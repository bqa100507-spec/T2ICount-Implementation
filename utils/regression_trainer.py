from torch.utils.data import DataLoader, Subset, default_collate
import torch
import logging
from utils.helper import SaveHandler, AverageMeter
from utils.trainer import Trainer
from models.build import build_t2icount
from datasets.dataset import ObjectCount
import numpy as np
import os
import time
import random
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from utils.checkpoints import load_trusted_legacy_checkpoint
from utils.ssim_loss import cal_avg_ms_ssim
from utils.inference import predict_count
from utils.train_subset import select_train_subset_indices


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _atomic_torch_save(payload, path):
    temporary_path = '{}.tmp'.format(path)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    den = torch.stack(transposed_batch[1], 0)  # the number of points is not fixed, keep it as a list of tensor
    prompt = transposed_batch[2]
    prompt_attn_mask = torch.stack(transposed_batch[3], 0)
    img_attn_mask = torch.stack(transposed_batch[4], 0)
    return images, den, prompt, prompt_attn_mask, img_attn_mask


def validate_train_sample_options(args):
    if args.train_samples > 0 and args.smoke_train_samples > 0:
        raise ValueError(
            '--train-samples and --smoke-train-samples cannot be used '
            'simultaneously.'
        )


def apply_smoke_train_sample_limit(datasets, sample_limit):
    if sample_limit > 0:
        train_dataset = datasets['train']
        effective_samples = min(sample_limit, len(train_dataset))
        datasets['train'] = Subset(
            train_dataset, range(effective_samples)
        )
        print(
            'Smoke training sample limiter active: using {} of {} '
            'training samples.'.format(
                effective_samples, len(train_dataset)
            )
        )
    return datasets


def apply_train_sample_subset(datasets, sample_limit, subset_seed):
    if sample_limit > 0:
        train_dataset = datasets['train']
        effective_samples = min(sample_limit, len(train_dataset))
        indices = select_train_subset_indices(
            len(train_dataset), sample_limit, subset_seed
        )
        datasets['train'] = Subset(train_dataset, indices)
        print(
            'Training subset active: using {} of {} training samples '
            'with subset seed {}.'.format(
                effective_samples, len(train_dataset), subset_seed
            )
        )
    return datasets


def progress_dataloader(dataloader, description):
    return tqdm(dataloader, desc=description)


class Reg_Trainer(Trainer):
    def setup(self):
        args = self.args
        validate_train_sample_options(args)
        if args.seed != -1:
            setup_seed(args.seed)
            print('Random seed is set as {}'.format(args.seed))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_device(torch.cuda.current_device())
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('Using {} gpus'.format(self.device_count))
        else:
            raise Exception('GPU is not available')

        self.d_ratio = args.downsample_ratio

        self.model = build_t2icount(
            args.config,
            args.sd_path,
            args.clip_path,
            device=self.device,
            mode='train',
            unet_config={
                'base_size': self.args.crop_size,
                'max_attn_size': self.args.crop_size // self.d_ratio,
                'attn_selector': 'down_cross+up_cross',
            },
        )

        self.datasets = {x: ObjectCount(args.data_dir,
                                        crop_size=args.crop_size,
                                        downsample_ratio=self.d_ratio,
                                        method=x,
                                        concat_size=args.concat_size,
                                        tokenizer=self.model.clip.tokenizer)
                         for x in ['train', 'val', 'test']}
        apply_train_sample_subset(
            self.datasets, args.train_samples, args.train_subset_seed
        )
        apply_smoke_train_sample_limit(
            self.datasets, args.smoke_train_samples
        )

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          collate_fn=(train_collate if x=='train' else default_collate),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val', 'test']}
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.unet.parameters(),
             'lr': args.lr * 0.1,
             'weight_decay': args.weight_decay * 0.1},
            {'params': self.model.decoder.parameters(),
             'lr': args.lr,
             'weight_decay': args.weight_decay}])

        self.start_epoch = args.start_epoch
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_list = SaveHandler(num=args.max_num)

        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                self._load_training_checkpoint(args.resume)
            elif suf == 'pth':
                raise ValueError(
                    'A .pth file contains model weights only and cannot resume '
                    'optimizer/epoch state. Use a training .tar checkpoint.'
                )

    def _load_training_checkpoint(self, path):
        checkpoint = load_trusted_legacy_checkpoint(path, 'cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get(
            'next_epoch', int(checkpoint['epoch']) + 1
        )
        self.best_mae = checkpoint.get('best_mae', np.inf)
        self.best_mse = checkpoint.get('best_mse', np.inf)
        _restore_rng_state(checkpoint.get('rng_state'))
        logging.info(
            'Resumed full training state from %s at epoch %d',
            path,
            self.start_epoch,
        )

    def _save_training_checkpoint(self):
        save_path = os.path.join(
            self.save_dir, '{}_ckpt.tar'.format(self.epoch)
        )
        _atomic_torch_save({
            'format_version': 2,
            'epoch': self.epoch,
            'next_epoch': self.epoch + 1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'rng_state': _capture_rng_state(),
        }, save_path)
        self.save_list.append(save_path)

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            logging.info('-' * 50 + "Epoch:{}/{}".format(epoch, args.epochs - 1) + '-' * 50)
            self.epoch = epoch
            self.train_epoch()
            if self.epoch >= args.start_val and self.epoch % self.args.val_epoch == 0:
                self.val_epoch()
            if self.epoch % 5 == 0:
                self._save_training_checkpoint()

    def train_epoch(self):
        epoch_reg_loss = AverageMeter()
        epoch_RRC1_loss = AverageMeter()
        epoch_RRC2_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()

        train_dataloader = self.dataloaders['train']
        train_progress = progress_dataloader(
            train_dataloader, 'Train epoch {}'.format(self.epoch)
        )
        for step, (input, den_map, caption, prompt_attn_mask, img_attn_mask) in enumerate(
                train_progress):
            inputs = input.to(self.device)
            gt_den_maps = den_map.to(self.device) * 60
            gt_prompt_attn_mask = prompt_attn_mask.to(self.device).unsqueeze(2).unsqueeze(3)
            gt_img_attn_mask = img_attn_mask.to(self.device)
            self.model.set_train()
            with torch.set_grad_enabled(True):
                N = inputs.shape[0]
                pred_den, sim_x2, sim_x1, fused_cross_attn = self.model(inputs, caption, gt_prompt_attn_mask)
                fused_cross_attn_ = fused_cross_attn * gt_img_attn_mask
                AN = fused_cross_attn_ >= 0.3 
                reg_loss = get_reg_loss(pred_den, gt_den_maps, threshold=1e-3 * 60)
                P = gt_den_maps >= (1e-3 * 60)
                rrc_loss_stage1 = RRC_loss(sim_x2, AN, P)
                rrc_loss_stage2 = RRC_loss(sim_x1, AN, P)

                epoch_reg_loss.update(reg_loss.item(), N)
                epoch_RRC1_loss.update(rrc_loss_stage1.item(), N)
                epoch_RRC2_loss.update(rrc_loss_stage2.item(), N)
                loss = reg_loss + 0.01 * rrc_loss_stage1 + 0.01 * rrc_loss_stage2

                gt_counts = torch.sum(gt_den_maps.view(N, -1), dim=1).detach().cpu().numpy() / 60
                pred_counts = torch.sum(pred_den.view(N, -1), dim=1).detach().cpu().numpy() / 60
                diff = pred_counts - gt_counts
                epoch_mae.update(np.mean(np.abs(diff)).item(), N)
                epoch_mse.update(np.mean(diff * diff), N)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                completed_steps = step + 1
                if (
                    completed_steps % 10 == 0
                    or completed_steps == len(train_dataloader)
                ):
                    train_progress.set_postfix(
                        reg='{:.4f}'.format(epoch_reg_loss.getAvg()),
                        rrc1='{:.4f}'.format(epoch_RRC1_loss.getAvg()),
                        rrc2='{:.4f}'.format(epoch_RRC2_loss.getAvg()),
                        mae='{:.2f}'.format(epoch_mae.getAvg()),
                        refresh=False,
                    )

        logging.info(
            'Epoch {} Train, reg:{:.4f}, RRC_stage1:{:.4f}, RRC_stage2:{:.4f}, mae:{:.2f}, mse:{:.2f}, Cost: {:.1f} sec '
            .format(self.epoch, epoch_reg_loss.getAvg(), epoch_RRC1_loss.getAvg(), epoch_RRC2_loss.getAvg(), epoch_mae.getAvg(),
                    np.sqrt(epoch_mse.getAvg()), (time.time() - epoch_start)))

    def val_epoch(self):
        epoch_start = time.time()
        mae, mse = self._evaluate_split('val')

        logging.info('Epoch {} Val, MAE: {:.2f}, MSE: {:.2f} Cost {:.1f} sec'
                     .format(self.epoch, mae, mse, (time.time() - epoch_start)))

        model_state_dict = self.model.state_dict()

        if (mae + mse) < (self.best_mae + self.best_mse):
            self.best_mae = mae
            self.best_mse = mse
            _atomic_torch_save(
                model_state_dict,
                os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.epoch)),
            )
            logging.info("Save best model: MAE: {:.2f} MSE:{:.2f} model epoch {}".format(mae, mse, self.epoch))
            self.test_epoch()
        print("Best Result: MAE: {:.2f} MSE:{:.2f}".format(self.best_mae, self.best_mse))

    def test_epoch(self):
        epoch_start = time.time()
        mae, mse = self._evaluate_split('test')

        logging.info('Epoch {} Test, MAE: {:.2f}, MSE: {:.2f} Cost {:.1f} sec'
                     .format(self.epoch, mae, mse, (time.time() - epoch_start)))

    def _evaluate_split(self, split):
        self.model.set_eval()
        epoch_res = []
        evaluation_progress = progress_dataloader(
            self.dataloaders[split], split.capitalize()
        )
        for inputs, gt_counts, captions, prompt_attn_mask, name in evaluation_progress:
            inputs = inputs.to(self.device)
            prediction = predict_count(
                self.model,
                inputs,
                captions[0],
                prompt_attn_mask,
                batch_size=self.args.batch_size,
                patch_size=self.args.crop_size,
                stride=self.args.stride,
            )
            epoch_res.append(gt_counts[0].item() - prediction)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        return mae, mse

def get_normalized_map(density_map):
    B, C, H, W = density_map.size()
    mu_sum = density_map.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mu_normed = density_map / (mu_sum + 1e-6)
    return mu_normed


def get_reg_loss(pred, gt, threshold, level=3, window_size=3):
    mask = gt > threshold
    loss_ssim = cal_avg_ms_ssim(pred * mask, gt * mask, level=level,
                                window_size=window_size)
    mu_normed = get_normalized_map(pred)
    gt_mu_normed = get_normalized_map(gt)
    tv_loss = (nn.L1Loss(reduction='none')(mu_normed, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)
    return loss_ssim + 0.1 * tv_loss


def RRC_loss(simi, ambiguous_negative_map, positive_map):
    pos = (1 - simi) * positive_map
    neg = torch.clamp(simi, min=0) * (ambiguous_negative_map == 0) * (positive_map == 0)

    pos_num = positive_map.flatten(1).sum(dim=1)
    neg_num = ((ambiguous_negative_map == 0) * (positive_map == 0)).flatten(1).sum(dim=1)
    loss = 2 * pos.flatten(1).sum(dim=1) / (pos_num + 1e-7) + neg.flatten(1).sum(dim=1) / (neg_num + 1e-7)
    return loss.mean()
