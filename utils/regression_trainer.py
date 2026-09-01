from torch.utils.data import DataLoader, Subset, default_collate
import torch
import logging
import math
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
from utils.rich_prompt_training import (
    build_rich_checkpoint_config,
    load_rich_prompt_bank,
    validate_resume_rich_config,
)


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
    baseline_batch = images, den, prompt, prompt_attn_mask, img_attn_mask
    if len(transposed_batch) == 5:
        return baseline_batch

    rich_samples = transposed_batch[5]
    rich_batch = {
        'source_image_name': tuple(
            sample['source_image_name'] for sample in rich_samples
        ),
        'rich_compatible': torch.tensor(
            [sample['rich_compatible'] for sample in rich_samples],
            dtype=torch.bool,
        ),
        'detailed_prompt': tuple(
            sample['detailed_prompt'] for sample in rich_samples
        ),
        'generalized_prompt': tuple(
            sample['generalized_prompt'] for sample in rich_samples
        ),
        'detailed_prompt_attn_mask': torch.stack(
            [sample['detailed_prompt_attn_mask'] for sample in rich_samples], 0
        ),
        'generalized_prompt_attn_mask': torch.stack(
            [sample['generalized_prompt_attn_mask'] for sample in rich_samples], 0
        ),
    }
    return baseline_batch + (rich_batch,)


def validate_train_sample_options(args):
    if args.train_samples > 0 and args.smoke_train_samples > 0:
        raise ValueError(
            '--train-samples and --smoke-train-samples cannot be used '
            'simultaneously.'
        )
    rich_prompt_bank = getattr(args, 'rich_prompt_bank', None)
    consistency_weight = getattr(args, 'rich_consistency_weight', 0.0)
    diagnostic_steps = getattr(args, 'rich_loss_diagnostic_steps', 0)
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError('--rich-consistency-weight must be finite and non-negative.')
    if diagnostic_steps < 0:
        raise ValueError('--rich-loss-diagnostic-steps must be zero or greater.')
    if not rich_prompt_bank and (
        consistency_weight != 0.0 or diagnostic_steps != 0
    ):
        raise ValueError(
            '--rich-consistency-weight and --rich-loss-diagnostic-steps '
            'require --rich-prompt-bank.'
        )
    if rich_prompt_bank and getattr(args, 'smoke_train_samples', 0) > 0:
        raise ValueError(
            '--rich-prompt-bank cannot be combined with the infrastructure-only '
            '--smoke-train-samples limiter.'
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


def _selected_train_dataset_info(train_dataset):
    if isinstance(train_dataset, Subset):
        base_dataset = train_dataset.dataset
        indices = list(train_dataset.indices)
    else:
        base_dataset = train_dataset
        indices = list(range(len(base_dataset)))
    image_names = [
        os.path.basename(base_dataset.im_list[index]) for index in indices
    ]
    return base_dataset, image_names


def rich_consistency_loss(class_density, detailed_density, generalized_density):
    """Return the three pairwise MSE terms and their symmetric mean."""
    class_detailed = F.mse_loss(class_density, detailed_density)
    class_generalized = F.mse_loss(class_density, generalized_density)
    detailed_generalized = F.mse_loss(
        detailed_density, generalized_density
    )
    consistency = (
        class_detailed + class_generalized + detailed_generalized
    ) / 3
    return {
        'class_detailed': class_detailed,
        'class_generalized': class_generalized,
        'detailed_generalized': detailed_generalized,
        'mean': consistency,
    }


def combine_rich_sample_weighted_loss(
    class_batch_loss,
    class_rich_loss,
    detailed_loss,
    generalized_loss,
    consistency_loss,
    rich_count,
    total_count,
    consistency_weight,
):
    """Replace compatible class losses without multiplying sample weight."""
    if rich_count < 1 or rich_count > total_count:
        raise ValueError('rich_count must be between 1 and total_count.')
    rich_supervised = (
        class_rich_loss + detailed_loss + generalized_loss
    ) / 3
    rich_fraction = float(rich_count) / float(total_count)
    combined = class_batch_loss + rich_fraction * (
        rich_supervised
        + consistency_weight * consistency_loss
        - class_rich_loss
    )
    return combined, rich_supervised


def _numeric(value):
    if value is None:
        return float('nan')
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


def log_rich_loss_diagnostic(
    step,
    rich_count,
    incompatible_count,
    class_loss,
    detailed_loss,
    generalized_loss,
    consistency_terms,
    combined_loss,
    class_mean_count=None,
    detailed_mean_count=None,
    generalized_mean_count=None,
):
    """Log scalar diagnostics only; return the unchanged optimization loss."""
    consistency_terms = consistency_terms or {}
    logging.info(
        (
            'Rich diagnostic step=%d compatible=%d incompatible=%d '
            't2i[class=%.6g detailed=%.6g generalized=%.6g] '
            'mse[class_detailed=%.6g class_generalized=%.6g '
            'detailed_generalized=%.6g mean=%.6g] combined=%.6g '
            'pred_count[class=%.4g detailed=%.4g generalized=%.4g]'
        ),
        step,
        rich_count,
        incompatible_count,
        _numeric(class_loss),
        _numeric(detailed_loss),
        _numeric(generalized_loss),
        _numeric(consistency_terms.get('class_detailed')),
        _numeric(consistency_terms.get('class_generalized')),
        _numeric(consistency_terms.get('detailed_generalized')),
        _numeric(consistency_terms.get('mean')),
        _numeric(combined_loss),
        _numeric(class_mean_count),
        _numeric(detailed_mean_count),
        _numeric(generalized_mean_count),
    )
    return combined_loss


def _t2i_loss_components(outputs, gt_den_maps, gt_img_attn_mask):
    pred_den, sim_x2, sim_x1, fused_cross_attn = outputs
    ambiguous_negative = (
        fused_cross_attn * gt_img_attn_mask
    ) >= 0.3
    positive = gt_den_maps >= (1e-3 * 60)
    reg_loss = get_reg_loss(
        pred_den, gt_den_maps, threshold=1e-3 * 60
    )
    rrc_loss_stage1 = RRC_loss(sim_x2, ambiguous_negative, positive)
    rrc_loss_stage2 = RRC_loss(sim_x1, ambiguous_negative, positive)
    return {
        'reg': reg_loss,
        'rrc1': rrc_loss_stage1,
        'rrc2': rrc_loss_stage2,
        'total': reg_loss + 0.01 * rrc_loss_stage1 + 0.01 * rrc_loss_stage2,
    }


def _slice_outputs(outputs, indices):
    return tuple(output.index_select(0, indices) for output in outputs)


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
        self.rich_prompt_bank = None
        self.rich_prompt_config = None
        rich_prompt_bank_path = getattr(args, 'rich_prompt_bank', None)
        if rich_prompt_bank_path:
            base_train_dataset, selected_image_names = (
                _selected_train_dataset_info(self.datasets['train'])
            )
            self.rich_prompt_bank = load_rich_prompt_bank(
                rich_prompt_bank_path,
                train_samples=args.train_samples,
                train_subset_seed=args.train_subset_seed,
                selected_image_names=selected_image_names,
                class_by_image=base_train_dataset.cls_dict,
            )
            base_train_dataset.rich_prompt_records = (
                self.rich_prompt_bank.records
            )
            self.rich_prompt_config = build_rich_checkpoint_config(
                self.rich_prompt_bank,
                consistency_weight=args.rich_consistency_weight,
                train_samples=args.train_samples,
                train_subset_seed=args.train_subset_seed,
            )
            logging.info(
                'Rich prompt training enabled: bank=%s samples=%d '
                'fingerprint=%s consistency_weight=%.6g',
                self.rich_prompt_config['prompt_bank_filename'],
                len(selected_image_names),
                self.rich_prompt_config['prompt_bank_fingerprint'],
                args.rich_consistency_weight,
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
        self._rich_diagnostic_steps_logged = 0

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
        validate_resume_rich_config(
            getattr(self, 'rich_prompt_config', None),
            checkpoint.get('rich_prompt_config'),
        )
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
        payload = {
            'format_version': 2,
            'epoch': self.epoch,
            'next_epoch': self.epoch + 1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'rng_state': _capture_rng_state(),
        }
        if getattr(self, 'rich_prompt_config', None) is not None:
            payload['rich_prompt_config'] = dict(self.rich_prompt_config)
        _atomic_torch_save(payload, save_path)
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
        args = self.args
        epoch_reg_loss = AverageMeter()
        epoch_RRC1_loss = AverageMeter()
        epoch_RRC2_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_rich_supervised = AverageMeter()
        epoch_rich_consistency = AverageMeter()
        rich_compatible_seen = 0
        train_samples_seen = 0
        epoch_start = time.time()

        train_dataloader = self.dataloaders['train']
        train_progress = progress_dataloader(
            train_dataloader, 'Train epoch {}'.format(self.epoch)
        )
        for step, batch in enumerate(train_progress):
            if len(batch) == 5:
                input, den_map, caption, prompt_attn_mask, img_attn_mask = batch
                rich_batch = None
            else:
                (
                    input,
                    den_map,
                    caption,
                    prompt_attn_mask,
                    img_attn_mask,
                    rich_batch,
                ) = batch
            inputs = input.to(self.device)
            gt_den_maps = den_map.to(self.device) * 60
            gt_prompt_attn_mask = prompt_attn_mask.to(self.device).unsqueeze(2).unsqueeze(3)
            gt_img_attn_mask = img_attn_mask.to(self.device)
            self.model.set_train()
            with torch.set_grad_enabled(True):
                N = inputs.shape[0]
                class_outputs = self.model(
                    inputs, caption, gt_prompt_attn_mask
                )
                pred_den = class_outputs[0]
                class_components = _t2i_loss_components(
                    class_outputs, gt_den_maps, gt_img_attn_mask
                )
                reg_loss = class_components['reg']
                rrc_loss_stage1 = class_components['rrc1']
                rrc_loss_stage2 = class_components['rrc2']

                epoch_reg_loss.update(reg_loss.item(), N)
                epoch_RRC1_loss.update(rrc_loss_stage1.item(), N)
                epoch_RRC2_loss.update(rrc_loss_stage2.item(), N)
                loss = class_components['total']

                rich_count = 0
                class_rich_loss = None
                detailed_loss = None
                generalized_loss = None
                consistency_terms = None
                class_mean_count = None
                detailed_mean_count = None
                generalized_mean_count = None
                if rich_batch is not None:
                    compatible = rich_batch['rich_compatible']
                    rich_indices_cpu = torch.nonzero(
                        compatible, as_tuple=False
                    ).flatten()
                    rich_count = rich_indices_cpu.numel()
                    rich_compatible_seen += rich_count
                    train_samples_seen += N
                    if rich_count > 0:
                        rich_indices = rich_indices_cpu.to(self.device)
                        rich_gt_den_maps = gt_den_maps.index_select(
                            0, rich_indices
                        )
                        rich_img_attn_mask = gt_img_attn_mask.index_select(
                            0, rich_indices
                        )
                        class_rich_outputs = _slice_outputs(
                            class_outputs, rich_indices
                        )
                        class_rich_components = _t2i_loss_components(
                            class_rich_outputs,
                            rich_gt_den_maps,
                            rich_img_attn_mask,
                        )
                        class_rich_loss = class_rich_components['total']

                        selected = rich_indices_cpu.tolist()
                        detailed_captions = tuple(
                            rich_batch['detailed_prompt'][index]
                            for index in selected
                        )
                        generalized_captions = tuple(
                            rich_batch['generalized_prompt'][index]
                            for index in selected
                        )
                        detailed_masks = rich_batch[
                            'detailed_prompt_attn_mask'
                        ].index_select(0, rich_indices_cpu).to(
                            self.device
                        ).unsqueeze(2).unsqueeze(3)
                        generalized_masks = rich_batch[
                            'generalized_prompt_attn_mask'
                        ].index_select(0, rich_indices_cpu).to(
                            self.device
                        ).unsqueeze(2).unsqueeze(3)
                        rich_inputs = inputs.index_select(0, rich_indices)

                        detailed_outputs = self.model(
                            rich_inputs, detailed_captions, detailed_masks
                        )
                        generalized_outputs = self.model(
                            rich_inputs, generalized_captions,
                            generalized_masks
                        )
                        detailed_components = _t2i_loss_components(
                            detailed_outputs,
                            rich_gt_den_maps,
                            rich_img_attn_mask,
                        )
                        generalized_components = _t2i_loss_components(
                            generalized_outputs,
                            rich_gt_den_maps,
                            rich_img_attn_mask,
                        )
                        detailed_loss = detailed_components['total']
                        generalized_loss = generalized_components['total']
                        consistency_terms = rich_consistency_loss(
                            class_rich_outputs[0],
                            detailed_outputs[0],
                            generalized_outputs[0],
                        )
                        loss, rich_supervised = (
                            combine_rich_sample_weighted_loss(
                                class_components['total'],
                                class_rich_loss,
                                detailed_loss,
                                generalized_loss,
                                consistency_terms['mean'],
                                rich_count,
                                N,
                                args.rich_consistency_weight,
                            )
                        )
                        epoch_rich_supervised.update(
                            rich_supervised.detach().item(), rich_count
                        )
                        epoch_rich_consistency.update(
                            consistency_terms['mean'].detach().item(),
                            rich_count,
                        )
                        if (
                            self._rich_diagnostic_steps_logged
                            < args.rich_loss_diagnostic_steps
                        ):
                            class_mean_count = (
                                class_rich_outputs[0]
                                .detach().flatten(1).sum(1).mean() / 60
                            )
                            detailed_mean_count = (
                                detailed_outputs[0]
                                .detach().flatten(1).sum(1).mean() / 60
                            )
                            generalized_mean_count = (
                                generalized_outputs[0]
                                .detach().flatten(1).sum(1).mean() / 60
                            )

                    diagnostic_limit = args.rich_loss_diagnostic_steps
                    if self._rich_diagnostic_steps_logged < diagnostic_limit:
                        log_rich_loss_diagnostic(
                            self._rich_diagnostic_steps_logged + 1,
                            rich_count,
                            N - rich_count,
                            class_rich_loss,
                            detailed_loss,
                            generalized_loss,
                            consistency_terms,
                            loss,
                            class_mean_count,
                            detailed_mean_count,
                            generalized_mean_count,
                        )
                        self._rich_diagnostic_steps_logged += 1

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
        if getattr(self, 'rich_prompt_config', None) is not None:
            rich_fraction = (
                float(rich_compatible_seen) / train_samples_seen
                if train_samples_seen else 0.0
            )
            logging.info(
                'Epoch %d Rich, supervised:%.6f, consistency:%.6f, '
                'compatible:%d/%d (%.4f)',
                self.epoch,
                epoch_rich_supervised.getAvg(),
                epoch_rich_consistency.getAvg(),
                rich_compatible_seen,
                train_samples_seen,
                rich_fraction,
            )

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
