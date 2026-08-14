import torch
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import h5py
import torchvision.transforms.functional as F
import random
from PIL import Image, ImageOps
from torchvision import transforms
import cv2
import json
import csv
import math
from pathlib import Path

from utils.clip import load_clip_tokenizer


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class ObjectCount(Dataset):
    def __init__(self, root, crop_size, downsample_ratio, method='train', concat_size=224,
                 tokenizer=None, clip_path=None):
        super(ObjectCount, self).__init__()
        #self.im_list = sorted(glob(os.path.join(root, 'images/*.jpg')))
        assert crop_size % downsample_ratio == 0
        assert method in ['train', 'val', 'test'], f"Invalid method: {method}. Must be 'train', 'val', or 'test'."
        self.crop_size = crop_size
        self.down_ratio = downsample_ratio
        self.method = method
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        if tokenizer is None and clip_path is None:
            raise ValueError("ObjectCount requires tokenizer or an explicit local clip_path.")
        self.tokenizer = tokenizer or load_clip_tokenizer(clip_path)
        self.concat_size = concat_size

        with open(os.path.join(root, 'FSC_147/Train_Test_Val_FSC_147.json'), 'r') as f:
            data_split = json.load(f)[method]
            self.im_list = [os.path.join(root, 'images_384_VarV2', x) for x in data_split]

        with open(os.path.join(root, 'FSC_147/annotation_FSC147_384.json'), 'r') as f:
            self.annotations = json.load(f)

        self.cls_dict = {}
        with open(os.path.join(root, 'FSC_147/ImageClasses_FSC147.txt'), "r", encoding="utf-8") as f:
            for line in f:
                self.cls_dict[line.strip().split('\t')[0]] = line.strip().split('\t')[1]

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        im_path = self.im_list[item]
        den_path = im_path.replace('images_384_VarV2', 'gt_density_map_adaptive_384_VarV2').replace('jpg', 'npy')

        img = Image.open(im_path).convert('RGB')
        cls_name = self.cls_dict[os.path.basename(im_path)]
        pts = self.annotations[os.path.basename(im_path)]['points']

        prompt = cls_name

        prompt_attn_mask = torch.zeros(77)
        cls_name_tokens = self.tokenizer(cls_name, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        prompt_attn_mask[1: 1 + cls_name_length] = 1

        if self.method == 'train':
            if random.random() > 0.5:
                out_img = img
                wd, ht = img.size
                den_map = np.load(den_path)
                img_attn_map = np.ones((ht, wd))

            else:
                if random.random() > 0.5:
                    rand_img = random.sample(self.im_list, 1)[0]
                    rand_cls = self.cls_dict[os.path.basename(rand_img)]

                    out_img = img
                    if rand_cls != cls_name:
                        wd, ht = img.size
                        den_map = np.zeros((ht, wd))
                        prompt = rand_cls
                        img_attn_map = np.zeros((ht, wd))
                        prompt_attn_mask = torch.zeros(77)
                        cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
                        cls_name_length = cls_name_tokens['input_ids'].shape[1]
                        prompt_attn_mask[1: 1 + cls_name_length] = 1
                    else:
                        wd, ht = img.size
                        den_map = np.load(den_path)
                        img_attn_map = np.ones((ht, wd))
                else:
                    rand_imgs = random.sample(self.im_list, 3)
                    imgs_info = []
                    wd, ht = img.size
                    i, j, h, w = random_crop(ht, wd, self.concat_size, self.concat_size)
                    img = F.crop(img, i, j, h, w)
                    den_map = np.load(den_path)[i: (i + h), j: (j + w)]
                    img_attn_map = np.ones((self.concat_size, self.concat_size))
                    imgs_info.append({'img': img, 'den_map': den_map, 'img_attention_map': img_attn_map})
                    for rand_img in rand_imgs:
                        extra_img = Image.open(rand_img).convert('RGB')
                        wd, ht = extra_img.size
                        i, j, h, w = random_crop(ht, wd, self.concat_size, self.concat_size)
                        extra_img = F.crop(extra_img, i, j, h, w)
                        if self.cls_dict[os.path.basename(rand_img)] == cls_name:
                            extra_den_map = np.load(rand_img.replace('images_384_VarV2', 'gt_density_map_adaptive_384_VarV2').replace('jpg', 'npy'))[i: (i + h), j: (j + w)]
                            extra_img_attention = np.ones((self.concat_size, self.concat_size))
                        else:
                            extra_den_map = np.zeros((self.concat_size, self.concat_size))
                            extra_img_attention = np.zeros((self.concat_size, self.concat_size))
                        imgs_info.append({'img': extra_img, 'den_map': extra_den_map, 'img_attention_map': extra_img_attention})

                    random.shuffle(imgs_info)
                    out_img = Image.new('RGB', (self.concat_size * 2, self.concat_size * 2))
                    out_img.paste(imgs_info[0]['img'], (0, 0))
                    out_img.paste(imgs_info[1]['img'], (self.concat_size, 0))
                    out_img.paste(imgs_info[2]['img'], (0, self.concat_size))
                    out_img.paste(imgs_info[3]['img'], (self.concat_size, self.concat_size))

                    den_map = np.zeros((self.concat_size * 2, self.concat_size * 2))
                    den_map[0:self.concat_size, 0:self.concat_size] = imgs_info[0]['den_map']
                    den_map[0:self.concat_size, self.concat_size:self.concat_size * 2] = imgs_info[1]['den_map']
                    den_map[self.concat_size:self.concat_size * 2, 0:self.concat_size] = imgs_info[2]['den_map']
                    den_map[self.concat_size:self.concat_size * 2, self.concat_size:self.concat_size * 2] = imgs_info[3][
                        'den_map']

                    img_attn_map = np.zeros((self.concat_size * 2, self.concat_size * 2))
                    img_attn_map[0:self.concat_size, 0:self.concat_size] = imgs_info[0]['img_attention_map']
                    img_attn_map[0:self.concat_size, self.concat_size:self.concat_size * 2] = imgs_info[1][
                        'img_attention_map']
                    img_attn_map[self.concat_size:self.concat_size * 2, 0:self.concat_size] = imgs_info[2][
                        'img_attention_map']
                    img_attn_map[self.concat_size:self.concat_size * 2, self.concat_size:self.concat_size * 2] = \
                        imgs_info[3]['img_attention_map']

            img, den_map, img_attn_map = self.train_transform_density(out_img, den_map, img_attn_map)
            return img, den_map, prompt, prompt_attn_mask, img_attn_map
        else:
            return self.transform(img), len(pts), prompt, prompt_attn_mask, os.path.basename(im_path).split('.')[0]

    def train_transform_density(self, img, den_map, img_attention_map):
        wd, ht = img.size
        if random.random() >= 0.5:
            re_size = random.random() * 1 + 1
            wd = int(wd * re_size)
            ht = int(ht * re_size)
            img = img.resize((wd, ht), Image.Resampling.BICUBIC)
            den_map = cv2.resize(den_map, (wd, ht), interpolation=cv2.INTER_CUBIC) / (re_size ** 2)
            img_attention_map = cv2.resize(img_attention_map, (wd, ht), interpolation=cv2.INTER_NEAREST)

        i, j, h, w = random_crop(ht, wd, self.crop_size, self.crop_size)
        img = F.crop(img, i, j, h, w)
        den_map = den_map[i: (i + h), j: (j + w)]
        den_map = den_map.reshape([h // self.down_ratio, self.down_ratio, w // self.down_ratio, self.down_ratio]).sum(
            axis=(1, 3))
        img_attention_map = img_attention_map[i: (i + h), j: (j + w)]
        img_attention_map = cv2.resize(img_attention_map, (int(w / 8), int(h / 8)), interpolation=cv2.INTER_NEAREST)

        if random.random() > 0.5:
            img = F.hflip(img)
            den_map = np.fliplr(den_map)
            img_attention_map = np.fliplr(img_attention_map)

        return self.transform(img), torch.from_numpy(den_map.copy()).float().unsqueeze(0), torch.from_numpy(img_attention_map.copy()).float().unsqueeze(0)


class IDCIA(Dataset):
    STAININGS = ('DAPI', 'TuJ1', 'MAP2ab', 'RIP', 'GFAP', 'Nestin', 'Ki67')

    def __init__(self, root, split='test', preprocess='raw'):
        if split != 'test':
            raise ValueError("IDCIA zero-shot evaluation only supports the official test split.")

        if preprocess not in ('raw', 'autocontrast'):
            raise ValueError(
                "IDCIA preprocess must be 'raw' or 'autocontrast'."
            )

        self.root = Path(root)
        self.preprocess = preprocess
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        image_index = self._index_files(self.root / 'images', {'.tif', '.tiff', '.png', '.jpg', '.jpeg'})
        annotation_index = self._index_files(self.root / 'ground_truth', {'.csv'})
        staining_names = {name.casefold(): name for name in self.STAININGS}

        self.samples = []
        for image_name in self._read_split(self.root / 'test.csv'):
            image_key = image_name.casefold()
            annotation_key = (Path(image_name).stem + '.csv').casefold()
            if image_key not in image_index:
                raise FileNotFoundError("IDCIA test image not found: {}".format(image_name))
            if annotation_key not in annotation_index:
                raise FileNotFoundError("IDCIA annotation not found for: {}".format(image_name))

            image_path = image_index[image_key]
            staining_key = image_path.parent.name.casefold()
            if staining_key not in staining_names:
                raise ValueError("Unknown IDCIA staining directory: {}".format(image_path.parent.name))

            annotation_path = annotation_index[annotation_key]
            self.samples.append({
                'image_path': image_path,
                'image_name': image_path.name,
                'staining': staining_names[staining_key],
                'gt_count': self._count_valid_coordinates(annotation_path),
            })

        if not self.samples:
            raise ValueError("IDCIA test split is empty: {}".format(self.root / 'test.csv'))

    @staticmethod
    def _index_files(directory, extensions):
        if not directory.is_dir():
            raise FileNotFoundError("IDCIA directory not found: {}".format(directory))

        index = {}
        for path in directory.rglob('*'):
            if not path.is_file() or path.suffix.casefold() not in extensions:
                continue
            key = path.name.casefold()
            if key in index:
                raise ValueError("Duplicate IDCIA filename: {}".format(path.name))
            index[key] = path
        return index

    @staticmethod
    def _read_split(split_path):
        if not split_path.is_file():
            raise FileNotFoundError("IDCIA split file not found: {}".format(split_path))

        image_names = []
        with split_path.open('r', newline='', encoding='utf-8-sig') as split_file:
            for row in csv.reader(split_file):
                if row and row[0].strip():
                    image_names.append(row[0].strip())
        return image_names

    @staticmethod
    def _count_valid_coordinates(annotation_path):
        count = 0
        with annotation_path.open('r', newline='', encoding='utf-8-sig') as annotation_file:
            reader = csv.DictReader(annotation_file)
            fields = {field.casefold(): field for field in (reader.fieldnames or [])}
            if 'x' not in fields or 'y' not in fields:
                raise ValueError("IDCIA annotation must contain X,Y columns: {}".format(annotation_path))

            for row in reader:
                try:
                    x = float(row[fields['x']])
                    y = float(row[fields['y']])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    count += 1
        return count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]

        with Image.open(sample['image_path']) as image:
            if self.preprocess == 'autocontrast':
                # IDCIA images are grayscale.
                # Apply contrast stretching before converting to RGB.
                image = image.convert('L')
                image = ImageOps.autocontrast(image)

            image = image.convert('RGB')

        return (
            self.transform(image),
            sample['gt_count'],
            sample['image_name'],
            sample['staining'],
        )


