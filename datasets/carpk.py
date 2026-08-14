import torch.nn as nn
import glob
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import torch

from utils.clip import load_clip_tokenizer


class CARPK(Dataset):
    def __init__(self, root, info, tokenizer=None, clip_path=None):
        self.im_list = sorted(glob.glob(os.path.join(root, 'Images/*.png')))
                
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.label_list = []
        try:
            with open(info) as f:
                for i in f:
                    self.label_list.append(i.strip())
        except:
            raise Exception("The path to label list is incorrect.")
        
        split_list = []
        for i in self.im_list:
            if os.path.basename(i).replace('.png', '') in self.label_list:
                split_list.append(i)
        self.im_list = split_list
        if tokenizer is None and clip_path is None:
            raise ValueError("CARPK requires tokenizer or an explicit local clip_path.")
        self.tokenizer = tokenizer or load_clip_tokenizer(clip_path)

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):

        im_path = self.im_list[item]

        img = Image.open(im_path).convert('RGB')
        ann_path = im_path.replace('png', 'txt').replace('Images', 'Annotations')
        print(os.path.basename(im_path))

        with open(ann_path, 'r') as f:
            count = sum(1 for line in f)

        prompt = 'cars'

        prompt_attn_mask = torch.zeros(77)
        cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        prompt_attn_mask[1: 1 + cls_name_length] = 1

        return self.transform(img), count, prompt, prompt_attn_mask, os.path.basename(im_path).split('.')[0]
