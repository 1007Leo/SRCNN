import torch
import torchvision.transforms.v2 as transforms

from torch.utils.data import Dataset

import os
import random

from PIL import Image

from utils import get_paths, generate_lr

class Div2kDataset(Dataset):
    def __init__(self, path, train=True, repeat=1, upscale_factor=3, patch_size=48):
        self.lr = []
        self.hr = []
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.train = train
        self.repeat = repeat

        if train:
            self.lr_path = path + "/DIV2K_train_LR_bicubic/X" + str(upscale_factor)
            self.hr_path = path + "/DIV2K_train_HR"
        else:
            self.lr_path = path + "/DIV2K_valid_LR_bicubic/X" + str(upscale_factor)
            self.hr_path = path + "/DIV2K_valid_HR"

        if not os.path.isdir(self.lr_path):
            generate_lr(self.hr_path, upscale_factor, "test")

        self.lr_img_paths = get_paths(self.lr_path)
        self.hr_img_paths = get_paths(self.hr_path)

    def __len__(self):
        return len(self.lr_img_paths) * self.repeat

    def __getitem__(self, idx):
        idx //= self.repeat
        
        lr = Image.open(self.lr_img_paths[idx]).convert('RGB')
        hr = Image.open(self.hr_img_paths[idx]).convert('RGB')

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True)
        ])
        if not self.train:
          transform = transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True)
        ])
          
        if self.train:
            lr, hr = self._get_patch(lr, hr, self.patch_size, self.upscale_factor)
        lr, hr = transform(lr, hr)
        return lr, hr

    def _get_patch(self, lr, hr, patch_size, scale):
        lh, lw = lr.size

        hr_patch_size = patch_size
        lr_patch_size = hr_patch_size // scale

        lx = random.randint(0, lw - lr_patch_size)
        ly = random.randint(0, lh - lr_patch_size)

        hx, hy = scale * lx, scale * ly
        
        lr_patch = lr.crop((ly, lx, ly + lr_patch_size, lx + lr_patch_size))
        hr_patch = hr.crop((hy, hx, hy + hr_patch_size, hx + hr_patch_size))

        return lr_patch, hr_patch

class ImageDataset(Dataset):
    def __init__(self, folder_path, upscale_factor=3, train=False, bic_up=False, segment_size=(33, 33), stride=14):
        self.folder_name = folder_path.split('/')[-1]
        hr_path = folder_path+"/"+self.folder_name+"_HR"
        lr_path = folder_path+"/"+self.folder_name+"_LR_bicubic/X"+str(upscale_factor)

        self.lr_paths = get_paths(lr_path)
        self.hr_paths = get_paths(hr_path)

        if not os.path.isdir(lr_path) or len(self.lr_paths) == 0:
            generate_lr(hr_path, upscale_factor)
            self.lr_paths = get_paths(lr_path)

        self.lr_tensor = []
        self.hr_tensor = []
        
        transform = transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True)
        ])

        for lr_path, hr_path in zip(self.lr_paths, self.hr_paths):
            lr_img = Image.open(lr_path).convert('RGB')
            hr_img = Image.open(hr_path).convert('RGB')

            h,w = lr_img.size
            if bic_up:
                lr_img = lr_img.resize((h * upscale_factor, w * upscale_factor), Image.BICUBIC)
            hr_img = hr_img.crop((0,0,h * upscale_factor, w * upscale_factor))

            lr_img, hr_img = transform(lr_img, hr_img)

            if train:
                c, h, w = hr_img.shape
                for x in range(0, h - segment_size[0] + 1, stride):
                    for y in range(0, w - segment_size[1] + 1, stride):
                        lr_segment = lr_img[:, x:x+segment_size[0], y:y+segment_size[1]]
                        hr_segment = hr_img[:, x:x+segment_size[0], y:y+segment_size[1]]

                        self.lr_tensor.append(lr_segment)
                        self.hr_tensor.append(hr_segment)
            else:
                self.lr_tensor.append(lr_img)
                self.hr_tensor.append(hr_img)

    def get_name(self):
        return self.folder_name
    
    def get_paths(self):
        return self.hr_paths, self.lr_paths

    def __len__(self):
        return len(self.lr_tensor)

    def __getitem__(self, idx):
        return self.lr_tensor[idx], self.hr_tensor[idx]