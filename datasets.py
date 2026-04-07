import os
import numpy as np
from torch.utils.data import Dataset
import glob
import tifffile as tf
import torch
import torch.nn.functional as F

class SimpleDataset(Dataset):
    def __init__(self, opt):
        super(SimpleDataset, self).__init__()
        self.max_T = opt['max_T']
        self.scale_factor = opt['sf']
        self.lr = opt['lr_path']
        self.hr = opt['hr_path']
        self.lr_paths = []
        self.hr_paths = []
        for path in self.lr:
            days = os.listdir(path)
            for day in days:
                self.lr_paths.extend(glob.glob(fr"{path}\{day}\*.tif"))
        for path in self.hr:
            days = os.listdir(path)
            for day in days:
                self.hr_paths.extend(glob.glob(fr"{path}\{day}\*.tif"))

    def __len__(self):
        assert len(self.lr_paths) == len(self.hr_paths), "Training lr and hr size error!"
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = tf.imread(self.lr_paths[idx]) / self.max_T
        hr = tf.imread(self.hr_paths[idx]) / self.max_T
        lr = np.expand_dims(lr, axis=0)
        hr = np.expand_dims(hr, axis=0)

        hr = torch.from_numpy(hr.copy())  # NumPy → Tensor
        lr = torch.from_numpy(lr.copy())

        lr = F.interpolate(lr.unsqueeze(0), scale_factor=self.scale_factor, mode='bicubic', align_corners=False).squeeze(0)
        lr = (lr - 0.5) / 0.5  # [0,1] → [-1,1]
        hr = (hr - 0.5) / 0.5  # [0,1] → [-1,1]
        return {'lr': lr, 'hr': hr, 'res': lr, 'f_name': self.hr_paths[idx]}

class FrequencyDataset(Dataset):
    def __init__(self, opt):
        super(FrequencyDataset, self).__init__()
        self.max_T = opt['max_T']
        self.scale_factor = opt['sf']
        self.f = opt['frequency']
        self.lr = opt['lr_path']
        self.hr = opt['hr_path']
        self.lr_paths = []
        self.hr_paths = []
        for path in self.lr:
            days = os.listdir(path)
            for day in days:
                self.lr_paths.extend(glob.glob(fr"{path}\{day}\{self.f}*.tif"))
        for path in self.hr:
            days = os.listdir(path)
            for day in days:
                self.hr_paths.extend(glob.glob(fr"{path}\{day}\{self.f}*.tif"))
        print(self.lr_paths)

    def __len__(self):
        assert len(self.lr_paths)==len(self.hr_paths), "Training lr and hr size error!"
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = tf.imread(self.lr_paths[idx]) / self.max_T
        hr = tf.imread(self.hr_paths[idx]) / self.max_T
        lr = np.expand_dims(lr, axis=0)
        hr = np.expand_dims(hr, axis=0)

        hr = torch.from_numpy(hr.copy())  # NumPy → Tensor
        lr = torch.from_numpy(lr.copy())

        lr = F.interpolate(lr.unsqueeze(0), scale_factor=self.scale_factor, mode='bicubic', align_corners=False).squeeze(0)
        lr = (lr - 0.5) / 0.5  # [0,1] → [-1,1]
        hr = (hr - 0.5) / 0.5  # [0,1] → [-1,1]

        return {'lr': lr, 'hr': hr, 'res': lr, 'f_name': self.hr_paths[idx]}
