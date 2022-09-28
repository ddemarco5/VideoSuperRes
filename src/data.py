"""

"""
import os

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset

from skimage import io

opj = os.path.join

class AnimeHDDataset(Dataset):

    def __init__(self, root_dir, train):

        self.root_dir = root_dir
        self.train = train
        self.image_paths = [opj(root_dir, x) for x in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def transform(self, full_image):

        crop_func = transforms.RandomCrop(512)
        resize_func1 = transforms.Resize(128)
        resize_func2 = transforms.Resize(256)
        to_tensor = transforms.ToTensor()

        full_image = to_tensor(full_image)

        y = crop_func(full_image)
        #x = resize_func1(y)
        x = resize_func2(y)

        #x = (x / 127.5) - 1.
        #y = (y / 127.5) - 1.

        return x, y

    def __getitem__(self, idx):

        print('pid: ', os.getpid(), ' - getting index[', idx, ']')

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]

        image = io.imread(image_path)

        return self.transform(image)
