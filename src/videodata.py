<<<<<<< Updated upstream
"""

"""
import os
import time

from math import floor
from threading import Thread

import numpy

from skimage import io

import av
import torch
import src.libav_functions
import torchvision.transforms as transforms

from av import VideoFormat
from torchvision.utils import save_image
from src.video_thread_test import ThreadedDecoder
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def __init__(self, root_dir, train, cache_size, patch_size, num_ds):
        self.root_dir = root_dir
        self.train = train

        self.patch_size = patch_size
        self.input_size = self.patch_size // num_ds

        #just one for now
        self.data_decoder = ThreadedDecoder(root_dir, cache_size)
        #self.data_decoder.start()

    def __len__(self):
        return self.data_decoder.get_length()

    def __getitem__(self, idx):

        #print("requested index", idx)
        #if self.data_decoder.active_buf is self.data_decoder.buf_1:
        #    print("1", end="", flush=True)
        #elif self.data_decoder.active_buf is self.data_decoder.buf_2:
        #    print("2", end="", flush=True)
        return self.transform(self.data_decoder.active_buf[idx].copy())

    def swap(self):
        self.data_decoder.swap()

    # returns the number of epochs we can achive before we get full data
    # coverage
    def get_epochs_per_dataset(self):
        return self.data_decoder.get_num_chunks()

    def transform(self, full_image):

        crop_func = transforms.RandomCrop(self.patch_size)
        resize_func = transforms.Resize(self.input_size)

        to_tensor = transforms.ToTensor()

        full_image = to_tensor(full_image)

        if self.train:
            y = crop_func(full_image)
            x = resize_func(y)
            return x, y

        return full_image
=======
"""

"""
import os
import time
import random

from math import floor
from threading import Thread

import numpy

from skimage import io

import av
import torch
import src.libav_functions as helper_funcs
import torchvision.transforms as transforms

from av import VideoFormat
from torchvision.utils import save_image
from src.video_frame_loader import ThreadedDecoder
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def __init__(self, root_dir, train, cache_size, patch_size, num_ds, patch_ds_factor):
        self.root_dir = root_dir
        self.train = train

        self.input_size = patch_size // num_ds

        # We want to downscale our original patch size to lessen any
        # video compression artifacts in the training data
        self.patch_ds_factor = patch_ds_factor
        self.patch_size = patch_size * patch_ds_factor

        #just one for now
        self.data_decoder = ThreadedDecoder(root_dir, cache_size)
        #self.data_decoder.start()

    def __len__(self):
        return self.data_decoder.get_length()

    def __getitem__(self, idx):

        #print("requested index", idx)
        #if self.data_decoder.active_buf is self.data_decoder.buf_1:
        #    print("1", end="", flush=True)
        #elif self.data_decoder.active_buf is self.data_decoder.buf_2:
        #    print("2", end="", flush=True)

        # downsized and ground truth
        small, ground_truth = self.transform(self.data_decoder.active_buf[idx].copy())

        #return self.transform(self.data_decoder.active_buf[idx].copy())
        return (small, ground_truth)

    def swap(self):
        self.data_decoder.swap()

    # returns the number of epochs we can achive before we get full data
    # coverage
    def get_epochs_per_dataset(self):
        return self.data_decoder.get_num_chunks()

    def transform(self, full_image):

        crop_func = transforms.RandomCrop(self.patch_size)
        resize_func = transforms.Resize(self.input_size)
        patch_downscale_func = transforms.Resize(self.patch_size // self.patch_ds_factor)

        to_tensor = transforms.ToTensor()
        # convert it to a tensor to work on it
        full_image = to_tensor(full_image)

        if self.train:
            y = crop_func(full_image)
            # Random color jitter
            #y = transforms.ColorJitter(hue=0.3)(y)

            # Shittify our cropped image
            bitrate = random.randrange(50, 1000, 5) # random number of kilobits per second
            y_enc = helper_funcs.compress_frame(transforms.ToPILImage()(y), "libx264", bitrate)
            y_enc = to_tensor(y_enc.copy()) # y.copy() is to avoid the totensor numpy array is not writable issue

            # Downscale our patch to give us a better chance of removing compression noise
            # and increasing fidelity as a whole
            y = patch_downscale_func(y)

            # Give our image a random blur between a range
            #x = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))(y)
        
            x = resize_func(y_enc)
            
            
            #exit(0)
            return x, y

        return full_image
>>>>>>> Stashed changes
