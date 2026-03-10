import os.path
from data.base_dataset import BaseDataset, get_transform

from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np


class one2allstyleDataset(BaseDataset):
    """
    This dataset class can load tif datasets with multiple channels.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B', 'Style_1')  # create a path '/path/to/data/trainB'
        self.opt.num_style = opt.num_style

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        # print(self.dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.A_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # Apply image transformation
        transform = get_transform(self.opt)

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_img = Image.open(A_path).convert('RGB')
        A_img = transform(A_img)
        A_concat = A_img

        B_path = self.B_paths[index % self.B_size]
        B_img = Image.open(B_path).convert('RGB')
        B_concat = transform(B_img)

        # image_tile = np.transpose(image_tile, (2, 0, 1))
        # image_tile_concat = image_tile
        for style in range(self.opt.num_style-1):
            im_path = B_path.replace('Style_1','Style_'+str(style+2))
            B_img = Image.open(im_path).convert('RGB')
            B_img = transform(B_img)
            B_concat = np.concatenate((B_concat, B_img), axis=0)

            A_concat = np.concatenate((A_concat, A_img), axis=0)

        A = A_concat
        B = B_concat
        # print(A.shape)
        # print(B.shape)
        # aa

        # A = random_transform(A_img)
        # B = random_transform(B_img)
        # A = A / np.max(A)
        # B = B / np.max(B)
        # A = torch.from_numpy(A.copy()).float()
        # B = torch.from_numpy(B.copy()).float()
        # print(A.shape)
        # print('bb')

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)