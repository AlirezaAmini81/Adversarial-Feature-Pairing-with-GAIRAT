import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

# from .utils import check_integrity, download_and_extract_archive
# from .vision import VisionDataset

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch 

class gcn_zca_CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:

        self.root = root
        self.transform = transform 
        self.target_transform = target_transform
        self.train = train  # training set or test set
        
        # if download:
        #     self.download()

        if self.train:
            self.filename = f'{self.root}/train.npz'
        else:
            self.filename = f'{self.root}/test.npz'

        # preprocessor
        # self.preprocessor = 

        loaded = np.load(f'{self.filename}')
        self.data = loaded['X']
        self.targets = loaded['Y'].squeeze()

        self.data = self.data.reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        print(self.data.shape)
        print(self.targets.shape)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        #### MUST USE WITH toPil()
        # assert transforms.ToPILImage in transforms

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img.copy(), target.copy()
        # return img, target
        return torch.tensor(img.copy(), dtype=torch.float32),  torch.tensor(target.copy(), dtype=torch.int64) # torch.LongTensor(target)

    def __len__(self) -> int:
        return len(self.data)

    # def download(self) -> None:
    #     if self._check_integrity():
    #         print("Files already downloaded and verified")
    #         return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
