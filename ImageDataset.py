"""
Author: ta
Date and time: 20/11/2018 - 23:56
"""

from PIL import Image
from torch.utils.data import Dataset

from utils import get_nb_files, load_image


class ImageDataset(Dataset):
    def __init__(self, dir_x, nb_channels=3, file_format="png"):
        assert nb_channels in [1, 3]
        self.nb_channels = nb_channels
        self.nb_files = get_nb_files(dir_x)
        self.dir_x = dir_x
        self.file_format = file_format

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = self.dir_x / '{}.{}'.format(idx, self.file_format)
        if self.nb_channels == 3:
            x = load_image(filename).transpose((2, 0, 1)) / 255.0
        else:
            x = load_image(filename) / 255.0

        return x


class ImageTransformDataset(Dataset):
    def __init__(self, dir_x, transform, nb_channels=3, file_format="png"):
        assert nb_channels in [1, 3]
        self.nb_channels = nb_channels
        self.nb_files = get_nb_files(dir_x)
        self.dir_x = dir_x
        self.transform = transform
        self.file_format = file_format

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = self.dir_x / '{}.{}'.format(idx, self.file_format)
        if self.nb_channels == 3:
            x = self.transform(self.pil_loader(filename)) / 255.0
        else:
            x = self.transform(self.pil_loader(filename)) / 255.0

        return x
