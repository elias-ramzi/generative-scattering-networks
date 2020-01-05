"""
Author: ta
Date and time: 20/11/2018 - 23:56
"""
import os

import numpy as np
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


class EmbeddingsDataset(Dataset):
    def __init__(self, dir_z):
        self.nb_files = get_nb_files(dir_z)
        self.dir_z = dir_z

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = os.path.join(self.dir_z, '{}.npy'.format(idx))
        return np.load(filename)


class EmbeddingsImagesDataset(Dataset):
    def __init__(self, dir_z, dir_x, nb_channels=3):
        assert get_nb_files(dir_z) == get_nb_files(dir_x)
        assert nb_channels in [1, 3]

        self.nb_files = get_nb_files(dir_z)

        self.nb_channels = nb_channels

        self.dir_z = dir_z
        self.dir_x = dir_x

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = os.path.join(self.dir_z, '{}.npy'.format(idx))
        z = np.load(filename)

        filename = os.path.join(self.dir_x, '{}.png'.format(idx))
        if self.nb_channels == 3:
            x = ((np.ascontiguousarray(Image.open(filename), dtype=np.uint8)
                  .transpose((2, 0, 1)) / 127.5) - 1.0)
        else:
            x = np.expand_dims(
                np.ascontiguousarray(Image.open(filename), dtype=np.uint8),
                axis=-1)
            x = (x.transpose((2, 0, 1)) / 127.5) - 1.0

        sample = {'z': z, 'x': x}
        return sample


class EmbeddingsTransformDataset(Dataset):
    def __init__(
        self, dir_z, dir_x, transform,
        nb_channels=3,
        file_format="jpg",
        nb_files=None
    ):
        if nb_files is None:
            nb_file_z = get_nb_files(dir_z)
            assert nb_file_z == get_nb_files(dir_x)
            self.nb_files = nb_file_z
        else:
            # useful when working on colab
            # colab cannot read folder with too many files
            self.nb_files = nb_files

        assert nb_channels in [1, 3]
        self.nb_channels = nb_channels

        self.dir_z = dir_z
        self.dir_x = dir_x
        self.file_format = file_format
        self.transform = transform

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = os.path.join(self.dir_z, '{}.npy'.format(idx))
        z = np.load(filename)

        filename = os.path.join(self.dir_x, '{}.{}'.format(idx, self.file_format))

        if self.nb_channels == 3:
            x = (self.transform(self.pil_loader(filename)) / 127.5) - 1
        else:
            x = ((self.transform(self.pil_loader(filename)) / 127.5) - 1).unsqueeze(0)

        return {'z': z, 'x': x}
