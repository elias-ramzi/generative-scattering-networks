"""
Author: ta
Date and time: 20/11/2018 - 22:59
"""

import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from tqdm import tqdm

from ImageDataset import ImageTransformDataset


class ScateringTransform:

    HOME_DATASET = Path('~/datasets/').expanduser()

    def __init__(
        self,
        dataset_name,
        dataset_attribute,
        J,
        nb_channels=3,
        input_shape=128,
        batch_size=512,
        device="gpu",
        num_workers=1,
        file_format='jpg'
    ):
        """
        parameters
        ----------
        dataset_name : string, name of dataset located in ~/datasets
        dataset_attribute : string, dataset transformed in ~/datasets/<dataset_name>
        J : iterator, different scaling to perform the scattering transform with
        """
        self.dataset_name = dataset_name
        self.dataset_attribute = dataset_attribute
        self.J = J
        assert nb_channels in [1, 3]
        self.nb_channels = nb_channels
        self.input_shape = input_shape
        self.batch_size = batch_size
        assert device.lower() in ['cpu', 'gpu']
        self.device = device
        self.num_workers = num_workers
        self.file_format = file_format

        self.dir_dataset = ScateringTransform.HOME_DATASET / 'celeba_hq' / dataset_attribute
        operations = [
            Resize((self.input_shape, self.input_shape)),
            ToTensor(),
        ]
        self.transform = Compose(operations)

    def __call__(self):
        if self.device == "gpu":
            os.environ["KYMATIO_BACKEND_2D"] = "skcuda"
        import kymatio
        print('[*] Backend for Scattering2D: {}'.format(kymatio.scattering2d.backend.NAME))

        for J in self.J:
            dir_to_save = dir_to_save = (
                ScateringTransform.HOME_DATASET
                / 'celeba_hq'
                / '{}_SJ{}'.format(self.dataset_attribute, J)
            )
            dir_to_save.mkdir()

            if J == 0:
                scattering = None
            else:
                scattering = kymatio.Scattering2D(J, (self.input_shape, self.input_shape))
                if self.device == 'gpu':
                    scattering.cuda()

            self.compute_scattering(self.dir_dataset / 'train', dir_to_save / 'train', scattering)
            self.compute_scattering(self.dir_dataset / 'test', dir_to_save / 'test', scattering)

    def compute_scattering(self, dir_images, dir_to_save, scattering):
        dir_to_save.mkdir()

        dataset = ImageTransformDataset(
            dir_images,
            self.transform,
            nb_channels=self.nb_channels,
            file_format=self.file_format,
        )
        dataloader = DataLoader(
            dataset,
            self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        for idx_batch, current_batch in enumerate(tqdm(dataloader)):
            images = (current_batch.float().cuda() if self.device == 'gpu'
                      else current_batch.float())
            if scattering is not None:
                s_images = scattering(images).cpu().numpy()
                s_images = np.reshape(
                    s_images,
                    (s_images.shape[0], -1, s_images.shape[-1], s_images.shape[-1])
                )
            else:
                s_images = images.cpu().numpy()
            for idx_local in range(s_images.shape[0]):
                idx_global = idx_local + idx_batch * self.batch_size
                filename = dir_to_save / '{}.npy'.format(idx_global)
                temp = s_images[idx_local]
                np.save(filename, temp)


if __name__ == '__main__':
    ScateringTransform(
        'celeba_hq',
        '1024_rgb',
        range(4, 5),
        device='cpu',
        num_workers=6
    )()
