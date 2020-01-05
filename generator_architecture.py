# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor

from config import DATASETS_PATH
from datasets import EmbeddingsTransformDataset


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)


class Generator(nn.Module):
    def __init__(
        self,
        nb_channels_first_layer,
        z_dim,
        size_first_layer=4,
        num_channel=3,
        linear_bias=True,
        conv_kernel=5,
    ):
        super(Generator, self).__init__()

        nb_channels_input = nb_channels_first_layer * 32

        self.main = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=size_first_layer * size_first_layer * nb_channels_input,
                      bias=linear_bias),
            View(-1, nb_channels_input, size_first_layer, size_first_layer),
            nn.BatchNorm2d(nb_channels_input, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),

            ConvBlock(nb_channels_first_layer * 32, nb_channels_first_layer * 16,
                      conv_kernel=conv_kernel,   upsampling=True),
            ConvBlock(nb_channels_first_layer * 16, nb_channels_first_layer * 8,
                      conv_kernel=conv_kernel, upsampling=True),
            ConvBlock(nb_channels_first_layer * 8, nb_channels_first_layer * 4,
                      conv_kernel=conv_kernel, upsampling=True),
            ConvBlock(nb_channels_first_layer * 4, nb_channels_first_layer * 2,
                      conv_kernel=conv_kernel, upsampling=True),
            ConvBlock(nb_channels_first_layer * 2, nb_channels_first_layer,
                      conv_kernel=conv_kernel, upsampling=True),
            ConvBlock(nb_channels_first_layer, nb_channels_output=num_channel,
                      conv_kernel=conv_kernel, tanh=True)
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


class ConvBlock(nn.Module):
    def __init__(
        self,
        nb_channels_input,
        nb_channels_output,
        conv_kernel=5,
        upsampling=False,
        tanh=False,
    ):
        super(ConvBlock, self).__init__()

        self.tanh = tanh
        self.upsampling = upsampling

        padding = (conv_kernel - 1) // 2

        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(nb_channels_input, nb_channels_output, conv_kernel, bias=False)
        self.bn_layer = nn.BatchNorm2d(nb_channels_output, eps=0.001, momentum=0.9)

    def forward(self, input_tensor):
        if self.upsampling:
            output = F.interpolate(input_tensor, scale_factor=2,
                                   mode='bilinear', align_corners=False)
        else:
            output = input_tensor

        output = self.pad(output)
        output = self.conv(output)
        output = self.bn_layer(output)

        if self.tanh:
            output = torch.tanh(output)
        else:
            output = F.relu(output)

        return output


def weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


if __name__ == '__main__':
    dir_datasets = Path(DATASETS_PATH).expanduser()
    dataset = 'celeba_hq'
    dataset_attribute = '1024_rgb'
    embedding_attribute = 'SJ4_PCA_200'

    dir_x_train = dir_datasets / dataset / '{0}'.format(dataset_attribute) / 'train'
    dir_z_train = (dir_datasets
                   / dataset
                   / '{0}_{1}'.format(dataset_attribute, embedding_attribute)
                   / 'train')

    operations = [
        Resize((128, 128)),
        ToTensor(),
    ]
    transform = Compose(operations)

    dataset = EmbeddingsTransformDataset(
        dir_z_train, dir_x_train, transform, file_format="jpg")
    fixed_dataloader = DataLoader(dataset, batch_size=30)
    fixed_batch = next(iter(fixed_dataloader))

    CHANNELS_LINER_LAYER = 64
    Z_DIM = 200
    SIZE_LINEAR_LAYER = 4

    input_tensor = fixed_batch['z'].float()  # .cuda()
    g = Generator(
        CHANNELS_LINER_LAYER,
        Z_DIM,
        size_first_layer=SIZE_LINEAR_LAYER,
        num_channel=3,
        conv_kernel=5,
    )
    g.apply(weights_init)
    # g.cuda()
    g.train()
    with torch.no_grad():
        output = g.forward(input_tensor)
    save_image(output[:16].data, 'tmp/temp.png', nrow=4)
