# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from datasets import EmbeddingsTransformDataset

dir_datasets = Path('~/datasets/').expanduser()
dataset = 'celeba_hq'
dataset_attribute = '1024_rgb'
embedding_attribute = 'SJ4'

transform = Compose([
    Resize((128, 128)),
    ToTensor(),
])

dir_x_train = dir_datasets / dataset / '{0}'.format(dataset_attribute) / 'train'
dir_z_train = dir_datasets / dataset / '{0}_{1}'.format(dataset_attribute, embedding_attribute) / 'train'

dataset = EmbeddingsTransformDataset(dir_z_train, dir_x_train, transform, file_format="jpg")
fixed_dataloader = DataLoader(dataset, batch_size=256)
fixed_batch = next(iter(fixed_dataloader))

x = fixed_batch['x'].numpy()
z = fixed_batch['z'].numpy()

min_distance = np.inf
i_tilde = 0
j_tilde = 0

distances = list()
for i in range(256):
    for j in range(256):
        if i < j:
            temp = (z[i] - z[j]) ** 2
            temp = np.sum(temp)
            temp = np.sqrt(temp)

            if temp < min_distance:
                min_distance = temp
                i_tilde = i
                j_tilde = j

            distances.append(temp)

distances = np.array(distances)
print('Most similar indexes:', i_tilde, j_tilde)

print('Min distances:', distances.min())

plt.hist(distances)
plt.show()
