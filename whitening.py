import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_nb_files


class EmbeddingsDataset(Dataset):
    def __init__(self, dir_z):
        self.nb_files = get_nb_files(dir_z)
        self.dir_z = dir_z

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = os.path.join(self.dir_z, '{}.npy'.format(idx))
        return np.load(filename)


def compute_withening(
    dir_to_save,
    dir_z,
    Scaler,
    PCA,
    batch_size=256,
    num_workers=6,
    training=False
):
    dataset = EmbeddingsDataset(
        dir_z,
    )
    dataloader = DataLoader(
        dataset,
        batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    if training:
        for idx_batch, current_batch in enumerate(tqdm(dataloader, desc='Fitting the Scaler')):
            current_batch = current_batch.view(current_batch.shape[0], -1).numpy()
            Scaler = Scaler.partial_fit(current_batch)

        for idx_batch, current_batch in enumerate(tqdm(dataloader, desc='Fitting the PCA')):
            current_batch = current_batch.view(current_batch.shape[0], -1).numpy()
            current_batch = Scaler.transform(current_batch)
            PCA = PCA.partial_fit(current_batch)

    dir_to_save.mkdir()
    for idx_batch, current_batch in enumerate(tqdm(dataloader, desc='Computing Whitening')):
        current_batch = current_batch.view(current_batch.shape[0], -1).numpy()
        current_batch = Scaler.transform(current_batch)
        current_batch = PCA.transform(current_batch)

        for idx_local in range(current_batch.shape[0]):
            idx_global = idx_local + idx_batch * batch_size
            filename = dir_to_save / '{}.npy'.format(idx_global)
            temp = current_batch[idx_local]
            np.save(filename, temp)


if __name__ == '__main__':
    N_COMPONENTS = 200
    BATCH_SIZE = 300
    assert BATCH_SIZE > N_COMPONENTS

    home_datasets = Path('~/datasets/').expanduser()
    embedding_attribute = '1024_rgb_SJ4'
    dir_dataset = home_datasets / 'celeba_hq' / embedding_attribute

    whitened = home_datasets / 'celeba_hq' / f'1024_rgb_SJ4_PCA_{N_COMPONENTS}'
    whitened.mkdir()

    Scaler = StandardScaler(with_mean=True, with_std=False)
    PCA = IncrementalPCA(n_components=N_COMPONENTS, whiten=True)
    compute_withening(
        whitened / 'train',
        dir_dataset / 'train',
        Scaler,
        PCA,
        batch_size=BATCH_SIZE,
        num_workers=6,
        training=True,
    )

    compute_withening(
        whitened / 'test',
        dir_dataset / 'test',
        Scaler,
        PCA,
        batch_size=BATCH_SIZE,
        num_workers=6,
        training=False,
    )
