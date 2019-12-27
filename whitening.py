from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EmbeddingsDataset


def compute_withening(
    dir_to_save,
    dir_z,
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
        for idx_batch, current_batch in\
                enumerate(tqdm(dataloader, desc='Fitting the PCA')):
            current_batch = current_batch.view(current_batch.shape[0], -1).numpy()
            if current_batch.shape[0] >= PCA.n_components:
                PCA = PCA.partial_fit(current_batch)

    dir_to_save.mkdir()
    for idx_batch, current_batch in\
            enumerate(tqdm(dataloader, desc='Computing Whitening')):
        current_batch = current_batch.view(current_batch.shape[0], -1).numpy()
        current_batch = PCA.transform(current_batch)

        for idx_local in range(current_batch.shape[0]):
            idx_global = idx_local + idx_batch * batch_size
            filename = dir_to_save / '{}.npy'.format(idx_global)
            temp = current_batch[idx_local]
            np.save(filename, temp)


if __name__ == '__main__':
    N_COMPONENTS = 400
    BATCH_SIZE = 512
    assert BATCH_SIZE >= N_COMPONENTS

    NUM_WORKERS = 3

    home_datasets = Path('~/datasets/').expanduser()
    embedding_attribute = '1024_rgb_SJ4'
    dir_dataset = home_datasets / 'celeba_hq' / embedding_attribute

    whitened = home_datasets / 'celeba_hq' / f'{embedding_attribute}_PCA_{N_COMPONENTS}'
    whitened.mkdir()

    PCA = IncrementalPCA(n_components=N_COMPONENTS, whiten=True)
    compute_withening(
        whitened / 'train',
        dir_dataset / 'train',
        PCA,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        training=True,
    )

    compute_withening(
        whitened / 'test',
        dir_dataset / 'test',
        PCA,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        training=False,
    )
