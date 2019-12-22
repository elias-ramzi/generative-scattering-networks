import os
from pathlib import Path
from glob import glob

# You should create the folder in your home ~/datasets
# Then download https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv
# Extract the pictures
# rename the folder celeba_hq
# So now you should have something like ~/datasets/celeba_hq
# Now you need to increment the folder to something like :
# ~/datasets/celeba_hq/1024_rgb
# Then you can run the following script which will seperate the picture in Train/Test
# You can change the Train size

TRAIN_SPLIT = 0.9


if __name__ == '__main__':
    dir_datasets = Path('~/datasets/').expanduser()
    dir_celeba = dir_datasets / 'celeba_hq'
    dir_attributes = dir_celeba / '1024_rgb'
    dir_attributes.mkdir()
    dir_train = dir_attributes / 'train'
    dir_train.mkdir()
    dir_test = dir_attributes / 'test'
    dir_test.mkdir()

    paths = glob(str(dir_attributes / '*'))
    paths = sorted(paths, key=lambda pth: pth.split('/')[1].split('.')[0])

    for pth in paths[:int(len(paths)*TRAIN_SPLIT)]:
        new_path = dir_train / os.path.basename(pth)
        os.rename(pth, new_path)

    ext = paths[0].split('.')[-1]
    for i, pth in enumerate(paths[int(len(paths)*TRAIN_SPLIT):]):
        new_path = dir_train / f"{i}.{ext}"
        os.rename(pth, new_path)
