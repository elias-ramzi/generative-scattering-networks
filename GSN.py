# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Resize
from tqdm import tqdm

from generator_architecture import Generator, weights_init
from datasets import EmbeddingsTransformDataset
from utils import create_folder, AverageMeter, now, get_hms, uint8_image


class GSN:
    def __init__(self, parameters):
        dir_datasets = Path('~/datasets').expanduser()
        dir_experiments = Path('~/experiments').expanduser()

        dataset = parameters['dataset']
        dataset_attribute = parameters['dataset_attribute']
        embedding_attribute = parameters['embedding_attribute']

        self.image_size = parameters['image_size']
        self.transorm = Compose([
            Resize((self.image_size, self.image_size)),
            ToTensor(),
        ])

        self.dim = parameters['dim']
        self.nb_channels_first_layer = parameters['nb_channels_first_layer']
        self.size_first_layer = parameters['size_first_layer']
        self.num_channel = parameters['num_channel']
        self.linear_bias = parameters['linear_bias']
        self.conv_kernel = parameters['conv_kernel']

        name_experiment = parameters['name_experiment']

        self.num_workers = parameters['num_workers']

        self.dir_x_train = dir_datasets / dataset / dataset_attribute / 'train'
        self.dir_x_test = dir_datasets / dataset / dataset_attribute / 'test'
        self.dir_z_train = (dir_datasets / dataset
                            / '{0}_{1}'.format(dataset_attribute, embedding_attribute)
                            / 'train')
        self.dir_z_test = (dir_datasets / dataset
                           / '{0}_{1}'.format(dataset_attribute, embedding_attribute)
                           / 'test')

        self.dir_experiment = dir_experiments / 'gsn' / name_experiment
        self.dir_models = self.dir_experiment / 'models'
        self.dir_logs_train = self.dir_experiment / 'logs_train'
        self.dir_logs_test = self.dir_experiment / 'logs_test'

        self.batch_size = parameters['batch_size']
        self.nb_epochs_to_save = 1

    def make_dirs(self):
        self.dir_experiment.mkdir(parents=True)
        self.dir_models.mkdir()
        self.dir_logs_train.mkdir()
        self.dir_logs_test.mkdir()

    def instantiate_generator(self):
        return Generator(
            self.nb_channels_first_layer,
            self.dim,
            self.size_first_layer,
            self.num_channel,
            self.linear_bias,
            self.conv_kernel,
        )

    def train(self, epoch_to_restore=0):
        g = self.instantiate_generator()

        if epoch_to_restore == 0:
            self.make_dirs()
            g.apply(weights_init)
        else:
            filename_model = self.dir_models / 'epoch_{}.pth'.format(epoch_to_restore)
            g.load_state_dict(torch.load(filename_model))

        g.cuda()
        g.train()

        dataset_train = EmbeddingsTransformDataset(self.dir_z_train, self.dir_x_train,
                                                   self.transorm)
        dataloader_train = DataLoader(dataset_train, self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, pin_memory=True)
        dataset_test = EmbeddingsTransformDataset(self.dir_z_test, self.dir_x_test,
                                                  self.transorm)
        dataloader_test = DataLoader(dataset_test, self.batch_size, shuffle=True,
                                     num_workers=self.num_workers, pin_memory=True)

        criterion = torch.nn.L1Loss()

        optimizer = optim.Adam(g.parameters())
        writer_train = SummaryWriter(log_dir=str(self.dir_logs_train))
        writer_test = SummaryWriter(log_dir=str(self.dir_logs_test))

        sample = next(iter(dataloader_train))
        writer_train.add_graph(g, sample['z'].float().cuda())

        try:
            epoch = epoch_to_restore
            while True:
                start_time = time.time()

                train_l1_loss = AverageMeter()
                g.train()
                for _ in range(self.nb_epochs_to_save):
                    epoch += 1

                    for idx_batch, current_batch in enumerate(tqdm(
                        dataloader_train,
                        desc=f'Training for epoch {epoch}')
                    ):
                        g.zero_grad()
                        x = Variable(current_batch['x']).float().cuda()
                        z = Variable(current_batch['z']).float().cuda()
                        g_z = g.forward(z)

                        loss = criterion(g_z, x)
                        loss.backward()
                        optimizer.step()
                        train_l1_loss.update(loss)
                        writer_train.add_scalar('metrics/l1_loss', train_l1_loss.avg, epoch)

                g.eval()
                with torch.no_grad():
                    test_l1_loss = AverageMeter()
                    for idx_batch, current_batch in\
                            enumerate(tqdm(dataloader_test, desc='Testing model')):
                        if idx_batch == 32:
                            break
                        x = current_batch['x'].float().cuda()
                        z = current_batch['z'].float().cuda()
                        g_z = g.forward(z)
                        loss = criterion(g_z, x)
                        test_l1_loss.update(loss)

                    writer_test.add_scalar('metrics/l1_loss', test_l1_loss.avg, epoch)
                    images = make_grid(g_z.data[:16], nrow=4, normalize=True)
                    writer_test.add_image('generations', images, epoch)

                    print(" Validation : Train Loss : {:.4f}, Test Loss : {:.4f}"
                          .format(train_l1_loss.avg, test_l1_loss.avg))

                if epoch % self.nb_epochs_to_save == 0:
                    filename = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
                    torch.save(g.state_dict(), filename)

                end_time = time.time()
                print("[*] Finished epoch {} in {}"
                      .format(epoch, get_hms(end_time - start_time)))

        finally:
            print('[*] Closing Writer.')
            writer_train.close()
            writer_test.close()

    def save_originals(self):
        def _save_originals(dir_z, dir_x, train_test):
            dataset = EmbeddingsTransformDataset(dir_z, dir_x, self.transorm)
            fixed_dataloader = DataLoader(dataset, 16)
            fixed_batch = next(iter(fixed_dataloader))

            temp = make_grid(fixed_batch['x'], nrow=4).numpy().transpose((1, 2, 0))

            filename_images = os.path.join(
                self.dir_experiment,
                'originals_{}.png'.format(train_test))
            Image.fromarray(uint8_image(temp)).save(filename_images)

        _save_originals(self.dir_z_train, self.dir_x_train, 'train')
        _save_originals(self.dir_z_test, self.dir_x_test, 'test')

    def compute_errors(self, epoch):
        filename_model = self.dir_models / 'epoch_{}.pth'.format(epoch)
        g = self.instantiate_generator()
        g.cuda()
        g.load_state_dict(torch.load(filename_model))
        g.eval()

        with torch.no_grad():
            criterion = torch.nn.MSELoss()

            def _compute_error(dir_z, dir_x, train_test):
                dataset = EmbeddingsTransformDataset(dir_z, dir_x, self.transorm)
                dataloader = DataLoader(
                    dataset, batch_size=512, num_workers=4, pin_memory=True
                )

                error = 0

                for idx_batch, current_batch in enumerate(tqdm(dataloader)):
                    x = current_batch['x'].float().cuda()
                    z = current_batch['z'].float().cuda()
                    g_z = g.forward(z)

                    error += criterion(g_z, x).data.cpu().numpy()

                error /= len(dataloader)

                print('Error for {}: {}'.format(train_test, error))

            _compute_error(self.dir_z_train, self.dir_x_train, 'train')
            _compute_error(self.dir_z_test, self.dir_x_test, 'test')

    def get_generator(self, epoch_to_load):
        filename_model = self.dir_models / 'epoch_{}.pth'.format(epoch_to_load)
        g = self.instantiate_generator()
        g.load_state_dict(torch.load(filename_model))
        # g.cuda()
        g.eval()
        return g

    def conditional_generation(self, epoch_to_load, idx_image, z_initial_idx, z_end_idx):
        g = self.get_generator(epoch_to_load)

        dir_to_save = (
            self.dir_experiment
            / 'conditional_generation_epoch{}_img{}_zi{}_ze{}_{}'.format(
                epoch_to_load,
                idx_image,
                z_initial_idx,
                z_end_idx,
                now()))
        dir_to_save.mkdir()

        with torch.no_grad():
            def _generate_random(dir_z, dir_x):
                dataset = EmbeddingsTransformDataset(dir_z, dir_x, self.transorm)
                fixed_dataloader = DataLoader(dataset, idx_image + 1, shuffle=False)
                fixed_batch = next(iter(fixed_dataloader))

                x = fixed_batch['x'][[idx_image]]
                filename_images = os.path.join(
                    dir_to_save,
                    'original.png'.format(epoch_to_load))
                temp = make_grid(x.data, nrow=1).cpu().numpy().transpose((1, 2, 0))
                Image.fromarray(uint8_image(temp)).save(filename_images)

                z0 = fixed_batch['z'][[idx_image]].numpy()
                nb_samples = 16
                batch_z = np.repeat(z0, nb_samples, axis=0)
                batch_z[:, z_initial_idx:z_end_idx] =\
                    np.random.randn(nb_samples, z_end_idx - z_initial_idx)
                z = torch.from_numpy(batch_z).float().cuda()

                g_z = g.forward(z)
                filename_images = os.path.join(
                    dir_to_save,
                    'modified.png'.format(epoch_to_load))
                temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
                Image.fromarray(uint8_image(temp)).save(filename_images)

            _generate_random(self.dir_z_train, self.dir_x_train)

    def generate_from_model(self, epoch_to_load):
        g = self.get_generator(epoch_to_load)

        dir_to_save = (self.dir_experiment
                       / 'generations_epoch{}_{}'.format(epoch_to_load, now()))
        dir_to_save.mkdir()

        with torch.no_grad():
            def _generate_from_model(dir_z, dir_x, train_test):
                dataset = EmbeddingsTransformDataset(dir_z, dir_x, self.transorm)
                fixed_dataloader = DataLoader(dataset, 16)
                fixed_batch = next(iter(fixed_dataloader))

                z = fixed_batch['z'].float().cuda()
                g_z = g.forward(z)
                filename_images = (dir_to_save
                                   / 'epoch_{}_{}.png'.format(epoch_to_load, train_test))
                temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
                Image.fromarray(uint8_image(temp)).save(filename_images)

            _generate_from_model(self.dir_z_train, self.dir_x_train, 'train')
            _generate_from_model(self.dir_z_test, self.dir_x_test, 'test')

            def _generate_path(dir_z, dir_x, train_test):
                dataset = EmbeddingsTransformDataset(dir_z, dir_x, self.transorm)
                fixed_dataloader = DataLoader(dataset, 2, shuffle=True)
                fixed_batch = next(iter(fixed_dataloader))

                z0 = fixed_batch['z'][[0]].numpy()
                z1 = fixed_batch['z'][[1]].numpy()

                batch_z = np.copy(z0)

                nb_samples = 100

                interval = np.linspace(0, 1, nb_samples)
                for t in interval:
                    if t > 0:
                        # zt = normalize((1 - t) * z0 + t * z1)
                        zt = (1 - t) * z0 + t * z1
                        batch_z = np.vstack((batch_z, zt))

                z = torch.from_numpy(batch_z).float().cuda()
                g_z = g.forward(z)

                # filename_images = os.path.join(self.dir_experiment, 'path_epoch_{}_{}.png'
                # .format(epoch, train_test))
                # temp = make_grid(g_z.data, nrow=nb_samples).cpu().numpy().transpose((1, 2, 0))
                # Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

                g_z = g_z.data.cpu().numpy().transpose((0, 2, 3, 1))

                folder_to_save = (dir_to_save
                                  / 'epoch_{}_{}_path'.format(epoch_to_load, train_test))
                create_folder(folder_to_save)

                for idx in range(nb_samples):
                    filename_image = os.path.join(folder_to_save, '{}.png'.format(idx))
                    Image.fromarray(uint8_image(g_z[idx])).save(filename_image)

            _generate_path(self.dir_z_train, self.dir_x_train, 'train')
            _generate_path(self.dir_z_test, self.dir_x_test, 'test')

            def _generate_random():
                nb_samples = 16
                z = np.random.randn(nb_samples, self.dim)
                # norms = np.sqrt(np.sum(z ** 2, axis=1))
                # norms = np.expand_dims(norms, axis=1)
                # norms = np.repeat(norms, self.dim, axis=1)
                # z /= norms

                z = torch.from_numpy(z).float().cuda()
                g_z = g.forward(z)
                filename_images = os.path.join(
                    dir_to_save,
                    'epoch_{}_random.png'.format(epoch_to_load))
                temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
                Image.fromarray(uint8_image(temp)).save(filename_images)

            _generate_random()

    def analyze_model(self, epoch):
        filename_model = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
        g = self.instantiate_generator()
        g.cuda()
        g.load_state_dict(torch.load(filename_model))
        g.eval()

        nb_samples = 50
        batch_z = np.zeros((nb_samples, 32 * self.nb_channels_first_layer, 4, 4))
        # batch_z = np.maximum(5*np.random.randn(
        # nb_samples, 32 * self.nb_channels_first_layer, 4, 4), 0)
        # batch_z = 5 * np.random.randn(nb_samples, 32 * self.nb_channels_first_layer, 4, 4)

        for i in range(4):
            for j in range(4):
                batch_z[:, :, i, j] = create_path(nb_samples)
        # batch_z[:, :, 0, 0] = create_path(nb_samples)
        # batch_z[:, :, 0, 1] = create_path(nb_samples)
        # batch_z[:, :, 1, 0] = create_path(nb_samples)
        # batch_z[:, :, 1, 1] = create_path(nb_samples)
        batch_z = np.maximum(batch_z, 0)

        z = Variable(torch.from_numpy(batch_z)).type(torch.FloatTensor).cuda()
        temp = g.main._modules['4'].forward(z)
        for i in range(5, 10):
            temp = g.main._modules['{}'.format(i)].forward(temp)

        g_z = temp.data.cpu().numpy().transpose((0, 2, 3, 1))

        folder_to_save = os.path.join(
            self.dir_experiment,
            'epoch_{}_path_after_linear_only00_path'.format(epoch))
        create_folder(folder_to_save)

        for idx in range(nb_samples):
            filename_image = os.path.join(folder_to_save, '{}.png'.format(idx))
            Image.fromarray(uint8_image(g_z[idx])).save(filename_image)


def create_path(nb_samples):
    z0 = 5 * np.random.randn(1, 32 * 32)
    z1 = 5 * np.random.randn(1, 32 * 32)

    # z0 = np.zeros((1, 32 * 32))
    # z1 = np.zeros((1, 32 * 32))

    # z0[0, 0] = -20
    # z1[0, 0] = 20

    batch_z = np.copy(z0)

    interval = np.linspace(0, 1, nb_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))

    return batch_z
