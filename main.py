# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""
from utils import create_name_experiment
from GSN import GSN

parameters = dict()
parameters['image_size'] = 128

parameters['dataset'] = 'celeba_hq'
parameters['dataset_attribute'] = '1024_rgb'
parameters['dim'] = 200
parameters['embedding_attribute'] = 'SJ4_PCA_{}'.format(parameters['dim'])

parameters['nb_channels_first_layer'] = 64
parameters['size_first_layer'] = 4
parameters['num_channel'] = 3
parameters['linear_bias'] = True
parameters['conv_kernel'] = 7

parameters['num_workers'] = 7
parameters['batch_size'] = 30

parameters['name_experiment'] = create_name_experiment(parameters, 'pilot_dsencoder')

gsn = GSN(parameters)
gsn.train(19)
# gsn.save_originals()
# gsn.generate_from_model(17)
# gsn.compute_errors(17)
# gsn.analyze_model(17)
# gsn.conditional_generation(17, idx_image=76, z_initial_idx=0, z_end_idx=160)
