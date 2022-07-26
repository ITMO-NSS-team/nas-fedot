import os
from nas.utils.utils import project_root

project_root = project_root()
tests_root = os.path.join(project_root, 'tests', 'unit', 'test')
verbose_values = {0: -1, 'auto': 1, 1: 1, 2: 1}
batch_norm_probability = 0.4
default_nodes_params = {
    'conv2d': {'layer_type': 'conv2d', 'activation': 'relu', 'conv_kernel_size': [3, 3],
               'conv_strides': [2, 2], 'num_of_filters': 16, 'pool_size': [2, 2],
               'pool_strides': [2, 2], 'pool_type': 'max_pool2d'},
    'dropout': {'drop': 0.2},
    'batch_normalization': {'epsilon': 0.001, 'momentum': 0.99},
    'dense': {'activation': 'relu', 'layer_type': 'dense', 'neurons': 121},
    'serial_connection': {'layer_type': 'serial_connection'},
    'flatten': None}
