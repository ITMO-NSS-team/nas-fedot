import os
from nas.utils.utils import project_root

PROJECT_ROOT = project_root()
TESTING_ROOT = os.path.join(PROJECT_ROOT, 'tests', 'unit', 'test_data')
VERBOSE_VAL = {0: -1, 'auto': 1, 1: 1, 2: 1}
BATCH_NORM_PROB = 0.4
DEFAULT_NODES_PARAMS = {
    'conv2d': {'layer_type': 'conv2d', 'activation': 'relu', 'kernel_size': [3, 3],
               'conv_strides': [2, 2], 'num_of_filters': 16, 'pool_size': [2, 2],
               'pool_strides': [2, 2], 'pool_type': 'max_pool2d'},
    'dropout': {'layer_type': 'dropout',
                'drop': 0.2},
    'batch_normalization': {'layer_type': 'batch_normalization', 'epsilon': 0.001, 'momentum': 0.99},
    'dense': {'activation': 'relu', 'layer_type': 'dense', 'neurons': 121},
    'serial_connection': {'layer_type': 'serial_connection'},
    'flatten': None}
