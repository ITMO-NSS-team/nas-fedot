verbose_values = {0: -1, 'auto': 1, 1: 1, 2: 1}

batch_norm_probability = 0.4

default_nodes_params = {
    'conv2d': {'layer_type': 'conv2d', 'activation': 'relu', 'kernel_size': [3, 3],
               'conv_strides': [2, 2], 'neurons': 16, 'pool_size': [2, 2],
               'pool_strides': [2, 2], 'pool_type': 'max_pool2d'},
    'dropout': {'drop': 0.2},
    'batch_normalization': {'epsilon': 0.001, 'momentum': 0.99},
    'dense': {'activation': 'relu', 'layer_type': 'dense', 'neurons': 121},
    'serial_connection': {'layer_type': 'serial_connection'},
    'flatten': None}

default_split_params = {
    'k_fold': {'n_splits': 5, 'shuffle': True},
    'holdout': {'train_size': 0.8, 'random_state': 42}}
