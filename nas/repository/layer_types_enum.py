from __future__ import annotations

from enum import Enum


class FrameworkTypesEnum(Enum):
    keras = 'keras'
    torch = 'torch'


class LayersPoolEnum(Enum):
    conv2d = 'conv2d'
    batch_norm2d = 'batch_norm2d'
    dilation_conv2d = 'dilation_conv2d'
    flatten = 'flatten'
    linear = 'linear'
    dropout = 'dropout'
    adaptive_pool2d = 'adaptive_pool2d'
    pooling2d = 'pooling2d'


class ActivationTypesIdsEnum(Enum):
    # softmax = 'softmax'
    elu = 'elu'
    selu = 'selu'
    softplus = 'softplus'
    relu = 'relu'
    softsign = 'softsign'
    tanh = 'tanh'
    hard_sigmoid = 'hard_sigmoid'
    sigmoid = 'sigmoid'
    linear = 'linear'
