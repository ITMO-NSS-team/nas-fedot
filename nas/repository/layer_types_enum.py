from enum import Enum


class FrameworkTypesEnum(Enum):
    keras = 'keras'
    torch = 'torch'


class LayersPoolEnum(Enum):
    conv2d = 'conv2d'
    conv2d_1x1 = 'conv2d_1x1'
    conv2d_3x3 = 'conv2d_3x3'
    conv2d_5x5 = 'conv2d_5x5'
    conv2d_7x7 = 'conv2d_7x7'
    dilation_conv2d = 'dilation_conv2d'
    flatten = 'flatten'
    dense = 'dense'
    dropout = 'dropout'
    batch_norm = 'batch_norm'


