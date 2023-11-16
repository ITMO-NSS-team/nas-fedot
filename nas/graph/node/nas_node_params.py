from __future__ import annotations

import random
from math import floor
from typing import TYPE_CHECKING, Dict

from nas.repository.layer_types_enum import LayersPoolEnum

random.seed(1)


if TYPE_CHECKING:
    from nas.composer.requirements import ModelRequirements


random.seed(1)


# TODO rewrite with builder pattern
def _get_padding(padding: int, kernel: int):
    """
    This function checks that the given padding value matches to kernel size.
    """
    if floor(kernel / 2) < padding:
        padding = [floor(kernel / 2), floor(kernel / 2)]
    return padding


class NasNodeFactory:
    def __init__(self, requirements: ModelRequirements = None):
        self.global_requirements = requirements

    def get_node_params(self, node_name: LayersPoolEnum, **params):
        supportable_nodes = {'conv2d': self.conv2d,
                             'linear': self.linear,
                             'dropout': self.dropout,
                             'pooling2d': self.pooling,
                             'adaptive_pool2d': self.ada_pool2d,
                             'batch_norm2d': self.batch_normalization,
                             'flatten': self.flatten}
        layer_params_fun = supportable_nodes.get(node_name.value)

        if layer_params_fun is None:
            raise ValueError(f'Wrong node name {node_name}')

        layer_params = layer_params_fun(self.global_requirements, **params)
        bn_prob = .5 if self.global_requirements is None else self.global_requirements.fc_requirements.batch_norm_prob
        bn_cond1 = random.uniform(0, 1) < bn_prob or 'momentum' in params.items()
        bn_cond2 = node_name not in [LayersPoolEnum.adaptive_pool2d, LayersPoolEnum.pooling2d,
                                     LayersPoolEnum.flatten, LayersPoolEnum.dropout, LayersPoolEnum.batch_norm2d]
        if bn_cond1 and bn_cond2:
            layer_params = {**layer_params, **self.batch_normalization(self.global_requirements, **params)}

        return layer_params

    @staticmethod
    def conv2d(requirements: ModelRequirements = None, **kwargs) -> Dict:
        params = {}
        if requirements is not None:
            out_shape = random.choice(requirements.conv_requirements.neurons_num)
            kernel_size = random.choice(requirements.conv_requirements.kernel_size)
            activation = random.choice(requirements.fc_requirements.activation_types).value
            stride = random.choice(requirements.conv_requirements.conv_strides)
            padding = _get_padding(random.choice(requirements.conv_requirements.padding), kernel_size)
        else:
            out_shape = kwargs.get('out_shape')
            kernel_size = kwargs.get('kernel_size')
            activation = kwargs.get('activation', 'relu')
            stride = kwargs.get('stride', [1, 1])
            padding = kwargs.get('padding', 0)
        params['out_shape'] = out_shape
        params['kernel_size'] = kernel_size
        params['activation'] = activation
        params['stride'] = stride
        params['padding'] = padding
        return params

    @staticmethod
    def pooling(requirements: ModelRequirements, **kwargs) -> Dict:
        params = {}
        if requirements is not None:
            pooling_size = random.choice(requirements.conv_requirements.pool_size)
            pooling_stride = random.choice(requirements.conv_requirements.pool_strides)
            mode = random.choice(requirements.conv_requirements.pooling_mode)
            padding = _get_padding(random.choice(requirements.conv_requirements.padding), pooling_size)
        else:
            pooling_size = kwargs.get('pool_size')
            pooling_stride = kwargs.get('pool_stride')
            padding = kwargs.get('padding', 0)
            mode = kwargs.get('mode')
        params['pool_size'] = pooling_size
        params['pool_stride'] = pooling_stride
        params['mode'] = mode
        params['padding'] = padding
        return params

    @staticmethod
    def ada_pool2d(requirements: ModelRequirements = None, **kwargs):
        params = {}
        if requirements is not None:
            out_shape = 1
            mode = random.choice(requirements.conv_requirements.pooling_mode)
        else:
            out_shape = kwargs.get('out_shape')
            mode = kwargs['mode']
        params['out_shape'] = out_shape
        params['mode'] = mode
        return params

    @staticmethod
    def linear(requirements: ModelRequirements = None, **kwargs) -> Dict:
        params = {}
        if requirements is not None:
            out_shape = random.choice(requirements.fc_requirements.neurons_num)
            activation = random.choice(requirements.fc_requirements.activation_types).value
        else:
            out_shape = kwargs.get('out_shape')
            activation = kwargs.get('activation', 'relu')
        params['out_shape'] = out_shape
        params['activation'] = activation
        return params

    @staticmethod
    def batch_normalization(requirements: ModelRequirements = None, **kwargs) -> Dict:
        params = {}
        if requirements is not None:
            momentum = .99
            epsilon = .001
        else:
            momentum = kwargs.get('momentum', .99)
            epsilon = kwargs.get('epsilon', .001)
        params['momentum'] = momentum
        params['epsilon'] = epsilon
        return params

    @staticmethod
    def dropout(requirements: ModelRequirements = None, **kwargs) -> Dict:
        params = {}
        if requirements is not None:
            drop_proba = random.randint(0, int(requirements.fc_requirements.max_dropout_val * 100)) / 100
        else:
            drop_proba = kwargs.get('drop', .8)
        params['drop'] = drop_proba
        return params

    @staticmethod
    def flatten(*args, **kwargs) -> Dict:
        return {}
