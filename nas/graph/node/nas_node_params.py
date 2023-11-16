from __future__ import annotations

import random
from dataclasses import dataclass
from math import floor
from typing import TYPE_CHECKING, Dict, Sequence

from nas.repository.layer_types_enum import LayersPoolEnum

random.seed(1)


if TYPE_CHECKING:
    from nas.composer.requirements import ModelRequirements


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
                             # 'average_pool2d': self.pooling,
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
            # padding = 'same' if requirements.conv_requirements.padding is None \
            #     else random.choice(requirements.conv_requirements.padding)
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


@dataclass
class GraphLayers:
    @staticmethod
    def _batch_normalization(requirements: ModelRequirements, layer_params: dict) -> dict:
        if random.uniform(0, 1) < requirements.fc_requirements.batch_norm_prob:
            # TODO add mode variety
            layer_params['momentum'] = 0.99
            layer_params['epsilon'] = 0.001
        return layer_params

    @staticmethod
    def _pool2d(requirements: ModelRequirements) -> dict:
        layer_params = dict()
        layer_params['pool_size'] = random.choice(requirements.conv_requirements.pool_size)
        layer_params['pool_strides'] = random.choice(requirements.conv_requirements.pool_strides)
        return layer_params

    @staticmethod
    def _max_pool2d(requirements: ModelRequirements) -> dict:
        layer_params = GraphLayers._pool2d(requirements)
        return layer_params

    @staticmethod
    def _average_pool2d(requirements: ModelRequirements) -> dict:
        layer_params = GraphLayers._pool2d(requirements)
        return layer_params

    @staticmethod
    def _base_conv2d(requirements: ModelRequirements) -> dict:
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = dict()
        layer_parameters['activation'] = random.choice(requirements.fc_requirements.activation_types).value
        layer_parameters['conv_strides'] = random.choice(requirements.conv_requirements.conv_strides)
        layer_parameters['neurons'] = random.choice(requirements.conv_requirements.neurons_num)
        layer_parameters['padding'] = 'same' if not requirements.conv_requirements.padding else random.choice(
            requirements.conv_requirements.padding)
        return GraphLayers._batch_normalization(requirements, layer_parameters)

    @staticmethod
    def _conv2d_1x1(requirements: ModelRequirements) -> dict:
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = GraphLayers._base_conv2d(requirements)
        layer_parameters['kernel_size'] = [1, 1]
        return layer_parameters

    @staticmethod
    def _conv2d_3x3(requirements: ModelRequirements) -> dict:
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = GraphLayers._base_conv2d(requirements)
        layer_parameters['kernel_size'] = [3, 3]
        return layer_parameters

    @staticmethod
    def _conv2d_5x5(requirements: ModelRequirements) -> dict:
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = GraphLayers._base_conv2d(requirements)
        layer_parameters['kernel_size'] = [5, 5]
        return layer_parameters

    @staticmethod
    def _conv2d_7x7(requirements: ModelRequirements) -> dict:
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = GraphLayers._base_conv2d(requirements)
        layer_parameters['kernel_size'] = [7, 7]
        return layer_parameters

    @staticmethod
    def _dilation_conv2d(requirements: ModelRequirements) -> dict:
        raise NotImplementedError(f'Dilation conv layers currently is unsupported')
        #
        # layer_parameters = GraphLayers._base_conv2d(requirements)
        # layer_parameters['dilation_rate'] = random.choice(requirements.conv_requirements.dilation_rate)
        # layer_parameters['kernel_size'] = [3, 3]
        # return layer_parameters

    @staticmethod
    def _dense(requirements: ModelRequirements) -> dict:
        layer_parameters = dict()
        layer_parameters['activation'] = random.choice(requirements.fc_requirements.activation_types).value
        layer_parameters['neurons'] = random.choice(requirements.fc_requirements.neurons_num)
        return GraphLayers._batch_normalization(requirements, layer_parameters)

    @staticmethod
    def _dropout(requirements: ModelRequirements) -> dict:
        layer_parameters = dict()
        layer_parameters['drop'] = random.randint(0, int(requirements.fc_requirements.max_dropout_val * 100)) / 100
        return layer_parameters

    @staticmethod
    def _flatten(*args, **kwargs) -> dict:
        return {'n_jobs': 1}

    def layer_params_by_type(self, layer_type: LayersPoolEnum, requirements: ModelRequirements) -> dict:
        layers = {
            # LayersPoolEnum.conv2d_1x1: self._conv2d_1x1,
            # LayersPoolEnum.conv2d_3x3: self._conv2d_3x3,
            # LayersPoolEnum.conv2d_5x5: self._conv2d_5x5,
            # LayersPoolEnum.conv2d_7x7: self._conv2d_7x7,
            LayersPoolEnum.dilation_conv2d: self._dilation_conv2d,
            LayersPoolEnum.flatten: self._flatten,
            LayersPoolEnum.linear: self._dense,
            LayersPoolEnum.dropout: self._dropout,
            LayersPoolEnum.pooling2d: self._pool2d,
            # LayersPoolEnum.max_pool2d: self._max_pool2d,
            # LayersPoolEnum.average_poold2: self._average_pool2d
        }

        if layer_type in layers:
            return layers[layer_type](requirements)
        else:
            raise NotImplementedError
