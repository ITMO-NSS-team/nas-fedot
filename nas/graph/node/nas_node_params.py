from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

import tensorflow as tf

from nas.repository.layer_types_enum import LayersPoolEnum

if TYPE_CHECKING:
    from nas.composer.requirements import ModelRequirements


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
    def _get_pool_params(requirements: ModelRequirements) -> dict:
        layer_params = dict()
        layer_params['pool_size'] = random.choice(requirements.conv_requirements.pool_size)
        layer_params['pool_strides'] = random.choice(requirements.conv_requirements.pool_strides)
        return layer_params

    @staticmethod
    def _max_pool2d(requirements: ModelRequirements) -> dict:
        layer_params = GraphLayers._get_pool_params(requirements)
        return layer_params

    @staticmethod
    def _average_pool2d(requirements: ModelRequirements) -> dict:
        layer_params = GraphLayers._get_pool_params(requirements)
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
            LayersPoolEnum.conv2d_1x1: self._conv2d_1x1,
            LayersPoolEnum.conv2d_3x3: self._conv2d_3x3,
            LayersPoolEnum.conv2d_5x5: self._conv2d_5x5,
            LayersPoolEnum.conv2d_7x7: self._conv2d_7x7,
            LayersPoolEnum.dilation_conv2d: self._dilation_conv2d,
            LayersPoolEnum.flatten: self._flatten,
            LayersPoolEnum.dense: self._dense,
            LayersPoolEnum.dropout: self._dropout,
            LayersPoolEnum.max_pool2d: self._max_pool2d,
            LayersPoolEnum.average_poold2: self._average_pool2d
        }

        if layer_type in layers:
            return layers[layer_type](requirements)
        else:
            raise NotImplementedError


class KerasLayersEnum(Enum):
    conv2d = tf.keras.layers.Conv2D
    dense = tf.keras.layers.Dense
    dilation_conv = partial(tf.keras.layers.Conv2D, dilation_rate=(2, 2))
    flatten = tf.keras.layers.Flatten
    batch_normalization = tf.keras.layers.BatchNormalization
    dropout = tf.keras.layers.Dropout
