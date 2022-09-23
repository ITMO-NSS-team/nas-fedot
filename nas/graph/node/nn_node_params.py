from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

import tensorflow as tf

from nas.repository.layer_types_enum import LayersPoolEnum


if TYPE_CHECKING:
    from nas.composer.nn_composer_requirements import NNComposerRequirements


@dataclass
class GraphLayers:
    @staticmethod
    def _base_conv2d(requirements):
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = dict()

        # layer_parameters['name'] = LayersPoolEnum.conv2d
        layer_parameters['activation'] = random.choice(requirements.activation_types).value
        layer_parameters['conv_strides'] = random.choice(requirements.conv_requirements.conv_strides)
        layer_parameters['num_of_filters'] = random.choice(requirements.conv_requirements.filters)
        layer_parameters['pool_size'] = random.choice(requirements.conv_requirements.pool_size)
        layer_parameters['pool_strides'] = random.choice(requirements.conv_requirements.pool_strides)
        layer_parameters['pool_type'] = random.choice(requirements.conv_requirements.pool_types)
        return layer_parameters

    def _conv2d_1x1(self, requirements):
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = self._base_conv2d(requirements)
        layer_parameters['kernel_size'] = [1, 1]
        return layer_parameters

    def _conv2d_3x3(self, requirements):
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = self._base_conv2d(requirements)
        layer_parameters['kernel_size'] = [3, 3]
        return layer_parameters

    def _conv2d_5x5(self, requirements):
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = self._base_conv2d(requirements)
        layer_parameters['kernel_size'] = [5, 5]
        return layer_parameters

    def _conv2d_7x7(self, requirements):
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = self._base_conv2d(requirements)
        layer_parameters['kernel_size'] = [7, 7]
        return layer_parameters

    def _dilation_conv2d(self, requirements):
        layer_parameters = self._base_conv2d(requirements)

        layer_parameters['dilation_rate'] = random.choice(requirements.conv_requirements.dilation_rate)
        layer_parameters['kernel_size'] = [3, 3]
        return layer_parameters

    @staticmethod
    def _dense(requirements):
        layer_parameters = dict()
        layer_parameters['activation'] = random.choice(requirements.activation_types).value
        layer_parameters['neurons'] = random.choice(requirements.fc_requirements.neurons_num)
        return layer_parameters

    @staticmethod
    def _dropout(requirements):
        layer_parameters = dict()

        layer_parameters['drop'] = random.randint(1, requirements.max_drop_size * 10) / 10
        return layer_parameters

    @staticmethod
    def _flatten(*args):
        return {'n_jobs': 1}

    def layer_by_type(self, layer_type: LayersPoolEnum, requirements: NNComposerRequirements.nn_requirements):
        layers = {
            LayersPoolEnum.conv2d_1x1: self._conv2d_1x1(requirements),
            LayersPoolEnum.conv2d_3x3: self._conv2d_3x3(requirements),
            LayersPoolEnum.conv2d_5x5: self._conv2d_5x5(requirements),
            LayersPoolEnum.conv2d_7x7: self._conv2d_7x7(requirements),
            LayersPoolEnum.dilation_conv2d: self._dilation_conv2d(requirements),
            LayersPoolEnum.flatten: self._flatten(requirements),
            LayersPoolEnum.dense: self._dense(requirements),
            LayersPoolEnum.dropout: self._dropout(requirements)
        }

        if layer_type in layers:
            return layers[layer_type]
        else:
            raise NotImplementedError


class KerasLayersEnum(Enum):
    conv2d = tf.keras.layers.Conv2D
    dense = tf.keras.layers.Dense
    dilation_conv = partial(tf.keras.layers.Conv2D, dilation_rate=(2, 2))
    flatten = tf.keras.layers.Flatten
    batch_normalization = tf.keras.layers.BatchNormalization
    dropout = tf.keras.layers.Dropout
