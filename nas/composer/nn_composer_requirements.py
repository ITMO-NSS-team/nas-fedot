from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Optional, Union

from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.genetic.operators.mutation import MutationStrengthEnum

from nas.repository.layer_types_enum import LayersPoolEnum, ActivationTypesIdsEnum


def load_default_requirements() -> NNComposerRequirements:
    primary_nodes_list = [LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_5x5,
                          LayersPoolEnum.conv2d_7x7]
    fc_requirements = FullyConnectedRequirements()
    conv_requirements = ConvRequirements(input_data_shape=[64, 64])
    model_requirements = ModelRequirements(fc_requirements, conv_requirements, primary=primary_nodes_list, epochs=20, )
    requirements = NNComposerRequirements(5, model_requirements=model_requirements, opt_epochs=5)
    return requirements


@dataclass
class FullyConnectedRequirements:
    min_number_of_neurons: int = 32
    max_number_of_neurons: int = 256

    def __post_init__(self):
        if self.min_number_of_neurons < 2:
            raise ValueError(f'min_num_of_neurons value {self.min_number_of_neurons} is unacceptable')
        if self.max_number_of_neurons < 2:
            raise ValueError(f'max_num_of_neurons value {self.max_number_of_neurons} is unacceptable')

    @property
    def neurons_num(self) -> List[int]:
        neurons = [self.min_number_of_neurons]
        i = self.min_number_of_neurons
        while i < self.max_number_of_neurons:
            i *= 2
            neurons.append(i)
        return neurons


@dataclass
class ConvRequirements:
    input_data_shape: List[Union[int, float], Union[int, float]]

    conv_strides: List[List[int]] = None
    pool_size: List[List[int]] = None
    pool_strides: List[List[int]] = None
    dilation_rate: List[int] = None
    color_mode: Optional[str] = 'color'
    min_filters_num: int = 32
    max_filters_num: int = 128

    def __post_init__(self):
        if not self.dilation_rate:
            self.dilation_rate = [1]
        if self.min_filters_num < 2:
            raise ValueError(f'min_filters value {self.min_filters_num} is unacceptable.')
        if self.max_filters_num < 2:
            raise ValueError(f'max_filters value {self.max_filters_num} is unacceptable.')
        if not self.conv_strides:
            self.conv_strides = [[1, 1]]
        if not all([side_size >= 3 for side_size in self.input_data_shape]):
            raise ValueError(f'Specified image size {self.input_data_shape} is unacceptable.')
        if not self.channels_num:
            raise ValueError(f'Wrong color mode.')

    def set_output_shape(self, output_shape: int) -> ConvRequirements:
        # TODO add output shape check
        self.max_filters_num = output_shape
        self.min_filters_num = output_shape
        return self

    def set_conv_params(self, stride: int) -> ConvRequirements:
        self.conv_strides = [[stride, stride]]
        return self

    def set_pooling_params(self, stride: int, size: int) -> ConvRequirements:
        self.pool_size = [[size, size]]
        self.pool_strides = [[stride, stride]]
        return self

    @staticmethod
    def _get_image_channels_num(color_mode) -> int:
        channels_dict = {'color': 3,
                         'grayscale': 1}
        return channels_dict.get(color_mode)

    @property
    def channels_num(self) -> int:
        color_mode = str.lower(self.color_mode)
        return self._get_image_channels_num(color_mode)

    @property
    def input_shape(self) -> List[Union[int, float], Union[int, float], int]:
        return [*self.input_data_shape, self.channels_num]

    @property
    def filters_num(self) -> List[int]:
        filters_num = [self.min_filters_num]
        i = self.min_filters_num
        while i < self.max_filters_num:
            i = i * 2
            filters_num.append(i)
        return filters_num


@dataclass
class ModelRequirements:
    num_of_classes: int
    conv_requirements: ConvRequirements
    fc_requirements: FullyConnectedRequirements = FullyConnectedRequirements()

    primary: Optional[List[LayersPoolEnum]] = None
    secondary: Optional[List[LayersPoolEnum]] = None

    activation_types: List[ActivationTypesIdsEnum] = None

    _batch_norm_prob: float = .5
    _dropout_prob: float = .5
    _max_dropout_val: float = .5
    _has_skip_connection: Optional[bool] = False

    epochs: int = 1
    batch_size: int = 32

    max_num_of_conv_layers: int = 6
    min_num_of_conv_layers: int = 4

    max_nn_depth: int = 3
    min_nn_depth: int = 1

    def __post_init__(self):
        if not self.primary:
            self.primary = [LayersPoolEnum.conv2d_3x3]
        if not self.secondary:
            self.secondary = [LayersPoolEnum.dropout, LayersPoolEnum.batch_norm, LayersPoolEnum.dense,
                              LayersPoolEnum.max_pool2d, LayersPoolEnum.average_poold2]
        if not self.activation_types:
            self.activation_types = [activation_func for activation_func in ActivationTypesIdsEnum]
        if self.epochs < 1:
            raise ValueError('Epoch number must be at least 1 or greater')
        if self.max_dropout_val >= 1:
            raise ValueError(f'max_drop_size value {self.max_dropout_val} is unacceptable')

    @property
    def batch_norm_prob(self) -> float:
        return self._batch_norm_prob

    def set_batch_norm_prob(self, prob: float) -> ModelRequirements:
        self._batch_norm_prob = prob
        return self

    @property
    def dropout_prob(self) -> float:
        return self._dropout_prob

    def set_dropout_prob(self, prob: float) -> ModelRequirements:
        self._dropout_prob = prob
        return self

    @property
    def max_dropout_val(self) -> float:
        return self._max_dropout_val

    def set_max_dropout_val(self, val: float) -> ModelRequirements:
        if 0 < val < 1:
            self._max_dropout_val = val
        else:
            raise ValueError('Given dropout value is unacceptable.')
        return self

    @property
    def has_skip_connection(self) -> bool:
        return self._has_skip_connection

    @property
    def max_possible_depth(self):
        return self.max_nn_depth + self.max_num_of_conv_layers


@dataclass
class NNComposerRequirements(PipelineComposerRequirements):
    model_requirements: ModelRequirements = None
    opt_epochs: int = None

    def __post_init__(self):
        # TODO type fix
        self.primary = self.model_requirements.primary
        self.secondary = self.model_requirements.secondary
        self.max_depth = self.model_requirements.max_possible_depth
        self.mutation_strength = MutationStrengthEnum.strong


if __name__ == '__main__':
    r = load_default_requirements()
    print('D!')
