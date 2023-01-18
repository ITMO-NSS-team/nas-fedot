from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import List, Optional, Union

from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.genetic.operators.mutation import MutationStrengthEnum

from nas.repository.layer_types_enum import LayersPoolEnum, ActivationTypesIdsEnum


def _get_image_channels_num(color_mode) -> int:
    channels_dict = {'color': 3,
                     'grayscale': 1}
    return channels_dict.get(color_mode)


def get_nearest_power_of_2(number: int) -> int:
    if not number & (number - 1):
        return number
    return int('1' + (len(bin(number)) - 2) * '0', 2)


def get_list_of_power_of_2(min_value: int, max_value: int) -> List[int]:
    return [2 ** n for n in range(int(log2(max_value)) + 1) if 2 ** n >= min_value]


def load_default_requirements() -> NNComposerRequirements:
    primary_nodes_list = [LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_5x5,
                          LayersPoolEnum.conv2d_7x7]
    fc_requirements = BaseLayerRequirements()
    conv_requirements = ConvRequirements()
    model_requirements = ModelRequirements(input_data_shape=[64, 64], num_of_classes=5, color_mode='color',
                                           conv_requirements=conv_requirements, fc_requirements=fc_requirements,
                                           primary=primary_nodes_list, epochs=20)
    requirements = NNComposerRequirements(5, model_requirements=model_requirements, opt_epochs=5)
    return requirements


@dataclass
class BaseLayerRequirements:
    min_number_of_neurons: int = 32
    max_number_of_neurons: int = 256

    _batch_norm_prob: float = .5
    _dropout_prob: float = .5
    _max_dropout_val: float = .5

    activation_types: List[ActivationTypesIdsEnum] = None

    def __post_init__(self):
        self.min_number_of_neurons = get_nearest_power_of_2(self.min_number_of_neurons)
        self.max_number_of_neurons = get_nearest_power_of_2(self.max_number_of_neurons)

        if not self.activation_types:
            self.activation_types = [activation_func for activation_func in ActivationTypesIdsEnum]
        if self.max_number_of_neurons < self.min_number_of_neurons:
            raise ValueError('Min number of neurons in requirements can not be greater than max number of neurons.')
        if self.min_number_of_neurons < 2:
            raise ValueError(f'{self.min_number_of_neurons.__name__} of {self.min_number_of_neurons} is unacceptable.')
        if self.max_number_of_neurons < 2:
            raise ValueError(f'{self.max_number_of_neurons.__name__} of {self.max_number_of_neurons} is unacceptable.')
        if self.max_dropout_val >= 1:
            raise ValueError(f'Max dropout value {self.max_dropout_val} is unacceptable')

    @property
    def neurons_num(self) -> List[int]:
        return get_list_of_power_of_2(self.min_number_of_neurons, self.max_number_of_neurons)

    @property
    def batch_norm_prob(self) -> float:
        return self._batch_norm_prob

    def set_batch_norm_prob(self, prob: float) -> BaseLayerRequirements:
        self._batch_norm_prob = prob
        return self

    @property
    def dropout_prob(self) -> float:
        return self._dropout_prob

    def set_dropout_prob(self, prob: float) -> BaseLayerRequirements:
        self._dropout_prob = prob
        return self

    @property
    def max_dropout_val(self) -> float:
        return self._max_dropout_val

    def set_max_dropout_val(self, val: float) -> BaseLayerRequirements:
        if 0 < val < 1:
            self._max_dropout_val = val
        else:
            raise ValueError('Given dropout value is unacceptable.')
        return self


@dataclass
class ConvRequirements(BaseLayerRequirements):
    conv_strides: List[List[int]] = None
    pool_size: List[List[int]] = None
    pool_strides: List[List[int]] = None
    dilation_rate: List[int] = None
    color_mode: Optional[str] = 'color'

    def __post_init__(self):
        if not self.dilation_rate:
            self.dilation_rate = [1]
        if not self.conv_strides:
            self.conv_strides = [[1, 1]]

    def set_output_shape(self, output_shape: int) -> ConvRequirements:
        # TODO add output shape check
        self.max_number_of_neurons = output_shape
        self.min_number_of_neurons = output_shape
        return self

    def set_conv_params(self, stride: int) -> ConvRequirements:
        self.conv_strides = [[stride, stride]]
        return self

    def set_pooling_params(self, stride: int, size: int) -> ConvRequirements:
        self.pool_size = [[size, size]]
        self.pool_strides = [[stride, stride]]
        return self


@dataclass
class ModelRequirements:
    input_data_shape: List[int, int]
    conv_requirements: ConvRequirements
    fc_requirements: BaseLayerRequirements
    color_mode: str = 'color'
    num_of_classes: int = None

    primary: Optional[List[LayersPoolEnum]] = None
    secondary: Optional[List[LayersPoolEnum]] = None

    _has_skip_connection: Optional[bool] = False

    epochs: int = 1
    batch_size: int = 32

    max_num_of_conv_layers: int = 6
    min_num_of_conv_layers: int = 4

    max_nn_depth: int = 3
    min_nn_depth: int = 1

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError(f'{self.epochs} is unacceptable number of train epochs.')
        if not all([side_size >= 3 for side_size in self.input_data_shape]):
            raise ValueError(f'Specified image size {self.input_data_shape} is unacceptable.')
        if not self.primary:
            self.primary = [LayersPoolEnum.conv2d_3x3]
        if not self.secondary:
            self.secondary = [LayersPoolEnum.dropout, LayersPoolEnum.batch_norm, LayersPoolEnum.dense,
                              LayersPoolEnum.max_pool2d, LayersPoolEnum.average_poold2]

    @property
    def channels_num(self) -> int:
        color_mode = str.lower(self.color_mode)
        return _get_image_channels_num(color_mode)

    @property
    def input_shape(self) -> List[Union[int, float], Union[int, float], int]:
        return [*self.input_data_shape, self.channels_num]

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
