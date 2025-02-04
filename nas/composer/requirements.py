from __future__ import annotations

import copy
from dataclasses import dataclass, field
from math import log2
from typing import List, Optional, Union, Tuple, Collection

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


def get_nearest_multiple_of_power_of_2(number: int, factor: int) -> int:
    """
    Such 2^k * factor that is closest to the given number
    """
    return get_nearest_power_of_2(number // factor) * factor


def get_list_of_multiples_of_power_of_2(min_value: int, max_value: int, factor: int) -> List[int]:
    return [factor * n for n in get_list_of_power_of_2(min_value // factor, max_value // factor)]


def load_default_requirements() -> NNComposerRequirements:
    primary_nodes_list = [LayersPoolEnum.conv2d, LayersPoolEnum.adaptive_pool2d, LayersPoolEnum.pooling2d]
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
            raise ValueError(f'Min number of neurons = {self.min_number_of_neurons} is unacceptable.')
        if self.max_number_of_neurons < 2:
            raise ValueError(f'Max number of neurons = {self.max_number_of_neurons} is unacceptable.')
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
    filter_size_factor: int = 1
    conv_strides: Optional[List[int], Tuple[int]] = None
    pool_size: Optional[List[int], Tuple[int]] = None
    pool_strides: Optional[List[int], Tuple[int]] = None
    pooling_mode: Optional[List[str], Tuple[str]] = None
    dilation_rate: Optional[List[int]] = None
    padding: Union[str, Collection[Collection[int]]] = None
    kernel_size: Union[List[int], Tuple[int]] = None

    supplementary_pooling_prob: float = .5  # Nobody knows which order of magnitude this should be (or pretends so) → it's set to .5

    def __post_init__(self):
        if not self.dilation_rate:
            self.dilation_rate = [1]
        if not self.conv_strides:
            self.conv_strides = [1]
        if not self.pool_size:
            self.pool_size = [2]
        if not self.pool_strides:
            self.pool_strides = copy.deepcopy(self.pool_size)
        if self.pooling_mode is None:
            self.pooling_mode = ['max', 'avg']
        if self.kernel_size is None:
            self.kernel_size = [3, 5, 7]
        if self.padding is None:
            self.padding = [1, 2, 3]

        if not hasattr(self.conv_strides, '__iter__'):
            raise ValueError('Pool of possible strides must be an iterable object')

    @property
    def neurons_num(self) -> List[int]:
        return get_list_of_multiples_of_power_of_2(self.min_number_of_neurons, self.max_number_of_neurons,
                                                   self.filter_size_factor)

    def force_output_shape(self, output_shape: int) -> ConvRequirements:
        self.max_number_of_neurons = output_shape
        self.min_number_of_neurons = output_shape
        return self

    def set_conv_params(self, stride: int) -> ConvRequirements:
        if self.conv_strides:
            self.conv_strides.append([stride, stride])
        else:
            self.conv_strides = [[stride, stride]]
        return self

    def set_pooling_params(self, stride: int, size: int) -> ConvRequirements:
        self.pool_size = [[size, size]]
        self.pool_strides = [[stride, stride]]
        return self

    def set_pooling_stride(self, stride: int) -> ConvRequirements:
        if self.pool_strides:
            self.pool_strides.append([stride, stride])
        else:
            self.force_pooling_stride(stride)
        return self

    def set_pooling_size(self, size: int) -> ConvRequirements:
        if self.pool_size:
            self.pool_size.append([size, size])
        else:
            self.force_pooling_size(size)
        return self

    def force_pooling_size(self, pool_size: int) -> ConvRequirements:
        self.pool_size = [[pool_size, pool_size]]
        return self

    def force_pooling_stride(self, pool_stride: int) -> ConvRequirements:
        self.pool_strides = [[pool_stride, pool_stride]]
        return self

    def force_conv_params(self, stride: int) -> ConvRequirements:
        self.conv_strides = [[stride, stride]]
        return self


@dataclass
class KANSplineRequirements:
    grid_size: List[int] = field(default_factory=lambda: [5, 10])  # 3?
    spline_order: List[int] = field(default_factory=lambda: [3, ])  # 2, 5?
    # Shouldn't depend on torch, no need for excessive hyperparameters → just use SiLU
    # base_activation: List[torch.nn.Module] = field(default_factory=lambda: [torch.nn.SiLU])

    # Auto-scaled anyway
    # grid_range: List[Tuple[int, int]] = field(default_factory=lambda: [(-1, 1)])


@dataclass
class KANLinearRequirements(KANSplineRequirements):
    min_number_of_neurons: int = 32
    max_number_of_neurons: int = 256

    def __post_init__(self):
        self.min_number_of_neurons = get_nearest_power_of_2(self.min_number_of_neurons)
        self.max_number_of_neurons = get_nearest_power_of_2(self.max_number_of_neurons)

        if self.max_number_of_neurons < self.min_number_of_neurons:
            raise ValueError('Min number of neurons in requirements cannot be greater than max number of neurons.')
        if self.min_number_of_neurons < 2:
            raise ValueError(f'Min number of neurons = {self.min_number_of_neurons} is unacceptable.')
        if self.max_number_of_neurons < 2:
            raise ValueError(f'Max number of neurons = {self.max_number_of_neurons} is unacceptable.')

    @property
    def neurons_num(self) -> List[int]:
        return get_list_of_power_of_2(self.min_number_of_neurons, self.max_number_of_neurons)


@dataclass
class KANConvRequirements(KANSplineRequirements):
    min_number_of_neurons: int = 2
    max_number_of_neurons: int = 32
    kernel_size: List[int] = field(default_factory=lambda: [3, 5])

    pooling_prob: float = .8
    pooling_kernel_size: List[int] = field(default_factory=lambda: [2, ])  # Could be 3?

    # Padding would be computed automatically form kernel size to maintain the feature map size:
    # padding: List[int] = field(default_factory=lambda: [0, 1])

    filter_size_factor: int = 3

    def __post_init__(self):
        if self.max_number_of_neurons < self.min_number_of_neurons:
            raise ValueError(
                'Min number of convolutions in requirements cannot be greater than max number of convolutions.')
        if self.min_number_of_neurons < 1:
            raise ValueError(f'Min number of convolutions = {self.min_number_of_neurons} is unacceptable.')
        if self.max_number_of_neurons < 1:
            raise ValueError(f'Max number of convolutions = {self.max_number_of_neurons} is unacceptable.')

    @property
    def neurons_num(self) -> List[int]:
        return get_list_of_multiples_of_power_of_2(self.min_number_of_neurons, self.max_number_of_neurons,
                                                   self.filter_size_factor)


@dataclass
class ModelRequirements:
    input_data_shape: List[int]

    conv_requirements: ConvRequirements = None
    fc_requirements: BaseLayerRequirements = None
    kan_conv_requirements: KANConvRequirements = None
    kan_linear_requirements: KANLinearRequirements = None

    color_mode: str = 'color'
    num_of_classes: int = None

    primary: Optional[List[LayersPoolEnum]] = None
    secondary: Optional[List[LayersPoolEnum]] = None

    epochs: int = 1
    batch_size: int = 32

    max_num_of_conv_layers: int = 6
    min_num_of_conv_layers: int = 4

    max_nn_depth: int = 3
    min_nn_depth: int = 1

    def __post_init__(self):
        if not self.conv_requirements:
            self.conv_requirements = ConvRequirements()
        if not self.fc_requirements:
            self.fc_requirements = BaseLayerRequirements()
        if not self.kan_conv_requirements:
            self.kan_conv_requirements = KANConvRequirements()
        if not self.kan_linear_requirements:
            self.kan_linear_requirements = KANLinearRequirements()

        # Regular convs can work with any dimensions:
        # self.conv_requirements.filter_size_factor = _get_image_channels_num(self.color_mode)
        self.kan_conv_requirements.filter_size_factor = _get_image_channels_num(self.color_mode)

        if self.epochs < 1:
            raise ValueError(f'{self.epochs} is unacceptable number of train epochs.')
        if not all([side_size >= 3 for side_size in self.input_data_shape]):
            raise ValueError(f'Specified image size {self.input_data_shape} is unacceptable.')
        if not self.channels_num:
            raise ValueError(f'{self.color_mode} if unacceptable')
        if not self.primary:
            self.primary = [LayersPoolEnum.conv2d]
        if not self.secondary:
            self.secondary = [LayersPoolEnum.dropout, LayersPoolEnum.linear,
                              LayersPoolEnum.pooling2d, LayersPoolEnum.adaptive_pool2d]

    @property
    def channels_num(self) -> int:
        color_mode = str.lower(self.color_mode)
        return _get_image_channels_num(color_mode)

    @property
    def input_shape(self) -> List[Union[int, int], Union[int, int], int]:
        return [*self.input_data_shape, self.channels_num]

    @property
    def has_skip_connection(self) -> bool:
        return self._has_skip_connection

    @property
    def max_depth(self):
        return self.max_nn_depth + self.max_num_of_conv_layers

    @property
    def min_depth(self):
        return self.min_nn_depth + self.min_num_of_conv_layers

    def linear_is_kan(self):
        if LayersPoolEnum.kan_linear in self.secondary:
            return True
        elif LayersPoolEnum.linear in self.secondary:
            return False
        raise ValueError('No linear layer in the secondary list')

    def conv_is_kan(self):
        if LayersPoolEnum.kan_conv2d in self.primary:
            return True
        elif LayersPoolEnum.conv2d in self.primary:
            return False
        raise ValueError('No linear layer in the primary list')


@dataclass
class NNComposerRequirements(PipelineComposerRequirements):
    model_requirements: ModelRequirements = None
    opt_epochs: int = 5
    split_ratio: float = .8

    optimization_fitting_timeout_seconds: Optional[int] = 60 * 12

    def __post_init__(self):
        self.primary = self.model_requirements.primary
        self.secondary = self.model_requirements.secondary
        self.max_depth = self.model_requirements.max_depth
        self.mutation_strength = MutationStrengthEnum.strong
        if not 0 < self.split_ratio < 1:
            raise ValueError(f'{self.split_ratio} is unacceptable.')
        if self.opt_epochs < 1:
            raise ValueError(f'{self.opt_epochs} is unacceptable number of optimization epochs.')

    def set_model_requirements(self, model_requirements: ModelRequirements) -> NNComposerRequirements:
        self.model_requirements = model_requirements
        return self
