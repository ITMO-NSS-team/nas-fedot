from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Optional

from fedot.core.optimisers.gp_comp.operators.mutation import MutationStrengthEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements

from nas.repository.layer_types_enum import LayersPoolEnum, ActivationTypesIdsEnum

_possible_color_modes = {'RGB': 3, 'Gray': 1}


def permissible_kernel_parameters_correct(image_size: List[float], kernel_sizes: List[List[int]],
                                          strides: List[List[int]],
                                          pooling: bool):
    # TODO _update parameters checker
    for i, kernel_size in enumerate(kernel_sizes):
        for j, stride in enumerate(strides):
            is_strides_permissible = all([stride[i] < kernel_size[i] for i in range(len(stride))])
            is_kernel_size_permissible = all([kernel_size[i] < image_size[i] for i in range(len(stride))])

            if not is_strides_permissible:
                if pooling:
                    strides[j] = [2, 2]
                else:
                    strides[j] = [1, 1]
            if not is_kernel_size_permissible:
                kernel_sizes[i] = [2, 2]


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
    input_shape: Optional[List[float]] = None
    color_mode: Optional[str] = None

    cnn_secondary: List[LayersPoolEnum] = None  # Additional node type that can be placed in conv part of the graph
    dilation_rate: List[int] = None
    min_filters: int = 32
    max_filters: int = 128
    conv_strides: List[List[int]] = None
    pool_size: List[List[int]] = None
    pool_strides: List[List[int]] = None
    conv_layer: Optional[List[str]] = None

    # pool_types: Optional[List[str]] = None

    def __post_init__(self):
        if not self.dilation_rate:
            self.dilation_rate = [2]
        if not self.color_mode:
            self.color_mode = 'RGB'
        if not self.input_shape:
            self.input_shape = [64, 64]
        if self.color_mode:
            self.input_shape.append(3) if self.color_mode == 'RGB' else self.input_shape.append(1)
        if self.min_filters < 2:
            raise ValueError(f'min_filters value {self.min_filters} is unacceptable')

        if self.max_filters < 2:
            raise ValueError(f'max_filters value {self.max_filters} is unacceptable')
        if not self.conv_strides:
            self.conv_strides = [[1, 1]]
        if not all([side_size >= 3 for side_size in self.input_shape]):
            raise ValueError(f'Specified image size is unacceptable')
        # TODO Extend checker method. Current one doesn't fit it's name.
        # permissible_kernel_parameters_correct(self.input_shape, self.kernel_size, self.conv_strides, False)
        if self._has_pooling():
            permissible_kernel_parameters_correct(self.input_shape, self.pool_size, self.pool_strides, True)

    def _has_pooling(self) -> bool:
        has_pooling = LayersPoolEnum.max_pool2d or LayersPoolEnum.average_poold2 in self.cnn_secondary
        has_pool_params = self.pool_size and self.pool_strides is not None
        return has_pooling and has_pool_params

    @property
    def num_of_channels(self):
        return _possible_color_modes.get(self.color_mode)

    @property
    def neurons(self):
        neurons = [self.min_filters]
        i = self.min_filters
        while i < self.max_filters:
            i = i * 2
            neurons.append(i)
        return neurons


@dataclass
class NNRequirements:
    fc_requirements: FullyConnectedRequirements = FullyConnectedRequirements()
    conv_requirements: ConvRequirements = ConvRequirements()

    primary: Optional[List[LayersPoolEnum]] = None
    secondary: Optional[List[LayersPoolEnum]] = None
    activation_types: List[ActivationTypesIdsEnum] = None
    batch_norm_prob: Optional[float] = None
    dropout_prob: Optional[float] = None
    has_skip_connection: Optional[bool] = False

    max_possible_parameters: int = 8e7

    epochs: int = 1
    batch_size: int = 12

    max_num_of_conv_layers: int = 6
    min_num_of_conv_layers: int = 4

    max_nn_depth: int = 3
    min_nn_depth: int = 1

    max_drop_size: int = 0.5

    def __post_init__(self):
        if not self.activation_types:
            self.activation_types = [activation_func for activation_func in ActivationTypesIdsEnum]
        if not self.max_num_of_conv_layers:
            self.max_num_of_conv_layers = 4
        if self.max_drop_size > 1:
            self.max_drop_size = 1
        if not self.batch_size:
            self.batch_size = 16
        if not self.batch_norm_prob:
            self.batch_norm_prob = 0.5
        if not self.dropout_prob:
            self.dropout_prob = 0.5
        if self.epochs < 1:
            raise ValueError('Epoch number must be at least 1 or greater')
        if self.max_drop_size >= 1:
            raise ValueError(f'max_drop_size value {self.max_drop_size} is unacceptable')

    @property
    def max_possible_depth(self):
        return self.max_nn_depth + self.max_num_of_conv_layers

    def set_output_shape(self, output_shape: int):
        # TODO add output shape check
        self.conv_requirements.max_filters = output_shape
        self.conv_requirements.min_filters = output_shape
        return self

    def set_conv_params(self, stride: int):
        self.conv_requirements.conv_strides = [[stride, stride]]
        return self

    def set_pooling_params(self, stride: int, size: int) -> NNRequirements:
        self.conv_requirements.pool_size = [[size, size]]
        self.conv_requirements.pool_strides = [[stride, stride]]
        return self

    def set_batch_norm_prob(self, prob: float):
        self.batch_norm_prob = prob
        return self


@dataclass
class OptimizerRequirements:
    opt_epochs: Optional[int] = 5

    def __post_init__(self):
        pass


@dataclass
class DataRequirements:
    n_classes: int = None
    split_params: Optional[dict] = None

    def __post_init__(self):
        if not self.split_params:
            self.split_params = {'cv_folds': 5}
        if self.n_classes and self.n_classes < 2:
            raise ValueError(f'number of classes {self.n_classes} is not acceptable')


@dataclass
class NNComposerRequirements(PipelineComposerRequirements):
    data_requirements: DataRequirements = DataRequirements()
    optimizer_requirements: OptimizerRequirements = OptimizerRequirements()
    nn_requirements: NNRequirements = NNRequirements()
    max_pipeline_fit_time: Optional[datetime.timedelta] = datetime.timedelta(hours=1)
    # pop_size: Optional[int] = 10
    num_of_gens: Optional[int] = 15
    default_parameters: Optional[dict] = None
    _max_depth = 0

    def __post_init__(self):
        if self.data_requirements.split_params.get('cv_folds'):
            self.cv_folds = self.data_requirements.split_params.get('cv_folds')
        self.primary = self.nn_requirements.primary
        self.secondary = self.nn_requirements.secondary
        self.max_depth = self.nn_requirements.max_possible_depth

        self.mutation_strength = MutationStrengthEnum.strong

    @property
    def max_depth(self):
        return self.nn_requirements.max_possible_depth

    @max_depth.setter
    def max_depth(self, value):
        self._max_depth = value

    @staticmethod
    def get_default_requirements(**specified_parameters):
        pass
