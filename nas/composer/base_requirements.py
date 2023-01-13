from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Optional

from fedot.core.optimisers.gp_comp.operators.mutation import MutationStrengthEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements

from nas.repository.layer_types_enum import LayersPoolEnum, ActivationTypesIdsEnum


@dataclass
class FullyConnectedRequirements:
    """TODO"""
    min_number_of_neurons: int = 32
    max_number_of_neurons: int = 256

    @property
    def out_features(self) -> List[int]:
        neurons = [self.min_number_of_neurons]
        i = self.min_number_of_neurons
        while i < self.max_number_of_neurons:
            i *= 2
            neurons.append(i)
        return neurons

    def __post_init__(self):
        if self.min_number_of_neurons < 2:
            raise ValueError(f'min_num_of_neurons value {self.min_number_of_neurons} is unacceptable')
        if self.max_number_of_neurons < 2:
            raise ValueError(f'max_num_of_neurons value {self.max_number_of_neurons} is unacceptable')


@dataclass
class ConvRequirements:
    source_shape: Optional[List[float]]
    number_of_channels: Optional[int] = None

    cnn_primary: list[LayersPoolEnum] = None  # List of main nodes like conv nodes with different kernel size
    cnn_secondary: List[LayersPoolEnum] = None  # Additional node type that can be placed in conv part of the graph
    dilation_rate: List[int] = None
    min_out_channels: int = 32
    max_out_channels: int = 128
    conv_strides: List[List[int]] = None
    pool_size: List[List[int]] = None
    pool_strides: List[List[int]] = None
    conv_layer: Optional[List[str]] = None

    def __post_init__(self):
        if not self.cnn_primary:
            self.cnn_primary = [LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_5x5,
                                LayersPoolEnum.conv2d_7x7]
        if not self.conv_strides:
            self.conv_strides = [[1, 1]]
        if not self.pool_size:
            self.pool_size = [[2, 2]]
        if not self.pool_strides:
            self.pool_strides = self.pool_size
        if not self.number_of_channels:
            self.number_of_channels = 3
        if self.number_of_channels:
            if len(self.source_shape) < 3:
                self.source_shape.append(self.number_of_channels)
        if self.min_out_channels < 2:
            raise ValueError(f'min_filters value {self.min_out_channels} is unacceptable')
        if self.max_out_channels < 2:
            raise ValueError(f'max_filters value {self.max_out_channels} is unacceptable')
        if not all([side_size >= 3 for side_size in self.source_shape]):
            raise ValueError(f'Specified image size: {self.source_shape} is unacceptable')
        if self.unacceptable_image_size():
            raise ValueError(f'Image size {self.source_shape} is unacceptable')

    def unacceptable_image_size(self) -> bool:
        for stride in self.conv_strides:
            for side in self.source_shape[:-1]:
                if any([s > side for s in stride]):
                    return True

    @property
    def out_channels(self):
        neurons = [self.min_out_channels]
        i = self.min_out_channels
        while i < self.max_out_channels:
            i = i * 2
            neurons.append(i)
        return neurons


@dataclass
class ModelRequirements:
    """TODO"""
    fc_requirements: FullyConnectedRequirements
    conv_requirements: ConvRequirements

    activation_types: List[ActivationTypesIdsEnum] = None

    batch_norm_prob: Optional[float] = .8
    dropout_prob: Optional[float] = .8
    has_skip_connection: Optional[bool] = False

    max_possible_parameters: int = 8e7

    epochs: int = 1  # train epochs
    batch_size: int = 32

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
        if self.epochs < 1:
            raise ValueError('Epoch number must be at least 1 or greater')
        if self.max_drop_size >= 1:
            raise ValueError(f'max_drop_size value {self.max_drop_size} is unacceptable')

    @property
    def max_possible_depth(self):
        return self.max_nn_depth + self.max_num_of_conv_layers

    def set_output_shape(self, output_shape: int):
        # TODO add output shape check
        self.conv_requirements.max_out_channels = output_shape
        self.conv_requirements.min_out_channels = output_shape
        return self

    def set_conv_params(self, stride: int):
        self.conv_requirements.conv_strides = [[stride, stride]]
        return self

    def set_pooling_params(self, stride: int, size: int) -> ModelRequirements:
        self.conv_requirements.pool_size = [[size, size]]
        self.conv_requirements.pool_strides = [[stride, stride]]
        return self

    def set_batch_norm_prob(self, prob: float):
        self.batch_norm_prob = prob
        return self


@dataclass
class DataParams:
    n_classes: int = None
    cv_folds: Optional[int] = None  # k_fold if not None

    def __post_init__(self):
        if self.n_classes and self.n_classes < 2:
            raise ValueError(f'number of classes {self.n_classes} is not acceptable')


@dataclass
class NNComposerRequirements(PipelineComposerRequirements):
    data_params: DataParams = None
    nn_requirements: ModelRequirements = None
    opt_epochs: int = 1
    _max_depth = 0

    def __post_init__(self):
        if not self.data_params:
            raise ValueError('Data parameters must be specified.')

        self.primary = self.nn_requirements.conv_requirements.cnn_primary  # main node types e.g. different convs
        self.secondary = self.nn_requirements.conv_requirements.cnn_secondary  # additional nodes like poolings, dense etc

        self.cv_folds = self.data_params.cv_folds
        self.max_depth = self.nn_requirements.max_possible_depth
        self.mutation_strength = MutationStrengthEnum.strong

    @property
    def max_depth(self):
        return self.nn_requirements.max_possible_depth

    @staticmethod
    def get_default_requirements(**specified_parameters):
        pass

    @max_depth.setter
    def max_depth(self, value):
        self._max_depth = value

    @staticmethod
    def default_params(multiclass: bool = True):
        """Returns predefined requirements presets."""
        secondary_nodes_list = [LayersPoolEnum.average_poold2, LayersPoolEnum.max_pool2d]
        conv_requirements = ConvRequirements(source_shape=[64, 64], cnn_secondary=secondary_nodes_list)
        fc_requirements = FullyConnectedRequirements()
        model_requirements = ModelRequirements(conv_requirements=conv_requirements, fc_requirements=fc_requirements)
        data_params = DataParams(n_classes=3 if multiclass else 2, cv_folds=None)
        return NNComposerRequirements(num_of_generations=2, data_params=data_params,
                                      nn_requirements=model_requirements)


if __name__ == '__main__':
    params = NNComposerRequirements.default_params()
    print('Done!')
