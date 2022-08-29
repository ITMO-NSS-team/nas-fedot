import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements, \
    MutationStrengthEnum

from nas.model.nn.layers_keras import activation_types

_possible_color_modes = {'RGB': 3, 'Gray': 1}


def permissible_kernel_parameters_correct(image_size: List[float], kernel_size: List[int],
                                          strides: List[int],
                                          pooling: bool) -> Tuple[List[int], List[int]]:
    is_strides_permissible = all([strides[i] < kernel_size[i] for i in range(len(strides))])
    is_kernel_size_permissible = all([kernel_size[i] < image_size[i] for i in range(len(strides))])
    if not is_strides_permissible:
        if pooling:
            strides = [2, 2]
        else:
            strides = [1, 1]
    if not is_kernel_size_permissible:
        kernel_size = [2, 2]
    return kernel_size, strides


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
    def neurons_num(self):
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

    cnn_secondary: List[str] = None

    min_filters: int = 32
    max_filters: int = 128
    kernel_size: List[int] = None
    conv_strides: List[int] = None
    pool_size: List[int] = None
    pool_strides: List[int] = None
    conv_layer: List[str] = None
    pool_types: List[str] = None

    def __post_init__(self):
        if not self.input_shape:
            self.input_shape = [64, 64, 3]
        if self.min_filters < 2:
            raise ValueError(f'min_filters value {self.min_filters} is unacceptable')
        if self.max_filters < 2:
            raise ValueError(f'max_filters value {self.max_filters} is unacceptable')
        if not self.kernel_size:
            self.kernel_size = [3, 3]
        if not self.conv_strides:
            self.conv_strides = [1, 1]
        if not self.pool_size:
            self.pool_size = [2, 2]
        if not self.pool_strides:
            self.pool_strides = [2, 2]
        if not self.pool_types:
            self.pool_types = ['max_pool2d', 'average_pool2d']
        if not all([side_size >= 3 for side_size in self.input_shape]):
            raise ValueError(f'Specified image size is unacceptable')

        self.kernel_size, self.conv_strides = permissible_kernel_parameters_correct(self.input_shape,
                                                                                    self.kernel_size,
                                                                                    self.conv_strides, False)
        self.pool_size, self.pool_strides = permissible_kernel_parameters_correct(self.input_shape,
                                                                                  self.pool_size,
                                                                                  self.pool_strides, True)

    @property
    def num_of_channels(self):
        return _possible_color_modes.get(self.color_mode)

    @property
    def filters(self):
        filters = [self.min_filters]
        i = self.min_filters
        while i < self.max_filters:
            i = i * 2
            filters.append(i)
        return filters


@dataclass
class NNRequirements:
    fc_requirements: FullyConnectedRequirements = FullyConnectedRequirements()
    conv_requirements: ConvRequirements = ConvRequirements()

    primary: Optional[List[str]] = None
    secondary: Optional[List[str]] = None

    activation_types = activation_types
    epochs: int = 1
    batch_size: int = 12

    max_num_of_conv_layers: int = 6
    min_num_of_conv_layers: int = 4

    max_nn_depth: int = 3
    min_nn_depth: int = 1

    max_drop_size: int = 0.5
    batch_norm_prob: Optional[float] = None
    dropout_prob: Optional[float] = None

    has_skip_connection: Optional[bool] = False

    def __post_init__(self):
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
        if not self.primary:
            self.primary = ['conv2d']
        if not self.secondary:
            self.secondary = ['dense']

    @property
    def max_possible_depth(self):
        return self.max_nn_depth + self.max_num_of_conv_layers


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
    pop_size: Optional[int] = 10
    num_of_gens: Optional[int] = 15
    default_parameters: Optional[dict] = None

    def __post_init__(self):
        if self.data_requirements.split_params.get('cv_folds'):
            self.cv_folds = self.data_requirements.split_params.get('cv_folds')
        self.primary = self.nn_requirements.primary
        self.secondary = self.nn_requirements.secondary
        self.max_depth = self.nn_requirements.max_possible_depth

        self.mutation_strength = MutationStrengthEnum.strong
