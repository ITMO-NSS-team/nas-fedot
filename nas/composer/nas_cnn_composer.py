import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements

from nas.nn.layers_keras import activation_types


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
class GPNNComposerRequirements(PipelineComposerRequirements):
    conv_kernel_size: List[int] = None
    conv_strides: List[int] = None
    pool_size: List[int] = None
    pool_strides: List[int] = None
    min_num_of_neurons: int = 32
    max_num_of_neurons: int = 256
    min_filters: int = 32
    max_filters: int = 128
    channels_num: int = 3
    max_drop_size: int = 0.5
    input_shape: List[int] = None
    conv_layer: List[str] = None
    cnn_secondary: List[str] = None
    pool_types: List[str] = None
    epochs: int = 5
    batch_size: int = 12
    num_of_classes: int = 10
    activation_types = activation_types
    max_num_of_conv_layers: int = 6
    min_num_of_conv_layers: int = 4
    max_nn_depth: int = 6
    min_nn_depth: int = 1
    max_number_of_skips: int = None
    has_skip_connection: bool = False
    skip_connections_id: Optional[List[int]] = None
    shortcuts_len: Optional[int] = None
    batch_norm_prob: Optional[float] = None
    dropout_prob: Optional[float] = None
    default_parameters: Optional[dict] = None

    def __post_init__(self):
        if not self.timeout:
            self.timeout = datetime.timedelta(hours=20)
        if not self.conv_kernel_size:
            self.conv_kernel_size = [3, 3]
        if not self.conv_strides:
            self.conv_strides = [1, 1]
        if not self.pool_size:
            self.pool_size = [2, 2]
        if not self.pool_strides:
            self.pool_strides = [2, 2]
        if not self.secondary:
            self.secondary = ['dense']
        if not self.pool_types:
            self.pool_types = ['max_pool2d', 'average_pool2d']
        if not self.primary:
            self.primary = ['conv2d']
        if self.max_drop_size > 1:
            self.max_drop_size = 1
        if not self.max_num_of_conv_layers:
            self.max_num_of_conv_layers = 4
        if not self.batch_size:
            self.batch_size = 16
        # if not self.skip_connections_id:
        #     self.skip_connections_id = [0, 4, 8]
        # if not self.shortcuts_len:
        #     self.shortcuts_len = 4
        if not self.batch_norm_prob:
            self.batch_norm_prob = 0.5
        if not self.dropout_prob:
            self.dropout_prob = 0.5
        if self.epochs < 1:
            raise ValueError('Epoch number must be at least 1 or greater')
        if not all([side_size >= 3 for side_size in self.input_shape]):
            raise ValueError(f'Specified image size is unacceptable')
        self.conv_kernel_size, self.conv_strides = permissible_kernel_parameters_correct(self.input_shape,
                                                                                         self.conv_kernel_size,
                                                                                         self.conv_strides, False)
        self.pool_size, self.pool_strides = permissible_kernel_parameters_correct(self.input_shape,
                                                                                  self.pool_size,
                                                                                  self.pool_strides, True)
        self.max_depth = self.max_nn_depth + self.max_num_of_conv_layers

        if self.min_num_of_neurons < 1:
            raise ValueError(f'min_num_of_neurons value is unacceptable')
        if self.max_num_of_neurons < 1:
            raise ValueError(f'max_num_of_neurons value is unacceptable')
        if self.max_drop_size > 1:
            raise ValueError(f'max_drop_size value is unacceptable')
        if self.channels_num > 3 or self.channels_num < 1:
            raise ValueError(f'channels_num value must be anywhere from 1 to 3')
        if self.epochs < 1:
            raise ValueError(f'epochs number less than 1')
        if self.batch_size < 1:
            raise ValueError(f'batch size less than 1')
        if self.min_filters < 2:
            raise ValueError(f'min_filters value is unacceptable')
        if self.max_filters < 2:
            raise ValueError(f'max_filters value is unacceptable')
        if not self.max_number_of_skips and self.has_skip_connection:
            self.max_number_of_skips = 3

    @property
    def filters(self):
        filters = [self.min_filters]
        i = self.min_filters
        while i < self.max_filters:
            i = i * 2
            filters.append(i)
        return filters
