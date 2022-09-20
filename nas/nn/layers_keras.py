from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Any

from fedot.core.utils import DEFAULT_PARAMS_STUB
from tensorflow.keras import layers

from nas.graph.node.nn_graph_node import NNNode
from nas.utils.default_parameters import default_nodes_params


class ActivationTypesIdsEnum(Enum):
    softmax = 'softmax'
    elu = 'elu'
    selu = 'selu'
    softplus = 'softplus'
    relu = 'relu'
    softsign = 'softsign'
    tanh = 'tanh'
    hard_sigmoid = 'hard_sigmoid'
    sigmoid = 'sigmoid'
    linear = 'linear'


@dataclass
class LayerParams:
    layer_type: str
    pool_type: str = None
    neurons: int = None
    activation: str = None
    drop: float = None
    pool_size: Tuple[int, int] = None
    kernel_size: Tuple[int, int] = None
    conv_strides: Tuple[int, int] = None
    pool_strides: Tuple[int, int] = None
    num_of_filters: int = None
    output_shape: List[float] = None
    momentum: float = None
    epsilon: float = None


def make_dense_layer(idx: int, input_layer: Any, current_node: NNNode):
    """
    This function generates dense layer from given node parameters

    :param idx: layer index
    :param input_layer: input Keras layer
    :param current_node: current node NNNode type
    """
    layer_params = _get_layer_params(current_node)
    neurons_num = layer_params.get('neurons')
    activation = layers.Activation(layer_params['activation'])
    dense_layer = layers.Dense(units=neurons_num, name=f'dense_layer_{idx}')(input_layer)
    if 'momentum' in layer_params:
        dense_layer = _add_batch_norm(dense_layer, layer_params)
    dense_layer = activation(dense_layer)
    if 'drop' in layer_params:
        dense_layer = _make_dropout_layer(dense_layer, layer_params)
    return dense_layer


def _get_layer_params(current_node: NNNode):
    if current_node.content['params'] == DEFAULT_PARAMS_STUB:
        layer_params = default_nodes_params[current_node.content['name']]
    else:
        layer_params = current_node.content.get('params')
    return layer_params


def _make_dropout_layer(input_layer: Any, params):
    """
    This function generates dropout layer from given node parameters

    :param input_layer: input Keras layer
    """
    drop = params.get('drop')
    dropout = layers.Dropout(drop)
    dropout_layer = dropout(input_layer)
    return dropout_layer


def make_conv_layer(idx: int, input_layer: Any, current_node: NNNode = None, is_free_node: bool = False):
    """
    This function generates convolutional layer from given node and adds pooling layer if node doesn't belong to any of
    skip connection blocks

    :param idx: layer index
    :param input_layer: input Keras layer
    :param current_node: current node NNNode type
    :param is_free_node: is node not belongs to any of the skip connection blocks
    """
    # Conv layer params
    layer_params = _get_layer_params(current_node)
    kernel_size = layer_params['kernel_size']
    conv_strides = layer_params['conv_strides'] if is_free_node else [1, 1]
    filters_num = layer_params['num_of_filters']
    dilation_rate = layer_params.get('dilation_rate', 1)
    activation = layers.Activation(layer_params['activation'])
    conv_layer = layers.Conv2D(filters=filters_num, kernel_size=kernel_size, strides=conv_strides,
                               name=f'conv_layer_{idx}', padding='same', dilation_rate=dilation_rate)(input_layer)
    if 'momentum' in layer_params:
        conv_layer = _add_batch_norm(input_layer=conv_layer, layer_params=layer_params)
    conv_layer = activation(conv_layer)
    # Add pooling
    if is_free_node:
        if layer_params.get('pool_size'):
            pool_size = layer_params['pool_size']
            pool_strides = layer_params['pool_strides']
            if layer_params['pool_type'] == 'max_pool2d':
                pooling = layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same')
            elif layer_params['pool_type'] == 'average_pool2d':
                pooling = layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, padding='same')
            else:
                raise ValueError('Wrong pooling type!')
            conv_layer = pooling(conv_layer)
    if 'drop' in layer_params:
        conv_layer = _make_dropout_layer(conv_layer, layer_params)

    return conv_layer


def make_skip_connection_block(idx: int, input_layer: Any, current_node, layers_dict: dict):
    """
    This function implements skip connection if current node has any.
    Returns concatenate of two layers as result in node has skip connections. Otherwise, returns current layer as result

    :param idx: layer index
    :param input_layer: input Keras layer
    :param current_node: current node
    :param layers_dict: dictionary with skip connection start/end pairs of nodes
    """
    if current_node in layers_dict:
        tmp = layers_dict.pop(current_node)
        start_layer = tmp.pop(0)
        input_layer = layers.concatenate([start_layer, input_layer],
                                         axis=-1, name=f'residual_end_{idx}')
        input_layer = layers.Activation('relu')(input_layer)
        layers_dict[current_node] = tmp
    return input_layer


def _add_batch_norm(input_layer: Any, layer_params):
    """
    Method that adds batch normalization layer if current node has batch_norm parameters

    :param input_layer: input Keras layer
    """
    batch_norm_layer = layers.BatchNormalization(momentum=layer_params['momentum'],
                                                 epsilon=layer_params['epsilon'])(input_layer)
    return batch_norm_layer


class KerasLayerBuilder:
    def __init__(self):
        self._builder_function = None

    def with_builder_function(self, layer_func):
        self._builder_function = layer_func
        return self

    def build(self, idx, input_layer, current_node, **kwargs):
        self._builder_function(idx, input_layer, current_node, **kwargs)
