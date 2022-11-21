from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple, List, Any

from fedot.core.utils import DEFAULT_PARAMS_STUB
from tensorflow.keras import layers

from nas.graph.node.nn_graph_node import NNNode
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.default_parameters import default_nodes_params


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
    output_shape: List[float] = None
    momentum: float = None
    epsilon: float = None


def make_dense_layer(idx: int, input_layer: Any, current_node: NNNode):
    """
    This function generates dense layer from given node parameters

    :param idx: layer index
    :param input_layer: input_layer Keras layer
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

    :param input_layer: input_layer Keras layer
    """
    drop = params.get('drop')
    dropout = layers.Dropout(drop)
    dropout_layer = dropout(input_layer)
    return dropout_layer


def make_pooling_layer(idx: int, input_layer: Any, current_node: NNNode, is_free_node: bool):
    layer_params = _get_layer_params(current_node)
    pool_size = layer_params.get('pool_size', [2, 2])
    pool_strides = layer_params.get('pool_strides')
    # hotfix 
    if current_node.content['name'] == LayersPoolEnum.max_pool2d.value:
        pool_layer = layers.MaxPooling2D(pool_size, pool_strides)(input_layer)
    else:
        pool_layer = layers.AveragePooling2D(pool_size, pool_strides)(input_layer)
    return pool_layer


def make_conv_layer(idx: int, input_layer: Any, current_node: NNNode = None, is_free_node: bool = False):
    """
    This function generates convolutional layer from given node and adds pooling layer if node doesn't belong to any of
    skip connection blocks

    :param idx: layer index
    :param input_layer: input_layer Keras layer
    :param current_node: current node NNNode type
    :param is_free_node: is node not belongs to any of the skip connection blocks
    """
    # Conv layer params
    layer_params = _get_layer_params(current_node)
    kernel_size = layer_params['kernel_size']
    conv_strides = layer_params['conv_strides']
    filters_num = layer_params['neurons']
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


def _add_shortcut_conv(input_layer, out_shape: int):
    """Adds to skip connection's shortcut a conv 1x1 layer to fix dimension difference"""
    stride = math.ceil(out_shape/input_layer.shape[-1])
    layer_to_add = layers.Conv2D(out_shape, 1, stride, padding='valid')(input_layer)
    layer_to_add = layers.BatchNormalization()(layer_to_add)
    return layer_to_add


def make_skip_connection_block(idx: int, input_layer: Any, current_node, layers_dict: dict):
    """
    This function implements skip connection if current node has any.
    Returns concatenate of two layers as result in node has skip connections. Otherwise, returns current layer as result

    :param idx: layer index
    :param input_layer: input_layer Keras layer
    :param current_node: current node
    :param layers_dict: dictionary with skip connection start/end pairs of nodes
    """
    if current_node in layers_dict:
        tmp = layers_dict.pop(current_node)
        start_layer = tmp.pop(0)
        # TODO extend to different strides
        if not start_layer.shape[-1] == input_layer.shape[-1]:
            # if current_node.nodes_from[0].content['params']['conv_strides'] != [1, 1]:
            out_shape = input_layer.shape[-1]
            start_layer = _add_shortcut_conv(input_layer=start_layer, out_shape=out_shape)
        input_layer = layers.add([start_layer, input_layer])
        # else:# if start layer has stride 2 => create conv 1x1 layer in shortcut
        #     input_layer = layers.concatenate([start_layer, input_layer],
        #                                      axis=-1, name=f'residual_end_{idx}')
        input_layer = layers.Activation('relu')(input_layer)
        layers_dict[current_node] = tmp
    return input_layer


def _add_batch_norm(input_layer: Any, layer_params):
    """
    Method that adds batch normalization layer if current node has batch_norm parameters

    :param input_layer: input_layer Keras layer
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
