import math
from typing import List

import tensorflow
from fedot.core.utils import DEFAULT_PARAMS_STUB

from nas.graph.node.nn_graph_node import NNNode
from nas.model.branch_manager import GraphBranchManager
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.default_parameters import default_nodes_params


def _get_layer_params(current_node: NNNode) -> dict:
    if current_node.content['params'] == DEFAULT_PARAMS_STUB:
        layer_params = default_nodes_params[current_node.content['name']]
    else:
        layer_params = current_node.content.get('params')
    return layer_params


def with_skip_connections(layer_func):
    def _add_equalize_conv(list_of_layers: List, input_layer):
        for i in range(len(list_of_layers)):
            if list_of_layers[i].shape[-1] != input_layer.shape[-1]:
                out_shape = input_layer.shape[-1]
                list_of_layers[i] = KerasLayers().add_shortcut_conv(list_of_layers[i], out_shape)

    def wrapper(*args, **kwargs):
        branch_manager: GraphBranchManager = kwargs.get('../model/branch_manager.py')
        # node and it's layer representation
        current_node = kwargs.get('node')
        input_layer = layer_func(*args, **kwargs)

        # add to active branches new branches
        # branch_manager.py.add_and_update(current_node, input_layer)
        # branch_manager.py._add_branch(current_node, input_layer)

        if len(current_node.nodes_from) > 1:
            # for cases where len(current_node.nodes_from) > 1 add skip connection
            # also add dimension equalizer for cases which have different dimensions
            # layer_to_add = branch_manager.py.get_last_layer(current_node)
            layers_to_add = branch_manager.get_last_layer(current_node.nodes_from[1:])
            if len(layers_to_add) > 2:
                pass
            _add_equalize_conv(layers_to_add, input_layer)
            layers_to_add.append(input_layer)
            # dimensions check. add conv to equalize dimensions in shortcuts if different
            input_layer = tensorflow.keras.layers.Add()(layers_to_add)

        # _update active branches
        # branch_manager.py.update_branch(current_node, input_layer)
        return input_layer

    return wrapper


def with_activation(layer_func):
    def add_activation_to_layer(*args, **kwargs):
        layer_params = _get_layer_params(kwargs.get('node'))
        activation_type = layer_params.get('activation')
        input_layer = layer_func(*args, **kwargs)
        if activation_type:
            activation = tensorflow.keras.layers.Activation(activation_type)
            input_layer = activation(input_layer)
        return input_layer

    return add_activation_to_layer


def with_batch_norm(layer_func):
    def add_batch_norm_to_layer(*args, **kwargs):
        layer_params = _get_layer_params(kwargs.get('node'))
        momentum = layer_params.get('momentum')
        input_layer = layer_func(*args, **kwargs)
        if momentum:
            epsilon = layer_params.get('epsilon')
            batch_norm = tensorflow.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
            input_layer = batch_norm(input_layer)
        return input_layer

    return add_batch_norm_to_layer


def with_dropout(layer_func):
    def add_dropout_to_layer(*args, **kwargs):
        layer_params = _get_layer_params(kwargs.get('node'))
        drop = layer_params.get('drop')
        input_layer = layer_func(*args, **kwargs)
        if drop:
            dropout = tensorflow.keras.layers.Dropout(drop)
            input_layer = dropout(input_layer)
        return input_layer

    return add_dropout_to_layer


class KerasLayers:
    @staticmethod
    def downsample_block(input_layer, out_shape: int, current_node: NNNode):
        """Adds to skip connection's shortcut a conv 1x1 layer to fix dimension difference"""
        layer_params = _get_layer_params(current_node)
        # stride = layer_params.get('conv_strides')
        # if stride != 1:
        stride = math.ceil(out_shape[-1] / input_layer.shape[-1])
        layer_to_add = tensorflow.keras.layers.Conv2D(out_shape[-1], 1, stride, padding='valid')(input_layer)
        layer_to_add = tensorflow.keras.layers.BatchNormalization()(layer_to_add)
        return layer_to_add

    @staticmethod
    def conv2d(node: NNNode, input_layer: tensorflow.Tensor, *args, **kwargs):
        layer_params = _get_layer_params(node)

        kernel_size = layer_params['kernel_size']
        strides = layer_params['conv_strides']
        filters = layer_params['neurons']

        conv2d_layer = tensorflow.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                                      padding='same')
        return conv2d_layer(input_layer)

    @staticmethod
    def pool(node: NNNode, input_layer, *args, **kwargs):
        layer_params = _get_layer_params(current_node=node)
        pool_size = layer_params.get('pool_size', [2, 2])
        pool_strides = layer_params.get('pool_strides')
        # hotfix
        if node.content['name'] == LayersPoolEnum.max_pool2d.value:
            pool_layer = tensorflow.keras.layers.MaxPooling2D(pool_size, pool_strides, padding='same')(input_layer)
        else:
            pool_layer = tensorflow.keras.layers.AveragePooling2D(pool_size, pool_strides, padding='same')(input_layer)
        return pool_layer

    @staticmethod
    def flatten(node: NNNode, input_layer: tensorflow.Tensor, *args, **kwargs):
        return tensorflow.keras.layers.Flatten()(input_layer)

    @staticmethod
    def dense(node: NNNode, input_layer: tensorflow.Tensor, *args, **kwargs):
        layer_params = _get_layer_params(node)
        units = layer_params['neurons']
        dense_layer = tensorflow.keras.layers.Dense(units=units)
        return dense_layer(input_layer)

    @staticmethod
    def activation(node: NNNode, input_layer: tensorflow.Tensor, *args, **kwargs):
        layer_params = _get_layer_params(node)
        activation_type = layer_params.get('activation')
        if activation_type:
            activation = tensorflow.keras.layers.Activation(activation_type)
            input_layer = activation(input_layer)
        return input_layer

    @staticmethod
    def dropout(node: NNNode, input_layer: tensorflow.Tensor, *args, **kwargs):
        layer_params = _get_layer_params(node)
        drop = layer_params.get('drop')
        if drop:
            dropout = tensorflow.keras.layers.Dropout(drop)
            input_layer = dropout(input_layer)
        return input_layer

    @staticmethod
    def batch_norm(node: NNNode, input_layer: tensorflow.Tensor, *args, **kwargs):
        layer_params = _get_layer_params(node)
        momentum = layer_params.get('momentum')
        epsilon = layer_params.get('epsilon')
        if momentum:
            batch_norm = tensorflow.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
            input_layer = batch_norm(input_layer)
        return input_layer

    @staticmethod
    def _get_node_type(node: NNNode) -> str:
        node_type = node.content['name']
        if 'conv2d' in node_type:
            return 'conv2d'
        elif 'pool' in node_type:
            return 'pool'
        return node_type

    @classmethod
    def convert_by_node_type(cls, node: NNNode, input_layer: tensorflow.Tensor, branch_manager: GraphBranchManager, **kwargs):
        layer_types = {
            'conv2d': cls.conv2d,
            'dense': cls.dense,
            'flatten': cls.flatten,
            'pool': cls.pool
        }
        node_type = cls._get_node_type(node)

        return layer_types[node_type](node=node, input_layer=input_layer, branch_manager=branch_manager, **kwargs)
