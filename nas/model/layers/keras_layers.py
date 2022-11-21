from __future__ import annotations
import math
from enum import Enum
from typing import List, TYPE_CHECKING

import tensorflow
from fedot.core.utils import DEFAULT_PARAMS_STUB
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.default_parameters import default_nodes_params

if TYPE_CHECKING:
    from nas.graph.node.nn_graph_node import NNNode
    from nas.model.branch_manager import GraphBranchManager


def _get_layer_params(current_node: NNNode) -> dict:
    if current_node.content['params'] == DEFAULT_PARAMS_STUB:
        layer_params = default_nodes_params[current_node.content['name']]
    else:
        layer_params = current_node.content.get('params')
    return layer_params


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
    def convert_by_node_type(cls, node: NNNode, input_layer: tensorflow.Tensor, branch_manager: GraphBranchManager,
                             **kwargs):
        layer_types = {
            'conv2d': cls.conv2d,
            'dense': cls.dense,
            'flatten': cls.flatten,
            'pool': cls.pool
        }
        node_type = cls._get_node_type(node)

        return layer_types[node_type](node=node, input_layer=input_layer, branch_manager=branch_manager, **kwargs)
