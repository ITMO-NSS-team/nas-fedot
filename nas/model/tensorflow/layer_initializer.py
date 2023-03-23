from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Union

import tensorflow as tf
from golem.core.dag.graph_node import GraphNode

from nas.graph.node.nas_graph_node import NasNode


@dataclass
class LayerInitializer:
    def __init__(self):
        self._layers_dictionary = {'conv2d': self.conv2d,
                                   'dense': self.dense,
                                   'dropout': self.dropout,
                                   'batch_norm': self.batch_norm,
                                   'max_pool2d': partial(self.pooling, mode='max'),
                                   'average_pool2d': partial(self.pooling, mode='avg'),
                                   'flatten': self.flatten}

    @staticmethod
    def conv2d(node: NasNode):
        padding = 'same'  # 'valid'  # TODO
        kernel_size = node.content['params']['kernel_size']  # TODO
        filters = node.content['params']['neurons']
        strides = node.content['params']['conv_strides']
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

    @staticmethod
    def dense(node: NasNode):
        filters = node.content['params']['neurons']
        return tf.keras.layers.Dense(filters)

    @staticmethod
    def pooling(node: NasNode, mode: str):
        pool_size = node.content['params']['pool_size']
        pool_strides = node.content['params']['pool_strides']
        if mode == 'max':
            return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides)
        elif mode == 'avg':
            return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides)
        else:
            raise ValueError(f'{mode} is wrong pooling mode.')

    @staticmethod
    def flatten(node: NasNode):
        return tf.keras.layers.Flatten()

    @staticmethod
    def dropout(node: NasNode):
        dropout_value = node.content['params']['drop']
        return tf.keras.layers.Dropout(dropout_value)

    @staticmethod
    def batch_norm(node: NasNode):
        momentum = node.content['params']['momentum']
        epsilon = node.content['params']['epsilon']
        return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    @staticmethod
    def activation(node: NasNode):
        activation_function_name = node.content['params'].get('activation')
        return tf.keras.layers.Activation(activation=activation_function_name)

    def initialize_layer(self, node: Union[NasNode, GraphNode]):
        node_layers = []
        name = node.content['name']
        if 'conv' in name:
            name = 'conv2d'
        node_layers.append(self._layers_dictionary[name](node))
        if 'epsilon' in node.parameters.keys():
            node_layers.append(self.batch_norm(node))

        return node_layers