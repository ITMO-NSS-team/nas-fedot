from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from nas.graph.cnn.cnn_graph import NasGraph
    from nas.graph.node.nn_graph_node import NNNode


class GraphStruct:
    def __init__(self, graph: NasGraph):
        self.head = graph.graph_struct[0]
        self.graph = graph
        self._iterator = 0
        self.skip_connections_list = None

    def __len__(self):
        return len(self.graph.nodes)

    def __getitem__(self, item):
        """returns all children nodes of node by it's id"""
        return self.graph.graph_struct[item]

    def get_children(self, node: NNNode):
        return self.graph.node_children(node)

    def reset(self):
        self._iterator = 0


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
    def conv2d(node: NNNode):
        kernel_size = 3  # TODO
        padding = 'valid'  # TODO
        filters = node.content['params']['neurons']
        strides = node.content['params']['conv_strides']
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

    @staticmethod
    def dense(node: NNNode):
        filters = node.content['params']['neurons']
        return tf.keras.layers.Dense(filters)

    @staticmethod
    def dropout(node: NNNode):
        dropout_value = node.content['params']['drop']
        return tf.keras.layers.Dropout(dropout_value)

    @staticmethod
    def batch_norm(node: NNNode):
        momentum = node.content['params']['momentum']
        epsilon = node.content['params']['epsilon']
        return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    @staticmethod
    def pooling(node: NNNode, mode: str):
        pool_size = node.content['params']['pool_size']
        pool_strides = node.content['params']['pool_strides']
        if mode == 'max':
            return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides)
        elif mode == 'avg':
            return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides)
        else:
            raise ValueError(f'{mode} is wrong pooling mode.')

    @staticmethod
    def flatten(node: NNNode):
        return tf.keras.layers.Flatten()

    def initialize_layer(self, node: NNNode):
        name = node.content['name']
        if 'conv' in name:
            name = 'conv2d'
        return self._layers_dictionary[name](node)
