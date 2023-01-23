from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from nas.graph.node.nn_graph_node import NNNode


@dataclass
class LayerInitializer:
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
