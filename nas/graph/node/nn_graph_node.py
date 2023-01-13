from __future__ import annotations
from typing import Optional, List, Tuple, Union

import numpy as np
from fedot.core.optimisers.graph import OptNode

from nas.graph.node.nn_node_params import GraphLayers


def get_node_params_by_type(node, requirements):
    return GraphLayers().layer_by_type(node, requirements)


def calculate_output_shape(node: NNNode) -> np.ndarray:
    """Returns input_layer shape of node"""
    # define node type
    is_conv = 'conv' in node.content['name']
    is_flatten = 'flatten' in node.content['name']
    if is_conv:
        return count_conv_layer_params(node)
    if is_flatten:
        return count_flatten_layer_params(node)
    else:
        return count_fc_layer_params(node)


def count_conv_layer_params(node: NNNode):
    input_shape = node.input_shape
    input_filters = input_shape[-1]
    kernel_size = node.content['params'].get('kernel_size')
    neurons = node.content['params'].get('neurons')
    params = (np.dot(*kernel_size) * input_filters + 1) * neurons
    return params


def count_flatten_layer_params(node):
    return np.prod(node.source_shape)


def count_fc_layer_params(node):
    return np.prod(node.output_shape) + 1


class NNNode(OptNode):
    def __init__(self, content: dict, nodes_from: Optional[List] = None):
        super().__init__(content, nodes_from)
        self.nodes_from = nodes_from

    def __str__(self):
        return str(self.content['name'])

    def __repr__(self):
        return self.__str__()

    @property
    def input_channels(self):
        if not self.nodes_from:
            return self.content['params'].get('input_channels')
        else:
            return self.nodes_from[0].output_channels

    @property
    def output_channels(self):
        return self.content['params']['out_channels']

    @property
    def node_params(self):
        return calculate_output_shape(self)
