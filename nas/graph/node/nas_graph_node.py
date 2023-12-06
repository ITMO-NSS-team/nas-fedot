from __future__ import annotations

from typing import Optional, List

import numpy as np
from golem.core.optimisers.graph import OptNode


def calculate_output_shape(node: NasNode) -> np.ndarray:
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


def count_conv_layer_params(node: NasNode):
    input_shape = node.input_shape
    input_filters = input_shape[-1]
    kernel_size = node.content['params'].get('kernel_size')
    neurons = node.content['params'].get('neurons')
    params = (np.dot(*kernel_size) * input_filters + 1) * neurons
    return params


def count_flatten_layer_params(node):
    return np.prod(node.input_shape)


def count_fc_layer_params(node):
    return np.prod(node.output_shape) + 1


class NasNode(OptNode):
    def __init__(self, content: dict, nodes_from: Optional[List] = None):
        super().__init__(content, nodes_from)
        self.nodes_from = nodes_from
        self._input_shape = None
        if 'params' in content:
            self.content = content

    def __str__(self):
        return str(self.content['name'])

    def __repr__(self):
        return self.__str__()
