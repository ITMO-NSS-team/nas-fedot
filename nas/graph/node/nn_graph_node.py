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
    return np.prod(node.input_shape)


def count_fc_layer_params(node):
    return np.prod(node.output_shape) + 1


class NNNode(OptNode):
    def __init__(self, content: dict, nodes_from: Optional[List] = None,
                 input_shape: Union[List[float], Tuple[float]] = None):
        super().__init__(content, nodes_from)
        self.nodes_from = nodes_from
        self._input_shape = None
        if 'params' in content:
            self.content = content
            # self.content['name'] = self.content['params']['name']

    def __str__(self):
        return str(self.content['name'])

    def __repr__(self):
        return self.__str__()

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, val):
        self._input_shape = val

    @property
    def output_shape(self) -> List:
        is_conv = 'conv' in self.content['name']
        is_flatten = 'flatten' in self.content['name']
        if is_conv:
            return [*self.content['params'].get('kernel_size'), self.content['params'].get('neurons')]
        if is_flatten:
            parent_node = self.nodes_from[0]
            return [np.prod(parent_node.output_shape).tolist()]
        else:
            return [*self.input_shape, self.content['params'].get('neurons')]

    @property
    def node_params(self):
        return calculate_output_shape(self)
