from __future__ import annotations

import datetime
from typing import Callable

from nas.composer.nn_composer_requirements import DataRequirements, ConvRequirements, FullyConnectedRequirements, \
    NNRequirements, OptimizerRequirements, NNComposerRequirements
from nas.graph.cnn.cnn_graph import NNGraph
from nas.graph.cnn.resnet_builder import ResNetGenerator
from nas.graph.node.nn_graph_node import NNNode
from nas.nn import ActivationTypesIdsEnum
from nas.repository.layer_types_enum import LayersPoolEnum


# import tensorflow


def print_all_childs(graph: NNGraph):
    for node in graph.graph_struct:
        print(f'for node {node} childs are: {graph.node_children(node)}')


def placeholder(*args):
    """func for convert NNNode to keras layer"""
    print('placeholder')
    return args


class ListNode:
    def __init__(self, value: NNNode):
        self._value = value

    @property
    def value(self) -> Callable:
        return placeholder(self._value)

    def __repr__(self):
        return self._value.content['name']


class Struct:
    def __init__(self, graph: NNGraph):
        self.head = graph.graph_struct[0]
        self.graph = graph
        self.skip_connections_list = None

    def __len__(self):
        return len(self.graph.nodes)

    def __getitem__(self, item):
        if item > 0:
            node = self.graph.graph_struct[item]
            node = self.graph.node_children(node)
        else:
            node = [self.graph.graph_struct[item]]

        return node


if __name__ == '__main__':

    secondary_nodes = []
    for i in range(len(s)):
        nodes = s[i]
        if nodes[0] in secondary_nodes:
            print(2)
        secondary_nodes.extend(nodes[1::])

    print(1)
