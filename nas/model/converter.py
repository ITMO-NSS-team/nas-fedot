from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from nas.graph.cnn.cnn_graph import NNGraph
    from nas.graph.node.nn_graph_node import NNNode


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
        self._iterator = 0
        self.skip_connections_list = None

    def __len__(self):
        return len(self.graph.nodes)

    def __getitem__(self, item):
        """returns all children nodes of node by it's id"""
        return self.graph.graph_struct[item]
        # node = self.graph._graph_struct[item]
        # return self.get_children(node)
        # if item == 0:
        #     node = [self.head]
        # else:
        #     node = self.graph._graph_struct[item - 1]
        #     node = self.graph.node_children(node)
        # return node

    def get_children(self, node: NNNode):
        return self.graph.node_children(node)

    def reset(self):
        self._iterator = 0
