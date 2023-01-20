from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from nas.graph.cnn.cnn_graph import NasGraph
    from nas.graph.node.nn_graph_node import NNNode


class Struct:
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
