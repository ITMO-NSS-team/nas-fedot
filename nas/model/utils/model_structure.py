from __future__ import annotations

from typing import TYPE_CHECKING, Union, List, Tuple, Optional

import numpy as np
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy

if TYPE_CHECKING:
    from nas.graph.cnn_graph import NasGraph
    from nas.graph.node.nas_graph_node import NasNode


class ModelStructure:
    """Support class for model building."""

    def __init__(self, graph: NasGraph):
        self.head = graph.graph_struct[0]
        self.graph = graph
        self._iterator = 0
        self.skip_connections_list = None

    def __len__(self):
        return len(self.graph.nodes)

    def __getitem__(self, item) -> Union[List, Tuple, GraphNode]:
        """returns all children nodes of node by node id."""
        return self.graph.graph_struct[item]

    def get_children(self, node: NasNode) -> Optional[List, Tuple, GraphNode]:
        """
        Returns all children by given node.
        :param node: node which children should get
        :return: list of all children nodes.
        """
        return self.graph.node_children(node)

    def reset(self):
        self._iterator = 0


class _ModelStructure:
    def __init__(self, graph: NasGraph):
        self.iterator = 0
        self.iterator_max_value = len(graph.nodes)
        self._nodes_matrix = np.zeros((len(graph.nodes), len(graph.nodes)), dtype=int)
        self._graph = graph
        self.init_matrix()

    @property
    def graph(self):
        return self._graph

    @property
    def nodes_matrix(self):
        return self._nodes_matrix

    def __len__(self):
        return len(self.nodes_matrix)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator < self.iterator_max_value:
            val = self.nodes_matrix[:, self.iterator]
            self.iterator += 1
            return self.get_indices_from_vector(val)
            # mb return non-zero indices instead of vector of numbers.
        else:
            self.iterator = 0
            raise StopIteration

    def init_matrix(self):
        _nodes_id = {node: node_id for node_id, node in enumerate(self.graph.nodes)}
        for node_id, node in enumerate(self.graph.nodes):
            children_nodes = self.graph.node_children(node)
            if not children_nodes:
                continue
            for children in children_nodes:
                self.nodes_matrix[_nodes_id[children]][node_id] = 1

    @staticmethod
    def get_indices_from_vector(vector_num: np.ndarray):
        return np.nonzero(vector_num)


if __name__ == '__main__':
    from nas.graph.cnn_graph import NasGraph

    graph = NasGraph.load('/home/staeros/work/nas_graph/skip_connection_parallel/graph.json')
    hierarchy = ordered_subnodes_hierarchy(graph.root_node)
    struct = _ModelStructure(graph)
    for n in struct:
        pass
    print(1)
