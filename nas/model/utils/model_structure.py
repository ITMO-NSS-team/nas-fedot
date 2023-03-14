from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, List, Tuple, Optional

import numpy as np
import tensorflow.python.keras.metrics
from golem.core.dag.convert import graph_structure_as_nx_graph
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


@dataclass
class _ModelStructure:
    # def __init__(self, graph: NasGraph):
    #     self.current_node_id = 0
    #     self.max_node_id = len(graph.nodes)
    #     self._nodes_matrix = np.zeros((len(graph.nodes), len(graph.nodes)), dtype=int)
    #     self._graph = graph
    #     self.residual_connections = None
    #     self.initialize_matrix()
    graph: NasGraph
    graph_adjacency = None
    nx_struct = None
    current_node_id: int = None
    max_node_id: int = None
    nodes_matrix: np.ndarray = None
    nodes_hierarchy = None

    def __post_init__(self):
        nx_graph, self.nx_struct = graph_structure_as_nx_graph(graph)
        self.graph_adjacency = [(n, nbrdict) for n, nbrdict in nx_graph.adjacency()]
        # self.graph_adjacency = {hash(self.nx_struct[node]): map(hash, [self.nx_struct[node_uuid] for node_uuid in successors]) for node, successors in nx_graph.adjacency()}
        self.nodes_matrix = np.zeros((len(graph.nodes), len(graph.nodes)), dtype=int)
        self.initialize_matrix()
        self.reset_current_node_id()

    def __len__(self):
        return len(self.nodes_matrix)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple:
        # Returns current node id and indices of its children.
        if self.current_node_id < self.max_node_id:
            val = self.nodes_matrix[:, self.current_node_id]
            self.current_node_id += 1
            return self.current_node_id, self.get_indices_from_vector(val)
            # mb return non-zero indices instead of vector of numbers.
        else:
            self.current_node_id = 0
            raise StopIteration

    def initialize_matrix(self):
        _nodes_id = {node: node_id for node_id, node in enumerate(self.graph.nodes)}
        for node_id, node in enumerate(self.graph.nodes):
            children_nodes = self.graph.node_children(node)
            if not children_nodes:
                continue
            for children in children_nodes:
                self.nodes_matrix[_nodes_id[children]][node_id] = 1

    @staticmethod
    def get_indices_from_vector(vector_num: np.ndarray):
        return np.nonzero(vector_num)[0]

    def reset_current_node_id(self):
        self.current_node_id = np.where(~self.nodes_matrix.any(axis=1))[0][0]
        return self


if __name__ == '__main__':
    from nas.graph.cnn_graph import NasGraph
    from nas.model.tensorflow.tf_model import BaseNasTFModel, NasTFModel
    from nas.graph.graph_builder.base_graph_builder import BaseGraphBuilder
    from nas.graph.graph_builder.resnet_builder import ResNetGenerator
    from nas.composer.nn_composer_requirements import load_default_requirements

    # graph = NasGraph.load('/home/staeros/work/nas_graph/skip_connection_parallel/graph.json')
    builder = BaseGraphBuilder().set_builder(ResNetGenerator(model_requirements=load_default_requirements().model_requirements))
    graph = builder.build()
    hierarchy = ordered_subnodes_hierarchy(graph.root_node)
    struct = _ModelStructure(graph)

    # for n in struct:
    #     pass

    model = NasTFModel(model=BaseNasTFModel(struct, n_classes=75))

    model.compile_model(metrics=[tensorflow.keras.metrics.Accuracy()],
                        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),
                        loss='categorical_crossentropy')

    # input_ = tensorflow.keras.layers.Input(shape=(32, 32, 3))

    model.model.build((None, 224, 224, 3))

    print(1)
