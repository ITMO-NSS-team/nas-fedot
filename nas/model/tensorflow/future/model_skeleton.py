from typing import List, Union

from golem.core.dag.graph_node import GraphNode

from nas.graph.cnn_graph import NasGraph
from nas.graph.node.nas_graph_node import NasNode
from nas.model.tensorflow.future.tf_layer_initializer import LayerInitializer
from nas.model.utils.branch_manager import GraphBranchManager


class ModelSkeleton:
    def __init__(self, graph: NasGraph, layer_initializer: LayerInitializer = LayerInitializer(),
                 branch_manager: GraphBranchManager = GraphBranchManager()):
        # self.model_struct = {node: layer_initializer.initialize_layer(node) for node in graph.graph_struct}
        self._graph = graph
        self._layer_initializer = layer_initializer
        self._branch_manger = branch_manager

    @property
    def model_struct(self) -> dict:
        return {node: self._layer_initializer.initialize_layer(node) for node in self._graph.graph_struct}

    @property
    def model_layers(self) -> List:
        return list(self.model_struct.values())

    @property
    def model_nodes(self) -> List:
        return list(self.model_struct.keys())

    @property
    def branch_manager(self):
        return self._branch_manger

    def get_children(self, node: Union[GraphNode, NasNode]):
        return self._graph.node_children(node)
