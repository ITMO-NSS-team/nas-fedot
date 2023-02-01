from __future__ import annotations

from typing import TYPE_CHECKING, Union, List, Tuple, Optional

from golem.core.dag.graph_node import GraphNode

from nas.model.utils.branch_manager import GraphBranchManager

if TYPE_CHECKING:
    from nas.graph.cnn_graph import NasGraph
    from nas.graph.node.nas_graph_node import NasNode


class ModelStructure:
    """Support class for model building."""
    def __init__(self, graph: NasGraph, branch_manager: Optional[GraphBranchManager] = None):
        self.head = graph.graph_struct[0]
        self.graph = graph
        self._iterator = 0
        self.skip_connections_list = None
        # TODO
        self.branch_manager = branch_manager

    def __len__(self):
        return len(self.graph.nodes)

    def __getitem__(self, item) -> Union[List, Tuple, GraphNode]:
        """returns all children nodes of node by node id."""
        return self.graph.graph_struct[item]

    def set_branch_manager(self, branch_manager: Optional[GraphBranchManager]):
        self.branch_manager = branch_manager
        return self

    def get_children(self, node: NasNode) -> Optional[List, Tuple, GraphNode]:
        """
        Returns all children by given node.
        :param node: node which children should get
        :return: list of all children nodes.
        """
        return self.graph.node_children(node)

    def reset(self):
        self._iterator = 0
