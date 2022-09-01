from typing import (Optional)
from random import choice

from fedot.core.optimisers.opt_node_factory import OptNodeFactory

from nas.graph.cnn.cnn_graph_node import NNNode
from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.graph.cnn.cnn_builder import get_layer_params


class NNNodeFactory(OptNodeFactory):
    def __init__(self, requirements: NNComposerRequirements, advisor):
        super().__init__(requirements, advisor)
        self._pool_conv_nodes = self.requirements.primary
        self._pool_fc_nodes = self.requirements.secondary

    def exchange_node(self, node: NNNode) -> Optional[NNNode]:
        if node.content['name'] in self._pool_conv_nodes:
            candidates = self._pool_conv_nodes
        else:
            candidates = self._pool_fc_nodes

        candidates = self.advisor.propose_change(current_operation_id=str(node.content['name']),
                                                 possible_operations=candidates)

        return self._return_node(candidates)

    def get_parent_node(self,
                        node: NNNode,
                        primary: bool) -> Optional[NNNode]:
        parent_operations_ids = None
        possible_operations = self._pool_conv_nodes if node.content['name'] in self._pool_conv_nodes \
            else self._pool_fc_nodes
        if node.nodes_from:
            parent_operations_ids = [str(n.content['name']) for n in node.nodes_from]

        candidates = self.advisor.propose_parent(current_operation_id=str(node.content['name']),
                                                 parent_operations_ids=parent_operations_ids,
                                                 possible_operations=possible_operations)
        return self._return_node(candidates)

    def get_node(self, primary: bool) -> Optional[NNNode]:
        candidates = self.requirements.primary if primary else self._pool_fc_nodes
        return self._return_node(candidates)

    def _return_node(self, candidates):
        if not candidates:
            return None
        layer_name = choice(candidates)
        layer_params = get_layer_params(layer_name, self.requirements)
        return NNNode(content={'name': layer_name,
                               'params': layer_params})