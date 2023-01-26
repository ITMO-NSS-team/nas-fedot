from random import choice
from typing import (Optional, List)

from golem.core.optimisers.opt_node_factory import OptNodeFactory

from nas.composer.nn_composer_requirements import ModelRequirements
from nas.graph.node.nn_graph_node import NNNode, get_node_params_by_type
from nas.repository.layer_types_enum import LayersPoolEnum


class NNNodeFactory(OptNodeFactory):
    def __init__(self, requirements: ModelRequirements, advisor):
        self.requirements = requirements
        self.advisor = advisor
        self._pool_conv_nodes = self.requirements.primary
        self._pool_fc_nodes = self.requirements.secondary

    def _get_possible_candidates(self, node: NNNode) -> List[LayersPoolEnum]:
        if 'conv' in node.content['name']:
            return self._pool_conv_nodes
        else:
            return self._pool_fc_nodes

    def exchange_node(self, node: NNNode) -> Optional[NNNode]:
        candidates = self._get_possible_candidates(node)
        candidates = self.advisor.propose_change(node=node,
                                                 possible_operations=candidates)

        return self._return_node(candidates)

    def get_parent_node(self, node: NNNode, **kwargs) -> Optional[NNNode]:
        parent_operations_ids = None
        possible_operations = self._get_possible_candidates(node)
        if node.nodes_from:
            parent_operations_ids = [str(n.content['name']) for n in node.nodes_from]

        candidates = self.advisor.propose_parent(node=node,
                                                 possible_operations=possible_operations)
        return self._return_node(candidates)

    def get_child_node(self, node: NNNode, **kwargs) -> Optional[NNNode]:
        possible_operations = self._get_possible_candidates(node)
        candidates = self.advisor.propose_child(node=node, possible_operations=possible_operations)
        return self._return_node(candidates)

    def get_node(self, is_primary: bool) -> Optional[NNNode]:
        candidates = self._pool_conv_nodes if is_primary else self._pool_fc_nodes
        return self._return_node(candidates)

    def _return_node(self, candidates):
        if not candidates:
            return None
        layer_name = choice(candidates)
        layer_params = get_node_params_by_type(layer_name, self.requirements)
        return NNNode(content={'name': layer_name.value,
                               'params': layer_params})
