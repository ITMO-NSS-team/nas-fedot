import random
from typing import List, Optional

from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes

from nas.composer.requirements import ModelRequirements
from nas.graph.BaseGraph import NasGraph
from nas.graph.builder.base_graph_builder import GraphGenerator
from nas.graph.node.nas_graph_node import NasNode
from nas.graph.node.nas_node_params import NasNodeFactory
from nas.operations.validation_rules.cnn_val_rules import model_has_several_roots, \
    model_has_wrong_number_of_flatten_layers, model_has_no_conv_layers, \
    model_has_several_starts, model_has_dim_mismatch
from nas.repository.layer_types_enum import LayersPoolEnum

random.seed(1)


def _add_skip_connections(graph: NasGraph, params):
    skip_connections_id = params[0]
    shortcut_len = params[1]
    for current_node in skip_connections_id:
        is_first_conv = current_node <= graph.cnn_depth[0]
        is_second_conv = current_node + shortcut_len < graph.cnn_depth[0]
        if is_first_conv == is_second_conv and (current_node + shortcut_len) < len(graph.nodes):
            graph.nodes[current_node + shortcut_len].nodes_from.append(graph.nodes[current_node])
        else:
            print('Wrong connection. Connection dropped.')


class ConvGraphMaker(GraphGenerator):
    def __init__(self, requirements: ModelRequirements,
                 initial_struct: Optional[List] = None, max_generation_attempts: int = 100):
        self._initial_struct = initial_struct
        self._requirements = requirements
        self._rules = [model_has_several_starts, model_has_no_conv_layers, model_has_wrong_number_of_flatten_layers,
                       model_has_several_roots, has_no_cycle, has_no_self_cycled_nodes, model_has_dim_mismatch]
        self._generation_attempts = max_generation_attempts

    @property
    def initial_struct(self):
        return self._initial_struct

    @property
    def requirements(self):
        return self._requirements

    @staticmethod
    def _get_skip_connection_params(graph):
        """Method for skip connection parameters generation"""
        connections = set()
        skips_len = random.randint(2, len(graph.nodes) // 2)
        max_number_of_skips = len(graph.nodes) // 3
        for _ in range(max_number_of_skips):
            node_id = random.randint(0, len(graph.nodes))
            connections.add(node_id)
        return connections, skips_len

    def check_generated_graph(self, graph: NasGraph) -> bool:
        for rule in self._rules:
            try:
                rule(graph)
            except ValueError:
                return False
        return True

    def _generate_from_scratch(self):
        total_conv_nodes = random.randint(self.requirements.min_num_of_conv_layers,
                                          self.requirements.max_num_of_conv_layers)
        total_fc_nodes = random.randint(self.requirements.min_nn_depth,
                                        self.requirements.max_nn_depth)
        # hotfix
        zero_node = random.choice(self.requirements.primary)
        graph_nodes = [zero_node]
        for i in range(1, total_conv_nodes + total_fc_nodes):
            if i == 0:
                node = random.choice(self.requirements.primary)
            elif i < total_conv_nodes:
                node = random.choice(self.requirements.primary + self.requirements.secondary) \
                    if i != total_conv_nodes - 1 else LayersPoolEnum.flatten
            else:
                node = random.choice([LayersPoolEnum.dropout, LayersPoolEnum.linear])
            graph_nodes.append(node)
        return graph_nodes

    def _add_node(self, node_to_add: LayersPoolEnum, parent_node: NasNode, node_name=None):
        node_params = NasNodeFactory(self.requirements).get_node_params(
            node_to_add)  # get_node_params_by_type(node_to_add, self.requirements)
        node = NasNode(content={'name': node_to_add.value, 'params': node_params}, nodes_from=parent_node)
        return node

    def build(self) -> NasGraph:
        for _ in range(self._generation_attempts):
            graph = NasGraph()
            parent_node = None
            graph_nodes = self.initial_struct if self.initial_struct else self._generate_from_scratch()
            for node in graph_nodes:
                node = self._add_node(node, parent_node)
                parent_node = [node]
                graph.add_node(node)
            if self.requirements.has_skip_connection:
                _add_skip_connections(graph, self._get_skip_connection_params(graph))
            if self.check_generated_graph(graph):
                return graph
        raise ValueError(f"Max number of generation attempts was reached and graph verification wasn't successful."
                         f"Try different requirements.")

    @staticmethod
    def load_graph(path) -> NasGraph:
        graph = NasGraph.load(path)
        return graph
