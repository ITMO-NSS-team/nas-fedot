import random
from typing import List, Optional

from nas.composer.nn_composer_requirements import ModelRequirements
from nas.graph.cnn.cnn_graph import NasGraph
from nas.graph.grpah_generator import GraphGenerator
from nas.graph.node.nn_graph_node import NNNode, get_node_params_by_type
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
    def __init__(self, param_restrictions: ModelRequirements,
                 initial_struct: Optional[List] = None):
        self._initial_struct = initial_struct
        self._param_restrictions = param_restrictions

    @property
    def initial_struct(self):
        return self._initial_struct

    @property
    def param_restrictions(self):
        return self._param_restrictions

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

    def _generate_from_scratch(self):
        total_conv_nodes = random.randint(self.param_restrictions.min_num_of_conv_layers,
                                          self.param_restrictions.max_num_of_conv_layers)
        total_fc_nodes = random.randint(self.param_restrictions.min_nn_depth,
                                        self.param_restrictions.max_nn_depth)
        # hotfix
        zero_node = random.choice(self.param_restrictions.primary)
        graph_nodes = [zero_node]
        for i in range(1, total_conv_nodes + total_fc_nodes):
            if i < total_conv_nodes:
                node = random.choice(self.param_restrictions.primary) \
                    if i != total_conv_nodes - 1 else LayersPoolEnum.flatten
            else:
                node = random.choice(self.param_restrictions.secondary)
            graph_nodes.append(node)
        return graph_nodes

    def _set_input_shape(self, graph, input_shape) -> NasGraph:
        graph.input_shape = input_shape
        return graph

    def _add_node(self, node_to_add, parent_node):
        node_params = get_node_params_by_type(node_to_add, self.param_restrictions)
        node = NNNode(content={'name': node_to_add.value, 'params': node_params}, nodes_from=parent_node)
        return node

    def build(self) -> NasGraph:
        is_correct_graph = False
        while not is_correct_graph:
            graph = NasGraph()
            parent_node = None
            graph_nodes = self.initial_struct if self.initial_struct else self._generate_from_scratch()
            for node in graph_nodes:
                node = self._add_node(node, parent_node)
                parent_node = [node]
                graph.add_node(node)
            if self.param_restrictions.has_skip_connection:
                _add_skip_connections(graph, self._get_skip_connection_params(graph))
            graph = self._set_input_shape(graph, self.param_restrictions.conv_requirements.input_shape)
            total_params = graph.get_trainable_params()
            if total_params < self.param_restrictions.max_possible_parameters:
                is_correct_graph = True
        return graph

    @staticmethod
    def load_graph(path) -> NasGraph:
        graph = NasGraph.load(path)
        return graph
