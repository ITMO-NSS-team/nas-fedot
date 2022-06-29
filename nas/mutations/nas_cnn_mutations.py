from random import random, choice
from typing import Any

from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from nas.composer.cnn.cnn_builder import get_layer_params
from nas.composer.cnn.cnn_graph_node import CNNNode
from nas.composer.cnn.cnn_graph import CNNGraph
from nas.utils.utils import seed_all

seed_all(1)


def cnn_simple_mutation(graph: CNNGraph, requirements, **kwargs) -> Any:
    node_mutation_probability = requirements.mutation_prob
    for i, node in enumerate(graph.nodes[::-1]):
        if random() < node_mutation_probability and not node.content['name'] == 'flatten':
            old_node_type = node.content['name']
            if old_node_type == 'conv2d':
                new_layer_params = get_layer_params(old_node_type, requirements)
            else:
                node_type = choice(requirements.secondary)
                new_layer_params = get_layer_params(node_type, requirements)
            new_nodes_from = None if not node.nodes_from else [node.nodes_from[0]]
            new_node = CNNNode(nodes_from=new_nodes_from,
                               content={'name': new_layer_params["layer_type"],
                                        'params': new_layer_params})
            if 'momentum' in node.content['params']:
                new_node.content['params']['momentum'] = node.content['params']['momentum']
                new_node.content['params']['epsilon'] = node.content['params']['epsilon']
            graph.update_node(node, new_node)
    return graph


mutation_by_type = {
    MutationTypesEnum.simple: cnn_simple_mutation,
}
