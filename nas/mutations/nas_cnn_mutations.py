from random import random, choice
from typing import Any

from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from nas.composer.cnn.cnn_builder import get_layer_params
from nas.composer.cnn.cnn_graph_node import CNNNode


def cnn_simple_mutation(graph: Any, requirements, **kwargs) -> Any:
    node_mutation_probability = requirements.mutation_prob
    nn_structure = graph.nodes[::-1]
    for i, node in enumerate(nn_structure):
        flatten = node.content['name'] == 'flatten'
        if random() < node_mutation_probability and not flatten:
            old_node_type = node.content['name']
            if old_node_type == 'conv2d':
                activation = choice(requirements.activation_types).value
                new_layer_params = {'layer_type': old_node_type, 'activation': activation,
                                    'kernel_size': node.content['params']['kernel_size'],
                                    'conv_strides': node.content['params']['conv_strides'],
                                    'pool_size': node.content['params']['pool_size'],
                                    'pool_strides': node.content['params']['pool_strides'],
                                    'pool_type': choice(requirements.pool_types),
                                    'num_of_filters': choice(requirements.filters)}
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
