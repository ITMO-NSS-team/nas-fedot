from random import random, choice, sample
from typing import Any
from copy import deepcopy

from fedot.core.dag.validation_rules import ERROR_PREFIX
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, MAX_NUM_OF_ATTEMPTS
from nas.graph_cnn_gp_operators import get_random_layer_params
from nas.composer.graph_gp_cnn_composer import GPNNComposerRequirements, NNNode, NNGraph
from nas.layer import LayerTypesIdsEnum


def has_no_flatten_skip(graph: 'NNGraph'):
    reversed_graph = deepcopy(graph.nodes)[::-1]
    skip_connection_parents_list = []
    for node in reversed_graph:
        if len(node.nodes_from) > 1:
            skip_connection_parents_list.extend(node.nodes_from[1:])
            for parent in skip_connection_parents_list:
                is_wrong_connection = ['conv' in parent.content and 'conv' not in node.content]
                if is_wrong_connection:
                    raise ValueError(f'{ERROR_PREFIX} Graph has cycles')
    return True


def cnn_simple_mutation(graph: Any, requirements: GPNNComposerRequirements, params: GraphGenerationParams,
                        max_depth) -> Any:
    node_mutation_probability = requirements.mutation_prob
    nn_structure = graph.nodes
    secondary_nodes = requirements.secondary
    for i, node in enumerate(nn_structure):
        if 'conv' in node.content:
            if random() < node_mutation_probability:
                old_node_type = node.content['params']['layer_type']
                if old_node_type == LayerTypesIdsEnum.conv2d.value:
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
                    new_layer_params = get_random_layer_params(node_type, requirements)
                new_nodes_from = None if not node.nodes_from else [node.nodes_from[0]]
                new_node = NNNode(nodes_from=new_nodes_from,
                                  content={'name': new_layer_params["layer_type"],
                                           'params': new_layer_params, 'conv': True, 'batch_norm': False})
                # TODO
                if node.content['batch_norm']:
                    new_node.content['batch_norm'] = True
                    new_node.content['params']['momentum'] = node.content['params']['momentum']
                    new_node.content['params']['epsilon'] = node.content['params']['epsilon']
                graph.update_node(node, new_node)
        else:
            if random() < node_mutation_probability:
                if node.nodes_from:
                    new_node_type = choice(secondary_nodes)
                    new_layer_params = get_random_layer_params(new_node_type, requirements)
                    new_nodes_from = None if not node.nodes_from else node.nodes_from
                    new_node = NNNode(nodes_from=new_nodes_from,
                                      content={'name': new_layer_params["layer_type"],
                                               'params': new_layer_params})
                if node.content['batch_norm']:
                    new_node.content['batch_norm'] = True
                    new_node.content['params']['momentum'] = node.content['params']['momentum']
                    new_node.content['params']['epsilon'] = node.content['params']['epsilon']
                try:
                    graph.update_node(node, new_node)
                except Exception as ex:
                    print(f'error in updating nodes: {ex}')
    return graph


mutation_by_type = {
    MutationTypesEnum.simple: cnn_simple_mutation,
}
