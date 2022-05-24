from random import random, choice
from typing import Any

from fedot.core.dag.validation_rules import ERROR_PREFIX
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from nas.graph_cnn_gp_operators import get_layer_params
from nas.composer.graph_gp_cnn_composer import GPNNComposerRequirements, NNNode, NNGraph


def flatten_check(graph: 'NNGraph'):
    cnt = 0
    for node in graph.nodes:
        if node.content['name'] == 'flatten':
            cnt += 1
            if cnt > 1:
                raise ValueError(f'{ERROR_PREFIX} Graph should have only one flatten layer')
    return True


def has_no_flatten_skip(graph: 'NNGraph'):
    for node in graph.free_nodes:
        if node.content['name'] == 'flatten':
            return True
    raise ValueError(f'{ERROR_PREFIX} Graph has wrong skip connections')


def graph_has_several_starts(graph: 'NNGraph'):
    cnt = 0
    for node in graph.graph_struct:
        if not node.nodes_from:
            cnt += 1
        if cnt > 1:
            raise ValueError(f'{ERROR_PREFIX} Graph has more than one start node')
    return True


def graph_has_wrong_structure(graph: 'NNGraph'):
    if graph.graph_struct[0].content['name'] != 'conv2d':
        raise ValueError(f'{ERROR_PREFIX} Graph has no conv layers in conv part')
    return True


def cnn_simple_mutation(graph: Any, requirements, **kwargs) -> Any:
    was_flatten = True
    node_mutation_probability = requirements.mutation_prob
    nn_structure = graph.nodes[::-1]
    secondary_nodes = requirements.secondary
    for i, node in enumerate(nn_structure):
        if node.content['name'] == 'flatten':
            was_flatten = False
            continue
        if not was_flatten:
            if random() < node_mutation_probability:
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
                new_node = NNNode(nodes_from=new_nodes_from,
                                  content={'name': new_layer_params["layer_type"],
                                           'params': new_layer_params})
                if 'momentum' in node.content['params']:
                    new_node.content['params']['momentum'] = node.content['params']['momentum']
                    new_node.content['params']['epsilon'] = node.content['params']['epsilon']
                graph.update_node(node, new_node)
        else:
            if random() < node_mutation_probability:
                new_node_type = choice(secondary_nodes)
                new_layer_params = get_layer_params(new_node_type, requirements)
                new_nodes_from = None if not node.nodes_from else node.nodes_from
                new_node = NNNode(nodes_from=new_nodes_from,
                                  content={'name': new_layer_params["layer_type"],
                                           'params': new_layer_params})
                if 'momentum' in node.content['params']:
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
