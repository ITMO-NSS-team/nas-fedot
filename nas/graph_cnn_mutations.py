from random import random, choice
from typing import Any

from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from nas.graph_cnn_gp_operators import get_random_layer_params
from nas.composer.graph_gp_cnn_composer import GPNNComposerRequirements
from nas.composer.graph_gp_cnn_composer import NNNode
from nas.graph_keras_eval import generate_structure
from nas.layer import LayerTypesIdsEnum, LayerParams


def cnn_simple_mutation(graph: Any, requirements: GPNNComposerRequirements, params: GraphGenerationParams,
                        max_depth) -> Any:
    node_mutation_probability = requirements.mutation_prob
    cnn_structure = graph.cnn_depth
    nn_structure = generate_structure(graph.root_node)[::-1]
    secondary_nodes = requirements.secondary
    for i, node in enumerate(nn_structure):
        if i < cnn_structure:
            if random() < node_mutation_probability:
                old_node_type = node.content['params'].layer_type
                if old_node_type == LayerTypesIdsEnum.conv2d.value:
                    activation = choice(requirements.activation_types).value
                    new_layer_params = LayerParams(layer_type=old_node_type, activation=activation,
                                                   kernel_size=node.content['params'].kernel_size,
                                                   conv_strides=node.content['params'].conv_strides,
                                                   pool_size=node.content['params'].pool_size,
                                                   pool_strides=node.content['params'].pool_strides,
                                                   pool_type=choice(requirements.pool_types),
                                                   num_of_filters=choice(requirements.filters))
                else:
                    node_type = choice(requirements.secondary)
                    new_layer_params = get_random_layer_params(node_type, requirements)
                new_nodes_from = None if not node.nodes_from else node.nodes_from
                new_node = NNNode(nodes_from=new_nodes_from,
                                  content={'name': f'{new_layer_params.layer_type}',
                                           'conv': True, 'params': new_layer_params})
                graph.update_node(node, new_node)
        else:
            if random() < node_mutation_probability:
                if node.nodes_from:
                    new_node_type = choice(secondary_nodes)
                    new_layer_params = get_random_layer_params(new_node_type, requirements)
                    new_nodes_from = None if not node.nodes_from else node.nodes_from
                    new_node = NNNode(nodes_from=new_nodes_from,
                                      content={'name': new_layer_params.layer_type,
                                               'params': new_layer_params})
                try:
                    graph.update_node(node, new_node)
                except Exception as ex:
                    print(f'error in updating nodes: {ex}')
    return graph


mutation_by_type = {
    MutationTypesEnum.simple: cnn_simple_mutation,
}
