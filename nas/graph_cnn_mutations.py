from random import random, choice
from typing import Any
from random import randint
from functools import partial

from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
# from fedot_old.core.composer.optimisers.gp_operators import nodes_from_height, node_height, node_depth
from nas.cnn_gp_operators import get_random_layer_params, random_nn_branch, random_cnn, \
    conv_output_shape
from nas.composer.graph_gp_cnn_composer import GPNNComposerRequirements
from nas.graph_nas_node import NNNodeGenerator
from nas.composer.graph_gp_cnn_composer import CustomGraphModel, CustomGraphAdapter, CustomGraphNode
from nas.graph_keras_eval import generate_structure
from nas.layer import LayerTypesIdsEnum, LayerParams


# from fedot_old.core.composer.visualisation import ComposerVisualiser


def cnn_simple_mutation(graph: Any, requirements: GPNNComposerRequirements, params: GraphGenerationParams,
                        max_depth) -> Any:
    node_mutation_probability = requirements.mutation_prob
    cnn_structure = graph.cnn_depth
    nn_structure = generate_structure(graph.root_node)[::-1]
    # for node in cnn_structure:
    #     if random() < node_mutation_probability:
    #         old_node_type = node.layer_params.layer_type
    #         if old_node_type == LayerTypesIdsEnum.conv2d:
    #             activation = choice(requirements.activation_types)
    #             new_layer_params = LayerParams(layer_type=old_node_type, activation=activation,
    #                                            kernel_size=node.layer_params.kernel_size,
    #                                            conv_strides=node.layer_params.conv_strides,
    #                                            pool_size=node.layer_params.pool_size,
    #                                            pool_strides=node.layer_params.pool_strides,
    #                                            pool_type=choice(requirements.pool_types),
    #                                            num_of_filters=choice(requirements.filters))
    #         else:
    #             node_type = choice(requirements.secondary)
    #             new_layer_params = get_random_layer_params(node_type, requirements)
    #         new_node = NNNodeGenerator.secondary_node(layer_params=new_layer_params,
    #                                                   content={'name': new_layer_params.layer_type})
    #         graph.update_node(node, new_node)
    secondary_nodes = requirements.secondary
    primary_nodes = requirements.primary
    for i, node in enumerate(nn_structure):
        if i < cnn_structure:
            if random() < node_mutation_probability:
                old_node_type = node.layer_params.layer_type
                if old_node_type == LayerTypesIdsEnum.conv2d.value:
                    activation = choice(requirements.activation_types).value
                    new_layer_params = LayerParams(layer_type=old_node_type, activation=activation,
                                                   kernel_size=node.layer_params.kernel_size,
                                                   conv_strides=node.layer_params.conv_strides,
                                                   pool_size=node.layer_params.pool_size,
                                                   pool_strides=node.layer_params.pool_strides,
                                                   pool_type=choice(requirements.pool_types),
                                                   num_of_filters=choice(requirements.filters))
                else:
                    node_type = choice(requirements.secondary)
                    new_layer_params = get_random_layer_params(node_type, requirements)
                new_nodes_from = None if not node.nodes_from else node.nodes_from
                new_node = CustomGraphNode(nodes_from=new_nodes_from, layer_params=new_layer_params,
                                           content={'name': new_layer_params.layer_type})
                graph.update_node(node, new_node)
        else:
            if random() < node_mutation_probability:
                if node.nodes_from:
                    new_node_type = choice(secondary_nodes)
                    new_layer_params = get_random_layer_params(new_node_type, requirements)
                    new_nodes_from = None if not node.nodes_from else node.nodes_from
                    new_node = CustomGraphNode(nodes_from=new_nodes_from, layer_params=new_layer_params,
                                               content={'name': new_layer_params.layer_type})
                # else:
                #     new_node_type = choice(primary_nodes)
                #     new_layer_params = get_random_layer_params(new_node_type, requirements)
                #     new_node = CustomGraphNode(layer_params=new_layer_params,
                #                                content={'name': new_layer_params.layer_type})
                try:
                    graph.update_node(node, new_node)
                except Exception as ex:
                    print(f'error in updating nodes: {ex}')
    graph.show()
    return graph


def cnn_growth_mutation(graph: Any, params, local_growth: bool = True) -> Any:
    cnn_part_growth_mutation(graph=graph, params=params)
    nn_growth_mutation(graph=graph, params=params, local_growth=local_growth)
    return graph


def cnn_part_growth_mutation(graph: Any, params) -> Any:
    point_in_chain = randint(1, len(graph.cnn_nodes))
    old_nodes = graph.cnn_nodes[:point_in_chain]
    if len(old_nodes) % 2:
        num_of_conv = ((len(old_nodes) - 1) / 2.) + 1
    else:
        num_of_conv = len(old_nodes) / 2.
    max_num_of_conv = params.requirements.max_num_of_conv_layers - num_of_conv
    min_num_of_conv = params.requirements.min_num_of_conv_layers - num_of_conv
    if max_num_of_conv > 0:
        if old_nodes[len(old_nodes) - 1].layer_params.layer_type in params.requirements.conv_types:
            node_type = choice(params.requirements.cnn_secondary)
            layer_params = get_random_layer_params(node_type, params.requirements)
            new_node = params.secondary_node_func(layer_params=layer_params)
            graph.add_cnn_node(new_node)
        image_size = params.requirements.image_size
        for node in old_nodes:
            if node.layer_params.layer_type in params.requirements.conv_types:
                image_size = conv_output_shape(node, image_size)
        graph.replace_cnn_nodes(new_nodes=old_nodes)
        random_cnn(secondary_node_func=params.secondary_node_func, requirements=params.requirements,
                   graph=graph, max_num_of_conv=max_num_of_conv, min_num_of_conv=min_num_of_conv, image_size=image_size)


def nn_growth_mutation(graph: Any, params, local_growth=True) -> Any:
    ComposerVisualiser.visualise(graph)
    primary_nodes = params.requirements.primary
    secondary_nodes = params.requirements.secondary
    random_layer_in_chain = randint(0, node_depth(graph.root_node))
    print(random_layer_in_chain)
    print(node_depth(graph.root_node) + 1)
    node_from_chain = choice(nodes_from_height(graph.root_node, random_layer_in_chain))
    if local_growth:
        is_primary_node_selected = (not node_from_chain.nodes_from) or (
                node_from_chain.nodes_from and node_from_chain != graph.root_node and randint(0, 1))
    else:
        is_primary_node_selected = randint(0, 1) and not node_height(graph, node_from_chain) \
                                                         < params.requirements.max_depth
    if is_primary_node_selected:
        new_node_type = choice(primary_nodes)
        new_layer_params = get_random_layer_params(new_node_type, params.requirements)
        new_subtree = params.primary_node_func(layer_params=new_layer_params)
        graph.replace_node_with_parents(node_from_chain, new_subtree)
    else:
        if local_growth:
            max_depth = node_depth(node_from_chain)
        else:
            max_depth = params.requirements.max_depth - random_layer_in_chain
        new_node_type = choice(secondary_nodes)
        new_layer_params = get_random_layer_params(new_node_type, params.requirements)
        new_subtree = params.secondary_node_func(layer_params=new_layer_params)
        offspring_size = randint(params.requirements.min_arity, params.requirements.max_arity)
        for _ in range(offspring_size):
            random_nn_branch(secondary_node_func=params.secondary_node_func,
                             primary_node_func=params.primary_node_func,
                             requirements=params.requirements,
                             max_depth=max_depth, start_height=(node_height(graph, node_from_chain)),
                             node_parent=new_subtree)
        graph.replace_node_with_parents(node_from_chain, new_subtree)


mutation_by_type = {
    MutationTypesEnum.simple: cnn_simple_mutation,
    MutationTypesEnum.growth: partial(cnn_growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(cnn_growth_mutation, local_growth=True),
}