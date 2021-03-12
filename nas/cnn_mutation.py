from random import random, choice
from typing import Any
from random import randint
from functools import partial
from fedot.core.composer.optimisers.mutation import MutationTypesEnum, MutationParams, get_mutation_prob
from fedot.core.composer.optimisers.gp_operators import nodes_from_height, node_height, node_depth
from nas.cnn_gp_operators import get_random_layer_params, random_nn_branch, random_cnn, \
    conv_output_shape
from nas.keras_eval import generate_structure
from nas.layer import LayerTypesIdsEnum, LayerParams
from fedot.core.composer.visualisation import ComposerVisualiser


def cnn_simple_mutation(chain: Any, parameters: MutationParams, node_mutation_probability: float = 0.6) -> Any:
    cnn_structure = chain.cnn_nodes
    nn_structure = generate_structure(chain.root_node)
    for node in cnn_structure:
        if random() < node_mutation_probability:
            old_node_type = node.layer_params.layer_type
            if old_node_type == LayerTypesIdsEnum.conv2d:
                activation = choice(parameters.requirements.activation_types)
                new_layer_params = LayerParams(layer_type=old_node_type, activation=activation,
                                               kernel_size=node.layer_params.kernel_size,
                                               conv_strides=node.layer_params.conv_strides,
                                               pool_size=node.layer_params.pool_size,
                                               pool_strides=node.layer_params.pool_strides,
                                               pool_type=choice(parameters.requirements.pool_types),
                                               num_of_filters=choice(parameters.requirements.filters))
            else:
                node_type = choice(parameters.requirements.secondary)
                new_layer_params = get_random_layer_params(node_type, parameters.requirements)
            new_node = parameters.secondary_node_func(layer_params=new_layer_params)
            chain.update_cnn_node(node, new_node)
    secondary_nodes = parameters.requirements.secondary
    primary_nodes = parameters.requirements.primary
    for node in nn_structure:
        if random() < node_mutation_probability:
            if node.nodes_from:
                new_node_type = choice(secondary_nodes)
                new_layer_params = get_random_layer_params(new_node_type, parameters.requirements)
                new_node = parameters.secondary_node_func(layer_params=new_layer_params)
            else:
                new_node_type = choice(primary_nodes)
                new_layer_params = get_random_layer_params(new_node_type, parameters.requirements)
                new_node = parameters.primary_node_func(layer_params=new_layer_params)
            chain.update_node(node, new_node)
    return chain


def cnn_growth_mutation(chain: Any, parameters: MutationParams, local_growth: bool = True) -> Any:
    cnn_part_growth_mutation(chain=chain, parameters=parameters)
    nn_growth_mutation(chain=chain, parameters=parameters, local_growth=local_growth)
    return chain


def cnn_part_growth_mutation(chain: Any, parameters: MutationParams) -> Any:
    point_in_chain = randint(1, len(chain.cnn_nodes))
    old_nodes = chain.cnn_nodes[:point_in_chain]
    if len(old_nodes) % 2:
        num_of_conv = ((len(old_nodes) - 1) / 2.) + 1
    else:
        num_of_conv = len(old_nodes) / 2.
    max_num_of_conv = parameters.requirements.max_num_of_conv_layers - num_of_conv
    min_num_of_conv = parameters.requirements.min_num_of_conv_layers - num_of_conv
    if max_num_of_conv > 0:
        if old_nodes[len(old_nodes) - 1].layer_params.layer_type in parameters.requirements.conv_types:
            node_type = choice(parameters.requirements.cnn_secondary)
            layer_params = get_random_layer_params(node_type, parameters.requirements)
            new_node = parameters.secondary_node_func(layer_params=layer_params)
            chain.add_cnn_node(new_node)
        image_size = parameters.requirements.image_size
        for node in old_nodes:
            if node.layer_params.layer_type in parameters.requirements.conv_types:
                image_size = conv_output_shape(node, image_size)
        chain.replace_cnn_nodes(new_nodes=old_nodes)
        random_cnn(secondary_node_func=parameters.secondary_node_func, requirements=parameters.requirements,
                   chain=chain, max_num_of_conv=max_num_of_conv, min_num_of_conv=min_num_of_conv, image_size=image_size)


def nn_growth_mutation(chain: Any, parameters: MutationParams, local_growth=True) -> Any:
    ComposerVisualiser.visualise(chain)
    primary_nodes = parameters.requirements.primary
    secondary_nodes = parameters.requirements.secondary
    random_layer_in_chain = randint(0, node_depth(chain.root_node))
    print(random_layer_in_chain)
    print(node_depth(chain.root_node) + 1)
    node_from_chain = choice(nodes_from_height(chain.root_node, random_layer_in_chain))
    if local_growth:
        is_primary_node_selected = (not node_from_chain.nodes_from) or (
                node_from_chain.nodes_from and node_from_chain != chain.root_node and randint(0, 1))
    else:
        is_primary_node_selected = randint(0, 1) and not node_height(chain, node_from_chain) \
                                                         < parameters.requirements.max_depth
    if is_primary_node_selected:
        new_node_type = choice(primary_nodes)
        new_layer_params = get_random_layer_params(new_node_type, parameters.requirements)
        new_subtree = parameters.primary_node_func(layer_params=new_layer_params)
        chain.replace_node_with_parents(node_from_chain, new_subtree)
    else:
        if local_growth:
            max_depth = node_depth(node_from_chain)
        else:
            max_depth = parameters.requirements.max_depth - random_layer_in_chain
        new_node_type = choice(secondary_nodes)
        new_layer_params = get_random_layer_params(new_node_type, parameters.requirements)
        new_subtree = parameters.secondary_node_func(layer_params=new_layer_params)
        offspring_size = randint(parameters.requirements.min_arity, parameters.requirements.max_arity)
        for _ in range(offspring_size):
            random_nn_branch(secondary_node_func=parameters.secondary_node_func,
                             primary_node_func=parameters.primary_node_func,
                             requirements=parameters.requirements,
                             max_depth=max_depth, start_height=(node_height(chain, node_from_chain)),
                             node_parent=new_subtree)
        chain.replace_node_with_parents(node_from_chain, new_subtree)


mutation_by_type = {
    MutationTypesEnum.simple: cnn_simple_mutation,
    MutationTypesEnum.growth: partial(cnn_growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(cnn_growth_mutation, local_growth=True),
}
