from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from random import random, choice, randint
from typing import (Any, Callable)
from functools import partial
from fedot.core.composer.chain import Chain, List
from fedot.core.composer.optimisers.gp_operators import nodes_from_height, node_depth, random_ml_chain, node_height


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    local_growth = 'local_growth'
    reduce = 'reduce'
    none = 'none'


class MutationStrengthEnum(Enum):
    weak = 0
    mean = 1
    strong = 2


@dataclass
class MutationParams:
    secondary_node_func: Callable = None
    primary_node_func: Callable = None
    chain_class: Any = None
    requirements: Any = None


def get_mutation_prob(mut_id, root_node):
    default_mutation_prob = 0.7
    if mut_id == MutationStrengthEnum.weak.value:
        mutation_strength = 0.2
        return mutation_strength / (node_depth(root_node) + 1)
    elif mut_id == MutationStrengthEnum.mean.value:
        mutation_strength = 1.0
        return mutation_strength / (node_depth(root_node) + 1)
    elif mut_id == MutationStrengthEnum.strong.value:
        mutation_strength = 5.0
        return mutation_strength / (node_depth(root_node) + 1)
    else:
        return default_mutation_prob


def mutation(types: List[MutationTypesEnum], chain_class, chain: Chain, requirements,
             secondary_node_func: Callable = None, primary_node_func: Callable = None, types_dict: dict = None) -> Any:
    mutation_prob = requirements.mutation_prob
    if mutation_prob and random() > mutation_prob:
        return deepcopy(chain)

    type = choice(types)
    if type == MutationTypesEnum.none:
        new_chain = deepcopy(chain)
    elif type in types_dict.keys():
        mutation_params = MutationParams(secondary_node_func=secondary_node_func,
                                         primary_node_func=primary_node_func, chain_class=chain_class,
                                         requirements=requirements)
        return types_dict[type](deepcopy(chain), mutation_params)
    else:
        raise ValueError(f'Required mutation not found: {type}')
    return new_chain


def simple_mutation(chain: Any, parameters: MutationParams) -> Any:
    node_mutation_probability = get_mutation_prob(mut_id=parameters.requirements.mutation_strength.value,
                                                  root_node=chain.root_node)

    def replace_node_to_random_recursive(node: Any) -> Any:
        if node.nodes_from:
            if random() < node_mutation_probability:
                secondary_node = parameters.secondary_node_func(model_type=choice(parameters.requirements.secondary),
                                                                nodes_from=node.nodes_from)
                chain.update_node(node, secondary_node)
            for child in node.nodes_from:
                replace_node_to_random_recursive(child)
        else:
            if random() < node_mutation_probability:
                primary_node = parameters.primary_node_func(model_type=choice(parameters.requirements.primary))
                chain.update_node(node, primary_node)

    replace_node_to_random_recursive(chain.root_node)

    return chain


def growth_mutation(chain: Any, parameters: MutationParams, local_growth=True) -> Any:
    random_layer_in_chain = randint(0, chain.depth - 1)
    node_from_chain = choice(nodes_from_height(chain.root_node, random_layer_in_chain))
    if local_growth:
        is_primary_node_selected = (not node_from_chain.nodes_from) or (
                node_from_chain.nodes_from and node_from_chain != chain.root_node and randint(0, 1))
    else:
        is_primary_node_selected = randint(0, 1) and not node_height(chain, node_from_chain) \
                                                         < parameters.requirements.max_depth

    if is_primary_node_selected:
        new_subtree = parameters.primary_node_func(model_type=choice(parameters.requirements.primary))
    else:
        if local_growth:
            max_depth = node_depth(node_from_chain)
        else:

            max_depth = parameters.requirements.max_depth - node_height(chain, node_from_chain)
        new_subtree = random_ml_chain(parameters.chain_class, parameters.secondary_node_func,
                                      parameters.primary_node_func, parameters.requirements,
                                      max_depth=max_depth).root_node
    chain.replace_node_with_parents(node_from_chain, new_subtree)
    return chain


def reduce_mutation(chain: Any, parameters: MutationParams) -> Any:
    nodes = [node for node in chain.nodes if not node is chain.root_node]
    node_to_del = choice(nodes)
    childs = chain.node_childs(node_to_del)
    is_possible_to_delete = all([len(child.nodes_from) - 1 >= parameters.requirements.min_arity for child in childs])
    if is_possible_to_delete:
        chain.delete_node(node_to_del)
    else:
        primary_node = parameters.primary_node_func(model_type=choice(parameters.requirements.primary))
        chain.replace_node_with_parents(node_to_del, primary_node)
    return chain


mutation_by_type = {
    MutationTypesEnum.simple: simple_mutation,
    MutationTypesEnum.growth: partial(growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(growth_mutation, local_growth=True),
    MutationTypesEnum.reduce: reduce_mutation
}
