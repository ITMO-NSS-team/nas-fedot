from fedot.core.optimisers.gp_comp.operators.crossover import subtree_crossover
from random import randint
from typing import Any
from nas.layer import LayerTypesIdsEnum, LayerParams
from nas.nas_node import NNNodeGenerator
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum


def cnn_crossover(graph_first: Any, graph_second: Any, max_depth, params) -> Any:
    max_num_of_conv = params.max_num_of_conv_layers
    min_num_of_conv = params.min_num_of_conv_layers
    is_crossoover_permissible = True
    num_of_genes = []
    num_of_conv = []
    parts = []
    for i, ind in enumerate((graph_first.cnn_nodes, graph_second.cnn_nodes)):
        end_point = (((len(ind) - 1) / 2) + 1) if len(ind) % 2 else len(ind) / 2.

        if i == 0:
            crossover_point = randint(1, end_point)
        else:
            if num_of_conv[0] == max_num_of_conv:
                is_crossoover_permissible = False
                break
            else:
                if num_of_conv[0] < min_num_of_conv:
                    end_point = end_point - (min_num_of_conv - num_of_conv[0]) + 1
            crossover_point = randint(0, end_point - 1)

        genes = ((crossover_point - 1) * 2) + 1 if crossover_point == end_point and len(
            ind) % 2 else crossover_point * 2
        num_of_genes.append(genes)
        if i == 0:
            parts.append(graph_first.cnn_nodes[:num_of_genes[0]])
        else:
            parts.append(graph_second.cnn_nodes[num_of_genes[1]:])

        num_of_conv.append(len(
            [True for node in parts[i] if node.layer_params.layer_type == LayerTypesIdsEnum.conv2d]))

    if is_crossoover_permissible:
        num_of_conv_in_first, num_of_conv_in_second = num_of_conv
        part_from_first, part_from_second = parts
        if num_of_conv_in_first + num_of_conv_in_second > max_num_of_conv:
            permissible_num_of_conv_second = max_num_of_conv - num_of_conv_in_first
            part_for_delete_in_second = (num_of_conv_in_second - permissible_num_of_conv_second) * 2 if not len(
                graph_second.cnn_nodes) % 2 else ((num_of_conv_in_second - permissible_num_of_conv_second) - 1) * 2 + 1
            part_from_second = part_from_second[:-int(part_for_delete_in_second)]

        additional_layer = []
        if part_from_first[len(part_from_first) - 1].layer_params.layer_type == LayerTypesIdsEnum.conv2d and \
                part_from_second[0].layer_params.layer_type == LayerTypesIdsEnum.conv2d:
            layer_params = LayerParams(layer_type=LayerTypesIdsEnum.serial_connection)
            new_node = NNNodeGenerator.secondary_node(layer_params=layer_params)
            additional_layer.append(new_node)
        old_nodes = part_from_first + additional_layer
        new_nodes = part_from_second
    else:
        old_nodes = parts[0]
        new_nodes = []

    graph_first.replace_cnn_nodes(new_nodes=old_nodes + new_nodes)
    return graph_first


def cnn_subtree_crossover(graph_first: Any, graph_second: Any, max_depth, params) -> Any:
    if graph_first.depth - 1 == 0 and graph_second.depth - 1 == 0:
        graph_first.replace_node_with_parents(graph_first.root_node, graph_second.root_node)
    else:
        graph_first, graph_second = subtree_crossover(graph_first, graph_second, max_depth)
    graph_first_res = cnn_crossover(graph_first, graph_second, max_depth, params)
    return graph_first_res
