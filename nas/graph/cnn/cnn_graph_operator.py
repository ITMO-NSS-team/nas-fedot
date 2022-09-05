import random
from math import floor
from typing import (List, Any, Callable)

from nas.graph.cnn.cnn_builder import get_layer_params
from nas.graph.cnn.cnn_graph import NNGraph
from nas.utils.var import batch_norm_probability


def _add_flatten_node(node_func: Callable, current_node: Any, graph: Any):
    flatten_node = node_func(nodes_from=[current_node],
                             content={'name': 'flatten', 'params': False})
    graph._add_node(flatten_node)


def _add_batch_norm2node(node, requirements=None):
    batch_norm_params = get_layer_params('batch_normalization', requirements)
    node.content['params'] = node.content['params'] | batch_norm_params


def _create_conv2d_node(node_func: Callable, requirements=None):
    layer_params = get_layer_params('conv2d', requirements)
    new_node = node_func(content={'name': f'{layer_params["layer_type"]}',
                                  'params': layer_params}, nodes_from=None)
    if random.uniform(0, 1) > batch_norm_probability:
        _add_batch_norm2node(new_node, requirements)
    return new_node


def _create_secondary_node(node_func: Callable, requirements=None):
    new_node = random.choice(requirements.cnn_secondary)
    layer_params = get_layer_params(new_node, requirements)
    new_node = node_func(content={'name': layer_params['layer_type'], 'params': layer_params},
                         nodes_from=None)
    if random.uniform(0, 1) > batch_norm_probability:
        _add_batch_norm2node(new_node, requirements)
    return new_node


def _create_primary_nn_node(node_func: Callable, requirements=None):
    new_node = random.choice(requirements.primary)
    layer_params = get_layer_params(new_node, requirements)
    new_node = node_func(content={'name': layer_params['layer_type'], 'params': layer_params},
                         nodes_from=None)
    if random.uniform(0, 1) > batch_norm_probability:
        _add_batch_norm2node(new_node, requirements)
    return new_node


def _create_dropout_node(node_func: Callable, requirements=None):
    new_node = 'dropout'
    layer_params = get_layer_params(new_node, requirements)
    new_node = node_func(content={'name': layer_params['layer_type'],
                                  'params': layer_params}, nodes_from=None)
    return new_node


# TODO fix


def is_image_has_permissible_size(image_size, min_size: int = 2):
    return all([side_size >= min_size for side_size in image_size])


def output_dimension(input_dimension: float, kernel_size: int, stride: int) -> float:
    output_dim = ((input_dimension - kernel_size) / stride) + 1
    return output_dim


def conv_output_shape(node, image_size):
    image_size = [
        output_dimension(image_size[i], node.content['params'].kernel_size[i], node.content['params'].conv_strides[i])
        for i in range(len(image_size))]
    if node.layer_params['pool_size']:
        image_size = [
            output_dimension(image_size[i], node.content['params'].pool_size[i], node.content['params'].pool_strides[i])
            for i in range(len(image_size))]
        image_size = [floor(side_size) for side_size in image_size]
    return image_size


def generate_initial_graph(graph_class: Callable, node_func: Callable, node_list: List, requirements=None) -> NNGraph:
    """
    Method for initial graph generation from defined nodes list.

    :param graph_class: Initial graph class
    :param node_func: Node class
    :param node_list: List of nodes to graph generation
    :param requirements: List of parameters with generation restrictions. If none then default params will be chosen
    :return: graph:
    """

    def _add_node_to_graph(node_type: str, parent_node=None):
        parent = None if not parent_node else [parent_node]
        layer_params = get_layer_params(node_type, requirements)
        new_node = node_func(nodes_from=parent, content={'name': node_type,
                                                         'params': layer_params})
        graph._add_node(new_node)
        return new_node

    def _add_skip_connections(nodes_id: List[int], shortcuts_length: int = 2):
        for current_node in nodes_id:
            is_first_conv = current_node <= graph.cnn_depth
            is_second_conv = current_node + shortcuts_length < graph.cnn_depth
            if is_first_conv == is_second_conv and (current_node + shortcuts_length) < len(graph.nodes):
                graph.nodes[current_node + shortcuts_length].nodes_from.append(graph.nodes[current_node])
            else:
                print('Wrong connection. Connection dropped.')

    graph = graph_class()
    created_node = None
    for node in node_list:
        created_node = _add_node_to_graph(node_type=node, parent_node=created_node)
    if requirements.has_skip_connection:
        _add_skip_connections(nodes_id=requirements.skip_connections_id, shortcuts_length=requirements.shortcuts_len)

    return graph


# TODO optimize skip connections for both cases
def add_skip_connections(graph: NNGraph):
    """
    Add random skip connection to given graph_class

    :param graph: initial graph_class
    """
    skip_list = ['flatten', 'dropout', 'serial_connection']
    max_depth = len(graph.nodes)
    skip_connection_nodes_num = random.randint(0, max_depth // 2)
    skip_connection_prob = 0.2
    for _ in range(skip_connection_nodes_num):
        was_flatten = False
        for node_id in range(max_depth - 1):
            if graph.nodes[node_id].content['name'] in skip_list:
                was_flatten = True if graph.nodes[node_id].content['name'] == 'flatten' else was_flatten
                continue
            destination_node_id = node_id
            graph_node = graph.nodes[node_id]
            is_residual = random.uniform(0, 1) > skip_connection_prob
            if is_residual:
                if not was_flatten:
                    destination_node_id = random.randint(node_id, graph.cnn_depth - 1)
                elif was_flatten:
                    destination_node_id = random.randint(node_id, max_depth - 1)
                if destination_node_id == node_id or destination_node_id == node_id + 1:
                    continue
                graph.nodes[destination_node_id].nodes_from.append(graph_node)


def random_conv_graph_generation(graph_class: Callable, node_func: Callable, requirements) -> NNGraph:
    """
    Method for random graph_class generation with given requirements

    :param graph_class: type of graph_class to generate
    :param node_func: type of generated node
    :param requirements: list of requirements to nodes
    """

    def _random_cnn(max_num_of_conv: int = None, min_num_of_conv: int = None, parent_nodes: Any = None):
        """
        Method for generation convolutional part of the graph_class

        :param max_num_of_conv: max number of nodes in conv part of the graph_class
        :param min_num_of_conv: min number of nodes in conv part of the graph_class
        :param parent_nodes: node from conv part will grow
        """
        max_num_of_conv = max_num_of_conv if max_num_of_conv is not None else requirements.max_num_of_conv_layers
        min_num_of_conv = min_num_of_conv if min_num_of_conv is not None else requirements.min_num_of_conv_layers
        num_of_conv = random.randint(min_num_of_conv, max_num_of_conv)

        def _growth_conv_node(node_parent: Any = None, depth: int = None, img_size: List[float] = None):
            depth = 0 if depth is None else depth
            nodes_from = None if node_parent is None else [node_parent]
            new_node = _create_conv2d_node(node_func=node_func, requirements=requirements)
            new_node.nodes_from = nodes_from
            depth += 1
            if depth > num_of_conv:
                return
            graph._add_node(new_node)
            nodes_from = new_node
            if depth < num_of_conv - 1:
                new_node = _create_secondary_node(node_func=node_func,
                                                  requirements=requirements)
                new_node.nodes_from = [nodes_from]
                depth += 1
                if depth > num_of_conv:
                    return
                graph._add_node(new_node)
            else:
                is_dropout = random.randint(0, 1)
                if is_dropout:
                    new_node = _create_dropout_node(node_func=node_func, requirements=requirements)
                    depth += 1
                    if depth > num_of_conv:
                        return
                    graph._add_node(new_node)
                    new_node.nodes_from = [nodes_from]
            nodes_from = new_node
            if depth < num_of_conv:
                _growth_conv_node(node_parent=nodes_from, img_size=img_size, depth=depth)

        parent_nodes = parent_nodes if parent_nodes else None
        _growth_conv_node(node_parent=parent_nodes)
        _add_flatten_node(node_func, graph.nodes[-1], graph)

    def _random_nn(root_node: Any = None, max_depth: int = None):
        """
        Method that generates dense part of the graph_class

        :param root_node: parent node where dense part starts grow from
        :param max_depth: max number of nodes in dense part
        """
        max_depth = max_depth if max_depth is not None else requirements.max_nn_depth

        def _growth_nn(node_parent: Any = None, depth: int = None):
            nodes_from = [node_parent] if node_parent is not None else None
            depth = 0 if depth is None else depth
            new_node = _create_primary_nn_node(node_func=node_func, requirements=requirements)
            new_node.nodes_from = nodes_from
            depth += 1
            if depth > max_depth:
                return
            graph._add_node(new_node)
            nodes_from = new_node
            if depth < max_depth - 1:
                new_node = _create_secondary_node(node_func=node_func, requirements=requirements)
                new_node.nodes_from = [nodes_from]
                depth += 1
                if depth > max_depth:
                    return
                graph._add_node(new_node)
                nodes_from = new_node
            else:
                is_additional_dense = random.randint(0, 1)
                if is_additional_dense:
                    new_node = _create_primary_nn_node(node_func=node_func, requirements=requirements)
                    new_node.nodes_from = [nodes_from]
                    depth += 1
                    if depth > max_depth:
                        return
                    graph._add_node(new_node)
                    nodes_from = new_node
            if depth < max_depth:
                _growth_nn(node_parent=nodes_from, depth=depth)

        root_node = root_node if root_node else None
        _growth_nn(node_parent=root_node)

    graph = graph_class()
    _random_cnn()
    _random_nn(root_node=graph.nodes[-1])
    if requirements.has_skip_connection:
        add_skip_connections(graph)

    if not hasattr(graph, 'parent_operators'):
        setattr(graph, 'parent_operators', [])
    return graph
