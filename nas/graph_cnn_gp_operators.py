import random
from math import floor
from typing import (Tuple, List, Any, Callable)

VERBOSE_VAL = {0: -1, 'auto': 1}
BATCH_NORM_PROB = 0.4
DEFAULT_NODES_PARAMS = {
    'conv2d': {'layer_type': 'conv2d', 'activation': 'relu', 'kernel_size': (3, 3),
               'conv_strides': (2, 2), 'num_of_filters': 16, 'pool_size': (2, 2),
               'pool_strides': (2, 2), 'pool_type': 'max_pool2d'},
    'dropout': {'layer_type': 'dropout',
                'drop': 0.2},
    'batch_norm': {'layer_type': 'batch_normalization', 'epsilon': 0.001, 'momentum': 0.99},
    'dense': {'activation': 'relu', 'layer_type': 'dense', 'neurons': 121},
    'serial_connection': {'layer_type': 'serial_connection'},
    'flatten': None}


def _add_flatten_node(node_func: Callable, current_node: Any, graph: Any):
    flatten_node = node_func(nodes_from=[current_node],
                             content={'name': 'flatten', 'params': False})
    graph.add_node(flatten_node)


def _add_batch_norm2node(node, requirements=None):
    batch_norm_params = get_layer_params('batch_normalization', requirements)
    node.content['params'] = node.content['params'] | batch_norm_params


def create_conv2d_node(node_func: Callable, requirements=None, image_size: List = None):
    if requirements is not None:
        conv_node_type = random.choice(requirements.conv_types)
        activation = random.choice(requirements.activation_types).value
        kernel_size = requirements.conv_kernel_size
        conv_strides = requirements.conv_strides
        num_of_filters = random.choice(requirements.filters)
        pool_size = None
        pool_strides = None
        pool_type = None
        if is_image_has_permissible_size(image_size, 2):
            img_size = [output_dimension(image_size[i], requirements.pool_size[i], requirements.pool_strides[i]) for i
                        in
                        range(len(image_size))]
            if is_image_has_permissible_size(image_size, 2):
                img_size = [
                    floor(output_dimension(img_size[i], requirements.pool_size[i], requirements.pool_strides[i]))
                    for i in range(len(img_size))]
                if is_image_has_permissible_size(img_size, 2):
                    pool_size = requirements.pool_size
                    pool_strides = requirements.pool_strides
                    pool_type = random.choice(requirements.pool_types)
        else:
            return
        layer_params = {'layer_type': conv_node_type, 'activation': activation, 'kernel_size': kernel_size,
                        'conv_strides': conv_strides, 'num_of_filters': num_of_filters, 'pool_size': pool_size,
                        'pool_strides': pool_strides, 'pool_type': pool_type}
    else:
        layer_params = get_layer_params('conv2d', requirements)
    new_node = node_func(content={'name': f'{layer_params["layer_type"]}',
                                  'params': layer_params}, nodes_from=None)
    if random.uniform(0, 1) > BATCH_NORM_PROB:
        _add_batch_norm2node(new_node, requirements)
    return new_node, img_size


def create_secondary_node(node_func: Callable, requirements=None):
    new_node = random.choice(requirements.cnn_secondary)
    layer_params = get_layer_params(new_node, requirements)
    new_node = node_func(content={'name': layer_params['layer_type'], 'params': layer_params},
                         nodes_from=None)
    if random.uniform(0, 1) > BATCH_NORM_PROB:
        _add_batch_norm2node(new_node, requirements)
    return new_node


def create_primary_nn_node(node_func: Callable, requirements=None):
    new_node = random.choice(requirements.primary)
    layer_params = get_layer_params(new_node, requirements)
    new_node = node_func(content={'name': layer_params['layer_type'], 'params': layer_params},
                         nodes_from=None)
    if random.uniform(0, 1) > BATCH_NORM_PROB:
        _add_batch_norm2node(new_node, requirements)
    return new_node


def create_dropout_node(node_func: Callable, requirements=None):
    new_node = 'dropout'
    layer_params = get_layer_params(new_node, requirements)
    new_node = node_func(content={'name': layer_params['layer_type'],
                                  'params': layer_params}, nodes_from=None)
    return new_node


def get_layer_params(layer_type: str, requirements=None):
    if requirements is None:
        return DEFAULT_NODES_PARAMS[layer_type]
    else:
        return _get_random_layer_params(layer_type, requirements)


def _get_random_layer_params(layer_type: str, requirements):
    layer_params = None
    if layer_type == 'serial_connection':
        layer_params = {'layer_type': layer_type}
    elif layer_type == 'dropout':
        drop_value = random.randint(1, (requirements.max_drop_size * 10)) / 10
        layer_params = {'layer_type': layer_type, 'drop': drop_value}
    elif layer_type == 'batch_normalization':
        momentum = random.uniform(0, 1)
        epsilon = random.uniform(0, 1)
        layer_params = {'layer_type': layer_type, 'momentum': momentum, 'epsilon': epsilon}
    elif layer_type == 'dense':
        activation = random.choice(requirements.activation_types).value
        neurons = random.randint(requirements.min_num_of_neurons, requirements.max_num_of_neurons)
        layer_params = {'layer_type': layer_type, 'neurons': neurons, 'activation': activation}
    return layer_params


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


def generate_static_graph(graph_class: 'NNGraph', node_func: Callable, node_list: List,
                          has_skip_connections: bool = False, skip_connections_id: List[int] = None,
                          shortcuts_len: int = None) -> 'NNGraph':
    """
    Method for initial graph generation from defined nodes list.

    :param graph_class: Initial graph class
    :param node_func: Node class
    :param node_list: list of nodes to graph generation
    :param has_skip_connections: is graph has skip connections. If True, skip connections will be
    added to graph after generation
    :param skip_connections_id: indices of nodes where skip connection starts
    :param shortcuts_len: len of skip connection's shortcut
    :return: graph:
    """

    def _add_node_to_graph(node_type: str, node_params: dict = DEFAULT_NODES_PARAMS, parent_node=None):
        parent = None if not parent_node else [parent_node]
        # is_conv_node = node_type.startswith('conv')
        new_node = node_func(nodes_from=parent, content={'name': node_type,
                                                         'params': node_params[node_type]})
        graph.add_node(new_node)
        return new_node

    def _add_skip_connections(nodes_id: List[int], shortcuts_length: int = 2):
        for current_node in nodes_id:
            is_first_conv = current_node < graph.cnn_depth
            is_second_conv = current_node + shortcuts_length < graph.cnn_depth
            if is_first_conv == is_second_conv and (current_node + shortcuts_length) < len(graph.nodes):
                graph.nodes[current_node + shortcuts_length].nodes_from.append(graph.nodes[current_node])
            else:
                print('Wrong connection. Connection dropped.')

    graph = graph_class()
    created_node = None
    for node in node_list:
        created_node = _add_node_to_graph(node_type=node, parent_node=created_node)
    if has_skip_connections:
        _add_skip_connections(nodes_id=skip_connections_id, shortcuts_length=shortcuts_len)

    return graph


# TODO optimize skip connections for both cases
def add_skip_connections(graph: 'NNGraph'):
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


def random_conv_graph_generation(graph_class: Callable, node_func: Callable, requirements) -> 'NNGraph':
    """
    Method for random graph_class generation with given requirements

    :param graph_class: type of graph_class to generate
    :param node_func: type of generated node
    :param requirements: list of requirements to nodes
    """

    def _random_cnn(max_num_of_conv: int = None,
                    min_num_of_conv: int = None, image_size: List[float] = None, parent_nodes: Any = None):
        """
        Method for generation convolutional part of the graph_class

        :param max_num_of_conv: max number of nodes in conv part of the graph_class
        :param min_num_of_conv: min number of nodes in conv part of the graph_class
        :param image_size: input image size
        :param parent_nodes: node from conv part will grow
        """
        max_num_of_conv = max_num_of_conv if max_num_of_conv is not None else requirements.max_num_of_conv_layers
        min_num_of_conv = min_num_of_conv if min_num_of_conv is not None else requirements.min_num_of_conv_layers
        num_of_conv = random.randint(min_num_of_conv, max_num_of_conv)

        if image_size is None:
            image_size = requirements.image_size
        else:
            image_size = image_size

        def _growth_conv_node(node_parent: Any = None, depth: int = None, img_size: List[float] = None):
            depth = 0 if depth is None else depth
            # TODO add nodes from to node creation func;
            #  also add skip connection creation which can be turned on and off
            nodes_from = None if node_parent is None else [node_parent]
            new_node, img_size = create_conv2d_node(node_func=node_func, requirements=requirements,
                                                    image_size=img_size)
            new_node.nodes_from = nodes_from
            depth += 1
            if depth > num_of_conv:
                return
            graph.add_node(new_node)
            nodes_from = new_node
            if depth < num_of_conv - 1:
                new_node = create_secondary_node(node_func=node_func,
                                                 requirements=requirements)
                new_node.nodes_from = [nodes_from]
                depth += 1
                if depth > num_of_conv:
                    return
                graph.add_node(new_node)
            else:
                is_dropout = random.randint(0, 1)
                if is_dropout:
                    new_node = create_dropout_node(node_func=node_func, requirements=requirements)
                    depth += 1
                    if depth > num_of_conv:
                        return
                    graph.add_node(new_node)
                    new_node.nodes_from = [nodes_from]
            nodes_from = new_node
            if depth < num_of_conv:
                _growth_conv_node(node_parent=nodes_from, img_size=img_size, depth=depth)

        parent_nodes = parent_nodes if parent_nodes else None
        _growth_conv_node(node_parent=parent_nodes, img_size=image_size)
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
            new_node = create_primary_nn_node(node_func=node_func, requirements=requirements)
            new_node.nodes_from = nodes_from
            depth += 1
            if depth > max_depth:
                return
            graph.add_node(new_node)
            nodes_from = new_node
            if depth < max_depth - 1:
                new_node = create_secondary_node(node_func=node_func, requirements=requirements)
                new_node.nodes_from = [nodes_from]
                depth += 1
                if depth > max_depth:
                    return
                graph.add_node(new_node)
                nodes_from = new_node
            else:
                is_additional_dense = random.randint(0, 1)
                if is_additional_dense:
                    new_node = create_primary_nn_node(node_func=node_func, requirements=requirements)
                    new_node.nodes_from = [nodes_from]
                    depth += 1
                    if depth > max_depth:
                        return
                    graph.add_node(new_node)
                    nodes_from = new_node
            if depth < max_depth:
                _growth_nn(node_parent=nodes_from, depth=depth)

        root_node = root_node if root_node else None
        _growth_nn(node_parent=root_node)

    graph = graph_class()
    _random_cnn()
    _random_nn(root_node=graph.nodes[-1])
    # add_skip_connections(graph_class)

    if not hasattr(graph, 'parent_operators'):
        setattr(graph, 'parent_operators', [])
    return graph


def one_side_parameters_correction(input_dimension: float, kernel_size: int, stride: int) -> \
        Tuple[int, int]:
    output_dim = output_dimension(input_dimension, kernel_size, stride)
    if not float(output_dim).is_integer():
        if kernel_size + 1 < input_dimension:
            kernel_size = kernel_size + 1
        while kernel_size > input_dimension:
            kernel_size = kernel_size - 1
        while not float(
                output_dimension(input_dimension, kernel_size, stride)).is_integer() or stride > input_dimension:
            stride = stride - 1
    return kernel_size, stride


def permissible_kernel_parameters_correct(image_size: List[float], kernel_size: Tuple[int, int],
                                          strides: Tuple[int, int],
                                          pooling: bool) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    is_strides_permissible = all([strides[i] < kernel_size[i] for i in range(len(strides))])
    is_kernel_size_permissible = all([kernel_size[i] < image_size[i] for i in range(len(strides))])
    if not is_strides_permissible:
        if pooling:
            strides = (2, 2)
        else:
            strides = (1, 1)
    if not is_kernel_size_permissible:
        kernel_size = (2, 2)
    return kernel_size, strides


def kernel_parameters_correction(input_image_size: List[float], kernel_size: Tuple[int, int],
                                 strides: Tuple[int, int], pooling: bool) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    kernel_size, strides = permissible_kernel_parameters_correct(input_image_size, kernel_size, strides, pooling)
    if len(set(input_image_size)) == 1:
        new_kernel_size, new_strides = one_side_parameters_correction(input_image_size[0], kernel_size[0],
                                                                      strides[0])
        if new_kernel_size != kernel_size:
            kernel_size = tuple([new_kernel_size for _ in range(len(input_image_size))])
        if new_strides != strides:
            strides = tuple([new_strides for _ in range(len(input_image_size))])
    else:
        new_kernel_size = []
        new_strides = []
        for i in range(len(input_image_size)):
            params = one_side_parameters_correction(input_image_size[i], kernel_size[i], strides[i])
            new_kernel_size.append(params[0])
            new_strides.append(params[1])
        kernel_size = tuple(new_kernel_size) if kernel_size != tuple(new_kernel_size) else kernel_size
        strides = tuple(new_strides) if strides != tuple(new_strides) else strides
    return kernel_size, strides
