from random import choice, randint
from math import floor
from typing import (Tuple, List, Any, Callable)
from nas.layer import LayerTypesIdsEnum, LayerParams
from nas.keras_eval import generate_structure


def output_dimension(input_dimension: float, kernel_size: int, stride: int) -> float:
    output_dim = ((input_dimension - kernel_size) / stride) + 1
    return output_dim


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
                                 strides: Tuple[int, int], pooling: bool) -> Tuple[
    Tuple[int, int], Tuple[int, int]]:
    kernel_size, strides = permissible_kernel_parameters_correct(input_image_size, kernel_size, strides, pooling)
    if len(set(input_image_size)) == 1:
        new_kernel_size, new_strides = one_side_parameters_correction(input_image_size[0], kernel_size[0],
                                                                      strides[0])
        if new_kernel_size != kernel_size:
            kernel_size = tuple([new_kernel_size for i in range(len(input_image_size))])
        if new_strides != strides:
            strides = tuple([new_strides for i in range(len(input_image_size))])
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


def is_image_has_permissible_size(image_size, min_size: int = 2):
    return all([side_size >= min_size for side_size in image_size])


def random_cnn(node_func: Callable, requirements, graph: Any = None, max_num_of_conv: int = None,
               min_num_of_conv: int = None, image_size: List[float] = None, parent_nodes: Any = None) -> Any:
    max_num_of_conv = max_num_of_conv if not max_num_of_conv is None else requirements.max_num_of_conv_layers
    min_num_of_conv = min_num_of_conv if not min_num_of_conv is None else requirements.max_num_of_conv_layers
    num_of_conv = randint(min_num_of_conv, max_num_of_conv)
    graph.cnn_depth = num_of_conv

    if image_size is None:
        current_image_size = requirements.image_size
    else:
        current_image_size = image_size

    def one_cnn_branch_growth(node_parent: Any = None, img_size: List[float] = current_image_size, depth: int = None,
                              total_convs: int = num_of_conv):
        # Conv layer parameters
        depth = 0 if depth is None else depth
        nodes_from = None if not depth else [node_parent]
        conv_node_type = choice(requirements.conv_types)
        activation = choice(requirements.activation_types).value
        kernel_size = requirements.conv_kernel_size
        conv_strides = requirements.conv_strides
        num_of_filters = choice(requirements.filters)
        pool_size = None
        pool_strides = None
        pool_type = None
        if is_image_has_permissible_size(img_size, 2):
            img_size = [output_dimension(img_size[i], kernel_size[i], conv_strides[i]) for i in
                        range(len(kernel_size))]

            if is_image_has_permissible_size(img_size, 2):
                img_size = [floor(output_dimension(img_size[i], requirements.pool_size[i],
                                                   requirements.pool_strides[i])) for i in
                            range(len(img_size))]
                if is_image_has_permissible_size(img_size, 2):
                    pool_size = requirements.pool_size
                    pool_strides = requirements.pool_strides
                    pool_type = choice(requirements.pool_types)
        else:
            return

        # Add conv layers
        layer_params = LayerParams(layer_type=conv_node_type, activation=activation, kernel_size=kernel_size,
                                   conv_strides=conv_strides, num_of_filters=num_of_filters, pool_size=pool_size,
                                   pool_strides=pool_strides, pool_type=pool_type)
        new_conv_node = node_func(nodes_from=nodes_from,
                                  content={'name': layer_params.layer_type},
                                  layer_params=layer_params)
        graph.add_node(new_conv_node)
        nodes_from = [new_conv_node]
        if pool_size is None:
            return

        # Add secondary layers
        if depth < total_convs - 1:
            new_secondary_node_type = choice(requirements.cnn_secondary)
            layer_params = get_random_layer_params(new_secondary_node_type, requirements)
            new_secondary_node = node_func(nodes_from=nodes_from,
                                           content={'name': layer_params.layer_type},
                                           layer_params=layer_params)
            graph.add_node(new_secondary_node)
        else:
            add_dropout_layer = randint(0, 1)
            if add_dropout_layer:
                new_secondary_node_type = LayerTypesIdsEnum.dropout.value
                layer_params = get_random_layer_params(new_secondary_node_type, requirements)
                new_secondary_node = node_func(nodes_from=nodes_from,
                                               content={'name': layer_params.layer_type},
                                               layer_params=layer_params)
                graph.add_node(new_secondary_node)
            else:
                new_secondary_node = None

        nodes_from = new_secondary_node if not new_secondary_node is None else nodes_from

        if depth < total_convs:
            one_cnn_branch_growth(node_parent=nodes_from, img_size=img_size, depth=depth+2)

    parent_nodes = parent_nodes if parent_nodes else None
    one_cnn_branch_growth(node_parent=parent_nodes, img_size=current_image_size, total_convs=num_of_conv)
    return graph.nodes[-1]


def random_nn_branch(node_func: Callable, requirements, graph: Any = None, max_depth=None, start_height: int = None,
                     node_parent=None) -> Any:
    max_depth = max_depth if not max_depth is None else requirements.max_depth

    def nn_branch_growth(node_parent: Any = None, offspring_size: int = None, depth: int = None,
                         total_nodes: int = max_depth):
        nodes_from = [node_parent] if node_parent else None
        depth = 0 if depth is None else depth

        new_node_type = choice(requirements.primary)
        layer_params = get_random_layer_params(new_node_type, requirements)
        new_node = node_func(nodes_from=nodes_from,
                             content={'name': layer_params.layer_type},
                             layer_params=layer_params)
        if graph:
            graph.add_node(new_node)
        nodes_from = [new_node]

        # Add secondary layers
        if depth < total_nodes - 1:
            new_secondary_node_type = choice(requirements.secondary)
            layer_params = get_random_layer_params(new_secondary_node_type, requirements)
            new_secondary_node = node_func(nodes_from=nodes_from,
                                           content={'name': layer_params.layer_type},
                                           layer_params=layer_params)
            graph.add_node(new_secondary_node)
        else:
            add_dense_layer = randint(0, 1)
            if add_dense_layer:
                new_secondary_node_type = LayerTypesIdsEnum.dense.value
                layer_params = get_random_layer_params(new_secondary_node_type, requirements)
                new_secondary_node = node_func(nodes_from=nodes_from,
                                               content={'name': layer_params.layer_type},
                                               layer_params=layer_params)
                graph.add_node(new_secondary_node)
            else:
                new_secondary_node = None

        nodes_from = new_secondary_node if not new_secondary_node is None else nodes_from

        if depth < total_nodes:
            nn_branch_growth(node_parent=nodes_from, depth=depth + 2)


    # def branch_growth(node_parent: Any = None, offspring_size: int = None, height: int = None):
    #
    #     height = 0 if height is None else height
    #     is_max_depth_exceeded = height >= max_depth - 1
    #     is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
    #     primary = is_max_depth_exceeded or is_primary_node_selected
    #     if primary:
    #         node_type = choice(requirements.primary)
    #     else:
    #         node_type = choice(requirements.secondary)
    #         offspring_size = offspring_size if not offspring_size is None else randint(requirements.min_arity,
    #                                                                                    requirements.max_arity)
    #     layer_params = get_random_layer_params(node_type, requirements)
    #
    #     if primary:
    #         new_node = secondary_node_func(layer_params=layer_params)
    #     else:
    #         new_node = secondary_node_func(layer_params=layer_params)
    #         for _ in range(offspring_size):
    #             branch_growth(node_parent=new_node, height=height + 1)
    #     if graph:
    #         graph.add_node(new_node)
    #     if node_parent:
    #         node_parent.nodes_from.append(new_node)

    node_parent = node_parent if node_parent else None
    # branch_growth(primary_node_parent=node_parent)
    nn_branch_growth(node_parent=node_parent, depth=start_height)


def random_cnn_graph(graph_class: Any, node_func: Callable, requirements) -> Any:
    graph = graph_class()
    # left (cnn part) branch of tree generation
    node_parent = random_cnn(graph=graph, node_func=node_func, requirements=requirements)
    # Right (fully connected nn) branch of tree generation
    random_nn_branch(graph=graph, node_func=node_func, requirements=requirements, start_height=0,
                     node_parent=node_parent)
    if not hasattr(graph, 'parent_operators'):
        setattr(graph, 'parent_operators', [])
    return graph


def get_random_layer_params(type: str, requirements) -> LayerParams:
    layer_params = None
    if type == LayerTypesIdsEnum.serial_connection.value:
        layer_params = LayerParams(layer_type=type)
    elif type == LayerTypesIdsEnum.dropout.value:
        drop = randint(1, (requirements.max_drop_size * 10)) / 10
        layer_params = LayerParams(layer_type=type, drop=drop)
    if type == LayerTypesIdsEnum.dense.value:
        activation = choice(requirements.activation_types).value
        neurons = randint(requirements.min_num_of_neurons, requirements.max_num_of_neurons)
        layer_params = LayerParams(layer_type=type, neurons=neurons, activation=activation)
    return layer_params


def check_cnn_branch(root_node: Any, image_size: List[int]):
    image_size = branch_output_shape(root_node, image_size)
    return is_image_has_permissible_size(image_size, 2)


def branch_output_shape(root: Any, image_size: List[float], subtree_to_delete: Any = None):
    structure = generate_structure(root)
    if subtree_to_delete:
        nodes = subtree_to_delete.ordered_subnodes_hierarchy
        structure = [node for node in structure if not node in nodes]
    for node in structure:
        if node.layer_params.layer_type == LayerTypesIdsEnum.conv2d:
            image_size = conv_output_shape(node, image_size)

    return image_size


def conv_output_shape(node, image_size):
    image_size = [
        output_dimension(image_size[i], node.layer_params.kernel_size[i], node.layer_params.conv_strides[i]) for
        i in range(len(image_size))]
    if node.layer_params.pool_size:
        image_size = [
            output_dimension(image_size[i], node.layer_params.pool_size[i], node.layer_params.pool_strides[i])
            for i in range(len(image_size))]
        image_size = [floor(side_size) for side_size in image_size]
    return image_size
