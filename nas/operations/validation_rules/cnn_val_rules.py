from fedot.core.dag.verification_rules import ERROR_PREFIX

from nas.graph.cnn.cnn_graph import NNGraph
from nas.graph.node.params_counter import get_shape, add_shortcut_and_check


def parameters_check(graph: NNGraph):
    try:
        total_params = graph.get_trainable_params()
        if total_params > 8e7:
            raise ValueError(f'{ERROR_PREFIX} Neural network has too many trainable parameters')
    except Exception as ex:
        raise ValueError(f'{ERROR_PREFIX} Cannot count trainable parameters due exception {ex}.')


def unique_node_types(graph: NNGraph):
    if len(set(map(str, graph.nodes))) < 3:
        raise ValueError(f'{ERROR_PREFIX} CNN should has at least 3 unique layer types')


def flatten_count(graph: NNGraph):
    if len(graph.cnn_depth) != 1:
        raise ValueError(f'{ERROR_PREFIX} wrong number of flatten layer in CNN')


def has_no_flatten_skip(graph: NNGraph):
    for node in graph.free_nodes:
        if node.content['name'] == 'flatten':
            return True
    raise ValueError(f'{ERROR_PREFIX} Graph has wrong skip connections')


def graph_has_several_starts(graph: NNGraph):
    cnt = 0
    for node in graph.graph_struct:
        if not node.nodes_from:
            cnt += 1
        if cnt > 1:
            raise ValueError(f'{ERROR_PREFIX} Graph has more than one start node')
    return True


def graph_has_wrong_structure(graph: NNGraph):
    nodes_after_flatten = [str(node) for node in graph.graph_struct[graph.cnn_depth[0]:] if 'conv' in str(node)]
    if len(nodes_after_flatten) != 0:
        raise ValueError(f'{ERROR_PREFIX} Graph has wrong structure')
    return True


def graph_has_wrong_structure_tmp(graph: NNGraph):
    nodes_before_flatten = [str(node) for node in graph.graph_struct[:graph.cnn_depth[0]] if 'dense' in str(node)]
    if len(nodes_before_flatten) != 0:
        raise ValueError(f'{ERROR_PREFIX} Graph has wrong structure')
    return True


def graph_has_several_root_nodes(graph: NNGraph):
    if len(graph.root_node) > 1:
        raise ValueError(f'{ERROR_PREFIX} Graph has several root nodes')


def dimensions_check(graph: NNGraph) -> int:
    # if not any(shape == output_shape for shape in main_shapes):
    #     Add conv 1x1 with different strides and compare again. If not equal, raise the error
    #     add_shortcut_and_check(shape, output_shape)
    input_shape = graph.input_shape
    dimension_cache = {}
    try:
        for node in graph.graph_struct:
            output_shape = get_shape(input_shape, node)
            dimension_cache[node] = output_shape
            input_shape = output_shape
            if len(node.nodes_from) > 1:
                # shape list where skip connection starts
                main_shapes = [dimension_cache.get(parent) for parent in node.nodes_from[1::]]
                for shape in main_shapes:
                    if not shape == output_shape:
                        new_shape = add_shortcut_and_check(shape, output_shape)
                        if not new_shape == output_shape:
                            raise ValueError(f'{ERROR_PREFIX} shapes {shape} and {output_shape} are not equal')
    except TypeError:
        raise ValueError('TEMPORAL ERROR MESSAGE. Type error occurred during validation. TEMPORAL ERROR MESSAGE')
    return True


def tmp_dense_in_conv(graph: NNGraph):
    for node in graph.graph_struct:
        if 'conv' in node.content['name']:
            parent_nodes = [n.content['name'] for n in node.nodes_from]
            if 'dense' in parent_nodes:
                raise ValueError(f'{ERROR_PREFIX} dense layer in conv part!')
    return True
