from collections import Counter

from fedot.core.dag.verification_rules import ERROR_PREFIX

from nas.graph.cnn.cnn_graph import NNGraph


def unique_node_types(graph: NNGraph):
    if len(set(map(str, graph.nodes))) < 3:
        raise ValueError(f'{ERROR_PREFIX} CNN should has at least 3 unique layer types')


def flatten_count(graph: NNGraph):
    nodes_counter = Counter(list(map(str, graph.nodes)))
    if nodes_counter['flatten'] != 1:
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
    nodes_after_flatten = [str(node) for node in graph.graph_struct[graph.cnn_depth:]]
    if 'conv2d' in nodes_after_flatten:
        raise ValueError(f'{ERROR_PREFIX} Graph has wrong structure')
    return True
