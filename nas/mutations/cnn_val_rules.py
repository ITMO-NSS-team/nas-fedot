from fedot.core.dag.validation_rules import ERROR_PREFIX

from nas.composer.cnn.cnn_graph import CNNGraph


def flatten_check(graph: CNNGraph):
    cnt = 0
    for node in graph.nodes:
        if node.content['name'] == 'flatten':
            cnt += 1
            if cnt > 1:
                raise ValueError(f'{ERROR_PREFIX} Graph should have only one flatten layer')
    return True


def has_no_flatten_skip(graph: CNNGraph):
    for node in graph.free_nodes:
        if node.content['name'] == 'flatten':
            return True
    raise ValueError(f'{ERROR_PREFIX} Graph has wrong skip connections')


def graph_has_several_starts(graph: CNNGraph):
    cnt = 0
    for node in graph.graph_struct:
        if not node.nodes_from:
            cnt += 1
        if cnt > 1:
            raise ValueError(f'{ERROR_PREFIX} Graph has more than one start node')
    return True


def graph_has_wrong_structure(graph: CNNGraph):
    for node in graph.graph_struct[graph.cnn_depth:]:
        if node.content['name'] == 'conv2d':
            raise ValueError(f'{ERROR_PREFIX} Graph has wrong structure')
    return True
