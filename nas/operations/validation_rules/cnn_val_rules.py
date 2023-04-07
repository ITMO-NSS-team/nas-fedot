from typing import Union, List

from golem.core.dag.graph_node import GraphNode
from golem.core.dag.verification_rules import ERROR_PREFIX

from nas.graph.BaseGraph import NasGraph
from nas.graph.node.nas_graph_node import NasNode


def model_has_several_starts(graph: NasGraph):
    if hasattr(graph.root_node, '__iter__'):
        raise ValueError(f'{ERROR_PREFIX} model must not has more than 1 start.')


def model_has_wrong_number_of_flatten_layers(graph: NasGraph):
    flatten_count = 0
    for node in graph.graph_struct[::-1]:
        if node.content['name'] == 'flatten':
            flatten_count += 1
    if flatten_count != 1:
        raise ValueError(f'{ERROR_PREFIX} model has wrong number of flatten layers.')


def conv_net_check_structure(graph: NasGraph):
    prohibited_node_types = ['average_pool2d', 'max_pool2d', 'conv2d']
    for node in graph.graph_struct[::-1]:
        node_name = 'conv2d' if 'conv' in node.content['name'] else node.content['name']
        if node_name == 'flatten':
            return True
        elif node_name in prohibited_node_types:
            raise ValueError(f'{ERROR_PREFIX} node {node} can not be after flatten layer.')


def model_has_no_conv_layers(graph: NasGraph):
    was_flatten = False
    was_conv = False
    for node in graph.graph_struct:
        node_name = 'conv2d' if 'conv' in node.content['name'] else node.content['name']
        if node_name == 'conv2d':
            was_conv = True
        elif node_name == 'flatten':
            was_flatten = True
    if not was_conv and was_flatten:
        raise ValueError(f'{ERROR_PREFIX} model has no convolutional layers.')


# TODO change checker into recursive(?) output_shape calculation.
def model_has_dimensional_conflict(graph: NasGraph):
    for node in graph.graph_struct:
        if len(node.nodes_from) > 1:
            not_conv_node = ['conv' not in n.content['name'] for n in node.nodes_from]
            if not_conv_node:
                raise ValueError(f'{ERROR_PREFIX} model has has different layer output shapes which may lead to error.')
            has_different_shapes = node.nodes_from[-1].parameters['neurons'] != node.parameters['neurons']
            has_fmap_decrease = node.nodes_from[-1].parameters['conv_strides'] != [1, 1]
            if any([has_fmap_decrease, has_different_shapes]):
                raise ValueError(f'{ERROR_PREFIX} model has has different layer output shapes which may lead to error.')
    return True


def check_dimensions(graph: NasGraph):
    _shape=[]
    def _shape_calc_recursive(node: Union[NasNode, GraphNode]):
        node_type = node.name
        neurons_num = node.parameters.get('neurons')
        kernel_size = node.parameters.get('kernel_size')

        # padding = 1 in fact padding SAME
        if not node.nodes_from:
            input_shapes = [_shape_calc_recursive(ancestor_node) if node.nodes_from else
                            _shape for ancestor_node in node.nodes_from]

        # conv2d case
        output_shape = None


    return _shape_calc_recursive(graph.root_node)