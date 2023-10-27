import torch
from golem.core.dag.verification_rules import ERROR_PREFIX

from nas.graph.BaseGraph import NasGraph
from nas.model.model_interface import NeuralSearchModel
from nas.model.pytorch.base_model import NASTorchModel


def model_has_several_roots(graph: NasGraph):
    if hasattr(graph.root_node, '__iter__'):
        raise ValueError(f'{ERROR_PREFIX} model must not has more than 1 root node.')


def model_has_several_starts(graph: NasGraph):
    starts = 0
    for node in graph.nodes:
        n = 0 if node.nodes_from else 1
        starts += n
        if starts > 1:
            raise ValueError(f'{ERROR_PREFIX} model must not has more than 1 start.')


def model_has_wrong_number_of_flatten_layers(graph: NasGraph):
    flatten_count = 0
    for node in graph.nodes:
        if node.content['name'] == 'flatten':
            flatten_count += 1
    if flatten_count != 1:
        raise ValueError(f'{ERROR_PREFIX} model has wrong number of flatten layers.')


def conv_net_check_structure(graph: NasGraph):
    prohibited_node_types = ['average_pool2d', 'max_pool2d', 'conv2d']
    for node in graph.nodes:
        node_name = 'conv2d' if 'conv' in node.content['name'] else node.content['name']
        if node_name == 'flatten':
            return True
        elif node_name in prohibited_node_types:
            raise ValueError(f'{ERROR_PREFIX} node {node} can not be after flatten layer.')


def model_has_no_conv_layers(graph: NasGraph):
    was_flatten = False
    was_conv = False
    for node in graph.nodes:
        node_name = 'conv2d' if 'conv' in node.content['name'] else node.content['name']
        if node_name == 'conv2d':
            was_conv = True
        elif node_name == 'flatten':
            was_flatten = True
    if not was_conv and was_flatten:
        raise ValueError(f'{ERROR_PREFIX} model has no convolutional layers.')


def model_has_dim_mismatch(graph: NasGraph):
    try:
        with torch.no_grad():
            m = NeuralSearchModel(NASTorchModel).compile_model(graph, (24, 24, 3), 3).model
            m.to('cpu')
            m.forward(torch.rand((3, 3, 224, 224)))
    except RuntimeError:
        raise ValueError(f'{ERROR_PREFIX} graph has dimension conflict.')
    return True

