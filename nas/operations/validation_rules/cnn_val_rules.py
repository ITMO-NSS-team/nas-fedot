from typing import Optional

import keras.utils.layer_utils
from golem.core.dag.verification_rules import ERROR_PREFIX

from nas.graph.cnn_graph import NasGraph
from nas.model.tensorflow.base_model import KerasModelMaker
from nas.model.utils.model_structure import ModelStructure


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


class ConvNetChecker:
    params_limit: int = 5e7
    error_message: Optional[str] = None
    model: Optional = None

    @classmethod
    def check_cnn(cls, graph: NasGraph):
        cls.error_message = None
        cls.model = None
        rules_list = [rule for rule in dir(cls) if callable(getattr(cls, rule)) and rule.startswith('r_')]
        for rule in rules_list:
            getattr(cls, rule)(graph)
            if cls.error_message:
                cls.clear_memory()
                raise ValueError(cls.error_message)
        cls.clear_memory()
        return True

    @staticmethod
    def r_is_buildable(graph: NasGraph):
        input_shape = [12, 12, 3]
        num_classes = 3
        try:
            ConvNetChecker.model = KerasModelMaker(input_shape, graph, ModelStructure, num_classes).build()
        except Exception as ex:
            ConvNetChecker.error_message = f'Exception {ex} occurred. Model cannot be built. {ERROR_PREFIX}.'

    @staticmethod
    def r_params_count(*args, **kwargs):
        params = keras.utils.layer_utils.count_params(ConvNetChecker.model.trainable_variables)
        if params > ConvNetChecker.params_limit:
            ConvNetChecker.error_message = f'{ERROR_PREFIX}. Graph has too many trainable params: {params}'

    @staticmethod
    def r_has_correct_struct(graph: NasGraph):
        graph_struct = graph.graph_struct
        node = graph_struct[0]
        if 'conv' not in node.content['name']:
            ConvNetChecker.error_message = f'{ERROR_PREFIX}. CNN should starts with Conv layer.'

    @classmethod
    def clear_memory(cls):
        if cls.model:
            del cls.model
            cls.model = None
        if cls.error_message:
            cls.error_message = None
