import os

import keras.utils.layer_utils
import tensorflow
from golem.core.dag.verification_rules import ERROR_PREFIX
from typing import Tuple, Optional

from nas.graph.cnn.cnn_graph import NasGraph
from nas.graph.node.params_counter import get_shape, add_shortcut_and_check
from nas.model.nn.tf_model import ModelMaker
from nas.model.utils import converter


def parameters_check(graph: NasGraph):
    try:
        total_params = graph.get_trainable_params()
        if total_params > 8e7:
            raise ValueError(f'{ERROR_PREFIX} Neural network has too many trainable parameters')
    except Exception as ex:
        raise ValueError(f'{ERROR_PREFIX} Cannot count trainable parameters due exception {ex}.')


def unique_node_types(graph: NasGraph):
    if len(set(map(str, graph.nodes))) < 3:
        raise ValueError(f'{ERROR_PREFIX} CNN should has at least 3 unique layer types')


def flatten_count(graph: NasGraph):
    if len(graph.cnn_depth) != 1:
        raise ValueError(f'{ERROR_PREFIX} wrong number of flatten layer in CNN')


def has_no_flatten_skip(graph: NasGraph):
    for node in graph.free_nodes:
        if node.content['name'] == 'flatten':
            return True
    raise ValueError(f'{ERROR_PREFIX} Graph has wrong skip connections')


def graph_has_several_starts(graph: NasGraph):
    cnt = 0
    for node in graph.graph_struct:
        if not node.nodes_from:
            cnt += 1
        if cnt > 1:
            raise ValueError(f'{ERROR_PREFIX} Graph has more than one start node')
    return True


def graph_has_wrong_structure(graph: NasGraph):
    nodes_after_flatten = [str(node) for node in graph.graph_struct[graph.cnn_depth[0]:] if 'conv' in str(node)]
    if len(nodes_after_flatten) != 0:
        raise ValueError(f'{ERROR_PREFIX} Graph has wrong structure')
    return True


def graph_has_wrong_structure_tmp(graph: NasGraph):
    nodes_before_flatten = [str(node) for node in graph.graph_struct[:graph.cnn_depth[0]] if 'dense' in str(node)]
    if len(nodes_before_flatten) != 0:
        raise ValueError(f'{ERROR_PREFIX} Graph has wrong structure')
    return True


def graph_has_several_root_nodes(graph: NasGraph):
    if len(graph.root_node) > 1:
        raise ValueError(f'{ERROR_PREFIX} Graph has several root nodes')


def dimensions_check(graph: NasGraph) -> int:
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


def is_architecture_is_correct(graph: NasGraph):
    # with tensorflow.device('/cpu:0'):
    input_shape = graph.input_shape
    num_classes = 2  # any int number
    try:
        model = ModelMaker(input_shape, graph, converter.GraphStruct, num_classes).build()
        params = keras.utils.layer_utils.count_params(model.trainable_variables)
        if params > 1.5e5:
            raise ValueError
        graph.unfit()
    except ValueError as ex:
        graph.unfit()
        raise ValueError(f'{ERROR_PREFIX}. Model building failed with exception {ex}. Model is unacceptable')
    return True


def tmp_dense_in_conv(graph: NasGraph):
    for node in graph.graph_struct:
        if 'conv' in node.content['name']:
            parent_nodes = [n.content['name'] for n in node.nodes_from]
            if 'dense' in parent_nodes:
                raise ValueError(f'{ERROR_PREFIX} dense layer in conv part!')
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
        # with tensorflow.device('/cpu:0'):
        for rule in rules_list:
            getattr(cls, rule)(graph)
            if cls.error_message:
                cls.clear_memory()
                raise ValueError(cls.error_message)
        cls.clear_memory()
        return True

    @staticmethod
    def r_is_buildable(graph: NasGraph):
        input_shape = [24, 24, 3]
        num_classes = 3
        try:
            ConvNetChecker.model = ModelMaker(input_shape, graph, converter.GraphStruct, num_classes).build()
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


if __name__ == '__main__':
    graph = NasGraph.load('/home/staeros/graph.json')
    lst = []
    for i in range(50):
        lst.append(ModelMaker(graph.input_shape, graph, converter.GraphStruct, 3).build())
    print(1)
