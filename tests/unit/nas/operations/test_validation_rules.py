import pathlib

from nas.operations.validation_rules.cnn_val_rules import *
from nas.utils.utils import set_root, project_root

tests_path = pathlib.Path(project_root(), 'tests', 'unit', 'test_data')
set_root(tests_path)


def test_flatten_count_no_flatten():
    graph = NNGraph.load('./graph_without_flatten_node.json')
    successful_check = False
    try:
        flatten_count(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_flatten_count_several_flattens():
    graph = NNGraph.load('./graph_several_flatten_and_starts.json')
    successful_check = False
    try:
        flatten_count(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_unique_node_types():
    graph = NNGraph.load('./graph_no_conv.json')
    successful_check = False
    try:
        unique_node_types(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_no_flatten_skip():
    graph = NNGraph.load('./graph_with_flatten_skip.json')
    successful_check = False
    try:
        has_no_flatten_skip(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_several_starts():
    graph = NNGraph.load('./graph_several_flatten_and_starts.json')
    successful_check = False
    try:
        graph_has_several_starts(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_wrong_cnn_struct():
    graph = NNGraph.load('./graph_conv_after_flatten.json')
    successful_check = False
    try:
        graph_has_wrong_structure(graph)
    except ValueError:
        successful_check = True
    assert successful_check
