import os

from nas.graph_cnn_mutations import has_no_flatten_skip, graph_has_several_starts, flatten_check, \
    graph_has_wrong_structure
from nas.composer.cnn_graph_node import CNNNode
from nas.composer.cnn_graph import CNNGraph
from nas.composer.cnn_graph_operator import generate_initial_graph
from nas.utils.var import TESTING_ROOT


def generate_graph():
    nodes_list = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'flatten', 'dense', 'dropout', 'dense', 'dense']
    graph = generate_initial_graph(CNNGraph, CNNNode, nodes_list)
    return graph


def test_has_no_flatten_skip():
    graph = CNNGraph.load(os.path.join(TESTING_ROOT, 'graph_with_flatten_skip.json'))
    successful_check = False
    try:
        has_no_flatten_skip(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_graph_has_several_starts():
    graph = CNNGraph.load(os.path.join(TESTING_ROOT, 'several_starts_graph.json'))
    successful_check = False
    try:
        graph_has_several_starts(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_graph_has_wrong_structure():
    graph = CNNGraph.load(os.path.join(TESTING_ROOT, 'no_conv_graph.json'))
    successful_check = False
    try:
        graph_has_wrong_structure(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_flatten_check():
    graph = CNNGraph.load(os.path.join(TESTING_ROOT, 'graph_few_flatten.json'))
    successful_check = False
    try:
        flatten_check(graph)
    except ValueError:
        successful_check = True
    assert successful_check
