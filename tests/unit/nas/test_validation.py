import os

from nas.graph_cnn_mutations import has_no_flatten_skip, graph_has_several_starts, flatten_check
from nas.composer.graph_gp_cnn_composer import NNGraph, NNNode
from nas.graph_cnn_gp_operators import generate_initial_graph
from nas.var import TESTING_ROOT


def generate_graph():
    nodes_list = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'flatten', 'dense', 'dropout', 'dense', 'dense']
    graph = generate_initial_graph(NNGraph, NNNode, nodes_list)
    return graph


# TODO move to is valid graph.
#  Generate several random graphs and check them on validation rules for checking is the graph generation func correct
def test_validation():
    rules_list = [has_no_flatten_skip, graph_has_several_starts, flatten_check]
    graph = generate_graph()
    successful_generation = False
    for rule in rules_list:
        successful_generation = rule(graph)
        if not successful_generation:
            break
    assert successful_generation


def test_has_no_flatten_skip():
    graph = NNGraph.load(os.path.join(TESTING_ROOT, 'graph_with_flatten_skip.json'))
    successful_check = False
    try:
        has_no_flatten_skip(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_graph_has_wrong_structure():
    raise NotImplementedError


def test_flatten_check():
    graph = NNGraph.load(os.path.join(TESTING_ROOT, 'graph_few_flatten.json'))
    successful_check = False
    try:
        flatten_check(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_cnn_mutation():
    raise NotImplementedError
