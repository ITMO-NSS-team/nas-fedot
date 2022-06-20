import os

from nas.graph_cnn_mutations import has_no_flatten_skip, graph_has_several_starts, flatten_check, \
    graph_has_wrong_structure
from nas.composer.gp_cnn_composer import GPNNComposerRequirements
from nas.composer.cnn_graph_node import CNNNode
from nas.composer.cnn_graph import CNNGraph
from nas.composer.cnn_graph_operator import generate_initial_graph
from nas.utils.var import TESTING_ROOT
from nas.cnn_builder import NASDirector, CNNBuilder

requirements = GPNNComposerRequirements(input_shape=[120, 120, 3], pop_size=1,
                                        num_of_generations=1, max_num_of_conv_layers=4,
                                        max_nn_depth=3, primary=['conv2d'], secondary=['dense'],
                                        batch_size=4, epochs=1,
                                        has_skip_connection=True, skip_connections_id=[0, 2, 5], shortcuts_len=2)


def generate_graph():
    director = NASDirector()
    director.set_builder(CNNBuilder(requirements=requirements))
    graph = director.create_nas_graph()
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
