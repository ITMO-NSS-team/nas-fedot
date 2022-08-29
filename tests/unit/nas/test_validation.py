import os

from nas.operations.evaluation.mutations import flatten_check, has_no_flatten_skip, graph_has_several_starts, \
    graph_has_wrong_structure
from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.graph.nn_graph.cnn.cnn_graph import NNGraph
from nas.utils.var import tests_root
from nas.graph.nn_graph.cnn import NNGraphBuilder, CNNBuilder

requirements = NNComposerRequirements(input_shape=[120, 120, 3], pop_size=1,
                                      num_of_generations=1, max_num_of_conv_layers=4,
                                      max_nn_depth=3, primary=['conv2d'], secondary=['dense'],
                                      batch_size=4, epochs=1,
                                      has_skip_connection=True, skip_connections_id=[0, 2, 5], shortcuts_len=2)


def generate_graph():
    director = NNGraphBuilder()
    director.set_builder(CNNBuilder(requirements=requirements))
    graph = director.create_nas_graph()
    return graph


def test_has_no_flatten_skip():
    graph = NNGraph.load(os.path.join(tests_root, 'graph_with_flatten_skip.json'))
    successful_check = False
    try:
        has_no_flatten_skip(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_graph_has_several_starts():
    graph = NNGraph.load(os.path.join(tests_root, 'several_starts_graph.json'))
    successful_check = False
    try:
        graph_has_several_starts(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_graph_has_wrong_structure():
    graph = NNGraph.load(os.path.join(tests_root, 'no_conv_graph.json'))
    successful_check = False
    try:
        graph_has_wrong_structure(graph)
    except ValueError:
        successful_check = True
    assert successful_check


def test_flatten_check():
    graph = NNGraph.load(os.path.join(tests_root, 'graph_few_flatten.json'))
    successful_check = False
    try:
        flatten_check(graph)
    except ValueError:
        successful_check = True
    assert successful_check
