import os

from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes

from nas.cnn_builder import CNNBuilder, NASDirector
from nas.composer.cnn_graph import CNNGraph
from nas.composer.cnn_graph_node import CNNNode
from nas.composer.cnn_graph_operator import random_conv_graph_generation
from nas.composer.gp_cnn_composer import GPNNComposerRequirements
from nas.graph_cnn_mutations import has_no_flatten_skip, flatten_check, graph_has_several_starts
from nas.utils.var import TESTING_ROOT, DEFAULT_NODES_PARAMS

NODES_LIST = ['conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dense', 'dense']
REQUIREMENTS = GPNNComposerRequirements(input_shape=[120, 120, 3], pop_size=1,
                                        num_of_generations=1, max_num_of_conv_layers=4,
                                        max_nn_depth=3, primary=['conv2d'], secondary=['dense'],
                                        batch_size=4, epochs=1,
                                        has_skip_connection=True, skip_connections_id=[0, 2, 5], shortcuts_len=2,
                                        batch_norm_prob=-1, dropout_prob=-1,
                                        default_parameters=DEFAULT_NODES_PARAMS)


def test_is_cnn_builder_correct():
    director = NASDirector()
    director.set_builder(CNNBuilder(NODES_LIST, requirements=REQUIREMENTS))
    loaded_static_graph = CNNGraph.load(os.path.join(TESTING_ROOT, 'static_graph.json'))
    generated_static_graph = director.create_nas_graph()
    assert loaded_static_graph.operator.is_graph_equal(generated_static_graph)


def test_static_graph_generation():
    r = GPNNComposerRequirements(input_shape=[120, 120, 3], pop_size=1,
                                 num_of_generations=1, max_num_of_conv_layers=4,
                                 max_nn_depth=3, primary=['conv2d'], secondary=['dense'],
                                 batch_size=4, epochs=1,
                                 has_skip_connection=True,
                                 )
    director = NASDirector()
    director.set_builder(CNNBuilder(NODES_LIST, requirements=r))
    graphs = []
    for _ in range(10):
        graph = director.create_nas_graph()
        graphs.append(graph)
    for g in graphs:
        for val in [has_no_flatten_skip, flatten_check, has_no_cycle, has_no_self_cycled_nodes,
                    graph_has_several_starts, has_no_self_cycled_nodes]:
            try:
                val(g)
            except ValueError:
                assert False
    assert True


def test_graph_has_right_dtype():
    for _ in range(10):
        director = NASDirector()
        director.set_builder(CNNBuilder(NODES_LIST, requirements=REQUIREMENTS))
        graph = director.create_nas_graph()
        assert type(graph) == CNNGraph
