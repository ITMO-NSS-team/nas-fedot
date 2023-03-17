import tensorflow as tf

from nas.composer.requirements import load_default_requirements
from nas.graph.cnn_graph import NasGraph
from tests.unit.nas.utilities import get_graph


def test_generated_graph_len():
    requirements = load_default_requirements()
    max_depth = requirements.max_depth
    for _ in range(100):
        graph = get_graph()
        if not max_depth >= len(graph):
            assert False
    assert True


def test_generated_graph_nodes_num():
    for _ in range(100):
        graph = get_graph()
        has_correct_nodes_num = len(graph.nodes) > 1
        if not has_correct_nodes_num:
            assert False
    assert True


def test_graph_type():
    for _ in range(100):
        graph = get_graph()
        assert isinstance(graph, NasGraph)


def test_graph_model_build():
    for _ in range(100):
        graph = get_graph()
        try:
            graph.model = graph.compile_model([32, 32, 3], 'binary_crossentropy',
                                              optimizer=tf.keras.optimizers.Adam, n_classes=3)
        except ValueError:
            assert False
    assert True
