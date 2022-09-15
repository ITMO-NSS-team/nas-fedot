from nas.graph.cnn.cnn_graph import NNGraph
from nas.nn.keras_graph_converter import build_nn_from_graph
from tests.unit.nas.utility_functions import get_requirements, get_graph


def test_graph_type():
    for _ in range(100):
        graph = get_graph()
        assert isinstance(graph, NNGraph)


def test_is_valid_graph():
    for _ in range(100):
        graph = get_graph()
        assert len(graph.nodes) > 1


def test_is_graph_trainable():
    is_valid = True
    for _ in range(100):
        graph = get_graph()
        try:
            build_nn_from_graph(graph, n_classes=4, requirements=get_requirements())
        except (ValueError, MemoryError) as ex:
            is_valid = False
        assert is_valid
