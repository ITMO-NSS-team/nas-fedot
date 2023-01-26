from nas.composer.nn_composer_requirements import load_default_requirements
from nas.graph.cnn.cnn_builder import ConvGraphMaker
from nas.graph.cnn.cnn_graph import NasGraph
from nas.graph.graph_builder import NNGraphBuilder


def _get_graph():
    requirements = load_default_requirements()
    builder = NNGraphBuilder()
    cnn_builder = ConvGraphMaker(requirements=requirements.model_requirements)
    builder.set_builder(cnn_builder)
    return builder.build()


def test_generated_graph_len():
    pass


def test_generated_graph_nodes_num():
    for _ in range(100):
        graph = _get_graph()
        has_correct_nodes_num = len(graph.nodes) > 1
        if not has_correct_nodes_num:
            assert False
    assert True


def test_graph_type():
    for _ in range(100):
        graph = _get_graph()
        assert isinstance(graph, NasGraph)
