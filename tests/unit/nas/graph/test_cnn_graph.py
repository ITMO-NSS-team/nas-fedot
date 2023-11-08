from nas.composer.requirements import load_default_requirements
from nas.graph.base_graph import NasGraph
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
