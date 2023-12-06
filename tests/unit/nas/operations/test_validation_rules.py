import pathlib

from nas.composer.requirements import load_default_requirements
from nas.graph.node.nas_graph_node import NasNode
from nas.graph.node.nas_node_params import NasNodeFactory
from nas.operations.validation_rules.cnn_val_rules import *
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.utils import set_root, project_root
from tests.unit.nas.utilities import get_graph

tests_path = pathlib.Path(project_root(), 'tests', 'unit', 'test_data')
set_root(tests_path)


def _get_node(node_name: LayersPoolEnum):
    requirements = load_default_requirements()
    node_factory = NasNodeFactory(requirements.model_requirements)
    node_params = node_factory.get_node_params(node_name)
    node = NasNode(content={'name': node_name.value, 'params': node_params}, nodes_from=None)
    return node


def test_model_has_several_starts():
    success = False
    graph = get_graph()
    node = _get_node(LayersPoolEnum.linear)
    graph.add_node(node)
    try:
        model_has_several_roots(graph)
    except ValueError:
        success = True
    assert success


def test_model_has_wrong_number_of_flatten_layers():
    success = False
    node = _get_node(LayersPoolEnum.flatten)
    graph = get_graph()
    graph.add_node(node)
    try:
        model_has_wrong_number_of_flatten_layers(graph)
    except ValueError:
        success = True
    assert success


def test_conv_net_check_structure():
    success = False
    graph = get_graph()
    graph.add_node(_get_node(LayersPoolEnum.adaptive_pool2d))
    graph.add_node(_get_node(LayersPoolEnum.pooling2d))
    graph.add_node(_get_node(LayersPoolEnum.conv2d))
    try:
        conv_net_check_structure(graph)
    except ValueError:
        success = True
    assert success
