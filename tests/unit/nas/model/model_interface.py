from nas.graph.builder.base_graph_builder import BaseGraphBuilder
from nas.graph.builder.resnet_builder import ResNetBuilder


def test_resnet_builder_inputs():
    is_correct = False
    resnet_types = ['resnet_18', 'ResNet18', 'rESNET18', None]
    graph_builder = BaseGraphBuilder().set_builder(ResNetBuilder())
    for resnet in resnet_types:
        try:
            graph_builder.build(resnet)
        except ValueError:
            is_correct = True
    assert is_correct


def test_different_resnet_types():
    is_correct = True
    resnet_types = {'resnet_18': 18, 'resnet_34': 34, 'resnet_50': 50}
    graph_builder = BaseGraphBuilder().set_builder(ResNetBuilder())
    for resnet in resnet_types.keys():
        graph = graph_builder.build(resnet)
        if not graph.depth == resnet_types[resnet] + 1:
            is_correct = False
    assert is_correct
