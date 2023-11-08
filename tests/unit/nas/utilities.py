from nas.composer.requirements import load_default_requirements
from nas.graph.base_graph import NasGraph
from nas.graph.builder.base_graph_builder import BaseGraphBuilder
from nas.graph.builder.cnn_builder import ConvGraphMaker


def get_graph() -> NasGraph:
    requirements = load_default_requirements()
    builder = BaseGraphBuilder()
    cnn_builder = ConvGraphMaker(requirements=requirements.model_requirements)
    builder.set_builder(cnn_builder)
    return builder.build()
