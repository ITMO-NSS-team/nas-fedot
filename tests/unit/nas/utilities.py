from nas.composer.nn_composer_requirements import load_default_requirements
from nas.graph.cnn.cnn_builder import ConvGraphMaker
from nas.graph.cnn.cnn_graph import NasGraph
from nas.graph.graph_builder import NNGraphBuilder


def get_graph() -> NasGraph:
    requirements = load_default_requirements()
    builder = NNGraphBuilder()
    cnn_builder = ConvGraphMaker(requirements=requirements.model_requirements)
    builder.set_builder(cnn_builder)
    return builder.build()
