from typing import Sequence, List
from nas.graph.grpah_generator import GraphGenerator
from nas.graph.cnn.cnn_graph import NNGraph


class NNGraphBuilder:
    _builder = None

    def set_builder(self, builder: GraphGenerator):
        self._builder = builder

    def build(self):
        return self._builder.build()

    def set_initial_graph(self, graph_to_set):
        pass

    def build_from_existed_graph(self, path):
        return self._builder.load_graph(path)
