from typing import Sequence, List
from nas.graph.grpah_generator import GraphGenerator


class NNGraphBuilder:
    _builder = None

    def set_builder(self, builder: GraphGenerator):
        self._builder = builder

    def build(self):
        return self._builder.build()

    def set_initial_graph(self, graph_to_set):
        pass
