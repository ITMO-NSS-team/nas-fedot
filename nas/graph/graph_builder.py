from nas.graph.grpah_generator import GraphGenerator


class NNGraphBuilder:
    _builder = None

    def set_builder(self, builder: GraphGenerator):
        self._builder = builder

    def create_nas_graph(self):
        return self._builder.build_graph()
