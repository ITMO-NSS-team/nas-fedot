from abc import ABC, abstractmethod


class GraphGenerator(ABC):
    @abstractmethod
    def _add_node(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError


class BaseGraphBuilder:
    _builder = None

    def set_builder(self, builder: GraphGenerator):
        self._builder = builder
        return self

    def build(self, *args, **kwargs):
        return self._builder.build(*args, **kwargs)

    def set_initial_graph(self, graph_to_set):
        pass

    def build_from_existed_graph(self, path):
        return self._builder.load_graph(path)
