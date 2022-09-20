from abc import ABC, abstractmethod


class GraphGenerator(ABC):
    @abstractmethod
    def _add_node(self, node_type, node_from):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError
