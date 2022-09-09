from abc import ABC, abstractmethod


class GraphGenerator(ABC):
    @abstractmethod
    def _add_node(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError
