from copy import deepcopy
from typing import Any

from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.utils import DEFAULT_PARAMS_STUB

from nas.utils.var import DEFAULT_NODES_PARAMS


class CustomGraphAdapter(DirectAdapter):
    def __init__(self, base_graph_class=None, base_node_class=None, log=None):
        super().__init__(base_graph_class=base_graph_class, base_node_class=base_node_class, log=log)
        self.base_graph_params = {}

    def adapt(self, adaptee: Any) -> OptGraph:
        opt_graph = deepcopy(adaptee)
        opt_graph.__class__ = OptGraph
        for node in opt_graph.nodes:
            self.base_graph_params[node.distance_to_primary_level] = node.content['params']
            node.__class__ = OptNode
        return opt_graph

    def restore(self, opt_graph: OptGraph):
        obj = deepcopy(opt_graph)
        obj.__class__ = self.base_graph_class
        for node in obj.nodes:
            node.__class__ = self.base_node_class
            if node.content['params'] == DEFAULT_PARAMS_STUB:
                node.content['params'] = DEFAULT_NODES_PARAMS[node.content['name']]
        return obj
