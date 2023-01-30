from fedot.core.utils import DEFAULT_PARAMS_STUB
from golem.core.optimisers.graph import OptNode

from nas.graph.node.nas_graph_node import NasNode
from nas.utils.default_parameters import default_nodes_params


class NasNodeOperatorAdapter:
    def adapt(self, adaptee) -> OptNode:
        adaptee.__class__ = OptNode
        return adaptee

    def restore(self, node) -> NasNode:
        obj = node
        obj.__class__ = NasNode
        if obj.content['params'] == DEFAULT_PARAMS_STUB:
            node_name = obj.content.get('name')
            obj.content = default_nodes_params[node_name]
        return obj
