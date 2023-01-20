from typing import TYPE_CHECKING
from fedot.core.utils import DEFAULT_PARAMS_STUB
from golem.core.optimisers.graph import OptNode

from nas.graph.node.nn_graph_node import NNNode
from nas.utils.default_parameters import default_nodes_params

if TYPE_CHECKING:
    from nas.graph.cnn.cnn_graph import NasGraph


class NNNodeOperatorAdapter:
    def adapt(self, adaptee) -> OptNode:
        adaptee.__class__ = OptNode
        return adaptee

    def restore(self, node) -> NNNode:
        obj = node
        obj.__class__ = NNNode
        if obj.content['params'] == DEFAULT_PARAMS_STUB:
            node_name = obj.content.get('name')
            obj.content = default_nodes_params[node_name]
        return obj
