from typing import Optional, List
from fedot.core.optimisers.graph import OptNode
from nas.graph.node.nn_node_params import GraphLayers


def get_node_params_by_type(node, requirements):
    return GraphLayers().layer_by_type(node, requirements)


class NNNode(OptNode):
    def __init__(self, content: dict, nodes_from: Optional[List] = None):
        super().__init__(content, nodes_from)
        self.nodes_from = nodes_from
        if 'params' in content:
            self.content = content
            self.content['name'] = self.content['name'].value

    def __str__(self):
        return str(self.content['name'])

    def __repr__(self):
        return self.__str__()
