from typing import List, Union

from golem.core.dag.graph_node import GraphNode

from nas.graph.node.nn_graph_node import NNNode


class GraphBranchManager:
    def __init__(self):
        self._streams = dict()

    @property
    def streams(self):
        return self._streams

    def __setitem__(self, key, value: dict):
        self._streams[key] = value

    def find_by_node(self, node: NNNode) -> int:
        for key in self._streams.keys():
            if node == self._streams[key]['node']:
                return key

    def get_parent_layer(self, node: NNNode) -> dict:
        key = self.find_by_node(node)
        return self._streams.pop(key)

    def add_and_update(self, node: NNNode, layer, childrens):
        _added_branches_keys = []
        self.update_keys()
        if self._streams:
            self._update(node, layer, _added_branches_keys)
        new_connections = self.new_connections(childrens)
        if new_connections:
            for i in range(new_connections):
                self._add(node, layer, _added_branches_keys)

    def _add(self, node: NNNode, layer, ids: List):
        key = len(self._streams.keys())
        self.__setitem__(key=key, value={'node': node, 'layer': layer})
        ids.append(key)

    def _update(self, current_node: Union[GraphNode, NNNode], layer, new_branches: List):
        # Update all existed branches where last node == current_node
        # Update branches that have nodes_from[0] as parent
        if len(current_node.nodes_from) > 1:
            nodes_to_update = current_node.nodes_from[:-1]
        else:
            nodes_to_update = current_node.nodes_from
        for node_to_update in nodes_to_update:
            key = self.find_by_node(node_to_update)
            if key is not None:
                if key not in new_branches:
                    new_val = {'node': current_node, 'layer': layer}
                    self._streams.update({key: new_val})
                else:
                    continue

    def update_keys(self):
        iter_keys = list(self._streams.keys())
        for actual_key, required_key in zip(iter_keys, range(len(iter_keys))):
            if actual_key != required_key:
                self.__setitem__(required_key, self._streams.pop(actual_key))

    @property
    def existed_connections(self):
        nodes = [self._streams[i]['node'] for i in self._streams.keys()]
        return nodes

    def new_connections(self, nodes: List[NNNode]) -> int:
        if not self._streams:
            return len(nodes)
        else:
            current_node_children = nodes[1:]
            existed_connections = self.existed_connections
            new_connections = [i for i in current_node_children if i not in existed_connections]
        return len(new_connections)
