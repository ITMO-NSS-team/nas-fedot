from typing import (List, Optional)

from nas.layer import LayerParams, LayerTypesIdsEnum
from fedot.core.dag.graph_node import GraphNode
from fedot.core.optimisers.graph import OptNode


class NNNode(OptNode):
    def __init__(self, content, nodes_from: Optional[List['NNNode']], layer_params: LayerParams):
        super().__init__(content, nodes_from)
        self.nodes_from = nodes_from
        self.layer_params = layer_params

    def __str__(self):
        type = self.layer_params.layer_type
        if type == LayerTypesIdsEnum.conv2d:
            layer_name = f'{type.value}\n{self.layer_params.activation.value, self.layer_params.num_of_filters}'
            if self.layer_params.pool_size:
                if self.layer_params.pool_type == LayerTypesIdsEnum.maxpool2d:
                    layer_name += '\n maxpool2d'
                elif self.layer_params.pool_type == LayerTypesIdsEnum.averagepool2d:
                    layer_name += '\n averagepool2d'
        elif type == LayerTypesIdsEnum.dense:
            layer_name = f'{type.value}({self.layer_params.neurons})'
        elif type == LayerTypesIdsEnum.dropout:
            layer_name = f'{type.value}({self.layer_params.drop})'
        else:
            layer_name = type.value
        return layer_name

    @property
    def ordered_subnodes_hierarchy(self) -> List['NNNode']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.ordered_subnodes_hierarchy
        return nodes


class NNNodeGenerator:
    @staticmethod
    def primary_node(layer_params: LayerParams) -> NNNode:
        return PrimaryNode(layer_params=layer_params)

    @staticmethod
    def secondary_node(layer_params: LayerParams = None,
                       nodes_from: Optional[List['NNNode']] = None) -> NNNode:
        return SecondaryNode(nodes_from=nodes_from, layer_params=layer_params)


class PrimaryNode(NNNode):
    def __init__(self, layer_params: LayerParams):
        super().__init__(nodes_from=None, layer_params=layer_params)


class SecondaryNode(NNNode):
    def __init__(self, nodes_from: Optional[List['NNNode']],
                 layer_params: LayerParams):
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, layer_params=layer_params)
