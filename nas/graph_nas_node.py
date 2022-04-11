from typing import (List, Optional)

from nas.layer import LayerParams, LayerTypesIdsEnum
from fedot.core.optimisers.graph import OptNode


class NNNode(OptNode):
    def __init__(self, content, nodes_from: Optional[List['NNNode']]):
        super().__init__(content, nodes_from)
        self.nodes_from = nodes_from
        if 'params' in content:
            self.content = content

    def __str__(self):
        type = self.content['params'].layer_type
        if type == LayerTypesIdsEnum.conv2d:
            layer_name = f'{type}\n{self.content["params"].activation, self.content["params"].num_of_filters}'
            if self.content['params'].pool_size:
                if self.content['params'].pool_type == LayerTypesIdsEnum.maxpool2d.name:
                    layer_name += '\n maxpool2d'
                elif self.content['params'].pool_type == LayerTypesIdsEnum.averagepool2d.name:
                    layer_name += '\n averagepool2d'
        elif type == LayerTypesIdsEnum.dense.name:
            layer_name = f'{type}({self.content["params"].neurons})'
        elif type == LayerTypesIdsEnum.dropout.name:
            layer_name = f'{type}({self.content["params"].drop})'
        else:
            layer_name = type
        return layer_name

