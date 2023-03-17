import math
from dataclasses import dataclass
from typing import List, Union

import numpy as np

from nas.composer.requirements import NNComposerRequirements
from nas.graph.node.nas_graph_node import NasNode, get_node_params_by_type
from nas.repository.layer_types_enum import LayersPoolEnum


def add_shortcut_and_check(input_shape: List, output_shape: List) -> bool:
    """Adds conv 1x1 in shortcut with different strides and сompare"""

    stride = math.ceil(input_shape[0] / output_shape[0])
    layer_type = LayersPoolEnum.conv2d_1x1
    requirements = NNComposerRequirements()
    layer_params = get_node_params_by_type(layer_type, requirements.model_requirements)
    shortcut_node = NasNode(content={'name': layer_type.value, 'params': layer_params})
    shortcut_node.content['params']['conv_strides'] = [stride, stride]
    shortcut_node.content['params']['neurons'] = output_shape[-1]
    shape = get_shape(input_shape, shortcut_node)
    return shape


def get_shape(input_shape: List, node: NasNode) -> List:
    return ParamCounter().get_output_shape(node, input_shape)


def count_node_params(node: NasNode, input_shape: List) -> List:
    pass


@dataclass
class ParamCounter:
    @staticmethod
    def _conv(input_shape: Union[List, np.ndarray], node: NasNode) -> Union[np.ndarray, List]:
        layer_params = node.content['params']
        stride = layer_params['conv_strides'][0]
        output_array = [math.ceil(i / stride) for i in input_shape[:2]]
        channels_num = layer_params['neurons']
        output_array.append(channels_num)
        return output_array

    @staticmethod
    def _fully_connected(input_shape: Union[List, np.ndarray], node: NasNode) -> List:
        layer_params = node.content['params']
        output = layer_params['neurons']
        return [output]

    @staticmethod
    def _pooling(input_shape: List, node: NasNode) -> List:
        layer_params = node.content['params']
        pool_stride = layer_params['pool_strides']
        output = [math.ceil(i / pool_stride[0]) for i in input_shape[:2:]]
        output.append(input_shape[-1])
        return output

    @staticmethod
    def _flatten(input_shape: List[float], node: NasNode) -> List:
        output = math.prod(input_shape)
        return [output]

    @staticmethod
    def _get_type(name: str):
        if 'conv' in name:
            return 'conv'
        if 'dense' in name:
            return 'fully_connected'
        if 'flatten' in name:
            return 'flatten'
        if 'pool' in name:
            return 'pooling'

    def get_output_shape(self, node: NasNode, input_shape) -> Union[np.ndarray, List]:
        layer_types = {
            'conv': self._conv,
            'fully_connected': self._fully_connected,
            'flatten': self._flatten,
            'pooling': self._pooling
        }
        _node_type = self._get_type(node.content['name'])
        if _node_type in layer_types:
            return layer_types[_node_type](input_shape, node)
        else:
            raise ValueError
