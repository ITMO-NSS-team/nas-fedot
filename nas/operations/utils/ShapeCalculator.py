from typing import Callable, Dict, List, Union

from math import floor, ceil
import numpy as np

from nas.graph.builder.resnet_builder import ResNetBuilder
from nas.graph.node.nas_graph_node import NasNode


def layer_type_from_node(node_name: str) -> str:
    """
    Auxiliary function to get exact node type by its name.
    """
    node_name = 'pooling' if 'pool' in node_name else node_name
    node_name = 'conv2d' if 'conv' in node_name else node_name
    return node_name


class ShapeCalculator:
    def __init__(self):
        self.nodes_list = {'conv2d': self.conv2d,
                           'dense': self.dense,
                           'pooling': self.pooling,
                           'dropout': self.dropout,
                           'flatten': self.flatten}

    def __call__(self, node: NasNode, input_shape: Union[np.ndarray, List]) -> np.ndarray:
        node_shape_func = self._shape_func_by_node(node.name)
        output_shape = node_shape_func(input_shape=input_shape, node_params=node.parameters)
        return np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape

    def _shape_func_by_node(self, node_name) -> Callable:
        node_name = layer_type_from_node(node_name)
        return self.nodes_list[node_name]

    @staticmethod
    def conv2d(input_shape: Union[np.ndarray, List], node_params: Dict):
        kernel_size = node_params.get('kernel_size')
        conv_stride = node_params.get('conv_strides', (1, 1))
        dilation_rate = node_params.get('dilation', (1, 1))
        padding = node_params.get('padding', (0, 0))
        out_channels = node_params.get('neurons')
        if padding == 'same':
            out_h = floor(input_shape[0] / conv_stride[0])
            out_w = floor(input_shape[1] / conv_stride[1])
        else:
            out_h = floor(((input_shape[0] - (2 * padding[0]) - (dilation_rate[0] * (kernel_size[0] - 1)) - 1) /
                           conv_stride[0]) + 1)
            out_w = floor(((input_shape[1] - (2 * padding[1]) - (dilation_rate[1] * (kernel_size[1] - 1)) - 1) /
                           conv_stride[1]) + 1)
        out_h = 1 if not out_h else out_w
        out_w = 1 if not out_w else out_w
        out_shape = [out_h, out_w, out_channels]
        return out_shape

    @staticmethod
    def dense(input_shape: Union[np.ndarray, List], node_params: Dict):
        input_features = node_params.get('neurons')
        return [input_features, input_shape[1]]

    @staticmethod
    def pooling(input_shape: Union[np.ndarray, List], node_params: Dict):
        pool_stride = node_params.get('pool_stride')
        out_h = floor(input_shape[0] / pool_stride[0])
        out_w = floor(input_shape[1] / pool_stride[1])
        out_channels = input_shape[-1]
        out_h = 1 if not out_h else out_w
        out_w = 1 if not out_w else out_w
        out_shape = [out_h, out_w, out_channels]
        return out_shape

    @staticmethod
    def flatten(input_shape: Union[np.ndarray, List], *args, **kwargs):
        return np.prod(np.array(input_shape))

    @staticmethod
    def dropout(input_shape: Union[np.ndarray, List], **kwargs):
        return input_shape
