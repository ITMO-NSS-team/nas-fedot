from typing import Callable, Dict, List

import numpy as np

from nas.graph.node.nas_graph_node import NasNode


class ShapeCalculator:
    def __init__(self):
        nodes_list = {'conv2d': self.conv2d,
                      'dense': self.dense,
                      'pooling': self.pooling,
                      'dropout': self.dropout,
                      'flatten': self.flatten}

    def __call__(self, node: NasNode, input_shape: List):
        self._shape_func_by_node(node)

    def _shape_func_by_node(self, node) -> Callable:
        node_params = node.parameters
        node_name = node.content['name']


    @staticmethod
    def conv2d(input_shape: List, node_params: Dict):
        kernel_size = node_params.get('kernel_size')
        conv_stride = node_params.get('conv_strides', (1, 1))
        dilation_rate = node_params.get('dilation', (1, 1))
        padding = node_params.get('padding', (0, 0))
        out_channels = node_params.get('neurons')
        conv_params = zip(input_shape, padding, dilation_rate, kernel_size, conv_stride)
        if padding == 'same':
            out_shape = [img_side // conv_side for img_side, conv_side in zip(input_shape, conv_stride)]
        else:
            out_shape = [((h - 2 * p - d * (k_size - 1) - 1) / s) + 1 for h, p, d, k_size, s in conv_params]
        out_shape.append(out_channels)
        return out_shape

    @staticmethod
    def dense(input_shape: List, node_params: Dict):
        input_features = node_params.get('neurons')
        return [input_features[0], input_shape[1]]

    @staticmethod
    def pooling(node_params: Dict, input_shape: List):
        pool_stride = node_params.get('pool_stride')
        return [size / stride for size, stride in zip(input_shape, pool_stride)]

    @staticmethod
    def flatten(input_shape: List, **kwargs):
        return np.prod(np.array(input_shape))

    @staticmethod
    def dropout(input_shape: List, **kwargs):
        return input_shape
