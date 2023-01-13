from enum import Enum
from typing import List

import torch.nn

from nas.graph.node.nn_graph_node import NNNode


class LayerFactory:
    def parse_shape_parameters(self, node: NNNode):
        pass

    @staticmethod
    def conv2d(node: NNNode):
        parameters = node.content['params']

        in_channels = parameters['in_channels']  # number of channels in input image
        out_channels = parameters['out_channels']  # number of output channels
        kernel_size = parameters['kernel_size']  # conv layer kernel size

        stride = parameters.get('stride', 1)
        padding = parameters.get('padding', 0)
        dilation = parameters.get('dilation', 1)

        return torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               dilation=dilation)


class BaseLayersTypes(Enum):
    conv2d = torch.nn.Conv2d
    dense = 'dense'
    dropout = 'dropout'
    batch_norm = 'batch_norm'
    max_pool2d = 'max_pool2d'
    avg_pool2d = 'average_pool2d'
    flatten = 'flatten'


class LayerInitializer:
    def __init__(self, layers_pool: List):
        self.layers_pool = layers_pool

    def __call__(self, node: NNNode):
        """
        Returns pytorch layer object based on given node.
        node: NNNode to convert into pytorch layer
        """

        layer_type = node.content['name']
        if layer_type not in self.layers_pool:
            raise ValueError(f'Possible layers pool does not contain layer {layer_type}')

        return self.make_layer_from_node(node)

    def make_layer_from_node(self, node: NNNode):
        layer_parameters = node.content
        return
