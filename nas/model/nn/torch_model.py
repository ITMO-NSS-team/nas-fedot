from dataclasses import dataclass

import torch

from nas.graph.node.nn_graph_node import NNNode
from nas.model.layers.base_layers import LayerInitializer


@dataclass
class LayerMaker:
    @staticmethod
    def conv2d_initialize(node: NNNode):
        params = node.content['params']
        in_channels = params['input_channels']
        out_channels = params['out_channels']
        kernel_size = params['kernel_size']
        stride = params['stride']

        return torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    @staticmethod
    def dense_initialize(node: NNNode):
        params = node.content['params']
        in_features = params['input_channels']
        out_features = params['out_channels']

        return torch.nn.Linear(in_features, out_features)

    @staticmethod
    def pool_initialize(node: NNNode):
        params = node.content['params']
        kernel_size = params['pool_size']
        stride = params['pool_strides']
        if node.content['name'] == 'max_pool2d':
            return torch.nn.MaxPool2d(kernel_size, stride)
        elif node.content['name'] == 'avg_pool2d':
            return torch.nn.AvgPool2d(kernel_size, stride)

    @staticmethod
    def dropout_initialize(node: NNNode):
        params = node.content['params']
        drop_ratio = params['drop']
        return torch.nn.Dropout(drop_ratio)

    @staticmethod
    def batch_norm_initialize(node: NNNode):
        params = node.content['params']
        return torch.nn.BatchNorm2d


    @staticmethod
    def make_layer(node: NNNode):
        layer_name = LayerMaker.layer_type(node.content['name'])
        layer = LayerMaker.layer_initializer_functions[layer_name]
        shape_parameters = node.content['params']
        return

    @staticmethod
    def layer_type(node_name: str):
        if 'conv2d' in node_name:
            return 'conv2d'
        return node_name

    def __call__(self, node: NNNode, *args, **kwargs):
        layer_type = self.layer_type(node.content['name'])
        return self.layer_initializer_functions[layer_type]

    def layer_initializer_functions(self):
        _layer_initializer_functions = {'conv2d': self.conv2d_initialize,
                                        'flatten': torch.nn.Flatten,
                                        'dense': self.dense_initialize,
                                        'dropout': self.dropout_initialize,
                                        'batch_norm': self.batch_norm_initialize,
                                        'average_pool2d': self.pool_initialize,
                                        'max_pool2d': self.pool_initialize}


class BaseModel(torch.nn.Module):
    def __init__(self, graph, requirements, layer_initializer: LayerInitializer):
        super().__init__()
        self._layer_initializer = layer_initializer
        self.layers = [self._layer_initializer(node) for node in graph.graph_sturct]
