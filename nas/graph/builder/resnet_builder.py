from golem.core.dag.graph_node import GraphNode

from nas.composer.requirements import *
from nas.graph.base_graph import NasGraph
from nas.graph.builder.base_graph_builder import GraphGenerator
from nas.graph.node.nas_graph_node import NasNode
from nas.graph.node.nas_node_params import NasNodeFactory
from nas.repository.existing_cnn_enum import CNNEnum
from nas.repository.layer_types_enum import LayersPoolEnum


class CNNRepository:
    def _build_resnet_34(self):
        pass

    def architecture_by_type(self, model_type: CNNEnum):
        models = {
            CNNEnum.resnet34: self._build_resnet_34()
        }


@dataclass
class ResnetConfig:
    conv_params = {'resnet_18': ({'kernel_size': [3, 3], 'padding': [1, 1]},
                                 {'kernel_size': [3, 3], 'padding': [1, 1]}),
                   'resnet_34': ({'kernel_size': [3, 3], 'padding': [1, 1]},
                                 {'kernel_size': [3, 3], 'padding': [1, 1]}),
                   'resnet_50': ({'kernel_size': [1, 1], 'padding': 0},
                                 {'kernel_size': [3, 3], 'padding': [1, 1]},
                                 {'kernel_size': [1, 1], 'padding': 0}),
                   'resnet_101': ({'kernel_size': [1, 1], 'padding': 0},
                                  {'kernel_size': [3, 3], 'padding': [1, 1]},
                                  {'kernel_size': [1, 1], 'padding': 0}),
                   'resnet_152': ({'kernel_size': [1, 1], 'padding': 0},
                                  {'kernel_size': [3, 3], 'padding': [1, 1]},
                                  {'kernel_size': [1, 1], 'padding': 0})}
    blocks_num = {'resnet_18': (2, 2, 2, 2), 'resnet_34': (3, 4, 6, 3),
                  'resnet_50': (3, 4, 6, 3), 'resnet_101': (3, 4, 23, 3),
                  'resnet_152': (3, 8, 36, 3)}
    padding = {'conv3x3': [1, 1], 'conv1x1': 0}


class ResNetBuilder(GraphGenerator):
    def __init__(self, *args, **kwargs):
        self.resnet_type: Optional[str] = kwargs.get('model_type')
        self._graph: Optional[NasGraph] = None

    @staticmethod
    def _default_conv_padding(k_size: Union[List[int], Tuple[int, int], int],
                              padding: Union[List[int], Tuple[int, int], int] = None):
        if padding is not None:
            padding = [1, 1] if (k_size == 3 or k_size == [3, 3]) else 0
        return padding

    def _add_node(self, node_name: LayersPoolEnum, parent_node: NasNode = None, *args, **params):
        if not parent_node:
            parent_node = [self._graph.nodes[-1]] if self._graph.nodes else []
        else:
            parent_node = [parent_node]
        node_parameters = NasNodeFactory().get_node_params(node_name, **params)
        node = NasNode(content={'name': node_name.value, 'params': node_parameters}, nodes_from=parent_node)
        self._graph.add_node(node)
        return self

    def _make_skip_connection(self, start_node: Union[NasNode, GraphNode], end_node: Union[NasNode, GraphNode],
                              input_stride: List[int], downsample, expansion, stride):
        # downsample = True if input_stride == [2, 2] else False
        out_shape = end_node.parameters['out_shape'] if end_node.parameters.get('out_shape') is not None \
            else end_node.nodes_from[0].parameters['out_shape']
        if downsample:
            self._add_node(LayersPoolEnum.conv2d, ativation='relu', stride=stride, kernel_size=[1, 1],
                           out_shape=out_shape, parent_node=start_node, momentum=.99, epsilon=.001)
            start_node = self._graph.nodes[-1]
        end_node.nodes_from.append(start_node)
        return self

    def _add_block(self, input_shape: int, blocks_num: int, block_expansion: int = 1, conv_stride=1):
        skip_connection_end = None
        downsample = input_shape * block_expansion != input_shape or conv_stride != 1
        for step in range(blocks_num):
            skip_connection_start = self._graph.root_node if skip_connection_end is None else skip_connection_end
            conv_stride = 1 if step > 1 else conv_stride
            self._add_node(LayersPoolEnum.conv2d, ativation='relu', stride=conv_stride, out_shape=input_shape,
                           parent_node=skip_connection_start, momentum=.99, epsilon=.001,
                           **ResnetConfig.conv_params[self.resnet_type][0])
            self._add_node(LayersPoolEnum.conv2d, ativation='relu', stride=[1, 1], out_shape=input_shape,
                           momentum=.99, epsilon=.001, **ResnetConfig.conv_params[self.resnet_type][1])
            if self.resnet_type in ['resnet_50', 'resnet_101', 'resnet_152']:
                self._add_node(LayersPoolEnum.conv2d, ativation='relu', stride=[1, 1],
                               out_shape=input_shape * block_expansion, momentum=.99, epsilon=.001,
                               **ResnetConfig.conv_params[self.resnet_type][2])
            skip_connection_end = self._graph.nodes[-1]
            self._make_skip_connection(skip_connection_start, skip_connection_end, conv_stride, downsample,
                                       block_expansion, conv_stride)
        return self

    def build(self, resnet_type: Optional[str] = None) -> NasGraph:
        self.resnet_type = resnet_type if resnet_type else self.resnet_type
        input_shapes = [64, 128, 256, 512]
        block_expansion = 1 if self.resnet_type in ['resnet_18, resnet_34'] else 4
        if self.resnet_type not in ResnetConfig.blocks_num.keys():
            raise ValueError(f'Builder cannot build "{self.resnet_type}" model.')

        self._graph = NasGraph()
        self._add_node(LayersPoolEnum.conv2d, activation='relu', stride=[2, 2], out_shape=64, kernel_size=[7, 7],
                       momentum=.99, epsilon=.001, padding=[3, 3])
        self._add_node(LayersPoolEnum.pooling2d, pool_size=[3, 3], pool_stride=[2, 2], mode='max', padding=[1, 1])
        for i, input_shape in enumerate(input_shapes):
            self._add_block(input_shape, blocks_num=ResnetConfig.blocks_num[self.resnet_type][i],
                            block_expansion=block_expansion)
        self._add_node(LayersPoolEnum.adaptive_pool2d, out_shape=[1, 1], mode='avg', parent_node=self._graph.root_node)
        self._add_node(LayersPoolEnum.flatten, parent_node=self._graph.root_node)
        return self._graph
