from golem.core.dag.graph_node import GraphNode

from nas.composer.requirements import *
from nas.graph.BaseGraph import NasGraph
from nas.graph.builder.base_graph_builder import GraphGenerator
from nas.graph.node.nas_graph_node import NasNode, get_node_params_by_type
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


class ResNetBuilder(GraphGenerator):
    def __init__(self, *args, **kwargs):
        self.resnet_type: Optional[str] = kwargs.get('model_type')
        # self.requirements = load_default_requirements().model_requirements
        self._graph: Optional[NasGraph] = None
        self.block_kernel_size = {'resnet_18': ([3, 3], [3, 3]),
                                  'resnet_34': ([3, 3], [3, 3]),
                                  'resnet_50': ([3, 3], [1, 1], [3, 3]),
                                  'resnet_101': ([3, 3], [1, 1], [3, 3]),
                                  'resnet_152': ([3, 3], [1, 1], [3, 3])}
        self._blocks_num = {'resnet_18': (2, 2, 2, 2), 'resnet_34': (3, 4, 6, 3),
                            'resnet_50': (3, 4, 6, 3), 'resnet_101': (3, 4, 23, 3),
                            'resnet_152': (3, 8, 36, 3)}

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
                              input_stride: List[int]):
        downsample = True if input_stride == [2, 2] else False
        out_shape = end_node.parameters['out_shape'] if end_node.parameters.get('out_shape') is not None \
            else end_node.nodes_from[0].parameters['out_shape']
        if downsample:
            self._add_node(LayersPoolEnum.conv2d, ativation='relu', stride=[2, 2],
                           out_shape=out_shape, parent_node=start_node, momentum=.99, epsilon=.001)
            start_node = self._graph.nodes[-1]
        end_node.nodes_from.append(start_node)
        return self

    def _add_block(self, input_shape: int, blocks_num: int, downsample: bool = False, block_expansion: int = 1):
        skip_connection_end = None
        for step in range(blocks_num):
            skip_connection_start = self._graph.nodes[-1] if not skip_connection_end else skip_connection_end
            conv_stride = [2, 2] if step == 0 and downsample else [1, 1]

            self._add_node(LayersPoolEnum.conv2d, ativation='relu', stride=conv_stride, out_shape=input_shape,
                           kernel_size=self.block_kernel_size[self.resnet_type][0], parent_node=skip_connection_start,
                           momentum=.99, epsilon=.001)
            self._add_node(LayersPoolEnum.conv2d, ativation='relu', stride=[1, 1], out_shape=input_shape,
                           momentum=.99, epsilon=.001, kernel_size=self.block_kernel_size[self.resnet_type][1])
            if self.resnet_type in ['resnet_50', 'resnet_101', 'resnet_152']:
                self._add_node(LayersPoolEnum.conv2d, ativation='relu', stride=[1, 1], out_shape=input_shape,
                               momentum=.99, epsilon=.001, kernel_size=self.block_kernel_size[self.resnet_type][2])
            skip_connection_end = self._graph.nodes[-1]
            self._make_skip_connection(skip_connection_start, skip_connection_end, conv_stride)
        return self

    def build(self, resnet_type: Optional[str] = None) -> NasGraph:
        self.resnet_type = resnet_type if resnet_type else self.resnet_type
        block_expansion = 1 if self.resnet_type in ['resnet_18, resnet_34'] else 4
        if self.resnet_type not in self._blocks_num.keys():
            raise ValueError(f'Builder cannot build "{self.resnet_type}" model.')

        self._graph = NasGraph()
        self._add_node(LayersPoolEnum.conv2d, activation='relu', stride=[2, 2], out_shape=64, kernel_size=[7, 7],
                       momentum=.99, epsilon=.001)
        self._add_node(LayersPoolEnum.max_pool2d, pool_size=[3, 3], pool_stride=[2, 2])
        self._add_block(64, blocks_num=self._blocks_num[self.resnet_type][0], block_expansion=block_expansion)
        self._add_block(128, blocks_num=self._blocks_num[self.resnet_type][1], downsample=True,
                        block_expansion=block_expansion)
        self._add_block(256, blocks_num=self._blocks_num[self.resnet_type][2], downsample=True,
                        block_expansion=block_expansion)
        self._add_block(512, blocks_num=self._blocks_num[self.resnet_type][3], downsample=True,
                        block_expansion=block_expansion)
        self._add_node(LayersPoolEnum.average_poold2, pool_size=[1, 1])
        self._add_node(LayersPoolEnum.flatten)
        return self._graph
