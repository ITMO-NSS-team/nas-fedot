from golem.core.dag.graph_node import GraphNode

from nas.composer.requirements import *
from nas.graph.BaseGraph import NasGraph
from nas.graph.builder.base_graph_builder import GraphGenerator
from nas.graph.node.nas_graph_node import NasNode, get_node_params_by_type
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
        self.requirements = load_default_requirements().model_requirements
        self._graph: Optional[NasGraph] = None
        self._blocks_num = {'resnet_18': (2, 2, 2, 2), 'resnet_34': (3, 4, 6, 3),
                            'resnet_50': (3, 4, 6, 3), 'resnet_101': (3, 4, 23, 3),
                            'resnet_152': (3, 8, 36, 3)}
        self._k_shape = {'resnet_18': (LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_3x3),
                         'resnet_34': (LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_3x3),
                         'resnet_50': (LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3,
                                       LayersPoolEnum.conv2d_1x1),
                         'resnet_101': (LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3,
                                        LayersPoolEnum.conv2d_1x1),
                         'resnet_152': (LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3,
                                        LayersPoolEnum.conv2d_1x1)}

    def _add_node(self, layer_type, parent_node: NasNode = None, *args, **kwargs):
        if not parent_node:
            parent_node = [self._graph.nodes[-1]] if self._graph.nodes else []
        else:
            parent_node = [parent_node]

        additional_layer_params: dict = kwargs.get('additional_layer_params')
        node_params = get_node_params_by_type(layer_type, self.requirements)
        if additional_layer_params:
            node_params.update(additional_layer_params)
        node = NasNode(content={'name': layer_type.value, 'params': node_params}, nodes_from=parent_node)

        self._graph.add_node(node)
        return self

    def _make_skip_connection(self, start_node: Union[NasNode, GraphNode], end_node: Union[NasNode, GraphNode],
                              input_stride: List[int]):
        downsample = True if input_stride == [2, 2] else False
        if downsample:
            self._add_node(LayersPoolEnum.conv2d_1x1,
                           additional_layer_params={'activation': 'relu', 'conv_strides': [2, 2],
                                                    'neurons': end_node.parameters['neurons']},
                           parent_node=start_node)
            start_node = self._graph.nodes[-1]
        end_node.nodes_from.append(start_node)
        return self

    def _add_block(self, input_shape: int, blocks_num: int, downsample: bool = False):
        skip_connection_end = None
        for step in range(blocks_num):
            skip_connection_start = self._graph.nodes[-1] if not skip_connection_end else skip_connection_end
            conv_stride = [2, 2] if step == 0 and downsample else [1, 1]

            self._add_node(self._k_shape[self.resnet_type][0],
                           additional_layer_params={'activation': 'relu', 'conv_strides': conv_stride,
                                                    'neurons': input_shape}, parent_node=skip_connection_start)
            self._add_node(self._k_shape[self.resnet_type][1],
                           additional_layer_params={'activation': 'relu', 'conv_strides': [1, 1],
                                                    'neurons': input_shape})
            if self.resnet_type in ['resnet_50', 'resnet_101', 'resnet_152']:
                self._add_node(self._k_shape[self.resnet_type][2],
                               additional_layer_params={'activation': 'relu', 'conv_strides': [1, 1],
                                                        'neurons': input_shape})
            skip_connection_end = self._graph.nodes[-1]
            self._make_skip_connection(skip_connection_start, skip_connection_end, conv_stride)
        return self

    def build(self, resnet_type: Optional[str] = None) -> NasGraph:
        self.resnet_type = resnet_type if resnet_type else self.resnet_type
        if self.resnet_type not in self._blocks_num.keys():
            raise ValueError(f'Builder cannot build "{self.resnet_type}" model.')

        self._graph = NasGraph()
        self._add_node(LayersPoolEnum.conv2d_7x7, additional_layer_params={'activation': 'relu', 'conv_strides': [2, 2],
                                                                           'neurons': 64})
        self._add_node(LayersPoolEnum.max_pool2d, additional_layer_params={'pool_size': [3, 3], 'pool_stride': [2, 2]})
        self._add_block(64, blocks_num=self._blocks_num[self.resnet_type][0])
        self._add_block(128, blocks_num=self._blocks_num[self.resnet_type][1], downsample=True)
        self._add_block(256, blocks_num=self._blocks_num[self.resnet_type][2], downsample=True)
        self._add_block(512, blocks_num=self._blocks_num[self.resnet_type][3], downsample=True)
        self._add_node(LayersPoolEnum.flatten)
        return self._graph
