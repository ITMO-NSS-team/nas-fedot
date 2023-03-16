from copy import deepcopy

from golem.core.dag.graph_node import GraphNode

from nas.composer.nn_composer_requirements import *
from nas.graph.cnn_graph import NasGraph
from nas.graph.graph_builder.base_graph_builder import GraphGenerator
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


# TODO rework resnet builder
class ResNetGenerator(GraphGenerator):
    """
    This class generates ResNet-like graph as initial assumption
    """

    def __init__(self, model_requirements: ModelRequirements):
        self._model_requirements = deepcopy(model_requirements)
        self._model_requirements.conv_requirements.force_output_shape(64)
        self._model_requirements.fc_requirements.set_batch_norm_prob(1)

    def _add_node(self, node_to_add: LayersPoolEnum, node_requirements: ModelRequirements,
                  parent_node: List[NasNode] = None, stride: int = None, pool_size: int = None,
                  pool_stride: int = None) -> NasNode:
        layer_requirements = deepcopy(node_requirements)
        if stride:
            layer_requirements.conv_requirements.force_conv_params(stride)
        elif pool_size or pool_stride:
            layer_requirements.conv_requirements.force_pooling_size(pool_size)
            layer_requirements.conv_requirements.force_pooling_stride(pool_stride)
            layer_requirements.conv_requirements.set_pooling_params(size=pool_size, stride=pool_stride)

        node_params = get_node_params_by_type(node_to_add, layer_requirements)
        node = NasNode(content={'name': node_to_add.value, 'params': node_params}, nodes_from=parent_node)
        return node

    def _add_to_block(self, block: NasGraph, node_to_add: LayersPoolEnum, requirements: ModelRequirements,
                      parent_node: Optional[List[Union[GraphNode, NasNode]]], stride: int):
        node_to_add = self._add_node(node_to_add, parent_node=parent_node, node_requirements=requirements,
                                     stride=stride)
        block.add_node(node_to_add)

    def _growth_one_block(self, graph: NasGraph, output_shape: int, number_of_sub_blocks: int):
        for i in range(number_of_sub_blocks):
            stride = 2 if (i == 0 and output_shape != 64) else 1
            block_to_add = self._generate_sub_block(output_shape, stride)
            nodes_to_add = block_to_add.graph_struct
            skip_connection_start = graph.graph_struct[-1]
            for node in nodes_to_add:
                if not node.nodes_from:
                    prev_node = graph.graph_struct[-1]
                    node.nodes_from.append(prev_node)
                graph.add_node(node)
            graph.graph_struct[-1].nodes_from.append(skip_connection_start)

    def _generate_sub_block(self, output_shape: int, stride: int) -> NasGraph:
        res_block = NasGraph()
        block_requirements = deepcopy(self._model_requirements)
        block_requirements.conv_requirements.force_output_shape(output_shape)

        self._add_to_block(res_block, LayersPoolEnum.conv2d_3x3, block_requirements, None, stride)
        self._add_to_block(res_block, LayersPoolEnum.conv2d_3x3, block_requirements, [res_block.graph_struct[-1]],
                           1)
        return res_block

    def build(self) -> NasGraph:
        self._model_requirements.max_num_of_conv_layers = 35
        self._model_requirements.max_nn_depth = 3

        resnet = NasGraph()

        node = self._add_node(LayersPoolEnum.conv2d_7x7, node_requirements=self._model_requirements, stride=2)
        resnet.add_node(node)

        pooling = self._add_node(LayersPoolEnum.max_pool2d, node_requirements=self._model_requirements,
                                 pool_size=3, pool_stride=2, parent_node=[node])
        resnet.add_node(pooling)
        self._growth_one_block(resnet, 64, 3)
        self._growth_one_block(resnet, 128, 4)
        self._growth_one_block(resnet, 256, 6)
        self._growth_one_block(resnet, 512, 3)
        pooling = self._add_node(LayersPoolEnum.average_poold2, node_requirements=self._model_requirements,
                                 parent_node=[resnet.graph_struct[-1]])
        resnet.add_node(pooling)
        flatten = self._add_node(LayersPoolEnum.flatten, node_requirements=self._model_requirements,
                                 parent_node=[pooling])
        resnet.add_node(flatten)
        resnet.input_shape = self._model_requirements.input_shape
        return resnet


class _ResNetBuilder(GraphGenerator):
    def __init__(self, *args, **kwargs):
        self.resnet_type: Optional[str] = None
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

    def build(self, resnet_type) -> NasGraph:
        self.resnet_type = resnet_type
        self._graph = NasGraph()
        self._add_node(LayersPoolEnum.conv2d_7x7, additional_layer_params={'activation': 'relu', 'conv_strides': [2, 2],
                                                                           'neurons': 64})
        self._add_node(LayersPoolEnum.max_pool2d, additional_layer_params={'pool_size': [3, 3], 'pool_stride': [2, 2]})
        self._add_block(64, blocks_num=self._blocks_num[resnet_type][0])
        self._add_block(128, blocks_num=self._blocks_num[resnet_type][1], downsample=True)
        self._add_block(256, blocks_num=self._blocks_num[resnet_type][2], downsample=True)
        self._add_block(512, blocks_num=self._blocks_num[resnet_type][3], downsample=True)
        self._add_node(LayersPoolEnum.flatten)
        return self._graph

        # {'activation': 'relu', 'conv_strides': [2, 2], 'neurons': 64, 'momentum': 0.99, 'epsilon': 0.001,
        #  'kernel_size': [7, 7]}


if __name__ == '__main__':
    resnet = _ResNetBuilder().build('resnet_34')
    resnet.show()
    print(1)
