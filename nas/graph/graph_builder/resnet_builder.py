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
            layer_requirements.conv_requirements.set_pooling_params(pool_size, pool_stride)

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
