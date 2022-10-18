from copy import deepcopy
from typing import List
from functools import reduce, partial

from nas.composer.nn_composer_requirements import ConvRequirements, NNRequirements
from nas.graph.cnn.cnn_builder import ConvGraphMaker
from nas.graph.cnn.cnn_graph import NNGraph
from nas.nn import ActivationTypesIdsEnum
from nas.repository.existing_cnn_enum import CNNEnum
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.graph.node.nn_graph_node import NNNode, get_node_params_by_type


# def concat_graphs(*graph_list: NNGraph):
#     def _concat_nn_graphs(graph_1: NNGraph, graph_2: NNGraph):
#         nodes_to_add = graph_2.graph_struct
#         for node in nodes_to_add:
#             prev_node = graph_1.graph_struct[-1]
#             node.nodes_from.append(prev_node)
#             graph_1.add_node(node)
#         return graph_1
#
#     result_graph = graph_list[0]
#     graph_iterator = iter(graph_list)
#     next(graph_iterator)
#     for graph in graph_iterator:
#         result_graph = _concat_nn_graphs(result_graph, graph)
#     return result_graph

def concat_graphs(graph_1, graph_2):
    nodes_to_add = graph_2.graph_struct
    skip_connection_start = graph_1.graph_struct[-1]
    for node in nodes_to_add:
        prev_node = graph_1.graph_struct[-1]
        node.nodes_from.append(prev_node)
        graph_1.add_node(node)

    graph_1.graph_struct[-1].nodes_from.append(skip_connection_start)
    return graph_1


class CNNRepository:
    def _build_resnet_34(self):
        pass

    def architecture_by_type(self, model_type: CNNEnum):
        models = {
            CNNEnum.resnet34: self._build_resnet_34()
        }


class ResNetBuilder:
    _requirements: NNRequirements = None

    def set_requirements_for_resnet(self, requirements: NNRequirements):
        self._requirements = requirements

    def set_output_shape(self, output_shape: int) -> NNRequirements:
        # TODO add output shape check
        self._requirements.conv_requirements.max_filters = output_shape
        self._requirements.conv_requirements.min_filters = output_shape
        return self._requirements

    def set_conv_params(self, stride: int) -> NNRequirements:
        self._requirements.conv_requirements.conv_strides = [[stride, stride]]
        return self._requirements

    def set_pooling_params(self, pool_type: List[str], stride: int = 2, size: int = 2) -> NNRequirements:
        self._requirements.conv_requirements.pool_types = pool_type
        self._requirements.conv_requirements.pool_size = [size, size]
        self._requirements.conv_requirements.pool_strides = stride
        return self._requirements

    def _build_resnet_block(self,input_block: NNGraph, output_shape: int, flag: int) -> NNGraph:
        '''conv2d
        batch_norm
        relu
        conv2d
        batch_norm
        relu'''
        shortcut_node = False
        initial_struct = [LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_3x3]  # * max_possible_nodes[output_shape]

        block_requirements = self.set_output_shape(output_shape).set_conv_params(1)

        resnet_block = ConvGraphMaker(initial_struct=initial_struct,
                                      param_restrictions=block_requirements).build()
        if not flag and output_shape != 64:
            resnet_block.graph_struct[0].content['params']['conv_strides'] = [2, 2]
            # shortcut_node_name = LayersPoolEnum.conv2d_1x1
            # shortcut_node = get_node_params_by_type(shortcut_node_name, block_requirements.set_conv_params(2))
            # shortcut_node = NNNode(content={'name': shortcut_node_name.value, 'params': shortcut_node}, nodes_from=None)

        return concat_graphs(input_block, resnet_block)

    def build34(self, input_shape: float = 20, mode: str = 'RGB'):

        conv_req = ConvRequirements(input_shape=[input_shape, input_shape], color_mode=mode)

        self.set_requirements_for_resnet(
            NNRequirements(conv_requirements=conv_req, activation_types=[ActivationTypesIdsEnum.relu]))
        self._requirements.set_batch_norm_prob(1)
        input_node_params = deepcopy(self._requirements)
        input_node_params.set_pooling_params(['max_pool2d'], 2, 3).set_conv_params(2)

        graph_builder = ConvGraphMaker(initial_struct=[LayersPoolEnum.conv2d_7x7],
                                       param_restrictions=input_node_params)

        resnet_graph = graph_builder.build()

        for i in range(2):
            resnet_graph = self._build_resnet_block(resnet_graph, 64, i)
        for i in range(4):
            resnet_graph = self._build_resnet_block(resnet_graph, 128, i)
        for i in range(6):
            resnet_graph = self._build_resnet_block(resnet_graph, 256, i)
        for i in range(3):
            resnet_graph = self._build_resnet_block(resnet_graph, 512, i)

        # block_64 = [self._build_resnet_block(resnet_graph, 64, i) for i in range(2)]
        # block_128 = [self._build_resnet_block(128, i) for i in range(4)]
        # block_256 = [self._build_resnet_block(256, i) for i in range(6)]
        # block_512 = [self._build_resnet_block(512, i) for i in range(3)]
        #
        # resnet_graph = concat_graphs(resnet_graph, *block_64, *block_128, *block_256, *block_512)

        return resnet_graph


if __name__ == '__main__':
    # graph_1 = ResNetBuilder._build_resnet_block(64)
    # graph_2 = ResNetBuilder._build_resnet_block(128)
    # r_graph = concat_nn_graphs(graph_1, graph_2)
    graph = ResNetBuilder().build34()
    print('Done!')
