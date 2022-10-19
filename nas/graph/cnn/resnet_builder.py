from copy import deepcopy

from nas.composer.nn_composer_requirements import *
from nas.graph.cnn.cnn_builder import ConvGraphMaker
from nas.graph.cnn.cnn_graph import NNGraph
from nas.nn import ActivationTypesIdsEnum, build_nn_from_graph
from nas.repository.existing_cnn_enum import CNNEnum
from nas.repository.layer_types_enum import LayersPoolEnum


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

    def _build_resnet_block(self, input_block: NNGraph, output_shape: int, flag: int) -> NNGraph:
        '''conv2d
        batch_norm
        relu
        conv2d
        batch_norm
        relu'''
        shortcut_node = False
        initial_struct = [LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_3x3]  # * max_possible_nodes[output_shape]

        block_requirements = deepcopy(self._requirements)
        block_requirements.set_output_shape(output_shape).set_conv_params(1).set_pooling_params(None, None, None)

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

        resnet_graph.graph_struct[-1].content['params']['pool_type'] = 'average_pool2d'
        resnet_graph.graph_struct[-1].content['params']['pool_strides'] = None
        resnet_graph.graph_struct[-1].content['params']['pool_size'] = [2, 2]

        return resnet_graph


if __name__ == '__main__':
    graph = ResNetBuilder().build34()

    cv_folds = 2
    image_side_size = 20
    batch_size = 8
    epochs = 1
    optimization_epochs = 1
    # conv_layers_pool = [LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_5x5,

    data_requirements = DataRequirements(split_params={'cv_folds': cv_folds})
    conv_requirements = ConvRequirements(input_shape=[image_side_size, image_side_size],
                                         color_mode='RGB',
                                         min_filters=32, max_filters=64,
                                         conv_strides=[[1, 1]],
                                         pool_size=[[2, 2]], pool_strides=[[2, 2]],
                                         pool_types=['max_pool2d', 'average_pool2d'])
    fc_requirements = FullyConnectedRequirements(min_number_of_neurons=32,
                                                 max_number_of_neurons=64)
    nn_requirements = NNRequirements(conv_requirements=conv_requirements,
                                     fc_requirements=fc_requirements,
                                     primary=[LayersPoolEnum.conv2d_3x3],
                                     secondary=[LayersPoolEnum.dense],
                                     epochs=epochs, batch_size=batch_size,
                                     max_nn_depth=1, max_num_of_conv_layers=10,
                                     has_skip_connection=True
                                     )
    optimizer_requirements = OptimizerRequirements(opt_epochs=optimization_epochs)

    requirements = NNComposerRequirements(data_requirements=data_requirements,
                                          optimizer_requirements=optimizer_requirements,
                                          nn_requirements=nn_requirements,
                                          timeout=datetime.timedelta(hours=200),
                                          num_of_generations=1)

    build_nn_from_graph(graph, 75, requirements)
    print('Done!')
