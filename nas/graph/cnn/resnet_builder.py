from copy import deepcopy
from typing import Union

from golem.core.dag.graph_node import GraphNode

from nas.composer.nn_composer_requirements import *
from nas.graph.grpah_generator import GraphGenerator
from nas.graph.node.nn_graph_node import NNNode, get_node_params_by_type
from nas.repository.existing_cnn_enum import CNNEnum
from nas.repository.layer_types_enum import LayersPoolEnum, ActivationTypesIdsEnum
from nas.graph.cnn.cnn_graph import NNGraph


class CNNRepository:
    def _build_resnet_34(self):
        pass

    def architecture_by_type(self, model_type: CNNEnum):
        models = {
            CNNEnum.resnet34: self._build_resnet_34()
        }


class ResNetGenerator(GraphGenerator):
    def __init__(self, model_requirements: ModelRequirements):
        self._model_requirements = deepcopy(model_requirements)
        self._model_requirements.conv_requirements.set_output_shape(64)  # TODO fix
        self._model_requirements.fc_requirements.set_batch_norm_prob(1)

    def _add_node(self, node_to_add: LayersPoolEnum, node_requirements: ModelRequirements,
                  parent_node: List[NNNode] = None, stride: int = None, pool_size: int = None,
                  pool_stride: int = None) -> NNNode:
        layer_requirements = deepcopy(node_requirements)
        if stride:
            layer_requirements.conv_requirements.set_conv_params(stride)
        elif pool_size or pool_stride:
            layer_requirements.conv_requirements.set_pooling_params(pool_size, pool_stride)

        node_params = get_node_params_by_type(node_to_add, layer_requirements)
        node = NNNode(content={'name': node_to_add.value, 'params': node_params}, nodes_from=parent_node)
        return node

    def _add_to_block(self, block: NNGraph, node_to_add, requirements: ModelRequirements,
                      parent_node: List[Union[GraphNode, NNNode]], stride: int):
        node_to_add = self._add_node(node_to_add, parent_node=parent_node, node_requirements=requirements,
                                     stride=stride)
        block.add_node(node_to_add)

    def _growth_one_block(self, graph: NNGraph, output_shape: int, number_of_sub_blocks: int):
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

    def _generate_sub_block(self, output_shape: int, stride: int) -> NNGraph:
        res_block = NNGraph()
        block_requirements = deepcopy(self._model_requirements)
        block_requirements.conv_requirements.set_output_shape(output_shape)

        self._add_to_block(res_block, LayersPoolEnum.conv2d_3x3, block_requirements, None, stride)
        self._add_to_block(res_block, LayersPoolEnum.conv2d_3x3, block_requirements, [res_block.graph_struct[-1]],
                           1)
        return res_block

    def build(self) -> NNGraph:
        self._model_requirements.max_num_of_conv_layers = 35
        self._model_requirements.max_nn_depth = 3

        resnet = NNGraph()

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


if __name__ == '__main__':
    cv_folds = 2
    image_side_size = 256
    batch_size = 8
    epochs = 1
    optimization_epochs = 1
    # conv_layers_pool = [LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_5x5,

    data_requirements = DataRequirements(split_params={'cv_folds': cv_folds})
    conv_requirements = ConvRequirements(input_shape=[image_side_size, image_side_size],
                                         cnn_secondary=[LayersPoolEnum.max_pool2d, LayersPoolEnum.average_poold2],
                                         color_mode='RGB',
                                         min_number_of_neurons=32, max_number_of_neurons=64,
                                         conv_strides=[[1, 1]],
                                         pool_size=[[2, 2]], pool_strides=[[2, 2]])
    fc_requirements = BaseLayerRequirements(min_number_of_neurons=32,
                                            max_number_of_neurons=64)
    nn_requirements = ModelRequirements(conv_requirements=conv_requirements,
                                        fc_requirements=fc_requirements,
                                        primary=[LayersPoolEnum.conv2d_3x3],
                                        secondary=[LayersPoolEnum.dense],
                                        epochs=epochs, batch_size=batch_size,
                                        max_nn_depth=1, max_num_of_conv_layers=10,
                                        has_skip_connection=True, activation_types=[ActivationTypesIdsEnum.relu]
                                        )
    optimizer_requirements = OptimizerRequirements(opt_epochs=optimization_epochs)

    requirements = NNComposerRequirements(data_requirements=data_requirements,
                                          optimizer_requirements=optimizer_requirements,
                                          model_requirements=nn_requirements,
                                          timeout=datetime.timedelta(hours=200),
                                          num_of_generations=1)
    graph = ResNetGenerator(requirements.model_requirements).build()


    print('Done!')
