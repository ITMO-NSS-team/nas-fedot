from __future__ import annotations
import datetime
import pathlib
from typing import Callable, Optional, List, TYPE_CHECKING

import tensorflow
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from keras import optimizers

from nas.data import Preprocessor, setup_data, DataGenerator
from nas.graph.graph_builder import NNGraphBuilder
from nas.model.layers.keras_layers import KerasLayers
from nas.model.utils import converter
from nas.model.utils.branch_manager import GraphBranchManager
from nas.operations.evaluation.metrics.metrics import get_predictions, calculate_validation_metric
from nas.repository.layer_types_enum import LayersPoolEnum

if TYPE_CHECKING:
    from nas.graph.cnn.cnn_graph import NNGraph
    from nas.graph.node.nn_graph_node import NNNode


class ModelMaker:
    def __init__(self, input_shape: List, graph: NNGraph, converter: Callable, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self._graph_struct = converter(graph)
        self._branch_manager = GraphBranchManager()

        self._output_shape = 1 if num_classes == 2 else num_classes
        self._activation_func = 'sigmoid' if num_classes == 2 else 'softmax'
        self._loss_func = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

        self._input = tensorflow.keras.layers.Input(shape=input_shape)
        self._body = self._make_body
        self._classifier = tensorflow.keras.layers.Dense(self._output_shape, activation=self._activation_func)

    @staticmethod
    def _make_one_layer(input_layer, node: NNNode, branch_manager: GraphBranchManager, downsample: Optional[Callable]):
        layer = KerasLayers.convert_by_node_type(node=node, input_layer=input_layer, branch_manager=branch_manager)
        layer = KerasLayers.batch_norm(node=node, input_layer=layer)

        if len(node.nodes_from) > 1:
            # return
            parent_layer = branch_manager.get_parent_layer(node=node.nodes_from[1])['layer']
            if downsample and parent_layer.shape != layer.shape:
                parent_layer = downsample(parent_layer, layer.shape, node)
            layer = tensorflow.keras.layers.Add()([layer, parent_layer])

        layer = KerasLayers.activation(node=node, input_layer=layer)
        layer = KerasLayers.dropout(node=node, input_layer=layer)
        return layer

    def _make_body(self, inputs, **kwargs):
        # x = self._make_one_layer(inputs, self._graph_struct.head, self._branch_manager, None)
        x = inputs
        for node in self._graph_struct:
            x = self._make_one_layer(input_layer=x, node=node, branch_manager=self._branch_manager,
                                     downsample=KerasLayers.downsample_block)  # create layer with batch_norm
            # _update active deprecated branches after each layer creation
            self._branch_manager.add_and_update(node, x, self._graph_struct.get_children(node))
        return x

    def build(self):
        inputs = self._input
        body = self._body(inputs)
        del self._branch_manager
        self._branch_manager = None
        output = self._classifier(body)
        model = tensorflow.keras.Model(inputs=inputs, outputs=output, name='nas_model')

        return model


if __name__ == '__main__':
    from nas.graph.cnn.resnet_builder import ResNetGenerator
    import os
    import nas.data.load_images as loader
    import nas.composer.nn_composer_requirements as nas_requirements

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    dataset_path = pathlib.Path('../datasets/butterfly_cls/train')
    task = Task(TaskTypesEnum.classification)
    data = loader.NNData.data_from_folder(dataset_path, task)

    cv_folds = 2
    image_side_size = 12
    batch_size = 8
    epochs = 1
    optimization_epochs = 3
    conv_layers_pool = [LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_5x5,
                        LayersPoolEnum.conv2d_7x7]

    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    data_requirements = nas_requirements.DataRequirements(split_params={'cv_folds': cv_folds})
    conv_requirements = nas_requirements.ConvRequirements(input_shape=[image_side_size, image_side_size],
                                                          color_mode='RGB',
                                                          min_filters_num=32, max_filters_num=64,
                                                          conv_strides=[[1, 1]],
                                                          pool_size=[[2, 2]], pool_strides=[[2, 2]],
                                                          cnn_secondary=[LayersPoolEnum.max_pool2d,
                                                                         LayersPoolEnum.average_poold2])
    fc_requirements = nas_requirements.FullyConnectedRequirements(min_number_of_neurons=32,
                                                                  max_number_of_neurons=64)
    nn_requirements = nas_requirements.ModelRequirements(conv_requirements=conv_requirements,
                                                         fc_requirements=fc_requirements,
                                                         primary=conv_layers_pool,
                                                         secondary=[LayersPoolEnum.dense],
                                                         epochs=epochs, batch_size=batch_size,
                                                         max_nn_depth=1, max_num_of_conv_layers=10,
                                                         has_skip_connection=True
                                                         )
    optimizer_requirements = nas_requirements.OptimizerRequirements(opt_epochs=optimization_epochs)

    requirements = nas_requirements.NNComposerRequirements(data_requirements=data_requirements,
                                                           optimizer_requirements=optimizer_requirements,
                                                           model_requirements=nn_requirements,
                                                           timeout=datetime.timedelta(hours=20),
                                                           num_of_generations=2, early_stopping_iterations=100)

    transformations = [tensorflow.convert_to_tensor]
    data_preprocessor = Preprocessor()
    data_preprocessor.set_image_size((image_side_size, image_side_size)).set_features_transformations(transformations)

    graph_generation_function = NNGraphBuilder()
    graph_generation_function.set_builder(ResNetGenerator(param_restrictions=requirements.model_requirements))

    graph = graph_generation_function.build()
    graph.model = ModelMaker(requirements.model_requirements.conv_requirements.input_shape, graph, converter.Struct,
                             train_data.num_classes).build()

    train_data, val_data = train_test_data_setup(train_data, shuffle_flag=True)

    train_generator = setup_data(train_data, requirements.model_requirements.batch_size, data_preprocessor, 'train',
                                 DataGenerator, True)
    val_generator = setup_data(val_data, requirements.model_requirements.batch_size, data_preprocessor, 'train',
                               DataGenerator, True)

    graph.fit(train_generator, val_generator, requirements=requirements, num_classes=train_data.num_classes,
              verbose=1, optimization=False, shuffle=True)

    predicted_labels, predicted_probabilities = get_predictions(graph, test_data, data_preprocessor)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(test_data, predicted_probabilities, predicted_labels)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    print(1)
