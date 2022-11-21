import datetime
from typing import Callable, Optional, List

import tensorflow
from keras import optimizers

from nas.composer.nn_composer_requirements import NNComposerRequirements, OptimizerRequirements, DataRequirements, \
    ConvRequirements, FullyConnectedRequirements, NNRequirements
from nas.graph.node.nn_graph_node import NNNode
from nas.model import converter
from nas.model.branch_manager import GraphBranchManager
from nas.model.layers.keras_layers import KerasLayers
from nas.model.layers.keras_layers import ActivationTypesIdsEnum
from nas.repository.layer_types_enum import LayersPoolEnum

from nas.graph.cnn.cnn_graph import NNGraph


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
        layer = KerasLayers().convert_by_node_type(node=node, input_layer=input_layer, branch_manager=branch_manager)
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
        x = self._make_one_layer(inputs, self._graph_struct.head, self._branch_manager, None)
        for node in self._graph_struct:
            # for node in nodes:
            x = self._make_one_layer(input_layer=x, node=node, branch_manager=self._branch_manager,
                                     downsample=KerasLayers.downsample_block)  # create layer with batch_norm
            # _update active deprecated branches after each layer creation
            self._branch_manager.add_and_update(node, x, self._graph_struct.get_children(node))
        return x

    def build(self):
        inputs = self._input
        body = self._body(inputs)
        output = self._classifier(body)
        model = tensorflow.keras.Model(inputs=inputs, outputs=output, name='nas_model')

        model.compile(loss=self._loss_func, optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])

        return model

if __name__ == '__main__':
    from nas.graph.cnn.resnet_builder import ResNetGenerator
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
                                         min_filters=32, max_filters=64,
                                         conv_strides=[[1, 1]],
                                         pool_size=[[2, 2]], pool_strides=[[2, 2]])
    fc_requirements = FullyConnectedRequirements(min_number_of_neurons=32,
                                                 max_number_of_neurons=64)
    nn_requirements = NNRequirements(conv_requirements=conv_requirements,
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
                                          nn_requirements=nn_requirements,
                                          timeout=datetime.timedelta(hours=200),
                                          num_of_generations=1)
    graph = ResNetGenerator(requirements.nn_requirements).build()

    print(tensorflow.config.list_physical_devices('GPU'))
    dataset = tensorflow.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = dataset
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    s = ModelMaker(x_train.shape[1:], graph, converter.Struct, 10).build()

    print(s.summary())

    s.fit(x_train, y_train, epochs=2)

    print(1)
