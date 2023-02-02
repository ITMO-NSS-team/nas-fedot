import pathlib
from typing import Optional, Union, List, Tuple

import tensorflow as tf
from fedot.core.repository.tasks import Task, TaskTypesEnum

import nas
from nas.composer.nn_composer_requirements import load_default_requirements
from nas.data import Preprocessor, KerasDataset
from nas.data.dataset.builder import BaseNasDatasetBuilder
from nas.graph.cnn_graph import NasGraph
from nas.graph.graph_builder.base_graph_builder import BaseGraphBuilder
from nas.graph.graph_builder.resnet_builder import ResNetGenerator
from nas.model.tensorflow.future.model_skeleton import ModelSkeleton
from nas.model.tensorflow.future.tf_activations_enum import KerasActivations
from nas.model.tensorflow.future.tf_layer_initializer import LayerInitializer
from nas.model.utils.branch_manager import GraphBranchManager


def test_model():
    import cv2


    requirements = load_default_requirements()
    graph_generation_function = BaseGraphBuilder()
    graph_generation_function.set_builder(ResNetGenerator(model_requirements=requirements.model_requirements))
    graph = graph_generation_function.build()

    model = NasModel(graph, [224, 224, 3], n_classes=75)

    dataset_path = pathlib.Path('/home/staeros/work/datasets/butterfly_cls/train')
    task = Task(TaskTypesEnum.classification)
    data = nas.data.nas_data.BaseNasImageData.data_from_folder(dataset_path, task)

    data_preprocessor = Preprocessor((224, 224))

    data_transformer = BaseNasDatasetBuilder(dataset_cls=KerasDataset,
                                             batch_size=16,
                                             shuffle=True).set_data_preprocessor(data_preprocessor)
    data_generator = data_transformer.build(data, mode='train')

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=[tf.metrics.Accuracy()])
    model.fit(data_generator)


    print('Done')


class NasModel(tf.keras.Model):
    def __init__(self, graph: NasGraph, input_shape: Union[List, Tuple], n_classes: Optional[int] = None):
        super().__init__()
        # self._input_layer = tf.keras.layers.Input(shape=input_shape, name='input', dtype=tf.float32)
        self.model_structure = ModelSkeleton(graph)
        self.model_layer = self.model_structure.model_layers
        output_shape = 1 if not n_classes else n_classes
        self.classifier = tf.keras.layers.Dense(output_shape, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # inputs = self._input_layer(inputs)
        for node in self.model_structure.model_nodes:
            layer = self.model_structure.model_struct[node]
            inputs = layer(inputs)
            if node.content['params'].get('epsilon'):
                batch_norm = LayerInitializer.batch_norm(node)
                inputs = batch_norm(inputs)
            if len(node.nodes_from) > 1:
                parent_layer = self.model_structure.branch_manager.get_parent_layer(node=node.nodes_from[1])['layer']
                inputs = tf.keras.layers.Add()([inputs, parent_layer])
            if node.content['params'].get('activation_func'):
                activation_func = tf.keras.layers.Activation(node.content['params'].get('activation_func'))
                inputs = activation_func(inputs)

            self.model_structure.branch_manager.add_and_update(node, inputs, self.model_structure.get_children(node))

        output = self.classifier(inputs)
        return output



