import gc
import json
import os
import pathlib
from typing import List, Union, Optional

import keras.backend
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.engine.functional import Functional

from fedot.core.dag.graph_node import GraphNode
from fedot.core.data.data import OutputData
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.serializers import Serializer
from fedot.core.utils import DEFAULT_PARAMS_STUB
from fedot.core.visualisation.graph_viz import NodeColorType

from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.graph.node.nn_graph_node import NNNode
from nas.operations.evaluation.callbacks.bad_performance_callback import CustomCallback
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.model.nn.tf_model import ModelMaker
from nas.model.utils import converter
# hotfix
from nas.utils.default_parameters import default_nodes_params
from nas.utils.utils import set_root, seed_all, project_root, clear_keras_session

set_root(project_root())
convolutional_types = (LayersPoolEnum.conv2d, LayersPoolEnum.dilation_conv2d)
seed_all(1)


class NNNodeOperatorAdapter:
    def adapt(self, adaptee) -> OptNode:
        adaptee.__class__ = OptNode
        return adaptee

    def restore(self, node) -> NNNode:
        obj = node
        obj.__class__ = NNNode
        if obj.content['params'] == DEFAULT_PARAMS_STUB:
            node_name = obj.content.get('name')
            obj.content = default_nodes_params[node_name]
        return obj


class NNGraph(OptGraph):

    def __init__(self, nodes=(), model=None):
        super().__init__(nodes)
        self._model = model
        self._input_shape = None
        self._weights = None

    def __repr__(self):
        return f"{self.depth}:{self.length}:{self.cnn_depth[0]}"

    def __eq__(self, other) -> bool:
        return self is other

    def show(self, save_path: Optional[Union[os.PathLike, str]] = None, engine: str = 'matplotlib',
             node_color: Optional[NodeColorType] = None, dpi: int = 100,
             node_size_scale: float = 1.0, font_size_scale: float = 1.0, edge_curvature_scale: float = 1.0):
        super().show(save_path, 'pyvis', node_color, dpi, node_size_scale, font_size_scale, edge_curvature_scale)

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, val):
        self._input_shape = val

    @property
    def model(self) -> Functional:
        return self._model

    @model.setter
    def model(self, value: Union[Functional]):
        self._model = value

    @model.deleter
    def model(self):
        del self._model
        self._model = None
        gc.collect()

    @property
    def free_nodes(self):
        free_nodes = []
        skip_connections_start_nodes = set()
        for node in self.graph_struct[::-1]:
            if len(skip_connections_start_nodes) == 0:
                free_nodes.append(node)
            is_skip_connection_end = len(node.nodes_from) > 1
            if is_skip_connection_end:
                skip_connections_start_nodes.update(node.nodes_from[1:])
            if node in skip_connections_start_nodes:
                skip_connections_start_nodes.remove(node)
        return free_nodes

    @property
    def _node_adapter(self):
        return NNNodeOperatorAdapter()

    @property
    def cnn_depth(self):
        flatten_id = [ind for ind, node in enumerate(self.graph_struct) if node.content['name'] == 'flatten']
        return flatten_id

    def get_trainable_params(self):
        total_params = 0
        output_shape = self.input_shape
        for node in self.graph_struct:
            node.input_shape = output_shape
            layer_params = node.node_params
            total_params += layer_params
            output_shape = node.output_shape
        return total_params

    def fit(self, train_generator, val_generator, requirements: NNComposerRequirements, num_classes: int,
            verbose='auto', optimization: bool = True, shuffle: bool = False, **kwargs):
        loss_func = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

        epochs = requirements.optimizer_requirements.opt_epochs if optimization else requirements.nn_requirements.epochs
        # lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-2, decay_steps=epochs, decay_rate=0.96, staircase=True)
        lr = 1e-4
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        batch_size = requirements.nn_requirements.batch_size
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        model_metrics = tensorflow.keras.metrics.CategoricalAccuracy() if num_classes > 2 else \
            tensorflow.keras.metrics.Accuracy()
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                           verbose=1, min_delta=1e-4, mode='min')
        callbacks_list = [early_stopping, reduce_lr_loss]
        if optimization:
            callbacks_list.append(CustomCallback())

        tf.keras.backend.clear_session()
        input_shape = requirements.nn_requirements.conv_requirements.input_shape
        self.model = ModelMaker(input_shape, self, converter.Struct, num_classes).build()
        self.model.compile(loss=loss_func, optimizer=optimizer, metrics=[model_metrics])
        self.model.fit(train_generator, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       validation_data=val_generator, shuffle=shuffle, callbacks=callbacks_list)

    def predict(self, test_data, batch_size=1, output_mode: str = 'default', **kwargs):
        if not self.model:
            raise AttributeError("Graph doesn't have a model yet")

        is_multiclass = test_data.num_classes > 2

        predictions = self.model.predict(test_data, batch_size)
        if output_mode == 'labels':
            predictions = self._probs2labels(predictions, is_multiclass)

        return OutputData(idx=test_data.idx, features=test_data.features, predict=predictions,
                          task=test_data.task, data_type=test_data.data_type)

    @staticmethod
    def _probs2labels(predictions, is_multiclass):
        if is_multiclass:
            return np.argmax(predictions, axis=-1)
        else:
            return np.where(predictions > .5, 1, 0)

    def fit_with_cache(self, *args, **kwargs):
        # TODO
        return False

    def save(self, path: Union[str, os.PathLike, pathlib.Path]):
        """Save graph and fitted model to json and mdf5 formats"""
        full_path = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        full_path = full_path.resolve()

        model_path = full_path / 'fitted_model.h5'
        if self.model:
            self.model.save(model_path)
            self.model = None

        graph = json.dumps(self, indent=4, cls=Serializer)
        with open(full_path / 'graph.json', 'w') as f:
            f.write(graph)

    @staticmethod
    def load(path: Union[str, os.PathLike, pathlib.Path]):
        """load graph from json file"""
        with open(path, 'r') as json_file:
            json_data = json_file.read()
            return json.loads(json_data, cls=Serializer)

    @property
    def graph_struct(self) -> List[Union[NNNode, GraphNode]]:
        if 'conv' in self.nodes[0].content['name']:
            return self.nodes
        else:
            return self.nodes[::-1]

    @staticmethod
    def release_memory(**kwargs):
        clear_keras_session(**kwargs)
        # gc.collect()

    def unfit(self, **kwargs):
        if self.model:
            del self.model
        if hasattr(self, '_weights'):
            del self._weights
        keras.backend.clear_session()

        # clear_keras_session(**kwargs)
        # gc.collect()

    def reset_weights(self):
        pass
        # for ix, l in enumerate(self.model.layers):
        #     if hasattr(l, "kernel_initializer"):
        #         l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        #     if hasattr(l, "bias_initializer"):
        #         l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        #     if hasattr(l, "recurrent_initializer"):
        #         l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
