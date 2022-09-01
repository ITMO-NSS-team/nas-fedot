import pathlib
from typing import List, Union

import json
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.engine.functional import Functional

from fedot.core.data.data import OutputData
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.serializers import Serializer
from fedot.core.utils import DEFAULT_PARAMS_STUB

from nas.graph.cnn.cnn_graph_node import NNNode
from nas.composer.nn_composer_requirements import NNComposerRequirements

# hotfix
from nas.utils.var import default_nodes_params
from nas.utils.utils import set_root, seed_all, project_root
from nas.model.nn.keras_graph_converter import build_nn_from_graph

set_root(project_root())
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
    # TODO add parent class NNGraph inherited from OptGraph with base nn logic e.g. fit, predict, save etc
    #  And create child class for CNN's with CNN specific parameters e.g. number of conv_layers.

    def __init__(self, nodes=(), model=None):
        super().__init__(nodes)
        self._model = model

    def __repr__(self):
        return f"{self.depth}:{self.length}:{self.cnn_depth}"

    def __eq__(self, other) -> bool:
        return self is other

    @property
    def model(self) -> Functional:
        return self._model

    @model.setter
    def model(self, value: Union[Functional]):
        self._model = value

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
        for idx, node in enumerate(self.graph_struct):
            if node.content['name'] == 'flatten':
                return idx

    def fit(self, train_generator, val_generator, requirements: NNComposerRequirements, num_classes: int,
            verbose='auto', optimization: bool = True, shuffle: bool = False):

        epochs = requirements.optimizer_requirements.opt_epochs if optimization else requirements.nn_requirements.epochs
        batch_size = requirements.nn_requirements.batch_size

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
                                           verbose=1, min_delta=1e-4, mode='min')
        callbacks_list = [early_stopping, reduce_lr_loss]

        if not self.model:
            build_nn_from_graph(self, num_classes, requirements)

        self.model.fit(train_generator, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       validation_data=val_generator, shuffle=shuffle, callbacks=callbacks_list)

        return self

    def predict(self, test_data, batch_size=1, output_mode: str = 'default', **kwargs):
        if not self.model:
            raise AttributeError("Graph doesn't have a model yet")
        # hotfix
        is_multiclass = test_data.num_classes > 2

        # test_generator = temporal_setup_data(test_data, batch_size, self._preprocessor, DataGenerator)
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
    def graph_struct(self) -> List:
        if self.nodes[0].content['name'] != 'conv2d':
            return self.nodes[::-1]
        else:
            return self.nodes
