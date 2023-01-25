import gc
import json
import os
import pathlib
from typing import List, Union, Optional, Tuple, Callable

import keras.backend
import tensorflow as tf
from fedot.core.data.data import OutputData
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.graph import OptGraph
from golem.serializers import Serializer
from golem.visualisation.graph_viz import NodeColorType
from tensorflow.python.keras.engine.functional import Functional

from nas.graph.node.nn_graph_node import NNNode
from nas.graph.utils import probs2labels
from nas.model.nn.tf_model import KerasModelMaker
from nas.model.utils import converter
# hotfix
from nas.utils.utils import seed_all, clear_keras_session

seed_all(1)


class NasGraph(OptGraph):
    def __init__(self, nodes: Optional[List[NNNode]] = ()):
        super().__init__(nodes)
        self._model = None

    def __repr__(self):
        return f"{self.depth}:{self.length}:{self.cnn_depth[0]}"

    def __eq__(self, other) -> bool:
        return self is other

    def show(self, save_path: Optional[Union[os.PathLike, str]] = None, engine: str = 'pyvis',
             node_color: Optional[NodeColorType] = None, dpi: int = 100,
             node_size_scale: float = 1.0, font_size_scale: float = 1.0, edge_curvature_scale: float = 1.0):
        super().show(save_path, engine, node_color, dpi, node_size_scale, font_size_scale, edge_curvature_scale)

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
    def cnn_depth(self):
        flatten_id = [ind for ind, node in enumerate(self.graph_struct) if node.content['name'] == 'flatten']
        return flatten_id

    def compile_model(self, input_shape: Union[List[int], Tuple[int]], loss_function: str, metrics: List,
                      model_builder: Callable = KerasModelMaker, n_classes: Optional[int] = None,
                      learning_rate: float = 1e-3, optimizer: Callable = None):
        optimizer = optimizer(learning_rate=learning_rate)

        self.model = model_builder(input_shape, self, converter.GraphStruct, n_classes).build()
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
        return self

    def fit(self, train_generator, validation_generator, epoch_num: int = 5, batch_size: int = 32,
            callbacks: List = None, verbose='auto', **kwargs):
        tf.keras.backend.clear_session()
        self.model.fit(train_generator, batch_size=batch_size, epochs=epoch_num, verbose=verbose,
                       validation_data=validation_generator, callbacks=callbacks)

    def predict(self, test_data, batch_size=1, output_mode: str = 'default', **kwargs) -> OutputData:
        if not self.model:
            raise AttributeError("Graph doesn't have a model yet")

        is_multiclass = test_data.num_classes > 2
        predictions = self.model.predict(test_data, batch_size)
        if output_mode == 'labels':
            predictions = probs2labels(predictions, is_multiclass)

        return OutputData(idx=test_data.idx, features=test_data.features, predict=predictions,
                          task=test_data.task, data_type=test_data.data_type)

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
