import gc
import json
import os
import pathlib
from typing import List, Union, Optional

from fedot.core.data.data import OutputData
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.graph import OptGraph
from golem.serializers import Serializer
from golem.visualisation.graph_viz import NodeColorType

from nas.graph.graph_utils import probs2labels
from nas.graph.node.nas_graph_node import NasNode
# from nas.model.model_interface import BaseModelInterface
# hotfix
from nas.utils.utils import seed_all

seed_all(1)


class NasGraph(OptGraph):
    def __init__(self, nodes: Optional[List[NasNode]] = ()):
        super().__init__(nodes)
        self._model_interface = None

    def __repr__(self):
        return f"{self.depth}:{self.length}:{self.cnn_depth[0]}"

    def __eq__(self, other) -> bool:
        return self is other

    def show(self, save_path: Optional[Union[os.PathLike, str]] = None, engine: str = 'pyvis',
             node_color: Optional[NodeColorType] = None, dpi: int = 100,
             node_size_scale: float = 1.0, font_size_scale: float = 1.0, edge_curvature_scale: float = 1.0):
        super().show(save_path, engine, node_color, dpi, node_size_scale, font_size_scale, edge_curvature_scale)

    @property
    def model_interface(self):
        return self._model_interface

    @model_interface.setter
    def model_interface(self, value):
        self._model_interface = value

    # @property
    # def cnn_depth(self):
    #     flatten_id = [ind for ind, node in enumerate(self.graph_struct) if node.content['name'] == 'flatten']
    #     return flatten_id

    def compile_model(self, output_shape: int = 1, **additional_params):
        # self.model_interface = model_builder(input_shape, self, ModelStructure, n_classes).build()
        # self.model_interface.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
        self.model_interface.compile_model(self, output_shape, **additional_params)
        return self

    def fit(self, train_data, val_data, epochs: int = 5, batch_size: int = 32,
            callbacks: List = None, verbose='auto', **kwargs):
        # tf.keras.backend.clear_session()
        self.model_interface.fit(train_data, val_data, batch_size=batch_size, epochs=epochs, verbose=verbose,
                                 callbacks=callbacks)
        return self

    def predict(self, test_data, batch_size=1, output_mode: str = 'default', **kwargs) -> OutputData:
        if not self.model_interface:
            raise AttributeError("Graph doesn't have a model yet")

        is_multiclass = test_data.num_classes > 2
        predictions = self.model_interface.predict(test_data, batch_size)
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
        if self.model_interface:
            self.model_interface.save(model_path)
            del self._model_interface
            self.model_interface = None

        graph = json.dumps(self, indent=4, cls=Serializer)
        with open(full_path / 'graph.json', 'w') as f:
            f.write(graph)

    @staticmethod
    def load(path: Union[str, os.PathLike, pathlib.Path]):
        """load graph from json file"""
        with open(path, 'r') as json_file:
            json_data = json_file.read()
            return json.loads(json_data, cls=Serializer)

    # @property
    # def graph_struct(self) -> List[Union[NasNode, GraphNode]]:
    #     if 'conv' in self.nodes[0].content['name']:
    #         return self.nodes
    #     else:
    #         return self.nodes[::-1]

    @staticmethod
    def release_memory(**kwargs):
        pass
        # clear_keras_session(**kwargs)
        # gc.collect()

    def unfit(self, **kwargs):
        del self._model_interface
        self.model_interface = None
        # keras.backend.clear_session()
        gc.collect()
