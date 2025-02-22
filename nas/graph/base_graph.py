import gc
import json
import os
import pathlib
from typing import List, Union, Optional

from fedot.core.data.data import OutputData
from golem.core.optimisers.graph import OptGraph
from golem.serializers import Serializer
from golem.visualisation.graph_viz import NodeColorType

from nas.graph.graph_utils import probs2labels
from nas.graph.node.nas_graph_node import NasNode
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

    def predict(self, test_data, batch_size=1, output_mode: str = 'default', **kwargs) -> OutputData:
        if not self.model_interface:
            raise AttributeError("Graph doesn't have a model yet")

        is_multiclass = test_data.num_classes > 2
        predictions = self.model_interface.validate(test_data, batch_size)
        if output_mode == 'labels':
            predictions = probs2labels(predictions, is_multiclass)

        return OutputData(idx=test_data.idx, features=test_data.features, predict=predictions,
                          task=test_data.task, data_type=test_data.data_type)

    def fit_with_cache(self, *args, **kwargs):
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

    @staticmethod
    def release_memory(**kwargs):
        pass

    def unfit(self, **kwargs):
        pass

    def get_singleton_node_by_name(self, node_name) -> NasNode:
        for node in self.nodes:
            if node.name == node_name:
                return node
        raise ValueError("One node with name {} should be in the graph".format(node_name))

    def get_input_node(self):
        """ Such that nodes_from is empty """
        for node in self.nodes:
            if not node.nodes_from:
                return node
        raise ValueError("Input node not found")
