import json
import os
import pathlib

from fedot.core.data.data import InputData, OutputData
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.serializers import Serializer
from fedot.core.utils import DEFAULT_PARAMS_STUB

from nas.composer.cnn.cnn_graph_node import CNNNode
from nas.nn.nas_keras_eval import create_nn_model, keras_model_fit, keras_model_predict

# hotfix
from nas.utils.var import default_nodes_params


class CNNGraph(OptGraph):
    # Temporal fix
    INDIVIDUAL = 1
    GENERATION = 1

    def __init__(self, nodes=None, fitted_model=None):
        super().__init__(nodes)
        self.model = fitted_model

    def __repr__(self):
        return f"{self.depth}:{self.length}:{self.cnn_depth}"

    def __eq__(self, other) -> bool:
        return self is other

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
    def has_skips(self):
        return len(self.free_nodes) != len(self.nodes)

    @property
    def _node_adapter(self):
        return NNNodeOperatorAdapter()

    @property
    def cnn_depth(self):
        for idx, node in enumerate(self.graph_struct):
            if node.content['name'] == 'flatten':
                return idx

    def fit(self, input_data: InputData, verbose=False, requirements=None, train_epochs: int = None):
        train_epochs = requirements.epochs if train_epochs is None else train_epochs
        if not self.model:
            self.model = create_nn_model(self, requirements.input_shape, input_data.num_classes)
        train_predicted = keras_model_fit(self.model, input_data, verbose=verbose, batch_size=requirements.batch_size,
                                          epochs=train_epochs, graph=self, ind=CNNGraph.INDIVIDUAL,
                                          gen=CNNGraph.GENERATION)
        CNNGraph.INDIVIDUAL += 1
        if CNNGraph.INDIVIDUAL > requirements.pop_size:
            CNNGraph.GENERATION += 1
        return train_predicted

    def predict(self, input_data: InputData, output_mode: str = 'default', is_multiclass: bool = False) -> OutputData:
        evaluation_result = keras_model_predict(self.model, input_data, output_mode, is_multiclass=is_multiclass)
        return evaluation_result

    def save(self, path: str = None):
        res = json.dumps(self, indent=4, cls=Serializer)
        with open(os.path.join(path, 'graph.json'), 'w') as f:
            f.write(res)

    @staticmethod
    def load(path: str):
        """load graph from json file"""
        with open(path, 'r') as json_file:
            json_data = json_file.read()
            return json.loads(json_data, cls=Serializer)

    @property
    def graph_struct(self):
        if self.nodes[0].content['name'] != 'conv2d':
            return self.nodes[::-1]
        else:
            return self.nodes


class NNNodeOperatorAdapter:
    def adapt(self, adaptee) -> OptNode:
        adaptee.__class__ = OptNode
        return adaptee

    def restore(self, node) -> CNNNode:
        obj = node
        obj.__class__ = CNNNode
        if obj.content['params'] == DEFAULT_PARAMS_STUB:
            node_name = obj.content.get('name')
            obj.content = default_nodes_params[node_name]
        return obj
