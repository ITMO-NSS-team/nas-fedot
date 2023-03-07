from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Callable, Optional, List, TYPE_CHECKING

import tensorflow
from fedot.core.data.data import InputData

from nas.model.tensorflow.future.tf_layer_initializer import LayerInitializer
from nas.model.tensorflow.tf_layers import KerasLayers
from nas.model.utils.branch_manager import GraphBranchManager
from nas.model.utils.model_structure import _ModelStructure

if TYPE_CHECKING:
    from nas.graph.cnn_graph import NasGraph
    from nas.graph.node.nas_graph_node import NasNode


class ModelBuilder:
    def __init__(self):
        self._model_class = Optional[Callable] = None

    def set_model_class(self, model_class) -> ModelBuilder:
        self._model_class = model_class
        return self

    def build(self, graph: NasGraph, **kwargs):
        return self._model_class(graph, kwargs)


class KerasModelMaker:
    def __init__(self, input_shape: List, graph: NasGraph, converter: Callable, num_classes: int = None):
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
    def _make_one_layer(input_layer, node: NasNode, branch_manager: GraphBranchManager, downsample: Optional[Callable]):
        layer = KerasLayers.convert_by_node_type(node=node, input_layer=input_layer, branch_manager=branch_manager)
        if node.content['params'].get('epsilon'):
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


class BaseNasTFModel(tensorflow.keras.Model):
    def __init__(self, model_struct: _ModelStructure, n_classes: int = None, input_shape=None):
        super().__init__()
        self.model_layers = None
        self.classifier = None
        self.model_structure = model_struct
        self.initialize_layers(n_classes)

    def initialize_layers(self, n_classes: int):
        output_shape = n_classes if n_classes > 2 else 1
        activation_function = 'categorical_crossentropy' if output_shape > 1 else 'binary_crossentropy'
        self.model_layers = [LayerInitializer().initialize_layer(node) for node in self.model_structure.graph]
        self.classifier = tensorflow.keras.layers.Dense(output_shape, activation=activation_function)

    def calculate_one_layer(self, inputs, layers):
        for layer in layers:
            inputs.pop(layers)

    def call(self, inputs, training=None, mask=None):
        input_layer = {self.model_structure.graph.nodes[0]: self.model_layers[0](inputs)}  # TODO
        for layers_lst in self.model_structure:
            input_layer = self.calculate_one_layer(input_layer, layers_lst)


class BaseModelInterface(ABC):
    """
    Class Interface for model handling
    """

    def __init__(self, model):
        self.model = model

    @staticmethod
    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compile_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError


class NasTFModel(BaseModelInterface):
    def __init__(self, model: tensorflow.keras.Model):
        super().__init__(model)

    @staticmethod
    def prepare_data(*args, **kwargs):
        pass

    def compile_model(self, callbacks, metrics, **additional_params):
        pass

    def fit(self, train_data: InputData, val_data: InputData, epochs, batch_size):
        train_generator = self.prepare_data(train_data)
        val_generator = self.prepare_data(val_data)
        pass

    def predict(self, *args, **kwargs):
        pass
