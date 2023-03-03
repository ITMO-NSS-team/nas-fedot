from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Callable, Optional, List, TYPE_CHECKING, Type

import tensorflow

from nas.model.tensorflow.future.tf_layer_initializer import LayerInitializer
from nas.model.tensorflow.tf_layers import KerasLayers
from nas.model.utils.branch_manager import GraphBranchManager
from nas.model.utils.model_structure import ModelStructure

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
    def __init__(self, input_shape: List, graph: NasGraph, converter: Type[ModelStructure], num_classes: int = None):
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


class BaseNasModel(ABC):
    def __init__(self, graph, model_structure_builder, *args, **kwargs):
        self.model = None

    @property
    def model_builder_class(self):
        return self._model_builder_class

    def set_model_builder(self, model_builder_class):
        self._model_builder_class = model_builder_class

    @abstractmethod
    def compile_model(self, graph: NasGraph, model_structure_builder: Type[ModelStructure]):
        raise NotImplementedError

    @abstractmethod
    def fit(self, train_data, val_data, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data):
        raise NotImplementedError


class BaseModelInterface(ABC):
    def __init__(self, graph: NasGraph, model_struct_builder: Type[ModelStructure]):
        self.model: BaseNasModel  # TF or Torch model obj

    @abstractmethod
    def fit(self, train_data, val_data, **additional_params):
        # train self.model with prerequirements
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data, **additional_params):
        raise NotImplementedError




class NasModelInterface(BaseModelInterface):
    def __init__(self, graph, model_structure_builder):
        super().__init__()


class TensorflowModel(tensorflow.keras.Model):
    def __init__(self, graph: NasGraph, structure_builder: Type[ModelStructure], input_shape, n_classes: Optional[int] = None):
        super().__init__()
        self._model_layers = None
        self.output_layer = None
        self._layers_hierarchy = structure_builder(graph)
        self.initialize_layers(input_shape, n_classes)

    def initialize_layers(self, input_shape, output_shape):
        output_shape = 1 if output_shape <= 2 else output_shape
        activation_func = 'sigmoid' if output_shape == 2 else 'softmax'
        layer_initializer = LayerInitializer()

        self._model_layers = [layer_initializer.initialize_layer(node) for node in self._layers_hierarchy.graph.nodes]
        self.output_layer = tensorflow.keras.layers.Dense(self._output_shape, activation=activation_func)

    def call(self, inputs, training=None, mask=None):
        return


class TensorflowModelInterface(BaseNasModel):
    def __init__(self, graph: NasGraph, model_structure_builder: Type[ModelStructure]):
        super().__init__()
        self.model = TensorflowModel(graph, model_structure_builder)

    def fit(self, train_data, val_data, dataset_builder):
        train_dataset = dataset_builder.build(train_data)
        val_dataset = dataset_builder.build(val_data)
        self.model.compile()
        self.model.train(train_dataset, val_dataset)
        return self

    def predict(self, data):
        return
