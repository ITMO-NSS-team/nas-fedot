from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Callable, Optional, List, TYPE_CHECKING, Union

import tensorflow
from fedot.core.data.data import InputData
from golem.core.dag.graph_node import GraphNode

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
    def __init__(self, model_struct: _ModelStructure, n_classes: int = None):
        super().__init__()
        self.model_layers = None
        self.classifier = None
        self.model_structure = model_struct
        self.initialize_layers(n_classes)

        self._inputs_dict = dict()

    def initialize_layers(self, n_classes: int):
        output_shape = n_classes if n_classes > 2 else 1
        activation_function = 'softmax' if output_shape > 1 else 'sigmoid'
        self.model_layers = {hash(node): LayerInitializer().initialize_layer(node) for node in
                             self.model_structure.graph.nodes}
        self.classifier = tensorflow.keras.layers.Dense(output_shape, activation=activation_function)

    def make_one_layer(self, inputs, next_node_ids: List):
        #  assemble layer
        tmp_inputs = None  # temporal var for cases where original inputs is required to be saved
        # (e.g. sudden switch from main branch to residual)
        current_node_id = self.model_structure.current_node_id
        current_node = self.model_structure.graph.nodes[current_node_id]

        # if current node was part of residual branch (e.g. if there are more than 1 layer in branch)
        # then INPUT should be returned instead of LAYER_OUTPUT,  and LAYER_OUTPUT should be saved in
        # SELF._INPUTS_DICT[next_node_ids[0]]

        if current_node_id in self._inputs_dict.keys() and len(next_node_ids) == 1:
            tmp_inputs = inputs
            inputs = self._inputs_dict[current_node_id]

        layer_func = self.model_layers[current_node_id]
        # if not self._inputs_dict.get(current_node_id) else \
        # self._inputs_dict.pop(current_node_id)

        layer_output = layer_func(inputs)

        # skip connection assemble
        if len(current_node.nodes_from) > 1:
            #  extract layer result for skip connection assemble
            skip_connection_start_layer = self._inputs_dict.pop(next_node_ids)  # if it is a list,
            # then iterate over it and extract each layer simultaneously. For cases where
            # there are more than 1 skip connection for node.

            layer_output = tensorflow.keras.layers.add([layer_output, *skip_connection_start_layer])

        # applying activation function, batch_norm, dropout pooling.

        activation = current_node.content['params'].get('activation')  # TODO
        if activation:
            layer_output = tensorflow.keras.activations.relu(layer_output)

        # update self._inputs_dict by new id if there are several children for current node.
        # next_node_ids is a list where 0 id is a main path and other ids are residual path (TODO check it)
        if len(next_node_ids) > 1 or tmp_inputs:
            self._inputs_dict[next_node_ids[-1]] = layer_output
        if tmp_inputs:
            return tmp_inputs
        return layer_output

    def make_model_forward_pass_recursive(self, data_input):
        visited_nodes = set()
        # save output of layers whom have more than 1 outputs in following format: hash(node): layer_output
        outputs_to_save = dict()

        def abs_make_layer(node: Union[NasNode, GraphNode]):

            # inputs: previous layer output (not shortcut)
            # get layer func
            layer_key = hash(node)
            node_layer = self.model_layers[layer_key]
            layer_inputs = None
            # if node is not in visited nodes, we simply calculate its output
            if node in visited_nodes:
                return outputs_to_save[layer_key]

            # store nodes in outputs_to_save if they have more than one successor or if it has more than 1 predecessor
            first_condition = len(self.model_structure.graph.node_children(node)) > 1 or len(node.nodes_from) > 1
            second_condition = layer_key not in outputs_to_save.keys()
            if first_condition and second_condition:
                outputs_to_save[layer_key] = None

            # to calculate output result we need to know layer_inputs. it could be obtained
            # by recursive calculating output of parent nodes or
            # by using inputs which are first layer inputs (original data).
            if node.parameters.get('neurons') == 128 and len(node.nodes_from) > 1:
                print()

            layer_inputs = [abs_make_layer(parent) for parent in node.nodes_from] if node.nodes_from else [data_input]

            # knowing layer inputs and layer func, calculate output of this layer
            # if node already in visited, then it has more than 1 child (it has several edges that led to itself)
            # hence its output already stored in outputs_to_save, and we could reuse its result as output.

            output = node_layer(layer_inputs[0])

            output = LayerInitializer.batch_norm(node)(output) if node.content['params'].get('epsilon') else output
            output = LayerInitializer.activation(node)(output) if node.content['params'].get('activation') else output
            output = LayerInitializer.dropout(node)(output) if node.content['params'].get('drop') else output

            if len(node.nodes_from) > 1:
                shortcut = layer_inputs[-1]
                output = tensorflow.keras.layers.Add()([output, shortcut])

            # at this step we have complete layer output which could be
            # stored to outputs dict to further skip connections assemble.
            if layer_key in outputs_to_save.keys():
                outputs_to_save[layer_key] = output

            # add node to visited
            visited_nodes.add(node)

            return output

        root_node = self.model_structure.graph.root_node

        return abs_make_layer(root_node)

    def call(self, inputs, training=None, mask=None):
        inputs = self.make_model_forward_pass_recursive(inputs)
        output = self.classifier(inputs)

        return output


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

    def compile_model(self, metrics, optimizer,
                      loss: Union[str, tensorflow.keras.losses.Loss], eagerly_flag: bool = True):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=eagerly_flag)
        return self

    def fit(self, train_data: InputData, val_data: InputData, epochs, batch_size):
        train_generator = self.prepare_data(train_data)
        val_generator = self.prepare_data(val_data)
        pass

    def predict(self, *args, **kwargs):
        pass
