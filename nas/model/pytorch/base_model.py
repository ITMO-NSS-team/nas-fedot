from __future__ import annotations

from typing import Union, Optional

import numpy as np
import torch
import torch.nn
import tqdm
from golem.core.dag.graph_node import GraphNode
from torch.utils.data import Dataset

from nas.graph.BaseGraph import NasGraph
from nas.graph.node.nas_graph_node import NasNode
from nas.model.model_interface import BaseModelInterface
from nas.model.pytorch.layers.layer_initializer import TorchLayerFactory


class NASTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_layers = None
        self.output_layer = None
        self._forward_pass = None
        self._inputs_dict = None

    def init_model(self, input_shape: int, out_shape: int, graph: NasGraph, **kwargs):
        output_shape = out_shape if out_shape > 2 else 1
        model_state = {}
        for node in graph.nodes:
            if node.name == 'flatten':
                continue

            # layer = LayerInitializer(node, input_shape)

            # TODO rewrite it with custom layers
            layer_dict = TorchLayerFactory.get_layer(node)
            input_shape = self.get_input_shape(node) if self.get_input_shape(node) is not None else input_shape
            layer_dict['weighted_layer'] = layer_dict['weighted_layer'](node, input_dim=input_shape)
            if layer_dict.get('normalization') is not None:
                out_shape = node.parameters.get('out_shape', input_shape)
                layer_dict['normalization'] = layer_dict['normalization'](node, input_dim=out_shape)
            model_state[node] = layer_dict
        self.model_layers = model_state
        self.output_layer = torch.nn.Linear(input_shape, output_shape)

    @staticmethod
    def get_input_shape(node: NasNode):
        input_shape = None
        parent_node = [] if not node.nodes_from else node.nodes_from[0]
        if parent_node:
            input_shape = parent_node.parameters.get('out_shape')
        return input_shape

    def build_forward_pass(self, inputs: torch.Tensor, graph: NasGraph):
        visited_nodes = set()
        node_to_save = dict()

        def _forward_pass_one_layer_recursive(node: Union[GraphNode, NasNode]):
            layer_state_dict = self.model_layers[node] if node.name != 'flatten' else \
                {'weighted_layer': torch.nn.Flatten()}
            if node in visited_nodes:
                return node_to_save[node]
            first_save_cond = len(graph.node_children(node)) > 1 or len(node.nodes_from) > 1
            second_save_cond = node not in node_to_save.keys()
            if first_save_cond and second_save_cond:
                node_to_save[node] = None
            layer_inputs = [_forward_pass_one_layer_recursive(parent) for parent in node.nodes_from] \
                if node.nodes_from else [inputs]
            output = layer_state_dict['weighted_layer'](layer_inputs[0])
            if layer_state_dict.get('normalization') is not None:
                output = layer_state_dict['normalization'](output)
            if len(node.nodes_from) > 1:
                shortcut = layer_inputs[-1]
                output += shortcut
            if node.name not in ['pooling2d', 'dropout', 'adaptive_pool2d', 'flatten']:
                output = TorchLayerFactory.get_activation(node.parameters['activation'])()(output)
            # output = dropout(node)(output)
            if node in node_to_save.keys():
                node_to_save[node] = output

            visited_nodes.add(node)
            return output

        out = _forward_pass_one_layer_recursive(graph.root_node)
        return self.output_layer(out)


class TorchModel(BaseModelInterface):
    def __init__(self, model_class: torch.nn.Module, graph: NasGraph, input_shape: int, out_shape: int):
        super().__init__(model_class)
        self._device = None
        self._writer = None
        self._model = model_class().init_model(input_shape, out_shape, graph)

    def __call__(self, data: Optional[torch.Tensor, np.ndarray], **kwargs):
        data = torch.Tensor(data)
        self.model.eval()
        data = data.to(self._device)
        with torch.no_grad():
            return self.model(data)

    def fit(self, train_data: Dataset, batch_size: int, train_parameters, opt_epochs: int = None,
            val_data: Optional[Dataset] = None):
        epochs = opt_epochs if opt_epochs is not None else train_parameters.epochs
        callbacks = train_parameters.callbacks
        scheduler = train_parameters.scheduler
        optimizer = train_parameters.optimizer
        metrics = train_parameters.metrics
        loss = train_parameters.loss_func

        train_loop = tqdm.trange(epochs, position=0)
        for epoch in train_loop:
            train_loop.set_description(f'Epoch [{epoch + 1}/{epochs}')
            train_logs = self._one_epoch_train(train_data, loss, optimizer, scheduler, metrics)
            val_logs = {} if val_data is None else self._one_epochs_val(val_data, loss, metrics)
            train_loop.set_postfix(learning_rate=optimizer.param_groups[0]['lr'],
                                   **train_logs, **val_logs)

    def predict(self, *args, **kwargs):
        pass
