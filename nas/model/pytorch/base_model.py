from __future__ import annotations

from typing import Union, Optional

import numpy as np
import torch
import torch.nn
import tqdm
from golem.core.dag.graph_node import GraphNode
from torch.utils.data import Dataset, DataLoader

from nas.graph.BaseGraph import NasGraph
from nas.graph.node.nas_graph_node import NasNode
from nas.model.model_interface import BaseModelInterface
from nas.model.pytorch.layers.layer_initializer import TorchLayerFactory


def get_node_input_channels(node: Union[GraphNode, NasNode]):
    n = node.nodes_from[0]
    while not n.parameters.get('out_shape'):
        n = n.nodes_from[0]
    return n.parameters.get('out_shape')


class NASTorchModel(torch.nn.Module):
    """
    Implementation of Pytorch model class for graph described architectures.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model_layers = None
        self.output_layer = None

        self._graph = None

    # def get_input_shape(self, node: NasNode):
    #     input_shape = None
    #     parent_node = [] if not node.nodes_from else node.nodes_from[0]
    #     parent_layer = self.__getattr__(f'node_{parent_node.uid}_n') if hasattr(self,
    #                                                                             f'node_{parent_node.uid}_n') else self.__getattr__(
    #         f'node_{parent_node.uid}')
    #     if parent_node:
    #         input_shape = parent_node.parameters.get('out_shape')
    #     return input_shape

    def set_device(self, device):
        self.to(device)

    def get_input_shape(self, node: Union[NasNode, GraphNode]):
        w_node = node
        dim_node = None
        while w_node.name in ['flatten', 'pooling2d', 'adaptive_pool2d', 'batch_norm2d']:
            if w_node.name in ['pooling2d', 'adaptive_pool2d', 'batch_norm2d']:
                dim_node = w_node
            w_node = w_node.nodes_from[0]
        name_to_search = f'node_{w_node.uid}'
        layer_name = self.__getattr__(f'{name_to_search}_n') if hasattr(self, f'{name_to_search}_n') \
            else self.__getattr__(f'{name_to_search}')

        if dim_node:
            pool_layer = self.__getattr__(f'node_{dim_node.uid}_n') if hasattr(self, f'node_{dim_node.uid}_n') \
                else self.__getattr__(f'node_{dim_node.uid}')
            return np.dot(*pool_layer.output_size) * layer_name.out_channels
        else:
            return np.dot(*layer_name.kernel_size) * layer_name.out_channels

    def init_model(self, in_shape: int, out_shape: int, graph: NasGraph, **kwargs):
        self._graph = graph
        visited_nodes = set()
        out_shape = out_shape if out_shape > 2 else 1

        def _init_layer(node: Union[GraphNode, NasNode]):
            if node.nodes_from:
                for n in node.nodes_from:
                    _init_layer(n)
            if node not in visited_nodes:
                layer_func = TorchLayerFactory.get_layer(node)
                input_shape = in_shape if not node.nodes_from else get_node_input_channels(node)
                layer = layer_func['weighted_layer'](node, input_dim=input_shape)
                self.__setattr__(f'node_{node.uid}', layer)
                if layer_func.get('normalization'):
                    output_shape = layer.out_channels
                    self.__setattr__(f'node_{node.uid}_n', layer_func['normalization'](node, input_dim=output_shape))
                visited_nodes.add(node)

        _init_layer(graph.root_node)
        linear_input = self.get_input_shape(graph.root_node)
        # node = graph.root_node if not graph.root_node.name == 'flatten' else graph.root_node.nodes_from[0]
        # name_to_search = f'node_{node.uid}'
        # layer_name = self.__getattr__(f'{name_to_search}_n') if hasattr(self, f'{name_to_search}_n') \
        #     else self.__getattr__(f'{name_to_search}')

        self.output_layer = torch.nn.Linear(linear_input, out_shape)

    # def init_model(self, input_shape: int, out_shape: int, graph: NasGraph, **kwargs):
    #     output_shape = out_shape if out_shape > 2 else 1
    #     for node in graph.nodes:
    #         if node.name == 'flatten':
    #             continue
    #         layer_dict = TorchLayerFactory.get_layer(node)
    #         input_shape = self.get_input_shape(node) if self.get_input_shape(node) is not None else input_shape
    #         layer = layer_dict['weighted_layer'](node, input_dim=input_shape)
    #         self.__setattr__(f'node_{node.uid}', layer)
    #         if layer_dict.get('normalization') is not None:
    #             out_shape = layer.out_channel  # node.parameters.get('out_shape', input_shape)
    #             self.__setattr__(f'node_{node.uid}_n', layer_dict['normalization'](node, input_dim=out_shape))
    #     self.output_layer = torch.nn.Linear(input_shape, output_shape)
    #     self._graph = graph

    def forward(self, inputs: torch.Tensor):
        visited_nodes = set()
        node_to_save = dict()

        def _forward_pass_one_layer_recursive(node: Union[GraphNode, NasNode]):
            layer_name = f'node_{node.uid}'
            layer_state_dict = self.__getattr__(layer_name) if node.name != 'flatten' else \
                torch.nn.Flatten()
            if node in visited_nodes:
                return node_to_save[node]
            first_save_cond = len(self._graph.node_children(node)) > 1 or len(node.nodes_from) > 1
            second_save_cond = node not in node_to_save.keys()
            if first_save_cond and second_save_cond:
                node_to_save[node] = None
            layer_inputs = [_forward_pass_one_layer_recursive(parent) for parent in node.nodes_from] \
                if node.nodes_from else [inputs]
            output = layer_state_dict(layer_inputs[0])
            if hasattr(self, f'{layer_name}_n'):
                output = self.__getattr__(f'{layer_name}_n')(output)
            if len(node.nodes_from) > 1:
                shortcut = layer_inputs[-1]
                output += shortcut
            if node.name not in ['pooling2d', 'dropout', 'adaptive_pool2d', 'flatten']:
                output = TorchLayerFactory.get_activation(node.parameters['activation'])()(output)
            if node in node_to_save.keys():
                node_to_save[node] = output
            visited_nodes.add(node)
            return output

        out = _forward_pass_one_layer_recursive(self._graph.root_node)
        return self.output_layer(out)

    def _one_epoch_train(self, train_data: DataLoader, optimizer, loss_fn, device):
        running_loss = 0
        for batch_id, (features_batch, targets_batch) in enumerate(train_data):
            features_batch, targets_batch = features_batch.to(device), targets_batch.to(device)
            optimizer.zero_grad()
            outputs = self.__call__(features_batch)
            loss = loss_fn(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss = running_loss / len(train_data)
        # TODO add tb writer
        return running_loss

    def _one_epoch_val(self, val_data: DataLoader, loss_fn, device):
        running_loss = 0
        for batch_id, (features_batch, targets_batch) in enumerate(val_data):
            features_batch, targets_batch = features_batch.to(device), targets_batch.to(device)
            outputs = self.__call__(features_batch)
            loss = loss_fn(outputs, targets_batch)
            running_loss += loss.item()
        return running_loss / len(val_data)

    def fit(self, train_data: DataLoader,
            loss,
            val_data: Optional[DataLoader] = None,
            optimizer=torch.optim.AdamW,
            epochs: int = 1,
            device: str = 'cpu',
            **kwargs):
        self.set_device(device)
        metrics = dict()
        optim = optimizer(self.parameters(), lr=kwargs.get('lr', 1e-3))
        pbar = tqdm.trange(epochs, desc='Fitting graph', leave=False)
        for epoch in pbar:
            self.train(mode=True)
            prefix = f'Epoch {epoch} / {epochs}:'
            pbar.set_description(f'{prefix} Train')
            train_loss = self._one_epoch_train(train_data, optim, loss, device)
            metrics['train_loss'] = train_loss
            if val_data:
                pbar.set_description(f'{prefix} Validation')
                with torch.no_grad():
                    self.eval()
                    val_loss = self._one_epoch_val(val_data, loss, device)
                metrics['val_loss'] = val_loss
            pbar.set_postfix(**metrics)

    def predict(self, test_data: DataLoader, device: str = 'cpu'):
        self.set_device(device)
        self.eval()
        results = []
        with torch.no_grad():
            for features, targets in test_data:
                features, targets = features.to(device), targets.to(device)
                predictions = self.__call__(features)
                results.extend(torch.argmax(predictions, dim=-1).cpu().detach().tolist())
                # results.append(torch.argmax(predictions, dim=-1).item())
        return np.array(results)


class TorchModel(BaseModelInterface):
    """
    Base class that implements all required logic to work with NasGraphs as it was torch models.
    """

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
