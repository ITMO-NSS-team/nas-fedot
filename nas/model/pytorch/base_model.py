from typing import Union

import torch
from golem.core.dag.graph_node import GraphNode

from nas.graph.BaseGraph import NasGraph
from nas.graph.node.nas_graph_node import NasNode
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
            layer_dict = TorchLayerFactory.get_layer(node)
            layer_dict['weighted_layer'] = layer_dict['weighted_layer'](input_shape, node)
            input_shape = layer_dict['weighted_layer'].out_features
            layer_dict['normalization'] = layer_dict['normalization'](input_shape, node)
            model_state[node] = layer_dict
        self.model_layers = model_state
        self.output_layer = torch.nn.Linear(input_shape, output_shape)

    def build_forward_pass(self, inputs: torch.Tensor, graph: NasGraph):
        visited_nodes = set()
        node_to_save = dict()

        def _forward_pass_one_layer_recursive(node: Union[GraphNode, NasNode]):
            layer_state_dict = self.model_layers[node]
            if node in visited_nodes:
                return node_to_save[node]
            first_save_cond = len(graph.node_children(node)) > 1 or len(node.nodes_from) > 1
            second_save_cond = node not in node_to_save.keys()
            if first_save_cond and second_save_cond:
                node_to_save[node] = None
            layer_inputs = [_forward_pass_one_layer_recursive(parent) for parent in node.nodes_from] \
                if node.nodes_from else [inputs]
            output = layer_state_dict['weighted_layer'](layer_inputs[0])
            output = layer_state_dict['normalization'](output)
            if len(node.nodes_from) > 1:
                shortcut = layer_inputs[-1]
                output += shortcut

            output = activation(output)
            output = dropout(output)
            return output

        _forward_pass_one_layer_recursive(graph.root_node)
        return
