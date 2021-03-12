from fedot.core.composer.chain import Chain
from fedot.core.models.data import InputData
from nas.nas_node import NNNode
from nas.keras_eval import create_nn_model, keras_model_fit, keras_model_predict


class NASChain(Chain):
    def __init__(self, nodes=None, cnn_nodes=None, fitted_model=None):
        super().__init__(nodes)
        self.cnn_nodes = cnn_nodes if not cnn_nodes is None else []
        self.model = fitted_model

    def __eq__(self, other) -> bool:
        return self is other

    def add_cnn_node(self, new_node: NNNode):
        """
        Append new node to chain list

        """
        self.cnn_nodes.append(new_node)

    def update_cnn_node(self, old_node: NNNode, new_node: NNNode):
        index = self.cnn_nodes.index(old_node)
        self.cnn_nodes[index] = new_node

    def replace_cnn_nodes(self, new_nodes):
        self.cnn_nodes = new_nodes

    def fit(self, input_data: InputData, verbose=False, input_shape: tuple = None,
            min_filters: int = None, max_filters: int = None, classes: int = 3, batch_size=24, epochs=15):
        if not self.model:
            self.model = create_nn_model(self, input_shape, classes)
        train_predicted = keras_model_fit(self.model, input_data, verbose=True, batch_size=batch_size, epochs=epochs)
        return train_predicted

    def predict(self, input_data: InputData):
        evaluation_result = keras_model_predict(self.model, input_data)
        return evaluation_result
