from typing import Tuple

from fedot.core.composer.metrics import ROCAUC, Logloss, Accuracy
from fedot.core.data.data import InputData, OutputData

from nas.data import KerasDataset
from nas.data.dataset.builder import BaseNasDatasetBuilder
from nas.data.setup_data import setup_data


# Hotfix
def get_predictions(graph, data: InputData, data_transformer: BaseNasDatasetBuilder) -> Tuple[OutputData, OutputData]:
    multiclass = data.num_classes > 2
    data_generator_to_predict = data_transformer.build(data, batch_size=1, mode='test')
    predicted_labels = graph.predict(data_generator_to_predict, output_mode='labels', is_multiclass=multiclass)
    predicted_probabilities = graph.predict(data_generator_to_predict, output_mode='default', is_multiclass=multiclass)
    return predicted_labels, predicted_probabilities


def calculate_validation_metric(data,
                                predicted_probabilities, predicted_labels) -> Tuple[float, float, float]:
    # Metrics calculation
    roc_auc_score = -ROCAUC.metric(reference=data, predicted=predicted_probabilities)
    log_loss_score = Logloss.metric(reference=data, predicted=predicted_probabilities)
    accuracy = -Accuracy.metric(reference=data, predicted=predicted_labels)
    return roc_auc_score, log_loss_score, accuracy
