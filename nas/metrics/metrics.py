from typing import Tuple

from fedot.core.data.data import InputData
from fedot.core.composer.metrics import ROCAUC, Logloss, Accuracy


def get_predictions(graph, data: InputData):
    multiclass = data.num_classes > 2
    predicted_labels = graph.predict(data, output_mode='label', is_multiclass=multiclass)
    predicted_probabilities = graph.predict(data, output_mode='default', is_multiclass=multiclass)
    return predicted_labels, predicted_probabilities


def calculate_validation_metric(data,
                                predicted_probabilities, predicted_labels) -> Tuple[float, float, float]:
    # Metrics calculation
    roc_auc_score = -ROCAUC.metric(reference=data, predicted=predicted_probabilities)
    log_loss_score = Logloss.metric(reference=data, predicted=predicted_probabilities)
    accuracy = -Accuracy.metric(reference=data, predicted=predicted_labels)
    return roc_auc_score, log_loss_score, accuracy
