from typing import Tuple
from fedot.core.data.data import InputData
from nas.composer.cnn_graph import NNGraph

from fedot.core.composer.metrics import ROCAUC, Logloss, Accuracy


def calculate_validation_metric(graph: NNGraph, dataset_to_validate: InputData) -> Tuple[float, float, float]:
    # Labels and probabilities prediction for metrics calculation
    multiclass = dataset_to_validate.num_classes > 2
    predicted_labels = graph.predict(dataset_to_validate, output_mode='label', is_multiclass=multiclass)
    predicted_probabilities = graph.predict(dataset_to_validate, output_mode='default', is_multiclass=multiclass)
    # Metrics calculation
    roc_auc_score = -ROCAUC.metric(reference=dataset_to_validate, predicted=predicted_probabilities)
    log_loss_score = Logloss.metric(reference=dataset_to_validate, predicted=predicted_probabilities)
    accuracy = -Accuracy.metric(reference=dataset_to_validate, predicted=predicted_labels)
    return roc_auc_score, log_loss_score, accuracy
