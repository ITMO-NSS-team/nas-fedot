from typing import Tuple
from fedot.core.data.data import InputData, OutputData
from nas.composer.graph_gp_cnn_composer import CustomGraphModel

from sklearn.metrics import roc_auc_score as roc_auc
from fedot.core.composer.metrics import ROCAUC, Logloss, Accuracy

import statistics


def _compute_roc_auc(dataset_to_validate: InputData, predictions: OutputData, is_multiclass: bool = False):
    if not is_multiclass:
        roc_auc_value = -ROCAUC.metric(reference=dataset_to_validate, predicted=predictions)
    else:
        roc_auc_value = []
        for prediction, true in zip(predictions.predict, dataset_to_validate.target):
            roc_auc_score = roc_auc(y_true=true, y_score=prediction)
            roc_auc_value.append(roc_auc_score)
        roc_auc_value = statistics.mean(roc_auc_value)
    return roc_auc_value


def calculate_validation_metric(graph: CustomGraphModel, dataset_to_validate: InputData,
                                is_multiclass: bool = False) -> Tuple[float, float, float]:
    # Labels and probabilities prediction for metrics calculation
    predicted_labels = graph.predict(dataset_to_validate, output_mode='label', is_multiclass=is_multiclass)
    predicted_probabilities = graph.predict(dataset_to_validate, output_mode='default', is_multiclass=is_multiclass)
    # Metrics calculation
    try:
        roc_auc_score = _compute_roc_auc(dataset_to_validate=dataset_to_validate,
                                         predictions=predicted_probabilities, is_multiclass=is_multiclass)
    except ValueError as error:
        print(error)
        roc_auc_score = 0
    log_loss_score = Logloss.metric(reference=dataset_to_validate, predicted=predicted_probabilities)
    accuracy = -Accuracy.metric(reference=dataset_to_validate, predicted=predicted_labels)
    return roc_auc_score, log_loss_score, accuracy
