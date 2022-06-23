from typing import Tuple, List, Union

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.composer.metrics import ROCAUC, Logloss, Accuracy
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

from nas.composer.cnn.cnn_graph import CNNGraph


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


# def plot_confusion_matrix(data: InputData, predicted_labels):
#
#     confusion_matrix = tf.math.confusion_matrix(labels=data.target, predictions=predicted_labels.predict,
#                                                 num_classes=data.num_classes).numpy()
#     _plot_matrix(confusion_matrix, classes=np.unique(data.target))
#
#
# def _plot_matrix(cm, classes, title='Confusion matrix'):
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  ha='center',
#                  color='white' if cm[i, j] > thresh else 'black')
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
