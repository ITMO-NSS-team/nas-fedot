from pathlib import Path
from typing import List
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(confusion_matrix, class_names: List, normalize: bool = False, save=None, cmap=plt.cm.YlGn):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    if normalize:
        confusion_matrix = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
                                     decimals=2)

    threshold = confusion_matrix.max() / 2.

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        color = 'white' if confusion_matrix[i, j] > threshold else 'black'
        plt.text(j, i, confusion_matrix[i, j], horizontalalignment='center', color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        save_path = str(Path(save) / 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
    return figure


# class ConfusionMatrix:
#     @staticmethod
#     def get_matrix(data, labels):
#         y_true = labels
#         y_pred = data.predict
#         return confusion_matrix(y_true=y_true, y_pred=y_pred)
#
#     @staticmethod
#     def plot(cm, class_names, normalize, color_map, save_folder):
#         figure = plt.figure(figsize=(8, 8))
#         plt.imshow(cm, interpolation='nearest', cmap=color_map)
#         plt.title('Confusion matrix')
#         tick_marks = np.arange(len(class_names))
#         plt.xticks(tick_marks, class_names, rotation=45)
#         plt.yticks(tick_marks, class_names)
#
#         if normalize:
#             cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
#                            decimals=2)
#
#         threshold = cm.max() / 2.
#
#         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#             color = 'white' if confusion_matrix[i, j] > threshold else 'black'
#             plt.text(j, i, confusion_matrix[i, j], horizontalalignment='center', color=color)
#         plt.tight_layout()
#         plt.ylabel('True label')
#         plt.xlabel('Predicted label')
#         if save_folder:
#             ConfusionMatrix._save(save_folder)
#         return figure
#
#     @staticmethod
#     def _save(path):
#         full_path = str(Path(path, 'confusion_matrix.png'))
#         plt.savefig(full_path)
#         plt.close()
