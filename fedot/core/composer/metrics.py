from abc import abstractmethod

from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss, accuracy_score

from fedot.core.chain_validation import validate
import numpy as np
from fedot.core.composer.chain import Chain
from fedot.core.models.data import InputData


def from_maximised_metric(metric_func):
    def wrapper(*args, **kwargs):
        return -metric_func(*args, **kwargs)

    return wrapper


class ChainMetric:
    @staticmethod
    @abstractmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        raise NotImplementedError()


class RmseMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        results = chain.predict(reference_data)
        return mean_squared_error(y_true=reference_data.target, y_pred=results.predict)


class MaeMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        results = chain.predict(reference_data)
        return mean_squared_error(y_true=reference_data.target, y_pred=results.predict)


class RocAucMetric(ChainMetric):
    @staticmethod
    @from_maximised_metric
    def get_value(chain: Chain, reference_data: InputData) -> float:
        try:
            # validate(chain)
            results = chain.predict(reference_data)
            score = round(roc_auc_score(y_score=results.predict,
                                        y_true=reference_data.target), 3)
        except Exception as ex:
            print(ex)
            score = 0.5

        return score


class LogLossMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        try:
            # validate(chain)
            results = chain.predict(reference_data)
            y_pred = [np.float64(predict[0]) for predict in results.predict]
            score = round(log_loss(y_true=reference_data.target,
                                   y_pred=y_pred), 3)
        except Exception as ex:
            print(ex)
            score = 0.5

        return score


class LogLossMulticlassMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        try:
            # validate(chain)
            results = chain.predict(reference_data)
            y_pred = [np.float64(predict) for predict in results.predict]
            score = round(log_loss(y_true=reference_data.target,
                                   y_pred=y_pred), 3)
        except Exception as ex:
            print(ex)
            score = 0.5

        return score


class AccuracyScore(ChainMetric):
    @staticmethod
    @from_maximised_metric
    def get_value(chain: Chain, reference_data: InputData) -> float:
        try:
            # validate(chain)
            results = chain.predict(reference_data)
            y_pred = [round(predict[0]) for predict in results.predict]
            score = round(accuracy_score(y_true=reference_data.target,
                                         y_pred=y_pred), 3)
        except Exception as ex:
            print(ex)
            score = 0.5

        return score

class AccuracyMulticlassScore(ChainMetric):
    @staticmethod
    @from_maximised_metric
    def get_value(chain: Chain, reference_data: InputData) -> float:
        try:
            # validate(chain)
            results = chain.predict(reference_data)
            y_pred = [round(predict) for predict in results.predict]
            score = round(accuracy_score(y_true=reference_data.target,
                                         y_pred=y_pred), 3)
        except Exception as ex:
            print(ex)
            score = 0.5

        return score

# TODO: reference_data = None ?
class StructuralComplexityMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        return chain.depth ** 2 + chain.length


class NodeNum(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        return chain.length
