import os
import sys
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "fedot"))

import datetime
from typing import Optional, Tuple
from sklearn.metrics import roc_auc_score as roc_auc, log_loss, accuracy_score
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.composer.chain import Chain
from fedot.core.models.model import *
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.models.data import InputData
from nas.composer.gp_cnn_composer import GPNNComposer, GPNNComposerRequirements
from nas.layer import LayerTypesIdsEnum
from nas.cnn_data import from_json

random.seed(2)
np.random.seed(2)


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> Tuple[float, float, float]:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    y_pred = [np.float64(predict[0]) for predict in predicted.predict]
    log_loss_value = log_loss(y_true=dataset_to_validate.target,
                              y_pred=y_pred)
    y_pred = [round(predict[0]) for predict in predicted.predict]
    accuracy_score_value = accuracy_score(y_true=dataset_to_validate.target,
                                          y_pred=y_pred)

    return roc_auc_value, log_loss_value, accuracy_score_value


def run_iceberg_classification_problem(file_path,
                                       max_lead_time: datetime.timedelta = datetime.timedelta(days=2),
                                       gp_optimiser_params: Optional[GPChainOptimiserParameters] = None, ):
    dataset_to_compose, dataset_to_validate = from_json(file_path)
    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    cnn_secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]
    conv_types = [LayerTypesIdsEnum.conv2d]
    pool_types = [LayerTypesIdsEnum.maxpool2d, LayerTypesIdsEnum.averagepool2d]
    nn_primary = [LayerTypesIdsEnum.dense]
    nn_secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]
    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.log_loss)
    # additional metrics
    # metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
    # metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.accuracy)

    composer_requirements = GPNNComposerRequirements(
        conv_types=conv_types, pool_types=pool_types, cnn_secondary=cnn_secondary,
        primary=nn_primary, secondary=nn_secondary, min_arity=2, max_arity=2,
        max_depth=3, pop_size=20, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time, image_size=[75, 75], train_epochs_num=10)

    # Create GP-based composer
    composer = GPNNComposer()

    gp_optimiser_params = gp_optimiser_params if gp_optimiser_params else None
    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function,
                                                is_visualise=True, optimiser_parameters=gp_optimiser_params)

    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True, input_shape=(75, 75, 3), epochs=25)

    print('Best model structure:')
    for node in chain_evo_composed.cnn_nodes:
        print(node)
    for node in chain_evo_composed.nodes:
        print(node)

    json_file = 'model.json'
    model_json = chain_evo_composed.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)

    ComposerVisualiser.visualise(chain_evo_composed)

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(chain_evo_composed, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    return roc_on_valid_evo_composed, chain_evo_composed


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition

    file_path = 'data/iceberg/train.json'
    full_path = file_path

    run_iceberg_classification_problem(full_path)
