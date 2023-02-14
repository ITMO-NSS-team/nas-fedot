import os
import pathlib
import sys
from typing import TypeVar, Any

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.objective import DataSource
from fedot.core.repository.tasks import TaskTypesEnum, Task
from golem.core.dag.graph import Graph
from golem.core.log import default_log
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveEvaluate
from golem.core.optimisers.objective.objective import to_fitness, Objective

from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.data.dataset.builder import ImageDatasetBuilder
from nas.graph.cnn_graph import NasGraph

G = TypeVar('G', Graph, OptGraph)


def _exceptions_save(graph: NasGraph, error_msg: Exception):
    data_folder = pathlib.Path('../debug_data')
    data_folder.mkdir(parents=True, exist_ok=True)
    folder = len(list(data_folder.iterdir()))
    save_path = data_folder / str(folder)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'log.txt', 'w') as file:
        file.write(str(error_msg))
    graph.save(save_path)


class NasObjectiveEvaluate(ObjectiveEvaluate[G]):
    def __init__(self, objective: Objective, data_producer: DataSource, data_transformer: ImageDatasetBuilder,
                 requirements: NNComposerRequirements, pipeline_cache: Any = None,
                 preprocessing_cache: Any = None, eval_n_jobs: int = 1, optimization_verbose=None, **objective_kwargs):
        # Add cache
        super().__init__(objective, eval_n_jobs, **objective_kwargs)
        self._data_producer = data_producer
        self._requirements = requirements
        self._pipeline_cache = pipeline_cache
        self._preprocessing_cache = preprocessing_cache
        self._data_transformer = data_transformer
        self._optimization_verbose = optimization_verbose
        self._log = default_log(self)

    def one_fold_train(self, graph: NasGraph, data: InputData, **kwargs):
        import tensorflow
        from nas.operations.evaluation.callbacks.bad_performance_callback import PerformanceCheckingCallback

        if not self._optimization_verbose == 'silent':
            fold_id = kwargs.pop('fold_id')
            self._log.message(f'\nTrain fold number: {fold_id}')

        shuffle = True if data.task != Task(TaskTypesEnum.ts_forecasting) else False
        data_to_train, data_to_validate = train_test_data_setup(data, shuffle_flag=shuffle, stratify=data.target)

        train_dataset = self._data_transformer.build(data_to_train, mode='train')
        validation_dataset = self._data_transformer.build(data_to_validate, mode='val')
        model_requirements = self._requirements.model_requirements

        loss_func = 'binary_crossentropy' if model_requirements.num_of_classes == 2 else 'categorical_crossentropy'
        metrics = tensorflow.keras.metrics.Accuracy()
        optimizer = tensorflow.keras.optimizers.Adam
        callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'),
                     tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=3, verbose=1,
                                                                  min_delta=1e-4, mode='min'),
                     PerformanceCheckingCallback()]

        graph.compile_model(input_shape=model_requirements.input_shape, loss_function=loss_func, metrics=[metrics],
                            optimizer=optimizer, n_classes=model_requirements.num_of_classes)
        graph.fit(train_dataset, validation_dataset, epoch_num=self._requirements.opt_epochs,
                  batch_size=model_requirements.batch_size, callbacks=callbacks, **kwargs)

    def calculate_objective(self, graph: NasGraph, reference_data: InputData) -> Fitness:
        test_dataset = self._data_transformer.build(reference_data, mode='test', batch_size=1)
        return self._objective(graph, reference_data=test_dataset)

    def evaluate(self, graph: NasGraph) -> Fitness:
        # super().evaluate(graph)
        if not self._optimization_verbose == 'silent':
            self._log.info('Fit for graph has started.')
        graph_id = graph.root_node.descriptive_id
        folds_metrics = []

        for fold_id, (train_data, test_data) in enumerate(self._data_producer()):
            try:
                self.one_fold_train(graph, train_data, log=self._log, fold_id=fold_id + 1)
            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                self._log.warning(f'\nContinuing after graph fit error {ex}\n '
                                  f'In {file_name}\n line {exc_tb.tb_lineno}\n.')
            else:
                evaluated_fitness = self.calculate_objective(graph, reference_data=test_data)
                if evaluated_fitness.valid:
                    folds_metrics.append(evaluated_fitness.values)
                    if not self._optimization_verbose == 'silent':
                        self._log.message(f'\nFor fold {fold_id + 1} fitness {evaluated_fitness}.')
                else:
                    self._log.warning(f'\nContinuing after objective evaluation error for graph: {graph_id}')

                if folds_metrics:
                    folds_metrics = tuple(np.mean(folds_metrics, axis=0))
                    self._log.message(f'\nEvaluated metrics: {folds_metrics}')
                else:
                    folds_metrics = None
            finally:
                graph.unfit()
        return to_fitness(folds_metrics, self._objective.is_multi_objective)

    def calculate_objective_with_cache(self, graph, train_data, fold_id=None, n_jobs=-1):
        graph.try_load_from_cache()
        # Load layer weights if ``cache`` is provided and if there are already
        # fitted layers in individual history
        graph.fit()

        if self._pipeline_cache is not None:
            self._pipeline_cache.save()
