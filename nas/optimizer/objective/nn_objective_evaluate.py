import os
import sys
import pathlib
from typing import Optional, TypeVar, Any

import keras.backend
import tensorflow as tf
import numpy as np

from fedot.core.dag.graph import Graph
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import default_log
from fedot.core.optimisers.fitness import Fitness
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import ObjectiveEvaluate, DataSource
from fedot.core.optimisers.objective.objective import to_fitness
from fedot.core.repository.tasks import TaskTypesEnum, Task

from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.data.data_generator import Preprocessor, DataGenerator
from nas.data.setup_data import setup_data
from nas.graph.cnn.cnn_graph import NNGraph
from nas.model.utils import converter
from nas.model.nn.tf_model import ModelMaker

G = TypeVar('G', Graph, OptGraph)


def _exceptions_save(graph: NNGraph, error_msg: Exception):
    data_folder = pathlib.Path('../debug_data')
    data_folder.mkdir(parents=True, exist_ok=True)
    folder = len(list(data_folder.iterdir()))
    save_path = data_folder / str(folder)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'log.txt', 'w') as file:
        file.write(str(error_msg))
    graph.save(save_path)


class NNObjectiveEvaluate(ObjectiveEvaluate[G]):
    def __init__(self, objective, data_producer: DataSource, preprocessor: Preprocessor,
                 requirements: NNComposerRequirements, pipeline_cache: Any = None,
                 preprocessing_cache: Any = None, eval_n_jobs: int = 1, **objective_kwargs):
        # Add cache
        super().__init__(objective, eval_n_jobs, **objective_kwargs)
        self._data_producer = data_producer
        self._requirements = requirements
        self._pipeline_cache = pipeline_cache
        self._preprocessing_cache = preprocessing_cache
        self._preprocessor = preprocessor
        self._log = default_log(self)

    def one_fold_train(self, graph: NNGraph, data: InputData, fold_id: Optional[int] = None, **kwargs) -> None:
        graph.release_memory(**kwargs)

        shuffle = True if data.task != Task(TaskTypesEnum.ts_forecasting) else False
        data_to_train, data_to_validate = train_test_data_setup(data, shuffle_flag=shuffle, stratify=data.target)
        train_generator = setup_data(data_to_train, self._requirements.nn_requirements.batch_size, self._preprocessor,
                                     'train', DataGenerator, shuffle)
        validation_generator = setup_data(data_to_validate, self._requirements.nn_requirements.batch_size,
                                          self._preprocessor, 'train', DataGenerator, shuffle)

        graph.fit(train_generator, validation_generator, self._requirements, data.num_classes,
                  shuffle=shuffle, **kwargs)

    def calculate_objective(self, graph: NNGraph, reference_data: InputData, fold_id: Optional[int] = None) -> Fitness:
        test_generator = setup_data(reference_data, 1, self._preprocessor, 'test', DataGenerator, False)
        return self._objective(graph, reference_data=test_generator)

    def evaluate(self, graph: NNGraph) -> Fitness:
        graph.log = self._log
        graph_id = graph.root_node.descriptive_id

        self._log.info('Fit for graph has started.')

        folds_metrics = []
        for fold_id, (train_data, test_data) in enumerate(self._data_producer()):
            # self.one_fold_train(graph, train_data, fold_id, log=self._log)
            try:
                self.one_fold_train(graph, train_data, fold_id, log=self._log)
            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                self._log.warning(f'Continuing after graph fit error {ex}\n '
                                  f'in {file_name}\n line {exc_tb.tb_lineno}\n')
                graph.model = None
                keras.backend.clear_session()
                # graph.release_memory()
                # graph.unfit()
                _exceptions_save(graph, ex)
                continue

            evaluated_fitness = self.calculate_objective(graph, reference_data=test_data)
            if evaluated_fitness.valid:
                folds_metrics.append(evaluated_fitness.values)
            else:
                self._log.warning(f'Continuing after objective evaluation error for graph: {graph_id}')
                # graph.release_memory()
                graph.model = None
                keras.backend.clear_session()
                continue

        if folds_metrics:
            folds_metrics = tuple(np.mean(folds_metrics, axis=0))
            self._log.message(f'Evaluated metrics: {folds_metrics}')
            graph._weights = None
        else:
            folds_metrics = None

        graph.model = None
        keras.backend.clear_session()
        return to_fitness(folds_metrics, self._objective.is_multi_objective)

    def calculate_objective_with_cache(self, graph, train_data, fold_id=None, n_jobs=-1):
        graph.try_load_from_cache()  # Load layer weights if ``cache`` is provided and if there are already
        # fitted layers in individual history
        graph.fit()

        if self._pipeline_cache is not None:
            self._pipeline_cache.save()
