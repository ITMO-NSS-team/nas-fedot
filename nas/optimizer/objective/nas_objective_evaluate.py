import os
import pathlib
import sys
from typing import TypeVar, Any

import numpy as np
import torch.nn
# import tensorflow
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
from torch.optim import AdamW
from torch.utils.data import DataLoader

from nas.composer.requirements import NNComposerRequirements
from nas.graph.BaseGraph import NasGraph
from nas.model.model_interface import BaseModelInterface
# from nas.operations.evaluation.callbacks.bad_performance_callback import PerformanceCheckingCallback

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


class NasObjectiveEvaluate(ObjectiveEvaluate):
    def __init__(self,
                 objective: Objective,
                 data_producer: DataSource,
                 model_interface: BaseModelInterface,
                 requirements: NNComposerRequirements,
                 dataset_builder,
                 pipeline_cache: Any = None,
                 preprocessing_cache: Any = None,
                 eval_n_jobs: int = 1,
                 optimization_verbose=None, **objective_kwargs):
        # Add cache
        super().__init__(objective, eval_n_jobs, **objective_kwargs)
        self._dataset_builder = dataset_builder
        self._data_producer = data_producer
        self._requirements = requirements
        self._pipeline_cache = pipeline_cache
        self._preprocessing_cache = preprocessing_cache
        self._optimization_verbose = optimization_verbose
        self.model_interface = model_interface
        self._log = default_log(self)

    def one_fold_train(self, graph: NasGraph, data: InputData, **kwargs):
        if not self._optimization_verbose == 'silent':
            fold_id = kwargs.pop('fold_id')
            self._log.message(f'\nTrain fold number: {fold_id}')

        shuffle = True if data.task != Task(TaskTypesEnum.ts_forecasting) else False
        train_data, val_data = train_test_data_setup(data, shuffle_flag=shuffle, stratify=data.target)
        n_classes = train_data.num_classes
        train_dataset = self._dataset_builder.build(train_data)
        val_dataset = self._dataset_builder.build(val_data)
        trainer = self.model_interface.build(graph=graph, input_shape=len(train_dataset[0][0]), output_shape=n_classes)
        batch_size = self._requirements.model_requirements.batch_size
        train_dataset = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = DataLoader(val_dataset, batch_size=batch_size)

        # setting up callbacks, loss and optimizer to trainer
        trainer.set_callbacks(kwargs.get('callbacks'))
        optimizer_cls = AdamW
        trainer.set_fit_params(optimizer_cls, loss_func=torch.nn.CrossEntropyLoss())
        epochs = self._requirements.opt_epochs
        trainer.fit_model(train_data=train_dataset, val_data=val_dataset, epochs=epochs)
        return trainer

    def objective_on_fold(self, graph: NasGraph, reference_data: InputData) -> Fitness:
        return self._objective(graph, reference_data=reference_data)

    def nn_objective_on_fold(self, trainer, reference_data: InputData):
        # prepare data
        test_data = DataLoader(self._dataset_builder.build(reference_data))
        predictions = trainer.predict(test_data)
        print()


    def evaluate(self, graph: NasGraph) -> Fitness:
        if not self._optimization_verbose == 'silent':
            self._log.info('Fit for graph has started.')
        graph_id = graph.root_node.descriptive_id

        graph.model_interface = self.model_interface
        folds_metrics = []

        for fold_id, (train_data, test_data) in enumerate(self._data_producer()):
            fitted_model = self.one_fold_train(graph, train_data, log=self._log, fold_id=fold_id + 1)
            evaluated_fitness = self.nn_objective_on_fold(trainer=fitted_model, reference_data=test_data)
            # try:
            #     self.one_fold_train(graph, train_data, log=self._log, fold_id=fold_id + 1)
            # except Exception as ex:
            #     exc_type, exc_obj, exc_tb = sys.exc_info()
            #     file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            #     self._log.warning(f'\nContinuing after graph fit error {ex}\n '
            #                       f'In {file_name}\n line {exc_tb.tb_lineno}\n.')
            # else:
            #     evaluated_fitness = self.objective_on_fold(graph, reference_data=test_data)
            #     if evaluated_fitness.valid:
            #         folds_metrics.append(evaluated_fitness.values)
            #         if not self._optimization_verbose == 'silent':
            #             self._log.message(f'\nFor fold {fold_id + 1} fitness {evaluated_fitness}.')
            #     else:
            #         self._log.warning(f'\nContinuing after objective evaluation error for graph: {graph_id}')
            #
            #     if folds_metrics:
            #         folds_metrics = tuple(np.mean(folds_metrics, axis=0))
            #         self._log.message(f'\nEvaluated metrics: {folds_metrics}')
            #     else:
            #         folds_metrics = None
            # finally:
            #     graph.unfit()
        return to_fitness(folds_metrics, self._objective.is_multi_objective)

    def calculate_objective_with_cache(self, graph, train_data, fold_id=None, n_jobs=-1):
        graph.try_load_from_cache()
        # Load layer weights if ``cache`` is provided and if there are already
        # fitted layers in individual history
        graph.fit()

        if self._pipeline_cache is not None:
            self._pipeline_cache.save()
