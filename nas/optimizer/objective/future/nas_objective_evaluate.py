import os
import sys

import numpy as np
import torch
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.objective import DataSource
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.log import default_log
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.objective import ObjectiveEvaluate, Objective
from golem.core.optimisers.objective.objective import to_fitness
from torch.utils.data import DataLoader

from nas.composer.requirements import NNComposerRequirements
from nas.data.dataset.builder import BaseNNDatasetBuilder
from nas.graph.BaseGraph import NasGraph
from nas.model.constructor import ModelConstructor
from nas.model.model_interface import NeuralSearchModel


# OPT_TYPES = Union[Type[torch.optim.Optimizer, torch.optim.AdamW]]

# TODO rewrite dock strings; pass flag to Dataset builder to disable data transformations.
class NASObjectiveEvaluate(ObjectiveEvaluate):
    """
    This class defines how Objective will be evaluated for neural network like graph structure.
    """

    def __init__(self,
                 objective: Objective,
                 # optimizer,
                 # loss_func,
                 data_producer: DataSource,
                 model_trainer_builder: ModelConstructor,
                 requirements: NNComposerRequirements,
                 nn_dataset_builder: BaseNNDatasetBuilder,
                 verbose_level=None,
                 eval_n_jobs: int = 1,
                 # callbacks: Optional[Sequence] = None,
                 **objective_kwargs):
        super().__init__(objective=objective, eval_n_jobs=eval_n_jobs, **objective_kwargs)
        self.verbose_level = verbose_level
        self._data_producer = data_producer
        self._dataset_builder = nn_dataset_builder
        self._model_trainer_builder = model_trainer_builder
        self._requirements = requirements
        # self._callbacks = callbacks
        # self._optimizer = optimizer
        # self._loss_func = loss_func
        self._log = default_log(self)

    def evaluate(self, graph: NasGraph) -> Fitness:
        # create pbar only if there is evaluation on folds
        # pbar = tqdm(self._data_producer()) if self._requirements.cv_folds > 1 else self._data_producer()
        fold_metrics = []
        for fold_id, (train_data, test_data) in enumerate(self._data_producer()):
            fitted_model = self._graph_fit(graph, train_data, log=self._log, fold_id=fold_id + 1)
            fold_fitness = self._evaluate_fitted_model(fitted_model, test_data, graph, log=self._log,
                                                       fold_id=fold_id + 1)
            if fold_fitness.valid:
                fold_metrics.append(fold_fitness.values)
            else:
                self._log.warning(f'\nContinuing after objective evaluation error.')

            if fold_metrics:
                fold_metrics = tuple(np.mean(fold_metrics, axis=0))
                self._log.message(f'\nEvaluated metrics: {fold_metrics}')
            del fitted_model
            # else:
            #     fold_metrics = None
            # try:
            #     fitted_model = self._graph_fit(graph, train_data, log=self._log, fold_id=fold_id + 1)
            # except Exception as ex:
            #     exc_type, exc_obj, exc_tb = sys.exc_info()
            #     file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            #     self._log.warning(f'\nContinuing after graph fit error {ex}\n '
            #                       f'In {file_name}\n line {exc_tb.tb_lineno}\n.')
            # else:
            #     fold_fitness = self._evaluate_fitted_model(fitted_model, test_data, graph, log=self._log,
            #                                                fold_id=fold_id + 1)
            #     if fold_fitness.valid:
            #         fold_metrics.append(fold_fitness.values)
            #     else:
            #         self._log.warning(f'\nContinuing after objective evaluation error.')
            #
            #     if fold_metrics:
            #         fold_metrics = tuple(np.mean(fold_metrics, axis=0))
            #         self._log.message(f'\nEvaluated metrics: {fold_metrics}')
            #     else:
            #         fold_metrics = None
            # finally:
            #     del fitted_model
        return to_fitness(fold_metrics, self._objective.is_multi_objective)

    def _graph_fit(self, graph: NasGraph, train_data: InputData, log, fold_id) -> NeuralSearchModel:
        """
        This method compiles and fits a neural network based on given parameters and graph structure.

        Args:
             graph - Graph with defined search space of operations to apply during training process;
             train_data - dataset used as an entry point into the pipeline fitting procedure;

         Returns:
             Fitted model object
        """
        shuffle_flag = train_data.task != Task(TaskTypesEnum.ts_forecasting)
        classes = train_data.num_classes
        batch_size = self._requirements.model_requirements.batch_size
        opt_epochs = self._requirements.opt_epochs

        opt_data, val_data = train_test_data_setup(train_data, shuffle_flag=shuffle_flag, stratify=train_data.target)
        opt_dataset = DataLoader(self._dataset_builder.build(opt_data), batch_size=batch_size)
        val_dataset = DataLoader(self._dataset_builder.build(val_data), batch_size=batch_size)

        input_shape = self._requirements.model_requirements.input_shape[-1]
        trainer = self._model_trainer_builder.build(input_shape=input_shape, output_shape=classes, graph=graph)
        trainer.fit_model(train_data=opt_dataset, val_data=val_dataset, epochs=opt_epochs)
        return trainer

    def _evaluate_fitted_model(self, fitted_model: NeuralSearchModel, test_data: InputData, graph: NasGraph,
                               log, fold_id):
        """
        Method for graph's fitness estimation on given data. Estimates fitted model fitness.
        """
        complexity_matrics = [m(graph) for _, m in self._objective.complexity_metrics.items()]

        shuffle_flag = test_data.task != Task(TaskTypesEnum.ts_forecasting)
        test_dataset = DataLoader(self._dataset_builder.build(test_data),
                                  batch_size=self._requirements.model_requirements.batch_size)

        predictions = fitted_model.predict(test_dataset)
        loss = fitted_model.loss_function(torch.Tensor(predictions), torch.Tensor(test_data.target)).item()
        return to_fitness([*complexity_matrics, loss], self._objective.is_multi_objective)
        # for metric in self._objective.metrics:
