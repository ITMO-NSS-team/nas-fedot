import pathlib
import tensorflow as tf

from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum

from nas.utils.utils import project_root, set_root
from nas.data.load_images import NNData
from nas.graph.nn_graph.cnn.cnn_graph import NNGraph
from nas.data.data_generator import Preprocessor, DataGenerator
from nas.data.setup_data import setup_data

project_root = project_root()
set_root(project_root)


def main():
    # Task and objective function
    task = Task(TaskTypesEnum.classification)

    objective = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    objective = Objective(objective)

    # Load dataset from folders
    dataset_path = pathlib.Path('../datasets/train')
    butterfly_dataset = NNData.data_from_folder(dataset_path, task)

    # Setup Loader, Preprocessor and Generator for InputData
    transformations = [tf.convert_to_tensor]
    preprocessor = Preprocessor()
    preprocessor.set_image_size((20, 20)).set_features_transformations(transformations)

    # Setup data for fit/predict
    test_generator = setup_data(butterfly_dataset, batch_size=1, data_preprocessor=preprocessor, mode='test',
                                data_generator=DataGenerator, shuffle=False)

    # Load optimized graph and fitted model
    graph_path = pathlib.Path('../_results/debug/master/2022-08-30/graph.json')
    model_path = pathlib.Path('../_results/debug/master/2022-08-30/fitted_model.h5')

    model = tf.keras.models.load_model(model_path)
    graph = NNGraph.load(str(graph_path))
    graph.model = model

    # Objective calculation
    fitness = objective(graph, reference_data=test_generator)

    print(fitness)

    print('Done!')


if __name__ == '__main__':
    main()
