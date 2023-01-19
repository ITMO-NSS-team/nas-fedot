import pathlib

from fedot.core.repository.tasks import Task, TaskTypesEnum

from nas.data.load_images import NasData
from nas.data.dataset.tf_dataset import KerasDataset
from nas.data.dataset.builder import BaseNasDatasetBuilder
from nas.utils.utils import set_root, project_root

set_root(project_root())

def test_dataset_params_switch():
    data = NasData.data_from_folder(pathlib.Path(project_root(), 'example_datasets/butterfly_cls'),
                                    Task(TaskTypesEnum.classification))
    dataset = BaseNasDatasetBuilder(KerasDataset, batch_size=1)
    generator_1 = dataset.build(data, mode='train')
    generator_2 = dataset.build(data, mode='train', batch_size=2)
    generator_3 = dataset.build(data, mode='train', batch_size=3)
    assert len(generator_1[0][0]) == len(generator_1[0][1]) == 1
    assert len(generator_2[0][0]) == len(generator_2[0][1]) == 2
    assert len(generator_3[0][0]) == len(generator_3[0][1]) == 3



