import datetime
import os
from typing import NamedTuple, Type, Union, Optional

import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST

from cases.mnist.cached_datasets import EuroSAT, CachedFashionMNIST, CachedMNIST
from nas.utils.random_split_hack import random_split
from nas.utils.utils import project_root


class RunConfiguration(NamedTuple):
    dataset_cls: Union[Type[EuroSAT], Type[CachedFashionMNIST], Type[CachedMNIST]]
    linear_is_kan: bool
    conv_is_kan: bool
    repetitions_for_final_choices: int
    history_path_instead_of_evolution: Optional[Union[os.PathLike, str]]

    @staticmethod
    def construct_dataset(dataset_cls):
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        if dataset_cls == EuroSAT:
            image_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
            ]
        else:
            image_transforms = [
                transforms.Grayscale(),
            ]

        transform = transforms.Compose(image_transforms + common_transforms)
        num_classes = 10  # Holds for both EuroSAT and MNIST variations

        def one_hot_encode(target):
            return torch.nn.functional.one_hot(torch.tensor(target), num_classes=num_classes).float()

        dataset_path = project_root() / "cases/mnist"

        eager = True
        if dataset_cls in [EuroSAT, ]:
            dataset_train, dataset_test = random_split(
                EuroSAT(root=dataset_path, transform=transform, target_transform=one_hot_encode, eager=eager,
                        cache_before_transform=True),
                [.7, .3]
            )
            assert num_classes == len(dataset_train.dataset.classes)
        # elif dataset_cls in [MNIST, FashionMNIST]:
        #     dataset_train = dataset_cls(root=dataset_path, train=True, download=True, transform=transform,
        #                                 target_transform=one_hot_encode)
        #     dataset_test = dataset_cls(root=dataset_path, train=False, download=True, transform=transform,
        #                                target_transform=one_hot_encode)
        #     assert num_classes == len(dataset_train.classes)
        elif dataset_cls in [CachedMNIST, CachedFashionMNIST]:
            dataset_train = dataset_cls(root=dataset_path, train=True, transform=transform,
                                        target_transform=one_hot_encode,
                                        eager=eager,
                                        cache_before_transform=False)
            dataset_test = dataset_cls(root=dataset_path, train=False, transform=transform,
                                       target_transform=one_hot_encode,
                                       eager=eager,
                                       cache_before_transform=False)
            assert num_classes == len(dataset_train.classes)
        else:
            raise ValueError("Unknown dataset: " + dataset_cls)

        return dataset_train, dataset_test

    def obtain_descriptive_name_suffix(self) -> str:
        if self.dataset_cls in [FashionMNIST, CachedFashionMNIST]:
            dataset_name = "fashion-mnist"
        elif self.dataset_cls in [MNIST, CachedMNIST]:
            dataset_name = "mnist"
        elif self.dataset_cls in [EuroSAT]:
            dataset_name = "eurosat"
        else:
            raise ValueError("Unknown-dataset")

        if self.linear_is_kan and self.conv_is_kan:
            arch_name = "dense-kan"
        elif self.linear_is_kan and not self.conv_is_kan:
            arch_name = "cnn-kan"
        elif not self.linear_is_kan and not self.conv_is_kan:
            arch_name = "cnn"
        else:
            ValueError("Unknown arch")

        return f"{arch_name}-{dataset_name}"


def obtain_history_dir_name(full_name: str = None, deduplicate: bool = True):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if full_name is None:
        return f'./_results/unnamed-runs/{timestamp}'
    else:
        basic_path = f'_results/{full_name}'
        if deduplicate and os.path.exists(basic_path):
            basic_path += "_" + timestamp
        return basic_path


def get_canonical_history_dir(cfg: RunConfiguration, full_name: Optional[str] = None,
                              name_prefix: Optional[str] = None, deduplicate: bool = True):
    # Choose name:
    assert full_name is None or name_prefix is None

    if full_name is None and name_prefix is not None:
        full_name = f"{name_prefix}-{cfg.obtain_descriptive_name_suffix()}"
    elif full_name is None:
        full_name = cfg.obtain_descriptive_name_suffix()  # For now, just don't create unnamed runs
    return obtain_history_dir_name(full_name, deduplicate=deduplicate)

MODEL_NAMES = ["cnn", "kan", "dense-kan", "cnn-kan"]
DATASET_NAMES = ["mnist", "fashion-mnist", "eurosat"]

def parse_run_name(run_name: str) -> (str, str, str):
    """Run name is formatted as {prefix}-{model}-{dataset}.
    Parses it into three parts or raises an Exception if the format is invalid.
    """
    # Sort the dataset and model names by length in descending order to match the longest names first
    sorted_datasets = sorted(DATASET_NAMES, key=len, reverse=True)
    sorted_models = sorted(MODEL_NAMES, key=len, reverse=True)

    # Initialize variables to store the found dataset and model
    dataset = None
    model = None
    prefix = None

    # Try to match the dataset name from the end of the run_name
    for ds_name in sorted_datasets:
        if run_name.endswith(f"-{ds_name}"):
            dataset = ds_name
            # Remove the dataset part from the run_name
            run_name_without_dataset = run_name[:-(len(ds_name) + 1)]
            break
    else:
        raise Exception(f"Invalid dataset name in run_name: {run_name}")

    # Try to match the model name from the end of the remaining run_name
    for model_name in sorted_models:
        if run_name_without_dataset.endswith(f"-{model_name}"):
            model = model_name
            # Remove the model part from the run_name
            prefix = run_name_without_dataset[:-(len(model_name) + 1)]
            break
    else:
        raise Exception(f"Invalid model name in run_name: {run_name}")

    return prefix, model, dataset


if __name__ == '__main__':
    test_run_names = [
        "experiment-cnn-mnist",
        "exp1-cnn-kan-fashion-mnist",
        "test-dense-kan-eurosat",
        "cnn-mnist",
        "test-kan-fashion-mnist",
        "invalidmodel-mnist",
        "cnn-invaliddataset",
        "experiment1-cnn-kan-fashion-mnist-extra",
        "experiment1--cnn--mnist",
        "cnn-mnist-",
    ]

    for run_name in test_run_names:
        try:
            prefix, model, dataset = parse_run_name(run_name)
            print(f"Run Name: '{run_name}' -> Prefix: '{prefix}', Model: '{model}', Dataset: '{dataset}'")
        except Exception as e:
            print(f"Run Name: '{run_name}' -> Error: {e}")
