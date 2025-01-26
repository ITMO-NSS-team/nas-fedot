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
