#!/bin/bash

source ~/nas-fedot-kan/venv/bin/activate

# Add /app and /app/cases/mnist to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:~/nas-fedot-kan/:~/nas-fedot-kan/cases/mnist"

# Run the Python script
python ~/nas-fedot-kan/cases/mnist/mnist_classification.py


