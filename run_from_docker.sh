#!/bin/bash

# Add /app and /app/cases/mnist to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:/app:/app/cases/mnist"

# Run the Python script
python /app/cases/mnist/mnist_classification.py

