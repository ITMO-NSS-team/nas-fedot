# Contains python3.8
FROM nvcr.io/nvidia/pytorch:21.10-py3

WORKDIR /app

# Install the Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# RUN pip install opencv-python==4.8.0.74
RUN pip uninstall -y opencv-python
RUN pip install opencv-python-headless==4.5.5.64

RUN apt update
RUN apt install -y tmux

# Copy the rest of the application code into the container
COPY . /app
#CMD ["python", "cases/mnist/mnist_classification.py"]
CMD ["git", "pull", "origin", "kan"]

# Hang the container
CMD ["tail", "-f", "/dev/null"]
