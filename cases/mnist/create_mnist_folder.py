import os
import pathlib

from fedot.core.repository.tasks import Task, TaskTypesEnum
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage

from nas.data.nas_data import InputDataNN
from nas.utils.utils import project_root


def save_mnist_to_folder(mnist_data, root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    to_pil = ToPILImage()
    for idx, (image, label) in enumerate(mnist_data):
        label_dir = os.path.join(root_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        image_path = os.path.join(label_dir, f"{idx}.png")
        pil_image = to_pil(image)
        pil_image.save(image_path)


# Root directory to save images (same for train and test parts, because splitting is done in fedot-nas)
mnist_root_dir = project_root() / "cases/mnist/mnist_dataset"

# Load Pytorch's MNIST
train_data = MNIST(root='./mnist_dataset', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='./mnist_dataset', train=False, download=True, transform=ToTensor())

# Save train and test data
# save_mnist_to_folder(train_data, os.path.join(mnist_root_dir, 'train'))
# save_mnist_to_folder(test_data, os.path.join(mnist_root_dir, 'test'))

# …to the same folder
save_mnist_to_folder(train_data, mnist_root_dir)
save_mnist_to_folder(test_data, mnist_root_dir)

data_path = pathlib.Path(mnist_root_dir)
task = Task(task_type=TaskTypesEnum.classification)

input_data = InputDataNN.data_from_folder(data_path, task)

print(input_data)