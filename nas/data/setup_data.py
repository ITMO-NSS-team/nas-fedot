import tensorflow as tf
from fedot.core.data.data import InputData

from nas.data.data_generator import Loader, DataGenerator, Preprocessor


def setup_data(input_data: InputData, batch_size, data_preprocessor, mode,
               data_generator, shuffle) -> tf.keras.utils.Sequence:
    dataset_loader = Loader(input_data)
    if mode == 'train':
        shuffle = shuffle
    else:
        shuffle = False
    return data_generator(dataset_loader, data_preprocessor, batch_size, shuffle)
