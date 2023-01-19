import tensorflow as tf
from fedot.core.data.data import InputData

from nas.data.data_generator import ImageLoader


def setup_data(input_data: InputData, batch_size, data_preprocessor, mode,
               data_generator, shuffle) -> tf.keras.utils.Sequence:
    """This function converts FEDOT's InputData to generator format"""
    dataset_loader = ImageLoader(input_data)
    if mode == 'train':
        shuffle = shuffle
    else:
        shuffle = False
    return data_generator(dataset_loader, data_preprocessor, batch_size, shuffle)
