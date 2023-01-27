import io

import tensorflow as tf
from matplotlib import pyplot as plt


def plot2image(figure):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(figure)
    buffer.seek(0)

    img = tf.image.decode_png(buffer.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)

    return img
