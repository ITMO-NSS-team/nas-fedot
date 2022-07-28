import pathlib
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CustomDataAugmentator(ImageDataGenerator):
    def __init__(self, transformations, **kwargs):
        self.preprocessing_function = transformations

    def _apply_transformation(self):
        pass
