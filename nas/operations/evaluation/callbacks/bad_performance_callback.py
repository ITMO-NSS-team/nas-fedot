import datetime
import tensorflow


class CustomCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.time_restriction = datetime.timedelta(seconds=400)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self.end_time = datetime.datetime.now()
        time_delta = self.end_time - self.start_time
        if time_delta >= self.time_restriction:
            self.model.stop_training = True
