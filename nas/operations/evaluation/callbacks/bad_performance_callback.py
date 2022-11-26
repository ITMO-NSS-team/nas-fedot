import datetime
import tensorflow


class CustomCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.time_restriction = datetime.timedelta(seconds=500)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self.end_time = datetime.datetime.now()
        _loss = logs.get('val_loss')
        _acc = logs.get('val_acc', 0)
        time_delta = self.end_time - self.start_time
        cond_1 = time_delta >= self.time_restriction
        cond_2 = _loss > 100
        cond_3 = _acc < 0.015
        if cond_1 and any([cond_2, cond_3]):
            self.model.stop_training = True
