import datetime

import tensorflow


class PerformanceChecker(tensorflow.keras.callbacks.Callback):
    """
    Callback for model performance evaluation during optimization.
    After each epoch evaluates model performance metrics like runtime, loss and if they are not fit to desired range,
    model considered as invalid
    """
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None
        self.time_restriction = datetime.timedelta(seconds=250)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self.end_time = datetime.datetime.now()
        _loss = logs.get('val_loss')
        _acc = logs.get('val_categorical_accuracy', 0)
        time_delta = self.end_time - self.start_time
        cond_1 = time_delta >= self.time_restriction
        cond_2 = _loss > 100
        cond_3 = _acc < 0.015
        cond_4 = time_delta > datetime.timedelta(seconds=450)
        if cond_1 and any([cond_2, cond_3, cond_4]):
            self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        self.end_time = datetime.datetime.now()
        time_delta = self.end_time - self.start_time
        if time_delta > datetime.timedelta(seconds=100):
            self.model.stop_training = True
