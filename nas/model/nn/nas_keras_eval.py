import pathlib
import datetime
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from fedot.core.data.data import OutputData

from nas.utils import utils, var
from nas.data.data_generator import temporal_setup_data

utils.set_root(var.project_root)


def _keras_model_prob2labels(predictions: np.array, is_multiclass: bool = False) -> np.array:
    if is_multiclass:
        output = []
        for prediction in predictions:
            values = []
            for val in prediction:
                values.append(round(val))
            output.append(np.float64(values))
    else:
        output = np.zeros(predictions.shape)
        for i, prediction in enumerate(predictions):
            class_prediction = np.argmax(prediction)
            output[i][class_prediction] = 1
    return output


def keras_model_fit(model, train_data, val_data, verbose: bool = True,
                    batch_size: int = 16, epochs: int = 10, **kwargs):
    gen = kwargs.get('gen', datetime.date.day)
    ind = kwargs.get('ind', datetime.datetime.hour)
    graph = kwargs.get('graph', None)
    logdir = kwargs.get('results_path')
    is_multiclass = train_data.num_classes > 2

    train_generator = temporal_setup_data(train_data, batch_size, is_multiclass)
    val_generator = temporal_setup_data(val_data, batch_size, is_multiclass)

    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
                                       verbose=1, min_delta=1e-4, mode='min')
    if logdir:
        logdir = pathlib.Path(logdir) / str(gen) / str(ind)
        # custom_callback_handler = nas_callbacks.NASCallbackTF(data_producer,
        #                                                    [nas_callbacks.F1ScoreCallback, nas_callbacks.RAMProfiler,
        #                                                        nas_callbacks.GPUProfiler], log_path=logdir)
        mcp_save = ModelCheckpoint(str(logdir / 'model' / 'mdl_wts.hdf5'), save_best_only=True, monitor='val_loss',
                                   mode='min')
        # tensorboard_callback = TensorBoard(
        #     log_dir=logdir,
        #     histogram_freq=1)
        callbacks = [early_stopping, mcp_save, reduce_lr_loss]
    else:
        callbacks = [early_stopping, reduce_lr_loss]
    # if graph and logdir:
    #     graph_plotter = nas_callbacks.GraphPlotter(graph, log_path=logdir)
    #     callbacks.append(graph_plotter)

    model.fit(train_generator,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              val_generator=val_generator,
              shuffle=True,
              callbacks=callbacks)
    return keras_model_predict(model, val_data, batch_size, is_multiclass=is_multiclass)


def keras_model_predict(model, input_data, batch_size, output_mode: str = 'default',
                        is_multiclass: bool = False) -> OutputData:
    generator_to_predict = temporal_setup_data(input_data, batch_size, is_multiclass)
    evaluation_result = model.predict(generator_to_predict)
    if output_mode == 'label':
        if is_multiclass:
            evaluation_result = np.argmax(evaluation_result, axis=1)
        else:
            evaluation_result = np.where(evaluation_result > 0.5, 1, 0)
    return OutputData(idx=input_data.idx,
                      features=input_data.features,
                      predict=evaluation_result,
                      task=input_data.task, data_type=input_data.data_type)
