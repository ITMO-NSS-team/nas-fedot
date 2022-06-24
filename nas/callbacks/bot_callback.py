import tensorflow as tf
import requests
import matplotlib.pyplot as plt


class BotCallback(tf.keras.callbacks.Callback):
    def __init__(self, access_token):
        self.access_token = access_token
        self.ping_url = 'https://api.telegram.org/bot' + str(self.access_token) + '/getUpdates'
        self.response = requests.get(self.ping_url).json()
        self.chat_id = self.response['result'][0]['message']['chat']['id']

    def send_message(self, message):
        self.ping_url = 'https://api.telegram.org/bot' + str(self.access_token) + '/sendMessage?' + \
                        'chat_id=' + str(self.chat_id) + \
                        '&parse_mode=Markdown' + \
                        '&text=' + message
        self.response = requests.get(self.ping_url)

    def send_photo(self, filepath):
        file_ = open(filepath, 'rb')
        file_dict = {'photo': file_}
        self.ping_url = 'https://api.telegram.org/bot' + str(self.access_token) + '/sendPhoto?' + \
                        'chat_id=' + str(self.chat_id)
        self.response = requests.post(self.ping_url, files=file_dict)
        file_.close()

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        message = ' Epoch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(epoch, logs['acc'],
                                                                                               logs['loss'])
        message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_acc'],
                                                                                         logs['val_loss'])
        self.send_message(message)


class Plotter(BotCallback):
    def __init__(self, access_token):
        super().__init__(access_token)
        self.batch = None
        self.epoch = None
        self.train_loss = None
        self.val_loss = None
        self.train_acc = None
        self.val_acc = None
        self.fig = None
        self.logs = None

    def on_train_begin(self, logs=None):
        self.batch = 0
        self.epoch = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.fig = plt.figure(figsize=(200, 100))
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        self.epoch.append(epoch)
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        self.train_acc.append(logs['acc'])
        self.val_acc.append(logs['val_acc'])

    def on_train_end(self, logs=None):
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        ax1.plot(self.epoch, self.train_loss, label='Training Loss')
        ax1.plot(self.epoch, self.val_loss, label='Validation Loss')
        ax1.legend()

        ax2.plot(self.epoch, self.train_acc, label='Training Accuracy')
        ax2.plot(self.epoch, self.val_acc, label='Validation Accuracy')
        ax2.legend()

        plt.savefig('Accuracy and Loss plot.jpg')
        plt.close()
        self.send_photo('./Accuracy and Loss plot.jpg')
