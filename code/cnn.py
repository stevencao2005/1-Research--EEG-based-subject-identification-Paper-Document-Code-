
"""
@author: Steven Cao"""

# CNN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers

#IMPORT ALL NEEDED MODULES

#Standard library imports
import datetime
import numpy as np
import time

#Third party imports
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import tensorflow.keras as keras

#Local application imports
from utils import calculate_metrics
from utils import save_logs

class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True, batch_size=16, lr=0.009, nb_epochs=10):
        self.output_directory = output_directory
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.batch_size = batch_size
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.h5')

        return

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)
        output_layer  = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)
        model         = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])

        #CREATING MODEL CALLBACKS
        file_path        = self.output_directory + 'best_model.h5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
        log_dir              = self.output_directory + 'tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch")

        self.callbacks = [model_checkpoint, tensorboard_callback]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training

        mini_batch_size = self.batch_size

        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.h5')

        model = keras.models.load_model(self.output_directory + 'best_model.h5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

        keras.backend.clear_session()

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):

        # LOADING THE MODEL
        start_time = time.time()
        model_path = self.output_directory + 'best_model.h5'
        model = keras.models.load_model(model_path)

        # PREDICTING THE SAMPLES FROM THE TEST DATASET
        time1 = time.time()
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        time2 = time.time()
        print("time to go through the test set:", time2 - time1)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
