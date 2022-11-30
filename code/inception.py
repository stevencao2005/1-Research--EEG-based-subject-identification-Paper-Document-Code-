
"""
@author: Steven Cao"""

#IMPORT ALL NEEDED MODULES

#Standard library imports
import datetime
import math
import numpy as np
import time

#Third party imports
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from timeit import default_timer as timer


#Local application imports
from utils import calculate_metrics
from utils import save_logs
from utils import save_test_duration


"""
The Inception Network
    Parameters:
        - output_directory (str): the str to the output directory which will contain the results.
        - input_shape (int,int) : the shape of each sample 
        - nb_classes (int)      : the number of classes 
        - verbose (boolean)     : whether or not you want to print out the training process.
        - batch_size (int)      : how many samples are you sending to the model before it updates its weights
        - lr (float)            : the learning rate of the model
        - nb_filters (int)      : how many filters for each convolutional layer
        - depth (int)           : how many inception blocks the Inception Network will have
        - kernel_size (int)     : the size of each kernel for each convolutional layer
        - nb_epochs (int)       : the number of epochs
    Return: none

"""

class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, lr=0.1,
                 nb_filters=64, use_residual=True, use_bottleneck=True, depth=9, kernel_size=41, nb_epochs=5): #nb_epochs=1500
        self.output_directory = output_directory
        self.nb_filters       = nb_filters
        self.use_residual     = use_residual
        self.use_bottleneck   = use_bottleneck
        self.depth            = depth
        self.kernel_size      = kernel_size - 1
        self.callbacks        = None
        self.batch_size       = batch_size
        self.bottleneck_size  = 32
        self.nb_epochs        = nb_epochs
        self.lr               = lr
        self.verbose          = verbose
        self.input_shape      = input_shape
        self.nb_classes       = nb_classes

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.h5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        conv_list     = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))
        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6     = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='elu')(x)
        x = keras.layers.Dropout(0.5)(x)

        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        # shortcut_y = keras.layers.Dropout(0.4)(shortcut_y)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('elu')(x)
        x = keras.layers.Dropout(0.5)(x)

        return x

    def build_model(self, input_shape, nb_classes):
        # input_reduced = keras.layers.GlobalAveragePooling1D(input_shape)
        input_shape = (input_shape[0], input_shape[1])
        input_layer = keras.layers.Input(input_shape)
        x           = input_layer
        input_res   = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer    = keras.layers.GlobalAveragePooling1D()(x)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model        = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # decay_rate = self.lr/self.nb_epochs
        # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(self.lr, momentum=0.99),
        #               metrics=['accuracy'])
        plot_model(model, to_file=self.output_directory+'model_architecture.png', show_shapes=True, show_layer_names=True)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])

        #CREATING MODEL CALLBACKS
        reduce_lr            = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=30, min_lr=0.0001)

        file_path1            = self.output_directory + 'best_trained_model.h5'
        model_checkpoint1     = keras.callbacks.ModelCheckpoint(filepath=file_path1, monitor='loss', mode='min', save_best_only=True)

        file_path2            = self.output_directory + 'best_testing_model.h5'
        model_checkpoint2     = keras.callbacks.ModelCheckpoint(filepath=file_path2, monitor='val_loss', mode='min', save_best_only=True)

        log_dir              = self.output_directory + 'tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch")
        # earlystopping        = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=25)

        self.callbacks = [reduce_lr, model_checkpoint1, model_checkpoint2, tensorboard_callback]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error no gpu')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        class TimingCallback(keras.callbacks.Callback):
            def __init__(self, logs={}):
                self.logs = []
                self.time_logs = []
                self.batch_size = 64
                self.batch_num = math.ceil(x_train.shape[0] / self.batch_size)
                self.batches_done = 0
                self.batches_left = 0
                self.time_left = 0
                self.prev_time = 0
                self.current_time = 0

            def on_epoch_begin(self, epoch, logs={}):
                self.prev_time = timer()
                self.batches_done = epoch * self.batch_num + self.params["steps"]

            def on_epoch_end(self, epoch, logs={}):
                self.batches_left = self.params["epochs"] * self.batch_num - self.batches_done
                self.current_time = timer()
                self.time_logs.append((self.current_time - self.prev_time) / self.batch_num)
                self.time_left = datetime.timedelta(
                    seconds=self.batches_left * (sum(self.time_logs) / len(self.time_logs)))
                print("     time left:", self.time_left)
                self.prev_time = timer()
        # class CELoss_Bounded_Callback(keras.callbacks.Callback):
        #     def __init__(self, x_train, y_train, x_val, y_val):
        #         self.x_train = x_train
        #         self.y_train = y_train
        #         self.x_val   = x_val
        #         self.y_val   = y_val
        #         self.validation_loss_list = []
        #         self.training_loss_list   = []
        #
        #     def on_epoch_end(self, epoch, logs={}):
        #         y_test_pred = np.asarray(self.model.predict(self.x_val))
        #
        #         enc = OneHotEncoder(categories='auto')
        #         enc.fit(y_true.reshape(-1, 1))
        #         y_true_hot = enc.transform(y_true.reshape(-1, 1)).toarray()
        #
        #         test_losses = []
        #         for label, pred in zip(y_true_hot, y_test_pred):
        #             pred /= pred.sum(axis=-1, keepdims=True)
        #             loss = np.sum(label * -np.log(pred), axis=-1, keepdims=False)
        #             if loss > 2.3:
        #                 loss = 2.3
        #             test_losses.append(loss)
        #         self.validation_loss_list.append(np.mean(test_losses))
        #
        #         y_train_pred = np.asarray(self.model.predict(self.x_train))
        #
        #         train_losses = []
        #         for label, pred in zip(y_train, y_train_pred):
        #             pred /= pred.sum(axis=-1, keepdims=True)
        #             loss = np.sum(label * -np.log(pred), axis=-1, keepdims=False)
        #             if loss > 2.3:
        #                 loss = 2.3
        #             train_losses.append(loss)
        #         self.training_loss_list.append(np.mean(train_losses))

        estimateRemTime = TimingCallback()
        # ce_loss         = CELoss_Bounded_Callback(x_train, y_train, x_val, y_val)
        self.callbacks.append(estimateRemTime)
        # self.callbacks.append(ce_loss)
        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.h5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val, return_df_metrics=False)

        #SAVE PREDICTIONS
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        #CONVERT THE PREDICTED FROM BINARY TO INTEGER
        y_pred     = np.argmax(y_pred, axis=1)
        # validation_loss_list = ce_loss.validation_loss_list
        # training_loss_list = ce_loss.training_loss_list
        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)
        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):

        #LOADING THE MODEL
        start_time = time.time()
        model_path = self.output_directory + 'best_testing_model.h5'
        model = keras.models.load_model(model_path)

        #PREDICTING THE SAMPLES FROM THE TEST DATASET
        time1  = time.time()
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        time2  = time.time()
        print("time to go through the test set:", time2 - time1)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
