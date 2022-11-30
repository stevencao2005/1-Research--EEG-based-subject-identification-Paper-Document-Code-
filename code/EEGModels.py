
"""
@author: Steven Cao"""

#IMPORT ALL NEEDED MODULES

#Standard library imports
import datetime
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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


class Classifier_EEGNET:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, lr=0.002,#lr=0.0009 #0.003
                 nb_filters=32, nb_epochs=50): #50

        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.callbacks = None
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose
        self.input_shape = input_shape
        self.nb_classes = nb_classes

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes, Chans=14, Samples=128,
               dropoutRate=0.5, kernLength=64, F1=8,
               D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        Chans          = input_shape[0]
        Samples        = input_shape[1]
        kernLength     = 128

        input1 = Input(shape=(Chans, Samples, 1))

        ##################################################################
        input_shape = (Chans, Samples, 1)
        block1 = Conv2D(F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)

        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutType(dropoutRate)(block1)

        block2 = SeparableConv2D(F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutType(dropoutRate)(block2)

        flatten = Flatten(name='flatten')(block2)
        dense   = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)
        model   = Model(inputs=input1, outputs=softmax)

        plot_model(model, to_file=self.output_directory+'model_architecture.png', show_shapes=True, show_layer_names=True)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.lr),
                      metrics=['accuracy'])
        # metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        #CREATING MODEL CALLBACKS
        reduce_lr            = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=30, min_lr=0.0001)

        file_path1            = self.output_directory + 'best_trained_model.h5'
        model_checkpoint1     = keras.callbacks.ModelCheckpoint(filepath=file_path1, monitor='loss', mode='min', save_best_only=True)

        file_path2            = self.output_directory + 'best_testing_model.h5'
        model_checkpoint2     = keras.callbacks.ModelCheckpoint(filepath=file_path2, monitor='val_accuracy', mode='max', save_best_only=True)

        log_dir              = self.output_directory + 'tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch")

        earlystopping        = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)

        self.callbacks = [reduce_lr, model_checkpoint1, model_checkpoint2, tensorboard_callback, earlystopping]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):


        # if not tf.config.list_physical_devices('GPU'):
        #     print('error no gpu')
        #     exit()
        # if not tf.test.is_gpu_available:
        #     print('error no gpu')
        #     exit()
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
        class CELoss_Bounded_Callback(keras.callbacks.Callback):
            def __init__(self, x_train, y_train, x_val, y_val):
                self.x_train = x_train
                self.y_train = y_train
                self.x_val   = x_val
                self.y_val   = y_val
                self.validation_loss_list = []
                self.training_loss_list   = []

            def on_epoch_end(self, epoch, logs={}):
                y_test_pred = np.asarray(self.model.predict(self.x_val))

                enc = OneHotEncoder(categories='auto')
                enc.fit(y_true.reshape(-1, 1))
                y_true_hot = enc.transform(y_true.reshape(-1, 1)).toarray()

                test_losses = []
                for label, pred in zip(y_true_hot, y_test_pred):
                    pred /= pred.sum(axis=-1, keepdims=True)
                    loss = np.sum(label * -np.log(pred), axis=-1, keepdims=False)
                    if loss > 2.3:
                        loss = 2.3
                    test_losses.append(loss)
                self.validation_loss_list.append(np.mean(test_losses))

                y_train_pred = np.asarray(self.model.predict(self.x_train))

                train_losses = []
                for label, pred in zip(y_train, y_train_pred):
                    pred /= pred.sum(axis=-1, keepdims=True)
                    loss = np.sum(label * -np.log(pred), axis=-1, keepdims=False)
                    if loss > 2.3:
                        loss = 2.3
                    train_losses.append(loss)
                self.training_loss_list.append(np.mean(train_losses))

        #two custom callbacks
        estimateRemTime = TimingCallback()
        ce_loss         = CELoss_Bounded_Callback(x_train, y_train, x_val, y_val)
        self.callbacks.append(estimateRemTime)
        self.callbacks.append(ce_loss)


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
        validation_loss_list = ce_loss.validation_loss_list
        training_loss_list = ce_loss.training_loss_list
        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration, validation_loss_list, training_loss_list)
        keras.backend.clear_session()

        return df_metrics


    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):

        #LOADING THE MODEL
        start_time = time.time()
        model_path = self.output_directory + 'best_testing_model.h5'
        model      = keras.models.load_model(model_path)

        #PREDICTING THE SAMPLES FROM THE TEST DATASET
        time1  = time.time()
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        time2  = time.time()
        print("time to go through the test set:", time2-time1)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred


def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    # input1   = Input(shape = (Chans, Samples, 1)) -- debug -- we give 4-second signals, not one-second signal
    input1   = Input(shape = (Chans, Samples*4, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


class Classifier_EEGNET_SSVEP:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, lr=0.009,
                 nb_filters=32, nb_epochs=30): #nb_epochs=1500

        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.callbacks = None
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes, Chans=64, Samples=128,
               dropoutRate=0.5, kernLength=64, F1=8,
               D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        n_feature_maps = 64
        Chans          = input_shape[0]
        Samples        = input_shape[1]
        input1         = Input(shape=(Chans, Samples, 1))

        ##################################################################
        block1 = Conv2D(F1, (1, kernLength), padding='same',
                        input_shape=(Chans, Samples, 1),
                        use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                                 depth_multiplier=D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutType(dropoutRate)(block1)

        block2 = SeparableConv2D(F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutType(dropoutRate)(block2)

        flatten = Flatten(name='flatten')(block2)
        dense   = Dense(nb_classes, name='dense')(flatten)
        softmax = Activation('softmax', name='softmax')(dense)
        model   = Model(inputs=input1, outputs=softmax)

        plot_model(model, to_file=self.output_directory+'model_architecture.png', show_shapes=True, show_layer_names=True)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.009),
                      metrics=['accuracy'])


        #CREATING MODEL CALLBACKS
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path        = self.output_directory + 'best_model.h5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        log_dir          = self.output_directory + 'tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch")

        self.callbacks = [reduce_lr, model_checkpoint, tensorboard_callback]

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

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.h5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        #SAVE PREDICTIONS
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        #CONVERT THE PREDICTED FROM BINARY TO INTEGER
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics


    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):

        #LOADING THE MODEL
        start_time = time.time()
        model_path = self.output_directory + 'best_model.h5'
        model = keras.models.load_model(model_path)

        #PREDICTING THE SAMPLES FROM THE TEST DATASET
        time1  = time.time()
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        time2  = time.time()
        print("time to go through the test set:", time2-time1)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred


def EEGNet_SSVEP(nb_classes = 12, Chans = 8, Samples = 256, 
             dropoutRate = 0.5, kernLength = 256, F1 = 96, 
             D = 1, F2 = 96, dropoutType = 'Dropout'):
    """ SSVEP Variant of EEGNet, as used in [1]. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. 
      D               : number of spatial filters to learn within each temporal
                        convolution.
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
      
      
    [1]. Waytowich, N. et. al. (2018). Compact Convolutional Neural Networks
    for Classification of Asynchronous Steady-State Visual Evoked Potentials.
    Journal of Neural Engineering vol. 15(6). 
    http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

def EEGNet_old(nb_classes, Chans = 64, Samples = 128, regRate = 0.0001,
           dropoutRate = 0.25, kernels = [(2, 32), (8, 4)], strides = (2, 4)):
    """ Keras Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)

    This model is the original EEGNet model proposed on arxiv
            https://arxiv.org/abs/1611.08024v2
    
    with a few modifications: we use striding instead of max-pooling as this 
    helped slightly in classification performance while also providing a 
    computational speed-up. 
    
    Note that we no longer recommend the use of this architecture, as the new
    version of EEGNet performs much better overall and has nicer properties.
    
    Inputs:
        
        nb_classes     : total number of final categories
        Chans, Samples : number of EEG channels and samples, respectively
        regRate        : regularization rate for L1 and L2 regularizations
        dropoutRate    : dropout fraction
        kernels        : the 2nd and 3rd layer kernel dimensions (default is 
                         the [2, 32] x [8, 4] configuration)
        strides        : the stride size (note that this replaces the max-pool
                         used in the original paper)
    
    """

    # start the model
    input_main   = Input((Chans, Samples))
    layer1       = Conv2D(16, (Chans, 1), input_shape=(Chans, Samples, 1),
                                 kernel_regularizer = l1_l2(l1=regRate, l2=regRate))(input_main)
    layer1       = BatchNormalization()(layer1)
    layer1       = Activation('elu')(layer1)
    layer1       = Dropout(dropoutRate)(layer1)
    
    permute_dims = 2, 1, 3
    permute1     = Permute(permute_dims)(layer1)
    
    layer2       = Conv2D(4, kernels[0], padding = 'same', 
                            kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                            strides = strides)(permute1)
    layer2       = BatchNormalization()(layer2)
    layer2       = Activation('elu')(layer2)
    layer2       = Dropout(dropoutRate)(layer2)
    
    layer3       = Conv2D(4, kernels[1], padding = 'same',
                            kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                            strides = strides)(layer2)
    layer3       = BatchNormalization()(layer3)
    layer3       = Activation('elu')(layer3)
    layer3       = Dropout(dropoutRate)(layer3)
    
    flatten      = Flatten(name = 'flatten')(layer3)
    
    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


class Classifier_DeepConvNet:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, lr=0.009,
                 nb_filters=32, nb_epochs=30): #nb_epochs=1500

        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.callbacks = None
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes, Chans=64, Samples=128,
               dropoutRate=0.5, kernLength=64, F1=8,
               D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        n_feature_maps = 64
        Chans          = input_shape[0]
        Samples        = input_shape[1]

        #########################################
        input_main = Input((Chans, Samples, 1))
        block1 = Conv2D(25, (1, 5),
                        input_shape=(Chans, Samples, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(25, (Chans, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
        block1 = Activation('elu')(block1)
        block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1 = Dropout(dropoutRate)(block1)

        block2 = Conv2D(50, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
        block2 = Activation('elu')(block2)
        block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2 = Dropout(dropoutRate)(block2)

        block3 = Conv2D(100, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
        block3 = Activation('elu')(block3)
        block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3 = Dropout(dropoutRate)(block3)

        block4 = Conv2D(200, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
        block4 = Activation('elu')(block4)
        block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4 = Dropout(dropoutRate)(block4)

        flatten = Flatten()(block4)
        dense   = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)
        model   = Model(inputs=input_main, outputs=softmax)

        plot_model(model, to_file=self.output_directory+'model_architecture.png', show_shapes=True, show_layer_names=True)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.009),
                      metrics=['accuracy'])

        #CREATING MODEL CALLBACKS
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.h5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        log_dir          = self.output_directory + 'tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch")

        self.callbacks = [reduce_lr, model_checkpoint, tensorboard_callback]

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

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.h5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        #SAVE PREDICTIONS
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        #CONVERT THE PREDICTED FROM BINARY TO INTEGER
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):

        #LOADING THE MODEL
        start_time = time.time()
        model_path = self.output_directory + 'best_model.h5'
        model = keras.models.load_model(model_path)

        #PREDICTING THE SAMPLES FROM THE TEST DATASET
        time1  = time.time()
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        time2  = time.time()
        print("time to go through the test set:", time2-time1)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred


def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   


class Classifier_ShallowConvNet:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, lr=0.009,
                 nb_filters=32, nb_epochs=30): #nb_epochs=1500

        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.callbacks = None
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes, Chans=64, Samples=128,
               dropoutRate=0.5, kernLength=64, F1=8,
               D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        n_feature_maps = 64
        Chans          = input_shape[0]
        Samples        = input_shape[1]

        ###################################
        input_main = Input((Chans, Samples, 1))
        block1 = Conv2D(40, (1, 13),
                        input_shape=(Chans, Samples, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(40, (Chans, 1), use_bias=False,
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
        block1 = Activation(square)(block1)
        block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
        block1 = Activation(log)(block1)
        block1 = Dropout(dropoutRate)(block1)

        flatten = Flatten()(block1)
        dense   = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)
        model   = Model(inputs=input_main, outputs=softmax)

        plot_model(model, to_file=self.output_directory+'model_architecture.png', show_shapes=True, show_layer_names=True)

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        #CREATING MODEL CALLBACKS
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.h5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        log_dir = self.output_directory + 'tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=True, update_freq="epoch")

        self.callbacks = [reduce_lr, model_checkpoint, tensorboard_callback]

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

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.h5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        #SAVE PREDICTIONS
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        #CONVERT THE PREDICTED FROM BINARY TO INTEGER
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics

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


def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


