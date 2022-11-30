
"""
@author: Steven Cao"""

#IMPORT ALL NEEDED MODULES

#Standard library imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import mne
import time
import utils
import pandas as pd

#Third party imports
import random
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import tensorboard
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


#Local application imports
from utils import calculate_metrics
from utils import case_by_case_analysis
from utils import create_directory
from utils import one_hot_encoder_only_ONE, one_hot_encoder_train_and_test
from utils import gettingInfoDL
from utils import generate_results_csv
from utils import read_all_datasets
from utils import read_dataset
from utils import standardizing_ONE_dataset, standardizing_datasets, standardizing_the_dataset_ALL
from utils import transform_mts_to_ucr_format
from utils import visualizeEEG_in_time_and_frequency_domain
from utils import visualize_filter
from utils import viz_cam
from utils import viz_for_survey_paper
from utils import HUMAN_train_test_no_split, HUMAN_train_test_split, MICE_train_test_no_split, MICE_train_test_split


#FOR FITTING THE MODEL ONTO THE DATASET
def fit_classifier_CV():

    #getting the train/test datasets

    global x_train
    global y_train
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test  = datasets_dict[dataset_name][2]
    y_test  = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    #transform the labels from integers to one hot vectors
    y_train, y_test = one_hot_encoder_train_and_test(y_train, y_test)

    # standardizing the train/test datasets for each channel
    x_train_preprocessed = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test_preprocessed  = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    channels             = [i for i in range(x_test.shape[1])]
    for channel in channels:
        x_train_channelData, x_test_channelData         = x_train[:, channel, :], x_test[:, channel, :]
        x_train_channelData, x_test_channelData         = standardizing_datasets(x_train_channelData,x_test_channelData)
        x_train_preprocessed[:, channel:channel + 1, :] = np.expand_dims(x_train_channelData, axis=1)
        x_test_preprocessed[:, channel:channel + 1, :]  = np.expand_dims(x_test_channelData, axis=1)

    x_train, y_train = shuffle(x_train_preprocessed, y_train)
    x_test, y_test   = shuffle(x_test_preprocessed, y_test)

    # save original y because later we will use binary
    y_true  = np.argmax(y_test, axis=1)

    y_train_true = list(np.argmax(y_train, axis=1))
    subjects = [i for i in range(9)]
    train_samples = dict()
    for subject in subjects:
        train_samples[str(subject)] = 0
    for i in y_train_true:
        for subject in subjects:
            if i == subject:
                train_samples[str(subject)] = train_samples[str(subject)] + 1



    #initializing the classifier to be trained on
    input_shape = x_train.shape[1:]
    classifier  = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    #saving the parameters of this trial
    info = gettingInfoDL(classifier)
    info.to_csv(output_directory + 'info.csv', index=False)

    #fitting the classifier using the preprocessed data
    classifier.fit(x_train, y_train, x_test, y_test, y_true)

#FOR FITTING THE MODEL ONTO THE DATASET
def fit_classifier_CV_train_mice_test_human():

    #getting the train/test datasets
    X_mice  = datasets_dict['mice'][0]
    Y_mice  = datasets_dict['mice'][1]
    X_human = datasets_dict['human'][0]
    Y_human = datasets_dict['human'][1]

    x_train, x_test, y_train, y_test = MICE_train_test_split(sfreq=256, lowpass=50, epoch_len=28, overlap=.9,
                                                                parent_folder=os.path.abspath('.') + '/data_preprocessed_mice_only_both_NR_euclidean_alignment')  #

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    y_train, y_test = one_hot_encoder_train_and_test(y_train, y_test)

    # standardizing the train/test datasets for each channel
    x_train_preprocessed = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test_preprocessed = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    channels = [i for i in range(x_test.shape[1])]
    for channel in channels:
        x_train_channelData, x_test_channelData = x_train[:, channel, :], x_test[:, channel, :]
        x_train_channelData, x_test_channelData = standardizing_datasets(x_train_channelData, x_test_channelData)
        x_train_preprocessed[:, channel:channel + 1, :] = np.expand_dims(x_train_channelData, axis=1)
        x_test_preprocessed[:, channel:channel + 1, :] = np.expand_dims(x_test_channelData, axis=1)

    x_train, y_train = shuffle(x_train_preprocessed, y_train)
    x_test, y_test = shuffle(x_test_preprocessed, y_test)

    # save original y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    # initializing the classifier to be trained on
    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    #saving the parameters of this trial
    info = gettingInfoDL(classifier)
    info.to_csv(output_directory + 'info.csv', index=False)
    i=1
    #fitting the classifier using the preprocessed data
    classifier.fit(x_train, y_train, x_test, y_test, y_true)
    i=1
#FOR FITTING THE MODEL ONTO THE DATASET
def fit_classifier_CV_train_mice_test_human_OLD():

    #getting the train/test datasets
    X_mice  = datasets_dict['mice'][0]
    Y_mice  = datasets_dict['mice'][1]
    X_human = datasets_dict['human'][0]
    Y_human = datasets_dict['human'][1]


    nb_classes = len(np.unique(Y_mice))

    #transform the labels from integers to one hot vectors
    Y_mice = one_hot_encoder_only_ONE(Y_mice)
    Y_human = one_hot_encoder_only_ONE(Y_human)


    # standardizing the train/test datasets for each channel
    X_mice_preprocessed = np.zeros((X_mice.shape[0], X_mice.shape[1], X_mice.shape[2]))
    X_human_preprocessed  = np.zeros((X_human.shape[0], X_human.shape[1], X_human.shape[2]))
    channels_human             = [i for i in range(X_human.shape[1])]
    for channel in channels_human:
        X_human_channelData  = X_human[:, channel, :]
        X_human_channelData  = standardizing_ONE_dataset(X_human_channelData)
        X_human_preprocessed[:, channel:channel + 1, :]  = np.expand_dims(X_human_channelData, axis=1)

    channels_mice             = [i for i in range(X_mice.shape[1])]
    for channel in channels_mice:
        X_mice_channelData   = X_mice[:, channel, :]
        X_mice_channelData   = standardizing_ONE_dataset(X_mice_channelData)
        X_mice_preprocessed[:, channel:channel + 1, :] = np.expand_dims(X_mice_channelData, axis=1)

    X_mice, Y_mice     = shuffle(X_mice_preprocessed, Y_mice)
    X_human, Y_human   = shuffle(X_human_preprocessed, Y_human)

    X_human_reshaped = np.reshape(X_human, (X_human.shape[0] * X_human.shape[1], X_human.shape[2]))
    Y_human_reshaped = np.zeros((Y_human.shape[0] * 6, Y_human.shape[1]))
    index = 0
    for value_sample in Y_human:
        for time in range(6):
            Y_human_reshaped[index, :] = value_sample
            index+=1
    X_human = np.expand_dims(X_human_reshaped, axis=1).copy()
    Y_human = Y_human_reshaped.copy()
    # save original y because later we will use binary
    y_true  = np.argmax(Y_human, axis=1)

    #initializing the classifier to be trained on
    input_shape = X_mice.shape[1:]
    classifier  = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    #saving the parameters of this trial
    info = gettingInfoDL(classifier)
    info.to_csv(output_directory + 'info.csv', index=False)
    i=1
    #fitting the classifier using the preprocessed data
    classifier.fit(X_mice, Y_mice, X_human, Y_human, y_true)
    i=1
def fit_classifier_IV():

    #getting the train/test datasets
    X_dataset = datasets_dict[dataset_name][0]
    Y_dataset = datasets_dict[dataset_name][1]

    z = 0
    total_metrics = np.zeros((4, 40))
    accuracy_model = np.zeros((4, 1))


    mice = 1
    if mice == 0:
        Control_human = ['102', '208', '457', '495', '556', '563', '744','XVZ2FYATE8M0SSF',
                                       'XVZ2FYAQH8YMGKY', 'XVZ2FYATE8X4YXQ', 'XVZ2FYATE84ZTFV', 'XVZ2FYATE8AJWX0',  # 9
                                       'XVZ2FYATE8BBO87', 'XVZ2FFAG8875MNV', 'XVZ2FYATE8ZYTB2', 'XVZ2FYATE8YDANN']  # 7
        Tbi_human = ['244', '340', '399', '424', '488', '510', '670''XVZ2FYAQH8WVIUC',
                                   'XVZ2FYATE84MSWI', 'XVZ2FYATE8B9R6X', 'XVZ2FYATE8DFIYL', 'XVZ2FYATE8FN4DS',  # 12
                                   'XVZ2FYATE8HSYB3', 'XVZ2FYATE8I41U0', 'XVZ2FYATE8JWW0A', 'XVZ2FYATE8K9U90',
                                   'XVZ2FYATE8W7FI6', 'XVZ2FYATE8Z362L', 'XVZ2FFAG885GFUG']  # 7

        Control_index_combinations = []
        Tbi_index_combinations = []
        # for i in range(40):
        #     Control_index_combinations.append(random.randint(0, 15))
        #     Tbi_index_combinations.append(random.randint(0, 18))

        # took 14 out
        Control_index_combinations = [14, 15, 3, 13, 1, 5, 11, 0, 7, 2, 2, 3, 4, 5, 3, 15, 5,
                                      11, 13, 14, 4, 13, 11, 9, 6, 0, 2, 5, 12, 14, 7, 14, 2, 8,
                                      12, 8, 5, 12, 8, 7]

        # took 1 out
        Tbi_index_combinations = [1, 12, 2, 9, 10, 17, 3, 17, 17, 5, 16, 12, 10, 14, 17, 17, 17,
                                  9, 9, 8, 13, 6, 18, 2, 3, 0, 2, 12, 10, 17, 10, 13, 1, 4,
                                  17, 3, 11, 5, 7, 14]

        num_control = len(Control_human)
        num_tbi = len(Tbi_human)
    elif mice == 1:

        #new
        # Control_mice = ['m010_TPE01_BaselineDay2_sham', 'm010_TPE03_BaselineDay2_sham','m010_TPE06_BaselineDay2_sham',
        #                 'm010_TPE08_BaselineDay2_sham','m010_TPE10_BaselineDay2_sham', 'm010_TPE12_BaselineDay2_sham',
        #                 'm010_TPE15_BaselineDay2_sham', 'm010_TPE18_BaselineDay2_sham','m010_TPE19_BaselineDay2_sham']

        #old
        # Control_mice = ['Sham102_BL5', 'Sham103_BL5', 'Sham104_BL5', 'Sham107_BL5', 'Sham108_BL5']

        #both
        Control_mice = ['Sham102_BL5', 'Sham103_BL5', 'Sham104_BL5', 'Sham107_BL5', 'Sham108_BL5',
                        'm010_TPE01_BaselineDay2_sham', 'm010_TPE03_BaselineDay2_sham', 'm010_TPE06_BaselineDay2_sham',
                        'm010_TPE08_BaselineDay2_sham', 'm010_TPE10_BaselineDay2_sham', 'm010_TPE12_BaselineDay2_sham',
                        'm010_TPE15_BaselineDay2_sham', 'm010_TPE18_BaselineDay2_sham', 'm010_TPE19_BaselineDay2_sham'
                        ]

        #new
        # Tbi_mice = ['m001_M60_BaselineDay1_CCI', 'm001_L56_BaselineDay1_CCI', 'm001_L55_BaselineDay1_CCI',
        #             'm001_K51_BaselineDay1_CCI', 'm001_J43_BaselineDay1_CCI', 'm001_I41_BaselineDay1_CCI',
        #             'm001_H36_BaselineDay1_CCI', 'm001_G29_BaselineDay1_CCI', 'm001_E20_BaselineDay1_CCI']

        #old
        # Tbi_mice = ['TBI101_BL5', 'TBI102_BL5', 'TBI103_BL5', 'TBI104_BL5', 'TBI106_BL5']

        #both
        Tbi_mice = ['TBI101_BL5', 'TBI102_BL5', 'TBI103_BL5', 'TBI104_BL5', 'TBI106_BL5',
                    'm001_M60_BaselineDay1_CCI', 'm001_L56_BaselineDay1_CCI', 'm001_L55_BaselineDay1_CCI',
                    'm001_K51_BaselineDay1_CCI', 'm001_J43_BaselineDay1_CCI', 'm001_I41_BaselineDay1_CCI',
                    'm001_H36_BaselineDay1_CCI', 'm001_G29_BaselineDay1_CCI', 'm001_E20_BaselineDay1_CCI'
                    ]
        Control_index_combinations = []
        Tbi_index_combinations = []

        # old: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4
        # new: 8, 3, 5, 2, 1, 6, 7, 0, 4, 5, 7, 8, 3, 4, 0
        Control_index_combinations = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6,  7, 7, 8,  8,  9, 9,  10, 10, 11, 11, 12, 12, 13, 13]     #both: 0,  4, 7, 9, 8, 2, 3, 1 , 5, 11, 12
        Tbi_index_combinations     = [1, 4, 2, 5, 3, 6, 4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9,  13, 10,  0, 11,  1, 12,  2,  4]     #both: 10, 3, 6, 1, 2, 4, 5, 12, 9, 10,  5
        # old: 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4
        # new: 1, 2, 4, 7, 5, 8, 6, 1, 3, 4, 8, 3, 6, 8, 2

        num_control = len(Control_mice)
        num_tbi = len(Tbi_mice)


    i=1
    for key in range(len(Control_index_combinations)):
        if key == 14:
            c = 0
        i = Control_index_combinations[key]
        j = Tbi_index_combinations[key]
        for k in range(1):
            mice = 1
            if mice == 0:
                x_train, x_test, y_train, y_test = HUMAN_train_test_no_split(sfreq=200, lowpass=50,epoch_len=30, overlap=.9,
                                IV_control=Control_human[i], IV_tbi=Tbi_human[j],
                               parent_folder= os.path.abspath('.') + '/data_preprocessed')
            elif mice == 1:
                x_train, x_test, y_train, y_test = MICE_train_test_no_split(sfreq=256, lowpass=50,epoch_len=28, overlap=.9,
                                IV_control=Control_mice[i], IV_tbi=Tbi_mice[j],
                               parent_folder= os.path.abspath('.') + '/data_preprocessed_mice_only_both_NR')

            nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

            #transform the labels from integers to one hot vectors
            y_train, y_test = one_hot_encoder_train_and_test(y_train, y_test)

            #standardizing the train/test datasets for each channel
            x_train_preprocessed = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
            x_test_preprocessed  = np.zeros((x_test.shape[0] , x_test.shape[1] , x_test.shape[2]))
            channels             = [i for i in range(x_test.shape[1])]
            for channel in channels:
                x_train_channelData, x_test_channelData      = x_train[:,channel,:], x_test[:,channel,:]
                x_train_channelData, x_test_channelData      = standardizing_datasets(x_train_channelData, x_test_channelData)
                x_train_preprocessed[:,channel:channel+1, :] = np.expand_dims(x_train_channelData, axis=1)
                x_test_preprocessed[:,channel:channel+1,:]   = np.expand_dims(x_test_channelData, axis=1)
            x_train, y_train = shuffle(x_train_preprocessed, y_train)
            x_test, y_test = shuffle(x_test_preprocessed, y_test)

            # save original y because later we will use binary
            y_true  = np.argmax(y_test, axis=1)

            #initializing the classifier to be trained on
            input_shape = x_train.shape[1:]
            classifier  = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

            #saving the parameters of this trial
            info = gettingInfoDL(classifier)
            info.to_csv(output_directory + 'info.csv', index=False)

            #fitting the classifier using the preprocessed data
            classifier.fit(x_train, y_train, x_test, y_test, y_true)
            y_pred, df_metrics = predict_IV(x_test, y_test, y_true, output_directory,
                                         return_df_metrics=True)
            accuracy_model[0,k] = float(df_metrics['accuracy'])
            accuracy_model[1,k] = float(df_metrics['precision'])
            accuracy_model[2,k] = float(df_metrics['recall'])
            accuracy_model[3,k] = float(df_metrics['f1 score'])
            # accuracy_model[4,k] = float(df_metrics['auc_score'])
            # accuracy_model[5,k] = float(df_metrics['specificity'])
        m=0
        average_accuracy_model = np.mean(accuracy_model, axis=1)
        total_metrics[:, z] = average_accuracy_model

        z = z + 1
    l = 0
    average_total_metrics = np.mean(total_metrics[:, :25], axis=1)

    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['accuracy', 'precision', 'recall', 'f1 score']) #, 'auc_score', 'specificity'
    res['accuracy']    = average_total_metrics[0]
    res['precision']   = average_total_metrics[1]
    res['recall']      = average_total_metrics[2]
    res['f1 score']    = average_total_metrics[3]
    # res['auc_score']   = average_total_metrics['auc_score']
    # res['specificity'] = average_total_metrics['specificity']

    res.to_csv(output_directory + 'df_metrics_IV.csv', index=False)



#FOR AFTER FITTING THE MODEL, AND WANT TO LOAD THE MODEL AND ANALYZE ITS PERFORMANCE
def fitted_classifier_CV():

    #GETTING THE TRAIN/TEST DATASETS
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test  = datasets_dict[dataset_name][2]
    y_test  = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    #TRANSFROM THE LABELS FROM INTEGERS TO ONE HOT VECTORS
    enc     = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test  = enc.transform(y_test.reshape(-1, 1)).toarray()

    #STANDARDIZING THE TRAIN/TEST DATASETS FOR EACH CHANNEL
    # standardizing the train/test datasets for each channel
    x_train_preprocessed = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test_preprocessed = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    channels = [i for i in range(x_test.shape[1])]
    for channel in channels:
        x_train_channelData, x_test_channelData = x_train[:, channel, :], x_test[:, channel, :]
        x_train_channelData, x_test_channelData = standardizing_datasets(x_train_channelData, x_test_channelData)
        x_train_preprocessed[:, channel:channel + 1, :] = np.expand_dims(x_train_channelData, axis=1)
        x_test_preprocessed[:, channel:channel + 1, :] = np.expand_dims(x_test_channelData, axis=1)

    x_train, y_train = shuffle(x_train_preprocessed, y_train)
    x_test, y_test   = shuffle(x_test_preprocessed, y_test)

    # save orignal y because later we will use binary
    y_true  = np.argmax(y_test, axis=1)

    #INITIALIZING THE CLASSIFIER TO BE TRAINED ON
    input_shape = x_train.shape[1:]
    classifier  = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    #LOADING THE MODEL AND EVALUATE IT USING THE TEST DATASET
    y_pred, df_metrics = predict(x_test, y_true, x_train, y_train, y_test, output_directory, return_df_metrics=True)

    i=1
    hist = pd.read_csv(os.path.abspath('.') + '/results/eegnet/BED/history.csv')
    plot_accuracy_metric_dict(hist, output_directory + 'epochs_accuracy.png')

    return x_test, y_true, x_train, y_train, y_test, y_pred, df_metrics

def fitted_classifier_CV_train_mice_test_human():
    # getting the train/test datasets
    X_human = datasets_dict['human'][0]
    Y_human = datasets_dict['human'][1]

    x_train, x_test, y_train, y_test = HUMAN_train_test_split(sfreq=200, lowpass=50, epoch_len=6.4, overlap=.9,
                                                             parent_folder=os.path.abspath(
                                                                 '.') + '/data_preprocessed')
    X_human = np.append(x_train, x_test, axis=0)
    Y_human = np.append(y_train, y_test, axis=0)
    nb_classes = len(np.unique(Y_human))

    # transform the labels from integers to one hot vectors
    Y_human = one_hot_encoder_only_ONE(Y_human)

    # standardizing the train/test datasets for each channel
    X_human_preprocessed = np.zeros((X_human.shape[0], X_human.shape[1], X_human.shape[2]))
    channels_human = [i for i in range(X_human.shape[1])]
    for channel in channels_human:
        X_human_channelData = X_human[:, channel, :]
        X_human_channelData = standardizing_ONE_dataset(X_human_channelData)
        X_human_preprocessed[:, channel:channel + 1, :] = np.expand_dims(X_human_channelData, axis=1)

    X_human, Y_human = shuffle(X_human_preprocessed, Y_human)
    method =1
    if method == 0:
        X_human_reshaped = np.reshape(X_human, (X_human.shape[0] * X_human.shape[1], X_human.shape[2]))
        Y_human_reshaped = np.zeros((Y_human.shape[0] * 6, Y_human.shape[1]))
        index = 0
        for value_sample in Y_human:
            for time in range(6):
                Y_human_reshaped[index, :] = value_sample
                index += 1
        X_human = np.expand_dims(X_human_reshaped, axis=1).copy()
        Y_human = Y_human_reshaped.copy()
    elif method == 1:
        X_human_reshaped = np.reshape(X_human, (X_human.shape[0], X_human.shape[1] * X_human.shape[2]), order="A")
        X_human = np.expand_dims(X_human_reshaped, axis=1).copy()
    # save original y because later we will use binary

    # import matplotlib
    # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(6, figsize=(20, 6))
    # for i in range(6):
    #     ax[i].plot(X_human[0, i, :])
    # plt.show()
    #
    # import matplotlib
    # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, figsize=(20, 6))
    # ax.plot(X_human_reshaped[0, :])
    # plt.show()
    # initializing the classifier to be trained on
    input_shape = X_human.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
    I=1

    X_train, X_test, Y_train, Y_test = train_test_split(X_human, Y_human, test_size=0.2, random_state=42)
    y_true = np.argmax(Y_human, axis=1)
    # y_pred, df_metrics = continue_Training(X_train, X_test, Y_train, Y_test, y_true, output_directory, return_df_metrics=True)
    #LOADING THE MODEL AND EVALUATE IT USING THE TEST DATASET
    y_pred, df_metrics = predict(X_human, Y_human, y_true, output_directory, return_df_metrics=True)

    i=1
    # hist = pd.read_csv(os.path.abspath('.') + '/results/eegnet/BED/history.csv')
    # plot_accuracy_metric_dict(hist, output_directory + 'epochs_accuracy.png')

    return x_test, y_true, x_train, y_train, y_test, y_pred, df_metrics


def fitted_classifier_CV_train_mice_retrain_human_test_human():


    Control_human = ['102', '208', '457', '495', '556', '563', '744', 'XVZ2FYATE8M0SSF',
                     'XVZ2FYAQH8YMGKY', 'XVZ2FYATE8X4YXQ', 'XVZ2FYATE84ZTFV', 'XVZ2FYATE8AJWX0',  # 9
                     'XVZ2FYATE8BBO87', 'XVZ2FFAG8875MNV', 'XVZ2FYATE8ZYTB2', 'XVZ2FYATE8YDANN']  # 7
    Tbi_human = ['244', '340', '399', '424', '488', '510', '670', 'XVZ2FYAQH8WVIUC',
                 'XVZ2FYATE84MSWI', 'XVZ2FYATE8B9R6X', 'XVZ2FYATE8DFIYL', 'XVZ2FYATE8FN4DS',  # 12
                 'XVZ2FYATE8HSYB3', 'XVZ2FYATE8I41U0', 'XVZ2FYATE8JWW0A', 'XVZ2FYATE8K9U90',
                 'XVZ2FYATE8W7FI6', 'XVZ2FYATE8Z362L', 'XVZ2FFAG885GFUG']  # 7

    Control_index_combinations = []
    Tbi_index_combinations = []
    # for i in range(40):
    #     Control_index_combinations.append(random.randint(0, 15))
    #     Tbi_index_combinations.append(random.randint(0, 18))

    # took 14 out
    Control_index_combinations = [14, 15, 3, 13, 1, 5, 11, 0, 7, 2, 2, 3, 4, 5, 3, 15, 5,
                                  11, 13, 14, 4, 13, 11, 9, 6, 0, 2, 5, 12, 14, 7, 14, 2, 8,
                                  12, 8, 5, 12, 8, 7]

    # took 1 out
    Tbi_index_combinations     = [1, 12, 2, 9, 10, 17, 3, 17, 17, 5, 16, 12, 10, 14, 17, 17, 17,
                                  9, 9, 8, 13, 6, 18, 2, 3, 0, 2, 12, 10, 17, 10, 13, 1, 4,
                                  17, 3, 11, 5, 7, 14]

    num_control = len(Control_human)
    num_tbi = len(Tbi_human)

    i = Control_index_combinations[3]
    j = Tbi_index_combinations[3]
    for k in range(1):
        x_train, x_test, y_train, y_test = HUMAN_train_test_no_split(sfreq=200, lowpass=50, epoch_len=6.4,
                                                                     overlap=.9,
                                                                     IV_control=Control_human[i],
                                                                     IV_tbi=Tbi_human[j],
                                                                     parent_folder=os.path.abspath(
                                                                         '.') + '/data_preprocessed')
        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

        # transform the labels from integers to one hot vectors
        y_train, y_test = one_hot_encoder_train_and_test(y_train, y_test)

        # standardizing the train/test datasets for each channel
        x_train_preprocessed = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        x_test_preprocessed = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        channels = [i for i in range(x_test.shape[1])]
        for channel in channels:
            x_train_channelData, x_test_channelData = x_train[:, channel, :], x_test[:, channel, :]
            x_train_channelData, x_test_channelData = standardizing_datasets(x_train_channelData, x_test_channelData)
            x_train_preprocessed[:, channel:channel + 1, :] = np.expand_dims(x_train_channelData, axis=1)
            x_test_preprocessed[:, channel:channel + 1, :] = np.expand_dims(x_test_channelData, axis=1)
        x_train, y_train = shuffle(x_train_preprocessed, y_train)
        x_test, y_test = shuffle(x_test_preprocessed, y_test)

        #----reshape to make it compatible to the inputLayer
        X_train_reshaped = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]), order="A")
        x_train = np.expand_dims(X_train_reshaped, axis=1).copy()

        X_test_reshaped = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]), order="A")
        x_test = np.expand_dims(X_test_reshaped, axis=1).copy()

        # save original y because later we will use binary
        y_true = np.argmax(y_test, axis=1)

        # initializing the classifier to be trained on
        input_shape = x_train.shape[1:]


        #RETRAINING THE MODEL USING HUMAN DATA, THEN LOADING THE RE-TRAINED MODEL AND EVALUATE IT USING THE TEST DATASET
        y_pred, df_metrics = continue_Training_and_TEST(x_train, x_test, y_train, y_test, y_true, output_directory, return_df_metrics=True)



    i=1
    # hist = pd.read_csv(os.path.abspath('.') + '/results/eegnet/BED/history.csv')
    # plot_accuracy_metric_dict(hist, output_directory + 'epochs_accuracy.png')

    return x_test, y_true, x_train, y_train, y_test, y_pred, df_metrics

def fitted_classifier_IV():

    #getting the train/test datasets
    X_dataset = datasets_dict[dataset_name][0]
    Y_dataset = datasets_dict[dataset_name][1]

    z = 0
    total_metrics = np.zeros((6, 40))
    accuracy_model = np.zeros((6, 10))

    Control_human = ['102', '208', '457', '495', '556', '563', '744','XVZ2FYATE8M0SSF',
                                   'XVZ2FYAQH8YMGKY', 'XVZ2FYATE8X4YXQ', 'XVZ2FYATE84ZTFV', 'XVZ2FYATE8AJWX0',  # 9
                                   'XVZ2FYATE8BBO87', 'XVZ2FFAG8875MNV', 'XVZ2FYATE8ZYTB2', 'XVZ2FYATE8YDANN']  # 7
    Tbi_human = ['244', '340', '399', '424', '488', '510', '670''XVZ2FYAQH8WVIUC',
                               'XVZ2FYATE84MSWI', 'XVZ2FYATE8B9R6X', 'XVZ2FYATE8DFIYL', 'XVZ2FYATE8FN4DS',  # 12
                               'XVZ2FYATE8HSYB3', 'XVZ2FYATE8I41U0', 'XVZ2FYATE8JWW0A', 'XVZ2FYATE8K9U90',
                               'XVZ2FYATE8W7FI6', 'XVZ2FYATE8Z362L', 'XVZ2FFAG885GFUG']  # 7

    Control_index_combinations = []
    Tbi_index_combinations = []
    # for i in range(40):
    #     Control_index_combinations.append(random.randint(0, 15))
    #     Tbi_index_combinations.append(random.randint(0, 18))

    #took 14 out
    Control_index_combinations = [14, 15,  3, 13,  1,  5, 11,  0,  7,  2,  2,  3,  4,  5,  3, 15,  5,
          11, 13, 14,  4, 13, 11,  9,  6,  0,  2,  5, 12, 14,  7, 14,  2,  8,
          12,  8,  5, 12,  8,  7]

    #took 1 out
    Tbi_index_combinations = [1, 12,  2,  9, 10, 17,  3, 17, 17,  5, 16, 12, 10, 14, 17, 17, 17,
           9,  9,  8, 13,  6, 18,  2,  3,  0,  2, 12, 10, 17, 10, 13,  1,  4,
          17,  3, 11,  5,  7, 14]



    x_train, x_test, y_train, y_test = train_test_no_split(sfreq=200, lowpass=50,epoch_len=30, overlap=.9,
                    IV_control=Control_human[0], IV_tbi=Tbi_human[0],
                   parent_folder= os.path.abspath('.') + '/data_preprocessed')

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    #transform the labels from integers to one hot vectors
    y_train, y_test = one_hot_encoder(y_train, y_test)

    #standardizing the train/test datasets for each channel
    x_train_preprocessed = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test_preprocessed  = np.zeros((x_test.shape[0] , x_test.shape[1] , x_test.shape[2]))
    channels             = [i for i in range(x_test.shape[1])]
    for channel in channels:
        x_train_channelData, x_test_channelData      = x_train[:,channel,:], x_test[:,channel,:]
        x_train_channelData, x_test_channelData      = standardizing_datasets(x_train_channelData, x_test_channelData)
        x_train_preprocessed[:,channel:channel+1, :] = np.expand_dims(x_train_channelData, axis=1)
        x_test_preprocessed[:,channel:channel+1,:]   = np.expand_dims(x_test_channelData, axis=1)
    x_train, y_train = shuffle(x_train_preprocessed, y_train)
    x_test, y_test   = shuffle(x_test_preprocessed, y_test)

    # save original y because later we will use binary
    y_true  = np.argmax(y_test, axis=1)

    #initializing the classifier to be trained on
    input_shape = x_train.shape[1:]
    classifier  = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    #LOADING THE MODEL AND EVALUATE IT USING THE TEST DATASET
    y_pred, df_metrics = predict(x_test, y_true, x_train, y_train, y_test, output_directory, return_df_metrics=True)

    return x_test, y_true, x_train, y_train, y_test, y_pred, df_metrics


def predict_IV(x_test, y_test, y_true, output_directory, return_df_metrics=True):

    #LOADING THE MODEL
    model_path  = output_directory + 'best_testing_model.h5'
    model       = keras.models.load_model(model_path)
    # model.load_weights(output_directory + 'model_init.h5')
    #PREDICTING THE SAMPLES FROM THE TEST DATASET

    start_time  = time.time()
    y_pred      = model.predict(x_test, batch_size=64)
    print("time to go through the test set:", time.time()-start_time)

    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(y_true.reshape(-1, 1))
    y_true_hot  = enc.transform(y_true.reshape(-1, 1)).toarray()

    # losses = []
    # for label, pred in zip(y_true_hot, y_pred):
    #     pred /= pred.sum(axis=-1, keepdims=True)
    #     losses.append(np.sum(label * -np.log(pred), axis=-1, keepdims=False))
    # print(losses)
    # print(np.mean(losses))
    # [0.10536055  0.8046684  0.0618754]

    # import matplotlib.pyplot as plt
    # matplotlib.use("TkAgg")
    # plt.hist(losses, bins=50)
    # plt.show()

    if return_df_metrics:
        y_pred     = np.argmax(y_pred, axis=1)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        return y_pred, df_metrics
def predict(x_test, y_test, y_true, output_directory, return_df_metrics=True):

    #LOADING THE MODEL
    model_path  = output_directory + 'last_model.h5'
    model       = keras.models.load_model(model_path)
    # model.load_weights(output_directory + 'model_init.h5')
    #PREDICTING THE SAMPLES FROM THE TEST DATASET

    start_time  = time.time()
    y_pred      = model.predict(x_test, batch_size=64)
    print("time to go through the test set:", time.time()-start_time)

    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(y_true.reshape(-1, 1))
    y_true_hot  = enc.transform(y_true.reshape(-1, 1)).toarray()

    # losses = []
    # for label, pred in zip(y_true_hot, y_pred):
    #     pred /= pred.sum(axis=-1, keepdims=True)
    #     losses.append(np.sum(label * -np.log(pred), axis=-1, keepdims=False))
    # print(losses)
    # print(np.mean(losses))
    # [0.10536055  0.8046684  0.0618754]

    # import matplotlib.pyplot as plt
    # matplotlib.use("TkAgg")
    # plt.hist(losses, bins=50)
    # plt.show()

    if return_df_metrics:
        y_pred     = np.argmax(y_pred, axis=1)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        return y_pred, df_metrics

def continue_Training_and_TEST(X_train, X_test, Y_train, Y_test, y_true, output_directory, return_df_metrics=True):

        # LOADING THE MODEL
        model_path = output_directory + 'last_model.h5'
        model = keras.models.load_model(model_path)
        # model.load_weights(output_directory + 'model_init.h5')
        # PREDICTING THE SAMPLES FROM THE TEST DATASET

        mini_batch_size = 64

        class CELoss_Bounded_Callback(keras.callbacks.Callback):
            def __init__(self, x_train, y_train, x_val, y_val):
                self.x_train = x_train
                self.y_train = y_train
                self.x_val   = x_val
                self.y_val   = y_val
                self.validation_loss_list = []
                self.training_loss_list   = []

            def on_epoch_end(self, epoch, logs={}):

                enc = OneHotEncoder(categories='auto')
                enc.fit(y_true.reshape(-1, 1))
                y_true_hot = enc.transform(y_true.reshape(-1, 1)).toarray()

                y_train_pred = np.asarray(self.model.predict(self.x_train))
                train_losses = []
                for label, pred in zip(self.y_train, y_train_pred):
                    pred /= pred.sum(axis=-1, keepdims=True)
                    loss = np.sum(label * -np.log(pred), axis=-1, keepdims=False)
                    if loss > 2.3:
                        loss = 2.3
                    train_losses.append(loss)
                print('')
                print('training loss', np.mean(train_losses))
                self.training_loss_list.append(np.mean(train_losses))

                y_test_pred = np.asarray(self.model.predict(self.x_val))
                test_losses = []
                for label, pred in zip(y_true_hot, y_test_pred):
                    pred /= pred.sum(axis=-1, keepdims=True)
                    loss = np.sum(label * -np.log(pred), axis=-1, keepdims=False)
                    if loss > 2.3:
                        loss = 2.3
                    test_losses.append(loss)
                print('validation loss', np.mean(test_losses))
                self.validation_loss_list.append(np.mean(test_losses))


        #two custom callbacks
        ce_loss         = CELoss_Bounded_Callback(X_train, Y_train, X_test, Y_test)

        file_path1            = output_directory + 'best_re_trained_model.h5'
        model_checkpoint1     = keras.callbacks.ModelCheckpoint(filepath=file_path1, monitor='loss', mode='min', save_best_only=True)

        file_path2            = output_directory + 'best_re_testing_model.h5'
        model_checkpoint2     = keras.callbacks.ModelCheckpoint(filepath=file_path2, monitor='val_loss', mode='min', save_best_only=True)

        callbacks = [ce_loss, model_checkpoint1, model_checkpoint2]

        hist = model.fit(X_train, Y_train, batch_size=mini_batch_size, epochs=5,
                              verbose=True, validation_data=(X_test, Y_test), callbacks=callbacks)

        i=1
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.figure()
        fig = plt.plot(ce_loss.validation_loss_list)
        plt.show()

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.figure()
        fig = plt.plot(ce_loss.training_loss_list)
        plt.show()
        model.save(output_directory + 'retrained_last_model.h5')

        # LOADING THE MODEL
        start_time = time.time()
        model_path = output_directory + 'best_re_testing_model.h5'
        model = keras.models.load_model(model_path)

        # PREDICTING THE SAMPLES FROM THE TEST DATASET
        time1 = time.time()
        y_pred = model.predict(X_test, batch_size=mini_batch_size)
        time2 = time.time()
        print("time to go through the test set:", time2 - time1)
        y_pred = np.argmax(y_pred, axis=1)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        i=1
        return y_pred, df_metrics




def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):

    #GETTING THE CLASSIFIER THAT WE WANT TO BE TRAINED ON THE DATA
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'eegnet':
        from classifiers import EEGModels
        return EEGModels.Classifier_EEGNET(output_directory, input_shape, nb_classes, verbose)


if __name__ == '__main__':
    #STARTING CODE
    root_dir = os.path.abspath('.')

    # DIRECTIONS ON THE DATASET TO USE, THE CLASSIFIER TO BE USED, AND THE VALIDATION METHOD
    #BED OR TBI ; inception OR resnet OR eegnet ; IV OR CV OR TL:MH
    sys.argv.extend(['BED', 'resnet', 'CV'])

    # INITIALIZING VARIABLES FOR CREATING A DIRECTORY TO PUT THE RESULTS OF THE MODEL IN
    dataset_name        = sys.argv[1]
    classifier_name     = sys.argv[2]
    output_directory    = root_dir + '/results/' + classifier_name + '/' + dataset_name + '/'
    test_dir_df_metrics = output_directory + 'df_metrics.csv'
    print('Method: ', dataset_name, classifier_name)


    # IF THE MODEL HAS ALREADY BEEN TRAINED ON, THEN WE CAN LOAD IT AND ANALYZE ITS PERFORMANCE.
    # IF NOT TRAINED YET, THEN WE CAN TRAIN THE MODEL.
    if sys.argv[3] == 'CV' and os.path.exists(test_dir_df_metrics):
        print('Already done')
        loadModelAnswer = input("Would you like to load the best model:  (y/n)")
        if loadModelAnswer == 'y':
            # READING THE DATASET AND SPLITTING IT INTO TRAIN/TEST DATASETS
            datasets_dict                                                = read_dataset(root_dir, dataset_name)

            # LOADING THE ALREADY FITTED CLASSIFIER AND EVALUATING IT
            x_test, y_true, x_train, y_train, y_test, y_pred, df_metrics = fitted_classifier_CV()
            predictions                                                  = case_by_case_analysis(y_true, y_pred)
            # visualizeEEG_in_time_and_frequency_domain(subject=8, amountOfSamples=1, predictions=predictions
            #                                           , dataset=x_test)
    if sys.argv[3] == 'IV' and os.path.exists(test_dir_df_metrics):
        print('Already done')
        loadModelAnswer = input("Would you like to load the best model:  (y/n)")
        if loadModelAnswer == 'y':
            # READING THE DATASET AND SPLITTING IT INTO TRAIN/TEST DATASETS
            datasets_dict                                                = read_dataset(root_dir, dataset_name)

            # LOADING THE ALREADY FITTED CLASSIFIER AND EVALUATING IT
            x_test, y_true, x_train, y_train, y_test, y_pred, df_metrics = fitted_classifier_IV()
            predictions                                                  = case_by_case_analysis(y_true, y_pred)
            # visualizeEEG_in_time_and_frequency_domain(subject=8, amountOfSamples=1, predictions=predictions
            #
    elif sys.argv[3] == 'IV':
        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, dataset_name)
        fit_classifier_IV()
    elif sys.argv[3] == 'TL:MH' and os.path.exists(test_dir_df_metrics):
        print('Already done')
        loadModelAnswer = input("Would you like to load the best model:  (y/n)")
        if loadModelAnswer == 'y':
            # READING THE DATASET AND SPLITTING IT INTO TRAIN/TEST DATASETS
            datasets_dict = read_dataset(root_dir, dataset_name)

            # LOADING THE ALREADY FITTED CLASSIFIER AND EVALUATING IT
            x_test, y_true, x_train, y_train, y_test, y_pred, df_metrics = fitted_classifier_CV_train_mice_test_human()
            predictions = case_by_case_analysis(y_true, y_pred)
            # visualizeEEG_in_time_and_frequency_domain(subject=8, amountOfSamples=1, predictions=predictions
    elif sys.argv[3] == 'TL:MH':
        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, dataset_name)
        fit_classifier_CV_train_mice_test_human()
    else:
        # CREATING THE OUTPUT DIRECTORY WHERE ALL THE RESULTS FROM THE MODEL BEING TRAINED WILL BE LOCATED
        create_directory(output_directory)

        # READING THE DATASET AND SPLITTING IT INTO TRAIN/TEST DATASETS
        datasets_dict = read_dataset(root_dir, dataset_name)

        # FITTING THE CLASSIFIER ONTO THE DATASET AND EVALUATING THE MODEL
        fit_classifier_CV()

        # CREATING A DIRECTORY TO SAY THAT WE FINISHED TRAINING AND EVALUATING THE MODEL
        print('DONE')
        create_directory(output_directory + '/DONE')

    pass


