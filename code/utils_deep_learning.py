
"""
@author: Steven Cao"""

#IMPORT ALL NEEDED MODULES

#Standard library imports
from builtins import print
import numpy as np
import operator
import os
import glob
import math
import mne
import pandas as pd
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

#Third party imports
import keras

from scipy.interpolate import interp1d
from scipy.io import loadmat
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


#Local application imports
import utils

#FUNCTIONS
def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory

def read_dataset(root_dir, dataset_name):
    #LOAD THE DATASET AND SPLIT IT INTO TRAIN/TEST DATASETS
    if sys.argv[3] == 'CV':
        datasets_dict = {}
        cur_root_dir = root_dir.replace('-temp', '')

        file_name = cur_root_dir + '/' + dataset_name + '/'
        x_train = np.load(file_name + 'X_train.npy')
        y_train = np.load(file_name + 'y_train.npy')
        x_test = np.load(file_name + 'X_test.npy')
        y_test = np.load(file_name + 'y_test.npy')

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    elif sys.argv[3] == 'IV':
        datasets_dict = {}
        cur_root_dir = root_dir.replace('-temp', '')

        file_name = cur_root_dir + '/' + dataset_name + '/'
        X = np.load(file_name + 'X.npy')
        Y = np.load(file_name + 'Y.npy')
        datasets_dict[dataset_name] = (X.copy(), Y.copy())
    elif sys.argv[3] == 'TL:MH':
        datasets_dict = {}
        cur_root_dir = root_dir.replace('-temp', '')

        file_name = cur_root_dir + '/' + dataset_name + '/'
        X_human = np.load(file_name + 'X.npy')
        Y_human = np.load(file_name + 'Y.npy')
        datasets_dict['human'] = (X_human.copy(), Y_human.copy())

        file_name = cur_root_dir + '/' + dataset_name + '/'
        X_mice = np.load(file_name + 'X_mice.npy')
        Y_mice = np.load(file_name + 'Y_mice.npy')
        datasets_dict['mice'] = (X_mice.copy(), Y_mice.copy())


    # elif archive_name == 'UCRArchive_2018':
    #     root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
    #     # df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
    #     #
    #     # df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)
    #     df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.txt',delim_whitespace=True, header=None)
    #     df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.txt', delim_whitespace=True, header=None)
    #
    #
    #     y_train = df_train.values[:, 0]
    #     y_test = df_test.values[:, 0]
    #
    #     x_train = df_train.drop(columns=[0])
    #     x_test = df_test.drop(columns=[0])
    #
    #     x_train.columns = range(x_train.shape[1])
    #     x_test.columns = range(x_test.shape[1])
    #
    #     x_train = x_train.values
    #     x_test = x_test.values
    #
    #     # znorm
    #     std_ = x_train.std(axis=1, keepdims=True)
    #     std_[std_ == 0] = 1.0
    #     x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
    #
    #     std_ = x_test.std(axis=1, keepdims=True)
    #     std_[std_ == 0] = 1.0
    #     x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
    #
    #     datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
    #                                    y_test.copy())
    # else:
    #     file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
    #     x_train, y_train = readucr(file_name + '_TRAIN')
    #     x_test, y_test = readucr(file_name + '_TEST')
    #     datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
    #                                    y_test.copy())

    return datasets_dict

def read_all_datasets(root_dir, archive_name, split_val=False):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []

    if archive_name == 'mts_archive':

        for dataset_name in MTS_DATASET_NAMES:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset + 'x_train.npy')
            y_train = np.load(root_dir_dataset + 'y_train.npy')
            x_test = np.load(root_dir_dataset + 'x_test.npy')
            y_test = np.load(root_dir_dataset + 'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
    elif archive_name == 'UCRArchive_2018':
        for dataset_name in DATASET_NAMES_2018:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'

            df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

            df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    else:
        for dataset_name in DATASET_NAMES:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict

def lowpass_filter(arr, sfreq, lowpass):
    '''
    Performs low-pass filtering.
    '''
    return mne.filter.filter_data(arr, sfreq=sfreq, l_freq=None, h_freq=lowpass, verbose=0)

def add_epochs(X_lst, y_lst, X, y, sfreq, epoch_len, overlap):
    '''
    Add epochs to the list of data for training or testing.
    '''
    window = epoch_len * sfreq
    for i in np.arange(X.shape[1] // (window * (1 - overlap))) * sfreq:
        start = int(i)
        stop = int(i + window)
        epoch = X[:, start:stop]
        # only use epochs with length == window
        if epoch.shape[-1] == window:
            X_lst.append(epoch)
            y_lst.append(y)
def HUMAN_train_test_no_split(sfreq=200,
                        lowpass=50,
                        epoch_len=4,
                        overlap=.9,
                        IV_control='102',
                        IV_tbi='244',
                        parent_folder = None,
                        data_folder=None):
    '''
    Lowpass filter, cut data into epochs, and split them into training and
    test sets.

    NOTE: the third session of every subject is used as test data.

    Parameters:
        - sfreq (int): sampling rate. Default at 256.
        - lowpass (int, float): lowpass frequency. Default at 50.
        - epoch_len (int, float): the length of every epoch in seconds. Default
        at 10 seconds.
        - overlap (float): the proportion by which every pair of contiguous
        epochs overlaps. Must be within [0, 1]. Default at 0.9.
        - parent_folder (str): the parent directory of the folders containing
        preprocessed files. Default is None under the assumption that this
        script file is already in the parent directory.
        - data_folder (str): the folder storing the training and test data.
        Default is None, which entails dumping those data to the same folder
        hosting this script.

    Return: none

    '''
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    Control_human = ['102', '208', '457', '495', '556', '563', '744', 'XVZ2FYATE8M0SSF',
                     'XVZ2FYAQH8YMGKY', 'XVZ2FYATE8X4YXQ', 'XVZ2FYATE84ZTFV', 'XVZ2FYATE8AJWX0',
                     'XVZ2FYATE8BBO87', 'XVZ2FFAG8875MNV', 'XVZ2FYATE8ZYTB2', 'XVZ2FYATE8YDANN']
    Tbi_human = ['244', '340', '399', '424', '488', '510', '670', 'XVZ2FYAQH8WVIUC',
                 'XVZ2FYATE84MSWI', 'XVZ2FYATE8B9R6X', 'XVZ2FYATE8DFIYL', 'XVZ2FYATE8FN4DS',
                 'XVZ2FYATE8HSYB3', 'XVZ2FYATE8I41U0', 'XVZ2FYATE8JWW0A', 'XVZ2FYATE8K9U90',
                 'XVZ2FYATE8W7FI6', 'XVZ2FYATE8Z362L', 'XVZ2FFAG885GFUG']

    if parent_folder != None:
        subject_dirs = glob.glob(os.path.abspath(parent_folder) + os.sep + '*')
    else:
        parent_folder = '.' + os.sep
        subject_dirs = glob.glob('.' + os.sep + '*')

    for d in subject_dirs:
        subject = d.split(os.sep)[-1]
        for session_dir in glob.glob(d + os.sep + '*.npy'):
            X = lowpass_filter(np.load(session_dir), sfreq, lowpass)
            X = shuffle(X)

            if subject in Control_human:
                y = 'control'
                if subject != IV_control:
                    add_epochs(X_train, y_train, X, y, sfreq, epoch_len, overlap)
                else:
                    add_epochs(X_test, y_test, X, y, sfreq, epoch_len, overlap)
            else:
                y = 'tbi'
                if subject != IV_tbi:
                    add_epochs(X_train, y_train, X, y, sfreq, epoch_len, overlap)
                else:
                    add_epochs(X_test, y_test, X, y, sfreq, epoch_len, overlap)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def HUMAN_train_test_split(sfreq=200,
                        lowpass=50,
                        epoch_len=4,
                        overlap=.9,
                        parent_folder = None,
                        data_folder=None):
    '''
    Lowpass filter, cut data into epochs, and split them into training and
    test sets.

    NOTE: the third session of every subject is used as test data.

    Parameters:
        - sfreq (int): sampling rate. Default at 256.
        - lowpass (int, float): lowpass frequency. Default at 50.
        - epoch_len (int, float): the length of every epoch in seconds. Default
        at 10 seconds.
        - overlap (float): the proportion by which every pair of contiguous
        epochs overlaps. Must be within [0, 1]. Default at 0.9.
        - parent_folder (str): the parent directory of the folders containing
        preprocessed files. Default is None under the assumption that this
        script file is already in the parent directory.
        - data_folder (str): the folder storing the training and test data.
        Default is None, which entails dumping those data to the same folder
        hosting this script.

    Return: none

    '''
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    Control_human = ['102', '208', '457', '495', '556', '563', '744', 'XVZ2FYATE8M0SSF',
                     'XVZ2FYAQH8YMGKY', 'XVZ2FYATE8X4YXQ', 'XVZ2FYATE84ZTFV', 'XVZ2FYATE8AJWX0',
                     'XVZ2FYATE8BBO87', 'XVZ2FFAG8875MNV', 'XVZ2FYATE8ZYTB2', 'XVZ2FYATE8YDANN']
    Tbi_human = ['244', '340', '399', '424', '488', '510', '670', 'XVZ2FYAQH8WVIUC',
                 'XVZ2FYATE84MSWI', 'XVZ2FYATE8B9R6X', 'XVZ2FYATE8DFIYL', 'XVZ2FYATE8FN4DS',
                 'XVZ2FYATE8HSYB3', 'XVZ2FYATE8I41U0', 'XVZ2FYATE8JWW0A', 'XVZ2FYATE8K9U90',
                 'XVZ2FYATE8W7FI6', 'XVZ2FYATE8Z362L', 'XVZ2FFAG885GFUG']

    if parent_folder != None:
        subject_dirs = glob.glob(os.path.abspath(parent_folder) + os.sep + '*')
    else:
        parent_folder = '.' + os.sep
        subject_dirs = glob.glob('.' + os.sep + '*')

    for d in subject_dirs:
        subject = d.split(os.sep)[-1]
        for session_dir in glob.glob(d + os.sep + '*.npy'):
            X = lowpass_filter(np.load(session_dir), sfreq, lowpass)
            X = shuffle(X)

            if subject in Control_human:
                y = 'control'
                X_train_data = X[:, :int(X.shape[1] * 0.8)]
                add_epochs(X_train, y_train, X_train_data, y, sfreq, epoch_len, overlap)
                X_test_data = X[:, int(X.shape[1] * 0.8):]
                add_epochs(X_test, y_test, X_test_data, y, sfreq, epoch_len, overlap)
            else:
                y = 'tbi'
                X_train_data = X[:, :int(X.shape[1] * 0.8)]
                add_epochs(X_train, y_train, X_train_data, y, sfreq, epoch_len, overlap)
                X_test_data = X[:, int(X.shape[1] * 0.8):]
                add_epochs(X_test, y_test, X_test_data, y, sfreq, epoch_len, overlap)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def MICE_train_test_no_split(sfreq=200,
                        lowpass=50,
                        epoch_len=4,
                        overlap=.9,
                        IV_control='102',
                        IV_tbi='244',
                        parent_folder = None,
                        data_folder=None):
    '''
    Lowpass filter, cut data into epochs, and split them into training and
    test sets.

    NOTE: the third session of every subject is used as test data.

    Parameters:
        - sfreq (int): sampling rate. Default at 256.
        - lowpass (int, float): lowpass frequency. Default at 50.
        - epoch_len (int, float): the length of every epoch in seconds. Default
        at 10 seconds.
        - overlap (float): the proportion by which every pair of contiguous
        epochs overlaps. Must be within [0, 1]. Default at 0.9.
        - parent_folder (str): the parent directory of the folders containing
        preprocessed files. Default is None under the assumption that this
        script file is already in the parent directory.
        - data_folder (str): the folder storing the training and test data.
        Default is None, which entails dumping those data to the same folder
        hosting this script.

    Return: none

    '''
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    Control_mice = ['Sham102_BL5', 'Sham103_BL5', 'Sham104_BL5', 'Sham107_BL5', 'Sham108_BL5',
                    'm010_TPE01_BaselineDay2_sham', 'm010_TPE03_BaselineDay2_sham',
                    'm010_TPE06_BaselineDay2_sham', 'm010_TPE08_BaselineDay2_sham',
                    'm010_TPE10_BaselineDay2_sham', 'm010_TPE12_BaselineDay2_sham',
                    'm010_TPE15_BaselineDay2_sham', 'm010_TPE18_BaselineDay2_sham',
                    'm010_TPE19_BaselineDay2_sham']
    Tbi_mice = ['TBI101_BL5', 'TBI102_BL5', 'TBI103_BL5', 'TBI104_BL5', 'TBI106_BL5',
                'm001_M60_BaselineDay1_CCI', 'm001_L56_BaselineDay1_CCI', 'm001_L55_BaselineDay1_CCI',
                'm001_K51_BaselineDay1_CCI', 'm001_J43_BaselineDay1_CCI', 'm001_I41_BaselineDay1_CCI',
                'm001_H36_BaselineDay1_CCI', 'm001_G29_BaselineDay1_CCI', 'm001_E20_BaselineDay1_CCI']

    if parent_folder != None:
        subject_dirs = glob.glob(os.path.abspath(parent_folder) + os.sep + '*')
    else:
        parent_folder = '.' + os.sep
        subject_dirs = glob.glob('.' + os.sep + '*')

    for d in subject_dirs:
        subject = d.split(os.sep)[-1]
        for session_dir in glob.glob(d + os.sep + '*.npy'):
            X = lowpass_filter(np.load(session_dir), sfreq, lowpass)
            X = shuffle(X)

            if subject in Control_mice:
                y = 'control'
                if subject != IV_control:
                    add_epochs(X_train, y_train, X, y, sfreq, epoch_len, overlap)
                else:
                    add_epochs(X_test, y_test, X, y, sfreq, epoch_len, overlap)
            else:
                y = 'tbi'
                if subject != IV_tbi:
                    add_epochs(X_train, y_train, X, y, sfreq, epoch_len, overlap)
                else:
                    add_epochs(X_test, y_test, X, y, sfreq, epoch_len, overlap)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def MICE_train_test_split(sfreq=200,
                        lowpass=50,
                        epoch_len=4,
                        overlap=.9,
                        parent_folder = None,
                        data_folder=None):
    '''
    Lowpass filter, cut data into epochs, and split them into training and
    test sets.

    NOTE: the third session of every subject is used as test data.

    Parameters:
        - sfreq (int): sampling rate. Default at 256.
        - lowpass (int, float): lowpass frequency. Default at 50.
        - epoch_len (int, float): the length of every epoch in seconds. Default
        at 10 seconds.
        - overlap (float): the proportion by which every pair of contiguous
        epochs overlaps. Must be within [0, 1]. Default at 0.9.
        - parent_folder (str): the parent directory of the folders containing
        preprocessed files. Default is None under the assumption that this
        script file is already in the parent directory.
        - data_folder (str): the folder storing the training and test data.
        Default is None, which entails dumping those data to the same folder
        hosting this script.

    Return: none

    '''
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    Control_mice = ['Sham102_BL5', 'Sham103_BL5', 'Sham104_BL5', 'Sham107_BL5', 'Sham108_BL5',
                    'm010_TPE01_BaselineDay2_sham', 'm010_TPE03_BaselineDay2_sham',
                    'm010_TPE06_BaselineDay2_sham', 'm010_TPE08_BaselineDay2_sham',
                    'm010_TPE10_BaselineDay2_sham', 'm010_TPE12_BaselineDay2_sham',
                    'm010_TPE15_BaselineDay2_sham', 'm010_TPE18_BaselineDay2_sham',
                    'm010_TPE19_BaselineDay2_sham']
    Tbi_mice = ['TBI101_BL5', 'TBI102_BL5', 'TBI103_BL5', 'TBI104_BL5', 'TBI106_BL5',
                'm001_M60_BaselineDay1_CCI', 'm001_L56_BaselineDay1_CCI', 'm001_L55_BaselineDay1_CCI',
                'm001_K51_BaselineDay1_CCI', 'm001_J43_BaselineDay1_CCI', 'm001_I41_BaselineDay1_CCI',
                'm001_H36_BaselineDay1_CCI', 'm001_G29_BaselineDay1_CCI', 'm001_E20_BaselineDay1_CCI']

    if parent_folder != None:
        subject_dirs = glob.glob(os.path.abspath(parent_folder) + os.sep + '*')
    else:
        parent_folder = '.' + os.sep
        subject_dirs = glob.glob('.' + os.sep + '*')

    for d in subject_dirs:
        subject = d.split(os.sep)[-1]
        for session_dir in glob.glob(d + os.sep + '*.npy'):
            X = lowpass_filter(np.load(session_dir), sfreq, lowpass)
            X = shuffle(X)

            if subject in Control_mice:
                y = 'control'
                X_train_data = X[:, :int(X.shape[1] * 0.8)]
                add_epochs(X_train, y_train, X_train_data, y, sfreq, epoch_len, overlap)
                X_test_data = X[:, int(X.shape[1] * 0.8):]
                add_epochs(X_test, y_test, X_test_data, y, sfreq, epoch_len, overlap)
            else:
                y = 'tbi'
                X_train_data = X[:, :int(X.shape[1] * 0.8)]
                add_epochs(X_train, y_train, X_train_data, y, sfreq, epoch_len, overlap)
                X_test_data = X[:, int(X.shape[1] * 0.8):]
                add_epochs(X_test, y_test, X_test_data, y, sfreq, epoch_len, overlap)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n = x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])

    n = x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length

def transform_to_same_length(x, n_var, max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')
            new_ts = f(idx_new)
            ucr_x[i, :, j] = new_ts

    return ucr_x

def transform_mts_to_ucr_format():
    mts_root_dir = '/mnt/Other/mtsdata/'
    mts_out_dir = '/mnt/nfs/casimir/archives/mts_archive/'
    for dataset_name in MTS_DATASET_NAMES:
        # print('dataset_name',dataset_name)

        out_dir = mts_out_dir + dataset_name + '/'

        # if create_directory(out_dir) is None:
        #     print('Already_done')
        #     continue

        a = loadmat(mts_root_dir + dataset_name + '/' + dataset_name + '.mat')
        a = a['mts']
        a = a[0, 0]

        dt = a.dtype.names
        dt = list(dt)

        for i in range(len(dt)):
            if dt[i] == 'train':
                x_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'test':
                x_test = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'trainlabels':
                y_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'testlabels':
                y_test = a[i].reshape(max(a[i].shape))

        # x_train = a[1][0]
        # y_train = a[0][:,0]
        # x_test = a[3][0]
        # y_test = a[2][:,0]

        n_var = x_train[0].shape[0]

        max_length = get_func_length(x_train, x_test, func=max)
        min_length = get_func_length(x_train, x_test, func=min)

        print(dataset_name, 'max', max_length, 'min', min_length)
        print()
        # continue

        x_train = transform_to_same_length(x_train, n_var, max_length)
        x_test = transform_to_same_length(x_test, n_var, max_length)

        # save them
        np.save(out_dir + 'x_train.npy', x_train)
        np.save(out_dir + 'y_train.npy', y_train)
        np.save(out_dir + 'x_test.npy', x_test)
        np.save(out_dir + 'y_test.npy', y_test)

        print('Done')
def one_hot_encoder_only_ONE(Y):
    enc     = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(Y.reshape(-1, 1))
    Y = enc.transform(Y.reshape(-1, 1)).toarray()
    return Y
def one_hot_encoder_train_and_test(y_train, y_test):
    enc     = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test  = enc.transform(y_test.reshape(-1, 1)).toarray()
    return y_train, y_test

def standardizing_ONE_dataset(X):
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    return X

def standardizing_the_dataset_ALL(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train, X_test

def standardizing_datasets(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test  = sc.transform(X_test)
    return X_train, X_test


def gettingInfoDL(classifier):
    info = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                       columns=['lr', 'batch_size', 'nb_epochs', 'nb_filters', 'input_shape', 'nb_classes'])
    info['lr']              = classifier.lr
    info['batch_size']      = classifier.batch_size
    info['nb_epochs']       = classifier.nb_epochs
    info['nb_filters']      = classifier.nb_filters
    info['input_shape']     = str(classifier.input_shape)
    info['nb_classes']      = classifier.nb_classes

    return info


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 7), dtype=np.float), index=[0],
                       columns=['accuracy', 'precision', 'recall', 'f1 score', 'duration', 'auc_score', 'specificity'])
    results = []
    labels  = list(np.unique(y_true))

    res['accuracy']  = accuracy_score(y_true, y_pred)
    res['recall']    = recall_score(y_true, y_pred,labels=labels, average='macro')
    res['precision'] = precision_score(y_true, y_pred,labels=labels, average='macro')

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['f1 score'] = f1_score(y_true, y_pred,labels=labels, average='macro')
    res['duration']    = duration

    if len(np.unique(labels)) == 2:
        # Confusion Matrix
        tn, fp, fn, tp       = confusion_matrix(y_true, y_pred, labels=labels).ravel()
        res['specificity']   = tn / (tn + fp)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        res['auc_score']     = metrics.auc(fpr, tpr)
    else:
        res['specificity']   = "N/A"
        res['auc_score']     = "N/A"
    return res

def case_by_case_analysis(y_true, y_pred):
    correctPredictionIndexes = []
    indexes                  = np.arange(1, len(y_true) + 1)
    labels                   = np.unique(y_true)
    subjectNames             = [i for i in labels]
    subjectNames_str         = [str(i) for i in labels]
    confusionMX              = confusion_matrix(y_true, y_pred)
    print(confusionMX)

    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=subjectNames_str))



    print("amount of samples predicted correctly for each subject")
    for i in range(confusionMX.shape[0]):
        print('    subject' + str(i), confusionMX[i, i])


    predictions = dict()
    for name in subjectNames:
        predictions[name] = dict()
        predictions[name]['correct predictions'] = {}
        predictions[name]['incorrect predictions'] = {}
    for i, value in enumerate(y_true):
        for j in subjectNames:
            if value == j:
                subjectName = value
        if y_pred[i] == y_true[i]:
            predictions[subjectName]['correct predictions'][i] = (indexes[i] - 1, y_pred[i], y_true[i])
        else:
            predictions[subjectName]['incorrect predictions'][i] = (indexes[i] - 1, y_pred[i], y_true[i])

    subjects = subjectNames.copy()
    subjectClassify = subjectNames.copy()
    values = dict()
    for subjectName in subjects:
        values[subjectName] = dict()
        for subject in subjectClassify:
            values[subjectName][subject] = 0
    for subjectName in subjectNames:
        subject = predictions[subjectName]['incorrect predictions']
        for sample in list(subject.items()):
            predictedValue = sample[1][1]
            for i in subjects:
                if predictedValue == i:
                    values[subjectName][i] = values[subjectName][i] + 1
    for subjectName in subjectNames:
        subject = values[subjectName]
        print('subject',subjectName)
        for subjectName in subjectClassify:
            print('   {0}   {1}'.format(subjectName, subject[subjectName]))
    from sklearn.metrics import cohen_kappa_score
    Coh_k_s = cohen_kappa_score(y_true, y_pred)

    from sklearn.metrics import matthews_corrcoef
    Mat_corf = matthews_corrcoef(y_true, y_pred)

    display =0
    if display ==1:
        for i in predictions:
            for j in predictions[i]:
                for k in predictions[i][j]:
                    sampleNum = predictions[i][j][k]
                    print("Sample {0}: Predicted as {1}  Actual Value is {2}".format(sampleNum[0],sampleNum[1], sampleNum[2]))

    return predictions

def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

def generate_results_csv(output_file_name, root_dir):
    res = pd.DataFrame(data=np.zeros((0, 7), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'precision', 'accuracy', 'recall', 'duration'])
    for classifier_name in CLASSIFIERS:
        for archive_name in ARCHIVE_NAMES:
            datasets_dict = read_all_datasets(root_dir, archive_name)
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                for dataset_name in datasets_dict.keys():
                    output_dir = root_dir + '/results/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + dataset_name + '/' + 'df_metrics.csv'
                    if not os.path.exists(output_dir):
                        continue
                    df_metrics = pd.read_csv(output_dir)
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['archive_name'] = archive_name
                    df_metrics['dataset_name'] = dataset_name
                    res = pd.concat((res, df_metrics), axis=0, sort=False)

    res.to_csv(root_dir + output_file_name, index=False)
    # aggreagte the accuracy for iterations on same dataset
    res = pd.DataFrame({
        'accuracy': res.groupby(
            ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    }).reset_index()

    return res

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_accuracy_metric(hist, file_name, metric='accuracy'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_accuracy_metric_dict(hist, file_name, metric= 'accuracy'):
    plt.figure()
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def save_logs_t_leNet(output_directory, hist, y_pred, y_true, duration):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

def save_logs(output_directory, hist, y_pred, y_true, duration, validation_loss_list, training_loss_list, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    validation_loss_list
    training_loss_list

    # plot losses and accuracy
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')
    plot_accuracy_metric(hist, output_directory + 'epochs_accuracy.png')

    return df_metrics

def visualize_filter(root_dir):
    import tensorflow.keras as keras
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    filters = model.layers[1].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    colors = [(255 / 255, 160 / 255, 14 / 255), (181 / 255, 87 / 255, 181 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255)]

    idx = 10
    idx_filter = 1

    filter = filters[:, 0, idx_filter]

    plt.figure(1)
    plt.plot(filter + 0.5, color='gray', label='filter')
    for c in classes:
        c_x_train = x_train[np.where(y_train == c)]
        convolved_filter_1 = new_feed_forward([c_x_train])[0]

        idx_c = int(c) - 1

        plt.plot(c_x_train[idx], color=colors[idx_c], label='class' + str(idx_c) + '-raw')
        plt.plot(convolved_filter_1[idx, :, idx_filter], color=colors_conv[idx_c], label='class' + str(idx_c) + '-conv')
        plt.legend()

    plt.savefig(root_dir + 'convolution-' + dataset_name + '.pdf')

    return 1

def viz_perf_themes(root_dir, df):
    df_themes = df.copy()
    themes_index = []
    # add the themes
    for dataset_name in df.index:
        themes_index.append(utils.constants.dataset_types[dataset_name])

    themes_index = np.array(themes_index)
    themes, themes_counts = np.unique(themes_index, return_counts=True)
    df_themes.index = themes_index
    df_themes = df_themes.rank(axis=1, method='min', ascending=False)
    df_themes = df_themes.where(df_themes.values == 1)
    df_themes = df_themes.groupby(level=0).sum(axis=1)
    df_themes['#'] = themes_counts

    for classifier in CLASSIFIERS:
        df_themes[classifier] = df_themes[classifier] / df_themes['#'] * 100
    df_themes = df_themes.round(decimals=1)
    df_themes.to_csv(root_dir + 'tab-perf-theme.csv')

def viz_perf_train_size(root_dir, df):
    df_size = df.copy()
    train_sizes = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(datasets_dict[dataset_name][0])
        train_sizes.append(train_size)

    train_sizes = np.array(train_sizes)
    bins = np.array([0, 100, 400, 800, 99999])
    train_size_index = np.digitize(train_sizes, bins)
    train_size_index = bins[train_size_index]

    df_size.index = train_size_index
    df_size = df_size.rank(axis=1, method='min', ascending=False)
    df_size = df_size.groupby(level=0, axis=0).mean()
    df_size = df_size.round(decimals=2)

    print(df_size.to_string())
    df_size.to_csv(root_dir + 'tab-perf-train-size.csv')

def viz_perf_classes(root_dir, df):
    df_classes = df.copy()
    class_numbers = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(np.unique(datasets_dict[dataset_name][1]))
        class_numbers.append(train_size)

    class_numbers = np.array(class_numbers)
    bins = np.array([0, 3, 4, 6, 8, 13, 9999])
    class_numbers_index = np.digitize(class_numbers, bins)
    class_numbers_index = bins[class_numbers_index]

    df_classes.index = class_numbers_index
    df_classes = df_classes.rank(axis=1, method='min', ascending=False)
    df_classes = df_classes.groupby(level=0, axis=0).mean()
    df_classes = df_classes.round(decimals=2)

    print(df_classes.to_string())
    df_classes.to_csv(root_dir + 'tab-perf-classes.csv')

def viz_perf_length(root_dir, df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths = np.array(lengths)
    bins = np.array([0, 81, 251, 451, 700, 1001, 9999])
    lengths_index = np.digitize(lengths, bins)
    lengths_index = bins[lengths_index]

    df_lengths.index = lengths_index
    df_lengths = df_lengths.rank(axis=1, method='min', ascending=False)
    df_lengths = df_lengths.groupby(level=0, axis=0).mean()
    df_lengths = df_lengths.round(decimals=2)

    print(df_lengths.to_string())
    df_lengths.to_csv(root_dir + 'tab-perf-lengths.csv')

def viz_plot(root_dir, df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths_index = np.array(lengths)

    df_lengths.index = lengths_index

    plt.scatter(x=df_lengths['fcn'], y=df_lengths['resnet'])
    plt.ylim(ymin=0, ymax=1.05)
    plt.xlim(xmin=0, xmax=1.05)
    # df_lengths['fcn']
    plt.savefig(root_dir + 'plot.pdf')

def viz_for_survey_paper(root_dir, filename='results-ucr-mts.csv'):
    df = pd.read_csv(root_dir + filename, index_col=0)
    df = df.T
    df = df.round(decimals=2)

    # get table performance per themes
    # viz_perf_themes(root_dir,df)

    # get table performance per train size
    # viz_perf_train_size(root_dir,df)

    # get table performance per classes
    # viz_perf_classes(root_dir,df)

    # get table performance per length
    # viz_perf_length(root_dir,df)

    # get plot
    viz_plot(root_dir, df)

def viz_cam(root_dir):
    import tensorflow.keras as keras
    import sklearn
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'

    if dataset_name == 'Gun_Point':
        save_name = 'GunPoint'
    else:
        save_name = dataset_name
    max_length = 2000
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    y_test = datasets_dict[dataset_name][3]

    # transform to binary labels
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    w_k_c = model.layers[-1].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    for c in classes:
        plt.figure()
        count = 0
        c_x_train = x_train[np.where(y_train == c)]
        for ts in c_x_train:
            ts = ts.reshape(1, -1, 1)
            [conv_out, predicted] = new_feed_forward([ts])
            pred_label = np.argmax(predicted)
            orig_label = np.argmax(enc.transform([[c]]))
            if pred_label == orig_label:
                cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
                for k, w in enumerate(w_k_c[:, orig_label]):
                    cas += w * conv_out[0, :, k]

                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100

                x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                # linear interpolation to smooth
                f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                y = f(x)
                # if (y < -2.2).any():
                #     continue
                f = interp1d(range(ts.shape[1]), cas)
                cas = f(x).astype(int)
                plt.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=2, vmin=0, vmax=100, linewidths=0.0)
                if dataset_name == 'Gun_Point':
                    if c == 1:
                        plt.yticks([-1.0, 0.0, 1.0, 2.0])
                    else:
                        plt.yticks([-2, -1.0, 0.0, 1.0, 2.0])
                count += 1

        cbar = plt.colorbar()
        # cbar.ax.set_yticklabels([100,75,50,25,0])
        plt.savefig(root_dir + '/temp/' + classifier + '-cam-' + save_name + '-class-' + str(int(c)) + '.png',
                    bbox_inches='tight', dpi=1080)

def visualizeEEG_in_time_and_frequency_domain(subject, amountOfSamples, predictions, dataset):

    #GETTING THE LIST OF ALL THE SAMPLE INDEXES THAT WAS PREDICTED INCORRECTLY FOR THAT SUBJECT
    subjectData              = predictions[subject]
    incorrectPredictions     = subjectData['incorrect predictions']

    #GETTING THE INDEXES OF THE SAMPLES THAT WE WANT TO VISUALIZE
    incorrectPredictionsLess = list(incorrectPredictions.items())[:amountOfSamples]


    for sampleInfo in incorrectPredictionsLess:

        #GETTING THE SAMPLE FROM THE DATASET
        sampleNum = sampleInfo[0]
        sample    = dataset[sampleNum, 0, :]

        #PLOTTING THE SAMPLE IN BOTH THE TIME AND FREQUENCY DOMAIN
        matplotlib.use("TkAgg")
        fig, ax = plt.subplots(2)
        ax[0].plot(sample)
        ax[1].magnitude_spectrum(sample, Fs =256)
        plt.show()

    pass


