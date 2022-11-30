"""
@authors: Steven Cao: https://github.com/stevencao2005,
"""


#import torch
import os
#from torch.utils.data import Dataset
import mne
import time
import pandas as pd
import numpy as np
from tensorpac import Pac
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import mne

from scipy.integrate import simps
from scipy import signal
import scipy.signal as sig
from scipy.linalg import sqrtm,logm

from tensorpac import Pac
import math
from time import process_time
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn_extra.cluster import KMedoids
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mne
import glob, os
from sklearn.utils import shuffle

'''
Script for producing training and testing data.
'''

class Preprocessing_For_DL_Models():
    def __init__(self):
        pass

    def create_directory(self,directory_path):
        if os.path.exists(directory_path):
            return None
        else:
            try:
                os.makedirs(directory_path)
            except:
                # in case another machine created the path meanwhile !:(
                return None
            return directory_path
    def tbi_format_to_numOfSamples_and_Signals(self, Tbi_filtered, Tbi_human):

        channel = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

        tbi_subjects               = [name for name in Tbi_human]
        tbi_subjects_channels_name = list(Tbi_filtered.keys())

        Tbi_filtered_reshaped_by_channels = Tbi_filtered.copy()
        for subject in tbi_subjects_channels_name:
            subject_channel = Tbi_filtered[subject]
            subject_channel_reshaped = np.reshape(subject_channel,
                                                  (len(channel), subject_channel.shape[1] * subject_channel.shape[2]))
            Tbi_filtered_reshaped_by_channels[subject] = subject_channel_reshaped

        # --create directory_path
        for name in tbi_subjects:
            directory_path = os.path.abspath('.') + '/data_preprocessed/' + name
            self.create_directory(directory_path)

        for name in tbi_subjects:
            subject_data = Tbi_filtered_reshaped_by_channels[name]
            file_name = os.path.abspath('.') + '/data_preprocessed/' + name + '/preprocessedSubjectData.npy'
            np.save(file_name, subject_data)

    def MICE_tbi_format_to_Signals(self, Tbi_filtered, Tbi_mice, p_mice):


        tbi_subjects               = [name for name in Tbi_mice]
        tbi_subjects_channels_name = list(Tbi_filtered.keys())

        Tbi_filtered_reshaped_by_channels = Tbi_filtered.copy()
        for subject in tbi_subjects_channels_name:
            subject_channel = Tbi_filtered[subject]
            subject_channel_reshaped = np.reshape(subject_channel,
                                                  (1, subject_channel.shape[0] * subject_channel.shape[1]))
            Tbi_filtered_reshaped_by_channels[subject] = subject_channel_reshaped

        # --create directory_path
        for name in tbi_subjects:
            directory_path = os.path.abspath('.') + '/data_preprocessed_mice/' + name
            self.create_directory(directory_path)

        for name in tbi_subjects:
            #--used only for euclidean alignment
            subject_data = Tbi_filtered_reshaped_by_channels[name]
            # subject_data = Tbi_filtered_reshaped_by_channels['tbi_'+ p_mice["sleep_stage"] + name]
            file_name = os.path.abspath('.') + '/data_preprocessed_mice/' + name + '/preprocessedSubjectData.npy'
            np.save(file_name, subject_data)

    def control_format_to_numOfSamples_and_Signals(self, Control_filtered, Control_human):

        channel = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

        control_subjects = [name for name in Control_human]
        control_subjects_channels_name = list(Control_filtered.keys())

        Control_filtered_reshaped_by_channels = Control_filtered.copy()
        for subject in control_subjects_channels_name:
            subject_channel = Control_filtered[subject]
            subject_channel_reshaped = np.reshape(subject_channel,
                                                  (len(channel), subject_channel.shape[1] * subject_channel.shape[2]))
            Control_filtered_reshaped_by_channels[subject] = subject_channel_reshaped

        # --create directory_path
        for name in control_subjects:
            directory_path = os.path.abspath('.') + '/data_preprocessed/' + name
            self.create_directory(directory_path)

        for name in control_subjects:
            subject_data = Control_filtered_reshaped_by_channels[name]
            file_name = os.path.abspath('.') + '/data_preprocessed/' + name + '/preprocessedSubjectData.npy'
            np.save(file_name, subject_data)

    def MICE_control_format_to_Signals(self, Control_filtered, Control_mice, p_mice):

        control_subjects = [name for name in Control_mice]
        control_subjects_channels_name = list(Control_filtered.keys())

        Control_filtered_reshaped_by_channels = Control_filtered.copy()
        for subject in control_subjects_channels_name:
            subject_channel = Control_filtered[subject]
            subject_channel_reshaped = np.reshape(subject_channel,
                                                  (1, subject_channel.shape[0] * subject_channel.shape[1]))
            Control_filtered_reshaped_by_channels[subject] = subject_channel_reshaped

        # --create directory_path
        for name in control_subjects:
            directory_path = os.path.abspath('.') + '/data_preprocessed_mice/' + name
            self.create_directory(directory_path)

        for name in control_subjects:
            #--used only for euclidean alignment technique
            subject_data = Control_filtered_reshaped_by_channels[name]
            # subject_data = Control_filtered_reshaped_by_channels['control_'+ p_mice["sleep_stage"] + name]
            file_name = os.path.abspath('.') + '/data_preprocessed_mice/' + name + '/preprocessedSubjectData.npy'
            np.save(file_name, subject_data)



    def lowpass_filter(self, arr, sfreq, lowpass):
        '''
        Performs low-pass filtering.
        '''
        return mne.filter.filter_data(arr, sfreq=sfreq, l_freq=None, h_freq=lowpass)

    def add_epochs(self, X_lst, y_lst, X, y, sfreq, epoch_len, overlap):
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

    def train_test_split_old(self,
                         sfreq=200,
                         lowpass=50,
                         epoch_len=4,
                         overlap=.9,
                         parent_folder=None,
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
            subject = d.split(os.sep)[-1][1:]
            for session_dir in glob.glob(d + os.sep + '*.npy'):
                X = self.lowpass_filter(np.load(session_dir), sfreq, lowpass)
                X = shuffle(X)
                if subject in Control_human:
                    y = 'control'
                    X_train_data = X[:, :int(X.shape[1] * 0.8)]
                    self.add_epochs(X_train, y_train, X_train_data, y, sfreq, epoch_len, overlap)
                    X_test_data = X[:, int(X.shape[1] * 0.8):]
                    self.add_epochs(X_test, y_test, X_test_data, y, sfreq, epoch_len, overlap)
                else:
                    y = 'tbi'
                    X_train_data = X[:, :int(X.shape[1] * 0.8)]
                    self.add_epochs(X_train, y_train, X_train_data, y, sfreq, epoch_len, overlap)
                    X_test_data = X[:, int(X.shape[1] * 0.8):]
                    self.add_epochs(X_test, y_test, X_test_data, y, sfreq, epoch_len, overlap)
        if data_folder == None:
            data_folder = '.' + os.sep
        else:
            data_folder = os.path.abspath(data_folder) + os.sep
        np.save(data_folder + 'X_train.npy', X_train)
        np.save(data_folder + 'X_test.npy', X_test)
        np.save(data_folder + 'y_train.npy', y_train)
        np.save(data_folder + 'y_test.npy', y_test)

        pass

    def train_test_split_save(self,
                         sfreq=200,
                         lowpass=50,
                         epoch_len=4,
                         overlap=.9,
                         IV_control = '102',
                         IV_tbi     = '244',
                         parent_folder=None,
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
                X = self.lowpass_filter(np.load(session_dir), sfreq, lowpass)
                X = shuffle(X)

                if subject in Control_human:
                    y = 'control'
                    if subject != IV_control:
                        self.add_epochs(X_train, y_train, X, y, sfreq, epoch_len, overlap)
                    else:
                        self.add_epochs(X_test, y_test, X, y, sfreq, epoch_len, overlap)
                else:
                    y = 'tbi'
                    if subject != IV_tbi:
                        self.add_epochs(X_train, y_train, X, y, sfreq, epoch_len, overlap)
                    else:
                        self.add_epochs(X_test, y_test, X, y, sfreq, epoch_len, overlap)
        if data_folder == None:
            data_folder = '.' + os.sep
        else:
            data_folder = os.path.abspath(data_folder) + os.sep
        np.save(data_folder + 'X_train.npy', X_train)
        np.save(data_folder + 'X_test.npy', X_test)
        np.save(data_folder + 'y_train.npy', y_train)
        np.save(data_folder + 'y_test.npy', y_test)

        pass
    def train_test_no_split(self,
                         sfreq=200,
                         lowpass=50,
                         epoch_len=4,
                         overlap=.9,
                         IV_control = '102',
                         IV_tbi     = '244',
                         parent_folder=None,
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
                X = self.lowpass_filter(np.load(session_dir), sfreq, lowpass)
                X = shuffle(X)

                if subject in Control_human:
                    y = 'control'
                    if subject != IV_control:
                        self.add_epochs(X_train, y_train, X, y, sfreq, epoch_len, overlap)
                    else:
                        self.add_epochs(X_test, y_test, X, y, sfreq, epoch_len, overlap)
                else:
                    y = 'tbi'
                    if subject != IV_tbi:
                        self.add_epochs(X_train, y_train, X, y, sfreq, epoch_len, overlap)
                    else:
                        self.add_epochs(X_test, y_test, X, y, sfreq, epoch_len, overlap)
        return X_train, X_test, y_train, y_test

    def form_into_two_datasets(self,
                         sfreq=200,
                         lowpass=50,
                         epoch_len=4,
                         overlap=.9,
                         parent_folder=None,
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
        X_dataset = []
        Y_dataset = []


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
                data = np.load(session_dir)
                X = self.lowpass_filter(data, sfreq, lowpass)
                X = shuffle(X)
                self.add_epochs(X_dataset, Y_dataset, X, subject, sfreq, epoch_len, overlap)
        if data_folder == None:
            data_folder = '.' + os.sep
        else:
            data_folder = os.path.abspath(data_folder) + os.sep
        np.save(data_folder + 'X.npy', X_dataset)
        np.save(data_folder + 'Y.npy', Y_dataset)

        i=1
        pass


    def MICE_form_into_two_datasets(self,
                         sfreq=200,
                         lowpass=50,
                         epoch_len=4,
                         overlap=.9,
                         parent_folder=None,
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
        X_dataset = []
        Y_dataset = []


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
                data = np.load(session_dir)
                X = self.lowpass_filter(data, sfreq, lowpass)
                X = shuffle(X)
                self.add_epochs(X_dataset, Y_dataset, X, subject, sfreq, epoch_len, overlap)
        if data_folder == None:
            data_folder = '.' + os.sep
        else:
            data_folder = os.path.abspath(data_folder) + os.sep
        np.save(data_folder + 'X_mice.npy', X_dataset)
        np.save(data_folder + 'Y_mice.npy', Y_dataset)

        pass


