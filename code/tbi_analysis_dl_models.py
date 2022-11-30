
"""
@author: Steven Cao"""

#IMPORT ALL NEEDED MODULES

#Standard library imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys
import time
import utils

#Third party imports
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import tensorboard
import tensorflow as tf
from tensorflow import keras

#Local application imports
from inception import Classifier_INCEPTION
from utils import calculate_metrics
from utils import case_by_case_analysis
from utils import create_directory
from utils import one_hot_encoder
from utils import gettingInfoDL
from utils import generate_results_csv
from utils import read_all_datasets
from utils import read_dataset
from utils import standardizing_the_dataset
from utils import transform_mts_to_ucr_format
from utils import visualizeEEG_in_time_and_frequency_domain
from utils import visualize_filter
from utils import viz_cam
from utils import viz_for_survey_paper


# --- preprocessing the features -----
file_name = os.path.abspath(".") + '/tbi_preprocessed_features_dataset.pkl'
start_time = time.time()
Total_dataframe = pickle.load(open(file_name, "rb"))
print(time.time() - start_time)

dataFile = os.path.abspath(".") + '/human_eeg_data.pkl'
time1 = time.time()
with open(dataFile, 'rb') as file:
    data = pickle.load(file)
print(time.time() - time1)
Control_human = data['Control_human']
Tbi_human = data['Tbi_human']

# ----- getting the preprocessed raw dataset
import pickle
file_name = os.path.abspath(".") + '/tbi_preprocessed_control_dataset.pkl'
start_time = time.time()
Control_filtered = pickle.load(open(file_name, "rb"))
print(time.time() - start_time)

file_name = os.path.abspath(".") + '/tbi_preprocessed_tbi_dataset.pkl'
start_time = time.time()
Tbi_filtered = pickle.load(open(file_name, "rb"))
print(time.time() - start_time)

channel = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

tbi_subjects = ['control_W' + name for name in Tbi_human]
tbi_subjects_channels_name = list(Tbi_filtered.keys())

Tbi_filtered_reshaped_by_channels = Tbi_filtered.copy()
for subject in tbi_subjects_channels_name:
    subject_channel = Tbi_filtered[subject]
    subject_channel_reshaped = np.reshape(subject_channel,
                                          (subject_channel.shape[0] * subject_channel.shape[1]))
    Tbi_filtered_reshaped_by_channels[subject] = np.expand_dims(subject_channel_reshaped, axis=0)

Tbi_filtered_reshaped_by_subjects = dict()
for subject in tbi_subjects:
    Tbi_filtered_reshaped_by_subjects[subject] = np.zeros(
        (6, subject_channel.shape[0] * subject_channel.shape[1]))
for subject in tbi_subjects_channels_name:
    subject_channel = Tbi_filtered_reshaped_by_channels[subject]
    for name in tbi_subjects:
        if name in subject:
            Tbi_filtered_reshaped_by_subjects[name][int(channel.index(subject[-2:]))] = subject_channel

# --create directory_path
for name in tbi_subjects:
    directory_path = os.path.abspath('.') + '/data_preprocessed/' + name[8:]
    create_directory(directory_path)

for name in tbi_subjects:
    subject_data = Tbi_filtered_reshaped_by_subjects[name]
    file_name = os.path.abspath('.') + '/data_preprocessed/' + name[8:] + '/preprocessedSubjectData.npy'
    np.save(file_name, subject_data)

channel = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

control_subjects = ['control_W' + name for name in Control_human]
control_subjects_channels_name = list(Control_filtered.keys())

Control_filtered_reshaped_by_channels = Control_filtered.copy()
for subject in control_subjects_channels_name:
    subject_channel = Control_filtered[subject]
    subject_channel_reshaped = np.reshape(subject_channel,
                                          (subject_channel.shape[0] * subject_channel.shape[1]))
    Control_filtered_reshaped_by_channels[subject] = np.expand_dims(subject_channel_reshaped, axis=0)

Control_filtered_reshaped_by_subjects = dict()
for subject in control_subjects:
    Control_filtered_reshaped_by_subjects[subject] = np.zeros(
        (6, subject_channel.shape[0] * subject_channel.shape[1]))
for subject in control_subjects_channels_name:
    subject_channel = Control_filtered_reshaped_by_channels[subject]
    for name in control_subjects:
        if name in subject:
            Control_filtered_reshaped_by_subjects[name][int(channel.index(subject[-2:]))] = subject_channel

# --create directory_path
for name in control_subjects:
    directory_path = os.path.abspath('.') + '/data_preprocessed/' + name[8:]
    create_directory(directory_path)

for name in control_subjects:
    subject_data = Control_filtered_reshaped_by_subjects[name]
    file_name = os.path.abspath('.') + '/data_preprocessed/' + name[8:] + '/preprocessedSubjectData.npy'
    np.save(file_name, subject_data)

