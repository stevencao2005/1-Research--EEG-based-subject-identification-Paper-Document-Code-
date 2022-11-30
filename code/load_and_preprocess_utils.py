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


from mpl_toolkits.mplot3d import Axes3D


class Loading_human():
    def __init__(self):
        self.p = None
        self.Control = None
        self.Control_human = None
        self.Control_sleep_label = None

        self.Tbi = None
        self.Tbi_human = None
        self.Tbi_sleep_label = None
        pass

    class functions(object):
        def __init__(self):
            pass

        def extract_human_data2(self, parameters, class_label, dataset_label):

            no_data = []
            data = {}
            sleep_label = {}
            subjects = parameters[dataset_label + '_' + class_label + '_human']

            for l in subjects:

                # print subject name---------------------------------------------------
                print()
                print('Subject ', l)

                # load stage file------------------------------------------------------
                if dataset_label == 'dataset2':
                    a = pd.read_csv(parameters['data_folder'] + '\\' + l + '_Stage.txt', header=None, index_col=None)
                elif dataset_label == 'dataset1':
                    df = pd.read_csv(parameters['data_folder'] + '\\' + l + '_Stage.txt', header=None, index_col=None)
                    df = df.drop(df.index[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
                    x = df.loc[16]
                    a = x.str.split(expand=True)

                    for j in range(17, len(df) + 16):
                        x = df.loc[j]
                        b = x.str.split(expand=True)
                        a = a.append(b)

                a.columns = a.iloc[0]
                a = a[a.iloc[:, 20] != 'Stg']

                # load data file------------------------------------------------------
                if parameters['ica'] == 'Y':
                    raw = mne.io.read_raw_fif(parameters['data_folder'] + '\\' + l + "_ica.fif", preload=True,
                                              verbose=None)
                elif parameters['ica'] == 'N':
                    raw = mne.io.read_raw_edf(parameters['data_folder'] + '\\' + l + ".edf", preload=True, verbose=None)

                # filter in time domain-----------------------------------------------
                if parameters['filter_extract'] == 'tim':
                    filt_raw = raw.copy()
                    filt_raw.load_data().filter(p['lf2'], p['hf2'])
                    raw = filt_raw

                # total data-----------------------------------------------------------
                raw_eeg = raw[:, :][0]
                print(np.shape(raw_eeg))

                temp = set(parameters['channel'])
                index = [i for i, val in enumerate(raw.ch_names) if val in temp]
                raw_data = (raw_eeg[index, :])

                count = 0
                j = 0
                data_control = np.zeros((len(parameters['channel']), 1))

                b1 = []

                for i in range(len(a)):
                    if a.iloc[i, 20] in parameters['sleep_stage']:
                        b1.append(a.iloc[i, 20])
                        # print(i+1)
                        count = count + 1
                        if dataset_label == 'dataset2':
                            data_control = np.concatenate((data_control, (raw_data[:,
                                                                          i * parameters[dataset_label + '_fsh'] *
                                                                          parameters[
                                                                              'epoch_len']:i * parameters[
                                                                              dataset_label + '_fsh'] * parameters[
                                                                                               'epoch_len'] + (
                                                                                                       parameters[
                                                                                                           dataset_label + '_fsh'] *
                                                                                                       parameters[
                                                                                                           'epoch_len'])])),
                                                          axis=1)
                        elif dataset_label == 'dataset1':
                            data_control = np.concatenate((data_control, (raw_data[:,
                                                                          i * parameters[dataset_label + '_fsh'] *
                                                                          parameters[
                                                                              'epoch_len']:i * parameters[
                                                                              dataset_label + '_fsh'] * parameters[
                                                                                               'epoch_len'] + (
                                                                                                       parameters[
                                                                                                           dataset_label + '_fsh'] *
                                                                                                       parameters[
                                                                                                           'epoch_len'])]) * 10 ** 6),
                                                          axis=1)
                            # print(j)
                        j = j + parameters['epoch_len'] * parameters[dataset_label + '_fsh']
                data_control = data_control[:, 1:]
                print(np.shape(data_control))
                if np.shape(data_control)[1] != 0:
                    data_control = data_control - data_control.mean(axis=1, keepdims=True)
                    name = l
                    data[name] = data_control
                    if class_label == 'Control':
                        b1 = [x + '0' for x in b1]
                        sleep_label[name] = b1
                    else:
                        b1 = [x + '1' for x in b1]
                        sleep_label[name] = b1

                else:
                    if l in no_data:
                        pass
                    else:
                        no_data.append(l)

            # print subjects with no sleep stage-------------------------------------
            if len(no_data) != 0:
                print('----------------------------------------------------------')
                print('No ' + parameters['sleep_stage'] + ' stage in ' + class_label + ' subjects ', no_data)

            # create new list of subjects who have particular sleep stage-------------
            ind = np.where(np.in1d(subjects, no_data))[0]
            new_list = [val for n, val in enumerate(subjects) if n not in ind]
            print('==========================================================')

            return data, sleep_label, new_list, no_data


    def load_parameters(self):
        p = dict()
        # -----------------------------------------------------------------------------
        # Dataset parameters
        p['data_folder'] = os.path.abspath('.') + '/data'

        p['data'] = ['dataset1', 'dataset2']

        p['dataset1_Control_human'] = ['102', '208', '457', '495', '556', '563', '744'] #7
        p['dataset1_Tbi_human'] = ['244', '340', '399', '424', '488', '510', '670'] #7
        p['dataset1_fsh'] = 200

        p['dataset2_Control_human'] = ['XVZ2FYATE8M0SSF',
                         'XVZ2FYAQH8YMGKY', 'XVZ2FYATE8X4YXQ', 'XVZ2FYATE84ZTFV', 'XVZ2FYATE8AJWX0', #9
                         'XVZ2FYATE8BBO87', 'XVZ2FFAG8875MNV', 'XVZ2FYATE8ZYTB2', 'XVZ2FYATE8YDANN']
        p['dataset2_Tbi_human'] = ['XVZ2FYAQH8WVIUC',
                     'XVZ2FYATE84MSWI', 'XVZ2FYATE8B9R6X', 'XVZ2FYATE8DFIYL', 'XVZ2FYATE8FN4DS', #12
                     'XVZ2FYATE8HSYB3', 'XVZ2FYATE8I41U0', 'XVZ2FYATE8JWW0A', 'XVZ2FYATE8K9U90',
                     'XVZ2FYATE8W7FI6', 'XVZ2FYATE8Z362L', 'XVZ2FFAG885GFUG']

        #Control: 16
        #Tbi: 19
        p['dataset2_fsh'] = 200

        p['age'] = {
            '102': [48],
            '208': [26],
            '457': [33],
            '495': [59],
            '399': [30],
            '556': [59],
            '563': [32],
            '603': [61],
            '744': [33],
            '153': [31],
            '244': [49],
            '340': [40],
            '424': [26],
            '488': [30],
            '510': [32],
            '670': [61],
            '440': [27],
            '610': [58],
            '101': [38],
            'XVZ2FYAQH8WM6TP': [39],
            'XVZ2FYAQH8WVIUC': [50],
            'XVZ2FYATE84MSWI': [34],
            'XVZ2FYATE8B9R6X': [33],
            'XVZ2FYATE8DFIYL': [32],
            'XVZ2FYATE8FA7E2': [59],
            'XVZ2FYATE8FN4DS': [30],
            'XVZ2FYATE8HSYB3': [23],
            'XVZ2FYATE8I41U0': [47],
            'XVZ2FYATE8IF50T': [31],
            'XVZ2FYATE8JWW0A': [43],
            'XVZ2FYATE8K9U90': [25],
            'XVZ2FYATE8W7FI6': [29],
            'XVZ2FYATE8YIMAH': [46],
            'XVZ2FYATE8Z362L': [40],
            'XVZ2FFAG885GFUG': [27],
            'XVZ2FFAG888P52H': [35],
            'XVZ2FFAG88ACGI4': [64],
            'XVZ2FYATE8M0SSF': [37],
            'XVZ2FYATE8X4YXQ': [55],
            'XVZ2FYATE8AALDJ': [33],
            'XVZ2FYATE875N3G': [36],
            'XVZ2FFAG889F317': [58],
            'XVZ2FYATE86FLYZ': [30],
            'XVZ2FYATE8B60OJ': [26],
            'XVZ2FYAQH8YMGKY': [31],
            'XVZ2FYATE84ZTFV': [43],
            'XVZ2FYATE8AJWX0': [72],
            'XVZ2FYATE8BBO87': [69],
            'XVZ2FYATE8TW5E7': [64],
            'XVZ2FYAQH8XLFTM': [29],
            'XVZ2FYAQH906525': [64],
            'XVZ2FYAQH90QFES': [69],
            'XVZ2FYATE8245MT': [64],
            'XVZ2FYATE83RBU3': [37],
            'XVZ2FYATE87IKI5': [69],
            'XVZ2FYATE8ALXGF': [26],
            'XVZ2FYATE8U1TQ5': [22],
            'XVZ2FYATE8E94H2': [32],
            'XVZ2FFAG886UF84': [84],
            'XVZ2FFAG8875MNV': [67],
            'XVZ2FYATE8ZYTB2': [76],
            'XVZ2FYATE8YDANN': [67],
            'XVZ2FYATE8XUPX7': [63]

        }

        # -----------------------------------------------------------------------------
        # Filter parameters
        p['freq_band'] = 'normal'
        # 'delta','theta','alpha','sigma','beta','gama','normal'

        p['filter_extract'] = 'freq'
        # 'tim','freq'

        # Filter using mne while extracting
        if p['filter_extract'] == 'tim':
            p['lf2'] = 0.5
            p['hf2'] = 50

        p['ica'] = 'Y'
        # 'Y','N'

        # -----------------------------------------------------------------------------
        # data parameters

        p['sleep_stage'] = ['W']
        # ['W','N1','N2','N3','R']

        p['channel'] = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
        # ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1', 'EOG-L', 'EOG-R', 'Chin', 'L Leg', 'R Leg', 'ECG', 'Snore', 'Airflow', 'P-Flo', 'C-FLOW', 'Chest', 'Abdomen', 'SpO2', 'C-Press', 'R-R', 'EtCO2']

        p['epoch_len'] = 30
        # sec

        p['duration'] = 10
        # value in min or 'NA'

        # -----------------------------------------------------------------------------
        # feature parameters
        p['features'] = ['absolute_power', 'relative_power', 'slow_fast', 'frequency amplitude asymmetry', 'phase synchrony',
                         'coherence', 'hjorth', 'spectral_entropy', 'phase amplitude coupling']

        p['max_features'] = 30
        self.p = p
        return p

    def load_data_old(self):

        #CAN ONLY BE USED OUTSIDE OF THE CLASS
        p       = self.p

        for i in p['data']:
            print('Loading {}...................'.format(i))
            vars()[i + '_Control'], vars()[i + '_Control_sleep_label'], vars()[i + '_Control_human'], vars()[
                i + '_no_data_Control'] = self.functions().extract_human_data2(p, 'Control', i)
            vars()[i + '_Tbi'], vars()[i + '_Tbi_sleep_label'], vars()[i + '_Tbi_human'], vars()[
                i + '_no_data_Tbi'] = self.functions().extract_human_data2(p, 'Tbi', i)

        Control_datasets = [x + '_Control' for x in p['data']]
        Control_dataset_sleep_label = [x + '_Control_sleep_label' for x in p['data']]
        Control_dataset_human = [x + '_Control_human' for x in p['data']]
        Control = {}
        Control_sleep_label = {}
        Control_human = []
        for i in range(len(Control_datasets)):
            Control = {**Control, **globals()[Control_datasets[i]]}
            Control_sleep_label = {**Control_sleep_label, **globals()[Control_dataset_sleep_label[i]]}
            Control_human.extend(globals()[Control_dataset_human[i]])

        Control_dict = {'Control': Control, 'Control_dataset_human': Control_dataset_human,
                        'Control_dataset_sleep_label': Control_dataset_sleep_label, 'Control_datasets':Control_datasets,
                        'Control_human': Control_human, 'Control_sleep_label':Control_sleep_label}

        Tbi_datasets = [x + '_Tbi' for x in p['data']]
        Tbi_dataset_sleep_label = [x + '_Tbi_sleep_label' for x in p['data']]
        Tbi_dataset_human = [x + '_Tbi_human' for x in p['data']]
        Tbi = {}
        Tbi_sleep_label = {}
        Tbi_human = []
        for i in range(len(Tbi_datasets)):
            Tbi = {**Tbi, **globals()[Tbi_datasets[i]]}
            Tbi_sleep_label = {**Tbi_sleep_label, **globals()[Tbi_dataset_sleep_label[i]]}
            Tbi_human.extend(globals()[Tbi_dataset_human[i]])

        Tbi_dict = {'Tbi': Tbi, 'Tbi_dataset_human': Tbi_dataset_human,
                        'Tbi_dataset_sleep_label': Tbi_dataset_sleep_label, 'Tbi_datasets':Tbi_datasets,
                        'Tbi_human': Tbi_human, 'Tbi_sleep_label':Tbi_sleep_label}

        return Control_dict, Tbi_dict


        pass

    def load_data(self):

        p       = self.p
        classes = ['Control', 'Tbi']

        import time, os
        import pickle
        dataFile = os.path.abspath('.') + '/human_eeg_data_02_15_2022.pkl'
        time1 = time.time()
        if os.path.exists(dataFile):
            with open(dataFile, 'rb') as file:
                data_dict = pickle.load(file)
        else:
            for numOfClass, i in enumerate(p['data']):
                print('Loading {}...................'.format(i))
                oneClass = classes[numOfClass]
                if i == 'dataset1':
                    dataset1_Control, dataset1_Control_sleep_label,\
                    dataset1_Control_human, dataset1_no_data_Control = self.functions().extract_human_data2(p, 'Control', i)
                    dataset1_Tbi, dataset1_Tbi_sleep_label,\
                    dataset1_Tbi_human, dataset1_no_data_Tbi = self.functions().extract_human_data2(p, 'Tbi', i)
                elif i == 'dataset2':
                    dataset2_Control, dataset2_Control_sleep_label,\
                    dataset2_Control_human, dataset2_no_data_Control = self.functions().extract_human_data2(p, 'Control', i)
                    dataset2_Tbi, dataset2_Tbi_sleep_label,\
                    dataset2_Tbi_human, dataset2_no_data_Tbi = self.functions().extract_human_data2(p, 'Tbi', i)

            dataset1_Control.update(dataset2_Control)
            dataset1_Control_human.extend(dataset2_Control_human)
            dataset1_Control_sleep_label.update(dataset2_Control_sleep_label)
            dataset1_no_data_Control.extend(dataset2_no_data_Control)

            Control = dataset1_Control
            Control_human = dataset1_Control_human
            Control_sleep_label = dataset1_Control_sleep_label


            self.Control = Control
            self.Control_human = Control_human
            self.Control_sleep_label = Control_sleep_label


            dataset1_Tbi.update(dataset2_Tbi)
            dataset1_Tbi_human.extend(dataset2_Tbi_human)
            dataset1_Tbi_sleep_label.update(dataset2_Tbi_sleep_label)
            dataset1_no_data_Tbi.extend(dataset2_no_data_Tbi)

            Tbi = dataset1_Tbi
            Tbi_human = dataset1_Tbi_human
            Tbi_sleep_label = dataset1_Tbi_sleep_label


            self.Tbi = Tbi
            self.Tbi_human = Tbi_human
            self.Tbi_sleep_label = Tbi_sleep_label


            Control_dict = {'Control': Control, 'Control_sleep_label': Control_sleep_label, 'Control_human': Control_human}
            Tbi_dict     = {'Tbi': Tbi, 'Tbi_sleep_label': Tbi_sleep_label, 'Tbi_human': Tbi_human}

            data_dict = {'Control_dict': Control_dict, 'Tbi_dict': Tbi_dict}
            with open(dataFile, 'wb') as file:
                pickle.dump(data_dict, file, pickle.HIGHEST_PROTOCOL)

        return data_dict

    def create_epoch(self, data, p, class_label):
        epoch_data = {}
        for ele in data:
            if ele in p['dataset1_' + class_label + '_human']:
                p['fs'] = p['dataset1_fsh']
            elif ele in p['dataset2_' + class_label + '_human']:
                p['fs'] = p['dataset2_fsh']

            num_epoch = int(np.shape(data[ele])[1] / (p['epoch_len'] * p['fs']))
            a = []

            for i in range(num_epoch):
                a.append(
                    data[ele][:, i * p['fs'] * p['epoch_len']:i * p['fs'] * p['epoch_len'] + p['fs'] * p['epoch_len']])
            epoch_data[ele] = np.transpose(a, (1, 2, 0))

        return epoch_data

    def artifact_human_removal3(self, data, p):
        AR_rem = {}
        m2 = {}
        for i in data:
            m2[i] = np.zeros(np.shape(data[i])[2], dtype=bool)
            for j in range(len(p['channel'])):
                a = (np.max(data[i][j], axis=0)) - (np.min(data[i][j], axis=0))
                z = ((a - np.mean(a)) / np.std(a))
                avglatlist = np.arange(1, a.shape[0] + 1)
                m = np.abs(z) > 2.5
                # mask.append(m)
                m2[i] = np.logical_or(m2[i], m)

                # a=(np.var(data[i][j],axis=0))
                a = np.sum(np.abs(data[i][j]), axis=0)
                z = ((a - np.mean(a)) / np.std(a))
                m = np.abs(z) > 2.5
                # mask.append(m)
                m2[i] = np.logical_or(m2[i], m)

            AR_rem[i] = np.delete(data[i], avglatlist[m2[i]] - 1, 2)
            mini = 10 ** 10

            for ele in AR_rem:
                if np.shape(AR_rem[ele])[2] < mini:
                    mini = np.shape(AR_rem[ele])[2]
        return AR_rem, mini, m2

    def label_sel_old(self, data, mask):
        for i in data:
            data[i] = (np.delete(data[i], mask[i]))
        return data

    def label_sel(self, data, mask):
        for i in data:
            print(i)
            data[i] = (np.delete(data[i][0:len(mask[i])], mask[i]))
        return data

    def length_sel_old(self, data, label, p):
        for ele in data:
            if p['duration'] != 'NA':
                data[ele] = data[ele][:, :, 0:p['duration'] * int(60 / p['epoch_len'])]
                label[ele] = label[ele][0:p['duration'] * int(60 / p['epoch_len'])]
            elif p['duration'] == 'NA':
                data[ele] = data[ele]
                label[ele] = label[ele]

        return data, label

    def length_sel(self, data, label, p):
        for ele in data:
            if p['duration'] != 'NA':
                data[ele] = data[ele][:, :, 0:p['duration'] * int(60 / p['epoch_len'])]
                label[ele] = label[ele][0:p['duration'] * int(60 / p['epoch_len'])]
            elif p['duration'] == 'NA':
                data[ele] = data[ele]
                label[ele] = label[ele]

        return data, label

    def filt(self, sig, p, lf1, lf2, hf2, hf1):
        x = sig - np.mean(sig)
        no_samp = len(x)
        mfft = round(np.log(len(x)) / np.log(2) + 0.5) + 1
        nfft = int(2 ** mfft)

        xf = np.fft.fft(x, int(nfft))
        xf2 = np.zeros((nfft), dtype='complex_')
        lp2 = int(np.fix((hf1 / p['fs']) * nfft))
        lp22 = int(np.fix((hf2 / p['fs']) * nfft))
        wl1 = np.ones(lp2)
        wl1[lp22:lp2] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(lp2 - lp22) + 1) / (lp2 - lp22))

        wl2 = wl1[::-1]
        xf2[0:lp2] = np.multiply(xf[0:lp2], np.transpose(wl1))
        xf2[nfft - lp2:nfft] = np.multiply(xf[nfft - lp2:nfft], np.transpose(wl2))

        qf1 = lf1
        lq2 = int(np.fix((qf1 / p['fs']) * nfft + 1))
        qf2 = lf2
        lq22 = int(np.fix((qf2 / p['fs']) * nfft + 1))  # fix(0.5*lp2);
        ql1 = np.zeros(lq22)
        ql1[lq2:lq22] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(lq22 - lq2) + 1) / (lq22 - lq2))
        ql2 = ql1[::-1]
        xf2[0:lq22] = np.multiply(xf2[0:lq22], ql1)
        xf2[nfft - lq22:nfft] = np.multiply(xf2[nfft - lq22:nfft], ql2)

        xi = np.real(np.fft.ifft(xf2, nfft))
        xi = xi[0:no_samp]

        return xi

    def freq_filt(self, unfiltered_data, p, class_label):
        filtered_data = {}

        if p['freq_band'] == 'delta':
            lf1 = 0.5
            lf2 = 1
            hf2 = 4
            hf1 = 4.5
        elif p['freq_band'] == 'theta':
            lf1 = 3.5
            lf2 = 4
            hf2 = 8
            hf1 = 8.5
        elif p['freq_band'] == 'alpha':
            lf1 = 7.5
            lf2 = 8
            hf2 = 12
            hf1 = 12.5
        elif p['freq_band'] == 'sigma':
            lf1 = 12.5
            lf2 = 13
            hf2 = 16
            hf1 = 16.5
        elif p['freq_band'] == 'beta':
            lf1 = 16.5
            lf2 = 17
            hf2 = 25
            hf1 = 25.5
        elif p['freq_band'] == 'gama':
            lf1 = 29.5
            lf2 = 30
            hf2 = 40
            hf1 = 40.5
        elif p['freq_band'] == 'slow':
            lf1 = 0.5
            lf2 = 1
            hf2 = 6
            hf1 = 6.5
        elif p['freq_band'] == 'normal':
            lf1 = 0.5
            lf2 = 1
            hf2 = 50
            hf1 = 50.5

        for i in unfiltered_data:
            if i in p['dataset1_' + class_label + '_human']:
                p['fs'] = p['dataset1_fsh']
            elif i in p['dataset2_' + class_label + '_human']:
                p['fs'] = p['dataset2_fsh']

            filtered_data[i] = np.zeros(np.shape(unfiltered_data[i]))

            for j in range(np.shape(unfiltered_data[i])[0]):
                for k in range(np.shape(unfiltered_data[i])[2]):
                    filtered_data[i][j, :, k] = self.filt(unfiltered_data[i][j, :, k], p, lf1, lf2, hf2, hf1)

        return filtered_data

    def feature_human_spectral(self, filtered, p, class_label):
        # https://raphaelvallat.com/bandpower.html

        if 'absolute_power' in p['features'] or 'relative_power' in p['features'] or 'spectral_entropy' in p[
            'features']:

            delta_low, delta_high = 0.5, 4
            theta_low, theta_high = 4, 8
            alpha_low, alpha_high = 8, 12
            sigma_low, sigma_high = 13, 16
            beta_low, beta_high = 16, 25
            gama_low, gama_high = 30, 40
            total_low, total_high = 0.5, 50

            # q={}
            band_power = {}
            rel_power = {}
            spec_ent = {}
            slow_fast = {}
            for ele in filtered:
                if ele in p['dataset1_' + class_label + '_human']:
                    p['fs'] = p['dataset1_fsh']
                elif ele in p['dataset2_' + class_label + '_human']:
                    p['fs'] = p['dataset2_fsh']
                win = int(2 / 0.5) * p['fs']

                # q[ele]=[[] for _ in range(2)]
                band_power[ele] = [[] for _ in range(8)]
                rel_power[ele] = [[] for _ in range(6)]
                spec_ent[ele] = [[] for _ in range(6)]
                slow_fast[ele] = [[] for _ in range(3)]
                for i in range(np.shape(filtered[ele])[2]):
                    a, b, c, d, e, f, g, h, l, m = np.zeros((3, np.shape(filtered[ele])[0])), np.zeros(
                        (4, np.shape(filtered[ele])[0])), np.zeros((4, np.shape(filtered[ele])[0])), np.zeros(
                        (4, np.shape(filtered[ele])[0])), np.zeros((1, np.shape(filtered[ele])[0])), np.zeros(
                        (1, np.shape(filtered[ele])[0])), np.zeros((4, np.shape(filtered[ele])[0])), np.zeros(
                        (4, np.shape(filtered[ele])[0])), np.zeros((4, np.shape(filtered[ele])[0])), np.zeros(
                        (3, np.shape(filtered[ele])[0]))
                    for j in range(np.shape(filtered[ele])[0]):
                        freqs, psd = signal.welch(filtered[ele][j, :, i], p['fs'], nperseg=win)
                        norm_psd = psd / np.sum(psd)
                        freq_res = freqs[1] - freqs[0]

                        idx_total = np.logical_and(freqs >= total_low, freqs <= total_high)
                        a[0, j] = simps(psd[idx_total], dx=freq_res)
                        a[1, j] = simps(norm_psd[idx_total], dx=freq_res)
                        a[2, j] = - np.sum(a[1, j] * np.log2(a[1, j]))

                        idx_delta = np.logical_and(freqs >= delta_low, freqs <= delta_high)
                        b[0, j] = simps(psd[idx_delta], dx=freq_res)
                        b[1, j] = simps(norm_psd[idx_delta], dx=freq_res)
                        b[2, j] = - np.sum(b[1, j] * np.log2(b[1, j]))
                        b[3, j] = b[0, j] / a[0, j]

                        idx_theta = np.logical_and(freqs >= theta_low, freqs <= theta_high)
                        c[0, j] = simps(psd[idx_theta], dx=freq_res)
                        c[1, j] = simps(norm_psd[idx_theta], dx=freq_res)
                        c[2, j] = - np.sum(c[1, j] * np.log2(c[1, j]))
                        c[3, j] = c[0, j] / a[0, j]

                        idx_alpha = np.logical_and(freqs >= alpha_low, freqs <= alpha_high)
                        d[0, j] = simps(psd[idx_alpha], dx=freq_res)
                        d[1, j] = simps(norm_psd[idx_alpha], dx=freq_res)
                        d[2, j] = - np.sum(d[1, j] * np.log2(d[1, j]))
                        d[3, j] = d[0, j] / a[0, j]

                        idx_alpha1 = np.logical_and(freqs >= 8, freqs <= 10)
                        e[0, j] = simps(psd[idx_alpha1], dx=freq_res)
                        # alpha1_rel_power[ele][i] = alpha1_power[ele][i] / total_power[ele][i]

                        idx_alpha2 = np.logical_and(freqs >= 10, freqs <= 12)
                        f[0, j] = simps(psd[idx_alpha2], dx=freq_res)
                        # alpha2_rel_power[ele][i] = alpha2_power[ele][i] / total_power[ele][i]

                        idx_sigma = np.logical_and(freqs >= sigma_low, freqs <= sigma_high)
                        g[0, j] = simps(psd[idx_sigma], dx=freq_res)
                        g[1, j] = simps(norm_psd[idx_sigma], dx=freq_res)
                        g[2, j] = - np.sum(g[1, j] * np.log2(g[1, j]))
                        g[3, j] = g[0, j] / a[0, j]

                        idx_beta = np.logical_and(freqs >= beta_low, freqs <= beta_high)
                        h[0, j] = simps(psd[idx_beta], dx=freq_res)
                        h[1, j] = simps(norm_psd[idx_beta], dx=freq_res)
                        h[2, j] = - np.sum(h[1, j] * np.log2(h[1, j]))
                        h[3, j] = h[0, j] / a[0, j]

                        idx_gama = np.logical_and(freqs >= gama_low, freqs <= gama_high)
                        l[0, j] = simps(psd[idx_gama], dx=freq_res)
                        l[1, j] = simps(norm_psd[idx_gama], dx=freq_res)
                        l[2, j] = - np.sum(l[1, j] * np.log2(l[1, j]))
                        l[3, j] = l[0, j] / a[0, j]

                        m[0, j] = c[0, j] / e[0, j]
                        m[1, j] = c[0, j] / f[0, j]
                        m[2, j] = e[0, j] / f[0, j]

                    # q[ele][0].append(a[0,:])
                    # q[ele][1].append(a[2,:])

                    band_power[ele][0].append(b[0, :])
                    band_power[ele][1].append(c[0, :])
                    band_power[ele][2].append(d[0, :])
                    band_power[ele][3].append(g[0, :])
                    band_power[ele][4].append(h[0, :])
                    band_power[ele][5].append(l[0, :])
                    band_power[ele][6].append(e[0, :])
                    band_power[ele][7].append(f[0, :])

                    rel_power[ele][0].append(b[3, :])
                    rel_power[ele][1].append(c[3, :])
                    rel_power[ele][2].append(d[3, :])
                    rel_power[ele][3].append(g[3, :])
                    rel_power[ele][4].append(h[3, :])
                    rel_power[ele][5].append(l[3, :])

                    spec_ent[ele][0].append(b[2, :])
                    spec_ent[ele][1].append(c[2, :])
                    spec_ent[ele][2].append(d[2, :])
                    spec_ent[ele][3].append(g[2, :])
                    spec_ent[ele][4].append(h[2, :])
                    spec_ent[ele][5].append(l[2, :])

                    slow_fast[ele][0].append(m[0, :])
                    slow_fast[ele][1].append(m[1, :])
                    slow_fast[ele][2].append(m[2, :])

                band_power[ele] = np.transpose(band_power[ele], (0, 2, 1))
                rel_power[ele] = np.transpose(rel_power[ele], (0, 2, 1))
                spec_ent[ele] = np.transpose(spec_ent[ele], (0, 2, 1))
                slow_fast[ele] = np.transpose(slow_fast[ele], (0, 2, 1))

        # =========================================================================================================

        if 'frequency amplitude asymmetry' in p['features']:
            n = {}

            for ele in filtered:
                z = 0
                n[ele] = np.zeros((6, int(np.shape(band_power[ele][2])[0] * (np.shape(band_power[ele][2])[0] - 1) / 2),
                                   np.shape(band_power[ele][2])[1]))
                for i in range(np.shape(band_power[ele][2])[0]):
                    for j in range(np.shape(band_power[ele][2])[0]):
                        if j > i:
                            for k in range(len(p['channel'])):
                                n[ele][k, z, :] = (band_power[ele][k][i] - band_power[ele][k][j]) / (
                                        band_power[ele][k][i] + band_power[ele][k][j])

                            z = z + 1

        # =========================================================================================================

        if 'phase amplitude coupling' in p['features']:
            r = {}

            for ele in filtered:

                r[ele] = np.zeros((2, np.shape(filtered[ele])[0], np.shape(filtered[ele])[2]))
                q1 = Pac(idpac=(2, 0, 0), f_pha=[4, 8], f_amp=[30, 40], dcomplex='hilbert', n_bins=18, verbose=False)

                for i in range((np.shape(filtered[ele])[2])):
                    for j in range((np.shape(filtered[ele])[0])):
                        r[ele][0, j, i] = q1.filterfit(p['fs'], filtered[ele][j, :, i])

                q1 = Pac(idpac=(2, 0, 0), f_pha=[4, 8], f_amp=[8, 12], dcomplex='hilbert', n_bins=18, verbose=False)

                for i in range((np.shape(filtered[ele])[2])):
                    for j in range((np.shape(filtered[ele])[0])):
                        r[ele][1, j, i] = q1.filterfit(p['fs'], filtered[ele][j, :, i])

        # =========================================================================================================

        return band_power, rel_power, slow_fast, spec_ent, n, r

        # =========================================================================================================
        # =========================================================================================================

    def feature_human_connectivity(self, filtered, p, class_label):
        win = int(2 / 0.5) * p['fs']
        delta_low, delta_high = 0.5, 4
        theta_low, theta_high = 4, 8
        alpha_low, alpha_high = 8, 12
        sigma_low, sigma_high = 13, 16
        beta_low, beta_high = 16, 25
        gama_low, gama_high = 30, 40
        total_low, total_high = 0.5, 50

        if 'coherence' in p['features']:
            m = {}
            for ele in filtered:
                z = 0
                m[ele] = np.zeros(
                    (6, int(np.shape(filtered[ele])[0] * (np.shape(filtered[ele])[0] - 1) / 2),
                     np.shape(filtered[ele])[2]))
                for a in range(len(p['channel'])):
                    for b in range(len(p['channel'])):
                        if a > b:
                            for i in range(np.shape(filtered[ele])[2]):
                                f, Cxy = signal.coherence(filtered[ele][a, :, i], filtered[ele][b, :, i], p['fs'],
                                                          'hann',
                                                          win)
                                idx_delta = np.logical_and(f >= delta_low, f <= delta_high)
                                m[ele][0, z, i] = np.mean(Cxy[idx_delta])

                                idx_theta = np.logical_and(f >= theta_low, f <= theta_high)
                                m[ele][1, z, i] = np.mean(Cxy[idx_theta])

                                idx_alpha = np.logical_and(f >= alpha_low, f <= alpha_high)
                                m[ele][2, z, i] = np.mean(Cxy[idx_alpha])

                                idx_sigma = np.logical_and(f >= sigma_low, f <= sigma_high)
                                m[ele][3, z, i] = np.mean(Cxy[idx_sigma])

                                idx_beta = np.logical_and(f >= beta_low, f <= beta_high)
                                m[ele][4, z, i] = np.mean(Cxy[idx_beta])

                                idx_gama = np.logical_and(f >= gama_low, f <= gama_high)
                                m[ele][5, z, i] = np.mean(Cxy[idx_gama])
                            z = z + 1

        # =========================================================================================================

        if 'phase synchrony' in p['features']:
            def hilphase2(y1, y2, n_sample):
                sig1_hill = signal.hilbert(y1)
                sig2_hill = signal.hilbert(y2)
                phase_y1 = np.unwrap(np.angle(sig1_hill))
                phase_y2 = np.unwrap(np.angle(sig2_hill))
                Inst_phase_diff = phase_y1 - phase_y2
                avg_phase = np.average(Inst_phase_diff)

                perc10w = math.floor(n_sample / 10)
                phase_y1 = phase_y1[perc10w:-perc10w]
                phase_y2 = phase_y2[perc10w:-perc10w]

                plv = np.abs(np.sum(np.exp(1j * (phase_y1 - phase_y2)))) / len(phase_y1)

                return plv, avg_phase

            # ----------------------------------------------------------------------------------------------------------
            def cal_plv(filtered_data, p):
                r = {}
                n_samples = p['epoch_len'] * p['fs']

                for ele in filtered_data:
                    z = 0
                    r[ele] = np.zeros(
                        (2, int(np.shape(filtered_data[ele])[0] * (np.shape(filtered_data[ele])[0] - 1) / 2),
                         np.shape(filtered_data[ele])[2]))
                    for a in range(len(p['channel'])):
                        for b in range(len(p['channel'])):
                            if a > b:
                                for i in range(np.shape(filtered_data[ele])[2]):
                                    r[ele][0, z, i], r[ele][1, z, i] = hilphase2(filtered_data[ele][a, :, i],
                                                                                 filtered_data[ele][b, :, i], n_samples)
                                    # r[ele][3,z,i]=np.corrcoef(filtered[ele][a,:,i],filtered[ele][b,:,i])[0,1]
                                z = z + 1
                return r

            # ----------------------------------------------------------------------------------------------------------
            filtered_data = {}
            phase_syn = [[] for _ in range(6)]
            freq_band = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'gama']
            for i in range(len(freq_band)):
                p['freq_band'] = freq_band[i]
                filtered_data[freq_band[i]] = self.freq_filt(filtered, p, class_label)
                phase_syn[i] = cal_plv(filtered_data[freq_band[i]], p)

        # =========================================================================================================

        return m, filtered_data, phase_syn, freq_band

        # =========================================================================================================
        # =========================================================================================================

    def feature_human_nonlin(self, filtered_data, p, freq_band):
        def Hjorth(extract):
            yt = extract

            diff_yt = np.diff(yt, axis=1)
            diff2_yt = np.diff(diff_yt, axis=1)
            Activity_Hjorth = np.var(yt, axis=1)
            var_D_yt = np.var(diff_yt, axis=1)
            var_2D_yt = np.var(diff2_yt, axis=1)
            Mobility_Hjorth = np.sqrt(var_D_yt / Activity_Hjorth)
            Mobility_Hjorth2 = np.sqrt(var_2D_yt / var_D_yt)

            Complexity_Hjorth = Mobility_Hjorth2 / Mobility_Hjorth

            return Activity_Hjorth, Mobility_Hjorth, Complexity_Hjorth

        if 'hjorth' in p['features']:
            Activity_Hjorth, Mobility_Hjorth, Complexity_Hjorth = {}, {}, {}

            for i in freq_band:
                Activity_Hjorth[i], Mobility_Hjorth[i], Complexity_Hjorth[i] = {}, {}, {}
                for j in filtered_data[i]:
                    Activity_Hjorth[i][j], Mobility_Hjorth[i][j], Complexity_Hjorth[i][j] = Hjorth(filtered_data[i][j])
        return Activity_Hjorth, Mobility_Hjorth, Complexity_Hjorth

    def create_dataframe_human(self, Control_BP_human, Control_RP_human, Control_SF_human, Control_spec_ent_human,
                               Control_FAA_human, Control_PAC_human, Control_coh_human, Control_phase_syn_human,
                               Activity_Control_human, Mobility_Control_human, Complexity_Control_human, Control_human,
                               freq_band, p):
        datfrm = {}
        for i in Control_human:
            datfrm[i] = pd.DataFrame({})
            if 'absolute_power' in p['features']:
                for j in range(len(freq_band)):
                    for k in range(len(p['channel'])):
                        datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                            freq_band[j] + '_power_' + p['channel'][k]: np.log10(Control_BP_human[i][j, k, :]),
                            'alpha1' + '_power_' + p['channel'][k]: np.log10(Control_BP_human[i][6, k, :]),
                            'alpha2' + '_power_' + p['channel'][k]: np.log10(Control_BP_human[i][7, k, :]),
                        })], axis=1)
            if 'relative_power' in p['features']:
                for j in range(len(freq_band)):
                    for k in range(len(p['channel'])):
                        datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                            freq_band[j] + '_rel_power_' + p['channel'][k]: np.log10(
                                (Control_RP_human[i][j, k, :]) / (1 - (Control_RP_human[i][j, k, :])))
                        })], axis=1)

            if 'hjorth' in p['features']:
                for j in range(len(freq_band)):
                    for k in range(len(p['channel'])):
                        datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                            freq_band[j] + '_activity_hjorth_' + p['channel'][k]:
                                Activity_Control_human[freq_band[j]][i][
                                    k],
                            freq_band[j] + '_mobility_hjorth_' + p['channel'][k]:
                                Mobility_Control_human[freq_band[j]][i][
                                    k],
                            freq_band[j] + '_complexity_hjorth_' + p['channel'][k]:
                                Complexity_Control_human[freq_band[j]][i][k],
                        })], axis=1)

            if 'phase amplitude coupling' in p['features']:
                for k in range(len(p['channel'])):
                    datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                        'theta_gama_PAC_' + p['channel'][k]: Control_PAC_human[i][0, k, :],
                        'theta_alpha_PAC_' + p['channel'][k]: Control_PAC_human[i][1, k, :],
                    })], axis=1)

            if 'slow_fast' in p['features']:
                for k in range(len(p['channel'])):
                    datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                        'alpha1_alpha2_' + p['channel'][k]: np.log10(
                            (Control_SF_human[i][0, k, :]) / (100 - (Control_SF_human[i][0, k, :]))),
                        'theta_alpha1_' + p['channel'][k]: np.log10(
                            (Control_SF_human[i][1, k, :]) / (100 - (Control_SF_human[i][1, k, :]))),
                        'theta_alpha2_' + p['channel'][k]: np.log10(
                            (Control_SF_human[i][2, k, :]) / (100 - (Control_SF_human[i][2, k, :]))),
                    })], axis=1)

            if 'frequency amplitude asymmetry' in p['features']:
                for j in range(len(freq_band)):
                    z = 0
                    for k in range(len(p['channel'])):
                        for l in range(len(p['channel'])):
                            if l > k:
                                datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                                    p['channel'][k] + '_' + p['channel'][l] + '_' + freq_band[j]: np.log10(
                                        (2 + (Control_FAA_human[i][j, z, :])) / (2 - (Control_FAA_human[i][j, z, :])))
                                })], axis=1)
                                z = z + 1

            if 'coherence' in p['features']:
                for j in range(len(freq_band)):
                    z = 0
                    for k in range(len(p['channel'])):
                        for l in range(len(p['channel'])):
                            if k > l:
                                datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                                    'Coh_' + freq_band[j] + '_' + p['channel'][k] + '_' + p['channel'][l]: np.log10(
                                        (Control_coh_human[i][j, z, :]) / (1 - (Control_coh_human[i][j, z, :]))),
                                    'PLV_' + freq_band[j] + '_' + p['channel'][k] + '_' + p['channel'][l]:
                                        Control_phase_syn_human[j][i][0, z, :],
                                    'Phase_' + freq_band[j] + '_' + p['channel'][k] + '_' + p['channel'][l]:
                                        Control_phase_syn_human[j][i][1, z, :],
                                })], axis=1)
                                z = z + 1

            if 'spectral_entropy' in p['features']:
                for j in range(len(freq_band)):
                    for k in range(len(p['channel'])):
                        datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                            'spec_ent_' + freq_band[j] + '_' + p['channel'][k]: -np.log10(
                                1 - (Control_spec_ent_human[i][j, k, :])),
                        })], axis=1)

            datfrm[i]['age'] = np.ones(len(datfrm[i])) * p['age'][i][0]
            datfrm[i]['sleep_label'] = [x[0] for x in self.Control_sleep_label['102']]
            datfrm[i]['Tbi_label'] = [x[1] for x in self.Control_sleep_label['102']]

        return datfrm

    # Age regression
    def age_reg_control_human(self, data):
        m = np.zeros((len(data.iloc[0]) - 3))
        b = np.zeros((len(data.iloc[0]) - 3))

        for i in range(len(data.iloc[0]) - 3):
            # print(data.columns[i])
            y = np.array(data.iloc[:, i], dtype=float)
            x = np.log10(np.array(data['age'], dtype=float))
            m[i], b[i] = np.polyfit(x, y, 1)
            data.iloc[:, i] = y - x * m[i]

        return data, m, b

    def age_reg_tbi_human(self, data, m, b):
        for i in range(len(data.iloc[0]) - 3):
            # print(data.columns[i])
            y = np.array(data.iloc[:, i], dtype=float)
            x = np.log10(np.array(data['age'], dtype=float))
            # m[i], b[i] = np.polyfit(x,y, 1)
            data.iloc[:, i] = y - x * m[i]

        return data

    # Mean removal and feature normalization
    def MR_control(self, data):
        col_mean = [0] * (len(data.iloc[0]) - 2)
        col_std = [0] * (len(data.iloc[0]) - 2)
        for i in range(len(data.iloc[0]) - 2):
            col_mean[i] = np.mean(data.iloc[:, i])
            data.iloc[:, i] = data.iloc[:, i] - col_mean[i]
            col_std[i] = np.std(data.iloc[:, i])
            data.iloc[:, i] = data.iloc[:, i] / col_std[i]
        return data, col_mean, col_std

    def MR_tbi(self, new_big_df, col_mean_test, col_std_test):
        for i in range(len(new_big_df.iloc[0]) - 2):
            # col_mean[i]=np.mean(new_big_df.iloc[:,i])
            new_big_df.iloc[:, i] = new_big_df.iloc[:, i] - col_mean_test[i]
            # col_std[i]=np.max(new_big_df.iloc[:,i])
            new_big_df.iloc[:, i] = new_big_df.iloc[:, i] / col_std_test[i]
        return new_big_df

    def fea_sel(self, data, Y, p):
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=1, verbose=False)
        rfe_selector.fit(data, Y)
        # rfe_support = rfe_selector.get_support()
        # rfe_feature = data.loc[:,rfe_support].columns.tolist()

        X = np.array(data.columns)
        Y = rfe_selector.ranking_

        Z = [x for _, x in sorted(zip(Y, X))]

        df = pd.DataFrame()

        for i in (Z):
            # ----changed to solve the columns must be same length as key -- debug 02_13_2022
            # df[Z] = data[Z]
            df = data[Z]
        i = 1
        return df.iloc[:, 0:p['max_features']], df.iloc[:, 0:p['max_features']].columns

    #--added Training_Tbi_human
    def la3(self, training, testing, Training_Control_human, Testing_Control_human, Training_Tbi_human, Testing_Tbi_human, p_human, n_clusters):
        sub_human_ea = {}
        for i in range(len(Training_Control_human)):
            sub_human_ea[Training_Control_human[i]] = training.iloc[
                                                      i * p_human['duration'] * int(60 / p_human['epoch_len']):i *
                                                                                                               p_human[
                                                                                                                   'duration'] * int(
                                                          60 / p_human['epoch_len']) + p_human[
                                                                                                                   'duration'] * int(
                                                          60 / p_human['epoch_len'])]
        for j in range(len(Training_Tbi_human)):
            sub_human_ea[Training_Tbi_human[j]] = training.iloc[
                                                  (i + j + 1) * p_human['duration'] * int(60 / p_human['epoch_len']):(
                                                                                                                             i + j + 1) *
                                                                                                                     p_human[
                                                                                                                         'duration'] * int(
                                                      60 / p_human['epoch_len']) + p_human['duration'] * int(
                                                      60 / p_human['epoch_len'])]

        # extract test subject according to class
        testing_sub_human_ea = {}
        for i in range(len(Testing_Control_human)):
            testing_sub_human_ea[Testing_Control_human[i]] = testing.iloc[
                                                             i * p_human['duration'] * int(
                                                                 60 / p_human['epoch_len']):i *
                                                                                            p_human[
                                                                                                'duration'] * int(
                                                                 60 / p_human['epoch_len']) + (p_human[
                                                                                                   'duration'] * int(
                                                                 60 / p_human['epoch_len'])), :]
        for j in range(len(Testing_Tbi_human)):
            testing_sub_human_ea[Testing_Tbi_human[j]] = testing.iloc[(i + j + 1) * p_human['duration'] * int(
                60 / p_human['epoch_len']):(i + j + 1) * p_human['duration'] * int(60 / p_human['epoch_len']) + (
                    p_human['duration'] * int(60 / p_human['epoch_len'])), :]
        # ----------------------------------------------------------------
        # calculate the cov of test subject
        c_test_human = {}
        C_test_avg_human = {}
        X_test_mr_human = {}
        for i in range(len(Testing_Control_human)):
            c = np.zeros((len(testing_sub_human_ea[Testing_Control_human[i]]), 1))
            for j in range(len(testing_sub_human_ea[Testing_Control_human[i]])):
                c[j, :] = np.cov(testing_sub_human_ea[Testing_Control_human[i]].iloc[j, :-1])
            X_test_mr_human[Testing_Control_human[i]] = testing_sub_human_ea[Testing_Control_human[i]].iloc[:, :-1].sub(
                testing_sub_human_ea[Testing_Control_human[i]].iloc[:, :-1].mean(axis=1), axis=0)
            c_test_human[Testing_Control_human[i]] = c
            C_test_avg_human[Testing_Control_human[i]] = [[np.mean(c_test_human[Testing_Control_human[i]])]]

        for i in range(len(Testing_Tbi_human)):
            c = np.zeros((len(testing_sub_human_ea[Testing_Tbi_human[i]]), 1))
            for j in range(len(testing_sub_human_ea[Testing_Tbi_human[i]])):
                c[j, :] = np.cov(testing_sub_human_ea[Testing_Tbi_human[i]].iloc[j, :-1])
            X_test_mr_human[Testing_Tbi_human[i]] = testing_sub_human_ea[Testing_Tbi_human[i]].iloc[:, :-1].sub(
                testing_sub_human_ea[Testing_Tbi_human[i]].iloc[:, :-1].mean(axis=1), axis=0)
            c_test_human[Testing_Tbi_human[i]] = c
            C_test_avg_human[Testing_Tbi_human[i]] = [[np.mean(c_test_human[Testing_Tbi_human[i]])]]
        # ----------------------------------------------------------------
        # combine cov of test subjects
        Ci_test = np.zeros((1, 1))
        for i in range(len(Testing_Control_human)):
            Ci_test = np.append(Ci_test, c_test_human[Testing_Control_human[i]], axis=0)

        for i in range(len(Testing_Tbi_human)):
            Ci_test = np.append(Ci_test, c_test_human[Testing_Tbi_human[i]], axis=0)
        Ci_test = Ci_test[1:, :]
        # ----------------------------------------------------------------
        # perform k-medoid clustering depending on how many true labeled datapoint required
        kmedoids2 = KMedoids(n_clusters, random_state=0).fit(Ci_test)
        # print(kmedoids2.medoid_indices_)
        # ----------------------------------------------------------------
        # label the centers of clusters
        mask = [True] * n_clusters
        for i in range(n_clusters):
            if kmedoids2.medoid_indices_[i] < (
                    len(Testing_Control_human) * p_human['duration'] * int(60 / p_human['epoch_len'])):
                mask[i] = False
        mask2 = [not (x) for x in mask]
        if len(np.unique(mask2)) > 1:
            ci_Control_ref = [[np.mean(kmedoids2.cluster_centers_[mask2]).tolist()]]
            ci_Tbi_ref = [[np.mean(kmedoids2.cluster_centers_[mask]).tolist()]]
            # ----------------------------------------------------------------
            # Calculate cov of human
            c_human = {}
            C_avg_human = {}
            X_mr_human = {}
            for i in range(len(Training_Control_human)):
                c = np.zeros((len(sub_human_ea[Training_Control_human[i]]), 1))
                for j in range(len(sub_human_ea[Training_Control_human[i]])):
                    c[j, :] = np.cov(sub_human_ea[Training_Control_human[i]].iloc[j, :-1])
                X_mr_human[Training_Control_human[i]] = sub_human_ea[Training_Control_human[i]].iloc[:, :-1].sub(
                    sub_human_ea[Training_Control_human[i]].iloc[:, :-1].mean(axis=1), axis=0)
                c_human[Training_Control_human[i]] = c
                C_avg_human[Training_Control_human[i]] = [[np.mean(c_human[Training_Control_human[i]])]]

            for i in range(len(Training_Tbi_human)):
                c = np.zeros((len(sub_human_ea[Training_Tbi_human[i]]), 1))
                for j in range(len(sub_human_ea[Training_Tbi_human[i]])):
                    c[j, :] = np.cov(sub_human_ea[Training_Tbi_human[i]].iloc[j, :-1])
                X_mr_human[Training_Tbi_human[i]] = sub_human_ea[Training_Tbi_human[i]].iloc[:, :-1].sub(
                    sub_human_ea[Training_Tbi_human[i]].iloc[:, :-1].mean(axis=1), axis=0)
                c_human[Training_Tbi_human[i]] = c
                C_avg_human[Training_Tbi_human[i]] = [[np.mean(c_human[Training_Tbi_human[i]])]]
                # ----------------------------------------------------------------
            R_human = {}
            R_human['target_control'] = (sqrtm(ci_Control_ref))
            R_human['target_tbi'] = (sqrtm(ci_Tbi_ref))
            # ----------------------------------------------------------------
            # Calculate alligned human data
            A_human = {}
            control_new_human = {}
            tbi_new_human = {}
            for i in range(len(Training_Control_human)):
                # print('control #',i+1)
                R_human[Training_Control_human[i]] = np.linalg.inv(sqrtm(C_avg_human[Training_Control_human[i]]))
                A_human[Training_Control_human[i]] = np.matmul(R_human['target_control'],
                                                               R_human[Training_Control_human[i]])
                a = np.zeros(np.shape(sub_human_ea[Training_Control_human[i]].iloc[:, :-1]))
                for j in range(len(sub_human_ea[Training_Control_human[i]])):
                    a[j, :] = np.matmul(A_human[Training_Control_human[i]],
                                        np.array(X_mr_human[Training_Control_human[i]].iloc[j, :]).reshape(1, len(
                                            X_mr_human[Training_Control_human[i]].iloc[j, :])))
                control_new_human[Training_Control_human[i]] = pd.DataFrame(a,
                                                                            columns=sub_human_ea[Training_Control_human[
                                                                                i]].columns[:-1])
                control_new_human[Training_Control_human[i]]['Tbi_label'] = 0.0

            for i in range(len(Training_Tbi_human)):
                # print('tbi #',i+1)
                R_human[Training_Tbi_human[i]] = np.linalg.inv(sqrtm(C_avg_human[Training_Tbi_human[i]]))
                A_human[Training_Tbi_human[i]] = np.matmul(R_human['target_tbi'], R_human[Training_Tbi_human[i]])
                a = np.zeros(np.shape(sub_human_ea[Training_Tbi_human[i]].iloc[:, :-1]))
                for j in range(len(sub_human_ea[Training_Tbi_human[i]])):
                    a[j, :] = np.matmul(A_human[Training_Tbi_human[i]],
                                        np.array(X_mr_human[Training_Tbi_human[i]].iloc[j, :]).reshape(1, len(
                                            X_mr_human[Training_Tbi_human[i]].iloc[j, :])))
                tbi_new_human[Training_Tbi_human[i]] = pd.DataFrame(a,
                                                                    columns=sub_human_ea[Training_Tbi_human[i]].columns[
                                                                            :-1])
                tbi_new_human[Training_Tbi_human[i]]['Tbi_label'] = 1.0
            # ----------------------------------------------------------------
            # Calculate new cov of human sub
            pi = {}

            for i in range(len(Training_Control_human)):
                v = np.zeros((len(control_new_human[Training_Control_human[i]]), 1))
                for j in range(len(control_new_human[Training_Control_human[i]])):
                    v[j, :] = np.cov(control_new_human[Training_Control_human[i]].iloc[j, :-1])
                pi[Training_Control_human[i]] = v

            for i in range(len(Training_Tbi_human)):
                v = np.zeros((len(tbi_new_human[Training_Tbi_human[i]]), 1))
                for j in range(len(tbi_new_human[Training_Tbi_human[i]])):
                    v[j, :] = np.cov(tbi_new_human[Training_Tbi_human[i]].iloc[j, :-1])
                pi[Training_Tbi_human[i]] = v
            # ----------------------------------------------------------------
            # Concatinate data
            Dt_control = pd.DataFrame()
            for k in (Training_Control_human):
                Dt_control = Dt_control.append(control_new_human[k], ignore_index=True)

            Dt_tbi = pd.DataFrame()
            for k in (Training_Tbi_human):
                Dt_tbi = Dt_tbi.append(tbi_new_human[k], ignore_index=True)
        else:
            Dt_control = pd.DataFrame()
            for k in (Training_Control_human):
                Dt_control = Dt_control.append(sub_human_ea[k], ignore_index=True)

            Dt_tbi = pd.DataFrame()
            for k in (Training_Tbi_human):
                Dt_tbi = Dt_tbi.append(sub_human_ea[k], ignore_index=True)

        return Dt_control, Dt_tbi

    # CORAL
    def coral(self, training, testing):
        Cs = np.cov(training.iloc[:, 0:-1].astype(
            'float64').T)  # +np.eye(len(np.cov(big_df_Training_mice.iloc[:,0:-1].astype('float64').T)))
        Ct = np.cov(testing.iloc[:, 0:-1].astype(
            'float64').T)  # +np.eye(len(np.cov(big_df_Training_human.iloc[:,0:-1].astype('float64').T)))

        Ds_changed = np.matmul(training.iloc[:, 0:-1], np.linalg.inv(sqrtm(Cs)))
        Ds_star = np.matmul(Ds_changed, sqrtm(Ct))
        training.iloc[:, 0:-1] = np.real(Ds_star)
        return training, testing

    def coral_main(self, testing, training):
        # plot_coral(training,testing)

        c2 = np.zeros((len(testing), 1))
        for i in range(len(testing)):
            c2[i, 0] = np.cov(testing.iloc[i, :-1])

        kmedoids3 = KMedoids(n_clusters=10, random_state=0).fit(c2)

        dt = pd.DataFrame(columns=testing.columns)
        for i in range(len(kmedoids3.medoid_indices_)):
            dt = dt.append(testing.iloc[kmedoids3.medoid_indices_[i], :], ignore_index=True)

        training, dt = self.coral(training, dt)
        return training

class Loading_mice():
    def __init__(self):
        self.p_mice = None
        self.data_folder_mice = None
        self.Old_Control_mice = None
        self.Old_Tbi_mice     = None
        self.New_Control_mice = None
        self.New_Tbi_mice     = None
        self.data_mice        = None

        pass

    class functions(object):
        def __init__(self):
            pass

        # Extract data

        # Extract data from .edf file
        def extract_mice_sleepdata(self, mice, p, data_folder, label, label2):
            no_data = []
            if label == 'Control':
                control = {}
            else:
                tbi = {}

            if label2 == 'new':
                df2 = pd.read_csv(data_folder + '\\' + 'Channel_Contents' + '.txt', header=None, index_col=None,
                                  error_bad_lines=False)
                groups = df2.iloc[1:, 0].str.split(expand=True)
                groups.columns = ['File', 'EMG', 'EEG', 'EEG', 'Group', 'Label']

            for l in range(len(mice)):
                print()
                print('Subject #', mice[l])

                if label2 == 'new':
                    df = pd.read_csv(data_folder + '\\' + mice[l] + '_Stages.txt', error_bad_lines=False,
                                     sep='delimiter', header=None)
                    if label == 'Control':
                        nu = 16
                        df = df.drop(df.index[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
                    elif label == 'Tbi':
                        nu = 0
                    x = df.loc[nu]
                    a = x.str.split('\t', expand=True)
                    for j in range(nu, len(df) + nu - 1):
                        x = df.loc[j]
                        b = x.str.split('\t', expand=True)
                        a = a.append(b)
                    a.columns = a.iloc[0]
                    a = a[a.iloc[:, 3] != 'Stage']
                    a = a.iloc[2:]
                elif label2 == 'old':
                    df = pd.read_csv(data_folder + '\\' + mice[l] + '_Stages.csv', sep='delimiter', header=None)
                    df = df.drop(df.index[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]])
                    # df.iloc[0:5,:]
                    x = df.loc[18]
                    a = x.str.split(',', expand=True)
                    for j in range(19, len(df) + 18):
                        x = df.loc[j]
                        b = x.str.split(',', expand=True)
                        a = a.append(b)
                    a.columns = a.iloc[0]
                    a = a[a.iloc[:, 3] != 'Stage']
                    a = a.iloc[2:]

                raw = mne.io.read_raw_edf(data_folder + '\\' + mice[l] + ".edf", preload=True, verbose=None)

                if p['filter_extract'] == 'tim':
                    filt_raw = raw.copy()
                    filt_raw.load_data().filter(p['lf2'], p['hf2'])
                    raw = filt_raw

                raw_eeg = raw[:, :][0]
                if label == 'Control' and label2 == 'old':
                    if mice[l] == 'Sham104_BL5':
                        index = raw.ch_names.index('SHA' + mice[l][4:7] + '-E')
                    elif mice[l] == 'Sham105_BL5':
                        index = raw.ch_names.index('SH' + mice[l][4:7] + '-E')
                    else:
                        index = raw.ch_names.index('SHM' + mice[l][4:7] + '-E')
                elif label == 'Tbi' and label2 == 'old':
                    if mice[l] == 'TBI106_BL5':
                        index = raw.ch_names.index('TBI' + mice[l][3:6] + 'E')
                    else:
                        index = raw.ch_names.index('TBI' + mice[l][3:6] + '-E')
                elif label2 == 'new':
                    ch_inter = (groups.iloc[groups[groups['File'] == mice[l]].index.values - 1, 2])
                    if len(ch_inter.iloc[0]) == 1:
                        ch = 'Ch0' + (ch_inter.iloc[0])
                    else:
                        ch = 'Ch' + (ch_inter.iloc[0])
                    for i in range(len(raw.ch_names)):
                        if raw.ch_names[i][0:4] == ch:
                            index = i
                raw_data = (raw_eeg[index, :])

                count = 0
                j = 0
                data_control = []

                for i in range(len(a)):
                    if a.iloc[i, 2] == p['sleep_stage']:
                        # print(i+1)
                        count = count + 1
                        data_control[j:j + 4 * p['fs']] = (raw_data[i * p['fs'] * 4:i * p['fs'] * 4 + (p['fs'] * 4)])
                        # print(j)
                        j = j + 4 * p['fs']

                if len(data_control) != 0:
                    data_control = data_control - np.mean(data_control)

                    if label == 'Control':
                        # name = 'control_'+p['sleep_stage']+mice[l]#+'_'+p['channel'][e]#.split('-')[0]
                        name = mice[l]
                        control[name] = data_control.reshape((1, len(data_control)))
                        # print(data_control)
                    else:
                        # name = 'tbi_'+p['sleep_stage']+mice[l]#+'_'+p['channel'][e]#.split('-')[0]
                        name = mice[l]
                        tbi[name] = data_control.reshape((1, len(data_control)))
                        # print(data_tbi)
                else:
                    if mice[l] in no_data:
                        pass
                    else:
                        no_data.append(mice[l])

                print('Subject data shape ', np.shape(data_control))
                # print(count)
            if len(no_data) != 0:
                print('----------------------------------------------------------')
                print('No ' + p['sleep_stage'] + ' stage in ' + label + ' subjects ', no_data)

            #    for ij in mice:
            #        print(mice)
            #        if ij in no_data:
            #            mice.remove(ij)
            #            print(ij)
            #            print(mice)
            ind = np.where(np.in1d(mice, no_data))[0]
            new_list = [val for n, val in enumerate(mice) if n not in ind]
            print('==========================================================')

            if label == 'Control':
                return control, new_list, no_data
            else:
                return tbi, new_list, no_data
    pass

    def load_parameters(self):
        # Initial values

        p_mice = dict()
        # -----------------------------------------------------------------------------
        # Filtering Parameters

        p_mice['fs'] = 256

        p_mice['filter_extract'] = 'freq'
        # 'tim','freq'
        # Filter using mne while extracting
        if p_mice['filter_extract'] == 'tim':
            p_mice['lf2'] = 0.5
            p_mice['hf2'] = 50

        p_mice['ica'] = 'N'

        p_mice['sleep_stage'] = 'NR'   #NR
        # W,NR, R

        p_mice['epoch_len'] = 28
        # sec

        p_mice['duration'] = 30
        #(min) 0.0104

        p_mice['features'] = ['absolute_power', 'relative_power', 'slow_fast', 'frequency amplitude asymmetry',
                              'phase synchrony', 'coherence', 'hjorth', 'spectral_entropy', 'phase amplitude coupling']
        p_mice['max_features'] = 30

        self.p_mice = p_mice

        return self.p_mice

    def load_other_parameters(self):
        # value in min or 'NA'
        self.data_folder_mice = os.path.abspath(".") + '/data_mice_01'
        self.Old_Control_mice = ['Sham102_BL5', 'Sham103_BL5', 'Sham104_BL5', 'Sham107_BL5', 'Sham108_BL5']
        self.Old_Tbi_mice = ['TBI101_BL5', 'TBI102_BL5', 'TBI103_BL5', 'TBI104_BL5', 'TBI106_BL5']
        self.New_Control_mice = ['m010_TPE01_BaselineDay2_sham', 'm010_TPE03_BaselineDay2_sham',
                            'm010_TPE06_BaselineDay2_sham', 'm010_TPE08_BaselineDay2_sham',
                            'm010_TPE10_BaselineDay2_sham', 'm010_TPE12_BaselineDay2_sham',
                            'm010_TPE15_BaselineDay2_sham', 'm010_TPE18_BaselineDay2_sham',
                            'm010_TPE19_BaselineDay2_sham']
        self.New_Tbi_mice = ['m001_M60_BaselineDay1_CCI', 'm001_L56_BaselineDay1_CCI', 'm001_L55_BaselineDay1_CCI',
                        'm001_K51_BaselineDay1_CCI', 'm001_J43_BaselineDay1_CCI', 'm001_I41_BaselineDay1_CCI',
                        'm001_H36_BaselineDay1_CCI', 'm001_G29_BaselineDay1_CCI', 'm001_E20_BaselineDay1_CCI']
        self.data_mice = 'both'
        # 'old','new','both'

        return self.data_folder_mice, self.Old_Control_mice, self.Old_Tbi_mice, self.New_Control_mice, self.New_Tbi_mice, self.data_mice


    def load_data(self):

        import time, os
        import pickle
        dataFile = os.path.abspath('.') + '/mice_eeg_data_NR_both_07_14_2022.pkl'
        time1 = time.time()
        if os.path.exists(dataFile):
            with open(dataFile, 'rb') as file:
                data_dict = pickle.load(file)
        else:
            if self.data_mice == 'old':
                print('Loading old mice data................')
                Control_data_mice, Control_mice, no_data_control_mice = self.functions().extract_mice_sleepdata(self.Old_Control_mice, self.p_mice,
                                                                                               self.data_folder_mice,
                                                                                               'Control', 'old')

                Tbi_data_mice, Tbi_mice, no_data_tbi_mice = self.functions().extract_mice_sleepdata(self.Old_Tbi_mice, self.p_mice,
                                                                                               self.data_folder_mice, 'Tbi', 'old')
                print()
            elif self.data_mice == 'new':
                print('Loading new mice data................')
                Control_data_mice, Control_mice, no_data_control_mice = self.functions().extract_mice_sleepdata(self.New_Control_mice, self.p_mice,
                                                                                               self.data_folder_mice,
                                                                                               'Control', 'new')
                Tbi_data_mice, Tbi_mice, no_data_tbi_mice = self.functions().extract_mice_sleepdata(self.New_Tbi_mice, self.p_mice,
                                                                                   self.data_folder_mice, 'Tbi', 'new')
                print()
            elif self.data_mice == 'both':
                print('Loading old data................')
                Old_Control_data_mice, Old_Control_mice, old_no_data_control_mice = self.functions().extract_mice_sleepdata(
                    self.Old_Control_mice, self.p_mice, self.data_folder_mice, 'Control', 'old')
                Old_Tbi_data_mice, Old_Tbi_mice, old_no_data_tbi_mice = self.functions().extract_mice_sleepdata(self.Old_Tbi_mice, self.p_mice,
                                                                                               self.data_folder_mice, 'Tbi',
                                                                                               'old')
                print()
                print('Loading new data................')
                New_Control_data_mice, New_Control_mice, new_no_data_control_mice = self.functions().extract_mice_sleepdata(
                    self.New_Control_mice, self.p_mice, self.data_folder_mice, 'Control', 'new')
                New_Tbi_data_mice, New_Tbi_mice, new_no_data_tbi_mice = self.functions().extract_mice_sleepdata(self.New_Tbi_mice, self.p_mice,
                                                                                               self.data_folder_mice, 'Tbi',
                                                                                               'new')
                print()
                Control_data_mice = {**Old_Control_data_mice, **New_Control_data_mice}
                Control_mice = Old_Control_mice + New_Control_mice
                no_data_control_mice = old_no_data_control_mice + new_no_data_control_mice
                Tbi_data_mice = {**Old_Tbi_data_mice, **New_Tbi_data_mice}
                Tbi_mice = Old_Tbi_mice + New_Tbi_mice
                no_data_tbi_mice = old_no_data_tbi_mice + new_no_data_tbi_mice


            Control_dict = {'Control_data_mice': Control_data_mice, 'Control_mice': Control_mice, 'no_data_control_mice': no_data_control_mice}
            Tbi_dict     = {'Tbi_data_mice': Tbi_data_mice, 'Tbi_mice': Tbi_mice, 'no_data_tbi_mice': no_data_tbi_mice}

            data_dict = {'Control_dict': Control_dict, 'Tbi_dict': Tbi_dict}
            with open(dataFile, 'wb') as file:
                pickle.dump(data_dict, file, pickle.HIGHEST_PROTOCOL)

        return data_dict

    # Create epochs

    def create_epoch(self, data, p):
        epoch_data = {}
        for ele in data:
            a = []
            num_epoch = int(len(data[ele]) / (p['epoch_len'] * p['fs']))

            for i in range(num_epoch):
                a.append(
                    data[ele][i * p['fs'] * p['epoch_len']:i * p['fs'] * p['epoch_len'] + p['fs'] * p['epoch_len']])
            epoch_data[ele] = a
        return epoch_data

    # Artifact removal

    def artifact_mice_removal3(self, data, p, mice, label):
        if label == 'Control':
            name = 'control_' + p['sleep_stage']
        elif label == 'Tbi':
            name = 'tbi_' + p['sleep_stage']

        AR_rem = {}
        m2 = {}
        for i in mice:
            # mask=np.array((len(p['channel']),len(Control_epoch['control_R101_C3-A2'])))
            # mask=[]
            m2[i] = np.zeros(len(data[name + i]), dtype=bool)

            a = (np.max(data[name + i], axis=1) - np.min(data[name + i], axis=1))
            z = ((a - np.mean(a)) / np.std(a))
            avglatlist = np.arange(1, a.shape[0] + 1)
            m = np.abs(z) > 2.5
            # mask.append(m)
            m2[i] = np.logical_or(m2[i], m)

            # a=(np.var(data[name+i+'_'+j],axis=1))
            a = np.sum(np.abs(data[name + i]), axis=1)
            z = ((a - np.mean(a)) / np.std(a))
            m = np.abs(z) > 2.5
            # mask.append(m)
            m2[i] = np.logical_or(m2[i], m)

            AR_rem[name + i] = np.delete(data[name + i], avglatlist[m2[i]] - 1, 0)

        mini = 10 ** 10

        for ele in AR_rem:
            if len(AR_rem[ele]) < mini:
                mini = len(AR_rem[ele])
        # print('.......')
        return AR_rem, mini, m2

    # Select number of epochs

    def length_sel(self, data, p):
        for ele in data:
            if p['duration'] != 'NA':
                data[ele] = data[ele][0:p['duration'] * int(60 / p['epoch_len'])]
            elif p['duration'] == 'NA':
                data[ele] = data[ele]
        return data

    # Filter

    def filt(self, sig, p, lf1, lf2, hf2, hf1):
        x = sig - np.mean(sig)
        no_samp = len(x)
        mfft = round(np.log(len(x)) / np.log(2) + 0.5) + 1
        nfft = int(2 ** mfft)

        xf = np.fft.fft(x, int(nfft))
        xf2 = np.zeros((nfft), dtype='complex_')
        lp2 = int(np.fix((hf1 / p['fs']) * nfft))
        lp22 = int(np.fix((hf2 / p['fs']) * nfft))
        wl1 = np.ones(lp2)
        wl1[lp22:lp2] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(lp2 - lp22) + 1) / (lp2 - lp22))

        wl2 = wl1[::-1]
        xf2[0:lp2] = np.multiply(xf[0:lp2], np.transpose(wl1))
        xf2[nfft - lp2:nfft] = np.multiply(xf[nfft - lp2:nfft], np.transpose(wl2))

        qf1 = lf1
        lq2 = int(np.fix((qf1 / p['fs']) * nfft + 1))
        qf2 = lf2
        lq22 = int(np.fix((qf2 / p['fs']) * nfft + 1))  # fix(0.5*lp2);
        ql1 = np.zeros(lq22)
        ql1[lq2:lq22] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(lq22 - lq2) + 1) / (lq22 - lq2))
        ql2 = ql1[::-1]
        xf2[0:lq22] = np.multiply(xf2[0:lq22], ql1)
        xf2[nfft - lq22:nfft] = np.multiply(xf2[nfft - lq22:nfft], ql2)

        xi = np.real(np.fft.ifft(xf2, nfft))
        xi = xi[0:no_samp]

        return xi

    def freq_filt(self, unfiltered_data, p):
        filtered_data = {}

        if p['freq_band'] == 'delta':
            lf1 = 0.5
            lf2 = 1
            hf2 = 4
            hf1 = 4.5
        elif p['freq_band'] == 'theta':
            lf1 = 3.5
            lf2 = 4
            hf2 = 8
            hf1 = 8.5
        elif p['freq_band'] == 'alpha':
            lf1 = 7.5
            lf2 = 8
            hf2 = 12
            hf1 = 12.5
        elif p['freq_band'] == 'sigma':
            lf1 = 12.5
            lf2 = 13
            hf2 = 16
            hf1 = 16.5
        elif p['freq_band'] == 'beta':
            lf1 = 16.5
            lf2 = 17
            hf2 = 25
            hf1 = 25.5
        elif p['freq_band'] == 'gama':
            lf1 = 29.5
            lf2 = 30
            hf2 = 40
            hf1 = 40.5
        elif p['freq_band'] == 'slow':
            lf1 = 0.5
            lf2 = 1
            hf2 = 6
            hf1 = 6.5
        elif p['freq_band'] == 'normal':
            lf1 = 0.5
            lf2 = 1
            hf2 = 50
            hf1 = 50.5

        for i in unfiltered_data:
            # print(lf2)
            filtered_data[i] = np.zeros(np.shape(unfiltered_data[i]))

            for j in range(len(unfiltered_data[i])):
                filtered_data[i][j] = self.filt(unfiltered_data[i][j], p, lf1, lf2, hf2, hf1)

        return filtered_data





