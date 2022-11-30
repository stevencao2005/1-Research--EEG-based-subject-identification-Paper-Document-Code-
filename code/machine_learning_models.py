"""
@authors: Steven Cao: https://github.com/stevencao2005,
          Manoj Vishwanath: https://github.com/ManojVishwanath

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


class Models(object):

    def __init__(self):
        pass

    # Decision_Tree Report
    def Dtree_Classifer(self, X_train, X_test, y_train, y_test):
        dtree = DecisionTreeClassifier()  # random_state=10)
        dtree.fit(X_train, y_train)
        predictions = dtree.predict(X_test)

        # print('Accuracy and Confusion matrix for Dtree:')
        # Check accuracy between predictions from Dtree and y_test
        accuracy_score1 = accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
        auc_sco = metrics.auc(fpr, tpr)
        prec_sco = precision_score(y_test, predictions, average='weighted')
        recall_sco = recall_score(y_test, predictions, average='weighted')
        f1_sco = f1_score(y_test, predictions, average='weighted')

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        spec_sco = tn / (tn + fp)
        # print(matrix1)
        # Report
        # print('Classification report for decision tree')
        # print(classification_report(y_test,predictions))

        return accuracy_score1, auc_sco, prec_sco, spec_sco, recall_sco, f1_sco

    # kNN Report
    def KNN_Classifer(self, X_train, X_test, y_train, y_test, fixed_k):
        # Calculating error for K values between 1 and 40 in KNN
        '''
        error = []
        for i in range(2, 50, 1):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error.append(np.mean(pred_i != y_test))

        plt.figure(figsize=(12, 6))
        plt.plot(range(2, 50, 1), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
        plt.title('Error Rate V/S K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        '''
        k = fixed_k

        # print('K:', k)
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # print('Accuracy and Confusion matrix for KNN:')
        # Check accuracy between predictions from KNN Classifier and y_test
        accuracy_score2 = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc_sco = metrics.auc(fpr, tpr)
        prec_sco = precision_score(y_test, y_pred, average='weighted')
        recall_sco = recall_score(y_test, y_pred, average='weighted')
        f1_sco = f1_score(y_test, y_pred, average='weighted')
        # print('accuracy=', accuracy_score2)
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec_sco = tn / (tn + fp)
        # print(matrix2)
        # Report
        # print('Classification report for kNN')
        # print(classification_report(y_test, y_pred))

        return accuracy_score2, fixed_k, auc_sco, prec_sco, spec_sco, recall_sco, f1_sco

    # MLP Classifier report
    def MLP_Classifer(self, X_train, X_test, y_train, y_test):
        mlp = MLPClassifier(hidden_layer_sizes=(100, 10), max_iter=10000)  # , random_state=11)
        mlp.fit(X_train, y_train)
        predictions = mlp.predict(X_test)
        # print('Accuracy and Confusion matrix for MLP:')
        # Check accuracy between predictions from MLPClassifier and y_test
        accuracy_score3 = accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
        auc_sco = metrics.auc(fpr, tpr)  # print('accuracy=', accuracy_score3)
        prec_sco = precision_score(y_test, predictions, average='weighted')
        recall_sco = recall_score(y_test, predictions, average='weighted')
        f1_sco = f1_score(y_test, predictions, average='weighted')

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        spec_sco = tn / (tn + fp)
        # print(matrix3)
        # Report
        # print('Classification report for Neural_network')
        # print(classification_report(y_test,predictions))
        # plt.plot(index,Compare)
        # plt.show()

        return accuracy_score3, auc_sco, prec_sco, spec_sco, recall_sco, f1_sco

    # Random forest Classifier Report
    def Random_Forest_Classifer(self, X_train, X_test, y_train, y_test):
        clf = RandomForestClassifier()  # random_state=40)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # print('Accuracy and Confusion matrix for Random Forest:')
        # Check accuracy between predictions from Randomforest Classifier and y_test
        accuracy_score4 = accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
        auc_sco = metrics.auc(fpr, tpr)
        prec_sco = precision_score(y_test, predictions, average='weighted')
        recall_sco = recall_score(y_test, predictions, average='weighted')
        f1_sco = f1_score(y_test, predictions, average='weighted')
        # print('accuracy=', accuracy_score4)
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        spec_sco = tn / (tn + fp)
        # print(matrix4)
        # Report
        # print('Classification report for Random forest')
        # print(classification_report(y_test,predictions))

        return accuracy_score4, auc_sco, prec_sco, spec_sco, recall_sco, f1_sco

    # SVC Classifier Report
    def SVC_Classifer(self, X_train, X_test, y_train, y_test):
        svclassifier = SVC(kernel='rbf')  # , random_state=74)
        svclassifier.fit(X_train, y_train)
        predictions = svclassifier.predict(X_test)

        # print('Accuracy and Confusion matrix for SVC:')
        # Check accuracy between predictions from SVC Classifier and y_test
        accuracy_score5 = accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
        auc_sco = metrics.auc(fpr, tpr)
        prec_sco = precision_score(y_test, predictions, average='weighted')
        recall_sco = recall_score(y_test, predictions, average='weighted')
        f1_sco = f1_score(y_test, predictions, average='weighted')
        # print('accuracy=', accuracy_score5)
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        spec_sco = tn / (tn + fp)
        # print(matrix5)
        # Report
        # print('Classification report for SVM')
        # print(classification_report(y_test,predictions))

        return accuracy_score5, auc_sco, prec_sco, spec_sco, recall_sco, f1_sco

    # XGBoost Classifier Report
    def XGboost_Classifer(self, X_train, X_test, y_train, y_test):
        model = XGBClassifier(verbosity=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy_score6 = accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
        auc_sco = metrics.auc(fpr, tpr)
        prec_sco = precision_score(y_test, predictions, average='weighted')
        recall_sco = recall_score(y_test, predictions, average='weighted')
        f1_sco = f1_score(y_test, predictions, average='weighted')

        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        spec_sco = tn / (tn + fp)

        return accuracy_score6, auc_sco, prec_sco, spec_sco, recall_sco, f1_sco

    # ML Classifier
    def ML_Classifier(self, Training_Dataframe, Testing_Dataframe):
        # ------------------------------------------------------------------------------------Training/Testing human Set------------------------------------------------------------------
        big_df_Training_human = shuffle(Training_Dataframe)
        big_df_Testing_human = shuffle(Testing_Dataframe)

        # Make train and test set

        X_train_human = big_df_Training_human.drop('Tbi_label', axis=1)
        y_train_human = big_df_Training_human['Tbi_label']
        y_train_human = y_train_human.astype(int)
        X_test_human = big_df_Testing_human.drop('Tbi_label', axis=1)
        y_test_human = big_df_Testing_human['Tbi_label']
        y_test_human = y_test_human.astype(int)

        # --------------------------------------------------------------------------------------ML Classifiers----------------------------------------------------------------------------

        accuracy_Dtree = np.zeros((6, 10))
        accuracy_KNN1 = np.zeros((6, 10))
        accuracy_KNN2 = np.zeros((6, 10))
        accuracy_KNN3 = np.zeros((6, 10))
        accuracy_MLP = np.zeros((6, 10))
        accuracy_Random_Forest = np.zeros((6, 10))
        accuracy_SVC = np.zeros((6, 10))
        accuracy_XGB = np.zeros((6, 10))
        # k_fix = [3,5,7]

        for i in range(10):

            # Decision_Tree Report
            accuracy_Dtree[0, i], accuracy_Dtree[1, i], accuracy_Dtree[2, i], accuracy_Dtree[3, i], accuracy_Dtree[
                4, i], \
            accuracy_Dtree[5, i] = self.Dtree_Classifer(X_train_human, X_test_human, y_train_human, y_test_human)

            # MLP Classifier report
            accuracy_MLP[0, i], accuracy_MLP[1, i], accuracy_MLP[2, i], accuracy_MLP[3, i], accuracy_MLP[4, i], \
            accuracy_MLP[5, i] = self.MLP_Classifer(X_train_human, X_test_human, y_train_human, y_test_human)

            # Random forest Classifier Report
            accuracy_Random_Forest[0, i], accuracy_Random_Forest[1, i], accuracy_Random_Forest[2, i], \
            accuracy_Random_Forest[3, i], accuracy_Random_Forest[4, i], accuracy_Random_Forest[
                5, i] = self.Random_Forest_Classifer(X_train_human, X_test_human, y_train_human, y_test_human)

            # SVC Classifier Report
            accuracy_SVC[0, i], accuracy_SVC[1, i], accuracy_SVC[2, i], accuracy_SVC[3, i], accuracy_SVC[4, i], \
            accuracy_SVC[5, i] = self.SVC_Classifer(X_train_human, X_test_human, y_train_human, y_test_human)


            # kNN Report and Calculating error for K values between 1 and 40
            accuracy_KNN1[0, i], k_1, accuracy_KNN1[1, i], accuracy_KNN1[2, i], accuracy_KNN1[3, i], accuracy_KNN1[
                4, i], \
            accuracy_KNN1[5, i] = self.KNN_Classifer(X_train_human, X_test_human, y_train_human, y_test_human, 5)
            accuracy_KNN2[0, i], k_2, accuracy_KNN2[1, i], accuracy_KNN2[2, i], accuracy_KNN2[3, i], accuracy_KNN2[
                4, i], \
            accuracy_KNN2[5, i] = self.KNN_Classifer(X_train_human, X_test_human, y_train_human, y_test_human, 11)
            accuracy_KNN3[0, i], k_3, accuracy_KNN3[1, i], accuracy_KNN3[2, i], accuracy_KNN3[3, i], accuracy_KNN3[
                4, i], \
            accuracy_KNN3[5, i] = self.KNN_Classifer(X_train_human, X_test_human, y_train_human, y_test_human, 19)
            # XGBoost
            accuracy_XGB[0, i], accuracy_XGB[1, i], accuracy_XGB[2, i], accuracy_XGB[3, i], accuracy_XGB[4, i], \
            accuracy_XGB[5, i] = self.XGboost_Classifer(X_train_human, X_test_human, y_train_human, y_test_human)

        return np.mean(accuracy_Dtree, axis=1) * 100, np.mean(accuracy_KNN1, axis=1) * 100, k_1, np.mean(accuracy_KNN2,
                                                                                                         axis=1) * 100, k_2, np.mean(
            accuracy_KNN3, axis=1) * 100, k_3, np.mean(accuracy_Random_Forest, axis=1) * 100, np.mean(accuracy_MLP,
                                                                                                      axis=1) * 100, np.mean(
            accuracy_SVC, axis=1) * 100, np.mean(accuracy_XGB, axis=1) * 100
