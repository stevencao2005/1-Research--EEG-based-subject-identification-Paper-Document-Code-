"""
@authors: Steven Cao: https://github.com/stevencao2005,
          Manoj Vishwanath: https://github.com/ManojVishwanath

"""


#IMPORT ALL NEEDED MODULES
import os
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
from load_and_preprocess_utils import Loading_human
from machine_learning_models import Models
from preprocess_for_dl_models import Preprocessing_For_DL_Models

#INITIALIZING PARAMETERS
t1_start               = time.time()
load                   = Loading_human()
p                      = load.load_parameters()
age_regg               = 'Y'


#LOAD DATA
data_dict              = load.load_data()

Control_dict           = data_dict['Control_dict']
Tbi_dict               = data_dict['Tbi_dict']
Control                = Control_dict['Control']
Control_human          = Control_dict['Control_human']
Control_sleep_label    = Control_dict['Control_sleep_label']
Tbi                    = Tbi_dict['Tbi']
Tbi_human              = Tbi_dict['Tbi_human']
Tbi_sleep_label        = Tbi_dict['Tbi_sleep_label']


#CREATE EPOCHS
print('Creating human epochs of length '+str(p['epoch_len'])+' sec..............')
Control_epoch_human                                    = load.create_epoch(Control,p,'Control')
Tbi_epoch_human                                        = load.create_epoch(Tbi,p,'Tbi')
print()

#ARTIFACT REMOVAL
print('Removing human epochs with artifacts.......')
Control_AR_human,min_control_human,Control_mask_human  = load.artifact_human_removal3(Control_epoch_human,p)
Tbi_AR_human,min_tbi_human,Tbi_mask_human              = load.artifact_human_removal3(Tbi_epoch_human,p)

min_dur_human                                          = int(min(min_control_human,min_tbi_human)*p['epoch_len']/60)
print('Minimum duration human '+str(min_dur_human)+' min')
print()

#SELECTING CORRESPONDING LABELS
Control_sleep_label = load.label_sel(Control_sleep_label,Control_mask_human)
Tbi_sleep_label     = load.label_sel(Tbi_sleep_label,Tbi_mask_human)

if p['duration'] != 'NA' and p['duration']<min_dur_human:
    print('Choosing first '+str(p['duration'])+'min data from each human sub')
    Control_AR_human,Control_sleep_label     = load.length_sel(Control_AR_human,Control_sleep_label,p)
    Tbi_AR_human,Tbi_sleep_label             = load.length_sel(Tbi_AR_human,Tbi_sleep_label,p)
elif p['duration']>=min_dur_human:
    p['duration']=min_dur_human
    print('Choosing first '+str(p['duration'])+'min data from each human sub')
    Control_AR_human,Control_sleep_label     = load.length_sel(Control_AR_human,Control_sleep_label,p)
    Tbi_AR_human,Tbi_sleep_label             = load.length_sel(Tbi_AR_human,Tbi_sleep_label,p)

#FILTERING
print('Filtering human epochs in '+p['freq_band']+' range ............')
Control_filtered_human                       = load.freq_filt(Control_AR_human,p,'Control')
Tbi_filtered_human                           = load.freq_filt(Tbi_AR_human,p,'Tbi')
print()

#FILTER THE AMOUNT OF SUBJECTS
control_remove = ['102', '208', '457', '495', '556', '563', '744']
for subject in control_remove:
    Control_filtered_human.pop(subject)
    Control_human.remove(subject)
tbi_remove = ['244', '340', '399', '424', '488', '510', '670', 'XVZ2FYATE8W7FI6', 'XVZ2FYATE8Z362L', 'XVZ2FFAG885GFUG']
for subject in tbi_remove:
    Tbi_filtered_human.pop(subject)
    Tbi_human.remove(subject)


i=1

#--------------------------------PREPROCESSING FOR THE DEEP LEARNING MODELS---------------------------
preprocess = Preprocessing_For_DL_Models()
preprocess.control_format_to_numOfSamples_and_Signals(Control_filtered_human, Control_human)
preprocess.tbi_format_to_numOfSamples_and_Signals(Tbi_filtered_human, Tbi_human)
# X_train, X_test, Y_train, Y_test = preprocess.train_test_split(sfreq=200, lowpass=50,epoch_len=4, overlap=.9,
#                             IV_control='XVZ2FYATE8M0SSF', IV_tbi='XVZ2FYAQH8WVIUC', parent_folder=os.path.abspath('.')+'/data_preprocessed')
preprocess.form_into_two_datasets(sfreq=200, lowpass=50,epoch_len=4, overlap=.9,parent_folder=os.path.abspath('.')+'/data_preprocessed')

#--stop here if you want to train on the deep learning models----
i=1


#--------------------------------------FEATURES----------------------------------------------------

#CALCULATING THE FEATURES

#spectral features
print('Calculating spectral features human.........')
Control_BP_human,Control_RP_human,Control_SF_human,Control_spec_ent_human,Control_FAA_human,Control_PAC_human  = load.feature_human_spectral(Control_filtered_human,p,'Control')
Tbi_BP_human,Tbi_RP_human,Tbi_SF_human,Tbi_spec_ent_human,Tbi_FAA_human,Tbi_PAC_human      =    load.feature_human_spectral(Tbi_filtered_human,p,'Tbi')
print()

#connectivity features
print('Calculating connectivity features human.........')
Control_coh_human,Control_filtered_data_human,Control_phase_syn_human,freq_band  =  load.feature_human_connectivity(Control_filtered_human,p,'Control')
Tbi_coh_human,Tbi_filtered_data_human,Tbi_phase_syn_human,freq_band      =    load.feature_human_connectivity(Tbi_filtered_human,p,'Tbi')
print()

#non-linear features
print('Calculating Non-linear features human...........')
Activity_Control_human,Mobility_Control_human,Complexity_Control_human  =    load.feature_human_nonlin(Control_filtered_data_human,p,freq_band)
Activity_Tbi_human,Mobility_Tbi_human,Complexity_Tbi_human     =    load.feature_human_nonlin(Tbi_filtered_data_human,p,freq_band)
print()


#CREATING HUMAN DATAFRAME
print('Creating human dataframe.........................')
Control_frame_human  = load.create_dataframe_human(Control_BP_human,Control_RP_human,Control_SF_human,Control_spec_ent_human,Control_FAA_human,Control_PAC_human,Control_coh_human,Control_phase_syn_human,Activity_Control_human,Mobility_Control_human,Complexity_Control_human,Control_human,freq_band,p)
Tbi_frame_human     =  load.create_dataframe_human(Tbi_BP_human,Tbi_RP_human,Tbi_SF_human,Tbi_spec_ent_human,Tbi_FAA_human,Tbi_PAC_human,Tbi_coh_human,Tbi_phase_syn_human,Activity_Tbi_human,Mobility_Tbi_human,Complexity_Tbi_human,Tbi_human,freq_band,p)
print()


#AGE REGRESSION
Total_control_dataframe_human = pd.DataFrame()

for i in (Control_frame_human):
    Total_control_dataframe_human          = Total_control_dataframe_human.append(Control_frame_human[i], ignore_index=True)
Total_control_dataframe_human              = Total_control_dataframe_human.drop('sleep_label', axis=1)
Total_control_dataframe_human['Tbi_label'] = Total_control_dataframe_human['Tbi_label'].replace(str(0),0)

Total_tbi_dataframe_human = pd.DataFrame()

for i in (Tbi_frame_human):
    Total_tbi_dataframe_human              = Total_tbi_dataframe_human.append(Tbi_frame_human[i], ignore_index=True)
Total_tbi_dataframe_human                  = Total_tbi_dataframe_human.drop('sleep_label', axis=1)
Total_tbi_dataframe_human['Tbi_label']     = Total_tbi_dataframe_human['Tbi_label'].replace(str(0),1)

if age_regg == 'Y':
    print('Perform age regression human..................')
    Total_control_dataframe_human, slop_human, yintercept_human = load.age_reg_control_human(Total_control_dataframe_human)
    Total_tbi_dataframe_human = load.age_reg_tbi_human(Total_tbi_dataframe_human, slop_human, yintercept_human)
    print()

Total_dataframe_human = Total_control_dataframe_human.append(Total_tbi_dataframe_human, ignore_index=True)


# Random Sampling
p['max_features']=int(np.sqrt(len(Total_dataframe_human)))
print("No. of features = %d"%(p['max_features']))

Random=shuffle(Total_dataframe_human)

#z-score normalization
Random, col_mean_test, col_std_test = load.MR_control(Random)

Random_train_fea = Random.iloc[:,0:-1]
Random_Y         = Random.iloc[:,-1]

#feature selection
Random_training,random_selected_fea = load.fea_sel(Random_train_fea,Random_Y,p)


#MODEL TRAINING -- CROSS VALIDATION

#decision tree classifier
dtree = DecisionTreeClassifier()
random_dtree_scores = cross_val_score(dtree, Random_training, Random_Y, cv=10)

#knn classifier: 5
classifier = KNeighborsClassifier(n_neighbors=5)
random_k1_scores = cross_val_score(classifier, Random_training, Random_Y, cv=10)

#knn classifier: 11
classifier = KNeighborsClassifier(n_neighbors=11)
random_k2_scores = cross_val_score(classifier, Random_training, Random_Y, cv=10)

#knn classifier: 19
classifier = KNeighborsClassifier(n_neighbors=19)
random_k3_scores = cross_val_score(classifier, Random_training, Random_Y, cv=10)

#mlp classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,10),max_iter=10000)
random_mlp_scores = cross_val_score(mlp, Random_training, Random_Y, cv=10)

#random forest classifier
clf = RandomForestClassifier()
random_rf_scores = cross_val_score(clf, Random_training, Random_Y, cv=10)

#support vector machine classifier
svclassifier = SVC(kernel='rbf')
random_svc_scores = cross_val_score(svclassifier, Random_training, Random_Y, cv=10)

#xgboost classifier
model = XGBClassifier(verbosity=0)
random_XGB_scores = cross_val_score(model, Random_training, Random_Y, cv=10)

print('=======================================================================')
print('Random Sampling 10 fold CV')
print('RS dtree accuracy %0.2f (+/-%0.2f)' % (np.mean(random_dtree_scores)*100,np.std(random_dtree_scores)*100))
print('RS k5 accuracy %0.2f (+/-%0.2f)' % (np.mean(random_k1_scores)*100,np.std(random_k1_scores)*100))
print('RS k11 accuracy %0.2f (+/-%0.2f)' % (np.mean(random_k2_scores)*100,np.std(random_k2_scores)*100))
print('RS k19 accuracy %0.2f (+/-%0.2f)' % (np.mean(random_k3_scores)*100,np.std(random_k3_scores)*100))
print('RS NN accuracy %0.2f (+/-%0.2f)' % (np.mean(random_mlp_scores)*100,np.std(random_mlp_scores)*100))
print('RS RF accuracy %0.2f (+/-%0.2f)' % (np.mean(random_rf_scores)*100,np.std(random_rf_scores)*100))
print('RS SVM accuracy %0.2f (+/-%0.2f)' % (np.mean(random_svc_scores)*100,np.std(random_svc_scores)*100))
print('RS XGBoost accuracy %0.2f (+/-%0.2f)' % (np.mean(random_XGB_scores)*100,np.std(random_XGB_scores)*100))
print('=======================================================================')



# Scaled['label']=Y
sub = {}
for i in range(len(Control_human)):
    sub[Control_human[i]] = Total_dataframe_human.iloc[
                            i * p['duration'] * int(60 / p['epoch_len']):i * p['duration'] * int(60 / p['epoch_len']) +
                                                                         p['duration'] * int(60 / p['epoch_len'])]
for j in range(len(Tbi_human)):
    sub[Tbi_human[j]] = Total_dataframe_human.iloc[
                        (i + j + 1) * p['duration'] * int(60 / p['epoch_len']):(i + j + 1) * p['duration'] * int(
                            60 / p['epoch_len']) + p['duration'] * int(60 / p['epoch_len'])]

# %%
# Individual validation
print('Individual validation (Leave 2 out)')
num_control_human = len(Control_human)
num_tbi_human = len(Tbi_human)

count1 = 0
count_fea = {}
for i in Total_dataframe_human.columns[:-1]:
    count_fea[i] = 0

z = 0


num_iter = 1

dtree = np.zeros((6, num_iter))
k1 = np.zeros((6, num_iter))
k2 = np.zeros((6, num_iter))
k3 = np.zeros((6, num_iter))
rf = np.zeros((6, num_iter))
nn = np.zeros((6, num_iter))
svecm = np.zeros((6, num_iter))
xgb = np.zeros((6, num_iter))

dtree2 = np.zeros((6, num_iter))
k12 = np.zeros((6, num_iter))
k22 = np.zeros((6, num_iter))
k32 = np.zeros((6, num_iter))
rf2 = np.zeros((6, num_iter))
nn2 = np.zeros((6, num_iter))
svecm2 = np.zeros((6, num_iter))
xgb2 = np.zeros((6, num_iter))

dtree3 = np.zeros((6, num_iter))
k13 = np.zeros((6, num_iter))
k23 = np.zeros((6, num_iter))
k33 = np.zeros((6, num_iter))
rf3 = np.zeros((6, num_iter))
nn3 = np.zeros((6, num_iter))
svecm3 = np.zeros((6, num_iter))
xgb3 = np.zeros((6, num_iter))

selected_fea = {}

x = np.random.randint(0, num_control_human, num_iter)
y = np.random.randint(0, num_tbi_human, num_iter)


for a in range(num_iter):
    i = x[a]
    j = y[a]

    Training_Control_human = []
    Training_Tbi_human = []
    Testing_Control_human = []
    Testing_Tbi_human = []
    Training_Control_human = [x for x in Control_human if x != Control_human[i]]
    Training_Tbi_human = [x for x in Tbi_human if x != Tbi_human[j]]

    Testing_Control_human.append(Control_human[i])
    Testing_Tbi_human.append(Tbi_human[j])

    Testing = sub[Testing_Control_human[0]].append(sub[Testing_Tbi_human[0]], ignore_index=True)

    Training = pd.DataFrame()

    for k in (Training_Control_human):
        Training = Training.append(sub[k], ignore_index=True)

    for k in (Training_Tbi_human):
        Training = Training.append(sub[k], ignore_index=True)

    Training, col_mean_test, col_std_test = load.MR_control(Training)
    Testing = load.MR_tbi(Testing, col_mean_test, col_std_test)

    #    print('-----------------------------------------------------------------------')
    print(a)

    train_fea = Training.iloc[:, 0:-1]
    Y = Training.iloc[:, -1]

    training, selected_fea[Testing_Control_human[0], Testing_Tbi_human[0]] = load.fea_sel(train_fea, Y, p)
    # training['label'] = Y
    #------ changed the label to tbi_label for newer versions -- debug 02_13_2022
    training['Tbi_label'] = Y

    #    print(selected_fea[Testing_Control_human[0],Testing_Tbi_human[0]])
    for g in selected_fea[Testing_Control_human[0], Testing_Tbi_human[0]]:
        count1 = count_fea[g]
        count_fea[g] = count1 + 1

    testing = Testing[training.columns]

    #USING MACHINE LEARNING CLASSIFIERS
    models = Models()
    dtree[:, z], k1[:, z], k1_val, k2[:, z], k2_val, k3[:, z], k3_val, rf[:, z], nn[:, z], svecm[:, z], xgb[:,
                                                                                                        z] = models.ML_Classifier(
        training, testing)
    training_coral = load.coral_main(testing, training)
    dtree3[:, z], k13[:, z], k1_val, k23[:, z], k2_val, k33[:, z], k3_val, rf3[:, z], nn3[:, z], svecm3[:, z], xgb3[:,
                                                                                                               z] = models.ML_Classifier(
        training_coral, testing)

    Dt_sham3, Dt_tbi3 = load.la3(training, testing, Training_Control_human, Testing_Control_human, Training_Tbi_human, Testing_Tbi_human, p, 3)
    new_training_human = Dt_sham3.append(Dt_tbi3, ignore_index=True)
    dtree2[:, z], k12[:, z], k1_val, k22[:, z], k2_val, k32[:, z], k3_val, rf2[:, z], nn2[:, z], svecm2[:, z], xgb2[:,
                                                                                                               z] = models.ML_Classifier(
        new_training_human, testing)

    z = z + 1

i=1
print("------------------------------------------------------------------------")
print("Over all Dtree Accuracy: %0.2f" % (np.mean(dtree[0,:])))
print("Over all RF Accuracy: %0.2f " %(np.mean(rf[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f" % (k1_val,np.mean(k1[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f " %(k2_val,np.mean(k2[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f" %(k3_val,np.mean(k3[0,:])))
print("Over all SVM Accuracy: %0.2f" %(np.mean(svecm[0,:])))
print("Over all NN Accuracy: %0.2f " %(np.mean(nn[0,:])))
print("Over all XGB Accuracy: %0.2f " %(np.mean(xgb[0,:])))
print("===========================================================")

print("------------------------------------------------------------------------")
print("Over all Dtree Accuracy: %0.2f" % (np.mean(dtree2[0,:])))
print("Over all RF Accuracy: %0.2f " %(np.mean(rf2[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f" % (k1_val,np.mean(k12[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f " %(k2_val,np.mean(k22[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f" %(k3_val,np.mean(k32[0,:])))
print("Over all SVM Accuracy: %0.2f" %(np.mean(svecm2[0,:])))
print("Over all NN Accuracy: %0.2f " %(np.mean(nn2[0,:])))
print("Over all XGB Accuracy: %0.2f " %(np.mean(xgb2[0,:])))
print("===========================================================")

print("------------------------------------------------------------------------")
print("Over all Dtree Accuracy: %0.2f" % (np.mean(dtree3[0,:])))
print("Over all RF Accuracy: %0.2f " %(np.mean(rf3[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f" % (k1_val,np.mean(k13[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f " %(k2_val,np.mean(k23[0,:])))
print("Over all KNN k= %d Accuracy: %0.2f" %(k3_val,np.mean(k33[0,:])))
print("Over all SVM Accuracy: %0.2f" %(np.mean(svecm3[0,:])))
print("Over all NN Accuracy: %0.2f " %(np.mean(nn3[0,:])))
print("Over all XGB Accuracy: %0.2f " %(np.mean(xgb3[0,:])))
print("===========================================================")








print('END')
t1_stop = process_time()
print ((t1_stop-t1_start)/60, "min")\

i=1

