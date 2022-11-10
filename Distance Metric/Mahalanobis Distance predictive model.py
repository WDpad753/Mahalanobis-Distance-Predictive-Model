# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:07:29 2022

@author: Admin
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs, plot_confusion_matrix
import winsound

########################################################################################################################
############################################# DataFrame Production #####################################################
########################################################################################################################
# Used to obtain the DataSet from the folder:
damage_case_level = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4']

####### Full Datasets:
## Full Dataset
##### Individual Study:
# Study 1:
url_Case_study_1_LDA_feature_matrix = '~//Case Study 1.txt'

# Study 2:
url_Case_study_2_LDA_feature_matrix = '~//Case Study 2.txt'

########################################################################################################################

# Used to get full Feature Datasets:
def set_pandas_options() -> None:
    pd.options.display.max_columns = 500
    pd.options.display.max_rows = 500
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None
    # pd.options.display.precision = 2  # set as needed

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def cov_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")
        
def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    # for i in range(len(diff)):
    #     md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    md.append(np.sqrt(np.diag(np.linalg.multi_dot([diff, inv_covariance_matrix, diff.T]))))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    
    # # Tukeys method:
    # PDF_Q1 = np.quantile(dist, 0.25)
    # PDF_Q3 = np.quantile(dist, 0.75)
    # IQR = PDF_Q3 - PDF_Q1
    # threshold = PDF_Q3 + (1.5 * IQR)
    
    # Assuming is Chi-distributed data:    
    # threshold = np.sqrt(st.chi2.ppf((1-(k/100)), df=np.array(dist).shape[0]))    #degrees of freedom = number of variables
    
    # # Assuming is norm-distributed data:
    # k = 0.997 if extreme else 0.95
    # params = st.norm.fit(dist)
    # # Separate parts of parameters
    # arg = params[:-2]
    # loc = params[-2]
    # scale = params[-1]
    # # Get sane start and end points of distribution
    # threshold = st.norm.ppf(k, *arg, loc=loc, scale=scale) if arg else dist.ppf(k, loc=loc, scale=scale)
    # threshold = st.norm.ppf(k, loc=np.mean(dist), scale=np.sqrt(np.var(dist)))    # loc = mean of distance training, scale = std of the distance training
    # threshold = st.norm.ppf(0.997, loc=np.mean(dist), scale=np.sqrt(np.var(dist)))    # loc = mean of distance training, scale = std of the distance training
    # dist_pdf = sns.distplot(dist, bins = 10, kde= True, color = 'blue').get_lines()[0].get_data()
    # threshold = np.quantile(dist_pdf, 0.95)
    return threshold

class MahalanobisOneclassClassifier():
    def __init__(self, X_train, threshold):
        self.X_train = X_train
        self.threshold = threshold
        print('Critical value is: ', self.threshold)

    def predict_proba(self, X_test):
        mahalanobis_dist = X_test
        return mahalanobis_dist

    def predict(self, X_test):
        # predict_lst = []
        dist = self.predict_proba(X_test)
        dist = dist.transpose()
        dist = dist.to_numpy()
        # print(dist[0])
        predict_lst = [int(dist_val >= self.threshold) for dist_val in dist[0]]
        # for i in range(len(dist)):
        #     if dist[i] >= threshold:
        #         predict_lst.append(int(dist[i]))
        return predict_lst

average_msd_accuracy = []
start_code = time.time()

for k in range(0, 2, 1):
########################################################################################################################
################################################# Study 1 #############################################################
########################################################################################################################
    if k == 0:
        ### Full Dataset:
        dataset = np.loadtxt(fname=url_Case_study_1_LDA_feature_matrix, skiprows=1, usecols=range(3))
        dataset_df = pd.DataFrame(dataset)
        dataset_df = dataset_df.iloc[: , 1:6]
        dataset = dataset_df.to_numpy()
        print(dataset)
        dataset_shape = dataset.shape
        print(dataset_shape)
        data_df = pd.DataFrame(dataset)
        data = data_df.assign(Label=damage_case_level)
        print(data)
        # Separating LDA Damage cases:
        Case_0_LDA_dataframe = data.loc[data['Label'] == '0']
        Case_1_LDA_dataframe = data.loc[data['Label'] == '1']
        Case_2_LDA_dataframe = data.loc[data['Label'] == '2']
        Case_3_LDA_dataframe = data.loc[data['Label'] == '3']
        Case_4_LDA_dataframe = data.loc[data['Label'] == '4']
        frames = [Case_0_LDA_dataframe, Case_1_LDA_dataframe, Case_2_LDA_dataframe, Case_3_LDA_dataframe, Case_4_LDA_dataframe]
        dataframe_SDY_1 = pd.concat(frames)        
        ## Training dataframe:
        Case_0_LDA_dataframe_train = Case_0_LDA_dataframe.iloc[:7]
        Case_1_LDA_dataframe_train = Case_1_LDA_dataframe.iloc[:7]
        Case_2_LDA_dataframe_train = Case_2_LDA_dataframe.iloc[:7]
        Case_3_LDA_dataframe_train = Case_3_LDA_dataframe.iloc[:7]
        Case_4_LDA_dataframe_train = Case_4_LDA_dataframe.iloc[:7]
        frames_train = [Case_0_LDA_dataframe_train, Case_1_LDA_dataframe_train, Case_2_LDA_dataframe_train, Case_3_LDA_dataframe_train, Case_4_LDA_dataframe_train]
        dataframe_SDY_1_train = pd.concat(frames_train)
        ## Testing dataframe:
        Case_0_LDA_dataframe_test = Case_0_LDA_dataframe.iloc[-3:]
        Case_1_LDA_dataframe_test = Case_1_LDA_dataframe.iloc[-3:]
        Case_2_LDA_dataframe_test = Case_2_LDA_dataframe.iloc[-3:]
        Case_3_LDA_dataframe_test = Case_3_LDA_dataframe.iloc[-3:]
        Case_4_LDA_dataframe_test = Case_4_LDA_dataframe.iloc[-3:]
        frames_test = [Case_0_LDA_dataframe_test, Case_1_LDA_dataframe_test, Case_2_LDA_dataframe_test, Case_3_LDA_dataframe_test, Case_4_LDA_dataframe_test]
        dataframe_SDY_1_test = pd.concat(frames_test)
        dataframe = dataframe_SDY_1
        dataframe_train = dataframe_SDY_1_train
        dataframe_test = dataframe_SDY_1_test
        dataframe_train.reset_index(drop=True, inplace=True)
        dataframe_test.reset_index(drop=True, inplace=True)
        Study_no = 1
########################################################################################################################

########################################################################################################################
################################################# Study 2 #############################################################
########################################################################################################################
    elif k == 1:
        ### Full Dataset:
        dataset = np.loadtxt(fname=url_Case_study_2_LDA_feature_matrix, skiprows=1, usecols=range(3))
        dataset_df = pd.DataFrame(dataset)
        dataset_df = dataset_df.iloc[: , 1:6]
        dataset = dataset_df.to_numpy()
        print(dataset)
        dataset_shape = dataset.shape
        print(dataset_shape)
        data_df = pd.DataFrame(dataset)
        data = data_df.assign(Label=damage_case_level)
        # Separating LDA Damage cases:
        Case_0_LDA_dataframe = data.loc[data['Label'] == '0']
        Case_1_LDA_dataframe = data.loc[data['Label'] == '1']
        Case_2_LDA_dataframe = data.loc[data['Label'] == '2']
        Case_3_LDA_dataframe = data.loc[data['Label'] == '3']
        Case_4_LDA_dataframe = data.loc[data['Label'] == '4']
        frames = [Case_0_LDA_dataframe, Case_1_LDA_dataframe, Case_2_LDA_dataframe, Case_3_LDA_dataframe, Case_4_LDA_dataframe]
        dataframe_SDY_2 = pd.concat(frames)        
        ## Training dataframe:
        Case_0_LDA_dataframe_train = Case_0_LDA_dataframe.iloc[:7]
        Case_1_LDA_dataframe_train = Case_1_LDA_dataframe.iloc[:7]
        Case_2_LDA_dataframe_train = Case_2_LDA_dataframe.iloc[:7]
        Case_3_LDA_dataframe_train = Case_3_LDA_dataframe.iloc[:7]
        Case_4_LDA_dataframe_train = Case_4_LDA_dataframe.iloc[:7]
        frames_train = [Case_0_LDA_dataframe_train, Case_1_LDA_dataframe_train, Case_2_LDA_dataframe_train, Case_3_LDA_dataframe_train, Case_4_LDA_dataframe_train]
        dataframe_SDY_2_train = pd.concat(frames_train)
        ## Testing dataframe:
        Case_0_LDA_dataframe_test = Case_0_LDA_dataframe.iloc[-3:]
        Case_1_LDA_dataframe_test = Case_1_LDA_dataframe.iloc[-3:]
        Case_2_LDA_dataframe_test = Case_2_LDA_dataframe.iloc[-3:]
        Case_3_LDA_dataframe_test = Case_3_LDA_dataframe.iloc[-3:]
        Case_4_LDA_dataframe_test = Case_4_LDA_dataframe.iloc[-3:]
        frames_test = [Case_0_LDA_dataframe_test, Case_1_LDA_dataframe_test, Case_2_LDA_dataframe_test, Case_3_LDA_dataframe_test, Case_4_LDA_dataframe_test]
        dataframe_SDY_2_test = pd.concat(frames_test)
        dataframe = dataframe_SDY_2
        dataframe_train = dataframe_SDY_2_train
        dataframe_test = dataframe_SDY_2_test
        dataframe_train.reset_index(drop=True, inplace=True)
        dataframe_test.reset_index(drop=True, inplace=True)
        Study_no = 2
########################################################################################################################
    
################################################# All Study ##########################################################
    
    dataframe.fillna(value=-99999, inplace=True)
    dataframe_train.fillna(value=-99999, inplace=True)
    dataframe_test.fillna(value=-99999, inplace=True)
    
    # Used to obtain the dataset:
    set_pandas_options()
    print('Full DataFrame:')
    print(dataframe)
    
    print('Training DataFrame:')
    print(dataframe_train)
    
    print('Testing DataFrame:')
    print(dataframe_test)
    
    features = [0, 1]
    full_features = [0, 1]
    labels = ['Label']

    ######## Anomaly Detection:
    ## Using Mahalanobis distance metric:
    # Inputting the damage cases:
    Case_0_LDA_dataframe = Case_0_LDA_dataframe.drop(['Label'], axis=1)  # df.columns is zero-based pd.Index
    Case_1_LDA_dataframe = Case_1_LDA_dataframe.drop(['Label'], axis=1)  # df.columns is zero-based pd.Index
    Case_2_LDA_dataframe = Case_2_LDA_dataframe.drop(['Label'], axis=1)  # df.columns is zero-based pd.Index
    Case_3_LDA_dataframe = Case_3_LDA_dataframe.drop(['Label'], axis=1)  # df.columns is zero-based pd.Index
    Case_4_LDA_dataframe = Case_4_LDA_dataframe.drop(['Label'], axis=1)  # df.columns is zero-based pd.Index
        
    data_train = np.array(Case_0_LDA_dataframe.values)
    data_test_C1 = np.array(Case_1_LDA_dataframe.values)
    data_test_C2 = np.array(Case_2_LDA_dataframe.values)
    data_test_C3 = np.array(Case_3_LDA_dataframe.values)
    data_test_C4 = np.array(Case_4_LDA_dataframe.values)
    
    data_train_df = pd.DataFrame(Case_0_LDA_dataframe.values)
    data_test_df_C1 =  pd.DataFrame(Case_1_LDA_dataframe.values)
    data_test_df_C2 =  pd.DataFrame(Case_2_LDA_dataframe.values)
    data_test_df_C3 =  pd.DataFrame(Case_3_LDA_dataframe.values)
    data_test_df_C4 =  pd.DataFrame(Case_4_LDA_dataframe.values)
    
    # Calculating the covariance matrix:
    covar_matrix, inv_covar_matrix = cov_matrix(data=data_train)
    
    # Calculating the mean value for the input variables:
    mean_distr = data_train_df.mean(axis=0)
    # rob_cov = MinCovDet(random_state=0).fit(data_train_df)
    # robust_mean = rob_cov.location_  #robust mean
    # mean_distr = robust_mean
    
    # Calculating the Mahalanobis distance and threshold value to flag datapoints as an anomaly:
    dist_test_C1 = MahalanobisDist(inv_covar_matrix, mean_distr, data_test_df_C1, verbose=True)
    dist_test_C2 = MahalanobisDist(inv_covar_matrix, mean_distr, data_test_df_C2, verbose=True)
    dist_test_C3 = MahalanobisDist(inv_covar_matrix, mean_distr, data_test_df_C3, verbose=True)
    dist_test_C4 = MahalanobisDist(inv_covar_matrix, mean_distr, data_test_df_C4, verbose=True)
    dist_train = MahalanobisDist(inv_covar_matrix, mean_distr, data_train_df, verbose=True)
    threshold = MD_threshold(dist_train, extreme = False)
    
    # Distribution of Threshold value for flagging an anomaly:
    plt.figure()
    sns.distplot(np.square(dist_train),bins = 10, kde= False)
    # plt.xlim([0.0,15])
    plt.show()
    
    plt.figure()
    sns.distplot(dist_train, bins = 10, kde= True, color = 'green');
    # plt.xlim([0.0,5])
    plt.xlabel('Mahalanobis dist')
    plt.show()
    
    anomaly_train = pd.DataFrame(index=data_train_df.index)
    anomaly_train['Mob_dist']= dist_train[0]
    anomaly_train['Thresh'] = threshold
    # If Mob_dist above threshold: Flag as anomaly
    anomaly_train['Anomaly'] = anomaly_train['Mob_dist'] > anomaly_train['Thresh']
    anomaly_train['Case'] = 'C0'
    anomaly_train.index = data_train_df.index
    
    anomaly_C1 = pd.DataFrame(index=data_test_df_C1.index)
    anomaly_C1['Mob_dist']= dist_test_C1[0]
    anomaly_C1['Thresh'] = threshold
    # If Mob_dist above threshold: Flag as anomaly
    anomaly_C1['Anomaly'] = anomaly_C1['Mob_dist'] > anomaly_C1['Thresh']
    anomaly_C1['Case'] = 'C1'
    anomaly_C1.index = data_test_df_C1.index
    anomaly_C1.head()
    
    anomaly_C2 = pd.DataFrame(index=data_test_df_C2.index)
    anomaly_C2['Mob_dist']= dist_test_C2[0]
    anomaly_C2['Thresh'] = threshold
    # If Mob_dist above threshold: Flag as anomaly
    anomaly_C2['Anomaly'] = anomaly_C2['Mob_dist'] > anomaly_C2['Thresh']
    anomaly_C2['Case'] = 'C2'
    anomaly_C2.index = data_test_df_C2.index
    anomaly_C2.head()
    
    anomaly_C3 = pd.DataFrame(index=data_test_df_C3.index)
    anomaly_C3['Mob_dist']= dist_test_C3[0]
    anomaly_C3['Thresh'] = threshold
    # If Mob_dist above threshold: Flag as anomaly
    anomaly_C3['Anomaly'] = anomaly_C3['Mob_dist'] > anomaly_C3['Thresh']
    anomaly_C3['Case'] = 'C3'
    anomaly_C3.index = data_test_df_C3.index
    anomaly_C3.head()
    
    anomaly_C4 = pd.DataFrame(index=data_test_df_C4.index)
    anomaly_C4['Mob_dist']= dist_test_C4[0]
    anomaly_C4['Thresh'] = threshold
    # If Mob_dist above threshold: Flag as anomaly
    anomaly_C4['Anomaly'] = anomaly_C4['Mob_dist'] > anomaly_C4['Thresh']
    anomaly_C4['Case'] = 'C4'
    anomaly_C4.index = data_test_df_C4.index
    anomaly_C4.head()
    
    final_scored_md = pd.concat([anomaly_train, anomaly_C1, anomaly_C2, anomaly_C3, anomaly_C4], ignore_index=True)
    # final_scored_md = pd.concat([anomaly_train, anomaly_C1, anomaly_C2, anomaly_C3, anomaly_C4])
    print(final_scored_md)
    
    # Plotting the observation vs Mahalanobis distance:
    final_scored_len = final_scored_md.shape
    obser = np.arange(1, final_scored_len[0]+1)
    dfc = final_scored_md.query('Mob_dist > Thresh')
    # obser_dfc = np.arange(dfc.shape)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Observation', fontsize = 15)
    ax.set_ylabel('Mahalanobis distance', fontsize = 15)
    ax.set_title('Mahalanobis distance plot of Example {0}'.format(Study_no), fontsize = 20)
    targets = ['C0', 'C1', 'C2', 'C3', 'C4']
    colors = ['blue', 'yellow', 'green', 'cyan', 'purple']
    final_scored_md_gp = final_scored_md.groupby("Case")
    for name, group in final_scored_md_gp:
        ax.scatter(group.index, group['Mob_dist'], s = 50, label=name)
    ax.axhline(y=threshold, color='k', linestyle='--')
    ax.scatter(dfc.index, dfc['Mob_dist'], c = 'red', s = 50, label='Anomaly')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()
    
    
    ######## Classification part:
    # Separating MSD Damage cases:
    Case_0_MSD = final_scored_md.loc[final_scored_md['Case'] == 'C0']
    Case_1_MSD = final_scored_md.loc[final_scored_md['Case'] == 'C1']
    Case_2_MSD = final_scored_md.loc[final_scored_md['Case'] == 'C2']
    Case_3_MSD = final_scored_md.loc[final_scored_md['Case'] == 'C3']
    Case_4_MSD = final_scored_md.loc[final_scored_md['Case'] == 'C4']
    
    ## Training dataframe:
    # Case_0_MSD_train = Case_0_MSD.iloc[:7]
    # Case_1_MSD_train = Case_1_MSD.iloc[:7]
    # Case_2_MSD_train = Case_2_MSD.iloc[:7]
    # Case_3_MSD_train = Case_3_MSD.iloc[:7]
    # Case_4_MSD_train = Case_4_MSD.iloc[:7]
    Case_0_MSD_train = Case_0_MSD.sample(n=7)
    Case_1_MSD_train = Case_1_MSD.sample(n=7)
    Case_2_MSD_train = Case_2_MSD.sample(n=7)
    Case_3_MSD_train = Case_3_MSD.sample(n=7)
    Case_4_MSD_train = Case_4_MSD.sample(n=7)
    frames_train = [Case_0_MSD_train, Case_1_MSD_train, Case_2_MSD_train, Case_3_MSD_train, Case_4_MSD_train]
    dataframe_train = pd.concat(frames_train)
    
    ## Testing dataframe:
    # Case_0_MSD_test = Case_0_MSD.iloc[-3:]
    # Case_1_MSD_test = Case_1_MSD.iloc[-3:]
    # Case_2_MSD_test = Case_2_MSD.iloc[-3:]
    # Case_3_MSD_test = Case_3_MSD.iloc[-3:]
    # Case_4_MSD_test = Case_4_MSD.iloc[-3:]
    Case_0_MSD_test = Case_0_MSD.sample(n=4)
    Case_1_MSD_test = Case_1_MSD.sample(n=4)
    Case_2_MSD_test = Case_2_MSD.sample(n=4)
    Case_3_MSD_test = Case_3_MSD.sample(n=4)
    Case_4_MSD_test = Case_4_MSD.sample(n=4)
    frames_test = [Case_0_MSD_test, Case_1_MSD_test, Case_2_MSD_test, Case_3_MSD_test, Case_4_MSD_test]
    dataframe_test = pd.concat(frames_test)
       
    # Keeping MSD column in training dataset:
    dataframe_train = dataframe_train.drop(['Case', 'Thresh'], axis=1)  # df.columns is zero-based pd.Index
    true_y_class_train = dataframe_train['Anomaly']
    true_y_class_train = true_y_class_train.astype(int)
    true_y_class_train = true_y_class_train.transpose()
    true_y_class_train = true_y_class_train.to_numpy()
    dataframe_train = dataframe_train.drop(['Anomaly'], axis=1)  # df.columns is zero-based pd.Index
    
    # Keeping MSD column in testing dataset:
    dataframe_test = dataframe_test.drop(['Case', 'Thresh'], axis=1)  # df.columns is zero-based pd.Index
    true_y_class_test = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    clf = MahalanobisOneclassClassifier(dataframe_train, threshold)
    mahalanobis_dist = clf.predict_proba(dataframe_test)
    pred_mahalanobis_dist_class = clf.predict(dataframe_test)
    print(mahalanobis_dist)
    print(pred_mahalanobis_dist_class)
    
    # Pred and Truth
    test_acc = accuracy_score(true_y_class_test, pred_mahalanobis_dist_class) * 100
    print('The test set accuracy is %4.2f%%' % test_acc)
    
    # Obtaining the report of the model:
    print('Report of MSD: ')
    print(classification_report(y_true=true_y_class_test, y_pred=pred_mahalanobis_dist_class))
    
    targets = ['0', '1']
    
    cnf_matrix = confusion_matrix(y_true=true_y_class_test, y_pred=pred_mahalanobis_dist_class)
    
    print('Confusion Matrix of MSD: ')
    print(cnf_matrix)
    
    # Obtaining number of labels:
    labels = list(set(true_y_class_test))
    labels.sort()
    print("Total labels: %s -> %s" % (len(labels), labels))
    
    # Obtaining the dataframe of the confusion matrix:
    df_conf = pd.DataFrame(data=confusion_matrix(true_y_class_test, pred_mahalanobis_dist_class, labels=labels), columns=labels,index=labels)
    print('Confusion Matrix Dataframe:')
    print(df_conf)
    
    # Local (metrics per class) #
    tps = {}
    fps = {}
    fns = {}
    precision_local = {}
    recall_local = {}
    f1_local = {}
    accuracy_local = {}
    for label in labels:
        tps[label] = df_conf.loc[label, label]
        fps[label] = df_conf[label].sum() - tps[label]
        fns[label] = df_conf.loc[label].sum() - tps[label]
        tp, fp, fn = tps[label], fps[label], fns[label]
    
        precision_local[label] = tp / (tp + fp) if (tp + fp) > 0. else 0.
        recall_local[label] = tp / (tp + fn) if (tp + fp) > 0. else 0.
        p, r = precision_local[label], recall_local[label]
    
        f1_local[label] = 2. * p * r / (p + r) if (p + r) > 0. else 0.
        accuracy_local[label] = tp / (tp + fp + fn) if (tp + fp + fn) > 0. else 0.
    
    print('\n')
    print("#-- Local measures --#")
    print("True Positives:", tps)
    print("False Positives:", fps)
    print("False Negatives:", fns)
    print("Precision:", precision_local)
    print("Recall:", recall_local)
    print("F1-Score:", f1_local)
    print("Accuracy:", accuracy_local)
    
    # Global metrics #
    micro_averages = {}
    macro_averages = {}
    
    correct_predictions = sum(tps.values())
    den = sum(list(tps.values()) + list(fps.values()))
    micro_averages["Precision"] = 1. * correct_predictions / den if den > 0. else 0.
    
    den = sum(list(tps.values()) + list(fns.values()))
    micro_averages["Recall"] = 1. * correct_predictions / den if den > 0. else 0.
    
    micro_avg_p, micro_avg_r = micro_averages["Precision"], micro_averages["Recall"]
    micro_averages["F1-score"] = 2. * micro_avg_p * micro_avg_r / (micro_avg_p + micro_avg_r) if (micro_avg_p + micro_avg_r) > 0. else 0.
    
    macro_averages["Precision"] = np.mean(list(precision_local.values()))
    macro_averages["Recall"] = np.mean(list(recall_local.values()))
    
    macro_avg_p, macro_avg_r = macro_averages["Precision"], macro_averages["Recall"]
    macro_averages["F1-Score"] = np.mean(list(f1_local.values()))
    
    total_predictions = df_conf.values.sum()
    accuracy_global = correct_predictions / total_predictions if total_predictions > 0. else 0.
    
    print('\n')
    print("#-- Global measures --#")
    print("Micro-Averages:", micro_averages)
    print("Macro-Averages:", macro_averages)
    print("Correct predictions:", correct_predictions)
    print("Total predictions:", total_predictions)
    print("Accuracy:", accuracy_global * 100)
    
    # TN (True Negative) #
    tns = {}
    for label in set(true_y_class_test):
        tns[label] = len(true_y_class_test) - (tps[label] + fps[label] + fns[label])
    print("True Negatives:", tns)
    
    accuracy_local_new = {}
    for label in labels:
        tp, fp, fn, tn = tps[label], fps[label], fns[label], tns[label]
        accuracy_local_new[label] = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0. else 0.
    
    total_true = sum(list(tps.values()) + list(tns.values()))
    total_predictions = sum(list(tps.values()) + list(tns.values()) + list(fps.values()) + list(fns.values()))
    accuracy_global_new = 1. * total_true / total_predictions if total_predictions > 0. else 0.
    
    print("Accuracy (per class), with TNs:", accuracy_local_new)
    print("Accuracy (per class), without TNs:", accuracy_local)
    print("Accuracy (global), with TNs:", accuracy_global_new)
    print("Accuracy (global), without TNs:", accuracy_global)
    
    print('\n')
    
    fig_1, ax_1 = plot_confusion_matrix(conf_mat=cnf_matrix, colorbar=True, show_absolute=True, show_normed=False, class_names=targets)
    plt.title('Confusion matrix of MSD Model of Example {0}'.format(Study_no))
    
    fig_2, ax_2 = plot_confusion_matrix(conf_mat=cnf_matrix, colorbar=True, show_absolute=False, show_normed=True, class_names=targets)
    plt.title('Normalized MSD confusion matrix of Example {0}'.format(Study_no))
    # plt.show()
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')

print('\n')
print("[INFO] Time took to run code: {:.2f} seconds".format(time.time() - start_code))
print("[INFO] Time took to run code: {:.2f} minutes".format(round((time.time() - start_code) / 60)))
print("[INFO] Time took to run code: {:.2f} hours".format(round(((time.time() - start_code) / 60) / 12)))

########################################################################################################################
########################################################################################################################
########################################################################################################################

## Notification:
duration = 5000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
winsound.MessageBeep()