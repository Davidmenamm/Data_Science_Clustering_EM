""" Reading and Preprocessing Actions"""

# imports
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import manifold as mf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler



"""
@param path, location of input dataset file
@param balance, if class balancing is applied to the read dataset
returns the the features and target of the dataset """
def readDataset(inPath):
    # read csv
    df_raw = pd.read_csv(inPath)
    # # replace empty values with column average
    fill_mean = lambda col : col.fillna(col.mean())
    df_filled = df_raw.apply(fill_mean, axis=0) # axis 0 each column
    # normalize
    df_normalize = np.nan_to_num(minMaxNormalization(df_filled))
    # return
    return df_normalize



# ordering based on the best auc
def formatTable(data, ordered):
    # get values
    table = []
    idx_auc = 0
    for topology in data:
        for config in topology:
            row = []
            # add model info
            row.extend( [str(info) for info in config['model']] )
            # add results
            row.append(str(config['acc_avg']))
            row.append(str(config['acc_std']))
            row.append(str(config['prec_avg']))
            row.append(str(config['prec_std']))
            row.append(str(config['rec_avg']))
            row.append(str(config['rec_std']))
            row.append(str(config['auc_avg']))
            row.append(str(config['auc_std']))
            row.append(str(config['loss_avg']))
            row.append(str(config['loss_std']))
            # index where auc_avg is located among all lists
            if idx_auc == 0:
                idx_auc = row.index(str(config['auc_avg']))
            # append to table data
            table.append(row)
    # order table
    if ordered:
        table = sorted(table, key= lambda item : item[idx_auc], reverse=True)
    # return
    return table



# get top info
def getTopInfo(topConfig, data):
    # get classification vectors
    y_pred_true = []
    y_pred_proba = []
    y_pred_categ = []
    loss_epoch = []
    # iterate topologies
    for topology in data:
        # iterate hyperparams
        for config in topology:
            # top config identifier
            topConfigId = []
            # first for elements identify each topology
            topConfigId.append(topConfig[0])
            topConfigId.append(topConfig[1])
            topConfigId.append(topConfig[2])
            topConfigId.append(topConfig[3])
            # find top model
            validate = True
            for identifier in config['model']:
                if str(identifier) not in topConfigId:
                    validate = False
            if validate:
                y_pred_true = config['y_true_sum']
                y_pred_proba = config['y_pred_proba_sum']
                y_pred_categ = config['y_pred_categ_sum']
                loss_epoch = config['loss_epoch_avg']
    # return
    return y_pred_true, y_pred_proba, y_pred_categ, loss_epoch



# additional min max normalization
def minMaxNormalization(df):
    # normalize min max
    normalized_df = (df-df.min())/(df.max()-df.min())
    print(normalized_df)
        
    # return
    return normalized_df