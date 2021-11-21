""" Apply binary classification with ANN """

# imports
import pandas as pd
import itertools as itl
import numpy as np
from os import listdir, path
from sklearn import mixture
from graph import graphCurve
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot as plt


# run one or multiple NN configuration(s)
def classify(df, k_range, num_init, random_state):
    # run for all k, from greater than 2
    clusters_list = []
    aic = []
    bic = []
    for k_value in k_range:
        # gaussin mixture model
        gmm = mixture.GaussianMixture(n_components=k_value, n_init=num_init, random_state=random_state, init_params='random', covariance_type='diag')
        # fit
        gmm.fit(df)
        # predict
        clusters_list.append( gmm.predict(df).tolist() )
        # metrics
        aic.append( gmm.aic(df) )
        bic.append( gmm.bic(df) )
    # return
    return clusters_list, aic, bic