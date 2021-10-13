"""
Preprocessing definitions.
"""
import numpy as np


def preprocess(X_train, X_test, preproc_type='std'):
    """This function defines multiples preprocessing procedures."""
    X_train_pp, X_test_pp = np.zeros_like(X_train), np.zeros_like(X_test)
    if preproc_type == 'scaling':
        X_train_pp = np.nan_to_num(X_train,nan=-750)/10000
        X_test_pp = np.nan_to_num(X_test,nan=-750)/10000

    elif preproc_type == 'std':
        mu = np.nanmean(X_train)
        sigma = np.nanstd(X_train)
        X_train_pp = (X_train-mu)/sigma
        mu_adj = np.nanmean(X_train_pp)
        X_train_pp[np.isnan(X_train_pp)] = mu_adj

        X_test_pp = (X_test-mu)/sigma
        X_test_pp[np.isnan(X_test_pp)] = mu_adj

    elif preproc_type == 'max-min':
        delta = np.nanmax(X_train)-np.nanmin(X_train)
        min_val = np.nanmin(X_train)
        X_train_pp = (X_train-min_val)/delta
        mu_adj = np.nanmin(X_train_pp) #?
        X_train_pp[np.isnan(X_train_pp)] = mu_adj

        X_test_pp = (X_test-min_val)/delta
        X_test_pp[np.isnan(X_test_pp)] = mu_adj

    return X_train_pp, X_test_pp
