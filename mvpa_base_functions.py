from __future__ import division
import math 
import pandas as pd
import numpy as np
from scipy import stats

def force_binary_accuracy(tsY, prob_pred):
    nts_subj = int(0.5*prob_pred.shape[0])
    pred_Y = np.nan*np.zeros_like(tsY)
    for i in range(nts_subj):
        if prob_pred[i] > prob_pred[i+nts_subj]:
            pred_Y[i] = 1
            pred_Y[i+nts_subj] = 0
        elif prob_pred[i] < prob_pred[i+nts_subj]:
            pred_Y[i] = 0
            pred_Y[i+nts_subj] = 1
        else:# probability = 0.5, randomly assign label
            v = np.random.randint(0,2)
            pred_Y[i] = v
            pred_Y[i+nts_subj] = 1-v
            
    acc = np.mean(pred_Y==tsY)
    acc_vec = np.zeros_like(pred_Y)
    acc_vec[pred_Y==tsY] = 1
    return acc, acc_vec

def permute_Y(Y):
    rY = Y.copy()
    rvals = np.random.uniform(0,1,Y.shape[0])
    hfsamp = int(0.5*Y.shape[0])
    for i in range(hfsamp):
        if rvals[i]>0.5:
            t = rY[i]
            tt = rY[i+hfsamp]
            rY[i] = tt
            rY[i+hfsamp] = t
    return rY


def permute_group(group):
    rgroup = group.copy()
    idx = np.random.permutation(int(0.5*rgroup.shape[0]))
    rgroup[:int(0.5*rgroup.shape[0])] = idx
    rgroup[int(0.5*rgroup.shape[0]):] = idx
    return rgroup

def weight_transform(X, coef_vals):
    data = X
    weights = coef_vals.T
    n_samples, n_dim = data.shape
    Nbricks = 100  
    scale_param = np.cov(np.matmul(weights.T, data.T))
    pattern_unscaled = np.zeros((n_dim,1));

    data = data - np.mean(data, axis=0, keepdims=True)

    Nbricks = min(Nbricks,n_dim);
    brick_n = np.floor(n_dim / Nbricks);


    #Randomly assign TRs into one of Nfolds
    for n in range(Nbricks):
#         print(n)
        #Select id group
        if n<Nbricks-1:
            ids = np.arange(n*brick_n, (n+1)*brick_n)
        elif n==Nbricks-1:
            ids = np.arange(n*brick_n, n_dim)
        ids = ids.astype(int)
        #Generate unscaled forward models values
        val = data[:,ids]
        data_cov = np.matmul(val.T, data) / (n_samples-1)
        t = np.matmul(data_cov, weights)
        pattern_unscaled[ids] = t

    pattern = pattern_unscaled / scale_param; # like cov(X)*W * inv(W'*X')
    return pattern