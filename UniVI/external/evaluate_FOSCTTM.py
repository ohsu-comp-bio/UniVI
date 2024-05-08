#https://bitbucket.org/noblelab/2020_mmdma_pytorch/src/master/evaluate.py

#Author: Ritambhara Singh (ritambhara@brown.edu)
#Created on: 5 Feb 2020

#Script to calculate performance scores for learnt embeddings of MMD algorithm

import numpy as np
import math
import os
import sys
import random

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

import matplotlib.cm
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calc_sil(x1_mat,x2_mat,x1_lab,x2_lab): #function to calculate Silhouette scores

    x = np.concatenate((x1_mat,x2_mat))
    lab = np.concatenate((x1_lab,x2_lab))

    sil_score = silhouette_samples(x,lab)


    avg = np.mean(sil_score)
    
    return avg

def calc_frac(x1_mat,x2_mat): #function to calculate FOSCTTM values
	nsamp = x1_mat.shape[0]
	total_count = nsamp * (nsamp -1)
	rank=0
	for row_idx in range(nsamp):
		euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
		true_nbr = euc_dist[row_idx]
		sort_euc_dist = sorted(euc_dist)
		rank+=sort_euc_dist.index(true_nbr)
	
	frac = float(rank)/total_count
	return frac
	
