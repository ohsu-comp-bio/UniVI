from typing import Union, Literal, List
from pathlib import Path, PosixPath
import numpy as np
import os
import scipy

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import anndata

from scvt.utilities._utils import Lists

def model_feature_matching(df_new, df_ref):
    assert isinstance(df_new, pd.DataFrame)
    assert isinstance(df_ref, pd.DataFrame)
    
    ''' mean vector based on df_new expression level '''
    feature_mean_vector = df_new.mean(1)
    features_common = df_new.columns.intersection(df_ref.columns).tolist()
    #features_missing = Lists.items_not_in_list(df_new.columns.tolist(), df_ref.columns.tolist())
    features_missing = Lists.items_not_in_list(df_ref.columns.tolist(), features_common)
    
    nrow = df_new.shape[0]
    ncol = df_ref.shape[1]

    mtx_none = np.array([[None for j in range(ncol)] for i in range(nrow)])
    df_out = pd.DataFrame(mtx_none, index=df_new.index, columns=df_ref.columns)

    for col_gene in df_ref.columns.tolist():
        if col_gene in df_new.columns.tolist():
            df_out[col_gene] = df_new[col_gene] 
        else:
            df_out[col_gene] = feature_mean_vector
    print(f'{len(features_common)} features are in common')
    print(f'{len(features_missing)} features are imputed by mean value')
    
    return { 'df_imputed': df_out, 'missing': features_missing, 'common': features_common}


def anndata_sanity_check(adata, key_adata_layer):
    ''' check if it is anndata.AnnData'''
    assert isinstance(adata, anndata.AnnData), "is not anndata.AnnData"
    ''' check if data layer is valid '''
    assert key_adata_layer in adata.layers.keys(), "Unknown data type"
    ''' ensure data is np.ndarray '''
    assert isinstance(adata.layers[key_adata_layer], (np.ndarray, scipy.sparse.csr.csr_matrix)),\
			"Should be np.ndarray or scipy.sparse.csr.csr_matrix"

def calc_N_train_val_test(
    n_total: int, 
    train_fraction: float, 
    val_fraction: float
):
    ''' example: calc_N_train_val_test(n_samples, 0.8, 0.1)'''
    assert 0.0 < train_fraction + val_fraction <= 1.0
    
    from sklearn.model_selection._split import _validate_shuffle_split
    n_train, n_val = _validate_shuffle_split(n_samples=n_total, 
                           train_size=train_fraction, test_size=val_fraction)
    n_test = n_total - (n_train + n_val)
    
    return (n_train, n_val, n_test)

def get_idx_train_val_test(
    n_total: int, 
    train_fraction: float, 
    val_fraction: float,
    shuffle: bool=True,
    seed: int=42
):
    n_train, n_val, n_test = calc_N_train_val_test(n_total, train_fraction, val_fraction)

    idx_whole = np.arange(n_total)
    
    if shuffle:
        idx_whole = np.random.RandomState(seed).permutation(n_total)
    
    idx_train = idx_whole[:n_train]
    idx_val = idx_whole[n_train:n_train+n_val]
    idx_test = idx_whole[n_train+n_val:]

    return (idx_train, idx_val, idx_test)

class SCDataset(Dataset):
    '''
    dataset = SCDataset(adata, 'counts')
    '''
    def __init__(
        self,
        adata: anndata.AnnData,
        key_adata_layer: Literal['counts', 'lib_normed_log1p', 'z_scaled']
    ):
        self.key_adata_layer = key_adata_layer
        anndata_sanity_check(adata, key_adata_layer)

        print(f"* SCDataset loading using {self.key_adata_layer}")
        X = adata.layers[self.key_adata_layer]
        if isinstance(X, scipy.sparse.csr.csr_matrix):
            X = X.copy().toarray()
        self.X = X

    def __getitem__(self, idx: List[int]):
        return self.X[idx]

    def __len__(self):
        return self.X.shape[0]


class SCDataClass:
    '''
    scd = SCDataClass(adata, 'z_scaled', train_fraction=0.8, val_fraction=0.1,
                                 batch_size=100, seed=42)
    '''
    
    def __init__(
        self,
        adata: anndata.AnnData, 
        key_adata_layer: Literal['counts', 'lib_normed_log1p', 'z_scaled'],
        train_fraction: float, 
        val_fraction: float,
        batch_size: int=100,
        seed: int=42    
    ):

        ''' set anndata ''' 
        self.adata = adata
        self.key_adata_layer = key_adata_layer
        
        ''' set dataset ''' 
        self.dataset = SCDataset(adata, key_adata_layer=self.key_adata_layer)
        
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.seed = seed
       
        ''' set dataloader  ''' 
        _scdl = SCDataLoader(dataset=self.dataset, 
                                train_fraction=self.train_fraction, 
                                val_fraction=self.val_fraction,
                                batch_size=self.batch_size,
                                seed=self.seed
                            )
        self.dataloaders = _scdl.get_dataloaders()
        
        ''' some attributes for train/ test/ validation ''' 
        self.idx_train = _scdl.idx_train
        self.idx_val = _scdl.idx_val
        self.idx_test = _scdl.idx_test

        self.n_train = len(self.idx_train)
        self.n_val = len(self.idx_val)
        self.n_test = len(self.idx_test)

class SCDataLoader:
    '''
    scdl = SCDataLoader(dataset=dataset, train_fraction=0.8, val_fraction=0.1)
    loaders = scdl.get_dataloaders()
    iter(loaders['train']).next()
    '''
    def __init__(
        self,
        adata: anndata.AnnData, 
        key_adata_layer: Literal['counts', 'lib_normed_log1p', 'z_scaled'],        
        train_fraction: float, 
        val_fraction: float,
        batch_size: int=100,
        seed: int=42    
    ):
    
        anndata_sanity_check(adata, key_adata_layer)
        
        '''
        print('\n')
        print(f'* Getting DataLoaders from SCDataset using key "{key_adata_layer}"')
        print(f'- train_fraction : {train_fraction}')
        print(f'- val_fraction : {val_fraction}')
        print(f'- batch_size : {batch_size}')
        print(f'- seed : {seed}')
        print('\n')
        '''

        n_total = len(adata)

        ''' return shuffled indices '''
        idx_train, idx_val, idx_test = get_idx_train_val_test(
					n_total=n_total, 
                             		train_fraction=train_fraction, 
					val_fraction=val_fraction,
                             		shuffle=True, 
					seed=seed)
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        self.batch_size = batch_size

        self.dataset_train = SCDataset(adata[idx_train].copy(), key_adata_layer)
        self.dataset_val = SCDataset(adata[idx_val].copy(), key_adata_layer)
        self.dataset_test = SCDataset(adata[idx_test].copy(), key_adata_layer)

        self.dataloaders = self.get_dataloaders()

    def get_dataloaders(self) -> dict:
        '''return dictionary for loaders'''
        
        ''' already shuffled indices, so no further shuffling '''
        return {
            'train': DataLoader(self.dataset_train, batch_size=self.batch_size, 
				shuffle=False),

            'test': DataLoader(self.dataset_test, batch_size=self.batch_size, 
				shuffle=False),

            'val': DataLoader(self.dataset_val, batch_size=self.batch_size, 
				shuffle=False)
        }

class SCPairedDataset(Dataset):
    '''
    dataset = SCPairedDataset(adata_rna, adata_adt, 'counts')
    '''
    def __init__(
        self,
        adata1: anndata.AnnData,
        adata2: anndata.AnnData,
        key_adata_layer: Literal['counts', 'lib_normed_log1p', 'z_scaled']
    ):
        self.key_adata_layer = key_adata_layer

        anndata_sanity_check(adata1, key_adata_layer)
        anndata_sanity_check(adata2, key_adata_layer)
        self._check_paired(adata1, adata2)

        #print(f"* SCPairedDataset loading using {self.key_adata_layer}")
        X1 = adata1.layers[self.key_adata_layer]
        X2 = adata2.layers[self.key_adata_layer]

        if isinstance(X1, scipy.sparse.csr.csr_matrix):
            X1 = X1.copy().toarray()
        if isinstance(X2, scipy.sparse.csr.csr_matrix):
            X2 = X2.copy().toarray()

        self.X1 = X1
        self.X2 = X2

    def _check_paired(self, adata1, adata2):
        assert all(adata1.obs.index == adata2.obs.index), \
                            "two obs. ids should be the same"

    def __getitem__(self, idx: List[int]):
        return self.X1[idx], self.X2[idx]
        #return {"d1": self.X1[idx], "d2": self.X2[idx]}

    def __len__(self):
        return self.X1.shape[0] 


class SCPairedDataLoader:
    '''
    scdl = SCPairedDataLoader(adata1, adata2, 'counts', train_fraction=0.8, val_fraction=0.1)
    loaders = scdl.get_dataloaders()
    iter(loaders['train']).next()
    '''
    def __init__(
        self,
        adata1: anndata.AnnData,
        adata2: anndata.AnnData,
        key_adata_layer: Literal['counts', 'lib_normed_log1p', 'z_scaled'],        
        train_fraction: float,
        val_fraction: float,
        batch_size: int=100,
        seed: int=42
    ):

        anndata_sanity_check(adata1, key_adata_layer)
        anndata_sanity_check(adata2, key_adata_layer)
        #assert all(adata1.obs.index == adata2.obs.index), "Should have the common ids"

        '''
        print('\n')
        print(f'* Getting DataLoaders from two anndata objects using key "{key_adata_layer}"')
        print(f'- train_fraction : {train_fraction}')
        print(f'- val_fraction : {val_fraction}')
        print(f'- batch_size : {batch_size}')
        print(f'- seed : {seed}')
        print('\n')
        '''

        n_total = len(adata1)

        ''' return shuffled indices '''
        idx_train, idx_val, idx_test = get_idx_train_val_test(
                                        n_total=n_total,
                                        train_fraction=train_fraction,
                                        val_fraction=val_fraction,
                                        shuffle=True,
                                        seed=seed)
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        self.batch_size = batch_size
        
        self.dataset_train = SCPairedDataset(adata1[idx_train].copy(), \
					adata2[idx_train].copy(), key_adata_layer)
        self.dataset_val = SCPairedDataset(adata1[idx_val].copy(), \
					adata2[idx_val].copy(), key_adata_layer)
        self.dataset_test = SCPairedDataset(adata1[idx_test].copy(), \
					adata2[idx_test].copy(), key_adata_layer)

        self.dataloaders = self.get_dataloaders()

    def get_dataloaders(self) -> dict:
        '''return dictionary for loaders'''

        ''' already shuffled indices, so no further shuffling '''
        return {
            'train': DataLoader(self.dataset_train, batch_size=self.batch_size,
                                shuffle=False),

            'test': DataLoader(self.dataset_test, batch_size=self.batch_size,
                                shuffle=False),

            'val': DataLoader(self.dataset_val, batch_size=self.batch_size,
                                shuffle=False)
        }

