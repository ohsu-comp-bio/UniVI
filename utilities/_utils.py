from collections import OrderedDict
import itertools
from typing import Optional, Union
import inspect
import os
import sys
import time
import pathlib
from pathlib import Path
import scipy
import numpy as np
import pandas as pd
import seaborn as sns

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scvt._settings import DataDefault


import torch

def get_ident_matrix(df, adata_ref):
    ident = adata_ref.obs.loc[df.index, 'ident']
    categories = sorted(ident.unique().tolist())
    lst_ident = ident.values.tolist()
    df_ident = df.copy()
    df_ident['ident'] = pd.Categorical(lst_ident, categories=categories)
    return df_ident

def get_ident_adata(lst_barcodes, adata_ref):
    ''' temporaray: is it necessary?? '''
    return adata_ref.obs.loc[lst_barcodes, 'ident'].values.tolist()


def get_markers(m=['m1','m2'], ident='Mono'):
    if m == 'm1':
        return DataDefault.MARKERS_DIC_RNA[ident]
    elif m == 'm1':
        return DataDefault.MARKERS_DIC_ADT[ident]    

class Lists:
    def items_in_list(list_target, list_ref):
        intersect = set(list_target).intersection(set(list_ref))
        return list(intersect)

    def items_not_in_list(list_target, list_ref):
        intersect = set(list_target).intersection(set(list_ref))
        return list(set(list_target).difference(set(intersect)))

    def exist_all_in_lists(list_target, list_ref):
        exist_all = all([element in list_ref for element in list_target])
        return exist_all

    def flatten(list_target):
        return list(itertools.chain(*list_target))

    def list_to_hash_value(list_target, dic_ref):
        lst_converted = []
        for key in list_target:
            converted = dic_ref[key]
            lst_converted.append(converted)
            
        return lst_converted

def lst_unique_from_dic(dic: dict):
    ordered = OrderedDict()
    for k, lst_values in dic.items():
        for value in lst_values:
            ordered[value] = 1
    values_unique = list(ordered.keys())
    
    return values_unique

def substitute_zs_dim(zs, lst_idx, value=0, kind_output="np"):
    
    if isinstance(zs, pd.DataFrame):
        index = zs.index
        columns = zs.columns
    else:
        index = [i for i in range(zs.shape[0])]        
        columns = ['zs' + str(i) for i in range(1, zs.shape[1]+1)]
    
    df_zs = check_mtx_to_df(zs)
    np_zs = df_zs.values
    np_zs[:, lst_idx] = value
    
    zs_off = np_zs
    if kind_output == "np":
        zs_off = np_zs
    elif kind_output == "tensor":
        zs_off = torch.from_numpy(np_zs)
    elif kind_output == "df":
        zs_off = pd.DataFrame(np_zs, index=index, columns=columns)
    
    return zs_off

def try_find_dense_part(x, n_clusters):
    '''
    example
    xp_sorted = _sort_matrix(xp, model_spectral)
    print(heatmap_from_mtx(x_sorted[:70, 70:135]))
    '''
    def _cluster_spectral(data, n_clusters, seed=42):
        #n_clusters = (5,9)  # for adt
        n_clusters = n_clusters
        model = SpectralBiclustering(n_clusters=n_clusters, method='log',
                                     random_state=seed)
        model.fit(data)
        return model

    def _sort_matrix(data, model):
        fit_data = data[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]
        return fit_data
            
    model_spectral = _cluster_spectral(x, n_clusters)
    x_sorted = _sort_matrix(x, model_spectral)    
    print(heatmap_from_mtx(x_sorted))
    return x_sorted

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - np.log(value.size(dim))

def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

def check_mtx_to_df(
        mtx: Union[pd.DataFrame, np.ndarray, torch.Tensor]
):
    if isinstance(mtx, pd.DataFrame):
        df = mtx.copy()
        df.index = mtx.index
        df.columns = mtx.columns
    elif isinstance(mtx, np.ndarray):
        df = pd.DataFrame(mtx)
    elif isinstance(mtx, scipy.sparse.csr.csr_matrix):
        df = pd.DataFrame(mtx.toarray())
    elif isinstance(mtx, torch.Tensor):
        df = pd.DataFrame(tensor_to_numpy(mtx))
    else:
        raise ValueError(f"Invalid matrix type")
    return df


def tensor_to_df(tensor, index=None, columns=None):
    df = pd.DataFrame(tensor_to_numpy(tensor))
    if index is not None:
        df.index = index
    if columns is not None:
        df.columns = columns
    return df

def tensor_to_numpy(tensor):
    assert isinstance(tensor, torch.Tensor), "is not a tensor"
    if tensor.is_cuda:
        np_arr = tensor.cpu().detach().numpy()
    else:
        np_arr = tensor.detach().numpy()
    return np_arr

def tree(directory):
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')

def load_defaults():
    ''' load global default values from HOME directory '''
    import json
    with open('defaults.json', "r") as fh:
        return json.load(fh)

def args2json(args, filepath: pathlib.PosixPath):
    import json
    with open(filepath, 'w') as fh:
        json.dump(args.__dict__, fh)

def str_to_path(filepath: str):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f'{filepath} not exists')
    return filepath

def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def create_run_dir(prefix: Optional[str] = "test"
) -> str:
    runPath = Path('run/' + prefix)
    runPath.mkdir(parents=True, exist_ok=True)
    return runPath

def joint_embed(mtx1, mtx2, reduction='umap'):
    assert mtx1.shape == mtx2.shape
    nrow, ncol = mtx1.shape
    df1 = check_mtx_to_df(mtx1)
    df2 = check_mtx_to_df(mtx2)
    df_concat = pd.concat([df1,df2])
    if reduction == 'umap':
        embeded = embed_umap(df_concat.to_numpy())
    elif reduction == 'tsne':
        embeded = embed_tsne(df_concat.to_numpy())
    elif reduction == 'pca':
        embeded = embed_pca(df_concat.to_numpy())
    df_embeded = pd.DataFrame(embeded)
    df_embeded_1 = df_embeded.iloc[:nrow,:]
    df_embeded_2 = df_embeded.iloc[nrow:,:]
    
    return df_embeded_1, df_embeded_2

def embed_pca(
    data: np.ndarray,
    n_components = 2,
) -> np.ndarray:
    ''' Embed on 2-D PCA space '''
    embedding = PCA(n_components=n_components)
    return embedding.fit_transform(data)

def embed_tsne(
    data: np.ndarray, 
    perplexity: Optional[int] = 30
) -> np.ndarray:
    ''' Embed on 2-D tSNE space (later 3-d ???) '''
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=12.0,
        learning_rate=200.0,
        n_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-07,
        metric='euclidean',
        init='random',
        verbose=0,
        random_state=None,
        method='barnes_hut',
        angle=0.5,
        n_jobs=None
    )
    return embedding.fit_transform(data)

def embed_umap(
    data: np.ndarray, 
    n_neighbors: Optional[int] = 20,
    seed: int = 42
) -> np.ndarray:
    ''' Embed on 2-D UMAP space (later 3-d ???) '''
    #https://github.com/lmcinnes/umap/issues/27

    np.random.seed(seed)
    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric='euclidean',
        n_epochs=None,
        learning_rate=1.0,
        init='spectral',
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=seed,
        metric_kwds=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric='categorical',
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=seed,
        verbose=False
    )
    return embedding.fit_transform(data)


'''
Utils For CHO 
'''
#https://stackoverflow.com/questions/49084842/creating-a-python-object-from-a-json-string
class dic2obj(object):
    ''' d: dict '''
    def __init__(self, d):
        if type(d) is str:
            d = json.loads(d)
        self.from_dict(d)

    def from_dict(self, d):
        self.__dict__ = {}
        for key, value in d.items():
            if type(value) is dict:
                value = dic2obj(value)
            self.__dict__[key] = value

    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            if type(value) is dic2obj:
                value = value.to_dict()
            d[key] = value
        return d

    def __repr__(self):
        return str(self.to_dict())

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]


class Inspect:
    ''' Inspect object member functions and attributes '''
    def __init__(self, obj, is_verbose=False):
        self.obj = obj
        self.is_verbose = is_verbose
    
    def show(self):
        return [ self.attributes(), self.functions() ]
        
    def functions(self):
        #print(f'* Functions of {self.obj}')
        dict_result = {}
        for key, val in inspect.getmembers(self.obj):
            if inspect.isroutine(val) and not (key.startswith('__')):
                dict_result[key] = val

        if self.is_verbose:
            return dict_result
        else:
            return list(dict_result.keys())
                    
    def attributes(self):
        #print(f'* Attributes of {self.obj}')
        dict_result = {}
        for key, val in inspect.getmembers(self.obj):
            if not inspect.isroutine(val) and not (key.startswith('__')):
                dict_result[key] = val

        if self.is_verbose:
            return dict_result
        else:
            return list(dict_result.keys())
                    
class Timer:
    import time

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))

# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.filepath = filepath

    def write(self, message):
        with open (self.filepath, "a", encoding = 'utf-8') as self.log:            
            self.log.write(message)
        #self.terminal.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def data_to_device(x, device):
    if isinstance(x, list):
        return (x[0].to(device), x[1].to(device))
    else:
        return x.to(device)

def dic_params_print(params: dict):
    if isinstance(params, dic2obj):
        params = params.to_dict()
    
    assert isinstance(params, dict)
    print('\n')
    print('* Running parameters')
    for k, v in params.items():
        print(f'[{k}]')
        for k2, v2 in v.items():
            print(f'    - {k2}: {v2}')
    print('\n')


