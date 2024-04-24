from collections import OrderedDict
from typing import Literal, Union, List
from pathlib import Path
import anndata
import scipy

from plotnine import *
from plotnine.data import *

from sklearn.cluster import SpectralBiclustering

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import json
import pickle
import re

import torch
from torch import optim
from torch import nn

import scanpy as sc

from scvt._settings import DataPath, DataDefault 
from scvt._objectives import objective
from scvt.utilities._utils import data_to_device, Logger, Timer, dic2obj, dic_params_print, tensor_to_df, tensor_to_numpy, df_to_tensor, get_device, embed_umap, embed_tsne, embed_pca, kl_divergence, check_mtx_to_df, lst_unique_from_dic, Lists

from scvt.utilities._stats import MVGLikelihood

from scvt.models import _vae
from scvt.models._vae import VAE
from scvt._VAE import VAEMap
from scvt.models._mmvae import MMVAE
from scvt.models._utils import init_weights, EarlyStopping
from scvt.models import _utils as model_utils
from scvt.datasets._datasets import SCPairedDataLoader, anndata_sanity_check, model_feature_matching
from scvt.datasets._external import get_cell_ident_from_reference

from scvt.external.evaluate_FOSCTTM import calc_frac

import importlib
from scvt import _objectives
from scvt.plotting import plots
importlib.reload(_objectives)
importlib.reload(_vae)
importlib.reload(model_utils)
importlib.reload(plots)

from scvt._objectives import elbo, objective
from scvt.plotting.plots import grid_display, heatmap_sample_idx, heatmap_from_mtx, gg_point_embed, gg_point_pair_by_umap, gg_point_pair_embed, boxplot_from_mtx, gg_point_z_activation, gg_point_feature_active, gg_point_scatter

class CrossMap(VAEMap):

    def __init__(
        self,
        dpath_home: str, 
        adata1,
        adata2,
        params = None, 
        pretrained=False
    ):
        self._epoch_current = None
        self._dic_best_samples = None
        self._dic_degs = None
        self._degs_unique = None
        self.FOSCTTM = None
        self.adata_pair = None

        ''' initialization ''' 
        self.verbose = False
        self.adata1 = adata1
        self.adata2 = adata2
        self.pretrained = pretrained
        self.dpath_home = Path(dpath_home)
        self.prj_name = self.dpath_home.name

        if params is None:
            params = self._load_default_params()

        self._init_working_dirs()

        ''' initialize logger ''' 
        #sys.stdout = Logger(self.fpath_log)

        self._init_params(params)

        ''' initialize random seed ''' 
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)

        self._init_device()

        self._init_dataclass()

        self._init_model()


    @property
    def dic_best_samples(self):
        if self._dic_best_samples is None:
            self.get_sample_best_run()
        return self._dic_best_samples

    @property
    def degs_unique(self):
        if self._degs_unique is None:
            self.get_DEGs_per_ident()
        return self._degs_unique

    @property
    def dic_degs(self):
        if self._dic_degs is None:
            self.get_DEGs_per_ident()
        return self._dic_degs

    @classmethod
    def load(cls, dpath_home):

        ''' pretrained file paths '''
        fpath_adata1 = Path(dpath_home).joinpath('adata1.h5ad')
        fpath_adata2 = Path(dpath_home).joinpath('adata2.h5ad')
        fpath_params = Path(dpath_home).joinpath('params.json')
        fpath_objs = Path(dpath_home).joinpath('objs.pkl')
        fpath_objs_dic = Path(dpath_home).joinpath('objs_dic.pkl')

        ''' load adata '''
        adata1 = anndata.read(fpath_adata1)
        adata2 = anndata.read(fpath_adata2)

        ''' load params  '''
        with open(fpath_params, 'r') as fh:
            params = json.load(fh)

        ''' load attributes  '''
        with open(fpath_objs, 'rb') as f: 
            epoch_current, dic_degs, FOSCTTM = pickle.load(f)

        ''' create instance ''' 
        instance = cls(dpath_home, adata1, adata2, params, pretrained=True)

        ''' load df_loss  '''
        instance.df_loss = pd.read_csv(instance.fpath_loss, header=0, index_col=0)

        if instance.fpath_checkpoint.exists():
            ''' load model-state-dict if pretrained ''' 
            checkpoint = torch.load(instance.fpath_checkpoint, map_location=instance.device)
        elif instance.fpath_checkpoint_best.exists():
            checkpoint = torch.load(instance.fpath_checkpoint_best, map_location=instance.device)
            epoch_last = checkpoint['epoch']
            ''' update df_loss in accordance with the best model state''' 
            instance.df_loss = instance.df_loss.iloc[:epoch_last,:]
        else:
            raise ValueError("Model state not exists")

        instance._epoch_current = checkpoint['epoch']
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        ''' load model-state-dict if pretrained ''' 
        checkpoint_best = torch.load(instance.fpath_checkpoint_best, map_location=instance.device)
        instance._epoch_best = checkpoint_best['epoch']
        #print(f'\n- Current epoch: {instance._epoch_current}')
        #print(f'- Best epoch: {instance._epoch_best}')

        ''' load attributes ''' 
        dic_best_samples = torch.load(instance.fpath_objs_dic, map_location=instance.device)

        instance._epoch_current = epoch_current
        instance._dic_best_samples = dic_best_samples
        instance._dic_degs = dic_degs
        instance.FOSCTTM = FOSCTTM

        return instance


    def save(self):
        ''' save adata '''
        self.adata1.write(self.fpath_adata1)
        self.adata2.write(self.fpath_adata2)

        ''' save model state '''
        checkpoint = { 'epoch': self._epoch_current,
                       'model_state_dict': self.model.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict() }
        torch.save(checkpoint, self.fpath_checkpoint)

        ''' save loss history '''
        self.df_loss.to_csv(self.fpath_loss, index=True)

        ''' save attributes '''
        lst_objs = [
        	self._epoch_current,
        	self._dic_degs,
        	self.FOSCTTM
	]
        with open(self.fpath_objs, 'wb') as f: 
            pickle.dump(lst_objs, f)

        ''' save best sample attributes '''
        torch.save(self._dic_best_samples, self.fpath_objs_dic)


    def _init_model(self):

        self.model1 = VAE(**self.param_model_arch_1)
        self.model2 = VAE(**self.param_model_arch_2)
        self.model = MMVAE(self.model1, self.model2, dim_latent=self.params.dim_latent)
        self.model = self.model.to(self.device)
        init_weights(self.model)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                               lr=1e-3)

    def _init_working_dirs(self):

        ''' working home directory '''
        self.dpath_home.mkdir(parents=True, exist_ok=True)
        self.fpath_adata1 = Path(self.dpath_home).joinpath('adata1.h5ad')
        self.fpath_adata2 = Path(self.dpath_home).joinpath('adata2.h5ad')
        self.fpath_params = Path(self.dpath_home).joinpath('params.json')
        self.fpath_checkpoint = Path(self.dpath_home).joinpath('checkpoint.pt')
        self.fpath_checkpoint_best = Path(self.dpath_home).joinpath('checkpoint_best.pt')
        self.fpath_objs = Path(self.dpath_home).joinpath('objs.pkl')
        self.fpath_objs_dic = Path(self.dpath_home).joinpath('objs_dic.pkl')
        self.fpath_log = Path(self.dpath_home).joinpath('run.log')
        self.fpath_loss = Path(self.dpath_home).joinpath('loss.csv')

        ''' analysis directory '''
        self.dpath_anal = Path(self.dpath_home).joinpath('anal')
        self.dpath_anal.mkdir(parents=True, exist_ok=True)
        self.dpath_anal_feature = Path(self.dpath_anal).joinpath('feature')
        self.dpath_anal_feature.mkdir(parents=True, exist_ok=True)
        self.dpath_anal_grad = Path(self.dpath_anal).joinpath('grad')
        self.dpath_anal_grad.mkdir(parents=True, exist_ok=True)


    def _init_dataclass(self):

        if self.params.dec_model == "nb":
            key_adata_layer = 'counts'
        elif self.params.dec_model == "gaussian":
            key_adata_layer = 'z_scaled'
        else:
            print(f'Invalid decoder model: {self.params.dec_model}')


        self.scdl = SCPairedDataLoader(
        		adata1 = self.adata1,
        		adata2 = self.adata2,
        		key_adata_layer = key_adata_layer,
        		train_fraction = self.params.train_fraction,
        		val_fraction = self.params.val_fraction,
        		batch_size = self.params.batch_size,
        		seed = self.params.seed
        )

    def _load_default_params(self):
        with open(DataPath.DEFAULT_PARAMS_MULTI, 'r') as fh:
            params = json.load(fh)
        return params

    def _init_params(self, params):

        ''' param file handling '''
        if not self.pretrained: 
            with open(self.fpath_params, 'w') as fh:
                json.dump(params, fh)

        ''' different part of parameters '''
        self.param_early_stopping = params['early_stopping']
        self.param_model_arch_1 = params['model_arch_1']
        self.param_model_arch_2 = params['model_arch_2']
        self.param_train = params['train']

        ''' print out running parameters '''
        if self.verbose:
            dic_params_print(params)

        dic_params = {**self.param_train, **self.param_early_stopping, 
				**self.param_model_arch_1, **self.param_model_arch_2}
        self.params = dic2obj(dic_params)


    def embed(self, 
	mtx_in: Union[np.ndarray, torch.Tensor, pd.DataFrame], 
	reduction="umap", n_neighbors=20
    ):
        df_mtx = check_mtx_to_df(mtx_in)

        if reduction == "umap":
            embeded = embed_umap(df_mtx.values, n_neighbors, self.params.seed)
        elif reduction == "tsne":
            embeded = embed_tsne(df_mtx.values)
        elif reduction == "pca":
            embeded = embed_pca(df_mtx.values)
        else:
            raise ValueError(f"Invalid embedding reduction method: {reduction}")

        if isinstance(mtx_in, np.ndarray):
            return embeded
        elif isinstance(mtx_in, torch.Tensor):
            return torch.from_numpy(embeded)
        elif isinstance(mtx_in, pd.DataFrame):
            return pd.DataFrame(embeded, index=mtx_in.index, columns=["e1","e2"])

    #def generate(self, sample_size=self.params.batch_size, n_mc_samples=1, save=False):
    def generate(self, kind: Literal["np", "tensor", "df"]="np", sample_size=100, save=False):

        if save:
            x_gen = self.model.generate(sample_size)
            tensor_to_df(x_gen).to_csv(
                self._filename_epoch(self.dpath_anal,
                                        "mtx_xg_", self.epoch_current), index=True)
        else:
            xg = self.model.generate(sample_size)
            if kind == 'np':
                xg = tensor_to_numpy(xg)
            elif kind == 'df':
                xg = tensor_to_df(xg)
            elif kind == 'tensor':
                xg = xg
            else:
                raise ValueError(f"Invalid data type: {kind}")
            return xg


    def get_latent_features(self, x_in, save=False):
        if isinstance(x_in, list) or isinstance(x_in, tuple):
            return self._get_latent_both(x_in, save=save)
        else:
            return self._get_latent_either(x_in, save=save)

    def _get_latent_both(self, 
	xs: List[Union[np.ndarray, torch.Tensor]], 
	save=False
    ):
        x1, x2 = xs
        assert type(x1) == type(x2)

        df_x1 = check_mtx_to_df(x1)
        df_x2 = check_mtx_to_df(x2)
        x1 = df_to_tensor(df_x1)
        x2 = df_to_tensor(df_x2)

        print(f"using data from modalities")
        zss = self.model.get_latent_features([x1,x2])
        z_avg = (zss[0]+zss[1])/2

        if save:
            tensor_to_df(zss[0]).to_csv(
                self._filename_epoch(self.dpath_anal,
                                        "mtx_z1", self.epoch_current), index=True)
            tensor_to_df(zss[1]).to_csv(
                self._filename_epoch(self.dpath_anal,
                                        "mtx_z2", self.epoch_current), index=True)
            return 

        if isinstance(xs[0], torch.Tensor):
            result = {'z':z_avg, 'z1':zss[0], 'z2':zss[1]}
        elif isinstance(xs[0], np.ndarray):
            result = {'z':z_avg.cpu().numpy(), 'z1':zss[0].cpu().numpy(), 'z2':zss[1].cpu().numpy()}
        elif isinstance(xs[0], pd.DataFrame):
            columns_zs = ['z'+str(i+1) for i in range(zss[0].shape[1])]
            df_z_avg = tensor_to_df(z_avg, index=df_x1.index, columns=columns_zs)
            df_z1 = tensor_to_df(zss[0], index=df_x1.index, columns=columns_zs)
            df_z2 = tensor_to_df(zss[1], index=df_x2.index, columns=columns_zs)
            result = {'z':df_z_avg, 'z1':df_z1, 'z2':df_z2}
        return result

    
    def reconstruct(self, x_in, save=False):
        if isinstance(x_in, list):
            return self._recon_both(x_in, save=save)
        else:
            return self._recon_either(x_in)

    def _recon_both(self, 
	xs: List[Union[pd.DataFrame, np.ndarray, torch.Tensor]]=None, save=False
    ):
        if xs is None:
            #print('im here'); assert False
            df_x1, df_x2 = self.dic_best_samples['xs']
        else:
            df_x1 = check_mtx_to_df(xs[0])
            df_x2 = check_mtx_to_df(xs[1])

        x1 = df_to_tensor(df_x1)
        x2 = df_to_tensor(df_x2)

        [[xp11, xp12], [xp21, xp22]] = self.model.reconstruct([x1,x2])  
        df_xp11 = tensor_to_df(xp11, index=df_x1.index, columns=df_x1.columns)
        df_xp12 = tensor_to_df(xp12, index=df_x2.index, columns=df_x2.columns)
        df_xp21 = tensor_to_df(xp21, index=df_x1.index, columns=df_x1.columns)
        df_xp22 = tensor_to_df(xp22, index=df_x2.index, columns=df_x2.columns)

        dic_recon = dict()
        dic_recon['xp1'] = df_xp11
        dic_recon['xpp2'] = df_xp12
        dic_recon['xpp1'] = df_xp21
        dic_recon['xp2'] = df_xp22

        return dic_recon


    def _save_layers_to_file(self, layers):

        for name_layer, feature in layers.outputs.items():
            tensor_to_df(feature).to_csv(
		self._filename_epoch(self.dpath_anal_feature, 
				name_layer, self.epoch_current), index=True)


    def _filename_epoch(self, dpath, prefix, epoch, ext='.csv'):
        fname = "_".join([prefix, 'ep', str(epoch), ext])
        #fname = re.sub("_\.", ".", fname)
        fname = re.sub("_.csv", ".csv", fname)
        return dpath.joinpath(fname)


    def plot_features_compare(self, m, lst_features):
        assert isinstance(lst_features, list), f"features should be in a list"
        ncols=len(lst_features)
        fpath_merge = self.dpath_anal.joinpath('plot_umap_amerge.png')
        all_lst_paths_img = []
        for key_score in ['counts', 'xp', 'xpp']:
            for feature in lst_features:
    
                fname = 'plot_feature_' + feature + '_' + key_score + '.png'
                fpath_feature = self.dpath_anal.joinpath(fname)
    
                g_feature = self.plot_feature(feature, m=m, key_score=key_score)
    
                if g_feature is None:
                    #raise ValueError(f"invalid feature: {feature} in {key_score}")
                    print(f"invalid feature: {feature} in {key_score}")
                    continue

                g_feature.save(fpath_feature, verbose=False)
                all_lst_paths_img.append(fpath_feature)

                ''' save only existing images '''
                lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=ncols, lst_titles=[],
                                                figratio=1.0, filepath_out=fpath_merge)

    def plot_features(self, lst_features, m, key_score='counts', ncols=1):
        '''
        [ Example 1 ]
        n_all = cm.adata1.obsm['Z'].shape[1]
        zs_name = ['z' + str(i) for i in range(1,n_all+1)]
        cm.plot_features(zs_name, m='m1', key_score='Z', ncols=10)

        [ Example 2 ]
        z = cm.adata1.obsm['Z']
        cm.adata1.obsm['Z_pca'] = embed_pca(z, n_components=5)

        n_all = cm.adata1.obsm['Z_pca'].shape[1]
        zs_name = ['z' + str(i) for i in range(1,n_all+1)]
        cm.plot_features(zs_name, m='m1', key_score='Z_pca', ncols=10)
        '''
        assert isinstance(lst_features, list), f"features should be in a list"
        fpath_merge = self.dpath_anal.joinpath('plot_umap_amerge.png')
    
        all_lst_paths_img = []
        
        for feature in lst_features:
            fname = 'plot_feature_' + feature + '_' + key_score + '.png'
            fpath_feature = self.dpath_anal.joinpath(fname)
            
            g_feature = self.plot_feature(feature, m=m, key_score=key_score)
            if g_feature is None:
                continue
    
            g_feature.save(fpath_feature, verbose=False)
            all_lst_paths_img.append(fpath_feature)

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]
    
        grid_display(lst_paths_imgs, ncols=ncols, lst_titles=[],
                                                figratio=1.0, filepath_out=fpath_merge)


    def plot_feature(self, feature, m='m1', key_score='lib_normed_log1p', figure_size=(2,2)):
    
        # umap coordinates
        try: 
            mtx_umap = self.adata1.obsm['Z_umap']
        except:
            raise ValueError(f"not set in .obsm['Z_umap']. try .save_latent_features() first")

        # select modality first 
        if m == 'm1':
            adata_score = self.adata1
        elif m == 'm2':
            adata_score = self.adata2
        else: 
            raise ValueError(f"should be either 'm1' or 'm2'")

        # for high-dimensional features 
        if key_score in ['counts', 'lib_normed_log1p', 'z_scaled', 'xp', 'xpp']:

            # for labeling
            if key_score in ['counts', 'lib_normed_log1p', 'z_scaled']:
                data_type = 'x'
            elif key_score == 'xp':
                data_type = 'xp'
            elif key_score == 'xpp':
                data_type = 'xpp'
    
            try:
                score_all = adata_score.layers[key_score]
                if isinstance(score_all, scipy.sparse.csr.csr_matrix):
                    score_all = adata_score.layers[key_score].toarray()            
                score = score_all[:,adata_score.var.index==feature]
    
                if isinstance(score, scipy.sparse.csr.csr_matrix):
                    score = score.toarray()
                
                if score.shape[1] == 0:
                    print(f"no info for feature '{feature}'")
                    return None
            except:
                raise ValueError(f"no info for .layers[{key_score}]")
            
        # for low-dimensional features 
        elif key_score in ['Z', 'Z_pca']:
            m = 'Z'
            if key_score == 'Z':
                data_type = 'active'
            elif key_score == 'Z_pca':
                data_type = 'pca'

            feature = feature.upper()
            assert feature.startswith('Z')
            nth = int(feature.split('Z')[1])
            try:
                score = self.adata1.obsm[key_score][:, nth-1]  # only in adata1
                score = np.expand_dims(score, 1)
            except:
                raise ValueError(f"no info for adata1.obsm[{key_score}]")
        else:
            raise ValueError(f"invalid key_score: {key_score}")

        title = feature + '_' + data_type

        return gg_point_feature_active(mtx_umap, score, title=title, figure_size=figure_size)


    def plot_features_recon(self, xs=None, es=None, ident='NK', lst_feature_names_pairs=None):

        ''' default input '''
        if xs is None or es is None:
            xs = self.dic_best_samples['xs']
            es = self.dic_best_samples['es']

        ''' do reconstruction '''
        #[[xp11, xp12], [xp21, xp22]] = self.reconstruct(xs)
        dic_recon = self.reconstruct(xs)
        xp11 = dic_recon['xp1']
        xp12 = dic_recon['xpp2']
        xp21 = dic_recon['xpp1']
        xp22 = dic_recon['xp2']

        ''' list of degs '''
        if lst_feature_names_pairs is None:
            lst_degs_1 = self.dic_degs[0][ident]
            lst_degs_2 = self.dic_degs[1][ident]
            #lst_degs_1 = self.degs_unique[0]
            #lst_degs_2 = self.degs_unique[1]
        else:
            lst_degs_1 = lst_feature_names_pairs[0]
            lst_degs_2 = lst_feature_names_pairs[1]

        ''' for modality 1 '''
        g_features_recon_x1 = self.plot_features_one(xs[0],es[0], lst_degs_1, color='red')  # original
        g_features_recon_xp11 = self.plot_features_one(xp11,es[0], lst_degs_1, color='red')  # reconstruction
        g_features_recon_xp21 = self.plot_features_one(xp21,es[0], lst_degs_1, color='red') # cross-generation
        
        ''' for modality 2 '''
        g_features_recon_x2 = self.plot_features_one(xs[1],es[1], lst_degs_2, color='blue')  # original
        g_features_recon_xp22 = self.plot_features_one(xp22,es[1], lst_degs_2, color='blue') # recontruction
        g_features_recon_xp12 = self.plot_features_one(xp12,es[1], lst_degs_2, color='blue')  # cross-generation
        
        ''' define image file names '''
        fpath_features_recon_x1_all = self.dpath_anal.joinpath('plot_features_recon_x1_all.png')
        fpath_features_recon_x1 = self.dpath_anal.joinpath('plot_features_recon_x1.png')
        fpath_features_recon_xp11 = self.dpath_anal.joinpath('plot_features_recon_xp11.png')
        fpath_features_recon_xp21 = self.dpath_anal.joinpath('plot_features_recon_xp21.png')

        fpath_features_recon_x2_all = self.dpath_anal.joinpath('plot_features_recon_x2_all.png')
        fpath_features_recon_x2 = self.dpath_anal.joinpath('plot_features_recon_x2.png')
        fpath_features_recon_xp22 = self.dpath_anal.joinpath('plot_features_recon_xp22.png')
        fpath_features_recon_xp12 = self.dpath_anal.joinpath('plot_features_recon_xp12.png')

        fname_merge = 'amerge_features_recon_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' save images '''
        g_features_recon_x1.save(fpath_features_recon_x1, verbose=False)
        g_features_recon_xp11.save(fpath_features_recon_xp11, verbose=False)
        g_features_recon_xp21.save(fpath_features_recon_xp21, verbose=False)

        g_features_recon_x2.save(fpath_features_recon_x2, verbose=False)
        g_features_recon_xp22.save(fpath_features_recon_xp22, verbose=False)
        g_features_recon_xp12.save(fpath_features_recon_xp12, verbose=False)

        ''' list of filepaths '''
        lst_paths_img_1 = [
        	fpath_features_recon_x1,
        	fpath_features_recon_xp11,
        	fpath_features_recon_xp21
        ]
        grid_display(lst_paths_img_1, ncols=1, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_features_recon_x1_all)
        lst_paths_img_2 = [
        	fpath_features_recon_x2,
        	fpath_features_recon_xp22,
        	fpath_features_recon_xp12
        ]
        grid_display(lst_paths_img_2, ncols=1, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_features_recon_x2_all)
        lst_paths_img = [
        	fpath_features_recon_x1_all,
        	fpath_features_recon_x2_all
        ]
        grid_display(lst_paths_img, ncols=1, lst_titles=['Modality 1', 'Modality 2'],
                                        figratio=1.0, filepath_out=fpath_merge)

    def plot_features_one(self, df_x, df_es, lst_feature_names=None, modality='', color='red'):

        ''' default genes defined '''
        lst_feature_names_default = DataDefault.FEATURE_GENES
        #lst_feature_names_default = DataDefault.MARKERS_DIC_RNA['Mono']

        ''' init genes: common genes in two modality '''
        if lst_feature_names is None:
            lst_feature_names = lst_feature_names_default

        ''' need to add for checking if genes exists'''
        df_selected = df_x[lst_feature_names]

        return gg_point_feature_active(df_es, df_selected, title=self.prj_name+modality, color=color)


    def plot_features_OLD(self, xs=None, es=None, lst_feature_names=None):

        ''' default input '''
        if xs is None or es is None:
            xs = self.dic_best_samples['xs']
            es = self.dic_best_samples['es']

        ''' default genes defined '''
        lst_feature_names_default = DataDefault.FEATURE_GENES

        fpath_features_1 = self.dpath_anal.joinpath('plot_features_1.png')
        fpath_features_2 = self.dpath_anal.joinpath('plot_features_2.png')
        fname_merge = 'amerge_features_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' init genes: common genes in two modality '''
        if lst_feature_names is None:
            lst_feature_names = lst_feature_names_default

        df_x1, df_x2 = xs
        df_es1, df_es2 = es
        g_features_1 = self.plot_features_one(df_x1, df_es1, lst_feature_names, color='red')
        g_features_2 = self.plot_features_one(df_x2, df_es2, lst_feature_names, color='blue')

        g_features_1.save(fpath_features_1, verbose = False)
        g_features_2.save(fpath_features_2, verbose = False)

        ''' list of filepaths '''
        lst_paths_img = [
                fpath_features_1,
                fpath_features_2
        ]

        grid_display(lst_paths_img, ncols=1, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)

    def create_eval_file(self, outpath='eval_synomicvae.h5ad'):
        try:
            adata_Z = anndata.AnnData(self.adata1.obsm['Z'], obs=self.adata1.obs)
        except:
            raise ValueError(f"no obsm['Z'] defined. try .save_latent_features first")
        adata_Z.write(outpath)
        print(f"{outpath} file created.")
        return adata_Z


    def get_anndata_pair(self):
        '''
        z1 = cm.adata1.obsm['z']
        z2 = cm.adata2.obsm['z']
        zadata1 = anndata.AnnData(z1)
        zadata1.obs = cm.adata1.obs
        zadata2 = anndata.AnnData(z2)
        zadata2.obs = cm.adata2.obs

        data_id = 'hao_sample'
        infile1 = dirpath_home + '/anal/eval_synomicvae_' + data_id + '_z1.h5ad'
        infile2 = dirpath_home + '/anal/eval_synomicvae_' + data_id + '_z2.h5ad'

        zadata1.write(infile1)
        zadata1.write(infile2)

        outfile_combined = dirpath_home + '/anal/aint_synomicvae_batch_' + data_id + '.h5ad'
        
        aint = create_comb_anndata_to_eval(infile1, infile2, outpath=outfile_combined)
        '''
    
        if self.adata_pair is not None:
            return self.adata_pair
    
        try: 
            z1 = self.adata1.obsm['z']
            z2 = self.adata2.obsm['z']
        except:
            raise ValueError(f"run .save_latent_features first")
    
        # anndata 1
        adata_ref = self.adata1
        ad1 = anndata.AnnData(z1)
        ad1.obs = adata_ref.obs
    
        # anndata 2
        adata_ref = self.adata2
        ad2 = anndata.AnnData(z2)
        ad2.obs = adata_ref.obs

        # paring and run umap
        ad_pair = ad1.concatenate(ad2)
        sc.pp.neighbors(ad_pair, use_rep='X')
        sc.tl.umap(ad_pair)
        self.adata_pair = ad_pair
    
        return ad_pair


    def plot_umap_pair(self, lst_nsamples=[100], figure_size=(3,3), ncols=1, lab1="m1", lab2="m2"):
    
        """ wrapper for plot_umap_combined('ident') """
        fpath_merge = self.dpath_anal.joinpath('plot_umap_pair_amerge.png')
    
        all_lst_paths_img = []
        for nsample in lst_nsamples:
            fname = 'plot_umap_pair_' + str(nsample) + '.png'
            fpath_part = self.dpath_anal.joinpath(fname)
            
            g_part = self._plot_umap_pair(nsample, 
					lab1=lab1, lab2=lab2,
					figure_size=figure_size)

            g_part.save(fpath_part, verbose=False)
            all_lst_paths_img.append(fpath_part)

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=ncols, lst_titles=[],
                                            figratio=1.0, filepath_out=fpath_merge)


    def _plot_umap_pair(self, n=200, show_text=False, figure_size=(4,4), lab1="m1", lab2="m2"):
    
        if self.adata_pair is None:
            self.get_anndata_pair()
    
        nrow = int(self.adata_pair.shape[0] / 2)
    
        a1 = self.adata_pair[:nrow,:]
        a2 = self.adata_pair[nrow:,:]
    
        e1 = a1[:n].obsm['X_umap']
        e2 = a2[:n].obsm['X_umap']
    
        return gg_point_pair_embed(e1, e2, show_text=show_text,
				lbl_mod1=lab1, lbl_mod2=lab2,
				figure_size=figure_size)


    def plot_umap(self, lst_key_obs=['ident'], ncols=1):
    
        """ wrapper for plot_umap_combined('ident') """
        fpath_merge = self.dpath_anal.joinpath('plot_umap_amerge.png')
    
        all_lst_paths_img = []
        for key_obs in lst_key_obs:
            fname = 'plot_umap_' + key_obs + '.png'
            fpath_key_obs = self.dpath_anal.joinpath(fname)
            
            g_key_obs = self._plot_umap_combined(key_obs)
        
            g_key_obs.save(fpath_key_obs, verbose=False)
            all_lst_paths_img.append(fpath_key_obs)

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=ncols, lst_titles=[],
                                            figratio=1.0, filepath_out=fpath_merge)

    def plot_umap_each(self, m='m1', lst_key_obs=['ident'], ncols=1):
        
        fpath_merge = self.dpath_anal.joinpath('plot_umap_' + m + '_amerge.png')
    
        all_lst_paths_img = []
        for key_obs in lst_key_obs:
            fname = 'plot_umap_' + key_obs + '.png'
            fpath_key_obs = self.dpath_anal.joinpath(fname)
            
            g_key_obs = self._plot_umap_each(m, key_obs)
        
            g_key_obs.save(fpath_key_obs, verbose=False)
            all_lst_paths_img.append(fpath_key_obs)

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]
    
        grid_display(lst_paths_imgs, ncols=ncols, lst_titles=[],
                                            figratio=1.0, filepath_out=fpath_merge)

    def _plot_umap_each(self, 
            m='m1', 
            key_obs='ident',
            figure_size=(3,3)
    ):
        if m == 'm1':
            adata = self.adata1
        elif m == 'm2':
            adata = self.adata2
        else:
            raise ValueError(f"should be either 'm1' or 'm2': {m}")
        
        try:
            lst_ident = adata.obs[key_obs].tolist()
        except:
            print(f"no key defined in .obs: {key_obs}")
            lst_ident = None
        
        try:
            colors = self._get_ident_colors(key_obs)
            #colors = adata.uns[key_obs+'_colors']
        except:
            print(f"no colors defined in .obsm: {key_obs}_colors")
            colors = None
        
        X_umap = adata.obsm['X_umap']
        title = key_obs
        g = gg_point_embed(X_umap, lst_ident, colors=colors, 
			figure_size=figure_size, title=title)
        return g

    def _plot_umap_combined(self, 
            key_obs='ident',
            figure_size=(3,3)
    ):
        """ plot_umap for combined Z from .obsm['Z_umap'] in adata1
            , which is defined beforehand by sc.tl.umap() 
        """

        adata = self.adata1
    
        try:
            lst_ident = adata.obs[key_obs].tolist()
        except:
            print(f"no key defined in .obs: {key_obs}")
            lst_ident = None
            
        try:
            colors = self._get_ident_colors(key_obs)
            #colors = adata.uns[key_obs+'_colors']
        except:
            print(f"no colors defined in .obs: {key_obs}_colors")
            colors = None
        
        try:
            X_umap = adata.obsm['Z_umap']        
        except:
            raise ValueError(f".obsm['Z_umap'] should be defined first. try .save_latent_features() first.")
            
        title = key_obs
        g = gg_point_embed(X_umap, lst_ident, colors=colors, 
			figure_size=figure_size, title=title)
        return g


    def save_recon_features(self):
        """ Generate slots"""
        x1 = self.adata1.layers['z_scaled']
        x2 = self.adata2.layers['z_scaled']

        #[[xp11, xp12], [xp21, xp22]] = self.reconstruct([x1,x2])
        dic_recon = self.reconstruct([x1,x2])
        self.adata1.layers['xp'] = dic_recon['xp1']
        self.adata1.layers['xpp'] = dic_recon['xpp1']
        self.adata2.layers['xp'] = dic_recon['xp2']
        self.adata2.layers['xpp'] = dic_recon['xpp2']


    def save_latent_features(self):
        """ Generate slots

        self.adata1.obsm['mZ']
        self.adata1.obsm['Z_umap']

        self.adata1.obsm['Z']
        self.adata1.obsm['X_umap']

        self.adata2.obsm['Z']
        self.adata2.obsm['X_umap']
	"""

        ''' load the best model '''
        self.update_to_best_epoch()

        ''' get samples '''
        X1 = self.adata1.X
        X2 = self.adata2.X

        ''' get latent features'''
        dic_Z = self.get_latent_features([X1,X2])

        ''' save to the slots '''
        # save Zs
        self.adata1.obsm['Z'] = dic_Z['z']
        self.adata1.obsm['z'] = dic_Z['z1']
        self.adata2.obsm['z'] = dic_Z['z2']

        # calculate umap for multimodal Z first, then save it to 'Z_umap'
        sc.pp.neighbors(self.adata1, use_rep='Z')
        sc.tl.umap(self.adata1)
        self.adata1.obsm['Z_umap'] = self.adata1.obsm['X_umap'] # umap for whole mZ

        # for Z1
        sc.pp.neighbors(self.adata1, use_rep='z')
        sc.tl.umap(self.adata1) # saved in self.adata1.obsm['X_umap']

        # for Z2
        sc.pp.neighbors(self.adata2, use_rep='z')
        sc.tl.umap(self.adata2) # saved in self.adata2.obsm['X_umap']

        #''' FOSCTTM score '''
        #self.FOSCTTM = calc_frac(zs1, zs2)


    def get_sample_best_run(self, kind: Literal["np", "tensor", "df"]="df", for_reset=False):

        ''' load model '''
        self.update_to_best_epoch()

        ''' get sample annotation '''
        df_x1, df_x2 = self.get_sample_data("df")

        ''' run model '''
        x1 = df_to_tensor(df_x1)
        x2 = df_to_tensor(df_x2)

        qzxs, pxzs, zss = self.model([x1, x2])

        zs1 = zss[0].squeeze()
        zs2 = zss[1].squeeze()

        ''' embedding by umap '''
        es1 = self.embed(zs1)
        es2 = self.embed(zs2)

        ''' format output '''
        x1 = tensor_to_numpy(x1)
        x2 = tensor_to_numpy(x2)
        zs1 = tensor_to_numpy(zs1)
        zs2 = tensor_to_numpy(zs2)
        es1 = tensor_to_numpy(es1)
        es2 = tensor_to_numpy(es2)

        ''' FOSCTTM score '''
        self.FOSCTTM = calc_frac(zs1, zs2)

        ''' formatting for dataframe output '''
        df_zs1 = pd.DataFrame(zs1)
        df_zs1.columns = ['z'+str(i) for i in range(1, zs1.shape[1]+1)]
        df_zs1.index = df_x1.index

        df_zs2 = pd.DataFrame(zs2)
        df_zs2.columns = ['z'+str(i) for i in range(1, zs2.shape[1]+1)]
        df_zs2.index = df_x1.index

        df_es1 = pd.DataFrame(es1, columns=["dim1", "dim2"], index=df_x1.index)
        df_es2 = pd.DataFrame(es2, columns=["dim1", "dim2"], index=df_x2.index)

        if(kind == "df"):
            x1,x2 = df_x1, df_x2
            zs1, zs2 = df_zs1, df_zs2
            es1, es2 = df_es1, df_es2
        elif(kind == "np"):
            pass
        else:
            raise ValueError(f"Invalid data type : {kind}")

        ''' attributes for best sample run for plotting reproducility '''
        if self._dic_best_samples is None or for_reset:
            self._dic_best_samples = {	'xs': [df_x1, df_x2], 
					'zss': [df_zs1, df_zs2], 
					'es': [df_es1, df_es2], 
					'pxzs': pxzs, 
					'qzxs': qzxs }

        print(f'\n- FOSCTTM of {self.prj_name}: {self.FOSCTTM:.2f}')
        return [x1,x2], [zs1,zs2], [es1,es2], pxzs, qzxs


    def get_sample_data(self, 
	kind: Literal["np", "tensor", "df"]="np"
    ):
        xs = list(self.scdl.dataloaders['test'])[0]

        if kind == "np":
            return xs[0].data.numpy(), xs[1].data.numpy()
        elif kind == "tensor":
            x1 = xs[0].to(self.device)
            x2 = xs[1].to(self.device)
            return [x1,x2]
        elif kind == "df":
            ''' get annotation from anndata object 1 ''' 
            obs_index1 = list(self.adata1[self.scdl.idx_test[:self.params.batch_size]].obs.index)
            df_sample1 = pd.DataFrame(tensor_to_numpy(xs[0]), index=obs_index1, columns=self.adata1.var.index)
            ''' get annotation from anndata object 2 ''' 
            obs_index2 = list(self.adata2[self.scdl.idx_test[:self.params.batch_size]].obs.index)
            df_sample2 = pd.DataFrame(tensor_to_numpy(xs[1]), index=obs_index2, columns=self.adata2.var.index)
            return df_sample1, df_sample2
        else:
            raise ValueError("Invalid data type: {kind}")


    def plot_z_embed_ident(self, zs=None, reduction='umap', key_ident='ident', colors=None):
        ''' set default '''
        if zs is None:
            zs = self.dic_best_samples['zss'][0]
        assert isinstance(zs, pd.DataFrame), "should be dataframe with barcode information"

        ''' get ident for each barcode from reference '''
        lst_barcodes = list(zs.index)

        #ident = get_cell_ident_from_reference(lst_barcodes, data_id='cit_hao', str_level=key_ident)
        ident = self.adata1[lst_barcodes, ].obs[key_ident].tolist()

        colors = self._get_ident_colors(key_ident)

        title = self.prj_name + '_' + key_ident

        embeded = self.embed(zs, reduction=reduction)
        g_embeded = gg_point_embed(embeded, lst_ident=ident, title=title, colors=colors)

        return g_embeded

    def plot_z_embed_pair(self, zss=None, reduction='umap'):
        ''' set default '''
        if zss is None:
            zss = self.dic_best_samples['zss']

        ''' tensors umap-embeded '''
        zs1, zs2 = zss 
        zs1 = check_mtx_to_df(zs1)
        zs2 = check_mtx_to_df(zs2)

        title = self.prj_name + '_' + f'{self.FOSCTTM:.2f}'
        return gg_point_pair_by_umap(zs1, zs2, 
                                    title=title, figure_size=(5,5))
        #z1_embeded = self.embed(zs1.values, reduction=reduction)
        #z2_embeded = self.embed(zs2.values, reduction=reduction)
        #z1_embeded = self.embed(zs1, reduction=reduction)
        #z2_embeded = self.embed(zs2, reduction=reduction)
        #return gg_point_pair_by_umap(z1_embeded, z2_embeded,
                                    #title=title, figure_size=(5,5))

    def plot_z_embed_all(self, zss=None, reduction='umap'):
        if 'hao' in self.prj_name:
            self._plot_z_embed_all_hao()
        else:
            self._plot_z_embed_all_etc()

    def _plot_z_embed_all_etc(self, zss=None, reduction='umap'):
        ''' set default '''
        if zss is None:
            zss = self.dic_best_samples['zss']

        ''' tensors umap-embeded '''
        zs1, zs2 = zss
        z = (zs1+zs2)/2.0

        ''' filepaths for output images '''
        fpath_embed_umap_pair = self.dpath_anal.joinpath('plot_z_embed_umap_pair.png')
        fpath_embed_umap_z_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z_1.png')
        fpath_embed_umap_z1_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_ident.png')
        fpath_embed_umap_z2_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_ident.png')
        fname_merge = 'amerge_z_embed_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' plot z_embed_pair '''
        g_pair = self.plot_z_embed_pair(zss, reduction=reduction)

        ''' embedding and plotting '''
        g_z_embed_umap_z_ident = self.plot_z_embed_ident(z, reduction=reduction, key_ident='ident')
        g_z_embed_umap_z1_ident = self.plot_z_embed_ident(zs1, reduction=reduction, key_ident='ident')
        g_z_embed_umap_z2_ident = self.plot_z_embed_ident(zs2, reduction=reduction, key_ident='ident')

        ''' save output image files '''
        g_z_embed_umap_z_ident.save(fpath_embed_umap_z_1, verbose=False)
        g_z_embed_umap_z1_ident.save(fpath_embed_umap_z1_1, verbose=False)
        g_z_embed_umap_z2_ident.save(fpath_embed_umap_z2_1, verbose=False)
        g_pair.save(fpath_embed_umap_pair, verbose=False)

        ''' save only existing images '''
        all_lst_paths_img = [
                fpath_embed_umap_z1_1,
                fpath_embed_umap_z_1,
                fpath_embed_umap_z2_1,
                fpath_embed_umap_pair,
        ]

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=2, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)



    def _plot_z_embed_all_hao(self, zss=None, reduction='umap'):
        ''' set default '''
        if zss is None:
            zss = self.dic_best_samples['zss']

        ''' tensors umap-embeded '''
        zs1, zs2 = zss
        z = (zs1+zs2)/2.0

        ''' filepaths for output images '''
        fpath_embed_umap_pair = self.dpath_anal.joinpath('plot_z_embed_umap_pair.png')
        fpath_embed_umap_z_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z_1.png')
        fpath_embed_umap_z1_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_ident.png')
        fpath_embed_umap_z1_2 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_level2.png')
        fpath_embed_umap_z1_3 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_level3.png')
        fpath_embed_umap_z2_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_ident.png')
        fpath_embed_umap_z2_2 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_level2.png')
        fpath_embed_umap_z2_3 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_level3.png')
        fname_merge = 'amerge_z_embed_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' plot z_embed_pair '''
        g_pair = self.plot_z_embed_pair(zss, reduction=reduction)

        ''' embedding and plotting '''
        g_z_embed_umap_z_ident = self.plot_z_embed_ident(z, reduction=reduction, key_ident='ident')

        g_z_embed_umap_z1_ident = self.plot_z_embed_ident(zs1, reduction=reduction, key_ident='ident')
        g_z_embed_umap_z1_level2 = self.plot_z_embed_ident(zs1, reduction=reduction, key_ident='level2')
        g_z_embed_umap_z1_level3 = self.plot_z_embed_ident(zs1, reduction=reduction, key_ident='level3')

        g_z_embed_umap_z2_ident = self.plot_z_embed_ident(zs2, reduction=reduction, key_ident='ident')
        g_z_embed_umap_z2_level2 = self.plot_z_embed_ident(zs2, reduction=reduction, key_ident='level2')
        g_z_embed_umap_z2_level3 = self.plot_z_embed_ident(zs2, reduction=reduction, key_ident='level3')

        ''' save output image files '''
        g_z_embed_umap_z_ident.save(fpath_embed_umap_z_1, verbose=False)

        g_z_embed_umap_z1_ident.save(fpath_embed_umap_z1_1, verbose=False)
        g_z_embed_umap_z1_level2.save(fpath_embed_umap_z1_2, verbose=False)
        g_z_embed_umap_z1_level3.save(fpath_embed_umap_z1_3, verbose=False)

        g_z_embed_umap_z2_ident.save(fpath_embed_umap_z2_1, verbose=False)
        g_z_embed_umap_z2_level2.save(fpath_embed_umap_z2_2, verbose=False)
        g_z_embed_umap_z2_level3.save(fpath_embed_umap_z2_3, verbose=False)
        g_pair.save(fpath_embed_umap_pair, verbose=False)

        ''' save only existing images '''
        all_lst_paths_img = [
                fpath_embed_umap_z1_3,
                fpath_embed_umap_z1_2,
                fpath_embed_umap_z1_1,
                fpath_embed_umap_z_1,
                fpath_embed_umap_z2_3,
                fpath_embed_umap_z2_2,
                fpath_embed_umap_z2_1,
                fpath_embed_umap_pair,
        ]

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=4, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)

    def plot_z_embed_pair_sample_id(self, es=None, figure_size=(16,16)):
        ''' set default '''
        if es is None:
            es = self.dic_best_samples['es']

        ''' filepaths for output images '''
        fpath_pair_embed_sample_id = self.dpath_anal.joinpath('plot_pair_embed_sample_id.png')
        fname_merge = 'amerge_pair_embed_sample_id_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' tensors umap-embeded '''
        e1, e2 = es

        ########################### need to be updated: gg_point_pair_embed
        g_pair_embed_sample_id = gg_point_pair_embed(e1, e2, show_text=True,
					title=self.prj_name, figure_size=figure_size)

        ''' save output image files '''
        g_pair_embed_sample_id.save(fpath_pair_embed_sample_id, verbose=False)

        ''' save only existing images '''
        all_lst_paths_img = [
        	fpath_pair_embed_sample_id
        ]

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=1, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)


    def plot_all(self):
        #xs, zss, es, pxzs, qzxs = self.dic_best_samples.values()

        xs = self.dic_best_samples['xs']
        zss = self.dic_best_samples['zss']
        es = self.dic_best_samples['es']
        pxzs = self.dic_best_samples['pxzs']
        qzxs = self.dic_best_samples['qzxs']

        print("- plotting loss")
        self.plot_loss()
        print("- plotting reconstruction")
        self.plot_recon(xs)
        print("- plotting z embedding")
        #self.plot_z_embed_all(reduction='pca')
        self.plot_z_embed_all(reduction='umap')
        #self.plot_z_embed_all(reduction='tsne')
        print("- plotting z boxplot")
        self.plot_z_boxplot(zss, qzxs)
        print("- plotting z activation")
        self.plot_z_activation(zss)
        print("- plotting feature activation")
        self.plot_features(xs, es)
        print("- plotting feature reconstruction")
        self.plot_features_recon(xs, es)
        self.show_plot()


    def plot_z_boxplot(self, zss=None, qzxs=None):
        ''' set default '''
        if zss is None or qzxs is None:
            zss = self.dic_best_samples['zss']
            qzxs = self.dic_best_samples['qzxs']

        ''' define paths for output image files ''' 
        fpath_bp_z1 = self.dpath_anal.joinpath('plot_bp_z1.png')
        fpath_bp_z2 = self.dpath_anal.joinpath('plot_bp_z2.png')
        fpath_bp_kl1 = self.dpath_anal.joinpath('plot_bp_kl1.png')
        fpath_bp_kl2 = self.dpath_anal.joinpath('plot_bp_kl2.png')
        fname_merge = 'amerge_z_bp_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' dataframe for zss '''
        df_zs1 = check_mtx_to_df(zss[0])
        df_zs2 = check_mtx_to_df(zss[1])
        zs_names = ['z'+str(i) for i in range(1, df_zs1.shape[1]+1)]

        ''' boxplot for zss '''
        df_zs1.columns = zs_names
        bp_z1 = boxplot_from_mtx(df_zs1, xlab="Zs", ylab="Activation", title=self.prj_name+'_zs1')

        df_zs2.columns = zs_names 
        bp_z2 = boxplot_from_mtx(df_zs2, xlab="Zs",
			ylab="Activation",
			title=self.prj_name+'_zs2')

        ''' boxplot for kl divergence '''
        pz = _get_pz(zss[0])
        kl1 = kl_divergence(qzxs[0], pz) ######## kl between post and prior without beta
        kl2 = kl_divergence(qzxs[1], pz) ######## kl between post and prior without beta

        df_kl1 = tensor_to_df(kl1)
        df_kl1.columns = zs_names
        bp_kl1 = boxplot_from_mtx(df_kl1, 
			xlab="Zs",
			ylab="kl(q(z|x)||p(z))",
			title=self.prj_name+'_kl_1')

        df_kl2 = tensor_to_df(kl2)
        df_kl2.columns = zs_names
        bp_kl2 = boxplot_from_mtx(df_kl2, 
			xlab="Zs",
			ylab="kl(q(z|x)||p(z))",
			title=self.prj_name+'_kl_2')

        ''' save boxplots '''
        bp_z1.save(fpath_bp_z1, verbose = False)
        bp_z2.save(fpath_bp_z2, verbose = False)
        bp_kl1.save(fpath_bp_kl1, verbose = False)
        bp_kl2.save(fpath_bp_kl2, verbose = False)

        ''' list of filepaths '''
        lst_paths_imgs = [
                fpath_bp_z1,
                fpath_bp_z2,
                fpath_bp_kl1,
                fpath_bp_kl2
        ]
        grid_display(lst_paths_imgs, ncols=2, lst_titles=[],
					figratio=1.0, filepath_out=fpath_merge)

    def plot_z_activation(self, 
        zss: Union[pd.DataFrame, np.ndarray, torch.Tensor] = None
    ):
        ''' set default '''
        if zss is None:
            zss = self.dic_best_samples['zss']

        ''' define output image file '''
        fpath_z_active = self.dpath_anal.joinpath('plot_z_active.png')
        fpath_z_active1 = self.dpath_anal.joinpath('plot_z_active1.png')
        fpath_z_active2 = self.dpath_anal.joinpath('plot_z_active2.png')
        fpath_merge = self.dpath_anal.joinpath('amerge_z_active_' + self.prj_name + '.png')

        '''dataframe for zss'''
        df_zs1 = check_mtx_to_df(zss[0])
        df_zs2 = check_mtx_to_df(zss[1])

        '''umap and plotting'''
        g_z_active1 = gg_point_z_activation(df_zs1.values, self.prj_name+'_Z1')
        g_z_active2 = gg_point_z_activation(df_zs2.values, self.prj_name+'_Z2')

        g_z_active1.save(fpath_z_active1, verbose=False)
        g_z_active2.save(fpath_z_active2, verbose=False)

        lst_paths_imgs = [
        	fpath_z_active1,
        	fpath_z_active2
        ]

        grid_display(lst_paths_imgs, ncols=1, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)

    def plot_recon(self, xs: List[pd.DataFrame]=None, top_n=10):

        def get_columns(dic_genes):
            ordered = OrderedDict()
            for ident, features in dic_genes.items():
                for feature in features:
                    ordered[feature] = 1
            columns_selected = list(ordered.keys())
            return columns_selected

        def get_index():
            x, zs, es, pxz, qzx = self.dic_best_samples.values()
            #ident = get_cell_ident_from_reference(list(x[0].index))
            bcs_target = list(x[0].index)
            ident = self.adata1[bcs_target, ].obs['ident'].tolist()

            sample_idx_sorted = np.argsort(ident)
            return sample_idx_sorted

        ''' sorted index '''
        if self.dic_degs is None:
            dic_genes_1, dic_genes_2 = self.get_DEGs_per_ident(top_n)
        else:
            dic_genes_1, dic_genes_2 = self.dic_degs

        columns_1 = get_columns(dic_genes_1)
        columns_2 = get_columns(dic_genes_2)
        index = get_index()

        ''' define paths for output image files '''
        fpath_hm_x1 = self.dpath_anal.joinpath('plot_hm_x1.png')
        fpath_hm_x2 = self.dpath_anal.joinpath('plot_hm_x2.png')
        fpath_hm_xp11 = self.dpath_anal.joinpath('plot_hm_xp11.png')
        fpath_hm_xp12 = self.dpath_anal.joinpath('plot_hm_xp12.png')
        fpath_hm_xp21 = self.dpath_anal.joinpath('plot_hm_xp21.png')
        fpath_hm_xp22 = self.dpath_anal.joinpath('plot_hm_xp22.png')
        fname_merge = 'amerge_xp_hm_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        if xs is None:
            df_x1, df_x2 = self.dic_best_samples['xs']
        else:
            ''' input cleaning '''
            df_x1, df_x2 = xs

        assert type(df_x1) == type(df_x2)
        assert isinstance(df_x1, pd.DataFrame)

        ''' do reconstruction '''
        #xs = [df_to_tensor(df_x1), df_to_tensor(df_x2)]
        #[[df_xp11,df_xp12],[df_xp21,df_xp22]] = self.reconstruct(xs)
        #[[df_xp11,df_xp12],[df_xp21,df_xp22]] = self.reconstruct([df_x1,df_x2])

        dic_recon = self.reconstruct([df_x1, df_x2])
        xp11 = dic_recon['xp1'] 
        xp12 = dic_recon['xpp2'] 
        xp21 = dic_recon['xpp1'] 
        xp22 = dic_recon['xp2'] 

        '''
        x_elbos = objective(self.params.loss_func, self.model, xs,
                        n_mc_samples=self.params.n_mc_samples, beta=self.params.beta)
        '''
        x_elbos = [345., 342.]

        #x_elbo_str = str(np.round(-x_elbos[0].item(), 2))
        x_elbo_str = '345'

        #df_xp11 = tensor_to_df(xp[0][0], index=df_x1.index, columns=df_x1.columns)
        #df_xp12 = tensor_to_df(xp[0][1], index=df_x2.index, columns=df_x2.columns)
        #df_xp21 = tensor_to_df(xp[1][0], index=df_x1.index, columns=df_x1.columns)
        #df_xp22 = tensor_to_df(xp[1][1], index=df_x2.index, columns=df_x2.columns)
        #print(df_xp22); assert False

        ''' heatmap labeling  '''
        hm_xp_kwargs = {'xlab': "Feature", 'ylab': "Sample",
                        'title': self.prj_name + '_xp_' + x_elbo_str, 'figure_size': (4,3) }

        hm_x_kwargs = {'xlab': "Feature", 'ylab': "Sample",
                        'title': self.prj_name + '_x', 'figure_size': (4,3) }

        ''' for rna, modality 1 '''
        hm_x1 = heatmap_from_mtx(df_x1[columns_1].iloc[index].values, **hm_x_kwargs)
        hm_xp11 = heatmap_from_mtx(df_xp11[columns_1].iloc[index].values, **hm_x_kwargs)
        hm_xp21 = heatmap_from_mtx(df_xp21[columns_1].iloc[index].values, **hm_x_kwargs)

        ''' for adt, modality 2 '''
        hm_x2 = heatmap_from_mtx(df_x2[columns_2].iloc[index].values, **hm_x_kwargs)
        hm_xp12 = heatmap_from_mtx(df_xp12[columns_2].iloc[index].values, **hm_x_kwargs)
        hm_xp22 = heatmap_from_mtx(df_xp22[columns_2].iloc[index].values, **hm_x_kwargs)

        ''' save images  '''
        hm_x1.save(fpath_hm_x1, verbose = False)
        hm_x2.save(fpath_hm_x2, verbose = False)
        hm_xp11.save(fpath_hm_xp11, verbose = False)
        hm_xp12.save(fpath_hm_xp12, verbose = False)
        hm_xp21.save(fpath_hm_xp21, verbose = False)
        hm_xp22.save(fpath_hm_xp22, verbose = False)

        ''' list of filepaths '''
        lst_paths_img = [
                fpath_hm_x1,
                fpath_hm_xp11,
                fpath_hm_xp21,
                fpath_hm_x2,
                fpath_hm_xp22,
                fpath_hm_xp12
        ]

        ''' list of titles '''
        lst_titles = [ ]
        grid_display(lst_paths_img, ncols=3, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)


    def plot_loss(self):

        ''' setup filepaths ''' 
        fpath_loss_all = self.dpath_anal.joinpath('plot_loss_all.png')
        fpath_loss = self.dpath_anal.joinpath('plot_loss.png')
        fpath_loss1 = self.dpath_anal.joinpath('plot_loss1.png')
        fpath_loss2 = self.dpath_anal.joinpath('plot_loss2.png')

        fname_merge = 'amerge_loss_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' loss figure all ''' 
        fig = self.df_loss.plot().get_figure()
        fig = self.df_loss.loc[:, ["train_loss", "val_loss",
        				"train_loss1", "val_loss1",
        				"train_loss2", "val_loss2" ]].plot(title="ALL").get_figure()
        fig.savefig(fpath_loss_all)
        plt.close(fig)

        ''' loss figure loss ''' 
        fig = self.df_loss.loc[:, ["train_loss", "val_loss"]].plot(title="LOSS").get_figure()
        fig.savefig(fpath_loss)
        plt.close(fig)

        ''' loss figure loss1 ''' 
        fig = self.df_loss.loc[:, ["train_loss1", "val_loss1"]].plot(title="recon").get_figure()
        fig.savefig(fpath_loss1)
        plt.close(fig)

        ''' loss figure loss2 ''' 
        fig = self.df_loss.loc[:, ["train_loss2", "val_loss2"]].plot(title="kl").get_figure()
        fig.savefig(fpath_loss2)
        plt.close(fig)

        ''' list of filepaths ''' 
        lst_paths_imgs = [
        	fpath_loss_all,
        	fpath_loss,
        	fpath_loss1,
        	fpath_loss2
        ]
 
        grid_display(lst_paths_imgs, ncols=2, lst_titles=[], 
					figratio=1.0, filepath_out=fpath_merge)

    def show_plot(self, 
        target: Literal['all', 'z_embed', 'pair', 'loss', 'recon', 'z_actvie', 'features']='all'
    ):
        fpath_merge_all = self.dpath_anal.joinpath('amerge_all_' + self.prj_name + '.png')
        fpath_merge_hm = self.dpath_anal.joinpath('amerge_xp_hm_' + self.prj_name + '.png')
        fpath_merge_loss = self.dpath_anal.joinpath('amerge_loss_' + self.prj_name + '.png')
        fpath_merge_embed = self.dpath_anal.joinpath('amerge_z_embed_' + self.prj_name + '.png')
        fpath_merge_z_active = self.dpath_anal.joinpath('amerge_z_active_' + self.prj_name + '.png')
        fpath_merge_features = self.dpath_anal.joinpath('amerge_features_' + self.prj_name + '.png')
        fpath_merge_features_recon = self.dpath_anal.joinpath('amerge_features_recon_' + self.prj_name + '.png')
        fpath_merge_features_recon_1 = self.dpath_anal.joinpath('plot_features_recon_x1_all.png')
        fpath_merge_features_recon_2 = self.dpath_anal.joinpath('plot_features_recon_x2_all.png')
        fpath_merge_bp = self.dpath_anal.joinpath('amerge_z_bp_' + self.prj_name + '.png')

        fpath_merge_embed_sample_id = self.dpath_anal.joinpath('amerge_z_embed_pair_sample_id_' + self.prj_name + '.png')

        fpath_tmp = self.dpath_anal.joinpath('tmp_' + self.prj_name + '.png')
        filepath_out = fpath_tmp

        if target == 'z_embed':
            all_lst_path_imgs = [ fpath_merge_embed ]
        elif target == 'pair':
            all_lst_path_imgs = [ fpath_merge_embed_sample_id ]
        elif target == 'loss':
            all_lst_path_imgs = [ fpath_merge_loss ]
        elif target == 'recon':
            all_lst_path_imgs = [ fpath_merge_hm ]
        elif target == 'z_active':
            all_lst_path_imgs = [ fpath_merge_z_active ]
        elif target == 'features':
            all_lst_path_imgs = [ fpath_merge_features ]
        elif target == 'all':
            all_lst_path_imgs = [
                    fpath_merge_embed,
                    fpath_merge_z_active,
                    fpath_merge_features_recon_1,
                    fpath_merge_features_recon_2,
                    fpath_merge_hm,
                    fpath_merge_bp,
                    fpath_merge_loss,
        	    fpath_merge_embed_sample_id
            ]
            filepath_out = fpath_merge_all
        else:
            raise ValueError(f"Invalid keyword: {target}")
        
        lst_path_imgs = [fpath for fpath in all_lst_path_imgs if fpath.exists()]
        
        grid_display(lst_path_imgs, ncols=1, lst_titles=[],
					figratio=1.0, filepath_out=filepath_out)


    def get_DEGs_per_ident(self, n=10, key_deg_adata_uns='deg_test'):
        self.run_DEG_test()

        ''' dictionary for degs for each ident  '''
        dic_deg_1 = self.top_n_features(self.adata1, n=n, key_deg_adata_uns=key_deg_adata_uns)
        dic_deg_2 = self.top_n_features(self.adata2, n=n, key_deg_adata_uns=key_deg_adata_uns)
        self._dic_degs = dic_deg_1, dic_deg_2

        ''' dictionary for degs_unique in each modality'''
        degs_unique_1 = lst_unique_from_dic(dic_deg_1)
        degs_unique_2 = lst_unique_from_dic(dic_deg_2)
        self._degs_unique = degs_unique_1, degs_unique_2

        return dic_deg_1, dic_deg_2

    def run_DEG_test(self, 
	key_layer='z_scaled', 
	key_obs='ident', 
	method='wilcoxon', 
	key_added='deg_test'
    ):
        assert key_obs in self.adata1.obs.keys(), \
                        f'no key defined in adata1.obs: {key_obs}, run first add_ident_to_adata'
        assert key_obs in self.adata2.obs.keys(), \
                        f'no key defined in adata2.obs: {key_obs}, run first add_ident_to_adata'
        print(f"Running DEG Test using '{key_layer}', '{key_obs}', '{method}' saved in slot '{key_added}'")

        sc.tl.rank_genes_groups(self.adata1, groupby=key_obs, 
					layer=key_layer, use_raw=False,
					method=method, key_added=key_added)

        sc.tl.rank_genes_groups(self.adata2, groupby=key_obs, 
					layer=key_layer, use_raw=False,
					method=method, key_added=key_added)


    def _transfer_input_matched(self, adata_query, m):

        if self.params.dec_model == 'gaussian':
            key_adata_layer = 'z_scaled'
        elif self.params.dec_model == 'nb':
            key_adata_layer = 'counts'
        anndata_sanity_check(adata_query, key_adata_layer)

        ''' model input matching '''
        if m == 'm1':
            adata = self.adata1
        elif m == 'm2':
            adata = self.adata2
        else:
            raise ValueError("Invalid modality keyword")
        df_query = adata_query.to_df(layer=key_adata_layer)

        df_ref = adata.to_df(layer=key_adata_layer)
        dic_matched_input = model_feature_matching(df_query, df_ref)

        return dic_matched_input


    def label_transfer(self, 
	adata_query, 
	m=None, 
	key_added_pred_z='pred_z', 
	key_added_pred_ident='pred_ident'
    ):
        """ Predict cell-type labels of new dataset 
        1) estimate latent features with respect to the query anndata(s)
        2) save latent features into slot of 'key_added_pred_z'
        3) predict cell-type based on the estimated latent features using MVGLikelihood
        4) save cell-type into slot of 'key_added_pred_ident'
        """
        both_modality = False

        # get latent features 
        if isinstance(adata_query, list) and m is None:
            both_modality = True
            df1_imputed = self._transfer_input_matched(adata_query[0], m='m1')['df_imputed']
            df2_imputed = self._transfer_input_matched(adata_query[1], m='m2')['df_imputed']
            df_zs_q = self.get_latent_features([df1_imputed, df2_imputed])['z']
        else:
            mtx_imputed = self._transfer_input_matched(adata_query, m=m)['df_imputed']
            df_zs_q = self.get_latent_features(mtx_imputed)

        # for cell-type reference
        df_ident_ref = pd.DataFrame(self.adata1.obsm['Z'])
        df_ident_ref['ident'] = self.adata1.obs['ident'].tolist()

        # predict cell-type based on reference annotation
        mvg = MVGLikelihood(df_ident_ref)
        lst_predict_ident = mvg.predict_celltype_from_z(df_zs_q)['predict'].tolist()

        # save z_mtx and ident on slots
        if both_modality:
            adata_query[0].obs[key_added_pred_ident] = lst_predict_ident
            adata_query[0].obsm[key_added_pred_z] = df_zs_q.values
            adata_query[1].obs[key_added_pred_ident] = lst_predict_ident
            adata_query[1].obsm[key_added_pred_z] = df_zs_q.values
        else:
            adata_query.obs[key_added_pred_ident] = lst_predict_ident
            adata_query.obsm[key_added_pred_z] = df_zs_q.values

        return 


    def _get_latent_either(self, x_in, save=False):
        df_x = check_mtx_to_df(x_in)
        x = df_to_tensor(df_x)

        if x.shape[1] == self.adata1.shape[1]:
            print(f"x in first modality")
            zs = self.model.get_latent_x1(x)
            columns_zs = ['z'+str(i+1) for i in range(zs.shape[1])]

        elif x.shape[1] == self.adata2.shape[1]:
            print(f"x in second modality")
            zs = self.model.get_latent_x2(x)
            columns_zs = ['z'+str(i+1) for i in range(zs.shape[1])]

        else:
            raise ValueError(f"invalid input shape: {x_in.shape}")

        if isinstance(x_in, np.ndarray):
            return zs.cpu().detach().numpy()
        elif isinstance(x_in, torch.Tensor):
            return zs
        elif isinstance(x_in, pd.DataFrame):
            return tensor_to_df(zs, index=df_x.index, columns=columns_zs)


    def _recon_either(self, x_in):

        df_x = check_mtx_to_df(x_in)
        x = df_to_tensor(df_x)

        if x.shape[1] == self.adata1.shape[1]:
            print('For first modality')
            xp, xpp = self.model.reconstruct_x1(x)
            columns_xp=self.adata1.var.index.tolist()
            columns_xpp=self.adata2.var.index.tolist()

        elif x.shape[1] == self.adata2.shape[1]:
            print('For second modality')
            xp, xpp = self.model.reconstruct_x2(x)
            columns_xp=self.adata2.var.index.tolist()
            columns_xpp=self.adata1.var.index.tolist()

        else:
            raise ValueError(f"invalid input shape: {x_in.shape}")

        dic_recon = dict()
        if isinstance(x_in, torch.Tensor):
            dic_recon['xp'] = xp
            dic_recon['xpp'] = xpp
        elif isinstance(x_in, np.ndarray):
            dic_recon['xp'] = xp.cpu().detach().numpy()
            dic_recon['xpp'] = xpp.cpu().detach().numpy()
        elif isinstance(x_in, pd.DataFrame):
            df_xp = tensor_to_df(xp, index=df_x.index, columns=columns_xp)
            df_xpp = tensor_to_df(xpp, index=df_x.index, columns=columns_xpp)
            dic_recon['xp'] = df_xp
            dic_recon['xpp'] = df_xpp

        return dic_recon

    def recon_from_z1(self, z1):
        """maybe deprecated """

        df_z1 = check_mtx_to_df(z1)
        z1 = df_to_tensor(df_z1)

        xg11, xg12 = self.model.recon_from_z1(z1)
        df_xg11 = tensor_to_df(xg11, index=df_z1.index, columns=self.adata1.var.index.tolist())
        df_xg12 = tensor_to_df(xg12, index=df_z1.index, columns=self.adata2.var.index.tolist())

        return df_xg11, df_xg12

    def recon_from_z2(self, z2):
        """maybe deprecated """

        df_z2 = check_mtx_to_df(z2)
        z2 = df_to_tensor(df_z2)

        xg22, xg21 = self.model.recon_from_z2(z2)
        df_xg22 = tensor_to_df(xg22, index=df_z2.index, columns=self.adata2.var.index.tolist())
        df_xg21 = tensor_to_df(xg21, index=df_z2.index, columns=self.adata1.var.index.tolist())

        return df_xg22, df_xg21

    def get_ident_matrix(self, df_mtx):
        ''' return new dataframe with ident 
        '''
        assert all(self.adata1.obs.index == self.adata2.obs.index)

        lst_barcodes_query = df_mtx.index.tolist()
        assert Lists.exist_all_in_lists(lst_barcodes_query, self.adata1.obs.index.tolist())

        lst_ident_query = self.adata1.obs.loc[lst_barcodes_query, 'ident'].values.tolist()

        df_ident = df_mtx.copy()
        df_ident['ident'] = pd.Categorical(lst_ident_query, ordered=True,
                                categories=sorted(self.adata1.obs.ident.unique().tolist()))
        return df_ident



    def anal_corr(self, x1='x', x2='xp',
        figure_size = (3,3),
        color = 'red',
        xlab = 'x',
        ylab = 'y',              
    ):
        df_corr_1 = self._get_corr_features(x1)
        df_corr_2 = self._get_corr_features(x2)
    
        lst_corr_1 = df_corr_1['Corr'].values
        lst_corr_2 = df_corr_2['Corr'].values

        print(gg_point_scatter(lst_corr_1, lst_corr_2,
                        figure_size=figure_size,
                        color=color,
                        xlab=xlab,
                        ylab=ylab)
        )
        return df_corr_1, df_corr_2

    def _get_pair_protein_rna(self, lst_protein=None):
    
    
        # get reference protein-RNA pair
        path = '/home/groups/precepts/chhy/kci/kci_20200321_multi_modal_vae/scvt/scripts/bioref/'
        fpath = path + 'ref_protein_gene.info'
        df_ref = pd.read_csv(fpath)  # 'protein', 'rna'
    
        if lst_protein is None:
            lst_protein = df_ref['protein'].values.tolist()
    
        dic_to_rna = dict(zip(df_ref['protein'].values, df_ref['rna'].values,))
        dic_to_pro = dict(zip(df_ref['rna'].values, df_ref['protein'].values,))
    
        # check if those RNAs are in adata1
        genes_in_adata = self.adata1.var.index.tolist()
        rna_in_ref = set(df_ref['rna'].tolist())
        lst_rna_valid = list(set(rna_in_ref).intersection(set(genes_in_adata)))
    
        lst_protein_valid = Lists.list_to_hash_value(lst_rna_valid, dic_to_pro)
    
        return lst_protein_valid, lst_rna_valid

    def _get_corr_features(self, kind='x'):
    
        def _corr(x, y):
            return scipy.stats.pearsonr(x, y)[0]

        def _corr_mtx(x, y):
            assert x.shape == y.shape

            lst_corr = []
            for i in range(x.shape[1]):
                lst_corr.append(_corr(x[:,i], y[:,i]))

            return lst_corr

        lst_pro, lst_rna = self._get_pair_protein_rna()
    
        if kind == 'x':
            x = self.adata1[:,lst_rna].layers['z_scaled']
            y = self.adata2[:,lst_pro].layers['z_scaled']
        elif kind == 'xp':
            x = self.adata1[:,lst_rna].layers['xp']
            y = self.adata2[:,lst_pro].layers['xp']
        elif kind == 'xpp':
            x = self.adata1[:,lst_rna].layers['xpp']
            y = self.adata2[:,lst_pro].layers['xpp']
    
        lst_corr = _corr_mtx(x, y)
        df_corr = pd.DataFrame()
        df_corr['protein'] = lst_pro
        df_corr['mRNA'] = lst_rna
        df_corr['Corr'] = lst_corr
    
        return df_corr

    def plot_violin(keys, m='m1', groupby='ident', layer='xp', save=None):
        if 'm1' in m:
            adata = self.adata1
        elif 'm2' in m:
            adata = self.adata2
    
        sc.pl.violin(adata, keys=keys, groupby=groupby, use_raw=False, layer=layer, size=0, save=save)


