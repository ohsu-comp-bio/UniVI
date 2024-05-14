from collections import OrderedDict
from typing import Literal, Union
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


from UniVI.utilities import _utils 
from UniVI._settings import DataPath, DataDefault, ColorIdent
from UniVI._objectives import objective
from UniVI.utilities._utils import Logger, Timer, dic2obj, dic_params_print, tensor_to_df, tensor_to_numpy, df_to_tensor, get_device, data_to_device, embed_umap, kl_divergence, check_mtx_to_df, substitute_zs_dim, lst_unique_from_dic, Lists
from UniVI.models._vae import VAE
from UniVI.models._utils import init_weights, EarlyStopping
from UniVI.models import _utils as model_utils
from UniVI.datasets._datasets import SCDataLoader, anndata_sanity_check, model_feature_matching
from UniVI.datasets._external import get_cell_ident_from_reference


import importlib
from UniVI import _objectives
from UniVI.plotting import plots
importlib.reload(_objectives)
importlib.reload(model_utils)
importlib.reload(plots)
importlib.reload(_utils)

from UniVI._objectives import elbo, objective
from UniVI.plotting.plots import grid_display, heatmap_sample_idx, heatmap_from_mtx, gg_point_ident, gg_point_embed, boxplot_from_mtx, gg_point_z_activation, gg_point_feature_active

class VAEMap(object):

    def __init__(
        self,
        dpath_home: str, 
        adata,
        params = None, 
        pretrained=False
    ):
        self._epoch_current = None
        self._dic_best_samples = None
        self._dic_degs = None


        ''' initialization ''' 
        self.adata = adata
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
    def epoch_best(self):

        return self._epoch_best
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

    @property
    def epoch_current(self):
        return self._epoch_current

    def update_to_last_epoch(self):
        ''' load last model-state-dict ''' 
        checkpoint  = torch.load(self.fpath_checkpoint, map_location=self.device)
        self._epoch_current = checkpoint['epoch']
        print(f'Loading the last model state from epoch {self.epoch_current}')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def update_to_best_epoch(self):
        ''' load best model-state-dict ''' 
        checkpoint_best = torch.load(self.fpath_checkpoint_best, map_location=self.device)
        self._epoch_best = checkpoint_best['epoch']
        print(f'\n* Loading the best model state from epoch {self.epoch_best}')
        self.model.load_state_dict(checkpoint_best['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_best['optimizer_state_dict'])

    @classmethod
    def load(cls, dpath_home):

        ''' pretrained file paths '''
        fpath_adata = Path(dpath_home).joinpath('adata.h5ad')
        fpath_params = Path(dpath_home).joinpath('params.json')
        fpath_objs = Path(dpath_home).joinpath('objs.pkl')
        fpath_objs_dic = Path(dpath_home).joinpath('objs_dic.pkl')

        ''' load adata '''
        adata = anndata.read(fpath_adata)

        ''' load params  '''
        with open(fpath_params, 'r') as fh:
            params = json.load(fh)

        ''' load attributes  '''
        with open(fpath_objs, 'rb') as f:
            epoch_current, dic_degs = pickle.load(f)

        ''' create instance ''' 
        instance = cls(dpath_home, adata, params, pretrained=True)

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

        return instance

    def save(self):
        ''' save adata '''
        self.adata.write(self.fpath_adata)

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
                self._dic_degs
        ]
        with open(self.fpath_objs, 'wb') as f: 
            pickle.dump(lst_objs, f)

        ''' save best sample attributes '''
        torch.save(self._dic_best_samples, self.fpath_objs_dic)


    def get_optimizer(self, model):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    def _init_model(self):

        self.model = VAE(**self.param_model_arch)
        self.model = self.model.to(self.device)
        init_weights(self.model)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                               lr=1e-3)

    def _init_working_dirs(self):

        ''' working home directory '''
        self.dpath_home.mkdir(parents=True, exist_ok=True)
        self.fpath_adata = Path(self.dpath_home).joinpath('adata.h5ad')
        self.fpath_params = Path(self.dpath_home).joinpath('params.json')
        self.fpath_checkpoint = Path(self.dpath_home).joinpath('checkpoint.pt')
        self.fpath_checkpoint_best = Path(self.dpath_home).joinpath('checkpoint_best.pt')
        self.fpath_objs = Path(self.dpath_home).joinpath('objs.pkl')
        self.fpath_objs_dic = Path(self.dpath_home).joinpath('objs_dic.pkl')
        self.fpath_log = Path(self.dpath_home).joinpath('run.log')
        self.fpath_loss = Path(self.dpath_home).joinpath('loss.csv')

        ''' analysis directory '''
        self.dpath_anal = Path(self.dpath_home).joinpath('analysis')
        self.dpath_anal.mkdir(parents=True, exist_ok=True)
        self.dpath_anal_feature = Path(self.dpath_anal).joinpath('feature')
        self.dpath_anal_feature.mkdir(parents=True, exist_ok=True)
        self.dpath_anal_grad = Path(self.dpath_anal).joinpath('grad')
        self.dpath_anal_grad.mkdir(parents=True, exist_ok=True)


    def _init_device(self):
        if self.params.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

    def _init_dataclass(self):

        if self.params.dec_model == "nb":
            key_adata_layer = 'counts'
        elif self.params.dec_model == "gaussian":
            key_adata_layer = 'z_scaled'
        else:
            print(f'Invalid decoder model: {self.params.dec_model}')

        self.scdl = SCDataLoader(
        		adata = self.adata,
        		key_adata_layer = key_adata_layer,
        		train_fraction = self.params.train_fraction,
        		val_fraction = self.params.val_fraction,
        		batch_size = self.params.batch_size,
        		seed = self.params.seed
        )

    def _load_default_params(self):
        with open(DataPath.DEFAULT_PARAMS, 'r') as fh:
            params = json.load(fh)
        return params

    def _init_params(self, params):

        ''' param file handling '''
        if not self.pretrained: 
            with open(self.fpath_params, 'w') as fh:
                json.dump(params, fh)

        ''' different part of parameters '''
        self.param_early_stopping = params['early_stopping']
        self.param_model_arch = params['model_arch']
        self.param_train = params['train']

        ''' print out running parameters '''
        dic_params_print(params)

        dic_params = {**self.param_train, **self.param_early_stopping, **self.param_model_arch}
        self.params = dic2obj(dic_params)

    def print_params(self):
        dic_params_print(self.params)

    def learn(self, n_epoch_new=None):

        ''' df_loss table '''
        columns_loss = ['train_loss', 'train_loss1', 'train_loss2', 
			'val_loss', 'val_loss1', 'val_loss2', 
			'test_loss', 'test_loss1', 'test_loss2']

        def _init_epoch(self, n_epoch_new):
            ''' if to restart learning ''' 
            if n_epoch_new is not None and self.epoch_current is not None:
                assert isinstance(n_epoch_new, int)
                if n_epoch_new > self.epoch_current:
                    #print(f'epoch_current: {self.epoch_current}')
                    #print(f'n_epoch_new: {n_epoch_new}')
                    is_continued = True
                else: 
                    raise ValueError("should be larger than {self.epoch_current}")
            else:
                epoch_start = 1
                n_epochs = self.params.n_epochs
                return epoch_start, n_epochs, None

            if is_continued:
                n_epochs = n_epoch_new
                epoch_start = self.epoch_current + 1
                print(f'Learning from epoch {epoch_start} to {n_epoch_new}')
                ''' load previous df_loss  '''
                
                df_loss_prev = None
                if self.fpath_loss.exists():
                    df_loss_file = pd.read_csv(self.fpath_loss, header=0, index_col=0)
                    if self.df_loss.shape[0] > 0:
                        df_loss_prev = self.df_loss \
				if self.df_loss.shape[0] > df_loss_file.shape[0] else df_loss_file
                    else:
                        df_loss_prev = df_loss_file
                else:
                    df_loss_prev = self.df_loss if self.df_loss is not None else None

            return epoch_start, n_epochs, df_loss_prev


        with Timer('LEARNING'):

            ''' init learning epochs ''' 
            epoch_start, n_epochs, df_loss= _init_epoch(self, n_epoch_new)

            ''' init earlystopping ''' 
            earlystopping = EarlyStopping(**self.param_early_stopping, 
						path=self.fpath_checkpoint_best)

            ''' per-epoch learning ''' 
            for epoch in range(epoch_start, n_epochs+1):

                self._epoch_current = epoch
    
                train_loss = self._train_per_epoch()
                print(f'epoch {epoch}: train {train_loss[0]:.3f}')

                val_loss = self._validate_per_epoch()
                print(f'epoch {epoch}: val   {val_loss[0]:.3f}')

                test_loss = self._test_per_epoch()
                print(f'epoch {epoch}: test  {test_loss[0]:.3f}')
    
                ''' loss in a row '''
                arow_loss = [*train_loss, *val_loss, *test_loss]

                ''' df_loss '''
                if epoch == 1:
                    df_loss = pd.DataFrame([arow_loss])
                    df_loss.columns = list(columns_loss)
                    self.df_loss = df_loss
                else:
                    ''' per epoch loss in a row '''
                    df_loss_epoch = pd.DataFrame([arow_loss])
                    df_loss_epoch.columns = list(columns_loss)
                    ''' append to existing df '''
                    df_loss = df_loss.append(df_loss_epoch)
                    df_loss.index = list(range(1, df_loss.shape[0]+1))
                    self.df_loss = df_loss

                ''' save df_loss to a file '''
                self.df_loss.to_csv(self.fpath_loss, index=True)

                ''' check if early-stopping is happening '''
                earlystopping.__call__(val_loss[0], epoch, self.model, self.optimizer)
                if np.isnan(train_loss[0]) or earlystopping.early_stop:
                    self._epoch_best = earlystopping.epoch_best
                    print(f'Early stopping at {epoch}/{self.params.n_epochs} => best at {self._epoch_best}')
                    break


    def _train_per_epoch(self):
    
        self.model.train()

        n_iter = len(self.scdl.dataloaders['train'])
        n_samples = len(self.scdl.idx_train)
    
        loss_sum = 0
        loss1_sum = 0
        loss2_sum = 0
    
        for i, x in enumerate(self.scdl.dataloaders['train']):
            x = data_to_device(x, self.device)

            self.optimizer.zero_grad()
            loss_batch = objective(self.params.loss_func, self.model, x, 
				beta=self.params.beta)
            loss_batch[0].backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2) 

            self.optimizer.step()
        
            loss_sum += loss_batch[0].item()
            loss1_sum += loss_batch[1].item()
            loss2_sum += loss_batch[2].item()
        
        #abbb
        #loss_epoch = loss_sum / n_samples
        #loss1_epoch = loss1_sum / n_samples
        #loss2_epoch = loss2_sum / n_samples
        loss_epoch = loss_sum / n_iter
        loss1_epoch = loss1_sum / n_iter
        loss2_epoch = loss2_sum / n_iter
        #print(f'{self.epoch_current}: train {loss_epoch:.3f}') # abby
    
        return (loss_epoch, loss1_epoch, loss2_epoch)    


    def _validate_per_epoch(self):
    
        n_iter = len(self.scdl.dataloaders['val'])
        n_samples = len(self.scdl.idx_val)
        loss_sum = 0
        loss1_sum = 0
        loss2_sum = 0
    
        self.model.eval()
        with torch.no_grad():
            for i, x in enumerate(self.scdl.dataloaders['val']):
                x = data_to_device(x, self.device)
            
                loss_batch = objective(self.params.loss_func, self.model, x, 
				beta=self.params.beta)

                #print(f'     - val_per_epoch.data {i}: {loss_batch[0].data}')
                #print(f'     - val_per_epoch.grad {i}: {loss_batch[0].grad}')
                #print(f'\n')
                loss_sum += loss_batch[0].item()
                loss1_sum += loss_batch[1].item()
                loss2_sum += loss_batch[2].item()

        #abbb
        #loss_epoch = loss_sum / n_samples
        #loss1_epoch = loss1_sum / n_samples
        #loss2_epoch = loss2_sum / n_samples
        loss_epoch = loss_sum / n_iter
        loss1_epoch = loss1_sum / n_iter
        loss2_epoch = loss2_sum / n_iter
        #print(f'{self.epoch_current}: val, {loss_epoch:.3f}') # abby

        return (loss_epoch, loss1_epoch, loss2_epoch)    


    def _test_per_epoch(self):
    
        n_iter = len(self.scdl.dataloaders['test'])
        n_samples = len(self.scdl.idx_test)
        loss_sum = 0
        loss1_sum = 0
        loss2_sum = 0
        
        self.model.eval()
        with torch.no_grad():
            for i, x in enumerate(self.scdl.dataloaders['val']):
                x = data_to_device(x, self.device)
                
                loss_batch = objective(self.params.loss_func, self.model, x, 
				beta=self.params.beta)

                #print(f'     - test_per_epoch.data {i}: {loss_batch[0].data}')
                #print(f'     - test_per_epoch.grad {i}: {loss_batch[0].grad}')
                #print(f'\n')

                loss_sum += loss_batch[0].item()
                loss1_sum += loss_batch[1].item()
                loss2_sum += loss_batch[2].item()

                if i == 0:
                    #self.reconstruct(x, save=True)
                    self.get_latent_features(x, save=False)
    
        #abbb
        #loss_epoch = loss_sum / n_samples
        #loss1_epoch = loss1_sum / n_samples
        #loss2_epoch = loss2_sum / n_samples
        loss_epoch = loss_sum / n_iter
        loss1_epoch = loss1_sum / n_iter
        loss2_epoch = loss2_sum / n_iter
        #print(f'{self.epoch_current}: test, {loss_epoch:.3f}') # abby

        return (loss_epoch, loss1_epoch, loss2_epoch)    

    def embed(self, 
	data: Union[pd.DataFrame, np.ndarray, torch.Tensor], reduction="umap", 
	n_neighbors=20
    ) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = tensor_to_numpy(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        assert isinstance(data, np.ndarray)
        if reduction == "umap":
            return embed_umap(data, n_neighbors, self.params.seed)
        else:
            raise ValueError(f"Invalid embedding method: {reduction}")



    def generate_with_zs(self,
                zs,
                kind: Literal["np", "tensor", "df"]="np",
                save=False):

        df_zs = check_mtx_to_df(zs)
        xg = self.model.generate_with_zs(df_to_tensor(df_zs))

        if kind == 'np':
            xg = tensor_to_numpy(xg)
        elif kind == 'df':
            xg = tensor_to_df(xg)
        elif kind == 'tensor':
            xg = xg
        else:
            raise ValueError(f"Invalid data type: {kind}")
        return xg


    def generate(self, 
                lst_idx_zeros=[],
    		kind: Literal["np", "tensor", "df"]="np",
                sample_size=100,
		save=False):

        if save:
            x_gen = self.model.generate(lst_idx_zeros, sample_size)
            tensor_to_df(x_gen).to_csv(
                self._filename_epoch(self.dpath_anal,
                                        "mtx_xg_", self.epoch_current), index=True)
        else:
            xg = self.model.generate(lst_idx_zeros, sample_size)
            if kind == 'np':
                xg = tensor_to_numpy(xg)
            elif kind == 'df':
                xg = tensor_to_df(xg)
            elif kind == 'tensor':
                xg = xg
            else:
                raise ValueError(f"Invalid data type: {kind}")
            return xg

    def get_latent_features(self, x_in: Union[np.ndarray, torch.Tensor, pd.DataFrame], save=False):

        df_x = check_mtx_to_df(x_in)
        x = df_to_tensor(df_x)
        zs = self.model.get_latent_features(x)

        if save:
            tensor_to_df(zs).to_csv(
                self._filename_epoch(self.dpath_anal,
                                        "mtx_z", self.epoch_current), index=True)
            return

        # return dic for consistency with MMVAE
        if isinstance(x_in, torch.Tensor):
            result = {'z': zs }
        elif isinstance(x_in, np.ndarray):
            result = {'z': zs.cpu().numpy() }
        elif isinstance(x_in, pd.DataFrame):
            columns_zs = ['z'+str(i+1) for i in range(zs.shape[1])]
            df_zs = tensor_to_df(zs, index=df_x.index, columns=columns_zs)
            result = {'z': df_zs }
        return result


    def reconstruct(self, x: Union[pd.DataFrame, np.ndarray, torch.Tensor]=None, save=False):

        if x is None:
            #print('im here'); assert False
            df_x = self.dic_best_samples['x']
        else:
            df_x = check_mtx_to_df(x)

        x = df_to_tensor(df_x)
        xp = self.model.reconstruct(x)
        df_xp = tensor_to_df(xp)
        df_xp.index = df_x.index
        df_xp.columns = df_x.columns

        if save:
            df_xp.to_csv(
		self._filename_epoch(self.dpath_anal, 
					"mtx_xp", self.epoch_current), index=True)
            #self._save_layers_to_file(layers)
        else:
            return df_xp

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


    def plot_features_recon(self, x=None, es=None, lst_feature_names=None):

        ''' default input '''
        if x is None or es is None:
            x = self.dic_best_samples['x']
            es = self.dic_best_samples['es']

        ''' do reconstruction '''
        xp = self.reconstruct(x)

        ''' list of degs '''
        if lst_feature_names is None:
            #lst_degs = self.dic_degs['Mono']
            lst_degs = self.degs_unique
        else:
            lst_degs = lst_feature_names

        ''' for modality 1 '''
        g_features_recon_x = self.plot_features_one(x, es, lst_degs, color='red')  # original
        g_features_recon_xp = self.plot_features_one(xp, es, lst_degs, color='red')  # reconstruction

        ''' define image file names '''
        fpath_features_recon_x = self.dpath_anal.joinpath('plot_features_recon_x.png')
        fpath_features_recon_xp = self.dpath_anal.joinpath('plot_features_recon_xp.png')

        fname_merge = 'amerge_features_recon_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' save images '''
        g_features_recon_x.save(fpath_features_recon_x, verbose=False)
        g_features_recon_xp.save(fpath_features_recon_xp, verbose=False)

        ''' list of filepaths '''
        lst_paths_img = [
                fpath_features_recon_x,
                fpath_features_recon_xp
        ]
        grid_display(lst_paths_img, ncols=1, lst_titles=['Original', 'Reconstructed'],
                                        figratio=1.0, filepath_out=fpath_merge)


    def plot_features_compare(self, lst_features):
        assert isinstance(lst_features, list), f"features should be in a list"
        ncols=len(lst_features)
        fpath_merge = self.dpath_anal.joinpath('plot_umap_amerge.png')
        all_lst_paths_img = []
        for key_score in ['counts', 'xp']:
            for feature in lst_features:

                fname = 'plot_feature_' + feature + '_' + key_score + '.png'
                fpath_feature = self.dpath_anal.joinpath(fname)

                g_feature = self.plot_feature(feature, key_score=key_score)

                if g_feature is None:
                    raise ValueError(f"invalid feature: {feature} in {key_score}")

                g_feature.save(fpath_feature, verbose=False)
                all_lst_paths_img.append(fpath_feature)

                ''' save only existing images '''
                lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=ncols, lst_titles=[],
                                                figratio=1.0, filepath_out=fpath_merge)


    def plot_features(self, lst_features, key_score='counts', ncols=1):

        assert isinstance(lst_features, list), f"features should be in a list"
        fpath_merge = self.dpath_anal.joinpath('plot_umap_amerge.png')

        all_lst_paths_img = []

        for feature in lst_features:
            fname = 'plot_feature_' + feature + '_' + key_score + '.png'
            fpath_feature = self.dpath_anal.joinpath(fname)

            g_feature = self.plot_feature(feature, key_score=key_score)
            if g_feature is None:
                continue

            g_feature.save(fpath_feature, verbose=False)
            all_lst_paths_img.append(fpath_feature)

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=ncols, lst_titles=[],
                                                figratio=1.0, filepath_out=fpath_merge)


    def plot_feature(self, feature, key_score='lib_normed_log1p', figure_size=(2,2)):

        # umap coordinates
        try:
            mtx_umap = self.adata.obsm['X_umap']  # Z_umap for multi-modal 
        except:
            raise ValueError(f"not set in .obsm['Z_umap']. try .save_latent_features() first")

        # select modality first 
        adata_score = self.adata

        # for high-dimensional features 
        if key_score in ['counts', 'lib_normed_log1p', 'z_scaled', 'xp']:

            # for labeling
            if key_score in ['counts', 'lib_normed_log1p', 'z_scaled']:
                data_type = 'x'
            elif key_score == 'xp':
                data_type = 'xp'

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
        elif key_score in ['Z']:
            #m = 'Z'
            feature = feature.upper()
            assert feature.startswith('Z')
            nth = int(feature.split('Z')[1])
            try:
                score = self.adata.obsm[key_score][:, nth-1]  # only in adata
                score = np.expand_dims(score, 1)
            except:
                raise ValueError(f"no info for adata.obsm[{key_score}]")

        else:
            raise ValueError(f"invalid key_score: {key_score}")

        title = feature + '_' + data_type

        return gg_point_feature_active(mtx_umap, score, title=title, figure_size=figure_size)







    def plot_umap(self, lst_key_obs=['ident'], ncols=1):

        """ wrapper for plot_umap_combined('ident') """
        fpath_merge = self.dpath_anal.joinpath('plot_umap_amerge.png')

        all_lst_paths_img = []
        for key_obs in lst_key_obs:
            fname = 'plot_umap_' + key_obs + '.png'
            fpath_key_obs = self.dpath_anal.joinpath(fname)

            g_key_obs = self._plot_umap(key_obs)

            g_key_obs.save(fpath_key_obs, verbose=False)
            all_lst_paths_img.append(fpath_key_obs)

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=ncols, lst_titles=[],
                                            figratio=1.0, filepath_out=fpath_merge)

    def _plot_umap(self,
            key_obs='ident',
            figure_size=(3,3)
    ):
        try:
            lst_ident = self.adata.obs[key_obs].tolist()
        except:
            raise ValueError(f"no key defined in .obs: {key_obs}")

        try:
            colors = self._get_ident_colors(key_obs)
            #colors = self.adata.uns[key_obs+'_colors']
        except:
            print(f"no colors defined in .obsm: {key_obs}_colors")
            colors = None

        X_umap = self.adata.obsm['X_umap']
        title = key_obs
        g = gg_point_embed(X_umap, lst_ident, colors=colors,
                        figure_size=figure_size, title=title)
        return g


    def save_recon_features(self):
        x = self.adata.layers['z_scaled']
        xp = self.reconstruct(x)
        self.adata.layers['xp'] = xp


    def save_latent_features(self):
        """ Generate slots
        self.adata.obsm['Z']
        self.adata.obsm['X_umap']
        """

        ''' load the best model '''
        self.update_to_best_epoch()

        ''' get samples '''
        X = self.adata.X

        ''' get latent features'''
        dic_Z = self.get_latent_features(X)

        ''' save to the slots '''
        # save Zs
        self.adata.obsm['Z'] = dic_Z['z']

        # for Z1
        sc.pp.neighbors(self.adata, use_rep='Z')
        sc.tl.umap(self.adata) # saved in self.adata.obsm['X_umap']


    def get_sample_best_run(self, kind: Literal["np", "tensor", "df"]="np", reduction='umap'):
        self.update_to_best_epoch()

        ''' get sample annotation ''' 
        df_x = self.get_sample_data("df")

        ''' run model ''' 
        x = df_to_tensor(df_x)
        qzx, pxz, zs = self.model(x)
        zs = zs.squeeze()
        df_zs = tensor_to_df(zs)
        df_zs.columns = ['z'+str(i) for i in range(1, zs.shape[1]+1)]
        df_zs.index = df_x.index

        ''' embedding by umap ''' 
        es = self.embed(zs, reduction=reduction)
        df_es = pd.DataFrame(es, columns=["dim1", "dim2"], index=df_x.index)

        ''' format output ''' 
        if(kind == "np"):
            x = tensor_to_numpy(x)
            zs = tensor_to_numpy(zs)
        elif(kind == "df"):
            x = df_x
            zs = df_zs
            es = df_es
        else:
            raise ValueError(f"Invalid data type : {kind}")

        ''' attributes for best sample run for plotting reproducility '''
        self._dic_best_samples = {'x': df_x, 'zs': df_zs, 'es': df_es, 'pxz': pxz, 'qzx': qzx }

        print(f'\n- Project ID: {self.prj_name}')
        return x, zs, es, pxz, qzx


    def get_sample_data(self, kind: Literal["np", "tensor", "df"]="np"):

        x = list(self.scdl.dataloaders['test'])[0]

        if kind == "np":
            return x.data.numpy()
        elif kind == "tensor":
            x = x.to(self.device)
            return x
        elif kind == "df":
            ''' get annotation from anndata object ''' 
            obs_index = list(self.adata[self.scdl.idx_test[:self.params.batch_size]].obs.index)
            df_sample = pd.DataFrame(tensor_to_numpy(x), index=obs_index, columns=self.adata.var.index)
            return df_sample
        else:
            raise ValueError("Invalid data type: {kind}")

    ''' not finished '''
    def plot_z_embed_off(self, zs, lst_idx_zeros):
        zs_off = substitute_zs_dim(zs, lst_idx_zeros, value=0, kind_output="np")
        print(zs_off.shape)
        return self.plot_z_embed(zs_off)


    def plot_recon_variation(self, zs):

        fname_merge = 'amerge_recon_variation_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        n_zs = zs.shape[1]
        value_replace = 0
        lst_path_imgs = []

        for i in range(n_zs):
            for j in range(n_zs):
                zs_replace = substitute_zs_dim(zs, [i,j], value_replace)
                xg = self.generate_with_zs(zs_replace)
                g = heatmap_from_mtx(xg[:50,:50], legend_position="none")
        
                fname_recon_variation_i = 'plot_recon_variation_v' + str(value_replace) + '_'  +str([i,j])+'.png' 
                fpath_recon_variation_i = self.dpath_anal.joinpath(fname_recon_variation_i)
                g.save(fpath_recon_variation_i, verbose=False)
                lst_path_imgs.append(fpath_recon_variation_i)

        grid_display(lst_path_imgs, ncols=10, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)


    def plot_z_embed_off_ident_all(self, zs=None, key_ident='ident'):
        ''' set default '''
        if zs is None:
            zs = self.dic_best_samples['zs']

        fname_merge = 'amerge_z_embed_off_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        lst_path_imgs = []
        for i in range(zs.shape[1]):
            fpath_z_embed_off_i = self.dpath_anal.joinpath('plot_z_embed_off_' + str(i+1) + '_' + key_ident + '.png')
            g_z_embed_off = self.plot_z_embed_off_ident(zs, [i], key_ident)
            g_z_embed_off.save(fpath_z_embed_off_i, verbose=False)
            lst_path_imgs.append(fpath_z_embed_off_i)

        grid_display(lst_path_imgs, ncols=10, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)


    def plot_z_embed_off_ident(self, zs, lst_idx_zeros, key_ident='ident'):
        assert isinstance(zs, pd.DataFrame), "should be dataframe with barcode information"
        zs_off = substitute_zs_dim(zs, lst_idx_zeros, value=0, kind_output="df")
        title = '~' + str(['z'+str(i+1) for i in lst_idx_zeros])
        return self.plot_z_embed_ident(zs_off, key_ident, title)
        

    def _get_ident_colors(self, key_ident):

        if 'hao' in self.prj_name:
            if key_ident == 'ident' or key_ident == 'level1':
                colors = ColorIdent.ident_hao
            elif key_ident == 'level2':
                colors = ColorIdent.level2_hao
            elif key_ident == 'level3':
                colors = ColorIdent.level3_hao

        elif 'atac' in self.prj_name:
            if key_ident == 'ident':
                colors = ColorIdent.ident_atac

        else:
            print(f"no color key defined yet for {key_ident}")
            colors = None

        return colors

    def plot_z_embed_ident(self, zs=None, key_ident='ident', title=None):
        ''' set default '''
        if zs is None:
            zs = self.dic_best_samples['zs']

        assert isinstance(zs, pd.DataFrame), "should be dataframe with barcode information"
        ''' get ident for each barcode from reference '''
        lst_barcodes = list(zs.index)
        #ident = get_cell_ident_from_reference(lst_barcodes, data_id='cit_hao', str_level=key_ident)
        #ident = self.adata[lst_barcodes, ].obs['ident'].tolist()
        ident = self.adata[lst_barcodes, ].obs[key_ident].tolist() 

        colors = self._get_ident_colors(key_ident)

        if title is None:
            title = self.prj_name + '_' + key_ident

        embeded = self.embed(zs)
        g_embeded = gg_point_embed(embeded, lst_ident=ident, title=title)
        return g_embeded

    def plot_z_embed_all(self, zs=None):
        ''' set default '''
        if zs is None:
            zs = self.dic_best_samples['zs']

        ''' filepaths for output images '''
        fpath_embed_umap_1 = self.dpath_anal.joinpath('plot_z_embed_umap_ident.png')
        fpath_embed_umap_2 = self.dpath_anal.joinpath('plot_z_embed_umap_level2.png')
        fpath_embed_umap_3 = self.dpath_anal.joinpath('plot_z_embed_umap_level3.png')
        fname_merge = 'amerge_z_embed_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' embedding and plotting '''
        g_z_embed_umap_ident = self.plot_z_embed_ident(zs, key_ident='ident')
        g_z_embed_umap_level2 = self.plot_z_embed_ident(zs, key_ident='level2')
        g_z_embed_umap_level3 = self.plot_z_embed_ident(zs, key_ident='level3')

        g_z_embed_umap_ident.save(fpath_embed_umap_1, verbose=False)
        g_z_embed_umap_level2.save(fpath_embed_umap_2, verbose=False)
        g_z_embed_umap_level3.save(fpath_embed_umap_3, verbose=False)

        all_lst_paths_img = [
        	fpath_embed_umap_3,
        	fpath_embed_umap_2,
        	fpath_embed_umap_1
        ]

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=3, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)

    def plot_all(self, top_n=10):
        #x, zs, es, pxz, qzx = self.dic_best_samples.values()

        x = self.dic_best_samples['x']
        zs = self.dic_best_samples['zs']
        es = self.dic_best_samples['es']
        pxz = self.dic_best_samples['pxz']
        qzx = self.dic_best_samples['qzx']

        print("- plotting loss")
        self.plot_loss() # done
        print("- plotting reconstruction")
        self.plot_recon(x, top_n)
        print("- plotting z embedding")
        self.plot_z_embed_all(zs)
        #print("- plotting z boxplot")
        #self.plot_z_boxplot(zs, qzx)
        print("- plotting z activation")
        self.plot_z_activation(zs)
        print("- plotting feature activation")
        self.plot_features(x, es)
        print("- plotting feature reconstruction")
        self.plot_features_recon(x, es)
        #self.plot_z_embed_off_ident_all(zs)
        self.show_plot()

    def plot_z_boxplot(self, 
	zs: Union[pd.DataFrame, np.ndarray, torch.Tensor],
	qzx = None
    ):
        ''' define paths for output image files ''' 
        fpath_bp_z = self.dpath_anal.joinpath('plot_bp_z.png')
        fpath_bp_kl = self.dpath_anal.joinpath('plot_bp_kl.png')
        fname_merge = 'amerge_z_bp_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' dataframe for zs '''
        df_zs = check_mtx_to_df(zs)
        zs_names = ['z'+str(i) for i in range(1, df_zs.shape[1]+1)]

        ''' boxplot for zs '''
        df_zs.columns = zs_names
        bp_z = boxplot_from_mtx(df_zs, 
			xlab="Zs",
			ylab="Activation",
			title=self.prj_name+'_zs'
		)
        bp_z.save(fpath_bp_z, verbose = False)

        ''' list of filepaths '''
        lst_paths_imgs = [ fpath_bp_z ]

        if qzx is not None:
            ''' boxplot for kl divergence '''
            pz = _get_pz(zs)
            kl = kl_divergence(qzx, pz) ######## kl between post and prior without beta
            df_kl = tensor_to_df(kl)
            df_kl.columns = zs_names
            bp_kl = boxplot_from_mtx(df_kl, 
			    xlab="Zs",
			    ylab="kl(q(z|x)||p(z))",
			    title=self.prj_name+'_kl'
		    )
            bp_kl.save(fpath_bp_kl, verbose = False)
            lst_paths_imgs.append(fpath_bp_kl)

        ''' list of filepaths '''
        grid_display(lst_paths_imgs, ncols=2, lst_titles=[],
					figratio=1.0, filepath_out=fpath_merge)

    def plot_z_activation(self,
	zs: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ):
        ''' define output image file '''
        fpath_z_active = self.dpath_anal.joinpath('plot_z_active.png')
        fpath_merge = self.dpath_anal.joinpath('amerge_z_active_' + self.prj_name + '.png')

        '''umap and plotting'''
        df_zs = check_mtx_to_df(zs)
        g_z_active = gg_point_z_activation(df_zs.values)

        g_z_active.save(fpath_z_active, verbose=False)

        lst_paths_imgs = [
                fpath_z_active
        ]

        grid_display(lst_paths_imgs, ncols=1, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)

    def plot_z_embed(self, zs, lst_ident=None, categories=None, title=None):
        g_embeded = gg_point_embed(zs, lst_ident, categories, title)
        ''' ggplot '''
        return g_embeded


    def plot_recon(self, x=None, top_n=10):
        def get_index():
            x, zs, es, pxz, qzx = self.dic_best_samples.values()
            #ident = get_cell_ident_from_reference(list(x.index))
            bcs_target = list(x.index)
            ident = self.adata[bcs_target, ].obs['ident'].tolist()
            sample_idx_sorted = np.argsort(ident)
            return sample_idx_sorted

        ''' sorted index '''
        if self.dic_degs is None:
            dic_genes = self.get_DEGs_per_ident(top_n)
        else:
            dic_genes = self.dic_degs

        columns = lst_unique_from_dic(dic_genes)
        index = get_index()

        ''' define paths for output image files '''
        fpath_hm_x = self.dpath_anal.joinpath('plot_hm_x.png')
        fpath_hm_xp = self.dpath_anal.joinpath('plot_hm_xp.png')
        fname_merge = 'amerge_xp_hm_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' input sanity checking '''
        if x is None:
            df_x = self.dic_best_samples['x']
        else:
            df_x = x

        assert isinstance(df_x, pd.DataFrame)

        ''' do reconstruction '''
        x = df_to_tensor(df_x)
        df_xp = self.reconstruct(x.float().to(self.device))
        df_xp.index = df_x.index
        df_xp.columns = df_x.columns

        ''' calculate elbo '''
        x_elbos = objective(self.params.loss_func, self.model, x,
                        beta=self.params.beta)

        x_elbo_str = str(np.round(-x_elbos[0].item(), 2))

        hm_xp_kwargs = {'xlab': "Feature", 'ylab': "Sample",
                        'title': self.prj_name + '_xp_' + x_elbo_str, 'figure_size': (4,3) }

        hm_x_kwargs = {'xlab': "Feature", 'ylab': "Sample",
                        'title': self.prj_name + '_x', 'figure_size': (4,3) }

        ''' sort for plotting '''
        hm_x = heatmap_from_mtx(df_x[columns].iloc[index].values, **hm_x_kwargs)
        hm_xp = heatmap_from_mtx(df_xp[columns].iloc[index].values, **hm_xp_kwargs)

        hm_x.save(fpath_hm_x, verbose = False)
        hm_xp.save(fpath_hm_xp, verbose = False)

        ''' list of filepaths '''
        lst_paths_img = [
                fpath_hm_x,
                fpath_hm_xp
        ]

        ''' list of titles '''
        lst_titles = [
                self.prj_name + '_x',
                self.prj_name + '_xp_' + x_elbo_str
        ]

        grid_display(lst_paths_img, ncols=2, lst_titles=[],
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
	target: Literal['all', 'z_embed', 'loss', 'recon', 'z_actvie', 'features']='all'
    ):
        fpath_merge_all = self.dpath_anal.joinpath('amerge_all_' + self.prj_name + '.png')
        fpath_merge_hm = self.dpath_anal.joinpath('amerge_xp_hm_' + self.prj_name + '.png')
        fpath_merge_loss = self.dpath_anal.joinpath('amerge_loss_' + self.prj_name + '.png')
        fpath_merge_embed = self.dpath_anal.joinpath('amerge_z_embed_' + self.prj_name + '.png')
        fpath_merge_embed_off = self.dpath_anal.joinpath('amerge_z_embed_off_' + self.prj_name + '.png')
        fpath_merge_z_active = self.dpath_anal.joinpath('amerge_z_active_' + self.prj_name + '.png')
        fpath_merge_features = self.dpath_anal.joinpath('amerge_features_' + self.prj_name + '.png')
        fpath_merge_features_recon = self.dpath_anal.joinpath('amerge_features_recon_' + self.prj_name + '.png')
        fpath_merge_bp = self.dpath_anal.joinpath('amerge_z_bp_' + self.prj_name + '.png')
        fpath_merge_recon_variation = self.dpath_anal.joinpath('amerge_recon_variation_' + self.prj_name + '.png')

        fpath_tmp = self.dpath_anal.joinpath('tmp_' + self.prj_name + '.png')
        filepath_out = fpath_tmp

        if target == 'z_embed':
            all_lst_path_imgs = [ fpath_merge_embed ]
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
                    fpath_merge_embed_off,
                    fpath_merge_z_active,
                    fpath_merge_features_recon,
        	    fpath_merge_hm,
        	    fpath_merge_bp,
        	    fpath_merge_loss,
                    fpath_merge_recon_variation
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
        self._dic_degs = self.top_n_features(self.adata, n=n, key_deg_adata_uns=key_deg_adata_uns)
        ''' dictionary for degs_unique in each modality'''
        self._degs_unique = lst_unique_from_dic(self._dic_degs)
        return self._dic_degs


    def run_DEG_test(self, 
        key_layer='z_scaled', 
        key_obs='ident', 
        method='wilcoxon', 
        key_added='deg_test'
    ):
        assert key_obs in self.adata.obs.keys(), \
                        f'no key defined in adata1.obs: {key_obs}, run first add_ident_to_adata'
        print(f"Running DEG Test using")
        print(f"{key_layer}, {key_obs}, {method}\nsaved in slot {key_added}")

        sc.tl.rank_genes_groups(self.adata, groupby=key_obs, 
                                        layer=key_layer, use_raw=False,
                                        method=method, key_added=key_added)


    def top_n_features(self, adata, n=10, key_deg_adata_uns='deg_test'):
        assert key_deg_adata_uns in adata.uns.keys(), f'no key in adata.uns: {key_adata_uns}, run first runDEGTest'
        dic_top_n = dict()

        np_record = adata.uns[key_deg_adata_uns]['names']
        dic_ident = {name:np_record[name] for name in np_record.dtype.names}
        for ident in dic_ident.keys():
            ranked_features = adata.uns[key_deg_adata_uns]['names'][ident]
            top_n = ranked_features[:n]
            dic_top_n[ident] = list(top_n)
        return dic_top_n

    def transfer_input_matched(self, adata_query):
    
        if self.params.dec_model == 'gaussian':
            key_adata_layer = 'z_scaled'
        elif self.params.dec_model == 'nb':
            key_adata_layer = 'counts'
        anndata_sanity_check(adata_query, key_adata_layer)
    
        ''' model input matching '''
        df_query = adata_query.to_df(layer=key_adata_layer)
        df_ref = self.adata.to_df(layer=key_adata_layer)
        dic_matched_input = model_feature_matching(df_query, df_ref)
    
        return dic_matched_input

    def transfer_learning(self, adata_query):
        dic_matched_input = self.transfer_input_matched(adata_query)
        df_zs = self.get_latent_features(dic_matched_input['df_imputed'].values)
        lst_ident = None
        if 'ident' in adata_query.obs:
            lst_ident=adata_query.obs.ident.values
        g_z_embed = gg_point_embed(df_zs, lst_ident=lst_ident)
        print(g_z_embed)


    def get_ident(self, lst_barcodes, adata_ref):
        ''' temporaray: is it necessary?? '''
        return adata_ref.obs.loc[lst_barcodes, 'ident'].values.tolist()

    def get_ident_matrix(self, df_mtx):
        ''' return new dataframe with ident 
        '''
        lst_barcodes_query = df_mtx.index.tolist()
        assert Lists.exist_all_in_lists(lst_barcodes_query, self.adata.obs.index.tolist())

        lst_ident_query = self.adata.obs.loc[lst_barcodes_query, 'ident'].values.tolist()

        df_ident = df_mtx.copy()
        df_ident['ident'] = pd.Categorical(lst_ident_query, ordered=True,
                                categories=sorted(self.adata.obs.ident.unique().tolist()))
        return df_ident


