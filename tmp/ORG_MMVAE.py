from typing import Literal, Union, List
from pathlib import Path
import anndata

from plotnine import *
from plotnine.data import *

from sklearn.cluster import SpectralBiclustering

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import json
import re

import torch
from torch import optim
from torch import nn

from scvt._settings import DataPath, DataDefault
from scvt._objectives import objective
from scvt.utilities._utils import Logger, Timer, dic2obj, dic_params_print, tensor_to_df, tensor_to_numpy, df_to_tensor, get_device, embed_umap, kl_divergence, check_mtx_to_df
from scvt.models._vae import VAE
from scvt.models._mmvae import MMVAE
from scvt.models._utils import init_weights, EarlyStopping
from scvt.models import _utils as model_utils
from scvt.datasets._datasets import SCPairedDataLoader
from scvt.datasets._external import get_cell_ident_from_reference

from scvt.external.evaluate_FOSCTTM import calc_frac

import importlib
from scvt import _objectives
from scvt.plotting import plots
importlib.reload(_objectives)
importlib.reload(model_utils)
importlib.reload(plots)

from scvt._objectives import elbo, iwae, objective, _get_pz
from scvt.plotting.plots import grid_display, heatmap_sample_idx, heatmap_from_mtx, gg_point_embed, gg_point_embed_pair, boxplot_from_mtx, gg_point_z_activation, gg_point_feature_active

class CrossMap(object):

    def __init__(
        self,
        dpath_home: str, 
        adata1,
        adata2,
        params = None, 
        pretrained=False
    ):
        self._epoch_current = None
        self.dic_samples = None
        self.FOSCTTM = None

        ''' initialization ''' 
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

    '''
    @property
    def FOSCTTM(self):
        return self.FOSCTTM
    '''

    @property
    def epoch_best(self):
        return self._epoch_best

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
        fpath_adata1 = Path(dpath_home).joinpath('adata1.h5ad')
        fpath_adata2 = Path(dpath_home).joinpath('adata2.h5ad')
        fpath_params = Path(dpath_home).joinpath('params.json')

        ''' load adata '''
        adata1 = anndata.read(fpath_adata1)
        adata2 = anndata.read(fpath_adata2)

        ''' load params  '''
        with open(fpath_params, 'r') as fh:
            params = json.load(fh)

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

    def _init_model(self):

        self.model1 = VAE(**self.param_model_arch_1)
        self.model2 = VAE(**self.param_model_arch_2)
        self.model = MMVAE(self.model1, self.model2)
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
        self.fpath_log = Path(self.dpath_home).joinpath('run.log')
        self.fpath_loss = Path(self.dpath_home).joinpath('loss.csv')

        ''' analysis directory '''
        self.dpath_anal = Path(self.dpath_home).joinpath('anal')
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
            key_adata_layer = 'count'
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
        dic_params_print(params)

        dic_params = {**self.param_train, **self.param_early_stopping, 
				**self.param_model_arch_1, **self.param_model_arch_2}
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


        with Timer('LEARNING Multi-Modal'):

            ''' init learning epochs ''' 
            epoch_start, n_epochs, df_loss= _init_epoch(self, n_epoch_new)

            ''' init earlystopping ''' 
            earlystopping = EarlyStopping(**self.param_early_stopping, 
						path=self.fpath_checkpoint_best)

            ''' per-epoch learning ''' 
            for epoch in range(epoch_start, n_epochs+1):

                self._epoch_current = epoch
    
                train_loss = self._train_per_epoch()
                val_loss = self._validate_per_epoch()
                test_loss = self._test_per_epoch()

                print(f'{epoch}: train {train_loss[0]:.3f}')
                print(f'{epoch}: val   {val_loss[0]:.3f}')
                print(f'{epoch}: test  {test_loss[0]:.3f}')
    
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

        #n_iter = len(self.scdl.dataloaders['train'])
        n_iter = len(self.scdl.dataloaders['test'])
    
        loss_sum = 0
        loss1_sum = 0
        loss2_sum = 0
    
        #for i, (x1,x2) in enumerate(self.scdl.dataloaders['train']):
        for i, (x1,x2) in enumerate(self.scdl.dataloaders['test']):

            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            x = [x1, x2]

            self.optimizer.zero_grad()
            loss_batch = objective(self.params.loss_func, self.model, x, 
				n_mc_samples=self.params.n_mc_samples, beta=self.params.beta)
            loss_batch[0].backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2) 

            self.optimizer.step()
        
            loss_sum += loss_batch[0].item()
            loss1_sum += loss_batch[1].item()
            loss2_sum += loss_batch[2].item()
        
        loss_epoch = loss_sum / n_iter
        loss1_epoch = loss1_sum / n_iter
        loss2_epoch = loss2_sum / n_iter
        #print(f'{self.epoch_current}: train {loss_epoch:.3f}') # abby
    
        return (loss_epoch, loss1_epoch, loss2_epoch)    


    def _validate_per_epoch(self):
    
        n_iter = len(self.scdl.dataloaders['val'])
        loss_sum = 0
        loss1_sum = 0
        loss2_sum = 0
    
        self.model.eval()
        with torch.no_grad():
            for i, (x1,x2) in enumerate(self.scdl.dataloaders['val']):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                x = [x1, x2]
            
                loss_batch = objective(self.params.loss_func, self.model, x, 
				n_mc_samples=self.params.n_mc_samples, beta=self.params.beta)

                loss_sum += loss_batch[0].item()
                loss1_sum += loss_batch[1].item()
                loss2_sum += loss_batch[2].item()

        loss_epoch = loss_sum / n_iter
        loss1_epoch = loss1_sum / n_iter
        loss2_epoch = loss2_sum / n_iter
        #print(f'{self.epoch_current}: val, {loss_epoch:.3f}') # abby

        return (loss_epoch, loss1_epoch, loss2_epoch)    


    def _test_per_epoch(self):
    
        n_iter = len(self.scdl.dataloaders['test'])
        loss_sum = 0
        loss1_sum = 0
        loss2_sum = 0
        
        self.model.eval()
        with torch.no_grad():
            for i, (x1,x2) in enumerate(self.scdl.dataloaders['val']):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                x = [x1, x2]
                
                loss_batch = objective(self.params.loss_func, self.model, x, 
				n_mc_samples=self.params.n_mc_samples, beta=self.params.beta)

                loss_sum += loss_batch[0].item()
                loss1_sum += loss_batch[1].item()
                loss2_sum += loss_batch[2].item()

                if i == 0:
                    self.reconstruct(x, save=True)
                    self.get_latent_features(x, save=True)
    
        loss_epoch = loss_sum / n_iter
        loss1_epoch = loss1_sum / n_iter
        loss2_epoch = loss2_sum / n_iter
        #print(f'{self.epoch_current}: test, {loss_epoch:.3f}') # abby

        return (loss_epoch, loss1_epoch, loss2_epoch)    

    def embed(self, data: Union[np.ndarray, torch.Tensor], method="umap", n_neighbors=20):
        if isinstance(data, torch.Tensor):
            data = tensor_to_numpy(data)
        assert isinstance(data, np.ndarray)
        if method == "umap":
            return embed_umap(data, n_neighbors, self.params.seed)
        else:
            raise ValueError(f"Invalid embedding method: {method}")


    #def generate(self, sample_size=self.params.batch_size, n_mc_samples=1, save=False):
    def generate(self, 
    		kind: Literal["np", "tensor", "df"]="np",
		sample_size=100, n_mc_samples=1, save=False):

        if save:
            x_gen = self.model.generate(sample_size, n_mc_samples)
            tensor_to_df(x_gen).to_csv(
                self._filename_epoch(self.dpath_anal,
                                        "mtx_xg_", self.epoch_current), index=True)
        else:
            xg = self.model.generate(sample_size, n_mc_samples)
            if kind == 'np':
                xg = tensor_to_numpy(xg)
            elif kind == 'df':
                xg = tensor_to_df(xg)
            elif kind == 'tensor':
                xg = xg
            else:
                raise ValueError(f"Invalid data type: {kindh}")
            return xg

    def get_latent_features(self, 
	xs: List[Union[np.ndarray, torch.Tensor]], 
	save=False, n_mc_samples=1
    ):
        x1, x2 = xs
        assert type(x1) == type(x2)

        if isinstance(x1, np.ndarray):
            x1, x2 = torch.Tensor(x1), torch.Tensor(x2)
            zss = self.model.get_latent_features([x1, x2], n_mc_samples)
            return [zss[0].data.numpy(), zss[1].data.numpy()]

        assert isinstance(x1, torch.Tensor)

        if save:
            zss = self.model.get_latent_features(xs, n_mc_samples)
            tensor_to_df(zss[0]).to_csv(
                self._filename_epoch(self.dpath_anal,
                                        "mtx_z1", self.epoch_current), index=True)
            tensor_to_df(zss[1]).to_csv(
                self._filename_epoch(self.dpath_anal,
                                        "mtx_z2", self.epoch_current), index=True)
        else:
            return self.model.get_latent_features(xs, n_mc_samples)

    
    def reconstruct(self, x: Union[np.ndarray, torch.Tensor], save=False):
        recon = self.model.reconstruct(x)
        return recon

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            xp, _ = self.model.reconstruct(x)
            return xp.data.numpy()
        assert isinstance(x, torch.Tensor)

        if save:
            xp, layers = self.model.reconstruct(x)
            tensor_to_df(xp).to_csv(
		self._filename_epoch(self.dpath_anal, 
					"mtx_xp", self.epoch_current), index=True)
            #self._save_layers_to_file(layers)
        else:
            xp, _ = self.model.reconstruct(x)
            return xp

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


    def plot_features_one(self, df_x, df_es, lst_feature_names=None, modality=''):

        ''' default genes defined '''
        lst_feature_names_default = DataDefault.FEATURE_GENES

        ''' init genes: common genes in two modality '''
        if lst_feature_names is None:
            lst_feature_names = lst_feature_names_default

        ''' need to add for checking if genes exists'''
        df_selected = df_x[lst_feature_names]

        return gg_point_feature_active(df_es, df_selected, title=self.prj_name+modality)


    def plot_features(self, lst_feature_names=None):

        ''' default genes defined '''
        lst_feature_names_default = DataDefault.FEATURE_GENES

        fpath_features_1 = self.dpath_anal.joinpath('plot_features_1.png')
        fpath_features_2 = self.dpath_anal.joinpath('plot_features_2.png')
        fname_merge = 'amerge_features_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' init genes: common genes in two modality '''
        if lst_feature_names is None:
            lst_feature_names = lst_feature_names_default

        ''' init dataframe '''
        if self.dic_samples is None:
            self.get_sample_best_run()

        df_x1, df_x2 = self.dic_samples['xs']
        df_es1, df_es2 = self.dic_samples['ess']
        g_features_1 = self.plot_features_one(df_x1, df_es1, lst_feature_names)
        g_features_2 = self.plot_features_one(df_x2, df_es2, lst_feature_names)

        g_features_1.save(fpath_features_1, verbose = False)
        g_features_2.save(fpath_features_2, verbose = False)

        ''' list of filepaths '''
        lst_paths_img = [
                fpath_features_1,
                fpath_features_2
        ]

        grid_display(lst_paths_img, ncols=1, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)


    def get_sample_best_run(self, kind: Literal["np", "tensor", "df"]="np"):
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
        self.dic_samples = {'xs': [df_x1, df_x2], 'zss': [df_zs1, df_zs2], 'ess': [df_es1, df_es2], 'pxzs': pxzs, 'qzxs': qzxs }

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


    def plot_z_embed(self, zs, lst_ident=None, categories=None, title=None):
        g_embeded = gg_point_embed(zs, lst_ident, categories=categories, title=title)
        ''' ggplot '''
        return g_embeded

    def plot_z_embed_ident(self, zs, str_level='level1'):
        assert isinstance(zs, pd.DataFrame), "should be dataframe with barcode information"
        ''' get ident for each barcode from reference '''
        lst_barcodes = list(zs.index)
        ident = get_cell_ident_from_reference(lst_barcodes,
                            data_id='cit_hao', str_level=str_level)
        title = self.prj_name + '_' + str_level

        return self.plot_z_embed(zs, lst_ident=ident, categories=None, title=title)

    def plot_z_embed_pair(self, zss):
        ''' tensors umap-embeded '''
        zs1, zs2 = zss 
        zs1 = check_mtx_to_df(zs1)
        zs2 = check_mtx_to_df(zs2)
        z1_embeded = self.embed(zs1.values)
        z2_embeded = self.embed(zs2.values)
        title = self.prj_name + '_' + f'{self.FOSCTTM:.2f}'
        return gg_point_embed_pair(z1_embeded, z2_embeded,
                                    title=title, figure_size=(5,5))

    def plot_z_embed_all(self, zss):

        ''' filepaths for output images '''
        fpath_embed_umap_pair = self.dpath_anal.joinpath('plot_z_embed_umap_pair.png')
        fpath_embed_umap_z1_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_level1.png')
        fpath_embed_umap_z1_2 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_level2.png')
        fpath_embed_umap_z1_3 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_level3.png')
        fpath_embed_umap_z2_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_level1.png')
        fpath_embed_umap_z2_2 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_level2.png')
        fpath_embed_umap_z2_3 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_level3.png')
        fname_merge = 'amerge_z_embed_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' tensors umap-embeded '''
        zs1, zs2 = zss

        ''' plot z_embed_pair '''
        g_pair = self.plot_z_embed_pair(zss)

        ''' embedding and plotting '''
        g_z_embed_umap_z1_level1 = self.plot_z_embed_ident(zs1, str_level='level1')
        g_z_embed_umap_z1_level2 = self.plot_z_embed_ident(zs1, str_level='level2')
        g_z_embed_umap_z1_level3 = self.plot_z_embed_ident(zs1, str_level='level3')

        g_z_embed_umap_z2_level1 = self.plot_z_embed_ident(zs2, str_level='level1')
        g_z_embed_umap_z2_level2 = self.plot_z_embed_ident(zs2, str_level='level2')
        g_z_embed_umap_z2_level3 = self.plot_z_embed_ident(zs2, str_level='level3')

        ''' save output image files '''
        g_z_embed_umap_z1_level1.save(fpath_embed_umap_z1_1, verbose=False)
        g_z_embed_umap_z1_level2.save(fpath_embed_umap_z1_2, verbose=False)
        g_z_embed_umap_z1_level3.save(fpath_embed_umap_z1_3, verbose=False)

        g_z_embed_umap_z2_level1.save(fpath_embed_umap_z2_1, verbose=False)
        g_z_embed_umap_z2_level2.save(fpath_embed_umap_z2_2, verbose=False)
        g_z_embed_umap_z2_level3.save(fpath_embed_umap_z2_3, verbose=False)
        g_pair.save(fpath_embed_umap_pair, verbose=False)


        ''' save only existing images '''
        all_lst_paths_img = [
                fpath_embed_umap_z1_3,
                fpath_embed_umap_z1_2,
                fpath_embed_umap_z1_1,
                fpath_embed_umap_pair,
                fpath_embed_umap_z2_3,
                fpath_embed_umap_z2_2,
                fpath_embed_umap_z2_1
        ]

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=4, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)



    def plot_z_embed_all_OLD(self):
        self.plot_z_embed(label=True, str_level="level1")
        self.plot_z_embed(label=True, str_level="level2")
        self.plot_z_embed(label=True, str_level="level3")
        self.show_plot_z_embed()

    def show_plot_z_embed_OLD(self):
        fpath_embed_umap_pair = self.dpath_anal.joinpath('plot_z_embed_umap_pair.png')
        fpath_embed_umap_z1_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_level1.png')
        fpath_embed_umap_z1_2 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_level2.png')
        fpath_embed_umap_z1_3 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_level3.png')
        fpath_embed_umap_z2_1 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_level1.png')
        fpath_embed_umap_z2_2 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_level2.png')
        fpath_embed_umap_z2_3 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_level3.png')
        fname_merge = 'amerge_z_embed_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        all_lst_paths_img = [
        	fpath_embed_umap_z1_3,
        	fpath_embed_umap_z1_2,
        	fpath_embed_umap_z1_1,
        	fpath_embed_umap_pair,
        	fpath_embed_umap_z2_3,
        	fpath_embed_umap_z2_2,
        	fpath_embed_umap_z2_1,
        	fpath_embed_umap_pair
        ]

        ''' save only existing images '''
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]

        grid_display(lst_paths_imgs, ncols=4, lst_titles=[],
                                        figratio=1.0, filepath_out=fpath_merge)

    def plot_all_ORG(self):
        self.update_to_best_epoch()
        self.plot_loss()
        self.plot_recon()
        self.plot_z_embed_all()
        self.plot_z_boxplot()
        self.plot_z_activation()
        self.show_plot()

    def plot_all(self):
        #x, zs, es, pxz, qzx = self.get_sample_best_run("df")
        xs, zss, es, pxzs, qzxs = self.get_sample_best_run("df")
        print("- plotting loss")
        self.plot_loss()
        print("- plotting reconstruction")
        self.plot_recon(xs)
        print("- plotting z embedding")
        self.plot_z_embed_all(zss)
        print("- plotting z boxplot")
        self.plot_z_boxplot(zss, qzxs)
        print("- plotting z activation")
        self.plot_z_activation(zss)
        print("- plotting feature activation")
        self.plot_features()
        self.show_plot()


    def plot_z_boxplot(self, zss, qzxs):

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

    def plot_z_embed_OLD(self, label=True, str_level="level1"):

        ''' define output image file '''
        fpath_embed_umap_z1 = self.dpath_anal.joinpath('plot_z_embed_umap_z1_' + str_level +'.png')
        fpath_embed_umap_z2 = self.dpath_anal.joinpath('plot_z_embed_umap_z2_' + str_level +'.png')
        fpath_embed_umap_pair = self.dpath_anal.joinpath('plot_z_embed_umap_pair.png')

        ''' get sample data as dataframe with barcodes and feature names '''
        df_x1, df_x2 = self.get_sample_data("df")
        lst_barcodes = list(df_x1.index)

        ''' latent space embedding '''
        x1 = df_to_tensor(df_x1)
        x2 = df_to_tensor(df_x2)
        zss = self.get_latent_features([x1, x2])

        ''' tensors umap-embeded '''
        z1_embeded = self.embed(zss[0])
        z2_embeded = self.embed(zss[1])

        if label:
            ''' get ident for each barcode from reference '''
            ident = get_cell_ident_from_reference(lst_barcodes,
                                data_id='cit_hao', str_level=str_level)
            ''' embeded dataframe '''
            df1_embeded = pd.DataFrame(z1_embeded, columns=['x','y'])
            df2_embeded = pd.DataFrame(z2_embeded, columns=['x','y'])

            g_pair = gg_point_embed_pair(df1_embeded, df2_embeded, 
					title=self.prj_name, figure_size=(5,5))

            ''' for ident labeling '''
            df1_embeded['ident'] = pd.Categorical(ident)
            df2_embeded['ident'] = pd.Categorical(ident)

            ''' ggplot '''
            g_z1 = gg_point_ident_single(df1_embeded, title=self.prj_name, figure_size=(5,5))
            g_z2 = gg_point_ident_single(df2_embeded, title=self.prj_name, figure_size=(5,5))

            ''' save ggplot '''
            g_z1.save(fpath_embed_umap_z1, verbose = False)
            g_z2.save(fpath_embed_umap_z2, verbose = False)
            g_pair.save(fpath_embed_umap_pair, verbose = False)
            #print(g_z1)
            #print(g_z2)
            #print(g_pair)

        else:
            print('need to implement here!!!')
            return


    def plot_z_activation(self,
        zss: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ):
        ''' define output image file '''
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

    def plot_z_embed_OLD(self):

        ''' define output image file '''
        fpath_embed_umap = self.dpath_anal.joinpath('plot_z_embed_umap.png')

        ''' get sample data as dataframe with barcodes and feature names '''
        xs = self.get_sample_data("tensor")

        ''' latent space embedding '''
        zss = self.get_latent_features(xs)

        '''umap and plotting'''
        g_embed = gg_point_embed_pair(self.embed(zss[0]), self.embed(zss[1]))
        print(g_embed)
        '''save '''
        g_embed.save(fpath_embed_umap)


    def plot_recon(self, xs):

        def _cluster_spectral(data, n_clusters=(3,3), seed=42):
            model = SpectralBiclustering(n_clusters=n_clusters, method='log',
                                     random_state=seed)
            model.fit(data)
            return model

        def _sort_matrix(data, model):
            fit_data = data[np.argsort(model.row_labels_)]
            fit_data = fit_data[:, np.argsort(model.column_labels_)]
            return fit_data

        ''' define paths for output image files ''' 
        fpath_hm_x1 = self.dpath_anal.joinpath('plot_hm_x1.png')
        fpath_hm_x2 = self.dpath_anal.joinpath('plot_hm_x2.png')
        fpath_hm_xp11 = self.dpath_anal.joinpath('plot_hm_xp11.png')
        fpath_hm_xp12 = self.dpath_anal.joinpath('plot_hm_xp12.png')
        fpath_hm_xp21 = self.dpath_anal.joinpath('plot_hm_xp21.png')
        fpath_hm_xp22 = self.dpath_anal.joinpath('plot_hm_xp22.png')
        fname_merge = 'amerge_xp_hm_' + self.prj_name + '.png'
        fpath_merge = self.dpath_anal.joinpath(fname_merge)

        ''' input cleaning ''' 
        assert type(xs[0]) == type(xs[1])

        if isinstance(xs[0], pd.DataFrame):
            x1 = df_to_tensor(xs[0])
            x2 = df_to_tensor(xs[1])
        elif isinstance(xs[0], np.ndarray):
            x1 = torch.from_numpy(xs[0])
            x2 = torch.from_numpy(xs[1])
        assert isinstance(x1, torch.Tensor)

        ''' do reconstruction ''' 
        xs = [x1, x2]
        xp = self.reconstruct(xs)
        x_elbos = objective(self.params.loss_func, self.model, xs, 
			n_mc_samples=self.params.n_mc_samples, beta=self.params.beta)

        x_elbo_str = str(np.round(-x_elbos[0].item(), 2))

        ''' tensor to numpy ''' 
        x1 = tensor_to_numpy(x1)
        x2 = tensor_to_numpy(x2)

        xp11 = tensor_to_numpy(xp[0][0])
        xp12 = tensor_to_numpy(xp[0][1])
        xp21 = tensor_to_numpy(xp[1][0])
        xp22 = tensor_to_numpy(xp[1][1])

        ##########################################
        hm_xp_kwargs = {'xlab': "Feature", 'ylab': "Sample",
        		'title': self.prj_name + '_xp_' + x_elbo_str, 'figure_size': (4,3) }

        hm_x_kwargs = {'xlab': "Feature", 'ylab': "Sample",
        		'title': self.prj_name + '_x', 'figure_size': (4,3) }

        ''' for rna, modality 1 '''
        n_clusters = (7,8)
        model_spectral = _cluster_spectral(x1, n_clusters, self.params.seed)
        x1_sorted = _sort_matrix(x1, model_spectral)
        xp11_sorted = _sort_matrix(xp11, model_spectral)
        xp21_sorted = _sort_matrix(xp21, model_spectral)

        hm_x1 = heatmap_from_mtx(x1_sorted[30:100, 1950:], **hm_x_kwargs)
        hm_xp11 = heatmap_from_mtx(xp11_sorted[30:100, 1950:], **hm_xp_kwargs)
        hm_xp21 = heatmap_from_mtx(xp21_sorted[30:100, 1950:], **hm_xp_kwargs)

        ''' for adt, modality 2 '''
        n_clusters = (6,7)
        model_spectral = _cluster_spectral(x2, n_clusters, self.params.seed)
        x2_sorted = _sort_matrix(x2, model_spectral)
        xp12_sorted = _sort_matrix(xp12, model_spectral)
        xp22_sorted = _sort_matrix(xp22, model_spectral)

        hm_x2 = heatmap_from_mtx(x2_sorted[:70, 70:135], **hm_x_kwargs)
        hm_xp12 = heatmap_from_mtx(xp12_sorted[:70, 70:135], **hm_xp_kwargs)
        hm_xp22 = heatmap_from_mtx(xp22_sorted[:70, 70:135], **hm_xp_kwargs)

        ##########################################
        # temporary
        ##########################################

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

    def show_plot(self, target: Literal['all', 'loss', 'recon', 'latent']='all'):
        fpath_merge_all = self.dpath_anal.joinpath('amerge_all_' + self.prj_name + '.png')
        fpath_merge_hm = self.dpath_anal.joinpath('amerge_xp_hm_' + self.prj_name + '.png')
        fpath_merge_loss = self.dpath_anal.joinpath('amerge_loss_' + self.prj_name + '.png')
        fpath_merge_embed = self.dpath_anal.joinpath('amerge_z_embed_' + self.prj_name + '.png')
        fpath_merge_z_active = self.dpath_anal.joinpath('amerge_z_active_' + self.prj_name + '.png')
        fpath_merge_features = self.dpath_anal.joinpath('amerge_features_' + self.prj_name + '.png')
        fpath_merge_bp = self.dpath_anal.joinpath('amerge_z_bp_' + self.prj_name + '.png')

        all_lst_paths_img = [
        	fpath_merge_embed,
        	fpath_merge_z_active,
        	fpath_merge_features,
        	fpath_merge_hm,
        	fpath_merge_bp,
        	fpath_merge_loss
	]
        
        lst_paths_imgs = [fpath for fpath in all_lst_paths_img if fpath.exists()]
        
        grid_display(lst_paths_imgs, ncols=1, lst_titles=[],
					figratio=1.0, filepath_out=fpath_merge_all)

