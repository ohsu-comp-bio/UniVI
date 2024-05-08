from torch.distributions import Normal
from typing import Union, Literal
from torch import distributions as dist
from torch import nn
import torch
import torch.nn.functional as F

from collections import OrderedDict

from UniVI.models import base
import importlib
importlib.reload(base)

import UniVI
from UniVI._settings import Constants
from UniVI.models.base import Encoder, DecoderGaussian, DecoderNB
from UniVI.models._utils import register_hook_vae
from UniVI.utilities._utils import log_mean_exp

import importlib
#from UniVI import _objectives
from UniVI.models import base
#importlib.reload(_objectives)
#importlib.reload(model_utils)
importlib.reload(base)

class VAE(nn.Module):
    
    def __init__(
        self,
        dec_model: Literal["gaussian", "nb"],
        dim_features: int,
        dim_latent: int,
        dim_hidden: int,
        n_hidden_layers: int = 1
    ):
        super().__init__()

        decoder = None
        pxz_ = None
        if dec_model == "gaussian":
            _pxz = dist.Normal
            decoder = DecoderGaussian(dim_latent=dim_latent, 
				      dim_out=dim_features, 
                                      dim_hidden=dim_hidden, 
				      n_hidden_layers=n_hidden_layers)
        elif dec_model == "nb":
            _pxz = dist.NegativeBinomial
            decoder = DecoderNB(dim_latent=dim_latent, 
   			        dim_out=dim_features, 
                                dim_hidden=dim_hidden, 
				n_hidden_layers=n_hidden_layers)
        else: 
            raise ValueError("Invalid decoder model")

        self.encoder = Encoder(dim_in=dim_features, 
			       dim_latent=dim_latent, 
                               dim_hidden=dim_hidden, 
			       n_hidden_layers=n_hidden_layers)

        self.decoder = decoder
        
        self._pxz = _pxz
        self._qzx = dist.Normal
        self.qzx_mu = None
        self.qzx_sigma = None

        self._pz = dist.Normal
        self._pz_theta = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim_latent), requires_grad=False),
            nn.Parameter(torch.zeros(1, dim_latent), requires_grad=True) # for training
	])

    @property
    def pz_params(self):
        pz_mu = self._pz_theta[0]
        _logvar = self._pz_theta[1]
        _logvar = F.softplus(_logvar) 
        pz_sigma = torch.exp(_logvar/2) + Constants.eta
        return pz_mu, pz_sigma

    def _get_pxz(self, zs):
        pxz = None        
        if self.decoder.decoder_type == "gaussian":
            pxz_mu, pxz_sigma = self.decoder(zs)
            pxz = self._pxz(pxz_mu, pxz_sigma)
    
        elif self.decoder.decoder_type == "nb":
            pxz_totalcount, pxz_logits = self.decoder(zs)
            pxz = self._pxz(total_count=pxz_totalcount, logits=pxz_logits)   

        else:
            raise ValueError(f'Unsupported distribution: {type(self.decoder)}')
        return pxz
        
    def forward(self, x):
        self.qzx_mu, self.qzx_sigma = self.encoder(x)
        qzx = self._qzx(self.qzx_mu, self.qzx_sigma)

        zs = qzx.rsample()
        pxz = self._get_pxz(zs)

        return qzx, pxz, zs    
    
    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            qzx_mu, qzx_sigma = self.encoder(x)
            qzx = self._qzx(qzx_mu, qzx_sigma)
            zs = qzx.rsample()
            pxz = self._get_pxz(zs)
            x_recon = pxz.mean

        return x_recon

    def get_latent_features(self, x):
        self.eval()
        with torch.no_grad():
            qzx_mu, qzx_sigma = self.encoder(x)
            qzx = self._qzx(qzx_mu, qzx_sigma)
            zs = qzx.rsample()
        return zs

    def generate_with_zs(self, zs):
        self.eval()
        with torch.no_grad():
            pxz = self._get_pxz(zs)
            xg = pxz.sample()
            return xg  # batch_size x n_features

    def generate(self, lst_idx_zeros=[], batch_size=100):
        print('need to be checked first: def generate in vae.py')
        assert False
        self.eval()
        with torch.no_grad():
            zs = self.pz.rsample([batch_size])
            zs = zs.squeeze(1)
            if len(lst_idx_zeros) > 0:
                zs[:, lst_idx_zeros] = 0
            pxz = self._get_pxz(zs)
            xg = pxz.sample()
            return xg  # batch_size x n_features


