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

class MMVAE(nn.Module):
    def __init__(self, *vaes, dim_latent):
        super().__init__()
        self.vaes = nn.ModuleList([vae for vae in vaes])
        self._pz = dist.Normal
        self._pz_theta = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim_latent), requires_grad=False),
            nn.Parameter(torch.zeros(1, dim_latent), requires_grad=True) # for training
        ])

    @property
    def pz_theta(self):
        pz_mu = self._pz_theta[0]
        _logvar = self._pz_theta[1]
        _logvar = F.softplus(_logvar)
        pz_sigma = torch.exp(_logvar/2) + Constants.eta
        return pz_mu, pz_sigma

    # Updated this commented-out code block to below because of issues when performing training using the negative-binomial
    # decoder model. There were invalid values being passed into pxzs[i][j] = vae._pxz(*vae.decoder(zs)) which threw several
    # errors. These have thorough debugging print statements in the reworked function below.
    def forward(self, x):

        self.check_mu = [None, None]
        self.check_sigma = [None, None]

        qzxs, zss = [], []
        pxzs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for i, vae in enumerate(self.vaes):
            qzx, pxz, zs = vae(x[i])
            qzxs.append(qzx)
            zss.append(zs)
            pxzs[i][i] = pxz 

            self.check_mu[i] = vae.qzx_mu
            self.check_sigma[i] = vae.qzx_sigma

        for i, zs in enumerate(zss):
            for j, vae in enumerate(self.vaes):
                if i != j:
                    pxzs[i][j] = vae._pxz(*vae.decoder(zs))

        return qzxs, pxzs, zss
    
    '''
    def forward(self, x):
        """
        Forward pass for MMVAE with support for both Gaussian and NB decoders.
        """
        self.check_mu = [None, None]
        self.check_sigma = [None, None]

        qzxs, zss = [], []
        pxzs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for i, vae in enumerate(self.vaes):
            qzx, pxz, zs = vae(x[i])
            qzxs.append(qzx)
            zss.append(zs)
            pxzs[i][i] = pxz 

            self.check_mu[i] = vae.qzx_mu
            self.check_sigma[i] = vae.qzx_sigma

        for i, zs in enumerate(zss):
            for j, vae in enumerate(self.vaes):
                if i != j:
                    try:
                        # Decode zs based on the type of decoder
                        if isinstance(vae.decoder, DecoderNB):
                            # Negative Binomial processing
                            total_count, logits = vae.decoder(zs)
                            probs = torch.sigmoid(logits)
                            probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)
                            total_count = torch.nn.functional.softplus(total_count) + Constants.eta
                            pxzs[i][j] = vae._pxz(total_count=total_count, probs=probs)

                        elif isinstance(vae.decoder, DecoderGaussian):
                            # Gaussian processing
                            mu, sigma = vae.decoder(zs)
                            sigma = torch.clamp(sigma, min=1e-6)  # Avoid invalid values
                            pxzs[i][j] = vae._pxz(loc=mu, scale=sigma)

                        else:
                            raise ValueError(f"Unsupported decoder type: {type(vae.decoder)}")

                    except Exception as e:
                        print(f"[ERROR] Failed to create pxz for VAE {i} -> VAE {j}")
                        print(f"Latent variables (zs): min={zs.min().item()}, max={zs.max().item()}, mean={zs.mean().item()}")
                        print(f"Error: {e}")
                        raise e

        return qzxs, pxzs, zss
    '''

    def recon_from_z1(self, z1):
        self.eval()
        with torch.no_grad():
            pxzs11 = self.vaes[0]._pxz(*self.vaes[0].decoder(z1))
            pxzs12 = self.vaes[0]._pxz(*self.vaes[1].decoder(z1))
        return pxzs11.mean, pxzs12.mean # self cross


    def recon_from_z2(self, z2):
        self.eval()
        with torch.no_grad():
            pxzs22 = self.vaes[1]._pxz(*self.vaes[1].decoder(z2))
            pxzs21 = self.vaes[1]._pxz(*self.vaes[0].decoder(z2))
        return pxzs22.mean, pxzs21.mean # self cross


    def reconstruct_x1(self, x1):
        self.eval()
        with torch.no_grad():
            zs = self.vaes[0].get_latent_features(x1)
            pxzs11_mean, pxzs12_mean = self.recon_from_z1(zs)
        return pxzs11_mean, pxzs12_mean # self, cross

    def reconstruct_x2(self, x2):
        self.eval()
        with torch.no_grad():
            zs = self.vaes[1].get_latent_features(x2)
            pxzs22_mean, pxzs21_mean = self.recon_from_z2(zs)
        return pxzs22_mean, pxzs21_mean # self, cross


    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            zss = self.get_latent_features(x)
            #xps = [[px_z.mean.squeeze() for px_z in r] for r in px_zs]
            z = (zss[0] + zss[1]) / 2.
            pxz11, pxz12 = self.recon_from_z1(z)
            pxz22, pxz21 = self.recon_from_z2(z)
        return [[pxz11, pxz12],[pxz21, pxz22]]


    def get_latent_x1(self, x1):
        self.eval()
        with torch.no_grad():
            _, _, zs = self.vaes[0].forward(x1)
        return zs

    def get_latent_x2(self, x2):
        self.eval()
        with torch.no_grad():
            _, _, zs = self.vaes[1].forward(x2)
        return zs

    def get_latent_features(self, x):
        self.eval()
        with torch.no_grad():
            _, _, zss = self.forward(x)
        return [zss[0], zss[1]]

