from torch.distributions import Normal
from typing import Union
from torch import nn
import torch

import scvt
from scvt.models.base import DecoderGaussian, DecoderNB

class VAE(nn.Module):
    
    def __init__(
        self,
        pxz_class: torch.distributions,
        encoder: nn.Module,
        decoder: nn.Module
    ):
        super().__init__()
        
        self.qzx_class = Normal
        self.pxz_class = pxz_class
        self.encoder = encoder
        self.decoder = decoder

        grad = {'requires_grad': False}
        self.pz_param_mean = nn.Parameter(torch.zeros(1, encoder.dim_latent), requires_grad=False)
        self.pz_param_logvar = nn.Parameter(torch.zeros(1, encoder.dim_latent), **grad)

    '''
    def pz_params(self):
        return self.pz_param_mean, torch.exp(nn.Softmax(self.pz_param_logvar, dim=-1)) + Constants.eta
    '''
    
    #def forward(self, x, n_mc_samples=1):
    def forward(self, x):
        qzx_param_mean, qzx_param_var = self.encoder(x)
        qzx = self.qzx_class(qzx_param_mean, qzx_param_var.sqrt())
        
        #zs = qzx.rsample(torch.Size([n_mc_samples]))
        zs = qzx.rsample()

        pxz = None        
        if self.decoder.decoder_type == "gaussian":
            pxz_param_mean, pxz_param_var = self.decoder(zs)
            pxz = self.pxz_class(pxz_param_mean, pxz_param_var.sqrt())
    
        elif self.decoder.decoder_type == "nb":
            pxz_param_total_count, pxz_param_logits = self.decoder(zs)
            pxz = self.pxz_class(total_count=pxz_param_total_count, logits=pxz_param_logits)   

        else:
            raise ValueError(f'Unsupported distribution: {type(self.decoder)}')
        
        return qzx, pxz, zs    
    
    def calc_reconstruct_loss(self, x):
        _, pxz, _ = self.forward(x)
        log_likelihood = pxz.log_prob(x).sum(dim=-1).sum() / x.shape[0] ### ??? if use K??
        return -log_likelihood

    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            _, pxz, _ = self.forward(x)
            x_recon = pxz.mean
        return x_recon

    def get_latent_features(self, x):
        self.eval()
        with torch.no_grad():
            qzx_param_mean, qzx_param_var = self.encoder(x)
            qzx = self.qzx_class(qzx_param_mean, qzx_param_var.sqrt())
            zs = qzx.rsample()
        return zs




