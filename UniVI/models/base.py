from collections import OrderedDict
import torch
from torch import nn

import UniVI
import importlib
importlib.reload(UniVI._settings)
from UniVI._settings import Constants
importlib.reload(UniVI._settings)

def create_unit_layers_wo_BATCHNORM(
        dim_in: int,
        dim_out: int,
        name_layer: str='fc',
        n_hidden_layers: int=1,
        dropout_rate: float=0.2
):
    ''' For multiple layers, should have the same in and out dimension '''
    if n_hidden_layers > 1:
        assert dim_in == dim_out,\
                 "should have the same dimension for multiple hidden layers"

    ''' Unit fully-connected layer '''
    a_layer = nn.Sequential(
               nn.Linear(in_features=dim_in, out_features=dim_out, bias=True),
               nn.ReLU(inplace=True),
               nn.Dropout(p=dropout_rate, inplace=False)
    )

    ''' Return multiple fully-connected layers '''
    return nn.Sequential(
        OrderedDict(
            [ ( f'{name_layer}_{i}', a_layer ) for i in range(n_hidden_layers) ]
        )
    )


def create_unit_layers(
        dim_in: int, 
        dim_out: int, 
        name_layer: str='fc',
        n_hidden_layers: int=1,
        dropout_rate: float=0.1
):
    ''' For multiple layers, should have the same in and out dimension '''
    if n_hidden_layers > 1:
        assert dim_in == dim_out,\
                 "should have the same dimension for multiple hidden layers"

    ''' Unit fully-connected layer '''
    a_layer = nn.Sequential(
               nn.Linear(in_features=dim_in, out_features=dim_out, bias=True),
               nn.BatchNorm1d(dim_out, 
                              eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
               nn.ReLU(inplace=True),
               nn.Dropout(p=dropout_rate, inplace=False)
    )

    ''' Return multiple fully-connected layers '''
    return nn.Sequential(
        OrderedDict(
            [ ( f'{name_layer}_{i}', a_layer ) for i in range(n_hidden_layers) ]
        )
    )

def create_unit_module_head(
        dim_in: int,
        dim_hidden: int,
        n_hidden_layers: int=1,
        name_module: str='module',
        dropout_rate: float=0.1
):
    a_module = nn.Sequential(
        OrderedDict({
            'fc_beg': create_unit_layers(dim_in, dim_hidden, 'beg'),
            'fc_hid': create_unit_layers(dim_hidden, dim_hidden, 'hid', n_hidden_layers)
        })
    )
    ''' Return a named module '''
    return nn.Sequential(OrderedDict({name_module: a_module}))

def create_unit_module_tail(
        dim_latent: int,
        dim_hidden: int,
        n_hidden_layers: int=1,
        name_module: str='module',
        dropout_rate: float=0.1
):
    a_module = nn.Sequential(
        OrderedDict({
            'fc_beg': create_unit_layers(dim_latent, dim_hidden, 'beg'),
            'fc_hid': create_unit_layers(dim_hidden, dim_hidden, 'hid', n_hidden_layers)
        })
    )
    ''' Return a named module '''
    return nn.Sequential(OrderedDict({name_module: a_module}))


def create_unit_module_ORG(
        dim_in: int, 
        dim_out: int, 
        dim_hidden: int, 
        n_hidden_layers: int=1,
        name_module: str='module',
        dropout_rate: float=0.1
):
    a_module = nn.Sequential(
        OrderedDict({
            'BeginLayer': create_unit_layers(dim_in, dim_hidden, 'beg'),
            'HiddenLayer': create_unit_layers(dim_hidden, dim_hidden, 'hid', n_hidden_layers),
            'EndLayer': create_unit_layers(dim_hidden, dim_out, 'end'),
        })
    )
    ''' Return a named module '''
    return nn.Sequential(OrderedDict({name_module: a_module}))


class Encoder(nn.Module):
    '''
    Example
    -------
    enc = Encoder(2000, 20, 256, 3)
    '''
    def __init__(
        self,
        dim_in: int,
        dim_latent: int,
        dim_hidden: int,
        n_hidden_layers: int=1,
        name_module: str='Encoder',
        dropout_rate: float=0.1
    ):
        super().__init__()

        self.dim_latent = dim_latent

        self.enc_hid = create_unit_module_head(
            dim_in = dim_in,
            dim_hidden = dim_hidden,
            n_hidden_layers = n_hidden_layers,
            name_module = name_module,
            dropout_rate = dropout_rate
        )

        ''' for latent bottleneck layers: bn '''
        self._bn_mu = nn.Linear(dim_hidden, dim_latent)
        self._bn_logvar = nn.Linear(dim_hidden, dim_latent)

    def forward(self, x):
        
        '''
        print(x)
        print(x.shape)
        print(x.dtype)
        '''
        
        x = self.enc_hid(x)
        
        '''
        print(x)
        print(x.shape)
        print(x.dtype)
        '''
        
        ''' for mean '''
        qzx_param_mu = self._bn_mu(x)
        ''' for variance '''
        _bn_logvar = self._bn_logvar(x)
        _bn_logvar = nn.functional.softplus(_bn_logvar)
        qzx_param_sigma = torch.exp(_bn_logvar/2) + Constants.eta

        return qzx_param_mu, qzx_param_sigma

class DecoderGaussian(nn.Module):
    ''' not having a rsample function '''
    def __init__(
        self,
        dim_latent: int, 
        dim_out: int, 
        dim_hidden: int, 
        n_hidden_layers: int=1,
        name_module: str='DecoderGaussian',
        dropout_rate: float=0.1
    ):
        super().__init__()

        self.decoder_type = "gaussian"

        self.dec_hid = create_unit_module_tail(
            dim_latent = dim_latent,
            dim_hidden = dim_hidden,
            n_hidden_layers = n_hidden_layers,
            name_module = name_module,
            dropout_rate = dropout_rate
        )

        ''' for output layer ''' 
        self._dec_mu = nn.Linear(dim_hidden, dim_out)
        self._dec_logvar = nn.Linear(dim_hidden, dim_out)

    def forward(self, z):

        #z = z.mean(0)
        z = self.dec_hid(z)

        ''' for mean '''
        pxz_param_mu = self._dec_mu(z)

        ''' for variance '''
        _dec_logvar = self._dec_logvar(z)
        _dec_logvar = nn.functional.softplus(_dec_logvar)
        pxz_param_sigma = torch.exp(_dec_logvar/2) + Constants.eta

        return pxz_param_mu, pxz_param_sigma


class DecoderNB(nn.Module):
    '''
    Example
    -------
    dec = DecoderNB(20, 2000, 256, 3)
    '''

    def __init__(
        self,
        dim_latent: int,
        dim_out: int,
        dim_hidden: int,
        n_hidden_layers: int=1,
        name_module: str='DecoderNB',
        dropout_rate: float=0.1
    ):
        super().__init__()
        
        self.dim_latent = dim_latent
        self.dim_hidden = dim_hidden

        self.decoder_type = "nb"

        self.dec_hid = create_unit_module_tail(
            dim_latent = dim_latent,
            dim_hidden = dim_hidden,
            n_hidden_layers = n_hidden_layers,
            name_module = name_module,
            dropout_rate = dropout_rate
        )

        ''' for output layer '''
        self._dec_totalcount = nn.Linear(dim_hidden, dim_out)
        self._dec_logits = nn.Linear(dim_hidden, dim_out)

    #def forward(self, zs):
        
        ''' reshaping for out layers ''' 
        #zs = self.dec_hid(zs) 

        ''' for total_count '''
        #_dec_totalcount = self._dec_totalcount(zs) 
        #_dec_totalcount = nn.functional.softplus(_dec_totalcount)
        #pxz_param_total_count = torch.clamp(_dec_totalcount,
        #                                    max=Constants.clamp_max)  ####?? needs clamps
        ''' for logits '''
        #_dec_logits = self._dec_logits(zs)
        #pxz_param_logits = torch.clamp(_dec_logits,
        #                               min=-Constants.clamp_max, max=Constants.clamp_max)

        ''' reshaping back to original params shape ''' 
        #return pxz_param_total_count, pxz_param_logits
    
    # New forward() function due to the generation of invalid values outside of what is expected by 
    # a negative binomial distribution. Might still need to set the min and max variables for both the
    # pxz_param_total_count variable and the pxz_param_logits variable. This would keep the code more
    # consistent.
    def forward(self, zs):
        ''' Reshaping for output layers '''
        zs = self.dec_hid(zs) 

        ''' Compute total_count '''
        _dec_totalcount = self._dec_totalcount(zs) 
        _dec_totalcount = nn.functional.softplus(_dec_totalcount)
        pxz_param_total_count = torch.clamp(_dec_totalcount, max=Constants.clamp_max)

        ''' Compute logits '''
        _dec_logits = self._dec_logits(zs)

        # Ensure logits are clamped to avoid extreme probabilities
        _dec_logits = torch.clamp(_dec_logits, min=-10, max=10)

        # Compute probabilities
        pxz_param_logits = torch.sigmoid(_dec_logits)
        pxz_param_logits = torch.clamp(pxz_param_logits, min=1e-6, max=1 - 1e-6)

        return pxz_param_total_count, pxz_param_logits






