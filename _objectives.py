import torch
from torch.distributions import Normal
from scvt.utilities._utils import log_mean_exp, kl_divergence, get_device, df_to_tensor

from scvt.external.evaluate_FOSCTTM import calc_frac
from scvt.utilities._utils import tensor_to_numpy, get_device


from numpy import prod
import numpy as np
import pandas as pd


def objective(loss_func, model, x, beta=1.0):
    #print(loss_func); assert False
    #if loss_func == "iwae":
        #(loss, loss1, loss2) = iwae(model, x, n_mc_samples=n_mc_samples, beta=beta)
    if loss_func == "elbo":
        (loss, loss1, loss2) = elbo(model, x, beta=beta)
    elif loss_func == "m_elbo":
        (loss, loss1, loss2) = m_elbo(model, x, beta=beta)
    else:
        raise ValueError("Invalid loss function: {loss_func}")
    return (loss, loss1, loss2)


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def two_el(value):
     if value.ndim == 1:
         return round(value[0].item(),2)
     elif value.ndim == 2:
         return round(value[0][0].item(),2)

def pp(lst, label=""):
    log = [str(v) for v in lst]
    log = "\t".join(log)
    print(label, log)


def MMD(x, y):
    # https://www.kaggle.com/onurtunali/maximum-mean-discrepancy
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz 
    
    XX, YY, XY = (torch.zeros(xx.shape).to(get_device()),
                  torch.zeros(xx.shape).to(get_device()),
                  torch.zeros(xx.shape).to(get_device()))
    
    #bandwidth_range = [0.2, 0.5, 0.9, 1.3]
    bandwidth_range = [0.1, 0.2, 0.5, 0.9, 1.3, 2.0]
    for a in bandwidth_range:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1
            
    return torch.mean(XX + YY - 2. * XY)


def m_elbo(model, x, beta=1.0):
    
    x1, x2 = x
    n_batch, _ = x1.shape
    '''
    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)
    '''
    '''
    print(x1)
    print(x1.dtype)
    print(x1.shape)
    
    print(x2)
    print(x2.dtype)
    print(x2.dtype)
    '''

    assert type(x1) == type(x2)
    if isinstance(x1, pd.DataFrame):
        x1 = df_to_tensor(x1)
        x2 = df_to_tensor(x2)
    elif isinstance(x1, np.ndarray):
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)

    # feature ratio between two modalities
    feature_ratio = round(x1.shape[1] / x2.shape[1], ndigits=1)

    # forward
    '''
    print(x1, x2)
    print(x1.shape, x2.shape)
    print(x1.dtype, x2.dtype)
    '''
    
    qzxs, pxzs, zss = model([x1,x2])
    qzx1, qzx2 = qzxs
    [[pxz11, pxz12], [pxz21, pxz22]] = pxzs
    z1, z2  = zss

    # KLD
    kld1 = kl_divergence(qzx1, model._pz(*model.pz_theta)).sum(-1)
    kld2 = kl_divergence(qzx2, model._pz(*model.pz_theta)).sum(-1)
    klds = [kld1, kld2]

    # extra KLD
    klds_ex1 = kl_divergence(qzx1, qzx2).sum(-1)
    klds_ex2 = kl_divergence(qzx2, qzx1).sum(-1)
    klds_ex = [klds_ex1, klds_ex2]

    # MMD penalty term
    mmd = MMD(z1, z2)

    # reconstruction error for four decoders
    lpxz11 = pxz11.log_prob(x1).sum(-1)
    lpxz12 = feature_ratio * pxz12.log_prob(x2).sum(-1)
    lpxz21 = pxz21.log_prob(x1).sum(-1)
    lpxz22 = feature_ratio * pxz22.log_prob(x2).sum(-1)

    # weight
    w12 = (qzx1.log_prob(z2.detach()) - qzx2.log_prob(z2.detach()).detach()).exp().sum(-1)
    w21 = (qzx2.log_prob(z1.detach()) - qzx1.log_prob(z1.detach()).detach()).exp().sum(-1)

    # reconstruction error
    lpxzs = [lpxz11, w12 * lpxz12, w21 * lpxz21, lpxz22]

    #obj = 0.5*(torch.stack(lpxzs).sum(0) - beta*( torch.stack(klds).sum(0) + torch.stack(klds_ex).sum(0) )).sum(0) - mmd
    obj = 0.5*(torch.stack(lpxzs).sum(0) - beta*( torch.stack(klds).sum(0) + torch.stack(klds_ex).sum(0) )).sum(0) 
    #obj = (torch.stack(lpxzs).sum(0) - beta*( torch.stack(klds).sum(0) + torch.stack(klds_ex).sum(0) )).sum(0) - mmd
    #obj = (torch.stack(lpxzs).sum(0) - beta*( torch.stack(klds).sum(0) + torch.stack(klds_ex).sum(0) )).sum(0)

    recon_error = torch.stack(lpxzs).sum(0).sum(0)
    kld = torch.stack(klds).sum(0).sum(0)
    kld_ex = torch.stack(klds_ex).sum(0).sum(0)
    kld_sum = kld + kld_ex

    # to debug
    res = []
    res.append(round(obj.item(), 2))
    res.append(round(recon_error.item(), 2))
    res.append(round(kld_sum.item(), 2))
    res.append(round(mmd.item(), 3))
    res.append(beta)
    res.append(round(kld.item(), 2))
    res.append(round(kld_ex.item(), 2))
    #pp(res, label="LL:")

    #return (-obj, -recon_error, kld_sum)
    return (-obj/n_batch, -recon_error/n_batch, kld_sum/n_batch)


def elbo(model, x, beta=1.0):
    """Computes E_{p(x)}[ELBO] """
    pz = model._pz(*model.pz_params)
    qz_x, px_z, zs = model(x)

    kld = beta * kl_divergence(qz_x, pz).sum(-1)
    lpx_z = px_z.log_prob(x).sum(-1)
    elbo = (lpx_z - kld).sum()

    #print(-elbo.item(), -lpx_z.sum().item(), kld.sum().item())
    #assert False
    return (-elbo, -lpx_z.sum(), kld.sum())




