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
    if loss_func == "iwae":
        (loss, loss1, loss2) = iwae(model, x, n_mc_samples=n_mc_samples, beta=beta)
    elif loss_func == "elbo":
        (loss, loss1, loss2) = elbo(model, x, beta=beta)
    elif loss_func == "m_elbo":
        (loss, loss1, loss2) = m_elbo(model, x, beta=beta)
    else:
        raise ValueError("Invalid loss function: {loss_func}")
    return (loss, loss1, loss2)

#######################
# temporary code 
#######################
def _get_pz(zs):
    #return Normal(torch.zeros([*zs.size()]).to(get_device()), torch.ones([*zs.size()]).to(get_device()))
    return Normal(torch.zeros([*zs.shape]).to(get_device()), torch.ones([*zs.shape]).to(get_device()))


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


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def MMD(x, y, kernel='rbf'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


def m_elbo(model, x, beta=1.0):
    x1, x2 = x
    assert type(x1) == type(x2)
    if isinstance(x1, pd.DataFrame):
        x1 = df_to_tensor(x1)
        x2 = df_to_tensor(x2)
    elif isinstance(x1, np.ndarray):
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)

    feature_ratio = round(x1.shape[1] / x2.shape[1], ndigits=1)

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

    # MMD
    mmd = MMD(z1, z2, kernel='rbf')
    #mmd = MMD(z1, z2, kernel='multiscale')

    # for four decoders 
    lpxz11 = pxz11.log_prob(x1).sum(-1)

    w12 = (qzx1.log_prob(z2.detach()) - qzx2.log_prob(z2.detach()).detach()).exp().sum(-1)
    lpxz12 = feature_ratio * pxz12.log_prob(x2).sum(-1)

    w21 = (qzx2.log_prob(z1.detach()) - qzx1.log_prob(z1.detach()).detach()).exp().sum(-1)
    lpxz21 = pxz21.log_prob(x1).sum(-1)

    lpxz22 = feature_ratio * pxz22.log_prob(x2).sum(-1)

    lpxzs = [lpxz11, w12 * lpxz12, w21 * lpxz21, lpxz22]


    dis = torch.stack(klds_ex).sum(0).sum(0)  #success

    obj = 0.5 * ( torch.stack(lpxzs).sum(0) - beta*torch.stack(klds).sum(0) )  # success !!!!!

    ############################################
    # original, no beta
    #obj = 0.5 * ( torch.stack(lpxzs).sum(0) - beta*torch.stack(klds).sum(0) )  # not good
    ############################################

    obj_sum = obj.sum(0)
    loss = obj_sum - beta * dis - mmd # success!!!!!
    #loss = obj_sum - mmd 

    res = []
    res.append(two_el(kld1))
    res.append(two_el(kld2))
    res.append(two_el(klds_ex1))
    res.append(two_el(klds_ex2))
    res.append(two_el(lpxz11))
    res.append(two_el(lpxz12))
    res.append(two_el(lpxz21))
    res.append(two_el(lpxz22))
    res.append(round(obj_sum.item(), 2))
    #res.append(round(dis.item(), 3))
    #res.append(two_el(dis))
    res.append(round(mmd.item(), 3))
    res.append(round(loss.item(), 2))
    res.append(two_el(w12))
    res.append(two_el(w21))
    #pp(res, label="LL:")

    return (-loss, loss, loss)


def get_lik_ratio(n_x1, n_x2):
    return n_x1/n_x2


def iwae(model, x, n_mc_samples=1, beta=1.0):

    batch_size = x.shape[0]

    qzx, pxz, zs = model(x, n_mc_samples)
    #pz = model._pz(*model.pz_params())
    pz = _get_pz(zs) #abby

    ''' sum over features '''
    lpz = pz.log_prob(zs).sum(-1)
    #lpxz = pxz.log_prob(x).view(*pxz.batch_shape[:2], -1).sum(-1)
    lpxz = pxz.log_prob(x).sum(-1)
    lqzx = qzx.log_prob(zs).sum(-1)

    ''' importance-weighted elbo '''
    #loss = log_mean_exp(lpz + lpxz - lqzx).sum()
    loss = log_mean_exp(lpz + lpxz - beta*lqzx).sum() / batch_size # org, works better but why?
    #loss = log_mean_exp(lpxz - beta*(lqzx - lpz)).sum() / batch_size # new bad xx never work

    ''' individual loss components for tracking '''
    mlpz = log_mean_exp(lpz).sum() / batch_size
    mlpxz = log_mean_exp(lpxz).sum() / batch_size
    mlqzx = log_mean_exp(beta*lqzx).sum() / batch_size

    return (-loss, -mlpxz, -mlqzx)


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




