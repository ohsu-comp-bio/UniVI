import torch
from torch.distributions import Normal
from scvt.utilities._utils import log_mean_exp, kl_divergence, get_device

from scvt.external.evaluate_FOSCTTM import calc_frac
from scvt.utilities._utils import tensor_to_numpy, get_device


from numpy import prod
import numpy as np


# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def objective(loss_func, model, x, n_mc_samples=1, beta=1.0):
    #print(loss_func); assert False
    if loss_func == "iwae":
        (loss, loss1, loss2) = iwae(model, x, n_mc_samples=n_mc_samples, beta=beta)
    elif loss_func == "elbo":
        (loss, loss1, loss2) = elbo(model, x, n_mc_samples=n_mc_samples, beta=beta)
    elif loss_func == "dreg":
        (loss, loss1, loss2) = dreg(model, x, n_mc_samples=n_mc_samples)
    elif loss_func == "m_iwae":
        (loss, loss1, loss2) = m_iwae(model, x, n_mc_samples=n_mc_samples, beta=beta)
    elif loss_func == "m_elbo":
        (loss, loss1, loss2) = m_elbo(model, x, n_mc_samples=n_mc_samples, beta=beta)
    elif loss_func == "m_elbo_org":
        (loss, loss1, loss2) = m_elbo_org(model, x, n_mc_samples=n_mc_samples, beta=beta)
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

def m_elbo_best(model, x, n_mc_samples=1, beta=1.0):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    pz = model._pz(*model.pz_params)  # by hyeyoung

    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []

    kld_z01 = kl_divergence(qz_xs[0], qz_xs[1])
    kld_z10 = kl_divergence(qz_xs[1], qz_xs[0])

    for r, qz_x in enumerate(qz_xs):
        #kld = kl_divergence(qz_x, model.pz(*model.pz_params))  # original

        kld = kl_divergence(qz_x, pz)
        klds.append(kld.sum(-1))  ############> ORIGINAL

        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(*px_zs[d][d].batch_shape[:2], -1)

            lpx_z = (lpx_z * model.vaes[d].llik_scaling).sum(-1) #>>>>>>>>> ORIGINAL

            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)

            lpx_zs.append(lwt.exp() * lpx_z)

    # 1) original objective
    obj_1 = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))

    # 2) put two constraints 
    obj_2 = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0) - kld_z01.sum(-1) - kld_z10.sum(-1))

    # 3) put two constraints + kld-beta 100
    #obj_3 = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - 100 * torch.stack(klds).sum(0) - 100 * kld_z01.sum(-1) - 100 * kld_z10.sum(-1) )
    obj_3 = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - beta * torch.stack(klds).sum(0) - beta * kld_z01.sum(-1) - beta * kld_z10.sum(-1) )


   ############3
    # for logging 
    ############3
    # put .mean(0).sum() in common
    lpx_zs_sum = torch.stack(lpx_zs).sum(0).mean(0).sum()
    klds_sum = torch.stack(klds).sum(0).mean(0).sum()
    kld_z01_sum = kld_z01.sum(-1).mean(0).sum()
    kld_z10_sum = kld_z10.sum(-1).mean(0).sum()
    #kld_z01_sum = kld_z01.sum(0).mean(0).sum() # ????????????????
    #kld_z10_sum = kld_z10.sum(0).mean(0).sum() # ????????????????

    ############3
    # select !!! 
    ############3
    obj = obj_3
    obj_sum = obj.mean(0).sum()
    ############3

    for_log = [lpx_zs_sum, klds_sum, kld_z01_sum, kld_z10_sum, obj_sum]

    #return obj.mean(0).sum(), for_log
    out =  obj.mean(0).sum() / x[0].shape[0]
    return (-out, -out, -out)




def m_elbo_org_OLD(model, x, K=1, beta=1):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []

    lik = 1.0
    pz1 = _get_pz(zss[0])
    pz2 = _get_pz(zss[1])

    qzx1 = qz_xs[0]
    qzx2 = qz_xs[1]
    kld_z01 = kl_divergence(qzx1, qzx2).sum(-1)
    kld_z10 = kl_divergence(qzx2, qzx1).sum(-1)
    kld_ex = [kld_z01, kld_z10]

    for r, qz_x in enumerate(qz_xs):
        #kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        kld = kl_divergence(qz_x, pz1)
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(*px_zs[d][d].batch_shape[:2], -1)
            #print(f'in for: {lpx_z.shape}')

            if lpx_z.shape[-1] == 224:
                lik = 9.0
            lpx_z = (lpx_z * lik).sum(-1)

            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)
            lpx_zs.append(lwt.exp() * lpx_z)
    #for lpxz in lpx_zs:
        #print(lpxz.shape)
    #assert False
    #obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - beta*torch.stack(klds).sum(0)) - beta*torch.stack(kld_ex).sum(0)
    loss = obj.mean(0).sum() / x[0].shape[0]
    return (-loss, -loss, -loss)




def m_elbo_before(model, xs, n_mc_samples=1, beta=1):

    ''' model run '''
    qzxs, pxzs, zss = model(xs, n_mc_samples)

    ''' define variables '''
    qzx1, qzx2 = qzxs
    [[pxz11, pxz12], [pxz21, pxz22]] = pxzs
    z1, z2  = zss
    x1, x2 = xs
    N = x1.shape[0]
    lik = get_lik_ratio(x1.shape[1], x2.shape[1])

    ''' kl divergence '''
    pz = model._pz(*model.pz_params)
    kld_pz = kl_divergence(pz, _get_pz(z1)).sum(-1)

    kld1 = kl_divergence(qzx1, pz).sum(-1)
    kld2 = kl_divergence(qzx2, pz).sum(-1)
    klds = [kld1, kld2]

    kld_z12 = kl_divergence(qzx1, qzx2).sum(-1)
    kld_z21 = kl_divergence(qzx2, qzx1).sum(-1)

    wt1 = (qzx1.log_prob(z1.detach()) - qzx2.log_prob(z1).detach()).sum(-1).exp()
    lpxz1 = pxz11.log_prob(x1).sum(-1) + wt1 * 9* pxz12.log_prob(x2).sum(-1)

    wt2 = (qzx2.log_prob(z2.detach()) - qzx1.log_prob(z2).detach()).sum(-1).exp()
    lpxz2 = wt2 * pxz21.log_prob(x1).sum(-1) + 9* pxz22.log_prob(x2).sum(-1)

    lpxzs = [lpxz1, lpxz2]

    #print(wt1.mean(0).sum())
    #print(wt2.mean(0).sum())
    #print(wt1.shape)
    #print(wt2.shape)
    #assert False

    LPXZ = torch.stack(lpxzs).mean(0).sum() / x1.shape[0]
    KLD = beta * torch.stack(klds).mean(0).sum() / x1.shape[0]

    OBJ = 0.5 * (LPXZ - KLD)

    return (-OBJ, -LPXZ, KLD)

def get_lik_ratio(n_x1, n_x2):
    return n_x1/n_x2


def m_elbo(model, xs, n_mc_samples=1, beta=1):

    ''' model run '''
    qzxs, pxzs, zss = model(xs, n_mc_samples)

    ''' variables '''
    qzx1, qzx2 = qzxs
    [[pxz11, pxz12], [pxz21, pxz22]] = pxzs
    z1, z2  = zss
    x1, x2 = xs
    N = x1.shape[0]
    lik = get_lik_ratio(x1.shape[1], x2.shape[1])

    ''' prior  '''
    pz = model._pz(*model.pz_params)
    kld_pz = kl_divergence(pz, _get_pz(z1)).sum(-1)

    ''' kl divergence '''
    kld1 = kl_divergence(qzx1, pz).sum(-1)
    kld2 = kl_divergence(qzx2, pz).sum(-1)
    klds = [kld1, kld2]

    ''' kl divergence extra '''
    kld_z12 = kl_divergence(qzx1, qzx2).sum(-1)
    kld_z21 = kl_divergence(qzx2, qzx1).sum(-1)


    wt1 = (qzx1.log_prob(z1.detach()) - qzx2.log_prob(z1).detach()).sum(-1).exp()
    #wt1 = 1
    lpxz1 = pxz11.log_prob(x1).sum(-1) + wt1 * lik* pxz12.log_prob(x2).sum(-1)

    wt2 = (qzx2.log_prob(z2.detach()) - qzx1.log_prob(z2).detach()).sum(-1).exp()
    #wt2 = 1
    lpxz2 = wt2 * pxz21.log_prob(x1).sum(-1) + lik* pxz22.log_prob(x2).sum(-1)

    lpxzs = [lpxz1, lpxz2]
    #obj = 1/2 * ((torch.stack(lpxzs).sum(0) - torch.stack(klds).sum(0)))

    RECON = 1/2 * torch.stack(lpxzs).sum(0)
    KLD = 1/2 * torch.stack(klds).sum(0)

    #print(f'RECON: {RECON.shape}')
    #print(f'KLD: {KLD.shape}')

    RECON = RECON.mean(0).sum()
    KLD = KLD.sum()
    constraints =  (kld_z12 + kld_z21 + (kld_z12 - kld_z21)**2).sum() # m_elbo_new
    #constraints = (kld_z12 + kld_z21).sum() # m_elbo_old
    KLTERM = beta*(KLD + constraints)

    #OBJ = RECON - beta*KLD
    OBJ = RECON - KLTERM

    print(f'\n')
    print(f'KLD mean: {KLD/N}')
    print(f'kld1: {kld1.sum()/N}')
    print(f'kld2: {kld2.sum()/N}')
    print(f'kld_z12: {kld_z12.sum()/N}')
    print(f'kld_z21: {kld_z21.sum()/N}')
    print(f'constraints: {constraints/N}')
    print(f'KLTERM: {KLTERM/N}')
    print(f'RECON: {RECON/N}')
    print(f'lpxz1: {lpxz1.sum()/N}')
    print(f'lpxz2: {lpxz2.sum()/N}')
    print(f'wt1: {wt1.sum()/N}')
    print(f'wt2: {wt2.sum()/N}')
    print(f'OBJ: {OBJ/N}')
    #assert False
    return (-OBJ/N, -RECON/N, KLTERM/N)

    lw1 = (lpxz1 + lpz1 - lqzx1 - beta*kld_z12 - beta*kld_z21 - 1*(kld_z12 - kld_z21)**2) # 1 x 300
    lw2 = (lpxz2 + lpz1 - lqzx2 - beta*kld_z12 - beta*kld_z21 - 1*(kld_z12 - kld_z21)**2) # 1 x 300
    #lw1 = (lpxz1 + lpz1 - lqzx1) #xxxxxxx
    #lw2 = (lpxz2 + lpz1 - lqzx2) #xxxxxxx


def m_elbo_org(model, xs, n_mc_samples=1, beta=1):

    ''' model run '''
    qzxs, pxzs, zss = model(xs, n_mc_samples)

    ''' variables '''
    qzx1, qzx2 = qzxs
    [[pxz11, pxz12], [pxz21, pxz22]] = pxzs
    z1, z2  = zss
    x1, x2 = xs
    N = x1.shape[0]
    lik = get_lik_ratio(x1.shape[1], x2.shape[1])

    ''' prior  '''
    pz = model._pz(*model.pz_params)
    kld_pz = kl_divergence(pz, _get_pz(z1)).sum(-1)

    ''' kl divergence '''
    kld1 = kl_divergence(qzx1, pz).sum(-1)
    kld2 = kl_divergence(qzx2, pz).sum(-1)
    klds = [kld1, kld2]

    ''' kl divergence extra '''
    kld_z12 = kl_divergence(qzx1, qzx2).sum(-1)
    kld_z21 = kl_divergence(qzx2, qzx1).sum(-1)


    wt1 = (qzx1.log_prob(z1.detach()) - qzx2.log_prob(z1).detach()).sum(-1).exp()
    lpxz1 = pxz11.log_prob(x1).sum(-1) + wt1 * lik* pxz12.log_prob(x2).sum(-1)

    wt2 = (qzx2.log_prob(z2.detach()) - qzx1.log_prob(z2).detach()).sum(-1).exp()
    lpxz2 = wt2 * pxz21.log_prob(x1).sum(-1) + lik* pxz22.log_prob(x2).sum(-1)

    lpxzs = [lpxz1, lpxz2]
    #obj = 1/2 * ((torch.stack(lpxzs).sum(0) - torch.stack(klds).sum(0)))

    RECON = 1/2 * torch.stack(lpxzs).sum(0)
    KLD = 1/2 * torch.stack(klds).sum(0)

    #print(f'RECON: {RECON.shape}')
    #print(f'KLD: {KLD.shape}')

    RECON = RECON.mean(0).sum()
    KLD = KLD.sum()

    OBJ = RECON - beta*KLD

    #print(f'RECON: {RECON}')
    #print(f'KLD: {KLD}')
    #print(f'OBJ: {OBJ}')
    #assert False
    return (-OBJ/N, -RECON/N, beta*KLD/N)


    ''' loss component for modality 1 '''
    lpz1 = pz.log_prob(z1).sum(-1)
    lqzx1 = log_mean_exp(torch.stack([qzx1.log_prob(z1), qzx2.log_prob(z1)])).sum(-1) #*********
    lpxz1 = log_mean_exp(torch.stack([pxz11.log_prob(x1).sum(-1), lik*pxz12.log_prob(x2).sum(-1)])) #*********

    ''' loss component for modality 2 '''
    lpz2 = pz.log_prob(z2).sum(-1)
    lqzx2 = log_mean_exp(torch.stack([qzx1.log_prob(z2), qzx2.log_prob(z2)])).sum(-1) #*********
    lpxz2 = log_mean_exp(torch.stack([pxz21.log_prob(x1).sum(-1), lik*pxz22.log_prob(x2).sum(-1)])) #********

    ''' loss component for modality 2 '''
    lw1 = (lpxz1 + lpz1 - lqzx1 - beta*kld_z12 - beta*kld_z21 - 1*(kld_z12 - kld_z21)**2) # 1 x 300
    lw2 = (lpxz2 + lpz1 - lqzx2 - beta*kld_z12 - beta*kld_z21 - 1*(kld_z12 - kld_z21)**2) # 1 x 300
    #lw1 = (lpxz1 + lpz1 - lqzx1) #xxxxxxx
    #lw2 = (lpxz2 + lpz1 - lqzx2) #xxxxxxx

    lw = log_mean_exp(torch.stack([lw1, lw2]))
    #lw = torch.stack([lw1, lw2]).sum(0) # xxx
    #lw = torch.stack([lw1, lw2]).mean(0)

    #print(lw1.shape)
    #print(lw2.shape)
    #print(lw.shape)
    #print(lik)

    #lw = lw.sum()
    lw = lw.mean(0).sum()

    return ( -lw/N, -lw1.mean(0).sum()/N, -lw2.mean(0).sum()/N )

    #KLD = beta * torch.stack(klds).mean(0).sum() / x1.shape[0]
    #OBJ = 0.5 * (LPXZ - KLD)
    #return (-OBJ, -LPXZ, KLD)



def m_iwae(model, xs, n_mc_samples=1, beta=1):

    ''' model run '''
    qzxs, pxzs, zss = model(xs, n_mc_samples)

    ''' variables '''
    qzx1, qzx2 = qzxs
    [[pxz11, pxz12], [pxz21, pxz22]] = pxzs
    z1, z2  = zss
    x1, x2 = xs
    N = x1.shape[0]
    #lik = get_lik_ratio(x1.shape[1], x2.shape[1])
    lik = 1.0 ##################################################################

    ''' prior  '''
    pz = model._pz(*model.pz_params)
    kld_pz = kl_divergence(pz, _get_pz(z1)).mean(-1)

    ''' kl divergence '''
    kld1 = kl_divergence(qzx1, pz).mean(-1)
    kld2 = kl_divergence(qzx2, pz).mean(-1)
    klds = [kld1, kld2]

    ''' kl divergence extra '''
    kld_z12 = kl_divergence(qzx1, qzx2).mean(-1)
    kld_z21 = kl_divergence(qzx2, qzx1).mean(-1)

    ''' loss component for modality 1 '''
    lpz1 = pz.log_prob(z1).mean(-1).mean(0)
    #lqzx1 = log_mean_exp(torch.stack([qzx1.log_prob(z1).mean(0), qzx1.log_prob(z2).mean(0)])).mean(-1) # test
    lqzx1 = log_mean_exp(torch.stack([qzx1.log_prob(z1).mean(0), qzx2.log_prob(z1).mean(0)])).mean(-1) # oooo
    #lpxz1 = log_mean_exp(torch.stack([pxz11.log_prob(x1).mean(-1).mean(0), lik*pxz12.log_prob(x2).mean(-1).mean(0)])) #*********
    lpxz1 = log_mean_exp(torch.stack([pxz11.log_prob(x1).mean(-1).mean(0), lik*pxz12.log_prob(x2).mean(-1).mean(0)])) #*********

    ''' loss component for modality 2 '''
    lpz2 = pz.log_prob(z2).mean(-1).mean(0)
    #lqzx2 = log_mean_exp(torch.stack([qzx2.log_prob(z1).mean(0), qzx2.log_prob(z2).mean(0)])).mean(-1) # test
    lqzx2 = log_mean_exp(torch.stack([qzx1.log_prob(z2).mean(0), qzx2.log_prob(z2).mean(0)])).mean(-1) # oooo
    #lpxz2 = log_mean_exp(torch.stack([pxz21.log_prob(x1).mean(-1).mean(0), lik*pxz22.log_prob(x2).mean(-1).mean(0)])) #********
    lpxz2 = log_mean_exp(torch.stack([pxz21.log_prob(x1).mean(-1).mean(0), lik*pxz22.log_prob(x2).mean(-1).mean(0)])) #********

    ''' loss component for modality 2 '''
    #constraints = beta * ( kld_z12 + kld_z21 ) + (kld_z12 - kld_z21)**2 # 1 x 300
    #constraints = beta * ( kld_z12 + kld_z21 ) # maybe
    #constraints = ( kld_z12 + kld_z21 + kld_pz)  # xxxxx
    constraints = ( kld_z12 + kld_z21 )  # xxxxx

    lw1 = lpxz1 + lpz1 - beta*(lqzx1 + constraints)
    lw2 = lpxz2 + lpz2 - beta*(lqzx2 + constraints)

    # TEST
    #lw1 = lpxz1 - beta*(lqzx1 + constraints - lpz1) # xxxxxx
    #lw2 = lpxz2 - beta*(lqzx2 + constraints - lpz2) # xxxxxx

    #lw1 = lw1.mean(0) # testj
    #lw2 = lw2.mean(0) # testj

    #lw1 = (lpxz1 + lpz1 - 1*lqzx1 - constraints ) #********
    #lw2 = (lpxz2 + lpz2 - 1*lqzx2 - constraints ) #********

    #lw1 = (lpxz1 + lpz1 - lqzx1 - beta*kld_z12 - beta*kld_z21 - 2*(kld_z12 - kld_z21)**2) # 1 x 300
    #lw2 = (lpxz2 + lpz2 - lqzx2 - beta*kld_z12 - beta*kld_z21 - 2*(kld_z12 - kld_z21)**2) # 1 x 300
    #lw1 = (lpxz1 + lpz1 - lqzx1) #xxxxxxx
    #lw2 = (lpxz2 + lpz2 - lqzx2) #xxxxxxx

    lw = log_mean_exp(torch.stack([lw1, lw2]))
    #lw = lw.mean(0).mean()
    lw = lw.mean()

    print(f'\n')
    print(f'kld1: {kld1.mean()/N}')
    print(f'kld2: {kld2.mean()/N}')
    print(f'kld_z12: {kld_z12.mean()/N}')
    print(f'kld_z21: {kld_z21.mean()/N}')
    print(f'constraints: {constraints.mean()/N}')
    print(f'lpz1: {lpz1.mean()/N}')
    print(f'lpz2: {lpz2.mean()/N}')
    print(f'lpxz1: {lpxz1.mean()/N}')
    print(f'lpxz2: {lpxz2.mean()/N}')
    print(f'lqzx1: {lqzx1.mean()/N}')
    print(f'lqzx2: {lqzx2.mean()/N}')
    print(f'OBJ: {lw/N}')
    #assert False


    return ( -lw/N, -lw1.mean(0).mean()/N, constraints.mean()/N )

    #KLD = beta * torch.stack(klds).mean(0).sum() / x1.shape[0]
    #OBJ = 0.5 * (LPXZ - KLD)
    #return (-OBJ, -LPXZ, KLD)



    #print(z1.shape)
    #print(z2.shape)
    #assert False
    #fos = calc_frac(tensor_to_numpy(z1.mean(0)), tensor_to_numpy(z2.mean(0)))
    #print(fos)
    #print(fos.shape)
    #fos = np.ndarray(fos)
    #assert False
    #fos = torch.from_numpy(fos).float().to(get_device())
    #fos = torch.as_tensor(fos)
    #print(fos)
    #lw = lw.sum() - 2*kld_z12.sum() - 2*kld_z21.sum() - fos # fos
    #lw = lw.sum() - 2*kld_z12.sum() - 2*kld_z21.sum() - 2*(kld_z12.sum() - kld_z21.sum())**2 - 50*fos  # fos2
    #lw = lw.sum() - 2*(kld_z12.sum() - kld_z21.sum())**2 - 50*fos  # fos_new #xxxxxxxxxxxx

    #lw = lw.sum() - 2*kld_z12.sum() - 2*kld_z21.sum() - 2*(kld_z12.sum() - kld_z21.sum())**2 # dbl_all  #**********
    #lw = lw.sum() - 2*kld_z12.sum() - 2*kld_z21.sum() - 5*(kld_z12.sum() - kld_z21.sum())**2 # dbl_all  ooooo 
    #lw = lw.sum()
    #lw = lw.sum() - 2*kld_z12.sum() - 2*kld_z21.sum() # rem_sq xxxxxxxxxxxxx
    #lw = lw.sum() - 2*kld_z12.sum() - 2*kld_z21.sum() - 2*(kld_z12.sum() - kld_z21.sum())**2 - 2*kld_pz.sum()# dbl_all_pz 
    #xxxxxxxxxxxxxxxxxx

    #lw = lw.sum() - kld_z12.sum()**2 - kld_z21.sum()**2 # two_sq xxxxxxxx
    #lw = lw.sum() - kld_z12.sum()**2 - kld_z21.sum()**2 - (kld_z12.sum() - kld_z21.sum())**2# two_sq_all xxxxxx

    #print(f'lw1: {lw1.mean(0).sum(0)/N}')
    #print(f'lw2: {lw2.mean(0).sum(0)/N}')
    #print(f'lw: {lw/N}')
    #print(f'kld_z12: {kld_z12.mean(0).sum(0)/N}')
    #print(f'kld_z21: {kld_z21.mean(0).sum(0)/N}')



def m_elbo_scvt(model, xs, n_mc_samples=1, beta=1):

    x1 = xs[0]
    x2 = xs[1]

    qzxs, pxzs, zss = model(xs, n_mc_samples)
    #print(zss[0].sum(-1).mean(0).sum())

    qzx1 = qzxs[0]
    qzx2 = qzxs[1]

    pxz11 = pxzs[0][0]
    pxz12 = pxzs[0][1]
    pxz21 = pxzs[1][0]
    pxz22 = pxzs[1][1]


    zs1 = zss[0]
    zs2 = zss[1]

    pz1 = _get_pz(zs1)
    pz2 = _get_pz(zs2)

    '''
    print(f'x1: {x1.shape}')
    print(f'x2: {x2.shape}')
    print(f'pz1: {pz1.batch_shape}')
    print(f'pz2: {pz2.batch_shape}')
    print(f'qzx1: {qzx1.batch_shape}')
    print(f'qzx2: {qzx2.batch_shape}')
    print(f'pxz11:{ pxz11.batch_shape }')
    print(f'pxz12:{ pxz12.batch_shape }')
    print(f'pxz21:{ pxz21.batch_shape }')
    print(f'pxz22:{ pxz22.batch_shape }')
    '''

    kld_z01 = kl_divergence(qzx1, qzx2).sum(-1)
    kld_z10 = kl_divergence(qzx2, qzx1).sum(-1)

    klds = [kl_divergence(qzx1, pz1).sum(-1), kl_divergence(qzx2, pz2).sum(-1)]


    lw1 = (qzx1.log_prob(zs2) - qzx2.log_prob(zs2)).sum(-1)
    lw2 = (qzx2.log_prob(zs1) - qzx1.log_prob(zs1)).sum(-1)
    lw1=lw1.exp()
    lw2=lw2.exp()

    #print(lw1.exp())
    #print(lw2.exp())
    #print(lw1)
    #print(lw2)
    #lw1=1
    #lw2=1
    #assert False

    fpxz11 = pxz11.log_prob(x1).sum(-1)
    fpxz12 = lw1 * pxz12.log_prob(x2).sum(-1) * 9
    fpxz21 = lw2 * pxz21.log_prob(x1).sum(-1)
    fpxz22 = pxz22.log_prob(x2).sum(-1) * 9
    #print('\n')
    #print(fpxz11.sum())
    #print(fpxz12.sum())
    #print(fpxz21.sum())
    #print(fpxz22.sum())
    ##beta = 100
    #assert False

    lpxzs = [fpxz11,fpxz12,fpxz21,fpxz22]


    # 1) original objective
    #obj_1 = (1 / 2) * (torch.stack(lpxzs).sum(0) - torch.stack(klds).sum(0))

    obj_2 = (1 / 2) * (torch.stack(lpxzs).sum(0) - beta * torch.stack(klds).sum(0))

    #obj_3 = (1 / 2) * (torch.stack(lpxzs).sum(0) - beta * torch.stack(klds).sum(0) - beta * kld_z01.sum(0) - beta * kld_z10.sum(0) )  #XXXXXXXXXXXXXXX
    obj_3 = (1 / 2) * (torch.stack(lpxzs).sum(0) - beta * torch.stack(klds).sum(0) - beta * kld_z01 - beta * kld_z10 )

    obj = obj_3 #########


    #print(f'obj_1: {obj_1.shape}')
    #print(f'obj_3: {obj_3.shape}')
    ############3
    # for logging 
    ############3
    # put .mean(0).sum() in common
    lpxzs_sum = torch.stack(lpxzs).sum(0).mean(0).sum()
    klds_sum = torch.stack(klds).sum(0).mean(0).sum()
    kld_z01_sum = kld_z01.sum(-1).mean(0).sum()
    kld_z10_sum = kld_z10.sum(-1).mean(0).sum()
    #kld_z01_sum = kld_z01.sum(0).mean(0).sum() # ????????????????
    #kld_z10_sum = kld_z10.sum(0).mean(0).sum() # ????????????????

    ############3
    # select !!! 
    ############3
    #obj_sum = obj.mean(0).sum()
    ############3

    #loss = obj.mean(0).sum() / x1[0]
    loss = obj.mean(0).sum() / x1.shape[0]
    #print(f'loss: {loss}')
    #print(f'loss: {loss.shape}')
    #assert False
    return (-loss, -loss, -loss)




def _m_iwae(model, x, n_mc_samples=1, beta=1.0):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, n_mc_samples)

    qzx1 = qz_xs[0]
    qzx2 = qz_xs[1]
    kld_z01 = kl_divergence(qz_xs[0], qz_xs[1]).sum(-1)
    kld_z10 = kl_divergence(qz_xs[1], qz_xs[0]).sum(-1)
    kld_ex = [kld_z01, kld_z10]

    pz_normal = _get_pz(zss[0])
    pz = model._pz(*model.pz_params)
    kld_pz = kl_divergence(pz, pz_normal).sum(-1)

    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model._pz(*model.pz_params).log_prob(zss[r]).sum(-1)

        # original
        #lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))

        # abby + pz xxxx
        #lqz_x = torch.stack( [ qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs ]) + torch.stack(kld_ex) + torch.stack(kld_pz)

        # abby  xxxxxxxxxxxxxxxxxxxxxxx
        lqz_x = torch.stack( [ qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs ]) + torch.stack(kld_ex).mean(0)

        # best
        #lqz_x = torch.stack( [ qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs ]).mean(0) + kld_z01 + kld_z10
        #lqz_x = torch.stack( [ qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs ]).mean(0)

        #lqz_x = torch.stack( [ qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs ]).sum(0) + kld_z01 + kld_z10 #XXXXXXXXXXX
        # pz xxxxxxx
        #lqz_x = torch.stack( [ qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs ]).mean(0) + kld_z01 + kld_z10 + kld_pz

        #lqz_x = log_mean_exp(beta*lqz_x)

        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        #lw = lpz + lpx_z - beta*lqz_x
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size



def m_iwae_what(model, x, n_mc_samples=1, beta=1.0):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    #S = compute_microbatch_split(x, n_mc_samples)
    #x_split = zip(*[_x.split(S) for _x in x])
    #lw = [_m_iwae(model, _x, n_mc_samples) for _x in x_split]
    lw = _m_iwae(model, x, n_mc_samples, beta)
    #lw = torch.cat(lw, 1)  # concat on batch
    #return log_mean_exp(lw).sum()
    out =  log_mean_exp(lw).sum() / x[0].shape[0]
    return (-out, -out, -out)



def m_iwae_org(model, x, n_mc_samples=1, beta=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    lw = [_m_iwae(model, _x, n_mc_samples) for _x in x_split]
    lw = torch.cat(lw, 1)  # concat on batch
    return log_mean_exp(lw).sum()


def _m_iwae_org(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size




def m_iwae_best(model, xs, n_mc_samples=1, beta=1):

    x1 = xs[0]
    x2 = xs[1]

    qzxs, pxzs, zss = model(xs, n_mc_samples)
    #print(zss[0].sum(-1).mean(0).sum())

    qzx1 = qzxs[0]
    qzx2 = qzxs[1]

    pxz11 = pxzs[0][0]
    pxz12 = pxzs[0][1]
    pxz21 = pxzs[1][0]
    pxz22 = pxzs[1][1]

    '''
    print(f'x1: {x1.shape}')
    print(f'x2: {x2.shape}')
    print(f'pxz11:{ pxz11.batch_shape }')
    print(f'pxz12:{ pxz12.batch_shape }')
    print(f'pxz21:{ pxz21.batch_shape }')
    print(f'pxz22:{ pxz22.batch_shape }')


   pz = model._pz(*model.pz_params)  # by hyeyoung

    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []

    kld_z01 = kl_divergence(qz_xs[0], qz_xs[1])
    kld_z10 = kl_divergence(qz_xs[1], qz_xs[0])

    for r, qz_x in enumerate(qz_xs):
        #kld = kl_divergence(qz_x, model.pz(*model.pz_params))  # original

        kld = kl_divergence(qz_x, pz)
        klds.append(kld.sum(-1))  ############> ORIGINAL



    '''

    zs1 = zss[0]
    zs2 = zss[1]

    #pz1 = _get_pz(zs1)
    #pz2 = _get_pz(zs2)
    pz = model._pz(*model.pz_params)  # by hyeyoung

    lpz1 = pz.log_prob(zs1).sum(-1)
    lpz2 = pz.log_prob(zs2).sum(-1)

    beta2 = 1
    lqzxs = beta * log_mean_exp(\
		torch.stack([qzx1.log_prob(zs1), qzx2.log_prob(zs2)]).sum(-1)) \
		+ beta2 * (kl_divergence(qzx1, qzx2).sum(-1) + kl_divergence(qzx2, qzx1).sum(-1))
    '''
    lqzxs = beta * log_mean_exp(\
			torch.stack([qzx1.log_prob(zs1), \
			qzx2.log_prob(zs2)]).sum(-1)) 
    '''

    lpxz11 = pxz11.log_prob(x1).sum(-1)
    lpxz12 = pxz12.log_prob(x2).sum(-1) * 18

    lpxz21 = pxz21.log_prob(x1).sum(-1)
    lpxz22 = pxz22.log_prob(x2).sum(-1)* 18


    # original best
    lpxz1 = torch.stack([lpxz11,lpxz12]).sum(0)
    lpxz2 = torch.stack([lpxz22,lpxz21]).sum(0) 

    '''
    lpxz1 = torch.stack([lpxz11,lpxz21]).sum(0)
    lpxz2 = torch.stack([lpxz12,lpxz22]).sum(0) 
    '''

    lw1 = lpz1 + lpxz1 - lqzxs
    lw2 = lpz2 + lpxz2 - lqzxs

    lws = torch.cat([lw1, lw2])

    loss = log_mean_exp(lws).sum()/x1.shape[0]
    #print(-loss)
    return (-loss, -loss, -loss)


def _dreg(model, x, n_mc_samples):
    """DREG estimate for log p_\theta(x) -- fully vectorised."""
    #_, px_z, zs = model(x, n_mc_samples)
    qzx, pxz, zs = model(x, n_mc_samples)

    pz = _get_pz(zs) #abby

    lpz = pz.log_prob(zs).sum(-1)
    lpxz = pxz.log_prob(x).view(*pxz.batch_shape[:2], -1).sum(-1)

    #qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi

    qzx = model._qzx(model.qzx_mu.detach(), model.qzx_var.sqrt().detach())
    lqzx = qzx.log_prob(zs).sum(-1)
    lw = lpz + lpxz - lqzx
    #print('lqzx.shape: ',lqzx.shape)
    #print('zs.shape: ',zs.shape)

    return lw, zs


def dreg(model, x, n_mc_samples=1):
    """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    lw, zs = _dreg(model, x, n_mc_samples)

    #lw = torch.cat(lw, 1)  # concat on batch
    #zs = torch.cat(zs, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return ((grad_wt * lw).sum(), 0.1, 0.1)


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


def elbo(model, x, n_mc_samples=1, beta=1.0):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, zs = model(x)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) 
    #print(qz_x)
    #print(model._pz(*model.pz_params))
    #assert False
    kld = kl_divergence(qz_x, model._pz(*model.pz_params))
    #print(kld)
    #assert False
    out = (lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum()
    out = out / x.shape[0]
    return (-out, -out, -out)

def m_elbo_old(model, x, n_mc_samples=1, beta=1.0):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """

    qz_xs, px_zs, zss = model(x)

    #lik = 1.0
    #pz1 = _get_pz(zss[0])
    #pz2 = _get_pz(zss[1])

    qzx1 = qz_xs[0]
    qzx2 = qz_xs[1]
    kld_z01 = kl_divergence(qz_xs[0], qz_xs[1]).sum(-1)
    kld_z10 = kl_divergence(qz_xs[1], qz_xs[0]).sum(-1)
    kld_ex = [kld_z01, kld_z10]



    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model._pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            if x[d].shape[1] == 224:
                lik = 9.
            else:
                lik = 1.
            lpx_z = px_zs[d][d].log_prob(x[d]).view(*px_zs[d][d].batch_shape[:2], -1)
            lpx_z = (lpx_z * lik).sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)
            lpx_zs.append(lwt.exp() * lpx_z)
    # original
    #obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))

    # abby
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - beta*torch.stack(klds).sum(0) - beta*torch.stack(kld_ex).sum(0))
    out = obj.mean(0).sum() / x[0].shape[0]
    #return obj.mean(0).sum()
    return (-out, -out, -out)



def elbo_my(model, x, n_mc_samples=1, beta=1.0):

    batch_size = x.shape[0]

    qzx, pxz, zs = model(x)
    #pz = model._pz(*model.pz_params())
    pz = _get_pz(zs) #abby
    
    kl_beta = beta*kl_divergence(qzx, pz).sum(-1)
    #kl = kl_divergence(qzx, pz).sum(-1)
    recon = pxz.log_prob(x).sum(-1)
    
    #batch_size = 1 ######################## for test
    loss = (recon - kl_beta).mean(0).sum() / batch_size
    #print(f'             in elbo: {-loss:.2f}')

    mrecon = recon.mean(0).sum() / batch_size
    mkl = kl_beta.mean(0).sum() / batch_size
    
    return (-loss, -mrecon, mkl)



