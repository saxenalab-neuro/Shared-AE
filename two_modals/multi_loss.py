"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
from torch.distributions import LowRankMultivariateNormal
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import LowRankMultivariateNormal
# import torch.nn as nn
import math
import torch.distributions as dist
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence


def get_kernel(X, Z, ksize):
    # Expanding X at dimension 1 and subtracting Z to calculate pairwise squared distances
    G = torch.sum((X.unsqueeze(1) - Z)**2, dim=-1)  # Gram matrix
    
    # Calculating the Gaussian kernel
    G = torch.exp(-G / ksize) / (math.sqrt(2 * math.pi * ksize) * torch.ones_like(-G/ksize))
    return G

class csLoss(nn.Module):
    def __init__(self,ks):
        super(csLoss, self).__init__()
        self.ksize=ks
    def forward(self, X,Z):

        ksize=self.ksize
        Gxx = get_kernel(X, X,ksize)
        Gzz = get_kernel(Z, Z,ksize)
        Gxz = get_kernel(X, Z,ksize)
        r = torch.log(torch.sqrt(torch.mean(Gxx) * torch.mean(Gzz) + 1e-5) / (torch.mean(Gxz) + 1e-5))
        return r
        
class klLoss(nn.Module):
    def __init__(self,model_precision=10,z_dim=12):
        super(klLoss, self).__init__()
        self.model_precision=model_precision
        self.z_dim=z_dim
    def forward(self,latent_dist,z,x,x_rec):
        elbo=-0.5 * (torch.sum(torch.pow(z,2)) + self.z_dim * np.log(2*np.pi))
        pxz_term = -0.5 * np.prod(x.shape[1:]) * (np.log(2*np.pi/self.model_precision))
        l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec.view(x.shape[0],-1), 2), dim=1)
        pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
        elbo = elbo + pxz_term
        elbo = elbo + torch.sum(latent_dist.entropy())
        return -elbo


class klLoss_(nn.Module):
    def __init__(self,prior,model_precision=2,z_dim=32):
        super(klLoss_, self).__init__()
        self.model_precision= nn.Parameter(torch.Tensor([model_precision]))
        self.z_dim=z_dim
        self.prior=prior
        
    def gaussian_likelihood(self, x_hat,  x):
        scale = torch.exp(self.model_precision)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))
        
    def kl_divergence_loss(self,z_dist,prior):
        return kl_divergence(z_dist,prior).sum(-1).sum()

    def kl_divergence_approx(p, q):

        log_p = p.log_prob()
        log_q = q.log_prob()
        
        kl_div = (log_p - log_q).mean()
        
        return kl_div
        
    def forward(self,latent_dist,z,x,x_rec,beta):
        #### E log(p(x|z)) ####
        device=z.get_device()

        reconstruction_loss = nn.BCELoss(reduction='sum')

        if self.prior=='beta':
            p = dist.Beta(torch.ones_like(latent_dist.concentration1),torch.ones_like(latent_dist.concentration0))
        elif self.prior=='normal':
            p = dist.Normal(torch.zeros_like(latent_dist.mean),torch.ones_like(latent_dist.stddev))
        elif self.prior=='dirichlet':
            p = dist.Dirichlet(torch.ones_like(latent_dist.concentration))
        elif self.prior=='laplace':
            p = dist.Laplace(torch.zeros_like(latent_dist.loc),torch.ones_like(latent_dist.scale))

        else:
            p = dist.LowRankMultivariateNormal(torch.zeros(z.shape).to(device), torch.unsqueeze(torch.ones(z.shape),-1).to(device),torch.ones(z.shape).to(device))#torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = latent_dist
        loss=reconstruction_loss(x_rec, x)+ beta*self.kl_divergence_loss(q,p)

        
        return loss
class jsloss(nn.Module):
    def __init__(self):
        super(jsloss, self).__init__()
        
    def forward(self,net_1_logits, net_2_logits):

        net_1_probs = F.softmax(net_1_logits, dim=0)
        net_2_probs = F.softmax(net_2_logits, dim=0)
        
        total_m = 0.5 * (net_1_probs + net_1_probs)
        
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
        return (0.5 * loss)