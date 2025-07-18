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

class multicsLoss(nn.Module):
    def __init__(self,ks):
        super(multicsLoss, self).__init__()
        self.ksize=ks
    def forward(self, X,Y,Z):

        ksize=self.ksize
        sigma=3
        Gxx = get_kernel(X, X,ksize)
        Gzz = get_kernel(Z, Z,ksize)
        Gxz = get_kernel(X, Z,ksize)
        
        Gxy = get_kernel(X, Y,ksize)
        Gyy = get_kernel(Y, Y,ksize)
        Gyz = get_kernel(Y, Z,ksize)
        
        r1 = torch.log(sigma*torch.sqrt(torch.mean(Gxx) * torch.mean(Gzz) + 1e-5) / (torch.mean(Gxz) + 1e-5))
        r2 = torch.log(sigma*torch.sqrt(torch.mean(Gyy) * torch.mean(Gzz) + 1e-5) / (torch.mean(Gyz) + 1e-5))
        r3 = torch.log(sigma*torch.sqrt(torch.mean(Gxx) * torch.mean(Gyy) + 1e-5) / (torch.mean(Gxy) + 1e-5))
        r=r1*10+r2+r3*10
        return r
        
import itertools

class GeneralizedMultiCSLoss(nn.Module):
    def __init__(self, ksize, sigma=3.0, pair_weights=None):
        super(GeneralizedMultiCSLoss, self).__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.pair_weights = pair_weights  # Optional dict of {(i,j): weight}

    def forward(self, *modalities):
        """
        Accepts a variable number of modality tensors.
        Each modality should be a tensor of shape [batch_size, feature_dim]
        """
        n = len(modalities)
        loss = 0.0
        eps = 1e-5
        
        for i, j in itertools.combinations(range(n), 2):
            A, B = modalities[i], modalities[j]
            G_aa = get_kernel(A, A, self.ksize)
            G_bb = get_kernel(B, B, self.ksize)
            G_ab = get_kernel(A, B, self.ksize)

            term = torch.log(
                self.sigma * torch.sqrt(torch.mean(G_aa) * torch.mean(G_bb) + eps) / 
                (torch.mean(G_ab) + eps)
            )
            weight = 1.0
            if self.pair_weights and (i, j) in self.pair_weights:
                weight = self.pair_weights[(i, j)]
            loss += weight * term

        return loss
    
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

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

