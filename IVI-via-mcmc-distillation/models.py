import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


class GMM(object):
    name = 'gmm'
    def __init__(self, n_clusters=8, sigma=0.1, r=1.):
        thetas = torch.linspace(0, 2*math.pi, n_clusters+1)[:-1]
        self.means = torch.stack((thetas.cos(), thetas.sin()), dim=1) * r
        self.std = sigma
        self.n_clusters = n_clusters
        
        self.n_dim = self.means.size()[1]
    
    def sample(self, n_samp=1000):
        samp_clusters = torch.randint(0, self.n_clusters, (n_samp,)).long()
        return torch.randn(n_samp, 2) * self.std + self.means[samp_clusters]
        
    def logp(self, x):
        squared_term = -torch.sum((x.unsqueeze(1) - self.means.unsqueeze(0))**2, dim=-1)/2./self.std**2
        return torch.logsumexp(squared_term, dim=-1) - math.log(self.n_clusters) - math.log(2*math.pi) - math.log(self.std**2)
    
    def score(self, x):
        diff_term = x.unsqueeze(1) - self.means.unsqueeze(0)
        squared_term = -torch.sum(diff_term**2, dim=-1)/2./self.std**2
        wts_term = F.softmax(squared_term, dim=-1)
        
        return torch.sum(-diff_term/self.std**2 * wts_term.unsqueeze(2), dim=1)
    
    
class LR(object):
    name = 'lr'
    def __init__(self, data):
        self.X, self.Y = data
        self.N = len(self.X)
        
        self.YX = torch.mm(self.Y.t(), self.X)
    
    def sample(self, n_samp=1000, stepsz=0.01, max_iter=1000):
        samp_beta = torch.zeros(n_samp, 2)
        for i in range(1, max_iter+1):
            samp_beta = samp_beta + stepsz * self.score(samp_beta) + math.sqrt(2*stepsz) * torch.randn(n_samp, 2)
        
        return samp_beta
        
    def logp(self, beta):
        inner_prod = torch.mm(self.X, beta.t())
        return torch.mm(self.Y.t(), inner_prod) - torch.sum(torch.log1p(torch.exp(inner_prod)), dim=0)
    
    def score(self, beta):
        inner_prod = torch.mm(self.X, beta.t())
        return self.YX - torch.sum(torch.sigmoid(inner_prod).unsqueeze(2) * self.X.unsqueeze(1), dim=0)