import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

from models.networks import SIMINet
from models.target_models import target_distribution
from tqdm import tqdm
from utils.annealing import annealing
from utils.parse_config import parse_config
import logging
import time
from utils.kernels import *



class KernelSIVI(object):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.target = self.config.target_score
        self.trainpara = self.config.train
        self.num_iters = self.trainpara.num_perepoch * self.config.train.num_epochs
        self.iter_idx = 0
        self.kernel = {"gaussian":gaussian_kernel, "laplace":laplace_kernel,
                       "IMQ":IMQ_kernel, "Riesz": Riesz_kernel}[self.config.kernel]
        
    def preprocess(self):
        self.save_fig_to_path = os.path.join(self.config.log_path, "fig")
        os.makedirs(self.save_fig_to_path,exist_ok=True)

    def loss_from_samples(self, X, neg_score, X_aux, neg_score_aux, alpha=1.0):
        """Original conditional-score loss, or its Gaussian Stein-kernel form."""
        estimator = getattr(self.trainpara, "estimator", "conditional")
        sx = alpha * self.target_model.score(X)
        sy = alpha * self.target_model.score(X_aux)
        detach_width = (getattr(self.trainpara, "stein_detach_bandwidth", True)
                        if estimator == "stein" else self.trainpara.detach)
        kernel = self.kernel
        if (self.config.kernel == "gaussian"
                and getattr(self.trainpara, "pairwise_backend", "upstream") == "matmul"):
            kernel = gaussian_kernel_matmul
        fixed_h = getattr(self.trainpara, "fixed_bandwidth", -1)
        if estimator == "conditional":
            K, h = kernel(X, X_aux, h=fixed_h, get_width=True, detach=detach_width)
            loss = (((sx + neg_score) @ (sy + neg_score_aux).T) * K).mean()
        elif estimator == "stein":
            if self.config.kernel != "gaussian":
                raise ValueError("The local Stein implementation requires a Gaussian kernel")
            K, h = kernel(X, X_aux, h=fixed_h, get_width=True, detach=detach_width)
            h2 = h * h
            distances = (X.square().sum(1, keepdim=True) + X_aux.square().sum(1)[None]
                         - 2 * (X @ X_aux.T)).clamp_min(0)
            # (sx - sy) dot (x - y), without a [N,N,D] temporary.
            cross = ((sx * X).sum(1, keepdim=True) + (sy * X_aux).sum(1)[None]
                     - sx @ X_aux.T - X @ sy.T)
            inner = sx @ sy.T + cross / h2 + X.shape[1] / h2 - distances / h2**2
            loss = (inner * K).mean()
        else:
            raise ValueError(f"Unknown estimator: {estimator}")
        return loss, h

    def training_loss(self, iteration):
        Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim], device=self.device)
        X, neg_score = self.SemiVInet(Z)
        Z_aux = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim], device=self.device)
        X_aux, neg_score_aux = self.SemiVInet(Z_aux)
        alpha = annealing(iteration, warm_up_interval=self.num_iters // self.trainpara.warm_ratio,
                          anneal=self.trainpara.annealing)
        return self.loss_from_samples(X, neg_score, X_aux, neg_score_aux, alpha)
    def learn(self):
        self.preprocess()

        self.target_model = target_distribution[self.target](self.device)
        self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)

        annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)

        optimizer_VI = torch.optim.Adam(self.SemiVInet.parameters(), lr = self.trainpara.lr_SIMI, betas=(.9, .99))
        
        scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)

        loss_list = []
        for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):
            with torch.no_grad():
                X = self.SemiVInet.sampling(num = self.config.sampling.num)
                torch.save(X,os.path.join(self.save_fig_to_path, str(self.iter_idx)+'.pt'))
                torch.save(self.SemiVInet.state_dict(), os.path.join(self.save_fig_to_path, "model"+str(self.iter_idx)+'.pt'))  
            if (epoch - 1)%self.config.sampling.visual_time ==0 or epoch ==self.trainpara.num_epochs:
                if self.target in ["banana", "multimodal", "x_shaped"]:
                    save_to_path = os.path.join(self.save_fig_to_path, str(self.iter_idx)+'.png')
                    bbox = {"multimodal":[-5, 5, -5, 5], "banana":[-3.5,3.5,-6,1], "x_shaped":[-5,5,-5,5]}
                    quiver_plot = False
                    self.target_model.contour_plot(bbox[self.target], fnet = None, samples=X.cpu().numpy(), save_to_path=save_to_path, quiver = quiver_plot, t = self.iter_idx)

            for i in range(1, self.trainpara.num_perepoch+1):
                self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i
                # ============================================================== #
                #                      Train the SemiVInet                       #
                # ============================================================== #
                Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                X, neg_score_implicit = self.SemiVInet(Z)
                if not self.trainpara.ustat:
                    Z_aux = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                    X_aux, neg_score_implicit_aux = self.SemiVInet(Z_aux)
                    loss_kxy, _ = self.loss_from_samples(
                        X, neg_score_implicit, X_aux, neg_score_implicit_aux,
                        annealing_coef(self.iter_idx))
                else:
                    compu_targetscore = self.target_model.score(X) * annealing_coef(self.iter_idx)
                    x_xaux_dist = torch.matmul((compu_targetscore + neg_score_implicit),(compu_targetscore  + neg_score_implicit).T)
                    x_xaux_dist.fill_diagonal_(0)
                    loss_kxy = (x_xaux_dist * (self.kernel(X,X))).mean()
                    optimizer_VI.zero_grad()
                    loss_kxy.backward()
                optimizer_VI.zero_grad()
                loss_kxy.backward()
                optimizer_VI.step()
                scheduler_VI.step()
                if i % (self.trainpara.num_perepoch//2) == 0: 
                    loss_list.append(loss_kxy.item())
                    logger.info('{} Iter {} | Loss: {:.4f}'.format(time.asctime(time.localtime(time.time())), self.iter_idx, loss_kxy))
                    logger.info("neg_score_implicit: {:.4f}".format(neg_score_implicit.abs().mean().item()))
        torch.save(X.detach().cpu(),os.path.join(self.save_fig_to_path, str(self.iter_idx)+'.pt'))
        np.save(os.path.join(self.save_fig_to_path, "loss"+'.npy'), np.array(loss_list))
if __name__ == "__main__":
    seednow = 2022
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    random.seed(seednow)
    torch.backends.cudnn.deterministic = True
        
    config = parse_config()

    datetimelabel = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if config.log_stick else "Now"
    config.log_path = os.path.join("expkernelSIVI", config.target_score, "{}".format(datetimelabel))
    os.makedirs(config.log_path,exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(os.path.join(config.log_path,"final.log"))
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.info('Training with the following settings:')
    for name, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            for name, value in vars(value).items():
                logger.info('{} : {}'.format(name, value))
        else:
            logger.info('{} : {}'.format(name, value))
    config.logger = logger
    task = KernelSIVI(config)
    task.learn()


