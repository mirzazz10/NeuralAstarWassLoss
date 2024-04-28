"""Helper functions for training
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import random
import re
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
from neural_astar.planner.astar import VanillaAstar

import torch

def wasLoss(p, t1, t2, t1_weights=None, t2_weights=None):
    total_loss = 0.0
    batch_size = t1.shape[0]
    
    for i in range(batch_size):
        t1_values = torch.squeeze(t1[i, :, :])
        t2_values = torch.squeeze(t2[i, :, :])
        t1_values = t1_values.reshape( -1)
        t2_values = t2_values.reshape( -1)  
                
        t1_sorter = torch.argsort(t1_values)
        t2_sorter = torch.argsort(t2_values)

        all_values = torch.cat((t1_values, t2_values))
        all_values, _ = torch.sort(all_values)

        # Compute the differences between pairs of successive values of t1 and t2.
        deltas = torch.diff(all_values)

        # Get the respective positions of the values of t1 and t2 among the values of
        # both distributions.
        t1_cdf_indices = torch.searchsorted(t1_values[t1_sorter], all_values[:-1], right=True)
        t2_cdf_indices = torch.searchsorted(t2_values[t2_sorter], all_values[:-1], right=True)

        # Calculate the CDFs of t1 and t2 using their weights, if specified.
        if t1_weights is None:
            t1_cdf = t1_cdf_indices.float() / t1_values.size(0)
        else:
            t1_sorted_cumweights = torch.cat((torch.tensor([0.], dtype=t1_values.dtype, device=t1_values.device),
                                              t1_weights[t1_sorter].cumsum(dim=0)))
            t1_cdf = t1_sorted_cumweights[t1_cdf_indices] / t1_sorted_cumweights[-1]

        if t2_weights is None:
            t2_cdf = t2_cdf_indices.float() / t2_values.size(0)
        else:
            t2_sorted_cumweights = torch.cat((torch.tensor([0.], dtype=t2_values.dtype, device=t2_values.device),
                                              t2_weights[t2_sorter].cumsum(dim=0)))
            t2_cdf = t2_sorted_cumweights[t2_cdf_indices] / t2_sorted_cumweights[-1]

        # Compute the value of the integral based on the CDFs.
        if p == 1:
            loss = torch.sum(torch.abs(t1_cdf - t2_cdf) * deltas)
        elif p == 2:
            loss = torch.sqrt(torch.sum(torch.square(t1_cdf - t2_cdf) * deltas))
        else:
            loss = torch.pow(torch.sum(torch.pow(torch.abs(t1_cdf - t2_cdf), p) * deltas), 1/p)
        
        total_loss += loss
    
    # Calculate mean loss across batches
    mean_loss = total_loss / batch_size
    
    return mean_loss

# Example usage:
# loss = _cdf_distance(p, t1, t2, t1_weights=None, t2_weights=None)


def _cdf_distance(p, t1_values, t2_values, t1_weights=None, t2_weights=None):
    # t1_values, t1_weights = _validate_distribution(t1, t1_weights)
    # t2_values, t2_weights = _validate_distribution(t2, t2_weights)

    t1_sorter = torch.argsort(t1_values)
    t2_sorter = torch.argsort(t2_values)

    all_values = torch.cat((t1_values, t2_values))
    all_values, _ = torch.sort(all_values)

    # Compute the differences between pairs of successive values of t1 and t2.
    deltas = torch.diff(all_values)

    # Get the respective positions of the values of t1 and t2 among the values of
    # both distributions.
    t1_cdf_indices = torch.searchsorted(t1_values[t1_sorter], all_values[:-1], right=True)
    t2_cdf_indices = torch.searchsorted(t2_values[t2_sorter], all_values[:-1], right=True)

    # Calculate the CDFs of t1 and t2 using their weights, if specified.
    
    t1_cdf = t1_cdf_indices.float() / t1_values.size(0)
    t2_cdf = t2_cdf_indices.float() / t2_values.size(0)
    
    if p == 2:
        return torch.sqrt(torch.sum(torch.square(t1_cdf - t2_cdf) * deltas))
    return torch.pow(torch.sum(torch.pow(torch.abs(t1_cdf - t2_cdf), p) * deltas), 1/p)


class wasserteinLoss(nn.Module):
    def __init__(self, p=2):
        super(wasserteinLoss, self).__init__()
        self.p = p

    def forward(self, t1, t2, t1_weights=None, t2_weights=None):
        # Call the _cdf_distance function passing the tensors and other parameters
        # loss = _cdf_distance(self.p, t1, t2, t1_weights, t2_weights)
        loss = wasLoss(self.p, t1, t2, t1_weights, t2_weights)
        print( "Wassertein Loss", loss)
        return loss


def load_from_ptl_checkpoint(checkpoint_path: str) -> dict:
    """
    Load model weights from PyTorch Lightning checkpoint.

    Args:
        checkpoint_path (str): (parent) directory where .ckpt is stored.

    Returns:
        dict: model state dict
    """

    ckpt_file = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))[-1]
    print(f"load {ckpt_file}")
    state_dict = torch.load(ckpt_file)["state_dict"]
    state_dict_extracted = dict()
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted


class PlannerModule(pl.LightningModule):
    def __init__(self, planner, config):
        super().__init__()
        self.planner = planner
        self.vanilla_astar = VanillaAstar()
        self.config = config

    def forward(self, map_designs, start_maps, goal_maps):
        return self.planner(map_designs, start_maps, goal_maps)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(self.planner.parameters(), self.config.params.lr)

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)        
        # loss = nn.L1Loss()(outputs.histories, opt_trajs)
        loss = wasserteinLoss(2)(outputs.histories, opt_trajs)        
        self.log("metrics/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = wasserteinLoss(2)(outputs.histories, opt_trajs)
        # loss = nn.L1Loss()(outputs.histories, opt_trajs)
        self.log("metrics/val_loss", loss)

        # For shortest path problems:
        if map_designs.shape[1] == 1:
            va_outputs = self.vanilla_astar(map_designs, start_maps, goal_maps)
            pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            p_opt = (pathlen_astar == pathlen_model).mean()

            exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

            h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

            self.log("metrics/p_opt", p_opt)
            self.log("metrics/p_exp", p_exp)
            self.log("metrics/h_mean", h_mean)

        return loss


def set_global_seeds(seed: int) -> None:
    """
    Set random seeds

    Args:
        seed (int): random seed
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
