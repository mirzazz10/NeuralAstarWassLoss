"""
Testing the code on the test data
Author: Rahman Baig Mirza
Reference: https://github.com/omron-sinicx/neural-astar
"""

from __future__ import annotations

import os
from glob import glob
import numpy as np
import hydra
import pytorch_lightning as pl
import torch
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import set_global_seeds
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.training import load_from_ptl_checkpoint
from tqdm import tqdm
import re

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

@hydra.main(config_path="config", config_name="train")
def main(config):
    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_dataloader(
        config.dataset + ".npz", "train", config.params.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        config.dataset + ".npz", "valid", config.params.batch_size, shuffle=False
    )

    test_loader = create_dataloader(
        config.dataset + ".npz", "test", 1, shuffle=False
    ) 
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    neural_astar = NeuralAstar(encoder_arch='Unet').to(device)
    neural_astar.load_state_dict(load_from_ptl_checkpoint("C://Users//mirza//ASU//MS_master_folder//semester_3//perceptionRobotics//baseLines//neural-astar//model//mixed_064_moore_c16//lightning_logs"))    
    neural_astar.eval()
    vanilla_astar = VanillaAstar().to(device)
    vanilla_astar.eval()

    total_hmean = []
    total_p_opt = []
    total_p_exp = []

    for test_batch in tqdm(test_loader):
        
        map_designs, start_maps, goal_maps, _ = test_batch
        if map_designs.shape[1] == 1:
            na_outputs = neural_astar(map_designs.to(device), start_maps.to(device), goal_maps.to(device))
            va_outputs = vanilla_astar(map_designs.to(device), start_maps.to(device), goal_maps.to(device))

            pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            pathlen_model = na_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            p_opt = (pathlen_astar == pathlen_model).mean()
            exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            exp_na = na_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

            h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

            total_hmean.append( h_mean)
            total_p_opt.append( p_opt)
            total_p_exp.append( p_exp)


    mean_hmean = np.mean( total_hmean)
    mean_p_opt = np.mean( total_p_opt)
    mean_p_exp = np.mean( total_p_exp) 
    print( mean_hmean, mean_p_exp, mean_p_opt)


if __name__ == "__main__":
    
    main()
