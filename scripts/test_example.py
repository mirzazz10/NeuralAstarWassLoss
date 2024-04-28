"""
Running inference for a single example and generating gif file 
Author: Rahman Baig Mirza
Reference: https://github.com/omron-sinicx/neural-astar
"""
import torch
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.training import load_from_ptl_checkpoint
import matplotlib.pyplot as plt
from neural_astar.utils.data import create_dataloader
import re
import matplotlib.pyplot as plt
from neural_astar.utils.data import visualize_results
from glob import glob
import moviepy.editor as mpy


def load_from_ptl_checkpoint(checkpoint_path: str) -> dict:
    """
    Load model weights from PyTorch Lightning checkpoint.

    Args:
        checkpoint_path (str): (parent) directory where .ckpt is stored.

    Returns:
        dict: model state dict
    """    
    print(f"load {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)["state_dict"]
    state_dict_extracted = dict()
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted

class inferenceModel():
    def __init__(self, model_arch, checkpointPath1, checkpointPath2):
        self.checkpointPath1 = checkpointPath1
        self.checkpointPath2 = checkpointPath2
        self.neural_astar_Normal = NeuralAstar(encoder_arch= model_arch).to(device)
        self.neural_astar_Normal.load_state_dict(load_from_ptl_checkpoint(self.checkpointPath1))
        self.neural_astar_Was = NeuralAstar(encoder_arch= model_arch).to(device)
        self.neural_astar_Was.load_state_dict(load_from_ptl_checkpoint(self.checkpointPath2))
        self.vanilla_astar = VanillaAstar().to(device)    

    def createGIF( self, plannerType, map_designs, outputs):
        frames = [
        visualize_results(
            map_designs, intermediate_results, scale=4
        )
        for intermediate_results in outputs.intermediate_results
        ]
        # print( "outputs intermediate results", outputs.intermediate_results )
        clip = mpy.ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
        clip.write_gif(f"testRuns/video_{plannerType}.gif")

    def _do_inference( self, datasetPath, mode, batch_size):
        dataloader = create_dataloader(datasetPath, mode, batch_size)
        map_designs, start_maps, goal_maps, _ = next(iter(dataloader))
        fig, axes = plt.subplots(2, 3, figsize=[8, 5])
        axes[0, 0].imshow(map_designs.numpy()[0, 0])
        axes[0, 0].set_title("map_design")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(start_maps.numpy()[0, 0])
        axes[0, 1].set_title("start_map")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(goal_maps.numpy()[0, 0])
        axes[0, 2].set_title("goal_map")
        axes[0, 2].axis("off")
        plt.savefig('./testRuns/input.png')
        
        self.neural_astar_Normal.eval()
        self.vanilla_astar.eval()
        self.neural_astar_Was.eval()
        na_outputs_Normal = self.neural_astar_Normal(map_designs.to(device), start_maps.to(device), goal_maps.to(device), True)
        na_outputs_Wass = self.neural_astar_Was(map_designs.to(device), start_maps.to(device), goal_maps.to(device), True)
        va_outputs = self.vanilla_astar(map_designs.to(device), start_maps.to(device), goal_maps.to(device), True)

        fig, axes = plt.subplots(3, 1, figsize=[12, 4])        
        axes[0].imshow(visualize_results(map_designs, na_outputs_Normal))
        axes[0].set_title("Neural A*")
        axes[0].axis("off")
        axes[1].imshow(visualize_results(map_designs, na_outputs_Wass))
        axes[1].set_title("Wass A*")
        axes[1].axis("off")
        axes[2].imshow(visualize_results(map_designs, va_outputs))
        axes[2].set_title("Vanilla A*")
        axes[2].axis("off")
        plt.savefig('./testRuns/output.png')

        self.createGIF( "L1", map_designs, na_outputs_Normal )
        self.createGIF( "Wass", map_designs, na_outputs_Wass )
        self.createGIF( "Vanilla", map_designs, va_outputs )
        print( "Output saved")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_arch = "Unet"     
    ckptPath1 = r"\model\version_15128766\checkpoints\epoch=255-step=8192.ckpt"
    ckptPath2 = r"C:\Users\mirza\ASU\MS_master_folder\semester_3\perceptionRobotics\finalUnetRuns\finalOutput\WasserteinLoss\version_15100342\checkpoints\epoch=354-step=11360.ckpt"    
    model = inferenceModel( model_arch, ckptPath1, ckptPath2)
    batch_size = 1
    datasetPath = './planning-datasets/data/street/mixed_064_moore_c16.npz'
    mode = "test"    
    model._do_inference( datasetPath, mode, batch_size)