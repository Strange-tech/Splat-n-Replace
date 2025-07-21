import torch
from utils.graphics_utils import geom_transform_quat, geom_transform_points
import numpy as np
import os
import yaml
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, instanced_render
import sys
from scene import InstScene
from scene.gaussian_model import InstGaussianModel
from scene.vanilla_gaussian_model import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import BasicPointCloud
import cv2
from torchvision.utils import save_image

with open("./arguments/hyper_param.yaml", "r") as f:
        hyper_param = yaml.safe_load(f)

SCENE_NAME = hyper_param["SCENE_NAME"]

if __name__ == "__main__":

    inst_gaussians = InstGaussianModel(sh_degree=3)
    inst_gaussians.load_chkpnt(f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}/chkpnt10000.pth")

    bg_gaussians = GaussianModel(sh_degree=3)
    bg_gaussians.load_ply(f"/root/autodl-tmp/data/{SCENE_NAME}/seg_inst/bg.ply")

#     for k,v in inst_gaussians.features_rest_offsets.items():
#         for dense_tensor in v:
#                 total_elements = dense_tensor.numel()
#                 zero_count = (dense_tensor == 0).sum().item()
#                 print(zero_count, total_elements)

    parser = ArgumentParser(description="Rendering script for Splat-n-Replace")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    scene = InstScene(lp.extract(args), inst_gaussians)
    cameras = scene.getTrainCameras()

    inst_gaussians.instancing()

    save_path = f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}/rendered_images"
    os.makedirs(save_path, exist_ok=True)

    for idx, view in enumerate(cameras):
        # Render the scene
        rendered_image = instanced_render(view, inst_gaussians, bg_gaussians, pp.extract(args), background)["render"]
        # Save the rendered image
        save_image(rendered_image, f'{save_path}/{idx}.jpg')

        break

    # inst_gaussians.save_ply(f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}/instanced_gaussians.ply", instancing=True)

    # All done
    print("\nRender complete.")
