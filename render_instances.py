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
from copy import deepcopy
from torchvision.utils import save_image
from scene.cameras import Camera

with open("./arguments/hyper_param.yaml", "r") as f:
    hyper_param = yaml.safe_load(f)

SCENE_NAME = hyper_param["SCENE_NAME"]


if __name__ == "__main__":

    model_dict = torch.load(
        f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}/chkpnt30000.pth"
    )
    all_template_gs = []
    for k, model in model_dict.items():
        template_gs = InstGaussianModel(sh_degree=3)
        template_gs.restore(model_args=model, training_args=None)
        print(template_gs.get_xyz.shape, len(template_gs._features_dc_offsets))
        for offset in template_gs._features_rest_offsets:
            total_elements = offset.numel()
            zero_count = (offset == 0).sum().item()
            print(zero_count, total_elements)
        all_template_gs.append(template_gs)

    bg_gaussians = GaussianModel(sh_degree=3)
    bg_gaussians.load_ply(f"/root/autodl-tmp/data/{SCENE_NAME}/seg_inst/bg.ply")

    parser = ArgumentParser(description="Rendering script for Splat-n-Replace")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.source_path = f"/root/autodl-tmp/data/{SCENE_NAME}"
    args.model_path = f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}"

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    scene = InstScene(lp.extract(args), all_template_gs)
    cameras = scene.getTrainCameras()

    for temp_gs in all_template_gs:
        temp_gs.instancing()
        temp_gs.save_ply(
            f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}/inst_gs_{temp_gs.template_id}.ply",
            instancing=True,
        )

    save_path = f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}/rendered_images"
    os.makedirs(save_path, exist_ok=True)

    view = deepcopy(cameras[0])

    ### test_3
    view.R = np.array([[-0.66, -0.03, -0.74], [-0.72, 0.24, 0.63], [0.16, 0.96, -0.19]])
    view.T = np.array([4.37, -0.95, 2.04])

    ### test_2
    # view.R = np.array([[-0.55, -0.34, 0.75], [0.41, 0.66, 0.61], [-0.71, 0.65, -0.22]])
    # view.T = np.array([-3.94, -3.43, 1.67])

    ### test
    # view.R = np.array([[0.23, -0.23, 0.94], [0.70, 0.71, 0.00], [-0.67, 0.66, 0.32]])
    # view.T = np.array([-5.05, -2.45, 6.23])
    view.update()

    # rendered_image = instanced_render(
    #     view, all_template_gs, bg_gaussians, pp.extract(args), background
    # )["render"]
    # save_image(rendered_image, f"{save_path}/test_3.jpg")

    # All done
    print("\nRender complete.")
