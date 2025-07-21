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
import gc
import json

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch.nn.functional as F

with open("./arguments/hyper_param.yaml", "r") as f:
    hyper_param = yaml.safe_load(f)

SCENE_NAME = hyper_param["SCENE_NAME"]
empty_gaussian_threshold = hyper_param["gaussian"]["empty_gaussian_threshold"]
mask_iterations_ratio = hyper_param["gaussian"]["mask_iterations_ratio"]


def masked_interval(mask: torch.Tensor, s: int, e: int):
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask[s:e] = mask[s:e]
    return new_mask


def save_image_pair_cv2(pred_img, gt_img, step, save_dir="./tmp"):
    import os

    os.makedirs(save_dir, exist_ok=True)

    pred_np = (
        pred_img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255
    ).astype("uint8")
    gt_np = (gt_img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
        "uint8"
    )

    # OpenCV 是 BGR 格式，所以要转换
    cv2.imwrite(
        f"{save_dir}/pred_{step:05d}.png", cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(f"{save_dir}/gt_{step:05d}.png", cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))


# training的逻辑是，前一半迭代，用掩码盖住背景，专门训练实例；后一半迭代，加入背景，训练整体视觉效果
def training(
    dataset,
    opt,
    pipe,
    all_temp_gs,
    bg_gs,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    first_iter = 0
    scene = InstScene(dataset, all_temp_gs)

    # construct the mask of every temp_gaussian
    for temp_gs in all_temp_gs:
        temp_gs.training_setup(opt)

    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     inst_gs.restore(model_params, opt)

    if bg_gs is not None:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        mask_note = {}
        start_idx = 0
        for temp_gs in all_temp_gs:
            temp_gs.instancing()
            mask_note[temp_gs.template_id] = [
                start_idx,
                start_idx + temp_gs.get_full_xyz.shape[0],
            ]
            start_idx += temp_gs.get_full_xyz.shape[0]

        if bg_gs is None:
            # bg_color = np.random.rand(3)
            # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        iter_start.record()

        for temp_gs in all_temp_gs:
            temp_gs.update_learning_rate(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                temp_gs.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # start_time = time.time()
        render_pkg = instanced_render(
            viewpoint_cam, all_temp_gs, bg_gs, pipe, background
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        # image = render_pkg["render"]
        # print("render time", time.time() - start_time)

        # remove the unseen image
        if visibility_filter.sum().item() < empty_gaussian_threshold:
            # print(f"Iteration {iteration}: No visible points, skipping this iteration.")
            continue

        gt_image = viewpoint_cam.original_image.cuda()

        if iteration < int(mask_iterations_ratio * opt.iterations):
            mask = torch.load(
                f"{dataset.source_path}/instance_masks/{viewpoint_cam.image_name}.pt"
            )
            mask3 = mask.unsqueeze(0).expand_as(gt_image)  # 变成 3 通道掩码
            gt_image = gt_image * mask3
            image = image * mask3

        if iteration % 1000 == 0:
            save_image_pair_cv2(
                image,
                gt_image,
                iteration,
                save_dir=f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}/training_images",
            )

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            mse = F.mse_loss(image, gt_image)
            psnr = -10 * torch.log10(mse + 1e-8)  # 避免 log(0)
            progress_bar.set_postfix(
                {"Loss": f"{ema_loss_for_log:.{7}f}", "PSNR": f"{psnr.item():.{2}f}"}
            )
            progress_bar.update()
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                for temp_gs in all_temp_gs:
                    m_n = mask_note[temp_gs.template_id]
                    v_f_1 = masked_interval(visibility_filter, m_n[0], m_n[1])
                    v_f_2 = visibility_filter[m_n[0] : m_n[1]]

                    # Keep track of max radii in image-space for pruning
                    viewspace_point_tensor_grad = viewspace_point_tensor.grad

                    temp_gs.max_radii2D[v_f_2] = torch.max(
                        temp_gs.max_radii2D[v_f_2], radii[v_f_1]
                    )
                    temp_gs.add_densification_stats(
                        viewspace_point_tensor_grad[m_n[0] : m_n[1]], v_f_2
                    )

                    if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        size_threshold = (
                            20 if iteration > opt.opacity_reset_interval else None
                        )
                        temp_gs.densify_and_prune(
                            opt.densify_grad_threshold,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                        )

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        temp_gs.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                for temp_gs in all_temp_gs:
                    temp_gs.optimizer.step()
                    temp_gs.optimizer.zero_grad(set_to_none=True)

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((inst_gs.capture(), iteration), f"/root/autodl-tmp/3dgs_output/{SCENE_NAME}/chkpnt{iteration}.pth")


if __name__ == "__main__":

    with open(f"/root/autodl-tmp/data/{SCENE_NAME}/scene_graph.json", "r") as f:
        scene_graph = json.load(f)

    all_template_gs = []

    for temp_map in scene_graph:
        template_id = temp_map["template_id"]
        print(f"Processing template: {template_id}")

        template_gs = InstGaussianModel(sh_degree=3)
        # 1. 加载所有实例，转换到模板空间
        all_instances = []
        transforms = []
        for inst in temp_map["instances"]:
            inst_path = (
                f'/root/autodl-tmp/data/{SCENE_NAME}/seg_inst/{inst["instance_id"]}.ply'
            )
            gs = GaussianModel(sh_degree=3)
            gs.load_ply(inst_path)
            # 注意：这里转置了一下，是为了方便与xyz做矩阵乘法
            t = torch.from_numpy(np.array(inst["transform"]).T).float().to("cuda")
            gs._xyz = geom_transform_points(gs.get_xyz, t)
            gs._rotation = geom_transform_quat(gs.get_rotation, t.T)
            all_instances.append(gs)
            transforms.append(torch.inverse(t))

        # 2. 合并所有模型为 shared_model（几何 & shared SH）
        template_gs.merge(all_instances)

        # 3. 为每个实例初始化 SH offset（同 shape）
        features_dc_offsets = [
            torch.zeros_like(template_gs.get_features_dc, requires_grad=True)
            for _ in all_instances
        ]
        features_rest_offsets = [
            torch.zeros_like(template_gs.get_features_rest, requires_grad=True)
            for _ in all_instances
        ]

        template_gs.set_template_id(template_id)
        template_gs.set_transforms(transforms)
        template_gs.set_features_dc_offsets(features_dc_offsets)
        template_gs.set_features_rest_offsets(features_rest_offsets)

        all_template_gs.append(template_gs)

    # bg_gaussians = None
    bg_gaussians = GaussianModel(sh_degree=3)
    bg_gaussians.load_ply(f"/root/autodl-tmp/data/{SCENE_NAME}/seg_inst/bg.ply")
    bg_gaussians.frozen()

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=np.random.randint(10000, 20000))
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[5_000, 10_000]
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        all_template_gs,
        bg_gaussians,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
