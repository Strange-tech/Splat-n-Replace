import torch
from utils.graphics_utils import geom_transform_quat, geom_transform_points
import numpy as np
import os
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, instanced_render
import sys
from instanced_scene import InstScene, InstGaussianModel
from scene import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import BasicPointCloud
import gc
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, bg_gs, inst_gs, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    scene = InstScene(dataset, inst_gs)
    inst_gs.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        inst_gs.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        inst_gs.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            inst_gs.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # start_time = time.time()
        render_pkg = instanced_render(viewpoint_cam, bg_gs, inst_gs, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # print("render time", time.time() - start_time)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # gt_mask = gt_image.sum(dim = 0)[None,:,:]
        # gt_mask[gt_mask != 0] = 1

        # mask_loss = - ((gt_mask * mask).sum() + 0.15 * ((1-gt_mask) * mask).sum())

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss = mask_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                inst_gs.max_radii2D[visibility_filter] = torch.max(inst_gs.max_radii2D[visibility_filter], radii[visibility_filter])
                inst_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    inst_gs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    inst_gs.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                inst_gs.optimizer.step()
                inst_gs.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((inst_gs.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":

    template_names = ["armchair", "ottoman", "sofa"]

    shared_gaussians = InstGaussianModel(sh_degree=3)
    
    bg_gaussians = GaussianModel(sh_degree=3)
    bg_gaussians.load_ply("./segmentation_res/background/bg.ply")

    shared_xyz = None
    shared_color = None
    template_intervals = {}
    transforms = {}
    features_dc_offsets = {}
    features_rest_offsets = {}

    for template_name in template_names:

        instance_paths = [f'./segmentation_res/{template_name}/1/', f'./segmentation_res/{template_name}/2/']

        template_gs = GaussianModel(sh_degree=3)
        # 1. 加载所有实例，转换到模板空间
        all_instances = []
        for i, path in enumerate(instance_paths):
            gs = GaussianModel(sh_degree=3)
            gs.load_ply(f'{path}{template_name}.ply')
            # 注意：这里转置了一下，是为了方便与xyz做矩阵乘法
            t = torch.from_numpy(np.load(f'{path}transform.npy').T).float().to("cuda")
            # print(transform)
            gs._xyz = geom_transform_points(gs.get_xyz, t)
            gs._rotation = geom_transform_quat(gs.get_rotation, t.T)
            # gs.save_ply(f"./tmp/test/{i}.ply")
            all_instances.append(gs)
            if template_name not in transforms:
                transforms[template_name] = []
            transforms[template_name].append(torch.inverse(t))

        # 2. 合并所有模型为 shared_model（几何 & shared SH）
        template_gs.merge(all_instances)

        # 3. 为每个实例初始化 SH offset（同 shape）
        features_dc_offsets[template_name] = [torch.zeros_like(template_gs.get_features_dc, requires_grad=True) for _ in all_instances]
        features_rest_offsets[template_name] = [torch.zeros_like(template_gs.get_features_rest, requires_grad=True) for _ in all_instances]

        if shared_xyz is None:
            start_idx = 0
            shared_xyz = template_gs.get_xyz.detach().cpu().numpy()
        else:
            start_idx = shared_xyz.shape[0]
            shared_xyz = np.concatenate([shared_xyz, template_gs.get_xyz.detach().cpu().numpy()], axis=0)
        end_idx = shared_xyz.shape[0]

        if shared_color is None:
            shared_color = template_gs.get_features_dc.detach().cpu().numpy()
        else:
            shared_color = np.concatenate([shared_color, template_gs.get_features_dc.detach().cpu().numpy()], axis=0)
        template_intervals[template_name] = [start_idx, end_idx]
        
    shared_color = np.squeeze(shared_color)
    
    shared_pc = BasicPointCloud(shared_xyz, shared_color, np.zeros((shared_xyz.shape[0], 3)))
    shared_gaussians.create_from_pcd(shared_pc, spatial_lr_scale=0.1)
    shared_gaussians.set_template_intervals(template_intervals)
    shared_gaussians.set_transforms(transforms)
    shared_gaussians.set_features_dc_offsets(features_dc_offsets)
    shared_gaussians.set_features_rest_offsets(features_rest_offsets)

    # print(vars(shared_gaussians))
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), bg_gaussians, shared_gaussians, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
