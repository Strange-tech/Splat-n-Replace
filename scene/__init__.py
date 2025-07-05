#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_ff import FeatureGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    inst_gaussians : GaussianModel
    feature_inst_gaussians : FeatureGaussianModel

    # target: feature, seg, scene
    def __init__(self, args : ModelParams, inst_gaussians : GaussianModel=None, feature_inst_gaussians: FeatureGaussianModel=None, load_iteration=None, feature_load_iteration=None, shuffle=True, resolution_scales=[1.0], init_from_3dgs_pcd=False, target='scene', mode='train', sample_rate = 1.0):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.feature_loaded_iter = None
        self.inst_gaussians = inst_gaussians
        self.feature_inst_gaussians = feature_inst_gaussians

        if load_iteration:
            if load_iteration == -1:
                if mode == 'train':
                    # only load feature inst_gaussians when doing segmentation
                    if target == 'seg' or target == 'coarse_seg_everything':
                        self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="feature") if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    elif target == 'scene':
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    elif target == 'feature' or target == 'contrastive_feature':
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    else:
                        assert False and "Unknown target!"
                elif mode == 'eval':
                    if target == 'seg':
                        self.feature_loaded_iter = None
                        self.feature_inst_gaussians = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="seg")
                    elif target == 'scene':
                        self.feature_inst_gaussians = None
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    elif target in ['feature', 'contrastive_feature']:
                        self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target=target) if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                        self.loaded_iter = -1 if inst_gaussians is None else searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target='scene')
                    elif target == 'coarse_seg_everything':
                        self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target=target)
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    else:
                        assert False and "Unknown target!"
            else:
                self.loaded_iter = load_iteration
                if mode == 'train':
                    if target == 'seg' or target == 'coarse_seg_everything':
                        self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="feature") if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                    elif target == 'scene' or 'feature' in target:
                        self.feature_loaded_iter = None
                    else:
                        assert False and "Unknown target!"
                elif mode == 'eval':
                    if target == 'seg':
                        self.feature_loaded_iter = None
                        self.feature_inst_gaussians = None
                    elif target == 'scene':
                        self.feature_inst_gaussians = None
                        self.feature_loaded_iter = None
                    elif target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature':
                        self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target=target) if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                        self.loaded_iter = -1
                    else:
                        assert False and "Unknown target!"

            print("Loading trained model at iteration {}, {}".format(self.loaded_iter, self.feature_loaded_iter))
            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
        # used for testing lerf transforms,json
        # and not os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print(f"Allow Camera Principle Point Shift: {args.allow_principle_point_shift}")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, need_features = args.need_features, need_masks = args.need_masks, sample_rate = sample_rate, allow_principle_point_shift = args.allow_principle_point_shift, replica = 'replica' in args.model_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
        #     print("Found transforms.json file, assuming Lerf data set!")
        #     scene_info = sceneLoadTypeCallbacks["Lerf"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # Load or initialize scene / seg inst_gaussians
        if self.loaded_iter and self.inst_gaussians is not None:
            if mode == 'train':
                self.inst_gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "scene_point_cloud.ply"))
            else:
                if target == 'coarse_seg_everything':
                    self.inst_gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "scene_point_cloud.ply"))
                elif 'feature' not in target:
                    self.inst_gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            target+"_point_cloud.ply"))
                else:
                    self.inst_gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "scene_point_cloud.ply"))

        elif self.inst_gaussians is not None:
            self.inst_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # Load or initialize feature inst_gaussians
        if self.feature_loaded_iter and self.feature_inst_gaussians is not None:
            if target == 'feature' or target == 'seg':
                self.feature_inst_gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.feature_loaded_iter),
                                                        "feature_point_cloud.ply"))
            elif target == 'coarse_seg_everything':
                if mode == 'train':
                    self.feature_inst_gaussians.load_ply_from_3dgs(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "scene_point_cloud.ply"))
                elif mode == 'eval':
                    self.feature_inst_gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.feature_loaded_iter),
                                                        "coarse_seg_everything_point_cloud.ply"))
            elif target == 'contrastive_feature':
                if mode == 'train':
                    self.feature_inst_gaussians.load_ply_from_3dgs(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "scene_point_cloud.ply"))
                elif mode == 'eval':
                    self.feature_inst_gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.feature_loaded_iter),
                                                        "contrastive_feature_point_cloud.ply"))


        elif self.feature_inst_gaussians is not None:
            if target=='feature' and init_from_3dgs_pcd:
                print("Initialize feature inst_gaussians from 3DGS point cloud")
                self.feature_inst_gaussians.create_from_pcd(fetchPly(
                        os.path.join(
                            self.model_path, 
                            "point_cloud",
                            "iteration_" + str(searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene") if (self.loaded_iter is None or self.loaded_iter == -1) else self.loaded_iter), 
                            "scene_point_cloud.ply"
                        ), 
                        only_xyz=True
                    ), 
                self.cameras_extent
                )
            elif target == 'contrastive_feature':
                if mode == 'train':
                    self.feature_inst_gaussians.load_ply_from_3dgs(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "scene_point_cloud.ply"))
                elif mode == 'eval':
                    self.feature_inst_gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.feature_loaded_iter),
                                                        "contrastive_feature_point_cloud.ply"))
            else:
                print("Initialize feature inst_gaussians from Colmap point cloud")
                self.feature_inst_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration, target='scene'):
        assert target != 'feature' and "Please use save_feature() to save feature inst_gaussians!"
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.inst_gaussians.save_ply(os.path.join(point_cloud_path, target+"_point_cloud.ply"))

    def save_mask(self, iteration, id = 0):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.inst_gaussians.save_mask(os.path.join(point_cloud_path, f"seg_point_cloud_{id}.npy"))

    def save_feature(self, iteration, target = 'coarse_seg_everything', smooth_weights = None, smooth_type = None, smooth_K = 16):
        assert self.feature_inst_gaussians is not None and (target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature')
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.feature_inst_gaussians.save_ply(os.path.join(point_cloud_path, f"{target}_point_cloud.ply"), smooth_weights, smooth_type, smooth_K)


    # def save_coarse_seg_everything(self, iteration):
    #     assert self.feature_inst_gaussians is not None
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.feature_inst_gaussians.save_ply(os.path.join(point_cloud_path, "coarse_seg_everything_point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


# import os
# import random
# import json
# from utils.system_utils import searchForMaxIteration
# from scene.dataset_readers import sceneLoadTypeCallbacks
# # from scene.gaussian_model import GaussianModel
# from scene.gaussian_model_inst import InstGaussianModel
# from arguments import ModelParams
# from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# class Scene:

#     inst_gaussians : InstGaussianModel

#     def __init__(self, args : ModelParams, inst_gaussians : InstGaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
#         """b
#         :param path: Path to colmap scene main folder.
#         """
#         self.model_path = args.model_path
#         self.loaded_iter = None
#         self.inst_gaussians = inst_gaussians

#         if load_iteration:
#             if load_iteration == -1:
#                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
#             else:
#                 self.loaded_iter = load_iteration
#             print("Loading trained model at iteration {}".format(self.loaded_iter))

#         self.train_cameras = {}
#         self.test_cameras = {}

#         if os.path.exists(os.path.join(args.source_path, "sparse")):
#             scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
#         elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
#             print("Found transforms_train.json file, assuming Blender data set!")
#             scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
#         else:
#             assert False, "Could not recognize scene type!"

#         if not self.loaded_iter:
#             with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
#                 dest_file.write(src_file.read())
#             json_cams = []
#             camlist = []
#             if scene_info.test_cameras:
#                 camlist.extend(scene_info.test_cameras)
#             if scene_info.train_cameras:
#                 camlist.extend(scene_info.train_cameras)
#             for id, cam in enumerate(camlist):
#                 json_cams.append(camera_to_JSON(id, cam))
#             with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
#                 json.dump(json_cams, file)

#         if shuffle:
#             random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
#             random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

#         self.cameras_extent = scene_info.nerf_normalization["radius"]

#         for resolution_scale in resolution_scales:
#             print("Loading Training Cameras")
#             self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
#             print("Loading Test Cameras")
#             self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

#         if self.loaded_iter:
#             self.inst_gaussians.load_ply(os.path.join(self.model_path,
#                                                            "point_cloud",
#                                                            "iteration_" + str(self.loaded_iter),
#                                                            "point_cloud.ply"), args.train_test_exp)
#         else:
#             self.inst_gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

#     def save(self, iteration):
#         point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
#         self.inst_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
#         exposure_dict = {
#             image_name: self.inst_gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
#             for image_name in self.inst_gaussians.exposure_mapping
#         }

#         with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
#             json.dump(exposure_dict, f, indent=2)

#     def getTrainCameras(self, scale=1.0):
#         return self.train_cameras[scale]

#     def getTestCameras(self, scale=1.0):
#         return self.test_cameras[scale]