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
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model_animal import GaussianModel
from scene.gaussian_model_latent_strands import GaussianModelHair
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos
import numpy as np
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, pointcloud_path=None, shuffle=True, resolution_scales=[1.0], scene_suffix="", scale_factor=1, resolution=(1024, 1024), use_test_split=False, num_views=-1):
        """
        :param path: Path to colmap scene main folder.
        """
        
        args.eval=use_test_split
        
        
        print('use eval', args.eval, load_iteration)
        
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, 2, args.interpolate_cameras, args.speed_up, args.max_frames, args.frame_offset)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "projection.npy")) or os.path.exists(os.path.join(args.source_path, "cameras.npz")):
            print('load synth camera')
            scene_info = sceneLoadTypeCallbacks["Synthetic"](args.source_path, args.images, args.eval, scale_factor=scale_factor, resolution=resolution, num_views=num_views)
        
        else:
            assert False, "Could not recognize scene type!"

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_suffix)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_suffix)

        if self.loaded_iter:
            if len(pointcloud_path) > 0:
                print(f'Loading point cloud from {pointcloud_path}')
                self.gaussians.load_ply(pointcloud_path, self.cameras_extent)
            else:
                try:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                     f"point_cloud{scene_suffix}",
                                                     "iteration_" + str(self.loaded_iter),
                                                     "raw_point_cloud.ply"), self.cameras_extent)
                except Exception as e:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                     f"point_cloud",
                                                     "iteration_" + str(self.loaded_iter),
                                                     "raw_point_cloud.ply"), self.cameras_extent)
                
        else:        
            if pointcloud_path is not None:
                
                self.gaussians.load_ply_mesh(pointcloud_path, self.cameras_extent)
            else:
                xyz = np.random.random((point_cloud.shape[0], 3)) * 2.6 - 1.3
                shs = np.random.random((point_cloud.shape[0], 3)) / 255.0

                pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((point_cloud.shape[0], 3)))

            
            
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]