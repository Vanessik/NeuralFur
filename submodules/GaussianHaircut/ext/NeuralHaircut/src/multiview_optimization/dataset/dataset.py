from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.io import imread
import os
import torch
import cv2 as cv
import sys
import json
from .openpose_data import OpenposeData
import pickle



def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    
    return intrinsics, pose


class Multiview_dataset(Dataset):
    def __init__(self, 
                 data_path, 
                 fitted_camera_path, 
                 use_scale, 
                 views_idx='',  
                 device='cuda',
                 batch_size=1):
        self.device = device

        self.scale_path = f'{data_path}/scale.pickle' if use_scale else ''
        self.image_path = f'{data_path}/images_4'
        self.openpose_kp_path = f'{data_path}/openpose/json'
        self.pixie_init_path = f'{data_path}/initialization_pixie'
        self.batch_size = batch_size
        
        imgs_list = [im_name.split('.')[0] for im_name in sorted(os.listdir(self.image_path))]
        
        self.fitted_camera_path = fitted_camera_path
        if fitted_camera_path:
            world_mats_dict = pickle.load(open(fitted_camera_path, 'rb'))
            imgs_list = [im_name for im_name in imgs_list if im_name in world_mats_dict.keys()]

 
        if views_idx:
            with open(views_idx, 'rb') as f:
                filter_idx = pickle.load(f) 
                imgs_list =  [imgs_list[i] for i in filter_idx]

        images_np = np.stack([imread(os.path.join(self.image_path, f'{im_name}.png')) for im_name in imgs_list])
        images = torch.from_numpy((images_np.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)).float()

        if fitted_camera_path:
            print('Overwriting COLMAP cameras with optimized cameras')
            world_mats_np = [world_mats_dict[im_name].numpy() for im_name in imgs_list]
            scale_mats_np = [np.eye(4) for _ in imgs_list]

        else:
            self.cams = np.load(f'{data_path}/cameras.npz', allow_pickle=True)
            world_mats_np = [self.cams[f'world_mat_{im_name}'] for im_name in imgs_list]

            scale_mat = np.eye(4, dtype=np.float32)        
            if self.scale_path:
                with open(self.scale_path, 'rb') as f:
                    transform = pickle.load(f)
                    print('upload transform', transform, self.scale_path)
                    scale_mat[:3, :3] *= transform['scale']
                    scale_mat[:3, 3] = np.array(transform['translation'])
                scale_mats_np = [scale_mat for _ in range(len(imgs_list) )]
            else:
                scale_mats_np = [self.cams[f'scale_mat_{im_name}'] for im_name in imgs_list]
        
        intrinsics_all = []
        pose_all = []
        proj_all = []
        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            proj_all.append(torch.from_numpy(P.copy()).float())
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())

        intrinsics_all = torch.stack(intrinsics_all).to(device) # [n_images, 4, 4]
        pose_all = torch.stack(pose_all).to(device) # [n_images, 4, 4]
        proj_all = torch.stack(proj_all).to(device)

        if views_idx:
            with open(views_idx, 'rb') as f:
                filter_idx = pickle.load(f) 
                
                pose_all =  pose_all[filter_idx]
                intrinsics_all = intrinsics_all[filter_idx]
        else:
            filter_idx = np.arange(len(imgs_list))

        intrinsics_all = intrinsics_all[:, :3, :3]
        
        lmks_dict = pickle.load(open(f'{data_path}/face_alignment/lmks_2d.pkl', 'rb'))
        lmks3d_dict = pickle.load(open(f'{data_path}/face_alignment/lmks_3d.pkl', 'rb'))
        
        lmks = [lmks_dict[im_name] if im_name in lmks_dict.keys() else None for im_name in imgs_list]
        lmks3d = [lmks3d_dict[im_name] if im_name in lmks3d_dict.keys() else None for im_name in imgs_list]

        # took views that have openpose keypoints
        data_openpose_dict = {}
        sns = sorted(os.listdir(self.openpose_kp_path))
        for sn in sns:
            with open(os.path.join(self.openpose_kp_path, sn), 'r') as f:
                data_openpose_dict[sn.replace('_keypoints.json', '')] = json.load(f)
        data_openpose = [data_openpose_dict[im_name] for im_name in imgs_list]

        self.good_views  = []
        self.view_scores = []
        
        mapping = dict(zip(filter_idx, np.arange(len(imgs_list))))
        unmapping = dict(zip(np.arange(len(imgs_list)), filter_idx))
        print(unmapping)
        
        
        for i in range(len(data_openpose)):
            if i in filter_idx:
                if len(data_openpose[i]['people']) > 0:
                    score = np.asarray(data_openpose[i]['people'][0]['face_keypoints_2d']).reshape(-1, 3)[:, 2].mean()
                    if score > 0.6:
                        self.good_views.append(mapping[i])
                        self.view_scores.append(score)       

        self.num_views = len(self.good_views)

        to_keep = [lmks[i] is not None for i in self.good_views]
        self.good_views = [self.good_views[i] for i in range(self.num_views) if to_keep[i]] # For some views otained landmarks could be bad
        self.view_scores = [self.view_scores[i] for i in range(self.num_views) if to_keep[i]] # For some views otained landmarks could be bad

        self.nimages = min(len(self.good_views), self.batch_size)
#         self.good_views = np.array(self.good_views)[:self.nimages]
        
        i_sorted = np.argsort(self.view_scores)[::-1]
        self.good_views = [self.good_views[i] for i in i_sorted]

        self.good_views = np.sort(np.array(self.good_views)[:self.nimages])
        
        self.openpose_data = OpenposeData(path=self.openpose_kp_path, views=self.good_views, device=self.device, filter_views_mapping=unmapping)    

        self.lmks = torch.from_numpy(np.stack([lmks[i] for i in self.good_views]))
        self.lmks3d = torch.from_numpy(np.stack([lmks3d[i] for i in self.good_views]))
        self.images = images[self.good_views]
        self.proj_all = proj_all[self.good_views]
        self.poses = torch.inverse(pose_all[self.good_views])
        self.intrinsics_all = intrinsics_all[self.good_views]

        print('Fitting using the following views:', [imgs_list[i] for i in self.good_views])
    
    def get_filter_views(self):
        return torch.tensor(self.good_views).to(self.device)
        
    def __getitem__(self, index):
        print(index)
        return {
                'img': self.images[index].to(self.device), 
                'lmks': self.lmks[index].to(self.device),
                'lmks3d':self.lmks3d[index].to(self.device), 
                'projection': self.proj_all[index].to(self.device),
                'extrinsics_rvec': self.poses[index, :3, :3].to(self.device),
                'extrinsics_tvec': self.poses[index, :3, 3].to(self.device),
                'frame_ids': torch.tensor(index, dtype=torch.long).to(self.device),
                'intrinsics': self.intrinsics_all[index].to(self.device),
                'openpose_lmks': self.openpose_data.get_sample(index)
               } 
    
    def __len__(self):
        return self.nimages 