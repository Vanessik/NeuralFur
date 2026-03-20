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

import torch
import torch.nn.functional as F
from math import pi

import numpy as np



class SDFQuery:
    def __init__(self, sdf_path='',
                 bounds_min='',
                 bounds_max='',
                device='cuda'):
        """
        sdf_tensor : (1,1,D,H,W) tensor representing the SDF grid
        bounds_min : (3,) tensor for minimum XYZ bounds
        bounds_max : (3,) tensor for maximum XYZ bounds
        """
        
        self.device = device
        self.grid_size = 32
        self.sdf_tensor = torch.from_numpy(np.load(sdf_path)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        self.bounds_min = torch.tensor(np.load(bounds_min), dtype=torch.float32, device=self.device)
        self.bounds_max = torch.tensor(np.load(bounds_max), dtype=torch.float32, device=self.device)
        self.normals = self.compute_sdf_normals().permute(3, 0, 1, 2).unsqueeze(0) 
        self.feat_tenzor = torch.cat((self.sdf_tensor, self.normals), 1)

    def compute_sdf_normals(self):
        """
        sdf_grid: [D, H, W] torch tensor (3D SDF grid)
        voxel_size: float
        Returns: [D, H, W, 3] tensor of unit normals
        """
        sdf_grid = self.sdf_tensor.squeeze(0).squeeze(0)
        D, H, W = sdf_grid.shape

        # Compute finite differences in central region
        dx = (sdf_grid[2:, 1:-1, 1:-1] - sdf_grid[:-2, 1:-1, 1:-1]) / (2 * self.grid_size)
        dy = (sdf_grid[1:-1, 2:, 1:-1] - sdf_grid[1:-1, :-2, 1:-1]) / (2 * self.grid_size)
        dz = (sdf_grid[1:-1, 1:-1, 2:] - sdf_grid[1:-1, 1:-1, :-2]) / (2 * self.grid_size)

        normals = torch.stack([dx, dy, dz], dim=-1)  # [D-2, H-2, W-2, 3]
        normals = F.normalize(normals, dim=-1)

        # Pad manually: replicate border values
        def replicate_border(tensor):
            # tensor: [D, H, W, C]
            tensor = torch.cat([tensor[0:1], tensor, tensor[-1:]], dim=0)  # depth
            tensor = torch.cat([tensor[:, 0:1], tensor, tensor[:, -1:]], dim=1)  # height
            tensor = torch.cat([tensor[:, :, 0:1], tensor, tensor[:, :, -1:]], dim=2)  # width
            return tensor

        normals_padded = replicate_border(normals)
        return normals_padded  # [D, H, W, 3]



    def query(self, points):
        """
        points : (N,3) tensor of XYZ coordinates in world space

        Returns:
            sdf_vals : (N,) tensor of sampled SDF values
        """
        points = points.clamp(self.bounds_min, self.bounds_max)
        
        B, C, D, H, W = self.sdf_tensor.shape

        # Normalize points to [-1, 1]
        p_norm = 2.0 * (points - self.bounds_min[None]) / (self.bounds_max - self.bounds_min)[None] - 1.0
        grid = p_norm.view(1, -1, 1, 1, 3)
        grid = grid[..., [2, 1, 0]]  # Swap to (z, y, x) order expected by grid_sample

        # Trilinear sampling
        sampled = F.grid_sample(
            self.feat_tenzor,
            grid,
            mode='bilinear',
            align_corners=True
        )
        

        # Flatten result
        sdf_vals = sampled[:, :1].view(-1)
        sdf_normals = sampled[:, 1:].permute(0, 2, 1, 3, 4).squeeze(0).squeeze(-1).squeeze(-1)
        
        return sdf_vals, sdf_normals
    
    
    


def l1_loss(network_output, gt, weight = None, mask = None):
    loss = torch.abs(network_output - gt)
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss.mean()

def ce_loss(network_output, gt):
    return F.binary_cross_entropy(network_output.clamp(1e-3, 1.0 - 1e-3), gt)


def strand_curvature_signature(strands):
    """
    Returns a (N, M-2) tensor of bending angles per strand
    """
    v1 = strands[:, 1:-1, :] - strands[:, :-2, :]
    v2 = strands[:, 2:, :] - strands[:, 1:-1, :]
    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
    angles = torch.acos((v1 * v2).sum(dim=-1).clamp(-1.0, 1.0))
    return angles  # (N, M-2)


def shape_consistency_loss(strands):
    """
    Enforces all strands to have similar curvature profiles
    """
    descriptors = strand_curvature_signature(strands)  # (N, M-2)
    mean_descriptor = descriptors.mean(dim=0, keepdim=True)  # (1, M-2)
    loss = torch.mean((descriptors - mean_descriptor) ** 2)
    return loss


def or_loss(network_output, gt, confs = None, weight = None, mask = None):
    weight = torch.ones_like(gt[:1]) if weight is None else weight
    loss = torch.minimum(
        (network_output - gt).abs(),
        torch.minimum(
            (network_output - gt - 1).abs(), 
            (network_output - gt + 1).abs()
        ))
    loss = loss * pi
    if confs is not None:
        loss = loss * confs - (confs + 1e-7).log()    
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss * weight


