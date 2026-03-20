import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from torch import nn
from torch.nn import functional as F

import itertools
import pickle

import numpy as np
import torchvision

from .texture import UNet, positional_encoding_mlp, MLP
from .strand_prior import Decoder

from torchvision.transforms import functional as TF
import sys

import accelerate
from copy import deepcopy
import os
import trimesh
import cv2
import pathlib
from pytorch3d.ops import knn_points

sys.path.append(os.path.join(sys.path[0], 'k-diffusion'))
from k_diffusion import config 


from src.utils.util import param_to_buffer, positional_encoding
from src.utils.geometry import barycentric_coordinates_of_projection, face_vertices
from src.utils.sample_points_from_meshes import sample_points_from_meshes
from src.diffusion_prior.diffusion import make_denoiser_wrapper
import json


def safe_normalize(v, dim=-1, eps=1e-8):
    return v / (v.norm(dim=dim, keepdim=True) + eps)



def assign_groups_by_nearest_vertex(origins, vertex_positions, vertex_groups):
    """
    Assigns group to each origin based on its nearest vertex using PyTorch3D knn_points.
    
    Args:
        origins: [N, 3] tensor
        vertex_positions: [V, 3] tensor
        vertex_groups: [V] tensor
    
    Returns:
        strand_groups: [N] tensor with group IDs for each origin
    """
    # Reshape for PyTorch3D batch API
    origins_batched = origins.unsqueeze(0)          # [1, N, 3]
    vertex_positions_batched = vertex_positions.unsqueeze(0)  # [1, V, 3]

    # Perform k-NN with k=1 (nearest vertex)
    knn = knn_points(origins_batched, vertex_positions_batched, K=1)

    nearest_idx = knn.idx[0, :, 0]  # [N] - nearest vertex indices
    strand_groups = vertex_groups[nearest_idx]  # assign group from nearest vertex

    return strand_groups



def smooth_values_torch(origins: torch.Tensor, values: torch.Tensor, k: int = 5):
    """
    Smooth scalar values based on k-nearest neighbors using PyTorch3D's knn_points.
    """
    # Reshape for pytorch3d: [1, N, 3]
    origins_batched = origins.unsqueeze(0)  # [1, N, 3]
    
    # Perform k-NN search (query against itself)
    knn = knn_points(origins_batched, origins_batched, K=k, return_sorted=True)
    idx = knn.idx[0]  # [N, k] - indices of neighbors

    # Gather values of k-nearest neighbors
    neighbor_values = values[idx]  # [N, k]

    # Smooth by averaging neighbor values
    smoothed_values = neighbor_values.mean(dim=1)  # [N]

    return smoothed_values



def get_face_adjacency(tri_mesh):
    """
    Given a trimesh.Trimesh object, return a dictionary mapping each face index
    to a list of neighboring face indices.
    """
    adjacency = tri_mesh.face_adjacency
    num_faces = len(tri_mesh.faces)

    # Build face-to-neighbor list
    neighbors = [[] for _ in range(num_faces)]
    for f0, f1 in adjacency:
        neighbors[f0].append(f1)
        neighbors[f1].append(f0)
    return neighbors

def fix_invalid_basis_with_trimesh_neighbors(R, valid, tri_mesh):
    neighbors = get_face_adjacency(tri_mesh)
    R_fixed = R.clone()
    for i, is_valid in enumerate(valid):
        if not is_valid:
            for j in neighbors[i]:
                if valid[j]:
                    R_fixed[i] = R[j]
                    break
            else:
                # fallback: identity
                R_fixed[i] = torch.eye(3, device=R.device)
    return R_fixed

def fix_normals_with_neighbors(mesh):
    """
    Fix NaN or zero-length normals in a mesh by averaging neighboring face normals.
    """
    faces = mesh.faces_packed()  # (F, 3)
    verts = mesh.verts_packed()  # (V, 3)

    # Compute raw face normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    raw_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)

    # Detect degenerate normals (zero-length or NaN)
    degenerate_mask = (raw_normals.norm(dim=-1) < 1e-6) | torch.isnan(raw_normals).any(dim=-1)
    valid_mask = ~degenerate_mask

    # Normalize valid normals
    face_normals = torch.zeros_like(raw_normals)
    face_normals[valid_mask] = safe_normalize(raw_normals[valid_mask])

    if degenerate_mask.any():
        print(f"Fixing {degenerate_mask.sum().item()} degenerate face normals...")

        # Step 1: Aggregate valid face normals to vertex normals
        num_faces = faces.shape[0]
        num_verts = verts.shape[0]

        # Create a zeroed vertex normal buffer
        vert_normals = torch.zeros((num_verts, 3), device=verts.device)

        # Accumulate valid face normals to their vertices
        for i in range(3):
            vert_normals = vert_normals.index_add(0, faces[valid_mask, i], face_normals[valid_mask])

        vert_normals = safe_normalize(vert_normals)

        # Step 2: For degenerate faces, average the normals at their 3 vertices
        deg_faces = faces[degenerate_mask]  # (D, 3)
        repaired = vert_normals[deg_faces].mean(dim=1)  # (D, 3)
        repaired = safe_normalize(repaired)

        face_normals[degenerate_mask] = repaired

    return face_normals  # (F, 3)




def downsample_texture(rect_size, downsample_size):
    b = torch.linspace(0, rect_size**2 - 1, rect_size**2, device="cuda").reshape(rect_size, rect_size)
    
    patch_size = rect_size // downsample_size
    unf = torch.nn.Unfold(
        kernel_size=patch_size,
        stride=patch_size).cuda()
    unfo = unf(b[None, None]).reshape(-1, downsample_size**2)
    idx = torch.randint(low=0, high=patch_size**2, size=(1,), device="cuda")
    idx_ = idx.repeat(downsample_size**2,)
    choosen_val = unfo[idx_, torch.arange(downsample_size**2, device="cuda")]
    x = choosen_val // rect_size
    y = choosen_val % rect_size 
    return x.long(), y.long()


class OptimizableTexturedStrands(nn.Module):
    def __init__(self, 
                 path_to_mesh, 
                 path_to_scale,
                 path_to_uvmap,
                 num_strands,
                 max_num_strands,
                 texture_size,
                 geometry_descriptor_size,
                 appearance_descriptor_size,
                 decoder_checkpoint_path,
                 num_guiding_strands=0,
                 fix_normals=False,
                 path_to_classes='',
                 normalize_decoder_output=False,
                 mapping_length=None,
                 texture_type='unet',
                 num_freqs=6,
                 precomputed_tangent='',
                 mapping_gravity=None,
                 use_metrical_space=False,
                 eye_dists_VQA=-1,
                 eye_dists_geom='',
                 metric_length=2.5,
                 smooth_annots=False,
                 fix_length=False, 
                 smooth_annots_k=5
                 ):
        
        super().__init__()
        file_path = pathlib.Path(__file__).parent.resolve()

        self.fix_length = fix_length
        self.use_metrical_space = use_metrical_space
        
        self.metric_scale = torch.tensor([1], device='cuda').float()
        if self.use_metrical_space:
            eyes_geom = np.array(trimesh.load(eye_dists_geom).vertices)
            eyes_geom_dists = np.linalg.norm(eyes_geom[0] - eyes_geom[1])
            
            self.metric_scale = torch.tensor([eyes_geom_dists / eye_dists_VQA], device='cuda').float()
            
            
        
        self.fix_normals = fix_normals
        self.normalize_decoder_output = normalize_decoder_output
        self.precomputed_tangent = precomputed_tangent

        self.use_guiding_strands = num_guiding_strands is not None and num_guiding_strands > 0
        self.num_guiding_strands = num_guiding_strands if self.use_guiding_strands else 0

        self.num_strands = num_strands - self.num_guiding_strands

        if self.normalize_decoder_output and self.use_metrical_space is False:
            self.length_param = nn.Parameter(torch.tensor([0.09]), requires_grad=True)
        
        else:
            self.length_param = torch.tensor([metric_length], device='cuda').float()
            
        
        self.scale_decoder = 1
        
        verts, faces, _ = load_obj(path_to_mesh, device='cuda')
        if path_to_scale:
            with open(path_to_scale, 'rb') as f:
                self.transform = pickle.load(f)
            verts = ((verts - torch.tensor(self.transform['translation'], device=verts.device)) / self.transform['scale']).float()
        
        if len(self.precomputed_tangent) > 0:
            self.scalp_mesh = Meshes(verts=verts[None], faces=faces.verts_idx[None]).cuda()
            try:
                scalp_uvs = torch.load(path_to_uvmap).cuda()[None].float() # generated in Blender uv map for the scalp
            except Exception as e:
                scalp_uvs = None
        else:
            try:
                scalp_uvs = torch.load(path_to_uvmap).cuda()[None].float()
                self.scalp_mesh = Meshes(verts=verts[None], faces=faces.verts_idx[None], textures=TexturesVertex(scalp_uvs)).cuda()
                self.scalp_mesh.textures = TexturesVertex(scalp_uvs)
            except Exception as e:
                self.scalp_mesh = Meshes(verts=verts[None], faces=faces.verts_idx[None]).cuda()
                scalp_uvs = None

        self.max_num_strands = max_num_strands
        self.register_buffer('local2world', self.init_scalp_basis(scalp_uvs))

        self.strand_length_scale = None
        self.strand_gravity = None

        self.path_to_classes = path_to_classes
        self.verts = verts

        self.mapping_length = mapping_length
        self.mapping_gravity = mapping_gravity

        self.smooth_annots = smooth_annots
        self.smooth_annots_k = smooth_annots_k

        self.face_index_map = None
        if len(self.path_to_classes) > 0:
            self.init_length_scale()
           
            nonzero_length_verts = []

            for part, weight in self.mapping_length.items():
                if weight > 0:
                    nonzero_length_verts += self.vertex_group_dict[part]

            self.create_nonzero_length_mesh(nonzero_length_verts, scalp_uvs)

        self.geometry_descriptor_size = geometry_descriptor_size
        self.appearance_descriptor_size = appearance_descriptor_size

        # Sample fixed origin points
        if len(self.precomputed_tangent) > 0:
            origins, face_idx = sample_points_from_meshes(self.scalp_mesh, num_samples=max_num_strands, return_textures=False)

            try: 
                _, uvs, _ = sample_points_from_meshes(self.scalp_mesh, num_samples=max_num_strands, return_textures=True)
            except Exception as e:
                print(e)
                uvs = None
                       
        else:
            origins, uvs, face_idx = sample_points_from_meshes(self.scalp_mesh, num_samples=max_num_strands, return_textures=True)
        
        self.register_buffer('origins', origins[0])
       
    
 
        if self.face_index_map is not None:
            face_idx = self.face_index_map[face_idx]

        if len(self.path_to_classes) > 0:
            self.init_strand_groups()
        
        
        if uvs is not None:
            self.register_buffer('uvs', uvs[0])
        
        # Get transforms for the samples
        self.local2world.data = self.local2world[face_idx[0]]
        self.num_freqs =num_freqs
        
        self.texture_type = texture_type

        if self.texture_type == 'unet':
            mgrid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, texture_size)]*2))[None].cuda()
            self.register_buffer('encoder_input', positional_encoding(mgrid, 6))
            # Initialize the texture decoder network
            self.texture_decoder = UNet(self.encoder_input.shape[1], geometry_descriptor_size + appearance_descriptor_size, bilinear=True)
        
        elif self.texture_type == 'mlp':
            center = (self.origins.max(0)[0] + self.origins.min(0)[0]) / 2
            scale = (self.origins.max(0)[0] - self.origins.min(0)[0]).max()
            roots_norm = (self.origins - center) / (scale / 2 + 1e-8)

            self.pos_enc = positional_encoding_mlp(roots_norm, num_freqs=num_freqs)  # (10000, 2*20 + 2 = 22)
            
            self.texture_decoder = MLP(in_dim=self.pos_enc.shape[-1], out_dim=geometry_descriptor_size + appearance_descriptor_size)
        else:
            print('texture type')
        
        # Decoder predicts the strands from the embeddings
        decoder_checkpoint_path = f'{file_path}/../../pretrained_models/strand_prior/strand_ckpt.pth'
        self.strand_decoder = Decoder(None, latent_dim=geometry_descriptor_size, length=99).eval()
        self.strand_decoder.load_state_dict(torch.load(decoder_checkpoint_path)['decoder'])
        param_to_buffer(self.strand_decoder)

        
    
    
    def create_nonzero_length_mesh(self, nonzero_length_verts, scalp_uvs):
    
            full_scalp_list = sorted(nonzero_length_verts)
            a = np.array(full_scalp_list)
            b = np.arange(len(a))
            vertex_map = dict(zip(a, b))  # maps original vertex idx -> new vertex idx

            faces_masked = []
            face_orig_index_map = []  # stores original face index for each new face

            for i, face in enumerate(self.scalp_mesh.faces_packed()):
                v0, v1, v2 = face.tolist()
                if v0 in vertex_map and v1 in vertex_map and v2 in vertex_map:
                    new_face = torch.tensor([
                        vertex_map[v0],
                        vertex_map[v1],
                        vertex_map[v2]
                    ])
                    faces_masked.append(new_face)
                    face_orig_index_map.append(i)  # map to original face index

            # Final subsampled mesh
            new_verts = self.scalp_mesh.verts_packed()[full_scalp_list]
            new_faces = torch.stack(faces_masked)

            if  scalp_uvs is not None:
                scalp_uvs = scalp_uvs[:, full_scalp_list]
                
                self.scalp_mesh = Meshes(verts=new_verts[None].float().cuda(),faces=new_faces[None].cuda(), textures=TexturesVertex(scalp_uvs)).cuda()
                self.scalp_mesh.textures = TexturesVertex(scalp_uvs)

            else:
        
                self.scalp_mesh = Meshes(
                    verts=new_verts[None].float().cuda(),
                    faces=new_faces[None].cuda()
                )
            # Save the face index mapping
            self.face_index_map = torch.tensor(face_orig_index_map, device='cuda')
            
            
            
            
    
    def init_length_scale(self):
        with open(self.path_to_classes, 'r') as f:
            self.vertex_group_dict = json.load(f)

        group_names = list(self.vertex_group_dict.keys())
        group_to_id = {name: i for i, name in enumerate(group_names)}  # e.g., 'tail': 0, ...

        self.group_to_id = group_to_id
        V = self.verts.shape[0]
        self.vertex_groups = torch.full((V,), fill_value=-1, dtype=torch.long, device=self.verts.device)  # initialize with -1

        for group_name, vertex_ids in self.vertex_group_dict.items():
            group_id = group_to_id[group_name]
            self.vertex_groups[vertex_ids] = group_id

        assert (self.vertex_groups >= 0).all(), "Some vertices were not assigned a group"
        
        
        
        
        
    def  init_strand_groups(self):
        group_names = list(self.vertex_group_dict.keys())

        self.strand_groups = assign_groups_by_nearest_vertex(self.origins, self.verts, self.vertex_groups)

        selected_mask = ((self.strand_groups == self.group_to_id['ears']) | 
                 (self.strand_groups == self.group_to_id['face']) | 
                 (self.strand_groups == self.group_to_id['leg_rear']) | 
                 (self.strand_groups == self.group_to_id['leg_front']) | 
                 (self.strand_groups == self.group_to_id['paw_pads']) | 
                 (self.strand_groups == self.group_to_id['paws']) | 
                 (self.strand_groups == self.group_to_id['inner_earcanal'])).float()
        
        self.good_idx_face = torch.where(selected_mask > 0)[0]

        # Build tensor of length values
        length_per_group = torch.tensor(
            [self.mapping_length[name] for name in group_names],
            device=self.strand_groups.device, dtype=torch.float32
        )  # shape (5,)
        
        if len(self.mapping_gravity) > 0:
            gravity_per_group = torch.tensor(
            [self.mapping_gravity[name] for name in group_names],
            device=self.strand_groups.device, dtype=torch.float32
        )  # shape (5,)
            
            self.strand_gravity = gravity_per_group[self.strand_groups]
        
        # Assign length per strand
        self.strand_length_scale = length_per_group[self.strand_groups]  # shape (N,) 
        
        
        if self.smooth_annots:
            self.strand_length_scale = smooth_values_torch(self.origins, self.strand_length_scale, k=self.smooth_annots_k)

        
        
    def init_pos_enc(self):
            center = (self.origins.max(0)[0] + self.origins.min(0)[0]) / 2
            scale = (self.origins.max(0)[0] - self.origins.min(0)[0]).max()
            roots_norm = (self.origins - center) / (scale / 2 + 1e-8)

            self.pos_enc = positional_encoding_mlp(roots_norm, num_freqs=self.num_freqs)  # (10000, 2*20 + 2 = 22)
            
        
    def basis_deformation(self):   
        # recompute basis
        # recompute transform
        pass
    
    
    def mesh_update(self, new_mesh):
#         update mesh geometry and origins based on barycentric
        # recompute origins
        pass

    def init_scalp_basis(self, scalp_uvs):         

        scalp_verts, scalp_faces = self.scalp_mesh.verts_packed()[None], self.scalp_mesh.faces_packed()[None]
        scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0] 
        
        # Define normal axis
        origin_v = scalp_face_verts.mean(1)

        if self.fix_normals:
            origin_n = fix_normals_with_neighbors(self.scalp_mesh)
        else:
            origin_n = self.scalp_mesh.faces_normals_packed()
            
        origin_n = safe_normalize(origin_n)
        
        
        if len(self.precomputed_tangent) > 0:
            origin_t = torch.tensor(np.load(self.precomputed_tangent), device='cuda').float()
        else:
            # Define tangent axis
            full_uvs = scalp_uvs[0][scalp_faces[0]]
            bs = full_uvs.shape[0]
            concat_full_uvs = torch.cat((full_uvs, torch.zeros(bs, full_uvs.shape[1], 1, device=full_uvs.device)), -1)
            new_point = concat_full_uvs.mean(1).clone()
            new_point[:, 0] += 0.00001
            bary_coords = barycentric_coordinates_of_projection(new_point, concat_full_uvs).unsqueeze(1)
            full_verts = scalp_verts[0][scalp_faces[0]]
            origin_t = (bary_coords @ full_verts).squeeze(1) - full_verts.mean(1)
        
        origin_t = safe_normalize(origin_t)

        # Define bitangent axis
        origin_b = torch.cross(origin_n, origin_t, dim=-1)
        origin_b = safe_normalize(origin_b)

        # Construct transform from global to local (for each point)
        R = torch.stack([origin_t, origin_b, origin_n], dim=1)

        det_R = torch.linalg.det(R)  # (F,)
        valid = det_R.abs() > 1e-6



        # Convert your PyTorch3D mesh to Trimesh (if needed)
        scalp_verts_np = self.scalp_mesh.verts_packed().cpu().numpy()
        scalp_faces_np = self.scalp_mesh.faces_packed().cpu().numpy()
        trimesh_mesh = trimesh.Trimesh(vertices=scalp_verts_np, faces=scalp_faces_np, process=False)

        # Fix R using neighbor info
        R_fixed = fix_invalid_basis_with_trimesh_neighbors(R, valid, trimesh_mesh)

        det_R_fixed = torch.linalg.det(R_fixed)
        valid_R_fixed = det_R_fixed.abs() > 1e-6

        # local to global 
        R_inv = torch.linalg.inv(R_fixed)

        return R_inv
        

    def forward(self, it=None, new_mesh=None): 
        
        if new_mesh is not None:
            self.mesh_update(new_mesh)
            self.basis_deformation()
            
        
        diffusion_dict = {}
        num_strands = self.num_guiding_strands if self.use_guiding_strands else self.num_strands

        idx = torch.randperm(self.max_num_strands, device=self.verts.device)[:num_strands]
        origins = self.origins[idx]

        try:
            uvs = self.uvs[idx]
        except Exception as e:
            uvs = None
            
        local2world = self.local2world[idx]
        
        scale_length = self.strand_length_scale[idx] if self.strand_length_scale is not None else None
        strand_gravity  = self.strand_gravity[idx] if self.strand_gravity is not None else None

        
        # Generate texture
        if self.texture_type == 'unet':
            texture = self.texture_decoder(self.encoder_input)
            z = F.grid_sample(texture, uvs[None, None])[0, :, 0].transpose(0, 1)

        elif self.texture_type == 'mlp':
            z = self.texture_decoder(self.pos_enc[idx])

        else:
            print('Error with texture')

        z_geom = z[:, :self.geometry_descriptor_size]
        
        if self.appearance_descriptor_size:
            z_app = z[:, self.geometry_descriptor_size:]
        else:
            z_app = None

        # Decode strabds
        v = self.strand_decoder(z_geom) / self.scale_decoder  # [num_strands, strand_length - 1, 3]

        p_local = torch.cat([
                torch.zeros_like(v[:, -1:, :]), 
                torch.cumsum(v, dim=1)
            ], 
            dim=1
        )
        
        
        if self.use_guiding_strands:
            idx = torch.randperm(self.max_num_strands, device=self.verts.device)[:self.num_strands]
            origins_gdn = origins
            uvs_gdn = uvs
            local2world_gdn = local2world
            p_local_gdn = p_local
            scale_length_gdn = scale_length
            strand_gravity_gdn = strand_gravity
            
            origins_int = self.origins[idx]

            scale_length_int = self.strand_length_scale[idx] if self.strand_length_scale is not None else None
            strand_gravity_int  = self.strand_gravity[idx] if self.strand_gravity is not None else None
            
            try:
                uvs_int = self.uvs[idx]
            except Exception as e:
                uvs_int = None

                
            local2world_int = self.local2world[idx]
            
            # Find K nearest neighbours for each of the interpolated points in the UV space
            K = 4
            dist = ((origins_int.view(-1, 1, 3) - origins_gdn.view(1, -1, 3)) ** 2).sum(-1)
            knn_dist, knn_idx = torch.sort(dist, dim=1)
            w = 1 / (knn_dist[:, :K] + 1e-7)
            w = w / w.sum(dim=-1, keepdim=True)
            
            p_local_int_nearest = p_local[knn_idx[:, 0]]            
            p_local_int_bilinear = (p_local[knn_idx[:, :K]] * w[:, :, None, None]).sum(dim=1)
            
            # Calculate cosine similarity between neighbouring guiding strands to get blending alphas (eq. 4 of HAAR)
            knn_v = v[knn_idx[:, :K]]
            csim_full = torch.nn.functional.cosine_similarity(knn_v.view(-1, K, 1, 99, 3), knn_v.view(-1, 1, K, 99, 3), dim=-1).mean(-1) # num_guiding_strands x K x K
            j, k = torch.triu_indices(K, K, device=csim_full.device).split([1, 1], dim=0)
            i = torch.arange(self.num_guiding_strands, device=csim_full.device).repeat_interleave(j.shape[1])
            j = j[0].repeat(self.num_guiding_strands)
            k = k[0].repeat(self.num_guiding_strands)
            csim = csim_full[i, j, k].view(self.num_guiding_strands, -1).mean(-1)
            
            alpha = torch.where(csim <= 0.9, 1 - 1.63 * csim**5, 0.4 - 0.4 * csim)
            alpha_int = (alpha[knn_idx[:, :K]] * w).sum(dim=1)[:, None, None]
            p_local_int = p_local_int_nearest * alpha_int + p_local_int_bilinear * (1 - alpha_int)

            uvs = torch.cat([uvs_gdn, uvs_int]) if uvs_int is not None else None
            
            local2world = torch.cat([local2world_gdn, local2world_int])
            p_local = torch.cat([p_local_gdn, p_local_int])
            
            origins = torch.cat([origins_gdn, origins_int])
            scale_length = torch.cat([scale_length_gdn, scale_length_int]) if self.strand_length_scale is not None else None
            strand_gravity  = torch.cat([strand_gravity_gdn, strand_gravity_int]) if self.strand_gravity is not None else None

            if self.appearance_descriptor_size:
                # Get latents for the samples

                if self.texture_type == 'unet':
                    # Get latents for the samples
                    z_int = F.grid_sample(texture, uvs_int[None, None])[0, :, 0].transpose(0, 1) # num_strands, C

                elif self.texture_type == 'mlp':
                    z_int = self.texture_decoder(self.pos_enc[idx])
                else:
                    print('Error with texture')

                z_app_int = z_int[:, self.geometry_descriptor_size:]
                z_app = torch.cat([z_app, z_app_int])
            
                
                
                
        strand_world = (local2world[:, None] @ p_local[..., None])[:, :, :3, 0] #10000x100x3
        
        if self.normalize_decoder_output:

            diff = strand_world[:, 1:, :] - strand_world[:, :-1, :]  # (N, 99, 3)
            arc_length = diff.norm(dim=2).sum(dim=1, keepdim=True).unsqueeze(-1)  # (N, 1, 1)
            
            normalized_arc = strand_world / (arc_length + 1e-8)

            
            if scale_length is not None and self.fix_length is False:
                scaled_strands = normalized_arc * self.length_param  * scale_length[..., None, None] * self.metric_scale[..., None, None]
            
            else:
                scaled_strands = normalized_arc * self.length_param * self.metric_scale[..., None, None]
                
            
            p =  scaled_strands + origins[:, None] # [num_strands, strang_length, 3]
            

        else:
            p =  strand_world + origins[:, None] # [num_strands, strang_length, 3]

        return p, uvs, local2world, p_local, z_geom, z_app, diffusion_dict, strand_gravity
    

    def forward_inference(self, num_strands):
        # To sample more strands at inference stage
        self.num_strands = num_strands
        
        # Sample from the fixed origins
        torch.manual_seed(0)
        idx = torch.randperm(self.max_num_strands, device=self.verts.device)[:num_strands]
        origins = self.origins[idx]
        
        try:
            uvs = self.uvs[idx]
        except Exception as e:
            uvs = None
            
        local2world = self.local2world[idx]
        scale_length = self.strand_length_scale[idx] if self.strand_length_scale is not None else None
        strand_gravity  = self.strand_gravity[idx] if self.strand_gravity is not None else None
        
        if self.texture_type == 'unet':
            texture = self.texture_decoder(self.encoder_input)
            # Get latents for the samples
            z = F.grid_sample(texture, uvs[None, None])[0, :, 0].transpose(0, 1) # num_strands, C
            
        elif self.texture_type == 'mlp':
            z = self.texture_decoder(self.pos_enc[idx])
        else:
            print('Error with texture')
            
        z_geom = z[:, :self.geometry_descriptor_size]

        if self.appearance_descriptor_size:
            z_app = z[:, self.geometry_descriptor_size:]
        else:
            z_app = None
        
        strands_list = []
        p_local_list = []
        for i in range(self.num_strands // 10000):
            l, r = i * 10000, (i+1) * 10000
            z_geom_batch = z_geom[l:r]
            v = self.strand_decoder(z_geom_batch) / self.scale_decoder # [num_strands, strand_length - 1, 3]
        
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )
            strand_world = (local2world[l:r][:, None] @ p_local[..., None])[:, :, :3, 0]
            
            
            if self.normalize_decoder_output:
                diff = strand_world[:, 1:, :] - strand_world[:, :-1, :]  # (N, 99, 3)
                arc_length = diff.norm(dim=2).sum(dim=1, keepdim=True).unsqueeze(-1)  # (N, 1, 1)
                normalized_arc = strand_world / (arc_length + 1e-8)
                
                            
                if scale_length is not None and self.fix_length is False:
                    scaled_strands = normalized_arc * self.length_param  * scale_length[l:r][..., None, None] * self.metric_scale[..., None, None]

                else:
                    scaled_strands = normalized_arc * self.length_param * self.metric_scale[..., None, None]
                
                
                p =  scaled_strands + origins[l:r][:, None] # [num_strands, strang_length, 3]

            else:
                p =  strand_world + origins[l:r][:, None] # [num_strands, strang_length, 3]

            strands_list.append(p)
            p_local_list.append(p_local)
        return torch.cat(strands_list, dim=0), uvs, local2world, torch.cat(p_local_list, dim=0), z_geom, z_app, strand_gravity