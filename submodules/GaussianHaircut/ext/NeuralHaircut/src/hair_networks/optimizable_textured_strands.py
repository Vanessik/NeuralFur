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

from .texture import UNet
from .strand_prior import Decoder

from torchvision.transforms import functional as TF
import sys

import accelerate
from copy import deepcopy
import os
import trimesh
import cv2
import pathlib

sys.path.append(os.path.join(sys.path[0], 'k-diffusion'))
from k_diffusion import config 

from src.utils.util import param_to_buffer, positional_encoding
from src.utils.geometry import barycentric_coordinates_of_projection, face_vertices
from src.utils.sample_points_from_meshes import sample_points_from_meshes
from src.diffusion_prior.diffusion import make_denoiser_wrapper



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
                 num_strands,
                 max_num_strands,
                 texture_size,
                 geometry_descriptor_size,
                 appearance_descriptor_size,
                 decoder_checkpoint_path,
                 path_to_scale=None,
                 cut_scalp=None, 
                 diffusion_cfg=None,
                 data_dir=None,
                 flame_mesh_dir=None,
                 num_guiding_strands=None,
#                  exp_name=None
                 ):
        super().__init__()
        file_path = pathlib.Path(__file__).parent.resolve()
        scalp_vert_idx = torch.load(f'{file_path}/../../data/new_scalp_vertex_idx.pth').long().cuda() # indices of scalp vertices
        scalp_faces = torch.load(f'{file_path}/../../data/new_scalp_faces.pth')[None].cuda() # faces that form a scalp
        scalp_uvs = torch.load(f'{file_path}/../../data/improved_neural_haircut_uvmap.pth').cuda()[None] # generated in Blender uv map for the scalp

        # Load FLAME head mesh
        if flame_mesh_dir is not None:
#             print('inside flame', flame_mesh_dir)
            verts, faces, _ = load_obj(f'{flame_mesh_dir}/stage_3/mesh_final.obj', device='cuda')
        else:
            verts, faces, _ = load_obj(path_to_mesh, device='cuda')

            
            
        
        # Transform head mesh if it's not in unit sphere (same scale used for world-->unit_sphere transform)
        self.transform = None
        
#         print('usssssse scale in NH', path_to_scale)
#         input()
        if path_to_scale:
            with open(path_to_scale, 'rb') as f:
                self.transform = pickle.load(f)
            verts = (verts - torch.tensor(self.transform['translation'], device=verts.device)) / self.transform['scale']
       
            
        head_mesh =  Meshes(verts=[(verts)], faces=[faces.verts_idx]).cuda()
        
        # Scaling factor, as decoder pretrained on synthetic data with fixed head scale
        usc_scale = torch.tensor([[0.2579, 0.4082, 0.2580]]).cuda()
        head_scale = head_mesh.verts_packed().max(0)[0] - head_mesh.verts_packed().min(0)[0]
        self.scale_decoder = (usc_scale / head_scale).mean()
        
        scalp_verts = head_mesh.verts_packed()[None, scalp_vert_idx]
        scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0]
        
        # Extract scalp mesh from head
        self.scalp_mesh = Meshes(verts=scalp_verts, faces=scalp_faces, textures=TexturesVertex(scalp_uvs)).cuda()
        
        # If we want to use different scalp vertices for scene
        cut_scalp=False
        if cut_scalp:
            if data_dir is not None:
                print(flame_mesh_dir)
                with open(f'{flame_mesh_dir}/cut_scalp_verts.pickle', 'rb') as f:
                    full_scalp_list = sorted(pickle.load(f))
            else:
                with open(cut_scalp, 'rb') as f:
                    full_scalp_list = sorted(pickle.load(f))
                
            a = np.array(full_scalp_list)
            b = np.arange(a.shape[0])
            d = dict(zip(a, b))
            
            faces_masked = []
            for face in self.scalp_mesh.faces_packed():
                if face[0] in full_scalp_list and face[1] in full_scalp_list and  face[2] in full_scalp_list:
                    faces_masked.append(torch.tensor([d[int(face[0])], d[int(face[1])], d[int(face[2])]]))

            scalp_uvs = scalp_uvs[:, full_scalp_list]
            self.scalp_mesh = Meshes(verts=self.scalp_mesh.verts_packed()[None, full_scalp_list].float(), faces=torch.stack(faces_masked)[None].cuda(), textures=TexturesVertex(scalp_uvs)).cuda()
        
        self.scalp_mesh.textures = TexturesVertex(scalp_uvs)

        # For 3D interpolation
        self.use_guiding_strands = num_guiding_strands is not None and num_guiding_strands > 0
        self.num_guiding_strands = num_guiding_strands if self.use_guiding_strands else 0

        self.num_strands = num_strands - self.num_guiding_strands
        self.max_num_strands = max_num_strands
        self.geometry_descriptor_size = geometry_descriptor_size
        self.appearance_descriptor_size = appearance_descriptor_size

        mgrid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, texture_size)]*2))[None].cuda()
        self.register_buffer('encoder_input', positional_encoding(mgrid, 6))
        
        # Initialize the texture decoder network
        self.texture_decoder = UNet(self.encoder_input.shape[1], geometry_descriptor_size + appearance_descriptor_size, bilinear=True)

        self.register_buffer('local2world', self.init_scalp_basis(scalp_uvs))

        # Sample fixed origin points
        origins, uvs, face_idx = sample_points_from_meshes(self.scalp_mesh, num_samples=max_num_strands, return_textures=True)
        self.register_buffer('origins', origins[0])
        self.register_buffer('uvs', uvs[0])

        # Get transforms for the samples
        self.local2world.data = self.local2world[face_idx[0]]
        
        # For uniform faces selection
        self.N_faces =  self.scalp_mesh.faces_packed()[None].shape[1]   
        # self.m, self.q = self.num_strands // self.N_faces, self.num_strands % self.N_faces
        
        # if self.use_guiding_strands:
        #     self.m_gdn, self.q_gdn = self.num_guiding_strands // self.N_faces, self.num_guiding_strands % self.N_faces
        
        self.faces_dict = {}
        for idx, f in enumerate(face_idx[0].cpu().numpy()):
            try:
                self.faces_dict[f].append(idx)
            except KeyError:
                self.faces_dict[f] = [idx]
        
        idxes, counts = face_idx[0].unique(return_counts=True)
        self.faces_count_dict = dict(zip(idxes.cpu().numpy(), counts.cpu().numpy()))
        
        # Decoder predicts the strands from the embeddings
        decoder_checkpoint_path = f'{file_path}/../../pretrained_models/strand_prior/strand_ckpt.pth'
        self.strand_decoder = Decoder(None, latent_dim=geometry_descriptor_size, length=99).eval()
        self.strand_decoder.load_state_dict(torch.load(decoder_checkpoint_path)['decoder'])
        param_to_buffer(self.strand_decoder)

        # Diffusion prior model
        self.use_diffusion = diffusion_cfg['use_diffusion']  

        if self.use_diffusion:
            ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=diffusion_cfg['model']['skip_stages'] > 0)
            self.accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=1)

            # Initialize diffusion model
            inner_model = config.make_model(diffusion_cfg)
            model = make_denoiser_wrapper(diffusion_cfg)(inner_model)
            self.model_ema = deepcopy(model).cuda()
            self.model_ema.eval()
            
            # Upload pretrained on synthetic data checkpoint
            diffusion_checkpoint_path = f'{file_path}/../../pretrained_models/diffusion_prior/wo_bug_blender_uv_00130000.pth'
            ckpt = torch.load(diffusion_checkpoint_path, map_location='cpu')
            self.accelerator.unwrap_model(self.model_ema.inner_model).load_state_dict(ckpt['model_ema'])
            param_to_buffer(self.model_ema)

            self.diffusion_input = diffusion_cfg['model']['input_size'][0]
            self.sample_density = config.make_sample_density(diffusion_cfg['model'])
            self.start_denoise = diffusion_cfg['start_denoise']
            self.diffuse_bs = diffusion_cfg['diffuse_bs']
            
            # Load scalp mask for hairstyle
            if data_dir is not None:
                self.diffuse_mask = f'{flame_mesh_dir}/scalp_data/dif_mask.png'
            else:
                self.diffuse_mask = diffusion_cfg.get('diffuse_mask', None) 
            
            if os.path.exists(self.diffuse_mask) and self.diffuse_mask:
                print(f'Loading diffuse mask {self.diffuse_mask}')
                self.diffuse_mask = torch.tensor(cv2.imread(self.diffuse_mask) / 255)[:, :, :1].squeeze(-1).cuda()
            else:
                self.diffuse_mask = torch.ones(256, 256).cuda()

            self.rect_size = texture_size
            self.downsample_size = self.diffusion_input

            b = torch.linspace(0, self.rect_size**2 - 1, self.rect_size**2, device="cuda").reshape(self.rect_size, self.rect_size)
            self.patch_size = self.rect_size // self.downsample_size
            unf = torch.nn.Unfold(
                kernel_size=self.patch_size,
                stride=self.patch_size).cuda()
            self.unfo = unf(b[None, None]).reshape(-1, self.downsample_size**2) # all unfolds

    def init_scalp_basis(self, scalp_uvs):         

        scalp_verts, scalp_faces = self.scalp_mesh.verts_packed()[None], self.scalp_mesh.faces_packed()[None]
        scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0] 
        
        # Define normal axis
        origin_v = scalp_face_verts.mean(1)
        origin_n = self.scalp_mesh.faces_normals_packed()
        origin_n /= origin_n.norm(dim=-1, keepdim=True)
        
        # Define tangent axis
        full_uvs = scalp_uvs[0][scalp_faces[0]]
        bs = full_uvs.shape[0]
        concat_full_uvs = torch.cat((full_uvs, torch.zeros(bs, full_uvs.shape[1], 1, device=full_uvs.device)), -1)
        new_point = concat_full_uvs.mean(1).clone()
        new_point[:, 0] += 0.001
        bary_coords = barycentric_coordinates_of_projection(new_point, concat_full_uvs).unsqueeze(1)
        full_verts = scalp_verts[0][scalp_faces[0]]
        origin_t = (bary_coords @ full_verts).squeeze(1) - full_verts.mean(1)
        origin_t /= origin_t.norm(dim=-1, keepdim=True)
        
        assert torch.where((bary_coords.reshape(-1, 3) > 0).sum(-1) != 3)[0].shape[0] == 0
        
        # Define bitangent axis
        origin_b = torch.cross(origin_n, origin_t, dim=-1)
        origin_b /= origin_b.norm(dim=-1, keepdim=True)

        # Construct transform from global to local (for each point)
        R = torch.stack([origin_t, origin_b, origin_n], dim=1) 
        
        # local to global 
        R_inv = torch.linalg.inv(R) 
        
        return R_inv
        
    def forward(self, it=None): 
        
        # Generate texture
        texture = self.texture_decoder(self.encoder_input)
        texture_res = texture.shape[-1]
        
        # Use diffusion prior
        diffusion_dict = {}
        
        if self.use_diffusion and it is not None and it >= self.start_denoise:
            geo_texture = texture[:, :self.geometry_descriptor_size]

            idx = torch.randint(low=0, high=self.patch_size**2, size=(self.diffuse_bs, 1), device="cuda")
            idx_i = idx.repeat(1, self.downsample_size**2).view(-1)
            idx_j = torch.arange(self.downsample_size**2, device="cuda").view(1, self.downsample_size**2).repeat(self.diffuse_bs, 1).view(-1)
            choosen_val = self.unfo[idx_i, idx_j]
            x = choosen_val // self.rect_size
            y = choosen_val % self.rect_size

            textures = geo_texture[:, :, x.long(), y.long()].split(self.downsample_size**2, dim=-1)
            diffusion_texture = torch.cat(textures).reshape(self.diffuse_bs, geo_texture.shape[1], self.diffusion_input, self.diffusion_input)

            noise = torch.randn_like(diffusion_texture)
            sigma = self.sample_density([diffusion_texture.shape[0]], device='cuda')
            mask = None
            if self.diffuse_mask is not None:
                mask = torch.nn.functional.interpolate(self.diffuse_mask[None][None], size=(self.diffusion_input, self.diffusion_input))
            L_diff, pred_image, noised_image = self.model_ema.loss_wo_logvar(diffusion_texture, noise, sigma, mask=mask, unet_cond=None)

            diffusion_dict['L_diff'] = L_diff.mean()       
        
        # m = self.m_gdn if self.use_guiding_strands else self.m
        # q = self.q_gdn if self.use_guiding_strands else self.q
        num_strands = self.num_guiding_strands if self.use_guiding_strands else self.num_strands

        # # Sample idxes from texture
        # if m:
        #     # If the #sampled strands > #scalp faces, then we try to sample more uniformly for better convergence
        #     f_idx, count = torch.cat((torch.arange(self.N_faces).repeat(m), torch.randperm(self.N_faces)[:q])).unique(return_counts=True)
            
        #     current_iter =  dict(zip(f_idx.cpu().numpy(), count.cpu().numpy()))
        #     iter_idx = []

        #     for i in range(self.N_faces):
        #         cur_idx_list = torch.tensor(self.faces_dict[i])[torch.randperm(self.faces_count_dict[i])[:current_iter[i]]].tolist()
        #         iter_idx.append(cur_idx_list)
        #     idx = torch.tensor(list(itertools.chain(*iter_idx)))
        # else:
        idx = torch.randperm(self.max_num_strands, device=texture.device)[:num_strands]

        origins = self.origins[idx]
        uvs = self.uvs[idx]
        local2world = self.local2world[idx]

        # Get latents for the samples
        z = F.grid_sample(texture, uvs[None, None])[0, :, 0].transpose(0, 1) # num_strands, C
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
            # # Sample the remaining indices from the texture            
            # if self.m:
            #     # If the #sampled strands > #scalp faces, then we try to sample more uniformly for better convergence
            #     f_idx, count = torch.cat((torch.arange(self.N_faces).repeat(self.m), torch.randperm(self.N_faces)[:self.q])).unique(return_counts=True)
                
            #     current_iter =  dict(zip(f_idx.cpu().numpy(), count.cpu().numpy()))
            #     iter_idx = []

            #     for i in range(self.N_faces):
            #         cur_idx_list = torch.tensor(self.faces_dict[i])[torch.randperm(self.faces_count_dict[i])[:current_iter[i]]].tolist()
            #         iter_idx.append(cur_idx_list)
            #     idx = torch.tensor(list(itertools.chain(*iter_idx)))
            # else:
            idx = torch.randperm(self.max_num_strands, device=texture.device)[:self.num_strands]
            
            origins_gdn = origins
            uvs_gdn = uvs
            local2world_gdn = local2world
            p_local_gdn = p_local
            
            origins_int = self.origins[idx]
            uvs_int = self.uvs[idx]
            local2world_int = self.local2world[idx]
            
            # Find K nearest neighbours for each of the interpolated points in the UV space
            K = 4

            dist = ((uvs_int.view(-1, 1, 2) - uvs_gdn.view(1, -1, 2))**2).sum(-1) # num_strands x num_guiding_strands
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

            uvs = torch.cat([uvs_gdn, uvs_int])
            local2world = torch.cat([local2world_gdn, local2world_int])
            p_local = torch.cat([p_local_gdn, p_local_int])
            
            origins = torch.cat([origins_gdn, origins_int])

            if self.appearance_descriptor_size:
                # Get latents for the samples
                z_int = F.grid_sample(texture, uvs_int[None, None])[0, :, 0].transpose(0, 1) # num_strands, C
                z_app_int = z_int[:, self.geometry_descriptor_size:]
                z_app = torch.cat([z_app, z_app_int])

        p = (local2world[:, None] @ p_local[..., None])[:, :, :3, 0] + origins[:, None] # [num_strands, strang_length, 3]

        return p, uvs, local2world, p_local, z_geom, z_app, diffusion_dict
    

    def forward_inference(self, num_strands): 
        
        # To sample more strands at inference stage
        texture = self.texture_decoder(self.encoder_input)
        self.num_strands = num_strands
        
        # Sample from the fixed origins
        torch.manual_seed(0)
        idx = torch.randperm(self.max_num_strands, device=texture.device)[:num_strands]
        origins = self.origins[idx]
        uvs = self.uvs[idx]
        local2world = self.local2world[idx]

        # Get latents for the samples
        z = F.grid_sample(texture, uvs[None, None])[0, :, 0].transpose(0, 1) # num_strands, C
        
        z_geom = z[:, :self.geometry_descriptor_size]

        if self.appearance_descriptor_size:
            z_app = z[:, self.geometry_descriptor_size:]
        else:
            z_app = None
        
        strands_list = []
        p_local_list = []
        for i in range(self.num_strands // 2500):
            l, r = i * 2500, (i+1) * 2500
            z_geom_batch = z_geom[l:r]
            v = self.strand_decoder(z_geom_batch) / self.scale_decoder # [num_strands, strand_length - 1, 3]
        
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )
            p = (local2world[l:r][:, None] @ p_local[..., None])[:, :, :3, 0] + origins[l:r][:, None] # [num_strands, strang_length, 3]
            strands_list.append(p)
            p_local_list.append(p_local)
        return torch.cat(strands_list, dim=0), uvs, local2world, torch.cat(p_local_list, dim=0), z_geom, z_app