import os
import sys
from pyhocon import ConfigFactory
from pathlib import Path

import torch

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.cameras import  FoVPerspectiveCameras
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.io import load_ply, save_ply, load_obj, save_obj, load_objs_as_meshes

from src.models.dataset import Dataset, MonocularDataset
from src.utils.geometry import face_vertices

import argparse
import yaml

from tqdm import tqdm
import numpy as np
import cv2
import pickle
from skimage.draw import polygon
from PIL import Image



def create_scalp_mask(scalp_mesh, scalp_uvs):
    img = np.zeros((256, 256, 1), 'uint8')
    
    for i in range(scalp_mesh.faces_packed().shape[0]):
        text = scalp_uvs[0][scalp_mesh.faces_packed()[i]].reshape(-1, 2).cpu().numpy()
        poly = 255/2 * (text + 1)
        rr, cc = polygon(poly[:,0], poly[:,1], img.shape)
        img[rr,cc, :] = (255)

    scalp_mask = np.flip(img.transpose(1, 0, 2), axis=0)
    return scalp_mask


def create_visibility_map(camera, rasterizer, mesh, hair_mask):
    fragments = rasterizer(mesh, cameras=camera)
    pix_to_face = fragments.pix_to_face  
    packed_faces = mesh.faces_packed() 
    packed_verts = mesh.verts_packed() 
    vertex_visibility_map = torch.zeros(packed_verts.shape[0]) 
    faces_visibility_map = torch.zeros(packed_faces.shape[0])
    pix_to_face = torch.where(hair_mask[None, ..., None], pix_to_face, -1 * torch.ones_like(pix_to_face))
    pix_to_face_vis = pix_to_face.clone() >= 0
    visible_faces = pix_to_face.unique()[1:] # not take -1
    visible_verts_idx = packed_faces[visible_faces] 
    unique_visible_verts_idx = torch.unique(visible_verts_idx)
    vertex_visibility_map[unique_visible_verts_idx] = 1.0
    faces_visibility_map[torch.unique(visible_faces)] = 1.0
    return vertex_visibility_map, faces_visibility_map, pix_to_face_vis


def check_visiblity_of_faces(cams, dataset, meshRasterizer, full_mesh, data_dir, n_views=2):
    # collect visibility maps
    os.makedirs(f'{data_dir}/flame_fitting/scalp_data/vis', exist_ok=True)
    vis_maps = []
    for cam in tqdm(range(len(cams))):
        hair_mask = torch.from_numpy((cv2.imread(dataset.hair_masks_lis[cam]) / 255.0).astype(np.float32)).cuda()[..., 0] > 0.5
        v, _, vis = create_visibility_map(cams[cam], meshRasterizer, full_mesh, hair_mask)
        Image.fromarray((vis[0, ..., 0].cpu().numpy() * 255).astype('uint8')).save(f'{data_dir}/flame_fitting/scalp_data/vis/%06d.jpg' % cam)
        vis_maps.append(v)

    # took faces that were visible at least from n_views to reduce noise
    vis_mask = (torch.stack(vis_maps).sum(0) > n_views).float()

    #idx = torch.nonzero(vis_mask).squeeze(-1).tolist()
    #idx = [i for i in idx if i < mesh_head.verts_packed().shape[0]]
    #indices_mapping = {j: i for i, j in enumerate(idx)}

    #face_faces = []
    #face_idx = torch.tensor(idx).to('cuda')
    #vertex = full_mesh.verts_packed()[face_idx]

    #for fle, i in enumerate(full_mesh.faces_packed()):
    #    if i[0] in face_idx and i[1] in face_idx and i[2] in face_idx:
    #        face_faces += [[indices_mapping[i[0].item()], indices_mapping[i[1].item()], indices_mapping[i[2].item()]]]
    #return vertex, torch.tensor(face_faces)
    return vis_mask

    
def main(args):
    
    save_path = args.data_dir
                             
    # upload mesh hair and head
    #verts_hair, faces_hair = load_ply(os.path.join(args.data_dir, '3d_gaussian_splatting', args.exp_name, 'mesh_cropped', 'iteration_30000', 'hair_mesh.ply'))
    #mesh_hair =  Meshes(verts=[verts_hair.float().to(args.device)], faces=[faces_hair.to(args.device)])

    mesh_head = load_objs_as_meshes([os.path.join(args.data_dir, 'flame_fitting', 'stage_3', 'mesh_final.obj')], device=args.device)

    # indices of scalp vertices
    scalp_vert_idx = torch.load(f'{args.project_dir}/data/new_scalp_vertex_idx.pth').long().cuda()
    # faces that form a scalp
    scalp_faces = torch.load(f'{args.project_dir}/data/new_scalp_faces.pth')[None].cuda() 
    scalp_uvs = torch.load(f'{args.project_dir}/data/new_scalp_uvcoords.pth')[None].cuda()

    # Convert the head mesh into a scalp mesh
    scalp_verts = mesh_head.verts_packed()[None, scalp_vert_idx]
    scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0]
    scalp_mesh = Meshes(verts=scalp_verts, faces=scalp_faces).cuda()

    # take from config dataset parameters
    with open(f'{args.project_dir}/{args.conf_path}', 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader))
        conf = yaml.load(replaced_conf, Loader=yaml.Loader)

    dataset = MonocularDataset(conf['dataset'], args.data_dir)

    raster_settings_mesh = RasterizationSettings(
                        image_size=(dataset.H, dataset.W), 
                        blur_radius=0.000, 
                        faces_per_pixel=1, 
                    )

    # init camera
    R = torch.ones(1, 3, 3)
    t = torch.ones(1, 3)
    cam_intr = torch.ones(1, 4, 4)
    size = torch.tensor([dataset.H, dataset.W]).to(args.device)

    cam = cameras_from_opencv_projection(
        camera_matrix=cam_intr.cuda(), 
        R=R.cuda(),
        tvec=t.cuda(),
        image_size=size[None].cuda()
    ).cuda()

    # init mesh rasterization
    meshRasterizer = MeshRasterizer(cam, raster_settings_mesh)

    #mesh_hair.textures = TexturesVertex(verts_features=torch.zeros_like(mesh_hair.verts_packed()).float().cuda()[None])
    mesh_head.textures = TexturesVertex(verts_features=torch.ones_like(mesh_head.verts_packed()).float().cuda()[None])

    # join hair and bust mesh to handle occlusions
    #full_mesh = join_meshes_as_scene([mesh_head, mesh_hair])
    full_mesh = mesh_head

    # add dataset cameras
    intrinsics_all = dataset.intrinsics_all #intrinsics
    pose_all_inv = torch.inverse(dataset.pose_all) #extrinsics

    cams_dataset = [
        cameras_from_opencv_projection(
            camera_matrix=intrinsics_all[idx][None].cuda(), 
            R=pose_all_inv[idx][:3, :3][None].cuda(),
            tvec=pose_all_inv[idx][:3, 3][None].cuda(),
            image_size=size[None].cuda()
        ).cuda() for idx in range(dataset.n_images)
    ]

    cams = cams_dataset
    #vis_vertex, vis_face = check_visiblity_of_faces(cams, head_masks, meshRasterizer, full_mesh, mesh_head, n_views=args.n_views)
    n_views = int(len(dataset.images_lis) * 0.25)
    vis_vertex_mask = check_visiblity_of_faces(cams, dataset, meshRasterizer, full_mesh, args.data_dir, n_views=n_views)

    sorted_idx = torch.where(vis_vertex_mask.bool()[scalp_vert_idx])[0]
    print(torch.where(~vis_vertex_mask.bool()[scalp_vert_idx])[0])

    # Cut new scalp
    a = np.array(sorted(sorted_idx.cpu()))
    b = np.arange(a.shape[0])
    d = dict(zip(a,b))

    full_scalp_list = sorted(sorted_idx)

    save_path = os.path.join(args.data_dir, 'flame_fitting', 'scalp_data')
    os.makedirs(save_path, exist_ok=True)

    faces_masked = []
    for face in scalp_mesh.faces_packed():
#         print(face[0] , full_scalp_list)
#         input()
        if face[0] in full_scalp_list and face[1] in full_scalp_list and  face[2] in full_scalp_list:
            faces_masked.append(torch.tensor([d[int(face[0])], d[int(face[1])], d[int(face[2])]]))
#         print(faces_masked, full_scalp_list)
    save_obj(os.path.join(save_path, 'scalp.obj'), scalp_mesh.verts_packed()[full_scalp_list], torch.stack(faces_masked))

    with open(os.path.join(save_path, 'cut_scalp_verts.pickle'), 'wb') as f:
        pickle.dump(list(torch.tensor(sorted_idx).detach().cpu().numpy()), f)
    
    # Create scalp mask for diffusion
    scalp_uvs = scalp_uvs[:, full_scalp_list]    
    scalp_mesh = load_objs_as_meshes([os.path.join(save_path, 'scalp.obj')], device=args.device)
    
    scalp_mask = create_scalp_mask(scalp_mesh, scalp_uvs)
    cv2.imwrite(os.path.join(save_path, 'dif_mask.png'), scalp_mask)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--conf_path', default='configs/monocular/neural_strands.yaml', type=str)
    parser.add_argument('--project_dir', default="", type=str)
    parser.add_argument('--data_dir', default="", type=str)
    parser.add_argument('--exp_name', default="", type=str) 
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--img_size', default=2160, type=int)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)