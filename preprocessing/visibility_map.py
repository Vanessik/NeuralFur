import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import imageio
import cv2 as cv

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer import TexturesVertex, HardPhongShader, PointLights, MeshRenderer
from pytorch3d.io import load_obj

from common import create_parser, get_data_path


def load_K_Rt_from_P(P):
    """Decompose a 3x4 projection matrix into intrinsics and pose."""
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


def create_visibility_map(camera, rasterizer, mesh):
    """Rasterize mesh from a camera view and return per-vertex/face visibility."""
    fragments = rasterizer(mesh, cameras=camera)
    pix_to_face = fragments.pix_to_face
    packed_faces = mesh.faces_packed()
    packed_verts = mesh.verts_packed()

    vertex_visibility_map = torch.zeros(packed_verts.shape[0])
    faces_visibility_map = torch.zeros(packed_faces.shape[0])

    visible_faces = pix_to_face.unique()[1:]  # skip -1
    visible_verts_idx = packed_faces[visible_faces]
    unique_visible_verts_idx = torch.unique(visible_verts_idx)

    vertex_visibility_map[unique_visible_verts_idx] = 1.0
    faces_visibility_map[torch.unique(visible_faces)] = 1.0
    return vertex_visibility_map, faces_visibility_map


def main(animal, root_path, mesh_name="furless_reshaped.obj",
         annotations_name="annotations_furless_reshaped.json",
         exclude_groups=None):
    if exclude_groups is None:
        exclude_groups = ["paw_pads", "eyes", "nosetip"]

    data_path = get_data_path(root_path, animal)
    device = "cuda"

    mesh_path = os.path.join(data_path, mesh_name)
    cam_path = os.path.join(data_path, "projection.npy")
    annots_path = os.path.join(data_path, annotations_name)

    verts, faces, _ = load_obj(mesh_path)
    mesh = Meshes(
        verts=[verts.float().to(device)],
        faces=[faces.verts_idx.to(device)],
    )

    # Load an image to get dimensions
    sample_img = os.path.join(data_path, "images_2", "0000.png")
    H, W = Image.open(sample_img).size[::-1]

    raster_settings = RasterizationSettings(
        image_size=(int(H), int(W)),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Load annotations and build per-vertex group IDs
    with open(annots_path, "r") as f:
        vertex_group_dict = json.load(f)

    group_names = list(vertex_group_dict.keys())
    group_to_id = {name: i for i, name in enumerate(group_names)}

    V = verts.shape[0]
    vertex_groups = torch.full((V,), fill_value=-1, dtype=torch.long, device=verts.device)
    for group_name, vertex_ids in vertex_group_dict.items():
        vertex_groups[vertex_ids] = group_to_id[group_name]

    # Build mask for excluded body parts (paw_pads, eyes, nosetip)
    masks = []
    for g in exclude_groups:
        if g in group_to_id:
            masks.append(vertex_groups == group_to_id[g])
    selected_mask = torch.zeros(V, dtype=torch.float, device=verts.device)
    for m in masks:
        selected_mask = torch.max(selected_mask, m.float())

    texture = selected_mask[..., None].repeat(1, 3)[None].to(device)
    mesh.textures = TexturesVertex(verts_features=texture)

    # Load cameras
    cameras = np.load(cam_path)
    intr_list, extr_list = [], []
    for frame_id in range(len(cameras)):
        intr, pose = load_K_Rt_from_P(cameras[frame_id, :3, :])
        intr_list.append(intr)
        extr_list.append(pose)

    intrinsics = torch.tensor(np.stack(intr_list), device=device).float()
    pose_inv = torch.inverse(torch.tensor(np.stack(extr_list), device=device)).float()
    size = torch.tensor([int(H), int(W)], device=device)

    # Build per-frame cameras
    cams_dataset = [
        cameras_from_opencv_projection(
            camera_matrix=intrinsics[idx][None],
            R=pose_inv[idx][:3, :3][None],
            tvec=pose_inv[idx][:3, 3][None],
            image_size=size[None],
        ).to(device)
        for idx in range(len(pose_inv))
    ]

    # Render bald masks per frame
    save_path = os.path.join(data_path, "masks_2", "bald")
    os.makedirs(save_path, exist_ok=True)

    for frame in tqdm(range(len(cams_dataset)), desc="Rendering bald masks"):
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cams_dataset[frame], raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cams_dataset[frame], lights=lights),
        )

        image = renderer(mesh)
        bg_mask = (image[..., :3].mean(dim=-1) > 0.99).float()
        my_mask = (image[..., :3].mean(dim=-1) < 0.2).float()
        union_mask = torch.max(bg_mask, my_mask)

        mask_np = union_mask[0].detach().cpu().numpy()
        mask_img = (mask_np * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(save_path, f"{frame:04d}.png"), mask_img)


if __name__ == "__main__":
    parser = create_parser("Create visibility masks for bald regions (paw_pads, eyes, nosetip)")
    parser.add_argument("--mesh_name", default="furless_reshaped.obj",
                        help="Mesh filename inside data directory")
    parser.add_argument("--annotations_name", default="annotations_furless_reshaped.json",
                        help="Annotations JSON filename")
    parser.add_argument("--exclude_groups", nargs="+", default=["paw_pads", "eyes", "nosetip"],
                        help="Body part groups to mark as bald")
    args = parser.parse_args()

    main(args.animal, args.root_path,
         mesh_name=args.mesh_name,
         annotations_name=args.annotations_name,
         exclude_groups=args.exclude_groups)
