import os
import shutil
import numpy as np
import trimesh

from common import create_parser, get_data_path, get_scene_type
from animal_config import neus_dict, postfixx


def main(animal, neus_root_path, root_path):
    scene_type = get_scene_type(animal)
    scene_neus_path = neus_dict[animal]

    src_path = os.path.join(root_path, f"{animal}_{postfixx[animal]}", scene_type)
    save_path = get_data_path(root_path, animal)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "images_2"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "masks_2", "hair"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "masks_2", "body"), exist_ok=True)

    original_image_list = sorted(os.listdir(os.path.join(src_path, "images")))
    print(f"Found {len(original_image_list)} images")

    for idx in range(len(original_image_list)):
        img_name = original_image_list[idx]
        padded_name = f"{idx:04d}.png"

        shutil.copy(
            os.path.join(src_path, "images", img_name),
            os.path.join(save_path, "images_2", padded_name),
        )
        shutil.copy(
            os.path.join(src_path, "silhouette", img_name),
            os.path.join(save_path, "masks_2", "hair", padded_name),
        )
        shutil.copy(
            os.path.join(src_path, "silhouette", img_name),
            os.path.join(save_path, "masks_2", "body", padded_name),
        )

    # Save cameras
    cam_path = os.path.join(src_path, "cameras.npz")
    np.save(os.path.join(save_path, "projection.npy"), np.load(cam_path)["arr_0"])

    scale_src = os.path.join(src_path, "scale.pickle")
    if os.path.exists(scale_src):
        shutil.copy(scale_src, os.path.join(save_path, "scale.pickle"))

    # Export NeuS meshes as OBJ (high-res and low-res)
    neus_mesh_dir = os.path.join(neus_root_path, scene_neus_path, scene_type, "wmask", "meshes")

    mesh_hr = trimesh.load(os.path.join(neus_mesh_dir, "00300000.ply"))
    mesh_hr.export(os.path.join(save_path, "neus.obj"))

    mesh_lr = trimesh.load(os.path.join(neus_mesh_dir, "00290000.ply"))
    mesh_lr.export(os.path.join(save_path, "neus_lr.obj"))


if __name__ == "__main__":
    parser = create_parser("Convert NeuS data to GaussianHaircut format")
    parser.add_argument("--neus_root_path", required=True, help="Path to NeuS exp folder")
    args = parser.parse_args()

    main(args.animal, args.neus_root_path, args.root_path)
