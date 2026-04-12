import os
import numpy as np
from PIL import Image
import torch
import pickle
import argparse
from tqdm import tqdm

from common import get_scene_type
from animal_config import timesteps_dict, scales_dict


def campose_to_extrinsic(camposes):
    if camposes.shape[1] != 12:
        raise Exception("Wrong campose data structure!")

    res = np.zeros((camposes.shape[0], 4, 4))
    res[:, 0:3, 2] = camposes[:, 0:3]
    res[:, 0:3, 0] = camposes[:, 3:6]
    res[:, 0:3, 1] = camposes[:, 6:9]
    res[:, 0:3, 3] = camposes[:, 9:12]
    res[:, 3, 3] = 1.0
    return res


def read_intrinsics(fn_instrinsic):
    with open(fn_instrinsic) as fo:
        data = fo.readlines()

    Ks, i = [], 0
    while i < len(data):
        if len(data[i]) > 6:
            a = np.array([float(x) for x in data[i].split()]); i += 1
            b = np.array([float(x) for x in data[i].split()]); i += 1
            c = np.array([float(x) for x in data[i].split()])
            Ks.append(np.vstack([a, b, c]))
        i += 1
    return np.stack(Ks)


def main(animal, root_path, save_root_path):
    scene_type = get_scene_type(animal)
    timestep = str(timesteps_dict[animal])

    save_path = os.path.join(save_root_path, f"{animal}_processed", scene_type)
    scale_name = scales_dict.get(animal, "")
    path_to_scale = os.path.join(save_root_path, scale_name) if scale_name else None

    os.makedirs(save_path, exist_ok=True)

    save_path_img = os.path.join(save_path, "images")
    save_path_silh = os.path.join(save_path, "silhouette")
    os.makedirs(save_path_img, exist_ok=True)
    os.makedirs(save_path_silh, exist_ok=True)

    img_dir = os.path.join(root_path, animal, "img", scene_type, timestep)
    rgb_paths = sorted([f for f in os.listdir(img_dir) if "alpha" not in f])
    silh_paths = sorted([f for f in os.listdir(img_dir) if "alpha" in f])

    for idx in tqdm(range(len(rgb_paths)), desc="Copying images"):
        pil_img = Image.open(os.path.join(img_dir, rgb_paths[idx]))
        pil_silh = Image.open(os.path.join(img_dir, silh_paths[idx]))
        pil_img.save(os.path.join(save_path_img, rgb_paths[idx]))
        pil_silh.save(os.path.join(save_path_silh, rgb_paths[idx]))

    # Prepare cameras
    cam_intr = os.path.join(root_path, animal, "Intrinsic.inf")
    cam_extr = os.path.join(root_path, animal, "CamPose.inf")

    camposes = torch.from_numpy(campose_to_extrinsic(np.loadtxt(cam_extr)))
    extr_matx = torch.inverse(camposes).float()
    intr_matx = torch.Tensor(read_intrinsics(cam_intr)).float()

    full_intr_matx = torch.eye(4)[None].repeat(intr_matx.shape[0], 1, 1)
    full_intr_matx[:, :3, :3] = intr_matx

    all_cams = np.array(full_intr_matx @ extr_matx)
    np.savez(os.path.join(save_path, "cameras_wo_scale.npz"), all_cams)

    scale_mat = np.eye(4, dtype=np.float32)
    if path_to_scale and os.path.exists(path_to_scale):
        try:
            with open(path_to_scale, "rb") as f:
                transform = pickle.load(f)
            print(f"Loaded scale: {transform['scale']}")
            scale_mat[:3, :3] *= transform["scale"]
            scale_mat[:3, 3] = transform["translation"]
        except Exception:
            print("Could not load scale file, continuing without scale")
    else:
        print("No scale file provided")

    proj = all_cams @ scale_mat
    scaled_extr = extr_matx @ torch.from_numpy(scale_mat).float()

    np.save(os.path.join(save_path, "cameras_intr.npy"), full_intr_matx)
    np.save(os.path.join(save_path, "cameras_extr.npy"), scaled_extr)
    np.save(os.path.join(save_path, "cameras_extr_wo_scale.npy"), extr_matx)
    np.savez(os.path.join(save_path, "cameras.npz"), proj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw Artemis data to NeuS format")
    parser.add_argument("--animal", "-a", required=True, help="Animal name")
    parser.add_argument("--root_path", required=True, help="Path to raw Artemis data")
    parser.add_argument("--save_path", required=True, help="Path to save processed data")
    args = parser.parse_args()

    main(args.animal, args.root_path, args.save_path)
