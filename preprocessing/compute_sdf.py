import os
import numpy as np
import trimesh
import trimesh.proximity
import torch
import torch.nn.functional as F

from common import create_parser, get_data_path


def mesh_to_sdf(path, grid_size=64):
    """Compute a signed distance field on a regular grid from a mesh."""
    mesh = trimesh.load(path, process=True)

    bbox = mesh.bounds
    min_bound = bbox[0] - 0.02
    max_bound = bbox[1] + 0.02
    lin = [np.linspace(min_bound[i], max_bound[i], grid_size) for i in range(3)]
    grid = np.stack(np.meshgrid(*lin, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)

    sdf = trimesh.proximity.signed_distance(mesh, points)
    sdf_grid = sdf.reshape((grid_size, grid_size, grid_size))
    sdf_grid *= -1
    return sdf_grid, points, min_bound, max_bound


def make_sdf_tensor(sdf_grid_np, min_bound, max_bound, device="cpu"):
    """Convert a NumPy SDF grid (D,H,W) into a torch tensor (1,1,D,H,W)."""
    sdf = torch.from_numpy(sdf_grid_np).float().to(device)
    return (
        sdf.unsqueeze(0).unsqueeze(0),
        torch.tensor(min_bound, dtype=torch.float32, device=device),
        torch.tensor(max_bound, dtype=torch.float32, device=device),
    )


def query_sdf(sdf_tensor, bounds_min, bounds_max, points):
    """
    Query the SDF volume at given world-space points using trilinear interpolation.

    Args:
        sdf_tensor: (1,1,D,H,W)
        bounds_min: (3,) tensor
        bounds_max: (3,) tensor
        points: (N,3) tensor in world space

    Returns:
        sdf_vals: (N,) differentiable w.r.t. points
    """
    p_norm = 2.0 * (points - bounds_min[None]) / (bounds_max - bounds_min)[None] - 1.0
    grid = p_norm.view(1, -1, 1, 1, 3)
    grid = grid[..., [2, 1, 0]]

    sampled = F.grid_sample(
        sdf_tensor, grid, mode="bilinear", align_corners=True,
    )
    return sampled.view(-1)


def main(animal, root_path, mode="furless_reshaped", grid_size=32):
    data_path = get_data_path(root_path, animal)

    mesh_path = os.path.join(data_path, f"{mode}.obj")
    sdf_grid, points, min_bound, max_bound = mesh_to_sdf(mesh_path, grid_size=grid_size)

    np.save(os.path.join(data_path, "sdf_grid.npy"), sdf_grid)
    np.save(os.path.join(data_path, "min_bound.npy"), min_bound)
    np.save(os.path.join(data_path, "max_bound.npy"), max_bound)
    print(f"Saved SDF grid ({grid_size}^3) to {data_path}")


if __name__ == "__main__":
    parser = create_parser("Compute signed distance field from mesh")
    parser.add_argument("--mode", "-m", default="furless_reshaped",
                        help="Mesh name prefix (e.g., furless_reshaped)")
    parser.add_argument("--grid_size", type=int, default=32,
                        help="SDF grid resolution")
    args = parser.parse_args()

    main(args.animal, args.root_path, mode=args.mode, grid_size=args.grid_size)
