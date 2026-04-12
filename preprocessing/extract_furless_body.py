import os
import numpy as np
import trimesh
import json
from collections import defaultdict
from scipy.spatial import cKDTree

os.environ["PYOPENGL_PLATFORM"] = "egl"
import mesh_to_sdf
import mcubes

from common import create_parser, get_data_path
from animal_config import effective_fur_thickness_cm


def laplacian_smooth_scalar_field(mesh, field, iterations=10):
    """Uniform Laplacian smoothing on a scalar per-vertex field."""
    neighbors = defaultdict(list)
    for i, nbrs in enumerate(mesh.vertex_neighbors):
        neighbors[i] = nbrs

    smoothed = field.copy()
    for _ in range(iterations):
        new_field = smoothed.copy()
        for i in range(len(smoothed)):
            nbrs = neighbors[i]
            if nbrs:
                new_field[i] = np.mean([smoothed[j] for j in nbrs])
        smoothed = new_field
    return smoothed


def main(animal, root_path, eye_dist_vqa, smooth_iterations=100, voxel_resolution=256):
    data_path = get_data_path(root_path, animal)

    mesh_path = os.path.join(data_path, "neus.obj")
    annots_path = os.path.join(data_path, "annotations.json")
    eye_landmarks_path = os.path.join(data_path, "eyes.ply")

    mesh = trimesh.load(mesh_path)

    with open(annots_path, "r") as f:
        data = json.load(f)

    fur_thickness = effective_fur_thickness_cm[animal]

    # Compute metric scale from eye landmarks
    eyes_geom = np.array(trimesh.load(eye_landmarks_path).vertices)
    eyes_geom_dist = np.linalg.norm(eyes_geom[0] - eyes_geom[1])
    metric_scale = eyes_geom_dist / eye_dist_vqa

    # Build per-vertex shrinkage field
    shrinkage_field = np.zeros(len(mesh.vertices))
    for part_name, thickness in fur_thickness.items():
        indices = data.get(part_name, [])
        if not indices:
            continue
        shrinkage = metric_scale * thickness
        shrinkage_field[indices] = shrinkage

    normals = mesh.vertex_normals

    # Save raw (unsmoothed) shrinkage mesh
    vertices_raw = mesh.vertices - normals * shrinkage_field[:, np.newaxis]
    mesh_raw = trimesh.Trimesh(vertices=vertices_raw, faces=mesh.faces)
    raw_path = os.path.join(data_path, f"shrunk_metric_raw_{animal}.obj")
    mesh_raw.export(raw_path)
    print(f"Saved unsmoothed shrinkage mesh to {raw_path}")

    # Laplacian smoothing on shrinkage field
    smoothed_shrinkage = laplacian_smooth_scalar_field(mesh, shrinkage_field, iterations=smooth_iterations)

    # Save smoothed shrinkage mesh
    vertices_smoothed = mesh.vertices - normals * smoothed_shrinkage[:, np.newaxis]
    mesh_smoothed = trimesh.Trimesh(vertices=vertices_smoothed, faces=mesh.faces)
    smooth_path = os.path.join(data_path, f"shrunk_metric_smoothed_{animal}.obj")
    mesh_smoothed.export(smooth_path)
    print(f"Saved smoothed shrinkage mesh to {smooth_path}")

    # --- SDF-based mesh reconstruction ---
    mesh_input = trimesh.load(mesh_path, force="mesh")
    if not mesh_input.is_watertight:
        print("[WARNING] Input mesh is not watertight -- results may be unreliable.")

    original_vertices = mesh_input.vertices
    bbox_min, bbox_max = mesh_input.bounds
    scale = bbox_max - bbox_min
    scale[scale == 0] = 1e-6

    mesh_scaled = mesh_input.copy()
    mesh_scaled.vertices = (mesh_input.vertices - bbox_min) / scale

    sdf_grid = mesh_to_sdf.mesh_to_voxels(
        mesh_scaled,
        voxel_resolution=voxel_resolution,
        surface_point_method="scan",
        sign_method="normal",
        scan_count=100,
    )

    D, H, W = sdf_grid.shape
    x = np.linspace(0, 1, D)
    y = np.linspace(0, 1, H)
    z = np.linspace(0, 1, W)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid_points_unit = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    grid_points_world = grid_points_unit * scale + bbox_min

    # Map shrinkage to grid via KDTree
    tree = cKDTree(original_vertices)
    _, nearest_vertex_indices = tree.query(grid_points_world)
    local_shrinkage = smoothed_shrinkage[nearest_vertex_indices]

    # Modify SDF and run marching cubes
    sdf_flat = sdf_grid.flatten()
    modified_sdf = (sdf_flat + local_shrinkage).reshape(sdf_grid.shape)
    verts_voxel, faces_voxel = mcubes.marching_cubes(modified_sdf, 0.0)

    verts_unit = verts_voxel / (voxel_resolution - 1)
    verts_world = verts_unit * scale + bbox_min

    shrunken_mesh = trimesh.Trimesh(vertices=verts_world, faces=faces_voxel)
    output_path = os.path.join(data_path, f"shrunken_{animal}_sdf_only.obj")
    shrunken_mesh.export(output_path)
    print(f"Shrunken mesh exported to {output_path}")


if __name__ == "__main__":
    parser = create_parser("Extract furless body by shrinking mesh along normals")
    parser.add_argument("--eye_dist_vqa", type=float, required=True,
                        help="Real-world eye distance in cm (from ChatGPT annotation)")
    parser.add_argument("--smooth_iterations", type=int, default=100,
                        help="Number of Laplacian smoothing iterations")
    parser.add_argument("--voxel_resolution", type=int, default=256,
                        help="Voxel resolution for SDF marching cubes")
    args = parser.parse_args()

    main(args.animal, args.root_path, args.eye_dist_vqa,
         smooth_iterations=args.smooth_iterations,
         voxel_resolution=args.voxel_resolution)
