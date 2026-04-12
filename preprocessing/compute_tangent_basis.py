import os
import torch
import numpy as np
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes

from common import create_parser, get_data_path
from mesh_utils import safe_normalize, fix_normals_with_neighbors, orient_face_tangents_parallel_transport, export_normals_visualization_with_cones


def main(animal, root_path, mode="furless_reshaped", visualize_tan=False):
    data_path = get_data_path(root_path, animal)

    scalp_path = os.path.join(data_path, f"{mode}.obj")
    tan_path = os.path.join(data_path, f"field_{mode}.npy")

    device = "cuda"
    scalp_mesh = load_objs_as_meshes([scalp_path], device=device)
    scalp_mesh = Meshes(
        verts=scalp_mesh.verts_packed()[None],
        faces=scalp_mesh.faces_packed()[None],
    ).cuda()

    tangents = torch.tensor(np.load(tan_path), device=device).float()
    origin_t = safe_normalize(tangents)

    origin_n = fix_normals_with_neighbors(scalp_mesh)
    origin_n = safe_normalize(origin_n)

    mesh = trimesh.load_mesh(scalp_path, process=True)
    oriented_tangents, _ = orient_face_tangents_parallel_transport(
        mesh, origin_t.detach().cpu().numpy(), seed_face=0,
    )

    save_path = os.path.join(data_path, f"tan_{mode}.npy")
    np.save(save_path, oriented_tangents)
    print(f"Saved oriented tangents to {save_path}")

    if visualize_tan:
        scalp_verts = scalp_mesh.verts_packed()
        scalp_faces = scalp_mesh.faces_packed()
        export_normals_visualization_with_cones(
            scalp_verts, scalp_faces,
            torch.tensor(oriented_tangents),
            filename=os.path.join(data_path, f"fixed_tangents_{animal}.ply"),
            scale=0.02,
        )


if __name__ == "__main__":
    parser = create_parser("Compute and orient tangent basis on mesh")
    parser.add_argument("--mode", "-m", default="furless_reshaped",
                        help="Mesh name prefix (e.g., furless_reshaped)")
    parser.add_argument("--visualize_tan", action="store_true",
                        help="Export tangent visualization as PLY")
    args = parser.parse_args()

    main(args.animal, args.root_path, mode=args.mode, visualize_tan=args.visualize_tan)
