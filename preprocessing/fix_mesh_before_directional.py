import os
import trimesh
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import pymeshlab

from common import create_parser, get_data_path


def compute_vertex_adjacency(n_vertices, faces):
    """Build a sparse adjacency matrix from face indices."""
    I, J = [], []
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            I.extend([v1, v2])
            J.extend([v2, v1])
    data = np.ones(len(I), dtype=np.float32)
    return coo_matrix((data, (I, J)), shape=(n_vertices, n_vertices)).tocsr()


def laplacian_smooth(mesh, iterations=10, lambda_factor=0.5):
    """Apply Laplacian smoothing to a mesh."""
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    n = len(vertices)

    adjacency = compute_vertex_adjacency(n, faces)
    deg = np.array(adjacency.sum(axis=1)).flatten()
    deg_inv = 1.0 / (deg + 1e-8)
    deg_inv_matrix = csr_matrix((deg_inv, (np.arange(n), np.arange(n))), shape=(n, n))
    normalized_laplacian = deg_inv_matrix @ adjacency

    for _ in range(iterations):
        displacement = normalized_laplacian @ vertices - vertices
        vertices += lambda_factor * displacement

    smoothed = mesh.copy()
    smoothed.vertices = vertices
    return smoothed


def clean_mesh(input_path, output_path, min_faces=20000, smooth=False, iterations=10):
    """Clean a mesh: remove degenerate/duplicate faces, fill holes, optionally smooth."""
    mesh = trimesh.load(input_path, force="mesh")

    print("[INFO] Running mesh cleaning...")
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.fill_holes()

    components = mesh.split(only_watertight=False)
    print(f"[INFO] Found {len(components)} connected components.")
    large_components = [c for c in components if len(c.faces) >= min_faces]

    if len(large_components) == 0:
        raise ValueError(f"No components found with >= {min_faces} faces.")

    mesh = trimesh.util.concatenate(large_components)
    mesh.remove_unreferenced_vertices()

    if smooth:
        print("[INFO] Applying Laplacian smoothing...")
        mesh = laplacian_smooth(mesh, iterations=iterations, lambda_factor=0.4)

    mesh.export(output_path)
    print(f"Saved cleaned mesh to {output_path}")


def subdivide(path, target_faces, output_path):
    """Decimate mesh to target face count using pymeshlab."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    ms.apply_filter("meshing_merge_close_vertices")
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=target_faces,
        preservenormal=True,
        preservetopology=True,
    )
    ms.save_current_mesh(output_path)
    print(f"Decimated mesh to {target_faces} faces -> {output_path}")


def main(animal, root_path, input_mesh, target_faces_hr=160000, target_faces_lr=10000):
    data_path = get_data_path(root_path, animal)

    input_path = os.path.join(data_path, input_mesh)
    repaired_path = os.path.join(data_path, f"repaired_smoothed_{animal}.obj")
    save_hr = os.path.join(data_path, "furless_reshaped.obj")
    save_lr = os.path.join(data_path, "furless_reshaped_lr.obj")

    # Step 1: Clean and smooth
    clean_mesh(input_path, repaired_path, min_faces=20000, smooth=True, iterations=10)

    # Step 2: Decimate to high-res and low-res
    subdivide(repaired_path, target_faces_hr, save_hr)
    subdivide(repaired_path, target_faces_lr, save_lr)

    # Step 3: Final clean (no smoothing)
    clean_mesh(save_hr, save_hr, min_faces=20000, smooth=False)
    clean_mesh(save_lr, save_lr, min_faces=2000, smooth=False)


if __name__ == "__main__":
    parser = create_parser("Clean and decimate mesh before Directional processing")
    parser.add_argument("--input_mesh", "-i", required=True,
                        help="Input mesh filename inside data directory (e.g., shrunken_panda_sdf_only.obj)")
    parser.add_argument("--target_faces_hr", type=int, default=160000,
                        help="Target face count for high-res output")
    parser.add_argument("--target_faces_lr", type=int, default=10000,
                        help="Target face count for low-res output")
    args = parser.parse_args()

    main(args.animal, args.root_path, args.input_mesh,
         target_faces_hr=args.target_faces_hr,
         target_faces_lr=args.target_faces_lr)
