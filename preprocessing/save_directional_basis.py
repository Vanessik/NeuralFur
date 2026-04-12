import os
import numpy as np
import trimesh
from collections import deque
from scipy.spatial import cKDTree

from common import create_parser, get_data_path


def load_dmat(path):
    """Load a .dmat file (Directional library format)."""
    with open(path, "r") as f:
        cols = int(f.readline().strip())
        rows = int(f.readline().strip())
        data = []
        for line in f:
            vals = line.strip().split()
            data.extend(float(v) for v in vals)
    return np.array(data).reshape(rows, cols)


def fill_nan_faces(mesh, raw_field, n_directions=2):
    """Fill NaN face fields using BFS adjacency, then spatial nearest-neighbor fallback."""
    num_faces = len(mesh.faces)
    field = raw_field.reshape(num_faces, n_directions, 3).copy()

    valid_mask = ~np.isnan(field).any(axis=(1, 2))

    # Build face adjacency
    adjacent_faces = [[] for _ in range(num_faces)]
    for a, b in mesh.face_adjacency:
        adjacent_faces[a].append(b)
        adjacent_faces[b].append(a)

    # BFS fill
    invalid_indices = np.where(~valid_mask)[0]
    for idx in invalid_indices:
        queue = deque([idx])
        visited = set()
        while queue:
            current = queue.popleft()
            visited.add(current)
            for nb in adjacent_faces[current]:
                if valid_mask[nb]:
                    field[idx] = field[nb]
                    valid_mask[idx] = True
                    break
                elif nb not in visited:
                    queue.append(nb)
            if valid_mask[idx]:
                break

    # Nearest-neighbor fallback
    still_invalid = ~valid_mask
    if still_invalid.any():
        print(f"Falling back to nearest-neighbor for {still_invalid.sum()} faces")
        face_centroids = mesh.triangles_center
        tree = cKDTree(face_centroids[valid_mask])
        _, nn_indices = tree.query(face_centroids[still_invalid])
        field[still_invalid] = field[valid_mask][nn_indices]

    assert np.isnan(field).sum() == 0, "Some NaN values remain after filling"
    return field


def main(animal, root_path, input_field, input_mesh, output):
    data_path = get_data_path(root_path, animal)

    mesh_path = os.path.join(data_path, input_mesh)
    field_path = os.path.join(data_path, input_field)
    save_path = os.path.join(data_path, output)

    mesh = trimesh.load(mesh_path)
    raw_face_field = load_dmat(field_path).T

    filled_field = fill_nan_faces(mesh, raw_face_field, n_directions=2)

    # Extract and normalize the second tangent direction
    t1 = filled_field[:, 1]
    norms = np.linalg.norm(t1, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    t1 = t1 / norms

    np.save(save_path, t1)
    print(f"Saved normalized tangent field to {save_path}")


if __name__ == "__main__":
    parser = create_parser("Process Directional library output into normalized tangent field")
    parser.add_argument("--input_field", default="rawFaceField.dmat",
                        help="Input .dmat file from Directional library")
    parser.add_argument("--input_mesh", default="furless_reshaped.obj",
                        help="Mesh file used with Directional")
    parser.add_argument("--output", "-o", default="field_furless_reshaped.npy",
                        help="Output .npy filename for the tangent field")
    args = parser.parse_args()

    main(args.animal, args.root_path, args.input_field, args.input_mesh, args.output)
