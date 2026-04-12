import os
import numpy as np
import trimesh
import json
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pickle
from pytorch3d.io import load_obj

from common import create_parser, get_data_path
from animal_config import source_paths_SMAL, source_paths_SMAL_furless, manes


def nearest_label_transfer(mesh_1_verts, mesh_2_verts, annotation_json_path, use_mane, output_json_path=None):
    """
    For each vertex in mesh_2, find the closest vertex in mesh_1.
    If that mesh_1 vertex is annotated, assign the same label to the mesh_2 vertex.
    """
    with open(annotation_json_path, "r") as f:
        annotations = json.load(f)

    if not use_mane and "mane" in annotations:
        del annotations["mane"]

    vertex_to_label = {}
    for label, indices in annotations.items():
        for idx in indices:
            vertex_to_label[idx] = label

    mesh1_kdtree = cKDTree(mesh_1_verts)
    _, nearest_indices = mesh1_kdtree.query(mesh_2_verts, k=1)

    mesh2_annotations = {}
    for mesh2_idx, mesh1_idx in enumerate(nearest_indices):
        if mesh1_idx in vertex_to_label:
            label = vertex_to_label[mesh1_idx]
            if label not in mesh2_annotations:
                mesh2_annotations[label] = []
            mesh2_annotations[label].append(mesh2_idx)

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(mesh2_annotations, f, indent=2)

    return mesh2_annotations


def convert_ndarray(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    return obj


def main(animal, root_path, annotation_json, input_mesh_path, full_shape=False):
    if full_shape:
        source_paths = source_paths_SMAL
        postf = ""
    else:
        source_paths = source_paths_SMAL_furless
        postf = "_furless"

    use_mane = manes[animal]
    source_path = source_paths[animal]
    data_path = get_data_path(root_path, animal)

    transfer_mesh = trimesh.load(os.path.join(data_path, input_mesh_path))
    annots_name = "annotations_" + input_mesh_path.replace(".obj", ".json")

    save_annots_path = os.path.join(data_path, annots_name)
    save_mesh_name = os.path.join(data_path, f"colored{postf}.ply")
    save_smal_model = os.path.join(data_path, f"smal{postf}.obj")

    source_mesh = trimesh.load(source_path)

    # Animal-specific coordinate transformations for SMAL alignment
    if animal == "cat":
        scale_path = os.path.join(data_path, "scale.pickle")
        with open(scale_path, "rb") as f:
            scale_mat = pickle.load(f)
        source_mesh.vertices -= scale_mat["translation"]
        source_mesh.vertices /= scale_mat["scale"]

    elif animal == "synth_tiger":
        pass  # no transformation needed

    elif animal == "fox":
        rot_x = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        rot_y = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        source_mesh.apply_transform(rot_x)
        source_mesh.apply_transform(rot_y)

    else:
        # Default: beagle_dog, panda, whiteTiger, etc.
        rot_x = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        rot_y = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
        source_mesh.apply_transform(rot_x)
        source_mesh.apply_transform(rot_y)

    source_mesh.export(save_smal_model)

    # Transfer labels
    dense_labels = nearest_label_transfer(
        source_mesh.vertices, transfer_mesh.vertices,
        annotation_json, use_mane, output_json_path=save_annots_path,
    )

    # Assign distinct colors per part
    part_names = list(dense_labels.keys())
    num_parts = len(part_names)
    cmap = plt.cm.get_cmap("tab20", num_parts)

    part_colors = {
        part: (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        for i, part in enumerate(part_names)
    }

    num_vertices = len(transfer_mesh.vertices)
    colors = np.ones((num_vertices, 3), dtype=np.uint8) * 255

    for part, indices in dense_labels.items():
        colors[indices] = part_colors[part]

    transfer_mesh.visual = trimesh.visual.ColorVisuals(transfer_mesh, vertex_colors=colors)
    transfer_mesh.export(save_mesh_name)

    # Add "body" label for unannotated vertices
    mesh_path = os.path.join(data_path, input_mesh_path)
    with open(save_annots_path, "r") as f:
        data = json.load(f)

    verts, faces, _ = load_obj(mesh_path, device="cuda")
    num_verts = verts.shape[0]

    used_indices = set()
    for key in data:
        used_indices.update(data[key])

    body_indices = [i for i in range(num_verts) if i not in used_indices]
    data["body"] = body_indices

    print(f"Annotation keys: {list(data.keys())}")
    print(f"Total categories (including body): {len(data)}")

    new_annots = convert_ndarray(data)
    with open(save_annots_path, "w") as f:
        json.dump(new_annots, f, indent=4)


if __name__ == "__main__":
    parser = create_parser("Transfer SMAL annotations to target mesh")
    parser.add_argument("--annotation_json", "-j", required=True,
                        help="Path to SMAL annotation JSON file")
    parser.add_argument("--input_mesh_path", "-i", default="furless_reshaped.obj",
                        help="Mesh filename inside the data directory")
    parser.add_argument("--full_shape", action="store_true",
                        help="Use full SMAL shape instead of furless")
    args = parser.parse_args()

    main(args.animal, args.root_path, args.annotation_json, args.input_mesh_path, args.full_shape)
