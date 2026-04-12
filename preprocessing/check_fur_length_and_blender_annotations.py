import os
import numpy as np
import json
from pytorch3d.io import load_obj

from common import create_parser, get_data_path


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


def main(animal, root_path, mode="furless_reshaped"):
    data_path = get_data_path(root_path, animal)

    annots_path = os.path.join(data_path, f"annotations_{mode}.json")
    mesh_path = os.path.join(data_path, f"{mode}.obj")

    with open(annots_path, "r") as f:
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
    print(f"Unannotated vertices assigned to 'body': {len(body_indices)}")

    new_annots = convert_ndarray(data)
    with open(annots_path, "w") as f:
        json.dump(new_annots, f, indent=4)

    print(f"Updated annotations saved to {annots_path}")


if __name__ == "__main__":
    parser = create_parser("Validate annotations and assign unannotated vertices to 'body'")
    parser.add_argument("--mode", "-m", default="furless_reshaped",
                        help="Mesh name prefix (e.g., furless_reshaped)")
    args = parser.parse_args()

    main(args.animal, args.root_path, mode=args.mode)
