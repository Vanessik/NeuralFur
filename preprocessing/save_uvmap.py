import os
import torch
from pytorch3d.io import load_obj

from common import create_parser, get_data_path


def main(animal, root_path, obj_path, save_name):
    data_path = get_data_path(root_path, animal)

    data = os.path.join(data_path, obj_path)
    verts, faces, uv = load_obj(data)

    uv_idx = faces.textures_idx.reshape(-1)
    v_idx = faces.verts_idx.reshape(-1)

    V = verts.shape[0]
    vertex_uv = torch.zeros((V, 2), dtype=uv.verts_uvs.dtype)
    seen = torch.zeros((V,), dtype=torch.bool)

    for corner in range(uv_idx.shape[0]):
        v = v_idx[corner].item()
        if not seen[v]:
            vertex_uv[v] = uv.verts_uvs[uv_idx[corner]]
            seen[v] = True

    save_path = os.path.join(data_path, save_name)
    torch.save(vertex_uv, save_path)
    print(f"Saved vertex UVs to {save_path}")


if __name__ == "__main__":
    parser = create_parser("Extract vertex UVs from OBJ file")
    parser.add_argument("--input", "-i", required=True,
                        help="Input OBJ filename (e.g., furless_reshaped_uv.obj)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output PT filename (e.g., furless_reshaped_uv.pt)")
    args = parser.parse_args()

    main(args.animal, args.root_path, args.input, args.output)
