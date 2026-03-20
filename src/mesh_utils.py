# src/mesh_utils.py

import torch
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix
from collections import defaultdict, deque

def safe_normalize(v, dim=-1, eps=1e-8):
    return v / (v.norm(dim=dim, keepdim=True) + eps)

def fix_normals_with_neighbors(mesh):
    """
    Fix degenerate face normals (zero-length or NaN) using vertex averaging.
    """
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    raw_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    degenerate_mask = (raw_normals.norm(dim=-1) < 1e-6) | torch.isnan(raw_normals).any(dim=-1)
    valid_mask = ~degenerate_mask

    face_normals = torch.zeros_like(raw_normals)
    face_normals[valid_mask] = safe_normalize(raw_normals[valid_mask])

    if degenerate_mask.any():
        print(f"Fixing {degenerate_mask.sum().item()} degenerate face normals...")
        num_verts = verts.shape[0]
        vert_normals = torch.zeros((num_verts, 3), device=verts.device)

        for i in range(3):
            vert_normals = vert_normals.index_add(0, faces[valid_mask, i], face_normals[valid_mask])
        vert_normals = safe_normalize(vert_normals)

        deg_faces = faces[degenerate_mask]
        repaired = safe_normalize(vert_normals[deg_faces].mean(dim=1))
        face_normals[degenerate_mask] = repaired

    return face_normals



def create_line_meshes(points, lines, radius=0.001, sections=6):
    """
    Create thin cylinder meshes for each line segment between points.
    points: (N, 3) numpy array
    lines: (M, 2) numpy array of indices into points
    """
    line_meshes = []
    for i0, i1 in lines:
        start = points[i0]
        end = points[i1]
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-8:
            continue
        direction /= length
        
        # Create cylinder along z-axis
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        
        # Align cylinder axis with direction vector
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-8:
            dot = np.dot(z_axis, direction)
            if dot < 0:
                cyl.apply_transform(rotation_matrix(np.pi, [1, 0, 0]))
        else:
            axis /= axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            cyl.apply_transform(rotation_matrix(angle, axis))
        
        # Move cylinder to start position
        cyl.apply_translation(start)
        line_meshes.append(cyl)
    
    if len(line_meshes) == 0:
        return None
    combined = trimesh.util.concatenate(line_meshes)
    return combined


def export_normals_visualization_with_cones(scalp_verts, scalp_faces, origin_n, filename="scalp_normals_with_cones.ply", scale=0.01):
    """
    Visualize per-face normals as arrows: cylinder + cone at the tip.
    """
    scalp_verts_np = scalp_verts.cpu().numpy()
    scalp_faces_np = scalp_faces.cpu().numpy()
    origin_n_np = origin_n.cpu().numpy()
    
    face_verts = scalp_verts_np[scalp_faces_np]  # (F,3,3)
    face_centers = face_verts.mean(axis=1)        # (F,3)
    
    # Parameters for arrow parts
    shaft_radius = scale * 0.005

    cone_height = scale * 0.2  # bigger cone
    cone_radius = shaft_radius * 4.0

    
    # Prepare arrays for cylinder line points and edges
    # Each normal has 2 points (start, end)
    line_points = np.zeros((scalp_faces_np.shape[0] * 2, 3), dtype=np.float64)
    lines = []
    
    for i in range(scalp_faces_np.shape[0]):
        base_idx = i * 2
        origin = face_centers[i]
        normal = origin_n_np[i]
        
        line_points[base_idx] = origin
        line_points[base_idx + 1] = origin + normal * scale
        
        lines.append([base_idx, base_idx + 1])
    
    # Create cylinder shafts for normals
    shaft_mesh = create_line_meshes(line_points, lines, radius=shaft_radius)
    
    # Create cones at the tip of each normal
    cones = []
    z_axis = np.array([0, 0, 1])
    for i in range(scalp_faces_np.shape[0]):
        tip = face_centers[i] + origin_n_np[i] * scale
        
        normal = origin_n_np[i]
        
        # Create cone aligned with z-axis
        cone = trimesh.creation.cone(radius=cone_radius, height=cone_height, sections=16)

        # Compute rotation from z-axis to normal
        axis = np.cross(z_axis, normal)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-8:
            dot = np.dot(z_axis, normal)
            if dot < 0:
                cone.apply_transform(rotation_matrix(np.pi, [1, 0, 0]))
        else:
            axis /= axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
            cone.apply_transform(rotation_matrix(angle, axis))

        # Move cone so base aligns exactly at the tip of the normal line
        cone.apply_translation(tip - normal * cone_height)
        
        cones.append(cone)
    
    if len(cones) > 0:
        cones_mesh = trimesh.util.concatenate(cones)
    else:
        cones_mesh = None
    
    # Create scalp mesh
    scalp_mesh = trimesh.Trimesh(vertices=scalp_verts_np, faces=scalp_faces_np, process=False)
    
    # Combine everything
    mesh_list = [scalp_mesh]
    if shaft_mesh is not None:
        mesh_list.append(shaft_mesh)
    if cones_mesh is not None:
        mesh_list.append(cones_mesh)
    
    scene = trimesh.Scene(mesh_list)
    scene.export(filename)
    print(f"Saved scalp + normals with cones visualization to {filename}")
    
    
def orient_face_tangents_parallel_transport(mesh, face_tangents, seed_face=0):
    """
    Orient face tangents consistently using parallel transport across shared edges.

    Parameters:
    - mesh: trimesh.Trimesh
    - face_tangents: (F, 3) numpy array
    - seed_face: index to start orientation

    Returns:
    - consistent_tangents: (F, 3) numpy array
    """
    n_faces = len(mesh.faces)
    consistent = face_tangents.copy()
    visited = np.zeros(n_faces, dtype=bool)
    adjacency = mesh.face_adjacency

    # Build face-to-neighbor map
    face_neighbors = defaultdict(list)
    for f0, f1 in adjacency:
        face_neighbors[f0].append(f1)
        face_neighbors[f1].append(f0)

    queue = deque([seed_face])
    visited[seed_face] = True
    flip_count = 0

    while queue:
        current = queue.popleft()
        for neighbor in face_neighbors[current]:
            if visited[neighbor]:
                continue
            # Get shared edge
            current_face = mesh.faces[current]
            neighbor_face = mesh.faces[neighbor]
            shared = np.intersect1d(current_face, neighbor_face)
            if len(shared) != 2:
                continue  # not a real shared edge

            # Edge direction
            v0, v1 = mesh.vertices[shared]
            edge_dir = v1 - v0
            edge_dir /= np.linalg.norm(edge_dir)

            # Project tangents onto the plane of current/neighbor face
            t0 = consistent[current]
            t1 = consistent[neighbor]
            if np.dot(t0, t1) < 0:
                consistent[neighbor] = -t1
                flip_count += 1

            visited[neighbor] = True
            queue.append(neighbor)

    print(f"[✓] Finished tangent orientation via transport. Flipped {flip_count} tangents.")
    return consistent, flip_count