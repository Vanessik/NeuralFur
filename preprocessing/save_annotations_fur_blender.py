import bpy
import json
import os

user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(user_desktop, "fur_annotations.json")

obj = bpy.context.object
mesh = obj.data
groups = obj.vertex_groups

annotations = {}

# Iterate through vertex groups
for group in groups:
    indices = [v.index for v in mesh.vertices if group.index in [g.group for g in v.groups]]
    annotations[group.name] = indices

# Save to JSON
with open(output_path, 'w') as f:
    json.dump(annotations, f, indent=4)

print(f"Saved annotation JSON to: {output_path}")
