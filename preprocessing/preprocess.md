# Preprocessing on Artemis Data [WIP]

Steps to process raw Artemis data from scratch. All scripts referenced below are in this directory.

### Prerequisites

- **Directional** -- to obtain tangent basis on the mesh
- **NeuS** -- to obtain full animal geometry (included as submodule)
- **SMAL** -- to obtain parametric animal model (SMALify)

### Download raw data

Download the Dynamic Furry Animals (DFA) dataset from [Artemis](https://github.com/HaiminLuo/Artemis):

```bash
animal="panda"
unzip ${animal}.zip -d /path/to/raw/Artemis/
```

## 1. Prepare images, silhouettes, and cameras

Converts raw Artemis captures (images, alpha masks, camera intrinsics/extrinsics) into the format expected by NeuS. Outputs images, silhouettes, and camera matrices (projection, intrinsic, extrinsic).

```bash
python preprocessing/preprocessing_all_data.py \
  --animal panda \
  --root_path /path/to/raw/Artemis \
  --save_path /path/to/save/Artemis
```

This creates `<save_path>/<animal>_processed/<motion_seq>/` containing:
- `images/` and `silhouette/` -- input views and alpha masks
- `cameras.npz` -- projection matrices (intrinsic x extrinsic x scale)
- `cameras_intr.npy`, `cameras_extr.npy` -- separate intrinsic and extrinsic matrices

## 2. Reconstruct surface with NeuS

Runs NeuS to obtain an SDF-based mesh reconstruction from the multi-view images.

```bash
cd submodules/NeuS
scene_name="panda"
data_dir="/path/to/Artemis/${scene_name}_processed"
python exp_runner.py --mode train --conf ./confs/wmask_artemis.conf --case walk --dataset artemis --scene ${scene_name} --data_dir ${data_dir}
python exp_runner.py --mode validate_mesh --conf ./confs/wmask_artemis.conf --case walk --dataset artemis --scene ${scene_name} --data_dir ${data_dir} --is_continue
```

The reconstructed mesh is saved to `submodules/NeuS/exp/<scene_name>/<case>/wmask/meshes/`.

## 3. Prepare data in GaussianHaircut format

Converts the NeuS-processed data into the directory structure expected by GaussianHaircut: padded image filenames, hair/body masks, projection matrices, and the NeuS mesh exported as OBJ.

```bash
python preprocessing/prepare_data_in_GH_format.py \
  --animal panda \
  --neus_root_path ./submodules/NeuS/exp \
  --root_path /path/to/Artemis \
  --save_root_path /path/to/Artemis
```

## 4. Calculate orientation maps

Computes hair/fur orientation maps using Gabor filters. These provide directional supervision for strand optimization.

```bash
cd submodules/GaussianHaircut/src/preprocessing
DATA_PATH="/path/to/Artemis/panda_processed_GH/walk"
python calc_orientation_maps.py \
  --img_path $DATA_PATH/images_2 \
  --mask_path $DATA_PATH/masks_2/body \
  --orient_dir $DATA_PATH/orientations_2/angles \
  --conf_dir $DATA_PATH/orientations_2/vars \
  --filtered_img_dir $DATA_PATH/orientations_2/filtered_imgs \
  --vis_img_dir $DATA_PATH/orientations_2/vis_imgs
```

## 5. Fit SMAL model

Fit a SMAL parametric animal model to the reconstructed mesh using [SMALify](https://github.com/silviazuffi/smalify). This produces a body model with semantic vertex groups.

## 6. Annotate and transfer SMAL body part annotations

### 6a. Annotate SMAL model in Blender [DONE and same across animals]

Open `annotate_smal.blend` in Blender, paint vertex groups for body parts (legs, belly, tail, ears, etc.), and export them as JSON by running `./preprocessing/save_annotations_fur_blender.py` from within Blender.

### 6b. Transfer annotations to target mesh

Transfers vertex group annotations from the fitted SMAL model to the target mesh using nearest-neighbor matching. These annotations define per-region fur length and gravity directions.

```bash
python preprocessing/transfer_smal_to_neus.py \
  --animal panda \
  --root_path /path/to/Artemis \
  --annotation_json ./data/part_annotations_SMAL.json \
  --input_mesh_path furless_reshaped.obj
```

```bash
python preprocessing/check_fur_length_and_blender_annotations.py
```

## 7. Annotate per-part fur properties with ChatGPT

For each new animal, we use ChatGPT to obtain three per-part annotations by sending two reference images (frontal and side views):

1. **Fur length** (`mapping_length`) -- fur length in cm per body part
2. **Effective fur thickness** (`effective_fur_thickness_cm`) -- how much to shrink the mesh inward to get the furless body (used in step 8)
3. **Fur growing direction** (`mapping_gravity`) -- a 3D direction vector per body part indicating how fur grows/hangs

Additionally, ask ChatGPT for the **distance between the eyeballs** in cm (`eye_dists_VQA`), used for metric scale estimation.

See [prompts.md](prompts.md) for the exact prompts to use. Paste the results into:
- **YAML config** (`submodules/GaussianHaircut/src/arguments/metrical_panda_furless_15k_small.yaml`): fur length (`mapping_length`), fur growing direction (`mapping_gravity`), and eye distance (`eye_dists_VQA`)
- **`src/animal_config.py`**: effective fur thickness (`effective_fur_thickness_cm`, used in step 8 for mesh shrinkage)

See `src/animal_config.py` for reference values across different animals.

## 8. Extract furless body mesh

Shrinks the NeuS mesh inward along vertex normals based on per-part fur thickness to obtain the furless (skin) body mesh. The shrinkage is defined per body part using `effective_fur_thickness_cm` values and scaled to metric space using the eye distance. The shrinkage field is Laplacian-smoothed across the surface for continuity, then the shrunken mesh is converted to SDF, reconstructed via marching cubes, and cleaned (degenerate faces removed, largest component kept).

The key steps are:

1. Load the NeuS mesh and body part annotations
2. Build a per-vertex shrinkage field from `effective_fur_thickness_cm` per body part
3. Laplacian-smooth the shrinkage field (100 iterations)
4. Displace vertices inward along normals by the smoothed shrinkage
5. Convert to SDF, run marching cubes at resolution 256, and clean the result
6. Export as `furless_reshaped.obj`

## 9. Compute tangent field with Directional

Clean the NeuS mesh to remove some artifacts:

```bash
python fix_mesh_before_directional.py
```

then use the [Directional](https://github.com/avaxman/Directional) library to compute a tangent field on the mesh surface. The output field is then refined using parallel transport:

```bash
python preprocessing/compute_tangent_basis.py \
  --animal panda \
  --root_path /path/to/Artemis \
  --save_root_path /path/to/Artemis \
  --mode furless_reshaped 
```
To visualize obtained field add also `--visualize_tan`.


## 10. Create eye landmarks and measure eye distance

Extract eye keypoints from the Artemis dataset or from SMAL model and save as `eyes.ply`.


```bash
python preprocessing/extract_eyes.py \
  --animal panda \
  --root_path /path/to/Artemis
```


## 11. Losses

### 11a Compute SDF volume

Computes a signed distance field on a regular grid from the furless body mesh. Used as a penetration loss during fur optimization.

```bash
python preprocessing/compute_sdf.py \
  --animal panda \
  --root_path /path/to/Artemis \
  --save_root_path /path/to/Artemis \
  --mode furless_reshaped \
  --grid_size 32
```

### 11b Save bald mask

```bash
python preprocessing/visibility_faces.py
```