# NeuralFur

Fur reconstruction from multi-view images using 3D Gaussian Splatting.

## Prerequisites

- CUDA 11.8 ([download](https://developer.nvidia.com/cuda-11-8-0-download-archive))
- Conda
- Blender 3.6 (optional, for strand visualization) ([download](https://www.blender.org/download/lts/3-6))

Make sure `PATH` includes `<CUDA_DIR>/bin` and `LD_LIBRARY_PATH` includes `<CUDA_DIR>/lib64`.

## Installation

```bash
cd submodules/GaussianHaircut
bash install.sh
```

This will clone external dependencies (`pytorch3d`, `simple-knn`, `glm`), create a conda environment, and install all required packages. NeuralHaircut is already included in `ext/`.

## Data

Download the preprocessed data from [Google Drive (link)](https://drive.google.com/uc?id=1rIxvQKXVaMZ6Xzx7MAmHfucnbezLSbZt) and place it into your desired location, e.g.:

```
/path/to/data/panda_processed_GH2/walk/
```
```
pip install gdown
gdown https://drive.google.com/uc?id=1rIxvQKXVaMZ6Xzx7MAmHfucnbezLSbZt
unzip *.zip
```

The data directory should contain:
- `sdf_grid.npy`, `min_bound.npy`, `max_bound.npy` -- SDF volume for penetration loss
- `neus_lr.obj` -- reconstructed mesh (used for chamfer loss)
- `furless.obj` -- furless body mesh (used for strand initialization)
- `tan_furless.npy` -- precomputed tangent directions
- `annotations_furless_reshaped2.json` -- body part annotations (controls per-region fur length and gravity)
- `eyes.ply` -- eye landmarks (used for metric scale estimation)
- `images/` -- input images
- `masks/` -- segmentation masks
- `orientations/` -- orientation maps

## Configuration

Edit `submodules/GaussianHaircut/simple_run_panda.sh` and set your paths:

```bash
CUDA_HOME=/path/to/cuda-11.8       # your CUDA installation
DATA_PATH=/path/to/data/panda_processed_GH2/walk/
ENV_PATH=/path/to/conda/env        # conda environment name or path
```

`DATA_PATH` is passed to the training and export scripts via `--data_root`, which automatically replaces the `DATA_ROOT` placeholder in the YAML config. No need to edit the YAML config separately.

The YAML config (`src/arguments/metrical_panda_furless_15k_small.yaml`) contains animal-specific parameters like per-region fur length, gravity directions, and loss settings. These don't need to change between runs of the same animal.

## Running

```bash
cd submodules/GaussianHaircut
bash simple_run_panda.sh
```

The pipeline runs two stages:
1. **Fur strand reconstruction** -- optimizes strand geometry and appearance using orientation, mask, chamfer, SDF, shape consistency, and gravity losses
2. **Strand export** -- exports the reconstructed strands as a `.ply` point cloud

Results are saved to the `SAVE_EXP_PATH` directory. Use Tensorboard to monitor training progress.

## Project Structure

```
NeuralFur/
  submodules/
    GaussianHaircut/
      src/
        train_latent_fur.py          # main training script
        preprocessing/export_fur.py  # strand export
        scene/                       # scene, cameras, gaussian models
        gaussian_renderer/           # rendering
        utils/                       # losses, camera utils, etc.
        arguments/                   # config and CLI argument parsing
      ext/
        NeuralHaircut/               # strand prior, texture networks
        diff_gaussian_rasterization_hair/  # custom CUDA rasterizer
      simple_run_panda.sh            # example run script
      install.sh                     # installation script
```

## TODO

- [x] Code for processed scenes
- [x] [Reconstruction results](https://drive.google.com/drive/folders/1Gsqqr5wyE0ciJIbZ0EM87U_zkEVD1JgV)
- [ ] Preprocessing pipeline for new data (deadline: April 12, 2026)


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{sklyarova_kabadayi_2025neuralfur,
    title   = {NeuralFur: Animal Fur Reconstruction from Multi-view Images},
    author  = {Sklyarova, Vanessa and Kabadayi, Berna and Yiannakidis, Anastasios and Becherini, Giorgio and Black, Michael J. and Thies, Justus},
    journal = {arXiv},
    month   = {January},
    year    = {2026}
}
```