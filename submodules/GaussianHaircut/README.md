# GaussianHaircut (NeuralFur fork)

Strand-based fur reconstruction using 3D Gaussian Splatting. This is a modified version of [GaussianHaircut](https://github.com/eth-ait/GaussianHaircut) adapted for animal fur reconstruction.

## Installation

### Prerequisites
- CUDA 11.8
- Conda

### Setup

```bash
bash install.sh
```

This will:
- Clone `pytorch3d`, `simple-knn`, and `glm` (for `diff_gaussian_rasterization_hair`) into `ext/`
- Create a conda environment and pip-install local packages: `pytorch3d`, `simple-knn`, `diff_gaussian_rasterization_hair`
- Download pretrained strand prior models for NeuralHaircut

## Usage

See the main [NeuralFur README](../../README.md) for full instructions on data preparation and configuration.

### Quick start

1. Set your paths in `simple_run_panda.sh` (`CUDA_HOME`, `DATA_PATH`, `ENV_PATH`)
2. Run:

```bash
bash simple_run_panda.sh
```

### Pipeline

The script runs two stages:

**Stage 1: Strand reconstruction** (`src/train_latent_fur.py`)

Optimizes fur strand geometry and appearance by jointly rendering body gaussians (frozen) and hair strands. Losses used:
- Orientation loss -- aligns rendered strand directions with GT orientation maps
- Mask loss -- matches rendered fur silhouette to GT masks
- Chamfer loss -- attracts strands toward the body surface
- SDF penetration loss -- penalizes strands going inside the body
- Shape consistency loss -- enforces similar curvature across strands
- Gravity loss -- encourages strands to follow per-region gravity directions

**Stage 2: Strand export** (`src/preprocessing/export_fur.py`)

Exports the optimized strands as a `.ply` point cloud.

## License

Based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). For terms, see LICENSE_3DGS. The rest is distributed under CC BY-NC-SA 4.0.

## Citation

```
@inproceedings{zakharov2024gh,
   title = {Human Hair Reconstruction with Strand-Aligned 3D Gaussians},
   author = {Zakharov, Egor and Sklyarova, Vanessa and Black, Michael J and Nam, Giljoo and Thies, Justus and Hilliges, Otmar},
   booktitle = {European Conference of Computer Vision (ECCV)},
   year = {2024}
}
```

## Acknowledgements

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Neural Haircut](https://github.com/SamsungLabs/NeuralHaircut): strand prior and hairstyle diffusion prior
- [HAAR](https://github.com/Vanessik/HAAR): hair upsampling
