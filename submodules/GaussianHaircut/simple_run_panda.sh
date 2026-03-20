#!/bin/bash

# === CUDA setup (adjust to your system) ===
export CUDA_HOME=/is/software/nvidia/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export PATH=$PATH:$CUDA_HOME/bin

# === General settings ===
GPU="0"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
eval "$(conda shell.bash hook)"

# === Data settings ===
animal="panda"
motion_type="walk"
data_path="/fast//vsklyarova/Projects/Artemis/Artemis/"
DATA_PATH="$data_path/${animal}_processed_GH/${motion_type}/"
SAVE_EXP_PATH="../results"
ENV_PATH="/home/vsklyarova/miniconda3/envs/eccv_gaus_hair_copy"

# === Training iterations ===
n_iter_body_stage=30000
n_iter_fur_stage=20000

# === Resolution ===
scale_factor=1
resolution_val="1920 1080"

# === Loss weights ===
dorient=1000        # orientation loss
dmask=0.1           # mask loss
dchamfer=20         # chamfer (attraction to surface)
dsdf=1              # SDF penetration
dshape=0.01         # shape consistency
gravity_consist=1   # gravity direction
strand_scale=0.0025

# === Config ===
config_path="$PROJECT_DIR/src/arguments/metrical_panda_furless_15k_small.yaml"
EXP_NAME="final"

# Find a free port for the GUI server
while :; do
    PORT=$(( ( RANDOM % 64512 ) + 1024 ))
    if ! lsof -i:"$PORT" &>/dev/null; then
        break
    fi
done

# =============================================
# Stage 1: Fur strand reconstruction
# =============================================
conda deactivate && conda activate $ENV_PATH && cd $PROJECT_DIR/src

CUDA_VISIBLE_DEVICES="$GPU" python train_latent_fur.py \
    -s $DATA_PATH \
    -m "$DATA_PATH/3d_gaussian_splatting/stage1" -r 1 \
    --model_path_hair "$SAVE_EXP_PATH/$animal/strands_reconstruction/$EXP_NAME" \
    --pointcloud_path_head "$DATA_PATH/furless.obj" \
    --hair_conf_path "$config_path" \
    --data_root "$DATA_PATH" \
    --lambda_dmask $dmask \
    --lambda_dorient $dorient \
    --lambda_sdf $dsdf \
    --lambda_chamfer $dchamfer \
    --lambda_shape_consist $dshape \
    --lambda_gravity_consist $gravity_consist \
    --strand_scale $strand_scale \
    --iteration_data $n_iter_body_stage \
    --iterations $n_iter_fur_stage \
    --scale_factor $scale_factor \
    --resolution_val $resolution_val \
    --port "$PORT" \
    --binarize_masks \
    --mask_bald \
    --use_test_split

# =============================================
# Stage 2: Export reconstructed strands
# =============================================
save_path="$SAVE_EXP_PATH/$animal/strands_reconstruction/$EXP_NAME"
conda deactivate && conda activate $ENV_PATH && cd $PROJECT_DIR/src

CUDA_VISIBLE_DEVICES="$GPU" python preprocessing/export_fur.py \
    --data_dir $DATA_PATH \
    --hair_conf_path "$config_path" \
    --data_root "$DATA_PATH" \
    --model_name $save_path \
    --iter $n_iter_fur_stage \
    --n_strands 100000 \
    --save_pts_per_strand 100
