# Prerequisites:
#
# 1. Install CUDA 11.8
#    Follow intructions on https://developer.nvidia.com/cuda-11-8-0-download-archive
#    Make sure that
#      -   PATH includes <CUDA_DIR>/bin
#      -   LD_LIBRARY_PATH includes <CUDA_DIR>/lib64
#    If needed, restart bash environment

#    The environment was tested only with this CUDA version

# 2. Install Blender 3.6 to create strand visualizations
#    Follow instructions on https://www.blender.org/download/lts/3-6
#

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

# Save parent dir
PROJECT_DIR=$PWD

# Pull external libraries (NeuralHaircut is already included in ext/)
cd $PROJECT_DIR/ext && git clone https://github.com/facebookresearch/pytorch3d
cd $PROJECT_DIR/ext/pytorch3d && git checkout 2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7
cd $PROJECT_DIR/ext && git clone https://github.com/camenduru/simple-knn
cd $PROJECT_DIR/ext/diff_gaussian_rasterization_hair/third_party && git clone https://github.com/g-truc/glm
cd $PROJECT_DIR/ext/diff_gaussian_rasterization_hair/third_party/glm && git checkout 5c46b9c07008ae65cb81ab79cd677ecc1934b903


# Install environment
cd $PROJECT_DIR && conda env create -f environment.yml
conda activate gaussian_splatting_hair

# Download Neural Haircut files
cd $PROJECT_DIR/ext/NeuralHaircut
gdown --folder https://drive.google.com/drive/folders/1TCdJ0CKR3Q6LviovndOkJaKm8S1T9F_8
cd $PROJECT_DIR/ext/NeuralHaircut/pretrained_models/diffusion_prior # downloads updated diffusion prior
gdown 1_9EOUXHayKiGH5nkrayncln3d6m1uV7f