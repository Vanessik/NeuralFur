#!/bin/bash
uname -a
#date
#env
date
CS_PATH='/mnt/data/humanparsing/LIP'
LR=3.0e-3
WD=5e-4
BS=8
GPU_IDS=0,1,2,3
RESTORE_FROM='./dataset/resnet101-imagenet.pth'
INPUT_SIZE='473,473'  
SNAPSHOT_DIR='./snapshots'
DATASET='train'
NUM_CLASSES=20 
EPOCHS=150

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

    python -m torch.distributed.launch --nproc_per_node=4 --nnode=1 \
    --node_rank=0 --master_addr=222.32.33.224 --master_port 29500 train.py \
       --data-dir ${CS_PATH} \
       --random-mirror\
       --random-scale\
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --epochs ${EPOCHS}

# python evaluate.py
