#!/bin/bash

# CS_PATH='./dataset/LIP'
CS_PATH='/mnt/data/humanparsing/LIP'
BS=1
GPU_IDS='1'
INPUT_SIZE='473,473'
SNAPSHOT_FROM='./snapshots/LIP_epoch_149.pth'
DATASET='val'
NUM_CLASSES=20

CUDA_VISIBLE_DEVICES=1 python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}
