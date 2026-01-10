#!/bin/bash

# Example script for running ImageNet experiments with the new abstracted interface
# This replaces the old main_imagenet1k.py calls

IMAGENET_DIR="/path/to/ImageNet"  # Update this path
EXP_DIR="./experiments/imagenet"

mkdir -p $EXP_DIR

# Example 1: ResNet-18 with pretrained weights and 5 single neuron permutations
python ../src/main.py \
    --dataset=imagenet \
    --data_dir="$IMAGENET_DIR" \
    --batch_size=128 \
    --experiments_path="$EXP_DIR/resnet18_num_alphas=21_num_perms=5_seed=1001.pth" \
    --num_alphas=21 \
    --num_batches=10 \
    --num_perms=5 \
    --num_workers=16 \
    --random_state=1001 \
    --model=resnet18 \
    --pretrained

# Example 2: ResNet-18 with pretrained weights and full layer permutation
python ../src/main.py \
    --dataset=imagenet \
    --data_dir="$IMAGENET_DIR" \
    --batch_size=128 \
    --experiments_path="$EXP_DIR/resnet18_num_alphas=21_full_perm=1_seed=1001.pth" \
    --num_alphas=21 \
    --num_batches=10 \
    --num_perms=1 \
    --num_workers=16 \
    --random_state=1001 \
    --model=resnet18 \
    --pretrained \
    --full_perm

# Example 3: ResNet-50 with pretrained weights and 50 single neuron permutations
python ../src/main.py \
    --dataset=imagenet \
    --data_dir="$IMAGENET_DIR" \
    --batch_size=128 \
    --experiments_path="$EXP_DIR/resnet50_num_alphas=21_num_perms=50_seed=1001.pth" \
    --num_alphas=21 \
    --num_batches=10 \
    --num_perms=50 \
    --num_workers=16 \
    --random_state=1001 \
    --model=resnet50 \
    --pretrained
