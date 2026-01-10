#!/bin/bash

# Example script for running CIFAR-10 experiments
# CIFAR-10 will be automatically downloaded to the specified data directory

CIFAR_DIR="./data/cifar10"
EXP_DIR="./experiments/cifar10"

mkdir -p $EXP_DIR

# Example 1: ResNet-18 with 5 single neuron permutations
python ../src/main.py \
    --dataset=cifar10 \
    --data_dir="$CIFAR_DIR" \
    --batch_size=128 \
    --experiments_path="$EXP_DIR/resnet18_num_alphas=21_num_perms=5_seed=1001.pth" \
    --num_alphas=21 \
    --num_batches=10 \
    --num_perms=5 \
    --num_workers=4 \
    --random_state=1001 \
    --model=resnet18

# Example 2: ResNet-18 with full layer permutation
python ../src/main.py \
    --dataset=cifar10 \
    --data_dir="$CIFAR_DIR" \
    --batch_size=128 \
    --experiments_path="$EXP_DIR/resnet18_num_alphas=21_full_perm=1_seed=1001.pth" \
    --num_alphas=21 \
    --num_batches=10 \
    --num_perms=1 \
    --num_workers=4 \
    --random_state=1001 \
    --model=resnet18 \
    --full_perm

# Example 3: ResNet-50 with 50 single neuron permutations
python ../src/main.py \
    --dataset=cifar10 \
    --data_dir="$CIFAR_DIR" \
    --batch_size=128 \
    --experiments_path="$EXP_DIR/resnet50_num_alphas=21_num_perms=50_seed=1001.pth" \
    --num_alphas=21 \
    --num_batches=10 \
    --num_perms=50 \
    --num_workers=4 \
    --random_state=1001 \
    --model=resnet50
