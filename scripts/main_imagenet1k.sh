#!/bin/bash
#SBATCH --array=0-11%10
#SBATCH --error=/cluster/home/varsen01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --ntasks=16
#SBATCH --output=/cluster/home/varsen01/slurmlog/out/log_%j.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh 
source ~/.bashrc
conda activate l3d_2024f_cuda12_1
IMAGENET_DIR="/cluster/tufts/hugheslab/datasets/ImageNet"
EXP_DIR="/cluster/home/varsen01/permutation-invariance/experiments/imagenet1k"

# Define an array of commands
experiments=(
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=5_seed=1001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=5 --num_workers=0 --random_state=1001 --model=resnet18'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=5_seed=2001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=5 --num_workers=0 --random_state=2001 --model=resnet18'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=50_seed=1001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=50 --num_workers=0 --random_state=1001 --model=resnet18'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=50_seed=2001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=50 --num_workers=0 --random_state=2001 --model=resnet18'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=500_seed=1001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=500 --num_workers=0 --random_state=1001 --model=resnet18'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=500_seed=2001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=500 --num_workers=0 --random_state=2001 --model=resnet18'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=1_seed=1001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=1 --num_workers=0 --random_state=1001 --model=resnet18 --full_perm'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=1_seed=2001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=1 --num_workers=0 --random_state=2001 --model=resnet18 --full_perm'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=5_seed=1001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=5 --num_workers=0 --random_state=1001 --model=resnet18 --full_perm'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=5_seed=2001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=5 --num_workers=0 --random_state=2001 --model=resnet18 --full_perm'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=10_seed=1001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=10 --num_workers=0 --random_state=1001 --model=resnet18 --full_perm'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=10_seed=2001.pth" --imagenet1k_dir="'$IMAGENET_DIR'" --num_alphas=21 --num_batches=10 --num_perms=10 --num_workers=0 --random_state=2001 --model=resnet18 --full_perm'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
