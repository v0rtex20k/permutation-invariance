#!/bin/bash
#SBATCH --array=0-11%10
#SBATCH --error=/cluster/home/varsen01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --ntasks=16
#SBATCH --output=/cluster/home/varsen01/slurmlog/out/log_%j.out
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

source ~/miniconda3/etc/profile.d/conda.sh 
source ~/.bashrc
conda activate l3d_2024f_cuda12_1
CIFAR_DIR="/cluster/home/varsen01/permutation-invariance/data/cifar10"
EXP_DIR="/cluster/home/varsen01/permutation-invariance/experiments/cifar10"

mkdir -p "$CIFAR_DIR"
mkdir -p "$EXP_DIR"

# Define an array of commands mirroring the ImageNet sweep but targeting CIFAR-10
experiments=(
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=5_seed=1001.pth" --num_alphas=21 --num_batches=10 --num_perms=5 --num_workers=8 --random_state=1001 --model=resnet18'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=5_seed=2001.pth" --num_alphas=21 --num_batches=10 --num_perms=5 --num_workers=8 --random_state=2001 --model=resnet18'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=50_seed=1001.pth" --num_alphas=21 --num_batches=10 --num_perms=50 --num_workers=8 --random_state=1001 --model=resnet18'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=50_seed=2001.pth" --num_alphas=21 --num_batches=10 --num_perms=50 --num_workers=8 --random_state=2001 --model=resnet18'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=500_seed=1001.pth" --num_alphas=21 --num_batches=10 --num_perms=500 --num_workers=8 --random_state=1001 --model=resnet18'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_num_perms=500_seed=2001.pth" --num_alphas=21 --num_batches=10 --num_perms=500 --num_workers=8 --random_state=2001 --model=resnet18'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=1_seed=1001.pth" --num_alphas=21 --num_batches=10 --num_perms=1 --num_workers=8 --random_state=1001 --model=resnet18 --full_perm'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=1_seed=2001.pth" --num_alphas=21 --num_batches=10 --num_perms=1 --num_workers=8 --random_state=2001 --model=resnet18 --full_perm'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=5_seed=1001.pth" --num_alphas=21 --num_batches=10 --num_perms=5 --num_workers=8 --random_state=1001 --model=resnet18 --full_perm'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=5_seed=2001.pth" --num_alphas=21 --num_batches=10 --num_perms=5 --num_workers=8 --random_state=2001 --model=resnet18 --full_perm'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=10_seed=1001.pth" --num_alphas=21 --num_batches=10 --num_perms=10 --num_workers=8 --random_state=1001 --model=resnet18 --full_perm'
    'python ../src/main.py --dataset=cifar10 --data_dir="'$CIFAR_DIR'" --batch_size=128 --experiments_path="'$EXP_DIR'/resnet18_num_alphas=21_full_perm=10_seed=2001.pth" --num_alphas=21 --num_batches=10 --num_perms=10 --num_workers=8 --random_state=2001 --model=resnet18 --full_perm'
)

# Run the command indexed by the SLURM array task ID
eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
