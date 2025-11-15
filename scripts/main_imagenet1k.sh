#!/bin/bash
#SBATCH --array=0-14%10
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=64g
#SBATCH --ntasks=16
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=1_random_state=1001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=1 --num_workers=0 --random_state=1001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=1_random_state=2001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=1 --num_workers=0 --random_state=2001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=1_random_state=3001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=1 --num_workers=0 --random_state=3001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=10_random_state=1001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=10 --num_workers=0 --random_state=1001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=10_random_state=2001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=10 --num_workers=0 --random_state=2001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=10_random_state=3001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=10 --num_workers=0 --random_state=3001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=100_random_state=1001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=100 --num_workers=0 --random_state=1001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=100_random_state=2001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=100 --num_workers=0 --random_state=2001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=100_random_state=3001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=100 --num_workers=0 --random_state=3001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=1000_random_state=1001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=1000 --num_workers=0 --random_state=1001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=1000_random_state=2001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=1000 --num_workers=0 --random_state=2001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=1000_random_state=3001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=1000 --num_workers=0 --random_state=3001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=10000_random_state=1001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=10000 --num_workers=0 --random_state=1001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=10000_random_state=2001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=10000 --num_workers=0 --random_state=2001'
    'python ../src/main_imagenet1k.py --batch_size=128 --experiments_path="/cluster/tufts/hugheslab/eharve06/permutation-invariance/experiments/imagenet1k/batch_size=128_num_alphas=11_num_batches=10_num_perms=10000_random_state=3001.pth" --imagenet1k_dir="/cluster/tufts/hugheslab/datasets/ImageNet" --num_alphas=11 --num_batches=10 --num_perms=10000 --num_workers=0 --random_state=3001'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
