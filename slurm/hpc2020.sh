#!/bin/bash --login
#SBATCH --job-name=aifs-fw-bw-o1280
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --qos=ng
#SBATCH --exclude=ac6-320
#SBATCH --time=0:20:00
#SBATCH -o %x-%j.out


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /ec/res4/hpcperm/naco/aifs/aifs-fw-bw/env.sh
srun python main.py -C 512