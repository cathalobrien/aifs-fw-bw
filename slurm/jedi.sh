#!/bin/bash --login
#SBATCH --job-name=aifs-fw-bw-o1280
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=0:10:00
#SBATCH --account=jureap5
#SBATCH --hint=nomultithread
#SBATCH -o %x-%j.out


#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /ec/res4/hpcperm/naco/aifs/aifs-fw-bw/env.sh
srun python main.py -C 512 -r o1280 --slurm -c "edge"
