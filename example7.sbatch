#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --job-name=NAS-mrpcemb
#SBATCH --output=automrpcembshift.log
#SBATCH --time=48:00:00

module load cuda/10.1

prefix=${HOME}/NAS-CatastrophicForgetting
srun ${prefix}/slurm_example.sh configembshiftmrpc.json ${prefix}
