#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --job-name=NASforClassification
#SBATCH --output=automnli.log
#SBATCH --time=48:00:00

module load cuda/10.1

prefix=${HOME}/NASforSentenceEmbeddingHeads
srun ${prefix}/slurm_example.sh configmnli.json ${prefix}
