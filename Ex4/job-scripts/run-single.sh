#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH -t 00:00:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --account courses
#SBATCH --partition courses

module load gcc cuda cmake openmpi

srun ../build/reduce-single reduce-single.result
