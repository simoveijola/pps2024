#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH -t 00:00:59

#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2

#SBATCH --account courses
#SBATCH --partition courses

module load gcc cuda cmake openmpi

srun ../build/reduce-mpi reduce-mpi.result
