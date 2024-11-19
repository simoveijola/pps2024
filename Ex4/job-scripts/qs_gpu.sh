#!/bin/bash -l
#SBATCH --account courses
#SBATCH --partition courses
#### Standard parameters
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=4000
#SBATCH --nodes=1         #Use one node
#SBATCH --ntasks=1        #One task
#SBATCH --gpus-per-node=1
##SBATCH --output=prog.out


module purge
module load gcc cmake openmpi cuda

srun ../build/quicksort-gpu


