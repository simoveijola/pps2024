#!/bin/bash -l
#SBATCH --account courses
#SBATCH --partition courses
#### Standard parameters
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=4000
#### For a small MPI job:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
####End of large MPI job.
#SBATCH --gpus-per-node=2
##SBATCH --output=prog.out

##module purge
####Needed to use CUDA-aware MPI
##export OMPI_MCA_opal_warn_on_missing_libcuda=0
##module use /share/apps/scibuilder-spack/aalto-centos7-dev/2023-01/lmod/linux-centos7-x86_64/Core
##module load gcc/11.3.0 cmake/3.26.3 openmpi/4.1.5

module load gcc cuda cmake openmpi

time srun ../build/quicksort-distributed-gpu

