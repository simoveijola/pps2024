#!/bin/bash -l
#### Job submission script example to a CPU queue using hybrid programming
#### model
#### (Remember for Slurm #SBATCH, one # is active, two or
#### more ## is commented out)
####
#### If you have permanent Triton account (not made just for this
#### course), you can comment this out (comment = two `##`) and
#### maybe you will have shorter queue time.
#SBATCH --account courses
#SBATCH --partition courses-cpu
#### General resource parameters:
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=500M
#### Number of nodes, number of MPI processes is nodes x ntasks
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
####Specify output file, otherwise slurm-<jobid>.out generated
#SBATCH --cpus-per-task=4
#SBATCH --output=DE.out
####Special resource allocation, do not use unless instructed

export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge   # unload all current modules
module load gcc
module load openmpi

rm -f HEAT_RESTART.dat

time srun heat_mpi 4000 4000 1000

