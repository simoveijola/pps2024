#!/bin/bash -l
#SBATCH --output=advec.out
##SBATCH --account=courses
##SBATCH --partition=courses-cpu
#SBATCH --mem-per-cpu=4000
#SBATCH --time=00:05:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1

module load gcc openmpi
mpicc -o ADVC advec_correctness.c -lm

nprocx=2
nprocy=2
domain_nx=$((nprocx*10))
domain_ny=$((nprocy*10))
iterations=10
nnodes=$((nprocx*nprocy))

time srun ADVC -- $nprocx $nprocy $domain_nx $domain_ny $iterations
