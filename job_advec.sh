#!/bin/bash -l
#SBATCH --output=advec.out
#SBATCH --account=courses
#SBATCH --partition=courses-cpu
#SBATCH --mem-per-cpu=4000
#SBATCH --time=00:05:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1

module load gcc openmpi
mpicc -o ADV advec_wave_2D.c -lm

nprocx=2
nprocy=2
domain_nx=$((nprocx*5))
domain_ny=$((nprocy*5))
iterations=2
nnodes=$((nprocx*nprocy))

time srun ADV $nprocx $nprocy $domain_nx $domain_ny $iterations
