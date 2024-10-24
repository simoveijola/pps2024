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

# iterate with different sizes and iteration counts
for iter in 50
do
	time srun ADV -- $nprocx $nprocy $((3*nprocx)) $((3*nprocy)) $iter >> times_iter_${iter}_size_3.out
done
