# Exercise 4 - Distributed Computing on GPUs

## Introduction

In this exercise, you will implement functions to find the maximum integer in an array distributed across multiple GPUs (devices).
Additionally you will extend the quicksort you have implemented in exercise 2. to work on multiple GPUs


The exercise consists of three subtasks related to reductions:

  1. Implement the functions `reduce_kernel` and `reduce` in `src/reduce-single.cu`. Your task here is to find the maximum integer in an array allocated on a single device and return the result with `reduce`.
  
  2. Extend your implementation to work on multiple devices on a single node in `src/reduce-multi.cu`. The functions to implement are `reduce_kernel` (you can re-use your solution from subtask (1) and `reduce`. The array is now distributed across multiple devices and each device should compute a single result. You can perform the final reduction step on the host by transferring the data with the device-to-host variant of `cudaMemcpy`. In your slurm command, you should allocate a single task (`--ntasks=1`) but multiple devices (`--gpus=n`).

  3. Extend your implementation to work on multiple devices on multiple nodes using MPI in `src/reduce-mpi.cu`. The instructions are otherwise the same as in subtask (2), however, in this exercise you should allocate one process per device (for example `--ntasks-per-node=2 --nodes=2 --gpus-per-node=2`) and ensure your implementation works on multiple nodes.


And one subtask related to quicksort:

  4. Extend your quicksort implementation to work on GPUs. Start first by implementing `quicksort` in `src/quicksort_gpu.cu`. Then extend the GPU implementation to multiple GPU by implementing `quicksort_distributed` in `quicksort_distributed_gpu.cu` (you are allowed to and highly encouraged to re-use your previous solutions in exercise 2. for this).   In this exercise you should allocate one process per device (for example `--ntasks-per-node=2 --nodes=2 --gpus-per-node=2`).


> Note: The performance of your implementation is not graded and to get full points, you only need to ensure your implementations give the correct results. However, you should use the hardware relatively efficiently: an implementation that uses a single CUDA thread in task 1, or a single device in tasks 2 or 3, will receive 0 points. Additionally try to use CUDA-aware MPI instead of first moving data to the CPU and then communicating in task 4.

## Returnables

The solutions should be returned in a single zip file named `<your student number>.zip`, for example `12345.zip`. The archive should contain at least `src/reduce-mpi.cu`, `src/reduce-multi.cu`, `src/reduce-single.cu`, `src/quicksort_gpu.c` and `src/quicksort_distributed_gpu.cu`. You create the archive with the command `zip <your student number>.zip -r src/` (note the `-r` flag: the archive must contain the `src` directory). 

## Getting started

Run the following commands to get started:

```Bash
module load gcc cuda cmake openmpi
cd pps-2024-exercises/exercise_sheets/sheet4
mkdir build && cd build
cmake .. && make -j
cd .. && mkdir yourrundir && cd yourrundir
sbatch ../job_scripts/run_XX.sh
```

Three binaries should appear corresponding to tasks 1, 2, and 3: `reduce-single`, `reduce-multi`, `reduce-mpi`.
Two binaries should appear corresponding to task 4: `quicksort-gpu` and `quicksort-distributed-gpu`
There are some batch scripts in `sheet4/job-scripts` to get you started.


## Hints

  * See `src/main.cu` for the CPU model solution used for grading. Your implementation should give the same output.

  * Only the interface function `int reduce(const int* arr, const size_t count)` declared in `src/reduce.cuh` is used for grading. You can add additional helper functions if needed.

  * Remember to use `cudaSetDevice` to select the correct device in task 2 and 3.

  * If your implementation is very slow, recall how GPUs differ from CPUs (lecture slides). Do not try to create a `for` loop inside the kernel that loops over all `n` elements. In addition, you should strive to avoid branch divergence and ensure that the workload is distributed evenly across the stream processors (CUDA cores).

  * For an introduction to parallel reductions and hints on more advanced optimization techniques, see [NVIDIA's slides on parallel reductions](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).

  * For doing a stable quicksort the prefix sum algorithm is helpful. For an introduction to parallel prescan and hints on optimization techniques see [NVIDIA’s article](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).

