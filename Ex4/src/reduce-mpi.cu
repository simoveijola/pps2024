#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"


// some global variables
constexpr int countPerBlock = 2048;
constexpr int blocksPerStream = 56*16; 
constexpr int threadsPerBlock = 128*2;

template<unsigned int blockSize>
__device__ void warp_reduce(volatile int *data, int i) {
  if(blockSize >= 64) data[i] = max(data[i], data[i + 32]);
  if(blockSize >= 32) data[i] = max(data[i], data[i + 16]);
  if(blockSize >= 16) data[i] = max(data[i], data[i + 8]);
  if(blockSize >= 8) data[i] = max(data[i], data[i + 4]);
  if(blockSize >= 4) data[i] = max(data[i], data[i + 2]);
  if(blockSize >= 2) data[i] = max(data[i], data[i + 1]);
}
// The device kernel for finding the maximum integer in `arr`
// 'count': number of consecutive element to handle in a block
// 'n': global size of the array
// 'out': array for result storage
template<unsigned int blockSize>
__global__ void reduce_kernel(const int* arr, const size_t n, const int streamId, int* out)
{
  // EXERCISE 4: Your code here
  // local data storage of size blockDim.x
  __shared__ int aux[threadsPerBlock];
  int tid = threadIdx.x, bid = blockIdx.x + streamId*blocksPerStream, bdim = blockDim.x;
  int i = tid + countPerBlock*bid; // one block takes care of count sized subarray
  // initialize aux tid
  aux[tid] = 0;
  // last element to handle for this block is 'last-1'
  int last = min(countPerBlock*(bid+1), (int)n);
  while(i < last) {
    aux[tid] = max(aux[tid], arr[i]);
    i+=bdim;
  }
  __syncthreads();

  if(blockSize >= 512) {
    if(tid < 256) aux[tid] = max(aux[tid], aux[tid + 256]);
    __syncthreads();
  }
  if(blockSize >= 256) {
    if(tid < 128) aux[tid] = max(aux[tid], aux[tid + 128]);
    __syncthreads();
  }
  if(blockSize >= 128) {
    if(tid < 64) aux[tid] = max(aux[tid], aux[tid + 64]);
    __syncthreads();
  }

  // reduce inside the warp
  if(tid < 32) warp_reduce<threadsPerBlock>(aux, tid);
  // store the maximum of the block to the corresponding place in 'out'
  if(tid == 0) {
    out[bid] = aux[0];
  }
}

int
reduce(const int* base_arr, const size_t base_count)
{
  // Do not modify: helper code for distributing `base_arr` to multiple GPUs
  int nprocs, pid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);

  const size_t count = base_count / nprocs;
  ERRCHK(count * nprocs == base_count);

  int* arr = (int*)malloc(count * sizeof(arr[0]));
  MPI_Scatter(base_arr, count, MPI_INT, arr, count, MPI_INT, 0, MPI_COMM_WORLD);

  if(pid == 0) {
    int results[nprocs];
  }

  // EXERCISE 4: Your code here
  // Input:
  //  arr    - An array of integers assigned to the current rank (process)
  //  count  - The number of integers in an array per rank (process). You can
  //           assume that count is divisible by the number
  //           of ranks, i.e. count * nprocs = base_count
  //
  // Return: the maximum integer in base_arr. The correct result needs to be
  //         returned only by rank 0. The return values of other ranks are
  //         not checked.
  //
  // Task: the integer array (base_arr) is now distributed across the ranks.
  //       Each rank holds a subset of the array stored in `arr` residing in
  //       host memory. Your task is to first map the rank (i.e. process id or
  //       pid) to a device id. Then, you should allocate memory on that device,
  //       transfer the data, and apply the reduction with the selected device,
  //       in a similar fashion as in task 2. Finally, you should combine the
  //       results of each process/device any way you like (for example using
  //       MPI_Gather or MPI_Reduce) and return the result.

  int num_devices;
  ERRCHK_CUDA(cudaGetDeviceCount(&num_devices));
  size_t dcount = count / num_devices;
  ERRCHK(dcount * num_devices ==
         count); // Require count divisible with num_devices

  int* darr[num_devices];
  const size_t bytes = dcount * sizeof(darr[0][0]);
  for (int i = 0; i < num_devices; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&darr[i], bytes);
  }

  // here on the variables define sizes for one device only
  const int blockSize = threadsPerBlock < dcount ? threadsPerBlock : dcount;
  const int blockCount = (dcount+countPerBlock-1)/countPerBlock;
  // now # streams is multiplied with number of devices, otherwise the same as previously
  const int nstreamPerDev = (blockCount+blocksPerStream-1)/blocksPerStream;

  // init streams and device/host auxiliary arrays
  cudaStream_t streams[num_devices*nstreamPerDev];
  int *out_d[num_devices];
  int *out[num_devices];

  for(int j = 0; j < num_devices; ++j) {
    cudaSetDevice(j);
    cudaMalloc((void**)&out_d[j], blockCount*sizeof(int));
    cudaMallocHost((void**)&out[j], blockCount*sizeof(int));
    for(int i = 0; i < nstreamPerDev; ++i) {
      cudaStreamCreate(&streams[i*num_devices + j]);	
    }
  }

  for(int j = 0; j < num_devices; ++j) { // then for each iteration start the corresponding stage on each device
    cudaSetDevice(j);
    for(int i = 0; i < nstreamPerDev; ++i) { // first loop through streams per device
      // transfer one stream amount of data
      int offset = i*blocksPerStream*countPerBlock;
      int count = i != nstreamPerDev-1 ? blocksPerStream*countPerBlock : dcount-offset;
      int blocks = i != nstreamPerDev-1 ? blocksPerStream : blockCount-(nstreamPerDev-1)*blocksPerStream;
      // transfer data to device and add padding when needed
      cudaMemcpyAsync(darr[j]+offset, &arr[j * dcount + offset], count*sizeof(int), 
                      cudaMemcpyHostToDevice, streams[i*num_devices + j]);
      if(i == nstreamPerDev-1) {/*pad the end*/
        cudaMemsetAsync(darr[j] + dcount, INT_MIN, 
                        threadsPerBlock, streams[i*num_devices + j]);
      }
      // reduce
      reduce_kernel<threadsPerBlock><<<blocks, threadsPerBlock, threadsPerBlock, streams[i*num_devices + j]>>>(darr[j], dcount, i, out_d[j]);
      // transfer data to CPU
      cudaMemcpyAsync(out[j]+i*blocksPerStream, out_d[j]+i*blocksPerStream, blocks*sizeof(int), 
                      cudaMemcpyDeviceToHost, streams[i*num_devices + j]);
      cudaEventDestroy(startker);
      cudaEventDestroy(stopker);
    }
  }

  for(int j = 0; j < num_devices; ++j) {
    cudaSetDevice(j);
    cudaDeviceSynchronize();
  } 
  // reduce the blocks sequentially with CPU as this is only a tiny fraction of calculations
  // no reason to invoke GPU kernels.
  int maximum = INT_MIN;
  for(int j = 0; j < num_devices; ++j) {
    cudaSetDevice(j);
    //cudaDeviceSynchronize();
    for(int i = 0; i < blockCount; ++i) {
      maximum = maximum > out[j][i] ? maximum : out[j][i]; 
    }
    cudaFree(darr[j]);
    cudaFree(out_d[j]);
    cudaFreeHost(out[j]);
    for(int i = 0; i < nstreamPerDev; ++i) {
      cudaStreamDestroy(streams[i*num_devices + j]);
    }
  }

  MPI_Gather(&maximum, 1, MPI_INT, results, nprocs, MPI_INT, 0, MPI_COMM_WORLD);

  if(pid == 0) {
    for(int i = 0; i < nprocs; ++i) {
      maximum = maximum > results[i] ? maximum : results[i]; 
    }
  }

  return maximum;

}
