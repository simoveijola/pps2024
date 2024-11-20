#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "errchk.h"
#include "reduce.cuh"
#include <cuda_runtime.h>

// some global variables
constexpr int countPerBlock = 2048;
constexpr int blocksPerStream = 28;
constexpr int threadsPerBlock = 128;

Template<int blockSize>
__device__ void
warp_reduce(volatile int* data, int i) {
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
Template<int blockSize>
__global__ void
reduce_kernel(const int* arr, const size_t n, const int streamId, int* out)
{
  // EXERCISE 4: Your code here
  // local data storage of size blockDim.x
  __shared__ int aux[threadsPerBlock];
  int tid = threadIdx.x, bid = blockIdx.x + streamId*blocksPerStream, bdim = blockDim.x;
  int i = tid + countPerBlock*bid; // one block takes care of count sized subarray
  // initialize aux tid
  aux[tid] = 0;
  // last element to handle for this block is 'last-1'
  int last = min(countPerBlock*(bid+1), n);
  while(i < last) {
    aux[tid] = max(aux[tid], arr[i]);
    i+=bdim;
  }

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
  if(tid < 32) warp_reduce(aux, tid);
  // store the maximum of the block to the corresponding place in 'out'
  if(tid == 0) out[bid] = aux[0];
}

// The host function for calling the reduction kernel
int
reduce(const int* arr, const size_t initial_count)
{
  // EXERCISE 4: Your code here
  // Input:
  //  arr           - An array of integers
  //  initial_count - The number of integers in `arr`
  //
  // Return: the maximum integer in `arr`
  //
  // Task: allocate memory on the GPU, transfer `arr` into
  // the allocated space, and apply `reduce_kernel` iteratively on it
  // to find the maximum integer. Finally, move the result back to host
  // memory and return it.
  
  // The nodes we use have P100 GPU so that 28 blocks of 128 threads
  // can run concurrently, thus I have limited the block count of
  // in one stream to be little over 28 (56) blocks, and I 
  // perform data transfer at the same time to hide data transfer latency

  // this current solution is not optimal for small arrays but should be 
  // efficient division of work for large arrays

  // init parallelism variables
  const int blockSize = min(threadsPerBlock, initial_count);
  const int blockCount = (initial_count+countPerBlock-1)/countPerBlock;
  const int nstream = (blockCount+blocksPerStream-1)/blocksPerStream;

  // init streams
  cudaStream_t streams[nstream];
  for(int i = 0; i < nstream; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  int *arr_d, *out_d;
  // allocate device arrays, pad the end to avoid memory access errors
  cudaMalloc((void**)arr_d, (initial_count+threadsPerBlock)*sizeof(int)); 
  cudaMalloc((void**)out_d, blockCount*sizeof(int));

  int *out;
  cudaMallocHost((void**)&out, blockCount*sizeof(int));

  for(int i = 0; i < nstream; ++i) {
    // transfer one stream amount of data
    int offset = i*blocksPerStream*countPerBlock;
    int count = min((i+1)*blocksPerStream*countPerBlock, initial_count-offset);
    int blocks = i != nstream-1 ? blocksPerStream : blockCount-(nstream-1)*blocksPerStream;
    // transfer data to device and add padding when needed
    cudaMemcpyAsync(arr_d+offset, arr+offset, count*sizeof(int), 
                    cudaMemcpyHostToDevice, streams[i]);
    if(i == nstream-1) {/*pad the end*/
      cudaMemsetAsync(arr_d + initial_count, INT_MIN, 
                      threadsPerBlock, streams[i]);
    }
    // reduce
    reduce_kernel< threadsPerBlock ><<<blocks, threadsPerBlock, threadsPerBlock, streams[i]>>>(arr_d, initial_count, i, out_d);
    // transfer data to CPU
    cudaMemcpyAsync(out+i*blocksPerStream, out_d+i*blocksPerStream, blocks*sizeof(int), 
                    cudaMemcpyDeviceToHost, streams[i]);
  }

  cudaDeviceSynchronize();
  // reduce the blocks sequentially with CPU as this is only a tiny fraction of calculations
  // no reason to invoke GPU kernels.
  int maximum = INT_MIN;
  for(int i = 0; i < blockCount; ++i) {
    maximum = max(maximum, out[i]); 
  }

  cudaFree(arr_d);
  cudaFree(out_d);
  cudaFreeHost(out);

  for(int i = 0; i < nstream; ++i) {
    cudaStreamDestroy(streams[i]);
  }

  return max;
}

