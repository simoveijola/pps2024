#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

// some global variables
constexpr int countPerBlock = 2048;
constexpr int blocksPerStream = 28*8;
constexpr int threadsPerBlock = 128;

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

// The host function for calling the reduction kernel
int
reduce(const int* arr, const size_t count)
{
  // Do not modify: helper code for distributing `arr` to multiple GPUs
  int num_devices;
  ERRCHK_CUDA(cudaGetDeviceCount(&num_devices));
  size_t dcount = count / num_devices;
  ERRCHK(dcount * num_devices ==
         count); // Require count divisible with num_devices

  int* darr[num_devices];
  // cudaStream_t streams[num_devices];
  const size_t bytes = dcount * sizeof(darr[0][0]);
  for (int i = 0; i < num_devices; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&darr[i], bytes);
    // cudaMemcpy(darr[i], &arr[i * dcount], bytes, cudaMemcpyHostToDevice);
    // cudaStreamCreate(&streams[i]);
  }

  // EXERCISE 4: Your code here
  // Input:
  //  darr[num_devices] - An array of integers distributed among the available
  //  devices.
  //                      For example, darr[0] is the array resident on device
  //                      #0.
  //  dcount            - The number of integers in an array per device. You can
  //                      assume that the initial count is divisible by the
  //                      number of devices, i.e. dcount * num_devices = count
  //
  // Return: the maximum integer across all device arrays in darr[]
  //
  // Task: the data is now stored in an array of array pointers (see darr
  // explanation above).
  //       Your task is to apply the reduction function implemented in task 1
  //       on all devices in parallel, combine their results, and return the
  //       final result. You can do the final reduction step on the host, but
  //       otherwise use the GPU resources relatively efficiently.
  //
  // Feel free to use CUDA streams to improve the efficiency of your code
  // (not required for full points)

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
  float kernelTime = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
	
  cudaEventRecord(start, 0);

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
      cudaEvent_t startker, stopker;
      cudaEventCreate(&startker);
      cudaEventCreate(&stopker);
      cudaEventRecord(startker, streams[i*num_devices + j]);
      reduce_kernel<threadsPerBlock><<<blocks, threadsPerBlock, threadsPerBlock, streams[i*num_devices + j]>>>(darr[j], dcount, i, out_d[j]);
      //cudaStreamSynchronize(streams[i*num_devices + j]);
      cudaEventRecord(stopker, streams[i*num_devices + j]);
      float kerTime = 0;
      cudaEventElapsedTime(&kerTime, startker, stopker);
      kernelTime += kerTime;
      // transfer data to CPU
      cudaMemcpyAsync(out[j]+i*blocksPerStream, out_d[j]+i*blocksPerStream, blocks*sizeof(int), 
                      cudaMemcpyDeviceToHost, streams[i*num_devices + j]);
      cudaEventDestroy(startker);
      cudaEventDestroy(stopker);
    }
  }
  cudaEventRecord(stop, 0); 
  // reduce the blocks sequentially with CPU as this is only a tiny fraction of calculations
  // no reason to invoke GPU kernels.
  int maximum = INT_MIN;
  for(int j = 0; j < num_devices; ++j) {
    cudaSetDevice(j);
    cudaDeviceSynchronize();
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

  float totalTime = 0;
  cudaEventElapsedTime(&totalTime, start, stop);
  printf("total time of execution = %f, kernel execution = %f\n", totalTime, kernelTime);
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return maximum;
}
