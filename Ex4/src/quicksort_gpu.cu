#include "bits/stdc++.h"
#include <cuda_runtime_api.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

constexpr int elementsPerBlock = 256;
constexpr int threadsPerBlock = 128;

// makes the comparisons and saves the results to lt, eq, and gt
__global__ void compare_kernel(float *data, int *lt, int *eq, int *gt, float pivot, int start, int end) {
        int ia = threadIdx.x, ib = blockIdx.x;
        int offset = start + ib*elementsPerBlock;
        int id = ia;
        int endBlock = min(elementsPerBlock, end-offset);
        
        while(id < endBlock) {
                float element = data[offset + id];
                // conviniently zeros the elements left from previous iterations at the same time as sets the new values
                lt[offset + id] = element < pivot ? 1 : 0;
                eq[offset + id] = element == pivot ? 1 : 0;
                gt[offset + id] = element > pivot ? 1 : 0;
                id += threadsPerBlock;
        }
}
// distributes data from 'tmp' to 'data'
__global__ void distribute_kernel(float *data, float *tmp, int *lt, int *eq, int *gt, int countLt, int countEq, int countGt, float pivot, int start, int end) {
        int ia = threadIdx.x, ib = blockIdx.x;
        int offset = start + ib*elementsPerBlock;
        int id = ia;
        int endBlock = min(elementsPerBlock, end-offset);
        while(id < endBlock) {
                float element = tmp[offset + id];
                // conviniently zeros the elements left from previous iterations at the same time as sets the new values
                if(element < pivot){
                        data[start + lt[offset + id]] = element;
                } else if(element == pivot) {
                        data[start + countLt + eq[offset + id]] = element;
                } else {
                        data[start + countLt + countEq + gt[offset + id]] = element;
                }
                id += threadsPerBlock;
        }
}
// partitions the data with respect to the pivot in stable manner
// returns pair of indices that tell where the less than array ends, and where the greater than array starts
std::pair<int, int> partition(float *data, float *tmp, int *lt, int *eq, int *gt, float pivot, int start, int end) {
        int n = end-start;
        int numBlocks = (n+elementsPerBlock-1)/elementsPerBlock;
        // check the comparisons using compare kernel
        compare_kernel<<<numBlocks, threadsPerBlock>>>(data, lt, eq, gt, pivot, start, end);
        // move the data to tmp
        //copy_kernel<<<256, 256>>>(data, tmp, start, end);
        cudaMemcpy(tmp+start, data+start, n*sizeof(float), cudaMemcpyDeviceToDevice);
        // need the thrust wrappers to run thrust exclusive scan
	thrust::device_ptr<int> tlt(lt);
        thrust::device_ptr<int> teq(eq);
        thrust::device_ptr<int> tgt(gt);

        // store the last elements of lt, gt and eq. Then we can add these to the exclusive sum to get
        // the total sum of elements. This is rather redundant unfortunately.
        int countLess, countEqual, countGreater; 
        cudaMemcpy(&countLess, lt+end-1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&countEqual, eq+end-1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&countGreater, gt+end-1, sizeof(int), cudaMemcpyDeviceToHost);
        
        // in-place exclusive scans for all these arrays
        thrust::exclusive_scan(tlt+start, tlt+end, tlt+start);
        thrust::exclusive_scan(teq+start, teq+end, teq+start);
        thrust::exclusive_scan(tgt+start, tgt+end, tgt+start);
	// update total counts
        int exSumLess, exSumEqual, exSumGreater; 
        cudaMemcpy(&exSumLess, lt+end-1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&exSumEqual, eq+end-1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&exSumGreater, gt+end-1, sizeof(int), cudaMemcpyDeviceToHost);
        countLess += exSumLess;
        countEqual += exSumEqual;
        countGreater += exSumGreater;
        
        // now distribute the data back from 'tmp' to 'data'
        distribute_kernel<<<numBlocks, threadsPerBlock>>>(data, tmp, lt, eq, gt, countLess, countEqual, countGreater, pivot, start, end);

        return std::make_pair(countLess, countLess+countEqual);
}

// here we asume that every array is on device
void quicksort_device(float pivot, int start, int end, float *data, float *tmp, int *lt, int *eq, int *gt) {
        int n = end-start;
	if(n <= 1) {
		return;
	}
	std::pair<int,int> edges = partition(data, tmp, lt, eq, gt, pivot, start, end);
        int end1 = start + edges.first, start2 = start + edges.second;
        float piv1 = 0., piv2 = 0.;
        // choose the pivots from the middle of the data and transfer to host
        cudaMemcpy(&piv1, data+(start+end1)/2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&piv2, data+(start2+end)/2, sizeof(float), cudaMemcpyDeviceToHost);
        // recursively call the function again to sort the less than and greater than sides
        quicksort_device(piv1, start, end1, data, tmp, lt, eq, gt);
        quicksort_device(piv2, start2, end, data, tmp, lt, eq, gt);
}

void quicksort(float pivot, int start, int end, float* &data)
{
/**
        Exercise 4: Your code here
        Input:
                pivot: a pivot value based on which to split the array in to less and greater elems
                start: starting index of the range to be sorted
                end: exclusive ending index of the range to be sorted
                data: array of floats allocated on the GPU to sort in range start till end
        Return:
                upon return the array range should be sorted
        Task:
                to sort the array using the idea of quicksort in a stable manner
                a sort is stable if it maintains the relative order of elements with equal values
		during the sorting try to keep the data movement of the CPU and GPU as small as possible
		and ideally only move scalars between them
	Hint:
		prefix sum is a beneficial primitive in this case that you are recommended to use:
		https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
**/
        int n = end-start;
        if(n <= 1) {
                return;
        }
        // idea here is to allocate the GPU memory only once to reduce overhead.
        // different partitioned sides operate on different indices of these arrays, avoiding
        // race conditions
        float *dataGPU = NULL, *tmpGPU = NULL;
       	int *ltGPU = NULL, *eqGPU = NULL, *gtGPU = NULL;
        //cudaMalloc((void**)&dataGPU, n*sizeof(float));
        cudaMalloc((void**)&tmpGPU, n*sizeof(float));
        cudaMalloc((void**)&ltGPU, n*sizeof(int));
        cudaMalloc((void**)&eqGPU, n*sizeof(int));
        cudaMalloc((void**)&gtGPU, n*sizeof(int));
        // move data to GPU
        //cudaMemcpy(dataGPU, data+start, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(ltGPU, 0, n*sizeof(int));
	cudaMemset(eqGPU, 0, n*sizeof(int));
	cudaMemset(gtGPU, 0, n*sizeof(int));
        // use device quicksort
        quicksort_device(pivot, start, end, data, tmpGPU, ltGPU, eqGPU, gtGPU);
        // move data back to host from device
        //cudaMemcpy(data+start, dataGPU, n*sizeof(float), cudaMemcpyDeviceToHost);
        // free the device arrays
        //cudaFree(dataGPU);
        cudaFree(tmpGPU);
        cudaFree(ltGPU);
        cudaFree(eqGPU);
        cudaFree(gtGPU);
}



