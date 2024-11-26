
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

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

void quicksort(float pivot, int start, int end, float* data, MPI_Comm comm, float *tmp, int* lt, int* eq, int *gt, int mpiend)
{

        int n = end-start;
        if(n <= 1) {
                return;
        }
        
        int nprocs, rank;
        MPI_Comm newcomm;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if(nprocs == 1) {
                // sort this processes subarray using the sequential sorting
                quicksort_device(pivot, start, end, data, tmp, lt, eq, gt);
                // this is the last recursive operation after which we have sorted our block
                // we may just as well wait for all the processes to finnish i.e. no reason to use non-blocking methods
                // Firts we query for the size of each part of the data
                int nproc_global, len, rankg;
                MPI_Comm_size(MPI_COMM_WORLD, &nproc_global);
                MPI_Comm_rank(MPI_COMM_WORLD, &rankg);
                int proclengths[nproc_global], displs[nproc_global];
                displs[0] = 0;
                len = mpiend-start;
                // debugging prints
                //printf("rank %i sorted elements between [%i, %i]\n", rankg, start, end);
                MPI_Allgather(&len, 1, MPI_FLOAT, proclengths,
                        1, MPI_FLOAT, MPI_COMM_WORLD);
                // calculate the offsets for data placements
                for(int i = 1; i < nproc_global; ++i) { displs[i] = displs[i-1] + proclengths[i-1]; }
                // gather sorted data to all processes
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                        data, proclengths, displs, MPI_FLOAT, MPI_COMM_WORLD);
                return;
        } else {
                // always devide the processes in two, the left and the right side.
                int color = rank < nprocs/2 ? 0 : 1;
                MPI_Comm_split(comm, color, 0, &newcomm);
                // then partition the required part of the data in the correct way and continue
                // recursively, until no more processes to split the data with
                if(color == 0) {
                        std::pair<int,int> edges = partition(data, tmp, lt, eq, gt, pivot, start, end);
                        int newend = start + edges.first;
                        int mpiend = start + edges.second;
                        float piv = 0.;
                        // choose the pivots from the middle of the data and transfer to host
                        cudaMemcpy(&piv, data+(start+newend)/2, sizeof(float), cudaMemcpyDeviceToHost);
                        // pass also the value of next start so that we can gather the equal elements later with mpi
                        // color 0 nodes take care of this
                        quicksort(piv, start, newend, data, newcomm, tmp, lt, eq, gt, mpiend);
                } else {
                        std::pair<int,int> edges = partition(data, tmp, lt, eq, gt, pivot, start, end);
                        int newstart = start + edges.second;
                        float piv = 0.;
                        // choose the pivots from the middle of the data and transfer to host
                        cudaMemcpy(&piv, data+(newstart+end)/2, sizeof(float), cudaMemcpyDeviceToHost);
                        quicksort(piv, newstart, end, data, newcomm, tmp, lt, eq, gt, end);
                }
        }	
        
}

void quicksort_distributed(float pivot, int start, int end, float* &data, MPI_Comm comm)
{
/**
        Exercise 4: Your code here
        Input:
                pivot: a pivot value based on which to split the array in to less and greater elems
                start: starting index of the range to be sorted
                end: exclusive ending index of the range to be sorted
                data: array of floats allocated on the GPU to sort in range start till end
                      the array is the same for each MPI process
                comm: the communicator of the MPI processes
        Return:
                upon return each MPI process should have their array sorted
        Task:
                To sort the array using the idea of quicksort in a stable manner using each MPI process.
                A sort is stable if it maintains the relative order of elements with equal values.

                Truly split the work among the process i.e. don't have each process independently sort the array.
                You should stick to the idea of quicksort also for the splitting of the array between processes.
                As an example you should not split the array into N equal sized parts (where N is the number of processors),
                sort them and then combine then (this would be following mergesort). Instead the splitting should be based on the
                pivot values

		Furthermore for the communication try to use CUDA aware MPI instead of first loading the data to host
		and then doing the communication 

        Hint:
		!!You should be able to combine and copy your solutions in quicksort_gpu.cu and quicksort_distributed.cu!!
        **/
        //printf("rank %i, nprocs = %i, start = %i, end= %i\n", rank, nprocs, start, end);

        int n = end-start;
        if(n <= 1) {
                return;
        }
        // cuda allocations: do only once
        float *dataGPU = NULL, *tmpGPU = NULL;
        int *ltGPU = NULL, *eqGPU = NULL, *gtGPU = NULL;
        //cudaMalloc((void**)&dataGPU, n*sizeof(float));
        cudaMalloc((void**)&tmpGPU, n*sizeof(float));
        cudaMalloc((void**)&ltGPU, n*sizeof(int));
        cudaMalloc((void**)&eqGPU, n*sizeof(int));
        cudaMalloc((void**)&gtGPU, n*sizeof(int));

        cudaMemset(ltGPU, 0, n*sizeof(int));
        cudaMemset(eqGPU, 0, n*sizeof(int));
        cudaMemset(gtGPU, 0, n*sizeof(int));

        // then call quicksort function, which first partitions the data for each node recursively, and then calls the previous gpu
        // of quicksort
        quicksort(pivot, start, end, data, comm, tmpGPU, ltGPU, eqGPU, gtGPU, end);

        // free temporary cuda arrays
        cudaFree(tmpGPU);
        cudaFree(ltGPU);
        cudaFree(eqGPU);
        cudaFree(gtGPU);
}
