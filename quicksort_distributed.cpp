#include <mpi.h>
#include <bits/stdc++.h>
#include "quicksort.h"

/**
        Exercise 2: Your code here
        Input:
                pivot: a pivot value based on which to split the array in to less and greater elems
                start: starting index of the range to be sorted
                end: exclusive ending index of the range to be sorted
                data: array of floats to sort in range start till end
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
	
	       	 	

	Hints:
		why is the communicator given to you?
		maybe it could be a good idea to use MPI_Comm_split?
		could you reuse your quicksort implementation for the base case of a single process?

**/     


using namespace std;

int partition(float *data float pivot, int start, int end, bool lte) {
        vector<float> partitioned;
        for(int i = start; i < end; ++i) {
                if(lte && data[i] <= pivot) {
                        partitioned.push_back(data[i]);
                } else if(data[i] > pivot) {
                        partitioned.push_back(data[i]);
                }
        }
        int len = partitioned.size();
        if(lte) {
                for(int i = 0; i < len; ++i) {
                        data[start + i] = partitioned[i];
                }
        } else {
                int s = end-len;
                for(int i = 0; i < len; ++i) {
                        data[s + i] = partitioned[i];
                }
        }
        return len;
}

float pivot_selection(int n, std::vector<float> candidates, float *aux) {
	if(n < 5) return aux[0];
	for(int k = 0; k < 5; k++) {
		int ind = (k+1)*n/5;
		candidates[k] = aux[ind];
	}
	nth_element(candidates.begin(), candidates.begin() + 2, candidates.end());
	return candidates[2];
}

void quicksort_distributed(float pivot, int start, int end, float* &data,  MPI_Comm comm)
{

        int nprocs, rank;
        MPI_Comm newcomm;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if(nprocs == 1) {
                quicksort(pivot, start, end, data);
                // this is the last recursive operation after which we have sorted our block
                // we may just as well wait for all the processes to finnish i.e. no reason to use non-blocking methods
                // Firts we query for the size of each part of the data
                int nproc_global, len;
                MPI_Comm_size(MPI_COMM_WORLD, &nproc_global);
                int proclengths[nproc_global], displs[nproc_global];
                displs[0] = 0;
                len = end-start;

                MPI_Allgather(&len, 1, MPI_FLOAT, proclengths,
                              1, MPI_FLOAT, MPI_COMM_WORLD);
                // calculate the offsets for data placements
                for(int i = 1; i < nproc_global; ++i) { displs[i] = displs[i-1] + proclengths[i-1]; }
                // gather sorted data to all processes
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                               data, proclengths, displs, MPI_FLOAT, MPI_COMM_WORLD);
        } else {
                // for pivot selection
                float piv;
                std::vector<float> candidates(5);
                // always devide the processes in two, the left and the right side.
                int color = rank < nprocs/2;
                MPI_Comm_split(comm, color, 0, &newcomm);
                // then partition the required part of the data in the correct way and continue
                // recursively, until no more processes to split the data with
                if(color == 0) {
                        int newend = partition(data, pivot, start, end, true);
                        // select the new pivot
                        piv = newend-start > 0 ? pivot_selection(newend-start, candidates, data + start) : 0.;
                        //recursive step
                        quicksort_distributed(piv, start, newend, data, newcomm)
                } else {
                        int newstart = partition(data, pivot, start, end, false);
                        // select the pivot
                        piv = end-newstart > 0 ? pivot_selection(end-newstart, candidates, data + newstart) : 0.;
                        // recursive step
                        quicksort_distributed(piv, newstart, end, data, newcomm)
                }
        }

}
