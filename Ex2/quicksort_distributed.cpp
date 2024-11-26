#include <mpi.h>
#include <bits/stdc++.h>

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

// lte tells whether we gather data left-side (less than or equal) of the pivot or not
// stable due to linear traversal and storing of the data in the same order
int partition(float *data, float pivot, int start, int end, bool lte) {
        vector<float> partitioned;
        for(int i = start; i < end; ++i) {
                if(lte && data[i] <= pivot) {
                        partitioned.push_back(data[i]);
                } else if(!lte && data[i] > pivot) {
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

float pivot_selection(int n, std::vector<float> candidates, float *data) {
	if(n < 5) return data[0];
	for(int k = 0; k < 5; k++) {
		int ind = (k+1)*n/5;
		candidates[k] = data[ind];
	}
	nth_element(candidates.begin(), candidates.begin() + 2, candidates.end());
	return candidates[2];
}
// the sequential solution
void quicksort(float pivot, int start, int end, float* &data)
{
	int n = end-start;
	if(n <= 1) {
		return;
	}
	std::vector<float> less, equal, greater;

	for(int k = start; k < end; ++k) {
		//printf("%f %f\n", data[k], pivot);
		if(data[k] < pivot) {
			less.push_back(data[k]);
		} else if(data[k] == pivot) {
			equal.push_back(data[k]);
		} else {
			greater.push_back(data[k]);
		}
	}

	int end1 = start + less.size(), start2 = start + less.size() + equal.size();
	// std::cout << start << " " << end1 << " " << start2 << " " << end << std::endl;
	float piv1, piv2;
	std::vector<float> candidates(5);
	// better pivot selection
        std::copy(less.begin(), less.end(), data+start);
	std::copy(equal.begin(), equal.end(), data+start+less.size());
	std::copy(greater.begin(), greater.end(), data+start+less.size()+equal.size());
	piv1 = less.size() > 0 ? pivot_selection(less.size(), candidates, data + start) : 0.;
	piv2 = greater.size() > 0 ? pivot_selection(greater.size(), candidates, data + end - greater.size()) : 0.;
	quicksort(piv1, start, end1, data);
	quicksort(piv2, start2, end, data);
}

// why is this stable?
// the sequential merge sort is stable and the partitioning algorithm used here
// to divide the data for each process is stable, due to linear traversal and storing
// of the data
void quicksort_distributed(float pivot, int start, int end, float* &data,  MPI_Comm comm)
{
        
	int nprocs, rank;
        MPI_Comm newcomm;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);
	//printf("rank %i, nprocs = %i, start = %i, end= %i\n", rank, nprocs, start, end);

        if(nprocs == 1) {
		// sort this processes subarray using the sequential sorting
                quicksort(pivot, start, end, data);
                // this is the last recursive operation after which we have sorted our block
                // we may just as well wait for all the processes to finnish i.e. no reason to use non-blocking methods
                // Firts we query for the size of each part of the data
                int nproc_global, len, rankg;
                MPI_Comm_size(MPI_COMM_WORLD, &nproc_global);
		MPI_Comm_rank(MPI_COMM_WORLD, &rankg);
                int proclengths[nproc_global], displs[nproc_global];
                displs[0] = 0;
                len = end-start;
		// debugging prints
		//printf("rank %i sorted elements between [%i, %i]\n", rankg, start, end);
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
                int color = rank < nprocs/2 ? 0 : 1;
                MPI_Comm_split(comm, color, 0, &newcomm);
                // then partition the required part of the data in the correct way and continue
                // recursively, until no more processes to split the data with
                if(color == 0) {
                        int newend = start + partition(data, pivot, start, end, true);
                        // select the new pivot
                        piv = newend-start > 0 ? pivot_selection(newend-start, candidates, data + start) : 0.;
                        //recursive step
                        quicksort_distributed(piv, start, newend, data, newcomm);
                } else {
                        int newstart = end - partition(data, pivot, start, end, false);
                        // select the pivot
                        piv = end-newstart > 0 ? pivot_selection(end-newstart, candidates, data + newstart) : 0.;
                        // recursive step
                        quicksort_distributed(piv, newstart, end, data, newcomm);
                }
        }	
}

