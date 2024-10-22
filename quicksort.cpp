#include "bits/stdc++.h"

float pivot_selection(int n, std::vector<float> candidates, std::vector<float> &aux) {
	if(n < 5) return aux[0];
	for(int k = 0; k < 5; k++) {
		int ind = (k+1)*n/5;
		candidates[k] = aux[ind];
	}
	std::nth_element(candidates.begin(), candidates.begin() + 2, candidates.end());
	return candidates[2];
}

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
	piv1 = less.size() > 0 ? pivot_selection(less.size(), candidates, less) : 0.;
	piv2 = greater.size() > 0 ? pivot_selection(greater.size(), candidates, greater) : 0.;

	std::copy(less.begin(), less.end(), data+start);
	std::copy(equal.begin(), equal.end(), data+start+less.size());
	std::copy(greater.begin(), greater.end(), data+start+less.size()+equal.size());

	quicksort(piv1, start, end1, data);
	quicksort(piv2, start2, end, data);

}

/*

bool is_sorted(float* data, int n){
    for(int i=1;i<n;i++){
        if(data[i]<data[i-1]){
            return false;
        }
    }
    return true;
}


int main() {
	constexpr int size = 10000;
	float* data = (float*)malloc(size*sizeof(float));
	srand(12345678);
    for(int i=0;i<size;i++){
        data[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

	quicksort(data[0],0,size,data);

	bool sorted = is_sorted(data,size);
    if(sorted){
        printf("Is sorted at rank\n");
    }else{
        printf("Not sorted at rank!!\n");
    }
	
	for (int i = 0; i < size; ++i) {
		printf("%f ", data[i]);
	}
	printf("\n");
	
	return 0;
}

*/