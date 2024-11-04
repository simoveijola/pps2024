/* Main solver routines for heat equation solver */

#include "immintrin.h"
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include "heat.h"
// as
typedef double v4d __attribute__ ((vector_size (4*sizeof(double))));
const v4d v4d_zeros = {0., 0., 0., 0.};
const v4d v4d_coeffs = {-2., -2., -2., -2.};
const __m256i shuffle1 = {1,2,3,4};
const __m256i shuffle2 = {3,4,5,6};


/* Exchange the boundary values */
void exchange_init(field *temperature, parallel_data *parallel)
{
    int width;
    width = temperature->ny + 2;
    // Send to the up, receive from down
    int ind = idx(1, 0, width);
    MPI_Isend(&temperature->data[ind], 1, parallel->rowtype,
              parallel->nup, 11, parallel->comm, &parallel->requests[0]);
    ind = idx(temperature->nx + 1, 0, width);
    MPI_Irecv(&temperature->data[ind], 1, parallel->rowtype, 
              parallel->ndown, 11, parallel->comm, &parallel->requests[1]);
    
    // Send to the down, receive from up
    
    ind = idx(temperature->nx, 0, width);
    MPI_Isend(&temperature->data[ind], 1, parallel->rowtype, 
              parallel->ndown, 12, parallel->comm, &parallel->requests[2]);
    ind = idx(0, 0, width);
    MPI_Irecv(&temperature->data[ind], 1, parallel->rowtype,
              parallel->nup, 12, parallel->comm, &parallel->requests[3]);
   
    // Send to the left, receive from right
   
    ind = idx(0, 1, width);
    MPI_Isend(&temperature->data[ind], 1, parallel->columntype,
              parallel->nleft, 13, parallel->comm, &parallel->requests[4]); 
    ind = idx(0, temperature->ny + 1, width);
    MPI_Irecv(&temperature->data[ind], 1, parallel->columntype, 
      		    parallel->nright, 13, parallel->comm, &parallel->requests[5]); 
     
    // Send to the right, receive from left   
    ind = idx(0, temperature->ny, width);
    MPI_Isend(&temperature->data[ind], 1, parallel->columntype,
              parallel->nright, 14, parallel->comm, &parallel->requests[6]);
    ind = 0;
    MPI_Irecv(&temperature->data[ind], 1, parallel->columntype,
              parallel->nleft, 14, parallel->comm, &parallel->requests[7]);

}

/* complete the non-blocking communication */
void exchange_finalize(parallel_data *parallel)
{
    MPI_Waitall(8, &parallel->requests[0], MPI_STATUSES_IGNORE);
}

/* Update the temperature values using five-point stencil */
void evolve_interior(field *curr, field *prev, double a, double dt)
{
    int i, j;
    int ic, iu, id, il, ir; // indexes for center, up, down, left, right
    int width, ny, nx;
    ny = curr->ny, nx = curr->nx;
    width = ny + 2;
    double dx2, dy2;

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    dx2 = prev->dx * prev->dx;
    dy2 = prev->dy * prev->dy;
    /*
    #pragma omp parallel for schedule(static)
    for (i = 2; i < curr->nx; i++) {
        for (j = 2; j < curr->ny; j++) {
            ic = idx(i, j, width);
            iu = idx(i+1, j, width);
            id = idx(i-1, j, width);
            ir = idx(i, j+1, width);
            il = idx(i, j-1, width);
            curr->data[ic] = prev->data[ic] + a * dt *
                               ((prev->data[iu] -
                                 2.0 * prev->data[ic] +
                                 prev->data[id]) / dx2 +
                                (prev->data[ir] -
                                 2.0 * prev->data[ic] +
                                 prev->data[il]) / dy2);
        }
    }
    */
    double invdx2 = 1. / dx2;
    double invdy2 = 1. / dy2;
    double adt = a * dt;
    
    #pragma omp parallel for
    for (i=2; i < nx; i++) {
        ic = idx(i, 2, width);
        iu = idx(i+1, 2, width);
        id = idx(i-1, 2, width);
        ir = idx(i, 3, width);
        il = idx(i, 1, width);
        for (j=2; j < ny; j++) {
            curr->data[ic] = prev->data[ic] + adt *
                               ((prev->data[iu++] -
                                 2.0 * prev->data[ic] +
                                 prev->data[id++]) * invdx2 +
                                (prev->data[ir++] -
                                 2.0 * prev->data[ic] +
                                 prev->data[il++]) * invdy2);
	    ic++;
        }
    }
    
    
}

/* Update the temperature values using five-point stencil */
/* update only the border-dependent regions of the field */
void evolve_edges(field *curr, field *prev, parallel_data *parallel, double a, double dt)
{
    // receives we are waiting for in the order of data origin: down, up, right, left
    MPI_Request receives[4] = {parallel->requests[1], parallel->requests[3], parallel->requests[5], parallel->requests[7]};
    int tasksleft = 4;
    // some common variables
    int width = curr->ny + 2;
    double dx2 = prev->dx * prev->dx, dy2 = prev->dy * prev->dy;

    #pragma omp parallel 
    {
        #pragma omp master
        {
        while(tasksleft > 0) {
            // Wait for some receive to be ready 
            int req;
            MPI_Waitany(4, receives, &req, MPI_STATUSES_IGNORE);
            tasksLeft--;
            // Assing the ready to be handled task to some thread
            #pragma omp task private(req) 
            {
                int i, j;
                int ic, iu, id, il, ir; // indexes for center, up, down, left, right

                /* Determine the temperature field at next time step
                * As we have fixed boundary conditions, the outermost gridpoints
                * are not updated. */

                if(req == 1) { // received from up
                    i = 1;
                    for (j = 2; j < curr->ny; j++) {
                        ic = idx(i, j, width);
                        iu = idx(i+1, j, width);
                        id = idx(i-1, j, width);
                        ir = idx(i, j+1, width);
                        il = idx(i, j-1, width);
                        curr->data[ic] = prev->data[ic] + a * dt *
                                        ((prev->data[iu] -
                                            2.0 * prev->data[ic] +
                                            prev->data[id]) / dx2 +
                                            (prev->data[ir] -
                                            2.0 * prev->data[ic] +
                                            prev->data[il]) / dy2);
                    }
                } else if(req == 0) { // received from down
                    i = curr -> nx;
                    for (j = 2; j < curr->ny; j++) {
                        ic = idx(i, j, width);
                        iu = idx(i+1, j, width);
                        id = idx(i-1, j, width);
                        ir = idx(i, j+1, width);
                        il = idx(i, j-1, width);
                        curr->data[ic] = prev->data[ic] + a * dt *
                                        ((prev->data[iu] -
                                            2.0 * prev->data[ic] +
                                            prev->data[id]) / dx2 +
                                            (prev->data[ir] -
                                            2.0 * prev->data[ic] +
                                            prev->data[il]) / dy2);
                    }
                } else if(req == 2) { // received from left
                    j = 1;
                    for (i = 2; i < curr->nx; i++) {
                        ic = idx(i, j, width);
                        iu = idx(i+1, j, width);
                        id = idx(i-1, j, width);
                        ir = idx(i, j+1, width);
                        il = idx(i, j-1, width);
                        curr->data[ic] = prev->data[ic] + a * dt *
                                        ((prev->data[iu] -
                                            2.0 * prev->data[ic] +
                                            prev->data[id]) / dx2 +
                                            (prev->data[ir] -
                                            2.0 * prev->data[ic] +
                                            prev->data[il]) / dy2);
                    }
                } else { // received from right
                    j = curr -> ny;
                    for (i = 2; i < curr->nx; i++) {
                        ic = idx(i, j, width);
                        iu = idx(i+1, j, width);
                        id = idx(i-1, j, width);
                        ir = idx(i, j+1, width);
                        il = idx(i, j-1, width);
                        curr->data[ic] = prev->data[ic] + a * dt *
                                        ((prev->data[iu] -
                                            2.0 * prev->data[ic] +
                                            prev->data[id]) / dx2 +
                                            (prev->data[ir] -
                                            2.0 * prev->data[ic] +
                                            prev->data[il]) / dy2);
                    }
                }
            }
        }
        }
    }
    // after all data has been received, and calculated, calculate the last corner values
    for(int i = 1; i < curr->nx+1; i+=curr->nx-1) {
        for(int j = 1; j < curr->ny+1; j+=curr->ny-1) {
            int ic = idx(i, j, width);
            int iu = idx(i+1, j, width);
            int id = idx(i-1, j, width);
            int ir = idx(i, j+1, width);
            int il = idx(i, j-1, width);
            curr->data[ic] = prev->data[ic] + a * dt *
                            ((prev->data[iu] -
                                2.0 * prev->data[ic] +
                                prev->data[id]) / dx2 +
                                (prev->data[ir] -
                                2.0 * prev->data[ic] +
                                prev->data[il]) / dy2);
        }
    }
}

/*
            #pragma unroll
            for(int row = 0; row < 3; ++row) {
                leftside[row] = prev -> data[idx(i+row, j-1, width)];
                rightside[row] = prev -> data[idx(i+row, j+4, width)];
            }

            // PREVIOUS AND NEXT ROW ABOVE AND BELOW THESE THREE
            vec[0] = _mm256_loadu_pd(prev -> data + idx(i-1, j, width));
            vec[4] = _mm256_loadu_pd(prev -> data + idx(i+3, j, width));

            // THE IDEA: these operations should be automatically such that the processor pipelines
            // these operations for the three rows and performs them out of order

            vec[1] = _mm256_loadu_pd(prev -> data + idx(i, j, width));
            vec[2] = _mm256_loadu_pd(prev -> data + idx(i+1, j, width));
            vec[3] = _mm256_loadu_pd(prev -> data + idx(i+2, j, width));

            vec_sl[0] = __builtin_shuffle(vec[1], vecz, shuffle1);
            vec_sl[1] = __builtin_shuffle(vec[2], vecz, shuffle1);
            vec_sl[2] = __builtin_shuffle(vec[3], vecz, shuffle1);
            // shift the vectors right
            vec_sr[0] = __builtin_shuffle(vecz, vec[1], shuffle2);
            vec_sr[1] = __builtin_shuffle(vecz, vec[2], shuffle2);
            vec_sr[2] = __builtin_shuffle(vecz, vec[3], shuffle2);
            // add the right side and left side values together
            vec_sl[0] += vec_sr[0];
            vec_sl[1] += vec_sr[1];
            vec_sl[2] += vec_sr[2];
            // multiply center value with the coefficient and add the previous row at the same time
            res[0] = _mm256_fmadd_pd(coeff, vec[1], vec[0]);
            res[1] = _mm256_fmadd_pd(coeff, vec[2], vec[1]);
            res[2] = _mm256_fmadd_pd(coeff, vec[3], vec[2]);
            vec_sl[0] = _mm256_fmadd_pd(coeff, vec[1], vec_sl[0]);
            vec_sl[1] = _mm256_fmadd_pd(coeff, vec[2], vec_sl[1]);
            vec_sl[2] = _mm256_fmadd_pd(coeff, vec[3], vec_sl[2]);
            vec_sl[0][0] += leftside[0];
            vec_sl[1][0] += leftside[1];
            vec_sl[2][0] += leftside[2];
            vec_sl[0][3] += rightside[0];
            vec_sl[1][3] += rightside[1];
            vec_sl[2][3] += rightside[2];
            // add the row below to this row and multiply by the 1/dx2
            res[0] = invdx2*(res[0] + vec[2]);
            res[1] = invdx2*(res[1] + vec[3]);
            res[2] = invdx2*(res[2] + vec[4]);
            // add the two scaled terms together and store in res
            res[0] = _mm256_fmadd_pd(invdy2, vec_sl[0], res[0]);
            res[1] = _mm256_fmadd_pd(invdy2, vec_sl[1], res[1]);
            res[2] = _mm256_fmadd_pd(invdy2, vec_sl[2], res[2]);
            // once more use fmadd to multiply res with a*dt and add previous vector
            res[0] = _mm256_fmadd_pd(adt, res[0], vec[1]);
            res[1] = _mm256_fmadd_pd(adt, res[1], vec[2]);
            res[2] = _mm256_fmadd_pd(adt, res[2], vec[3]);
            // store the final vectors into the memory
            _mm256_storeu_pd(curr -> data + idx(i, j, width), res[0]);
            _mm256_storeu_pd(curr -> data + idx(i+1, j, width), res[1]);
            _mm256_storeu_pd(curr -> data + idx(i+2, j, width), res[2]);
*/

/*
            prev_row = _mm256_load_pd(prev->data[idx(i-1, j, width)]);
            vec1 = _mm256_load_pd(prev -> data + idx(i, j, width));
            vec2 = _mm256_load_pd(prev -> data + idx(i+1, j, width));
            vec3 = _mm256_load_pd(prev -> data + idx(i+2, j, width));
            vec4 = _mm256_load_pd(prev -> data + idx(i+3, j, width));
            next_row = _mm256_load_pd(prev -> data + idx(i+4, j, width));

            vec1_sl = __builtin_shuffle(vec1, vecz, {1,2,3,4});
            vec2_sl = __builtin_shuffle(vec2, vecz, {1,2,3,4});
            vec3_sl = __builtin_shuffle(vec3, vecz, {1,2,3,4});
            vec4_sl = __builtin_shuffle(vec4, vecz, {1,2,3,4});

            vec1_sr = __builtin_shuffle(vec_pr[0], vec1, {3,4,5,6});
            vec2_sr = __builtin_shuffle(vec_pr[1], vec2, {3,4,5,6});
            vec3_sr = __builtin_shuffle(vec_pr[2], vec3, {3,4,5,6});
            vec4_sr = __builtin_shuffle(vec_pr[3], vec4, {3,4,5,6});

            res1 = _mm256_fmadd_pd(coeff, vec1, prev_row);
            res2 = _mm256_fmadd_pd(coeff, vec2, vec1);
            res3 = _mm256_fmadd_pd(coeff, vec3, vec2);
            res4 = _mm256_fmadd_pd(coeff, vec4, vec3);

            vec1_sl += vec1_sr;
            vec2_sl += vec2_sr;
            vec3_sl += vec3_sr;
            vec4_sl += vec4_sr;

            res1 += vec2;
            res2 += vec3;
            res3 += vec4;
            res4 += next_row;

            res1 += vec1_sl;
            res2 += vec2_sl;
            res3 += vec3_sl;
            res4 += vec4_sl;

            vec_pr[0] += vec1;
            vec_pr[1] += vec2;
            vec_pr[2] += vec3;
            vec_pr[3] += vec4;

            if(j > 2) {
                _mm256_store_pd(prev->data + idx(i-1, j-4, width), vec_pr[0])
                vec_pr[1] += vec2;
                vec_pr[2] += vec3;
                vec_pr[3] += vec4;
            }
            */
