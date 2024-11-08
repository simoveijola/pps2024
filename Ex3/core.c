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
void exchange_init(field *temperature, parallel_data *parallel, MPI_Request *requests)
{	
    int width, height, stepx, stepy, nthreads;
    //int ind;
    width = temperature->ny + 2;
    height = temperature->nx + 2;

    nthreads = 1;//omp_get_max_threads();
    stepx = height/nthreads;
    stepy = width/nthreads;
    // the work of sending and receiving
    #pragma omp for nowait
    for(int i = 0; i < 4; ++i) {
        // define the offset for the thread
        int startx = 0;//stepx*i;
        int starty = 0;//stepy*i;
        int ind;
        // send up, receive down
	if(i == 0)
        {
            ind = idx(1, starty, width);
            MPI_Isend(&temperature->data[ind], 1, parallel->rowtype_full,
                    parallel->nup, 11, parallel->comm, &requests[i*0 + 0]);

            ind = idx(temperature->nx + 1, starty, width);
            MPI_Irecv(&temperature->data[ind], 1, parallel->rowtype_full, 
                    parallel->ndown, 11, parallel->comm, &requests[i*0 + 1]);
        }
        // send down, receive up
	if(i == 1)
        {
            ind = idx(temperature->nx, starty, width);
            MPI_Isend(&temperature->data[ind], 1, parallel->rowtype_full, 
                    parallel->ndown, 12, parallel->comm, &requests[i*0 + 2]);

            ind = idx(0, starty, width);
            MPI_Irecv(&temperature->data[ind], 1, parallel->rowtype_full,
                    parallel->nup, 12, parallel->comm, &requests[i*0 + 3]);
        }
        // send left, receive right
        if(i == 2)
	{
            ind = idx(startx, 1, width);
            MPI_Isend(&temperature->data[ind], 1, parallel->columntype_full,
                    parallel->nleft, 13, parallel->comm, &requests[i*0 + 4]); 

            ind = idx(startx, temperature->ny + 1, width);
            MPI_Irecv(&temperature->data[ind], 1, parallel->columntype_full, 
                        parallel->nright, 13, parallel->comm, &requests[i*0 + 5]);
        }
        // send right, receive left
        if(i == 3)
	{
            ind = idx(startx, temperature->ny, width);
            MPI_Isend(&temperature->data[ind], 1, parallel->columntype_full,
                    parallel->nright, 14, parallel->comm, &requests[i*0 + 6]);

            ind = startx;
            MPI_Irecv(&temperature->data[ind], 1, parallel->columntype_full,
                    parallel->nleft, 14, parallel->comm, &requests[i*0 + 7]);
        }
    }

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

    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    dx2 = prev->dx * prev->dx;
    dy2 = prev->dy * prev->dy;
    /*
    
    #pragma omp for
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
    
    #pragma omp for
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
void evolve_edges(field *curr, field *prev, parallel_data *parallel, double a, double dt,  MPI_Request *requests)
{
    // some common variables
    double dx2 = prev->dx * prev->dx, dy2 = prev->dy * prev->dy;
    int width, height, stepx, stepy, nthreads;
    //int ind;
    width = curr->ny + 2;
    height = curr->nx + 2;

    nthreads = 1;// omp_get_max_threads();
    stepx = height/nthreads;
    stepy = width/nthreads;
    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.

    // send the data in parallel with maximum number of threads to divide 
    // the work of sending and receiving
    #pragma omp for
    for(int k = 0; k < 4; ++k) {
        int i, j, ic, iu, id, ir, il;

        int startx = 2;//k > 0 ? stepx*k : 2;
        int endx = curr->nx;//k < nthreads-1 ? stepx*(k+1) : curr->nx;
        int starty = 2;//k > 0 ? stepy*k : 2;
        int endy = curr->ny;//k < nthreads-1 ? stepy*(k+1) : curr->ny;
        
	if(k == 0) {
	MPI_Waitall(2, &requests[2], MPI_STATUSES_IGNORE);
        i = 1;
	for (j = starty; j < endy; j++) {
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
	if(k == 1) {
	MPI_Waitall(2, &requests[0], MPI_STATUSES_IGNORE);
        i = curr -> nx;
        for (j = starty; j < endy; j++) {
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
	if(k == 2) {
	MPI_Waitall(2, &requests[6], MPI_STATUSES_IGNORE);
        j = 1;
        for (i = startx; i < endx; i++) {
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
	if(k == 3) {
	MPI_Waitall(2, &requests[4], MPI_STATUSES_IGNORE);
        j = curr -> ny;
        for (i = startx; i < endx; i++) {
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
