#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

static const int halo_width=2;    // fill in halo_width
int rank, nprocs;

const float pi=3.14159;
const float u_x=0.5 , u_y=0.5 , c_amp=1;  // choose velocity components and amplitude of initial condition
const float cdt=.3;               // safety factor for timestep (experiment!)
static float dx, dy;              // grid spacings

float ugrad_upw(int i, int j, int ny, float data[][ny]){

    // u.grad operator with upwinding acting on field in data at point i,j.
    //
    const float coeff[]={-3./2.,4./2.,-1./2.};
    float sum_x=0., sum_y=0.;

    int inc = -copysign(1.0, u_x);
    for (int k=0; k<=halo_width; k++){
        sum_x += coeff[k]*data[i+inc*k][j];
    }
    sum_x *= fabs(u_x)/dx;

    inc = -copysign(1.0, u_y);
    for (int k=0; k<=halo_width; k++){
        sum_y += coeff[k]*data[i][j+inc*k];
    }
    sum_y *= fabs(u_y)/dy;

    return sum_x + sum_y;
}

// not used at the moment as we use MPI_Cart
int find_proc(int ipx, int ipy, int npx)
{
    return ipy*npx + ipx;
}

int* find_proc_coords(int rank, int npx, int npy)
{
    int coords[2];
    coords[0] = rank % npx;
    coords[1] = rank / npx;
    return coords;
}

void initcond(int nx, int ny, float x[], float y[], float data[][ny+2*halo_width])
{
    // Initialisation of field in data: harmonic function in x (can be modified to a harmonic in y or x and y):
    for (int ix = halo_width; ix < halo_width+nx; ++ix)
    {
        for (int iy = halo_width; iy < halo_width+ny; ++iy)
        {
            data[ix][iy] = c_amp*sin((double) x[ix]);
            // other choices:
            //data[ix][iy] = c_amp*sin((double) y[iy]);
            //data[ix][iy] = c_amp*sin((double) x[ix])*sin((double) y[iy]);
        }
    }
}

void rhs(const int xrange[2], const int yrange[2], int ny, float data[][ny+2*halo_width], float d_data[][ny+2*halo_width])
{
    //Right-hand side d_data of pde for field in data for a subdomain defined by xrange, yrange:
    int ix,iy;

    for (ix = xrange[0]; ix < xrange[1]; ++ix) {
        for (iy = yrange[0]; iy < yrange[1]; ++iy)
        {
            d_data[ix][iy] = ugrad_upw(ix, iy, ny, data);
        }
    }
}

FILE*
get_file_ptr(const char* prefix, const int pid)
{
    char name[4098];
    sprintf(name,"%s%d",prefix,pid);
    return fopen(name,"w");
}
int main(int argc, char** argv)
{   

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check compatibility of argv parameters!

    int nprocx = atoi(argv[1]); 
    int nprocy = atoi(argv[2]);

    int domain_nx = atoi(argv[3]),                 // number of gridpoints in x direction
        subdomain_nx = domain_nx/nprocx,                            // subdomain x-size w/o halos
        subdomain_mx = subdomain_nx + 2*halo_width;                            //                  with halos

    int domain_ny = atoi(argv[4]),                 // number of gridpoints in y direction
        subdomain_ny = domain_ny/nprocy,                           // subdomain y-size w/o halos
        subdomain_my = subdomain_ny + 2*halo_width;                        //                  with halos

    // Find neighboring processes!
    int *proc_coords;
    MPI_Comm newComm;

    const int dim = 2;
    const int dims[2] = {nprocx, nprocy};
    const int periodic[2] = {1,1};
    // create cartesian coordinates, new communicators, and find the new rank
    // and coordinates for this process
    MPI_Cart_create(MPI_COMM_WORLD, dim, dims, periodic, 1, &newComm);
    MPI_Comm_rank(newComm, &rank);
    MPI_Cart_coords(newComm, rank, dim, proc_coords);
    int ipx=proc_coords[0], ipy=proc_coords[1];
    
    // int nL_coords[2], nR_coords[2], nU_coords[2], nD_coords[2];
    int nL_rank, nR_rank, nU_rank, nD_rank;
    // Find the neighbour ranks
    MPI_Cart_shift(newComm, 0, 1, &nU_rank, &nD_rank);
    MPI_Cart_shift(newComm, 1, 1, &nL_rank, &nR_rank);


    float data[subdomain_mx][subdomain_my], d_data[subdomain_mx][subdomain_my];

    float xextent=2.*pi, yextent=2.*pi;            // domain has extents 2 pi x 2 pi

    // Set grid spacings dx, dy:
    dx=xextent/domain_nx, dy=yextent/domain_ny;

    float x[subdomain_mx], y[subdomain_my];
    int ix, iy;

    // Populate grid coordinate arrays x,y (equidistant): 
    for (ix=0;ix<subdomain_mx; ix++) x[ix] = (ipx*subdomain_nx - halo_width + ix + 0.5)*dx;
    for (iy=0;iy<subdomain_my; iy++) y[iy] = (ipy*subdomain_ny - halo_width + iy + 0.5)*dy;

    // Initialisation of data.
    initcond(subdomain_nx, subdomain_ny, x, y, data);

    // Think about convenient data types to access non-contiguous portions of array data!
    // MPI_Type_vectors seem useful
    MPI_Datatype col_t, row_t;
    // number of rows blocks (without halo) each with size 2 and stride of number of elements in a row
    // (taking halo to account) to get next element from the same two columns.
    MPI_Type_vector(subdomain_nx, 2, subdomain_my, MPI_FLOAT, &col_t);
    // number of rows (2) of length (subdomain_ny) having stride of row length with halo
    // between the starts of the rows
    MPI_Type_vector(2, subdomain_ny, subdomain_my, MPI_FLOAT, &row_t);
    MPI_Type_commit(&row_t); MPI_Type_commit(&col_t);

    // Create MPI Window and buffer for one-sided communication 
    // Buffer has size enough for 2 columns and 2 rows of data from
    // neighbouring processes from the sides corresponding to the velocity
    // first element are for the 2 columns, and the rest for the 2 rows
    MPI_Win win;
    const int bsize = 2*subdomain_nx + 2*subdomain_ny;
    float ndata[bsize];
    MPI_Win_create(ndata, bsize*sizeof(float), sizeof(float), MPI_INFO_NULL, newComm, &win);

    unsigned int iterations = atoi(argv[5]);       // number of iterations=timesteps

    if (u_x==0 && u_y==0) {
      if (rank==0) printf("velocity=0 - no meaningful simulation!");
      exit(1);
    }

    // CFL condition for timestep:
    float dt = cdt*(u_x==0 ? (u_y==0 ? 0 : dy/u_y) : (u_y==0 ? dx/u_x : fmin(dx/u_x,dy/u_y)));

    if (rank==0) printf("dt= %f \n",dt);

    float t=0.;

    //Setup subdomain bounds
    int ixstart = halo_width;
    int iystart = halo_width;

    int ixstop = halo_width+subdomain_nx;
    int iystop = halo_width+subdomain_ny;

    // Consider proper synchronization measures!

    // Initialize timing!

    // whether we retrieve data from left (true) or right (false), and up or down
    // We use put here, so if left == true, we place data to the right and so on
    int left = u_x > 0 ? 1 : 0;
    int up = u_y > 0 ? 1 : 0;
    // First ranges that we can calculate without depending on the outside data
    int xrange_1[2] = {ixstart + ((int)up)*halo_width, ixstop - ((int)!up)*halo_width};
    int yrange_1[2] = {iystart + ((int)left)*halo_width, iystop - ((int)!left)*halo_width};
    // Second ranges that we can calculate after getting outside data
    int xrange_2[2] = up == 1 ? {ixstart, ixstart + halo_width} : {ixstop-halo_width, ixstop};
    int yrange_2[2] = left == 1 ? {iystart, iystart + halo_width} : {iystop-halo_width, iystop};

    FILE* fptr_approx = get_file_ptr("field_chunk_approximated_", rank);

    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        // Get the data from neighbors!
        // synchronize at first
        MPI_Win_fence(0, win);
        if(left) {
            // starting address to send to the process on the right. 
            // First row of non-halo data and the second last column of that data 
            // send starting from column: halo_width + (subdomain_ny - halo_width) = subdomain_ny
            float *start_addr = &data[halo_width][subdomain_ny];
            MPI_Put(start_addr, 1, col_t, nR_rank, 0, 2*subdomain_nx, MPI_FLOAT, win);
        } else {
            // put the data from the left side of the array to the left process
            float *start_addr = &data[halo_width][halo_width];
            MPI_Put(start_addr, 1, col_t, nL_rank, 0, 2*subdomain_nx, MPI_FLOAT, win);
        }
        if(up) {
            // send data down so 2 rows halo_width + subdomain_nx - halo_width =
            // subdomain_nx and the next one subdomain_nx + 1
            float *start_addr = &data[subdomain_nx][halo_width];
            // now offset of the column data being sent from some other process to the target
            MPI_Put(start_addr, 1, row_t, nD_rank, 2*subdomain_nx, 2*subdomain_ny, MPI_FLOAT, win);
        } else {
            // put the upper two rows of data in the upper-side process
            float *start_addr = &data[halo_width][halo_width];
            // now offset of the column data being sent from some other process to the target
            MPI_Put(start_addr, 1, row_t, nL_rank, 2*subdomain_nx, 2*subdomain_ny, MPI_FLOAT, win);
        }
        // Compute rhs. Think about concurrency of computation and data fetching by MPI_Get!
        // calculate first the subarray that is not affected by the data from other processes:

        rhs(xrange_1, yrange_1, subdomain_ny, data, d_data);
        
        // synchronize so that all data has been moved around inside a window
        MPI_Win_fence(0, win);
        // Data arrived -> compute stencils in all points that *are* affected by halo points.
        // first copy the data from the buffer attached to the window
        int offsetx = up == 1 ? 0 : subdomain_nx + halo_width;
        int offsety = left == 1 ? 0 : subdomain_ny + halo_width;

        for(int i = 0; i < 2; ++i) {
            for(int j = 0; j < subdomain_ny; ++j) {
                data[offsetx + i][halo_width + j] = ndata[2*subdomain_nx + i*subdomain_ny + j];
            }
        }
        for(int i = 0; i < subdomain_nx; ++i) {
            for(int j = 0; j < 2; ++j) {
                data[halo_width + i][offsety + j] = ndata[i*2 + j];
            }
        }

        rhs(xrange_2, yrange_2, subdomain_ny, data, d_data);

        // Update field in data using rhs in d_data (Euler's method):
        for (ix = ixstart; ix < ixstop; ++ix) {
            for (iy = iystart; iy < iystop; ++iy)
            {
                data[ix][iy] += dt*d_data[ix][iy];

                fprintf(fptr_approx,"%f ",data[ix][iy]);
            }
        }
        t = t+dt;
        fprintf(fptr_approx,"\n");
        // Output solution for checking/visualisation with choosable cadence!
    }
    fclose(fptr_approx);

    // Finalize timing!

    // analytic solution in array data_an:
    float data_an[iterations][subdomain_mx][subdomain_my], xshift[subdomain_mx], yshift[subdomain_my];

    // Construct file name for data chunk of process.
    FILE* fptr_analytical = get_file_ptr("field_chunk_analytical_",rank);

    t=0.;
    for (int iter=0; iter<iterations; iter++) {
        for (ix=0;ix<subdomain_mx;ix++) xshift[ix] = x[ix] - u_x*t;
        for (iy=0;iy<subdomain_my;iy++) yshift[iy] = y[iy] - u_y*t;

        initcond(subdomain_nx, subdomain_ny, xshift, yshift, (float (*)[subdomain_my]) &data_an[iter][0][0]);

        if (u_y==0.) {
          for (int ix=ixstart; ix < ixstop; ++ix) fprintf(fptr_analytical,"%f ",data_an[iter][ix][iystart]);
	    }
        else if(u_x == 0.) {
          for (int iy=iystart; iy < iystop; ++iy) fprintf(fptr_analytical,"%f ",data_an[iter][ixstart][iy]);
	    } else {
          for (int ix=ixstart; ix < ixstop; ++ix)  {
            for (int iy=iystart; iy < iystop; ++iy) {
                fprintf(fptr_analytical,"%f ",data_an[iter][ix][iy]);
            }
          }
	    }
      fprintf(fptr_analytical,"\n");
      t += dt;
    }
    fclose(fptr_analytical);

    MPI_Win_free(&win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;

}
