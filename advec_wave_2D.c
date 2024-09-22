#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

static const int halo_width=2;    // fill in halo_width
static int rank;

const float pi=3.14159;
const float u_x=0.5 , u_y=0.5 , c_amp=1  // choose velocity components and amplitude of initial condition
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
    sum_x *= abs(u_x)/dx;

    inc = -copysign(1.0, u_y);
    for (int k=0; k<=halo_width; k++){
        sum_y += coeff[k]*data[i][j+inc*k];
    }
    sum_y *= abs(u_y)/dy;

    return sum_x + sum_y;
}

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

void rhs(const int xrange[2], const int yrange[2], int ny, float data[][ny], float d_data[][ny])
{
    //Right-hand side d_data of pde for field in data for a subdomain defined by xrange, yrange:
    int ix,iy;

    for (ix = xrange[0]; ix < xrange[1]; ++ix)
        for (iy = yrange[0]; iy < yrange[1]; ++iy)
        {
            d_data[ix][iy] = ugrad_upw(ix, iy, ny, data);
        }
}

void fill_wbuffer(float *wbuf, const float *data, int rangex[2], int rangey[2]) {

    int dif1 = rangex[1]-rangex[0]+1;
    int dif2 = rangey[1]-rangey[0]+1;

    for(int i = 0; i < dif1, ++i) {
        for(int j = 0; j < dif2; ++j) {
            int ind = i*(dif2) + j;
            wbuf[ind] = data[rangex[0] + i][rangey[0] + j]
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

    int *proc_coords = find_proc_coords(rank,nprocx,nprocy);
    ipx=proc_coords[0]; ipy=proc_coords[1];

    // Find neighboring processes!

    int domain_nx = atoi(argv[3]),                 // number of gridpoints in x direction
        subdomain_nx = domain_nx/nprocx;                            // subdomain x-size w/o halos
        subdomain_mx = subdomain_nx + 2*halo_width;                            //                  with halos

    int domain_ny = atoi(argv[4]),                 // number of gridpoints in y direction
        subdomain_ny = domain_ny/nprocy;                            // subdomain y-size w/o halos
        subdomain_my = subdomain_ny + 2*halo_width;                        //                  with halos


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


    // Perhaps vector of (2, sumbdomain_nx) size for the right and left sides, (as subdomain_nx corresponds to number of rows)
    // and vector of (subdomain_ny, 2) size for the top and bottom sides (as subdomain_ny corresponds to number of columns)
    /*
    float trows[2*subdomain_ny], brows[2*subdomain_ny], rcols[2*subdomain_nx], lcols[2*subdomain_nx];
    int rx1 = {halo_width, halo_width + 1}, ry1 = {halo_width, halo_width + 1}
    int rx2 = {subdomain_mx - 1 - halo_width - 1, subdomain_mx - 1 - halo_width}, ry2 = {subdomain_my - 1 - halo_width - 1, subdomain_my - 1 - halo_width};
    int rx3 = {halo_width, subdomain_mx - 1 - halo_width}, ry3 = {halo_width, subdomain_my - 1 - halo_width};

    // fill the buffers with corresponding data
    fill_wbuffer(trows, data, rx1, ry3); fill_wbuffer(brows, data, rx2, ry3); fill_wbuffer(lcols, data, rx3, ry1); fill_wbuffer(rcols, data, rx3, ry2);

    */

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

    //Create MPI Windows for one-sided communication 
    MPI_Win win;
    MPI_Win_create(data, subdomain_mx*subdomain_my*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win)

    unsigned int iterations = atoi(argv[5]);       // number of iterations=timesteps

    if (u_x==0 && u_y==0) {
      if (rank==0) printf("velocity=0 - no meaningful simulation!");
      exit(1);
    }

    // CFL condition for timestep:
    float dt = cdt*(u_x==0 ? (u_y==0 ? 0 : dy/u_y) : (u_y==0 ? dx/u_x : fmin(dx/u_x,dy/u_y)));

    if (rank==0) printf("dt= %f \n",dt);

    float t=0.;

    // Consider proper synchronization measures!

    // Initialize timing!


    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        // Get the data from neighbors!

        // Compute rhs. Think about concurrency of computation and data fetching by MPI_Get!

        // Data arrived -> compute stencils in all points that *are* affected by halo points.

        // Update field in data using rhs in d_data (Euler's method):
        for (ix = ixstart; ix < ixstop; ++ix) {
            for (iy = iystart; iy < iystop; ++iy)
            {
                data[ix][iy] += dt*d_data[ix][iy];
            }
        }
        t = t+dt;

        // Output solution for checking/visualisation with choosable cadence!
    }

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

        if (u_y==0.)
          for (int ix=ixstart; ix < ixstop; ++ix) fprintf(fptr_analytical,"%f ",data_an[iter][ix][iystart]);
        else if(u_x == 0.)
          for (int iy=iystart; iy < iystop; ++iy) fprintf(fptr_analytical,"%f ",data_an[iter][ixstart][iy]);
        else
          for (int ix=ixstart; ix < ixstop; ++ix) fprintf(fptr_analytical,"%f ",data_an[iter][ix][iystart]);
            for (int iy=iystart; iy < iystop; ++iy) fprintf(fptr_analytical,"%f ",data_an[iter][ixstart][iy]);
		fprintf(fptr_analytical,"%f ",data_an[iter][ix][iy]);
      fprintf(fptr_analytical,"\n");
      t += dt;
    }
    fclose(fptr_analytical);

    MPI_Win_free(&win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
