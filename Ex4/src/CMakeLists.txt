# Single
add_executable(reduce-single main.cu reduce-single.cu)

# Multi
add_executable(reduce-multi main.cu reduce-multi.cu)

# MPI
find_package(MPI REQUIRED)
add_executable(reduce-mpi main.cu reduce-mpi.cu)
target_link_libraries(reduce-mpi MPI::MPI_C)
target_compile_options(reduce-mpi PRIVATE -DUSE_MPI=1)

add_executable(quicksort-gpu qs_main.cu  quicksort_gpu.cu)
add_executable(quicksort-distributed-gpu qs_main.cu quicksort_distributed_gpu.cu)

target_link_libraries(quicksort-distributed-gpu MPI::MPI_C)

target_compile_options(quicksort-gpu PRIVATE -DGPU=1 -DDISTRIBUTED=0)
target_compile_options(quicksort-distributed-gpu PRIVATE -DGPU=1 -DDISTRIBUTED=1)
