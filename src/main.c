// mpicc -O3 src/main.c src/solver.c src/matrix.c src/vector.c src/mmio.c -I src
// mpirun -np 4 ./a.out data/epb3.mtx

#include "solver.h"

int main(int argc, char *argv[]) {
    
    MPI_Init(&argc, &argv);

    int numprocs, myid, namelen;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(proc_name, &namelen);

    char *filename = argv[1];

    INFO_Matrix A_info;

    A_info.recvcounts = (int *)malloc(numprocs * sizeof(int));
    A_info.displs = (int *)malloc(numprocs * sizeof(int));

	CSR_Matrix *A_loc = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
	csr_init_matrix(A_loc);

    MPI_csr_load_matrix(filename, A_loc, &A_info);

    if (A_info.cols != A_info.rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    double *x_loc, *r_loc, *x, *r;
    int vec_size = A_info.rows;
    int vec_loc_size = A_loc->rows;
    x_loc = (double *)malloc(vec_loc_size * sizeof(double));
    r_loc = (double *)malloc(vec_loc_size * sizeof(double));
    x = (double *)malloc(vec_size * sizeof(double));
    r = (double *)malloc(vec_size * sizeof(double));

    for (int i = 0; i < vec_size; i++) {
        x[i] = 1; // 厳密解はすべて1
    }

    csr_mv(A_loc, x, r_loc);

    MPI_Allgatherv(r_loc, vec_loc_size, MPI_DOUBLE, r, A_info.recvcounts, A_info.displs, MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < vec_size; i++) {
        x[i] = 0; // 初期値はすべて0
    }

    int total_iter;

    //total_iter = bicgstab(A_loc, &A_info, x, r);
    //total_iter = ca_bicgstab(A_loc, &A_info, x, r);
    //total_iter = pipe_bicgstab(A_loc, &A_info, x, r);
    total_iter = pipe_bicgstab_rr(A_loc, &A_info, x, r, 30, 6);

    if (myid == 0) printf("x[0] = %f, Total iter: %d\n", x[0], total_iter);

	csr_free_matrix(A_loc); free(A_loc);
    free(x_loc); free(r_loc); free(x); free(r);
    free(A_info.recvcounts);  free(A_info.displs);

	MPI_Finalize();
	return 0;
}