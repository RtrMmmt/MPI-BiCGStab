// mpicc -O3 src/test.c src/solver.c src/matrix.c src/vector.c src/mmio.c -I src -lm
// mpirun -np 4 ./a.out data/atmosmodd.mtx bicgstab
// mpirun -np 4 ./a.out data/atmosmodd.mtx pipe_bicgstab_rr 30 6

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

	CSR_Matrix *A_loc_diag = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
    CSR_Matrix *A_loc_offd = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
	csr_init_matrix(A_loc_diag);
    csr_init_matrix(A_loc_offd);

    MPI_csr_load_matrix_block(filename, A_loc_diag, A_loc_offd, &A_info);

    if (A_info.cols != A_info.rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    double *x_loc, *r_loc, *x, *r;
    int vec_size = A_info.rows;
    int vec_loc_size = A_loc_diag->rows;
    x_loc = (double *)malloc(vec_loc_size * sizeof(double));
    r_loc = (double *)malloc(vec_loc_size * sizeof(double));
    x = (double *)malloc(vec_size * sizeof(double));
    r = (double *)malloc(vec_size * sizeof(double));

    for (int i = 0; i < vec_loc_size; i++) x_loc[i] = 1; // 厳密解はすべて1
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);
    for (int i = 0; i < vec_loc_size; i++) x_loc[i] = 0; // 初期値はすべて0

    int total_iter;

    if(myid == 0) printf("bicgstab\n");
    total_iter = bicgstab(A_loc_diag, A_loc_offd, &A_info, x_loc, r_loc);

    for (int i = 0; i < vec_loc_size; i++) x_loc[i] = 1; // 厳密解はすべて1
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);
    for (int i = 0; i < vec_loc_size; i++) x_loc[i] = 0; // 初期値はすべて0

    if(myid == 0) printf("ca_bicgstab\n");
    total_iter = ca_bicgstab(A_loc_diag, A_loc_offd, &A_info, x_loc, r_loc);

    for (int i = 0; i < vec_loc_size; i++) x_loc[i] = 1; // 厳密解はすべて1
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);
    for (int i = 0; i < vec_loc_size; i++) x_loc[i] = 0; // 初期値はすべて0

    if(myid == 0) printf("pipe_bicgstab\n");
    total_iter = pipe_bicgstab(A_loc_diag, A_loc_offd, &A_info, x_loc, r_loc);

    for (int i = 0; i < vec_loc_size; i++) x_loc[i] = 1; // 厳密解はすべて1
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);
    for (int i = 0; i < vec_loc_size; i++) x_loc[i] = 0; // 初期値はすべて0

    if(myid == 0) printf("pipe_bicgstab_rr\n");
    total_iter = pipe_bicgstab_rr(A_loc_diag, A_loc_offd, &A_info, x_loc, r_loc, 30, 6); 

	csr_free_matrix(A_loc_diag); free(A_loc_diag);
    csr_free_matrix(A_loc_offd); free(A_loc_offd);
    free(x_loc); free(r_loc); free(x); free(r);
    free(A_info.recvcounts);  free(A_info.displs);

	MPI_Finalize();
	return 0;
}