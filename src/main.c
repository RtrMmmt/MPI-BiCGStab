/******************************************************************************
 * macでのコンパイルと実行コマンド
 * mpicc -O3 src/main.c src/solver.c src/matrix.c src/vector.c src/mmio.c -I src -lm
 * mpirun -np 4 ./a.out data/atmosmodd.mtx bicgstab
 * mpirun -np 4 ./a.out data/atmosmodd.mtx pipe_bicgstab_rr 30 6
 ******************************************************************************/

#include "solver.h"

#define DISPLAY_NODE_INFO   /* ノード数とプロセス数の表示 */

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int numprocs, myid, namelen;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(proc_name, &namelen);

#ifdef DISPLAY_NODE_INFO
    /* ノード数とプロセス数をカウント */
    char *all_proc_names = NULL;
    if (myid == 0) {
        all_proc_names = (char *)malloc(numprocs * MPI_MAX_PROCESSOR_NAME * sizeof(char));
    }
    MPI_Gather(proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_proc_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        int *proc_count_per_node = (int *)calloc(numprocs, sizeof(int));
        int unique_nodes = 0;

        for (int i = 0; i < numprocs; i++) {
            int found = 0;
            for (int j = 0; j < unique_nodes; j++) {
                if (strncmp(&all_proc_names[i * MPI_MAX_PROCESSOR_NAME], 
                            &all_proc_names[j * MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME) == 0) {
                    proc_count_per_node[j]++;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                strncpy(&all_proc_names[unique_nodes * MPI_MAX_PROCESSOR_NAME], 
                        &all_proc_names[i * MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME);
                proc_count_per_node[unique_nodes]++;
                unique_nodes++;
            }
        }

        printf("Node: %d, Proc: %d\n", unique_nodes, numprocs);

        free(proc_count_per_node);
    }

    if (all_proc_names != NULL) {
        free(all_proc_names);
    }
#endif

    double start_time, end_time, total_time;

    if (argc < 3) {
        if (myid == 0) {
            printf("Usage: %s <matrix file> <method> [options]\n", argv[0]);
            printf("Methods:\n");
            printf("  bicgstab\n");
            printf("  ca_bicgstab\n");
            printf("  pipe_bicgstab\n");
            printf("  pipe_bicgstab_rr <r> <s>\n");
        }
        MPI_Finalize();
        return 1;
    }

    char *filename = argv[1];
    char *method = argv[2];

    /* 行列の初期化 */
    INFO_Matrix A_info;
    A_info.recvcounts = (int *)malloc(numprocs * sizeof(int));
    A_info.displs = (int *)malloc(numprocs * sizeof(int));
	CSR_Matrix *A_loc_diag = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
    CSR_Matrix *A_loc_offd = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
	csr_init_matrix(A_loc_diag);
    csr_init_matrix(A_loc_offd);

    /* 行列の読み取り */
    start_time = MPI_Wtime();
    MPI_csr_load_matrix_block(filename, A_loc_diag, A_loc_offd, &A_info);
    end_time = MPI_Wtime();
    if (myid == 0) printf("IO time      : %e [sec.]\n", end_time - start_time);

    if (A_info.cols != A_info.rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    /* ベクトルの初期化 */
    double *x_loc, *r_loc, *x, *r;
    int vec_size = A_info.rows;
    int vec_loc_size = A_loc_diag->rows;
    x_loc = (double *)malloc(vec_loc_size * sizeof(double));
    r_loc = (double *)malloc(vec_loc_size * sizeof(double));
    x = (double *)malloc(vec_size * sizeof(double));
    r = (double *)malloc(vec_size * sizeof(double));

    for (int i = 0; i < vec_loc_size; i++) {
        x_loc[i] = 1; /* 厳密解はすべて1 */
    }

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);

    for (int i = 0; i < vec_loc_size; i++) {
        x_loc[i] = 0; /* 初期値はすべて0 */
    }

    int total_iter;

    /* 実行 */
    if (strcmp(method, "bicgstab") == 0) {
        total_iter = bicgstab(A_loc_diag, A_loc_offd, &A_info, x_loc, r_loc);
    } else if (strcmp(method, "ca_bicgstab") == 0) {
        total_iter = ca_bicgstab(A_loc_diag, A_loc_offd, &A_info, x_loc, r_loc);
    } else if (strcmp(method, "pipe_bicgstab") == 0) {
        total_iter = pipe_bicgstab(A_loc_diag, A_loc_offd, &A_info, x_loc, r_loc);
    } else if (strcmp(method, "pipe_bicgstab_rr") == 0) {
        if (argc != 5) {
            if (myid == 0) printf("Usage for pipe_bicgstab_rr: %s <matrix file> pipe_bicgstab_rr <r> <s>\n", argv[0]);
            MPI_Finalize();
            return 1;
        }
        int krr = atoi(argv[3]);
        int nrr = atoi(argv[4]);
        total_iter = pipe_bicgstab_rr(A_loc_diag, A_loc_offd, &A_info, x_loc, r_loc, krr, nrr);
    } else {
        if (myid == 0) printf("Unknown method: %s\n", method);
        MPI_Finalize();
        return 1;
    }

    //if (myid == 0) printf("x[0] = %f, Total iter: %d\n", x_loc[0], total_iter);

	csr_free_matrix(A_loc_diag); free(A_loc_diag);
    csr_free_matrix(A_loc_offd); free(A_loc_offd);
    free(x_loc); free(r_loc); free(x); free(r);
    free(A_info.recvcounts);  free(A_info.displs);

	MPI_Finalize();
	return 0;
}
