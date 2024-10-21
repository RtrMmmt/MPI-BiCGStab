/******************************************************************************
 * macでのコンパイルと実行コマンド
 * mpicc -O3 src/test_shifted.c src/shifted_solver.c src/matrix.c src/vector.c src/mmio.c -I src -lm
 * mpirun -np 4 ./a.out data/atmosmodd.mtx
 ******************************************************************************/

#include "shifted_solver.h"

#define DISPLAY_NODE_INFO   /* ノード数とプロセス数の表示 */

#define SIGMA_LENGTH 100
#define SEED 0

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

    char *filename = argv[1];

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
    int sigma_len = SIGMA_LENGTH;
    double sigma[sigma_len];
    int seed = SEED;

    for (int i = 0; i < sigma_len; ++i) {
        sigma[i] = i * 0.01 + 0.01;
    }
    double *x_loc_set, *r_loc, *x, *r;
    int vec_size = A_info.rows;
    int vec_loc_size = A_loc_diag->rows;
    x_loc_set = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    r_loc = (double *)malloc(vec_loc_size * sizeof(double));
    x = (double *)malloc(vec_size * sizeof(double));
    r = (double *)malloc(vec_size * sizeof(double));

    for (int i = 0; i < vec_loc_size; i++) {
        x_loc_set[seed * vec_loc_size + i] = 1; /* 厳密解はすべて1 */
    }

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, &x_loc_set[seed * vec_loc_size], x, r_loc);
    my_daxpy(vec_loc_size, sigma[seed], &x_loc_set[seed * vec_loc_size], r_loc);

    double *ans_loc = (double *)malloc(vec_loc_size * sizeof(double));
    my_dcopy(vec_loc_size, r_loc, ans_loc);

    for (int i = 0; i < vec_loc_size * sigma_len; i++) {
        x_loc_set[i] = 0; /* 初期値はすべて0 */
    }

    int total_iter;
    /* 実行 */
    //total_iter = shifted_bicgstab(A_loc_diag, A_loc_offd, &A_info, x_loc_set, r_loc, sigma, sigma_len);
    total_iter = shifted_lopbicgstab(A_loc_diag, A_loc_offd, &A_info, x_loc_set, r_loc, sigma, sigma_len, seed);
    //total_iter = shifted_lopbicgstab_v2(A_loc_diag, A_loc_offd, &A_info, x_loc_set, r_loc, sigma, sigma_len, seed);

    for (int i = 0; i < sigma_len; i++) {
        //if (i != 0) csr_shift_diagonal(A_loc_diag, 0.01);
        //MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, &x_loc_set[i * vec_loc_size], x, r_loc);
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, &x_loc_set[i * vec_loc_size], x, r_loc);
        my_daxpy(vec_loc_size, sigma[i], &x_loc_set[i * vec_loc_size], r_loc);

        double diff;
        double local_diff_norm_2 = 0;
        double local_ans_norm_2 = 0;
        for (int j = 0; j < vec_loc_size; j++) {
            diff = r_loc[j] - ans_loc[j];
            local_diff_norm_2 += diff * diff;
            local_ans_norm_2 += ans_loc[j] * ans_loc[j];
        }
        double global_diff_norm_2, global_ans_norm_2;
        MPI_Allreduce(&local_diff_norm_2, &global_diff_norm_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_ans_norm_2, &global_ans_norm_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double rerative_error = sqrt(global_diff_norm_2) / sqrt(global_ans_norm_2);
        if (myid == 0) {
            if (i == seed) printf("#seed: %.2f, relative error: %e\n", sigma[i], rerative_error);
            else printf("sigma: %.2f, relative error: %e\n", sigma[i], rerative_error);
        }
    }

	csr_free_matrix(A_loc_diag); free(A_loc_diag);
    csr_free_matrix(A_loc_offd); free(A_loc_offd);
    free(x_loc_set); free(r_loc); free(x); free(r);
    free(A_info.recvcounts);  free(A_info.displs);

	MPI_Finalize();
	return 0;
}
