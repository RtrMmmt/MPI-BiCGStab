#include "shifted_solver.h"

#define EPS 1.0e-15   /* 収束判定条件 */
#define MAX_ITER 1000 /* 最大反復回数 */

#define MEASURE_TIME /* 時間計測 */

#define DISPLAY_RESIDUAL /* 残差表示 */
#define OUT_ITER 100     /* 残差の表示間隔 */

int shifted_bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *X_loc, double *r_loc, double *sigma, int sigma_len) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc_diag->rows;

    int k, max_iter;
    double tol;
    double *r_hat_loc, *q_loc, *s_loc, *y_loc, *vec;

    

    double dot_r, dot_zero, rTr, rTs, qTy, yTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, rTy_req, yTy_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    r_hat_loc = (double *)malloc(vec_loc_size * sizeof(double));
    q_loc = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc = (double *)malloc(vec_loc_size * sizeof(double));

    vec = (double *)malloc(vec_size * sizeof(double));


}