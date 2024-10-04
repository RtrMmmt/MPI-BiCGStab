#include "shifted_solver.h"

#define IDX(set, i, rows) ((set) + (i) * (rows))

#define EPS 1.0e-15   /* 収束判定条件 */
#define MAX_ITER 1000 /* 最大反復回数 */

#define MEASURE_TIME /* 時間計測 */

#define DISPLAY_RESIDUAL /* 残差表示 */
#define OUT_ITER 100     /* 残差の表示間隔 */

int shifted_bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#ifdef MEASURE_TIME
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc_diag->rows;

    int i, j;

    int k, max_iter;
    double tol;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *tau_set, *xi_old_set, *xi_curr_set, *xi_new_set;
    double alpha_old, beta_old;

    double dot_r, dot_zero, rTr, rTs, rTy, yTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, rTy_req, yTy_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    r_old_loc = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc = (double *)malloc(vec_loc_size * sizeof(double));

    vec = (double *)malloc(vec_size * sizeof(double));

    p_loc_set = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set = (double *)malloc(sigma_len * sizeof(double));
    beta_set = (double *)malloc(sigma_len * sizeof(double));
    omega_set = (double *)malloc(sigma_len * sizeof(double));
    tau_set = (double *)malloc(sigma_len * sizeof(double));
    xi_old_set = (double *)malloc(sigma_len * sizeof(double));
    xi_curr_set = (double *)malloc(sigma_len * sizeof(double));
    xi_new_set = (double *)malloc(sigma_len * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      /* (r#,r) */
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   /* r_hat = r = b */
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, p_loc_set + i * vec_loc_size);
        beta_set[i] = 0.0;
        tau_set[i] = 1.0;
        xi_old_set[i] = 1.0;
        xi_curr_set[i] = 1.0;
        xi_new_set[i] = 1.0;
    }
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    while (dot_r > tol * tol * dot_zero && k < max_iter) {
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, p_loc_set, vec, s_loc);  /* s <- A p0 */
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  /* rTs <- (r#,s) */
        my_dcopy(vec_loc_size, r_loc, r_old_loc);
        alpha_old = alpha_set[0];
        beta_old = beta_set[0];
        MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);

        alpha_set[0] = rTr / rTs;   /* alpha0 <- (r#,r)/(r#,s) */
        my_daxpy(vec_loc_size, -alpha_set[0], s_loc, r_loc);   /* q <- r - alpha0 s */

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  /* y <- A q */
        rTy = my_ddot(vec_loc_size, r_loc, y_loc);  /* (q,y) */
        MPI_Iallreduce(MPI_IN_PLACE, &rTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTy_req);
        yTy = my_ddot(vec_loc_size, y_loc, y_loc);  /* (y,y) */
        MPI_Iallreduce(MPI_IN_PLACE, &yTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &yTy_req);
        MPI_Wait(&rTy_req, MPI_STATUS_IGNORE);
        MPI_Wait(&yTy_req, MPI_STATUS_IGNORE);

        omega_set[0] = rTy / yTy;  /* omega <- (q,y)/(y,y) */
        my_daxpy(vec_loc_size, alpha_set[0], p_loc_set, x_loc_set);     /* x0 <- x0 + alpha0 p0 + omega q */
        my_daxpy(vec_loc_size, omega_set[0], r_loc, x_loc_set);         /* ------------------------------ */
        my_daxpy(vec_loc_size, -omega_set[0], y_loc, r_loc);            /* r <- q - omega0 y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    /* (r,r) */
        MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  /* (r#,r) */
        MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta_set[0] = (alpha_set[0] / omega_set[0]) * (rTr / rTr_old);   /* beta0 <- (alpha0 / omega0) * ((r#,r)/(r#,r)) */
        my_dscal(vec_loc_size, beta_set[0], p_loc_set);                         /* p <- r + beta p - beta omega s */
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc_set);                          /*                                */
        my_daxpy(vec_loc_size, -beta_set[0]*omega_set[0], s_loc, p_loc_set);    /* ------------------------------ */
        k++;

        rTr_old = rTr;




    }
    




}