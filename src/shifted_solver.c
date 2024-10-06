#include "shifted_solver.h"

#define IDX(set, i, rows) ((set) + (i) * (rows))

#define EPS 1.0e-12   /* 収束判定条件 */
#define MAX_ITER 500 /* 最大反復回数 */

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
    double max_xi;

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
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   /* r# <- r = b */
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);
        beta_set[i] = 0.0;
        tau_set[i] = 1.0;
        xi_old_set[i] = 1.0;
        xi_curr_set[i] = 1.0;
        xi_new_set[i] = 1.0;
    }
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;
    dot_zero = rTr;
    max_xi = 1.0;

    while (max_xi * max_xi * dot_r > tol * tol * dot_zero && k < max_iter) {
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[0], vec, s_loc);  /* s <- A p0 */
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  /* rTs <- (r#,s) */
        MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        my_dcopy(vec_loc_size, r_loc, r_old_loc);
        alpha_old = alpha_set[0];
        beta_old = beta_set[0];
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha_set[0] = rTr / rTs;   /* alpha0 <- (r#,r)/(r#,s) */
        for (j = 1; j < sigma_len; j++) {
            xi_new_set[j] = (xi_curr_set[j] * xi_old_set[j] * alpha_old) / (alpha_set[0] * beta_old * (xi_old_set[j] - xi_curr_set[j]) + xi_old_set[j] * alpha_old * (1.0 + alpha_set[0] * sigma[j]));
            alpha_set[j] = xi_new_set[j] / (xi_curr_set[j] * alpha_set[0]);
        }
        my_daxpy(vec_loc_size, -alpha_set[0], s_loc, r_loc);   /* q <- r - alpha0 s */

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  /* y <- A q */
        rTy = my_ddot(vec_loc_size, r_loc, y_loc);  /* (q,y) */
        MPI_Iallreduce(MPI_IN_PLACE, &rTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTy_req);
        yTy = my_ddot(vec_loc_size, y_loc, y_loc);  /* (y,y) */
        MPI_Iallreduce(MPI_IN_PLACE, &yTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &yTy_req);
        MPI_Wait(&rTy_req, MPI_STATUS_IGNORE);
        MPI_Wait(&yTy_req, MPI_STATUS_IGNORE);

        omega_set[0] = rTy / yTy;  /* omega <- (q,y)/(y,y) */
        for (j = 1; j < sigma_len; j++) {
            omega_set[j] = omega_set[0] / (1.0 + omega_set[0] * sigma[j]);
            my_daxpy(vec_loc_size, omega_set[j] * tau_set[j] * xi_new_set[j], r_loc, x_loc_set + j * vec_loc_size);
            my_daxpy(vec_loc_size, alpha_set[0], p_loc_set + j * vec_loc_size, x_loc_set + j * vec_loc_size);
            tau_set[j] = tau_set[j] / (1.0 + omega_set[0] * sigma[j]);
            my_daxpy(vec_loc_size, omega_set[j] * tau_set[j] * xi_new_set[j] / alpha_set[j], r_loc, p_loc_set + j * vec_loc_size);
            my_daxpy(vec_loc_size, - omega_set[j] * tau_set[j] * xi_curr_set[j] / alpha_set[j], r_old_loc, p_loc_set + j * vec_loc_size);
        }
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
        max_xi = 1.0;

        for (j = 1; j < sigma_len; j++) {
            beta_set[j] = pow(xi_new_set[j] / xi_curr_set[j], 2) * beta_set[0];
            my_dscal(vec_loc_size, beta_set[j], p_loc_set + j * vec_loc_size);
            my_daxpy(vec_loc_size, tau_set[j] * xi_new_set[j], r_loc, p_loc_set + j * vec_loc_size);
            xi_old_set[j] = xi_curr_set[j];
            xi_curr_set[j] = xi_new_set[j];
            if (fabs(xi_curr_set[j]) > max_xi) {
                max_xi = fabs(xi_curr_set[j]);
            }
        }

        my_dscal(vec_loc_size, beta_set[0], p_loc_set);                         /* p0 <- r + beta p0 - beta omega s */
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc_set);                          /*                                */
        my_daxpy(vec_loc_size, -beta_set[0]*omega_set[0], s_loc, p_loc_set);    /* ------------------------------ */

        k++;

#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e, Max_Xi: %e\n", k, sqrt(dot_r / dot_zero), max_xi);
        }
#endif

    }


#ifdef MEASURE_TIME
        end_time = MPI_Wtime();
        total_time = end_time - start_time;
#endif

    if (myid == 0) {
        printf("Total iter   : %d\n", k);
        printf("Final r      : %e\n", sqrt(dot_r / dot_zero));
#ifdef MEASURE_TIME
        printf("Total time   : %e [sec.] \n", total_time);
        printf("Avg time/iter: %e [sec.] \n", total_time / k);
#endif
    }

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(tau_set); free(xi_old_set); free(xi_curr_set); free(xi_new_set);

    return k;

}