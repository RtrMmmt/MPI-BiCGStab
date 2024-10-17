#include "shifted_solver.h"

#define IDX(set, i, rows) ((set) + (i) * (rows))

#define EPS 1.0e-12   /* 収束判定条件 */
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
    double max_xi, abs_xi;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *q_loc, *vec;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *tau_set, *xi_old_set, *xi_curr_set, *xi_new_set;
    double alpha_old, beta_old;

    double dot_r, dot_zero, rTr, rTs, rTy, yTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, rTy_req, yTy_req;
    MPI_Request reqs1[2];
    MPI_Request reqs2[2];

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    q_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    tau_set     = (double *)malloc(sigma_len * sizeof(double));
    xi_old_set  = (double *)malloc(sigma_len * sizeof(double));
    xi_curr_set = (double *)malloc(sigma_len * sizeof(double));
    xi_new_set  = (double *)malloc(sigma_len * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      /* (r#,r) */
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   /* r# <- r = b */
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    /* p[sigma] <- b */
        beta_set[i] = 0.0;      /* beta[sigma] <- 0 */
        alpha_set[i] = 1.0;     /* alpha[sigma] <- 1 */
        xi_old_set[i] = 1.0;    /* xi_old[sigma] <- 1 */
        xi_curr_set[i] = 1.0;   /* xi_curr[sigma] <- 1 */
        tau_set[i] = 1.0;       /* tau[sigma] <- 1 */
    }
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;    /* (r,r) */
    dot_zero = rTr; /* (r#,r#) */
    max_xi = 1.0;   /* max(|xi_curr|) */

    while (max_xi * max_xi * dot_r > tol * tol * dot_zero && k < max_iter) {

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[0], vec, s_loc);  /* s <- A p[0] */
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  /* rTs <- (r#,s) */
        for (j = 1; j < sigma_len; j++) {
            beta_set[j] = (xi_curr_set[j] / xi_old_set[j]) * (xi_curr_set[j] / xi_old_set[j]) * beta_set[0]; /* beta[sigma] <- (xi_new[sigma] / xi_curr[sigma])^2 beta[0] */
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      /* p[sigma] <- tau[sigma] xi_new[sigma] r +  beta[sigma] p[sigma] */
            my_daxpy(vec_loc_size, tau_set[j] * xi_curr_set[j], r_loc, &p_loc_set[j * vec_loc_size]);
        }
        my_dcopy(vec_loc_size, r_loc, r_old_loc);   /* r_old <- r */
        alpha_old = alpha_set[0];       /* alpha_old <- alpha[0] */
        beta_old = beta_set[0];         /* beta_old <- beta[0] */
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha_set[0] = rTr / rTs;   /* alpha[0] <- (r#,r)/(r#,s) */
        my_daxpy(vec_loc_size, -alpha_set[0], s_loc, r_loc);   /* q <- r - alpha[0] s */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  /* y <- A q */

        rTy = my_ddot(vec_loc_size, r_loc, y_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTy_req);  /* (q,y) */
        yTy = my_ddot(vec_loc_size, y_loc, y_loc); MPI_Iallreduce(MPI_IN_PLACE, &yTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &yTy_req);  /* (y,y) */
        for (j = 1; j < sigma_len; j++) {
            xi_new_set[j] = (xi_curr_set[j] * xi_old_set[j] * alpha_old) / (alpha_set[0] * beta_old * (xi_old_set[j] - xi_curr_set[j]) + xi_old_set[j] * alpha_old * (1.0 + alpha_set[0] * sigma[j]));
            /* xi_new[sigma] = (xi_curr[sigma] * xi_old[sigma] * alpha_old) / (alpha[0] * beta_old * (xi_old[sigma] - xi_curr[sigma]) + xi_old[sigma] * alpha_old * (1.0 + alpha[0] * sigma[sigma])) */
            alpha_set[j] = (xi_new_set[j] / xi_curr_set[j]) * alpha_set[0];     /* alpha[sigma] <- (xi_new[sigma] / xi_curr[sigma]) alpha[0] */
        }
        MPI_Wait(&rTy_req, MPI_STATUS_IGNORE);
        MPI_Wait(&yTy_req, MPI_STATUS_IGNORE);

        omega_set[0] = rTy / yTy;  /* omega[0] <- (q,y)/(y,y) */
        my_daxpy(vec_loc_size, alpha_set[0], &p_loc_set[0], x_loc_set);     /* x[0] <- x[0] + alpha[0] p[0] + omega[0] q */
        my_daxpy(vec_loc_size, omega_set[0], r_loc, x_loc_set);
        my_dcopy(vec_loc_size, r_loc, q_loc);   /* q <- r */
        my_daxpy(vec_loc_size, -omega_set[0], y_loc, r_loc);            /* r <- q - omega[0] y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  /* (r,r) */
        rTr_old = rTr;      /* r_old <- (r#,r) */
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  /* (r#,r) */
        for (j = 1; j < sigma_len; j++) {
            omega_set[j] = omega_set[0] / (1.0 + omega_set[0] * sigma[j]);      /* omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) */
            my_daxpy(vec_loc_size, omega_set[j] * tau_set[j] * xi_new_set[j], q_loc, &x_loc_set[j * vec_loc_size]);     /* x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] tau[sigma] xi_new[sigma] q */
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] * tau_set[j] * xi_new_set[j] / alpha_set[j], q_loc, &p_loc_set[j * vec_loc_size]);      /* p[sigma] <- p[sigma] + omega[sigma] tau[sigma] / alpha[sigma] (xi_new[sigma] q - xi_curr[sigma] r_old) */
            my_daxpy(vec_loc_size, - omega_set[j] * tau_set[j] * xi_curr_set[j] / alpha_set[j], r_old_loc, &p_loc_set[j * vec_loc_size]);
            tau_set[j] = tau_set[j] / (1.0 + omega_set[0] * sigma[j]);      /* tau[sigma] <- tau[sigma] / (1 + omega[0] sigma) */
        }
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta_set[0] = (alpha_set[0] / omega_set[0]) * (rTr / rTr_old);   /* beta[0] <- (alpha[0] / omega[0]) ((r#,r)/(r#,r)) */
        max_xi = 1.0;
        for (j = 1; j < sigma_len; j++) {
            abs_xi = fabs(xi_curr_set[j] * tau_set[j]);
            if (abs_xi > max_xi) max_xi = abs_xi;   /* max(|xi_curr|) */
        }
        my_dcopy(sigma_len, xi_curr_set, xi_old_set);   /* xi_old[sigma] <- xi_curr[sigma] */
        my_dcopy(sigma_len, xi_new_set, xi_curr_set);   /* xi_curr[sigma] <- xi_new[sigma] */
        my_dscal(vec_loc_size, beta_set[0], p_loc_set);     /* p[0] <- r + beta[0] p[0] - beta[0] omega[0] s */
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc_set);
        my_daxpy(vec_loc_size, -beta_set[0] * omega_set[0], s_loc, p_loc_set);

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

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(q_loc);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(tau_set); free(xi_old_set); free(xi_curr_set); free(xi_new_set);

    return k;

}