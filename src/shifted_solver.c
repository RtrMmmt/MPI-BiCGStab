#include "shifted_solver.h"

#define IDX(set, i, rows) ((set) + (i) * (rows))

#define EPS 1.0e-12   /* 収束判定条件 */
#define MAX_ITER 1000 /* 最大反復回数 */

#define MEASURE_TIME /* 時間計測 */

//#define DISPLAY_RESIDUAL /* 残差表示 */
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

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec;

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
    if (myid == 0) printf("rTr = %e\n", rTr);

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
        //if (myid == 0) printf("alpha[0] = %e\n", alpha_set[0]);
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
        for (j = 1; j < sigma_len; j++) {
            omega_set[j] = omega_set[0] / (1.0 + omega_set[0] * sigma[j]);      /* omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) */
            my_daxpy(vec_loc_size, omega_set[j] * tau_set[j] * xi_new_set[j], r_loc, &x_loc_set[j * vec_loc_size]);     /* x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] tau[sigma] xi_new[sigma] q */
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] * tau_set[j] * xi_new_set[j] / alpha_set[j], r_loc, &p_loc_set[j * vec_loc_size]);      /* p[sigma] <- p[sigma] + omega[sigma] tau[sigma] / alpha[sigma] (xi_new[sigma] q - xi_curr[sigma] r_old) */
            my_daxpy(vec_loc_size, - omega_set[j] * tau_set[j] * xi_curr_set[j] / alpha_set[j], r_old_loc, &p_loc_set[j * vec_loc_size]);
        }
        my_daxpy(vec_loc_size, -omega_set[0], y_loc, r_loc);            /* r <- q - omega[0] y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  /* (r,r) */
        rTr_old = rTr;      /* r_old <- (r#,r) */
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  /* (r#,r) */
        for (j = 1; j < sigma_len; j++) {
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

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(tau_set); free(xi_old_set); free(xi_curr_set); free(xi_new_set);

    return k;

}

int shifted_lopbicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

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
    double max_zeta_pi, abs_zeta_pi;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set, *pi_old_set, *pi_new_set;
    double alpha_old, beta_old;

    double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req;
    //MPI_Request reqs1[2];
    //MPI_Request reqs2[2];

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); /* 安全のためゼロで初期化(下でもOK) */
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));
    pi_new_set  = (double *)malloc(sigma_len * sizeof(double));
    pi_old_set  = (double *)malloc(sigma_len * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      /* (r#,r) */
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   /* r# <- r = b */
    for (i = 0; i < sigma_len; i++) {
        //my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    /* p[sigma] <- b */
        beta_set[i] = 0.0;      /* beta[sigma] <- 0 */
        alpha_set[i] = 1.0;     /* alpha[sigma] <- 1 */
        eta_set[i] = 0.0;       /* eta[sigma] <- 0 */
        pi_old_set[i] = 1.0;    /* pi_old[sigma] <- 1 */
        pi_new_set[i] = 1.0;    /* pi_new[sigma] <- 1 */
        zeta_set[i] = 1.0;       /* zeta[sigma] <- 1 */
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;    /* (r,r) */
    dot_zero = rTr; /* (r#,r#) */
    max_zeta_pi = 1.0;   /* max(|1/(zeta pi)|) */

    while (max_zeta_pi * max_zeta_pi * dot_r > tol * tol * dot_zero && k < max_iter) {

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  /* s <- (A + sigma[seed] I) p[seed] */
        my_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  /* rTs <- (r#,s) */
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            beta_set[j] = (pi_old_set[j] / pi_new_set[j]) * (pi_old_set[j] / pi_new_set[j]) * beta_set[seed]; /* beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] */
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      /* p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] */
            my_daxpy(vec_loc_size, 1.0 / (pi_new_set[j] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }
        my_dcopy(sigma_len, pi_new_set, pi_old_set);   /* pi_old[sigma] <- pi_new[sigma] */
        my_dcopy(vec_loc_size, r_loc, r_old_loc);   /* r_old <- r */
        alpha_old = alpha_set[seed];       /* alpha_old <- alpha[seed] */
        beta_old = beta_set[seed];         /* beta_old <- beta[seed] */
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha_set[seed] = rTr / rTs;   /* alpha[seed] <- (r#,r)/(r#,s) */
        my_daxpy(vec_loc_size, -alpha_set[seed], s_loc, r_loc);   /* q <- r - alpha[seed] s */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  /* y <- (A + sigma[seed] I) q */
        my_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);

        qTq = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  /* (q,q) */
        qTy = my_ddot(vec_loc_size, r_loc, y_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  /* (q,y) */
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            eta_set[j] = (beta_old / alpha_old) * alpha_set[seed] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_set[seed] * pi_old_set[j];
            /* eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] */
            pi_new_set[j] = eta_set[j] + pi_old_set[j];    /* pi_new[sigma] <- eta[sigma] + pi_old[sigma] */
            alpha_set[j] = (pi_old_set[j] / pi_new_set[j]) * alpha_set[seed];     /* alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] */
        }
        MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
        MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);

        omega_set[seed] = qTq / qTy;  /* omega[seed] <- (q,q)/(q,y) */
        my_daxpy(vec_loc_size, alpha_set[seed], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     /* x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q */
        my_daxpy(vec_loc_size, omega_set[seed], r_loc, &x_loc_set[seed * vec_loc_size]);
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            omega_set[j] = omega_set[seed] / (1.0 - omega_set[seed] * (sigma[seed] - sigma[j]));      /* omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) */
            my_daxpy(vec_loc_size, omega_set[j] / (pi_new_set[j] * zeta_set[j]), r_loc, &x_loc_set[j * vec_loc_size]);     /* x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q */
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_new_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);    /* p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) */
            my_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_old_set[j]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            zeta_set[j] = (1.0 - omega_set[seed] * (sigma[seed] - sigma[j])) * zeta_set[j];      /* zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] */
        }
        my_daxpy(vec_loc_size, -omega_set[seed], y_loc, r_loc);            /* r <- q - omega[seed] y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  /* (r,r) */
        rTr_old = rTr;      /* r_old <- (r#,r) */
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  /* (r#,r) */
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta_set[seed] = (alpha_set[seed] / omega_set[seed]) * (rTr / rTr_old);   /* beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) */
        max_zeta_pi = 1.0;
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_new_set[j]));
            if (abs_zeta_pi > max_zeta_pi) max_zeta_pi = abs_zeta_pi;   /* max(|1/(zeta[sigma] pi[sigma])|) */
        }
        my_dscal(vec_loc_size, beta_set[seed], &p_loc_set[seed * vec_loc_size]);     /* p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s */
        my_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -beta_set[seed] * omega_set[seed], s_loc, &p_loc_set[seed * vec_loc_size]);

        k++;

#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e, Max_Zeta_Pi: %e\n", k, sqrt(dot_r / dot_zero), max_zeta_pi);
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
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set); free(pi_old_set); free(pi_new_set);

    return k;

}


int shifted_lopbicgstab_v2(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

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
    double max_zeta_pi, abs_zeta_pi;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec, *q_loc;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set, *pi_old_set, *pi_new_set;
    double alpha_old, beta_old;

    double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req;
    //MPI_Request reqs1[2];
    //MPI_Request reqs2[2];

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    q_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); /* 安全のためゼロで初期化(下でもOK) */
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));
    pi_new_set  = (double *)malloc(sigma_len * sizeof(double));
    pi_old_set  = (double *)malloc(sigma_len * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      /* (r#,r) */
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   /* r# <- r = b */
    for (i = 0; i < sigma_len; i++) {
        //my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    /* p[sigma] <- b */
        beta_set[i] = 0.0;      /* beta[sigma] <- 0 */
        alpha_set[i] = 1.0;     /* alpha[sigma] <- 1 */
        eta_set[i] = 0.0;       /* eta[sigma] <- 0 */
        pi_old_set[i] = 1.0;    /* pi_old[sigma] <- 1 */
        pi_new_set[i] = 1.0;    /* pi_new[sigma] <- 1 */
        zeta_set[i] = 1.0;       /* zeta[sigma] <- 1 */
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;    /* (r,r) */
    dot_zero = rTr; /* (r#,r#) */
    max_zeta_pi = 1.0;   /* max(|1/(zeta pi)|) */

    while (max_zeta_pi * max_zeta_pi * dot_r > tol * tol * dot_zero && k < max_iter) {

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  /* s <- (A + sigma[seed] I) p[seed] */
        my_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  /* rTs <- (r#,s) */
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            beta_set[j] = (pi_old_set[j] / pi_new_set[j]) * (pi_old_set[j] / pi_new_set[j]) * beta_set[seed]; /* beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] */
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      /* p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] */
            my_daxpy(vec_loc_size, 1.0 / (pi_new_set[j] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }
        my_dcopy(sigma_len, pi_new_set, pi_old_set);   /* pi_old[sigma] <- pi_new[sigma] */
        my_dcopy(vec_loc_size, r_loc, r_old_loc);   /* r_old <- r */
        alpha_old = alpha_set[seed];       /* alpha_old <- alpha[seed] */
        beta_old = beta_set[seed];         /* beta_old <- beta[seed] */
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha_set[seed] = rTr / rTs;   /* alpha[seed] <- (r#,r)/(r#,s) */
        my_daxpy(vec_loc_size, -alpha_set[seed], s_loc, r_loc);   /* q <- r - alpha[seed] s */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  /* y <- (A + sigma[seed] I) q */
        my_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);

        qTq = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  /* (q,q) */
        qTy = my_ddot(vec_loc_size, r_loc, y_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  /* (q,y) */
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            eta_set[j] = (beta_old / alpha_old) * alpha_set[seed] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_set[seed] * pi_old_set[j];
            /* eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] */
            pi_new_set[j] = eta_set[j] + pi_old_set[j];    /* pi_new[sigma] <- eta[sigma] + pi_old[sigma] */
            alpha_set[j] = (pi_old_set[j] / pi_new_set[j]) * alpha_set[seed];     /* alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] */
        }
        my_dcopy(vec_loc_size, r_loc, q_loc);  /* q <- r */
        MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
        MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);

        omega_set[seed] = qTq / qTy;  /* omega[seed] <- (q,q)/(q,y) */
        my_daxpy(vec_loc_size, alpha_set[seed], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     /* x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q */
        my_daxpy(vec_loc_size, omega_set[seed], r_loc, &x_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -omega_set[seed], y_loc, r_loc);            /* r <- q - omega[seed] y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  /* (r,r) */
        rTr_old = rTr;      /* r_old <- (r#,r) */
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  /* (r#,r) */
        max_zeta_pi = 1.0;
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            omega_set[j] = omega_set[seed] / (1.0 - omega_set[seed] * (sigma[seed] - sigma[j]));      /* omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) */
            my_daxpy(vec_loc_size, omega_set[j] / (pi_new_set[j] * zeta_set[j]), q_loc, &x_loc_set[j * vec_loc_size]);     /* x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q */
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_new_set[j]), q_loc, &p_loc_set[j * vec_loc_size]);    /* p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) */
            my_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_old_set[j]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            zeta_set[j] = (1.0 - omega_set[seed] * (sigma[seed] - sigma[j])) * zeta_set[j];      /* zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] */
            
            abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_new_set[j]));
            if (abs_zeta_pi > max_zeta_pi) max_zeta_pi = abs_zeta_pi;   /* max(|1/(zeta[sigma] pi[sigma])|) */
        }
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta_set[seed] = (alpha_set[seed] / omega_set[seed]) * (rTr / rTr_old);   /* beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) */
        my_dscal(vec_loc_size, beta_set[seed], &p_loc_set[seed * vec_loc_size]);     /* p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s */
        my_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -beta_set[seed] * omega_set[seed], s_loc, &p_loc_set[seed * vec_loc_size]);

        k++;

#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e, Max_Zeta_Pi: %e\n", k, sqrt(dot_r / dot_zero), max_zeta_pi);
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
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set); free(pi_old_set); free(pi_new_set);

    return k;

}

int shifted_lopbicgstab_nooverlap(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

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
    double max_zeta_pi, abs_zeta_pi;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set, *pi_old_set, *pi_new_set;
    double alpha_old, beta_old;

    double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req;
    //MPI_Request reqs1[2];
    //MPI_Request reqs2[2];

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); /* 安全のためゼロで初期化(下でもOK) */
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));
    pi_new_set  = (double *)malloc(sigma_len * sizeof(double));
    pi_old_set  = (double *)malloc(sigma_len * sizeof(double));

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
        eta_set[i] = 0.0;       /* eta[sigma] <- 0 */
        pi_old_set[i] = 1.0;    /* pi_old[sigma] <- 1 */
        pi_new_set[i] = 1.0;    /* pi_new[sigma] <- 1 */
        zeta_set[i] = 1.0;       /* zeta[sigma] <- 1 */
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;    /* (r,r) */
    dot_zero = rTr; /* (r#,r#) */
    max_zeta_pi = 1.0;   /* max(|1/(zeta pi)|) */

    while (max_zeta_pi * max_zeta_pi * dot_r > tol * tol * dot_zero && k < max_iter) {

        my_dcopy(sigma_len, pi_new_set, pi_old_set);   /* pi_old[sigma] <- pi_new[sigma] */
        my_dcopy(vec_loc_size, r_loc, r_old_loc);   /* r_old <- r */
        alpha_old = alpha_set[seed];       /* alpha_old <- alpha[seed] */
        beta_old = beta_set[seed];         /* beta_old <- beta[seed] */

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  /* s <- (A + sigma[seed] I) p[seed] */
        my_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  /* rTs <- (r#,s) */
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha_set[seed] = rTr / rTs;   /* alpha[seed] <- (r#,r)/(r#,s) */
        my_daxpy(vec_loc_size, -alpha_set[seed], s_loc, r_loc);   /* q <- r - alpha[seed] s */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  /* y <- (A + sigma[seed] I) q */
        my_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);
        qTq = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  /* (q,q) */
        qTy = my_ddot(vec_loc_size, r_loc, y_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  /* (q,y) */
        MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
        MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);

        omega_set[seed] = qTq / qTy;  /* omega[seed] <- (q,q)/(q,y) */
        my_daxpy(vec_loc_size, alpha_set[seed], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     /* x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q */
        my_daxpy(vec_loc_size, omega_set[seed], r_loc, &x_loc_set[seed * vec_loc_size]);
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            eta_set[j] = (beta_old / alpha_old) * alpha_set[seed] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_set[seed] * pi_old_set[j];
            /* eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] */
            pi_new_set[j] = eta_set[j] + pi_old_set[j];    /* pi_new[sigma] <- eta[sigma] + pi_old[sigma] */
            alpha_set[j] = (pi_old_set[j] / pi_new_set[j]) * alpha_set[seed];     /* alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] */
            omega_set[j] = omega_set[seed] / (1.0 - omega_set[seed] * (sigma[seed] - sigma[j]));      /* omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) */
            my_daxpy(vec_loc_size, omega_set[j] / (pi_new_set[j] * zeta_set[j]), r_loc, &x_loc_set[j * vec_loc_size]);     /* x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q */
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_new_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);    /* p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) */
            my_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_old_set[j]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            zeta_set[j] = (1.0 - omega_set[seed] * (sigma[seed] - sigma[j])) * zeta_set[j];      /* zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] */
        }
        my_daxpy(vec_loc_size, -omega_set[seed], y_loc, r_loc);            /* r <- q - omega[seed] y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  /* (r,r) */
        rTr_old = rTr;      /* r_old <- (r#,r) */
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  /* (r#,r) */
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta_set[seed] = (alpha_set[seed] / omega_set[seed]) * (rTr / rTr_old);   /* beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) */
        max_zeta_pi = 1.0;
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_new_set[j]));
            if (abs_zeta_pi > max_zeta_pi) max_zeta_pi = abs_zeta_pi;   /* max(|1/(zeta[sigma] pi[sigma])|) */
        }
        my_dscal(vec_loc_size, beta_set[seed], &p_loc_set[seed * vec_loc_size]);     /* p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s */
        my_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -beta_set[seed] * omega_set[seed], s_loc, &p_loc_set[seed * vec_loc_size]);

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            beta_set[j] = (pi_old_set[j] / pi_new_set[j]) * (pi_old_set[j] / pi_new_set[j]) * beta_set[seed]; /* beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] */
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      /* p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] */
            my_daxpy(vec_loc_size, 1.0 / (pi_new_set[j] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

        k++;

#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e, Max_Zeta_Pi: %e\n", k, sqrt(dot_r / dot_zero), max_zeta_pi);
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
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set); free(pi_old_set); free(pi_new_set);

    return k;

}

int shifted_pipe_lopbicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

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
    double max_zeta_pi, abs_zeta_pi;

    double *r_old_loc, *r_hat_loc, *s_loc, *z_loc, *w_loc, *v_loc, *t_loc, *vec;
    
    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set, *pi_old_set, *pi_new_set;
    double alpha_old, beta_old;

    double dot_r, dot_zero, rTr, rTw, wTw, rTs, rTz, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTw_req, wTw_req, rTs_req, rTz_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    z_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    w_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    v_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    t_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); /* 安全のためゼロで初期化(下でもOK) */
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));
    pi_new_set  = (double *)malloc(sigma_len * sizeof(double));
    pi_old_set  = (double *)malloc(sigma_len * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    /* 初期化 */
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);   /* (r,r) */

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);  /* w <- (A + sigma[seed] I) r */
    my_daxpy(vec_loc_size, sigma[seed], r_loc, w_loc);
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (r,w) */

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);  /* t <- (A + sigma[seed] I) w */
    my_daxpy(vec_loc_size, sigma[seed], w_loc, t_loc);

    my_dcopy(vec_loc_size, r_loc, r_hat_loc);       /* r# <- r */
    for (i = 0; i < sigma_len; i++) {
        //my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    /* p[sigma] <- b */
        beta_set[i] = 0.0;      /* beta[sigma] <- 0 */
        alpha_set[i] = 1.0;     /* alpha[sigma] <- 1 */
        eta_set[i] = 0.0;       /* eta[sigma] <- 0 */
        pi_old_set[i] = 1.0;    /* pi_old[sigma] <- 1 */
        pi_new_set[i] = 1.0;    /* pi_new[sigma] <- 1 */
        zeta_set[i] = 1.0;       /* zeta[sigma] <- 1 */
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
    MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);

    alpha_old = 1.0;    /* alpha_old <- 1 */
    alpha_set[seed] = rTr / rTw;  /* alpha[seed] <- (r,r)/(r,w) */
    dot_r = rTr;    /* (r,r) */
    dot_zero = rTr; /* (r#,r#) */
    max_zeta_pi = 1.0;   /* max(|1/(zeta pi)|) */

    /* 反復 */
    while (max_zeta_pi * max_zeta_pi * dot_r > tol * tol * dot_zero && k < max_iter) {
        
        my_daxpy(vec_loc_size, -omega_set[seed], s_loc, &p_loc_set[seed * vec_loc_size]);   /* p[seed] <- r + beta[seed] (p[seed] - omega[seed] s) */
        my_dscal(vec_loc_size, beta_set[seed], &p_loc_set[seed * vec_loc_size]);            /*                             */
        my_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);                /* --------------------------- */
        my_daxpy(vec_loc_size, -omega_set[seed], z_loc, s_loc);     /* s <- w + beta[seed] (s - omega[seed] z) */
        my_dscal(vec_loc_size, beta_set[seed], s_loc);              /*                             */
        my_daxpy(vec_loc_size, 1.0, w_loc, s_loc);                  /* --------------------------- */
        my_daxpy(vec_loc_size, -omega_set[seed], v_loc, z_loc);     /* z <- t + beta[seed] (z - omega[seed] v) */
        my_dscal(vec_loc_size, beta_set[seed], z_loc);              /*                             */
        my_daxpy(vec_loc_size, 1.0, t_loc, z_loc);                  /* --------------------------- */
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            beta_set[j] = (pi_old_set[j] / pi_new_set[j]) * (pi_old_set[j] / pi_new_set[j]) * beta_set[seed]; /* beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] */
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      /* p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] */
            my_daxpy(vec_loc_size, 1.0 / (pi_new_set[j] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }
        my_dcopy(vec_loc_size, r_loc, r_old_loc);   /* r_old <- r */
        my_daxpy(vec_loc_size, -alpha_set[seed], s_loc, r_loc);     /* q <- r - alpha[seed] s */
        my_daxpy(vec_loc_size, -alpha_set[seed], z_loc, w_loc);     /* y <- w - alpha[seed] z */
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (q,y) */
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);   /* (y,y) */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, z_loc, vec, v_loc);  /* v <- (A + sigma[seed] I) z */
        my_daxpy(vec_loc_size, sigma[seed], z_loc, v_loc);
        my_dcopy(sigma_len, pi_new_set, pi_old_set);   /* pi_old[sigma] <- pi_new[sigma] */
        beta_old = beta_set[seed];         /* beta_old <- beta[seed] */
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            eta_set[j] = (beta_old / alpha_old) * alpha_set[seed] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_set[seed] * pi_old_set[j];
            /* eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] */
            pi_new_set[j] = eta_set[j] + pi_old_set[j];    /* pi_new[sigma] <- eta[sigma] + pi_old[sigma] */
            alpha_set[j] = (pi_old_set[j] / pi_new_set[j]) * alpha_set[seed];     /* alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] */
        }
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega_set[seed] = rTw / wTw;  /* omega[seed] <- (q,y)/(y,y) */
        my_daxpy(vec_loc_size, alpha_set[seed], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);    /* x <- x + alpha[seed] p[seed] + omega[seed] q */
        my_daxpy(vec_loc_size, omega_set[seed], r_loc, &x_loc_set[seed * vec_loc_size]);    /* -------------------------- */
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            omega_set[j] = omega_set[seed] / (1.0 - omega_set[seed] * (sigma[seed] - sigma[j]));      /* omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) */
            my_daxpy(vec_loc_size, omega_set[j] / (pi_new_set[j] * zeta_set[j]), r_loc, &x_loc_set[j * vec_loc_size]);     /* x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q */
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_new_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);    /* p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) */
            my_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_old_set[j]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            zeta_set[j] = (1.0 - omega_set[seed] * (sigma[seed] - sigma[j])) * zeta_set[j];      /* zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] */
        }
        my_daxpy(vec_loc_size, -omega_set[seed], w_loc, r_loc);   /* r <- q - omega[seed] y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);   /* (r,r) */
        my_daxpy(vec_loc_size, -alpha_set[seed], v_loc, t_loc);   /* w <- y - omega (t - alpha[seed] v) */
        my_daxpy(vec_loc_size, -omega_set[seed], t_loc, w_loc);   /* ---------------------------- */
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);   /* (r#,r) */
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (r#,w) */
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);   /* (r#,s) */
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);   /* (r#,z) */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);  /* t <- (A + sigma[seed] I) w */
        my_daxpy(vec_loc_size, sigma[seed], w_loc, t_loc);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTz_req, MPI_STATUS_IGNORE);

        beta_set[seed] = (alpha_set[seed] / omega_set[seed]) * (rTr / rTr_old);           /* beta[seed] <- (alpha[seed] / omega[seed]) * ((r#,r)/(r#,r)) */
        alpha_old = alpha_set[seed];       /* alpha_old <- alpha[seed] */
        alpha_set[seed] = rTr / (rTw + beta_set[seed] * (rTs - omega_set[seed] * rTz));   /* alpha[seed] <- (r#,r) / {(r#,w) + beta[seed] ((r#,s) - omega[seed] (r#,z))} */
        max_zeta_pi = 1.0;
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_new_set[j]));
            if (abs_zeta_pi > max_zeta_pi) max_zeta_pi = abs_zeta_pi;   /* max(|1/(zeta[sigma] pi[sigma])|) */
        }

        k++;

        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e, Max_Zeta_Pi: %e\n", k, sqrt(dot_r / dot_zero), max_zeta_pi);
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

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(z_loc); free(w_loc); free(v_loc); free(t_loc); free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set); free(pi_old_set); free(pi_new_set);

    return k;
}