#include "shifted_switching_solver.h"

#define IDX(set, i, rows) ((set) + (i) * (rows))

#define EPS 1.0e-12   /* 収束判定条件 */
#define MAX_ITER 1000 /* 最大反復回数 */

#define MEASURE_TIME /* 時間計測 */

//#define DISPLAY_RESIDUAL /* 残差表示 */
#define DISPLAY_SIGMA_RESIDUAL /* sigma毎の残差表示 */
#define OUT_ITER 1     /* 残差の表示間隔 */

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

    int k, max_iter, stop_count;
    double tol;
    double max_zeta_pi, abs_zeta_pi;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set, *pi_old_set, *pi_new_set;
    double alpha_old, beta_old;

    double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req;

    bool *stop_flag;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;
    stop_count = 0;

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

    stop_flag   = (bool *)calloc(sigma_len, sizeof(bool)); /* Falseで初期化 */

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      /* (r#,r) */
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   /* r# <- r = b */
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    /* p[sigma] <- b */
        beta_set[i]   = 0.0;  /* beta[sigma] <- 0 */
        alpha_set[i]  = 1.0;  /* alpha[sigma] <- 1 */
        eta_set[i]    = 0.0;  /* eta[sigma] <- 0 */
        pi_old_set[i] = 1.0;  /* pi_old[sigma] <- 1 */
        pi_new_set[i] = 1.0;  /* pi_new[sigma] <- 1 */
        zeta_set[i]   = 1.0;  /* zeta[sigma] <- 1 */
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;    /* (r,r) */
    dot_zero = rTr; /* (r#,r#) */
    max_zeta_pi = 1.0;   /* max(|1/(zeta pi)|) */

    while (stop_count < sigma_len && k < max_iter) {

        my_dcopy(vec_loc_size, r_loc, r_old_loc);       /* r_old <- r */
        my_dcopy(sigma_len, pi_new_set, pi_old_set);    /* pi_old[sigma] <- pi_new[sigma] */
        alpha_old = alpha_set[seed];    /* alpha_old <- alpha[seed] */
        beta_old = beta_set[seed];      /* beta_old <- beta[seed] */

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
            if (stop_flag[j]) continue;
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
        my_dscal(vec_loc_size, beta_set[seed], &p_loc_set[seed * vec_loc_size]);     /* p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s */
        my_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -beta_set[seed] * omega_set[seed], s_loc, &p_loc_set[seed * vec_loc_size]);

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            if (stop_flag[j]) continue;
            beta_set[j] = (pi_old_set[j] / pi_new_set[j]) * (pi_old_set[j] / pi_new_set[j]) * beta_set[seed]; /* beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] */
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      /* p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] */
            my_daxpy(vec_loc_size, 1.0 / (pi_new_set[j] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
#endif
        for (j = 0; j < sigma_len; j++) {
            if (stop_flag[j]) {
#ifdef DISPLAY_SIGMA_RESIDUAL
                if (myid == 0 && k % OUT_ITER == 0) printf("------------ ");
#endif
                continue;
            }
            if (j == seed) {
                abs_zeta_pi = 1.0;
            } else {
                abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_new_set[j]));
            }
#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("%e ", abs_zeta_pi * sqrt(dot_r / dot_zero));
#endif
            if (abs_zeta_pi * abs_zeta_pi * dot_r <= tol * tol * dot_zero) {
                stop_flag[j] = true;
                stop_count++;
            }
        }
#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) printf("\n");
#endif

        k++;

#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e\n", k, sqrt(dot_r / dot_zero));
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
    free(stop_flag);

    return k;

}