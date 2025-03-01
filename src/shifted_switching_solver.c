#include "shifted_switching_solver.h"

#define IDX(set, i, rows) ((set) + (i) * (rows))

#define EPS 1.0e-12   // 収束判定条件 
#define MAX_ITER 1000 // 最大反復回数 

#define MEASURE_TIME // 時間計測 
//#define MEASURE_SECTION_TIME // セクション時間計測
//#define DISPLAY_SECTION_TIME // セクション時間表示

//#define DISPLAY_RESULT // 結果表示 
//#define DISPLAY_RESIDUAL // 途中の残差表示 
//#define DISPLAY_SIGMA_RESIDUAL // 途中のsigma毎の残差表示 
#define OUT_ITER 1     // 残差の表示間隔 

//#define DISPLAY_ERROR


int shifted_lopbicg(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
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

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); // 一応ゼロで初期化(下でもOK) 
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));
    pi_new_set  = (double *)malloc(sigma_len * sizeof(double));
    pi_old_set  = (double *)malloc(sigma_len * sizeof(double));

    stop_flag   = (bool *)calloc(sigma_len, sizeof(bool)); // Falseで初期化 

#ifdef MEASURE_SECTION_TIME
        double seed_time, shift_time;
        double section_start_time, section_end_time;
        seed_time = 0; shift_time = 0;
#endif

    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      // (r#,r) 
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   // r# <- r = b 
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    // p[sigma] <- b 
        alpha_set[i]  = 1.0;  // alpha[sigma]  <- 1 
        beta_set[i]   = 0.0;  // beta[sigma]   <- 0         
        eta_set[i]    = 0.0;  // eta[sigma]    <- 0 
        pi_old_set[i] = 1.0;  // pi_old[sigma] <- 1 
        pi_new_set[i] = 1.0;  // pi_new[sigma] <- 1 
        zeta_set[i]   = 1.0;  // zeta[sigma]   <- 1 
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;    // (r,r) 
    dot_zero = rTr; // (r#,r#) 
    max_zeta_pi = 1.0;   // max(|1/(zeta pi)|) 

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
        start_time = MPI_Wtime();
#endif

    while (stop_count < sigma_len && k < max_iter) {

        my_dcopy(vec_loc_size, r_loc, r_old_loc);       // r_old <- r 
        my_dcopy(sigma_len, pi_new_set, pi_old_set);    // pi_old[sigma] <- pi_new[sigma] 
        alpha_old = alpha_set[seed];    // alpha_old <- alpha[seed] 
        beta_old = beta_set[seed];      // beta_old <- beta[seed] 

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  // s <- (A + sigma[seed] I) p[seed] 
        my_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);

        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  // rTs <- (r#,s) 
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha_set[seed] = rTr / rTs;   // alpha[seed] <- (r#,r)/(r#,s) 
        my_daxpy(vec_loc_size, -alpha_set[seed], s_loc, r_loc);   // q <- r - alpha[seed] s (qはrを再利用する)
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  // y <- (A + sigma[seed] I) q 
        my_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);
        qTq = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  // (q,q) 
        qTy = my_ddot(vec_loc_size, r_loc, y_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  // (q,y) 
        MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
        MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);

        omega_set[seed] = qTq / qTy;  // omega[seed] <- (q,q)/(q,y) 
        my_daxpy(vec_loc_size, alpha_set[seed], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     // x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q 
        my_daxpy(vec_loc_size, omega_set[seed], r_loc, &x_loc_set[seed * vec_loc_size]);

#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;   // seedはスキップ 
            if (stop_flag[j]) continue;  // 収束したものはスキップ 
            eta_set[j] = (beta_old / alpha_old) * alpha_set[seed] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_set[seed] * pi_old_set[j];
            // eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] 
            pi_new_set[j] = eta_set[j] + pi_old_set[j];    // pi_new[sigma] <- eta[sigma] + pi_old[sigma] 
            alpha_set[j] = (pi_old_set[j] / pi_new_set[j]) * alpha_set[seed];     // alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] 
            omega_set[j] = omega_set[seed] / (1.0 - omega_set[seed] * (sigma[seed] - sigma[j]));      // omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) 
            my_daxpy(vec_loc_size, omega_set[j] / (pi_new_set[j] * zeta_set[j]), r_loc, &x_loc_set[j * vec_loc_size]);     // x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q 
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_new_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);    // p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) 
            my_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_old_set[j]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            zeta_set[j] = (1.0 - omega_set[seed] * (sigma[seed] - sigma[j])) * zeta_set[j];      // zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] 
        }

#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        shift_time += section_end_time - section_start_time;
#endif

        my_daxpy(vec_loc_size, -omega_set[seed], y_loc, r_loc);            // r <- q - omega[seed] y 
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  // (r,r) 
        rTr_old = rTr;      // r_old <- (r#,r) 
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  // (r#,r) 
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta_set[seed] = (alpha_set[seed] / omega_set[seed]) * (rTr / rTr_old);   // beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) 
        my_dscal(vec_loc_size, beta_set[seed], &p_loc_set[seed * vec_loc_size]);     // p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s 
        my_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -beta_set[seed] * omega_set[seed], s_loc, &p_loc_set[seed * vec_loc_size]);

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            if (stop_flag[j]) continue;
            beta_set[j] = (pi_old_set[j] / pi_new_set[j]) * (pi_old_set[j] / pi_new_set[j]) * beta_set[seed]; // beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] 
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      // p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] 
            my_daxpy(vec_loc_size, 1.0 / (pi_new_set[j] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
#endif

#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
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

#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        shift_time += section_end_time - section_start_time;
#endif

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


#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
        end_time = MPI_Wtime();
        total_time = end_time - start_time;
#endif

#ifdef MEASURE_SECTION_TIME
    seed_time = total_time - shift_time;
#endif

    if (myid == 0) {
#ifdef DISPLAY_RESULT
        printf("Total iter   : %d\n", k - 1);
        printf("Final r      : %e\n", sqrt(dot_r / dot_zero));
        printf("x            : %e\n", x_loc_set[seed * vec_loc_size]);
#endif
#ifdef MEASURE_TIME
        printf("Total time   : %e [sec.] \n", total_time);
        printf("Avg time/iter: %e [sec.] \n", total_time / k);
#endif
#ifdef MEASURE_SECTION_TIME
        printf("Seed time    : %e [sec.]\n", seed_time);
        printf("Shift time   : %e [sec.]\n", shift_time);
#endif
    }

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set); free(pi_old_set); free(pi_new_set);
    free(stop_flag);

    return k;

}


int shifted_lopbicg_switching(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
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
    int max_sigma;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec, *q_loc_copy;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set;
    double alpha_old, beta_old;

    double *alpha_seed_archive, *beta_seed_archive, *omega_seed_archive;
    double *pi_archive_set;

    double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req, vec_req;

    bool *stop_flag;

    k = 1;
    tol = EPS;
    max_iter = MAX_ITER + 1;
    stop_count = 0;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    q_loc_copy  = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); // 一応ゼロで初期化(下でもOK) 
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));


    // seed switching で使うので履歴を保存する 
    alpha_seed_archive  = (double *)malloc(max_iter * sizeof(double));
    beta_seed_archive   = (double *)malloc(max_iter * sizeof(double));
    omega_seed_archive  = (double *)malloc(max_iter * sizeof(double));
    pi_archive_set      = (double *)malloc(max_iter * sigma_len * sizeof(double));

    stop_flag   = (bool *)calloc(sigma_len, sizeof(bool)); // Falseで初期化 

#ifdef DISPLAY_ERROR
    double *ans_loc = (double *)malloc(vec_loc_size * sizeof(double));   
    double *temp    = (double *)malloc(vec_loc_size * sizeof(double));
    for (i = 0; i < vec_loc_size; i++) {
        temp[i] = 1.0;
    }
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, temp, vec, ans_loc);
    my_daxpy(vec_loc_size, sigma[seed], temp, ans_loc);
    free(temp);
#endif

#ifdef MEASURE_SECTION_TIME
        double seed_time, shift_time, switch_time;
        double section_start_time, section_end_time;
        seed_time = 0; shift_time = 0; switch_time = 0;
#endif

    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      // (r#,r) 
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   // r# <- r = b 
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    // p[sigma] <- b 
        alpha_set[i]  = 1.0;  // alpha[sigma]  <- 1 
        beta_set[i]   = 0.0;  // beta[sigma]   <- 0 
        eta_set[i]    = 0.0;  // eta[sigma]    <- 0 
        pi_archive_set[i * max_iter + 0] = 1.0;
        pi_archive_set[i * max_iter + 1] = 1.0;
        zeta_set[i]   = 1.0;  // zeta[sigma]   <- 1 
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;    // (r,r) 
    dot_zero = rTr; // (r#,r#) 
    max_zeta_pi = 1.0;   // max(|1/(zeta pi)|) 

    alpha_seed_archive[0] = 1.0;
    beta_seed_archive[0]  = 0.0;

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
        start_time = MPI_Wtime();
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0) printf("Seed : %d\n", seed);
#endif

    while (stop_count < sigma_len && k < max_iter) {

        my_dcopy(vec_loc_size, r_loc, r_old_loc);       // r_old <- r 

        //MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  // s <- (A + sigma[seed] I) p[seed]
        MPI_Iallgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        for (int l = 0; l < vec_loc_size; l++) {
            s_loc[l] = 0.0;
        }
        mult(A_loc_diag, &p_loc_set[seed * vec_loc_size], s_loc);
        MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
        mult(A_loc_offd, vec, s_loc);

        my_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  // rTs <- (r#,s) 
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha_seed_archive[k] = rTr / rTs;   // alpha[seed] <- (r#,r)/(r#,s) 
        my_daxpy(vec_loc_size, -alpha_seed_archive[k], s_loc, r_loc);   // q <- r - alpha[seed] s 
        my_dcopy(vec_loc_size, r_loc, q_loc_copy);  // q_copy <- q (q_copyにr_locをコピー　シード方程式を一つにまとめるため)

        //MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  // y <- (A + sigma[seed] I) q 
        MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        for (int l = 0; l < vec_loc_size; l++) {
            y_loc[l] = 0.0;
        }
        mult(A_loc_diag, r_loc, y_loc);
        MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
        mult(A_loc_offd, vec, y_loc);

        my_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);
        qTq = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  // (q,q) 
        qTy = my_ddot(vec_loc_size, r_loc, y_loc); MPI_Iallreduce(MPI_IN_PLACE, &qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  // (q,y) 
        MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
        MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);

        omega_seed_archive[k] = qTq / qTy;  // omega[seed] <- (q,q)/(q,y) 
        my_daxpy(vec_loc_size, alpha_seed_archive[k], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     // x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q 
        my_daxpy(vec_loc_size, omega_seed_archive[k], r_loc, &x_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -omega_seed_archive[k], y_loc, r_loc);            // r <- q - omega[seed] y 
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  // (r,r) 
        rTr_old = rTr;      // r_old <- (r#,r) 
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc); MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  // (r#,r) 
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta_seed_archive[k] = (alpha_seed_archive[k] / omega_seed_archive[k]) * (rTr / rTr_old);   // beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) 
        my_dscal(vec_loc_size, beta_seed_archive[k], &p_loc_set[seed * vec_loc_size]);     // p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s 
        my_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -beta_seed_archive[k] * omega_seed_archive[k], s_loc, &p_loc_set[seed * vec_loc_size]);

#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            if (stop_flag[j]) continue;
            eta_set[j] = (beta_seed_archive[k - 1] / alpha_seed_archive[k - 1]) * alpha_seed_archive[k] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_seed_archive[k] * pi_archive_set[j * max_iter + (k - 1)];
            // eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] 
            pi_archive_set[j * max_iter + k] = eta_set[j] + pi_archive_set[j * max_iter + (k - 1)];    // pi_new[sigma] <- eta[sigma] + pi_old[sigma] 
            alpha_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * alpha_seed_archive[k];     // alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] 
            omega_set[j] = omega_seed_archive[k] / (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j]));      // omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) 
            my_daxpy(vec_loc_size, omega_set[j] / (pi_archive_set[j * max_iter + k] * zeta_set[j]), q_loc_copy, &x_loc_set[j * vec_loc_size]);     // x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q 
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + k]), q_loc_copy, &p_loc_set[j * vec_loc_size]);    // p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) 
            my_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + (k - 1)]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            zeta_set[j] = (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j])) * zeta_set[j];      // zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] 
            beta_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * beta_seed_archive[k]; // beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] 
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      // p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] 
            my_daxpy(vec_loc_size, 1.0 / (pi_archive_set[j * max_iter + k] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
#endif
        max_zeta_pi = 1.0;
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
                abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_archive_set[j * max_iter + k]));
            }
#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("%e ", abs_zeta_pi * sqrt(dot_r / dot_zero));
#endif
            if (abs_zeta_pi * abs_zeta_pi * dot_r <= tol * tol * dot_zero) {
                stop_flag[j] = true;
                stop_count++;
            } else {
                if (abs_zeta_pi > max_zeta_pi) {
                    max_zeta_pi = abs_zeta_pi;
                    max_sigma = j;
                }
            }
        }
#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) printf("\n");
#endif

#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        shift_time += section_end_time - section_start_time;
#endif

#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif

        // seed switching 
        if (stop_flag[seed] && stop_count < sigma_len) {
#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0) printf("Seed : %d\n", max_sigma);
#endif
            for (i = 1; i <= k; i++) {
                alpha_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * alpha_seed_archive[i];
                beta_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * beta_seed_archive[i];
                omega_seed_archive[i] = omega_seed_archive[i] / (1.0 - omega_seed_archive[i] * (sigma[seed] - sigma[max_sigma]));
            }
            my_dscal(vec_loc_size, 1.0 / (zeta_set[max_sigma] * pi_archive_set[max_sigma * max_iter + k]), r_loc);

            for (j = 0; j < sigma_len; j++) {
                eta_set[j]    = 0.0;  // eta[sigma]    <- 0 
                //pi_archive_set[j * max_iter + 0] = 1.0;
                zeta_set[j]   = 1.0;  // zeta[sigma]   <- 1 
            }
            //alpha_seed_archive[0] = 1.0;
            //beta_seed_archive[0]  = 0.0;

            for (i = 1; i <= k; i++) {
                for (j = 0; j < sigma_len; j++) {
                    if (stop_flag[j]) continue;
                    if (j == max_sigma) continue;
                    eta_set[j] = (beta_seed_archive[i - 1] / alpha_seed_archive[i - 1]) * alpha_seed_archive[i] * eta_set[j] - (sigma[max_sigma] - sigma[j]) * alpha_seed_archive[i] * pi_archive_set[j * max_iter + (i - 1)];
                    pi_archive_set[j * max_iter + i] = eta_set[j] + pi_archive_set[j * max_iter + (i - 1)];
                    zeta_set[j] = (1.0 - omega_seed_archive[i] * (sigma[max_sigma] - sigma[j])) * zeta_set[j];
                }
            }

            for (j = 0; j < sigma_len; j++) {
                if (stop_flag[j]) continue;
                if (j == max_sigma) continue;
                if (myid == 0) printf("sigma[%d] eta: %f, pi: %f, zeta: %f\n", j, eta_set[j], pi_archive_set[j * max_iter + k], zeta_set[j]);
            }

            seed = max_sigma;
            if (myid == 0) printf("k: %d, seed: %d, remain: %d\n", k, seed, sigma_len - stop_count);
        }

#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        switch_time += section_end_time - section_start_time;
#endif

        k++;

#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e\n", k, sqrt(dot_r / dot_zero));
        }
#endif

    }


#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
#endif

#ifdef MEASURE_SECTION_TIME
    seed_time = total_time - shift_time - switch_time;
#endif

    if (myid == 0) {
#ifdef DISPLAY_RESIDUAL
        printf("Final r      : %e\n", sqrt(dot_r / dot_zero));
#endif
#ifdef MEASURE_TIME
        printf("Total iter   : %d\n", k - 1);
        printf("Total time   : %e [sec.] \n", total_time);
        printf("Avg time/iter: %e [sec.] \n", total_time / k);
#endif
#ifdef MEASURE_SECTION_TIME
        printf("Seed time    : %e [sec.]\n", seed_time);
        printf("Shift time   : %e [sec.]\n", shift_time);
        printf("Switch time  : %e [sec.]\n", switch_time);
#endif
    }

#ifdef DISPLAY_ERROR
    if (myid == 0) {
        printf("seed(0:seed, 1:shift), sigma, relative error\n");
    }

    for (int i = 0; i < sigma_len; i++) {
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &x_loc_set[i * vec_loc_size], vec, r_loc);
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

        double rerative_error = sqrt(global_diff_norm_2) / sqrt(global_ans_norm_2); //ノルムで相対誤差を計算
        if (myid == 0) {
            if (i == seed) printf("0, %e, %e\n", sigma[i], rerative_error);
            else if (i % 10 == 0) printf("1, %e, %e\n", sigma[i], rerative_error);
        }
    }
    free(ans_loc);
#endif

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(q_loc_copy);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set);
    free(alpha_seed_archive); free(beta_seed_archive); free(omega_seed_archive); free(pi_archive_set);
    free(stop_flag);

    return k;

}


int shifted_lopbicg_switching_noovlp(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
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
    int max_sigma;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec, *q_loc_copy;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set;
    double alpha_old, beta_old;

    double *alpha_seed_archive, *beta_seed_archive, *omega_seed_archive;
    double *pi_archive_set;

    double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req, vec_req;

    bool *stop_flag;

    k = 1;
    tol = EPS;
    max_iter = MAX_ITER + 1;
    stop_count = 0;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    q_loc_copy  = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); // 一応ゼロで初期化(下でもOK) 
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));


    // seed switching で使うので履歴を保存する 
    alpha_seed_archive  = (double *)malloc(max_iter * sizeof(double));
    beta_seed_archive   = (double *)malloc(max_iter * sizeof(double));
    omega_seed_archive  = (double *)malloc(max_iter * sizeof(double));
    pi_archive_set      = (double *)malloc(max_iter * sigma_len * sizeof(double));

    stop_flag   = (bool *)calloc(sigma_len, sizeof(bool)); // Falseで初期化 

#ifdef MEASURE_SECTION_TIME
        double seed_time, shift_time, switch_time;
        double agv_1_time, mult_diag_1_time, mult_offd_1_time;
        double agv_2_time, mult_diag_2_time, mult_offd_2_time;
        double ared_time;
        double section_start_time, section_end_time;
        double seed_start_time, seed_end_time;

        double seed_iter_time, shift_iter_time;
        double agv_1_iter_time, mult_diag_1_iter_time, mult_offd_1_iter_time;
        double agv_2_iter_time, mult_diag_2_iter_time, mult_offd_2_iter_time;
        double ared_iter_time;

        seed_time = 0; shift_time = 0; switch_time = 0;
        agv_1_time = 0; mult_diag_1_time = 0; mult_offd_1_time = 0;
        agv_2_time = 0; mult_diag_2_time = 0; mult_offd_2_time = 0;
        ared_time = 0;
#endif

    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      // (r#,r) 
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   // r# <- r = b 
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    // p[sigma] <- b 
        alpha_set[i]  = 1.0;  // alpha[sigma]  <- 1 
        beta_set[i]   = 0.0;  // beta[sigma]   <- 0 
        eta_set[i]    = 0.0;  // eta[sigma]    <- 0 
        pi_archive_set[i * max_iter + 0] = 1.0;
        pi_archive_set[i * max_iter + 1] = 1.0;
        zeta_set[i]   = 1.0;  // zeta[sigma]   <- 1 
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;    // (r,r) 
    dot_zero = rTr; // (r#,r#) 
    max_zeta_pi = 1.0;   // max(|1/(zeta pi)|) 

    alpha_seed_archive[0] = 1.0;
    beta_seed_archive[0]  = 0.0;

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
        start_time = MPI_Wtime();
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0) printf("Seed : %d\n", seed);
#endif

    while (stop_count < sigma_len && k < max_iter) {

#ifdef MEASURE_SECTION_TIME
        seed_start_time = MPI_Wtime();
#endif

        my_dcopy(vec_loc_size, r_loc, r_old_loc);       // r_old <- r 

        //MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  // s <- (A + sigma[seed] I) p[seed]
#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif
        MPI_Iallgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        agv_1_time += section_end_time - section_start_time;
        agv_1_iter_time = section_end_time - section_start_time;
        section_start_time = MPI_Wtime();
#endif
        for (int l = 0; l < vec_loc_size; l++) {
            s_loc[l] = 0.0;
        }
        mult(A_loc_diag, &p_loc_set[seed * vec_loc_size], s_loc);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        mult_diag_1_time += section_end_time - section_start_time;
        mult_diag_1_iter_time = section_end_time - section_start_time;
        section_start_time = MPI_Wtime();
#endif
        //MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
        mult(A_loc_offd, vec, s_loc);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        mult_offd_1_time += section_end_time - section_start_time;
        mult_offd_1_iter_time = section_end_time - section_start_time;
#endif

        my_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);
#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif
        MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  // rTs <- (r#,s) 
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        ared_time += section_end_time - section_start_time;
        ared_iter_time = section_end_time - section_start_time;
#endif

        alpha_seed_archive[k] = rTr / rTs;   // alpha[seed] <- (r#,r)/(r#,s) 
        my_daxpy(vec_loc_size, -alpha_seed_archive[k], s_loc, r_loc);   // q <- r - alpha[seed] s 
        my_dcopy(vec_loc_size, r_loc, q_loc_copy);  // q_copy <- q (q_copyにr_locをコピー　シード方程式を一つにまとめるため)

        //MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  // y <- (A + sigma[seed] I) q 
#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif
        MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        agv_2_time += section_end_time - section_start_time;
        agv_2_iter_time = section_end_time - section_start_time;
        section_start_time = MPI_Wtime();
#endif
        for (int l = 0; l < vec_loc_size; l++) {
            y_loc[l] = 0.0;
        }
        mult(A_loc_diag, r_loc, y_loc);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        mult_diag_2_time += section_end_time - section_start_time;
        mult_diag_2_iter_time = section_end_time - section_start_time;
        section_start_time = MPI_Wtime();
#endif
        //MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
        mult(A_loc_offd, vec, y_loc);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        mult_offd_2_time += section_end_time - section_start_time;
        mult_offd_2_iter_time = section_end_time - section_start_time;
#endif

        my_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);
        qTq = my_ddot(vec_loc_size, r_loc, r_loc); //MPI_Iallreduce(MPI_IN_PLACE, &qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  // (q,q) 
        qTy = my_ddot(vec_loc_size, r_loc, y_loc); //MPI_Iallreduce(MPI_IN_PLACE, &qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  // (q,y) 
#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif
        MPI_Iallreduce(MPI_IN_PLACE, &qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);
        MPI_Iallreduce(MPI_IN_PLACE, &qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);
        MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
        MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        ared_time += section_end_time - section_start_time;
        ared_iter_time += section_end_time - section_start_time;
#endif

        omega_seed_archive[k] = qTq / qTy;  // omega[seed] <- (q,q)/(q,y) 
        my_daxpy(vec_loc_size, alpha_seed_archive[k], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     // x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q 
        my_daxpy(vec_loc_size, omega_seed_archive[k], r_loc, &x_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -omega_seed_archive[k], y_loc, r_loc);            // r <- q - omega[seed] y 
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc); //MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  // (r,r) 
        rTr_old = rTr;      // r_old <- (r#,r) 
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc); //MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  // (r#,r) 
#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif
        MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        ared_time += section_end_time - section_start_time;
        ared_iter_time += section_end_time - section_start_time;
#endif

        beta_seed_archive[k] = (alpha_seed_archive[k] / omega_seed_archive[k]) * (rTr / rTr_old);   // beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) 
        my_dscal(vec_loc_size, beta_seed_archive[k], &p_loc_set[seed * vec_loc_size]);     // p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s 
        my_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_daxpy(vec_loc_size, -beta_seed_archive[k] * omega_seed_archive[k], s_loc, &p_loc_set[seed * vec_loc_size]);

#ifdef MEASURE_SECTION_TIME
        seed_end_time = MPI_Wtime();
        seed_time += seed_end_time - seed_start_time;
        seed_iter_time = seed_end_time - seed_start_time;
        section_start_time = MPI_Wtime();
#endif

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            if (stop_flag[j]) continue;
            eta_set[j] = (beta_seed_archive[k - 1] / alpha_seed_archive[k - 1]) * alpha_seed_archive[k] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_seed_archive[k] * pi_archive_set[j * max_iter + (k - 1)];
            // eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] 
            pi_archive_set[j * max_iter + k] = eta_set[j] + pi_archive_set[j * max_iter + (k - 1)];    // pi_new[sigma] <- eta[sigma] + pi_old[sigma] 
            alpha_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * alpha_seed_archive[k];     // alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] 
            omega_set[j] = omega_seed_archive[k] / (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j]));      // omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) 
            my_daxpy(vec_loc_size, omega_set[j] / (pi_archive_set[j * max_iter + k] * zeta_set[j]), q_loc_copy, &x_loc_set[j * vec_loc_size]);     // x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q 
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + k]), q_loc_copy, &p_loc_set[j * vec_loc_size]);    // p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) 
            my_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + (k - 1)]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            zeta_set[j] = (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j])) * zeta_set[j];      // zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] 
            beta_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * beta_seed_archive[k]; // beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] 
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      // p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] 
            my_daxpy(vec_loc_size, 1.0 / (pi_archive_set[j * max_iter + k] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        shift_time += section_end_time - section_start_time;
        shift_iter_time = section_end_time - section_start_time;
#endif

#ifdef DISPLAY_SECTION_TIME
        if (myid == 0 && k == 1) {
            printf("iter, unsolved, seed, agv_1, mult_diag_1, mult_offd_1, agv_2, mult_diag_2, mult_offd_2, ared, shift\n");
        }

        if (myid == 0 && k % OUT_ITER == 0) {
            printf("%d, %d, %e, %e, %e, %e, %e, %e, %e, %e, %e\n", k, sigma_len - stop_count, seed_iter_time, agv_1_iter_time, mult_diag_1_iter_time, mult_offd_1_iter_time, agv_2_iter_time, mult_diag_2_iter_time, mult_offd_2_iter_time, ared_iter_time, shift_iter_time);
        }
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
#endif
        max_zeta_pi = 1.0;
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
                abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_archive_set[j * max_iter + k]));
            }
#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("%e ", abs_zeta_pi * sqrt(dot_r / dot_zero));
#endif
            if (abs_zeta_pi * abs_zeta_pi * dot_r <= tol * tol * dot_zero) {
                stop_flag[j] = true;
                stop_count++;
            } else {
                if (abs_zeta_pi > max_zeta_pi) {
                    max_zeta_pi = abs_zeta_pi;
                    max_sigma = j;
                }
            }
        }
#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) printf("\n");
#endif

#ifdef MEASURE_SECTION_TIME
        section_start_time = MPI_Wtime();
#endif

        // seed switching 
        if (stop_flag[seed] && stop_count < sigma_len) {
#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0) printf("Seed : %d\n", max_sigma);
#endif
            for (i = 1; i <= k; i++) {
                alpha_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * alpha_seed_archive[i];
                beta_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * beta_seed_archive[i];
                omega_seed_archive[i] = omega_seed_archive[i] / (1.0 - omega_seed_archive[i] * (sigma[seed] - sigma[max_sigma]));
            }
            my_dscal(vec_loc_size, 1.0 / (zeta_set[max_sigma] * pi_archive_set[max_sigma * max_iter + k]), r_loc);

            for (j = 0; j < sigma_len; j++) {
                eta_set[j]    = 0.0;  // eta[sigma]    <- 0 
                //pi_archive_set[j * max_iter + 0] = 1.0;
                zeta_set[j]   = 1.0;  // zeta[sigma]   <- 1 
            }
            //alpha_seed_archive[0] = 1.0;
            //beta_seed_archive[0]  = 0.0;

            for (i = 1; i <= k; i++) {
                for (j = 0; j < sigma_len; j++) {
                    if (stop_flag[j]) continue;
                    if (j == max_sigma) continue;
                    eta_set[j] = (beta_seed_archive[i - 1] / alpha_seed_archive[i - 1]) * alpha_seed_archive[i] * eta_set[j] - (sigma[max_sigma] - sigma[j]) * alpha_seed_archive[i] * pi_archive_set[j * max_iter + (i - 1)];
                    pi_archive_set[j * max_iter + i] = eta_set[j] + pi_archive_set[j * max_iter + (i - 1)];
                    zeta_set[j] = (1.0 - omega_seed_archive[i] * (sigma[max_sigma] - sigma[j])) * zeta_set[j];
                }
            }

            seed = max_sigma;
        }

#ifdef MEASURE_SECTION_TIME
        section_end_time = MPI_Wtime();
        switch_time += section_end_time - section_start_time;
#endif

        k++;

#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e\n", k, sqrt(dot_r / dot_zero));
        }
#endif

    }


#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
#endif

    if (myid == 0) {
#ifdef DISPLAY_RESIDUAL
        printf("Total iter   : %d\n", k - 1);
        printf("Final r      : %e\n", sqrt(dot_r / dot_zero));
#endif
#ifdef MEASURE_TIME
        printf("Total time   : %e [sec.] \n", total_time);
        //printf("Avg time/iter: %e [sec.] \n", total_time / k);
#endif
#ifdef MEASURE_SECTION_TIME
        printf("Seed time    : %e [sec.]\n", seed_time);
        printf(" 1 Agv time   : %e [sec.]\n", agv_1_time);
        printf(" 1 Mult_diag  : %e [sec.]\n", mult_diag_1_time);
        printf(" 1 Mult_offd  : %e [sec.]\n", mult_offd_1_time);
        printf(" 2 Agv time   : %e [sec.]\n", agv_2_time);
        printf(" 2 Mult_diag  : %e [sec.]\n", mult_diag_2_time);
        printf(" 2 Mult_offd  : %e [sec.]\n", mult_offd_2_time);
        printf(" Ared time    : %e [sec.]\n", ared_time);
        printf("Shift time   : %e [sec.]\n", shift_time);
        printf("Switch time  : %e [sec.]\n", switch_time);
#endif
    }

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(q_loc_copy);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set);
    free(alpha_seed_archive); free(beta_seed_archive); free(omega_seed_archive); free(pi_archive_set);
    free(stop_flag);

    return k;

}