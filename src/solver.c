#include "solver.h"

#define  EPS        1.0e-9  /* 収束判定条件 */
#define  MAX_ITER   10000   /* 最大反復回数 */

#define  MEASURE_TIME       /* 時間計測 */

#define  DISPLAY_RESIDUAL   /* 残差表示 */
#define  OUT_ITER   100     /* 残差表示の反復間隔 */

/******************************************************************************
 * @fn      bicgstab
 * @brief   BiCGSTAB法
 * @param   A_loc_diag : 行列Aの対角ブロック
 * @param   A_loc_offd : 行列Aの非対角ブロック
 * @param   A_info     : 行列Aの情報
 * @param   x_loc      : 解ベクトル
 * @param   r_loc      : 右辺ベクトルおよび残差ベクトル
 * @return  実行した反復回数
 * @sa
 * @detail  行列、ベクトルを行方向に分割して受け取り、BiCGSTAB法を実行
 ******************************************************************************/
int bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc, double *r_loc) {

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

    int k, max_iter;
    double tol;
    double *Ax_loc, *r_hat_loc, *s_loc, *y_loc, *p_loc, *vec;
    double dot_r, dot_zero, rTr, rTs, rTy, yTy, rTr_old, alpha, beta, omega;
    MPI_Request dot_r_req, rTr_req, rTs_req, rTy_req, yTy_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    Ax_loc      = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    p_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    /* 初期化 */
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc); /* Ax <- A x */
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);    /* r <- b - Ax */
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);       /* r# <- r */
    my_dcopy(vec_loc_size, r_loc, p_loc);           /* p <- r */
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);      /* (r#,r) */
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;
    dot_zero = rTr;

    /* 反復 */
    while (dot_r > tol * tol * dot_zero && k < max_iter) {

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, p_loc, vec, s_loc);  /* s <- A p */
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  /* rTs <- (r#,s) */
        MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha = rTr / rTs;  /* alpha <- (r#,r)/(r#,s) */
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);   /* q <- r - alpha s */

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  /* y <- A q */
        rTy = my_ddot(vec_loc_size, r_loc, y_loc);  /* (q,y) */
        MPI_Iallreduce(MPI_IN_PLACE, &rTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTy_req);
        yTy = my_ddot(vec_loc_size, y_loc, y_loc);  /* (y,y) */
        MPI_Iallreduce(MPI_IN_PLACE, &yTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &yTy_req);
        MPI_Wait(&rTy_req, MPI_STATUS_IGNORE);
        MPI_Wait(&yTy_req, MPI_STATUS_IGNORE);

        omega = rTy / yTy;  /* omega <- (q,y)/(y,y) */
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);    /* x <- x + alpha p + omega q */
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);    /*                            */
        my_daxpy(vec_loc_size, -omega, y_loc, r_loc);   /* -------------------------- */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    /* (r,r) */
        MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  /* (r#,r) */
        MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta = (alpha / omega) * (rTr / rTr_old);   /* beta <- (alpha / omega) * ((r#,r)/(r#,r)) */
        my_dscal(vec_loc_size, beta, p_loc);                /* p <- r + beta p - beta omega s */
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc);          /*                                */
        my_daxpy(vec_loc_size, -beta*omega, s_loc, p_loc);  /* ------------------------------ */
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

    free(Ax_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(p_loc); free(vec);

    return k;
}

/******************************************************************************
 * @fn      ca_bicgstab
 * @brief   Communication avoiding BiCGSTAB法
 * @param   A_loc_diag : 行列Aの対角ブロック
 * @param   A_loc_offd : 行列Aの非対角ブロック
 * @param   A_info     : 行列Aの情報
 * @param   x_loc      : 解ベクトル
 * @param   r_loc      : 右辺ベクトルおよび残差ベクトル
 * @return  実行した反復回数
 * @sa
 * @detail  行列、ベクトルを行方向に分割して受け取り、CA-BiCGSTAB法を実行
 ******************************************************************************/
int ca_bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc, double *r_loc) {

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

    int k, max_iter;
    double tol;
    double *Ax_loc, *r_hat_loc, *s_loc, *z_loc, *w_loc, *p_loc, *vec;
    double dot_r, dot_zero, rTr, rTw, wTw, rTs, rTz, rTr_old, alpha, beta, omega;
    MPI_Request dot_r_req, rTr_req, rTw_req, wTw_req, rTs_req, rTz_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    Ax_loc      = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    z_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    w_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    p_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    /* 初期化 */
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc); /* Ax <- A x */
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);    /* r <- b - Ax */
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);       /* r# <- r */
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);   /* (r,r) */

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);  /* w <- A r */
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (r,w) */
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
    MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);

    alpha = rTr / rTw;  /* alpha <- (r,r)/(r,w) */
    beta = 0;           /* beta  <- 0 */
    dot_r = rTr;
    dot_zero = rTr;

    /* 反復 */
    while (dot_r > tol * tol * dot_zero && k < max_iter) {
        my_daxpy(vec_loc_size, -omega, s_loc, p_loc);   /* p <- r + beta (p - omega s) */
        my_dscal(vec_loc_size, beta, p_loc);            /*                             */
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc);      /* --------------------------- */
        my_daxpy(vec_loc_size, -omega, z_loc, s_loc);   /* s <- w + beta (s - omega z) */
        my_dscal(vec_loc_size, beta, s_loc);            /*                             */
        my_daxpy(vec_loc_size, 1.0, w_loc, s_loc);      /* --------------------------- */

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, s_loc, vec, z_loc);  /* z <- A s */
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);   /* q <- r - alpha s */
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);   /* y <- w - alpha z */
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);      MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (q,y) */
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);      MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);   /* (y,y) */
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;  /* omega <- (q,y)/(y,y) */
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);    /* x <- x + alpha p + omega q */
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);    /* -------------------------- */
        my_daxpy(vec_loc_size, -omega, w_loc, r_loc);   /* r <- q - omega y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);   /* (r,r) */

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);  /* w <- A r */
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);   /* (r#,r) */
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (r#,w) */
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);   /* (r#,s) */
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);   /* (r#,z) */
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTz_req, MPI_STATUS_IGNORE);
        beta = (alpha / omega) * (rTr / rTr_old);   /* beta <- (alpha / omega) * ((r#,r)/(r#,r)) */
        alpha = rTr / (rTw + beta * (rTs - omega * rTz));   /* beta <- (r#,r) / {(r#,r) + beta ((r#,s) - omega (r#,z))} */

        k++;

        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
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

    free(Ax_loc); free(r_hat_loc); free(s_loc); free(z_loc); free(w_loc); free(p_loc); free(vec);

    return k;
}

/******************************************************************************
 * @fn      pipe_bicgstab
 * @brief   Pipelined BiCGSTAB法
 * @param   A_loc_diag : 行列Aの対角ブロック
 * @param   A_loc_offd : 行列Aの非対角ブロック
 * @param   A_info     : 行列Aの情報
 * @param   x_loc      : 解ベクトル
 * @param   r_loc      : 右辺ベクトルおよび残差ベクトル
 * @return  実行した反復回数
 * @sa
 * @detail  行列、ベクトルを行方向に分割して受け取り、Pipelined BiCGSTAB法を実行
 ******************************************************************************/
int pipe_bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc, double *r_loc) {
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

    int k, max_iter;
    double tol;
    double *Ax_loc, *r_hat_loc, *s_loc, *z_loc, *w_loc, *p_loc, *v_loc, *t_loc, *vec;
    double dot_r, dot_zero, rTr, rTw, wTw, rTs, rTz, rTr_old, alpha, beta, omega;
    MPI_Request dot_r_req, rTr_req, rTw_req, wTw_req, rTs_req, rTz_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    Ax_loc      = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    z_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    w_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    p_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    v_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    t_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    /* 初期化 */
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc); /* Ax <- A x */
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);    /* r <- b - Ax */
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);       /* r# <- r */
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);   /* (r,r) */

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);  /* w <- A r */
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (r,w) */

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);  /* t <- A w */
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
    MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);

    alpha = rTr / rTw;  /* alpha <- (r,r)/(r,w) */
    beta = 0;           /* beta  <- 0 */
    dot_r = rTr;
    dot_zero = rTr;

    /* 反復 */
    while (dot_r > tol * tol * dot_zero && k < max_iter) {
        my_daxpy(vec_loc_size, -omega, s_loc, p_loc);   /* p <- r + beta (p - omega s) */
        my_dscal(vec_loc_size, beta, p_loc);            /*                             */
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc);      /* --------------------------- */
        my_daxpy(vec_loc_size, -omega, z_loc, s_loc);   /* s <- w + beta (s - omega z) */
        my_dscal(vec_loc_size, beta, s_loc);            /*                             */
        my_daxpy(vec_loc_size, 1.0, w_loc, s_loc);      /* --------------------------- */
        my_daxpy(vec_loc_size, -omega, v_loc, z_loc);   /* z <- t + beta (z - omega v) */
        my_dscal(vec_loc_size, beta, z_loc);            /*                             */
        my_daxpy(vec_loc_size, 1.0, t_loc, z_loc);      /* --------------------------- */
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);   /* q <- r - alpha s */
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);   /* y <- w - alpha z */
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (q,y) */
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);   /* (y,y) */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, z_loc, vec, v_loc);  /* v <- A z ドット積の通信をオーバーラップ */
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;  /* omega <- (q,y)/(y,y) */
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);    /* x <- x + alpha p + omega q */
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);    /* -------------------------- */
        my_daxpy(vec_loc_size, -omega, w_loc, r_loc);   /* r <- q - omega y */
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);   /* (r,r) */
        my_daxpy(vec_loc_size, -alpha, v_loc, t_loc);   /* w <- y - omega (t - alpha v) */
        my_daxpy(vec_loc_size, -omega, t_loc, w_loc);   /* ---------------------------- */
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);   /* (r#,r) */
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (r#,w) */
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);   /* (r#,s) */
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);   /* (r#,z) */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);  /* t <- A w ドット積の通信をオーバーラップ */
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTz_req, MPI_STATUS_IGNORE);

        beta = (alpha / omega) * (rTr / rTr_old);           /* beta <- (alpha / omega) * ((r#,r)/(r#,r)) */
        alpha = rTr / (rTw + beta * (rTs - omega * rTz));   /* alpha <- (r#,r) / {(r#,w) + beta ((r#,s) - omega (r#,z))} */

        k++;

        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
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

    free(Ax_loc); free(r_hat_loc); free(s_loc); free(z_loc); free(w_loc); free(p_loc); free(v_loc); free(t_loc); free(vec);

    return k;
}

/******************************************************************************
 * @fn      pipe_bicgstab_rr
 * @brief   Pipelined BiCGSTAB法 with Residual replacement
 * @param   A_loc_diag : 行列Aの対角ブロック
 * @param   A_loc_offd : 行列Aの非対角ブロック
 * @param   A_info     : 行列Aの情報
 * @param   x_loc      : 解ベクトル
 * @param   r_loc      : 右辺ベクトルおよび残差ベクトル
 * @param   krr        : 残差置換の周期
 * @param   nrr        : 残差置換を行う最大回数
 * @return  実行した反復回数
 * @sa
 * @detail  行列、ベクトルを行方向に分割して受け取り、残差置換付きPipelined BiCGSTAB法を実行
 ******************************************************************************/
int pipe_bicgstab_rr(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc, double *r_loc, int krr, int nrr) {
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

    int k, max_iter;
    double tol;
    double *b_loc, *Ax_loc, *r_hat_loc, *s_loc, *z_loc, *w_loc, *p_loc, *v_loc, *t_loc, *vec;
    double dot_r, dot_zero, rTr, rTw, wTw, rTs, rTz, rTr_old, alpha, beta, omega;
    MPI_Request dot_r_req, rTr_req, rTw_req, wTw_req, rTs_req, rTz_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    b_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    Ax_loc      = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    z_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    w_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    p_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    v_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    t_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    /* 初期化 */
    my_dcopy(vec_loc_size, r_loc, b_loc);  /* b <- r */
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc); /* Ax <- A x */
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);    /* r <- b - Ax */
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);       /* r# <- r */
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);   /* (r,r) */

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);  /* w <- A r */
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (r,w) */

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);  /* t <- A w */
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
    MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);

    alpha = rTr / rTw;  /* alpha <- (r,r)/(r,w) */
    beta = 0;           /* beta  <- 0 */
    dot_r = rTr;
    dot_zero = rTr;

    while (dot_r > tol * tol * dot_zero && k < max_iter) {
        my_daxpy(vec_loc_size, -omega, s_loc, p_loc);   /* p <- r + beta (p - omega s) */
        my_dscal(vec_loc_size, beta, p_loc);            /*                             */
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc);      /* --------------------------- */

        if (k % krr == 0 && k > 0 && k <= krr * nrr) {  /* 残差置換 */
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, p_loc, vec, s_loc);  /* s <- A p */
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, s_loc, vec, z_loc);  /* z <- A s */
        } else {
            my_daxpy(vec_loc_size, -omega, z_loc, s_loc);   /* s <- w + beta (s - omega z) */
            my_dscal(vec_loc_size, beta, s_loc);            /*                             */
            my_daxpy(vec_loc_size, 1.0, w_loc, s_loc);      /* --------------------------- */
            my_daxpy(vec_loc_size, -omega, v_loc, z_loc);   /* z <- t + beta (z - omega v) */
            my_dscal(vec_loc_size, beta, z_loc);            /*                             */
            my_daxpy(vec_loc_size, 1.0, t_loc, z_loc);      /* --------------------------- */
        }

        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);       /* q <- r - alpha s */
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);       /* y <- w - alpha z */
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (q,y) */
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);   /* (y,y) */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, z_loc, vec, v_loc);  /* v <- A z ドット積の通信をオーバーラップ */
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;  /* omega <- (q,y)/(y,y) */
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);    /* x <- x + alpha p + omega q */
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);    /* -------------------------- */

        if (k % krr == 0 && k > 0 && k <= krr * nrr) {  /* 残差置換 */
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc); /* Ax <- A x */
            my_dcopy(vec_loc_size, b_loc, r_loc);   /* r <- b */
            my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);    /* r <- b - Ax */
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);  /* w <- A r */
        } else {
            my_daxpy(vec_loc_size, -omega, w_loc, r_loc);   /* r <- q - omega y */
            my_daxpy(vec_loc_size, -alpha, v_loc, t_loc);   /* w <- y - omega (t - alpha v) */
            my_daxpy(vec_loc_size, -omega, t_loc, w_loc);   /* ---------------------------- */
        }

        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);   /* (r,r) */
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);   /* (r#,r) */
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);   /* (r#,w) */
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);   /* (r#,s) */
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);   /* (r#,z) */
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);  /* t <- A w ドット積の通信をオーバーラップ */
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTz_req, MPI_STATUS_IGNORE);

        beta = (alpha / omega) * (rTr / rTr_old);           /* beta <- (alpha / omega) * ((r#,r)/(r#,r)) */
        alpha = rTr / (rTw + beta * (rTs - omega * rTz));   /* alpha <- (r#,r) / {(r#,w) + beta ((r#,s) - omega (r#,z))} */

        k++;

        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
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

    free(b_loc);
    free(Ax_loc); free(r_hat_loc); free(s_loc); free(z_loc); free(w_loc); free(p_loc); free(v_loc); free(t_loc); free(vec);

    return k;
}