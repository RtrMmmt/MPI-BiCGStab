#include "solver.h"

#define  EPS        1.0e-9
#define  MAX_ITER   4000
#define  OUT_ITER   100
#define  MEASURE_TIME
//#define  DISPLAY_RESIDUAL

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

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc);
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);
    my_dcopy(vec_loc_size, r_loc, p_loc);
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;
    dot_zero = rTr;

    while (dot_r > tol * tol * dot_zero && k < max_iter) {

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, p_loc, vec, s_loc);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha = rTr / rTs;
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);
        rTy = my_ddot(vec_loc_size, r_loc, y_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &rTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTy_req);
        yTy = my_ddot(vec_loc_size, y_loc, y_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &yTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &yTy_req);
        MPI_Wait(&rTy_req, MPI_STATUS_IGNORE);
        MPI_Wait(&yTy_req, MPI_STATUS_IGNORE);

        omega = rTy / yTy;
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);
        my_daxpy(vec_loc_size, -omega, y_loc, r_loc);
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

        beta = (alpha / omega) * (rTr / rTr_old);
        my_dscal(vec_loc_size, beta, p_loc);
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc);
        my_daxpy(vec_loc_size, -beta*omega, s_loc, p_loc);
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

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc);
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
    MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);

    alpha = rTr / rTw;
    beta = 0;
    dot_r = rTr;
    dot_zero = rTr;

    while (dot_r > tol * tol * dot_zero && k < max_iter) {
        my_daxpy(vec_loc_size, -omega, s_loc, p_loc);
        my_dscal(vec_loc_size, beta, p_loc);
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc);
        my_daxpy(vec_loc_size, -omega, z_loc, s_loc);
        my_dscal(vec_loc_size, beta, s_loc);
        my_daxpy(vec_loc_size, 1.0, w_loc, s_loc);
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, s_loc, vec, z_loc);
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);      MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);      MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);
        my_daxpy(vec_loc_size, -omega, w_loc, r_loc);
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTz_req, MPI_STATUS_IGNORE);
        beta = (alpha / omega) * (rTr / rTr_old);
        alpha = rTr / (rTw + beta * (rTs - omega * rTz));

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

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc);
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
    MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);

    alpha = rTr / rTw;
    beta = 0;
    dot_r = rTr;
    dot_zero = rTr;

    while (dot_r > tol * tol * dot_zero && k < max_iter) {
        my_daxpy(vec_loc_size, -omega, s_loc, p_loc);
        my_dscal(vec_loc_size, beta, p_loc);
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc);
        my_daxpy(vec_loc_size, -omega, z_loc, s_loc);
        my_dscal(vec_loc_size, beta, s_loc);
        my_daxpy(vec_loc_size, 1.0, w_loc, s_loc);
        my_daxpy(vec_loc_size, -omega, v_loc, z_loc);
        my_dscal(vec_loc_size, beta, z_loc);
        my_daxpy(vec_loc_size, 1.0, t_loc, z_loc);
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, z_loc, vec, v_loc);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);
        my_daxpy(vec_loc_size, -omega, w_loc, r_loc);
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        my_daxpy(vec_loc_size, -alpha, v_loc, t_loc);
        my_daxpy(vec_loc_size, -omega, t_loc, w_loc);
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTz_req, MPI_STATUS_IGNORE);

        beta = (alpha / omega) * (rTr / rTr_old);
        alpha = rTr / (rTw + beta * (rTs - omega * rTz));

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

    my_dcopy(vec_loc_size, r_loc, b_loc);
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc);
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);

    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
    MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);

    alpha = rTr / rTw;
    beta = 0;
    dot_r = rTr;
    dot_zero = rTr;

    while (dot_r > tol * tol * dot_zero && k < max_iter) {
        my_daxpy(vec_loc_size, -omega, s_loc, p_loc);
        my_dscal(vec_loc_size, beta, p_loc);
        my_daxpy(vec_loc_size, 1.0, r_loc, p_loc);

        if (k % krr == 0 && k > 0 && k <= krr * nrr) {
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, p_loc, vec, s_loc);
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, s_loc, vec, z_loc);
        } else {
            my_daxpy(vec_loc_size, -omega, z_loc, s_loc);
            my_dscal(vec_loc_size, beta, s_loc);
            my_daxpy(vec_loc_size, 1.0, w_loc, s_loc);
            my_daxpy(vec_loc_size, -omega, v_loc, z_loc);
            my_dscal(vec_loc_size, beta, z_loc);
            my_daxpy(vec_loc_size, 1.0, t_loc, z_loc);
        }

        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, z_loc, vec, v_loc);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);

        if (k % krr == 0 && k > 0 && k <= krr * nrr) {
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, x_loc, vec, Ax_loc);
            my_dcopy(vec_loc_size, b_loc, r_loc);
            my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, w_loc);
        } else {
            my_daxpy(vec_loc_size, -omega, w_loc, r_loc);
            my_daxpy(vec_loc_size, -alpha, v_loc, t_loc);
            my_daxpy(vec_loc_size, -omega, t_loc, w_loc);
        }

        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, w_loc, vec, t_loc);
        MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        MPI_Wait(&rTz_req, MPI_STATUS_IGNORE);

        beta = (alpha / omega) * (rTr / rTr_old);
        alpha = rTr / (rTw + beta * (rTs - omega * rTz));

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