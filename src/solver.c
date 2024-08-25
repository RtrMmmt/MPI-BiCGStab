#include "solver.h"

#define  EPS        1.0e-15
#define  MAX_ITER   4000
#define  OUT_ITER   100
#define  MEASURE_TIME
#define  DISPLAY_RESIDUAL

int bicgstab(CSR_Matrix *A_loc, INFO_Matrix *A_info, double *x, double *r) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#ifdef MEASURE_TIME
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc->rows;

    int k, max_iter;
    double tol;
    double *x_loc, *r_loc, *Ax_loc, *r_hat_loc, *s_loc, *y_loc, *p_loc, *p;
    double dot_r, dot_zero, rTr, rTs, rTy, yTy, rTr_old, alpha, beta, omega;
    MPI_Request r_req, p_req, dot_r_req, rTr_req, rTs_req, rTy_req, yTy_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    x_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    r_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    Ax_loc      = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    p_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    p           = (double *)malloc(vec_size * sizeof(double)); //pはrで代用できるが、わかりやすいので用意

    int start_idx = A_info->displs[myid];
    for (int i = 0; i < vec_loc_size; i++) {
        r_loc[i] = r[start_idx + i];
    }

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    csr_mv(A_loc, x, Ax_loc);
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
    MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, p, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &p_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);
    my_dcopy(vec_loc_size, r_loc, p_loc);
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);
    MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    dot_r = rTr;
    dot_zero = rTr;

    while (dot_r > tol * tol * dot_zero && k < max_iter) {
        MPI_Wait(&p_req, MPI_STATUS_IGNORE);

        csr_mv(A_loc, p, s_loc);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);

        alpha = rTr / rTs;
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);
        MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, r, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &r_req);
        MPI_Wait(&r_req, MPI_STATUS_IGNORE);

        csr_mv(A_loc, r, y_loc);
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
        MPI_Iallgatherv(p_loc, vec_loc_size, MPI_DOUBLE, p, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &p_req);
        k++;

#ifdef DISPLAY_RESIDUAL
        if (myid == 0 && k % OUT_ITER == 0) {
            printf("Iteration: %d, Residual: %e\n", k, sqrt(dot_r / dot_zero));
        }
#endif
    }

    MPI_Wait(&p_req, MPI_STATUS_IGNORE);

#ifdef MEASURE_TIME
        end_time = MPI_Wtime();
        total_time = end_time - start_time;
#endif

    MPI_Allgatherv(x_loc, vec_loc_size, MPI_DOUBLE, x, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("Final Residual: %e\n", sqrt(dot_r / dot_zero));
#ifdef MEASURE_TIME
        printf("Total time: %e seconds\n", total_time);
        printf("Average time per iteration: %e seconds\n", total_time / k);
#endif
    }

    free(x_loc); free(r_loc);
    free(Ax_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(p_loc); free(p);

    return k;
}

int ca_bicgstab(CSR_Matrix *A_loc, INFO_Matrix *A_info, double *x, double *r) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#ifdef MEASURE_TIME
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc->rows;

    int k, max_iter;
    double tol;
    double *x_loc, *r_loc, *Ax_loc, *r_hat_loc, *s_loc, *z_loc, *w_loc, *p_loc, *s;
    double dot_r, dot_zero, rTr, rTw, wTw, rTs, rTz, rTr_old, alpha, beta, omega;
    MPI_Request r_req, s_req, dot_r_req, rTr_req, rTw_req, wTw_req, rTs_req, rTz_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    x_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    r_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    Ax_loc      = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    z_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    w_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    p_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    s           = (double *)malloc(vec_size * sizeof(double)); //sはrで代用できるが、わかりやすいので用意

    int start_idx = A_info->displs[myid];
    for (int i = 0; i < vec_loc_size; i++) {
        r_loc[i] = r[start_idx + i];
    }

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    csr_mv(A_loc, x, Ax_loc);
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
    MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, r, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &r_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    MPI_Wait(&r_req, MPI_STATUS_IGNORE);

    csr_mv(A_loc, r, w_loc);
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
        MPI_Iallgatherv(s_loc, vec_loc_size, MPI_DOUBLE, s, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &s_req);
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);
        MPI_Wait(&s_req, MPI_STATUS_IGNORE);

        csr_mv(A_loc, s, z_loc);
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);      MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);      MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);
        my_daxpy(vec_loc_size, -omega, w_loc, r_loc);
        MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, r, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &r_req);
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        MPI_Wait(&r_req, MPI_STATUS_IGNORE);

        csr_mv(A_loc, r, w_loc);
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

    MPI_Allgatherv(x_loc, vec_loc_size, MPI_DOUBLE, x, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("Final Residual: %e\n", sqrt(dot_r / dot_zero));
#ifdef MEASURE_TIME
        printf("Total time: %e seconds\n", total_time);
        printf("Average time per iteration: %e seconds\n", total_time / k);
#endif
    }

    free(x_loc); free(r_loc);
    free(Ax_loc); free(r_hat_loc); free(s_loc); free(z_loc); free(w_loc); free(p_loc); free(s);

    return k;
}

int pipe_bicgstab(CSR_Matrix *A_loc, INFO_Matrix *A_info, double *x, double *r) {
    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#ifdef MEASURE_TIME
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc->rows;

    int k, max_iter;
    double tol;
    double *x_loc, *r_loc, *Ax_loc, *r_hat_loc, *s_loc, *z_loc, *w_loc, *p_loc, *v_loc, *t_loc, *z, *w;
    double dot_r, dot_zero, rTr, rTw, wTw, rTs, rTz, rTr_old, alpha, beta, omega;
    MPI_Request r_req, z_req, w_req, dot_r_req, rTr_req, rTw_req, wTw_req, rTs_req, rTz_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    x_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    r_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    Ax_loc      = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    z_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    w_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    p_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    v_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    t_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    z           = (double *)malloc(vec_size * sizeof(double));
    w           = (double *)malloc(vec_size * sizeof(double));

    int start_idx = A_info->displs[myid];
    for (int i = 0; i < vec_loc_size; i++) {
        r_loc[i] = r[start_idx + i];
    }

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    csr_mv(A_loc, x, Ax_loc);
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
    MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, r, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &r_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    MPI_Wait(&r_req, MPI_STATUS_IGNORE);

    csr_mv(A_loc, r, w_loc);
    MPI_Iallgatherv(w_loc, vec_loc_size, MPI_DOUBLE, w, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &w_req);
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
    MPI_Wait(&w_req, MPI_STATUS_IGNORE);

    csr_mv(A_loc, w, t_loc);
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
        MPI_Iallgatherv(z_loc, vec_loc_size, MPI_DOUBLE, z, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &z_req);
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);
        MPI_Wait(&z_req, MPI_STATUS_IGNORE);
        csr_mv(A_loc, z, v_loc);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);
        my_daxpy(vec_loc_size, -omega, w_loc, r_loc);
        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        my_daxpy(vec_loc_size, -alpha, v_loc, t_loc);
        my_daxpy(vec_loc_size, -omega, t_loc, w_loc);
        MPI_Iallgatherv(w_loc, vec_loc_size, MPI_DOUBLE, w, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &w_req);
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);
        MPI_Wait(&w_req, MPI_STATUS_IGNORE);
        csr_mv(A_loc, w, t_loc);
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

    MPI_Allgatherv(x_loc, vec_loc_size, MPI_DOUBLE, x, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("Final Residual: %e\n", sqrt(dot_r / dot_zero));
#ifdef MEASURE_TIME
        printf("Total time: %e seconds\n", total_time);
        printf("Average time per iteration: %e seconds\n", total_time / k);
#endif
    }

    free(x_loc); free(r_loc);
    free(Ax_loc); free(r_hat_loc); free(s_loc); free(z_loc); free(w_loc); free(p_loc); free(v_loc); free(t_loc); free(z); free(w);

    return k;
}

int pipe_bicgstab_rr(CSR_Matrix *A_loc, INFO_Matrix *A_info, double *x, double *r, int krr, int nrr) {
    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#ifdef MEASURE_TIME
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc->rows;

    int k, max_iter;
    double tol;
    double *b_loc, *x_loc, *r_loc, *Ax_loc, *r_hat_loc, *s_loc, *z_loc, *w_loc, *p_loc, *v_loc, *t_loc, *z, *w, *p, *s;
    double dot_r, dot_zero, rTr, rTw, wTw, rTs, rTz, rTr_old, alpha, beta, omega;
    MPI_Request r_req, z_req, w_req, p_req, dot_r_req, rTr_req, rTw_req, wTw_req, rTs_req, rTz_req;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;

    b_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    x_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    r_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    Ax_loc      = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    z_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    w_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    p_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    v_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    t_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    z           = (double *)malloc(vec_size * sizeof(double));
    w           = (double *)malloc(vec_size * sizeof(double));
    p           = (double *)malloc(vec_size * sizeof(double));
    s           = (double *)malloc(vec_size * sizeof(double));

    int start_idx = A_info->displs[myid];
    for (int i = 0; i < vec_loc_size; i++) {
        r_loc[i] = r[start_idx + i];
    }

#ifdef MEASURE_TIME
        start_time = MPI_Wtime();
#endif

    my_dcopy(vec_loc_size, r_loc, b_loc);
    csr_mv(A_loc, x, Ax_loc);
    my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
    MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, r, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &r_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);
    rTr = my_ddot(vec_loc_size, r_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    MPI_Wait(&r_req, MPI_STATUS_IGNORE);

    csr_mv(A_loc, r, w_loc);
    MPI_Iallgatherv(w_loc, vec_loc_size, MPI_DOUBLE, w, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &w_req);
    rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
    MPI_Wait(&w_req, MPI_STATUS_IGNORE);

    csr_mv(A_loc, w, t_loc);
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
            MPI_Allgatherv(p_loc, vec_loc_size, MPI_DOUBLE, p, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
            csr_mv(A_loc, p, s_loc);
            MPI_Allgatherv(s_loc, vec_loc_size, MPI_DOUBLE, s, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
            csr_mv(A_loc, s, z_loc);
        } else {
            my_daxpy(vec_loc_size, -omega, z_loc, s_loc);
            my_dscal(vec_loc_size, beta, s_loc);
            my_daxpy(vec_loc_size, 1.0, w_loc, s_loc);
            my_daxpy(vec_loc_size, -omega, v_loc, z_loc);
            my_dscal(vec_loc_size, beta, z_loc);
            my_daxpy(vec_loc_size, 1.0, t_loc, z_loc);
        }

        MPI_Iallgatherv(z_loc, vec_loc_size, MPI_DOUBLE, z, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &z_req);
        my_daxpy(vec_loc_size, -alpha, s_loc, r_loc);
        my_daxpy(vec_loc_size, -alpha, z_loc, w_loc);
        rTw = my_ddot(vec_loc_size, r_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        wTw = my_ddot(vec_loc_size, w_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &wTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &wTw_req);
        MPI_Wait(&z_req, MPI_STATUS_IGNORE);
        csr_mv(A_loc, z, v_loc);
        MPI_Wait(&rTw_req, MPI_STATUS_IGNORE);
        MPI_Wait(&wTw_req, MPI_STATUS_IGNORE);

        omega = rTw / wTw;
        my_daxpy(vec_loc_size, alpha, p_loc, x_loc);
        my_daxpy(vec_loc_size, omega, r_loc, x_loc);

        if (k % krr == 0 && k > 0 && k <= krr * nrr) {
            MPI_Allgatherv(x_loc, vec_loc_size, MPI_DOUBLE, x, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
            csr_mv(A_loc, x, Ax_loc);
            my_dcopy(vec_loc_size, b_loc, r_loc);
            my_daxpy(vec_loc_size, -1.0, Ax_loc, r_loc);
            MPI_Allgatherv(r_loc, vec_loc_size, MPI_DOUBLE, r, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
            csr_mv(A_loc, r, w_loc);
        } else {
            my_daxpy(vec_loc_size, -omega, w_loc, r_loc);
            my_daxpy(vec_loc_size, -alpha, v_loc, t_loc);
            my_daxpy(vec_loc_size, -omega, t_loc, w_loc);
        }

        dot_r = my_ddot(vec_loc_size, r_loc, r_loc);    MPI_Iallreduce(MPI_IN_PLACE, &dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);
        MPI_Iallgatherv(w_loc, vec_loc_size, MPI_DOUBLE, w, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &w_req);
        rTr_old = rTr;
        rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
        rTw = my_ddot(vec_loc_size, r_hat_loc, w_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTw_req);
        rTs = my_ddot(vec_loc_size, r_hat_loc, s_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);
        rTz = my_ddot(vec_loc_size, r_hat_loc, z_loc);  MPI_Iallreduce(MPI_IN_PLACE, &rTz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTz_req);
        MPI_Wait(&w_req, MPI_STATUS_IGNORE);
        csr_mv(A_loc, w, t_loc);
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

    MPI_Allgatherv(x_loc, vec_loc_size, MPI_DOUBLE, x, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("Final Residual: %e\n", sqrt(dot_r / dot_zero));
#ifdef MEASURE_TIME
        printf("Total time: %e seconds\n", total_time);
        printf("Average time per iteration: %e seconds\n", total_time / k);
#endif
    }

    free(b_loc); free(x_loc); free(r_loc);
    free(Ax_loc); free(r_hat_loc); free(s_loc); free(z_loc); free(w_loc); free(p_loc); free(v_loc); free(t_loc); free(z); free(w); free(p); free(s);

    return k;
}