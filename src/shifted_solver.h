#ifndef SHIFTED_SOLVER_H
#define SHIFTED_SOLVER_H

#include <math.h>
#include <time.h>

#include "matrix.h"
#include "vector.h"

typedef struct {
    unsigned int rows;
    unsigned int cols;
    double *val;
} DENSE_Matrix;

int shifted_bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len);
int shifted_lopbicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);
int shifted_lopbicgstab_v2(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);
int shifted_pipe_lopbicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);

#endif