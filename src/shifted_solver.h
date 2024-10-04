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

int shifted_bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, DENSE_Matrix *X_loc, double *r_loc, double *sigma);

#endif