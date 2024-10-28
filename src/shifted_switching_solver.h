#ifndef SHIFTED_SOLVER_H
#define SHIFTED_SOLVER_H

#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "matrix.h"
#include "vector.h"

int shifted_lopbicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);

#endif