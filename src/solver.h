#ifndef SOLVER_H
#define SOLVER_H

#include <math.h>
#include <time.h> 

#include "matrix.h"
#include "vector.h"

int bicgstab(CSR_Matrix *A_loc, INFO_Matrix *A_info, double *x, double *r);
int ca_bicgstab(CSR_Matrix *A_loc, INFO_Matrix *A_info, double *x, double *r);
int pipe_bicgstab(CSR_Matrix *A_loc, INFO_Matrix *A_info, double *x, double *r);
int pipe_bicgstab_rr(CSR_Matrix *A_loc, INFO_Matrix *A_info, double *x, double *r, int krr, int nrr);

#endif