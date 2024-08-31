#ifndef SOLVER_H
#define SOLVER_H

#include <math.h>
#include <time.h> 

#include "matrix.h"
#include "vector.h"

int bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc, double *r_loc);
int ca_bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc, double *r_loc);
int pipe_bicgstab(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc, double *r_loc);
int pipe_bicgstab_rr(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc, double *r_loc, int krr, int nrr);

#endif