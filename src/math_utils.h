#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stdlib.h>

void mat_vec_mult(double *A, double *x, double *y, size_t n);
double calc_spectral_radius(double *A, size_t n);
void rescale_matrix(double *A, size_t n, double target_rho);
void mat_transpose(double *A, double *A_T, size_t rows, size_t cols);
void mat_mat_mult(double *A, double *B, double *C, size_t r1, size_t c1, size_t c2);
int solve_linear_system_lud(double *A, double *b, double *x, size_t n);

#endif // MATH_UTILS_H
