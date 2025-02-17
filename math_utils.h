#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stdlib.h>

void mat_vec_mult(double *A, double *x, double *y, size_t n);
double calc_spectral_radius(double *A, size_t n);
void rescale_matrix(double *A, size_t n, double target_rho);

#endif // MATH_UTILS_H
