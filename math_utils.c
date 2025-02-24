#include "math_utils.h"
#include <stdlib.h>
#include <math.h>

// Matrix-vector multiplication: y = A * x
void mat_vec_mult(double *A, double *x, double *y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = 0.0;
        for (size_t j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// Power Iteration to estimate spectral radius
double calc_spectral_radius(double *A, size_t n) {

    const unsigned int MAX_ITER = 1000;
    const double TOLERANCE = 1e-6;
 
    double* x = (double *)malloc(n * sizeof(double));
    double* y = (double *)malloc(n * sizeof(double));

    // Initialize x with 1s
    for (size_t i = 0; i < n; i++) x[i] = 1.0;

    double lambda_old = 0.0, lambda_new = 0.0;
    
    for (unsigned int iter = 0; iter < MAX_ITER; iter++) {
        mat_vec_mult(A, x, y, n);  // y = A * x

        // Compute largest magnitude value in y (Rayleigh Quotient estimate)
        lambda_new = 0.0;
        for (size_t i = 0; i < n; i++) {
            if (fabs(y[i]) > lambda_new) {
                lambda_new = fabs(y[i]);
                //double max_index = i;
            }
        }

        // Normalize y
        for (size_t i = 0; i < n; i++) x[i] = y[i] / lambda_new;

        // Convergence check
        if (fabs(lambda_new - lambda_old) < TOLERANCE) break;
        lambda_old = lambda_new;
    }

    free(x);
    free(y);
    return lambda_new;
}

void rescale_matrix(double* A, size_t n, double target_rho) {
    double rho = calc_spectral_radius(A, n);
    double rescale_factor = target_rho / rho;

    // rescale all values, such that the matrix has 
    // spectral radius = target_rho
    for (size_t i = 0; i < (n * n); i++) {
        A[i] *= rescale_factor;  
    }
}

