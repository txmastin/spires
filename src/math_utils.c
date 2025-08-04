#include "math_utils.h"
#include <stdlib.h>
#include <stdio.h>
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

/**
 * @brief Transposes a matrix.
 * @param A The input matrix (rows x cols).
 * @param A_T The output transposed matrix (cols x rows).
 */
void mat_transpose(double *A, double *A_T, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

/**
 * @brief Multiplies two matrices: C = A * B.
 * @param A Input matrix of size (r1 x c1).
 * @param B Input matrix of size (c1 x c2).
 * @param C Output matrix of size (r1 x c2).
 */
void mat_mat_mult(double *A, double *B, double *C, size_t r1, size_t c1, size_t c2) {
    for (size_t i = 0; i < r1; i++) {
        for (size_t j = 0; j < c2; j++) {
            C[i * c2 + j] = 0.0;
            for (size_t k = 0; k < c1; k++) {
                C[i * c2 + j] += A[i * c1 + k] * B[k * c2 + j];
            }
        }
    }
}

/**
 * @brief Solves a system of linear equations Ax = b using LU decomposition.
 * This function decomposes A into L and U, then solves Ly = b (forward substitution)
 * and finally Ux = y (backward substitution).
 *
 * @param A The n x n coefficient matrix. This matrix will be modified in place.
 * @param b The n x 1 vector of constants.
 * @param x The n x 1 solution vector (output).
 * @param n The size of the system.
 * @return 0 on success, -1 on failure (e.g., singular matrix).
 */
int solve_linear_system_lud(double *A, double *b, double *x, size_t n) {
    // --- Step 1: LU Decomposition (Doolittle's method) ---
    for (size_t i = 0; i < n; i++) {
        // Upper Triangle
        for (size_t k = i; k < n; k++) {
            double sum = 0.0;
            for (size_t j = 0; j < i; j++) {
                sum += A[i * n + j] * A[j * n + k];
            }
            A[i * n + k] = A[i * n + k] - sum;
        }
        // Lower Triangle
        for (size_t k = i + 1; k < n; k++) {
            if (A[i * n + i] == 0.0) {
                fprintf(stderr, "Error: Matrix is singular and cannot be inverted.\n");
                return -1; // Avoid division by zero
            }
            double sum = 0.0;
            for (size_t j = 0; j < i; j++) {
                sum += A[k * n + j] * A[j * n + i];
            }
            A[k * n + i] = (A[k * n + i] - sum) / A[i * n + i];
        }
    }

    // --- Step 2: Forward substitution (solves Ly = b for y) ---
    double y[n];
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < i; j++) {
            sum += A[i * n + j] * y[j];
        }
        y[i] = b[i] - sum;
    }

    // --- Step 3: Backward substitution (solves Ux = y for x) ---
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < (int)n; j++) {
            sum += A[i * n + j] * x[j];
        }
        if (A[i * n + i] == 0.0) {
            fprintf(stderr, "Error: Matrix is singular.\n");
            return -1;
        }
        x[i] = (y[i] - sum) / A[i * n + i];
    }

    return EXIT_SUCCESS; // Success
}
