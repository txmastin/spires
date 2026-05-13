#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stdlib.h>
#include "reservoir.h"

#ifdef __cplusplus
extern "C" {
#endif

// cpu (always available)
void   mat_vec_mult(double *A, double *x, double *y, size_t n);
void   cblas_mat_vec_mult(struct reservoir *r, const double *input_vector,
                          double *external_inputs);
double calc_spectral_radius(double *A, size_t n);
void   rescale_matrix(double *A, size_t n, double target_rho);
void   mat_transpose(double *A, double *A_T, size_t rows, size_t cols);
void   mat_mat_mult(double *A, double *B, double *C,
                    size_t r1, size_t c1, size_t c2);
int    solve_linear_system_lud(double *A, double *b, double *x, size_t n);

// cuda (requires compiling with USE_CUDA=1 flag)
#ifdef USE_CUDA
void cuda_init_reservoir(struct reservoir *r);
void cuda_step_reservoir(struct reservoir *r, const double *input_vector);
void cuda_get_spikes(struct reservoir *r, double *spike_out);
void cuda_free_reservoir(struct reservoir *r);
void cuda_copy_state(struct reservoir *r, double *out);
void call_peek();
void cuda_reset_reservoir(struct reservoir *r);
void cuda_alloc_state_buffer(struct reservoir *r, size_t series_length);
void cuda_collect_state(struct reservoir *r, size_t t);
void cuda_get_state_buffer(struct reservoir *r, double *h_X, size_t series_length);
void cuda_free_state_buffer(struct reservoir *r);
#endif

#ifdef __cplusplus
}
#endif

#endif /* MATH_UTILS_H */
