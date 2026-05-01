/*
#include <stdlib.h>
#include <reservoir.h>

#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#ifdef USE_CUDA
  #ifdef __CUDACC__
  __global__ void mat_mat_mult_kernel(const double * __restrict__ A,
                                    const double * __restrict__ B,
                                    double       * __restrict__ C,
                                    size_t r1, size_t c1, size_t c2);
                                    
  
  #endif
  #ifdef __cplusplus
  extern "C" {
  #endif
  void cuda_malloc_init(struct reservoir *r);
  void cublas_dot_product(size_t num_neurons, 
                        const double *W_row, 
                        const int a,  
                        const double *last_spikes, 
                        const int b,
                        double *recurrent_input);
  void cublas_matvec(struct reservoir *r,
                               const double *last_spikes,    
                               double *recurrent_inputs); 
  void cuda_init_reservoir(struct reservoir *r);
  void cuda_step_reservoir(struct reservoir *r, const double* input_vector);
  void cuda_free_reservoir(struct reservoir *r);
  #ifdef __cplusplus
  }
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif
  void mat_vec_mult(double *A, double *x, double *y, size_t n);
  void cblas_mat_vec_mult(struct reservoir *r, const double *input_vector, double *external_inputs);
  double calc_spectral_radius(double *A, size_t n);
  void rescale_matrix(double *A, size_t n, double target_rho);
  void mat_transpose(double *A, double *A_T, size_t rows, size_t cols);
  void mat_mat_mult(double *A, double *B, double *C, size_t r1, size_t c1, size_t c2);
  int solve_linear_system_lud(double *A, double *b, double *x, size_t n);

#ifdef __cplusplus
}
#endif

#endif // MATH_UTILS_H
*/

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stdlib.h>
#include "reservoir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── CPU math (always available) ─────────────────────────────────────── */
void   mat_vec_mult(double *A, double *x, double *y, size_t n);
void   cblas_mat_vec_mult(struct reservoir *r, const double *input_vector,
                          double *external_inputs);
double calc_spectral_radius(double *A, size_t n);
void   rescale_matrix(double *A, size_t n, double target_rho);
void   mat_transpose(double *A, double *A_T, size_t rows, size_t cols);
void   mat_mat_mult(double *A, double *B, double *C,
                    size_t r1, size_t c1, size_t c2);
int    solve_linear_system_lud(double *A, double *b, double *x, size_t n);

/* ── CUDA backend (only when compiled in) ────────────────────────────── */
#ifdef USE_CUDA
void cuda_init_reservoir(struct reservoir *r);
void cuda_step_reservoir(struct reservoir *r, const double *input_vector);
void cuda_get_spikes(struct reservoir *r, double *spike_out);
void cuda_free_reservoir(struct reservoir *r);
void cuda_copy_state(struct reservoir *r, double *out);
void call_peek();
void cuda_reset_reservoir(struct reservoir *r);
#endif

#ifdef __cplusplus
}
#endif

#endif /* MATH_UTILS_H */
