#ifndef SYNAPSE_SIMPLE_H
#define SYNAPSE_SIMPLE_H

#include <stdlib.h>

/* SYNAPSE_SIMPLE: instantaneous scalar-weight multiply (recurrent_input =
 * sum_j w_ij * spike_j). No params used.
 *
 * Opaque: internal storage (CSR vs. dense) is fully private to simple.c; the
 * backend choice is baked in at construction time (two separate build
 * functions) rather than threaded through every call, so callers/dispatchers
 * never need to know which backend is active. */
struct simple_synapse_data;

struct simple_synapse_data *synapse_simple_build_sparse(const double *dense, size_t n);
struct simple_synapse_data *synapse_simple_build_dense(const double *dense, size_t n);
void   synapse_simple_free(struct simple_synapse_data *d);
void   synapse_simple_to_dense(const struct simple_synapse_data *d, double *dense_out);
double synapse_simple_row_dot(const struct simple_synapse_data *d, size_t row, const double *x);
void   synapse_simple_scale(struct simple_synapse_data *d, double factor);
double synapse_simple_spectral_radius(const struct simple_synapse_data *d);

#endif // SYNAPSE_SIMPLE_H
