#ifndef SYNAPSE_PSC_HOMOGENEOUS_H
#define SYNAPSE_PSC_HOMOGENEOUS_H

#include <stdlib.h>

/* PSC_HOMOGENEOUS: exponential postsynaptic-current filter with one tau_syn
 * shared by the whole reservoir. params layout: params[0] = tau_syn.
 *
 * Exponential filtering is linear, so with a uniform tau_syn, filtering once
 * per presynaptic neuron (a length-n trace vector) is mathematically
 * equivalent to filtering per edge -- weight storage/row-dot math is
 * therefore identical to SYNAPSE_SIMPLE and reused internally; only the
 * trace buffer and tau_syn are new. */
struct psc_homogeneous_synapse_data;

struct psc_homogeneous_synapse_data *synapse_psc_homogeneous_build_sparse(const double *dense, size_t n, const double *params);
struct psc_homogeneous_synapse_data *synapse_psc_homogeneous_build_dense(const double *dense, size_t n, const double *params);
void   synapse_psc_homogeneous_free(struct psc_homogeneous_synapse_data *d);
void   synapse_psc_homogeneous_to_dense(const struct psc_homogeneous_synapse_data *d, double *dense_out);
double synapse_psc_homogeneous_row_dot(const struct psc_homogeneous_synapse_data *d, size_t row, const double *x);
void   synapse_psc_homogeneous_scale(struct psc_homogeneous_synapse_data *d, double factor);
double synapse_psc_homogeneous_spectral_radius(const struct psc_homogeneous_synapse_data *d);
const double *synapse_psc_homogeneous_prepare(struct psc_homogeneous_synapse_data *d, const double *spikes, double dt);

#endif // SYNAPSE_PSC_HOMOGENEOUS_H
