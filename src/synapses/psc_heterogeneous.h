#ifndef SYNAPSE_PSC_HETEROGENEOUS_H
#define SYNAPSE_PSC_HETEROGENEOUS_H

#include <stdlib.h>

/* PSC_HETEROGENEOUS: exponential postsynaptic-current filter with an
 * independent tau sampled per connection. params layout:
 * params[0] = tau_min, params[1] = tau_max (tau ~ log-uniform on
 * [tau_min, tau_max] -- i.e. log(tau) is uniform, so tau spans the range
 * evenly on a log scale rather than over-representing the high end -- sampled
 * while building the structure).
 *
 * Unlike PSC_HOMOGENEOUS, per-edge tau means the filter can't be collapsed
 * to per-neuron state -- this genuinely needs its own per-edge trace and
 * tau storage (own struct, not reusing simple.c). */
struct psc_heterogeneous_synapse_data;

struct psc_heterogeneous_synapse_data *synapse_psc_heterogeneous_build_sparse(const double *dense, size_t n, const double *params);
struct psc_heterogeneous_synapse_data *synapse_psc_heterogeneous_build_dense(const double *dense, size_t n, const double *params);
void   synapse_psc_heterogeneous_free(struct psc_heterogeneous_synapse_data *d);
void   synapse_psc_heterogeneous_to_dense(const struct psc_heterogeneous_synapse_data *d, double *dense_out);
double synapse_psc_heterogeneous_row_dot(const struct psc_heterogeneous_synapse_data *d, size_t row, const double *x);
void   synapse_psc_heterogeneous_scale(struct psc_heterogeneous_synapse_data *d, double factor);
double synapse_psc_heterogeneous_spectral_radius(const struct psc_heterogeneous_synapse_data *d);
const double *synapse_psc_heterogeneous_prepare(struct psc_heterogeneous_synapse_data *d, const double *spikes, double dt);

#endif // SYNAPSE_PSC_HETEROGENEOUS_H
