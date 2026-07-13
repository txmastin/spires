#include "psc_homogeneous.h"
#include "simple.h"
#include <math.h>

struct psc_homogeneous_synapse_data {
    struct simple_synapse_data *weights;
    size_t n;
    double tau_syn;
    double *trace;   // length n, one per presynaptic neuron
};

static struct psc_homogeneous_synapse_data *build_common(struct simple_synapse_data *weights,
                                                          size_t n, const double *params)
{
    if (!weights)
        return NULL;

    struct psc_homogeneous_synapse_data *d = malloc(sizeof(*d));
    if (!d) {
        synapse_simple_free(weights);
        return NULL;
    }
    d->weights = weights;
    d->n = n;
    d->tau_syn = params ? params[0] : 1.0;
    d->trace = calloc(n, sizeof(double));
    return d;
}

struct psc_homogeneous_synapse_data *synapse_psc_homogeneous_build_sparse(const double *dense, size_t n,
                                                                           const double *params)
{
    return build_common(synapse_simple_build_sparse(dense, n), n, params);
}

struct psc_homogeneous_synapse_data *synapse_psc_homogeneous_build_dense(const double *dense, size_t n,
                                                                          const double *params)
{
    return build_common(synapse_simple_build_dense(dense, n), n, params);
}

void synapse_psc_homogeneous_free(struct psc_homogeneous_synapse_data *d)
{
    if (!d)
        return;
    synapse_simple_free(d->weights);
    free(d->trace);
    free(d);
}

void synapse_psc_homogeneous_to_dense(const struct psc_homogeneous_synapse_data *d, double *dense_out)
{
    synapse_simple_to_dense(d->weights, dense_out);
}

double synapse_psc_homogeneous_row_dot(const struct psc_homogeneous_synapse_data *d, size_t row, const double *x)
{
    return synapse_simple_row_dot(d->weights, row, x);
}

void synapse_psc_homogeneous_scale(struct psc_homogeneous_synapse_data *d, double factor)
{
    synapse_simple_scale(d->weights, factor);
}

double synapse_psc_homogeneous_spectral_radius(const struct psc_homogeneous_synapse_data *d)
{
    return synapse_simple_spectral_radius(d->weights);
}

const double *synapse_psc_homogeneous_prepare(struct psc_homogeneous_synapse_data *d, const double *spikes, double dt)
{
    /* Normalized so the filter's DC gain is exactly 1 (steady-state trace for
     * constant firing -> 1, matching a raw spike's magnitude) regardless of
     * tau_syn -- an unnormalized trace[i] = trace[i]*decay + spikes[i] has
     * steady-state gain 1/(1-decay) ~= tau_syn/dt, which silently amplifies
     * the effective recurrent input far beyond what rescale_weights's
     * spectral-radius calibration (calibrated for O(1) spike inputs) accounts
     * for, and drives the reservoir into runaway saturation. */
    double decay = exp(-dt / d->tau_syn);
    for (size_t i = 0; i < d->n; i++)
        d->trace[i] = d->trace[i] * decay + (1.0 - decay) * spikes[i];
    return d->trace;
}
