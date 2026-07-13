#include "synapse.h"

struct synapse_matrix synapse_build_from_dense(const double *dense, size_t n,
                                                enum synapse_type type,
                                                enum synapse_backend backend,
                                                const double *synapse_params)
{
    struct synapse_matrix w = {0};
    w.type = type;
    w.backend = backend;
    w.n = n;

    switch (type) {
    case SYNAPSE_SIMPLE:
        w.data = (backend == SYNAPSE_SPARSE)
            ? (void *)synapse_simple_build_sparse(dense, n)
            : (void *)synapse_simple_build_dense(dense, n);
        break;
    case SYNAPSE_PSC_HOMOGENEOUS:
        w.data = (backend == SYNAPSE_SPARSE)
            ? (void *)synapse_psc_homogeneous_build_sparse(dense, n, synapse_params)
            : (void *)synapse_psc_homogeneous_build_dense(dense, n, synapse_params);
        break;
    case SYNAPSE_PSC_HETEROGENEOUS:
        w.data = (backend == SYNAPSE_SPARSE)
            ? (void *)synapse_psc_heterogeneous_build_sparse(dense, n, synapse_params)
            : (void *)synapse_psc_heterogeneous_build_dense(dense, n, synapse_params);
        break;
    }

    return w;
}

void synapse_free(struct synapse_matrix *w)
{
    if (!w || !w->data)
        return;
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        synapse_simple_free((struct simple_synapse_data *)w->data);
        break;
    case SYNAPSE_PSC_HOMOGENEOUS:
        synapse_psc_homogeneous_free((struct psc_homogeneous_synapse_data *)w->data);
        break;
    case SYNAPSE_PSC_HETEROGENEOUS:
        synapse_psc_heterogeneous_free((struct psc_heterogeneous_synapse_data *)w->data);
        break;
    }
    w->data = NULL;
    w->n = 0;
}

void synapse_to_dense(const struct synapse_matrix *w, double *dense_out)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        synapse_simple_to_dense((const struct simple_synapse_data *)w->data, dense_out);
        break;
    case SYNAPSE_PSC_HOMOGENEOUS:
        synapse_psc_homogeneous_to_dense((const struct psc_homogeneous_synapse_data *)w->data, dense_out);
        break;
    case SYNAPSE_PSC_HETEROGENEOUS:
        synapse_psc_heterogeneous_to_dense((const struct psc_heterogeneous_synapse_data *)w->data, dense_out);
        break;
    }
}

double synapse_row_dot(const struct synapse_matrix *w, size_t row, const double *x)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        return synapse_simple_row_dot((const struct simple_synapse_data *)w->data, row, x);
    case SYNAPSE_PSC_HOMOGENEOUS:
        return synapse_psc_homogeneous_row_dot((const struct psc_homogeneous_synapse_data *)w->data, row, x);
    case SYNAPSE_PSC_HETEROGENEOUS:
        return synapse_psc_heterogeneous_row_dot((const struct psc_heterogeneous_synapse_data *)w->data, row, x);
    }
    return 0.0;
}

void synapse_scale(struct synapse_matrix *w, double factor)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        synapse_simple_scale((struct simple_synapse_data *)w->data, factor);
        break;
    case SYNAPSE_PSC_HOMOGENEOUS:
        synapse_psc_homogeneous_scale((struct psc_homogeneous_synapse_data *)w->data, factor);
        break;
    case SYNAPSE_PSC_HETEROGENEOUS:
        synapse_psc_heterogeneous_scale((struct psc_heterogeneous_synapse_data *)w->data, factor);
        break;
    }
}

double synapse_spectral_radius(const struct synapse_matrix *w)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        return synapse_simple_spectral_radius((const struct simple_synapse_data *)w->data);
    case SYNAPSE_PSC_HOMOGENEOUS:
        return synapse_psc_homogeneous_spectral_radius((const struct psc_homogeneous_synapse_data *)w->data);
    case SYNAPSE_PSC_HETEROGENEOUS:
        return synapse_psc_heterogeneous_spectral_radius((const struct psc_heterogeneous_synapse_data *)w->data);
    }
    return 0.0;
}

const double *synapse_prepare(struct synapse_matrix *w, const double *spikes, double dt)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        return spikes;
    case SYNAPSE_PSC_HOMOGENEOUS:
        return synapse_psc_homogeneous_prepare((struct psc_homogeneous_synapse_data *)w->data, spikes, dt);
    case SYNAPSE_PSC_HETEROGENEOUS:
        return synapse_psc_heterogeneous_prepare((struct psc_heterogeneous_synapse_data *)w->data, spikes, dt);
    }
    return spikes;
}
