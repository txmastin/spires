#include "simple.h"
#include "../sparse.h"
#include "../math_utils.h"
#include <string.h>
#include <cblas.h>

struct simple_synapse_data {
    int is_sparse;
    size_t n;
    union {
        struct csr_matrix csr;
        double *dense;
    } storage;
};

struct simple_synapse_data *synapse_simple_build_sparse(const double *dense, size_t n)
{
    struct simple_synapse_data *d = malloc(sizeof(*d));
    if (!d)
        return NULL;
    d->is_sparse = 1;
    d->n = n;
    d->storage.csr = csr_build_from_dense(dense, n);
    return d;
}

struct simple_synapse_data *synapse_simple_build_dense(const double *dense, size_t n)
{
    struct simple_synapse_data *d = malloc(sizeof(*d));
    if (!d)
        return NULL;
    d->is_sparse = 0;
    d->n = n;
    d->storage.dense = malloc(n * n * sizeof(double));
    if (d->storage.dense)
        memcpy(d->storage.dense, dense, n * n * sizeof(double));
    return d;
}

void synapse_simple_free(struct simple_synapse_data *d)
{
    if (!d)
        return;
    if (d->is_sparse)
        csr_free(&d->storage.csr);
    else
        free(d->storage.dense);
    free(d);
}

void synapse_simple_to_dense(const struct simple_synapse_data *d, double *dense_out)
{
    if (d->is_sparse)
        csr_to_dense(&d->storage.csr, dense_out);
    else
        memcpy(dense_out, d->storage.dense, d->n * d->n * sizeof(double));
}

double synapse_simple_row_dot(const struct simple_synapse_data *d, size_t row, const double *x)
{
    if (d->is_sparse)
        return csr_row_dot(&d->storage.csr, row, x);
    return cblas_ddot((int)d->n, &d->storage.dense[row * d->n], 1, x, 1);
}

void synapse_simple_scale(struct simple_synapse_data *d, double factor)
{
    if (d->is_sparse)
        csr_scale(&d->storage.csr, factor);
    else
        cblas_dscal((int)(d->n * d->n), factor, d->storage.dense, 1);
}

double synapse_simple_spectral_radius(const struct simple_synapse_data *d)
{
    if (d->is_sparse)
        return csr_spectral_radius(&d->storage.csr, d->n);
    return calc_spectral_radius(d->storage.dense, d->n);
}
