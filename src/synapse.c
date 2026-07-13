#include "synapse.h"
#include "math_utils.h"
#include <string.h>
#include <cblas.h>

struct synapse_matrix synapse_build_from_dense(const double *dense, size_t n,
                                                enum synapse_type type,
                                                enum synapse_backend backend)
{
    struct synapse_matrix w = {0};
    w.type = type;
    w.backend = backend;
    w.n = n;

    switch (type) {
    case SYNAPSE_SIMPLE:
        switch (backend) {
        case SYNAPSE_SPARSE:
            w.data.csr = csr_build_from_dense(dense, n);
            break;
        case SYNAPSE_DENSE:
            w.data.dense = malloc(n * n * sizeof(double));
            if (w.data.dense)
                memcpy(w.data.dense, dense, n * n * sizeof(double));
            break;
        }
        break;
    }

    return w;
}

void synapse_free(struct synapse_matrix *w)
{
    if (!w)
        return;
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        switch (w->backend) {
        case SYNAPSE_SPARSE:
            csr_free(&w->data.csr);
            break;
        case SYNAPSE_DENSE:
            free(w->data.dense);
            w->data.dense = NULL;
            break;
        }
        break;
    }
    w->n = 0;
}

void synapse_to_dense(const struct synapse_matrix *w, double *dense_out)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        switch (w->backend) {
        case SYNAPSE_SPARSE:
            csr_to_dense(&w->data.csr, dense_out);
            break;
        case SYNAPSE_DENSE:
            memcpy(dense_out, w->data.dense, w->n * w->n * sizeof(double));
            break;
        }
        break;
    }
}

double synapse_row_dot(const struct synapse_matrix *w, size_t row, const double *x)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        switch (w->backend) {
        case SYNAPSE_SPARSE:
            return csr_row_dot(&w->data.csr, row, x);
        case SYNAPSE_DENSE:
            return cblas_ddot((int)w->n, &w->data.dense[row * w->n], 1, x, 1);
        }
        break;
    }
    return 0.0;
}

void synapse_scale(struct synapse_matrix *w, double factor)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        switch (w->backend) {
        case SYNAPSE_SPARSE:
            csr_scale(&w->data.csr, factor);
            break;
        case SYNAPSE_DENSE:
            cblas_dscal((int)(w->n * w->n), factor, w->data.dense, 1);
            break;
        }
        break;
    }
}

double synapse_spectral_radius(const struct synapse_matrix *w)
{
    switch (w->type) {
    case SYNAPSE_SIMPLE:
        switch (w->backend) {
        case SYNAPSE_SPARSE:
            return csr_spectral_radius(&w->data.csr, w->n);
        case SYNAPSE_DENSE:
            return calc_spectral_radius(w->data.dense, w->n);
        }
        break;
    }
    return 0.0;
}
