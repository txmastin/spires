#ifndef SYNAPSE_H
#define SYNAPSE_H

#include "sparse.h"

/* Which synapse model. Only one exists today (SYNAPSE_SIMPLE = plain weights).
 * Future hot-swappable models (distributional, biorealistic, memcapacitive, ...)
 * are a separate, later effort — no placeholder values yet. */
enum synapse_type {
    SYNAPSE_SIMPLE = 0
};

/* Storage/compute backend, only meaningful within SYNAPSE_SIMPLE today.
 * Explicit and user-selectable (see spires_synapse_backend in spires.h) rather
 * than auto-switched, so behavior stays predictable and reproducible.
 *
 * SYNAPSE_SPARSE (CSR) wins at low connectivity (typical reservoir-computing
 * topologies, ~1-20% density). SYNAPSE_DENSE (plain array + BLAS) can win at
 * higher connectivity, since cblas_ddot/cblas_dgemv vectorize over contiguous
 * memory better than CSR's gather-via-col_idx access pattern. */
enum synapse_backend {
    SYNAPSE_SPARSE = 0,
    SYNAPSE_DENSE
};

struct synapse_matrix {
    enum synapse_type type;
    enum synapse_backend backend;
    size_t n;
    union {
        struct csr_matrix csr;
        double *dense;   // n*n, row-major, malloc'd
    } data;
};

struct synapse_matrix synapse_build_from_dense(const double *dense, size_t n,
                                                enum synapse_type type,
                                                enum synapse_backend backend);
void   synapse_free(struct synapse_matrix *w);
void   synapse_to_dense(const struct synapse_matrix *w, double *dense_out);
double synapse_row_dot(const struct synapse_matrix *w, size_t row, const double *x);
void   synapse_scale(struct synapse_matrix *w, double factor);
double synapse_spectral_radius(const struct synapse_matrix *w);

#endif // SYNAPSE_H
