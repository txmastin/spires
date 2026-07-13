#ifndef SYNAPSE_H
#define SYNAPSE_H

#include "synapses/simple.h"
#include "synapses/psc_homogeneous.h"
#include "synapses/psc_heterogeneous.h"

/* Which synapse model. Future hot-swappable models (distributional,
 * biorealistic, memcapacitive, ...) are a separate, later effort. */
enum synapse_type {
    SYNAPSE_SIMPLE = 0,
    SYNAPSE_PSC_HOMOGENEOUS,
    SYNAPSE_PSC_HETEROGENEOUS
};

/* Storage/compute backend. Explicit and user-selectable (see
 * spires_synapse_backend in spires.h) rather than auto-switched, so behavior
 * stays predictable and reproducible.
 *
 * SYNAPSE_SPARSE (CSR) wins at low connectivity (typical reservoir-computing
 * topologies, ~1-20% density). SYNAPSE_DENSE (plain array + BLAS) can win at
 * higher connectivity, since cblas_ddot/cblas_dgemv vectorize over contiguous
 * memory better than CSR's gather-via-col_idx access pattern. */
enum synapse_backend {
    SYNAPSE_SPARSE = 0,
    SYNAPSE_DENSE
};

/* Opaque: internal storage is fully private to the corresponding
 * implementation file under src/synapses/ (mirrors neuron.c's `void *neuron`
 * dispatch). */
struct synapse_matrix {
    enum synapse_type type;
    enum synapse_backend backend;
    size_t n;
    void *data;
};

/* synapse_params layout is documented per type in the corresponding header
 * under src/synapses/; SYNAPSE_SIMPLE ignores it. */
struct synapse_matrix synapse_build_from_dense(const double *dense, size_t n,
                                                enum synapse_type type,
                                                enum synapse_backend backend,
                                                const double *synapse_params);
void   synapse_free(struct synapse_matrix *w);
void   synapse_to_dense(const struct synapse_matrix *w, double *dense_out);
double synapse_row_dot(const struct synapse_matrix *w, size_t row, const double *x);
void   synapse_scale(struct synapse_matrix *w, double factor);
double synapse_spectral_radius(const struct synapse_matrix *w);

/* Called once per micro-step, before the per-neuron row_dot pass (it mutates
 * shared per-synapse-model state, so must run single-threaded/first).
 * Returns the vector row_dot should be called with -- for SYNAPSE_SIMPLE this
 * is `spikes` unchanged; for stateful models it's an internally-maintained
 * filtered trace (or unused/NULL, for models like PSC_HETEROGENEOUS that
 * read their own per-edge trace directly inside row_dot instead). */
const double *synapse_prepare(struct synapse_matrix *w, const double *spikes, double dt);

#endif // SYNAPSE_H
