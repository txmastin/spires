#ifndef SPIRES_H
#define SPIRES_H

/* Public, single-header API for spires.
 *
 * Outside users should only include this file:
 *     #include <spires.h>
 *
 * Coding style:
 *  - snake_case identifiers
 *  - typedefs only for user-facing enums/opaque handles/config
 *  - no hidden allocations in hot paths
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------
 * Public status codes
 * ---------------------------- */
typedef enum {
    SPIRES_OK = 0,
    SPIRES_ERR_INVALID_ARG,
    SPIRES_ERR_ALLOC,
    SPIRES_ERR_INTERNAL
} spires_status;

/* Opaque handle (implementation is private) */
typedef struct spires_reservoir spires_reservoir;

/* ----------------------------
 * Public enums mirroring backend options
 * (ordering should match your backend enums)
 * ---------------------------- */
typedef enum {
    SPIRES_CONN_RANDOM = 0,
    SPIRES_CONN_SMALL_WORLD,
    SPIRES_CONN_SCALE_FREE
} spires_connectivity_type;

typedef enum {
    SPIRES_NEURON_LIF_DISCRETE = 0,
    SPIRES_NEURON_LIF_BIO,
    SPIRES_NEURON_FLIF_CAPUTO,
    SPIRES_NEURON_FLIF_GL,
    SPIRES_NEURON_FLIF_DIFFUSIVE
} spires_neuron_type;

/* ----------------------------
 * Creation-time configuration
 * (kept minimal; forwards to create_reservoir(...) as-is)
 * ---------------------------- */
typedef struct {
    size_t num_neurons;
    size_t num_inputs;
    size_t num_outputs;
    double spectral_radius;
    double ei_ratio;
    double input_strength;
    double connectivity;          /* density or similar per backend */
    double dt;
    spires_connectivity_type connectivity_type;
    spires_neuron_type       neuron_type;
    double *neuron_params;        /* forwarded to init_neuron; caller owns */
} spires_reservoir_config;

/* ----------------------------
 * Lifecycle
 * ---------------------------- */
spires_status spires_reservoir_create(const spires_reservoir_config *cfg,
                                      spires_reservoir **out_r);
void          spires_reservoir_destroy(spires_reservoir *r);
spires_status spires_reservoir_reset(spires_reservoir *r);

/* ----------------------------
 * Stepping
 * ---------------------------- */
/* Step once with an input vector u_t of length num_inputs.
 * Pass NULL for zeros. No output buffer here; use state or your own output accessor.
 */
spires_status spires_step(spires_reservoir *r, const double *u_t);



/* ----------------------------
 * Running (forward inferencing)
 * ---------------------------- */
/* Run the reservoir on a series of inputs.
 * input_series: flattened [series_length x num_inputs] (for Din=1 just length=series_length)
 * Returns a newly malloc'd array of length series_length with the readout outputs.
 * Caller must free().
 */

double *spires_run(spires_reservoir *r, const double *input_series, size_t series_length);


/* ----------------------------
 * Training
 * ---------------------------- */
spires_status spires_train_online(spires_reservoir *r,
                                  const double *target_vec, double lr);

spires_status spires_train_ridge(spires_reservoir *r,
                                 const double *input_series,
                                 const double *target_series,
                                 size_t series_length, double lambda);

/* ----------------------------
 * State access
 * ---------------------------- */
/* Returns a newly malloc'd copy of the current neuron state (length = num_neurons).
 * Caller must free().
 */
double *spires_read_state_copy(spires_reservoir *r);

/* Compute current readout y = W_out * state (+ b). 
 * Caller provides an array sized to num_outputs. */
spires_status spires_compute_output(spires_reservoir *r, double *out);


/* ----------------------------
 * Introspection
 * ---------------------------- */
size_t spires_num_neurons(const spires_reservoir *r);
size_t spires_num_inputs(const spires_reservoir *r);
size_t spires_num_outputs(const spires_reservoir *r);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SPIRES_H */
