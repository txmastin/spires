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

/* Which synapse model governs the recurrent connections. Distinct from
 * spires_connectivity_type, which decides the topology (who's connected to
 * whom) rather than how a connection's weight is modeled. Future hot-swappable
 * models (distributional, biorealistic, memcapacitive, ...) are a separate,
 * later effort.
 *
 * synapse_params layout (see spires_reservoir_config.synapse_params below):
 *   SPIRES_SYNAPSE_SIMPLE:              unused.
 *   SPIRES_SYNAPSE_PSC_HOMOGENEOUS:      params[0] = tau_syn (shared by all
 *                                        connections; exponential PSC filter).
 *   SPIRES_SYNAPSE_PSC_HETEROGENEOUS:    params[0] = tau_min, params[1] =
 *                                        tau_max (per-connection tau sampled
 *                                        log-uniform on [tau_min, tau_max]).
 *   SPIRES_SYNAPSE_FRACTIONAL:           params[0] = alpha (0,1], params[1] =
 *                                        tau_syn, params[2] = T_mem (finite
 *                                        memory-window truncation). Exact
 *                                        GL-discretized fractional relaxation
 *                                        equation for synaptic current,
 *                                        tau_syn*D^alpha(I) = -I + spike(t) --
 *                                        impulse response is the Mittag-Leffler
 *                                        relaxation function, which reduces to
 *                                        SPIRES_SYNAPSE_PSC_HOMOGENEOUS at
 *                                        alpha=1. O(mem_len) per neuron per step.
 *   SPIRES_SYNAPSE_FRACTIONAL_MULTIEXP:  params[0] = alpha (target order),
 *                                        params[1] = tau_syn, params[2] =
 *                                        tau_min, params[3] = tau_max,
 *                                        params[4] = N (number of exponential
 *                                        components). Cheaper O(N) per neuron
 *                                        per step approximation of
 *                                        SPIRES_SYNAPSE_FRACTIONAL's
 *                                        Mittag-Leffler response, via the exact
 *                                        Mittag-Leffler spectral density
 *                                        discretized over N log-spaced
 *                                        timescales in [tau_min, tau_max] --
 *                                        expect it to track the exact model
 *                                        well inside that window and diverge
 *                                        outside it. */
typedef enum {
    SPIRES_SYNAPSE_SIMPLE = 0,
    SPIRES_SYNAPSE_PSC_HOMOGENEOUS,
    SPIRES_SYNAPSE_PSC_HETEROGENEOUS,
    SPIRES_SYNAPSE_FRACTIONAL,
    SPIRES_SYNAPSE_FRACTIONAL_MULTIEXP
} spires_synapse_type;

/* Storage/compute backend, applies to all synapse types above. Explicit,
 * user-selectable choice (not auto-switched): SPARSE (CSR) wins at low
 * connectivity (typical reservoir-computing topologies); DENSE (BLAS) can
 * win at higher connectivity. */
typedef enum {
    SPIRES_SYNAPSE_SPARSE = 0,
    SPIRES_SYNAPSE_DENSE
} spires_synapse_backend;

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
    spires_synapse_type      synapse_type;    /* default 0 = SPIRES_SYNAPSE_SIMPLE */
    spires_synapse_backend   synapse_backend; /* default 0 = SPIRES_SYNAPSE_SPARSE */
    double *synapse_params;       /* forwarded as-is; layout documented above per synapse_type; caller owns */
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

spires_status spires_train_rls(spires_reservoir *r,
                               const double *input_series,
                               const double *target_series,
                               size_t series_length,
                               double delta, double lambda);

spires_status spires_coarse_grain(const spires_reservoir *r,
                                  double weight_threshold,
                                  spires_reservoir **out_r);

/* ----------------------------
 * State access
 * ---------------------------- */
/* Returns a newly malloc'd copy of the current neuron state (length = num_neurons).
 * Caller must free().
 */
double *spires_copy_reservoir_state(spires_reservoir *r);

/* Reads the current neuron state into a caller-provided buffer
 * (length = num_neurons). No allocation.
 */
spires_status spires_read_reservoir_state(spires_reservoir *r, double *buffer);

/* Reads the current spike state (0.0 or 1.0 per neuron) into a caller-provided
 * buffer (length = num_neurons). Call immediately after spires_step to get
 * spikes from that timestep. No allocation.
 */
spires_status spires_read_spike_state(spires_reservoir *r, double *buffer);

/* Returns a newly malloc'd copy of the recurrent weight matrix W
 * (length = num_neurons * num_neurons, row-major). Caller must free().
 */
double *spires_copy_weights(const spires_reservoir *r);

/* Reads the recurrent weight matrix W into a caller-provided buffer
 * (length = num_neurons * num_neurons, row-major). No allocation.
 */
spires_status spires_read_weights(const spires_reservoir *r, double *buffer);

/* Compute current readout y = W_out * state (+ b). 
 * Caller provides an array sized to num_outputs. */
spires_status spires_compute_output(spires_reservoir *r, double *out);


/* ----------------------------
 * Introspection
 * ---------------------------- */
size_t spires_num_neurons(const spires_reservoir *r);
size_t spires_num_inputs(const spires_reservoir *r);
size_t spires_num_outputs(const spires_reservoir *r);


/* --------------------------
 *  Optimizer stuff
 *  ------------------------- */

struct spires_opt_budget {
	double	data_fraction;    /* 0..1 */
	int	num_seeds;
	double	time_limit_sec;   /* 0 = no limit */
	int	_reserved_i0;
	double	_reserved_d0;
};

enum spires_metric_kind { SPIRES_METRIC_AUROC = 0, SPIRES_METRIC_AUPRC = 1 };

struct spires_opt_score {
	double	lambda_var;       /* penalty on std(metric) */
	double	lambda_cost;      /* penalty on cost proxy */
	int	metric;           /* enum spires_metric_kind */
	int	_reserved_i0;
};

struct spires_opt_result {
	spires_reservoir_config  best_config;
        double                   best_log10_ridge;
	double	                 best_score;   /* mean − λ_var*std − λ_cost*cost */
	double	                 metric_mean;
	double	                 metric_std;
};

int spires_optimize(const spires_reservoir_config *base_config,
                    const struct spires_opt_budget *budgets,
		    int num_budgets,
		    const struct spires_opt_score *score,
		    struct spires_opt_result *out,
                    const double *input_series,
		    const double *target_series,
		    size_t series_length);


#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SPIRES_H */
