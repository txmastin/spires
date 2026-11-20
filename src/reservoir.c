#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cblas.h>
#include <omp.h>
#include <string.h> // for memcpy
#include <math.h>

// #include "spires.h"
#include "reservoir.h"
#include "math_utils.h"
#include "synapse.h"



/* Neurons that draw random numbers get a seed derived from the reservoir
 * seed and their index, not a draw from reservoir->rng, so the weight
 * stream is the same whatever the neuron type. */
static void seed_neurons(struct reservoir *r)
{
    if (r->neuron_type != LIF_POWERLAW_REFRACTORY) return;
    for (size_t i = 0; i < r->num_neurons; i++) {
        uint64_t s = r->seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1);
        lif_powerlaw_refractory_set_seed(r->neurons[i], s);
    }
}

/* **** create reservoir ***** */

struct reservoir *create_reservoir(const struct reservoir_params *p)
{
    if (p == NULL) {
        fprintf(stderr, "Error creating reservoir. Parameters are NULL!\n");
        return NULL;
    }

    struct reservoir *reservoir = malloc(sizeof(*reservoir));
    if (reservoir == NULL) {
        fprintf(stderr, "Error allocating memory for reservoir of size %zu\n", p->num_neurons);
        return NULL;
    }

    reservoir->num_neurons = p->num_neurons;
    reservoir->num_inputs = p->num_inputs;
    reservoir->num_outputs = p->num_outputs;
    reservoir->spectral_radius = p->spectral_radius;
    reservoir->ei_ratio = p->ei_ratio;
    reservoir->input_strength = p->input_strength;
    reservoir->connectivity = p->connectivity;
    reservoir->dt = p->dt;
    reservoir->connectivity_type = p->connectivity_type;
    reservoir->neuron_type = p->neuron_type;
    reservoir->neuron_params = p->neuron_params;
    reservoir->synapse_type = p->synapse_type;
    reservoir->synapse_backend = p->synapse_backend;
    reservoir->synapse_params = p->synapse_params;
    reservoir->ei_mode = p->ei_mode;
    reservoir->neuron_sign = NULL;

    reservoir->plasticity.type = p->plasticity_type;
    reservoir->plasticity.params = p->plasticity_params;
    reservoir->plasticity.x_pre_e = NULL;
    reservoir->plasticity.x_post_e = NULL;
    reservoir->plasticity.x_pre_i = NULL;
    reservoir->plasticity.x_post_i = NULL;
    reservoir->plasticity.neuron_sign = NULL;
    if (p->plasticity_type != PLASTICITY_NONE) {
        reservoir->plasticity.x_pre_e  = calloc(p->num_neurons, sizeof(double));
        reservoir->plasticity.x_post_e = calloc(p->num_neurons, sizeof(double));
        reservoir->plasticity.x_pre_i  = calloc(p->num_neurons, sizeof(double));
        reservoir->plasticity.x_post_i = calloc(p->num_neurons, sizeof(double));
        if (!reservoir->plasticity.x_pre_e || !reservoir->plasticity.x_post_e ||
            !reservoir->plasticity.x_pre_i || !reservoir->plasticity.x_post_i) {
            fprintf(stderr, "Failed to allocate plasticity traces\n");
            free(reservoir->plasticity.x_pre_e);
            free(reservoir->plasticity.x_post_e);
            free(reservoir->plasticity.x_pre_i);
            free(reservoir->plasticity.x_post_i);
            free(reservoir);
            return NULL;
        }
    }
    reservoir->seed = p->seed;
    reservoir->use_washout = p->use_washout;
    reservoir->washout = p->washout;
    reservoir->steps_since_reset = 0;
    spires_rng_seed(&reservoir->rng, p->seed);


    reservoir->neurons = malloc(p->num_neurons * sizeof(void*));

    if (!(reservoir->neurons)) {
        fprintf(stderr, "Memory allocation failed for reservoir neurons\n");
        free(reservoir);
        return NULL;
    }

    void *local_shared_neuron_data = NULL; // set to NULL to start, optionally change it during neuron instantiation

    #pragma omp parallel for // parallelizable because race conditions are handled by init_neuron
    for (size_t i = 0; i < p->num_neurons; i++) {
        reservoir->neurons[i] = init_neuron(p->neuron_type, p->neuron_params, p->dt, &local_shared_neuron_data);
    }
    
    reservoir->shared_neuron_data = local_shared_neuron_data; // malloc'd in function if flif_gl is used, must free
    seed_neurons(reservoir);

    return reservoir;
}

int init_reservoir(struct reservoir *r) 
{
    if (r == NULL) {
        fprintf(stderr, "Error initializing reservoir. Reservoir not created!\n");
        return EXIT_FAILURE;
    }
    init_weights(r);
    rescale_weights(r);
    randomize_output_layer(r);
    return EXIT_SUCCESS;
}

/* Writes num_outputs * input_length doubles into output_series, supplied by
 * the caller. run_reservoir() is this plus the allocation. */
int run_reservoir_into(struct reservoir *r, double *input_series,
                       size_t input_length, double *output_series)
{
    if (!r) {
        fprintf(stderr, "Error running reservoir. Reservoir not initialized!\n");
        return EXIT_FAILURE;
    }

    if (r->dt <= 0.0) {
        fprintf(stderr, "Error running reservoir. dt must be greater than 0.\n");
        return EXIT_FAILURE;
    }

    if (!input_series || !output_series)
        return EXIT_FAILURE;

    size_t num_inputs = r->num_inputs;
    size_t num_outputs = r->num_outputs;

    for (size_t i = 0; i < input_length; i ++) {
        const double *current_input = &input_series[i * num_inputs];
        step_reservoir(r, current_input);

        double *current_output = &output_series[i * num_outputs];
        compute_output(r, current_output);
    }

    return EXIT_SUCCESS;
}

double *run_reservoir(struct reservoir *r, double *input_series, size_t input_length) 
{
    if (!r) {
        fprintf(stderr, "Error running reservoir. Reservoir not initialized!\n");
        return NULL;
    }

    double *output_series = malloc(r->num_outputs * input_length * sizeof(double));
    if (output_series == NULL) {
        fprintf(stderr, "Error intializing memory for reservoir outputs!\n");
        return NULL;
    }

    if (run_reservoir_into(r, input_series, input_length, output_series) != EXIT_SUCCESS) {
        free(output_series);
        return NULL;
    }

    return output_series;    
}

void step_reservoir(struct reservoir *r, const double *input_vector) 
{
    if (!r) {
        fprintf(stderr, "Error running reservoir. Reservoir not initialized!\n");
        return;
    }

    if (r->dt <= 0.0) {
        fprintf(stderr, "Error running reservoir. dt must be greater than 0.\n");
        return;
    }

    // The macro step duration is implicitly 1.0.
    // Calculate how many internal micro-steps to run.
    int num_micro_steps = (int)llround(1.0 / r->dt);
    size_t num_neurons = r->num_neurons;
    size_t num_inputs = r->num_inputs;

    // --- Pre-calculate constant external inputs for this entire macro step ---
    double external_inputs[num_neurons];
    // W_in * input_vector
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                num_neurons, num_inputs, r->input_strength, r->W_in, num_inputs,
                input_vector, 1, 0.0, external_inputs, 1);
    

    // --- Double buffer to hold spikes, which evolve at the micro-step scale ---
    // doubled for parallel computation / preventing race conditions
    double last_spikes[num_neurons];
    double new_spikes[num_neurons]; 
    
    for (size_t i = 0; i < num_neurons; i++) {
        last_spikes[i] = get_neuron_spike(r->neurons[i], r->neuron_type);
    }

    // --- Internal simulation loop (the "micro-steps") ---
    for (int t = 0; t < num_micro_steps; t++) {
        // Advance any per-synapse-model state (e.g. PSC decay traces) once per
        // micro-step, before the parallel per-neuron pass reads it.
        const double *effective_input = synapse_prepare(&r->W, last_spikes, r->dt);

        #pragma omp parallel for // omp ftw
        for (size_t i = 0; i < num_neurons; i++) {
            double recurrent_input = synapse_row_dot(&r->W, i, effective_input);

            double total_input = external_inputs[i] + recurrent_input;
            update_neuron(r->neurons[i], r->neuron_type, total_input, r->dt);
            new_spikes[i] = get_neuron_spike(r->neurons[i], r->neuron_type);
        }

        /* Decay, then update weights against strictly-earlier activity, then
         * fold in this step's spikes. One spike vector serves both roles:
         * a neuron's spike is presynaptic for its outgoing synapses and
         * postsynaptic for its incoming ones. */
        plasticity_decay(&r->plasticity, num_neurons, r->dt);
        synapse_apply_plasticity(&r->W, new_spikes, &r->plasticity, r->dt);
        plasticity_accumulate(&r->plasticity, new_spikes, num_neurons);

        memcpy(last_spikes, new_spikes, num_neurons * sizeof(double));
    }
    r->steps_since_reset++;
}

/* Whether the state after the current step falls inside the washout. Steps
 * are counted from 1, so with washout W the first W states are excluded. */
static int in_washout(const struct reservoir *r)
{
    return r->use_washout && r->steps_since_reset <= r->washout;
}


// compute_output calculates the sum of neuron states * W_out
int compute_output(struct reservoir *reservoir, double *output_vector) 
{
    if (!reservoir || !output_vector) {
        fprintf(stderr, "Error: Reservoir or output vector is NULL.\n");
        return EXIT_FAILURE;
    }

    size_t num_neurons = reservoir->num_neurons;
    size_t num_outputs = reservoir->num_outputs;

    // read the current state of all neurons into a temporary local array.
    // using a VLA on the stack
    double state[num_neurons];
    for (size_t i = 0; i < num_neurons; i++) {
        state[i] = get_neuron_state(reservoir->neurons[i], reservoir->neuron_type);
    }

    // compute the output vector: output_vector = W_out * state
    // W_out is a [num_outputs x num_neurons] matrix
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                num_outputs, num_neurons, 1.0, reservoir->W_out, num_neurons,
                state, 1, 0.0, output_vector, 1);
    return EXIT_SUCCESS;
}

// compute_activity returns the sum of spikes
double compute_activity(struct reservoir *reservoir) 
{
    double total_activity = 0.0;
    double spike; 
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        spike = get_neuron_spike(reservoir->neurons[i], reservoir->neuron_type);
        total_activity += spike;
    }
    
    return total_activity;
}

// reads the neuron->V for each neuron into buffer
void read_reservoir_state(struct reservoir *reservoir, double *buffer) 
{
    if (buffer == NULL)
        fprintf(stderr, "Error reading reservoir state, buffer uninitialized\n");

    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        buffer[i] = get_neuron_state(reservoir->neurons[i], reservoir->neuron_type);
    }
}


// reads the spike state (0.0 or 1.0) for each neuron into buffer
void read_reservoir_spikes(struct reservoir *reservoir, double *buffer)
{
    if (buffer == NULL) {
        fprintf(stderr, "Error reading spike state, buffer uninitialized\n");
        return;
    }
    for (size_t i = 0; i < reservoir->num_neurons; i++)
        buffer[i] = get_neuron_spike(reservoir->neurons[i], reservoir->neuron_type);
}

// copies the neuron->V for each neuron into state
double *copy_reservoir_state(struct reservoir *reservoir)
{
    double *state = malloc(reservoir->num_neurons * sizeof(double));
    if (state == NULL) {
        fprintf(stderr, "Memory allocation error for reservoir state\n");
        return NULL;
    }
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        state[i] = get_neuron_state(reservoir->neurons[i], reservoir->neuron_type);
    }
    return state; // caller will need to free()
}

void train_output_iteratively(struct reservoir *r, double *target_vector, double lr)
{
    if (!r || !target_vector) {
        fprintf(stderr, "Error: Reservoir or target vector is NULL.\n");
        return;
    }
    if (in_washout(r))
        return;

    size_t num_neurons = r->num_neurons;
    size_t num_outputs = r->num_outputs;

    // 1. Get the current state of the reservoir's neurons
    double *state = copy_reservoir_state(r);
    if (!state) {
        fprintf(stderr, "Error: Failed to read reservoir state.\n");
        return;
    }

    // 2. Get the current prediction vector from the reservoir
    double prediction[num_outputs];
    compute_output(r, prediction);

    // 3. Loop through each output channel to update its corresponding weights
    for (size_t i = 0; i < num_outputs; i++) {
        // Calculate the error for this specific output channel
        double error = target_vector[i] - prediction[i];

        // Get a pointer to the start of the i-th row of the W_out matrix
        double *w_out_row = &r->W_out[i * num_neurons];

        // Apply the learning rule to this row of the output weights:
        // W_out_row += learning_rate * error * state_vector
        cblas_daxpy(num_neurons, lr * error, state, 1, w_out_row, 1);
    }

    // Clean up the allocated state vector
    free(state);
}

void free_reservoir(struct reservoir *reservoir) 
{
    if (!reservoir) { 
        return;
    }

    free(reservoir->neuron_sign);
    reservoir->neuron_sign = NULL;
    free(reservoir->plasticity.x_pre_e);
    free(reservoir->plasticity.x_post_e);
    free(reservoir->plasticity.x_pre_i);
    free(reservoir->plasticity.x_post_i);
    reservoir->plasticity.x_pre_e = NULL;
    reservoir->plasticity.x_post_e = NULL;
    reservoir->plasticity.x_pre_i = NULL;
    reservoir->plasticity.x_post_i = NULL;

    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        free_neuron(reservoir->neurons[i], reservoir->neuron_type);
        reservoir->neurons[i] = NULL;
    }

    if (reservoir->neuron_type == FLIF_GL && reservoir->shared_neuron_data != NULL) {
		free(reservoir->shared_neuron_data);
	}

    free(reservoir->neurons);
    free(reservoir->W_in);
    free(reservoir->W_out);
    synapse_free(&reservoir->W);
    reservoir->W_in = NULL;
    reservoir->W_out = NULL;
    free(reservoir);
    reservoir = NULL;
}


double generate_weight(struct spires_rng *rng, double ei_ratio)
{
    double magnitude = spires_rng_double(rng);
    if (spires_rng_double(rng) < ei_ratio) {
        return magnitude;
    }
    else {
        return -magnitude;
    }
}

/**
 * @brief Initializes the weight matrices for a reservoir.
 * * @param reservoir A pointer to the reservoir to initialize.
 * @return EXIT_SUCCESS or EXIT_FAILURE.
 * * @note Randomness comes from the reservoir's own RNG, seeded from
 * reservoir_params.seed at construction. Construction is therefore
 * reproducible and safe to run on several threads at once; the global
 * srand()/rand() pair is no longer involved.
 */

/* --- helpers for weight init ----- */

static inline double urand01(struct spires_rng *rng)
{
    return spires_rng_double(rng);
}

static inline int has_edge(const struct edge_rows *er, size_t n, size_t i, size_t j)
{
    (void)n;
    return edge_rows_has(er, i, j);
}

/* neuron_sign is NULL under EI_PER_SYNAPSE, leaving the per-connection sign
 * from generate_weight() untouched. Under Dale's law it holds one sign per
 * neuron, and the presynaptic neuron j fixes the sign of the edge. */
static inline void add_edge(struct edge_rows *er, size_t n, struct spires_rng *rng,
                            double ei_ratio, const signed char *neuron_sign,
                            size_t i, size_t j)
{
    (void)n;
    double w = generate_weight(rng, ei_ratio);
    if (neuron_sign)
        w = (neuron_sign[j] > 0) ? fabs(w) : -fabs(w);
    edge_rows_append(er, i, j, w);
}

/* Rewiring can pick a column whose entry was zeroed earlier in the same pass.
 * has_edge() reports that as absent, so appending would leave two entries for
 * one column; overwrite instead. */
static inline void add_edge_over(struct edge_rows *er, size_t n, struct spires_rng *rng,
                                 double ei_ratio, const signed char *neuron_sign,
                                 size_t i, size_t j)
{
    (void)n;
    double w = generate_weight(rng, ei_ratio);
    if (neuron_sign)
        w = (neuron_sign[j] > 0) ? fabs(w) : -fabs(w);
    edge_rows_set(er, i, j, w);
}

int init_weights(struct reservoir *reservoir)
{
    reservoir->W_in = malloc(reservoir->num_neurons * reservoir->num_inputs * sizeof(double));
    if (!reservoir->W_in) {
        fprintf(stderr, "Error allocating memory for W_in, size of reservoir: %zu\n",
                reservoir->num_neurons);
        return EXIT_FAILURE;
    }

    /* zero init output weights */
    reservoir->W_out = calloc(reservoir->num_neurons * reservoir->num_outputs, sizeof(double));
    if (!reservoir->W_out) {
        fprintf(stderr, "Error allocating memory for W_out, size of reservoir: %zu\n",
                reservoir->num_neurons);
        free(reservoir->W_in);
        reservoir->W_in = NULL;
        return EXIT_FAILURE;
    }

    /* Recurrent weights are accumulated as edges rather than into a dense
     * n*n scratch buffer, which is what used to dominate construction memory.
     * Rows are pre-sized to the expected out-degree so generation rarely has
     * to grow one; unassigned pairs simply never appear. */
    size_t hint = (size_t)(reservoir->connectivity * (double)reservoir->num_neurons * 1.15) + 8;
    if (hint > reservoir->num_neurons) hint = reservoir->num_neurons;
    struct edge_rows W_edges;
    if (edge_rows_init(&W_edges, reservoir->num_neurons, hint) != 0) {
        fprintf(stderr, "Error allocating scratch memory for W, size of reservoir: %zu\n",
                reservoir->num_neurons);
        free(reservoir->W_in);
        free(reservoir->W_out);
        reservoir->W_in = NULL;
        reservoir->W_out = NULL;
        return EXIT_FAILURE;
    }

    /* common: input weights always get initialized regardless of connectivity model */
    {
        size_t nin = reservoir->num_neurons * reservoir->num_inputs;
        for (size_t k = 0; k < nin; k++)
            reservoir->W_in[k] = generate_weight(&reservoir->rng, reservoir->ei_ratio);
    }

    /* Dale's law: one identity per neuron, drawn once, so that every synapse
     * leaving it shares a sign. NULL under EI_PER_SYNAPSE, which leaves the
     * per-connection signs from generate_weight() alone and consumes no extra
     * draws -- so default networks are unchanged. */
    signed char *neuron_sign = NULL;
    if (reservoir->ei_mode == EI_PER_NEURON) {
        neuron_sign = malloc(reservoir->num_neurons * sizeof(*neuron_sign));
        reservoir->neuron_sign = neuron_sign;
        if (!neuron_sign) {
            fprintf(stderr, "Failed to allocate E/I identity table\n");
            edge_rows_free(&W_edges);
            return EXIT_FAILURE;
        }
        for (size_t k = 0; k < reservoir->num_neurons; k++)
            neuron_sign[k] = (urand01(&reservoir->rng) < reservoir->ei_ratio) ? 1 : -1;
    }

    switch (reservoir->connectivity_type) {
        case RANDOM: {
            /* Bernoulli directed graph with edge prob = connectivity, no self-loops */
            size_t n = reservoir->num_neurons;
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    if (i == j)
                        continue;
                    if (urand01(&reservoir->rng) < reservoir->connectivity)
                        add_edge(&W_edges, n, &reservoir->rng, reservoir->ei_ratio, neuron_sign, i, j);
                }
            }
        } break;

        case SMALL_WORLD: {
            /* Watts–Strogatz on a directed ring:
             * - base out-degree K approximates density: K = round(connectivity * (n-1))
             * - ensure K >= 1
             * - rewire each ring edge with probability p = 0.1 (tweak if you like)
             */
            size_t n = reservoir->num_neurons;
            if (n < 3) break;

            int K = (int)((reservoir->connectivity) * (double)(n - 1) + 0.5);
            if (K < 1) K = 1;
            if (K >= (int)n) K = (int)n - 1;

            double p = 0.1;             /* default WS rewiring prob */
            /* 1) ring lattice: each i connects to K forward neighbors (directed),
             *    giving directed density K/(n-1) = connectivity, consistent with
             *    RANDOM. The undirected projection is a WS lattice of degree 2K. */
            for (size_t i = 0; i < n; i++) {
                for (int s = 1; s <= K; s++) {
                    size_t j = (i + (size_t)s) % n;   /* forward neighbor */
                    if (i == j) continue;
                    add_edge(&W_edges, n, &reservoir->rng, reservoir->ei_ratio, neuron_sign, i, j);
                }
            }
            /* 2) rewire each (i -> i+s) with prob p to a random j != i, no duplicate edges */
            for (size_t i = 0; i < n; i++) {
                for (int s = 1; s <= K; s++) {
                    size_t j_old = (i + (size_t)s) % n;
                    if (urand01(&reservoir->rng) < p) {
                        /* drop old edge */
                        edge_rows_set(&W_edges, i, j_old, 0.0);
                        /* choose a new target j_new */
                        size_t j_new;
                        int attempts = 0;
                        do {
                            j_new = (size_t)(urand01(&reservoir->rng) * (double)n);
                            if (j_new >= n) j_new = n - 1;
                            if (++attempts > 10 * (int)n) break; /* fail-safe */
                        } while (j_new == i || has_edge(&W_edges, n, i, j_new));
                        if (j_new != i)
                            add_edge_over(&W_edges, n, &reservoir->rng, reservoir->ei_ratio, neuron_sign, i, j_new);
                    }
                }
            }
        } break;
        
        case SCALE_FREE: {
            /* Undirected BA → random orientation (both in/out heavy-tailed) */

            size_t n = reservoir->num_neurons;
            if (n < 2) break;

            /* Map connectivity to BA parameter m (edges per new node) */
            int m = (int)((reservoir->connectivity) * (double)(n - 1) + 0.5);
            if (m < 1)      m = 1;
            if (m >= (int)n) m = (int)n - 1;

            /* Seed size m0: small connected core (clique). */
            int m0 = m > 2 ? m : 2;
            if (m0 >= (int)n) m0 = (int)n - 1;

            /* Optional initial attractiveness (deg + A). A=1 softens hubs a bit; A=0 is classic BA. */
            const double A = 1.0;

            /* Temporary undirected adjacency (byte per entry; fine for N~few thousands).
               If you prefer, use a bitset, but this keeps code simple. */
            unsigned char *adj = calloc(n * n, sizeof(unsigned char));
            int *deg = calloc(n, sizeof(int));
            if (!adj || !deg) {
                fprintf(stderr, "alloc fail in SCALE_FREE\n");
                free(adj); free(deg);
                return EXIT_FAILURE;
            }

            /* --- Seed: clique on m0 nodes (undirected, no self) --- */
            for (int u = 0; u < m0; u++) {
                for (int v = u + 1; v < m0; v++) {
                    adj[(size_t)u * n + (size_t)v] = 1;
                    adj[(size_t)v * n + (size_t)u] = 1;
                    deg[u]++; deg[v]++;
                }
            }

            /* --- Growth: for v = m0..n-1, connect to m existing nodes via PA on (deg + A) --- */
            for (size_t v = (size_t)m0; v < n; v++) {
                int added = 0;
                int guard = 0;
                while (added < m && guard < 50 * m) {
                    /* total "attractiveness" over existing nodes */
                    double total = 0.0;
                    for (size_t u = 0; u < v; u++) total += (double)deg[u] + A;
                    if (total <= 0.0) total = (double)v; /* fallback uniform */

                    /* roulette-wheel pick */
                    double r = urand01(&reservoir->rng) * total, acc = 0.0;
                    size_t pick = 0;
                    for (size_t u = 0; u < v; u++) {
                        acc += (double)deg[u] + A;
                        if (r <= acc) { pick = u; break; }
                    }

                    if (pick != v && !adj[v * n + pick]) {
                        adj[v * n + pick] = 1;
                        adj[pick * n + v] = 1;
                        deg[v]++; deg[pick]++;
                        added++;
                    }
                    guard++;
                }

                /* Fallback to random unique partners if PA got stuck (rare on small N) */
                while (added < m) {
                    size_t pick = (size_t)(urand01(&reservoir->rng) * (double)v);
                    if (pick >= v) pick = v - 1;
                    if (pick != v && !adj[v * n + pick]) {
                        adj[v * n + pick] = 1;
                        adj[pick * n + v] = 1;
                        deg[v]++; deg[pick]++;
                        added++;
                    }
                }
            }

            /* --- Orient each undirected edge randomly and assign weight --- */
            for (size_t i = 0; i < n; i++) {
                for (size_t j = i + 1; j < n; j++) {
                    if (!adj[i * n + j]) continue;
                    if (urand01(&reservoir->rng) < 0.5) {
                        add_edge(&W_edges, n, &reservoir->rng, reservoir->ei_ratio, neuron_sign, i, j);
                    } else {
                        add_edge(&W_edges, n, &reservoir->rng, reservoir->ei_ratio, neuron_sign, j, i);
                    }
                }
            }

            free(adj);
            free(deg);
        } break;
    }

    reservoir->W = synapse_build_from_rows(&W_edges, reservoir->num_neurons,
                                           reservoir->synapse_type, reservoir->synapse_backend,
                                           reservoir->synapse_params, &reservoir->rng);
    edge_rows_free(&W_edges);
    reservoir->plasticity.neuron_sign = reservoir->neuron_sign;

    return EXIT_SUCCESS;
}


int rescale_weights(struct reservoir *reservoir) 
{
    // rescale weights to ensure spectral radius
    double current_spectral_radius = synapse_spectral_radius(&reservoir->W);
    if (current_spectral_radius > 1e-9) { // avoid division by zero
        double rescaling_factor = reservoir->spectral_radius / current_spectral_radius;
        synapse_scale(&reservoir->W, rescaling_factor);
    }
    return EXIT_SUCCESS;
}

int randomize_output_layer(struct reservoir *reservoir) 
{
    if (reservoir == NULL) {
        fprintf(stderr, "Failed to randomize output weights. Reservoir not initialized!\n");
        return EXIT_FAILURE;
    }

    double ei_ratio = 0.5; // good overall starting point for rich outputs
    for (size_t i = 0; i < reservoir->num_neurons * reservoir->num_outputs; i++) {
        reservoir->W_out[i] = generate_weight(&reservoir->rng, ei_ratio);
    }
    return EXIT_SUCCESS;
}

void train_output_ridge_regression(struct reservoir *reservoir, double *input_series, 
                                double *target_series, size_t series_length, double lambda) 
{
    if (reservoir == NULL) {
        fprintf(stderr, "Error training. Reservoir not initialized!");
        return;
    }

    size_t num_neurons = reservoir->num_neurons;
    size_t num_inputs = reservoir->num_inputs;
    size_t num_outputs = reservoir->num_outputs;

    // --- Step 1: Collect reservoir states (X) over the entire series ---
    // The matrix X will be (series_length x num_neurons)
    double *X = malloc(series_length * num_neurons * sizeof(double));
    if (!X) {
        fprintf(stderr, "Failed to allocate memory for state matrix X.\n");
        return;
    }
    
    reset_reservoir(reservoir);
    // Run the reservoir and record the state of every neuron at every timestep
    for (size_t t = 0; t < series_length; t++) {
        const double *current_input = &input_series[t * num_inputs];
        step_reservoir(reservoir, current_input);
        read_reservoir_state(reservoir, &X[t * num_neurons]);
    }

    // --- Step 2: Construct the matrices for the normal equation A*W = B ---
    // A = X'X + lambda*I  (size: num_neurons x num_neurons)
    // B = X'Y             (size: num_neurons x 1)
    //
    // Under washout the first `washout` states, the start-up transient, are
    // harvested but left out of the fit: X and Y are taken from row `skip`.
    size_t skip = reservoir->use_washout ? reservoir->washout : 0;
    if (skip >= series_length) {
        fprintf(stderr, "Error training. Washout leaves no states to fit.\n");
        free(X);
        return;
    }
    double *X_fit = X + skip * num_neurons;
    size_t n_fit = series_length - skip;

    double *X_T = malloc(num_neurons * n_fit * sizeof(double));
    double *A = malloc(num_neurons * num_neurons * sizeof(double));
    double *B = malloc(num_neurons * sizeof(double));
    if (!X_T || !A || !B) {
        fprintf(stderr, "Failed to allocate memory for regression matrices.\n");
        free(X); free(X_T); free(A); free(B);
        return;
    }

    // Calculate X_T (transpose of X)
    mat_transpose(X_fit, X_T, n_fit, num_neurons);

    // Calculate A = X_T * X
    mat_mat_mult(X_T, X_fit, A, num_neurons, n_fit, num_neurons);

    // Add the ridge term: A = A + lambda * I
    for (size_t i = 0; i < num_neurons; i++) {
        A[i * num_neurons + i] += lambda;
    }

    // --- Step 3: Solve for each output channel in parallel ---
    #pragma omp parallel for
    for (size_t i = 0; i < num_outputs; i++) {
        // Each thread allocates its own copy of A on the heap to prevent stack overflow.
        double *A_copy = malloc(num_neurons * num_neurons * sizeof(double));
        if (!A_copy) {
            fprintf(stderr, "Error: Failed to allocate memory for A_copy in thread %d\n", omp_get_thread_num());
            continue; // Skip this iteration if memory allocation fails
        }
        memcpy(A_copy, A, num_neurons * num_neurons * sizeof(double));

        // Create the target vector 'b' for this specific output channel: b = X_T * Y_i
        double *b = malloc(num_neurons * sizeof(double));
        if (!b) { 
            free(A_copy); // Clean up before skipping
            continue;
        }
        
        // First, extract the i-th column from the target_series matrix into Y_i
        double *Y_i = malloc(n_fit * sizeof(double));
        if (!Y_i) {
            free(A_copy);
            free(b);
            continue;
        }
        for(size_t t = 0; t < n_fit; t++) {
            Y_i[t] = target_series[(t + skip) * num_outputs + i];
        }

        // Then, compute b = X_T * Y_i
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    num_neurons, n_fit, 1.0, X_T, n_fit,
                    Y_i, 1, 0.0, b, 1);

        // Solve A*w = b for w, where w is the i-th row of W_out
        double *w_solution_row = &reservoir->W_out[i * num_neurons];
        
        // Check the status of the linear solve
        int status = solve_linear_system_lud(A_copy, b, w_solution_row, num_neurons);
        if (status != 0) {
            // Print a warning instead of exiting, as other threads may be succeeding.
            fprintf(stderr, "Warning: Ridge regression failed for output channel %zu.\n", i);
        }

        // Clean up all memory allocated within the loop
        free(A_copy);
        free(b);
        free(Y_i);
    }

    // --- Step 4: Cleanup ---
    free(X);
    free(X_T);
    free(A);
}

void train_output_rls(struct reservoir *reservoir, double *input_series,
                      double *target_series, size_t series_length,
                      double delta, double lambda)
{
    if (reservoir == NULL) {
        fprintf(stderr, "Error training. Reservoir not initialized!");
        return;
    }

    size_t num_neurons = reservoir->num_neurons;
    size_t num_inputs  = reservoir->num_inputs;
    size_t num_outputs = reservoir->num_outputs;

    // --- Step 1: Allocate the inverse correlation matrix P ---
    double *P = calloc(num_neurons * num_neurons, sizeof(double));
    if (!P) {
        fprintf(stderr, "Failed to allocate memory for RLS correlation matrix P.\n");
        return;
    }

    // --- Step 2: Allocate workspace arrays ---
    double *state = malloc(num_neurons * sizeof(double));
    double *Px    = malloc(num_neurons * sizeof(double));
    double *Wx    = malloc(num_outputs * sizeof(double));
    if (!state || !Px || !Wx) {
        fprintf(stderr, "Failed to allocate memory for RLS workspace.\n");
        free(P); free(state); free(Px); free(Wx);
        return;
    }

    // --- Step 3: Initialize P = (1/delta) * I ---
    double inv_delta = 1.0 / delta;
    for (size_t i = 0; i < num_neurons; i++)
        P[i * num_neurons + i] = inv_delta;

    // --- Step 4: Zero W_out and reset reservoir state ---
    memset(reservoir->W_out, 0, num_outputs * num_neurons * sizeof(double));
    reset_reservoir(reservoir);

    // --- Step 5: RLS update loop ---
    for (size_t t = 0; t < series_length; t++) {
        const double *current_input  = &input_series[t * num_inputs];
        const double *current_target = &target_series[t * num_outputs];

        step_reservoir(reservoir, current_input);
        if (in_washout(reservoir))
            continue;   // start-up transient: step, but do not update
        read_reservoir_state(reservoir, state);

        // Px = P * x
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    num_neurons, num_neurons, 1.0, P, num_neurons,
                    state, 1, 0.0, Px, 1);

        // denom = lambda + x' * Px
        double denom = lambda + cblas_ddot(num_neurons, state, 1, Px, 1);

        // k = Px / denom  (in-place; Px now holds the gain vector k)
        cblas_dscal(num_neurons, 1.0 / denom, Px, 1);

        // e = target - W_out * x  (Wx holds the error vector)
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    num_outputs, num_neurons, 1.0, reservoir->W_out, num_neurons,
                    state, 1, 0.0, Wx, 1);
        for (size_t i = 0; i < num_outputs; i++)
            Wx[i] = current_target[i] - Wx[i];

        // W_out += e * k'  (rank-1 update across all output channels)
        cblas_dger(CblasRowMajor, num_outputs, num_neurons,
                   1.0, Wx, 1, Px, 1,
                   reservoir->W_out, num_neurons);

        // P -= denom * k * k'
        // Shortcut: since P is symmetric, k * x'P = denom * k * k'
        cblas_dger(CblasRowMajor, num_neurons, num_neurons,
                   -denom, Px, 1, Px, 1,
                   P, num_neurons);

        // P /= lambda  (forgetting factor scaling)
        if (lambda != 1.0)
            cblas_dscal(num_neurons * num_neurons, 1.0 / lambda, P, 1);
    }

    // --- Step 6: Cleanup ---
    free(P);
    free(state);
    free(Px);
    free(Wx);
}

// Function to reset neurons in reservoir (helpful for fractional order neuron models)
void reset_reservoir(struct reservoir *reservoir)
{
    if (reservoir == NULL) {
        fprintf(stderr, "Error resetting reservoir. Reservoir not initialized!\n");
    }
    void *local_ptr = reservoir->shared_neuron_data;
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        free_neuron(reservoir->neurons[i], reservoir->neuron_type);
        reservoir->neurons[i] = init_neuron(reservoir->neuron_type, reservoir->neuron_params, reservoir->dt, &local_ptr);
    }
    seed_neurons(reservoir);
    reservoir->steps_since_reset = 0;
}

struct reservoir *coarse_grain_reservoir(const struct reservoir *r, double weight_threshold)
{
    if (!r) {
        fprintf(stderr, "Error in coarse-graining: reservoir is NULL.\n");
        return NULL;
    }

    size_t num_neurons = r->num_neurons;
    size_t num_inputs  = r->num_inputs;
    size_t num_outputs = r->num_outputs;

    double *W_work = malloc(num_neurons * num_neurons * sizeof(double));
    if (!W_work) {
        fprintf(stderr, "Failed to allocate W working copy for coarse-graining.\n");
        return NULL;
    }

    double *W_in_work = malloc(num_neurons * num_inputs * sizeof(double));
    if (!W_in_work) {
        fprintf(stderr, "Failed to allocate W_in working copy for coarse-graining.\n");
        free(W_work);
        return NULL;
    }

    double *V = malloc(num_neurons * sizeof(double));
    if (!V) {
        fprintf(stderr, "Failed to allocate neuron state buffer for coarse-graining.\n");
        free(W_work); free(W_in_work);
        return NULL;
    }

    int *alive = malloc(num_neurons * sizeof(int));
    if (!alive) {
        fprintf(stderr, "Failed to allocate alive array for coarse-graining.\n");
        free(W_work); free(W_in_work); free(V);
        return NULL;
    }

    double *W_out_work = malloc(num_outputs * num_neurons * sizeof(double));
    if (!W_out_work) {
        fprintf(stderr, "Failed to allocate W_out working copy for coarse-graining.\n");
        free(W_work); free(W_in_work); free(V); free(alive);
        return NULL;
    }

    synapse_to_dense(&r->W, W_work);
    memcpy(W_in_work, r->W_in, num_neurons * num_inputs * sizeof(double));
    memcpy(W_out_work, r->W_out, num_outputs * num_neurons * sizeof(double));
    for (size_t i = 0; i < num_neurons; i++) {
        V[i]     = get_neuron_state(r->neurons[i], r->neuron_type);
        alive[i] = 1;
    }

    int merged = 1;
    while (merged) {
        merged = 0;
        for (size_t i = 0; i < num_neurons; i++) {
            for (size_t j = i + 1; j < num_neurons; j++) {
                if (W_work[i * num_neurons + j] <= weight_threshold &&
                    W_work[j * num_neurons + i] <= weight_threshold)
                    continue;

                merged = 1;

                V[i] = (V[i] + V[j]) / 2.0;
                V[j] = 0.0;
                alive[j] = 0;

                for (size_t k = 0; k < num_neurons; k++) {
                    if (k == i || k == j)
                        continue;
                    double w_ik = W_work[i * num_neurons + k];
                    double w_jk = W_work[j * num_neurons + k];
                    double new_w = (w_ik * w_jk == 0.0) ? w_ik + w_jk
                                                         : (w_ik + w_jk) / 2.0;
                    W_work[i * num_neurons + k] = new_w;
                    W_work[k * num_neurons + i] = new_w;
                }

                for (size_t k = 0; k < num_inputs; k++) {
                    double w_ik = W_in_work[i * num_inputs + k];
                    double w_jk = W_in_work[j * num_inputs + k];
                    W_in_work[i * num_inputs + k] = (w_ik * w_jk == 0.0) ? w_ik + w_jk
                                                                           : (w_ik + w_jk) / 2.0;
                    W_in_work[j * num_inputs + k] = 0.0;
                }

                for (size_t o = 0; o < num_outputs; o++) {
                    W_out_work[o * num_neurons + i] += W_out_work[o * num_neurons + j];
                    W_out_work[o * num_neurons + j]  = 0.0;
                }

                for (size_t k = 0; k < num_neurons; k++) {
                    W_work[j * num_neurons + k] = 0.0;
                    W_work[k * num_neurons + j] = 0.0;
                }
            }
        }
    }

    size_t num_super = 0;
    for (size_t i = 0; i < num_neurons; i++) {
        if (alive[i])
            num_super++;
    }

    size_t *old_to_new = malloc(num_neurons * sizeof(size_t));
    if (!old_to_new) {
        fprintf(stderr, "Failed to allocate index map for coarse-grained reservoir.\n");
        free(W_work); free(W_in_work); free(W_out_work); free(V); free(alive);
        return NULL;
    }

    size_t idx = 0;
    for (size_t i = 0; i < num_neurons; i++)
        old_to_new[i] = alive[i] ? idx++ : (size_t)-1;

    double *W_new = calloc(num_super * num_super, sizeof(double));
    if (!W_new) {
        fprintf(stderr, "Failed to allocate recurrent weights for coarse-grained reservoir.\n");
        free(W_work); free(W_in_work); free(W_out_work); free(V); free(alive); free(old_to_new);
        return NULL;
    }

    double *W_in_new = calloc(num_super * num_inputs, sizeof(double));
    if (!W_in_new) {
        fprintf(stderr, "Failed to allocate input weights for coarse-grained reservoir.\n");
        free(W_work); free(W_in_work); free(W_out_work); free(V); free(alive); free(old_to_new);
        free(W_new);
        return NULL;
    }

    double *W_out_new = calloc(num_outputs * num_super, sizeof(double));
    if (!W_out_new) {
        fprintf(stderr, "Failed to allocate output weights for coarse-grained reservoir.\n");
        free(W_work); free(W_in_work); free(W_out_work); free(V); free(alive); free(old_to_new);
        free(W_new); free(W_in_new);
        return NULL;
    }

    for (size_t i = 0; i < num_neurons; i++) {
        if (!alive[i])
            continue;
        size_t ni = old_to_new[i];
        for (size_t j = 0; j < num_neurons; j++) {
            if (!alive[j])
                continue;
            W_new[ni * num_super + old_to_new[j]] = W_work[i * num_neurons + j];
        }
        for (size_t k = 0; k < num_inputs; k++)
            W_in_new[ni * num_inputs + k] = W_in_work[i * num_inputs + k];
        for (size_t o = 0; o < num_outputs; o++)
            W_out_new[o * num_super + ni] = W_out_work[o * num_neurons + i];
    }

    free(W_work); free(W_in_work); free(W_out_work); free(old_to_new);

    struct reservoir_params cg_params = {
        .num_neurons       = num_super,
        .num_inputs        = num_inputs,
        .num_outputs       = num_outputs,
        .spectral_radius   = r->spectral_radius,
        .ei_ratio          = r->ei_ratio,
        .input_strength    = r->input_strength,
        .connectivity      = r->connectivity,
        .dt                = r->dt,
        .connectivity_type = r->connectivity_type,
        .neuron_type       = r->neuron_type,
        .neuron_params     = r->neuron_params,
        .synapse_type      = r->synapse_type,
        .synapse_backend   = r->synapse_backend,
        .synapse_params    = r->synapse_params,
        /* Coarse-graining does not preserve Dale's law: neurons are merged on
         * strong positive weights, and W_ij > 0 fixes j's identity but not
         * i's, so a merge can combine an excitatory and an inhibitory neuron.
         * The result is reported honestly as per-synapse rather than carrying
         * an ei_mode it no longer satisfies. */
        .ei_mode           = EI_PER_SYNAPSE,
        .plasticity_type   = PLASTICITY_NONE,
        .seed              = r->seed,
        .use_washout       = r->use_washout,
        .washout           = r->washout,
    };
    struct reservoir *new_r = create_reservoir(&cg_params);
    if (!new_r) {
        fprintf(stderr, "Failed to create coarse-grained reservoir.\n");
        free(W_new); free(W_in_new); free(W_out_new); free(V); free(alive);
        return NULL;
    }

    new_r->W = synapse_build_from_dense(W_new, num_super, r->W.type, r->W.backend, r->synapse_params, &new_r->rng);
    free(W_new);
    new_r->W_in  = W_in_new;
    new_r->W_out = W_out_new;

    idx = 0;
    for (size_t i = 0; i < num_neurons; i++) {
        if (!alive[i])
            continue;
        set_neuron_state(new_r->neurons[idx], new_r->neuron_type, V[i]);
        idx++;
    }

    free(V); free(alive);

    return new_r;
}

double *copy_reservoir_weights(const struct reservoir *r)
{
    if (!r || !r->W.n)
        return NULL;
    size_t n = r->num_neurons;
    double *buf = malloc(n * n * sizeof(double));
    if (!buf)
        return NULL;
    synapse_to_dense(&r->W, buf);
    return buf;
}

void read_reservoir_weights(const struct reservoir *r, double *buffer)
{
    if (!r || !r->W.n || !buffer)
        return;
    synapse_to_dense(&r->W, buffer);
}

double *copy_reservoir_readout(const struct reservoir *r)
{
    if (!r)
        return NULL;
    size_t n = r->num_outputs * r->num_neurons;
    double *buf = malloc(n * sizeof(double));
    if (!buf)
        return NULL;
    memcpy(buf, r->W_out, n * sizeof(double));
    return buf;
}

void read_reservoir_readout(const struct reservoir *r, double *buffer)
{
    if (!r || !buffer)
        return;
    size_t n = r->num_outputs * r->num_neurons;
    memcpy(buffer, r->W_out, n * sizeof(double));
}
