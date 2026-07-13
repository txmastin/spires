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
#include "sparse.h"



/* **** create reservoir ***** */

struct reservoir *create_reservoir(size_t num_neurons, size_t num_inputs, size_t num_outputs, 
                                double spectral_radius, double ei_ratio, double input_strength, 
                                double connectivity, double dt, enum connectivity_type connectivity_type, 
                                enum neuron_type neuron_type, double *neuron_params) 
{

    struct reservoir *reservoir = malloc(sizeof(*reservoir));
    if (reservoir == NULL) {
        fprintf(stderr, "Error allocating memory for reservoir of size %zu\n", num_neurons);
        return NULL;
    }
    
    reservoir->num_neurons = num_neurons;
    reservoir->num_inputs = num_inputs;
    reservoir->num_outputs = num_outputs;
    reservoir->spectral_radius = spectral_radius;
    reservoir->ei_ratio = ei_ratio;
    reservoir->input_strength = input_strength;
    reservoir->connectivity = connectivity; 
    reservoir->dt = dt;
    reservoir->connectivity_type = connectivity_type;
    reservoir->neuron_type = neuron_type;
    reservoir->neuron_params = neuron_params;
    
    
    reservoir->neurons = malloc(num_neurons * sizeof(void*));
    
    if (!(reservoir->neurons)) {
        fprintf(stderr, "Memory allocation failed for reservoir neurons\n");
        free(reservoir); 
        return NULL;
    }

    void *local_shared_neuron_data = NULL; // set to NULL to start, optionally change it during neuron instantiation
    
    #pragma omp parallel for // parallelizable because race conditions are handled by init_neuron
    for (size_t i = 0; i < num_neurons; i++) {
        reservoir->neurons[i] = init_neuron(neuron_type, neuron_params, dt, &local_shared_neuron_data); 
    }
    
    reservoir->shared_neuron_data = local_shared_neuron_data; // malloc'd in function if flif_gl is used, must free

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

double *run_reservoir(struct reservoir *r, double *input_series, size_t input_length) 
{
    if (!r) {
        fprintf(stderr, "Error running reservoir. Reservoir not initialized!\n");
        return NULL;
    }

    if (r->dt <= 0.0) {
        fprintf(stderr, "Error running reservoir. dt must be greater than 0.\n");
        return NULL;
    }


    size_t num_inputs = r->num_inputs;
    size_t num_outputs = r->num_outputs;

    double *output_series = malloc(num_outputs * input_length * sizeof(double));
    if (output_series == NULL) {
        fprintf(stderr, "Error intializing memory for reservoir outputs!\n");
        return NULL;
    }

    for (size_t i = 0; i < input_length; i ++) {
        const double *current_input = &input_series[i * num_inputs];
        step_reservoir(r, current_input);

        double *current_output = &output_series[i * num_outputs];
        compute_output(r, current_output);
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
        #pragma omp parallel for // omp ftw
        for (size_t i = 0; i < num_neurons; i++) {
            // Sparse dot product: recurrent_input = W_row • last_spikes, skipping zeros
            double recurrent_input = csr_row_dot(&r->W, i, last_spikes);

            double total_input = external_inputs[i] + recurrent_input;
            update_neuron(r->neurons[i], r->neuron_type, total_input, r->dt);
            new_spikes[i] = get_neuron_spike(r->neurons[i], r->neuron_type);
        }

        memcpy(last_spikes, new_spikes, num_neurons * sizeof(double)); 
    }
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
    csr_free(&reservoir->W);
    reservoir->W_in = NULL;
    reservoir->W_out = NULL;
    free(reservoir);
    reservoir = NULL;
}


double generate_weight(double ei_ratio) 
{
    double magnitude = (double)rand() / RAND_MAX;
    if (((double)rand() / RAND_MAX) < ei_ratio) {
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
 * * @note The user is responsible for seeding the random number generator 
 * before initializing a reservoir. This should be done once at the beginning
 * of the main() application by calling srand(time(NULL)).
 */

/* --- helpers for weight init ----- */

static inline double urand01(void)
{
    return (double)rand() / (double)RAND_MAX;
}

static inline int has_edge(const double *W, size_t n, size_t i, size_t j)
{
    return W[i * n + j] != 0.0;
}

static inline void add_edge(double *W_dense, size_t n, double ei_ratio, size_t i, size_t j)
{
    W_dense[i * n + j] = generate_weight(ei_ratio);
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

    /* zero init recurrent weights scratch buffer; unassigned stay 0.0.
     * Topology generation below fills this dense buffer exactly as before;
     * it is converted to sparse CSR storage at the end of this function. */
    double *W_dense = calloc(reservoir->num_neurons * reservoir->num_neurons, sizeof(double));
    if (!W_dense) {
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
            reservoir->W_in[k] = generate_weight(reservoir->ei_ratio);
    }

    switch (reservoir->connectivity_type) {
        case RANDOM: {
            /* Bernoulli directed graph with edge prob = connectivity, no self-loops */
            size_t n = reservoir->num_neurons;
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    if (i == j)
                        continue;
                    if (urand01() < reservoir->connectivity)
                        add_edge(W_dense, n, reservoir->ei_ratio, i, j);
                }
            }
        } break;

        case SMALL_WORLD: {
            /* Watts–Strogatz on a directed ring:
             * - base out-degree K approximates density: K = round(connectivity * (n-1))
             * - ensure K >= 2 and even
             * - rewire each ring edge with probability p = 0.1 (tweak if you like)
             */
            size_t n = reservoir->num_neurons;
            if (n < 3) break;

            int K = (int)((reservoir->connectivity) * (double)(n - 1) + 0.5);
            if (K < 2) K = 2;
            if (K % 2) K++;             /* even */
            if (K >= (int)n) K = (int)n - 1;

            double p = 0.1;             /* default WS rewiring prob */
            /* 1) ring lattice: each i connects to K/2 forward neighbors (directed) */
            int half = K / 2;
            for (size_t i = 0; i < n; i++) {
                for (int s = 1; s <= half; s++) {
                    size_t j = (i + (size_t)s) % n;   /* forward neighbor */
                    if (i == j) continue;
                    add_edge(W_dense, n, reservoir->ei_ratio, i, j);
                }
            }
            /* 2) rewire each (i -> i+s) with prob p to a random j != i, no duplicate edges */
            for (size_t i = 0; i < n; i++) {
                for (int s = 1; s <= half; s++) {
                    size_t j_old = (i + (size_t)s) % n;
                    if (urand01() < p) {
                        /* drop old edge */
                        W_dense[i * n + j_old] = 0.0;
                        /* choose a new target j_new */
                        size_t j_new;
                        int attempts = 0;
                        do {
                            j_new = (size_t)(urand01() * (double)n);
                            if (j_new >= n) j_new = n - 1;
                            if (++attempts > 10 * (int)n) break; /* fail-safe */
                        } while (j_new == i || has_edge(W_dense, n, i, j_new));
                        if (j_new != i)
                            add_edge(W_dense, n, reservoir->ei_ratio, i, j_new);
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
                    double r = urand01() * total, acc = 0.0;
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
                    size_t pick = (size_t)(urand01() * (double)v);
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
                    if (urand01() < 0.5) {
                        W_dense[i * n + j] = generate_weight(reservoir->ei_ratio);
                    } else {
                        W_dense[j * n + i] = generate_weight(reservoir->ei_ratio);
                    }
                }
            }

            free(adj);
            free(deg);
        } break;
    }

    reservoir->W = csr_build_from_dense(W_dense, reservoir->num_neurons);
    free(W_dense);

    return EXIT_SUCCESS;
}


int rescale_weights(struct reservoir *reservoir) 
{
    // rescale weights to ensure spectral radius
    double current_spectral_radius = csr_spectral_radius(&reservoir->W, reservoir->num_neurons);
    if (current_spectral_radius > 1e-9) { // avoid division by zero
        double rescaling_factor = reservoir->spectral_radius / current_spectral_radius;
        csr_scale(&reservoir->W, rescaling_factor);
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
        reservoir->W_out[i] = generate_weight(ei_ratio);
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

    double *X_T = malloc(num_neurons * series_length * sizeof(double));
    double *A = malloc(num_neurons * num_neurons * sizeof(double));
    double *B = malloc(num_neurons * sizeof(double));
    if (!X_T || !A || !B) {
        fprintf(stderr, "Failed to allocate memory for regression matrices.\n");
        free(X); free(X_T); free(A); free(B);
        return;
    }

    // Calculate X_T (transpose of X)
    mat_transpose(X, X_T, series_length, num_neurons);

    // Calculate A = X_T * X
    mat_mat_mult(X_T, X, A, num_neurons, series_length, num_neurons);

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
        double *Y_i = malloc(series_length * sizeof(double));
        if (!Y_i) { 
            free(A_copy); 
            free(b);
            continue; 
        }
        for(size_t t = 0; t < series_length; t++) {
            Y_i[t] = target_series[t * num_outputs + i];
        }
        
        // Then, compute b = X_T * Y_i
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    num_neurons, series_length, 1.0, X_T, series_length,
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

    csr_to_dense(&r->W, W_work);
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

    struct reservoir *new_r = create_reservoir(
        num_super, num_inputs, num_outputs,
        r->spectral_radius, r->ei_ratio, r->input_strength,
        r->connectivity, r->dt,
        r->connectivity_type, r->neuron_type, r->neuron_params);
    if (!new_r) {
        fprintf(stderr, "Failed to create coarse-grained reservoir.\n");
        free(W_new); free(W_in_new); free(W_out_new); free(V); free(alive);
        return NULL;
    }

    new_r->W = csr_build_from_dense(W_new, num_super);
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
    if (!r || !r->W.row_ptr)
        return NULL;
    size_t n = r->num_neurons;
    double *buf = malloc(n * n * sizeof(double));
    if (!buf)
        return NULL;
    csr_to_dense(&r->W, buf);
    return buf;
}

void read_reservoir_weights(const struct reservoir *r, double *buffer)
{
    if (!r || !r->W.row_ptr || !buffer)
        return;
    csr_to_dense(&r->W, buffer);
}
