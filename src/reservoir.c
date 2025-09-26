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

void step_reservoir(struct reservoir *r, double *input_vector) 
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
           // --- OPTIMIZATION using cblas_ddot ---
            // Get a pointer to the start of the i-th row of the weight matrix W
            const double *W_row = &r->W[i * num_neurons];

            // Calculate the dot product: recurrent_input = W_row • last_spikes
            double recurrent_input = cblas_ddot(num_neurons, W_row, 1, last_spikes, 1);

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

// return the neuron->V for each neuron
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
    free(reservoir->W);
    reservoir->W_in = NULL;
    reservoir->W_out = NULL;
    reservoir->W = NULL;
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

static inline void add_edge(struct reservoir *r, size_t i, size_t j)
{
    r->W[i * r->num_neurons + j] = generate_weight(r->ei_ratio);
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

    /* zero init recurrent weights; unassigned stay 0.0 */
    reservoir->W = calloc(reservoir->num_neurons * reservoir->num_neurons, sizeof(double));
    if (!reservoir->W) {
        fprintf(stderr, "Error allocating memory for W, size of reservoir: %zu\n",
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
                        add_edge(reservoir, i, j);
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
                    add_edge(reservoir, i, j);
                }
            }
            /* 2) rewire each (i -> i+s) with prob p to a random j != i, no duplicate edges */
            for (size_t i = 0; i < n; i++) {
                for (int s = 1; s <= half; s++) {
                    size_t j_old = (i + (size_t)s) % n;
                    if (urand01() < p) {
                        /* drop old edge */
                        reservoir->W[i * n + j_old] = 0.0;
                        /* choose a new target j_new */
                        size_t j_new;
                        int attempts = 0;
                        do {
                            j_new = (size_t)(urand01() * (double)n);
                            if (j_new >= n) j_new = n - 1;
                            if (++attempts > 10 * (int)n) break; /* fail-safe */
                        } while (j_new == i || has_edge(reservoir->W, n, i, j_new));
                        if (j_new != i)
                            add_edge(reservoir, i, j_new);
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
                        reservoir->W[i * n + j] = generate_weight(reservoir->ei_ratio);
                    } else {
                        reservoir->W[j * n + i] = generate_weight(reservoir->ei_ratio);
                    }
                }
            }

            free(adj);
            free(deg);
        } break;
    }

    return EXIT_SUCCESS;
}


int rescale_weights(struct reservoir *reservoir) 
{
    // rescale weights to ensure spectral radius
    double current_spectral_radius = calc_spectral_radius(reservoir->W, reservoir->num_neurons);
    if (current_spectral_radius > 1e-9) { // avoid division by zero
        double rescaling_factor = reservoir->spectral_radius / current_spectral_radius; 
        for (size_t i = 0; i < (reservoir->num_neurons * reservoir->num_neurons); i++) {
                reservoir->W[i] *= rescaling_factor;
        }
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

        double *current_state = copy_reservoir_state(reservoir); // copy_reservoir_state allocates memory
        if (current_state) {
            memcpy(&X[t * num_neurons], current_state, num_neurons * sizeof(double));
            free(current_state); // so we must not forget to free it
        }
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
