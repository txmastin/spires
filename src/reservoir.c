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

    for (size_t i = 0; i < num_neurons; i++) {
        reservoir->neurons[i] = init_neuron(neuron_type, neuron_params, dt); 
    }

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
double *read_reservoir_state(struct reservoir *reservoir) 
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
    double *state = read_reservoir_state(r);
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

int init_weights(struct reservoir *reservoir) 
{
    reservoir->W_in = malloc(reservoir->num_neurons * reservoir->num_inputs * sizeof(double)); 
    if(reservoir->W_in == NULL) {
        fprintf(stderr, "Error allocating memory for W_in, size of reservoir: %zu\n", reservoir->num_neurons);
        return EXIT_FAILURE;
    }
    
    reservoir->W_out = calloc(reservoir->num_neurons * reservoir->num_outputs, sizeof(double)); // set output weights to zero initially
    if(reservoir->W_out == NULL) {
        fprintf(stderr, "Error allocating memory for W_out, size of reservoir: %zu\n", reservoir->num_neurons);
        free(reservoir->W_in);
        reservoir->W_in = NULL; 
        return EXIT_FAILURE;
    }
    
    // calloc here to ensure non-assigned weights are automatically clamped to 0.0 to simplify the matrix operations
    reservoir->W = calloc(reservoir->num_neurons * reservoir->num_neurons, sizeof(double)); 
    if (reservoir->W == NULL) {
        fprintf(stderr, "Error allocating memory for W, size of reservoir: %zu\n", reservoir->num_neurons);
        free(reservoir->W_in);
        free(reservoir->W_out);
        reservoir->W_in = NULL;
        reservoir->W_out = NULL;
        return EXIT_FAILURE;
    }

    switch(reservoir->connectivity_type) {
        case RANDOM:
            for (size_t i = 0; i < reservoir->num_neurons * reservoir->num_inputs; i++) {
                reservoir->W_in[i] = generate_weight(reservoir->ei_ratio);
            }
            for (size_t i = 0; i < reservoir->num_neurons; i++) {
                for(size_t j = 0; j < reservoir->num_neurons; j++) {
                    if (i == j) continue; // no-self connections
                    if(((double)rand() / RAND_MAX) < reservoir->connectivity) {
                        reservoir->W[i * reservoir->num_neurons + j] = generate_weight(reservoir->ei_ratio);
                    }
                } 
            }
            break;

        case SMALL_WORLD:
            fprintf(stderr, "Small-world connectivity not yet implemented.\n");
            break;

        case SCALE_FREE:
            fprintf(stderr, "Scale-free connectivity not yet implemented.\n");
            break;
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

        double *current_state = read_reservoir_state(reservoir); // read_reservoir_state allocates memory
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

    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        free_neuron(reservoir->neurons[i], reservoir->neuron_type);
        reservoir->neurons[i] = init_neuron(reservoir->neuron_type, reservoir->neuron_params, reservoir->dt);
    }
}
