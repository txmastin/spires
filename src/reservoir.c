#include <stdlib.h>
#include <stdio.h>
#include <time.h>
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
        reservoir->neurons[i] = init_neuron(neuron_type, neuron_params); 
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

// todo break inputs off to send only to 'num_inputs' neurons

double *run_reservoir(struct reservoir *r, double *input_series, size_t input_length) 
{
    if (!r || r->dt <= 0) {
        fprintf(stderr, "Error running reservoir. Reservoir not initialized!\n");
        return NULL;
    }

    double *reservoir_outputs = malloc(input_length * sizeof(double));
    if (reservoir_outputs == NULL) {
        fprintf(stderr, "Error intializing memory for reservoir outputs!\n");
        return NULL;
    }

    for (size_t i = 0; i < input_length; i ++) {
        step_reservoir(r, input_series[i]);
        reservoir_outputs[i] = compute_output(r);
    }

    return reservoir_outputs;    
}

void step_reservoir(struct reservoir *r, double input) 
{
    if (!r || r->dt <= 0) return;

    // The macro step duration is implicitly 1.0.
    // Calculate how many internal micro-steps to run.
    int num_micro_steps = (int)(1.0 / r->dt);
    size_t num_neurons = r->num_neurons;

    // --- Pre-calculate constant external inputs for this entire macro step ---
    double external_inputs[num_neurons];
    for (size_t i = 0; i < num_neurons; i++) {
        external_inputs[i] = input * r->W_in[i] * r->input_strength;
    }

    // --- Buffer to hold spikes, which evolve at the micro-step scale ---
    // We get the spikes from the previous macro_step to start.
    double last_spikes[num_neurons];
    for (size_t i = 0; i < num_neurons; i++) {
        last_spikes[i] = get_neuron_spike(r->neurons[i], r->neuron_type);
    }

    // --- Internal simulation loop (the "micro-steps") ---
    for (int t = 0; t < num_micro_steps; t++) {
        for (size_t i = 0; i < num_neurons; i++) {
            // 1. Calculate recurrent input based on spikes from the LAST micro-step
            double recurrent_input = 0.0;
            for (size_t j = 0; j < num_neurons; j++) {
                recurrent_input += last_spikes[j] * r->W[i * num_neurons + j];
            }

            // 2. Total input is the sum of the constant external and dynamic recurrent inputs
            double total_input = external_inputs[i] + recurrent_input;

            // 3. Update the neuron's internal state by one 'dt'
            update_neuron(r->neurons[i], r->neuron_type, total_input, r->dt);
        }

        // 4. Record all spikes that occurred in THIS micro-step to be used in the NEXT one
        for (size_t i = 0; i < num_neurons; i++) {
            last_spikes[i] = get_neuron_spike(r->neurons[i], r->neuron_type);
        }
    }
}


// compute_output calculates the sum of neuron states * W_out
double compute_output(struct reservoir *reservoir) 
{
    double output = 0.0;
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        output += get_neuron_state(reservoir->neurons[i], reservoir->neuron_type) * reservoir->W_out[i];
    }

    return output;
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

void train_output_iteratively(struct reservoir *reservoir, double target, double lr) 
{
    double *state = read_reservoir_state(reservoir); // malloc'd in function, free here
    if (state == NULL) { return; }
    double prediction = compute_output(reservoir);
    double error = target - prediction;

    // update W_out
    for (size_t i =0; i < reservoir->num_neurons; i++) {
        reservoir->W_out[i] += state[i] * error * lr;
    }

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
    reservoir->W_in = malloc(reservoir->num_neurons * sizeof(double)); 
    if(reservoir->W_in == NULL) {
        fprintf(stderr, "Error allocating memory for W_in, size of reservoir: %zu\n", reservoir->num_neurons);
        return EXIT_FAILURE;
    }
    
    reservoir->W_out = calloc(reservoir->num_neurons, sizeof(double)); // set output weights to zero initially
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
            for (size_t i = 0; i < reservoir->num_neurons; i++) {
                reservoir->W_in[i] = generate_weight(reservoir->ei_ratio);
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
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        reservoir->W_out[i] = generate_weight(ei_ratio);
    }
    return EXIT_SUCCESS;
}

void train_output_ridge_regression(struct reservoir *reservoir, double *input_series, 
    double *target_series, size_t series_length, double lambda) 
{
    size_t num_neurons = reservoir->num_neurons;

    // --- Step 1: Collect reservoir states (X) over the entire series ---
    // The matrix X will be (series_length x num_neurons)
    double *X = malloc(series_length * num_neurons * sizeof(double));
    if (!X) {
        fprintf(stderr, "Failed to allocate memory for state matrix X.\n");
        return;
    }

    // Run the reservoir and record the state of every neuron at every timestep
    for (size_t t = 0; t < series_length; t++) {
        step_reservoir(reservoir, input_series[t]);
        double *current_state = read_reservoir_state(reservoir);
        for (size_t n = 0; n < num_neurons; n++) {
            X[t * num_neurons + n] = current_state[n];
        }
        free(current_state);
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

    // Calculate B = X_T * Y (where Y is the target_series)
    for (size_t i = 0; i < num_neurons; i++) {
        B[i] = 0.0;
        for (size_t j = 0; j < series_length; j++) {
            B[i] += X_T[i * series_length + j] * target_series[j];
        }
    }

    // --- Step 3: Solve the system A * W_out = B for W_out ---
    // The solution will be stored directly in the reservoir's W_out.
    int status = solve_linear_system_lud(A, B, reservoir->W_out, num_neurons);
    if (status != 0) {
        fprintf(stderr, "Ridge regression failed. The system may be ill-conditioned.\n");
    }

    // --- Step 4: Cleanup ---
    free(X);
    free(X_T);
    free(A);
    free(B);
}

// Function to reset neurons in reservoir (helpful for fractional order neuron models)
void reset_reservoir(struct reservoir *reservoir) 
{
    if (reservoir == NULL) {
        fprintf(stderr, "Error resetting reservoir. Reservoir not initialized!\n");
    }

    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        free_neuron(reservoir->neurons[i], reservoir->neuron_type);
        reservoir->neurons[i] = init_neuron(reservoir->neuron_type, reservoir->neuron_params);
    }
}
