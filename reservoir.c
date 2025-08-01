#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "reservoir.h"
#include "math_utils.h"

struct Reservoir* create_reservoir(
    size_t num_neurons, size_t num_inputs, size_t num_outputs, 
    double learning_rate, double spectral_radius, double ei_ratio, double input_strength, double connectivity, double dt,
    enum ConnectivityType connectivity_type, enum NeuronType neuron_type, double *neuron_params) {

    struct Reservoir *reservoir = malloc(sizeof(*reservoir));
    if (reservoir == NULL) {
        fprintf(stderr, "Error allocating memory for reservoir of size %zu\n", num_neurons);
        return NULL;
    }
    
    reservoir->num_neurons = num_neurons;
    reservoir->num_inputs = num_inputs;
    reservoir->num_outputs = num_outputs;
    reservoir->learning_rate = learning_rate;
    reservoir->spectral_radius = spectral_radius;
    reservoir->ei_ratio = ei_ratio;
    reservoir->input_strength = input_strength;
    reservoir->connectivity = connectivity; 
    reservoir->dt = dt;
    reservoir->connectivity_type = connectivity_type;
    reservoir->neuron_type = neuron_type;
    
    reservoir->neurons = (void**)malloc(num_neurons * sizeof(void*));
    
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


// todo break inputs off to send only to 'num_inputs' neurons


/* old version
void step_reservoir(struct Reservoir *r, double input) {
    for (size_t i = 0; i < r->num_neurons; i++) {
        double neuron_input = input * r->W_in[i] * r->input_strength;

        for (size_t j = 0; j < r->num_neurons; j++) {
            neuron_input += get_neuron_spike(r->neurons[j], r->neuron_type) * r->W[i * r->num_neurons + j];
        }

        update_neuron(r->neurons[i], r->neuron_type, neuron_input, r->dt);
    }
}
*/


void step_reservoir(struct Reservoir *r, double input) {
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

/****** OLD VERSION ******/
/*
void step_reservoir(struct Reservoir *reservoir, double input) {
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        double neuron_input = input * reservoir->W_in[i] * reservoir->input_strength;
        for (size_t j = 0; j < reservoir->num_neurons; j++) {
            // add inputs from other spiking neurons
            neuron_input += get_neuron_spike(reservoir->neurons[j], reservoir->neuron_type) * reservoir->W[i * reservoir->num_neurons + j];
        } 
        update_neuron(reservoir->neurons[i], reservoir->neuron_type, neuron_input);
    }
}

void run_reservoir(struct Reservoir *reservoir, double *inputs, size_t input_length) {
    for (size_t i = 0; i < input_length; i++) {
        double input = inputs[i]; 
        step_reservoir(reservoir, input); 
    }
}
*/


// compute_output calculates the sum of neuron states * W_out
double compute_output(struct Reservoir *reservoir) {
    double output = 0.0;
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        output += get_neuron_state(reservoir->neurons[i], reservoir->neuron_type) * reservoir->W_out[i];
    }

    return output;
}

// compute_activity returns the sum of spikes
double compute_activity(struct Reservoir *reservoir) {
    double total_activity = 0.0;
    double spike; 
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        spike = get_neuron_spike(reservoir->neurons[i], reservoir->neuron_type);
        total_activity += spike;
    }
    
    return total_activity;
}

// return the neuron->V for each neuron
double *read_reservoir_state(struct Reservoir *reservoir) {
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

void train_output_iteratively(struct Reservoir *reservoir, double target) {
    double *state = read_reservoir_state(reservoir); // malloc'd in function, free here
    if (state == NULL) { return; }
    double prediction = compute_output(reservoir);
    double error = target - prediction;

    // update W_out
    for (size_t i =0; i < reservoir->num_neurons; i++) {
        reservoir->W_out[i] += state[i] * error * reservoir->learning_rate;
    }

    free(state);
}

void free_reservoir(struct Reservoir *reservoir) {
    if (!reservoir) { return; }
    
    printf("Freeing reservoir at address: %p\n", (void*)reservoir);
    printf("Freeing neurons at address: %p\n", (void*)reservoir->neurons);
    printf("Freeing W_in at address: %p\n", (void*)reservoir->W_in);
    printf("Freeing W_out at address: %p\n", (void*)reservoir->W_out);
    printf("Freeing W at address: %p\n", (void*)reservoir->W);

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


double generate_weight(double ei_ratio) {
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

int init_weights(struct Reservoir *reservoir) {
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

int rescale_weights(struct Reservoir *reservoir) {
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

int randomize_output_layer(struct Reservoir *reservoir) {
    if (reservoir == NULL) {
        fprintf(stderr, "Failed to randomize output weights. Reservoir not initialized!");
        return EXIT_FAILURE;
    }

    double ei_ratio = 0.5; // good overall starting point for rich outputs
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        reservoir->W_out[i] = generate_weight(ei_ratio);
    }
    return EXIT_SUCCESS;
}
