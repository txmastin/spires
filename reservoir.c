#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "reservoir.h"
#include "math_utils.h"

// Create a reservoir of neurons
struct Reservoir* create_reservoir(
    size_t num_neurons, size_t num_inputs, size_t num_outputs, 
    double spectral_radius, double input_strength, double connectivity, 
    enum ConnectivityType connectivity_type, enum NeuronType neuron_type) {

    struct Reservoir *reservoir = malloc(sizeof(*reservoir));
    if (reservoir == NULL) {
        fprintf(stderr, "Error allocating memory for reservoir of size %zu\n", num_neurons);
        return NULL;
    }
    
    reservoir->num_neurons = num_neurons;
    reservoir->num_inputs = num_inputs;
    reservoir->num_outputs = num_outputs;
    reservoir->spectral_radius = spectral_radius;
    reservoir->input_strength = input_strength;
    reservoir->connectivity = connectivity; 
    reservoir->connectivity_type = connectivity_type;
    reservoir->neuron_type = neuron_type;
    
    reservoir->neurons = (void**)malloc(num_neurons * sizeof(void*));
    
    if (!(reservoir->neurons)) {
        fprintf(stderr, "Memory allocation failed for reservoir neurons\n");
        free(reservoir); 
        return NULL;
    }

    for (size_t i = 0; i < num_neurons; i++) {
        reservoir->neurons[i] = init_neuron(neuron_type); 
    }

    return reservoir;
}

// send input to each neuron
// todo break inputs off to send only to 'num_inputs' neurons
void step_reservoir(struct Reservoir *reservoir, double input) {
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        double neuron_input = input * reservoir->W_in[i] * reservoir->input_strength;
        for (size_t j = i; j < reservoir->num_neurons; j++) {
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

// read reservoir calculates the sum of neuron states * W_out
double read_reservoir(struct Reservoir *reservoir) {
    double output = 0.0;
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        output += get_neuron_state(reservoir->neurons[i], reservoir->neuron_type) * reservoir->W_out[i];
    }

    return output;
}

double read_spikes(struct Reservoir *reservoir) {
    double total_activity = 0.0;
    double spike; 
    for (size_t i = 0; i < reservoir->num_neurons; i++) {
        spike = get_neuron_spike(reservoir->neurons[i], reservoir->neuron_type);
        total_activity += spike;
        //printf("%f ", spike);
    }
    printf("%f", total_activity);
    
    return total_activity;
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

// Initialize weights
int init_weights(struct Reservoir *reservoir) {
    reservoir->W_in = malloc(reservoir->num_neurons * sizeof(double)); 
    if(reservoir->W_in == NULL) {
        fprintf(stderr, "Error allocating memory for W_in, size of reservoir: %zu\n", reservoir->num_neurons);
        return EXIT_FAILURE;
    }
    
    reservoir->W_out = malloc(reservoir->num_neurons * sizeof(double));
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
    
    // look into replacing rand() in the future for better statistical distribution 
    srand(time(NULL));

    switch(reservoir->connectivity_type) {
        case DENSE:
            for (size_t i = 0; i < reservoir->num_neurons; i++) {
                reservoir->W_in[i] = (double)rand() / RAND_MAX;
                reservoir->W_out[i] = (double)rand() / RAND_MAX;
                for(size_t j = 0; j < reservoir->num_neurons; j++) {
                    reservoir->W[i * reservoir->num_neurons + j] = (double)rand() / RAND_MAX;
                } 
            }
            break;
 
        case SPARSE:
            for (size_t i = 0; i < reservoir->num_neurons; i++) {
                reservoir->W_in[i] = (double)rand() / RAND_MAX;
                reservoir->W_out[i] = (double)rand() / RAND_MAX;
                for(size_t j = 0; j < reservoir->num_neurons; j++) {
                    if(((double)rand() / RAND_MAX) > reservoir->connectivity) {
                        reservoir->W[i * reservoir->num_neurons + j] = (double)rand() / RAND_MAX;
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
