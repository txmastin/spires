#include <stdio.h>
#include <stdlib.h>
#include "reservoir.h"
#include "math_utils.h"

int main(void) {
    // Define reservoir parameters
    size_t num_neurons = 100; 
    size_t num_inputs = 100;
    size_t num_outputs = 100;
    double spectral_radius = 0.95;
    double sparsity = 0.2;
    double input_strength = 1.0;
    enum ConnectivityType connectivity = DENSE;
    enum NeuronType neuron_type = LIF;

    // Create reservoir
    struct Reservoir *reservoir = create_reservoir(num_neurons, num_inputs, num_outputs,
                                            spectral_radius, sparsity, input_strength,
                                            connectivity, neuron_type);
    
    if (!reservoir) {
        fprintf(stderr, "Error: Failed to create reservoir\n");
        return EXIT_FAILURE;
    }

    printf("Reservoir created successfully with %zu neurons.\n", num_neurons);
    
    int init_err = init_weights(reservoir); 
    if(!init_err) { printf("Weights initialized successfully\n");}
    
    printf("Allocated weights:\n");
    
//    for (size_t i = 0; i < reservoir->num_neurons; i++) {
//        for (size_t j = 0; j < reservoir->num_neurons; j++) {
//            printf("%lf ", reservoir->W[i*reservoir->num_neurons + j]);
//        }
//        printf("\n");
//    }

    rescale_matrix(reservoir->W, reservoir->num_neurons, reservoir->spectral_radius);
 
    double spec_radius = calc_spectral_radius(reservoir->W, reservoir->num_neurons);

    printf("Spectral Radius:\t%lf\n", spec_radius);
  
        // Create test inputs
    size_t input_length = 100; 
    double inputs[input_length];
    for (size_t i = 0; i < input_length; i++) {
        inputs[i] = 2.0;  // Simple constant input
    }
    
    for (size_t i = 0; i < input_length; i++) {
        printf("%f ", inputs[i]);
    }
    printf("\n");
    // run inputs through reservoir for num_epochs
    size_t num_epochs = 5;
    
    double reservoir_state = read_reservoir(reservoir);
    printf("Reservoir state: %lf\n", reservoir_state);
    
    double current_spikes = 0.0;
    for (size_t i = 0; i < input_length; i++) { 
        step_reservoir(reservoir, inputs[i]);
        reservoir_state = read_reservoir(reservoir);
        current_spikes = read_spikes(reservoir); 
        
        printf("\ntotal spikes:\t%f\n", current_spikes);
    }

    // Free memory
    free_reservoir(reservoir);
    reservoir = NULL;
    // Check if memory was successfully freed
    if (reservoir == NULL) {
        printf("Reservoir successfully freed.\n");
    } else {
        printf("Warning: Reservoir was not properly freed.\n");
    }

    printf("Successfull reached end of program.\n");
    
    return EXIT_SUCCESS;
}

