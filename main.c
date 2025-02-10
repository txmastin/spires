#include <stdio.h>
#include <stdlib.h>
#include "reservoir.h"

int main() {
    // Define reservoir parameters
    int num_neurons = 100;
    int input_size = 10;
    int output_size = 10;
    double spectral_radius = 0.9;
    double sparsity = 0.5;
    double weight_scale = 0.1;
    ConnectivityType connectivity = DENSE;
    NeuronType neuron_type = LIF;

    // Create reservoir
    Reservoir *reservoir = create_reservoir(num_neurons, input_size, output_size,
                                            spectral_radius, sparsity, weight_scale,
                                            connectivity, neuron_type);
    
    if (!reservoir) {
        fprintf(stderr, "Error: Failed to create reservoir\n");
        return 1;
    }

    printf("Reservoir created successfully with %d neurons.\n", num_neurons);

    // Create dummy input
    double inputs[input_size];
    for (int i = 0; i < input_size; i++) {
        inputs[i] = (double)i / input_size;  // Simple ramp input
    }

    // Update the reservoir a few times
    for (int t = 0; t < 5; t++) {
        update_reservoir(reservoir, inputs);
        printf("Reservoir updated at time step %d.\n", t);
    }

    // Free memory
    free_reservoir(reservoir);

    // Check if memory was successfully freed
    if (reservoir == NULL) {
        printf("Reservoir successfully freed.\n");
    } else {
        printf("Warning: Reservoir was not properly freed.\n");
    }

    return 0;
}

