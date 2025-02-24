#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "reservoir.h"
#include "math_utils.h"

void save_data(double *values, size_t length) {
    FILE *file = fopen("../plotting/data.txt", "w");
    if (!file) {
        printf("Error opening file!\n");
        return;
    }
    
    for (size_t i = 0; i <= length; i++) {
        fprintf(file, "%lf\n", values[i]);
    }
    
    fclose(file);
}


int main(void) {
    size_t num_neurons = 100; 
    size_t num_inputs = 100;
    size_t num_outputs = 100;
    double learning_rate = 0.0005;
    double spectral_radius = 0.9;
    double sparsity = 0.175;
    double input_strength = 5.0;
    enum ConnectivityType connectivity = DENSE;
    enum NeuronType neuron_type = LIF;
    double neuron_params[4] = {0.0, 0.7, 0.0, 0.3};
    struct Reservoir *reservoir = create_reservoir(num_neurons, num_inputs, num_outputs, learning_rate, spectral_radius, sparsity, input_strength, connectivity, neuron_type, neuron_params);
    
    if (!reservoir) {
        fprintf(stderr, "Error: Failed to create reservoir\n");
        return EXIT_FAILURE;
    }

    printf("Reservoir created successfully with %zu neurons.\n", num_neurons);
    
    int init_err = init_weights(reservoir); 
    if(!init_err) { printf("Weights initialized successfully\n");}
    
    rescale_matrix(reservoir->W, reservoir->num_neurons, reservoir->spectral_radius);
 
    double spec_radius = calc_spectral_radius(reservoir->W, reservoir->num_neurons);

    printf("Spectral Radius:\t%lf\n", spec_radius);
  
    // Create test inputs
    size_t input_length = 1000; 
    double inputs[input_length];
    for (size_t i = 0; i < input_length; i++) {
        inputs[i] = sin(i)+1;  
    }

    double reservoir_output;
 
    size_t num_epochs = 100;
    double *acc_trace = malloc(num_epochs * sizeof(double));
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        double avg_err = 0.0;
        for (size_t i = 0; i < input_length; i++) { 
            step_reservoir(reservoir, inputs[i]);
            train_output_layer(reservoir, inputs[i]);
            reservoir_output = compute_output(reservoir);
            double error = inputs[i] - reservoir_output;
            avg_err += fabs(error);
        }
        avg_err /= input_length;
        acc_trace[epoch] = avg_err;
        printf("%f\n", avg_err);
    } 

    save_data(acc_trace, num_epochs);
    
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

