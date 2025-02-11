#include <stdlib.h>
#include <stdio.h>
#include "reservoir.h"

// Create a reservoir of neurons
Reservoir* create_reservoir(int num_neurons, int num_inputs, int num_outputs, double spectral_radius, double input_strength, double connectivity, ConnectivityType connectivity_type, NeuronType neuron_type) {
    Reservoir *reservoir = (Reservoir*)malloc(sizeof(Reservoir));
    if(reservoir == NULL) {
        fprintf(stderr, "Error allocating memory for reservoir of size %d\n", num_neurons);
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

    for (int i = 0; i < num_neurons; i++) {
        reservoir->neurons[i] = init_neuron(neuron_type); 
    }

    return reservoir;
   }

// Update all neurons in the reservoir
void update_reservoir(Reservoir *reservoir, double *inputs) {
    for (int i = 0; i < reservoir->num_neurons; i++) {
        update_neuron(reservoir->neurons[i], inputs);
    }
}

// Free reservoir memory
void free_reservoir(Reservoir **reservoir) {
    if (!reservoir || !(*reservoir)) { return; }

    printf("Freeing reservoir at address: %p\n", (void*)(*reservoir));
    printf("Freeing neurons at address: %p\n", (void*)(*reservoir)->neurons);
    printf("Freeing W_in at address: %p\n", (void*)(*reservoir)->W_in);
    printf("Freeing W_out at address: %p\n", (void*)(*reservoir)->W_out);
    printf("Freeing W at address: %p\n", (void*)(*reservoir)->W);

    for (int i = 0; i < (*reservoir)->num_neurons; i++) {
        free_neuron(&((*reservoir)->neurons[i]), (*reservoir)->neuron_type);
    }
    free((*reservoir)->neurons);
    free((*reservoir)->W_in);
    free((*reservoir)->W_out);
    free((*reservoir)->W);
    //(*reservoir)->W_in = NULL;
    //(*reservoir)->W_out = NULL;
    //(*reservoir)->W = NULL;
    free(*reservoir);
    *reservoir = NULL;
}

// Initialize weights
int init_weights(Reservoir *reservoir) {
    reservoir->W_in = (double *)malloc(reservoir->num_neurons * sizeof(double)); 
    if(reservoir->W_in == NULL) {
        fprintf(stderr, "Error allocating memory for W_in, size of reservoir: %d\n", reservoir->num_neurons);
        return 1;
    }
    
    reservoir->W_out = (double *)malloc(reservoir->num_neurons * sizeof(double));
    if(reservoir->W_out == NULL) {
        fprintf(stderr, "Error allocating memory for W_out, size of reservoir: %d\n", reservoir->num_neurons);
        free(reservoir->W_in);
        reservoir->W_in = NULL; 
        return 1;
    }
    
    // calloc here to ensure non-assigned weights are automatically clamped to 0.0, e.g., in sparse connectivity
    reservoir->W = (double *)calloc(reservoir->num_neurons * reservoir->num_neurons, sizeof(double)); 
    
    if (reservoir->W == NULL) {
        fprintf(stderr, "Error allocating memory for W, size of reservoir: %d\n", reservoir->num_neurons);
        free(reservoir->W_in);
        free(reservoir->W_out);
        reservoir->W_in = NULL;
        reservoir->W_out = NULL;
        return 1;
    }

    switch(reservoir->connectivity_type) {
        case DENSE:
            for (int i = 0; i < reservoir->num_neurons; i++) {
                reservoir->W_in[i] = (double)rand() / RAND_MAX;
                reservoir->W_out[i] = (double)rand() / RAND_MAX;
                for(int j = 0; j < reservoir->num_neurons; j++) {
                    reservoir->W[i * reservoir->num_neurons + j] = (double)rand() / RAND_MAX;
                } 
            }
            break;
 
        case SPARSE:
            for (int i = 0; i < reservoir->num_neurons; i++) {
                reservoir->W_in[i] = (double)rand() / RAND_MAX;
                reservoir->W_out[i] = (double)rand() / RAND_MAX;
                for(int j = 0; j < reservoir->num_neurons; j++) {
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

    return 0;
}
