#include<stdio.h>
#include<stdlib.h>
#include "neuron.h"

// Create a neuron with random weights
void* init_neuron(NeuronType type) {
    void* neuron = NULL;
    switch(type) {
        case LIF: 
            neuron = (LIFNeuron *)init_LIF();
            break;
        case FLIF:
            neuron = (FLIFNeuron *)init_FLIF();
            break;
        default:
            fprintf(stderr, "Neuron type unavailable.\n");
    }
    return neuron;
}

// Update neuron state based on inputs
void update_neuron(void *neuron, double *inputs) {
    
}

// Free allocated memory
void free_neuron(void** neuron, NeuronType type) {
    if (!neuron || !(*neuron)) { return; };
    switch(type) {
        case LIF:
            free_LIF((LIFNeuron**)neuron);
            break;
        case FLIF:
            free_FLIF((FLIFNeuron**)neuron);
            break;
    }
    *neuron = NULL;
}

