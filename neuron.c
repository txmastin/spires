#include<stdio.h>
#include<stdlib.h>
#include "neuron.h"

void* init_neuron(enum NeuronType type) {
    void* neuron = NULL;
    switch(type) {
        case LIF: 
            neuron = (LIFNeuron *)init_LIF(0.0,0.7,0.0,0.2);
            break;
        case FLIF:
            neuron = (FLIFNeuron *)init_FLIF(0.0,0.7,0.0,0.2,0.9);
            break;
        default:
            fprintf(stderr, "Neuron type unavailable.\n");
    }
    return neuron;
}

void update_neuron(void *neuron, double *inputs) {
    
}

void free_neuron(void *neuron, enum NeuronType type) {
    if (!neuron) { return; };
    switch(type) {
        case LIF:
            free_LIF((LIFNeuron*)neuron);
            break;
        case FLIF:
            free_FLIF((FLIFNeuron*)neuron);
            break;
    }
    neuron = NULL;
}

