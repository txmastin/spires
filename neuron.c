#include<stdio.h>
#include<stdlib.h>
#include "neuron.h"

void* init_neuron(enum NeuronType type, double *neuron_params) {
    void* neuron = NULL;
    switch(type) {
        case LIF: 
            neuron = (LIFNeuron *)init_LIF(neuron_params);
            break;
        case FLIF:
            neuron = (FLIFNeuron *)init_FLIF(neuron_params);
            break;
        case FLIF_Caputo:
            neuron = (FLIFCaputoNeuron *)init_FLIF_Caputo(neuron_params);
            break;
        case FLIF_GL:
            neuron = (FLIFGLNeuron *)init_FLIF_GL(neuron_params);
            break;
        default:
            fprintf(stderr, "Neuron type unavailable.\n");
    }
    return neuron;
}

void update_neuron(void *neuron, enum NeuronType type, double input) {
    // update neuron based on type  
    switch(type) {
        case LIF:
            update_LIF((LIFNeuron *)neuron, input);
            break;
        case FLIF:
            update_FLIF((FLIFNeuron *)neuron, input);
            break;
        case FLIF_Caputo:
            update_FLIF_Caputo((FLIFCaputoNeuron *)neuron, input);
            break;
        case FLIF_GL:
            update_FLIF_GL((FLIFGLNeuron *)neuron, input);
            break;
        default:
            fprintf(stderr, "Neuron type unavailable.\n");
    }
}

double get_neuron_state(void *neuron, enum NeuronType type) {
    double val; 
    switch(type) {
        case LIF:
            val = ((LIFNeuron *)neuron)->V;
            break;
        case FLIF:
            val = ((FLIFNeuron *)neuron)->V;
            break;
        case FLIF_Caputo:
            val = ((FLIFCaputoNeuron *)neuron)->V;
            break;
        case FLIF_GL:
            val = ((FLIFGLNeuron *)neuron)->V;
            break;
        default:
            fprintf(stderr, "Neuron type unavailable.\n");
    }
    return val;
}

double get_neuron_spike(void *neuron, enum NeuronType type) {
    double spike; 
    switch(type) {
        case LIF:
            spike = ((LIFNeuron*)neuron)->spike;
            break;
        case FLIF:
            spike = ((FLIFNeuron*)neuron)->spike;
            break;
        case FLIF_Caputo:
            spike = ((FLIFCaputoNeuron *)neuron)->spike;
            break;
        case FLIF_GL:
            spike = ((FLIFGLNeuron *)neuron)->spike;
            break;
        default:
            fprintf(stderr, "Neuron type unavailable\n");
    }
    return spike;
}

void free_neuron(void *neuron, enum NeuronType type) {
    if (!neuron) { return; }
    switch(type) {
        case LIF:
            free_LIF((LIFNeuron*)neuron);
            break;
        case FLIF:
            free_FLIF((FLIFNeuron*)neuron);
            break;
        case FLIF_Caputo:
            free_FLIF_Caputo((FLIFCaputoNeuron*)neuron);
            break;
        case FLIF_GL:
            free_FLIF_GL((FLIFGLNeuron*)neuron);
            break;
    }
    neuron = NULL;
}

