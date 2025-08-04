#include<stdio.h>
#include<stdlib.h>
#include "neuron.h"

void* init_neuron(enum NeuronType type, double *neuron_params) {
    void* neuron = NULL;
    switch(type) {
        case LIF_DISCRETE: 
            neuron = (LIFDiscreteNeuron *)init_LIF_Discrete(neuron_params);
            break;
        case LIF_BIO: 
            neuron = (LIFBioNeuron *)init_LIF_Bio(neuron_params);
            break;
        case FLIF:
            neuron = (FLIFNeuron *)init_FLIF(neuron_params);
            break;
        case FLIF_CAPUTO:
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

void update_neuron(void *neuron, enum NeuronType type, double input, double dt) {
    // update neuron based on type  
    switch(type) {
        case LIF_DISCRETE:
            update_LIF_Discrete((LIFDiscreteNeuron *)neuron, input);
            break;
        case LIF_BIO:
            update_LIF_Bio((LIFBioNeuron *)neuron, input, dt);
            break;
        case FLIF:
            update_FLIF((FLIFNeuron *)neuron, input, dt);
            break;
        case FLIF_CAPUTO:
            update_FLIF_Caputo((FLIFCaputoNeuron *)neuron, input, dt);
            break;
        case FLIF_GL:
            update_FLIF_GL((FLIFGLNeuron *)neuron, input, dt);
            break;
        default:
            fprintf(stderr, "Neuron type unavailable.\n");
    }
}

double get_neuron_state(void *neuron, enum NeuronType type) {
    double val; 
    switch(type) {
        case LIF_DISCRETE:
            val = ((LIFDiscreteNeuron *)neuron)->V;
            break;
        case LIF_BIO:
            val = ((LIFBioNeuron *)neuron)->V;
            break;
        case FLIF:
            val = ((FLIFNeuron *)neuron)->V;
            break;
        case FLIF_CAPUTO:
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
        case LIF_DISCRETE:
            spike = ((LIFDiscreteNeuron*)neuron)->spike;
            break;
        case LIF_BIO:
            spike = ((LIFBioNeuron*)neuron)->spike;
            break;
        case FLIF:
            spike = ((FLIFNeuron*)neuron)->spike;
            break;
        case FLIF_CAPUTO:
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
        case LIF_DISCRETE:
            free_LIF_Discrete((LIFDiscreteNeuron*)neuron);
            break;
        case LIF_BIO:
            free_LIF_Bio((LIFBioNeuron*)neuron);
            break;
        case FLIF:
            free_FLIF((FLIFNeuron*)neuron);
            break;
        case FLIF_CAPUTO:
            free_FLIF_Caputo((FLIFCaputoNeuron*)neuron);
            break;
        case FLIF_GL:
            free_FLIF_GL((FLIFGLNeuron*)neuron);
            break;
    }
    neuron = NULL;
}

