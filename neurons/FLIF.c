#include <stdlib.h>
#include <stdio.h>

#include "FLIF.h"

FLIFNeuron* init_FLIF(double *neuron_params) {
    FLIFNeuron *neuron = (FLIFNeuron *)malloc(sizeof(FLIFNeuron));
    if(NULL == neuron) { 
        fprintf(stderr, "Error allocating memory for FLIFNeuron\n"); 
        return NULL;
    } 
    neuron->V = neuron_params[0];
    neuron->V_th = neuron_params[1];
    neuron->V_0 = neuron_params[2];
    neuron->leak_rate = neuron_params[3];
    neuron->alpha = neuron_params[4];
    neuron->spike = 0;
    
    return neuron;
}

void update_FLIF(FLIFNeuron *neuron, double inputs){
    
}

void free_FLIF(FLIFNeuron *neuron) {
    if (neuron) { 
        free(neuron);
        neuron = NULL;
    }
} 
