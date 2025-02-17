#include <stdlib.h>
#include <stdio.h>

#include "FLIF.h"

FLIFNeuron* init_FLIF(double V, double V_th, double V_0, double leak_rate, double alpha) {
    FLIFNeuron *neuron = (FLIFNeuron *)malloc(sizeof(FLIFNeuron));
    if(NULL == neuron) { 
        fprintf(stderr, "Error allocating memory for FLIFNeuron\n"); 
        return NULL;
    } 
    neuron->V = V;
    neuron->V_th = V_th;
    neuron->V_0 = V_0;
    neuron->leak_rate = leak_rate;
    neuron->alpha = alpha;
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
