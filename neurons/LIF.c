#include <stdlib.h>
#include <stdio.h>

#include "LIF.h"

LIFNeuron* init_LIF(double V, double V_th, double V_0, double leak_rate) {
    LIFNeuron *neuron = (LIFNeuron *)malloc(sizeof(LIFNeuron));
    if(NULL == neuron) { 
        fprintf(stderr, "Error allocating memory for LIFNeuron\n"); 
        return NULL;
    } 
    neuron->V = V;
    neuron->V_th = V_th;
    neuron->V_0 = V_0;
    neuron->leak_rate = leak_rate;
    neuron->spike = 0.0;
    
    return neuron;
}

void update_LIF(LIFNeuron *neuron, double input){
    // if the neuron is already spiking, reset it
    if (neuron->spike == 1.0) { neuron->spike = 0.0; }

    // subtract the 'leak' and integrate the input
    neuron->V = (1.0-neuron->leak_rate) * neuron->V + input;

    // check if the neuron spikes
    if (neuron->V > neuron->V_th) {
        neuron->V = neuron->V_0;
        neuron->spike = 1.0; 
    }
}

void free_LIF(LIFNeuron *neuron) {
    if (neuron) { 
        free(neuron);
        neuron = NULL;
    }
} 
