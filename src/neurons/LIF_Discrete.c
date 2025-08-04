#include <stdlib.h>
#include <stdio.h>

#include "LIF_Discrete.h"

LIFDiscreteNeuron* init_LIF_Discrete(double *neuron_params) { 
    LIFDiscreteNeuron *neuron = (LIFDiscreteNeuron *)malloc(sizeof(LIFDiscreteNeuron));
    if(NULL == neuron) { 
        fprintf(stderr, "Error allocating memory for LIFDiscreteNeuron\n"); 
        return NULL;
    } 
    neuron->V_0 = neuron_params[0];
    neuron->V_th = neuron_params[1];
    neuron->leak_rate = neuron_params[2];
    neuron->V = neuron->V_0;
    neuron->spike = 0.0;
    
    return neuron;
}

void update_LIF_Discrete(LIFDiscreteNeuron *neuron, double input){
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

void free_LIF_Discrete(LIFDiscreteNeuron *neuron) {
    if (neuron) { 
        free(neuron);
        neuron = NULL;
    }
} 
