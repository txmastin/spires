#include <stdlib.h>
#include <stdio.h>
#include "lif_discrete.h"

struct lif_discrete_neuron *init_lif_discrete(double *neuron_params) 
{
    struct lif_discrete_neuron *neuron = (struct lif_discrete_neuron *)malloc(sizeof(struct lif_discrete_neuron));
    if(NULL == neuron) { 
        fprintf(stderr, "Error allocating memory for struct lif_discrete_neuron\n"); 
        return NULL;
    } 
    neuron->V_0 = neuron_params[0];
    neuron->V_th = neuron_params[1];
    neuron->leak_rate = neuron_params[2];
    neuron->bias = neuron_params[3];
    neuron->V = neuron->V_0;
    neuron->spike = 0.0;
    
    return neuron;
}

void update_lif_discrete(struct lif_discrete_neuron *neuron, double input){
    // if the neuron is already spiking, reset it
    if (neuron->spike == 1.0) { neuron->spike = 0.0; }

    // subtract the 'leak' and integrate the input
    neuron->V = (1.0-neuron->leak_rate) * neuron->V + input + neuron->bias;

    // check if the neuron spikes
    if (neuron->V > neuron->V_th) {
        neuron->V = neuron->V_0;
        neuron->spike = 1.0; 
    }
}

void free_lif_discrete(struct lif_discrete_neuron *neuron) {
    if (neuron) { 
        free(neuron);
        neuron = NULL;
    }
} 
