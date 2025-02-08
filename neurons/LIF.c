#include <stdlib.h>
#include "LIF.h"

LIFNeuron init_LIF(double V, double V_th, double V_0, double, leak_rate) {
    LIFNeuron neuron;
    neuron.V = V;
    neuron.V_th = V_th;
    neuron.V_0 = V_0;
    neuron.leak_rate = leak_rate;
    neuron.spiking = 0;
    
    return neuron;
}

void update_LIF(LIFNeuron *neuron, double *inputs){
    
}

void free_LIF(LIFNeuron *neuron) {
    free(neuron);
    neuron = NULL;
} 
