#include <stdlib.h>
#include "LIF_Bio.h"

LIFBioNeuron* init_LIF_Bio(double *neuron_params) {
    LIFBioNeuron *n = malloc(sizeof(LIFBioNeuron));
    n->V_0  = neuron_params[0];  // reset voltage 
    n->V_th   = neuron_params[1];  // threshold
    n->tau   = neuron_params[2];  // membrane time constant
    n->bias  = neuron_params[3];  // constant bias
    n->V = n->V_0; // initial voltage set to reset voltage
    n->spike = 0.0;
    return n;
}

void update_LIF_Bio(LIFBioNeuron *n, double input, double dt) {
    if (n->spike == 1.0) {
        n->spike = 0.0;
    }

    double dV = (-n->V + input + n->bias) / n->tau;
    n->V += dV * dt;

    if (n->V >= n->V_th) {
        n->V = n->V_0;
        n->spike = 1.0;
    }
}

void free_LIF_Bio(LIFBioNeuron *n) {
    free(n);
}

