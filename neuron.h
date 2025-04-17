#ifndef NEURON_H
#define NEURON_H

#include "neurons/LIF.h"
#include "neurons/FLIF.h"
#include "neurons/FLIF_Caputo.h"
#include "neurons/FLIF_GL.h"


enum NeuronType {
    LIF,
    FLIF,
    FLIF_Caputo,
    FLIF_GL
};

void* init_neuron(enum NeuronType type, double *neuron_params);
void update_neuron(void *neuron, enum NeuronType type, double input);
void free_neuron(void *neuron, enum NeuronType type);
double get_neuron_state(void *neuron, enum NeuronType type);
double get_neuron_spike(void *neuron, enum NeuronType type);

#endif // NEURON_H

