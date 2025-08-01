#ifndef NEURON_H
#define NEURON_H

#include "neurons/LIF_Discrete.h"
#include "neurons/LIF_Bio.h"
#include "neurons/FLIF.h"
#include "neurons/FLIF_Caputo.h"
#include "neurons/FLIF_GL.h"


enum NeuronType {
    LIF_DISCRETE,
    LIF_BIO,
    FLIF,
    FLIF_CAPUTO,
    FLIF_GL
};

void* init_neuron(enum NeuronType type, double *neuron_params);
void update_neuron(void *neuron, enum NeuronType type, double input, double dt);
void free_neuron(void *neuron, enum NeuronType type);
double get_neuron_state(void *neuron, enum NeuronType type);
double get_neuron_spike(void *neuron, enum NeuronType type);

#endif // NEURON_H

