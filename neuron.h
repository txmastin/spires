#ifndef NEURON_H
#define NEURON_H

#include "neurons/LIF.h"
#include "neurons/FLIF.h"

enum NeuronType {
    LIF,
    FLIF
};

void* init_neuron(enum NeuronType neuron_type);
void update_neuron(void *neuron, double *inputs);
void free_neuron(void *neuron, enum NeuronType type);

#endif // NEURON_H

