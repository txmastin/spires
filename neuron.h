#ifndef NEURON_H
#define NEURON_H

#include "neurons/LIF.h"
#include "neurons/FLIF.h"

typedef enum {
    LIF,
    FLIF
} NeuronType;

void* init_neuron(NeuronType neuron_type);
void update_neuron(void *neuron, double *inputs);
void free_neuron(void **neuron, NeuronType type);

#endif // NEURON_H

