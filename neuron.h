#ifndef NEURON_H
#define NEURON_H

#include <stdlib.h>
#include <stdio.h>

// Neuron structure
typedef struct {
    double membrane_potential;
    double threshold;
    double *weights;  // Synaptic weights
    int num_inputs;
} Neuron;

// Function prototypes
Neuron create_neuron(int num_inputs);
void update_neuron(Neuron *neuron, double *inputs);
void free_neuron(Neuron *neuron);

#endif // NEURON_H

