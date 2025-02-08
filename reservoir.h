#ifndef RESERVOIR_H
#define RESERVOIR_H

#include "neuron.h"

typedef enum {
    DENSE,
    SPARSE,
    SMALL_WORLD,
    SCALE_FREE
} ConnectivityType;

typedef struct {
    void **neurons; // ptr to void ptr used to dynamically change neuron type
    int num_neurons;
    int num_inputs;
    int num_outputs; 
    double spectral_radius;
    double input_strength;
    double connectivity;
    double *W_in;
    double *W_out;
    double *W;
    ConnectivityType connectivity_type;
    NeuronType neuron_type;
} Reservoir;

Reservoir* create_reservoir(int num_neurons, int num_inputs, int num_outputs, double spectral_radius, double input_strength, double connectivity, ConnectivityType connectivity_type, NeuronType neuron_type);
void update_reservoir(Reservoir *reservoir, double *inputs);
void free_reservoir(Reservoir *reservoir);
void init_weights(Reservoir *reservoir);

#endif // RESERVOIR_H

