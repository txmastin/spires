#ifndef RESERVOIR_H
#define RESERVOIR_H

#include "neuron.h"

enum ConnectivityType {
    DENSE,
    SPARSE,
    SMALL_WORLD,
    SCALE_FREE
};

struct Reservoir {
    void **neurons; 
    int num_neurons;
    int num_inputs;
    int num_outputs; 
    double spectral_radius;
    double input_strength;
    double connectivity;
    double *W_in;
    double *W_out;
    double *W;
    enum ConnectivityType connectivity_type;
    enum NeuronType neuron_type;
};

struct Reservoir* create_reservoir(
    int num_neurons, int num_inputs, int num_outputs,
    double spectral_radius, double input_strength, double connectivity, 
    enum ConnectivityType connectivity_type, enum NeuronType neuron_type);

void update_reservoir(struct Reservoir *reservoir, double *inputs);
void free_reservoir(struct Reservoir *reservoir);
int init_weights(struct Reservoir *reservoir);

#endif // RESERVOIR_H

