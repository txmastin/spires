#ifndef RESERVOIR_H
#define RESERVOIR_H

#include "neuron.h"

enum ConnectivityType {
    RANDOM,
    SMALL_WORLD,
    SCALE_FREE
};

struct Reservoir {
    void **neurons; 
    size_t num_neurons;
    size_t num_inputs;     // for now, default to num_neurons
    size_t num_outputs;    // ditto here
    double spectral_radius;
    double ei_ratio;
    double input_strength;
    double connectivity;
    double dt;
    double *W_in;
    double *W_out;
    double *W;
    enum ConnectivityType connectivity_type;
    enum NeuronType neuron_type;
    double *neuron_params;
};

struct Reservoir* create_reservoir(
    size_t num_neurons, size_t num_inputs, size_t num_outputs,
    double spectral_radius, double ei_ratio, double input_strength, double connectivity, double dt,
    enum ConnectivityType connectivity_type, enum NeuronType neuron_type, double *neuron_params);

double compute_output(struct Reservoir *reservoir);
double compute_activity(struct Reservoir *reservoir);
void step_reservoir(struct Reservoir *reservoir, double input);
double *run_reservoir(struct Reservoir *reservoir, double *input_series, size_t input_length);
void free_reservoir(struct Reservoir *reservoir);
int init_weights(struct Reservoir *reservoir);
int randomize_output_layer(struct Reservoir *reservoir);
int rescale_weights(struct Reservoir *reservoir);
int init_reservoir(struct Reservoir *reservoir);
double *read_reservoir_state(struct Reservoir *reservoir);
void train_output_iteratively(struct Reservoir *reservoir, double target, double lr);
void train_output_ridge_regression(struct Reservoir *reservoir, double *input_series, double *target_series, size_t series_length, double lambda);
void reset_reservoir(struct Reservoir *reservoir);
#endif // RESERVOIR_H

