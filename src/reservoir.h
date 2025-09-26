#ifndef RESERVOIR_H
#define RESERVOIR_H

#include "neuron.h"

enum connectivity_type {
    RANDOM,
    SMALL_WORLD,
    SCALE_FREE
};

struct reservoir {
    void **neurons; 
    size_t num_neurons;
    size_t num_inputs;      
    size_t num_outputs;     
    double spectral_radius;
    double ei_ratio;
    double input_strength;
    double connectivity;
    double dt;
    double *W_in;
    double *W_out;
    double *W;
    double *shared_neuron_data; // optionally used in some neuron types
    enum connectivity_type connectivity_type;
    enum neuron_type neuron_type;
    double *neuron_params;
};

struct reservoir* create_reservoir(
    size_t num_neurons, size_t num_inputs, size_t num_outputs,
    double spectral_radius, double ei_ratio, double input_strength, double connectivity, double dt,
    enum connectivity_type connectivity_type, enum neuron_type neuron_type, double *neuron_params);

int compute_output(struct reservoir *reservoir, double *output_vector);
double compute_activity(struct reservoir *reservoir);
void step_reservoir(struct reservoir *reservoir, double *input_vector);
double *run_reservoir(struct reservoir *reservoir, double *input_series, size_t input_length); // input_series is a flattened array of input_vector * num_inputs
void free_reservoir(struct reservoir *reservoir);
int init_weights(struct reservoir *reservoir);
int randomize_output_layer(struct reservoir *reservoir);
int rescale_weights(struct reservoir *reservoir);
int init_reservoir(struct reservoir *reservoir);
void read_reservoir_state(struct reservoir *reservoir, double *buffer);
double *copy_reservoir_state(struct reservoir *reservoir);
void train_output_iteratively(struct reservoir *reservoir, double *target_vector, double lr);
void train_output_ridge_regression(struct reservoir *reservoir, double *input_series, 
                                double *target_series, size_t series_length, double lambda);
void reset_reservoir(struct reservoir *reservoir);
#endif // RESERVOIR_H

