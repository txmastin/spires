#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "reservoir.h"

// Generate a sine wave input
void generate_sine_wave(double *buffer, size_t length, double freq, double sample_rate) {
    for (size_t i = 0; i < length; i++) {
        buffer[i] = sin(2.0 * M_PI * freq * i / sample_rate);
    }
}

int main(void) {
    // reservoir parameters 
    size_t num_neurons = 32;
    size_t num_inputs = 32;
    size_t num_outputs = 32;
    double lr = 0.001;
    double rho = 0.9;
    double ei_ratio = 0.8;
    double input_strength = 0.1;
    double connectivity = 0.2;
    double dt = 0.01;  
    enum NeuronType neuron_type = FLIF_GL;
    enum ConnectivityType connectivity_type = RANDOM;
  
    double neuron_params[10] = {
    1.0,    // params[0]: Cm - Membrane capacitance (e.g., 1.0 nF)
    0.05,   // params[1]: gl - Leak conductance (results in a 20ms time constant)
    -65.0,  // params[2]: Vl - Resting/leak potential (mV)
    -50.0,  // params[3]: V_th - Firing threshold (mV)
    -65.0,  // params[4]: V_0 / Vreset - Reset potential (mV)
    30.0,   // params[5]: Vpeak - Spike peak for visualization (mV)
    0.85,   // params[6]: alpha - Fractional order (0 < alpha <= 1)
    3.0,    // params[7]: tref - Refractory period (ms)
    150.0,  // params[8]: Tmem - Memory duration for fractional derivative (ms)
    dt     // params[9]: dt - Simulation time step (ms)
    };

    // neuron parameters
    // Create reservoir
    struct Reservoir *res = create_reservoir(num_neurons, num_inputs, num_outputs, lr, rho, ei_ratio, input_strength, connectivity, dt, connectivity_type, neuron_type, neuron_params);
    init_weights(res);
    rescale_weights(res);
    randomize_output_layer(res);

    // Generate input sine wave
    size_t timesteps = 100;
    double freq = 0.05;
    double sample_rate = 1.0;  // matches dt
    double input_series[timesteps];

    generate_sine_wave(input_series, timesteps, freq, sample_rate);

    size_t num_epochs = 1000;
    double *acc_trace = malloc(num_epochs * sizeof(double));

    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        double avg_err = 0.0;
        for (size_t i = 0; i < timesteps; i++) {
            step_reservoir(res, input_series[i]);
            train_output_iteratively(res, input_series[i]);
            double reservoir_output = compute_output(res);
            double error = input_series[i] - reservoir_output;
            avg_err += fabs(error);
        }
        avg_err /= timesteps;
        acc_trace[epoch] = avg_err;
        printf("%f\n", avg_err);
    }

    free(acc_trace);


    // Cleanup
    free_reservoir(res);
    return 0;
}
