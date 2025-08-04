#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "reservoir.h"


void generate_noisy_sine_wave(double *buffer, size_t length, double freq, double sample_rate, double noise_gain) {
    for (size_t i = 0; i < length; i++) {
        double clean_signal = sin(2.0 * M_PI * freq * i / sample_rate);
        double noise = (noise_gain * rand() / RAND_MAX) - (noise_gain / 2.0);
        buffer[i] =  clean_signal + noise;
    }
}

int main(void) {
    srand(time(NULL));
    FILE *output_file = fopen("output_signals.dat", "w");

    if (output_file == NULL) {
        fprintf(stderr, "Error: Could not open data files for writing.\n");
        return 1;
    }

    // Generate input sine wave
    size_t timesteps = 50;
    double freq = 0.05;
    double sample_rate = 1.0;  
    double input_series[timesteps];
    double noise_gain = 0.0;

    generate_noisy_sine_wave(input_series, timesteps, freq, sample_rate, noise_gain);

    double *target_series = input_series;
    double series_length = timesteps;
    double lambda = 0.1;
    
    // reservoir parameters 
    size_t num_neurons = 32;
    size_t num_inputs = 32;
    size_t num_outputs = 32;
    double rho = 0.9;
    double ei_ratio = 0.8;
    double input_strength = 1.0;
    double connectivity = 0.2;
    double dt = 0.01;
    enum NeuronType neuron_type = FLIF_GL;
    enum ConnectivityType connectivity_type = RANDOM;
 
    // neuron parameters
    double fractional_neuron_params[] = {
        1.0,    // params[0]: V_th
        0.0,    // params[1]: V_reset
        0.0,    // params[2]: V_rest
        20.0,   // params[3]: tau_m
        0.8,    // params[4]: alpha
        dt,     // params[5]: dt
        timesteps,   // params[6]: Tmem
        0.0
    };

   double discrete_neuron_params[] = {
        0.0, // params[0]: V_0
        1.0, // params[1]: V_th
        0.2, // params[2]: leak_rate
    };
    
    double *neuron_params = NULL;
    switch(neuron_type){
        case LIF_DISCRETE:
            neuron_params = discrete_neuron_params;
            break;
        case FLIF_GL:
            neuron_params = fractional_neuron_params;
            break;
        default:
            fprintf(stderr, "Error: Could not initialize neuron parameters, neuron type unavailable");
            break;
    };
            
    
    // Create reservoir
    struct Reservoir *res = create_reservoir(num_neurons, num_inputs, num_outputs, rho, ei_ratio, input_strength, connectivity, dt, connectivity_type, neuron_type, neuron_params);
    init_weights(res);
    rescale_weights(res);
    randomize_output_layer(res);
    
    // run training and inferencing
    train_output_ridge_regression(res, input_series, target_series, series_length, lambda); 
    reset_reservoir(res); // necessary for fractional neurons 
    double *reservoir_outputs = run_reservoir(res, input_series, series_length); 
    
    // saving final inference output of the reservoir
    fprintf(output_file, "# Timestep Input_Signal Reservoir_Output\n");
    for (size_t i = 0; i < series_length; i++) {
        fprintf(output_file, "%zu %f %f\n", i, target_series[i], reservoir_outputs[i]);
    }

    // Cleanup
    free_reservoir(res);
    fclose(output_file);

    return 0;
}


