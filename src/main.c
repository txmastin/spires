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

void generate_mackey_glass(double *buffer, size_t length, double x0, double tau, double beta, double gamma, int n) {
    double mg_dt = 1.0;
    // Calculate the required length of the history buffer in discrete steps
    int history_len = (int)ceil(tau / mg_dt);

    // Allocate and initialize the history buffer
    double *history = (double *)malloc(history_len * sizeof(double));
    if (history == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for Mackey-Glass history.\n");
        return;
    }
    for (int i = 0; i < history_len; i++) {
        history[i] = x0;
    }

    // Set the first point of the output buffer
    buffer[0] = x0;
    double x_t = x0; // Current value of x

    // Main simulation loop to generate the series with 4th order runge-kutta method
    for (size_t i = 0; i < length - 1; i++) {
        // Get the delayed value x(t - tau) from the history buffer
        double x_tau = history[i % history_len];

        // k1: slope at the beginning of the interval
        double k1 = mg_dt * (beta * x_tau / (1.0 + pow(x_tau, n)) - gamma * x_t);

        // k2: slope at the midpoint, using k1
        x_tau = history[(i + history_len / 2) % history_len]; // Approx. delay for midpoint
        double k2 = mg_dt * (beta * x_tau / (1.0 + pow(x_tau, n)) - gamma * (x_t + 0.5 * k1));

        // k3: slope at the midpoint, using k2
        double k3 = mg_dt * (beta * x_tau / (1.0 + pow(x_tau, n)) - gamma * (x_t + 0.5 * k2));

        // k4: slope at the end of the interval, using k3
        x_tau = history[(i + 1) % history_len]; // Approx. delay for endpoint
        double k4 = mg_dt * (beta * x_tau / (1.0 + pow(x_tau, n)) - gamma * (x_t + k3));

        // Update the current value using the weighted average of the slopes
        x_t += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        // Store the new value in the history buffer (for future delayed lookups)
        history[(i + 1) % history_len] = x_t;

        // Store the new value in the output buffer
        buffer[i + 1] = x_t;
    }

    // Cleanup
    free(history);
}

int main(void) {
    srand(time(NULL));
    FILE *output_file = fopen("data/output_signals.dat", "w");

    if (output_file == NULL) {
        fprintf(stderr, "Error: Could not open data files for writing.\n");
        return 1;
    }
    
    /*
    // Generate input sine wave
    size_t timesteps = 50;
    double freq = 0.05;
    double sample_rate = 1.0;  
    double *input_series = malloc(timesteps * sizeof(double);
    double noise_gain = 3.0;

    generate_noisy_sine_wave(input_series, timesteps, freq, sample_rate, noise_gain);
    */

    // Generate chaotic mackey-glass signal
    // Parameters for chaotic behavior
    size_t timesteps = 100;
    double x0 = 0.1;
    double tau = 20; // Must be > 17 for chaos
    double beta = 0.2;
    double gamma = 0.1;
    int n = 10;

    // Allocate buffer for the output
    double *input_series = (double *)malloc(timesteps * sizeof(double));
    if (input_series == NULL) {
        return 1;
    }

    // Generate the series
    generate_mackey_glass(input_series, timesteps, x0, tau, beta, gamma, n);

    double *target_series = input_series;
    double series_length = timesteps;
    double lambda = 0.01;
    
    // reservoir parameters 
    size_t num_neurons = 200;
    size_t num_inputs = 200;
    size_t num_outputs = 200;
    double rho = 0.9;
    double ei_ratio = 0.8;
    double input_strength = 1.0;
    double connectivity = 0.1;
    double dt = 0.01;
    enum NeuronType neuron_type = FLIF_GL;
    enum ConnectivityType connectivity_type = RANDOM;
 
    // neuron parameters
    double fractional_neuron_params[] = {
        1.0,    // params[0]: V_th
        0.0,    // params[1]: V_reset
        0.0,    // params[2]: V_rest
        20.0,   // params[3]: tau_m
        0.4,    // params[4]: alpha
        dt,     // params[5]: dt
        timesteps,   // params[6]: Tmem
        0.25     // params[7]: bias
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
    init_reservoir(res); 
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


