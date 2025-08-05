#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "reservoir.h" // Include your full reservoir library header

/**
 * @brief Runs an integration test for a single neuron model within the reservoir.
 * @param neuron_params The parameters for the neuron model.
 * @param input_current The constant input current to apply.
 * @param duration The total duration of the simulation in abstract time units.
 */
void run_single_neuron_integration_test(double* neuron_params, double input_current, double duration) {
    printf("\n--- Running Integration Test ---\n");
    printf("Input Current = %f\n",  input_current);

    // ========================================================================
    // 1. CONFIGURE RESERVOIR FOR A SINGLE NEURON TEST
    // ========================================================================
    size_t num_neurons = 1;
    double dt = neuron_params[5]; // dt must match the neuron's internal dt

    // These parameters are disabled or set to neutral values for a single neuron test
    double spectral_radius = 0.0; // No recurrent connections
    double connectivity = 0.0;    // No recurrent connections
    double input_strength = 1.0;  // Pass input through without scaling

    // Create the reservoir with our single FLIF_GL neuron
    struct Reservoir *res = create_reservoir(
        num_neurons,
        num_neurons, // num_inputs
        num_neurons, // num_outputs
        spectral_radius,
        0.8, // ei_ratio (not relevant for single neuron)
        input_strength,
        connectivity,
        dt,
        RANDOM,      // connectivity_type (not relevant)
        FLIF_GL,     // The neuron type we are testing
        neuron_params
    );

    if (!res) {
        printf("Reservoir initialization failed.\n");
        return;
    }

    // Initialize weights. W will be all zeros, W_in will be random.
    // We will manually set W_in[0] for a controlled test.
    init_weights(res);
    res->W_in[0] = 1.0; // Ensure input current is passed with a weight of 1

    // ========================================================================
    // 2. RUN THE SIMULATION
    // ========================================================================
    int num_steps = (int)(duration / dt);
    double first_spike_time = -1.0;

    for (int i = 0; i < num_steps; i++) {
        // The step_reservoir function handles the micro-stepping internally
        step_reservoir(res, input_current);

        // Check for a spike
        if (get_neuron_spike(res->neurons[0], res->neuron_type) == 1.0 && first_spike_time < 0) {
            first_spike_time = i * dt;
        }
    }

    // ========================================================================
    // 3. PRINT RESULTS AND CLEAN UP
    // ========================================================================
    if (first_spike_time >= 0) {
        printf("Result: First spike at t = %.2f\n", first_spike_time);
    } else {
        printf("Result: Neuron did not spike.\n");
    }

    free_reservoir(res);
}


int test_neuron(void) {
    printf("=====================================================\n");
    printf("  Integration Test for FLIF_GL in Reservoir System   \n");
    printf("=====================================================\n");

    // Define the base parameters for our FLIF_GL neuron
    double base_params[] = {
        1.0,    // V_th
        0.0,    // V_reset
        0.0,    // V_rest
        20.0,   // tau_m
        0.9,    // alpha
        1.0,    // dt (using 1.0 for easy time interpretation)
        200.0  // Tmem
    };
    double input_current = 0.1;
    double duration = 500.0;

    // --- TEST 1: No Bias ---
    // This should exactly match the standalone test result for no bias.
    run_single_neuron_integration_test(base_params, input_current, duration);

    return 0;
}

