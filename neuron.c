#include "neuron.h"

// Create a neuron with random weights
Neuron create_neuron(int num_inputs) {
    Neuron neuron;
    neuron.membrane_potential = 0.0;  // Resting potential
    neuron.threshold = 0.9;
    neuron.num_inputs = num_inputs;
    neuron.weights = (double *)malloc(num_inputs * sizeof(double));

    if (!neuron.weights) {
        fprintf(stderr, "Memory allocation failed for weights\n");
        exit(1);
    }

    // Initialize random weights
    for (int i = 0; i < num_inputs; i++) {
        neuron.weights[i] = (double)rand() / RAND_MAX * 2.0 - 1.0; // Random between -1 and 1
    }

    return neuron;
}

// Update neuron state based on inputs
void update_neuron(Neuron *neuron, double *inputs) {
    double weighted_sum = 0.0;
    
    for (int i = 0; i < neuron->num_inputs; i++) {
        weighted_sum += inputs[i] * neuron->weights[i];
    }

    neuron->membrane_potential += weighted_sum;

    if (neuron->membrane_potential >= neuron->threshold) {
        printf("Neuron spiked!\n");
        neuron->membrane_potential = 0.0;  // Reset after spike
    }
}

// Free allocated memory
void free_neuron(Neuron *neuron) {
    free(neuron->weights);
}

