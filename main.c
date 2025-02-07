#include "reservoir.c"
#include "neuron.c"
#include <time.h>

int main() {
    srand(time(NULL)); // Seed for random weights

    int num_neurons = 10;
    int num_inputs = 5;
    Reservoir reservoir = create_reservoir(num_neurons, num_inputs);

    // Dummy inputs for testing
    double inputs[5] = {1.0, 1.0, 1.0, 1.0, 1.0};

    // Simulate a few timesteps
    for (int t = 0; t < 10; t++) {
        printf("Timestep %d:\n", t);
        update_reservoir(&reservoir, inputs);
    }

    // Clean up
    free_reservoir(&reservoir);
    return 0;
}

