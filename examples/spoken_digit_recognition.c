#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "reservoir.h"

// --- Data Simulation ---

#define NUM_SAMPLES_TRAIN 50
#define NUM_SAMPLES_TEST 10
#define NUM_CLASSES 10       // Digits 0-9
#define NUM_FEATURES 13      // Example: 13 MFCCs per time slice
#define SEQUENCE_LENGTH 50   // Each audio clip has 50 time steps

/**
 * @brief Generates a mock dataset for training and testing.
 * @param[out] train_x Pointer to store the training input features.
 * @param[out] train_y Pointer to store the one-hot encoded training labels.
 * @param[out] test_x  Pointer to store the testing input features.
 * @param[out] test_y  Pointer to store the actual integer labels for testing.
 */
void generate_mock_dataset(double **train_x, double **train_y, double **test_x, int **test_y) {
    printf("Generating mock dataset...\n");

    // Allocate memory for the flattened 2D arrays
    *train_x = malloc(NUM_SAMPLES_TRAIN * SEQUENCE_LENGTH * NUM_FEATURES * sizeof(double));
    *train_y = malloc(NUM_SAMPLES_TRAIN * SEQUENCE_LENGTH * NUM_CLASSES * sizeof(double));
    *test_x = malloc(NUM_SAMPLES_TEST * SEQUENCE_LENGTH * NUM_FEATURES * sizeof(double));
    *test_y = malloc(NUM_SAMPLES_TEST * sizeof(int));

    // Generate training data
    for (int i = 0; i < NUM_SAMPLES_TRAIN; i++) {
        int digit = rand() % NUM_CLASSES; // The correct digit for this sample
        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            // Create a feature vector (e.g., noisy sine waves centered around the digit)
            for (int f = 0; f < NUM_FEATURES; f++) {
                (*train_x)[i * SEQUENCE_LENGTH * NUM_FEATURES + t * NUM_FEATURES + f] =
                    sin((double)t / 5.0 + f) + ((double)rand() / RAND_MAX - 0.5) * 0.5 + digit;
            }
            // Create the one-hot encoded target vector
            for (int c = 0; c < NUM_CLASSES; c++) {
                (*train_y)[i * SEQUENCE_LENGTH * NUM_CLASSES + t * NUM_CLASSES + c] = (c == digit) ? 1.0 : 0.0;
            }
        }
    }

    // Generate testing data
    for (int i = 0; i < NUM_SAMPLES_TEST; i++) {
        int digit = rand() % NUM_CLASSES;
        (*test_y)[i] = digit;
        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            for (int f = 0; f < NUM_FEATURES; f++) {
                 (*test_x)[i * SEQUENCE_LENGTH * NUM_FEATURES + t * NUM_FEATURES + f] =
                    sin((double)t / 5.0 + f) + ((double)rand() / RAND_MAX - 0.5) * 0.5 + digit;
            }
        }
    }
}

/**
 * @brief Finds the index of the maximum value in an array.
 * @param array The array to search.
 * @param size The size of the array.
 * @return The index of the maximum value.
 */
int argmax(const double *array, size_t size) {
    if (size == 0) return -1;
    int max_idx = 0;
    for (size_t i = 1; i < size; i++) {
        if (array[i] > array[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

int main(void) {
    srand(time(NULL));

    // --- 1. Load Data ---
    double *train_x, *train_y;
    double *test_x;
    int *test_y;
    generate_mock_dataset(&train_x, &train_y, &test_x, &test_y);

    // --- 2. Setup Reservoir Parameters ---
    printf("Configuring reservoir...\n");
    size_t num_neurons = 1000;
    double spectral_radius = 0.99;
    double connectivity = 0.2;
    double input_strength = 1.0;
    double dt = 0.1;
    double lambda = 0.01; // Ridge regression regularization

    // neuron parameters
    double fractional_neuron_params[] = {
        1.0,    // params[0]: V_th
        0.0,    // params[1]: V_reset
        0.0,    // params[2]: V_rest
        20.0,   // params[3]: tau_m
        0.3,    // params[4]: alpha
        dt,     // params[5]: dt
        SEQUENCE_LENGTH,   // params[6]: Tmem
        0.3     // params[7]: bias
    };

    double discrete_neuron_params[] = {
        0.0, // params[0]: V_0
        1.0, // params[1]: V_th
        0.2, // params[2]: leak_rate
        0.05 // params[3]: bias
    };
    enum neuron_type neuron_type = FLIF_GL;
    double *neuron_params = NULL;
    switch(neuron_type) {
        case LIF_DISCRETE:
            neuron_params = discrete_neuron_params;
            break;
        case FLIF_GL:
            neuron_params = fractional_neuron_params;
            break;
        default:
            fprintf(stderr, "Error: Could not initialize neuron parameters, neuron type unavailable.\n");
            break;
    };

    // --- 3. Create and Initialize Reservoir ---
    struct reservoir *res = create_reservoir(
        num_neurons, NUM_FEATURES, NUM_CLASSES,
        spectral_radius, 0.8, input_strength, connectivity, dt,
        RANDOM, neuron_type, neuron_params
    );
    if (!res) { return 1; }
    init_reservoir(res);

    // --- 4. Train the Reservoir ---
    // Note: For simplicity, we train on the entire dataset at once.
    // In a real scenario, you might train on each sample individually.
    printf("Training reservoir...\n");
    train_output_ridge_regression(res, train_x, train_y, NUM_SAMPLES_TRAIN * SEQUENCE_LENGTH, lambda);
    printf("Training complete.\n");

    // --- 5. Test the Reservoir ---
    printf("\n--- Running Inference ---\n");
    int correct_predictions = 0;
    for (int i = 0; i < NUM_SAMPLES_TEST; i++) {
        // Get the input sequence for the current test sample
        const double *current_test_input = &test_x[i * SEQUENCE_LENGTH * NUM_FEATURES];

        // Run the reservoir on this sequence
        // Note: The output is the reservoir's state *after* the final timestep
        double final_output[NUM_CLASSES];
        reset_reservoir(res);
        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            const double *input_slice = &current_test_input[t * NUM_FEATURES];
            step_reservoir(res, input_slice);
        }
        // Get the output only after the full sequence has been processed
        compute_output(res, final_output);

        // Get the predicted class
        int predicted_digit = argmax(final_output, NUM_CLASSES);
        int actual_digit = test_y[i];

        printf("Sample %d: Predicted = %d, Actual = %d ", i, predicted_digit, actual_digit);
        if (predicted_digit == actual_digit) {
            printf("(Correct)\n");
            correct_predictions++;
        } else {
            printf("(Incorrect)\n");
        }
    }

    // --- 6. Print Final Results and Cleanup ---
    double accuracy = (double)correct_predictions / NUM_SAMPLES_TEST * 100.0;
    printf("\nFinal Accuracy: %.2f%%\n", accuracy);

    free(train_x);
    free(train_y);
    free(test_x);
    free(test_y);
    free_reservoir(res);

    return 0;
}

