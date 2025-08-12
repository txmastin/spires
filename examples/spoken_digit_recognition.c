#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "reservoir.h"

// --- Data Loading ---

#define NUM_SAMPLES_TRAIN 2500
#define NUM_SAMPLES_TEST 500
#define NUM_SAMPLES 3000     // Total samples in FSDD
#define NUM_CLASSES 10       // Digits 0-9
#define NUM_FEATURES 13      // Must match N_MFCC in Python script
#define SEQUENCE_LENGTH 25   // Must match MAX_LEN in Python script

/**
 * @brief Loads the preprocessed features and labels from text files.
 * @param[out] all_features Pointer to store the flattened feature data.
 * @param[out] all_labels   Pointer to store the integer labels.
 * @return 0 on success, -1 on failure.
 */
int load_preprocessed_data(double **all_features, int **all_labels) {
    const char* features_path = "/home/xenos/programming/datasets/free-spoken-digit-dataset/data/spoken_digit_features.txt";
    const char* labels_path = "/home/xenos/programming/datasets/free-spoken-digit-dataset/data/spoken_digit_labels.txt";

    
    printf("Loading data from text files...\n");

    // Allocate memory
    *all_features = malloc(NUM_SAMPLES * SEQUENCE_LENGTH * NUM_FEATURES * sizeof(double));
    *all_labels = malloc(NUM_SAMPLES * sizeof(int));
    if (!*all_features || !*all_labels) {
        fprintf(stderr, "Error: Memory allocation failed for dataset.\n");
        return -1;
    }

    // Open and read features file
    FILE *f_features = fopen(features_path, "r");
    if (!f_features) {
        fprintf(stderr, "Error: Could not open features file: %s\n", features_path);
        return -1;
    }
    for (int i = 0; i < NUM_SAMPLES * SEQUENCE_LENGTH * NUM_FEATURES; i++) {
        if (fscanf(f_features, "%lf", &(*all_features)[i]) != 1) {
            fprintf(stderr, "Error reading features file.\n");
            fclose(f_features);
            return -1;
        }
    }
    fclose(f_features);

    // Open and read labels file
    FILE *f_labels = fopen(labels_path, "r");
    if (!f_labels) {
        fprintf(stderr, "Error: Could not open labels file: %s\n", labels_path);
        return -1;
    }
    for (int i = 0; i < NUM_SAMPLES; i++) {
        if (fscanf(f_labels, "%d", &(*all_labels)[i]) != 1) {
            fprintf(stderr, "Error reading labels file.\n");
            fclose(f_labels);
            return -1;
        }
    }
    fclose(f_labels);

    printf("Data loading complete.\n");
    return 0;
}
// In spoken_digit_recognition.c

void split_dataset(
    double *all_features, int *all_labels,
    double **train_x, double **train_y,
    double **test_x, int **test_y)
{
    size_t feature_vec_size = SEQUENCE_LENGTH * NUM_FEATURES;
    size_t target_vec_size = SEQUENCE_LENGTH * NUM_CLASSES; // Size for one-hot targets

    // Allocate train/test arrays
    *train_x = malloc(NUM_SAMPLES_TRAIN * feature_vec_size * sizeof(double));
    *train_y = malloc(NUM_SAMPLES_TRAIN * target_vec_size * sizeof(double)); // <-- CORRECTED SIZE
    *test_x  = malloc(NUM_SAMPLES_TEST * feature_vec_size * sizeof(double));
    *test_y  = malloc(NUM_SAMPLES_TEST * sizeof(int));

    if (!*train_x || !*train_y || !*test_x || !*test_y) {
        fprintf(stderr, "Error allocating train/test arrays.\n");
        exit(1);
    }

    // Create and shuffle indices (your code is perfect here)
    int indices[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++) indices[i] = i;
    srand((unsigned)time(NULL));
    for (int i = NUM_SAMPLES - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // --- Fill train set ---
    for (int i = 0; i < NUM_SAMPLES_TRAIN; i++) {
        int idx = indices[i];
        // Copy the feature data (your code is perfect here)
        memcpy(&(*train_x)[i * feature_vec_size],
               &all_features[idx * feature_vec_size],
               feature_vec_size * sizeof(double));
        
        // --- NEW: Create the one-hot encoded target vectors ---
        int correct_digit = all_labels[idx];
        // For every timestep in this sample...
        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            // ...create a one-hot vector.
            for (int c = 0; c < NUM_CLASSES; c++) {
                (*train_y)[i * target_vec_size + t * NUM_CLASSES + c] = (c == correct_digit) ? 1.0 : 0.0;
            }
        }
    }

    // --- Fill test set (your code is perfect here) ---
    for (int i = 0; i < NUM_SAMPLES_TEST; i++) {
        int idx = indices[NUM_SAMPLES_TRAIN + i];
        memcpy(&(*test_x)[i * feature_vec_size],
               &all_features[idx * feature_vec_size],
               feature_vec_size * sizeof(double));
        (*test_y)[i] = all_labels[idx];
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

    double *all_features;
    int *all_labels;
    
    if (load_preprocessed_data(&all_features, &all_labels) != 0) {
        return 1;
    }

    double *train_x, *train_y;
    double *test_x;
    int *test_y;

    split_dataset(all_features, all_labels,
                  &train_x, &train_y,
                  &test_x, &test_y);

    // --- 2. Setup Reservoir Parameters ---
    printf("Configuring reservoir...\n");
    size_t num_neurons = 500;
    double spectral_radius = 0.99;
    double connectivity = 0.17;
    double input_strength = 1.0;
    double dt = 0.1;
    double lambda = 0.01; // Ridge regression regularization

    // neuron parameters
    double fractional_neuron_params[] = {
        1.0,    // params[0]: V_th
        0.0,    // params[1]: V_reset
        0.0,    // params[2]: V_rest
        20.0,   // params[3]: tau_m
        0.2,    // params[4]: alpha
        dt,     // params[5]: dt
        SEQUENCE_LENGTH,   // params[6]: Tmem
        0.35     // params[7]: bias
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

