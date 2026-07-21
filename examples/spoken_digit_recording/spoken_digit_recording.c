#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cblas.h>
#include <spires.h>
#include "reservoir.h"
#include "neuron.h"
#include "synapse.h"

/*
 * Local mirror of the opaque handle — must match the definition in spires_api.c.
 * Needed to reach impl->W, impl->W_in, and impl->neurons for CSV recording.
 */
struct spires_reservoir {
    struct reservoir *impl;
};

#define NUM_SAMPLES_TRAIN 2500
#define NUM_SAMPLES_TEST 500
#define NUM_SAMPLES 3000
#define NUM_CLASSES 10
#define NUM_FEATURES 13
#define SEQUENCE_LENGTH 25

static inline double bias_from_alpha(double alpha) {
    const double b_inf = 0.057;
    const double A     = 0.986;
    const double k     = 7.083;
    double b = b_inf + A * exp(-k * alpha);
    if (b < b_inf) b = b_inf;
    return b;
}

int load_preprocessed_data(double **all_features, int **all_labels) {
    const char* features_path = "/home/xenos/programming/datasets/free-spoken-digit-dataset/data/spoken_digit_features.txt";
    const char* labels_path   = "/home/xenos/programming/datasets/free-spoken-digit-dataset/data/spoken_digit_labels.txt";

    printf("Loading data from text files...\n");

    *all_features = malloc(NUM_SAMPLES * SEQUENCE_LENGTH * NUM_FEATURES * sizeof(double));
    *all_labels   = malloc(NUM_SAMPLES * sizeof(int));
    if (!*all_features || !*all_labels) {
        fprintf(stderr, "Error: Memory allocation failed for dataset.\n");
        return -1;
    }

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

void split_dataset(
    double *all_features, int *all_labels,
    double **train_x, double **train_y,
    double **test_x, int **test_y)
{
    size_t feature_vec_size = SEQUENCE_LENGTH * NUM_FEATURES;
    size_t target_vec_size  = SEQUENCE_LENGTH * NUM_CLASSES;

    *train_x = malloc(NUM_SAMPLES_TRAIN * feature_vec_size * sizeof(double));
    *train_y = malloc(NUM_SAMPLES_TRAIN * target_vec_size  * sizeof(double));
    *test_x  = malloc(NUM_SAMPLES_TEST  * feature_vec_size * sizeof(double));
    *test_y  = malloc(NUM_SAMPLES_TEST  * sizeof(int));

    if (!*train_x || !*train_y || !*test_x || !*test_y) {
        fprintf(stderr, "Error allocating train/test arrays.\n");
        exit(1);
    }

    int indices[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++) indices[i] = i;
    srand((unsigned)time(NULL));
    for (int i = NUM_SAMPLES - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i]; indices[i] = indices[j]; indices[j] = temp;
    }

    for (int i = 0; i < NUM_SAMPLES_TRAIN; i++) {
        int idx = indices[i];
        memcpy(&(*train_x)[i * feature_vec_size],
               &all_features[idx * feature_vec_size],
               feature_vec_size * sizeof(double));
        int correct_digit = all_labels[idx];
        for (int t = 0; t < SEQUENCE_LENGTH; t++)
            for (int c = 0; c < NUM_CLASSES; c++)
                (*train_y)[i * target_vec_size + t * NUM_CLASSES + c] = (c == correct_digit) ? 1.0 : 0.0;
    }

    for (int i = 0; i < NUM_SAMPLES_TEST; i++) {
        int idx = indices[NUM_SAMPLES_TRAIN + i];
        memcpy(&(*test_x)[i * feature_vec_size],
               &all_features[idx * feature_vec_size],
               feature_vec_size * sizeof(double));
        (*test_y)[i] = all_labels[idx];
    }
}

int argmax(const double *array, size_t size) {
    if (size == 0) return -1;
    int max_idx = 0;
    for (size_t i = 1; i < size; i++)
        if (array[i] > array[max_idx]) max_idx = (int)i;
    return max_idx;
}

/*
 * record_inference_csvs
 *
 * Runs a single forward pass of `sample_input` (shape [SEQUENCE_LENGTH x NUM_FEATURES])
 * through `res`, recording at every internal micro-step (1.0 / dt per macro step).
 *
 * Output CSVs:
 *   spikes.csv              — [(SEQUENCE_LENGTH * micro_steps) rows x num_neurons cols]
 *   membrane_potentials.csv — [(SEQUENCE_LENGTH * micro_steps) rows x num_neurons cols]
 *   adjacency_matrix.csv    — [num_neurons rows x num_neurons cols]
 */
static void record_inference_csvs(spires_reservoir *res,
                                   const double *sample_input,
                                   size_t num_neurons)
{
    struct reservoir *impl = res->impl;
    int num_micro_steps = (int)llround(1.0 / impl->dt);
    int total_rows = SEQUENCE_LENGTH * num_micro_steps;

    double *ext_input   = malloc(num_neurons * sizeof(double));
    double *last_spikes = malloc(num_neurons * sizeof(double));
    double *new_spikes  = malloc(num_neurons * sizeof(double));
    double *v_buf       = malloc(num_neurons * sizeof(double));

    if (!ext_input || !last_spikes || !new_spikes || !v_buf) {
        fprintf(stderr, "Error: allocation failed in record_inference_csvs\n");
        goto cleanup;
    }

    FILE *f_spikes = fopen("spikes.csv", "w");
    FILE *f_v      = fopen("membrane_potentials.csv", "w");
    if (!f_spikes || !f_v) {
        fprintf(stderr, "Error: could not open output CSV files\n");
        if (f_spikes) fclose(f_spikes);
        if (f_v)      fclose(f_v);
        goto cleanup;
    }

    spires_reservoir_reset(res);

    for (size_t i = 0; i < num_neurons; i++)
        last_spikes[i] = get_neuron_spike(impl->neurons[i], impl->neuron_type);

    for (int t = 0; t < SEQUENCE_LENGTH; t++) {
        const double *input_slice = sample_input + t * NUM_FEATURES;

        /* Compute W_in * input once per macro-step, matching step_reservoir's logic */
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    (int)num_neurons, NUM_FEATURES,
                    impl->input_strength, impl->W_in, NUM_FEATURES,
                    input_slice, 1, 0.0, ext_input, 1);

        for (int micro = 0; micro < num_micro_steps; micro++) {
            for (size_t i = 0; i < num_neurons; i++) {
                double recurrent = synapse_row_dot(&impl->W, i, last_spikes);
                update_neuron(impl->neurons[i], impl->neuron_type,
                              ext_input[i] + recurrent, impl->dt);
                new_spikes[i] = get_neuron_spike(impl->neurons[i], impl->neuron_type);
                v_buf[i]      = get_neuron_state(impl->neurons[i], impl->neuron_type);
            }

            for (size_t i = 0; i < num_neurons; i++) {
                fprintf(f_spikes, "%.6f%s", new_spikes[i], i + 1 < num_neurons ? "," : "\n");
                fprintf(f_v,      "%.6f%s", v_buf[i],      i + 1 < num_neurons ? "," : "\n");
            }

            memcpy(last_spikes, new_spikes, num_neurons * sizeof(double));
        }
    }

    fclose(f_spikes);
    fclose(f_v);

    /* Adjacency matrix — one write, independent of time */
    FILE *f_adj = fopen("adjacency_matrix.csv", "w");
    if (!f_adj) {
        fprintf(stderr, "Error: could not open adjacency_matrix.csv\n");
        goto cleanup;
    }
    double *W_dense = malloc(num_neurons * num_neurons * sizeof(double));
    if (!W_dense) {
        fprintf(stderr, "Error: allocation failed for adjacency matrix dump\n");
        fclose(f_adj);
        goto cleanup;
    }
    synapse_to_dense(&impl->W, W_dense);
    for (size_t i = 0; i < num_neurons; i++) {
        for (size_t j = 0; j < num_neurons; j++) {
            fprintf(f_adj, "%.6f%s",
                    W_dense[i * num_neurons + j],
                    j + 1 < num_neurons ? "," : "\n");
        }
    }
    free(W_dense);
    fclose(f_adj);

    printf("Wrote %d rows (SEQUENCE_LENGTH=%d x %d micro-steps) to "
           "spikes.csv, membrane_potentials.csv, and adjacency_matrix.csv\n",
           total_rows, SEQUENCE_LENGTH, num_micro_steps);

cleanup:
    free(ext_input);
    free(last_spikes);
    free(new_spikes);
    free(v_buf);
}

int main(void) {
    srand(time(NULL));

    double *all_features;
    int    *all_labels;
    if (load_preprocessed_data(&all_features, &all_labels) != 0)
        return 1;

    double *train_x, *train_y;
    double *test_x;
    int    *test_y;
    split_dataset(all_features, all_labels,
                  &train_x, &train_y,
                  &test_x, &test_y);

    printf("Configuring reservoir...\n");

    size_t num_neurons    = 400;
    double desired_degree = 15;
    double spectral_radius = 0.99;
    double connectivity   = desired_degree / ((double)num_neurons - 1);
    double input_strength = 1.0;
    double ei_ratio       = 0.8;
    double dt             = 0.1;
    double lambda         = 0.0;
    double alpha          = 0.50;

    double neuron_params[] = {
        1.0,                    /* V_th    */
        0.0,                    /* V_reset */
        0.0,                    /* V_rest  */
        20.0,                   /* tau_m   */
        alpha,                  /* alpha   */
        10.0,                   /* Tmem    */
        bias_from_alpha(alpha), /* bias    */
        0.0                     /* t_ref   */
    };

    spires_reservoir_config cfg = {
        .num_neurons      = num_neurons,
        .num_inputs       = NUM_FEATURES,
        .num_outputs      = NUM_CLASSES,
        .spectral_radius  = spectral_radius,
        .ei_ratio         = ei_ratio,
        .input_strength   = input_strength,
        .connectivity     = connectivity,
        .dt               = dt,
        .connectivity_type = SPIRES_CONN_SCALE_FREE,
        .neuron_type      = SPIRES_NEURON_FLIF_GL,
        .neuron_params    = neuron_params
    };

    spires_reservoir *res = NULL;
    if (spires_reservoir_create(&cfg, &res) != SPIRES_OK || !res) {
        fprintf(stderr, "Error: failed to create reservoir\n");
        return 1;
    }

    printf("Training reservoir...\n");
    if (spires_train_ridge(res, train_x, train_y,
                           NUM_SAMPLES_TRAIN * SEQUENCE_LENGTH, lambda) != SPIRES_OK) {
        fprintf(stderr, "ridge training failed\n");
        return 1;
    }
    printf("Training complete.\n");

    printf("Recording inference CSVs for test sample 0 (label=%d)...\n", test_y[0]);
    record_inference_csvs(res, &test_x[0], num_neurons);

    /* Run full test set to report accuracy */
    int correct = 0;
    for (int i = 0; i < NUM_SAMPLES_TEST; i++) {
        const double *input = &test_x[i * SEQUENCE_LENGTH * NUM_FEATURES];
        double output[NUM_CLASSES];
        spires_reservoir_reset(res);
        for (int t = 0; t < SEQUENCE_LENGTH; t++)
            spires_step(res, input + t * NUM_FEATURES);
        spires_compute_output(res, output);
        if (argmax(output, NUM_CLASSES) == test_y[i])
            correct++;
    }
    printf("Test accuracy: %.2f%%\n", (double)correct / NUM_SAMPLES_TEST * 100.0);

    spires_reservoir_destroy(res);
    free(train_x); free(train_y);
    free(test_x);  free(test_y);
    free(all_features); free(all_labels);
    return 0;
}
