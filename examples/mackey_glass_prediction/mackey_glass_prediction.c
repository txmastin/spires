#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <spires.h>

void generate_mackey_glass(double *buffer, size_t length, double x0, double tau, double beta, double gamma, int n)
{
    double mg_dt = 1.0;
    // Calculate the required length of the history buffer in discrete steps
    int history_len = (int)ceil(tau / mg_dt);

    // Allocate and initialize the history buffer
    double *history = malloc(history_len * sizeof(double));
    if (history == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for Mackey-Glass history.\n");
        return;
    }
    for (int i = 0; i < history_len; i++) {
        history[i] = x0;
    }

    buffer[0] = x0;
    double x_t = x0; // Current value of x

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

    free(history);
}


int main(void) 
{
    srand(time(NULL));
    FILE *output_file = fopen("data/output_signals.dat", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error: Could not open data files for writing.\n");
        return 1;
    }

    /* ---- Generate Mackey–Glass ---- */
    size_t timesteps = 1000;
    double x0 = 0.1;
    double tau = 17;       // Must be > 17 for chaos
    double beta = 0.2;
    double gamma = 0.1;
    int n = 10;

    double *input_series = malloc(timesteps * sizeof(double));
    if (input_series == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for input_series.\n");
        fclose(output_file);
        return 1;
    }
    
    generate_mackey_glass(input_series, timesteps, x0, tau, beta, gamma, n);


    int rc;
    const size_t Din  = 1;  /* Mackey–Glass is scalar input */
    const size_t Dout = 1;  /* predict next value */
    const size_t horizon = 84;

    const size_t series_length  = timesteps - horizon;
    
    /* Build target series: y[t] = x[t + horizon].
     * We optimize/train on T = series_length - horizon samples. */
    if (series_length <= horizon) {
            fprintf(stderr, "series too short for horizon=%zu\n", horizon);
            return 1;
    }
    size_t T = series_length;

    double *target_series = malloc(T * Dout * sizeof(double));
    if (!target_series) {
            fprintf(stderr, "alloc failure for target_series\n");
            return 1;
    }
    for (size_t t = 0; t < T; t++)
            target_series[t] = input_series[t + horizon];

    /* ---- Neuron Hyperparameters ---- */
    double fractional_neuron_params[] = {
        1.0,            // V_th
        0.0,            // V_reset
        0.0,            // V_rest
        20.0,           // tau_m
        0.7,            // alpha
        (double)timesteps, // Tmem
        0.1             // bias
    };
    
    /* Base config (“ball-park” defaults) */
    spires_reservoir_config base = {
            .num_neurons       = 100,
            .num_inputs        = Din,
            .num_outputs       = Dout,
            .spectral_radius   = 0.90,
            .ei_ratio          = 0.80,
            .input_strength    = 1.00,
            .connectivity      = 0.20,
            .dt                = 0.1,
            .connectivity_type = SPIRES_CONN_RANDOM,
            .neuron_type       = SPIRES_NEURON_FLIF_GL,
            .neuron_params     = fractional_neuron_params,   /* alpha will be set internally by optimizer */
    };

    /* Two-stage budget: quick pass on half data, then full data */
    struct spires_opt_budget buds[] = {
            { .data_fraction = 0.8, .num_seeds = 1, .time_limit_sec = 0.0 },
    };
    struct spires_opt_score score = {
            .lambda_var  = 0.0,
            .lambda_cost = 0.0,
            .metric      = SPIRES_METRIC_AUROC, /* placeholder; optimizer uses 1/(1+MSE) internally */
    };
    struct spires_opt_result out;

    /* Optimize hyperparameters using your task data */
    rc = spires_optimize(&base, buds, (int)(sizeof(buds)/sizeof(buds[0])),
                         &score, &out,
                         /* input  */ input_series,   /* flattened [T x Din]  */
                         /* target */ target_series,  /* flattened [T x Dout] */
                         /* T */     T);
    if (rc) {
            fprintf(stderr, "spires_optimize failed (rc=%d)\n", rc);
            free(target_series);
            return 1;
    }

    /* Train final reservoir with the best config and ridge on ALL T samples */
    spires_reservoir *R = NULL;
    if (spires_reservoir_create(&out.best_config, &R) != SPIRES_OK || !R) {
            fprintf(stderr, "failed to create reservoir\n");
            free(target_series);
            return 1;
    }

    const double lambda = pow(10.0, out.best_log10_ridge);
    if (spires_train_ridge(R, input_series, target_series, T, lambda) != SPIRES_OK) {
            fprintf(stderr, "ridge training failed\n");
            spires_reservoir_destroy(R);
            free(target_series);
            return 1;
    }

    spires_reservoir_reset(R);

    double *reservoir_outputs = spires_run(R, input_series, series_length);
    if (!reservoir_outputs) {
        fprintf(stderr, "inference failed (spires_run returned NULL)\n");
        spires_reservoir_destroy(R);
        free(input_series);
        fclose(output_file);
        return 1;
    }

    fprintf(output_file, "# Timestep Input_Signal Reservoir_Output\n");
    for (size_t i = 0; i < series_length; i++) {
        fprintf(output_file, "%zu %f %f\n", i, input_series[i], reservoir_outputs[i]);
    }

    free(reservoir_outputs);
    spires_reservoir_destroy(R);
    fclose(output_file);
    free(input_series);
    return 0;
}

