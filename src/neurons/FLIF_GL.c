#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FLIF_GL.h"

// Helper function to compute the GL coefficients
static void compute_gl_coeffs(double* coeffs_array, double alpha, int N) {
    if (!coeffs_array) return;
    coeffs_array[0] = 1.0;
    for (int k = 1; k < N; k++) {
        coeffs_array[k] = coeffs_array[k - 1] * (1.0 - (alpha + 1.0) / (double)k);
    }
}

// init_FLIF_GL conforms to the existing `init_neuron` wrapper
FLIFGLNeuron* init_FLIF_GL(double* params) {
    FLIFGLNeuron* n = (FLIFGLNeuron*)malloc(sizeof(FLIFGLNeuron));
    if (!n) return NULL;

    // Map parameters from the array passed by main.c
    // This order must match the array in your main function.
    n->V_th   = params[0];
    n->V_reset = params[1];
    n->V_rest = params[2];
    n->tau_m  = params[3];
    n->alpha  = params[4];
    n->dt     = params[5]; // This is the micro-step dt from the reservoir
    double Tmem = params[6]; // Memory duration in ms, from params
    n->bias = params[7];

    // Initialize internal state
    n->internal_step = 0;
    n->V = n->V_rest;
    n->spike = 0.0;

    // Setup the circular history buffer based on memory duration and micro-step dt
    n->mem_len = (Tmem > 0 && n->dt > 0) ? (int)(Tmem / n->dt) : 2000;
    if (n->mem_len > MAX_MEM_LEN) n->mem_len = MAX_MEM_LEN;

    n->V_history = (double*)malloc(n->mem_len * sizeof(double));
    n->coeffs    = (double*)malloc(n->mem_len * sizeof(double));

    // Initialize history buffer to the resting potential
    for (int i = 0; i < n->mem_len; i++) {
        n->V_history[i] = n->V_rest;
    }

    // Pre-compute coefficients for the length of our history buffer
    compute_gl_coeffs(n->coeffs, n->alpha, n->mem_len);

    return n;
}

// update_FLIF_GL conforms to the existing `update_neuron` wrapper signature
void update_FLIF_GL(FLIFGLNeuron* n, double input, double dt) {
    // Reset spike from the previous micro-step
    if (n->spike == 1.0) n->spike = 0.0;
    
    // Get the current position (head) in the circular buffer
    int head = n->internal_step % n->mem_len;

    // Get the voltage from the previous micro-step, V[n-1]
    int prev_idx = (head - 1 + n->mem_len) % n->mem_len;
    double V_prev = n->V_history[prev_idx];

    // 1. Calculate the history term using the circular buffer.
    double history = 0.0;
    int limit = (n->internal_step < n->mem_len) ? n->internal_step : n->mem_len - 1;
    for (int k = 1; k <= limit; k++) {
        int idx = (head - k + n->mem_len) % n->mem_len;
        history += n->coeffs[k] * n->V_history[idx];
    }

    // 2. Calculate the "right-hand side" of the differential equation.
    double rhs = (-(V_prev - n->V_rest) / n->tau_m) + input + n->bias;

    // 3. The direct update rule, using the micro-step dt from the reservoir.
    n->V = pow(dt, n->alpha) * rhs - history;

    // 4. Spike-and-reset logic.
    if (n->V >= n->V_th) {
        n->V = n->V_reset;
        n->spike = 1.0;
    }

    // Store the newly calculated voltage in the history buffer at the current position
    n->V_history[head] = n->V;

    // Increment the neuron's internal micro-step counter
    n->internal_step++;
}

void free_FLIF_GL(FLIFGLNeuron* n) {
    if (!n) return;
    free(n->V_history);
    free(n->coeffs);
    free(n);
}
