#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flif_gl.h"


struct flif_gl_neuron *init_flif_gl(double *params, double dt, double *coeffs) 
{
    struct flif_gl_neuron* n = malloc(sizeof(struct flif_gl_neuron));
    if (!n) return NULL;

    // Map parameters from the array passed by main.c
    // This order must match the array in your main function.
    n->V_th   = params[0];
    n->V_reset = params[1];
    n->V_rest = params[2];
    n->tau_m  = params[3];
    n->alpha  = params[4];
    double T_mem = params[5]; // Memory duration
    n->bias = params[6];
    n->t_ref = params[7]; // Absolute refractory period

    // copy coeffs:
    n->coeffs = coeffs;

    // Initialize internal state
    n->internal_step = 0;
    n->V = n->V_rest;
    n->spike = 0.0;
    // Start strictly past t_ref so the very first update integrates
    // normally regardless of t_ref's value (in particular t_ref=0 must not
    // make the first call spuriously refractory: t_prev=2*t_ref would give
    // 0>0=false there, which is wrong).
    n->t_prev = n->t_ref + dt;

    // Setup the circular history buffer based on memory duration and micro-step dt
    n->mem_len = (T_mem > 0 && dt > 0) ? (int)(T_mem / dt) : 2000;
    if (n->mem_len > MAX_MEM_LEN) n->mem_len = MAX_MEM_LEN;

    n->V_history = malloc(n->mem_len * sizeof(double));

    // Initialize history buffer to the resting potential
    for (int i = 0; i < n->mem_len; i++) {
        n->V_history[i] = n->V_rest;
    }

    return n;
}

void update_flif_gl(struct flif_gl_neuron* n, double input, double dt)
{
    // Reset spike from the previous micro-step
    if (n->spike == 1.0) n->spike = 0.0;

    // Get the current position (head) in the circular buffer. Must be
    // computed every call (not just when integrating) so the buffer stays
    // correctly time-indexed through refractory periods -- coeffs[k] are
    // fixed-lag weights in absolute time steps, not "steps since active".
    int head = n->internal_step % n->mem_len;

    if (n->t_prev > n->t_ref) {
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
            n->t_prev = 0.0;
        }
    } else {
        // Absolute refractory period: held at reset, dynamics not
        // integrated. Physically, this is the 2A -> A reaction of a DP
        // scheme -- a spike temporarily removes the neuron from the active
        // pool, which is what prevents runaway full-synchrony lock-in.
        n->V = n->V_reset;
    }

    // Store the newly calculated voltage in the history buffer at the current position
    n->V_history[head] = n->V;

    // Increment the neuron's internal micro-step counter and refractory clock
    n->internal_step++;
    n->t_prev += dt;
}

void free_flif_gl(struct flif_gl_neuron* n) 
{
    if (!n) return;
    free(n->V_history);
    free(n);
}
