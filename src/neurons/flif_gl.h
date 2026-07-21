#ifndef FLIF_GL_H
#define FLIF_GL_H

#define MAX_MEM_LEN 20000

struct flif_gl_neuron {
    double V_th;
    double V_reset;
    double V_rest;
    double tau_m;
    double alpha;
    double bias;
    double t_ref;   // Absolute refractory period (same time units as dt/tau_m)

    // Neuron state variables
    double V;       // Current membrane potential
    double spike;   // Current spike state (0.0 or 1.0)
    double t_prev;  // Time elapsed since last spike (see t_ref)

    // Internal state for self-contained simulation
    long internal_step; // Tracks the neuron's own "micro-steps"
    int mem_len;        // The actual size of the history buffer below

    // Memory components
    double * V_history;  // A circular buffer for recent voltage history
    double * coeffs;     // Pre-computed Grünwald-Letnikov coefficients

};

// Function Prototypes
struct flif_gl_neuron* init_flif_gl(double* params, double dt, double *coeffs);
void update_flif_gl(struct flif_gl_neuron* n, double input, double dt);
void free_flif_gl(struct flif_gl_neuron* n);

#endif // FLIF_GL_H
