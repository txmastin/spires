#ifndef FLIF_GL_H
#define FLIF_GL_H

// The MAX_MEM_LEN is a safeguard for the history buffer size.
#define MAX_MEM_LEN 20000

typedef struct {
    // Core model parameters from the Python script
    double V_th;
    double V_reset;
    double V_rest;
    double tau_m;
    double alpha;
    double dt;
    double bias;

    // Neuron state variables
    double V;       // Current membrane potential
    double spike;   // Current spike state (0.0 or 1.0)

    // Internal state for self-contained simulation
    long internal_step; // Tracks the neuron's own "micro-steps"
    int mem_len;        // The actual size of the history buffer below

    // Memory components
    double* V_history;  // A circular buffer for recent voltage history
    double* coeffs;     // Pre-computed Grünwald-Letnikov coefficients

} FLIFGLNeuron;

// Function Prototypes
FLIFGLNeuron* init_FLIF_GL(double* params);
void update_FLIF_GL(FLIFGLNeuron* n, double input, double dt);
void free_FLIF_GL(FLIFGLNeuron* n);

#endif // FLIF_GL_H
