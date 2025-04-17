#ifndef FLIF_GL_H
#define FLIF_GL_H

#define MAX_MEM_LEN 10000

typedef struct {
    double V, V_th, V_0, spike;
    double Cm, gl, Vl, Vreset, Vpeak;
    double alpha, tref, tprev, dt, kr;
    double *DeltaM, *coeffs;
    int mem_len;
    int step;
} FLIFGLNeuron;

FLIFGLNeuron* init_FLIF_GL(double *params);
void update_FLIF_GL(FLIFGLNeuron *neuron, double input);
void free_FLIF_GL(FLIFGLNeuron *neuron);

#endif // FLIF_GL_H
