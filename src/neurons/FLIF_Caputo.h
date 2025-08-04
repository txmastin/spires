#ifndef FLIF_CAPUTO_H
#define FLIF_CAPUTO_H

#define MAX_MEM_LEN 20000

typedef struct {
    double V, V_th, V_0, spike;
    double Cm, gl, Vl, Vreset, Vpeak;
    double alpha, tref, tprev, dt, kr;
    double *DeltaM, *coeffs;
    int mem_len;
    int step;
} FLIFCaputoNeuron;

FLIFCaputoNeuron* init_FLIF_Caputo(double *params);
void update_FLIF_Caputo(FLIFCaputoNeuron *neuron, double input, double dt);
void free_FLIF_Caputo(FLIFCaputoNeuron *neuron);

#endif // FLIF_CAPUTO_H
