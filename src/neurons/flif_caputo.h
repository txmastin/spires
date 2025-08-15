#ifndef FLIF_CAPUTO_H
#define FLIF_CAPUTO_H

#define MAX_MEM_LEN 20000

struct flif_caputo_neuron {
    double V, V_th, V_0, spike;
    double C_m, g_l, V_l, V_reset, V_peak;
    double alpha, t_ref, t_prev, kr;
    double *delta_mem, *coeffs;
    int mem_len;
    int step;
};

struct flif_caputo_neuron* init_flif_caputo(double *params, double dt);
void update_flif_caputo(struct flif_caputo_neuron *neuron, double input, double dt);
void free_flif_caputo(struct flif_caputo_neuron *neuron);

#endif // FLIF_CAPUTO_H
