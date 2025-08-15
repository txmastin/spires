#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flif_caputo.h"

struct flif_caputo_neuron *init_flif_caputo(double *params, double dt) 
{
    struct flif_caputo_neuron *n = malloc(sizeof(struct flif_caputo_neuron));
    if (!n) return NULL;

    n->C_m = params[0];
    n->g_l = params[1];
    n->V_l = params[2];
    n->V_th = params[3];
    n->V_0 = params[4];
    n->V_reset = params[4];
    n->V_peak = params[5];
    n->alpha = params[6];
    n->t_ref = params[7];
    double T_mem = params[8];

    n->V = n->V_0;
    n->spike = 0;
    n->t_prev = 2 * n->t_ref;
    n->step = 0;

    n->mem_len = (T_mem > 0) ? (int)(T_mem / dt) : MAX_MEM_LEN;
    if (n->mem_len > MAX_MEM_LEN) n->mem_len = MAX_MEM_LEN;

    n->delta_mem = calloc(n->mem_len - 2, sizeof(double));
    n->coeffs = malloc((n->mem_len - 2) * sizeof(double));

    int i;
    for (i = 0; i < n->mem_len - 2; i++) {
        double x = i + 2;
        n->coeffs[i] = pow(x, 1.0 - n->alpha) - pow(x - 1.0, 1.0 - n->alpha);
    }

    n->kr = pow(dt, n->alpha) * tgamma(2.0 - n->alpha);
    return n;
}

void update_flif_caputo(struct flif_caputo_neuron *n, double input, double dt) 
{
    if (n->spike == 1.0) n->spike = 0.0;

    if (n->t_prev > n->t_ref) {
        double dV = (-n->g_l * (n->V - n->V_l) + input) / n->C_m;
        double markov = n->kr * dV;

        for (int i = n->mem_len - 3; i > 0; i--) n->delta_mem[i] = n->delta_mem[i - 1];
        n->delta_mem[0] = n->V - n->V_0;

        double mem = 0;
        for (int i = 0; i < n->mem_len - 2; i++) mem += n->coeffs[i] * n->delta_mem[i];

        n->V += markov - mem;

        if (n->V >= n->V_th) {
            n->V = n->V_peak;
            n->spike = 1.0;
            n->V = n->V_0;
            n->t_prev = 0.0;
        }
    } else {
        n->V = n->V_0;
    }

    n->t_prev += dt;
    n->step++;
}

void free_flif_caputo(struct flif_caputo_neuron *n) 
{
    if (!n) return;
    free(n->delta_mem);
    free(n->coeffs);
    free(n);
}
