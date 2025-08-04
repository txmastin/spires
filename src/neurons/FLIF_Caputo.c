#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FLIF_Caputo.h"

FLIFCaputoNeuron* init_FLIF_Caputo(double *params) {
    FLIFCaputoNeuron *n = malloc(sizeof(FLIFCaputoNeuron));
    if (!n) return NULL;

    n->Cm = params[0];
    n->gl = params[1];
    n->Vl = params[2];
    n->V_th = params[3];
    n->V_0 = params[4];
    n->Vreset = params[4];
    n->Vpeak = params[5];
    n->alpha = params[6];
    n->tref = params[7];
    double Tmem = params[8];
    n->dt = params[9];

    n->V = n->V_0;
    n->spike = 0;
    n->tprev = 2 * n->tref;
    n->step = 0;

    n->mem_len = (Tmem > 0) ? (int)(Tmem / n->dt) : MAX_MEM_LEN;
    if (n->mem_len > MAX_MEM_LEN) n->mem_len = MAX_MEM_LEN;

    n->DeltaM = calloc(n->mem_len - 2, sizeof(double));
    n->coeffs = malloc((n->mem_len - 2) * sizeof(double));

    int i;
    for (i = 0; i < n->mem_len - 2; i++) {
        double x = i + 2;
        n->coeffs[i] = pow(x, 1.0 - n->alpha) - pow(x - 1.0, 1.0 - n->alpha);
    }

    n->kr = pow(n->dt, n->alpha) * tgamma(2.0 - n->alpha);
    return n;
}

void update_FLIF_Caputo(FLIFCaputoNeuron *n, double input, double dt) {
    if (n->spike == 1.0) n->spike = 0.0;

    if (n->tprev > n->tref) {
        double dV = (-n->gl * (n->V - n->Vl) + input) / n->Cm;
        double Markov = n->kr * dV;

        for (int i = n->mem_len - 3; i > 0; i--) n->DeltaM[i] = n->DeltaM[i - 1];
        n->DeltaM[0] = n->V - n->V_0;

        double Mem = 0;
        for (int i = 0; i < n->mem_len - 2; i++) Mem += n->coeffs[i] * n->DeltaM[i];

        n->V += Markov - Mem;

        if (n->V >= n->V_th) {
            n->V = n->Vpeak;
            n->spike = 1.0;
            n->V = n->V_0;
            n->tprev = 0.0;
        }
    } else {
        n->V = n->V_0;
    }

    n->tprev += n->dt;
    n->step++;
}

void free_FLIF_Caputo(FLIFCaputoNeuron *n) {
    if (!n) return;
    free(n->DeltaM);
    free(n->coeffs);
    free(n);
}
