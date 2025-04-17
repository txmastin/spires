#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FLIF_GL.h"

FLIFGLNeuron* init_FLIF_GL(double *params) {
    FLIFGLNeuron *n = malloc(sizeof(FLIFGLNeuron));
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

    n->DeltaM = calloc(n->mem_len - 1, sizeof(double));
    n->coeffs = malloc((n->mem_len - 1) * sizeof(double));

    n->coeffs[0] = 1.0;
    for (int i = 1; i < n->mem_len - 1; i++) {
        n->coeffs[i] = n->coeffs[i - 1] * (1.0 - (1.0 + n->alpha) / (double)i);
    }

    n->kr = pow(n->dt, n->alpha);
    return n;
}

void update_FLIF_GL(FLIFGLNeuron *n, double input) {
    if (n->spike == 1.0) n->spike = 0.0;

    if (n->tprev > n->tref) {
        double dV = (-n->gl * (n->V - n->Vl) + input) / n->Cm;

        for (int i = n->mem_len - 2; i > 0; i--) n->DeltaM[i] = n->DeltaM[i - 1];
        n->DeltaM[0] = n->V;

        double Mem = 0;
        for (int i = 0; i < n->mem_len - 1; i++) Mem += n->coeffs[i] * n->DeltaM[i];

        n->V = n->kr * dV - Mem;

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

void free_FLIF_GL(FLIFGLNeuron *n) {
    if (!n) return;
    free(n->DeltaM);
    free(n->coeffs);
    free(n);
}
