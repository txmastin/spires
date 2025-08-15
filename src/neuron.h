#ifndef NEURON_H
#define NEURON_H

#include "neurons/lif_discrete.h"
#include "neurons/lif_bio.h"
#include "neurons/flif_caputo.h"
#include "neurons/flif_gl.h"
#include "neurons/flif_diffusive.h"


enum neuron_type {
    LIF_DISCRETE,
    LIF_BIO,
    FLIF_CAPUTO,
    FLIF_GL,
    FLIF_DIFFUSIVE
};

void* init_neuron(enum neuron_type type, double *neuron_params, double dt);
void update_neuron(void *neuron, enum neuron_type type, double input, double dt);
void free_neuron(void *neuron, enum neuron_type type);
double get_neuron_state(void *neuron, enum neuron_type type);
double get_neuron_spike(void *neuron, enum neuron_type type);

#endif // NEURON_H

