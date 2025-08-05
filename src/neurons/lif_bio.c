#include <stdlib.h>
#include "lif_bio.h"

struct lif_bio_neuron* init_lif_bio(double *neuron_params) 
{
    struct lif_bio_neuron *n = malloc(sizeof(struct lif_bio_neuron));
    n->V_0  = neuron_params[0];  // reset voltage 
    n->V_th   = neuron_params[1];  // threshold
    n->tau   = neuron_params[2];  // membrane time constant
    n->bias  = neuron_params[3];  // constant bias
    n->V = n->V_0; // initial voltage set to reset voltage
    n->spike = 0.0;
    return n;
}

void update_lif_bio(struct lif_bio_neuron *n, double input, double dt) 
{
    if (n->spike == 1.0) {
        n->spike = 0.0;
    }

    double dV = (-n->V + input + n->bias) / n->tau;
    n->V += dV * dt;

    if (n->V >= n->V_th) {
        n->V = n->V_0;
        n->spike = 1.0;
    }
}

void free_lif_bio(struct lif_bio_neuron *n) 
{
    free(n);
}

