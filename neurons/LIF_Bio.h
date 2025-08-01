#ifndef LIF_BIO_H
#define LIF_BIO_H

typedef struct {
    double V;
    double V_th;
    double V_0;
    double tau;
    double bias;
    double spike;
} LIFBioNeuron;

LIFBioNeuron* init_LIF_Bio(double *neuron_params);
void update_LIF_Bio(LIFBioNeuron *neuron, double input, double dt);
void free_LIF_Bio(LIFBioNeuron *neuron);

#endif

