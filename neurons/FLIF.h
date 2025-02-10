#ifndef FLIF_H
#define FLIF_H

typedef struct {
    double V;
    double V_th;
    double V_0;
    double leak_rate;
    int spiking;
    double alpha;
} FLIFNeuron;

FLIFNeuron* init_FLIF(double V, double V_th, double V_0, double leak_rate, double alpha);
void update_FLIF(FLIFNeuron *neuron, double *inputs);
void free_FLIF(FLIFNeuron **neuron);

#endif
