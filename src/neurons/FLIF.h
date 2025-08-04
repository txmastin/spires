#ifndef FLIF_H
#define FLIF_H

typedef struct {
    double V;
    double V_th;
    double V_0;
    double leak_rate;
    double spike;
    double alpha;
} FLIFNeuron;

FLIFNeuron* init_FLIF(double *neuron_params);
void update_FLIF(FLIFNeuron *neuron, double inputs, double dt);
void free_FLIF(FLIFNeuron *neuron);

#endif // FLIF_H
