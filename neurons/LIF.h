#ifndef LIF_H
#define LIF_H

typedef struct {
    double V;
    double V_th;
    double V_0;
    double leak_rate;
    int spiking;
} LIFNeuron;

LIFNeuron* init_LIF(double V, double V_th, double V_0, double leak_rate);
void update_LIF(LIFNeuron *neuron, double *inputs);
void free_LIF(LIFNeuron *neuron);

#endif // LIF_H
