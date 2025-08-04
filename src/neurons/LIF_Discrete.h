#ifndef LIF_DISCRETE_H
#define LIF_DISCRETE_H

typedef struct {
    double V;
    double V_th;
    double V_0;
    double leak_rate;
    double spike;
} LIFDiscreteNeuron;

LIFDiscreteNeuron* init_LIF_Discrete(double *neuron_params);
void update_LIF_Discrete(LIFDiscreteNeuron *neuron, double input);
void free_LIF_Discrete(LIFDiscreteNeuron *neuron);

#endif // LIF_DISCRETE_H
