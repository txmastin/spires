#ifndef LIF_DISCRETE_H
#define LIF_DISCRETE_H

struct lif_discrete_neuron {
    double V;
    double V_th;
    double V_0;
    double leak_rate;
    double spike;
    double bias;
};

struct lif_discrete_neuron* init_lif_discrete(double *neuron_params);
void update_lif_discrete(struct lif_discrete_neuron *neuron, double input);
void free_lif_discrete(struct lif_discrete_neuron *neuron);

#endif // LIF_DISCRETE_H
