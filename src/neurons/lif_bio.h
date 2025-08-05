#ifndef LIF_BIO_H
#define LIF_BIO_H

struct lif_bio_neuron {
    double V;
    double V_th;
    double V_0;
    double tau;
    double bias;
    double spike;
};

struct lif_bio_neuron *init_lif_bio(double *neuron_params);
void update_lif_bio(struct lif_bio_neuron *neuron, double input, double dt);
void free_lif_bio(struct lif_bio_neuron *neuron);

#endif

