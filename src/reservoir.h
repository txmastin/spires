#ifndef RESERVOIR_H
#define RESERVOIR_H

#include "neuron.h"
#include "synapse.h"
#include "rng.h"
#include "plasticity.h"

enum connectivity_type {
    RANDOM,
    SMALL_WORLD,
    SCALE_FREE
};

/* Sign assignment for recurrent weights; see spires_ei_mode. */
enum ei_mode {
    EI_PER_SYNAPSE,
    EI_PER_NEURON
};

struct reservoir {
    void **neurons; 
    size_t num_neurons;
    size_t num_inputs;      
    size_t num_outputs;     
    double spectral_radius;
    double ei_ratio;
    double input_strength;
    double connectivity;
    double dt;
    double *W_in;
    double *W_out;
    struct synapse_matrix W;
    double *shared_neuron_data; // optionally used in some neuron types
    enum connectivity_type connectivity_type;
    enum neuron_type neuron_type;
    enum synapse_type synapse_type;
    enum synapse_backend synapse_backend;
    enum ei_mode ei_mode;
    double *neuron_params;
    double *synapse_params;
    /* Dale's law identities, length num_neurons; NULL under EI_PER_SYNAPSE.
     * A cache: under Dale's law any non-zero in column j already reveals
     * neuron j's sign. Plasticity rules must therefore preserve weight sign,
     * or this silently disagrees with W. */
    signed char *neuron_sign;
    struct plasticity_state plasticity;   /* traces owned here */
    uint64_t seed;              // seed this reservoir was built from
    struct spires_rng rng;      // per-reservoir stream; replaces global rand()
    int use_washout;
    size_t washout;             // states after reset excluded from training
    size_t steps_since_reset;   // macro steps since construction or reset
};

/* Construction parameters. Grouped so that adding one does not lengthen a
 * positional argument list where same-typed arguments can be transposed
 * silently. Mirrors spires_reservoir_config; the public API bridges between
 * them in spires_api.c. */
struct reservoir_params {
    size_t num_neurons;
    size_t num_inputs;
    size_t num_outputs;
    double spectral_radius;
    double ei_ratio;
    double input_strength;
    double connectivity;
    double dt;
    enum connectivity_type connectivity_type;
    enum neuron_type neuron_type;
    double *neuron_params;
    enum synapse_type synapse_type;
    enum synapse_backend synapse_backend;
    double *synapse_params;
    enum ei_mode ei_mode;
    enum plasticity_type plasticity_type;
    double *plasticity_params;
    /* Seed for this reservoir's own RNG. 0 is an ordinary seed, not a
     * sentinel: construction is reproducible by default. Callers wanting a
     * different network per run pass spires_random_seed(). */
    uint64_t seed;
    int use_washout;
    size_t washout;
};

struct reservoir* create_reservoir(const struct reservoir_params *p);

int compute_output(struct reservoir *reservoir, double *output_vector);
double compute_activity(struct reservoir *reservoir);
void step_reservoir(struct reservoir *reservoir, const double *input_vector);
double *run_reservoir(struct reservoir *reservoir, double *input_series, size_t input_length); // input_series is a flattened array of input_vector * num_inputs
int run_reservoir_into(struct reservoir *reservoir, double *input_series, size_t input_length, double *output_series); // output_series holds num_outputs * input_length doubles, caller-allocated
void free_reservoir(struct reservoir *reservoir);
int init_weights(struct reservoir *reservoir);
int randomize_output_layer(struct reservoir *reservoir);
int rescale_weights(struct reservoir *reservoir);
int init_reservoir(struct reservoir *reservoir);
void read_reservoir_state(struct reservoir *reservoir, double *buffer);
void read_reservoir_spikes(struct reservoir *reservoir, double *buffer);
double *copy_reservoir_state(struct reservoir *reservoir);
void train_output_iteratively(struct reservoir *reservoir, double *target_vector, double lr);
void train_output_ridge_regression(struct reservoir *reservoir, double *input_series,
                                double *target_series, size_t series_length, double lambda);
void train_output_rls(struct reservoir *reservoir, double *input_series,
                      double *target_series, size_t series_length,
                      double delta, double lambda);
void reset_reservoir(struct reservoir *reservoir);
struct reservoir *coarse_grain_reservoir(const struct reservoir *r, double weight_threshold);
double *copy_reservoir_weights(const struct reservoir *r);
void    read_reservoir_weights(const struct reservoir *r, double *buffer);
double *copy_reservoir_readout(const struct reservoir *r);
void    read_reservoir_readout(const struct reservoir *r, double *buffer);
#endif // RESERVOIR_H

