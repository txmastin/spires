#ifndef SPIRES_H
#define SPIRES_H


#include <stdlib.h> /* For size_t */

/*
 * This guard allows the header to be used in C++ code,
 * a standard practice for portable C libraries.
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * enum spires_neuron_type - Defines the neuron model for the reservoir.
 */
enum spires_neuron_type {
	LIF_DISCRETE,
	LIF_BIO,
	FLIF,
	FLIF_CAPUTO,
	FLIF_GL
};

/**
 * enum spires_connectivity_type - Defines the internal connection topology.
 */
enum spires_connectivity_type {
	RANDOM,
	SMALL_WORLD,
	SCALE_FREE
};

/*
 * Forward declaration of the main reservoir struct.
 * This makes it an "incomplete type". Users can have pointers
 * to it, but cannot access its members directly, ensuring encapsulation.
 */
struct spires_reservoir;

/**
 * spires_create() - Create and allocate a new spiking reservoir.
 * @num_neurons:	Number of neurons in the reservoir.
 * @num_inputs:		Number of external input channels.
 * @num_outputs:	Number of output channels.
 * @spectral_radius:	The desired spectral radius (rho) of the weights.
 * @ei_ratio:		Ratio of excitatory to inhibitory neurons.
 * @input_strength:	Scaling factor for the input weights.
 * @connectivity:	Connection probability for the internal weight matrix.
 * @dt:			Simulation timestep.
 * @connectivity_type:	The connection topology to generate.
 * @neuron_type:	The neuron model to use for all neurons.
 * @neuron_params:	Pointer to an array of parameters for the chosen model.
 *
 * This function allocates memory for a new reservoir and its components.
 * The internal weights are not initialized until spires_init() is called.
 *
 * Return: A pointer to the new struct spires_reservoir on success,
 * NULL on memory allocation failure.
 */
struct spires_reservoir *spires_create(
        size_t num_neurons, size_t num_inputs, size_t num_outputs,
	double spectral_radius, double ei_ratio, double input_strength,
	double connectivity, double dt,
	enum spires_connectivity_type connectivity_type,
	enum spires_neuron_type neuron_type, double *neuron_params);

/**
 * spires_free() - Free all memory associated with a reservoir.
 * @res:	The reservoir to free. Can be NULL.
 */
void spires_free(struct spires_reservoir *res);

/**
 * spires_init() - Initialize or re-initialize the reservoir's weights.
 * @res:	The reservoir to initialize.
 *
 * Generates the internal and input weight matrices according to the
 * specified connectivity and parameters.
 *
 * Return: 0 on success, or a negative error code on failure.
 */
int spires_init(struct spires_reservoir *res);

/**
 * spires_reset() - Reset the state of all neurons in the reservoir.
 * @res:	The reservoir whose state will be reset.
 */
void spires_reset(struct spires_reservoir *res);

/**
 * spires_train_ridge_regression() - Train the reservoir's output weights.
 * @res:		The reservoir to train.
 * @input_series:	An array of input values.
 * @target_series:	An array of target output values.
 * @series_length:	The number of timesteps in the series.
 * @lambda:		The regularization parameter for ridge regression.
 */
void spires_train_ridge_regression(struct spires_reservoir *res,
	const double *input_series, const double *target_series,
	size_t series_length, double lambda);

/**
 * spires_run() - Run the reservoir in inference mode.
 * @res:		The reservoir to run.
 * @input_series:	An array of input values.
 * @series_length:	The number of timesteps in the input series.
 *
 * Feeds the input series into the trained reservoir and collects the output.
 * Assumes the reservoir has already been trained.
 *
 * Return: A pointer to a newly allocated array containing the output.
 * The caller is responsible for freeing this memory. Returns NULL
 * on allocation failure.
 */
double *spires_run(struct spires_reservoir *res,
	const double *input_series, size_t series_length);


#ifdef __cplusplus
}
#endif

#endif /* SPIRES_H */
