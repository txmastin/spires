#ifndef CROSSBAR_INTERNAL_H
#define CROSSBAR_INTERNAL_H

#include <spires.h>
#include <stddef.h>

typedef struct {
	double g_min;
	double g_max;
	double alpha;
	double max_abs_weight;
} conductance_mapping;

int convert_weights_to_resistances(const spires_reservoir *reservoir,
				   size_t num_neurons, size_t num_outputs,
				   double r_on, double r_off,
				   double **resistances_out,
				   conductance_mapping *mapping);

int decode_crossbar_output(size_t num_neurons, size_t num_outputs,
			   size_t num_timesteps, const double *voltages,
			   const double *resistances, double load_resistance,
			   const conductance_mapping *mapping,
			   double row_voltage_scaler, double *decoded_outputs);

int generate_crossbar_netlist(const spires_crossbar_readout_config *config,
			      const char *netlist_path,
			      const double *resistances);

#endif
