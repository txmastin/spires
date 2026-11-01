#include "crossbar_circuit.h"
#include "crossbar_models.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Converts the spires trained weights into differential pair resistances for
// crossbar
int convert_weights_to_resistances(const spires_reservoir *reservoir,
				   size_t num_neurons, size_t num_outputs,
				   double r_on, double r_off,
				   double **resistances_out,
				   conductance_mapping *mapping)
{
	if (!reservoir || num_neurons == 0 || num_outputs == 0 || r_on <= 0.0 ||
	    r_off <= r_on || !resistances_out || !mapping) {
		return -1;
	}

	*resistances_out = NULL;
	if (num_outputs > SIZE_MAX / 2 ||
	    num_neurons > SIZE_MAX / num_outputs) {
		return -1;
	}

	size_t weight_count = num_neurons * num_outputs;
	/* Conductance cannot be negative, so each signed output needs separate
	 * positive and negative current paths. */
	size_t num_physical_columns = num_outputs * 2;
	if (num_neurons > SIZE_MAX / num_physical_columns ||
	    num_neurons * num_physical_columns > SIZE_MAX / sizeof(double)) {
		return -1;
	}

	double *readout = malloc(weight_count * sizeof(*readout));
	if (!readout) {
		fprintf(stderr, "Failed to allocate readout weights\n");
		return -1;
	}
	if (spires_read_readout(reservoir, readout) != SPIRES_OK) {
		fprintf(stderr, "Failed to read SPIRES readout weights\n");
		free(readout);
		return -1;
	}

	double *resistances =
	    malloc(num_neurons * num_physical_columns * sizeof(*resistances));
	if (!resistances) {
		fprintf(stderr, "Failed to allocate crossbar resistances\n");
		free(readout);
		return -1;
	}

	/* A single scale factor preserves relative weights across every output
	 * while fitting the largest magnitude into the device conductance
	 * range. */
	double max_abs_weight = 0.0;
	for (size_t i = 0; i < weight_count; i++) {
		if (!isfinite(readout[i])) {
			fprintf(stderr,
				"Invalid readout weight at index %zu: %g\n", i,
				readout[i]);
			free(readout);
			free(resistances);
			return -1;
		}

		double abs_weight = fabs(readout[i]);
		if (abs_weight > max_abs_weight) {
			max_abs_weight = abs_weight;
		}
	}
	if (max_abs_weight == 0.0) {
		fprintf(stderr,
			"Cannot map an all-zero readout to resistances\n");
		free(readout);
		free(resistances);
		return -1;
	}

	mapping->g_min = 1.0 / r_off;
	mapping->g_max = 1.0 / r_on;
	mapping->max_abs_weight = max_abs_weight;

	mapping->alpha = (mapping->g_max - mapping->g_min) / max_abs_weight;

	for (size_t neuron = 0; neuron < num_neurons; neuron++) {
		for (size_t output = 0; output < num_outputs; output++) {
			size_t weight_index = output * num_neurons + neuron;

			size_t positive_column = 2 * output;
			size_t negative_column = positive_column + 1;

			double positive_conductance;
			double negative_conductance;

			size_t positive_index =
			    neuron * num_physical_columns + positive_column;
			size_t negative_index =
			    neuron * num_physical_columns + negative_column;

			double weight = readout[weight_index];

			/* Keeping the inactive device at g_min respects the
			 * finite off resistance; the shared baseline cancels on
			 * subtraction. */
			if (weight >= 0.0) {
				positive_conductance =
				    mapping->g_min + mapping->alpha * weight;
				negative_conductance = mapping->g_min;
			} else {
				positive_conductance = mapping->g_min;
				negative_conductance =
				    mapping->g_min + mapping->alpha * (-weight);
			}

			resistances[positive_index] =
			    1.0 / positive_conductance;
			resistances[negative_index] =
			    1.0 / negative_conductance;
		}
	}

	*resistances_out = resistances;
	free(readout);

	return 0;
}

/********** Crossbar Generator **********/
static int write_netlist_header(FILE *file,
				const spires_crossbar_readout_config *config)
{
	return fprintf(file,
		       "* SPIRES online crossbar\n"
		       "* Rows: %zu\n"
		       "* Columns: %zu\n"
		       "\n%s\n",
		       config->num_neurons, config->num_outputs * 2,
		       crossbar_model_get(config->model)->netlist) < 0
		   ? -1
		   : 0;
}

static int write_external_inputs(FILE *file,
				 const spires_crossbar_readout_config *config)
{
	if (fprintf(file, "\n* SPIRES reservoir states\n") < 0)
		return -1;

	for (size_t row = 0; row < config->num_neurons; row++) {
		if (fprintf(file, "VROW%zu row%zu 0 dc 0 external\n", row,
			    row) < 0)
			return -1;
	}

	return 0;
}

static int write_memristor_array(FILE *file,
				 const spires_crossbar_readout_config *config,
				 const double *resistances)
{
	size_t columns = config->num_outputs * 2;

	if (fprintf(file, "\n* Memristor array\n") < 0)
		return -1;

	for (size_t row = 0; row < config->num_neurons; row++) {
		for (size_t column = 0; column < columns; column++) {
			if (fprintf(file,
				    "X%zu_%zu row%zu col%zu %s"
				    " PARAMS: Rinit=%.12g\n",
				    row, column, row, column,
				    crossbar_model_get(config->model)->subcircuit_name,
				    resistances[row * columns + column]) < 0)
				return -1;
		}
	}

	return 0;
}

static int write_column_loads(FILE *file,
			      const spires_crossbar_readout_config *config)
{
	size_t columns = config->num_outputs * 2;

	if (fprintf(file, "\n* Column loads\n") < 0)
		return -1;

	for (size_t column = 0; column < columns; column++) {
		if (fprintf(file, "RLOAD%zu col%zu 0 %.17g\n", column, column,
			    config->load_resistance) < 0)
			return -1;
	}

	return 0;
}

int generate_crossbar_netlist(const spires_crossbar_readout_config *config,
			      const char *netlist_path,
			      const double *resistances)
{
	if (!crossbar_model_get(config->model)) {
		return -1;
	}

	FILE *file = fopen(netlist_path, "w");

	if (!file)
		return -1;

	if (write_netlist_header(file, config) != 0 ||
	    write_external_inputs(file, config) != 0 ||
	    write_memristor_array(file, config, resistances) != 0 ||
	    write_column_loads(file, config) != 0 ||
	    fprintf(file, "\n* Online simulation\n.tran %.17g %.17g uic\n",
		    config->time_step,
		    config->num_timesteps * config->time_step) < 0 ||
	    fprintf(file, ".save time") < 0) {
		fclose(file);
		return -1;
	}

	for (size_t column = 0; column < config->num_outputs * 2; column++) {
		if (fprintf(file, " v(col%zu)", column) < 0) {
			fclose(file);
			return -1;
		}
	}

	if (fprintf(file, "\n.end\n") < 0 || fclose(file) != 0)
		return -1;

	return 0;
}

// decodes the read column voltage outputs back to readable format
int decode_crossbar_output(size_t num_neurons, size_t num_outputs,
			   size_t num_timesteps, const double *voltages,
			   const double *resistances, double load_resistance,
			   const conductance_mapping *mapping,
			   double row_voltage_scaler, double *decoded_outputs)
{
	if (num_neurons == 0 || num_outputs == 0 || num_timesteps == 0 ||
	    !voltages || !resistances || !mapping || !decoded_outputs) {
		return -1;
	}

	if (load_resistance <= 0.0 || mapping->alpha == 0.0 ||
	    row_voltage_scaler == 0.0) {
		return -1;
	}

	size_t num_physical_columns = num_outputs * 2;

	for (size_t timestep = 0; timestep < num_timesteps; timestep++) {
		for (size_t output = 0; output < num_outputs; output++) {
			size_t positive_column;
			size_t negative_column;
			size_t positive_voltage_index;
			size_t negative_voltage_index;
			size_t output_index;
			double positive_conductance_sum = 0.0;
			double negative_conductance_sum = 0.0;
			double positive_voltage;
			double negative_voltage;
			double positive_load_current;
			double negative_load_current;
			double positive_source_current;
			double negative_source_current;

			positive_column = 2 * output;
			negative_column = positive_column + 1;

			for (size_t neuron = 0; neuron < num_neurons;
			     neuron++) {
				size_t positive_resistance_index;
				size_t negative_resistance_index;

				positive_resistance_index =
				    neuron * num_physical_columns +
				    positive_column;

				negative_resistance_index =
				    neuron * num_physical_columns +
				    negative_column;

				positive_conductance_sum +=
				    1.0 /
				    resistances[positive_resistance_index];

				negative_conductance_sum +=
				    1.0 /
				    resistances[negative_resistance_index];
			}

			positive_voltage_index =
			    timestep * num_physical_columns + positive_column;

			negative_voltage_index =
			    timestep * num_physical_columns + negative_column;

			positive_voltage = voltages[positive_voltage_index];
			negative_voltage = voltages[negative_voltage_index];

			positive_load_current =
			    positive_voltage / load_resistance;
			negative_load_current =
			    negative_voltage / load_resistance;

			/* Column voltage also drops across every device. Adding
			 * that lost current recovers the source-side dot
			 * product. */
			positive_source_current =
			    positive_load_current +
			    positive_voltage * positive_conductance_sum;

			negative_source_current =
			    negative_load_current +
			    negative_voltage * negative_conductance_sum;

			output_index = timestep * num_outputs + output;

			/* Subtraction cancels the shared off-state conductance;
			 * the denominator reverses weight and input-voltage
			 * scaling. */
			decoded_outputs[output_index] =
			    (positive_source_current -
			     negative_source_current) /
				    (mapping->alpha * row_voltage_scaler);
		}
	}

	return 0;
}
