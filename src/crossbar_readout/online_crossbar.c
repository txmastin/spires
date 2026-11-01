#define _POSIX_C_SOURCE 200809L

#include <spires.h>

#include "crossbar_circuit.h"
#include "crossbar_models.h"
#include "online_crossbar.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <ctype.h>
#include <math.h>
#include <ngspice/sharedspice.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

typedef struct Online_Crossbar Online_Crossbar;
static pthread_mutex_t ngspice_mutex = PTHREAD_MUTEX_INITIALIZER;

struct Online_Crossbar {
	spires_crossbar_readout_config config;
	const char *netlist_path;
	double *resistances;
	conductance_mapping mapping;
	double *state_ring;
	double *results;
	size_t published_count;
	size_t consumed_count;
	size_t result_count;
	int started;
	int simulation_done;
	int failed;
	int mutex_initialized;
	int state_condition_initialized;
	int progress_condition_initialized;
	pthread_mutex_t mutex;
	pthread_cond_t state_available;
	pthread_cond_t progress;
};

static int online_crossbar_init(const spires_crossbar_readout_config *config,
				const spires_reservoir *reservoir,
				const char *netlist_path,
				Online_Crossbar **crossbar_out);

static int online_crossbar_start(Online_Crossbar *crossbar);

static int online_crossbar_submit(Online_Crossbar *crossbar, size_t timestep,
				  const double *state, double *previous_output,
				  int *output_ready);

static int online_crossbar_finish(Online_Crossbar *crossbar,
				  double *final_output);

static void online_crossbar_destroy(Online_Crossbar *crossbar);

spires_status run_crossbar_readout_into(const spires_crossbar_readout_config *config,
					spires_reservoir *reservoir,
					const double *input_series,
					size_t series_length, double *buffer)
{
	// validating configs
	if (!config || !reservoir || !input_series || series_length == 0 ||
	    config->num_neurons == 0 || config->num_outputs == 0 ||
	    config->num_timesteps != series_length ||
	    !crossbar_model_get(config->model) || !buffer) {
		return SPIRES_ERR_INVALID_ARG;
	}
	if (config->num_neurons > SIZE_MAX / sizeof(double)) {
		return SPIRES_ERR_INVALID_ARG;
	}
	if (config->num_neurons != spires_num_neurons(reservoir) ||
	    config->num_outputs != spires_num_outputs(reservoir) ||
	    spires_num_inputs(reservoir) == 0) {
		return SPIRES_ERR_INVALID_ARG;
	}

	Online_Crossbar *crossbar = NULL;
	double *state = malloc(config->num_neurons * sizeof(*state));
	if (!state) {
		return SPIRES_ERR_ALLOC;
	}

	/* Each run gets a private netlist so concurrent callers cannot collide.
	 */
	char netlist_path[] = "/tmp/spires_crossbar_XXXXXX";
	int netlist_fd = mkstemp(netlist_path);
	if (netlist_fd < 0 || close(netlist_fd) != 0) {
		free(state);
		unlink(netlist_path);
		return SPIRES_ERR_INTERNAL;
	}
	/* libngspice keeps process-global state, so independent readouts must
	 * not initialize or drive it concurrently. */
	pthread_mutex_lock(&ngspice_mutex);

	// starting the crossbar
	if (spires_reservoir_reset(reservoir) != SPIRES_OK ||
	    online_crossbar_init(config, reservoir, netlist_path,
				 &crossbar) != 0 ||
	    online_crossbar_start(crossbar) != 0) {
		free(state);
		online_crossbar_destroy(crossbar);
		unlink(netlist_path);
		pthread_mutex_unlock(&ngspice_mutex);
		return SPIRES_ERR_INTERNAL;
	}

	size_t num_inputs = spires_num_inputs(reservoir);
	for (size_t timestep = 0; timestep < series_length; timestep++) {
		const double *input = input_series + timestep * num_inputs;
		int output_ready = 0;
		/* The background simulation returns output[t-1] while state[t]
		 * is being published, avoiding a simulation restart per
		 * timestep. */
		double *previous =
		    timestep == 0
			? NULL
			: buffer + (timestep - 1) * config->num_outputs;
		if (spires_step(reservoir, input) != SPIRES_OK ||
		    spires_read_reservoir_state(reservoir, state) !=
			SPIRES_OK ||
		    online_crossbar_submit(crossbar, timestep, state, previous,
					   &output_ready) != 0 ||
		    output_ready != (timestep != 0)) {
			fprintf(stderr,
				"Crossbar readout failed at timestep %zu\n",
				timestep);
			free(state);
			online_crossbar_destroy(crossbar);
			unlink(netlist_path);
			pthread_mutex_unlock(&ngspice_mutex);
			return SPIRES_ERR_INTERNAL;
		}
	}

	int status = online_crossbar_finish(
	    crossbar, buffer + (series_length - 1) * config->num_outputs);
	free(state);
	online_crossbar_destroy(crossbar);
	unlink(netlist_path);
	pthread_mutex_unlock(&ngspice_mutex);
	return status == 0 ? SPIRES_OK : SPIRES_ERR_INTERNAL;
}

double *run_crossbar_readout(const spires_crossbar_readout_config *config,
			     spires_reservoir *reservoir,
			     const double *input_series, size_t series_length)
{
	if (!config || series_length == 0 || config->num_outputs == 0 ||
	    series_length > SIZE_MAX / config->num_outputs ||
	    series_length * config->num_outputs > SIZE_MAX / sizeof(double)) {
		return NULL;
	}

	double *buffer =
	    malloc(series_length * config->num_outputs * sizeof(*buffer));
	if (!buffer) {
		return NULL;
	}

	if (run_crossbar_readout_into(config, reservoir, input_series, series_length,
				 buffer) != SPIRES_OK) {
		free(buffer);
		return NULL;
	}

	return buffer;
}

/********** ngSpice callback functions **********/
// ignores random ngspice output, unless its an error
static int callback_text(char *text, int ident, void *user_data)
{
	(void)ident;
	(void)user_data;

	if (text &&
	    (strncmp(text, "stderr", 6) == 0 || strstr(text, "Error") != NULL ||
	     strstr(text, "error") != NULL)) {
		fprintf(stderr, "ngspice: %s\n", text);
	}

	return 0;
}

// ignores ngspice vector information that the readout does not need
static int callback_init_data(pvecinfoall values, int ident, void *user_data)
{
	(void)values;
	(void)ident;
	(void)user_data;
	return 0;
}

// marks the simulation as failed and wakes any waiting threads
static void fail_locked(Online_Crossbar *crossbar)
{
	crossbar->failed = 1;
	pthread_cond_broadcast(&crossbar->state_available);
	pthread_cond_broadcast(&crossbar->progress);
}

// marks the simulation as failed if ngspice exits unexpectedly
static int callback_exit(int status, NG_BOOL immediate, NG_BOOL quit_exit,
			 int ident, void *user_data)
{
	(void)status;
	(void)immediate;
	(void)quit_exit;
	(void)ident;

	Online_Crossbar *crossbar = user_data;
	pthread_mutex_lock(&crossbar->mutex);
	fail_locked(crossbar);
	pthread_mutex_unlock(&crossbar->mutex);
	return 0;
}

// records when the ngspice background simulation finishes
static int callback_background(NG_BOOL running, int ident, void *user_data)
{
	(void)ident;
	Online_Crossbar *crossbar = user_data;

	/* libngspice passes false at worker start and true at worker exit. */
	if (!running) {
		return 0;
	}

	pthread_mutex_lock(&crossbar->mutex);
	crossbar->simulation_done = 1;
	if (crossbar->result_count < crossbar->config.num_timesteps) {
		fail_locked(crossbar);
	}

	pthread_cond_broadcast(&crossbar->progress);
	pthread_cond_broadcast(&crossbar->state_available);
	pthread_mutex_unlock(&crossbar->mutex);

	return 0;
}

// gets the crossbar row number from an ngspice source name
static int row_from_name(const char *name, size_t *row_out)
{
	const char *digits = name + strlen(name);
	while (digits > name && isdigit((unsigned char)digits[-1])) {
		digits--;
	}

	if (*digits == '\0') {
		return -1;
	}

	char *end = NULL;
	unsigned long value = strtoul(digits, &end, 10);
	if (end == digits || *end != '\0') {
		return -1;
	}

	*row_out = (size_t)value;

	return 0;
}

// converts the current ngspice simulation time into a reservoir timestep
static size_t timestep_for_time(const Online_Crossbar *crossbar, double time)
{
	if (time <= 0.0) {
		return 0;
	}

	double scaled = time / crossbar->config.time_step;

	/* Hold state[t] over the full interval (t*dt, (t+1)*dt]. */
	double interval = ceil(scaled - 1.0e-9);

	size_t timestep = interval <= 1.0 ? 0 : (size_t)interval - 1;
	if (timestep >= crossbar->config.num_timesteps) {
		timestep = crossbar->config.num_timesteps - 1;
	}

	return timestep;
}

// gives ngspice the reservoir voltage for a requested crossbar row
static int callback_voltage(double *voltage, double time, char *name, int ident,
			    void *user_data)
{
	(void)ident;
	Online_Crossbar *crossbar = user_data;
	size_t row;

	if (!name || row_from_name(name, &row) != 0 ||
	    row >= crossbar->config.num_neurons) {
		fprintf(stderr, "ngspice: unrecognized external source %s\n",
			name ? name : "(null)");
		return 1;
	}

	size_t timestep = timestep_for_time(crossbar, time);
	pthread_mutex_lock(&crossbar->mutex);

	while (!crossbar->failed && crossbar->published_count <= timestep) {
		pthread_cond_wait(&crossbar->state_available, &crossbar->mutex);
	}

	if (crossbar->failed) {
		pthread_mutex_unlock(&crossbar->mutex);
		return 1;
	}

	*voltage =
	    crossbar->config.row_voltage_scaler *
	    crossbar->state_ring[(timestep % 2) * crossbar->config.num_neurons +
				 row];

	if (crossbar->consumed_count < timestep + 1) {
		crossbar->consumed_count = timestep + 1;
		pthread_cond_broadcast(&crossbar->progress);
	}

	pthread_mutex_unlock(&crossbar->mutex);
	return 0;
}

// collects and decodes crossbar output voltages at each timestep
static int callback_data(pvecvaluesall values, int count, int ident,
			 void *user_data)
{
	(void)count;
	(void)ident;
	Online_Crossbar *crossbar = user_data;
	double time = -1.0;
	double *columns =
	    calloc(crossbar->config.num_outputs * 2, sizeof(*columns));
	unsigned char *found =
	    calloc(crossbar->config.num_outputs * 2, sizeof(*found));

	if (!columns || !found) {
		free(columns);
		free(found);
		pthread_mutex_lock(&crossbar->mutex);
		fail_locked(crossbar);
		pthread_mutex_unlock(&crossbar->mutex);
		return 1;
	}

	/* Extract the sample time and each indexed crossbar column voltage from
	 * ngspice's vector batch. */
	for (int i = 0; i < values->veccount; i++) {
		pvecvalues value = values->vecsa[i];
		if (value->is_scale || strcmp(value->name, "time") == 0) {
			time = value->creal;
			continue;
		}
		const char *col = strstr(value->name, "col");
		if (!col) {
			continue;
		}
		char *end = NULL;
		unsigned long index = strtoul(col + 3, &end, 10);
		if (end != col + 3 &&
		    index < crossbar->config.num_outputs * 2) {
			columns[index] = value->creal;
			found[index] = 1;
		}
	}

	if (time >= 0.0) {
		double scaled = time / crossbar->config.time_step;
		double rounded = nearbyint(scaled);

		if (fabs(scaled - rounded) <= 1.0e-7 && rounded >= 1.0 &&
		    (size_t)rounded <= crossbar->config.num_timesteps) {
			size_t timestep = (size_t)rounded - 1;

			int complete = 1;
			for (size_t i = 0; i < crossbar->config.num_outputs * 2;
			     i++) {
				complete = complete && found[i];
			}

			if (complete) {
				double *decoded =
				    crossbar->results +
				    timestep * crossbar->config.num_outputs;

				if (decode_crossbar_output(
					crossbar->config.num_neurons,
					crossbar->config.num_outputs, 1,
					columns, crossbar->resistances,
					crossbar->config.load_resistance,
					&crossbar->mapping,
					crossbar->config.row_voltage_scaler,
					decoded) == 0) {
					pthread_mutex_lock(&crossbar->mutex);

					if (crossbar->result_count <
					    timestep + 1) {
						crossbar->result_count =
						    timestep + 1;
					}

					if (timestep + 1 <
					    crossbar->config.num_timesteps) {
						ngSpice_SetBkpt(
						    (timestep + 2) *
						    crossbar->config.time_step);
					}

					pthread_cond_broadcast(
					    &crossbar->progress);
					pthread_mutex_unlock(&crossbar->mutex);
				}
			}
		}
	}

	free(columns);
	free(found);
	return 0;
}

/********** Crossbar Control **********/
static int config_is_valid(const spires_crossbar_readout_config *config)
{
	if (!config || config->num_neurons == 0 || config->num_outputs == 0 ||
	    config->num_timesteps == 0 || !isfinite(config->time_step) ||
	    config->time_step <= 0.0 || !isfinite(config->row_voltage_scaler) ||
	    config->row_voltage_scaler == 0.0 ||
	    !isfinite(config->load_resistance) ||
	    config->load_resistance <= 0.0 || !isfinite(config->r_on) ||
	    !isfinite(config->r_off) || config->r_on <= 0.0 ||
	    config->r_off <= config->r_on ||
	    !crossbar_model_get(config->model)) {
		return 0;
	}

	if (config->num_outputs > SIZE_MAX / 2 ||
	    config->num_neurons > SIZE_MAX / 2 ||
	    config->num_timesteps > SIZE_MAX / config->num_outputs ||
	    config->num_timesteps * config->num_outputs >
		SIZE_MAX / sizeof(double)) {
		return 0;
	}

	double stop_time = config->num_timesteps * config->time_step;
	return isfinite(stop_time) && stop_time > 0.0;
}

// allocates the online crossbar and generates its ngspice netlist
static int online_crossbar_init(const spires_crossbar_readout_config *config,
				const spires_reservoir *reservoir,
				const char *netlist_path,
				Online_Crossbar **crossbar_out)
{
	if (!crossbar_out) {
		return -1;
	}
	*crossbar_out = NULL;
	if (!reservoir || !netlist_path || !config_is_valid(config)) {
		return -1;
	}

	Online_Crossbar *crossbar = calloc(1, sizeof(*crossbar));
	if (!crossbar) {
		return -1;
	}
	crossbar->config = *config;
	crossbar->netlist_path = netlist_path;
	if (pthread_mutex_init(&crossbar->mutex, NULL) != 0) {
		goto fail;
	}
	crossbar->mutex_initialized = 1;
	if (pthread_cond_init(&crossbar->state_available, NULL) != 0) {
		goto fail;
	}
	crossbar->state_condition_initialized = 1;
	if (pthread_cond_init(&crossbar->progress, NULL) != 0) {
		goto fail;
	}
	crossbar->progress_condition_initialized = 1;

	crossbar->state_ring = calloc(2 * config->num_neurons, sizeof(double));
	crossbar->results =
	    calloc(config->num_timesteps * config->num_outputs, sizeof(double));
	if (!crossbar->state_ring || !crossbar->results) {
		goto fail;
	}
	if (convert_weights_to_resistances(
		reservoir, config->num_neurons, config->num_outputs,
		config->r_on, config->r_off, &crossbar->resistances,
		&crossbar->mapping) != 0) {
		goto fail;
	}

	if (generate_crossbar_netlist(&crossbar->config, crossbar->netlist_path,
				      crossbar->resistances) != 0) {
		goto fail;
	}

	*crossbar_out = crossbar;
	return 0;

fail:
	online_crossbar_destroy(crossbar);
	return -1;
}

// starts the persistent ngspice simulation in the background
static int online_crossbar_start(Online_Crossbar *crossbar)
{
	if (!crossbar || crossbar->started) {
		return -1;
	}

	/* Register callbacks before loading the netlist so ngspice can request
	 * reservoir states as soon as the transient analysis begins. */
	if (ngSpice_Init(callback_text, callback_text, callback_exit,
			 callback_data, callback_init_data, callback_background,
			 crossbar) != 0) {
		return -1;
	}

	if (ngSpice_Init_Sync(callback_voltage, NULL, NULL, NULL, crossbar) !=
	    0) {
		ngSpice_Reset();
		return -1;
	}

	char command[4096];
	int written = snprintf(command, sizeof(command), "source %s",
			       crossbar->netlist_path);
	if (written < 0 || (size_t)written >= sizeof(command) ||
	    ngSpice_Command(command) != 0) {
		ngSpice_Reset();
		return -1;
	}
	crossbar->started = 1;

	/* Breakpoints force result callbacks at the exact sample boundaries
	 * used by the reservoir, rather than relying on ngspice's adaptive
	 * stepping. */
	ngSpice_SetBkpt(crossbar->config.time_step);
	if (ngSpice_Command("bg_run") != 0) {
		ngSpice_Reset();
		crossbar->started = 0;
		return -1;
	}
	return 0;
}

// submits one reservoir state and returns the previous timestep's output
static int online_crossbar_submit(Online_Crossbar *crossbar, size_t timestep,
				  const double *state, double *previous_output,
				  int *output_ready)
{
	if (!crossbar || !state || !output_ready || !crossbar->started ||
	    timestep >= crossbar->config.num_timesteps ||
	    timestep != crossbar->published_count ||
	    (timestep != 0 && !previous_output)) {
		return -1;
	}

	pthread_mutex_lock(&crossbar->mutex);

	/* Keep two states in flight at most: the ring has two slots, and
	 * allowing a third publication would overwrite data ngspice may still
	 * be reading. */
	while (!crossbar->failed && timestep >= crossbar->consumed_count + 2) {
		pthread_cond_wait(&crossbar->progress, &crossbar->mutex);
	}

	if (crossbar->failed) {
		pthread_mutex_unlock(&crossbar->mutex);
		return -1;
	}

	memcpy(crossbar->state_ring +
		   (timestep % 2) * crossbar->config.num_neurons,
	       state, crossbar->config.num_neurons * sizeof(double));
	crossbar->published_count++;
	pthread_cond_broadcast(&crossbar->state_available);

	*output_ready = timestep != 0;
	if (timestep != 0) {

		/* The current state is consumed asynchronously; wait until
		 * ngspice has decoded the preceding sample before exposing that
		 * output. */
		while (!crossbar->failed && crossbar->result_count < timestep) {
			pthread_cond_wait(&crossbar->progress,
					  &crossbar->mutex);
		}

		if (crossbar->failed) {
			pthread_mutex_unlock(&crossbar->mutex);
			return -1;
		}

		memcpy(previous_output,
		       crossbar->results +
			   (timestep - 1) * crossbar->config.num_outputs,
		       crossbar->config.num_outputs * sizeof(double));
	}

	pthread_mutex_unlock(&crossbar->mutex);
	return 0;
}

// waits for and returns the final crossbar output
static int online_crossbar_finish(Online_Crossbar *crossbar,
				  double *final_output)
{
	if (!crossbar || !final_output ||
	    crossbar->published_count != crossbar->config.num_timesteps) {
		return -1;
	}
	pthread_mutex_lock(&crossbar->mutex);

	/* Completion is reported asynchronously, so the caller must wait for
	 * all requested samples before reading the final output. */
	while (!crossbar->failed &&
	       crossbar->result_count < crossbar->config.num_timesteps) {
		pthread_cond_wait(&crossbar->progress, &crossbar->mutex);
	}

	if (crossbar->failed) {
		pthread_mutex_unlock(&crossbar->mutex);
		return -1;
	}

	memcpy(final_output,
	       crossbar->results + (crossbar->config.num_timesteps - 1) *
				       crossbar->config.num_outputs,
	       crossbar->config.num_outputs * sizeof(double));
	pthread_mutex_unlock(&crossbar->mutex);
	return 0;
}

// stops ngspice and frees all online crossbar resources
static void online_crossbar_destroy(Online_Crossbar *crossbar)
{
	if (!crossbar) {
		return;
	}

	if (crossbar->started && ngSpice_running()) {
		/* Stop the worker before freeing callback state; otherwise
		 * ngspice could access the buffers after they are released. */
		ngSpice_Command("bg_halt");
		if (crossbar->mutex_initialized) {
			pthread_mutex_lock(&crossbar->mutex);
			while (!crossbar->simulation_done &&
			       ngSpice_running()) {
				pthread_cond_wait(&crossbar->progress,
						  &crossbar->mutex);
			}
			pthread_mutex_unlock(&crossbar->mutex);
		}
	}

	if (crossbar->started) {
		/* Reset only after the worker has stopped so libngspice
		 * releases its references to the netlist and callback context
		 * safely. */
		ngSpice_Reset();
	}

	free(crossbar->resistances);
	free(crossbar->state_ring);
	free(crossbar->results);

	if (crossbar->state_condition_initialized) {
		pthread_cond_destroy(&crossbar->state_available);
	}
	if (crossbar->progress_condition_initialized) {
		pthread_cond_destroy(&crossbar->progress);
	}
	if (crossbar->mutex_initialized) {
		pthread_mutex_destroy(&crossbar->mutex);
	}

	free(crossbar);
}
