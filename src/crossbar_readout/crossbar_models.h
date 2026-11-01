#ifndef CROSSBAR_MODELS_H
#define CROSSBAR_MODELS_H

#include <spires.h>

struct crossbar_model_definition {
	const char *subcircuit_name;
	const char *netlist;
};

const struct crossbar_model_definition *
crossbar_model_get(spires_crossbar_model model);

#endif
