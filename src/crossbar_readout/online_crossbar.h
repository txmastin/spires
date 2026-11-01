#ifndef ONLINE_CROSSBAR_H
#define ONLINE_CROSSBAR_H

#include <spires.h>

spires_status run_crossbar_readout_into(
    const spires_crossbar_readout_config *config, spires_reservoir *reservoir,
    const double *input_series, size_t series_length, double *buffer);

double *run_crossbar_readout(
    const spires_crossbar_readout_config *config, spires_reservoir *reservoir,
    const double *input_series, size_t series_length);

#endif
