#include <stdlib.h>
#include <string.h>

#include "spires.h"

/* Private backend headers */
#include "reservoir.h"
#include "neuron.h"

/* The public opaque handle just wraps a backend pointer. */
struct spires_reservoir {
    struct reservoir *impl; /* owned */
};

/* --------------- lifecycle --------------- */
spires_status spires_reservoir_create(const spires_reservoir_config *cfg,
                                      spires_reservoir **out_r)
{
    if (!cfg || !out_r)
        return SPIRES_ERR_INVALID_ARG;

    enum connectivity_type conn = (enum connectivity_type)cfg->connectivity_type;
    enum neuron_type ntype = (enum neuron_type)cfg->neuron_type;

    struct reservoir *impl = create_reservoir(cfg->num_neurons,
                                              cfg->num_inputs,
                                              cfg->num_outputs,
                                              cfg->spectral_radius,
                                              cfg->ei_ratio,
                                              cfg->input_strength,
                                              cfg->connectivity,
                                              cfg->dt,
                                              conn,
                                              ntype,
                                              cfg->neuron_params);
    if (!impl)
        return SPIRES_ERR_INTERNAL;

    if (init_reservoir(impl) != 0) {
        free_reservoir(impl);
        return SPIRES_ERR_INTERNAL;
    }

    spires_reservoir *r = malloc(sizeof(*r));
    if (!r) {
        free_reservoir(impl);
        return SPIRES_ERR_ALLOC;
    }
    r->impl = impl;
    *out_r = r;
    return SPIRES_OK;
}

void spires_reservoir_destroy(spires_reservoir *r)
{
    if (!r)
        return;
    if (r->impl)
        free_reservoir(r->impl);
    free(r);
}

spires_status spires_reservoir_reset(spires_reservoir *r)
{
    if (!r || !r->impl)
        return SPIRES_ERR_INVALID_ARG;
    reset_reservoir(r->impl);
    return SPIRES_OK;
}

/* --------------- stepping --------------- */
spires_status spires_step(spires_reservoir *r, const double *u_t)
{
    if (!r || !r->impl)
        return SPIRES_ERR_INVALID_ARG;

    double *tmp = NULL;
    if (u_t) {
        tmp = malloc(sizeof(double) * r->impl->num_inputs);
        if (!tmp)
            return SPIRES_ERR_ALLOC;
        memcpy(tmp, u_t, sizeof(double) * r->impl->num_inputs);
    }
    step_reservoir(r->impl, tmp);
    if (tmp)
        free(tmp);
    return SPIRES_OK;
}

/* --------------- training --------------- */
spires_status spires_train_online(spires_reservoir *r,
                                  const double *target_vec, double lr)
{
    if (!r || !r->impl || !target_vec)
        return SPIRES_ERR_INVALID_ARG;
    /* backend function expects non-const; we know it only reads */
    train_output_iteratively(r->impl, (double *)target_vec, lr);
    return SPIRES_OK;
}

spires_status spires_train_ridge(spires_reservoir *r,
                                 const double *input_series,
                                 const double *target_series,
                                 size_t series_length, double lambda)
{
    if (!r || !r->impl || !input_series || !target_series)
        return SPIRES_ERR_INVALID_ARG;
    train_output_ridge_regression(r->impl,
                                  (double *)input_series,
                                  (double *)target_series,
                                  series_length, lambda);
    return SPIRES_OK;
}

/* --------------- state --------------- */
double *spires_read_state_copy(spires_reservoir *r)
{
    if (!r || !r->impl)
        return NULL;
    /* backend returns malloc'd buffer; caller must free */
    return read_reservoir_state(r->impl);
}

/* --------------- introspection --------------- */
size_t spires_num_neurons(const spires_reservoir *r)
{
    return (r && r->impl) ? r->impl->num_neurons : 0;
}

size_t spires_num_inputs(const spires_reservoir *r)
{
    return (r && r->impl) ? r->impl->num_inputs : 0;
}

size_t spires_num_outputs(const spires_reservoir *r)
{
    return (r && r->impl) ? r->impl->num_outputs : 0;
}
