#include "fractional.h"
#include "simple.h"
#include <math.h>
#include <string.h>

#ifndef FRAC_MAX_MEM_LEN
#define FRAC_MAX_MEM_LEN 20000
#endif

struct fractional_synapse_data {
    struct simple_synapse_data *weights;
    size_t n;
    double alpha;
    double tau_syn;
    double T_mem;
    int mem_len;         /* 0 until lazily initialized on the first _prepare call
                           * (dt isn't known at build time) */
    double *coeffs;      /* mem_len, shared GL derivative coefficients */
    double *I;           /* n, current synaptic current per neuron (returned) */
    double *I_history;   /* n * mem_len, circular buffer of I's own past values */
    long internal_step;
};

static struct fractional_synapse_data *build_common(struct simple_synapse_data *weights,
                                                     size_t n, const double *params)
{
    if (!weights)
        return NULL;
    struct fractional_synapse_data *d = malloc(sizeof(*d));
    if (!d) {
        synapse_simple_free(weights);
        return NULL;
    }
    d->weights = weights;
    d->n = n;
    d->alpha   = params ? params[0] : 1.0;
    d->tau_syn = params ? params[1] : 1.0;
    d->T_mem   = params ? params[2] : 10.0;
    d->mem_len = 0;
    d->coeffs = NULL;
    d->I_history = NULL;
    d->internal_step = 0;
    d->I = calloc(n, sizeof(double));
    return d;
}

struct fractional_synapse_data *synapse_fractional_build_sparse(const double *dense, size_t n,
                                                                  const double *params)
{
    return build_common(synapse_simple_build_sparse(dense, n), n, params);
}

struct fractional_synapse_data *synapse_fractional_build_dense(const double *dense, size_t n,
                                                                 const double *params)
{
    return build_common(synapse_simple_build_dense(dense, n), n, params);
}

void synapse_fractional_free(struct fractional_synapse_data *d)
{
    if (!d)
        return;
    synapse_simple_free(d->weights);
    free(d->coeffs);
    free(d->I);
    free(d->I_history);
    free(d);
}

void synapse_fractional_to_dense(const struct fractional_synapse_data *d, double *dense_out)
{
    synapse_simple_to_dense(d->weights, dense_out);
}

double synapse_fractional_row_dot(const struct fractional_synapse_data *d, size_t row, const double *x)
{
    return synapse_simple_row_dot(d->weights, row, x);
}

void synapse_fractional_scale(struct fractional_synapse_data *d, double factor)
{
    synapse_simple_scale(d->weights, factor);
}

double synapse_fractional_spectral_radius(const struct fractional_synapse_data *d)
{
    return synapse_simple_spectral_radius(d->weights);
}

/* Same recursion as neuron.c:compute_gl_coeffs (GL fractional-DERIVATIVE
 * coefficients, alternating sign for k>=1) -- duplicated locally rather than
 * exposing that internal helper across modules. Correct for a derivative;
 * do NOT confuse with a fractional-integral kernel (which would need a
 * different, all-positive recursion and would not represent an actual
 * relaxation ODE). */
static void compute_gl_derivative_coeffs(double *coeffs, double alpha, int N)
{
    coeffs[0] = 1.0;
    for (int k = 1; k < N; k++)
        coeffs[k] = coeffs[k - 1] * (1.0 - (alpha + 1.0) / (double)k);
}

const double *synapse_fractional_prepare(struct fractional_synapse_data *d, const double *spikes, double dt)
{
    if (d->mem_len == 0) {
        int mem_len = (d->T_mem > 0 && dt > 0) ? (int)(d->T_mem / dt) : 2000;
        if (mem_len > FRAC_MAX_MEM_LEN) mem_len = FRAC_MAX_MEM_LEN;
        if (mem_len < 1) mem_len = 1;
        d->mem_len = mem_len;
        d->coeffs = malloc((size_t)mem_len * sizeof(double));
        compute_gl_derivative_coeffs(d->coeffs, d->alpha, mem_len);
        d->I_history = calloc(d->n * (size_t)mem_len, sizeof(double));
    }

    int mem_len = d->mem_len;
    double dt_alpha = pow(dt, d->alpha);
    long step = d->internal_step;
    int head = (int)(step % mem_len);
    int limit = (step < mem_len) ? (int)step : mem_len - 1;

    #pragma omp parallel for
    for (size_t i = 0; i < d->n; i++) {
        double *hist = &d->I_history[i * (size_t)mem_len];
        int prev_idx = (head - 1 + mem_len) % mem_len;
        double I_prev = hist[prev_idx];

        double history_term = 0.0;
        for (int k = 1; k <= limit; k++) {
            int idx = (head - k + mem_len) % mem_len;
            history_term += d->coeffs[k] * hist[idx];
        }

        double rhs = (-I_prev + spikes[i]) / d->tau_syn;
        double I_new = dt_alpha * rhs - history_term;

        hist[head] = I_new;
        d->I[i] = I_new;
    }

    d->internal_step++;
    return d->I;
}
