#include "fractional_multiexp.h"
#include "simple.h"
#include <math.h>
#include <string.h>

#define FRAC_PI 3.14159265358979323846

struct fractional_multiexp_synapse_data {
    struct simple_synapse_data *weights;
    size_t n;
    int N;
    double *tau;      /* N */
    double *c;        /* N, normalized weights (sum to 1) */
    double *traces;   /* N * n, per-component per-neuron trace */
    double *combined; /* n, weighted sum returned by prepare */
};

static struct fractional_multiexp_synapse_data *build_common(struct simple_synapse_data *weights,
                                                              size_t n, const double *params)
{
    if (!weights)
        return NULL;

    double alpha   = params ? params[0] : 0.5;
    double tau_syn = params ? params[1] : 1.0;
    double tau_min = params ? params[2] : 1.0;
    double tau_max = params ? params[3] : 20.0;
    int N = params ? (int)(params[4] + 0.5) : 8;
    if (N < 1) N = 1;

    struct fractional_multiexp_synapse_data *d = malloc(sizeof(*d));
    if (!d) {
        synapse_simple_free(weights);
        return NULL;
    }
    d->weights = weights;
    d->n = n;
    d->N = N;
    d->tau = malloc((size_t)N * sizeof(double));
    d->c   = malloc((size_t)N * sizeof(double));

    /* Degenerate case: alpha essentially 1 means the true Mittag-Leffler
     * spectral density collapses to a single delta function at tau_syn
     * (E_1 is a single exponential, not a continuous spectrum) --
     * sin(alpha*pi) is ~0 there, so the quadrature below would divide by
     * ~0. Handle it honestly as one component rather than producing NaNs. */
    double sin_apia = sin(alpha * FRAC_PI);
    if (fabs(sin_apia) < 1e-9) {
        d->N = 1;
        d->tau[0] = tau_syn;
        d->c[0] = 1.0;
        d->traces = calloc(n, sizeof(double));
        d->combined = calloc(n, sizeof(double));
        return d;
    }

    double log_tau_min = log(tau_min);
    double log_tau_max = log(tau_max);
    double d_log_tau = (N > 1) ? (log_tau_max - log_tau_min) / (N - 1) : 0.0;
    double cos_apia = cos(alpha * FRAC_PI);

    double raw[N];
    double sum_raw = 0.0;
    for (int i = 0; i < N; i++) {
        double log_tau_i = log_tau_min + i * d_log_tau;
        double tau_i = exp(log_tau_i);
        double x = tau_syn / tau_i;
        double K = (1.0 / FRAC_PI) * pow(x, alpha - 1.0) * sin_apia /
                   (pow(x, 2.0 * alpha) + 2.0 * pow(x, alpha) * cos_apia + 1.0);
        double w = K * x * d_log_tau; /* |d(log x)| = d_log_tau */
        if (w < 0.0) w = 0.0;         /* guard tiny negative numerical noise */
        /* The theory's quadrature weight `w` multiplies an *unnormalized*
         * exp(-t/tau_i) (area = tau_i). Each component filter below is
         * instead individually area-normalized to 1 (same convention as
         * psc_homogeneous.c), which is equivalent to (1/tau_i)*exp(-t/tau_i)
         * in the continuum limit -- so the mixing coefficient must be scaled
         * by tau_i to compensate, or longer-tau_i components get
         * systematically under-weighted relative to what the spectral
         * density calls for. */
        raw[i] = w * tau_i;
        d->tau[i] = tau_i;
        sum_raw += raw[i];
    }
    for (int i = 0; i < N; i++)
        d->c[i] = (sum_raw > 0.0) ? raw[i] / sum_raw : 1.0 / N;

    d->traces = calloc((size_t)N * n, sizeof(double));
    d->combined = calloc(n, sizeof(double));
    return d;
}

struct fractional_multiexp_synapse_data *synapse_fractional_multiexp_build_sparse(const double *dense, size_t n,
                                                                                   const double *params)
{
    return build_common(synapse_simple_build_sparse(dense, n), n, params);
}

struct fractional_multiexp_synapse_data *synapse_fractional_multiexp_build_dense(const double *dense, size_t n,
                                                                                  const double *params)
{
    return build_common(synapse_simple_build_dense(dense, n), n, params);
}

void synapse_fractional_multiexp_free(struct fractional_multiexp_synapse_data *d)
{
    if (!d)
        return;
    synapse_simple_free(d->weights);
    free(d->tau);
    free(d->c);
    free(d->traces);
    free(d->combined);
    free(d);
}

void synapse_fractional_multiexp_to_dense(const struct fractional_multiexp_synapse_data *d, double *dense_out)
{
    synapse_simple_to_dense(d->weights, dense_out);
}

double synapse_fractional_multiexp_row_dot(const struct fractional_multiexp_synapse_data *d, size_t row, const double *x)
{
    return synapse_simple_row_dot(d->weights, row, x);
}

void synapse_fractional_multiexp_scale(struct fractional_multiexp_synapse_data *d, double factor)
{
    synapse_simple_scale(d->weights, factor);
}

double synapse_fractional_multiexp_spectral_radius(const struct fractional_multiexp_synapse_data *d)
{
    return synapse_simple_spectral_radius(d->weights);
}

const double *synapse_fractional_multiexp_prepare(struct fractional_multiexp_synapse_data *d, const double *spikes, double dt)
{
    int N = d->N;
    size_t n = d->n;

    #pragma omp parallel for
    for (int c = 0; c < N; c++) {
        double decay = exp(-dt / d->tau[c]);
        double *trace_c = &d->traces[(size_t)c * n];
        for (size_t i = 0; i < n; i++)
            trace_c[i] = trace_c[i] * decay + (1.0 - decay) * spikes[i];
    }

    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (int c = 0; c < N; c++)
            sum += d->c[c] * d->traces[(size_t)c * n + i];
        d->combined[i] = sum;
    }

    return d->combined;
}
