#include "psc_heterogeneous.h"
#include <string.h>
#include <math.h>

struct psc_heterogeneous_synapse_data {
    int is_sparse;
    size_t n;
    size_t nnz;         // nnz if sparse, n*n if dense
    size_t *row_ptr;    // n+1, NULL if dense
    size_t *col_idx;    // nnz,  NULL if dense
    double *weights;    // nnz (sparse) or n*n (dense)
    double *tau;        // parallel to weights, sampled per edge at build time
    double *trace;      // parallel to weights, mutable each micro-step
};

static inline double urand01(void)
{
    return (double)rand() / (double)RAND_MAX;
}

/* tau ~ log-uniform on [tau_min, tau_max]: log(tau) is uniform, so tau spans
 * the range evenly on a log scale rather than over-representing the high end
 * the way a plain linear Uniform[tau_min, tau_max] draw would -- appropriate
 * since synaptic/membrane time constants realistically span orders of
 * magnitude. */
static inline double sample_log_uniform_tau(double tau_min, double tau_max)
{
    double log_min = log(tau_min);
    double log_max = log(tau_max);
    return exp(log_min + urand01() * (log_max - log_min));
}

static struct psc_heterogeneous_synapse_data *build(const double *dense, size_t n,
                                                     const double *params, int is_sparse)
{
    double tau_min = params ? params[0] : 1.0;
    double tau_max = params ? params[1] : 1.0;

    struct psc_heterogeneous_synapse_data *d = malloc(sizeof(*d));
    if (!d)
        return NULL;
    d->is_sparse = is_sparse;
    d->n = n;
    d->row_ptr = NULL;
    d->col_idx = NULL;

    if (is_sparse) {
        size_t nnz = 0;
        for (size_t k = 0; k < n * n; k++)
            if (dense[k] != 0.0)
                nnz++;

        d->nnz = nnz;
        d->row_ptr = malloc((n + 1) * sizeof(size_t));
        d->col_idx = malloc(nnz * sizeof(size_t));
        d->weights = malloc(nnz * sizeof(double));
        d->tau     = malloc(nnz * sizeof(double));
        d->trace   = calloc(nnz, sizeof(double));

        size_t idx = 0;
        for (size_t i = 0; i < n; i++) {
            d->row_ptr[i] = idx;
            const double *row = &dense[i * n];
            for (size_t j = 0; j < n; j++) {
                if (row[j] != 0.0) {
                    d->col_idx[idx] = j;
                    d->weights[idx] = row[j];
                    d->tau[idx] = sample_log_uniform_tau(tau_min, tau_max);
                    idx++;
                }
            }
        }
        d->row_ptr[n] = idx;
    } else {
        d->nnz = n * n;
        d->weights = malloc(n * n * sizeof(double));
        d->tau     = malloc(n * n * sizeof(double));
        d->trace   = calloc(n * n, sizeof(double));
        for (size_t k = 0; k < n * n; k++) {
            d->weights[k] = dense[k];
            d->tau[k] = sample_log_uniform_tau(tau_min, tau_max);
        }
    }

    return d;
}

struct psc_heterogeneous_synapse_data *synapse_psc_heterogeneous_build_sparse(const double *dense, size_t n,
                                                                               const double *params)
{
    return build(dense, n, params, 1);
}

struct psc_heterogeneous_synapse_data *synapse_psc_heterogeneous_build_dense(const double *dense, size_t n,
                                                                              const double *params)
{
    return build(dense, n, params, 0);
}

void synapse_psc_heterogeneous_free(struct psc_heterogeneous_synapse_data *d)
{
    if (!d)
        return;
    free(d->row_ptr);
    free(d->col_idx);
    free(d->weights);
    free(d->tau);
    free(d->trace);
    free(d);
}

void synapse_psc_heterogeneous_to_dense(const struct psc_heterogeneous_synapse_data *d, double *dense_out)
{
    if (d->is_sparse) {
        memset(dense_out, 0, d->n * d->n * sizeof(double));
        for (size_t i = 0; i < d->n; i++)
            for (size_t k = d->row_ptr[i]; k < d->row_ptr[i + 1]; k++)
                dense_out[i * d->n + d->col_idx[k]] = d->weights[k];
    } else {
        memcpy(dense_out, d->weights, d->n * d->n * sizeof(double));
    }
}

double synapse_psc_heterogeneous_row_dot(const struct psc_heterogeneous_synapse_data *d, size_t row, const double *x)
{
    (void)x; /* ignored: this type reads its own internally-maintained trace */
    double sum = 0.0;
    if (d->is_sparse) {
        size_t start = d->row_ptr[row], end = d->row_ptr[row + 1];
        for (size_t k = start; k < end; k++)
            sum += d->weights[k] * d->trace[k];
    } else {
        const double *wrow = &d->weights[row * d->n];
        const double *trow = &d->trace[row * d->n];
        for (size_t j = 0; j < d->n; j++)
            sum += wrow[j] * trow[j];
    }
    return sum;
}

void synapse_psc_heterogeneous_scale(struct psc_heterogeneous_synapse_data *d, double factor)
{
    for (size_t k = 0; k < d->nnz; k++)
        d->weights[k] *= factor;
}

/* Power iteration over weights only (tau/trace aren't connection strengths).
 * This is now the third near-identical copy of this algorithm (see also
 * math_utils.c:calc_spectral_radius and sparse.c:csr_spectral_radius) --
 * a reasonable candidate to consolidate behind a shared matvec-callback
 * helper if a fourth caller ever shows up. */
static void matvec(const struct psc_heterogeneous_synapse_data *d, const double *x, double *y)
{
    if (d->is_sparse) {
        for (size_t i = 0; i < d->n; i++) {
            double sum = 0.0;
            for (size_t k = d->row_ptr[i]; k < d->row_ptr[i + 1]; k++)
                sum += d->weights[k] * x[d->col_idx[k]];
            y[i] = sum;
        }
    } else {
        for (size_t i = 0; i < d->n; i++) {
            double sum = 0.0;
            const double *wrow = &d->weights[i * d->n];
            for (size_t j = 0; j < d->n; j++)
                sum += wrow[j] * x[j];
            y[i] = sum;
        }
    }
}

double synapse_psc_heterogeneous_spectral_radius(const struct psc_heterogeneous_synapse_data *d)
{
    const unsigned int max_iter = 1000;
    const double tol = 1e-6;
    size_t n = d->n;

    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++)
        x[i] = 1.0;

    double lambda_old = 0.0, lambda_new = 0.0;
    for (unsigned int iter = 0; iter < max_iter; iter++) {
        matvec(d, x, y);

        lambda_new = 0.0;
        for (size_t i = 0; i < n; i++) {
            double a = fabs(y[i]);
            if (a > lambda_new)
                lambda_new = a;
        }
        for (size_t i = 0; i < n; i++)
            x[i] = y[i] / lambda_new;

        if (fabs(lambda_new - lambda_old) < tol)
            break;
        lambda_old = lambda_new;
    }

    free(x);
    free(y);
    return lambda_new;
}

const double *synapse_psc_heterogeneous_prepare(struct psc_heterogeneous_synapse_data *d, const double *spikes, double dt)
{
    /* Normalized (DC gain 1 per edge) for the same reason as
     * psc_homogeneous.c's _prepare -- see the comment there. */
    if (d->is_sparse) {
        #pragma omp parallel for
        for (size_t i = 0; i < d->n; i++) {
            for (size_t k = d->row_ptr[i]; k < d->row_ptr[i + 1]; k++) {
                double decay = exp(-dt / d->tau[k]);
                d->trace[k] = d->trace[k] * decay + (1.0 - decay) * spikes[d->col_idx[k]];
            }
        }
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < d->n; i++) {
            for (size_t j = 0; j < d->n; j++) {
                size_t k = i * d->n + j;
                double decay = exp(-dt / d->tau[k]);
                d->trace[k] = d->trace[k] * decay + (1.0 - decay) * spikes[j];
            }
        }
    }
    return NULL; /* unused: row_dot reads internal trace directly */
}
