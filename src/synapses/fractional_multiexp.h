#ifndef SYNAPSE_FRACTIONAL_MULTIEXP_H
#define SYNAPSE_FRACTIONAL_MULTIEXP_H

#include <stdlib.h>

/* SYNAPSE_FRACTIONAL_MULTIEXP: efficient O(N)-per-step approximation of
 * SYNAPSE_FRACTIONAL's Mittag-Leffler impulse response, built from a weighted
 * sum of N ordinary single-exponential filters (same O(1)-per-component math
 * as SYNAPSE_PSC_HOMOGENEOUS, just N of them). Weights derived from the exact
 * Mittag-Leffler spectral (Bernstein) density,
 *
 *   E_alpha(-(t/tau_syn)^alpha) = integral_0^inf exp(-s*t) * rho(s) ds
 *
 * discretized over N log-spaced timescales in [tau_min, tau_max] -- not a
 * heuristic power-law weighting, the actual closed-form density for this
 * function. Expect close tracking of the true SYNAPSE_FRACTIONAL curve within
 * [tau_min, tau_max] and visible divergence outside that window (a finite sum
 * of exponentials can't reproduce arbitrarily early curvature or an infinite
 * tail) -- that's the intended "true vs. approximate" comparison.
 *
 * params[0] = alpha (0,1) -- note: alpha very close to 1 is a degenerate case
 *             (the exact Mittag-Leffler spectral density collapses to a
 *             single delta function, since E_1 is just one exponential) and
 *             is handled as a special case, not a numerical quadrature.
 * params[1] = tau_syn
 * params[2] = tau_min
 * params[3] = tau_max
 * params[4] = N (number of exponential components) */
struct fractional_multiexp_synapse_data;

struct fractional_multiexp_synapse_data *synapse_fractional_multiexp_build_sparse(const double *dense, size_t n, const double *params);
struct fractional_multiexp_synapse_data *synapse_fractional_multiexp_build_dense(const double *dense, size_t n, const double *params);
void   synapse_fractional_multiexp_free(struct fractional_multiexp_synapse_data *d);
void   synapse_fractional_multiexp_to_dense(const struct fractional_multiexp_synapse_data *d, double *dense_out);
double synapse_fractional_multiexp_row_dot(const struct fractional_multiexp_synapse_data *d, size_t row, const double *x);
void   synapse_fractional_multiexp_scale(struct fractional_multiexp_synapse_data *d, double factor);
double synapse_fractional_multiexp_spectral_radius(const struct fractional_multiexp_synapse_data *d);
const double *synapse_fractional_multiexp_prepare(struct fractional_multiexp_synapse_data *d, const double *spikes, double dt);

#endif // SYNAPSE_FRACTIONAL_MULTIEXP_H
