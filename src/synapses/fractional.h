#ifndef SYNAPSE_FRACTIONAL_H
#define SYNAPSE_FRACTIONAL_H

#include <stdlib.h>

/* SYNAPSE_FRACTIONAL: exact GL-discretized fractional relaxation equation for
 * synaptic current: tau_syn * D^alpha(I) = -I + spike(t) (Caputo/GL fractional
 * derivative, symmetric convention -- both terms scaled by 1/tau_syn, unlike
 * flif_gl.c's own asymmetric convention, so alpha=1 collapses to exactly the
 * existing area-normalized SYNAPSE_PSC_HOMOGENEOUS filter rather than an
 * unnormalized one -- see synapse module docs / plan for the derivation).
 *
 * The impulse response of this equation is the classical Mittag-Leffler
 * relaxation function E_alpha(-(t/tau_syn)^alpha): reduces to the ordinary
 * exponential at alpha=1, has a genuine power-law tail for large t, and a
 * distinct stretched-exponential-like onset for small t -- richer than a
 * hand-shaped power-law kernel, because it's the impulse response of an
 * actual fractional-order ODE, not a curve-fit.
 *
 * params[0] = alpha (0,1]
 * params[1] = tau_syn
 * params[2] = T_mem (finite memory-window truncation; mem_len = T_mem/dt,
 *             capped -- purely a numerical fidelity/cost tradeoff, mirrors
 *             flif_gl's T_mem role) */
struct fractional_synapse_data;

struct fractional_synapse_data *synapse_fractional_build_sparse(const double *dense, size_t n, const double *params);
struct fractional_synapse_data *synapse_fractional_build_dense(const double *dense, size_t n, const double *params);
void   synapse_fractional_free(struct fractional_synapse_data *d);
void   synapse_fractional_to_dense(const struct fractional_synapse_data *d, double *dense_out);
double synapse_fractional_row_dot(const struct fractional_synapse_data *d, size_t row, const double *x);
void   synapse_fractional_scale(struct fractional_synapse_data *d, double factor);
double synapse_fractional_spectral_radius(const struct fractional_synapse_data *d);
const double *synapse_fractional_prepare(struct fractional_synapse_data *d, const double *spikes, double dt);

#endif // SYNAPSE_FRACTIONAL_H
