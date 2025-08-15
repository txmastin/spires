#ifndef FLIF_DIFFUSIVE_H
#define FLIF_DIFFUSIVE_H

// flif_diffusive (double-precision): power-law refractory neuron (no fractional derivative).
// Alpha ∈ [0,1] controls the refractory law:
//   α = 1 → deterministic refractory t_ref (standard LIF with absolute refractory)
//   α = 0 → uniform refractory on [t_min, t_max]
//   0<α<1 → mixture: point mass at t_ref with weight m(α)=α^γ and truncated power-law tail with slope q(α)=QMAX·α
//
// Params layout:
//   params[0] = V_th
//   params[1] = V_reset
//   params[2] = V_rest
//   params[3] = tau_m
//   params[4] = alpha
//   params[5] = T_mem_or_tref
//                 - if α<1: upper truncation t_max for refractory sampling (if <=0, default chosen)
//                 - if α≥1: deterministic refractory t_ref (seconds; if <=0, no refractory)
//   params[6] = bias
//
// Compile-time knobs (override with -D flags as needed):
#ifndef FLIF_DIFF_TMIN_MULT
#define FLIF_DIFF_TMIN_MULT 1.0    // t_min = MULT * dt
#endif
#ifndef FLIF_DIFF_TMAX_MULT
#define FLIF_DIFF_TMAX_MULT 100.0  // fallback t_max when params[6] <= 0 in α<1 mode
#endif
#ifndef FLIF_DIFF_GAMMA
#define FLIF_DIFF_GAMMA 2.0        // collapse weight m(α) = α^γ
#endif
#ifndef FLIF_DIFF_QMAX
#define FLIF_DIFF_QMAX 6.0         // power-law slope q(α) = QMAX * α  (q=0 → uniform)
#endif

#ifndef MAX_MEM_LEN
#define MAX_MEM_LEN 20000          // parity with flif_gl
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

struct flif_diffusive_neuron {
    // Parameters (double precision)
    double V_th, V_reset, V_rest, tau_m, alpha, bias;

    // State
    double V;
    double spike;        // 0.0 or 1.0

    // Time bookkeeping
    long   internal_step;
    int    mem_len;      // derived horizon in steps (parity with flif_gl)

    // Refractory gate
    long   refractory_until;  // step index (exclusive) until which spikes are suppressed
    double t_min;             // lower truncation (≈ dt * FLIF_DIFF_TMIN_MULT) in seconds
    double t_max;             // upper truncation (α<1 mode) in seconds
    double t_ref;             // deterministic refractory (α≥1 mode) in seconds

    // Cached Euler gain
    double g_euler;           // dt / tau_m

    // RNG state (SplitMix64)
    unsigned long long rng;
};

// API (double throughout)
struct flif_diffusive_neuron* init_flif_diffusive(double* params, double dt);
void   update_flif_diffusive(struct flif_diffusive_neuron* n, double input, double dt);
void   free_flif_diffusive(struct flif_diffusive_neuron* n);

// Optional deterministic seeding for reproducibility
void   flif_diffusive_set_seed(struct flif_diffusive_neuron* n, unsigned long long seed);

#ifdef __cplusplus
}
#endif

#endif // FLIF_DIFFUSIVE_H
