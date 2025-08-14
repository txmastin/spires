#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "flif_diffusive.h"

// ---------------- RNG (SplitMix64) ----------------
static inline uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97f4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// 53-bit uniform in [0,1)
static inline double u01(uint64_t *state) {
    uint64_t r = splitmix64_next(state);
    return (r >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

// Inverse-CDF sampler for truncated power t^{-q} on [tmin,tmax].
// q=0 → uniform, q=1 → log-uniform, general q handled by closed form.
static inline double sample_trunc_power(double tmin, double tmax, double q, double u){
    if (tmax <= tmin) return tmin;
    if (q <= 1e-12) { // uniform
        return tmin + u * (tmax - tmin);
    }
    if (fabs(q - 1.0) < 1e-12) { // log-uniform
        return tmin * pow(tmax / tmin, u);
    }
    const double a = 1.0 - q;
    const double A = pow(tmin, a), B = pow(tmax, a);
    return pow((1.0 - u) * A + u * B, 1.0 / a);
}

// Alpha-controlled refractory sampler (single knob).
// alpha in [0,1]:
//   m(alpha) = alpha^FLIF_DIFF_GAMMA  → probability of deterministic refractory t_ref (delta mass)
//   q(alpha) = FLIF_DIFF_QMAX * alpha → tail slope for truncated power on [tmin,tmax]
static inline double sample_alpha_refractory(double tmin, double tmax,
                                             double alpha, double t_ref,
                                             uint64_t *rng)
{
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;

    const double m = pow(alpha, (double)FLIF_DIFF_GAMMA);
    const double q = (double)FLIF_DIFF_QMAX * alpha;

    double u0 = u01(rng);
    if (u0 < m) return t_ref; // collapse to deterministic refractory as alpha→1

    double u1 = u01(rng);
    return sample_trunc_power(tmin, tmax, q, u1);
}

static inline long steps_from_time(double W, double dt){
    if (W <= 0.0) return 0;
    long s = (long)ceil(W / dt);
    return (s < 0) ? 0 : s;
}

// ---------------- Public API ----------------

// params: [V_th, V_reset, V_rest, tau_m, alpha, dt, T_mem_or_tref, bias]
struct flif_diffusive_neuron* init_flif_diffusive(double* params)
{
    if (!params) return NULL;
    struct flif_diffusive_neuron* n = (struct flif_diffusive_neuron*)malloc(sizeof *n);
    if (!n) return NULL;

    n->V_th   = params[0];
    n->V_reset= params[1];
    n->V_rest = params[2];
    n->tau_m  = params[3];
    n->alpha  = params[4];
    n->dt     = params[5];
    double P6 = params[6];   // T_mem_or_tref
    n->bias   = params[7];

    // Guards
    if (n->tau_m <= 0.0) n->tau_m = 1.0;
    if (n->dt    <= 0.0) n->dt    = 1e-3;
    if (n->alpha <  0.0) n->alpha = 0.0;
    if (n->alpha >  1.0) n->alpha = 1.0;

    n->V = n->V_rest;
    n->spike = 0.0;
    n->internal_step = 0;
    n->refractory_until = -1;

    // Cached Euler gain for LIF subthreshold
    n->g_euler = n->dt / n->tau_m;

    if (n->alpha < 1.0) {
        // Heavy-tailed refractory mode
        n->t_min = (double)FLIF_DIFF_TMIN_MULT * n->dt;
        n->t_max = (P6 > 0.0) ? P6 : (double)FLIF_DIFF_TMAX_MULT * n->tau_m;
        if (n->t_max <= n->t_min) n->t_max = n->t_min * 2.0;
        n->t_ref = n->dt; // default point for mixture branch

        // mem_len parity with flif_gl: span horizon over dt
        int mem_len = (int)(n->t_max / n->dt);
        if (mem_len < 2) mem_len = 2;
        if (mem_len > MAX_MEM_LEN) mem_len = MAX_MEM_LEN;
        n->mem_len = mem_len;
    } else {
        // Deterministic refractory (standard LIF with optional absolute refractory)
        n->t_ref = (P6 > 0.0) ? P6 : 0.0; // 0 → no absolute refractory
        n->t_min = (double)FLIF_DIFF_TMIN_MULT * n->dt; // defined for completeness
        n->t_max = n->t_ref;
        n->mem_len = (int)((n->t_ref > 0.0) ? (n->t_ref / n->dt) : 2);
        if (n->mem_len < 2) n->mem_len = 2;
        if (n->mem_len > MAX_MEM_LEN) n->mem_len = MAX_MEM_LEN;
    }

    // Seed RNG deterministically from pointer + params (use 64-bit chunks of the doubles)
    uint64_t seed = 0x9E3779B97f4A7C15ULL ^ (uintptr_t)n;
    for (int i=0;i<8;i++) {
        uint64_t bits = 0;
        memcpy(&bits, &params[i], sizeof(bits));
        seed ^= bits + 0x9E3779B97f4A7C15ULL;
        seed = splitmix64_next(&seed);
    }
    n->rng = seed;

    return n;
}

void flif_diffusive_set_seed(struct flif_diffusive_neuron* n, unsigned long long seed)
{
    if (!n) return;
    if (seed == 0ULL) seed = 0xD1B54A32D192ED03ULL;
    n->rng = (uint64_t)seed;
    (void)splitmix64_next(&n->rng);
    (void)splitmix64_next(&n->rng);
}

void update_flif_diffusive(struct flif_diffusive_neuron* n, double input, double dt)
{
    if (!n) return;

    // Clear spike flag
    if (n->spike == 1.0) n->spike = 0.0;

    // If dt changes, update caches/bounds (no reallocations)
    if (dt > 0.0 && dt != n->dt) {
        n->dt = dt;
        n->g_euler = n->dt / n->tau_m;
        if (n->alpha < 1.0) {
            n->t_min = (double)FLIF_DIFF_TMIN_MULT * n->dt;
            int mem_len = (int)(n->t_max / n->dt);
            if (mem_len < 2) mem_len = 2;
            if (mem_len > MAX_MEM_LEN) mem_len = MAX_MEM_LEN;
            n->mem_len = mem_len;
        } else {
            n->mem_len = (int)((n->t_ref > 0.0) ? (n->t_ref / n->dt) : 2);
            if (n->mem_len < 2) n->mem_len = 2;
            if (n->mem_len > MAX_MEM_LEN) n->mem_len = MAX_MEM_LEN;
        }
    }

    // Subthreshold Euler LIF using previous V
    double V_prev = n->V;
    double rhs = -((double)V_prev - (double)n->V_rest) + input + n->bias;
    double Vn  = V_prev + n->g_euler * rhs;
    n->V = Vn;

    // Spike gate
    int can_spike = (n->internal_step >= n->refractory_until);
    if (can_spike && n->V >= n->V_th) {
        // Spike + reset
        n->V = n->V_reset;
        n->spike = 1.0;

        long steps = 0;
        if (n->alpha < 1.0) {
            // Alpha-controlled mixture refractory
            double W = sample_alpha_refractory(n->t_min, n->t_max, n->alpha, n->t_ref, (uint64_t*)&n->rng);
            steps = steps_from_time(W, n->dt);
            if (steps < 1) steps = 1; // enforce at least one step
        } else {
            // Deterministic refractory (standard LIF mode)
            steps = steps_from_time(n->t_ref, n->dt);
            // steps may be zero if t_ref==0 → immediate eligibility next step
        }
        n->refractory_until = n->internal_step + steps;
    }

    n->internal_step++;
}

void free_flif_diffusive(struct flif_diffusive_neuron* n)
{
    if (!n) return;
    free(n);
}
