/*
 * AGILE: Adaptive Gradient-Informed Lévy Exploration
 * Reference C API (header)
 *
 * Explore with Lévy-biased steps until patience is exhausted, then
 * restore best-so-far and run deterministic refinement (GD or SPSA).
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SPIRES_AGILE_H
#define SPIRES_AGILE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Options controlling AGILE behavior. */
struct agile_options {
	/* Lévy / exploration */
	double mu;              /* Lévy exponent in (1,3] */
	double eta_min;         /* min step size (normalized space) */
	double eta_max;         /* max step size (normalized space) */
	double patience0;       /* initial exploration budget (steps) */
	double patience_decay;  /* alpha in (0,1): patience *= alpha on improve */

	/* Refinement (deterministic polishing) */
	double refine_step;     /* step size for refinement (default: eta_min) */
	int    refine_max_iters;/* cap on refinement iterations */
	int    refine_no_improve; /* stop if no improvement for this many iters */

	/* SPSA (used when no gradient is provided) */
	double spsa_eps;        /* small perturbation for SPSA (normalized coords) */

	/* Bounds & projection */
	int    reflect_at_bounds; /* 0 = clamp, 1 = reflect */

	/* RNG and logging */
	uint64_t seed;          /* PRNG seed for reproducibility */
	int      verbose;       /* >0 prints progress to stderr */
};

/* Objective: return loss value; lower is better. */
typedef double (*agile_loss_fn)(const double *theta, int dim, void *ctx);

/* Optional gradient: return 0 on success; non-zero means fallback to SPSA. */
typedef int (*agile_grad_fn)(const double *theta, int dim, double *grad_out, void *ctx);

/* Summary of a run. */
struct agile_report {
	double best_loss;
	int    best_iter;       /* iteration where best was found */
	int    total_evals;     /* number of objective evaluations performed */
	int    total_iters;     /* iterations including refinement */
	int    exited_refinement; /* 1 if refinement phase was entered */
};

/*
 * Optimize.
 *
 * Inputs:
 *   - theta0:      initial point (length dim) in caller's parameter space
 *   - lower/upper: per-coordinate bounds (same space as theta0)
 *   - loss_fn:     objective (deterministic under ctx+seed)
 *   - grad_fn:     optional gradient (NULL → SPSA used as proxy)
 *   - ctx:         user context pointer passed to callbacks
 *   - opts:        options (NULL → internal defaults)
 *
 * Outputs:
 *   - theta_best:  best point found (same space as theta0)
 *   - report:      run statistics (optional; can be NULL)
 *
 * Notes:
 *   * For strictly-positive variables, pass a transformed coordinate
 *     (e.g., log10(lambda)) so that bounds are linear.
 */
int agile_optimize(
	const double *theta0,
	const double *lower,
	const double *upper,
	int dim,
	agile_loss_fn loss_fn,
	agile_grad_fn grad_fn,
	void *ctx,
	const struct agile_options *opts,
	double *theta_best,
	struct agile_report *report
);

#ifdef __cplusplus
}
#endif

#endif /* SPIRES_AGILE_H */
