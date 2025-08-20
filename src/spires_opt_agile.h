/*
 * spires_opt_agile.h - SPIRES hyperparameter optimization via AGILE
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SPIRES_OPT_AGILE_H
#define SPIRES_OPT_AGILE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SPIRES hyperparameters (continuous / relaxed).
 * Note: n_cont and topo_cont are relaxed; the implementation snaps them
 * to discrete {200,400,800,1200} and {0,1,2} at evaluation time.
 */
struct spires_opt_params {
	double	alpha;		/* [0.20, 0.80] */
	double	spectral_radius;	/* [0.75, 1.05] */
	double	input_gain;		/* [0.20, 1.50] */
	double	connectivity;		/* [0.08, 0.35] */
	double	e_over_i;		/* [0.70, 1.10] */
	double	log10_ridge;		/* [-5, -1] for 1e-5..1e-1 */
	double	n_cont;		/* relaxed neuron count; snaps to {200,400,800,1200} */
	double	topo_cont;	/* relaxed topology idx; snaps to {0=RND,1=SW,2=SF} */
};

/* Evaluation budget (multi-fidelity). */
struct spires_budget {
	double	data_fraction;	/* 0..1 */
	int	num_seeds;
	double	time_limit_sec;	/* optional; 0 = no limit */
};

/* Result summary. */
struct spires_result {
	struct spires_opt_params  best_params;
	double			  best_score;		/* mean − λ_var*std − λ_cost*cost */
	double			  best_metric_mean;
	double			  best_metric_std;
};

/* User-provided evaluator.
 * Build/run a SPIRES reservoir and compute metrics for params p under 'budget'.
 * Return 0 on success; non-zero signals a fatal error (optimizer penalizes).
 * Set *feasible = 0 to indicate a health-check failure (divergence, etc.).
 */
typedef int (*spires_eval_fn)(const struct spires_opt_params *p,
			      const struct spires_budget *budget,
			      void *user_ctx,
			      /* outputs: */
			      double *metric_mean, double *metric_std,
			      double *cost_proxy, int *feasible);

/* Scoring options (robustness + cost penalties). */
struct spires_score_opts {
	double	lambda_var;	/* penalty multiplier for std */
	double	lambda_cost;	/* penalty multiplier for cost proxy */
	int	use_auprc;	/* 1 = metric is AUPRC; 0 = AUROC/other (informational) */
};

/* Top-level entrypoint: run AGILE across budgets and return best params.
 * Returns 0 on success.
 */
int spires_optimize_with_agile(spires_eval_fn eval,
			       void *user_ctx,
			       const struct spires_budget *budgets,
			       int num_budgets,
			       const struct spires_score_opts *score_opts,
			       struct spires_result *out);

#ifdef __cplusplus
}
#endif

#endif /* SPIRES_OPT_AGILE_H */

