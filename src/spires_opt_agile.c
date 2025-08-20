/*
 * spires_opt_agile.c - SPIRES hyperparameter optimization via AGILE
 *
 * SPDX-License-Identifier: MIT
 */

#include "spires_opt_agile.h"
#include "agile.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ---- Helpers: discrete projection for relaxed vars ---- */

static int round_N(double n_cont)
{
	/* Allowed sizes: 200, 400, 800, 1200 */
	static const int choices[] = { 200, 400, 800, 1200 };
	int i, best = choices[0];
	double bestd = fabs(n_cont - choices[0]);

	for (i = 1; i < (int)(sizeof(choices) / sizeof(choices[0])); i++) {
		double d = fabs(n_cont - choices[i]);

		if (d < bestd) {
			bestd = d;
			best = choices[i];
		}
	}
	return best;
}

static int round_topology(double t_cont)
{
	/* 0=RANDOM, 1=SMALL_WORLD, 2=SCALE_FREE */
	int idx = (int)llround(t_cont);

	if (idx < 0)
		idx = 0;
	if (idx > 2)
		idx = 2;
	return idx;
}

/* Build params from theta vector (order fixed below). */
static void theta_to_params(const double *th, struct spires_opt_params *p)
{
	p->alpha		= th[0];
	p->spectral_radius	= th[1];
	p->input_gain		= th[2];
	p->connectivity		= th[3];
	p->e_over_i		= th[4];
	p->log10_ridge		= th[5];
	p->n_cont		= th[6];
	p->topo_cont		= th[7];
}

/* Bounds for AGILE (same order as theta). */
static void fill_bounds(double *lo, double *hi)
{
	lo[0] = 0.20;  hi[0] = 0.80;  /* alpha */
	lo[1] = 0.75;  hi[1] = 1.05;  /* spectral_radius */
	lo[2] = 0.20;  hi[2] = 1.50;  /* input_gain */
	lo[3] = 0.08;  hi[3] = 0.35;  /* connectivity */
	lo[4] = 0.70;  hi[4] = 1.10;  /* e_over_i */
	lo[5] = -5.0;  hi[5] = -1.0;  /* log10_ridge: 1e-5..1e-1 */
	lo[6] = 200.0; hi[6] = 1200.0;/* N (relaxed) */
	lo[7] = 0.0;   hi[7] = 2.0;   /* topo idx (relaxed) */
}

/* Default start near your ball-park config. */
static void default_start(double *th0)
{
	th0[0] = 0.55;  /* alpha */
	th0[1] = 0.90;  /* rho */
	th0[2] = 1.00;  /* input_gain */
	th0[3] = 0.20;  /* connectivity */
	th0[4] = 0.80;  /* E:I */
	th0[5] = -2.0;  /* ridge = 1e-2 */
	th0[6] = 400.0; /* N */
	th0[7] = 0.0;   /* topo = RANDOM */
}

/* Score: mean − λ_var*std − λ_cost*cost. Higher is better. */
static double make_score(double mean, double std, double cost,
			 const struct spires_score_opts *sopt)
{
	/* sopt->use_auprc available if you want to branch weighting later */
	return mean - sopt->lambda_var * std - sopt->lambda_cost * cost;
}

/* Context passed to AGILE loss function. */
struct spires_obj_ctx {
	spires_eval_fn eval;
	void *user_ctx;
	const struct spires_budget *budget;
	const struct spires_score_opts *sopt;
};

/* Black-box loss for AGILE: returns negative score (AGILE minimizes). */
static double spires_loss_fn(const double *theta, int dim, void *ctxp)
{
	struct spires_obj_ctx *ctx = (struct spires_obj_ctx *)ctxp;
	struct spires_opt_params p;
	double mean = 0.0, std = 0.0, cost = 0.0;
	int feasible = 1;
	int rc;

	(void)dim;

	theta_to_params(theta, &p);

	/* Snap relaxed vars for the actual evaluation; AGILE still moves in R^d. */
	p.n_cont = round_N(p.n_cont);
	p.topo_cont = round_topology(p.topo_cont);

	rc = ctx->eval(&p, ctx->budget, ctx->user_ctx, &mean, &std, &cost, &feasible);
	if (rc != 0 || !feasible)
		return 1e6; /* heavy penalty */

	return -make_score(mean, std, cost, ctx->sopt);
}

/* Optional: if you have cheap partial derivatives for some coords, you can
 * implement a mixed gradient here. Otherwise AGILE will use SPSA.
 */
static int spires_grad_fn(const double *theta, int dim, double *g_out, void *ctxp)
{
	(void)theta;
	(void)dim;
	(void)g_out;
	(void)ctxp;
	return -1; /* force SPSA */
}

int spires_optimize_with_agile(spires_eval_fn eval,
			       void *user_ctx,
			       const struct spires_budget *budgets,
			       int num_budgets,
			       const struct spires_score_opts *score_opts,
			       struct spires_result *out)
{
	const int D = 8; /* number of decision variables in theta */
	double lo[D], hi[D], th0[D], th_best[D];
	struct agile_options aopt = {
		.mu = 1.3,
		.eta_min = 1e-3,
		.eta_max = 0.3,
		.patience0 = 100.0,
		.patience_decay = 0.9,
		.refine_step = 1e-2,
		.refine_max_iters = 600,
		.refine_no_improve = 60,
		.spsa_eps = 1e-3,
		.reflect_at_bounds = 1,
		.seed = 0xA91EULL,
		.verbose = 1
	};
        printf("[WRAP] spires_optimize_with_agile entered; aopt.verbose=%d\n", aopt.verbose); fflush(stdout);
	struct agile_report rep = { 0 };
	struct spires_obj_ctx ctx = {
		.eval = eval,
		.user_ctx = user_ctx,
		.budget = NULL,
		.sopt = score_opts
	};
	int i, rc = 0;

	if (!eval || !budgets || num_budgets <= 0 || !score_opts || !out)
		return -1;

	fill_bounds(lo, hi);
	default_start(th0);

	/* Run at successive budgets; keep promoting best-so-far. */
	for (i = 0; i < num_budgets; i++) {
		int rc2;

		ctx.budget = &budgets[i];
		rc2 = agile_optimize(th0, lo, hi, D,
				     spires_loss_fn, spires_grad_fn,
				     &ctx, &aopt, th_best, &rep);
		if (rc2 != 0) {
			rc = rc2;
			break;
		}
		/* Use the best from this stage as the start for the next. */
		memcpy(th0, th_best, sizeof(th0));
	}

	/* Decode best and assemble result. */
	if (rc == 0 && out) {
		struct spires_opt_params pbest;
		double mean = 0.0, std = 0.0, cost = 0.0;
		int feasible = 1;

		theta_to_params(th_best, &pbest);
		pbest.n_cont = round_N(pbest.n_cont);
		pbest.topo_cont = round_topology(pbest.topo_cont);

		/* Evaluate at full budget (last one). */
		ctx.budget = &budgets[num_budgets - 1];
		(void)eval(&pbest, ctx.budget, user_ctx, &mean, &std, &cost, &feasible);

		out->best_params = pbest;
		out->best_metric_mean = mean;
		out->best_metric_std = std;
		out->best_score = make_score(mean, std, cost, score_opts);
	}

	return rc;
}

