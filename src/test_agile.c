#include "agile.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/* ---------- Quadratic bowl: f(theta) = sum_i (theta_i - 3)^2 ---------- */
static double quad_loss(const double *theta, int dim, void *ctx)
{
	int i;
	double s = 0.0;
	(void)ctx;
	for (i = 0; i < dim; i++) {
		double d = theta[i] - 3.0;
		s += d * d;
	}
	return s;
}

static int quad_grad(const double *theta, int dim, double *g_out, void *ctx)
{
	int i;
	(void)ctx;
	for (i = 0; i < dim; i++)
		g_out[i] = 2.0 * (theta[i] - 3.0);
	return 0;
}

/* Assert helpers */
static void expect(int cond, const char *msg)
{
	if (!cond) {
		fprintf(stderr, "ASSERTION FAILED: %s\n", msg);
		assert(cond);
	}
}

static void run_test_gradient_mode(void)
{
	const int dim = 4;
	double lower[4], upper[4], theta0[4], theta_best[4];
	int i;

	for (i = 0; i < dim; i++) {
		lower[i] = -10.0;
		upper[i] =  10.0;
		theta0[i] = 0.0;
	}

	struct agile_options opt = {
		.mu = 1.3,
		.eta_min = 1e-3,
		.eta_max = 0.3,
		.patience0 = 20.0,
		.patience_decay = 0.85,
		.refine_step = 1e-2,
		.refine_max_iters = 1000,
		.refine_no_improve = 50,
		.spsa_eps = 1e-3,
		.reflect_at_bounds = 1,
		.seed = 0xDEADBEEFULL,
		.verbose = 0
	};
	struct agile_report rep = {0};

	int rc = agile_optimize(theta0, lower, upper, dim,
				quad_loss, quad_grad, NULL,
				&opt, theta_best, &rep);
	expect(rc == 0, "agile_optimize returned error in gradient mode");
	expect(rep.exited_refinement == 1, "did not enter refinement in gradient mode");
	expect(rep.best_iter <= rep.total_iters, "best_iter exceeds total_iters");

	double best_loss = quad_loss(theta_best, dim, NULL);
	expect(best_loss < 1e-3, "best_loss not sufficiently small in gradient mode");

	/* Check proximity to optimum (3.0 per coord) */
	for (i = 0; i < dim; i++)
		expect(fabs(theta_best[i] - 3.0) < 1e-2, "theta_best not close to 3.0 in gradient mode");

	printf("[OK] gradient mode: best_loss=%.6g, total_evals=%d, total_iters=%d\n",
	       best_loss, rep.total_evals, rep.total_iters);
}

static void run_test_spsa_mode(void)
{
	const int dim = 3;
	double lower[3], upper[3], theta0[3], theta_best[3];
	int i;

	for (i = 0; i < dim; i++) {
		lower[i] = 0.0;
		upper[i] = 5.0;
		theta0[i] = (i == 0) ? 0.1 : 0.0; /* near bound to test projection */
	}

	struct agile_options opt = {
		.mu = 1.5,
		.eta_min = 1e-3,
		.eta_max = 0.2,
		.patience0 = 25.0,
		.patience_decay = 0.9,
		.refine_step = 5e-3,
		.refine_max_iters = 1500,
		.refine_no_improve = 80,
		.spsa_eps = 1e-3,
		.reflect_at_bounds = 1,
		.seed = 123456u,
		.verbose = 0
	};
	struct agile_report rep = {0};

	int rc = agile_optimize(theta0, lower, upper, dim,
				quad_loss, NULL, NULL,
				&opt, theta_best, &rep);
	expect(rc == 0, "agile_optimize returned error in SPSA mode");
	expect(rep.exited_refinement == 1, "did not enter refinement in SPSA mode");
	expect(rep.best_iter <= rep.total_iters, "best_iter exceeds total_iters (SPSA mode)");

	double best_loss = quad_loss(theta_best, dim, NULL);
	expect(best_loss < 5e-3, "best_loss not sufficiently small in SPSA mode");

	for (i = 0; i < dim; i++) {
		expect(theta_best[i] >= lower[i] - 1e-12 &&
		       theta_best[i] <= upper[i] + 1e-12,
		       "theta_best outside bounds in SPSA mode");
	}

	printf("[OK] SPSA mode: best_loss=%.6g, total_evals=%d, total_iters=%d\n",
	       best_loss, rep.total_evals, rep.total_iters);
}

int main(void)
{
	run_test_gradient_mode();
	run_test_spsa_mode();

	printf("All AGILE unit tests passed.\n");
	return 0;
}
