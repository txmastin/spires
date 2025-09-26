#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* Library include */
#include "spires.h"

/* Private backend headers */
#include "reservoir.h"
#include "neuron.h"
#include "spires_opt_agile.h"


/* The public opaque handle wraps a backend pointer. */
struct spires_reservoir {
    struct reservoir *impl; /* owned */
};

/* --------------- lifecycle --------------- */
spires_status spires_reservoir_create(const spires_reservoir_config *cfg,
                                      spires_reservoir **out_r)
{
    if (!cfg || !out_r)
        return SPIRES_ERR_INVALID_ARG;

    /* Validate dt once, up front (strict: 1.0 must be an integer multiple of dt) */
    if (cfg->dt <= 0.0 || cfg->dt > 1.0) {
        fprintf(stderr, "dt must be in (0, 1.0]\n");
        return SPIRES_ERR_INVALID_ARG;
    }
    double steps_d = 1.0 / cfg->dt;
    int steps = (int)llround(steps_d);
    if (steps < 1 || fabs(steps_d - (double)steps) > 1e-12) {
        fprintf(stderr, "To ensure proper time simulation, dt must evenly divide 1.0 (1/dt≈%.12f)\n", steps_d);
        return SPIRES_ERR_INVALID_ARG;
    }

    enum connectivity_type conn = (enum connectivity_type)cfg->connectivity_type;
    enum neuron_type ntype = (enum neuron_type)cfg->neuron_type;

    struct reservoir *impl = create_reservoir(cfg->num_neurons,
                                              cfg->num_inputs,
                                              cfg->num_outputs,
                                              cfg->spectral_radius,
                                              cfg->ei_ratio,
                                              cfg->input_strength,
                                              cfg->connectivity,
                                              cfg->dt,           /* same dt flows to backend */
                                              conn,
                                              ntype,
                                              cfg->neuron_params);
    if (!impl)
        return SPIRES_ERR_INTERNAL;

    if (init_reservoir(impl) != 0) {
        free_reservoir(impl);
        return SPIRES_ERR_INTERNAL;
    }

    spires_reservoir *r = malloc(sizeof(*r));
    if (!r) {
        free_reservoir(impl);
        return SPIRES_ERR_ALLOC;
    }
    r->impl = impl;
    *out_r  = r;
    return SPIRES_OK;
}

void spires_reservoir_destroy(spires_reservoir *r)
{
    if (!r)
        return;
    if (r->impl)
        free_reservoir(r->impl);
    free(r);
}

spires_status spires_reservoir_reset(spires_reservoir *r)
{
    if (!r || !r->impl)
        return SPIRES_ERR_INVALID_ARG;
    reset_reservoir(r->impl);
    return SPIRES_OK;
}

/* --------------- stepping --------------- */
spires_status spires_step(spires_reservoir *r, const double *u_t)
{
    if (!r || !r->impl)
        return SPIRES_ERR_INVALID_ARG;

    double *tmp = NULL;
    if (u_t) {
        tmp = malloc(sizeof(double) * r->impl->num_inputs);
        if (!tmp)
            return SPIRES_ERR_ALLOC;
        memcpy(tmp, u_t, sizeof(double) * r->impl->num_inputs);
    }
    step_reservoir(r->impl, tmp);
    if (tmp)
        free(tmp);
    return SPIRES_OK;
}


/* --------------- running ---------------- */
double *spires_run(spires_reservoir *r, const double *input_series, size_t series_length)
{
    if (!r || !r->impl || !input_series)
        return NULL;
    /* backend function returns malloc'd predictions; caller frees */
    return run_reservoir(r->impl, (double *)input_series, series_length);
}



/* --------------- training --------------- */
spires_status spires_train_online(spires_reservoir *r,
                                  const double *target_vec, double lr)
{
    if (!r || !r->impl || !target_vec)
        return SPIRES_ERR_INVALID_ARG;
    /* backend function expects non-const; we know it only reads */
    train_output_iteratively(r->impl, (double *)target_vec, lr);
    return SPIRES_OK;
}

spires_status spires_train_ridge(spires_reservoir *r,
                                 const double *input_series,
                                 const double *target_series,
                                 size_t series_length, double lambda)
{
    if (!r || !r->impl || !input_series || !target_series)
        return SPIRES_ERR_INVALID_ARG;
    train_output_ridge_regression(r->impl,
                                  (double *)input_series,
                                  (double *)target_series,
                                  series_length, lambda);
    return SPIRES_OK;
}

/* --------------- state --------------- */
double *spires_copy_reservoir_state(spires_reservoir *r)
{
    if (!r || !r->impl)
        return NULL;
    /* backend returns malloc'd buffer; caller must free */
    return copy_reservoir_state(r->impl);
}

spires_status spires_read_reservoir_state(spires_reservoir *r, double *buffer)
{
    if (!r || !r->impl || !buffer) {
        fprintf(stderr, "Error reading reservoir state, buffer not initialized!\n");
        return SPIRES_ERR_INVALID_ARG;
    }
    read_reservoir_state(r->impl, buffer);
    return SPIRES_OK;
}

spires_status spires_compute_output(spires_reservoir *r, double *out)
{
    if (!r || !r->impl || !out)
        return SPIRES_ERR_INVALID_ARG;

    /* backend compute_output(struct reservoir*, double* out) -> 0 on success */
    if (compute_output(r->impl, out) != 0)
        return SPIRES_ERR_INTERNAL;

    return SPIRES_OK;
}

/* --------------- introspection --------------- */
size_t spires_num_neurons(const spires_reservoir *r)
{
    return (r && r->impl) ? r->impl->num_neurons : 0;
}

size_t spires_num_inputs(const spires_reservoir *r)
{
    return (r && r->impl) ? r->impl->num_inputs : 0;
}

size_t spires_num_outputs(const spires_reservoir *r)
{
    return (r && r->impl) ? r->impl->num_outputs : 0;
}

/* ---------------- optimization ---------------- */
extern int spires_core_evaluate_config(const spires_reservoir_config *cfg,
                                       double log10_ridge,
                                       double data_fraction,
                                       int num_seeds,
                                       double time_limit_sec,
                                       double *metric_mean, double *metric_std,
                                       double *cost_proxy, int *feasible);

static size_t snap_N(double x) 
{
    static const size_t choices[] = {200, 400, 800, 1200};
    size_t best = choices[0]; double bestd = fabs(x - (double)choices[0]);
    for (size_t i = 1; i < sizeof(choices)/sizeof(choices[0]); i++) {
        double d = fabs(x - (double)choices[i]);
        if (d < bestd) { bestd = d; best = choices[i]; }
    }
    return best;
}
static int snap_topo(double t)
{
    int k = (int)llround(t);
    if (k < 0) k = 0; if (k > 2) k = 2;
    return k;
}

struct opt_ctx
{
    const spires_reservoir_config *base;  /* not owned */
    const double *x;                      /* input data */
    const double *y;                      /* target series */
    size_t n;                             /* series length */
    int alpha_index;                      /* where α lives in neuron_params (e.g., 0) */
};

static double eval_mse(spires_reservoir *r,
		       const double *x, const double *y,
		       size_t off, size_t len,
		       size_t din, size_t dout)
{
	double *yhat = (double *)malloc(sizeof(double) * (dout ? dout : 1));
	if (!yhat)
		return INFINITY;

	const double *x_base = x + off * din;
	const double *y_base = y + off * dout;

	double se = 0.0;

	for (size_t t = 0; t < len; t++) {
		const double *u_t = x_base + t * din;

		/* advance one step */
		if (spires_step(r, u_t) != SPIRES_OK) {
			free(yhat);
			return INFINITY;
		}

		/* read current outputs (length = dout) */
		if (spires_compute_output(r, yhat) != SPIRES_OK) {
			free(yhat);
			return INFINITY;
		}

		/* accumulate squared error across outputs */
		const double *y_t = y_base + t * dout;
		for (size_t o = 0; o < dout; o++) {
			double e = y_t[o] - yhat[o];
			se += e * e;
		}
	}

	free(yhat);

	/* average over all time steps and outputs */
	if (len == 0 || dout == 0)
		return INFINITY;

	return se / (double)(len * dout);
}

/* adapter called by AGILE each time */
static int eval_adapter(const struct spires_opt_params *pr,
			const struct spires_budget *budget,
			void *user_ctx,
			double *metric_mean, double *metric_std,
			double *cost_proxy, int *feasible)
{
	const struct opt_ctx *ctx = (const struct opt_ctx *)user_ctx;

	/* clone and override tuned fields */
	spires_reservoir_config cfg = *ctx->base;
	cfg.num_neurons       = snap_N(pr->n_cont);
	cfg.spectral_radius   = pr->spectral_radius;
	cfg.input_strength    = pr->input_gain;
	cfg.connectivity      = pr->connectivity;
	cfg.ei_ratio          = pr->e_over_i;
	cfg.connectivity_type = (spires_connectivity_type) snap_topo(pr->topo_cont);

	double local_params[8] = {0};
	if (!cfg.neuron_params)
		cfg.neuron_params = local_params;
	cfg.neuron_params[ctx->alpha_index] = pr->alpha;

        const size_t din  = cfg.num_inputs  ? cfg.num_inputs  : 1;
        const size_t dout = cfg.num_outputs ? cfg.num_outputs : 1;

	/* split train/valid by data_fraction */
	size_t n_train = (size_t)((double)ctx->n * fmax(0.0, fmin(1.0, budget->data_fraction)));
	if (n_train < 8 || ctx->n <= n_train) {
		*feasible = 0;
		*metric_mean = 0.0;
		*metric_std  = 1.0;
		*cost_proxy  = 1.0;
		return 0;
	}
	size_t n_valid = ctx->n - n_train;

	const int seeds = budget->num_seeds > 0 ? budget->num_seeds : 1;
	double sum = 0.0, sum2 = 0.0;
	int ok = 0;

	for (int s = 0; s < seeds; s++) {
		spires_reservoir *r = NULL;
		if (spires_reservoir_create(&cfg, &r) != SPIRES_OK || !r)
			continue;

		/* train ridge on train slice */
		const double *x_train = ctx->x + 0;
		const double *y_train = ctx->y + 0;
		const double lambda = pow(10.0, pr->log10_ridge);

		if (spires_train_ridge(r, x_train, y_train, n_train, lambda) != SPIRES_OK) {
			spires_reservoir_destroy(r);
			continue;
		}

		/* evaluate on holdout */
		if (spires_reservoir_reset(r) != SPIRES_OK) {
			spires_reservoir_destroy(r);
			continue;
		}
		double mse = eval_mse(r, ctx->x, ctx->y, n_train, n_valid, din, dout);
		spires_reservoir_destroy(r);

		if (!isfinite(mse) || mse < 0.0)
			continue;

		/* convert to higher-is-better score */
		double metric = 1.0 / (1.0 + mse);
		sum  += metric;
		sum2 += metric * metric;
		ok++;
	}

	if (ok == 0) {
		*feasible = 0;
		*metric_mean = 0.0;
		*metric_std  = 1.0;
		*cost_proxy  = 1.0;
		return 0;
	}

	double mean = sum / (double)ok;
	double var  = fmax(0.0, (sum2 / (double)ok) - mean * mean);

	*metric_mean = mean;
	*metric_std  = sqrt(var);

	/* simple cost proxy: size × inverse sparsity */
	double sparsity = cfg.connectivity > 1e-8 ? cfg.connectivity : 1e-8;
	*cost_proxy = (double)cfg.num_neurons * (1.0 / sparsity) * 1e-6;

	*feasible = 1;
	return 0;
}

int spires_optimize(const spires_reservoir_config *base_config,
		    const struct spires_opt_budget *budgets_pub,
		    int num_budgets,
		    const struct spires_opt_score *score_pub,
		    struct spires_opt_result *out_pub,
                    const double *input_series,
                    const double *target_series,
                    size_t series_length)
{
        printf("[SPIRES] spires_optimize entered\n"); fflush(stdout);
	if (!base_config || !budgets_pub || num_budgets <= 0 || !score_pub || !out_pub)
		return -1;

	struct spires_budget *buds = (struct spires_budget *)calloc((size_t)num_budgets, sizeof(*buds));
	if (!buds) return -2;

	for (int i = 0; i < num_budgets; i++) {
		buds[i].data_fraction  = budgets_pub[i].data_fraction;
		buds[i].num_seeds      = budgets_pub[i].num_seeds;
		buds[i].time_limit_sec = budgets_pub[i].time_limit_sec;
	}

	struct spires_score_opts sopt = {
		.lambda_var  = score_pub->lambda_var,
		.lambda_cost = score_pub->lambda_cost,
		.use_auprc   = (score_pub->metric == SPIRES_METRIC_AUPRC)
	};

	struct spires_result ires = {0};
	struct opt_ctx ctx = {
            .base = base_config,
            .x = input_series,
            .y = target_series,
            .n = series_length,
            .alpha_index = 4
        }; 
        
        printf("[SPIRES] calling wrapper (budgets=%d)\n", num_budgets); fflush(stdout);

	int rc = spires_optimize_with_agile(eval_adapter, &ctx, buds, num_budgets, &sopt, &ires);
	free(buds);
	if (rc)
            return rc;

	/* map internal best to public result */
	out_pub->best_config = *base_config;
	out_pub->best_config.num_neurons       = snap_N(ires.best_params.n_cont);
	out_pub->best_config.spectral_radius   = ires.best_params.spectral_radius;
	out_pub->best_config.input_strength    = ires.best_params.input_gain;
	out_pub->best_config.connectivity      = ires.best_params.connectivity;
	out_pub->best_config.ei_ratio          = ires.best_params.e_over_i;
	out_pub->best_config.connectivity_type = (spires_connectivity_type)snap_topo(ires.best_params.topo_cont);

	out_pub->best_log10_ridge = ires.best_params.log10_ridge;
	out_pub->metric_mean      = ires.best_metric_mean;
	out_pub->metric_std       = ires.best_metric_std;
	out_pub->best_score       = ires.best_score;
	return 0;
}
