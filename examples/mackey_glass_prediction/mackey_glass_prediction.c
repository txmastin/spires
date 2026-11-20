/*
 * One-step-ahead prediction of the Mackey-Glass chaotic time series.
 *
 * A reservoir of Grunwald-Letnikov fractional LIF neurons is driven by x(t)
 * and a ridge readout is trained to predict x(t+1). The readout is fit on the
 * first N_TRAIN samples, discarding a washout at the start, and scored on the
 * N_TEST samples that follow, which it never saw during training.
 *
 * Output: prints train and test NRMSE, and writes
 *   data/output_signals.dat  (t, target x(t+1), prediction, is_test)
 * for plotting with mg_results.gp.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <spires.h>

#define N_TRAIN 2000   /* samples used to fit the readout            */
#define N_TEST  1000   /* held-out samples used only for scoring     */
#define WASHOUT 200    /* start-up transient excluded from training  */

/*
 * Mackey-Glass delay equation
 *     dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t),
 * integrated with RK4 at dt = 0.1 (the delayed value held across each step)
 * and sampled once per unit time. The first 1000 samples are discarded so the
 * series starts on the chaotic attractor. With tau = 17 the series is chaotic.
 */
static int generate_mackey_glass(double *buffer, size_t length, double x0,
                                 double tau, double beta, double gamma, double n)
{
    const double dt = 0.1;
    const size_t every = 10, discard = 1000;
    size_t lag = (size_t)lround(tau / dt);
    size_t steps = (length + discard) * every + lag + 1;

    double *x = malloc(steps * sizeof *x);
    if (!x) return -1;

    for (size_t i = 0; i <= lag; i++) x[i] = x0;   /* constant initial history */
    for (size_t i = lag; i + 1 < steps; i++) {
        double xt = x[i], xd = x[i - lag];
        double drive = beta * xd / (1.0 + pow(xd, n));
        double k1 = drive - gamma * xt;
        double k2 = drive - gamma * (xt + 0.5 * dt * k1);
        double k3 = drive - gamma * (xt + 0.5 * dt * k2);
        double k4 = drive - gamma * (xt + dt * k3);
        x[i + 1] = xt + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    for (size_t t = 0; t < length; t++) buffer[t] = x[lag + (discard + t) * every];

    free(x);
    return 0;
}

/* Root-mean-square error normalized by the standard deviation of the target. */
static double nrmse(const double *pred, const double *target, size_t n)
{
    double mean = 0.0, var = 0.0, err = 0.0;
    for (size_t i = 0; i < n; i++) mean += target[i];
    mean /= (double)n;
    for (size_t i = 0; i < n; i++) {
        var += (target[i] - mean) * (target[i] - mean);
        err += (pred[i] - target[i]) * (pred[i] - target[i]);
    }
    return sqrt(err / var);
}

int main(void)
{
    const size_t T = N_TRAIN + N_TEST;
    int ret = 1;
    spires_reservoir *R = NULL;
    double *pred = NULL;
    FILE *out = NULL;

    /* ---- Data: input x(t), target x(t+1) ---- */
    double *series = malloc((T + 1) * sizeof *series);
    double *target = malloc(T * sizeof *target);
    if (!series || !target) { fprintf(stderr, "allocation failed\n"); goto done; }

    if (generate_mackey_glass(series, T + 1, 1.2, 17.0, 0.2, 0.1, 10.0) != 0) {
        fprintf(stderr, "Mackey-Glass allocation failed\n");
        goto done;
    }
    const double *input = series;              /* x(t),   t = 0 .. T-1 */
    for (size_t t = 0; t < T; t++) target[t] = series[t + 1];

    /* ---- Neurons: Grunwald-Letnikov fractional LIF ----
     * Order: [V_th, V_reset, V_rest, tau_m, alpha, T_mem, bias, t_ref].
     * alpha = 1 is ordinary LIF; lower it (e.g. 0.8) for power-law memory. */
    double neuron_params[8] = {
        1.0,     /* V_th    firing threshold                 */
        0.0,     /* V_reset potential after a spike          */
        0.0,     /* V_rest  resting potential                */
        2.0,     /* tau_m   membrane time constant (samples) */
        1.0,     /* alpha   fractional order, in (0, 1]      */
        200.0,   /* T_mem   memory window (samples)          */
        0.0,     /* bias    constant input current           */
        0.0      /* t_ref   refractory period                */
    };

    spires_reservoir_config cfg = {
        .num_neurons       = 500,
        .num_inputs        = 1,
        .num_outputs       = 1,
        .spectral_radius   = 0.5,
        .ei_ratio          = 0.8,
        .input_strength    = 1.0,
        .connectivity      = 0.1,
        .dt                = 1.0,
        .connectivity_type = SPIRES_CONN_RANDOM,
        .neuron_type       = SPIRES_NEURON_FLIF_GL,
        .neuron_params     = neuron_params,
        .seed              = 42,       /* change for a different network */
        .use_washout       = 1,
        .washout           = WASHOUT,
    };

    if (spires_reservoir_create(&cfg, &R) != SPIRES_OK) {
        fprintf(stderr, "failed to create reservoir\n");
        goto done;
    }

    /* ---- Train on the first N_TRAIN samples only ---- */
    const double lambda = 1.0;   /* ridge regularization */
    if (spires_train_ridge(R, input, target, N_TRAIN, lambda) != SPIRES_OK) {
        fprintf(stderr, "ridge training failed\n");
        goto done;
    }

    /* ---- Run from rest over the whole series; score the held-out part ---- */
    spires_reservoir_reset(R);
    pred = spires_run(R, input, T);
    if (!pred) { fprintf(stderr, "inference failed\n"); goto done; }

    printf("train NRMSE (after washout): %.4f\n",
           nrmse(pred + WASHOUT, target + WASHOUT, N_TRAIN - WASHOUT));
    printf("test  NRMSE (held out)     : %.4f\n",
           nrmse(pred + N_TRAIN, target + N_TRAIN, N_TEST));

    out = fopen("data/output_signals.dat", "w");
    if (!out) { fprintf(stderr, "could not open data/output_signals.dat\n"); goto done; }
    fprintf(out, "# t target_x(t+1) prediction is_test\n");
    for (size_t t = WASHOUT; t < T; t++)
        fprintf(out, "%zu %f %f %d\n", t, target[t], pred[t], t >= N_TRAIN);

    ret = 0;

done:
    if (out) fclose(out);
    free(pred);
    spires_reservoir_destroy(R);
    free(target);
    free(series);
    return ret;
}
