// mc_memory_capacity_sweep.c
// Build: clang -O2 -Wall -Wextra -o mc_mc_sweep mc_memory_capacity_sweep.c -lm
// Link with your spires lib and include path for "reservoir.h".

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "reservoir.h"   // your API (create_reservoir, init_reservoir, step_reservoir, compute_output, etc.)

// ===================== Experiment Config =====================
#define NUM_FEATURES 1

// Linear and nonlinear memory targets (match your Python)
#define K_LIN 30         // linear delays k=1..30
#define K_NL  5          // nonlinear Legendre delays k=1..5 (P2 and P3)
#define NUM_CLASSES (K_LIN + 2*K_NL)

// Timings (feel free to tweak smaller for quick checks)
#define SEQUENCE_LENGTH    10000     // total steps per split
#define WASHOUT_STEPS       500
#define TEST_WASHOUT_STEPS  500

// Reservoir hyperparams
static const size_t NUM_NEURONS      = 4000;
static const int    AVG_DEGREE       = 15;
static const double SPECTRAL_RADIUS  = 0.9;
static const double EI_RATIO         = 0.8;
static const double INPUT_STRENGTH   = 1.0;
static const double DT               = 1.0;

// Connectivity as probability = K/(N-1) for ER(RANDOM)
static const enum connectivity_type C_TYPE = RANDOM;
static const enum neuron_type       N_TYPE = FLIF_GL;

// Ridge regularization
static const double LAMBDA_RIDGE = 1e-4;

// Alpha sweep
static const double ALPHA_START = 0.10;
static const double ALPHA_END   = 1.00;
static const double ALPHA_STEP  = 0.05;

// Fractional neuron params layout you specified:
// [0] V_th, [1] V_reset, [2] V_rest, [3] tau_m, [4] alpha, [5] dt, [6] Tmem, [7] bias
static const double V_TH   = 1.0;
static const double V_RESET= 0.0;
static const double V_REST = 0.0;
static const double TAU_M  = 20.0;
static const int    TMEM   = 2048;

// ===================== RNG & Math Helpers =====================
static inline uint64_t xorshift64star(uint64_t* s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}
static inline double rng_uniform_m1p1(uint64_t* s) {
    // Uniform in (-1,1)
    return ((xorshift64star(s) >> 11) * (1.0/9007199254740992.0)) * 2.0 - 1.0;
}

// Legendre polynomials P2, P3 (on u in [-1,1])
static inline double P2(double x){ return 0.5*(3.0*x*x - 1.0); }
static inline double P3(double x){ return 0.5*(5.0*x*x*x - 3.0*x); }

// Pearson r^2 (scale/offset invariant)
static double pearson_r2(const double* a, const double* b, size_t n) {
    double ma=0, mb=0;
    for (size_t i=0;i<n;i++){ ma+=a[i]; mb+=b[i]; }
    ma/= (double)n; mb/= (double)n;
    double num=0, da=0, db=0;
    for (size_t i=0;i<n;i++){
        double xa=a[i]-ma, xb=b[i]-mb;
        num += xa*xb; da += xa*xa; db += xb*xb;
    }
    if (da<=1e-12 || db<=1e-12) return 0.0;
    double r = num / sqrt(da*db);
    double r2 = r*r;
    return r2 < 0 ? 0.0 : (r2 > 1.0 ? 1.0 : r2);
}

// Closed-form alpha -> bias mapping (fit to your data; easy to tweak)
static inline double bias_from_alpha(double alpha) {
    const double b_inf = 0.057;   // plateau for high alpha
    const double A     = 0.986;   // amplitude
    const double k     = 7.083;   // decay rate
    double b = b_inf + A * exp(-k * alpha);
    if (b < b_inf) b = b_inf;
    return b;
}

// ===================== Target Builders =====================
// Build Y with linear (k=1..K_LIN) and nonlinear (P2/P3 for k=1..K_NL) targets.
// Y is time-major (T x NUM_CLASSES). We leave invalid early rows as 0 and
// only score on valid tails.
static void build_targets_lin_nl(const double* u_raw, size_t T_total, size_t washout,
                                 double* Y /* T_total x NUM_CLASSES */)
{
    memset(Y, 0, T_total*NUM_CLASSES*sizeof(double));
    for (size_t t=0; t<T_total; ++t) {
        // Linear
        for (size_t k=1; k<=K_LIN; ++k) {
            if (t >= washout + k) {
                Y[t*NUM_CLASSES + (k-1)] = u_raw[t - k];
            }
        }
        // Nonlinear
        for (size_t k=1; k<=K_NL; ++k) {
            if (t >= washout + k) {
                double uk = u_raw[t - k];
                Y[t*NUM_CLASSES + (K_LIN + (k-1))]        = P2(uk);
                Y[t*NUM_CLASSES + (K_LIN + K_NL + (k-1))] = P3(uk);
            }
        }
    }
}

// ===================== Main =====================
int main(void){
    const size_t T_train_total = SEQUENCE_LENGTH;
    const size_t T_test_total  = SEQUENCE_LENGTH;

    // Connectivity from desired average degree
    double CONNECTIVITY = (NUM_NEURONS > 1) ? (double)AVG_DEGREE / (double)(NUM_NEURONS - 1) : 0.0;

    // Allocate input and targets (time-major)
    double* train_x = (double*)calloc(T_train_total*NUM_FEATURES, sizeof(double));
    double* train_y = (double*)calloc(T_train_total*NUM_CLASSES, sizeof(double));
    double* test_x  = (double*)calloc(T_test_total *NUM_FEATURES, sizeof(double));
    double* test_y  = (double*)calloc(T_test_total *NUM_CLASSES, sizeof(double));
    if(!train_x||!train_y||!test_x||!test_y){ fprintf(stderr,"OOM\n"); return 1; }

    // Generate independent train/test input streams in [-1,1] (raw for targets AND drive)
    uint64_t s1=1234567ULL, s2=7654321ULL;
    for (size_t t=0; t<T_train_total; ++t) train_x[t] = rng_uniform_m1p1(&s1);
    for (size_t t=0; t<T_test_total;  ++t) test_x [t] = rng_uniform_m1p1(&s2);

    // Build linear + nonlinear targets from raw inputs
    build_targets_lin_nl(train_x, T_train_total, WASHOUT_STEPS,      train_y);
    build_targets_lin_nl(test_x,  T_test_total,  TEST_WASHOUT_STEPS, test_y);

    // Output files
    FILE *fp = fopen("data/alpha_mc.dat", "w");
    if (!fp) { fprintf(stderr, "Failed to open alpha_mc.dat\n"); return 1; }
    fprintf(fp, "# alpha\tMC_lin\tMC_nl\tMC_total\n");

    // Sweep alpha
    for (double alpha = ALPHA_START; alpha <= ALPHA_END + 1e-12; alpha += ALPHA_STEP) {
        // Bias from alpha (closed form)
        double bias = bias_from_alpha(alpha);

        // Neuron params array (your layout)
        double fractional_neuron_params[] = {
            /*[0] V_th   */  V_TH,
            /*[1] V_reset*/  V_RESET,
            /*[2] V_rest */  V_REST,
            /*[3] tau_m  */  TAU_M,
            /*[4] alpha  */  alpha,
            /*[5] dt     */  DT,
            /*[6] Tmem   */  TMEM,
            /*[7] bias   */  bias
        };

        // Create/init reservoir
        struct reservoir *res = create_reservoir(
            NUM_NEURONS, NUM_FEATURES, NUM_CLASSES,
            SPECTRAL_RADIUS, EI_RATIO, INPUT_STRENGTH,
            /* connectivity prob */ CONNECTIVITY, DT, C_TYPE, N_TYPE, fractional_neuron_params
        );
        if (!res) { fprintf(stderr,"create_reservoir failed (alpha=%.3f)\n", alpha); fclose(fp); return 1; }
        if (init_reservoir(res) != 0) { fprintf(stderr,"init_reservoir failed (alpha=%.3f)\n", alpha); free_reservoir(res); fclose(fp); return 1; }

        // Train multi-output ridge on train split
        train_output_ridge_regression(
            res, train_x, train_y,
            (size_t)T_train_total, LAMBDA_RIDGE
        );

        // Evaluate on test split: stream inputs and read predictions
        reset_reservoir(res);
        double* yhat = (double*)calloc(T_test_total*NUM_CLASSES, sizeof(double));
        if(!yhat){ fprintf(stderr,"OOM yhat\n"); free_reservoir(res); fclose(fp); return 1; }

        double out_buf[NUM_CLASSES];
        for (size_t t=0; t<T_test_total; ++t){
            // step with scalar input (time-major)
            step_reservoir(res, &test_x[t]);
            compute_output(res, out_buf);
            memcpy(&yhat[t*NUM_CLASSES], out_buf, NUM_CLASSES*sizeof(double));
        }

        // Compute Pearson r^2 per target on valid tail and sum
        double MC_lin = 0.0, MC_nl = 0.0;

        // Linear block
        for (size_t k = 1; k <= K_LIN; ++k) {
            size_t col = (k-1);
            size_t t0  = TEST_WASHOUT_STEPS + k;
            if (t0 >= T_test_total) continue;
            size_t n = T_test_total - t0;

            double *yt  = (double*)malloc(n*sizeof(double));
            double *yph = (double*)malloc(n*sizeof(double));
            if(!yt||!yph){ fprintf(stderr,"OOM temp\n"); free(yt); free(yph); free(yhat); free_reservoir(res); fclose(fp);  return 1; }

            for (size_t i=0;i<n;i++){
                yt[i]  =  test_y[(t0+i)*NUM_CLASSES + col];
                yph[i] = yhat[(t0+i)*NUM_CLASSES + col];
            }
            MC_lin += pearson_r2(yt, yph, n);
            free(yt); free(yph);
        }

        // Nonlinear P2 / P3 blocks
        for (size_t k = 1; k <= K_NL; ++k) {
            size_t colP2 = K_LIN + (k-1);
            size_t colP3 = K_LIN + K_NL + (k-1);
            size_t t0    = TEST_WASHOUT_STEPS + k;
            if (t0 >= T_test_total) continue;
            size_t n = T_test_total - t0;

            double *y2t  = (double*)malloc(n*sizeof(double));
            double *y2ph = (double*)malloc(n*sizeof(double));
            double *y3t  = (double*)malloc(n*sizeof(double));
            double *y3ph = (double*)malloc(n*sizeof(double));
            if(!y2t||!y2ph||!y3t||!y3ph){ fprintf(stderr,"OOM temp\n"); free(y2t); free(y2ph); free(y3t); free(y3ph); free(yhat); free_reservoir(res); fclose(fp); return 1; }

            for (size_t i=0;i<n;i++){
                y2t[i]  =  test_y[(t0+i)*NUM_CLASSES + colP2];
                y2ph[i] = yhat[(t0+i)*NUM_CLASSES + colP2];
                y3t[i]  =  test_y[(t0+i)*NUM_CLASSES + colP3];
                y3ph[i] = yhat[(t0+i)*NUM_CLASSES + colP3];
            }
            MC_nl += pearson_r2(y2t, y2ph, n);
            MC_nl += pearson_r2(y3t, y3ph, n);
            free(y2t); free(y2ph); free(y3t); free(y3ph);
        }

        double MC_total = MC_lin + MC_nl;

        // Log
        printf("alpha=%.3f  bias=%.4f  MC_lin=%.3f  MC_nl=%.3f  MC_total=%.3f\n",
               alpha, bias, MC_lin, MC_nl, MC_total);
        fprintf(fp, "%.6f\t%.6f\t%.6f\t%.6f\n", alpha, MC_lin, MC_nl, MC_total);

        // Cleanup per-alpha
        free(yhat);
        free_reservoir(res);
    }

    fclose(fp);

    free(train_x); free(train_y); free(test_x); free(test_y);

    return 0;
}

