// mc_memory_capacity.c
// Memory Capacity (MC) experiment using your spires reservoir exactly like your FSDD example.
// Only includes your header; no API prototypes are re-declared here.
// Build (example): clang -O2 -Wall -Wextra -o mc_memory_capacity mc_memory_capacity.c -lm
// Link against your spires lib and include path for "reservoir.h".

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "reservoir.h"   // your library header

// ===================== User Config =====================
#define NUM_FEATURES 1
#define K_MAX        20
#define NUM_CLASSES  K_MAX

#define NUM_SAMPLES_TRAIN  1
#define NUM_SAMPLES_TEST   1
#define SEQUENCE_LENGTH    6000
#define WASHOUT_STEPS      500
#define TEST_WASHOUT_STEPS 500

// Reservoir hyperparams (tune as you like)
static const size_t NUM_NEURONS      = 40;
static const double SPECTRAL_RADIUS  = 0.9;
static const double EI_RATIO         = 0.8;       
static const double INPUT_STRENGTH   = 1.0;
static const double CONNECTIVITY     = 0.2;
static const enum connectivity_type C_TYPE = RANDOM;         
static const enum neuron_type N_TYPE = FLIF_GL;         

// Fractional neuron params layout (your example):
// [0] V_th, [1] V_reset, [2] V_rest, [3] tau_m, [4] alpha, [5] dt, [6] Tmem, [7] bias
static const double DT     = 1.0;
static const int    TMEM   = SEQUENCE_LENGTH;    // or SEQUENCE_LENGTH
static const double ALPHA  = 0.1;     // sweep this as needed
static const double BIAS   = 0.55;

#define LAMBDA_RIDGE 1e-4

// ===================== Small Utilities =====================
static inline uint64_t xorshift64star(uint64_t* s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}
static inline double rng_uniform_m1p1(uint64_t* s) {
    return ((xorshift64star(s) >> 11) * (1.0/9007199254740992.0)) * 2.0 - 1.0;
}
static void zscore_inplace(double* x, size_t n, double* mean_out, double* std_out) {
    double sum=0.0, sum2=0.0;
    for (size_t i=0;i<n;i++){ sum += x[i]; sum2 += x[i]*x[i]; }
    double mu = sum / (double)n;
    double var = fmax(1e-12, sum2/(double)n - mu*mu);
    double sd = sqrt(var);
    for (size_t i=0;i<n;i++) x[i] = (x[i]-mu)/sd;
    if (mean_out) *mean_out = mu;
    if (std_out)  *std_out  = sd;
}
static double compute_R2(const double* y, const double* yhat, size_t n) {
    double mu=0.0; for (size_t i=0;i<n;i++) mu += y[i]; mu /= (double)n;
    double sse=0.0, sst=0.0;
    for (size_t i=0;i<n;i++){ double e=y[i]-yhat[i]; sse+=e*e; double d=y[i]-mu; sst+=d*d; }
    if (sst < 1e-12) return 0.0;
    double r2 = 1.0 - sse/sst;
    if (r2 < 0.0) r2 = 0.0; if (r2 > 1.0) r2 = 1.0;
    return r2;
}

// Build y[t][k-1] = u[t-k] when valid (t >= washout + k); else 0 (we'll only score valid rows).
static void build_mc_targets(const double* x, size_t T_total, size_t washout,
                             size_t Kmax, double* y_out /*(T_total x Kmax)*/)
{
    memset(y_out, 0, T_total*Kmax*sizeof(double));
    for (size_t t=0; t<T_total; t++){
        for (size_t k=1; k<=Kmax; k++){
            if (t >= washout + k){
                size_t t_src = t - k;
                y_out[t*Kmax + (k-1)] = x[t_src*NUM_FEATURES + 0];
            }
        }
    }
}

// ===================== Main (matches your style) =====================
int main(void){
    const size_t T_train_total = NUM_SAMPLES_TRAIN * SEQUENCE_LENGTH;
    const size_t T_test_total  = NUM_SAMPLES_TEST  * SEQUENCE_LENGTH;

    // Allocate flat, time-major arrays (same shape convention as your example)
    double* train_x = (double*)calloc(T_train_total*NUM_FEATURES, sizeof(double));
    double* train_y = (double*)calloc(T_train_total*NUM_CLASSES, sizeof(double));
    double* test_x  = (double*)calloc(T_test_total *NUM_FEATURES, sizeof(double));
    double* test_y  = (double*)calloc(T_test_total *NUM_CLASSES, sizeof(double));
    if(!train_x||!train_y||!test_x||!test_y){ fprintf(stderr,"OOM\n"); return 1; }

    // Inputs: i.i.d. U(-1,1), z-scored (good practice for ridge)
    uint64_t s1=1234567ULL, s2=7654321ULL;
    for (size_t t=0; t<T_train_total; t++) train_x[t*NUM_FEATURES+0] = rng_uniform_m1p1(&s1);
    for (size_t t=0; t<T_test_total;  t++) test_x [t*NUM_FEATURES+0] = rng_uniform_m1p1(&s2);
    double mu, sd;
    zscore_inplace(train_x, T_train_total*NUM_FEATURES, &mu, &sd);
    zscore_inplace(test_x,  T_test_total *NUM_FEATURES, &mu, &sd);

    // Delay targets
    build_mc_targets(train_x, T_train_total, WASHOUT_STEPS,      K_MAX, train_y);
    build_mc_targets(test_x,  T_test_total,  TEST_WASHOUT_STEPS, K_MAX, test_y);

    // Neuron params as a raw double array (your layout)
    double fractional_neuron_params[] = {
        /*[0] V_th   */  1.0,
        /*[1] V_reset*/  0.0,
        /*[2] V_rest */  0.0,
        /*[3] tau_m  */  20.0,
        /*[4] alpha  */  ALPHA,
        /*[5] dt     */  DT,
        /*[6] Tmem   */  TMEM,
        /*[7] bias   */  BIAS
    };

    // --- Your exact pattern: create -> init -> train ---
    struct reservoir *res = create_reservoir(
        NUM_NEURONS, NUM_FEATURES, NUM_CLASSES,
        SPECTRAL_RADIUS, EI_RATIO, INPUT_STRENGTH,
        CONNECTIVITY, DT, C_TYPE, N_TYPE, fractional_neuron_params
    );
    if (!res) { fprintf(stderr,"create_reservoir failed\n"); return 1; }
    if (init_reservoir(res) != 0) { fprintf(stderr,"init_reservoir failed\n"); return 1; }

    train_output_ridge_regression(
        res, train_x, train_y,
        (size_t)(NUM_SAMPLES_TRAIN * SEQUENCE_LENGTH), LAMBDA_RIDGE
    );

    // --- Evaluation: same streaming style as your example ---
    reset_reservoir(res);

    double* yhat = (double*)calloc(T_test_total*NUM_CLASSES, sizeof(double));
    if(!yhat){ fprintf(stderr,"OOM yhat\n"); return 1; }

    double out_buf[NUM_CLASSES];
    for (size_t t=0; t<T_test_total; t++){
        step_reservoir(res, &test_x[t*NUM_FEATURES + 0]);   // <--- use your example's step()
        compute_output(res, out_buf);                       // <--- and your example's readout()
        memcpy(&yhat[t*NUM_CLASSES], out_buf, NUM_CLASSES*sizeof(double));
    }

    // Score only on the valid tail per delay
    double MC = 0.0;
    printf("#Delay\tR2\n");
    for (size_t k=0; k<K_MAX; k++){
        size_t t0 = TEST_WASHOUT_STEPS + (k+1);
        if (t0 >= T_test_total) { printf("%zu\t0.000000\n", k+1); continue; }
        size_t n = T_test_total - t0;

        // Gather contiguous column k over valid rows
        double *y = (double*)malloc(n*sizeof(double));
        double *yh= (double*)malloc(n*sizeof(double));
        if(!y||!yh){ fprintf(stderr,"OOM temp\n"); return 1; }
        for (size_t i=0;i<n;i++){
            y[i]  = test_y[(t0+i)*NUM_CLASSES + k];
            yh[i] = yhat[(t0+i)*NUM_CLASSES + k];
        }
        double r2 = compute_R2(y, yh, n);
        free(y); free(yh);

        MC += r2;
        printf("%zu\t%.6f\n", k+1, r2);
    }
    printf("Total_MC\t%.6f\n", MC);

    // Cleanup
    free(train_x); free(train_y); free(test_x); free(test_y); free(yhat);
    free_reservoir(res);
    return 0;
}
