#include "agile.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <alloca.h>

// ------------------- Tiny RNG (PCG32) -------------------
// Public domain PCG from O'Neill (minimal variant)
static inline uint32_t pcg32_next(uint64_t *state)
{
    uint64_t old = *state;
    *state = old * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline double rnd_uniform(uint64_t *state)
{
    /* Return uniform in [0,1). Using 32-bit mantissa from PCG. */
    return (pcg32_next(state) * (1.0/4294967296.0));
}


// ------------------- Helpers -------------------
static inline double clamp(double x, double a, double b)
{
    return x < a ? a : (x > b ? b : x);
}

static inline void reflect_inplace(double *x, const double *lo, const double *hi, int d)
{
    for(int i=0;i<d;i++) {
        if(x[i] < lo[i]) {
            double over = lo[i] - x[i];
            double span = hi[i] - lo[i];
            double k = floor(over/span);
            double r = over - k*span;
            x[i] = lo[i] + r; // one reflection is enough after modulo
        } else if(x[i] > hi[i]) {
            double over = x[i] - hi[i];
            double span = hi[i] - lo[i];
            double k = floor(over/span);
            double r = over - k*span;
            x[i] = hi[i] - r;
        }
    }
}

static inline double vec_norm(const double *g, int d)
{
    double s=0.0; 
    for(int i=0;i<d;i++) {
        s += g[i]*g[i];
    }
    return sqrt(s);
}

static inline void vec_scale(double *y, const double *x, double a, int d)
{
    for(int i=0;i<d;i++) {
        y[i] = a * x[i];
    }
}

static inline void vec_axpy(double *y, const double *x, double a, int d)
{
    for(int i=0;i<d;i++) {
        y[i] += a * x[i];
    }
}

static inline void vec_copy(double *y, const double *x, int d)
{
    memcpy(y, x, (size_t)d*sizeof(double));
}

// Random unit direction in R^d
static void random_unit(double *u, int d, uint64_t *state)
{
    double s=0.0; 
    for(int i=0;i<d;i++) { 
        double z = rnd_uniform(state)*2.0-1.0; 
        u[i]=z; 
        s+=z*z; 
    }
    /* small eps added to ensure no division by zero. 
     * since n is ultimately dependent on a uniform sampling, 
     * this shouldn't affect the randomness too much */
    double n = sqrt(s) + 1e-18; 
    for(int i=0;i<d;i++) {
        u[i] /= n;
    }
}


// Truncated power‑law (inverse CDF) with exponent mu>1 on [a,b]
static double sample_powerlaw(double mu, double a, double b, uint64_t *state)
{
    if(mu <= 1.0) 
        mu = 1.000001; // guard

    double u = rnd_uniform(state);
    double c1 = pow(a, 1.0 - mu);
    double c2 = pow(b, 1.0 - mu);
    double x = pow(c1 + (c2 - c1)*u, 1.0/(1.0 - mu));
    
    return x;
}


// SPSA gradient estimate (two‑point)
static void spsa_grad(const double *theta, const double *lo, 
                      const double *hi, int d,
                      double eps, agile_loss_fn f, 
                      void *ctx, uint64_t *state, double *g_out)
{
    // Rademacher +/-1 perturbation
    static const double signs[2] = { -1.0, 1.0 };
    double *delta = (double*)alloca((size_t)d*sizeof(double));
    double *tp = (double*)alloca((size_t)d*sizeof(double));
    double *tm = (double*)alloca((size_t)d*sizeof(double));


    for(int i=0;i<d;i++) {
        delta[i] = signs[pcg32_next(state)&1];
        tp[i] = theta[i] + eps*delta[i];
        tm[i] = theta[i] - eps*delta[i];
        // project to bounds by clamping /* SPSA uses clamp for numerical stability (ignores reflect flag) */
        tp[i] = clamp(tp[i], lo[i], hi[i]);
        tm[i] = clamp(tm[i], lo[i], hi[i]);
    }

    double fp = f(tp, d, ctx);
    double fm = f(tm, d, ctx);
    double scale = (fp - fm)/(2.0*eps);
    
    for(int i=0;i<d;i++) {
        g_out[i] = scale * delta[i];
    }
}


// Projection helper
static void project(double *theta, const double *lo, const double *hi, int d, int reflect)
{
    if(reflect) {
        reflect_inplace(theta, lo, hi, d);
    }
    else {
        for(int i=0;i<d;i++) {
            theta[i] = clamp(theta[i], lo[i], hi[i]);
        }
    }
}

// Main optimize
int agile_optimize(const double *theta0, const double *lower, const double *upper,
                   int dim, agile_loss_fn loss_fn, agile_grad_fn grad_fn, void *ctx,
                   const struct agile_options *opts_in, double *theta_best, struct agile_report *report)
{
    // Defaults
    struct agile_options opt = {
        .mu=1.3, .eta_min=1e-3, .eta_max=5e-1,
        .patience0=100.0, .patience_decay=0.9,
        .refine_step=0.0, .refine_max_iters=200, .refine_no_improve=30,
        .spsa_eps=1e-3,
        .reflect_at_bounds=1,
        .seed=0xC0FFEEULL,
        .verbose=1
    };

    if(opts_in) 
        opt = *opts_in;
     
    if(opt.refine_step <= 0.0) 
        opt.refine_step = opt.eta_min;
    
    printf("[AGILE] entered: dim=%d, verbose=%d, patience0:%0.0f\n", dim, opt.verbose, opt.patience0); fflush(stdout);
 
    // Working buffers
    double *theta = (double*)malloc((size_t)dim*sizeof(double));
    double *theta_cand = (double*)malloc((size_t)dim*sizeof(double));
    double *g = (double*)malloc((size_t)dim*sizeof(double));
    
    if(!theta || !theta_cand || !g) {
        free(theta);
        free(theta_cand);
        free(g);
        fprintf(stderr, "Error allocating memory during optimization. Aborting!\n");
        return EXIT_FAILURE;
    }


    uint64_t rng = opt.seed ? opt.seed : 0xA57B17E5ULL;


    // Init
    vec_copy(theta, theta0, dim);
    project(theta, lower, upper, dim, opt.reflect_at_bounds);


    double loss = loss_fn(theta, dim, ctx);
    double best_loss = loss;
    vec_copy(theta_best, theta, dim);
    int total_evals = 1;
    int total_iters = 0;
    int best_iter = 0;

    // Exploration loop
    double patience = opt.patience0;
    int steps_since_improve = 0;

    int in_refinement = 0;

    while(1) {
        total_iters++;
        // Compute gradient (true or SPSA)
        int have_grad = 0;
        if(grad_fn)
            have_grad = (grad_fn(theta, dim, g, ctx) == 0);

        if(!have_grad) {
            spsa_grad(theta, lower, upper, dim, opt.spsa_eps, loss_fn, ctx, &rng, g); 
            total_evals += 2; 
        }

        double gn = vec_norm(g, dim);
        double dir_buf[64]; // stack fast‑path for small dims
        double *dvec = (dim <= 64)? dir_buf : (double*)malloc((size_t)dim*sizeof(double));
        
        if(gn > 1e-18)
            vec_scale(dvec, g, -1.0/gn, dim);
        else
            random_unit(dvec, dim, &rng);


        // Lévy step size
        double eta = sample_powerlaw(opt.mu, opt.eta_min, opt.eta_max, &rng);


        // Propose candidate
        vec_copy(theta_cand, theta, dim);
        vec_axpy(theta_cand, dvec, eta, dim);
        project(theta_cand, lower, upper, dim, opt.reflect_at_bounds);


        double cand_loss = loss_fn(theta_cand, dim, ctx);
        total_evals++;


        if(cand_loss < loss) {
            // Accept move
            vec_copy(theta, theta_cand, dim);
            loss = cand_loss;
            steps_since_improve = 0;
            patience *= opt.patience_decay;

            
            if (patience < 1.0)
                patience = 1.0;
            // store best location if improved 
            if(loss < best_loss) {
                best_loss = loss;
                best_iter = total_iters;
                vec_copy(theta_best, theta, dim);
            }

            if(opt.verbose) {
                printf("[AGILE] iter %d improved: loss=%.6g, patience=%.2f\n", total_iters, loss, patience);
                fflush(stdout);
            }
        }
        else {
            steps_since_improve++;
        }


        if(dim > 64 && dvec) 
            free(dvec);

        // Switch to refinement when budget exhausted
        if(steps_since_improve >= (int)ceil(patience)) {
            if(opt.verbose) {
                printf("[AGILE] switch to refinement at iter %d (best=%.6g)\n", total_iters, best_loss);
                fflush(stdout);
            }
            in_refinement = 1;
            break;
        }

        // Optional: user may set a global iteration cap via refine_max_iters==0 meaning no refinement only exploration; not used here
        if(total_iters > 1000000) 
            break; // safety
    }


    // ---------------- Refinement phase ----------------
    if(in_refinement) {
        vec_copy(theta, theta_best, dim); // restore best‑so‑far
        loss = best_loss;
        int no_improve = 0;
        for(int k=0; k<opt.refine_max_iters; ++k){
            total_iters++;
            int have_grad = 0;
            if(grad_fn)
                have_grad = (grad_fn(theta, dim, g, ctx) == 0);
            if(!have_grad) {
                spsa_grad(theta, lower, upper, dim, opt.spsa_eps, loss_fn, ctx, &rng, g);
                total_evals += 2;
            }


            // Gradient descent step
            vec_axpy(theta, g, -opt.refine_step, dim);
            project(theta, lower, upper, dim, opt.reflect_at_bounds);
            double new_loss = loss_fn(theta, dim, ctx);
            total_evals++;
            if(new_loss < loss - 1e-12) {
                loss = new_loss;
                if(loss < best_loss){
                    best_loss = loss;
            best_iter = total_iters;
 
                    vec_copy(theta_best, theta, dim);
                }
                no_improve = 0;
            } else {
            no_improve++;
            }
            if(no_improve >= opt.refine_no_improve) 
                break;
        }
    }

    if(report) {
        report->best_loss = best_loss;
        report->best_iter = best_iter;
        report->total_evals = total_evals;
        report->total_iters = total_iters;
        report->exited_refinement = in_refinement;
    }


    free(theta);
    free(theta_cand);
    free(g);

    return EXIT_SUCCESS;
}

