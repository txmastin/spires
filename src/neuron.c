#include<stdio.h>
#include<stdlib.h>
#include "neuron.h"

/***** small helper function for exiting *****/
static void _handle_unknown_neuron_type(const char *function_name)
{
    fprintf(stderr, "ERROR in %s: Neuron type unavailable. Exiting gracefully.\n", function_name);
    exit(EXIT_FAILURE);
}


// Helper function to compute the GL coefficients
static void compute_gl_coeffs(double *coeffs_array, double alpha, int N)
{
    if (!coeffs_array) return;
    coeffs_array[0] = 1.0;
    for (int k = 1; k < N; k++) {
        coeffs_array[k] = coeffs_array[k - 1] * (1.0 - (alpha + 1.0) / (float)k);
    }
}

void *init_neuron(enum neuron_type type, double *neuron_params, double dt, void **shared_neuron_data) 
{
    void* neuron = NULL;
    switch(type) {
        case LIF_DISCRETE: 
            neuron = (struct lif_discrete_neuron *)init_lif_discrete(neuron_params);
            break;
        case LIF_BIO: 
            neuron = (struct lif_bio_neuron *)init_lif_bio(neuron_params);
            break;
        case FLIF_CAPUTO:
            neuron = (struct flif_caputo_neuron *)init_flif_caputo(neuron_params, dt);
            break;
        case FLIF_GL:
            #pragma omp critical
            {
                if (*shared_neuron_data == NULL) {
                    const double alpha = neuron_params[4];
                    const double T_mem = neuron_params[5]; 
                    int mem_len = (T_mem > 0 && dt > 0) ? (int)(T_mem / dt) : 2000;
                    double *coeffs_arr = malloc(mem_len * sizeof(*coeffs_arr));
                    if (coeffs_arr != NULL) {
                        compute_gl_coeffs(coeffs_arr, alpha, mem_len);
                        *shared_neuron_data = coeffs_arr;
                    } else {
                        fprintf(stderr, "Error allocating memory to fractional memory trace. Exiting!");
                        exit(EXIT_FAILURE);
                    }
                }
            }
            neuron = (struct flif_gl_neuron *)init_flif_gl(neuron_params, dt, *shared_neuron_data);
            break;
        case FLIF_DIFFUSIVE:
            neuron = (struct flif_diffusive_neuron *)init_flif_diffusive(neuron_params, dt);
            break;
        default:
            _handle_unknown_neuron_type(__func__);
    }
    return neuron;
}

void update_neuron(void *neuron, enum neuron_type type, double input, double dt) 
{
    // update neuron based on type  
    switch(type) {
        case LIF_DISCRETE:
            update_lif_discrete((struct lif_discrete_neuron *)neuron, input);
            break;
        case LIF_BIO:
            update_lif_bio((struct lif_bio_neuron *)neuron, input, dt);
            break;
        case FLIF_CAPUTO:
            update_flif_caputo((struct flif_caputo_neuron *)neuron, input, dt);
            break;
        case FLIF_GL:
            update_flif_gl((struct flif_gl_neuron *)neuron, input, dt);
            break;
        case FLIF_DIFFUSIVE:
            update_flif_diffusive((struct flif_diffusive_neuron *)neuron, input, dt);
            break;
        default:
            _handle_unknown_neuron_type(__func__);
    }
}

double get_neuron_state(void *neuron, enum neuron_type type) 
{
    double val; 
    switch(type) {
        case LIF_DISCRETE:
            val = ((struct lif_discrete_neuron *)neuron)->V;
            break;
        case LIF_BIO:
            val = ((struct lif_bio_neuron *)neuron)->V;
            break;
        case FLIF_CAPUTO:
            val = ((struct flif_caputo_neuron *)neuron)->V;
            break;
        case FLIF_GL:
            val = ((struct flif_gl_neuron *)neuron)->V;
            break;
        case FLIF_DIFFUSIVE:
            val = ((struct flif_diffusive_neuron *)neuron)->V;
            break;
        default:
            _handle_unknown_neuron_type(__func__);
    }
    return val;
}

double get_neuron_spike(void *neuron, enum neuron_type type) 
{
    double spike; 
    switch(type) {
        case LIF_DISCRETE:
            spike = ((struct lif_discrete_neuron *)neuron)->spike;
            break;
        case LIF_BIO:
            spike = ((struct lif_bio_neuron *)neuron)->spike;
            break;
        case FLIF_CAPUTO:
            spike = ((struct flif_caputo_neuron *)neuron)->spike;
            break;
        case FLIF_GL:
            spike = ((struct flif_gl_neuron *)neuron)->spike;
            break;
        case FLIF_DIFFUSIVE:
            spike = ((struct flif_diffusive_neuron *)neuron)->spike;
            break;
        default:
            _handle_unknown_neuron_type(__func__);
    }
    return spike;
}

void free_neuron(void *neuron, enum neuron_type type) 
{
    if (!neuron) { return; }
    switch(type) {
        case LIF_DISCRETE:
            free_lif_discrete((struct lif_discrete_neuron *)neuron);
            break;
        case LIF_BIO:
            free_lif_bio((struct lif_bio_neuron *)neuron);
            break;
        case FLIF_CAPUTO:
            free_flif_caputo((struct flif_caputo_neuron *)neuron);
            break;
        case FLIF_GL:
            free_flif_gl((struct flif_gl_neuron *)neuron);
            break;
        case FLIF_DIFFUSIVE:
            free_flif_diffusive((struct flif_diffusive_neuron *)neuron);
            break;
    }
    neuron = NULL;
}

