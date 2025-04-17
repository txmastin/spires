#include <stdio.h>
#include <stdlib.h>
#include "neurons/FLIF_Caputo.h"

int main(void) {
    // --- Params ---
    double Cm = 500.0;
    double gl = 25.0;
    double Vl = 0.0;
    double Vth = 20.0;
    double V0 = 0.0;
    double Vpeak = 90.0;
    double alpha = 0.6;
    double tref = 5.0;
    double Tmem = 2000.0;
    double dt = 0.05;
    double Iapp = 870.0;

    // simulation length in ms
    double T = 6000.0;
    int n_steps = (int)(T / dt);

    // --- Param array ---
    double neuron_params[10] = {
        Cm, gl, Vl, Vth, V0, Vpeak, alpha, tref, Tmem, dt
    };

    // --- Init neuron ---
    FLIFCaputoNeuron *neuron = init_FLIFCaputo(neuron_params);

    // --- Open files for saving ---
    FILE *fv = fopen("c_voltage.txt", "w");
    FILE *fs = fopen("c_spikes.txt", "w");

    // --- Simulation ---
    for (int i = 0; i < n_steps; i++) {
        double t = i * dt;

        update_FLIFCaputo(neuron, Iapp);

        fprintf(fv, "%lf\n", neuron->V);
        if (neuron->spike > 0.5) {
            fprintf(fs, "%lf\n", t / 1000.0); // seconds, like Python
        }
    }

    fclose(fv);
    fclose(fs);
    free_FLIFCaputo(neuron);

    return 0;
}

