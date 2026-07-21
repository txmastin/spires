# spires

A high-performance spiking reservoir computing library written in C.

Spires provides configurable spiking neural network reservoirs with multiple neuron models, including fractional-order leaky integrate-and-fire neurons, multiple network topologies, and batch/online readout training.

## Features

- **5 neuron models**: discrete LIF, biophysical LIF, and three fractional-order variants (Caputo, Grunwald-Letnikov, diffusive)
- **3 network topologies**: Erdos-Renyi random, Watts-Strogatz small-world, Barabasi-Albert scale-free
- **Readout training**: ridge regression (batch) and online delta rule
- **Coarse-graining**: renormalization-group-inspired reservoir compression that preserves critical dynamics
- **Built-in optimizer**: AGILE-based hyperparameter search with multi-fidelity budgets
- **OpenMP parallelism**: multi-threaded neuron updates via BLAS
- **Clean C API**: opaque handles, status codes, no hidden allocations in hot paths

## Background

A spiking reservoir is a class of reservoir computing system that utilizes the temporal dynamics of spiking activity for computation. A reservoir consists of $N$ interconnected spiking neurons where each neuron integrates input from connected neurons over time, accumulating a membrane potential $v\_i$. In a standard leaky integrate-and-fire (LIF) neuron, the membrane dynamics follow:

$$\frac{dv\_i}{dt} = -\frac{v\_i(t) - v\_{rest}}{\tau\_m} + I\_i(t),$$

where $\tau\_m$ is the membrane time constant and $I\_i(t) = \sum\_{j=1}^{N} W\_{ij} S\_j(t) + \gamma W\_{in,i} u(t)$ is the total synaptic input comprising recurrent spike impulses and external input.

Spires extends this with **fractional-order LIF neurons** [[2]](#references), which replace the integer-order derivative with a fractional derivative of order $\alpha \in (0, 1]$:

$$D^\alpha v\_i(t) = -\frac{v\_i(t) - v\_{rest}}{\tau\_m} + I\_i(t).$$

This is discretized using the Grünwald–Letnikov (GL) scheme [[2]](#references):

$$v\_i[n] = \Delta t^\alpha \left(-\frac{v\_i[n-1] - v\_{rest}}{\tau\_m} + I\_i[n]\right) - \sum\_{k=1}^{L} w\_k^{(\alpha)} v\_i[n-k],$$

where $w\_k^{(\alpha)}$ are the Grunwald-Letnikov coefficients ($w\_0 = 1$, $w\_k = w\_{k-1}(1 - (\alpha + 1)/k)$) and $L = T\_{mem} / \Delta t$ is the memory horizon. When $\alpha = 1$, the history sum vanishes and the model recovers the standard LIF. For $0 < \alpha < 1$, the history sum introduces long-range temporal memory, producing spike-frequency adaptation and richer dynamics without additional state variables.

When the membrane potential $v\_i$ surpasses the firing threshold $\theta$, the neuron emits a spike:

$$S\_i[n] = \begin{cases} 1, & \text{if } v\_i[n] \geq \theta, \\\\ 0, & \text{otherwise}, \end{cases}$$

and the membrane potential resets to $v\_{reset}$.

The neurons are connected together with weights $W$:

$$W = \rho \frac{W\_0}{\lambda\_{max}(W\_0)},$$

where $W\_0$ is a random weight matrix with sparse connectivity $c$, $\lambda\_{max}(W\_0)$ is the largest eigenvalue of $W\_0$, and $\rho$ is the desired spectral radius.

A readout layer with weights $W\_{out}$ maps the high-dimensional reservoir state into an output $y(t)$:

$$y(t) = W\_{out} \cdot \mathbf{v}(t).$$

The readout is trained to minimize the error between the target and predicted output while the internal reservoir weights remain fixed.

## Neuron Models

| Type | Enum | Parameters |
|------|------|------------|
| Discrete LIF | `SPIRES_NEURON_LIF_DISCRETE` | `[V_0, V_th, leak_rate, bias]` |
| Biophysical LIF | `SPIRES_NEURON_LIF_BIO` | `[V_0, V_th, tau, bias]` |
| Fractional LIF (Caputo) | `SPIRES_NEURON_FLIF_CAPUTO` | `[C_m, g_l, V_l, V_th, V_reset, V_peak, alpha, t_ref, T_mem]` |
| Fractional LIF (Grunwald-Letnikov) | `SPIRES_NEURON_FLIF_GL` | `[V_th, V_reset, V_rest, tau_m, alpha, T_mem, bias, t_ref]` |
| Fractional LIF (Diffusive) | `SPIRES_NEURON_FLIF_DIFFUSIVE` | `[V_th, V_reset, V_rest, tau_m, alpha, T_mem, bias]` |

The fractional-order parameter `alpha` controls the order of the fractional derivative in the membrane potential dynamics. When `alpha = 1.0`, the neuron reduces to a standard LIF. Values `0 < alpha < 1` introduce long-range temporal memory through the Grünwald–Letnikov (or Caputo/diffusive) discretization of the fractional derivative [[2]](#references), enabling spike-frequency adaptation and richer temporal dynamics. The parameter α simultaneously governs memory capacity, active information storage, and information transfer — each peaking at distinct values of α — making it a continuous control parameter over the reservoir's information-processing regime.

The `T_mem` parameter controls the memory horizon of the fractional derivative. The number of history samples retained is `L = T_mem / dt`. Larger values increase accuracy of the fractional approximation at the cost of computation and memory.

`SPIRES_NEURON_FLIF_GL`'s `t_ref` sets an absolute refractory period (same time units as `dt`/`tau_m`): after firing, the neuron is held at `V_reset` (dynamics not integrated, no threshold check) until `t_ref` time units have elapsed. `t_ref = 0` disables refractoriness entirely. Non-zero `t_ref` blocks `round(t_ref / dt)` subsequent steps after each spike.

## Network Topologies

| Type | Enum | Description |
|------|------|-------------|
| Random | `SPIRES_CONN_RANDOM` | Erdos-Renyi: each connection exists independently with probability `connectivity` |
| Small-World | `SPIRES_CONN_SMALL_WORLD` | Watts-Strogatz: regular ring lattice with random rewiring |
| Scale-Free | `SPIRES_CONN_SCALE_FREE` | Barabasi-Albert preferential attachment: hub-and-spoke structure |

The weight matrix is initialized randomly and then rescaled so that its spectral radius (largest absolute eigenvalue) equals the configured `spectral_radius`. This controls the echo state property of the reservoir.

## Readout Training

### Ridge Regression (batch)

Given a recorded state trajectory $\Phi \in \mathbb{R}^{T \times N}$ and targets $Y \in \mathbb{R}^{T \times M}$, ridge regression solves for the output weights in closed form:

$$W\_{out} = ({\Phi}^\intercal \Phi + \lambda I)^{-1} {\Phi}^\intercal Y,$$

where $\lambda$ is a regularization parameter. This is the recommended training method — it is fast, deterministic, and produces good results for most tasks. Use `spires_train_ridge()`.

### Recursive Least Squares (online)

RLS is a second-order online method that maintains a running estimate of the inverse correlation matrix $P \in \mathbb{R}^{N \times N}$ and updates the output weights incrementally:

$$\mathbf{k}[t] = \frac{P[t-1]\,\mathbf{v}(t)}{\lambda + \mathbf{v}(t)^\intercal P[t-1]\,\mathbf{v}(t)},$$

$$W\_{out}[t] = W\_{out}[t-1] + \mathbf{e}(t)\,\mathbf{k}[t]^\intercal,$$

$$P[t] = \frac{1}{\lambda}\left(P[t-1] - \mathbf{k}[t]\,\mathbf{v}(t)^\intercal P[t-1]\right),$$

where $\lambda \in (0, 1]$ is a forgetting factor and $\mathbf{e}(t) = y\_{\text{target}}(t) - W\_{\text{out}}[t-1]\,\mathbf{v}(t)$ is the prediction error. $P$ is initialised as $P[0] = \delta^{-1} I$, where $\delta$ is a small positive constant controlling the strength of the initial prior. Compared to the delta rule, RLS converges in far fewer steps by using second-order curvature information. Use `spires_train_rls()`.

### Online Delta Rule

The delta rule updates output weights one step at a time with a single scaled outer product:

$$W\_{out} \leftarrow W\_{out} + \mu \cdot \epsilon(t) \cdot \mathbf{v}(t)^\intercal,$$

where $\mu$ is the learning rate and $\epsilon(t) = y\_{target}(t) - y(t)$ is the prediction error. Unlike RLS, it requires no auxiliary matrices — memory overhead is $O(N)$ and per-step compute is $O(N \cdot D\_{out})$ regardless of reservoir size. Convergence is slower than RLS, but the minimal footprint makes it well suited to embedded or resource-constrained deployments where memory is more limiting than training efficiency. Use `spires_train_online()`.

## Coarse-Graining

Spires implements the coarse-graining (renormalization) algorithm from [[1]](#references). The algorithm iteratively merges pairs of neurons connected by strong positive weights — neurons that share a high synaptic weight perform redundant computation and can be treated as a single unit.

When two neurons $i$ and $j$ satisfy $W_{ij} > w_{th}$ or $W_{ji} > w_{th}$, they are merged into a single super-neuron with:

- **State**: $v_i \leftarrow (v_i + v_j) / 2$
- **Recurrent weights**: averaged (or preserved if only one direction is non-zero) to maintain effective drive
- **Input weights**: same averaging rule applied to $W_{in}$
- **Readout weights**: $W_{out}[\cdot, i] \leftarrow W_{out}[\cdot, i] + W_{out}[\cdot, j]$, so the merged neuron's averaged state still produces the same output contribution

Merging cascades until no remaining pair exceeds the threshold, producing a compressed reservoir.

For **avalanche-critical reservoirs** — those whose recurrent dynamics produce neuronal avalanches with a power-law size distribution, which occurs when the branching ratio $\sigma \approx 1$ — the macro-scale dynamics are invariant under this transformation. The coarse-grained reservoir is a renormalization group fixed point: its avalanche statistics are scale-invariant, and the same readout trained on the original reservoir transfers without retraining. For non-critical reservoirs, dynamics change under coarse-graining and retraining is required.

Use `spires_coarse_grain()` to produce a new, smaller reservoir from an existing one. The weight threshold can be set as a high percentile of the positive weight distribution to control how aggressively the network is compressed.

## Hyperparameter Optimization

Spires includes a built-in optimizer based on **AGILE** (Adaptive Gradient-Informed Levy Exploration). AGILE combines gradient-informed search with Levy flight exploration:

1. **Exploration phase**: searches the hyperparameter space using gradient-directed steps with Levy-distributed step sizes, which produce occasional long jumps that help escape local minima. When an analytical gradient is not provided, SPSA (Simultaneous Perturbation Stochastic Approximation) estimates the gradient numerically.
2. **Refinement phase**: once a patience budget is exhausted without improvement, AGILE restores the best-known point and switches to deterministic local polishing with smaller step sizes.

The optimizer tunes reservoir size, spectral radius, input gain, connectivity, excitatory/inhibitory ratio, network topology, fractional order, and ridge regularization simultaneously. It supports multi-fidelity evaluation budgets — running quick, low-seed evaluations early and more expensive, multi-seed evaluations to confirm promising candidates.

Use `spires_optimize()` to run the optimizer. See `examples/mackey_glass_prediction/` for a complete example.

## Quick Start

```c
#include <stdio.h>
#include <stdlib.h>
#include <spires.h>

int main(void)
{
    /* Configure a 200-neuron reservoir with fractional-order LIF neurons */
    double neuron_params[] = {
        1.0,    /* V_th    — firing threshold    */
        0.0,    /* V_reset — reset voltage       */
        0.0,    /* V_rest  — resting potential    */
        20.0,   /* tau_m   — membrane time const  */
        0.7,    /* alpha   — fractional order     */
        100.0,  /* T_mem   — memory horizon       */
        0.1     /* bias    — constant input bias  */
    };

    spires_reservoir_config cfg = {
        .num_neurons       = 200,
        .num_inputs        = 1,
        .num_outputs       = 1,
        .spectral_radius   = 0.9,
        .ei_ratio          = 0.8,
        .input_strength    = 1.0,
        .connectivity      = 0.2,
        .dt                = 0.1,
        .connectivity_type = SPIRES_CONN_RANDOM,
        .neuron_type       = SPIRES_NEURON_FLIF_GL,
        .neuron_params     = neuron_params,
    };

    /* Create and initialize */
    spires_reservoir *res = NULL;
    if (spires_reservoir_create(&cfg, &res) != SPIRES_OK) {
        fprintf(stderr, "Failed to create reservoir\n");
        return 1;
    }

    /* Step through some input */
    double input[] = {0.5};
    for (int t = 0; t < 100; t++)
        spires_step(res, input);

    /* Read the reservoir state */
    double state[200];
    spires_read_reservoir_state(res, state);
    printf("state[0] = %f\n", state[0]);

    /* Clean up */
    spires_reservoir_destroy(res);
    return 0;
}
```

## Examples

The `examples/` directory contains two complete applications:

- **`mackey_glass_prediction/`** — chaotic time series prediction with automatic hyperparameter optimization
- **`spoken_digit_recognition/`** — FSDD spoken digit classification using fractional-order neurons

## Building

### Dependencies

- **OpenBLAS** (BLAS implementation)
- **LAPACKE** (C interface to LAPACK)
- **OpenMP** (parallelism, typically bundled with your compiler)
- A C99-compatible compiler (clang or gcc)

On Debian/Ubuntu:
```sh
sudo apt install libopenblas-dev liblapacke-dev
```

On Gentoo:
```sh
emerge sci-libs/openblas sci-libs/lapack
```

### Compile

```sh
make
```

This produces `lib/libspires.a`. To use spires in your project, link against the static library:

```sh
cc -O2 -I/path/to/spires/include my_program.c \
    -L/path/to/spires/lib -lspires -lopenblas -llapacke -lm -fopenmp
```

## API Reference

All public functions and types are declared in a single header:

```c
#include <spires.h>
```

### Status Codes

Every function that can fail returns a `spires_status`:

| Code | Meaning |
|------|---------|
| `SPIRES_OK` | Success |
| `SPIRES_ERR_INVALID_ARG` | NULL pointer or invalid parameter |
| `SPIRES_ERR_ALLOC` | Memory allocation failure |
| `SPIRES_ERR_INTERNAL` | Backend computation error |

### Configuration

```c
typedef struct {
    size_t num_neurons;                    /* reservoir size */
    size_t num_inputs;                     /* input dimensionality */
    size_t num_outputs;                    /* output dimensionality */
    double spectral_radius;                /* eigenvalue scaling of W */
    double ei_ratio;                       /* excitatory/inhibitory ratio */
    double input_strength;                 /* input weight scaling */
    double connectivity;                   /* connection density [0, 1] */
    double dt;                             /* integration timestep; 1/dt must be integer */
    spires_connectivity_type connectivity_type;
    spires_neuron_type       neuron_type;
    double *neuron_params;                 /* neuron-specific parameters (caller-owned) */
} spires_reservoir_config;
```

### Lifecycle

```c
/* Create a reservoir from a config. Caller must destroy with spires_reservoir_destroy(). */
spires_status spires_reservoir_create(const spires_reservoir_config *cfg,
                                      spires_reservoir **out_r);

/* Free all resources. Safe to call on NULL. */
void spires_reservoir_destroy(spires_reservoir *r);

/* Reset neuron states to initial conditions. Weights are preserved. */
spires_status spires_reservoir_reset(spires_reservoir *r);
```

### Stepping and Running

```c
/* Advance the reservoir by one timestep with input vector u_t (length = num_inputs). */
spires_status spires_step(spires_reservoir *r, const double *u_t);

/* Run the reservoir over a full input series. Returns a malloc'd output array
 * of length (series_length * num_outputs). Caller must free(). */
double *spires_run(spires_reservoir *r, const double *input_series, size_t series_length);
```

### Coarse-Graining

```c
/* Produce a smaller reservoir by merging neuron pairs whose recurrent weight
 * exceeds weight_threshold. Returns a new reservoir via out_r; the original
 * is unchanged. Caller must destroy out_r with spires_reservoir_destroy().
 *
 * The threshold is typically set as a high percentile of the positive weight
 * distribution (e.g. the 99th percentile to merge only the top 1%). */
spires_status spires_coarse_grain(const spires_reservoir *r,
                                  double weight_threshold,
                                  spires_reservoir **out_r);
```

### Training

```c
/* Batch ridge regression: solves for W_out over the full input/target series. */
spires_status spires_train_ridge(spires_reservoir *r,
                                 const double *input_series,
                                 const double *target_series,
                                 size_t series_length, double lambda);

/* Recursive least squares: second-order online method, processes the full series
 * sequentially. delta initialises P = (1/delta)*I; lambda is the forgetting factor
 * in (0, 1] (use 1.0 for no forgetting). Converges faster than the delta rule at
 * the cost of O(N^2) memory for the inverse correlation matrix. */
spires_status spires_train_rls(spires_reservoir *r,
                               const double *input_series,
                               const double *target_series,
                               size_t series_length,
                               double delta, double lambda);

/* Online delta rule: single-step weight update. O(N) memory; suited to embedded
 * or resource-constrained deployments where footprint matters more than convergence speed. */
spires_status spires_train_online(spires_reservoir *r,
                                  const double *target_vec, double lr);
```

### State Access

```c
/* Copy current neuron states into a caller-provided buffer (length = num_neurons). */
spires_status spires_read_reservoir_state(spires_reservoir *r, double *buffer);

/* Returns a malloc'd copy of the state (length = num_neurons). Caller must free(). */
double *spires_copy_reservoir_state(spires_reservoir *r);

/* Compute readout output y = W_out * state. Caller provides buffer (length = num_outputs). */
spires_status spires_compute_output(spires_reservoir *r, double *out);
```

### Introspection

```c
size_t spires_num_neurons(const spires_reservoir *r);
size_t spires_num_inputs(const spires_reservoir *r);
size_t spires_num_outputs(const spires_reservoir *r);
```

## References

[1] Mastin, T. & Teuscher, C. Coarse-Graining Spiking Reservoirs: Reducing Reservoir Size While Preserving Critical Dynamics. *Northeast Journal of Complex Systems (NEJCS)* **7**(2), Article 2 (2025). https://doi.org/10.63562/2577-8439.1106

[2] Mastin, T., Anderson, N., McNeal, S., Tubbin, M. & Teuscher, C. Fractional-order systems for neuromorphic computing: software and hardware opportunities and challenges. *npj Unconventional Computing* **3**, 24 (2026). https://doi.org/10.1038/s44335-026-00070-8

## License

MIT License. See [LICENSE](LICENSE) for details.
