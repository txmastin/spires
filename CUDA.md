# CUDA Backend

This document covers the CUDA acceleration path in `src/math_utils.cu`. It targets developers who know C/CUDA but are new to this codebase.

The CUDA path only supports **FLIF-GL neurons**. Other neuron models run on the CPU.

---

## Architecture overview

The CUDA backend is opaque to the rest of the library. The `reservoir` struct holds a `void *cuda_backend` pointer; the C side never touches the fields inside it. Everything GPU-related goes through the `extern "C"` functions in `math_utils.cu`.

### `cuda_backend` struct

```c
struct cuda_backend {
    /* Weight matrices (float, device) */
    float *d_W;               // [N × N] recurrent weights
    float *d_W_in;            // [N × M] input weights

    /* Per-step scratch (float, device) */
    float *d_input_vector;    // [M]  current input
    float *d_external_inputs; // [N]  W_in × input result
    float *d_recurrent;       // [N]  W × spikes result

    /* Ping-pong spike buffers (float, device) */
    float *d_spikes_a;        // [N]
    float *d_spikes_b;        // [N]

    /* Voltage history (float, device) */
    float *d_V_history;       // [mem_len × N], time-major, circular

    /* Training state matrix (float, device) — only live during training */
    float *d_X;               // [series_length × N]
    size_t X_series_len;

    /* Cached neuron parameters */
    float V_rest, V_th, V_reset, tau_m, bias, alpha, dt_alpha;

    int mem_len;              // GL history buffer depth (= T_mem / dt)
    int internal_step;        // global step counter; drives circular buffer head

    double *host_spikes;      // pinned host buffer [N] for D2H transfers

    cublasHandle_t cublas;
    cudaStream_t   stream;
};
```

All work runs on a single CUDA stream. The cuBLAS handle is bound to that stream on creation, so cuBLAS calls and custom kernel launches are automatically ordered.

### `__constant__` memory

```c
#define MAX_LEN 2048
__constant__ float c_coeffs[MAX_LEN];
```

The Grünwald–Letnikov coefficients (`c_k`) are uploaded once during `cuda_init_reservoir` and read by every thread in `update_flif_gl_kernel`. Placing them in constant memory gives broadcast caching: all threads in a warp read the same coefficient at the same step, so the access hits the constant cache rather than global memory.

### Voltage history layout

`d_V_history` is a 2-D array of shape `[mem_len × N]`, stored row-major. Each row is one time-slot; each column is one neuron. The buffer is circular: `internal_step % mem_len` gives the write slot (`head`). Reading history walks backwards from `head - 1`.

Row-major with neurons as columns means the FLIF kernel (`update_flif_gl_kernel`) accesses history for each neuron `i` at stride `N` per time-step — one global memory load per coefficient per neuron per kernel invocation.

---

## Per-step execution pipeline

`cuda_step_reservoir` drives one **macro-step** (one external timestep). With `dt = 0.1`, `micro_steps = 1/dt = 10`.

```
[host]  convert input double[] → float[]
        cudaMemcpyAsync → d_input_vector          (H2D, async)

[GPU]   cublasSgemv: d_W_in × d_input_vector
                  → d_external_inputs             (done once per macro-step)

for t in 0..micro_steps:
    [GPU]  cublasSgemv: d_W × last_spikes → d_recurrent
    [GPU]  update_flif_gl_kernel(d_external_inputs, d_recurrent,
                                 d_V_history, next_spikes, ...)
    swap(last_spikes, next_spikes)   // pointer swap, no copy
```

The spike buffers (`d_spikes_a` / `d_spikes_b`) are swapped by pointer each micro-step, not by copying data.

`d_external_inputs` is computed once and reused across all micro-steps within a macro-step. This is valid because the external input is constant within a macro-step.

### `update_flif_gl_kernel`

One thread per neuron. For neuron `i`:

1. Walk back through `d_V_history` up to `limit` slots, accumulating `sum(c_coeffs[k] * V[n-k, i])`.
2. Compute the RHS: `-(V_prev - V_rest) / tau_m + external[i] + recurrent[i] + bias`.
3. New voltage: `V = dt_alpha * rhs - history_sum`.
4. Threshold check: if `V >= V_th`, set `V = V_reset` and `spike = 1`.
5. Write `V` to `d_V_history[head * N + i]` and `spike` to `next_spikes[i]`.

`limit` is clamped to `min(internal_step, mem_len - 1)` so early steps don't read uninitialized history.

---

## Function reference

### `cuda_init_reservoir(reservoir *r)`

Allocates and initializes the full backend for reservoir `r`. Called once at reservoir creation.

- Reads neuron parameters from `r->neurons[0]` (all neurons share the same params in the FLIF-GL model).
- Converts `r->W` and `r->W_in` from `double` to `float` on the host, then uploads to device.
- Uploads GL coefficients to `c_coeffs` constant memory.
- Allocates `d_V_history` and initializes all slots to `V_rest`.
- Zeros `d_spikes_a` and `d_spikes_b`.
- Creates the cuBLAS handle and CUDA stream.
- Stores the backend pointer in `r->cuda_backend`.

**Precondition:** `r->neurons[0]` must be a `flif_gl_neuron`. `n0->mem_len` must be ≤ `MAX_LEN` (2048).

---

### `cuda_step_reservoir(reservoir *r, const double *input_vector)`

Advances the reservoir by one macro-step. See [Per-step execution pipeline](#per-step-execution-pipeline) above.

**No CPU–GPU synchronization** is performed here. The call returns while GPU work may still be in flight.

---

### `cuda_copy_state(reservoir *r, double *out)`

Copies the current membrane voltages to `out` (length `N`).

- Calls `cudaStreamSynchronize` to flush all pending GPU work.
- Does a single blocking D2H copy of the most recent row in `d_V_history`.
- Converts float → double.

Use this for **inference/readout** when you need the state on the host immediately. Do not call this in a training loop (use `cuda_collect_state` instead).

---

### `cuda_reset_reservoir(reservoir *r)`

Resets neuron state to initial conditions without freeing device memory.

- Fills all of `d_V_history` with `V_rest` via `fill_float_kernel` (async).
- Zeros both spike buffers via `cudaMemsetAsync`.
- Resets `internal_step` to 0.

All operations are async on the stream. Returns before GPU work completes.

---

### `cuda_alloc_state_buffer(reservoir *r, size_t series_length)`

Allocates `d_X` on the device: `[series_length × N]` float matrix. Call once before a training run.

Memory cost: `series_length × N × 4` bytes. For N=1000, 62,500 steps → ~250 MB.

---

### `cuda_collect_state(reservoir *r, size_t t)`

Async D2D copy of the current voltage row into row `t` of `d_X`. No CPU sync, no PCIe transfer.

Must be called after each `cuda_step_reservoir` call during training. `d_X` must have been allocated with `cuda_alloc_state_buffer`.

---

### `cuda_get_state_buffer(reservoir *r, double *h_X, size_t series_length)`

Flushes the stream, then bulk-copies `d_X` to `h_X` in a single D2H transfer. Converts float → double for the CPU-side ridge regression solver.

This is the **only synchronization point** in the training loop. Call it once after all `cuda_step_reservoir` / `cuda_collect_state` calls are done.

---

### `cuda_free_state_buffer(reservoir *r)`

Frees `d_X` and zeroes the pointer. Call after `cuda_get_state_buffer` to reclaim device memory.

---

### `cuda_free_reservoir(reservoir *r)`

Frees all device memory, destroys the cuBLAS handle and stream, and frees the `cuda_backend` struct. Sets `r->cuda_backend = NULL`.

---

## Training pipeline

The sequence for `spires_train_ridge` with CUDA:

```
cuda_alloc_state_buffer(r, series_length)

for t in 0..series_length:
    cuda_step_reservoir(r, input[t])        // all async, GPU runs ahead
    cuda_collect_state(r, t)                // async D2D into d_X[t]

cuda_get_state_buffer(r, h_X, series_length) // one sync + one bulk D2H

cuda_free_state_buffer(r)

// CPU: solve W_out = (h_X' h_X + λI)^-1 h_X' Y  via cblas + LAPACKE
```

The critical design decision: **all GPU → CPU synchronization is deferred to a single call** at the end of the loop. The CPU races ahead queuing 62,500 async operations; the GPU runs continuously without stalling. Before this design, each `cuda_copy_state` call inserted a `cudaStreamSynchronize` every timestep, creating 62,500 sync barriers.

---

## Performance notes and known bottlenecks

### What has been done

**Eliminated per-step sync barriers.** Previously, collecting the state matrix required a blocking D2H copy each timestep (`cudaStreamSynchronize` × 62,500). The async D2D collect pattern (`cuda_collect_state` + `cuda_get_state_buffer`) removed all of these.

**Float32 throughout.** Weights, voltage history, and all intermediate buffers are `float`. On consumer NVIDIA GPUs (RTX series), double-precision throughput is 1/32 of float-precision throughput. The conversion from the double-typed public API happens once at init (`cuda_init_reservoir`) and once per macro-step for the input vector.

### Remaining bottlenecks (priority order)

**1. Ridge regression on CPU (high impact)**

After `cuda_get_state_buffer`, the state matrix `h_X` is transferred to host and the regression (`X'X`, `X'Y`, Cholesky solve) runs on CPU via cblas/LAPACKE. For N=1000, series_length=62,500, `X'X` is 1000×1000 and `X'Y` involves a 62,500×1000 matrix — feasible on CPU, but moving this to cuBLAS (`cublasSsyrk`) + cuSolver (`cusolverDnSpotrf`) would eliminate the D2H transfer entirely and parallelize the regression.

**2. Memory bandwidth in `update_flif_gl_kernel` (medium impact)**

The GL history loop reads up to 100 rows of `d_V_history` from global memory per kernel invocation. With N=1000, each row is 4 KB; 100 rows = 400 KB per macro-step per micro-step. A tile of neurons loaded cooperatively into shared memory would reduce redundant global memory transactions. Arithmetic intensity is ~0.25 FLOP/byte — well below the roofline ridge (~5–15 FLOP/byte on modern GPUs).

**3. N is too small for good GPU occupancy (medium impact, model-dependent)**

N=1000 → 4 blocks of 256 threads. Most SMs sit idle. At N=2000–4000, occupancy improves substantially. Increasing N changes the whole optimization story.

**4. CUDA Graphs for the micro-step loop (low impact)**

The 10 cuBLAS calls + 10 kernel launches per macro-step have a fixed structure and could be captured as a CUDA Graph to reduce CPU launch overhead. Worth considering only after bottlenecks 1–3 are addressed.

### Profiling commands

```bash
# Timeline: spot idle gaps, sync barriers, transfer overhead
nsys profile --stats=true --trace=cuda,nvtx -o profile_report ./spoken_digit_recognition
nsys-ui profile_report.nsys-rep

# Kernel deep dive: bandwidth utilization, occupancy, warp efficiency
ncu --set full --kernel-name update_flif_gl_kernel -o kernel_report ./spoken_digit_recognition
ncu-ui kernel_report.ncu-rep

# Roofline
ncu --set roofline --kernel-name update_flif_gl_kernel ./spoken_digit_recognition
```
