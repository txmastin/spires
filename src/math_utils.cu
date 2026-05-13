#include "math_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define MAX_LEN 2048
__constant__ float c_coeffs[MAX_LEN];

#define TILE_SIZE 32 // for tiled cuda kernel

#define PEEK(label) do { \
    cudaError_t _e = cudaPeekAtLastError(); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[PEEK %s] error pending: %s (%d)\n", \
                label, cudaGetErrorString(_e), (int)_e); \
    } \
} while (0)

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _err = (call);                                             \
    if (_err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d in %s: %s (%d)\n",            \
                __FILE__, __LINE__, #call,                                 \
                cudaGetErrorString(_err), (int)_err);                      \
        exit(1);                                                           \
    }                                                                      \
} while (0)

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error\n"); \
        exit(1); \
    }

struct cuda_backend {
    float *d_W, *d_W_in;
    float *d_input_vector, *d_external_inputs, *d_recurrent;
    float *d_spikes_a, *d_spikes_b;
    float *d_V_history;
    float *d_X;           // training state matrix [series_length × N], device-side
    size_t X_series_len;

    float V_rest, V_th, V_reset, tau_m, bias, alpha, dt_alpha;

    int mem_len;
    int internal_step;

    double *host_spikes;

    cublasHandle_t cublas;
    cudaStream_t   stream;
};

__global__ void fill_float_kernel(float *p, float v, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void update_flif_gl_kernel(
    const float * __restrict__ external_inputs, // [N]
    const float * __restrict__ recurrent,       // [N]
    float       * __restrict__ V_history,       // [mem_len x N], time-major
    float       * __restrict__ next_spikes,     // [N]  output
    float V_rest,
    float V_th,
    float V_reset,
    float tau_m,
    float bias,
    float dt_alpha,
    int head,       // current write slot in circular buffer
    int prev_idx,   // slot containing V_{n-1}
    int limit,      // how many history steps are valid
    int mem_len,
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // --- 1. History term: sum_{k=1}^{limit} c_k * V_{n-k,i} ---
    float history = 0.0f;
    int idx = prev_idx;
    for (int k = 1; k <= limit; ++k) {
        history += c_coeffs[k] * V_history[(size_t)idx * N + i];
        idx = (idx == 0) ? mem_len - 1 : idx - 1;
    }

    // --- 2. RHS of the fractional LIF equation ---
    const float V_prev = V_history[(size_t)prev_idx * N + i];
    const float rhs = -(V_prev - V_rest) / tau_m
                     + external_inputs[i] + recurrent[i] + bias;

    // --- 3. GL direct update ---
    float V = dt_alpha * rhs - history;

    // --- 4. Spike and reset ---
    float spike = 0.0f;
    if (V >= V_th) {
        V     = V_reset;
        spike = 1.0f;
    }

    // --- 5. Write outputs ---
    V_history[(size_t)head * N + i] = V;
    next_spikes[i] = spike;
}

extern "C" void call_peek()
{
    PEEK("calling...");
}

extern "C" void cuda_init_reservoir(struct reservoir *r)
{
    cudaFree(0);
    cudaGetLastError();
    PEEK("cuda_init_reservoir start");

    struct cuda_backend *cb = (struct cuda_backend *)calloc(1, sizeof *cb);
    if (!cb) { fprintf(stderr, "OOM allocating cuda_backend\n"); exit(1); }

    const size_t N = r->num_neurons;
    const size_t M = r->num_inputs;

    struct flif_gl_neuron *n0 = (struct flif_gl_neuron *)r->neurons[0];
    cb->V_th    = (float)n0->V_th;
    cb->V_reset = (float)n0->V_reset;
    cb->V_rest  = (float)n0->V_rest;
    cb->tau_m   = (float)n0->tau_m;
    cb->bias    = (float)n0->bias;
    cb->alpha   = (float)n0->alpha;
    cb->dt_alpha = (float)pow(r->dt, n0->alpha);
    cb->mem_len  = n0->mem_len;
    cb->internal_step = 0;

    if (n0->mem_len > MAX_LEN) {
        fprintf(stderr, "mem_len %d exceeds MAX_LEN %d\n",
            n0->mem_len, MAX_LEN);
        exit(1);
    }

    // Convert GL coefficients to float and upload to __constant__ memory
    float *coeffs_f = (float *)malloc(n0->mem_len * sizeof(float));
    for (int k = 0; k < n0->mem_len; k++) coeffs_f[k] = (float)n0->coeffs[k];
    CUDA_CHECK(cudaMemcpyToSymbol(c_coeffs, coeffs_f, n0->mem_len * sizeof(float)));
    free(coeffs_f);

    CUDA_CHECK(cudaMallocHost(&cb->host_spikes, N * sizeof(double)));

    // Allocate float weight matrices on device
    CUDA_CHECK(cudaMalloc(&cb->d_W,    N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cb->d_W_in, N * M * sizeof(float)));

    // Convert and upload W (double → float)
    float *W_f = (float *)malloc(N * N * sizeof(float));
    for (size_t k = 0; k < N * N; k++) W_f[k] = (float)r->W[k];
    CUDA_CHECK(cudaMemcpy(cb->d_W, W_f, N * N * sizeof(float), cudaMemcpyHostToDevice));
    free(W_f);

    // Convert and upload W_in (double → float)
    float *W_in_f = (float *)malloc(N * M * sizeof(float));
    for (size_t k = 0; k < N * M; k++) W_in_f[k] = (float)r->W_in[k];
    CUDA_CHECK(cudaMemcpy(cb->d_W_in, W_in_f, N * M * sizeof(float), cudaMemcpyHostToDevice));
    free(W_in_f);

    // Allocate float scratch buffers
    CUDA_CHECK(cudaMalloc(&cb->d_input_vector,    M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cb->d_external_inputs, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cb->d_recurrent,       N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cb->d_spikes_a,        N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cb->d_spikes_b,        N * sizeof(float)));

    // V_history: time-major [mem_len x N], init to V_rest
    CUDA_CHECK(cudaMalloc(&cb->d_V_history, (size_t)cb->mem_len * N * sizeof(float)));
    float *tmp = (float *)malloc((size_t)cb->mem_len * N * sizeof(float));
    for (size_t k = 0; k < (size_t)cb->mem_len * N; k++) tmp[k] = cb->V_rest;
    CUDA_CHECK(cudaMemcpy(cb->d_V_history, tmp,
                          (size_t)cb->mem_len * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(tmp);

    // Initial spikes = 0
    CUDA_CHECK(cudaMemset(cb->d_spikes_a, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(cb->d_spikes_b, 0, N * sizeof(float)));

    // cuBLAS and stream
    CUDA_CHECK(cudaStreamCreate(&cb->stream));
    CUBLAS_CHECK(cublasCreate(&cb->cublas));
    CUBLAS_CHECK(cublasSetStream(cb->cublas, cb->stream));

    r->cuda_backend = cb;
}

extern "C" void cuda_step_reservoir(struct reservoir *r, const double *input_vector)
{
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    const int    micro_steps = (int)llround(1.0 / r->dt);
    const int    N = (int)r->num_neurons;
    const int    M = (int)r->num_inputs;
    const float  zero_f = 0.0f, one_f = 1.0f;
    const float  input_strength_f = (float)r->input_strength;

    // Convert input vector double → float and upload
    float input_f[M];
    for (int j = 0; j < M; j++) input_f[j] = (float)input_vector[j];
    CUDA_CHECK(cudaMemcpyAsync(cb->d_input_vector, input_f,
                               M * sizeof(float),
                               cudaMemcpyHostToDevice, cb->stream));

    CUBLAS_CHECK(cublasSgemv(cb->cublas, CUBLAS_OP_T,
                             M, N,
                             &input_strength_f,
                             cb->d_W_in, M,
                             cb->d_input_vector, 1,
                             &zero_f,
                             cb->d_external_inputs, 1));

    float *last = cb->d_spikes_a;
    float *next = cb->d_spikes_b;
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    for (int t = 0; t < micro_steps; ++t) {

        CUBLAS_CHECK(cublasSgemv(cb->cublas, CUBLAS_OP_T,
                                 N, N,
                                 &one_f,
                                 cb->d_W, N,
                                 last, 1,
                                 &zero_f,
                                 cb->d_recurrent, 1));

        const int head     = cb->internal_step % cb->mem_len;
        const int prev_idx = (head == 0) ? cb->mem_len - 1 : head - 1;
        const int limit    = (cb->internal_step < cb->mem_len)
                             ? cb->internal_step : cb->mem_len - 1;

        update_flif_gl_kernel<<<blocks, threads, 0, cb->stream>>>(
            cb->d_external_inputs, cb->d_recurrent,
            cb->d_V_history, next,
            cb->V_rest, cb->V_th, cb->V_reset,
            cb->tau_m, cb->bias, cb->dt_alpha,
            head, prev_idx, limit, cb->mem_len, N);

        CUDA_CHECK(cudaGetLastError());

        cb->internal_step++;

        float *tmp = last; last = next; next = tmp;
    }

    cb->d_spikes_a = last;
    cb->d_spikes_b = next;
}

// Copies current reservoir state (float) to host buffer (double).
extern "C" void cuda_copy_state(struct reservoir *r, double *out)
{
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    const size_t N = r->num_neurons;
    const int last_head =
        (cb->internal_step - 1 + cb->mem_len) % cb->mem_len;

    CUDA_CHECK(cudaStreamSynchronize(cb->stream));

    float *tmp_f = (float *)malloc(N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(tmp_f,
                          cb->d_V_history + (size_t)last_head * N,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t k = 0; k < N; k++) out[k] = (double)tmp_f[k];
    free(tmp_f);
}

extern "C" void cuda_reset_reservoir(struct reservoir *r)
{
    if (!r || !r->cuda_backend) return;
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    const size_t N = r->num_neurons;
    const size_t total = (size_t)cb->mem_len * N;

    fill_float_kernel<<<(total + 255) / 256, 256, 0, cb->stream>>>(
        cb->d_V_history, cb->V_rest, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(cb->d_spikes_a, 0, N * sizeof(float), cb->stream));
    CUDA_CHECK(cudaMemsetAsync(cb->d_spikes_b, 0, N * sizeof(float), cb->stream));

    cb->internal_step = 0;
}

extern "C" void cuda_alloc_state_buffer(struct reservoir *r, size_t series_length)
{
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    const size_t N = r->num_neurons;
    CUDA_CHECK(cudaMalloc(&cb->d_X, series_length * N * sizeof(float)));
    cb->X_series_len = series_length;
}

// Device-to-device async copy: writes current V into row t of d_X (no CPU sync).
extern "C" void cuda_collect_state(struct reservoir *r, size_t t)
{
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    const size_t N = r->num_neurons;
    const int last_head = (cb->internal_step - 1 + cb->mem_len) % cb->mem_len;
    CUDA_CHECK(cudaMemcpyAsync(
        cb->d_X + t * N,
        cb->d_V_history + (size_t)last_head * N,
        N * sizeof(float),
        cudaMemcpyDeviceToDevice,
        cb->stream));
}

// Single host sync + D2H copy, with float → double conversion for CPU ridge regression.
extern "C" void cuda_get_state_buffer(struct reservoir *r, double *h_X, size_t series_length)
{
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    const size_t N = r->num_neurons;
    CUDA_CHECK(cudaStreamSynchronize(cb->stream));

    float *h_X_f = (float *)malloc(series_length * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_X_f, cb->d_X,
                          series_length * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t k = 0; k < series_length * N; k++) h_X[k] = (double)h_X_f[k];
    free(h_X_f);
}

extern "C" void cuda_free_state_buffer(struct reservoir *r)
{
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    if (cb->d_X) {
        cudaFree(cb->d_X);
        cb->d_X = NULL;
        cb->X_series_len = 0;
    }
}

extern "C" void cuda_free_reservoir(struct reservoir *r)
{
    printf("calling cuda free res\n");
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    if (!cb) return;

    cudaFree(cb->d_W);
    cudaFree(cb->d_W_in);
    cudaFree(cb->d_input_vector);
    cudaFree(cb->d_external_inputs);
    cudaFree(cb->d_recurrent);
    cudaFree(cb->d_spikes_a);
    cudaFree(cb->d_spikes_b);
    cudaFree(cb->d_V_history);
    cudaFree(cb->d_X);
    cudaFreeHost(cb->host_spikes);

    cublasDestroy(cb->cublas);
    cudaStreamDestroy(cb->stream);
    free(cb);
    r->cuda_backend = NULL;
}

// Matrix-vector multiplication: y = A * x
extern "C" void mat_vec_mult(double *A, double *x, double *y, size_t n)
{
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		    n, n, 1.0, A, n, x, 1, 0.0, y, 1);
}

// Power Iteration to estimate spectral radius
extern "C" double calc_spectral_radius(double *A, size_t n)
{
	const unsigned int max_iter = 1000;
	const double tol = 1e-6;

	double x[n];
	double y[n];

	size_t i;
	for (i = 0; i < n; i++)
		x[i] = 1.0;

	double lambda_old = 0.0;
	double lambda_new = 0.0;

	unsigned int iter;
	for (iter = 0; iter < max_iter; iter++) {
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			    n, n, 1.0, A, n, x, 1, 0.0, y, 1);

		lambda_new = 0.0;
		for (i = 0; i < n; i++) {
			double abs_y = fabs(y[i]);
			if (abs_y > lambda_new)
				lambda_new = abs_y;
		}

		for (i = 0; i < n; i++)
			x[i] = y[i] / lambda_new;

		if (fabs(lambda_new - lambda_old) < tol)
			break;

		lambda_old = lambda_new;
	}

	return lambda_new;
}


extern "C" void rescale_matrix(double *A, size_t n, double target_rho)
{
	double rho = calc_spectral_radius(A, n);
	double rescale_factor = target_rho / rho;
	cblas_dscal(n * n, rescale_factor, A, 1);
}

extern "C" void mat_transpose(double *A, double *A_T, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

__global__ void mat_mat_mult_kernel(const double * __restrict__ A,
                                    const double * __restrict__ B,
                                    double       * __restrict__ C,
                                    size_t r1, size_t c1, size_t c2)
{
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double acc = 0.0;

    int numTiles = (c1 + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {

        int aCol = t * TILE_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] =
            (row < r1 && aCol < c1) ? A[row * c1 + aCol] : 0.0;

        int bRow = t * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < c1 && col < c2) ? B[bRow * c2 + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < r1 && col < c2)
        C[row * c2 + col] = acc;
}

void mat_mat_mult(double *A, double *B, double *C,
		  size_t r1, size_t c1, size_t c2)
{
    printf("entering: mat_mat_mul cblas\n");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    r1, c2, c1,
		    1.0, A, c1,
		    B, c2,
		    0.0, C, c2);
}

extern "C" int solve_linear_system_lud(double *A, double *b, double *x, size_t n)
{
	int info, *ipiv;
	double *A_copy, *b_copy;
	size_t i;

	ipiv = (int *)malloc(n * sizeof(int));
	if (!ipiv) {
		fprintf(stderr, "Error: malloc failed for ipiv\n");
		return -1;
	}

	A_copy = (double *)malloc(n * n * sizeof(double));
	if (!A_copy) {
		fprintf(stderr, "Error: malloc failed for A_copy\n");
		free(ipiv);
		return -1;
	}
	memcpy(A_copy, A, n * n * sizeof(double));

	b_copy = (double *)malloc(n * sizeof(double));
	if (!b_copy) {
		fprintf(stderr, "Error: malloc failed for b_copy\n");
		free(ipiv);
		free(A_copy);
		return -1;
	}
	memcpy(b_copy, b, n * sizeof(double));

	info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, A_copy, n, ipiv, b_copy, 1);

	if (info != 0) {
		if (info < 0)
			fprintf(stderr, "Error: Argument %d had illegal value\n", -info);
		else
			fprintf(stderr, "Error: U[%d,%d] is exactly zero. Matrix is singular.\n", info, info);
		free(ipiv);
		free(A_copy);
		free(b_copy);
		return -1;
	}

	for (i = 0; i < n; i++)
		x[i] = b_copy[i];

	free(ipiv);
	free(A_copy);
	free(b_copy);

	return 0;
}
