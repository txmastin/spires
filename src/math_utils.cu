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
__constant__ double c_coeffs[MAX_LEN];

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
    double *d_W, *d_W_in;
    double *d_input_vector, *d_external_inputs, *d_recurrent;
    double *d_spikes_a, *d_spikes_b;
    double *d_V_history;

    // Uniform scalar parameters
    double V_rest, V_th, V_reset, tau_m, bias, alpha, dt_alpha;

    int mem_len;
    int internal_step;

    double *host_spikes;

    cublasHandle_t cublas;
    cudaStream_t   stream;
};

__global__ void fill_double_kernel(double *p, double v, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void update_flif_gl_kernel(
    const double * __restrict__ external_inputs, // [N]
    const double * __restrict__ recurrent,       // [N]
    double       * __restrict__ V_history,       // [mem_len x N], time-major
    double       * __restrict__ next_spikes,     // [N]  output
    double V_rest,
    double V_th,
    double V_reset,
    double tau_m,
    double bias,
    double dt_alpha,
    int head,       // current write slot in circular buffer
    int prev_idx,   // slot containing V_{n-1}
    int limit,      // how many history steps are valid
    int mem_len,
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // --- 1. History term: sum_{k=1}^{limit} c_k * V_{n-k,i} ---
    // Walk backwards through the circular buffer.
    // c_coeffs lives in __constant__ memory — one broadcast read per k.
    double history = 0.0;
    int idx = prev_idx;
    for (int k = 1; k <= limit; ++k) {
        // V_history is time-major: neuron i at time slot idx is [idx*N + i]
        history += c_coeffs[k] * V_history[(size_t)idx * N + i];
        idx = (idx == 0) ? mem_len - 1 : idx - 1;
    }

    // --- 2. RHS of the fractional LIF equation ---
    const double V_prev = V_history[(size_t)prev_idx * N + i];
    const double rhs = -(V_prev - V_rest) / tau_m
                     + external_inputs[i] + recurrent[i] + bias;

    // --- 3. GL direct update (Eq. 15) ---
    double V = dt_alpha * rhs - history;

    // --- 4. Spike and reset ---
    double spike = 0.0;
    if (V >= V_th) {
        V     = V_reset;
        spike = 1.0;
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
    cudaFree(0);                  // benign call that triggers init
    cudaGetLastError();           
    PEEK("cuda_init_reservoir start");

    struct cuda_backend *cb = (struct cuda_backend *)calloc(1, sizeof *cb);
    if (!cb) { fprintf(stderr, "OOM allocating cuda_backend\n"); exit(1); }

    const size_t N = r->num_neurons;
    const size_t M = r->num_inputs;

    // Pull scalar params from neuron[0] — uniform across all neurons per paper.
    struct flif_gl_neuron *n0 = (struct flif_gl_neuron *)r->neurons[0];
    cb->V_th    = n0->V_th;
    cb->V_reset = n0->V_reset;
    cb->V_rest  = n0->V_rest;
    cb->tau_m   = n0->tau_m;
    cb->bias    = n0->bias;
    cb->alpha   = n0->alpha;
    cb->dt_alpha = pow(r->dt, n0->alpha);
    cb->mem_len  = n0->mem_len;
    cb->internal_step = 0;

    if (n0->mem_len > MAX_LEN) {
        fprintf(stderr, "mem_len %d exceeds MAX_LEN %d\n",
            n0->mem_len, MAX_LEN);
        exit(1);
    }

    // Upload GL coefficients to __constant__ memory (one broadcast, free at runtime)
    CUDA_CHECK(cudaMemcpyToSymbol(c_coeffs, n0->coeffs,
                                  n0->mem_len * sizeof(double)));


    CUDA_CHECK(cudaMallocHost(&cb->host_spikes, N * sizeof(double))); 

    // Allocate and upload weight matrices
    CUDA_CHECK(cudaMalloc(&cb->d_W,    N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cb->d_W_in, N * M * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cb->d_W,    r->W,    N*N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cb->d_W_in, r->W_in, N*M*sizeof(double), cudaMemcpyHostToDevice));

    // Allocate scratch buffers
    CUDA_CHECK(cudaMalloc(&cb->d_input_vector,    M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cb->d_external_inputs, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cb->d_recurrent,       N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cb->d_spikes_a,        N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cb->d_spikes_b,        N * sizeof(double)));

    // V_history: time-major [mem_len x N], init to V_rest
    CUDA_CHECK(cudaMalloc(&cb->d_V_history, (size_t)cb->mem_len * N * sizeof(double)));

    // Initialize V_history to V_rest on device
    // Easiest: fill a host buffer and upload
    double *tmp = (double *)malloc((size_t)cb->mem_len * N * sizeof(double));
    for (size_t k = 0; k < (size_t)cb->mem_len * N; k++) tmp[k] = cb->V_rest;
    CUDA_CHECK(cudaMemcpy(cb->d_V_history, tmp,
                          (size_t)cb->mem_len * N * sizeof(double),
                          cudaMemcpyHostToDevice));
    free(tmp);

    // Initial spikes = 0
    CUDA_CHECK(cudaMemset(cb->d_spikes_a, 0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(cb->d_spikes_b, 0, N * sizeof(double)));

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
    const double zero = 0.0, one = 1.0;
    const double input_strength = r->input_strength;

    // host -> device
    CUDA_CHECK(cudaMemcpyAsync(cb->d_input_vector, input_vector,
                               M * sizeof(double),
                               cudaMemcpyHostToDevice, cb->stream));

    CUBLAS_CHECK(cublasDgemv(cb->cublas, CUBLAS_OP_T,
                             M, N,              // (rows, cols) of the col-major matrix
                             &input_strength,
                             cb->d_W_in, M,     // lda = M (row stride of row-major W_in)
                             cb->d_input_vector, 1,
                             &zero,
                             cb->d_external_inputs, 1));

    double *last = cb->d_spikes_a;
    double *next = cb->d_spikes_b;
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    for (int t = 0; t < micro_steps; ++t) {

        CUBLAS_CHECK(cublasDgemv(cb->cublas, CUBLAS_OP_T,
                                 N, N,
                                 &one,
                                 cb->d_W, N,
                                 last, 1,
                                 &zero,
                                 cb->d_recurrent, 1));

        // Circular buffer indices
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

        CUDA_CHECK(cudaGetLastError());  // catch launch errors immediately

        cb->internal_step++;

        // Swap buffers (no copy, just pointer swap)
        double *tmp = last; last = next; next = tmp;
    }

    // Persist buffer assignments for next macro step
    cb->d_spikes_a = last;
    cb->d_spikes_b = next;
}

extern "C" void cuda_copy_state(struct reservoir *r, double *out)
{
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    const size_t N = r->num_neurons;
    const int last_head =
        (cb->internal_step - 1 + cb->mem_len) % cb->mem_len;

    CUDA_CHECK(cudaStreamSynchronize(cb->stream));
    CUDA_CHECK(cudaMemcpy(out,
                          cb->d_V_history + (size_t)last_head * N,
                          N * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

extern "C" void cuda_reset_reservoir(struct reservoir *r)
{
    if (!r || !r->cuda_backend) return;
    struct cuda_backend *cb = (struct cuda_backend *)r->cuda_backend;
    const size_t N = r->num_neurons;
    const size_t total = (size_t)cb->mem_len * N;

    /* fill V_history with V_rest using a small kernel — avoids host scratch */
    fill_double_kernel<<<(total + 255) / 256, 256, 0, cb->stream>>>(
        cb->d_V_history, cb->V_rest, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(cb->d_spikes_a, 0, N * sizeof(double), cb->stream));
    CUDA_CHECK(cudaMemsetAsync(cb->d_spikes_b, 0, N * sizeof(double), cb->stream));

    cb->internal_step = 0;
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
    cudaFree(cb->host_spikes);

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
		/* y = A * x */
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			    n, n, 1.0, A, n, x, 1, 0.0, y, 1);

		/* Estimate largest magnitude component (Rayleigh estimate) */
		lambda_new = 0.0;
		for (i = 0; i < n; i++) {
			double abs_y = fabs(y[i]);
			if (abs_y > lambda_new)
				lambda_new = abs_y;
		}

		/* Normalize y -> x */
		for (i = 0; i < n; i++)
			x[i] = y[i] / lambda_new;

		/* Check convergence */
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
/*
void rescale_matrix(double* A, size_t n, double target_rho) 
{
    double rho = calc_spectral_radius(A, n);
    double rescale_factor = target_rho / rho;

    // rescale all values, such that the matrix has 
    // spectral radius = target_rho
    for (size_t i = 0; i < (n * n); i++) {
        A[i] *= rescale_factor;  
    }
}
*/
/**
 * @brief Transposes a matrix.
 * @param A The input matrix (rows x cols).
 * @param A_T The output transposed matrix (cols x rows).
 */
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

    // register variable for accumulating the dot product for this threads output element
    // keeping this in a register is faster that assigning it to global or shared memory
    double acc = 0.0;

    // Sweep tiles across the shared dimension (c1)
    int numTiles = (c1 + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {

        // Load one tile of A (row-major): rows of A, columns = tile strip
        int aCol = t * TILE_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] =
            (row < r1 && aCol < c1) ? A[row * c1 + aCol] : 0.0;

        // Load one tile of B (row-major): rows = tile strip, columns of B
        int bRow = t * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < c1 && col < c2) ? B[bRow * c2 + col] : 0.0;

        // all threads in the block must reach this point before they proceed
        __syncthreads();

        // Accumulate dot product for this tile
        #pragma unroll // Compiler optimization
        for (int k = 0; k < TILE_SIZE; k++)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < r1 && col < c2)
        C[row * c2 + col] = acc;
}

/**
 * @brief Multiplies two matrices: C = A * B.
 * @param A Input matrix of size (r1 x c1).
 * @param B Input matrix of size (c1 x c2).
 * @param C Output matrix of size (r1 x c2).
 */

/*
// Host wrapper using tiled kernel
extern "C" void mat_mat_mult(double *A, double *B, double *C,
                  size_t r1, size_t c1, size_t c2)
{

    printf("calling mat_mat_mult (CUDA VERSION)\n");
 
    double *d_A, *d_B, *d_C;
    size_t bytesA = r1 * c1 * sizeof(double);
    size_t bytesB = c1 * c2 * sizeof(double);
    size_t bytesC = r1 * c2 * sizeof(double);

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((c2 + TILE_SIZE - 1) / TILE_SIZE,
              (r1 + TILE_SIZE - 1) / TILE_SIZE);

    mat_mat_mult_kernel<<<grid, block>>>(d_A, d_B, d_C, r1, c1, c2);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
*/

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
/*
void mat_mat_mult(double *A, double *B, double *C, size_t r1, size_t c1, size_t c2) 
{
    for (size_t i = 0; i < r1; i++) {
        for (size_t j = 0; j < c2; j++) {
            C[i * c2 + j] = 0.0;
            for (size_t k = 0; k < c1; k++) {
                C[i * c2 + j] += A[i * c1 + k] * B[k * c2 + j];
            }
        }
    }
}
*/
/**
 * @brief Solves a system of linear equations Ax = b using LU decomposition.
 * This function decomposes A into L and U, then solves Ly = b (forward substitution)
 * and finally Ux = y (backward substitution).
 *
 * @param A The n x n coefficient matrix. This matrix will be modified in place.
 * @param b The n x 1 vector of constants.
 * @param x The n x 1 solution vector (output).
 * @param n The size of the system.
 * @return 0 on success, -1 on failure (e.g., singular matrix).
 */
extern "C" int solve_linear_system_lud(double *A, double *b, double *x, size_t n)
{
	int info, *ipiv;
	double *A_copy, *b_copy;
	size_t i;

	/* Allocate pivot array */
	ipiv = (int *)malloc(n * sizeof(int));
	if (!ipiv) {
		fprintf(stderr, "Error: malloc failed for ipiv\n");
		return -1;
	}

	/* Make copies of A and b because dgesv overwrites inputs */
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

	/* Solve system: A_copy * x = b_copy */
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

	/* Copy solution to output vector */
	for (i = 0; i < n; i++)
		x[i] = b_copy[i];

	free(ipiv);
	free(A_copy);
	free(b_copy);

	return 0;
}
/*
int solve_linear_system_lud(double *A, double *b, double *x, size_t n) 
{
    // --- Step 1: LU Decomposition (Doolittle's method) ---
    for (size_t i = 0; i < n; i++) {
        // Upper Triangle
        for (size_t k = i; k < n; k++) {
            double sum = 0.0;
            for (size_t j = 0; j < i; j++) {
                sum += A[i * n + j] * A[j * n + k];
            }
            A[i * n + k] = A[i * n + k] - sum;
        }
        // Lower Triangle
        for (size_t k = i + 1; k < n; k++) {
            if (A[i * n + i] == 0.0) {
                fprintf(stderr, "Error: Matrix is singular and cannot be inverted.\n");
                return -1; // Avoid division by zero
            }
            double sum = 0.0;
            for (size_t j = 0; j < i; j++) {
                sum += A[k * n + j] * A[j * n + i];
            }
            A[k * n + i] = (A[k * n + i] - sum) / A[i * n + i];
        }
    }

    // --- Step 2: Forward substitution (solves Ly = b for y) ---
    double y[n];
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < i; j++) {
            sum += A[i * n + j] * y[j];
        }
        y[i] = b[i] - sum;
    }

    // --- Step 3: Backward substitution (solves Ux = y for x) ---
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < (int)n; j++) {
            sum += A[i * n + j] * x[j];
        }
        if (A[i * n + i] == 0.0) {
            fprintf(stderr, "Error: Matrix is singular.\n");
            return -1;
        }
        x[i] = (y[i] - sum) / A[i * n + i];
    }

    return EXIT_SUCCESS; 
}
*/
