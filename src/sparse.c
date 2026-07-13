#include "sparse.h"
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <omp.h>

struct csr_matrix csr_build_from_dense(const double *dense, size_t n)
{
    struct csr_matrix m = {0};
    m.n = n;

    size_t nnz = 0;
    for (size_t k = 0; k < n * n; k++)
        if (dense[k] != 0.0)
            nnz++;

    m.nnz = nnz;
    m.row_ptr = malloc((n + 1) * sizeof(size_t));
    m.col_idx = malloc(nnz * sizeof(size_t));
    m.values  = malloc(nnz * sizeof(double));
    if (!m.row_ptr || (nnz && (!m.col_idx || !m.values))) {
        free(m.row_ptr); free(m.col_idx); free(m.values);
        struct csr_matrix empty = {0};
        return empty;
    }

    size_t idx = 0;
    for (size_t i = 0; i < n; i++) {
        m.row_ptr[i] = idx;
        const double *row = &dense[i * n];
        for (size_t j = 0; j < n; j++) {
            if (row[j] != 0.0) {
                m.col_idx[idx] = j;
                m.values[idx]  = row[j];
                idx++;
            }
        }
    }
    m.row_ptr[n] = idx;

    return m;
}

void csr_free(struct csr_matrix *m)
{
    if (!m)
        return;
    free(m->row_ptr);
    free(m->col_idx);
    free(m->values);
    m->row_ptr = NULL;
    m->col_idx = NULL;
    m->values  = NULL;
    m->n = 0;
    m->nnz = 0;
}

void csr_to_dense(const struct csr_matrix *m, double *dense_out)
{
    memset(dense_out, 0, m->n * m->n * sizeof(double));
    for (size_t i = 0; i < m->n; i++) {
        for (size_t k = m->row_ptr[i]; k < m->row_ptr[i + 1]; k++) {
            dense_out[i * m->n + m->col_idx[k]] = m->values[k];
        }
    }
}

double csr_row_dot(const struct csr_matrix *m, size_t row, const double *x)
{
    double sum = 0.0;
    size_t start = m->row_ptr[row];
    size_t end   = m->row_ptr[row + 1];
    for (size_t k = start; k < end; k++)
        sum += m->values[k] * x[m->col_idx[k]];
    return sum;
}

void csr_spmv(const struct csr_matrix *m, const double *x, double *y)
{
    #pragma omp parallel for
    for (size_t i = 0; i < m->n; i++) {
        y[i] = csr_row_dot(m, i, x);
    }
}

void csr_scale(struct csr_matrix *m, double factor)
{
    if (m->nnz)
        cblas_dscal((int)m->nnz, factor, m->values, 1);
}

// Power iteration to estimate spectral radius (mirrors math_utils.c:calc_spectral_radius)
double csr_spectral_radius(const struct csr_matrix *m, size_t n)
{
    const unsigned int max_iter = 1000;
    const double tol = 1e-6;

    double x[n];
    double y[n];

    for (size_t i = 0; i < n; i++)
        x[i] = 1.0;

    double lambda_old = 0.0;
    double lambda_new = 0.0;

    for (unsigned int iter = 0; iter < max_iter; iter++) {
        csr_spmv(m, x, y);

        lambda_new = 0.0;
        for (size_t i = 0; i < n; i++) {
            double abs_y = fabs(y[i]);
            if (abs_y > lambda_new)
                lambda_new = abs_y;
        }

        for (size_t i = 0; i < n; i++)
            x[i] = y[i] / lambda_new;

        if (fabs(lambda_new - lambda_old) < tol)
            break;

        lambda_old = lambda_new;
    }

    return lambda_new;
}
