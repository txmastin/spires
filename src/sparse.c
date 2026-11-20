#include "sparse.h"
#include "math_utils.h"
#include <string.h>
#include <math.h>
#include <cblas.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

// Power iteration to estimate spectral radius.
// Only converges when the dominant eigenvalue is real; kept for reference.
double csr_spectral_radius_power_iteration(const struct csr_matrix *m, size_t n)
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

struct csr_matvec_ctx {
    const struct csr_matrix *m;
};

static void csr_matvec(void *ctx, const double *x, double *y)
{
    csr_spmv(((const struct csr_matvec_ctx *)ctx)->m, x, y);
}

double csr_spectral_radius(const struct csr_matrix *m, size_t n)
{
    (void)n;    /* n == m->n; kept for call-site compatibility */
    struct csr_matvec_ctx ctx = { m };
    return spectral_radius_arnoldi(csr_matvec, &ctx, m->n);
}

/* ---- edge_rows: O(nnz) stand-in for the dense construction scratch ---- */

int edge_rows_init(struct edge_rows *er, size_t n, size_t per_row_hint)
{
    er->n   = n;
    er->len = calloc(n, sizeof(size_t));
    er->cap = calloc(n, sizeof(size_t));
    er->col = calloc(n, sizeof(size_t *));
    er->val = calloc(n, sizeof(double *));
    if (!er->len || !er->cap || !er->col || !er->val) {
        edge_rows_free(er);
        return -1;
    }
    for (size_t i = 0; per_row_hint && i < n; i++) {
        er->col[i] = malloc(per_row_hint * sizeof(size_t));
        er->val[i] = malloc(per_row_hint * sizeof(double));
        if (!er->col[i] || !er->val[i]) {
            edge_rows_free(er);
            return -1;
        }
        er->cap[i] = per_row_hint;
    }
    return 0;
}

void edge_rows_free(struct edge_rows *er)
{
    if (er->col) for (size_t i = 0; i < er->n; i++) free(er->col[i]);
    if (er->val) for (size_t i = 0; i < er->n; i++) free(er->val[i]);
    free(er->col); free(er->val); free(er->len); free(er->cap);
    er->col = NULL; er->val = NULL; er->len = NULL; er->cap = NULL;
    er->n = 0;
}

/* Linear scan: rows hold ~connectivity*n entries, and only SMALL_WORLD's
 * rewiring probes repeatedly. The dense buffer answered this in O(1). */
static size_t er_find(const struct edge_rows *er, size_t i, size_t j)
{
    const size_t *c = er->col[i];
    for (size_t k = 0, e = er->len[i]; k < e; k++)
        if (c[k] == j) return k;
    return (size_t)-1;
}

static int er_grow(struct edge_rows *er, size_t i)
{
    if (er->len[i] == er->cap[i]) {
        /* 1.5x rather than 2x: with rows pre-sized to their expected length,
         * an overflowing row is usually only slightly over, and doubling
         * would waste most of what it claims. */
        size_t nc = er->cap[i] + er->cap[i] / 2 + 8;
        size_t *c = realloc(er->col[i], nc * sizeof(size_t));
        if (!c) return -1;
        er->col[i] = c;
        double *v = realloc(er->val[i], nc * sizeof(double));
        if (!v) return -1;
        er->val[i] = v;
        er->cap[i] = nc;
    }
    return 0;
}

void edge_rows_append(struct edge_rows *er, size_t i, size_t j, double w)
{
    if (er_grow(er, i) != 0) return;
    er->col[i][er->len[i]] = j;
    er->val[i][er->len[i]] = w;
    er->len[i]++;
}

int edge_rows_set(struct edge_rows *er, size_t i, size_t j, double w)
{
    size_t k = er_find(er, i, j);
    if (k != (size_t)-1) { er->val[i][k] = w; return 0; }
    if (er_grow(er, i) != 0) return -1;
    er->col[i][er->len[i]] = j;
    er->val[i][er->len[i]] = w;
    er->len[i]++;
    return 0;
}

int edge_rows_has(const struct edge_rows *er, size_t i, size_t j)
{
    size_t k = er_find(er, i, j);
    return k != (size_t)-1 && er->val[i][k] != 0.0;
}

void edge_rows_to_dense(const struct edge_rows *er, double *dense_out)
{
    size_t n = er->n;
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < er->len[i]; k++)
            dense_out[i * n + er->col[i][k]] = er->val[i][k];
}

/* Rows are accumulated in generation order, but a dense row scan emits
 * ascending columns, so each row is sorted before it is written out. */
struct er_pair { size_t col; size_t seq; double val; };

/* By column, then by insertion order, so a duplicated column leaves the most
 * recently written entry last in its run -- the one a dense assignment would
 * have kept. qsort is not stable, hence the explicit sequence number. */
static int er_pair_cmp(const void *a, const void *b)
{
    const struct er_pair *p = a, *q = b;
    if (p->col != q->col) return (p->col > q->col) - (p->col < q->col);
    return (p->seq > q->seq) - (p->seq < q->seq);
}

struct csr_matrix csr_build_from_rows(const struct edge_rows *er)
{
    struct csr_matrix m = {0};
    size_t n = er->n;
    m.n = n;

    size_t nnz = 0, widest = 0;
    for (size_t i = 0; i < n; i++) {
        if (er->len[i] > widest) widest = er->len[i];
        nnz += er->len[i];          /* upper bound; duplicates and zeros drop out below */
    }

    m.nnz = nnz;
    m.row_ptr = malloc((n + 1) * sizeof(size_t));
    m.col_idx = malloc(nnz * sizeof(size_t));
    m.values  = malloc(nnz * sizeof(double));
    struct er_pair *buf = widest ? malloc(widest * sizeof(*buf)) : NULL;
    if (!m.row_ptr || (nnz && (!m.col_idx || !m.values)) || (widest && !buf)) {
        free(m.row_ptr); free(m.col_idx); free(m.values); free(buf);
        struct csr_matrix empty = {0};
        return empty;
    }

    size_t idx = 0;
    for (size_t i = 0; i < n; i++) {
        m.row_ptr[i] = idx;
        size_t cnt = 0;
        for (size_t k = 0; k < er->len[i]; k++) {
            buf[cnt].col = er->col[i][k];
            buf[cnt].seq = k;
            buf[cnt].val = er->val[i][k];
            cnt++;
        }
        qsort(buf, cnt, sizeof(*buf), er_pair_cmp);
        for (size_t k = 0; k < cnt; k++) {
            if (k + 1 < cnt && buf[k + 1].col == buf[k].col)
                continue;                  /* superseded by a later write */
            /* Zeros are dropped only after the last write for the column has
             * been identified: a trailing 0.0 erases the edge, exactly as
             * assigning 0.0 into a dense cell would. */
            if (buf[k].val == 0.0)
                continue;
            m.col_idx[idx] = buf[k].col;
            m.values[idx]  = buf[k].val;
            idx++;
        }
    }
    m.row_ptr[n] = idx;
    /* nnz was an upper bound: duplicate columns collapse above, so the real
     * count is only known once the rows have been written. */
    m.nnz = idx;

    free(buf);
    return m;
}
