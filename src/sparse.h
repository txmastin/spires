#ifndef SPARSE_H
#define SPARSE_H

#include <stdlib.h>

/* Compressed Sparse Row matrix. row_ptr has length n+1; col_idx/values have
 * length nnz. Row i's entries live in [row_ptr[i], row_ptr[i+1]). */
struct csr_matrix {
    size_t n;
    size_t nnz;
    size_t *row_ptr;
    size_t *col_idx;
    double *values;
};

struct csr_matrix csr_build_from_dense(const double *dense, size_t n);
void   csr_free(struct csr_matrix *m);
void   csr_to_dense(const struct csr_matrix *m, double *dense_out);
double csr_row_dot(const struct csr_matrix *m, size_t row, const double *x);
void   csr_spmv(const struct csr_matrix *m, const double *x, double *y);
void   csr_scale(struct csr_matrix *m, double factor);
double csr_spectral_radius(const struct csr_matrix *m, size_t n);

#endif // SPARSE_H
