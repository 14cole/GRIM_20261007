/* GHOST far-field Galerkin block quadrature. No third-party source.
   Mirrors the numpy far pass in twod/operators.py operation for operation:
   same distances, the same Horner table evaluation as table.c, and the same
   accumulation order, so each far pair receives the same partial sums.
   Pairs outside `far` are skipped; the caller scales them by zero anyway.
   Every call owns its outputs; immutable tables can be shared across threads. */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define GHOST_EPS 1e-12
#define GHOST_PI 3.14159265358979323846
#define MAX_WIDTH 8

typedef struct {
    int intervals, degree;
    const double *bounds, *coeff;
    double k_re, k_im;
} table_t;

/* Large-argument Hankel expansion for distances beyond a partial table:
   (j/4) H0^(2)(kr) and (j/4) k H1^(2)(kr). Tables stop at -Im(k) r = 128, so
   |kr| >= 128 and eight terms are far below double precision. */
static void asymptotic(const table_t *t, double r, double *out) {
    double complex k = t->k_re + I * t->k_im;
    double complex z = k * r;
    double complex root = csqrt(2.0 / (GHOST_PI * z));
    for (int nu = 0; nu < 2; ++nu) {
        double mu = 4.0 * nu * nu;
        double complex term = 1.0, sum = 1.0, factor = -I / z;
        for (int m = 1; m < 8; ++m) {
            double odd = 2.0 * m - 1.0;
            term *= (mu - odd * odd) / (8.0 * m) * factor;
            sum += term;
        }
        double complex phase = cexp(-I * (z - nu * GHOST_PI / 2.0 - GHOST_PI / 4.0));
        double complex value = 0.25 * I * root * phase * sum;
        if (nu == 1) value *= k;
        out[2 * nu] = creal(value);
        out[2 * nu + 1] = cimag(value);
    }
}

/* Returns 0, or 1 below the table (the caller falls back to exact kernels). */
static int kernel(const table_t *t, double r, int *hint, double *out) {
    const double *bounds = t->bounds;
    int intervals = t->intervals, degree = t->degree;
    if (!(r >= bounds[0])) return 1;
    if (r > bounds[intervals]) {
        asymptotic(t, r, out);
        return 0;
    }
    /* The previous interval usually still holds r; otherwise search exactly as
       table.c does. Bounds increase strictly, so both pick the same interval. */
    int lo = *hint;
    if (!(lo >= 0 && lo < intervals && r >= bounds[lo] && (r < bounds[lo + 1] || lo + 1 == intervals))) {
        int hi = intervals;
        lo = 0;
        while (lo + 1 < hi) {
            int mid = lo + (hi - lo) / 2;
            if (r < bounds[mid]) hi = mid; else lo = mid;
        }
        *hint = lo;
    }
    double x = (r - bounds[lo]) / (bounds[lo + 1] - bounds[lo]);
    const double *base = t->coeff + (int64_t)lo * (degree + 1) * 4;
    for (int c = 0; c < 2; ++c) {
        int offset = 2 * c;
        double re = base[4 * degree + offset], im = base[4 * degree + offset + 1];
        for (int j = degree - 1; j >= 0; --j) {
            re = re * x + base[4 * j + offset];
            im = im * x + base[4 * j + offset + 1];
        }
        out[2 * c] = re;
        out[2 * c + 1] = im;
    }
    return 0;
}

/* acc[ab] (+)= c * value, matching numpy's complex-by-real products. */
static inline void first_or_add(double *acc, int first, double re, double im, double c) {
    if (first) {
        acc[0] = re * c;
        acc[1] = im * c;
    } else {
        acc[0] = acc[0] + re * c;
        acc[1] = acc[1] + im * c;
    }
}

/* The public wrapper supplies constant widths for the linear, quadratic and
   cubic bases. Inlining lets the compiler unroll those short contractions;
   distances, interpolation and the order of every sum remain unchanged. */
static inline __attribute__((always_inline)) int ghost_far_block_impl(
    int64_t mb, int64_t nb, int q, int width,
    const double *obs_pts, const double *src_pts,
    const double *qw, const double *phi,
    const double *obs_norm, const double *src_norm,
    const uint8_t *far,
    int obs_normal_deriv, int want_s, int want_k, int mirrored,
    double k_re, double k_im,
    int intervals, const double *bounds, int degree, const double *coeff,
    double *acc_s, double *acc_k, double *acc_kt) {
    if (mb < 0 || nb < 0 || q < 1 || width < 1 || width > MAX_WIDTH || intervals < 1 ||
        degree < 0 || degree > 32 || (!want_s && !want_k) ||
        (want_s && !acc_s) || (want_k && !acc_k) || (want_k && mirrored && !acc_kt))
        return 3;
    table_t table = {intervals, degree, bounds, coeff, k_re, k_im};
    int with_kt = want_k && mirrored;
    int64_t plane = mb * nb;
    for (int64_t i = 0; i < mb; ++i) {
        const double *on = obs_norm + 2 * i;
        int hint = -1;
        for (int64_t j = 0; j < nb; ++j) {
            int64_t pair = i * nb + j;
            if (!far[pair]) continue;
            const double *sn = src_norm + 2 * j;
            const double *n_ij = obs_normal_deriv ? on : sn;
            const double *n_ji = obs_normal_deriv ? sn : on;
            for (int qi = 0; qi < q; ++qi) {
                double part_s[2 * MAX_WIDTH], part_k[2 * MAX_WIDTH], part_kt[2 * MAX_WIDTH];
                double ox = obs_pts[(i * q + qi) * 2], oy = obs_pts[(i * q + qi) * 2 + 1];
                for (int qj = 0; qj < q; ++qj) {
                    double dx = ox - src_pts[(j * q + qj) * 2];
                    double dy = oy - src_pts[(j * q + qj) * 2 + 1];
                    double dist = sqrt(dx * dx + dy * dy);
                    if (!(dist >= GHOST_EPS)) dist = GHOST_EPS;
                    double value[4];
                    if (kernel(&table, dist, &hint, value)) return 2;
                    double w = qw[qj];
                    const double *phi_s = phi + qj * width;
                    int first = qj == 0;
                    double dk_re = 0., dk_im = 0.;
                    if (want_k) {
                        double proj = dx * n_ij[0] + dy * n_ij[1];
                        proj = proj / dist;
                        dk_re = value[2] * proj;
                        dk_im = value[3] * proj;
                        if (obs_normal_deriv) { dk_re = -dk_re; dk_im = -dk_im; }
                    }
                    for (int b = 0; b < width; ++b) {
                        double c = w * phi_s[b];
                        if (want_s) first_or_add(part_s + 2 * b, first, value[0], value[1], c);
                        if (want_k) first_or_add(part_k + 2 * b, first, dk_re, dk_im, c);
                    }
                    if (with_kt) {
                        double proj = dx * n_ji[0] + dy * n_ji[1];
                        proj = proj / dist;
                        double re = value[2] * proj, im = value[3] * proj;
                        if (!obs_normal_deriv) { re = -re; im = -im; }
                        for (int a = 0; a < width; ++a)
                            first_or_add(part_kt + 2 * a, first, re, im, w * phi_s[a]);
                    }
                }
                double wo = qw[qi];
                const double *phi_o = phi + qi * width;
                for (int a = 0; a < width; ++a) {
                    double c = wo * phi_o[a];
                    for (int b = 0; b < width; ++b) {
                        int64_t at = ((int64_t)(width * a + b) * plane + pair) * 2;
                        if (want_s) {
                            acc_s[at] = acc_s[at] + part_s[2 * b] * c;
                            acc_s[at + 1] = acc_s[at + 1] + part_s[2 * b + 1] * c;
                        }
                        if (want_k) {
                            acc_k[at] = acc_k[at] + part_k[2 * b] * c;
                            acc_k[at + 1] = acc_k[at + 1] + part_k[2 * b + 1] * c;
                        }
                    }
                }
                if (with_kt) {
                    for (int b = 0; b < width; ++b) {
                        double c = wo * phi_o[b];
                        for (int a = 0; a < width; ++a) {
                            int64_t at = ((int64_t)(width * a + b) * plane + pair) * 2;
                            acc_kt[at] = acc_kt[at] + part_kt[2 * a] * c;
                            acc_kt[at + 1] = acc_kt[at + 1] + part_kt[2 * a + 1] * c;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

EXPORT int ghost_far_block(
    int64_t mb, int64_t nb, int q, int width,
    const double *obs_pts, const double *src_pts,
    const double *qw, const double *phi,
    const double *obs_norm, const double *src_norm,
    const uint8_t *far,
    int obs_normal_deriv, int want_s, int want_k, int mirrored,
    double k_re, double k_im,
    int intervals, const double *bounds, int degree, const double *coeff,
    double *acc_s, double *acc_k, double *acc_kt) {
    if (width == 2) return ghost_far_block_impl(mb, nb, q, 2, obs_pts, src_pts, qw, phi, obs_norm, src_norm, far, obs_normal_deriv, want_s, want_k, mirrored, k_re, k_im, intervals, bounds, degree, coeff, acc_s, acc_k, acc_kt);
    if (width == 3) return ghost_far_block_impl(mb, nb, q, 3, obs_pts, src_pts, qw, phi, obs_norm, src_norm, far, obs_normal_deriv, want_s, want_k, mirrored, k_re, k_im, intervals, bounds, degree, coeff, acc_s, acc_k, acc_kt);
    if (width == 4) return ghost_far_block_impl(mb, nb, q, 4, obs_pts, src_pts, qw, phi, obs_norm, src_norm, far, obs_normal_deriv, want_s, want_k, mirrored, k_re, k_im, intervals, bounds, degree, coeff, acc_s, acc_k, acc_kt);
    return ghost_far_block_impl(mb, nb, q, width, obs_pts, src_pts, qw, phi, obs_norm, src_norm, far, obs_normal_deriv, want_s, want_k, mirrored, k_re, k_im, intervals, bounds, degree, coeff, acc_s, acc_k, acc_kt);
}

/* Tile scatter with the weighting fused in: SystemScatter's
   ufunc.at of (values * scale) * weight[row], without forming the weighted tile.
   Each destination entry gets v*s (complex by real, or complex by complex), then
   times the complex route weight, then one add -- exactly the numpy operations.
   NumPy's complex product on FMA hardware rounds as re = fma(ar, br, -(ai*bi)),
   im = fma(ar, bi, ai*br); `fused` selects that form (the caller probes numpy),
   otherwise the plain two-rounding form. -ffp-contract=off keeps both exact. */
#define GHOST_ROW_BLOCK 16

#define CMUL_PLAIN(re, im, br, bi) do {                                                    \
        double r_ = (re) * (br) - (im) * (bi);                                             \
        double i_ = (re) * (bi) + (im) * (br);                                             \
        (re) = r_; (im) = i_; } while (0)
#define CMUL_FUSED(re, im, br, bi) do {                                                    \
        double r_ = __builtin_fma((re), (br), -((im) * (bi)));                             \
        double i_ = __builtin_fma((re), (bi), (im) * (br));                                \
        (re) = r_; (im) = i_; } while (0)

/* One weighted contribution: v * s (kind 1 real, 2 complex), then * w, then add. */
#define SCATTER_ENTRY(CMUL)                                                                \
    do {                                                                                   \
        const double *v = values + 2 * (b * vb + i * vi + j * vj);                         \
        double re = v[0], im = v[1];                                                       \
        if (scale_kind == 1) {                                                             \
            double f = scale[i * si + j * sj];                                             \
            re = re * f; im = im * f;                                                      \
        } else if (scale_kind == 2) {                                                      \
            const double *f = scale + 2 * (i * si + j * sj);                               \
            CMUL(re, im, f[0], f[1]);                                                      \
        }                                                                                  \
        if (weights) {                                                                     \
            const double *w = weights + 2 * rows[i];                                       \
            CMUL(re, im, w[0], w[1]);                                                      \
        }                                                                                  \
        double *t = matrix + 2 * (r * row_stride + c * column_stride);                     \
        t[0] = t[0] + re; t[1] = t[1] + im;                                                \
    } while (0)

#define DEFINE_SCATTER(NAME, ATTRIBUTE, CMUL)                                              \
static ATTRIBUTE void NAME(                                                                \
    int64_t m, int64_t n, int width, const int64_t *rows, const int64_t *columns,          \
    const int64_t *row_map, const int64_t *column_map,                                     \
    const double *values, int64_t vb, int64_t vi, int64_t vj,                              \
    int scale_kind, const double *scale, int64_t si, int64_t sj, const double *weights,    \
    double *matrix, int64_t row_stride, int64_t column_stride, int blocked) {              \
    for (int64_t b = 0; b < width; ++b) {                                                  \
        if (blocked) {                                                                     \
            for (int64_t i0 = 0; i0 < m; i0 += GHOST_ROW_BLOCK) {                          \
                int64_t i1 = i0 + GHOST_ROW_BLOCK < m ? i0 + GHOST_ROW_BLOCK : m;          \
                for (int64_t j = 0; j < n; ++j) {                                          \
                    int64_t c = column_map[columns[j * width + b]];                        \
                    if (c < 0) continue;                                                   \
                    for (int64_t i = i0; i < i1; ++i) {                                    \
                        int64_t r = row_map[rows[i]];                                      \
                        if (r < 0) continue;                                               \
                        SCATTER_ENTRY(CMUL);                                               \
                    }                                                                      \
                }                                                                          \
            }                                                                              \
            continue;                                                                      \
        }                                                                                  \
        for (int64_t i = 0; i < m; ++i) {                                                  \
            int64_t r = row_map[rows[i]];                                                  \
            if (r < 0) continue;                                                           \
            for (int64_t j = 0; j < n; ++j) {                                              \
                int64_t c = column_map[columns[j * width + b]];                            \
                if (c < 0) continue;                                                       \
                SCATTER_ENTRY(CMUL);                                                       \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
}

DEFINE_SCATTER(scatter_plain, , CMUL_PLAIN)
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
DEFINE_SCATTER(scatter_fused, __attribute__((target("fma"))), CMUL_FUSED)
#define GHOST_HAVE_FUSED 1
#else
#define GHOST_HAVE_FUSED 0
#endif

/* 1 when the mapped destination rows of `rows` are distinct, else 0. */
static int rows_distinct(int64_t m, const int64_t *rows, const int64_t *row_map) {
    int64_t lo = INT64_MAX, hi = -1, count = 0, previous = -1;
    int increasing = 1;
    for (int64_t i = 0; i < m; ++i) {
        int64_t r = row_map[rows[i]];
        if (r < 0) continue;
        if (r <= previous) increasing = 0;
        previous = r;
        if (r < lo) lo = r;
        if (r > hi) hi = r;
        ++count;
    }
    if (increasing || count < 2) return 1;
    int64_t span = hi - lo + 1;
    if (span > ((int64_t)1 << 26)) return 0;
    unsigned char *seen = (unsigned char *)calloc((size_t)((span + 7) / 8), 1);
    if (!seen) return 0;
    int distinct = 1;
    for (int64_t i = 0; i < m && distinct; ++i) {
        int64_t r = row_map[rows[i]];
        if (r < 0) continue;
        int64_t k = r - lo;
        if (seen[k >> 3] & (1u << (k & 7))) distinct = 0;
        seen[k >> 3] |= (unsigned char)(1u << (k & 7));
    }
    free(seen);
    return distinct;
}

/* matrix[row_map[rows[i]], column_map[columns[j*width + b]]] += (v(b,i,j) * s(i,j)) * w[rows[i]]
   for b, i, j, skipping unmapped nodes. v(b,i,j) = values[b*vb + i*vi + j*vj] and
   s(i,j) = scale[i*si + j*sj] use element strides (complex values; real or complex
   scale), so a transposed or strided accumulator plane needs no contiguous copy;
   scale_kind 0/1/2 is no, real or complex scale; weights may be NULL (unit weight).

   Entries are added in the order of ufunc.at over (b, i, j). When the destination
   rows run contiguously (row_stride 1) and the rows map to distinct destination
   rows, the loop is blocked over 16 rows with j outside: for fixed b an entry can
   then repeat only across j, whose order is kept, so every sum is bitwise the same
   while writes walk down contiguous columns. Returns 1 on an index outside the
   destination, 3 on invalid arguments, 4 if a fused product was requested without
   compiler support. */
EXPORT int ghost_scatter_tile(
    int64_t m, int64_t n, int width,
    const int64_t *rows, const int64_t *columns,
    const int64_t *row_map, const int64_t *column_map, int64_t node_count,
    const double *values, int64_t vb, int64_t vi, int64_t vj,
    int scale_kind, const double *scale, int64_t si, int64_t sj,
    const double *weights, int fused,
    double *matrix, int64_t nrows, int64_t ncols, int64_t row_stride, int64_t column_stride) {
    if (m < 0 || n < 0 || width < 1 || node_count < 0 || nrows < 0 || ncols < 0 ||
        scale_kind < 0 || scale_kind > 2 || (scale_kind && !scale) || !values || !matrix) return 3;
    if (fused && !GHOST_HAVE_FUSED) return 4;
    for (int64_t i = 0; i < m; ++i) {
        if (rows[i] < 0 || rows[i] >= node_count) return 1;
        if (row_map[rows[i]] >= nrows) return 1;
    }
    for (int64_t j = 0; j < n * width; ++j) {
        if (columns[j] < 0 || columns[j] >= node_count) return 1;
        if (column_map[columns[j]] >= ncols) return 1;
    }
    int blocked = row_stride == 1 && rows_distinct(m, rows, row_map);
#if GHOST_HAVE_FUSED
    if (fused) {
        scatter_fused(m, n, width, rows, columns, row_map, column_map, values, vb, vi, vj,
                      scale_kind, scale, si, sj, weights, matrix, row_stride, column_stride, blocked);
        return 0;
    }
#endif
    scatter_plain(m, n, width, rows, columns, row_map, column_map, values, vb, vi, vj,
                  scale_kind, scale, si, sj, weights, matrix, row_stride, column_stride, blocked);
    return 0;
}

/* 1 when this build can form NumPy's fused complex product. */
EXPORT int ghost_scatter_has_fused(void) { return GHOST_HAVE_FUSED; }

/* matrix[row_map[rows[i]], column_map[columns[j, b]]] += values[b, i, j] for b, i, j in
   order, skipping unmapped nodes: SystemScatter.scatter_add_columns for one route. The
   caller forms the weighted values with numpy, so sums match its ufunc.at exactly. */
EXPORT int ghost_scatter_columns(
    int64_t m, int64_t n, int width,
    const int64_t *rows, const int64_t *columns,
    const int64_t *row_map, const int64_t *column_map, int64_t node_count,
    const double *values,
    double *matrix, int64_t nrows, int64_t ncols, int64_t row_stride, int64_t column_stride) {
    if (m < 0 || n < 0 || width < 1 || node_count < 0 || nrows < 0 || ncols < 0) return 3;
    for (int64_t i = 0; i < m; ++i)
        if (rows[i] < 0 || rows[i] >= node_count) return 1;
    for (int64_t j = 0; j < n * width; ++j)
        if (columns[j] < 0 || columns[j] >= node_count) return 1;
    for (int b = 0; b < width; ++b) {
        for (int64_t i = 0; i < m; ++i) {
            int64_t r = row_map[rows[i]];
            if (r < 0) continue;
            if (r >= nrows) return 1;
            const double *v = values + ((int64_t)b * m + i) * n * 2;
            for (int64_t j = 0; j < n; ++j) {
                int64_t c = column_map[columns[j * width + b]];
                if (c < 0) continue;
                if (c >= ncols) return 1;
                double *target = matrix + 2 * (r * row_stride + c * column_stride);
                target[0] = target[0] + v[2 * j];
                target[1] = target[1] + v[2 * j + 1];
            }
        }
    }
    return 0;
}
