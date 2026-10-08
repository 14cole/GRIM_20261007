/*
 * Phase-7c native sampling kernel for the BoR streaming assembly
 * (bor_streaming.py).  Fills the azimuthal integrand tiles that dominate
 * the streamed far-block build:
 *
 *   sample_g     g(xi)  = exp(-j k R)/(4 pi R)          (EFIE Green orders)
 *   sample_mfie  the four MFIE bracket functions * p(R)
 *   sample_ibc   the four IBC (rotated-PV) bracket functions * p(R)
 *   with p(R) = (1 + j k R) exp(-j k R)/(4 pi R^3)
 *
 * Real wavenumber only (the streaming path serves the exterior/air region;
 * bor_streaming falls back to the NumPy sampler for complex k).  Output
 * arrays are interleaved complex doubles laid out [rows, np, nxi].
 * Coincident points are clamped to R = 1e-30 -> large-but-finite garbage;
 * the caller zeroes every near/adjacent-element pair after the FFT exactly
 * as the table path does.
 *
 * Later entry points (all optional for the Python callers, which detect them
 * with hasattr and fall back to NumPy): near_mfie, near_brackets,
 * sample_g_pairs, sample_brackets_pairs (paired, real/complex k), and the
 * near-rule kernels trig_moments, parity_moments, near_green and
 * near_brackets_stable (all skip exp(ki R) for a real k: GHOST_DECAY).
 *
 * Build and load-check with ghost_backend/bor/native/build_kernel.py (it
 * compiles -O3 -std=c99 [-fopenmp], no relaxed floating point).
 */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define R_MIN 1e-30
/* exp(ki R) of exp(-j k R) with k = kr + j ki: exactly 1 for a real k, so
 * skipping the call there changes no value. */
#define GHOST_DECAY(ki, x) ((ki) == 0.0 ? 1.0 : exp(x))
#if defined(_MSC_VER)
#define GHOST_RESTRICT __restrict
#else
#define GHOST_RESTRICT restrict
#endif

void sample_g(int nr, int np_, int nxi,
              const double *GHOST_RESTRICT rho_p,
              const double *GHOST_RESTRICT z_p,
              const double *GHOST_RESTRICT rho_q,
              const double *GHOST_RESTRICT z_q,
              double k, const double *GHOST_RESTRICT sin2_tab,
              double *GHOST_RESTRICT out)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < np_; j++) {
            double dr = rho_p[i] - rho_q[j];
            double dz = z_p[i] - z_q[j];
            double d2 = dr * dr + dz * dz;
            double rr4 = 4.0 * rho_p[i] * rho_q[j];
            double *o = out + 2 * (size_t)nxi * ((size_t)i * np_ + j);
            for (int l = 0; l < nxi; l++) {
                double R = sqrt(d2 + rr4 * sin2_tab[l]);
                if (R < R_MIN) R = R_MIN;
                double a = 1.0 / (4.0 * M_PI * R);
                double kr = k * R;
                o[2 * l] = a * cos(kr);
                o[2 * l + 1] = -a * sin(kr);
            }
        }
    }
}

/* shared bracket-point core: computes p(R) (complex) and R components */
static inline void pR_floor(double Rx, double Ry, double Rz, double k, double floor_R,
                            double *R_out, double *p_re, double *p_im)
{
    double R = sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
    if (R < floor_R) R = floor_R;
    double pre = 1.0 / (4.0 * M_PI * R * R * R);
    double kr = k * R;
    double c = cos(kr), s = sin(kr);
    /* (1 + j kr)(c - j s) = (c + kr s) + j (kr c - s) */
    *p_re = pre * (c + kr * s);
    *p_im = pre * (kr * c - s);
    *R_out = R;
}

static inline void pR(double Rx, double Ry, double Rz, double k,
                      double *R_out, double *p_re, double *p_im)
{
    pR_floor(Rx, Ry, Rz, k, R_MIN, R_out, p_re, p_im);
}

void sample_mfie(int nr, int np_, int nxi,
                 const double *rho_p, const double *z_p,
                 const double *tr_p, const double *tz_p,
                 const double *rho_q, const double *z_q,
                 const double *tr_q, const double *tz_q,
                 double k, const double *cx_tab, const double *sx_tab,
                 double *o_tt, double *o_tf, double *o_ft, double *o_ff)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < np_; j++) {
            double Rz = z_p[i] - z_q[j];
            size_t base = 2 * (size_t)nxi * ((size_t)i * np_ + j);
            double *tt = o_tt + base, *tf = o_tf + base;
            double *ft = o_ft + base, *ff = o_ff + base;
            for (int l = 0; l < nxi; l++) {
                double cx = cx_tab[l], sx = sx_tab[l];
                double Rx = rho_p[i] - rho_q[j] * cx;
                double Ry = rho_q[j] * sx;
                double R, p_re, p_im;
                pR(Rx, Ry, Rz, k, &R, &p_re, &p_im);
                double WtR = tr_p[i] * Rx + tz_p[i] * Rz;
                double WfR = Ry;
                double nR = -tz_p[i] * Rx + tr_p[i] * Rz;
                double n_tq = -tz_p[i] * tr_q[j] * cx + tr_p[i] * tz_q[j];
                double n_fq = -tz_p[i] * sx;
                double Wt_tq = tr_p[i] * tr_q[j] * cx + tz_p[i] * tz_q[j];
                double Wt_fq = tr_p[i] * sx;
                double Wf_tq = -tr_q[j] * sx;
                double Wf_fq = cx;
                double f;
                f = -(WtR * n_tq - Wt_tq * nR);
                tt[2 * l] = f * p_re; tt[2 * l + 1] = f * p_im;
                f = -(WtR * n_fq - Wt_fq * nR);
                tf[2 * l] = f * p_re; tf[2 * l + 1] = f * p_im;
                f = -(WfR * n_tq - Wf_tq * nR);
                ft[2 * l] = f * p_re; ft[2 * l + 1] = f * p_im;
                f = -(WfR * n_fq - Wf_fq * nR);
                ff[2 * l] = f * p_re; ff[2 * l + 1] = f * p_im;
            }
        }
    }
}

void sample_ibc(int nr, int np_, int nxi,
                const double *rho_p, const double *z_p,
                const double *tr_p, const double *tz_p,
                const double *rho_q, const double *z_q,
                const double *tr_q, const double *tz_q,
                double k, const double *cx_tab, const double *sx_tab,
                double *o_tt, double *o_tf, double *o_ft, double *o_ff)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < np_; j++) {
            double Rz = z_p[i] - z_q[j];
            size_t base = 2 * (size_t)nxi * ((size_t)i * np_ + j);
            double *tt = o_tt + base, *tf = o_tf + base;
            double *ft = o_ft + base, *ff = o_ff + base;
            for (int l = 0; l < nxi; l++) {
                double cx = cx_tab[l], sx = sx_tab[l];
                double Rx = rho_p[i] - rho_q[j] * cx;
                double Ry = rho_q[j] * sx;
                double R, p_re, p_im;
                pR(Rx, Ry, Rz, k, &R, &p_re, &p_im);
                double Wt_nq = -tr_p[i] * tz_q[j] * cx + tz_p[i] * tr_q[j];
                double Wf_nq = tz_q[j] * sx;
                double D = rho_p[i] * cx - rho_q[j];
                double R_tq = tr_q[j] * D + tz_q[j] * Rz;
                double R_fq = rho_p[i] * sx;
                double R_nq = -tz_q[j] * D + tr_q[j] * Rz;
                double Wt_tq = tr_p[i] * tr_q[j] * cx + tz_p[i] * tz_q[j];
                double Wt_fq = tr_p[i] * sx;
                double Wf_tq = -tr_q[j] * sx;
                double Wf_fq = cx;
                double f;
                f = Wt_nq * R_tq - Wt_tq * R_nq;
                tt[2 * l] = f * p_re; tt[2 * l + 1] = f * p_im;
                f = Wt_nq * R_fq - Wt_fq * R_nq;
                tf[2 * l] = f * p_re; tf[2 * l + 1] = f * p_im;
                f = Wf_nq * R_tq - Wf_tq * R_nq;
                ft[2 * l] = f * p_re; ft[2 * l + 1] = f * p_im;
                f = Wf_nq * R_fq - Wf_fq * R_nq;
                ff[2 * l] = f * p_re; ff[2 * l + 1] = f * p_im;
            }
        }
    }
}

/*
 * Paired near-field bracket sampler.
 *
 * sample_mfie walks an nr x np_ outer product against one shared xi grid,
 * which is the streamed far-block layout.  The near rule instead holds a flat
 * list of point PAIRS, and its sinh-graded quadrature gives every pair its own
 * xi row.  Same bracket algebra either way, so this entry takes the paired
 * layout and a grid that is shared (xi_per_pair = 0) or per pair
 * (xi_per_pair = 1), covering _mfie_brackets and the near rule's
 * brackets_grid.  Output arrays are interleaved complex doubles [npair, nxi].
 *
 * NEAR_R_MIN mirrors the 1e-300 clamp both NumPy callers apply, not the
 * 1e-30 the streamed path uses.
 */
#define NEAR_R_MIN 1e-300

/* Paired MFIE/IBC sampling for real or complex k.  No relaxed floating-point
 * flags: the Python reference and its quadrature/error checks remain valid.
 * family == 0 is MFIE; family == 1 is the source-normal IBC operator. */
void near_brackets(int family, int npair, int nxi,
                   const double *rho_p, const double *z_p,
                   const double *tr_p, const double *tz_p,
                   const double *rho_q, const double *z_q,
                   const double *tr_q, const double *tz_q,
                   double kr, double ki, const double *xi, int xi_per_pair,
                   double *o_tt, double *o_tf, double *o_ft, double *o_ff)
{
    /* The bounded Python pair queue owns near concurrency. Avoid multiplying
     * its worker count by an unconfigured OpenMP team in each caller thread. */
    for (int i = 0; i < npair; i++) {
        const double *grid = xi + (xi_per_pair ? (size_t)i * nxi : 0);
        double rp = rho_p[i], rq = rho_q[i], rz = z_p[i] - z_q[i];
        double tp = tr_p[i], zp = tz_p[i], tq = tr_q[i], zq = tz_q[i];
        for (int l = 0; l < nxi; l++) {
            double cx = cos(grid[l]), sx = sin(grid[l]);
            double rx = rp - rq * cx, ry = rq * sx;
            double r = fmax(sqrt(rx*rx + ry*ry + rz*rz), NEAR_R_MIN);
            double a = kr*r, b = ki*r, c = cos(a), s = sin(a);
            double scale = GHOST_DECAY(ki, b) / (4.0*M_PI*r*r*r);
            double pre = ((1.0-b)*c + a*s)*scale;
            double pim = (a*c - (1.0-b)*s)*scale;
            double wt_tq = tp*tq*cx + zp*zq, wt_fq = tp*sx;
            double wf_tq = -tq*sx, wf_fq = cx;
            double tt, tf, ft, ff;
            if (family == 0) {
                double wtr = tp*rx + zp*rz, nr = -zp*rx + tp*rz;
                double ntq = -zp*tq*cx + tp*zq, nfq = -zp*sx;
                tt = -(wtr*ntq - wt_tq*nr);
                tf = -(wtr*nfq - wt_fq*nr);
                ft = -(ry*ntq - wf_tq*nr);
                ff = -(ry*nfq - wf_fq*nr);
            } else {
                double wtnq = -tp*zq*cx + zp*tq, wfnq = zq*sx;
                double d = rp*cx - rq;
                double rtq = tq*d + zq*rz, rfq = rp*sx, rnq = -zq*d + tq*rz;
                tt = wtnq*rtq - wt_tq*rnq;
                tf = wtnq*rfq - wt_fq*rnq;
                ft = wfnq*rtq - wf_tq*rnq;
                ff = wfnq*rfq - wf_fq*rnq;
            }
            size_t j = 2*((size_t)i*nxi + l);
            o_tt[j] = tt*pre; o_tt[j+1] = tt*pim;
            o_tf[j] = tf*pre; o_tf[j+1] = tf*pim;
            o_ft[j] = ft*pre; o_ft[j+1] = ft*pim;
            o_ff[j] = ff*pre; o_ff[j+1] = ff*pim;
        }
    }
}

void near_mfie(int npair, int nxi,
               const double *GHOST_RESTRICT rho_p,
               const double *GHOST_RESTRICT z_p,
               const double *GHOST_RESTRICT tr_p,
               const double *GHOST_RESTRICT tz_p,
               const double *GHOST_RESTRICT rho_q,
               const double *GHOST_RESTRICT z_q,
               const double *GHOST_RESTRICT tr_q,
               const double *GHOST_RESTRICT tz_q,
               double k, const double *GHOST_RESTRICT xi, int xi_per_pair,
               double *GHOST_RESTRICT o_tt, double *GHOST_RESTRICT o_tf,
               double *GHOST_RESTRICT o_ft, double *GHOST_RESTRICT o_ff)
{
    /* Paired sampling shares the outer near-task CPU reservation. */
    for (int i = 0; i < npair; i++) {
        const double *xi_row = xi + (xi_per_pair ? (size_t)i * nxi : (size_t)0);
        double Rz = z_p[i] - z_q[i];
        double trp = tr_p[i], tzp = tz_p[i], trq = tr_q[i], tzq = tz_q[i];
        double n_tq_c = tzp * trq, n_tq_k = trp * tzq;
        double Wt_tq_c = trp * trq, Wt_tq_k = tzp * tzq;
        size_t base = 2 * (size_t)nxi * (size_t)i;
        double *tt = o_tt + base, *tf = o_tf + base;
        double *ft = o_ft + base, *ff = o_ff + base;
        for (int l = 0; l < nxi; l++) {
            double cx = cos(xi_row[l]), sx = sin(xi_row[l]);
            double Rx = rho_p[i] - rho_q[i] * cx;
            double Ry = rho_q[i] * sx;
            double R, p_re, p_im;
            pR_floor(Rx, Ry, Rz, k, NEAR_R_MIN, &R, &p_re, &p_im);
            double WtR = trp * Rx + tzp * Rz;
            double WfR = Ry;
            double nR = -tzp * Rx + trp * Rz;
            double n_tq = -n_tq_c * cx + n_tq_k;
            double n_fq = -tzp * sx;
            double Wt_tq = Wt_tq_c * cx + Wt_tq_k;
            double Wt_fq = trp * sx;
            double Wf_tq = -trq * sx;
            double Wf_fq = cx;
            double f;
            f = -(WtR * n_tq - Wt_tq * nR);
            tt[2 * l] = f * p_re; tt[2 * l + 1] = f * p_im;
            f = -(WtR * n_fq - Wt_fq * nR);
            tf[2 * l] = f * p_re; tf[2 * l + 1] = f * p_im;
            f = -(WfR * n_tq - Wf_tq * nR);
            ft[2 * l] = f * p_re; ft[2 * l + 1] = f * p_im;
            f = -(WfR * n_fq - Wf_fq * nR);
            ff[2 * l] = f * p_re; ff[2 * l + 1] = f * p_im;
        }
    }
}

/*
 * Paired-layout samplers for the grouped (banded) azimuthal quadrature.
 *
 * bor.kernels.banded_modal_kernels groups far point pairs by the FFT size
 * their own radius and meridian distance require and evaluates each group on
 * the half grid xi in [-pi, 0]: the Green's function is even in xi and the
 * bracket components have exact parity, so every pair of a group shares one
 * grid and only half of it is needed.  These entries take that flat pair
 * list, a real or complex wavenumber k = kr + j ki (Im k <= 0 for a passive
 * medium: exp(-j k R) = exp(ki R) (cos(kr R) - j sin(kr R))), the tabulated
 * trigonometry of the shared grid, and run the pair loop on an OpenMP team of
 * `nthreads` (<= 0: the runtime default).  The bracket algebra is the one of
 * near_brackets above; the outputs are interleaved complex doubles
 * [npair, nxi].  R is floored at NEAR_R_MIN like the near samplers; the
 * caller never sends coincident points here.
 */
static inline int team_size(int nthreads)
{
#ifdef _OPENMP
    return nthreads > 0 ? nthreads : omp_get_max_threads();
#else
    (void)nthreads;
    return 1;
#endif
}

void sample_g_pairs(int npair, int nxi,
                    const double *GHOST_RESTRICT rho_p,
                    const double *GHOST_RESTRICT z_p,
                    const double *GHOST_RESTRICT rho_q,
                    const double *GHOST_RESTRICT z_q,
                    double kr, double ki,
                    const double *GHOST_RESTRICT sin2_tab,
                    double *GHOST_RESTRICT out, int nthreads)
{
    const int team = team_size(nthreads);
    #pragma omp parallel for schedule(static) num_threads(team)
    for (int i = 0; i < npair; i++) {
        double dr = rho_p[i] - rho_q[i];
        double dz = z_p[i] - z_q[i];
        double d2 = dr * dr + dz * dz;
        double rr4 = 4.0 * rho_p[i] * rho_q[i];
        double *o = out + 2 * (size_t)nxi * (size_t)i;
        for (int l = 0; l < nxi; l++) {
            double R = sqrt(d2 + rr4 * sin2_tab[l]);
            if (R < NEAR_R_MIN) R = NEAR_R_MIN;
            double a = GHOST_DECAY(ki, ki * R) / (4.0 * M_PI * R);
            double kR = kr * R;
            o[2 * l] = a * cos(kR);
            o[2 * l + 1] = -a * sin(kR);
        }
    }
}

void sample_brackets_pairs(int family, int npair, int nxi,
                           const double *GHOST_RESTRICT rho_p,
                           const double *GHOST_RESTRICT z_p,
                           const double *GHOST_RESTRICT tr_p,
                           const double *GHOST_RESTRICT tz_p,
                           const double *GHOST_RESTRICT rho_q,
                           const double *GHOST_RESTRICT z_q,
                           const double *GHOST_RESTRICT tr_q,
                           const double *GHOST_RESTRICT tz_q,
                           double kr, double ki,
                           const double *GHOST_RESTRICT cx_tab,
                           const double *GHOST_RESTRICT sx_tab,
                           double *GHOST_RESTRICT o_tt, double *GHOST_RESTRICT o_tf,
                           double *GHOST_RESTRICT o_ft, double *GHOST_RESTRICT o_ff,
                           int nthreads)
{
    const int team = team_size(nthreads);
    #pragma omp parallel for schedule(static) num_threads(team)
    for (int i = 0; i < npair; i++) {
        double rp = rho_p[i], rq = rho_q[i], rz = z_p[i] - z_q[i];
        double tp = tr_p[i], zp = tz_p[i], tq = tr_q[i], zq = tz_q[i];
        size_t base = 2 * (size_t)nxi * (size_t)i;
        double *tt = o_tt + base, *tf = o_tf + base;
        double *ft = o_ft + base, *ff = o_ff + base;
        for (int l = 0; l < nxi; l++) {
            double cx = cx_tab[l], sx = sx_tab[l];
            double rx = rp - rq * cx, ry = rq * sx;
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r < NEAR_R_MIN) r = NEAR_R_MIN;
            double a = kr * r, b = ki * r, c = cos(a), s = sin(a);
            double scale = GHOST_DECAY(ki, b) / (4.0 * M_PI * r * r * r);
            double pre = ((1.0 - b) * c + a * s) * scale;
            double pim = (a * c - (1.0 - b) * s) * scale;
            double wt_tq = tp * tq * cx + zp * zq, wt_fq = tp * sx;
            double wf_tq = -tq * sx, wf_fq = cx;
            double vtt, vtf, vft, vff;
            if (family == 0) {
                double wtr = tp * rx + zp * rz, nr = -zp * rx + tp * rz;
                double ntq = -zp * tq * cx + tp * zq, nfq = -zp * sx;
                vtt = -(wtr * ntq - wt_tq * nr);
                vtf = -(wtr * nfq - wt_fq * nr);
                vft = -(ry * ntq - wf_tq * nr);
                vff = -(ry * nfq - wf_fq * nr);
            } else {
                double wtnq = -tp * zq * cx + zp * tq, wfnq = zq * sx;
                double d = rp * cx - rq;
                double rtq = tq * d + zq * rz, rfq = rp * sx, rnq = -zq * d + tq * rz;
                vtt = wtnq * rtq - wt_tq * rnq;
                vtf = wtnq * rfq - wt_fq * rnq;
                vft = wfnq * rtq - wf_tq * rnq;
                vff = wfnq * rfq - wf_fq * rnq;
            }
            tt[2 * l] = vtt * pre; tt[2 * l + 1] = vtt * pim;
            tf[2 * l] = vtf * pre; tf[2 * l + 1] = vtf * pim;
            ft[2 * l] = vft * pre; ft[2 * l + 1] = vft * pim;
            ff[2 * l] = vff * pre; ff[2 * l + 1] = vff * pim;
        }
    }
}

/*
 * Angular moments of weighted samples on per-pair grids, for the near rules.
 *
 * For every pair n, row r and order m = 0..count-1:
 *   out_cos[n, r, m] = sum_a X[n, r, a] cos(m xi[n, a])
 *   out_sin[n, r, m] = sum_a X[n, r, a] sin(m xi[n, a])   (when out_sin != NULL)
 * X is [n, rows, na] real (a complex weight is two rows), xi is [n, na].  The
 * trigonometric factors follow the rotation recurrence
 *   (c, s)_{m+1} = (c cos xi - s sin xi, s cos xi + c sin xi),
 * one multiply-add per sample and order instead of a cosine evaluation and
 * an [n, na, count] table; its rounding grows linearly with the order (a few
 * ulps at m = 100), far below the rules' 2e-8 refinement tolerance.  Pairs run
 * on an OpenMP team of `nthreads` (<= 0: the runtime default).
 *
 * Loop order: orders outermost, samples innermost.  The recurrence state of
 * one pair lives in four scratch rows (c, s, cos xi, sin xi) and every order
 * is one pass over the samples that forms the dot products of all rows and
 * advances the rotation, so the pass vectorizes across samples (two lanes)
 * with independent partial sums per row.  No relaxed floating point: every
 * moment of every entry point is
 *   (sum over even sample indices) + (sum over odd sample indices),
 * both accumulated in increasing sample order, so trig_moments,
 * parity_moments and the fused near kernels below agree bitwise.
 */

/* Rotation of the recurrence state by one step, samples [a0, na). */
static void rotate_state(int a0, int na, double *GHOST_RESTRICT c, double *GHOST_RESTRICT s,
                         const double *GHOST_RESTRICT c1, const double *GHOST_RESTRICT s1)
{
    for (int a = a0; a < na; a++) {
        const double ca = c[a], sa = s[a];
        c[a] = ca * c1[a] - sa * s1[a];
        s[a] = sa * c1[a] + ca * s1[a];
    }
}

/* Dot product with the two-lane partial sums of the convention above. */
static double dot_pair(int na, const double *GHOST_RESTRICT x, const double *GHOST_RESTRICT v)
{
    double even = 0.0, odd = 0.0;
    int a = 0;
    for (; a + 1 < na; a += 2) {
        even += x[a] * v[a];
        odd += x[a + 1] * v[a + 1];
    }
    if (a < na)
        even += x[a] * v[a];
    return even + odd;
}

/* One order of the parity projection: cosine moments of four rows e0..e3 and
 * sine moments of four rows o0..o3, then one rotation step, in one pass. */
static void pass_parity44(int na,
                          const double *GHOST_RESTRICT e0, const double *GHOST_RESTRICT e1,
                          const double *GHOST_RESTRICT e2, const double *GHOST_RESTRICT e3,
                          const double *GHOST_RESTRICT o0, const double *GHOST_RESTRICT o1,
                          const double *GHOST_RESTRICT o2, const double *GHOST_RESTRICT o3,
                          double *GHOST_RESTRICT c, double *GHOST_RESTRICT s,
                          const double *GHOST_RESTRICT c1, const double *GHOST_RESTRICT s1,
                          double *oc, double *os)
{
    double A0 = 0.0, B0 = 0.0, A1 = 0.0, B1 = 0.0, A2 = 0.0, B2 = 0.0, A3 = 0.0, B3 = 0.0;
    double P0 = 0.0, Q0 = 0.0, P1 = 0.0, Q1 = 0.0, P2 = 0.0, Q2 = 0.0, P3 = 0.0, Q3 = 0.0;
    int a = 0;
    for (; a + 1 < na; a += 2) {
        const double ca = c[a], cb = c[a + 1], sa = s[a], sb = s[a + 1];
        A0 += e0[a] * ca; B0 += e0[a + 1] * cb;
        A1 += e1[a] * ca; B1 += e1[a + 1] * cb;
        A2 += e2[a] * ca; B2 += e2[a + 1] * cb;
        A3 += e3[a] * ca; B3 += e3[a + 1] * cb;
        P0 += o0[a] * sa; Q0 += o0[a + 1] * sb;
        P1 += o1[a] * sa; Q1 += o1[a + 1] * sb;
        P2 += o2[a] * sa; Q2 += o2[a + 1] * sb;
        P3 += o3[a] * sa; Q3 += o3[a + 1] * sb;
        const double ka = c1[a], kb = c1[a + 1], ta = s1[a], tb = s1[a + 1];
        c[a] = ca * ka - sa * ta; c[a + 1] = cb * kb - sb * tb;
        s[a] = sa * ka + ca * ta; s[a + 1] = sb * kb + cb * tb;
    }
    if (a < na) {
        const double ca = c[a], sa = s[a];
        A0 += e0[a] * ca; A1 += e1[a] * ca; A2 += e2[a] * ca; A3 += e3[a] * ca;
        P0 += o0[a] * sa; P1 += o1[a] * sa; P2 += o2[a] * sa; P3 += o3[a] * sa;
        c[a] = ca * c1[a] - sa * s1[a];
        s[a] = sa * c1[a] + ca * s1[a];
    }
    oc[0] = A0 + B0; oc[1] = A1 + B1; oc[2] = A2 + B2; oc[3] = A3 + B3;
    os[0] = P0 + Q0; os[1] = P1 + Q1; os[2] = P2 + Q2; os[3] = P3 + Q3;
}

/* Two consecutive orders of the cosine moments of two rows (the real and
 * imaginary parts of the Green's rule), advancing the state by two steps:
 * four independent accumulator chains instead of two.  The intermediate state
 * is formed exactly as rotate_state would store it. */
static void pass_cos2_twice(int na, const double *GHOST_RESTRICT x0, const double *GHOST_RESTRICT x1,
                            double *GHOST_RESTRICT c, double *GHOST_RESTRICT s,
                            const double *GHOST_RESTRICT c1, const double *GHOST_RESTRICT s1,
                            double out0[2], double out1[2])
{
    double A0 = 0.0, B0 = 0.0, A1 = 0.0, B1 = 0.0;   /* order m   */
    double C0 = 0.0, D0 = 0.0, C1 = 0.0, D1 = 0.0;   /* order m+1 */
    int a = 0;
    for (; a + 1 < na; a += 2) {
        const double ca = c[a], cb = c[a + 1], sa = s[a], sb = s[a + 1];
        const double ka = c1[a], kb = c1[a + 1], ta = s1[a], tb = s1[a + 1];
        const double na_c = ca * ka - sa * ta, nb_c = cb * kb - sb * tb;
        const double na_s = sa * ka + ca * ta, nb_s = sb * kb + cb * tb;
        const double xa = x0[a], xb = x0[a + 1], ya = x1[a], yb = x1[a + 1];
        A0 += xa * ca; B0 += xb * cb;
        A1 += ya * ca; B1 += yb * cb;
        C0 += xa * na_c; D0 += xb * nb_c;
        C1 += ya * na_c; D1 += yb * nb_c;
        c[a] = na_c * ka - na_s * ta; c[a + 1] = nb_c * kb - nb_s * tb;
        s[a] = na_s * ka + na_c * ta; s[a + 1] = nb_s * kb + nb_c * tb;
    }
    if (a < na) {
        const double ca = c[a], sa = s[a];
        const double nc = ca * c1[a] - sa * s1[a], ns = sa * c1[a] + ca * s1[a];
        A0 += x0[a] * ca; A1 += x1[a] * ca;
        C0 += x0[a] * nc; C1 += x1[a] * nc;
        c[a] = nc * c1[a] - ns * s1[a];
        s[a] = ns * c1[a] + nc * s1[a];
    }
    out0[0] = A0 + B0; out1[0] = A1 + B1;
    out0[1] = C0 + D0; out1[1] = C1 + D1;
}

/* Moments of one pair from its recurrence state (initialized by the caller):
 * rows_c cosine rows xc[r] -> oc[r * count + m], rows_s sine rows xs[r] ->
 * os[r * count + m].  Dispatches the specialized passes; any other row count
 * takes per-row passes with the same arithmetic. */
static void pair_moments(int na, int count,
                         int rows_c, const double *const *xc, double *oc,
                         int rows_s, const double *const *xs, double *os,
                         double *c, double *s, const double *c1, const double *s1)
{
    if (rows_c == 4 && rows_s == 4) {
        for (int m = 0; m < count; m++) {
            double vc[4], vs[4];
            pass_parity44(na, xc[0], xc[1], xc[2], xc[3], xs[0], xs[1], xs[2], xs[3],
                          c, s, c1, s1, vc, vs);
            for (int r = 0; r < 4; r++) {
                oc[(size_t)r * count + m] = vc[r];
                os[(size_t)r * count + m] = vs[r];
            }
        }
        return;
    }
    if (rows_c == 2 && rows_s == 0) {
        int m = 0;
        for (; m + 1 < count; m += 2) {
            double v0[2], v1[2];
            pass_cos2_twice(na, xc[0], xc[1], c, s, c1, s1, v0, v1);
            oc[m] = v0[0]; oc[m + 1] = v0[1];
            oc[(size_t)count + m] = v1[0]; oc[(size_t)count + m + 1] = v1[1];
        }
        if (m < count) {
            oc[m] = dot_pair(na, xc[0], c);
            oc[(size_t)count + m] = dot_pair(na, xc[1], c);
        }
        return;
    }
    for (int m = 0; m < count; m++) {
        for (int r = 0; r < rows_c; r++)
            oc[(size_t)r * count + m] = dot_pair(na, xc[r], c);
        for (int r = 0; r < rows_s; r++)
            os[(size_t)r * count + m] = dot_pair(na, xs[r], s);
        if (m + 1 < count)
            rotate_state(0, na, c, s, c1, s1);
    }
}

/* Recurrence state of one pair: c = 1, s = 0, (c1, s1) = (cos, sin) of xi. */
static void init_state(int na, const double *grid, double *c, double *s, double *c1, double *s1)
{
    for (int a = 0; a < na; a++) {
        c1[a] = cos(grid[a]);
        s1[a] = sin(grid[a]);
        c[a] = 1.0;
        s[a] = 0.0;
    }
}

/* Shared driver of trig_moments and parity_moments. */
static void angular_moments(int n, int rows_c, int rows_s, int na, int count,
                            const double *Xc, const double *Xs, const double *xi,
                            double *out_cos, double *out_sin, int nthreads)
{
    const int team = team_size(nthreads);
    if (n <= 0 || count <= 0)
        return;
    #pragma omp parallel num_threads(team)
    {
        const size_t stride = (size_t)(na > 0 ? na : 1) + 4;
        double *scratch = (double *)malloc(sizeof(double) * 4 * stride);
        const double **rows = (const double **)malloc(sizeof(double *) * (size_t)(rows_c + rows_s + 1));
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            double *oc = out_cos ? out_cos + (size_t)i * rows_c * count : (double *)0;
            double *os = out_sin ? out_sin + (size_t)i * rows_s * count : (double *)0;
            if (scratch == (double *)0 || rows == (const double **)0) {
                /* No memory for the recurrence state: NaN, never stale data. */
                for (int j = 0; j < rows_c * count; j++) oc[j] = NAN;
                for (int j = 0; j < rows_s * count; j++) os[j] = NAN;
                continue;
            }
            double *c = scratch, *s = scratch + stride;
            double *c1 = scratch + 2 * stride, *s1 = scratch + 3 * stride;
            init_state(na, xi + (size_t)i * na, c, s, c1, s1);
            for (int r = 0; r < rows_c; r++)
                rows[r] = Xc + ((size_t)i * rows_c + r) * na;
            for (int r = 0; r < rows_s; r++)
                rows[rows_c + r] = Xs + ((size_t)i * rows_s + r) * na;
            pair_moments(na, count, rows_c, rows, oc, rows_s, rows + rows_c, os, c, s, c1, s1);
        }
        free((void *)rows);
        free(scratch);
    }
}

void trig_moments(int n, int rows, int na, int count,
                  const double *GHOST_RESTRICT X, const double *GHOST_RESTRICT xi,
                  double *GHOST_RESTRICT out_cos, double *out_sin, int nthreads)
{
    angular_moments(n, rows, out_sin ? rows : 0, na, count, X, X, xi,
                    out_cos, out_sin, nthreads);
}

/*
 * Fused parity projection of the near bracket rules.  The tangential MFIE/IBC
 * brackets have exact angular parity (tt/ff even, tf/ft odd), so the even rows
 * need only cosine moments and the odd rows only sine moments:
 *   out_cos[n, r, m] = sum_a XE[n, r, a] cos(m xi[n, a])   r < rows_even
 *   out_sin[n, r, m] = sum_a XO[n, r, a] sin(m xi[n, a])   r < rows_odd
 * One recurrence and one pass per order serve both (trig_moments would need
 * two calls, one of them computing discarded cosines).  The values equal
 * those of trig_moments(XE) cosines and trig_moments(XO) sines bitwise.
 */
void parity_moments(int n, int rows_even, int rows_odd, int na, int count,
                    const double *GHOST_RESTRICT XE, const double *GHOST_RESTRICT XO,
                    const double *GHOST_RESTRICT xi,
                    double *GHOST_RESTRICT out_cos, double *GHOST_RESTRICT out_sin,
                    int nthreads)
{
    angular_moments(n, rows_even, rows_odd, na, count, XE, XO, xi,
                    out_cos, out_sin, nthreads);
}

/*
 * Per-sample near-rule integrands, shared by the samplers and the fused
 * sample-and-project kernels below so both produce identical values.
 */
#if defined(_MSC_VER)
#define GHOST_INLINE static __forceinline
#else
#define GHOST_INLINE static inline __attribute__((always_inline))
#endif

/* g = exp(-j k R) / (4 pi R), R = sqrt(d2 + rr4 sin^2(x/2)).  Operation order
 * of the NumPy reference kernels._green_samples: exp(-jkR) first, then the
 * division by (4 pi) R. */
GHOST_INLINE void green_value_half(double d2, double rr4, double h, double kr, double ki,
                                   double *re, double *im)
{
    double R = sqrt(d2 + rr4 * (h * h));
    if (R < NEAR_R_MIN) R = NEAR_R_MIN;
    const double e = GHOST_DECAY(ki, ki * R), kR = kr * R, den = (4.0 * M_PI) * R;
    *re = (e * cos(kR)) / den;
    *im = (e * -sin(kR)) / den;
}

GHOST_INLINE void green_value(double d2, double rr4, double x, double kr, double ki,
                              double *re, double *im)
{
    green_value_half(d2, rr4, sin(0.5 * x), kr, ki, re, im);
}

/* The sampled MFIE/IBC brackets of near_brackets (same expressions, same
 * order), at cx = cos xi, sx = sin xi; v = tt, tf, ft, ff as (re, im). */
GHOST_INLINE void bracket_sampled(int family, double rp, double rq, double rz,
                                  double tp, double zp, double tq, double zq,
                                  double kr, double ki, double cx, double sx, double v[8])
{
    double rx = rp - rq * cx, ry = rq * sx;
    double r = fmax(sqrt(rx*rx + ry*ry + rz*rz), NEAR_R_MIN);
    double a = kr*r, b = ki*r, c = cos(a), s = sin(a);
    double scale = GHOST_DECAY(ki, b) / (4.0*M_PI*r*r*r);
    double pre = ((1.0-b)*c + a*s)*scale;
    double pim = (a*c - (1.0-b)*s)*scale;
    double wt_tq = tp*tq*cx + zp*zq, wt_fq = tp*sx;
    double wf_tq = -tq*sx, wf_fq = cx;
    double tt, tf, ft, ff;
    if (family == 0) {
        double wtr = tp*rx + zp*rz, nr = -zp*rx + tp*rz;
        double ntq = -zp*tq*cx + tp*zq, nfq = -zp*sx;
        tt = -(wtr*ntq - wt_tq*nr);
        tf = -(wtr*nfq - wt_fq*nr);
        ft = -(ry*ntq - wf_tq*nr);
        ff = -(ry*nfq - wf_fq*nr);
    } else {
        double wtnq = -tp*zq*cx + zp*tq, wfnq = zq*sx;
        double d = rp*cx - rq;
        double rtq = tq*d + zq*rz, rfq = rp*sx, rnq = -zq*d + tq*rz;
        tt = wtnq*rtq - wt_tq*rnq;
        tf = wtnq*rfq - wt_fq*rnq;
        ft = wfnq*rtq - wf_tq*rnq;
        ff = wfnq*rfq - wf_fq*rnq;
    }
    v[0] = tt*pre; v[1] = tt*pim;
    v[2] = tf*pre; v[3] = tf*pim;
    v[4] = ft*pre; v[5] = ft*pim;
    v[6] = ff*pre; v[7] = ff*pim;
}

/* Pair constants of the cancellation-free forms. */
typedef struct {
    double rp, rq, trp, tzp, trq, tzq, d_z, d2, X, D, A, N;
} stable_pair;

GHOST_INLINE stable_pair stable_setup(int family, double rp, double zp, double trp, double tzp,
                                      double rq, double zq, double trq, double tzq)
{
    stable_pair q;
    const double d_rho = rp - rq;
    q.rp = rp; q.rq = rq; q.trp = trp; q.tzp = tzp; q.trq = trq; q.tzq = tzq;
    q.d_z = zp - zq;
    q.d2 = d_rho * d_rho + q.d_z * q.d_z;
    q.X = trp * tzq - tzp * trq;
    q.D = trp * trq + tzp * tzq;
    if (family == 0) {
        q.A = trp * d_rho + tzp * q.d_z;
        q.N = -tzp * d_rho + trp * q.d_z;
    } else {
        q.A = trq * d_rho + tzq * q.d_z;
        q.N = -tzq * d_rho + trq * q.d_z;
    }
    return q;
}

/* kernels._stable_brackets at one angle: half = sin(xi/2), sx = sin xi. */
GHOST_INLINE void bracket_stable(int family, const stable_pair *q, double kr, double ki,
                                 double half, double sx, double v[8])
{
    const double rp = q->rp, rq = q->rq, trp = q->trp, tzp = q->tzp, trq = q->trq, tzq = q->tzq;
    const double X = q->X, D = q->D, A = q->A, N = q->N, d_z = q->d_z;
    const double h = 2.0 * (half * half);
    double R = sqrt(q->d2 + 2.0 * rp * rq * h);
    if (R < NEAR_R_MIN) R = NEAR_R_MIN;
    /* p = (1 + j k R) exp(-j k R) / (4 pi R^3) */
    const double a = kr * R, b = ki * R, c = cos(a), s = sin(a);
    const double scale = GHOST_DECAY(ki, b) / ((4.0 * M_PI) * R * R * R);
    const double pre = ((1.0 - b) * c + a * s) * scale;
    const double pim = (a * c - (1.0 - b) * s) * scale;
    double tt, tf, ft, ff;
    if (family == 0) {
        /* _stable_brackets returns -p * (bracket) for the MFIE family */
        tt = -((A * X - D * N) + h * (A * tzp * trq + trp * rq * X + D * tzp * rq + trp * trq * N));
        tf = -(-sx * d_z * (trp * trp + tzp * tzp));
        ft = -(sx * (rq * X + trq * N));
        ff = -(-tzp * rq * h - N * (1.0 - h));
    } else {
        tt = -(X * A + D * N) + h * (X * trq * rp + trp * tzq * A - D * tzq * rp + trp * trq * N);
        tf = -sx * (X * rp + trp * N);
        ft = sx * d_z * (trq * trq + tzq * tzq);
        ff = tzq * rp * h - N * (1.0 - h);
    }
    v[0] = tt * pre; v[1] = tt * pim;
    v[2] = tf * pre; v[3] = tf * pim;
    v[4] = ft * pre; v[5] = ft * pim;
    v[6] = ff * pre; v[7] = ff * pim;
}

/*
 * Near-rule Green's function samples on per-pair (xi_per_pair = 1) or shared
 * (0) grids: g = exp(-j k R)/(4 pi R), R = sqrt(d^2 + 4 rho_p rho_q sin^2(xi/2)),
 * k = kr + j ki, output interleaved complex [npair, nxi].  Agrees with the
 * NumPy reference kernels._green_samples to rounding.  Serial: near
 * preparation owns its concurrency.
 */
void near_green(int npair, int nxi,
                const double *GHOST_RESTRICT rho_p, const double *GHOST_RESTRICT z_p,
                const double *GHOST_RESTRICT rho_q, const double *GHOST_RESTRICT z_q,
                double kr, double ki, const double *GHOST_RESTRICT xi, int xi_per_pair,
                double *GHOST_RESTRICT out)
{
    for (int i = 0; i < npair; i++) {
        const double *grid = xi + (xi_per_pair ? (size_t)i * nxi : (size_t)0);
        const double dr = rho_p[i] - rho_q[i], dz = z_p[i] - z_q[i];
        const double d2 = dr * dr + dz * dz;
        const double rr4 = 4.0 * rho_p[i] * rho_q[i];
        double *o = out + 2 * (size_t)nxi * (size_t)i;
        for (int l = 0; l < nxi; l++)
            green_value(d2, rr4, grid[l], kr, ki, o + 2 * l, o + 2 * l + 1);
    }
}

/*
 * Cancellation-free MFIE (family 0) / IBC (family 1) brackets: the closed forms
 * of kernels._stable_brackets.  With h = 1 - cos xi = 2 sin^2(xi/2),
 *   R^2 = d^2 + 2 rho_p rho_q h,  X = t_p x t_q,  D = t_p . t_q,
 *   A, N = tangential/normal projections of (d_rho, d_z) (source frame for IBC),
 * every bracket is a sum of products whose O(h^2) parts cancel analytically, so
 * collinear pairs a distance d apart keep full relative accuracy where the
 * sampled forms lose (rho/d)^2 ulps.  Same layout and arguments as
 * near_brackets; serial.
 */
void near_brackets_stable(int family, int npair, int nxi,
                          const double *rho_p, const double *z_p,
                          const double *tr_p, const double *tz_p,
                          const double *rho_q, const double *z_q,
                          const double *tr_q, const double *tz_q,
                          double kr, double ki, const double *xi, int xi_per_pair,
                          double *o_tt, double *o_tf, double *o_ft, double *o_ff)
{
    for (int i = 0; i < npair; i++) {
        const double *grid = xi + (xi_per_pair ? (size_t)i * nxi : (size_t)0);
        const stable_pair q = stable_setup(family, rho_p[i], z_p[i], tr_p[i], tz_p[i],
                                           rho_q[i], z_q[i], tr_q[i], tz_q[i]);
        for (int l = 0; l < nxi; l++) {
            double v[8];
            bracket_stable(family, &q, kr, ki, sin(0.5 * grid[l]), sin(grid[l]), v);
            const size_t j = 2 * ((size_t)i * nxi + l);
            o_tt[j] = v[0]; o_tt[j + 1] = v[1];
            o_tf[j] = v[2]; o_tf[j + 1] = v[3];
            o_ft[j] = v[4]; o_ft[j + 1] = v[5];
            o_ff[j] = v[6]; o_ff[j + 1] = v[7];
        }
    }
}

/*
 * Fused near rules: build the angular nodes, sample, weight and project in
 * one call per group of pairs, with no intermediate arrays (ctypes releases
 * the GIL for the whole call).  The graded rule of bor.kernels
 * (_graded_near_kernels) gives every pair of a group
 *   - a sinh core on s = sin(xi/2) in [0, s_core]: with delta = d/a,
 *       vmax = asinh(s_core/delta), v = u vmax, s = delta sinh(v),
 *       xi = 2 asin(s), w = (w_u vmax)(2 delta cosh v)/sqrt(1 - s^2)
 *     for the Gauss rule (u_core, w_core) on [0, 1] (n_core = 0: no core);
 *   - the group's shared nodes/weights (geometric panels and tail, or the
 *     on-axis panel on [0, pi]).
 * The node formulas are those of bor.kernels._near_nodes, evaluated in the
 * same order.
 *
 * near_green_rule    out [npair, 2, count]: cosine moments of Re(w g), Im(w g)
 *                    (kernels._cosine_moments before its factor 2).
 * near_brackets_rule out_cos [npair, 4, count]: cosine moments of
 *                    2w (tt.re, ff.re, tt.im, ff.im); out_sin: sine moments of
 *                    2w (tf.re, ft.re, tf.im, ft.im) -- the rows of
 *                    kernels._project_parity_brackets.  stable != 0 selects the
 *                    cancellation-free forms (near_brackets_stable), else the
 *                    sampled ones (near_brackets).
 * Both equal the corresponding sampler (near_green / near_brackets[_stable])
 * followed by trig_moments / parity_moments on the same nodes, bitwise.
 * Pairs run on an OpenMP team of `nthreads` (<= 0: the runtime default); near
 * preparation passes 1 because it owns its concurrency.
 */
static int rule_nodes(double delta, double s_core, int n_core,
                      const double *u_core, const double *w_core,
                      int n_shared, const double *xi_shared, const double *w_shared,
                      double *xi, double *w)
{
    int na = 0;
    if (n_core > 0) {
        const double vmax = asinh(s_core / delta);
        for (int j = 0; j < n_core; j++) {
            const double v = u_core[j] * vmax;
            const double s = delta * sinh(v);
            xi[na] = 2.0 * asin(s);
            w[na] = (w_core[j] * vmax) * (2.0 * delta * cosh(v)) / sqrt(1.0 - s * s);
            na++;
        }
    }
    for (int j = 0; j < n_shared; j++) {
        xi[na] = xi_shared[j];
        w[na] = w_shared[j];
        na++;
    }
    return na;
}

/* Reuse libm values at shared nodes across every point pair. NULL preserves
 * the scalar evaluation if this optional, O(n_shared) allocation fails. */
static double *shared_rule_trig(int n_shared, const double *xi)
{
    if (n_shared <= 0) return NULL;
    double *values = (double *)malloc(3 * sizeof(double) * (size_t)n_shared);
    if (values == NULL) return NULL;
    for (int j = 0; j < n_shared; j++) {
        values[j] = cos(xi[j]);
        values[n_shared+j] = sin(xi[j]);
        values[2*(size_t)n_shared+j] = sin(0.5 * xi[j]);
    }
    return values;
}

void near_green_rule(int npair,
                     const double *GHOST_RESTRICT rho_p, const double *GHOST_RESTRICT z_p,
                     const double *GHOST_RESTRICT rho_q, const double *GHOST_RESTRICT z_q,
                     const double *GHOST_RESTRICT delta, double kr, double ki,
                     double s_core, int n_core, const double *u_core, const double *w_core,
                     int n_shared, const double *xi_shared, const double *w_shared,
                     int count, double *GHOST_RESTRICT out, int nthreads)
{
    const int team = team_size(nthreads);
    const int na_max = (n_core > 0 ? n_core : 0) + (n_shared > 0 ? n_shared : 0);
    if (npair <= 0 || count <= 0)
        return;
    double *shared = shared_rule_trig(n_shared, xi_shared);
    #pragma omp parallel num_threads(team)
    {
        const size_t stride = (size_t)(na_max > 0 ? na_max : 1) + 4;
        double *scratch = (double *)malloc(sizeof(double) * 8 * stride);
        #pragma omp for schedule(static)
        for (int i = 0; i < npair; i++) {
            double *o = out + (size_t)i * 2 * count;
            if (scratch == (double *)0) {
                for (int j = 0; j < 2 * count; j++) o[j] = NAN;
                continue;
            }
            double *xi = scratch, *w = scratch + stride;
            double *x0 = scratch + 2 * stride, *x1 = scratch + 3 * stride;
            double *c = scratch + 4 * stride, *s = scratch + 5 * stride;
            double *c1 = scratch + 6 * stride, *s1 = scratch + 7 * stride;
            const int na = rule_nodes(delta[i], s_core, n_core, u_core, w_core,
                                      n_shared, xi_shared, w_shared, xi, w);
            const double dr = rho_p[i] - rho_q[i], dz = z_p[i] - z_q[i];
            const double d2 = dr * dr + dz * dz;
            const double rr4 = 4.0 * rho_p[i] * rho_q[i];
            for (int a = 0; a < na; a++) {
                double re, im;
                const int j = a - (n_core > 0 ? n_core : 0);
                const double half = shared != NULL && j >= 0
                    ? shared[2*(size_t)n_shared+j] : sin(0.5 * xi[a]);
                green_value_half(d2, rr4, half, kr, ki, &re, &im);
                x0[a] = re * w[a];
                x1[a] = im * w[a];
                c1[a] = shared != NULL && j >= 0 ? shared[j] : cos(xi[a]);
                s1[a] = shared != NULL && j >= 0 ? shared[n_shared+j] : sin(xi[a]);
                c[a] = 1.0;
                s[a] = 0.0;
            }
            const double *rows[2];
            rows[0] = x0;
            rows[1] = x1;
            pair_moments(na, count, 2, rows, o, 0, (const double *const *)0, (double *)0,
                         c, s, c1, s1);
        }
        free(scratch);
    }
    free(shared);
}

void near_brackets_rule(int family, int stable, int npair,
                        const double *rho_p, const double *z_p,
                        const double *tr_p, const double *tz_p,
                        const double *rho_q, const double *z_q,
                        const double *tr_q, const double *tz_q,
                        const double *delta, double kr, double ki,
                        double s_core, int n_core, const double *u_core, const double *w_core,
                        int n_shared, const double *xi_shared, const double *w_shared,
                        int count, double *out_cos, double *out_sin, int nthreads)
{
    const int team = team_size(nthreads);
    const int na_max = (n_core > 0 ? n_core : 0) + (n_shared > 0 ? n_shared : 0);
    if (npair <= 0 || count <= 0)
        return;
    double *shared = shared_rule_trig(n_shared, xi_shared);
    #pragma omp parallel num_threads(team)
    {
        const size_t stride = (size_t)(na_max > 0 ? na_max : 1) + 4;
        double *scratch = (double *)malloc(sizeof(double) * 14 * stride);
        #pragma omp for schedule(static)
        for (int i = 0; i < npair; i++) {
            double *oc = out_cos + (size_t)i * 4 * count;
            double *os = out_sin + (size_t)i * 4 * count;
            if (scratch == (double *)0) {
                for (int j = 0; j < 4 * count; j++) oc[j] = os[j] = NAN;
                continue;
            }
            double *xi = scratch, *w = scratch + stride;
            double *e[4], *o[4];
            for (int r = 0; r < 4; r++) {
                e[r] = scratch + (size_t)(2 + r) * stride;
                o[r] = scratch + (size_t)(6 + r) * stride;
            }
            double *c = scratch + 10 * stride, *s = scratch + 11 * stride;
            double *c1 = scratch + 12 * stride, *s1 = scratch + 13 * stride;
            const int na = rule_nodes(delta[i], s_core, n_core, u_core, w_core,
                                      n_shared, xi_shared, w_shared, xi, w);
            const double rp = rho_p[i], rq = rho_q[i], rz = z_p[i] - z_q[i];
            const double tp = tr_p[i], zp = tz_p[i], tq = tr_q[i], zq = tz_q[i];
            const stable_pair q = stable_setup(family, rp, z_p[i], tp, zp, rq, z_q[i], tq, zq);
            for (int a = 0; a < na; a++) {
                double v[8];
                const int j = a - (n_core > 0 ? n_core : 0);
                c1[a] = shared != NULL && j >= 0 ? shared[j] : cos(xi[a]);
                s1[a] = shared != NULL && j >= 0 ? shared[n_shared+j] : sin(xi[a]);
                c[a] = 1.0;
                s[a] = 0.0;
                if (stable) {
                    const double half = shared != NULL && j >= 0
                        ? shared[2*(size_t)n_shared+j] : sin(0.5 * xi[a]);
                    bracket_stable(family, &q, kr, ki, half, s1[a], v);
                } else
                    bracket_sampled(family, rp, rq, rz, tp, zp, tq, zq, kr, ki, c1[a], s1[a], v);
                const double ww = w[a];
                e[0][a] = (2.0 * v[0]) * ww;   /* tt.re */
                e[1][a] = (2.0 * v[6]) * ww;   /* ff.re */
                e[2][a] = (2.0 * v[1]) * ww;   /* tt.im */
                e[3][a] = (2.0 * v[7]) * ww;   /* ff.im */
                o[0][a] = (2.0 * v[2]) * ww;   /* tf.re */
                o[1][a] = (2.0 * v[4]) * ww;   /* ft.re */
                o[2][a] = (2.0 * v[3]) * ww;   /* tf.im */
                o[3][a] = (2.0 * v[5]) * ww;   /* ft.im */
            }
            const double *ev[4], *ov[4];
            for (int r = 0; r < 4; r++) {
                ev[r] = e[r];
                ov[r] = o[r];
            }
            pair_moments(na, count, 4, ev, oc, 4, ov, os, c, s, c1, s1);
        }
        free(scratch);
    }
    free(shared);
}
