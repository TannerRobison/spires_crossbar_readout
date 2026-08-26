#include "math_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <string.h>


// Matrix-vector multiplication: y = A * x
void mat_vec_mult(double *A, double *x, double *y, size_t n)
{
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		    n, n, 1.0, A, n, x, 1, 0.0, y, 1);
}

/*
void mat_vec_mult(double *A, double *x, double *y, size_t n) 
{
    for (size_t i = 0; i < n; i++) {
        y[i] = 0.0;
        for (size_t j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}
*/

// Power Iteration to estimate spectral radius.
// Only converges when the dominant eigenvalue is real; kept for reference.
double calc_spectral_radius_power_iteration(double *A, size_t n)
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
/*
double calc_spectral_radius(double *A, size_t n) 
{

    const unsigned int MAX_ITER = 1000;
    const double TOLERANCE = 1e-6;
 
    double* x = malloc(n * sizeof(double));
    double* y = malloc(n * sizeof(double));

    // Initialize x with 1s
    for (size_t i = 0; i < n; i++) x[i] = 1.0;

    double lambda_old = 0.0, lambda_new = 0.0;
    
    for (unsigned int iter = 0; iter < MAX_ITER; iter++) {
        mat_vec_mult(A, x, y, n);  // y = A * x

        // Compute largest magnitude value in y (Rayleigh Quotient estimate)
        lambda_new = 0.0;
        for (size_t i = 0; i < n; i++) {
            if (fabs(y[i]) > lambda_new) {
                lambda_new = fabs(y[i]);
                //double max_index = i;
            }
        }

        // Normalize y
        for (size_t i = 0; i < n; i++) x[i] = y[i] / lambda_new;

        // Convergence check
        if (fabs(lambda_new - lambda_old) < TOLERANCE) break;
        lambda_old = lambda_new;
    }

    free(x);
    free(y);
    return lambda_new;
}
*/

// 2-norm via ddot; ~5x faster than dnrm2 here. Falls back if it overflows.
static double vec_norm2(size_t n, const double *x)
{
	double d = cblas_ddot((int)n, x, 1, x, 1);
	if (!isfinite(d))
		return cblas_dnrm2((int)n, x, 1);
	return sqrt(d);
}

/* Spectral radius via restarted Arnoldi. Projects A onto an orthonormal
 * Krylov basis and takes the largest-magnitude eigenvalue of the small
 * Hessenberg matrix, so a complex dominant pair needs no special case --
 * unlike power iteration, which rotates instead of converging when the E/I
 * split is balanced. matvec computes y = A*x; ctx is passed through so the
 * dense and sparse backends can share this. */
#define ARNOLDI_M            32
#define ARNOLDI_MAX_RESTARTS 40
#define ARNOLDI_TOL          1e-8

double spectral_radius_arnoldi(void (*matvec)(void *ctx, const double *x, double *y),
			       void *ctx, size_t n)
{
	if (!matvec || n == 0)
		return 0.0;

	size_t m = ((size_t)ARNOLDI_M < n) ? (size_t)ARNOLDI_M : n;

	double *Q  = malloc((m + 1) * n * sizeof(double));
	double *H  = malloc(m * m * sizeof(double));
	double *Hc = malloc(m * m * sizeof(double));  /* dgeev overwrites input */
	double *VR = malloc(m * m * sizeof(double));
	double *wr = malloc(m * sizeof(double));
	double *wi = malloc(m * sizeof(double));
	double *w  = malloc(n * sizeof(double));

	if (!Q || !H || !Hc || !VR || !wr || !wi || !w) {
		fprintf(stderr, "spectral_radius_arnoldi: allocation failed\n");
		free(Q); free(H); free(Hc); free(VR); free(wr); free(wi); free(w);
		return 0.0;
	}

	/* Non-constant start: a constant vector is an exact eigenvector of any
	 * matrix with constant row sums. Local LCG rather than rand() so the
	 * global stream used for weights is untouched, and fixed-seed so the
	 * calibrated radius is reproducible. */
	unsigned long s = 12345UL;
	for (size_t i = 0; i < n; i++) {
		s = s * 1103515245UL + 12345UL;
		Q[i] = ((double)((s >> 16) & 0x7fffUL) / 32767.0) - 0.5;
	}
	double nrm0 = vec_norm2(n, Q);
	if (nrm0 < 1e-300) {
		free(Q); free(H); free(Hc); free(VR); free(wr); free(wi); free(w);
		return 0.0;
	}
	cblas_dscal((int)n, 1.0 / nrm0, Q, 1);

	double rho = 0.0, prev = 0.0;

	for (unsigned int restart = 0; restart < ARNOLDI_MAX_RESTARTS; restart++) {
		memset(H, 0, m * m * sizeof(double));
		size_t mm = m;

		for (size_t j = 0; j < m; j++) {
			matvec(ctx, &Q[j * n], w);

			/* modified Gram-Schmidt against the existing basis */
			for (size_t i = 0; i <= j; i++) {
				double h = cblas_ddot((int)n, &Q[i * n], 1, w, 1);
				H[i * m + j] = h;
				cblas_daxpy((int)n, -h, &Q[i * n], 1, w, 1);
			}

			double nrm = vec_norm2(n, w);
			if (nrm < 1e-14) {      /* invariant subspace reached */
				mm = j + 1;
				break;
			}
			if (j + 1 < m)
				H[(j + 1) * m + j] = nrm;
			cblas_dcopy((int)n, w, 1, &Q[(j + 1) * n], 1);
			cblas_dscal((int)n, 1.0 / nrm, &Q[(j + 1) * n], 1);
		}

		/* leading mm x mm block, repacked to a contiguous mm stride */
		for (size_t i = 0; i < mm; i++)
			memcpy(&Hc[i * mm], &H[i * m], mm * sizeof(double));

		lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V',
					       (lapack_int)mm, Hc, (lapack_int)mm,
					       wr, wi, NULL, 1, VR, (lapack_int)mm);
		if (info != 0)
			break;

		size_t k = 0;
		double best = -1.0;
		for (size_t i = 0; i < mm; i++) {
			double mag = hypot(wr[i], wi[i]);
			if (mag > best) {
				best = mag;
				k = i;
			}
		}
		rho = best;

		if (restart > 0 &&
		    fabs(rho - prev) <= ARNOLDI_TOL * (rho > 0.0 ? rho : 1.0))
			break;
		prev = rho;

		/* Restart from the dominant Ritz vector lifted back to R^n. For a
		 * complex pair dgeev stores real and imaginary parts in consecutive
		 * columns; the real part alone is a valid real restart direction. */
		memset(w, 0, n * sizeof(double));
		for (size_t i = 0; i < mm; i++)
			cblas_daxpy((int)n, VR[i * mm + k], &Q[i * n], 1, w, 1);

		double nv = vec_norm2(n, w);
		if (nv < 1e-14)
			break;
		cblas_dcopy((int)n, w, 1, Q, 1);
		cblas_dscal((int)n, 1.0 / nv, Q, 1);
	}

	free(Q); free(H); free(Hc); free(VR); free(wr); free(wi); free(w);
	return rho;
}

struct dense_matvec_ctx {
	const double *A;
	size_t n;
};

static void dense_matvec(void *ctx, const double *x, double *y)
{
	const struct dense_matvec_ctx *c = (const struct dense_matvec_ctx *)ctx;
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		    (int)c->n, (int)c->n, 1.0, c->A, (int)c->n, x, 1, 0.0, y, 1);
}

double calc_spectral_radius(double *A, size_t n)
{
	struct dense_matvec_ctx ctx = { A, n };
	return spectral_radius_arnoldi(dense_matvec, &ctx, n);
}

void rescale_matrix(double *A, size_t n, double target_rho)
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
void mat_transpose(double *A, double *A_T, size_t rows, size_t cols) 
{
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

/**
 * @brief Multiplies two matrices: C = A * B.
 * @param A Input matrix of size (r1 x c1).
 * @param B Input matrix of size (c1 x c2).
 * @param C Output matrix of size (r1 x c2).
 */

void mat_mat_mult(double *A, double *B, double *C,
		  size_t r1, size_t c1, size_t c2)
{
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
int solve_linear_system_lud(double *A, double *b, double *x, size_t n)
{
	int info, *ipiv;
	double *A_copy, *b_copy;
	size_t i;

	/* Allocate pivot array */
	ipiv = malloc(n * sizeof(int));
	if (!ipiv) {
		fprintf(stderr, "Error: malloc failed for ipiv\n");
		return -1;
	}

	/* Make copies of A and b because dgesv overwrites inputs */
	A_copy = malloc(n * n * sizeof(double));
	if (!A_copy) {
		fprintf(stderr, "Error: malloc failed for A_copy\n");
		free(ipiv);
		return -1;
	}
	memcpy(A_copy, A, n * n * sizeof(double));

	b_copy = malloc(n * sizeof(double));
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
