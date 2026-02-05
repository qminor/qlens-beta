#include <iostream>
#include <vector>
#include <cmath>
#include "errors.h"
#include "sort.h"

#ifdef USE_MPI
#include "mpi.h"
#endif

using namespace std;

// Abstract base class for solving sparse linear equations by the preconditioned
// biconjugate gradient method. To use, declare a derived class in which the
// functions A_matrix_multiply, A_transpose_multiply, and preconditioner_solve
// are defined for your problem, along with any data that they need.
// Then call the solve function.

class CG_Solver : public Sort
{
	protected:
	int n, iterations;
	int itmax;
	double tol, err;
	double temp;
	bool will_continue;
	int nthreads, default_nthreads;
	int mpi_np, mpi_id;
#ifdef USE_MPI
	MPI_Comm *mpi_comm;
#endif
	bool find_determinant, ratio_mode; // ratio mode allows to find determinant even after solution has converged
	double log_determinant;

	public:
	CG_Solver() { find_determinant = false; log_determinant = -1e30; ratio_mode = false;}
	CG_Solver(const int nn, const double tol_in, const int itmax_in) : n(nn), tol(tol_in), itmax(itmax_in) { find_determinant = false; log_determinant = -1e30; ratio_mode = false; }
	void set_tolerances(const double tol_in, const int itmax_in) {
		tol=tol_in; itmax=itmax_in;
	}

	void solve(double* b, double* x);
	void get_error(int& it, double& error) { it=iterations; error=err; }
	void set_determinant_mode(bool detmode) { find_determinant = detmode; }
	void get_log_determinant(double& logdet) { logdet = log_determinant; }

	protected:
	void error_norm(double* sx, double& err);
	virtual void preconditioner_solve(double* b, double* x) = 0;
	virtual void A_matrix_multiply(const double* const x, double* const r) = 0;
	void set_thread_num(int nt_in);
};

class CG_sparse : public CG_Solver
{
	vector<double> Avec;
	vector<int> Aivec;
	double *A_sparse;
	int *A_index;
	int A_length;
	vector<int> *sorted_indices;
	vector<int> *sorted_indices_i;
	double *preconditioner;
	double *preconditioner_transpose;
	int *preconditioner_transpose_index;

	public:
	CG_sparse(double* As_in, int* Ai_in, const double tol_in, const int itmax_in, const int nt_in, const int mpi_np, const int mpi_id);
	CG_sparse(double** Amatrix, const int nn, const double tol_in, const int itmax_in, const int mpi_np, const int mpi_id);
#ifdef USE_MPI
	void set_MPI_comm(MPI_Comm* mpi_comm_in);
#endif
	~CG_sparse();

	void solve(double* b, double* x);
	double calculate_log_determinant();
	void A_matrix_multiply(const double* const x, double* const r);
	void incomplete_Cholesky_preconditioner();
	void preconditioner_solve(double* b, double* x);
	void Cholesky_preconditioner_solve(double* b, double* x);
	void indexx(int* arr, int* indx, int nn);
};

