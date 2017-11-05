#include "cg.h"
#include "mathexpr.h"
#include "sort.h"
#include "errors.h"
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include "mpi.h"
#endif

using namespace std;

CG_sparse::CG_sparse(double* As_in, int* Ai_in, const double tol_in, const int itmax_in, const int nt_in, const int mpi_np_in, const int mpi_id_in)
{
	mpi_id=mpi_id_in;
	mpi_np=mpi_np_in;
	tol=tol_in; itmax=itmax_in;
	A_sparse = As_in;
	A_index = Ai_in;
	n = A_index[0] - 1;
	A_length = A_index[n];
	preconditioner = NULL;
	preconditioner_transpose = NULL;
	preconditioner_transpose_index = NULL;

	sorted_indices = new vector<int>[n];
	sorted_indices_i = new vector<int>[n];
	int i,j;
	for (i=0; i < n; i++) {
		for (j=A_index[i]; j < A_index[i+1]; j++) {
			sorted_indices[A_index[j]].push_back(j);
			sorted_indices_i[A_index[j]].push_back(i);
		}
	}
	
	set_thread_num(nt_in);
}

CG_sparse::CG_sparse(double** Amatrix, const int nn, const double tol_in, const int itmax_in, const int mpi_np_in, const int mpi_id_in)
{
	mpi_id=mpi_id_in;
	mpi_np=mpi_np_in;
	n=nn; tol=tol_in; itmax=itmax_in;
	int i,j,k;

	Aivec.assign(n+1,0);
	Aivec[0] = n+1;
	for (j=0; j < n; j++) Avec.push_back(Amatrix[j][j]);
	k=n;
	Avec.push_back(0); // dummy element; Avec and Aivec should both now have n+1 elements
	for (i=0; i < n; i++) {
		for (j=i+1; j < n; j++) {
			if (fabs(Amatrix[i][j]) != 0) {
				Avec.push_back(Amatrix[i][j]);
				Aivec.push_back(j);
				k++;
			}
		}
		Aivec[i+1] = k+1;
	}

	A_sparse = Avec.data();
	A_index = Aivec.data();
	A_length = Aivec.size();

	sorted_indices = new vector<int>[n];
	sorted_indices_i = new vector<int>[n];
	for (i=0; i < n; i++) {
		for (j=A_index[i]; j < A_index[i+1]; j++) {
			sorted_indices[A_index[j]].push_back(j);
			sorted_indices_i[A_index[j]].push_back(i);
		}
	}

	preconditioner = NULL;
	preconditioner_transpose = NULL;
	preconditioner_transpose_index = NULL;
	set_thread_num(1);
}

#ifdef USE_MPI
void CG_sparse::set_MPI_comm(MPI_Comm* mpi_comm_in)
{
	mpi_comm = mpi_comm_in;
}
#endif

void CG_Solver::set_thread_num(int nt_in)
{
	#pragma omp parallel
	{
#ifdef USE_OPENMP
		#pragma omp master
		default_nthreads = omp_get_num_threads();
#endif
	}
	nthreads = nt_in;
}

void CG_Solver::solve(double* b, double* x)
{
	double ak,akden,bk,bkden=1.0,bknum,bnrm,dxnrm,xnrm,zm1nrm,znrm=0;
	static const double EPS=1.0e-14;
	int j,k;

	double *p = new double[n];
	double *r = new double[n];
	double *z = new double[n];
	double *alpha = new double[n];
	double *beta = new double[n];

	iterations=0;
	k=0;

#ifdef USE_OPENMP
	omp_set_num_threads(nthreads);
#endif
	
	#pragma omp parallel
	{
		int thread=0;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#endif
		A_matrix_multiply(x,r);
		#pragma omp barrier
		#pragma omp master
		{
			for (j=0;j<n;j++) {
				r[j]=b[j]-r[j];
			}
			preconditioner_solve(b,z);
			error_norm(z,bnrm);
			preconditioner_solve(r,z);
			error_norm(z,znrm);
			for (j=0;j<n;j++) {
				p[j]=0;
			}
		}

		#pragma omp barrier
		while (iterations < itmax)
		{
			#pragma omp barrier
			#pragma omp master
			{
				iterations++;
				bknum=0;
				for (j=0;j<n;j++) {
					bknum += z[j]*r[j];
				}
				bk=bknum/bkden;
				beta[k]=bk;
				for (j=0;j<n;j++) {
					p[j]=bk*p[j]+z[j];
				}
				bkden=bknum;
			}
			#pragma omp barrier
			A_matrix_multiply(p,z);
			#pragma omp barrier
			#pragma omp master
			{
				akden=0;
				for (j=0;j<n;j++) {
					akden += z[j]*p[j];
				}
				ak=bknum/akden;
				alpha[k]=ak;
				for (j=0;j<n;j++) {
					x[j] += ak*p[j];
					r[j] -= ak*z[j];
				}

				zm1nrm=znrm;
				preconditioner_solve(r,z);
				error_norm(z,znrm);
				temp = fabs(zm1nrm-znrm)/znrm;
				if (temp > EPS) {
					error_norm(p,dxnrm);
					dxnrm *= fabs(ak);
					err=znrm/fabs(zm1nrm-znrm)*dxnrm;
					will_continue = false;
				} else {
					err=znrm/bnrm;
					will_continue = true;
				}
			}
			#pragma omp barrier
			if (will_continue) continue;
			#pragma omp master
			{
				error_norm(x,xnrm);
				temp = err;
				if (temp <= 0.5*xnrm) {
					err /= xnrm;
					will_continue = false;
				}
				else {
					err=znrm/bnrm;
					will_continue = true;
				}
				if (++k >= n) k=0; // increase index k; if it has filled all n elements, start over
			}
			#pragma omp barrier
			if (will_continue) continue;
			if (err <= tol) break;
		}
	}
#ifdef USE_OPENMP
	omp_set_num_threads(default_nthreads);
#endif
	delete[] p;
	delete[] r;
	delete[] z;
	delete[] alpha;
	delete[] beta;
}

void CG_sparse::solve(double* b, double* x)
{
	// note, the sparse version uses a diagonal preconditioner specifically
	double ak,akden,bk,bkden=1.0,bknum,bnrm,dxnrm,xnrm,zm1nrm,znrm=0;
	static const double EPS=1.0e-14;
	int j,k;

	double p[n];
	double r[n];
	double z[n];
	double y[n];
	double y2[n];
	double alpha[n];
	double beta[n];
	double akk[n], bkk[n];

	double log_pre_det, log_pre_det_last, log_predet_temp;
	double errnorm;
	log_pre_det_last = 0;
	log_pre_det = 0;

	iterations=0;
	k=0;
	double rho[n], sigma[n], gamma[n]; // used to find determinant after solution has already converged
	double rnrm, old_rnrm=1.0, older_rnrm, signorm, old_signorm=1.0;

#ifdef USE_OPENMP
	omp_set_num_threads(nthreads);
#endif

	#pragma omp parallel
	{
		A_matrix_multiply(x,r);
		#pragma omp barrier
		#pragma omp master
		{
			rnrm=0;
			for (j=0;j<n;j++) {
				r[j] = b[j] - r[j];
				sigma[j] = r[j];
				rnrm += r[j]*r[j];
			}
			rnrm = sqrt(rnrm);
			signorm = rnrm;
			bnrm=0; znrm=0;
			for (j=0; j < n; j++) {
				temp = (A_sparse[j] != 0) ? b[j]/A_sparse[j] : b[j]; // diagonal preconditioner
				bnrm += temp*temp;
				z[j] = (A_sparse[j] != 0) ? r[j]/A_sparse[j] : r[j]; // diagonal preconditioner
				gamma[j] = z[j];
				znrm += z[j]*z[j];
			}
			bnrm = sqrt(bnrm);
			znrm = sqrt(znrm);
			for (j=0;j<n;j++) {
				p[j]=0;
				rho[j]=0;
			}
		}

		#pragma omp barrier
		while (iterations < itmax)
		{
			#pragma omp barrier
			if (!ratio_mode)
			{
				#pragma omp master
				{
					iterations++;
					bknum=0;
					for (j=0;j<n;j++) {
						bknum += z[j]*r[j];
					}
					bk=bknum/bkden;
					beta[k]=bk;
					for (j=0;j<n;j++) {
						p[j] = z[j] + bk*p[j];
					}
					bkden=bknum;
				}
				#pragma omp barrier
				A_matrix_multiply(p,y);
				#pragma omp barrier
				#pragma omp master
				{
					akden=0;
					for (j=0;j<n;j++) {
						akden += y[j]*p[j];
					}
					ak=bknum/akden;
					alpha[k]=ak;
					older_rnrm=old_rnrm;
					old_rnrm=rnrm;
					rnrm=0;
					for (j=0;j<n;j++) {
						x[j] += ak*p[j];
						r[j] -= ak*y[j];
						rnrm += r[j]*r[j];
					}
					rnrm = sqrt(rnrm);

					if (find_determinant) {
						if (k==0) {
							bkk[0] = 0;
							akk[0] = 1.0/alpha[0];
						} else {
							bkk[k] = sqrt(beta[k-1])/alpha[k-1];
							akk[k] = 1.0/alpha[k] + beta[k-1]/alpha[k-1];
						}
						log_predet_temp = log_pre_det;
						log_pre_det = log_pre_det + log(akk[k] - (bkk[k]*bkk[k])*exp(log_pre_det_last - log_pre_det));
						log_pre_det_last = log_predet_temp;
					}

					zm1nrm=znrm;
					znrm=0;
					for (j=0; j < n; j++) {
						z[j] = (A_sparse[j] != 0) ? r[j]/A_sparse[j] : r[j]; // diagonal preconditioner
						znrm += z[j]*z[j];
					}
					znrm = sqrt(znrm);
					temp = fabs(zm1nrm-znrm);
					if (temp > EPS*znrm) {
						errnorm = 0.0;
						for (j=0; j < n; j++) {
							errnorm += p[j]*p[j];
						}
						dxnrm = sqrt(errnorm);

						dxnrm *= fabs(ak);
						err=znrm/fabs(zm1nrm-znrm)*dxnrm;
						will_continue = false;
					} else {
						err=znrm/bnrm;
						will_continue = true;
					}
					if (++k >= n) k=0; // increase index k; if it has filled all n elements, start over
				}
				#pragma omp barrier
				if (will_continue) continue;
				#pragma omp barrier
				#pragma omp master
				{
					errnorm = 0.0;
					for (j=0; j < n; j++) {
						errnorm += x[j]*x[j];
					}
					xnrm = sqrt(errnorm);

					temp = err;
					if (temp <= 0.5*xnrm) {
						err /= xnrm;
						will_continue = false;
					}
					else {
						err=znrm/bnrm;
						will_continue = true;
					}
					if (((will_continue==false) and (err <= tol)) and (find_determinant) and (iterations < n)) {
						will_continue = true;
						ratio_mode = true;
						bkden = bkden / (old_rnrm*old_rnrm);
						old_signorm = old_rnrm / older_rnrm;
						signorm = rnrm / old_rnrm;
						for (j=0;j<n;j++) {
							rho[j] = p[j]/older_rnrm;
							gamma[j] = z[j]/old_rnrm;
							sigma[j] = r[j]/old_rnrm;
						}
					} // need at least n iterations to find determinant
				}
				#pragma omp barrier
				if (will_continue) continue;
				if (err <= tol) break;
			}
			else
			{
				#pragma omp master
				{
					iterations++;
					bknum=0;
					for (j=0;j<n;j++) {
						bknum += gamma[j]*sigma[j];
					}
					bk=bknum/bkden;
					beta[k]=bk;
					for (j=0;j<n;j++) {
						rho[j] = gamma[j] + bk*rho[j]/old_signorm;
					}
					bkden=bknum/(signorm*signorm);
				}
				#pragma omp barrier
				A_matrix_multiply(rho,y);
				#pragma omp barrier
				#pragma omp master
				{
					akden=0;
					for (j=0;j<n;j++) {
						akden += y[j]*rho[j];
					}
					ak=bknum/akden;
					alpha[k]=ak;
					old_signorm=signorm;
					signorm=0;
					for (j=0;j<n;j++) {
						sigma[j] = (sigma[j]-ak*y[j])/old_signorm;
						signorm += sigma[j]*sigma[j];
					}
					signorm = sqrt(signorm);

					if (find_determinant) {
						if (k==0) {
							bkk[0] = 0;
							akk[0] = 1.0/alpha[0];
						} else {
							bkk[k] = sqrt(beta[k-1])/alpha[k-1];
							akk[k] = 1.0/alpha[k] + beta[k-1]/alpha[k-1];
						}
						log_predet_temp = log_pre_det;
						double wtf = (bkk[k]*bkk[k])*exp(log_pre_det_last-log_pre_det);
						//if (akk[k] < wtf) cerr << "uh-oh: determinant is becoming negative wtf=" << wtf << endl;
						log_pre_det = log_pre_det + log(fabs(akk[k] - (bkk[k]*bkk[k])*exp(log_pre_det_last - log_pre_det)));
						log_pre_det_last = log_predet_temp;
					}

					for (j=0; j < n; j++) {
						gamma[j] = (A_sparse[j] != 0) ? sigma[j]/A_sparse[j] : sigma[j]; // diagonal preconditioner
					}
					if (iterations < n) will_continue = true;
					else will_continue = false;
					if (++k >= n) k=0; // increase index k; if it has filled all n elements, start over
				}
				#pragma omp barrier
				if (will_continue) continue;
				break;
			}
		}
	}

#ifdef USE_OPENMP
	omp_set_num_threads(default_nthreads);
#endif

	if (find_determinant) {
		if (iterations < n) die("should not allow less than n iterations when determinant mode is on (it=%i,n=%i)",iterations,n);
		double log_preconditioner_det = 0;
		for (int i=0; i < n; i++) log_preconditioner_det += log(A_sparse[i]);
		log_determinant = log_pre_det + log_preconditioner_det; // determinant of preconditioned matrix times determinant of the preconditioner itself
		//cout << "LOGDETS: " << log_pre_det << " " << log_preconditioner_det << endl;
		//cout << "Determinant: " << det << endl;
	}
	ratio_mode = false;
}

double CG_sparse::calculate_log_determinant()
{
	// note, the sparse version uses a diagonal preconditioner specifically
	double ak,akden,bk,bkden=1.0,bknum,bnrm,dxnrm,xnrm,zm1nrm,znrm=0;
	static const double EPS=1.0e-14;
	int j,k;

	double y[n];
	double alpha[n];
	double beta[n];
	double akk[n], bkk[n];

	double log_pre_det, log_pre_det_last, log_predet_temp;
	log_pre_det_last = 0;
	log_pre_det = 0;

	k=0;
	double rho[n], sigma[n], gamma[n];
	double signorm, old_signorm=1.0;

#ifdef USE_OPENMP
	omp_set_num_threads(nthreads);
#endif
	
	#pragma omp parallel
	{
		#pragma omp master
		{
			signorm=0;
			for (j=0;j<n;j++) {
				sigma[j] = 1.0; //starting point shouldn't matter, although there can be some rounding error that depends on initial sigma if n is large
				signorm += sigma[j]*sigma[j];
			}
			signorm = sqrt(signorm);
			for (j=0; j < n; j++) {
				gamma[j] = (A_sparse[j] != 0) ? sigma[j]/A_sparse[j] : sigma[j]; // diagonal preconditioner
			}
			for (j=0;j<n;j++) {
				rho[j]=0;
			}
		}

		#pragma omp barrier
		for (k=0; k < n; k++)
		{
			#pragma omp barrier
			#pragma omp master
			{
				bknum=0;
				for (j=0;j<n;j++) {
					bknum += gamma[j]*sigma[j];
				}
				bk=bknum/bkden;
				beta[k]=bk;
				for (j=0;j<n;j++) {
					rho[j] = gamma[j] + bk*rho[j]/old_signorm;
				}
				bkden=bknum/(signorm*signorm);
			}
			#pragma omp barrier
			A_matrix_multiply(rho,y);
			#pragma omp barrier
			#pragma omp master
			{
				akden=0;
				for (j=0;j<n;j++) {
					akden += y[j]*rho[j];
				}
				ak=bknum/akden;
				alpha[k]=ak;
				old_signorm=signorm;
				signorm=0;
				for (j=0;j<n;j++) {
					sigma[j] = (sigma[j]-ak*y[j])/old_signorm;
					signorm += sigma[j]*sigma[j];
				}
				signorm = sqrt(signorm);

				if (k==0) {
					bkk[0] = 0;
					akk[0] = 1.0/alpha[0];
				} else {
					bkk[k] = sqrt(beta[k-1])/alpha[k-1];
					akk[k] = 1.0/alpha[k] + beta[k-1]/alpha[k-1];
				}
				log_predet_temp = log_pre_det;
				log_pre_det = log_pre_det + log(akk[k] - (bkk[k]*bkk[k])*exp(log_pre_det_last - log_pre_det));
				log_pre_det_last = log_predet_temp;

				for (j=0; j < n; j++) {
					gamma[j] = (A_sparse[j] != 0) ? sigma[j]/A_sparse[j] : sigma[j]; // diagonal preconditioner
				}
			}
		}
	}

#ifdef USE_OPENMP
	omp_set_num_threads(default_nthreads);
#endif
	double log_preconditioner_det = 0;
	for (int i=0; i < n; i++) log_preconditioner_det += log(A_sparse[i]);
	log_determinant = log_pre_det + log_preconditioner_det; // determinant of preconditioned matrix times determinant of the preconditioner itself
	return log_determinant;
}

void CG_Solver::error_norm(double* sx, double& err)
{
	// Compute one of two norms for a vector sx[0..n-1]. Used by solve.
	static double ans;
	ans = 0.0;
	for (int i=0; i < n; i++) {
		ans += SQR(sx[i]);
	}
	err = sqrt(ans);
	//cout << "Error = " << err << endl;
	//#pragma omp for ordered reduction(+:ans)
	//#pragma omp for reduction(+:ans)
}

void CG_sparse::A_matrix_multiply(const double* const x, double* const r)
{
	int i,j;
	int mpi_chunk, mpi_i_start, mpi_i_end;
	mpi_chunk = n / mpi_np;
	mpi_i_start = mpi_id*mpi_chunk;
	if (mpi_id == mpi_np-1) mpi_chunk += (n % mpi_np); // assign the remainder elements to the last mpi process
	mpi_i_end = mpi_i_start + mpi_chunk;

	#pragma omp for schedule(static)
	for (i=mpi_i_start; i < mpi_i_end; i++) {
		r[i] = A_sparse[i] * x[i];
		for (j=A_index[i]; j < A_index[i+1]; j++) {
			r[i] += A_sparse[j] * x[A_index[j]];
		}
		for (j=0; j < sorted_indices[i].size(); j++) {
			r[i] += A_sparse[sorted_indices[i][j]] * x[sorted_indices_i[i][j]];
		}
	}

	#pragma omp master
	{
#ifdef USE_MPI
		int chunk, i_start;
		chunk = n / mpi_np;
		for (i=0; i < mpi_np; i++) {
			i_start = i*chunk;
			if (i == mpi_np-1) chunk += (n % mpi_np); // assign the remainder elements to the last mpi process
			//cout << "About to broadcast (process " << mpi_id << ", thread " << omp_get_thread_num() << ")...\n" << flush;
			MPI_Bcast(r + i_start,chunk,MPI_DOUBLE,i,(*mpi_comm));
		}
#endif
	}
}

void CG_sparse::incomplete_Cholesky_preconditioner()
{
	preconditioner = new double[A_length];
	int i,j,k;
	double pivotsum;

	for (i=0; i < A_length; i++) preconditioner[i] = A_sparse[i];

	preconditioner[0] = sqrt(preconditioner[0]);
	for (j=A_index[0]; j < A_index[1]; j++) preconditioner[j] /= preconditioner[0];
	pivotsum = preconditioner[0];

	for (i=1; i < n; i++) {
		// we skip the subtracting portion entirely, since this makes the decomposition unstable for the sparse lensing matrices
		if (preconditioner[i] <= 0) {
			warn("Incomplete Cholesky decomposition is failing: matrix is no longer positive-definite (row %i)",i);
			preconditioner[i] = pivotsum / i;
		}
		pivotsum += preconditioner[i];
		preconditioner[i] = sqrt(preconditioner[i]);
		for (j=A_index[i]; j < A_index[i+1]; j++) preconditioner[j] /= preconditioner[i];
	}

	preconditioner_transpose = new double[A_length];
	preconditioner_transpose_index = new int[A_length];

	int jl,jm,jp,ju,m,n2,noff,inc,iv;
	double v;

	n2=A_index[0];
	for (j=0; j < n2-1; j++) preconditioner_transpose[j] = preconditioner[j];
	int n_offdiag = A_index[n2-1] - A_index[0];
	int *offdiag_indx = new int[n_offdiag];
	int *offdiag_indx_transpose = new int[n_offdiag];
	for (i=0; i < n_offdiag; i++) offdiag_indx[i] = A_index[n2+i];
	indexx(offdiag_indx,offdiag_indx_transpose,n_offdiag);
	for (j=n2, k=0; j < A_index[n2-1]; j++, k++) {
		preconditioner_transpose_index[j] = offdiag_indx_transpose[k];
	}
	jp=0;
	for (k=A_index[0]; k < A_index[n2-1]; k++) {
		m = preconditioner_transpose_index[k] + n2;
		preconditioner_transpose[k] = preconditioner[m];
		for (j=jp; j < A_index[m]+1; j++)
			preconditioner_transpose_index[j]=k;
		jp = A_index[m] + 1;
		jl=0;
		ju=n2-1;
		while (ju-jl > 1) {
			jm = (ju+jl)/2;
			if (A_index[jm] > m) ju=jm; else jl=jm;
		}
		preconditioner_transpose_index[k]=jl;
	}
	for (j=jp; j < n2; j++) preconditioner_transpose_index[j] = A_index[n2-1];
	for (j=0; j < n2-1; j++) {
		jl = preconditioner_transpose_index[j+1] - preconditioner_transpose_index[j];
		noff=preconditioner_transpose_index[j];
		inc=1;
		do {
			inc *= 3;
			inc++;
		} while (inc <= jl);
		do {
			inc /= 3;
			for (k=noff+inc; k < noff+jl; k++) {
				iv = preconditioner_transpose_index[k];
				v = preconditioner_transpose[k];
				m=k;
				while (preconditioner_transpose_index[m-inc] > iv) {
					preconditioner_transpose_index[m] = preconditioner_transpose_index[m-inc];
					preconditioner_transpose[m] = preconditioner_transpose[m-inc];
					m -= inc;
					if (m-noff+1 <= inc) break;
				}
				preconditioner_transpose_index[m] = iv;
				preconditioner_transpose[m] = v;
			}
		} while (inc > 1);
	}
	delete[] offdiag_indx;
	delete[] offdiag_indx_transpose;
}

void CG_sparse::Cholesky_preconditioner_solve(double* b, double* x)
{
	int i,k;
	static double sum;

	for (i=0; i < n; i++) { // sum over rows
		sum = b[i];
		for (k=preconditioner_transpose_index[i]; k < preconditioner_transpose_index[i+1]; k++) {
			sum -= preconditioner_transpose[k]*x[preconditioner_transpose_index[k]]; //sum over columns
		}
		x[i] = sum / preconditioner_transpose[i];
	}
	for (i=n-1; i >= 0; i--) { // sum over rows
		sum = x[i];
		for (k=A_index[i]; k < A_index[i+1]; k++) {
			sum -= preconditioner[k]*x[A_index[k]]; //sum over columns
		}
		x[i] = sum / preconditioner[i];
	}
}

void CG_sparse::preconditioner_solve(double* r, double* x)
{
	// diagonal preconditioner
	for (int i=0; i < n; i++)
		x[i] = (A_sparse[i] != 0) ? r[i]/A_sparse[i] : r[i];
}

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
void CG_sparse::indexx(int* arr, int* indx, int nn)
{
	const int M=7, NSTACK=50;
	int i,indxt,ir,j,k,jstack=-1,l=0;
	double a,temp;
	int *istack = new int[NSTACK];
	ir = nn - 1;
	for (j=0; j < nn; j++) indx[j] = j;
	for (;;) {
		if (ir-l < M) {
			for (j=l+1; j <= ir; j++) {
				indxt=indx[j];
				a=arr[indxt];
				for (i=j-1; i >=l; i--) {
					if (arr[indx[i]] <= a) break;
					indx[i+1]=indx[i];
				}
				indx[i+1]=indxt;
			}
			if (jstack < 0) break;
			ir=istack[jstack--];
			l=istack[jstack--];
		} else {
			k=(l+ir) >> 1;
			SWAP(indx[k],indx[l+1]);
			if (arr[indx[l]] > arr[indx[ir]]) {
				SWAP(indx[l],indx[ir]);
			}
			if (arr[indx[l+1]] > arr[indx[ir]]) {
				SWAP(indx[l+1],indx[ir]);
			}
			if (arr[indx[l]] > arr[indx[l+1]]) {
				SWAP(indx[l],indx[l+1]);
			}
			i=l+1;
			j=ir;
			indxt=indx[l+1];
			a=arr[indxt];
			for (;;) {
				do i++; while (arr[indx[i]] < a);
				do j--; while (arr[indx[j]] > a);
				if (j < i) break;
				SWAP(indx[i],indx[j]);
			}
			indx[l+1]=indx[j];
			indx[j]=indxt;
			jstack += 2;
			if (jstack >= NSTACK) die("NSTACK too small in indexx");
			if (ir-i+1 >= j-l) {
				istack[jstack]=ir;
				istack[jstack-1]=i;
				ir=j-1;
			} else {
				istack[jstack]=j-1;
				istack[jstack-1]=l;
				l=i;
			}
		}
	}
	delete[] istack;
}
#undef SWAP(a,b)

CG_sparse::~CG_sparse()
{
	delete[] sorted_indices;
	delete[] sorted_indices_i;
	if (preconditioner != NULL) delete[] preconditioner;
	if (preconditioner_transpose != NULL) delete[] preconditioner_transpose;
	if (preconditioner_transpose_index != NULL) delete[] preconditioner_transpose_index;
}


