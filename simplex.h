#ifndef SIMPLEX_H
#define SIMPLEX_H

#include "rand.h"
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace std;

class Simplex : public Random
{
	bool initialized;
	double temp;
	double tinc, t0, tfinal;

	static const double t0_default;
	static const double tfinal_default;
	static const double tinc_default;
	static const int max_iterations_default;
	static const int max_iterations_anneal_default;

	public:
	double (Simplex::*func)(double*);
	double ftol;
	double yb;
	int ndim;
	double *pb;
	int mpts;
	double *y;
	double **p;
	double tt;
	double *disps;
	int max_iterations, max_iterations_anneal;
	double fmin;

	bool simplex_exit_status;

	Simplex() : Random(-10) {
		initialized = false;
		t0 = t0_default;
		tfinal = tfinal_default;
		tinc = tinc_default;
	}
	Simplex(double* point, const int& ndim_in, const double& vertex_displacement, const double& ftol_in, const int seed_in)
	{
		t0 = t0_default;
		tfinal = tfinal_default;
		tinc = tinc_default;
		initialized = false;
		double disps_in[ndim_in];
		for (int i=0; i < ndim_in; i++) disps_in[i] = vertex_displacement;
		initialize_simplex(point,ndim_in,disps_in,ftol_in,seed_in);
		func = NULL;
	}
	Simplex(double* point, const int& ndim_in, double* vertex_displacements, const double& ftol_in, const int seed_in)
	{
		t0 = t0_default;
		tfinal = tfinal_default;
		tinc = tinc_default;
		initialized = false;
		initialize_simplex(point,ndim_in,vertex_displacements,ftol_in,seed_in);
		func = NULL;
	}
	~Simplex();

	void initialize_simplex(double* point, const int& ndim_in, const double& vertex_displacement, const double& ftol_in, const int seed_in = 100000)
	{
		// Here we assign the same initial displacement amount for each vertex
		double disps_in[ndim_in];
		for (int i=0; i < ndim_in; i++) disps_in[i] = vertex_displacement;
		initialize_simplex(point,ndim_in,disps_in,ftol_in,seed_in);
	}
	void initialize_simplex(double* point, const int& ndim_in, double* vertex_displacements, const double& ftol_in, const int seed_in = 100000);
	void simplex_set_function(double (Simplex::*func_in)(double*)) { func = func_in; }
	void simplex_set_fmin(double fmin_in) { fmin = fmin_in; }
	void set_annealing_schedule_parameters(double t0_in, double tfinal_in, double tinc_in, double nmax_anneal_in, double nmax_in) {
		t0 = t0_in;
		tfinal = tfinal_in;
		tinc = tinc_in;
		max_iterations = nmax_in;
		max_iterations_anneal = nmax_anneal_in;
	}
	void simplex_minval(double x[], double &f)
	{
		for (int i=0; i < ndim; i++) x[i] = pb[i];
		f = yb;
	}
	void reset_simplex(double* point)
	{
		int i,j;
		for (i=0; i < mpts; i++) {
			for (j=0; j < ndim; j++)
				p[i][j]=point[j];
			if (i != 0) p[i][i-1] += disps[i-1];
		}
		yb = 1e30;
	}
	void downhill_simplex(int &nfunk, const int& nmax, const double& temperature);
	void get_psum(double* psum)
	{
		// the following code is from Numerical Recipes in C
		int n,m;
		double sum;
		for (n=0; n < ndim; n++) {
			for (sum=0.0, m=0; m < mpts; m++) sum += p[m][n];
			psum[n]=sum;
		}
	}

	double amotry(double* psum, const int &ihi, double& yhi, const double& fac);
	int downhill_simplex_anneal(bool verbal = false);
	void simplex_evaluate_bestfit_point() { yb=(this->*func)(pb); }
};

#endif // SIMPLEX_H
