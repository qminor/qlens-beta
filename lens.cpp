#include "qlens.h"
#include "pixelgrid.h"
#include "profile.h"
#include "sbprofile.h"
#include "mathexpr.h"
#include "vector.h"
#include "matrix.h"
#include "errors.h"
#include "romberg.h"
#include "spline.h"
#include "mcmchdr.h"
#include "hyp_2F1.h"
#include "cosmo.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <sys/stat.h>
using namespace std;

#ifdef USE_MULTINEST
#include "multinest.h"
#endif

#ifdef USE_POLYCHORD
#include "interfaces.hpp"
#endif

const int QLens::nmax_lens_planes = 100;
const double QLens::default_autogrid_initial_step = 1.0e-3;
const double QLens::default_autogrid_rmin = 1.0e-5;
const double QLens::default_autogrid_rmax = 1.0e5;
const double QLens::default_autogrid_frac = 2.1; // ****** NOTE: it might be better to make this depend on the axis ratio, since for q=1 you may need larger rfrac
const int QLens::max_cc_search_iterations = 8;
double QLens::galsubgrid_radius_fraction; // radius of perturber subgridding in terms of fraction of Einstein radius
double QLens::galsubgrid_min_cellsize_fraction; // minimum cell size for perturber subgridding in terms of fraction of Einstein radius
int QLens::galsubgrid_cc_splittings;
bool QLens::auto_store_cc_points;
const double QLens::perturber_einstein_radius_fraction = 0.2;
const double QLens::default_rmin_frac = 1e-4;
bool QLens::warnings;
bool QLens::newton_warnings; // newton_warnings: when true, displays warnings when Newton's method fails or returns anomalous results
bool QLens::use_scientific_notation;
bool QLens::use_ansi_output_during_fit;
double QLens::rmin_frac;
bool QLens::respline_at_end; // for creating deflection spline
int QLens::resplinesteps; // for creating deflection spline
string QLens::fit_output_filename;

int QLens::nthreads = 0;
lensvector *QLens::defs = NULL, **QLens::defs_subtot = NULL, *QLens::defs_i = NULL, *QLens::xvals_i = NULL;
lensmatrix *QLens::jacs = NULL, *QLens::hesses = NULL, **QLens::hesses_subtot = NULL, *QLens::hesses_i = NULL, *QLens::Amats_i = NULL;
int *QLens::indxs = NULL;

// The ParamSettings stuff should go in a separate cpp file!! Doesn't belong in lens.cpp
void ParamSettings::update_params(const int nparams_in, vector<string>& names, double* stepsizes_in)
{
	int i;
	if (nparams==nparams_in) {
		// update parameter names just in case
		for (i=0; i < nparams_in; i++) {
			param_names[i] = names[i];
		}
		return;
	}
	int newparams = nparams_in - nparams;
	ParamPrior** newpriors = new ParamPrior*[nparams_in];
	ParamTransform** newtransforms = new ParamTransform*[nparams_in];
	double* new_stepsizes = new double[nparams_in];
	bool* new_auto_stepsize = new bool[nparams_in];
	bool* new_subplot_param = new bool[nparams_in];
	bool* new_hist2d_param = new bool[nparams_in];
	double* new_penalty_limits_lo = new double[nparams_in];
	double* new_penalty_limits_hi = new double[nparams_in];
	double* new_prior_norms = new double[nparams_in];
	bool* new_use_penalty_limits = new bool[nparams_in];
	if (param_names != NULL) delete[] param_names;
	param_names = new string[nparams_in];
	if (nparams_in > nparams) {
		for (i=0; i < nparams; i++) {
			newpriors[i] = new ParamPrior(priors[i]);
			newtransforms[i] = new ParamTransform(transforms[i]);
			new_stepsizes[i] = stepsizes[i];
			new_auto_stepsize[i] = auto_stepsize[i];
			new_subplot_param[i] = subplot_param[i];
			new_hist2d_param[i] = hist2d_param[i];
			new_penalty_limits_lo[i] = penalty_limits_lo[i];
			new_penalty_limits_hi[i] = penalty_limits_hi[i];
			new_prior_norms[i] = prior_norms[i];
			new_use_penalty_limits[i] = use_penalty_limits[i];
			param_names[i] = names[i];
		}
		for (i=nparams; i < nparams_in; i++) {
			//cout << "New parameter " << i << ", setting plimits to false" << endl;
			newpriors[i] = new ParamPrior();
			newtransforms[i] = new ParamTransform();
			param_names[i] = names[i];
			new_stepsizes[i] = stepsizes_in[i];
			new_auto_stepsize[i] = true; // stepsizes for newly added parameters are set to 'auto'
			new_subplot_param[i] = false; 
			new_hist2d_param[i] = true; 
			new_penalty_limits_lo[i] = -1e30;
			new_penalty_limits_hi[i] = 1e30;
			new_prior_norms[i] = 1.0;
			new_use_penalty_limits[i] = false;
		}
	} else {
		for (i=0; i < nparams_in; i++) {
			newpriors[i] = new ParamPrior(priors[i]);
			newtransforms[i] = new ParamTransform(transforms[i]);
			new_stepsizes[i] = stepsizes[i];
			new_auto_stepsize[i] = auto_stepsize[i];
			new_subplot_param[i] = subplot_param[i];
			new_hist2d_param[i] = hist2d_param[i];
			new_penalty_limits_lo[i] = penalty_limits_lo[i];
			new_penalty_limits_hi[i] = penalty_limits_hi[i];
			new_prior_norms[i] = prior_norms[i];
			new_use_penalty_limits[i] = use_penalty_limits[i];
			param_names[i] = names[i];
		}
	}
	if (nparams > 0) {
		for (i=0; i < nparams; i++) {
			delete priors[i];
			delete transforms[i];
		}
		delete[] priors;
		delete[] transforms;
		delete[] stepsizes;
		delete[] auto_stepsize;
		delete[] subplot_param;
		delete[] hist2d_param;
		delete[] penalty_limits_lo;
		delete[] penalty_limits_hi;
		delete[] prior_norms;
		delete[] use_penalty_limits;
	}
	priors = newpriors;
	transforms = newtransforms;
	stepsizes = new_stepsizes;
	auto_stepsize = new_auto_stepsize;
	subplot_param = new_subplot_param;
	hist2d_param = new_hist2d_param;
	penalty_limits_lo = new_penalty_limits_lo;
	penalty_limits_hi = new_penalty_limits_hi;
	prior_norms = new_prior_norms;
	use_penalty_limits = new_use_penalty_limits;
	nparams = nparams_in;
}
void ParamSettings::insert_params(const int pi, const int pf, vector<string>& names, double* stepsizes_in)
{
	int i, j, np = pf-pi;
	int new_nparams = nparams + np;
	ParamPrior** newpriors = new ParamPrior*[new_nparams];
	ParamTransform** newtransforms = new ParamTransform*[new_nparams];
	double* new_stepsizes = new double[new_nparams];
	bool* new_auto_stepsize = new bool[new_nparams];
	bool* new_subplot_param = new bool[new_nparams];
	bool* new_hist2d_param = new bool[new_nparams];
	double* new_penalty_limits_lo = new double[new_nparams];
	double* new_penalty_limits_hi = new double[new_nparams];
	double* new_prior_norms = new double[new_nparams];
	bool* new_use_penalty_limits = new bool[new_nparams];
	string* new_param_names = new string[new_nparams];
	for (i=0; i < pi; i++) {
		newpriors[i] = new ParamPrior(priors[i]);
		newtransforms[i] = new ParamTransform(transforms[i]);
		new_auto_stepsize[i] = auto_stepsize[i];
		new_hist2d_param[i] = hist2d_param[i];
		new_subplot_param[i] = subplot_param[i];
		new_stepsizes[i] = (auto_stepsize[i]) ? stepsizes_in[i] : stepsizes[i]; // if stepsizes are set to 'auto', use new auto stepsizes since they might have changed
		new_penalty_limits_lo[i] = penalty_limits_lo[i];
		new_penalty_limits_hi[i] = penalty_limits_hi[i];
		new_prior_norms[i] = prior_norms[i];
		new_use_penalty_limits[i] = use_penalty_limits[i];
		new_param_names[i] = names[i];
	}
	for (i=pi; i < pf; i++) {
		newpriors[i] = new ParamPrior();
		newtransforms[i] = new ParamTransform();
		new_stepsizes[i] = stepsizes_in[i];
		new_auto_stepsize[i] = true; // stepsizes for newly added parameters are set to 'auto'
		new_hist2d_param[i] = true; // stepsizes for newly added parameters are set to 'auto'
		new_subplot_param[i] = false; // stepsizes for newly added parameters are set to 'auto'
		new_penalty_limits_lo[i] = -1e30;
		new_penalty_limits_hi[i] = 1e30;
		new_prior_norms[i] = 1.0;
		new_use_penalty_limits[i] = false;
		new_param_names[i] = names[i];
	}
	for (j=pf,i=pi; i < nparams; i++, j++) {
		newpriors[j] = new ParamPrior(priors[i]);
		newtransforms[j] = new ParamTransform(transforms[i]);
		new_auto_stepsize[j] = auto_stepsize[i];
		new_hist2d_param[j] = hist2d_param[i];
		new_subplot_param[j] = subplot_param[i];
		new_stepsizes[j] = (auto_stepsize[i]) ? stepsizes_in[j] : stepsizes[i]; // if stepsizes are set to 'auto', use new auto stepsizes since they might have changed
		new_penalty_limits_lo[j] = penalty_limits_lo[i];
		new_penalty_limits_hi[j] = penalty_limits_hi[i];
		new_prior_norms[j] = prior_norms[i];
		new_use_penalty_limits[j] = use_penalty_limits[i];
		new_param_names[j] = param_names[i];
		//cout << "Penalty limit " << i << " is now penalty limit " << j << ", which is " << use_penalty_limits[i] << " with range " << penalty_limits_lo[i] << " " << penalty_limits_hi[i] << endl;
	}
	if (nparams > 0) {
		for (i=0; i < nparams; i++) {
			delete priors[i];
			delete transforms[i];
		}
		delete[] priors;
		delete[] transforms;
		delete[] stepsizes;
		delete[] auto_stepsize;
		delete[] subplot_param;
		delete[] hist2d_param;
		delete[] penalty_limits_lo;
		delete[] penalty_limits_hi;
		delete[] prior_norms;
		delete[] use_penalty_limits;
		delete[] param_names;
	}
	priors = newpriors;
	transforms = newtransforms;
	stepsizes = new_stepsizes;
	auto_stepsize = new_auto_stepsize;
	subplot_param = new_subplot_param;
	hist2d_param = new_hist2d_param;
	penalty_limits_lo = new_penalty_limits_lo;
	penalty_limits_hi = new_penalty_limits_hi;
	prior_norms = new_prior_norms;
	use_penalty_limits = new_use_penalty_limits;
	param_names = new_param_names;
	nparams = new_nparams;
}
bool ParamSettings::remove_params(const int pi, const int pf)
{
	if (pf > nparams) return false;
	int i, j, np = pf-pi;
	if (np==nparams) {
		clear_params();
		return true;
	}
	int new_nparams = nparams - np;
	ParamPrior** newpriors = new ParamPrior*[new_nparams];
	ParamTransform** newtransforms = new ParamTransform*[new_nparams];
	double* new_stepsizes = new double[new_nparams];
	bool* new_auto_stepsize = new bool[new_nparams];
	bool* new_subplot_param = new bool[new_nparams];
	bool* new_hist2d_param = new bool[new_nparams];
	double* new_penalty_limits_lo = new double[new_nparams];
	double* new_penalty_limits_hi = new double[new_nparams];
	double* new_prior_norms = new double[new_nparams];
	bool* new_use_penalty_limits = new bool[new_nparams];
	string* new_param_names = new string[new_nparams];
	for (i=0; i < pi; i++) {
		newpriors[i] = new ParamPrior(priors[i]);
		newtransforms[i] = new ParamTransform(transforms[i]);
		new_stepsizes[i] = stepsizes[i];
		new_auto_stepsize[i] = auto_stepsize[i];
		new_subplot_param[i] = subplot_param[i];
		new_hist2d_param[i] = hist2d_param[i];
		new_penalty_limits_lo[i] = penalty_limits_lo[i];
		new_penalty_limits_hi[i] = penalty_limits_hi[i];
		new_prior_norms[i] = prior_norms[i];
		new_use_penalty_limits[i] = use_penalty_limits[i];
		new_param_names[i] = param_names[i];
	}
	for (i=pf,j=pi; i < nparams; i++, j++) {
		newpriors[j] = new ParamPrior(priors[i]);
		newtransforms[j] = new ParamTransform(transforms[i]);
		new_stepsizes[j] = stepsizes[i];
		new_auto_stepsize[j] = auto_stepsize[i];
		new_subplot_param[j] = subplot_param[i];
		new_hist2d_param[j] = hist2d_param[i];
		new_penalty_limits_lo[j] = penalty_limits_lo[i];
		new_penalty_limits_hi[j] = penalty_limits_hi[i];
		new_prior_norms[j] = prior_norms[i];
		new_use_penalty_limits[j] = use_penalty_limits[i];
		new_param_names[j] = param_names[i];
	}
	for (i=0; i < nparams; i++) {
		delete priors[i];
		delete transforms[i];
	}
	delete[] priors;
	delete[] transforms;
	delete[] stepsizes;
	delete[] auto_stepsize;
	delete[] subplot_param;
	delete[] hist2d_param;
	delete[] penalty_limits_lo;
	delete[] penalty_limits_hi;
	delete[] prior_norms;
	delete[] use_penalty_limits;
	delete[] param_names;
	priors = newpriors;
	transforms = newtransforms;
	stepsizes = new_stepsizes;
	auto_stepsize = new_auto_stepsize;
	subplot_param = new_subplot_param;
	hist2d_param = new_hist2d_param;
	penalty_limits_lo = new_penalty_limits_lo;
	penalty_limits_hi = new_penalty_limits_hi;
	prior_norms = new_prior_norms;
	use_penalty_limits = new_use_penalty_limits;
	param_names = new_param_names;
	nparams = new_nparams;
	return true;
}
void ParamSettings::add_dparam(string dparam_name)
{
	string* new_dparam_names = new string[n_dparams+1];
	bool* new_subplot_dparam = new bool[n_dparams+1];
	bool* new_hist2d_dparam = new bool[n_dparams+1];
	if (n_dparams > 0) {
		for (int i=0; i < n_dparams; i++) {
			new_dparam_names[i] = dparam_names[i];
			new_subplot_dparam[i] = subplot_dparam[i];
			new_hist2d_dparam[i] = hist2d_dparam[i];
		}
		delete[] dparam_names;
		delete[] subplot_dparam;
		delete[] hist2d_dparam;
	}
	new_dparam_names[n_dparams] = dparam_name;
	new_subplot_dparam[n_dparams] = false;
	new_hist2d_dparam[n_dparams] = true;
	n_dparams++;
	dparam_names = new_dparam_names;
	subplot_dparam = new_subplot_dparam;
	hist2d_dparam = new_hist2d_dparam;
}
void ParamSettings::remove_dparam(int dparam_number)
{
	string* new_dparam_names;
	bool* new_subplot_dparam;
	bool* new_hist2d_dparam;
	if (n_dparams > 1) {
		new_dparam_names = new string[n_dparams-1];
		new_subplot_dparam = new bool[n_dparams-1];
		new_hist2d_dparam = new bool[n_dparams-1];
		int i,j;
		for (i=0, j=0; i < n_dparams; i++) {
			if (i != dparam_number) {
				new_dparam_names[j] = dparam_names[i];
				new_subplot_dparam[j] = subplot_dparam[i];
				new_hist2d_dparam[j] = hist2d_dparam[i];
				j++;
			}
		}
	}
	delete[] dparam_names;
	delete[] subplot_dparam;
	delete[] hist2d_dparam;
	n_dparams--;
	if (n_dparams > 0) {
		dparam_names = new_dparam_names;
		subplot_dparam = new_subplot_dparam;
		hist2d_dparam = new_hist2d_dparam;
	} else {
		dparam_names = NULL;
		subplot_dparam = NULL;
		hist2d_dparam = NULL;
	}
}

void ParamSettings::print_priors()
{
	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter settings:\n";
	int max_length=0;
	for (int i=0; i < nparams; i++) {
		if (param_names[i].length() > max_length) max_length = param_names[i].length();
	}
	int extra_length;
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << param_names[i] << ": ";
		extra_length = max_length - param_names[i].length();
		for (int j=0; j < extra_length; j++) cout << " ";
		if ((nparams > 10) and (i < 10)) cout << " ";
		if (priors[i]->prior==UNIFORM_PRIOR) cout << "uniform prior";
		else if (priors[i]->prior==LOG_PRIOR) cout << "log prior";
		else if (priors[i]->prior==GAUSS_PRIOR) {
			cout << "gaussian prior (mean=" << priors[i]->gaussian_pos << ", sigma=" << priors[i]->gaussian_sig << ")";
		}
		else if (priors[i]->prior==GAUSS2_PRIOR) {
			cout << "multivariate gaussian prior, params=(" << i << "," << priors[i]->gauss_paramnums[1] << "), mean=(" << priors[i]->gauss_meanvals[0] << "," << priors[i]->gauss_meanvals[1] << "), sigs=(" << sqrt(priors[i]->covariance_matrix[0][0]) << "," << (sqrt(priors[i]->covariance_matrix[1][1])) << "), sqrt(sig12) = " << (sqrt(priors[i]->covariance_matrix[0][1]));
		} else if (priors[i]->prior==GAUSS2_PRIOR_SECONDARY) {
			cout << "multivariate gaussian prior, params=(" << priors[i]->gauss_paramnums[0] << "," << priors[i]->gauss_paramnums[1] << ")";
		}
		else die("Prior type unknown");
		if (transforms[i]->transform==NONE) ;
		else if (transforms[i]->transform==LOG_TRANSFORM) cout << ", log transformation";
		else if (transforms[i]->transform==GAUSS_TRANSFORM) cout << ", gaussian transformation (mean=" << transforms[i]->gaussian_pos << ", sigma=" << transforms[i]->gaussian_sig << ")";
		else if (transforms[i]->transform==LINEAR_TRANSFORM) cout << ", linear transformation A*" << param_names[i] << " + b (A=" << transforms[i]->a << ", b=" << transforms[i]->b << ")";
		else if (transforms[i]->transform==RATIO) cout << ", ratio transformation " << param_names[i] << "/" << param_names[transforms[i]->ratio_paramnum];
		if (transforms[i]->include_jacobian==true) cout << " (include Jacobian in likelihood)";
		cout << endl;
	}
}

void ParamSettings::print_stepsizes()
{
	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter initial stepsizes:\n";
	string *transformed_names = new string[nparams];
	transform_parameter_names(param_names,transformed_names,NULL,NULL);
	int max_length=0;
	for (int i=0; i < nparams; i++) {
		if (transformed_names[i].length() > max_length) max_length = transformed_names[i].length();
	}
	int extra_length;
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << transformed_names[i] << ": ";
		extra_length = max_length - transformed_names[i].length();
		for (int j=0; j < extra_length; j++) cout << " ";
		if ((nparams > 10) and (i < 10)) cout << " ";
		cout << stepsizes[i];
		if (auto_stepsize[i]) cout << " (auto)";
		cout << endl;
	}
	delete[] transformed_names;
}

void ParamSettings::print_penalty_limits()
{
	//for (int i=0; i < nparams; i++) {
		//if (use_penalty_limits[i]==true) {
			//cout << "USE_LIMITS " << i << endl;
		//}
	//}

	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter limits imposed on chi-square:\n";
	int max_length=0;
	for (int i=0; i < nparams; i++) {
		if (param_names[i].length() > max_length) max_length = param_names[i].length();
	}
	int extra_length;
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << param_names[i] << ": ";
		extra_length = max_length - param_names[i].length();
		for (int j=0; j < extra_length; j++) cout << " ";
		if ((nparams > 10) and (i < 10)) cout << " ";
		if (use_penalty_limits[i]==false) cout << "none" << endl;
		else {
			cout << "[";
			if (penalty_limits_lo[i]==-1e30) cout << "-inf";
			else cout << penalty_limits_lo[i];
			cout << ":";
			if (penalty_limits_hi[i]==1e30) cout << "inf";
			else cout << penalty_limits_hi[i];
			cout << "]" << endl;
		}
	}
}


void QLens::allocate_multithreaded_variables(const int& threads, const bool reallocate)
{
	if (xvals_i != NULL) {
		if (!reallocate) return;
		else deallocate_multithreaded_variables();
	}
	nthreads = threads;
	// Note: the grid construction is not being parallelized any more...if you decide to ditch it for good, then get rid of these multithreaded variables and replace by single-thread version
	xvals_i = new lensvector[nthreads];
	defs = new lensvector[nthreads];
	defs_subtot = new lensvector*[nthreads];
	defs_i = new lensvector[nthreads];
	jacs = new lensmatrix[nthreads];
	hesses = new lensmatrix[nthreads];
	hesses_subtot = new lensmatrix*[nthreads];
	Amats_i = new lensmatrix[nthreads];
	hesses_i = new lensmatrix[nthreads];
	for (int i=0; i < nthreads; i++) {
		defs_subtot[i] = new lensvector[nmax_lens_planes];
		hesses_subtot[i] = new lensmatrix[nmax_lens_planes];
	}
}

void QLens::deallocate_multithreaded_variables()
{
	if (xvals_i != NULL) {
		delete[] xvals_i;
		delete[] defs;
		delete[] defs_i;
		delete[] jacs;
		delete[] hesses;
		delete[] hesses_i;
		delete[] Amats_i;
		for (int i=0; i < nthreads; i++) {
			delete[] defs_subtot[i];
			delete[] hesses_subtot[i];
		}
		delete[] defs_subtot;
		delete[] hesses_subtot;

		xvals_i = NULL;
		defs = NULL;
		defs_i = NULL;
		jacs = NULL;
		hesses = NULL;
		hesses_i = NULL;
		Amats_i = NULL;
		defs_subtot = NULL;
		hesses_subtot = NULL;
	}
}

#ifdef USE_MUMPS
DMUMPS_STRUC_C *QLens::mumps_solver;

void QLens::setup_mumps()
{
	mumps_solver = new DMUMPS_STRUC_C;
	mumps_solver->par = 1; // this tells MUMPS that the host machine participates in calculation
}
#endif

void QLens::delete_mumps()
{
#ifdef USE_MUMPS
	delete mumps_solver;
#endif
}

#ifdef USE_MPI
void QLens::set_mpi_params(const int& mpi_id_in, const int& mpi_np_in, const int& mpi_ngroups_in, const int& group_num_in, const int& group_id_in, const int& group_np_in, int* group_leader_in, MPI_Group* group_in, MPI_Comm* comm, MPI_Group* mygroup, MPI_Comm* mycomm)
{
	mpi_id = mpi_id_in;
	mpi_np = mpi_np_in;
	mpi_ngroups = mpi_ngroups_in;
	group_id = group_id_in;
	group_num = group_num_in;
	group_np = group_np_in;
	group_leader = new int[mpi_ngroups];
	for (int i=0; i < mpi_ngroups; i++) group_leader[i] = group_leader_in[i];
	mpi_group = group_in;
	group_comm = comm;
	my_group = mygroup;
	my_comm = mycomm;
#ifdef USE_MUMPS
	setup_mumps();
#endif
}
#endif

void QLens::set_mpi_params(const int& mpi_id_in, const int& mpi_np_in)
{
	// This assumes only one 'group', so all MPI processes will work together for each likelihood evaluation
	mpi_id = mpi_id_in;
	mpi_np = mpi_np_in;
	mpi_ngroups = 1;
	group_id = mpi_id;
	group_num = 0;
	group_np = mpi_np;
	group_leader = NULL;

#ifdef USE_MPI
	MPI_Comm_group(MPI_COMM_WORLD, mpi_group);
	MPI_Comm_create(MPI_COMM_WORLD, *mpi_group, group_comm);
#ifdef USE_MUMPS
	setup_mumps();
#endif
#endif
}

QLens::QLens() : UCMC()
{
	lens_parent = NULL; // this is only set if creating from another lens
	mpi_id = 0;
	mpi_np = 1;
	group_np = 1;
	group_id = 0;
	group_num = 0;
	mpi_ngroups = 1;
	group_leader = NULL;
#ifdef USE_MPI
	mpi_group = NULL;
#endif

	int threads = 1;
#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		threads = omp_get_num_threads();
	}
#endif

	allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)
	hubble = 0.7;
	omega_matter = 0.3;
	set_cosmology(omega_matter,0.04,hubble,2.215);
	lens_redshift = 0.5;
	source_redshift = 2.0;
	syserr_pos = 0.0;
	wl_shear_factor = 1.0;
	user_changed_zsource = false; // keeps track of whether redshift has been manually changed; if so, then don't change it to redshift from data
	auto_zsource_scaling = true; // this automatically sets the reference source redshift (for kappa scaling) equal to the source redshift being used
	reference_source_redshift = 2.0; // this is the source redshift with respect to which the lens models are defined
	reference_zfactors = NULL;
	default_zsrc_beta_factors = NULL;
	source_redshifts = NULL;
	zfactors = NULL;
	beta_factors = NULL;
	lens_redshifts = NULL;
	lens_redshift_idx = NULL;
	zlens_group_size = NULL;
	zlens_group_lens_indx = NULL;
	n_lens_redshifts = 0;

	vary_hubble_parameter = false;
	hubble_lower_limit = 1e30; // These must be specified by user
	hubble_upper_limit = 1e30; // These must be specified by user

	vary_omega_matter_parameter = false;
	omega_matter_lower_limit = 1e30; // These must be specified by user
	omega_matter_upper_limit = 1e30; // These must be specified by user

	vary_syserr_pos_parameter = false;
	syserr_pos_lower_limit = 1e30; // These must be specified by user
	syserr_pos_upper_limit = 1e30; // These must be specified by user

	vary_wl_shear_factor_parameter = false;
	wl_shear_factor_lower_limit = 1e30; // These must be specified by user
	wl_shear_factor_upper_limit = 1e30; // These must be specified by user

	chisq_it=0;
	raw_chisq = -1e30;
	calculate_bayes_factor = false;
	reference_lnZ = -1e30; // used to calculate Bayes factors when two different models are run
	chisq_diagnostic = false;
	chisq_bestfit = 1e30;
	bestfit_flux = 0;
	display_chisq_status = false;
	chisq_display_frequency = 100; // Number of chi-square evaluations before displaying chi-square on screen
	show_wtime = false;
	terminal = TEXT;
	suppress_plots = false;
	verbal_mode = true;
	n_infiles = 0;
	infile = infile_list;
	quit_after_reading_file = false;
	quit_after_error = false;
	fitmethod = SIMPLEX;
	fit_output_dir = ".";
	auto_fit_output_dir = true; // name the output directory "chains_<label>" unless manually specified otherwise
	simplex_nmax = 10000;
	simplex_nmax_anneal = 1000;
	simplex_temp_initial = 0; // no simulated annealing by default
	simplex_temp_final = 1;
	simplex_cooling_factor = 0.9; // temperature decrement (multiplicative) for annealing schedule
	simplex_minchisq = -1e30;
	simplex_minchisq_anneal = -1e30;
	simplex_show_bestfit = false;
	n_livepts = 1000; // for nested sampling
	polychord_nrepeats = 5;
	mcmc_threads = 1;
	mcmc_tolerance = 1.01; // Gelman-Rubin statistic for T-Walk sampler
	mcmc_logfile = false;
	open_chisq_logfile = false;
	psf_convolution_mpi = false;
	use_input_psf_matrix = false;
	psf_threshold = 1e-1;
	n_image_prior = false;
	n_image_threshold = 1.5; // ************THIS SHOULD BE SPECIFIED BY THE USER, AND ONLY GETS USED IF n_image_prior IS SET TO 'TRUE'
	n_image_prior_sb_frac = 0.25; // ********ALSO SHOULD BE SPECIFIED BY THE USER, AND ONLY GETS USED IF n_image_prior IS SET TO 'TRUE'
	outside_sb_prior = false;
	outside_sb_prior_noise_frac = 1e10; // surface brightness threshold is given as multiple of data pixel noise (1e10 by default so it's effectively not used)
	outside_sb_prior_threshold = 0.3; // surface brightness threshold is given as fraction of max surface brightness
	einstein_radius_prior = false;
	einstein_radius_low_threshold = 0;
	einstein_radius_high_threshold = 1000;
	extended_mask_n_neighbors = -1;
	high_sn_frac = 0.5; // fraction of max SB; used to determine optimal source pixel size based on area the high S/N pixels cover when mapped to source plane
	subhalo_prior = false; // if on, this prior constrains any subhalos (with Pseudo-Jaffe profiles) to be positioned within the designated fit area (selected fit pixels only)
	use_custom_prior = false;
	nlens = 0;
	n_sb = 0;
	sbmin = -1e30;
	sbmax = 1e30;
	n_derived_params = 0;
	radial_grid = true;
	grid_xlength = 20; // default gridsize
	grid_ylength = 20;
	grid_xcenter = 0;
	grid_ycenter = 0;
	rmin_frac = default_rmin_frac;
	plot_ptsize = 1.2;
	plot_pttype = 7;

	fit_output_filename = "fit";
	auto_save_bestfit = false;
	fitmodel = NULL;
#ifdef USE_FITS
	fits_format = true;
#else
	fits_format = false;
#endif
	data_pixel_size = -1; // used for setting a pixel scale for FITS images (only if initialized to a positive number)
	n_fit_parameters = 0;
	n_sourcepts_fit = 0;
	//sourcepts_fit = NULL;
	//sourcepts_lower_limit = NULL;
	//sourcepts_upper_limit = NULL;
	//vary_sourcepts_x = NULL;
	//vary_sourcepts_y = NULL;
	borrowed_image_data = false;
	image_data = NULL;
	defspline = NULL;

	source_fit_mode = Point_Source;
	use_ansi_characters = false;
	chisq_tolerance = 1e-3;
	chisq_magnification_threshold = 0;
	chisq_imgsep_threshold = 0;
	chisq_imgplane_substitute_threshold = -1; // if > 0, will evaluate the source plane chi-square and if above the threshold, use instead of image plane chi-square (if imgplane_chisq is on)
	n_repeats = 1;
	calculate_parameter_errors = true;
	imgplane_chisq = false;
	use_magnification_in_chisq = true;
	use_magnification_in_chisq_during_repeats = true;
	include_central_image = true;
	include_imgpos_chisq = false;
	include_flux_chisq = false;
	include_weak_lensing_chisq = false;
	include_parity_in_chisq = false;
	include_time_delay_chisq = false;
	use_analytic_bestfit_src = false;
	n_images_penalty = false;
	fix_source_flux = false;
	source_flux = 1.0;
	param_settings = new ParamSettings;
	sim_err_pos = 0.005;
	sim_err_flux = 0.01;
	sim_err_td = 1;
	sim_err_shear = 0.1;

	image_pixel_data = NULL;
	image_pixel_grid = NULL;
	source_pixel_grid = NULL;
	sourcegrid_xmin = -1;
	sourcegrid_xmax = 1;
	sourcegrid_ymin = -1;
	sourcegrid_ymax = 1;
	sourcegrid_limit_xmin = -1e30;
	sourcegrid_limit_xmax = 1e30;
	sourcegrid_limit_ymin = -1e30;
	sourcegrid_limit_ymax = 1e30;
	auto_sourcegrid = true;
	auto_shapelet_scaling = true;
	auto_shapelet_center = true;
	shapelet_scale_mode = 0;
	ray_tracing_method = Interpolate;
	weight_interpolation_by_imgplane_area = true;
	interpolate_sb_3pt = true; // if false, will not use 3-point interpolation even when ray tracing method is set to "interpolate"
#ifdef USE_MUMPS
	inversion_method = MUMPS;
#else
#ifdef USE_UMFPACK
	inversion_method = UMFPACK;
#else
	inversion_method = CG_Method;
#endif
#endif
	parallel_mumps = false;
	show_mumps_info = false;
	regularization_method = Curvature;
	regularization_parameter = 0.5;
	regularization_parameter_lower_limit = 1e30; // These must be specified by user
	regularization_parameter_upper_limit = 1e30; // These must be specified by user
	vary_regularization_parameter = false;
	optimize_regparam = false;
	optimize_regparam_tol = 0.01; // this is the tolerance on log(regparam)
	optimize_regparam_minlog = -1;
	optimize_regparam_maxlog = 5;
	psf_width_x = 0;
	psf_width_y = 0;
	data_pixel_noise = 0;
	sim_pixel_noise = 0;
	sb_threshold = 0;
	noise_threshold = 0; // when optimizing the source pixel grid size, image pixels whose surface brightness < noise_threshold*pixel_noise are ignored
	n_image_pixels_x = 200;
	n_image_pixels_y = 200;
	srcgrid_npixels_x = 50;
	srcgrid_npixels_y = 50;
	auto_srcgrid_npixels = true;
	auto_srcgrid_set_pixel_size = false; // this feature is not working at the moment, so keep it off
	pixel_fraction = 0.3; // this should not be used if adaptive grid is being used
	pixel_fraction_lower_limit = 1e30; // These must be specified by user
	pixel_fraction_upper_limit = 1e30; // These must be specified by user
	vary_pixel_fraction = false; // varying the pixel fraction doesn't work if regularization is also varied (with source pixel regularization)
	srcgrid_xshift = 0;
	srcgrid_xshift_lower_limit = 1e30;
	srcgrid_xshift_upper_limit = 1e30;
	vary_srcgrid_xshift = false;
	srcgrid_yshift = 0;
	srcgrid_yshift_lower_limit = 1e30;
	srcgrid_yshift_upper_limit = 1e30;
	vary_srcgrid_yshift = false;
	srcgrid_size_scale = 0;
	srcgrid_size_scale_lower_limit = 1e30;
	srcgrid_size_scale_upper_limit = 1e30;
	vary_srcgrid_size_scale = false;
	Fmatrix = NULL;
	Fmatrix_index = NULL;
	Rmatrix = NULL;
	Rmatrix_index = NULL;
	Dvector = NULL;
	image_surface_brightness = NULL;
	foreground_surface_brightness = NULL;
	source_pixel_vector = NULL;
	source_pixel_n_images = NULL;
	active_image_pixel_i = NULL;
	active_image_pixel_j = NULL;
	image_pixel_location_Lmatrix = NULL;
	source_pixel_location_Lmatrix = NULL;
	Lmatrix = NULL;
	Lmatrix_index = NULL;
	psf_matrix = NULL;
	inversion_nthreads = 1;
	adaptive_grid = false;
	pixel_magnification_threshold = 7;
	pixel_magnification_threshold_lower_limit = 1e30; // These must be specified by user
	pixel_magnification_threshold_upper_limit = 1e30; // These must be specified by user
	base_srcpixel_imgpixel_ratio = 0.8; // for lowest mag source pixel, this sets fraction of image pixel area covered by it (when mapped to image plane)
	vary_magnification_threshold = false;
	exclude_source_pixels_beyond_fit_window = true;
	activate_unmapped_source_pixels = true;
	regrid_if_unmapped_source_subpixels = false;
	default_imgpixel_nsplit = 2;
	split_imgpixels = true;

	use_cc_spline = false;
	auto_ccspline = false;
	plot_critical_curves = &QLens::plot_sorted_critical_curves;
	cc_rmin = default_autogrid_rmin;
	cc_rmax = default_autogrid_rmax;
	source_plane_rscale = 0; // this must be found by the autogrid
	autogrid_frac = default_autogrid_frac;

	// the following variables are only relevant if use_cc_spline is on (so critical curves are splined)
	cc_thetasteps = 200;
	cc_splined = false;
	effectively_spherical = false;

	// parameters for the recursive grid
	enforce_min_cell_area = true; // this is option is obsolete, and should be removed (we should always enforce a min cell area!!!!)
	min_cell_area = 1e-4;
	usplit_initial = 16; // initial number of cell divisions in the r-direction
	wsplit_initial = 24; // initial number of cell divisions in the theta-direction
	splitlevels = 0; // number of times grid squares are recursively split (by default)...setting to zero is best, recursion slows down grid creation & searching
	cc_splitlevels = 2; // number of times grid squares are recursively split when containing a critical curve
	cc_neighbor_splittings = false;
	use_perturber_flags = false;
	subgrid_around_perturbers = true;
	subgrid_only_near_data_images = false; // if on, only subgrids around perturber galaxies (during fit) if a data image is within the determined subgridding radius (dangerous if not all images are observed!)
	galsubgrid_radius_fraction = 1.3;
	galsubgrid_min_cellsize_fraction = 0.25;
	galsubgrid_cc_splittings = 1;
	sorted_critical_curves = false;
	n_singular_points = 0;
	auto_store_cc_points = true;
	newton_magnification_threshold = 1000;
	reject_himag_images = true;
	reject_images_found_outside_cell = false;
	redundancy_separation_threshold = 1e-5;

	warnings = true;
	newton_warnings = false;
	set_sci_notation(true);
	use_ansi_output_during_fit = true;
	include_time_delays = false;
	autocenter = true; // this option tells qlens to center the grid on a particular lens (given by primary_lens_number)
	auto_gridsize_from_einstein_radius = true; // this option tells qlens to set the grid size based on the Einstein radius of a particular lens (given by primary_lens_number)
	autogrid_before_grid_creation = false; // this option (if set to true) tells qlens to optimize the grid size & position automatically (using autogrid) when grid is created
	primary_lens_number = 0;
	auto_set_primary_lens = true;
	include_secondary_lens = false; // turn on to use an additional secondary lens to set the grid size (useful if modeling DM halo + BCG)
	secondary_lens_number = 1;
	spline_frac = 1.8;
	tabulate_rmin = 1e-3;
	tabulate_qmin = 0.2;
	tabulate_logr_N = 2000;
	tabulate_phi_N = 200;
	tabulate_q_N = 10;
	grid = NULL;
	Gauss_NN = 1024;
	integral_tolerance = 5e-3;
	default_parameter_mode = 0;
	include_recursive_lensing = true;
	use_mumps_subcomm = true; // this option should probably be removed, but keeping it for now in case a problem with sub_comm turns up
	DerivedParamPtr = static_cast<void (UCMC::*)(double*,double*)> (&QLens::fitmodel_calculate_derived_params);
}

QLens::QLens(QLens *lens_in) : UCMC() // creates lens object with same settings as input lens; does NOT import the lens/source model configurations, however
{
	lens_parent = lens_in;
	verbal_mode = lens_in->verbal_mode;
	set_random_generator(lens_in);
	chisq_it=0;
	raw_chisq = -1e30;
	calculate_bayes_factor = lens_in->calculate_bayes_factor;
	reference_lnZ = lens_in->reference_lnZ; // used to calculate Bayes factors when two different models are run
	chisq_diagnostic = lens_in->chisq_diagnostic;
	chisq_bestfit = lens_in->chisq_bestfit;
	bestfit_flux = lens_in->bestfit_flux;
	display_chisq_status = lens_in->display_chisq_status;
	chisq_display_frequency = lens_in->chisq_display_frequency; // Number of chi-square evaluations before displaying chi-square on screen
	mpi_id = lens_in->mpi_id;
	mpi_np = lens_in->mpi_np;
	mpi_ngroups = lens_in->mpi_ngroups;
	group_id = lens_in->group_id;
	group_num = lens_in->group_num;
	group_np = lens_in->group_np;
	group_leader = lens_in->group_leader;
	if (lens_in->group_leader==NULL) group_leader = NULL;
	else {
		group_leader = new int[mpi_ngroups];
		for (int i=0; i < mpi_ngroups; i++) group_leader[i] = lens_in->group_leader[i];
	}
#ifdef USE_MPI
	group_comm = lens_in->group_comm;
	mpi_group = lens_in->mpi_group;
	my_comm = lens_in->my_comm;
	my_group = lens_in->my_group;
#endif

	hubble = lens_in->hubble;
	omega_matter = lens_in->omega_matter;
	syserr_pos = lens_in->syserr_pos;
	wl_shear_factor = lens_in->wl_shear_factor;
	set_cosmology(omega_matter,0.04,hubble,2.215);
	lens_redshift = lens_in->lens_redshift;
	source_redshift = lens_in->source_redshift;
	user_changed_zsource = lens_in->user_changed_zsource; // keeps track of whether redshift has been manually changed; if so, then don't change it to redshift from data
	auto_zsource_scaling = lens_in->auto_zsource_scaling;
	reference_source_redshift = lens_in->reference_source_redshift; // this is the source redshift with respect to which the lens models are defined
	// Dynamically allocated arrays like the ones below should probably just be replaced with container classes, as long as they don't slow
	// down the code significantly (check!)  It would make the code less bug-prone. On the other hand, if no one ever has to look at or mess with the code, then who cares?
	reference_zfactors = NULL; // this is the scaling for lensing quantities if the source redshift is different from the reference value
	default_zsrc_beta_factors = NULL; // this is the scaling for lensing quantities if the source redshift is different from the reference value
	source_redshifts = NULL;
	zfactors = NULL;
	beta_factors = NULL;
	lens_redshifts = NULL;
	n_lens_redshifts = lens_in->n_lens_redshifts;
	lens_redshift_idx = NULL;
	zlens_group_size = NULL;
	zlens_group_lens_indx = NULL;

	vary_hubble_parameter = lens_in->vary_hubble_parameter;
	hubble_lower_limit = lens_in->hubble_lower_limit; // These must be specified by user
	hubble_upper_limit = lens_in->hubble_upper_limit; // These must be specified by user

	vary_omega_matter_parameter = lens_in->vary_omega_matter_parameter;
	omega_matter_lower_limit = lens_in->omega_matter_lower_limit; // These must be specified by user
	omega_matter_upper_limit = lens_in->omega_matter_upper_limit; // These must be specified by user

	vary_syserr_pos_parameter = lens_in->vary_syserr_pos_parameter;
	syserr_pos_lower_limit = lens_in->syserr_pos_lower_limit; // These must be specified by user
	syserr_pos_upper_limit = lens_in->syserr_pos_upper_limit; // These must be specified by user

	vary_wl_shear_factor_parameter = lens_in->vary_wl_shear_factor_parameter;
	wl_shear_factor_lower_limit = lens_in->wl_shear_factor_lower_limit; // These must be specified by user
	wl_shear_factor_upper_limit = lens_in->wl_shear_factor_upper_limit; // These must be specified by user

	terminal = lens_in->terminal;
	show_wtime = lens_in->show_wtime;
	fit_output_dir = lens_in->fit_output_dir;
	auto_fit_output_dir = lens_in->auto_fit_output_dir;
	auto_save_bestfit = lens_in->auto_save_bestfit;
	fitmethod = lens_in->fitmethod;
	mcmc_threads = lens_in->mcmc_threads;
	simplex_nmax = lens_in->simplex_nmax;
	simplex_nmax_anneal = lens_in->simplex_nmax_anneal;
	simplex_temp_initial = lens_in->simplex_temp_initial;
	simplex_temp_final = lens_in->simplex_temp_final;
	simplex_cooling_factor = lens_in->simplex_cooling_factor; // temperature decrement (multiplicative) for annealing schedule
	simplex_minchisq = lens_in->simplex_minchisq;
	simplex_minchisq_anneal = lens_in->simplex_minchisq_anneal;
	simplex_show_bestfit = lens_in->simplex_show_bestfit;
	n_livepts = lens_in->n_livepts; // for nested sampling
	polychord_nrepeats = lens_in->polychord_nrepeats;
	mcmc_tolerance = lens_in->mcmc_tolerance; // for T-Walk sampler
	mcmc_logfile = lens_in->mcmc_logfile;
	open_chisq_logfile = lens_in->open_chisq_logfile;
	psf_convolution_mpi = lens_in->psf_convolution_mpi;
	use_input_psf_matrix = lens_in->use_input_psf_matrix;
	psf_threshold = lens_in->psf_threshold;
	n_image_prior = lens_in->n_image_prior;
	n_image_threshold = lens_in->n_image_threshold;
	n_image_prior_sb_frac = lens_in->n_image_prior_sb_frac;
	outside_sb_prior = lens_in->outside_sb_prior;
	outside_sb_prior_noise_frac = lens_in->outside_sb_prior_noise_frac; // surface brightness threshold is given as multiple of data pixel noise
	outside_sb_prior_threshold = lens_in->outside_sb_prior_threshold; // surface brightness threshold is given as fraction of max surface brightness
	einstein_radius_prior = lens_in->einstein_radius_prior;
	einstein_radius_low_threshold = lens_in->einstein_radius_low_threshold;
	einstein_radius_high_threshold = lens_in->einstein_radius_high_threshold;
	extended_mask_n_neighbors = lens_in->extended_mask_n_neighbors;

	high_sn_frac = lens_in->high_sn_frac; // fraction of max SB; used to determine optimal source pixel size based on area the high S/N pixels cover when mapped to source plane
	subhalo_prior = lens_in->subhalo_prior;
	use_custom_prior = lens_in->use_custom_prior;

	plot_ptsize = lens_in->plot_ptsize;
	plot_pttype = lens_in->plot_pttype;

	nlens = 0;
	n_sb = 0;
	n_derived_params = 0;
	radial_grid = lens_in->radial_grid;
	grid_xlength = lens_in->grid_xlength; // default gridsize
	grid_ylength = lens_in->grid_ylength;
	grid_xcenter = lens_in->grid_xcenter;
	grid_ycenter = lens_in->grid_ycenter;

	LogLikePtr = static_cast<double (UCMC::*)(double *)> (&QLens::fitmodel_loglike_point_source); // unnecessary, but just in case
	source_fit_mode = lens_in->source_fit_mode;
	use_ansi_characters = lens_in->use_ansi_characters;
	chisq_tolerance = lens_in->chisq_tolerance;
	chisq_magnification_threshold = lens_in->chisq_magnification_threshold;
	chisq_imgsep_threshold = lens_in->chisq_imgsep_threshold;
	chisq_imgplane_substitute_threshold = lens_in->chisq_imgplane_substitute_threshold;
	n_repeats = lens_in->n_repeats;
	calculate_parameter_errors = lens_in->calculate_parameter_errors;
	imgplane_chisq = lens_in->imgplane_chisq;
	use_magnification_in_chisq = lens_in->use_magnification_in_chisq;
	use_magnification_in_chisq_during_repeats = lens_in->use_magnification_in_chisq_during_repeats;
	include_central_image = lens_in->include_central_image;
	include_imgpos_chisq = lens_in->include_imgpos_chisq;
	include_flux_chisq = lens_in->include_flux_chisq;
	include_weak_lensing_chisq = lens_in->include_weak_lensing_chisq;
	include_parity_in_chisq = lens_in->include_parity_in_chisq;
	include_time_delay_chisq = lens_in->include_time_delay_chisq;
	use_analytic_bestfit_src = lens_in->use_analytic_bestfit_src;
	n_images_penalty = lens_in->n_images_penalty;
	fix_source_flux = lens_in->fix_source_flux;
	source_flux = lens_in->source_flux;
	param_settings = new ParamSettings(*lens_in->param_settings);
	sim_err_pos = lens_in->sim_err_pos;
	sim_err_flux = lens_in->sim_err_flux;
	sim_err_td = lens_in->sim_err_td;
	sim_err_shear = lens_in->sim_err_shear;

	fitmodel = NULL;
	fits_format = lens_in->fits_format;
	data_pixel_size = lens_in->data_pixel_size;
	n_fit_parameters = 0;
	n_sourcepts_fit = 0;
	//sourcepts_fit = NULL;
	//sourcepts_lower_limit = NULL;
	//sourcepts_upper_limit = NULL;
	//vary_sourcepts_x = NULL;
	//vary_sourcepts_y = NULL;
	borrowed_image_data = false;
	image_data = NULL;
	weak_lensing_data.input(lens_in->weak_lensing_data);
	defspline = NULL;

	image_pixel_data = NULL;
	image_pixel_grid = NULL;
	source_pixel_grid = NULL;
	sourcegrid_xmin = lens_in->sourcegrid_xmin;
	sourcegrid_xmax = lens_in->sourcegrid_xmax;
	sourcegrid_ymin = lens_in->sourcegrid_ymin;
	sourcegrid_ymax = lens_in->sourcegrid_ymax;
	sourcegrid_limit_xmin = lens_in->sourcegrid_limit_xmin;
	sourcegrid_limit_xmax = lens_in->sourcegrid_limit_xmax;
	sourcegrid_limit_ymin = lens_in->sourcegrid_limit_ymin;
	sourcegrid_limit_ymax = lens_in->sourcegrid_limit_ymax;
	auto_sourcegrid = lens_in->auto_sourcegrid;
	auto_shapelet_scaling = lens_in->auto_shapelet_scaling;
	auto_shapelet_center = lens_in->auto_shapelet_center;
	shapelet_scale_mode = lens_in->shapelet_scale_mode;
	weight_interpolation_by_imgplane_area = lens_in->weight_interpolation_by_imgplane_area;
	interpolate_sb_3pt = lens_in->interpolate_sb_3pt;

	regularization_method = lens_in->regularization_method;
	regularization_parameter = lens_in->regularization_parameter;
	regularization_parameter_lower_limit = lens_in->regularization_parameter_lower_limit;
	regularization_parameter_upper_limit = lens_in->regularization_parameter_upper_limit;
	vary_regularization_parameter = lens_in->vary_regularization_parameter;
	optimize_regparam = lens_in->optimize_regparam;
	optimize_regparam_tol = lens_in->optimize_regparam_tol; // this is the tolerance on log(regparam)
	optimize_regparam_minlog = lens_in->optimize_regparam_minlog;
	optimize_regparam_maxlog = lens_in->optimize_regparam_maxlog;

	ray_tracing_method = lens_in->ray_tracing_method;
	inversion_method = lens_in->inversion_method;
	parallel_mumps = lens_in->parallel_mumps;
	show_mumps_info = lens_in->show_mumps_info;

	psf_width_x = lens_in->psf_width_x;
	psf_width_y = lens_in->psf_width_y;
	data_pixel_noise = lens_in->data_pixel_noise;
	sim_pixel_noise = 0; // the fit model should never add random noise when generating lensed images
	sb_threshold = lens_in->sb_threshold;
	noise_threshold = lens_in->noise_threshold;
	n_image_pixels_x = lens_in->n_image_pixels_x;
	n_image_pixels_y = lens_in->n_image_pixels_y;
	n_image_pixels_x = lens_in->n_image_pixels_x;
	n_image_pixels_y = lens_in->n_image_pixels_y;
	srcgrid_npixels_x = lens_in->srcgrid_npixels_x;
	srcgrid_npixels_y = lens_in->srcgrid_npixels_y;
	auto_srcgrid_npixels = lens_in->auto_srcgrid_npixels;
	auto_srcgrid_set_pixel_size = lens_in->auto_srcgrid_set_pixel_size;

	pixel_fraction = lens_in->pixel_fraction;
	vary_pixel_fraction = lens_in->vary_pixel_fraction;
	srcgrid_xshift = lens_in->srcgrid_xshift;
	vary_srcgrid_xshift = lens_in->vary_srcgrid_xshift;
	srcgrid_yshift = lens_in->srcgrid_yshift;
	vary_srcgrid_yshift = lens_in->vary_srcgrid_yshift;
	srcgrid_size_scale = lens_in->srcgrid_size_scale;
	vary_srcgrid_size_scale = lens_in->vary_srcgrid_size_scale;
	Dvector = NULL;
	Fmatrix = NULL;
	Fmatrix_index = NULL;
	Rmatrix = NULL;
	Rmatrix_index = NULL;
	image_surface_brightness = NULL;
	foreground_surface_brightness = NULL;
	source_pixel_vector = NULL;
	source_pixel_n_images = NULL;
	active_image_pixel_i = NULL;
	active_image_pixel_j = NULL;
	Lmatrix_index = NULL;
	if (lens_in->psf_matrix==NULL) psf_matrix = NULL;
	else {
		psf_npixels_x = lens_in->psf_npixels_x;
		psf_npixels_y = lens_in->psf_npixels_y;
		psf_matrix = new double*[psf_npixels_x];
		int i,j;
		for (i=0; i < psf_npixels_x; i++) {
			psf_matrix[i] = new double[psf_npixels_y];
			for (j=0; j < psf_npixels_y; j++) psf_matrix[i][j] = lens_in->psf_matrix[i][j];
		}
	}
	image_pixel_location_Lmatrix = NULL;
	source_pixel_location_Lmatrix = NULL;
	Lmatrix = NULL;
	inversion_nthreads = lens_in->inversion_nthreads;
	adaptive_grid = lens_in->adaptive_grid;
	pixel_magnification_threshold = lens_in->pixel_magnification_threshold;
	pixel_magnification_threshold_lower_limit = lens_in->pixel_magnification_threshold_lower_limit;
	pixel_magnification_threshold_upper_limit = lens_in->pixel_magnification_threshold_upper_limit;
	vary_magnification_threshold = lens_in->vary_magnification_threshold;
	base_srcpixel_imgpixel_ratio = lens_in->base_srcpixel_imgpixel_ratio; // for lowest mag source pixel, this sets fraction of image pixel area covered by it (when mapped to image plane)
	exclude_source_pixels_beyond_fit_window = lens_in->exclude_source_pixels_beyond_fit_window;
	activate_unmapped_source_pixels = lens_in->activate_unmapped_source_pixels;
	regrid_if_unmapped_source_subpixels = lens_in->regrid_if_unmapped_source_subpixels;
	default_imgpixel_nsplit = lens_in->default_imgpixel_nsplit;
	split_imgpixels = lens_in->split_imgpixels;

	use_cc_spline = lens_in->use_cc_spline;
	auto_ccspline = lens_in->auto_ccspline;
	plot_critical_curves = &QLens::plot_sorted_critical_curves;
	cc_rmin = lens_in->cc_rmin;
	cc_rmax = lens_in->cc_rmax;
	source_plane_rscale = lens_in->source_plane_rscale;
	autogrid_frac = lens_in->autogrid_frac;

	// the following variables are only relevant if use_cc_spline is on (so critical curves are splined)
	cc_thetasteps = lens_in->cc_thetasteps;
	cc_splined = false;
	effectively_spherical = false;

	// parameters for the recursive grid
	enforce_min_cell_area = lens_in->enforce_min_cell_area;
	min_cell_area = lens_in->min_cell_area;
	usplit_initial = lens_in->usplit_initial; // initial number of cell divisions in the r-direction
	wsplit_initial = lens_in->wsplit_initial; // initial number of cell divisions in the theta-direction
	splitlevels = lens_in->splitlevels; // number of times grid squares are recursively split (by default)...minimum of one splitting is required
	cc_splitlevels = lens_in->cc_splitlevels; // number of times grid squares are recursively split when containing a critical curve
	cc_neighbor_splittings = lens_in->cc_neighbor_splittings;
	use_perturber_flags = lens_in->use_perturber_flags;
	subgrid_around_perturbers = lens_in->subgrid_around_perturbers;
	subgrid_only_near_data_images = lens_in->subgrid_only_near_data_images; // if on, only subgrids around perturber galaxies if a data image is within the determined subgridding radius
	galsubgrid_radius_fraction = lens_in->galsubgrid_radius_fraction;
	galsubgrid_min_cellsize_fraction = lens_in->galsubgrid_min_cellsize_fraction;
	galsubgrid_cc_splittings = lens_in->galsubgrid_cc_splittings;
	sorted_critical_curves = false;
	auto_store_cc_points = lens_in->auto_store_cc_points;
	n_singular_points = lens_in->n_singular_points;
	newton_magnification_threshold = lens_in->newton_magnification_threshold;
	reject_himag_images = lens_in->reject_himag_images;
	reject_images_found_outside_cell = lens_in->reject_images_found_outside_cell;
	redundancy_separation_threshold = lens_in->redundancy_separation_threshold;

	include_time_delays = lens_in->include_time_delays;
	autocenter = lens_in->autocenter;
	primary_lens_number = lens_in->primary_lens_number;
	auto_set_primary_lens = lens_in->auto_set_primary_lens;
	include_secondary_lens = lens_in->include_secondary_lens; // turn on to use an additional secondary lens to set the grid size (useful if modeling DM halo + BCG)
	secondary_lens_number = lens_in->secondary_lens_number;

	auto_gridsize_from_einstein_radius = lens_in->auto_gridsize_from_einstein_radius;
	autogrid_before_grid_creation = lens_in->autogrid_before_grid_creation; // this option (if set to true) tells qlens to optimize the grid size & position automatically when grid is created
	default_parameter_mode = lens_in->default_parameter_mode;
	spline_frac = lens_in->spline_frac;
	tabulate_rmin = lens_in->tabulate_rmin;
	tabulate_qmin = lens_in->tabulate_qmin;
	tabulate_logr_N = lens_in->tabulate_logr_N;
	tabulate_phi_N = lens_in->tabulate_phi_N;
	tabulate_q_N = lens_in->tabulate_q_N;

	grid = NULL;
	Gauss_NN = lens_in->Gauss_NN;
	integral_tolerance = lens_in->integral_tolerance;
	include_recursive_lensing = lens_in->include_recursive_lensing;
	use_mumps_subcomm = lens_in->use_mumps_subcomm;
}

void QLens::kappa_inverse_mag_sourcept(const lensvector& xvec, lensvector& srcpt, double &kap_tot, double &invmag, const int &thread, double* zfacs, double** betafacs)
{
	//cout << "CHECK " << zfacs[0] << " " << betafacs[0][0] << endl;
	double x = xvec[0], y = xvec[1];
	lensmatrix *jac = &jacs[thread];
	lensvector *def_tot = &defs[thread];

	if (!defspline)
	{
		if (n_lens_redshifts > 1) {
			lensvector *x_i = &xvals_i[thread];
			lensmatrix *A_i = &Amats_i[thread];
			lensvector *def = &defs_i[thread];
			lensvector **def_i = &defs_subtot[thread];
			lensmatrix *hess = &hesses_i[thread];
			lensmatrix **hess_i = &hesses_subtot[thread];

			int i,j;
			(*jac)[0][0] = 0;
			(*jac)[1][1] = 0;
			(*jac)[0][1] = 0;
			(*jac)[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			for (i=0; i < n_lens_redshifts; i++) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
					(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
					(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
					(*A_i)[0][0] -= (betafacs[i-1][j])*((*hess_i)[j])[0][0];
					(*A_i)[1][1] -= (betafacs[i-1][j])*((*hess_i)[j])[1][1];
					(*A_i)[1][0] -= (betafacs[i-1][j])*((*hess_i)[j])[1][0];
					(*A_i)[0][1] -= (betafacs[i-1][j])*((*hess_i)[j])[0][1];
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					lens_list[zlens_group_lens_indx[i][j]]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess));
					(*hess_i)[i][0][0] += (*hess)[0][0];
					(*hess_i)[i][1][1] += (*hess)[1][1];
					(*hess_i)[i][0][1] += (*hess)[0][1];
					(*hess_i)[i][1][0] += (*hess)[1][0];
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
				(*def_i)[i][0] *= zfacs[i];
				(*def_i)[i][1] *= zfacs[i];
				(*def_tot)[0] += (*def_i)[i][0];
				(*def_tot)[1] += (*def_i)[i][1];

				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];

				(*jac)[0][0] += (*hess_i)[i][0][0];
				(*jac)[1][1] += (*hess_i)[i][1][1];
				(*jac)[1][0] += (*hess_i)[i][1][0];
				(*jac)[0][1] += (*hess_i)[i][0][1];
			}
			kap_tot = ((*jac)[0][0] + (*jac)[1][1])/2;
		} else {
			(*jac)[0][0] = 0;
			(*jac)[1][1] = 0;
			(*jac)[0][1] = 0;
			(*jac)[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			kap_tot = 0;

			if (nthreads==1) {
				int j;
				double kap;
				(*jac)[0][0] = 0;
				(*jac)[1][1] = 0;
				(*jac)[0][1] = 0;
				(*jac)[1][0] = 0;
				(*def_tot)[0] = 0;
				(*def_tot)[1] = 0;
				kap_tot = 0;
				lensvector *def = &defs_i[0];
				lensmatrix *hess = &hesses_i[0];
				for (j=0; j < nlens; j++) {
					lens_list[j]->kappa_and_potential_derivatives(x,y,kap,(*def),(*hess));
					(*jac)[0][0] += (*hess)[0][0];
					(*jac)[1][1] += (*hess)[1][1];
					(*jac)[0][1] += (*hess)[0][1];
					(*jac)[1][0] += (*hess)[1][0];
					(*def_tot)[0] += (*def)[0];
					(*def_tot)[1] += (*def)[1];
					kap_tot += kap;
				}
			} else {
				// The following parallel scheme is useful for clusters when LOTS of perturbers are present
				double *hess00 = new double[nthreads];
				double *hess11 = new double[nthreads];
				double *hess01 = new double[nthreads];
				double *def0 = new double[nthreads];
				double *def1 = new double[nthreads];
				double *kapi = new double[nthreads];

				//cout << "Starting new deflection calculation..." << endl << flush;
				#pragma omp parallel
				{
					int thread2;
#ifdef USE_OPENMP
					thread2 = omp_get_thread_num();
#else
					thread2 = 0;
#endif
					lensvector *def = &defs_i[thread2];
					lensmatrix *hess = &hesses_i[thread2];
					//double hess00=0, hess11=0, hess01=0, def0=0, def1=0, kapi=0;
					int j;
					double kap;
					hess00[thread2] = 0;
					hess11[thread2] = 0;
					hess01[thread2] = 0;
					def0[thread2] = 0;
					def1[thread2] = 0;
					kapi[thread2] = 0;

					#pragma omp for schedule(dynamic)
					for (j=0; j < nlens; j++) {
						lens_list[j]->kappa_and_potential_derivatives(x,y,kap,(*def),(*hess));
						hess00[thread2] += (*hess)[0][0];
						hess11[thread2] += (*hess)[1][1];
						hess01[thread2] += (*hess)[0][1];
						def0[thread2] += (*def)[0];
						def1[thread2] += (*def)[1];
						kapi[thread2] += kap;
					}
					//#pragma omp critical
					//{
						//cout << "Thread " << thread2 << " finished" << endl << flush;
					//}
					//#pragma omp critical
					//{
						//(*jac)[0][0] += hess00;
						//(*jac)[1][1] += hess11;
						//(*jac)[0][1] += hess01;
						//(*jac)[1][0] += hess01;
						//(*def_tot)[0] += def0;
						//(*def_tot)[1] += def1;
						//kap_tot += kapi;
					//}
				}
				//cout << "Finished parallel part" << endl << flush;
				for (int j=0; j < nthreads; j++) {
					(*jac)[0][0] += hess00[j];
					(*jac)[1][1] += hess11[j];
					(*jac)[0][1] += hess01[j];
					(*jac)[1][0] += hess01[j];
					(*def_tot)[0] += def0[j];
					(*def_tot)[1] += def1[j];
					kap_tot += kapi[j];
				}
				delete[] hess00;
				delete[] hess11;
				delete[] hess01;
				delete[] def0;
				delete[] def1;
				delete[] kapi;
			}
			//double defx = (*def_tot)[0];
			//double defy = (*def_tot)[1];
			//double jac00 = (*jac)[0][0];

			(*jac)[0][0] *= zfacs[0];
			(*jac)[1][1] *= zfacs[0];
			(*jac)[0][1] *= zfacs[0];
			(*jac)[1][0] *= zfacs[0];
			(*def_tot)[0] *= zfacs[0];
			(*def_tot)[1] *= zfacs[0];
			kap_tot *= zfacs[0];
			//cout << "Finished def calc" << endl << flush;
		}
	}
	else {
		(*def_tot) = defspline->deflection(x,y);
		(*jac) = defspline->hessian(x,y);
		kap_tot = kappa(x,y,zfacs,betafacs);
	}
	srcpt[0] = x - (*def_tot)[0]; // this uses the lens equation, beta = theta - alpha
	srcpt[1] = y - (*def_tot)[1];

	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	invmag = determinant((*jac));
	//cout << "Finished def calc for real; invmag = " << invmag << ", srcpt0=" << srcpt[0] << " srcpt1=" << srcpt[1] << " kap=" << kap_tot << " " << endl << flush;
}

void QLens::sourcept_jacobian(const lensvector& xvec, lensvector& srcpt, lensmatrix& jac_tot, const int &thread, double* zfacs, double** betafacs)
{
	double x = xvec[0], y = xvec[1];
	lensvector *def_tot = &defs[thread];

	if (!defspline)
	{
		if (n_lens_redshifts > 1) {
			lensvector *x_i = &xvals_i[thread];
			lensmatrix *A_i = &Amats_i[thread];
			lensvector *def = &defs_i[thread];
			lensvector **def_i = &defs_subtot[thread];
			lensmatrix *hess = &hesses_i[thread];
			lensmatrix **hess_i = &hesses_subtot[thread];

			int i,j;
			jac_tot[0][0] = 0;
			jac_tot[1][1] = 0;
			jac_tot[0][1] = 0;
			jac_tot[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			for (i=0; i < n_lens_redshifts; i++) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
					(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
					(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
					(*A_i)[0][0] -= (betafacs[i-1][j])*((*hess_i)[j])[0][0];
					(*A_i)[1][1] -= (betafacs[i-1][j])*((*hess_i)[j])[1][1];
					(*A_i)[1][0] -= (betafacs[i-1][j])*((*hess_i)[j])[1][0];
					(*A_i)[0][1] -= (betafacs[i-1][j])*((*hess_i)[j])[0][1];
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					lens_list[zlens_group_lens_indx[i][j]]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess));
					(*hess_i)[i][0][0] += (*hess)[0][0];
					(*hess_i)[i][1][1] += (*hess)[1][1];
					(*hess_i)[i][0][1] += (*hess)[0][1];
					(*hess_i)[i][1][0] += (*hess)[1][0];
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
				(*def_i)[i][0] *= zfacs[i];
				(*def_i)[i][1] *= zfacs[i];
				(*def_tot)[0] += (*def_i)[i][0];
				(*def_tot)[1] += (*def_i)[i][1];

				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];

				jac_tot[0][0] += (*hess_i)[i][0][0];
				jac_tot[1][1] += (*hess_i)[i][1][1];
				jac_tot[1][0] += (*hess_i)[i][1][0];
				jac_tot[0][1] += (*hess_i)[i][0][1];
			}
		} else {
			jac_tot[0][0] = 0;
			jac_tot[1][1] = 0;
			jac_tot[0][1] = 0;
			jac_tot[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;

			if (nthreads==1) {
				lensvector *def = &defs_i[0];
				lensmatrix *hess = &hesses_i[0];
				int j;
				jac_tot[0][0] = 0;
				jac_tot[1][1] = 0;
				jac_tot[0][1] = 0;
				jac_tot[1][0] = 0;
				(*def_tot)[0] = 0;
				(*def_tot)[1] = 0;
				for (j=0; j < nlens; j++) {
					lens_list[j]->potential_derivatives(x,y,(*def),(*hess));
					jac_tot[0][0] += (*hess)[0][0];
					jac_tot[1][1] += (*hess)[1][1];
					jac_tot[0][1] += (*hess)[0][1];
					jac_tot[1][0] += (*hess)[1][0];
					(*def_tot)[0] += (*def)[0];
					(*def_tot)[1] += (*def)[1];
				}
			} else {
				// The following parallel scheme is useful for clusters when LOTS of perturbers are present
				double *hess00 = new double[nthreads];
				double *hess11 = new double[nthreads];
				double *hess01 = new double[nthreads];
				double *def0 = new double[nthreads];
				double *def1 = new double[nthreads];

				#pragma omp parallel
				{
					int thread2;
#ifdef USE_OPENMP
					thread2 = omp_get_thread_num();
#else
					thread2 = 0;
#endif
					lensvector *def = &defs_i[thread2];
					lensmatrix *hess = &hesses_i[thread2];
					//double hess00=0, hess11=0, hess01=0, def0=0, def1=0, kapi=0;
					int j;
					//double kap;
						hess00[thread2] = 0;
						hess11[thread2] = 0;
						hess01[thread2] = 0;
						def0[thread2] = 0;
						def1[thread2] = 0;

					#pragma omp for schedule(dynamic)
					for (j=0; j < nlens; j++) {
						lens_list[j]->potential_derivatives(x,y,(*def),(*hess));
						hess00[thread2] += (*hess)[0][0];
						hess11[thread2] += (*hess)[1][1];
						hess01[thread2] += (*hess)[0][1];
						def0[thread2] += (*def)[0];
						def1[thread2] += (*def)[1];
					}
					//#pragma omp critical
					//{
						//jac_tot[0][0] += hess00;
						//jac_tot[1][1] += hess11;
						//jac_tot[0][1] += hess01;
						//jac_tot[1][0] += hess01;
						//(*def_tot)[0] += def0;
						//(*def_tot)[1] += def1;
					//}
				}
				for (int j=0; j < nthreads; j++) {
					jac_tot[0][0] += hess00[j];
					jac_tot[1][1] += hess11[j];
					jac_tot[0][1] += hess01[j];
					jac_tot[1][0] += hess01[j];
					(*def_tot)[0] += def0[j];
					(*def_tot)[1] += def1[j];
				}
				delete[] hess00;
				delete[] hess11;
				delete[] hess01;
				delete[] def0;
				delete[] def1;
			}
			jac_tot[0][0] *= zfacs[0];
			jac_tot[1][1] *= zfacs[0];
			jac_tot[0][1] *= zfacs[0];
			jac_tot[1][0] *= zfacs[0];
			(*def_tot)[0] *= zfacs[0];
			(*def_tot)[1] *= zfacs[0];
		}
	}
	else {
		(*def_tot) = defspline->deflection(x,y);
		jac_tot = defspline->hessian(x,y);
	}
	srcpt[0] = x - (*def_tot)[0]; // this uses the lens equation, beta = theta - alpha
	srcpt[1] = y - (*def_tot)[1];

	jac_tot[0][0] = 1 - jac_tot[0][0];
	jac_tot[1][1] = 1 - jac_tot[1][1];
	jac_tot[0][1] = -jac_tot[0][1];
	jac_tot[1][0] = -jac_tot[1][0];
}

void QLens::create_and_add_lens(LensProfileName name, const int emode, const double zl, const double zs, const double mass_parameter, const double scale1, const double scale2, const double eparam, const double theta, const double xc, const double yc, const double special_param1, const double special_param2, const int pmode)
{
	// eparam can be either q (axis ratio) or epsilon (ellipticity) depending on the ellipticity mode
	
	LensProfile* new_lens;

	int old_emode = LensProfile::default_ellipticity_mode;
	if (emode != -1) LensProfile::default_ellipticity_mode = emode; // set ellipticity mode to user-specified value for this lens

		// *NOTE*: Gauss_NN and integral_tolerance should probably just be set as static variables in LensProfile, so they don't need to be passed in here

	Alpha* alphaptr;
	Shear* shearptr;
	Truncated_NFW* tnfwptr;
	switch (name) {
		case PTMASS:
			new_lens = new PointMass(zl, zs, mass_parameter, xc, yc, pmode, this); break;
		case SHEET:
			new_lens = new MassSheet(zl, zs, mass_parameter, xc, yc, this); break;
		case DEFLECTION:
			new_lens = new Deflection(zl, zs, scale1, scale2, this); break;
		case ALPHA:
			//new_lens = new Alpha(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, this); break; // the old way
			
			alphaptr = new Alpha();
			alphaptr->initialize_parameters(mass_parameter, scale1, scale2, eparam, theta, xc, yc);

			//alphaptr = new Alpha(mass_parameter, scale1, scale2, eparam, theta, xc, yc); // an alternative constructor to use; in this case you don't need to call initialize_parameters
		
			new_lens = alphaptr;
			break;
		case SHEAR:
			shearptr = new Shear();
			shearptr->initialize_parameters(eparam,theta,xc,yc);
			new_lens = shearptr;
			break;
			//new_lens = new Shear(zl, zs, eparam, theta, xc, yc, this); break;
		// Note: the Multipole profile is added using the function add_multipole_lens(..., this) because one of the input parameters is an int
		case nfw:
			new_lens = new NFW(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case TRUNCATED_nfw:
			tnfwptr = new Truncated_NFW(pmode,special_param1);
			tnfwptr->initialize_parameters(mass_parameter, scale1, scale2, eparam, theta, xc, yc);
			new_lens = tnfwptr;
			break;
			//new_lens = new Truncated_NFW(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, special_param1, pmode, this); break;
		case CORED_nfw:
			new_lens = new Cored_NFW(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case PJAFFE:
			new_lens = new PseudoJaffe(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case EXPDISK:
			new_lens = new ExpDisk(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, this); break;
		case HERNQUIST:
			new_lens = new Hernquist(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, this); break;
		case CORECUSP:
			if ((special_param1==-1000) or (special_param2==-1000)) die("special parameters need to be passed to create_and_add_lens(...) function for model CORECUSP");
			new_lens = new CoreCusp(zl, zs, mass_parameter, special_param1, special_param2, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case SERSIC_LENS:
			new_lens = new SersicLens(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case CORED_SERSIC_LENS:
			new_lens = new Cored_SersicLens(zl, zs, mass_parameter, scale1, scale2, special_param1, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case TESTMODEL: // Model for testing purposes
			new_lens = new TestModel(zl, zs, eparam, theta, xc, yc, Gauss_NN, integral_tolerance); break;
		default:
			die("Lens type not recognized");
	}
	if (emode != -1) LensProfile::default_ellipticity_mode = old_emode; // restore ellipticity mode to its default setting
	add_lens(new_lens,zl,zs);
}

/*
void QLens::create_and_add_lens(LensProfileName name, const int emode, const double zl, const double zs, const double mass_parameter, const double scale1, const double scale2, const double eparam, const double theta, const double xc, const double yc, const double special_param1, const double special_param2, const int pmode)
{
	// eparam can be either q (axis ratio) or epsilon (ellipticity) depending on the ellipticity mode
	
	add_new_lens_entry(zl);

	int old_emode = LensProfile::default_ellipticity_mode;
	if (emode != -1) LensProfile::default_ellipticity_mode = emode; // set ellipticity mode to user-specified value for this lens

		// *NOTE*: Gauss_NN and integral_tolerance should probably just be set as static variables in LensProfile, so they don't need to be passed in here

	switch (name) {
		case PTMASS:
			lens_list[nlens-1] = new PointMass(zl, zs, mass_parameter, xc, yc, pmode, this); break;
		case SHEET:
			lens_list[nlens-1] = new MassSheet(zl, zs, mass_parameter, xc, yc, this); break;
		case DEFLECTION:
			lens_list[nlens-1] = new Deflection(zl, zs, scale1, scale2, this); break;
		case ALPHA:
			lens_list[nlens-1] = new Alpha(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, this); break;
		case SHEAR:
			lens_list[nlens-1] = new Shear(zl, zs, eparam, theta, xc, yc, this); break;
		// Note: the Multipole profile is added using the function add_multipole_lens(..., this) because one of the input parameters is an int
		case nfw:
			lens_list[nlens-1] = new NFW(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case TRUNCATED_nfw:
			lens_list[nlens-1] = new Truncated_NFW(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, special_param1, pmode, this); break;
		case CORED_nfw:
			lens_list[nlens-1] = new Cored_NFW(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case PJAFFE:
			lens_list[nlens-1] = new PseudoJaffe(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case EXPDISK:
			lens_list[nlens-1] = new ExpDisk(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, this); break;
		case HERNQUIST:
			lens_list[nlens-1] = new Hernquist(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, this); break;
		case CORECUSP:
			if ((special_param1==-1000) or (special_param2==-1000)) die("special parameters need to be passed to create_and_add_lens(...) function for model CORECUSP");
			lens_list[nlens-1] = new CoreCusp(zl, zs, mass_parameter, special_param1, special_param2, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case SERSIC_LENS:
			lens_list[nlens-1] = new SersicLens(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case CORED_SERSIC_LENS:
			lens_list[nlens-1] = new Cored_SersicLens(zl, zs, mass_parameter, scale1, scale2, special_param1, eparam, theta, xc, yc, Gauss_NN, integral_tolerance, pmode, this); break;
		case TESTMODEL: // Model for testing purposes
			lens_list[nlens-1] = new TestModel(zl, zs, eparam, theta, xc, yc, Gauss_NN, integral_tolerance); break;
		default:
			die("Lens type not recognized");
	}
	if (emode != -1) LensProfile::default_ellipticity_mode = old_emode; // restore ellipticity mode to its default setting

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();
	if (auto_zsource_scaling) auto_zsource_scaling = false; // fix zsrc_ref now that a lens has been created, to make sure lens mass scale doesn't change when zsrc is varied
}
*/

void QLens::create_and_add_lens(const char *splinefile, const int emode, const double zl, const double zs, const double q, const double theta, const double qx, const double f, const double xc, const double yc)
{
	add_new_lens_entry(zl);

	int old_emode = LensProfile::default_ellipticity_mode;
	if (emode != -1) LensProfile::default_ellipticity_mode = emode; // set ellipticity mode to user-specified value for this lens
	lens_list[nlens-1] = new LensProfile(splinefile, zl, zs, q, theta, xc, yc, Gauss_NN, integral_tolerance, qx, f, this);
	if (emode != -1) LensProfile::default_ellipticity_mode = old_emode; // restore ellipticity mode to its default setting

	lens_list_vec.push_back(lens_list[nlens-1]); // used for Python wrapper

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();
}

void QLens::add_shear_lens(const double zl, const double zs, const double shear_p1, const double shear_p2, const double xc, const double yc)
{
	create_and_add_lens(SHEAR,-1,zl,zs,0,0,0,shear_p1,shear_p2,xc,yc);
}

void QLens::add_ptmass_lens(const double zl, const double zs, const double mass_parameter, const double xc, const double yc, const int pmode)
{
	create_and_add_lens(PTMASS,-1,zl,zs,mass_parameter,0,0,0,0,xc,yc,0,0,pmode);
}

void QLens::add_mass_sheet_lens(const double zl, const double zs, const double mass_parameter, const double xc, const double yc)
{
	create_and_add_lens(SHEET,-1,zl,zs,mass_parameter,0,0,0,0,xc,yc);
}

void QLens::add_multipole_lens(const double zl, const double zs, int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool kap, bool sine_term)
{
	add_new_lens_entry(zl);

	lens_list[nlens-1] = new Multipole(zl, zs, a_m, n, m, theta, xc, yc, kap, this, sine_term);
	lens_list_vec.push_back(lens_list[nlens-1]); // used for Python wrapper

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();
}

void QLens::add_tabulated_lens(const double zl, const double zs, int lnum, const double kscale, const double rscale, const double theta, const double xc, const double yc)
{
	// automatically set gridsize if the appropriate settings are turned on
	if (autogrid_before_grid_creation) autogrid();
	else {
		if (autocenter==true) {
			lens_list[primary_lens_number]->get_center_coords(grid_xcenter,grid_ycenter);
		}
		if (auto_gridsize_from_einstein_radius==true) {
			double re_major, reav;
			re_major = einstein_radius_of_primary_lens(reference_zfactors[lens_redshift_idx[primary_lens_number]],reav);
			if (re_major > 0.0) {
				double rmax = autogrid_frac*re_major;
				grid_xlength = 2*rmax;
				grid_ylength = 2*rmax;
				cc_rmax = rmax;
			}
		}
	}

	add_new_lens_entry(zl);

	lens_list[nlens-1] = new Tabulated_Model(zl, zs, kscale, rscale, theta, xc, yc, lens_list[lnum], tabulate_rmin, dmax(grid_xlength,grid_ylength), tabulate_logr_N, tabulate_phi_N,this);
	lens_list_vec.push_back(lens_list[nlens-1]); // used for Python wrapper

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();
}

void QLens::add_qtabulated_lens(const double zl, const double zs, int lnum, const double kscale, const double rscale, const double q, const double theta, const double xc, const double yc)
{
	// automatically set gridsize if the appropriate settings are turned on
	if (autogrid_before_grid_creation) autogrid();
	else {
		if (autocenter==true) {
			lens_list[primary_lens_number]->get_center_coords(grid_xcenter,grid_ycenter);
		}
		if (auto_gridsize_from_einstein_radius==true) {
			double re_major, reav;
			re_major = einstein_radius_of_primary_lens(reference_zfactors[lens_redshift_idx[primary_lens_number]],reav);

			if (re_major != 0.0) {
				double rmax = autogrid_frac*re_major;
				grid_xlength = 2*rmax;
				grid_ylength = 2*rmax;
				cc_rmax = rmax;
			}
		}
	}

	add_new_lens_entry(zl);

	lens_list[nlens-1] = new QTabulated_Model(zl, zs, kscale, rscale, q, theta, xc, yc, lens_list[lnum], tabulate_rmin, dmax(grid_xlength,grid_ylength), tabulate_logr_N, tabulate_phi_N, tabulate_qmin, tabulate_q_N, this);
	lens_list_vec.push_back(lens_list[nlens-1]); // used for Python wrapper

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();
}

bool QLens::add_tabulated_lens_from_file(const double zl, const double zs, const double kscale, const double rscale, const double theta, const double xc, const double yc, const string tabfileroot)
{
	string tabfilename;
	if (tabfileroot.find(".tab")==string::npos) tabfilename = tabfileroot + ".tab";
	else tabfilename = tabfileroot;
	ifstream tabfile(tabfilename.c_str());
	if (!tabfile.good()) return false;
	if (tabfile.eof()) return false;
	int i, j, k, rN, phiN;
	double dummy;
	string dummyname;
	tabfile >> dummyname;
	tabfile >> rN >> phiN;
	// check that the file length matches the number of fields expected from rN, phiN
	for (i=0; i < rN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < phiN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < rN; i++) {
		for (j=0; j < phiN; j++) {
			for (k=0; k < 7; k++) {
				if (tabfile.eof()) return false;
				tabfile >> dummy;
			}
		}
	}
	tabfile.clear();
	tabfile.seekg(0, ios::beg);

	add_new_lens_entry(zl);

	lens_list[nlens-1] = new Tabulated_Model(zl, zs, kscale, rscale, theta, xc, yc, tabfile, tabfilename, this);
	lens_list_vec.push_back(lens_list[nlens-1]); // used for Python wrapper

	for (i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();
	return true;
}

bool QLens::add_qtabulated_lens_from_file(const double zl, const double zs, const double kscale, const double rscale, const double q, const double theta, const double xc, const double yc, const string tabfileroot)
{
	string tabfilename;
	if (tabfileroot.find(".tab")==string::npos) tabfilename = tabfileroot + ".tab";
	else tabfilename = tabfileroot;
	ifstream tabfile(tabfilename.c_str());
	if (!tabfile.good()) return false;
	if (tabfile.eof()) return false;
	int i, j, k, l, rN, phiN, qN;
	double dummy;
	string dummyname;
	tabfile >> dummyname;
	tabfile >> rN >> phiN >> qN;
	// check that the file length matches the number of fields expected from rN, phiN
	for (i=0; i < rN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < phiN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < qN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < rN; i++) {
		for (j=0; j < phiN; j++) {
			for (l=0; l < qN; l++) {
				for (k=0; k < 7; k++) {
					if (tabfile.eof()) return false;
					tabfile >> dummy;
				}
			}
		}
	}
	tabfile.clear();
	tabfile.seekg(0, ios::beg);

	add_new_lens_entry(zl);

	lens_list[nlens-1] = new QTabulated_Model(zl, zs, kscale, rscale, q, theta, xc, yc, tabfile, this);
	lens_list_vec.push_back(lens_list[nlens-1]); // used for Python wrapper

	for (i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();
	return true;
}

void QLens::add_lens(LensProfile *new_lens, const double zl, const double zs)
{
	new_lens->set_integration_parameters(Gauss_NN, integral_tolerance);
	new_lens->setup_cosmology(this,zl,zs);

	add_new_lens_entry(new_lens->zlens);

	new_lens->lens_number = nlens-1;
	lens_list[nlens-1] = new_lens;
	lens_list_vec.push_back(lens_list[nlens-1]); // used for Python wrapper
	lens_list[nlens-1]->register_vary_flags();

	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();
	if (auto_zsource_scaling) auto_zsource_scaling = false; // fix zsrc_ref now that a lens has been created, to make sure lens mass scale doesn't change when zsrc is varied
}

void QLens::add_new_lens_entry(const double zl)
{
	LensProfile** newlist = new LensProfile*[nlens+1];
	int* new_lens_redshift_idx = new int[nlens+1];
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			newlist[i] = lens_list[i];
			new_lens_redshift_idx[i] = lens_redshift_idx[i];
		}
		delete[] lens_list;
		delete[] lens_redshift_idx;
	}
	bool new_redshift;
	add_new_lens_redshift(zl,nlens,new_lens_redshift_idx);
	lens_redshift_idx = new_lens_redshift_idx;
	lens_list = newlist;
	nlens++;

	//int j,k;
	//if (n_lens_redshifts > 1) {
		//cout << "Beta matrix:\n";
		//for (j=0; j < n_lens_redshifts-1; j++) {
			//for (k=0; k < j+1; k++) cout << default_zsrc_beta_factors[j][k] << " ";
			//cout << endl;
		//}
		//cout << endl;
	//}
}

void QLens::add_new_lens_redshift(const double zl, const int lens_i, int* zlens_idx)
{
	int i, j, k, znum;
	bool new_redshift = true;
	for (i=0; i < n_lens_redshifts; i++) {
		if (lens_redshifts[i]==zl) { znum = i; new_redshift = false; break; }
	}
	if (new_redshift) {
		znum = n_lens_redshifts;
		double *new_lens_redshifts = new double[n_lens_redshifts+1];
		int *new_zlens_group_size = new int[n_lens_redshifts+1];
		int **new_zlens_group_lens_indx = new int*[n_lens_redshifts+1];
		int *new_zlens_group_lens_indx_col = new int[1];
		double *new_reference_zfactors = new double[n_lens_redshifts+1];
		new_zlens_group_lens_indx_col[0] = lens_i;
		for (i=0; i < n_lens_redshifts; i++) {
			if (zl < lens_redshifts[i]) {
				znum = i;
				break;
			}
		}
		for (i=0; i < znum; i++) {
			new_lens_redshifts[i] = lens_redshifts[i];
			new_zlens_group_lens_indx[i] = zlens_group_lens_indx[i];
			new_zlens_group_size[i] = zlens_group_size[i];
			new_reference_zfactors[i] = reference_zfactors[i];
		}
		new_lens_redshifts[znum] = zl;
		new_zlens_group_lens_indx[znum] = new_zlens_group_lens_indx_col;
		new_zlens_group_size[znum] = 1;
		new_reference_zfactors[znum] = kappa_ratio(zl,source_redshift,reference_source_redshift);
		for (i=znum; i < n_lens_redshifts; i++) {
			new_lens_redshifts[i+1] = lens_redshifts[i];
			new_zlens_group_lens_indx[i+1] = zlens_group_lens_indx[i];
			new_zlens_group_size[i+1] = zlens_group_size[i];
			new_reference_zfactors[i+1] = reference_zfactors[i];
		}
		if (n_lens_redshifts > 0) {
			delete[] lens_redshifts;
			delete[] zlens_group_lens_indx;
			delete[] zlens_group_size;
			delete[] reference_zfactors;
		}
		lens_redshifts = new_lens_redshifts;
		zlens_group_lens_indx = new_zlens_group_lens_indx;
		zlens_group_size = new_zlens_group_size;
		reference_zfactors = new_reference_zfactors;

		double **new_default_zsrc_beta_factors;
		if (n_lens_redshifts > 0) {
			// later you can improve on this so it doesn't have to recalculate previous beta matrix elements, but for now I just want to get
			// it up and running quickly
			new_default_zsrc_beta_factors = new double*[n_lens_redshifts];
			for (i=1; i < n_lens_redshifts+1; i++) {
				new_default_zsrc_beta_factors[i-1] = new double[i];
				if (include_recursive_lensing) {
					for (j=0; j < i; j++) new_default_zsrc_beta_factors[i-1][j] = calculate_beta_factor(lens_redshifts[j],lens_redshifts[i],source_redshift); // from cosmo.cpp
				} else {
					for (j=0; j < i; j++) new_default_zsrc_beta_factors[i-1][j] = 0;
				}
			}
			if (default_zsrc_beta_factors != NULL) {
				for (i=0; i < n_lens_redshifts-1; i++) {
					delete[] default_zsrc_beta_factors[i];
				}
				delete[] default_zsrc_beta_factors;
			}
			default_zsrc_beta_factors = new_default_zsrc_beta_factors;
		}

		double **new_zfactors;
		double ***new_beta_factors;
		if (n_sourcepts_fit > 0) {
			new_zfactors = new double*[n_sourcepts_fit];
			new_beta_factors = new double**[n_sourcepts_fit];
			for (i=0; i < n_sourcepts_fit; i++) {
				new_zfactors[i] = new double[n_lens_redshifts+1];
				for (j=0; j < znum; j++) {
					new_zfactors[i][j] = zfactors[i][j];
				}
				new_zfactors[i][znum] = kappa_ratio(zl,source_redshifts[i],reference_source_redshift);
				for (j=znum; j < n_lens_redshifts; j++) {
					new_zfactors[i][j+1] = zfactors[i][j];
				}

				if (n_lens_redshifts > 0) {
					new_beta_factors[i] = new double*[n_lens_redshifts];
					for (j=1; j < n_lens_redshifts+1; j++) {
						new_beta_factors[i][j-1] = new double[j];
						if (include_recursive_lensing) {
							for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],source_redshifts[i]); // from cosmo.cpp
						} else {
							for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = 0;
						}
					}
				} else new_beta_factors[i] = NULL;
			}
			if (zfactors != NULL) {
				for (i=0; i < n_sourcepts_fit; i++) delete[] zfactors[i];
				delete[] zfactors;
			}
			if (beta_factors != NULL) {
				for (i=0; i < n_sourcepts_fit; i++) {
					if (beta_factors[i] != NULL) {
						for (j=0; j < n_lens_redshifts-1; j++) {
							delete[] beta_factors[i][j];
						}
						if (n_lens_redshifts > 1) delete[] beta_factors[i];
					}
				}
				delete[] beta_factors;
			}
			zfactors = new_zfactors;
			beta_factors = new_beta_factors;
		}
		n_lens_redshifts++;
		//for (i=0; i < n_lens_redshifts; i++) {
			//cout << i << " " << lens_redshifts[i] << " " << reference_zfactors[i] << endl;
		//}
	} else {
		int *new_zlens_group_lens_indx_col = new int[zlens_group_size[znum]+1];
		for (i=0; i < zlens_group_size[znum]; i++) {
			new_zlens_group_lens_indx_col[i] = zlens_group_lens_indx[znum][i];
		}
		new_zlens_group_lens_indx_col[zlens_group_size[znum]] = lens_i;
		delete[] zlens_group_lens_indx[znum];
		zlens_group_lens_indx[znum] = new_zlens_group_lens_indx_col;
		zlens_group_size[znum]++;
	}
	zlens_idx[lens_i] = znum;
	if (new_redshift) {
		// we inserted a new redshift, so higher redshifts get bumped up an index
		for (j=0; j < nlens; j++) {
			if (j==lens_i) continue;
			if (zlens_idx[j] >= zlens_idx[lens_i]) zlens_idx[j]++;
		}
	}
}

void QLens::update_lens_redshift_data()
{
	for (int i=0; i < nlens; i++) {
		if (lens_redshifts[lens_redshift_idx[i]] != lens_list[i]->zlens) {
			double old_zlens = lens_redshifts[lens_redshift_idx[i]];
			double new_zlens = lens_list[i]->zlens;
			remove_old_lens_redshift(lens_redshift_idx[i],i,false); // this will only remove the redshift if there are no other lenses with the old redshift
			add_new_lens_redshift(new_zlens,i,lens_redshift_idx); // this will only add a new redshift if there are no other lenses with new redshift
		}
	}
	for (int i=0; i < nlens; i++) {
		if (lens_list[i]->lensed_center_coords) lens_list[i]->set_center_if_lensed_coords(); // for LOS perturbers whose lensed center coordinates are used as parameters (updates true center)
	}
}

void QLens::remove_old_lens_redshift(const int znum, const int lens_i, const bool removed_lens)
{
	int i, j, k, nlenses_with_znum=0, idx=-1;
	for (i=0; i < nlens; i++) {
		if (lens_redshift_idx[i]==znum) {
			nlenses_with_znum++;
			idx = i;
			if (nlenses_with_znum > 1) break;
		}
	}
	if (nlenses_with_znum==1) {
		double *new_lens_redshifts = new double[n_lens_redshifts-1];
		int *new_zlens_group_size = new int[n_lens_redshifts-1];
		int **new_zlens_group_lens_indx = new int*[n_lens_redshifts-1];
		for (i=0; i < znum; i++) {
			new_lens_redshifts[i] = lens_redshifts[i];
			new_zlens_group_size[i] = zlens_group_size[i];
			new_zlens_group_lens_indx[i] = zlens_group_lens_indx[i];
		}
		for (i=znum; i < n_lens_redshifts-1; i++) {
			new_lens_redshifts[i] = lens_redshifts[i+1];
			new_zlens_group_size[i] = zlens_group_size[i+1];
			new_zlens_group_lens_indx[i] = zlens_group_lens_indx[i+1];
		}
		if (lens_redshifts != NULL) delete[] lens_redshifts;
		delete[] zlens_group_lens_indx[znum];
		delete[] zlens_group_lens_indx;
		delete[] zlens_group_size;
		zlens_group_lens_indx = new_zlens_group_lens_indx;
		zlens_group_size = new_zlens_group_size;
		lens_redshifts = new_lens_redshifts;
		for (i=0; i < nlens; i++) {
			if (lens_redshift_idx[i] > znum) lens_redshift_idx[i]--;
		}

		double *new_reference_zfactors;
		double **new_default_zsrc_beta_factors;
		if (n_lens_redshifts==1) {
			delete[] reference_zfactors;
			reference_zfactors = NULL;
		} else {
			new_reference_zfactors = new double[n_lens_redshifts-1];
			for (j=0; j < znum; j++) new_reference_zfactors[j] = reference_zfactors[j];
			for (j=znum; j < n_lens_redshifts-1; j++) new_reference_zfactors[j] = reference_zfactors[j+1];
			delete[] reference_zfactors;
			reference_zfactors = new_reference_zfactors;
			if (n_lens_redshifts==2) {
				delete[] default_zsrc_beta_factors[0];
				delete[] default_zsrc_beta_factors;
				default_zsrc_beta_factors = NULL;
			} else {
				new_default_zsrc_beta_factors = new double*[n_lens_redshifts-2];
				for (i=1; i < n_lens_redshifts-1; i++) {
					new_default_zsrc_beta_factors[i-1] = new double[i];
					if (include_recursive_lensing) {
						for (j=0; j < i; j++) new_default_zsrc_beta_factors[i-1][j] = calculate_beta_factor(lens_redshifts[j],lens_redshifts[i],source_redshift); // from cosmo.cpp
					} else {
						for (j=0; j < i; j++) new_default_zsrc_beta_factors[i-1][j] = 0;
					}
				}
				if (default_zsrc_beta_factors != NULL) {
					for (i=0; i < n_lens_redshifts-1; i++) {
						delete[] default_zsrc_beta_factors[i];
					}
					delete[] default_zsrc_beta_factors;
				}
				default_zsrc_beta_factors = new_default_zsrc_beta_factors;
			}
		}

		double **new_zfactors;
		if (n_sourcepts_fit > 0) {
			if (n_lens_redshifts==1) {
				for (i=0; i < n_sourcepts_fit; i++) delete[] zfactors[i];
				delete[] zfactors;
				zfactors = NULL;
			} else {
				new_zfactors = new double*[n_sourcepts_fit];
				for (i=0; i < n_sourcepts_fit; i++) {
					new_zfactors[i] = new double[n_lens_redshifts-1];
					for (j=0; j < znum; j++) {
						new_zfactors[i][j] = zfactors[i][j];
					}
					for (j=znum; j < n_lens_redshifts-1; j++) {
						new_zfactors[i][j] = zfactors[i][j+1];
					}
				}
				for (i=0; i < n_sourcepts_fit; i++) delete[] zfactors[i];
				delete[] zfactors;
				zfactors = new_zfactors;

				double ***new_beta_factors;
				if (n_lens_redshifts==2) {
					for (i=0; i < n_sourcepts_fit; i++) {
						delete[] beta_factors[i][0];
						delete[] beta_factors[i];
						beta_factors[i] = NULL;
					}
					default_zsrc_beta_factors = NULL;
				} else {
					new_beta_factors = new double**[n_sourcepts_fit];
					for (i=0; i < n_sourcepts_fit; i++) {
						new_beta_factors[i] = new double*[n_lens_redshifts-2];
						for (j=1; j < n_lens_redshifts-1; j++) {
							new_beta_factors[i][j-1] = new double[j];
							if (include_recursive_lensing) {
								for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],source_redshifts[i]); // from cosmo.cpp
							} else {
								for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = 0;
							}
						}
					}
					if (beta_factors != NULL) {
						for (i=0; i < n_sourcepts_fit; i++) {
							for (j=0; j < n_lens_redshifts-1; j++) {
								delete[] beta_factors[i][j];
							}
							if (n_lens_redshifts > 1) delete[] beta_factors[i];
						}
						delete[] beta_factors;
					}
					beta_factors = new_beta_factors;
				}
			}
		}
		n_lens_redshifts--;
	} else {
		int *new_zlens_group_lens_indx_col = new int[zlens_group_size[znum]-1];
		for (i=0,j=0; i < zlens_group_size[znum]; i++) {
			if (zlens_group_lens_indx[znum][i] != lens_i) {
				new_zlens_group_lens_indx_col[j] = zlens_group_lens_indx[znum][i];
				j++;
			}
		}
		delete[] zlens_group_lens_indx[znum];
		zlens_group_lens_indx[znum] = new_zlens_group_lens_indx_col;
		zlens_group_size[znum]--;
	}
	if (removed_lens) {
		for (i=0; i < n_lens_redshifts; i++) {
			for (j=0; j < zlens_group_size[i]; j++) {
				// move all the lens indices greater than lens_i down by one, since we've removed lens_i
				if (zlens_group_lens_indx[i][j] > lens_i) zlens_group_lens_indx[i][j]--;
			}
		}
	}


}

void QLens::print_beta_matrices()
{
	int i,j,k;
	for (i=0; i < nlens; i++) cout << "Lens " << i << " redshift index: " << lens_redshift_idx[i] << endl;
	cout << "zfacs: ";
	for (i=0; i < n_lens_redshifts; i++) cout << reference_zfactors[i] << " ";
	cout << endl;
	cout << "zsrc=" << source_redshift << " beta matrix:\n";
	for (j=0; j < n_lens_redshifts-1; j++) {
		cout << "z=" << lens_redshifts[j] << ": ";
		for (k=0; k < j+1; k++) cout << default_zsrc_beta_factors[j][k] << " ";
		cout << endl;
	}

	if (source_fit_mode == Point_Source) {
		for (i=0; i < n_sourcepts_fit; i++) {
			cout << "source " << i << " beta matrix:\n";
			for (j=0; j < n_lens_redshifts-1; j++) {
				for (k=0; k < j+1; k++) cout << beta_factors[i][j][k] << " ";
				cout << endl;
			}
			cout << endl;
		}
	}
	//lensvector x,y;
	//x[0] = 2; x[1] = 0;
	//lensvector defp;
	//lens_list[0]->deflection(x[0],x[1],defp);
	//y = x - default_zsrc_beta_factors[0][0]*defp;
	//cout << "map to lens plane 1: " << y[0] << " " << y[1] << endl;
}

bool QLens::save_tabulated_lens_to_file(int lnum, const string tabfileroot)
{
	int pos;
	string tabfilename = tabfileroot;
	if ((pos = tabfilename.find(".tab")) != string::npos) {
		tabfilename = tabfilename.substr(0,pos);
	}

	if ((lens_list[lnum]->get_lenstype() != TABULATED) and (lens_list[lnum]->get_lenstype() != QTABULATED)) return false;
	if (lens_list[lnum]->get_lenstype() == TABULATED) {
		Tabulated_Model temp_tablens((Tabulated_Model*) lens_list[lnum]);
		temp_tablens.output_tables(tabfilename);
	} else {
		QTabulated_Model temp_tablens((QTabulated_Model*) lens_list[lnum]);
		temp_tablens.output_tables(tabfilename);
	}
	return true;
}

bool QLens::set_lens_vary_parameters(const int lensnumber, boolvector &vary_flags)
{
	int pi, pf;
	get_lens_parameter_numbers(lensnumber,pi,pf);
	if (lens_list[lensnumber]->vary_parameters(vary_flags)==false) return false;
	if (pf > pi) param_settings->remove_params(pi,pf);
	return register_lens_vary_parameters(lensnumber);
}

bool QLens::register_lens_vary_parameters(const int lensnumber)
{
	int pi, pf, nparams;
	get_n_fit_parameters(nparams);
	dvector stepsizes(nparams);
	get_parameter_names();
	if (get_lens_parameter_numbers(lensnumber,pi,pf) == true) {
		//cout << "Inserting parameters " << pi << " to " << pf << endl;
		get_automatic_initial_stepsizes(stepsizes);
		//param_settings->print_penalty_limits();
		param_settings->insert_params(pi,pf,fit_parameter_names,stepsizes.array());
		//cout << "Inserting parameters done " << endl;
		//param_settings->print_penalty_limits();
		int index=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		lens_list[lensnumber]->get_auto_ranges(use_penalty_limits,lower,upper,index);
		//cout << "Updating lens plimits from " << pi << " to " << (pf-1) << endl;
		param_settings->update_specific_penalty_limits(pi,pf,use_penalty_limits,lower,upper);
		//param_settings->print_penalty_limits();
	}
	return true;
}

bool QLens::set_sb_vary_parameters(const int sbnumber, boolvector &vary_flags)
{
	int pi, pf, nparams;
	get_sb_parameter_numbers(sbnumber,pi,pf);
	if (sb_list[sbnumber]->vary_parameters(vary_flags)==false) return false;
	if (pf > pi) param_settings->remove_params(pi,pf);
	get_n_fit_parameters(nparams);
	dvector stepsizes(nparams);
	get_parameter_names();
	if (get_sb_parameter_numbers(sbnumber,pi,pf) == true) {
		get_automatic_initial_stepsizes(stepsizes);
		param_settings->insert_params(pi,pf,fit_parameter_names,stepsizes.array());
		int index=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		sb_list[sbnumber]->get_auto_ranges(use_penalty_limits,lower,upper,index);
		//cout << "Updating lens plimits from " << pi << " to " << (pf-1) << endl;
		param_settings->update_specific_penalty_limits(pi,pf,use_penalty_limits,lower,upper);
	}
	return true;
}

void QLens::update_parameter_list()
{
	// One slight issue that should be fixed: for the "extra" parameters like regularization, hubble constant, etc., the stepsizes
	// and plimits are not preserved if one of the extra parameters is removed and it's not the last one on the list. There should
	// be a more specific update such that just those parameters are removed (using remove_params(...))
	get_n_fit_parameters(n_fit_parameters);
	if (n_fit_parameters > 0) {
		dvector stepsizes(n_fit_parameters);
		get_parameter_names();
		get_automatic_initial_stepsizes(stepsizes);
		param_settings->update_params(n_fit_parameters,fit_parameter_names,stepsizes.array());
	} else {
		param_settings->clear_params();
	}
}

void QLens::remove_lens(int lensnumber)
{
	int pi, pf;
	get_lens_parameter_numbers(lensnumber,pi,pf);

	if ((lensnumber >= nlens) or (nlens==0)) { warn(warnings,"Specified lens does not exist"); return; }
	LensProfile** newlist = new LensProfile*[nlens-1];
	int* new_lens_redshift_idx;
	if (nlens > 1) new_lens_redshift_idx = new int[nlens-1];
	int i,j;
	for (i=0; i < nlens; i++) {
		if ((i != lensnumber) and (lens_list[i]->center_anchored==true) and (lens_list[i]->get_center_anchor_number()==lensnumber)) lens_list[i]->delete_center_anchor();
		if ((i != lensnumber) and (lens_list[i]->anchor_special_parameter==true) and (lens_list[i]->get_special_parameter_anchor_number()==lensnumber)) lens_list[i]->delete_special_parameter_anchor();
		if (i != lensnumber) lens_list[i]->unanchor_parameter(lens_list[lensnumber]); // this unanchors the lens if any of its parameters are anchored to the lens being deleted
	}
	for (i=0; i < n_sb; i++) {
		// If any source profiles are anchored to the center of this lens (typically to model foreground light), delete the anchor
		if ((sb_list[i]->center_anchored==true) and (sb_list[i]->get_center_anchor_number()==lensnumber)) sb_list[i]->delete_center_anchor();
	}
	remove_old_lens_redshift(lens_redshift_idx[lensnumber], lensnumber, true); // removes the lens redshift from the list if no other lenses share that redshift
	for (i=0,j=0; i < nlens; i++) {
		if (i != lensnumber) {
			newlist[j] = lens_list[i];
			new_lens_redshift_idx[j] = lens_redshift_idx[i];
			j++;
		}
	}
	delete lens_list[lensnumber];
	delete[] lens_list;
	delete[] lens_redshift_idx;
	nlens--;
	lens_list_vec.erase(lens_list_vec.begin()+lensnumber); // Used for Python wrapper

	lens_list = newlist;
	if (nlens > 0) lens_redshift_idx = new_lens_redshift_idx;
	else lens_redshift_idx = NULL;
	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	if (auto_ccspline) automatically_determine_ccspline_mode();

	param_settings->remove_params(pi,pf);
	update_parameter_list();
	get_parameter_names(); // parameter names must be updated whenever lens models are removed/added
	for (int i=n_derived_params-1; i >= 0; i--) {
		if (dparam_list[i]->lensnum_param==lensnumber) {
			if (mpi_id==0) cout << "Removing derived param " << i << endl;
			remove_derived_param(i);
		}
	}
}

void QLens::clear_lenses()
{
	if (nlens > 0) {
		lens_list_vec.clear(); // Used for Python wrapper
		int pi, pf, pi_min=1000, pf_max=0;
		// since all the lens parameters are blocked together in the param_settings list, we just need to find the initial and final parameters to remove
		for (int i=0; i < nlens; i++) {
			get_lens_parameter_numbers(i,pi,pf);
			if (pi < pi_min) pi_min=pi;
			if (pf > pf_max) pf_max=pf;
		}	
		for (int i=0; i < nlens; i++) {
			delete lens_list[i];
		}	
		delete[] lens_list;
		param_settings->remove_params(pi_min,pf_max);
		nlens = 0;
		update_parameter_list(); // this is necessary to keep the parameter priors, transforms preserved
		if (lens_redshift_idx != NULL) {
			delete[] lens_redshift_idx;
			lens_redshift_idx = NULL;
		}
		if (lens_redshifts != NULL) {
			delete[] lens_redshifts;
			lens_redshifts = NULL;
		}
		if (zlens_group_size != NULL) {
			delete[] zlens_group_size;
			zlens_group_size = NULL;
		}
		if (zlens_group_lens_indx != NULL) {
			for (int i=0; i < n_lens_redshifts; i++) delete[] zlens_group_lens_indx[i];
			delete[] zlens_group_lens_indx;
			zlens_group_lens_indx = NULL;
		}
		if (reference_zfactors != NULL) {
			delete[] reference_zfactors;
			reference_zfactors = NULL;
		}
		if (zfactors != NULL) {
			for (int i=0; i < n_sourcepts_fit; i++) delete[] zfactors[i];
			delete[] zfactors;
			zfactors = NULL;
		}
		if (default_zsrc_beta_factors != NULL) {
			for (int i=0; i < n_lens_redshifts-1; i++) delete[] default_zsrc_beta_factors[i];
			delete[] default_zsrc_beta_factors;
			default_zsrc_beta_factors = NULL;
		}
		if (beta_factors != NULL) {
			for (int i=0; i < n_sourcepts_fit; i++) {
				for (int j=0; j < n_lens_redshifts-1; j++) delete[] beta_factors[i][j];
				if (n_lens_redshifts > 1) delete[] beta_factors[i];
			}
			delete[] beta_factors;
			beta_factors = NULL;
		}

		n_lens_redshifts = 0;

		reset_grid();
		get_parameter_names(); // parameter names must be updated whenever lens models are removed/added
		clear_derived_params();
	}
}

void QLens::set_source_redshift(const double zsrc)
{
	source_redshift = zsrc;
	int i,j;
	if (auto_zsource_scaling) {
		reference_source_redshift = source_redshift;
		for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = 1.0;
		for (i=0; i < n_sourcepts_fit; i++) {
			for (j=0; j < n_lens_redshifts; j++) {
				zfactors[i][j] = kappa_ratio(lens_redshifts[j],source_redshifts[i],reference_source_redshift);
			}
		}
	} else {
		for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = kappa_ratio(lens_redshifts[i],source_redshift,reference_source_redshift);
	}
	recalculate_beta_factors();
	//reset_grid();
}

void QLens::set_reference_source_redshift(const double zsrc)
{
	if (nlens > 0) { warn("zsrc_ref cannot be changed if any lenses have already been created"); return; }
	int i,j;
	reference_source_redshift = zsrc;
	if (auto_zsource_scaling==true) auto_zsource_scaling = false; // Now that zsrc_ref has been set explicitly, don't automatically change it if zsrc is changed
	for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = kappa_ratio(lens_redshifts[i],source_redshift,reference_source_redshift);
	reset_grid();
	if (n_sourcepts_fit > 0) {
		for (i=0; i < n_sourcepts_fit; i++) {
			for (j=0; j < n_lens_redshifts; j++) {
				zfactors[i][j] = kappa_ratio(lens_redshifts[j],source_redshifts[i],reference_source_redshift);
			}
		}
	}
}

void QLens::recalculate_beta_factors()
{
	int i,j;
	if (n_lens_redshifts > 1) {
		for (i=1; i < n_lens_redshifts; i++) {
			if (include_recursive_lensing) {
				for (j=0; j < i; j++) default_zsrc_beta_factors[i-1][j] = calculate_beta_factor(lens_redshifts[j],lens_redshifts[i],source_redshift);
			} else {
				for (j=0; j < i; j++) default_zsrc_beta_factors[i-1][j] = 0;
			}
		}
	}
}

void QLens::toggle_major_axis_along_y(bool major_axis_along_y)
{
	if (LensProfile::orient_major_axis_north != major_axis_along_y) {
		LensProfile::orient_major_axis_north = major_axis_along_y;
		if (nlens > 0) {
			if (major_axis_along_y) {
				for (int i=0; i < nlens; i++) lens_list[i]->shift_angle_minus_90();
			} else {
				for (int i=0; i < nlens; i++) lens_list[i]->shift_angle_90();
			}
		}
	}
}

void QLens::automatically_determine_ccspline_mode()
{
	// this feature should be made obsolete. It's really not necessary to spline the critical curves anymore
	bool kappa_present = false;
	for (int i=0; i < nlens; i++) {
		if ((autocenter==true) and (i==primary_lens_number)) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
		if ((lens_list[i]->get_lenstype() != SHEAR) and (lens_list[i]->get_lenstype() != MULTIPOLE)) kappa_present = true;
	}
	if (kappa_present==false) set_ccspline_mode(false);
	else if ((test_for_elliptical_symmetry()==true) and (test_for_singularity()==false)) set_ccspline_mode(true);
	else set_ccspline_mode(false);
}

bool QLens::test_for_elliptical_symmetry()
{
	bool elliptical_symmetry_present = true;
	if (nlens > 0) {
		double xc, yc;
		for (int i=0; i < nlens; i++) {
			lens_list[i]->get_center_coords(xc,yc);
			if ((abs(xc-grid_xcenter) > Grid::image_pos_accuracy) or (abs(yc-grid_ycenter) > Grid::image_pos_accuracy))
				elliptical_symmetry_present = false;
		}
	}
	return elliptical_symmetry_present;
}

bool QLens::test_for_singularity()
{
	// if kappa goes like r^n near the origin where n <= -1, then a radial critical curve will not form
	// (this is because the deflection, which goes like r^(n+1), must increase as you go outward in order
	// to have a radial critical curve; this will only happen for n>-1). Here we test for this for the
	// relevant models where this can occur
	bool singular = false;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			if ((lens_list[i]->get_lenstype() == PJAFFE) and (lens_list[i]->core_present()==false)) singular = true;
			else if ((lens_list[i]->get_lenstype() == ALPHA) and (lens_list[i]->get_inner_logslope() <= -1) and (lens_list[i]->core_present()==false)) singular = true;
			else if (lens_list[i]->get_lenstype() == PTMASS) singular = true;
				// a radial critical curve will occur if a core is present, OR if alpha > 1 (since kappa goes like r^n where n=alpha-2)
		}
	}
	return singular;
}

void QLens::record_singular_points(double *zfacs)
{
	// if kappa goes like r^n near the origin where n <= -1, then a radial critical curve will not form
	// (this is because the deflection, which goes like r^(n+1), must increase as you go outward in order
	// to have a radial critical curve; this will only happen for n>-1). Here we test for this for the
	// relevant models where this can occur
	singular_pts.clear();
	double xc, yc;
	bool singular;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			singular = false;
			if (zfacs[lens_redshift_idx[i]] != 0.0) {
				if ((lens_list[i]->get_lenstype() == PJAFFE) and (lens_list[i]->core_present()==false)) singular = true;
				else if ((lens_list[i]->get_lenstype() == ALPHA) and (lens_list[i]->get_inner_logslope() <= -1) and (lens_list[i]->core_present()==false)) singular = true;
				else if (lens_list[i]->get_lenstype() == PTMASS) singular = true;
					// a radial critical curve will occur if a core is present, OR if alpha > 1 (since kappa goes like r^n where n=alpha-2)
				if (singular) {
					lens_list[i]->get_center_coords(xc,yc);
					lensvector singular_pt(xc,yc);
					singular_pts.push_back(singular_pt);
				}
			}
		}
	}
	n_singular_points = singular_pts.size();
}

void QLens::add_source_object(SB_ProfileName name, double sb_norm, double scale, double scale2, double logslope_param, double q, double theta, double xc, double yc)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			newlist[i] = sb_list[i];
		delete[] sb_list;
	}

	switch (name) {
		case GAUSSIAN:
			newlist[n_sb] = new Gaussian(sb_norm, scale, q, theta, xc, yc); break;
		case SERSIC:
			newlist[n_sb] = new Sersic(sb_norm, scale, logslope_param, q, theta, xc, yc); break;
		case CORED_SERSIC:
			newlist[n_sb] = new Cored_Sersic(sb_norm, scale, logslope_param, scale2, q, theta, xc, yc); break;
		case TOPHAT:
			newlist[n_sb] = new TopHat(sb_norm, scale, q, theta, xc, yc); break;
		default:
			die("Surface brightness profile type not recognized");
	}
	n_sb++;
	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}

void QLens::add_shapelet_source(const double amp00, const double sig_x, const double q, const double theta, const double xc, const double yc, const int nmax, const bool truncate)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			newlist[i] = sb_list[i];
		delete[] sb_list;
	}

	newlist[n_sb] = new Shapelet(amp00, sig_x, q, theta, xc, yc, nmax, truncate);
	n_sb++;
	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}

void QLens::add_multipole_source(int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool sine_term)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			newlist[i] = sb_list[i];
		delete[] sb_list;
	}

	newlist[n_sb] = new SB_Multipole(a_m, n, m, theta, xc, yc, sine_term);
	n_sb++;
	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}

void QLens::add_source_object(const char *splinefile, double q, double theta, double qx, double f, double xc, double yc)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			newlist[i] = sb_list[i];
		delete[] sb_list;
	}
	newlist[n_sb++] = new SB_Profile(splinefile, q, theta, xc, yc, qx, f);

	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}

/*
void QLens::remove_source_object(int sb_number)
{
	if ((sb_number >= n_sb) or (n_sb == 0)) { warn(warnings,"Specified source object does not exist"); return; }
	SB_Profile** newlist = new SB_Profile*[n_sb-1];
	int i,j;
	for (i=0, j=0; i < n_sb; i++)
		if (i != sb_number) { newlist[j] = sb_list[i]; j++; }
	delete[] sb_list;
	n_sb--;

	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}
*/

void QLens::remove_source_object(int sb_number)
{
	int pi, pf;
	get_sb_parameter_numbers(sb_number,pi,pf);

	if ((sb_number >= n_sb) or (n_sb==0)) { warn(warnings,"Specified source object does not exist"); return; }
	SB_Profile** newlist = new SB_Profile*[n_sb-1];
	int i,j;
	for (i=0, j=0; i < n_sb; i++) {
		if (i != sb_number) {
			newlist[j] = sb_list[i];
			j++;
		}
	}
	delete sb_list[sb_number];
	delete[] sb_list;
	n_sb--;

	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;

	param_settings->remove_params(pi,pf);
	update_parameter_list();
	get_parameter_names(); // parameter names must be updated whenever lens models are removed/added
}

void QLens::clear_source_objects()
{
	if (n_sb > 0) {
		int pi, pf, pi_min=1000, pf_max=0;
		// since all the source parameters are blocked together in the param_settings list, we just need to find the initial and final parameters to remove
		for (int i=0; i < n_sb; i++) {
			get_sb_parameter_numbers(i,pi,pf);
			if (pi < pi_min) pi_min=pi;
			if (pf > pf_max) pf_max=pf;
		}	
		for (int i=0; i < n_sb; i++) {
			delete sb_list[i];
		}	
		delete[] sb_list;
		param_settings->remove_params(pi_min,pf_max);
		n_sb = 0;
		update_parameter_list(); // this is necessary to keep the parameter priors, transforms preserved
		get_parameter_names(); // parameter names must be updated whenever source models are removed/added
	}
}

void QLens::print_source_list(bool show_vary_params)
{
	cout << resetiosflags(ios::scientific);
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			cout << i << ". ";
			sb_list[i]->print_parameters();
			if (show_vary_params)
				sb_list[i]->print_vary_parameters();
		}
	}
	else cout << "No source models have been specified" << endl;
	cout << endl;
	if (use_scientific_notation) cout << setiosflags(ios::scientific);
}

void QLens::add_derived_param(DerivedParamType type_in, double param, int lensnum, double param2, bool use_kpc)
{
	DerivedParam** newlist = new DerivedParam*[n_derived_params+1];
	if (n_derived_params > 0) {
		for (int i=0; i < n_derived_params; i++)
			newlist[i] = dparam_list[i];
		delete[] dparam_list;
	}
	if (param2 == -1e30) newlist[n_derived_params] = new DerivedParam(type_in,param,lensnum,-1,use_kpc);
	else newlist[n_derived_params] = new DerivedParam(type_in,param,lensnum,param2,use_kpc);
	n_derived_params++;
	dparam_list = newlist;
	param_settings->add_dparam(dparam_list[n_derived_params-1]->name);
}

void QLens::remove_derived_param(int dparam_number)
{
	if ((dparam_number >= n_derived_params) or (n_derived_params == 0)) { warn(warnings,"Specified derived parameter does not exist"); return; }
	DerivedParam** newlist = new DerivedParam*[n_derived_params-1];
	int i,j;
	for (i=0, j=0; i < n_derived_params; i++)
		if (i != dparam_number) { newlist[j] = dparam_list[i]; j++; }
		else delete dparam_list[i];
	delete[] dparam_list;
	n_derived_params--;

	dparam_list = newlist;
	param_settings->remove_dparam(dparam_number);
}

void QLens::rename_derived_param(int dparam_number, string newname, string new_latex_name)
{
	if (dparam_number >= n_derived_params) die("Specified derived parameter does not exist");
	dparam_list[dparam_number]->rename(words[4],words[5]);
	param_settings->rename_dparam(dparam_number,newname);
}

void QLens::clear_derived_params()
{
	if (n_derived_params > 0) {
		for (int i=0; i < n_derived_params; i++)
			delete dparam_list[i];
		delete[] dparam_list;
		n_derived_params = 0;
	}
	param_settings->clear_dparams();
}

void QLens::print_derived_param_list()
{
	if (mpi_id==0) {
		if (n_derived_params > 0) {
			for (int i=0; i < n_derived_params; i++) {
				cout << i << ". " << flush;
				dparam_list[i]->print_param_description(this);
			}
		}
		else cout << "No derived parameters have been created" << endl;
	}
}

void QLens::set_gridcenter(double xc, double yc)
{
	grid_xcenter=xc;
	grid_ycenter=yc;
	if (autocenter) autocenter = false;
}

void QLens::set_gridsize(double xl, double yl)
{
	grid_xlength = xl;
	grid_ylength = yl;
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
}

void QLens::set_grid_corners(double xmin, double xmax, double ymin, double ymax)
{
	grid_xcenter = 0.5*(xmax+xmin);
	grid_ycenter = 0.5*(ymax+ymin);
	grid_xlength = xmax-xmin;
	grid_ylength = ymax-ymin;
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
}

void QLens::autogrid(double rmin, double rmax, double frac)
{
	cc_rmin = rmin;
	cc_rmax = rmax;
	autogrid_frac = frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

void QLens::autogrid(double rmin, double rmax)
{
	cc_rmin = rmin;
	cc_rmax = rmax;
	autogrid_frac = default_autogrid_frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

void QLens::autogrid() {
	cc_rmin = default_autogrid_rmin;
	cc_rmax = default_autogrid_rmax;
	autogrid_frac = default_autogrid_frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		//else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

bool QLens::create_grid(bool verbal, double *zfacs, double **betafacs, const int redshift_index) // the last (optional) argument indicates which images are being fit to; used to optimize the subgridding
{
	if (nlens==0) { warn(warnings, "no lens model is specified"); return false; }
	double mytime0, mytime;
#ifdef USE_OPENMP
	if (show_wtime) {
		mytime0=omp_get_wtime();
	}
#endif
	lensvector *centers;
	double *einstein_radii;
	int i_primary=0;
	if ((subgrid_around_perturbers) and (nlens > 1)) {
		centers = new lensvector[nlens];
		einstein_radii = new double[nlens];
		find_effective_lens_centers_and_einstein_radii(centers,einstein_radii,i_primary,zfacs,betafacs,verbal);
	}
	if (grid != NULL) {
		int rsp, thetasp;
		grid->get_usplit_initial(rsp);
		grid->get_wsplit_initial(thetasp);
		if ((rsp != usplit_initial) or (thetasp != wsplit_initial)) {
			delete grid;
			grid = NULL;
		}
		if ((auto_store_cc_points) and (use_cc_spline==false)) {
			critical_curve_pts.clear();
			caustic_pts.clear();
			length_of_cc_cell.clear();
			sorted_critical_curves = false;
			sorted_critical_curve.clear();
		}
	}
	record_singular_points(zfacs); // grid cells will split around singular points (e.g. center of point mass, etc.)

	Grid::set_splitting(usplit_initial, wsplit_initial, splitlevels, cc_splitlevels, min_cell_area, cc_neighbor_splittings);
	Grid::set_enforce_min_area(enforce_min_cell_area);
	Grid::set_lens(this);
	find_automatic_grid_position_and_size(zfacs);
	double rmax = 0.5*dmax(grid_xlength,grid_ylength);

	if ((verbal) and (mpi_id==0)) cout << "Creating grid..." << flush;
	if (grid != NULL) {
		if (radial_grid)
			grid->redraw_grid(rmin_frac*rmax, rmax, grid_xcenter, grid_ycenter, 1, zfacs, betafacs); // setting grid_q to 1 for the moment...I will play with that later
		else
			grid->redraw_grid(grid_xcenter, grid_ycenter, grid_xlength, grid_ylength, zfacs, betafacs);
	} else {
		if (radial_grid)
			grid = new Grid(rmin_frac*rmax, rmax, grid_xcenter, grid_ycenter, 1, zfacs, betafacs); // setting grid_q to 1 for the moment...I will play with that later
		else
			grid = new Grid(grid_xcenter, grid_ycenter, grid_xlength, grid_ylength, zfacs, betafacs);
	}
	if ((subgrid_around_perturbers) and (nlens > 1)) {
		subgrid_around_perturber_galaxies(centers,einstein_radii,i_primary,zfacs,betafacs,redshift_index);
		delete[] centers;
		delete[] einstein_radii;
	}
	if ((auto_store_cc_points==true) and (use_cc_spline==false)) grid->store_critical_curve_pts();
	if ((verbal) and (mpi_id==0)) {
		cout << "done" << endl;
#ifdef USE_OPENMP
		if (show_wtime) {
			mytime=omp_get_wtime() - mytime0;
			if (mpi_id==0) cout << "Wall time for creating grid: " << mytime << endl;
		}
#endif
	}

	return true;
}

void QLens::find_automatic_grid_position_and_size(double *zfacs)
{
	if (autogrid_before_grid_creation) autogrid();
	else {
		if (autocenter==true) {
			lens_list[primary_lens_number]->get_center_coords(grid_xcenter,grid_ycenter);
		}
		if (auto_gridsize_from_einstein_radius==true) {
			double re_major, reav;
			re_major = einstein_radius_of_primary_lens(zfacs[lens_redshift_idx[primary_lens_number]],reav);
			if (re_major != 0.0) {
				double rmax = autogrid_frac*re_major;
				grid_xlength = 2*rmax;
				grid_ylength = 2*rmax;
				cc_rmax = rmax;
			}
		}
	}
}

void QLens::set_primary_lens()
{
	double re, re_avg, largest_einstein_radius = 0;
	int i;
	for (i=0; i < nlens; i++) {
		if (reference_zfactors[lens_redshift_idx[i]] != 0.0) {
			lens_list[i]->get_einstein_radius(re,re_avg,reference_zfactors[lens_redshift_idx[i]]);
			if (re > largest_einstein_radius) {
				largest_einstein_radius = re;
				primary_lens_number = i;
			}
		}
	}
}

void QLens::find_effective_lens_centers_and_einstein_radii(lensvector *centers, double *einstein_radii, int& i_primary, double *zfacs, double **betafacs, bool verbal)
{
	double zlprim, zlsub, re_avg;
	double largest_einstein_radius = 0;
	int i;
	i_primary = 0;
	for (i=0; i < nlens; i++) {
		if (zfacs[lens_redshift_idx[i]] != 0.0) {
			lens_list[i]->get_einstein_radius(einstein_radii[i],re_avg,zfacs[lens_redshift_idx[i]]);
			if (einstein_radii[i] > largest_einstein_radius) {
				largest_einstein_radius = einstein_radii[i];
				zlprim = lens_list[i]->zlens;
				i_primary = i;
			}
		}
	}
	if (largest_einstein_radius==0) {
		if ((mpi_id==0) and (verbal)) warn("could not find primary lens; Einstein radii all returned zero, setting primary to lens 0");
		zlprim = lens_list[0]->zlens;
		i_primary = 0;
	}

	for (i=0; i < nlens; i++) {
		if (zfacs[lens_redshift_idx[i]] != 0.0) {
			zlsub = lens_list[i]->zlens;
			if ((zlsub > zlprim) and (include_recursive_lensing)) {
				if ((lens_list[i]->get_specific_parameter("xc_l",centers[i][0])==false) or (lens_list[i]->get_specific_parameter("yc_l",centers[i][1])==false)) {
					if (find_lensed_position_of_background_perturber(verbal,i,centers[i],zfacs,betafacs)==false) {
						if (verbal) warn("cannot find lensed position of background perturber");
						lens_list[i]->get_center_coords(centers[i][0],centers[i][1]);
					}
				}
			} else {
				lens_list[i]->get_center_coords(centers[i][0],centers[i][1]);
			}
		}
	}
}

void QLens::subgrid_around_perturber_galaxies(lensvector *centers, double *einstein_radii, const int ihost, double *zfacs, double **betafacs, const int redshift_index)
{
	if (grid==NULL) {
		if (create_grid(false,zfacs,betafacs)==false) die("Could not create recursive grid");
	}
	if (nlens==0) { warn(warnings,"No galaxies in lens lens_list"); return; }
	double largest_einstein_radius, xch, ych;
	xch = centers[ihost][0];
	ych = centers[ihost][1];
	largest_einstein_radius = einstein_radii[ihost];

	double xc,yc;
	lensvector center;
	int parity, n_perturbers=0;
	double *kappas = new double[nlens];
	double *parities = new double[nlens];
	bool *exclude = new bool[nlens];
	bool *include_as_primary_perturber = new bool[nlens];
	bool *included_as_secondary_perturber = new bool[nlens];
	for (int i=0; i < nlens; i++) {
		include_as_primary_perturber[i] = false;
		included_as_secondary_perturber[i] = false;
		exclude[i] = false;
	}
	vector<int> excluded;
	int i,j,k;
	bool within_grid;
	double grid_xmin, grid_xmax, grid_ymin, grid_ymax;
	grid_xmin = grid_xcenter - grid_xlength/2;
	grid_xmax = grid_xcenter + grid_xlength/2;
	grid_ymin = grid_ycenter - grid_ylength/2;
	grid_ymax = grid_ycenter + grid_ylength/2;
	//for (i=0; i < nlens; i++) {
				//if ((i==primary_lens_number) or ((centers[i][0]==xch) and (centers[i][1]==ych)) or ((!use_perturber_flags) and (einstein_radii[i] >= 0) and (einstein_radii[i] >= perturber_einstein_radius_fraction*largest_einstein_radius))) exclude[i] = false;
//
	//}
	for (i=0; i < nlens; i++) {
		if (!included_as_secondary_perturber[i]) {
			excluded.clear();
			within_grid = false;
			xc = centers[i][0];
			yc = centers[i][1];
			if ((xc >= grid_xmin) and (xc <= grid_xmax) and (yc >= grid_ymin) and (yc <= grid_ymax)) within_grid = true;
			if (zfacs[lens_redshift_idx[i]] != 0.0) {
				// lenses with Einstein radii < some fraction of the largest Einstein radius, and not co-centered with the largest lens, are considered perturbers.

				if ((((!use_perturber_flags) and (lens_list[i]->has_kapavg_profile()) and (einstein_radii[i] < perturber_einstein_radius_fraction*largest_einstein_radius)) or ((use_perturber_flags) and (lens_list[i]->perturber==true))) and (lens_list[i]->has_kapavg_profile()) and (within_grid) and (i != primary_lens_number)) {
					if ((xc != xch) or (yc != ych)) {
						center[0]=xc;
						center[1]=yc;
						exclude[i] = true;
						for (k=i+1; k < nlens; k++) {
							if ((centers[k][0]==xc) and (centers[k][1]==yc)) {
								exclude[k] = true;
								excluded.push_back(k);
							}
						}
						kappas[i] = kappa_exclude(center,exclude,zfacs,betafacs);
						parities[i] = sign(magnification_exclude(center,exclude,zfacs,betafacs)); // use the parity to help determine approx. size of critical curves
						// galaxies in positive-parity regions where kappa > 1 will form no critical curves, so don't subgrid around these
						exclude[i] = false;
						for (k=0; k < excluded.size(); k++) exclude[excluded[k]] = false; // reset the exclude flags
						if ((parities[i]==1) and (kappas[i] >= 1.0)) continue;
						else {
							n_perturbers++;
							include_as_primary_perturber[i] = true;
							for (k=0; k < excluded.size(); k++) included_as_secondary_perturber[excluded[k]] = true; // reset the exclude flags
						}
					}
				}
			}
		}
	}
	lensvector *galcenter = new lensvector[n_perturbers];
	bool *subgrid = new bool[n_perturbers];
	double *subgrid_radius = new double[n_perturbers];
	double *min_galsubgrid_cellsize = new double[n_perturbers];
	
	for (j=0; j < n_perturbers; j++) subgrid[j] = false;
	double rmax, kappa_at_center;
	j=0;
	for (i=0; i < nlens; i++) {
		if (include_as_primary_perturber[i]) {
			excluded.clear();
			within_grid = false;
			xc = centers[i][0];
			yc = centers[i][1];
			if ((xc >= grid_xmin) and (xc <= grid_xmax) and (yc >= grid_ymin) and (yc <= grid_ymax)) within_grid = true;
			if (zfacs[lens_redshift_idx[i]] != 0.0) {
				//cout << "Perturber (lens " << i << ") at " << xc << " " << yc << endl;
				// lenses co-centered with the primary lens, no matter how small, are not considered perturbers unless flagged specifically
				kappa_at_center = kappas[i];
				parity = parities[i]; // use the parity to help determine approx. size of critical curves

				// galaxies in positive-parity regions where kappa > 1 will form no critical curves, so don't subgrid around these
				galcenter[j][0]=xc;
				galcenter[j][1]=yc;

				exclude[i] = true;
				for (k=i+1; k < nlens; k++) {
					if ((centers[k][0]==xc) and (centers[k][1]==yc)) {
						exclude[k] = true;
						excluded.push_back(k);
					}
				}
				if (calculate_perturber_subgridding_scale(i,exclude,ihost,false,centers[i],rmax,zfacs,betafacs)==false) {
					warn("Satellite subgridding failed (NaN shear calculated); this may be because two or more subhalos are at the same position");
					delete[] subgrid;
					delete[] kappas;
					delete[] exclude;
					delete[] include_as_primary_perturber;
					delete[] included_as_secondary_perturber;
					delete[] parities;
					delete[] galcenter;
					delete[] subgrid_radius;
					delete[] min_galsubgrid_cellsize;
					return;
				}
				exclude[i] = false;
				for (k=0; k < excluded.size(); k++) exclude[excluded[k]] = false; // reset the exclude flags

				subgrid_radius[j] = galsubgrid_radius_fraction*rmax;
				min_galsubgrid_cellsize[j] = SQR(galsubgrid_min_cellsize_fraction*rmax);
				if (rmax > 0) subgrid[j] = true;
				//cout << "Nj=" << j << " i=" << i << endl;
				j++;
			}
		}
	}
	if ((subgrid_only_near_data_images) and (source_redshift_groups.size() > 0)) {
		int zindx = redshift_index;
		if (zindx==-1) zindx = 0;
		int k;
		double distsqr, min_distsqr;
		for (j=0; j < n_perturbers; j++) {
			min_distsqr = 1e30;
			for (i=source_redshift_groups[zindx]; i < source_redshift_groups[zindx+1]; i++) {
				for (k=0; k < image_data[i].n_images; k++) {
					distsqr = SQR(image_data[i].pos[k][0] - galcenter[j][0]) + SQR(image_data[i].pos[k][1] - galcenter[j][1]);
					if (distsqr < min_distsqr) min_distsqr = distsqr;
				}
			}
			if (min_distsqr > SQR(subgrid_radius[j])) subgrid[j] = false;
		}
	}

	if (n_perturbers > 0)
		grid->subgrid_around_galaxies(galcenter,n_perturbers,subgrid_radius,min_galsubgrid_cellsize,galsubgrid_cc_splittings,subgrid);

	delete[] subgrid;
	delete[] kappas;
	delete[] exclude;
	delete[] include_as_primary_perturber;
	delete[] included_as_secondary_perturber;

	delete[] parities;
	delete[] galcenter;
	delete[] subgrid_radius;
	delete[] min_galsubgrid_cellsize;
}

bool QLens::calculate_perturber_subgridding_scale(int lens_number, bool* perturber_list, int host_lens_number, bool verbose, lensvector& center, double& rmax_numerical, double *zfacs, double **betafacs)
{
	perturber_lens_number = lens_number;
	linked_perturber_list = perturber_list;
	subgridding_zfacs = zfacs;
	subgridding_betafacs = betafacs;
	perturber_center[0]=center[0]; perturber_center[1]=center[1];

	double zlsub, zlprim;
	zlsub = lens_list[perturber_lens_number]->zlens;
	zlprim = lens_list[0]->zlens;

	double dum, b;
	lens_list[host_lens_number]->get_einstein_radius(dum,b,zfacs[lens_redshift_idx[host_lens_number]]);

	double shear_angle, shear_tot;
	shear_exclude(perturber_center,shear_tot,shear_angle,linked_perturber_list,zfacs,betafacs);
	if (shear_angle*0.0 != 0.0) return false;
	theta_shear = degrees_to_radians(shear_angle);
	theta_shear -= M_PI/2.0;

	double (Brent::*dthetac_eq)(const double);
	dthetac_eq = static_cast<double (Brent::*)(const double)> (&QLens::galaxy_subgridding_scale_equation);
	static const double rmin = 1e-6;
	double rmax_precision = 0.3*sqrt(min_cell_area);
	double bound = 0.4*b;

	bool found_rmax1, found_rmax2;
	double rmax1, rmax2;
	double rmax_pos, rmax_pos_center, rmax_neg, rmax_pos_noperturb;

	subgridding_include_perturber = true;
	subgridding_parity_at_center = 1;
	found_rmax1 = BrentsMethod(dthetac_eq, rmax1, rmin, bound, rmax_precision);
	found_rmax2 = BrentsMethod(dthetac_eq, rmax2, -bound, rmin, rmax_precision);
	if ((found_rmax1) and (found_rmax2)) rmax_pos = dmax(rmax1,rmax2);
	else if (found_rmax1) rmax_pos = rmax1;
	else if (found_rmax2) rmax_pos = rmax2;
	else rmax_pos = 0;

	// Now we compare to where the critical curve is *without* a perturber; if the difference is not great compared to the
	// distance of the perturber from the critical curve, then we don't subgrid around this region (although we might still
	// subgrid if there is a smaller radial critical curve produced, which would give a nonzero rmax_neg).
	subgridding_include_perturber = false;
	found_rmax1 = BrentsMethod(dthetac_eq, rmax1, rmin, bound, rmax_precision);
	found_rmax2 = BrentsMethod(dthetac_eq, rmax2, -bound, -rmin, rmax_precision);
	if ((found_rmax1) and (found_rmax2)) rmax_pos_noperturb = dmax(rmax1,rmax2);
	else if (found_rmax1) rmax_pos_noperturb = rmax1;
	else if (found_rmax2) rmax_pos_noperturb = rmax2;
	else rmax_pos_noperturb = 0;
	if (rmax_pos != 0) {
		double rmax_ratio = abs((rmax_pos-rmax_pos_noperturb)/rmax_pos);
		if (rmax_ratio > 0.5) rmax_pos = abs(rmax_pos);
		else rmax_pos = 0;
	}

	subgridding_parity_at_center = -1;
	subgridding_include_perturber = true;
	found_rmax1 = BrentsMethod(dthetac_eq, rmax1, rmin, bound, rmax_precision);
	found_rmax2 = BrentsMethod(dthetac_eq, rmax2, -bound, -rmin, rmax_precision);
	rmax2 = abs(rmax2);
	if ((found_rmax1) and (found_rmax2)) rmax_neg = dmax(rmax1,rmax2);
	else if (found_rmax1) rmax_neg = rmax1;
	else if (found_rmax2) rmax_neg = rmax2;
	else rmax_neg = 0;
	//cout << "rmax_pos=" << rmax_pos << ", rmax_neg=" << rmax_neg << endl;

	rmax_numerical = dmax(rmax_neg,rmax_pos);
	if (zlsub > zlprim) rmax_numerical *= 1.1; // in this regime, rmax is often a bit underestimated, so this helps counteract that
		//cout << "RMAX: " << rmax_numerical << endl;
	//if (rmax_numerical==0.0) warn("could not find rmax");
	return true;
}

double QLens::galaxy_subgridding_scale_equation(const double r)
{
	double kappa0, shear0, lambda0, shear_angle, perturber_avg_kappa;
	lensvector x;
	x[0] = perturber_center[0] + r*cos(theta_shear);
	x[1] = perturber_center[1] + r*sin(theta_shear);
	if (subgridding_parity_at_center < 0) {
		kappa0 = kappa_exclude(perturber_center,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
		shear_exclude(perturber_center,shear0,shear_angle,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
		lambda0 = 1 - kappa0 + shear0;
	} else {
		kappa0 = kappa_exclude(x,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
		shear_exclude(x,shear0,shear_angle,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
		lambda0 = 1 - kappa0 - shear0;
	}
	double r_eff = r;

	perturber_avg_kappa = 0;
	if (subgridding_include_perturber) {
		double zlsub, zlprim;
		zlsub = lens_list[perturber_lens_number]->zlens;
		zlprim = lens_list[0]->zlens;

		if (zlsub > zlprim) {
			lensvector xp, xpc;
			lens_list[perturber_lens_number]->get_center_coords(xpc[0],xpc[1]);
			double zsrc0 = source_redshift;
			//cout << "ZLSUB ZSRC: " << zlsub << " " << zsrc0 << endl;
			set_source_redshift(zlsub);
			lensvector alpha;
			// BUG!!!!!!! subgridding_zfacs is not updated by set_source_redshift
			deflection(x,alpha,subgridding_zfacs,subgridding_betafacs);
			set_source_redshift(zsrc0);
			xp[0] = x[0] - alpha[0];
			xp[1] = x[1] - alpha[1];
			r_eff = sqrt(SQR(xp[0]-xpc[0])+SQR(xp[1]-xpc[1]));
		} else {
			r_eff = r;
		}
		for (int i=0; i < nlens; i++) {
			if (linked_perturber_list[i]) perturber_avg_kappa += subgridding_zfacs[lens_redshift_idx[i]]*lens_list[i]->kappa_avg_r(r_eff);
		}
		if (subgridding_parity_at_center > 0) {
			if (zlsub < zlprim) {
				int i1,i2;
				i1 = lens_redshift_idx[primary_lens_number];
				i2 = lens_redshift_idx[perturber_lens_number];
				double beta = subgridding_betafacs[i1-1][i2];
				double dr = 1e-5;
				double kappa0_p, shear0_p;
				lensvector xp;
				xp[0] = perturber_center[0] + (r+dr)*cos(theta_shear);
				xp[1] = perturber_center[1] + (r+dr)*sin(theta_shear);
				kappa0_p = kappa_exclude(xp,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
				shear_exclude(xp,shear0_p,shear_angle,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
				double k0deriv = (kappa0_p+shear0_p-kappa0-shear0)/dr;
				double fac = 1 - beta*(kappa0 + shear0 + r*k0deriv);
				perturber_avg_kappa *= 1 - beta*(kappa0 + shear0 + r*k0deriv);
			} else if (zlsub > zlprim) {
				int i1,i2;
				i1 = lens_redshift_idx[primary_lens_number];
				i2 = lens_redshift_idx[perturber_lens_number];
				double beta = subgridding_betafacs[i2-1][i1];
				perturber_avg_kappa *= 1 - beta*(kappa0 + shear0);
			}
		}
	}

	return (lambda0 - perturber_avg_kappa);
}

void QLens::plot_shear_field(double xmin, double xmax, int nx, double ymin, double ymax, int ny, const string filename)
{
	int i, j, k;
	double x, y;
	double xstep = (xmax-xmin)/(nx-1);
	double ystep = (ymax-ymin)/(ny-1);
	double scale = 0.3*dmin(xstep,ystep);
	int compass_steps = 2;
	double compass_step = scale / (compass_steps-1);
	lensvector pos;
	double kapval,shearval,shear_angle,xp,yp,t;
	ofstream sout;
	open_output_file(sout,filename);
	for (i=0, x=xmin; i < nx; i++, x += xstep) {
		for (j=0, y=ymin; j < ny; j++, y += ystep) {
			pos[0]=x; pos[1]=y;
			shear(pos,shearval,shear_angle,0,reference_zfactors,default_zsrc_beta_factors);
			kapval = kappa(pos,reference_zfactors,default_zsrc_beta_factors);
			shearval /= (1-kapval); // reduced shear
			shear_angle *= M_PI/180.0;
			for (k=-compass_steps+1; k < compass_steps; k++)
			{
				t = k*compass_step;
				xp = x + t*cos(shear_angle);
				yp = y + t*sin(shear_angle);
				sout << xp << " " << yp << endl;
			}
			sout << endl;
		}
	}
	sout.close();
}

void QLens::plot_weak_lensing_shear_data(const bool include_model_shear, const string filename)
{
	double *zfacs = new double[n_lens_redshifts];
	int i, j, k;
	double x, y;
	double shearval,shear_angle,shear1,shear2,xp,yp,t;
	double model_shear1,model_shear2,model_shearval,model_shear_angle,xmp,ymp,tm;
	double xmin=1e30, xmax=-1e30, ymin=1e30, ymax=-1e30, min_shear=1e30, max_shear=-1e30;
	for (i=0; i < weak_lensing_data.n_sources; i++) {
		shear1 = weak_lensing_data.reduced_shear1[i];
		shear2 = weak_lensing_data.reduced_shear2[i];
		shearval = sqrt(shear1*shear1+shear2*shear2);
		x = weak_lensing_data.pos[i][0];
		y = weak_lensing_data.pos[i][1];
		if (x < xmin) xmin = x;
		if (x > xmax) xmax = x;
		if (y < ymin) ymin = y;
		if (y > ymax) ymax = y;
		if (shearval < min_shear) min_shear = shearval;
		if (shearval > max_shear) max_shear = shearval;
	}
	int nsteps_approx = (int) sqrt(weak_lensing_data.n_sources);
	double xstep = (xmax-xmin)/nsteps_approx;
	double ystep = (ymax-ymin)/nsteps_approx;
	double scale_factor = 1.7; // slightly enlarges the "arrows" so they're easier to see on the screen
	double scale = scale_factor*dmin(xstep,ystep)/2.0;
	double zsrc;
	int compass_steps = 2;
	double compass_step, model_compass_step;

	ofstream sout;
	open_output_file(sout,filename);
	for (i=0; i < weak_lensing_data.n_sources; i++) {
		x = weak_lensing_data.pos[i][0];
		y = weak_lensing_data.pos[i][1];
		zsrc = weak_lensing_data.zsrc[i];
		for (int i=0; i < n_lens_redshifts; i++) {
			zfacs[i] = kappa_ratio(lens_redshifts[i],zsrc,reference_source_redshift);
		}
		shear1 = weak_lensing_data.reduced_shear1[i];
		shear2 = weak_lensing_data.reduced_shear2[i];
		shearval = sqrt(shear1*shear1+shear2*shear2);
		shear_angle = atan(abs(shear2/shear1));
		compass_step = scale*(shearval/max_shear) / (compass_steps-1);
		if (shear1 < 0) {
			if (shear2 < 0)
				shear_angle = shear_angle - M_PI;
			else
				shear_angle = M_PI - shear_angle;
		} else if (shear2 < 0) {
			shear_angle = -shear_angle;
		}
		shear_angle *= 0.5;

		if (include_model_shear) {
			lensvector xvec(x,y);
			reduced_shear_components(xvec,model_shear1,model_shear2,0,zfacs);
			model_shearval = sqrt(model_shear1*model_shear1 + model_shear2*model_shear2);
			model_shear_angle = atan(abs(model_shear2/model_shear1));
			if (model_shear1 < 0) {
				if (model_shear2 < 0)
					model_shear_angle = model_shear_angle - M_PI;
				else
					model_shear_angle = M_PI - model_shear_angle;
			} else if (model_shear2 < 0) {
				model_shear_angle = -model_shear_angle;
			}
			model_shear_angle *= 0.5;
			model_compass_step = scale*(model_shearval/max_shear) / (compass_steps-1);
		}

		for (k=-compass_steps+1; k < compass_steps; k++)
		{
			t = k*compass_step;
			tm = k*model_compass_step;
			xp = x + t*cos(shear_angle);
			yp = y + t*sin(shear_angle);
			if (include_model_shear) {
				xmp = x + tm*cos(model_shear_angle);
				ymp = y + tm*sin(model_shear_angle);
				sout << xp << " " << yp << " " << xmp << " " << ymp << endl;
			} else {
				sout << xp << " " << yp << endl;
			}
		}
		sout << endl;
	}
	sout.close();
	delete[] zfacs;
}

/*
// The following function uses the series expansions derived in Minor et al. 2017, but it's better to simply use a root finder, so
// this approach is deprecated
void QLens::calculate_critical_curve_perturbation_radius(int lens_number, bool verbose, double &rmax, double& mass_enclosed)
{
	// the analytic formulas require a Pseudo-Jaffe or isothermal profile, and they only work for subhalos in the plane of the lens
	// if one of these conditions isn't satisfied, just use the numerical root-finding version instead
	if (((lens_list[lens_number]->get_lenstype()!=PJAFFE) and (lens_list[lens_number]->get_lenstype()!=ALPHA)) or (lens_list[lens_number]->zlens != lens_list[0]->zlens))
	{
		double avg_sigma_enclosed;
		calculate_critical_curve_perturbation_radius_numerical(lens_number,verbose,rmax,avg_sigma_enclosed,mass_enclosed);
		return;
	}
	//this assumes the host halo is lens number 0 (and is centered at the origin), and corresponding external shear (if present) is lens number 1
	double xc, yc, b, alpha, bs, rt, dum, q, shear_ext, phi, phi_0, phi_p, theta_s;
	double host_xc, host_yc;
	double reference_zfactor = reference_zfactors[lens_redshift_idx[lens_number]];
	lens_list[lens_number]->get_center_coords(xc,yc);
	lens_list[0]->get_center_coords(host_xc,host_yc);
	theta_s = sqrt(SQR(xc-host_xc) + SQR(yc-host_yc));
	phi = atan(abs((yc-host_yc)/(xc-host_xc)));
	if ((xc-host_xc) < 0) {
		if ((yc-host_yc) < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if ((yc-host_yc) < 0) {
		phi = -phi;
	}

	bool is_pjaffe;
	if (lens_list[lens_number]->get_lenstype()==PJAFFE) {
		is_pjaffe = true;
		double params[10];
		lens_list[lens_number]->get_parameters(params);
		bs = params[0];
		rt = params[1];
	} else {
		is_pjaffe = false;
		lens_list[lens_number]->get_einstein_radius(dum,bs,reference_zfactor);
	}
	double host_params[10];
	lens_list[0]->get_parameters(host_params);
	alpha = host_params[1];
	lens_list[0]->get_einstein_radius(dum,b,reference_zfactor);
	lens_list[0]->get_q_theta(q,phi_0);
	double gamma = alpha-1;
	double aprime = alpha;

	if (lens_list[1]->get_lenstype()==SHEAR) lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
	else { shear_ext = 0; phi_p=0; }
	if (LensProfile::orient_major_axis_north==true) {
		phi_0 += M_HALFPI;
		phi_p += M_HALFPI;
	}
	double shear_tot, sigma, eta, mu, delta, epsilon, xi, zeta, rmax_analytic, kappa0_at_sub, phi_normal_to_shear;
	sigma = 1.0/sqrt(q*SQR(cos(phi-phi_0)) + SQR(sin(phi-phi_0))/q);
	kappa0_at_sub = 0.5*(1-gamma)*pow(b*sigma/theta_s,1+gamma);

	double dphi, dphi_prime, gg;
	complex<double> hyp, complex_g;
	dphi = phi-phi_0;
	while (dphi <= -M_HALFPI) dphi += M_2PI;
	while (dphi >= M_2PI) dphi -= M_2PI;
	dphi_prime = atan(abs(tan(dphi)/q));
	if (dphi > M_HALFPI) { // dphi_prime must be in the same quadrant as dphi
		if (dphi <= M_PI) dphi_prime = M_PI - dphi_prime;
		else if (dphi <= 1.5*M_PI) dphi_prime += M_PI;
		else if (dphi <= M_2PI) dphi_prime = -dphi_prime;
	} else if (dphi < 0) dphi_prime = -dphi_prime;
	hyp = hyp_2F1(1.0,aprime/2.0,2.0-aprime/2.0,-(1-q)/(1+q)*polar(1.0,2*dphi_prime));
	complex_g = 1.0 - sqrt(q)*(4.0*(1-aprime)/((1+q)*(2-aprime)))*hyp/sigma*polar(1.0,dphi_prime-dphi);
	gg = sqrt(norm(complex_g));
	double gg_q = alpha/(2-alpha);
	if (verbose) cout << "shear/kappa = " << gg << " q=1 version: " << gg_q << endl;

	double cg = kappa0_at_sub*(1+gg);
	shear_tot = sqrt(SQR(gg*kappa0_at_sub) + shear_ext*shear_ext + 2*gg*kappa0_at_sub*shear_ext*cos(2*(phi-phi_p)));
	eta = 1 + gg*kappa0_at_sub - shear_tot;
	if (shear_ext==0) phi_normal_to_shear = phi;
	else phi_normal_to_shear = asin((gg*kappa0_at_sub*sin(2*phi)+shear_ext*sin(2*phi_p))/shear_tot) / 2;

	if (verbose) cout << "phi_normal = " << radians_to_degrees(phi_normal_to_shear) << ", phi = " << radians_to_degrees(phi) << ", dphi=" << radians_to_degrees(phi_normal_to_shear - phi) << endl;
	if (is_pjaffe) {
		double beta, bsq;
		double y=sqrt(theta_s*bs/(aprime*eta));
		if (verbose) cout << "y=" << y << endl;
		beta = rt/y;
		bsq = beta*beta;
		delta = 1 + 2*beta - 2*sqrt(1+bsq) + 1.0/sqrt(1+bsq);
		xi = beta-sqrt(1+bsq)+1.0/sqrt(1+bsq);
		epsilon = -(bs/y)*xi;
	} else {
		delta = 1;
		epsilon = 0;
	}


	double theta_on_cc = b*sigma*pow((1+gg)*(1-gamma)/(2*eta),1.0/(1+gamma));
	double dtheta = theta_s - theta_on_cc;
	//dtheta=0;
	mu = eta - epsilon + cg*gamma;
	zeta = 0.5*(bs*delta - (eta-epsilon-cg)*theta_s);
	rmax_analytic = (sqrt(bs*theta_s*delta*mu+SQR(zeta))+zeta)/mu;

	// Now for eta, we will use the formula for eta on the critical curve for the isothermal case, which works amazingly well
	double eta_on_cc_iso = (1-shear_ext*shear_ext)/(1 + shear_ext*cos(2*(phi-phi_p))); // isothermal
	//double eta_on_cc_not_iso = (1+gg*shear_ext*cos(2*(phi-phi_p)))*(-1 + sqrt(1 + (gg*gg-1)*(1-shear_ext*shear_ext)/SQR(1+gg*shear_ext*cos(2*(phi-phi_p)))))/(gg-1);
	double dtt = dtheta/theta_on_cc;
	dtt = dtt - (aprime+1)*dtt*dtt/2.0;
	mu = (aprime*eta_on_cc_iso)*(1 + (1-aprime)*dtt + (xi/(aprime*eta_on_cc_iso))*sqrt(bs/b));
	zeta = 0.5*(bs*delta - aprime*eta_on_cc_iso*theta_s*dtt - sqrt(b*bs)*(theta_s/b)*xi);
	double rmax_analytic2 = (1.0/mu)*(sqrt(bs*theta_s*delta*mu+SQR(zeta))+zeta);

	// for the approximate solutions on c.c., expanding subhalo deflection around sqrt(b*bs) seems to work better
	if (is_pjaffe) {
		double beta, bsq;
		beta = rt/sqrt(b*bs);
		bsq = beta*beta;
		delta = 1 + 2*beta - 2*sqrt(1+bsq) + 1.0/sqrt(1+bsq);
		xi = beta-sqrt(1+bsq)+1.0/sqrt(1+bsq);
		epsilon = -sqrt(bs/b)*xi;
	}

	// the next approximation assumes the subhalo is located on the (unperturbed) critical curve
	double eta_on_cc, xx, rmax_on_cc, rmax0;
	double lambda = pow(0.5*(1+gg)*(1-gamma)*pow(eta_on_cc_iso,gamma),1.0/(1+gamma));
	xx = sqrt(theta_s/(b*aprime*delta*eta_on_cc_iso));
	rmax_on_cc = delta*xx*(1-xi*xx/2.0+SQR(xx*xi)/8.0)*sqrt(b*bs) + (delta/(2*eta_on_cc_iso*aprime))*(1-xx*xi)*bs;

	// rough form for non-lens modelers to use
	double xxx = xi*sqrt(theta_s/(b*delta*aprime));
	rmax0 = sqrt(delta*theta_s*bs/aprime)*(1-xxx/2.0 + xxx*xxx/8.0) + delta*bs/(2*aprime)*(1-xxx);
	//double mass_rmax = M_PI*bs*(rmax_analytic - sqrt(bs*b + SQR(rmax_analytic)) + sqrt(bs*b))/aprime;

	double shear_angle, rmax_numerical, totshear;
	perturber_lens_number = lens_number;
	perturber_center[0]=xc; perturber_center[1]=yc;
	shear_exclude(perturber_center,totshear,shear_angle,perturber_lens_number,reference_zfactors,default_zsrc_beta_factors);
	theta_shear = degrees_to_radians(shear_angle);
	theta_shear -= M_PI/2.0;
	double (Brent::*dthetac_eq)(const double);
	dthetac_eq = static_cast<double (Brent::*)(const double)> (&QLens::subhalo_perturbation_radius_equation);
	double bound = 2*sqrt(b*bs);
	rmax_numerical = abs(BrentsMethod_Inclusive(dthetac_eq,-bound,bound,1e-5));
	double avg_kappa = reference_zfactors[lens_redshift_idx[perturber_lens_number]]*lens_list[perturber_lens_number]->kappa_avg_r(rmax_numerical);
	double zlsub, zlprim;
	zlsub = lens_list[perturber_lens_number]->zlens;
	zlprim = lens_list[0]->zlens;
	double menc = avg_kappa*M_PI*SQR(rmax_numerical)*sigma_crit_kpc(zlsub,reference_source_redshift);

	if (verbose) {
		cout << "direction of maximum warping = " << radians_to_degrees(theta_shear) << endl;
		cout << "theta_c=" << theta_on_cc << endl;
		cout << "dtheta/theta_c=" << (theta_s-theta_on_cc)/theta_on_cc << endl;
		//cout << "mu=" << mu << endl;
		//cout << "zeta=" << zeta << " zeta2=" << zeta2 << endl;
		//cout << "cosp = " << cos(2*(phi-phi_p)) << endl;
		//cout << "cfactor = " << cg*(1+gamma) << endl;
		cout << "eta = " << eta << ", eta_on_cc = " << eta_on_cc << ", eta_on_cc_iso = " << eta_on_cc_iso << endl;
		cout << "sigma = " << sigma << endl;
		cout << "lambda = " << lambda << endl;
		cout << "zeta = " << zeta << endl;
		cout << "theta_s  = " << theta_s << endl;
		cout << "theta_s (on c.c., approx) = " << theta_on_cc << endl << endl;
		cout << "rmax_numerical = " << rmax_numerical << endl;
		cout << "rmax_analytic = " << rmax_analytic << " (fractional error = " << (rmax_analytic-rmax_numerical)/rmax_numerical << ")" << endl;
		cout << "rmax_analytic_approx = " << rmax_analytic2 << " (fractional error = " << (rmax_analytic2-rmax_numerical)/rmax_numerical << ")" << endl;
		cout << "rmax (if on c.c.) = " << rmax_on_cc << " (fractional error = " << (rmax_on_cc-rmax_numerical)/rmax_numerical << ")" << endl;
		cout << "rmax (rough, if on c.c.) = " << rmax0 << " (fractional error = " << (rmax0-rmax_numerical)/rmax_numerical << ")" << endl;
		cout << "avg_kappa/alpha = " << avg_kappa/alpha << endl;
		cout << "mass_enclosed/alpha = " << menc/alpha << endl;
		cout << "mass_enclosed/alpha/eta = " << menc/alpha/eta << endl;
	}
	mass_enclosed = menc/alpha;
	rmax = rmax_analytic;
}
*/

bool QLens::find_lensed_position_of_background_perturber(bool verbal, int lens_number, lensvector& pos, double *zfacs, double **betafacs)
{
	double zlsub;
	zlsub = lens_list[lens_number]->zlens;
	lensvector perturber_center;
	lens_list[lens_number]->get_center_coords(perturber_center[0],perturber_center[1]);
	double zsrc0 = source_redshift;
	bool subgrid_setting = subgrid_around_perturbers;
	find_automatic_grid_position_and_size(zfacs);
	bool auto0 = auto_gridsize_from_einstein_radius;
	bool auto1 = autogrid_before_grid_creation;
	subgrid_around_perturbers = false;
	auto_gridsize_from_einstein_radius = false;
	autogrid_before_grid_creation = false;
	set_source_redshift(zlsub);
	create_grid(false,zfacs,betafacs);
	int n_images, img_i;
	image *img = get_images(perturber_center, n_images, false);
	if (n_images == 0) {
		reset_grid();
		set_source_redshift(zsrc0);
		return false;
	}
	img_i = 0;
	if (n_images > 1) {
		if ((mpi_id==0) and (verbal)) {
			warn("Well this is interesting. Perturber maps to more than one place in the primary lens plane! Using image furthest from primary lens center");
			cout << "Positions of lensed perturber:\n";
		}
		double rsq, rsqmax=-1e30;
		double xc0, yc0;
		lens_list[primary_lens_number]->get_center_coords(xc0,yc0);
		for (int ii=0; ii < n_images; ii++) {
			rsq = SQR(img[ii].pos[0]-xc0) + SQR(img[ii].pos[1]-yc0);
			if (rsq > rsqmax) {
				rsqmax = rsq;
				img_i = ii;
			}
			if ((mpi_id==0) and (verbal)) cout << img[ii].pos[0] << " " << img[ii].pos[1] << endl;
		}
	}
	set_source_redshift(zsrc0);
	subgrid_around_perturbers = subgrid_setting;
	auto_gridsize_from_einstein_radius = false;
	autogrid_before_grid_creation = false;
	reset_grid();
	pos[0] = img[img_i].pos[0];
	pos[1] = img[img_i].pos[1];
	return true;
}

bool QLens::calculate_critical_curve_perturbation_radius_numerical(int lens_number, bool verbal, double& rmax_numerical, double& avg_sigma_enclosed, double& mass_enclosed, double& rmax_perturber_z, double &avgkap_scaled_to_primary_lensplane, bool subtract_unperturbed)
{
	perturber_lens_number = lens_number;
	bool *perturber_list = new bool[nlens];
	for (int i=0; i < nlens; i++) perturber_list[i] = false;
	perturber_list[lens_number] = true;
	linked_perturber_list = perturber_list;
	double xc, yc, host_xc, host_yc, b, dum, alpha, shear_ext, phi, phi_p;

	double zlsub, zlprim;
	zlsub = lens_list[perturber_lens_number]->zlens;
	zlprim = lens_list[primary_lens_number]->zlens;

	double reference_zfactor = reference_zfactors[lens_redshift_idx[perturber_lens_number]];
	if (zlsub > zlprim) {
		if ((lens_list[lens_number]->get_specific_parameter("xc_l",perturber_center[0])==false) or (lens_list[lens_number]->get_specific_parameter("yc_l",perturber_center[1])==false)) {
			if (find_lensed_position_of_background_perturber(verbal,lens_number,perturber_center,reference_zfactors,default_zsrc_beta_factors)==false) {
				warn("could not find lensed position of background perturber");
				delete[] perturber_list;
				return false;
			}
		}
		xc = perturber_center[0];
		yc = perturber_center[1];
		if ((mpi_id==0) and (verbal)) cout << "Perturber located at (" << xc << "," << yc << ") in primary lens plane\n";
	} else {
		lens_list[perturber_lens_number]->get_center_coords(xc,yc);
		perturber_center[0]=xc; perturber_center[1]=yc;
	}

	lens_list[primary_lens_number]->get_center_coords(host_xc,host_yc);
	phi = atan(abs((yc-host_yc)/(xc-host_xc)));
	if ((xc-host_xc) < 0) {
		if ((yc-host_yc) < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if ((yc-host_yc) < 0) {
		phi = -phi;
	}

	if ((primary_lens_number < (nlens-1)) and (lens_list[primary_lens_number+1]->get_lenstype()==SHEAR)) lens_list[primary_lens_number+1]->get_q_theta(shear_ext,phi_p); // assumes that if there is external shear present, it comes after the primary lens in the lens list
	else { shear_ext = 0; phi_p=0; }
	if (LensProfile::orient_major_axis_north==true) {
		phi_p += M_HALFPI;
	}

	double shear_angle, shear_tot;
	shear_exclude(perturber_center,shear_tot,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
	if (shear_angle*0.0 != 0.0) {
		warn("could not calculate shear at position of perturber");
		rmax_numerical = 0.0;
		mass_enclosed = 0.0;
		delete[] perturber_list;
		return false;
	}

	lens_list[primary_lens_number]->get_einstein_radius(dum,b,reference_zfactors[lens_redshift_idx[primary_lens_number]]);

	theta_shear = degrees_to_radians(shear_angle);
	theta_shear -= M_PI/2.0;
	double (Brent::*dthetac_eq)(const double);
	dthetac_eq = static_cast<double (Brent::*)(const double)> (&QLens::subhalo_perturbation_radius_equation);
	double bound = 0.6*b;
	rmax_numerical = BrentsMethod_Inclusive(dthetac_eq,-bound,bound,1e-5,verbal);
	//if ((rmax_numerical==bound) or (rmax_numerical==-bound)) {
		//rmax_numerical = 0.0; // subhalo too far from critical curve to cause a meaningful "local" perturbation
		//mass_enclosed = 0.0;
		//delete[] perturber_list;
		//return true;
	//}
	if (zlsub > zlprim) {
		lensvector x;
		x[0] = perturber_center[0] + rmax_numerical*cos(theta_shear);
		x[1] = perturber_center[1] + rmax_numerical*sin(theta_shear);
		lensvector xp, xpc;
		lens_list[perturber_lens_number]->get_center_coords(xpc[0],xpc[1]);
		double zsrc0 = source_redshift;
		set_source_redshift(zlsub);
		lensvector defp;
		deflection(x,defp,reference_zfactors,default_zsrc_beta_factors);
		set_source_redshift(zsrc0);
		xp[0] = x[0] - defp[0];
		xp[1] = x[1] - defp[1];
		rmax_perturber_z = sqrt(SQR(xp[0]-xpc[0])+SQR(xp[1]-xpc[1]));
	} else rmax_perturber_z = abs(rmax_numerical);

	double avg_kappa = reference_zfactors[lens_redshift_idx[perturber_lens_number]]*lens_list[perturber_lens_number]->kappa_avg_r(rmax_perturber_z);

	if (lens_list[primary_lens_number]->lenstype==ALPHA) {
		double host_params[10];
		lens_list[primary_lens_number]->get_parameters(host_params);
		alpha = host_params[1];
	} else {
		alpha = 1.0;
	}

	double subhalo_rc = sqrt(SQR(perturber_center[0]-host_xc)+SQR(perturber_center[1]-host_yc));

	double theta_c_unperturbed, rmax_relative = rmax_numerical;
	if ((subtract_unperturbed) or (verbal)) {
		double (Brent::*dthetac_eq_nosub)(const double);
		dthetac_eq_nosub = static_cast<double (Brent::*)(const double)> (&QLens::perturbation_radius_equation_nosub);
		theta_c_unperturbed = BrentsMethod_Inclusive(dthetac_eq_nosub,-bound,bound,1e-5,verbal);
		rmax_relative = abs(rmax_numerical-theta_c_unperturbed);
		if (subtract_unperturbed) rmax_numerical = rmax_relative;
		theta_c_unperturbed += subhalo_rc; // now it's actually theta_c relative to the primary lens center
	}
	//double delta_s_over_thetac = subhalo_rc/theta_c_unperturbed - 1;
	//double r_over_rc = abs(rmax_numerical)/(subhalo_rc + abs(rmax_numerical));
	//double ktilde_approx = 1 - alpha*(delta_s_over_thetac + r_over_rc);
	//double ktilde_approx2 = 1 - alpha*(r_over_rc);
		//double blergh2_approx = 1 - alpha*(delta_s_over_thetac + 2*r_over_rc);

	double kpc_to_arcsec_sub = 206.264806/angular_diameter_distance(zlsub);
	// the following quantities are scaled by 1/alpha
	avg_sigma_enclosed = avg_kappa*sigma_crit_kpc(zlsub,reference_source_redshift);
	mass_enclosed = avg_sigma_enclosed*M_PI*SQR(rmax_numerical/kpc_to_arcsec_sub);

	double menc_scaled_to_primary_lensplane = 0;
	avgkap_scaled_to_primary_lensplane = 0;
	double k0deriv=0;
	//double avgkap_scaled2 = 0;
	double kappa0;
	if (include_recursive_lensing) {
		if (zlsub < zlprim) {
			//double kappa0, shear_tot, shear_angle;
			lensvector x;
			x[0] = perturber_center[0] + rmax_numerical*cos(theta_shear);
			x[1] = perturber_center[1] + rmax_numerical*sin(theta_shear);
			kappa0 = kappa_exclude(x,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(x,shear_tot,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);

			int i1,i2;
			i1 = lens_redshift_idx[primary_lens_number];
			i2 = lens_redshift_idx[perturber_lens_number];
			double beta = default_zsrc_beta_factors[i1-1][i2];
			double dr = 1e-5;
			double kappa0_p, shear_tot_p;
			lensvector xp;
			xp[0] = perturber_center[0] + (rmax_numerical+dr)*cos(theta_shear);
			xp[1] = perturber_center[1] + (rmax_numerical+dr)*sin(theta_shear);
			kappa0_p = kappa_exclude(xp,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(xp,shear_tot_p,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			double kappa0_m, shear_tot_m;
			lensvector xm;
			xm[0] = perturber_center[0] + (rmax_numerical-dr)*cos(theta_shear);
			xm[1] = perturber_center[1] + (rmax_numerical-dr)*sin(theta_shear);
			kappa0_m = kappa_exclude(xm,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(xm,shear_tot_m,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			k0deriv = (kappa0_p+shear_tot_p-kappa0_m-shear_tot_m)/(2*dr);
			double mass_scale_factor = (sigma_crit_kpc(zlprim,reference_source_redshift) / sigma_crit_kpc(zlsub,reference_source_redshift))*SQR(rmax_numerical/rmax_perturber_z)*(1 - beta*(kappa0 + shear_tot + rmax_numerical*k0deriv));
			//double fac1 = (sigma_crit_kpc(zlprim,reference_source_redshift) / sigma_crit_kpc(zlsub,reference_source_redshift));
			//double fac2 = (1 - beta*(kappa0 + shear_tot + rmax_numerical*k0deriv));
			//cout << fac1 << " " << fac2 << " " << mass_scale_factor << endl;
			menc_scaled_to_primary_lensplane = mass_enclosed*mass_scale_factor;
			avgkap_scaled_to_primary_lensplane = avg_kappa*(1-beta*(kappa0+shear_tot+abs(rmax_numerical)*k0deriv));

			//double ktilde = kappa0+shear_tot;
			//double blergh = abs(rmax_numerical)*k0deriv;
			//double blergh2 = ktilde + blergh;
			//double blergh2_approx0 = 1 - alpha*(delta_s_over_thetac + 2*r_over_rc);
			//double blergh2_approx = 1 - alpha*(2*r_over_rc);
			//avgkap_scaled2 = avg_kappa*(1-beta*blergh2_approx);
			//cout << "r*k0deriv=" << blergh << endl;
			//cout << "blergh=" << blergh2 << " approx=" << blergh2_approx << " better_approx=" << blergh2_approx0 << endl;
		} else if (zlsub > zlprim) {
			//double kappa0, shear_tot, shear_angle;
			lensvector x;
			x[0] = perturber_center[0] + rmax_numerical*cos(theta_shear);
			x[1] = perturber_center[1] + rmax_numerical*sin(theta_shear);
			kappa0 = kappa_exclude(x,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(x,shear_tot,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			int i1,i2;
			i1 = lens_redshift_idx[primary_lens_number];
			i2 = lens_redshift_idx[perturber_lens_number];
			double beta = default_zsrc_beta_factors[i2-1][i1];
			double mass_scale_factor = (sigma_crit_kpc(zlprim,reference_source_redshift) / sigma_crit_kpc(zlsub,reference_source_redshift))*(1 - beta*(kappa0 + shear_tot));
			menc_scaled_to_primary_lensplane = mass_enclosed*mass_scale_factor;
			avgkap_scaled_to_primary_lensplane = avg_kappa*(1-beta*(kappa0+shear_tot));

			//double ktilde = kappa0+shear_tot;
			//double blergh2 = ktilde;
			//double blergh2_approx = 1 - alpha*(r_over_rc);
			//double blergh2_approx0 = 1 - alpha*(delta_s_over_thetac + r_over_rc);
			//avgkap_scaled2 = avg_kappa*(1-beta*blergh2_approx);
			//cout << "blergh=" << blergh2 << " approx=" << blergh2_approx << " better_approx=" << blergh2_approx0 << endl;
		}
	} else {
		double mass_scale_factor = (sigma_crit_kpc(zlprim,reference_source_redshift) / sigma_crit_kpc(zlsub,reference_source_redshift))*SQR(rmax_numerical/rmax_perturber_z);
		menc_scaled_to_primary_lensplane = mass_enclosed*mass_scale_factor;
		avgkap_scaled_to_primary_lensplane = avg_kappa;
	}
	//if (mpi_id==0) cout << "CHECK0: " << rmax_numerical << " " << rmax_perturber_z << " " << avg_kappa << " " << avgkap_scaled_to_primary_lensplane << " ... " << kappa0 << " " << shear_tot << " " << k0deriv << endl;

	//double avgkap_check,menc_check;
	//get_perturber_avgkappa_scaled(lens_number,rmax_numerical,avgkap_check,menc_check);
	double rmax_kpc = rmax_numerical/kpc_to_arcsec_sub;

	if ((mpi_id==0) and (verbal)) {
		lensvector x;
		x[0] = perturber_center[0] + rmax_numerical*cos(theta_shear);
		x[1] = perturber_center[1] + rmax_numerical*sin(theta_shear);
		cout << "direction of maximum warping = " << radians_to_degrees(theta_shear) << endl;
		cout << "rmax_numerical = " << rmax_numerical << " (rmax_kpc=" << rmax_kpc << ")" << endl;
		cout << "rmax_relative = " << rmax_relative << endl;
		cout << "subhalo_rc = " << subhalo_rc << endl;
		cout << "thetac = " << theta_c_unperturbed << endl;
		//cout << "delta: " << delta_s_over_thetac << endl;
		//cout << "r_over_rc: " << r_over_rc << endl;
		//cout << "ktilde: " << ktilde << endl;
		cout << "rmax location: (" << x[0] << "," << x[1] << ")\n";
		if (zlsub > zlprim) cout << "rmax_perturber_z = " << rmax_perturber_z << endl;
		cout << "avg_kappa/alpha = " << avg_kappa/alpha << endl;
		if (avgkap_scaled_to_primary_lensplane != 0) {
			//double avgkaperr = (avgkap_scaled_to_primary_lensplane-avgkap_scaled2)/avgkap_scaled_to_primary_lensplane;
			cout << "avg_kappa(primary_lens_plane)/alpha = " << avgkap_scaled_to_primary_lensplane/alpha << endl;
			//cout << "avg_kappa_approx(primary_lens_plane)/alpha = " << avgkap_scaled2/alpha << " (err=" << avgkaperr << ")" << endl;
		}
		cout << "avg_sigma_enclosed = " << avg_sigma_enclosed << endl;
		cout << "mass_enclosed = " << mass_enclosed << endl;
		if (menc_scaled_to_primary_lensplane != 0) cout << "mass(primary_lens_plane) = " << menc_scaled_to_primary_lensplane << endl;
	}
	//rmax_numerical = abs(rmax_numerical);
	delete[] perturber_list;
	return true;
}

void QLens::get_perturber_avgkappa_scaled(int lens_number, const double r0, double &avgkap_scaled, double &menc_scaled, double &avgkap0, bool verbal)
{
	bool *perturber_list = new bool[nlens];
	for (int i=0; i < nlens; i++) perturber_list[i] = false;
	perturber_list[lens_number] = true;

	double zlsub, zlprim;
	zlsub = lens_list[lens_number]->zlens;
	zlprim = lens_list[primary_lens_number]->zlens;

	if (zlsub > zlprim) {
		if ((lens_list[lens_number]->get_specific_parameter("xc_l",perturber_center[0])==false) or (lens_list[lens_number]->get_specific_parameter("yc_l",perturber_center[1])==false)) {
			if (find_lensed_position_of_background_perturber(verbal,lens_number,perturber_center,reference_zfactors,default_zsrc_beta_factors)==false) {
				warn("could not find lensed position of background perturber");
				delete[] perturber_list;
				die();
			}
		}
	} else {
		lens_list[perturber_lens_number]->get_center_coords(perturber_center[0],perturber_center[1]);
	}


	double kappa0, shear_tot, shear_angle;
	lensvector x;
	x[0] = perturber_center[0] + r0*cos(theta_shear);
	x[1] = perturber_center[1] + r0*sin(theta_shear);
	kappa0 = kappa_exclude(x,perturber_list,reference_zfactors,default_zsrc_beta_factors);
	shear_exclude(x,shear_tot,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);
	avgkap0 = 1 - kappa0 - shear_tot;

	double r;
	if ((zlsub > zlprim) and (include_recursive_lensing)) {
		lensvector x;
		x[0] = perturber_center[0] + r0*cos(theta_shear);
		x[1] = perturber_center[1] + r0*sin(theta_shear);
		lensvector xp, xpc;
		lens_list[lens_number]->get_center_coords(xpc[0],xpc[1]);
		double zsrc0 = source_redshift;
		set_source_redshift(zlsub);
		lensvector defp;
		deflection(x,defp,reference_zfactors,default_zsrc_beta_factors);
		set_source_redshift(zsrc0);
		xp[0] = x[0] - defp[0];
		xp[1] = x[1] - defp[1];
		r = sqrt(SQR(xp[0]-xpc[0])+SQR(xp[1]-xpc[1]));
	} else r = abs(r0);

	double avg_kappa = reference_zfactors[lens_redshift_idx[lens_number]]*lens_list[lens_number]->kappa_avg_r(r);

	double avg_sigma_enclosed = avg_kappa*sigma_crit_arcsec(zlsub,reference_source_redshift);
	double mass_enclosed = avg_sigma_enclosed*M_PI*SQR(r0);

	menc_scaled = 0;
	avgkap_scaled = 0;
	double k0deriv=0;
	if (include_recursive_lensing) {
		if (zlsub < zlprim) {
			//double kappa0, shear_tot, shear_angle;
			//lensvector x;
			//x[0] = perturber_center[0] + r0*cos(theta_shear);
			//x[1] = perturber_center[1] + r0*sin(theta_shear);
			//kappa0 = kappa_exclude(x,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			//shear_exclude(x,shear_tot,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);

			int i1,i2;
			i1 = lens_redshift_idx[primary_lens_number];
			i2 = lens_redshift_idx[lens_number];
			double beta = default_zsrc_beta_factors[i1-1][i2];
			double dr = 1e-5;
			double kappa0_p, shear_tot_p;
			lensvector xp;
			xp[0] = perturber_center[0] + (r0+dr)*cos(theta_shear);
			xp[1] = perturber_center[1] + (r0+dr)*sin(theta_shear);
			kappa0_p = kappa_exclude(xp,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(xp,shear_tot_p,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			double kappa0_m, shear_tot_m;
			lensvector xm;
			xm[0] = perturber_center[0] + (r0-dr)*cos(theta_shear);
			xm[1] = perturber_center[1] + (r0-dr)*sin(theta_shear);
			kappa0_m = kappa_exclude(xm,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(xm,shear_tot_m,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			k0deriv = (kappa0_p+shear_tot_p-kappa0_m-shear_tot_m)/(2*dr);
			double mass_scale_factor = (sigma_crit_kpc(zlprim,reference_source_redshift) / sigma_crit_kpc(zlsub,reference_source_redshift))*SQR(r0/r)*(1 - beta*(kappa0 + shear_tot + r0*k0deriv));
			//double fac1 = (sigma_crit_kpc(zlprim,reference_source_redshift) / sigma_crit_kpc(zlsub,reference_source_redshift));
			//double fac2 = (1 - beta*(kappa0 + shear_tot + r0*k0deriv));
			//cout << fac1 << " " << fac2 << " " << mass_scale_factor << endl;
			menc_scaled = mass_enclosed*mass_scale_factor;
			avgkap_scaled = avg_kappa*(1-beta*(kappa0+shear_tot+abs(r0)*k0deriv));
		} else if (zlsub > zlprim) {
			//double kappa0, shear_tot, shear_angle;
			//lensvector x;
			//x[0] = perturber_center[0] + r0*cos(theta_shear);
			//x[1] = perturber_center[1] + r0*sin(theta_shear);
			//kappa0 = kappa_exclude(x,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			//shear_exclude(x,shear_tot,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			int i1,i2;
			i1 = lens_redshift_idx[primary_lens_number];
			i2 = lens_redshift_idx[lens_number];
			double beta = default_zsrc_beta_factors[i2-1][i1];
			double mass_scale_factor = (sigma_crit_kpc(zlprim,reference_source_redshift) / sigma_crit_kpc(zlsub,reference_source_redshift))*(1 - beta*(kappa0 + shear_tot));
			menc_scaled = mass_enclosed*mass_scale_factor;
			avgkap_scaled = avg_kappa*(1-beta*(kappa0+shear_tot));
		}
	} else {
		double mass_scale_factor = (sigma_crit_kpc(zlprim,reference_source_redshift) / sigma_crit_kpc(zlsub,reference_source_redshift));
		menc_scaled = mass_enclosed*mass_scale_factor;
		avgkap_scaled = avg_kappa;
	}
	if ((verbal) and (mpi_id==0)) cout << "CHECK: " << r0 << " " << r << " " << avg_kappa << " " << avgkap_scaled << " ... " << kappa0 << " " << shear_tot << " " << k0deriv << endl;

	delete[] perturber_list;
}


double QLens::subhalo_perturbation_radius_equation(const double r)
{
	double kappa0, shear0, shear_angle, subhalo_avg_kappa;
	lensvector x;
	x[0] = perturber_center[0] + r*cos(theta_shear);
	x[1] = perturber_center[1] + r*sin(theta_shear);
	kappa0 = kappa_exclude(x,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
	shear_exclude(x,shear0,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);

	double zlsub, zlprim;
	zlsub = lens_list[perturber_lens_number]->zlens;
	zlprim = lens_list[0]->zlens;

	double r_eff;
	if (zlsub > zlprim) {
		lensvector xp, xpc;
		lens_list[perturber_lens_number]->get_center_coords(xpc[0],xpc[1]);
		double zsrc0 = source_redshift;
		set_source_redshift(zlsub);
		lensvector alpha;
		deflection(x,alpha,reference_zfactors,default_zsrc_beta_factors);
		set_source_redshift(zsrc0);
		xp[0] = x[0] - alpha[0];
		xp[1] = x[1] - alpha[1];
		r_eff = sqrt(SQR(xp[0]-xpc[0])+SQR(xp[1]-xpc[1]));
	} else {
		r_eff = r;
	}
	subhalo_avg_kappa = 0;
	for (int i=0; i < nlens; i++) {
		if (linked_perturber_list[i]) subhalo_avg_kappa += reference_zfactors[lens_redshift_idx[i]]*lens_list[i]->kappa_avg_r(r_eff);
	}
	if (zlsub < zlprim) {
		int i1,i2;
		i1 = lens_redshift_idx[primary_lens_number];
		i2 = lens_redshift_idx[perturber_lens_number];
		double beta = default_zsrc_beta_factors[i1-1][i2];
		double dr = 1e-5;
		double kappa0_p, shear0_p;
		lensvector xp;
		xp[0] = perturber_center[0] + (r+dr)*cos(theta_shear);
		xp[1] = perturber_center[1] + (r+dr)*sin(theta_shear);
		kappa0_p = kappa_exclude(xp,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		shear_exclude(xp,shear0_p,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		double kappa0_m, shear0_m;
		lensvector xm;
		xm[0] = perturber_center[0] + (r-dr)*cos(theta_shear);
		xm[1] = perturber_center[1] + (r-dr)*sin(theta_shear);
		kappa0_m = kappa_exclude(xm,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		shear_exclude(xm,shear0_m,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		double k0deriv = (kappa0_p+shear0_p-kappa0_m-shear0_m)/(2*dr);
		subhalo_avg_kappa *= 1 - beta*(kappa0 + shear0 + r*k0deriv);
	} else if (zlsub > zlprim) {
		int i1,i2;
		i1 = lens_redshift_idx[primary_lens_number];
		i2 = lens_redshift_idx[perturber_lens_number];
		double beta = default_zsrc_beta_factors[i2-1][i1];
		subhalo_avg_kappa *= 1 - beta*(kappa0 + shear0);
	}
	return (1 - kappa0 - shear0 - subhalo_avg_kappa);
}

double QLens::perturbation_radius_equation_nosub(const double r)
{
	double kappa0, shear0, shear_angle;
	lensvector x;
	x[0] = perturber_center[0] + r*cos(theta_shear);
	x[1] = perturber_center[1] + r*sin(theta_shear);
	kappa0 = kappa_exclude(x,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
	shear_exclude(x,shear0,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
	return (1 - kappa0 - shear0);
}

bool QLens::get_einstein_radius(int lens_number, double& re_major_axis, double& re_average)
{
	if (lens_number >= nlens) { warn("lens %i has not been created",lens_number); return false; }
	lens_list[lens_number]->get_einstein_radius(re_major_axis,re_average,reference_zfactors[lens_redshift_idx[lens_number]]);
	return true;
}

double QLens::inverse_magnification_r(const double r)
{
	lensmatrix jac;
	hessian(grid_xcenter + r*cos(theta_crit), grid_ycenter + r*sin(theta_crit), jac, 0, reference_zfactors, default_zsrc_beta_factors);
	jac[0][0] = 1 - jac[0][0];
	jac[1][1] = 1 - jac[1][1];
	jac[0][1] = -jac[0][1];
	jac[1][0] = -jac[1][0];
	return determinant(jac);
}

double QLens::source_plane_r(const double r)
{
	lensvector x,def;
	x[0] = grid_xcenter + r*cos(theta_crit);
	x[1] = grid_ycenter + r*sin(theta_crit);
	find_sourcept(x,def,0,reference_zfactors,default_zsrc_beta_factors);
	def[0] -= grid_xcenter; // this assumes the deflection is approximately zero at the center of the grid (roughly true if any perturber gal's are small)
	def[1] -= grid_ycenter;
	return def.norm();
}

void QLens::create_deflection_spline(int steps)
{
	spline_deflection(0.5*grid_xlength*spline_frac, 0.5*grid_ylength*spline_frac, steps);
}

void QLens::spline_deflection(double xl, double yl, int steps)
{
	dvector xtable(steps+1);
	dvector ytable(steps+1);
	dmatrix defxmatrix(steps+1);
	dmatrix defymatrix(steps+1);
	dmatrix defxxmatrix(steps+1);
	dmatrix defyymatrix(steps+1);
	dmatrix defxymatrix(steps+1);

	double xmin, xmax, ymin, ymax;
	xmin = -xl; xmax = xl;
	ymin = -yl; ymax = yl;
	double x, y, xstep, ystep;
	xstep = (xmax-xmin)/steps;
	ystep = (ymax-ymin)/steps;

	int i, j;
	lensvector def;
	lensmatrix hess;
	for (i=0, x=xmin; i <= steps; i++, x += xstep) {
		xtable[i] = x;
		for (j=0, y=ymin; j <= steps; j++, y += ystep) {
			if (i==0) ytable[j] = y;		// Only needs to be done the first time around (hence "if i==0")
			deflection(x,y,def,0,reference_zfactors,default_zsrc_beta_factors);
			hessian(x,y,hess,0,reference_zfactors,default_zsrc_beta_factors);
			defxmatrix[i][j] = def[0];
			defymatrix[i][j] = def[1];
			defxxmatrix[i][j] = hess[0][0];
			defyymatrix[i][j] = hess[1][1];
			defxymatrix[i][j] = hess[0][1];
		}
	}

	if (defspline) delete defspline; // delete previous spline
	defspline = new Defspline;
	defspline->ax.input(xtable, ytable, defxmatrix);
	defspline->ay.input(xtable, ytable, defymatrix);
	defspline->axx.input(xtable, ytable, defxxmatrix);
	defspline->ayy.input(xtable, ytable, defyymatrix);
	defspline->axy.input(xtable, ytable, defxymatrix);
}

bool QLens::get_deflection_spline_info(double &xmax, double &ymax, int &nsteps)
{
	if (!defspline) return false;
	xmax = defspline->xmax();
	ymax = defspline->ymax();
	nsteps = defspline->nsteps();
	return true;
}

bool QLens::unspline_deflection()
{
	if (!defspline) return false;
	delete defspline;
	defspline = NULL;
	return true;
}

bool QLens::autospline_deflection(int steps)
{
	double (Brent::*mag_r)(const double);
	mag_r = static_cast<double (Brent::*)(const double)> (&QLens::inverse_magnification_r);

	double mag0, mag1, root0, root1x, root1y;
	bool found_first_root;
	double r, rstep, step_increment, step_increment_change;
	rstep = cc_rmin;
	step_increment = 1.1;
	step_increment_change = 0.5;
	int i;
	for (i=0, theta_crit=0; i < 2; i++, theta_crit += M_PI/2)  // just samples point on x-axis and y-axis
	{
		for (;;)
		{
			mag1 = inverse_magnification_r(rstep);
			found_first_root = false;
			for (r=rstep; r < cc_rmax; r += ((rstep *= step_increment)/step_increment))
			{
				mag0 = mag1;
				mag1 = inverse_magnification_r(r+rstep);
				if (mag0*mag1 < 0) {
					if (!found_first_root) {
						root0 = BrentsMethod(mag_r, r, r+rstep, 1e-3);
							found_first_root = true;
					} else {
						if (i==0) root1x = BrentsMethod(mag_r, r, r+rstep, 1e-3);
						if (i==1) root1y = BrentsMethod(mag_r, r, r+rstep, 1e-3);
						break;
					}
				}
			}
			if (r >= cc_rmax) {
				if (cc_rmin > 1e-5) cc_rmin = ((cc_rmin/10) > 1e-5) ? cc_rmin/10 : 1e-5;
				else step_increment = 1 + (step_increment-1)*step_increment_change;
				cc_rmax *= 1.5;
				rstep = cc_rmin;
			} else {
				if (i==0)	// adjust the scale of rstep if it is too small
				{
					rstep = cc_rmin;
					double rmin_frac, cc0_max_rfrac_range;
					rmin_frac = root0 / cc_rmin;
					cc0_max_rfrac_range = 1.1; // This is the (fractional) margin allowed for the inner cc radius to vary
					while (rmin_frac > 2*cc0_max_rfrac_range) {
						rstep *= 2;
						cc_rmin *= 2;
						rmin_frac /= 2;
					}
				}
				break;
			}
		}
	}
	grid_xlength = spline_frac*autogrid_frac*root1x;
	grid_ylength = spline_frac*autogrid_frac*root1y;
	spline_deflection(grid_xlength,grid_ylength,steps);
	return true;
}

Vector<dvector> QLens::find_critical_curves(bool &check)
{
	Vector<dvector> rcrit(2);
	rcrit[0].input(cc_thetasteps+1);
	rcrit[1].input(cc_thetasteps+1);

	double (Brent::*mag_r)(const double);
	mag_r = static_cast<double (Brent::*)(const double)> (&QLens::inverse_magnification_r);

	respline_at_end = false;
	resplinesteps = 0;
	double mag0, mag1;
	bool found_first_root;
	double rstep, thetastep, step_increment, step_increment_change, beginning_increment;
	thetastep = 2*M_PI/cc_thetasteps;
	beginning_increment = 1.2;
	step_increment_change = 0.5;
	bool first_iteration = true;
	double tangential_crit_total = 0;
	int i, iterations;
	double r;
	for (i=0, theta_crit=0; i < cc_thetasteps; i++, theta_crit += thetastep)
	{
		iterations = 0;
		rcrit[0][i] = 0; rcrit[1][i] = 0;
		step_increment = beginning_increment;
		for (;;)
		{
			iterations++;
			rstep = default_autogrid_initial_step;
			mag1 = inverse_magnification_r(cc_rmin);
			found_first_root = false;
			for (r=cc_rmin; r < cc_rmax-rstep; r += ((rstep *= step_increment)/step_increment))
			{
				mag0 = mag1;
				mag1 = inverse_magnification_r(r+rstep);
				if (mag0*mag1 < 0) {
					if (!found_first_root) {
						rcrit[0][i] = BrentsMethod(mag_r, r, r+rstep, 1e-3);
						if (rcrit[0][i] < 1e-6) die("catastrophic failure--critical curves smaller than 1e-6");
						found_first_root = true;
					} else {
						rcrit[1][i] = BrentsMethod(mag_r, r, r+rstep, 1e-3);
						if (rcrit[1][i] < 1e-6) die("catastrophic failure--critical curves smaller than 1e-6");
						tangential_crit_total += rcrit[1][i];
						break;
					}
				}
			}
			if (r+rstep >= cc_rmax) {
				if (defspline) {
					resplinesteps = defspline->nsteps();
					respline_at_end = true;
					unspline_deflection();
					if (cc_rmin > 1e-5) cc_rmin = ((cc_rmin/10) > 1e-5) ? cc_rmin/10 : 1e-5;
					cc_rmax *= 1.5;
					warn(warnings, "could not find critical curves after automatic deflection spline; deleting spline and trying again...");
					i = 0; theta_crit = 0;
				} else {
					if (iterations >= max_cc_search_iterations)
					{
						check = false;
						if (!found_first_root)
							warn(warnings, "could not find any critical curves along theta = %g after %i iterations",theta_crit,iterations);
						else
							warn(warnings, "could not find two critical curves along theta = %g after %i iterations",theta_crit,iterations);
						return rcrit;
					}
					step_increment = 1 + (step_increment-1)*step_increment_change;
				}
				rstep = default_autogrid_initial_step;
			} else {
				if (first_iteration)	// adjust the scale of rstep if it is too small
				{
					double rmin_frac, cc0_max_rfrac_range;
					rmin_frac = rcrit[0][0] / cc_rmin;
					cc0_max_rfrac_range = 10; // This is the (fractional) margin allowed for the inner cc radius to vary
													  // --must be large, or else it might skip over both curves if they are close!
					while (rmin_frac > 2*cc0_max_rfrac_range) {
						cc_rmin *= 2;
						rmin_frac /= 2;
					}
					if (cc_rmin > rcrit[0][0]/cc0_max_rfrac_range) cc_rmin /= 2;
					first_iteration = false;
				}
				else break;
			}
		}
	}
	rcrit[0][cc_thetasteps] = rcrit[0][0];
	rcrit[1][cc_thetasteps] = rcrit[1][0];

	check = true;
	return rcrit;
}

bool QLens::spline_critical_curves(bool verbal)
{
	respline_at_end = false;	// this is for deflection spline; will become true if def. needs to be resplined
	if (cc_splined==true) {
		delete[] ccspline;
		delete[] caustic;
	}
	if ((verbal) and (mpi_id==0)) cout << "Splining critical curves..." << endl;
	bool check;
	Vector<dvector> rcrit = find_critical_curves(check);
	if (check==false) {
		cc_splined = false;	// critical curves could not be found
		return false;
	}
	Vector<dvector> rcaust(2);
	double theta, thetastep;
	thetastep = 2*M_PI/cc_thetasteps;
	rcaust[0].input(cc_thetasteps+1);
	rcaust[1].input(cc_thetasteps+1);
	dvector theta_table(cc_thetasteps+1);
	dvector caust0_theta_table(cc_thetasteps+1);
	dvector caust1_theta_table(cc_thetasteps+1);
	double xp, yp;
	lensvector x, caust0, caust1;
	bool seems_spherical = false;
	for (int i=0; i < cc_thetasteps; i++) {
		theta_table[i] = i*thetastep;
		xp = cos(theta_table[i]); yp = sin(theta_table[i]);
		x[0] = grid_xcenter + rcrit[0][i]*xp;
		x[1] = grid_ycenter + rcrit[0][i]*yp;
		find_sourcept(x,caust0,0,reference_zfactors,default_zsrc_beta_factors);
		rcaust[0][i] = norm(caust0[0]-grid_xcenter,caust0[1]-grid_ycenter);
		caust0_theta_table[i] = angle(caust0[0]-grid_xcenter,caust0[1]-grid_ycenter);
		if (!(isspherical())) {
			x[0] = grid_xcenter + rcrit[1][i]*xp;
			x[1] = grid_ycenter + rcrit[1][i]*yp;
			find_sourcept(x,caust1,0,reference_zfactors,default_zsrc_beta_factors);
			rcaust[1][i] = norm(caust1[0]-grid_xcenter,caust1[1]-grid_ycenter);
			caust1_theta_table[i] = angle(caust1[0]-grid_xcenter,caust1[1]-grid_ycenter);
			if ((i==0) and (rcaust[1][i] < 1e-3))
				seems_spherical = true;
			else if ((seems_spherical == true) and (rcaust[1][i] > 1e-3))
				seems_spherical = false;
		} else {
			rcaust[1][i] = 0;
			caust1_theta_table[i] = theta_table[i];
		}
		if ((rcaust[0][i] < 0) or (rcaust[1][i] < 0)) {
			cc_splined = false;
			warn(warnings, "Cannot spline critical curves; caustics not well-defined");
			return false;
		}
	}
	if (seems_spherical==true) {
		effectively_spherical = true;
		for (int i=0; i < cc_thetasteps; i++)
			rcaust[1][i] = 0;
	}

	// put the (theta,r) points in order for splining
	theta_table[cc_thetasteps] = 2*M_PI;
	sort(cc_thetasteps, caust0_theta_table, rcaust[0]);
	sort(cc_thetasteps, caust1_theta_table, rcaust[1]);
	caust0_theta_table[cc_thetasteps] = 2*M_PI + caust0_theta_table[0];
	caust1_theta_table[cc_thetasteps] = 2*M_PI + caust1_theta_table[0];
	rcaust[0][cc_thetasteps] = rcaust[0][0];
	rcaust[1][cc_thetasteps] = rcaust[1][0];
	
	ccspline = new Spline[2];
	caustic = new Spline[2];
	int c0steps, c1steps;
	ccspline[0].input(theta_table, rcrit[0]);
	ccspline[1].input(theta_table, rcrit[1]);
	caustic[0].input(caust0_theta_table, rcaust[0]);
	caustic[1].input(caust1_theta_table, rcaust[1]);
		
	cc_splined = true;

	return check;
}

bool QLens::plot_splined_critical_curves(string critfile)
{
	if (!use_cc_spline) { warn("Cannot plot critical curves from spline, spline option is disabled"); return false; }
	if (!cc_splined) {
		if (spline_critical_curves()==false) return false;
	}
	ofstream crit;
	open_output_file(crit,critfile);
	if (use_scientific_notation) crit << setiosflags(ios::scientific);
	int nn = 500;
	double thetastep, rcrit0, rcrit1, rcaust0, rcaust1, xp, yp;
	thetastep = 2*M_PI/nn;
	double theta;
	int i;
	for (theta=0, i=0; i <= nn; i++, theta += thetastep) {
		rcrit0 = crit0_interpolate(theta);
		rcaust0 = caust0_interpolate(theta);
		xp = cos(theta); yp = sin(theta);
		crit << grid_xcenter+rcrit0*xp << "\t" << grid_ycenter+rcrit0*yp << "\t" << grid_xcenter+rcaust0*xp << "\t" << grid_ycenter+rcaust0*yp << endl;
	}
	crit << endl;
	for (theta=0, i=0; i <= nn; i++, theta += thetastep) {
		rcrit1 = crit1_interpolate(theta);
		rcaust1 = caust1_interpolate(theta);
		xp = cos(theta); yp = sin(theta);
		crit << grid_xcenter+rcrit1*xp << "\t" << grid_ycenter+rcrit1*yp << "\t" << grid_xcenter+rcaust1*xp << "\t" << grid_ycenter+rcaust1*yp << endl;
	}
	return true;
}

double QLens::caust0_interpolate(double theta)
{
	// sometimes caustic starts at some small angle, so make sure theta is in range
	if ((theta < caustic[0].xmin()) and (theta+2*M_PI < caustic[0].xmax()))
		theta += 2*M_PI;
	return caustic[0].splint(theta);
}
double QLens::caust1_interpolate(double theta)
{
	// sometimes caustic starts at some small angle, so make sure theta is in range
	if (effectively_spherical) return 0.0;
	if ((theta < caustic[1].xmin()) and (theta+2*M_PI < caustic[1].xmax()))
		theta += 2*M_PI;
	return caustic[1].splint(theta);
}

bool QLens::find_optimal_gridsize()
{
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}

	double (Brent::*mag_r)(const double);
	mag_r = static_cast<double (Brent::*)(const double)> (&QLens::inverse_magnification_r);

	double mag0, mag1;
	bool found_first_root;
	int thetasteps = 40;
	double rstep, thetastep, step_increment;
	thetastep = 2*M_PI/thetasteps;
	step_increment = 1.2;
	bool first_iteration = true;
	int i;
	double r, min_r, max_r, global_rmin=1e30, global_rmax=0;
	double max_x, max_y, global_xmax=0, global_ymax=0;
	for (i=0, theta_crit=0; i < thetasteps; i++, theta_crit += thetastep)
	{
		min_r = 1e30; max_r = 0;
		if (global_rmax > 0) rstep = 0.2*global_rmax;
		else rstep = default_autogrid_initial_step;
		mag1 = inverse_magnification_r(cc_rmin);
		found_first_root = false;
		for (r=cc_rmin; r < cc_rmax-rstep; r += ((rstep *= step_increment)/step_increment))
		{
			mag0 = mag1;
			mag1 = inverse_magnification_r(r+rstep);
			if (mag0*mag1 < 0) {
				if (!found_first_root) {
					min_r = BrentsMethod(mag_r, r, r+rstep, 1e-2*rstep);
					found_first_root = true;
				} else {
					max_r = BrentsMethod(mag_r, r, r+rstep, 1e-2*rstep);
				}
			}
		}
		if (!found_first_root) continue;
		if (min_r > max_r) max_r = min_r;
		max_x = abs(max_r*cos(theta_crit));
		max_y = abs(max_r*sin(theta_crit));
		if (min_r < global_rmin) {
			global_rmin = min_r;
			source_plane_rscale = source_plane_r(min_r);
		}
		if (max_r > global_rmax) {
			global_rmax = max_r;
		}
		if (max_x > global_xmax) global_xmax = max_x;
		if (max_y > global_ymax) global_ymax = max_y;
	}
	if ((global_xmax == 0) or (global_ymax == 0)) return false;
	grid_xlength = 2*(global_xmax*autogrid_frac);
	grid_ylength = 2*(global_ymax*autogrid_frac);
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	return true;
}

void QLens::sort_critical_curves()
{
	if (grid==NULL) {
		if (create_grid(false,reference_zfactors,default_zsrc_beta_factors)==false) { warn("Could not create recursive grid"); return; }
	}
	sorted_critical_curve.clear();
	int n_cc_pts = critical_curve_pts.size();
	if (n_cc_pts == 0) return;
	int n_cc = 1;
	double dist_threshold; // this should be defined by the smallest grid cell size
	double dist_threshold_frac = 2;
	vector<lensvector> critical_curves_temp = critical_curve_pts;
	vector<lensvector> caustics_temp = caustic_pts;
	vector<double> length_of_cell_temp = length_of_cc_cell;
	critical_curve new_critical_curve;
	lensvector displacement, last_pt;
	last_pt[0] = critical_curves_temp[0][0];
	last_pt[1] = critical_curves_temp[0][1];
	new_critical_curve.cc_pts.push_back(critical_curves_temp[0]);
	new_critical_curve.caustic_pts.push_back(caustics_temp[0]);
	new_critical_curve.length_of_cell.push_back(length_of_cell_temp[0]);
	dist_threshold = dist_threshold_frac*length_of_cell_temp[0];
	critical_curves_temp.erase(critical_curves_temp.begin());
	caustics_temp.erase(caustics_temp.begin());
	length_of_cell_temp.erase(length_of_cell_temp.begin());
	n_cc_pts--;

	int i, i_closest_pt, i_retry=0;
	double dist, shortest_dist;
	lensvector disp_from_first;
	while (n_cc_pts > 0) {
		shortest_dist = 1e30;
		for (i=0; i < n_cc_pts; i++) {
			displacement[0] = critical_curves_temp[i][0] - last_pt[0];
			displacement[1] = critical_curves_temp[i][1] - last_pt[1];
			dist = displacement.norm();
			if (dist < shortest_dist) {
				shortest_dist = dist;
				i_closest_pt = i;
			}
		}
		if (shortest_dist > dist_threshold) {
			disp_from_first[0] = last_pt[0] - new_critical_curve.cc_pts[0][0];
			disp_from_first[1] = last_pt[1] - new_critical_curve.cc_pts[0][1];
			if (disp_from_first.norm() > dist_threshold) {
				// Since it seems we're not closing the curve, maybe the issue is that the cell size changed as we traversed the critical curve.
				// Let's increase the distance threshold and try again (up to 3 tries).
				if (i_retry < 10) {
					i_retry++;
					dist_threshold *= 1.5;
					continue;
				}
			}
			// store this critical curve, move on to the next one
			sorted_critical_curve.push_back(new_critical_curve);
			new_critical_curve.cc_pts.clear();
			new_critical_curve.caustic_pts.clear();
			n_cc++;
			i_retry=0;
		}
		last_pt[0] = critical_curves_temp[i_closest_pt][0];
		last_pt[1] = critical_curves_temp[i_closest_pt][1];
		new_critical_curve.cc_pts.push_back(critical_curves_temp[i_closest_pt]);
		new_critical_curve.caustic_pts.push_back(caustics_temp[i_closest_pt]);
		new_critical_curve.length_of_cell.push_back(length_of_cell_temp[i_closest_pt]);
		dist_threshold = dist_threshold_frac*length_of_cell_temp[i_closest_pt];
		critical_curves_temp.erase(critical_curves_temp.begin()+i_closest_pt);
		caustics_temp.erase(caustics_temp.begin()+i_closest_pt);
		length_of_cell_temp.erase(length_of_cell_temp.begin()+i_closest_pt);
		n_cc_pts--;
	}
	sorted_critical_curve.push_back(new_critical_curve);
	sorted_critical_curves = true;
}

bool QLens::plot_sorted_critical_curves(string critfile)
{
	if (!sorted_critical_curves) sort_critical_curves();

	if (critfile != "") {
		ofstream crit;
		open_output_file(crit,critfile);
		if (use_scientific_notation) crit << setiosflags(ios::scientific);
		int n_cc = sorted_critical_curve.size();
		if (n_cc==0) return false;
		for (int j=0; j < n_cc; j++) {
			for (int k=0; k < sorted_critical_curve[j].cc_pts.size(); k++) {
				crit << sorted_critical_curve[j].cc_pts[k][0] << " " << sorted_critical_curve[j].cc_pts[k][1] << " " << sorted_critical_curve[j].caustic_pts[k][0] << " " << sorted_critical_curve[j].caustic_pts[k][1] << " " << sorted_critical_curve[j].length_of_cell[k] << endl;
			}
			// connect the first and last points to make a closed curve
			crit << sorted_critical_curve[j].cc_pts[0][0] << " " << sorted_critical_curve[j].cc_pts[0][1] << " " << sorted_critical_curve[j].caustic_pts[0][0] << " " << sorted_critical_curve[j].caustic_pts[0][1] << " " << sorted_critical_curve[j].length_of_cell[0] << endl;
			if (j < n_cc-1) crit << endl; // separates the critical curves in the plot
		}
	}
	return true;
}

double QLens::einstein_radius_of_primary_lens(const double zfac, double &reav)
{
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	// this calculates the Einstein radius of the "macro" lens model (treating the lens as spherical), ignoring any lens components that are not centered on the primary lens
	double rmin_einstein_radius = 1e-6;
	double rmax_einstein_radius = 1e4;
	double xc0, yc0, xc1, yc1, xc, yc;
	lens_list[primary_lens_number]->get_center_coords(xc0,yc0);
	centered = new bool[nlens];
	centered[primary_lens_number]=true;
	bool multiple_lenses = false;
	if (include_secondary_lens) {
		centered[secondary_lens_number]=true;
		lens_list[secondary_lens_number]->get_center_coords(xc1,yc1);
		multiple_lenses = true;
	}
	for (int j=0; j < nlens; j++) {
		if (j==primary_lens_number) continue;
		if ((include_secondary_lens) and (j==secondary_lens_number)) continue;
		lens_list[j]->get_center_coords(xc,yc);
		if (((xc==xc0) and (yc==yc0)) or ((include_secondary_lens) and ((xc==xc1) and (yc==yc1)))) {
			// If a lens is selected as the "secondary" lens (e.g. a BCG), then it will be treated as co-centered with the primary even if there's an offset;
			// the same is true for any other lenses co-centered with the secondary
			if (lens_list[j]->kapavgptr_rsq_spherical != NULL) {
				multiple_lenses = true;
				centered[j]=true;
				if (multiple_lenses==false) multiple_lenses = true;
			} else centered[j] = false;
		}
		else centered[j]=false;
	}
	if (multiple_lenses==false) {
		delete[] centered;
		double re;
		lens_list[primary_lens_number]->get_einstein_radius(re,reav,zfac);
		return re;
	}
	zfac_re = zfac;
	if ((einstein_radius_root(rmin_einstein_radius)*einstein_radius_root(rmax_einstein_radius)) > 0) {
		// multiple imaging does not occur with this lens
		delete[] centered;
		return 0;
	}
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&QLens::einstein_radius_root);
	reav = BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-3);
	double fprim, fprim_max = -1e30;
	for (int j=0; j < nlens; j++) {
		if (centered[j]) {
			fprim = lens_list[primary_lens_number]->get_f_major_axis(); // use the primary lens's axis ratio to approximate the major axis of critical curve
			if (fprim > fprim_max) fprim_max = fprim;
		}
	}
	double re_maj_approx = fprim_max*reav;
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating Einstein radius: " << wtime << endl;
	}
#endif
	delete[] centered;
	return re_maj_approx;
}

double QLens::einstein_radius_root(const double r)
{
	double kapavg=0;
	for (int j=0; j < nlens; j++) {
		if (centered[j]) kapavg += zfac_re*lens_list[j]->kappa_avg_r(r);
	}
	return (kapavg-1);
}

void QLens::plot_total_kappa(double rmin, double rmax, int steps, const char *kname, const char *kdname)
{
	double r, rstep, total_kappa, total_dkappa;
	rstep = pow(rmax/rmin, 1.0/steps);
	int i,j;
	ofstream kout(kname);
	ofstream kdout;
	if (kdname != NULL) kdout.open(kdname);
	if (use_scientific_notation) kout << setiosflags(ios::scientific);
	if (use_scientific_notation) kdout << setiosflags(ios::scientific);
	double arcsec_to_kpc = angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
	double sigma_cr_kpc = sigma_crit_kpc(lens_redshift, reference_source_redshift);
	double kap, kap2;
	double theta, thetastep;
	int thetasteps = 200;
	thetastep = 2*M_PI/thetasteps;
	double x, y, x2, y2, dr;
	dr = 1e-1*rmin*(rstep-1);
	
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}
	double *onezfac = new double[n_lens_redshifts];
	for (i=0; i < n_lens_redshifts; i++) onezfac[i] = 1.0;
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		total_kappa = 0;
		total_dkappa = 0;
		for (j=0, theta=0; j < thetasteps; j++, theta += thetastep) {
			x = grid_xcenter + r*cos(theta);
			y = grid_ycenter + r*sin(theta);
			x2 = (r+dr)*cos(theta);
			y2 = (r+dr)*sin(theta);
			kap = kappa(x,y,onezfac,default_zsrc_beta_factors);
			kap2 = kappa(x2,y2,onezfac,default_zsrc_beta_factors);
			total_kappa += kap;
			total_dkappa += (kap2 - kap)/dr;
		}
		total_kappa /= thetasteps;
		total_dkappa /= thetasteps;
		kout << r << " " << total_kappa << " " << r*arcsec_to_kpc << " " << total_kappa*sigma_cr_kpc << endl;
		if (kdname != NULL) kdout << r << " " << total_dkappa << r*arcsec_to_kpc << " " << total_dkappa*sigma_cr_kpc/arcsec_to_kpc << endl;
	}
	delete[] onezfac;

	/*
	double rsq;
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		rsq = r*r;
		kout << r << " ";
		if (kdname != NULL) kdout << r << " ";
		total_kappa = 0;
		if (kdname != NULL) total_dkappa = 0;
		for (int j=0; j < nlens; j++) {
			if (centered[j]) {
				// this ignores off-center lenses (perturbers) since we are plotting the radial profile; ellipticity is also ignored
				kap = lens_list[j]->kappa_rsq(rsq);
				if (kdname != NULL) dkap = lens_list[j]->kappa_rsq_deriv(rsq);
				total_kappa += kap;
				if (kdname != NULL) total_dkappa += dkap;
			}
		}
		kout << total_kappa << endl;
		if (kdname != NULL) kdout << fabs(total_dkappa) << endl;
	}
	*/
}

double QLens::einstein_radius_single_lens(const double src_redshift, const int lensnum)
{
	double re_avg,re_major,zfac;
	zfac = kappa_ratio(lens_list[lensnum]->zlens,src_redshift,reference_source_redshift);
	lens_list[lensnum]->get_einstein_radius(re_major,re_avg,zfac);
	return re_avg;
}

double QLens::total_kappa(const double r, const int lensnum, const bool use_kpc)
{
	double total_kappa;
	int j;
	double kap, kap2;
	double theta, thetastep;
	int thetasteps = 200;
	thetastep = 2*M_PI/thetasteps;
	double x, y;
	double z, r_arcsec = r;
	if (lensnum==-1) z = lens_list[primary_lens_number]->get_redshift();
	else z = lens_list[lensnum]->get_redshift();
	if (use_kpc) r_arcsec *= 206.264806/angular_diameter_distance(z);
	
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}
	total_kappa = 0;
	double *onezfac = new double[n_lens_redshifts];
	for (j=0; j < n_lens_redshifts; j++) onezfac[j] = 1.0; // this ensures that the reference source redshift is used, which is appropriate to each lens
	for (j=0, theta=0; j < thetasteps; j++, theta += thetastep) {
		x = grid_xcenter + r_arcsec*cos(theta);
		y = grid_ycenter + r_arcsec*sin(theta);
		if (lensnum==-1) kap = kappa(x,y,onezfac,default_zsrc_beta_factors);
		else kap = lens_list[lensnum]->kappa(x,y);
		total_kappa += kap;
	}
	delete[] onezfac;
	total_kappa /= thetasteps;
	return total_kappa;
}

double QLens::total_dkappa(const double r, const int lensnum, const bool use_kpc)
{
	double total_dkappa;
	int j;
	double kap, kap2;
	double theta, thetastep;
	int thetasteps = 200;
	thetastep = 2*M_PI/thetasteps;
	double x, y, x2, y2, dr;
	dr = 1e-5;
	double z, r_arcsec = r;
	if (lensnum==-1) z = lens_list[primary_lens_number]->get_redshift();
	else z = lens_list[lensnum]->get_redshift();
	if (use_kpc) r_arcsec *= 206.264806/angular_diameter_distance(z);
	
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}
	total_dkappa = 0;
	double *onezfac = new double[n_lens_redshifts];
	for (j=0; j < n_lens_redshifts; j++) onezfac[j] = 1.0;
	for (j=0, theta=0; j < thetasteps; j++, theta += thetastep) {
		x = grid_xcenter + r_arcsec*cos(theta);
		y = grid_ycenter + r_arcsec*sin(theta);
		x2 = (r_arcsec+dr)*cos(theta);
		y2 = (r_arcsec+dr)*sin(theta);
		if (lensnum==-1) {
			kap = kappa(x,y,onezfac,default_zsrc_beta_factors);
			kap2 = kappa(x2,y2,onezfac,default_zsrc_beta_factors);
		} else {
			kap = lens_list[lensnum]->kappa(x,y);
			kap2 = lens_list[lensnum]->kappa(x2,y2);
		}
		total_dkappa += (kap2 - kap)/dr;
	}
	total_dkappa /= thetasteps;
	return total_dkappa;
}

void QLens::plot_mass_profile(double rmin, double rmax, int rpts, const char *massname)
{
	double r, rstep, kavg;
	rstep = pow(rmax/rmin, 1.0/(rpts-1));
	int i;
	ofstream mout(massname);
	if (use_scientific_notation) mout << setiosflags(ios::scientific);
	double arcsec_to_kpc = angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
	double sigma_cr_arcsec = sigma_crit_arcsec(lens_redshift, reference_source_redshift);
	mout << "#radius(arcsec) mass(m_solar) radius(kpc)\n";
	for (i=0, r=rmin; i < rpts; i++, r *= rstep) {
		kavg = 0;
		for (int j=0; j < nlens; j++) {
			kavg += lens_list[j]->kappa_avg_r(r);
		}
		mout << r << " " << sigma_cr_arcsec*M_PI*kavg*r*r << " " << r*arcsec_to_kpc << endl;
	}
}

void QLens::plot_kappa_profile(int l, double rmin, double rmax, int steps, const char *kname, const char *kdname)
{
	if (l >= nlens) { warn("lens %i does not exist", l); return; }
	lens_list[l]->plot_kappa_profile(rmin,rmax,steps,kname,kdname);
}

bool QLens::isspherical()
{
	bool all_spherical = true;
	for (int i=0; i < nlens; i++)
		if (!(lens_list[i]->isspherical())) { all_spherical = false; break; }
	return all_spherical;
}

void QLens::print_lensing_info_at_point(const double x, const double y)
{
	lensvector point, alpha, beta;
	double sheartot, shear_angle;
	point[0] = x; point[1] = y;
	deflection(point,alpha,reference_zfactors,default_zsrc_beta_factors);
	//lensvector alpha2;
	//custom_deflection(point[0],point[1],alpha2);
	shear(point,sheartot,shear_angle,0,reference_zfactors,default_zsrc_beta_factors);
	beta[0] = point[0] - alpha[0];
	beta[1] = point[1] - alpha[1];
	double kappaval = kappa(point,reference_zfactors,default_zsrc_beta_factors);
	cout << "kappa = " << kappaval << endl;
	cout << "deflection = (" << alpha[0] << "," << alpha[1] << ")\n";
	//cout << "custom deflection = (" << alpha2[0] << "," << alpha2[1] << ")\n";
	cout << "potential = " << potential(point,reference_zfactors,default_zsrc_beta_factors) << endl;
	cout << "magnification = " << magnification(point,0,reference_zfactors,default_zsrc_beta_factors) << endl;
	cout << "shear = " << sheartot << ", shear_angle=" << shear_angle << endl;
	cout << "reduced_shear1 = " << sheartot*cos(2*shear_angle*M_PI/180.0)/(1-kappaval) << " reduced_shear2 = " << sheartot*sin(2*shear_angle*M_PI/180.0)/(1-kappaval) << endl;
	cout << "sourcept = (" << beta[0] << "," << beta[1] << ")\n";

	/*
	if (n_lens_redshifts > 1) {
		lensvector xl;
		for (int i=1; i < n_lens_redshifts; i++) {
			map_to_lens_plane(i,x,y,xl,0,reference_zfactors,default_zsrc_beta_factors);
			cout << "x(z=" << lens_redshifts[i] << "): (" << xl[0] << "," << xl[1] << ")" << endl;
		}

		int i,j;
		for (i=1; i < n_lens_redshifts; i++) {
			for (j=0; j < i; j++) cout << "beta(" << i << "," << j << "): " << default_zsrc_beta_factors[i-1][j] << endl;
		}
		for (i=0; i < n_lens_redshifts; i++) cout << "zfac(" << i << "): " << reference_zfactors[i] << endl;
	}
	*/
	if (n_sb > 0) {
		double sb = find_surface_brightness(point);
		cout << "surface brightness = " << sb << endl;
	}
	cout << endl;
	//cout << "shear/kappa = " << sheartot/kappa(point) << endl;
}


bool QLens::make_random_sources(int nsources, const char *outfilename)
{
	if (!use_cc_spline) { warn(warnings,"Random sources is only supported in critical curve spline mode (ccspline)"); return false; }
	ofstream outfile(outfilename);
	if (!cc_splined)
		if (spline_critical_curves()==false) return false;	// in case critical curves could not be found
	int sources_inside = 0;
	double theta, r, theta_step, rmax;
	int theta_n, theta_count = 400;
	theta_step = 2*M_PI/(theta_count-1);
	rmax = dmax(caust0_interpolate(0), caust1_interpolate(0));
	for (theta_n=0, theta=0; theta_n < theta_count; theta_n++, theta += theta_step) {
		r = dmax(caust0_interpolate(theta), caust1_interpolate(theta));
		if (r > rmax) rmax = r;
	}

	while (sources_inside < nsources)
	{
		theta = RandomNumber() * 2*M_PI;
		r = sqrt(RandomNumber()) * rmax;
		if (r < dmax(caust0_interpolate(theta), caust1_interpolate(theta))) {
			sources_inside++;
			outfile << grid_xcenter+r*cos(theta) << "\t" << grid_ycenter+r*sin(theta) << endl;
		}
	}
	return true;
}

void QLens::make_source_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename)
{
	ofstream sourcetab(source_filename.c_str());
	int i,j;
	double x,y,xstep,ystep;
	xstep = (xmax-xmin)/(xsteps-1);
	ystep = (ymax-ymin)/(ysteps-1);
	for (i=0, x=xmin; i < xsteps; i++, x += xstep)
		for (j=0, y=ymin; j < ysteps; j++, y += ystep)
			sourcetab << x << " " << y << endl;
}

void QLens::make_source_ellipse(const double xcenter, const double ycenter, const double major_axis, const double q, const double angle_degrees, const int n_subellipses, const int points_per_ellipse, string source_filename)
{
	ofstream source_file; open_output_file(source_file,source_filename);

	double da, dtheta, angle;
	da = major_axis/(n_subellipses-1);
	dtheta = M_2PI/points_per_ellipse;
	angle = (M_PI/180)*angle_degrees;
	double a, theta, x, y;

	int i,j;
	for (i=1, a=da; i < n_subellipses; i++, a += da)
	{
		for (j=0, theta=0; j < points_per_ellipse; j++, theta += dtheta)
		{
			x = a*cos(theta); y = a*q*sin(theta);
			source[0] = xcenter + x*cos(angle) - y*sin(angle);
			source[1] = ycenter + x*sin(angle) + y*cos(angle);
			source_file << source[0] << " " << source[1] << endl;
		}
	}
}

void QLens::raytrace_image_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename)
{
	ofstream sourcetab(source_filename.c_str());
	int i,j;
	double x,y,xs,ys,xstep,ystep;
	lensvector point, alpha;
	xstep = (xmax-xmin)/(xsteps-1);
	ystep = (ymax-ymin)/(ysteps-1);
	for (i=0, x=xmin; i < xsteps; i++, x += xstep) {
		for (j=0, y=ymin; j < ysteps; j++, y += ystep) {
			point[0] = x; point[1] = y;
			deflection(point,alpha,reference_zfactors,default_zsrc_beta_factors);
			xs = point[0] - alpha[0];
			ys = point[1] - alpha[1];
			sourcetab << xs << " " << ys << endl;
		}
	}
}

bool QLens::total_cross_section(double &area)
{
	if (!use_cc_spline) { warn(warnings,"Determining total cross section is only supported when ccspline mode is on"); return false; }
	if (!cc_splined)
		if (spline_critical_curves()==false) return false;	// in case critical curves could not be found
	double (Romberg::*csptr)(const double);
	csptr = static_cast<double (Romberg::*)(const double)> (&QLens::total_cross_section_integrand);
	area = romberg(csptr, 0, 2*M_PI,1e-6,5);

	return true;
}

double QLens::total_cross_section_integrand(const double theta)
{
	return (0.5 * SQR(dmax(caust0_interpolate(theta), caust1_interpolate(theta))));
}

/*
double QLens::make_perturber_population(const double number_density, const double rmax, const double b, const double a)
{
	int N = (int) (number_density*M_PI*rmax*rmax);
	int realizations = 3000;
	double r, theta, alpha_x, alpha_y, defsqr, defnorm, defsqr2;
	double mean_alpha_x=0, mean_alpha_y=0, mean_defsqr=0, mean_defsqr2=0;
	int i,j;
	for (j=0; j < realizations; j++) {
		alpha_x=alpha_y=0;
		defsqr2=0;
		for (i=0; i < N; i++) {
			r = sqrt(RandomNumber())*rmax;
			theta = RandomNumber()*2*M_PI;
			defnorm = b*(1+(a-sqrt(r*r+a*a))/r);
			alpha_x += -defnorm*cos(theta);
			alpha_y += -defnorm*sin(theta);
		}
		defsqr = SQR(alpha_x) + SQR(alpha_y);
		mean_defsqr += defsqr;
		mean_alpha_x += alpha_x;
		mean_alpha_y += alpha_y;
	}
	mean_defsqr /= realizations;
	mean_alpha_x /= realizations;
	mean_alpha_y /= realizations;
	//cout << "Root-mean square deflection: " << mean_defsqr << " " << mean_defsqr2 << endl;
	//cout << "Mean deflection: " << mean_alpha_x << " " << mean_alpha_y << endl;
	return mean_defsqr;
}

void QLens::plot_perturber_deflection_vs_area()
{
	int i,nn = 30;
	double r,rmin,rmax,rstep,logrstep,defsqr_avg;
	rmin=5;
	rmax=5000;
	rstep = (rmax-rmin)/nn;
	logrstep = pow(rmax/rmin,1.0/(nn-1));
	for (i=0, r=rmin; i < nn; i++, r *= logrstep) {
		defsqr_avg = make_perturber_population(0.04,r,0.1,0.6);
		cout << r << " " << log(r) << " " << defsqr_avg << endl;
	}
}
*/

/********************************* Functions for point image data (reading, writing, simulating etc.) *********************************/

bool QLens::add_simulated_image_data(const lensvector &sourcept)
{
	int i,j,k,n_images;
	if (nlens==0) { warn("no lens model has been created"); return false; }
	image *imgs = get_images(sourcept, n_images, false);
	if (n_images==0) { warn("could not find any images; no data added"); return false; }

	//bool *new_vary_sourcepts_x = new bool[n_sourcepts_fit+1];
	//bool *new_vary_sourcepts_y = new bool[n_sourcepts_fit+1];
	//lensvector *new_sourcepts_fit = new lensvector[n_sourcepts_fit+1];
	ImageData *new_image_data = new ImageData[n_sourcepts_fit+1];
	//lensvector* new_sourcepts_lower_limit;
	//lensvector* new_sourcepts_upper_limit;
	//if (sourcepts_upper_limit != NULL) {
		//new_sourcepts_lower_limit = new lensvector[n_sourcepts_fit+1];
		//new_sourcepts_upper_limit = new lensvector[n_sourcepts_fit+1];
	//}
	double *new_redshifts, **new_zfactors, ***new_beta_factors;
	new_redshifts = new double[n_sourcepts_fit+1];
	if (n_lens_redshifts > 0) new_zfactors = new double*[n_sourcepts_fit+1];
	new_beta_factors = new double**[n_sourcepts_fit+1];
	for (i=0; i < n_sourcepts_fit; i++) {
		new_redshifts[i] = source_redshifts[i];
		//new_sourcepts_fit[i] = sourcepts_fit[i];
		//new_vary_sourcepts_x[i] = vary_sourcepts_x[i];
		//new_vary_sourcepts_y[i] = vary_sourcepts_y[i];
		new_image_data[i].input(image_data[i]);
		//if (sourcepts_upper_limit != NULL) {
			//new_sourcepts_upper_limit[i] = sourcepts_upper_limit[i];
			//new_sourcepts_lower_limit[i] = sourcepts_lower_limit[i];
		//}
		if (n_lens_redshifts > 0) new_zfactors[i] = zfactors[i];
		if (n_lens_redshifts > 1) new_beta_factors[i] = beta_factors[i];
	}
	new_redshifts[n_sourcepts_fit] = source_redshift;
	if (n_lens_redshifts > 0) {
		new_zfactors[n_sourcepts_fit] = new double[n_lens_redshifts];
		for (j=0; j < n_lens_redshifts; j++) {
			new_zfactors[n_sourcepts_fit][j] = kappa_ratio(lens_redshifts[j],source_redshift,reference_source_redshift);
		}
	}
	if (n_lens_redshifts > 1) {
		new_beta_factors[n_sourcepts_fit] = new double*[n_lens_redshifts-1];
		for (j=1; j < n_lens_redshifts; j++) {
			new_beta_factors[n_sourcepts_fit][j-1] = new double[j];
			if (include_recursive_lensing) {
				for (k=0; k < j; k++) new_beta_factors[n_sourcepts_fit][j-1][k] = calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],source_redshift);
			} else {
				for (k=0; k < j; k++) new_beta_factors[n_sourcepts_fit][j-1][k] = 0;
			}
		}
	} else new_beta_factors[n_sourcepts_fit] = NULL;
	if (n_sourcepts_fit > 0) {
		delete[] image_data;
		//delete[] sourcepts_fit;
		//delete[] vary_sourcepts_x;
		//delete[] vary_sourcepts_y;
	}
	if (source_redshifts != NULL) delete[] source_redshifts;
	if (zfactors != NULL) delete[] zfactors;
	if (beta_factors != NULL) delete[] beta_factors;
	source_redshifts = new_redshifts;
	if (n_lens_redshifts > 0) {
		zfactors = new_zfactors;
		beta_factors = new_beta_factors;
	}

	bool include_image[n_images];
	double min_td=1e30;
	for (i=0; i < n_images; i++) {
		// central maxima images have positive parity and kappa > 1, so use this to exclude them if desired
		if ((include_central_image==false) and (imgs[i].parity == 1) and (kappa(imgs[i].pos,reference_zfactors,default_zsrc_beta_factors) > 1)) include_image[i] = false;
		else include_image[i] = true;
		imgs[i].pos[0] += sim_err_pos*NormalDeviate();
		imgs[i].pos[1] += sim_err_pos*NormalDeviate();
		imgs[i].mag *= source_flux; // now imgs[i].mag is in fact the flux, not just the magnification
		imgs[i].mag += sim_err_flux*NormalDeviate();
		if (include_time_delays) {
			imgs[i].td += sim_err_td*NormalDeviate();
			if (imgs[i].td < min_td) min_td = imgs[i].td;
		}
	}
	if (include_time_delays) {
		for (int i=0; i < n_images; i++) {
			imgs[i].td -= min_td;
		}
	}
	sourcepts_fit.push_back(sourcept);
	vary_sourcepts_x.push_back(true);
	vary_sourcepts_y.push_back(true);
	//new_sourcepts_fit[n_sourcepts_fit] = sourcept;
	//new_vary_sourcepts_x[n_sourcepts_fit] = true;
	//new_vary_sourcepts_y[n_sourcepts_fit] = true;
	new_image_data[n_sourcepts_fit].input(n_images,imgs,sim_err_pos,sim_err_flux,sim_err_td,include_image,include_time_delays);
	n_sourcepts_fit++;
	image_data = new_image_data;
	//sourcepts_fit = new_sourcepts_fit;
	//vary_sourcepts_x = new_vary_sourcepts_x;
	//vary_sourcepts_y = new_vary_sourcepts_y;

	//if (sourcepts_upper_limit != NULL) {
		//delete[] sourcepts_upper_limit;
		//delete[] sourcepts_lower_limit;
		lensvector lower; lower[0] = -1e30; lower[1] = -1e30;
		lensvector upper; upper[0] = 1e30; upper[1] = 1e30;
		sourcepts_lower_limit.push_back(lower);
		sourcepts_upper_limit.push_back(upper);

		//new_sourcepts_lower_limit[n_sourcepts_fit-1][0] = -1e30;
		//new_sourcepts_lower_limit[n_sourcepts_fit-1][1] = -1e30;
		//new_sourcepts_upper_limit[n_sourcepts_fit-1][0] = 1e30;
		//new_sourcepts_upper_limit[n_sourcepts_fit-1][1] = 1e30;
		//sourcepts_upper_limit = new_sourcepts_upper_limit;
		//sourcepts_lower_limit = new_sourcepts_lower_limit;
	//}
	sort_image_data_into_redshift_groups();
	include_imgpos_chisq = true;
	return true;
}

void QLens::write_image_data(string filename)
{
	ofstream outfile(filename.c_str());
	if (use_scientific_notation==true) outfile << setiosflags(ios::scientific);
	else {
		outfile << setprecision(6);
		outfile << fixed;
	}
	if (data_info != "") outfile << "# data_info: " << data_info << endl;
	outfile << "zlens = " << lens_redshift << endl;
	outfile << n_sourcepts_fit << " # number of source points" << endl;
	for (int i=0; i < n_sourcepts_fit; i++) {
		outfile << image_data[i].n_images << " " << source_redshifts[i] << " # number of images, source redshift" << endl;
		image_data[i].write_to_file(outfile);
	}
}

bool QLens::load_image_data(string filename)
{
	int i,j,k;
	ifstream data_infile(filename.c_str());
	if (!data_infile.is_open()) { warn("Error: input file '%s' could not be opened",filename.c_str()); return false; }

	int n_datawords;
	vector<string> datawords;

	if (read_data_line(data_infile,datawords,n_datawords)==false) return false;
	if ((n_datawords==2) and (datawords[0]=="zlens")) {
		double zlens;
		if (datastring_convert(datawords[1],zlens)==false) { warn("data file has incorrect format; could not read lens redshift"); return false; }
		if (zlens < 0) { warn("invalid redshift; redshift must be greater than zero"); return false; }
		lens_redshift = zlens;
		if (read_data_line(data_infile,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	}
	if (n_datawords != 1) { warn("input data file has incorrect format; first line should specify number of source points"); return false; }
	int nsrcfit;
	if (datastring_convert(datawords[0],nsrcfit)==false) { warn("data file has incorrect format; could not read number of source points"); return false; }
	if (nsrcfit <= 0) { warn("number of source points must be greater than zero"); return false; }
	n_sourcepts_fit = nsrcfit;

	//if (sourcepts_fit != NULL) {
		//delete[] sourcepts_fit;
		//delete[] vary_sourcepts_x;
		//delete[] vary_sourcepts_y;
	//}
	if (source_redshifts != NULL) delete[] source_redshifts;
	if (zfactors != NULL) {
		for (i=0; i < n_sourcepts_fit; i++) delete[] zfactors[i];
		delete[] zfactors;
	}
	if (beta_factors != NULL) {
		for (i=0; i < n_sourcepts_fit; i++) {
			if (beta_factors[i] != NULL) {
				for (j=0; j < n_lens_redshifts-1; j++) delete[] beta_factors[i][j];
				if (n_lens_redshifts > 1) delete[] beta_factors[i];
			}
		}
		delete[] beta_factors;
	}

	//sourcepts_fit = new lensvector[n_sourcepts_fit];
	//vary_sourcepts_x = new bool[n_sourcepts_fit];
	//vary_sourcepts_y = new bool[n_sourcepts_fit];
	source_redshifts = new double[n_sourcepts_fit];
	if (n_lens_redshifts > 0) {
		zfactors = new double*[n_sourcepts_fit];
		beta_factors = new double**[n_sourcepts_fit];
	}
	lensvector zero; zero[0]=0; zero[1]=0;
	for (i=0; i < n_sourcepts_fit; i++) {
		sourcepts_fit.push_back(zero);
		vary_sourcepts_x.push_back(true);
		vary_sourcepts_y.push_back(true);
		source_redshifts[i] = source_redshift;
		if (n_lens_redshifts > 0) {
			zfactors[i] = new double[n_lens_redshifts];
			if (n_lens_redshifts > 1) beta_factors[i] = new double*[n_lens_redshifts-1];
			else beta_factors[i] = NULL;
		}
	}
	if ((fitmethod != POWELL) and (fitmethod != SIMPLEX)) {
		//if (sourcepts_lower_limit != NULL) delete[] sourcepts_lower_limit;
		sourcepts_lower_limit.resize(n_sourcepts_fit);
		//if (sourcepts_upper_limit != NULL) delete[] sourcepts_upper_limit;
		sourcepts_upper_limit.resize(n_sourcepts_fit);
	}

	if (image_data != NULL) delete[] image_data;
	image_data = new ImageData[n_sourcepts_fit];
	int nn;
	bool zsrc_given_in_datafile = false;
	for (i=0; i < n_sourcepts_fit; i++) {
		if (read_data_line(data_infile,datawords,n_datawords)==false) { 
			warn("data file could not be read; unexpected end of file"); 
			clear_image_data();
			return false;
		}
		if ((n_datawords != 1) and (n_datawords != 2)) {
			warn("input data file has incorrect format; invalid number of images for source point %i",i);
			clear_image_data();
			return false;
		}
		if (datastring_convert(datawords[0],nn)==false) {
			warn("data file has incorrect format; could not read number of images for source point %i",i);
			clear_image_data();
			return false;
		}
		if (n_datawords==2) {
			if (datastring_convert(datawords[1],source_redshifts[i])==false) {
				warn("data file has incorrect format; could not read redshift for source point %i",i);
				clear_image_data();
				return false;
			}
			zsrc_given_in_datafile = true;
		}
		if (nn==0) warn("no images in data file for source point %i",i);
		image_data[i].input(nn);
		for (j=0; j < nn; j++) {
			if (read_data_line(data_infile,datawords,n_datawords)==false) {
				warn("data file could not be read; unexpected end of file"); 
				clear_image_data();
				return false;
			}
			if ((n_datawords != 5) and (n_datawords != 7)) {
				warn("input data file has incorrect format; wrong number of data entries for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[0],image_data[i].pos[j][0])==false) {
				warn("image position x-coordinate has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[1],image_data[i].pos[j][1])==false) {
				warn("image position y-coordinate has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[2],image_data[i].sigma_pos[j])==false) {
				warn("image position measurement error has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[3],image_data[i].flux[j])==false) {
				warn("image flux has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[4],image_data[i].sigma_f[j])==false) {
				warn("image flux measurement error has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (n_datawords==7) {
				if (datastring_convert(datawords[5],image_data[i].time_delays[j])==false) {
					warn("image time delay has incorrect format; could not read entry for source point %i, image number %i",i,j);
					clear_image_data();
					return false;
				}
				if (datastring_convert(datawords[6],image_data[i].sigma_t[j])==false) {
					warn("image time delay has incorrect format; could not read entry for source point %i, image number %i",i,j);
					n_sourcepts_fit=0; delete[] image_data; image_data = NULL;
					clear_image_data();
					return false;
				}
			} else {
				image_data[i].time_delays[j] = 0;
				image_data[i].sigma_t[j] = 0;
			}
		}
	}
	if (zsrc_given_in_datafile) {
		if (!user_changed_zsource) {
			source_redshift = source_redshifts[0];
			if (auto_zsource_scaling) {
				reference_source_redshift = source_redshifts[0];
				for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = 1.0;
			}
			else {
				for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = kappa_ratio(lens_redshifts[i],source_redshift,reference_source_redshift);
			}
		}
		// if source redshifts are given in the datafile, turn off auto scaling of zsrc_ref so user can experiment with different zsrc values if desired (without changing zsrc_ref)
		auto_zsource_scaling = false;
	}

	if (n_lens_redshifts > 0) {
		for (i=0; i < n_sourcepts_fit; i++) {
			for (j=0; j < n_lens_redshifts; j++) {
				zfactors[i][j] = kappa_ratio(lens_redshifts[j],source_redshifts[i],reference_source_redshift);
			}
		}
	}

	if (n_lens_redshifts > 1) {
		for (i=0; i < n_sourcepts_fit; i++) {
			for (j=1; j < n_lens_redshifts; j++) {
				beta_factors[i][j-1] = new double[j];
				if (include_recursive_lensing) {
					for (k=0; k < j; k++) beta_factors[i][j-1][k] = calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],source_redshifts[i]); // from cosmo.cpp
				} else {
					for (k=0; k < j; k++) beta_factors[i][j-1][k] = 0;
				}
			}
		}
	}

	sort_image_data_into_redshift_groups();

	int ncombs, max_combinations = -1;
	int n;
	for (i=0; i < n_sourcepts_fit; i++) {
		ncombs = image_data[i].n_images * (image_data[i].n_images-1) / 2;
		if (ncombs > max_combinations) max_combinations = ncombs;
	}
	double *distsqrs = new double[max_combinations];
	for (i=0; i < n_sourcepts_fit; i++) {
		n=0;
		for (k=0; k < image_data[i].n_images; k++) {
			for (j=k+1; j < image_data[i].n_images; j++) {
				distsqrs[n] = SQR(image_data[i].pos[k][0] - image_data[i].pos[j][0]) + SQR(image_data[i].pos[k][1] - image_data[i].pos[j][1]);
				n++;
			}
		}
		sort(n,distsqrs);
		image_data[i].max_distsqr = distsqrs[n-1]; // this saves the maximum distance between any pair of images (useful for image chi-square for missing image penalty values)
	}
	delete[] distsqrs;

	//cout << "n_redshift_groups=" << source_redshift_groups.size()-1 << endl;
	//for (i=0; i < source_redshift_groups.size(); i++) {
		//cout << source_redshift_groups[i] << endl;
	//}

	include_imgpos_chisq = true;
	return true;
}

void QLens::sort_image_data_into_redshift_groups()
{
	// Reorganize, if necessary, so that image sets with the same source redshift are listed together. This makes it easy to assign image sets with
	// different source planes to different MPI processes in the image plane chi-square.
	// We aren't trying to sort the groups from low to high redshift, only to make sure like redshifts occur in groups.
	//

	bool sort_sourcept_limits = false;
	int i,k,l,j=0;

	ImageData *sorted_image_data = new ImageData[n_sourcepts_fit];
	double *sorted_redshifts = new double[n_sourcepts_fit];
	double **sorted_zfactors;
	double ***sorted_beta_factors;
	if (n_lens_redshifts > 0) {
		sorted_zfactors = new double*[n_sourcepts_fit];
		sorted_beta_factors = new double**[n_sourcepts_fit];
		for (i=0; i < n_sourcepts_fit; i++) sorted_beta_factors[i] = beta_factors[i];
	}
	bool *sorted_vary_sourcepts_x = new bool[n_sourcepts_fit];
	bool *sorted_vary_sourcepts_y = new bool[n_sourcepts_fit];
	lensvector *sorted_sourcepts_upper_limit;
	lensvector *sorted_sourcepts_lower_limit;
	if (!sourcepts_upper_limit.empty()) {
		sort_sourcept_limits = true;
		sorted_sourcepts_upper_limit = new lensvector[n_sourcepts_fit];
		sorted_sourcepts_lower_limit = new lensvector[n_sourcepts_fit];
	}
	source_redshift_groups.clear();
	source_redshift_groups.push_back(0);
	bool *assigned = new bool[n_sourcepts_fit];
	for (i=0; i < n_sourcepts_fit; i++) assigned[i] = false;
	for (i=0; i < n_sourcepts_fit; i++) {
		if (!assigned[i]) {
			sorted_image_data[j].input(image_data[i]);
			sorted_redshifts[j] = source_redshifts[i];
			sorted_vary_sourcepts_x[j] = vary_sourcepts_x[i];
			sorted_vary_sourcepts_y[j] = vary_sourcepts_y[i];
			if (sort_sourcept_limits) {
				sorted_sourcepts_upper_limit[j] = sourcepts_upper_limit[i];
				sorted_sourcepts_lower_limit[j] = sourcepts_lower_limit[i];
			}
			if (n_lens_redshifts > 0) {
				sorted_zfactors[j] = zfactors[i];
				sorted_beta_factors[j] = beta_factors[i];
			}
			assigned[i] = true;
			j++;
			for (k=i+1; k < n_sourcepts_fit; k++) {
				if (!assigned[k]) {
					if (source_redshifts[k]==source_redshifts[i]) {
						sorted_image_data[j].input(image_data[k]);
						sorted_redshifts[j] = source_redshifts[k];
						if (n_lens_redshifts > 0) {
							sorted_zfactors[j] = zfactors[k];
							sorted_beta_factors[j] = beta_factors[k];
						}
						sorted_vary_sourcepts_x[j] = vary_sourcepts_x[k];
						sorted_vary_sourcepts_y[j] = vary_sourcepts_y[k];
						if (sort_sourcept_limits) {
							sorted_sourcepts_upper_limit[j] = sourcepts_upper_limit[k];
							sorted_sourcepts_lower_limit[j] = sourcepts_lower_limit[k];
						}
						assigned[k] = true;
						j++;
					}
				}
			}
			source_redshift_groups.push_back(j); // this stores the last index for each group of image sets with the same redshift
		}
	}
	delete[] image_data;
	delete[] source_redshifts;
	if (n_lens_redshifts > 0) {
		delete[] zfactors;
	}
	if (n_lens_redshifts > 0) delete[] beta_factors;
	for (int i=0; i < n_sourcepts_fit; i++) {
		vary_sourcepts_x[i] = sorted_vary_sourcepts_x[i];
		vary_sourcepts_y[i] = sorted_vary_sourcepts_y[i];
	}
	delete[] sorted_vary_sourcepts_x;
	delete[] sorted_vary_sourcepts_y;
	delete[] assigned;
	image_data = sorted_image_data;
	source_redshifts = sorted_redshifts;
	if (n_lens_redshifts > 0) {
		zfactors = sorted_zfactors;
		beta_factors = sorted_beta_factors;
	}
	//vary_sourcepts_x = sorted_vary_sourcepts_x;
	//vary_sourcepts_y = sorted_vary_sourcepts_y;
	if (sort_sourcept_limits) {
		for (int i=0; i < n_sourcepts_fit; i++) {
			sourcepts_upper_limit[i] = sorted_sourcepts_upper_limit[i];
			sourcepts_lower_limit[i] = sorted_sourcepts_lower_limit[i];
		}
		delete[] sorted_sourcepts_upper_limit;
		delete[] sorted_sourcepts_lower_limit;
		//sourcepts_upper_limit = sorted_sourcepts_upper_limit;
		//sourcepts_lower_limit = sorted_sourcepts_lower_limit;
	}
}

void QLens::remove_image_data(int image_set)
{
	//what about upper/lower limits (if they exist)? CHECK THIS
	if (image_set >= n_sourcepts_fit) { warn(warnings,"Specified image dataset has not been loaded"); return; }
	if (n_sourcepts_fit==1) { clear_image_data(); return; }
	//bool *new_vary_sourcepts_x = new bool[n_sourcepts_fit-1];
	//bool *new_vary_sourcepts_y = new bool[n_sourcepts_fit-1];
	//lensvector *new_sourcepts_fit = new lensvector[n_sourcepts_fit-1];
	sourcepts_fit.erase(sourcepts_fit.begin()+image_set);
	vary_sourcepts_x.erase(vary_sourcepts_x.begin()+image_set);
	vary_sourcepts_y.erase(vary_sourcepts_y.begin()+image_set);
	ImageData *new_image_data = new ImageData[n_sourcepts_fit-1];
	int i,j,k;
	double *new_redshifts, **new_zfactors, ***new_beta_factors;
	new_redshifts = new double[n_sourcepts_fit-1];
	if (n_lens_redshifts > 0) {
		new_zfactors = new double*[n_sourcepts_fit-1];
		new_beta_factors = new double**[n_sourcepts_fit-1];
	}
	for (i=0,j=0; i < n_sourcepts_fit; i++) {
		if (i != image_set) {
			new_image_data[j].input(image_data[i]);
			//new_sourcepts_fit[j] = sourcepts_fit[i];
			new_redshifts[j] = source_redshifts[i];
			//new_vary_sourcepts_x[j] = vary_sourcepts_x[i];
			//new_vary_sourcepts_y[j] = vary_sourcepts_y[i];
			if (n_lens_redshifts > 0) {
				new_zfactors[j] = zfactors[i];
				new_beta_factors[j] = beta_factors[i];
			}
			j++;
		} else {
			if (n_lens_redshifts > 0) {
				delete[] zfactors[i];
				if (beta_factors[i] != NULL) {
					for (k=0; k < n_lens_redshifts-1; k++) delete[] beta_factors[i][k];
					if (n_lens_redshifts > 1) delete[] beta_factors[i];
				}
			}
		}
	}
	delete[] source_redshifts;
	delete[] image_data;
	//delete[] sourcepts_fit;
	//delete[] vary_sourcepts_x;
	//delete[] vary_sourcepts_y;

	n_sourcepts_fit--;
	image_data = new_image_data;
	//sourcepts_fit = new_sourcepts_fit;
	source_redshifts = new_redshifts;
	if (n_lens_redshifts > 0) {
		delete[] zfactors;
		delete[] beta_factors;
		zfactors = new_zfactors;
		beta_factors = new_beta_factors;
	}
	//vary_sourcepts_x = new_vary_sourcepts_x;
	//vary_sourcepts_y = new_vary_sourcepts_y;

	sort_image_data_into_redshift_groups(); // this updates redshift_groups, in case there are no other image sets that shared the redshift of the one being deleted
}

bool QLens::plot_srcpts_from_image_data(int dataset_number, ofstream* srcfile, const double srcpt_x, const double srcpt_y, const double flux)
{
	// flux is an optional argument; if not specified, its default is -1, meaning fluxes will not be calculated or displayed
	if ((use_cc_spline) and (!cc_splined) and (spline_critical_curves()==false)) return false;
	if (dataset_number >= n_sourcepts_fit) { warn("specified dataset number does not exist"); return false; }

	int i,n_srcpts = image_data[dataset_number].n_images;
	lensvector *srcpts = new lensvector[n_srcpts];
	for (i=0; i < n_srcpts; i++) {
		find_sourcept(image_data[dataset_number].pos[i],srcpts[i],0,zfactors[dataset_number],beta_factors[dataset_number]);
	}

	if (use_scientific_notation==false) {
		cout << setprecision(6);
		cout << fixed;
	}

	double* time_delays_mod;
	if (include_time_delays) {
		double td_factor;
		time_delays_mod = new double[n_srcpts];
		double min_td_obs, min_td_mod;
		double pot;
		td_factor = time_delay_factor_arcsec(lens_redshift,source_redshifts[dataset_number]);
		min_td_obs=1e30;
		min_td_mod=1e30;
		for (i=0; i < n_srcpts; i++) {
			pot = potential(image_data[dataset_number].pos[i],zfactors[dataset_number],beta_factors[dataset_number]);
			time_delays_mod[i] = 0.5*(SQR(image_data[dataset_number].pos[i][0] - srcpts[i][0]) + SQR(image_data[dataset_number].pos[i][1] - srcpts[i][1])) - pot;
			if (time_delays_mod[i] < min_td_mod) min_td_mod = time_delays_mod[i];
		}
		for (i=0; i < n_srcpts; i++) {
			time_delays_mod[i] -= min_td_mod;
			if (time_delays_mod[i] != 0.0) time_delays_mod[i] *= td_factor; // td_factor contains the cosmological factors and is in units of days
		}
	}

	if (mpi_id==0) {
		cout << "# Source " << dataset_number << " from fit: " << srcpt_x << " " << srcpt_y << endl << endl;
		cout << "#imgpos_x\timgpos_y\tsrcpos_x\tsrcpos_y";
		if (flux != -1.0) cout << "\timage flux";
		if (include_time_delays) cout << "\ttime_delay (days)";
		cout << endl;
		double imgflux;
		for (i=0; i < n_srcpts; i++) {
			cout << image_data[dataset_number].pos[i][0] << "\t" << image_data[dataset_number].pos[i][1] << "\t" << srcpts[i][0] << "\t" << srcpts[i][1];
			if (srcfile != NULL) (*srcfile) << srcpts[i][0] << "\t" << srcpts[i][1];
			if (flux != -1) {
				imgflux = flux/inverse_magnification(image_data[dataset_number].pos[i],0,zfactors[dataset_number],beta_factors[dataset_number]);
				cout << "\t" << imgflux;
			}
			if (include_time_delays) {
				cout << "\t" << time_delays_mod[i];
			}
			cout << endl;
			if (srcfile != NULL) (*srcfile) << endl;
		}
		cout << endl;
	}
	if (use_scientific_notation==false)
		cout.unsetf(ios_base::floatfield);
	if (include_time_delays) delete[] time_delays_mod;

	delete[] srcpts;
	return true;
}

vector<ImageDataSet> QLens::export_to_ImageDataSet()
{
	vector<ImageDataSet> image_data_sets;
	image_data_sets.clear();
	image_data_sets.resize(n_sourcepts_fit);
	int i,j;
	for (i=0; i < n_sourcepts_fit; i++) {
		image_data_sets[i].set_n_images(image_data[i].n_images);
		image_data_sets[i].zsrc = source_redshifts[i];
		for (j=0; j < image_data[i].n_images; j++) {
			image_data_sets[i].images[j].pos[0] = image_data[i].pos[j][0];
			image_data_sets[i].images[j].pos[1] = image_data[i].pos[j][1];
			image_data_sets[i].images[j].flux = image_data[i].flux[j];
			image_data_sets[i].images[j].td = image_data[i].time_delays[j];
			image_data_sets[i].images[j].sigma_pos = image_data[i].sigma_pos[j];
			image_data_sets[i].images[j].sigma_flux = image_data[i].sigma_f[j];
			image_data_sets[i].images[j].sigma_td = image_data[i].sigma_t[j];
		}
	}
	return image_data_sets;
}

bool QLens::load_weak_lensing_data(string filename)
{
	int i,j,k;
	ifstream data_infile(filename.c_str());
	if (!data_infile.is_open()) { warn("Error: input file '%s' could not be opened",filename.c_str()); return false; }

	int n_datawords;
	vector<string> datawords;
	int n_wl_sources = 0;
	while (read_data_line(data_infile,datawords,n_datawords)) n_wl_sources++;
	data_infile.close();

	if (n_wl_sources==0) return false;
	data_infile.open(filename.c_str());

	weak_lensing_data.input(n_wl_sources);
	for (j=0; j < n_wl_sources; j++) {
		if (read_data_line(data_infile,datawords,n_datawords)==false) { 
			weak_lensing_data.clear();
			return false;
		}
		weak_lensing_data.id[j] = datawords[0];
		if (datastring_convert(datawords[1],weak_lensing_data.pos[j][0])==false) {
			warn("weak lensing source x-coordinate has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[2],weak_lensing_data.pos[j][1])==false) {
			warn("weak lensing source y-coordinate has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[3],weak_lensing_data.reduced_shear1[j])==false) {
			warn("weak lensing source reduced shear1 has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[4],weak_lensing_data.reduced_shear2[j])==false) {
			warn("weak lensing source reduced shear2 has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[5],weak_lensing_data.sigma_shear1[j])==false) {
			warn("source shear1 measurement error has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[6],weak_lensing_data.sigma_shear2[j])==false) {
			warn("source shear2 measurement error has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[7],weak_lensing_data.zsrc[j])==false) {
			warn("source redshift thas incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
	}

	// For starters, let's not store zfactors, we'll just calculate them when we evaluate the chi-square. We can always put
	// the zfactors in to save time later.
	//if (n_lens_redshifts > 0) {
		//for (i=0; i < n_sourcepts_fit; i++) {
			//for (j=0; j < n_lens_redshifts; j++) {
				//weak_lensing_data.zfactors[i][j] = kappa_ratio(lens_redshifts[j],source_redshifts[i],reference_source_redshift);
			//}
		//}
	//}

	// I don't think beta factors should matter for weak lensing, but you can check this later
	//if (n_lens_redshifts > 1) {
		//for (i=0; i < n_sourcepts_fit; i++) {
			//for (j=1; j < n_lens_redshifts; j++) {
				//beta_factors[i][j-1] = new double[j];
				//if (include_recursive_lensing) {
					//for (k=0; k < j; k++) beta_factors[i][j-1][k] = calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],source_redshifts[i]); // from cosmo.cpp
				//} else {
					//for (k=0; k < j; k++) beta_factors[i][j-1][k] = 0;
				//}
			//}
		//}
	//}

	include_weak_lensing_chisq = true;
	return true;
}

void QLens::add_simulated_weak_lensing_data(const string id, lensvector &sourcept, const double zsrc)
{
	double *zfacs = new double[n_lens_redshifts];

	for (int i=0; i < n_lens_redshifts; i++) {
		zfacs[i] = kappa_ratio(lens_redshifts[i],zsrc,reference_source_redshift);
	}
	double shear1, shear2;
	reduced_shear_components(sourcept,shear1,shear2,0,zfacs);
	shear1 += sim_err_shear*NormalDeviate();
	shear2 += sim_err_shear*NormalDeviate();
	weak_lensing_data.add_source(id,sourcept,shear1,shear2,sim_err_shear,sim_err_shear,zsrc);
	if (!include_weak_lensing_chisq) include_weak_lensing_chisq = true;
	delete[] zfacs;
}

void QLens::add_weak_lensing_data_from_random_sources(const int num_sources, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, const double r_exclude)
{
	int wl_index = weak_lensing_data.n_sources;
	lensvector src;
	string id_string;
	double zsrc;
	for (int i=0; i < num_sources; i++) {
		stringstream idstr;
		idstr << wl_index;
		idstr >> id_string;

		do {
			src[0] = (xmax-xmin)*RandomNumber() + xmin;
			src[1] = (ymax-ymin)*RandomNumber() + ymin;
		} while (sqrt(SQR(src[0]-grid_xcenter)+SQR(src[1]-grid_ycenter)) <= r_exclude);
		zsrc = (zmax-zmin)*RandomNumber() + zmin; // redshift
		add_simulated_weak_lensing_data(id_string,src,zsrc);

		wl_index++;
	}
}

bool QLens::read_data_line(ifstream& data_infile, vector<string>& datawords, int &n_datawords)
{
	static const int n_characters = 512;
	int pos;
	string word;
	n_datawords = 0;
	datawords.clear();
	do {
		char dataline[n_characters];
		data_infile.getline(dataline,n_characters);
		if (data_infile.gcount()==n_characters-1) {
			warn("the number of characters in a single line cannot exceed %i",n_characters);
			return false;
		}
		if ((data_infile.rdstate() & ifstream::eofbit) != 0) {
			return false;
		}
		string linestring(dataline);
		if ((pos = linestring.find("data_info: ")) != string::npos) {
			data_info = linestring.substr(pos+11);
		} else {
			remove_comments(linestring);
			istringstream datastream0(linestring.c_str());
			while (datastream0 >> word) datawords.push_back(word);
			n_datawords = datawords.size();
		}
	} while (n_datawords==0); // skip lines that are blank or only have comments
	remove_equal_sign_datafile(datawords,n_datawords);
	return true;
}

void QLens::remove_comments(string& instring)
{
	string instring_copy(instring);
	instring.clear();
	size_t comment_pos = instring_copy.find("#");
	if (comment_pos != string::npos) {
		instring = instring_copy.substr(0,comment_pos);
	} else instring = instring_copy;
}

void QLens::remove_equal_sign_datafile(vector<string>& datawords, int &n_datawords)
{
	int pos;
	if ((pos = datawords[0].find('=')) != string::npos) {
		// there's an equal sign in the first word, so remove it and separate into two datawords
		datawords.push_back("");
		for (int i=n_datawords-1; i > 0; i--) datawords[i+1] = datawords[i];
		datawords[1] = datawords[0].substr(pos+1);
		datawords[0] = datawords[0].substr(0,pos);
		n_datawords++;
	}
	else if ((n_datawords == 3) and (datawords[1]=="="))
	{
		// there's an equal sign in the second of three datawords (indicating a parameter assignment), so remove it and reduce to two datawords
		string word1,word2;
		word1=datawords[0]; word2=datawords[2];
		datawords.clear();
		datawords.push_back(word1);
		datawords.push_back(word2);
		n_datawords = 2;
	}
}

bool QLens::datastring_convert(const string& instring, int& outvar)
{
	datastream.clear(); // resets the error flags
	datastream.str(string()); // clears the stringstream
	datastream << instring;
	if (datastream >> outvar) return true;
	else return false;
}

bool QLens::datastring_convert(const string& instring, double& outvar)
{
	datastream.clear(); // resets the error flags
	datastream.str(string()); // clears the stringstream
	datastream << instring;
	if (datastream >> outvar) return true;
	else return false;
}

void QLens::clear_image_data()
{
	int i,j;
	if (image_data != NULL) {
		delete[] image_data;
		image_data = NULL;
	}
	sourcepts_fit.clear();
	vary_sourcepts_x.clear();
	vary_sourcepts_y.clear();
	sourcepts_lower_limit.clear();
	sourcepts_upper_limit.clear();
	//if (sourcepts_fit != NULL) {
		//delete[] sourcepts_fit;
		//delete[] vary_sourcepts_x;
		//delete[] vary_sourcepts_y;
		//sourcepts_fit = NULL;
	//}
	//if (sourcepts_lower_limit != NULL) {
		//delete[] sourcepts_lower_limit;
		//sourcepts_lower_limit = NULL;
	//}
	//if (sourcepts_upper_limit != NULL) {
		//delete[] sourcepts_upper_limit;
		//sourcepts_upper_limit = NULL;
	//}
	if (zfactors != NULL) {
		for (i=0; i < n_sourcepts_fit; i++) delete[] zfactors[i];
		delete[] zfactors;
		zfactors = NULL;
	}
	if (default_zsrc_beta_factors != NULL) {
		for (i=1; i < n_lens_redshifts; i++) delete[] default_zsrc_beta_factors[i-1];
		delete[] default_zsrc_beta_factors;
		default_zsrc_beta_factors = NULL;
	}
	if (beta_factors != NULL) {
		for (i=0; i < n_sourcepts_fit; i++) {
			for (j=1; j < n_lens_redshifts; j++) delete[] beta_factors[i][j-1];
			if (n_lens_redshifts > 1) delete[] beta_factors[i];
		}
		delete[] beta_factors;
		beta_factors = NULL;
	}
	if (source_redshifts != NULL) {
		delete[] source_redshifts;
		source_redshifts = NULL;
	}

	n_sourcepts_fit = 0;
}

void QLens::print_image_data(bool include_errors)
{
	if (mpi_id==0) {
		for (int i=0; i < n_sourcepts_fit; i++) {
			cout << "Source " << i << ": zsrc=" << source_redshifts[i] << endl;
			image_data[i].print_list(include_errors,use_scientific_notation);
		}
	}
}

void ImageData::input(const int &nn)
{
	n_images = nn;
	pos = new lensvector[n_images];
	flux = new double[n_images];
	time_delays = new double[n_images];
	sigma_pos = new double[n_images];
	sigma_f = new double[n_images];
	sigma_t = new double[n_images];
	use_in_chisq = new bool[n_images];
	max_distsqr = 1e30;
}

void ImageData::input(const ImageData& imgs_in)
{
	if (n_images != 0) {
		// delete arrays so we can re-create them
		delete[] pos;
		delete[] flux;
		delete[] time_delays;
		delete[] sigma_pos;
		delete[] sigma_f;
		delete[] sigma_t;
		delete[] use_in_chisq;
	}
	n_images = imgs_in.n_images;
	pos = new lensvector[n_images];
	flux = new double[n_images];
	time_delays = new double[n_images];
	sigma_pos = new double[n_images];
	sigma_f = new double[n_images];
	sigma_t = new double[n_images];
	use_in_chisq = new bool[n_images];
	for (int i=0; i < n_images; i++) {
		pos[i] = imgs_in.pos[i];
		flux[i] = imgs_in.flux[i];
		time_delays[i] = imgs_in.time_delays[i];
		sigma_pos[i] = imgs_in.sigma_pos[i];
		sigma_f[i] = imgs_in.sigma_f[i];
		sigma_t[i] = imgs_in.sigma_t[i];
		use_in_chisq[i] = true;
	}
	max_distsqr = imgs_in.max_distsqr;
}

void ImageData::input(const int &nn, image* images, const double sigma_pos_in, const double sigma_flux_in, const double sigma_td_in, bool* include, bool include_time_delays)
{
	// this function is used to store simulated data
	int n_images_include=0;
	for (int i=0; i < nn; i++) if (include[i]) n_images_include++;
	n_images = n_images_include;
	pos = new lensvector[n_images];
	flux = new double[n_images];
	time_delays = new double[n_images];
	sigma_pos = new double[n_images];
	sigma_f = new double[n_images];
	sigma_t = new double[n_images];
	use_in_chisq = new bool[n_images];
	int j=0;
	for (int i=0; i < nn; i++) {
		if (!include[i]) continue;
		pos[j] = images[i].pos;
		flux[j] = images[i].mag; // images[i].mag should be the flux, not the magnification
		if (include_time_delays) {
			time_delays[j] = images[i].td;
			sigma_t[j] = sigma_td_in;
		}
		else { time_delays[j] = 0; sigma_t[j] = 0; }
		sigma_pos[j] = sigma_pos_in;
		sigma_f[j] = sigma_flux_in;
		use_in_chisq[j] = true;
		j++;
	}
	max_distsqr = 1e30;
}


void ImageData::add_image(lensvector& pos_in, const double sigma_pos_in, const double flux_in, const double sigma_f_in, const double time_delay_in, const double sigma_t_in)
{
	int n_images_new = n_images+1;
	if (n_images != 0) {
		lensvector *new_pos = new lensvector[n_images_new];
		double *new_flux = new double[n_images_new];
		double *new_time_delays = new double[n_images_new];
		double *new_sigma_pos = new double[n_images_new];
		double *new_sigma_f = new double[n_images_new];
		double *new_sigma_t = new double[n_images_new];
		bool *new_use_in_chisq = new bool[n_images_new];
		for (int i=0; i < n_images; i++) {
			new_pos[i][0] = pos[i][0];
			new_pos[i][1] = pos[i][1];
			new_flux[i] = flux[i];
			new_time_delays[i] = time_delays[i];
			new_sigma_pos[i] = sigma_pos[i];
			new_sigma_f[i] = sigma_f[i];
			new_sigma_t[i] = sigma_t[i];
			new_use_in_chisq[i] = use_in_chisq[i];
		}
		delete[] pos;
		delete[] flux;
		delete[] time_delays;
		delete[] sigma_pos;
		delete[] sigma_f;
		delete[] sigma_t;
		delete[] use_in_chisq;
		pos = new_pos;
		flux = new_flux;
		time_delays = new_time_delays;
		sigma_pos = new_sigma_pos;
		sigma_f = new_sigma_f;
		sigma_t = new_sigma_t;
		use_in_chisq = new_use_in_chisq;
		n_images++;
	} else {
		n_images = 1;
		pos = new lensvector[n_images];
		flux = new double[n_images];
		time_delays = new double[n_images];
		sigma_pos = new double[n_images];
		sigma_f = new double[n_images];
		sigma_t = new double[n_images];
		use_in_chisq = new bool[n_images];
	}
	pos[n_images-1][0] = pos_in[0];
	pos[n_images-1][1] = pos_in[1];
	flux[n_images-1] = flux_in;
	time_delays[n_images-1] = time_delay_in;
	sigma_pos[n_images-1] = sigma_pos_in;
	sigma_f[n_images-1] = sigma_f_in;
	sigma_t[n_images-1] = sigma_t_in;
	use_in_chisq[n_images-1] = true;
}

bool ImageData::set_use_in_chisq(int image_i, bool use_in_chisq_in)
{
	if (image_i >= n_images) return false;
	use_in_chisq[image_i] = use_in_chisq_in;
	return true;
}

void ImageData::print_list(bool print_errors, bool use_sci)
{
	if (use_sci==false) {
		cout << setprecision(6);
		cout << fixed;
	}
	if (print_errors) cout << "#        pos_x(arcsec)\tpos_y(arcsec)\tsig_pos\t\tflux\t\tsig_flux";
	else cout << "#        pos_x\t\tpos_y\t\tflux";
	if (sigma_t[0] != 0) {
		if (print_errors) cout << "\ttime_delay(days)\tsigma_t\n";
		else cout << "\ttime_delay\n";
	}
	else cout << endl;
	for (int i=0; i < n_images; i++) {
		cout << "Image " << i << ": " << pos[i][0] << "\t" << pos[i][1];
		if (print_errors) cout << "\t" << sigma_pos[i];
		cout << "\t" << flux[i];
		if (print_errors) cout << "\t" << sigma_f[i];
		if (sigma_t[0] != 0) {
			cout << "\t" << time_delays[i];
			if (print_errors) cout << "\t\t" << sigma_t[i];
		}
		if (!use_in_chisq[i]) cout << "   (excluded from chisq)";
		cout << endl;
	}
	cout << endl;
	if (use_sci==false)
		cout.unsetf(ios_base::floatfield);
}

void ImageData::write_to_file(ofstream &outfile)
{
	for (int i=0; i < n_images; i++) {
		outfile << pos[i][0] << " " << pos[i][1];
		outfile << " " << sigma_pos[i];
		outfile << " " << flux[i];
		outfile << " " << sigma_f[i];
		if (sigma_t[0] != 0) {
			outfile << " " << time_delays[i];
			outfile << " " << sigma_t[i];
		}
		outfile << endl;
	}
}

ImageData::~ImageData()
{
	if (n_images != 0) {
		delete[] pos;
		delete[] flux;
		delete[] time_delays;
		delete[] sigma_pos;
		delete[] sigma_f;
		delete[] sigma_t;
		delete[] use_in_chisq;
	}
}

void WeakLensingData::input(const int &nn)
{
	if (n_sources != 0) {
		// delete arrays so we can re-create them
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
	}
	n_sources = nn;
	id = new string[n_sources];
	pos = new lensvector[n_sources];
	reduced_shear1 = new double[n_sources];
	reduced_shear2 = new double[n_sources];
	sigma_shear1 = new double[n_sources];
	sigma_shear2 = new double[n_sources];
	zsrc = new double[n_sources];
}

void WeakLensingData::input(const WeakLensingData& wl_in)
{
	if (n_sources != 0) {
		// delete arrays so we can re-create them
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
	}
	n_sources = wl_in.n_sources;
	id = new string[n_sources];
	pos = new lensvector[n_sources];
	reduced_shear1 = new double[n_sources];
	reduced_shear2 = new double[n_sources];
	sigma_shear1 = new double[n_sources];
	sigma_shear2 = new double[n_sources];
	zsrc = new double[n_sources];
	for (int i=0; i < n_sources; i++) {
		id[i] = wl_in.id[i];
		pos[i] = wl_in.pos[i];
		reduced_shear1[i] = wl_in.reduced_shear1[i];
		reduced_shear2[i] = wl_in.reduced_shear2[i];
		sigma_shear1[i] = wl_in.sigma_shear1[i];
		sigma_shear2[i] = wl_in.sigma_shear2[i];
		zsrc[i] = wl_in.zsrc[i];
	}
}

void WeakLensingData::add_source(const string id_in, lensvector& pos_in, const double g1_in, const double g2_in, const double g1_err_in, const double g2_err_in, const double zsrc_in)
{
	int n_sources_new = n_sources+1;
	if (n_sources != 0) {
		string *new_id = new string[n_sources_new];
		lensvector *new_pos = new lensvector[n_sources_new];
		double *new_reduced_shear1 = new double[n_sources_new];
		double *new_reduced_shear2 = new double[n_sources_new];
		double *new_sigma_shear1 = new double[n_sources_new];
		double *new_sigma_shear2 = new double[n_sources_new];
		double *new_zsrc = new double[n_sources_new];
		for (int i=0; i < n_sources; i++) {
			new_id[i] = id[i];
			new_pos[i][0] = pos[i][0];
			new_pos[i][1] = pos[i][1];
			new_reduced_shear1[i] = reduced_shear1[i];
			new_reduced_shear2[i] = reduced_shear2[i];
			new_sigma_shear1[i] = sigma_shear1[i];
			new_sigma_shear2[i] = sigma_shear2[i];
			new_zsrc[i] = zsrc[i];
		}
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
		id = new_id;
		pos = new_pos;
		reduced_shear1 = new_reduced_shear1;
		reduced_shear2 = new_reduced_shear2;
		sigma_shear1 = new_sigma_shear1;
		sigma_shear2 = new_sigma_shear2;
		zsrc = new_zsrc;
		n_sources++;
	} else {
		n_sources = 1;
		id = new string[n_sources];
		pos = new lensvector[n_sources];
		reduced_shear1 = new double[n_sources];
		reduced_shear2 = new double[n_sources];
		sigma_shear1 = new double[n_sources];
		sigma_shear2 = new double[n_sources];
		zsrc = new double[n_sources];
	}
	id[n_sources-1] = id_in;
	pos[n_sources-1][0] = pos_in[0];
	pos[n_sources-1][1] = pos_in[1];
	reduced_shear1[n_sources-1] = g1_in;
	reduced_shear2[n_sources-1] = g2_in;
	sigma_shear1[n_sources-1] = g1_err_in;
	sigma_shear2[n_sources-1] = g2_err_in;
	zsrc[n_sources-1] = zsrc_in;
}

void WeakLensingData::print_list(bool use_sci)
{
	if (use_sci==false) {
		cout << setprecision(6);
		cout << fixed;
	}
	cout << "# id\tpos_x(arcsec)\tpos_y(arcsec)\tg1\t\tg2\t\tsig_g1\t\tsig_g2\t\tzsrc\n";
	for (int i=0; i < n_sources; i++) {
		cout << id[i] << "\t" << pos[i][0] << "\t" << pos[i][1];
		cout << "\t" << reduced_shear1[i];
		cout << "\t" << reduced_shear2[i];
		cout << "\t" << sigma_shear1[i];
		cout << "\t" << sigma_shear2[i];
		cout << "\t" << zsrc[i] << endl;
	}
	cout << endl;
	if (use_sci==false)
		cout.unsetf(ios_base::floatfield);
}

void WeakLensingData::write_to_file(string filename)
{
	ofstream outfile(filename.c_str());
	for (int i=0; i < n_sources; i++) {
		outfile << id[i] << " ";
		outfile << pos[i][0] << " " << pos[i][1];
		outfile << " " << reduced_shear1[i];
		outfile << " " << reduced_shear2[i];
		outfile << " " << sigma_shear1[i];
		outfile << " " << sigma_shear2[i];
		outfile << " " << zsrc[i];
		outfile << endl;
	}
}

WeakLensingData::~WeakLensingData()
{
	if (n_sources != 0) {
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
	}
}

void WeakLensingData::clear()
{
	if (n_sources != 0) {
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
	}
	n_sources = 0;
}

/******************************************** Functions for lens model fitting ******************************************/

bool QLens::initialize_fitmodel(const bool running_fit_in)
{
	if (source_fit_mode == Point_Source) {
		if (((!include_weak_lensing_chisq) or (weak_lensing_data.n_sources==0)) and ((sourcepts_fit.empty()) or (image_data==NULL))) {
			warn("cannot do fit; image data points have not been defined");
			return false;
		}
		if ((image_data==NULL) and (include_imgpos_chisq)) { warn("cannot evaluate image position chi-square; no image data has been defined"); return false; }
		else if ((image_data==NULL) and (include_flux_chisq)) { warn("cannot evaluate image flux chi-square; no image data has been defined"); return false; }
		else if ((image_data==NULL) and (include_time_delay_chisq)) { warn("cannot evaluate image time delay chi-square; no image data has been defined"); return false; }
	} else {
		if (image_pixel_data==NULL) { warn("cannot do fit; image data pixels have not been loaded"); return false; }
	}
	if (fitmodel != NULL) delete fitmodel;
	fitmodel = new QLens(this);
	fitmodel->use_ansi_characters = running_fit_in;
	fitmodel->auto_ccspline = false;
	//fitmodel->set_gridcenter(grid_xcenter,grid_ycenter);

	int i,j,k;
	if (n_lens_redshifts > 0) {
		fitmodel->reference_zfactors = new double[n_lens_redshifts];
		fitmodel->default_zsrc_beta_factors = new double*[n_lens_redshifts-1];
		fitmodel->lens_redshifts = new double[n_lens_redshifts];
		fitmodel->lens_redshift_idx = new int[nlens];
		fitmodel->zlens_group_size = new int[n_lens_redshifts];
		fitmodel->zlens_group_lens_indx = new int*[n_lens_redshifts];
		for (i=0; i < n_lens_redshifts; i++) {
			fitmodel->reference_zfactors[i] = reference_zfactors[i];
			fitmodel->lens_redshifts[i] = lens_redshifts[i];
			fitmodel->zlens_group_size[i] = zlens_group_size[i];
			fitmodel->zlens_group_lens_indx[i] = new int[zlens_group_size[i]];
			//cout << "Redshift group " << i << " (z=" << fitmodel->lens_redshifts[i] << ", " << fitmodel->zlens_group_size[i] << " lenses)\n";
			for (j=0; j < zlens_group_size[i]; j++) {
				fitmodel->zlens_group_lens_indx[i][j] = zlens_group_lens_indx[i][j];
				//cout << fitmodel->zlens_group_lens_indx[i][j] << " ";
			}
			//cout << endl;
		}
		for (i=0; i < n_lens_redshifts-1; i++) {
			fitmodel->default_zsrc_beta_factors[i] = new double[i+1];
			for (j=0; j < i+1; j++) fitmodel->default_zsrc_beta_factors[i][j] = default_zsrc_beta_factors[i][j];
		}
		for (j=0; j < nlens; j++) fitmodel->lens_redshift_idx[j] = lens_redshift_idx[j];
	}

	fitmodel->borrowed_image_data = true;
	if (source_fit_mode == Point_Source) {
		fitmodel->image_data = image_data;
		fitmodel->n_sourcepts_fit = n_sourcepts_fit;
		fitmodel->sourcepts_fit = sourcepts_fit;
		fitmodel->vary_sourcepts_x = vary_sourcepts_x;
		fitmodel->vary_sourcepts_y = vary_sourcepts_y;
		fitmodel->sourcepts_lower_limit = sourcepts_lower_limit;
		fitmodel->sourcepts_upper_limit = sourcepts_upper_limit;
		//fitmodel->sourcepts_fit = new lensvector[n_sourcepts_fit];
		//if ((sourcepts_lower_limit != NULL) and (sourcepts_upper_limit != NULL)) {
			//fitmodel->sourcepts_lower_limit = new lensvector[n_sourcepts_fit];
			//fitmodel->sourcepts_upper_limit = new lensvector[n_sourcepts_fit];
		//}
		//fitmodel->vary_sourcepts_x = new bool[n_sourcepts_fit];
		//fitmodel->vary_sourcepts_y = new bool[n_sourcepts_fit];
		fitmodel->source_redshifts = new double[n_sourcepts_fit];
		fitmodel->zfactors = new double*[n_sourcepts_fit];
		fitmodel->beta_factors = new double**[n_sourcepts_fit];
		for (i=0; i < n_sourcepts_fit; i++) {
			//cout << "source " << i << " beta matrix:\n";
			//fitmodel->vary_sourcepts_x[i] = vary_sourcepts_x[i];
			//fitmodel->vary_sourcepts_y[i] = vary_sourcepts_y[i];
			//fitmodel->sourcepts_fit[i][0] = sourcepts_fit[i][0];
			//fitmodel->sourcepts_fit[i][1] = sourcepts_fit[i][1];
			//if ((sourcepts_lower_limit != NULL) and (sourcepts_upper_limit != NULL)) {
				//fitmodel->sourcepts_lower_limit[i][0] = sourcepts_lower_limit[i][0];
				//fitmodel->sourcepts_upper_limit[i][0] = sourcepts_upper_limit[i][0];
			//}
			fitmodel->source_redshifts[i] = source_redshifts[i];
			fitmodel->zfactors[i] = new double[n_lens_redshifts];
			fitmodel->beta_factors[i] = new double*[n_lens_redshifts-1];
			for (j=0; j < n_lens_redshifts; j++) fitmodel->zfactors[i][j] = zfactors[i][j];
			for (j=0; j < n_lens_redshifts-1; j++) {
				fitmodel->beta_factors[i][j] = new double[j+1];
				for (k=0; k < j+1; k++) fitmodel->beta_factors[i][j][k] = beta_factors[i][j][k];
				//for (k=0; k < j+1; k++) cout << beta_factors[i][j][k] << " ";
				//cout << endl;
			}
			//cout << endl;
		}
		fitmodel->source_redshift_groups = source_redshift_groups;
	} else {
		fitmodel->image_pixel_data = image_pixel_data;
		fitmodel->load_pixel_grid_from_data();
		if (source_pixel_grid != NULL) {
			// we do this because some of the static source grid parameters will be changed during fit (really should reorganize so this is not an issue)
			delete source_pixel_grid;
			source_pixel_grid = NULL;
		}
	}

	fitmodel->nlens = nlens;
	fitmodel->lens_list = new LensProfile*[nlens];
	for (i=0; i < nlens; i++) {
		switch (lens_list[i]->get_lenstype()) {
			case KSPLINE:
				fitmodel->lens_list[i] = new LensProfile(lens_list[i]); break;
			case ALPHA:
				fitmodel->lens_list[i] = new Alpha((Alpha*) lens_list[i]); break;
			case PJAFFE:
				fitmodel->lens_list[i] = new PseudoJaffe((PseudoJaffe*) lens_list[i]); break;
			case nfw:
				fitmodel->lens_list[i] = new NFW((NFW*) lens_list[i]); break;
			case TRUNCATED_nfw:
				fitmodel->lens_list[i] = new Truncated_NFW((Truncated_NFW*) lens_list[i]); break;
			case CORED_nfw:
				fitmodel->lens_list[i] = new Cored_NFW((Cored_NFW*) lens_list[i]); break;
			case HERNQUIST:
				fitmodel->lens_list[i] = new Hernquist((Hernquist*) lens_list[i]); break;
			case EXPDISK:
				fitmodel->lens_list[i] = new ExpDisk((ExpDisk*) lens_list[i]); break;
			case SHEAR:
				fitmodel->lens_list[i] = new Shear((Shear*) lens_list[i]); break;
			case MULTIPOLE:
				fitmodel->lens_list[i] = new Multipole((Multipole*) lens_list[i]); break;
			case CORECUSP:
				fitmodel->lens_list[i] = new CoreCusp((CoreCusp*) lens_list[i]); break;
			case SERSIC_LENS:
				fitmodel->lens_list[i] = new SersicLens((SersicLens*) lens_list[i]); break;
			case CORED_SERSIC_LENS:
				fitmodel->lens_list[i] = new Cored_SersicLens((Cored_SersicLens*) lens_list[i]); break;
			case PTMASS:
				fitmodel->lens_list[i] = new PointMass((PointMass*) lens_list[i]); break;
			case SHEET:
				fitmodel->lens_list[i] = new MassSheet((MassSheet*) lens_list[i]); break;
			case DEFLECTION:
				fitmodel->lens_list[i] = new Deflection((Deflection*) lens_list[i]); break;
			case TABULATED:
				fitmodel->lens_list[i] = new Tabulated_Model((Tabulated_Model*) lens_list[i]); break;
			case QTABULATED:
				fitmodel->lens_list[i] = new QTabulated_Model((QTabulated_Model*) lens_list[i]); break;
			default:
				die("lens type not supported for fitting");
		}
		fitmodel->lens_list[i]->lens = fitmodel; // point to the fitmodel, since the cosmology may be varied (by varying H0, e.g.)
	}
	for (i=0; i < nlens; i++) {
		// if the lens is anchored to another lens, re-anchor so that it points to the corresponding
		// lens in fitmodel (the lens whose parameters will be varied)
		if (fitmodel->lens_list[i]->center_anchored==true) fitmodel->lens_list[i]->anchor_center_to_lens(lens_list[i]->get_center_anchor_number());
		if (fitmodel->lens_list[i]->anchor_special_parameter==true) {
			LensProfile *parameter_anchor_lens = fitmodel->lens_list[lens_list[i]->get_special_parameter_anchor_number()];
			fitmodel->lens_list[i]->assign_special_anchored_parameters(parameter_anchor_lens,1,false);
		}
		for (j=0; j < fitmodel->lens_list[i]->get_n_params(); j++) {
			if (fitmodel->lens_list[i]->anchor_parameter[j]==true) {
				LensProfile *parameter_anchor_lens = fitmodel->lens_list[lens_list[i]->parameter_anchor_lens[j]->lens_number];
				int paramnum = fitmodel->lens_list[i]->parameter_anchor_paramnum[j];
				fitmodel->lens_list[i]->assign_anchored_parameter(j,paramnum,true,true,lens_list[i]->parameter_anchor_ratio[j],lens_list[i]->parameter_anchor_exponent[j],parameter_anchor_lens);
			}
		}
	}

	fitmodel->n_sb = n_sb;
	if (n_sb > 0) {
		fitmodel->sb_list = new SB_Profile*[n_sb];
		for (i=0; i < n_sb; i++) {
			switch (sb_list[i]->get_sbtype()) {
				case GAUSSIAN:
					fitmodel->sb_list[i] = new Gaussian((Gaussian*) sb_list[i]); break;
				case SERSIC:
					fitmodel->sb_list[i] = new Sersic((Sersic*) sb_list[i]); break;
				case CORED_SERSIC:
					fitmodel->sb_list[i] = new Cored_Sersic((Cored_Sersic*) sb_list[i]); break;
				case SB_MULTIPOLE:
					fitmodel->sb_list[i] = new SB_Multipole((SB_Multipole*) sb_list[i]); break;
				case SHAPELET:
					fitmodel->sb_list[i] = new Shapelet((Shapelet*) sb_list[i]); break;
				case TOPHAT:
					fitmodel->sb_list[i] = new TopHat((TopHat*) sb_list[i]); break;
				default:
					die("surface brightness profile type not supported for fitting");
			}
		}
		for (i=0; i < n_sb; i++) {
			if (fitmodel->sb_list[i]->center_anchored==true) fitmodel->sb_list[i]->anchor_center_to_lens(fitmodel->lens_list, sb_list[i]->get_center_anchor_number());
		}
	}

	fitmodel->fitmethod = fitmethod;
	fitmodel->n_fit_parameters = n_fit_parameters;
	fitmodel->lensmodel_fit_parameters = lensmodel_fit_parameters;
	fitmodel->srcmodel_fit_parameters = srcmodel_fit_parameters;
	if ((fitmethod!=POWELL) and (fitmethod!=SIMPLEX)) fitmodel->setup_limits();

	// testing out copying plimits from lens to fitmodel
	//get_n_fit_parameters(n_fit_parameters);
	//boolvector use_penalty_limits(n_fit_parameters);
	//dvector lower(n_fit_parameters), upper(n_fit_parameters);
	//param_settings->get_penalty_limits(use_penalty_limits,lower,upper);

	//for (int i=0; i < n_fit_parameters; i++) { if (use_penalty_limits[i] == true) cout << "Using plimits for " << i << endl; }
	//if (param_settings->use_penalty_limits[8] == true) cout << "WOWOWWOWOW Using plimits for 8" << endl; 
	//fitmodel->param_settings->update_penalty_limits(use_penalty_limits,lower,upper);

	//return;

	if (open_chisq_logfile) {
		string logfile_str = fit_output_dir + "/" + fit_output_filename + ".log";
		if (group_id==0) {
			if (group_num > 0) {
				// if there is more than one MPI group evaluating the likelihood, output a separate file for each group
				stringstream groupstream;
				string groupstr;
				groupstream << group_num;
				groupstream >> groupstr;
				logfile_str += "." + groupstr;
			}
			fitmodel->logfile.open(logfile_str.c_str());
			fitmodel->logfile << setprecision(10);
		}
	}

	//param_settings->print_penalty_limits();
	//cout << "FITMODEL:" << endl;
	//fitmodel->param_settings->print_penalty_limits();
	fitmodel->update_parameter_list();
	//cout << endl;
	return true;
}

void QLens::update_anchored_parameters_and_redshift_data()
{
	for (int i=0; i < nlens; i++) {
		if (lens_list[i]->center_anchored) lens_list[i]->update_anchor_center();
		if (lens_list[i]->anchor_special_parameter) lens_list[i]->update_special_anchored_params();
		lens_list[i]->update_anchored_parameters();
	}
	for (int i=0; i < n_sb; i++) {
		if (sb_list[i]->center_anchored) sb_list[i]->update_anchor_center();
	}
	update_lens_redshift_data();
	reset_grid();
}

bool QLens::update_fitmodel(const double* params)
{
	bool status = true;
	int i, index=0;
	for (i=0; i < nlens; i++) {
		fitmodel->lens_list[i]->update_fit_parameters(params,index,status);
	}
	fitmodel->update_anchored_parameters_and_redshift_data();
	if (source_fit_mode==Point_Source) {
		if (!use_analytic_bestfit_src) {
			for (i=0; i < n_sourcepts_fit; i++) {
				if (fitmodel->vary_sourcepts_x[i]) fitmodel->sourcepts_fit[i][0] = params[index++];
				if (fitmodel->vary_sourcepts_y[i]) fitmodel->sourcepts_fit[i][1] = params[index++];
			}
		}
	} else if ((source_fit_mode == Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		for (i=0; i < n_sb; i++) {
			fitmodel->sb_list[i]->update_fit_parameters(params,index,status);
		}
	} else if (source_fit_mode == Pixellated_Source) {
		if ((vary_regularization_parameter) and (regularization_method != None)) fitmodel->regularization_parameter = params[index++];
		if (vary_pixel_fraction) fitmodel->pixel_fraction = params[index++];
		if (vary_srcgrid_xshift) fitmodel->srcgrid_xshift = params[index++];
		if (vary_srcgrid_yshift) fitmodel->srcgrid_yshift = params[index++];
		if (vary_srcgrid_size_scale) fitmodel->srcgrid_size_scale = params[index++];
		if (vary_magnification_threshold) fitmodel->pixel_magnification_threshold = params[index++];
	}
	if (source_fit_mode == Shapelet_Source) {
		if ((vary_regularization_parameter) and (regularization_method != None)) fitmodel->regularization_parameter = params[index++];
	}
	if (vary_hubble_parameter) {
		fitmodel->hubble = params[index++];
		if (fitmodel->hubble < 0) status = false; // do not allow negative Hubble parameter
		fitmodel->set_cosmology(fitmodel->omega_matter,0.04,fitmodel->hubble,2.215);
	}
	if (vary_omega_matter_parameter) {
		fitmodel->omega_matter = params[index++];
		if (fitmodel->omega_matter < 0) status = false; // do not allow negative Hubble parameter
		fitmodel->set_cosmology(fitmodel->omega_matter,0.04,fitmodel->hubble,2.215);
	}
	if (vary_syserr_pos_parameter) {
		fitmodel->syserr_pos = params[index++];
		if (fitmodel->syserr_pos < 0) status = false; // do not allow negative syserr_pos parameter
	}
	if (vary_wl_shear_factor_parameter) {
		fitmodel->wl_shear_factor = params[index++];
	}

	if (index != n_fit_parameters) die("Index didn't go through all the fit parameters (%i)",n_fit_parameters);
	return status;
}

void QLens::output_analytic_srcpos(lensvector *beta_i)
{
	// Note: beta_i needs to have the same size as the number of image sets being fit, or else a segmentation fault will occur
	int i,j;
	lensvector beta_ji;
	lensmatrix mag, magsqr;
	lensmatrix amatrix, ainv;
	lensvector bvec;
	lensmatrix jac;

	double siginv, src_norm;
	for (i=0; i < n_sourcepts_fit; i++) {
		amatrix[0][0] = amatrix[0][1] = amatrix[1][0] = amatrix[1][1] = 0;
		bvec[0] = bvec[1] = 0;
		beta_i[i][0] = beta_i[i][1] = 0;
		src_norm=0;
		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].use_in_chisq[j]) {
				if (use_magnification_in_chisq) {
					sourcept_jacobian(image_data[i].pos[j],beta_ji,jac,0,zfactors[i],beta_factors[i]);
					mag = jac.inverse();
					lensmatsqr(mag,magsqr);
					siginv = 1.0/(SQR(image_data[i].sigma_pos[j]) + syserr_pos*syserr_pos);
					amatrix[0][0] += magsqr[0][0]*siginv;
					amatrix[1][0] += magsqr[1][0]*siginv;
					amatrix[0][1] += magsqr[0][1]*siginv;
					amatrix[1][1] += magsqr[1][1]*siginv;
					bvec[0] += (magsqr[0][0]*beta_ji[0] + magsqr[0][1]*beta_ji[1])*siginv;
					bvec[1] += (magsqr[1][0]*beta_ji[0] + magsqr[1][1]*beta_ji[1])*siginv;
				} else {
					find_sourcept(image_data[i].pos[j],beta_ji,0,zfactors[i],beta_factors[i]);
					siginv = 1.0/(SQR(image_data[i].sigma_pos[j]) + syserr_pos*syserr_pos);
					beta_i[i][0] += beta_ji[0]*siginv;
					beta_i[i][1] += beta_ji[1]*siginv;
					src_norm += siginv;
				}
			}
		}
		if (use_magnification_in_chisq) {
			if (amatrix.invert(ainv)==false) return;
			beta_i[i] = ainv*bvec;
		} else {
			beta_i[i][0] /= src_norm;
			beta_i[i][1] /= src_norm;
		}
	}
	return;
}

void QLens::set_analytic_sourcepts()
{
	lensvector *srcpts = new lensvector[n_sourcepts_fit];
	output_analytic_srcpos(srcpts);
	for (int i=0; i < n_sourcepts_fit; i++) {
		sourcepts_fit[i][0] = srcpts[i][0];
		sourcepts_fit[i][1] = srcpts[i][1];
	}
	delete[] srcpts;
}

double QLens::chisq_pos_source_plane()
{
	int i,j;
	double chisq=0;
	int n_images_hi=0;
	lensvector delta_beta, delta_theta;
	lensmatrix mag, magsqr;
	lensmatrix amatrix, ainv;
	lensvector bvec;
	lensmatrix jac;
	lensvector src_bf;
	lensvector *beta;

	for (i=0; i < n_sourcepts_fit; i++) {
		if (image_data[i].n_images > n_images_hi) n_images_hi = image_data[i].n_images;
	}
	double* mag00 = new double[n_images_hi];
	double* mag11 = new double[n_images_hi];
	double* mag01 = new double[n_images_hi];
	lensvector* beta_ji = new lensvector[n_images_hi];

	double sigsq, signormfac, siginv, src_norm;
	if (syserr_pos == 0.0) signormfac = 0.0; // signormfac is the correction to chi-square to account for unknown systematic error
	for (i=0; i < n_sourcepts_fit; i++) {
		amatrix[0][0] = amatrix[0][1] = amatrix[1][0] = amatrix[1][1] = 0;
		bvec[0] = bvec[1] = 0;
		src_bf[0] = src_bf[1] = 0;
		src_norm=0;
		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].use_in_chisq[j]) {
				if (use_magnification_in_chisq) {
					sourcept_jacobian(image_data[i].pos[j],beta_ji[j],jac,0,zfactors[i],beta_factors[i]);
					mag = jac.inverse();
					mag00[j] = mag[0][0];
					mag01[j] = mag[0][1];
					mag11[j] = mag[1][1];

					if (use_analytic_bestfit_src) {
						lensmatsqr(mag,magsqr);
						siginv = 1.0/(SQR(image_data[i].sigma_pos[j]) + syserr_pos*syserr_pos);
						amatrix[0][0] += magsqr[0][0]*siginv;
						amatrix[1][0] += magsqr[1][0]*siginv;
						amatrix[0][1] += magsqr[0][1]*siginv;
						amatrix[1][1] += magsqr[1][1]*siginv;
						bvec[0] += (magsqr[0][0]*beta_ji[j][0] + magsqr[0][1]*beta_ji[j][1])*siginv;
						bvec[1] += (magsqr[1][0]*beta_ji[j][0] + magsqr[1][1]*beta_ji[j][1])*siginv;
					}
				} else {
					find_sourcept(image_data[i].pos[j],beta_ji[j],0,zfactors[i],beta_factors[i]);
					if (use_analytic_bestfit_src) {
						siginv = 1.0/(SQR(image_data[i].sigma_pos[j]) + syserr_pos*syserr_pos);
						src_bf[0] += beta_ji[j][0]*siginv;
						src_bf[1] += beta_ji[j][1]*siginv;
						src_norm += siginv;
					}
				}
			}
		}
		if (use_analytic_bestfit_src) {
			if (use_magnification_in_chisq) {
				if (amatrix.invert(ainv)==false) return 1e30;
				src_bf = ainv*bvec;
			} else {
				src_bf[0] /= src_norm;
				src_bf[1] /= src_norm;
			}
			beta = &src_bf;
		} else {
			beta = &sourcepts_fit[i];
		}

		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].use_in_chisq[j]) {
				delta_beta[0] = (*beta)[0] - beta_ji[j][0];
				delta_beta[1] = (*beta)[1] - beta_ji[j][1];
				sigsq = SQR(image_data[i].sigma_pos[j]);
				if (syserr_pos != 0.0) {
					 signormfac = 2*log(1.0 + syserr_pos*syserr_pos/sigsq);
					 sigsq += syserr_pos*syserr_pos;
				}
				if (use_magnification_in_chisq) {
					delta_theta[0] = mag00[j] * delta_beta[0] + mag01[j] * delta_beta[1];
					delta_theta[1] = mag01[j] * delta_beta[0] + mag11[j] * delta_beta[1];
					chisq += delta_theta.sqrnorm() / sigsq + signormfac;
				} else {
					chisq += delta_beta.sqrnorm() / sigsq + signormfac;
				}
			}
		}
	}
	delete[] mag00;
	delete[] mag11;
	delete[] mag01;
	delete[] beta_ji;
	if ((group_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	return chisq;
}

double QLens::chisq_pos_image_plane()
{
	int n_redshift_groups = source_redshift_groups.size()-1;
	int mpi_chunk=n_redshift_groups, mpi_start=0;
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

	//cout << "Running chisq..." << endl;
	if (group_np > 1) {
		if (group_np > n_redshift_groups) die("Number of MPI processes per group cannot be greater than number of source planes in data being fit");
		mpi_chunk = n_redshift_groups / group_np;
		mpi_start = group_id*mpi_chunk;
		if (group_id == group_np-1) mpi_chunk += (n_redshift_groups % group_np); // assign the remainder elements to the last mpi process
	}

	if (use_analytic_bestfit_src) set_analytic_sourcepts();

	double chisq=0, chisq_part=0;

	int n_images, n_tot_images=0, n_tot_images_part=0;
	double sigsq, signormfac, chisq_each_srcpt, dist;
	if (syserr_pos == 0.0) signormfac = 0.0; // signormfac is the correction to chi-square to account for unknown systematic error
	int i,j,k,m,n;
	for (m=mpi_start; m < mpi_start + mpi_chunk; m++) {
		create_grid(false,zfactors[source_redshift_groups[m]],beta_factors[source_redshift_groups[m]],m);
		for (i=source_redshift_groups[m]; i < source_redshift_groups[m+1]; i++) {
			chisq_each_srcpt = 0;
			image *img = get_images(sourcepts_fit[i], n_images, false);
			n_visible_images = n_images;
			bool *ignore = new bool[n_images];
			for (j=0; j < n_images; j++) ignore[j] = false;

			for (j=0; j < n_images; j++) {
				if ((!ignore[j]) and (abs(img[j].mag) < chisq_magnification_threshold)) {
					ignore[j] = true;
					n_visible_images--;
				}
				if ((chisq_imgsep_threshold > 0) and (!ignore[j])) {
					for (k=j+1; k < n_images; k++) {
						if (!ignore[k]) {
							dist = sqrt(SQR(img[k].pos[0] - img[j].pos[0]) + SQR(img[k].pos[1] - img[j].pos[1]));
							if (dist < chisq_imgsep_threshold) {
								ignore[k] = true;
								n_visible_images--;
							}
						}
					}
				}
			}

			n_tot_images_part += n_visible_images;
			if ((n_images_penalty==true) and (n_visible_images > image_data[i].n_images)) {
				chisq_part += 1e30;
				continue;
			}

			int n_dists = n_visible_images*image_data[i].n_images;
			double *distsqrs = new double[n_dists];
			int *data_k = new int[n_dists];
			int *model_j = new int[n_dists];
			n=0;
			for (k=0; k < image_data[i].n_images; k++) {
				for (j=0; j < n_images; j++) {
					if (ignore[j]) continue;
					distsqrs[n] = SQR(image_data[i].pos[k][0] - img[j].pos[0]) + SQR(image_data[i].pos[k][1] - img[j].pos[1]);
					data_k[n] = k;
					model_j[n] = j;
					n++;
				}
			}
			if (n != n_dists) die("count of all data-model image combinations does not equal expected number (%i vs %i)",n,n_dists);
			sort(n_dists,distsqrs,data_k,model_j);
			int *closest_image_j = new int[image_data[i].n_images];
			int *closest_image_k = new int[n_images];
			double *closest_distsqrs = new double[image_data[i].n_images];
			for (k=0; k < image_data[i].n_images; k++) closest_image_j[k] = -1;
			for (j=0; j < n_images; j++) closest_image_k[j] = -1;
			int m=0;
			int mmax = dmin(n_visible_images,image_data[i].n_images);
			for (n=0; n < n_dists; n++) {
				if ((closest_image_j[data_k[n]] == -1) and (closest_image_k[model_j[n]] == -1)) {
					closest_image_j[data_k[n]] = model_j[n];
					closest_image_k[model_j[n]] = data_k[n];
					closest_distsqrs[data_k[n]] = distsqrs[n];
					m++;
					if (m==mmax) n = n_dists; // force loop to exit
				}
			}

			for (k=0; k < image_data[i].n_images; k++) {
				sigsq = SQR(image_data[i].sigma_pos[k]);
				if (syserr_pos != 0.0) {
					 signormfac = 2*log(1.0 + syserr_pos*syserr_pos/sigsq);
					 sigsq += syserr_pos*syserr_pos;
				}
				if (closest_image_j[k] != -1) {
					if (image_data[i].use_in_chisq[k]) {
						chisq_each_srcpt += closest_distsqrs[k]/sigsq + signormfac;
					}
				} else {
					// add a penalty value to chi-square for not reproducing this data image; the distance is twice the maximum distance between any pair of images
					chisq_each_srcpt += 4*image_data[i].max_distsqr/sigsq + signormfac;
				}
			}
			chisq_part += chisq_each_srcpt;
			delete[] ignore;
			delete[] distsqrs;
			delete[] data_k;
			delete[] model_j;
			delete[] closest_image_j;
			delete[] closest_image_k;
			delete[] closest_distsqrs;
		}
	}
#ifdef USE_MPI
	//cout << "chisq_part=" << chisq_part << ", group_id=" << group_id << endl;
	MPI_Allreduce(&chisq_part, &chisq, 1, MPI_DOUBLE, MPI_SUM, sub_comm);
	MPI_Allreduce(&n_tot_images_part, &n_tot_images, 1, MPI_INT, MPI_SUM, sub_comm);
	MPI_Comm_free(&sub_comm);
#else
	chisq = chisq_part;
	n_tot_images = n_tot_images_part;
#endif

	if ((group_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	n_visible_images = n_tot_images; // save the total number of visible images produced
	return chisq;
}

double QLens::chisq_pos_image_plane_diagnostic(const bool verbose, const bool output_residuals_to_file, double &rms_imgpos_err, int &n_matched_images, const string output_filename)
//double QLens::chisq_pos_image_plane_diagnostic(const bool verbose, double &rms_imgpos_err, int &n_matched_images)
{
	int n_redshift_groups = source_redshift_groups.size()-1;
	int mpi_chunk=n_redshift_groups, mpi_start=0;
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

	if (group_np > 1) {
		if (group_np > n_redshift_groups) die("Number of MPI processes per group cannot be greater than number of source planes in data being fit");
		mpi_chunk = n_redshift_groups / group_np;
		mpi_start = group_id*mpi_chunk;
		if (group_id == group_np-1) mpi_chunk += (n_redshift_groups % group_np); // assign the remainder elements to the last mpi process
	}

	if (use_analytic_bestfit_src) set_analytic_sourcepts();

	double chisq=0, chisq_part=0, rms_part=0;
	int n_images, n_tot_images=0, n_tot_images_part=0, n_matched_images_part=0;
	double sigsq, signormfac, chisq_each_srcpt, n_matched_images_each_srcpt, rms_err_each_srcpt, dist;
	rms_imgpos_err = 0;
	n_matched_images = 0;
	vector<double> closest_chivals, closest_xvals_model, closest_yvals_model, closest_xvals_data, closest_yvals_data;

	if (syserr_pos == 0.0) signormfac = 0.0; // signormfac is the correction to chi-square to account for unknown systematic error
	int i,j,k,m,n;
	for (m=mpi_start; m < mpi_start + mpi_chunk; m++) {
		create_grid(false,zfactors[source_redshift_groups[m]],beta_factors[source_redshift_groups[m]],m);
		if ((mpi_id==0) and (verbose)) cout << endl << "zsrc=" << source_redshifts[source_redshift_groups[m]] << ": grid = (" << (grid_xcenter-grid_xlength/2) << "," << (grid_xcenter+grid_xlength/2) << ") x (" << (grid_ycenter-grid_ylength/2) << "," << (grid_ycenter+grid_ylength/2) << ")" << endl;
		for (i=source_redshift_groups[m]; i < source_redshift_groups[m+1]; i++) {
			chisq_each_srcpt = 0;
			n_matched_images_each_srcpt = 0;
			rms_err_each_srcpt = 0;
			image *img = get_images(sourcepts_fit[i], n_images, false);
			n_visible_images = n_images;
			bool *ignore = new bool[n_images];
			for (j=0; j < n_images; j++) ignore[j] = false;

			for (j=0; j < n_images; j++) {
				if ((!ignore[j]) and (abs(img[j].mag) < chisq_magnification_threshold)) {
					ignore[j] = true;
					n_visible_images--;
				}
				if ((chisq_imgsep_threshold > 0) and (!ignore[j])) {
					for (k=j+1; k < n_images; k++) {
						if (!ignore[k]) {
							dist = sqrt(SQR(img[k].pos[0] - img[j].pos[0]) + SQR(img[k].pos[1] - img[j].pos[1]));
							if (dist < chisq_imgsep_threshold) {
								ignore[k] = true;
								n_visible_images--;
							}
						}
					}
				}
			}

			n_tot_images_part += n_visible_images;
			if ((n_images_penalty==true) and (n_visible_images > image_data[i].n_images)) {
				chisq_part += 1e30;
				if ((mpi_id==0) and (verbose)) cout << "nimg_penalty incurred for source " << i << " (# model images = " << n_visible_images << ", # data images = " << image_data[i].n_images << ")" << endl;
			}

			int n_dists = n_visible_images*image_data[i].n_images;
			double *distsqrs = new double[n_dists];
			int *data_k = new int[n_dists];
			int *model_j = new int[n_dists];
			n=0;
			for (k=0; k < image_data[i].n_images; k++) {
				for (j=0; j < n_images; j++) {
					if (ignore[j]) continue;
					distsqrs[n] = SQR(image_data[i].pos[k][0] - img[j].pos[0]) + SQR(image_data[i].pos[k][1] - img[j].pos[1]);
					data_k[n] = k;
					model_j[n] = j;
					n++;
				}
			}
			if (n != n_dists) die("count of all data-model image combinations does not equal expected number (%i vs %i)",n,n_dists);
			sort(n_dists,distsqrs,data_k,model_j);
			int *closest_image_j = new int[image_data[i].n_images];
			int *closest_image_k = new int[n_images];
			double *closest_distsqrs = new double[image_data[i].n_images];
			for (k=0; k < image_data[i].n_images; k++) closest_image_j[k] = -1;
			for (j=0; j < n_images; j++) closest_image_k[j] = -1;
			int m=0;
			int mmax = dmin(n_visible_images,image_data[i].n_images);
			for (n=0; n < n_dists; n++) {
				if ((closest_image_j[data_k[n]] == -1) and (closest_image_k[model_j[n]] == -1)) {
					closest_image_j[data_k[n]] = model_j[n];
					closest_image_k[model_j[n]] = data_k[n];
					closest_distsqrs[data_k[n]] = distsqrs[n];
					m++;
					if (m==mmax) n = n_dists; // force loop to exit
				}
			}

			double chisq_this_img, chi_x, chi_y;
			int this_src_nimgs = image_data[i].n_images;
			for (k=0; k < image_data[i].n_images; k++) {
				sigsq = SQR(image_data[i].sigma_pos[k]);
				if (syserr_pos != 0.0) {
					 signormfac = 2*log(1.0 + syserr_pos*syserr_pos/sigsq);
					 sigsq += syserr_pos*syserr_pos;
				}
				if ((mpi_id==0) and (verbose)) cout << "source " << i << ", image " << k << ": ";
				if (closest_image_j[k] != -1) {
					if (image_data[i].use_in_chisq[k]) {
						rms_err_each_srcpt += closest_distsqrs[k];
						n_matched_images_each_srcpt++;
						chisq_this_img = closest_distsqrs[k]/sigsq + signormfac;
						chi_x = (img[closest_image_j[k]].pos[0]-image_data[i].pos[k][0])/sqrt(sigsq);
						chi_y = (img[closest_image_j[k]].pos[1]-image_data[i].pos[k][1])/sqrt(sigsq);
						closest_chivals.push_back(abs(chi_x));
						closest_xvals_model.push_back(img[closest_image_j[k]].pos[0]);
						closest_yvals_model.push_back(img[closest_image_j[k]].pos[1]);
						closest_xvals_data.push_back(image_data[i].pos[k][0]);
						closest_yvals_data.push_back(image_data[i].pos[k][1]);
						closest_chivals.push_back(abs(chi_y));
						closest_xvals_model.push_back(img[closest_image_j[k]].pos[0]);
						closest_yvals_model.push_back(img[closest_image_j[k]].pos[1]);
						closest_xvals_data.push_back(image_data[i].pos[k][0]);
						closest_yvals_data.push_back(image_data[i].pos[k][1]);

						if ((mpi_id==0) and (verbose)) cout << "chi_x=" << chi_x << ", chi_y=" << chi_y << ", chisq=" << chisq_this_img << " matched to (" << img[closest_image_j[k]].pos[0] << "," << img[closest_image_j[k]].pos[1] << ")" << endl << flush;
						chisq_each_srcpt += chisq_this_img;
					}
					else if ((mpi_id==0) and (verbose)) cout << "ignored in chisq,  matched to (" << img[closest_image_j[k]].pos[0] << "," << img[closest_image_j[k]].pos[1] << ")" << endl << flush;
				} else {
					// add a penalty value to chi-square for not reproducing this data image; the distance is twice the maximum distance between any pair of images
					chisq_this_img += 4*image_data[i].max_distsqr/sigsq + signormfac;
					if ((mpi_id==0) and (verbose)) cout << "chisq=" << chisq_this_img << " (not matched to model image)" << endl << flush;
					chisq_each_srcpt += chisq_this_img;
				}
			}
			if ((mpi_id==0) and (verbose)) {
				for (k=0; k < n_images; k++) {
					if (closest_image_k[k] == -1) cout << "EXTRA IMAGE: source " << i << ", model image " << k << " (" << img[k].pos[0] << "," << img[k].pos[1] << "), magnification = " << img[k].mag << endl << flush;
				}
			}

			chisq_part += chisq_each_srcpt;
			rms_part += rms_err_each_srcpt;
			n_matched_images_part += n_matched_images_each_srcpt;
			delete[] ignore;
			delete[] distsqrs;
			delete[] data_k;
			delete[] model_j;
			delete[] closest_image_j;
			delete[] closest_image_k;
			delete[] closest_distsqrs;
		}
	}
	if (closest_chivals.size() != 2*n_matched_images_part) die("WTFWEFAEFL:KASEL:FKAWEL:KFWEL:FK!");
	//cout << "HI THERE " << closest_chivals.size() << " " << n_matched_images_part << endl;
	if ((mpi_id==0) and (verbose)) cout << endl;
#ifdef USE_MPI
	MPI_Allreduce(&chisq_part, &chisq, 1, MPI_DOUBLE, MPI_SUM, sub_comm);
	MPI_Allreduce(&rms_part, &rms_imgpos_err, 1, MPI_DOUBLE, MPI_SUM, sub_comm);
	MPI_Allreduce(&n_tot_images_part, &n_tot_images, 1, MPI_INT, MPI_SUM, sub_comm);
	MPI_Allreduce(&n_matched_images_part, &n_matched_images, 1, MPI_INT, MPI_SUM, sub_comm);
#else
	chisq = chisq_part;
	n_tot_images = n_tot_images_part;
	n_matched_images = n_matched_images_part;
	rms_imgpos_err = rms_part;
#endif
	rms_imgpos_err = sqrt(rms_imgpos_err/n_matched_images);
	double *chi_all_images = new double[2*n_matched_images];
	double *model_xvals_all_images = new double[2*n_matched_images];
	double *model_yvals_all_images = new double[2*n_matched_images];
	double *data_xvals_all_images = new double[2*n_matched_images];
	double *data_yvals_all_images = new double[2*n_matched_images];
	int *nmatched_parts = new int[group_np];

#ifdef USE_MPI
	int id=0;
	nmatched_parts[group_id] = 2*n_matched_images_part;
	for (id=0; id < group_np; id++) {
		MPI_Bcast(nmatched_parts+id, 1, MPI_INT, id, sub_comm);
	}
	int indx=0;
	for (id=0; id < group_np; id++) {
		if (group_id==id) {
			for (i=0; i < nmatched_parts[id]; i++) {
				chi_all_images[indx+i] = closest_chivals[i];
				model_xvals_all_images[indx+i] = closest_xvals_model[i];
				model_yvals_all_images[indx+i] = closest_yvals_model[i];
				data_xvals_all_images[indx+i] = closest_xvals_data[i];
				data_yvals_all_images[indx+i] = closest_yvals_data[i];
			}
		}

		indx += nmatched_parts[id];
	}
	indx=0;
	for (id=0; id < group_np; id++) {
		MPI_Bcast(chi_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		MPI_Bcast(model_xvals_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		MPI_Bcast(model_yvals_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		MPI_Bcast(data_xvals_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		MPI_Bcast(data_yvals_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		indx += nmatched_parts[id];
	}
	MPI_Comm_free(&sub_comm);
#else
	for (i=0; i < 2*n_matched_images; i++) {
		chi_all_images[i] = closest_chivals[i];
		model_xvals_all_images[i] = closest_xvals_model[i];
		model_yvals_all_images[i] = closest_yvals_model[i];
		data_xvals_all_images[i] = closest_xvals_data[i];
		data_yvals_all_images[i] = closest_yvals_data[i];
	}
#endif
	if ((group_id==0) and (output_residuals_to_file)) {
		sort(2*n_matched_images,chi_all_images,model_xvals_all_images,model_yvals_all_images,data_xvals_all_images,data_yvals_all_images);
		double frac;
		ofstream outfile(output_filename.c_str());
		outfile << "#chi fraction(>chi) model_x model_y data_x data_y" << endl;
		for (i=0; i < 2*n_matched_images; i++) {
			j = 2*n_matched_images-i-1;
			frac = ((double) j)/(2.0*n_matched_images);
			outfile << chi_all_images[i] << " " << frac << " " << model_xvals_all_images[i] << " " << model_yvals_all_images[i] << " " << data_xvals_all_images[i] << " " << data_yvals_all_images[i] << endl;
		}
	}

	if ((mpi_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	n_visible_images = n_tot_images; // save the total number of visible images produced
	if ((mpi_id==0) and (verbose)) cout << "Number of matched image pairs = " << n_matched_images <<", rms_imgpos_error = " << rms_imgpos_err << endl << endl;
	delete[] nmatched_parts;
	delete[] chi_all_images;
	delete[] model_xvals_all_images;
	delete[] model_yvals_all_images;
	delete[] data_xvals_all_images;
	delete[] data_yvals_all_images;
	return chisq;
}

/*
// what was this function for? I can't remember
void QLens::output_imgplane_chisq_vals()
{
	int n_redshift_groups = source_redshift_groups.size()-1;
	int mpi_chunk=n_redshift_groups, mpi_start=0;
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

	if (group_np > 1) {
		if (group_np > n_redshift_groups) die("Number of MPI processes per group cannot be greater than number of source planes in data being fit");
		mpi_chunk = n_redshift_groups / group_np;
		mpi_start = group_id*mpi_chunk;
		if (group_id == group_np-1) mpi_chunk += (n_redshift_groups % group_np); // assign the remainder elements to the last mpi process
	}

	if (use_analytic_bestfit_src) {
		lensvector *srcpts = new lensvector[n_sourcepts_fit];
		output_analytic_srcpos(srcpts);
		for (int i=0; i < n_sourcepts_fit; i++) {
			cout << "src" << i << "_x=" << srcpts[i][0] << " src" << i << "_y=" << srcpts[i][1];
			cout << endl;
		}

		for (int i=0; i < n_sourcepts_fit; i++) {
			sourcepts_fit[i][0] = srcpts[i][0];
			sourcepts_fit[i][1] = srcpts[i][1];
		}
		delete[] srcpts;
	}

	int n_images, n_tot_images=0, n_tot_images_part=0;
	double sigsq, dist;
	int i,j,k,m,n;
	for (m=mpi_start; m < mpi_start + mpi_chunk; m++) {
		create_grid(false,zfactors[source_redshift_groups[m]],beta_factors[source_redshift_groups[m]],m);
		//if (mpi_id==0) cout << endl << "zsrc=" << source_redshifts[source_redshift_groups[m]] << ": grid = (" << (grid_xcenter-grid_xlength/2) << "," << (grid_xcenter+grid_xlength/2) << ") x (" << (grid_ycenter-grid_ylength/2) << "," << (grid_ycenter+grid_ylength/2) << ")" << endl;
		for (i=source_redshift_groups[m]; i < source_redshift_groups[m+1]; i++) {
			image *img = get_images(sourcepts_fit[i], n_images, false);
			n_visible_images = n_images;
			bool *ignore = new bool[n_images];
			for (j=0; j < n_images; j++) ignore[j] = false;

			for (j=0; j < n_images; j++) {
				if ((!ignore[j]) and (abs(img[j].mag) < chisq_magnification_threshold)) {
					ignore[j] = true;
					n_visible_images--;
				}
				if ((chisq_imgsep_threshold > 0) and (!ignore[j])) {
					for (k=j+1; k < n_images; k++) {
						if (!ignore[k]) {
							dist = sqrt(SQR(img[k].pos[0] - img[j].pos[0]) + SQR(img[k].pos[1] - img[j].pos[1]));
							if (dist < chisq_imgsep_threshold) {
								ignore[k] = true;
								n_visible_images--;
							}
						}
					}
				}
			}

			n_tot_images_part += n_visible_images;

			int n_dists = n_visible_images*image_data[i].n_images;
			double *distsqrs = new double[n_dists];
			int *data_k = new int[n_dists];
			int *model_j = new int[n_dists];
			n=0;
			for (k=0; k < image_data[i].n_images; k++) {
				for (j=0; j < n_images; j++) {
					if (ignore[j]) continue;
					distsqrs[n] = SQR(image_data[i].pos[k][0] - img[j].pos[0]) + SQR(image_data[i].pos[k][1] - img[j].pos[1]);
					data_k[n] = k;
					model_j[n] = j;
					n++;
				}
			}
			if (n != n_dists) die("count of all data-model image combinations does not equal expected number (%i vs %i)",n,n_dists);
			sort(n_dists,distsqrs,data_k,model_j);
			int *closest_image_j = new int[image_data[i].n_images];
			int *closest_image_k = new int[n_images];
			double *closest_distsqrs = new double[image_data[i].n_images];
			for (k=0; k < image_data[i].n_images; k++) closest_image_j[k] = -1;
			for (j=0; j < n_images; j++) closest_image_k[j] = -1;
			int m=0;
			int mmax = dmin(n_visible_images,image_data[i].n_images);
			for (n=0; n < n_dists; n++) {
				if ((closest_image_j[data_k[n]] == -1) and (closest_image_k[model_j[n]] == -1)) {
					closest_image_j[data_k[n]] = model_j[n];
					closest_image_k[model_j[n]] = data_k[n];
					closest_distsqrs[data_k[n]] = distsqrs[n];
					m++;
					if (m==mmax) n = n_dists; // force loop to exit
				}
			}

			for (k=0; k < image_data[i].n_images; k++) {
				sigsq = SQR(image_data[i].sigma_pos[k]);
				if (syserr_pos != 0.0) {
					 sigsq += syserr_pos*syserr_pos;
				}
					if (closest_image_j[k] != -1) {
						if (image_data[i].use_in_chisq[k]) {
							//if (mpi_id==0) cout << "chisq=" << chisq_this_img << " matched to (" << img[closest_image_j[k]].pos[0] << "," << img[closest_image_j[k]].pos[1] << ")" << endl << flush;
							//cout << (img[closest_image_j[k]].pos[0]-image_data[i].pos[k][0])/sqrt(sigsq) << " " << (img[closest_image_j[k]].pos[1]-image_data[i].pos[k][1])/sqrt(sigsq) << endl << flush;
							cout << abs((img[closest_image_j[k]].pos[0]-image_data[i].pos[k][0])/sqrt(sigsq)) << " " << image_data[i].pos[k][0] << " " << image_data[i].pos[k][1] << " " << img[closest_image_j[k]].pos[0] << " " << img[closest_image_j[k]].pos[1] << endl;
							cout << abs((img[closest_image_j[k]].pos[1]-image_data[i].pos[k][1])/sqrt(sigsq)) << " " << image_data[i].pos[k][0] << " " << image_data[i].pos[k][1] << " " << img[closest_image_j[k]].pos[0] << " " << img[closest_image_j[k]].pos[1] << endl;
							//cout << img[closest_image_j[k]].pos[0] << " " << img[closest_image_j[k]].pos[1] << endl;
							//cout << image_data[i].pos[k][0] << " " << image_data[i].pos[k][1] << endl << endl;
						}
						//else if (mpi_id==0) cout << "ignored in chisq,  matched to (" << img[closest_image_j[k]].pos[0] << "," << img[closest_image_j[k]].pos[1] << ")" << endl << flush;
					}
			}

			delete[] ignore;
			delete[] distsqrs;
			delete[] data_k;
			delete[] model_j;
			delete[] closest_image_j;
			delete[] closest_image_k;
			delete[] closest_distsqrs;
		}
	}
	if (mpi_id==0) cout << endl;
#ifdef USE_MPI
	MPI_Comm_free(&sub_comm);
#endif

	n_visible_images = n_tot_images; // save the total number of visible images produced
	return;
}
*/

void QLens::output_model_source_flux(double *bestfit_flux)
{
	double chisq=0;
	int n_total_images=0;
	int i,j,k=0;

	for (i=0; i < n_sourcepts_fit; i++)
		for (j=0; j < image_data[i].n_images; j++) n_total_images++;
	double image_mag;

	lensmatrix jac;
	for (i=0; i < n_sourcepts_fit; i++) {
		double num=0, denom=0;
		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_f[j]==0) { k++; continue; }
			hessian(image_data[i].pos[j],jac,zfactors[i],beta_factors[i]);
			jac[0][0] = 1 - jac[0][0];
			jac[1][1] = 1 - jac[1][1];
			jac[0][1] = -jac[0][1];
			jac[1][0] = -jac[1][0];
			image_mag = 1.0/determinant(jac);
			if (include_parity_in_chisq) {
				num += image_data[i].flux[j] * image_mag / SQR(image_data[i].sigma_f[j]);
			} else {
				num += abs(image_data[i].flux[j] * image_mag) / SQR(image_data[i].sigma_f[j]);
			}
			denom += SQR(image_mag/image_data[i].sigma_f[j]);
			k++;
		}
		if (denom==0) bestfit_flux[i] = -1; // indicates we cannot find the source flux
		else bestfit_flux[i] = num/denom;
	}
}

double QLens::chisq_flux()
{
	double chisq=0;
	int n_images_hi=0;
	int i,j,k;

	for (i=0; i < n_sourcepts_fit; i++) {
		if (image_data[i].n_images > n_images_hi) n_images_hi = image_data[i].n_images;
	}
	double* image_mags = new double[n_images_hi];

	lensmatrix jac;
	double flux_src, num, denom;
	for (i=0; i < n_sourcepts_fit; i++) {
		k=0; num=0; denom=0;
		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_f[j]==0) { k++; continue; }
			hessian(image_data[i].pos[j],jac,zfactors[i],beta_factors[i]);
			jac[0][0] = 1 - jac[0][0];
			jac[1][1] = 1 - jac[1][1];
			jac[0][1] = -jac[0][1];
			jac[1][0] = -jac[1][0];
			image_mags[k] = 1.0/determinant(jac);
			if (include_parity_in_chisq) {
				num += image_data[i].flux[j] * image_mags[k] / SQR(image_data[i].sigma_f[j]);
			} else {
				num += abs(image_data[i].flux[j] * image_mags[k]) / SQR(image_data[i].sigma_f[j]);
			}
			denom += SQR(image_mags[k]/image_data[i].sigma_f[j]);
			k++;
		}

		if (fix_source_flux) {
			flux_src = source_flux; // only one source flux value is currently supported; later this should be generalized so that
											// some fluxes can be fixed and others parameterized
		}
		else {
			// the source flux is calculated analytically, rather than including it as a fit parameter (see Keeton 2001, section 4.2)
			flux_src = num / denom;
		}

		k=0;
		if (include_parity_in_chisq) {
			for (j=0; j < image_data[i].n_images; j++) {
				if (image_data[i].sigma_f[j]==0) { k++; continue; }
				chisq += SQR((image_data[i].flux[j] - image_mags[k++]*flux_src)/image_data[i].sigma_f[j]);
			}
		} else {
			for (j=0; j < image_data[i].n_images; j++) {
				if (image_data[i].sigma_f[j]==0) { k++; continue; }
				chisq += SQR((abs(image_data[i].flux[j]) - abs(image_mags[k++]*flux_src))/image_data[i].sigma_f[j]);
			}
		}
	}

	delete[] image_mags;
	return chisq;
}

double QLens::chisq_time_delays()
{
	double chisq=0;
	int n_images_hi=0;
	int i,j,k;

	for (i=0; i < n_sourcepts_fit; i++) {
		if (image_data[i].n_images > n_images_hi) n_images_hi = image_data[i].n_images;
	}

	double td_factor;
	double* time_delays_obs = new double[n_images_hi];
	double* time_delays_mod = new double[n_images_hi];
	double min_td_obs, min_td_mod;
	double pot;
	lensvector beta_ij;
	for (k=0, i=0; i < n_sourcepts_fit; i++) {
		td_factor = time_delay_factor_arcsec(lens_redshift,source_redshifts[i]);
		min_td_obs=1e30;
		min_td_mod=1e30;
		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_t[j]==0) continue;
			find_sourcept(image_data[i].pos[j],beta_ij,0,zfactors[i],beta_factors[i]);
			pot = potential(image_data[i].pos[j],zfactors[i],beta_factors[i]);
			time_delays_mod[j] = 0.5*(SQR(image_data[i].pos[j][0] - beta_ij[0]) + SQR(image_data[i].pos[j][1] - beta_ij[1])) - pot;
			if (time_delays_mod[j] < min_td_mod) min_td_mod = time_delays_mod[j];

			if (image_data[i].time_delays[j] < min_td_obs) min_td_obs = image_data[i].time_delays[j];
		}
		for (k=0, j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_t[j]==0) { k++; continue; }
			time_delays_mod[k] -= min_td_mod;
			if (time_delays_mod[k] != 0.0) time_delays_mod[k] *= td_factor; // td_factor contains the cosmological factors and is in units of days
			time_delays_obs[k] = image_data[i].time_delays[j] - min_td_obs;
			k++;
		}
		for (k=0, j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_t[j]==0) { k++; continue; }
			chisq += SQR((time_delays_obs[k] - time_delays_mod[k]) / image_data[i].sigma_t[j]);
			k++;
		}
	}
	if (chisq==0) warn("no time delay information has been used for chi-square");

	delete[] time_delays_obs;
	delete[] time_delays_mod;
	return chisq;
}

double QLens::chisq_weak_lensing()
{
	int i,j,nsrc = weak_lensing_data.n_sources;
	if (nsrc==0) return 0;
	double chisq=0;
	double g1,g2;
	double **zfacs = new double*[nsrc];
	for (i=0; i < nsrc; i++) {
		zfacs[i] = new double[n_lens_redshifts];
	}
	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif

		#pragma omp for private(i,j,g1,g2) schedule(static) reduction(+:chisq)
		for (i=0; i < nsrc; i++) {
			for (j=0; j < n_lens_redshifts; j++) {
				zfacs[i][j] = kappa_ratio(lens_redshifts[j],weak_lensing_data.zsrc[i],reference_source_redshift);
			}
			reduced_shear_components(weak_lensing_data.pos[i],g1,g2,thread,zfacs[i]);
			chisq += SQR((wl_shear_factor*g1-weak_lensing_data.reduced_shear1[i])/weak_lensing_data.sigma_shear1[i]) + SQR((wl_shear_factor*g2-weak_lensing_data.reduced_shear2[i])/weak_lensing_data.sigma_shear2[i]);

		}
	}
	for (i=0; i < nsrc; i++) delete[] zfacs[i];
	delete[] zfacs;
	return chisq;
}

bool QLens::output_weak_lensing_chivals(string filename)
{
	int i,j,nsrc = weak_lensing_data.n_sources;
	if (nsrc==0) return false;
	ofstream chifile(filename.c_str());
	double chi1, chi2;
	double g1,g2;
	double **zfacs = new double*[nsrc];
	for (i=0; i < nsrc; i++) {
		zfacs[i] = new double[n_lens_redshifts];
	}
	for (i=0; i < nsrc; i++) {
		for (j=0; j < n_lens_redshifts; j++) {
			zfacs[i][j] = kappa_ratio(lens_redshifts[j],weak_lensing_data.zsrc[i],reference_source_redshift);
		}
		reduced_shear_components(weak_lensing_data.pos[i],g1,g2,0,zfacs[i]);
		chi1 = (wl_shear_factor*g1-weak_lensing_data.reduced_shear1[i])/weak_lensing_data.sigma_shear1[i];
		chi2 = (wl_shear_factor*g2-weak_lensing_data.reduced_shear2[i])/weak_lensing_data.sigma_shear2[i];
		chifile << chi1 << " " << chi2 << endl;
	}
	for (i=0; i < nsrc; i++) delete[] zfacs[i];
	delete[] zfacs;
	return true;
}

void QLens::get_automatic_initial_stepsizes(dvector& stepsizes)
{
	int i, index=0;
	for (i=0; i < nlens; i++) lens_list[i]->get_auto_stepsizes(stepsizes,index);
	if (source_fit_mode==Point_Source) {
		if (!use_analytic_bestfit_src) {
			if (nlens > 0) {
				double avg_srcdist;
				int j,k,n_srcpts,n_src_pairs;
				for (i=0; i < n_sourcepts_fit; i++) {
					avg_srcdist=0;
					n_src_pairs=0;
					n_srcpts = image_data[i].n_images;
					lensvector *srcpts = new lensvector[n_srcpts];
					for (j=0; j < n_srcpts; j++) {
						find_sourcept(image_data[i].pos[j],srcpts[j],0,zfactors[i],beta_factors[i]);
						for (k=0; k < j; k++) {
							avg_srcdist += sqrt(SQR(srcpts[j][0] - srcpts[k][0]) + SQR(srcpts[j][1] - srcpts[k][1]));
							n_src_pairs++;
						}
					}
					avg_srcdist /= n_src_pairs;
					if (vary_sourcepts_x[i]) stepsizes[index++] = 0.25*avg_srcdist;
					if (vary_sourcepts_y[i]) stepsizes[index++] = 0.25*avg_srcdist;
					delete[] srcpts;
				}
			} else {
				for (i=0; i < n_sourcepts_fit; i++) {
					if (vary_sourcepts_x[i]) stepsizes[index++] = 0.1*grid_xlength; // nothing else to use, since there's no lens model yet
					if (vary_sourcepts_y[i]) stepsizes[index++] = 0.1*grid_ylength;
				}
			}
		}
	}
	else if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		for (i=0; i < n_sb; i++) sb_list[i]->get_auto_stepsizes(stepsizes,index);
	}
	if ((vary_regularization_parameter) and ((source_fit_mode==Pixellated_Source) or (source_fit_mode==Shapelet_Source)) and (regularization_method != None)) {
		stepsizes[index++] = 0.33*regularization_parameter;
	}
	if (vary_pixel_fraction) stepsizes[index++] = 0.3;
	if (vary_srcgrid_xshift) stepsizes[index++] = 0.3;
	if (vary_srcgrid_yshift) stepsizes[index++] = 0.3;
	if (vary_srcgrid_size_scale) stepsizes[index++] = 0.3;
	if (vary_magnification_threshold) stepsizes[index++] = 0.3;
	if (vary_hubble_parameter) stepsizes[index++] = 0.3;
	if (vary_omega_matter_parameter) stepsizes[index++] = 0.3;
	if (vary_syserr_pos_parameter) stepsizes[index++] = 0.1;
	if (vary_wl_shear_factor_parameter) stepsizes[index++] = 0.1;
	if (index != n_fit_parameters) die("Index didn't go through all the fit parameters when setting default stepsizes (%i vs %i)",index,n_fit_parameters);
}

void QLens::set_default_plimits()
{
	get_n_fit_parameters(n_fit_parameters);
	boolvector use_penalty_limits(n_fit_parameters);
	dvector lower(n_fit_parameters), upper(n_fit_parameters);
	int i, index=0;
	for (i=0; i < nlens; i++) {
		lens_list[i]->get_auto_ranges(use_penalty_limits,lower,upper,index);
	}
	if (source_fit_mode==Point_Source) {
		if (!use_analytic_bestfit_src) {
			for (i=0; i < n_sourcepts_fit; i++) {
				//if (vary_sourcepts_x[i]) index++;
				//if (vary_sourcepts_y[i]) index++;
				if (vary_sourcepts_x[i]) { use_penalty_limits[index]=false; index++; }
				if (vary_sourcepts_y[i]) { use_penalty_limits[index]=false; index++; }
			}
		}
	}
	else if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		for (i=0; i < n_sb; i++) sb_list[i]->get_auto_ranges(use_penalty_limits,lower,upper,index);
	}
	if ((vary_regularization_parameter) and ((source_fit_mode==Pixellated_Source) or (source_fit_mode==Shapelet_Source)) and (regularization_method != None)) {
		index++;
	}
	if (vary_pixel_fraction) index++;
	if (vary_srcgrid_xshift) index++;
	if (vary_srcgrid_yshift) index++;
	if (vary_srcgrid_size_scale) index++;
	if (vary_magnification_threshold) index++;
	if (vary_hubble_parameter) index++;
	if (vary_omega_matter_parameter) index++;
	if (vary_syserr_pos_parameter) index++;
	if (vary_wl_shear_factor_parameter) index++;
	if (index != n_fit_parameters) die("Index didn't go through all the fit parameters when setting default ranges (%i vs %i)",index,n_fit_parameters);
	param_settings->update_penalty_limits(use_penalty_limits,lower,upper);
}

void QLens::get_n_fit_parameters(int &nparams)
{
	lensmodel_fit_parameters = 0;
	srcmodel_fit_parameters = 0;
	for (int i=0; i < nlens; i++) lensmodel_fit_parameters += lens_list[i]->get_n_vary_params();
	nparams = lensmodel_fit_parameters;
	if (source_fit_mode==Point_Source) {
		if (!use_analytic_bestfit_src) {
			for (int i=0; i < n_sourcepts_fit; i++) {
				if (vary_sourcepts_x[i]) nparams++;
				if (vary_sourcepts_y[i]) nparams++;
			}
		}
	}
	else if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		for (int i=0; i < n_sb; i++)
			srcmodel_fit_parameters += sb_list[i]->get_n_vary_params();
		nparams += srcmodel_fit_parameters;
	}
	if ((vary_regularization_parameter) and ((source_fit_mode==Pixellated_Source) or (source_fit_mode==Shapelet_Source)) and (regularization_method != None)) nparams++;
	if (vary_pixel_fraction) nparams++;
	if (vary_srcgrid_xshift) nparams++;
	if (vary_srcgrid_yshift) nparams++;
	if (vary_srcgrid_size_scale) nparams++;
	if (vary_magnification_threshold) nparams++;
	if (vary_hubble_parameter) nparams++;
	if (vary_omega_matter_parameter) nparams++;
	if (vary_syserr_pos_parameter) nparams++;
	if (vary_wl_shear_factor_parameter) nparams++;
}

bool QLens::setup_fit_parameters(bool include_limits)
{
	//if (source_fit_mode==Point_Source) {
		//if (image_data==NULL) { warn("cannot do fit; image data points have not been loaded"); return false; }
		//if (sourcepts_fit==NULL) { warn("cannot do fit; initial source parameters have not been defined"); return false; }
	//} else if (source_fit_mode==Pixellated_Source) {
		//if (image_pixel_data==NULL) { warn("cannot do fit; image pixel data has not been loaded"); return false; }
	//}
	if (nlens==0) { warn("cannot do fit; no lens models have been defined"); return false; }
	get_n_fit_parameters(n_fit_parameters);
	if (n_fit_parameters==0) { warn("cannot do fit; no parameters are being varied"); return false; }
	fitparams.input(n_fit_parameters);
	int index = 0;
	for (int i=0; i < nlens; i++) lens_list[i]->get_fit_parameters(fitparams,index);
	if (index != lensmodel_fit_parameters) die("Index didn't go through all the lens model fit parameters (%i vs %i)",index,lensmodel_fit_parameters);
	if (source_fit_mode==Point_Source) {
		if (!use_analytic_bestfit_src) {
			for (int i=0; i < n_sourcepts_fit; i++) {
				if (vary_sourcepts_x[i]) fitparams[index++] = sourcepts_fit[i][0];
				if (vary_sourcepts_y[i]) fitparams[index++] = sourcepts_fit[i][1];
			}
		}
	} else if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		for (int i=0; i < n_sb; i++) sb_list[i]->get_fit_parameters(fitparams,index);
		int expected_index = lensmodel_fit_parameters + srcmodel_fit_parameters;
		if (index != expected_index) die("Index didn't go through all the lens+source model fit parameters (%i vs %i)",index,expected_index);
	}
	if ((vary_regularization_parameter) and ((source_fit_mode==Pixellated_Source) or (source_fit_mode==Shapelet_Source)) and (regularization_method != None)) fitparams[index++] = regularization_parameter;
	if (vary_pixel_fraction) fitparams[index++] = pixel_fraction;
	if (vary_srcgrid_xshift) fitparams[index++] = srcgrid_xshift;
	if (vary_srcgrid_yshift) fitparams[index++] = srcgrid_yshift;
	if (vary_srcgrid_size_scale) fitparams[index++] = srcgrid_size_scale;
	if (vary_magnification_threshold) fitparams[index++] = pixel_magnification_threshold;
	if (vary_hubble_parameter) fitparams[index++] = hubble;
	if (vary_omega_matter_parameter) fitparams[index++] = omega_matter;
	if (vary_syserr_pos_parameter) fitparams[index++] = syserr_pos;
	if (vary_wl_shear_factor_parameter) fitparams[index++] = wl_shear_factor;

	get_parameter_names();
	dvector stepsizes(n_fit_parameters);
	get_automatic_initial_stepsizes(stepsizes);
	param_settings->update_params(n_fit_parameters,fit_parameter_names,stepsizes.array());
	//set_default_plimits();
	param_settings->transform_parameters(fitparams.array());
	transformed_parameter_names.resize(n_fit_parameters);
	transformed_latex_parameter_names.resize(n_fit_parameters);
	param_settings->transform_parameter_names(fit_parameter_names.data(),transformed_parameter_names.data(),latex_parameter_names.data(),transformed_latex_parameter_names.data());

	if (include_limits) return setup_limits();
	return true;
}

bool QLens::setup_limits()
{
	upper_limits.input(n_fit_parameters);
	lower_limits.input(n_fit_parameters);
	upper_limits_initial.input(n_fit_parameters);
	lower_limits_initial.input(n_fit_parameters);
	int index=0;
	for (int i=0; i < nlens; i++) {
		if ((lens_list[i]->get_n_vary_params() > 0) and (lens_list[i]->get_limits(lower_limits,upper_limits,lower_limits_initial,upper_limits_initial,index)==false)) { warn("cannot do fit; limits have not been defined for lens %i",i); return false; }
	}
	if (index != lensmodel_fit_parameters) die("index didn't go through all the lens model fit parameters when setting upper/lower limits");
	if (source_fit_mode==Point_Source) {
		if (!use_analytic_bestfit_src) {
			for (int i=0; i < n_sourcepts_fit; i++) {
				if (vary_sourcepts_x[i]) {
					lower_limits[index] = sourcepts_lower_limit[i][0];
					lower_limits_initial[index] = lower_limits[index]; // make it possible to specify initial limits for source point!
					upper_limits[index] = sourcepts_upper_limit[i][0];
					upper_limits_initial[index] = upper_limits[index]; // make it possible to specify initial limits for source point!
					index++;
				}
				if (vary_sourcepts_y[i]) {
					lower_limits[index] = sourcepts_lower_limit[i][1];
					lower_limits_initial[index] = lower_limits[index]; // make it possible to specify initial limits for source point!
					upper_limits[index] = sourcepts_upper_limit[i][1];
					upper_limits_initial[index] = upper_limits[index]; // make it possible to specify initial limits for source point!
					index++;
				}
			}
		}
	} else if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		for (int i=0; i < n_sb; i++) {
			if ((sb_list[i]->get_n_vary_params() > 0) and (sb_list[i]->get_limits(lower_limits,upper_limits,lower_limits_initial,upper_limits_initial,index)==false)) { warn("cannot do fit; limits have not been defined for source %i",i); return false; }
		}
		int expected_index = lensmodel_fit_parameters + srcmodel_fit_parameters;
		if (index != expected_index) die("index didn't go through all the lens+source model fit parameters when setting upper/lower limits (%i vs %i)", index, expected_index);
	}
	if ((source_fit_mode == Pixellated_Source) or (source_fit_mode==Shapelet_Source)) {
		if ((vary_regularization_parameter) and (regularization_method != None)) {
			if ((regularization_parameter_lower_limit==1e30) or (regularization_parameter_upper_limit==1e30)) { warn("lower/upper limits must be set for regularization parameter (see 'regparam') before doing fit"); return false; }
			lower_limits[index] = regularization_parameter_lower_limit;
			lower_limits_initial[index] = lower_limits[index];
			upper_limits[index] = regularization_parameter_upper_limit;
			upper_limits_initial[index] = upper_limits[index];
			index++;
		}
		if (source_fit_mode==Pixellated_Source) {
			if (vary_pixel_fraction) {
				lower_limits[index] = pixel_fraction_lower_limit;
				lower_limits_initial[index] = lower_limits[index];
				upper_limits[index] = pixel_fraction_upper_limit;
				upper_limits_initial[index] = upper_limits[index];
				index++;
			}
			if (vary_srcgrid_xshift) {
				lower_limits[index] = srcgrid_xshift_lower_limit;
				lower_limits_initial[index] = lower_limits[index];
				upper_limits[index] = srcgrid_xshift_upper_limit;
				upper_limits_initial[index] = upper_limits[index];
				index++;
			}
			if (vary_srcgrid_yshift) {
				lower_limits[index] = srcgrid_yshift_lower_limit;
				lower_limits_initial[index] = lower_limits[index];
				upper_limits[index] = srcgrid_yshift_upper_limit;
				upper_limits_initial[index] = upper_limits[index];
				index++;
			}
			if (vary_srcgrid_size_scale) {
				lower_limits[index] = srcgrid_size_scale_lower_limit;
				lower_limits_initial[index] = lower_limits[index];
				upper_limits[index] = srcgrid_size_scale_upper_limit;
				upper_limits_initial[index] = upper_limits[index];
				index++;
			}

			if (vary_magnification_threshold) {
				lower_limits[index] = pixel_magnification_threshold_lower_limit;
				lower_limits_initial[index] = lower_limits[index];
				upper_limits[index] = pixel_magnification_threshold_upper_limit;
				upper_limits_initial[index] = upper_limits[index];
				index++;
			}
		}
	}
	if (vary_hubble_parameter) {
		lower_limits[index] = hubble_lower_limit;
		lower_limits_initial[index] = lower_limits[index];
		upper_limits[index] = hubble_upper_limit;
		upper_limits_initial[index] = upper_limits[index];
		index++;
	}
	if (vary_omega_matter_parameter) {
		lower_limits[index] = omega_matter_lower_limit;
		lower_limits_initial[index] = lower_limits[index];
		upper_limits[index] = omega_matter_upper_limit;
		upper_limits_initial[index] = upper_limits[index];
		index++;
	}
	if (vary_syserr_pos_parameter) {
		lower_limits[index] = syserr_pos_lower_limit;
		lower_limits_initial[index] = lower_limits[index];
		upper_limits[index] = syserr_pos_upper_limit;
		upper_limits_initial[index] = upper_limits[index];
		index++;
	}
	if (vary_wl_shear_factor_parameter) {
		lower_limits[index] = wl_shear_factor_lower_limit;
		lower_limits_initial[index] = lower_limits[index];
		upper_limits[index] = wl_shear_factor_upper_limit;
		upper_limits_initial[index] = upper_limits[index];
		index++;
	}
	if (index != n_fit_parameters) die("index didn't go through all the fit parameters when setting upper/lower limits (%i expected, %i found)",n_fit_parameters,index);
	param_settings->transform_limits(lower_limits.array(),upper_limits.array());
	param_settings->transform_limits(lower_limits_initial.array(),upper_limits_initial.array());
	param_settings->set_prior_norms(lower_limits.array(),upper_limits.array());
	for (int i=0; i < n_fit_parameters; i++) {
		if (lower_limits[i] > upper_limits[i]) {
			double temp = upper_limits[i]; upper_limits[i] = lower_limits[i]; lower_limits[i] = temp;
		}
		if (lower_limits_initial[i] > upper_limits_initial[i]) {
			double temp = upper_limits_initial[i]; upper_limits_initial[i] = lower_limits_initial[i]; lower_limits_initial[i] = temp;
		}
	}
	return true;
}

void QLens::get_parameter_names()
{
	get_n_fit_parameters(n_fit_parameters);
	fit_parameter_names.clear();
	latex_parameter_names.clear();
	vector<string> latex_parameter_subscripts;
	int i,j;
	for (i=0; i < nlens; i++) {
		lens_list[i]->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);
	}
	if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		int srcparams_start = fit_parameter_names.size();
		for (i=0; i < n_sb; i++) {
			sb_list[i]->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);
		}
		for (i=srcparams_start; i < fit_parameter_names.size(); i++) {
			fit_parameter_names[i] += "_src";
		}
	}
	// find any parameters with matching names and number them so they can be distinguished
	int count, n_names;
	n_names = fit_parameter_names.size();

	//if (source_fit_mode==Parameterized_Source) {
		//for (i=0; i < n_names; i++) {
			//cout << "PARAM " << i << ": " << fit_parameter_names[i] << endl;
		//}
		//die();
	//}

	string *new_parameter_names = new string[n_names];
	for (i=0; i < n_names; i++) {
		count=1;
		new_parameter_names[i] = fit_parameter_names[i];
		for (j=i+1; j < n_names; j++) {
			if (fit_parameter_names[j]==fit_parameter_names[i]) {
				if (count==1) {
					stringstream countstr;
					string countstring;
					countstr << count;
					countstr >> countstring;
					if (isdigit(new_parameter_names[i].at(new_parameter_names[i].length()-1))) new_parameter_names[i] += "_"; // in case parameter name already ends with a number
					new_parameter_names[i] += countstring;
					if (latex_parameter_subscripts[i].empty()) latex_parameter_subscripts[i] = countstring;
					else latex_parameter_subscripts[i] += "," + countstring;
					count++;
				}
				stringstream countstr;
				string countstring;
				countstr << count;
				countstr >> countstring;
				if (isdigit(fit_parameter_names[j].at(fit_parameter_names[j].length()-1))) fit_parameter_names[j] += "_"; // in case parameter name already ends with a number
				fit_parameter_names[j] += countstring;
				if (latex_parameter_subscripts[j].empty()) latex_parameter_subscripts[j] = countstring;
				else latex_parameter_subscripts[j] += "," + countstring;
				count++;
			}
		}
		fit_parameter_names[i] = new_parameter_names[i];
	}
	delete[] new_parameter_names;
	if (source_fit_mode==Point_Source) {
		if (!use_analytic_bestfit_src) {
			if (n_sourcepts_fit==1) {
				if (vary_sourcepts_x[0]) {
					fit_parameter_names.push_back("xsrc");
					latex_parameter_names.push_back("x");
					latex_parameter_subscripts.push_back("src");
				}
				if (vary_sourcepts_y[0]) {
					fit_parameter_names.push_back("ysrc");
					latex_parameter_names.push_back("y");
					latex_parameter_subscripts.push_back("src");
				}
			} else {
				for (i=0; i < n_sourcepts_fit; i++) {
					stringstream srcpt_num_str;
					string srcpt_num_string;
					srcpt_num_str << i;
					srcpt_num_str >> srcpt_num_string;
					if (vary_sourcepts_x[i]) {
						fit_parameter_names.push_back("xsrc" + srcpt_num_string);
						latex_parameter_names.push_back("x");
						latex_parameter_subscripts.push_back("src,"+srcpt_num_string);
					}
					if (vary_sourcepts_y[i]) {
						fit_parameter_names.push_back("ysrc" + srcpt_num_string);
						latex_parameter_names.push_back("y");
						latex_parameter_subscripts.push_back("src,"+srcpt_num_string);
					}
				}
			}
		}
	}
	else if ((vary_regularization_parameter) and ((source_fit_mode==Pixellated_Source) or (source_fit_mode==Shapelet_Source)) and (regularization_method != None)) {
		fit_parameter_names.push_back("regparam");
		latex_parameter_names.push_back("\\lambda");
		latex_parameter_subscripts.push_back("");
	}
	if (vary_pixel_fraction) {
		fit_parameter_names.push_back("pixel_fraction");
		latex_parameter_names.push_back("f");
		latex_parameter_subscripts.push_back("pixel");
	}
	if (vary_srcgrid_xshift) {
		fit_parameter_names.push_back("srcgrid_xshift");
		latex_parameter_names.push_back("\\delta x");
		latex_parameter_subscripts.push_back("sg");
	}
	if (vary_srcgrid_yshift) {
		fit_parameter_names.push_back("srcgrid_yshift");
		latex_parameter_names.push_back("\\delta y");
		latex_parameter_subscripts.push_back("sg");
	}
	if (vary_srcgrid_size_scale) {
		fit_parameter_names.push_back("srcgrid_scale");
		latex_parameter_names.push_back("f");
		latex_parameter_subscripts.push_back("sg");
	}
	if (vary_magnification_threshold) {
		fit_parameter_names.push_back("mag_threshold");
		latex_parameter_names.push_back("m");
		latex_parameter_subscripts.push_back("split");
	}
	if (vary_hubble_parameter) {
		fit_parameter_names.push_back("h0");
		latex_parameter_names.push_back("H");
		latex_parameter_subscripts.push_back("0");
	}
	if (vary_omega_matter_parameter) {
		fit_parameter_names.push_back("omega_m");
		latex_parameter_names.push_back("\\Omega");
		latex_parameter_subscripts.push_back("M");
	}
	if (vary_syserr_pos_parameter) {
		fit_parameter_names.push_back("syserr_pos");
		latex_parameter_names.push_back("\\sigma");
		latex_parameter_subscripts.push_back("sys");
	}
	if (vary_wl_shear_factor_parameter) {
		fit_parameter_names.push_back("wl_shearfac");
		latex_parameter_names.push_back("m");
		latex_parameter_subscripts.push_back("WL");
	}

	if (fit_parameter_names.size() != n_fit_parameters) die("get_parameter_names() did not assign names to all the fit parameters (%i vs %i)",n_fit_parameters,fit_parameter_names.size());
	for (i=0; i < n_fit_parameters; i++) {
		if (latex_parameter_subscripts[i] != "") latex_parameter_names[i] += "_{" + latex_parameter_subscripts[i] + "}";
	}
}

bool QLens::lookup_parameter_value(const string pname, double& pval)
{
	bool found_param = false;
	setup_fit_parameters(false);
	int i;
	for (i=0; i < n_fit_parameters; i++) {
		if (transformed_parameter_names[i]==pname) {
			found_param = true;
			pval = fitparams[i];
		}
	}
	if (!found_param) {
		for (i=0; i < n_derived_params; i++) {
			if (dparam_list[i]->name==pname) {
				found_param = true;
				pval = dparam_list[i]->get_derived_param(this);
			}
		}
	}
	return found_param;
}

void QLens::create_parameter_value_string(string &pvals)
{
	pvals = "";
	setup_fit_parameters(false);
	int i;
	for (i=0; i < n_fit_parameters; i++) {
		stringstream pvalstr;
		string pvalstring;
		pvalstr << fitparams[i];
		pvalstr >> pvalstring;
		pvals += pvalstring;
		if ((n_derived_params > 0) or (i < n_fit_parameters-1)) pvals += " ";
	}
	double pval;
	for (i=0; i < n_derived_params; i++) {
		pval = dparam_list[i]->get_derived_param(this);
		stringstream pvalstr;
		string pvalstring;
		pvalstr << pval;
		pvalstr >> pvalstring;
		pvals += pvalstring;
		if (i < n_derived_params-1) pvals += " ";
	}
}

bool QLens::get_lens_parameter_numbers(const int lens_i, int& pi, int& pf)
{
	if (lens_i >= nlens) { pf=pi=0; return false; }
	get_n_fit_parameters(n_fit_parameters);
	vector<string> dummy, dummy2, dummy3;
	for (int i=0; i < lens_i; i++) {
		lens_list[i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	}
	pi = dummy.size();
	if (pi == n_fit_parameters) { pf=pi=0; return false; }
	lens_list[lens_i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	pf = dummy.size();
	if (pf==pi) return false;
	return true;
}

bool QLens::get_sb_parameter_numbers(const int sb_i, int& pi, int& pf)
{
	if (sb_i >= n_sb) { pf=pi=0; return false; }
	get_n_fit_parameters(n_fit_parameters);
	vector<string> dummy, dummy2, dummy3;
	for (int i=0; i < sb_i; i++) {
		sb_list[i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	}
	pi = dummy.size();
	if (pi == n_fit_parameters) { pf=pi=0; return false; }
	sb_list[sb_i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	pf = dummy.size();
	pi += lensmodel_fit_parameters; // since lens fit parameters come before the source params
	pf += lensmodel_fit_parameters; // since lens fit parameters come before the source params
	if (pf==pi) return false;
	return true;
}

void QLens::fit_set_optimizations()
{
	temp_auto_ccspline = auto_ccspline;
	temp_auto_store_cc_points = auto_store_cc_points;
	temp_include_time_delays = include_time_delays;

	// turn the following features off because they add pointless overhead (they will be restored to their
	// former settings after the search is done)
	auto_ccspline = false;
	auto_store_cc_points = false;
	include_time_delays = false; // calculating time delays from images found not necessary during fit, since the chisq_time_delays finds time delays separately

	fisher_inverse.erase(); // reset parameter covariance matrix in case it was used in a previous fit
}

void QLens::fit_restore_defaults()
{
	auto_ccspline = temp_auto_ccspline;
	auto_store_cc_points = temp_auto_store_cc_points;
	include_time_delays = temp_include_time_delays;
	clear_raw_chisq(); // in case chi-square is being used as a derived parameter
	Grid::set_lens(this); // annoying that the grids can only point to one lens object--it would be better for the pointer to be non-static (implement this later)
}

double QLens::chisq_single_evaluation(bool show_diagnostics, bool show_status)
{
	if (setup_fit_parameters(false)==false) return -1e30;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	if (!initialize_fitmodel(false)) {
		raw_chisq = 1e30;
		if ((mpi_id==0) and (show_status)) warn(warnings,"Warning: could not evaluate chi-square function");
		return -1e30;
	}
	//fitmodel->param_settings->print_penalty_limits();

	double (QLens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	display_chisq_status = true;
	fitmodel->chisq_it = 0;
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	if (show_diagnostics) chisq_diagnostic = true;
	bool default_display_status = display_chisq_status;
	if (!show_status) display_chisq_status = false;

	//fitmodel->print_lens_list(true);
	//fitmodel->param_settings->print_penalty_limits();
	double chisqval = 2 * (this->*loglikeptr)(fitparams.array());
	if (einstein_radius_prior) fitmodel->get_einstein_radius_prior(true); // just to show what the Re prior is returning
	if (!show_status) display_chisq_status = default_display_status;
	//if ((mpi_id==0) and (show_status)) {
		//if (display_chisq_status) cout << endl;
		//cout << "2*loglike: " << chisqval << endl;
	//}
	if ((chisqval >= 1e30) and (mpi_id==0)) warn(warnings,"Your parameter values are returning a large \"penalty\" chi-square--this likely means one or\nmore parameters have unphysical values or are out of the bounds specified by 'fit plimits'");
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if ((mpi_id==0) and (show_status)) cout << "Wall time for likelihood evaluation: " << wtime << endl;
	}
#endif
	display_chisq_status = false;
	if (show_diagnostics) chisq_diagnostic = false;

	double rawchisqval = raw_chisq;
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return rawchisqval;
}

void QLens::plot_chisq_2d(const int param1, const int param2, const int n1, const double i1, const double f1, const int n2, const double i2, const double f2)
{
	if (setup_fit_parameters(false)==false) return;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	if (!initialize_fitmodel(false)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	if (param1 >= n_fit_parameters) { warn("Parameter %i does not exist (%i parameters total)",param1,n_fit_parameters); return; }
	if (param2 >= n_fit_parameters) { warn("Parameter %i does not exist (%i parameters total)",param2,n_fit_parameters); return; }

	double (QLens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	double step1 = (f1-i1)/n1;
	double step2 = (f2-i2)/n2;
	int i,j;
	double p1,p2;
	ofstream p1out("chisq2d.x");
	ofstream l1out("like2d.x");
	for (i=0, p1=i1; i <= n1; i++, p1 += step1) {
		p1out << p1 << endl;
		l1out << p1 << endl;
	}
	p1out.close();
	l1out.close();
	ofstream p2out("chisq2d.y");
	ofstream l2out("like2d.y");
	for (i=0, p2=i2; i <= n2; i++, p2 += step2) {
		p2out << p2 << endl;
		l2out << p2 << endl;
	}
	p2out.close();
	l2out.close();

	double chisqmin=1e30;
	dmatrix chisqvals(n1,n2);
	ofstream chisqout("chisq2d.dat");
	double p1min, p2min;
	for (j=0, p2=i2+0.5*step2; j < n2; j++, p2 += step2) {
		for (i=0, p1=i1+0.5*step1; i < n1; i++, p1 += step1) {
			cout << "p1=" << p1 << " p2=" << p2 << endl;
			fitparams[param1] = p1;
			fitparams[param2] = p2;
			chisqvals[i][j] = 2.0 * (this->*loglikeptr)(fitparams.array());
			if (chisqvals[i][j] < chisqmin) {
				chisqmin = chisqvals[i][j];
				p1min = p1;
				p2min = p2;
			}
			chisqout << chisqvals[i][j] << " ";
		}
		chisqout << endl;
	}
	chisqout.close();
	if (mpi_id==0) cout << "min chisq=" << chisqmin << ", occurs at (" << p1min << "," << p2min << ")\n";

	ofstream likeout("like2d.dat");
	for (i=0; i < n1; i++) {
		for (j=0; j < n2; j++) {
			likeout << exp(-0.5*SQR(chisqvals[i][j]-chisqmin)) << " ";
		}
		likeout << endl;
	}
	likeout.close();

	fit_restore_defaults();
}

void QLens::plot_chisq_1d(const int param, const int n, const double ip, const double fp, string filename)
{
	if (setup_fit_parameters(false)==false) return;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	if (!initialize_fitmodel(false)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	if (param >= n_fit_parameters) { warn("Parameter %i does not exist (%i parameters total)",param,n_fit_parameters); return; }

	double (QLens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	double step = (fp-ip)/n;
	int i,j;
	double p;

	double chisqmin=1e30;
	dvector chisqvals(n);
	ofstream chisqout(filename.c_str());
	double pmin;
	for (i=0, p=ip; i <= n; i++, p += step) {
		fitparams[param] = p;
		chisqvals[i] = 2.0 * (this->*loglikeptr)(fitparams.array());
		if (chisqvals[i] < chisqmin) {
			chisqmin = chisqvals[i];
			pmin = p;
		}
		chisqout << p << " " << chisqvals[i] << endl;
	}
	chisqout.close();
	if (mpi_id==0) cout << "min chisq=" << chisqmin << ", occurs at " << pmin << endl;

	fit_restore_defaults();
}

double QLens::chi_square_fit_simplex()
{
	if (setup_fit_parameters(false)==false) return 0.0;
	fit_set_optimizations();
	if (!initialize_fitmodel(false)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return 1e30;
	}

	double (Simplex::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Simplex::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (Simplex::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	dvector stepsizes(param_settings->stepsizes,n_fit_parameters);
	if (mpi_id==0) {
		cout << "Initial stepsizes: ";
		for (int i=0; i < n_fit_parameters; i++) cout << stepsizes[i] << " ";
		cout << endl << endl;
	}

	initialize_simplex(fitparams.array(),n_fit_parameters,stepsizes.array(),chisq_tolerance);
	simplex_set_display_bfpont(simplex_show_bestfit);
	simplex_set_function(loglikeptr);
	simplex_set_fmin(simplex_minchisq/2);
	simplex_set_fmin_anneal(simplex_minchisq_anneal/2);
	//int iterations = 0;
	//downhill_simplex(iterations,max_iterations,0); // last argument is temperature for simulated annealing, but there is no cooling schedule with this function
	set_annealing_schedule_parameters(simplex_temp_initial,simplex_temp_final,simplex_cooling_factor,simplex_nmax_anneal,simplex_nmax);
	int n_iterations;

	double chisq_initial = (this->*loglikeptr)(fitparams.array());
	if ((chisq_initial >= 1e30) and (mpi_id==0)) warn(warnings,"Your initial parameter values are returning a large \"penalty\" chi-square--this likely means\none or more parameters have unphysical values or are out of the bounds specified by 'fit plimits'");

	display_chisq_status = true;

	fitmodel->chisq_it = 0;
	bool verbal = (mpi_id==0) ? true : false;
	//if (simplex_show_bestfit) cout << endl; // since we'll need an extra line to display best-fit parameters during annealing
	if (use_ansi_output_during_fit) use_ansi_characters = true;
	else use_ansi_characters = false;
	n_iterations = downhill_simplex_anneal(verbal);
	simplex_minval(fitparams.array(),chisq_bestfit);
	chisq_bestfit *= 2; // since the loglike function actually returns 0.5*chisq
	int chisq_evals = fitmodel->chisq_it;
	fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
	if (display_chisq_status) {
		(this->*loglikeptr)(fitparams.array());
		if (mpi_id==0) cout << endl << endl;
	}
	//use_ansi_characters = false;

	bool turned_on_chisqmag = false;
	if (n_repeats > 0) {
		if ((source_fit_mode==Point_Source) and (!use_magnification_in_chisq) and (use_magnification_in_chisq_during_repeats) and (!imgplane_chisq)) {
			turned_on_chisqmag = true;
			use_magnification_in_chisq = true;
			fitmodel->use_magnification_in_chisq = true;
			simplex_evaluate_bestfit_point(); // need to re-evaluate and record the chi-square at the best-fit point since we are changing the chi-square function
			cout << "Now using magnification in position chi-square function during repeats...\n";
		}
		set_annealing_schedule_parameters(0,simplex_temp_final,simplex_cooling_factor,simplex_nmax_anneal,simplex_nmax); // repeats have zero temperature (just minimization)
		for (int i=0; i < n_repeats; i++) {
			if (mpi_id==0) cout << "Repeating optimization (trial " << i+1 << ")                                                  \n\n\n" << flush;
			//use_ansi_characters = true;
			n_iterations = downhill_simplex_anneal(verbal);
			//use_ansi_characters = false;
			simplex_minval(fitparams.array(),chisq_bestfit);
			chisq_bestfit *= 2; // since the loglike function actually returns 0.5*chisq
			chisq_evals += fitmodel->chisq_it;
			fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
			if (display_chisq_status) {
				(this->*loglikeptr)(fitparams.array());
				if (mpi_id==0) cout << endl << endl;
			}
		}
	}
	use_ansi_characters = false;
	bestfitparams.input(fitparams);

	display_chisq_status = false;
	if (mpi_id==0) {
		if (simplex_exit_status==true) {
			if (simplex_temp_initial==0) cout << "Downhill simplex converged after " << n_iterations << " iterations\n\n";
			else cout << "Downhill simplex converged after " << n_iterations << " iterations at final temperature T=0\n\n";
		} else {
			cout << "Downhill simplex interrupted after " << n_iterations << " iterations\n\n";
		}
	}

	if (source_fit_mode==Pixellated_Source) {
		if (fitmodel->source_pixel_grid != NULL) {
			fitmodel->source_pixel_grid->plot_surface_brightness("src_calc");
			fitmodel->image_pixel_grid->plot_surface_brightness("img_calc");
		} else warn("source pixel grid was not created during fit");
	} else if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		fitmodel->image_pixel_grid->plot_surface_brightness("img_calc");
	}

	bool fisher_matrix_is_nonsingular;
	if (calculate_parameter_errors) {
		if (mpi_id==0) cout << "Calculating parameter errors... (press CTRL-C to skip)" << endl;
		fisher_matrix_is_nonsingular = calculate_fisher_matrix(fitparams,stepsizes);
		if (fisher_matrix_is_nonsingular) bestfit_fisher_inverse.input(fisher_inverse);
		else bestfit_fisher_inverse.erase(); // just in case it was defined before
		if (mpi_id==0) cout << endl;
	}
	if (mpi_id==0) {
		if (use_scientific_notation) cout << setiosflags(ios::scientific);
		else {
			cout << resetiosflags(ios::scientific);
			cout.unsetf(ios_base::floatfield);
		}
		cout << "Best-fit model: 2*loglike = " << chisq_bestfit << " (after " << chisq_evals << " evals)" << endl;
		//cout << "Number of iterations: " << iterations << endl;
		for (int i=0; i < nlens; i++) fitmodel->lens_list[i]->reset_angle_modulo_2pi();
		fitmodel->print_lens_list(false);

		if (source_fit_mode == Point_Source) {
			lensvector *bestfit_src = new lensvector[n_sourcepts_fit];
			double *bestfit_flux;
			bool found_bestfit_flux = true;
			if (include_flux_chisq) {
				bestfit_flux = new double[n_sourcepts_fit];
				fitmodel->output_model_source_flux(bestfit_flux);
				for (int i=0; i < n_sourcepts_fit; i++) if (bestfit_flux[i] == -1) found_bestfit_flux = false;
			};
			if (!found_bestfit_flux) warn("Not all best-fit source fluxes could be calculated. If there are no measured fluxes in your\ndata, turn 'chisqflux' off.");

			if (use_analytic_bestfit_src) {
				fitmodel->output_analytic_srcpos(bestfit_src);
			} else {
				for (int i=0; i < n_sourcepts_fit; i++) bestfit_src[i] = fitmodel->sourcepts_fit[i];
			}
			for (int i=0; i < n_sourcepts_fit; i++) {
				cout << "src" << i << "_x=" << bestfit_src[i][0] << " src" << i << "_y=" << bestfit_src[i][1];
				if ((include_flux_chisq) and (found_bestfit_flux)) {
					cout << " src" << i << "_flux=" << bestfit_flux[i];
				}
				cout << endl;
			}
			delete[] bestfit_src;
			if (include_flux_chisq) delete[] bestfit_flux;
		}

		if ((vary_regularization_parameter) and (source_fit_mode == Pixellated_Source) and (regularization_method != None)) {
			cout << "regularization parameter lambda=" << fitmodel->regularization_parameter << endl;
		}
		if (vary_pixel_fraction) cout << "pixel fraction = " << fitmodel->pixel_fraction << endl;
		if (vary_srcgrid_xshift) cout << "sourcegrid x-shift = " << fitmodel->srcgrid_xshift << endl;
		if (vary_srcgrid_yshift) cout << "sourcegrid y-shift = " << fitmodel->srcgrid_yshift << endl;
		if (vary_srcgrid_size_scale) cout << "sourcegrid scale = " << fitmodel->srcgrid_size_scale << endl;
		if (vary_magnification_threshold) cout << "magnification threshold = " << fitmodel->pixel_magnification_threshold << endl;
		if (vary_hubble_parameter) cout << "h0 = " << fitmodel->hubble << endl;
		if (vary_omega_matter_parameter) cout << "omega_m = " << fitmodel->omega_matter << endl;
		if (vary_syserr_pos_parameter) cout << "syserr_pos = " << fitmodel->syserr_pos << endl;
		if (vary_wl_shear_factor_parameter) cout << "wl_shearfac = " << fitmodel->wl_shear_factor << endl;

		cout << endl;
		if (calculate_parameter_errors) {
			if (fisher_matrix_is_nonsingular) {
				cout << "Marginalized 1-sigma errors from Fisher matrix:\n";
				for (int i=0; i < n_fit_parameters; i++) {
					cout << transformed_parameter_names[i] << ": " << fitparams[i] << " +/- " << sqrt(abs(fisher_inverse[i][i])) << endl;
				}
			} else {
				cout << "Error: Fisher matrix is singular, marginalized errors cannot be calculated\n";
				for (int i=0; i < n_fit_parameters; i++)
					cout << transformed_parameter_names[i] << ": " << fitparams[i] << endl;
			}
		} else {
			for (int i=0; i < n_fit_parameters; i++)
				cout << transformed_parameter_names[i] << ": " << fitparams[i] << endl;
		}
		//cout << "\nNOTE: To adopt the best-fit model parameters, enter the command 'fit use_bestfit'.\n";
		cout << endl;
		if (auto_save_bestfit) output_bestfit_model();
	}

	if (turned_on_chisqmag) use_magnification_in_chisq = false; // restore chisqmag to original setting
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return chisq_bestfit;
}

double QLens::chi_square_fit_powell()
{
	if (setup_fit_parameters(false)==false) return 0.0;
	fit_set_optimizations();
	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return 1e30;
	}

	double (Powell::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Powell::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (Powell::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	initialize_powell(loglikeptr,chisq_tolerance);

	dvector stepsizes(param_settings->stepsizes,n_fit_parameters);
	if (mpi_id==0) {
		cout << "Initial stepsizes: ";
		for (int i=0; i < n_fit_parameters; i++) cout << stepsizes[i] << " ";
		cout << endl << endl;
	}

	double chisq_initial = (this->*loglikeptr)(fitparams.array());
	if ((chisq_initial >= 1e30) and (mpi_id==0)) warn(warnings,"Your initial parameter values are returning a large \"penalty\" chi-square--this likely means\none or more parameters have unphysical values or are out of the bounds specified by 'fit plimits'");

	display_chisq_status = true;

	fitmodel->chisq_it = 0;
	if (use_ansi_output_during_fit) use_ansi_characters = true;
	else use_ansi_characters = false;
	powell_minimize(fitparams.array(),n_fit_parameters,stepsizes.array());
	use_ansi_characters = false;
	chisq_bestfit = 2*(this->*loglikeptr)(fitparams.array());
	int chisq_evals = fitmodel->chisq_it;
	fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
	if (display_chisq_status) {
		(this->*loglikeptr)(fitparams.array());
		if (mpi_id==0) cout << endl;
	}

	bool turned_on_chisqmag = false;
	if (n_repeats > 0) {
		if ((source_fit_mode==Point_Source) and (!use_magnification_in_chisq) and (use_magnification_in_chisq_during_repeats) and (!imgplane_chisq)) {
			turned_on_chisqmag = true;
			use_magnification_in_chisq = true;
			fitmodel->use_magnification_in_chisq = true;
			cout << "Now using magnification in position chi-square function during repeats...\n";
		}
		for (int i=0; i < n_repeats; i++) {
			if (mpi_id==0) cout << "Repeating optimization (trial " << i+1 << ")\n";
			use_ansi_characters = true;
			powell_minimize(fitparams.array(),n_fit_parameters,stepsizes.array());
			use_ansi_characters = false;
			chisq_bestfit = 2*(this->*loglikeptr)(fitparams.array());
			chisq_evals += fitmodel->chisq_it;
			fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
			if (display_chisq_status) {
				(this->*loglikeptr)(fitparams.array());
				if (mpi_id==0) cout << endl;
			}
		}
	}
	bestfitparams.input(fitparams);

	display_chisq_status = false;

	if (group_id==0) fitmodel->logfile << "Optimization finished: min chisq = " << chisq_bestfit << endl;

	if (source_fit_mode==Pixellated_Source) {
		if (mpi_id==0) fitmodel->source_pixel_grid->plot_surface_brightness("src_calc");
		if (mpi_id==0) fitmodel->image_pixel_grid->plot_surface_brightness("img_calc");
	} else if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		if (mpi_id==0) fitmodel->image_pixel_grid->plot_surface_brightness("img_calc");
	}

	bool fisher_matrix_is_nonsingular;
	if (calculate_parameter_errors) {
		if (mpi_id==0) cout << "Calculating parameter errors... (press CTRL-C to skip)" << endl;
		fisher_matrix_is_nonsingular = calculate_fisher_matrix(fitparams,stepsizes);
		if (fisher_matrix_is_nonsingular) bestfit_fisher_inverse.input(fisher_inverse);
		else bestfit_fisher_inverse.erase(); // just in case it was defined before
		cout << endl;
	}
	if (mpi_id==0) {
		// The following is the same code as in the simplex(...) function, which is stupid. You should put it in a separate function so that you don't have two copies of the same code. DO THIS LATER!!!
		if (use_scientific_notation) cout << setiosflags(ios::scientific);
		else {
			cout << resetiosflags(ios::scientific);
			cout.unsetf(ios_base::floatfield);
		}

		cout << "\nBest-fit model: 2*loglike = " << chisq_bestfit << " (after " << chisq_evals << " evals)" << endl;
		update_fitmodel(fitparams.array());
		for (int i=0; i < nlens; i++) {
			fitmodel->lens_list[i]->reset_angle_modulo_2pi();
		}
		fitmodel->print_lens_list(false);

		if (source_fit_mode == Point_Source) {
			lensvector *bestfit_src = new lensvector[n_sourcepts_fit];
			double *bestfit_flux;
			if (include_flux_chisq) {
				bestfit_flux = new double[n_sourcepts_fit];
				fitmodel->output_model_source_flux(bestfit_flux);
			};
			if (use_analytic_bestfit_src) {
				fitmodel->output_analytic_srcpos(bestfit_src);
			} else {
				for (int i=0; i < n_sourcepts_fit; i++) bestfit_src[i] = fitmodel->sourcepts_fit[i];
			}
			for (int i=0; i < n_sourcepts_fit; i++) {
				cout << "src" << i << "_x=" << bestfit_src[i][0] << " src" << i << "_y=" << bestfit_src[i][1];
				if (include_flux_chisq) cout << " src" << i << "_flux=" << bestfit_flux[i];
				cout << endl;
			}
			delete[] bestfit_src;
			if (include_flux_chisq) delete[] bestfit_flux;
		}

		if ((vary_regularization_parameter) and (source_fit_mode == Pixellated_Source) and (regularization_method != None)) {
			cout << "regularization parameter lambda=" << fitmodel->regularization_parameter << endl;
		}
		if (vary_pixel_fraction) cout << "pixel fraction = " << fitmodel->pixel_fraction << endl;
		if (vary_srcgrid_xshift) cout << "source grid x-shift = " << fitmodel->srcgrid_xshift << endl;
		if (vary_srcgrid_yshift) cout << "source grid y-shift = " << fitmodel->srcgrid_yshift << endl;
		if (vary_srcgrid_size_scale) cout << "source grid scale = " << fitmodel->srcgrid_size_scale << endl;
		if (vary_magnification_threshold) cout << "magnification threshold = " << fitmodel->pixel_magnification_threshold << endl;
		if (vary_hubble_parameter) cout << "h0 = " << fitmodel->hubble << endl;
		if (vary_omega_matter_parameter) cout << "omega_m = " << fitmodel->omega_matter << endl;
		if (vary_syserr_pos_parameter) cout << "syserr_pos = " << fitmodel->syserr_pos << endl;
		if (vary_wl_shear_factor_parameter) cout << "wl_shearfac = " << fitmodel->wl_shear_factor << endl;
		cout << endl;
		if (calculate_parameter_errors) {
			if (fisher_matrix_is_nonsingular) {
				cout << "Marginalized 1-sigma errors from Fisher matrix:\n";
				for (int i=0; i < n_fit_parameters; i++) {
					cout << transformed_parameter_names[i] << ": " << fitparams[i] << " +/- " << sqrt(abs(fisher_inverse[i][i])) << endl;
				}
			} else {
				cout << "Error: Fisher matrix is singular, marginalized errors cannot be calculated\n";
				for (int i=0; i < n_fit_parameters; i++)
					cout << transformed_parameter_names[i] << ": " << fitparams[i] << endl;
			}
		} else {
			for (int i=0; i < n_fit_parameters; i++)
				cout << transformed_parameter_names[i] << ": " << fitparams[i] << endl;
		}
		cout << endl;
		if (auto_save_bestfit) output_bestfit_model();
	}

	if (turned_on_chisqmag) use_magnification_in_chisq = false; // restore chisqmag to original setting
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return chisq_bestfit;
}

void QLens::nested_sampling()
{
	if (setup_fit_parameters(true)==false) return;
	fit_set_optimizations();
	if ((mpi_id==0) and (fit_output_dir != ".")) {
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		if (system(rmstring.c_str()) != 0) warn("could not delete old output directory for nested sampling results"); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
		// I should probably give the nested sampling output a unique extension like ".nest" or something, so that mkdist can't ever confuse it with twalk output in the same dir
		// Do this later...
		create_output_directory();
	}

	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	InputPoint(fitparams.array(),upper_limits.array(),lower_limits.array(),upper_limits_initial.array(),lower_limits_initial.array(),n_fit_parameters);
	SetNDerivedParams(n_derived_params);

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	if (mpi_id==0) {
		// This code gets repeated in a few spots and should really be put in a separate function...DO THIS LATER!
		int i;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fit_parameters << " " << n_derived_params << endl;
		pnumfile.close();
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_fit_parameters; i++) pnamefile << transformed_parameter_names[i] << endl;
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list[i]->name << endl;
		pnamefile.close();
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_fit_parameters; i++) lpnamefile << transformed_parameter_names[i] << "\t" << transformed_latex_parameter_names[i] << endl;
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list[i]->name << "\t" << dparam_list[i]->latex_name << endl;
		lpnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < n_fit_parameters; i++)
		{
			prangefile << lower_limits[i] << " " << upper_limits[i] << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		if (param_markers != "") {
			string marker_str = fit_output_dir + "/" + fit_output_filename + ".markers";
			ofstream markerfile(marker_str.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

	double *param_errors = new double[n_fit_parameters];
#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	string filename = fit_output_dir + "/" + fit_output_filename;

	display_chisq_status = false; // just in case it was turned on
	double lnZ;

	use_ansi_characters = true;
	MonoSample(filename.c_str(),n_livepts,lnZ,fitparams.array(),param_errors,mcmc_logfile,NULL,chain_info,data_info);
	use_ansi_characters = false;
	bestfitparams.input(fitparams);
	chisq_bestfit = 2*(this->*LogLikePtr)(fitparams.array());

	//if (display_chisq_status) {
		//for (int i=0; i < n_sourcepts_fit; i++) cout << endl; // to get past the status signs for image position chi-square
		//cout << endl;
		//display_chisq_status = false;
	//}

#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for nested sampling: " << wt << endl;
	}
#endif

	if (mpi_id==0) {
		cout << endl;
		if (source_fit_mode == Point_Source) {
			lensvector *bestfit_src = new lensvector[n_sourcepts_fit];
			double *bestfit_flux;
			if (include_flux_chisq) {
				bestfit_flux = new double[n_sourcepts_fit];
				fitmodel->output_model_source_flux(bestfit_flux);
			};
			if (use_analytic_bestfit_src) {
				fitmodel->output_analytic_srcpos(bestfit_src);
			} else {
				for (int i=0; i < n_sourcepts_fit; i++) bestfit_src[i] = fitmodel->sourcepts_fit[i];
			}
			for (int i=0; i < n_sourcepts_fit; i++) {
				cout << "src" << i << "_x=" << bestfit_src[i][0] << " src" << i << "_y=" << bestfit_src[i][1];
				if (include_flux_chisq) cout << " src" << i << "_flux=" << bestfit_flux[i];
				cout << endl;
			}
			delete[] bestfit_src;
			if (include_flux_chisq) delete[] bestfit_flux;
		}

		cout << "\nBest-fit parameters and error estimates (from dispersions of chain output points):\n";
		for (int i=0; i < n_fit_parameters; i++) {
			cout << transformed_parameter_names[i] << ": " << fitparams[i] << " +/- " << param_errors[i] << endl;
		}
		cout << endl;
		output_bestfit_model();
	}
	delete[] param_errors;

	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
}

void QLens::multinest(const bool resume_previous, const bool skip_run)
{
#ifdef USE_MULTINEST
	if (setup_fit_parameters(true)==false) return;
	fit_set_optimizations();
	if ((mpi_id==0) and (!resume_previous) and (!skip_run) and (fit_output_dir != ".")) {
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		if (system(rmstring.c_str()) != 0) warn("could not delete old output directory for nested sampling results"); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
		// I should probably give the nested sampling output a unique extension like ".nest" or something, so that mkdist can't ever confuse it with twalk output in the same dir
		// Do this later...
		create_output_directory();
	}

	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	if (mpi_id==0) {
		// This code gets repeated in a few spots and should really be put in a separate function...DO THIS LATER!
		int i;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fit_parameters << " " << n_derived_params << endl;
		pnumfile.close();
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_fit_parameters; i++) pnamefile << transformed_parameter_names[i] << endl;
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list[i]->name << endl;
		pnamefile.close();
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_fit_parameters; i++) lpnamefile << transformed_parameter_names[i] << "\t" << transformed_latex_parameter_names[i] << endl;
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list[i]->name << "\t" << dparam_list[i]->latex_name << endl;
		lpnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < n_fit_parameters; i++)
		{
			prangefile << lower_limits[i] << " " << upper_limits[i] << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		if (param_markers != "") {
			string marker_str = fit_output_dir + "/" + fit_output_filename + ".markers";
			ofstream markerfile(marker_str.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	display_chisq_status = false; // just in case it was turned on
	string filename = fit_output_dir + "/" + fit_output_filename;

	 mcsampler_set_lensptr(this);

	int IS, mmodal, ceff, nPar, nClsPar, nlive, updInt, maxModes, seed, fb, resume, outfile, initMPI, maxiter;
	double efr, tol, Ztol, logZero;

	IS = 0;					// do Nested Importance Sampling (bad idea)
	mmodal = 0;					// do mode separation?
	ceff = 0;					// run in constant efficiency mode? (usually a VERY bad idea)
	efr = 0.8;				// set the required efficiency
	nlive = n_livepts;
	tol = 0.5;				// tol, defines the stopping criteria
	nPar = n_fit_parameters+n_derived_params;					// total no. of parameters including free & derived parameters
	nClsPar = n_fit_parameters;				// no. of parameters to do mode separation on
	updInt = 1000;				// after how many iterations feedback is required & the output files should be updated
							// note: posterior files are updated & dumper routine is called after every updInt*10 iterations
	Ztol = -1e90;				// all the modes with logZ < Ztol are ignored
	maxModes = 100;				// expected max no. of modes (used only for memory allocation)
	seed = 11+group_num;					// random no. generator seed, if < 0 then take the seed from system clock

	if (mpi_id==0) fb = 1;					// need feedback on standard output?
	else fb = 0;

	if (resume_previous) resume = 1;					// resume from a previous job?
	else resume = 0;

	outfile = 1;				// write output files?
	initMPI = 0;				// initialize MPI routines?, relevant only if compiling with MPI
							// set it to F if you want your main program to handle MPI initialization
	logZero = -1e90;			// points with loglike < logZero will be ignored by MultiNest
	maxiter = 0;				// max no. of iterations, a non-positive value means infinity. MultiNest will terminate if either it 
							// has done max no. of iterations or convergence criterion (defined through tol) has been satisfied
	void *context = 0;				// not required by MultiNest, any additional information user wants to pass
	int pWrap[n_fit_parameters];				// which parameters to have periodic boundary conditions?
	for (int i = 0; i < n_fit_parameters; i++) pWrap[i] = 0;
	//MPI_Fint fortran_comm = MPI_Comm_c2f((*group_comm));

	use_ansi_characters = true;

	if (!skip_run) {
#ifdef MULTINEST_MOD
		// This code uses a modified version of MultiNest that allows for the likelihood to be parallelized over a subset of MPI processes
		MPI_Group world_group;
		MPI_Comm world_comm;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);
		MPI_Comm_create(MPI_COMM_WORLD, world_group, &world_comm);

		MPI_Group leader_group;
		MPI_Comm leaders_comm;
		MPI_Group_incl(world_group, mpi_ngroups, group_leader, &leader_group);
		MPI_Comm_create(world_comm, leader_group, &leaders_comm);

		MPI_Fint fortran_comm = MPI_Comm_c2f(leaders_comm);

		int follower_rank = (group_id==0) ? -1 : group_num;

		nested::run(fortran_comm, follower_rank, mpi_ngroups, IS, mmodal, ceff, nlive, tol, efr, n_fit_parameters, nPar, nClsPar, maxModes, updInt, Ztol, filename.c_str(), seed, pWrap, fb, resume, outfile, initMPI, logZero, maxiter, multinest_loglikelihood, dumper_multinest, context);
#else
		nested::run(IS, mmodal, ceff, nlive, tol, efr, n_fit_parameters, nPar, nClsPar, maxModes, updInt, Ztol, filename.c_str(), seed, pWrap, fb, resume, outfile, initMPI, logZero, maxiter, multinest_loglikelihood, dumper_multinest, context);
#endif
	}

	bestfitparams.input(n_fit_parameters);

	use_ansi_characters = false;

	//if (display_chisq_status) {
		//for (int i=0; i < n_sourcepts_fit; i++) cout << endl; // to get past the status signs for image position chi-square
		//cout << endl;
		//display_chisq_status = false;
	//}

#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for nested sampling: " << wt << endl;
	}
#endif

	// Now convert the MultiNest output to a form that mkdist can read
	double lnZ = -1e30;
	double *xparams;
	double *params;
	double *covs;
	double *avgs;
	double minchisq = 1e30;
	int cont = 1;
	bool using_livepts_file = false;
	if (mpi_id==0) {
		cout << endl;
		string stats_filename = filename + "stats.dat";
		ifstream stats_in(stats_filename.c_str());
		if (!(stats_in.is_open())) warn("MultiNest output file %sstats.dat could not be found; evidence undetermined",filename.c_str());
		string dum;
		for (int i=0; i < 5; i++) {
			stats_in >> dum;
		}
		//double area=1.0;
		stats_in >> lnZ;
		stats_in.close();
		//for (int i=0; i < n_fit_parameters; i++) area *= (upper_limits[i]-lower_limits[i]);
		//lnZ += log(area);

		const int n_characters = 16384;
		char line[n_characters];

		string mnin_filename = filename + ".txt";
		ifstream mnin(mnin_filename.c_str());
		if (!(mnin.is_open())) {
			if (!(mnin.is_open())) {
				warn("MultiNest output file %s could not be found; will look for live points log file",mnin_filename.c_str());
			}
			mnin_filename = filename + "live.points";
			mnin.open(mnin_filename.c_str());
			using_livepts_file = true;
		}
		if (!(mnin.is_open())) {
			warn("MultiNest output file %s could not be found; chain cannot be processed",mnin_filename.c_str());
			cont = 0;
		} else {
			ofstream mnout(filename.c_str());
			if (data_info != "") mnout << "# DATA_INFO: " << data_info << endl;
			if (chain_info != "") mnout << "# CHAIN_INFO: " << chain_info << endl;
			mnout << "# Sampler: MultiNest, n_livepts = " << n_livepts << endl;
			mnout << "# lnZ = " << lnZ << endl;
			if (calculate_bayes_factor) {
				if (reference_lnZ==-1e30) reference_lnZ = lnZ; // first model being fit, so Bayes factor doesn't get calculated yet
				else {
					double log_bayes_factor = lnZ - reference_lnZ;
					mnout << "# Bayes factor: ln(Z/Z_ref) = " << log_bayes_factor << " Z/Z_ref = " << exp(log_bayes_factor) << " (lnZ_ref=" << reference_lnZ << ")" << endl;
					reference_lnZ = lnZ;
				}
			}

			double weight, chi2;
			int n_tot_params = n_fit_parameters + n_derived_params;
			xparams = new double[n_tot_params];
			params = new double[n_tot_params];
			covs = new double[n_tot_params];
			avgs = new double[n_tot_params];
			int i;
			double weighttot = 0;
			for (int i=0; i < n_tot_params; i++) {
				covs[i] = 0;
				avgs[i] = 0;
			}
			while ((mnin.getline(line,n_characters)) and (!mnin.eof())) {
				istringstream instream(line);
				if (!using_livepts_file) {
					instream >> weight;
					instream >> chi2;
				} else {
					weight = 0.0;
				}
				for (i=0; i < n_tot_params; i++) instream >> xparams[i];
				transform_cube(params,xparams);
				mnout << weight << "   ";
				for (i=0; i < n_fit_parameters; i++) mnout << params[i] << "   ";
				for (i=0; i < n_derived_params; i++) mnout << xparams[n_fit_parameters+i] << "   ";
				if (using_livepts_file) {
					double negloglike;
					instream >> negloglike;
					chi2 = -2*negloglike;
				}
				mnout << chi2 << endl;
				if (chi2 < minchisq) {
					minchisq = chi2;
					for (i=0; i < n_fit_parameters; i++) bestfitparams[i] = params[i];
				}
				for (i=0; i < n_tot_params; i++) {
					avgs[i] += weight*params[i];
					covs[i] += weight*params[i]*params[i];
				}
				weighttot += weight;
			}
			mnin.close();
			for (i=0; i < n_tot_params; i++) {
				if (weighttot==0.0) {
					avgs[i] = 0;
					covs[i] = 0;
				} else {
					avgs[i] /= weighttot;
					covs[i] = covs[i]/weighttot - avgs[i]*avgs[i];
				}
			}
		}
	}

#ifdef USE_MPI
		MPI_Bcast(&cont,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
	if (cont == 0) return;
#ifdef USE_MPI
		MPI_Bcast(bestfitparams.array(),n_fit_parameters,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	if (group_num==0) {
		chisq_bestfit = 2*(this->*LogLikePtr)(bestfitparams.array());
	}

	if (mpi_id==0) {
		if (source_fit_mode == Point_Source) {
			lensvector *bestfit_src = new lensvector[n_sourcepts_fit];
			double *bestfit_flux;
			if (include_flux_chisq) {
				bestfit_flux = new double[n_sourcepts_fit];
				fitmodel->output_model_source_flux(bestfit_flux);
			};
			if (use_analytic_bestfit_src) {
				fitmodel->output_analytic_srcpos(bestfit_src);
			} else {
				for (int i=0; i < n_sourcepts_fit; i++) bestfit_src[i] = fitmodel->sourcepts_fit[i];
			}
			for (int i=0; i < n_sourcepts_fit; i++) {
				cout << "src" << i << "_x=" << bestfit_src[i][0] << " src" << i << "_y=" << bestfit_src[i][1];
				if (include_flux_chisq) cout << " src" << i << "_flux=" << bestfit_flux[i];
				cout << endl;
			}
			delete[] bestfit_src;
			if (include_flux_chisq) delete[] bestfit_flux;
		}

		cout << endl << "Log-evidence: ln(Z) = " << lnZ << endl;
		cout << "\nBest-fit parameters and error estimates (from dispersions of chain output points):    (chisq=" << minchisq << ")\n";
		if (using_livepts_file) {
			for (int i=0; i < n_fit_parameters; i++) {
				cout << transformed_parameter_names[i] << ": " << bestfitparams[i] << endl;
			}
		} else {
			for (int i=0; i < n_fit_parameters; i++) {
				cout << transformed_parameter_names[i] << ": " << bestfitparams[i] << " +/- " << sqrt(covs[i]) << endl;
			}
		}
		cout << endl;
		output_bestfit_model();
		delete[] xparams;
		delete[] params;
		delete[] avgs;
		delete[] covs;
	}
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
#endif
}

void QLens::polychord(const bool resume_previous, const bool skip_run)
{
#ifdef USE_POLYCHORD
	if (setup_fit_parameters(true)==false) return;
	fit_set_optimizations();
	if ((mpi_id==0) and (!resume_previous) and (!skip_run) and (fit_output_dir != ".")) {
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		if (system(rmstring.c_str()) != 0) warn("could not delete old output directory for nested sampling results"); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
		// I should probably give the nested sampling output a unique extension like ".nest" or something, so that mkdist can't ever confuse it with twalk output in the same dir
		// Do this later...
		create_output_directory();
		//if (mpi_id==0) {
			//string cluster_dir = fit_output_dir + "/clusters";
			//struct stat sb;
			//stat(cluster_dir.c_str(),&sb);
			//if (S_ISDIR(sb.st_mode)==false)
				//mkdir(cluster_dir.c_str(),S_IRWXU | S_IRWXG);
		//}
	}

	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	if (mpi_id==0) {
		// This code gets repeated in a few spots and should really be put in a separate function...DO THIS LATER!
		int i;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fit_parameters << " " << n_derived_params << endl;
		pnumfile.close();
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_fit_parameters; i++) pnamefile << transformed_parameter_names[i] << endl;
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list[i]->name << endl;
		pnamefile.close();
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_fit_parameters; i++) lpnamefile << transformed_parameter_names[i] << "\t" << transformed_latex_parameter_names[i] << endl;
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list[i]->name << "\t" << dparam_list[i]->latex_name << endl;
		lpnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < n_fit_parameters; i++)
		{
			prangefile << lower_limits[i] << " " << upper_limits[i] << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		if (param_markers != "") {
			string marker_str = fit_output_dir + "/" + fit_output_filename + ".markers";
			ofstream markerfile(marker_str.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	display_chisq_status = false; // just in case it was turned on

	use_ansi_characters = true;

	mcsampler_set_lensptr(this);
	Settings settings(n_fit_parameters,n_derived_params);

	settings.nlive         = n_livepts;
	settings.num_repeats   = n_fit_parameters*polychord_nrepeats;
	settings.do_clustering = false;

	settings.precision_criterion = 1e-3;
	settings.logzero = -1e30;

	settings.base_dir      = fit_output_dir.c_str();
	settings.file_root     = fit_output_filename.c_str();

	settings.write_resume  = true;
	settings.read_resume   = resume_previous;
	settings.write_live    = true;
	settings.write_dead    = true;
	settings.write_stats   = true;

	settings.equals        = false;
	settings.posteriors    = true;
	settings.cluster_posteriors = false;

	settings.feedback      = 3;
	settings.compression_factor  = 0.36787944117144233;

	settings.boost_posterior= 1.0;

	if (!skip_run) {
		run_polychord(polychord_loglikelihood,polychord_prior,polychord_dumper,settings);
	}

	use_ansi_characters = false;

	//if (display_chisq_status) {
		//for (int i=0; i < n_sourcepts_fit; i++) cout << endl; // to get past the status signs for image position chi-square
		//cout << endl;
		//display_chisq_status = false;
	//}

#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for nested sampling: " << wt << endl;
	}
#endif

	bestfitparams.input(n_fit_parameters);
	// Now convert the PolyChord output to a form that mkdist can read
	double *params, *covs, *avgs;
	double lnZ;
	if (mpi_id==0) {
		const int n_characters = 16384;
		char line[n_characters];

		string filename = fit_output_dir + "/" + fit_output_filename;
		string stats_filename = filename + ".stats";
		ifstream stats_in(stats_filename.c_str());
		int i;
		for (i=0; i < 8; i++) stats_in.getline(line,n_characters); // skip past beginning lines
		string dum;
		for (i=0; i < 2; i++) {
			stats_in >> dum;
		}
		//double area=1.0;
		stats_in >> lnZ;
		stats_in.close();
		//for (i=0; i < n_fit_parameters; i++) area *= (upper_limits[i]-lower_limits[i]);
		//lnZ += log(area);

		string polyin_filename = filename + ".txt";
		ifstream polyin(polyin_filename.c_str());
		ofstream polyout(filename.c_str());
		if (data_info != "") polyout << "# DATA_INFO: " << data_info << endl;
		if (chain_info != "") polyout << "# CHAIN_INFO: " << chain_info << endl;
		polyout << "# Sampler: PolyChord, n_livepts = " << n_livepts << endl;
		polyout << "# lnZ = " << lnZ << endl;
		if (calculate_bayes_factor) {
			if (reference_lnZ==-1e30) reference_lnZ = lnZ; // first model being fit, so Bayes factor doesn't get calculated yet
			else {
				double log_bayes_factor = lnZ - reference_lnZ;
				polyout << "# Bayes factor: ln(Z/Z_ref) = " << log_bayes_factor << " Z/Z_ref = " << exp(log_bayes_factor) << " (lnZ_ref=" << reference_lnZ << ")" << endl;
				reference_lnZ = lnZ;
			}
		}

		double weight, chi2;
		double minchisq = 1e30;
		int n_tot_params = n_fit_parameters + n_derived_params;
		params = new double[n_tot_params];
		covs = new double[n_tot_params];
		avgs = new double[n_tot_params];
		double weighttot = 0;
		for (int i=0; i < n_tot_params; i++) {
			covs[i] = 0;
			avgs[i] = 0;
		}
		istringstream linestream;
		int ncols = n_tot_params + 2;
		stringstream *colstr = new stringstream[ncols];
		string *colstring = new string[ncols];
		while ((polyin.getline(line,n_characters)) and (!polyin.eof())) {
			linestream.clear();
			linestream.str(line);
			for (i=0; i < ncols; i++) linestream >> colstring[i];
			for (i=0; i < ncols; i++) {
				colstr[i].clear();
				colstr[i].str(colstring[i]);
			}
			colstr[0] >> weight;
			colstr[1] >> chi2;
			for (i=0; i < n_tot_params; i++) colstr[i+2] >> params[i];
			polyout << weight << "   ";
			for (i=0; i < n_tot_params; i++) polyout << params[i] << "   ";
			polyout << chi2 << endl;
			if (chi2 < minchisq) {
				minchisq = chi2;
				for (i=0; i < n_fit_parameters; i++) bestfitparams[i] = params[i];
			}
			for (i=0; i < n_tot_params; i++) {
				avgs[i] += weight*params[i];
				covs[i] += weight*params[i]*params[i];
			}
			weighttot += weight;
		}
		polyin.close();
		for (i=0; i < n_tot_params; i++) {
			avgs[i] /= weighttot;
			covs[i] = covs[i]/weighttot - avgs[i]*avgs[i];
		}
		delete[] colstr;
		delete[] colstring;
	}

#ifdef USE_MPI
	MPI_Bcast(bestfitparams.array(),n_fit_parameters,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	if (group_num==0) {
		chisq_bestfit = 2*(this->*LogLikePtr)(bestfitparams.array());
	}

	if (mpi_id==0) {
		cout << endl;
		if (source_fit_mode == Point_Source) {
			lensvector *bestfit_src = new lensvector[n_sourcepts_fit];
			double *bestfit_flux;
			if (include_flux_chisq) {
				bestfit_flux = new double[n_sourcepts_fit];
				fitmodel->output_model_source_flux(bestfit_flux);
			};
			if (use_analytic_bestfit_src) {
				fitmodel->output_analytic_srcpos(bestfit_src);
			} else {
				for (int i=0; i < n_sourcepts_fit; i++) bestfit_src[i] = fitmodel->sourcepts_fit[i];
			}
			for (int i=0; i < n_sourcepts_fit; i++) {
				cout << "src" << i << "_x=" << bestfit_src[i][0] << " src" << i << "_y=" << bestfit_src[i][1];
				if (include_flux_chisq) cout << " src" << i << "_flux=" << bestfit_flux[i];
				cout << endl;
			}
			delete[] bestfit_src;
			if (include_flux_chisq) delete[] bestfit_flux;
		}

		cout << endl << "Log-evidence: ln(Z) = " << lnZ << endl;
		cout << "\nBest-fit parameters and error estimates (from dispersions of chain output points):\n";
		for (int i=0; i < n_fit_parameters; i++) {
			cout << transformed_parameter_names[i] << ": " << bestfitparams[i] << " +/- " << sqrt(covs[i]) << endl;
		}
		cout << endl;
		output_bestfit_model();
		delete[] params;
		delete[] avgs;
		delete[] covs;
	}

	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
#endif
}

bool QLens::add_dparams_to_chain()
{
	// Should have an option to specify the extension to the filename for the new chain (which defaults to '.new' if nothing is specified). ADD THIS IN!!!!!!!!!!!!!
	// Should check whether any new derived parameters have the same name as one of the old derived parameters--this can happen if one accidently adds the
	// same derived parameters they had before (in addition to some new ones). Have it print an error if this is the case. ADD THIS FEATURE!!!!!!
	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	int i, nparams, n_dparams_old;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams_old;
	if (nparams != n_fit_parameters) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	pnumfile.close();

	if (setup_fit_parameters(true)==false) return false;
	fit_set_optimizations();
	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return false;
	}

	static const int n_characters = 5000;
	char dataline[n_characters];
	if (mpi_id==0) {
		int i;
		int n_dparams_tot = n_dparams_old + n_derived_params;
		int n_totparams_old = n_fit_parameters + n_dparams_old;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".new.nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fit_parameters << " " << n_dparams_tot << endl;
		pnumfile.close();

		string pnamefile_old_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".new.paramnames";
		ifstream pnamefile_old(pnamefile_old_str.c_str());
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_totparams_old; i++) {
			pnamefile_old.getline(dataline,n_characters);
			pnamefile << dataline << endl;
		}
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list[i]->name << endl;
		pnamefile.close();
		pnamefile_old.close();

		string lpnamefile_old_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".new.latex_paramnames";
		ifstream lpnamefile_old(lpnamefile_old_str.c_str());
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_totparams_old; i++) {
			lpnamefile_old.getline(dataline,n_characters);
			lpnamefile << dataline << endl;
		}
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list[i]->name << "\t" << dparam_list[i]->latex_name << endl;
		lpnamefile.close();
		lpnamefile_old.close();

		string prangefile_old_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		string prangefile_str = fit_output_dir + "/" + fit_output_filename + ".new.ranges";
		ifstream prangefile_old(prangefile_old_str.c_str());
		ofstream prangefile(prangefile_str.c_str());
		for (i=0; i < n_totparams_old; i++) {
			prangefile_old.getline(dataline,n_characters);
			prangefile << dataline << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		prangefile_old.close();
	}

	double *params = new double[n_fit_parameters];
	double *dparams_old = new double[n_dparams_old];
	double weight, chisq;
	string chain_old_str = fit_output_dir + "/" + fit_output_filename;
	string chain_str = fit_output_dir + "/" + fit_output_filename + ".new";
	ifstream chain_file_old0(chain_old_str.c_str());

	int j,line,nlines=0;
	while (!chain_file_old0.eof()) {
		chain_file_old0.getline(dataline,n_characters);
		if (dataline[0]=='#') continue;
		nlines++;
	}
	double **dparams_new = new double*[nlines];
	for (i=0; i < nlines; i++) dparams_new[i] = new double[n_derived_params];
	char **chain_lines = new char*[nlines];
	for (i=0; i < nlines; i++) chain_lines[i] = new char[5000];
	chain_file_old0.close();

	chain_file_old0.open(chain_old_str.c_str());
	for (line=0; line < nlines; line++) {
		chain_file_old0.getline(chain_lines[line],n_characters);
		if (chain_lines[line][0]=='#') { line--; continue; }
	}

	int nlines_chunk = nlines/20;
	if (mpi_id==0) cout << "Calculating derived parameters: [\033[20C]" << endl << endl << flush;
	int prev_icount, icount = 0;
	for (line=group_num; line < nlines; line += mpi_ngroups) {
		istringstream datastream(chain_lines[line]);
		datastream >> weight;
		for (i=0; i < n_fit_parameters; i++) {
			datastream >> params[i];
		}
		fitmodel_calculate_derived_params(params, dparams_new[line]);

		prev_icount = icount;
		icount = line/nlines_chunk;
		if ((mpi_id==0) and (prev_icount != icount)) {
			cout << "\033[2ACalculating derived parameters: [" << flush;
			for (j=0; j < icount; j++) cout << "=" << flush;
			cout << "\033[1B" << endl << flush;
		}
	}
	if (mpi_id==0) {
		cout << "\033[2ACalculating derived parameters: [" << flush;
		for (j=0; j < 20; j++) cout << "=" << flush;
		cout << "\033[1B" << endl << flush;
	}
	if (mpi_id==0) cout << endl;
	//cout << "icount=" << icount << " prev=" << prev_icount << "line=" << line << " chunk=" << nlines_chunk << endl;

#ifdef USE_MPI
	int id;
	for (int groupnum=0; groupnum < mpi_ngroups; groupnum++) {
		for (i=groupnum; i < nlines; i += mpi_ngroups) {
			id = group_leader[groupnum];
			MPI_Bcast(dparams_new[i],n_derived_params,MPI_DOUBLE,id,MPI_COMM_WORLD);
		}
	}
#endif

	if (mpi_id==0) {
		ofstream chain_file(chain_str.c_str());
		for (line=0; line < nlines; line++) {
			istringstream datastream(chain_lines[line]);
			datastream >> weight;
			chain_file << weight << "   ";
			for (i=0; i < n_fit_parameters; i++) {
				datastream >> params[i];
				chain_file << params[i] << "   ";
			}
			for (i=0; i < n_dparams_old; i++) {
				datastream >> dparams_old[i];
				chain_file << dparams_old[i] << "   ";
			}
			datastream >> chisq;
			for (i=0; i < n_derived_params; i++) chain_file << dparams_new[line][i] << "   ";
			chain_file << chisq << endl;
		}
		chain_file.close();
	}

	delete[] params;
	delete[] dparams_old;
	for (i=0; i < nlines; i++) {
		delete[] chain_lines[i];
		delete[] dparams_new[i];
	}
	delete[] dparams_new;
	delete[] chain_lines;
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return true;
}

bool QLens::adopt_point_from_chain(const unsigned long line_num)
{

	int i, nparams, n_dparams_old;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams_old;
	if (nparams != n_fit_parameters) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	pnumfile.close();

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[n_fit_parameters];

	string chain_str = fit_output_dir + "/" + fit_output_filename;
	ifstream chain_file(chain_str.c_str());

	unsigned long nlines=0;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		//if (dataline[0]=='#') cout << string(dataline) << endl;
		nlines++;
	}
	if (nlines < line_num) {
		warn("number of points in chain (%i) is less than the point number requested (%i)",nlines,line_num); return false;
	}

	chain_file.close();
	chain_file.open(chain_str.c_str());
	for (i=1; i <= line_num; i++) {
		chain_file.getline(dataline,n_characters);
	}
	if (dataline[0]=='#') { warn("line from chain file is a comment line"); return false; }
	istringstream datastream(dataline);
	double weight;
	datastream >> weight;
	for (i=0; i < n_fit_parameters; i++) {
		datastream >> params[i];
	}
	dvector chain_params(params,n_fit_parameters);
	adopt_model(chain_params);

	delete[] params;
	return true;
}

bool QLens::adopt_point_from_chain_paramrange(const int paramnum, const double minval, const double maxval)
{
	int i, nparams, n_dparams, n_tot_parameters, ndparam;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams;
	n_tot_parameters = nparams + n_dparams;
	if (nparams != n_fit_parameters) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	if (paramnum >= n_tot_parameters) { warn("parameter number is less than number of fit+derived parameters in chain"); return false; }
	pnumfile.close();
	if (paramnum >= n_fit_parameters) {
		ndparam = paramnum - n_fit_parameters;
		if (mpi_id==0) cout << "The parameter selected is derived parameter no. " << ndparam << endl;
	}

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[n_tot_parameters];

	string chain_str = fit_output_dir + "/" + fit_output_filename;
	ifstream chain_file(chain_str.c_str());

	unsigned long nline=0, line_num;
	double weight, max_weight = -1e30;
	double chisq, minchisq = 1e30;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		//if (dataline[0]=='#') cout << string(dataline) << endl;
		istringstream datastream(dataline);
		datastream >> weight;
		for (i=0; i < n_tot_parameters; i++) {
			datastream >> params[i];
		}
		datastream >> chisq;
		//cout << params[paramnum] << endl;
		if ((params[paramnum] > minval) and (params[paramnum] < maxval) and (chisq < minchisq)) {
			//max_weight = weight;
			minchisq = chisq;
			line_num = nline;
		}

		nline++;
	}
	if (mpi_id==0) cout << "Line number of point adopted: " << line_num << " (out of " << nline << " total lines)" << endl;
	//if (max_weight==-1e30) { warn("no points from chain fell within range min/max values for specified parameter"); return false; }
	if (minchisq==1e30) { warn("no points from chain fell within range min/max values for specified parameter"); return false; }

	chain_file.close();
	chain_file.open(chain_str.c_str());
	for (i=0; i <= line_num; i++) {
		chain_file.getline(dataline,n_characters);
	}
	if (dataline[0]=='#') { warn("line from chain file is a comment line"); return false; }
	istringstream datastream(dataline);
	datastream >> weight;
	for (i=0; i < n_fit_parameters; i++) {
		datastream >> params[i];
	}
	dvector chain_params(params,n_fit_parameters);
	adopt_model(chain_params);

	delete[] params;
	return true;
}

bool QLens::plot_kappa_profile_percentiles_from_chain(int lensnum, double rmin, double rmax, int nbins, const string kappa_filename)
{
	double zl = lens_list[lensnum]->get_redshift();
	double r, rstep = pow(rmax/rmin, 1.0/(nbins-1));
	double *rvals = new double[nbins];
	int i;
	for (i=0, r=rmin; i < nbins; i++, r *= rstep) {
		rvals[i] = r;
	}

	int nparams, n_dparams_old;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams_old;
	if (nparams != n_fit_parameters) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	pnumfile.close();

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[n_fit_parameters];

	string chain_str = fit_output_dir + "/" + fit_output_filename;
	ifstream chain_file(chain_str.c_str());

	unsigned long n_points=0;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		if (dataline[0]=='#') continue;
		n_points++;
	}
	chain_file.close();

	double **weights = new double*[nbins];
	double **weights2 = new double*[nbins];
	double **kappa_r_pts = new double*[nbins];
	double **kappa_avg_pts = new double*[nbins];
	for (i=0; i < nbins; i++) {
		kappa_r_pts[i] = new double[n_points];
		kappa_avg_pts[i] = new double[n_points];
		weights[i] = new double[n_points];
		weights2[i] = new double[n_points];
	}
	double *kappa_r_vals = new double[nbins];
	double *kappa_avg_vals = new double[nbins];

	chain_file.open(chain_str.c_str());
	int j=0;
	double weight, tot=0;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		if (dataline[0]=='#') continue;

		istringstream datastream(dataline);
		datastream >> weight;
		tot += weight;
		for (i=0; i < n_fit_parameters; i++) {
			datastream >> params[i];
		}
		dvector chain_params(params,n_fit_parameters);
		adopt_model(chain_params);
		//print_lens_list(false);
		lens_list[lensnum]->plot_kappa_profile(nbins,rvals,kappa_r_vals,kappa_avg_vals);
		for (i=0; i < nbins; i++) {
			kappa_r_pts[i][j] = kappa_r_vals[i];
			kappa_avg_pts[i][j] = kappa_avg_vals[i];
			//cout << "RK: " << rvals[i] << " " << kappa_r_pts[i][j] << endl;
			weights[i][j] = weight;
			weights2[i][j] = weight;
		}

		j++;
	}
	chain_file.close();

	for (i=0; i < nbins; i++) {
		sort(n_points,kappa_r_pts[i],weights[i]);
		sort(n_points,kappa_avg_pts[i],weights2[i]);
	}
	double kaplo1, kaplo2, kaphi1, kaphi2;
	double kaplo1_prev, kaplo2_prev, kaphi1_prev, kaphi2_prev;
	double slope_lo1, slope_lo2, slope_hi1, slope_hi2;
	double kavglo1, kavglo2, kavghi1, kavghi2;
	double mavglo1, mavglo2, mavghi1, mavghi2;
	ofstream outfile(kappa_filename.c_str());
	double sigma_cr_arcsec = sigma_crit_arcsec(zl, reference_source_redshift);
	double arcsec_to_kpc = angular_diameter_distance(zl)/(1e-3*(180/M_PI)*3600);
	double rval_kpc;
	for (i=0; i < nbins; i++) {
		kaplo1 = find_percentile(n_points, 0.02275, tot, kappa_r_pts[i], weights[i]);
		kaphi1 = find_percentile(n_points, 0.97725, tot, kappa_r_pts[i], weights[i]);
		kaplo2 = find_percentile(n_points, 0.15865, tot, kappa_r_pts[i], weights[i]);
		kaphi2 = find_percentile(n_points, 0.84135, tot, kappa_r_pts[i], weights[i]);
		if (i>0) {
			slope_lo1 = log(kaplo1/kaplo1_prev)/log(rvals[i]/rvals[i-1]);
			slope_hi1 = log(kaphi1/kaphi1_prev)/log(rvals[i]/rvals[i-1]);
			slope_lo2 = log(kaplo2/kaplo2_prev)/log(rvals[i]/rvals[i-1]);
			slope_hi2 = log(kaphi2/kaphi2_prev)/log(rvals[i]/rvals[i-1]);
		}
		kaplo1_prev = kaplo1;
		kaphi1_prev = kaphi1;
		kaplo2_prev = kaplo2;
		kaphi2_prev = kaphi2;
		if (i==0) {
			// stupid hack but it works
			kaplo1 = find_percentile(n_points, 0.02275, tot, kappa_r_pts[1], weights[1]);
			kaphi1 = find_percentile(n_points, 0.97725, tot, kappa_r_pts[1], weights[1]);
			kaplo2 = find_percentile(n_points, 0.15865, tot, kappa_r_pts[1], weights[1]);
			kaphi2 = find_percentile(n_points, 0.84135, tot, kappa_r_pts[1], weights[1]);
			slope_lo1 = log(kaplo1/kaplo1_prev)/log(rvals[i]/rvals[i-1]);
			slope_hi1 = log(kaphi1/kaphi1_prev)/log(rvals[i]/rvals[i-1]);
			slope_lo2 = log(kaplo2/kaplo2_prev)/log(rvals[i]/rvals[i-1]);
			slope_hi2 = log(kaphi2/kaphi2_prev)/log(rvals[i]/rvals[i-1]);
		}
		kavglo1 = find_percentile(n_points, 0.02275, tot, kappa_avg_pts[i], weights2[i]);
		kavghi1 = find_percentile(n_points, 0.97725, tot, kappa_avg_pts[i], weights2[i]);
		kavglo2 = find_percentile(n_points, 0.15865, tot, kappa_avg_pts[i], weights2[i]);
		kavghi2 = find_percentile(n_points, 0.84135, tot, kappa_avg_pts[i], weights2[i]);
		mavglo1 = kavglo1*M_PI*SQR(rvals[i])*sigma_cr_arcsec;
		mavghi1 = kavghi1*M_PI*SQR(rvals[i])*sigma_cr_arcsec;
		mavglo2 = kavglo2*M_PI*SQR(rvals[i])*sigma_cr_arcsec;
		mavghi2 = kavghi2*M_PI*SQR(rvals[i])*sigma_cr_arcsec;
		rval_kpc = rvals[i]*arcsec_to_kpc;
		outfile << rvals[i] << " " << rval_kpc << " " << kaplo1 << " " << kaphi1 << " " << kaplo2 << " " << kaphi2 << " " << kavglo1 << " " << kavghi1 << " " << kavglo2 << " " << kavghi2 << " " << mavglo1 << " " << mavghi1 << " " << mavglo2 << " " << mavghi2 << " " << slope_lo1 << " " << slope_hi1 << " " << slope_lo2 << " " << slope_hi2 << endl;
	}

	delete[] params;
	delete[] rvals;
	delete[] kappa_r_vals;
	delete[] kappa_avg_vals;
	for (i=0; i < nbins; i++) {
		delete[] kappa_r_pts[i];
		delete[] kappa_avg_pts[i];
		delete[] weights[i];
		delete[] weights2[i];
	}
	delete[] kappa_r_pts;
	delete[] kappa_avg_pts;
	delete[] weights;
	delete[] weights2;
	return true;
}

double QLens::find_percentile(const unsigned long npoints, const double pct, const double tot, double *pts, double *weights)
{
	double totsofar = 0;
	for (int j = 0; j < npoints; j++)
	{
		totsofar += weights[j];
		if (totsofar/tot >= pct)
		{
			return pts[j] + (pts[j-1] - pts[j])*(totsofar - pct*tot)/weights[j];
		}
	}
	return 0.0;
}

void QLens::test_fitmodel_invert()
{
	if (setup_fit_parameters(false)==false) return;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	initialize_fitmodel(false);
	fitmodel_loglike_extended_source_test(fitparams.array());
	if (mpi_id==0) {
		cout << endl;
		cout << "Testing inversion again to make sure values are consistent...\n";
	}
	fitmodel_loglike_extended_source_test(fitparams.array());
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
}

void QLens::chi_square_twalk()
{
	if (setup_fit_parameters(true)==false) return;
	fit_set_optimizations();
	if ((mpi_id==0) and (fit_output_dir != ".")) {
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		if (system(rmstring.c_str()) != 0) warn("could not delete old output directory for twalk results"); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
		create_output_directory();
	}
	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	InputPoint(fitparams.array(),upper_limits.array(),lower_limits.array(),upper_limits_initial.array(),lower_limits_initial.array(),n_fit_parameters);
	SetNDerivedParams(n_derived_params);

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	if (mpi_id==0) {
		// This code gets repeated in a few spots and should really be put in a separate function...DO THIS LATER!
		int i;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fit_parameters << " " << n_derived_params << endl;
		pnumfile.close();
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_fit_parameters; i++) pnamefile << transformed_parameter_names[i] << endl;
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list[i]->name << endl;
		pnamefile.close();
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_fit_parameters; i++) lpnamefile << transformed_parameter_names[i] << "\t" << transformed_latex_parameter_names[i] << endl;
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list[i]->name << "\t" << dparam_list[i]->latex_name << endl;
		lpnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < n_fit_parameters; i++)
		{
			prangefile << lower_limits[i] << " " << upper_limits[i] << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		if (param_markers != "") {
			string marker_str = fit_output_dir + "/" + fit_output_filename + ".markers";
			ofstream markerfile(marker_str.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	string filename = fit_output_dir + "/" + fit_output_filename;

	display_chisq_status = false; // just in case it was turned on

	use_ansi_characters = true;
	TWalk(filename.c_str(),0.9836,4,2.4,2.5,6.0,mcmc_tolerance,mcmc_threads,fitparams.array(),mcmc_logfile,NULL,chain_info,data_info);
	use_ansi_characters = false;
	bestfitparams.input(fitparams);
	chisq_bestfit = 2*(this->*LogLikePtr)(bestfitparams.array());

	//if (display_chisq_status) {
		//for (int i=0; i < n_sourcepts_fit; i++) cout << endl; // to get past the status signs for image position chi-square
		//cout << endl;
		//display_chisq_status = false;
	//}
#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for T-Walk: " << wt << endl;
	}
#endif
	if (mpi_id==0) {
		output_bestfit_model();
	}

	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
}

bool QLens::adopt_model(dvector &fitparams)
{
	if (nlens == 0) { if (mpi_id==0) warn(warnings,"No fit model has been specified"); return false; }
	if (n_fit_parameters == 0) { if (mpi_id==0) warn(warnings,"No best-fit point has been saved from a previous fit"); return false; }
	if (fitparams.size() != n_fit_parameters) {
		if (mpi_id==0) {
			if (fitparams.size()==0) warn(warnings,"fit has not been run; best-fit solution is not available");
			else warn(warnings,"Best-fit number of parameters does not match current number; this likely means your current lens/source model does not match the model that was used for fitting.");
		}
		return false;
	}
	int i, index=0;
	double transformed_params[n_fit_parameters];
	param_settings->inverse_transform_parameters(fitparams.array(),transformed_params);
	bool status;
	for (i=0; i < nlens; i++) {
		lens_list[i]->update_fit_parameters(transformed_params,index,status);
		lens_list[i]->reset_angle_modulo_2pi();
	}
	update_anchored_parameters_and_redshift_data();
	reset_grid(); // this will force it to redraw the critical curves if needed
	if (source_fit_mode == Point_Source) {
		if (use_analytic_bestfit_src) {
			output_analytic_srcpos(sourcepts_fit.data());
		} else {
			for (i=0; i < n_sourcepts_fit; i++) {
				if (vary_sourcepts_x[i]) sourcepts_fit[i][0] = transformed_params[index++];
				if (vary_sourcepts_y[i]) sourcepts_fit[i][1] = transformed_params[index++];
			}
		}
	} else if ((source_fit_mode == Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		for (i=0; i < n_sb; i++) {
			sb_list[i]->update_fit_parameters(transformed_params,index,status);
			sb_list[i]->reset_angle_modulo_2pi();
		}
	} else if (source_fit_mode == Pixellated_Source) {
		if ((vary_regularization_parameter) and (regularization_method != None))
			regularization_parameter = transformed_params[index++];
		if (vary_pixel_fraction) pixel_fraction = transformed_params[index++];
		if (vary_srcgrid_xshift) srcgrid_xshift = transformed_params[index++];
		if (vary_srcgrid_yshift) srcgrid_yshift = transformed_params[index++];
		if (vary_srcgrid_size_scale) srcgrid_size_scale = transformed_params[index++];
		if (vary_magnification_threshold) pixel_magnification_threshold = transformed_params[index++];
	}
	if (source_fit_mode == Shapelet_Source) {
		if ((vary_regularization_parameter) and (regularization_method != None))
			regularization_parameter = transformed_params[index++];
	}

	if (vary_hubble_parameter) {
		hubble = transformed_params[index++];
		set_cosmology(omega_matter,0.04,hubble,2.215);
	}
	if (vary_omega_matter_parameter) {
		omega_matter = transformed_params[index++];
		set_cosmology(omega_matter,0.04,hubble,2.215);
	}
	if (vary_syserr_pos_parameter) {
		syserr_pos = transformed_params[index++];
	}
	if (vary_wl_shear_factor_parameter) {
		wl_shear_factor = transformed_params[index++];
	}
	if ((index != n_fit_parameters) and (mpi_id==0)) die("Index didn't go through all the fit parameters (%i); this likely means your current lens model does not match the lens model that was used for fitting.",n_fit_parameters);
	if (source_fit_mode==Shapelet_Source) {
		// we run the following function so it updates amplitudes (and also sigma, xc, yc if using auto_shapelet_scaling)
		if (invert_surface_brightness_map_from_data(false)==false) warn("no data loaded; cannot update shapelet amplitudes");
	}
	return true;
}

int FISHER_KEEP_RUNNING = 1;

void fisher_sighandler(int sig)
{
	FISHER_KEEP_RUNNING = 0;
}

void fisher_quitproc(int sig)
{
	exit(0);
}

bool QLens::calculate_fisher_matrix(const dvector &params, const dvector &stepsizes)
{
	// this function calculates the marginalized error using the Gaussian approximation
	// (only accurate if we are near maximum likelihood point and it is close to Gaussian around this point)
	static const double increment2 = 1e-4;
	if ((mpi_id==0) and (source_fit_mode==Point_Source) and (!imgplane_chisq) and (!use_magnification_in_chisq)) warn("Fisher matrix errors may not be accurate if source plane chi-square is used without magnification");

	dmatrix fisher(n_fit_parameters,n_fit_parameters);
	fisher_inverse.erase();
	fisher_inverse.input(n_fit_parameters,n_fit_parameters);
	dvector xhi(params);
	dvector xlo(params);
	double x0, curvature;
	int i,j;
	double step, derivlo, derivhi;
	for (i=0; i < n_fit_parameters; i++) {
		x0 = params[i];
		xhi[i] += increment2*stepsizes[i];
		if ((param_settings->use_penalty_limits[i]==true) and (xhi[i] > param_settings->penalty_limits_hi[i])) xhi[i] = x0;
		xlo[i] -= increment2*stepsizes[i];
		if ((param_settings->use_penalty_limits[i]==true) and (xlo[i] < param_settings->penalty_limits_lo[i])) xlo[i] = x0;
		step = xhi[i] - xlo[i];
		for (j=0; j < n_fit_parameters; j++) {
			derivlo = loglike_deriv(xlo,j,stepsizes[j]);
			derivhi = loglike_deriv(xhi,j,stepsizes[j]);
			fisher[i][j] = (derivhi - derivlo) / step;
			if (fisher[i][j]*0.0) warn(warnings,"Fisher matrix element (%i,%i) calculated as 'nan'",i,j);
			//if (i==j) cout << abs(derivlo+derivhi) << " " << sqrt(abs(fisher[i][j])) << endl;
			if ((mpi_id==0) and (i==j) and (abs(derivlo+derivhi) > sqrt(abs(fisher[i][j])))) warn(warnings,"Derivatives along parameter %i indicate best-fit point may not be at a local minimum of chi-square",i);
			signal(SIGABRT, &fisher_sighandler);
			signal(SIGTERM, &fisher_sighandler);
			signal(SIGINT, &fisher_sighandler);
			signal(SIGUSR1, &fisher_sighandler);
			signal(SIGQUIT, &fisher_quitproc);
			if (!FISHER_KEEP_RUNNING) {
				fisher_inverse.erase();
				return false;
			}
		}
		xhi[i]=xlo[i]=x0;
	}

	double offdiag_avg;
	// average the off-diagonal elements to enforce symmetry
	for (i=1; i < n_fit_parameters; i++) {
		for (j=0; j < i; j++) {
			offdiag_avg = 0.5*(fisher[i][j]+ fisher[j][i]);
			//if (abs((fisher[i][j]-fisher[j][i])/offdiag_avg) > 0.01) die("Fisher off-diags differ by more than 1%!");
			fisher[i][j] = fisher[j][i] = offdiag_avg;
		}
	}
	bool nonsingular = fisher.check_nan();
	if (nonsingular) fisher.inverse(fisher_inverse,nonsingular);
	if (!nonsingular) {
		if (mpi_id==0) warn(warnings,"Fisher matrix is singular, cannot be inverted\n");
		fisher_inverse.erase();
		return false;
	}
	return true;
}

double QLens::loglike_deriv(const dvector &params, const int index, const double step)
{
	static const double increment = 1e-5;
	dvector xhi(params);
	dvector xlo(params);
	double dif, x0 = xhi[index];
	xhi[index] += increment*step;
	if ((param_settings->use_penalty_limits[index]==true) and (xhi[index] > param_settings->penalty_limits_hi[index])) xhi[index] = x0;
	xlo[index] -= increment*step;
	if ((param_settings->use_penalty_limits[index]==true) and (xlo[index] < param_settings->penalty_limits_lo[index])) xlo[index] = x0;
	dif = xhi[index] - xlo[index];
	double (QLens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}
	return (((this->*loglikeptr)(xhi.array()) - (this->*loglikeptr)(xlo.array())) / dif);
}

void QLens::output_bestfit_model()
{
	if (nlens == 0) { warn(warnings,"No fit model has been specified"); return; }
	if (n_fit_parameters == 0) { warn(warnings,"No best-fit point has been saved from a previous fit"); return; }
	if (bestfitparams.size() != n_fit_parameters) { warn(warnings,"Best-fit point number of params does not match current number"); return; }
	if (fit_output_dir != ".") create_output_directory();

	int i;
	string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
	ofstream pnamefile(pnamefile_str.c_str());
	for (i=0; i < n_fit_parameters; i++) pnamefile << transformed_parameter_names[i] << endl;
	for (i=0; i < n_derived_params; i++) pnamefile << dparam_list[i]->name << endl;
	pnamefile.close();
	string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
	ofstream lpnamefile(lpnamefile_str.c_str());
	for (i=0; i < n_fit_parameters; i++) lpnamefile << transformed_parameter_names[i] << "\t" << transformed_latex_parameter_names[i] << endl;
	for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list[i]->name << "\t" << dparam_list[i]->latex_name << endl;
	lpnamefile.close();

	string bestfit_filename = fit_output_dir + "/" + fit_output_filename + ".bf";
	int n,j;
	ofstream bf_out(bestfit_filename.c_str());
	bf_out << chisq_bestfit << " ";
	for (i=0; i < n_fit_parameters; i++) bf_out << bestfitparams[i] << " ";
	bf_out << endl;
	bf_out.close();

	string outfile_str = fit_output_dir + "/" + fit_output_filename + ".bestfit";
	ofstream outfile(outfile_str.c_str());
	if ((calculate_parameter_errors) and (bestfit_fisher_inverse.is_initialized()))
	{
		if (bestfit_fisher_inverse.rows() != n_fit_parameters) die("dimension of Fisher matrix does not match number of fit parameters (%i vs %i)",bestfit_fisher_inverse.rows(),n_fit_parameters);
		string fisher_inv_filename = fit_output_dir + "/" + fit_output_filename + ".pcov"; // inverse-fisher matrix is the parameter covariance matrix
		ofstream fisher_inv_out(fisher_inv_filename.c_str());
		for (i=0; i < n_fit_parameters; i++) {
			for (j=0; j < n_fit_parameters; j++) {
				fisher_inv_out << bestfit_fisher_inverse[i][j] << " ";
			}
			fisher_inv_out << endl;
		}

		outfile << "Best-fit model: 2*loglike = " << chisq_bestfit << endl;
		if ((include_flux_chisq) and (bestfit_flux != 0)) outfile << "Best-fit source flux = " << bestfit_flux << endl;
		outfile << endl;
		outfile << "Marginalized 1-sigma errors from Fisher matrix:\n";
		for (int i=0; i < n_fit_parameters; i++) {
			outfile << transformed_parameter_names[i] << ": " << bestfitparams[i] << " +/- " << sqrt(abs(bestfit_fisher_inverse[i][i])) << endl;
		}
		outfile << endl;
	} else {
		if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
			outfile << "Best-fit model: 2*loglike = " << chisq_bestfit << " (warning: errors omitted here because Fisher matrix was not calculated):\n";
		} else {
			outfile << "Best-fit model: 2*loglike = " << chisq_bestfit << endl;
		}
		if ((include_flux_chisq) and (bestfit_flux != 0)) outfile << "Best-fit source flux = " << bestfit_flux << endl;
		outfile << endl;
		for (int i=0; i < n_fit_parameters; i++) {
			outfile << transformed_parameter_names[i] << ": " << bestfitparams[i] << endl;
		}
		outfile << endl;
	}
	string prange_str = fit_output_dir + "/" + fit_output_filename + ".pranges";
	ofstream prangefile(prange_str.c_str());
	for (int i=0; i < n_fit_parameters; i++)
	{
		if (param_settings->use_penalty_limits[i])
			prangefile << param_settings->penalty_limits_lo[i] << " " << param_settings->penalty_limits_hi[i] << endl;
		else
			prangefile << "-1e30 1e30" << endl;
	}
	prangefile.close();
	string script_str = fit_output_dir + "/" + fit_output_filename + ".commands";
	ofstream scriptfile(script_str.c_str());
	for (int i=0; i < lines.size()-1; i++) {
		scriptfile << lines[i] << endl;
	}
	scriptfile.close();

	QLens* model;
	if (fitmodel != NULL) model = fitmodel;
	else model = this;
	// In order to save the commands for the best-fit model, we adopt the best-fit model in the fitmodel object (if available);
	// that way we're not forced to adopt it in the user-end lens object if the user doesn't want to
	if (model == fitmodel) {
		model->bestfitparams.input(bestfitparams);
		model->adopt_model(bestfitparams);
	}
	bool include_limits;
	if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) include_limits = false;
	else include_limits = true;
	string scriptfile_str = fit_output_dir + "/" + fit_output_filename + "_bf.in";
	model->output_lens_commands(scriptfile_str,include_limits);
	if (include_limits) {
		// save version without limits in case user wants to load best-fit model while in Simplex or Powell mode
		string scriptfile_str2 = fit_output_dir + "/" + fit_output_filename + "_bf_nolimits.in";
		model->output_lens_commands(scriptfile_str2,false);
	}
}

double QLens::get_einstein_radius_prior(const bool verbal)
{
	double re, loglike_penalty = 0;
	einstein_radius_of_primary_lens(reference_zfactors[lens_redshift_idx[primary_lens_number]],re);
	//loglike_penalty = SQR((re-einstein_radius_threshold)/0.1);
	if (re < einstein_radius_low_threshold) {
		loglike_penalty = pow(1-re+einstein_radius_low_threshold,40) - 1.0; // constructed so that penalty = 0 if the average n_image = n_image_threshold
		if ((mpi_id==0) and (verbal)) cout << "*NOTE: Einstein radius is below the low prior threshold (" << re << " vs. " << einstein_radius_low_threshold << "), resulting in penalty prior (loglike_penalty=" << loglike_penalty << ")" << endl;
	}
	else if (re > einstein_radius_high_threshold) {
		loglike_penalty = pow(1+re-einstein_radius_high_threshold,40) - 1.0; // constructed so that penalty = 0 if the average n_image = n_image_threshold
		if ((mpi_id==0) and (verbal)) cout << "*NOTE: Einstein radius is above the high prior threshold (" << re << " vs. " << einstein_radius_high_threshold << "), resulting in penalty prior (loglike_penalty=" << loglike_penalty << ")" << endl;
	}
	return loglike_penalty;
}

double QLens::fitmodel_loglike_point_source(double* params)
{
	bool showed_first_chisq = false; // used just to know whether to print a comma before showing the next chisq component
	double transformed_params[n_fit_parameters];
	if (params != NULL) {
		fitmodel->param_settings->inverse_transform_parameters(params,transformed_params);
		//fitmodel->param_settings->print_penalty_limits();
		bool penalty_incurred = false;
		for (int i=0; i < n_fit_parameters; i++) {
			if (fitmodel->param_settings->use_penalty_limits[i]==true) {
				//cout << "USE_LIMITS " << i << endl;
				if ((transformed_params[i] < fitmodel->param_settings->penalty_limits_lo[i]) or (transformed_params[i] > fitmodel->param_settings->penalty_limits_hi[i])) penalty_incurred = true;
			}
		}
		//fitmodel->param_settings->print_penalty_limits();
		if (penalty_incurred) return 1e30;
		if (update_fitmodel(transformed_params)==false) return 1e30;
		if (group_id==0) {
			if (fitmodel->logfile.is_open()) {
				for (int i=0; i < n_fit_parameters; i++) fitmodel->logfile << params[i] << " ";
			}
			fitmodel->logfile << flush;
		}
	}

	double loglike, chisq_total=0, chisq;
	if (include_imgpos_chisq) {
		bool used_imgplane_chisq; // keeps track of whether image plane chi-square gets used, since there is an option to switch from srcplane to imgplane below a given threshold
		double rms_err;
		int n_matched_imgs;
		if (imgplane_chisq) {
			used_imgplane_chisq = true;
			if (chisq_diagnostic) chisq = fitmodel->chisq_pos_image_plane_diagnostic(true,false,rms_err,n_matched_imgs);
			else chisq = fitmodel->chisq_pos_image_plane();
		}
		else {
			used_imgplane_chisq = false;
			chisq = fitmodel->chisq_pos_source_plane();
			if (chisq < chisq_imgplane_substitute_threshold) {
				if (chisq_diagnostic) chisq = fitmodel->chisq_pos_image_plane_diagnostic(true,false,rms_err,n_matched_imgs);
				else chisq = fitmodel->chisq_pos_image_plane();
				used_imgplane_chisq = true;
			}
		}
		if ((display_chisq_status) and (mpi_id==0)) {
			if (use_ansi_characters) cout << "\033[2A" << flush;
			if (include_imgpos_chisq) {
				if (used_imgplane_chisq) {
					if (!imgplane_chisq) cout << "imgplane_chisq: "; // so user knows the imgplane chi-square is being used (we're below the threshold to switch from srcplane to imgplane)
					int tot_data_images = 0;
					for (int i=0; i < n_sourcepts_fit; i++) tot_data_images += image_data[i].n_images;
					if (use_ansi_characters) cout << "# images: " << fitmodel->n_visible_images << " vs. " << tot_data_images << " data";
					if (fitmodel->chisq_it % chisq_display_frequency == 0) {
						if (!use_ansi_characters) cout << "# images: " << fitmodel->n_visible_images << " vs. " << tot_data_images << " data";
						cout << ", chisq_pos=" << chisq;
						if (syserr_pos != 0.0) {
							double signormfac, chisq_sys = chisq;
							int i,k;
							for (i=0; i < n_sourcepts_fit; i++) {
								for (k=0; k < image_data[i].n_images; k++) {
									signormfac = 2*log(1.0 + SQR(fitmodel->syserr_pos/image_data[i].sigma_pos[k]));
									chisq_sys -= signormfac;
								}
							}
							cout << ", chisq_pos_sys=" << chisq_sys;
						}
						showed_first_chisq = true;
					}
				} else {
					if (fitmodel->chisq_it % chisq_display_frequency == 0) {
						cout << "chisq_pos=" << chisq;
						// redundant and ugly! make it prettier later
						if (syserr_pos != 0.0) {
							double signormfac, chisq_sys = chisq;
							int i,k;
							for (i=0; i < n_sourcepts_fit; i++) {
								for (k=0; k < image_data[i].n_images; k++) {
									signormfac = 2*log(1.0 + SQR(fitmodel->syserr_pos/image_data[i].sigma_pos[k]));
									chisq_sys -= signormfac;
								}
							}
							cout << ", chisq_pos_sys=" << chisq_sys;
						}
						showed_first_chisq = true;
					}
				}
			}
		}
	} else {
		if ((display_chisq_status) and (mpi_id==0)) {
			if (use_ansi_characters) cout << "\033[2A" << flush;
			if ((fitmodel->chisq_it % chisq_display_frequency == 0) and (include_imgpos_chisq)) {
				cout << "chisq_pos=0";
				showed_first_chisq = true;
			}
		}
	}
	chisq_total += chisq;
	if (include_flux_chisq) {
		chisq = fitmodel->chisq_flux();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (showed_first_chisq) cout << ", ";
			else showed_first_chisq = true;
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_flux=" << chisq;
		}
	}
	if (include_time_delay_chisq) {
		chisq = fitmodel->chisq_time_delays();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (showed_first_chisq) cout << ", ";
			else showed_first_chisq = true;
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_td=" << chisq;
		}
	}
	if (include_weak_lensing_chisq) {
		chisq = fitmodel->chisq_weak_lensing();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (showed_first_chisq) cout << ", ";
			else showed_first_chisq = true;
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_weak_lensing=" << chisq;
		}
	}
	raw_chisq = chisq_total; // in case the chi-square is being used as a derived parameter
	fitmodel->raw_chisq = chisq_total;
	loglike = chisq_total/2.0;
	if (chisq*0.0 != 0.0) {
		warn("chi-square is returning NaN (%g)",chisq);
	}
 
 	if (params != NULL) {
		fitmodel->param_settings->add_prior_terms_to_loglike(params,loglike);
		fitmodel->param_settings->add_jacobian_terms_to_loglike(transformed_params,loglike);
		if (use_custom_prior) loglike += fitmodel_custom_prior();
	}
	if (einstein_radius_prior) loglike += fitmodel->get_einstein_radius_prior(false);
	if ((display_chisq_status) and (mpi_id==0)) {
		if (fitmodel->chisq_it % chisq_display_frequency == 0) {
			if (chisq_total != (2*loglike)) cout << ", chisq_tot=" << chisq_total;
			cout << ", -2*loglike=" << 2*loglike;
			cout << "                ";
			if (!use_ansi_characters) cout << endl;
		}
		if (use_ansi_characters) cout << endl << endl;
	}


	fitmodel->chisq_it++;
	return loglike;
}

double QLens::fitmodel_loglike_extended_source(double* params)
{
	double transformed_params[n_fit_parameters];
	if (params != NULL) {
		fitmodel->param_settings->inverse_transform_parameters(params,transformed_params);
		for (int i=0; i < n_fit_parameters; i++) {
			if (fitmodel->param_settings->use_penalty_limits[i]==true) {
				//cout << "parameter " << i << ": plimits " << fitmodel->param_settings->penalty_limits_lo[i] << fitmodel->param_settings->penalty_limits_hi[i] << endl;
				if ((transformed_params[i] < fitmodel->param_settings->penalty_limits_lo[i]) or (transformed_params[i] > fitmodel->param_settings->penalty_limits_hi[i])) return 1e30;
			}
			//else cout << "parameter " << i << ": no plimits " << endl;
		}
		if (update_fitmodel(transformed_params)==false) return 1e30;
		if (group_id==0) {
			if (fitmodel->logfile.is_open()) {
				for (int i=0; i < n_fit_parameters; i++) fitmodel->logfile << params[i] << " ";
				fitmodel->logfile << flush;
			}
		}
	}
	double loglike, chisq=0, chisq0;
	if (chisq != 2e30) {
		if ((source_fit_mode==Pixellated_Source) and ((fitmodel->regularization_parameter < 0) or (fitmodel->pixel_fraction <= 0))) chisq = 2e30;
		else chisq = fitmodel->invert_image_surface_brightness_map(chisq0,false);
	}

	raw_chisq = chisq0; // in case the chi-square is being used as a derived parameter
	fitmodel->raw_chisq = chisq0;
	loglike = chisq/2.0;
	if (params != NULL) {
		fitmodel->param_settings->add_prior_terms_to_loglike(params,loglike);
		fitmodel->param_settings->add_jacobian_terms_to_loglike(transformed_params,loglike);
		if (use_custom_prior) loglike = fitmodel_custom_prior();
	}
	if (einstein_radius_prior) loglike += fitmodel->get_einstein_radius_prior(false);
	if ((display_chisq_status) and (mpi_id==0)) {
		if (use_ansi_characters) cout << "\033[2A" << flush;
		cout << "chisq0=" << chisq0;
		cout << ", chisq_pix=" << chisq;
		cout << ", -2*loglike=" << 2*loglike;
		cout << "                " << endl;

		//cout << "\033[1A";
		if (use_ansi_characters) cout << endl;
	}

	fitmodel->chisq_it++;
	return loglike;
}

double QLens::loglike_point_source(double* params)
{
	// can use this version for testing purposes in case there is any doubt about whether the fitmodel version is faithfully reproducing the original
	double transformed_params[n_fit_parameters];
	param_settings->inverse_transform_parameters(params,transformed_params);
	for (int i=0; i < n_fit_parameters; i++) {
		if (param_settings->use_penalty_limits[i]==true) {
			if ((transformed_params[i] < param_settings->penalty_limits_lo[i]) or (transformed_params[i] > param_settings->penalty_limits_hi[i])) return 1e30;
		}
	}
	if (update_fitmodel(transformed_params)==false) return 1e30;
	if (group_id==0) {
		if (logfile.is_open()) {
			for (int i=0; i < n_fit_parameters; i++) logfile << params[i] << " ";
		}
		logfile << flush;
	}

	double loglike, chisq_total=0, chisq;
	if (imgplane_chisq) {
		chisq = chisq_pos_image_plane();
		if ((display_chisq_status) and (mpi_id==0)) {
			int tot_data_images = 0;
			for (int i=0; i < n_sourcepts_fit; i++) tot_data_images += image_data[i].n_images;
			cout << "# images: " << n_visible_images << " vs. " << tot_data_images << " data, ";
			if (chisq_it % chisq_display_frequency == 0) cout << "chisq_pos=" << chisq;
		}
	}
	else {
		chisq = chisq_pos_source_plane();
		if ((display_chisq_status) and (mpi_id==0)) {
			if (chisq_it % chisq_display_frequency == 0) cout << "chisq_pos=" << chisq;
		}
	}
	chisq_total += chisq;
	if (include_flux_chisq) {
		chisq = chisq_flux();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (chisq_it % chisq_display_frequency == 0) cout << ", chisq_flux=" << chisq;
		}
	}
	if (include_time_delay_chisq) {
		chisq = chisq_time_delays();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (chisq_it % chisq_display_frequency == 0) cout << ", chisq_td=" << chisq;
		}
	}
	if ((display_chisq_status) and (mpi_id==0)) {
		if (chisq_it % chisq_display_frequency == 0) cout << ", chisq_tot=" << chisq_total << "               ";
		cout << endl;
		//cout << "\033[1A";
	}

	loglike = chisq_total/2.0;

	param_settings->add_prior_terms_to_loglike(params,loglike);
	param_settings->add_jacobian_terms_to_loglike(transformed_params,loglike);
	if (use_custom_prior) loglike = fitmodel_custom_prior();
	chisq_it++;
	return loglike;
}

double QLens::fitmodel_loglike_extended_source_test(double* params)
{
	double transformed_params[n_fit_parameters];
	fitmodel->param_settings->inverse_transform_parameters(params,transformed_params);
	for (int i=0; i < n_fit_parameters; i++) {
		if (fitmodel->param_settings->use_penalty_limits[i]==true) {
			if ((transformed_params[i] < fitmodel->param_settings->penalty_limits_lo[i]) or (transformed_params[i] > fitmodel->param_settings->penalty_limits_hi[i])) return 1e30;
		}
	}
	if (update_fitmodel(transformed_params)==false) return 1e30;

	if (group_id==0) {
		if (fitmodel->logfile.is_open()) {
			for (int i=0; i < n_fit_parameters; i++) fitmodel->logfile << params[i] << " ";
			dvector subhalo_params;
		}
	}
	double loglike, chisq, chisq0;
	if (fitmodel->regularization_parameter < 0) chisq = 2e30;
	else if (fitmodel->pixel_fraction <= 0) chisq = 2e30;
	else chisq = fitmodel->invert_image_surface_brightness_map(chisq0,true);
	loglike = chisq/2.0;
	fitmodel->param_settings->add_prior_terms_to_loglike(params,loglike);
	fitmodel->param_settings->add_jacobian_terms_to_loglike(transformed_params,loglike);
	return loglike;
}

void QLens::fitmodel_calculate_derived_params(double* params, double* derived_params)
{
	double transformed_params[n_fit_parameters];
	fitmodel->param_settings->inverse_transform_parameters(params,transformed_params);
	if (update_fitmodel(transformed_params)==false) warn("derived params for point incurring penalty chi-square may give absurd results");
	for (int i=0; i < n_derived_params; i++) derived_params[i] = dparam_list[i]->get_derived_param(fitmodel);
}

double QLens::get_lens_parameter_using_default_pmode(const int paramnum, const int lensnum)
{
	if (lensnum >= nlens) die("lensnum exceeds number of lenses");
	int lens_nparams = lens_list[lensnum]->get_n_params();
	if (paramnum >= lens_nparams) die("for lensparam, lens parameter number exceeds total number of parameters in lens");
	double lensparam;
	double *lensparams = new double[lens_nparams];
	lens_list[lensnum]->get_parameters_pmode(default_parameter_mode,lensparams);
	lensparam = lensparams[paramnum];
	delete[] lensparams;
	return lensparam;
}

double QLens::fitmodel_custom_prior()
{
	//static const double rcore_threshold = 3.0;
	double cnfw_params[8];
	double rc, rs, rcore;
	if (fitmodel != NULL)
		fitmodel->lens_list[0]->get_parameters_pmode(0,cnfw_params);
	else
		lens_list[0]->get_parameters_pmode(0,cnfw_params); // used for the "test" command"
	rs = cnfw_params[1];
	rc = cnfw_params[2];
	//rcore = rc*(sqrt(1+8*rs/rc)-1)/4.0;
	//if (fitmodel==NULL) cout << "rcore: " << rcore << endl; // for testing purposes, using the "test" command
	//if (rcore < rcore_threshold) return 0.0;
	//else return 1e30+rcore; // penalty function
	if (rc < rs) return 0.0;
	else return 1e30+rc;
}

void QLens::set_Gauss_NN(const int& nn)
{
	Gauss_NN = nn;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->SetGaussLegendre(nn);
		}
	}
}

void QLens::set_integral_tolerance(const double& acc)
{
	integral_tolerance = acc;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->set_integral_tolerance(acc);
		}
	}
}

void QLens::reassign_lensparam_pointers_and_names()
{
	// parameter pointers should be reassigned if the parameterization mode has been changed (e.g., shear components turned on/off)
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->calculate_ellipticity_components(); // in case ellipticity components has been turned on
			lens_list[i]->assign_param_pointers();
			lens_list[i]->assign_paramnames();
		}
		set_default_plimits();
		update_parameter_list();
		if (mpi_id==0) cout << "NOTE: plimits have been reset, since lens parameterization has been changed" << endl;
	}
}

void QLens::reassign_sb_param_pointers_and_names()
{
	// parameter pointers should be reassigned if the parameterization mode has been changed
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			sb_list[i]->calculate_ellipticity_components(); // in case ellipticity components has been turned on
			sb_list[i]->assign_param_pointers();
			sb_list[i]->assign_paramnames();
		}
		set_default_plimits();
		update_parameter_list();
		if (mpi_id==0) cout << "NOTE: plimits have been reset, since source parameterization has been changed" << endl;
	}
}

void QLens::print_lens_cosmology_info(const int lmin, const int lmax)
{
	if (lmax >= nlens) return;
	double sigma_cr = sigma_crit_kpc(lens_redshift,reference_source_redshift);
	double dlens = angular_diameter_distance(lens_redshift);
	cout << "H0 = " << hubble*100 << " km/s/Mpc" << endl;
	cout << "omega_m = " << omega_matter << endl;
	//cout << "omega_lambda = " << 1-omega_matter << endl;
	cout << "zlens = " << lens_redshift << endl;
	cout << "zsrc = " << source_redshift << endl;
	cout << "D_lens: " << dlens << " Mpc  (angular diameter distance to lens plane)" << endl;
	double rhocrit = 1e-9*critical_density(lens_redshift);
	cout << "rho_crit(zlens): " << rhocrit << " M_sol/kpc^3" << endl;
	cout << "Sigma_crit(zlens,zsrc_ref): " << sigma_cr << " M_sol/kpc^2" << endl;
	double kpc_to_arcsec = 206.264806/angular_diameter_distance(lens_redshift);
	cout << "1 arcsec = " << (1.0/kpc_to_arcsec) << " kpc" << endl;
	cout << "sigma8 = " << rms_sigma8() << endl;
	cout << endl;
	if (nlens > 0) {
		for (int i=lmin; i <= lmax; i++) {
			lens_list[i]->output_cosmology_info(i);
		}
	}
	else cout << "No lens models have been specified" << endl << endl;
}

bool QLens::output_mass_r(const double r, const int lensnum, const bool use_kpc)
{
	if (lensnum >= nlens) return false;
	double zlens, sigma_cr, kpc_to_arcsec, r_arcsec, r_kpc, mass_r_2d, rho_r_3d, mass_r_3d;
	double zl = lens_list[lensnum]->zlens;
	sigma_cr = sigma_crit_arcsec(zl,reference_source_redshift);
	kpc_to_arcsec = 206.264806/angular_diameter_distance(zl);
	if (!use_kpc) {
		r_kpc = r/kpc_to_arcsec;
		r_arcsec = r;
	} else {
		r_kpc = r;
		r_arcsec = r*kpc_to_arcsec;
	}
	cout << "Radius: " << r_kpc << " kpc (" << r_arcsec << " arcsec)\n";
	mass_r_2d = sigma_cr*lens_list[lensnum]->mass_rsq(r_arcsec*r_arcsec);
	cout << "Mass enclosed (2D): " << mass_r_2d << " M_sol" << endl;
	bool converged;
	rho_r_3d = (sigma_cr*CUBE(kpc_to_arcsec))*lens_list[lensnum]->calculate_scaled_density_3d(r_arcsec,1e-4,converged);
	cout << "Density (3D): " << rho_r_3d << " M_sol/kpc^3" << endl;
	mass_r_3d = sigma_cr*lens_list[lensnum]->calculate_scaled_mass_3d(r_arcsec);
	//double mass_r_3d_unscaled = mass_r_3d/sigma_cr;
	//double rho_r_3d_noscale = lens_list[lensnum]->calculate_scaled_density_3d(r_arcsec);
	cout << "Mass enclosed (3D): " << mass_r_3d << " M_sol" << endl;
	//cout << "Mass enclosed (3D) unscaled: " << mass_r_3d_unscaled << " M_sol" << endl;
	//cout << "Density unscaled (3D): " << rho_r_3d_noscale << " arcsec^-1" << endl;
	cout << endl;
	return true;
}

double QLens::mass2d_r(const double r, const int lensnum, const bool use_kpc)
{
	double sigma_cr, mass_r_2d, z;
	z = lens_list[lensnum]->zlens;
	double r_arcsec = (use_kpc) ? r*206.264806/angular_diameter_distance(z) : r;

	sigma_cr = sigma_crit_arcsec(z,reference_source_redshift);
	mass_r_2d = sigma_cr*lens_list[lensnum]->mass_rsq(r_arcsec*r_arcsec);
	return mass_r_2d;
}

double QLens::mass3d_r(const double r, const int lensnum, const bool use_kpc)
{
	double sigma_cr, mass_r_3d, z;
	z = lens_list[lensnum]->zlens;
	double r_arcsec = (use_kpc) ? r*206.264806/angular_diameter_distance(z) : r;
	sigma_cr = sigma_crit_arcsec(z,reference_source_redshift);
	mass_r_3d = sigma_cr*lens_list[lensnum]->calculate_scaled_mass_3d(r_arcsec);
	return mass_r_3d;
}

double QLens::calculate_average_log_slope(const int lensnum, const double rmin, const double rmax, const bool use_kpc)
{
	double z = lens_list[lensnum]->zlens;
	double kpc_to_arcsec = 206.264806/angular_diameter_distance(z);
	double rmin_arcsec = rmin, rmax_arcsec = rmax;
	if (use_kpc) {
		rmin_arcsec *= kpc_to_arcsec;
		rmax_arcsec *= kpc_to_arcsec;
	}
	return lens_list[lensnum]->average_log_slope(rmin_arcsec,rmax_arcsec);
}

void QLens::print_lens_list(bool show_vary_params)
{
	cout << resetiosflags(ios::scientific);
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			cout << i << ". ";
			lens_list[i]->print_parameters();
			//string lline = lens_list[i]->get_parameters_string();
			//cout << lline << endl;
			if (show_vary_params)
				lens_list[i]->print_vary_parameters();
		}
		if (source_redshift != reference_source_redshift) cout << "NOTE: for all lenses, kappa is defined by zsrc_ref = " << reference_source_redshift << endl;
	}
	else cout << "No lens models have been specified" << endl;
	cout << endl;
	if (use_scientific_notation) cout << setiosflags(ios::scientific);
}

void QLens::output_lens_commands(string filename, const bool print_limits)
{
	ofstream scriptfile(filename.c_str());
	if (print_limits) scriptfile << "#limits included" << endl;
	else scriptfile << "#nolimits included" << endl;
	for (int i=0; i < nlens; i++) {
		lens_list[i]->print_lens_command(scriptfile,print_limits);
	}
	if (source_fit_mode == Point_Source) {
		if (!sourcepts_fit.empty()) {
			if (!use_analytic_bestfit_src) {
				scriptfile << "fit sourcept\n";
				if (!print_limits) {
					for (int i=0; i < n_sourcepts_fit; i++) scriptfile << sourcepts_fit[i][0] << " " << sourcepts_fit[i][1] << endl;
				} else {
					for (int i=0; i < n_sourcepts_fit; i++) {
						scriptfile << sourcepts_fit[i][0] << " " << sourcepts_fit[i][1] << endl;
						scriptfile << sourcepts_lower_limit[i][0] << " " << sourcepts_upper_limit[i][0] << endl;
						scriptfile << sourcepts_lower_limit[i][1] << " " << sourcepts_upper_limit[i][1] << endl;
					}
				}
			} else {
				scriptfile << "fit sourcept auto\n";
			}
		} else if (n_sourcepts_fit > 0) scriptfile << "# Warning: Initial source point parameters not chosen\n";
	}
	if ((source_fit_mode == Pixellated_Source) or (source_fit_mode==Shapelet_Source)) {
		if (vary_regularization_parameter) {
			if (!print_limits) {
				scriptfile << "regparam " << regularization_parameter << endl;
			} else {
				scriptfile << "regparam " << regularization_parameter_lower_limit << " " << regularization_parameter << " " << regularization_parameter_upper_limit << endl;
			}
		}
		if (source_fit_mode==Pixellated_Source) {
			if (vary_magnification_threshold) {
				if (!print_limits) {
					scriptfile << "srcpixel_mag_threshold " << pixel_magnification_threshold << endl;
				} else {
					scriptfile << "srcpixel_mag_threshold " << pixel_magnification_threshold_lower_limit << " " << pixel_magnification_threshold << " " << pixel_magnification_threshold_upper_limit << endl;
				}
			}
		}
	}
	if ((source_fit_mode == Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		if (print_limits) scriptfile << "#limits included" << endl;
		else scriptfile << "#nolimits included" << endl;
		for (int i=0; i < n_sb; i++) {
			sb_list[i]->print_source_command(scriptfile,print_limits);
		}
	}

	if (vary_hubble_parameter) {
		scriptfile << "hubble = " << hubble << endl;
		if (print_limits) {
			scriptfile << hubble_lower_limit << " " << hubble_upper_limit << endl;
		}
	}
	if (vary_omega_matter_parameter) {
		scriptfile << "omega_m = " << omega_matter << endl;
		if (print_limits) {
			scriptfile << omega_matter_lower_limit << " " << omega_matter_upper_limit << endl;
		}
	}
	if (vary_syserr_pos_parameter) {
		scriptfile << "syserr_pos = " << syserr_pos << endl;
		if (print_limits) {
			scriptfile << syserr_pos_lower_limit << " " << syserr_pos_upper_limit << endl;
		}
	}
	if (vary_wl_shear_factor_parameter) {
		scriptfile << "wl_shearfac = " << wl_shear_factor << endl;
		if (print_limits) {
			scriptfile << wl_shear_factor_lower_limit << " " << wl_shear_factor_upper_limit << endl;
		}
	}
}

void QLens::print_fit_model()
{
	print_lens_list(true);
	if (source_fit_mode == Point_Source) {
		if (!sourcepts_fit.empty()) {
			if (!use_analytic_bestfit_src) {
				if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
					cout << "Initial fit coordinates for source points:\n";
					for (int i=0; i < n_sourcepts_fit; i++) cout << "Source point " << i << ": (" << sourcepts_fit[i][0] << "," << sourcepts_fit[i][1] << ")\n";
				} else {
					cout << "Initial fit coordinates and limits for source points:\n";
					for (int i=0; i < n_sourcepts_fit; i++) {
						cout << "Source point " << i << ": (" << sourcepts_fit[i][0] << "," << sourcepts_fit[i][1] << ")\n";
						cout << "x" << i << ": [" << sourcepts_lower_limit[i][0] << ":" << sourcepts_upper_limit[i][0] << "]\n";
						cout << "y" << i << ": [" << sourcepts_lower_limit[i][1] << ":" << sourcepts_upper_limit[i][1] << "]\n";
					}
				}
				cout << endl;
			}
		} else if (n_sourcepts_fit > 0) cout << "Initial source point parameters not chosen\n";
	}
	else if ((source_fit_mode == Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
		if (n_sb > 0) {
			cout << "Source profile list:" << endl;
			print_source_list(true);
		}
	}
	if ((source_fit_mode == Pixellated_Source) or (source_fit_mode==Shapelet_Source)) {
		if (vary_regularization_parameter) {
			if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
				cout << "Regularization parameter: " << regularization_parameter << endl;
			} else {
				if ((regularization_parameter_lower_limit==1e30) or (regularization_parameter_upper_limit==1e30)) cout << "\nRegularization parameter: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
				else cout << "Regularization parameter: [" << regularization_parameter_lower_limit << ":" << regularization_parameter << ":" << regularization_parameter_upper_limit << "]\n";
			}
		}
		if (source_fit_mode==Pixellated_Source) {
			if (vary_magnification_threshold) {
				if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
					cout << "Pixel magnification threshold: " << pixel_magnification_threshold << endl;
				} else {
					if ((pixel_magnification_threshold_lower_limit==1e30) or (pixel_magnification_threshold_upper_limit==1e30)) cout << "\nPixel magnification threshold: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
					else cout << "Pixel magnification threshold: [" << pixel_magnification_threshold_lower_limit << ":" << pixel_magnification_threshold << ":" << pixel_magnification_threshold_upper_limit << "]\n";
				}
			}
			if (vary_pixel_fraction) {
				if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
					cout << "Source grid pixel fraction: " << pixel_fraction << endl;
				} else {
					if ((pixel_fraction_lower_limit==1e30) or (pixel_fraction_upper_limit==1e30)) cout << "\nSource grid pixel fraction: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
					else cout << "Source grid pixel fraction: [" << pixel_fraction_lower_limit << ":" << pixel_fraction << ":" << pixel_fraction_upper_limit << "]\n";
				}
			}
			if (vary_srcgrid_xshift) {
				if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
					cout << "Source grid x-shift: " << srcgrid_xshift << endl;
				} else {
					if ((srcgrid_xshift_lower_limit==1e30) or (srcgrid_xshift_upper_limit==1e30)) cout << "\nSource grid x-shift: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
					else cout << "Source grid x-shift: [" << srcgrid_xshift_lower_limit << ":" << srcgrid_xshift << ":" << srcgrid_xshift_upper_limit << "]\n";
				}
			}
			if (vary_srcgrid_yshift) {
				if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
					cout << "Source grid y-shift: " << srcgrid_yshift << endl;
				} else {
					if ((srcgrid_yshift_lower_limit==1e30) or (srcgrid_yshift_upper_limit==1e30)) cout << "\nSource grid x-shift: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
					else cout << "Source grid y-shift: [" << srcgrid_yshift_lower_limit << ":" << srcgrid_yshift << ":" << srcgrid_yshift_upper_limit << "]\n";
				}
			}
			if (vary_srcgrid_size_scale) {
				if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
					cout << "Source grid scale: " << srcgrid_size_scale << endl;
				} else {
					if ((srcgrid_size_scale_lower_limit==1e30) or (srcgrid_size_scale_upper_limit==1e30)) cout << "\nSource grid scale: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
					else cout << "Source grid scale: [" << srcgrid_size_scale_lower_limit << ":" << srcgrid_size_scale << ":" << srcgrid_size_scale_upper_limit << "]\n";
				}
			}
		}
	}
	if (vary_hubble_parameter) {
		if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
			cout << "Hubble parameter: " << hubble << endl;
		} else {
			if ((hubble_lower_limit==1e30) or (hubble_upper_limit==1e30)) cout << "\nHubble parameter: lower/upper limits not given (these must be set by 'h0' command before fit)\n";
			else cout << "Hubble parameter: [" << hubble_lower_limit << ":" << hubble << ":" << hubble_upper_limit << "]\n";
		}
	}
	if (vary_omega_matter_parameter) {
		if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
			cout << "omega_m parameter: " << omega_matter << endl;
		} else {
			if ((omega_matter_lower_limit==1e30) or (omega_matter_upper_limit==1e30)) cout << "\nomega_m parameter: lower/upper limits not given (these must be set by 'h0' command before fit)\n";
			else cout << "omega_m parameter: [" << omega_matter_lower_limit << ":" << omega_matter << ":" << omega_matter_upper_limit << "]\n";
		}
	}
	if (vary_syserr_pos_parameter) {
		if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
			cout << "Systematic error parameter: " << syserr_pos << endl;
		} else {
			if ((syserr_pos_lower_limit==1e30) or (syserr_pos_upper_limit==1e30)) cout << "\nsyserr_pos parameter: lower/upper limits not given (these must be set by 'syserr_pos' command before fit)\n";
			else cout << "Systematic error parameter: [" << syserr_pos_lower_limit << ":" << syserr_pos << ":" << syserr_pos_upper_limit << "]\n";
		}
	}
	if (vary_wl_shear_factor_parameter) {
		if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
			cout << "Weak lensing shear factor parameter: " << wl_shear_factor << endl;
		} else {
			if ((wl_shear_factor_lower_limit==1e30) or (wl_shear_factor_upper_limit==1e30)) cout << "\nwl_shearfac parameter: lower/upper limits not given (these must be set by 'wl_shear_factor' command before fit)\n";
			else cout << "Weak lensing shear factor parameter: [" << wl_shear_factor_lower_limit << ":" << wl_shear_factor << ":" << wl_shear_factor_upper_limit << "]\n";
		}
	}
}

void QLens::plot_ray_tracing_grid(double xmin, double xmax, double ymin, double ymax, int x_N, int y_N, string filename)
{

	lensvector **corner_pts = new lensvector*[x_N];
	lensvector **corner_sourcepts = new lensvector*[x_N];
	int i,j;
	for (i=0; i < x_N; i++) {
		corner_pts[i] = new lensvector[y_N];
		corner_sourcepts[i] = new lensvector[y_N];
	}

	double x,y;
	double pixel_xlength = (xmax-xmin)/(x_N-1);
	double pixel_ylength = (ymax-ymin)/(y_N-1);
	for (j=0, y=ymin; j < y_N; j++, y += pixel_ylength) {
		for (i=0, x=xmin; i < x_N; i++, x += pixel_xlength) {
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
			find_sourcept(corner_pts[i][j],corner_sourcepts[i][j],0,reference_zfactors,default_zsrc_beta_factors);
		}
	}
	ofstream outfile(filename.c_str());
	for (j=0, y=ymin; j < y_N-1; j++) {
		for (i=0, x=xmin; i < x_N-1; i++) {
			outfile << corner_sourcepts[i][j][0] << " " << corner_sourcepts[i][j][1] << endl;
			outfile << corner_sourcepts[i+1][j][0] << " " << corner_sourcepts[i+1][j][1] << endl;
			outfile << corner_sourcepts[i+1][j+1][0] << " " << corner_sourcepts[i+1][j+1][1] << endl;
			outfile << corner_sourcepts[i][j+1][0] << " " << corner_sourcepts[i][j+1][1] << endl;
			outfile << corner_sourcepts[i][j][0] << " " << corner_sourcepts[i][j][1] << endl;
			outfile << endl;
		}
	}
	for (i=0; i < x_N; i++) {
		delete[] corner_pts[i];
		delete[] corner_sourcepts[i];
	}
	delete[] corner_pts;
	delete[] corner_sourcepts;
}

void QLens::plot_logkappa_map(const int x_N, const int y_N, const string filename, const bool ignore_mask)
{
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = filename + ".x";
	string y_filename = filename + ".y";
	ofstream pixel_xvals(x_filename.c_str());
	ofstream pixel_yvals(y_filename.c_str());
	int i,j;
	for (i=0, x=xmin; i <= x_N; i++, x += xstep) {
		pixel_xvals << x << endl;
	}
	for (j=0, y=ymin; j <= y_N; j++, y += ystep) {
		pixel_yvals << y << endl;
	}
	pixel_xvals.close();
	pixel_yvals.close();

	string logkapname = filename + ".kappalog";
	ofstream logkapout(logkapname.c_str());

	double kap, mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	bool negkap = false; // Pseudo-elliptical models can produce negative kappa, so produce a warning if so
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			if ((!ignore_mask) and (image_pixel_data != NULL) and (!image_pixel_data->inside_mask(x,y))) logkapout << "NaN ";
			else {
				kap = kappa(pos,reference_zfactors,default_zsrc_beta_factors);
				//kap = kappa_exclude(pos,0,reference_zfactors,default_zsrc_beta_factors); // for looking at convergence of perturber
				if (kap < 0) {
					negkap = true;
					kap = abs(kap);
				}
				logkapout << log(kap)/log(10) << " ";
			}
		}
		logkapout << endl;
	}
	if (negkap==true) warn("kappa has negative values in some locations; plotting abs(kappa)");
}

void QLens::plot_logpot_map(const int x_N, const int y_N, const string filename)
{
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = filename + ".x";
	string y_filename = filename + ".y";
	ofstream pixel_xvals(x_filename.c_str());
	ofstream pixel_yvals(y_filename.c_str());
	int i,j;
	for (i=0, x=xmin; i <= x_N; i++, x += xstep) {
		pixel_xvals << x << endl;
	}
	for (j=0, y=ymin; j <= y_N; j++, y += ystep) {
		pixel_yvals << y << endl;
	}
	pixel_xvals.close();
	pixel_yvals.close();

	string logpotname = filename + ".potlog";
	ofstream logpotout(logpotname.c_str());

	double mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			pot = potential(pos,reference_zfactors,default_zsrc_beta_factors);
			logpotout << log(abs(pot))/log(10) << " ";
		}
		logpotout << endl;
	}
}

void QLens::plot_logmag_map(const int x_N, const int y_N, const string filename)
{
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = filename + ".x";
	string y_filename = filename + ".y";
	ofstream pixel_xvals(x_filename.c_str());
	ofstream pixel_yvals(y_filename.c_str());
	int i,j;
	for (i=0, x=xmin; i <= x_N; i++, x += xstep) {
		pixel_xvals << x << endl;
	}
	for (j=0, y=ymin; j <= y_N; j++, y += ystep) {
		pixel_yvals << y << endl;
	}
	pixel_xvals.close();
	pixel_yvals.close();

	string logmagname = filename + ".maglog";
	ofstream logmagout(logmagname.c_str());

	double mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			mag = magnification(pos,0,reference_zfactors,default_zsrc_beta_factors);
			logmagout << log(abs(mag))/log(10) << " ";
		}
		logmagout << endl;
	}
}

void QLens::plot_lensinfo_maps(string file_root, const int x_N, const int y_N, const int pert_residual)
{
	if (fit_output_dir != ".") create_output_directory();
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = fit_output_dir + "/" + file_root + ".x";
	string y_filename = fit_output_dir + "/" + file_root + ".y";
	ofstream pixel_xvals(x_filename.c_str());
	ofstream pixel_yvals(y_filename.c_str());
	int i,j;
	for (i=0, x=xmin; i <= x_N; i++, x += xstep) {
		pixel_xvals << x << endl;
	}
	for (j=0, y=ymin; j <= y_N; j++, y += ystep) {
		pixel_yvals << y << endl;
	}
	pixel_xvals.close();
	pixel_yvals.close();

	string kapname = fit_output_dir + "/" + file_root + ".kappa";
	string magname = fit_output_dir + "/" + file_root + ".mag";
	string invmagname = fit_output_dir + "/" + file_root + ".invmag";
	string shearname = fit_output_dir + "/" + file_root + ".shear";
	//string potname = fit_output_dir + "/" + file_root + ".pot";
	string defxname = fit_output_dir + "/" + file_root + ".defx";
	string defyname = fit_output_dir + "/" + file_root + ".defy";

	string logkapname = kapname + "log";
	string logmagname = magname + "log";
	string logshearname = shearname + "log";
	ofstream kapout(kapname.c_str());
	ofstream magout(magname.c_str());
	ofstream invmagout(invmagname.c_str());
	ofstream shearout(shearname.c_str());
	//ofstream potout(potname.c_str());
	ofstream defxout(defxname.c_str());
	ofstream defyout(defyname.c_str());
	ofstream logkapout(logkapname.c_str());
	ofstream logmagout(logmagname.c_str());
	ofstream logshearout(logshearname.c_str());

	bool *exclude = new bool[nlens];
	for (int i=0; i < nlens; i++) exclude[i] = false;
	if (pert_residual >= 0) {
		for (int i=0; i < nlens; i++) {
			if (i==pert_residual) exclude[i] = true;
		}
	}

	double kap, mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			kap = kappa(pos,reference_zfactors,default_zsrc_beta_factors);
			mag = magnification(pos,0,reference_zfactors,default_zsrc_beta_factors);
			invmag = inverse_magnification(pos,0,reference_zfactors,default_zsrc_beta_factors);
			shearval = shear(pos,0,reference_zfactors,default_zsrc_beta_factors);
			//pot = lens->potential(pos);
			deflection(pos,alpha,reference_zfactors,default_zsrc_beta_factors);
			if (pert_residual >= 0) {
				kap -= kappa_exclude(pos,exclude,reference_zfactors,default_zsrc_beta_factors);
				mag -= magnification_exclude(pos,exclude,0,reference_zfactors,default_zsrc_beta_factors);
				invmag -= inverse_magnification_exclude(pos,exclude,0,reference_zfactors,default_zsrc_beta_factors);
				shearval -= shear_exclude(pos,exclude,0,reference_zfactors,default_zsrc_beta_factors);
				//pot = lens->potential(pos);
				lensvector alpha_r;
				deflection_exclude(pos,exclude,alpha_r,reference_zfactors,default_zsrc_beta_factors);
				alpha[0] -= alpha_r[0];
				alpha[1] -= alpha_r[1];
			}

			kapout << kap << " ";
			magout << mag << " ";
			invmagout << invmag << " ";
			shearout << shearval << " ";
			//potout << pot << " ";
			defxout << alpha[0] << " ";
			defyout << alpha[1] << " ";
			if (kap==0) logkapout << "NaN ";
			else logkapout << log(kap)/log(10) << " ";
			if (mag==0) logmagout << "NaN ";
			else logmagout << log(abs(mag))/log(10) << " ";
			if (shearval==0) logshearout << "NaN ";
			else logshearout << log(abs(shearval))/log(10) << " ";
		}
		kapout << endl;
		invmagout << endl;
		shearout << endl;
		//potout << endl;
		defxout << endl;
		defyout << endl;
		logkapout << endl;
		logmagout << endl;
		logshearout << endl;
	}
	delete[] exclude;
}

// Pixel grid functions

void QLens::fit_los_despali()
{
	double mvir, z, logm, zstep;
	int i,nz = 10;
	double zmin = 0.05, zmax = 1.0;
	zstep = (zmax-zmin)/nz;
	//ofstream mzout("despali.dat");
	double lm, lowlm, hilm, defresid;
	double c;
	lowlm = 5.0;
	hilm = 10.0;
	int nlm = 140;
	int nx=512, ny=512;
	double lmstep = (hilm-lowlm)/(nlm-1);
	ofstream desout("despali_lm.dat");
	int j;
	double min_defresid;
	double lm_mindef;
	for (i=0, z=zmin; i < nz; i++, z += zstep) {
		min_defresid = 1e30;
		for (j=0, lm=lowlm; j < nlm; j++, lm += lmstep) {
			mvir = pow(10.0,lm);
			defresid = average_def_residual(nx,ny,nlens-1,z,mvir);
			if (defresid < min_defresid) {
				min_defresid = defresid;
				lm_mindef = lm;
			}
		}
		desout << z << " " << lm_mindef << " " << min_defresid << endl;
	}
}

double QLens::average_def_residual(const int x_N, const int y_N, const int perturber_lensnum, double z, double mvir)
{
	double mvir0,z0; // just to save original values
	if (lens_list[perturber_lensnum]->get_specific_parameter("mvir",mvir0)==false) die("could not find mvir parameter");
	if (lens_list[perturber_lensnum]->get_specific_parameter("z",z0)==false) die("could not find z parameter");

	double xc,yc,xl,yl;
	if (lens_list[perturber_lensnum]->get_specific_parameter("xc_l",xc)==false) die("could not find xc_l parameter");
	if (lens_list[perturber_lensnum]->get_specific_parameter("yc_l",yc)==false) die("could not find yc_l parameter");
	xl = sqrt(5);
	yl = sqrt(5);
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = xc-0.5*xl; xmax = xc+0.5*xl;
	ymin = yc-0.5*yl; ymax = yc+0.5*yl;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	int i,j;

	double **al_los_x = new double*[x_N];
	double **al_los_y = new double*[x_N];
	double **al_sub_x= new double*[x_N];
	double **al_sub_y= new double*[x_N];
	for (i=0; i < x_N; i++) {
		al_los_x[i] = new double[y_N];
		al_los_y[i] = new double[y_N];
		al_sub_x[i] = new double[y_N];
		al_sub_y[i] = new double[y_N];
	}

	bool *exclude = new bool[nlens];
	for (int i=0; i < nlens; i++) exclude[i] = false;
	for (int i=0; i < nlens; i++) {
		if (i==perturber_lensnum) exclude[i] = true;
	}

	lensvector alpha, alpha_unperturbed;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			deflection(pos,alpha,reference_zfactors,default_zsrc_beta_factors);
			//custom_deflection(pos[0],pos[1],alpha);
			al_sub_x[i][j] = alpha[0];
			al_sub_y[i][j] = alpha[1];
		}
	}
	if (lens_list[perturber_lensnum]->update_specific_parameter("z",z)==false) die("could not find parameter");
	if (lens_list[perturber_lensnum]->update_specific_parameter("mvir",mvir)==false) die("could not find parameter");

	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			deflection(pos,alpha,reference_zfactors,default_zsrc_beta_factors);
			//custom_deflection(pos[0],pos[1],alpha);
			al_los_x[i][j] = alpha[0];
			al_los_y[i][j] = alpha[1];
		}
	}

	double avg_def_resid = 0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			//cout << al_los_x[i][j] << " " << al_sub_x[i][j] << endl;
			avg_def_resid += SQR(al_los_x[i][j] - al_sub_x[i][j]) + SQR(al_los_y[i][j] - al_sub_y[i][j]);
		}
	}
	avg_def_resid /= (x_N*y_N);
	avg_def_resid = sqrt(avg_def_resid);

	if (lens_list[perturber_lensnum]->update_specific_parameter("z",z0)==false) die("could not find parameter");
	if (lens_list[perturber_lensnum]->update_specific_parameter("mvir",mvir0)==false) die("could not find parameter");

	delete[] exclude;
	for (i=0; i < x_N; i++) {
		delete[] al_los_x[i];
		delete[] al_los_y[i];
		delete[] al_sub_x[i];
		delete[] al_sub_y[i];
	}
	delete[] al_los_x;
	delete[] al_los_y;
	delete[] al_sub_x;
	delete[] al_sub_y;
	return avg_def_resid;
}

void QLens::find_optimal_sourcegrid_for_analytic_source()
{
	if (n_sb==0) { warn("no source objects have been specified"); return; }
	sb_list[0]->window_params(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax);
	if (n_sb > 1) {
		double xmin, xmax, ymin, ymax;
		for (int i=1; i < n_sb; i++) {
			if (!sb_list[i]->is_lensed) continue;
			sb_list[i]->window_params(xmin,xmax,ymin,ymax);
			if (xmin < sourcegrid_xmin) {
				if (xmin > sourcegrid_limit_xmin) sourcegrid_xmin = xmin;
				else sourcegrid_xmin = sourcegrid_limit_xmin;
			}
			if (xmax > sourcegrid_xmax) {
				if (xmax < sourcegrid_limit_xmax) sourcegrid_xmax = xmax;
				else sourcegrid_xmax = sourcegrid_limit_xmax;
			}
			if (ymin < sourcegrid_ymin) {
				if (ymin > sourcegrid_limit_ymin) sourcegrid_ymin = ymin;
				else sourcegrid_ymin = sourcegrid_limit_ymin;
			}
			if (ymax > sourcegrid_ymax) {
				if (ymax < sourcegrid_limit_ymax) sourcegrid_ymax = ymax;
				else sourcegrid_ymax = sourcegrid_limit_ymax;
			}
		}
	}
}

bool QLens::create_source_surface_brightness_grid(bool verbal, bool image_grid_already_exists)
{
	bool use_image_pixelgrid = false;
	if ((adaptive_grid) and (nlens==0)) { cerr << "Error: cannot ray trace source for adaptive grid; no lens model has been specified\n"; return false; }
	if ((adaptive_grid) or (((auto_sourcegrid) or (auto_srcgrid_npixels)) and (islens()))) use_image_pixelgrid = true;
	if (n_sb==0) { warn("no source objects have been specified"); return false; }

	if (use_image_pixelgrid) {
		if (!image_grid_already_exists) {
			double xmin,xmax,ymin,ymax;
			xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			xmax += 1e-10;
			ymax += 1e-10;
			if (image_pixel_grid != NULL) delete image_pixel_grid;
			image_pixel_grid = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y);
		}

		int n_imgpixels;
		if (auto_sourcegrid) find_optimal_sourcegrid_for_analytic_source();
		if (auto_srcgrid_npixels) {
			if (auto_srcgrid_set_pixel_size) // this option doesn't work well, DON'T USE RIGHT NOW
				image_pixel_grid->find_optimal_firstlevel_sourcegrid_npixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_imgpixels);
			else
				image_pixel_grid->find_optimal_sourcegrid_npixels(pixel_fraction,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_imgpixels);
			if ((verbal) and (mpi_id==0)) {
				cout << "Optimal sourcegrid number of pixels: " << srcgrid_npixels_x << " " << srcgrid_npixels_y << endl;
				cout << "Sourcegrid dimensions: " << sourcegrid_xmin << " " << sourcegrid_xmax << " " << sourcegrid_ymin << " " << sourcegrid_ymax << endl;
				cout << "Number of active image pixels expected: " << n_imgpixels << endl;
			}
		}

		if ((srcgrid_npixels_x < 2) or (srcgrid_npixels_y < 2)) {
			warn("too few source pixels for ray tracing");
			if (!image_grid_already_exists) {
				delete image_pixel_grid;
				image_pixel_grid = NULL;
			}
			return false;
		}
	}

	if (auto_sourcegrid) {
		if ((srcgrid_xshift != 0) or (srcgrid_yshift != 0)) {
			double srcpixel_xlength, srcpixel_ylength;
			srcpixel_xlength = (sourcegrid_xmax-sourcegrid_xmin) / srcgrid_npixels_x;
			srcpixel_ylength = (sourcegrid_ymax-sourcegrid_ymin) / srcgrid_npixels_y;
			if (srcgrid_xshift != 0) {
				sourcegrid_xmax += srcgrid_xshift * srcpixel_xlength;
				sourcegrid_xmin += srcgrid_xshift * srcpixel_xlength;
			}
			if (srcgrid_yshift != 0) {
				sourcegrid_ymax += srcgrid_yshift * srcpixel_ylength;
				sourcegrid_ymin += srcgrid_yshift * srcpixel_ylength;
			}
		}
		if (srcgrid_size_scale != 0) {
			double xwidth_adj = srcgrid_size_scale*(sourcegrid_xmax-sourcegrid_xmin);
			double ywidth_adj = srcgrid_size_scale*(sourcegrid_ymax-sourcegrid_ymin);
			sourcegrid_xmin -= xwidth_adj/2;
			sourcegrid_xmax += xwidth_adj/2;
			sourcegrid_ymin -= ywidth_adj/2;
			sourcegrid_ymax += ywidth_adj/2;
		}
	}

	SourcePixelGrid::set_splitting(srcgrid_npixels_x,srcgrid_npixels_y,1e-6);
	if (source_pixel_grid != NULL) delete source_pixel_grid;
	source_pixel_grid = new SourcePixelGrid(this,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax);
	if (use_image_pixelgrid) source_pixel_grid->set_image_pixel_grid(image_pixel_grid);
	if ((mpi_id==0) and (verbal)) {
		cout << "# of source pixels: " << source_pixel_grid->number_of_pixels << endl;
	}
	if (adaptive_grid) {
		source_pixel_grid->adaptive_subgrid();
		if ((mpi_id==0) and (verbal)) {
			cout << "# of source pixels after subgridding: " << source_pixel_grid->number_of_pixels << endl;
		}
	}
	source_pixel_grid->assign_surface_brightness();
	if ((use_image_pixelgrid) and (!image_grid_already_exists)) {
		delete image_pixel_grid;
		image_pixel_grid = NULL;
	}
	return true;
}

void QLens::load_source_surface_brightness_grid(string source_inputfile)
{
	if (source_pixel_grid != NULL) delete source_pixel_grid;
	source_pixel_grid = new SourcePixelGrid(this,source_inputfile,1e-6);
}

bool QLens::load_image_surface_brightness_grid(string image_pixel_filename_root)
{
	bool first_data_img = false;
	if (image_pixel_data==NULL) {
		image_pixel_data = new ImagePixelData();
		image_pixel_data->set_lens(this);
		first_data_img = true;
	}
	bool status = true;
	if (fits_format == true) {
		if (data_pixel_size <= 0) { // in this case no pixel scale has been specified, so we simply use the grid that has already been chosen
			double xmin,xmax,ymin,ymax;
			xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			status = image_pixel_data->load_data_fits(xmin,xmax,ymin,ymax,image_pixel_filename_root); // these functions are defined in the header pixelgrid.h
		} else {
			status = image_pixel_data->load_data_fits(data_pixel_size,image_pixel_filename_root); // these functions are defined in the header pixelgrid.h
		}
		// the pixel size may have been specified in the FITS file, in which case data_pixel_size was just set to something > 0
		if ((status==true) and (data_pixel_size > 0)) {
			double xmin,xmax,ymin,ymax;
			int npx, npy;
			image_pixel_data->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
			grid_xlength = xmax-xmin;
			grid_ylength = ymax-ymin;
			set_gridcenter(0.5*(xmin+xmax),0.5*(ymin+ymax));
		}
	} else {
		image_pixel_data->load_data(image_pixel_filename_root);
		double xmin,xmax,ymin,ymax;
		int npx, npy;
		image_pixel_data->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
		grid_xlength = xmax-xmin;
		grid_ylength = ymax-ymin;
		set_gridcenter(0.5*(xmin+xmax),0.5*(ymin+ymax));
	}
	if (status==false) {
		if (first_data_img) {
			delete image_pixel_data;
			image_pixel_data = NULL;
		}
		return false;
	}
	image_pixel_data->get_npixels(n_image_pixels_x,n_image_pixels_y);
	if (image_pixel_grid != NULL) {
		delete image_pixel_grid; // so when you invert, it will load a new image grid based on the data
		// This should be changed! There should be a separate image_pixel_grid for the data, vs. lensed images. That way, you don't have to do this!
		image_pixel_grid = NULL;
	}
	// Make sure the grid size & center are fixed now
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
	return true;
}

bool QLens::plot_lensed_surface_brightness(string imagefile, const int reduce_factor, bool output_fits, bool plot_residual, bool plot_foreground_only, bool show_mask_only, bool offload_to_data, bool show_extended_mask, bool show_noise_thresh, bool verbose)
{
	if ((source_fit_mode==Pixellated_Source) and (source_pixel_grid==NULL)) { warn("No source surface brightness map has been generated"); return false; }
	else if (n_sb==0) { warn("No surface brightness profiles have been defined"); return false; }
	if ((plot_residual==true) and (image_pixel_data==NULL)) { warn("cannot plot residual image, no pixel data image has been loaded"); return false; }
	double xmin,xmax,ymin,ymax;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xmax += 1e-10; // is this still necessary? Check
	ymax += 1e-10;
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y);
	if (image_pixel_data != NULL) image_pixel_grid->set_fit_window((*image_pixel_data)); 
	if (active_image_pixel_i != NULL) {
		delete[] active_image_pixel_i;
		delete[] active_image_pixel_j;
		active_image_pixel_i = NULL;
		active_image_pixel_j = NULL;
	}

	if (source_fit_mode==Pixellated_Source) {
		image_pixel_grid->set_source_pixel_grid(source_pixel_grid);
		source_pixel_grid->set_image_pixel_grid(image_pixel_grid);
		if (assign_pixel_mappings(verbose)==false) return false;
	}
	image_pixel_grid->find_surface_brightness(plot_foreground_only);
	vectorize_image_pixel_surface_brightness(); // note that in this case, the image pixel vector also contains the foreground
	if (reduce_factor==1) PSF_convolution_image_pixel_vector(verbose); // if reduce factor > 1, we'll do the PSF convolution after reducing the resolution
	store_image_pixel_surface_brightness();

	clear_pixel_matrices();

	if (reduce_factor > 1) {
		// If reduce_factor > 1, we will reduce the image to a lower resolution (with pixel size *= reduce_factor)
		double **sb_old = new double*[n_image_pixels_x];
		for (int i=0; i < n_image_pixels_x; i++) {
			sb_old[i] = new double[n_image_pixels_y];
			for (int j=0; j < n_image_pixels_x; j++) {
				sb_old[i][j] = image_pixel_grid->surface_brightness[i][j];
			}
		}
		delete image_pixel_grid;
		image_pixel_grid = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,sb_old,n_image_pixels_x,n_image_pixels_y,reduce_factor,xmin,xmax,ymin,ymax);
		n_image_pixels_x /= reduce_factor;
		n_image_pixels_y /= reduce_factor;
		vectorize_image_pixel_surface_brightness();
		PSF_convolution_image_pixel_vector(verbose);
		store_image_pixel_surface_brightness();
		clear_pixel_matrices();

		for (int i=0; i < n_image_pixels_x; i++) delete[] sb_old[i];
		delete[] sb_old;
	}

	if ((sim_pixel_noise != 0) or (data_pixel_noise != 0)) {
		if (verbose) {
			double total_signal, noise;
			if (sim_pixel_noise != 0) noise = sim_pixel_noise;
			else noise = data_pixel_noise;
			double signal_to_noise = image_pixel_grid->calculate_signal_to_noise(noise,total_signal);
			if (mpi_id==0) {
				cout << "Signal-to-noise ratio = " << signal_to_noise << " (using noise=" << noise << ")" << endl;
				cout << "Total integrated signal = " << total_signal << endl;
			}
		}
		image_pixel_grid->add_pixel_noise(sim_pixel_noise);
	}

	if (output_fits==false) {
		if (mpi_id==0) image_pixel_grid->plot_surface_brightness(imagefile,plot_residual,show_mask_only,show_extended_mask,show_noise_thresh);
	} else {
		if (mpi_id==0) image_pixel_grid->output_fits_file(imagefile,plot_residual);
	}

	int i,j;
	sbmax=-1e30;
	sbmin=1e30;
	// store max sb just in case we want to set the color bar scale using it
	for (i=0; i < n_image_pixels_x; i++) {
		for (j=0; j < n_image_pixels_y; j++) {
			if (image_pixel_grid->surface_brightness[i][j] > sbmax) sbmax = image_pixel_grid->surface_brightness[i][j];
			if (image_pixel_grid->surface_brightness[i][j] < sbmin) sbmin = image_pixel_grid->surface_brightness[i][j];
		}
	}
	if (offload_to_data) {
		if (image_pixel_data != NULL) delete image_pixel_data;
		image_pixel_data = new ImagePixelData();
		image_pixel_data->set_lens(this);
		image_pixel_data->load_from_image_grid();
		double xmin,xmax,ymin,ymax;
		int npx, npy;
		image_pixel_data->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
		grid_xlength = xmax-xmin;
		grid_ylength = ymax-ymin;
		set_gridcenter(0.5*(xmin+xmax),0.5*(ymin+ymax));
		image_pixel_data->get_npixels(n_image_pixels_x,n_image_pixels_y);
		// Make sure the grid size & center are fixed now
		if (autocenter) autocenter = false;
		if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
		if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
		// The sim_pixel_noise should now become data_pixel_noise, and to be safe, set sim_pixel_noise = 0 since we will be fitting now
		data_pixel_noise = sim_pixel_noise;
	}

	delete image_pixel_grid; // so when you invert, it will load a new image grid based on the data
	image_pixel_grid = NULL;
	return true;
}

double QLens::image_pixel_chi_square()
{
	if (image_pixel_grid==NULL) { warn("No image surface brightness map has been generated"); return -1e30; }
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return -1e30; }
	int i,j;
	if (image_pixel_data->npixels_x != image_pixel_grid->x_N) die("image surface brightness map does not have same dimensions of image pixel data");
	if (image_pixel_data->npixels_y != image_pixel_grid->y_N) die("image surface brightness map does not have same dimensions of image pixel data");
	double chisq=0;
	for (i=0; i < image_pixel_data->npixels_x; i++) {
		for (j=0; j < image_pixel_data->npixels_y; j++) {
			if (image_pixel_grid->maps_to_source_pixel)
				chisq += SQR(image_pixel_grid->surface_brightness[i][j] - image_pixel_data->surface_brightness[i][j]);
		}
	}
	return chisq;
}

void QLens::update_source_amplitudes_from_shapelets()
{
	if (source_pixel_vector == NULL) die("source pixel vector has not been created");
	SB_Profile* shapelet;
	for (int i=0; i < n_sb; i++) {
		if (sb_list[i]->sbtype==SHAPELET) {
			shapelet = sb_list[i];
			break; // currently only one shapelet source supported
		}
	}
	shapelet->get_amplitudes(source_pixel_vector);
}

void QLens::find_shapelet_scaling_parameters(const bool verbal)
{
	SB_Profile* shapelet;
	for (int i=0; i < n_sb; i++) {
		if (sb_list[i]->sbtype==SHAPELET) {
			shapelet = sb_list[i];
			break; // currently only one shapelet source supported
		}
	}
	double sig,xc,yc;
	image_pixel_grid->find_optimal_shapelet_scale(sig,xc,yc,verbal);
	if (auto_shapelet_scaling) shapelet->update_specific_parameter("sigma",sig);
	if (auto_shapelet_center) {
		shapelet->update_specific_parameter("xc",xc);
		shapelet->update_specific_parameter("yc",yc);
	}
	if ((mpi_id==0) and (verbal)) cout << "auto shapelet scaling: sig=" << sig << ", xc=" << xc << ", yc=" << yc << endl;
}

int QLens::get_shapelet_nn()
{
	SB_Profile* shapelet = NULL;
	for (int i=0; i < n_sb; i++) {
		if (sb_list[i]->sbtype==SHAPELET) {
			shapelet = sb_list[i];
			break; // currently only one shapelet source supported
		}
	}
	if (shapelet==NULL) die("no shapelet object found");
	return *(shapelet->indxptr);
}


void QLens::load_pixel_grid_from_data()
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return; }
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(this, reference_zfactors, default_zsrc_beta_factors, source_fit_mode,ray_tracing_method, (*image_pixel_data));
	image_pixel_grid->set_pixel_noise(data_pixel_noise);
}

void QLens::plot_image_pixel_grid()
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return; }
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(this, reference_zfactors, default_zsrc_beta_factors, source_fit_mode,ray_tracing_method, (*image_pixel_data));
	image_pixel_grid->redo_lensing_calculations();
	image_pixel_grid->plot_grid("map",false);
}

double QLens::invert_surface_brightness_map_from_data(bool verbal)
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return -1e30; }
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(this, reference_zfactors, default_zsrc_beta_factors, source_fit_mode, ray_tracing_method, (*image_pixel_data));
	image_pixel_grid->set_pixel_noise(data_pixel_noise);
	double chisq0;
	double chisq = invert_image_surface_brightness_map(chisq0,verbal);
	if (chisq == 2e30) {
		delete image_pixel_grid;
		image_pixel_grid = NULL;
	}
	return chisq;
}

double QLens::invert_image_surface_brightness_map(double &chisq0, bool verbal)
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return -1e30; }
	if (image_pixel_grid == NULL) { warn("No image surface brightness grid has been loaded"); return -1e30; }

	if (subhalo_prior) {
		int i, i_primary;
		double largest_einstein_radius, xch, ych;
		lensvector *centers = new lensvector[nlens];
		double *einstein_radii = new double[nlens];
		find_effective_lens_centers_and_einstein_radii(centers,einstein_radii,i_primary,reference_zfactors,default_zsrc_beta_factors,false);
		xch = centers[i_primary][0];
		xch = centers[i_primary][1];
		largest_einstein_radius = einstein_radii[i_primary];

		double xc,yc;
		for (i=0; i < nlens; i++) {
			if (i==i_primary) continue;
			if (lens_list[i]->lenstype==SHEAR) continue; // external shear cannot count as a subhalo
			xc = centers[i][0];
			yc = centers[i][1];
			if ((xc==xch) and (yc==ych)) continue; // lenses co-centered with primary lens not counted as subhalos
			if ((einstein_radii[i] > 0) and (einstein_radii[i] < perturber_einstein_radius_fraction*largest_einstein_radius)) {
				if (!image_pixel_data->test_if_in_fit_region(xc,yc)) {
					if ((mpi_id==0) and (verbal)) cout << "Subhalo outside fit region --> chisq = 2e30, will not invert image\n";
					if (logfile.is_open()) 
						logfile << "it=" << chisq_it << " chisq0=2e30" << endl;
					return 2e30;
				}
			}
		}
		delete[] einstein_radii;
		delete[] centers;
	}

	if ((mpi_id==0) and (verbal)) cout << "Number of data pixels in mask: " << image_pixel_data->n_required_pixels << endl;
	if (pixel_fraction <= 0) die("pixel fraction cannot be less than or equal to zero");
#ifdef USE_OPENMP
	double tot_wtime0, tot_wtime;
	if (show_wtime) {
		tot_wtime0 = omp_get_wtime();
	}
#endif
	bool at_least_one_zoom_lensed_src = false;
	for (int k=0; k < n_sb; k++) {
		if (sb_list[k]->is_lensed) {
			if (sb_list[k]->zoom_subgridding) at_least_one_zoom_lensed_src = true;
		}
	}

	if ((source_fit_mode==Pixellated_Source) or (source_fit_mode==Shapelet_Source) or (n_image_prior)) image_pixel_grid->redo_lensing_calculations();
	else if (at_least_one_zoom_lensed_src) image_pixel_grid->redo_lensing_calculations_corners();

	int i,j;
	double chisq = 0;
	if (source_fit_mode == Pixellated_Source) {
		if (auto_sourcegrid) image_pixel_grid->find_optimal_sourcegrid(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,sourcegrid_limit_xmin,sourcegrid_limit_xmax,sourcegrid_limit_ymin,sourcegrid_limit_ymax);
		int n_expected_imgpixels;
		if (auto_srcgrid_npixels) {
			if (auto_srcgrid_set_pixel_size)
				image_pixel_grid->find_optimal_firstlevel_sourcegrid_npixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_expected_imgpixels);
			else
				image_pixel_grid->find_optimal_sourcegrid_npixels(pixel_fraction,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_expected_imgpixels);
				srcgrid_npixels_x *= 2; // aim high, since many of the source grid pixels may lie outside the mask (we'll refine the # of pixels after drawing the grid once)
				srcgrid_npixels_y *= 2;
		}
		if ((srcgrid_npixels_x < 2) or (srcgrid_npixels_y < 2)) {
			if ((mpi_id==0) and (verbal)) cout << "Source grid has negligible size...cannot invert image\n";
			if (logfile.is_open()) 
				logfile << "it=" << chisq_it << " chisq0=2e30" << endl;
			return 2e30;
		}
		if (auto_sourcegrid) {
			if ((srcgrid_xshift != 0) or (srcgrid_yshift != 0)) {
				double srcpixel_xlength, srcpixel_ylength;
				srcpixel_xlength = (sourcegrid_xmax-sourcegrid_xmin) / srcgrid_npixels_x;
				srcpixel_ylength = (sourcegrid_ymax-sourcegrid_ymin) / srcgrid_npixels_y;
				if (srcgrid_xshift != 0) {
					sourcegrid_xmax += srcgrid_xshift * srcpixel_xlength;
					sourcegrid_xmin += srcgrid_xshift * srcpixel_xlength;
				}
				if (srcgrid_yshift != 0) {
					sourcegrid_ymax += srcgrid_yshift * srcpixel_ylength;
					sourcegrid_ymin += srcgrid_yshift * srcpixel_ylength;
				}
			}
			if (srcgrid_size_scale != 0) {
				double xwidth_adj = srcgrid_size_scale*(sourcegrid_xmax-sourcegrid_xmin);
				double ywidth_adj = srcgrid_size_scale*(sourcegrid_ymax-sourcegrid_ymin);
				sourcegrid_xmin -= xwidth_adj/2;
				sourcegrid_xmax += xwidth_adj/2;
				sourcegrid_ymin -= ywidth_adj/2;
				sourcegrid_ymax += ywidth_adj/2;
			}
		}
		SourcePixelGrid::set_splitting(srcgrid_npixels_x,srcgrid_npixels_y,1e-6);
		if (source_pixel_grid != NULL) delete source_pixel_grid;
#ifdef USE_OPENMP
		double srcgrid_wtime0, srcgrid_wtime;
		if (show_wtime) {
			srcgrid_wtime0 = omp_get_wtime();
		}
#endif
		source_pixel_grid = new SourcePixelGrid(this,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax);
		image_pixel_grid->set_source_pixel_grid(source_pixel_grid);
		source_pixel_grid->set_image_pixel_grid(image_pixel_grid);

		bool verbal_now = verbal;
		if (auto_srcgrid_npixels) verbal_now = false; // since we're gonna redo the sourcegrid a second time, don't show the first one (it would confuse people)
		if ((auto_srcgrid_npixels) or (n_image_prior)) source_pixel_grid->calculate_pixel_magnifications();
		if (auto_srcgrid_npixels) {
			if ((mpi_id==0) and (verbal_now)) cout << "Assigning pixel mappings...\n";
			if (assign_pixel_mappings(verbal_now)==false) {
				return 2e30;
			}
			double aspect_ratio = (sourcegrid_xmax-sourcegrid_xmin)/(sourcegrid_ymax-sourcegrid_ymin);

			double srcgrid_area_covered_frac = high_sn_srcgrid_overlap_area / ((sourcegrid_xmax-sourcegrid_xmin)*(sourcegrid_ymax-sourcegrid_ymin));
			srcgrid_npixels_x = (int) sqrt(pixel_fraction*image_pixel_grid->n_high_sn_pixels/srcgrid_area_covered_frac*aspect_ratio);
			srcgrid_npixels_y = (int) srcgrid_npixels_x/aspect_ratio;

			if (srcgrid_npixels_x < 3) srcgrid_npixels_x = 3;
			if (srcgrid_npixels_y < 3) srcgrid_npixels_y = 3;

			delete source_pixel_grid;
			SourcePixelGrid::set_splitting(srcgrid_npixels_x,srcgrid_npixels_y,1e-6);
			source_pixel_grid = new SourcePixelGrid(this,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax);
			image_pixel_grid->set_source_pixel_grid(source_pixel_grid);
			source_pixel_grid->set_image_pixel_grid(image_pixel_grid);
			if ((mpi_id==0) and (verbal)) {
				cout << "Optimal sourcegrid number of firstlevel pixels (active + inactive): " << srcgrid_npixels_x << " " << srcgrid_npixels_y << endl;
				cout << "Sourcegrid dimensions: " << sourcegrid_xmin << " " << sourcegrid_xmax << " " << sourcegrid_ymin << " " << sourcegrid_ymax << endl;
			}
		}
		if (adaptive_grid) {
			source_pixel_grid->adaptive_subgrid();
			if ((mpi_id==0) and (verbal)) {
				cout << "# of source pixels after subgridding: " << source_pixel_grid->number_of_pixels;
				if (auto_srcgrid_npixels) {
					double pix_frac = ((double) source_pixel_grid->number_of_pixels) / n_expected_imgpixels;
					cout << ", f=" << pix_frac;
				}
				cout << endl;
			}
		} else {
			if (n_image_prior) source_pixel_grid->calculate_pixel_magnifications();
		}
		if ((mpi_id==0) and (verbal)) cout << "Assigning pixel mappings...\n";
		if (assign_pixel_mappings(verbal)==false) {
			return 2e30;
		}

		if ((mpi_id==0) and (verbal)) {
			cout << "Number of active image pixels: " << image_npixels << endl;
		}
		//if (mpi_id==0) cout << "****Overlap area: " << total_srcgrid_overlap_area << endl;
		//if (mpi_id==0) cout << "****High S/N Overlap area: " << high_sn_srcgrid_overlap_area << endl;
		double src_pixel_area = ((sourcegrid_xmax-sourcegrid_xmin)*(sourcegrid_ymax-sourcegrid_ymin)) / (srcgrid_npixels_x*srcgrid_npixels_y);
		double est_nmapped = total_srcgrid_overlap_area / src_pixel_area;
		double est_pixfrac = est_nmapped / image_npixels;
		if ((mpi_id==0) and (verbal)) {
			double pixfrac = ((double) source_npixels) / image_npixels;
			cout << "Actual f = " << pixfrac << endl;
			if (auto_srcgrid_npixels) {
				double high_sn_pixfrac = ((double) source_npixels*high_sn_srcgrid_overlap_area/total_srcgrid_overlap_area) / image_pixel_grid->n_high_sn_pixels;
				cout << "Actual high S/N f = " << high_sn_pixfrac << endl;
			}
		}

#ifdef USE_OPENMP
		if (show_wtime) {
			srcgrid_wtime = omp_get_wtime() - srcgrid_wtime0;
			if (mpi_id==0) cout << "Wall time for creating source pixel grid: " << srcgrid_wtime << endl;
		}
#endif

		if ((mpi_id==0) and (verbal)) cout << "Initializing pixel matrices...\n";
		initialize_pixel_matrices(verbal);
		if (regularization_method != None) create_regularization_matrix();
		if (inversion_method==DENSE) {
			if (regularization_method != None) create_regularization_matrix_dense();
			convert_Lmatrix_to_dense();
			PSF_convolution_Lmatrix_dense(verbal);
		} else {
			PSF_convolution_Lmatrix(verbal);
		}
		image_pixel_grid->fill_surface_brightness_vector(); // note that image_pixel_grid just has the data pixel values stored in it
		calculate_foreground_pixel_surface_brightness();

		if ((mpi_id==0) and (verbal)) cout << "Creating lensing matrices...\n" << flush;
		if (inversion_method==DENSE) create_lensing_matrices_from_Lmatrix_dense(verbal);
		else create_lensing_matrices_from_Lmatrix(verbal);
#ifdef USE_OPENMP
		if (show_wtime) {
			tot_wtime = omp_get_wtime() - tot_wtime0;
			if (mpi_id==0) cout << "Total wall time before F-matrix inversion: " << tot_wtime << endl;
		}
#endif

		if ((mpi_id==0) and (verbal)) cout << "Inverting lens mapping...\n" << flush;

		if (inversion_method==MUMPS) invert_lens_mapping_MUMPS(verbal);
		else if (inversion_method==UMFPACK) invert_lens_mapping_UMFPACK(verbal);
		else if (inversion_method==DENSE) invert_lens_mapping_dense(verbal);
		else invert_lens_mapping_CG_method(verbal);

		if (inversion_method==DENSE) calculate_image_pixel_surface_brightness_dense();
		else calculate_image_pixel_surface_brightness();
#ifdef USE_OPENMP
		if (show_wtime) {
			tot_wtime = omp_get_wtime() - tot_wtime0;
			if (mpi_id==0) cout << "Total wall time for F-matrix construction + inversion: " << tot_wtime << endl;
		}
#endif
	} else if (source_fit_mode == Parameterized_Source) {
		image_pixel_grid->find_surface_brightness();
		vectorize_image_pixel_surface_brightness();
		PSF_convolution_image_pixel_vector(verbal);
		store_image_pixel_surface_brightness();

		if (n_image_prior) {
			create_source_surface_brightness_grid(true,true);
			image_pixel_grid->set_source_pixel_grid(source_pixel_grid);
			source_pixel_grid->set_image_pixel_grid(image_pixel_grid);
			source_pixel_grid->calculate_pixel_magnifications();
			source_pixel_grid->find_avg_n_images();
		}
	} else {
		// Shapelet source mode
		if ((auto_shapelet_scaling) or (auto_shapelet_center)) find_shapelet_scaling_parameters(verbal);
		initialize_pixel_matrices_shapelets(verbal);
		image_pixel_grid->fill_surface_brightness_vector(); // note that image_pixel_grid just has the data pixel values stored in it
		PSF_convolution_Lmatrix_dense(verbal);
		calculate_foreground_pixel_surface_brightness();

		if (regularization_method != None) create_regularization_matrix_dense();
		if ((mpi_id==0) and (verbal)) cout << "Creating lensing matrices...\n" << flush;
		create_lensing_matrices_from_Lmatrix_dense(verbal);

		int n_data_pixels=0;
		for (i=0; i < image_pixel_data->npixels_x; i++) {
			for (j=0; j < image_pixel_data->npixels_y; j++) {
				if (image_pixel_data->require_fit[i][j]) {
					n_data_pixels++;
				}
			}
		}
		if (optimize_regparam) optimize_regularization_parameter(verbal);
#ifdef USE_OPENMP
		if (show_wtime) {
			tot_wtime = omp_get_wtime() - tot_wtime0;
			if (mpi_id==0) cout << "Total wall time before F-matrix inversion: " << tot_wtime << endl;
		}
#endif
		if ((mpi_id==0) and (verbal)) cout << "Inverting lens mapping...\n" << flush;
		invert_lens_mapping_dense(verbal);
		calculate_image_pixel_surface_brightness_dense();
#ifdef USE_OPENMP
		if (show_wtime) {
			tot_wtime = omp_get_wtime() - tot_wtime0;
			if (mpi_id==0) cout << "Total wall time for F-matrix construction + inversion: " << tot_wtime << endl;
		}
#endif
		if (n_image_prior) {
			create_source_surface_brightness_grid(false,true);
			image_pixel_grid->set_source_pixel_grid(source_pixel_grid);
			source_pixel_grid->set_image_pixel_grid(image_pixel_grid);
			source_pixel_grid->calculate_pixel_magnifications();
			source_pixel_grid->find_avg_n_images();
		}
	}

	double covariance; // right now we're using a uniform uncorrelated noise for each pixel
	if (data_pixel_noise==0) covariance = 1; // doesn't matter what covariance is, since we won't be regularizing
	else covariance = SQR(data_pixel_noise);
	double dchisq, dchisq2;
	int img_index;
	int count=0;
	int n_data_pixels=0;
	for (i=0; i < image_pixel_data->npixels_x; i++) {
		for (j=0; j < image_pixel_data->npixels_y; j++) {
			if (image_pixel_data->require_fit[i][j]) {
				n_data_pixels++;
				if (image_pixel_grid->maps_to_source_pixel[i][j]) {
					img_index = image_pixel_grid->pixel_index[i][j];
					chisq += SQR(image_surface_brightness[img_index] - image_pixel_data->surface_brightness[i][j])/covariance; // generalize to full covariance matrix later
					count++;
				} else {
					chisq += SQR(image_pixel_data->surface_brightness[i][j])/covariance;
				}
			}
		}
	}
	chisq0 = chisq;

	if (group_id==0) {
		if (logfile.is_open()) {
			logfile << "it=" << chisq_it << " chisq0=" << chisq << " chisq0_per_pixel=" << chisq/image_pixel_data->n_required_pixels << " ";
			if (vary_pixel_fraction) logfile << "F=" << ((double) source_npixels)/image_npixels << " ";
		}
	}
	if ((mpi_id==0) and (verbal)) cout << "chisq0=" << chisq << " chisq0_per_pixel=" << chisq/image_pixel_data->n_required_pixels << endl;
	double chisqreg;
	if (source_fit_mode==Pixellated_Source) {
		if ((regularization_method != None) and (vary_regularization_parameter)) {
			// NOTE: technically, you should have these terms even if you do not vary the regularization parameter, since varying
			//       the lens parameters and/or the adaptive grid changes the determinants. However, this probably will not affect
			//       the inferred lens parameters because they are not very sensitive to the regularization. Play with this later!

			double Es=0;
			for (i=0; i < source_npixels; i++) {
				Es += Rmatrix[i]*SQR(source_pixel_vector[i]);
				for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
					Es += 2 * source_pixel_vector[i] * Rmatrix[j] * source_pixel_vector[Rmatrix_index[j]]; // factor of 2 since matrix is symmetric
				}
			}
			// Es here actually differs from its usual definition by a factor of 1/2, so we do not multiply by 2 (as we would normally do for chisq = -2*log(like))
			if (regularization_parameter != 0) {
				chisqreg = chisq + regularization_parameter*Es;
				chisq += regularization_parameter*Es - source_npixels*log(regularization_parameter) - Rmatrix_log_determinant;
				//cout << "src_np=" << source_npixels << " lambda=" << regularization_parameter << " Es=" << Es << " logdet=" << Rmatrix_log_determinant << endl;
			}
			chisq += Fmatrix_log_determinant;
			if (group_id==0) {
				if (logfile.is_open()) logfile << "reg=" << regularization_parameter << " chisq_reg=" << chisq << " ";
				if (logfile.is_open()) logfile << "logdet=" << Fmatrix_log_determinant << " Rlogdet=" << Rmatrix_log_determinant << " chisq_tot=" << chisq;
			}
		}
	} else if (source_fit_mode==Shapelet_Source) {
		double Es=0;
		if (regularization_method != None) {
			for (i=0; i < source_npixels; i++) {
				Es += Rmatrix_dense[i][i] * SQR(source_pixel_vector[i]); // so far we just use normalization Rmatrix, which is just diagonal
			}
			if (regularization_parameter != 0) {
				chisqreg = chisq + regularization_parameter*Es;
				chisq += regularization_parameter*Es - source_npixels*log(regularization_parameter) - Rmatrix_log_determinant;
			}
			chisq += Fmatrix_log_determinant;
		}
		//cout << "Rmatrix_det=" << Rmatrix_log_determinant << " Fmatrix_det=" << Fmatrix_log_determinant << " Es=" << Es << " regparam=" << regularization_parameter << endl << endl << endl;
	}
	if ((mpi_id==0) and (verbal)) cout << "chisqreg=" << chisqreg << ", n_data_pixels=" << n_data_pixels << endl;
	//chisq += n_data_pixels*log(M_2PI*covariance); // not critical because the data fit window and assumed pixel noise are not varied
	// to normalize the evidence properly

	if (n_image_prior) {
		double chisq_penalty;
		if ((mpi_id==0) and (verbal)) cout << "Average number of images: " << pixel_avg_n_image << endl;
		if (pixel_avg_n_image < n_image_threshold) {
			chisq_penalty = pow(1+n_image_threshold-pixel_avg_n_image,40) - 1.0; // constructed so that penalty = 0 if the average n_image = n_image_threshold
			chisq += chisq_penalty;
			if ((mpi_id==0) and (verbal)) cout << "*NOTE: average number of images is below the prior threshold (" << pixel_avg_n_image << " vs. " << n_image_threshold << "), resulting in penalty prior (chisq_penalty=" << chisq_penalty << ")" << endl;
		}
	}

	bool sb_outside_window = false;
	if (outside_sb_prior) {
		int old_imgpixel_nsplit = default_imgpixel_nsplit;
		if (default_imgpixel_nsplit > 2) {
			default_imgpixel_nsplit = 2; // it's waaaay too slow if many splittings
			image_pixel_grid->reset_nsplit();
		}
		if (source_fit_mode==Pixellated_Source) {
			clear_lensing_matrices();
			clear_pixel_matrices();
			if (extended_mask_n_neighbors == -1) image_pixel_grid->include_all_pixels();
			else image_pixel_grid->activate_extended_mask(); 
			assign_pixel_mappings(verbal);
			initialize_pixel_matrices(verbal);
			PSF_convolution_Lmatrix(verbal);
			source_pixel_grid->fill_surface_brightness_vector();
			calculate_image_pixel_surface_brightness();
		} else if (source_fit_mode==Shapelet_Source) {
			clear_pixel_matrices();
			if (extended_mask_n_neighbors == -1) image_pixel_grid->include_all_pixels();
			else image_pixel_grid->activate_extended_mask(); 

			image_pixel_grid->find_surface_brightness();
			vectorize_image_pixel_surface_brightness();
			PSF_convolution_image_pixel_vector(verbal);
			image_pixel_grid->load_data((*image_pixel_data)); // This restores pixel data values to image_pixel_grid (used for the inversion)

			// The following works, but is slower...after awhile you can probably delete these lines
			//initialize_pixel_matrices_shapelets(verbal);
			//update_source_amplitudes_from_shapelets();
			//PSF_convolution_Lmatrix_dense(verbal);
			//calculate_image_pixel_surface_brightness_dense();
		} else if (source_fit_mode==Parameterized_Source) {
			clear_pixel_matrices();
			if (extended_mask_n_neighbors == -1) image_pixel_grid->include_all_pixels();
			else image_pixel_grid->activate_extended_mask(); 
			image_pixel_grid->find_surface_brightness();
			vectorize_image_pixel_surface_brightness();
			PSF_convolution_image_pixel_vector(verbal);
			store_image_pixel_surface_brightness();
		}
		if (default_imgpixel_nsplit != old_imgpixel_nsplit) {
			default_imgpixel_nsplit = old_imgpixel_nsplit;
			image_pixel_grid->reset_nsplit();
		}
		double max_external_sb = -1e30, max_sb = -1e30;
		for (i=0; i < image_pixel_data->npixels_x; i++) {
			for (j=0; j < image_pixel_data->npixels_y; j++) {
				if ((image_pixel_data->require_fit[i][j]) and (image_pixel_grid->maps_to_source_pixel[i][j])) {
					img_index = image_pixel_grid->pixel_index[i][j];
					if (image_surface_brightness[img_index] > max_sb) {
						 max_sb = image_surface_brightness[img_index];
					}
				}
			}
		}
		double outside_sb_threshold = dmin(outside_sb_prior_noise_frac*data_pixel_noise,outside_sb_prior_threshold*max_sb);
		if ((verbal) and (mpi_id==0)) cout << "OUTSIDE SB THRESHOLD: " << outside_sb_threshold << endl;
		for (i=0; i < image_pixel_data->npixels_x; i++) {
			for (j=0; j < image_pixel_data->npixels_y; j++) {
				if ((!image_pixel_data->require_fit[i][j]) and ((image_pixel_data->extended_mask[i][j])) and (image_pixel_grid->maps_to_source_pixel[i][j])) {
					img_index = image_pixel_grid->pixel_index[i][j];
					//cout << image_surface_brightness[img_index] << endl;
					if (abs(image_surface_brightness[img_index]) >= outside_sb_threshold) {
						if (image_surface_brightness[img_index] > max_external_sb) {
							 max_external_sb = image_surface_brightness[img_index];
						}
					}
				}
			}
		}
		//cout << "WTF? " << max_sb << " " << max_external_sb << endl;
		if (max_external_sb > 0) {
			double chisq_penalty;
			sb_outside_window = true;
			chisq_penalty = pow(1+abs((max_external_sb-outside_sb_threshold)/outside_sb_threshold),60) - 1.0;
			chisq += chisq_penalty;
			if ((mpi_id==0) and (verbal)) cout << "*NOTE: surface brightness above the prior threshold (" << max_external_sb << " vs. " << outside_sb_threshold << ") has been found outside the selected fit region, resulting in penalty prior (chisq_penalty=" << chisq_penalty << ")" << endl;
		}
		image_pixel_grid->set_fit_window((*image_pixel_data));
	}

	if ((source_fit_mode==Pixellated_Source) or (source_fit_mode==Shapelet_Source))
	{
		if ((group_id==0) and (logfile.is_open())) {
			if (sb_outside_window) logfile << " -2*log(ev)=" << chisq << " (no priors; SB produced outside window)" << endl;
			else logfile << " -2*log(ev)=" << chisq << " (no priors)" << endl;
		}
		if ((mpi_id==0) and (verbal)) {
			cout << "-2*log(ev)=" << chisq << " (without priors)" << endl;
			if ((vary_pixel_fraction) or (vary_regularization_parameter)) cout << " logdet=" << Fmatrix_log_determinant << endl;
		}
	}
	if ((mpi_id==0) and (verbal)) cout << "number of image pixels included in chisq = " << count << endl;

	chisq_it++;

	if (source_fit_mode==Pixellated_Source) clear_lensing_matrices();
	clear_pixel_matrices();
	return chisq;
}

double QLens::calculate_chisq0_from_srcgrid(double &chisq0, bool verbal)
{

	if ((source_fit_mode==Pixellated_Source) and (source_pixel_grid==NULL)) { warn("No source surface brightness map has been generated"); return false; }
	if (((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) and (n_sb==0)) { warn("No surface brightness profiles have been defined"); return false; }
	if (image_pixel_data==NULL) { warn("no pixel data image has been loaded"); return false; }
	double xmin,xmax,ymin,ymax;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xmax += 1e-10; // is this still necessary? Check
	ymax += 1e-10;
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y);
	if (image_pixel_data != NULL) image_pixel_grid->set_fit_window((*image_pixel_data)); 
	//if (image_pixel_data != NULL) image_pixel_grid->set_neighbor_pixels(5); 
	if (active_image_pixel_i != NULL) {
		delete[] active_image_pixel_i;
		delete[] active_image_pixel_j;
		active_image_pixel_i = NULL;
		active_image_pixel_j = NULL;
	}

	if (source_fit_mode==Pixellated_Source) {
		image_pixel_grid->set_source_pixel_grid(source_pixel_grid);
		source_pixel_grid->set_image_pixel_grid(image_pixel_grid);
		if (assign_pixel_mappings(verbal)==false) return false;
	}
	image_pixel_grid->find_surface_brightness();
	vectorize_image_pixel_surface_brightness();
	PSF_convolution_image_pixel_vector(verbal); // if reduce factor > 1, we'll do the PSF convolution after reducing the resolution
	store_image_pixel_surface_brightness();

	double covariance; // right now we're using a uniform uncorrelated noise for each pixel
	if (data_pixel_noise==0) covariance = 1; // doesn't matter what covariance is, since we won't be regularizing
	else covariance = SQR(data_pixel_noise);
	double chisq = 0;
	int img_index;
	int i,j;
	for (i=0; i < image_pixel_data->npixels_x; i++) {
		for (j=0; j < image_pixel_data->npixels_y; j++) {
			if (image_pixel_data->require_fit[i][j]) {
				if (image_pixel_grid->maps_to_source_pixel[i][j]) {
					img_index = image_pixel_grid->pixel_index[i][j];
					chisq += SQR(image_surface_brightness[img_index] - image_pixel_data->surface_brightness[i][j])/covariance; // generalize to full covariance matrix later
				} else {
					chisq += SQR(image_pixel_data->surface_brightness[i][j])/covariance;
				}
			}
		}
	}
	chisq0 = chisq;
	clear_pixel_matrices();

	if (group_id==0) {
		if (logfile.is_open()) {
			logfile << "it=" << chisq_it << " chisq0=" << chisq << " chisq0_per_pixel=" << chisq/image_pixel_data->n_required_pixels << " ";
			if (vary_pixel_fraction) logfile << "F=" << ((double) source_npixels)/image_npixels << " ";
		}
	}
	if ((mpi_id==0) and (verbal)) cout << "chisq0=" << chisq << " chisq0_per_pixel=" << chisq/image_pixel_data->n_required_pixels << endl;
	return true;
}

int croot_lensnumber;

void QLens::plot_mc_curve(const int lensnumber, const double logm_min, const double logm_max, const string filename)
{
	// Uncomment below if you don't want to load lens configuration from a script first
	/*
	clear_lenses();
	LensProfile::use_ellipticity_components = true;
	Shear::use_shear_component_params = true;

	create_and_add_lens(ALPHA,1,lens_redshift,reference_source_redshift,1.3634,1.17163,0,0.0347001,-0.0100747,0.0152892,-0.00558392);
	add_shear_lens(lens_redshift,reference_source_redshift,0.0647257,-0.0575047,0.0152892,-0.00558392);
	lens_list[1]->anchor_center_to_lens(lens_list,0);
	create_and_add_lens(nfw,1,lens_redshift,reference_source_redshift,1e10,20.1015,0,0,0,0.18,-1.42,0,0,1);
	*/
	double mvir0,c0; // just to save original values
	if (lens_list[lensnumber]->get_specific_parameter("mvir",mvir0)==false) die("could not find mvir parameter");
	if (lens_list[lensnumber]->get_specific_parameter("c",c0)==false) die("could not find c parameter");

	croot_lensnumber = lensnumber;
	double rmax,rmax_true,avgsig,menc,menc_true,rmax_z,avgkap_scaled;
	if (!calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax_true,avgsig,menc_true,rmax_z,avgkap_scaled)) die("could not calculate critical curve perturbation radius");
	menc_true = mass2d_r(rmax_true,lensnumber,false);

	// overriding the above
	//rmax_true = 0.07;
	//menc_true = mass2d_r(rmax_true,lensnumber);

	rmax_true_mc = rmax_true; // for root finder
	menc_true_mc = menc_true; // for root finder

	cout << "rmax_true=" << rmax_true << " menc_true=" << menc_true << endl;

	double (Brent::*mc_eq)(const double);
	mc_eq = static_cast<double (Brent::*)(const double)> (&QLens::croot_eq);

	double mvir, c, logm, logm_step;
	int i,n_logm = 100;
	logm_step = (logm_max-logm_min)/(n_logm-1);
	double lowc, hic, lowf, hif;
	lowc = 1;
	hic = 1000000;
	ofstream mcout(filename.c_str());
	for (i=0, logm=logm_min; i < n_logm; i++, logm += logm_step) {
		mvir = pow(10,logm);
		if (lens_list[lensnumber]->update_specific_parameter("mvir",mvir)==false) die("could not find parameter");
		lowf = (this->*mc_eq)(lowc);
		hif = (this->*mc_eq)(hic);
		if ((lowf*hif) > 0.0) {
			warn("Root not bracketed for logm=%g\n",logm);
			break;
		}
		c = BrentsMethod(mc_eq, lowc, hic, 1e-5);
		calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
		mcout << logm << " " << c << " " << rmax << endl;
	}
	if (lens_list[lensnumber]->update_specific_parameter("mvir",mvir0)==false) die("could not find parameter");
	if (lens_list[lensnumber]->update_specific_parameter("c",c0)==false) die("could not find parameter");
}

double QLens::croot_eq(const double c)
{
	if (lens_list[croot_lensnumber]->update_specific_parameter("c",c)==false) die("could not find parameter");
	double avgsig, menc, rmax, avgkap_scaled, rmax_z;
	calculate_critical_curve_perturbation_radius_numerical(croot_lensnumber,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
	return (rmax - rmax_true_mc);
	//return (mass2d_r(rmax_true_mc,croot_lensnumber,false) - menc_true_mc);
}

void QLens::find_equiv_mvir(const double newc)
{
	// Uncomment below if you don't want to load lens configuration from a script first
	/*
	clear_lenses();
	LensProfile::use_ellipticity_components = true;
	Shear::use_shear_component_params = true;

	create_and_add_lens(ALPHA,1,lens_redshift,reference_source_redshift,1.3634,1.17163,0,0.0347001,-0.0100747,0.0152892,-0.00558392);
	add_shear_lens(lens_redshift,reference_source_redshift,0.0647257,-0.0575047,0.0152892,-0.00558392);
	lens_list[1]->anchor_center_to_lens(lens_list,0);
	create_and_add_lens(nfw,1,lens_redshift,reference_source_redshift,1e10,20.1015,0,0,0,0.18,-1.42,0,0,1);
	*/
	double rmax,rmax_true,avgsig,menc,menc_true,rmax_z,avgkap_scaled;
	if (!calculate_critical_curve_perturbation_radius_numerical(2,false,rmax_true,avgsig,menc_true,rmax_z,avgkap_scaled)) die("could not calculate critical curve perturbation radius");
	menc_true = mass2d_r(rmax_true,2,false);

	// overriding the above
	//rmax_true = 0.07;
	//menc_true = mass2d_r(rmax_true,2);

	rmax_true_mc = rmax_true; // for root finder
	menc_true_mc = menc_true; // for root finder

	double (Brent::*mc_eq)(const double);
	mc_eq = static_cast<double (Brent::*)(const double)> (&QLens::mroot_eq);

	if (lens_list[2]->update_specific_parameter("c",newc)==false) die("could not find parameter");
	double lognewm = BrentsMethod(mc_eq, 6, 12, 1e-5);
	double newm = pow(10,lognewm);
	cout << "rmax_true=" << rmax_true << " menc_true=" << menc_true << " mvir=" << newm << endl;

	if (lens_list[2]->update_specific_parameter("mvir",newm)==false) die("could not find parameter");
	reset_grid();
}

double QLens::mroot_eq(const double logm)
{
	double m = pow(10,logm);
	if (lens_list[2]->update_specific_parameter("mvir",m)==false) die("could not find parameter");
	return (mass2d_r(rmax_true_mc,2,false) - menc_true_mc);
}

int zroot_lensnumber;

double QLens::NFW_def_function(const double x)
{
	double xsq = x*x;
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}

//const double alpha_c = -0.105;
//const double alpha_c = -0.063;
//const double alpha_c = -0.046;
//const double gamma_c = -0.06;
const double alpha_c0 = -0.063;
const double alpha_c = -0.05;
const double gamma_c = -0.045;
const double delta_c = 0.025;
//const double alpha_c = 0;
double muroot_kprime;
double muroot_mvir0, muroot_rs0, muroot_rmax_p,muroot_z,muroot_x0,muroot_fzm,muroot_c0,muroot_z0,muroot_rs;

void QLens::plot_mz_curve(const int lensnumber, const double zmin, const double zmax, const double yslope_lowz, const double yslope_hiz, const bool keep_dr_const, const string filename)
{
	double mvir0,z0,c0,ycl0; // just to save original values
	if (lens_list[lensnumber]->get_specific_parameter("mvir",mvir0)==false) die("could not find mvir parameter");
	if (lens_list[lensnumber]->get_specific_parameter("yc_l",ycl0)==false) die("could not find ycl parameter");
	if (lens_list[lensnumber]->get_specific_parameter("z",z0)==false) die("could not find z parameter");
	if (lens_list[lensnumber]->get_specific_parameter("c",c0)==false) {
		double rs0;
		if (lens_list[lensnumber]->get_specific_parameter("rs_kpc",rs0)==false) die("could not find c or rs_kpc parameter");
		double nfwparams[10];
		lens_list[lensnumber]->get_parameters_pmode(1,nfwparams);
		c0 = nfwparams[1];
	}

	double alpha, alpha0, b0, b;
	if (lens_list[primary_lens_number]->lenstype==ALPHA) {
		double host_params[10];
		lens_list[primary_lens_number]->get_parameters(host_params);
		alpha0 = host_params[1];
		b0 = host_params[0];
	} else {
		alpha = 1.0;
		b0 = 1.3; // this is bullshit...
	}

	if (lens_list[lensnumber]->get_specific_parameter("xc_l",perturber_center[0])==false) die("could not find xc_l parameter");
	if (lens_list[lensnumber]->get_specific_parameter("yc_l",perturber_center[1])==false) die("could not find yc_l parameter");

	double host_xc, host_yc;
	lens_list[primary_lens_number]->get_center_coords(host_xc,host_yc);
	double subhalo_rc = sqrt(SQR(perturber_center[0]-host_xc)+SQR(perturber_center[1]-host_yc));

	zroot_lensnumber = lensnumber;
	double rmax,rmax_true,avgsig,menc,menc_true,rmax_z,avgkap_scaled,avgkap0,rmax_z_true,avgkap_scaled_true,avgsig_true;
	if (!calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax_true,avgsig,menc_true,rmax_z_true,avgkap_scaled_true)) die("could not calculate critical curve perturbation radius");
	cout << "avgkap_scaled_true = " << avgkap_scaled_true << endl;
	get_perturber_avgkappa_scaled(lensnumber,rmax_true,avgkap_scaled,menc,avgkap0);
	cout << "avgkap_scaled_check = " << avgkap_scaled << " " << avgkap0 << endl;
	menc_true = mass2d_r(rmax_true,lensnumber,false);
	avgsig_true = avgsig;

	lensvector rmax_loc;
	rmax_loc[0] = perturber_center[0] + rmax_true*cos(theta_shear);
	rmax_loc[1] = perturber_center[1] + rmax_true*sin(theta_shear);

	// overriding the above
	//rmax_true = 0.07;
	//menc_true = mass2d_r(rmax_true,lensnumber);

	rmax_true_mz = rmax_true; // for root finder
	menc_true_mz = menc_true; // for root finder
	rmax_z_true_mz = rmax_z_true; // for root finder
	avgkap_scaled_true_mz = avgkap_scaled_true; // for root finder

	cout << "rmax_true=" << rmax_true << " menc_true=" << menc_true << endl;
	//cout << "yslope=" << yslope << endl;

	double (Brent::*mz_eq)(const double);
	mz_eq = static_cast<double (Brent::*)(const double)> (&QLens::zroot_eq);

	double (Brent::*mu_eq)(const double);
	mu_eq = static_cast<double (Brent::*)(const double)> (&QLens::muroot_eq);

	double (Brent::*mr_eq)(const double);
	mr_eq = static_cast<double (Brent::*)(const double)> (&QLens::mrroot_eq);

	double (Brent::*mr_eq0)(const double);
	mr_eq0 = static_cast<double (Brent::*)(const double)> (&QLens::mrroot_eq0);

	double dp, dL = angular_diameter_distance(lens_redshift); // in Mpc

	double mvir, z, logm, zstep;
	int i,nz = 60;
	zstep = (zmax-zmin)/(nz-1);
	string filename_mz = filename + ".mz";
	string filename_zm = fit_output_dir + "/" + filename + ".zm";
	ofstream mzout(filename_mz.c_str());
	ofstream zmout(filename_zm.c_str());
	double lowlm, hilm, lowf, hif, fz, fzm, fz0;
	double c;
	lowlm = 6;
	hilm = 14;
	double nfwparams[10];
	double rs, rs0, x0;
	lens_list[lensnumber]->get_parameters_pmode(2,nfwparams);
	rs0 = nfwparams[1];
	cout << "rs_kpc=" << rs0 << endl;
	x0 = (rmax_true*dL/206.264806)/rs0;
	muroot_x0=x0;
	muroot_c0=c0;
	muroot_z0=z0;
		//ofstream muout("facmu.dat");
		ofstream kpout("kpout.dat");
	double ycl, rmax_y;
	double rmax_rel;
	for (i=0, z=zmin; i < nz; i++, z += zstep) {
		if (lens_list[lensnumber]->update_specific_parameter("z",z)==false) die("could not find parameter");
		//ycl = 0; 
		if (z >= 0.5) ycl = yslope_hiz*(z-0.5) + ycl0;
		else ycl = -yslope_lowz*(z-0.5) + ycl0;
		//else ycl = ycl0;
		//if (ycl > -1.4) ycl = -1.4;
		if (lens_list[lensnumber]->update_specific_parameter("yc_l",ycl)==false) die("could not find parameter");
		if (lens_list[lensnumber]->get_specific_parameter("xc_l",perturber_center[0])==false) die("could not find xc_l parameter");
		if (lens_list[lensnumber]->get_specific_parameter("yc_l",perturber_center[1])==false) die("could not find yc_l parameter");

		if (keep_dr_const) {
			rmax_true_mz = (rmax_loc[1] - perturber_center[1]) / (sin(theta_shear));
		}
		rmax_loc[0] = perturber_center[0] + rmax_true_mz*cos(theta_shear);
		rmax_loc[1] = perturber_center[1] + rmax_true_mz*sin(theta_shear);

		lowf = (this->*mz_eq)(lowlm);
		hif = (this->*mz_eq)(hilm);
		if ((lowf*hif) > 0.0) {
			warn("Root not bracketed for z=%g\n",z);
			break;
		}
		//if (z > 0.8) lowlm = 10.0;
		logm = BrentsMethod(mz_eq, lowlm, hilm, 1e-10);
		double mvir = pow(10,logm);
		if (lens_list[zroot_lensnumber]->update_specific_parameter("mvir",mvir)==false) die("could not find parameter");

		calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax_rel,avgsig,menc,rmax_z,avgkap_scaled,true);
		calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);

		double avgkap_scaled_bla, menc_scaled_bla;
		double avgkap_scaled_bla2, menc_scaled_bla2;
		double avgkap0_rmax;
		get_perturber_avgkappa_scaled(zroot_lensnumber,rmax,avgkap_scaled_bla2,menc_scaled_bla2,avgkap0_rmax,false);
		get_perturber_avgkappa_scaled(zroot_lensnumber,rmax_true_mz,avgkap_scaled_bla,menc_scaled_bla,avgkap0);
		cout << z << " " << logm << " " << rmax << " " << rmax_rel << " loc=(" << rmax_loc[0] << "," << rmax_loc[1] << ") ycl=" << ycl << " akap=" << avgkap_scaled_bla << " akap0=" << avgkap0 << " f=" << (this->*mz_eq)(logm) << endl;
		//cout << avgkap_scaled_bla << " " << avgkap_scaled_true_mz << " " << avgkap_scaled << " " << avgkap_scaled_bla2 << " " << avgkap0 << endl;
		kpout << z << " " << logm << " " << rmax << " " << rmax_true_mz << " " << avgkap_scaled_bla << " " << avgkap_scaled_true_mz << " " << ycl << endl;
		fz = calculate_sigpert_scale_factor(lens_redshift,z,source_redshift,rmax_true,alpha,subhalo_rc);
		fzm = calculate_menc_scale_factor(lens_redshift,z,source_redshift,rmax_true,alpha,subhalo_rc);
		if (lens_list[lensnumber]->get_specific_parameter("c",c)==false) {
			double rstry;
			if (lens_list[lensnumber]->get_specific_parameter("rs_kpc",rstry)==false) die("could not find c or rs_kpc parameter");
			double nfwparams[10];
			lens_list[lensnumber]->get_parameters_pmode(1,nfwparams);
			c = nfwparams[1];
		}

		lens_list[lensnumber]->get_parameters_pmode(2,nfwparams);
		rs = nfwparams[1];

		muroot_rs = rs;
		dp = angular_diameter_distance(z); // in Mpc
		double rmax_p = rmax_true*dp/206.264806;
		muroot_fzm=fzm;
		muroot_mvir0=mvir0;
		muroot_rs0=rs0;
		muroot_rmax_p=rmax_p;
		muroot_z=z;
/*

		double mu;
		double mui=0.1, muf = 1000;
		int nmu = 30000;
		double mustep = (muf-mui)/(nmu-1);
		int i;
		ofstream muout("facmu.dat");
		//for (i=0, mu=mui; i < nmu; i++, mu += mustep) {
		for (i=0, mu=mui; i < nmu; i++, mu += mustep) {
			//mu = 1000;
			//c = median_concentration_dutton(mu*mvir0,z);
			//c0 = median_concentration_dutton(mvir0,z0);
			//mu = mvir/mvir0;
			//double fac = (log(1+c) - c/(1+c)) / (log(1+c0) - c0/(1+c0));
			//double factest = pow(mu,alpha_c+gamma_c*(z-z0)+delta_c*SQR(z-z0));
			double sfac2 = (log(mu/2) + NFW_def_function(mu));
			double sfac3 = pow(mu,1-alpha_c0);
			muout << mu << " " << sfac2 << " " << sfac3 << endl;
			//muout << (z-z0) << " " << fac << " " << factest << endl;
			//muout << mu << " " << z << " " << fac << " " << factest << endl;
			//muout << mu << " " << (this->*mu_eq)(mu) << endl;
		}
		die();
*/

		muroot_kprime = fzm*pow(rmax_p/rs0,(-1+alpha_c0)/0.4333333)*(log(x0/2)+NFW_def_function(x0));
		double mu_expected = rmax_p/rs0*pow(mvir/mvir0,-0.4333333);
		double wtf = pow(mu_expected,(-1+alpha_c0)/0.43333333)*(log(mu_expected/2) + NFW_def_function(mu_expected)) - muroot_kprime;
		//double wtf = (this->*mu_eq)(mu_expected);
		//cout << "Trying for z=" << z << "... (expect mu=" << mu_expected << "; f(mu) = " << wtf << ") kprime=" << muroot_kprime << endl;
		double mu = 0.55*pow(muroot_kprime,-1.333333333);

		//double mu0 = BrentsMethod(mu_eq, 1e-3, 10, 1e-5);
		//kpout << muroot_kprime << " " << mu << " " << mu0 << endl;
		double mvir_root = mvir0*pow(mu*rs0/rmax_p,-1.0/0.43333333);
		double logm_root = log(mvir_root)/log(10);
		//cout << "m200=" << mvir_root << " actual=" << mvir << endl;
	
		//mu = BrentsMethod(mr_eq0, 1e-3, 1000000, 1e-9);
		//double mvir_check = mu*mvir0;
		//double logm_check = log(mvir_check)/log(10);

		double x_desp = (z < lens_redshift) ? (z/lens_redshift - 1) : (z > lens_redshift) ? ((z - lens_redshift) / (source_redshift - lens_redshift)) : 0;
		double logmvir_desp = (log(mvir0)/log(10)) + 0.41*x_desp + 0.57*SQR(x_desp) + 0.9 * CUBE(x_desp);
		double mvir_desp = pow(10,logmvir_desp);
		
		//mu = BrentsMethod(mr_eq, 1e-3, 1000000, 1e-9);
		//double mvir_root = mu*mvir0;
		//double logm_root = log(mvir_root)/log(10);

		//double c_check = median_concentration_dutton(mvir_root,z);
		//rs_check = pow(3*mvir_root/(4*M_PI*200*1e-9*critical_density(z)),1.0/3.0)/c_check;
		//x = (rmax_true*dp/206.264806)/rs_check;
		////cout << "RS: " << rs << " " << rs_check << endl;
		//double fac = (log(1+c_check) - c_check/(1+c_check)) / (log(1+c0) - c0/(1+c0));
		//double x = muroot_rmax_p/rs_check;
		//double sfac = (log(x/2) + NFW_def_function(x)) / (log(muroot_x0/2) + NFW_def_function(muroot_x0));
		//double fch = mu - (muroot_fzm*fac/sfac);
		//double testm = mvir0*fzm*((log(1+c_check)-c_check/(1+c_check))/(log(1+c0)-c0/(1+c0))) * ((log(x0/2)+NFW_def_function(x0))/(log(x/2)+NFW_def_function(x)));

		//mu = mvir/mvir0;
		//x = (rmax_true*dp/206.264806)/rs;
		//fac = (log(1+c) - c/(1+c)) / (log(1+c0) - c0/(1+c0));
		//x = rmax_p/rs;
		//sfac = (log(x/2) + NFW_def_function(x)) / (log(x0/2) + NFW_def_function(x0));
		//double fch2 = mu - ((menc/menc_true)*fac/sfac);
		//double mvcheck = mvir0*(menc/menc_true)*fac/sfac;
		//double mvcheck2 = mvir0*fzm*fac/sfac;

		//cout << "MVIR: " << testm << " " << mvcheck << " " << mvcheck2 << " " << mvir << " " << fch << " " << fch2 << endl;

		//double mvir = fz*pow(10,9.95018);
		//lens_list[lensnumber]->get_parameters_pmode(0,nfwparams);
		//double ks = nfwparams[0];
		//double sigcrit = sigma_crit_kpc(z,source_redshift);
		//double sigcrit0 = sigma_crit_kpc(lens_redshift,source_redshift);
		double dl = comoving_distance(lens_redshift);
		double dpert = comoving_distance(z);
		int i1,i2;
		i1 = lens_redshift_idx[primary_lens_number];
		i2 = lens_redshift_idx[lensnumber];
		double betafac;
		if (z > lens_redshift) betafac = default_zsrc_beta_factors[i2-1][i1];
		else betafac = 0;
		double dpert_tilde = dpert*(1-betafac);
		double dchidz = comoving_distance_derivative(z);
		double dndm = mass_function_ST(mvir,z);
		double dsigdlogm_dz = mvir*dndm*dpert*dpert_tilde*dchidz/SQR(206264.8);

		double dndm_desp = mass_function_ST(mvir_desp,z);
		double dsigdlogm_dz_desp = mvir_desp*dndm_desp*dpert*dpert_tilde*dchidz/SQR(206264.8);

		double dndm_mconst = mass_function_ST(mvir0,z);
		double dsigdlogm_dz_mconst = mvir0*dndm_mconst*dpert*dpert_tilde*dchidz/SQR(206264.8);

		//double ffac = 1 - beta;
		//double gfac = (z > lens_redshift) ? (1-beta) : 1;

		//double sengul_logmvir = log10((sigcrit0/sigcrit)*mvir0);

		//cout << "MVIR: " << mvir << " " << mvir_check << endl;
		//cout << "CHECK: " << fzm << " " << (fz*SQR(dp/dL)) << endl;
		//cout << "rs0=" << rs0 << " rs=" << rs << " rs2=" << rs_check << endl;
		//mzout << z << " " << logm << " " << rmax << " " << fz << " " << fzm << " " << mvir_ratio << " " << (avgsig/avgsig_true) << " " << (menc/menc_true) << " " << (mvir/mvir0) << " " << logm_check << " " << logm_root << " " << logmvir_desp << " " << dsigdlogm_dz << " " << dsigdlogm_dz_desp << endl;
		mzout << z << " " << logm << " " << rmax << " " << fz << " " << fzm << " " << (avgsig/avgsig_true) << " " << (menc/menc_true) << " " << (mvir/mvir0) << " " << logm_root << " " << logmvir_desp << " " << dsigdlogm_dz << " " << dsigdlogm_dz_desp << " " << dsigdlogm_dz_mconst << endl;
		zmout << logm << " " << z << endl;
	}
	if (lens_list[lensnumber]->update_specific_parameter("mvir",mvir0)==false) die("could not find parameter");
	if (lens_list[lensnumber]->update_specific_parameter("z",z0)==false) die("could not find parameter");
	if (lens_list[lensnumber]->update_specific_parameter("yc_l",ycl0)==false) die("could not find parameter");
}

double QLens::mrroot_eq0(const double mu)
{
	double mvir, c, c0, fac, rs, x, sfac;
	mvir = muroot_mvir0*mu;
	c = median_concentration_dutton(mvir,muroot_z);
	c0 = median_concentration_dutton(muroot_mvir0,muroot_z0);
	fac = (log(1+c) - c/(1+c)) / (log(1+c0) - c0/(1+c0));
	rs = pow(3*mvir/(4*M_PI*200*1e-9*critical_density(muroot_z)),1.0/3.0)/c;
	x = muroot_rmax_p/rs;
	sfac = (log(x/2) + NFW_def_function(x)) / (log(muroot_x0/2) + NFW_def_function(muroot_x0));
	return mu - (muroot_fzm*fac/sfac);
}

double QLens::mrroot_eq(const double mu)
{
	double mvir, c, c0, fac, rs, x, sfac;
	mvir = muroot_mvir0*mu;
	c = median_concentration_dutton(mvir,muroot_z);
	c0 = median_concentration_dutton(muroot_mvir0,muroot_z0);
	fac = (log(1+c) - c/(1+c)) / (log(1+c0) - c0/(1+c0));
	//fac = pow(mu,alpha_c+gamma_c*(muroot_z-muroot_z0)+delta_c*SQR(muroot_z-muroot_z0));
	//fac = 1 - 0.062*log(mu);
	//rs = pow(3*mvir/(4*M_PI*200*1e-9*critical_density(muroot_z)),1.0/3.0)/c;
	//rs = muroot_rs0*pow(mu,0.45333333333);
	//rs = muroot_rs0*pow(mu,0.433333333333);
	rs = muroot_rs0*pow(mu,0.43433333-0.026*muroot_z0)*pow(critical_density(muroot_z)/critical_density(muroot_z0),-1.0/3.0)*pow(10,-0.381*(exp(-0.617*pow(muroot_z,1.21)) - exp(-0.617*pow(muroot_z0,1.21))));
	//rs = muroot_rs0*pow(mu,0.43433333)*pow(critical_density(muroot_z)/critical_density(muroot_z0),-1.0/3.0)*pow(10,-0.381*(exp(-0.617*pow(muroot_z,1.21)) - exp(-0.617*pow(muroot_z0,1.21))))*pow(hubble*1e-12,-0.026*(muroot_z-muroot_z0)) * pow(mvir,-0.026*muroot_z)*pow(muroot_mvir0,0.026*muroot_z0);
	//rs = muroot_rs0*pow(mu,0.33333333)*pow(critical_density(muroot_z)/critical_density(muroot_z0),-1.0/3.0)*pow(mvir*hubble*1e-12,0.101-0.026*muroot_z)*pow(muroot_mvir0*hubble*1e-12,-0.101+0.026*muroot_z0)*pow(10,-0.381*(exp(-0.617*pow(muroot_z,1.21)) - exp(-0.617*pow(muroot_z0,1.21))));
	//rs = muroot_rs0*pow(mu,0.33333333)*pow(critical_density(muroot_z)/critical_density(muroot_z0),-1.0/3.0)*c0/c;
	x = muroot_rmax_p/rs;
	sfac = (log(x/2) + NFW_def_function(x)) / (log(muroot_x0/2) + NFW_def_function(muroot_x0));
	return mu - (muroot_fzm*fac/sfac);
}

double QLens::muroot_eq(const double mu)
{
	return pow(mu,(-1+alpha_c0)/0.43333333)*(log(mu/2) + NFW_def_function(mu)) - muroot_kprime;
	//return pow(mu,(-1+alpha_c)/0.43333333)*fac*(log(mu/2) + NFW_def_function(mu)) - muroot_kprime;
	//return pow(mu,(-1+alpha_c)/0.43333333)*fac*(log(mu2/2) + NFW_def_function(mu2)) - muroot_kprime;
}

double QLens::zroot_eq(const double logm)
{
	double mvir = pow(10,logm);
	if (lens_list[zroot_lensnumber]->update_specific_parameter("mvir",mvir)==false) die("could not find parameter");
	double avgkap_scaled, menc_scaled, avgkap0;
	get_perturber_avgkappa_scaled(zroot_lensnumber,rmax_true_mz,avgkap_scaled,menc_scaled,avgkap0);
	//cout << "WTF? " << avgkap_scaled << " " << avgkap0 << endl;
	return (avgkap_scaled - avgkap0);
	//return (avgkap_scaled - avgkap_scaled_true_mz);
	//return (mass2d_r(rmax_true_mz,zroot_lensnumber,false) - menc_true_mz);
	//double avgsig, menc, rmax, rmax_z, avgkap_scaled;
	//calculate_critical_curve_perturbation_radius_numerical(zroot_lensnumber,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
	//return (rmax - rmax_true_mz);
}

void QLens::plot_mz_bestfit(const int lensnumber, const double zmin, const double zmax, const double zstep, string filename)
{
	double mvir0,z0; // just to save original values
	if (lens_list[lensnumber]->get_specific_parameter("z",z0)==false) die("could not find z parameter");

	set_fitmethod(POWELL);
	boolvector varyflags;
	lens_list[lensnumber]->get_vary_flags(varyflags);
	int npar = varyflags.size();
	varyflags[npar-1] = false; // make sure the redshift won't be varied during fit
	set_lens_vary_parameters(lensnumber,varyflags);
	calculate_parameter_errors = false;

	double mvir, z, logm;
	string filename_mz = filename + ".mzb";
	string filename_zm = fit_output_dir + "/" + filename + ".zmb";
	ofstream mzout(filename_mz.c_str());
	ofstream zmout(filename_zm.c_str());
	z=z0;
	if (lens_list[lensnumber]->update_specific_parameter("z",z)==false) die("could not find parameter");
	chi_square_fit_powell();
	adopt_model(bestfitparams);
	dvector fitpar0(bestfitparams);
	if (lens_list[lensnumber]->get_specific_parameter("mvir",mvir0)==false) die("could not find mvir parameter");

	double rmax, rmax_rel;
	double avgsig,menc,rmax_z,avgkap_scaled; // these won't get used, but required for function for getting rmax
	for (z=z0; z > zmin; z -= zstep) {
		if (lens_list[lensnumber]->update_specific_parameter("z",z)==false) die("could not find parameter");
		chi_square_fit_powell();
		adopt_model(bestfitparams);
		if (lens_list[lensnumber]->get_specific_parameter("mvir",mvir)==false) die("could not find mvir parameter");
		logm = log10(mvir);

		calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax_rel,avgsig,menc,rmax_z,avgkap_scaled,true);
		calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);

		double x_desp = (z < lens_redshift) ? (z/lens_redshift - 1) : (z > lens_redshift) ? ((z - lens_redshift) / (source_redshift - lens_redshift)) : 0;
		double logmvir_desp = (log(mvir0)/log(10)) + 0.41*x_desp + 0.57*SQR(x_desp) + 0.9 * CUBE(x_desp);
		double mvir_desp = pow(10,logmvir_desp);
	
		double dpert = comoving_distance(z);
		int i1,i2;
		i1 = lens_redshift_idx[primary_lens_number];
		i2 = lens_redshift_idx[lensnumber];
		double betafac;
		if (z > lens_redshift) betafac = default_zsrc_beta_factors[i2-1][i1];
		else betafac = 0;
		double dpert_tilde = dpert*(1-betafac);
		double dchidz = comoving_distance_derivative(z);
		double dndm = mass_function_ST(mvir,z);
		double dsigdlogm_dz = mvir*dndm*dpert*dpert_tilde*dchidz/SQR(206264.8);

		double dndm_desp = mass_function_ST(mvir_desp,z);
		double dsigdlogm_dz_desp = mvir_desp*dndm_desp*dpert*dpert_tilde*dchidz/SQR(206264.8);

		mzout << z << " " << logm << " " << " " << rmax << " " << rmax_rel << " " << dsigdlogm_dz << " " << dsigdlogm_dz_desp << endl;
		zmout << logm << " " << z << endl;
	}

	if (lens_list[lensnumber]->update_specific_parameter("z",z0)==false) die("could not find parameter");
	adopt_model(fitpar0);
	mzout << endl;
	zmout << endl;

	for (z=z0; z < zmax; z += zstep) {
		if (lens_list[lensnumber]->update_specific_parameter("z",z)==false) die("could not find parameter");
		chi_square_fit_powell();
		adopt_model(bestfitparams);
		if (lens_list[lensnumber]->get_specific_parameter("mvir",mvir)==false) die("could not find mvir parameter");
		logm = log10(mvir);

		calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax_rel,avgsig,menc,rmax_z,avgkap_scaled,true);
		calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);

		double x_desp = (z < lens_redshift) ? (z/lens_redshift - 1) : (z > lens_redshift) ? ((z - lens_redshift) / (source_redshift - lens_redshift)) : 0;
		double logmvir_desp = (log(mvir0)/log(10)) + 0.41*x_desp + 0.57*SQR(x_desp) + 0.9 * CUBE(x_desp);
		double mvir_desp = pow(10,logmvir_desp);

		double dpert = comoving_distance(z);
		int i1,i2;
		i1 = lens_redshift_idx[primary_lens_number];
		i2 = lens_redshift_idx[lensnumber];
		double betafac;
		if (z > lens_redshift) betafac = default_zsrc_beta_factors[i2-1][i1];
		else betafac = 0;
		double dpert_tilde = dpert*(1-betafac);
		double dchidz = comoving_distance_derivative(z);
		double dndm = mass_function_ST(mvir,z);
		double dsigdlogm_dz = mvir*dndm*dpert*dpert_tilde*dchidz/SQR(206264.8);

		double dndm_desp = mass_function_ST(mvir_desp,z);
		double dsigdlogm_dz_desp = mvir_desp*dndm_desp*dpert*dpert_tilde*dchidz/SQR(206264.8);

		//double dndm_mconst = mass_function_ST(mvir0,z);
		//double dsigdlogm_dz_mconst = mvir0*dndm_mconst*dpert*dpert_tilde*dchidz/SQR(206264.8);

		//mzout << z << " " << logm << " " << " " << dsigdlogm_dz << " " << dsigdlogm_dz_desp << " " << dsigdlogm_dz_mconst << endl;
		mzout << z << " " << logm << " " << " " << rmax << " " << rmax_rel << " " << dsigdlogm_dz << " " << dsigdlogm_dz_desp << endl;
		zmout << logm << " " << z << endl;
	}

	//if (lens_list[lensnumber]->update_specific_parameter("mvir",mvir0)==false) die("could not find parameter");
	//if (lens_list[lensnumber]->update_specific_parameter("z",z0)==false) die("could not find parameter");
}

void QLens::find_bestfit_smooth_model(const int lensnumber)
{
	set_fitmethod(POWELL);
	boolvector varyflags;
	lens_list[lensnumber]->get_vary_flags(varyflags);
	int npar = varyflags.size();
	for (int i=0; i < npar; i++) varyflags[i] = false; // won't vary subhalo during fits
	set_lens_vary_parameters(lensnumber,varyflags);
	calculate_parameter_errors = false;

	double mvir_min = 1e5;
	double mvir_scalefac = 0.9;
	double mvir0, mvir;

	if (lens_list[lensnumber]->get_specific_parameter("mvir",mvir0)==false) die("could not find mvir parameter");

	double chisq = chi_square_fit_powell();
	adopt_model(bestfitparams);
	n_repeats = 10;
	double chisqprev;
	int i=0;
	do {
		cout << "Warm-up iteration " << i << ":" << endl;
		chisqprev = chisq;
		chisq = chi_square_fit_powell();
		adopt_model(bestfitparams);
		i++;
	} while (abs(chisq-chisqprev) > 0.05);

	double chisq0 = chisq;
	n_repeats = 1;
	for (mvir=mvir0; mvir > mvir_min; mvir *= mvir_scalefac) {
		if (lens_list[lensnumber]->update_specific_parameter("mvir",mvir)==false) die("could not find parameter");
		chisq = chi_square_fit_powell();
		adopt_model(bestfitparams);
	}
	double likeratio = exp(-(chisq-chisq0)/2);
	cout << "likeratio=" << likeratio << ", chisq=" << chisq << ", chisq0=" << chisq0 << endl;
}

/*
void QLens::find_perturber_surface_density(const int lensnumber, const double zmin, const double zmax)
{

	double mvir0,z0,c0; // just to save original values
	if (lens_list[lensnumber]->get_specific_parameter("mvir",mvir0)==false) die("could not find mvir parameter");
	if (lens_list[lensnumber]->get_specific_parameter("z",z0)==false) die("could not find z parameter");
	if (lens_list[lensnumber]->get_specific_parameter("c",c0)==false) {
		double rs0;
		if (lens_list[lensnumber]->get_specific_parameter("rs_kpc",rs0)==false) die("could not find c or rs_kpc parameter");
		double nfwparams[10];
		lens_list[lensnumber]->get_parameters_pmode(1,nfwparams);
		c0 = nfwparams[1];
	}

	double alpha;
	if (lens_list[primary_lens_number]->lenstype==ALPHA) {
		double host_params[10];
		lens_list[primary_lens_number]->get_parameters(host_params);
		alpha = host_params[1];
	} else {
		alpha = 1.0;
	}

	double reference_zfactor = reference_zfactors[lens_redshift_idx[perturber_lens_number]];
	if (lens_list[lensnumber]->get_specific_parameter("xc_l",perturber_center[0])==false) die("could not find xc_l parameter");
	if (lens_list[lensnumber]->get_specific_parameter("yc_l",perturber_center[1])==false) die("could not find yc_l parameter");

	zroot_lensnumber = lensnumber;
	double rmax,rmax_true,avgsig,menc,menc_true,rmax_z,avgkap_scaled,rmax_z_true,avgkap_scaled_true,avgsig_true;
	if (!calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax_true,avgsig,menc_true,rmax_z_true,avgkap_scaled_true)) die("could not calculate critical curve perturbation radius");
	cout << "avgkap_scaled_true = " << avgkap_scaled_true << endl;
	get_perturber_avgkappa_scaled(lensnumber,rmax_true,avgkap_scaled,menc);
	cout << "avgkap_scaled_check = " << avgkap_scaled << endl;
	menc_true = mass2d_r(rmax_true,lensnumber,false);
	avgsig_true = avgsig;

	rmax_true_mz = rmax_true; // for root finder
	menc_true_mz = menc_true; // for root finder
	rmax_z_true_mz = rmax_z_true; // for root finder
	avgkap_scaled_true_mz = avgkap_scaled_true; // for root finder

	cout << "rmax_true=" << rmax_true << " menc_true=" << menc_true << endl;

	double (Brent::*mz_eq)(const double);
	mz_eq = static_cast<double (Brent::*)(const double)> (&QLens::zroot_eq);

	double dp, dL = angular_diameter_distance(lens_redshift); // in Mpc

	int i;
	double lowlm, hilm, lowf, hif, fz, fzm, fz0;
	double c;
	lowlm = 6.0;
	hilm = 14.0;
	double nfwparams[10];
	double rs, rs0, x0, mvir_ratio;
	lens_list[lensnumber]->get_parameters_pmode(2,nfwparams);
	rs0 = nfwparams[1];
	cout << "rs_kpc=" << rs0 << endl;
	x0 = (rmax_true*dL/206.264806)/rs0;
	muroot_x0=x0;
	muroot_c0=c0;
	muroot_z0=z0;
		//ofstream muout("facmu.dat");
		ofstream kpout("kpout.dat");
	for (i=0, z=zmin; i < nz; i++, z += zstep) {
		if (lens_list[lensnumber]->update_specific_parameter("z",z)==false) die("could not find parameter");
		lowf = (this->*mz_eq)(lowlm);
		hif = (this->*mz_eq)(hilm);
		if ((lowf*hif) > 0.0) {
			warn("Root not bracketed for z=%g\n",z);
			break;
		}
		logm = BrentsMethod(mz_eq, lowlm, hilm, 1e-5);
		double mvir = pow(10,logm);
		calculate_critical_curve_perturbation_radius_numerical(lensnumber,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
		fz = calculate_sigpert_scale_factor(lens_redshift,z,source_redshift,rmax_true,alpha,subhalo_rc);
		fzm = calculate_menc_scale_factor(lens_redshift,z,source_redshift,rmax_true,alpha,subhalo_rc);
		if (lens_list[lensnumber]->get_specific_parameter("c",c)==false) {
			double rstry;
			if (lens_list[lensnumber]->get_specific_parameter("rs_kpc",rstry)==false) die("could not find c or rs_kpc parameter");
			double nfwparams[10];
			lens_list[lensnumber]->get_parameters_pmode(1,nfwparams);
			c = nfwparams[1];
		}

		lens_list[lensnumber]->get_parameters_pmode(2,nfwparams);
		rs = nfwparams[1];

		muroot_rs = rs;
		dp = angular_diameter_distance(z); // in Mpc
		double rmax_p = rmax_true*dp/206.264806;
		muroot_fzm=fzm;
		muroot_mvir0=mvir0;
		muroot_rs0=rs0;
		muroot_rmax_p=rmax_p;
		muroot_z=z;

		muroot_kprime = fzm*pow(rmax_p/rs0,(-1+alpha_c0)/0.4333333)*(log(x0/2)+NFW_def_function(x0));
		double mu_expected = rmax_p/rs0*pow(mvir/mvir0,-0.4333333);
		double wtf = pow(mu_expected,(-1+alpha_c0)/0.43333333)*(log(mu_expected/2) + NFW_def_function(mu_expected)) - muroot_kprime;
		//double wtf = (this->*mu_eq)(mu_expected);
		//cout << "Trying for z=" << z << "... (expect mu=" << mu_expected << "; f(mu) = " << wtf << ") kprime=" << muroot_kprime << endl;
		double mu = 0.55*pow(muroot_kprime,-1.333333333);
		double mu0 = BrentsMethod(mu_eq, 1e-3, 10, 1e-5);
		kpout << muroot_kprime << " " << mu << " " << mu0 << endl;
		double mvir_root = mvir0*pow(mu*rs0/rmax_p,-1.0/0.43333333);
		double logm_root = log(mvir_root)/log(10);
		//cout << "m200=" << mvir_root << " actual=" << mvir << endl;
	
		mu = BrentsMethod(mr_eq0, 1e-3, 1000000, 1e-9);
		double mvir_check = mu*mvir0;
		double logm_check = log(mvir_check)/log(10);
		
		//mu = BrentsMethod(mr_eq, 1e-3, 1000000, 1e-9);
		//double mvir_root = mu*mvir0;
		//double logm_root = log(mvir_root)/log(10);

		//double c_check = median_concentration_dutton(mvir_root,z);
		//rs_check = pow(3*mvir_root/(4*M_PI*200*1e-9*critical_density(z)),1.0/3.0)/c_check;
		//x = (rmax_true*dp/206.264806)/rs_check;
		////cout << "RS: " << rs << " " << rs_check << endl;
		//double fac = (log(1+c_check) - c_check/(1+c_check)) / (log(1+c0) - c0/(1+c0));
		//double x = muroot_rmax_p/rs_check;
		//double sfac = (log(x/2) + NFW_def_function(x)) / (log(muroot_x0/2) + NFW_def_function(muroot_x0));
		//double fch = mu - (muroot_fzm*fac/sfac);
		//double testm = mvir0*fzm*((log(1+c_check)-c_check/(1+c_check))/(log(1+c0)-c0/(1+c0))) * ((log(x0/2)+NFW_def_function(x0))/(log(x/2)+NFW_def_function(x)));

		//mu = mvir/mvir0;
		//x = (rmax_true*dp/206.264806)/rs;
		//fac = (log(1+c) - c/(1+c)) / (log(1+c0) - c0/(1+c0));
		//x = rmax_p/rs;
		//sfac = (log(x/2) + NFW_def_function(x)) / (log(x0/2) + NFW_def_function(x0));
		//double fch2 = mu - ((menc/menc_true)*fac/sfac);
		//double mvcheck = mvir0*(menc/menc_true)*fac/sfac;
		//double mvcheck2 = mvir0*fzm*fac/sfac;

		//cout << "MVIR: " << testm << " " << mvcheck << " " << mvcheck2 << " " << mvir << " " << fch << " " << fch2 << endl;

		//double mvir = fz*pow(10,9.95018);
		//lens_list[lensnumber]->get_parameters_pmode(0,nfwparams);
		//double ks = nfwparams[0];
		//double sigcrit = sigma_crit_kpc(z,source_redshift);
		//double sigcrit0 = sigma_crit_kpc(lens_redshift,source_redshift);
		double dl = angular_diameter_distance(lens_redshift);
		double dpert = angular_diameter_distance(z);
		int i1,i2;
		i1 = lens_redshift_idx[primary_lens_number];
		i2 = lens_redshift_idx[lensnumber];
		double beta;
		if (z < lens_redshift) beta = default_zsrc_beta_factors[i1-1][i2];
		else if (z > lens_redshift) beta = default_zsrc_beta_factors[i2-1][i1];
		else beta = 0;

		//double ffac = 1 - beta;
		//double gfac = (z > lens_redshift) ? (1-beta) : 1;

		//double sengul_logmvir = log10((sigcrit0/sigcrit)*mvir0);

		//cout << "MVIR: " << mvir << " " << mvir_check << endl;
		//cout << "CHECK: " << fzm << " " << (fz*SQR(dp/dL)) << endl;
		//cout << "rs0=" << rs0 << " rs=" << rs << " rs2=" << rs_check << endl;
		mzout << z << " " << logm << " " << rmax << " " << fz << " " << fzm << " " << mvir_ratio << " " << (avgsig/avgsig_true) << " " << (menc/menc_true) << " " << (mvir/mvir0) << " " << logm_check << " " << logm_root << endl;
		zmout << logm << " " << z << endl;
	}
	if (lens_list[lensnumber]->update_specific_parameter("mvir",mvir0)==false) die("could not find parameter");
	if (lens_list[lensnumber]->update_specific_parameter("z",z0)==false) die("could not find parameter");
}
*/



/*
double QLens::set_required_data_pixel_window(bool verbal)
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return -1e30; }
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(this, ray_tracing_method, (*image_pixel_data));

	int count;
	image_pixel_grid->assign_required_data_pixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,count,image_pixel_data);
	if ((mpi_id==0) and (verbal)) cout << "Number of data pixels in window required for fit: " << count << endl;
}
*/

/*
void QLens::generate_solution_chain_sdp81()
{
	double b_true, q_true, theta_true, xc_true, yc_true;
	double shear_x_true, shear_y_true;
	double ks_true, a_true, gamma_true, xs_true, ys_true;
	double b_fit, alpha_fit, q_fit, theta_fit, xc_fit, yc_fit;
	double shear_x_fit, shear_y_fit;
	double bs_fit, a_fit, xs_fit, ys_fit;
	double mtot_true, mtot_fit;

	bool include_subhalo = false;

	b_true = 1.606; q_true = 0.82; theta_true = 8.3; xc_true = -0.019; yc_true = -0.154;
	shear_x_true = 0.035803; shear_y_true = 0.003763;
	ks_true = 0.0089; a_true = 2.595; xs_true = -1.5; ys_true = -0.37; gamma_true = 2;
	mtot_true = M_PI*SQR(a_true)*ks_true*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
	cout << "Total subhalo mass: " << mtot_true << endl;

	double rs, r_initial, r_final, rstep, theta_sub;
	int rpoints = 15;
	r_initial = sqrt(SQR(xs_true)+SQR(ys_true));
	theta_sub = atan(ys_true/xs_true);
	r_final = 2.0;
	rstep = (r_final-r_initial)/(rpoints-1);

	double chisq, rmax, menc;
	warnings = false;
	string temp_data_filename = "tempdata.fits";
	string rmstring = "if [ -f " + temp_data_filename + " ]; then rm " + temp_data_filename + "; fi";
	fits_format = true;
	set_grid_corners(-3,3,-3,3);
	n_image_pixels_x = 500;
	n_image_pixels_y = 500;
	data_pixel_size = 0.012; // 6 arcseconds / 500 pixels long
	data_pixel_noise = 0.3;
	sim_pixel_noise = 0.3;
	psf_width_x = 0.01;
	psf_width_y = 0.01;
	clear_source_objects();
	add_source_object(GAUSSIAN,1.5,0.018,0,0.9,20,0.18,-0.118);
	add_source_object(GAUSSIAN,2,0.021,0,0.7,120,0.18,-0.208);
	sourcegrid_limit_xmin = 0.05; sourcegrid_limit_xmax = 0.3;
	sourcegrid_limit_ymin = -0.33; sourcegrid_limit_ymax = 0.02;
	regularization_method = Curvature;
	regularization_parameter = 31.5085;
	vary_regularization_parameter = true;
	pixel_magnification_threshold = 5;
	Shear::use_shear_component_params = true;
	source_fit_mode = Pixellated_Source;
	adaptive_grid = true;

	int i;

	for (int l=1; l < 2; l++) {
		if (l==1) include_subhalo = true;

		if (!include_subhalo) {
			b_fit = 1.61241; alpha_fit = 1.13264; q_fit = 0.824023; theta_fit = 8.22863; xc_fit = -0.028069; yc_fit = -0.149037;
			shear_x_fit = 0.0503836; shear_y_fit = 0.00956111;
		} else {
			b_fit = 1.607432701; alpha_fit = 1.38045557; q_fit = 0.7201057458; theta_fit = 7.769220326; xc_fit = -0.04439526632; yc_fit = -0.1574250004;
			shear_x_fit = 0.06842352955; shear_y_fit = 0.01062479437;
			bs_fit = 0.04560161822; xs_fit = -1.500479158; ys_fit = -0.3724162998;
		}

		string filename = "rmax_fit_xs";
		if (!include_subhalo) filename += "_nosub";
		else filename += "_sub";
		ofstream fitout((filename + ".dat").c_str());
		ofstream params_out((filename + "_params.dat").c_str());
		for (i=0, rs=r_initial; i < rpoints; i++, rs += rstep)
		{
			clear_lenses();
			auto_srcgrid_npixels = false;
			auto_sourcegrid = false;
			sourcegrid_xmin=-0.4;
			sourcegrid_xmax=0.3;
			sourcegrid_ymin=-0.6;
			sourcegrid_ymax=0.1999;
			srcgrid_npixels_x = 150;
			srcgrid_npixels_y = 150;

			xs_true = -rs*cos(theta_sub);
			ys_true = -rs*sin(theta_sub);
			create_and_add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
			add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
			create_and_add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
			//calculate_critical_curve_perturbation_radius_numerical(2,true,rmax,menc);
			//cout << endl;
			create_source_surface_brightness_grid(false);
			system(rmstring.c_str());
			plot_lensed_surface_brightness(temp_data_filename,true,false);
			load_image_surface_brightness_grid(temp_data_filename);
			image_pixel_data->set_no_required_data_pixels();
			image_pixel_data->set_required_data_annulus(-0.4,-0.2,0.97,1.43,90,270);
			image_pixel_data->set_required_data_pixels(1.75,2.05,-0.5,0.33);
			clear_lenses();

			auto_srcgrid_npixels = true;
			auto_sourcegrid = true;
			create_and_add_lens(ALPHA,b_fit,alpha_fit,0,q_fit,theta_fit,xc_fit,yc_fit);
			add_shear_lens(shear_x_fit,shear_y_fit,xc_fit,yc_fit);
			lens_list[1]->anchor_center_to_lens(lens_list,0);
			boolvector vary_flags(7);
			for (int i=0; i < 7; i++) vary_flags[i] = true;
			vary_flags[2] = false; // not varying core
			lens_list[0]->vary_parameters(vary_flags);
			boolvector shear_vary_flags(2);
			shear_vary_flags[0] = true;
			shear_vary_flags[1] = true;
			lens_list[1]->vary_parameters(shear_vary_flags);
			if (include_subhalo) {
				create_and_add_lens(PJAFFE,bs_fit,0,0,1,0,xs_fit,ys_fit);
				lens_list[2]->assign_special_anchored_parameters(lens_list[0],1,true); // calculates tidal radius
				boolvector pjaffe_vary_flags(7);
				for (int i=0; i < 7; i++) pjaffe_vary_flags[i] = false;
				pjaffe_vary_flags[0] = true;
				pjaffe_vary_flags[5] = true;
				pjaffe_vary_flags[6] = true;
				lens_list[2]->vary_parameters(pjaffe_vary_flags);
			}
			//print_lens_list(true);
			outside_sb_prior = true;
			chisq = chi_square_fit_simplex();
			adopt_model(bestfitparams);

			if (include_subhalo) {
				calculate_critical_curve_perturbation_radius_numerical(2,false,rmax,menc);
			}

			double host_params[10];
			double dum;
			lens_list[0]->get_parameters(host_params);
			b_fit = host_params[0];
			alpha_fit = host_params[1];
			lens_list[0]->get_q_theta(q_fit,theta_fit);
			theta_fit *= 180.0/M_PI;
			lens_list[0]->get_center_coords(xc_fit,yc_fit);
			double shear_ext,phi_p;
			lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
			shear_x_fit = -shear_ext*cos(2*phi_p+M_PI);
			shear_y_fit = -shear_ext*sin(2*phi_p+M_PI);
			if (include_subhalo) {
				double sub_params[10];
				lens_list[2]->get_parameters(sub_params);
				bs_fit = sub_params[0];
				a_fit = sub_params[1];
				lens_list[2]->get_center_coords(xs_fit,ys_fit);
			}
			//print_lens_list(false);
			cout << endl;

			double avg_kappa, menc_true;
			if (include_subhalo) {
				clear_lenses();
				create_and_add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
				add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
				create_and_add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
				avg_kappa = reference_zfactor*lens_list[2]->kappa_avg_r(rmax);
				menc_true = avg_kappa*M_PI*SQR(rmax)*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				mtot_fit = M_PI*a_fit*bs_fit*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				cout << "rmax = " << rmax << ", mass_enclosed_fit = " << menc << ", mass_enclosed_true = " << menc_true << endl;
				fitout << xs_true << " " << ys_true << " " << xs_fit << " " << ys_fit << " " << chisq << " " << rmax << " " << menc << " " << menc_true << " " << (menc_true-menc)/menc_true << " " << mtot_fit << " " << (mtot_true-mtot_fit)/mtot_true << endl;
			} else {
				fitout << xs_true << " " << rs << " " << chisq << endl;
			}
			params_out << "xs_true=" << xs_true << " ks_true=" << ks_true << " ";
			if (include_subhalo)
				params_out << bs_fit << " " << xs_fit << " " << ys_fit << " ";
			params_out << b_fit << " " << alpha_fit << " " << q_fit << " " << theta_fit << " " << xc_fit << " " << yc_fit << " " << shear_x_fit << " " << shear_y_fit << " " << endl;
		}

		// Now reset and move subhalo in the other direction...
		if (!include_subhalo) {
			b_fit = 1.61241; alpha_fit = 1.13264; q_fit = 0.824023; theta_fit = 8.22863; xc_fit = -0.028069; yc_fit = -0.149037;
			shear_x_fit = 0.0503836; shear_y_fit = 0.00956111;
		} else {
			b_fit = 1.607432701; alpha_fit = 1.38045557; q_fit = 0.7201057458; theta_fit = 7.769220326; xc_fit = -0.04439526632; yc_fit = -0.1574250004;
			shear_x_fit = 0.06842352955; shear_y_fit = 0.01062479437;
			bs_fit = 0.04560161822; xs_fit = -1.500479158; ys_fit = -0.3724162998;
		}

		for (i=0, rs=r_initial; i < rpoints; i++, rs -= rstep)
		{
			clear_lenses();
			auto_srcgrid_npixels = false;
			auto_sourcegrid = false;
			sourcegrid_xmin=-0.4;
			sourcegrid_xmax=0.3;
			sourcegrid_ymin=-0.6;
			sourcegrid_ymax=0.1999;
			srcgrid_npixels_x = 150;
			srcgrid_npixels_y = 150;

			xs_true = -rs*cos(theta_sub);
			ys_true = -rs*sin(theta_sub);
			create_and_add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
			add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
			create_and_add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
			//calculate_critical_curve_perturbation_radius_numerical(2,true,rmax,menc);
			//cout << endl;
			create_source_surface_brightness_grid(false);
			system(rmstring.c_str());
			plot_lensed_surface_brightness(temp_data_filename,true,false);
			load_image_surface_brightness_grid(temp_data_filename);
			image_pixel_data->set_no_required_data_pixels();
			image_pixel_data->set_required_data_annulus(-0.4,-0.2,0.97,1.43,90,270);
			image_pixel_data->set_required_data_pixels(1.75,2.05,-0.5,0.33);
			clear_lenses();

			auto_srcgrid_npixels = true;
			auto_sourcegrid = true;
			create_and_add_lens(ALPHA,b_fit,alpha_fit,0,q_fit,theta_fit,xc_fit,yc_fit);
			add_shear_lens(shear_x_fit,shear_y_fit,xc_fit,yc_fit);
			lens_list[1]->anchor_center_to_lens(lens_list,0);
			boolvector vary_flags(7);
			for (int i=0; i < 7; i++) vary_flags[i] = true;
			vary_flags[2] = false; // not varying core
			lens_list[0]->vary_parameters(vary_flags);
			boolvector shear_vary_flags(2);
			shear_vary_flags[0] = true;
			shear_vary_flags[1] = true;
			lens_list[1]->vary_parameters(shear_vary_flags);
			if (include_subhalo) {
				create_and_add_lens(PJAFFE,bs_fit,0,0,1,0,xs_fit,ys_fit);
				lens_list[2]->assign_special_anchored_parameters(lens_list[0],1,true); // calculates tidal radius
				boolvector pjaffe_vary_flags(7);
				for (int i=0; i < 7; i++) pjaffe_vary_flags[i] = false;
				pjaffe_vary_flags[0] = true;
				pjaffe_vary_flags[5] = true;
				pjaffe_vary_flags[6] = true;
				lens_list[2]->vary_parameters(pjaffe_vary_flags);
			}
			outside_sb_prior = true;
			chisq = chi_square_fit_simplex();
			adopt_model(bestfitparams);

			if (include_subhalo) {
				calculate_critical_curve_perturbation_radius_numerical(2,false,rmax,menc);
			}

			double host_params[10];
			double dum;
			lens_list[0]->get_parameters(host_params);
			b_fit = host_params[0];
			alpha_fit = host_params[1];
			lens_list[0]->get_q_theta(q_fit,theta_fit);
			theta_fit *= 180.0/M_PI;
			lens_list[0]->get_center_coords(xc_fit,yc_fit);
			double shear_ext,phi_p;
			lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
			shear_x_fit = -shear_ext*cos(2*phi_p+M_PI);
			shear_y_fit = -shear_ext*sin(2*phi_p+M_PI);
			double sub_params[10];
			lens_list[2]->get_parameters(sub_params);
			bs_fit = sub_params[0];
			a_fit = sub_params[1];
			lens_list[2]->get_center_coords(xs_fit,ys_fit);
			//print_lens_list(false);
			invert_surface_brightness_map_from_data(false);
			plotcrit("crit.dat");
			if (plot_lensed_surface_brightness("img_pixel")==true) {
				if (mpi_id==0) source_pixel_grid->plot_surface_brightness("src_pixel");
				run_plotter_range("srcpixel","");
				run_plotter_range("imgpixel","");
			}
			cout << endl;

			double avg_kappa, menc_true;
			if (include_subhalo) {
				clear_lenses();
				create_and_add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
				add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
				create_and_add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
				avg_kappa = reference_zfactor*lens_list[2]->kappa_avg_r(rmax);
				menc_true = avg_kappa*M_PI*SQR(rmax)*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				mtot_fit = M_PI*a_fit*bs_fit*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				cout << "rmax = " << rmax << ", mass_enclosed_fit = " << menc << ", mass_enclosed_true = " << menc_true << endl;
				fitout << xs_true << " " << ys_true << " " << xs_fit << " " << ys_fit << " " << chisq << " " << rmax << " " << menc << " " << menc_true << " " << (menc_true-menc)/menc_true << " " << mtot_fit << " " << (mtot_true-mtot_fit)/mtot_true << endl;
			} else {
				fitout << xs_true << " " << rs << " " << chisq << endl;
			}
			params_out << "xs_true=" << xs_true << " ks_true=" << ks_true << " ";
			if (include_subhalo)
				params_out << bs_fit << " " << xs_fit << " " << ys_fit << " ";
			params_out << b_fit << " " << alpha_fit << " " << q_fit << " " << theta_fit << " " << xc_fit << " " << yc_fit << " " << shear_x_fit << " " << shear_y_fit << " " << endl;

		}
		fitout.close();
		params_out.close();
	}

	// Now re-set and vary ks instead...
//	if (!include_subhalo) {
//		b_fit = b_true; alpha_fit = 1.0; q_fit = q_true; theta_fit = theta_true; xc_fit = xc_true; yc_fit = yc_true;
//		shear_x_fit = shear_x_true; shear_y_fit = shear_y_true;
//	} else {
//		//b_fit = 1.607432701; alpha_fit = 1.38045557; q_fit = 0.7201057458; theta_fit = 7.769220326; xc_fit = -0.04439526632; yc_fit = -0.1574250004;
//		//shear_x_fit = 0.06842352955; shear_y_fit = 0.01062479437;
//		//bs_fit = 0.04560161822; xs_fit = -1.500479158; ys_fit = -0.3724162998;
//		b_fit = b_true; alpha_fit = 1.0; q_fit = q_true; theta_fit = theta_true; xc_fit = xc_true; yc_fit = yc_true;
//		shear_x_fit = shear_x_true; shear_y_fit = shear_y_true;
//		bs_fit = 0.001; xs_fit = -1.5; ys_fit = -0.37;
//	}
//
//	double ks, ks_initial, ks_final, kstep;
//	int kspoints = 30;
//	ks_initial = 0.001;
//	ks_final = 0.015;
//	kstep = (ks_final-ks_initial)/(kspoints-1);
//
//	for (int l=1; l < 2; l++) {
//		if (l==1) include_subhalo = true;
//
//		string kfilename = "rmax_fit_ks";
//		if (!include_subhalo) kfilename += "_nosub";
//		else kfilename += "_sub";
//		ofstream kfitout((kfilename + ".dat").c_str());
//		ofstream kparams_out((kfilename + "_params.dat").c_str());
//
//		for (i=0, ks=ks_initial; i < kspoints; i++, ks += kstep)
//		{
//			ks_true = ks;
//			clear_lenses();
//			auto_srcgrid_npixels = false;
//			auto_sourcegrid = false;
//			sourcegrid_xmin=-0.4;
//			sourcegrid_xmax=0.3;
//			sourcegrid_ymin=-0.6;
//			sourcegrid_ymax=0.1999;
//			srcgrid_npixels_x = 150;
//			srcgrid_npixels_y = 150;
//
//			xs_true = -r_initial*cos(theta_sub);
//			ys_true = -r_initial*sin(theta_sub);
//			a_true = SQR(21.6347)*ks_true/1.606; // from formula for tidal radius of Munoz profile, see Minor et al. (2016)
//			create_and_add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
//			add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
//			create_and_add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
//			//calculate_critical_curve_perturbation_radius_numerical(2,true,rmax,menc);
//			//cout << endl;
//			system(rmstring.c_str());
//			plot_lensed_surface_brightness(temp_data_filename,true,false);
//			load_image_surface_brightness_grid(temp_data_filename);
//			image_pixel_data->set_no_required_data_pixels();
//			image_pixel_data->set_required_data_annulus(-0.4,-0.2,0.97,1.43,90,270);
//			image_pixel_data->set_required_data_pixels(1.75,2.05,-0.5,0.33);
//			clear_lenses();
//
//			auto_srcgrid_npixels = true;
//			auto_sourcegrid = true;
//			create_and_add_lens(ALPHA,b_fit,alpha_fit,0,q_fit,theta_fit,xc_fit,yc_fit);
//			add_shear_lens(shear_x_fit,shear_y_fit,xc_fit,yc_fit);
//			lens_list[1]->anchor_center_to_lens(lens_list,0);
//			boolvector vary_flags(7);
//			for (int i=0; i < 7; i++) vary_flags[i] = true;
//			vary_flags[2] = false; // not varying core
//			lens_list[0]->vary_parameters(vary_flags);
//			boolvector shear_vary_flags(2);
//			shear_vary_flags[0] = true;
//			shear_vary_flags[1] = true;
//			lens_list[1]->vary_parameters(shear_vary_flags);
//			if (include_subhalo) {
//				create_and_add_lens(PJAFFE,bs_fit,0,0,1,0,xs_fit,ys_fit);
//				lens_list[2]->assign_special_anchored_parameters(lens_list[0],1,true); // calculates tidal radius
//				boolvector pjaffe_vary_flags(7);
//				for (int i=0; i < 7; i++) pjaffe_vary_flags[i] = false;
//				pjaffe_vary_flags[0] = true;
//				pjaffe_vary_flags[5] = true;
//				pjaffe_vary_flags[6] = true;
//				lens_list[2]->vary_parameters(pjaffe_vary_flags);
//			}
//			outside_sb_prior = true;
//			chisq = chi_square_fit_simplex();
//			adopt_model(bestfitparams);
//
//			if (include_subhalo) {
//				calculate_critical_curve_perturbation_radius_numerical(2,false,rmax,menc);
//			}
//
//			double host_params[10];
//			double dum;
//			lens_list[0]->get_parameters(host_params);
//			b_fit = host_params[0];
//			alpha_fit = host_params[1];
//			lens_list[0]->get_q_theta(q_fit,theta_fit);
//			theta_fit *= 180.0/M_PI;
//			lens_list[0]->get_center_coords(xc_fit,yc_fit);
//			double shear_ext,phi_p;
//			lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
//			shear_x_fit = -shear_ext*cos(2*phi_p+M_PI);
//			shear_y_fit = -shear_ext*sin(2*phi_p+M_PI);
//			if (include_subhalo) {
//				double sub_params[10];
//				lens_list[2]->get_parameters(sub_params);
//				bs_fit = sub_params[0];
//				a_fit = sub_params[1];
//				lens_list[2]->get_center_coords(xs_fit,ys_fit);
//			}
//			print_lens_list(false);
//			cout << endl;
//			invert_surface_brightness_map_from_data(false);
//			plotcrit("crit.dat");
//			if (plot_lensed_surface_brightness("img_pixel")==true) {
//				if (mpi_id==0) source_pixel_grid->plot_surface_brightness("src_pixel");
//				run_plotter_range("srcpixel","");
//				run_plotter_range("imgpixel","");
//			}
//
//			double avg_kappa, menc_true;
//			if (include_subhalo) {
//				clear_lenses();
//				create_and_add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
//				add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
//				create_and_add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
//				avg_kappa = reference_zfactor*lens_list[2]->kappa_avg_r(rmax);
//				menc_true = avg_kappa*M_PI*SQR(rmax)*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
//				mtot_fit = M_PI*a_fit*bs_fit*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
//				mtot_true = M_PI*SQR(a_true)*ks_true*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
//				cout << "rmax = " << rmax << ", mass_enclosed_fit = " << menc << ", mass_enclosed_true = " << menc_true << endl;
//				kfitout << ks_true << " " << xs_fit << " " << bs_fit << " " << chisq << " " << rmax << " " << menc << " " << menc_true << " " << (menc_true-menc)/menc_true << " " << mtot_fit << " " << (mtot_true-mtot_fit)/mtot_true << endl;
//			} else {
//				kfitout << ks_true << " " << chisq << " " << endl;
//			}
//			kparams_out << "xs_true=" << xs_true << " ks_true=" << ks_true << " ";
//			if (include_subhalo)
//				kparams_out << bs_fit << " " << xs_fit << " " << ys_fit << " ";
//			kparams_out << b_fit << " " << alpha_fit << " " << q_fit << " " << theta_fit << " " << xc_fit << " " << yc_fit << " " << shear_x_fit << " " << shear_y_fit << endl;
//
//		}
//		kfitout.close();
//		kparams_out.close();
//	}

}
*/

void QLens::create_output_directory()
{
	if (mpi_id==0) {
		struct stat sb;
		stat(fit_output_dir.c_str(),&sb);
		if (S_ISDIR(sb.st_mode)==false)
			mkdir(fit_output_dir.c_str(),S_IRWXU | S_IRWXG);
	}
}

void QLens::open_output_file(ofstream &outfile, char* filechar_in)
{
	string filename_in(filechar_in);
	if (fit_output_dir != ".") create_output_directory(); // in case it hasn't been created already
	string filename = fit_output_dir + "/" + filename_in;
	outfile.open(filename.c_str());
}

void QLens::open_output_file(ofstream &outfile, string filename_in)
{
	if (fit_output_dir != ".") create_output_directory(); // in case it hasn't been created already
	string filename = fit_output_dir + "/" + filename_in;
	outfile.open(filename.c_str());
}

void QLens::reset_grid()
{
	if (defspline != NULL) {
		delete defspline;
		defspline = NULL;
	}
	if (grid != NULL) {
		delete grid;
		grid = NULL;
	}
	critical_curve_pts.clear();
	caustic_pts.clear();
	length_of_cc_cell.clear();
	sorted_critical_curves = false;
	sorted_critical_curve.clear();
	singular_pts.clear();
	if (cc_splined==true) {
		delete[] ccspline;
		delete[] caustic;
		cc_splined = false;
	}
}

void QLens::delete_ccspline()
{
	if (cc_splined==true) {
		delete[] ccspline;
		delete[] caustic;
		cc_splined = false;
	}
}

QLens::~QLens()
{
	int i,j;
	if (nlens > 0) {
		for (i=0; i < nlens; i++) {
			delete lens_list[i];
		}
		delete[] lens_redshift_idx;
		delete[] lens_list;
	}
	if (n_lens_redshifts > 0) delete[] lens_redshifts;

	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			delete sb_list[i];
		delete[] sb_list;
		n_sb = 0;
	}

	delete grid;
	delete param_settings;
	if (defspline != NULL) delete defspline;
	if (fitmodel != NULL) delete fitmodel;
	//if (sourcepts_fit != NULL) delete[] sourcepts_fit;
	//if (vary_sourcepts_x != NULL) delete[] vary_sourcepts_x;
	//if (vary_sourcepts_y != NULL) delete[] vary_sourcepts_y;
	if (psf_matrix != NULL) {
		for (int i=0; i < psf_npixels_x; i++) delete[] psf_matrix[i];
		delete[] psf_matrix;
	}
	if (source_redshifts != NULL) delete[] source_redshifts;
	if (zfactors != NULL) {
		for (i=0; i < n_sourcepts_fit; i++) delete[] zfactors[i];
		delete[] zfactors;
	}
	if (default_zsrc_beta_factors != NULL) {
		for (i=0; i < n_lens_redshifts-1; i++) delete[] default_zsrc_beta_factors[i];
		delete[] default_zsrc_beta_factors;
	}
	if (beta_factors != NULL) {
		for (i=0; i < n_sourcepts_fit; i++) {
			for (j=0; j < n_lens_redshifts-1; j++) delete[] beta_factors[i][j];
			if (n_lens_redshifts > 1) delete[] beta_factors[i];
		}
		delete[] beta_factors;
	}
	//if (sourcepts_upper_limit != NULL) delete[] sourcepts_upper_limit;
	//if (sourcepts_lower_limit != NULL) delete[] sourcepts_lower_limit;
	if ((image_data != NULL) and (borrowed_image_data==false)) delete[] image_data;
	if ((image_pixel_data != NULL) and (borrowed_image_data==false)) delete image_pixel_data;
	if (image_surface_brightness != NULL) delete[] image_surface_brightness;
	if (foreground_surface_brightness != NULL) delete[] foreground_surface_brightness;
	if (source_pixel_vector != NULL) delete[] source_pixel_vector;
	if (source_pixel_n_images != NULL) delete[] source_pixel_n_images;
	if (active_image_pixel_i != NULL) delete[] active_image_pixel_i;
	if (active_image_pixel_j != NULL) delete[] active_image_pixel_j;
	if (image_pixel_location_Lmatrix != NULL) delete[] image_pixel_location_Lmatrix;
	if (source_pixel_location_Lmatrix != NULL) delete[] source_pixel_location_Lmatrix;
	if (Lmatrix_index != NULL) delete[] Lmatrix_index;
	if (Lmatrix != NULL) delete[] Lmatrix;
	if (Dvector != NULL) delete[] Dvector;
	if (Fmatrix != NULL) delete[] Fmatrix;
	if (Fmatrix_index != NULL) delete[] Fmatrix_index;
	if (Rmatrix != NULL) delete[] Rmatrix;
	if (Rmatrix_index != NULL) delete[] Rmatrix_index;
	if (source_pixel_grid != NULL) delete source_pixel_grid;
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	if (group_leader != NULL) delete[] group_leader;
}

/***********************************************************************************************************************/

// POLYCHORD FUNCTIONS

QLens* lensptr;
double mcsampler_set_lensptr(QLens* lens_in)
{
	lensptr = lens_in;
	return 0.0;
}

double polychord_loglikelihood (double theta[], int nDims, double phi[], int nDerived)
{
	double logl = -lensptr->LogLikeFunc(theta);
	lensptr->DerivedParamFunc(theta,phi);
	return logl;
}

void polychord_prior (double cube[], double theta[], int nDims)
{
	lensptr->transform_cube(theta,cube);
}

void polychord_dumper(int ndead,int nlive,int npars,double* live,double* dead,double* logweights,double logZ, double logZerr)
{
}

void multinest_loglikelihood(double *Cube, int &ndim, int &npars, double &lnew, void *context)
{
	double *params = new double[ndim];
	lensptr->transform_cube(params,Cube);
	lnew = -lensptr->LogLikeFunc(params);
	lensptr->DerivedParamFunc(params,Cube+ndim);
	delete[] params;
}

void dumper_multinest(int &nSamples, int &nlive, int &nPar, double **physLive, double **posterior, double **paramConstr, double &maxLogLike, double &logZ, double &INSlogZ, double &logZerr, void *context)
{
	// convert the 2D Fortran arrays to C++ arrays
	
	// the posterior distribution
	// postdist will have nPar parameters in the first nPar columns & loglike value & the posterior probability in the last two columns
	
	int i, j;
	
	double postdist[nSamples][nPar + 2];
	for( i = 0; i < nPar + 2; i++ )
		for( j = 0; j < nSamples; j++ )
			postdist[j][i] = posterior[0][i * nSamples + j];
	
	// last set of live points
	// pLivePts will have nPar parameters in the first nPar columns & loglike value in the last column
	
	double pLivePts[nlive][nPar + 1];
	for( i = 0; i < nPar + 1; i++ )
		for( j = 0; j < nlive; j++ )
			pLivePts[j][i] = physLive[0][i * nlive + j];
}

void QLens::test_lens_functions()
{
	clear_lenses();
	load_image_data("alphafit.dat");

	Alpha *A = new Alpha();
	A->initialize_parameters(4.5,1,0,0.8,30,0.7,0.3);
	boolvector flags(7);
	flags[0] = true;
	flags[1] = false;
	flags[2] = false;
	flags[3] = true;
	flags[4] = true;
	flags[5] = true;
	flags[6] = true;
	//flags[7] = true;
	//param_settings->print_penalty_limits();
	A->set_vary_flags(flags);
	Shear *S = new Shear();
	S->initialize_parameters(0.02,10,0,0);
	boolvector flag2(4);
	flag2[0] = true;
	flag2[1] = true;
	flag2[2] = false;
	flag2[3] = false;
	S->set_vary_flags(flag2);
	add_lens(A,0.5,2);
	add_lens(S,0.5,2);
	use_analytic_bestfit_src = true;
	include_flux_chisq = true;

	chi_square_fit_simplex();
	use_bestfit();

	bool status;
	vector<ImageSet> imgsets = get_fit_imagesets(status);

	// The following shows how to access the image data in the "imgset" object
	/*
	cout << endl;
	for (int j=0; j < imgsets.size(); j++) {
		cout << "Source " << j << ": redshift = " << imgsets[j].zsrc << endl;
		cout << "Number of images: " << imgsets[j].n_images << endl;
		cout << "Source:  " << imgsets[j].src[0] << " " << imgsets[j].src[1] << endl;
		for (int i=0; i < imgsets[j].n_images; i++) cout << "Image" << i << ": " << imgsets[j].images[i].pos[0] << " " << imgsets[j].images[i].pos[1] << " " << imgsets[j].images[i].mag << " " << imgsets[j].imgflux(i) << endl; 
		cout << endl;
	}

	vector<ImageDataSet> imgdatasets = export_to_ImageDataSet();
	cout << "Image Data:" << endl;
	for (int j=0; j < imgdatasets.size(); j++) {
		cout << "Source " << j << ": redshift = " << imgdatasets[j].zsrc << endl;
		cout << "Number of images: " << imgdatasets[j].n_images << endl;
		for (int i=0; i < imgdatasets[j].n_images; i++) cout << "Image" << i << ": " << imgdatasets[j].images[i].pos[0] << " " << imgdatasets[j].images[i].pos[1] << " " << imgdatasets[j].images[i].flux << endl; 
		cout << endl;
	}
	*/



	//OR...you can print similar information by calling the following function:
	//imgset.print();

	/*
	cout << "Generating critical curves/caustics and plotting to files 'crit.dat' and 'caust.dat'..." << endl;
	plot_sorted_critical_curves(); // generates critical curves/caustics and stores them
	// The following shows how the critical curves/caustics are accessed using "sorted_critical_curve", which is a std::vector of
	// "critical_curve" objects (in qlens.h you can see what the critical_curve structure looks like). The length of sorted_critical_curve vector
	// tells you how many distinct critical curves there are (sometimes there is one, or two, or more...it depends on the lens).
	ofstream ccfile("crit.dat");
	ofstream caustic_file("caust.dat");
	int i,j;
	for (i=0; i < sorted_critical_curve.size(); i++ ) {
		for (j=0; j < sorted_critical_curve[i].cc_pts.size(); j++) {
			ccfile << sorted_critical_curve[i].cc_pts[j][0] << " " << sorted_critical_curve[i].cc_pts[j][1] << endl; // printing x- and y-values for critical curves
			caustic_file << sorted_critical_curve[i].caustic_pts[j][0] << " " << sorted_critical_curve[i].caustic_pts[j][1] << endl; // same for caustics
		}
		ccfile << endl;
		caustic_file << endl;
	}
	*/

	//delete A;
}
