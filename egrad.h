#ifndef EGRAD_H
#define EGRAD_H

#include "mathexpr.h"
#include "brent.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

class EllipticityGradient : public Brent
{
	protected:
	bool ellipticity_gradient; // if true, then allows a gradient in both ellipticity and position angle
	bool contours_overlap; // relevant for ellipticity gradient mode
	double overlap_log_penalty_prior;
	int egrad_ellipticity_mode; // same as ellipticity_mode in LensProfile, but only allows modes 0 or 1 right now
	int egrad_mode;
	double xi_initial_egrad, xi_final_egrad, xi_ref_egrad; // keeps track of the region where ellipticity/PA is changing
	bool center_gradient; // this is currently always false, but later I might add the option to have a center gradient
	bool fourier_gradient; // if true, allows a gradient in the Fourier amplitudes perturbing surface brightness (or kappa)
	bool use_linear_xivals; // for initial knot spacing

	int bspline_order; // set to 3 by default
	int n_bspline_knots_tot;
	double *geometric_param[4]; // this will point to the parameter arrays for q, theta, xc and yc
	double *geometric_knots[4]; // this is only needed if B-spline fitting (egrad_mode=0) is used
	double geometric_param_ref[4]; // used by egrad=2
	double geometric_param_dif[4]; // used by egrad=2
	int n_egrad_params[4];
	double **fourier_param;
	double **fourier_knots;
	int *fourier_grad_mvals;
	int *n_fourier_grad_params;
	int n_fourier_grad_modes;
	bool *angle_param_egrad;

	// For profile fitting from isophote fits
	int n_isophote_datapts;
	double *profile_fit_xivals, *profile_fit_logxivals, *sbprofile_data, *sbprofile_errors;
	double *profile_fit_data, *profile_fit_errs, *profile_fit_weights;
	double *profile_fit_egrad_params;
	double *profile_fit_bspline_coefs;
	int profile_fit_nparams, profile_fit_istart, egrad_paramnum;
	double profile_fit_min_knot_interval;

	void check_for_overlapping_contours();
	void allocate_bspline_work_arrays(const int n_data);
	void free_bspline_work_arrays();
	double fit_bspline_curve(double *knots, double *coefs);

	public:
	EllipticityGradient();
	~EllipticityGradient();
	bool setup_egrad_params(const int egrad_mode_in, const int ellipticity_mode_in, const dvector& egrad_params, int& n_egrad_params_tot, const int n_bspline_coefs, const dvector& knots, const double ximin = 0.1, const double ximax = 5, const double xiref = 1.5, const bool linear_xivals = false); // arbitrary default values for the last three arguments
	bool setup_fourier_grad_params(const int n_modes, const ivector& mvals, const dvector& fourier_grad_params, int& n_fourier_grad_params_tot, const dvector& knots);
	void set_egrad_ptr();
	void disable_egrad_mode(int& n_tot_egrad_params);
	void set_geometric_param_pointers_egrad(double **param, boolvector& angle_param, int& qi);
	void get_egrad_params(dvector& egrad_params);
	int get_egrad_nparams();
	int get_fgrad_nparams();
	void update_egrad_meta_parameters();
	void set_geometric_paramnames_egrad(std::vector<std::string>& paramnames, std::vector<std::string>& latex_paramnames, std::vector<std::string>& latex_param_subscripts, int &qi, std::string latex_suffix = "");
	void set_geometric_param_ranges_egrad(boolvector& set_auto_penalty_limits, dvector& penalty_lower_limits, dvector& penalty_upper_limits, int &param_i);
	void set_geometric_stepsizes_egrad(dvector& stepsizes, int &index);
	void set_fourier_paramnums(int *paramnum, int paramnum0);

	void ellipticity_function(const double xi, double& ep, double& theta);
	double angle_function(const double xi);
	void fourier_mode_function(const double xi, double* cosamp, double* sinamp);
	void fourier_mode_function(const double xi, const int mval, double& cosamp, double& sinamp);
	double (EllipticityGradient::*egrad_ptr)(const double xi, double *paramvals, const int param_index);
	double egrad_tanh_function(const double xi, double *paramvals, const int param_index);
	double egrad_bspline_function(const double xi, double *paramvals, const int param_index);
	double elliptical_radius_root(const double x, const double y);
	void plot_ellipticity_function(const double ximin, const double ximax, const int nn, const std::string suffix = "");
	void plot_fourier_functions(const double ximin, const double ximax, const int nn, const std::string suffix = "");
	void output_egrad_values_and_knots(std::ofstream& outfile);
	int get_egrad_mode() { return egrad_mode; }

	private:
	double elliptical_radius_root_eq(const double xi, const double &xi_root_x, const double &xi_root_y);
	double egrad_minq; // useful for finding expected range of elliptical radius at a given point (for root finder)
	// the following are used during the B-spline fitting, but are no longer used once the B-spline coefficients and knots have been determined.
	int bspline_nmax;
	double *bspline_work;
	int *bspline_iwork;
	double *bspline_weights;
};

struct IsophoteData {
	int n_xivals;
	bool use_xcyc;
	bool use_A34;
	bool use_A56;

	double *xivals, *logxivals;
	double *sb_avg_vals, *sb_errs;
	double *qvals, *q_errs;
	double *thetavals, *theta_errs;
	double *xcvals, *xc_errs;
	double *ycvals, *yc_errs;

	double *A3vals, *A3_errs;
	double *B3vals, *B3_errs;
	double *A4vals, *A4_errs;
	double *B4vals, *B4_errs;

	double *A5vals, *A5_errs;
	double *B5vals, *B5_errs;
	double *A6vals, *A6_errs;
	double *B6vals, *B6_errs;

	IsophoteData() {
		xivals=logxivals=sb_avg_vals=sb_errs=qvals=q_errs=thetavals=theta_errs=xcvals=xc_errs=ycvals=yc_errs=A3vals=A3_errs=B3vals=B3_errs=A4vals=A4_errs=B4vals=B4_errs=NULL;
		A5vals=A5_errs=B5vals=B5_errs=A6vals=A6_errs=B6vals=B6_errs=NULL;
		n_xivals = 0;
		use_xcyc = true;
		use_A34 = true;
		use_A56 = false;
	}
	IsophoteData(IsophoteData &iso_in);
	void input(const int n_xivals_in);
	IsophoteData(const int n_xivals_in);
	void input(const int n_xivals_in, double* xivals_in);
	bool load_profiles_noerrs(std::ifstream& profin, const double errfrac, const bool include_xcyc, const bool include_a34, const bool include_a56);
	bool load_profiles(std::ifstream& profin, const bool include_xcyc, const bool include_a34, const bool include_a56);
	void setnan();
	void plot_isophote_parameters(const std::string suffix);
	~IsophoteData();
};

#endif // EGRAD_H
