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
		use_A56 = false;
	}
	IsophoteData(IsophoteData &iso_in) {
		input(iso_in.n_xivals);
		use_A56 = iso_in.use_A56;
		for (int i=0; i < n_xivals; i++) {
			xivals[i] = iso_in.xivals[i];
			logxivals[i] = iso_in.logxivals[i];
			sb_avg_vals[i] = iso_in.sb_avg_vals[i];
			sb_errs[i] = iso_in.sb_errs[i];
			qvals[i] = iso_in.qvals[i];
			thetavals[i] = iso_in.thetavals[i];
			xcvals[i] = iso_in.xcvals[i];
			ycvals[i] = iso_in.ycvals[i];
			q_errs[i] = iso_in.q_errs[i];
			theta_errs[i] = iso_in.theta_errs[i];
			xc_errs[i] = iso_in.xc_errs[i];
			yc_errs[i] = iso_in.yc_errs[i];
			A3vals[i] = iso_in.A3vals[i];
			B3vals[i] = iso_in.B3vals[i];
			A4vals[i] = iso_in.A4vals[i];
			B4vals[i] = iso_in.B4vals[i];
			A3_errs[i] = iso_in.A3_errs[i];
			B3_errs[i] = iso_in.B3_errs[i];
			A4_errs[i] = iso_in.A4_errs[i];
			B4_errs[i] = iso_in.B4_errs[i];
		}
		if (use_A56) {
			for (int i=0; i < n_xivals; i++) {
				A5vals[i] = iso_in.A5vals[i];
				B5vals[i] = iso_in.B5vals[i];
				A6vals[i] = iso_in.A6vals[i];
				B6vals[i] = iso_in.B6vals[i];
				A5_errs[i] = iso_in.A5_errs[i];
				B5_errs[i] = iso_in.B5_errs[i];
				A6_errs[i] = iso_in.A6_errs[i];
				B6_errs[i] = iso_in.B6_errs[i];
			}
		}

	}
	void input(const int n_xivals_in) {
		n_xivals = n_xivals_in;
		xivals = new double[n_xivals];
		logxivals = new double[n_xivals];
		sb_avg_vals = new double[n_xivals];
		sb_errs = new double[n_xivals];
		qvals = new double[n_xivals];
		thetavals = new double[n_xivals];
		xcvals = new double[n_xivals];
		ycvals = new double[n_xivals];
		q_errs = new double[n_xivals];
		theta_errs = new double[n_xivals];
		xc_errs = new double[n_xivals];
		yc_errs = new double[n_xivals];

		A3vals = new double[n_xivals];
		B3vals = new double[n_xivals];
		A4vals = new double[n_xivals];
		B4vals = new double[n_xivals];
		A3_errs = new double[n_xivals];
		B3_errs = new double[n_xivals];
		A4_errs = new double[n_xivals];
		B4_errs = new double[n_xivals];

		A5vals = new double[n_xivals];
		B5vals = new double[n_xivals];
		A6vals = new double[n_xivals];
		B6vals = new double[n_xivals];
		A5_errs = new double[n_xivals];
		B5_errs = new double[n_xivals];
		A6_errs = new double[n_xivals];
		B6_errs = new double[n_xivals];

	}
	void input(const int n_xivals_in, double* xivals_in) {
		input(n_xivals_in);
		for (int i=0; i < n_xivals; i++) {
			xivals[i] = xivals_in[i];
			logxivals[i] = log(xivals[i])/ln10;
		}
	}
	void setnan() {
		if (n_xivals > 0) {
			for (int i=0; i < n_xivals; i++) {
				sb_avg_vals[i]=sb_errs[i]=qvals[i]=thetavals[i]=xcvals[i]=ycvals[i]=q_errs[i]=theta_errs[i]=xc_errs[i]=yc_errs[i]=NAN; // if any bestfit solutions can't be found, it will be left as NAN
				A3vals[i]=B3vals[i]=A4vals[i]=B4vals[i]=A3_errs[i]=B3_errs[i]=A4_errs[i]=B4_errs[i]=NAN;
			}
		}
	}
	void plot_isophote_parameters(const std::string suffix) {
		std::string sbname = "sbvals_" + suffix + ".dat";
		std::string qname = "qvals_" + suffix + ".dat";
		std::string thetaname = "thetavals_" + suffix + ".dat";
		std::string xcname = "xcvals_" + suffix + ".dat";
		std::string ycname = "ycvals_" + suffix + ".dat";

		std::string A3name = "A3vals_" + suffix + ".dat";
		std::string B3name = "B3vals_" + suffix + ".dat";
		std::string A4name = "A4vals_" + suffix + ".dat";
		std::string B4name = "B4vals_" + suffix + ".dat";

		std::string A5name = "A5vals_" + suffix + ".dat";
		std::string B5name = "B5vals_" + suffix + ".dat";
		std::string A6name = "A6vals_" + suffix + ".dat";
		std::string B6name = "B6vals_" + suffix + ".dat";

		std::ofstream sbout(sbname.c_str());
		std::ofstream qout(qname.c_str());
		std::ofstream thetaout(thetaname.c_str());
		std::ofstream xcout(xcname.c_str());
		std::ofstream ycout(ycname.c_str());

		std::ofstream A3out(A3name.c_str());
		std::ofstream B3out(B3name.c_str());
		std::ofstream A4out(A4name.c_str());
		std::ofstream B4out(B4name.c_str());


		std::ofstream A5out;
		std::ofstream B5out;
		std::ofstream A6out;
		std::ofstream B6out;
		if (use_A56) {
			A5out.open(A5name.c_str());
			B5out.open(B5name.c_str());
			A6out.open(A6name.c_str());
			B6out.open(B6name.c_str());
		}



		for (int i=0; i < n_xivals; i++) {
			sbout << xivals[i] << " " << sb_avg_vals[i] << " " << 2*sb_errs[i] << std::endl;
			qout << xivals[i] << " " << qvals[i] << " " << 2*q_errs[i] << std::endl;
			thetaout << xivals[i] << " " << radians_to_degrees(thetavals[i]) << " " << radians_to_degrees(2*theta_errs[i]) << std::endl;
			xcout << xivals[i] << " " << xcvals[i] << " " << 2*xc_errs[i] << std::endl;
			ycout << xivals[i] << " " << ycvals[i] << " " << 2*yc_errs[i] << std::endl;
			A3out << xivals[i] << " " << A3vals[i] << " " << 2*A3_errs[i] << std::endl;
			B3out << xivals[i] << " " << B3vals[i] << " " << 2*B3_errs[i] << std::endl;
			A4out << xivals[i] << " " << A4vals[i] << " " << 2*A4_errs[i] << std::endl;
			B4out << xivals[i] << " " << B4vals[i] << " " << 2*B4_errs[i] << std::endl;

			if (use_A56) {
				A5out << xivals[i] << " " << A5vals[i] << " " << 2*A5_errs[i] << std::endl;
				B5out << xivals[i] << " " << B5vals[i] << " " << 2*B5_errs[i] << std::endl;
				A6out << xivals[i] << " " << A6vals[i] << " " << 2*A6_errs[i] << std::endl;
				B6out << xivals[i] << " " << B6vals[i] << " " << 2*B6_errs[i] << std::endl;
			}
		}
	}
	~IsophoteData() {
		if (xivals != NULL) delete[] xivals;
		if (logxivals != NULL) delete[] logxivals;
		if (sb_avg_vals != NULL) delete[] sb_avg_vals;
		if (sb_errs != NULL) delete[] sb_errs;
		if (qvals != NULL) delete[] qvals;
		if (thetavals != NULL) delete[] thetavals;
		if (xcvals != NULL) delete[] xcvals;
		if (ycvals != NULL) delete[] ycvals;
		if (q_errs != NULL) delete[] q_errs;
		if (theta_errs != NULL) delete[] theta_errs;
		if (xc_errs != NULL) delete[] xc_errs;
		if (yc_errs != NULL) delete[] yc_errs;

		if (A3vals != NULL) delete[] A3vals;
		if (B3vals != NULL) delete[] B3vals;
		if (A4vals != NULL) delete[] A4vals;
		if (B4vals != NULL) delete[] B4vals;
		if (A3_errs != NULL) delete[] A3_errs;
		if (B3_errs != NULL) delete[] B3_errs;
		if (A4_errs != NULL) delete[] A4_errs;
		if (B4_errs != NULL) delete[] B4_errs;

		if (A5vals != NULL) delete[] A5vals;
		if (B5vals != NULL) delete[] B5vals;
		if (A6vals != NULL) delete[] A6vals;
		if (B6vals != NULL) delete[] B6vals;
		if (A5_errs != NULL) delete[] A5_errs;
		if (B5_errs != NULL) delete[] B5_errs;
		if (A6_errs != NULL) delete[] A6_errs;
		if (B6_errs != NULL) delete[] B6_errs;

	}
};

#endif // EGRAD_H
