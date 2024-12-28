#ifndef PROFILE_H
#define PROFILE_H

#include "mathexpr.h"
#include "gauss.h"
#include "spline.h"
#include "brent.h"
#include "egrad.h"
#include "lensvec.h"
//#include "sbprofile.h"
#include "romberg.h"
#include "cosmo.h"
#include <iostream>
#include <vector>
#include <complex>

class Sersic;
class DoubleSersic;
class Cored_Sersic;
class TopHatLens;
class SPLE;
class dPIE;
class NFW_Source;
class SB_Profile;

enum IntegrationMethod { Romberg_Integration, Gaussian_Quadrature, Gauss_Patterson_Quadrature, Fejer_Quadrature };
enum LensProfileName
{
	KSPLINE,
	sple_LENS,
	dpie_LENS,
	nfw,
	TRUNCATED_nfw,
	CORED_nfw,
	HERNQUIST,
	EXPDISK,
	CORECUSP,
	SERSIC_LENS,
	DOUBLE_SERSIC_LENS,
	CORED_SERSIC_LENS,
	MULTIPOLE,
	PTMASS,
	SHEAR,
	SHEET,
	DEFLECTION,
	TOPHAT_LENS,
	TABULATED,
	QTABULATED,
	TESTMODEL
};

struct LensIntegral;
class QLens;

class LensProfile : public Romberg, public GaussLegendre, public GaussPatterson, public ClenshawCurtis, public EllipticityGradient
{
	friend struct LensIntegral;
	friend class QLens;
	friend class SB_Profile;
	friend class Sersic;
	friend class Cored_Sersic;
	friend class SPLE;
	friend class dPIE;
	friend class ImagePixelGrid;

	// the following private declarations are specific to LensProfile and not derived classes
	private:
	Spline kspline;
	double kappa_splint(double);
	double kappa_rsq_deriv_splint(double);
	double qx_parameter, f_parameter;

	protected:
	LensProfileName lenstype;
	bool center_defined;
	bool lensed_center_coords; // option for line-of-sight perturber that makes the lensed position of the perturber the free parameters
	double zlens, zsrc_ref;
	double zlens_current; // used to check if zlens has been changed, in which case sigma_cr, etc. are updated
	double sigma_cr, kpc_to_arcsec;
	double q, theta, x_center, y_center; // four base parameters, which can be added to in derived lens models
	double xc_prime, yc_prime; // used if lensed_center_coords is set to true
	double f_major_axis; // used for defining elliptical radius; set in function set_q(q)
	double epsilon, epsilon1, epsilon2; // used for defining ellipticity, and/or components of ellipticity (epsilon1, epsilon2)
	double costheta, sintheta;
	double integral_tolerance;
	double theta_eff; // used for intermediate calculations if ellipticity components are being used
	double **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model
	bool perturber; // optional flag that can make the perturber subgridding faster, if used

	int n_params, n_vary_params;
	int lensprofile_nparams; // just the parameters that define the kappa profile (rather than the angular structure or center coord's)
	bool angle_param_exists;
	int ellipticity_paramnum; // used to keep track of ellipticity parameter (this feature is used only by qtab models)
	boolvector vary_params;
	boolvector angle_param; // used to keep track of angle parameters so they can be easily converted to degrees and displayed
	std::string model_name;
	std::string special_parameter_command;
	std::vector<std::string> paramnames;
	std::vector<std::string> latex_paramnames, latex_param_subscripts;
	boolvector set_auto_penalty_limits;
	dvector penalty_upper_limits, penalty_lower_limits;
	dvector stepsizes;
	bool include_limits;
	dvector lower_limits, upper_limits;
	dvector lower_limits_initial, upper_limits_initial;

	int n_fourier_modes; // Number of Fourier mode perturbations to elliptical density contours (zero by default)
	ivector fourier_mode_mvals, fourier_mode_paramnum;
	dvector fourier_mode_cosamp, fourier_mode_sinamp;
	Spline *fourier_integral_left_cos_spline;
	Spline *fourier_integral_left_sin_spline;
	Spline *fourier_integral_right_cos_spline;
	Spline *fourier_integral_right_sin_spline;
	bool fourier_integrals_splined;

	virtual void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void setup_base_lens_properties(const int np, const int lensprofile_np, const bool is_elliptical_lens, const int pmode_in = 0, const int subclass_in = -1);
	void copy_base_lensdata(const LensProfile* lens_in);
	void copy_source_data_to_lens(const SB_Profile* in);

	void set_nparams_and_anchordata(const int &n_params_in, const bool resize = false);
	void reset_anchor_lists();
	void set_spawned_mass_and_anchor_parameters(SB_Profile* sb_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void set_geometric_param_pointers(int qi);
	void set_geometric_paramnames(int qi);
	void set_angle(const double &theta_degrees);
	void set_angle_radians(const double &theta_in);
	void set_ellipticity_parameter(const double &q_in);
	void rotate(double&, double&);
	void rotate_back(double&, double&);

	void set_geometric_parameters(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	void set_angle_from_components(const double &comp_x, const double &comp_y);
	void set_center_if_lensed_coords();
	void set_integration_parameters(const int &nn, const double &acc);
	void copy_integration_tables(const LensProfile* lens_in);

	void set_integration_pointers();
	virtual void set_model_specific_integration_pointers();
	void update_meta_parameters_and_pointers();
	void update_angle_meta_params();
	void update_ellipticity_meta_parameters();
	void update_cosmology_meta_parameters();
	virtual void update_meta_parameters()
	{
		update_cosmology_meta_parameters();
		update_ellipticity_meta_parameters();
	}
	void calculate_ellipticity_components();
	void update_center_from_pixsrc_coords();

	double potential_numerical(const double, const double);
	double potential_spherical_default(const double x, const double y);
	void deflection_numerical(const double, const double, lensvector&);
	void deflection_spherical_default(const double, const double, lensvector&);
	void hessian_numerical(const double, const double, lensmatrix&);
	void hessian_spherical_default(const double, const double, lensmatrix&);
	void deflection_and_hessian_together(const double x, const double y, lensvector &def, lensmatrix& hess);
	void deflection_and_hessian_numerical(const double x, const double y, lensvector& def, lensmatrix& hess);
	void warn_if_not_converged(const bool& converged, const double &x, const double &y);

	double rmin_einstein_radius; // initial bracket used to find Einstein radius
	double rmax_einstein_radius; // initial bracket used to find Einstein radius
	double einstein_radius_root(const double r);
	double zfac; // for doing calculations at redshift other than the reference redshift
	double mass_intval; // for calculating 3d enclosed mass
	Spline *rho3d_logx_spline;

	double kappa_avg_spherical_integral(const double);
	double mass_enclosed_spherical_integrand(const double);
	double kapavg_spherical_generic(const double rsq);
	double potential_spherical_integral(const double rsq);

	double calculate_scaled_mass_3d_from_kappa(const double r);
	double calculate_scaled_mass_3d_from_analytic_rho3d(const double r);
	double mass3d_r_integrand_analytic(const double r);
	virtual double rho3d_r_integrand_analytic(const double r);

	double rho3d_w_integrand(const double w);
	double mass3d_r_integrand(const double r);
	double mass_inverse_rsq(const double u);
	double half_mass_radius_root(const double r);

	void kappa_deflection_and_hessian_from_elliptical_potential(const double x, const double y, double& kap, lensvector& def, lensmatrix& hess);
	void deflection_from_elliptical_potential(const double x, const double y, lensvector& def);
	void hessian_from_elliptical_potential(const double x, const double y, lensmatrix& hess);
	double kappa_from_elliptical_potential(const double x, const double y);

	public:
	int lens_number;
	bool center_anchored;
	LensProfile* center_anchor_lens;
	bool* anchor_parameter_to_lens;
	LensProfile** parameter_anchor_lens;
	int* parameter_anchor_paramnum;
	double* parameter_anchor_ratio;
	double* parameter_anchor_exponent;
	bool* anchor_parameter_to_source;
	SB_Profile** parameter_anchor_source;
	bool at_least_one_param_anchored;
	bool transform_center_coords_to_pixsrc_frame;

	bool anchor_special_parameter;
	LensProfile* special_anchor_lens;
	double special_anchor_factor;

	static IntegrationMethod integral_method;
	static bool orient_major_axis_north;
	static bool use_ellipticity_components; // if set to true, uses e_1 and e_2 as fit parameters instead of q and theta
	static bool integration_warnings;
	static int default_ellipticity_mode;
	static int default_fejer_nlevels;
	static int fourier_spline_npoints;
	QLens* qlens;
	int ellipticity_mode;
	int parameter_mode; // allows for different parametrizations
	int lens_subclass; // allows for different subclasses of lenses (e.g. multipole order m=0,1,2...); set to -1 if there are no subclasses defined
	std::string subclass_label;
	bool analytic_3d_density; // if true, uses analytic 3d density to find mass_3d(r); if false, finds deprojected 3d profile through integration

	LensProfile() {
		set_null_ptrs_and_values();
		qx_parameter = 1.0;
		setup_lens_properties();
	}
	LensProfile(const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int& nn, const double &acc, const double &qx_in, const double &f_in, QLens*);
	LensProfile(const LensProfile* lens_in);
	~LensProfile() {
		if (param != NULL) delete[] param;
		if (anchor_parameter_to_lens != NULL) delete[] anchor_parameter_to_lens;
		if (parameter_anchor_lens != NULL) delete[] parameter_anchor_lens;
		if (anchor_parameter_to_source != NULL) delete[] anchor_parameter_to_source;
		if (parameter_anchor_source != NULL) delete[] parameter_anchor_source;
		if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
		if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
		if (parameter_anchor_exponent != NULL) delete[] parameter_anchor_exponent;
		if (fourier_integral_left_cos_spline != NULL) delete[] fourier_integral_left_cos_spline;
		if (fourier_integral_left_cos_spline != NULL) delete[] fourier_integral_left_sin_spline;
		if (fourier_integral_left_cos_spline != NULL) delete[] fourier_integral_right_cos_spline;
		if (fourier_integral_left_cos_spline != NULL) delete[] fourier_integral_right_sin_spline;

	}
	void set_null_ptrs_and_values()
	{
		defptr = NULL;
		kapavgptr_rsq_spherical = NULL;
		potptr_rsq_spherical = NULL;
		hessptr = NULL;
		potptr = NULL;
		def_and_hess_ptr = NULL;
		anchor_parameter_to_lens = NULL;
		parameter_anchor_lens = NULL;
		anchor_parameter_to_source = NULL;
		parameter_anchor_source = NULL;
		parameter_anchor_paramnum = NULL;
		at_least_one_param_anchored = false;
		param = NULL;
		parameter_anchor_ratio = NULL;
		parameter_anchor_exponent = NULL;
		zlens = zlens_current = 0;
		zsrc_ref = 0;
		zfac = 1.0;
		sigma_cr = 0;
		fourier_integrals_splined = false;
		fourier_integral_left_cos_spline = NULL;
		fourier_integral_left_sin_spline = NULL;
		fourier_integral_right_cos_spline = NULL;
		fourier_integral_right_sin_spline = NULL;
		use_concentration_prior = false;
		qlens = NULL;
	}
	void setup_cosmology(QLens* lens_in, const double zlens_in, const double zsrc_in);

	// in all derived classes, each of the following function pointers can be redirected if analytic formulas
	// are used instead of the default numerical version
	double (LensProfile::*kapavgptr_rsq_spherical)(const double); // numerical: &LensProfile::kapavg_spherical_integral
	double (LensProfile::*potptr_rsq_spherical)(const double); // numerical: &LensProfile::potential_spherical_integral
	void (LensProfile::*defptr)(const double, const double, lensvector& def); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default
	void (LensProfile::*hessptr)(const double, const double, lensmatrix& hess); // numerical: &LensProfile::hessian_numerical or &LensProfile::hessian_spherical_default
	double (LensProfile::*potptr)(const double, const double); // numerical: &LensProfile::potential_numerical
	void (LensProfile::*def_and_hess_ptr)(const double, const double, lensvector& def, lensmatrix &hess); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default

	bool anchor_center_to_lens(const int &center_anchor_lens_number);
	void delete_center_anchor();
	bool setup_transform_center_coords_to_pixsrc_frame(const double dxc, const double dyc);
	bool enable_ellipticity_gradient(dvector& efunc_params, const int egrad_mode, const int n_bspline_coefs, const dvector& knots, const double ximin = 1e30, const double ximax = 1e30, const double xiref = 1.5, const bool linear_xivals = false, const bool copy_vary_setting = false, boolvector* vary_egrad = NULL);
	void add_fourier_mode(const int m_in, const double amp_in, const double phi_in, const bool vary1, const bool vary2);
	void remove_fourier_modes();
	bool enable_fourier_gradient(dvector& fourier_params, const dvector& knots, const bool copy_vary_settings = false, boolvector* vary_egrad = NULL);
	void find_egrad_paramnums(int& qi, int& qf, int& theta_i, int& theta_f, int& amp_i, int& amp_f);

	virtual void assign_param_pointers();
	virtual void assign_paramnames();
	bool register_vary_flags();
	bool set_vary_flags(boolvector &vary_flags);
	void get_vary_flags(boolvector &vary_flags);
	bool vary_parameters(const boolvector& vary_params_in);
	void set_limits(const dvector& lower, const dvector& upper);
	void set_limits(const dvector& lower, const dvector& upper, const dvector& lower_init, const dvector& upper_init);
	bool set_limits_specific_parameter(const std::string name_in, const double& lower, const double& upper);
	bool get_limits(dvector& lower, dvector& upper, dvector& lower0, dvector& upper0, int &index);
	void shift_angle_90();
	void shift_angle_minus_90();
	void reset_angle_modulo_2pi();
	virtual void set_auto_stepsizes();  // This *must* be redefined in all derived classes
	virtual void set_auto_ranges(); // This *must* be redefined in all derived classes

	void set_geometric_param_auto_stepsizes(int &index);
	void get_auto_stepsizes(dvector& stepsizes, int &index);
	void set_geometric_param_auto_ranges(int param_i);
	void get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index);

	virtual void get_fit_parameters(dvector& fitparams, int &index);
	void get_fit_parameter_names(std::vector<std::string>& paramnames_vary, std::vector<std::string> *latex_paramnames_vary = NULL, std::vector<std::string> *latex_subscripts_vary = NULL);
	virtual double get_parameter(const int i);
	virtual void get_parameters(double* params);
	bool get_specific_parameter(const std::string name_in, double& value);
	virtual void get_parameters_pmode(const int pmode_in, double* params);
	bool update_specific_parameter(const std::string name_in, const double& value);
	bool update_specific_parameter(const int paramnum, const double& value);
	void update_parameters(const double* params);
	void update_fit_parameters(const double* fitparams, int &index, bool& status);
	void update_ellipticity_parameter(const double param);
	void update_anchored_parameters();
	void update_anchor_center();
	virtual void assign_special_anchored_parameters(LensProfile*, const double factor, const bool just_created) {}
	virtual void update_special_anchored_params() {}
	void unassign_special_anchored_parameter() { anchor_special_parameter = false; }
	void copy_special_parameter_anchor(const LensProfile *lens_in);
	void delete_special_parameter_anchor();
	void assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, LensProfile* param_anchor_lens);
	void assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, SB_Profile* param_anchor_source);

	void copy_parameter_anchors(const LensProfile* lens_in);
	void unanchor_parameter(LensProfile* param_anchor_lens);
	void unanchor_parameter(SB_Profile* param_anchor_source);
	void print_parameters();
	std::string mkstring_doub(const double db);
	std::string mkstring_int(const int i);
	std::string get_parameters_string();
	void print_vary_parameters();
	void output_field_in_sci_notation(double* num, std::ofstream& scriptout, const bool space);
	virtual void print_lens_command(std::ofstream& scriptout, const bool use_limits);
	void output_lens_command_nofit(std::string& command);
	virtual void get_auxiliary_parameter(std::string& aux_paramname, double& aux_param) { aux_paramname = ""; aux_param = 0; } // used for outputting information of derived parameters

	// the following function MUST be redefined in all derived classes
	virtual double kappa_rsq(const double rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	// some of these functions can be redefined in the derived classes
	virtual double kappa_rsq_deriv(const double rsq);
	virtual void get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor);
	virtual double get_inner_logslope();
	virtual bool output_cosmology_info(const int lens_number = -1);
	void change_pmode(const int pmode_in);
	bool use_concentration_prior;
	virtual double concentration_prior(); // in NFW-like models, uses mass-concentration relation c(M,z) as a prior
	double average_log_slope(const double rmin, const double rmax);
	double average_log_slope_3d(const double rmin, const double rmax);
	virtual bool calculate_total_scaled_mass(double& total_mass);
	virtual double calculate_scaled_density_3d(const double r, const double tolerance, bool &converged);
	virtual double calculate_scaled_mass_3d(const double r);

	bool calculate_half_mass_radius(double& half_mass_radius, const double mtot_in = -10);
	double mass_rsq(const double rsq);

	virtual double kappa_avg_r(const double r);
	//void plot_kappa_profile(double rmin, double rmax, int steps, const char *kname, const char *kdname = NULL);
	void plot_kappa_profile(double rmin, double rmax, int steps, std::ofstream& kout, std::ofstream& kdout);
	void plot_kappa_profile(const int n_rvals, double* rvals, double* kapvals, double* kapavgvals);
	virtual bool core_present(); // this function is only used for certain derived classes (i.e. specific lens models)
	bool has_kapavg_profile();

	virtual void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess);
	virtual void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	virtual double potential(double x, double y);
	virtual double kappa(double x, double y);
	virtual void deflection(double x, double y, lensvector& def);
	virtual void hessian(double x, double y, lensmatrix& hess); // the Hessian matrix of the lensing potential (*not* the arrival time surface)
	double kappa_from_fourier_modes(const double x, const double y);
	void add_deflection_from_fourier_modes(const double x, const double y, lensvector& def);
	void add_hessian_from_fourier_modes(const double x, const double y, lensmatrix& hess);
	void spline_fourier_mode_integrals(const double rmin, const double rmax);

	public:
	bool isspherical() { return (q==1.0); }
	std::string get_model_name() { return model_name; }
	LensProfileName get_lenstype() { return lenstype; }
	void get_center_coords(double &xc, double &yc) { xc=x_center; yc=y_center; }
	void get_center_coords(lensvector &center) { center[0]=x_center; center[1]=y_center; }
	void get_q_theta(double &q_out, double& theta_out) { q_out=q; theta_out=theta; }
	double get_f_major_axis() { return f_major_axis; }
	double get_redshift() { return zlens; }
	int get_n_params() { return n_params; }
	int get_lensprofile_nparams() { return lensprofile_nparams; }
	int get_n_vary_params() { return n_vary_params; }
	int get_center_anchor_number() { return center_anchor_lens->lens_number; }
	virtual int get_special_parameter_anchor_number() { return -1; } // no special parameters can be center_anchored for the base class
	void set_theta(double theta_in) { theta=theta_in; update_angle_meta_params(); }
	void set_center(double xc_in, double yc_in) { x_center = xc_in; y_center = yc_in; }
	void set_include_limits(bool inc) { include_limits = inc; }
	void set_integral_tolerance(const double acc);
	void set_integral_warnings();
	void set_perturber(bool ispert) { perturber = ispert; }
	void set_lensed_center(bool lensed_xcyc) {
		lensed_center_coords = lensed_xcyc;
		xc_prime = x_center;
		yc_prime = y_center;
		set_center_if_lensed_coords();
		assign_paramnames();
		assign_param_pointers();
	}
	bool output_plates(const int n_plates);
};

struct LensIntegral : public Romberg
{
	LensProfile *profile;
	double xval, yval, xsqval, ysqval, fsqinv, xisq, u, epsilon, qfac, nval_plus_half, mnval_plus_half;
	int nval, emode;
	int mval, fourier_ival; // mval, fourier_ival are used for the Fourier mode integrals
	double phi0; // phi0 is used for Fourier mode integrals if ellipticity gradient is used
	bool cosmode;
	double *cosamps, *sinamps; // used for Fourier modes
	double *gausspoints, *gaussweights;
	double *pat_points, **pat_weights;
	double *pat_funcs;
	double **pat_funcs_mult;
	double *cc_points, **cc_weights;
	double *cc_funcs;
	double **cc_funcs_mult;
	int n_mult;

	LensIntegral()
	{
		cosamps=sinamps=NULL;
		n_mult = 0;
	}
	LensIntegral(LensProfile *profile_in, const double xval_in, const double yval_in, const double q = 1, const int n_mult_in = 0) : xval(xval_in), yval(yval_in)
	{
		cosamps=sinamps=NULL;
		initialize(profile_in,q,n_mult_in);
	}
	void initialize(LensProfile *profile_in, const double q = 1, const int n_mult_in = 0)
	{
		n_mult = n_mult_in;
		profile = profile_in;
		xsqval = xval*xval;
		ysqval = yval*yval;
		epsilon = 1 - q*q;
		emode = profile->ellipticity_mode;
		fsqinv = (emode==0) ? 1 : (emode==1) ? q : (emode==2) ? q : q*q/((1+q*q)/2); 
		phi0 = 0;
		gausspoints = profile->points;
		gaussweights = profile->weights;
		if (profile->integral_method==Gauss_Patterson_Quadrature) {
			pat_points = profile->pat_points;
			pat_weights = profile->pat_weights;
			if (n_mult > 0) {
				pat_funcs_mult = new double*[511];
				for (int i=0; i < 511; i++) pat_funcs_mult[i] = new double[n_mult];
			} else {
				pat_funcs = new double[511];
			}

		} else if (profile->integral_method==Fejer_Quadrature) {
			cc_points = profile->cc_points;
			cc_weights = profile->cc_weights;
			if (n_mult > 0) {
				cc_funcs_mult = new double*[profile->cc_N];
				for (int i=0; i < profile->cc_N; i++) cc_funcs_mult[i] = new double[n_mult];
			} else {
				cc_funcs = new double[profile->cc_N];
			}
		}
	}
	~LensIntegral() {
		if (profile->integral_method==Gauss_Patterson_Quadrature) {
			if (n_mult > 0) {
				for (int i=0; i < 511; i++) delete[] pat_funcs_mult[i];
				delete[] pat_funcs_mult;
			} else {
				delete[] pat_funcs;
			}
		} else if (profile->integral_method==Fejer_Quadrature) {
			if (n_mult > 0) {
				for (int i=0; i < profile->cc_N; i++) delete[] cc_funcs_mult[i];
				delete[] cc_funcs_mult;
			} else {
				delete[] cc_funcs;
			}
		}
	}
	double GaussIntegrate(double (LensIntegral::*func)(const double), const double a, const double b);
	double PattersonIntegrate(double (LensIntegral::*func)(const double), const double a, const double b, bool &converged);
	double FejerIntegrate(double (LensIntegral::*func)(double), double a, double b, bool &converged);

	// Functions for doing multiple integrals simultaneously
	void GaussIntegrate(void (LensIntegral::*func)(const double, double*), const double a, const double b, double* results, const int n_funcs);
	void PattersonIntegrate(void (LensIntegral::*func)(const double, double*), const double a, const double b, double* results, const int n_funcs, bool& converged);
	void FejerIntegrate(void (LensIntegral::*func)(const double, double*), const double a, const double b, double* results, const int n_funcs, bool& converged);

	double i_integrand_prime(const double w);
	double j_integrand_prime(const double w);
	double k_integrand_prime(const double w);
	//double i_integrand_v2(const double w);
	//double j_integrand_v2(const double w);
	//double k_integrand_v2(const double w);
	double i_integral(bool &converged);
	double j_integral(const int nval, bool &converged);
	double k_integral(const int nval, bool &converged);

	double i_integrand_egrad(const double w);
	//double j_integrand_egrad(const double w);
	//double k_integrand_egrad(const double w);
	//double jprime_integrand_egrad(const double w);

	double i_integral_egrad(bool &converged);
	//double j_integral_egrad(const int nval_in, bool &converged);
	//double k_integral_egrad(const int nval_in, bool &converged);
	//double jprime_integral_egrad(const int nval_in, bool &converged);

	void j_integrand_egrad_mult(const double w, double* jint);
	void k_integrand_egrad_mult(const double w, double* kint);
	void jprime_integrand_egrad_mult(const double w, double* jint);
	void j_integral_egrad_mult(double *jint, bool &converged);
	void k_integral_egrad_mult(double *kint, bool &converged);
	void jprime_integral_egrad_mult(double *jint, bool &converged);

	void calculate_fourier_integrals(const int mval_in, const int fourier_ival_in, const bool cosmode_in, const double rval, double& ileft, double& iright, bool &converged);
	double fourier_kappa_perturbation(const double r);
	double ileft_integrand(const double r);
	double iright_integrand(const double u); // here, u = 1/r
	double fourier_kappa_m(const double r, const double phi, const int mval_in, const double fourier_ival_in);

};

class SPLE_Lens : public LensProfile
{
	private:
	double alpha, bprime, sprime; // alpha=2D density log-slope, whereas bprime,sprime are defined along the major axis
	// Note that in emode=1, the actual fit parameters are bprime' = bprime*sqrt(q) and sprime' = sprime*sqrt(q), not bprime and sprime. (See the constructor function for more on how this is implemented.)
	double b, s;
	double qsq, ssq; // used in lensing calculations
	double gamma; // 3D density log-slope, which is an alternative parameter instead of alpha
	static const double euler_mascheroni;
	static const double def_tolerance;

	double kappa_rsq(const double);
	double kappa_rsq_deriv(const double);

	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	double kapavg_spherical_rsq_iso(const double rsq);
	void deflection_elliptical_iso(const double, const double, lensvector&);
	void hessian_elliptical_iso(const double, const double, lensmatrix&);
	double potential_spherical_rsq_iso(const double rsq);
	double potential_elliptical_iso(const double x, const double y);
	void deflection_elliptical_nocore(const double x, const double y, lensvector&);
	void deflection_and_hessian_elliptical_nocore(const double x, const double y, lensvector&, lensmatrix&);
	void hessian_elliptical_nocore(const double x, const double y, lensmatrix& hess);
	double potential_elliptical_nocore(const double x, const double y);
	double potential_spherical_rsq_nocore(const double rsq);
	std::complex<double> deflection_angular_factor(const double &phi);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode_in = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	SPLE_Lens(const int parameter_mode_in = 0)
	{
		set_null_ptrs_and_values();
		setup_lens_properties(parameter_mode_in);
	}
	SPLE_Lens(const double zlens_in, const double zsrc_in, const double &b_in, const double &slope_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, QLens* qlens_in);
	SPLE_Lens(const SPLE_Lens* lens_in);
	void initialize_parameters(const double &bb, const double &slope, const double &ss, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	SPLE_Lens(const double &bb, const double &slope, const double &ss, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in=0) : SPLE_Lens(parameter_mode_in) {
		initialize_parameters(bb,slope,ss,q_in,theta_degrees,xc_in,yc_in);
	}
	SPLE_Lens(SPLE* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double calculate_scaled_mass_3d(const double r);
	bool core_present() { return (sprime==0) ? false : true; }
	double get_inner_logslope() { return -alpha; }
	void get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor);
	bool output_cosmology_info(const int lens_number);
};

class dPIE_Lens : public LensProfile
{
	private:
	double b, s, a; // a is the truncation radius
	double sigma0, mtot, s_kpc, a_kpc; // alternate parametrizations

	// the following are meta-parameters used in lensing calculations
	double bprime, sprime, aprime; // these are the lengths along the major axis
	double qsq, ssq, asq;

	double kappa_rsq(const double);
	double kappa_rsq_deriv(const double);

	double kapavg_spherical_rsq(const double rsq);
	void deflection_elliptical(const double, const double, lensvector&);
	void hessian_elliptical(const double, const double, lensmatrix&);
	double potential_elliptical(const double x, const double y);
	double potential_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode_in = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	bool calculate_tidal_radius;
	int get_special_parameter_anchor_number() { return special_anchor_lens->lens_number; } // no special parameters can be anchored for the base class

	dPIE_Lens()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	dPIE_Lens(const double zlens_in, const double zsrc_in, const double &b_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode, QLens* qlens_in);
	void initialize_parameters(const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	dPIE_Lens(const dPIE_Lens* lens_in);
	dPIE_Lens(dPIE* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void get_parameters_pmode(const int pmode, double* params);
	void set_auto_stepsizes();
	void set_auto_ranges();
	void assign_special_anchored_parameters(LensProfile*, const double factor, const bool just_created);
	void update_special_anchored_params();

	void set_abs_params_from_sigma0();
	void set_abs_params_from_mtot();
	bool output_cosmology_info(const int lens_number = -1);
	double calculate_scaled_mass_3d(const double r);
	bool calculate_total_scaled_mass(double& total_mass);
	void get_einstein_radius(double& r1, double &r2, const double zfactor) { rmin_einstein_radius = 0.01*b; rmax_einstein_radius = 100*b; LensProfile::get_einstein_radius(r1,r2,zfactor); } 
	double get_tidal_radius() { return aprime; }
	bool core_present() { return (sprime==0) ? false : true; }
};

class NFW : public LensProfile
{
	private:
	double ks, rs;
	double m200, c200, rs_kpc; // alternate parametrizations

	double kappa_rsq(const double);
	double kappa_rsq_deriv(const double);
	double lens_function_xsq(const double&);

	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();
	void set_ks_rs_from_m200_c200();
	void set_ks_c200_from_m200_rs();

	public:
	NFW()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	NFW(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, QLens* qlens_in);
	void initialize_parameters(const double &p1_in, const double &p2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	NFW(const NFW* lens_in);
	NFW(NFW_Source* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	void assign_special_anchored_parameters(LensProfile*, const double factor, const bool just_created);
	void update_special_anchored_params();
	void get_parameters_pmode(const int pmode_in, double* params);

	double calculate_scaled_mass_3d(const double r);
	double concentration_prior();
	bool output_cosmology_info(const int lens_number = -1);
};

class Truncated_NFW : public LensProfile
{
	// This profile is the same as NFW, times a factor (1+(r/rt)^2)^-2 which smoothly truncates the halo (prescription from Baltz, Marshall & Oguri (2008))
	private:
	double ks, rs, rt;
	double m200, c200, rs_kpc, rt_kpc, tau200, tau_s; // alternate parametrizations

	double kappa_rsq(const double);
	double lens_function_xsq(const double&);
	double kapavg_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();
	void set_ks_rs_from_m200_c200();
	void set_ks_c200_from_m200_rs();

	public:
	Truncated_NFW(const int parameter_mode = 0, const int truncation_mode = 0)
	{
		set_null_ptrs_and_values();
		setup_lens_properties(parameter_mode,truncation_mode);
	}
	Truncated_NFW(const double zlens_in, const double zsrc_in, const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int truncation_mode_in, const int parameter_mode_in, QLens* qlens_in);
	void initialize_parameters(const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Truncated_NFW(const Truncated_NFW* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	void assign_special_anchored_parameters(LensProfile*, const double factor, const bool just_created);
	void update_special_anchored_params();
	void get_parameters_pmode(const int pmode_in, double* params);

	bool output_cosmology_info(const int lens_number = -1);
};

class Cored_NFW : public LensProfile
{
	// This profile goes like 1/(r+rc)/(r+rs)^2
	private:
	double ks, rs, rc;
	double m200, c200, beta, rs_kpc, rc_kpc; // alternate parametrization

	double kappa_rsq(const double);
	double kappa_rsq_deriv(const double rsq);
	double lens_function_xsq(const double&);
	double kapavg_spherical_rsq(const double rsq);
	//double potential_spherical_rsq(const double rsq);
	//double potential_lens_function_xsq(const double&);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();
	void set_ks_rs_from_m200_c200_beta();
	void set_ks_rs_from_m200_c200_rckpc();
	void set_ks_c200_from_m200_rs();

	public:
	Cored_NFW()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	Cored_NFW(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &rt_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, QLens* qlens_in);
	void initialize_parameters(const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Cored_NFW(const Cored_NFW* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	void assign_special_anchored_parameters(LensProfile*, const double factor, const bool just_created);
	void update_special_anchored_params();
	void get_parameters_pmode(const int pmode_in, double* params);
	double calculate_scaled_mass_3d(const double r);
	bool output_cosmology_info(const int lens_number);
};

class Hernquist : public LensProfile
{
	private:
	double ks, rs;

	double kappa_rsq(const double);
	double kappa_rsq_deriv(const double);
	double lens_function_xsq(const double);
	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	Hernquist()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	Hernquist(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const int &nn, const double &acc, QLens*);
	void initialize_parameters(const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Hernquist(const Hernquist* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
};

class ExpDisk : public LensProfile
{
	private:
	double k0, R_d;

	double kappa_rsq(const double);
	double kappa_rsq_deriv(const double);
	double kapavg_spherical_rsq(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	ExpDisk()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	ExpDisk(const double zlens_in, const double zsrc_in, const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, QLens*);
	void initialize_parameters(const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	ExpDisk(const ExpDisk* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	bool calculate_total_scaled_mass(double& total_mass);
};

class Shear : public LensProfile
{
	private:
	double shear, theta_eff;
	double shear1, shear2; // used when shear_components is turned on
	double kappa_rsq(const double) { return 0; }
	double kappa_rsq_deriv(const double) { return 0; }
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	void set_angle_from_components(const double &comp_x, const double &comp_y);

	public:
	Shear()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	Shear(const double zlens_in, const double zsrc_in, const double &shear_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens*);
	void initialize_parameters(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in);
	Shear(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in) : Shear() {
		initialize_parameters(shear_p1_in,shear_p2_in,xc_in,yc_in);
	}
	Shear(const Shear* lens_in);
	static bool use_shear_component_params; // if set to true, uses shear_1 and shear_2 as fit parameters instead of gamma and theta
	static bool angle_points_towards_perturber; // direction of hypothetical perturber differs from shear angle by 90 degrees

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	// here the base class deflection/hessian functions are overloaded because the angle is put in explicitly in the formulas (no rotation of the coordinates is needed)
	double potential(double, double);
	void potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess);
	void deflection(double, double, lensvector&);
	void hessian(double, double, lensmatrix&);
	void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess)
	{
		kap = 0;
		potential_derivatives(x,y,def,hess);
	}

	double kappa(double, double) { return 0; }
	void get_einstein_radius(double& r1, double& r2, const double zfactor) { r1=0; r2=0; }
};

class Multipole : public LensProfile
{
	private:
	int m;
	double A_n, n, theta_eff;
	bool kappa_multipole; // specifies whether it is a multipole in the potential or in kappa
	bool sine_term; // specifies whether it is a sine or cosine multipole term

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:

	Multipole()
	{
		sine_term = false; // default
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	Multipole(const double zlens_in, const double zsrc_in, const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, QLens*, const bool sine=false);
	void initialize_parameters(const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, const bool sine);
	Multipole(const Multipole* lens_in);

	// here the base class deflection/hessian functions are overloaded because the angle is put in explicitly in the formulas (no rotation of the coordinates is needed)
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess);
	void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	void deflection(double, double, lensvector&);
	void hessian(double, double, lensmatrix&);
	double potential(double, double);
	double kappa(double, double);
	double deflection_m0_spherical_r(const double r);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	void get_einstein_radius(double& r1, double& r2, const double zfactor);
};

class PointMass : public LensProfile
{
	private:
	double b; // Einstein radius of point mass
	double mtot; // alternative parameterization

	double kappa_rsq(const double rsq) { return 0; }
	double kappa_rsq_deriv(const double rsq) { return 0; }
	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	PointMass()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	PointMass(const double zlens_in, const double zsrc_in, const double &bb, const double &xc_in, const double &yc_in, const int parameter_mode_in, QLens*);
	void initialize_parameters(const double &p_in, const double &xc_in, const double &yc_in);
	PointMass(const PointMass* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double potential(double, double);
	double kappa(double, double);
	double kappa_avg_r(const double r);

	// here the base class deflection/hessian functions are overloaded because the potential has circular symmetry (no rotation of the coordinates is needed)
	void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess)
	{
		kap = 0;
		potential_derivatives(x,y,def,hess);
	}

	void deflection(double, double, lensvector&);
	void hessian(double, double, lensmatrix&);

	bool calculate_total_scaled_mass(double& total_mass);
	double calculate_scaled_mass_3d(const double r);
	void get_einstein_radius(double& r1, double& r2, const double zfactor);
};

class CoreCusp : public LensProfile
{
	private:
	double n, gamma, a, s, k0;
	static const double nstep; // this is for calculating the n=3 case, which requires extrapolation since F21 is singular for n=3
	static const double digamma_three_halves; // needed for the n=3 case
	bool set_k0_by_einstein_radius;
	double einstein_radius;
	double core_enclosed_mass;
	double digamma_term, beta_p1, beta_p2; // used for calculations of kappa, dkappa
	double r200_const;

	double kappa_rsq(const double);
	double kappa_rsq_deriv(const double rsq);
	double kapavg_spherical_rsq(const double rsq);

	double kappa_rsq_nocore(const double rsq_prime, const double atilde);
	double enclosed_mass_spherical_nocore(const double rsq_prime, const double atilde) { return enclosed_mass_spherical_nocore(rsq_prime,atilde,n); }
	double enclosed_mass_spherical_nocore(const double rsq_prime, const double atilde, const double nprime);
	double enclosed_mass_spherical_nocore_n3(const double rsq_prime, const double atilde, const double nprime);
	double enclosed_mass_spherical_nocore_limit(const double rsq, const double atilde, const double n_stepsize);
	double kappa_rsq_deriv_nocore(const double rsq_prime, const double atilde);
	void set_core_enclosed_mass();

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();
	void get_auxiliary_parameter(std::string& aux_paramname, double& aux_param) { if (set_k0_by_einstein_radius) aux_paramname = "k0"; aux_param = k0; }

	public:
	bool calculate_tidal_radius;
	int get_special_parameter_anchor_number() { return special_anchor_lens->lens_number; } // no special parameters can be anchored for the base class

	CoreCusp()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	CoreCusp(const double zlens_in, const double zsrc_in, const double &k0_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, QLens*);
	void initialize_parameters(const double &mass_param_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	CoreCusp(const CoreCusp* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	void assign_special_anchored_parameters(LensProfile*, const double factor, const bool just_created);
	void update_special_anchored_params();

	double rho3d_r_integrand_analytic(const double r);
	bool output_cosmology_info(const int lens_number);
	double r200_root_eq(const double r);
	bool core_present() { return (s==0) ? false : true; }
};

class SersicLens : public LensProfile
{
	friend class SB_Profile;
	friend class Sersic;

	private:
	double kappa0, b, n;
	double re; // effective radius
	double mstar; // total stellar mass (alternate parameterization)
	double def_factor; // used to calculate the spherical deflection

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq);
	double kapavg_spherical_rsq(const double rsq);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:

	SersicLens()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	SersicLens(const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, QLens*);
	void initialize_parameters(const double &p1_in, const double &Re_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	SersicLens(const SersicLens* lens_in);
	SersicLens(Sersic* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	bool output_cosmology_info(const int lens_number);
};

class DoubleSersicLens : public LensProfile
{
	friend class SB_Profile;
	friend class DoubleSersic;

	private:
	double kappa0, delta_k;
	double kappa0_1, b1, n1;
	double kappa0_2, b2, n2;
	double Reff1, Reff2; // effective radiukappa
	double mstar; // total stellar mass (alternate parameterization to kappa0)
	//double def_factor; // used to calculate the spherical deflection

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq);
	//double kapavg_spherical_rsq(const double rsq);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	//void set_model_specific_integration_pointers();

	public:

	DoubleSersicLens()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	DoubleSersicLens(const double zlens_in, const double zsrc_in, const double &p1_in, const double &delta_k_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, QLens*);
	void initialize_parameters(const double &p1_in, const double &delta_k_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	DoubleSersicLens(const DoubleSersicLens* lens_in);
	DoubleSersicLens(DoubleSersic* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	bool output_cosmology_info(const int lens_number);
};

class Cored_SersicLens : public LensProfile
{
	private:
	double kappa0, b, n;
	double re; // effective radius
	double rc; // core radius
	double mstar; // total stellar mass (alternate parameterization)
	double def_factor; // used to calculate the spherical deflection

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq);
	double kapavg_spherical_rsq(const double rsq);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:

	Cored_SersicLens()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	Cored_SersicLens(const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, QLens*);
	void initialize_parameters(const double &p1_in, const double &Re_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Cored_SersicLens(const Cored_SersicLens* lens_in);
	Cored_SersicLens(Cored_Sersic* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	bool output_cosmology_info(const int lens_number);
};

/*
class DoubleSersicLens : public LensProfile
{
	friend class SB_Profile;
	friend class DoubleSersic;

	private:
	double kappa0_1, b1, n1;
	double kappa0_2, b2, n2;
	double re1, re2; // effective radius
	double def_factor; // used to calculate the spherical deflection

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq);
	double kapavg_spherical_rsq(const double rsq);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:

	DoubleSersicLens()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	DoubleSersicLens(const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, QLens*);
	void initialize_parameters(const double &p1_in, const double &Re_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	DoubleSersicLens(const DoubleSersicLens* lens_in);
	DoubleSersicLens(DoubleSersic* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
};
*/



class MassSheet : public LensProfile
{
	private:
	double kext;

	double kappa_rsq(const double rsq) { return 0; }
	double kappa_rsq_deriv(const double rsq) { return 0; }
	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	MassSheet()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	MassSheet(const double zlens_in, const double zsrc_in, const double &kext_in, const double &xc_in, const double &yc_in, QLens*);
	void initialize_parameters(const double &kext_in, const double &xc_in, const double &yc_in);
	MassSheet(const MassSheet* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters() {}
	void set_auto_stepsizes();
	void set_auto_ranges();

	double potential(double, double);
	double kappa(double, double);

	// here the base class deflection/hessian functions are overloaded because the potential has circular symmetry (no rotation of the coordinates is needed)
	void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess)
	{
		kap = kext;
		potential_derivatives(x,y,def,hess);
	}

	void deflection(double, double, lensvector&);
	void hessian(double, double, lensmatrix&);

	void get_einstein_radius(double& r1, double& r2, const double zfactor) { r1=0; r2=0; }
};

class Deflection : public LensProfile
{
	private:
	double def_x, def_y;

	double kappa_rsq(const double rsq) { return 0; }
	double kappa_rsq_deriv(const double rsq) { return 0; }
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	Deflection()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	Deflection(const double zlens_in, const double zsrc_in, const double &defx_in, const double &defy_in, QLens*);
	void initialize_parameters(const double &defx_in, const double &defy_in);
	Deflection(const Deflection* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters() {}
	void set_auto_stepsizes();
	void set_auto_ranges();

	double potential(double, double);
	double kappa(double, double);

	// here the base class deflection/hessian functions are overloaded because the potential has circular symmetry (no rotation of the coordinates is needed)
	void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess)
	{
		kap = 0;
		potential_derivatives(x,y,def,hess);
	}

	void deflection(double, double, lensvector&);
	void hessian(double, double, lensmatrix&);

	void get_einstein_radius(double& r1, double& r2, const double zfactor) { r1=0; r2=0; }
};

class TopHatLens : public LensProfile
{
	private:
	double kap0, xi0;

	double kappa_rsq(const double);
	double kappa_rsq_deriv(const double);

	double kapavg_spherical_rsq(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	TopHatLens()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	TopHatLens(const double zlens_in, const double zsrc_in, const double &kap0_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, QLens*);
	void initialize_parameters(const double &kap0_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	TopHatLens(const TopHatLens* lens_in);

	void deflection_analytic(const double, const double, lensvector&);
	void hessian_analytic(const double, const double, lensmatrix&);
	double potential_analytic(const double x, const double y);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	//bool calculate_total_scaled_mass(double& total_mass);
};



class Tabulated_Model : public LensProfile
{
	private:
	double kscale;
	double rscale, rscale0, rscale_factor; // rscale is a parameter that can be varied, whereas rscale0 is set when the table is created or loaded from file
	int grid_logr_N, grid_phi_N;
	double grid_logrlength;
	double *grid_logrvals, *grid_phivals;
	double **kappa_vals, **pot_vals, **defx, **defy, **hess_xx, **hess_yy, **hess_xy;
	std::string original_lens_command; // used for saving commands to reproduce this model
	double original_kscale, original_rscale;
	bool loaded_from_file;

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq) { return 0; } // will not be used

	public:
	Tabulated_Model()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	Tabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, QLens*);
	Tabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double &xc, const double &yc, std::ifstream& tabfile, const std::string& tab_filename, QLens*);

	Tabulated_Model(const Tabulated_Model* lens_in);
	~Tabulated_Model();
	void output_tables(const std::string tabfile_root);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	void print_lens_command(std::ofstream& scriptout, const bool use_limits);

	double potential(double, double);
	void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess);
	double kappa(double, double);
	void deflection(double, double, lensvector&);
	void hessian(double, double, lensmatrix&);

	//void get_einstein_radius(double& r1, double& r2, const double zfactor) { r1=0; r2=0; } // cannot use this
};

class QTabulated_Model : public LensProfile
{
	private:
	double kscale;
	double rscale, rscale0, rscale_factor; // rscale is a parameter that can be varied, whereas rscale0 is set when the table is created or loaded from file
	int grid_logr_N, grid_phi_N, grid_q_N;
	int kval;
	bool original_emode;
	double ww, WW;
	double grid_logrlength, grid_qlength;
	double *grid_logrvals, *grid_phivals, *grid_qvals;
	double ***kappa_vals, ***pot_vals, ***defx, ***defy, ***hess_xx, ***hess_yy, ***hess_xy;

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq) { return 0; } // will not be used

	public:
	QTabulated_Model()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	QTabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, const double qmin, const int q_N, QLens*);
	QTabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, const double &xc, const double &yc, std::ifstream& tabfile, QLens*);

	QTabulated_Model(const QTabulated_Model* lens_in);
	~QTabulated_Model();
	void output_tables(const std::string tabfile_root);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double potential(double, double);
	void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess);
	double kappa(double, double);
	void deflection(double, double, lensvector&);
	void hessian(double, double, lensmatrix&);

	//void get_einstein_radius(double& r1, double& r2, const double zfactor) { r1=0; r2=0; } // cannot use this
};

// Model for testing purposes; can also be used as a template for a new lens model
class TestModel : public LensProfile
{
	private:

	double kappa_rsq(const double);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);

	// The following functions can be overloaded, but don't necessarily have to be
	//double kappa_rsq_deriv(const double rsq); // optional
	//void deflection(double, double, lensvector&);
	//void hessian(double, double, lensmatrix&);

	public:
	TestModel()
	{
		set_null_ptrs_and_values();
		setup_lens_properties();
	}
	TestModel(const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc);

	//double kappa(double, double);
	//void deflection(double, double, lensvector&);
	//void hessian(double, double, lensmatrix&);
	//double potential(double, double);
};

#endif // PROFILE_H
