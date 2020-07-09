#ifndef PROFILE_H
#define PROFILE_H

#include "mathexpr.h"
#include "gauss.h"
#include "spline.h"
#include "brent.h"
#include "lensvec.h"
#include "romberg.h"
#include "cosmo.h"
#include <iostream>
#include <vector>
#include <complex>

#ifdef USE_PYBIND
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

using namespace std;

enum IntegrationMethod { Romberg_Integration, Gaussian_Quadrature, Gauss_Patterson_Quadrature };
enum LensProfileName
{
	KSPLINE,
	ALPHA,
	PJAFFE,
	nfw,
	TRUNCATED_nfw,
	CORED_nfw,
	HERNQUIST,
	EXPDISK,
	CORECUSP,
	SERSIC_LENS,
	CORED_SERSIC_LENS,
	MULTIPOLE,
	PTMASS,
	SHEAR,
	SHEET,
	DEFLECTION,
	TABULATED,
	QTABULATED,
	TESTMODEL
};

struct LensIntegral;
class Lens;

class LensProfile : public Romberg, public GaussLegendre, public GaussPatterson, public Brent
{
	friend class LensIntegral;
	friend class Lens;
	friend class SB_Profile;

	// the following private declarations are specific to LensProfile and not derived classes
	private:
	Spline kspline;
	double kappa_splint(double);
	double kappa_rsq_deriv_splint(double);
	double qx_parameter, f_parameter;

	protected:
	LensProfileName lenstype;
	bool center_defined;
	double zlens, zsrc_ref;
	double zlens_current; // used to check if zlens has been changed, in which case sigma_cr, etc. are updated
	double sigma_cr, kpc_to_arcsec;
	double q, theta, x_center, y_center; // four base parameters, which can be added to in derived lens models
	double f_major_axis; // used for defining elliptical radius; set in function set_q(q)
	double epsilon, epsilon2; // used for defining ellipticity, or else components of ellipticity (epsilon, epsilon2)
	double costheta, sintheta;
	double integral_tolerance;
	double theta_eff; // used for intermediate calculations if ellipticity components are being used
	double **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model
	bool perturber;

	int n_params, n_vary_params;
	int angle_paramnum; // used to keep track of angle parameter so it can be easily converted to degrees and displayed
	int ellipticity_paramnum; // used to keep track of ellipticity parameter (this feature is used only by qtab models)
	boolvector vary_params;
	string model_name;
	string special_parameter_command;
	vector<string> paramnames;
	vector<string> latex_paramnames, latex_param_subscripts;
	boolvector set_auto_penalty_limits;
	dvector penalty_upper_limits, penalty_lower_limits;
	dvector stepsizes;
	bool include_limits;
	dvector lower_limits, upper_limits;
	dvector lower_limits_initial, upper_limits_initial;

	virtual void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void setup_base_lens(const int np, const bool is_elliptical_lens, const int pmode_in = 0, const int subclass_in = -1);
	void copy_base_lensdata(const LensProfile* lens_in);

	void set_nparams_and_anchordata(const int &n_params_in);
	void set_geometric_param_pointers(int qi);
	void set_geometric_paramnames(int qi);
	void set_angle(const double &theta_degrees);
	void set_angle_radians(const double &theta_in);
	void set_ellipticity_parameter(const double &q_in);
	void rotate(double&, double&);
	void rotate_back(double&, double&);

	void set_geometric_parameters(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	void set_angle_from_components(const double &comp_x, const double &comp_y);
	void set_default_base_settings(const int &nn, const double &acc);
	void set_integration_pointers();
	virtual void set_model_specific_integration_pointers();
	void update_meta_parameters_and_pointers();
	void update_angle_meta_params();
	void update_ellipticity_meta_parameters();
	void update_zlens_meta_parameters();
	virtual void update_meta_parameters()
	{
		update_zlens_meta_parameters();
		update_ellipticity_meta_parameters();
	}
	void calculate_ellipticity_components();

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

	double j_integral(const double, const double, const int, bool &converged);
	double k_integral(const double, const double, const int, bool &converged);
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

	// remove these later if it doesn't pan out
	void deflection_from_elliptical_potential_experimental(const double x, const double y, lensvector& def);
	void hessian_from_elliptical_potential_experimental(const double x, const double y, lensmatrix& hess);
	double kappa_from_elliptical_potential_experimental(const double x, const double y);
	double test_resq(const double x, const double y);
	double test_defx(const double x, const double y);
	double test_defy(const double x, const double y);

	public:
	int lens_number;
	bool center_anchored;
	LensProfile* center_anchor_lens;
	bool* anchor_parameter;
	LensProfile** parameter_anchor_lens;
	int* parameter_anchor_paramnum;
	double* parameter_anchor_ratio;
	double* parameter_anchor_exponent;

	bool anchor_special_parameter;
	LensProfile* special_anchor_lens;

	static IntegrationMethod integral_method;
	static bool orient_major_axis_north;
	static bool use_ellipticity_components; // if set to true, uses e_1 and e_2 as fit parameters instead of gamma and theta
	static bool output_integration_errors; // if set to true, uses e_1 and e_2 as fit parameters instead of gamma and theta
	static int default_ellipticity_mode;
	Lens* lens;
	int ellipticity_mode;
	int parameter_mode; // allows for different parametrizations
	int lens_subclass; // allows for different subclasses of lenses (e.g. multipole order m=0,1,2...); set to -1 if there are no subclasses defined
	string subclass_label;
	bool analytic_3d_density; // if true, uses analytic 3d density to find mass_3d(r); if false, finds deprojected 3d profile through integration

	LensProfile() : defptr(0), kapavgptr_rsq_spherical(0), potptr_rsq_spherical(0), hessptr(0), potptr(0), def_and_hess_ptr(0), qx_parameter(1), anchor_parameter(0), parameter_anchor_lens(0), parameter_anchor_paramnum(0), param(0), parameter_anchor_ratio(0), parameter_anchor_exponent(0), lens(0)
	{
		setup_lens_properties();
		set_default_base_settings(20,5e-3); // is this really necessary? check...
		zfac = 1.0;
		zlens = zlens_current = 0;
	}
	LensProfile(const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int& nn, const double &acc, const double &qx_in, const double &f_in, Lens*);
	LensProfile(const LensProfile* lens_in);
	~LensProfile() {
		if (param != NULL) delete[] param;
		if (anchor_parameter != NULL) delete[] anchor_parameter;
		if (parameter_anchor_lens != NULL) delete[] parameter_anchor_lens;
		if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
		if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
		if (parameter_anchor_exponent != NULL) delete[] parameter_anchor_exponent;
	}
	void setup_cosmology(Lens* lens_in, const double zlens_in, const double zsrc_in);

	// in all derived classes, each of the following function pointers can be redirected if analytic formulas
	// are used instead of the default numerical version
	double (LensProfile::*kapavgptr_rsq_spherical)(const double); // numerical: &LensProfile::kapavg_spherical_integral
	double (LensProfile::*potptr_rsq_spherical)(const double); // numerical: &LensProfile::potential_spherical_integral
	void (LensProfile::*defptr)(const double, const double, lensvector& def); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default
	void (LensProfile::*hessptr)(const double, const double, lensmatrix& hess); // numerical: &LensProfile::hessian_numerical or &LensProfile::hessian_spherical_default
	double (LensProfile::*potptr)(const double, const double); // numerical: &LensProfile::potential_numerical
	void (LensProfile::*def_and_hess_ptr)(const double, const double, lensvector& def, lensmatrix &hess); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default

	void anchor_center_to_lens(LensProfile** center_anchor_list, const int &center_anchor_lens_number);
	void delete_center_anchor();
	virtual void assign_param_pointers();
	virtual void assign_paramnames();
	bool vary_parameters(const boolvector& vary_params_in);
	void set_limits(const dvector& lower, const dvector& upper);
	void set_limits(const dvector& lower, const dvector& upper, const dvector& lower_init, const dvector& upper_init);
	bool get_limits(dvector& lower, dvector& upper, dvector& lower0, dvector& upper0, int &index);
	void shift_angle_90();
	void shift_angle_minus_90();
	void reset_angle_modulo_2pi();
	virtual void set_auto_stepsizes();  // This *must* be redefined in all derived classes
	virtual void set_auto_ranges(); // This *must* be redefined in all derived classes

	void set_auto_eparam_stepsizes(int eparam1_i, int eparam2_i);
	void get_auto_stepsizes(dvector& stepsizes, int &index);
	void set_geometric_param_auto_ranges(int param_i);
	void get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index);

	virtual void get_fit_parameters(dvector& fitparams, int &index);
	void get_fit_parameter_names(vector<string>& paramnames_vary, vector<string> *latex_paramnames_vary = NULL, vector<string> *latex_subscripts_vary = NULL);
	virtual void get_parameters(double* params);
	virtual void get_parameters_pmode(const int pmode_in, double* params);
	bool update_specific_parameter(const string name_in, const double& value);
	virtual void update_parameters(const double* params);
	virtual void update_fit_parameters(const double* fitparams, int &index, bool& status);
	void update_ellipticity_parameter(const double param);
	void update_anchored_parameters();
	void update_anchor_center();
	virtual void assign_special_anchored_parameters(LensProfile*, const double factor, const bool just_created) {}
	virtual void update_special_anchored_params() {}
	void unassign_special_anchored_parameter() { anchor_special_parameter = false; }
	void copy_special_parameter_anchor(const LensProfile *lens_in);
	void delete_special_parameter_anchor();
	void assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, LensProfile* param_anchor_lens);
	void copy_parameter_anchors(const LensProfile* lens_in);
	void unanchor_parameter(LensProfile* param_anchor_lens);
	void print_parameters();
	void print_vary_parameters();
	void output_field_in_sci_notation(double* num, ofstream& scriptout, const bool space);
	virtual void print_lens_command(ofstream& scriptout, const bool use_limits);
	void output_lens_command_nofit(string& command);
	virtual void get_auxiliary_parameter(string& aux_paramname, double& aux_param) { aux_paramname = ""; aux_param = 0; } // used for outputting information of derived parameters

	// the following function MUST be redefined in all derived classes
	virtual double kappa_rsq(const double rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	// these functions can be redefined in the derived classes, but don't have to be
	virtual double kappa_rsq_deriv(const double rsq);
	virtual void get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor);
	virtual double get_inner_logslope();
	virtual bool output_cosmology_info(const int lens_number = -1);
	double average_log_slope(const double rmin, const double rmax);
	double average_log_slope_3d(const double rmin, const double rmax);
	virtual bool calculate_total_scaled_mass(double& total_mass);
	virtual double calculate_scaled_density_3d(const double r, const double tolerance, bool &converged);
	virtual double calculate_scaled_mass_3d(const double r);

	bool calculate_half_mass_radius(double& half_mass_radius, const double mtot_in = -10);
	double mass_rsq(const double rsq);
	double get_theta() {return theta;} //  @EXCEL: NEED TO REMOVE

	virtual double kappa_avg_r(const double r);
	void plot_kappa_profile(double rmin, double rmax, int steps, const char *kname, const char *kdname = NULL);
	void plot_kappa_profile(const int n_rvals, double* rvals, double* kapvals, double* kapavgvals);
	virtual bool core_present(); // this function is only used for certain derived classes (i.e. specific lens models)
	bool has_kapavg_profile();

	virtual void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess);
	virtual void potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess);
	virtual double potential(double, double);
	virtual double kappa(double x, double y);
	virtual void deflection(double, double, lensvector&);
	virtual void hessian(double, double, lensmatrix&); // the Hessian matrix of the lensing potential (*not* the arrival time surface)

	bool isspherical() { return (q==1.0); }
	string get_model_name() { return model_name; }
	LensProfileName get_lenstype() { return lenstype; }
	void get_center_coords(double &xc, double &yc) { xc=x_center; yc=y_center; }
	void get_center_coords(lensvector &center) { center[0]=x_center; center[1]=y_center; }
	void get_q_theta(double &q_out, double& theta_out) { q_out=q; theta_out=theta; }
	double get_f_major_axis() { return f_major_axis; }
	double get_redshift() { return zlens; }
	int get_n_params() { return n_params; }
	int get_n_vary_params() { return n_vary_params; }
	int get_center_anchor_number() { return center_anchor_lens->lens_number; }
	virtual int get_special_parameter_anchor_number() { return -1; } // no special parameters can be center_anchored for the base class
	void set_theta(double theta_in) { theta=theta_in; update_angle_meta_params(); }
	void set_center(double xc_in, double yc_in) { x_center = xc_in; y_center = yc_in; }
	void set_include_limits(bool inc) { include_limits = inc; }
	void set_integral_tolerance(const double acc) { integral_tolerance = acc; SetGaussPatterson(acc,true); }
	void set_perturber(bool ispert) { perturber = ispert; }
};

struct LensIntegral : public Romberg
{
	LensProfile *profile;
	double xsqval, ysqval, fsqinv, xisq, u, one_minus_qsq, qfac, nval_plus_half;
	double *gausspoints, *gaussweights;
	double *pat_points, **pat_weights;
	double pat_funcs[511];
	int n_gausspoints;

	LensIntegral(LensProfile *profile_in, double xsqval_in, double ysqval_in, double q, int nval_in) : profile(profile_in), xsqval(xsqval_in), ysqval(ysqval_in)
	{
		one_minus_qsq = 1 - q*q;
		nval_plus_half = nval_in + 0.5;
		if (q != 1.0) fsqinv = 1.0/SQR(profile->f_major_axis);
		else fsqinv = 1.0; // even if the lens model itself is not spherical, we can still get the spherical calculations if needed by setting f=1 here
		n_gausspoints = profile->numberOfPoints;
		gausspoints = profile->points;
		gaussweights = profile->weights;
		pat_points = profile->pat_points;
		pat_weights = profile->pat_weights;
	}
	double GaussIntegrate(double (LensIntegral::*func)(const double), const double a, const double b);
	double PattersonIntegrate(double (LensIntegral::*func)(double), double a, double b, bool &converged);

	double i_integrand_prime(const double w);
	double j_integrand_prime(const double w);
	double k_integrand_prime(const double w);
	double i_integral(bool &converged);
	double j_integral(bool &converged);
	double k_integral(bool &converged);
};

class Alpha : public LensProfile
{
	private:
	double alpha, bprime, sprime;
	// Note that the actual fit parameters are bprime' = bprime*sqrt(q) and sprime' = sprime*sqrt(q), not bprime and sprime. (See the constructor function for more on how this is implemented.)
	double b, s;
	double qsq, ssq; // used in lensing calculations
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
	complex<double> deflection_angular_factor(const double &phi);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	Alpha() : LensProfile()
	{
		setup_lens_properties();
	}
	Alpha(const double zlens_in, const double zsrc_in, const double &b_in, const double &alpha_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens* lens_in);
	Alpha(const Alpha* lens_in);
	void initialize_parameters(const double &bb, const double &aa, const double &ss, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Alpha(const double &bb, const double &aa, const double &ss, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in) : Alpha() { initialize_parameters(bb,aa,ss,q_in,theta_degrees,xc_in,yc_in); }

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

class PseudoJaffe : public LensProfile
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

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	bool calculate_tidal_radius;
	int get_special_parameter_anchor_number() { return special_anchor_lens->lens_number; } // no special parameters can be anchored for the base class

	PseudoJaffe() : LensProfile()
	{
		setup_lens_properties();
	}
	PseudoJaffe(const double zlens_in, const double zsrc_in, const double &b_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode, Lens* lens_in);
	PseudoJaffe(const PseudoJaffe* lens_in);

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
	double median_c_factor; // used if concentration is set to factor*median value

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
	NFW() : LensProfile()
	{
		setup_lens_properties();
	}
	NFW(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in);
	NFW(const NFW* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	void assign_special_anchored_parameters(LensProfile*, const double factor, const bool just_created);
	void update_special_anchored_params();
	void get_parameters_pmode(const int pmode_in, double* params);

	double calculate_scaled_mass_3d(const double r);
	bool output_cosmology_info(const int lens_number = -1);
};

class Truncated_NFW : public LensProfile
{
	// This profile is the same as NFW, times a factor (1+(r/rt)^2)^-2 which smoothly truncates the halo (prescription from Baltz, Marshall & Oguri (2008))
	private:
	double ks, rs, rt;
	double m200, c200, rs_kpc, rt_kpc, tau200, tau_s; // alternate parametrizations
	double median_c_factor; // used if concentration is set to factor*median value

	double kappa_rsq(const double);
	double lens_function_xsq(const double&);
	double kapavg_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();
	void set_ks_rs_from_m200_c200();
	void set_ks_c200_from_m200_rs();

	public:
	Truncated_NFW() : LensProfile()
	{
		setup_lens_properties();
	}
	Truncated_NFW(const double zlens_in, const double zsrc_in, const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int truncation_mode_in, const int parameter_mode_in, Lens* lens_in);
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
	double median_c_factor;

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
	Cored_NFW() : LensProfile()
	{
		setup_lens_properties();
	}
	Cored_NFW(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &rt_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in);
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
	Hernquist() : LensProfile()
	{
		setup_lens_properties();
	}
	Hernquist(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens*);
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
	ExpDisk() : LensProfile()
	{
		setup_lens_properties();
	}
	ExpDisk(const double zlens_in, const double zsrc_in, const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens*);
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
	Shear() : LensProfile()
	{
		setup_lens_properties();
	}
	Shear(const double zlens_in, const double zsrc_in, const double &shear_in, const double &theta_degrees, const double &xc_in, const double &yc_in, Lens*);
	Shear(const Shear* lens_in);
	void initialize_parameters(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in);
	Shear(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in) : Shear() {
		initialize_parameters(shear_p1_in,shear_p2_in,xc_in,yc_in);
	}
	static bool use_shear_component_params; // if set to true, uses shear_1 and shear_2 as fit parameters instead of gamma and theta

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

	Multipole() : LensProfile()
	{
		setup_lens_properties();
	}
	Multipole(const double zlens_in, const double zsrc_in, const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, Lens*, const bool sine=false);
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
	PointMass() : LensProfile()
	{
		setup_lens_properties();
	}
	PointMass(const double zlens_in, const double zsrc_in, const double &bb, const double &xc_in, const double &yc_in, const int parameter_mode_in, Lens*);
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
	void get_auxiliary_parameter(string& aux_paramname, double& aux_param) { if (set_k0_by_einstein_radius) aux_paramname = "k0"; aux_param = k0; }

	public:
	bool calculate_tidal_radius;
	int get_special_parameter_anchor_number() { return special_anchor_lens->lens_number; } // no special parameters can be anchored for the base class

	CoreCusp() : LensProfile()
	{
		setup_lens_properties();
	}
	CoreCusp(const double zlens_in, const double zsrc_in, const double &k0_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens*);
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
	private:
	double kappa_e, b, n;
	double re; // effective radius
	double mstar; // total stellar mass (alternate parameterization)
	double def_factor; // used to calculate the spherical deflection

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq);
	double kapavg_spherical_rsq(const double rsq);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:

	SersicLens() : LensProfile()
	{
		setup_lens_properties();
	}
	SersicLens(const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens*);
	SersicLens(const SersicLens* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
};

class Cored_SersicLens : public LensProfile
{
	private:
	double kappa_e, b, n;
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

	Cored_SersicLens() : LensProfile()
	{
		setup_lens_properties();
	}
	Cored_SersicLens(const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens*);
	Cored_SersicLens(const Cored_SersicLens* lens_in);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
};

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
	MassSheet() : LensProfile()
	{
		setup_lens_properties();
	}
	MassSheet(const double zlens_in, const double zsrc_in, const double &kext_in, const double &xc_in, const double &yc_in, Lens*);
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
	Deflection() : LensProfile()
	{
		setup_lens_properties();
	}
	Deflection(const double zlens_in, const double zsrc_in, const double &defx_in, const double &defy_in, Lens*);
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

class Tabulated_Model : public LensProfile
{
	private:
	double kscale;
	double rscale, rscale0, rscale_factor; // rscale is a parameter that can be varied, whereas rscale0 is set when the table is created or loaded from file
	int grid_logr_N, grid_phi_N;
	double grid_logrlength;
	double *grid_logrvals, *grid_phivals;
	double **kappa_vals, **pot_vals, **defx, **defy, **hess_xx, **hess_yy, **hess_xy;
	string original_lens_command; // used for saving commands to reproduce this model
	double original_kscale, original_rscale;
	bool loaded_from_file;

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq) { return 0; } // will not be used

	public:
	Tabulated_Model() : LensProfile()
	{
		setup_lens_properties();
	}
	Tabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, Lens*);
	Tabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double &xc, const double &yc, ifstream& tabfile, const string& tab_filename, Lens*);

	Tabulated_Model(const Tabulated_Model* lens_in);
	~Tabulated_Model();
	void output_tables(const string tabfile_root);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	void print_lens_command(ofstream& scriptout);

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
	QTabulated_Model() : LensProfile()
	{
		setup_lens_properties();
	}
	QTabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, const double qmin, const int q_N, Lens*);
	QTabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, const double &xc, const double &yc, ifstream& tabfile, Lens*);

	QTabulated_Model(const QTabulated_Model* lens_in);
	~QTabulated_Model();
	void output_tables(const string tabfile_root);

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
	TestModel() : LensProfile()
	{
		setup_lens_properties();
	}
	TestModel(const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc);

	//double kappa(double, double);
	//void deflection(double, double, lensvector&);
	//void hessian(double, double, lensmatrix&);
	//double potential(double, double);
};

#endif // PROFILE_H
