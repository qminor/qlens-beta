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
#include <functional>
#include <iostream>
#include <vector>
#include <complex>
#include <map>

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

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

template <typename QScalar>
struct LensIntegral; // defined in lensintegral.h

class QLens; 

template <typename QScalar>
class LensParams
{
	public:
	QScalar q, theta, x_center, y_center; // four base parameters, which can be added to in derived lens models
	QScalar **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model
	QScalar epsilon, epsilon1, epsilon2; // used for defining ellipticity, and/or components of ellipticity (epsilon1, epsilon2)
	QScalar costheta, sintheta;
	QScalar theta_eff; // used for intermediate calculations if ellipticity components are being used
	QScalar xc_prime, yc_prime; // used if lensed_center_coords is set to true
};

template <typename QScalar>
class LensSpline_Params : public LensParams<QScalar>
{
	public:
	Spline<QScalar> kspline;
	QScalar qx_parameter, f_parameter;
};

class LensProfile : public EllipticityGradient
{
	friend struct LensIntegral<double>;
#ifdef USE_STAN
	friend struct LensIntegral<stan::math::var>;
#endif
	friend class QLens;
	friend class SB_Profile;
	friend class Sersic;
	friend class Cored_Sersic;
	friend class SPLE;
	friend class dPIE;
	friend class ImagePixelGrid;
	friend class Cosmology;

	using GaussQuad = GaussLegendre<std::function<double(const double)>,double>;
	using Patterson = GaussPatterson<std::function<double(const double)>,double>;
	using Fejer = ClenshawCurtis<std::function<double(const double)>,double>;

	//Spline<double> kspline;
	//double qx_parameter, f_parameter;

	public:
	LensParams<double>* lensparams; // this will point to the corresponding lensparams in the inherited classes
#ifdef USE_STAN
	LensParams<stan::math::var>* lensparams_dif; // this will point to the corresponding lensparams in the inherited classes
#endif

	private:
	LensSpline_Params<double> lensparams_spl;
#ifdef USE_STAN
	LensSpline_Params<stan::math::var> lensparams_spl_dif; // autodiff version
#endif
	template <typename QScalar>
	LensSpline_Params<QScalar>& assign_lensspline_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_spl_dif;
		else
#endif
		return lensparams_spl;
	}

	template <typename QScalar>
	LensParams<QScalar>& assign_lensparam_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return (*lensparams_dif);
		else
#endif
		return (*lensparams);
	}

	public:
	double zlens, zsrc_ref;
	double sigma_cr, kpc_to_arcsec;
	//double q, theta, x_center, y_center; // four base parameters, which can be added to in derived lens models

	protected:
	LensProfileName lenstype;
	bool center_defined;
	bool lensed_center_coords; // option for line-of-sight perturber that makes the lensed position of the perturber the free parameters
	double zlens_current; // used to check if zlens has been changed, in which case sigma_cr, etc. are updated
	//double xc_prime, yc_prime; // used if lensed_center_coords is set to true
	double f_major_axis; // used for defining elliptical radius
	//double epsilon, epsilon1, epsilon2; // used for defining ellipticity, and/or components of ellipticity (epsilon1, epsilon2)
	//double costheta, sintheta;
	//double theta_eff; // used for intermediate calculations if ellipticity components are being used
	//double **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model
	bool perturber; // optional flag that can make the perturber subgridding faster, if used

	int n_params, n_vary_params;
	int lensprofile_nparams; // just the parameters that define the kappa profile (rather than the angular structure or center coord's)
	bool angle_param_exists;
	int ellipticity_paramnum; // used to keep track of ellipticity parameter (this feature is used only by qtab models)
	boolvector vary_params;
	boolvector angle_param; // used to keep track of angle parameters so they can be easily converted to degrees and displayed
	std::string model_name;
	std::vector<std::string> paramnames;
	std::vector<std::string> latex_paramnames, latex_param_subscripts;
	boolvector set_auto_penalty_limits;
	Vector<double> penalty_upper_limits, penalty_lower_limits;
	Vector<double> stepsizes;
	bool include_limits;
	Vector<double> lower_limits, upper_limits;

	int n_fourier_modes; // Number of Fourier mode perturbations to elliptical density contours (zero by default)
	ivector fourier_mode_mvals, fourier_mode_paramnum;
	Vector<double> fourier_mode_cosamp, fourier_mode_sinamp;
	Spline<double> *fourier_integral_left_cos_spline;
	Spline<double> *fourier_integral_left_sin_spline;
	Spline<double> *fourier_integral_right_cos_spline;
	Spline<double> *fourier_integral_right_sin_spline;
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

	void set_integration_pointers();
	virtual void set_model_specific_integration_pointers();
	void update_meta_parameters_and_pointers();
	void update_angle_meta_params();
	void update_ellipticity_meta_parameters();
	void update_cosmology_meta_parameters(const bool force_update = false);
	virtual void update_meta_parameters()
	{
		update_cosmology_meta_parameters();
		update_ellipticity_meta_parameters();
	}
	void calculate_ellipticity_components();
	void update_center_from_pixsrc_coords(QLens* qlensptr);

	double potential_numerical(const double, const double);
	double potential_spherical_default(const double x, const double y);
	void deflection_numerical(const double, const double, lensvector<double>&);
	void deflection_spherical_default(const double, const double, lensvector<double>&);
	void hessian_numerical(const double, const double, lensmatrix<double>&);
	void hessian_spherical_default(const double, const double, lensmatrix<double>&);
	void deflection_and_hessian_together(const double x, const double y, lensvector<double> &def, lensmatrix<double>& hess);
	void deflection_and_hessian_numerical(const double x, const double y, lensvector<double>& def, lensmatrix<double>& hess);
	void warn_if_not_converged(const bool& converged, const double &x, const double &y);

	double rmin_einstein_radius; // initial bracket used to find Einstein radius
	double rmax_einstein_radius; // initial bracket used to find Einstein radius
	double einstein_radius_root(const double r);
	double zfac; // for doing calculations at redshift other than the reference redshift
	double mass_intval; // for calculating 3d enclosed mass
	Spline<double> *rho3d_logx_spline;

	template <typename QScalar>
	QScalar kappa_avg_spherical_integral(const QScalar);

	template <typename QScalar>
	QScalar mass_enclosed_spherical_integrand(const QScalar);
	template <typename QScalar>
	QScalar kapavg_spherical_generic(const QScalar rsq);
	template <typename QScalar>
	QScalar potential_spherical_integral(const QScalar rsq);

	double calculate_scaled_mass_3d_from_kappa(const double r);
	double calculate_scaled_mass_3d_from_analytic_rho3d(const double r);
	double mass3d_r_integrand_analytic(const double r);
	virtual double rho3d_r_integrand_analytic(const double r);

	double rho3d_w_integrand(const double w);
	double mass3d_r_integrand(const double r);
	double mass_inverse_rsq(const double u);
	double half_mass_radius_root(const double r);

	void kappa_deflection_and_hessian_from_elliptical_potential(const double x, const double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess);
	void deflection_from_elliptical_potential(const double x, const double y, lensvector<double>& def);
	void hessian_from_elliptical_potential(const double x, const double y, lensmatrix<double>& hess);
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

	inline static IntegrationMethod integral_method = Gauss_Patterson_Quadrature;
	inline static bool orient_major_axis_north = true;
	inline static bool use_ellipticity_components = false; // if set to true, uses e_1 and e_2 as fit parameters instead of q and theta
	inline static bool integration_warnings = true;
	inline static int default_ellipticity_mode = 1;
	inline static int default_fejer_nlevels = 12;
	inline static int fourier_spline_npoints = 336;
	inline static double integral_tolerance = 1e-3;
	Cosmology* cosmo;
	QLens* qlens;
	int ellipticity_mode;
	int parameter_mode; // allows for different parametrizations
	int lens_subclass; // allows for different subclasses of lenses (e.g. multipole order m=0,1,2...); set to -1 if there are no subclasses defined
	std::string subclass_label;
	bool analytic_3d_density; // if true, uses analytic 3d density to find mass_3d(r); if false, finds deprojected 3d profile through integration

	LensProfile() {
		//std::cout << "HUBBA WHA??" << std::endl;
		set_null_ptrs_and_values();
		//std::cout << "HUBBA WHA2??" << std::endl;
		//lensparams_spl.qx_parameter = 1.0;
		//std::cout << "HUBBA WHA3??" << std::endl;
		//setup_lens_properties();
		//std::cout << "HUBBA WHA4??" << std::endl;
	}
	LensProfile(const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in, Cosmology*);
	LensProfile(const LensProfile* lens_in);
	~LensProfile() {
		if (lensparams->param != NULL) delete[] lensparams->param;
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
		//lensparams->param = NULL;
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
		cosmo = NULL;
		lens_number = -1;
	}
	void set_qlens_pointer(QLens* qlens_in) { qlens = qlens_in; }
	void set_redshifts(const double zlens_in, const double zsrc_in);
	void setup_cosmology(Cosmology* cosmo_in);

	template <typename QScalar>
	using temp_defptr = void(*)(const QScalar, const QScalar, lensvector<QScalar>&);

	//void (LensProfile::*temp_defptr)(const QScalar, const QScalar, lensvector<QScalar>& def); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default

	// in all derived classes, each of the following function pointers can be redirected if analytic formulas
	// are used instead of the default numerical version
	double (LensProfile::*kapavgptr_rsq_spherical)(const double); // numerical: &LensProfile::kapavg_spherical_integral
	double (LensProfile::*potptr_rsq_spherical)(const double); // numerical: &LensProfile::potential_spherical_integral
	void (LensProfile::*defptr)(const double, const double, lensvector<double>& def); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default
	void (LensProfile::*hessptr)(const double, const double, lensmatrix<double>& hess); // numerical: &LensProfile::hessian_numerical or &LensProfile::hessian_spherical_default
	double (LensProfile::*potptr)(const double, const double); // numerical: &LensProfile::potential_numerical
	void (LensProfile::*def_and_hess_ptr)(const double, const double, lensvector<double>& def, lensmatrix<double> &hess); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default

#ifdef USE_STAN
	stan::math::var (LensProfile::*kapavgptr_rsq_spherical_autodif)(const stan::math::var); // numerical: &LensProfile::kapavg_spherical_integral
	stan::math::var (LensProfile::*potptr_rsq_spherical_autodif)(const stan::math::var); // numerical: &LensProfile::potential_spherical_integral
	void (LensProfile::*defptr_autodif)(const stan::math::var, const stan::math::var, lensvector<stan::math::var>& def); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default
	void (LensProfile::*hessptr_autodif)(const stan::math::var, const stan::math::var, lensmatrix<stan::math::var>& hess); // numerical: &LensProfile::hessian_numerical or &LensProfile::hessian_spherical_default
	stan::math::var (LensProfile::*potptr_autodif)(const stan::math::var, const stan::math::var); // numerical: &LensProfile::potential_numerical
	void (LensProfile::*def_and_hess_ptr_autodif)(const stan::math::var, const stan::math::var, lensvector<stan::math::var>& def, lensmatrix<stan::math::var> &hess); // numerical: &LensProfile::deflection_numerical or &LensProfile::deflection_spherical_default
#endif

	bool anchor_center_to_lens(const int &center_anchor_lens_number);
	void delete_center_anchor();
	bool setup_transform_center_coords_to_pixsrc_frame(const double dxc, const double dyc, QLens* qlensptr_in=NULL);
	bool enable_ellipticity_gradient(Vector<double>& efunc_params, const int egrad_mode, const int n_bspline_coefs, const Vector<double>& knots, const double ximin = 1e30, const double ximax = 1e30, const double xiref = 1.5, const bool linear_xivals = false, const bool copy_vary_setting = false, boolvector* vary_egrad = NULL);
	void add_fourier_mode(const int m_in, const double amp_in, const double phi_in, const bool vary1, const bool vary2);
	void remove_fourier_modes();
	bool enable_fourier_gradient(Vector<double>& fourier_params, const Vector<double>& knots, const bool copy_vary_settings = false, boolvector* vary_egrad = NULL);
	void find_egrad_paramnums(int& qi, int& qf, int& theta_i, int& theta_f, int& amp_i, int& amp_f);

	virtual void assign_param_pointers();
	virtual void assign_paramnames();
	bool register_vary_flags();
	bool set_vary_flags(boolvector &vary_flags);
	void get_vary_flags(boolvector &vary_flags);
	bool vary_parameters(const boolvector& vary_params_in);
	bool update_specific_varyflag(const std::string name_in, const bool flag);
	void set_limits(const Vector<double>& lower, const Vector<double>& upper);
	bool set_limits_specific_parameter(const std::string name_in, const double& lower, const double& upper);
	void update_limits(const double* lower, const double* upper, const bool* limits_changed, int& index);
	bool get_limits(Vector<double>& lower, Vector<double>& upper);
	bool get_limits(Vector<double>& lower, Vector<double>& upper, int &index);
	void shift_angle_90();
	void shift_angle_minus_90();
	void reset_angle_modulo_2pi();
	virtual void set_auto_stepsizes();  // This *must* be redefined in all derived classes
	virtual void set_auto_ranges(); // This *must* be redefined in all derived classes

	void set_geometric_param_auto_stepsizes(int &index);
	void get_auto_stepsizes(Vector<double>& stepsizes, int &index);
	void set_geometric_param_auto_ranges(int param_i);
	void get_auto_ranges(boolvector& use_penalty_limits, Vector<double>& lower, Vector<double>& upper, int &index);

	static void extract_geometric_params_from_map(double& q1, double& q2, double& xcp, double& ycp, std::map<std::string, double> dict)
	{ 
		if (!use_ellipticity_components) {
			try {
			 q1 = dict.at("q");
			} catch (...) {
			 q1 = 1.0;
			}
			try {
			 q2 = dict.at("theta");
			} catch (...) {
				q2 = 0.0;
			}
		} else {
			try {
			 q1 = dict.at("e1");
			 q2 = dict.at("e2");
			} catch (...) {
			 q1 = 0.0;
			 q2 = 0.0;
			}
		}
		try {
					 xcp = dict.at("xc");
		} catch (...) {
			xcp = 0.0;
		}
		try {
					 ycp = dict.at("yc");
		} catch (...) {
			ycp = 0.0;
		}
	}

	virtual void get_fit_parameters(double *fitparams, int &index);
	void get_fit_parameter_names(std::vector<std::string>& paramnames_vary, std::vector<std::string> *latex_paramnames_vary = NULL, std::vector<std::string> *latex_subscripts_vary = NULL);
	virtual double get_parameter(const int i);
	virtual void get_parameters(double* params);
	bool lookup_parameter_number(const std::string name_in, int& paramnum);
	bool check_parameter_name(const std::string name_in);
	bool get_specific_parameter(const std::string name_in, double& value);
	bool get_specific_limit(const std::string name_in, double& lower, double& upper);
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
	virtual void get_auxiliary_parameter(std::string& aux_paramname, double& aux_param) { aux_paramname = ""; aux_param = 0; } // used for outputting information of derived parameters

	// the following function MUST be redefined in all derived classes
	virtual double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	virtual double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	virtual stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	virtual stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	virtual void get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor);
	virtual double get_xi_parameter(const double zfactor);
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
	double einstein_radius();
	bool calculate_half_mass_radius(double& half_mass_radius, const double mtot_in = -10);
	double mass_rsq(const double rsq);

	virtual double kappa_avg_r(const double r);
	//void plot_kappa_profile(double rmin, double rmax, int steps, const char *kname, const char *kdname = NULL);
	void plot_kappa_profile(double rmin, double rmax, int steps, std::ofstream& kout, std::ofstream& kdout);
	void plot_kappa_profile(const int n_rvals, double* rvals, double* kapvals, double* kapavgvals);
	virtual bool core_present(); // this function is only used for certain derived classes (i.e. specific lens models)
	bool has_kapavg_profile();

	double elliptical_radius(double x, double y);
	virtual void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess);
	virtual void potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess);
	virtual double potential(double x, double y);
	virtual double kappa(double x, double y);
	virtual void deflection(double x, double y, lensvector<double>& def);
	virtual void hessian(double x, double y, lensmatrix<double>& hess); // the Hessian matrix of the lensing potential (*not* the arrival time surface)

	void kappa_and_dkappa_dR(double x, double y, double& kap, double& dkap); // this is just used for the 'xi' parameter
	double kappa_from_fourier_modes(const double x, const double y);
	void add_deflection_from_fourier_modes(const double x, const double y, lensvector<double>& def);
	void add_hessian_from_fourier_modes(const double x, const double y, lensmatrix<double>& hess);
	void spline_fourier_mode_integrals(const double rmin, const double rmax);

	public:
	bool isspherical() { return (lensparams->q==1.0); }
	std::string get_model_name() { return model_name; }
	LensProfileName get_lenstype() { return lenstype; }
	void get_center_coords(double &xc, double &yc) { xc=lensparams->x_center; yc=lensparams->y_center; }
	void get_center_coords(lensvector<double> &center) { center[0]=lensparams->x_center; center[1]=lensparams->y_center; }
	void get_q_theta(double &q_out, double& theta_out) { q_out=lensparams->q; theta_out=lensparams->theta; }
	double get_f_major_axis() { return f_major_axis; }
	double get_redshift() { return zlens; }
	int get_n_params() { return n_params; }
	int get_lensprofile_nparams() { return lensprofile_nparams; }
	int get_n_vary_params() { return n_vary_params; }
	bool get_vary_flag(const int paramnum) { return vary_params[paramnum]; }
	int get_center_anchor_number() { return center_anchor_lens->lens_number; }
	virtual int get_special_parameter_anchor_number() { return -1; } // no special parameters can be center_anchored for the base class
	void set_zsrc_ref(const double zsrc_ref_in) { zsrc_ref = zsrc_ref_in; }
	void set_theta(double theta_in) { lensparams->theta=theta_in; update_angle_meta_params(); }
	void set_center(double xc_in, double yc_in) { lensparams->x_center = xc_in; lensparams->y_center = yc_in; }
	void set_include_limits(bool inc) { include_limits = inc; }
	//void set_integral_tolerance(const double acc);
	//void set_integral_warnings();
	void set_perturber(bool ispert) { perturber = ispert; }
	void set_lensed_center(bool lensed_xcyc) {
		lensed_center_coords = lensed_xcyc;
		lensparams->xc_prime = lensparams->x_center;
		lensparams->yc_prime = lensparams->y_center;
		set_center_if_lensed_coords();
		assign_paramnames();
		assign_param_pointers();
	}
	bool output_plates(const int n_plates);
};

class SPLE_Lens : public LensProfile
{
	template <typename QScalar>
	class SPLE_Params : public LensParams<QScalar>
	{
		public:
		QScalar alpha, bprime, sprime;
		QScalar b, s, gamma;
		QScalar qsq, ssq_prime;
	};

	private:
	SPLE_Params<double> lensparams_sple;
#ifdef USE_STAN
	SPLE_Params<stan::math::var> lensparams_sple_dif; // autodiff version
#endif
	template <typename QScalar>
	SPLE_Params<QScalar>& assign_sple_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_sple_dif;
		else
#endif
		return lensparams_sple;
	}

	private:
	//double alpha, bprime, sprime; // alpha=2D density log-slope, whereas bprime,sprime are defined along the major axis
	// Note that in emode=1, the actual fit parameters are bprime' = bprime*sqrt(q) and sprime' = sprime*sqrt(q), not bprime and sprime. (See the constructor function for more on how this is implemented.)
	//double b, s;
	//double qsq, ssq_prime; // used in lensing calculations
	//double gamma; // 3D density log-slope, which is an alternative parameter instead of alpha
	inline static const double euler_mascheroni = 0.57721566490153286060;
	inline static const double def_tolerance = 1e-16;

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	double kapavg_spherical_rsq_iso(const double rsq);
	void deflection_elliptical_iso(const double, const double, lensvector<double>&);
	void hessian_elliptical_iso(const double, const double, lensmatrix<double>&);
	double potential_spherical_rsq_iso(const double rsq);
	double potential_elliptical_iso(const double x, const double y);
	void deflection_elliptical_nocore(const double x, const double y, lensvector<double>&);
	void deflection_and_hessian_elliptical_nocore(const double x, const double y, lensvector<double>&, lensmatrix<double>&);
	void hessian_elliptical_nocore(const double x, const double y, lensmatrix<double>& hess);
	double potential_elliptical_nocore(const double x, const double y);
	double potential_spherical_rsq_nocore(const double rsq);
	std::complex<double> deflection_angular_factor(const double &phi);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode_in = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	SPLE_Lens(const double zlens_in, const double zsrc_in, const double &b_in, const double &slope_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in);
	SPLE_Lens(const SPLE_Lens* lens_in);
	void initialize_parameters(const double &bb, const double &slope, const double &ss, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	SPLE_Lens(SPLE* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double calculate_scaled_mass_3d(const double r);
	bool core_present() { return (lensparams_sple.sprime==0) ? false : true; }
	double get_inner_logslope() { return -lensparams_sple.alpha; }
	void get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor = 1.0);
	bool output_cosmology_info(const int lens_number);
};

class dPIE_Lens : public LensProfile
{
	template <typename QScalar>
	class dPIE_Params : public LensParams<QScalar>
	{
		public:
		QScalar bprime, sprime, aprime;
		QScalar b, s, a;
		QScalar sigma0, mtot, s_kpc, a_kpc;
		QScalar qsq, ssq_prime, asq;
	};

	private:
	dPIE_Params<double> lensparams_dpie;
#ifdef USE_STAN
	dPIE_Params<stan::math::var> lensparams_dpie_dif; // autodiff version
#endif
	template <typename QScalar>
	dPIE_Params<QScalar>& assign_dpie_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_dpie_dif;
		else
#endif
		return lensparams_dpie;
	}

	private:
	//double b, s, a; // a is the truncation radius
	//double sigma0, mtot, s_kpc, a_kpc; // alternate parametrizations

	// the following are meta-parameters used in lensing calculations
	//double bprime, sprime, aprime; // these are the lengths along the major axis
	//double qsq, ssq_prime, asq;

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	double kapavg_spherical_rsq(const double rsq);
	void deflection_elliptical(const double, const double, lensvector<double>&);
	void hessian_elliptical(const double, const double, lensmatrix<double>&);
	double potential_elliptical(const double x, const double y);
	double potential_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode_in = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	bool calculate_tidal_radius;
	int get_special_parameter_anchor_number() { return this->special_anchor_lens->lens_number; } // no special parameters can be anchored for the base class

	dPIE_Lens(const double zlens_in, const double zsrc_in, const double &b_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode, Cosmology* cosmo_in);
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
	void get_einstein_radius(double& r1, double &r2, const double zfactor = 1.0) { this->rmin_einstein_radius = 0.01*lensparams_dpie.b; this->rmax_einstein_radius = 100*lensparams_dpie.b; LensProfile::get_einstein_radius(r1,r2,zfactor); } 
	double get_tidal_radius() { return lensparams_dpie.aprime; }
	bool core_present() { return (lensparams_dpie.sprime==0) ? false : true; }
};

class NFW : public LensProfile
{
	template <typename QScalar>
	class NFW_Params : public LensParams<QScalar>
	{
		public:
		QScalar ks, rs;
		QScalar m200, c200, rs_kpc; // alternate parametrizations
	};

	private:
	NFW_Params<double> lensparams_nfw;
#ifdef USE_STAN
	NFW_Params<stan::math::var> lensparams_nfw_dif; // autodiff version
#endif
	template <typename QScalar>
	NFW_Params<QScalar>& assign_nfw_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_nfw_dif;
		else
#endif
		return lensparams_nfw;
	}

	public:
	//double ks, rs;
	//double m200, c200, rs_kpc; // alternate parametrizations

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);
	template <typename QScalar>
	QScalar lens_function_xsq(const QScalar);

	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();
	void set_ks_rs_from_m200_c200();
	void set_ks_c200_from_m200_rs();

	public:
	NFW(const double zlens_in, const double zsrc_in, const double &p1_in, const double &p2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in);
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
	template <typename QScalar>
	class Truncated_NFW_Params : public LensParams<QScalar>
	{
		public:
		QScalar ks, rs, rt;
		QScalar m200, c200, rs_kpc, rt_kpc, tau200, tau_s; // alternate parametrizations
	};

	private:
	Truncated_NFW_Params<double> lensparams_tnfw;
#ifdef USE_STAN
	Truncated_NFW_Params<stan::math::var> lensparams_tnfw_dif; // autodiff version
#endif
	template <typename QScalar>
	Truncated_NFW_Params<QScalar>& assign_tnfw_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_tnfw_dif;
		else
#endif
		return lensparams_tnfw;
	}

	// This profile is the same as NFW, times a factor (1+(r/rt)^2)^-2 which smoothly truncates the halo (prescription from Baltz, Marshall & Oguri (2008))
	private:
	//double ks, rs, rt;
	//double m200, c200, rs_kpc, rt_kpc, tau200, tau_s; // alternate parametrizations

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return 0; } // no implementation in this class for kappa_rsq_deriv...still overloading the wrapper just to be safe
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return 0; }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	template <typename QScalar>
	QScalar lens_function_xsq(const QScalar);
	double kapavg_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();
	void set_ks_rs_from_m200_c200();
	void set_ks_c200_from_m200_rs();

	public:
	Truncated_NFW(const double zlens_in, const double zsrc_in, const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int truncation_mode_in, const int parameter_mode_in, Cosmology* cosmo_in);
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
	template <typename QScalar>
	class Cored_NFW_Params : public LensParams<QScalar>
	{
		public:
		QScalar ks, rs, rc;
		QScalar m200, c200, beta, rs_kpc, rc_kpc; // alternate parametrization
	};

	private:
	Cored_NFW_Params<double> lensparams_cnfw;
#ifdef USE_STAN
	Cored_NFW_Params<stan::math::var> lensparams_cnfw_dif; // autodiff version
#endif
	template <typename QScalar>
	Cored_NFW_Params<QScalar>& assign_cnfw_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_cnfw_dif;
		else
#endif
		return lensparams_cnfw;
	}

	// This profile goes like 1/(r+rc)/(r+rs)^2
	private:
	//double ks, rs, rc;
	//double m200, c200, beta, rs_kpc, rc_kpc; // alternate parametrization

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);
	template <typename QScalar>
	QScalar lens_function_xsq(const QScalar);
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
	Cored_NFW(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &rt_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in);
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
	template <typename QScalar>
	class Hernquist_Params : public LensParams<QScalar>
	{
		public:
		QScalar ks, rs;
	};

	private:
	Hernquist_Params<double> lensparams_hernquist;
#ifdef USE_STAN
	Hernquist_Params<stan::math::var> lensparams_hernquist_dif; // autodiff version
#endif
	template <typename QScalar>
	Hernquist_Params<QScalar>& assign_hernquist_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_hernquist_dif;
		else
#endif
		return lensparams_hernquist;
	}

	private:
	//double ks, rs;

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);
	template <typename QScalar>
	QScalar lens_function_xsq(const QScalar);
	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	double rho3d_r_integrand_analytic(const double r);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	Hernquist(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, Cosmology*);
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
	template <typename QScalar>
	class ExpDisk_Params : public LensParams<QScalar>
	{
		public:
		QScalar k0, R_d;
	};

	private:
	ExpDisk_Params<double> lensparams_expdisk;
#ifdef USE_STAN
	ExpDisk_Params<stan::math::var> lensparams_expdisk_dif; // autodiff version
#endif
	template <typename QScalar>
	ExpDisk_Params<QScalar>& assign_expdisk_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_expdisk_dif;
		else
#endif
		return lensparams_expdisk;
	}

	private:
	//double k0, R_d;

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);
	double kapavg_spherical_rsq(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	ExpDisk(const double zlens_in, const double zsrc_in, const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, Cosmology*);
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
	template <typename QScalar>
	class Shear_Params : public LensParams<QScalar>
	{
		public:
		QScalar shear, theta_eff;
		QScalar shear1, shear2;
	};

	private:
	Shear_Params<double> lensparams_shear;
#ifdef USE_STAN
	Shear_Params<stan::math::var> lensparams_shear_dif; // autodiff version
#endif
	template <typename QScalar>
	Shear_Params<QScalar>& assign_shear_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_shear_dif;
		else
#endif
		return lensparams_shear;
	}

	private:
	//double shear, theta_eff;
	//double shear1, shear2; // used when shear_components is turned on
	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq) { return 0; }
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq) {return 0; }

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	void set_angle_from_components(const double &comp_x, const double &comp_y);

	public:
	Shear(const double zlens_in, const double zsrc_in, const double &shear_in, const double &theta_degrees, const double &xc_in, const double &yc_in, Cosmology*);
	void initialize_parameters(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in);
	Shear(const Shear* lens_in);

	inline static bool use_shear_component_params = false; // if set to true, uses shear_1 and shear_2 as fit parameters instead of gamma and theta
	inline static bool angle_points_towards_perturber = false; // direction of hypothetical perturber differs from shear angle by 90 degrees

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	// here the base class deflection/hessian functions are overloaded because the angle is put in explicitly in the formulas (no rotation of the coordinates is needed)
	double potential(double, double);
	void potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess);
	void deflection(double, double, lensvector<double>&);
	void hessian(double, double, lensmatrix<double>&);
	void potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
	{
		kap = 0;
		potential_derivatives(x,y,def,hess);
	}

	double kappa(double, double) { return 0; }
	void get_einstein_radius(double& r1, double& r2, const double zfactor = 1.0) { r1=0; r2=0; }
};

class Multipole : public LensProfile
{
	template <typename QScalar>
	class Multipole_Params : public LensParams<QScalar>
	{
		public:
		QScalar A_n, n, theta_eff;
	};

	private:
	Multipole_Params<double> lensparams_mpole;
#ifdef USE_STAN
	Multipole_Params<stan::math::var> lensparams_mpole_dif; // autodiff version
#endif
	template <typename QScalar>
	Multipole_Params<QScalar>& assign_mpole_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_mpole_dif;
		else
#endif
		return lensparams_mpole;
	}

	private:
	int m;
	//double A_n, n, theta_eff;
	bool kappa_multipole; // specifies whether it is a multipole in the potential or in kappa
	bool sine_term; // specifies whether it is a sine or cosine multipole term

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:

	Multipole(const double zlens_in, const double zsrc_in, const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, Cosmology*, const bool sine=false);
	void initialize_parameters(const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, const bool sine);
	Multipole(const Multipole* lens_in);

	// here the base class deflection/hessian functions are overloaded because the angle is put in explicitly in the formulas (no rotation of the coordinates is needed)
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess);
	void potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess);
	void deflection(double, double, lensvector<double>&);
	void hessian(double, double, lensmatrix<double>&);
	double potential(double, double);
	double kappa(double, double);
	double deflection_m0_spherical_r(const double r);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	void get_einstein_radius(double& r1, double& r2, const double zfactor = 1.0);
};

class PointMass : public LensProfile
{
	template <typename QScalar>
	class PointMass_Params : public LensParams<QScalar>
	{
		public:
		QScalar b; // Einstein radius of point mass
		QScalar mtot; // alternative parameterization
	};

	private:
	PointMass_Params<double> lensparams_ptmass;
#ifdef USE_STAN
	PointMass_Params<stan::math::var> lensparams_ptmass_dif; // autodiff version
#endif
	template <typename QScalar>
	PointMass_Params<QScalar>& assign_ptmass_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_ptmass_dif;
		else
#endif
		return lensparams_ptmass;
	}

	private:
	//double b; // Einstein radius of point mass
	//double mtot; // alternative parameterization

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq) { return 0; }
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq) {return 0; }
	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	PointMass() {}
	PointMass(const double zlens_in, const double zsrc_in, const double &bb, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology*);
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
	void potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
	{
		kap = 0;
		potential_derivatives(x,y,def,hess);
	}

	void deflection(double, double, lensvector<double>&);
	void hessian(double, double, lensmatrix<double>&);

	bool calculate_total_scaled_mass(double& total_mass);
	double calculate_scaled_mass_3d(const double r);
	void get_einstein_radius(double& r1, double& r2, const double zfactor = 1.0);
};

class CoreCusp : public LensProfile
{
	template <typename QScalar>
	class CoreCusp_Params : public LensParams<QScalar>
	{
		public:
		QScalar n, gamma, a, s, k0;
		QScalar einstein_radius;
		QScalar core_enclosed_mass;
		QScalar digamma_term, beta_p1, beta_p2; // used for calculations of kappa, dkappa
		QScalar r200_const;

	};

	private:
	CoreCusp_Params<double> lensparams_cc;
#ifdef USE_STAN
	CoreCusp_Params<stan::math::var> lensparams_cc_dif; // autodiff version
#endif
	template <typename QScalar>
	CoreCusp_Params<QScalar>& assign_cc_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_cc_dif;
		else
#endif
		return lensparams_cc;
	}

	private:
	//double n, gamma, a, s, k0;
	inline static const double nstep = 0.2; // this is for calculating the n=3 case, which requires extrapolation since F21 is singular for n=3
	inline static const double digamma_three_halves = 0.036489973978435; // needed for the n=3 case

	bool set_k0_by_einstein_radius;

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	template <typename QScalar>
	QScalar kappa_rsq_nocore(const QScalar rsq_prime, const QScalar atilde);
	template <typename QScalar>
	QScalar kappa_rsq_deriv_nocore(const QScalar rsq_prime, const QScalar atilde);

	double kapavg_spherical_rsq(const double rsq);

	double enclosed_mass_spherical_nocore(const double rsq_prime, const double atilde) { return enclosed_mass_spherical_nocore(rsq_prime,atilde,lensparams_cc.n); }
	double enclosed_mass_spherical_nocore(const double rsq_prime, const double atilde, const double nprime);
	double enclosed_mass_spherical_nocore_n3(const double rsq_prime, const double atilde, const double nprime);
	double enclosed_mass_spherical_nocore_limit(const double rsq, const double atilde, const double n_stepsize);
	void set_core_enclosed_mass();

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();
	void get_auxiliary_parameter(std::string& aux_paramname, double& aux_param) { if (set_k0_by_einstein_radius) aux_paramname = "k0"; aux_param = lensparams_cc.k0; }

	public:
	bool calculate_tidal_radius;
	int get_special_parameter_anchor_number() { return special_anchor_lens->lens_number; } // no special parameters can be anchored for the base class

	CoreCusp(const double zlens_in, const double zsrc_in, const double &k0_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology*);
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
	bool core_present() { return (lensparams_cc.s==0) ? false : true; }
};

class SersicLens : public LensProfile
{
	friend class SB_Profile;
	friend class Sersic;

	template <typename QScalar>
	class Sersic_Params : public LensParams<QScalar>
	{
		public:
		QScalar kappa0, b, n;
		QScalar re; // effective radius
		QScalar mstar; // total stellar mass (alternate parameterization)
		QScalar def_factor; // used to calculate the spherical deflection
	};

	private:
	Sersic_Params<double> lensparams_sersic;
#ifdef USE_STAN
	Sersic_Params<stan::math::var> lensparams_sersic_dif; // autodiff version
#endif
	template <typename QScalar>
	Sersic_Params<QScalar>& assign_sersic_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_sersic_dif;
		else
#endif
		return lensparams_sersic;
	}

	private:
	//double kappa0, b, n;
	//double re; // effective radius
	//double mstar; // total stellar mass (alternate parameterization)
	//double def_factor; // used to calculate the spherical deflection

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	double kapavg_spherical_rsq(const double rsq);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:

	SersicLens(const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology*);
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

	template <typename QScalar>
	class DoubleSersic_Params : public LensParams<QScalar>
	{
		public:
		QScalar kappa0, delta_k;
		QScalar kappa0_1, b1, n1;
		QScalar kappa0_2, b2, n2;
		QScalar Reff1, Reff2; // effective radiukappa
		QScalar mstar; // total stellar mass (alternate parameterization to kappa0)
	};

	private:
	DoubleSersic_Params<double> lensparams_dsersic;
#ifdef USE_STAN
	DoubleSersic_Params<stan::math::var> lensparams_dsersic_dif; // autodiff version
#endif
	template <typename QScalar>
	DoubleSersic_Params<QScalar>& assign_dsersic_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_dsersic_dif;
		else
#endif
		return lensparams_dsersic;
	}

	private:
	//double kappa0, delta_k;
	//double kappa0_1, b1, n1;
	//double kappa0_2, b2, n2;
	//double Reff1, Reff2; // effective radiukappa
	//double mstar; // total stellar mass (alternate parameterization to kappa0)
	//double def_factor; // used to calculate the spherical deflection

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	//double kapavg_spherical_rsq(const double rsq);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	//void set_model_specific_integration_pointers();

	public:

	DoubleSersicLens(const double zlens_in, const double zsrc_in, const double &p1_in, const double &delta_k_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology*);
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
	template <typename QScalar>
	class Cored_Sersic_Params : public LensParams<QScalar>
	{
		public:
		QScalar kappa0, b, n;
		QScalar re; // effective radius
		QScalar rc; // core radius
		QScalar mstar; // total stellar mass (alternate parameterization)
		QScalar def_factor; // used to calculate the spherical deflection
	};

	private:
	Cored_Sersic_Params<double> lensparams_csersic;
#ifdef USE_STAN
	Cored_Sersic_Params<stan::math::var> lensparams_csersic_dif; // autodiff version
#endif
	template <typename QScalar>
	Cored_Sersic_Params<QScalar>& assign_csersic_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_csersic_dif;
		else
#endif
		return lensparams_csersic;
	}

	private:
	//double kappa0, b, n;
	//double re; // effective radius
	//double rc; // core radius
	//double mstar; // total stellar mass (alternate parameterization)
	//double def_factor; // used to calculate the spherical deflection

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	double kapavg_spherical_rsq(const double rsq);

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:

	Cored_SersicLens(const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology*);
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

class MassSheet : public LensProfile
{
	template <typename QScalar>
	class MassSheet_Params : public LensParams<QScalar>
	{
		public:
		QScalar kext;
	};

	private:
	MassSheet_Params<double> lensparams_sheet;
#ifdef USE_STAN
	MassSheet_Params<stan::math::var> lensparams_sheet_dif; // autodiff version
#endif
	template <typename QScalar>
	MassSheet_Params<QScalar>& assign_sheet_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_sheet_dif;
		else
#endif
		return lensparams_sheet;
	}

	private:
	//double kext;

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	double kapavg_spherical_rsq(const double rsq);
	double potential_spherical_rsq(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	MassSheet(const double zlens_in, const double zsrc_in, const double &kext_in, const double &xc_in, const double &yc_in, Cosmology*);
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
	void potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
	{
		kap = lensparams_sheet.kext;
		potential_derivatives(x,y,def,hess);
	}

	void deflection(double, double, lensvector<double>&);
	void hessian(double, double, lensmatrix<double>&);

	void get_einstein_radius(double& r1, double& r2, const double zfactor = 1.0) { r1=0; r2=0; }
};

class Deflection : public LensProfile
{
	template <typename QScalar>
	class Deflection_Params : public LensParams<QScalar>
	{
		public:
		QScalar def_x, def_y;
	};

	private:
	Deflection_Params<double> lensparams_defl;
#ifdef USE_STAN
	Deflection_Params<stan::math::var> lensparams_defl_dif; // autodiff version
#endif
	template <typename QScalar>
	Deflection_Params<QScalar>& assign_defl_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_defl_dif;
		else
#endif
		return lensparams_defl;
	}

	private:
	//double def_x, def_y;

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq) { return 0; }
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq) {return 0; }
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	Deflection(const double zlens_in, const double zsrc_in, const double &defx_in, const double &defy_in, Cosmology*);
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
	void potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
	{
		kap = 0;
		potential_derivatives(x,y,def,hess);
	}

	void deflection(double, double, lensvector<double>&);
	void hessian(double, double, lensmatrix<double>&);

	void get_einstein_radius(double& r1, double& r2, const double zfactor = 1.0) { r1=0; r2=0; }
};

class TopHatLens : public LensProfile
{
	template <typename QScalar>
	class TopHat_Params : public LensParams<QScalar>
	{
		public:
		QScalar kap0, xi0;
	};

	private:
	TopHat_Params<double> lensparams_tophat;
#ifdef USE_STAN
	TopHat_Params<stan::math::var> lensparams_tophat_dif; // autodiff version
#endif
	template <typename QScalar>
	TopHat_Params<QScalar>& assign_tophat_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensparams_tophat_dif;
		else
#endif
		return lensparams_tophat;
	}

	private:
	//double kap0, xi0;

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return kappa_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return kappa_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar kappa_rsq_deriv_impl(const QScalar rsq);

	double kapavg_spherical_rsq(const double rsq);
	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);
	void set_model_specific_integration_pointers();

	public:
	TopHatLens(const double zlens_in, const double zsrc_in, const double &kap0_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, Cosmology*);
	void initialize_parameters(const double &kap0_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	TopHatLens(const TopHatLens* lens_in);

	void deflection_analytic(const double, const double, lensvector<double>&);
	void hessian_analytic(const double, const double, lensmatrix<double>&);
	double potential_analytic(const double x, const double y);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();
	//bool calculate_total_scaled_mass(double& total_mass);
};

/*
class Tabulated_Model : public LensProfile
{
	private:
	double kscale;
	double rscale, rscale0, rscale_factor; // rscale is a parameter that can be varied, whereas rscale0 is set when the table is created or loaded from file
	int grid_logr_N, grid_phi_N;
	double grid_logrlength;
	double *grid_logrvals, *grid_phivals;
	double **kappa_vals, **pot_vals, **defx, **defy, **hess_xx, **hess_yy, **hess_xy;
	double original_kscale, original_rscale;
	bool loaded_from_file;

	double kappa_rsq(const double rsq);
	double kappa_rsq_deriv(const double rsq) { return 0; } // will not be used

	public:
	Tabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, Cosmology*);
	Tabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double &xc, const double &yc, std::ifstream& tabfile, const std::string& tab_filename, Cosmology*);

	Tabulated_Model(const Tabulated_Model* lens_in);
	~Tabulated_Model();
	void output_tables(const std::string tabfile_root);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double potential(double, double);
	void potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess);
	double kappa(double, double);
	void deflection(double, double, lensvector<double>&);
	void hessian(double, double, lensmatrix<double>&);

	//void get_einstein_radius(double& r1, double& r2, const double zfactor = 1.0) { r1=0; r2=0; } // cannot use this
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
	QTabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, const double qmin, const int q_N, Cosmology*);
	QTabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, const double &xc, const double &yc, std::ifstream& tabfile, Cosmology*);

	QTabulated_Model(const QTabulated_Model* lens_in);
	~QTabulated_Model();
	void output_tables(const std::string tabfile_root);

	void assign_paramnames();
	void assign_param_pointers();
	void update_meta_parameters();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double potential(double, double);
	void potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess);
	void kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess);
	double kappa(double, double);
	void deflection(double, double, lensvector<double>&);
	void hessian(double, double, lensmatrix<double>&);

	//void get_einstein_radius(double& r1, double& r2, const double zfactor = 1.0) { r1=0; r2=0; } // cannot use this
};
*/

// Model for testing purposes; can also be used as a template for a new lens model
class TestModel : public LensProfile
{
	private:

	double kappa_rsq(const double rsq) { return kappa_rsq_impl(rsq); }
	double kappa_rsq_deriv(const double rsq) { return 0; }
#ifdef USE_STAN
	stan::math::var kappa_rsq(const stan::math::var rsq) { return kappa_rsq_impl(rsq); }
	stan::math::var kappa_rsq_deriv(const stan::math::var rsq) { return 0; }
#endif

	template <typename QScalar>
	QScalar kappa_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	void setup_lens_properties(const int parameter_mode = 0, const int subclass = 0);

	// The following functions can be overloaded, but don't necessarily have to be
	//double kappa_rsq_deriv(const double rsq); // optional
	//void deflection(double, double, lensvector<double>&);
	//void hessian(double, double, lensmatrix<double>&);

	public:
	TestModel(const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);

	//double kappa(double, double);
	//void deflection(double, double, lensvector<double>&);
	//void hessian(double, double, lensmatrix<double>&);
	//double potential(double, double);
};

#endif // PROFILE_H
