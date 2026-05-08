#ifndef SBPROFILE_H
#define SBPROFILE_H

#include "mathexpr.h"
#include "spline.h"
#include "egrad.h"
#include "lensvec.h"
#include "profile.h"
//#include "GregsMathHdr.h"
#include "mcmchdr.h"
#include "simplex.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <map>

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

struct ImageData;
class LensProfile;
class PointSource;
class QLens;

enum SB_ProfileName { SB_SPLINE, GAUSSIAN, SERSIC, CORE_SERSIC, CORED_SERSIC, DOUBLE_SERSIC, sple, dpie, nfw_SOURCE, SHAPELET, MULTI_GAUSSIAN_EXPANSION, TOPHAT, SB_MULTIPOLE };

template <typename QScalar>
class SB_Params
{
	public:
	QScalar q, theta, x_center, y_center; // four base parameters, which can be added to in derived surface brightness models
	QScalar x_center_lensed, y_center_lensed; // used if lensed_center_coords is set to true
	QScalar **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model
	QScalar epsilon1, epsilon2; // used for defining ellipticity components
	QScalar costheta, sintheta;
	Vector<QScalar> fourier_mode_cosamp, fourier_mode_sinamp;
	QScalar c0; // "boxiness" parameter
	QScalar rt; // truncation radius parameter
	QScalar zsrc;

	SB_Params() {
		param = NULL;
	}
};

template <typename QScalar>
class SB_Spline_Params : public SB_Params<QScalar>
{
	public:
	Spline<QScalar> sb_spline;
	QScalar qx_parameter, f_parameter;
	SB_Spline_Params() : SB_Params<QScalar>() {
		qx_parameter = 1.0;
		f_parameter = 1.0;
	}
};

class SB_Profile : public EllipticityGradient, private UCMC, private Simplex
{
	friend class QLens;
	friend class LensProfile;
	friend class SersicLens;
	friend class DoubleSersicLens;
	friend class Cored_SersicLens;
	friend SPLE_Lens;
	friend dPIE_Lens;
	friend class NFW;
	friend class ImagePixelGrid;
	friend struct ImageData;
	private:
	//Spline<double> sb_spline;
	double sb_splint(double);
	//double qx_parameter, f_parameter;

	public:
	SB_Params<double>* sbparams; // this will point to the corresponding sbparams in the inherited classes
#ifdef USE_STAN
	SB_Params<stan::math::var>* sbparams_dif; // this will point to the corresponding sbparams in the inherited classes
#endif

	template <typename QScalar>
	SB_Params<QScalar>& assign_sbparam_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return (*sbparams_dif);
		else
#endif
		return (*sbparams);
	}

	protected:
	SB_Spline_Params<double> sbparams_spl;
#ifdef USE_STAN
	SB_Spline_Params<stan::math::var> sbparams_spl_dif; // autodiff version
#endif
	template <typename QScalar>
	SB_Spline_Params<QScalar>& assign_sbspline_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_spl_dif;
		else
#endif
		return sbparams_spl;
	}

	SB_ProfileName sbtype;
	bool lensed_center_coords; // option that makes the coordinates of a lensed image of the source the free parameters
	//double zsrc;
	int band;
	//double q, theta, x_center, y_center; // four base parameters, which can be added to in derived surface brightness models
	//double x_center_lensed, y_center_lensed; // used if lensed_center_coords is set to true
	//double epsilon1, epsilon2; // used for defining ellipticity, or else components of ellipticity (epsilon1, epsilon2)
	//double c0; // "boxiness" parameter
	//double rt; // truncation radius parameter
	//double costheta, sintheta; // used to store sines and cosines so they don't have to be recalculated many times

	//double **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model

	int n_params, n_vary_params;
	int sbprofile_nparams; // just the parameters that define the SB profile (rather than the angular structure or center coord's)
	bool angle_param_exists;
	boolvector angle_param; // used to keep track of angle parameters so they can be easily converted to degrees and displayed
	bool include_boxiness_parameter;
	bool include_truncation_radius;
	boolvector vary_params;
	std::string model_name;
	std::vector<std::string> paramnames;
	std::vector<std::string> latex_paramnames, latex_param_subscripts;
	boolvector set_auto_penalty_limits;
	Vector<double> penalty_upper_limits, penalty_lower_limits;
	Vector<double> stepsizes;
	bool include_limits;
	Vector<double> lower_limits, upper_limits;

	int n_fourier_modes; // Number of Fourier mode perturbations to elliptical isophotes (zero by default)
	ivector fourier_mode_mvals, fourier_mode_paramnum;
	Vector<double> fourier_mode_cosamp, fourier_mode_sinamp;

	void set_nparams(const int &n_params_in, const bool resize = false);
	void reset_anchor_lists();
	void setup_base_source_properties(const int np, const int sbprofile_np, const bool is_elliptical_source, const int pmode_in = 0);
	void copy_base_source_data(const SB_Profile* sb_in);
	//bool spawn_lens_model(SPLE_Lens* lens_model);

	void set_geometric_paramnames(int qi);

	template <typename QScalar>
	void set_geometric_param_pointers(int qi);
	template <typename QScalar>
	void set_ellipticity_parameter(const QScalar &q_in);
	template <typename QScalar>
	void set_geometric_parameters(const QScalar &q_in, const QScalar &theta_degrees, const QScalar &xc_in, const QScalar &yc_in, const QScalar &zsrc_in);
	template <typename QScalar>
	void set_geometric_parameters_radians(const QScalar &q_in, const QScalar &theta_in, const QScalar &xc_in, const QScalar &yc_in);
	template <typename QScalar>
	void set_angle_from_components(QScalar &comp_x, const QScalar &comp_y);
	template <typename QScalar>
	void set_center_if_lensed_coords();
	template <typename QScalar>
	void calculate_ellipticity_components();
	template <typename QScalar>
	void update_meta_parameters_and_pointers();
	template <typename QScalar>
	void update_angle_meta_params();
	template <typename QScalar>
	void update_ellipticity_meta_parameters();

	virtual void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	virtual void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	template <typename QScalar>
	void update_meta_parameters_impl()
	{
		update_ellipticity_meta_parameters<QScalar>();
	}

	template <typename QScalar>
	void set_angle(const QScalar &theta_degrees);
	template <typename QScalar>
	void set_angle_radians(const QScalar &theta_in);
	template <typename QScalar>
	void rotate(QScalar&, QScalar&);
	template <typename QScalar>
	void rotate_back(QScalar&, QScalar&);
#ifdef USE_STAN
	virtual void sync_autodif_parameters();
	void sync_autodif_geometric_parameters();
#endif

	public:
	static bool orient_major_axis_north;
	static bool use_sb_ellipticity_components; // if set to true, uses e_1 and e_2 as fit parameters instead of q and theta
	static int default_ellipticity_mode;
	static bool use_fmode_scaled_amplitudes; // if set to true, uses a_m = m*A_m and b_m = m*B_m as parameters instead of true amplitudes
	static bool fourier_sb_perturbation; // if true, add fourier modes to the surface brightness, rather than the elliptical radius
	static bool fourier_use_eccentric_anomaly; // use eccentric anomaly as angle for fourier modes? (preferable, but lensing multiploes must use polar angle)
	static double zoom_split_factor; 
	static double zoom_scale; 
	int ellipticity_mode;

	int sb_number;
	bool is_lensed; // Can be a lensed source, or a galaxy in the lens plane
	bool zoom_subgridding; // Useful if pixels are large compared to profile--subgrids to prevent undersampling
	bool center_anchored_to_lens, center_anchored_to_source, center_anchored_to_ptsrc;
	LensProfile* center_anchor_lens;
	SB_Profile* center_anchor_source;
	PointSource* center_anchor_ptsrc;
	SB_Profile** parameter_anchor_source;
	int* parameter_anchor_paramnum;
	double* parameter_anchor_ratio;
	double* parameter_anchor_exponent;
	bool* anchor_parameter_to_source;

	int *indxptr; // points to important integer values for subclasses that uses them (e.g. shapelets)
	QLens* qlens;
	int parameter_mode; // allows for different parametrizations

	SB_Profile() { set_null_ptrs_and_values(); }
	SB_Profile(const char *splinefile, const int band_in, const double &zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in, QLens* qlens_in);
	SB_Profile(const SB_Profile* sb_in);
	~SB_Profile() {
		if (sbparams->param != NULL) delete[] sbparams->param;
		if (anchor_parameter_to_source != NULL) delete[] anchor_parameter_to_source;
		if (parameter_anchor_source != NULL) delete[] parameter_anchor_source;
		if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
		if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
		if (parameter_anchor_exponent != NULL) delete[] parameter_anchor_exponent;
#ifdef USE_STAN
		if (sbparams_dif->param != NULL) delete[] sbparams_dif->param;
#endif
	}
	void set_null_ptrs_and_values()
	{
		anchor_parameter_to_source = NULL;
		parameter_anchor_source = NULL;
		parameter_anchor_paramnum = NULL;
		//param = NULL;
		parameter_anchor_ratio = NULL;
		parameter_anchor_exponent = NULL;
		qlens = NULL;
	}
	void set_qlens_pointer(QLens* qlens_in) { qlens = qlens_in; }

	void anchor_center_to_lens(LensProfile** center_anchor_list, const int &center_anchor_lens_number);
	void anchor_center_to_source(SB_Profile** center_anchor_list, const int &center_anchor_source_number);
	void anchor_center_to_ptsrc(PointSource** center_anchor_list, const int &center_anchor_ptsrc_number);
	void delete_center_anchor();
	bool enable_ellipticity_gradient(Vector<double>& efunc_params, const int egrad_mode, const int n_bspline_coefs, const Vector<double>& knots, const double ximin = 1e30, const double ximax = 1e30, const double xiref = 1.5, const bool linear_xivals = false, const bool copy_vary_setting = false, boolvector* vary_egrad = NULL);
	void disable_ellipticity_gradient();
	bool enable_fourier_gradient(Vector<double>& fourier_params, const Vector<double>& knots, const bool copy_vary_settings = false, boolvector* vary_egrad = NULL);

	// the following function MUST be redefined in all derived classes
	virtual void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	virtual void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	virtual void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	bool update_specific_varyflag(const std::string name_in, const bool flag);
	bool vary_parameters(const boolvector& vary_params_in);
	void add_fourier_mode(const int m_in, const double amp_in, const double phi_in, const bool vary1, const bool vary2);
	void add_boxiness_parameter(const double c0_in, const bool vary_c0);
	void add_truncation_radius(const double rt_in, const bool vary_rt);
	bool has_fourier_modes() { return (n_fourier_modes > 0) ? true : false; }
	bool fourier_mode_exists(const int mval) {
		bool mode_exists = false;
		for (int i=0; i < n_fourier_modes; i++) {
			if (fourier_mode_mvals[i]==mval) { mode_exists = true; break; }
		}
		return mode_exists;
	}
	void set_lensed(const bool isl) {
		is_lensed = isl;
		assign_paramnames();
	}
	void set_zoom_subgridding(const bool zoom) {
		zoom_subgridding = zoom;
	}

	void remove_fourier_modes();

	bool register_vary_flags();
	bool set_vary_flags(boolvector &vary_flags);
	void get_vary_flags(boolvector& vary_flags);
	void set_limits(const Vector<double>& lower, const Vector<double>& upper);
	bool set_limits_specific_parameter(const std::string name_in, const double& lower, const double& upper);
	void update_limits(const double* lower, const double* upper, const bool* limits_changed, int& index);
	bool get_limits(Vector<double>& lower, Vector<double>& upper, int &index);
	bool get_limits(Vector<double>& lower, Vector<double>& upper);

	void shift_angle_90();
	void shift_angle_minus_90();
	void reset_angle_modulo_2pi();
	virtual void set_auto_stepsizes();  // This *must* be redefined in all derived classes
	virtual void set_auto_ranges(); // This *must* be redefined in all derived classes

	void set_geometric_param_auto_stepsizes(int &index);
	void get_auto_stepsizes(Vector<double>& stepsizes, int &index);
	void set_geometric_param_auto_ranges(int param_i);
	void get_auto_ranges(boolvector& use_penalty_limits, Vector<double>& lower, Vector<double>& upper, int &index);

	static void extract_geometric_params_from_map(double& q1, double& q2, double& xcp, double& ycp, std::map<std::string, double> dict, std::set<std::string> &allowed)
	{ 
		if (!use_sb_ellipticity_components) {
			allowed.insert("q");
			allowed.insert("theta");
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
			allowed.insert("e1");
			allowed.insert("e2");
			try {
			 q1 = dict.at("e1");
			 q2 = dict.at("e2");
			} catch (...) {
			 q1 = 0.0;
			 q2 = 0.0;
			}
		}
		allowed.insert({"xc","yc"});
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
	void get_fit_parameter_names(std::vector<std::string>& paramnames_vary, std::vector<std::string> *latex_paramnames_vary = NULL, std::vector<std::string> *latex_subscripts_vary = NULL, const bool include_suffix = false);
	virtual void get_parameters(double* params);
	bool lookup_parameter_number(const std::string name_in, int& paramnum);
	bool check_parameter_name(const std::string name_in);
	bool get_specific_parameter(const std::string name_in, double& value);
	bool get_specific_limit(const std::string name_in, double& lower, double& upper);
	bool update_specific_parameter(const std::string name_in, const double& value);
	bool update_specific_parameter(const int paramnum, const double& value);

	void update_parameters(const double* params);
	template <typename QScalar>
	void update_fit_parameters(const QScalar* fitparams, int &index, bool& status);
	void update_anchored_parameters();
	bool update_anchored_parameters_to_source(const int src_i);
	void update_anchor_center();
	void assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, SB_Profile* param_anchor_source);
	void copy_parameter_anchors(const SB_Profile* sb_in);
	void unanchor_parameter(SB_Profile* param_anchor_source);

	bool fit_sbprofile_data(IsophoteData& isophote_data, const int fit_mode, const int n_livepts=500, const int mpi_np=1, const int mpi_id=0, const std::string fit_output_dir = "."); // for fitting to isophote data
	double sbprofile_loglike(const double *params);
	bool fit_egrad_profile_data(IsophoteData& isophote_data, const int egrad_param, const int fit_mode, const int n_livepts=500, const bool optimize_knots=false, const int mpi_np=1, const int mpi_id=0, const std::string fit_output_dir = ".");
	double profile_fit_loglike(const double *params);
	double profile_fit_loglike_bspline(const double *params);
	void find_egrad_paramnums(int& qi, int& qf, int& theta_i, int& theta_f, int& amp_i, int& amp_f);

	void plot_sb_profile(double rmin, double rmax, int steps, std::ofstream &sbout);
	void print_parameters(const bool show_band = false, const int band = 0);
	void print_params() { print_parameters(false,0); }
	void print_vary_parameters();
	std::string mkstring_doub(const double db);
	std::string mkstring_int(const int i);
	std::string get_parameters_string();

	// the following function MUST be redefined in all derived classes
	virtual double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	virtual double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	virtual stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	virtual stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar sb_rsq_deriv_impl(const QScalar rsq);
	template <typename QScalar>
	QScalar surface_brightness_r(const QScalar r);

	virtual void window_params(double& xmin, double& xmax, double& ymin, double& ymax);
	virtual double window_rmax();
	virtual double length_scale(); // retrieves characteristic length scale of object (used for zoom subgridding)

	// these functions can be redefined in the derived classes, but don't have to be
	virtual double surface_brightness(double x, double y) { return surface_brightness_impl(x,y); }
#ifdef USE_STAN
	virtual stan::math::var surface_brightness(stan::math::var x, stan::math::var y) { return surface_brightness_impl(x,y); }
#endif

	//virtual double calculate_Lmatrix_element(const double x, const double y, const int amp_index); // used by Shapelet subclass
	virtual void calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight); // used by Shapelet subclass
	virtual void calculate_gradient_Rmatrix_elements(double* Rmatrix_elements, int* Rmatrix_index);
	virtual void calculate_curvature_Rmatrix_elements(double* Rmatrix, int* Rmatrix_index);
	virtual void calculate_curvature_Rmatrix_elements_rvals(double *rvalsq, const int n_rvals, double* Rmatrix_elements);

	virtual void update_amplitudes(double*& ampvec); // used by Shapelet subclass
	virtual void get_regularization_param_ptr(double*& regparam_ptr); // for source objects that are regularized
	//virtual void get_amplitudes(double *ampvec); // used by Shapelet subclass
	virtual void update_indxptr(const int newval);
	virtual double get_scale_parameter();
	virtual void update_scale_parameter(const double scale);

	template <typename QScalar>
	QScalar surface_brightness_impl(QScalar x, QScalar y);

	//virtual double surface_brightness_zoom(const double x, const double y, const double pixel_xlength, const double pixel_ylength);
	template <typename QScalar>
	QScalar surface_brightness_zoom(lensvector<QScalar> &centerpt, lensvector<QScalar> &pt1, lensvector<QScalar> &pt2, lensvector<QScalar> &pt3, lensvector<QScalar> &pt4, const QScalar sb_noise);

	std::string get_model_name() { return model_name; }
	SB_ProfileName get_sbtype() { return sbtype; }
	void get_center_coords(double &xc, double &yc) { xc=sbparams->x_center; yc=sbparams->y_center; }
	double get_xcenter() { return sbparams->x_center; }
	double get_ycenter() { return sbparams->y_center; }
	int get_n_params() { return n_params; }
	int get_sbprofile_nparams() { return sbprofile_nparams; }
	void check_vary_params();
	int get_n_vary_params() { return n_vary_params; }
	bool get_vary_flag(const int paramnum) { return vary_params[paramnum]; }
	int get_center_anchor_number();
	void set_theta(double theta_in) {
		sbparams->theta=theta_in;
		update_angle_meta_params<double>();
#ifdef USE_STAN
		sbparams_dif->theta=theta_in;
		update_angle_meta_params<stan::math::var>();
#endif
	}
	void set_center(double xc_in, double yc_in) { sbparams->x_center = xc_in; sbparams->y_center = yc_in; }
	void set_include_limits(bool inc) { include_limits = inc; }
	void set_lensed_center(bool lensed_xcyc) {
		lensed_center_coords = lensed_xcyc;
		sbparams->x_center_lensed = sbparams->x_center;
		sbparams->y_center_lensed = sbparams->y_center;
		set_center_if_lensed_coords<double>();
		assign_paramnames();
		assign_param_pointers();
#ifdef USE_STAN
		assign_param_pointers_autodif();
#endif
	}
};

class Gaussian : public SB_Profile
{
	template <typename QScalar>
	class Gaussian_Params : public SB_Params<QScalar>
	{
		public:
		QScalar sbtot, max_sb, sig_x; // sig_x is the dispersion along the major axis
	};

	public:
	Gaussian_Params<double> sbparams_gaussian;
#ifdef USE_STAN
	Gaussian_Params<stan::math::var> sbparams_gaussian_dif; // autodiff version
#endif
	template <typename QScalar>
	Gaussian_Params<QScalar>& assign_gaussian_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_gaussian_dif;
		else
#endif
		return sbparams_gaussian;
	}

	private:
	//double sbtot, max_sb, sig_x; // sig_x is the dispersion along the major axis

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	Gaussian() : SB_Profile() {}
	Gaussian(const int band_in, const double &zsrc_in, const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	Gaussian(const Gaussian* sb_in);
	~Gaussian() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	//double surface_brightness_zoom(const double x, const double y, const double pixel_xlength, const double pixel_ylength);
	//double surface_brightness_zoom(lensvector<double> &centerpt, lensvector<double> &pt1, lensvector<double> &pt2, lensvector<double> &pt3, lensvector<double> &pt4);

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class Sersic : public SB_Profile
{
	friend class SersicLens;

	template <typename QScalar>
	class Sersic_Params : public SB_Params<QScalar>
	{
		public:
		QScalar s0, s_eff, b, n;
		QScalar Reff; // effective radius
	};

	public:
	Sersic_Params<double> sbparams_sersic;
#ifdef USE_STAN
	Sersic_Params<stan::math::var> sbparams_sersic_dif; // autodiff version
#endif
	template <typename QScalar>
	Sersic_Params<QScalar>& assign_sersic_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_sersic_dif;
		else
#endif
		return sbparams_sersic;
	}

	private:
	double s0, s_eff, b, n;
	double Reff; // effective radius

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	Sersic() : SB_Profile() {}
	Sersic(const int band_in, const double &zsrc_in, const double &s0_in, const double &Reff_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, QLens* qlens_in);
	Sersic(const Sersic* sb_in);
	~Sersic() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class Cored_Sersic : public SB_Profile
{
	friend class Cored_SersicLens;
	template <typename QScalar>
	class Cored_Sersic_Params : public SB_Params<QScalar>
	{
		public:
		QScalar s0, b, n, rc;
		QScalar Reff; // effective radius
	};

	public:
	Cored_Sersic_Params<double> sbparams_csersic;
#ifdef USE_STAN
	Cored_Sersic_Params<stan::math::var> sbparams_csersic_dif; // autodiff version
#endif
	template <typename QScalar>
	Cored_Sersic_Params<QScalar>& assign_csersic_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_csersic_dif;
		else
#endif
		return sbparams_csersic;
	}

	private:
	//double s0, b, n, rc;
	//double Reff; // effective radius

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	Cored_Sersic() : SB_Profile() {}
	Cored_Sersic(const int band_in, const double &zsrc_in, const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	Cored_Sersic(const Cored_Sersic* sb_in);
	~Cored_Sersic() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class CoreSersic : public SB_Profile
{
	template <typename QScalar>
	class CoreSersic_Params : public SB_Params<QScalar>
	{
		public:
		QScalar s0, b, k, n, rc, gamma, alpha;
		QScalar Reff; // effective radius
	};

	public:
	CoreSersic_Params<double> sbparams_coresersic;
#ifdef USE_STAN
	CoreSersic_Params<stan::math::var> sbparams_coresersic_dif; // autodiff version
#endif
	template <typename QScalar>
	CoreSersic_Params<QScalar>& assign_coresersic_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_coresersic_dif;
		else
#endif
		return sbparams_coresersic;
	}

	private:
	//double s0, k, n, rc, gamma, alpha;
	//double Reff; // effective radius

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	CoreSersic() : SB_Profile() {}
	CoreSersic(const int band_in, const double &zsrc_in, const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in, const double &gamma_in, const double &alpha_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	CoreSersic(const CoreSersic* sb_in);
	~CoreSersic() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class DoubleSersic : public SB_Profile
{
	friend class DoubleSersicLens;
	template <typename QScalar>
	class DoubleSersic_Params : public SB_Params<QScalar>
	{
		public:
		QScalar s0, delta_s;
		QScalar s0_1, b1, n1;
		QScalar s0_2, b2, n2;
		QScalar Reff1, Reff2; // effective radius
	};

	public:
	DoubleSersic_Params<double> sbparams_dsersic;
#ifdef USE_STAN
	DoubleSersic_Params<stan::math::var> sbparams_dsersic_dif; // autodiff version
#endif
	template <typename QScalar>
	DoubleSersic_Params<QScalar>& assign_dsersic_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_dsersic_dif;
		else
#endif
		return sbparams_dsersic;
	}

	private:
	//double s0, delta_s;
	//double s0_1, b1, n1;
	//double s0_2, b2, n2;
	//double Reff1, Reff2; // effective radius

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	DoubleSersic() : SB_Profile() {}
	DoubleSersic(const int band_in, const double &zsrc_in, const double &s0_in, const double &delta_s_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	DoubleSersic(const DoubleSersic* sb_in);
	~DoubleSersic() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class SPLE : public SB_Profile
{
	friend SPLE_Lens;
	template <typename QScalar>
	class SPLE_Params : public SB_Params<QScalar>
	{
		public:
		QScalar bs, s, alpha;
	};

	public:
	SPLE_Params<double> sbparams_sple;
#ifdef USE_STAN
	SPLE_Params<stan::math::var> sbparams_sple_dif; // autodiff version
#endif
	template <typename QScalar>
	SPLE_Params<QScalar>& assign_sple_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_sple_dif;
		else
#endif
		return sbparams_sple;
	}

	private:
	//double bs, s, alpha;

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	SPLE() : SB_Profile() {}
	SPLE(const int band_in, const double &zsrc_in, const double &b_in, const double &alpha_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, QLens* qlens_in);
	SPLE(const SPLE* sb_in);
	~SPLE() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class dPIE : public SB_Profile
{
	friend dPIE_Lens;
	template <typename QScalar>
	class dPIE_Params : public SB_Params<QScalar>
	{
		public:
		QScalar bs, s, a;
	};

	public:
	dPIE_Params<double> sbparams_dpie;
#ifdef USE_STAN
	dPIE_Params<stan::math::var> sbparams_dpie_dif; // autodiff version
#endif
	template <typename QScalar>
	dPIE_Params<QScalar>& assign_dpie_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_dpie_dif;
		else
#endif
		return sbparams_dpie;
	}

	private:
	//double bs, s, a;

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	dPIE() : SB_Profile() {}
	dPIE(const int band_in, const double &zsrc_in, const double &b_in, const double &alpha_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, QLens* qlens_in);
	dPIE(const dPIE* sb_in);
	~dPIE() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class NFW_Source : public SB_Profile
{
	friend class NFW;

	template <typename QScalar>
	class NFW_Params : public SB_Params<QScalar>
	{
		public:
		QScalar s0, rs;
	};

	public:
	NFW_Params<double> sbparams_nfw;
#ifdef USE_STAN
	NFW_Params<stan::math::var> sbparams_nfw_dif; // autodiff version
#endif
	template <typename QScalar>
	NFW_Params<QScalar>& assign_nfw_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_nfw_dif;
		else
#endif
		return sbparams_nfw;
	}

	private:
	//double s0, rs;

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	template <typename QScalar>
	QScalar nfw_function_xsq(const QScalar &xsq);

	public:
	NFW_Source() : SB_Profile() {}
	NFW_Source(const int band_in, const double &zsrc_in, const double &s0_in, const double &rs_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, QLens* qlens_in);
	NFW_Source(const NFW_Source* sb_in);
	~NFW_Source() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class Shapelet : public SB_Profile
{
	friend class QLens;
	template <typename QScalar>
	class Shapelet_Params : public SB_Params<QScalar>
	{
		public:
		QScalar sig; // sig is the average dispersion of the (0,0) shapelet which is Gaussian
		QScalar regparam; // regularization parameter for shapelets (if using)
		QScalar sig_factor; // used in pmode=2; sigma is set using a scaling factor of the dispersion of the source SB, instead of sigma itself
		QScalar **amps; // shapelet amplitudes
		Shapelet_Params() : SB_Params<QScalar>() { amps = NULL; }
	};

	public:
	Shapelet_Params<double> sbparams_shapelet;
#ifdef USE_STAN
	Shapelet_Params<stan::math::var> sbparams_shapelet_dif; // autodiff version
#endif
	template <typename QScalar>
	Shapelet_Params<QScalar>& assign_shapelet_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_shapelet_dif;
		else
#endif
		return sbparams_shapelet;
	}

	private:
	//double sig; // sig is the average dispersion of the (0,0) shapelet which is Gaussian
	//double regparam; // regularization parameter for shapelets (if using)
	//double sig_factor; // used in pmode=2; sigma is set using a scaling factor of the dispersion of the source SB, instead of sigma itself
	//double **amps; // shapelet amplitudes
	int n_shapelets;
	bool truncate_at_3sigma; // this truncates the shapelets at r = 2*sigma to eliminate edge effects

	//double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	//double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
//#ifdef USE_STAN
	//stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	//stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
//#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	Shapelet() : SB_Profile() {}
	Shapelet(const int band_in, const double &zsrc_in, const double &amp00, const double &sig_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int nn, const bool truncate_2sig, const int parameter_mode_in, QLens* qlens_in);
	Shapelet(const Shapelet* sb_in);
	~Shapelet() {
		if (sbparams_shapelet.amps != NULL) {
			for (int i=0; i < n_shapelets; i++) delete[] sbparams_shapelet.amps[i];
			delete[] sbparams_shapelet.amps;
		}
#ifdef USE_STAN
		if (sbparams_shapelet_dif.amps != NULL) {
			for (int i=0; i < n_shapelets; i++) delete[] sbparams_shapelet_dif.amps[i];
			delete[] sbparams_shapelet_dif.amps;
		}
#endif
	}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	double surface_brightness(double x, double y) { return surface_brightness_impl(x,y); }
#ifdef USE_STAN
	stan::math::var surface_brightness(stan::math::var x, stan::math::var y) { return surface_brightness_impl(x,y); }
#endif

	template <typename QScalar>
	QScalar surface_brightness_impl(QScalar x, QScalar y);

	double hermite_polynomial(const double x, const int n);

	//double surface_brightness_zoom(const double x, const double y, const double pixel_xlength, const double pixel_ylength);
	//double surface_brightness_zoom(lensvector<double> &centerpt, lensvector<double> &pt1, lensvector<double> &pt2, lensvector<double> &pt3, lensvector<double> &pt4);

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();
	//double calculate_Lmatrix_element(double x, double y, const int amp_index);
	void calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight);
	void calculate_gradient_Rmatrix_elements(double* Rmatrix_elements, int* Rmatrix_index);
	void calculate_curvature_Rmatrix_elements(double* Rmatrix, int* Rmatrix_index);
	void get_regularization_param_ptr(double*& regparam_ptr);
	void update_amplitudes(double*& ampvec);
	//void get_amplitudes(double *ampvec);
	double get_scale_parameter();
	void update_scale_parameter(const double scale);
	void update_indxptr(const int newval);

	double window_rmax();
	double length_scale();
};

class MGE : public SB_Profile
{
	friend class QLens;

	template <typename QScalar>
	class MGE_Params : public SB_Params<QScalar>
	{
		public:
		QScalar *amps; // MGE amplitudes
		QScalar *sigs; // MGE widths
		QScalar regparam; // regularization parameter for MGE
		MGE_Params() : SB_Params<QScalar>() { amps = NULL; sigs = NULL; }
	};

	public:
	MGE_Params<double> sbparams_mge;
#ifdef USE_STAN
	MGE_Params<stan::math::var> sbparams_mge_dif; // autodiff version
#endif
	template <typename QScalar>
	MGE_Params<QScalar>& assign_mge_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_mge_dif;
		else
#endif
		return sbparams_mge;
	}

	private:
	int n_gaussians;
	//double *amps; // MGE amplitudes
	//double *sigs; // MGE widths
	//double regparam; // regularization parameter for MGE
	double logsig_i, logsig_f;

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	template <typename QScalar>
	QScalar sb_rsq_deriv_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	MGE() : SB_Profile() {}
	MGE(const int band_in, const double zsrc_in, const double reg, const double amp0, const double sig_i, const double sig_f, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int nn, const int parameter_mode_in, QLens* qlens_in);
	MGE(const MGE* sb_in);
	~MGE() {
		if (sbparams_mge.amps != NULL) {
			delete[] sbparams_mge.amps;
			delete[] sbparams_mge.sigs;
		}
#ifdef USE_STAN
		if (sbparams_mge_dif.amps != NULL) {
			delete[] sbparams_mge_dif.amps;
			delete[] sbparams_mge_dif.sigs;
		}
#endif
	}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() { update_meta_parameters_impl<double>(); }
#ifdef USE_STAN
	void update_meta_parameters_autodif() { update_meta_parameters_impl<stan::math::var>(); }
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();
	template <typename QScalar>
	void update_meta_parameters_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();
	//double calculate_Lmatrix_element(double x, double y, const int amp_index);
	void calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight);
	void calculate_curvature_Rmatrix_elements_rvals(double *rvalsq, const int n_rvals, double* Rmatrix_elements);
	void get_regularization_param_ptr(double*& regparam_ptr);
	void update_amplitudes(double*& ampvec);
	//void get_amplitudes(double *ampvec);
	void update_indxptr(const int newval);

	double window_rmax();
	double length_scale();
};

class SB_Multipole : public SB_Profile
{
	template <typename QScalar>
	class SB_Multipole_Params : public SB_Params<QScalar>
	{
		public:
		QScalar A_n, r0, theta_eff;
	};

	public:
	SB_Multipole_Params<double> sbparams_sbmpole;
#ifdef USE_STAN
	SB_Multipole_Params<stan::math::var> sbparams_sbmpole_dif; // autodiff version
#endif
	template <typename QScalar>
	SB_Multipole_Params<QScalar>& assign_sbmpole_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_sbmpole_dif;
		else
#endif
		return sbparams_sbmpole;
	}

	private:
	int m;
	//double A_n, r0, theta_eff;
	bool sine_term; // specifies whether it is a sine or cosine multipole term

	//double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	//double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
//#ifdef USE_STAN
	//stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	//stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
//#endif

	//template <typename QScalar>
	//QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	//template <typename QScalar>
	//double sb_rsq_deriv_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	SB_Multipole() : SB_Profile() {}
	SB_Multipole(const int band_in, const double &zsrc_in, const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool sine, QLens* qlens_in);
	SB_Multipole(const SB_Multipole* sb_in);
	~SB_Multipole() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	double surface_brightness(double x, double y) { return surface_brightness_impl(x,y); }
#ifdef USE_STAN
	stan::math::var surface_brightness(stan::math::var x, stan::math::var y) { return surface_brightness_impl(x,y); }
#endif

	template <typename QScalar>
	QScalar surface_brightness_impl(QScalar x, QScalar y);

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() {}
#ifdef USE_STAN
	void update_meta_parameters_autodif() {}
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class TopHat : public SB_Profile
{
	template <typename QScalar>
	class TopHat_Params : public SB_Params<QScalar>
	{
		public:
		QScalar sb, rad; // sig_x is the dispersion along the major axis
	};

	public:
	TopHat_Params<double> sbparams_tophat;
#ifdef USE_STAN
	TopHat_Params<stan::math::var> sbparams_tophat_dif; // autodiff version
#endif
	template <typename QScalar>
	TopHat_Params<QScalar>& assign_tophat_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return sbparams_tophat_dif;
		else
#endif
		return sbparams_tophat;
	}

	private:
	//double sb, rad; // sig_x is the dispersion along the major axis

	double sb_rsq(const double rsq) { return sb_rsq_impl(rsq); }
	double sb_rsq_deriv(const double rsq) { return sb_rsq_deriv_impl(rsq); }
#ifdef USE_STAN
	stan::math::var sb_rsq(const stan::math::var rsq) { return sb_rsq_impl(rsq); }
	stan::math::var sb_rsq_deriv(const stan::math::var rsq) { return sb_rsq_deriv_impl(rsq); }
#endif

	template <typename QScalar>
	QScalar sb_rsq_impl(const QScalar rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	TopHat() : SB_Profile() {}
	TopHat(const int band_in, const double &zsrc_in, const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	TopHat(const TopHat* sb_in);
	~TopHat() {}
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void assign_param_pointers() { assign_param_pointers_impl<double>(); }
#ifdef USE_STAN
	void assign_param_pointers_autodif() { assign_param_pointers_impl<stan::math::var>(); }
#endif

	void update_meta_parameters() {}
#ifdef USE_STAN
	void update_meta_parameters_autodif() {}
#endif

	void assign_paramnames();
	template <typename QScalar>
	void assign_param_pointers_impl();

	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};


#endif // SBPROFILE_H
