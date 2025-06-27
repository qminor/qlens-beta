#ifndef SBPROFILE_H
#define SBPROFILE_H

#include "mathexpr.h"
#include "spline.h"
#include "egrad.h"
#include "lensvec.h"
//#include "profile.h"
//#include "GregsMathHdr.h"
#include "mcmchdr.h"
#include "simplex.h"
#include <cmath>
#include <iostream>
#include <vector>

struct ImagePixelData;
class LensProfile;
class QLens;

enum SB_ProfileName { SB_SPLINE, GAUSSIAN, SERSIC, CORE_SERSIC, CORED_SERSIC, DOUBLE_SERSIC, sple, dpie, nfw_SOURCE, SHAPELET, MULTI_GAUSSIAN_EXPANSION, TOPHAT, SB_MULTIPOLE };

class SB_Profile : public EllipticityGradient, private UCMC, private Simplex
{
	friend class QLens;
	friend class LensProfile;
	friend class SersicLens;
	friend class DoubleSersicLens;
	friend class Cored_SersicLens;
	friend class SPLE_Lens;
	friend class dPIE_Lens;
	friend class NFW;
	friend class ImagePixelGrid;
	friend struct ImagePixelData;
	private:
	Spline sb_spline;
	double sb_splint(double);
	double qx_parameter, f_parameter;

	protected:
	SB_ProfileName sbtype;
	bool lensed_center_coords; // option that makes the coordinates of a lensed image of the source the free parameters
	double q, theta, x_center, y_center; // four base parameters, which can be added to in derived surface brightness models
	double x_center_lensed, y_center_lensed; // used if lensed_center_coords is set to true
	double epsilon1, epsilon2; // used for defining ellipticity, or else components of ellipticity (epsilon1, epsilon2)
	double c0; // "boxiness" parameter
	double rt; // truncation radius parameter
	double costheta, sintheta;

	double **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model

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
	dvector penalty_upper_limits, penalty_lower_limits;
	dvector stepsizes;
	bool include_limits;
	dvector lower_limits, upper_limits;
	dvector lower_limits_initial, upper_limits_initial;

	int n_fourier_modes; // Number of Fourier mode perturbations to elliptical isophotes (zero by default)
	ivector fourier_mode_mvals, fourier_mode_paramnum;
	dvector fourier_mode_cosamp, fourier_mode_sinamp;

	void set_nparams(const int &n_params_in, const bool resize = false);
	void reset_anchor_lists();
	void setup_base_source_properties(const int np, const int sbprofile_np, const bool is_elliptical_source, const int pmode_in = 0);
	void copy_base_source_data(const SB_Profile* sb_in);
	//bool spawn_lens_model(SPLE_Lens* lens_model);

	void set_geometric_param_pointers(int qi);
	void set_geometric_paramnames(int qi);
	void set_ellipticity_parameter(const double &q_in);

	void set_geometric_parameters(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	void set_geometric_parameters_radians(const double &q_in, const double &theta_in, const double &xc_in, const double &yc_in);
	void set_angle_from_components(const double &comp_x, const double &comp_y);
	void set_center_if_lensed_coords();
	void calculate_ellipticity_components();
	void update_meta_parameters_and_pointers();
	void update_angle_meta_params();
	void update_ellipticity_meta_parameters();
	virtual void update_meta_parameters()
	{
		update_ellipticity_meta_parameters();
	}

	void set_angle(const double &theta_degrees);
	void set_angle_radians(const double &theta_in);
	void rotate(double&, double&);
	void rotate_back(double&, double&);
	static bool orient_major_axis_north;
	static bool use_sb_ellipticity_components; // if set to true, uses e_1 and e_2 as fit parameters instead of q and theta
	static int default_ellipticity_mode;
	static bool use_fmode_scaled_amplitudes; // if set to true, uses a_m = m*A_m and b_m = m*B_m as parameters instead of true amplitudes
	static bool fourier_sb_perturbation; // if true, add fourier modes to the surface brightness, rather than the elliptical radius
	static bool fourier_use_eccentric_anomaly; // use eccentric anomaly as angle for fourier modes? (preferable, but lensing multiploes must use polar angle)
	static double zoom_split_factor; 
	static double zoom_scale; 
	int ellipticity_mode;

	public:
	int sb_number;
	bool is_lensed; // Can be a lensed source, or a galaxy in the lens plane
	bool zoom_subgridding; // Useful if pixels are large compared to profile--subgrids to prevent undersampling
	bool center_anchored_to_lens, center_anchored_to_source;
	LensProfile* center_anchor_lens;
	SB_Profile* center_anchor_source;
	SB_Profile** parameter_anchor_source;
	int* parameter_anchor_paramnum;
	double* parameter_anchor_ratio;
	double* parameter_anchor_exponent;
	bool* anchor_parameter_to_source;

	int *indxptr; // points to important integer values for subclasses that uses them (e.g. shapelets)
	QLens* qlens;
	int parameter_mode; // allows for different parametrizations

	SB_Profile() : qx_parameter(1), param(0) { qlens = NULL; }
	SB_Profile(const char *splinefile, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in, QLens* qlens_in);
	SB_Profile(const SB_Profile* sb_in);
	~SB_Profile() {
		if (param != NULL) delete[] param;
		if (anchor_parameter_to_source != NULL) delete[] anchor_parameter_to_source;
		if (parameter_anchor_source != NULL) delete[] parameter_anchor_source;
		if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
		if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
		if (parameter_anchor_exponent != NULL) delete[] parameter_anchor_exponent;
	}
	void set_null_ptrs_and_values()
	{
		anchor_parameter_to_source = NULL;
		parameter_anchor_source = NULL;
		parameter_anchor_paramnum = NULL;
		param = NULL;
		parameter_anchor_ratio = NULL;
		parameter_anchor_exponent = NULL;
	}

	void anchor_center_to_lens(LensProfile** center_anchor_list, const int &center_anchor_lens_number);
	void anchor_center_to_source(SB_Profile** center_anchor_list, const int &center_anchor_source_number);
	void delete_center_anchor();
	bool enable_ellipticity_gradient(dvector& efunc_params, const int egrad_mode, const int n_bspline_coefs, const dvector& knots, const double ximin = 1e30, const double ximax = 1e30, const double xiref = 1.5, const bool linear_xivals = false, const bool copy_vary_setting = false, boolvector* vary_egrad = NULL);
	void disable_ellipticity_gradient();
	bool enable_fourier_gradient(dvector& fourier_params, const dvector& knots, const bool copy_vary_settings = false, boolvector* vary_egrad = NULL);

	virtual void assign_param_pointers();
	virtual void assign_paramnames();
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

	void get_vary_flags(boolvector& vary_flags);
	void set_limits(const dvector& lower, const dvector& upper);
	void set_limits(const dvector& lower, const dvector& upper, const dvector& lower_init, const dvector& upper_init);
	bool get_limits(dvector& lower, dvector& upper, dvector& lower0, dvector& upper0, int &index);
	bool get_limits(dvector& lower, dvector& upper, int &index);
	bool get_limits(dvector& lower, dvector& upper);

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
	void get_fit_parameter_names(std::vector<std::string>& paramnames_vary, std::vector<std::string> *latex_paramnames_vary = NULL, std::vector<std::string> *latex_subscripts_vary = NULL, const bool include_suffix = false);
	virtual void get_parameters(double* params);
	bool get_specific_parameter(const std::string name_in, double& value);
	bool update_specific_parameter(const std::string name_in, const double& value);
	bool update_specific_parameter(const int paramnum, const double& value);

	virtual void update_parameters(const double* params);
	virtual void update_fit_parameters(const double* fitparams, int &index, bool& status);
	void update_anchored_parameters();
	bool update_anchored_parameters_to_source(const int src_i);
	void update_anchor_center();
	void assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, SB_Profile* param_anchor_source);
	void copy_parameter_anchors(const SB_Profile* sb_in);
	void unanchor_parameter(SB_Profile* param_anchor_source);

	bool fit_sbprofile_data(IsophoteData& isophote_data, const int fit_mode, const int n_livepts=500, const int mpi_np=1, const int mpi_id=0, const std::string fit_output_dir = "."); // for fitting to isophote data
	double sbprofile_loglike(double *params);
	bool fit_egrad_profile_data(IsophoteData& isophote_data, const int egrad_param, const int fit_mode, const int n_livepts=500, const bool optimize_knots=false, const int mpi_np=1, const int mpi_id=0, const std::string fit_output_dir = ".");
	double profile_fit_loglike(double *params);
	double profile_fit_loglike_bspline(double *params);
	void find_egrad_paramnums(int& qi, int& qf, int& theta_i, int& theta_f, int& amp_i, int& amp_f);

	void plot_sb_profile(double rmin, double rmax, int steps, std::ofstream &sbout);
	void print_parameters(const double zs = -1);
	void print_vary_parameters();

	// the following items MUST be redefined in all derived classes
	virtual double sb_rsq(const double rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	virtual double sb_rsq_deriv(const double rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	virtual void window_params(double& xmin, double& xmax, double& ymin, double& ymax);
	virtual double window_rmax();
	virtual double length_scale(); // retrieves characteristic length scale of object (used for zoom subgridding)

	// these functions can be redefined in the derived classes, but don't have to be
	virtual double surface_brightness_r(const double r);
	virtual double surface_brightness(double x, double y);
	//virtual double calculate_Lmatrix_element(const double x, const double y, const int amp_index); // used by Shapelet subclass
	virtual void calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight); // used by Shapelet subclass
	virtual void calculate_gradient_Rmatrix_elements(double* Rmatrix_elements, int* Rmatrix_index);
	virtual void calculate_curvature_Rmatrix_elements(double* Rmatrix, int* Rmatrix_index);
	virtual void update_amplitudes(double*& ampvec); // used by Shapelet subclass
	virtual void get_regularization_param_ptr(double*& regparam_ptr); // for source objects that are regularized
	//virtual void get_amplitudes(double *ampvec); // used by Shapelet subclass
	virtual void update_indxptr(const int newval);
	virtual double surface_brightness_zeroth_order(double x, double y);
	virtual double get_scale_parameter();
	virtual void update_scale_parameter(const double scale);

	//virtual double surface_brightness_zoom(const double x, const double y, const double pixel_xlength, const double pixel_ylength);
	double surface_brightness_zoom(lensvector &centerpt, lensvector &pt1, lensvector &pt2, lensvector &pt3, lensvector &pt4, const double sb_noise);

	SB_ProfileName get_sbtype() { return sbtype; }
	void get_center_coords(double &xc, double &yc) { xc=x_center; yc=y_center; }
	int get_n_params() { return n_params; }
	int get_sbprofile_nparams() { return sbprofile_nparams; }
	void check_vary_params();
	int get_n_vary_params() { return n_vary_params; }
	int get_center_anchor_number();
	void set_include_limits(bool inc) { include_limits = inc; }
	void set_lensed_center(bool lensed_xcyc) {
		lensed_center_coords = lensed_xcyc;
		x_center_lensed = x_center;
		y_center_lensed = y_center;
		set_center_if_lensed_coords();
		assign_paramnames();
		assign_param_pointers();
	}

};

class Gaussian : public SB_Profile
{
	private:
	double sbtot, max_sb, sig_x; // sig_x is the dispersion along the major axis

	double sb_rsq(const double);

	public:
	Gaussian() : SB_Profile() {}
	Gaussian(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	Gaussian(const Gaussian* sb_in);
	~Gaussian() {}

	//double surface_brightness_zoom(const double x, const double y, const double pixel_xlength, const double pixel_ylength);
	//double surface_brightness_zoom(lensvector &centerpt, lensvector &pt1, lensvector &pt2, lensvector &pt3, lensvector &pt4);

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class Sersic : public SB_Profile
{
	friend class SersicLens;
	private:
	double s0, b, n;
	double Reff; // effective radius

	double sb_rsq(const double);

	public:
	Sersic() : SB_Profile() {}
	Sersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	Sersic(const Sersic* sb_in);
	~Sersic() {}

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class CoreSersic : public SB_Profile
{
	private:
	double s0, k, n, rc, gamma, alpha;
	double Reff; // effective radius

	double sb_rsq(const double);

	public:
	CoreSersic() : SB_Profile() {}
	CoreSersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in, const double &gamma_in, const double &alpha_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	CoreSersic(const CoreSersic* sb_in);
	~CoreSersic() {}

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class Cored_Sersic : public SB_Profile
{
	friend class Cored_SersicLens;
	private:
	double s0, b, n, rc;
	double Reff; // effective radius

	double sb_rsq(const double);

	public:
	Cored_Sersic() : SB_Profile() {}
	Cored_Sersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	Cored_Sersic(const Cored_Sersic* sb_in);
	~Cored_Sersic() {}

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class DoubleSersic : public SB_Profile
{
	friend class DoubleSersicLens;
	private:
	double s0, delta_s;
	double s0_1, b1, n1;
	double s0_2, b2, n2;
	double Reff1, Reff2; // effective radius

	double sb_rsq(const double);

	public:
	DoubleSersic() : SB_Profile() {}
	DoubleSersic(const double &s0_in, const double &delta_s_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	DoubleSersic(const DoubleSersic* sb_in);
	~DoubleSersic() {}

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class SPLE : public SB_Profile
{
	friend class SPLE_Lens;

	private:
	double bs, s, alpha;

	double sb_rsq(const double);

	public:
	SPLE() : SB_Profile() {}
	SPLE(const double &b_in, const double &alpha_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, QLens* qlens_in);
	SPLE(const SPLE* sb_in);

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class dPIE : public SB_Profile
{
	friend class dPIE_Lens;

	private:
	double bs, s, a;

	double sb_rsq(const double);

	public:
	dPIE() : SB_Profile() {}
	dPIE(const double &b_in, const double &alpha_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, QLens* qlens_in);
	dPIE(const dPIE* sb_in);

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class NFW_Source : public SB_Profile
{
	friend class NFW;

	private:
	double s0, rs;

	double sb_rsq(const double);
	double nfw_function_xsq(const double &xsq);

	public:
	NFW_Source() : SB_Profile() {}
	NFW_Source(const double &s0_in, const double &rs_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, QLens* qlens_in);
	NFW_Source(const NFW_Source* sb_in);

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class Shapelet : public SB_Profile
{
	friend class QLens;
	private:
	double sig; // sig is the average dispersion of the (0,0) shapelet which is Gaussian
	double regparam; // regularization parameter for shapelets (if using)
	double sig_factor; // used in pmode=2; sigma is set using a scaling factor of the dispersion of the source SB, instead of sigma itself
	double **amps; // shapelet amplitudes
	int n_shapelets;
	bool truncate_at_3sigma; // this truncates the shapelets at r = 2*sigma to eliminate edge effects

	public:
	Shapelet() : SB_Profile() { amps = NULL; }
	Shapelet(const double &amp00, const double &sig_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int nn, const bool truncate_2sig, const int parameter_mode_in, QLens* qlens_in);
	Shapelet(const Shapelet* sb_in);
	~Shapelet() {
		if (amps != NULL) {
			for (int i=0; i < n_shapelets; i++) delete[] amps[i];
			delete[] amps;
		}
	}

	double surface_brightness(double x, double y);
	double hermite_polynomial(const double x, const int n);

	//double surface_brightness_zoom(const double x, const double y, const double pixel_xlength, const double pixel_ylength);
	//double surface_brightness_zoom(lensvector &centerpt, lensvector &pt1, lensvector &pt2, lensvector &pt3, lensvector &pt4);

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();
	//double calculate_Lmatrix_element(double x, double y, const int amp_index);
	void calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight);
	void calculate_gradient_Rmatrix_elements(double* Rmatrix_elements, int* Rmatrix_index);
	void calculate_curvature_Rmatrix_elements(double* Rmatrix, int* Rmatrix_index);
	void get_regularization_param_ptr(double*& regparam_ptr);
	double surface_brightness_zeroth_order(double x, double y);
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
	private:
	int n_gaussians;
	double *amps; // shapelet amplitudes
	double *sigs; // shapelet widths
	double logsig_i, logsig_f;
	double regparam; // regularization parameter for MGE

	double sb_rsq(const double);
	double sb_rsq_deriv(const double rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models

	public:
	MGE() : SB_Profile() { amps = NULL; sigs = NULL; }
	MGE(const double reg, const double amp0, const double sig_i, const double sig_f, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int nn, const int parameter_mode_in, QLens* qlens_in);
	MGE(const MGE* sb_in);
	~MGE() {
		if (amps != NULL) {
			delete[] amps;
			delete[] sigs;
		}
	}

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();
	//double calculate_Lmatrix_element(double x, double y, const int amp_index);
	void calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight);
	void get_regularization_param_ptr(double*& regparam_ptr);
	void update_amplitudes(double*& ampvec);
	//void get_amplitudes(double *ampvec);
	void update_indxptr(const int newval);

	double window_rmax();
	double length_scale();
};

class SB_Multipole : public SB_Profile
{
	private:
	int m;
	double A_n, r0, theta_eff;
	bool sine_term; // specifies whether it is a sine or cosine multipole term

	public:
	SB_Multipole() : SB_Profile() {}
	SB_Multipole(const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool sine, QLens* qlens_in);
	SB_Multipole(const SB_Multipole* sb_in);
	~SB_Multipole() {}

	double surface_brightness(double x, double y);

	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};

class TopHat : public SB_Profile
{
	private:
	double sb, rad; // sig_x is the dispersion along the major axis

	double sb_rsq(const double);

	public:
	TopHat() : SB_Profile() {}
	TopHat(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in);
	TopHat(const TopHat* sb_in);
	~TopHat() {}

	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
	double length_scale();
};


#endif // SBPROFILE_H
