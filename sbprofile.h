#ifndef SBPROFILE_H
#define SBPROFILE_H

#include "mathexpr.h"
#include "spline.h"
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

enum SB_ProfileName { SB_SPLINE, GAUSSIAN, SERSIC, TOPHAT };

class SB_Profile
{
	private:
	Spline sb_spline;
	double sb_splint(double);
	double qx_parameter, f_parameter;

	protected:
	SB_ProfileName sbtype;
	double q, theta, x_center, y_center; // four base parameters, which can be added to in derived surface brightness models
	double epsilon, epsilon2; // used for defining ellipticity, or else components of ellipticity (epsilon, epsilon2)
	double costheta, sintheta;

	double **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model

	int n_params, n_vary_params;
	int angle_paramnum; // used to keep track of angle parameter so it can be easily converted to degrees and displayed
	int ellipticity_paramnum; // used to keep track of ellipticity parameter so it can be easily converted to degrees and displayed
	boolvector vary_params;
	string model_name;
	vector<string> paramnames;
	vector<string> latex_paramnames, latex_param_subscripts;
	boolvector set_auto_penalty_limits;
	dvector penalty_upper_limits, penalty_lower_limits;
	dvector stepsizes;
	bool include_limits;
	dvector lower_limits, upper_limits;
	dvector lower_limits_initial, upper_limits_initial;

	void set_nparams(const int &n_params_in);
	void copy_base_source_data(const SB_Profile* sb_in);

	void set_geometric_param_pointers(int qi);
	void set_geometric_paramnames(int qi);
	void set_ellipticity_parameter(const double &q_in);

	void set_geometric_parameters(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	void set_geometric_parameters_radians(const double &q_in, const double &theta_in, const double &xc_in, const double &yc_in);
	void set_angle_from_components(const double &comp_x, const double &comp_y);
	void update_meta_parameters_and_pointers();
	void update_angle_meta_params();
	virtual void update_meta_parameters()
	{
		update_angle_meta_params();
	}

	void set_angle(const double &theta_degrees);
	void set_angle_radians(const double &theta_in);
	void rotate(double&, double&);
	void rotate_back(double&, double&);
	static bool orient_major_axis_north;
	static bool use_ellipticity_components; // if set to true, uses e_1 and e_2 as fit parameters instead of gamma and theta

	public:
	int sb_number;

	SB_Profile() : qx_parameter(1), param(0) {}
	SB_Profile(const char *splinefile, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in);
	SB_Profile(const SB_Profile* sb_in);
	~SB_Profile() {
		if (param != NULL) delete[] param;
	}

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
	bool update_specific_parameter(const string name_in, const double& value);
	virtual void update_parameters(const double* params);
	virtual void update_fit_parameters(const double* fitparams, int &index, bool& status);
	void print_parameters();
	void print_vary_parameters();
	void output_field_in_sci_notation(double* num, ofstream& scriptout, const bool space);
	virtual void print_source_command(ofstream& scriptout, const bool use_limits);

	// the following items MUST be redefined in all derived classes
	virtual double sb_rsq(const double rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	virtual void window_params(double& xmin, double& xmax, double& ymin, double& ymax);
	virtual double window_rmax();

	// these functions can be redefined in the derived classes, but don't have to be
	virtual double surface_brightness_r(const double r);
	virtual double surface_brightness(double x, double y);

	SB_ProfileName get_sbtype() { return sbtype; }
	void get_center_coords(double &xc, double &yc) { xc=x_center; yc=y_center; }
	int get_n_params() { return n_params; }
	int get_n_vary_params() { return n_vary_params; }
};

class Gaussian : public SB_Profile
{
	private:
	double max_sb, sig_x; // sig_x is the dispersion along the major axis

	double sb_rsq(const double);

	public:
	Gaussian() : SB_Profile() {}
	Gaussian(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Gaussian(const Gaussian* sb_in);
	~Gaussian() {}

	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	double window_rmax();
};

class Sersic : public SB_Profile
{
	private:
	double s0, k, n; // sig_x is the dispersion along the major axis
	double Reff; // effective radius

	double sb_rsq(const double);

	public:
	Sersic() : SB_Profile() {}
	Sersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Sersic(const Sersic* sb_in);
	~Sersic() {}

	void update_meta_parameters();
	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	void print_parameters();
	double window_rmax();
};

class TopHat : public SB_Profile
{
	private:
	double sb, rad; // sig_x is the dispersion along the major axis

	double sb_rsq(const double);

	public:
	TopHat() : SB_Profile() {}
	TopHat(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	TopHat(const TopHat* sb_in);
	~TopHat() {}

	void assign_paramnames();
	void assign_param_pointers();
	void set_auto_stepsizes();
	void set_auto_ranges();

	void print_parameters();
	double window_rmax();
};


#endif // SBPROFILE_H
