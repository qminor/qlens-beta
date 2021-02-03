#include "sbprofile.h"
#include "qlens.h"
#include "mathexpr.h"
#include "errors.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

bool SB_Profile::orient_major_axis_north = false; // At the moment, this setting cannot be changed; it should probably be removed altogether
bool SB_Profile::use_sb_ellipticity_components = false;
bool SB_Profile::use_fmode_scaled_amplitudes = true;
double SB_Profile::zoom_split_factor = 2;
double SB_Profile::zoom_scale = 4;

SB_Profile::SB_Profile(const char *splinefile, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in)
{
	model_name = "sbspline";
	sbtype = SB_SPLINE;
	set_nparams(6);
	qx_parameter = qx_in;
	f_parameter = f_in;
	sb_spline.input(splinefile);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_param_pointers();
	assign_paramnames();
}

SB_Profile::SB_Profile(const SB_Profile* sb_in)
{
	qx_parameter = sb_in->qx_parameter;
	f_parameter = sb_in->f_parameter;
	copy_base_source_data(sb_in);
	sb_spline.input(sb_in->sb_spline);
}

void SB_Profile::copy_base_source_data(const SB_Profile* sb_in)
{
	model_name = sb_in->model_name;
	sbtype = sb_in->sbtype;
	sb_number = sb_in->sb_number;
	set_nparams(sb_in->n_params);
	center_anchored = sb_in->center_anchored;
	center_anchor_lens = sb_in->center_anchor_lens;
	is_lensed = sb_in->is_lensed;
	zoom_subgridding = sb_in->zoom_subgridding;

	q = sb_in->q;
	epsilon = sb_in->epsilon;
	epsilon2 = sb_in->epsilon2;
	set_angle_radians(sb_in->theta);
	x_center = sb_in->x_center;
	y_center = sb_in->y_center;

	paramnames = sb_in->paramnames;
	latex_paramnames = sb_in->latex_paramnames;
	latex_param_subscripts = sb_in->latex_param_subscripts;
	n_vary_params = sb_in->n_vary_params;
	vary_params.input(sb_in->vary_params);
	stepsizes.input(sb_in->stepsizes);
	set_auto_penalty_limits.input(sb_in->set_auto_penalty_limits);
	penalty_lower_limits.input(sb_in->penalty_lower_limits);
	penalty_upper_limits.input(sb_in->penalty_upper_limits);

	include_limits = sb_in->include_limits;
	if (include_limits) {
		lower_limits.input(sb_in->lower_limits);
		upper_limits.input(sb_in->upper_limits);
		lower_limits_initial.input(sb_in->lower_limits_initial);
		upper_limits_initial.input(sb_in->upper_limits_initial);
	}
	n_fourier_modes = sb_in->n_fourier_modes;
	n_contour_bumps = sb_in->n_contour_bumps;
	if (n_fourier_modes > 0) {
		fourier_mode_mvals.input(sb_in->fourier_mode_mvals);
		fourier_mode_cosamp.input(sb_in->fourier_mode_cosamp);
		fourier_mode_sinamp.input(sb_in->fourier_mode_sinamp);
		fourier_mode_paramnum.input(sb_in->fourier_mode_paramnum);
	}
	include_boxiness_parameter = sb_in->include_boxiness_parameter;
	include_truncation_radius = sb_in->include_truncation_radius;
	if (include_boxiness_parameter) c0 = sb_in->c0;
	if (include_truncation_radius) rt = sb_in->rt;
	if (n_contour_bumps > 0) {
		bump_drvals.input(sb_in->bump_drvals);
		bump_xcvals.input(sb_in->bump_xcvals);
		bump_ycvals.input(sb_in->bump_ycvals);
		bump_sigvals.input(sb_in->bump_sigvals);
		bump_e1vals.input(sb_in->bump_e1vals);
		bump_e2vals.input(sb_in->bump_e2vals);
		bump_qvals.input(sb_in->bump_qvals);
		bump_phivals.input(sb_in->bump_phivals);
		bump_xcprime.input(sb_in->bump_xcprime);
		bump_ycprime.input(sb_in->bump_ycprime);
		bump_paramnum.input(sb_in->bump_paramnum);
	}

	assign_param_pointers();
}

void SB_Profile::set_nparams(const int &n_params_in)
{
	n_params = n_params_in;
	center_anchored = false;
	include_boxiness_parameter = false;
	include_truncation_radius = false;
	is_lensed = true; // default
	zoom_subgridding = false; // default
	n_fourier_modes = 0;
	n_contour_bumps = 0;
	n_vary_params = 0;
	vary_params.input(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.input(n_params);
	set_auto_penalty_limits.input(n_params);
	penalty_lower_limits.input(n_params);
	penalty_upper_limits.input(n_params);

	param = new double*[n_params];
	for (int i=0; i < n_params; i++) {
		vary_params[i] = false;
	}
}

void SB_Profile::anchor_center_to_lens(LensProfile** center_anchor_list, const int &center_anchor_lens_number)
{
	if (!center_anchored) center_anchored = true;
	center_anchor_lens = center_anchor_list[center_anchor_lens_number];
	x_center = center_anchor_lens->x_center;
	y_center = center_anchor_lens->y_center;
}

void SB_Profile::delete_center_anchor()
{
	if (center_anchored) {
		center_anchored = false;
		center_anchor_lens = NULL;
	}
}

void SB_Profile::add_boxiness_parameter(const double c0_in, const bool vary_c0)
{
	if (include_boxiness_parameter) return;
	include_boxiness_parameter = true;
	c0 = c0_in;
	n_params++;

	vary_params.resize(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.resize(n_params);
	set_auto_penalty_limits.resize(n_params);
	penalty_lower_limits.resize(n_params);
	penalty_upper_limits.resize(n_params);
	if (vary_c0) n_vary_params++;

	vary_params[n_params-1] = vary_c0;
	paramnames[n_params-1] = "c0";
	latex_paramnames[n_params-1] = "c";
	latex_param_subscripts[n_params-1] = "0";
	stepsizes[n_params-1] = 0.1; // arbitrary
	set_auto_penalty_limits[n_params-1] = false;

	//delete[] param;
	//param = new double*[n_params];
	//assign_param_pointers();

	double **new_param = new double*[n_params];
	for (int i=0; i < n_params-1; i++) new_param[i] = param[i];
	new_param[n_params-1] = &c0;
	delete[] param;
	param = new_param;
}

void SB_Profile::add_truncation_radius(const double rt_in, const bool vary_rt)
{
	if (include_truncation_radius) return;
	include_truncation_radius = true;
	rt = rt_in;
	n_params++;

	vary_params.resize(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.resize(n_params);
	set_auto_penalty_limits.resize(n_params);
	penalty_lower_limits.resize(n_params);
	penalty_upper_limits.resize(n_params);
	if (vary_rt) n_vary_params++;

	vary_params[n_params-1] = vary_rt;
	paramnames[n_params-1] = "rt";
	latex_paramnames[n_params-1] = "r";
	latex_param_subscripts[n_params-1] = "t";
	stepsizes[n_params-1] = 0.1; // arbitrary
	set_auto_penalty_limits[n_params-1] = false;

	double **new_param = new double*[n_params];
	for (int i=0; i < n_params-1; i++) new_param[i] = param[i];
	new_param[n_params-1] = &rt;
	delete[] param;
	param = new_param;

	//delete[] param;
	//param = new double*[n_params];
	//assign_param_pointers();
}

void SB_Profile::add_contour_bump(const double amp, const double x, const double y, const double sig, const double e1, const double e2)
{
	n_contour_bumps++;
	bump_paramnum.resize(n_contour_bumps);
	bump_drvals.resize(n_contour_bumps);
	bump_xcvals.resize(n_contour_bumps);
	bump_ycvals.resize(n_contour_bumps);
	bump_sigvals.resize(n_contour_bumps);
	bump_e1vals.resize(n_contour_bumps);
	bump_e2vals.resize(n_contour_bumps);
	bump_phivals.resize(n_contour_bumps);
	bump_qvals.resize(n_contour_bumps);
	bump_xcprime.resize(n_contour_bumps);
	bump_ycprime.resize(n_contour_bumps);
	bump_paramnum[n_contour_bumps-1] = n_params;
	bump_drvals[n_contour_bumps-1] = amp;
	bump_xcvals[n_contour_bumps-1] = x;
	bump_ycvals[n_contour_bumps-1] = y;
	bump_sigvals[n_contour_bumps-1] = sig;
	bump_e1vals[n_contour_bumps-1] = e1;
	bump_e2vals[n_contour_bumps-1] = e2;
	n_params += 6;

	vary_params.resize(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.resize(n_params);
	set_auto_penalty_limits.resize(n_params);
	penalty_lower_limits.resize(n_params);
	penalty_upper_limits.resize(n_params);

	for (int i=6; i >= 1; i--) vary_params[n_params-i] = false;

	stringstream nstream;
	string nstring;
	int nbump = n_contour_bumps-1;
	nstream << nbump;
	nstream >> nstring;

	paramnames[n_params-6] = "cb" + nstring + "_dr";
	latex_paramnames[n_params-6] = "A";
	latex_param_subscripts[n_params-6] = "\\delta s" + nstring;
	paramnames[n_params-5] = "cb" + nstring + "_xc";
	latex_paramnames[n_params-5] = "x";
	latex_param_subscripts[n_params-5] = "\\delta s" + nstring;
	paramnames[n_params-4] = "cb" + nstring + "_yc";
	latex_paramnames[n_params-4] = "y";
	latex_param_subscripts[n_params-4] = "\\delta s" + nstring;
	paramnames[n_params-3] = "cb" + nstring + "_sig";
	latex_paramnames[n_params-3] = "\\sigma";
	latex_param_subscripts[n_params-3] = "\\delta s" + nstring;
	paramnames[n_params-2] = "cb" + nstring + "_e1";
	latex_paramnames[n_params-2] = "e";
	latex_param_subscripts[n_params-2] = "1,\\delta s" + nstring;
	paramnames[n_params-1] = "cb" + nstring + "_e2";
	latex_paramnames[n_params-1] = "e";
	latex_param_subscripts[n_params-1] = "2,\\delta s" + nstring;

	stepsizes[n_params-6] = 0.01; // arbitrary
	set_auto_penalty_limits[n_params-6] = false;
	stepsizes[n_params-5] = 0.01; // arbitrary
	set_auto_penalty_limits[n_params-5] = false;
	stepsizes[n_params-4] = 0.01; // arbitrary
	set_auto_penalty_limits[n_params-4] = false;
	stepsizes[n_params-3] = 0.01; // arbitrary
	set_auto_penalty_limits[n_params-3] = false;
	stepsizes[n_params-2] = 0.05; // arbitrary
	set_auto_penalty_limits[n_params-2] = false;
	stepsizes[n_params-1] = 0.05; // arbitrary
	set_auto_penalty_limits[n_params-1] = false;

	double **new_param = new double*[n_params];
	for (int i=0; i < n_params-6; i++) new_param[i] = param[i];
	new_param[n_params-6] = &bump_drvals[n_contour_bumps-1];
	new_param[n_params-5] = &bump_xcvals[n_contour_bumps-1];
	new_param[n_params-4] = &bump_ycvals[n_contour_bumps-1];
	new_param[n_params-3] = &bump_sigvals[n_contour_bumps-1];
	new_param[n_params-2] = &bump_e1vals[n_contour_bumps-1];
	new_param[n_params-1] = &bump_e2vals[n_contour_bumps-1];
	delete[] param;
	param = new_param;

	update_ellipticity_meta_parameters();

	//delete[] param;
	//param = new double*[n_params];
	//assign_param_pointers();
}

void SB_Profile::add_fourier_mode(const int m_in, const double amp_in, const double phi_in, const bool vary1, const bool vary2)
{
	n_fourier_modes++;
	fourier_mode_mvals.resize(n_fourier_modes);
	fourier_mode_cosamp.resize(n_fourier_modes);
	fourier_mode_sinamp.resize(n_fourier_modes);
	fourier_mode_paramnum.resize(n_fourier_modes);
	fourier_mode_mvals[n_fourier_modes-1] = m_in;
	fourier_mode_cosamp[n_fourier_modes-1] = amp_in;
	fourier_mode_sinamp[n_fourier_modes-1] = phi_in;
	fourier_mode_paramnum[n_fourier_modes-1] = n_params;
	n_params += 2;
	vary_params.resize(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.resize(n_params);
	set_auto_penalty_limits.resize(n_params);
	penalty_lower_limits.resize(n_params);
	penalty_upper_limits.resize(n_params);
	if (vary1) n_vary_params++;
	if (vary2) n_vary_params++;

	vary_params[n_params-2] = vary1;
	vary_params[n_params-1] = vary2;
	stringstream mstream;
	string mstring;
	mstream << m_in;
	mstream >> mstring;
	paramnames[n_params-2] = "A_" + mstring;
	paramnames[n_params-1] = "B_" + mstring;
	latex_paramnames[n_params-2] = "A";
	latex_param_subscripts[n_params-2] = mstring;
	latex_paramnames[n_params-1] = "B";
	latex_param_subscripts[n_params-1] = mstring;
	stepsizes[n_params-2] = 0.005; // arbitrary
	stepsizes[n_params-1] = 0.005; // arbitrary
	set_auto_penalty_limits[n_params-2] = false;
	set_auto_penalty_limits[n_params-1] = false;
	//for (int i=0; i < n_params; i++) cout << stepsizes[i] << " ";
	//cout << endl;

	delete[] param;
	param = new double*[n_params];
	assign_param_pointers();

	double **new_param = new double*[n_params];
	for (int i=0; i < n_params-2; i++) new_param[i] = param[i];
	new_param[n_params-2] = &fourier_mode_cosamp[n_fourier_modes-1];
	new_param[n_params-1] = &fourier_mode_sinamp[n_fourier_modes-1];
	delete[] param;
	param = new_param;
}

void SB_Profile::remove_fourier_modes()
{
	// This is not well-written, because it assumes the last parameters are fourier modes. Have it actually check the parameter name
	// before deleting it. FIX!
	n_params -= n_fourier_modes*2;
	vary_params.resize(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.resize(n_params);
	set_auto_penalty_limits.resize(n_params);
	penalty_lower_limits.resize(n_params);
	penalty_upper_limits.resize(n_params);
	n_vary_params = 0;
	for (int i=0; i < n_params; i++) if (vary_params[i]) n_vary_params++;

	n_fourier_modes = 0;
	fourier_mode_mvals.resize(n_fourier_modes);
	fourier_mode_sinamp.resize(n_fourier_modes);
	fourier_mode_cosamp.resize(n_fourier_modes);
	fourier_mode_paramnum.resize(n_fourier_modes);

	delete[] param;
	param = new double*[n_params];
	assign_param_pointers();
}

void SB_Profile::assign_param_pointers()
{
	param[0] = &qx_parameter;
	param[1] = &f_parameter;
	set_geometric_param_pointers(2);
}

void SB_Profile::set_geometric_param_pointers(int qi)
{
	// Sets parameter pointers for ellipticity (or axis ratio) and angle
	if (use_sb_ellipticity_components) {
		param[qi++] = &epsilon;
		param[qi++] = &epsilon2;
		angle_paramnum = -1; // there is no angle parameter if ellipticity components are being used
	} else {
		param[qi++] = &q;
		param[qi] = &theta;
		angle_paramnum = qi++;
	}
	param[qi++] = &x_center;
	param[qi++] = &y_center;
	if (include_boxiness_parameter) param[qi++] = &c0;
	if (include_truncation_radius) param[qi++] = &rt;
	if (n_contour_bumps > 0) {
		for (int i=0; i < n_contour_bumps; i++) {
			param[qi++] = &bump_drvals[i];
			param[qi++] = &bump_xcvals[i];
			param[qi++] = &bump_ycvals[i];
			param[qi++] = &bump_sigvals[i];
			param[qi++] = &bump_e1vals[i];
			param[qi++] = &bump_e2vals[i];
		}
	}
	if (n_fourier_modes > 0) {
		for (int i=0; i < n_fourier_modes; i++) {
			param[qi++] = &fourier_mode_cosamp[i];
			param[qi++] = &fourier_mode_sinamp[i];
		}
	}
}

bool SB_Profile::vary_parameters(const boolvector& vary_params_in)
{
	if (vary_params_in.size() != n_params) {
		return false;
	}
	// Save the old limits, if they exist
	dvector old_lower_limits(n_params);
	dvector old_upper_limits(n_params);
	int i=0,k=0;
	for (i=0; i < n_params; i++) {
		if (vary_params[i]) {
			old_lower_limits[i] = lower_limits[k];
			old_upper_limits[i] = upper_limits[k];
			k++;
		} else {
			old_lower_limits[i] = -1e30;
			old_upper_limits[i] = 1e30;
		}
	}
	if (k != n_vary_params) die("k != n_vary_params");

	n_vary_params=0;
	for (i=0; i < vary_params_in.size(); i++) {
		vary_params[i] = vary_params_in[i];
		if (vary_params_in[i]) {
			n_vary_params++;
		}
	}

	lower_limits.input(n_vary_params);
	upper_limits.input(n_vary_params);
	k=0;
	for (i=0; i < n_params; i++) {
		if (vary_params[i]) {
			lower_limits[k] = old_lower_limits[i];
			upper_limits[k] = old_upper_limits[i];
			k++;
		}
	}
	if (k != n_vary_params) die("k != n_vary_params");
	lower_limits_initial.input(lower_limits);
	upper_limits_initial.input(upper_limits);

	return true;
}

void SB_Profile::set_limits(const dvector& lower, const dvector& upper)
{
	include_limits = true;
	if (lower.size() != n_vary_params) die("number of parameters with lower limits does not match number of variable parameters");
	if (upper.size() != n_vary_params) die("number of parameters with upper limits does not match number of variable parameters");
	lower_limits = lower;
	upper_limits = upper;
	lower_limits_initial = lower;
	upper_limits_initial = upper;
}

void SB_Profile::set_limits(const dvector& lower, const dvector& upper, const dvector& lower_init, const dvector& upper_init)
{
	include_limits = true;
	if (lower.size() != n_vary_params) die("number of parameters with lower limits does not match number of variable parameters");
	if (upper.size() != n_vary_params) die("number of parameters with upper limits does not match number of variable parameters");
	lower_limits = lower;
	upper_limits = upper;
	lower_limits_initial = lower_init;
	upper_limits_initial = upper_init;
}

void SB_Profile::get_parameters(double* params)
{
	for (int i=0; i < n_params; i++) {
		if (i==angle_paramnum) params[i] = radians_to_degrees(*(param[i]));
		else params[i] = *(param[i]);
	}
}

void SB_Profile::update_parameters(const double* params)
{
	for (int i=0; i < n_params; i++) {
		if (i==angle_paramnum) *(param[i]) = degrees_to_radians(params[i]);
		else *(param[i]) = params[i];
	}
	update_meta_parameters();
}

bool SB_Profile::update_specific_parameter(const string name_in, const double& value)
{
	double* newparams = new double[n_params];
	get_parameters(newparams);
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if (paramnames[i]==name_in) {
			found_match = true;
			newparams[i] = value;
			break;
		}
	}
	if (found_match) update_parameters(newparams);
	delete[] newparams;
	return found_match;
}

void SB_Profile::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		for (int i=0; i < n_params; i++) {
			if (vary_params[i]==true) {
				if (i==angle_paramnum) {
					*(param[i]) = degrees_to_radians(fitparams[index++]);
					update_angle_meta_params();
				}
				else *(param[i]) = fitparams[index++];
			}
		}
		update_meta_parameters();
	}
}

void SB_Profile::update_anchor_center()
{
	if (center_anchored) {
		x_center = center_anchor_lens->x_center;
		y_center = center_anchor_lens->y_center;
	}
}

void SB_Profile::get_fit_parameters(dvector& fitparams, int &index)
{
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]==true) {
			if (i==angle_paramnum) fitparams[index++] = radians_to_degrees(*(param[i]));
			else fitparams[index++] = *(param[i]);
		}
	}
}

void SB_Profile::set_auto_stepsizes()
{
	stepsizes[0] = 0.1;
	stepsizes[1] = 0.1;
	set_auto_eparam_stepsizes(2,3);
	stepsizes[4] = 1.0;
	stepsizes[5] = 1.0;
	stepsizes[6] = 0.1;
}

void SB_Profile::set_auto_eparam_stepsizes(int eparam1_i, int eparam2_i)
{
	if (use_sb_ellipticity_components) {
		stepsizes[eparam1_i] = 0.1; // e1
		stepsizes[eparam2_i] = 0.1; // e2
	} else {
		stepsizes[eparam1_i] = 0.1; // q
		stepsizes[eparam2_i] = 20;  // angle stepsize
	}
}

void SB_Profile::get_auto_stepsizes(dvector& stepsizes_in, int &index)
{
	set_auto_stepsizes();
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]) stepsizes_in[index++] = stepsizes[i];
	}
}

void SB_Profile::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false;
	set_auto_penalty_limits[1] = false;
	set_geometric_param_auto_ranges(2);
}

void SB_Profile::set_geometric_param_auto_ranges(int param_i)
{
	if (use_sb_ellipticity_components) {
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = -1; penalty_upper_limits[param_i] = 1; param_i++;
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = -1; penalty_upper_limits[param_i] = 1; param_i++;
	} else {
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 5e-3; penalty_upper_limits[param_i] = 1; param_i++;
		set_auto_penalty_limits[param_i++] = false;
	}
	set_auto_penalty_limits[param_i++] = false;
	set_auto_penalty_limits[param_i++] = false;
}

void SB_Profile::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	set_auto_ranges();
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]) {
			if (set_auto_penalty_limits[i]) {
				use_penalty_limits[index] = true;
				lower[index] = penalty_lower_limits[i];
				upper[index] = penalty_upper_limits[i];
			} else {
				use_penalty_limits[index] = false;
			}
			index++;
		}
	}
}

void SB_Profile::get_fit_parameter_names(vector<string>& paramnames_vary, vector<string> *latex_paramnames_vary, vector<string> *latex_subscripts_vary)
{
	int i;
	for (i=0; i < n_params; i++) {
		if (vary_params[i]) {
			paramnames_vary.push_back(paramnames[i]);
			//cout << "NAME: " << paramnames_vary[i] << endl;
			if (latex_paramnames_vary != NULL) latex_paramnames_vary->push_back(latex_paramnames[i]);
			if (latex_subscripts_vary != NULL) latex_subscripts_vary->push_back(latex_param_subscripts[i]);
		}
	}
}

bool SB_Profile::get_limits(dvector& lower, dvector& upper, dvector& lower0, dvector& upper0, int &index)
{
	if ((include_limits==false) or (lower_limits.size() != n_vary_params)) return false;
	for (int i=0; i < n_vary_params; i++) {
		lower[index] = lower_limits[i];
		upper[index] = upper_limits[i];
		lower0[index] = lower_limits_initial[i];
		upper0[index] = upper_limits_initial[i];
		index++;
	}
	return true;
}

void SB_Profile::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "qx"; latex_paramnames[0] = "q"; latex_param_subscripts[0] = "x";
	paramnames[1] = "f"; latex_paramnames[1] = "f"; latex_param_subscripts[1] = "";
	set_geometric_paramnames(2);
}

void SB_Profile::set_geometric_paramnames(int qi)
{
	if (use_sb_ellipticity_components) {
		paramnames[qi] = "e1"; latex_paramnames[qi] = "e"; latex_param_subscripts[qi] = "1,src"; qi++;
		paramnames[qi] = "e2"; latex_paramnames[qi] = "e"; latex_param_subscripts[qi] = "2,src"; qi++;
	} else {
		paramnames[qi] = "q"; latex_paramnames[qi] = "q"; latex_param_subscripts[qi] = "src"; qi++;
		paramnames[qi] = "theta"; latex_paramnames[qi] = "\\theta"; latex_param_subscripts[qi] = "src"; qi++;
	}
	paramnames[qi] = "xc"; latex_paramnames[qi] = "x"; latex_param_subscripts[qi] = "c,src"; qi++;
	paramnames[qi] = "yc"; latex_paramnames[qi] = "y"; latex_param_subscripts[qi] = "c,src"; qi++;
}

void SB_Profile::set_geometric_parameters(const double &q1_in, const double &q2_in, const double &xc_in, const double &yc_in)
{
	qx_parameter = 1.0;

	if (use_sb_ellipticity_components) {
		epsilon = q1_in;
		epsilon2 = q2_in;
	} else {
		q = q1_in;
		if (q < 0) q = -q; // don't allow negative axis ratios
		if (q > 1) q = 1.0; // don't allow q>1
		theta = degrees_to_radians(q2_in);
	}
	x_center = xc_in;
	y_center = yc_in;
	update_ellipticity_meta_parameters();
}

void SB_Profile::set_geometric_parameters_radians(const double &q_in, const double &theta_in, const double &xc_in, const double &yc_in)
{
	qx_parameter = 1.0;
	q=q_in;
	if (q < 0) q = -q; // don't allow negative axis ratios
	if (q > 1) q = 1.0; // don't allow q>1
	set_angle_radians(theta_in);
	x_center = xc_in;
	y_center = yc_in;
}

void SB_Profile::calculate_ellipticity_components()
{
	if (use_sb_ellipticity_components) {
		double theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		epsilon = (1-q)*cos(2*theta_eff);
		epsilon2 = (1-q)*sin(2*theta_eff);
	}
}

void SB_Profile::update_ellipticity_meta_parameters()
{
	if (use_sb_ellipticity_components) {
		q = 1 - sqrt(SQR(epsilon) + SQR(epsilon2));
		set_angle_from_components(epsilon,epsilon2); // note this will automatically set the costheta, sintheta parameters
	} else {
		update_angle_meta_params(); // sets the costheta, sintheta meta-parameters
	}
	if (n_contour_bumps > 0) {
		for (int i=0; i < n_contour_bumps; i++) {
			bump_xcprime[i] = bump_xcvals[i] - x_center;
			bump_ycprime[i] = bump_ycvals[i] - y_center;
			if (theta != 0) rotate(bump_xcprime[i],bump_ycprime[i]);

			bump_qvals[i] = 1 - sqrt(SQR(bump_e1vals[i]) + SQR(bump_e2vals[i]));
			if (bump_qvals[i] < 0) bump_qvals[i] = 0.001;
			if (bump_e1vals[i]==0) {
				if (bump_e2vals[i] > 0) bump_phivals[i] = M_HALFPI;
				else bump_phivals[i] = -M_HALFPI;
			} else {
				bump_phivals[i] = atan(abs(bump_e2vals[i]/bump_e1vals[i]));
				if (bump_e1vals[i] < 0) {
					if (bump_e2vals[i] < 0)
						bump_phivals[i] = bump_phivals[i] - M_PI;
					else
						bump_phivals[i] = M_PI - bump_phivals[i];
				} else if (bump_e2vals[i] < 0) {
					bump_phivals[i] = -bump_phivals[i];
				}
			}
			bump_phivals[i] = 0.5*bump_phivals[i];
			while (bump_phivals[i] > M_PI) bump_phivals[i] -= M_PI;
			while (bump_phivals[i] <= 0) bump_phivals[i] += M_PI;
		}
	}
}

void SB_Profile::update_angle_meta_params()
{
	// trig functions are stored to save computation time later
	costheta = cos(theta);
	sintheta = sin(theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		double tmp = sintheta;
		sintheta = costheta;
		costheta = -tmp;
	}
}

double SB_Profile::sb_rsq(const double rsq) // this function should be redefined in all derived classes
{
	double r = sqrt(rsq);
	if (r < qx_parameter*sb_spline.xmin()) return (f_parameter*sb_spline.extend_inner_logslope(r/qx_parameter));
	if (r > qx_parameter*sb_spline.xmax()) return (f_parameter*sb_spline.extend_outer_logslope(r/qx_parameter));
	return (f_parameter*sb_spline.splint(r/qx_parameter));
}

void SB_Profile::shift_angle_90()
{
	// do this if the major axis orientation is changed (so the lens angles values are changed appropriately, even though the lens doesn't change)
	theta += M_HALFPI;
	while (theta > M_PI) theta -= M_PI;
}

void SB_Profile::shift_angle_minus_90()
{
	// do this if the major axis orientation is changed (so the lens angles values are changed appropriately, even though the lens doesn't change)
	theta -= M_HALFPI;
	while (theta <= -M_PI) theta += M_PI;
}

void SB_Profile::reset_angle_modulo_2pi()
{
	while (theta < -M_PI/2) theta += 2*M_PI;
	while (theta > 2*M_PI) theta -= 2*M_PI;
}

void SB_Profile::set_angle(const double &theta_degrees)
{
	theta = degrees_to_radians(theta_degrees);
	// trig functions are stored to save computation time later
	costheta = cos(theta);
	sintheta = sin(theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		double tmp = sintheta;
		sintheta = costheta;
		costheta = -tmp;
	}
}

void SB_Profile::set_angle_radians(const double &theta_in)
{
	theta = theta_in;
	// trig functions are stored to save computation time later
	costheta = cos(theta);
	sintheta = sin(theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		double tmp = sintheta;
		sintheta = costheta;
		costheta = -tmp;
	}
}

void SB_Profile::set_angle_from_components(const double &comp1, const double &comp2)
{
	double angle;
	if (comp1==0) {
		if (comp2 > 0) angle = M_HALFPI;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(comp2/comp1));
		if (comp1 < 0) {
			if (comp2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (comp2 < 0) {
			angle = -angle;
		}
	}
	angle = 0.5*angle;
	if (orient_major_axis_north) angle -= M_HALFPI;
	//while (angle > M_HALFPI) angle -= M_PI;
	//while (angle <= -M_HALFPI) angle += M_PI;
	while (angle > M_PI) angle -= M_PI;
	while (angle <= 0) angle += M_PI;
	set_angle_radians(angle);
}

inline void SB_Profile::rotate(double &x, double &y)
{
	// perform a counter-clockwise rotation of the coordinate system to match the coordinate system of the rotated galaxy
	double xp = x*costheta + y*sintheta;
	y = -x*sintheta + y*costheta;
	x = xp;
}

inline void SB_Profile::rotate_back(double &x, double &y)
{
	// perform a counter-clockwise rotation of the coordinate system to match the coordinate system of the rotated galaxy
	double xp = x*costheta - y*sintheta;
	y = x*sintheta + y*costheta;
	x = xp;
}

double SB_Profile::surface_brightness(double x, double y)
{
	// switch to coordinate system centered on surface brightness profile
	x -= x_center;
	y -= y_center;
	if (theta != 0) rotate(x,y);
	double phi_q; // used for Fourier modes
	if (n_fourier_modes > 0) {
		phi_q = atan(abs(y/(q*x)));
		if (x < 0) {
			if (y < 0)
				phi_q = phi_q - M_PI;
			else
				phi_q = M_PI - phi_q;
		} else if (y < 0) {
			phi_q = -phi_q;
		}
	}

	double rsq, rsq_ell = x*x + y*y/(q*q);
	if ((include_boxiness_parameter) and (c0 != 0.0)) {
		rsq = pow(pow(abs(x),c0+2.0) + pow(abs(y/q),c0+2.0),2.0/(c0+2.0));
	} else {
		rsq = rsq_ell;
	}
	if ((n_fourier_modes > 0) or (n_contour_bumps > 0)) {
		double fourier_factor = 1.0;
		if (use_fmode_scaled_amplitudes) {
			for (int i=0; i < n_fourier_modes; i++) {
				fourier_factor += (fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*phi_q) + fourier_mode_sinamp[i]*sin(fourier_mode_mvals[i]*phi_q))/fourier_mode_mvals[i];
			}
		} else {
			for (int i=0; i < n_fourier_modes; i++) {
				fourier_factor += fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*phi_q) + fourier_mode_sinamp[i]*sin(fourier_mode_mvals[i]*phi_q);
			}
		}
		if (n_contour_bumps > 0) {
			double phi_degrees;
			double r = sqrt(rsq_ell);
			double xprime, yprime, temp;
			for (int i=0; i < n_contour_bumps; i++) {
				//phi_degrees = phi_q*180/M_PI;
				//while (abs(phi_degrees-bump_xcvals[i]) > 180) {
					//if (phi_degrees > bump_xcvals[i]) phi_degrees -= 360;
					//else phi_degrees += 360;
				//}
				xprime = x - bump_xcprime[i];
				yprime = y - bump_ycprime[i];
				temp = xprime*cos(bump_phivals[i]) + yprime*sin(bump_phivals[i]);
				yprime = -xprime*sin(bump_phivals[i]) + yprime*cos(bump_phivals[i]);
				xprime = temp;

				fourier_factor -= (bump_drvals[i]/r)*exp(-(SQR(xprime) + SQR(yprime/bump_qvals[i]))/SQR(bump_sigvals[i])/2);
			}
		}
		rsq *= fourier_factor*fourier_factor;
	}
	double sb = sb_rsq(rsq);
	if (include_truncation_radius) sb *= pow(1+pow(rsq/(rt*rt),3),-2);
	//if (include_truncation_radius) {
		//double fac = pow(1.0+pow(rsq/(rt*rt),1.5),-1.33333);
		//cout << "r=" << sqrt(rsq) << " rt=" << rt << " fac=" << fac << endl;
		//sb *= fac;
	//}
	return sb;
}

//double SB_Profile::surface_brightness_zoom(const double x, const double y, const double pixel_xlength, const double pixel_ylength)
//double SB_Profile::surface_brightness_zoom(lensvector &centerpt, lensvector &pt1, lensvector &pt2, lensvector &pt3, lensvector &pt4)
//{
	//return 0; // cop-out, because we're only using this feature for Gaussian profiles at the moment
//}

double SB_Profile::surface_brightness_zoom(lensvector &centerpt, lensvector &pt1, lensvector &pt2, lensvector &pt3, lensvector &pt4)
{
	//double pt1[0], pt2[0], pt1[1], pt2[1];
	bool subgrid = false;
	//pt1[0] = x - pixel_xlength/2;
	//pt2[0] = x + pixel_xlength/2;
	//pt1[1] = y - pixel_ylength/2;
	//pt2[1] = y + pixel_ylength/2;

	double scale = zoom_scale*length_scale();
	lensvector sbcenter; sbcenter[0] = x_center; sbcenter[1] = y_center;
	lensvector d1, d2, d3;
	double product1, product2, product3;
	lensvector scpt1, scpt2, scpt3, scpt4;
	//cout << "SCALE : " << scale << endl;
	d1[0] = pt1[0]-centerpt[0];
	d1[1] = pt1[1]-centerpt[1];
	d2[0] = d1[0] * (scale/d1.norm()); // vector along d1 with length given by scale
	d2[1] = d1[1] * (scale/d1.norm()); // vector along d1 with length given by scale
	//d2 = d1 * (3*scale/d1.norm()); // vector along d1 with length given by scale
	//cout << "ANLGE: " << d1.angle() << " " << d2.angle() << endl;
	//cout << "WTF? " << d2.norm() << " " << (20*scale) << endl;
	scpt1 = pt1 + d2;
	//scpt1 = centerpt + d2;

	d1[0] = pt2[0]-centerpt[0];
	d1[1] = pt2[1]-centerpt[1];
	d1 = pt2-centerpt;
	d2[0] = d1[0] * (scale/d1.norm()); // vector along d1 with length given by scale
	d2[1] = d1[1] * (scale/d1.norm()); // vector along d1 with length given by scale
	//cout << "ANLGE: " << d1.angle() << " " << d2.angle() << endl;
	scpt2 = pt2 + d2;
	//scpt2 = centerpt + d2;

	d1[0] = pt3[0]-centerpt[0];
	d1[1] = pt3[1]-centerpt[1];
	d1 = pt3-centerpt;
	d2[0] = d1[0] * (scale/d1.norm()); // vector along d1 with length given by scale
	d2[1] = d1[1] * (scale/d1.norm()); // vector along d1 with length given by scale
	//d2 = d1 * (3*scale/d1.norm()); // vector along d1 with length given by scale
	//cout << "ANLGE: " << d1.angle() << " " << d2.angle() << endl;
	scpt3 = pt3 + d2;
	//scpt3 = centerpt + d2;

	d1[0] = pt4[0]-centerpt[0];
	d1[1] = pt4[1]-centerpt[1];
	d1 = pt4-centerpt;
	d2[0] = d1[0] * (scale/d1.norm()); // vector along d1 with length given by scale
	d2[1] = d1[1] * (scale/d1.norm()); // vector along d1 with length given by scale
	//d2 = d1 * (3*scale/d1.norm()); // vector along d1 with length given by scale
	//cout << "ANLGE: " << d1.angle() << " " << d2.angle() << endl;
	scpt4 = pt4 + d2;
	//scpt4 = centerpt + d2;

	d1[0] = sbcenter[0] - scpt3[0];
	d1[1] = sbcenter[1] - scpt3[1];
	d2[0] = sbcenter[0] - scpt2[0];
	d2[1] = sbcenter[1] - scpt2[1];
	d3[0] = sbcenter[0] - scpt1[0];
	d3[1] = sbcenter[1] - scpt1[1];
	product1 = d1 ^ d2;
	product2 = d3 ^ d1;
	product3 = d2 ^ d3;
	if ((product1 > 0) and (product2 > 0) and (product3 > 0)) subgrid = true;
	else if ((product1 < 0) and (product2 < 0) and (product3 < 0)) subgrid = true;
	else {
		d3[0] = sbcenter[0] - scpt4[0];
		d3[1] = sbcenter[1] - scpt4[1];
		product2 = d3 ^ d1;
		product3 = d2 ^ d3;
		if ((product1 > 0) and (product2 > 0) and (product3 > 0)) subgrid = true;
		else if ((product1 < 0) and (product2 < 0) and (product3 < 0)) subgrid = true;
	}

	// Now find the middle point of each side, and see if it is close enough to sbcenter
	// This could still fail if the cell is very large, and sbcenter is only close to some off-center point
	// on one of the edges, but hopefully it's enough to get the job done.
	if (!subgrid) {
		d1[0] = 0.5*(pt1[0] + pt2[0]);
		d1[1] = 0.5*(pt1[1] + pt2[1]);
		d2[0] = d1[0] - sbcenter[0];
		d2[1] = d1[1] - sbcenter[1];
		if (d2.norm() < scale) subgrid = true;

		d1[0] = 0.5*(pt1[0] + pt3[0]);
		d1[1] = 0.5*(pt1[1] + pt3[1]);
		d2[0] = d1[0] - sbcenter[0];
		d2[1] = d1[1] - sbcenter[1];
		if (d2.norm() < scale) subgrid = true;

		d1[0] = 0.5*(pt2[0] + pt4[0]);
		d1[1] = 0.5*(pt2[1] + pt4[1]);
		d2[0] = d1[0] - sbcenter[0];
		d2[1] = d1[1] - sbcenter[1];
		if (d2.norm() < scale) subgrid = true;

		d1[0] = 0.5*(pt3[0] + pt4[0]);
		d1[1] = 0.5*(pt3[1] + pt4[1]);
		d2[0] = d1[0] - sbcenter[0];
		d2[1] = d1[1] - sbcenter[1];
		if (d2.norm() < scale) subgrid = true;
	}

	/*
	if ((iii==25) and (jjj==13)) {
		//if (subgrid) cout << "SUBGRIDDING!" << endl;
		//else cout << " NOT SUBGRIDDING! WTF???!!!!" << endl;
		ofstream blargh("blargh.dat");
		blargh << pt1[0] << " " << pt1[1] << endl;
		blargh << pt2[0] << " " << pt2[1] << endl;
		blargh << pt4[0] << " " << pt4[1] << endl;
		blargh << pt3[0] << " " << pt3[1] << endl;
		blargh << pt1[0] << " " << pt1[1] << endl << endl;
		blargh << scpt1[0] << " " << scpt1[1] << endl;
		blargh << scpt2[0] << " " << scpt2[1] << endl;
		blargh << scpt4[0] << " " << scpt4[1] << endl;
		blargh << scpt3[0] << " " << scpt3[1] << endl;
		blargh << scpt1[0] << " " << scpt1[1] << endl << endl;
		ofstream blarghpt("blarghpt.dat");
		blarghpt << sbcenter[0] << " " << sbcenter[1] << endl;
		ofstream blarghpt2("blarghpt2.dat");
		blarghpt2 << centerpt[0] << " " << centerpt[1] << endl;
	}
	*/

	/*
	double pixel_xlength = dmax(pt2[0]-pt1[0],pt4[0]-pt3[0]); // a bit rough
	double pixel_ylength = dmax(pt3[1]-pt1[1],pt4[1]-pt2[1]); // a bit rough
	double pl = SQR(dmax(pixel_xlength,pixel_ylength))/4.0;
	double scale = length_scale();
	if ((pixel_xlength < scale/6) and (pixel_ylength < scale*q/6)) ; // grid already fine enough
	// Redo the below line using cross products!
	else if ((x_center >= pt1[0]) and (x_center <= pt2[0]) and (y_center >= pt1[1]) and (y_center <= pt2[1])) { subgrid = true; }
	else {
		double rsq;
		rsq = SQR(pt1[0]-x_center) + SQR(pt1[1]-y_center);
		if (rsq < 3*pl) subgrid = true;
		rsq = SQR(pt2[0]-x_center) + SQR(pt2[1]-y_center);
		if (rsq < 3*pl) subgrid = true;
		rsq = SQR(pt3[0]-x_center) + SQR(pt3[1]-y_center);
		if (rsq < 3*pl) subgrid = true;
		rsq = SQR(pt4[0]-x_center) + SQR(pt4[1]-y_center);
		if (rsq < 3*pl) subgrid = true;
	}
	*/
	if (!subgrid) return surface_brightness(centerpt[0],centerpt[1]);
	//if (subgrid) cout << "Subgridding! pt=" << centerpt[0] << "," << centerpt[1] << endl;
	double pixel_xlength, pixel_ylength;
	d1 = pt2 - pt1;
	d2 = pt4 - pt3;
	pixel_xlength = dmax(d1.norm(),d2.norm());
	d1 = pt3 - pt1;
	d2 = pt4 - pt2;
	pixel_ylength = dmax(d1.norm(),d2.norm());
	int xsplit, ysplit;
	double lscale = length_scale();
	xsplit = ((int) (zoom_split_factor*pixel_xlength/lscale)) + 1;
	ysplit = ((int) (zoom_split_factor*pixel_ylength/lscale)) + 1;
	//cout << "HI " << xsplit << " " << ysplit << endl;
	//nsplit = imax(xsplit,ysplit);
	//cout << "nsplit=" << nsplit << endl;
	double sb = 0;
	double u0, w0, xs, ys;
	int ii,jj;
	//double sbadd;
	//ofstream testsplit("testsplit.dat");

	for (ii=0; ii < xsplit; ii++) {
		u0 = ((double) (1+2*ii))/(2*xsplit);
		for (jj=0; jj < ysplit; jj++) {
			//xs = u0*pt1[0] + (1-u0)*pt2[0];
			w0 = ((double) (1+2*jj))/(2*ysplit);
			xs = (pt1[0]*u0 + pt2[0]*(1-u0))*w0 + (pt3[0]*u0 + pt4[0]*(1-u0))*(1-w0);
			ys = (pt1[1]*u0 + pt2[1]*(1-u0))*w0 + (pt3[1]*u0 + pt4[1]*(1-u0))*(1-w0);
			//ys = w0*pt1[1] + (1-w0)*pt2[1];
			//sbadd = surface_brightness(xs,ys);
			sb += surface_brightness(xs,ys);
			//testsplit << xs << " " << ys << endl;
		}
	}
	sb /= (xsplit*ysplit);
	//if (sb > 0) cout << "sb=" << sb << endl;

	return sb;
}



double SB_Profile::surface_brightness_r(const double r)
{
	return sb_rsq(r*r);
}

/*
double SB_Profile::calculate_Lmatrix_element(double x, double y, const int amp_index)
{
	return 0.0; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}
*/

void SB_Profile::calculate_Lmatrix_elements(double x, double y, double* Lmatrix_elements, const double weight)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

void SB_Profile::calculate_gradient_Rmatrix_elements(double** Rmatrix_elements, double &logdet)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}


void SB_Profile::update_amplitudes(double *ampvec)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

void SB_Profile::get_amplitudes(double *ampvec)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

void SB_Profile::print_parameters()
{
	cout << model_name;
	if (sbtype==SHAPELET) cout << "(n_shapelets=" << (*indxptr) << ")";
	if (!is_lensed) cout << "(unlensed)";
	if (zoom_subgridding) cout << "(zoom)";
	cout << ": ";
	for (int i=0; i < n_params; i++) {
		cout << paramnames[i] << "=";
		if (i==angle_paramnum) cout << radians_to_degrees(*(param[i])) << " degrees";
		else cout << *(param[i]);
		if (i != n_params-1) cout << ", ";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

void SB_Profile::print_vary_parameters()
{
	if (n_vary_params==0) {
		cout << "   parameters: none\n";
	} else {
		vector<string> paramnames_vary;
		get_fit_parameter_names(paramnames_vary);
		if (include_limits) {
			if (lower_limits_initial.size() != n_vary_params) cout << "   Warning: parameter limits not defined\n";
			else {
				cout << "   parameter limits:\n";
				for (int i=0; i < n_vary_params; i++) {
					if ((lower_limits_initial[i]==lower_limits[i]) and (upper_limits_initial[i]==upper_limits[i]))
						cout << "   " << paramnames_vary[i] << ": [" << lower_limits[i] << ":" << upper_limits[i] << "]\n";
					else
						cout << "   " << paramnames_vary[i] << ": [" << lower_limits[i] << ":" << upper_limits[i] << "], initial range: [" << lower_limits_initial[i] << ":" << upper_limits_initial[i] << "]\n";
				}
			}
		} else {
			cout << "   parameters: ";
			cout << paramnames_vary[0];
			for (int i=1; i < n_vary_params; i++) cout << ", " << paramnames_vary[i];
			cout << endl;
		}
	}
}

void SB_Profile::window_params(double& xmin, double& xmax, double& ymin, double& ymax)
{
	double rmax = window_rmax();
	xmin = -rmax;
	xmax = rmax;
	ymin = -q*rmax;
	ymax = q*rmax;
	if (theta != 0) {
		double xx[4], yy[4];
		xx[0]=xmin; yy[0]=ymin;
		xx[1]=xmax; yy[1]=ymin;
		xx[2]=xmax; yy[2]=ymax;
		xx[3]=xmin; yy[3]=ymax;
		xmin=1e30; xmax=-1e30; ymin=1e30; ymax=-1e30;
		for (int i=0; i < 4; i++) {
			rotate_back(xx[i],yy[i]);
			if (xx[i] < xmin) xmin=xx[i];
			if (xx[i] > xmax) xmax=xx[i];
			if (yy[i] < ymin) ymin=yy[i];
			if (yy[i] > ymax) ymax=yy[i];
		}
	}
	xmin += x_center;
	xmax += x_center;
	ymin += y_center;
	ymax += y_center;
}

double SB_Profile::window_rmax()
{
	return qx_parameter*sb_spline.xmax();
}

double SB_Profile::length_scale()
{
	return qx_parameter*sb_spline.xmax();
}

void SB_Profile::print_source_command(ofstream& scriptout, const bool use_limits)
{
	scriptout << setprecision(16);
	scriptout << "fit source " << model_name << " ";
	if (!is_lensed) scriptout << "-unlensed ";
	if (zoom_subgridding) scriptout << "-zoom ";

	for (int i=0; i < n_params; i++) {
		if (i==angle_paramnum) scriptout << radians_to_degrees(*(param[i]));
		else {
			// If this is an optional parameter, need to specify parameter name before the value
			if (paramnames[i]=="c0") scriptout << "c0="; // boxiness parameter
			else if (paramnames[i]=="rt") scriptout << "rt="; // truncation radius
			else {
				for (int j=0; j < n_fourier_modes; j++) {
					if (fourier_mode_paramnum[j]==i) scriptout << "f" << fourier_mode_mvals[j] << "="; // Fourier mode
				}
				for (int j=0; j < n_contour_bumps; j++) {
					if (bump_paramnum[j]==i) scriptout << "cb" << "="; // Fourier mode
				}
			}
			if (((*(param[i]) != 0.0) and (abs(*(param[i])) < 1e-3)) or (abs(*(param[i]))) > 1e3) output_field_in_sci_notation(param[i],scriptout,false);
			else scriptout << *(param[i]);
		}
		scriptout << " ";
	}
	string extra_arg;
	if (get_special_command_arg(extra_arg)) scriptout << extra_arg << " ";
	scriptout << endl;
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]) scriptout << "1 ";
		else scriptout << "0 ";
	}
	scriptout << endl;
	if ((use_limits) and (include_limits)) {
		if (lower_limits_initial.size() != n_vary_params) scriptout << "# Warning: parameter limits not defined\n";
		else {
			for (int i=0; i < n_vary_params; i++) {
				if ((lower_limits_initial[i]==lower_limits[i]) and (upper_limits_initial[i]==upper_limits[i])) {
					if ((((lower_limits[i] != 0.0) and (abs(lower_limits[i]) < 1e-3)) or (abs(lower_limits[i])) > 1e3) or (((upper_limits[i] != 0.0) and (abs(upper_limits[i]) < 1e-3)) or (abs(upper_limits[i])) > 1e3)) {
						output_field_in_sci_notation(&lower_limits[i],scriptout,true);
						output_field_in_sci_notation(&upper_limits[i],scriptout,false);
						scriptout << endl;
					} else {
						scriptout << lower_limits[i] << " " << upper_limits[i] << endl;
					}
				} else {
					if ((((lower_limits[i] != 0.0) and (abs(lower_limits[i]) < 1e-3)) or (abs(lower_limits[i])) > 1e3) or (((upper_limits[i] != 0.0) and (abs(upper_limits[i]) < 1e-3)) or (abs(upper_limits[i])) > 1e3)) {
						output_field_in_sci_notation(&lower_limits[i],scriptout,true);
						output_field_in_sci_notation(&upper_limits[i],scriptout,true);
						output_field_in_sci_notation(&lower_limits_initial[i],scriptout,true);
						output_field_in_sci_notation(&upper_limits_initial[i],scriptout,false);
						scriptout << endl;
					} else {
						scriptout << lower_limits[i] << " " << upper_limits[i] << " " << lower_limits_initial[i] << " " << upper_limits_initial[i] << endl;
					}

				}
			}
		}
	}
}

bool SB_Profile::get_special_command_arg(string &arg)
{
	return false; // overloaded for certain source objects to givce special command args
}


inline void SB_Profile::output_field_in_sci_notation(double* num, ofstream& scriptout, const bool space)
{
	scriptout << setiosflags(ios::scientific);
	scriptout << (*num);
	scriptout << resetiosflags(ios::scientific);
	if (space) scriptout << " ";
}

/********************************* Specific SB_Profile models (derived classes) *********************************/

Gaussian::Gaussian(const double &sbtot_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	model_name = "gaussian";
	sbtype = GAUSSIAN;
	set_nparams(6);
	sbtot = sbtot_in; sig_x = sig_x_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
	assign_param_pointers();
	assign_paramnames();
}

Gaussian::Gaussian(const Gaussian* sb_in)
{
	sbtot = sb_in->sbtot;
	sig_x = sb_in->sig_x;
	max_sb = sb_in->max_sb;
	copy_base_source_data(sb_in);
	update_meta_parameters();
}

void Gaussian::update_meta_parameters()
{
	max_sb = sbtot/(M_2PI*q*sig_x*sig_x);
	update_ellipticity_meta_parameters();
}

void Gaussian::assign_paramnames()
{
	paramnames[0] = "sbtot";     latex_paramnames[0] = "S";       latex_param_subscripts[0] = "max";
	paramnames[1] = "sigma"; latex_paramnames[1] = "\\sigma"; latex_param_subscripts[1] = "";
	set_geometric_paramnames(2);
}

void Gaussian::assign_param_pointers()
{
	param[0] = &sbtot;
	param[1] = &sig_x;
	set_geometric_param_pointers(2);
}

void Gaussian::set_auto_stepsizes()
{
	stepsizes[0] = (sbtot != 0) ? 0.1*sbtot : 0.1;
	stepsizes[1] = (sig_x != 0) ? 0.1*sig_x : 0.1; // arbitrary
	set_auto_eparam_stepsizes(2,3);
	stepsizes[4] = 0.1;
	stepsizes[5] = 0.1;
}

void Gaussian::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(2);
}

double Gaussian::sb_rsq(const double rsq)
{
	return max_sb*exp(-0.5*rsq/(sig_x*sig_x));
}

double Gaussian::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 7*sig_x;
}

double Gaussian::length_scale()
{
	return sig_x;
}

Sersic::Sersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	model_name = "sersic";
	sbtype = SERSIC;
	set_nparams(7);
	n = n_in;
	Reff = Reff_in;
	s0 = s0_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
	assign_param_pointers();
	assign_paramnames();
}

Sersic::Sersic(const Sersic* sb_in)
{
	s0 = sb_in->s0;
	n = sb_in->n;
	Reff = sb_in->Reff;
	copy_base_source_data(sb_in);
	update_meta_parameters();
}

void Sersic::update_meta_parameters()
{
	double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n); // from Cardone 2003 (or Ciotti 1999)
	k = b*pow(sqrt(q)/Reff,1.0/n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_ellipticity_meta_parameters();
}

void Sersic::assign_paramnames()
{
	paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	paramnames[1] = "Reff"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "eff";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	set_geometric_paramnames(3);
}

void Sersic::assign_param_pointers()
{
	param[0] = &s0;
	param[1] = &Reff;
	param[2] = &n;
	set_geometric_param_pointers(3);
}

void Sersic::set_auto_stepsizes()
{
	stepsizes[0] = 0.1; // arbitrary
	stepsizes[1] = 0.1; // arbitrary
	stepsizes[2] = 0.1; // arbitrary
	set_auto_eparam_stepsizes(3,4);
	stepsizes[5] = 0.1;
	stepsizes[6] = 0.1;
}

void Sersic::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(3);
}

double Sersic::sb_rsq(const double rsq)
{
	return s0*exp(-k*pow(rsq,0.5/n));
}

double Sersic::window_rmax()
{
	return pow(3.0/k,n);
}

double Sersic::length_scale()
{
	return Reff;
}


Cored_Sersic::Cored_Sersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	model_name = "csersic";
	sbtype = CORED_SERSIC;
	set_nparams(8);
	n = n_in;
	Reff = Reff_in;
	s0 = s0_in;
	rc = rc_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
	assign_param_pointers();
	assign_paramnames();
}

Cored_Sersic::Cored_Sersic(const Cored_Sersic* sb_in)
{
	s0 = sb_in->s0;
	n = sb_in->n;
	Reff = sb_in->Reff;
	rc = sb_in->rc;
	copy_base_source_data(sb_in);
	update_meta_parameters();
}

void Cored_Sersic::update_meta_parameters()
{
	double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	k = b*pow(sqrt(q)/Reff,1.0/n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_ellipticity_meta_parameters();
}

void Cored_Sersic::assign_paramnames()
{
	paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	paramnames[1] = "Reff"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "eff";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	paramnames[3] = "rc"; latex_paramnames[3] = "r"; latex_param_subscripts[3] = "c";
	set_geometric_paramnames(4);
}

void Cored_Sersic::assign_param_pointers()
{
	param[0] = &s0;
	param[1] = &Reff;
	param[2] = &n;
	param[3] = &rc;
	set_geometric_param_pointers(4);
}

void Cored_Sersic::set_auto_stepsizes()
{
	stepsizes[0] = 0.1; // arbitrary
	stepsizes[1] = 0.1; // arbitrary
	stepsizes[2] = 0.1; // arbitrary
	stepsizes[3] = 0.1; // arbitrary
	set_auto_eparam_stepsizes(4,5);
	stepsizes[6] = 0.1;
	stepsizes[7] = 0.1;
}

void Cored_Sersic::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_auto_penalty_limits[3] = true; penalty_lower_limits[3] = 0; penalty_upper_limits[3] = 1e30;
	set_geometric_param_auto_ranges(4);
}

double Cored_Sersic::sb_rsq(const double rsq)
{
	return s0*exp(-k*pow(rsq+rc*rc,0.5/n));
}

double Cored_Sersic::window_rmax()
{
	return pow(3.0/k,n);
}

double Cored_Sersic::length_scale()
{
	return Reff;
}

Shapelet::Shapelet(const double &amp00, const double &sig_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int nn, const bool truncate_2sig)
{
	model_name = "shapelet";
	sbtype = SHAPELET;
	set_nparams(5);
	sig = sig_in;
	n_shapelets = nn;
	indxptr = &n_shapelets;
	amps = new double*[n_shapelets];
	for (int i=0; i < n_shapelets; i++) amps[i] = new double[n_shapelets];
	for (int i=0; i < n_shapelets; i++) {
		for (int j=0; j < n_shapelets; j++) {
			amps[i][j] = 0;
		}
		amps[i][i] = amp00/(i+1); // just to start off
	}
	//amps[0][0] = amp00; // just to start off
	truncate_at_3sigma = truncate_2sig;

/*
	// testing stuff
	if (n_shapelets == 1) {
		amps[0][0] = amp00; 
	} else if (n_shapelets == 2) {
		//amps[0][1] = amp00;
		amps[1][0] = amp00;
	}
	else if (n_shapelets == 3) {
		//amps[0][2] = amp00;
		amps[2][2] = amp00;
	}
	else if (n_shapelets == 4) {
		//amps[0][3] = amp00;
		//amps[0][0] = amp00;
		amps[3][2] = amp00;
	}
	else if (n_shapelets == 5) {
		amps[4][3] = amp00;
	}
	else if (n_shapelets == 9) {
		amps[0][0] = amp00; 
		amps[4][2] = 0.5*amp00;
		amps[3][6] = 0.3*amp00;
		amps[8][8] = 0.2*amp00;
	}
*/

	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
	assign_param_pointers();
	assign_paramnames();
}

Shapelet::Shapelet(const Shapelet* sb_in)
{
	n_shapelets = sb_in->n_shapelets;
	indxptr = &n_shapelets;
	sig = sb_in->sig;
	amps = new double*[n_shapelets];
	for (int i=0; i < n_shapelets; i++) amps[i] = new double[n_shapelets];
	for (int i=0; i < n_shapelets; i++) {
		for (int j=0; j < n_shapelets; j++) {
			amps[i][j] = sb_in->amps[i][j];
		}
	}
	truncate_at_3sigma = sb_in->truncate_at_3sigma;
	copy_base_source_data(sb_in);
	update_meta_parameters();
}

void Shapelet::update_meta_parameters()
{
	update_ellipticity_meta_parameters();
}

void Shapelet::assign_paramnames()
{
	paramnames[0] = "sigma"; latex_paramnames[0] = "\\sigma"; latex_param_subscripts[0] = "";
	set_geometric_paramnames(1);
}

void Shapelet::assign_param_pointers()
{
	param[0] = &sig;
	set_geometric_param_pointers(1);
}

void Shapelet::set_auto_stepsizes()
{
	stepsizes[0] = (sig != 0) ? 0.1*sig : 0.1; // arbitrary
	set_auto_eparam_stepsizes(1,2);
	stepsizes[3] = 0.1;
	stepsizes[4] = 0.1;
}

void Shapelet::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_geometric_param_auto_ranges(1);
}

double Shapelet::surface_brightness(double x, double y)
{
	x -= x_center;
	y -= y_center;
	if ((truncate_at_3sigma) and (sqrt(x*x+y*y) > 3*sig)) return 0;
	if (theta != 0) rotate(x,y);

	double gaussfactor, sb, xarg, yarg, fac, lastfac, sqrtq;
	gaussfactor = (0.5641895835477563/(sig))*exp(-(q*x*x+y*y/q)/(2*sig*sig));
	double *hermvals_x = new double[n_shapelets];
	double *hermvals_y = new double[n_shapelets];
	hermvals_x[0] = 1.0;
	hermvals_y[0] = 1.0;
	sqrtq = sqrt(q);
	xarg = x*sqrtq/sig;
	yarg = y/(sqrtq*sig);
	if (n_shapelets > 1) {
		hermvals_x[1] = 2*xarg/SQRT2;
		hermvals_y[1] = 2*yarg/SQRT2;
	}
	lastfac = 1.0/SQRT2;
	int i,j;
	for (i=2; i < n_shapelets; i++) {
		fac = 1.0/sqrt(2*i);
		hermvals_x[i] = 2*(xarg*hermvals_x[i-1] - (i-1)*hermvals_x[i-2]*lastfac) * fac;
		hermvals_y[i] = 2*(yarg*hermvals_y[i-1] - (i-1)*hermvals_y[i-2]*lastfac) * fac;
		lastfac = fac;
	}
	sb = 0;
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			sb += amps[i][j]*hermvals_x[i]*hermvals_y[j];
		}
	}
	sb *= gaussfactor;

	delete[] hermvals_x;
	delete[] hermvals_y;
	return sb;
}

void Shapelet::calculate_Lmatrix_elements(double x, double y, double* Lmatrix_elements, const double weight)
{
	x -= x_center;
	y -= y_center;
	if ((truncate_at_3sigma) and (sqrt(x*x+y*y) > 3*sig)) return;
	if (theta != 0) rotate(x,y);

	double gaussfactor, xarg, yarg, fac, lastfac, sqrtq;
	gaussfactor = (0.5641895835477563/(sig))*exp(-(q*x*x+y*y/q)/(2*sig*sig));
	double *hermvals_x = new double[n_shapelets];
	double *hermvals_y = new double[n_shapelets];
	hermvals_x[0] = 1.0;
	hermvals_y[0] = 1.0;
	sqrtq = sqrt(q);
	xarg = x*sqrtq/sig;
	yarg = y/(sqrtq*sig);
	if (n_shapelets > 1) {
		hermvals_x[1] = 2*xarg/SQRT2;
		hermvals_y[1] = 2*yarg/SQRT2;
	}
	lastfac = 1.0/SQRT2;
	int i,j;
	for (i=2; i < n_shapelets; i++) {
		fac = 1.0/sqrt(2*i);
		hermvals_x[i] = 2*(xarg*hermvals_x[i-1] - (i-1)*hermvals_x[i-2]*lastfac) * fac;
		hermvals_y[i] = 2*(yarg*hermvals_y[i-1] - (i-1)*hermvals_y[i-2]*lastfac) * fac;
		lastfac = fac;
	}
	int n=0;
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			Lmatrix_elements[n++] += weight*gaussfactor*hermvals_x[i]*hermvals_y[j];
		}
	}

	delete[] hermvals_x;
	delete[] hermvals_y;
}

void Shapelet::calculate_gradient_Rmatrix_elements(double** Rmatrix_elements, double &logdet)
{
	if (sig==0) die("sigma cannot be zero!!");
	if ((q > 1) or (q <= 0)) die("q taking an absurd value");
	int i,j,n=0;
	logdet = 0;
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			Rmatrix_elements[n][n] = ((2*i+1)/q + (2*j+1)*q)/(2*sig*sig);
			if (Rmatrix_elements[n][n] < 0) die("negative element!!!");
			logdet += log(Rmatrix_elements[n][n]);
			n++;
		}
	}
}



/*
double Shapelet::calculate_Lmatrix_element(double x, double y, const int amp_index)
{
	x -= x_center;
	y -= y_center;
	//if ((x*x+y*y) > 2*sig) return 0;
	if (theta != 0) rotate(x,y);

	double gaussfactor, fac, lastfac, xarg, yarg, sqrtq;
	double hermval_x, hermval_x_prev, hermval_x_prev2;
	double hermval_y, hermval_y_prev, hermval_y_prev2;
	gaussfactor = (0.5641895835477563/(sig))*exp(-(q*x*x+y*y/q)/(2*sig*sig));

	int i,m,n;
	m = amp_index / n_shapelets;
	n = amp_index % n_shapelets;

	if (m==0) hermval_x = 1.0;
	if (n==0) hermval_y = 1.0;
	sqrtq = sqrt(q);
	xarg = x*sqrtq/sig;
	yarg = y/(sqrtq*sig);
	if (m > 0) {
		hermval_x = 2*xarg/SQRT2;
		hermval_x_prev = 1.0;
	}
	if (n > 0) {
		hermval_y = 2*yarg/SQRT2;
		hermval_y_prev = 1.0;
	}
	lastfac = 1.0/SQRT2;
	// since we'll be getting all the Lmatrix elements, maybe save time by storing all the Hermite polynomials
	// in a matrix? Would have to have a separate matrix for each thread, and each thread would have to iterate
	// over all the source amplitudes before moving to the next image pixel (and starting a new matrix).
	// Look into this later!
	for (i=2; i <= dmax(m,n); i++) {
		fac = 1.0/sqrt(2*i);
		if (i <= m) {
			hermval_x_prev2 = hermval_x_prev;
			hermval_x_prev = hermval_x;
			hermval_x = 2*(xarg*hermval_x_prev - (i-1)*hermval_x_prev2*lastfac) * fac;
		}
		if (i <= n) {
			hermval_y_prev2 = hermval_y_prev;
			hermval_y_prev = hermval_y;
			hermval_y = 2*(yarg*hermval_y_prev - (i-1)*hermval_y_prev2*lastfac) * fac;
		}
		lastfac = fac;
	}
	return (gaussfactor * hermval_x * hermval_y);
}
*/

void Shapelet::update_amplitudes(double *ampvec)
{
	int i,j,k=0;
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			amps[i][j] = ampvec[k++];
		}
	}
}

void Shapelet::get_amplitudes(double *ampvec)
{
	int i,j,k=0;
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			ampvec[k++] = amps[i][j];
		}
	}
}

double Shapelet::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	if (truncate_at_3sigma) return 2.3*sig;
	else return 2*sig*sqrt(n_shapelets);
}

double Shapelet::length_scale()
{
	if (truncate_at_3sigma) return sig;
	return sig*sqrt(n_shapelets);
}

bool Shapelet::get_special_command_arg(string &arg)
{
	stringstream nstr;
	string nstring;
	nstr << n_shapelets;
	nstr >> nstring;
	arg = "n=" + nstring;
	if (truncate_at_3sigma) {
		arg += " -truncate";
	}
	return true;
}


SB_Multipole::SB_Multipole(const double &A_m_in, const double r0_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool sine)
{
	model_name = "sbmpole";
	sbtype = SB_MULTIPOLE;
	//stringstream mstr;
	//string mstring;
	//mstr << m_in;
	//mstr >> mstring;
	//special_parameter_command = "m=" + mstring;
	sine_term = sine;
	set_nparams(5);

	r0 = r0_in;
	m = m_in;
	A_n = A_m_in;
	set_angle(theta_degrees);
	x_center = xc_in;
	y_center = yc_in;

	assign_param_pointers();
	assign_paramnames();
}

SB_Multipole::SB_Multipole(const SB_Multipole* sb_in)
{
	r0 = sb_in->r0;
	m = sb_in->m;
	A_n = sb_in->A_n;
	sine_term = sb_in->sine_term;
	copy_base_source_data(sb_in);
}

void SB_Multipole::assign_paramnames()
{
	string mstring;
	stringstream mstr;
	mstr << m;
	mstr >> mstring;
	if (sine_term) {
		paramnames[0] = "B_" + mstring;  latex_paramnames[0] = "B"; latex_param_subscripts[0] = mstring;
	} else {
		paramnames[0] =  "A_" + mstring; latex_paramnames[0] = "A"; latex_param_subscripts[0] = mstring;
	}
	paramnames[1] = "beta"; latex_paramnames[1] = "\\beta"; latex_param_subscripts[1] = "";
	paramnames[2] = "theta"; latex_paramnames[2] = "\\theta"; latex_param_subscripts[2] = "";
	paramnames[3] = "xc";    latex_paramnames[3] = "x";       latex_param_subscripts[3] = "c";
	paramnames[4] = "yc";    latex_paramnames[4] = "y";       latex_param_subscripts[4] = "c";
}

void SB_Multipole::assign_param_pointers()
{
	param[0] = &A_n; // here, A_n is actually the shear magnitude
	param[1] = &r0;
	param[2] = &theta; angle_paramnum = 2;
	param[3] = &x_center;
	param[4] = &y_center;
}

void SB_Multipole::set_auto_stepsizes()
{
	stepsizes[0] = 0.05;
	stepsizes[1] = 0.1;
	stepsizes[2] = 20;
	stepsizes[3] = 0.1; // very arbitrary, but a multipole term is usually center_anchored anyway
	stepsizes[4] = 0.1;
}

void SB_Multipole::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = false;
	set_auto_penalty_limits[4] = false;
}

double SB_Multipole::surface_brightness(double x, double y)
{
	x -= x_center;
	y -= y_center;
	double phi = atan(abs(y/x));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	double theta_eff = (sine_term) ? theta+M_HALFPI/m : theta;
	return A_n*exp(-sqrt(x*x+y*y)/r0) * cos(m*(phi-theta_eff));
}

double SB_Multipole::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 7*r0;
}

double SB_Multipole::length_scale()
{
	return r0;
}

TopHat::TopHat(const double &sb_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	model_name = "tophat";
	sbtype = TOPHAT;
	set_nparams(6);
	sb = sb_in; rad = rad_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_param_pointers();
	assign_paramnames();
}

TopHat::TopHat(const TopHat* sb_in)
{
	sb = sb_in->sb;
	rad = sb_in->rad;
	copy_base_source_data(sb_in);
}

void TopHat::assign_paramnames()
{
	paramnames[0] = "sb";     latex_paramnames[0] = "S";       latex_param_subscripts[0] = "top";
	paramnames[1] = "rad"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "top";
	set_geometric_paramnames(2);
}

void TopHat::assign_param_pointers()
{
	param[0] = &sb;
	param[1] = &rad;
	set_geometric_param_pointers(2);
}

void TopHat::set_auto_stepsizes()
{
	stepsizes[0] = 0.1; // arbitrary
	stepsizes[1] = 0.1; // arbitrary
	set_auto_eparam_stepsizes(2,3);
	stepsizes[4] = 0.1;
	stepsizes[5] = 0.1;
}

void TopHat::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(2);
}

double TopHat::sb_rsq(const double rsq)
{
	return (rsq < rad*rad) ? sb : 0.0;
}

double TopHat::window_rmax()
{
	return 2*rad;
}

double TopHat::length_scale()
{
	return rad;
}


