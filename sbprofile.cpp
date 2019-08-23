#include "sbprofile.h"
#include "qlens.h"
#include "mathexpr.h"
#include "errors.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

bool SB_Profile::orient_major_axis_north;
bool SB_Profile::use_ellipticity_components;

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
	is_lensed = sb_in->is_lensed;
	set_nparams(sb_in->n_params);
	set_geometric_parameters_radians(sb_in->q,sb_in->theta,sb_in->x_center,sb_in->y_center);

	paramnames = sb_in->paramnames;
	latex_paramnames = sb_in->latex_paramnames;
	latex_param_subscripts = sb_in->latex_param_subscripts;
	n_vary_params = sb_in->n_vary_params;
	vary_params.input(sb_in->vary_params);

	include_limits = sb_in->include_limits;
	if (include_limits) {
		lower_limits.input(sb_in->lower_limits);
		upper_limits.input(sb_in->upper_limits);
		lower_limits_initial.input(sb_in->lower_limits_initial);
		upper_limits_initial.input(sb_in->upper_limits_initial);
	}
	n_fourier_modes = sb_in->n_fourier_modes;
	if (n_fourier_modes > 0) {
		fourier_mode_mvals.input(sb_in->fourier_mode_mvals);
		fourier_mode_cosamp.input(sb_in->fourier_mode_cosamp);
		fourier_mode_sinamp.input(sb_in->fourier_mode_sinamp);
		fourier_mode_paramnum.input(sb_in->fourier_mode_paramnum);
	}
	include_boxiness_parameter = sb_in->include_boxiness_parameter;
	include_fmode_rscale = sb_in->include_fmode_rscale;
	if (include_boxiness_parameter) c0 = sb_in->c0;
	if (include_fmode_rscale) fmode_rscale = sb_in->fmode_rscale;
	assign_param_pointers();
}

void SB_Profile::set_nparams(const int &n_params_in)
{
	n_params = n_params_in;
	include_boxiness_parameter = false;
	include_fmode_rscale = false;
	is_lensed = true; // default
	n_fourier_modes = 0;
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

	delete[] param;
	param = new double*[n_params];
	assign_param_pointers();
}

void SB_Profile::add_fmode_rscale(const double rscale_in, const bool vary_rscale)
{
	if (include_fmode_rscale) return;
	include_fmode_rscale = true;
	fmode_rscale = rscale_in;
	n_params++;

	vary_params.resize(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.resize(n_params);
	set_auto_penalty_limits.resize(n_params);
	penalty_lower_limits.resize(n_params);
	penalty_upper_limits.resize(n_params);
	if (vary_rscale) n_vary_params++;

	vary_params[n_params-1] = vary_rscale;
	paramnames[n_params-1] = "rfsc";
	latex_paramnames[n_params-1] = "r";
	latex_param_subscripts[n_params-1] = "fsc";
	stepsizes[n_params-1] = 0.01; // arbitrary
	set_auto_penalty_limits[n_params-1] = false;

	delete[] param;
	param = new double*[n_params];
	assign_param_pointers();
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
}

void SB_Profile::remove_fourier_modes()
{
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
	if (use_ellipticity_components) {
		param[qi++] = &epsilon;
		param[qi++] = &epsilon2;
		angle_paramnum = -1; // there is no angle parameter if ellipticity components are being used
		ellipticity_paramnum = -1; // no single ellipticity parameter here
	} else {
		param[qi] = &q;
		ellipticity_paramnum = qi++;
		param[qi] = &theta;
		angle_paramnum = qi++;
	}
	param[qi++] = &x_center;
	param[qi++] = &y_center;
	if (include_boxiness_parameter) param[qi++] = &c0;
	if (include_fmode_rscale) param[qi++] = &fmode_rscale;
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
				//cout << "P" << i << ": " << *(param[i]) << " " << fitparams[index-1] << endl;
			}
		}
		update_meta_parameters();
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
	if (use_ellipticity_components) {
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
	if (use_ellipticity_components) {
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
	if (use_ellipticity_components) {
		paramnames[qi] = "e1"; latex_paramnames[qi] = "e"; latex_param_subscripts[qi] = "1,src"; qi++;
		paramnames[qi] = "e2"; latex_paramnames[qi] = "e"; latex_param_subscripts[qi] = "2,src"; qi++;
	} else {
		paramnames[qi] = "q"; latex_paramnames[qi] = "q"; latex_param_subscripts[qi] = "src"; qi++;
		paramnames[qi] = "theta"; latex_paramnames[qi] = "\\theta"; latex_param_subscripts[qi] = "src"; qi++;
	}
	paramnames[qi] = "xc"; latex_paramnames[qi] = "x"; latex_param_subscripts[qi] = "c,src"; qi++;
	paramnames[qi] = "yc"; latex_paramnames[qi] = "y"; latex_param_subscripts[qi] = "c,src"; qi++;
}

void SB_Profile::set_geometric_parameters(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	qx_parameter = 1.0;
	q=q_in;
	if (q < 0) q = -q; // don't allow negative axis ratios
	if (q > 1) q = 1.0; // don't allow q>1
	set_angle(theta_degrees);
	x_center = xc_in;
	y_center = yc_in;
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
	while (angle > M_HALFPI) angle -= M_PI;
	while (angle <= -M_HALFPI) angle += M_PI;
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

	double rsq = x*x + y*y/(q*q);
	double rscale_fac, cc0;
	if (include_fmode_rscale) {
		if (fmode_rscale==0) rscale_fac = 1.0;
		else rscale_fac = (erf((sqrt(rsq)*6.0/fmode_rscale-3.0))+1.0)/2;
	} else {
		rscale_fac = 1.0;
	}
	cc0 = c0*rscale_fac;
	if ((include_boxiness_parameter) and (c0 != 0.0)) {
		rsq = pow(pow(abs(x),cc0+2.0) + pow(abs(y/q),cc0+2.0),2.0/(cc0+2.0));
	//} else {
		//rsq = x*x + y*y/(q*q);
	}
	if (n_fourier_modes > 0) {
		double fourier_factor = 1.0;
		for (int i=0; i < n_fourier_modes; i++) {
			//fourier_factor += fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*(phi_q + fourier_mode_phivals[i]));
			fourier_factor += rscale_fac*(fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*phi_q) + fourier_mode_sinamp[i]*sin(fourier_mode_mvals[i]*phi_q));
		}
		rsq *= fourier_factor*fourier_factor;
	}
	return sb_rsq(rsq);
}

double SB_Profile::surface_brightness_r(const double r)
{
	return sb_rsq(r*r);
}

void SB_Profile::print_parameters()
{
	cout << model_name;
	if (!is_lensed) cout << "(unlensed)";
	cout << ": ";
	for (int i=0; i < n_params; i++) {
		cout << paramnames[i] << "=";
		if (i==angle_paramnum) cout << radians_to_degrees(*(param[i])) << " degrees";
		else cout << *(param[i]);
		if (i != n_params-1) cout << ", ";
	}
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

void SB_Profile::print_source_command(ofstream& scriptout, const bool use_limits)
{
	scriptout << setprecision(16);
	scriptout << "fit source " << model_name << " ";
	if (!is_lensed) scriptout << "-unlensed ";

	for (int i=0; i < n_params; i++) {
		if (i==angle_paramnum) scriptout << radians_to_degrees(*(param[i]));
		else {
			if (paramnames[i]=="c0") scriptout << "c0=";
			if (paramnames[i]=="rfsc") scriptout << "rfsc=";
			for (int j=0; j < n_fourier_modes; j++) {
				if (fourier_mode_paramnum[j]==i) scriptout << "f" << fourier_mode_mvals[j] << "=";
			}
			if (((*(param[i]) != 0.0) and (abs(*(param[i])) < 1e-3)) or (abs(*(param[i]))) > 1e3) output_field_in_sci_notation(param[i],scriptout,false);
			else scriptout << *(param[i]);
		}
		scriptout << " ";
	}
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

inline void SB_Profile::output_field_in_sci_notation(double* num, ofstream& scriptout, const bool space)
{
	scriptout << setiosflags(ios::scientific);
	scriptout << (*num);
	scriptout << resetiosflags(ios::scientific);
	if (space) scriptout << " ";
}

/********************************* Specific SB_Profile models (derived classes) *********************************/

Gaussian::Gaussian(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	model_name = "gaussian";
	sbtype = GAUSSIAN;
	set_nparams(6);
	max_sb = max_sb_in; sig_x = sig_x_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_param_pointers();
	assign_paramnames();
}

Gaussian::Gaussian(const Gaussian* sb_in)
{
	max_sb = sb_in->max_sb;
	sig_x = sb_in->sig_x;
	copy_base_source_data(sb_in);
}

void Gaussian::assign_paramnames()
{
	paramnames[0] = "sbmax";     latex_paramnames[0] = "S";       latex_param_subscripts[0] = "max";
	paramnames[1] = "sigma"; latex_paramnames[1] = "\\sigma"; latex_param_subscripts[1] = "";
	set_geometric_paramnames(2);
}

void Gaussian::assign_param_pointers()
{
	param[0] = &max_sb;
	param[1] = &sig_x;
	set_geometric_param_pointers(2);
}

void Gaussian::set_auto_stepsizes()
{
	stepsizes[0] = (max_sb != 0) ? 0.1*max_sb : 0.1;
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
	double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	k = b*pow(sqrt(q)/Reff,1.0/n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_angle_meta_params();
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
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
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
	ellipticity_paramnum = -1; // no ellipticity parameter here
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


