#include "profile.h"
#include "qlens.h"
#include "mathexpr.h"
#include "romberg.h"
#include "errors.h"
#include "cosmo.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

IntegrationMethod LensProfile::integral_method;
bool LensProfile::orient_major_axis_north;
bool LensProfile::use_ellipticity_components;
int LensProfile::default_ellipticity_mode;

LensProfile::LensProfile(const char *splinefile, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int& nn, const double& acc, const double &qx_in, const double &f_in)
{
	lenstype = KSPLINE;
	model_name = "kspline";
	setup_base_lens(6,true); // number of parameters = 6, is_elliptical_lens = true
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	qx_parameter = qx_in;
	f_parameter = f_in;
	kspline.input(splinefile);
	zfac = 1.0;

	set_integration_pointers();
}

void LensProfile::setup_base_lens(const int np, const bool is_elliptical_lens)
{
	set_nparams_and_anchordata(np);
	center_anchored = false;
	anchor_special_parameter = false;
	if (is_elliptical_lens) ellipticity_mode = default_ellipticity_mode;

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	assign_param_pointers();
	assign_paramnames();
}

LensProfile::LensProfile(const LensProfile* lens_in)
{
	qx_parameter = lens_in->qx_parameter;
	f_parameter = lens_in->f_parameter;
	kspline.input(lens_in->kspline);
	zfac = lens_in->zfac;

	copy_base_lensdata(lens_in);
	set_integration_pointers();
}

void LensProfile::copy_base_lensdata(const LensProfile* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	center_anchor_lens = lens_in->center_anchor_lens;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	set_default_base_settings(lens_in->numberOfPoints,lens_in->romberg_accuracy);

	q = lens_in->q;
	epsilon = lens_in->epsilon;
	epsilon2 = lens_in->epsilon2;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
}

void LensProfile::set_nparams_and_anchordata(const int &n_params_in)
{
	n_params = n_params_in;
	n_vary_params = 0;
	vary_params.input(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.input(n_params);
	set_auto_penalty_limits.input(n_params);
	penalty_lower_limits.input(n_params);
	penalty_upper_limits.input(n_params);

	anchor_parameter = new bool[n_params];
	parameter_anchor_lens = new LensProfile*[n_params];
	parameter_anchor_paramnum = new int[n_params];
	parameter_anchor_ratio = new double[n_params];
	param = new double*[n_params];
	for (int i=0; i < n_params; i++) {
		vary_params[i] = false;
		anchor_parameter[i] = false;
		parameter_anchor_lens[i] = NULL;
		parameter_anchor_paramnum[i] = -1;
		parameter_anchor_ratio[i] = 1.0;
	}
}

void LensProfile::anchor_center_to_lens(LensProfile** center_anchor_list, const int &center_anchor_lens_number)
{
	if (!center_anchored) center_anchored = true;
	center_anchor_lens = center_anchor_list[center_anchor_lens_number];
	x_center = center_anchor_lens->x_center;
	y_center = center_anchor_lens->y_center;
}

void LensProfile::delete_center_anchor()
{
	if (center_anchored) {
		center_anchored = false;
		center_anchor_lens = NULL;
	}
}

void LensProfile::delete_special_parameter_anchor()
{
	if (anchor_special_parameter) anchor_special_parameter = false;
}

void LensProfile::vary_parameters(const boolvector& vary_params_in)
{
	if (vary_params_in.size() != n_params) {
		if ((vary_params_in.size() == n_params-2) and (center_anchored)) {
			vary_params[n_params-2] = false;
			vary_params[n_params-1] = false;
		}
		else die("number of parameters to vary does not match total number of parameters");
	}
	n_vary_params=0;
	int i;
	for (i=0; i < vary_params_in.size(); i++) {
		vary_params[i] = vary_params_in[i];
		if (vary_params_in[i]) {
			n_vary_params++;
		}
	}
}

void LensProfile::set_limits(const dvector& lower, const dvector& upper)
{
	include_limits = true;
	if (lower.size() != n_vary_params) die("number of parameters with lower limits does not match number of variable parameters");
	if (upper.size() != n_vary_params) die("number of parameters with upper limits does not match number of variable parameters");
	lower_limits = lower;
	upper_limits = upper;
	lower_limits_initial = lower;
	upper_limits_initial = upper;
}

void LensProfile::set_limits(const dvector& lower, const dvector& upper, const dvector& lower_init, const dvector& upper_init)
{
	include_limits = true;
	if (lower.size() != n_vary_params) die("number of parameters with lower limits does not match number of variable parameters");
	if (upper.size() != n_vary_params) die("number of parameters with upper limits does not match number of variable parameters");
	lower_limits = lower;
	upper_limits = upper;
	lower_limits_initial = lower_init;
	upper_limits_initial = upper_init;
}

void LensProfile::get_parameters(double* params)
{
	for (int i=0; i < n_params; i++) {
		if (i==angle_paramnum) params[i] = radians_to_degrees(*(param[i]));
		else params[i] = *(param[i]);
	}
}

void LensProfile::update_parameters(const double* params)
{
	for (int i=0; i < n_params; i++) {
		if (i==angle_paramnum) *(param[i]) = degrees_to_radians(params[i]);
		else *(param[i]) = params[i];
	}
	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

bool LensProfile::update_specific_parameter(const string name_in, const double& value)
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

void LensProfile::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		for (int i=0; i < n_params; i++) {
			if (vary_params[i]==true) {
				if (i==angle_paramnum) {
					// the costheta, sintheta meta-parameters will be set in update_ellipticity_meta_parameters, which is called from update_meta_parameters for elliptical models
					*(param[i]) = degrees_to_radians(fitparams[index++]);
					update_angle_meta_params();
				}
				else *(param[i]) = fitparams[index++];
			}
		}
		update_meta_parameters();
		set_integration_pointers();
		set_model_specific_integration_pointers();
	}
}

void LensProfile::update_anchored_parameters()
{
	bool at_least_one_param_anchored = false;
	for (int i=0; i < n_params; i++) {
		if (anchor_parameter[i]) {
			(*param[i]) = parameter_anchor_ratio[i]*(*(parameter_anchor_lens[i]->param[parameter_anchor_paramnum[i]]));
			if (at_least_one_param_anchored==false) at_least_one_param_anchored = true;
		}
	}
	if (at_least_one_param_anchored) {
		update_meta_parameters();
	}
}

void LensProfile::update_anchor_center()
{
	if (center_anchored) {
		x_center = center_anchor_lens->x_center;
		y_center = center_anchor_lens->y_center;
	}
}

void LensProfile::get_fit_parameters(dvector& fitparams, int &index)
{
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]==true) {
			if (i==angle_paramnum) fitparams[index++] = radians_to_degrees(*(param[i]));
			else fitparams[index++] = *(param[i]);
		}
	}
}

void LensProfile::set_auto_stepsizes()
{
	stepsizes[0] = 0.1;
	stepsizes[1] = 0.1;
	set_auto_eparam_stepsizes(2,3);
	stepsizes[4] = 1.0; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein, but need zfactor
	stepsizes[5] = 1.0;
}

void LensProfile::set_auto_eparam_stepsizes(int eparam1_i, int eparam2_i)
{
	if (use_ellipticity_components) {
		stepsizes[eparam1_i] = 0.1; // e1
		stepsizes[eparam2_i] = 0.1; // e2
	} else {
		stepsizes[eparam1_i] = 0.1; // q or e
		stepsizes[eparam2_i] = 20;  // angle stepsize
	}
}

void LensProfile::get_auto_stepsizes(dvector& stepsizes_in, int &index)
{
	set_auto_stepsizes();
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]) stepsizes_in[index++] = stepsizes[i];
	}
}

void LensProfile::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false;
	set_auto_penalty_limits[1] = false;
	set_geometric_param_auto_ranges(2);
}

void LensProfile::set_geometric_param_auto_ranges(int param_i)
{
	if (use_ellipticity_components) {
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = -1; penalty_upper_limits[param_i] = 1; param_i++;
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = -1; penalty_upper_limits[param_i] = 1; param_i++;
	} else {
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 0; penalty_upper_limits[param_i] = 1; param_i++;
		set_auto_penalty_limits[param_i++] = false;
	}
	set_auto_penalty_limits[param_i++] = false;
	set_auto_penalty_limits[param_i++] = false;
}

void LensProfile::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
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

void LensProfile::get_fit_parameter_names(vector<string>& paramnames_vary, vector<string> *latex_paramnames_vary, vector<string> *latex_subscripts_vary)
{
	int i;
	for (i=0; i < n_params; i++) {
		if (vary_params[i]) {
			paramnames_vary.push_back(paramnames[i]);
			if (latex_paramnames_vary != NULL) latex_paramnames_vary->push_back(latex_paramnames[i]);
			if (latex_subscripts_vary != NULL) latex_subscripts_vary->push_back(latex_param_subscripts[i]);
		}
	}
}

bool LensProfile::get_limits(dvector& lower, dvector& upper, dvector& lower0, dvector& upper0, int &index)
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

void LensProfile::copy_parameter_anchors(const LensProfile* lens_in)
{
	// n_params *must* already be set before running this
	anchor_parameter = new bool[n_params];
	parameter_anchor_lens = new LensProfile*[n_params];
	parameter_anchor_paramnum = new int[n_params];
	param = new double*[n_params];
	parameter_anchor_ratio = new double[n_params];
	for (int i=0; i < n_params; i++) {
		anchor_parameter[i] = lens_in->anchor_parameter[i];
		parameter_anchor_lens[i] = lens_in->parameter_anchor_lens[i];
		parameter_anchor_paramnum[i] = lens_in->parameter_anchor_paramnum[i];
		parameter_anchor_ratio[i] = lens_in->parameter_anchor_ratio[i];
	}
	if (anchor_special_parameter) copy_special_parameter_anchor(lens_in);
}

void LensProfile::copy_special_parameter_anchor(const LensProfile *lens_in)
{
	special_anchor_lens = lens_in->special_anchor_lens;
}


void LensProfile::assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_anchor_ratio, LensProfile* param_anchor_lens)
{
	if (paramnum >= n_params) die("Parameter does not exist for this lens");
	if (anchor_paramnum >= param_anchor_lens->n_params) die("Parameter does not exist for lens you are anchoring to");
	anchor_parameter[paramnum] = true;
	parameter_anchor_lens[paramnum] = param_anchor_lens;
	parameter_anchor_paramnum[paramnum] = anchor_paramnum;
	if (!use_anchor_ratio) {
		parameter_anchor_ratio[paramnum] = 1.0;
		(*param[paramnum]) = *(param_anchor_lens->param[anchor_paramnum]);
	}
	else parameter_anchor_ratio[paramnum] = (*param[paramnum]) / (*(param_anchor_lens->param[anchor_paramnum]));

}

void LensProfile::unanchor_parameter(LensProfile* param_anchor_lens)
{
	// if any parameters are anchored to the lens in question, unanchor them (use this when you are deleting a lens, in case others are anchored to it)
	for (int i=0; i < n_params; i++) {
		if ((anchor_parameter[i]) and (parameter_anchor_lens[i] == param_anchor_lens)) {
			parameter_anchor_lens[i] = NULL;
			anchor_parameter[i] = false;
			parameter_anchor_paramnum[i] = -1;
			parameter_anchor_ratio[i] = 1.0;
		}
	}
}

void LensProfile::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "qx"; latex_paramnames[0] = "q"; latex_param_subscripts[0] = "x";
	paramnames[1] = "f"; latex_paramnames[1] = "f"; latex_param_subscripts[1] = "";
	set_geometric_paramnames(2);
}

void LensProfile::set_geometric_paramnames(int qi)
{
	if (use_ellipticity_components) {
		paramnames[qi] = "e1"; latex_paramnames[qi] = "e"; latex_param_subscripts[qi] = "1"; qi++;
		paramnames[qi] = "e2"; latex_paramnames[qi] = "e"; latex_param_subscripts[qi] = "2"; qi++;
	} else {
		if ((ellipticity_mode==2) or (ellipticity_mode==3)) {
			paramnames[qi] = "epsilon";  latex_paramnames[qi] = "\\epsilon"; latex_param_subscripts[qi] = ""; qi++;
		} else {
			paramnames[qi] = "q"; latex_paramnames[qi] = "q"; latex_param_subscripts[qi] = ""; qi++;
		}
		paramnames[qi] = "theta"; latex_paramnames[qi] = "\\theta"; latex_param_subscripts[qi] = ""; qi++;
	}
	if (!center_anchored) {
		paramnames[qi] = "xc"; latex_paramnames[qi] = "x"; latex_param_subscripts[qi] = "c"; qi++;
		paramnames[qi] = "yc"; latex_paramnames[qi] = "y"; latex_param_subscripts[qi] = "c"; qi++;
	}
}

void LensProfile::assign_param_pointers()
{
	param[0] = &qx_parameter;
	param[1] = &f_parameter;
	set_geometric_param_pointers(2);
}

void LensProfile::set_geometric_param_pointers(int qi)
{
	// Sets parameter pointers for ellipticity (or axis ratio) and angle
	if (use_ellipticity_components) {
		param[qi++] = &epsilon;
		param[qi++] = &epsilon2;
		angle_paramnum = -1; // there is no angle parameter if ellipticity components are being used
	} else {
		if ((ellipticity_mode==2) or (ellipticity_mode==3))
			param[qi++] = &epsilon;
		else
			param[qi++] = &q;
		param[qi] = &theta;
		angle_paramnum = qi++;
	}
	if (!center_anchored) {
		param[qi++] = &x_center;
		param[qi++] = &y_center;
	}
}

void LensProfile::set_geometric_parameters(const double &q1_in, const double &q2_in, const double &xc_in, const double &yc_in)
{
	if (use_ellipticity_components) {
		epsilon = q1_in;
		epsilon2 = q2_in;
	} else {
		set_ellipticity_parameter(q1_in);
		theta = degrees_to_radians(q2_in);
	}
	x_center = xc_in;
	y_center = yc_in;
	update_ellipticity_meta_parameters();
}

void LensProfile::print_parameters()
{
	if (ellipticity_mode==3) cout << "pseudo-";
	cout << model_name << ": ";
	for (int i=0; i < n_params-2; i++) {
		cout << paramnames[i] << "=";
		if (i==angle_paramnum) cout << radians_to_degrees(*(param[i])) << " degrees";
		else cout << *(param[i]);
		cout << ", ";
	}
	cout << "center=(" << x_center << "," << y_center << ")";
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	if ((ellipticity_mode != default_ellipticity_mode) and (ellipticity_mode != 3)) {
		if ((lenstype != SHEAR) and (lenstype != PTMASS) and (lenstype != MULTIPOLE) and (lenstype != SHEET))   // these models are not elliptical so emode is irrelevant
		cout << " (ellipticity mode = " << ellipticity_mode << ")"; // emode=3 is indicated by "pseudo-" name, not here
	}
	cout << endl;
}

void LensProfile::print_vary_parameters()
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
	bool parameter_anchored = false;
	for (int i=0; i < n_params; i++) {
		if (anchor_parameter[i]) parameter_anchored = true;
	}
	if (parameter_anchored) {
		cout << "   anchored parameters: ";
		int j=0;
		for (int i=0; i < n_params; i++) {
			if (anchor_parameter[i]) {
				if (j > 0) cout << ", ";
				cout << paramnames[i] << " --> (lens " << parameter_anchor_lens[i]->lens_number << ": " << parameter_anchor_lens[i]->paramnames[parameter_anchor_paramnum[i]];
				if (parameter_anchor_ratio[i] != 1.0) cout << "*" << parameter_anchor_ratio[i];
				cout << ")";
				j++;
			}
		}
		cout << endl;
	}
}

bool LensProfile::output_cosmology_info(const double zlens, const double zsrc, Cosmology* cosmo, const int lens_number)
{
	return false; // no cosmology-dependent physical parameters to calculate for this model
}

void LensProfile::set_ellipticity_parameter(const double &q_in)
{
	if (ellipticity_mode==0) {
		q = q_in; // axis ratio q = b/a
	} else if (ellipticity_mode==1) {
		q = q_in; // axis ratio q = b/a
	} else if (use_ellipticity_components) {
		q = q_in; // axis ratio q = b/a
	} else if ((ellipticity_mode==2) or (ellipticity_mode==3)) {
		epsilon = q_in; // axis ratio q = b/a
	}
	if (q < 0) q = -q; // don't allow negative axis ratios
	if (q > 1) q = 1.0; // don't allow q>1
	if (q==0) q = 0.001; // just to avoid catastrophe
}

void LensProfile::set_angle_from_components(const double &comp1, const double &comp2)
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

void LensProfile::set_default_base_settings(const int &nn, const double &acc)
{
	include_limits = false;
	rmin_einstein_radius = 1e-6; rmax_einstein_radius = 1e4;
	SetGaussLegendre(nn);
	romberg_accuracy = acc;
}

void LensProfile::update_meta_parameters_and_pointers()
{
	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

void LensProfile::set_model_specific_integration_pointers() {} // gets overloaded by some models

void LensProfile::set_integration_pointers() // Note: make sure the axis ratio q has been defined before calling this
{
	potptr = &LensProfile::potential_numerical;
	kapavgptr_rsq_spherical = &LensProfile::kappa_avg_spherical_integral;
	potptr_rsq_spherical = &LensProfile::potential_spherical_integral;
	if (q==1.0) {
		potptr = &LensProfile::potential_spherical_default;
		defptr = &LensProfile::deflection_spherical_default;
		hessptr = &LensProfile::hessian_spherical_default;
	} else {
		defptr = &LensProfile::deflection_numerical;
		hessptr = &LensProfile::hessian_numerical;
	}
}

double LensProfile::kappa_rsq(const double rsq) // this function should be redefined in all derived classes
{
	double r = sqrt(rsq);
	if (r < qx_parameter*kspline.xmin()) return (f_parameter*kspline.extend_inner_logslope(r/qx_parameter));
	if (r > qx_parameter*kspline.xmax()) return (f_parameter*kspline.extend_outer_logslope(r/qx_parameter));
	return (f_parameter*kspline.splint(r/qx_parameter));
}

void LensProfile::deflection_from_elliptical_potential(const double x, const double y, lensvector& def)
{
	// Formulas derived in Dumet-Montoya et al. (2012)
	double kapavg = (1-epsilon)*x*x + (1+epsilon)*y*y; // just r_ell^2 for the moment
	kapavg = (this->*kapavgptr_rsq_spherical)(kapavg);
	def[0] = kapavg*(1-epsilon)*x;
	def[1] = kapavg*(1+epsilon)*y;
}

void LensProfile::hessian_from_elliptical_potential(const double x, const double y, lensmatrix& hess)
{
	// Formulas derived in Dumet-Montoya et al. (2012)
	double cos2phi, sin2phi, exsq, eysq, rsq, gamma1, gamma2, kap_r, shearmag, kap;
	exsq = (1-epsilon)*x*x; // elliptical x^2
	eysq = (1+epsilon)*y*y; // elliptical y^2
	rsq = exsq+eysq; // elliptical r^2
	cos2phi = (exsq - eysq) / rsq;
	sin2phi = 2*q*(1+epsilon)*x*y/rsq;
	kap_r = kappa_rsq(rsq);
	shearmag = ((this->*kapavgptr_rsq_spherical)(rsq)) - kap_r; // shear from the spherical model
	kap = kap_r + epsilon*shearmag*cos2phi;
	gamma1 = -epsilon*kap_r - shearmag*cos2phi;
	gamma2 = -sqrt(1-epsilon*epsilon)*shearmag*sin2phi;
	hess[0][0] = kap + gamma1;
	hess[1][1] = kap - gamma1;
	hess[0][1] = gamma2;
	hess[1][0] = gamma2;
}

double LensProfile::kappa_from_elliptical_potential(const double x, const double y)
{
	// Formulas derived in Dumet-Montoya et al. (2012)
	double cos2phi, exsq, eysq, rsq, kap_r, shearmag;
	exsq = (1-epsilon)*x*x; // elliptical x^2
	eysq = (1+epsilon)*y*y; // elliptical y^2
	rsq = exsq+eysq; // elliptical r^2
	cos2phi = (exsq - eysq) / rsq;

	kap_r = kappa_rsq(rsq);
	shearmag = (this->*kapavgptr_rsq_spherical)(rsq) - kap_r;

	return (kap_r + epsilon*shearmag*cos2phi);
}

void LensProfile::shift_angle_90()
{
	// do this if the major axis orientation is changed (so the lens angles values are changed appropriately, even though the lens doesn't change)
	theta += M_HALFPI;
	while (theta > M_PI) theta -= M_PI;
}

void LensProfile::shift_angle_minus_90()
{
	// do this if the major axis orientation is changed (so the lens angles values are changed appropriately, even though the lens doesn't change)
	theta -= M_HALFPI;
	while (theta <= -M_PI) theta += M_PI;
}

void LensProfile::reset_angle_modulo_2pi()
{
	while (theta < -M_PI/2) theta += 2*M_PI;
	while (theta > 2*M_PI) theta -= 2*M_PI;
}

void LensProfile::set_angle(const double &theta_degrees)
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

void LensProfile::set_angle_radians(const double &theta_in)
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

void LensProfile::update_ellipticity_meta_parameters()
{
	// f_major_axis sets the major axis of the elliptical radius xi such that a = f*xi, and b = f*q*xi (and thus, xi = sqrt(x^2 + (y/q)^2)/f)
	if (use_ellipticity_components) {
		set_ellipticity_parameter(1 - sqrt(SQR(epsilon) + SQR(epsilon2)));
		// if ellipticity components are being used, we are automatically using the following major axis scaling
		set_angle_from_components(epsilon,epsilon2);
		f_major_axis = 1.0/sqrt(q); // defined such that a = xi/sqrt(q), and b = xi*sqrt(q)
	} else if (ellipticity_mode==0) {
		epsilon = 1 - q;
		f_major_axis = 1.0; // defined such that a = xi, and b = xi*q
	} else if (ellipticity_mode==1) {
		epsilon = 1 - q;
		// if ellipticity components are being used, we are automatically using the following major axis scaling
		f_major_axis = 1.0/sqrt(q); // defined such that a = xi/sqrt(q), and b = xi*sqrt(q)
	} else if ((ellipticity_mode==2) or (ellipticity_mode==3)) {
		q = sqrt((1-epsilon)/(1+epsilon));
		f_major_axis = sqrt((1+q*q)/2)/q; // defined such that a = xi/sqrt(1-e), and b = xi/sqrt(1+e), so that q = sqrt((1-e)/(1+e))
	}
	if (!use_ellipticity_components) update_angle_meta_params(); // sets the costheta, sintheta meta-parameters
}

void LensProfile::update_angle_meta_params()
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

void LensProfile::rotate(double &x, double &y)
{
	// perform a counter-clockwise rotation of the coordinate system to match the coordinate system of the rotated galaxy
	double xp = x*costheta + y*sintheta;
	y = -x*sintheta + y*costheta;
	x = xp;
}

double LensProfile::kappa(double x, double y)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	if ((ellipticity_mode==3) and (q != 1)) {
		return kappa_from_elliptical_potential(x,y);
	} else {
		return kappa_rsq((x*x + y*y/(q*q))/(f_major_axis*f_major_axis));
	}
}

double LensProfile::potential(double x, double y)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	if (ellipticity_mode==3) {
		return (this->*potptr_rsq_spherical)((1-epsilon)*x*x + (1+epsilon)*y*y); // ellipticity is put into the potential in this mode
	} else {
		return (this->*potptr)(x,y);
	}
}

void LensProfile::deflection(double x, double y, lensvector& def)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	if ((ellipticity_mode==3) and (q != 1)) {
		deflection_from_elliptical_potential(x,y,def);
	} else {
		(this->*defptr)(x,y,def);
	}
	if (sintheta != 0) def.rotate_back(costheta,sintheta);
}

void LensProfile::hessian(double x, double y, lensmatrix& hess)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	if ((ellipticity_mode==3) and (q != 1)) {
		hessian_from_elliptical_potential(x,y,hess);
	} else {
		(this->*hessptr)(x,y,hess);
	}
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

double LensProfile::kappa_avg_r(const double r)
{
	return (this->*kapavgptr_rsq_spherical)(r*r);
}

double LensProfile::einstein_radius_root(const double r)
{
	return (zfac*(this->*kapavgptr_rsq_spherical)(r*r)-1);
}

void LensProfile::get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor)
{
	if (kapavgptr_rsq_spherical==NULL) {
		re_major_axis=0;
		re_average=0;
		return;
	}
	zfac = zfactor;
	if ((einstein_radius_root(rmin_einstein_radius)*einstein_radius_root(rmax_einstein_radius)) > 0) {
		// multiple imaging does not occur with this lens
		re_major_axis = 0;
		re_average = 0;
		return;
	}
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&LensProfile::einstein_radius_root);
	re_average = BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-6);
	re_major_axis = re_average * f_major_axis;
	zfac = 1.0;
}

double LensProfile::kappa_rsq_deriv(const double rsq)
{
	static const double precision = 1e-6;
	double temp, h;
	h = precision*rsq;
	temp = rsq + h;
	h = temp - rsq; // silly NR trick
	return (kappa_rsq((rsq+h)/(qx_parameter*qx_parameter))-kappa_rsq((rsq-h)/(qx_parameter*qx_parameter)))/(2*h);
}

double LensProfile::get_inner_logslope()
{
	static const double h = 1e-6;
	double dlogh;
	dlogh = log(h);
	return ((log(kappa_rsq(SQR(exp(2*dlogh)))) - log(kappa_rsq(h*h))) / dlogh);
}

void LensProfile::plot_kappa_profile(double rmin, double rmax, int steps, const char *kname, const char *kdname)
{
	double r, rstep;
	rstep = pow(rmax/rmin, 1.0/steps);
	int i;
	ofstream kout(kname);
	ofstream kdout;
	if (kdname != NULL) kdout.open(kdname);
	kout << setiosflags(ios::scientific);
	if (kdname != NULL) kdout << setiosflags(ios::scientific);
	double kavg, rsq;
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		rsq = r*r;
		if (kapavgptr_rsq_spherical==NULL) kavg=0; // just in case there is no radial deflection function defined
		else kavg = (this->*kapavgptr_rsq_spherical)(rsq);
		kout << r << " " << kappa_rsq(rsq) << " " << kavg << " " << kavg*r << " " << M_PI*kavg*rsq << endl;
		if (kdname != NULL) kdout << r << " " << fabs(kappa_rsq_deriv(rsq)) << endl;
	}
}

void LensProfile::deflection_spherical_default(double x, double y, lensvector& def)
{
	double kapavg = x*x+y*y; // r^2 right now
	kapavg = (this->*kapavgptr_rsq_spherical)(kapavg);

	def[0] = kapavg*x;
	def[1] = kapavg*y;
}

double LensProfile::potential_spherical_default(const double x, const double y)
{
	return (this->*potptr_rsq_spherical)(x*x+y*y); // ellipticity is put into the potential in this mode
}

double LensProfile::kapavg_spherical_generic(const double rsq)
{
	return (this->*kapavgptr_rsq_spherical)(rsq);
}

double LensProfile::kappa_avg_spherical_integral(const double rsq)
{
	double ans;
	if (integral_method == Romberg_Integration)
	{
		double (Romberg::*sptr)(const double);
		sptr = static_cast<double (Romberg::*)(const double)> (&LensProfile::mass_enclosed_spherical_integrand);
		ans = (2.0/rsq)*romberg_open(sptr, 0, sqrt(rsq), romberg_accuracy, 5);
	}
	else if (integral_method == Gaussian_Quadrature)
	{
		double (GaussianIntegral::*sptr)(double);
		sptr = static_cast<double (GaussianIntegral::*)(double)> (&LensProfile::mass_enclosed_spherical_integrand);
		ans = (2.0/rsq)*NIntegrate(sptr,0,sqrt(rsq));
	}
	else die("unknown integral method");
	return ans;
}

double LensProfile::mass_enclosed_spherical_integrand(const double u) { return u*kappa_rsq(u*u); } // actually mass enclosed / (2*pi*sigma_cr)

void LensProfile::hessian_spherical_default(const double x, const double y, lensmatrix& hess)
{
	double rsq, kappa_avg, r_dfdr;
	rsq = x*x+y*y;
	kappa_avg = (this->*kapavgptr_rsq_spherical)(rsq);
	r_dfdr = 2*(kappa_rsq(rsq) - kappa_avg)/rsq; // Here, r_dfdr = (1/r)*d/dr(kappa_avg)

	hess[0][0] = kappa_avg + x*x*r_dfdr;
	hess[1][1] = kappa_avg + y*y*r_dfdr;
	hess[0][1] = x*y*r_dfdr;
	hess[1][0] = hess[0][1];
}

double LensProfile::potential_spherical_integral(const double rsq)
{
	LensIntegral lens_integral(this,rsq,0,1.0);
	return (0.5*lens_integral.i_integral());
}

void LensProfile::deflection_numerical(const double x, const double y, lensvector& def)
{
	def[0] = q*x*j_integral(x,y,0);
	def[1] = q*y*j_integral(x,y,1);
}

void LensProfile::hessian_numerical(const double x, const double y, lensmatrix& hess)
{
	hess[0][0] = 2*q*x*x*k_integral(x,y,0) + q*j_integral(x,y,0);
	hess[1][1] = 2*q*y*y*k_integral(x,y,2) + q*j_integral(x,y,1);
	hess[0][1] = 2*q*x*y*k_integral(x,y,1);
	hess[1][0] = hess[0][1];
}

double LensProfile::potential_numerical(const double x, const double y)
{
	LensIntegral lens_integral(this,x*x,y*y,q);
	return (0.5*q*lens_integral.i_integral());
}

inline double LensProfile::j_integral(const double x, const double y, const int n)
{
	LensIntegral lens_integral(this,x*x,y*y,q,n);
	return lens_integral.j_integral();
}

inline double LensProfile::k_integral(const double x, const double y, const int n)
{
	LensIntegral lens_integral(this,x*x,y*y,q,n);
	return lens_integral.k_integral();
}

bool LensProfile::core_present() { return false; }

double LensIntegral::i_integral()
{
	double ans;
	if (profile->integral_method == Romberg_Integration)
	{
		double (Romberg::*iptr)(const double);
		iptr = static_cast<double (Romberg::*)(const double)> (&LensIntegral::i_integrand_prime);
		ans = romberg_open(iptr, 0, 1, profile->romberg_accuracy, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		double (GaussianIntegral::*iptr)(double);
		iptr = static_cast<double (GaussianIntegral::*)(double)> (&LensIntegral::i_integrand_prime);
		ans = NIntegrate(iptr,0,1);
	}
	else die("unknown integral method");
	return ans;
}

double LensIntegral::j_integral()
{
	double ans;
	if (profile->integral_method == Romberg_Integration)
	{
		double (Romberg::*jptr)(const double);
		jptr = static_cast<double (Romberg::*)(const double)> (&LensIntegral::j_integrand_prime);
		ans = romberg_open(jptr, 0, 1, profile->romberg_accuracy, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		double (GaussianIntegral::*jptr)(double);
		jptr = static_cast<double (GaussianIntegral::*)(double)> (&LensIntegral::j_integrand_prime);
		ans = NIntegrate(jptr,0,1);
	}
	else die("unknown integral method");
	return ans;
}

double LensIntegral::k_integral()
{
	double ans;
	if (profile->integral_method == Romberg_Integration)
	{
		double (Romberg::*kptr)(const double);
		kptr = static_cast<double (Romberg::*)(const double)> (&LensIntegral::k_integrand_prime);
		ans = romberg_open(kptr, 0, 1, profile->romberg_accuracy, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		double (GaussianIntegral::*kptr)(double);
		kptr = static_cast<double (GaussianIntegral::*)(double)> (&LensIntegral::k_integrand_prime);
		ans = NIntegrate(kptr,0,1);
	}
	else die("unknown integral method");
	return ans;
}

// i,j,k integrals are in form similar to Keeton (2001), but generalized to allow for different
// definitions of the elliptical radius. I have also made the subsitution
// u=w*w (easier for Gaussian quadrature; makes kappa singularity more manageable)

double LensIntegral::i_integrand_prime(const double w)
{
	u = w*w;
	qfac = 1 - (1-qsq)*u;
	xisq = u*(xsqval + ysqval/qfac)*fsqinv;
	return (2*w*(xisq/u)*(profile->kapavg_spherical_generic)(xisq) / sqrt(qfac))/fsqinv;
}

double LensIntegral::j_integrand_prime(const double w)
{
	u = w*w;
	qfac = 1 - (1-qsq)*u;
	xisq = u*(xsqval + ysqval/qfac)*fsqinv;
	return (2*w*profile->kappa_rsq(xisq) / pow(qfac, nval+0.5));
}

double LensIntegral::k_integrand_prime(const double w)
{
	u = w*w;
	qfac = 1 - (1-qsq)*u;
	xisq = u*(xsqval + ysqval/qfac)*fsqinv;
	return fsqinv*(2*w*u*profile->kappa_rsq_deriv(xisq) / pow(qfac, nval+0.5));
}


