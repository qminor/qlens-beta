#include "profile.h"
#include "qlens.h"
#include "mathexpr.h"
#include "romberg.h"
#include "errors.h"
#include "cosmo.h"
#include "qlens.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

IntegrationMethod LensProfile::integral_method;
bool LensProfile::orient_major_axis_north;
bool LensProfile::use_ellipticity_components;
int LensProfile::default_ellipticity_mode;
bool LensProfile::output_integration_errors;

LensProfile::LensProfile(const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int& nn, const double& acc, const double &qx_in, const double &f_in, Lens* lens_in)
{
	setup_lens_properties();
	setup_cosmology(lens_in,zlens_in,zsrc_in);
	set_default_base_settings(nn,acc);

	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	qx_parameter = qx_in;
	f_parameter = f_in;
	kspline.input(splinefile);
	zfac = 1.0;

	set_integration_pointers();
}

void LensProfile::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = KSPLINE;
	model_name = "kspline";
	special_parameter_command = "";
	setup_base_lens(7,true); // number of parameters = 6, is_elliptical_lens = true
}

void LensProfile::setup_base_lens(const int np, const bool is_elliptical_lens, const int pmode_in, const int subclass_in)
{
	center_defined = true;
	parameter_mode = pmode_in;
	lens_subclass = subclass_in; // automatically set to -1 by default if there are no subclasses defined
	set_nparams_and_anchordata(np);
	center_anchored = false;
	anchor_special_parameter = false;
	if (is_elliptical_lens) ellipticity_mode = default_ellipticity_mode;
	else {
		f_major_axis = 1; // used for calculating approximate angle-averaged Einstein radius for non-elliptical lens models
		ellipticity_mode = -1; // indicates not an elliptical lens
	}
	analytic_3d_density = false; // this will be changed to 'true' for certain models (e.g. NFW)
	perturber = false; // default

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	assign_param_pointers();
	assign_paramnames();
}

void LensProfile::setup_cosmology(Lens* lens_in, const double zlens_in, const double zsrc_in)
{
	lens = lens_in;
	zlens = zlens_in;
	zlens_current = zlens_in;
	zsrc_ref = zsrc_in;
	sigma_cr = lens->sigma_crit_arcsec(zlens,zsrc_ref);
	kpc_to_arcsec = 206.264806/lens->angular_diameter_distance(zlens);
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
	lens = lens_in->lens;
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_defined = lens_in->center_defined;
	zlens = lens_in->zlens;
	zsrc_ref = lens_in->zsrc_ref;
	sigma_cr = lens_in->sigma_cr;
	kpc_to_arcsec = lens_in->kpc_to_arcsec;

	center_anchored = lens_in->center_anchored;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	center_anchor_lens = lens_in->center_anchor_lens;
	n_params = lens_in->n_params;
	parameter_mode = lens_in->parameter_mode;
	lens_subclass = lens_in->lens_subclass;
	special_parameter_command = lens_in->special_parameter_command;
	subclass_label = lens_in->subclass_label;
	ellipticity_mode = lens_in->ellipticity_mode;
	perturber = lens_in->perturber;
	analytic_3d_density = lens_in->analytic_3d_density;
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	stepsizes.input(lens_in->stepsizes);
	set_auto_penalty_limits.input(lens_in->set_auto_penalty_limits);
	penalty_lower_limits.input(lens_in->penalty_lower_limits);
	penalty_upper_limits.input(lens_in->penalty_upper_limits);
	set_default_base_settings(lens_in->numberOfPoints,lens_in->integral_tolerance);

	if (ellipticity_mode != -1) {
		q = lens_in->q;
		epsilon = lens_in->epsilon;
		epsilon2 = lens_in->epsilon2;
	}
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	include_limits = lens_in->include_limits;
	if (include_limits) {
		lower_limits.input(lens_in->lower_limits);
		upper_limits.input(lens_in->upper_limits);
		lower_limits_initial.input(lens_in->lower_limits_initial);
		upper_limits_initial.input(lens_in->upper_limits_initial);
	}
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
	parameter_anchor_exponent = new double[n_params];
	param = new double*[n_params];
	for (int i=0; i < n_params; i++) {
		vary_params[i] = false;
		anchor_parameter[i] = false;
		parameter_anchor_lens[i] = NULL;
		parameter_anchor_paramnum[i] = -1;
		parameter_anchor_ratio[i] = 1.0;
		parameter_anchor_exponent[i] = 1.0;
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

bool LensProfile::vary_parameters(const boolvector& vary_params_in)
{
	if (vary_params_in.size() != n_params) {
		if ((vary_params_in.size() == n_params-2) and (center_anchored)) {
			vary_params[n_params-2] = false;
			vary_params[n_params-1] = false;
		}
		else return false;
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

void LensProfile::get_parameters_pmode(const int pmode_in, double* params)
{
	// overload this function for models that have different parameter modes; allows
	// flexibility in obtaining parameters from different pmodes
	return get_parameters(params);
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

void LensProfile::update_ellipticity_parameter(const double eparam)
{
	// This function is only used by the "qtab" model at the moment
	*(param[ellipticity_paramnum]) = eparam;
	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

// You need to have a function at the model level, called here that reports a "false" status if parameter values are crazy!
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
			(*param[i]) = parameter_anchor_ratio[i]*pow(*(parameter_anchor_lens[i]->param[parameter_anchor_paramnum[i]]),parameter_anchor_exponent[i]);
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
	stepsizes[6] = 0.1;
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
		set_auto_penalty_limits[param_i] = true;
		if ((ellipticity_mode==2) or (ellipticity_mode==3)) {
			penalty_lower_limits[param_i] = 0; penalty_upper_limits[param_i] = 0.995;
		} else {
			penalty_lower_limits[param_i] = 5e-3; penalty_upper_limits[param_i] = 1;
		}
		param_i++;
		set_auto_penalty_limits[param_i] = false; penalty_lower_limits[param_i] = -1e30; penalty_upper_limits[param_i] = 1e30; param_i++;
	}
	set_auto_penalty_limits[param_i] = false; penalty_lower_limits[param_i] = -1e30; penalty_upper_limits[param_i] = 1e30; param_i++;
	set_auto_penalty_limits[param_i] = false; penalty_lower_limits[param_i] = -1e30; penalty_upper_limits[param_i] = 1e30; param_i++;
	set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 0.01; penalty_upper_limits[param_i] = zsrc_ref; param_i++;
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
			} else {
				use_penalty_limits[index] = false;
			}
			index++;
		}
	}
}

void LensProfile::get_fit_parameter_names(vector<string>& paramnames_vary, vector<string> *latex_paramnames_vary, vector<string> *latex_subscripts_vary)
{
	int i;
	//cout << "NPAR=" << n_params << endl;
	for (i=0; i < n_params; i++) {
		if (vary_params[i]) {
			//cout << "PAR " << i << "is being varied" << endl;
			paramnames_vary.push_back(paramnames[i]);
			if (latex_paramnames_vary != NULL) latex_paramnames_vary->push_back(latex_paramnames[i]);
			if (latex_subscripts_vary != NULL) latex_subscripts_vary->push_back(latex_param_subscripts[i]);
		}
		//else cout << "PAR " << i << "is NOT being varied" << endl;
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
	parameter_anchor_exponent = new double[n_params];
	for (int i=0; i < n_params; i++) {
		anchor_parameter[i] = lens_in->anchor_parameter[i];
		parameter_anchor_lens[i] = lens_in->parameter_anchor_lens[i];
		parameter_anchor_paramnum[i] = lens_in->parameter_anchor_paramnum[i];
		parameter_anchor_ratio[i] = lens_in->parameter_anchor_ratio[i];
		parameter_anchor_exponent[i] = lens_in->parameter_anchor_exponent[i];
	}
	if (anchor_special_parameter) copy_special_parameter_anchor(lens_in);
}

void LensProfile::copy_special_parameter_anchor(const LensProfile *lens_in)
{
	special_anchor_lens = lens_in->special_anchor_lens;
}


void LensProfile::assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, LensProfile* param_anchor_lens)
{
	if (paramnum >= n_params) die("Parameter does not exist for this lens");
	if (anchor_paramnum >= param_anchor_lens->n_params) die("Parameter does not exist for lens you are anchoring to");
	anchor_parameter[paramnum] = true;
	parameter_anchor_lens[paramnum] = param_anchor_lens;
	parameter_anchor_paramnum[paramnum] = anchor_paramnum;
	if ((!use_implicit_ratio) and (!use_exponent)) {
		parameter_anchor_ratio[paramnum] = 1.0;
		(*param[paramnum]) = *(param_anchor_lens->param[anchor_paramnum]);
	}
	else if (use_implicit_ratio) {
		parameter_anchor_exponent[paramnum] = 1.0;
		if ((*(param_anchor_lens->param[anchor_paramnum]))==0) {
			if (*param[paramnum]==0) parameter_anchor_ratio[paramnum] = 1.0;
			else die("cannot anchor to parameter with specified ratio if parameter is equal to zero");
		} else {
			parameter_anchor_ratio[paramnum] = (*param[paramnum]) / (*(param_anchor_lens->param[anchor_paramnum]));
		}
	}
	else if (use_exponent) {
		parameter_anchor_ratio[paramnum] = ratio;
		parameter_anchor_exponent[paramnum] = exponent;
	}
	update_anchored_parameters();
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
			parameter_anchor_exponent[i] = 1.0;
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
	paramnames[qi] = "z"; latex_paramnames[qi] = "z"; latex_param_subscripts[qi] = "l"; qi++;
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
		ellipticity_paramnum = -1; // no single ellipticity parameter here
	} else {
		if ((ellipticity_mode==2) or (ellipticity_mode==3))
			param[qi] = &epsilon;
		else
			param[qi] = &q;
		ellipticity_paramnum = qi++;
		param[qi] = &theta;
		angle_paramnum = qi++;
	}
	if (!center_anchored) {
		param[qi++] = &x_center;
		param[qi++] = &y_center;
	}
	param[qi++] = &zlens;
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
	cout << model_name << "(";
	if (lens_subclass != -1) cout << subclass_label << "=" << lens_subclass << ",";
	cout << "z=" << zlens << "): ";
	if (center_defined) {
		for (int i=0; i < n_params-3; i++) {
			cout << paramnames[i] << "=";
			if (i==angle_paramnum) cout << radians_to_degrees(*(param[i])) << " degrees";
			else cout << *(param[i]);
			cout << ", ";
		}
		cout << "xc=" << x_center << ", yc=" << y_center;
	} else {
		for (int i=0; i < n_params-1; i++) {
			cout << paramnames[i] << "=";
			if (i==angle_paramnum) cout << radians_to_degrees(*(param[i])) << " degrees";
			else cout << *(param[i]);
			if (i != n_params-2) cout << ", ";
		}
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	if ((ellipticity_mode != default_ellipticity_mode) and (ellipticity_mode != 3) and (ellipticity_mode != -1)) {
		if ((lenstype != SHEAR) and (lenstype != PTMASS) and (lenstype != MULTIPOLE) and (lenstype != SHEET) and (lenstype != TABULATED))   // these models are not elliptical so emode is irrelevant
		cout << " (ellipticity mode = " << ellipticity_mode << ")"; // emode=3 is indicated by "pseudo-" name, not here
	}
	double aux_param;
	string aux_paramname;
	get_auxiliary_parameter(aux_paramname,aux_param);
	if (aux_paramname != "") cout << " (" << aux_paramname << "=" << aux_param << ")";
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
				cout << paramnames[i] << " --> (lens " << parameter_anchor_lens[i]->lens_number << ": ";
				if ((parameter_anchor_ratio[i] != 1.0) or (parameter_anchor_exponent[i] != 1.0)) {
					cout << parameter_anchor_ratio[i] << "*" << parameter_anchor_lens[i]->paramnames[parameter_anchor_paramnum[i]];
					if (parameter_anchor_exponent[i] != 1.0) cout << "^" << parameter_anchor_exponent[i];
				}
				else cout << parameter_anchor_lens[i]->paramnames[parameter_anchor_paramnum[i]];
				cout << ")";
				j++;
			}
		}
		cout << endl;
	}
}

inline void LensProfile::output_field_in_sci_notation(double* num, ofstream& scriptout, const bool space)
{
	// I thought it would be cool to print scientific notation and omit all the zero's if it's an exact mantissa.
	// It doesn't work very well (due to binary vs. base 10 storage), so canning it for now, not a big deal
	//int exp;
	//double mantissa = frexp10((*num),&exp);
	//double mantissa;
   //exp = ((*num) == 0) ? 0 : 1 + (int)std::floor(std::log10(abs((*num))));
   //mantissa = (*num) * std::pow(10 , -exp);
	scriptout << setiosflags(ios::scientific);
	scriptout << (*num);
	scriptout << resetiosflags(ios::scientific);
	if (space) scriptout << " ";
}

void LensProfile::print_lens_command(ofstream& scriptout, const bool use_limits)
{
	scriptout << setprecision(16);
	//scriptout << setiosflags(ios::scientific);
	scriptout << "fit lens " << model_name << " ";
	if (ellipticity_mode != default_ellipticity_mode) {
		if ((lenstype != SHEAR) and (lenstype != PTMASS) and (lenstype != MULTIPOLE) and (lenstype != SHEET) and (lenstype != TABULATED))   // these models are not elliptical so emode is irrelevant
			scriptout << "emode=" << ellipticity_mode << " ";
	}
	if (parameter_mode != 0) scriptout << "pmode=" << parameter_mode << " ";
	if (special_parameter_command != "") scriptout << special_parameter_command << " ";

	if (center_defined) {
		for (int i=0; i < n_params-3; i++) {
			if ((anchor_parameter[i]) and (parameter_anchor_ratio[i]==1.0)) scriptout << "anchor=" << parameter_anchor_lens[i]->lens_number << "," << parameter_anchor_paramnum[i] << " ";
			else {
				if (i==angle_paramnum) scriptout << radians_to_degrees(*(param[i]));
				else {
					if (((*(param[i]) != 0.0) and (abs(*(param[i])) < 1e-3)) or (abs(*(param[i]))) > 1e3) output_field_in_sci_notation(param[i],scriptout,false);
					else scriptout << *(param[i]);
				}
				if (anchor_parameter[i]) scriptout << "/anchor=" << parameter_anchor_lens[i]->lens_number << "," << parameter_anchor_paramnum[i];
				scriptout << " ";
			}
		}
		if (center_anchored) scriptout << " anchor_center=" << center_anchor_lens->lens_number << endl;
		else scriptout << x_center << " " << y_center << " z=" << zlens << endl;
	} else {
		for (int i=0; i < n_params-1; i++) {
			if ((anchor_parameter[i]) and (parameter_anchor_ratio[i]==1.0)) scriptout << "anchor=" << parameter_anchor_lens[i]->lens_number << "," << parameter_anchor_paramnum[i] << " ";
			else {
				if (i==angle_paramnum) scriptout << radians_to_degrees(*(param[i]));
				else {
					if (((*(param[i]) != 0.0) and (abs(*(param[i])) < 1e-3)) or (abs(*(param[i]))) > 1e3) output_field_in_sci_notation(param[i],scriptout,false);
					else scriptout << *(param[i]);
				}
				if (anchor_parameter[i]) scriptout << "/anchor=" << parameter_anchor_lens[i]->lens_number << "," << parameter_anchor_paramnum[i];
				scriptout << " ";
			}
		}
		scriptout << " z=" << zlens << endl;
	}
	for (int i=0; i < n_params-1; i++) {
		if (vary_params[i]) scriptout << "1 ";
		else scriptout << "0 ";
	}
	if (vary_params[n_params-1]) scriptout << "varyz=1" << endl; // the last parameter is always the redshift, whose flag can be omitted if not being varied
	else scriptout << endl;
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

void LensProfile::output_lens_command_nofit(string& command)
{
	command += "lens " + model_name + " ";
	if (ellipticity_mode != default_ellipticity_mode) {
		if ((lenstype != SHEAR) and (lenstype != PTMASS) and (lenstype != MULTIPOLE) and (lenstype != SHEET) and (lenstype != TABULATED))   // these models are not elliptical so emode is irrelevant
		{
			stringstream emodestr;
			string emodestring;
			emodestr << ellipticity_mode;
			emodestr >> emodestring;
			command += "emode=" + emodestring + " ";
		}
	}
	if (special_parameter_command != "") command += special_parameter_command += " ";

	string xcstring = "";
	string ycstring = "";
	if (center_defined) {
		for (int i=0; i < n_params-3; i++) {
				stringstream paramstr;
				paramstr << setprecision(16);
				string paramstring;
			if (i==angle_paramnum) {
				paramstr << radians_to_degrees(*(param[i]));
			}
			else {
				paramstr << *(param[i]);
			}
			paramstr >> paramstring;
			command += paramstring + " ";
		}
		stringstream xcstr;
		stringstream ycstr;
		xcstr << setprecision(16);
		xcstr << x_center;
		xcstr >> xcstring;
		ycstr << setprecision(16);
		ycstr << y_center;
		ycstr >> ycstring;
	} else {
		for (int i=0; i < n_params-1; i++) {
				stringstream paramstr;
				paramstr << setprecision(16);
				string paramstring;
			if (i==angle_paramnum) {
				paramstr << radians_to_degrees(*(param[i]));
			}
			else {
				paramstr << *(param[i]);
			}
			paramstr >> paramstring;
			command += paramstring + " ";
		}
	}
	stringstream zlstr;
	string zlstring;
	zlstr << setprecision(16);
	zlstr << zlens;
	zlstr >> zlstring;
	command += xcstring + " " + ycstring + " z=" + zlstring;
}

bool LensProfile::output_cosmology_info(const int lens_number)
{
	bool mass_converged, rhalf_converged;
	double sigma_cr, mtot, rhalf;
	mass_converged = calculate_total_scaled_mass(mtot);
	if (mass_converged) {
		rhalf_converged = calculate_half_mass_radius(rhalf,mtot);
		sigma_cr = lens->sigma_crit_arcsec(zlens,zsrc_ref);
		mtot *= sigma_cr;
		if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
		cout << "total mass: " << mtot << " M_sol" << endl;
		double kpc_to_arcsec = 206.264806/lens->angular_diameter_distance(zlens);
		if (rhalf_converged) cout << "half-mass radius: " << rhalf/kpc_to_arcsec << " kpc (" << rhalf << " arcsec)" << endl;
		cout << endl;
	}
	return false; // no cosmology-dependent physical parameters to calculate for this model
}

double LensProfile::average_log_slope(const double rmin, const double rmax)
{
	double k1, k2;
	k1 = kappa_rsq(rmin*rmin);
	k2 = kappa_rsq(rmax*rmax);
	return log(k2/k1)/log(rmax/rmin);
}

double LensProfile::average_log_slope_3d(const double rmin, const double rmax)
{
	double rho1, rho2;
	bool converged;
	rho1 = calculate_scaled_density_3d(rmin*rmin,1e-5,converged);
	rho2 = calculate_scaled_density_3d(rmax*rmax,1e-5,converged);
	return log(rho2/rho1)/log(rmax/rmin);
}

bool LensProfile::calculate_total_scaled_mass(double& total_mass)
{
	double u, mass_u, mass_u_prev;
	double re_major_axis, re_average;
	static const double mtol = 1e-5;
	static const int nmax = 34;
	get_einstein_radius(re_major_axis, re_average, 1.0);
	if (re_major_axis==0) return false;
	if (this->kapavgptr_rsq_spherical==NULL) return false;
	u = 1.0/(re_average*re_average);
	mass_u = mass_inverse_rsq(u);
	int n = 0;
	do {
		mass_u_prev = mass_u;
		u /= 8.0;
		mass_u = mass_inverse_rsq(u);
		if (++n == nmax) break;
	} while (abs(mass_u-mass_u_prev) > (mtol*mass_u));
	total_mass = M_PI*mass_u;
	if (n==nmax) return false;
	else return true;
}

double LensProfile::mass_inverse_rsq(const double u)
{
	return (this->*kapavgptr_rsq_spherical)(1.0/u)/u; // leaving out the M_PI here because it will be tacked on afterword
}

double LensProfile::mass_rsq(const double rsq)
{
	if (this->kapavgptr_rsq_spherical==NULL) return 0;
	return M_PI*rsq*(this->*kapavgptr_rsq_spherical)(rsq);
}

double LensProfile::half_mass_radius_root(const double r)
{
	return (mass_rsq(r*r)-mass_intval);
}

bool LensProfile::calculate_half_mass_radius(double& half_mass_radius, const double mtot_in)
{
	if (kapavgptr_rsq_spherical==NULL) {
		return false;
	}
	double mtot;
	if (mtot_in==-10) {
		if (calculate_total_scaled_mass(mtot)==false) return false;
	} else mtot = mtot_in;
	mass_intval = mtot/2;
	if ((half_mass_radius_root(rmin_einstein_radius)*half_mass_radius_root(rmax_einstein_radius)) > 0) {
		return false;
	}
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&LensProfile::half_mass_radius_root);
	half_mass_radius = BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-6);
	return true;
}

double LensProfile::calculate_scaled_mass_3d(const double r)
{
	if (analytic_3d_density) return calculate_scaled_mass_3d_from_analytic_rho3d(r);
	else return calculate_scaled_mass_3d_from_kappa(r);
}

double LensProfile::calculate_scaled_mass_3d_from_kappa(const double r)
{
	static const int max_iter = 6;
	int rho3d_nn = 40; // starting value
	double re_major_axis, re_average;
	get_einstein_radius(re_major_axis, re_average, 1.0);
	if (re_average==0.0) {
		warn("Einstein radius is returning zero; cannot calculate 3d mass");
		return 0;
	}
	int i, j, iter=0;
	double R,Rmin,logx,logxmin,logxmax,logxstep;
	logxmin = -6; logxmax = log(r/re_average)/ln10;
	Rmin = re_average*pow(10,logxmin);
	bool converged, menc_converged, prev_converged, convergence_everywhere;
	bool converge_at_small_r;
	bool first_convergence;
	bool trouble_at_small_and_large_r;
	double convergence_beyond_radius;
	double tolerance = 1e-3;
	double quadtolerance = tolerance / 2.0; // make integral tolerance a bit stricter

	double *rho3dvals, *new_rho3dvals;
	double *logxvals;
	double temppat, mass3d, prev_mass3d;
	double (GaussPatterson::*mptr)(double);
	convergence_everywhere = true;
	converge_at_small_r = true;
	first_convergence = false;
	mptr = static_cast<double (GaussPatterson::*)(double)> (&LensProfile::mass3d_r_integrand);
	do {
		convergence_beyond_radius = 0;
		converged=true; // this refers to the individual integrals converging
		prev_converged=true; // this refers to previous logx value during loop over logx
		trouble_at_small_and_large_r = false;
		if (iter > 0) {
			prev_mass3d = mass3d;
			rho3d_nn = 2*(rho3d_nn-1) + 1; // double the number of steps (not points!)
		}
		new_rho3dvals = new double[rho3d_nn];
		logxvals = new double[rho3d_nn];
		logxstep = (logxmax-logxmin)/(rho3d_nn-1);
		for (i=0, logx=logxmin; i < rho3d_nn; i++, logx += logxstep) {
			logxvals[i] = logx;
			if ((iter > 0) and (i%2==0)) {
				j = i/2;
				new_rho3dvals[i] = rho3dvals[j];
			} else {
				R = re_average*pow(10,logx);
				new_rho3dvals[i] = calculate_scaled_density_3d(R,quadtolerance,converged);
				if (new_rho3dvals[i]*0.0 != 0.0) cout << "r(" << i << "): R=" << R << " nan value\n";
				if (converged==false) {
					convergence_everywhere = false;
					if ((converge_at_small_r==false) and (prev_converged==true)) trouble_at_small_and_large_r = true;
					if (i==0) converge_at_small_r = false;
				} else {
					if (prev_converged==false) {
						if ((converge_at_small_r == false) and (first_convergence == false)) {
							convergence_beyond_radius = R;
						}
					}
					if (first_convergence==false) first_convergence = true;
				}
				prev_converged = converged;
			}
		}
		if (iter > 0) {
			delete[] rho3dvals;
		}
		rho3dvals = new_rho3dvals;

		rho3d_logx_spline = new Spline(logxvals,rho3dvals,rho3d_nn);
		mass_intval = re_average;

		temppat = pat_tolerance;
		SetGaussPatterson(quadtolerance,false);
		mass3d = 4*M_PI*AdaptiveQuad(mptr,Rmin,r,menc_converged);
		SetGaussPatterson(temppat,true);
		delete[] logxvals;
		delete rho3d_logx_spline;
		if (iter > 0) {
			if (abs(mass3d-prev_mass3d) < tolerance*abs(mass3d)) break;
		}
	} while (++iter < max_iter);
	if (iter==max_iter) warn("Enclosed mass did not converge after nmax=%i iterations of refining 3d density profile",max_iter);

	delete[] rho3dvals;

	if (!convergence_everywhere) {
		if ((converge_at_small_r==false) and (!trouble_at_small_and_large_r)) {
			warn("Gauss-Patterson quadrature did not converge for R smaller than %g (tol=%g) (using NMAX=511 points)",convergence_beyond_radius,quadtolerance);
		} else warn("Gauss-Patterson quadrature did not achieve desired convergence (tol=%g) for all r after NMAX=511 points",quadtolerance);
	}
	if (menc_converged==false) warn("Gauss-Patterson quadrature did not converge for enclosed mass integral (tol=%g) using NMAX=511 points",quadtolerance);

	return mass3d;
}

double LensProfile::calculate_scaled_mass_3d_from_analytic_rho3d(const double r)
{
	bool menc_converged;
	double mass3d, tolerance = 1e-4;

	double temppat = pat_tolerance;
	SetGaussPatterson(tolerance,false);
	double (GaussPatterson::*mptr)(double);
	mptr = static_cast<double (GaussPatterson::*)(const double)> (&LensProfile::mass3d_r_integrand_analytic);
	mass3d = 4*M_PI*AdaptiveQuad(mptr,0,r,menc_converged);
	SetGaussPatterson(temppat,true);
	if (menc_converged==false) warn("Gauss-Patterson quadrature did not converge for enclosed mass integral (tol=%g) using NMAX=511 points",tolerance);
	return mass3d;
}

double LensProfile::mass3d_r_integrand(const double r)
{
	double logx = log(r/mass_intval)/ln10;
	if (logx < rho3d_logx_spline->xmin()) return (r*r*rho3d_logx_spline->extend_inner_logslope(logx));
	return r*r*rho3d_logx_spline->splint(logx);
}

double LensProfile::mass3d_r_integrand_analytic(const double r)
{
	return r*r*rho3d_r_integrand_analytic(r);
}

double LensProfile::rho3d_r_integrand_analytic(const double r)
{
	return 0; // This function must be overloaded in lens models where analytic_3d_density==true
}

double LensProfile::calculate_scaled_density_3d(const double r, const double tolerance, bool& converged)
{
	if (analytic_3d_density) return rho3d_r_integrand_analytic(r);
	mass_intval = r*r;
	double (GaussPatterson::*mptr)(double);
	mptr = static_cast<double (GaussPatterson::*)(double)> (&LensProfile::rho3d_w_integrand);
	double temppat = pat_tolerance;
	SetGaussPatterson(tolerance,false);
	double ans = -(2*r/M_PI)*AdaptiveQuad(mptr,0,1,converged);
	SetGaussPatterson(temppat,true);
	return ans;
}

double LensProfile::rho3d_w_integrand(const double w)
{
	double wsq = w*w;
	return kappa_rsq_deriv(mass_intval/wsq)/(wsq*sqrt(1-wsq));
	//double ans = kappa_rsq_deriv(mass_intval/wsq)/(wsq*sqrt(1-wsq));
	//if (ans*0.0 != 0.0) cout << "NAN! " << wsq << " " << mass_intval/wsq << " " << kappa_rsq_deriv(mass_intval/wsq) << endl;
	//return ans;
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
	if (q > 1) q = 1.0; // don't allow q>1
	if (q<=0) q = 0.001; // just to avoid catastrophe
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
	rmin_einstein_radius = 1e-5; rmax_einstein_radius = 1e4;
	SetGaussLegendre(nn);
	integral_tolerance = acc;
	SetGaussPatterson(acc,true);
}

void LensProfile::update_meta_parameters_and_pointers()
{
	set_integration_pointers();
	set_model_specific_integration_pointers();
	update_meta_parameters();
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
	def_and_hess_ptr = &LensProfile::deflection_and_hessian_together;
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

void LensProfile::deflection_from_elliptical_potential_experimental(const double x, const double y, lensvector& def)
{
	const double eta = 0.0;
	const double rse = 15;

	double denom = 1 + 0.5*eta*SQR(y/rse); 
	double kapavg_resq = (this->*kapavgptr_rsq_spherical)(((1-epsilon)*x*x + (1+epsilon)*y*y) / denom);

	def[0] = (kapavg_resq)*(1-epsilon)*x/denom;
	def[1] = (kapavg_resq)*((1+epsilon)*y/denom - (eta*y/(rse*rse))*((1-epsilon)*x*x+(1+epsilon)*y*y)/(2*denom*denom));
}

/*
double LensProfile::test_resq(const double x, const double y)
{
	const double eta = 0.1;
	const double rse = 20;
	double denom;
	//double bb=0.5*eta*(x*x)/rse;
	//double re0 = (1-epsilon)*x*x + (1+epsilon)*y*y;
	//double re = SQR(-bb + sqrt(bb*bb + re0));
	//cout << bb << " " << re0 << " " << re << endl;
	//return SQR(-bb + sqrt(bb*bb + re0));
	//double denom = 1 + 0.5*eta*SQR(x/rse); 
	if (eta > 0)
		denom = 1 + 0.5*eta*SQR(y/rse); 
	else
		denom = 1 + 0.5*abs(eta)*SQR(x/rse); 
	return ((1-epsilon)*x*x + (1+epsilon)*y*y) / denom;
}
*/

double LensProfile::test_defx(const double x, const double y)
{
	lensvector deff;
	deflection_from_elliptical_potential_experimental(x,y,deff);
	return deff[0];
}

double LensProfile::test_defy(const double x, const double y)
{
	lensvector deff;
	deflection_from_elliptical_potential_experimental(x,y,deff);
	return deff[1];
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
	//cout << "Trying " << x << " " << y << " " << epsilon << " " << rsq << endl;
	//double wtf = (this->*kapavgptr_rsq_spherical)(rsq); 
	//cout << "wtf=" << wtf << ", rsq=" << rsq << endl;
	//cout << "Trying rsq=" << rsq << endl;
	shearmag = ((this->*kapavgptr_rsq_spherical)(rsq)) - kap_r; // shear from the spherical model
	//cout << "shearmag=" << shearmag << ", rsq=" << rsq << endl;
	kap = kap_r + epsilon*shearmag*cos2phi;
	gamma1 = -epsilon*kap_r - shearmag*cos2phi;
	gamma2 = -sqrt(1-epsilon*epsilon)*shearmag*sin2phi;
	hess[0][0] = kap + gamma1;
	hess[1][1] = kap - gamma1;
	hess[0][1] = gamma2;
	hess[1][0] = gamma2;
	if ((shearmag * 0.0) != 0.0) die("die2");
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

void LensProfile::hessian_from_elliptical_potential_experimental(const double x, const double y, lensmatrix& hess)
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
	//hess[0][0] = kap + gamma1;
	//hess[1][1] = kap - gamma1;
	//hess[0][1] = gamma2;
	//hess[1][0] = gamma2;

	//double hess00, hess11, hess01;
	double dx = 1e-4;
	hess[0][0] = (test_defx(x+dx,y) - test_defx(x-dx,y))/(2*dx);
	hess[1][1] = (test_defy(x,y+dx) - test_defy(x,y-dx))/(2*dx);
	hess[0][1] = (test_defx(x,y+dx) - test_defx(x,y-dx))/(2*dx);
	hess[1][0] = hess[0][1];
	//cout << hess00 << " " << hess[0][0] << " " << hess11 << " " << hess[1][1] << " " << hess01 << " " << hess[0][1] << endl;
}

double LensProfile::kappa_from_elliptical_potential_experimental(const double x, const double y)
{
	// Formulas derived in Dumet-Montoya et al. (2012)
	double cos2phi, exsq, eysq, rsq, kap_r, shearmag;
	exsq = (1-epsilon)*x*x; // elliptical x^2
	eysq = (1+epsilon)*y*y; // elliptical y^2
	rsq = exsq+eysq; // elliptical r^2
	cos2phi = (exsq - eysq) / rsq;

	kap_r = kappa_rsq(rsq);
	shearmag = (this->*kapavgptr_rsq_spherical)(rsq) - kap_r;

	double hess00, hess11, hess01;
	double dx = 1e-4;
	hess00 = (test_defx(x+dx,y) - test_defx(x-dx,y))/(2*dx);
	hess11 = (test_defy(x,y+dx) - test_defy(x,y-dx))/(2*dx);
	hess01 = (test_defx(x,y+dx) - test_defx(x,y-dx))/(2*dx);
	double testkap1 = (hess00+hess11)/2;
	//cout << hess00/2 << " " << hess11/2 << " " << testkap1 << endl;

	//cout << hess00 << " " << hess[0][0] << " " << hess11 << " " << hess[1][1] << " " << hess01 << " " << hess[0][1] << endl;

	//double testkap0 = (kap_r + epsilon*shearmag*cos2phi);
	//cout << testkap0 << " " << testkap1 << endl;

	//return (kap_r + epsilon*shearmag*cos2phi);
	return testkap1;
}

void LensProfile::kappa_deflection_and_hessian_from_elliptical_potential(const double x, const double y, double& kap, lensvector& def, lensmatrix& hess)
{
	// Formulas derived in Dumet-Montoya et al. (2012)
	double cos2phi, sin2phi, exsq, eysq, rsq, gamma1, gamma2, kap_r, shearmag, kapavg;
	exsq = (1-epsilon)*x*x; // elliptical x^2
	eysq = (1+epsilon)*y*y; // elliptical y^2
	rsq = exsq+eysq; // elliptical r^2

	kapavg = (this->*kapavgptr_rsq_spherical)(rsq);

	def[0] = kapavg*(1-epsilon)*x;
	def[1] = kapavg*(1+epsilon)*y;

	cos2phi = (exsq - eysq) / rsq;
	sin2phi = 2*q*(1+epsilon)*x*y/rsq;
	kap_r = kappa_rsq(rsq);
	shearmag = kapavg - kap_r; // shear from the spherical model
	kap = kap_r + epsilon*shearmag*cos2phi;
	gamma1 = -epsilon*kap_r - shearmag*cos2phi;
	gamma2 = -sqrt(1-epsilon*epsilon)*shearmag*sin2phi;
	hess[0][0] = kap + gamma1;
	hess[1][1] = kap - gamma1;
	hess[0][1] = gamma2;
	hess[1][0] = gamma2;
	//if ((kapavg * 0.0) != 0.0) die("die1");
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

void LensProfile::update_zlens_meta_parameters()
{
	if (zlens != zlens_current) {
		sigma_cr = lens->sigma_crit_arcsec(zlens,zsrc_ref);
		kpc_to_arcsec = 206.264806/lens->angular_diameter_distance(zlens);
		zlens_current = zlens;
	}
}

void LensProfile::calculate_ellipticity_components()
{
	if ((ellipticity_mode != -1) and (use_ellipticity_components)) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		epsilon = (1-q)*cos(2*theta_eff);
		epsilon2 = (1-q)*sin(2*theta_eff);
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

void LensProfile::rotate_back(double &x, double &y)
{
	// perform a clockwise rotation of the coordinate system to transform from the coordinate system of the rotated galaxy
	double xp = x*costheta - y*sintheta;
	y = x*sintheta + y*costheta;
	x = xp;
}

double LensProfile::potential(double x, double y)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	if (ellipticity_mode==3) {
		return (this->*potptr_rsq_spherical)((1-epsilon)*x*x + (1+epsilon)*y*y); // ellipticity is put into the potential in this mode
		//return (this->*potptr_rsq_spherical)(test_resq(x,y)); // ellipticity is put into the potential in this mode
	} else {
		return (this->*potptr)(x,y);
	}
}

void LensProfile::potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	double kap; // not used here, but no significant overhead wasted so it's ok
	if ((ellipticity_mode==3) and (q != 1)) {
		kappa_deflection_and_hessian_from_elliptical_potential(x,y,kap,def,hess);
	} else {
		(this->*def_and_hess_ptr)(x,y,def,hess);
	}
	if (sintheta != 0) def.rotate_back(costheta,sintheta);
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

void LensProfile::kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	if ((ellipticity_mode==3) and (q != 1)) {
		kappa_deflection_and_hessian_from_elliptical_potential(x,y,kap,def,hess);
	} else {
		kap = kappa_rsq((x*x + y*y/(q*q))/(f_major_axis*f_major_axis));
		(this->*def_and_hess_ptr)(x,y,def,hess);
	}
	if (sintheta != 0) def.rotate_back(costheta,sintheta);
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

void LensProfile::deflection_and_hessian_together(const double x, const double y, lensvector &def, lensmatrix& hess)
{
	if ((defptr == &LensProfile::deflection_numerical) and (hessptr == &LensProfile::hessian_numerical)) {
		if ((abs(x) < 1e-14) and (abs(y) < 1e-14)) {
			def[0]=0;
			def[1]=0;
			hess[0][0]=0;
			hess[1][1]=0;
			hess[0][1]=0;
			hess[1][0]=0;
			return;
		}
		deflection_and_hessian_numerical(x,y,def,hess); // saves time to calculate deflection & hessian together, because they have two integrals in common
	} else {
		(this->*defptr)(x,y,def);
		(this->*hessptr)(x,y,hess);
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
		//deflection_from_elliptical_potential_experimental(x,y,def);
	} else {
		(this->*defptr)(x,y,def);
	}
	if (sintheta != 0) def.rotate_back(costheta,sintheta);
}

double LensProfile::kappa(double x, double y)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	if ((ellipticity_mode==3) and (q != 1)) {
		return kappa_from_elliptical_potential(x,y);
		//return kappa_from_elliptical_potential_experimental(x,y);
	} else {
		return kappa_rsq((x*x + y*y/(q*q))/(f_major_axis*f_major_axis));
	}
}

void LensProfile::hessian(double x, double y, lensmatrix& hess)
{
	// switch to coordinate system centered on lens profile
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	if ((ellipticity_mode==3) and (q != 1)) {
		hessian_from_elliptical_potential(x,y,hess);
		//hessian_from_elliptical_potential_experimental(x,y,hess);
	} else {
		(this->*hessptr)(x,y,hess);
	}
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

bool LensProfile::has_kapavg_profile()
{
	if (this->kapavgptr_rsq_spherical==NULL) return false;
	else return true;
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
	//cout << "RMIN " << rmin_einstein_radius << " " << einstein_radius_root(rmin_einstein_radius) << endl;
	//cout << "RMAX " << rmax_einstein_radius << " " << einstein_radius_root(rmax_einstein_radius) << endl;
	//int i, nn=200000;
	//double rf, r, rstep = (rmax_einstein_radius-rmin_einstein_radius)/(nn-1);
	//double (Brent::*bptr)(const double);
	//bptr = static_cast<double (Brent::*)(const double)> (&LensProfile::einstein_radius_root);
	//for (i=0, r=rmin_einstein_radius; i < nn; i++, r += rstep) {
		//rf = (this->*bptr)(r);
	//}
	//die();
	if ((einstein_radius_root(rmin_einstein_radius)*einstein_radius_root(rmax_einstein_radius)) > 0) {
		// multiple imaging does not occur with this lens
		re_major_axis = 0;
		re_average = 0;
		return;
	}
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&LensProfile::einstein_radius_root);
	re_major_axis = f_major_axis * BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-6);
	if (ellipticity_mode != -1) re_average = re_major_axis * sqrt(q);
	else re_average = re_major_axis; // not an elliptical lens, so q has no meaning
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
	double kavg, rsq, r_kpc, rho3d, scaled_rho;
	bool converged;
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		rsq = r*r;
		if (kapavgptr_rsq_spherical==NULL) kavg=0; // just in case there is no radial deflection function defined
		else kavg = (this->*kapavgptr_rsq_spherical)(rsq);
		r_kpc = r/kpc_to_arcsec;
		scaled_rho = calculate_scaled_density_3d(r,1e-4,converged);
		rho3d = (sigma_cr*CUBE(kpc_to_arcsec))*scaled_rho;

		//kout << r << " " << kappa_rsq(rsq) << " " << kavg << " " << kavg*r << " " << M_PI*kavg*rsq*sigma_cr << " " << r_kpc << " " << kappa_rsq(rsq)*sigma_cr << " " << rho3d << endl;
		//if (kdname != NULL) kdout << r << " " << 2*r*kappa_rsq_deriv(rsq) << endl;
	}
}

void LensProfile::plot_kappa_profile(const int n_rvals, double* rvals, double* kapvals, double* kapavgvals)
{
	double rsq;
	for (int i=0; i < n_rvals; i++) {
		rsq = SQR(rvals[i]);
		kapvals[i] = kappa_rsq(rsq);
		if (kapavgptr_rsq_spherical==NULL) kapavgvals[i] = 0; // just in case there is no radial deflection function defined
		else kapavgvals[i] = (this->*kapavgptr_rsq_spherical)(rsq);
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
		ans = (2.0/rsq)*romberg_open(sptr, 0, sqrt(rsq), integral_tolerance, 5);
	}
	else if (integral_method == Gaussian_Quadrature)
	{
		double (GaussianIntegral::*sptr)(double);
		sptr = static_cast<double (GaussianIntegral::*)(double)> (&LensProfile::mass_enclosed_spherical_integrand);
		ans = (2.0/rsq)*NIntegrate(sptr,0,sqrt(rsq));
	}
	else if (integral_method == Gauss_Patterson_Quadrature)
	{
		double (GaussPatterson::*sptr)(double);
		sptr = static_cast<double (GaussPatterson::*)(double)> (&LensProfile::mass_enclosed_spherical_integrand);
		bool converged;
		ans = (2.0/rsq)*AdaptiveQuad(sptr,0,sqrt(rsq),converged);
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
	bool converged;
	double ans;
	LensIntegral lens_integral(this,rsq,0,1.0,0);
	ans = 0.5*lens_integral.i_integral(converged);
	return ans;
}

void LensProfile::deflection_numerical(const double x, const double y, lensvector& def)
{
	if ((abs(x) < 1e-14) and (abs(y) < 1e-14)) {
		// return zero deflection, since there's a risk of getting 'NaN' if the center of the profile is evaluated
		def[0]=0;
		def[1]=0;
		return;
	}
	bool converged;
	def[0] = q*x*j_integral(x,y,0,converged);
	warn_if_not_converged(converged,x,y);
	def[1] = q*y*j_integral(x,y,1,converged);
	warn_if_not_converged(converged,x,y);
}

void LensProfile::hessian_numerical(const double x, const double y, lensmatrix& hess)
{
	if ((abs(x) < 1e-14) and (abs(y) < 1e-14)) {
		hess[0][0]=0;
		hess[1][1]=0;
		hess[0][1]=0;
		hess[1][0]=0;
		return;
	}
	bool converged, converged2;
	hess[0][0] = 2*q*x*x*k_integral(x,y,0,converged) + q*j_integral(x,y,0,converged2);
	warn_if_not_converged(converged,x,y);
	warn_if_not_converged(converged2,x,y);
	hess[1][1] = 2*q*y*y*k_integral(x,y,2,converged) + q*j_integral(x,y,1,converged2);
	warn_if_not_converged(converged,x,y);
	warn_if_not_converged(converged2,x,y);
	hess[0][1] = 2*q*x*y*k_integral(x,y,1,converged);
	warn_if_not_converged(converged,x,y);
	hess[1][0] = hess[0][1];
}

void LensProfile::deflection_and_hessian_numerical(const double x, const double y, lensvector& def, lensmatrix& hess)
{
	// You should make it save the kappa and kappa' values evaluated during J0 and K0 so it doesn't have to evaluate them again for
	// J1, K1 and K2 (unless higher order quadrature is required for convergence, in which case extra evaluations must be done). This
	// will save a significant amount of time, but might take some doing to implement
	bool converged;
	double jint0, jint1;
	jint0 = j_integral(x,y,0,converged);
	warn_if_not_converged(converged,x,y);
	jint1 = j_integral(x,y,1,converged);
	warn_if_not_converged(converged,x,y);
	def[0] = q*x*jint0;
	def[1] = q*y*jint1;
	hess[0][0] = 2*q*x*x*k_integral(x,y,0,converged) + q*jint0;
	warn_if_not_converged(converged,x,y);
	hess[1][1] = 2*q*y*y*k_integral(x,y,2,converged) + q*jint1;
	warn_if_not_converged(converged,x,y);
	hess[0][1] = 2*q*x*y*k_integral(x,y,1,converged);
	warn_if_not_converged(converged,x,y);
	hess[1][0] = hess[0][1];
}

inline void LensProfile::warn_if_not_converged(const bool& converged, const double &x, const double &y)
{
	if ((!converged) and (output_integration_errors)) {
		if (integral_method==Gauss_Patterson_Quadrature) {
			if ((lens->mpi_id==0) and (lens->warnings)) {
				cout << "*WARNING*: Gauss-Patterson did not converge (x=" << x << ",y=" << y << "); switched to 1023-pt. Gauss-Legendre                  " << endl;
				cout << "Lens: " << model_name << ", Params: ";
				for (int i=0; i < n_params; i++) {
					cout << paramnames[i] << "=";
					if (i==angle_paramnum) cout << radians_to_degrees(*(param[i])) << " degrees";
					else cout << *(param[i]);
					if (i != n_params-1) cout << ", ";
				}
				cout << "     " << endl;
				if (lens->running_fit) {
					cout << "\033[2A";
				}
			}
		}
	}
}

double LensProfile::potential_numerical(const double x, const double y)
{
	if (this->kapavgptr_rsq_spherical==NULL) return 0.0; // cannot calculate potential without a spherical deflection defined
	bool converged;
	double ans;
	LensIntegral lens_integral(this,x*x,y*y,q,0);
	ans = 0.5*q*lens_integral.i_integral(converged);
	warn_if_not_converged(converged,x,y);
	return ans;
}

inline double LensProfile::j_integral(const double x, const double y, const int n, bool &converged)
{
	LensIntegral lens_integral(this,x*x,y*y,q,n);
	return lens_integral.j_integral(converged);
}

inline double LensProfile::k_integral(const double x, const double y, const int n, bool &converged)
{
	LensIntegral lens_integral(this,x*x,y*y,q,n);
	return lens_integral.k_integral(converged);
}

bool LensProfile::core_present() { return false; }

double LensIntegral::i_integral(bool &converged)
{
	converged = true; // will change if convergence not achieved
	double ans;
	if (profile->integral_method == Romberg_Integration)
	{
		double (Romberg::*iptr)(const double);
		iptr = static_cast<double (Romberg::*)(const double)> (&LensIntegral::i_integrand_prime);
		ans = romberg_open(iptr, 0, 1, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		double (LensIntegral::*iptr)(double) = &LensIntegral::i_integrand_prime;
		ans = GaussIntegrate(iptr,0,1);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		double (LensIntegral::*iptr)(double) = &LensIntegral::i_integrand_prime;
		ans = PattersonIntegrate(iptr,0,1,converged);
	}
	else die("unknown integral method");
	return ans;
}

double LensIntegral::j_integral(bool &converged)
{
	converged = true; // will change if convergence not achieved
	double ans;
	if (profile->integral_method == Romberg_Integration)
	{
		double (Romberg::*jptr)(const double);
		jptr = static_cast<double (Romberg::*)(const double)> (&LensIntegral::j_integrand_prime);
		ans = romberg_open(jptr, 0, 1, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		double (LensIntegral::*jptr)(double) = &LensIntegral::j_integrand_prime;
		ans = GaussIntegrate(jptr,0,1);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		double (LensIntegral::*jptr)(double) = &LensIntegral::j_integrand_prime;
		ans = PattersonIntegrate(jptr,0,1,converged);
		/*
		if (!converged) {
			//int i, nn=511;
			//double w, wstep = 1.0/(nn-1);
			////for (i=0, w=0; i < nn; i++, w += wstep) {
			//for (i=0; i < nn; i++) {
				//w = 0.5 + 0.5*profile->pat_points[i];
				//cout << w << " " << j_integrand_prime(w) << endl;
			//}
			cout << endl << endl;
			double w = 0.999999;
			double u = w*w;
			double qfac = 1 - one_minus_qsq*u;
			double wtf2 = (2*w*profile->kappa_rsq(u*(xsqval + ysqval/qfac)*fsqinv) / pow(qfac, nval_plus_half));

			double wtf = 0.999999*((xsqval + ysqval/qfac)*fsqinv);
			cout << xsqval << " " << ysqval << " " << wtf << " " << qfac << " " << one_minus_qsq << endl;
			cout << endl;

			//die();
			int i, nn=1023;
			double wstep = 1.0/(nn-1);
			ofstream blargh("ctwtf.dat");

			//for (i=0, w=0; i < nn; i++, w += wstep) {
			for (i=0; i < nn; i++) {
				w = 0.5 + 0.5*profile->points[i];
				blargh << w << " " << j_integrand_prime(w) << endl;
			}

			//die();
		}
		*/
	}
	else die("unknown integral method");
	return ans;
}

double LensIntegral::k_integral(bool &converged)
{
	converged = true; // will change if convergence not achieved
	double ans;
	if (profile->integral_method == Romberg_Integration)
	{
		double (Romberg::*kptr)(const double);
		kptr = static_cast<double (Romberg::*)(const double)> (&LensIntegral::k_integrand_prime);
		ans = romberg_open(kptr, 0, 1, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		double (LensIntegral::*kptr)(double) = &LensIntegral::k_integrand_prime;
		ans = GaussIntegrate(kptr,0,1);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		double (LensIntegral::*kptr)(double) = &LensIntegral::k_integrand_prime;
		ans = PattersonIntegrate(kptr,0,1,converged);
	}

	else die("unknown integral method");
	return ans;
}

// i,j,k integrals are in form similar to Keeton (2001), but generalized to allow for different
// definitions of the elliptical radius. I have also made the substitution
// u=w*w (easier for Gaussian quadrature; makes kappa singularity more manageable)

double LensIntegral::i_integrand_prime(const double w)
{
	u = w*w;
	qfac = 1 - one_minus_qsq*u;
	xisq = u*(xsqval + ysqval/qfac)*fsqinv;
	return (2*w*(xisq/u)*(profile->kapavg_spherical_generic)(xisq) / sqrt(qfac))/fsqinv;
}

double LensIntegral::j_integrand_prime(const double w)
{
	u = w*w;
	qfac = 1 - one_minus_qsq*u;
	return (2*w*profile->kappa_rsq(u*(xsqval + ysqval/qfac)*fsqinv) / pow(qfac, nval_plus_half));
}

double LensIntegral::k_integrand_prime(const double w)
{
	u = w*w;
	qfac = 1 - one_minus_qsq*u;
	return fsqinv*(2*w*u*profile->kappa_rsq_deriv(u*(xsqval + ysqval/qfac)*fsqinv) / pow(qfac, nval_plus_half));
}

double LensIntegral::GaussIntegrate(double (LensIntegral::*func)(const double), const double a, const double b)
{
	double result = 0;

	for (int i = 0; i < n_gausspoints; i++)
		result += gaussweights[i]*(this->*func)(((a+b) + (b-a)*gausspoints[i])/2.0);

	return (b-a)*result/2.0;
}

double LensIntegral::PattersonIntegrate(double (LensIntegral::*func)(double), double a, double b, bool& converged)
{
	double result=0, result_old;
	int i, level=0, istep, istart;
	double absum = (a+b)/2, abdif = (b-a)/2;
	double *weightptr;
	converged = true; // will change to false if convergence not achieved

	int order, j;
	do {
		weightptr = pat_weights[level];
		result_old = result;
		order = profile->pat_orders[level];
		istep = (profile->pat_N+1) / (order+1);
		istart = istep - 1;
		istep *= 2;
		result = 0;
		for (j=0, i=istart; j < order; j += 2, i += istep) {
			pat_funcs[i] = (this->*func)(absum + abdif*pat_points[i]);
			result += weightptr[j]*pat_funcs[i];
		}
		for (j=1, i=istep-1; j < order; j += 2, i += istep) {
			result += weightptr[j]*pat_funcs[i];
		}
		if ((level > 1) and (abs(result-result_old) < profile->pat_tolerance*abs(result))) break;
	} while (++level < 9);

	if (level == 9) {
		profile->SetGaussLegendre(1023);
		gausspoints = profile->points;
		gaussweights = profile->weights;

		result = 0;
		for (int i = 0; i < profile->numberOfPoints; i++)
			result += gaussweights[i]*(this->*func)(absum + abdif*gausspoints[i]);

		converged = false;

		/*
		cerr << "level=" << level-1 << ", order=" << profile->pat_orders[level-1] << ", old_dif=" << abs((result-result_old)/result) << " (s=" << result << "), ";
		profile->SetGaussLegendre(profile->pat_orders[level-1]);
		gausspoints = profile->points;
		gaussweights = profile->weights;

		result_old = result;
		result = 0;

		for (int i = 0; i < profile->numberOfPoints; i++)
			result += gaussweights[i]*(this->*func)(absum + abdif*gausspoints[i]);

		//if (abs(result-result_old) > profile->pat_tolerance*abs(result)) converged = false;
		cerr << "new_dif=" << abs((result-result_old)/result) << " (s=" << result << "), ";
		profile->SetGaussLegendre(2*profile->pat_orders[level-1]);
		gausspoints = profile->points;
		gaussweights = profile->weights;

		result_old = result;
		result = 0;

		for (int i = 0; i < profile->numberOfPoints; i++)
			result += gaussweights[i]*(this->*func)(absum + abdif*gausspoints[i]);

		//if (abs(result-result_old) > profile->pat_tolerance*abs(result)) converged = false;
		cerr << "new_dif2=" << abs((result-result_old)/result) << " (s=" << result << ")" << endl;
		*/
	}

	return abdif*result;
}

