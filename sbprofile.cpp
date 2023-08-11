#include "sbprofile.h"
#include "qlens.h"
#include "egrad.h"
#include "mathexpr.h"
#include "errors.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

bool SB_Profile::orient_major_axis_north = false; // At the moment, this setting cannot be changed; it should probably be removed altogether
bool SB_Profile::use_sb_ellipticity_components = false;
int SB_Profile::default_ellipticity_mode = 1;
bool SB_Profile::use_fmode_scaled_amplitudes = false;
bool SB_Profile::fourier_use_eccentric_anomaly = true;
bool SB_Profile::fourier_sb_perturbation = false; // if true, add fourier modes to the surface brightness, rather than the elliptical radius
double SB_Profile::zoom_split_factor = 2;
double SB_Profile::zoom_scale = 4;
double SB_Profile::SB_noise = 0.1; // used to help determine subpixel splittings to resolve SB profiles (zoom mode)

SB_Profile::SB_Profile(const char *splinefile, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in, QLens* qlens_in)
{
	model_name = "sbspline";
	sbtype = SB_SPLINE;
	setup_base_source_properties(6,2,true);
	qlens = qlens_in;
	qx_parameter = qx_in;
	f_parameter = f_in;
	sb_spline.input(splinefile);
	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
}

void SB_Profile::setup_base_source_properties(const int np, const int sbprofile_np, const bool is_elliptical_source, const int pmode_in)
{
	set_null_ptrs_and_values(); // sets pointers to NULL to make sure qlens doesn't try to delete them during setup
	parameter_mode = pmode_in;
	center_anchored_to_lens = false;
	center_anchored_to_source = false;
	if (is_elliptical_source) {
		ellipticity_mode = default_ellipticity_mode;
	} else {
		ellipticity_mode = -1; // indicates not an elliptical source
	}
	n_fourier_modes = 0;
	include_boxiness_parameter = false;
	include_truncation_radius = false;
	is_lensed = true; // default
	zoom_subgridding = false; // default
	ellipticity_gradient = false;
	fourier_gradient = false;
	contours_overlap = false; // only relevant for ellipticity gradient mode
	overlap_log_penalty_prior = 0;
	lensed_center_coords = false;
	set_nparams(np);
	sbprofile_nparams = sbprofile_np;

	assign_param_pointers();
	assign_paramnames();

	include_limits = false;
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
	set_null_ptrs_and_values();
	model_name = sb_in->model_name;
	sbtype = sb_in->sbtype;
	qlens = sb_in->qlens;
	sb_number = sb_in->sb_number;
	set_nparams(sb_in->n_params);
	sbprofile_nparams = sb_in->sbprofile_nparams;
	parameter_mode = sb_in->parameter_mode;
	center_anchored_to_lens = sb_in->center_anchored_to_lens;
	center_anchor_lens = sb_in->center_anchor_lens;
	center_anchored_to_source = sb_in->center_anchored_to_source;
	center_anchor_source = sb_in->center_anchor_source;
	is_lensed = sb_in->is_lensed;
	zoom_subgridding = sb_in->zoom_subgridding;
	ellipticity_mode = sb_in->ellipticity_mode;
	ellipticity_gradient = sb_in->ellipticity_gradient;
	fourier_gradient = sb_in->fourier_gradient;
	lensed_center_coords = sb_in->lensed_center_coords;

	paramnames = sb_in->paramnames;
	latex_paramnames = sb_in->latex_paramnames;
	latex_param_subscripts = sb_in->latex_param_subscripts;
	angle_param.input(n_params);
	for (int i=0; i < n_params; i++) angle_param[i] = false; // the angle params will be recognized when assign_param_pointers() is called
	n_vary_params = sb_in->n_vary_params;
	vary_params.input(sb_in->vary_params);
	stepsizes.input(sb_in->stepsizes);
	set_auto_penalty_limits.input(sb_in->set_auto_penalty_limits);
	penalty_lower_limits.input(sb_in->penalty_lower_limits);
	penalty_upper_limits.input(sb_in->penalty_upper_limits);

	q = sb_in->q;
	epsilon1 = sb_in->epsilon1;
	epsilon2 = sb_in->epsilon2;
	angle_param_exists = sb_in->angle_param_exists;
	if (angle_param_exists) set_angle_radians(sb_in->theta);
	x_center = sb_in->x_center;
	y_center = sb_in->y_center;
	if (lensed_center_coords) {
		x_center_lensed = sb_in->x_center_lensed;
		y_center_lensed = sb_in->y_center_lensed;
	}

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
	include_truncation_radius = sb_in->include_truncation_radius;
	if (include_boxiness_parameter) c0 = sb_in->c0;
	if (include_truncation_radius) rt = sb_in->rt;
	if (ellipticity_gradient) {
		egrad_mode = sb_in->egrad_mode;
		egrad_ellipticity_mode = sb_in->egrad_ellipticity_mode;
		center_gradient = sb_in->center_gradient;
		int i,j;
		for (i=0; i < 4; i++) {
			n_egrad_params[i] = sb_in->n_egrad_params[i];
			geometric_param_ref[i] = sb_in->geometric_param_ref[i];
			geometric_param_dif[i] = sb_in->geometric_param_dif[i];
			geometric_param[i] = new double[n_egrad_params[i]];
			for (j=0; j < n_egrad_params[i]; j++) {
				geometric_param[i][j] = sb_in->geometric_param[i][j];
			}
		}
		 angle_param_egrad = new bool[n_egrad_params[1]]; // keeps track of which parameters are angles, so they can be converted to degrees when displayed
		for (j=0; j < n_egrad_params[1]; j++) {
			angle_param_egrad[j] = sb_in->angle_param_egrad[j];
		}
		xi_initial_egrad = sb_in->xi_initial_egrad;
		xi_final_egrad = sb_in->xi_final_egrad;
		xi_ref_egrad = sb_in->xi_ref_egrad;
		if (egrad_mode==0) {
			bspline_order = sb_in->bspline_order;
			n_bspline_knots_tot = sb_in->n_bspline_knots_tot;
			for (i=0; i < 4; i++) {
				geometric_knots[i] = new double[n_bspline_knots_tot];
				for (j=0; j < n_bspline_knots_tot; j++) geometric_knots[i][j] = sb_in->geometric_knots[i][j];
			}
		}
		contours_overlap = sb_in->contours_overlap;
		set_egrad_ptr();
	}
	if (fourier_gradient) {
		n_fourier_grad_modes = sb_in->n_fourier_grad_modes;
		fourier_grad_mvals = fourier_mode_mvals.array(); // fourier_grad_mvals is used in the egrad functions (which are inherited by this class)
		n_fourier_grad_params = new int[n_fourier_modes];
		int i,j,k;
		for (i=0; i < n_fourier_grad_modes; i++) {
			n_fourier_grad_params[i] = sb_in->n_fourier_grad_params[i];
		}
		int n_amps = n_fourier_grad_modes*2;
		fourier_param = new double*[n_amps];
		for (i=0,k=0; i < n_fourier_grad_modes; i++, k+=2) {
			fourier_param[k] = new double[n_fourier_grad_params[i]];
			fourier_param[k+1] = new double[n_fourier_grad_params[i]];
			for (j=0; j < n_fourier_grad_params[i]; j++) {
				fourier_param[k][j] = sb_in->fourier_param[k][j];
			}
			for (j=0; j < n_fourier_grad_params[i]; j++) {
				fourier_param[k+1][j] = sb_in->fourier_param[k+1][j];
			}
		}
		if (egrad_mode==0) {
			fourier_knots = new double*[n_amps];
			for (i=0; i < n_amps; i++) {
				fourier_knots[i] = new double[n_bspline_knots_tot];
				for (j=0; j < n_bspline_knots_tot; j++) fourier_knots[i][j] = sb_in->fourier_knots[i][j];
			}
		}
	}
	copy_parameter_anchors(sb_in);
	assign_param_pointers();
}

/*
bool SB_Profile::spawn_lens_model(Alpha* lens_model)
{
	cout << "About to spawn..." << endl;
	lens_model = new Alpha();
	lens_model->initialize_parameters(1.2, 1, 0, 0.8, 30, 0.01, 0.01);
	cout << "Spawned qlens model..." << endl;
	cout << "LENS NAME from SB: " << lens_model->get_model_name() << endl;
	return true;
}
*/

void SB_Profile::set_nparams(const int &n_params_in, const bool resize)
{
	int old_nparams = (resize) ? n_params : 0;
	n_params = n_params_in;
	vary_params.resize(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.input(n_params);
	set_auto_penalty_limits.input(n_params);
	penalty_lower_limits.input(n_params);
	penalty_upper_limits.input(n_params);
	angle_param.input(n_params);
	for (int i=0; i < n_params; i++) angle_param[i] = false;

	if (param != NULL) delete[] param;
	if (anchor_parameter_to_source != NULL) delete[] anchor_parameter_to_source;
	if (parameter_anchor_source != NULL) delete[] parameter_anchor_source;
	if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
	if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
	if (parameter_anchor_exponent != NULL) delete[] parameter_anchor_exponent;

	anchor_parameter_to_source = new bool[n_params];
	parameter_anchor_source = new SB_Profile*[n_params];
	parameter_anchor_paramnum = new int[n_params];
	parameter_anchor_ratio = new double[n_params];
	parameter_anchor_exponent = new double[n_params];

	param = new double*[n_params];
	for (int i=0; i < n_params; i++) {
		anchor_parameter_to_source[i] = false;
		parameter_anchor_source[i] = NULL;
		parameter_anchor_paramnum[i] = -1;
		parameter_anchor_ratio[i] = 1.0;
		parameter_anchor_exponent[i] = 1.0;
	}

	if (n_params > old_nparams) {
		for (int i=old_nparams; i < n_params; i++) {
			vary_params[i] = false;
		}
	}
	n_vary_params = 0;
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]) n_vary_params++;
	}
}

void SB_Profile::anchor_center_to_lens(LensProfile** center_anchor_list, const int &center_anchor_lens_number)
{
	if (!center_anchored_to_lens) center_anchored_to_lens = true;
	center_anchor_lens = center_anchor_list[center_anchor_lens_number];
	x_center = center_anchor_lens->x_center;
	y_center = center_anchor_lens->y_center;
}

void SB_Profile::anchor_center_to_source(SB_Profile** center_anchor_list, const int &center_anchor_source_number)
{
	if (!center_anchored_to_source) center_anchored_to_source = true;
	center_anchor_source = center_anchor_list[center_anchor_source_number];
	x_center = center_anchor_source->x_center;
	y_center = center_anchor_source->y_center;
}

int SB_Profile::get_center_anchor_number() {
	if (center_anchored_to_lens) return center_anchor_lens->lens_number;
	else if (center_anchored_to_source) return center_anchor_source->sb_number;
	else return -1;
}


void SB_Profile::delete_center_anchor()
{
	if (center_anchored_to_lens) {
		center_anchored_to_lens = false;
		center_anchor_lens = NULL;
	} else if (center_anchored_to_source) {
		center_anchored_to_source = false;
		center_anchor_source = NULL;
	}
}

void SB_Profile::add_boxiness_parameter(const double c0_in, const bool vary_c0)
{
	// NOTE: this should be fixed up so that it comes before the Fourier modes, and the paramnames, stepsizes etc. should be automatically
	// assigned in the general functions just as the Fourier modes are. FIX!
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

	angle_param.resize(n_params);
	angle_param[n_params-1] = false;

	//delete[] param;
	//param = new double*[n_params];
	//assign_param_pointers();

	double **new_param = new double*[n_params];
	for (int i=0; i < n_params-1; i++) new_param[i] = param[i];
	new_param[n_params-1] = &c0;
	delete[] param;
	param = new_param;
	reset_anchor_lists();
	assign_param_pointers();
	assign_paramnames();
}

void SB_Profile::add_truncation_radius(const double rt_in, const bool vary_rt)
{
	// NOTE: this should be fixed up so that it comes before the Fourier modes, and the paramnames, stepsizes etc. should be automatically
	// assigned in the general functions just as the Fourier modes are. FIX!
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
	angle_param.resize(n_params);
	angle_param[n_params-1] = false;

	double **new_param = new double*[n_params];
	for (int i=0; i < n_params-1; i++) new_param[i] = param[i];
	new_param[n_params-1] = &rt;
	delete[] param;
	param = new_param;
	reset_anchor_lists();
	assign_param_pointers();
	assign_paramnames();

	//delete[] param;
	//param = new double*[n_params];
	//assign_param_pointers();
}

void SB_Profile::add_fourier_mode(const int m_in, const double amp_in, const double amp2_in, const bool vary1, const bool vary2)
{
	n_fourier_modes++;
	fourier_mode_mvals.resize(n_fourier_modes);
	fourier_mode_cosamp.resize(n_fourier_modes);
	fourier_mode_sinamp.resize(n_fourier_modes);
	fourier_mode_paramnum.resize(n_fourier_modes);
	fourier_mode_mvals[n_fourier_modes-1] = m_in;
	fourier_mode_cosamp[n_fourier_modes-1] = amp_in;
	fourier_mode_sinamp[n_fourier_modes-1] = amp2_in;
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
	angle_param.resize(n_params);
	angle_param[n_params-2] = false;
	angle_param[n_params-1] = false;
	if (vary1) n_vary_params++;
	if (vary2) n_vary_params++;

	vary_params[n_params-2] = vary1;
	vary_params[n_params-1] = vary2;

	delete[] param;
	param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
	assign_paramnames();
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
	angle_param.resize(n_params);
	for (int i=0; i < n_params; i++) angle_param[i] = false;
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

bool SB_Profile::enable_ellipticity_gradient(dvector& efunc_params, const int egrad_mode, const int n_bspline_coefs, const dvector& knots, const double ximin, const double ximax, const double xiref, const bool linear_xivals, const bool copy_vary_settings, boolvector* vary_egrad)
{
	if (ellipticity_mode==-1) return false; // ellipticity gradient only works for sources that have elliptical isophotes
	if (ellipticity_mode > 1) return false; // only emode=0 or 1 is supported right now
	
	if ((egrad_mode==0) and (efunc_params[0]==-1e30)) { // in this case, the egrad params were never initialized
		// Not sure if I should do this here, or before calling enable_ellipticity_gradient?
		efunc_params.input(2*n_bspline_coefs+2);
		for (int i=0; i < n_bspline_coefs; i++) efunc_params[i] = q;
		for (int i=n_bspline_coefs; i < 2*n_bspline_coefs; i++) efunc_params[i] = radians_to_degrees(theta);
		efunc_params[2*n_bspline_coefs] = x_center;
		efunc_params[2*n_bspline_coefs+1] = y_center;
	}

	int n_egrad_params;
	if (setup_egrad_params(egrad_mode,ellipticity_mode,efunc_params,n_egrad_params,n_bspline_coefs,knots,ximin,ximax,xiref,linear_xivals)==false) {
		warn("could not set up egrad params properly");
		return false;
	}
	int new_nparams = n_params + n_egrad_params - 4; // we already had q, theta, xc and yc
	if (n_egrad_params < 4) {
		warn("could not setup egrad params; less than four egrad parameters were created");
		return false;
	}

	vary_params.resize(new_nparams);
	paramnames.resize(new_nparams);
	latex_paramnames.resize(new_nparams);
	latex_param_subscripts.resize(new_nparams);
	stepsizes.resize(new_nparams);
	set_auto_penalty_limits.resize(new_nparams);
	penalty_lower_limits.resize(new_nparams);
	penalty_upper_limits.resize(new_nparams);
	angle_param.resize(new_nparams);
	for (int i=sbprofile_nparams+4; i < n_params; i++) {
		vary_params[i+n_egrad_params-4] = vary_params[i];
		angle_param[i+n_egrad_params-4] = angle_param[i];
	}
	for (int i=0; i < n_fourier_modes; i++) fourier_mode_paramnum[i] += n_egrad_params - 4; // so it keeps track of where the Fourier modes are
	int j=0;
	for (int i=sbprofile_nparams; i < sbprofile_nparams + n_egrad_params; i++) {
		if (!copy_vary_settings) vary_params[i] = false;
		else vary_params[i] = (*vary_egrad)[j++];
		angle_param[i] = false; // the angle params will be set when the param pointers are set
	}
	n_params = new_nparams;
	delete[] param;
	param = new double*[n_params];

	reset_anchor_lists();
	assign_param_pointers();
	assign_paramnames();

	if (qlens != NULL) qlens->ellipticity_gradient = true;
	check_for_overlapping_contours();
	if (contours_overlap) {
		warn("contours overlap for chosen ellipticity gradient parameters");
		if (qlens != NULL) {
			qlens->contours_overlap = true;
			qlens->contour_overlap_log_penalty_prior = overlap_log_penalty_prior;
		}
	}
	return true;
}

void SB_Profile::disable_ellipticity_gradient()
{
	int n_tot_egrad_params;
	disable_egrad_mode(n_tot_egrad_params);
	int n_extra_egrad_params = n_tot_egrad_params - 4; // since we'll still have four geometric params: q,theta,xc,yc
	boolvector new_vary(vary_params);
	for (int i=sbprofile_nparams; i < sbprofile_nparams + n_tot_egrad_params; i++) new_vary[i] = false;
	vary_parameters(new_vary);
	n_params -= n_extra_egrad_params;
	for (int i=sbprofile_nparams + 4; i < n_params; i++) {
		// Now we move all info about remaining parameters (center coordinates, Fourier modes) down and then resize the arrays
		vary_params[i] = vary_params[i+n_extra_egrad_params];
		angle_param[i] = angle_param[i+n_extra_egrad_params];
	}
	vary_params.resize(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.resize(n_params);
	set_auto_penalty_limits.resize(n_params);
	penalty_lower_limits.resize(n_params);
	penalty_upper_limits.resize(n_params);
	angle_param.resize(n_params);
	delete[] param;
	param = new double*[n_params];

	assign_param_pointers();
	assign_paramnames();
	update_ellipticity_meta_parameters();
}

void SB_Profile::reset_anchor_lists()
{
	if (anchor_parameter_to_source != NULL) delete[] anchor_parameter_to_source;
	if (parameter_anchor_source != NULL) delete[] parameter_anchor_source;
	if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
	if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
	if (parameter_anchor_exponent != NULL) delete[] parameter_anchor_exponent;

	anchor_parameter_to_source = new bool[n_params];
	parameter_anchor_source = new SB_Profile*[n_params];
	parameter_anchor_paramnum = new int[n_params];
	parameter_anchor_ratio = new double[n_params];
	parameter_anchor_exponent = new double[n_params];

	// parameters should not be anchored before enable egrad or adding Fourier modes, since the anchors are deleted here
	for (int i=0; i < n_params; i++) {
		anchor_parameter_to_source[i] = false;
		parameter_anchor_source[i] = NULL;
		parameter_anchor_paramnum[i] = -1;
		parameter_anchor_ratio[i] = 1.0;
		parameter_anchor_exponent[i] = 1.0;
	}
}

bool SB_Profile::enable_fourier_gradient(dvector& fourier_params, const dvector& knots, const bool copy_vary_settings, boolvector* vary_fgrad)
{
	if (ellipticity_mode==-1) return false; // Fourier gradient only works for sources that have elliptical isophotes
	if (ellipticity_mode > 1) return false; // only emode=0 or 1 is supported right now
	if (n_fourier_modes==0) return false; // Fourier modes must already be present
	if ((include_boxiness_parameter) or (include_truncation_radius)) return false; // not compatible with these parameters (unless you move them to the end)

	if ((egrad_mode==0) and (fourier_params[0]==-1e30)) { // in this case, the fgrad params were never initialized
		// Not sure if I should do this here, or before calling enable_fourier_gradient?
		int n_bspline_coefs = n_bspline_knots_tot - bspline_order - 1;
		fourier_params.input(2*n_fourier_modes*n_bspline_coefs);
		int i,j=0,k;
		for (k=0; k < n_fourier_modes; k++) {
			for (i=0; i < n_bspline_coefs; i++) fourier_params[j++] = fourier_mode_cosamp[k];
			for (i=0; i < n_bspline_coefs; i++) fourier_params[j++] = fourier_mode_sinamp[k];
		}
	}

	int n_fourier_grad_params;
	if (setup_fourier_grad_params(n_fourier_modes,fourier_mode_mvals,fourier_params,n_fourier_grad_params,knots)==false) return false;
	int param_ndif = n_fourier_grad_params - 2*n_fourier_modes; // we already had Am, Bm amplitudes as parameters
	int new_nparams = n_params + param_ndif; // we already had Am, Bm amplitudes as parameters
	int fourier_istart = fourier_mode_paramnum[0];

	vary_params.resize(new_nparams);
	paramnames.resize(new_nparams);
	latex_paramnames.resize(new_nparams);
	latex_param_subscripts.resize(new_nparams);
	stepsizes.resize(new_nparams);
	set_auto_penalty_limits.resize(new_nparams);
	penalty_lower_limits.resize(new_nparams);
	penalty_upper_limits.resize(new_nparams);
	angle_param.resize(new_nparams);
	// The next part is only relevant if there are parameters after the Fourier modes
	for (int i=fourier_istart+2*n_fourier_modes; i < n_params; i++) {
		vary_params[i+param_ndif] = vary_params[i];
		angle_param[i+param_ndif] = angle_param[i];
	}
	int j=0;
	for (int i=fourier_istart; i < fourier_istart + n_fourier_grad_params; i++) {
		if (!copy_vary_settings) vary_params[i] = false;
		else vary_params[i] = (*vary_fgrad)[j++];
		angle_param[i] = false; // the angle params will be set when the param pointers are set
	}
	set_fourier_paramnums(fourier_mode_paramnum.array(),fourier_istart);
	//for (int i=0; i < n_fourier_modes; i++) cout << "fmode(" << i << ") start: " << fourier_mode_paramnum[i] << endl;
	//die();
	n_params = new_nparams;
	delete[] param;
	param = new double*[n_params];

	reset_anchor_lists();
	assign_param_pointers();
	assign_paramnames();

	if (qlens != NULL) qlens->ellipticity_gradient = true;
	check_for_overlapping_contours();
	if (contours_overlap) {
		warn("contours overlap for chosen ellipticity gradient parameters");
		if (qlens != NULL) {
			qlens->contours_overlap = true;
			qlens->contour_overlap_log_penalty_prior = overlap_log_penalty_prior;
		}
	}

	return true;
}

void SB_Profile::assign_param_pointers()
{
	param[0] = &qx_parameter;
	param[1] = &f_parameter;
	set_geometric_param_pointers(sbprofile_nparams);
}

void SB_Profile::set_geometric_param_pointers(int qi)
{
	// Sets parameter pointers for ellipticity (or axis ratio) and angle
	if (!ellipticity_gradient) {
		if (use_sb_ellipticity_components) {
			param[qi++] = &epsilon1;
			param[qi] = &epsilon2;
			angle_param[qi++] = false;
			angle_param_exists = false; // there is no angle parameter if ellipticity components are being used
		} else {
			param[qi++] = &q;
			param[qi] = &theta;
			angle_param[qi++] = true;
			angle_param_exists = true;
		}
		if (!lensed_center_coords) {
			param[qi++] = &x_center;
			param[qi++] = &y_center;
		} else {
			param[qi++] = &x_center_lensed;
			param[qi++] = &y_center_lensed;
		}

	} else {
		angle_param_exists = true;
		set_geometric_param_pointers_egrad(param,angle_param,qi); // NOTE: if fourier_gradient is turned on, the Fourier parameter pointers are also set in this function
	}

	if ((!fourier_gradient) and (n_fourier_modes > 0)) {
		for (int i=0; i < n_fourier_modes; i++) {
			param[qi++] = &fourier_mode_cosamp[i];
			param[qi++] = &fourier_mode_sinamp[i];
		}
	}
	if (include_boxiness_parameter) param[qi++] = &c0;
	if (include_truncation_radius) param[qi++] = &rt;
}

bool SB_Profile::vary_parameters(const boolvector& vary_params_in)
{
	if (vary_params_in.size() != n_params) {
		warn("vary params doesn't have the right size: %i vs %i",vary_params_in.size(),n_params);
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

void SB_Profile::get_vary_flags(boolvector& vary_flags)
{
	vary_flags.input(n_params);
	for (int i=0; i < n_params; i++) {
		vary_flags[i] = vary_params[i];
	}
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
		if (angle_param[i]) params[i] = radians_to_degrees(*(param[i]));
		else params[i] = *(param[i]);
	}
}

bool SB_Profile::get_specific_parameter(const string name_in, double& value)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if (paramnames[i]==name_in) {
			found_match = true;
			value = *(param)[i];
			break;
		}
	}
	return found_match;
}

void SB_Profile::update_parameters(const double* params)
{
	for (int i=0; i < n_params; i++) {
		if (angle_param[i]) *(param[i]) = degrees_to_radians(params[i]);
		else *(param[i]) = params[i];
	}
	update_meta_parameters();
	if (qlens != NULL) qlens->update_anchored_parameters_and_redshift_data();
	if (lensed_center_coords) set_center_if_lensed_coords();
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
	else {
		if ((sbtype==SHAPELET) and (name_in=="n")) {
			update_indxptr(value);
			found_match = true;
		}
	}
	delete[] newparams;
	return found_match;
}

bool SB_Profile::update_specific_parameter(const int paramnum, const double& value)
{
	if (paramnum >= n_params) return false;
	double* newparams = new double[n_params];
	get_parameters(newparams);
	for (int i=0; i < n_params; i++) {
		if (i==paramnum) {
			newparams[i] = value;
			break;
		}
	}
	update_parameters(newparams);
	delete[] newparams;
	return true;
}

void SB_Profile::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		for (int i=0; i < n_params; i++) {
			if (vary_params[i]==true) {
				if (angle_param[i]) {
					*(param[i]) = degrees_to_radians(fitparams[index++]);
					update_angle_meta_params();
				}
				else *(param[i]) = fitparams[index++];
			}
		}
		update_meta_parameters();
	}
}

void SB_Profile::update_anchored_parameters()
{
	bool at_least_one_param_anchored = false;
	for (int i=0; i < n_params; i++) {
		if (anchor_parameter_to_source[i]) {
			(*param[i]) = parameter_anchor_ratio[i]*pow(*(parameter_anchor_source[i]->param[parameter_anchor_paramnum[i]]),parameter_anchor_exponent[i]);
			if (at_least_one_param_anchored==false) at_least_one_param_anchored = true;
		}
	}
	if (at_least_one_param_anchored) {
		update_meta_parameters();
	}
}

bool SB_Profile::update_anchored_parameters_to_source(const int src_i)
{
	bool at_least_one_param_anchored = false;
	for (int i=0; i < n_params; i++) {
		if ((anchor_parameter_to_source[i]) and (parameter_anchor_source[i]->sb_number==src_i)) {
			(*param[i]) = parameter_anchor_ratio[i]*pow(*(parameter_anchor_source[i]->param[parameter_anchor_paramnum[i]]),parameter_anchor_exponent[i]);
			if (at_least_one_param_anchored==false) at_least_one_param_anchored = true;
		}
	}
	if (at_least_one_param_anchored) {
		update_meta_parameters();
	}
	return at_least_one_param_anchored;
}

void SB_Profile::update_anchor_center()
{
	if (center_anchored_to_lens) {
		x_center = center_anchor_lens->x_center;
		y_center = center_anchor_lens->y_center;
	} else if (center_anchored_to_source) {
		x_center = center_anchor_source->x_center;
		y_center = center_anchor_source->y_center;
	}
}

void SB_Profile::get_fit_parameters(dvector& fitparams, int &index)
{
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]==true) {
			if (angle_param[i]) fitparams[index++] = radians_to_degrees(*(param[i]));
			else fitparams[index++] = *(param[i]);
		}
	}
}

void SB_Profile::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1;
	stepsizes[index++] = 0.1;
	set_geometric_param_auto_stepsizes(index);
}

void SB_Profile::set_geometric_param_auto_stepsizes(int &index)
{
	if (!ellipticity_gradient) {
		if (use_sb_ellipticity_components) {
			stepsizes[index++] = 0.1; // e1
			stepsizes[index++] = 0.1; // e2
		} else {
			stepsizes[index++] = 0.1; // q
			stepsizes[index++] = 20;  // angle stepsize
		}
		stepsizes[index++] = 0.1; // xc
		stepsizes[index++] = 0.1; // yc
	} else {
		set_geometric_stepsizes_egrad(stepsizes,index);
	}
	if ((!fourier_gradient) and (n_fourier_modes > 0)) {
		for (int i=0; i < n_fourier_modes; i++) {
			stepsizes[index++] = 0.005;
			stepsizes[index++] = 0.005;
		}
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
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

void SB_Profile::set_geometric_param_auto_ranges(int param_i)
{
	if (!ellipticity_gradient) {
		if (use_sb_ellipticity_components) {
			set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = -1; penalty_upper_limits[param_i] = 1; param_i++;
			set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = -1; penalty_upper_limits[param_i] = 1; param_i++;
		} else {
			set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 5e-3; penalty_upper_limits[param_i] = 1; param_i++;
			set_auto_penalty_limits[param_i++] = false;
		}
		set_auto_penalty_limits[param_i++] = false;
		set_auto_penalty_limits[param_i++] = false;
	} else {
		set_geometric_param_ranges_egrad(set_auto_penalty_limits, penalty_lower_limits, penalty_upper_limits, param_i);
	}
	if ((!fourier_gradient) and (n_fourier_modes > 0)) {
		for (int i=0; i < n_fourier_modes; i++) {
			set_auto_penalty_limits[param_i++] = false;
			set_auto_penalty_limits[param_i++] = false;
		}
	}
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

void SB_Profile::get_fit_parameter_names(vector<string>& paramnames_vary, vector<string> *latex_paramnames_vary, vector<string> *latex_subscripts_vary, const bool include_suffix)
{
	int i;
	string suffix = "";
	if (include_suffix) {
		if (is_lensed) suffix = "_src";
		else suffix = "_fg";
	}
	for (i=0; i < n_params; i++) {
		if (vary_params[i]) {
			paramnames_vary.push_back(paramnames[i] + suffix);
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

bool SB_Profile::get_limits(dvector& lower, dvector& upper, int &index)
{
	if ((include_limits==false) or (lower_limits.size() != n_vary_params)) return false;
	for (int i=0; i < n_vary_params; i++) {
		lower[index] = lower_limits[i];
		upper[index] = upper_limits[i];
		index++;
	}
	return true;
}

bool SB_Profile::get_limits(dvector& lower, dvector& upper)
{
	if (include_limits==false) return false;
	lower.input(n_vary_params);
	upper.input(n_vary_params);
	for (int i=0; i < n_vary_params; i++) {
		lower[i] = lower_limits[i];
		upper[i] = upper_limits[i];
	}
	return true;
}

void SB_Profile::copy_parameter_anchors(const SB_Profile* sb_in)
{
	// n_params *must* already be set before running this
	anchor_parameter_to_source = new bool[n_params];
	parameter_anchor_source = new SB_Profile*[n_params];
	parameter_anchor_paramnum = new int[n_params];
	parameter_anchor_ratio = new double[n_params];
	parameter_anchor_exponent = new double[n_params];
	for (int i=0; i < n_params; i++) {
		anchor_parameter_to_source[i] = sb_in->anchor_parameter_to_source[i];
		parameter_anchor_source[i] = sb_in->parameter_anchor_source[i];
		parameter_anchor_paramnum[i] = sb_in->parameter_anchor_paramnum[i];
		parameter_anchor_ratio[i] = sb_in->parameter_anchor_ratio[i];
		parameter_anchor_exponent[i] = sb_in->parameter_anchor_exponent[i];
	}
}

void SB_Profile::assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, SB_Profile* param_anchor_source)
{
	if (paramnum >= n_params) die("Parameter does not exist for this source");
	if (anchor_paramnum >= param_anchor_source->n_params) die("Parameter does not exist for source you are anchoring to");
	anchor_parameter_to_source[paramnum] = true;
	parameter_anchor_source[paramnum] = param_anchor_source;
	//(*param_anchor_source->param[0]) = 88;
	parameter_anchor_paramnum[paramnum] = anchor_paramnum;
	if ((!use_implicit_ratio) and (!use_exponent)) {
		parameter_anchor_ratio[paramnum] = 1.0;
		(*param[paramnum]) = *(param_anchor_source->param[anchor_paramnum]);
	}
	else if (use_implicit_ratio) {
		parameter_anchor_exponent[paramnum] = 1.0;
		if ((*(param_anchor_source->param[anchor_paramnum]))==0) {
			if (*param[paramnum]==0) parameter_anchor_ratio[paramnum] = 1.0;
			else die("cannot anchor to parameter with specified ratio if parameter is equal to zero");
		} else {
			parameter_anchor_ratio[paramnum] = (*param[paramnum]) / (*(param_anchor_source->param[anchor_paramnum]));
		}
	}
	else if (use_exponent) {
		parameter_anchor_ratio[paramnum] = ratio;
		parameter_anchor_exponent[paramnum] = exponent;
	}
	update_anchored_parameters();
}

void SB_Profile::unanchor_parameter(SB_Profile* param_anchor_source)
{
	// if any parameters are anchored to the source in question, unanchor them (use this when you are deleting a source, in case others are anchored to it)
	for (int i=0; i < n_params; i++) {
		if ((anchor_parameter_to_source[i]) and (parameter_anchor_source[i] == param_anchor_source)) {
			parameter_anchor_source[i] = NULL;
			anchor_parameter_to_source[i] = false;
			parameter_anchor_paramnum[i] = -1;
			parameter_anchor_ratio[i] = 1.0;
			parameter_anchor_exponent[i] = 1.0;
		}
	}
}

void SB_Profile::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "qx"; latex_paramnames[0] = "q"; latex_param_subscripts[0] = "x";
	paramnames[1] = "f"; latex_paramnames[1] = "f"; latex_param_subscripts[1] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

void SB_Profile::set_geometric_paramnames(int qi)
{
	string suffix;
	if (is_lensed) suffix = "src";
	else suffix = "fg";
	if (!ellipticity_gradient) {
		if (use_sb_ellipticity_components) {
			paramnames[qi] = "e1"; latex_paramnames[qi] = "e"; latex_param_subscripts[qi] = "1," + suffix; qi++;
			paramnames[qi] = "e2"; latex_paramnames[qi] = "e"; latex_param_subscripts[qi] = "2," + suffix; qi++;
		} else {
			paramnames[qi] = "q"; latex_paramnames[qi] = "q"; latex_param_subscripts[qi] = suffix; qi++;
			paramnames[qi] = "theta"; latex_paramnames[qi] = "\\theta"; latex_param_subscripts[qi] = suffix; qi++;
		}
		paramnames[qi] = "xc"; latex_paramnames[qi] = "x"; latex_param_subscripts[qi] = "c," + suffix;
		if (lensed_center_coords) {
			paramnames[qi] += "_l";
			latex_param_subscripts[qi] += ",l";
		}
		qi++;
		paramnames[qi] = "yc"; latex_paramnames[qi] = "y"; latex_param_subscripts[qi] = "c," + suffix;
		if (lensed_center_coords) {
			paramnames[qi] += "_l";
			latex_param_subscripts[qi] += ",l";
		}
		qi++;
	} else {
		set_geometric_paramnames_egrad(paramnames, latex_paramnames, latex_param_subscripts, qi, ("," + suffix));
	}
	if ((!fourier_gradient) and (n_fourier_modes > 0)) {
		for (int i=0; i < n_fourier_modes; i++) {
			stringstream mstream;
			string mstring;
			mstream << fourier_mode_mvals[i];
			mstream >> mstring;
			paramnames[qi] = "A_" + mstring; latex_paramnames[qi] = "A"; latex_param_subscripts[qi] = mstring; qi++;
			paramnames[qi] = "B_" + mstring; latex_paramnames[qi] = "B"; latex_param_subscripts[qi] = mstring; qi++;
		}
	}
}

void SB_Profile::set_geometric_parameters(const double &q1_in, const double &q2_in, const double &xc_in, const double &yc_in)
{
	qx_parameter = 1.0;

	if (use_sb_ellipticity_components) {
		epsilon1 = q1_in;
		epsilon2 = q2_in;
	} else {
		q = q1_in;
		if (q < 0) q = -q; // don't allow negative axis ratios
		if (q > 1) q = 1.0; // don't allow q>1
		theta = degrees_to_radians(q2_in);
	}
	if (!lensed_center_coords) {
		x_center = xc_in;
		y_center = yc_in;
	} else {
		x_center_lensed = xc_in;
		y_center_lensed = yc_in;
		set_center_if_lensed_coords();
	}

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

void SB_Profile::set_center_if_lensed_coords()
{
	if (lensed_center_coords) {
		if (qlens==NULL) die("Cannot use lensed center coordinates if pointer to QLens object hasn't been assigned");
		lensvector xl;
		xl[0] = x_center_lensed;
		xl[1] = y_center_lensed;
		qlens->find_sourcept(xl,x_center,y_center,0,qlens->reference_zfactors,qlens->default_zsrc_beta_factors);
	}
}

void SB_Profile::calculate_ellipticity_components()
{
	if (use_sb_ellipticity_components) {
		double theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		epsilon1 = (1-q)*cos(2*theta_eff);
		epsilon2 = (1-q)*sin(2*theta_eff);
	}
}

void SB_Profile::update_ellipticity_meta_parameters()
{
	if (!ellipticity_gradient) {
		if (use_sb_ellipticity_components) {
			q = 1 - sqrt(SQR(epsilon1) + SQR(epsilon2));
			set_angle_from_components(epsilon1,epsilon2); // note this will automatically set the costheta, sintheta parameters
		} else {
			update_angle_meta_params(); // sets the costheta, sintheta meta-parameters
		}
	} else {
		//q = efunc_qi; // q shouldn't be used at all, but this is just in case ellipticity gradient is turned off
		//theta = efunc_theta_i; // theta shouldn't be used at all, but this is just in case ellipticity gradient is turned off
		q = geometric_param[0][0];
		if (q > 1.0) q = 1.0;
		theta = geometric_param[1][0];
		x_center = geometric_param[2][0];
		y_center = geometric_param[3][0];
		update_egrad_meta_parameters();
		check_for_overlapping_contours();
		if (qlens != NULL) {
			if (contours_overlap) {
				qlens->contours_overlap = true;
				qlens->contour_overlap_log_penalty_prior = overlap_log_penalty_prior;
			} else {
				qlens->contours_overlap = false;
			}
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
	// do this if the major axis orientation is changed (so the qlens angles values are changed appropriately, even though the qlens doesn't change)
	theta += M_HALFPI;
	while (theta > M_PI) theta -= M_PI;
}

void SB_Profile::shift_angle_minus_90()
{
	// do this if the major axis orientation is changed (so the qlens angles values are changed appropriately, even though the qlens doesn't change)
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
		else if (comp2==0) angle = 0.0;
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
	if ((!ellipticity_gradient) and (theta != 0)) rotate(x,y);

	double xisq, rsq;
	double fourier_factor = 0.0;

	rsq = x*x + y*y;
	if (!ellipticity_gradient) {
		double rsq_ell = x*x + y*y/(q*q);
		if (ellipticity_mode==1) rsq_ell *= q;
		if ((include_boxiness_parameter) and (c0 != 0.0)) {
			if (ellipticity_mode==0)
				xisq = pow(pow(abs(x),c0+2.0) + pow(abs(y/q),c0+2.0),2.0/(c0+2.0));
			else
				xisq = pow(pow(abs(x)*sqrt(q),c0+2.0) + pow(abs(y/sqrt(q)),c0+2.0),2.0/(c0+2.0));
		} else {
			xisq = rsq_ell;
		}
		if (n_fourier_modes > 0) {
			double phi_q; // used for Fourier modes
			if (fourier_use_eccentric_anomaly) phi_q = atan(y/(q*x));
			else phi_q = atan(y/x); // for SB perturbations, we don't use eccentric anomaly as our angle because the corresponding lensing multipoles can't either
				// Check the angle below!!!! Shouldn't you use angle c.c. from x-axis here? (comp to ellipse sampling angle in pixelgrid.cpp)
			if (x < 0) phi_q += M_PI;
			else if (y < 0) phi_q += M_2PI;

			if (!fourier_sb_perturbation) fourier_factor = 1.0;
			if (use_fmode_scaled_amplitudes) {
				for (int i=0; i < n_fourier_modes; i++) {
					fourier_factor += (fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*phi_q) + fourier_mode_sinamp[i]*sin(fourier_mode_mvals[i]*phi_q))/fourier_mode_mvals[i];
				}
			} else {
				for (int i=0; i < n_fourier_modes; i++) {
					fourier_factor += fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*phi_q) + fourier_mode_sinamp[i]*sin(fourier_mode_mvals[i]*phi_q);
				}
			}
			if (!fourier_sb_perturbation) xisq *= fourier_factor*fourier_factor;
		}
	} else {
		double xi = elliptical_radius_root(x,y);
		xisq = SQR(xi);
		if ((n_fourier_modes > 0) and (fourier_sb_perturbation)) {
			double ep, phi0;
			if (fourier_use_eccentric_anomaly) ellipticity_function(xi,ep,phi0);
			else ellipticity_function(sqrt(rsq),ep,phi0); // lensing multipoles depend on r, not xi, so we follow the same restriction here

			double costh, sinth, xp, yp, qq, phi_q;
			costh = cos(phi0);
			sinth = sin(phi0);
			xp = x*costh + y*sinth;
			yp = -x*sinth + y*costh;
			qq = sqrt(1-ep);

			if (fourier_use_eccentric_anomaly) phi_q = atan(yp/(qq*xp));
			else phi_q = atan(yp/xp);
			if (xp < 0) phi_q += M_PI;
			else if (yp < 0) phi_q += M_2PI;

			double *cosamps;
			double *sinamps;
			if (fourier_gradient) {
				cosamps = new double[n_fourier_modes];
				sinamps = new double[n_fourier_modes];
				if (fourier_use_eccentric_anomaly) fourier_mode_function(xi,cosamps,sinamps);
				else fourier_mode_function(sqrt(rsq),cosamps,sinamps); // lensing multipoles depend on r, not xi, so we follow the same restriction here
			} else {
				// No need to create new arrays, just have them point to fourier_mode_cosamp and fourier_mode_sinamp
				cosamps = fourier_mode_cosamp.array();
				sinamps = fourier_mode_sinamp.array();
			}

			//if (use_fmode_scaled_amplitudes) {
				//for (int i=0; i < n_fourier_modes; i++) {
					//fourier_factor += (cosamps[i]*cos(fourier_mode_mvals[i]*phi_q) + sinamps[i]*sin(fourier_mode_mvals[i]*phi_q))/fourier_mode_mvals[i];
				//}
			//} else {
				for (int i=0; i < n_fourier_modes; i++) {
					fourier_factor += cosamps[i]*cos(fourier_mode_mvals[i]*phi_q) + sinamps[i]*sin(fourier_mode_mvals[i]*phi_q);
				}
			//}
			if (fourier_gradient) {
				delete[] cosamps;
				delete[] sinamps;
			}
		}
	}
	double sb = sb_rsq(xisq);
	if ((n_fourier_modes > 0) and (fourier_sb_perturbation)) {
		// create virtual sb_rsq_deriv function in base class, and versions for all inherited classes, so you don't have to do this numerically for analytic models
		double sbderiv, h = 1e-5;

		if (fourier_use_eccentric_anomaly) {
			// we evaluate SB at non-elliptical radius because that's what the corresponding lensing multipoles have to do (to get deflections).
			if (xisq <= h) sbderiv = (sb_rsq(xisq + h) - sb_rsq(xisq))/(h);
			else sbderiv = (sb_rsq(xisq + h) - sb_rsq(xisq-h))/(2*h);
			sb += 2*fourier_factor*sbderiv*xisq; // this allows it to approximate perturbing the elliptical radius (via first order term in Taylor expansion in (r + dr))
		} else {
			if (rsq <= h) sbderiv = (sb_rsq(rsq + h) - sb_rsq(rsq))/(h);
			else sbderiv = (sb_rsq(rsq + h) - sb_rsq(rsq-h))/(2*h);
			sb += 2*fourier_factor*sbderiv*rsq; // this allows it to approximate perturbing the elliptical radius (via first order term in Taylor expansion in (r + dr))
		}
		//sb += fourier_factor; // for testing purposes
	}
	if (include_truncation_radius) sb *= pow(1+pow(xisq/(rt*rt),3),-2);

	if (sb*0.0 != 0.0) warn("surface brightness returning NAN");

	return sb;
}

double SB_Profile::surface_brightness_zoom(lensvector &centerpt, lensvector &pt1, lensvector &pt2, lensvector &pt3, lensvector &pt4)
{
	bool subgrid = false;
	bool contains_sbcenter = false;
	int xsplit, ysplit;

	lensvector sbcenter;
	sbcenter[0] = x_center;
	sbcenter[1] = y_center;

	lensvector d1, d2, d3;
	double product1, product2, product3;
	double r[4];
	d1[0] = sbcenter[0]-pt1[0];
	d1[1] = sbcenter[1]-pt1[1];
	r[0] = d1.norm();
	d2[0] = sbcenter[0]-pt2[0];
	d2[1] = sbcenter[1]-pt2[1];
	r[1] = d2.norm();
	d3[0] = sbcenter[0]-pt3[0];
	d3[1] = sbcenter[1]-pt3[1];
	r[2] = d3.norm();

	// Let's see if the cell contains the SB center inside it
	product1 = d1 ^ d2;
	product2 = d3 ^ d1;
	product3 = d2 ^ d3;
	if ((product1 > 0) and (product2 > 0) and (product3 > 0)) contains_sbcenter = true;
	else if ((product1 < 0) and (product2 < 0) and (product3 < 0)) contains_sbcenter = true;
	else {
		d3[0] = sbcenter[0]-pt3[0];
		d3[1] = sbcenter[1]-pt3[1];
		r[3] = d3.norm();
		product2 = d3 ^ d1;
		product3 = d2 ^ d3;
		if ((product1 > 0) and (product2 > 0) and (product3 > 0)) contains_sbcenter = true;
		else if ((product1 < 0) and (product2 < 0) and (product3 < 0)) contains_sbcenter = true;
	}
	double rmin = 1e30, rmax = -1e30;
	for (int i=0; i < 4; i++) {
		if (r[i] > rmax) rmax = r[i];
		if (r[i] < rmin) rmin = r[i];
	}

	if (contains_sbcenter) rmin = 0;
	//d1[0] = centerpt[0]-sbcenter[0];
	//d1[1] = centerpt[1]-sbcenter[1];
	//double center_r = d1.norm();

	if ((rmax-rmin) < length_scale()) {
		const int max_split = 20;
		if (contains_sbcenter) {
			// If the pixel contains the SB center, just split the max number of times and call it a day
			subgrid = true;
			xsplit = max_split;
			ysplit = max_split;
		} else {
			// If pixels are smaller than the half-light radius, we will estimate the curvature and use this to find the optimal subpixel scale that makes the
			// error in the estimated (integrated) SB over the subpixel smaller than the epsilon/6 of the pixel noise; we use this to determine # of splittings
			const double epsilon = 1; // in principle we could allow the user to change this to make the subgridding more aggressive. Implement later?
			double rminsq=rmin*rmin, rmaxsq=rmax*rmax;
			double sbderiv1, sbderiv2, h = 1e-5;
			if (rminsq <= h) sbderiv1 = 2*rmin*(sb_rsq(rminsq + h) - sb_rsq(rminsq))/(h);
			else sbderiv1 = 2*rmin*(sb_rsq(rminsq + h) - sb_rsq(rminsq-h))/(2*h);
			if (rmaxsq <= h) sbderiv2 = 2*rmax*(sb_rsq(rmaxsq + h) - sb_rsq(rmaxsq))/(h);
			else sbderiv2 = 2*rmax*(sb_rsq(rmaxsq + h) - sb_rsq(rmaxsq-h))/(2*h);
			double sbcurv_approx = sbderiv2-sbderiv1;
			double optimal_scale = 4*epsilon*SB_noise/sbcurv_approx;
			// Ideally, if the pixel size is greater than optimal scale, you'd increase the splittings, calculate sbcurv_approx again, and iterate until
			// pixel size is small enough. But this seems to work well enough as it is.
			double npix_approx = (rmax-rmin)/optimal_scale;
			if (npix_approx > 1) {
				subgrid = true;
				xsplit = ((int) npix_approx) + 1;
				ysplit = ((int) npix_approx) + 1;
				if (xsplit > max_split) {
					//cout << "xsplit wants to be " << xsplit << endl;
					xsplit = max_split; // limit on number of splittings
				}
				if (ysplit > max_split) {
					//cout << "ysplit wants to be " << ysplit << endl;
					ysplit = max_split; // limit on number of splittings
				}

				//double xavg = (pt1[0] + pt2[0] + pt3[0] + pt4[0])/4;
				//double yavg = (pt1[1] + pt2[1] + pt3[1] + pt4[1])/4;
				//cout << "xsplit= " << xsplit << " ysplit=" << ysplit << " (x=" << xavg << ",y=" << yavg << ",maxsplit=" << max_split << ")" << endl;
			}
			//if (npix_approx > 1) cout << "r= " << center_r << " sbcurv_approx=" << sbcurv_approx << " delta_R=" << (rmax-rmin) << " OPTIMAL SCALE: " << optimal_scale << " npix_approx=" << npix_approx << endl;
		}
	} else {
		// The following algorithm is for source pixels that are large compared to the half-light radius of the SB profile
		// Revisit this later? Seems a bit shoddy, and not very trustworthy for lensed sources, but maybe good enough for unlensed sources.
		double scale = zoom_scale*length_scale();
		lensvector scpt1, scpt2, scpt3, scpt4;
		d1[0] = pt1[0]-centerpt[0];
		d1[1] = pt1[1]-centerpt[1];
		d2[0] = d1[0] * (scale/d1.norm()); // vector along d1 with length given by scale
		d2[1] = d1[1] * (scale/d1.norm()); // vector along d1 with length given by scale
		scpt1 = pt1 + d2;

		d1[0] = pt2[0]-centerpt[0];
		d1[1] = pt2[1]-centerpt[1];
		d2[0] = d1[0] * (scale/d1.norm()); // vector along d1 with length given by scale
		d2[1] = d1[1] * (scale/d1.norm()); // vector along d1 with length given by scale
		scpt2 = pt2 + d2;

		d1[0] = pt3[0]-centerpt[0];
		d1[1] = pt3[1]-centerpt[1];
		d2[0] = d1[0] * (scale/d1.norm()); // vector along d1 with length given by scale
		d2[1] = d1[1] * (scale/d1.norm()); // vector along d1 with length given by scale
		scpt3 = pt3 + d2;

		d1[0] = pt4[0]-centerpt[0];
		d1[1] = pt4[1]-centerpt[1];
		d2[0] = d1[0] * (scale/d1.norm()); // vector along d1 with length given by scale
		d2[1] = d1[1] * (scale/d1.norm()); // vector along d1 with length given by scale
		scpt4 = pt4 + d2;

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

		// Now find the middle point of each side, and see if it is close enough to sbcenter. This could still fail if the
		// cell is very large, and sbcenter is only close to some off-center point on one of the edges, but hopefully it's
		// enough to get the job done.
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

		if (!subgrid) {
			double pixel_xlength, pixel_ylength;
			d1 = pt2 - pt1;
			d2 = pt4 - pt3;
			pixel_xlength = dmax(d1.norm(),d2.norm());
			d1 = pt3 - pt1;
			d2 = pt4 - pt2;
			pixel_ylength = dmax(d1.norm(),d2.norm());
			double lscale = 0.01*length_scale();
			xsplit = ((int) (zoom_split_factor*pixel_xlength/lscale)) + 1;
			ysplit = ((int) (zoom_split_factor*pixel_ylength/lscale)) + 1;
		}
	}
	if (!subgrid) return surface_brightness(centerpt[0],centerpt[1]);
	double sb = 0;
	double u0, w0, xs, ys;
	int ii,jj;

	// This splitting algorithm allows for the 'pixel' to not be rectangular, as in lensed sources. But it's still dicey for lensed 
	// sources since the subpixels in the source plane won't look exactly the same as splitting in the image plane and mapping each
	// subpixel to the source plane. It's always better (but costlier) to split the image pixels and then ray trace the subpixels to
	// the source plane.
	for (ii=0; ii < xsplit; ii++) {
		u0 = ((double) (1+2*ii))/(2*xsplit);
		for (jj=0; jj < ysplit; jj++) {
			w0 = ((double) (1+2*jj))/(2*ysplit);
			xs = (pt1[0]*u0 + pt2[0]*(1-u0))*w0 + (pt3[0]*u0 + pt4[0]*(1-u0))*(1-w0);
			ys = (pt1[1]*u0 + pt2[1]*(1-u0))*w0 + (pt3[1]*u0 + pt4[1]*(1-u0))*(1-w0);
			sb += surface_brightness(xs,ys);
		}
	}
	sb /= (xsplit*ysplit);

	return sb;
}

double SB_Profile::surface_brightness_r(const double r)
{
	return sb_rsq(r*r);
}

bool SB_Profile::fit_sbprofile_data(IsophoteData& isophote_data, const int fit_mode, const int n_livepts, const int mpi_np, const int mpi_id, const string fit_output_dir)
{
	// nested sampling: fitmode = 0
	// downhill simplex: fitmode = 1 or higher
	for (int i=0; i < sbprofile_nparams; i++) {
		if (!vary_params[i]) {
			if (mpi_id==0) warn("all sbprofile parameters must be allowed to vary (param %i set fixed)",i);
			return false;
		}
		if ((fit_mode<=0) and (lower_limits.size() != n_vary_params)) {
			if (mpi_id==0) warn("lower/upper prior limits have not been set for sbprofile parameters (limit size=%i, nvary=%i)",lower_limits.size(),n_vary_params);
			return false;
		}
	}
	if (sbprofile_nparams == 0) {
		if (mpi_id==0) warn("Not an ellipical sbprofile object; cannot do sbprofile fit");
		return false;
	}
	n_isophote_datapts = isophote_data.n_xivals;
	profile_fit_xivals = isophote_data.xivals;
	sbprofile_data = isophote_data.sb_avg_vals;
	sbprofile_errors = isophote_data.sb_errs;
	double *fitparams = new double[sbprofile_nparams];
	double *param_errors = new double[sbprofile_nparams];

	set_auto_ranges(); // this is so it can give a penalty prior if a parameter takes an absurd value
	if (fit_mode<=0) {
#ifdef USE_MPI
		Set_MCMC_MPI(mpi_np,mpi_id);
#endif
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&SB_Profile::sbprofile_loglike);
		InputPoint(fitparams,lower_limits.array(),upper_limits.array(),sbprofile_nparams);
		double lnZ;
		string filename = fit_output_dir + "/" + "sbprofile";

		string pnumfile_str = filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << sbprofile_nparams << " " << 0 << endl;
		pnumfile.close();

		string pnamefile_str = filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (int i=0; i < sbprofile_nparams; i++) {
			pnamefile << paramnames[i] << endl;
		}
		pnamefile.close();

		string prange_str = filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (int i=0; i < sbprofile_nparams; i++) {
			prangefile << lower_limits[i] << " " << upper_limits[i] << endl;
		}
		prangefile.close();

		MonoSample(filename.c_str(),n_livepts,lnZ,fitparams,param_errors,false);
		double chisq_bestfit = 2*(this->*LogLikePtr)(fitparams);
	} else {
		double (Simplex::*loglikeptr)(double*);
		loglikeptr = static_cast<double (Simplex::*)(double*)> (&SB_Profile::sbprofile_loglike);
		double *stepsizes = new double [sbprofile_nparams];
		for (int i=0; i < sbprofile_nparams; i++) {
			stepsizes[i] = 0.1; // arbitrary
			fitparams[i] = *(param[i]);
		}
		double chisq_tolerance = 1e-4;
		if (qlens != NULL) chisq_tolerance = qlens->chisq_tolerance;
		initialize_simplex(fitparams,sbprofile_nparams,stepsizes,chisq_tolerance);
		simplex_set_display_bfpont(true);
		simplex_set_function(loglikeptr);
		int it, nrep=1;
		while (nrep-- >= 0) {
			it=0;
			downhill_simplex(it,10000,0); // do final run with zero temperature
		}
		delete[] stepsizes;
	}

	delete[] fitparams;
	delete[] param_errors;
	profile_fit_xivals = NULL;
	sbprofile_data = NULL;
	sbprofile_errors = NULL;
	return true;
}

double SB_Profile::sbprofile_loglike(double *params)
{
	int i;
	double loglike=0;
	for (i=0; i < sbprofile_nparams; i++) {
		if ((set_auto_penalty_limits[i]) and ((params[i] < penalty_lower_limits[i]) or (params[i] > penalty_upper_limits[i]))) {
			//cout << "PENALTY! param " << i << " = " << params[i] << endl;
			return 1e30; // penalty prior
		}
		*(param[i]) = params[i];
	}
	update_meta_parameters();
	for (i=0; i < n_isophote_datapts; i++) {
		if (sbprofile_data[i]*0.0 != 0.0) continue;
		loglike += SQR((sbprofile_data[i] - sb_rsq(SQR(profile_fit_xivals[i])))/sbprofile_errors[i]);
	}
	loglike /= 2;
	return loglike;
}

bool SB_Profile::fit_egrad_profile_data(IsophoteData& isophote_data, const int egrad_param, const int fit_mode_in, const int n_livepts, const bool optimize_knots, const int mpi_np, const int mpi_id, const string fit_output_dir)
{
	// nested sampling: fitmode = 0
	// downhill simplex: fitmode = 1 or higher
	int fit_mode = fit_mode_in;
	if (ellipticity_gradient == false) {
		if (mpi_id==0) warn("ellipticity gradient must be on for egrad profile fitting");
		return false;
	}
#ifndef USE_FITPACK
	if (egrad_mode==0) {
		if (mpi_id==0) warn("cannot do B-spline fit without compiling with FITPACK");
		return false;
	}
#endif
	if (fit_mode==0) {
		if (egrad_mode==0) {
			if (mpi_id==0) warn("nested sampling is not currently set up with egrad_mode=0; switching to downhill simplex");
			fit_mode = 1;
		}
	}
	if (fit_mode < 0) fit_mode = 0; // this is a trick so that nested sampling can be run even in egrad_mode==0 at the end of an isofit call

	egrad_paramnum = egrad_param;
	n_isophote_datapts = isophote_data.n_xivals;
	if (egrad_param==0) {
		profile_fit_data = isophote_data.qvals;
		profile_fit_errs = isophote_data.q_errs;
	} else if (egrad_param==1) {
		profile_fit_data = isophote_data.thetavals;
		profile_fit_errs = isophote_data.theta_errs;
	} else if (egrad_param==2) {
		profile_fit_data = isophote_data.xcvals;
		profile_fit_errs = isophote_data.xc_errs;
	} else if (egrad_param==3) {
		profile_fit_data = isophote_data.ycvals;
		profile_fit_errs = isophote_data.yc_errs;
	} else if (egrad_param==4) {
		profile_fit_data = isophote_data.A3vals;
		profile_fit_errs = isophote_data.A3_errs;
	} else if (egrad_param==5) {
		profile_fit_data = isophote_data.B3vals;
		profile_fit_errs = isophote_data.B3_errs;
	} else if (egrad_param==6) {
		profile_fit_data = isophote_data.A4vals;
		profile_fit_errs = isophote_data.A4_errs;
	} else if (egrad_param==7) {
		profile_fit_data = isophote_data.B4vals;
		profile_fit_errs = isophote_data.B4_errs;
	} else if (egrad_param==8) {
		profile_fit_data = isophote_data.A5vals;
		profile_fit_errs = isophote_data.A5_errs;
	} else if (egrad_param==9) {
		profile_fit_data = isophote_data.B5vals;
		profile_fit_errs = isophote_data.B5_errs;
	} else if (egrad_param==10) {
		profile_fit_data = isophote_data.A6vals;
		profile_fit_errs = isophote_data.A6_errs;
	} else if (egrad_param==11) {
		profile_fit_data = isophote_data.B6vals;
		profile_fit_errs = isophote_data.B6_errs;
	}

	int i,j,k;
	if (egrad_param < 4) {
		if ((egrad_mode==0) and (fit_mode != 0)) {
			profile_fit_nparams = n_bspline_knots_tot - 2*bspline_order - 1;
			profile_fit_egrad_params = geometric_knots[egrad_param];
			profile_fit_bspline_coefs = geometric_param[egrad_param];
		} else {
			profile_fit_nparams = n_egrad_params[egrad_param];
			profile_fit_egrad_params = geometric_param[egrad_param];
		}
		profile_fit_istart = sbprofile_nparams;
		for (i=0; i < egrad_param; i++) profile_fit_istart += n_egrad_params[i];
	} else {
		int fparam, fmode, sinmode_num;
		fparam = egrad_param - 4;
		fmode = 3 + fparam / 2;
		sinmode_num = fparam % 2;
		bool fmode_found = false;
		for (i=0,k=0; i < n_fourier_modes; i++, k+=2) {
			if (fourier_mode_mvals[i]==fmode) {
				if ((egrad_mode==0) and (fit_mode != 0)) {
					profile_fit_nparams = n_bspline_knots_tot - 2*bspline_order - 1;
					profile_fit_egrad_params = fourier_knots[k+sinmode_num];
					profile_fit_bspline_coefs = fourier_param[k+sinmode_num];
				} else {
					profile_fit_nparams = n_fourier_grad_params[i];
					profile_fit_egrad_params = fourier_param[k+sinmode_num];
				}
				profile_fit_istart = fourier_mode_paramnum[i];
				if (sinmode_num==1) profile_fit_istart += profile_fit_nparams;
				egrad_paramnum = 4 + i;
				fmode_found = true;
				break;
			}
		}
		//cout << "fparam=" << fparam << " fmode=" << fmode << " sinmode_num=" << sinmode_num << " istart=" << profile_fit_istart << endl;
		if (!fmode_found) die("could not find Fourier mode");
	}

	double bspline_logximin, bspline_logximax;
	if (egrad_mode==0) {
		// note that B-spline curve fitting requires that xi_initial_egrad and xi_final_egrad be set to the initial/final data xi values
		xi_initial_egrad = isophote_data.xivals[0];
		xi_final_egrad = isophote_data.xivals[n_isophote_datapts-1];
		bspline_logximin = log(xi_initial_egrad)/ln10;
		bspline_logximax = log(xi_final_egrad)/ln10;

		// note that if fit_mode==0, we do MCMC and the knots will be fixed. If fit_mode == 1 and optimize_knots is set to true, knots will be optimized
		if ((fit_mode == 1) and (optimize_knots)) {
			int n_unique_knots = n_bspline_knots_tot - 2*bspline_order;
			double logxi, logxistep = (bspline_logximax-bspline_logximin)/(n_unique_knots-1);
			double xi, xistep = (xi_final_egrad-xi_initial_egrad)/(n_unique_knots-1);
			for (j=0; j < bspline_order; j++) {
				profile_fit_egrad_params[j] = bspline_logximin;
			}
			if (!use_linear_xivals) {
				for (j=0, logxi=bspline_logximin; j < n_unique_knots; j++, logxi += logxistep) profile_fit_egrad_params[j+bspline_order] = logxi;
			} else {
				for (j=0, xi=xi_initial_egrad; j < n_unique_knots; j++, xi += xistep) {
					profile_fit_egrad_params[j+bspline_order] = log(xi)/ln10;
				}
			}
			for (j=0; j < bspline_order; j++) profile_fit_egrad_params[n_bspline_knots_tot-bspline_order+j] = bspline_logximax;
		}
	} else {
		for (i=profile_fit_istart; i < profile_fit_istart + profile_fit_nparams; i++) {
			if (!vary_params[i]) {
				if (mpi_id==0) warn("all egrad profile parameters must be allowed to vary (param %i set fixed, profile_fit_nparams=%i, istart=%i)",i,profile_fit_nparams,profile_fit_istart);
				return false;
			}
			if ((fit_mode==0) and (lower_limits.size() != n_vary_params)) {
				if (mpi_id==0) warn("lower/upper prior limits have not been set for egrad profile parameters (limit size=%i, nvary=%i)",lower_limits.size(),n_vary_params);
				return false;
			}
		}
	}
	profile_fit_xivals = isophote_data.xivals;
	profile_fit_logxivals = isophote_data.logxivals;
	profile_fit_weights = new double[n_isophote_datapts];
	for (i=0; i < n_isophote_datapts; i++) profile_fit_weights[i] = 1.0/profile_fit_errs[i];

	double *fitparams = new double[profile_fit_nparams];
	double *param_errors = new double[profile_fit_nparams];
	
	if (egrad_mode==0) {
		allocate_bspline_work_arrays(n_isophote_datapts);
		for (i=0; i < profile_fit_nparams; i++) fitparams[i] = profile_fit_egrad_params[bspline_order+i+1] - profile_fit_egrad_params[bspline_order+i];
	}

	if (fit_mode==0) {
#ifdef USE_MPI
		Set_MCMC_MPI(mpi_np,mpi_id);
#endif
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&SB_Profile::profile_fit_loglike);
		double *lower = new double[profile_fit_nparams];
		double *upper = new double[profile_fit_nparams];
		for (i=0,j=0,k=0; i < n_params; i++) {
			if (vary_params[i]) {
				if (i >= profile_fit_istart) {
					lower[k] = lower_limits[j];
					upper[k] = upper_limits[j];
					k++;
				}
				j++;
			}
			if (k==profile_fit_nparams) break;
		}

		//cout << "nparams=" << profile_fit_nparams << endl;
		//cout << "LIMITS: " << endl;
		//for (i=0; i < profile_fit_nparams; i++) {
			//cout << lower[i] << " " << upper[i] << endl;
		//}
		InputPoint(fitparams,lower,upper,profile_fit_nparams);
		double lnZ;
		double chisq_bestfit;
		//double chisq_bestfit = 2*(this->*LogLikePtr)(fitparams);

		string fit_output_filename, egrad_istring;
		stringstream egrad_istr;
		egrad_istr << egrad_param;
		egrad_istr >> egrad_istring;
		fit_output_filename = "egrad_profile" + egrad_istring;

		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << profile_fit_nparams << " " << 0 << endl;
		pnumfile.close();

		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=profile_fit_istart, j=0; j < profile_fit_nparams; i++, j++) {
			pnamefile << paramnames[i] << endl;
		}
		pnamefile.close();

		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < profile_fit_nparams; i++)
		{
			prangefile << lower[i] << " " << upper[i] << endl;
		}
		prangefile.close();

		string filename = fit_output_dir + "/" + fit_output_filename;
		MonoSample(filename.c_str(),n_livepts,lnZ,fitparams,param_errors,false);
		chisq_bestfit = 2*(this->*LogLikePtr)(fitparams);
		if (mpi_id==0) cout << "chisq=" << chisq_bestfit << endl;
		delete[] lower;
		delete[] upper;
	} else {
		if ((egrad_mode==0) and ((!optimize_knots) or (fit_mode > 1))) {
			profile_fit_loglike_bspline(fitparams); // just fit B-spline with same knots as before; no need to call Simplex
		} else {
			double *stepsizes = new double [profile_fit_nparams];
			double (Simplex::*loglikeptr)(double*);
			if (egrad_mode==0) {
				loglikeptr = static_cast<double (Simplex::*)(double*)> (&SB_Profile::profile_fit_loglike_bspline);
				for (i=0; i < profile_fit_nparams; i++) {
					stepsizes[i] = fitparams[i]/4.0; // arbitrary
				}
				double min_data_interval = 1e30;
				for (i=0; i < n_isophote_datapts-1; i++) {
					if ((profile_fit_logxivals[i+1]-profile_fit_logxivals[i]) < min_data_interval) min_data_interval = (profile_fit_logxivals[i+1]-profile_fit_logxivals[i]);
				}
				// the minimum knot interval allowed is given in terms of a specified fraction of the spacing between data points
				profile_fit_min_knot_interval = 2*min_data_interval;
				//cout << "min interval: " << profile_fit_min_knot_interval << endl;
			} else {
				loglikeptr = static_cast<double (Simplex::*)(double*)> (&SB_Profile::profile_fit_loglike);
				for (i=profile_fit_istart, j=0; j < profile_fit_nparams; i++, j++) {
					if (angle_param[i]) stepsizes[j] = 20;
					else stepsizes[j] = 0.1;
					fitparams[j] = *(param[i]);
				}
			}
			double chisq_tolerance = 1e-4;
			if (qlens != NULL) chisq_tolerance = qlens->chisq_tolerance;

			initialize_simplex(fitparams,profile_fit_nparams,stepsizes,chisq_tolerance);
			simplex_set_display_bfpont(true);
			simplex_set_function(loglikeptr);
			int it, nrep=1;
			while (nrep-- >= 0) {
				it=0;
				downhill_simplex(it,10000,0); // do final run with zero temperature
			}
			delete[] stepsizes;
		}
	}
	//double loglike=0;
	//double pval;
	//for (i=0; i < n_isophote_datapts; i++) {
		//if (profile_fit_errs[i]>=1e30) continue; // in this case don't bother to include in chisq
		//pval = (this->*egrad_ptr)(profile_fit_xivals[i],profile_fit_egrad_params,egrad_paramnum);
		//loglike += SQR((profile_fit_data[i] - pval)/profile_fit_errs[i]);
		//cout << "datapt " << i << ": " << profile_fit_data[i] << " " << profile_fit_errs[i] << " " << pval << " logl=" << loglike << endl;
	//}
	//loglike /= 2;
	//cout << "LOGLIKE=" << loglike << endl;

	if (egrad_mode==0) {
		free_bspline_work_arrays();
	}
	if (egrad_param==0) update_ellipticity_meta_parameters(); // since the q-parameters have changed

	delete[] fitparams;
	delete[] param_errors;
	delete[] profile_fit_weights;
	profile_fit_xivals = NULL;
	profile_fit_logxivals = NULL;
	profile_fit_data = NULL;
	profile_fit_errs = NULL;
	profile_fit_weights = NULL;
	profile_fit_egrad_params = NULL;
	return true;
}

void SB_Profile::find_egrad_paramnums(int& qi, int& qf, int& theta_i, int& theta_f, int& amp_i, int& amp_f)
{
	qi = sbprofile_nparams;
	qf = qi + n_egrad_params[0];
	theta_i = qf;
	theta_f = theta_i + n_egrad_params[1];

	if (fourier_gradient) {
		amp_i = fourier_mode_paramnum[0];
		amp_f = amp_i;
		for (int i=0; i < n_fourier_modes; i++) {
			amp_f += 2*n_fourier_grad_params[i];
		}
	}
}

double SB_Profile::profile_fit_loglike(double *params)
{
	int i,j;
	double loglike=0;
	for (i=profile_fit_istart, j=0; j < profile_fit_nparams; i++, j++) {
		if (angle_param[i]) *(param[i]) = degrees_to_radians(params[j]);
		else *(param[i]) = params[j];
	}
	update_meta_parameters(); // this isn't really necessary now, but will become necessary if parameter transformations are made, e.g. ellipticity components
	//for (i=0; i < profile_fit_nparams; i++) {
		//cout << profile_fit_egrad_params[i] << endl;
	//}
	for (i=0; i < n_isophote_datapts; i++) {
		if (profile_fit_errs[i]>=1e30) continue; // in this case don't bother to include in chisq
		loglike += SQR((profile_fit_data[i] - (this->*egrad_ptr)(profile_fit_xivals[i],profile_fit_egrad_params,egrad_paramnum))/profile_fit_errs[i]);
	}
	loglike /= 2;
	//if (profile_fit_istart==22) {
		//cout << "params: " << params[0] << " " << params[1] << " " << params[2] << " " << params[3] << " " << "LOGLIKE=" << loglike << endl;
	//}
	return loglike;
}

double SB_Profile::profile_fit_loglike_bspline(double *params)
{
	// here, the nonlinear parameters are the knots that are being optimized
	int i;
	double tot_interval = 0;
	for (i=0; i < n_bspline_knots_tot-2*bspline_order-1; i++) {
		if (abs(params[i]) < profile_fit_min_knot_interval) { warn("knot interval too small; skipping B-spline fit"); return 1e30; }
		tot_interval += abs(params[i]);
		profile_fit_egrad_params[bspline_order+i+1] = profile_fit_egrad_params[bspline_order+i] + abs(params[i]);
	}
	for (i=0; i < bspline_order; i++) profile_fit_egrad_params[n_bspline_knots_tot-bspline_order+i] = profile_fit_egrad_params[n_bspline_knots_tot-bspline_order-1];

	//update_meta_parameters(); // this isn't really necessary now, but will become necessary if parameter transformations are made, e.g. ellipticity components
	if (tot_interval > 1.1*(log(xi_final_egrad/xi_initial_egrad)/ln10)) { warn("total interval too large (%g); skipping B-spline fit",pow(10,tot_interval)); return 1e30; } // penalty chisq

	double loglike = fit_bspline_curve(profile_fit_egrad_params,profile_fit_bspline_coefs);
	return loglike;
}

/*
double SB_Profile::calculate_Lmatrix_element(double x, double y, const int amp_index)
{
	return 0.0; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}
*/

void SB_Profile::calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

/*
void SB_Profile::calculate_gradient_Rmatrix_elements(double*& Rmatrix_elements, double &logdet)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}
*/

void SB_Profile::calculate_gradient_Rmatrix_elements(double* Rmatrix_elements, int* Rmatrix_index)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

void SB_Profile::calculate_curvature_Rmatrix_elements(double* Rmatrix, int* Rmatrix_index)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

void SB_Profile::update_amplitudes(double*& ampvec)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

double SB_Profile::surface_brightness_zeroth_order(double x, double y)
{
	return 0; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

double SB_Profile::get_scale_parameter()
{
	return 0; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

void SB_Profile::update_scale_parameter(const double scale)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

void SB_Profile::update_indxptr(const int newval)
{
	return;
}

/*
void SB_Profile::get_amplitudes(double *ampvec)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}
*/

void SB_Profile::plot_sb_profile(double rmin, double rmax, int steps, ofstream &sbout)
{
	double r, rstep;
	rstep = pow(rmax/rmin, 1.0/steps);
	int i;
	sbout << setiosflags(ios::scientific);
	bool converged;
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		sbout << r << " " << sb_rsq(r*r) << " " << endl;
	}
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
		if (angle_param[i]) cout << radians_to_degrees(*(param[i])) << " degrees";
		else cout << *(param[i]);
		if (i != n_params-1) cout << ", ";
	}
	if (center_anchored_to_lens) cout << " (center anchored to lens " << center_anchor_lens->lens_number << ")";
	else if (center_anchored_to_source) cout << " (center anchored to source " << center_anchor_source->sb_number << ")";
	if ((ellipticity_mode != default_ellipticity_mode) and (ellipticity_mode != -1)) {
		cout << " (";
		if (ellipticity_gradient) cout << "egrad=on,";
		if (fourier_gradient) cout << "fgrad=on,";
		cout << "emode=" << ellipticity_mode << ")"; // emode=3 is indicated by "pseudo-" name, not here
	} else {
		if (ellipticity_gradient) {
			if (fourier_gradient) cout << " (egrad,fgrad=on)";
			else cout << " (egrad=on)";
		}
	}
	if (lensed_center_coords) cout << " (xc=" << x_center << ", yc=" << y_center << ")";
	cout << endl;
	if ((ellipticity_gradient) and (egrad_mode==0)) {
		cout << "   q-knots: ";
		for (int i=bspline_order; i < n_bspline_knots_tot - bspline_order; i++) {
			cout << pow(10,geometric_knots[0][i]); // gives the elliptical radius values
			if (i < n_bspline_knots_tot-1) cout << " ";
		}
		cout << endl;
		cout << "   theta-knots: ";
		for (int i=bspline_order; i < n_bspline_knots_tot - bspline_order; i++) {
			cout << pow(10,geometric_knots[1][i]); // gives the elliptical radius values
			if (i < n_bspline_knots_tot-1) cout << " ";
		}
		cout << endl;
	}
	if ((fourier_gradient) and (egrad_mode==0)) {
		for (int j=0; j < n_fourier_grad_modes; j++) {
			stringstream mvalstr;
			string mvalstring;
			mvalstr << fourier_grad_mvals[j];
			mvalstr >> mvalstring;
			cout << "   A" << mvalstring << "-knots: ";
			for (int i=bspline_order; i < n_bspline_knots_tot - bspline_order; i++) {
				cout << pow(10,fourier_knots[0][i]); // gives the elliptical radius values
				if (i < n_bspline_knots_tot-1) cout << " ";
			}
			cout << endl;
			cout << "   B" << mvalstring << "-knots: ";
			for (int i=bspline_order; i < n_bspline_knots_tot - bspline_order; i++) {
				cout << pow(10,fourier_knots[1][i]); // gives the elliptical radius values
				if (i < n_bspline_knots_tot-1) cout << " ";
			}
			cout << endl;
		}
	}
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
	bool parameter_anchored = false;
	for (int i=0; i < n_params; i++) {
		if (anchor_parameter_to_source[i]) {
			parameter_anchored = true;
			cout << "Parameter " << i << " is anchored" << endl;
		}
	}
	if (parameter_anchored) {
		cout << "   anchored parameters: ";
		int j=0;
		for (int i=0; i < n_params; i++) {
			if (anchor_parameter_to_source[i]) {
				if (j > 0) cout << ", ";
				cout << paramnames[i] << " --> (source " << parameter_anchor_source[i]->sb_number << ": ";
				if ((parameter_anchor_ratio[i] != 1.0) or 
					(parameter_anchor_exponent[i] != 1.0)) {
					cout << parameter_anchor_ratio[i] << "*" << parameter_anchor_source[i]->paramnames[parameter_anchor_paramnum[i]];
					if (parameter_anchor_exponent[i] != 1.0) cout << "^" << parameter_anchor_exponent[i];
				}
				else cout << parameter_anchor_source[i]->paramnames[parameter_anchor_paramnum[i]];
				cout << ")";
				j++;
			}
		}
		cout << endl;
	}
}

void SB_Profile::window_params(double& xmin, double& xmax, double& ymin, double& ymax)
{
	double rmax = window_rmax();
	if (ellipticity_mode==0) {
		xmin = -rmax;
		xmax = rmax;
		ymin = -q*rmax;
		ymax = q*rmax;
	} else {
		double sqrtq = sqrt(q);
		xmin = -rmax/sqrtq;
		xmax = rmax/sqrtq;
		ymin = -sqrtq*rmax;
		ymax = sqrtq*rmax;
	}
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
	if (lensed_center_coords) scriptout << "-lensed_center ";

	for (int i=0; i < n_params; i++) {
		if (angle_param[i]) scriptout << radians_to_degrees(*(param[i]));
		else {
			// If this is an optional parameter, need to specify parameter name before the value
			if (paramnames[i]=="c0") scriptout << "c0="; // boxiness parameter
			else if (paramnames[i]=="rt") scriptout << "rt="; // truncation radius
			else {
				for (int j=0; j < n_fourier_modes; j++) {
					if (fourier_mode_paramnum[j]==i) scriptout << "f" << fourier_mode_mvals[j] << "="; // Fourier mode
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

Gaussian::Gaussian(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	model_name = "gaussian";
	sbtype = GAUSSIAN;
	setup_base_source_properties(6,2,true);
	qlens = qlens_in;
	max_sb = max_sb_in;
	sig_x = sig_x_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
}

Gaussian::Gaussian(const Gaussian* sb_in)
{
	max_sb = sb_in->max_sb;
	sig_x = sb_in->sig_x;
	sbtot = sb_in->sbtot;
	copy_base_source_data(sb_in);
	update_meta_parameters();
}

void Gaussian::update_meta_parameters()
{
	//max_sb = sbtot/(M_2PI*q*sig_x*sig_x);
	sbtot = max_sb*(M_2PI*sig_x*sig_x);
	if (ellipticity_mode==0) sbtot *= q;
	update_ellipticity_meta_parameters();
}

void Gaussian::assign_paramnames()
{
	paramnames[0] = "sbmax";     latex_paramnames[0] = "S";       latex_param_subscripts[0] = "max";
	paramnames[1] = "sigma"; latex_paramnames[1] = "\\sigma"; latex_param_subscripts[1] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

void Gaussian::assign_param_pointers()
{
	param[0] = &max_sb;
	param[1] = &sig_x;
	set_geometric_param_pointers(sbprofile_nparams);
}

void Gaussian::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = (max_sb != 0) ? 0.1*max_sb : 0.1;
	stepsizes[index++] = (sig_x != 0) ? 0.1*sig_x : 0.1; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void Gaussian::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
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

Sersic::Sersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	model_name = "sersic";
	sbtype = SERSIC;
	setup_base_source_properties(7,3,true);
	qlens = qlens_in;
	n = n_in;
	Reff = Reff_in;
	s0 = s0_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
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
	b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n); // from Cardone 2003 (or Ciotti 1999)
	//k = b*pow(1.0/Reff,1.0/n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_ellipticity_meta_parameters();
}

void Sersic::assign_paramnames()
{
	paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	paramnames[1] = "Reff"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "eff";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

void Sersic::assign_param_pointers()
{
	param[0] = &s0;
	param[1] = &Reff;
	param[2] = &n;
	set_geometric_param_pointers(sbprofile_nparams);
}

void Sersic::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void Sersic::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

double Sersic::sb_rsq(const double rsq)
{
	return s0*exp(-b*pow(rsq/(Reff*Reff),0.5/n));
}

double Sersic::window_rmax()
{
	double fac = pow(3.0/b,n);
	if (fac < 1) fac = 1;
	return Reff*fac;
}

double Sersic::length_scale()
{
	return Reff;
}

Cored_Sersic::Cored_Sersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	model_name = "csersic";
	sbtype = CORED_SERSIC;
	setup_base_source_properties(8,4,true);
	qlens = qlens_in;
	n = n_in;
	Reff = Reff_in;
	s0 = s0_in;
	rc = rc_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
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
	b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	//k = b*pow(1.0/Reff,1.0/n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_ellipticity_meta_parameters();
}

void Cored_Sersic::assign_paramnames()
{
	paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	paramnames[1] = "Reff"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "eff";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	paramnames[3] = "rc"; latex_paramnames[3] = "r"; latex_param_subscripts[3] = "c";
	set_geometric_paramnames(sbprofile_nparams);
}

void Cored_Sersic::assign_param_pointers()
{
	param[0] = &s0;
	param[1] = &Reff;
	param[2] = &n;
	param[3] = &rc;
	set_geometric_param_pointers(sbprofile_nparams);
}

void Cored_Sersic::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void Cored_Sersic::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_auto_penalty_limits[3] = true; penalty_lower_limits[3] = 0; penalty_upper_limits[3] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

double Cored_Sersic::sb_rsq(const double rsq)
{
	return s0*exp(-b*pow((rsq+rc*rc)/(Reff*Reff),0.5/n));
}

double Cored_Sersic::window_rmax()
{
	return Reff*pow(3.0/b,n);
}

double Cored_Sersic::length_scale()
{
	return Reff;
}

CoreSersic::CoreSersic(const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in,	const double &gamma_in, const double &alpha_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	model_name = "Csersic";
	sbtype = CORE_SERSIC;
	setup_base_source_properties(10,6,true);
	qlens = qlens_in;
	n = n_in;
	Reff = Reff_in;
	s0 = s0_in;
	rc = rc_in;
	gamma = gamma_in;
	alpha = alpha_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
}

CoreSersic::CoreSersic(const CoreSersic* sb_in)
{
	s0 = sb_in->s0;
	n = sb_in->n;
	Reff = sb_in->Reff;
	rc = sb_in->rc;
	gamma = sb_in->gamma;
	alpha = sb_in->alpha;
	copy_base_source_data(sb_in);
	update_meta_parameters();
}

void CoreSersic::update_meta_parameters()
{
	double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	k = b*pow(1.0/Reff,1.0/n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_ellipticity_meta_parameters();
}

void CoreSersic::assign_paramnames()
{
	paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	paramnames[1] = "Reff"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "eff";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	paramnames[3] = "rc"; latex_paramnames[3] = "r"; latex_param_subscripts[3] = "c";
	paramnames[4] = "gamma"; latex_paramnames[4] = "\\gamma"; latex_param_subscripts[4] = "";
	paramnames[5] = "alpha"; latex_paramnames[5] = "\\alpha"; latex_param_subscripts[5] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

void CoreSersic::assign_param_pointers()
{
	param[0] = &s0;
	param[1] = &Reff;
	param[2] = &n;
	param[3] = &rc;
	param[4] = &gamma;
	param[5] = &alpha;
	set_geometric_param_pointers(sbprofile_nparams);
}

void CoreSersic::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void CoreSersic::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_auto_penalty_limits[3] = true; penalty_lower_limits[3] = 0; penalty_upper_limits[3] = 1e30;
	set_auto_penalty_limits[4] = true; penalty_lower_limits[4] = 0; penalty_upper_limits[4] = 1e30;
	set_auto_penalty_limits[5] = true; penalty_lower_limits[5] = 0; penalty_upper_limits[5] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

double CoreSersic::sb_rsq(const double rsq)
{
	double rca = pow(rc,alpha);
	double ra = pow(rsq,alpha/2);
	return s0*pow(1+rca/ra,gamma/alpha)*exp(-k*pow(ra+rca,1.0/(alpha*n)));
}

double CoreSersic::window_rmax()
{
	return pow(3.0/k,n);
}

double CoreSersic::length_scale()
{
	return Reff;
}

DoubleSersic::DoubleSersic(const double &s0_in, const double &delta_s_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	model_name = "dsersic";
	sbtype = DOUBLE_SERSIC;
	setup_base_source_properties(10,6,true);
	qlens = qlens_in;
	s0 = s0_in;
	delta_s = delta_s_in;
	n1 = n1_in;
	Reff1 = Reff1_in;
	n2 = n2_in;
	Reff2 = Reff2_in;

	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
}

DoubleSersic::DoubleSersic(const DoubleSersic* sb_in)
{
	s0 = sb_in->s0;
	delta_s = sb_in->delta_s;
	n1 = sb_in->n1;
	Reff1 = sb_in->Reff1;
	n2 = sb_in->n2;
	Reff2 = sb_in->Reff2;
	copy_base_source_data(sb_in);
	update_meta_parameters();
}

void DoubleSersic::update_meta_parameters()
{
	s0_1 = s0*(1+delta_s)/2;
	s0_2 = s0*(1-delta_s)/2;
	b1 = 2*n1 - 0.33333333333333 + 4.0/(405*n1) + 46.0/(25515*n1*n1) + 131.0/(1148175*n1*n1*n1);
	b2 = 2*n2 - 0.33333333333333 + 4.0/(405*n2) + 46.0/(25515*n2*n2) + 131.0/(1148175*n2*n2*n2);
	update_ellipticity_meta_parameters();
}

void DoubleSersic::assign_paramnames()
{
	paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	paramnames[1] = "delta_s"; latex_paramnames[1] = "\\Delta"; latex_param_subscripts[1] = "s";
	paramnames[2] = "Reff1"; latex_paramnames[2] = "R"; latex_param_subscripts[2] = "eff,1";
	paramnames[3] = "n1"; latex_paramnames[3] = "n"; latex_param_subscripts[3] = "1";
	paramnames[4] = "Reff2"; latex_paramnames[4] = "R"; latex_param_subscripts[4] = "eff,2";
	paramnames[5] = "n2"; latex_paramnames[5] = "n"; latex_param_subscripts[5] = "2";

	set_geometric_paramnames(sbprofile_nparams);
}

void DoubleSersic::assign_param_pointers()
{
	param[0] = &s0;
	param[1] = &delta_s;
	param[2] = &Reff1;
	param[3] = &n1;
	param[4] = &Reff2;
	param[5] = &n2;
	set_geometric_param_pointers(sbprofile_nparams);
}

void DoubleSersic::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void DoubleSersic::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = false; penalty_lower_limits[1] = -1e30; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_auto_penalty_limits[3] = true; penalty_lower_limits[3] = 0; penalty_upper_limits[3] = 1e30;
	set_auto_penalty_limits[4] = true; penalty_lower_limits[4] = 0; penalty_upper_limits[4] = 1e30;
	set_auto_penalty_limits[5] = true; penalty_lower_limits[5] = 0; penalty_upper_limits[5] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

double DoubleSersic::sb_rsq(const double rsq)
{
	return (s0_1*exp(-b1*pow(rsq/(Reff1*Reff1),0.5/n1)) + s0_2*exp(-b2*pow(rsq/(Reff2*Reff2),0.5/n2)));
}

double DoubleSersic::window_rmax()
{
	double max1 = Reff1*pow(3.0/b1,n1);
	double max2 = Reff2*pow(3.0/b2,n2);
	return dmax(max1,max2);
}

double DoubleSersic::length_scale()
{
	return sqrt(Reff1*Reff1 + Reff2*Reff2);
}

SPLE::SPLE(const double &bb, const double &aa, const double &ss, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, QLens* cosmo_in)
{
	model_name = "sple";
	sbtype = sple;
	qlens = cosmo_in;
	setup_base_source_properties(7,3,true); // number of parameters = 7, is_elliptical_source = true
	bs = bb;
	alpha = aa;
	s = ss;
	if (s < 0) s = -s; // don't allow negative core radii
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
}

SPLE::SPLE(const SPLE* sb_in)
{
	bs = sb_in->bs;
	alpha = sb_in->alpha;
	s = sb_in->s;

	copy_base_source_data(sb_in);
	update_meta_parameters();
}

void SPLE::assign_paramnames()
{
	paramnames[0] = "bs";     latex_paramnames[0] = "b";       latex_param_subscripts[0] = "s";
	paramnames[1] = "alpha"; latex_paramnames[1] = "\\alpha"; latex_param_subscripts[1] = "";
	paramnames[2] = "s";     latex_paramnames[2] = "s";       latex_param_subscripts[2] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

void SPLE::assign_param_pointers()
{
	param[0] = &bs;
	param[1] = &alpha;
	param[2] = &s;
	set_geometric_param_pointers(sbprofile_nparams);
}

void SPLE::update_meta_parameters()
{
	update_ellipticity_meta_parameters();
	// these meta-parameters are used in analytic formulas for deflection, potential, etc.
}

void SPLE::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1*bs;
	stepsizes[index++] = 0.1;
	stepsizes[index++] = 0.02*bs; // this one is a bit arbitrary, but hopefully reasonable enough
	set_geometric_param_auto_stepsizes(index);
}

void SPLE::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 2;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

double SPLE::sb_rsq(const double rsq)
{
	return ((bs==0.0) ? 0.0 : ((2-alpha) * pow(bs*bs/(s*s+rsq), alpha/2) / 2));
}

double SPLE::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 3*bs;
}

double SPLE::length_scale()
{
	return bs;
}

Shapelet::Shapelet(const double &amp00, const double &scale_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int nn, const bool truncate, const int parameter_mode_in, QLens* qlens_in)
{
	model_name = "shapelet";
	sbtype = SHAPELET;
	int npar = 5;
	setup_base_source_properties(npar,1,false,parameter_mode_in);
	qlens = qlens_in;
	if (parameter_mode==0) {
		sig = scale_in;
		sig_factor = 1.0;
	} else {
		sig_factor = scale_in;
		sig = 1.0; // this will be set automatically using the 'find_shapelet_scaling_parameters' function in lens.cpp
	}
	n_shapelets = nn;
	indxptr = &n_shapelets;
	amps = new double*[n_shapelets];
	for (int i=0; i < n_shapelets; i++) amps[i] = new double[n_shapelets];
	for (int i=0; i < n_shapelets; i++) {
		for (int j=0; j < n_shapelets; j++) {
			amps[i][j] = 0;
		}
	}
	amps[0][0] = amp00;
	truncate_at_3sigma = truncate;

	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	update_meta_parameters();
}

Shapelet::Shapelet(const Shapelet* sb_in)
{
	n_shapelets = sb_in->n_shapelets;
	indxptr = &n_shapelets;
	sig = sb_in->sig;
	sig_factor = sb_in->sig_factor;
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
	int indx=0;
	if (parameter_mode==0) {
		paramnames[indx] = "sigma"; latex_paramnames[indx] = "\\sigma"; latex_param_subscripts[indx] = ""; indx++;
	} else {
		paramnames[indx] = "sigfac"; latex_paramnames[indx] = "f"; latex_param_subscripts[indx] = "\\sigma"; indx++;
	}
	set_geometric_paramnames(indx);
}

void Shapelet::assign_param_pointers()
{
	int indx=0;
	if (parameter_mode==0) {
		param[indx++] = &sig;
	} else {
		param[indx++] = &sig_factor;
	}
	set_geometric_param_pointers(indx);
}

void Shapelet::set_auto_stepsizes()
{
	int indx=0;
	if (parameter_mode==0) {
		stepsizes[indx++] = (sig != 0) ? 0.1*sig : 0.1; // arbitrary
	} else {
		stepsizes[indx++] = (sig_factor != 0) ? 0.1*sig_factor : 0.1; // arbitrary
	}
	set_geometric_param_auto_stepsizes(indx);
}

void Shapelet::set_auto_ranges()
{
	int indx=0;
	set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30; indx++;
	set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30; indx++;
	set_geometric_param_auto_ranges(indx);
}

double Shapelet::surface_brightness(double x, double y)
{
	x -= x_center;
	y -= y_center;
	if ((truncate_at_3sigma) and (sqrt(x*x+y*y) > 2.3*sig)) return 0;
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
	//cout << "AMP00: " << amps[0][0] << endl;

	delete[] hermvals_x;
	delete[] hermvals_y;
	return sb;
}

double Shapelet::surface_brightness_zeroth_order(double x, double y)
{
	x -= x_center;
	y -= y_center;
	if ((truncate_at_3sigma) and (sqrt(x*x+y*y) > 2.3*sig)) return 0;
	if (theta != 0) rotate(x,y);

	double gaussfactor = (0.5641895835477563/sig)*exp(-(q*x*x+y*y/q)/(2*sig*sig));
	return (amps[0][0]*gaussfactor);
}

void Shapelet::calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight)
{
	x -= x_center;
	y -= y_center;
	if ((truncate_at_3sigma) and (sqrt(x*x+y*y) > 2.3*sig)) return;
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
			*(Lmatrix_elements++) += weight*gaussfactor*hermvals_x[i]*hermvals_y[j];
		}
	}

	delete[] hermvals_x;
	delete[] hermvals_y;
}

void Shapelet::calculate_gradient_Rmatrix_elements(double* Rmatrix, int* Rmatrix_index)
{
	if (sig==0) die("sigma cannot be zero!!");
	int i,j,n=0;
	int indx = n_shapelets*n_shapelets + 1;
	double norm = 1.0/(2*sig*sig);
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			Rmatrix[n] = norm*((2*i+1) + (2*j+1));
			Rmatrix_index[n] = indx;

			/*
			// off-diagonal elements here are lower triangular, but we need upper triangular. FIX THIS!
			if (i > 1) {
				Rmatrix_index[indx] = n - 2*n_shapelets;
				Rmatrix[indx] = -norm*sqrt(i*(i-1));
				indx++;
			}
			if (j > 1) {
				Rmatrix_index[indx] = n - 2;
				Rmatrix[indx] = -norm*sqrt(j*(j-1));
				indx++;
			}
			*/
			n++;
		}
	}
	Rmatrix_index[n] = indx;
}

void Shapelet::calculate_curvature_Rmatrix_elements(double* Rmatrix, int* Rmatrix_index)
{
	if (sig==0) die("sigma cannot be zero!!");
	int i,j,n=0;
	double ip, jp;
	int indx = n_shapelets*n_shapelets + 1;
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			ip = sqrt(i*(i+1));
			jp = sqrt(j*(j+1));
			Rmatrix[n] = (4*(i*i+j*j) + 3*(i+j) + 6 + 2*i*j + 2*ip*jp + 2*(i+j)*(ip + jp))/(4*SQR(sig*sig));
			Rmatrix_index[n] = indx;
			n++;
		}
	}
	Rmatrix_index[n] = indx;
}

void Shapelet::update_amplitudes(double*& ampvec)
{
	int i,j,k=0;

	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			amps[i][j] = *(ampvec++);
		}
	}
}

/*
void Shapelet::get_amplitudes(double *ampvec)
{
	int i,j,k=0;
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			ampvec[k++] = amps[i][j];
		}
	}
}
*/

double Shapelet::get_scale_parameter()
{
	return sig;
}

void Shapelet::update_scale_parameter(const double scale)
{
	if (parameter_mode==0) {
		sig = scale;
	} else {
		sig = sig_factor*scale;
	}
}

void Shapelet::update_indxptr(const int newval)
{
	// indxptr points to n_shapelets
	int old_nn = n_shapelets;
	n_shapelets = newval;
	indxptr = &n_shapelets;
	double **newamps = new double*[n_shapelets];
	for (int i=0; i < n_shapelets; i++) newamps[i] = new double[n_shapelets];
	for (int i=0; i < n_shapelets; i++) {
		for (int j=0; j < n_shapelets; j++) {
			if ((i < old_nn) and (j < old_nn)) newamps[i][j] = amps[i][j];
			else newamps[i][j] = 0;
		}
	}
	for (int i=0; i < old_nn; i++) delete[] amps[i];
	delete[] amps;
	amps = newamps;
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
	if (truncate_at_3sigma) arg += " -truncate";
	return true;
}


SB_Multipole::SB_Multipole(const double &A_m_in, const double r0_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool sine, QLens* qlens_in)
{
	model_name = "sbmpole";
	sbtype = SB_MULTIPOLE;
	//stringstream mstr;
	//string mstring;
	//mstr << m_in;
	//mstr >> mstring;
	//special_parameter_command = "m=" + mstring;
	sine_term = sine;
	setup_base_source_properties(5,0,false);
	qlens = qlens_in;

	r0 = r0_in;
	m = m_in;
	A_n = A_m_in;
	set_angle(theta_degrees);
	x_center = xc_in;
	y_center = yc_in;
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
	param[2] = &theta; angle_param[2] = true;
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

TopHat::TopHat(const double &sb_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	model_name = "tophat";
	sbtype = TOPHAT;
	setup_base_source_properties(6,2,true);
	qlens = qlens_in;
	sb = sb_in; rad = rad_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
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
	set_geometric_param_pointers(sbprofile_nparams);
}

void TopHat::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void TopHat::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
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


