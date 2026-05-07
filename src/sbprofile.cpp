#include "sbprofile.h"
#include "qlens.h"
#include "egrad.h"
#include "mathexpr.h"
#include "errors.h"
#include "params.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

bool SB_Profile::orient_major_axis_north = false;
bool SB_Profile::use_sb_ellipticity_components = false;
int SB_Profile::default_ellipticity_mode = 1;
bool SB_Profile::use_fmode_scaled_amplitudes = false;
bool SB_Profile::fourier_use_eccentric_anomaly = true;
bool SB_Profile::fourier_sb_perturbation = false; // if true, add fourier modes to the surface brightness, rather than the elliptical radius
double SB_Profile::zoom_split_factor = 2;
double SB_Profile::zoom_scale = 4;

SB_Profile::SB_Profile(const char *splinefile, const int band_in, const double &zsrc_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in, QLens* qlens_in)
{
	sbparams = &sbparams_spl;
#ifdef USE_STAN
	sbparams_dif = &sbparams_spl_dif;
#endif
	model_name = "sbspline";
	sbtype = SB_SPLINE;
	band = band_in;
	setup_base_source_properties(6,2,true);
	qlens = qlens_in;
	set_geometric_parameters<double>(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	sbparams_spl.qx_parameter = qx_in;
	sbparams_spl.f_parameter = f_in;
	sbparams_spl.sb_spline.input(splinefile);
	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

void SB_Profile::setup_base_source_properties(const int np, const int sbprofile_np, const bool is_elliptical_source, const int pmode_in) // default pmode_in=0
{
	set_null_ptrs_and_values(); // sets pointers to NULL to make sure qlens doesn't try to delete them during setup
	sbparams->param = NULL;
	sbparams->zsrc = 0;
#ifdef USE_STAN
	sbparams_dif->param = NULL;
	sbparams_dif->zsrc = 0;
#endif

	parameter_mode = pmode_in;
	center_anchored_to_lens = false;
	center_anchored_to_source = false;
	center_anchored_to_ptsrc = false;
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
#ifdef USE_STAN
	assign_param_pointers_autodif();
#endif
	assign_paramnames();

	include_limits = false;
}

SB_Profile::SB_Profile(const SB_Profile* sb_in)
{
	sbparams = &sbparams_spl;
#ifdef USE_STAN
	sbparams_dif = &sbparams_spl_dif;
#endif
	SB_Spline_Params<double>& p = assign_sbspline_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.qx_parameter = sb_in->sbparams_spl.qx_parameter;
	p.f_parameter = sb_in->sbparams_spl.f_parameter;
	p.sb_spline.input(sb_in->sbparams_spl.sb_spline);
	copy_base_source_data(sb_in);
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

void SB_Profile::copy_base_source_data(const SB_Profile* sb_in)
{
	set_null_ptrs_and_values();
	model_name = sb_in->model_name;
	sbtype = sb_in->sbtype;
	band = sb_in->band;
	qlens = sb_in->qlens;
	sb_number = sb_in->sb_number;
	set_nparams(sb_in->n_params);
	sbprofile_nparams = sb_in->sbprofile_nparams;
	parameter_mode = sb_in->parameter_mode;
	center_anchored_to_lens = sb_in->center_anchored_to_lens;
	center_anchor_lens = sb_in->center_anchor_lens;
	center_anchored_to_source = sb_in->center_anchored_to_source;
	center_anchored_to_ptsrc = sb_in->center_anchored_to_ptsrc;
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

	sbparams->zsrc = sb_in->sbparams->zsrc;
	sbparams->q = sb_in->sbparams->q;
	sbparams->epsilon1 = sb_in->sbparams->epsilon1;
	sbparams->epsilon2 = sb_in->sbparams->epsilon2;
	angle_param_exists = sb_in->angle_param_exists;
	if (angle_param_exists) set_angle_radians(sb_in->sbparams->theta);
	sbparams->x_center = sb_in->sbparams->x_center;
	sbparams->y_center = sb_in->sbparams->y_center;
	if (lensed_center_coords) {
		sbparams->x_center_lensed = sb_in->sbparams->x_center_lensed;
		sbparams->y_center_lensed = sb_in->sbparams->y_center_lensed;
	}

	include_limits = sb_in->include_limits;
	if (include_limits) {
		lower_limits.input(sb_in->lower_limits);
		upper_limits.input(sb_in->upper_limits);
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
	if (include_boxiness_parameter) sbparams->c0 = sb_in->sbparams->c0;
	if (include_truncation_radius) sbparams->rt = sb_in->sbparams->rt;
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
#ifdef USE_STAN
	sync_autodif_parameters();
	assign_param_pointers_autodif();
#endif
}

#ifdef USE_STAN
void SB_Profile::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_spl_dif.qx_parameter = sbparams_spl.qx_parameter;
	sbparams_spl_dif.f_parameter = sbparams_spl.f_parameter;
	//sbparams_spl_dif.sb_spline.input(sbparams_spl.sb_spline);
}
#endif

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

	if (sbparams->param != NULL) delete[] sbparams->param;
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

	sbparams->param = new double*[n_params];
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
#ifdef USE_STAN
	sbparams_dif->param = new stan::math::var*[n_params];
#endif
}

void SB_Profile::anchor_center_to_lens(LensProfile** center_anchor_list, const int &center_anchor_lens_number)
{
	if (!center_anchored_to_lens) center_anchored_to_lens = true;
	center_anchor_lens = center_anchor_list[center_anchor_lens_number];
	sbparams->x_center = center_anchor_lens->lensparams->x_center;
	sbparams->y_center = center_anchor_lens->lensparams->y_center;
#ifdef USE_STAN
	sbparams_dif->x_center = center_anchor_lens->lensparams_dif->x_center;
	sbparams_dif->y_center = center_anchor_lens->lensparams_dif->y_center;
#endif
}

void SB_Profile::anchor_center_to_source(SB_Profile** center_anchor_list, const int &center_anchor_source_number)
{
	if (!center_anchored_to_source) center_anchored_to_source = true;
	center_anchor_source = center_anchor_list[center_anchor_source_number];
	sbparams->x_center = center_anchor_source->sbparams->x_center;
	sbparams->y_center = center_anchor_source->sbparams->y_center;
}

void SB_Profile::anchor_center_to_ptsrc(PointSource** center_anchor_list, const int &center_anchor_ptsrc_number)
{
	if (!center_anchored_to_ptsrc) center_anchored_to_ptsrc = true;
	center_anchor_ptsrc = center_anchor_list[center_anchor_ptsrc_number];
	sbparams->x_center = center_anchor_ptsrc->ptsrc_params.pos[0];
	sbparams->y_center = center_anchor_ptsrc->ptsrc_params.pos[1];
}

int SB_Profile::get_center_anchor_number() {
	if (center_anchored_to_lens) return center_anchor_lens->lens_number;
	else if (center_anchored_to_source) return center_anchor_source->sb_number;
	else if (center_anchored_to_ptsrc) return center_anchor_ptsrc->entry_number;
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
	} else if (center_anchored_to_ptsrc) {
		center_anchored_to_ptsrc = false;
		center_anchor_ptsrc = NULL;
	}
}

void SB_Profile::add_boxiness_parameter(const double c0_in, const bool vary_c0)
{
	// NOTE: this should be fixed up so that it comes before the Fourier modes, and the paramnames, stepsizes etc. should be automatically
	// assigned in the general functions just as the Fourier modes are. FIX!
	if (include_boxiness_parameter) return;
	include_boxiness_parameter = true;
	sbparams->c0 = c0_in;
#ifdef USE_STAN
	sbparams_dif->c0 = c0_in;
#endif
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

	delete[] sbparams->param;
	sbparams->param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
#ifdef USE_STAN
	delete[] sbparams_dif->param;
	sbparams_dif->param = new stan::math::var*[n_params];
	assign_param_pointers_autodif();
#endif
	assign_paramnames();
}

void SB_Profile::add_truncation_radius(const double rt_in, const bool vary_rt)
{
	// NOTE: this should be fixed up so that it comes before the Fourier modes, and the paramnames, stepsizes etc. should be automatically
	// assigned in the general functions just as the Fourier modes are. FIX!
	if (include_truncation_radius) return;
	include_truncation_radius = true;
	sbparams->rt = rt_in;
#ifdef USE_STAN
	sbparams_dif->rt = rt_in;
#endif
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

	delete[] sbparams->param;
	sbparams->param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
#ifdef USE_STAN
	delete[] sbparams_dif->param;
	sbparams_dif->param = new stan::math::var*[n_params];
	assign_param_pointers_autodif();
#endif
	assign_paramnames();
}

void SB_Profile::add_fourier_mode(const int m_in, const double amp_in, const double amp2_in, const bool vary1, const bool vary2)
{
	n_fourier_modes++;
	sbparams->fourier_mode_cosamp.resize(n_fourier_modes);
	sbparams->fourier_mode_sinamp.resize(n_fourier_modes);
	sbparams->fourier_mode_cosamp[n_fourier_modes-1] = amp_in;
	sbparams->fourier_mode_sinamp[n_fourier_modes-1] = amp2_in;
#ifdef USE_STAN
	sbparams_dif->fourier_mode_cosamp.resize(n_fourier_modes);
	sbparams_dif->fourier_mode_sinamp.resize(n_fourier_modes);
	sbparams_dif->fourier_mode_cosamp[n_fourier_modes-1] = amp_in;
	sbparams_dif->fourier_mode_sinamp[n_fourier_modes-1] = amp2_in;
#endif
	fourier_mode_mvals.resize(n_fourier_modes);
	fourier_mode_mvals[n_fourier_modes-1] = m_in;
	fourier_mode_paramnum.resize(n_fourier_modes);
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

	delete[] sbparams->param;
	sbparams->param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
#ifdef USE_STAN
	delete[] sbparams_dif->param;
	sbparams_dif->param = new stan::math::var*[n_params];
	assign_param_pointers_autodif();
#endif
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
	sbparams->fourier_mode_sinamp.resize(n_fourier_modes);
	sbparams->fourier_mode_cosamp.resize(n_fourier_modes);
#ifdef USE_STAN
	sbparams_dif->fourier_mode_sinamp.resize(n_fourier_modes);
	sbparams_dif->fourier_mode_cosamp.resize(n_fourier_modes);
#endif
	fourier_mode_mvals.resize(n_fourier_modes);
	fourier_mode_paramnum.resize(n_fourier_modes);

	delete[] sbparams->param;
	sbparams->param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
#ifdef USE_STAN
	delete[] sbparams_dif->param;
	sbparams_dif->param = new stan::math::var*[n_params];
	assign_param_pointers_autodif();
#endif
}

bool SB_Profile::enable_ellipticity_gradient(Vector<double>& efunc_params, const int egrad_mode, const int n_bspline_coefs, const Vector<double>& knots, const double ximin, const double ximax, const double xiref, const bool linear_xivals, const bool copy_vary_settings, boolvector* vary_egrad)
{
	if (ellipticity_mode==-1) return false; // ellipticity gradient only works for sources that have elliptical isophotes
	if (ellipticity_mode > 1) return false; // only emode=0 or 1 is supported right now
	
	SB_Spline_Params<double>& p = assign_sbspline_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	if ((egrad_mode==0) and (efunc_params[0]==-1e30)) { // in this case, the egrad params were never initialized
		// Not sure if I should do this here, or before calling enable_ellipticity_gradient?
		efunc_params.input(2*n_bspline_coefs+2);
		for (int i=0; i < n_bspline_coefs; i++) efunc_params[i] = p.q;
		for (int i=n_bspline_coefs; i < 2*n_bspline_coefs; i++) efunc_params[i] = radians_to_degrees(p.theta);
		efunc_params[2*n_bspline_coefs] = p.x_center;
		efunc_params[2*n_bspline_coefs+1] = p.y_center;
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
	delete[] sbparams->param;
	sbparams->param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
#ifdef USE_STAN
	delete[] sbparams_dif->param;
	sbparams_dif->param = new stan::math::var*[n_params];
	assign_param_pointers_autodif();
#endif
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

	delete[] sbparams->param;
	sbparams->param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
#ifdef USE_STAN
	delete[] sbparams_dif->param;
	sbparams_dif->param = new stan::math::var*[n_params];
	assign_param_pointers_autodif();
#endif
	assign_paramnames();
	update_ellipticity_meta_parameters<double>();
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

bool SB_Profile::enable_fourier_gradient(Vector<double>& fourier_params, const Vector<double>& knots, const bool copy_vary_settings, boolvector* vary_fgrad)
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
	delete[] sbparams->param;
	sbparams->param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
#ifdef USE_STAN
	delete[] sbparams_dif->param;
	sbparams_dif->param = new stan::math::var*[n_params];
	assign_param_pointers_autodif();
#endif
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

template <typename QScalar>
void SB_Profile::assign_param_pointers_impl()
{
	SB_Spline_Params<QScalar>& p = assign_sbspline_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.qx_parameter;
	p.param[1] = &p.f_parameter;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void SB_Profile::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void SB_Profile::assign_param_pointers_impl<stan::math::var>();
#endif

template <typename QScalar>
void SB_Profile::set_geometric_param_pointers(int qi)
{
	// Sets parameter pointers for ellipticity (or axis ratio) and angle
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	if (!ellipticity_gradient) {
		if (use_sb_ellipticity_components) {
			p.param[qi++] = &p.epsilon1;
			p.param[qi] = &p.epsilon2;
			angle_param[qi++] = false;
			angle_param_exists = false; // there is no angle parameter if ellipticity components are being used
		} else {
			p.param[qi++] = &p.q;
			p.param[qi] = &p.theta;
			angle_param[qi++] = true;
			angle_param_exists = true;
		}
		if (!lensed_center_coords) {
			p.param[qi++] = &p.x_center;
			p.param[qi++] = &p.y_center;
		} else {
			p.param[qi++] = &p.x_center_lensed;
			p.param[qi++] = &p.y_center_lensed;
		}

	} else {
		angle_param_exists = true;
		//set_geometric_param_pointers_egrad(param,angle_param,qi); // NOTE: if fourier_gradient is turned on, the Fourier parameter pointers are also set in this function
	}

	if ((!fourier_gradient) and (n_fourier_modes > 0)) {
		for (int i=0; i < n_fourier_modes; i++) {
			p.param[qi++] = &p.fourier_mode_cosamp[i];
			p.param[qi++] = &p.fourier_mode_sinamp[i];
		}
	}
	if (include_boxiness_parameter) p.param[qi++] = &p.c0;
	if (include_truncation_radius) p.param[qi++] = &p.rt;
}
template void SB_Profile::set_geometric_param_pointers<double>(int qi);
#ifdef USE_STAN
template void SB_Profile::set_geometric_param_pointers<stan::math::var>(int qi);
#endif

bool SB_Profile::vary_parameters(const boolvector& vary_params_in)
{
	if (vary_params_in.size() != n_params) {
		warn("vary params doesn't have the right size: %i vs %i",vary_params_in.size(),n_params);
		return false;
	}

	int pi, pf;
	if (qlens) qlens->get_sb_parameter_numbers(sb_number,pi,pf); // these are the old parameter numbers

	// Save the old limits, if they exist
	Vector<double> old_lower_limits(n_params);
	Vector<double> old_upper_limits(n_params);
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

	if (qlens) {
		if (pf > pi) qlens->param_list->remove_params(pi,pf);
		return qlens->register_sb_vary_parameters(sb_number);
	}

	return true;
}

bool SB_Profile::set_vary_flags(boolvector &vary_flags)
{
	boolvector new_vary_flags(n_params);
	if ((vary_flags.size() != n_params-1) and (vary_flags.size() != n_params)) return false;
	for (int i=0; i < vary_flags.size(); i++) new_vary_flags[i] = vary_flags[i];
	if (vary_flags.size() == n_params) new_vary_flags[n_params-1] = vary_flags[n_params-1];
	else new_vary_flags[n_params-1] = false; // if no vary flag is given for redshift, then assume it's not being varied
	if (vary_parameters(new_vary_flags)==false) return false;
	return true;
}

bool SB_Profile::update_specific_varyflag(const string name_in, const bool flag)
{
	int paramnum;
	if (!lookup_parameter_number(name_in,paramnum)) return false;
	boolvector new_vary_flags(vary_params);
	new_vary_flags[paramnum] = flag;
	if (vary_parameters(new_vary_flags)==false) return false;
	return true;
}

void SB_Profile::get_vary_flags(boolvector& vary_flags)
{
	vary_flags.input(n_params);
	for (int i=0; i < n_params; i++) {
		vary_flags[i] = vary_params[i];
	}
}

bool SB_Profile::register_vary_flags()
{
	// This function is called if there are already vary flags that have been set before adding the lens to the list
	if ((n_vary_params > 0) and (qlens != NULL))
		return qlens->register_sb_vary_parameters(sb_number);
	else return false;
}

void SB_Profile::set_limits(const Vector<double>& lower, const Vector<double>& upper)
{
	include_limits = true;
	if (lower.size() != n_vary_params) die("number of parameters with lower limits does not match number of variable parameters");
	if (upper.size() != n_vary_params) die("number of parameters with upper limits does not match number of variable parameters");
	lower_limits = lower;
	upper_limits = upper;
}

bool SB_Profile::set_limits_specific_parameter(const string name_in, const double& lower, const double& upper)
{
	if (n_vary_params==0) return false;
	int param_i = -1;
	int i,j;
	for (i=0,j=0; i < n_params; i++) {
		if (!vary_params[i]) continue;
		if (paramnames[i]==name_in) {
			param_i = j;
			break;
		}
		j++;
	}
	if (param_i != -1) {
		if (!include_limits) include_limits = true;
		lower_limits[param_i] = lower;
		upper_limits[param_i] = upper;
	}
	return (param_i != -1);
}

void SB_Profile::update_limits(const double* lower, const double* upper, const bool* limits_changed, int& index)
{
	// in this case, the limits are being updated from the fitparams list, so there is no need to call register_lens_prior_limits
	if (!include_limits) include_limits = true;
	for (int i=0; i < n_vary_params; i++) {
		if (limits_changed[index]) {
			lower_limits[i] = lower[index];
			upper_limits[i] = upper[index];
		}
		index++;
	}
}

void SB_Profile::get_parameters(double* params)
{
	SB_Params<double>& p = assign_sbparam_object<double>();
	for (int i=0; i < n_params; i++) {
		if (angle_param[i]) params[i] = radians_to_degrees(*(p.param[i]));
		else params[i] = *(p.param[i]);
	}
}

bool SB_Profile::check_parameter_name(const string name_in)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if (paramnames[i]==name_in) {
			found_match = true;
			break;
		}
	}
	return found_match;
}

bool SB_Profile::lookup_parameter_number(const string name_in, int& paramnum)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if (paramnames[i]==name_in) {
			found_match = true;
			paramnum = i;
			break;
		}
	}
	return found_match;
}

bool SB_Profile::get_specific_parameter(const string name_in, double& value)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if (paramnames[i]==name_in) {
			found_match = true;
			value = *(sbparams->param)[i];
			break;
		}
	}
	return found_match;
}

bool SB_Profile::get_specific_limit(const string name_in, double& lower, double& upper)
{
	if (include_limits==false) return false;
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if (paramnames[i]==name_in) {
			found_match = true;
			lower = lower_limits[i];
			upper = upper_limits[i];
			break;
		}
	}
	return found_match;
}

void SB_Profile::update_parameters(const double* params)
{
	SB_Params<double>& p = assign_sbparam_object<double>();
	for (int i=0; i < n_params; i++) {
		//cout << "Param " << i << ": " << params[i];
		//if (angle_param[i]) cout << " angle param" << endl;
		//else cout << " NOT angle param" << endl;
		if (angle_param[i]) *(p.param[i]) = degrees_to_radians(params[i]);
		else *(p.param[i]) = params[i];
	}
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
	if (qlens != NULL) {
		qlens->update_anchored_parameters_and_redshift_data();
		qlens->update_sb_fitparams(sb_number);
	}
	if (lensed_center_coords) set_center_if_lensed_coords<double>();
}

bool SB_Profile::update_specific_parameter(const string name_in, const double& value)
{
	SB_Params<double>& p = assign_sbparam_object<double>();
	bool found_match = false;
	int paramnum = -1;
	for (int i=0; i < n_params; i++) {
		if (paramnames[i]==name_in) {
			found_match = true;
			paramnum = i;
			break;
		}
	}
	if (found_match) {
		if (angle_param[paramnum]) *(p.param[paramnum]) = degrees_to_radians(value);
		else *(p.param[paramnum]) = value;
		update_meta_parameters();
		if (qlens != NULL) {
			qlens->update_anchored_parameters_and_redshift_data();
			qlens->update_sb_fitparams(sb_number);
		}
		if (lensed_center_coords) set_center_if_lensed_coords<double>();
#ifdef USE_STAN
		sync_autodif_parameters();
		update_meta_parameters_autodif();
#endif
	}
	else {
		if ((sbtype==SHAPELET) and (name_in=="n")) {
			update_indxptr(value);
			found_match = true;
		}
	}
	return found_match;
}

bool SB_Profile::update_specific_parameter(const int paramnum, const double& value)
{
	SB_Params<double>& p = assign_sbparam_object<double>();
	if (paramnum >= n_params) return false;
	//double* newparams = new double[n_params];
	//get_parameters(newparams);
	//newparams[paramnum] = value;
	if (angle_param[paramnum]) *(p.param[paramnum]) = degrees_to_radians(value);
	else *(p.param[paramnum]) = value;

	//update_parameters(newparams);
	//delete[] newparams;
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
	if (qlens != NULL) {
		qlens->update_anchored_parameters_and_redshift_data();
		qlens->update_sb_fitparams(sb_number);
	}

	if (lensed_center_coords) set_center_if_lensed_coords<double>();
	return true;
}

template <typename QScalar>
void SB_Profile::update_fit_parameters(const QScalar* fitparams, int &index, bool& status)
{
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	if (n_vary_params > 0) {
		for (int i=0; i < n_params; i++) {
			if (vary_params[i]==true) {
				if (angle_param[i]) {
					*(p.param[i]) = degrees_to_radians(fitparams[index++]);
					update_angle_meta_params<QScalar>();
				}
				else *(p.param[i]) = fitparams[index++];
			}
		}
#ifdef USE_STAN
		// if using autodif params, let's update the non-autodiff params too (or vice versa) for consistency. Maybe revisit this later? Might not be necessary
		if constexpr (std::is_same_v<QScalar, stan::math::var>) {
			for (int i=0; i < n_params; i++) {
				if (vary_params[i]==true) {
					*(sbparams->param[i]) = (*(sbparams_dif->param[i])).val();
				}
			}
		} else {
			for (int i=0; i < n_params; i++) {
				if (vary_params[i]==true) {
					*(sbparams_dif->param[i]) = (*(sbparams->param[i]));
				}
			}
		}
		update_meta_parameters_autodif();
#endif
		update_meta_parameters();
	}
}
template void SB_Profile::update_fit_parameters<double>(const double* fitparams, int &index, bool& status);
#ifdef USE_STAN
template void SB_Profile::update_fit_parameters<stan::math::var>(const stan::math::var* fitparams, int &index, bool& status);
#endif

void SB_Profile::update_anchored_parameters()
{
#ifdef USE_STAN
	using stan::math::pow;
#endif
	bool at_least_one_param_anchored = false;
	for (int i=0; i < n_params; i++) {
		if (anchor_parameter_to_source[i]) {
			(*sbparams->param[i]) = parameter_anchor_ratio[i]*pow(*(parameter_anchor_source[i]->sbparams->param[parameter_anchor_paramnum[i]]),parameter_anchor_exponent[i]);
#ifdef USE_STAN
			(*sbparams_dif->param[i]) = parameter_anchor_ratio[i]*pow(*(parameter_anchor_source[i]->sbparams_dif->param[parameter_anchor_paramnum[i]]),parameter_anchor_exponent[i]);
#endif
			if (at_least_one_param_anchored==false) at_least_one_param_anchored = true;
		}
	}
	if (at_least_one_param_anchored) {
		update_meta_parameters();
#ifdef USE_STAN
		update_meta_parameters_autodif();
#endif
	}
}

bool SB_Profile::update_anchored_parameters_to_source(const int src_i)
{
	bool at_least_one_param_anchored = false;
	for (int i=0; i < n_params; i++) {
		if ((anchor_parameter_to_source[i]) and (parameter_anchor_source[i]->sb_number==src_i)) {
			(*sbparams->param[i]) = parameter_anchor_ratio[i]*pow(*(parameter_anchor_source[i]->sbparams->param[parameter_anchor_paramnum[i]]),parameter_anchor_exponent[i]);
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
		sbparams->x_center = center_anchor_lens->lensparams->x_center;
		sbparams->y_center = center_anchor_lens->lensparams->y_center;
#ifdef USE_STAN
		sbparams_dif->x_center = center_anchor_lens->lensparams->x_center;
		sbparams_dif->y_center = center_anchor_lens->lensparams->y_center;
#endif
	} else if (center_anchored_to_source) {
		sbparams->x_center = center_anchor_source->sbparams->x_center;
		sbparams->y_center = center_anchor_source->sbparams->y_center;
#ifdef USE_STAN
		sbparams_dif->x_center = center_anchor_source->sbparams_dif->x_center;
		sbparams_dif->y_center = center_anchor_source->sbparams_dif->y_center;
#endif
	} else if (center_anchored_to_ptsrc) {
		sbparams->x_center = center_anchor_ptsrc->ptsrc_params.pos[0];
		sbparams->y_center = center_anchor_ptsrc->ptsrc_params.pos[1];
#ifdef USE_STAN
		sbparams_dif->x_center = center_anchor_ptsrc->ptsrc_params.pos[0];
		sbparams_dif->y_center = center_anchor_ptsrc->ptsrc_params.pos[1];
#endif
	}
}

void SB_Profile::get_fit_parameters(double *fitparams, int &index)
{
	SB_Params<double>& p = assign_sbparam_object<double>();
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]==true) {
			if (angle_param[i]) fitparams[index++] = radians_to_degrees(*(p.param[i]));
			else fitparams[index++] = *(p.param[i]);
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

void SB_Profile::get_auto_stepsizes(Vector<double>& stepsizes_in, int &index)
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

void SB_Profile::get_auto_ranges(boolvector& use_penalty_limits, Vector<double>& lower, Vector<double>& upper, int &index)
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

bool SB_Profile::get_limits(Vector<double>& lower, Vector<double>& upper, int &index)
{
	if ((include_limits==false) or (lower_limits.size() != n_vary_params)) return false;
	for (int i=0; i < n_vary_params; i++) {
		lower[index] = lower_limits[i];
		upper[index] = upper_limits[i];
		index++;
	}
	return true;
}

bool SB_Profile::get_limits(Vector<double>& lower, Vector<double>& upper)
{
	if ((include_limits==false) or (lower.size() != n_vary_params)) return false;
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
	if (paramnum >= n_params) die("Parameter %i does not exist for this source (nparams=%i)",paramnum,n_params);
	if (anchor_paramnum >= param_anchor_source->n_params) die("Parameter does not exist for source you are anchoring to");
	anchor_parameter_to_source[paramnum] = true;
	parameter_anchor_source[paramnum] = param_anchor_source;
	parameter_anchor_paramnum[paramnum] = anchor_paramnum;
	if ((!use_implicit_ratio) and (!use_exponent)) {
		parameter_anchor_ratio[paramnum] = 1.0;
		(*sbparams->param[paramnum]) = *(param_anchor_source->sbparams->param[anchor_paramnum]);
	}
	else if (use_implicit_ratio) {
		parameter_anchor_exponent[paramnum] = 1.0;
		if ((*(param_anchor_source->sbparams->param[anchor_paramnum]))==0) {
			if (*sbparams->param[paramnum]==0) parameter_anchor_ratio[paramnum] = 1.0;
			else die("cannot anchor to parameter with specified ratio if parameter is equal to zero");
		} else {
			parameter_anchor_ratio[paramnum] = (*sbparams->param[paramnum]) / (*(param_anchor_source->sbparams->param[anchor_paramnum]));
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

template <typename QScalar>
void SB_Profile::set_geometric_parameters(const QScalar &q1_in, const QScalar &q2_in, const QScalar &xc_in, const QScalar &yc_in, const QScalar &zsrc_in)
{
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();

	if (use_sb_ellipticity_components) {
		p.epsilon1 = q1_in;
		p.epsilon2 = q2_in;
	} else {
		p.q = q1_in;
		if (p.q < 0) p.q = -p.q; // don't allow negative axis ratios
		if (p.q > 1) p.q = 1.0; // don't allow q>1
		p.theta = degrees_to_radians(q2_in);
	}
	if (!lensed_center_coords) {
		p.x_center = xc_in;
		p.y_center = yc_in;
	} else {
		p.x_center_lensed = xc_in;
		p.y_center_lensed = yc_in;
		set_center_if_lensed_coords<QScalar>();
	}
	p.zsrc = zsrc_in;

	update_ellipticity_meta_parameters<QScalar>();
}
template void SB_Profile::set_geometric_parameters<double>(const double &q1_in, const double &q2_in, const double &xc_in, const double &yc_in, const double &zsrc_in);
#ifdef USE_STAN
template void SB_Profile::set_geometric_parameters<stan::math::var>(const stan::math::var &q1_in, const stan::math::var &q2_in, const stan::math::var &xc_in, const stan::math::var &yc_in, const stan::math::var &zsrc_in);
#endif

#ifdef USE_STAN
void SB_Profile::sync_autodif_geometric_parameters()
{
	sbparams_dif->zsrc = sbparams->zsrc;
	if (use_sb_ellipticity_components) {
		sbparams_dif->epsilon1 = sbparams->epsilon1;
		sbparams_dif->epsilon2 = sbparams->epsilon2;
	}
	sbparams_dif->q = sbparams->q;
	sbparams_dif->theta = sbparams->theta;
	sbparams_dif->x_center = sbparams->x_center;
	sbparams_dif->y_center = sbparams->y_center;
	if (lensed_center_coords) {
		sbparams_dif->x_center_lensed = sbparams->x_center_lensed;
		sbparams_dif->y_center_lensed = sbparams->y_center_lensed;
	}
	if ((!fourier_gradient) and (n_fourier_modes > 0)) {
		for (int i=0; i < n_fourier_modes; i++) {
			sbparams_dif->fourier_mode_cosamp[i] = sbparams->fourier_mode_cosamp[i];
			sbparams_dif->fourier_mode_sinamp[i] = sbparams->fourier_mode_cosamp[i];
		}
	}
}
#endif

template <typename QScalar>
void SB_Profile::set_geometric_parameters_radians(const QScalar &q_in, const QScalar &theta_in, const QScalar &xc_in, const QScalar &yc_in)
{
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	p.q=q_in;
	if (p.q < 0) p.q = -p.q; // don't allow negative axis ratios
	if (p.q > 1) p.q = 1.0; // don't allow q>1
	set_angle_radians(theta_in);
	p.x_center = xc_in;
	p.y_center = yc_in;
}
template void SB_Profile::set_geometric_parameters_radians<double>(const double &q_in, const double &theta_in, const double &xc_in, const double &yc_in);
#ifdef USE_STAN
template void SB_Profile::set_geometric_parameters_radians<stan::math::var>(const stan::math::var &q_in, const stan::math::var &theta_in, const stan::math::var &xc_in, const stan::math::var &yc_in);
#endif

template <typename QScalar>
void SB_Profile::set_center_if_lensed_coords()
{
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	if (lensed_center_coords) {
		if (qlens==NULL) die("Cannot use lensed center coordinates if pointer to QLens object hasn't been assigned");
		lensvector<QScalar> xl;
		xl[0] = p.x_center_lensed;
		xl[1] = p.y_center_lensed;
		qlens->find_sourcept<QScalar>(xl,p.x_center,p.y_center,0,qlens->reference_zfactors,qlens->default_zsrc_beta_factors);
	}
}
template void SB_Profile::set_center_if_lensed_coords<double>();
#ifdef USE_STAN
template void SB_Profile::set_center_if_lensed_coords<stan::math::var>();
#endif

template <typename QScalar>
void SB_Profile::calculate_ellipticity_components()
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
#endif
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	if (use_sb_ellipticity_components) {
		QScalar theta_eff = (orient_major_axis_north) ? p.theta + M_HALFPI : p.theta;
		p.epsilon1 = (1-p.q)*cos(2*theta_eff);
		p.epsilon2 = (1-p.q)*sin(2*theta_eff);
	}
}
template void SB_Profile::calculate_ellipticity_components<double>();
#ifdef USE_STAN
template void SB_Profile::calculate_ellipticity_components<stan::math::var>();
#endif

template <typename QScalar>
void SB_Profile::update_ellipticity_meta_parameters()
{
#ifdef USE_STAN
	using stan::math::sqrt;
#endif
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	if (!ellipticity_gradient) {
		if (use_sb_ellipticity_components) {
			p.q = 1 - sqrt(SQR(p.epsilon1) + SQR(p.epsilon2));
			set_angle_from_components(p.epsilon1,p.epsilon2); // note this will automatically set the costheta, sintheta parameters
		} else {
			update_angle_meta_params<QScalar>(); // sets the costheta, sintheta meta-parameters
		}
	} else {
		/*
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
		*/
	}
}
template void SB_Profile::update_ellipticity_meta_parameters<double>();
#ifdef USE_STAN
template void SB_Profile::update_ellipticity_meta_parameters<stan::math::var>();
#endif

template <typename QScalar>
void SB_Profile::update_angle_meta_params()
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
#endif
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	// trig functions are stored to save computation time later
	p.costheta = cos(p.theta);
	p.sintheta = sin(p.theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		QScalar tmp = p.sintheta;
		p.sintheta = p.costheta;
		p.costheta = -tmp;
	}
}
template void SB_Profile::update_angle_meta_params<double>();
#ifdef USE_STAN
template void SB_Profile::update_angle_meta_params<stan::math::var>();
#endif

template <typename QScalar>
QScalar SB_Profile::sb_rsq_impl(const QScalar rsq) // this function should be redefined in all derived classes
{
#ifdef USE_STAN
	using stan::math::sqrt;
#endif
	SB_Spline_Params<QScalar>& p = assign_sbspline_param_object<QScalar>();
	QScalar r = sqrt(rsq);
	if (r < p.qx_parameter*p.sb_spline.xmin()) return (p.f_parameter*p.sb_spline.extend_inner_logslope(r/p.qx_parameter));
	if (r > p.qx_parameter*p.sb_spline.xmax()) return (p.f_parameter*p.sb_spline.extend_outer_logslope(r/p.qx_parameter));
	return (p.f_parameter*p.sb_spline.splint(r/p.qx_parameter));
}
template double SB_Profile::sb_rsq_impl<double>(const double rsq); // this function should be redefined in all derived classes
#ifdef USE_STAN
template stan::math::var SB_Profile::sb_rsq_impl<stan::math::var>(const stan::math::var rsq); // this function should be redefined in all derived classes
#endif

template <typename QScalar>
QScalar SB_Profile::sb_rsq_deriv_impl(const QScalar rsq)
{
	SB_Spline_Params<QScalar>& p = assign_sbspline_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	static const QScalar precision = 1e-6;
	QScalar temp, h;
	h = precision*rsq;
	temp = rsq + h;
	h = temp - rsq; // silly NR trick
	return (sb_rsq((rsq+h)/(p.qx_parameter*p.qx_parameter))-sb_rsq((rsq-h)/(p.qx_parameter*p.qx_parameter)))/(2*h);
}
template double SB_Profile::sb_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var SB_Profile::sb_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

void SB_Profile::shift_angle_90()
{
	// do this if the major axis orientation is changed (so the qlens angles values are changed appropriately, even though the qlens doesn't change)
	sbparams->theta += M_HALFPI;
	while (sbparams->theta > M_PI) sbparams->theta -= M_PI;
}

void SB_Profile::shift_angle_minus_90()
{
	// do this if the major axis orientation is changed (so the qlens angles values are changed appropriately, even though the qlens doesn't change)
	sbparams->theta -= M_HALFPI;
	while (sbparams->theta <= -M_PI) sbparams->theta += M_PI;
}

void SB_Profile::reset_angle_modulo_2pi()
{
	while (sbparams->theta < -M_PI/2) sbparams->theta += 2*M_PI;
	while (sbparams->theta > 2*M_PI) sbparams->theta -= 2*M_PI;
}

template <typename QScalar>
void SB_Profile::set_angle(const QScalar &theta_degrees)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
#endif
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	p.theta = degrees_to_radians(theta_degrees);
	// trig functions are stored to save computation time later
	p.costheta = cos(p.theta);
	p.sintheta = sin(p.theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		QScalar tmp = p.sintheta;
		p.sintheta = p.costheta;
		p.costheta = -tmp;
	}
}
template void SB_Profile::set_angle<double>(const double &theta_degrees);
#ifdef USE_STAN
template void SB_Profile::set_angle<stan::math::var>(const stan::math::var &theta_degrees);
#endif

template <typename QScalar>
void SB_Profile::set_angle_radians(const QScalar &theta_in)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
#endif
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	p.theta = theta_in;
	// trig functions are stored to save computation time later
	p.costheta = cos(p.theta);
	p.sintheta = sin(p.theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		QScalar tmp = p.sintheta;
		p.sintheta = p.costheta;
		p.costheta = -tmp;
	}
}
template void SB_Profile::set_angle_radians<double>(const double &theta_in);
#ifdef USE_STAN
template void SB_Profile::set_angle_radians<stan::math::var>(const stan::math::var &theta_in);
#endif

template <typename QScalar>
void SB_Profile::set_angle_from_components(QScalar &comp1, const QScalar &comp2)
{
#ifdef USE_STAN
	using stan::math::atan2;
	if ((comp1==0) and (comp2==0)) comp1 += 1e-10; // for autodiff purposes, to avoid coordinate singularity
#endif
	QScalar angle = atan2(comp2,comp1);
	angle = 0.5*angle;
	if (orient_major_axis_north) angle -= M_HALFPI;
	//while (angle > M_HALFPI) angle -= M_PI;
	//while (angle <= -M_HALFPI) angle += M_PI;
	while (angle > M_PI) angle -= M_PI;
	while (angle <= 0) angle += M_PI;
	set_angle_radians<QScalar>(angle);
}
template void SB_Profile::set_angle_from_components<double>(double &comp1, const double &comp2);
#ifdef USE_STAN
template void SB_Profile::set_angle_from_components<stan::math::var>(stan::math::var &comp1, const stan::math::var &comp2);
#endif

template <typename QScalar>
void SB_Profile::rotate(QScalar &x, QScalar &y)
{
	// perform a counter-clockwise rotation of the coordinate system to match the coordinate system of the rotated galaxy
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	QScalar xp = x*p.costheta + y*p.sintheta;
	y = -x*p.sintheta + y*p.costheta;
	x = xp;
}
template void SB_Profile::rotate<double>(double &x, double &y);
#ifdef USE_STAN
template void SB_Profile::rotate<stan::math::var>(stan::math::var &x, stan::math::var &y);
#endif

template <typename QScalar>
void SB_Profile::rotate_back(QScalar &x, QScalar &y)
{
	// perform a counter-clockwise rotation of the coordinate system to match the coordinate system of the rotated galaxy
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	QScalar xp = x*p.costheta - y*p.sintheta;
	y = x*p.sintheta + y*p.costheta;
	x = xp;
}
template void SB_Profile::rotate_back<double>(double &x, double &y);
#ifdef USE_STAN
template void SB_Profile::rotate_back<stan::math::var>(stan::math::var &x, stan::math::var &y);
#endif

template <typename QScalar>
QScalar SB_Profile::surface_brightness_impl(QScalar x, QScalar y)
{
#ifdef USE_STAN
	using stan::math::abs;
	using stan::math::sqrt;
	using stan::math::atan;
	using stan::math::pow;
#endif
	// switch to coordinate system centered on surface brightness profile
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	x -= p.x_center;
	y -= p.y_center;
	if ((!ellipticity_gradient) and (p.theta != 0)) rotate(x,y);

	QScalar xisq, rsq;
	QScalar fourier_factor = 0.0;

	rsq = x*x + y*y;
	if (!ellipticity_gradient) {
		QScalar rsq_ell = x*x + y*y/(p.q*p.q);
		if (ellipticity_mode==1) rsq_ell *= p.q;
		if ((include_boxiness_parameter) and (p.c0 != 0.0)) {
			if (ellipticity_mode==0)
				xisq = pow(pow(abs(x),p.c0+2.0) + pow(abs(y/p.q),p.c0+2.0),2.0/(p.c0+2.0));
			else
				xisq = pow(pow(abs(x)*sqrt(p.q),p.c0+2.0) + pow(abs(y/sqrt(p.q)),p.c0+2.0),2.0/(p.c0+2.0));
		} else {
			xisq = rsq_ell;
		}
		if (n_fourier_modes > 0) {
			QScalar phi_q; // used for Fourier modes
			if (fourier_use_eccentric_anomaly) phi_q = atan(y/(p.q*x));
			else phi_q = atan(y/x); // for SB perturbations, we don't use eccentric anomaly as our angle because the corresponding lensing multipoles can't either
				// Check the angle below!!!! Shouldn't you use angle c.c. from x-axis here? (comp to ellipse sampling angle in pixelgrid.cpp)
			if (x < 0) phi_q += M_PI;
			else if (y < 0) phi_q += M_2PI;

			if (!fourier_sb_perturbation) fourier_factor = 1.0;
			if (use_fmode_scaled_amplitudes) {
				for (int i=0; i < n_fourier_modes; i++) {
					fourier_factor += (p.fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*phi_q) + p.fourier_mode_sinamp[i]*sin(fourier_mode_mvals[i]*phi_q))/fourier_mode_mvals[i];
				}
			} else {
				for (int i=0; i < n_fourier_modes; i++) {
					fourier_factor += p.fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*phi_q) + p.fourier_mode_sinamp[i]*sin(fourier_mode_mvals[i]*phi_q);
				}
			}
			if (!fourier_sb_perturbation) xisq *= fourier_factor*fourier_factor;
		}
	} else {
		/*
		QScalar xi = elliptical_radius_root(x,y);
		xisq = SQR(xi);
		if ((n_fourier_modes > 0) and (fourier_sb_perturbation)) {
			QScalar ep, phi0;
			if (fourier_use_eccentric_anomaly) ellipticity_function(xi,ep,phi0);
			else ellipticity_function(sqrt(rsq),ep,phi0); // lensing multipoles depend on r, not xi, so we follow the same restriction here

			QScalar costh, sinth, xp, yp, qq, phi_q;
			costh = cos(phi0);
			sinth = sin(phi0);
			xp = x*costh + y*sinth;
			yp = -x*sinth + y*costh;
			qq = sqrt(1-ep);

			if (fourier_use_eccentric_anomaly) phi_q = atan(yp/(qq*xp));
			else phi_q = atan(yp/xp);
			if (xp < 0) phi_q += M_PI;
			else if (yp < 0) phi_q += M_2PI;

			QScalar *cosamps;
			QScalar *sinamps;
			if (fourier_gradient) {
				cosamps = new QScalar[n_fourier_modes];
				sinamps = new QScalar[n_fourier_modes];
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
		*/
	}
	QScalar sb = sb_rsq(xisq);
	if ((n_fourier_modes > 0) and (fourier_sb_perturbation)) {
		// create virtual sb_rsq_deriv function in base class, and versions for all inherited classes, so you don't have to do this numerically for analytic models
		QScalar sbderiv, h = 1e-5;

		if (fourier_use_eccentric_anomaly) {
			if (xisq <= h) sbderiv = (sb_rsq(xisq + h) - sb_rsq(xisq))/(h);
			else sbderiv = (sb_rsq(xisq + h) - sb_rsq(xisq-h))/(2*h);
			sb += 2*fourier_factor*sbderiv*xisq; // this allows it to approximate perturbing the elliptical radius (via first order term in Taylor expansion in (r + dr))
		} else {
			// we evaluate SB at non-elliptical radius because that's what the corresponding lensing multipoles have to do (to get deflections).
			if (rsq <= h) sbderiv = (sb_rsq(rsq + h) - sb_rsq(rsq))/(h);
			else sbderiv = (sb_rsq(rsq + h) - sb_rsq(rsq-h))/(2*h);
			sb += 2*fourier_factor*sbderiv*rsq; // this allows it to approximate perturbing the elliptical radius (via first order term in Taylor expansion in (r + dr))
		}
		//sb += fourier_factor; // for testing purposes
	}
	if (include_truncation_radius) sb *= pow(1+pow(xisq/(p.rt*p.rt),3),-2);

	if (sb*0.0 != 0.0) warn("surface brightness returning NAN");

	return sb;
}
template double SB_Profile::surface_brightness_impl<double>(double x, double y);
#ifdef USE_STAN
template stan::math::var SB_Profile::surface_brightness_impl<stan::math::var>(stan::math::var x, stan::math::var y);
#endif

template <typename QScalar>
QScalar SB_Profile::surface_brightness_zoom(lensvector<QScalar> &centerpt, lensvector<QScalar> &pt1, lensvector<QScalar> &pt2, lensvector<QScalar> &pt3, lensvector<QScalar> &pt4, const QScalar sb_noise)
{
	SB_Params<QScalar>& p = assign_sbparam_object<QScalar>();
	bool subgrid = false;
	bool contains_sbcenter = false;
	int xsplit, ysplit;

	lensvector<QScalar> sbcenter;
	sbcenter[0] = p.x_center;
	sbcenter[1] = p.y_center;

	lensvector<QScalar> d1, d2, d3;
	QScalar product1, product2, product3;
	QScalar r[4];
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
	QScalar rmin = 1e30, rmax = -1e30;
	for (int i=0; i < 4; i++) {
		if (r[i] > rmax) rmax = r[i];
		if (r[i] < rmin) rmin = r[i];
	}

	if (contains_sbcenter) rmin = 0;
	//d1[0] = centerpt[0]-sbcenter[0];
	//d1[1] = centerpt[1]-sbcenter[1];
	//QScalar center_r = d1.norm();

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
			const QScalar epsilon = 1; // in principle we could allow the user to change this to make the subgridding more aggressive. Implement later?
			QScalar rminsq=rmin*rmin, rmaxsq=rmax*rmax;
			QScalar sbderiv1, sbderiv2, h = 1e-5;
			if (rminsq <= h) sbderiv1 = 2*rmin*(sb_rsq(rminsq + h) - sb_rsq(rminsq))/(h);
			else sbderiv1 = 2*rmin*(sb_rsq(rminsq + h) - sb_rsq(rminsq-h))/(2*h);
			if (rmaxsq <= h) sbderiv2 = 2*rmax*(sb_rsq(rmaxsq + h) - sb_rsq(rmaxsq))/(h);
			else sbderiv2 = 2*rmax*(sb_rsq(rmaxsq + h) - sb_rsq(rmaxsq-h))/(2*h);
			QScalar sbcurv_approx = sbderiv2-sbderiv1;
			QScalar optimal_scale = 4*epsilon*sb_noise/sbcurv_approx;
			// Ideally, if the pixel size is greater than optimal scale, you'd increase the splittings, calculate sbcurv_approx again, and iterate until
			// pixel size is small enough. But this seems to work well enough as it is.
			QScalar npix_approx = (rmax-rmin)/optimal_scale;
			if (npix_approx > 1) {
				subgrid = true;
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					xsplit = ((int) npix_approx.val()) + 1;
					ysplit = ((int) npix_approx.val()) + 1;
				} else
#endif
				{
					xsplit = ((int) npix_approx) + 1;
					ysplit = ((int) npix_approx) + 1;
				}
				if (xsplit > max_split) {
					//cout << "xsplit wants to be " << xsplit << endl;
					xsplit = max_split; // limit on number of splittings
				}
				if (ysplit > max_split) {
					//cout << "ysplit wants to be " << ysplit << endl;
					ysplit = max_split; // limit on number of splittings
				}

				//QScalar xavg = (pt1[0] + pt2[0] + pt3[0] + pt4[0])/4;
				//QScalar yavg = (pt1[1] + pt2[1] + pt3[1] + pt4[1])/4;
				//cout << "xsplit= " << xsplit << " ysplit=" << ysplit << " (x=" << xavg << ",y=" << yavg << ",maxsplit=" << max_split << ")" << endl;
			}
			//if (npix_approx > 1) cout << "r= " << center_r << " sbcurv_approx=" << sbcurv_approx << " delta_R=" << (rmax-rmin) << " OPTIMAL SCALE: " << optimal_scale << " npix_approx=" << npix_approx << endl;
		}
	} else {
		// The following algorithm is for source pixels that are large compared to the half-light radius of the SB profile
		// Revisit this later? Seems a bit shoddy, and not very trustworthy for lensed sources, but maybe good enough for unlensed sources.
		QScalar scale = zoom_scale*length_scale();
		lensvector<QScalar> scpt1, scpt2, scpt3, scpt4;
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
			QScalar pixel_xlength, pixel_ylength;
			d1 = pt2 - pt1;
			d2 = pt4 - pt3;
			pixel_xlength = maxval(d1.norm(),d2.norm());
			d1 = pt3 - pt1;
			d2 = pt4 - pt2;
			pixel_ylength = maxval(d1.norm(),d2.norm());
			QScalar lscale = 0.01*length_scale();
#ifdef USE_STAN
			if constexpr (std::is_same_v<QScalar, stan::math::var>) {
				xsplit = ((int) (zoom_split_factor*pixel_xlength/lscale).val()) + 1;
				ysplit = ((int) (zoom_split_factor*pixel_ylength/lscale).val()) + 1;
			} else
#endif
			{
				xsplit = ((int) (zoom_split_factor*pixel_xlength/lscale)) + 1;
				ysplit = ((int) (zoom_split_factor*pixel_ylength/lscale)) + 1;
			}
		}
	}
	if (!subgrid) return surface_brightness(centerpt[0],centerpt[1]);
	QScalar sb = 0;
	QScalar u0, w0, xs, ys;
	int ii,jj;

	// This splitting algorithm allows for the 'pixel' to not be rectangular, as in lensed sources. But it's still dicey for lensed 
	// sources since the subpixels in the source plane won't look exactly the same as splitting in the image plane and mapping each
	// subpixel to the source plane. It's always better (but costlier) to split the image pixels and then ray trace the subpixels to
	// the source plane.
	for (ii=0; ii < xsplit; ii++) {
		u0 = ((QScalar) (1+2*ii))/(2*xsplit);
		for (jj=0; jj < ysplit; jj++) {
			w0 = ((QScalar) (1+2*jj))/(2*ysplit);
			xs = (pt1[0]*u0 + pt2[0]*(1-u0))*w0 + (pt3[0]*u0 + pt4[0]*(1-u0))*(1-w0);
			ys = (pt1[1]*u0 + pt2[1]*(1-u0))*w0 + (pt3[1]*u0 + pt4[1]*(1-u0))*(1-w0);
			sb += surface_brightness(xs,ys);
		}
	}
	sb /= (xsplit*ysplit);

	return sb;
}
template double SB_Profile::surface_brightness_zoom<double>(lensvector<double> &centerpt, lensvector<double> &pt1, lensvector<double> &pt2, lensvector<double> &pt3, lensvector<double> &pt4, const double sb_noise);
#ifdef USE_STAN
template stan::math::var SB_Profile::surface_brightness_zoom<stan::math::var>(lensvector<stan::math::var> &centerpt, lensvector<stan::math::var> &pt1, lensvector<stan::math::var> &pt2, lensvector<stan::math::var> &pt3, lensvector<stan::math::var> &pt4, const stan::math::var sb_noise);
#endif

template <typename QScalar>
QScalar SB_Profile::surface_brightness_r(const QScalar r)
{
	return sb_rsq(r*r);
}
template double SB_Profile::surface_brightness_r<double>(const double r);
#ifdef USE_STAN
template stan::math::var SB_Profile::surface_brightness_r<stan::math::var>(const stan::math::var r);
#endif

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
		LogLikePtr = static_cast<double (UCMC::*)(const double*)> (&SB_Profile::sbprofile_loglike);
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
		double (Simplex::*loglikeptr)(const double*);
		loglikeptr = static_cast<double (Simplex::*)(const double*)> (&SB_Profile::sbprofile_loglike);
		double *stepsizes = new double [sbprofile_nparams];
		for (int i=0; i < sbprofile_nparams; i++) {
			stepsizes[i] = 0.1; // arbitrary
			fitparams[i] = *(sbparams->param[i]);
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

double SB_Profile::sbprofile_loglike(const double *params)
{
	int i;
	double loglike=0;
	for (i=0; i < sbprofile_nparams; i++) {
		if ((set_auto_penalty_limits[i]) and ((params[i] < penalty_lower_limits[i]) or (params[i] > penalty_upper_limits[i]))) {
			//cout << "PENALTY! param " << i << " = " << params[i] << endl;
			return 1e30; // penalty prior
		}
		*(sbparams->param[i]) = params[i];
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
		LogLikePtr = static_cast<double (UCMC::*)(const double*)> (&SB_Profile::profile_fit_loglike);
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
			double (Simplex::*loglikeptr)(const double*);
			if (egrad_mode==0) {
				loglikeptr = static_cast<double (Simplex::*)(const double*)> (&SB_Profile::profile_fit_loglike_bspline);
				//cout << "Fitting " << profile_fit_nparams << " independent knot intervals" << endl;
				for (i=0; i < profile_fit_nparams; i++) {
					stepsizes[i] = fitparams[i]/4.0; // arbitrary
				}
				double min_data_interval = 1e30;
				for (i=0; i < n_isophote_datapts-1; i++) {
					if ((profile_fit_logxivals[i+1]-profile_fit_logxivals[i]) < min_data_interval) min_data_interval = (profile_fit_logxivals[i+1]-profile_fit_logxivals[i]);
				}
				// the minimum knot interval allowed is given in terms of a specified fraction of the spacing between data points
				profile_fit_min_knot_interval = min_data_interval;
				if (mpi_id==0) cout << "min knot interval: " << profile_fit_min_knot_interval << endl;
			} else {
				loglikeptr = static_cast<double (Simplex::*)(const double*)> (&SB_Profile::profile_fit_loglike);
				for (i=profile_fit_istart, j=0; j < profile_fit_nparams; i++, j++) {
					if (angle_param[i]) stepsizes[j] = 20;
					else stepsizes[j] = 0.1;
					fitparams[j] = *(sbparams->param[i]);
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
	if (egrad_param==0) update_ellipticity_meta_parameters<double>(); // since the q-parameters have changed

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

double SB_Profile::profile_fit_loglike(const double *params)
{
	int i,j;
	double loglike=0;
	for (i=profile_fit_istart, j=0; j < profile_fit_nparams; i++, j++) {
		if (angle_param[i]) *(sbparams->param[i]) = degrees_to_radians(params[j]);
		else *(sbparams->param[i]) = params[j];
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

double SB_Profile::profile_fit_loglike_bspline(const double *params)
{
	// here, the nonlinear parameters are the knots that are being optimized
	int i;
	double tot_interval = 0;
	//cout <<  "Current knot intervals: " << endl;
	for (i=0; i < n_bspline_knots_tot-2*bspline_order-1; i++) {
		//cout << params[i] << " ";
		if (abs(params[i]) < profile_fit_min_knot_interval) { warn("knot interval too small; skipping B-spline fit"); return 1e30; }
		tot_interval += abs(params[i]);
		profile_fit_egrad_params[bspline_order+i+1] = profile_fit_egrad_params[bspline_order+i] + abs(params[i]);
	}
	//cout << "total interval: " << tot_interval << " ";
	for (i=0; i < bspline_order; i++) profile_fit_egrad_params[n_bspline_knots_tot-bspline_order+i] = profile_fit_egrad_params[n_bspline_knots_tot-bspline_order-1];

	//update_meta_parameters(); // this isn't really necessary now, but will become necessary if parameter transformations are made, e.g. ellipticity components
	if (tot_interval > 1.1*(log(xi_final_egrad/xi_initial_egrad)/ln10)) { warn("total interval too large (log: %g) (linear: %g); skipping B-spline fit",tot_interval,pow(10,tot_interval)); return 1e30; } // penalty chisq

	double loglike = fit_bspline_curve(profile_fit_egrad_params,profile_fit_bspline_coefs);
	//cout << "loglike=" << loglike << endl;
	return loglike;
}

/*
double SB_Profile::calculate_Lmatrix_element(double x, double y, const int amp_index)
{
	return 0.0; // this is only used in the derived classes Shapelet, MGE
}
*/

void SB_Profile::calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight)
{
	return; // this is only used in the derived classes Shapelet, MGE
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

void SB_Profile::calculate_curvature_Rmatrix_elements_rvals(double *rvalsq, const int n_rvals, double* Rmatrix_elements)
{
	return; // this is only used in the derived class MGE (but may be used by more profiles later)
}

void SB_Profile::get_regularization_param_ptr(double*& regparam_ptr)
{
	return; // this is only used in the derived class Shapelet (but may be used by more profiles later)
}

void SB_Profile::update_amplitudes(double*& ampvec)
{
	return; // this is only used in the derived classes Shapelet, MGE
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

void SB_Profile::print_parameters(const bool show_band, const int band)
{
	ios_base::fmtflags current_flags = cout.flags();
	if (current_flags & ios::scientific) cout << resetiosflags(ios::scientific);
	cout << model_name;
	bool parenthesis = false;
	string divider = "(";
	if (!is_lensed) {
		cout << "(unlensed";
		parenthesis = true;
	} else if (sbparams->zsrc > 0) {
		stringstream zstr;
		zstr << sbparams->zsrc;
		string zstring;
		zstr >> zstring;
		cout << "(zs=" << zstring;
		parenthesis = true;
	}
	if (parenthesis) divider = ",";
	
	if (sbtype==SHAPELET) { cout << divider << "n_shapelets=" << (*indxptr); parenthesis = true; }
	else if (sbtype==MULTI_GAUSSIAN_EXPANSION) { cout << divider << "n_gaussians=" << (*indxptr); parenthesis = true; }
	if (zoom_subgridding) { cout << divider << "zoom"; parenthesis = true; }
	if (parenthesis) cout << ")";
	cout << ": ";
	for (int i=0; i < n_params; i++) {
		cout << paramnames[i] << "=";
		if (angle_param[i]) cout << radians_to_degrees(*(sbparams->param[i])) << " degrees";
		else cout << *(sbparams->param[i]);
		if (i != n_params-1) cout << ", ";
	}
	if (center_anchored_to_lens) cout << " (center anchored to lens " << center_anchor_lens->lens_number << ")";
	else if (center_anchored_to_source) cout << " (center anchored to source " << center_anchor_source->sb_number << ")";
	else if (center_anchored_to_ptsrc) cout << " (center anchored to ptsrc " << center_anchor_ptsrc->entry_number << ")";
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
	if (lensed_center_coords) cout << " (xc=" << sbparams->x_center << ", yc=" << sbparams->y_center << ")";
	if (show_band) {
		cout << " (band=" << band << ")";
	}
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
	if (current_flags & ios::scientific) cout << setiosflags(ios::scientific);
}

void SB_Profile::print_vary_parameters()
{
	ios_base::fmtflags current_flags = cout.flags();
	if (current_flags & ios::scientific) cout << resetiosflags(ios::scientific);
	if (n_vary_params==0) {
		cout << "   parameters: none\n";
	} else {
		vector<string> paramnames_vary;
		get_fit_parameter_names(paramnames_vary);
		if (include_limits) {
			if (lower_limits.size() != n_vary_params) cout << "   Warning: parameter limits not defined\n";
			else {
				cout << "   parameter limits:\n";
				for (int i=0; i < n_vary_params; i++) {
					cout << "   " << paramnames_vary[i] << ": [" << lower_limits[i] << ":" << upper_limits[i] << "]\n";
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
	if (current_flags & ios::scientific) cout << setiosflags(ios::scientific);
}

string SB_Profile::mkstring_doub(const double db)
{
	stringstream dstr;
	string dstring;
	dstr << db;
	dstr >> dstring;
	return dstring;
}

string SB_Profile::mkstring_int(const int i)
{
	stringstream istr;
	string istring;
	istr << i;
	istr >> istring;
	return istring;
}

// This function is used by the Python wrapper
string SB_Profile::get_parameters_string()
{
	string paramstring = "";
	if (sb_number != -1) paramstring += mkstring_int(sb_number) + ". (";
	if (!is_lensed) paramstring += "unlensed): ";
	else paramstring += "z=" + mkstring_doub(sbparams->zsrc) + "): ";
	for (int i=0; i < n_params; i++) {
		paramstring += paramnames[i] + "=";
		if (angle_param[i]) paramstring += mkstring_doub(radians_to_degrees(*(sbparams->param[i]))) + " degrees";
		else paramstring += mkstring_doub(*(sbparams->param[i]));
		if (i != n_params-1) paramstring += ", ";
	}
	if (center_anchored_to_source) paramstring += " (center anchored to source " + mkstring_doub(center_anchor_source->sb_number) + ")";
	else if (center_anchored_to_lens) paramstring += " (center anchored to lens " + mkstring_doub(center_anchor_lens->lens_number) + ")";
	else if (center_anchored_to_ptsrc) paramstring += " (center anchored to ptsrc " + mkstring_doub(center_anchor_ptsrc->entry_number) + ")";

	if ((ellipticity_mode != default_ellipticity_mode) and (ellipticity_mode != -1)) {
		paramstring += " (";
		if (ellipticity_gradient) paramstring += "egrad=on,";
		if (fourier_gradient) paramstring += "fgrad=on,";
		paramstring += "emode=" + mkstring_int(ellipticity_mode) + ")"; // emode=3 is indicated by "pseudo-" name, not here
	} else {
		if (ellipticity_gradient) {
			if (fourier_gradient) paramstring += " (egrad,fgrad=on)";
			else paramstring += " (egrad=on)";
		}
	}
	//if (lensed_center_coords) paramstring += " (xc=" + mkstring_doub(x_center) + ", yc=" + mkstring_doub(y_center) + ")";
	//if (show_band) {
		//paramstring += " (band=" + mkstring_int(band) + ")";
	//}

	return paramstring;
}

void SB_Profile::window_params(double& xmin, double& xmax, double& ymin, double& ymax)
{
	double rmax = window_rmax();
	if (ellipticity_mode==0) {
		xmin = -rmax;
		xmax = rmax;
		ymin = -sbparams->q*rmax;
		ymax = sbparams->q*rmax;
	} else {
		double sqrtq = sqrt(sbparams->q);
		xmin = -rmax/sqrtq;
		xmax = rmax/sqrtq;
		ymin = -sqrtq*rmax;
		ymax = sqrtq*rmax;
	}
	if (sbparams->theta != 0) {
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
	xmin += sbparams->x_center;
	xmax += sbparams->x_center;
	ymin += sbparams->y_center;
	ymax += sbparams->y_center;
}

double SB_Profile::window_rmax()
{
	return sbparams_spl.qx_parameter*sbparams_spl.sb_spline.xmax();
}

double SB_Profile::length_scale()
{
	return sbparams_spl.qx_parameter*sbparams_spl.sb_spline.xmax();
}

/********************************* Specific SB_Profile models (derived classes) *********************************/

Gaussian::Gaussian(const int band_in, const double &zsrc_in, const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	sbparams = &sbparams_gaussian;
#ifdef USE_STAN
	sbparams_dif = &sbparams_gaussian_dif;
#endif
	model_name = "gaussian";
	sbtype = GAUSSIAN;
	band = band_in;
	setup_base_source_properties(6,2,true);
	qlens = qlens_in;
	sbparams_gaussian.max_sb = max_sb_in;
	sbparams_gaussian.sig_x = sig_x_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

Gaussian::Gaussian(const Gaussian* sb_in)
{
	sbparams = &sbparams_gaussian;
#ifdef USE_STAN
	sbparams_dif = &sbparams_gaussian_dif;
#endif
	copy_base_source_data(sb_in);
	sbparams_gaussian.max_sb = sb_in->sbparams_gaussian.max_sb;
	sbparams_gaussian.sig_x = sb_in->sbparams_gaussian.sig_x;
	sbparams_gaussian.sbtot = sb_in->sbparams_gaussian.sbtot;
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void Gaussian::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_gaussian_dif.max_sb = sbparams_gaussian.max_sb;
	sbparams_gaussian_dif.sig_x = sbparams_gaussian.sig_x;
	sbparams_gaussian_dif.sbtot = sbparams_gaussian.sbtot;
}
#endif

template <typename QScalar>
void Gaussian::update_meta_parameters_impl()
{
	update_ellipticity_meta_parameters<QScalar>();
	//max_sb = sbtot/(M_2PI*q*sig_x*sig_x);
	Gaussian_Params<QScalar>& p = assign_gaussian_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.sbtot = p.max_sb*(M_2PI*p.sig_x*p.sig_x);
	if (ellipticity_mode==0) p.sbtot *= p.q;
}
template void Gaussian::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void Gaussian::update_meta_parameters_impl<stan::math::var>();
#endif

void Gaussian::assign_paramnames()
{
	paramnames[0] = "sbmax";     latex_paramnames[0] = "S";       latex_param_subscripts[0] = "max";
	paramnames[1] = "sigma"; latex_paramnames[1] = "\\sigma"; latex_param_subscripts[1] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

template <typename QScalar>
void Gaussian::assign_param_pointers_impl()
{
	Gaussian_Params<QScalar>& p = assign_gaussian_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.max_sb;
	p.param[1] = &p.sig_x;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void Gaussian::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void Gaussian::assign_param_pointers_impl<stan::math::var>();
#endif

void Gaussian::set_auto_stepsizes()
{
	int index = 0;
	Gaussian_Params<double>& p = assign_gaussian_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	stepsizes[index++] = (p.max_sb != 0) ? 0.1*p.max_sb : 0.1;
	stepsizes[index++] = (p.sig_x != 0) ? 0.1*p.sig_x : 0.1; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void Gaussian::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

template <typename QScalar>
QScalar Gaussian::sb_rsq_impl(const QScalar rsq)
{
	Gaussian_Params<QScalar>& p = assign_gaussian_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	return p.max_sb*exp(-0.5*rsq/(p.sig_x*p.sig_x));
}
template double Gaussian::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Gaussian::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

double Gaussian::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 7*sbparams_gaussian.sig_x;
}

double Gaussian::length_scale()
{
	return sbparams_gaussian.sig_x;
}

Sersic::Sersic(const int band_in, const double &zsrc_in, const double &s_in, const double &Reff_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, QLens* qlens_in)
{
	sbparams = &sbparams_sersic;
#ifdef USE_STAN
	sbparams_dif = &sbparams_sersic_dif;
#endif
	model_name = "sersic";
	sbtype = SERSIC;
	band = band_in;
	setup_base_source_properties(7,3,true,parameter_mode_in);
	qlens = qlens_in;
	sbparams_sersic.n = n_in;
	sbparams_sersic.Reff = Reff_in;
	if (parameter_mode==0) {
		sbparams_sersic.s0 = s_in;
	} else {
		sbparams_sersic.s_eff = s_in;
	}
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

Sersic::Sersic(const Sersic* sb_in)
{
	sbparams = &sbparams_sersic;
#ifdef USE_STAN
	sbparams_dif = &sbparams_sersic_dif;
#endif
	sbparams_sersic.s0 = sb_in->sbparams_sersic.s0;
	sbparams_sersic.s_eff = sb_in->sbparams_sersic.s_eff;
	sbparams_sersic.n = sb_in->sbparams_sersic.n;
	sbparams_sersic.Reff = sb_in->sbparams_sersic.Reff;
	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void Sersic::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_sersic_dif.s0 = sbparams_sersic.s0;
	sbparams_sersic_dif.s_eff = sbparams_sersic.s_eff;
	sbparams_sersic_dif.n = sbparams_sersic.n;
	sbparams_sersic_dif.Reff = sbparams_sersic.Reff;
	sbparams_sersic_dif.b = sbparams_sersic.b;
}
#endif

template <typename QScalar>
void Sersic::update_meta_parameters_impl()
{
#ifdef USE_STAN
	using stan::math::exp;
	using stan::math::pow;
#endif
	Sersic_Params<QScalar>& p = assign_sersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.b = 2*p.n - 0.33333333333333 + 4.0/(405*p.n) + 46.0/(25515*p.n*p.n) + 131.0/(1148175*p.n*p.n*p.n); // from Cardone 2003 (or Ciotti 1999)
	if (parameter_mode==1) {
		p.s0 = p.s_eff * exp(p.b);
	}
	//k = b*pow(1.0/Reff,1.0/n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_ellipticity_meta_parameters<QScalar>();
}
template void Sersic::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void Sersic::update_meta_parameters_impl<stan::math::var>();
#endif

void Sersic::assign_paramnames()
{
	if (parameter_mode==0) {
		paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	} else {
		paramnames[0] = "s_eff"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "eff";
	}
	paramnames[1] = "Reff"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "eff";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

template <typename QScalar>
void Sersic::assign_param_pointers_impl()
{
	Sersic_Params<QScalar>& p = assign_sersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	if (parameter_mode==0) {
		p.param[0] = &p.s0;
	} else {
		p.param[0] = &p.s_eff;
	}
	p.param[1] = &p.Reff;
	p.param[2] = &p.n;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void Sersic::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void Sersic::assign_param_pointers_impl<stan::math::var>();
#endif

void Sersic::set_auto_stepsizes()
{
	int index = 0;
	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	if (parameter_mode==0) {
		stepsizes[index++] = (p.s0 > 0) ? 0.1*p.s0 : 0.1; 
	} else {
		stepsizes[index++] = (p.s_eff > 0) ? 0.1*p.s_eff : 0.1; 
	}
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.3; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void Sersic::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

template <typename QScalar>
QScalar Sersic::sb_rsq_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::exp;
	using stan::math::pow;
#endif
	Sersic_Params<QScalar>& p = assign_sersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	return p.s0*exp(-p.b*pow(rsq/(p.Reff*p.Reff),0.5/p.n));
}
template double Sersic::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Sersic::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

double Sersic::window_rmax()
{
	double fac = pow(3.0/sbparams_sersic.b,sbparams_sersic.n);
	if (fac < 1) fac = 1;
	return sbparams_sersic.Reff*fac;
}

double Sersic::length_scale()
{
	return sbparams_sersic.Reff;
}

Cored_Sersic::Cored_Sersic(const int band_in, const double &zsrc_in, const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	sbparams = &sbparams_csersic;
#ifdef USE_STAN
	sbparams_dif = &sbparams_csersic_dif;
#endif
	model_name = "csersic";
	sbtype = CORED_SERSIC;
	band = band_in;
	setup_base_source_properties(8,4,true);
	qlens = qlens_in;
	sbparams_csersic.n = n_in;
	sbparams_csersic.Reff = Reff_in;
	sbparams_csersic.s0 = s0_in;
	sbparams_csersic.rc = rc_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

Cored_Sersic::Cored_Sersic(const Cored_Sersic* sb_in)
{
	sbparams = &sbparams_csersic;
#ifdef USE_STAN
	sbparams_dif = &sbparams_csersic_dif;
#endif
	sbparams_csersic.s0 = sb_in->sbparams_csersic.s0;
	sbparams_csersic.n = sb_in->sbparams_csersic.n;
	sbparams_csersic.Reff = sb_in->sbparams_csersic.Reff;
	sbparams_csersic.rc = sb_in->sbparams_csersic.rc;
	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void Cored_Sersic::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_csersic_dif.s0 = sbparams_csersic.s0;
	sbparams_csersic_dif.n = sbparams_csersic.n;
	sbparams_csersic_dif.Reff = sbparams_csersic.Reff;
	sbparams_csersic_dif.rc = sbparams_csersic.rc;
	sbparams_csersic_dif.b = sbparams_csersic.b;
}
#endif

template <typename QScalar>
void Cored_Sersic::update_meta_parameters_impl()
{
	Cored_Sersic_Params<QScalar>& p = assign_csersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.b = 2*p.n - 0.33333333333333 + 4.0/(405*p.n) + 46.0/(25515*p.n*p.n) + 131.0/(1148175*p.n*p.n*p.n); // from Cardone 2003 (or Ciotti 1999)
	//k = b*pow(1.0/Reff,1.0/n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_ellipticity_meta_parameters<QScalar>();
}
template void Cored_Sersic::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void Cored_Sersic::update_meta_parameters_impl<stan::math::var>();
#endif

void Cored_Sersic::assign_paramnames()
{
	paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	paramnames[1] = "Reff"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "eff";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	paramnames[3] = "rc"; latex_paramnames[3] = "r"; latex_param_subscripts[3] = "c";
	set_geometric_paramnames(sbprofile_nparams);
}

template <typename QScalar>
void Cored_Sersic::assign_param_pointers_impl()
{
	Cored_Sersic_Params<QScalar>& p = assign_csersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.s0;
	p.param[1] = &p.Reff;
	p.param[2] = &p.n;
	p.param[3] = &p.rc;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void Cored_Sersic::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void Cored_Sersic::assign_param_pointers_impl<stan::math::var>();
#endif

void Cored_Sersic::set_auto_stepsizes()
{
	int index = 0;
	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	stepsizes[index++] = (p.s0 > 0) ? 0.1*p.s0 : 0.1; 
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

template <typename QScalar>
QScalar Cored_Sersic::sb_rsq_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::exp;
	using stan::math::pow;
#endif
	Cored_Sersic_Params<QScalar>& p = assign_csersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	return p.s0*exp(-p.b*pow((rsq+p.rc*p.rc)/(p.Reff*p.Reff),0.5/p.n));
}
template double Cored_Sersic::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Cored_Sersic::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

double Cored_Sersic::window_rmax()
{
	return sbparams_csersic.Reff*pow(3.0/sbparams_csersic.b,sbparams_csersic.n);
}

double Cored_Sersic::length_scale()
{
	return sbparams_csersic.Reff;
}

CoreSersic::CoreSersic(const int band_in, const double &zsrc_in, const double &s0_in, const double &Reff_in, const double &n_in, const double &rc_in,	const double &gamma_in, const double &alpha_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	sbparams = &sbparams_coresersic;
#ifdef USE_STAN
	sbparams_dif = &sbparams_coresersic_dif;
#endif
	model_name = "Csersic";
	sbtype = CORE_SERSIC;
	band = band_in;
	setup_base_source_properties(10,6,true);
	qlens = qlens_in;
	sbparams_coresersic.n = n_in;
	sbparams_coresersic.Reff = Reff_in;
	sbparams_coresersic.s0 = s0_in;
	sbparams_coresersic.rc = rc_in;
	sbparams_coresersic.gamma = gamma_in;
	sbparams_coresersic.alpha = alpha_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

CoreSersic::CoreSersic(const CoreSersic* sb_in)
{
	sbparams = &sbparams_coresersic;
#ifdef USE_STAN
	sbparams_dif = &sbparams_coresersic_dif;
#endif
	sbparams_coresersic.s0 = sb_in->sbparams_coresersic.s0;
	sbparams_coresersic.n = sb_in->sbparams_coresersic.n;
	sbparams_coresersic.Reff = sb_in->sbparams_coresersic.Reff;
	sbparams_coresersic.rc = sb_in->sbparams_coresersic.rc;
	sbparams_coresersic.gamma = sb_in->sbparams_coresersic.gamma;
	sbparams_coresersic.alpha = sb_in->sbparams_coresersic.alpha;
	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void CoreSersic::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_coresersic_dif.s0 = sbparams_coresersic.s0;
	sbparams_coresersic_dif.n = sbparams_coresersic.n;
	sbparams_coresersic_dif.Reff = sbparams_coresersic.Reff;
	sbparams_coresersic_dif.rc = sbparams_coresersic.rc;
	sbparams_coresersic_dif.b = sbparams_coresersic.b;
	sbparams_coresersic_dif.gamma = sbparams_coresersic.gamma;
	sbparams_coresersic_dif.alpha = sbparams_coresersic.alpha;
}
#endif

template <typename QScalar>
void CoreSersic::update_meta_parameters_impl()
{
	CoreSersic_Params<QScalar>& p = assign_coresersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.b = 2*p.n - 0.33333333333333 + 4.0/(405*p.n) + 46.0/(25515*p.n*p.n) + 131.0/(1148175*p.n*p.n*p.n); // from Cardone 2003 (or Ciotti 1999)
	p.k = p.b*pow(1.0/p.Reff,1.0/p.n);
	//s0 = L0_in/(M_PI*Reff*Reff*2*n*Gamma(2*n)/pow(b,2*n));
	update_ellipticity_meta_parameters<QScalar>();
}
template void CoreSersic::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void CoreSersic::update_meta_parameters_impl<stan::math::var>();
#endif

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

template <typename QScalar>
void CoreSersic::assign_param_pointers_impl()
{
	CoreSersic_Params<QScalar>& p = assign_coresersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.s0;
	p.param[1] = &p.Reff;
	p.param[2] = &p.n;
	p.param[3] = &p.rc;
	p.param[4] = &p.gamma;
	p.param[5] = &p.alpha;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void CoreSersic::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void CoreSersic::assign_param_pointers_impl<stan::math::var>();
#endif

void CoreSersic::set_auto_stepsizes()
{
	int index = 0;
	CoreSersic_Params<double>& p = assign_coresersic_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	stepsizes[index++] = (p.s0 > 0) ? 0.1*p.s0 : 0.1; 
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

template <typename QScalar>
QScalar CoreSersic::sb_rsq_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::exp;
	using stan::math::pow;
#endif
	CoreSersic_Params<QScalar>& p = assign_coresersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	QScalar rca = pow(p.rc,p.alpha);
	QScalar ra = pow(rsq,p.alpha/2);
	return p.s0*pow(1+rca/ra,p.gamma/p.alpha)*exp(-p.k*pow(ra+rca,1.0/(p.alpha*p.n)));
}
template double CoreSersic::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var CoreSersic::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

double CoreSersic::window_rmax()
{
	return sbparams_coresersic.Reff*pow(3.0/sbparams_coresersic.k,sbparams_coresersic.n);
}

double CoreSersic::length_scale()
{
	return sbparams_coresersic.Reff;
}

DoubleSersic::DoubleSersic(const int band_in, const double &zsrc_in, const double &s0_in, const double &delta_s_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	sbparams = &sbparams_dsersic;
#ifdef USE_STAN
	sbparams_dif = &sbparams_dsersic_dif;
#endif
	model_name = "dsersic";
	sbtype = DOUBLE_SERSIC;
	band = band_in;
	setup_base_source_properties(10,6,true);
	qlens = qlens_in;
	sbparams_dsersic.s0 = s0_in;
	sbparams_dsersic.delta_s = delta_s_in;
	sbparams_dsersic.n1 = n1_in;
	sbparams_dsersic.Reff1 = Reff1_in;
	sbparams_dsersic.n2 = n2_in;
	sbparams_dsersic.Reff2 = Reff2_in;

	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

DoubleSersic::DoubleSersic(const DoubleSersic* sb_in)
{
	sbparams = &sbparams_dsersic;
#ifdef USE_STAN
	sbparams_dif = &sbparams_dsersic_dif;
#endif
	sbparams_dsersic.s0 = sb_in->sbparams_dsersic.s0;
	sbparams_dsersic.delta_s = sb_in->sbparams_dsersic.delta_s;
	sbparams_dsersic.n1 = sb_in->sbparams_dsersic.n1;
	sbparams_dsersic.Reff1 = sb_in->sbparams_dsersic.Reff1;
	sbparams_dsersic.n2 = sb_in->sbparams_dsersic.n2;
	sbparams_dsersic.Reff2 = sb_in->sbparams_dsersic.Reff2;
	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void DoubleSersic::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_dsersic_dif.s0 = sbparams_dsersic.s0;
	sbparams_dsersic_dif.delta_s = sbparams_dsersic.delta_s;
	sbparams_dsersic_dif.s0_1 = sbparams_dsersic.s0_1;
	sbparams_dsersic_dif.n1 = sbparams_dsersic.n1;
	sbparams_dsersic_dif.Reff1 = sbparams_dsersic.Reff1;
	sbparams_dsersic_dif.s0_2 = sbparams_dsersic.s0_2;
	sbparams_dsersic_dif.n2 = sbparams_dsersic.n2;
	sbparams_dsersic_dif.Reff2 = sbparams_dsersic.Reff2;
	sbparams_dsersic_dif.b1 = sbparams_dsersic.b1;
	sbparams_dsersic_dif.b2 = sbparams_dsersic.b2;
}
#endif

template <typename QScalar>
void DoubleSersic::update_meta_parameters_impl()
{
	DoubleSersic_Params<QScalar>& p = assign_dsersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.s0_1 = p.s0*(1+p.delta_s)/2;
	p.s0_2 = p.s0*(1-p.delta_s)/2;
	p.b1 = 2*p.n1 - 0.33333333333333 + 4.0/(405*p.n1) + 46.0/(25515*p.n1*p.n1) + 131.0/(1148175*p.n1*p.n1*p.n1);
	p.b2 = 2*p.n2 - 0.33333333333333 + 4.0/(405*p.n2) + 46.0/(25515*p.n2*p.n2) + 131.0/(1148175*p.n2*p.n2*p.n2);
	update_ellipticity_meta_parameters<QScalar>();
}
template void DoubleSersic::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void DoubleSersic::update_meta_parameters_impl<stan::math::var>();
#endif

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

template <typename QScalar>
void DoubleSersic::assign_param_pointers_impl()
{
	DoubleSersic_Params<QScalar>& p = assign_dsersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.s0;
	p.param[1] = &p.delta_s;
	p.param[2] = &p.Reff1;
	p.param[3] = &p.n1;
	p.param[4] = &p.Reff2;
	p.param[5] = &p.n2;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void DoubleSersic::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void DoubleSersic::assign_param_pointers_impl<stan::math::var>();
#endif

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

template <typename QScalar>
QScalar DoubleSersic::sb_rsq_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::exp;
	using stan::math::pow;
#endif
	DoubleSersic_Params<QScalar>& p = assign_dsersic_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	return (p.s0_1*exp(-p.b1*pow(rsq/(p.Reff1*p.Reff1),0.5/p.n1)) + p.s0_2*exp(-p.b2*pow(rsq/(p.Reff2*p.Reff2),0.5/p.n2)));
}
template double DoubleSersic::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var DoubleSersic::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

double DoubleSersic::window_rmax()
{
	double max1 = sbparams_dsersic.Reff1*pow(3.0/sbparams_dsersic.b1,sbparams_dsersic.n1);
	double max2 = sbparams_dsersic.Reff2*pow(3.0/sbparams_dsersic.b2,sbparams_dsersic.n2);
	return dmax(max1,max2);
}

double DoubleSersic::length_scale()
{
	return sqrt(sbparams_dsersic.Reff1*sbparams_dsersic.Reff1 + sbparams_dsersic.Reff2*sbparams_dsersic.Reff2);
}

SPLE::SPLE(const int band_in, const double &zsrc_in, const double &bb, const double &aa, const double &ss, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	sbparams = &sbparams_sple;
#ifdef USE_STAN
	sbparams_dif = &sbparams_sple_dif;
#endif
	model_name = "sple";
	sbtype = sple;
	band = band_in;
	setup_base_source_properties(7,3,true); // number of parameters = 7, is_elliptical_source = true
	qlens = qlens_in;
	sbparams_sple.bs = bb;
	sbparams_sple.alpha = aa;
	sbparams_sple.s = ss;
	if (sbparams_sple.s < 0) sbparams_sple.s = -sbparams_sple.s; // don't allow negative core radii
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

SPLE::SPLE(const SPLE* sb_in)
{
	sbparams = &sbparams_sple;
#ifdef USE_STAN
	sbparams_dif = &sbparams_sple_dif;
#endif
	sbparams_sple.bs = sb_in->sbparams_sple.bs;
	sbparams_sple.alpha = sb_in->sbparams_sple.alpha;
	sbparams_sple.s = sb_in->sbparams_sple.s;

	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void SPLE::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_sple_dif.bs = sbparams_sple.bs;
	sbparams_sple_dif.s = sbparams_sple.s;
	sbparams_sple_dif.alpha = sbparams_sple.alpha;
}
#endif

void SPLE::assign_paramnames()
{
	paramnames[0] = "bs";     latex_paramnames[0] = "b";       latex_param_subscripts[0] = "s";
	paramnames[1] = "alpha"; latex_paramnames[1] = "\\alpha"; latex_param_subscripts[1] = "";
	paramnames[2] = "s";     latex_paramnames[2] = "s";       latex_param_subscripts[2] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

template <typename QScalar>
void SPLE::assign_param_pointers_impl()
{
	SPLE_Params<QScalar>& p = assign_sple_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.bs;
	p.param[1] = &p.alpha;
	p.param[2] = &p.s;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void SPLE::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void SPLE::assign_param_pointers_impl<stan::math::var>();
#endif

template <typename QScalar>
void SPLE::update_meta_parameters_impl()
{
	update_ellipticity_meta_parameters<QScalar>();
	// these meta-parameters are used in analytic formulas for deflection, potential, etc.
}
template void SPLE::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void SPLE::update_meta_parameters_impl<stan::math::var>();
#endif

void SPLE::set_auto_stepsizes()
{
	int index = 0;
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	stepsizes[index++] = 0.1*p.bs;
	stepsizes[index++] = 0.1;
	stepsizes[index++] = 0.02*p.bs; // this one is a bit arbitrary, but hopefully reasonable enough
	set_geometric_param_auto_stepsizes(index);
}

void SPLE::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 2;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

template <typename QScalar>
QScalar SPLE::sb_rsq_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::pow;
#endif
	SPLE_Params<QScalar>& p = assign_sple_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	return ((p.bs==0.0) ? 0.0 : ((2-p.alpha) * pow(p.bs*p.bs/(p.s*p.s+rsq), p.alpha/2) / 2));
}
template double SPLE::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var SPLE::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

double SPLE::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 3*sbparams_sple.bs;
}

double SPLE::length_scale()
{
	return sbparams_sple.bs;
}

dPIE::dPIE(const int band_in, const double &zsrc_in, const double &bb, const double &aa, const double &ss, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	sbparams = &sbparams_dpie;
#ifdef USE_STAN
	sbparams_dif = &sbparams_dpie_dif;
#endif
	model_name = "dpie";
	sbtype = dpie;
	qlens = qlens_in;
	band = band_in;
	setup_base_source_properties(7,3,true); // number of parameters = 7, is_elliptical_source = true
	sbparams_dpie.bs = bb;
	sbparams_dpie.a = aa;
	sbparams_dpie.s = ss;
	if (sbparams_dpie.s < 0) sbparams_dpie.s = -sbparams_dpie.s; // don't allow negative core radii
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

dPIE::dPIE(const dPIE* sb_in)
{
	sbparams = &sbparams_dpie;
#ifdef USE_STAN
	sbparams_dif = &sbparams_dpie_dif;
#endif
	sbparams_dpie.bs = sb_in->sbparams_dpie.bs;
	sbparams_dpie.a = sb_in->sbparams_dpie.a;
	sbparams_dpie.s = sb_in->sbparams_dpie.s;

	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void dPIE::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_dpie_dif.bs = sbparams_dpie.bs;
	sbparams_dpie_dif.a = sbparams_dpie.a;
	sbparams_dpie_dif.s = sbparams_dpie.s;
}
#endif

void dPIE::assign_paramnames()
{
	paramnames[0] = "bs";     latex_paramnames[0] = "b";       latex_param_subscripts[0] = "s";
	paramnames[1] = "a"; latex_paramnames[1] = "a"; latex_param_subscripts[1] = "";
	paramnames[2] = "s";     latex_paramnames[2] = "s";       latex_param_subscripts[2] = "";
	set_geometric_paramnames(sbprofile_nparams);
}

template <typename QScalar>
void dPIE::assign_param_pointers_impl()
{
	dPIE_Params<QScalar>& p = assign_dpie_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.bs;
	p.param[1] = &p.a;
	p.param[2] = &p.s;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void dPIE::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void dPIE::assign_param_pointers_impl<stan::math::var>();
#endif

template <typename QScalar>
void dPIE::update_meta_parameters_impl()
{
	update_ellipticity_meta_parameters<QScalar>();
	// these meta-parameters are used in analytic formulas for deflection, potential, etc.
}
template void dPIE::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void dPIE::update_meta_parameters_impl<stan::math::var>();
#endif

void dPIE::set_auto_stepsizes()
{
	int index = 0;
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	stepsizes[index++] = 0.1*p.bs;
	stepsizes[index++] = 0.1*p.a;
	stepsizes[index++] = 0.02*p.bs; // this one is a bit arbitrary, but hopefully reasonable enough
	set_geometric_param_auto_stepsizes(index);
}

void dPIE::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

template <typename QScalar>
QScalar dPIE::sb_rsq_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::pow;
#endif
	dPIE_Params<QScalar>& p = assign_dpie_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	return (0.5 * p.bs * (pow(p.s*p.s+rsq, -0.5) - pow(p.a*p.a+rsq,-0.5)));
}
template double dPIE::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var dPIE::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

double dPIE::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 3*dmax(sbparams_dpie.bs,sbparams_dpie.a);
}

double dPIE::length_scale()
{
	return sbparams_dpie.bs;
}

NFW_Source::NFW_Source(const int band_in, const double &zsrc_in, const double &s0_in, const double &rs_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	sbparams = &sbparams_nfw;
#ifdef USE_STAN
	sbparams_dif = &sbparams_nfw_dif;
#endif
	model_name = "nfw";
	sbtype = nfw_SOURCE;
	band = band_in;
	setup_base_source_properties(6,2,true); // number of parameters = 6, is_elliptical_source = true
	qlens = qlens_in;
	sbparams_nfw.s0 = s0_in;
	sbparams_nfw.rs = rs_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

NFW_Source::NFW_Source(const NFW_Source* sb_in)
{
	sbparams = &sbparams_nfw;
#ifdef USE_STAN
	sbparams_dif = &sbparams_nfw_dif;
#endif
	sbparams_nfw.s0 = sb_in->sbparams_nfw.s0;
	sbparams_nfw.rs = sb_in->sbparams_nfw.rs;

	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void NFW_Source::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_nfw_dif.s0 = sbparams_nfw.s0;
	sbparams_nfw_dif.rs = sbparams_nfw.rs;
}
#endif

void NFW_Source::assign_paramnames()
{
	paramnames[0] = "s0"; latex_paramnames[0] = "S"; latex_param_subscripts[0] = "0";
	paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	set_geometric_paramnames(sbprofile_nparams);
}

template <typename QScalar>
void NFW_Source::assign_param_pointers_impl()
{
	NFW_Params<QScalar>& p = assign_nfw_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.s0;
	p.param[1] = &p.rs;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void NFW_Source::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void NFW_Source::assign_param_pointers_impl<stan::math::var>();
#endif

template <typename QScalar>
void NFW_Source::update_meta_parameters_impl()
{
	update_ellipticity_meta_parameters<QScalar>();
	// these meta-parameters are used in analytic formulas for deflection, potential, etc.
}
template void NFW_Source::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void NFW_Source::update_meta_parameters_impl<stan::math::var>();
#endif

void NFW_Source::set_auto_stepsizes()
{
	int index = 0;
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	stepsizes[index++] = 0.2*p.s0;
	stepsizes[index++] = 0.2*p.rs;
	set_geometric_param_auto_stepsizes(index);
}

void NFW_Source::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(sbprofile_nparams);
}

template <typename QScalar>
QScalar NFW_Source::sb_rsq_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::abs;
	using stan::math::log;
#endif
	NFW_Params<QScalar>& p = assign_nfw_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	QScalar xsq = rsq/(p.rs*p.rs);
	if (xsq < 1e-6) return -p.s0*(2+log(xsq/4));
	else if (abs(xsq-1) < 1e-5) return 2*p.s0*(0.3333333333333333 - (xsq-1)/5.0); // formula on next line becomes unstable for x close to 1, this fixes it
	else return 2*p.s0*(1 - nfw_function_xsq(xsq))/(xsq - 1);
}
template double NFW_Source::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var NFW_Source::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar NFW_Source::nfw_function_xsq(const QScalar &xsq)
{
#ifdef USE_STAN
	using stan::math::atan;
	using stan::math::atanh;
	using stan::math::sqrt;
#endif
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}
template double NFW_Source::nfw_function_xsq<double>(const double &xsq);
#ifdef USE_STAN
template stan::math::var NFW_Source::nfw_function_xsq<stan::math::var>(const stan::math::var &xsq);
#endif

double NFW_Source::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 3*sbparams_nfw.rs;
}

double NFW_Source::length_scale()
{
	return sbparams_nfw.rs;
}

Shapelet::Shapelet(const int band_in, const double &zsrc_in, const double &amp00, const double &scale_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int nn, const bool truncate, const int parameter_mode_in, QLens* qlens_in)
{
	sbparams = &sbparams_shapelet;
#ifdef USE_STAN
	sbparams_dif = &sbparams_shapelet_dif;
#endif
	model_name = "shapelet";
	sbtype = SHAPELET;
	band = band_in;
	setup_base_source_properties(6,1,false,parameter_mode_in);
	qlens = qlens_in;
	if (parameter_mode==0) {
		sbparams_shapelet.sig = scale_in;
		sbparams_shapelet.sig_factor = 1.0;
	} else {
		sbparams_shapelet.sig_factor = scale_in;
		sbparams_shapelet.sig = 1.0; // this will be set automatically using the 'find_shapelet_scaling_parameters' function in lens.cpp
	}
	sbparams_shapelet.regparam = 100; // default
	n_shapelets = nn;
	indxptr = &n_shapelets;
	sbparams_shapelet.amps = new double*[n_shapelets];
	for (int i=0; i < n_shapelets; i++) sbparams_shapelet.amps[i] = new double[n_shapelets];
	for (int i=0; i < n_shapelets; i++) {
		for (int j=0; j < n_shapelets; j++) {
			sbparams_shapelet.amps[i][j] = 0;
		}
	}
	sbparams_shapelet.amps[0][0] = amp00;
	truncate_at_3sigma = truncate;

	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

Shapelet::Shapelet(const Shapelet* sb_in)
{
	sbparams = &sbparams_shapelet;
#ifdef USE_STAN
	sbparams_dif = &sbparams_shapelet_dif;
#endif
	n_shapelets = sb_in->n_shapelets;
	indxptr = &n_shapelets;
	sbparams_shapelet.sig = sb_in->sbparams_shapelet.sig;
	sbparams_shapelet.sig_factor = sb_in->sbparams_shapelet.sig_factor;
	sbparams_shapelet.regparam = sb_in->sbparams_shapelet.regparam;
	sbparams_shapelet.amps = new double*[n_shapelets];
	for (int i=0; i < n_shapelets; i++) sbparams_shapelet.amps[i] = new double[n_shapelets];
	for (int i=0; i < n_shapelets; i++) {
		for (int j=0; j < n_shapelets; j++) {
			sbparams_shapelet.amps[i][j] = sb_in->sbparams_shapelet.amps[i][j];
		}
	}
	truncate_at_3sigma = sb_in->truncate_at_3sigma;
	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void Shapelet::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_shapelet_dif.sig = sbparams_shapelet.sig;
	sbparams_shapelet_dif.regparam = sbparams_shapelet.regparam;
	sbparams_shapelet_dif.sig_factor = sbparams_shapelet.sig_factor;
	// what to do about the amplitudes? Do we sync them here?
}
#endif

template <typename QScalar>
void Shapelet::update_meta_parameters_impl()
{
	update_ellipticity_meta_parameters<QScalar>();
}
template void Shapelet::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void Shapelet::update_meta_parameters_impl<stan::math::var>();
#endif

void Shapelet::assign_paramnames()
{
	int indx=0;
	if (parameter_mode==0) {
		paramnames[indx] = "sigma"; latex_paramnames[indx] = "\\sigma"; latex_param_subscripts[indx] = ""; indx++;
	} else {
		paramnames[indx] = "sigfac"; latex_paramnames[indx] = "f"; latex_param_subscripts[indx] = "\\sigma"; indx++;
	}
	paramnames[indx] = "regparam"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = ""; indx++;
	set_geometric_paramnames(indx);
}

template <typename QScalar>
void Shapelet::assign_param_pointers_impl()
{
	Shapelet_Params<QScalar>& p = assign_shapelet_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	int indx=0;
	if (parameter_mode==0) {
		p.param[indx++] = &p.sig;
	} else {
		p.param[indx++] = &p.sig_factor;
	}
	p.param[indx++] = &p.regparam;
	set_geometric_param_pointers<QScalar>(indx);
}
template void Shapelet::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void Shapelet::assign_param_pointers_impl<stan::math::var>();
#endif

void Shapelet::set_auto_stepsizes()
{
	int indx=0;
	Shapelet_Params<double>& p = assign_shapelet_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	if (parameter_mode==0) {
		stepsizes[indx++] = (p.sig != 0) ? 0.1*p.sig : 0.1; // arbitrary
	} else {
		stepsizes[indx++] = (p.sig_factor != 0) ? 0.1*p.sig_factor : 0.1; // arbitrary
	}
	stepsizes[indx++] = 0.3*p.regparam; // arbitrary
	set_geometric_param_auto_stepsizes(indx);
}

void Shapelet::set_auto_ranges()
{
	int indx=0;
	set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30; indx++;
	set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30; indx++;
	set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30; indx++;
	set_geometric_param_auto_ranges(indx);
}

template <typename QScalar>
QScalar Shapelet::surface_brightness_impl(QScalar x, QScalar y)
{
#ifdef USE_STAN
	using stan::math::sqrt;
	using stan::math::exp;
#endif
	Shapelet_Params<QScalar>& p = assign_shapelet_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	if ((truncate_at_3sigma) and (sqrt(x*x+y*y) > 2.3*p.sig)) return 0;
	if (p.theta != 0) rotate(x,y);

	QScalar gaussfactor, sb, xarg, yarg, fac, lastfac, sqrtq;
	gaussfactor = (0.5641895835477563/(p.sig))*exp(-(p.q*x*x+y*y/p.q)/(2*p.sig*p.sig));
	QScalar *hermvals_x = new QScalar[n_shapelets];
	QScalar *hermvals_y = new QScalar[n_shapelets];
	hermvals_x[0] = 1.0;
	hermvals_y[0] = 1.0;
	sqrtq = sqrt(p.q);
	xarg = x*sqrtq/p.sig;
	yarg = y/(sqrtq*p.sig);
	if (n_shapelets > 1) {
		hermvals_x[1] = 2*xarg/M_SQRT2;
		hermvals_y[1] = 2*yarg/M_SQRT2;
	}
	lastfac = 1.0/M_SQRT2;
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
			sb += p.amps[i][j]*hermvals_x[i]*hermvals_y[j];
		}
	}
	sb *= gaussfactor;
	//cout << "AMP00: " << amps[0][0] << endl;

	delete[] hermvals_x;
	delete[] hermvals_y;
	return sb;
}
template double Shapelet::surface_brightness_impl<double>(double x, double y);
#ifdef USE_STAN
template stan::math::var Shapelet::surface_brightness_impl<stan::math::var>(stan::math::var x, stan::math::var y);
#endif

void Shapelet::calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight)
{
	Shapelet_Params<double>& p = assign_shapelet_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	if ((truncate_at_3sigma) and (sqrt(x*x+y*y) > 2.3*p.sig)) return;
	if (p.theta != 0) rotate(x,y);

	double gaussfactor, xarg, yarg, fac, lastfac, sqrtq;
	gaussfactor = (0.5641895835477563/(p.sig))*exp(-(p.q*x*x+y*y/p.q)/(2*p.sig*p.sig));
	double *hermvals_x = new double[n_shapelets];
	double *hermvals_y = new double[n_shapelets];
	hermvals_x[0] = 1.0;
	hermvals_y[0] = 1.0;
	sqrtq = sqrt(p.q);
	xarg = x*sqrtq/p.sig;
	yarg = y/(sqrtq*p.sig);
	if (n_shapelets > 1) {
		hermvals_x[1] = 2*xarg/M_SQRT2;
		hermvals_y[1] = 2*yarg/M_SQRT2;
	}
	lastfac = 1.0/M_SQRT2;
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
	Shapelet_Params<double>& p = assign_shapelet_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	if (p.sig==0) die("sigma cannot be zero!!");
	int i,j,n=0;
	int indx = n_shapelets*n_shapelets + 1;
	double norm = 1.0/(2*p.sig*p.sig);
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
	Shapelet_Params<double>& p = assign_shapelet_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	if (p.sig==0) die("sigma cannot be zero!!");
	int i,j,n=0;
	double ip, jp;
	int indx = n_shapelets*n_shapelets + 1;
	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			ip = sqrt(i*(i+1));
			jp = sqrt(j*(j+1));
			Rmatrix[n] = (4*(i*i+j*j) + 3*(i+j) + 6 + 2*i*j + 2*ip*jp + 2*(i+j)*(ip + jp))/(4*SQR(p.sig*p.sig));
			Rmatrix_index[n] = indx;
			n++;
		}
	}
	Rmatrix_index[n] = indx;
}

void Shapelet::get_regularization_param_ptr(double*& regparam_ptr)
{
	Shapelet_Params<double>& p = assign_shapelet_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	regparam_ptr = &p.regparam;
}

void Shapelet::update_amplitudes(double*& ampvec)
{
	Shapelet_Params<double>& p = assign_shapelet_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	int i,j,k=0;

	for (i=0; i < n_shapelets; i++) {
		for (j=0; j < n_shapelets; j++) {
			p.amps[i][j] = *(ampvec++);
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
	return sbparams_shapelet.sig;
}

void Shapelet::update_scale_parameter(const double scale)
{
	if (parameter_mode==0) {
		sbparams_shapelet.sig = scale;
	} else {
		sbparams_shapelet.sig = sbparams_shapelet.sig_factor*scale;
	}
}

void Shapelet::update_indxptr(const int newval)
{
	Shapelet_Params<double>& p = assign_shapelet_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	// indxptr points to n_shapelets
	int old_nn = n_shapelets;
	n_shapelets = newval;
	indxptr = &n_shapelets;
	double **newamps = new double*[n_shapelets];
	for (int i=0; i < n_shapelets; i++) newamps[i] = new double[n_shapelets];
	for (int i=0; i < n_shapelets; i++) {
		for (int j=0; j < n_shapelets; j++) {
			if ((i < old_nn) and (j < old_nn)) newamps[i][j] = p.amps[i][j];
			else newamps[i][j] = 0;
		}
	}
	for (int i=0; i < old_nn; i++) delete[] p.amps[i];
	delete[] p.amps;
	p.amps = newamps;
}

double Shapelet::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	if (truncate_at_3sigma) return 2.3*sbparams_shapelet.sig;
	else return 2*sbparams_shapelet.sig*sqrt(n_shapelets);
}

double Shapelet::length_scale()
{
	if (truncate_at_3sigma) return sbparams_shapelet.sig;
	return sbparams_shapelet.sig*sqrt(n_shapelets);
}


MGE::MGE(const int band_in, const double zsrc_in, const double reg, const double amp0, const double sig_i_in, const double sig_f_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int nn, const int parameter_mode_in, QLens* qlens_in)
{
	sbparams = &sbparams_mge;
#ifdef USE_STAN
	sbparams_dif = &sbparams_mge_dif;
#endif
	if (nn <= 0) die("must have n_gaussians > 0");
	model_name = "mge";
	sbtype = MULTI_GAUSSIAN_EXPANSION;
	band = band_in;
	setup_base_source_properties(5,1,false,parameter_mode_in);
	qlens = qlens_in;
	logsig_i = log(sig_i_in)/ln10;
	logsig_f = log(sig_f_in)/ln10;
	n_gaussians = nn;
	indxptr = &n_gaussians;
	sbparams_mge.regparam = reg;
	sbparams_mge.amps = new double[n_gaussians];
	sbparams_mge.sigs = new double[n_gaussians];
	int i;
	double logsig, logsigstep = (logsig_f - logsig_i) / (n_gaussians-1);
	for (i=0, logsig = logsig_i; i < n_gaussians; i++, logsig += logsigstep) {
		sbparams_mge.sigs[i] = pow(10,logsig);
		sbparams_mge.amps[i] = 0;
	}
	sbparams_mge.amps[0] = amp0;

	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

MGE::MGE(const MGE* sb_in)
{
	sbparams = &sbparams_mge;
#ifdef USE_STAN
	sbparams_dif = &sbparams_mge_dif;
#endif
	n_gaussians = sb_in->n_gaussians;
	logsig_i = sb_in->logsig_i;
	logsig_f = sb_in->logsig_f;
	indxptr = &n_gaussians;
	sbparams_mge.regparam = sb_in->sbparams_mge.regparam;
	sbparams_mge.amps = new double[n_gaussians];
	sbparams_mge.sigs = new double[n_gaussians];
	for (int i=0; i < n_gaussians; i++) {
		sbparams_mge.amps[i] = sb_in->sbparams_mge.amps[i];
		sbparams_mge.sigs[i] = sb_in->sbparams_mge.sigs[i];
	}
	copy_base_source_data(sb_in);
	update_meta_parameters();
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void MGE::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_mge_dif.regparam = sbparams_mge.regparam;
	// what to do about the amplitudes and widths? Do we sync them here?
}
#endif

template <typename QScalar>
void MGE::update_meta_parameters_impl()
{
	update_ellipticity_meta_parameters<QScalar>();
}
template void MGE::update_meta_parameters_impl<double>();
#ifdef USE_STAN
template void MGE::update_meta_parameters_impl<stan::math::var>();
#endif

void MGE::assign_paramnames()
{
	int indx=0;
	paramnames[indx] = "regparam"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = ""; indx++;
	set_geometric_paramnames(indx);
}

template <typename QScalar>
void MGE::assign_param_pointers_impl()
{
	MGE_Params<QScalar>& p = assign_mge_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	int indx=0;
	p.param[indx++] = &p.regparam;
	set_geometric_param_pointers<QScalar>(indx);
}
template void MGE::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void MGE::assign_param_pointers_impl<stan::math::var>();
#endif

void MGE::set_auto_stepsizes()
{
	int indx=0;
	MGE_Params<double>& p = assign_mge_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	stepsizes[indx++] = 0.3*p.regparam; // arbitrary
	set_geometric_param_auto_stepsizes(indx);
}

void MGE::set_auto_ranges()
{
	int indx=0;
	set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 1e-6; penalty_upper_limits[indx] = 1e30; indx++; // regparam
	set_geometric_param_auto_ranges(indx);
}

template <typename QScalar>
QScalar MGE::sb_rsq_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::exp;
#endif
	MGE_Params<QScalar>& p = assign_mge_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	QScalar sb=0;
	for (int i=0; i < n_gaussians; i++) {
		sb += p.amps[i]*exp(-rsq/SQR(p.sigs[i])/2)/M_SQRT_2PI/p.sigs[i];
	}
	return sb;
}
template double MGE::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var MGE::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar MGE::sb_rsq_deriv_impl(const QScalar rsq)
{
#ifdef USE_STAN
	using stan::math::exp;
#endif
	MGE_Params<QScalar>& p = assign_mge_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	QScalar sb_deriv=0;
	for (int i=0; i < n_gaussians; i++) {
		sb_deriv += -p.amps[i]*exp(-rsq/SQR(p.sigs[i])/2)/M_SQRT_2PI/CUBE(p.sigs[i])/2;
	}
	return sb_deriv;
}
template double MGE::sb_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var MGE::sb_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

void MGE::calculate_Lmatrix_elements(double x, double y, double*& Lmatrix_elements, const double weight)
{
	MGE_Params<double>& p = assign_mge_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	if (p.theta != 0) rotate(x,y);

	double xisq = x*x + y*y/(p.q*p.q);
	for (int i=0; i < n_gaussians; i++) {
		*(Lmatrix_elements++) += weight*exp(-xisq/SQR(p.sigs[i])/2)/M_SQRT_2PI/p.sigs[i];
	}
}

void MGE::calculate_curvature_Rmatrix_elements_rvals(double *rvalsq, const int n_rvals, double* Rmatrix_elements)
{
	//int i,j;
	//for (i=0; i < n_gaussians; i++) {
		//for (j=i; j < n_gaussians; j++) {
			//if ((i==0) and (j==0)) *(Rmatrix_elements) = 1;
			//else if (j==i) *(Rmatrix_elements) = 2;
			//else if (j==i+1) *(Rmatrix_elements) = -1;
			//else (*Rmatrix_elements) = 0;
			//Rmatrix_elements++;
		//}
	//}

	MGE_Params<double>& p = assign_mge_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	int i,j,k,l;
	double sigi5inv, sigj5inv, sigsqil, sigsqlj, sigl10inv;
	double sum_i, sum_j;
	for (i=0; i < n_gaussians; i++) {
		sigi5inv = pow(p.sigs[i],-1);
		for (j=i; j < n_gaussians; j++) {
			sigj5inv = pow(p.sigs[j],-1);
			//sigsqij = 1.0/(1.0/SQR(sigs[i]) + 1.0/SQR(sigs[j]));
			*(Rmatrix_elements) = 0;
			for (l=0; l < n_gaussians; l++) {
				sigsqil = 1.0/(1.0/SQR(p.sigs[i]) + 1.0/SQR(p.sigs[l]));
				sigsqlj = 1.0/(1.0/SQR(p.sigs[l]) + 1.0/SQR(p.sigs[j]));
				sigl10inv = pow(p.sigs[l],-2);
				sum_i = 0;
				for (k=0; k < n_rvals; k++) {
					sum_i += sqrt(rvalsq[k]*abs(rvalsq[k]-p.sigs[i]*p.sigs[i]))*exp(-rvalsq[k]*sigsqil/2);
				}
				sum_j = 0;
				for (k=0; k < n_rvals; k++) {
					sum_j += sqrt(rvalsq[k]*abs(rvalsq[k]-p.sigs[j]*p.sigs[j]))*exp(-rvalsq[k]*sigsqlj/2);
				}
				*(Rmatrix_elements) += sum_i*sum_j*sigi5inv*sigj5inv*sigl10inv;
			}
			Rmatrix_elements++;
		}
	}
}

void MGE::get_regularization_param_ptr(double*& regparam_ptr)
{
	regparam_ptr = &sbparams_mge.regparam;
}

void MGE::update_amplitudes(double*& ampvec)
{
	int i,j,k=0;

	MGE_Params<double>& p = assign_mge_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	for (i=0; i < n_gaussians; i++) {
		p.amps[i] = *(ampvec++);
		//cout << "AMP " << i << ": " << amps[i] << endl;
	}
}

/*
void MGE::get_amplitudes(double *ampvec)
{
	int i,j,k=0;
	for (i=0; i < n_gaussians; i++) {
		for (j=0; j < n_gaussians; j++) {
			ampvec[k++] = amps[i][j];
		}
	}
}
*/

void MGE::update_indxptr(const int newval)
{
	// indxptr points to n_gaussians
	int old_nn = n_gaussians;
	n_gaussians = newval;
	indxptr = &n_gaussians;

	MGE_Params<double>& p = assign_mge_param_object<double>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	double *newamps = new double[n_gaussians];
	double *newsigs = new double[n_gaussians];
	int i;
	double logsig, logsigstep = (logsig_f - logsig_i) / (n_gaussians-1);
	for (i=0, logsig = logsig_i; i < n_gaussians; i++, logsig += logsigstep) {
		newsigs[i] = pow(10,logsig);
		newamps[i] = 0;
	}
	if (p.amps != NULL) delete[] p.amps;
	if (p.sigs != NULL) delete[] p.sigs;
	p.amps = newamps;
	p.sigs = newsigs;
}

double MGE::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 3*sbparams_mge.sigs[n_gaussians-1];
}

double MGE::length_scale()
{
	return sbparams_mge.sigs[n_gaussians-1];
}


SB_Multipole::SB_Multipole(const int band_in, const double &zsrc_in, const double &A_m_in, const double r0_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool sine, QLens* qlens_in)
{
	sbparams = &sbparams_sbmpole;
#ifdef USE_STAN
	sbparams_dif = &sbparams_sbmpole_dif;
#endif
	model_name = "sbmpole";
	sbtype = SB_MULTIPOLE;
	band = band_in;
	//stringstream mstr;
	//string mstring;
	//mstr << m_in;
	//mstr >> mstring;
	sine_term = sine;
	setup_base_source_properties(5,0,false);
	qlens = qlens_in;

	sbparams_sbmpole.r0 = r0_in;
	m = m_in;
	sbparams_sbmpole.A_n = A_m_in;
	set_angle(theta_degrees);
	sbparams_sbmpole.x_center = xc_in;
	sbparams_sbmpole.y_center = yc_in;
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

SB_Multipole::SB_Multipole(const SB_Multipole* sb_in)
{
	sbparams = &sbparams_sbmpole;
#ifdef USE_STAN
	sbparams_dif = &sbparams_sbmpole_dif;
#endif
	sbparams_sbmpole.r0 = sb_in->sbparams_sbmpole.r0;
	m = sb_in->m;
	sbparams_sbmpole.A_n = sb_in->sbparams_sbmpole.A_n;
	sine_term = sb_in->sine_term;
	copy_base_source_data(sb_in);
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void SB_Multipole::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_sbmpole_dif.A_n = sbparams_sbmpole.A_n;
	sbparams_sbmpole_dif.r0 = sbparams_sbmpole.r0;
	sbparams_sbmpole_dif.theta_eff = sbparams_sbmpole.theta_eff;
}
#endif

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

template <typename QScalar>
void SB_Multipole::assign_param_pointers_impl()
{
	SB_Multipole_Params<QScalar>& p = assign_sbmpole_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.A_n; // here, A_n is actually the shear magnitude
	p.param[1] = &p.r0;
	p.param[2] = &p.theta; angle_param[2] = true;
	p.param[3] = &p.x_center;
	p.param[4] = &p.y_center;
}
template void SB_Multipole::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void SB_Multipole::assign_param_pointers_impl<stan::math::var>();
#endif

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

template <typename QScalar>
QScalar SB_Multipole::surface_brightness_impl(QScalar x, QScalar y)
{
#ifdef USE_STAN
	using stan::math::atan2;
	using stan::math::cos;
	using stan::math::exp;
	using stan::math::sqrt;
#endif
	SB_Multipole_Params<QScalar>& p = assign_sbmpole_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	QScalar phi = atan2(y,x);
	QScalar theta_eff = (sine_term) ? p.theta+M_HALFPI/m : p.theta;
	return p.A_n*exp(-sqrt(x*x+y*y)/p.r0) * cos(m*(phi-theta_eff));
}
template double SB_Multipole::surface_brightness_impl<double>(double x, double y);
#ifdef USE_STAN
template stan::math::var SB_Multipole::surface_brightness_impl<stan::math::var>(stan::math::var x, stan::math::var y);
#endif

double SB_Multipole::window_rmax() // used to define the window size for pixellated surface brightness maps
{
	return 7*sbparams_sbmpole.r0;
}

double SB_Multipole::length_scale()
{
	return sbparams_sbmpole.r0;
}

TopHat::TopHat(const int band_in, const double &zsrc_in, const double &sb_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, QLens* qlens_in)
{
	sbparams = &sbparams_tophat;
#ifdef USE_STAN
	sbparams_dif = &sbparams_tophat_dif;
#endif
	model_name = "tophat";
	sbtype = TOPHAT;
	band = band_in;
	setup_base_source_properties(6,2,true);
	qlens = qlens_in;
	sbparams_tophat.sb = sb_in; sbparams_tophat.rad = rad_in;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in,zsrc_in);
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

TopHat::TopHat(const TopHat* sb_in)
{
	sbparams = &sbparams_tophat;
#ifdef USE_STAN
	sbparams_dif = &sbparams_tophat_dif;
#endif
	sbparams_tophat.sb = sb_in->sbparams_tophat.sb;
	sbparams_tophat.rad = sb_in->sbparams_tophat.rad;
	copy_base_source_data(sb_in);
#ifdef USE_STAN
	sync_autodif_parameters();
	update_meta_parameters_autodif();
#endif
}

#ifdef USE_STAN
void TopHat::sync_autodif_parameters()
{
	sync_autodif_geometric_parameters();
	sbparams_tophat_dif.sb = sbparams_tophat.sb;
	sbparams_tophat_dif.rad = sbparams_tophat.rad;
}
#endif

void TopHat::assign_paramnames()
{
	paramnames[0] = "sb";     latex_paramnames[0] = "S";       latex_param_subscripts[0] = "top";
	paramnames[1] = "rad"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "top";
	set_geometric_paramnames(2);
}

template <typename QScalar>
void TopHat::assign_param_pointers_impl()
{
	TopHat_Params<QScalar>& p = assign_tophat_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	p.param[0] = &p.sb;
	p.param[1] = &p.rad;
	set_geometric_param_pointers<QScalar>(sbprofile_nparams);
}
template void TopHat::assign_param_pointers_impl<double>();
#ifdef USE_STAN
template void TopHat::assign_param_pointers_impl<stan::math::var>();
#endif

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

template <typename QScalar>
QScalar TopHat::sb_rsq_impl(const QScalar rsq)
{
	TopHat_Params<QScalar>& p = assign_tophat_param_object<QScalar>(); // this reference will point to either the <double> sbparams or <stan::math::var> sbparams for autodiff
	return (rsq < p.rad*p.rad) ? p.sb : 0.0;
}
template double TopHat::sb_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var TopHat::sb_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

double TopHat::window_rmax()
{
	return 2*sbparams_tophat.rad;
}

double TopHat::length_scale()
{
	return sbparams_tophat.rad;
}


