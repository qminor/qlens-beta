#include "profile.h"
#include "lensintegral.h"
#include "qlens.h"
#include "mathexpr.h"
#include "romberg.h"
#include "errors.h"
#include "cosmo.h"
#include "params.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

LensProfile::LensProfile(const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_spl;
	setup_lens_properties();
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);

	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	lensparams_spl.qx_parameter = qx_in;
	lensparams_spl.f_parameter = f_in;
	lensparams_spl.kspline.input(splinefile);
	zfac = 1.0;

	set_integration_pointers();
}

void LensProfile::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = KSPLINE;
	model_name = "kspline";
	setup_base_lens_properties(7,2,true); // number of parameters = 7, is_elliptical_lens = true
}

void LensProfile::setup_base_lens_properties(const int np, const int lensprofile_np, const bool is_elliptical_lens, const int pmode_in, const int subclass_in)
{
	set_null_ptrs_and_values(); // sets pointers to NULL to make sure qlens doesn't try to delete them during setup
	lensparams->param = NULL;
	center_defined = true;
	parameter_mode = pmode_in;
	lens_subclass = subclass_in; // automatically set to -1 by default if there are no subclasses defined
	set_nparams_and_anchordata(np);
	lensprofile_nparams = lensprofile_np;
	center_anchored = false;
	anchor_special_parameter = false;
	transform_center_coords_to_pixsrc_frame = false;
	if (is_elliptical_lens) {
		ellipticity_mode = default_ellipticity_mode;
	} else {
		f_major_axis = 1; // used for calculating approximate angle-averaged Einstein radius for non-elliptical lens models
		ellipticity_mode = -1; // indicates not an elliptical lens
	}
	n_fourier_modes = 0;
	ellipticity_gradient = false;
	contours_overlap = false; // only relevant for ellipticity gradient mode
	overlap_log_penalty_prior = 0;
	lensed_center_coords = false;
	analytic_3d_density = false; // this will be changed to 'true' for certain models (e.g. NFW)
	perturber = false; // default

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	assign_param_pointers();
	assign_paramnames();

	include_limits = false;
	rmin_einstein_radius = 1e-5;
	rmax_einstein_radius = 1e4;
}

void LensProfile::set_redshifts(const double zlens_in, const double zsrc_in)
{
	if (zlens==0) {
		// only set zlens, etc. if they haven't been set before. After zlens is set, it should only be changed by doing update_parameters(...)
		zlens = zlens_in;
		zlens_current = zlens_in;
		zsrc_ref = zsrc_in;
	}
}

void LensProfile::setup_cosmology(Cosmology* cosmo_in)
{
	if (cosmo != cosmo_in) {
		if (cosmo_in != NULL) {
			cosmo = cosmo_in;
			sigma_cr = cosmo->sigma_crit_arcsec(zlens,zsrc_ref);
			kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(zlens);
		}
	}
}

LensProfile::LensProfile(const LensProfile* lens_in)
{
	lensparams = &lensparams_spl;
#ifdef USE_STAN
	lensparams_dif = &lensparams_spl_dif;
#endif
	LensSpline_Params<double>& p = assign_lensspline_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	//const LensSpline_Params<double>& p_in = lens_in->assign_lensspline_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff

	p.qx_parameter = lens_in->lensparams_spl.qx_parameter;
	p.f_parameter = lens_in->lensparams_spl.f_parameter;
	p.kspline.input(lens_in->lensparams_spl.kspline);
	zfac = lens_in->zfac;

	copy_base_lensdata(lens_in);
	set_integration_pointers();
}

void LensProfile::copy_base_lensdata(const LensProfile* lens_in) // This must *always* get called by any derived class when copying another lens object
{
	cosmo = lens_in->cosmo;
	qlens = lens_in->qlens;
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_defined = lens_in->center_defined;
	zlens = lens_in->zlens;
	zsrc_ref = lens_in->zsrc_ref;
	sigma_cr = lens_in->sigma_cr;
	kpc_to_arcsec = lens_in->kpc_to_arcsec;

	center_anchored = lens_in->center_anchored;
	transform_center_coords_to_pixsrc_frame = lens_in->transform_center_coords_to_pixsrc_frame;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	center_anchor_lens = lens_in->center_anchor_lens;
	n_params = lens_in->n_params;
	lensprofile_nparams = lens_in->lensprofile_nparams;
	parameter_mode = lens_in->parameter_mode;
	lens_subclass = lens_in->lens_subclass;
	subclass_label = lens_in->subclass_label;
	ellipticity_mode = lens_in->ellipticity_mode;
	ellipticity_gradient = lens_in->ellipticity_gradient;
	fourier_gradient = lens_in->fourier_gradient;
	perturber = lens_in->perturber;
	lensed_center_coords = lens_in->lensed_center_coords;
	analytic_3d_density = lens_in->analytic_3d_density;

	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	angle_param.input(n_params);
	for (int i=0; i < n_params; i++) angle_param[i] = false; // the angle params will be recognized when assign_param_pointers() is called
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	stepsizes.input(lens_in->stepsizes);
	set_auto_penalty_limits.input(lens_in->set_auto_penalty_limits);
	penalty_lower_limits.input(lens_in->penalty_lower_limits);
	penalty_upper_limits.input(lens_in->penalty_upper_limits);
	integral_tolerance = lens_in->integral_tolerance;
	integration_warnings = lens_in->integration_warnings;

	if (ellipticity_mode != -1) {
		lensparams->q = lens_in->lensparams->q;
		lensparams->epsilon = lens_in->lensparams->epsilon;
		lensparams->epsilon1 = lens_in->lensparams->epsilon1;
		lensparams->epsilon2 = lens_in->lensparams->epsilon2;
	}

	f_major_axis = lens_in->f_major_axis;
	angle_param_exists = lens_in->angle_param_exists;
	if (angle_param_exists) set_angle_radians(lens_in->lensparams->theta);
	lensparams->x_center = lens_in->lensparams->x_center;
	lensparams->y_center = lens_in->lensparams->y_center;
	if ((lensed_center_coords) or (transform_center_coords_to_pixsrc_frame)) {
		lensparams->xc_prime = lens_in->lensparams->xc_prime;
		lensparams->yc_prime = lens_in->lensparams->yc_prime;
	}
	n_fourier_modes = lens_in->n_fourier_modes;
	if (n_fourier_modes > 0) {
		fourier_mode_mvals.input(lens_in->fourier_mode_mvals);
		fourier_mode_cosamp.input(lens_in->fourier_mode_cosamp);
		fourier_mode_sinamp.input(lens_in->fourier_mode_sinamp);
		fourier_mode_paramnum.input(lens_in->fourier_mode_paramnum);
	}
	if (ellipticity_gradient) {
		egrad_mode = lens_in->egrad_mode;
		egrad_ellipticity_mode = lens_in->egrad_ellipticity_mode;
		center_gradient = lens_in->center_gradient;
		int i,j;
		for (i=0; i < 4; i++) {
			n_egrad_params[i] = lens_in->n_egrad_params[i];
			geometric_param_ref[i] = lens_in->geometric_param_ref[i];
			geometric_param_dif[i] = lens_in->geometric_param_dif[i];
			geometric_param[i] = new double[n_egrad_params[i]];
			for (j=0; j < n_egrad_params[i]; j++) {
				geometric_param[i][j] = lens_in->geometric_param[i][j];
			}
		}
		 angle_param_egrad = new bool[n_egrad_params[1]]; // keeps track of which parameters are angles, so they can be converted to degrees when displayed
		for (j=0; j < n_egrad_params[1]; j++) {
			angle_param_egrad[j] = lens_in->angle_param_egrad[j];
		}
		xi_initial_egrad = lens_in->xi_initial_egrad;
		xi_final_egrad = lens_in->xi_final_egrad;
		xi_ref_egrad = lens_in->xi_ref_egrad;
		if (egrad_mode==0) {
			bspline_order = lens_in->bspline_order;
			n_bspline_knots_tot = lens_in->n_bspline_knots_tot;
			for (i=0; i < 4; i++) {
				geometric_knots[i] = new double[n_bspline_knots_tot];
				for (j=0; j < n_bspline_knots_tot; j++) geometric_knots[i][j] = lens_in->geometric_knots[i][j];
			}
		}
		contours_overlap = lens_in->contours_overlap;
		set_egrad_ptr();
	}
	if (fourier_gradient) {
		n_fourier_grad_modes = lens_in->n_fourier_grad_modes;
		fourier_grad_mvals = fourier_mode_mvals.array(); // fourier_grad_mvals is used in the egrad functions (which are inherited by this class)
		n_fourier_grad_params = new int[n_fourier_modes];
		int i,j,k;
		for (i=0; i < n_fourier_grad_modes; i++) {
			n_fourier_grad_params[i] = lens_in->n_fourier_grad_params[i];
		}
		int n_amps = n_fourier_grad_modes*2;
		fourier_param = new double*[n_amps];
		for (i=0,k=0; i < n_fourier_grad_modes; i++, k+=2) {
			fourier_param[k] = new double[n_fourier_grad_params[i]];
			fourier_param[k+1] = new double[n_fourier_grad_params[i]];
			for (j=0; j < n_fourier_grad_params[i]; j++) {
				fourier_param[k][j] = lens_in->fourier_param[k][j];
			}
			for (j=0; j < n_fourier_grad_params[i]; j++) {
				fourier_param[k+1][j] = lens_in->fourier_param[k+1][j];
			}
		}
		if (egrad_mode==0) {
			fourier_knots = new double*[n_amps];
			for (i=0; i < n_amps; i++) {
				fourier_knots[i] = new double[n_bspline_knots_tot];
				for (j=0; j < n_bspline_knots_tot; j++) fourier_knots[i][j] = lens_in->fourier_knots[i][j];
			}
		}
	}
	include_limits = lens_in->include_limits;
	if (include_limits) {
		lower_limits.input(lens_in->lower_limits);
		upper_limits.input(lens_in->upper_limits);
	}

	lensparams->param = new double*[n_params];
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
}

void LensProfile::copy_source_data_to_lens(const SB_Profile* sb_in)
{
	zlens = sb_in->zsrc;
	zlens_current = zlens;
	lensparams->q = sb_in->q;
	lensparams->epsilon1 = sb_in->epsilon1;
	lensparams->epsilon2 = sb_in->epsilon2;
	angle_param_exists = sb_in->angle_param_exists;
	if (angle_param_exists) set_angle_radians(sb_in->theta);
	lensparams->x_center = sb_in->x_center;
	lensparams->y_center = sb_in->y_center;
	ellipticity_mode = sb_in->ellipticity_mode;
	ellipticity_gradient = sb_in->ellipticity_gradient;
	fourier_gradient = sb_in->fourier_gradient;

	n_fourier_modes = sb_in->n_fourier_modes;
	if (n_fourier_modes > 0) {
		fourier_mode_mvals.input(sb_in->fourier_mode_mvals);
		fourier_mode_cosamp.input(sb_in->fourier_mode_cosamp);
		fourier_mode_sinamp.input(sb_in->fourier_mode_sinamp);
		fourier_mode_paramnum.input(sb_in->fourier_mode_paramnum);
		int n_new_params = n_params + 2*n_fourier_modes;
		set_nparams_and_anchordata(n_new_params);
	}

	if (ellipticity_gradient) {
		egrad_mode = sb_in->egrad_mode;
		egrad_ellipticity_mode = sb_in->egrad_ellipticity_mode;
		center_gradient = sb_in->center_gradient;
		f_major_axis = 1.0; // f_major_axis doesn't apply when egrad is on, but is still called in get_einstein_radius(...)
		int i,j;
		for (i=0; i < 4; i++) {
			n_egrad_params[i] = sb_in->n_egrad_params[i];
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
		int n_new_params = n_params + get_egrad_nparams() - 4;
		set_nparams_and_anchordata(n_new_params);
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
		int n_new_params = n_params + get_fgrad_nparams() - 2*n_fourier_modes;
		set_nparams_and_anchordata(n_new_params);
	}
	assign_paramnames();
	assign_param_pointers();
}

void LensProfile::set_nparams_and_anchordata(const int &n_params_in, const bool resize)
{
	int old_nparams = (resize) ? n_params : 0;
	n_params = n_params_in;
	vary_params.input(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.input(n_params);
	set_auto_penalty_limits.input(n_params);
	penalty_lower_limits.input(n_params);
	penalty_upper_limits.input(n_params);
	angle_param.input(n_params);
	for (int i=0; i < n_params; i++) angle_param[i] = false;

	if (lensparams->param != NULL) delete[] lensparams->param;
	if (anchor_parameter_to_lens != NULL) delete[] anchor_parameter_to_lens;
	if (parameter_anchor_lens != NULL) delete[] parameter_anchor_lens;
	if (anchor_parameter_to_source != NULL) delete[] anchor_parameter_to_source;
	if (parameter_anchor_source != NULL) delete[] parameter_anchor_source;
	if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
	if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
	if (parameter_anchor_exponent != NULL) delete[] parameter_anchor_exponent;

	anchor_parameter_to_lens = new bool[n_params];
	parameter_anchor_lens = new LensProfile*[n_params];
	anchor_parameter_to_source = new bool[n_params];
	parameter_anchor_source = new SB_Profile*[n_params];
	parameter_anchor_paramnum = new int[n_params];
	parameter_anchor_ratio = new double[n_params];
	parameter_anchor_exponent = new double[n_params];
	at_least_one_param_anchored = false;
	lensparams->param = new double*[n_params];
	for (int i=0; i < n_params; i++) {
		//vary_params[i] = false;
		anchor_parameter_to_lens[i] = false;
		parameter_anchor_lens[i] = NULL;
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

bool LensProfile::anchor_center_to_lens(const int &center_anchor_lens_number)
{
	if (qlens == NULL) return false;
	if (!center_anchored) center_anchored = true;
	center_anchor_lens = qlens->lens_list[center_anchor_lens_number];
	lensparams->x_center = center_anchor_lens->lensparams->x_center;
	lensparams->y_center = center_anchor_lens->lensparams->y_center;
	return true;
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

bool LensProfile::setup_transform_center_coords_to_pixsrc_frame(const double dxc, const double dyc, QLens* qlensptr_in)
{
	QLens* qlensptr;
	if (qlens == NULL) {
		qlensptr = qlensptr_in;
		if (qlensptr==NULL) return false;
	} else qlensptr = qlens;
	if (ellipticity_mode == -1) die("can only transform center coords for elliptical lens");
	if (!transform_center_coords_to_pixsrc_frame) transform_center_coords_to_pixsrc_frame = true;
	assign_param_pointers();
	assign_paramnames();
	lensparams->xc_prime = dxc;
	lensparams->yc_prime = dyc;
	update_center_from_pixsrc_coords(qlensptr);
	return true;
}

void LensProfile::set_spawned_mass_and_anchor_parameters(SB_Profile* sb_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper)
{
	if (vary_mass_parameter) {
		vary_params[0] = true;
		n_vary_params = 1;
		include_limits = include_limits_in;
		if (include_limits) {
			lower_limits.input(n_vary_params);
			upper_limits.input(n_vary_params);
			lower_limits[0] = mass_param_lower;
			upper_limits[0] = mass_param_upper;
		}
	}

	set_integration_pointers();
	set_model_specific_integration_pointers();
	// We don't update meta parameters yet because we still need to initialize the cosmology (since cosmology info couldn't be retrieved from source object)

	for (int i=1; i < n_params-1; i++) {
		// anchoring every parameter except the mass parameter (since stellar mass-to-light ratio is not known), and the redshift (since that's not a parameter in SB_Profile yet)
		anchor_parameter_to_source[i] = true;
		parameter_anchor_source[i] = (SB_Profile*) sb_in;
		parameter_anchor_paramnum[i] = i;
		parameter_anchor_ratio[i] = 1.0;
		(*lensparams->param[i]) = *(parameter_anchor_source[i]->param[i]);
		at_least_one_param_anchored = true;
	}
	update_anchored_parameters();
}

bool LensProfile::set_vary_flags(boolvector &vary_flags)
{
	bool omitted_center = false;
	if ((center_anchored) and vary_flags.size() == n_params-3) omitted_center = true;
	boolvector new_vary_flags(n_params);
	if ((vary_flags.size() != n_params-1) and (vary_flags.size() != n_params) and (!omitted_center)) return false;
	for (int i=0; i < vary_flags.size(); i++) new_vary_flags[i] = vary_flags[i];
	if (omitted_center) {
		new_vary_flags[n_params-3] = false;
		new_vary_flags[n_params-2] = false;
	}
	if (vary_flags.size() == n_params) new_vary_flags[n_params-1] = vary_flags[n_params-1];
	else new_vary_flags[n_params-1] = false; // if no vary flag is given for redshift, then assume it's not being varied
	if (vary_parameters(new_vary_flags)==false) return false;
	return true;
}

bool LensProfile::update_specific_varyflag(const string name_in, const bool flag)
{
	int paramnum;
	if (!lookup_parameter_number(name_in,paramnum)) return false;
	boolvector new_vary_flags(vary_params);
	new_vary_flags[paramnum] = flag;
	if (vary_parameters(new_vary_flags)==false) return false;
	return true;
}

void LensProfile::get_vary_flags(boolvector &vary_flags)
{
	vary_flags.input(n_params);
	for (int i=0; i < n_params; i++) vary_flags[i] = vary_params[i];
}

bool LensProfile::register_vary_flags()
{
	// This function is called if there are already vary flags that have been set before adding the lens to the list
	if ((n_vary_params > 0) and (qlens != NULL))
		return qlens->register_lens_vary_parameters(lens_number);
	else return false;
}

bool LensProfile::vary_parameters(const boolvector& vary_params_in)
{
	int pi, pf;
	if (qlens) qlens->get_lens_parameter_numbers(lens_number,pi,pf); // these are the old parameter numbers

	if (vary_params_in.size() != n_params) {
		if ((vary_params_in.size() == n_params-2) and (center_anchored)) {
			vary_params[n_params-2] = false;
			vary_params[n_params-1] = false;
		}
		else return false;
	}
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
		return qlens->register_lens_vary_parameters(lens_number);
	}
	return true;
}

void LensProfile::set_limits(const Vector<double>& lower, const Vector<double>& upper)
{
	include_limits = true;

	if (lower.size() != n_vary_params) die("number of parameters with lower limits does not match number of variable parameters (%i vs %i)",lower.size(),n_vary_params);
	if (upper.size() != n_vary_params) die("number of parameters with upper limits does not match number of variable parameters",upper.size(),n_vary_params);
	lower_limits = lower;
	upper_limits = upper;
	if (qlens != NULL) qlens->register_lens_prior_limits(lens_number);
}

bool LensProfile::set_limits_specific_parameter(const string name_in, const double& lower, const double& upper)
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
	if (qlens != NULL) qlens->register_lens_prior_limits(lens_number);
	return (param_i != -1);
}

void LensProfile::update_limits(const double* lower, const double* upper, const bool* limits_changed, int& index)
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

void LensProfile::get_parameters(double* params)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	for (int i=0; i < n_params; i++) {
		if (angle_param[i]) params[i] = radians_to_degrees(*(p.param[i]));
		else params[i] = *(p.param[i]);
	}
}

double LensProfile::get_parameter(const int i)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	if (i >= n_params) die("requested parameter with index greater than number of params");
	if (angle_param[i]) return radians_to_degrees(*(p.param[i]));
	else return *(p.param[i]);
}

bool LensProfile::check_parameter_name(const string name_in)
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

bool LensProfile::lookup_parameter_number(const string name_in, int& paramnum)
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

bool LensProfile::get_specific_parameter(const string name_in, double& value)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if (paramnames[i]==name_in) {
			found_match = true;
			value = *(lensparams->param)[i];
			break;
		}
	}
	return found_match;
}

bool LensProfile::get_specific_limit(const string name_in, double& lower, double& upper)
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

void LensProfile::get_parameters_pmode(const int pmode_in, double* params)
{
	// overload this function for models that have different parameter modes; allows
	// flexibility in obtaining parameters from different pmodes
	get_parameters(params);
	if (lensed_center_coords) {
		params[n_params-3] = lensparams->x_center;
		params[n_params-2] = lensparams->y_center;
	}
}

void LensProfile::update_parameters(const double* params)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	for (int i=0; i < n_params; i++) {
		if (angle_param[i]) *(p.param[i]) = degrees_to_radians(params[i]);
		else *(p.param[i]) = params[i];
	}
	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
	if (qlens != NULL) {
		qlens->update_anchored_parameters_and_redshift_data();
		qlens->update_lens_fitparams(lens_number);
	}
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
	if (found_match) {
		update_parameters(newparams);
	}
	delete[] newparams;
	if (qlens != NULL) {
		qlens->update_anchored_parameters_and_redshift_data();
		qlens->update_lens_fitparams(lens_number);
	}
	return found_match;
}

bool LensProfile::update_specific_parameter(const int paramnum, const double& value)
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
	if (qlens != NULL) {
		qlens->update_anchored_parameters_and_redshift_data();
		qlens->update_lens_fitparams(lens_number);
	}
	return true;
}

void LensProfile::update_ellipticity_parameter(const double eparam)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// This function is only used by the "qtab" model at the moment
	*(p.param[ellipticity_paramnum]) = eparam;
	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

// Perhaps should have a function at the model level, called here that reports a "false" status if parameter values are crazy!
void LensProfile::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	if (n_vary_params > 0) {
		for (int i=0; i < n_params; i++) {
			if (vary_params[i]==true) {
				if (angle_param[i]) {
					// the costheta, sintheta meta-parameters will be set in update_ellipticity_meta_parameters, which is called from update_meta_parameters for elliptical models
					*(p.param[i]) = degrees_to_radians(fitparams[index++]);
					update_angle_meta_params();
				}
				else *(p.param[i]) = fitparams[index++];
			}
		}
		update_meta_parameters();
		set_integration_pointers();
		set_model_specific_integration_pointers();
	}
	// NOTE: to save time, we do not update the qlens->param_list here, since this function is only run during model fitting
}

void LensProfile::update_anchored_parameters()
{
	LensParams<double>& p = assign_lensparam_object<double>();
	if (at_least_one_param_anchored) {
		for (int i=0; i < n_params; i++) {
			if (anchor_parameter_to_lens[i]) {
				(*p.param[i]) = parameter_anchor_ratio[i]*pow(*(parameter_anchor_lens[i]->lensparams->param[parameter_anchor_paramnum[i]]),parameter_anchor_exponent[i]);
				if (at_least_one_param_anchored==false) at_least_one_param_anchored = true;
			} else if (anchor_parameter_to_source[i]) {
				(*p.param[i]) = parameter_anchor_ratio[i]*pow(*(parameter_anchor_source[i]->param[parameter_anchor_paramnum[i]]),parameter_anchor_exponent[i]);
				if (at_least_one_param_anchored==false) at_least_one_param_anchored = true;
			}
		}
		update_meta_parameters();
		set_integration_pointers();
		set_model_specific_integration_pointers();
	}
}

void LensProfile::update_anchor_center()
{
	if (center_anchored) {
		lensparams->x_center = center_anchor_lens->lensparams->x_center;
		lensparams->y_center = center_anchor_lens->lensparams->y_center;
	}
}

void LensProfile::get_fit_parameters(double *fitparams, int &index)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]==true) {
			if (angle_param[i]) fitparams[index++] = radians_to_degrees(*(p.param[i]));
			else fitparams[index++] = *(p.param[i]);
		}
	}
}

void LensProfile::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1;
	stepsizes[index++] = 0.1;
	set_geometric_param_auto_stepsizes(index);
}

void LensProfile::set_geometric_param_auto_stepsizes(int &index)
{
	if (!ellipticity_gradient) {
		if (use_ellipticity_components) {
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
	stepsizes[index++] = 0.1; // z
}

void LensProfile::get_auto_stepsizes(Vector<double>& stepsizes_in, int &index)
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
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void LensProfile::set_geometric_param_auto_ranges(int param_i)
{
	if (!ellipticity_gradient) {
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
	} else {
		set_geometric_param_ranges_egrad(set_auto_penalty_limits, penalty_lower_limits, penalty_upper_limits, param_i);
	}
	if ((!fourier_gradient) and (n_fourier_modes > 0)) {
		for (int i=0; i < n_fourier_modes; i++) {
			set_auto_penalty_limits[param_i++] = false;
			set_auto_penalty_limits[param_i++] = false;
		}
	}
	set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 0.01; penalty_upper_limits[param_i] = zsrc_ref; param_i++;
}

void LensProfile::get_auto_ranges(boolvector& use_penalty_limits, Vector<double>& lower, Vector<double>& upper, int &index)
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

bool LensProfile::get_limits(Vector<double>& lower, Vector<double>& upper)
{
	if ((include_limits==false) or (lower_limits.size() != n_vary_params) or (lower.size() != n_vary_params)) return false;
	for (int i=0; i < n_vary_params; i++) {
		lower[i] = lower_limits[i];
		upper[i] = upper_limits[i];
	}
	return true;
}

bool LensProfile::get_limits(Vector<double>& lower, Vector<double>& upper, int &index)
{
	if ((include_limits==false) or (lower_limits.size() != n_vary_params)) return false;
	for (int i=0; i < n_vary_params; i++) {
		lower[index] = lower_limits[i];
		upper[index] = upper_limits[i];
		index++;
	}
	return true;
}

void LensProfile::copy_parameter_anchors(const LensProfile* lens_in)
{
	// n_params *must* already be set before running this
	anchor_parameter_to_lens = new bool[n_params];
	parameter_anchor_lens = new LensProfile*[n_params];
	anchor_parameter_to_source = new bool[n_params];
	parameter_anchor_source = new SB_Profile*[n_params];
	parameter_anchor_paramnum = new int[n_params];
	parameter_anchor_ratio = new double[n_params];
	parameter_anchor_exponent = new double[n_params];
	for (int i=0; i < n_params; i++) {
		anchor_parameter_to_lens[i] = lens_in->anchor_parameter_to_lens[i];
		parameter_anchor_lens[i] = lens_in->parameter_anchor_lens[i];
		anchor_parameter_to_source[i] = lens_in->anchor_parameter_to_source[i];
		parameter_anchor_source[i] = lens_in->parameter_anchor_source[i];
		parameter_anchor_paramnum[i] = lens_in->parameter_anchor_paramnum[i];
		parameter_anchor_ratio[i] = lens_in->parameter_anchor_ratio[i];
		parameter_anchor_exponent[i] = lens_in->parameter_anchor_exponent[i];
	}
	at_least_one_param_anchored = lens_in->at_least_one_param_anchored;
	if (anchor_special_parameter) copy_special_parameter_anchor(lens_in);
}

void LensProfile::copy_special_parameter_anchor(const LensProfile *lens_in)
{
	special_anchor_lens = lens_in->special_anchor_lens;
}


void LensProfile::assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, LensProfile* param_anchor_lens)
{
	if (paramnum >= n_params) die("Parameter does not exist for this qlens");
	if (anchor_paramnum >= param_anchor_lens->n_params) die("Parameter does not exist for lens you are anchoring to");
	if (anchor_parameter_to_source[paramnum]) { warn("cannot anchor a parameter to both a lens and a source object"); return; }
	anchor_parameter_to_lens[paramnum] = true;
	parameter_anchor_lens[paramnum] = param_anchor_lens;
	parameter_anchor_paramnum[paramnum] = anchor_paramnum;
	if ((!use_implicit_ratio) and (!use_exponent)) {
		parameter_anchor_ratio[paramnum] = 1.0;
		(*lensparams->param[paramnum]) = *(param_anchor_lens->lensparams->param[anchor_paramnum]);
	}
	else if (use_implicit_ratio) {
		parameter_anchor_exponent[paramnum] = 1.0;
		if ((*(param_anchor_lens->lensparams->param[anchor_paramnum]))==0) {
			if (*lensparams->param[paramnum]==0) parameter_anchor_ratio[paramnum] = 1.0;
			else die("cannot anchor to parameter with specified ratio if parameter is equal to zero");
		} else {
			parameter_anchor_ratio[paramnum] = (*lensparams->param[paramnum]) / (*(param_anchor_lens->lensparams->param[anchor_paramnum]));
		}
	}
	else if (use_exponent) {
		parameter_anchor_ratio[paramnum] = ratio;
		parameter_anchor_exponent[paramnum] = exponent;
	}
	at_least_one_param_anchored = true;
	update_anchored_parameters();
}

void LensProfile::assign_anchored_parameter(const int& paramnum, const int& anchor_paramnum, const bool use_implicit_ratio, const bool use_exponent, const double ratio, const double exponent, SB_Profile* param_anchor_source)
{
	if (paramnum >= n_params) die("Parameter does not exist for this source");
	if (anchor_paramnum >= param_anchor_source->n_params) die("Parameter does not exist for source you are anchoring to");
	if (anchor_parameter_to_lens[paramnum]) { warn("cannot anchor a parameter to both a lens and a source object"); return; }
	anchor_parameter_to_source[paramnum] = true;
	parameter_anchor_source[paramnum] = param_anchor_source;
	parameter_anchor_paramnum[paramnum] = anchor_paramnum;
	if ((!use_implicit_ratio) and (!use_exponent)) {
		parameter_anchor_ratio[paramnum] = 1.0;
		(*lensparams->param[paramnum]) = *(param_anchor_source->param[anchor_paramnum]);
	}
	else if (use_implicit_ratio) {
		parameter_anchor_exponent[paramnum] = 1.0;
		if ((*(param_anchor_source->param[anchor_paramnum]))==0) {
			if (*lensparams->param[paramnum]==0) parameter_anchor_ratio[paramnum] = 1.0;
			else die("cannot anchor to parameter with specified ratio if parameter is equal to zero");
		} else {
			parameter_anchor_ratio[paramnum] = (*lensparams->param[paramnum]) / (*(param_anchor_source->param[anchor_paramnum]));
		}
	}
	else if (use_exponent) {
		parameter_anchor_ratio[paramnum] = ratio;
		parameter_anchor_exponent[paramnum] = exponent;
	}
	at_least_one_param_anchored = true;
	update_anchored_parameters();
}

void LensProfile::unanchor_parameter(LensProfile* param_anchor_lens)
{
	// if any parameters are anchored to the lens in question, unanchor them (use this when you are deleting a lens, in case others are anchored to it)
	bool removed_anchor = false;
	for (int i=0; i < n_params; i++) {
		if ((anchor_parameter_to_lens[i]) and (parameter_anchor_lens[i] == param_anchor_lens)) {
			removed_anchor = true;
			parameter_anchor_lens[i] = NULL;
			anchor_parameter_to_lens[i] = false;
			parameter_anchor_paramnum[i] = -1;
			parameter_anchor_ratio[i] = 1.0;
			parameter_anchor_exponent[i] = 1.0;
		}
	}
	if (removed_anchor) {
		at_least_one_param_anchored = false;
		// check to see if any anchors are left
		for (int i=0; i < n_params; i++) {
			if ((anchor_parameter_to_lens[i]) or (anchor_parameter_to_source[i])) {
				at_least_one_param_anchored = true;
				break;
			}
		}
	}
}

void LensProfile::unanchor_parameter(SB_Profile* param_anchor_source)
{
	// if any parameters are anchored to the source in question, unanchor them (use this when you are deleting a source, in case others are anchored to it)
	bool removed_anchor = false;
	for (int i=0; i < n_params; i++) {
		if ((anchor_parameter_to_source[i]) and (parameter_anchor_source[i] == param_anchor_source)) {
			removed_anchor = true;
			parameter_anchor_source[i] = NULL;
			anchor_parameter_to_source[i] = false;
			parameter_anchor_paramnum[i] = -1;
			parameter_anchor_ratio[i] = 1.0;
			parameter_anchor_exponent[i] = 1.0;
		}
	}
	if (removed_anchor) {
		at_least_one_param_anchored = false;
		// check to see if any anchors are left
		for (int i=0; i < n_params; i++) {
			if ((anchor_parameter_to_lens[i]) or (anchor_parameter_to_source[i])) {
				at_least_one_param_anchored = true;
				break;
			}
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
	set_geometric_paramnames(lensprofile_nparams);
}

void LensProfile::set_geometric_paramnames(int qi)
{
	if (!ellipticity_gradient) {
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
			if (!transform_center_coords_to_pixsrc_frame) {
				paramnames[qi] = "xc"; latex_paramnames[qi] = "x"; latex_param_subscripts[qi] = "c";
			} else {
				paramnames[qi] = "delta_xc"; latex_paramnames[qi] = "\\Delta x"; latex_param_subscripts[qi] = "c";
			}
			if (lensed_center_coords) {
				paramnames[qi] += "_l";
				latex_param_subscripts[qi] += ",l";
			}
			qi++;
			if (!transform_center_coords_to_pixsrc_frame) {
				paramnames[qi] = "yc"; latex_paramnames[qi] = "y"; latex_param_subscripts[qi] = "c";
			} else {
				paramnames[qi] = "delta_yc"; latex_paramnames[qi] = "\\Delta y"; latex_param_subscripts[qi] = "c";
			}
			if (lensed_center_coords) {
				paramnames[qi] += "_l";
				latex_param_subscripts[qi] += ",l";
			}
			qi++;
		}
	} else {
		set_geometric_paramnames_egrad(paramnames, latex_paramnames, latex_param_subscripts, qi, ",src");
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
	paramnames[qi] = "z"; latex_paramnames[qi] = "z"; latex_param_subscripts[qi] = "l"; qi++;
}

void LensProfile::assign_param_pointers()
{
	LensSpline_Params<double>& p = assign_lensspline_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.param[0] = &p.qx_parameter;
	p.param[1] = &p.f_parameter;
	set_geometric_param_pointers(lensprofile_nparams);
}

void LensProfile::set_geometric_param_pointers(int qi)
{
	// Sets parameter pointers for ellipticity (or axis ratio) and angle
	LensParams<double>& p = assign_lensparam_object<double>();

	if (!ellipticity_gradient) {
		if (use_ellipticity_components) {
			p.param[qi++] = &p.epsilon1;
			p.param[qi] = &p.epsilon2;
			angle_param[qi++] = false;
			angle_param_exists = false; // there is no angle parameter if ellipticity components are being used
			ellipticity_paramnum = -1; // no single ellipticity parameter here
		} else {
			if ((ellipticity_mode==2) or (ellipticity_mode==3))
				p.param[qi] = &p.epsilon;
			else
				p.param[qi] = &p.q;
			ellipticity_paramnum = qi++;
			p.param[qi] = &p.theta;
			angle_param[qi++] = true;
			angle_param_exists = true;
		}
		if ((!lensed_center_coords) and (!transform_center_coords_to_pixsrc_frame)) {
			p.param[qi++] = &p.x_center;
			p.param[qi++] = &p.y_center;
		} else {
			p.param[qi++] = &p.xc_prime;
			p.param[qi++] = &p.yc_prime;
		}
	} else {
		angle_param_exists = true;
		set_geometric_param_pointers_egrad(p.param,angle_param,qi); // NOTE: if fourier_gradient is turned on, the Fourier parameter pointers are also set in this function
		// Still need to make lensed_center_coords compatible with egrad (or else forbid having both turned on!!!)
	}
	if ((!fourier_gradient) and (n_fourier_modes > 0)) {
		for (int i=0; i < n_fourier_modes; i++) {
			p.param[qi++] = &fourier_mode_cosamp[i];
			p.param[qi++] = &fourier_mode_sinamp[i];
		}
	}
	p.param[qi++] = &zlens;
}

void LensProfile::set_geometric_parameters(const double &par1_in, const double &par2_in, const double &xc_in, const double &yc_in)
{
	/* TRYING OUT THE LENSPARAMS VERSION */
	LensParams<double>& p = assign_lensparam_object<double>();
	if (use_ellipticity_components) {
		p.epsilon1 = par1_in;
		p.epsilon2 = par2_in;
	} else {
		set_ellipticity_parameter(par1_in);
		p.theta = degrees_to_radians(par2_in);
	}
	if ((!lensed_center_coords) and (!transform_center_coords_to_pixsrc_frame)) {
		p.x_center = xc_in;
		p.y_center = yc_in;
	} else {
		p.xc_prime = xc_in;
		p.yc_prime = yc_in;
		set_center_if_lensed_coords();
	}
	update_ellipticity_meta_parameters();

	/*
	LensParams<stan::math::var>& p2 = assign_lensparam_object<stan::math::var>();
	if (use_ellipticity_components) {
		p2.epsilon1 = par1_in;
		p2.epsilon2 = par2_in;
	} else {
		set_ellipticity_parameter(par1_in);
		p2.theta = degrees_to_radians(par2_in);
	}
	if ((!lensed_center_coords) and (!transform_center_coords_to_pixsrc_frame)) {
		p2.x_center = xc_in;
		p2.y_center = yc_in;
	} else {
		p2.xc_prime = xc_in;
		p2.yc_prime = yc_in;
		set_center_if_lensed_coords();
	}
	*/

}

void LensProfile::set_ellipticity_parameter(const double &param_in)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	if (ellipticity_mode==0) {
		p.q = param_in; // axis ratio q = b/a
	} else if (ellipticity_mode==1) {
		p.q = param_in; // axis ratio q = b/a
	} else if (ellipticity_mode==2) {
		p.epsilon = param_in; // axis ratio q = b/a
		p.q = (1-p.epsilon)/(1+p.epsilon);
			cout << "q=" << p.q << endl;
	} else if (ellipticity_mode==3) {
		p.epsilon = param_in; // axis ratio q = b/a
		p.q = sqrt((1-p.epsilon)/(1+p.epsilon));
	}
	if (p.q > 1) p.q = 1.0; // don't allow q>1
	if (p.q<=0) p.q = 0.001; // just to avoid catastrophe
}

void LensProfile::set_center_if_lensed_coords()
{
	LensParams<double>& p = assign_lensparam_object<double>();
	if (lensed_center_coords) {
		if (qlens==NULL) die("Cannot use lensed center coordinates if pointer to QLens object hasn't been assigned");
		lensvector<double> xl;
		qlens->map_to_lens_plane<double>(qlens->lens_redshift_idx[lens_number],p.xc_prime,p.yc_prime,xl,0,qlens->reference_zfactors,qlens->default_zsrc_beta_factors);
		p.x_center = xl[0];
		p.y_center = xl[1];
	}
}

double LensProfile::concentration_prior()
{
	return 0; // this prior is only used in the NFW-like models
}

void LensProfile::change_pmode(const int pmode_in) // WARNING! This does not check whether pmode exists or not for given lens
{
	parameter_mode = pmode_in;
	assign_param_pointers();
	assign_paramnames();
	update_meta_parameters_and_pointers();
}

bool LensProfile::output_cosmology_info(const int lens_number)
{
	bool mass_converged, rhalf_converged;
	double sigma_cr, mtot, rhalf;
	mass_converged = calculate_total_scaled_mass(mtot);
	if (mass_converged) {
		rhalf_converged = calculate_half_mass_radius(rhalf,mtot);
		sigma_cr = cosmo->sigma_crit_arcsec(zlens,zsrc_ref);
		mtot *= sigma_cr;
		if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
		cout << "total mass: " << mtot << " M_sol" << endl;
		//double kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(zlens);
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
	convergence_everywhere = true;
	converge_at_small_r = true;
	first_convergence = false;
	Patterson patterson;
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

		rho3d_logx_spline = new Spline<double>(logxvals,rho3dvals,rho3d_nn);
		mass_intval = re_average;

		patterson.SetGaussPatterson(quadtolerance,false);
		mass3d = 4*M_PI*patterson.AdaptiveQuad([this](auto x) { return mass3d_r_integrand(x); },Rmin,r,menc_converged);
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
	if ((integration_warnings) and (menc_converged==false)) warn("Gauss-Patterson quadrature did not converge for enclosed mass integral (tol=%g) using NMAX=511 points",quadtolerance);

	return mass3d;
}

double LensProfile::calculate_scaled_mass_3d_from_analytic_rho3d(const double r)
{
	bool menc_converged;
	double mass3d, tolerance = 1e-4;

	Patterson patterson(tolerance,false);
	mass3d = 4*M_PI*patterson.AdaptiveQuad([this](auto x) { return mass3d_r_integrand_analytic(x); },0,r,menc_converged);
	if ((integration_warnings) and (menc_converged==false)) warn("Gauss-Patterson quadrature did not converge for enclosed mass integral (tol=%g) using NMAX=511 points",tolerance);
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
	Patterson patterson(tolerance,false);
	patterson.SetGaussPatterson(tolerance,false);
	double ans = -(2*r/M_PI)*patterson.AdaptiveQuad([this](auto x) { return mass3d_r_integrand_analytic(x); },0,1,converged);
	return ans;
}

double LensProfile::rho3d_w_integrand(const double w)
{
	double wsq = w*w;
	return kappa_rsq_deriv(mass_intval/wsq)/(wsq*sqrt(1-wsq));
}

void LensProfile::set_angle_from_components(const double &comp1, const double &comp2)
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
	while (angle > M_HALFPI) angle -= M_PI;
	while (angle <= -M_HALFPI) angle += M_PI;
	set_angle_radians(angle);
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
	kapavgptr_rsq_spherical = &LensProfile::kappa_avg_spherical_integral<double>;
	potptr_rsq_spherical = &LensProfile::potential_spherical_integral<double>;
	if ((lensparams->q==1.0) and (!ellipticity_gradient)) {
		potptr = &LensProfile::potential_spherical_default;
		defptr = &LensProfile::deflection_spherical_default;
		hessptr = &LensProfile::hessian_spherical_default;
	} else {
		defptr = &LensProfile::deflection_numerical;
		hessptr = &LensProfile::hessian_numerical;
	}
	def_and_hess_ptr = &LensProfile::deflection_and_hessian_together;

#ifdef USE_STAN
	kapavgptr_rsq_spherical_autodif = &LensProfile::kappa_avg_spherical_integral<stan::math::var>;
	potptr_rsq_spherical_autodif = &LensProfile::potential_spherical_integral<stan::math::var>;
#endif
}

template <typename QScalar>
QScalar LensProfile::kappa_rsq_impl(const QScalar rsq) // this function should be redefined in all derived classes
{
	LensSpline_Params<QScalar>& p = assign_lensspline_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar r = sqrt(rsq);

	if (r < p.qx_parameter*p.kspline.xmin()) return (p.f_parameter*p.kspline.extend_inner_logslope(r/p.qx_parameter));
	if (r > p.qx_parameter*p.kspline.xmax()) return (p.f_parameter*p.kspline.extend_outer_logslope(r/p.qx_parameter));
	return (p.f_parameter*p.kspline.splint(r/p.qx_parameter));
}
template double LensProfile::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var LensProfile::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar LensProfile::kappa_rsq_deriv_impl(const QScalar rsq)
{
	LensSpline_Params<QScalar>& p = assign_lensspline_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	static const QScalar precision = 1e-6;
	QScalar temp, h;
	h = precision*rsq;
	temp = rsq + h;
	h = temp - rsq; // silly NR trick
	return (kappa_rsq((rsq+h)/(p.qx_parameter*p.qx_parameter))-kappa_rsq((rsq-h)/(p.qx_parameter*p.qx_parameter)))/(2*h);
}
template double LensProfile::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var LensProfile::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

void LensProfile::deflection_from_elliptical_potential(const double x, const double y, lensvector<double>& def)
{
	LensParams<double>& p = assign_lensparam_object<double>();

	// Formulas derived in Dumet-Montoya et al. (2012)
	double kapavg = (this->*kapavgptr_rsq_spherical)((1-p.epsilon)*x*x + (1+p.epsilon)*y*y);

	def[0] = kapavg*(1-p.epsilon)*x;
	def[1] = kapavg*(1+p.epsilon)*y;
}

void LensProfile::hessian_from_elliptical_potential(const double x, const double y, lensmatrix<double>& hess)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// Formulas derived in Dumet-Montoya et al. (2012)
	double cos2phi, sin2phi, exsq, eysq, rsq, gamma1, gamma2, kap_r, shearmag, kap;
	exsq = (1-p.epsilon)*x*x; // elliptical x^2
	eysq = (1+p.epsilon)*y*y; // elliptical y^2
	rsq = exsq+eysq; // elliptical r^2
	cos2phi = (exsq - eysq) / rsq;
	sin2phi = 2*p.q*(1+p.epsilon)*x*y/rsq;
	kap_r = kappa_rsq(rsq);
	shearmag = ((this->*kapavgptr_rsq_spherical)(rsq)) - kap_r; // shear from the spherical model
	kap = kap_r + p.epsilon*shearmag*cos2phi;
	gamma1 = -p.epsilon*kap_r - shearmag*cos2phi;
	gamma2 = -sqrt(1-p.epsilon*p.epsilon)*shearmag*sin2phi;
	hess[0][0] = kap + gamma1;
	hess[1][1] = kap - gamma1;
	hess[0][1] = gamma2;
	hess[1][0] = gamma2;
	if ((shearmag * 0.0) != 0.0) die("die2");
}

double LensProfile::kappa_from_elliptical_potential(const double x, const double y)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// Formulas derived in Dumet-Montoya et al. (2012)
	double cos2phi, exsq, eysq, rsq, kap_r, shearmag;
	exsq = (1-p.epsilon)*x*x; // elliptical x^2
	eysq = (1+p.epsilon)*y*y; // elliptical y^2
	rsq = exsq+eysq; // elliptical r^2
	cos2phi = (exsq - eysq) / rsq;

	kap_r = kappa_rsq(rsq);
	shearmag = (this->*kapavgptr_rsq_spherical)(rsq) - kap_r;

	return (kap_r + p.epsilon*shearmag*cos2phi);
}

void LensProfile::kappa_deflection_and_hessian_from_elliptical_potential(const double x, const double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// Formulas derived in Dumet-Montoya et al. (2012)
	double cos2phi, sin2phi, exsq, eysq, rsq, gamma1, gamma2, kap_r, shearmag, kapavg;
	exsq = (1-p.epsilon)*x*x; // elliptical x^2
	eysq = (1+p.epsilon)*y*y; // elliptical y^2
	rsq = exsq+eysq; // elliptical r^2

	kapavg = (this->*kapavgptr_rsq_spherical)(rsq);

	def[0] = kapavg*(1-p.epsilon)*x;
	def[1] = kapavg*(1+p.epsilon)*y;

	cos2phi = (exsq - eysq) / rsq;
	sin2phi = 2*p.q*(1+p.epsilon)*x*y/rsq;
	kap_r = kappa_rsq(rsq);
	shearmag = kapavg - kap_r; // shear from the spherical model
	kap = kap_r + p.epsilon*shearmag*cos2phi;
	gamma1 = -p.epsilon*kap_r - shearmag*cos2phi;
	gamma2 = -sqrt(1-p.epsilon*p.epsilon)*shearmag*sin2phi;
	hess[0][0] = kap + gamma1;
	hess[1][1] = kap - gamma1;
	hess[0][1] = gamma2;
	hess[1][0] = gamma2;
}

void LensProfile::shift_angle_90()
{
	// do this if the major axis orientation is changed (so the lens angles values are changed appropriately, even though the lens doesn't change)
	lensparams->theta += M_HALFPI;
	while (lensparams->theta > M_PI) lensparams->theta -= M_PI;
}

void LensProfile::shift_angle_minus_90()
{
	// do this if the major axis orientation is changed (so the lens angles values are changed appropriately, even though the lens doesn't change)
	lensparams->theta -= M_HALFPI;
	while (lensparams->theta <= -M_PI) lensparams->theta += M_PI;
}

void LensProfile::reset_angle_modulo_2pi()
{
	while (lensparams->theta < -M_PI/2) lensparams->theta += 2*M_PI;
	while (lensparams->theta > 2*M_PI) lensparams->theta -= 2*M_PI;
}

void LensProfile::set_angle(const double &theta_degrees)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	p.theta = degrees_to_radians(theta_degrees);
	// trig functions are stored to save computation time later
	p.costheta = cos(p.theta);
	p.sintheta = sin(p.theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		double tmp = p.sintheta;
		p.sintheta = p.costheta;
		p.costheta = -tmp;
	}
}

void LensProfile::set_angle_radians(const double &theta_in)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	p.theta = theta_in;
	// trig functions are stored to save computation time later
	p.costheta = cos(p.theta);
	p.sintheta = sin(p.theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		double tmp = p.sintheta;
		p.sintheta = p.costheta;
		p.costheta = -tmp;
	}
}

void LensProfile::update_cosmology_meta_parameters(const bool force_update)
{
	if ((cosmo != NULL) and ((force_update) or (zlens != zlens_current) or (cosmo->get_n_vary_params() > 0))) {
		sigma_cr = cosmo->sigma_crit_arcsec(zlens,zsrc_ref);
		kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(zlens);
		if (zlens != zlens_current) zlens_current = zlens;
	}
}

void LensProfile::calculate_ellipticity_components()
{
	LensParams<double>& p = assign_lensparam_object<double>();
	if ((ellipticity_mode != -1) and (use_ellipticity_components)) {
		p.theta_eff = (orient_major_axis_north) ? p.theta + M_HALFPI : p.theta;
		if ((ellipticity_mode==0) or (ellipticity_mode==1)) p.epsilon = 1-p.q;
		p.epsilon1 = p.epsilon*cos(2*p.theta_eff);
		p.epsilon2 = p.epsilon*sin(2*p.theta_eff);
	}
}

void LensProfile::update_ellipticity_meta_parameters()
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// f_major_axis sets the major axis of the elliptical radius xi such that a = f*xi, and b = f*q*xi (and thus, xi = sqrt(x^2 + (y/q)^2)/f)
	if (use_ellipticity_components) {
		if ((ellipticity_mode==0) or (ellipticity_mode==1)) set_ellipticity_parameter(1 - sqrt(SQR(p.epsilon1) + SQR(p.epsilon2)));
		else set_ellipticity_parameter(sqrt(SQR(p.epsilon1) + SQR(p.epsilon2)));
		// if ellipticity components are being used, we are automatically using the following major axis scaling
		set_angle_from_components(p.epsilon1,p.epsilon2);
	}

	//if (transform_center_coords_to_pixsrc_frame) update_center_from_pixsrc_coords();

	if (!ellipticity_gradient) {
		if (ellipticity_mode==0) {
			lensparams->epsilon = 1 - lensparams->q;
			f_major_axis = 1.0; // defined such that a = xi, and b = xi*q
		} else if (ellipticity_mode==1) {
			lensparams->epsilon = 1 - lensparams->q;
			// if ellipticity components are being used, we are automatically using the following major axis scaling
			f_major_axis = 1.0/sqrt(lensparams->q); // defined such that a = xi/sqrt(q), and b = xi*sqrt(q)
		} else if (ellipticity_mode==2) {
			lensparams->q = (1-lensparams->epsilon)/(1+lensparams->epsilon);
			f_major_axis = 1.0/sqrt(lensparams->q); // defined such that a = xi/sqrt(q), and b = xi*sqrt(q)
		} else if (ellipticity_mode==3) {
			lensparams->q = sqrt((1-lensparams->epsilon)/(1+lensparams->epsilon));
			f_major_axis = sqrt((1+lensparams->q*lensparams->q)/2)/lensparams->q; // defined such that a = xi/sqrt(1-e), and b = xi/sqrt(1+e), so that q = sqrt((1-e)/(1+e))
		}
		if (!use_ellipticity_components) update_angle_meta_params(); // sets the costheta, sintheta meta-parameters
	} else {
		lensparams->q = geometric_param[0][0];
		if (lensparams->q > 1.0) lensparams->q = 1.0;
		lensparams->theta = geometric_param[1][0];
		lensparams->x_center = geometric_param[2][0];
		lensparams->y_center = geometric_param[3][0];
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

void LensProfile::update_center_from_pixsrc_coords(QLens* qlensptr)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	double xcs, ycs;
	//cout << "lens " << lens_number << " finding approx source size... " << endl;
	// generalize this later so it can be anchored to an image_pixel_grid with the appropriate zsrc_i (doesn't have to be 0)
	qlensptr->find_source_centroid(0,xcs,ycs,false);
	//cout << "lens " << lens_number << " xcs,ycs: " << xcs << " " << ycs << endl;
	p.x_center = xcs + p.xc_prime;
	p.y_center = ycs + p.yc_prime;
}


void LensProfile::update_angle_meta_params()
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// trig functions are stored to save computation time later
	p.costheta = cos(p.theta);
	p.sintheta = sin(p.theta);
	if (orient_major_axis_north==true) {
		// this effectively alters theta by 90 degrees, so that the major axis will point along "north" (i.e. the y-axis)
		double tmp = p.sintheta;
		p.sintheta = p.costheta;
		p.costheta = -tmp;
	}
}

void LensProfile::rotate(double &x, double &y)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// perform a counter-clockwise rotation of the coordinate system to match the coordinate system of the rotated galaxy
	double xp = x*p.costheta + y*p.sintheta;
	y = -x*p.sintheta + y*p.costheta;
	x = xp;
}

void LensProfile::rotate_back(double &x, double &y)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// perform a clockwise rotation of the coordinate system to transform from the coordinate system of the rotated galaxy
	double xp = x*p.costheta - y*p.sintheta;
	y = x*p.sintheta + y*p.costheta;
	x = xp;
}

double LensProfile::elliptical_radius(double x, double y)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// switch to coordinate system centered on lens profile
	x -= p.x_center;
	y -= p.y_center;
	if ((!ellipticity_gradient) and (p.sintheta != 0)) rotate(x,y);
	double ans=0.0, xisq;

	if ((ellipticity_mode==3) and (p.q != 1)) {
		die("cannot get ellipticity radius in emode=3 since ellipticity is in the potential");
	} else {
		double xisq;
		if (!ellipticity_gradient) {
			xisq = (x*x + y*y/(p.q*p.q))/(f_major_axis*f_major_axis);
		} else {
			xisq = SQR(elliptical_radius_root(x,y));
		}
		ans = sqrt(xisq);
	}
	return ans;
}

double LensProfile::kappa(double x, double y)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// switch to coordinate system centered on lens profile
	x -= p.x_center;
	y -= p.y_center;
	if ((!ellipticity_gradient) and (p.sintheta != 0)) rotate(x,y);
	double ans;

	if ((ellipticity_mode==3) and (p.q != 1)) {
		return kappa_from_elliptical_potential(x,y);
	} else {
		double xisq, fourier_factor = 0;
		if (!ellipticity_gradient) {
			xisq = (x*x + y*y/(p.q*p.q))/(f_major_axis*f_major_axis);
		} else {
			xisq = SQR(elliptical_radius_root(x,y));
		}
		ans = kappa_rsq(xisq);
	}
	if (n_fourier_modes > 0) ans += kappa_from_fourier_modes(x,y);
	return ans;
}

void LensProfile::deflection(double x, double y, lensvector<double>& def)
{
	LensParams<double>& p = assign_lensparam_object<double>();

	// switch to coordinate system centered on lens profile
	//if (x*0.0 != 0.0) die("x is fucked going into def function");
	//cout << "CENTER: " << x_center << " " << y_center << endl;
	//if (x_center==1e30) die("x_center has not been set for lens %i",lens_number);
	//if (y_center==1e30) die("y_center has not been set for lens %i",lens_number);
	x -= p.x_center;
	y -= p.y_center;
	//if (x_center*0.0 != 0.0) die("center is fucked");
	//if (x*0.0 != 0.0) die("x is fucked but not center");
	if ((!ellipticity_gradient) and (p.sintheta != 0)) rotate(x,y);
	//if (x*0.0 != 0.0) die("fucked after rotation");
	if ((ellipticity_mode==3) and (p.q != 1)) {
		deflection_from_elliptical_potential(x,y,def);
	} else {
		(this->*defptr)(x,y,def);
	}

	if (n_fourier_modes > 0) {
		add_deflection_from_fourier_modes(x,y,def); // this adds the deflection from Fourier modes
	}
	if ((!ellipticity_gradient) and (p.sintheta != 0)) def.rotate_back(p.costheta,p.sintheta);
}

void LensProfile::hessian(double x, double y, lensmatrix<double>& hess)
{
	LensParams<double>& p = assign_lensparam_object<double>();

	// switch to coordinate system centered on lens profile
	x -= p.x_center;
	y -= p.y_center;
	if ((!ellipticity_gradient) and (p.sintheta != 0)) rotate(x,y);
	if ((ellipticity_mode==3) and (p.q != 1)) {
		hessian_from_elliptical_potential(x,y,hess);
	} else {
		(this->*hessptr)(x,y,hess);
	}
	if (n_fourier_modes > 0) {
		add_hessian_from_fourier_modes(x, y, hess);
	}
	if ((!ellipticity_gradient) and (p.sintheta != 0)) hess.rotate_back(p.costheta,p.sintheta);
}

double LensProfile::potential(double x, double y)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// switch to coordinate system centered on lens profile
	x -= p.x_center;
	y -= p.y_center;
	if ((!ellipticity_gradient) and (p.sintheta != 0)) rotate(x,y);
	if (ellipticity_mode==3) {
		return (this->*potptr_rsq_spherical)((1-p.epsilon)*x*x + (1+p.epsilon)*y*y); // ellipticity is put into the potential in this mode
	} else {
		return (this->*potptr)(x,y);
	}
}

void LensProfile::kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	// switch to coordinate system centered on lens profile
	x -= p.x_center;
	y -= p.y_center;
	if ((!ellipticity_gradient) and (p.sintheta != 0)) rotate(x,y);
	if ((ellipticity_mode==3) and (p.q != 1)) {
		kappa_deflection_and_hessian_from_elliptical_potential(x,y,kap,def,hess);
	} else {
		kap = kappa_rsq((x*x + y*y/(p.q*p.q))/(f_major_axis*f_major_axis));
		(this->*def_and_hess_ptr)(x,y,def,hess);
	}
	if (n_fourier_modes > 0) {
		add_deflection_from_fourier_modes(x,y,def); // this adds the deflection from Fourier modes
		add_hessian_from_fourier_modes(x,y,hess);
	}
	if ((!ellipticity_gradient) and (p.sintheta != 0)) {
		def.rotate_back(p.costheta,p.sintheta);
		hess.rotate_back(p.costheta,p.sintheta);
	}
	//double kapcheck = (hess[0][0]+hess[1][1])/2;
	//if ((!ellipticity_gradient) and (sintheta != 0)) rotate_back(x,y);
	//double kapcheck2 = kappa(x,y);
	//if (abs(kapcheck-kapcheck2) > 1e-3*abs(kapcheck2)) cout << "KAPPACHECK: " << kapcheck << " " << kapcheck2 << "(x=" << x << "," << y << ")" << endl;

}

void LensProfile::potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	double kap;
	kappa_and_potential_derivatives(x,y,kap,def,hess); // including kappa has no noticeable extra overhead
}

void LensProfile::deflection_and_hessian_together(const double x, const double y, lensvector<double> &def, lensmatrix<double>& hess)
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
	// /************ ADD HESSIAN FROM FOURIER MODES ************/ 
}

void LensProfile::kappa_and_dkappa_dR(double x, double y, double& kap, double& dkap)
{
	LensParams<double>& p = assign_lensparam_object<double>();
	if (lenstype==SHEET) {
		kap = kappa_rsq(0); // since R doesn't matter for a mass sheet
		dkap = kappa_rsq_deriv(0);
	} else {
		// switch to coordinate system centered on lens profile
		x -= p.x_center;
		y -= p.y_center;
		if ((!ellipticity_gradient) and (p.sintheta != 0)) rotate(x,y);
		double ell_radius_sq = (x*x + y*y/(p.q*p.q))/(f_major_axis*f_major_axis);
		kap = kappa_rsq(ell_radius_sq);
		dkap = 2*ell_radius_sq*kappa_rsq_deriv(ell_radius_sq)/sqrt(x*x+y*y); // this gives dkappa_dR where R is simply radius, not elliptical radius
	}
}

void LensProfile::add_fourier_mode(const int m_in, const double amp_in, const double amp2_in, const bool vary1, const bool vary2)
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

	vary_params[n_params-1] = vary_params[n_params-3]; // moving redshift to the end
	vary_params[n_params-3] = vary1;
	vary_params[n_params-2] = vary2;

	delete[] lensparams->param;
	lensparams->param = new double*[n_params];
	reset_anchor_lists();
	assign_param_pointers();
	assign_paramnames();
}

void LensProfile::remove_fourier_modes()
{
	// This is not well-written, because it assumes the last parameters (before the zlens parameter) are fourier modes. Have it actually check the parameter name
	// before deleting it. FIX!
	if (n_fourier_modes==0) return;
	vary_params[n_params-1 - n_fourier_modes*2] = vary_params[n_params-1]; // moving zlens parameter back
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

	delete[] lensparams->param;
	lensparams->param = new double*[n_params];
	assign_param_pointers();
}

bool LensProfile::enable_ellipticity_gradient(Vector<double>& efunc_params, const int egrad_mode, const int n_bspline_coefs, const Vector<double>& knots, const double ximin, const double ximax, const double xiref, const bool linear_xivals, const bool copy_vary_settings, boolvector* vary_egrad)
{
	if (ellipticity_mode==-1) return false; // ellipticity gradient only works for lenses that have elliptical isodensity contours
	if (ellipticity_mode > 1) return false; // only emode=0 or 1 is supported right now
	//NOTE: when new ellipticity functions are incorporated, n_efunc_params can be different from 4,
	//      and efunc_params will be able to have different # of parameters, etc.

	
	if ((egrad_mode==0) and (efunc_params[0]==-1e30)) { // in this case, the egrad params were never initialized
		// Not sure if I should do this here, or before calling enable_ellipticity_gradient?
		efunc_params.input(2*n_bspline_coefs+2);
		for (int i=0; i < n_bspline_coefs; i++) efunc_params[i] = lensparams->q;
		for (int i=n_bspline_coefs; i < 2*n_bspline_coefs; i++) efunc_params[i] = radians_to_degrees(lensparams->theta);
		efunc_params[2*n_bspline_coefs] = lensparams->x_center;
		efunc_params[2*n_bspline_coefs+1] = lensparams->y_center;
	}

	int n_egrad_params;
	if (setup_egrad_params(egrad_mode,ellipticity_mode,efunc_params,n_egrad_params,n_bspline_coefs,knots,ximin,ximax,xiref,linear_xivals)==false) {
		warn("could not set up egrad params properly");
		return false;
	}
	int new_nparams = n_params + n_egrad_params - 4; // we already had q, theta, xc and yc
	//cout << "WTF? " << n_egrad_params << " " << n_params << " " << new_nparams << " " << n_bspline_coefs << endl;
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
	for (int i=lensprofile_nparams+4; i < n_params; i++) {
		vary_params[i+n_egrad_params-4] = vary_params[i];
		angle_param[i+n_egrad_params-4] = angle_param[i];
	}
	for (int i=0; i < n_fourier_modes; i++) fourier_mode_paramnum[i] += n_egrad_params - 4; // so it keeps track of where the Fourier modes are
	int j=0;
	for (int i=lensprofile_nparams; i < lensprofile_nparams + n_egrad_params; i++) {
		if (!copy_vary_settings) vary_params[i] = false;
		else vary_params[i] = (*vary_egrad)[j++];
		angle_param[i] = false; // the angle params will be set when the param pointers are set
	}
	n_params = new_nparams;
	delete[] lensparams->param;
	lensparams->param = new double*[n_params];

	reset_anchor_lists();
	assign_param_pointers();
	assign_paramnames();

	update_ellipticity_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
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

void LensProfile::reset_anchor_lists()
{
	if (anchor_parameter_to_lens != NULL) delete[] anchor_parameter_to_lens;
	if (parameter_anchor_lens != NULL) delete[] parameter_anchor_lens;
	if (anchor_parameter_to_source != NULL) delete[] anchor_parameter_to_source;
	if (parameter_anchor_source != NULL) delete[] parameter_anchor_source;
	if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
	if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
	if (parameter_anchor_exponent != NULL) delete[] parameter_anchor_exponent;

	anchor_parameter_to_lens = new bool[n_params];
	parameter_anchor_lens = new LensProfile*[n_params];
	anchor_parameter_to_source = new bool[n_params];
	parameter_anchor_source = new SB_Profile*[n_params];
	parameter_anchor_paramnum = new int[n_params];
	parameter_anchor_ratio = new double[n_params];
	parameter_anchor_exponent = new double[n_params];

	// parameters should not be anchored before enable egrad, since the anchors are deleted here
	for (int i=0; i < n_params; i++) {
		anchor_parameter_to_lens[i] = false;
		parameter_anchor_lens[i] = NULL;
		anchor_parameter_to_source[i] = false;
		parameter_anchor_source[i] = NULL;
		parameter_anchor_paramnum[i] = -1;
		parameter_anchor_ratio[i] = 1.0;
		parameter_anchor_exponent[i] = 1.0;
	}
	at_least_one_param_anchored = false;
}

bool LensProfile::enable_fourier_gradient(Vector<double>& fourier_params, const Vector<double>& knots, const bool copy_vary_settings, boolvector* vary_fgrad)
{
	if (ellipticity_mode==-1) return false; // Fourier gradient only works for lenses that have elliptical isodensity contours
	if (ellipticity_mode > 1) return false; // only emode=0 or 1 is supported right now
	if (n_fourier_modes==0) return false; // Fourier modes must already be present

	if (egrad_mode==0) {
		int n_bspline_coefs = n_bspline_knots_tot - bspline_order - 1;
		// Not sure if I should do this here, or before calling enable_fourier_gradient?
		fourier_params.input(2*n_fourier_modes*n_bspline_coefs);
		int i,j,k;
		for (k=0; k < 2*n_fourier_modes; k++) {
			for (i=0,j=0; i < n_bspline_coefs; i++, j++) {
				fourier_params[j] = 0.0;
			}
		}
	}

	int n_fourier_grad_params;
	//NOTE: here we assume that ximin, ximax were set when enabling ellipticity gradient
	if (setup_fourier_grad_params(n_fourier_modes,fourier_mode_mvals,fourier_params,n_fourier_grad_params,knots)==false) {
		warn("could not set up fgrad params properly");
		return false;
	}
	int param_ndif = n_fourier_grad_params - 2*n_fourier_modes; // we already had q, theta, xc and yc
	int new_nparams = n_params + param_ndif; // we already had q, theta, xc and yc
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
	n_params = new_nparams;
	delete[] lensparams->param;
	lensparams->param = new double*[n_params];

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

void LensProfile::find_egrad_paramnums(int& qi, int& qf, int& theta_i, int& theta_f, int& amp_i, int& amp_f)
{
	qi = lensprofile_nparams;
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
	if ((einstein_radius_root(rmin_einstein_radius)*einstein_radius_root(rmax_einstein_radius)) > 0) {
		// multiple imaging does not occur with this lens
		re_major_axis = 0;
		re_average = 0;
		return;
	}
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&LensProfile::einstein_radius_root);
	re_major_axis = f_major_axis * BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-6);
	if (ellipticity_mode != -1) re_average = re_major_axis * sqrt(lensparams->q);
	else re_average = re_major_axis; // not an elliptical lens, so q has no meaning
	zfac = 1.0;
}

double LensProfile::einstein_radius()
{
	// uses qlens->reference_zfactors, and give re_average only
	double re_maj, re_avg;
	if (qlens==NULL) {
		warn("lens has not been added to qlens list; cannot find Einstein radius");
		return 0.0;
	}
	get_einstein_radius(re_maj,re_avg,qlens->reference_zfactors[qlens->lens_redshift_idx[lens_number]]);
	return re_avg;
}

double LensProfile::get_xi_parameter(const double zfactor)
{
	double r_ein, re_sq, kappa_e, dkappa_e;
	if (kapavgptr_rsq_spherical==NULL) {
		return -1e30;
	}
	zfac = zfactor;
	if ((einstein_radius_root(rmin_einstein_radius)*einstein_radius_root(rmax_einstein_radius)) >= 0) {
		// multiple imaging does not occur with this lens
		return -1e30;
	}
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&LensProfile::einstein_radius_root);
	r_ein = BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-6);

	re_sq = r_ein*r_ein;
	kappa_e = zfactor*kappa_rsq(re_sq);
	dkappa_e = 2*r_ein*zfactor*kappa_rsq_deriv(re_sq); // we express xi in terms of derivative of kappa, rather than second derivative of the deflection
	zfac = 1.0;
	return (2*r_ein*dkappa_e/(1-kappa_e)+2);
}

double LensProfile::get_inner_logslope()
{
	static const double h = 1e-6;
	double dlogh;
	dlogh = log(h);
	return ((log(kappa_rsq(SQR(exp(2*dlogh)))) - log(kappa_rsq(h*h))) / dlogh);
}

void LensProfile::plot_kappa_profile(double rmin, double rmax, int steps, ofstream &kout, ofstream &kdout)
{
	double r, rstep;
	rstep = pow(rmax/rmin, 1.0/steps);
	int i;
	kout << setiosflags(ios::scientific);
	if (kdout.is_open()) kdout << setiosflags(ios::scientific);
	double kavg, rsq, r_kpc, rho3d, scaled_rho;
	bool converged;
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		rsq = r*r;
		if (kapavgptr_rsq_spherical==NULL) kavg=0; // just in case there is no radial deflection function defined
		else kavg = (this->*kapavgptr_rsq_spherical)(rsq);
		r_kpc = r/kpc_to_arcsec;
		scaled_rho = calculate_scaled_density_3d(r,1e-4,converged);
		rho3d = (sigma_cr*CUBE(kpc_to_arcsec))*scaled_rho;

		kout << r << " " << kappa_rsq(rsq) << " " << kavg << " " << kavg*r << " " << M_PI*kavg*rsq*sigma_cr << " " << r_kpc << " " << kappa_rsq(rsq)*sigma_cr << " " << rho3d << endl;
		if (kdout.is_open()) kdout << r << " " << 2*r*kappa_rsq_deriv(rsq) << endl;
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

void LensProfile::deflection_spherical_default(const double x, const double y, lensvector<double>& def)
{
	double kapavg = (this->*kapavgptr_rsq_spherical)(x*x+y*y);

	def[0] = kapavg*x;
	def[1] = kapavg*y;
}

double LensProfile::potential_spherical_default(const double x, const double y)
{
	return (this->*potptr_rsq_spherical)(x*x+y*y); // ellipticity is put into the potential in this mode
}

template <typename QScalar>
QScalar LensProfile::kapavg_spherical_generic(const QScalar rsq)
{
#ifdef USE_STAN
	if constexpr (std::is_same_v<QScalar, stan::math::var>) {
		cout << "WTF is going wrong?" << endl;
		return (this->*kapavgptr_rsq_spherical_autodif)(rsq);
	}
	else
#endif
	return (this->*kapavgptr_rsq_spherical)(rsq);
}

template <typename QScalar>
QScalar LensProfile::kappa_avg_spherical_integral(const QScalar rsq)
{
#ifdef USE_STAN
	cout << "HERGGLEHARGLE?" << endl;
#endif
	QScalar ans;

	if (integral_method == Romberg_Integration)
	{
		Romberg<std::function<QScalar(const QScalar)>,QScalar> romberg;
		ans = (2.0/rsq)*romberg.integrate_open([this](auto x) { return mass_enclosed_spherical_integrand(x); }, 0, sqrt(rsq), integral_tolerance, 5);
	}
	else if (integral_method == Gaussian_Quadrature)
	{
		GaussLegendre<std::function<QScalar(const QScalar)>,QScalar> gauss;
		ans = (2.0/rsq)*gauss.NIntegrate([this](auto x) { return mass_enclosed_spherical_integrand(x); },0,sqrt(rsq));
	}
	else if (integral_method == Gauss_Patterson_Quadrature)
	{
		bool converged;
		GaussPatterson<std::function<QScalar(const QScalar)>,QScalar> patterson(integral_tolerance,integration_warnings);
		ans = (2.0/rsq)*patterson.AdaptiveQuad([this](auto x) { return mass_enclosed_spherical_integrand(x); },0,sqrt(rsq),converged);
	}
	else if (integral_method == Fejer_Quadrature)
	{
		bool converged;
		ClenshawCurtis<std::function<QScalar(const QScalar)>,QScalar> fejer(integral_tolerance,integration_warnings);
		ans = (2.0/rsq)*fejer.AdaptiveQuadCC([this](auto x) { return mass_enclosed_spherical_integrand(x); },0,sqrt(rsq),converged);
	}
	else die("unknown integral method");
	return ans;
}
template double LensProfile::kappa_avg_spherical_integral<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var LensProfile::kappa_avg_spherical_integral<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar LensProfile::mass_enclosed_spherical_integrand(const QScalar u)
{
	return u*kappa_rsq(u*u);
} // actually mass enclosed / (2*pi*sigma_cr)
template double LensProfile::mass_enclosed_spherical_integrand<double>(const double u);
#ifdef USE_STAN
template stan::math::var LensProfile::mass_enclosed_spherical_integrand<stan::math::var>(const stan::math::var u);
#endif


void LensProfile::hessian_spherical_default(const double x, const double y, lensmatrix<double>& hess)
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

template <typename QScalar>
QScalar LensProfile::potential_spherical_integral(const QScalar rsq)
{
	using std::sqrt;
	bool converged;
	QScalar ans;
	LensIntegral<QScalar> lens_integral(this,sqrt(rsq),0,1.0);
	ans = 0.5*lens_integral.i_integral(converged);

#ifdef USE_STAN
	stan::math::var anss;
	LensIntegral<stan::math::var> lens_integral_stan(this,sqrt(rsq),0,1.0);
	cout << "HI3" << endl;
	anss = 0.5*lens_integral_stan.i_integral(converged);
	cout << "ans=" << ans << endl;
	cout << "stan ans=" << anss << endl;
#endif

	return ans;
}

void LensProfile::deflection_numerical(const double x, const double y, lensvector<double>& def)
{
	if ((abs(x) < 1e-14) and (abs(y) < 1e-14)) {
		// return zero deflection, since there's a risk of getting 'NaN' if the center of the profile is evaluated
		def[0]=0;
		def[1]=0;
		return;
	}

	bool converged;
	if (!ellipticity_gradient) {
		//cout << "NOT DOING EGRAD" << endl;
		LensIntegral<double> lens_integral(this,x,y,lensparams->q);
		def[0] = x*lens_integral.j_integral(0,converged);
		warn_if_not_converged(converged,x,y);
		def[1] = y*lens_integral.j_integral(1,converged);
		warn_if_not_converged(converged,x,y);
		//cout << "j0=" << (def[0]/x) << " j1=" << (def[1]/y) << endl;
	} else {
		//cout << "DOING EGRAD" << endl;
		/*
		double jint0, jint1, jint2;
		//cout << "FIRST INTEGRAL" << endl;
		jint0 = lens_integral.j_integral_egrad(0,converged);
		warn_if_not_converged(converged,x,y);
		//cout << "SECOND INTEGRAL" << endl;
		jint1 = lens_integral.j_integral_egrad(1,converged);
		warn_if_not_converged(converged,x,y);
		//cout << "THIRD INTEGRAL" << endl;
		jint2 = lens_integral.j_integral_egrad(2,converged);
		warn_if_not_converged(converged,x,y);
		//cout << "DONE" << endl;
		//cout << "j0=" << jint0 << " j1=" << jint1 << " j2=" << jint2 << endl;
		// Later, switch this to the two-integral version to save time, since we aren't getting the hessian here
		def[0] = x*jint0 + y*jint1;
		def[1] = x*jint1 + y*jint2;
		LensIntegral lens_integral(this,x,y,lensparams->q);
		def[0] = lens_integral.jprime_integral_egrad(0,converged);
		warn_if_not_converged(converged,x,y);
		def[1] = lens_integral.jprime_integral_egrad(1,converged);
		warn_if_not_converged(converged,x,y);
		*/

		LensIntegral<double> lens_integral2(this,x,y,lensparams->q,2);
		lens_integral2.jprime_integral_egrad_mult(def.array(),converged);
		warn_if_not_converged(converged,x,y);
		/*
		double defcheck[2];
		lens_integral2.jprime_integral_egrad_mult(defcheck,converged);
		if (abs(def[0]-defcheck[0]) > 1e-3) {
			cout << "UH-OH! DEFCHECK_defx: " << def[0] << " " << defcheck[0] << endl;
		}
		if (abs(def[1]-defcheck[1]) > 1e-3) {
			cout << "UH-OH! DEFCHECK_defy: " << def[1] << " " << defcheck[1] << endl;
		}
		*/
	}
}

void LensProfile::hessian_numerical(const double x, const double y, lensmatrix<double>& hess)
{
	if ((abs(x) < 1e-14) and (abs(y) < 1e-14)) {
		hess[0][0]=0;
		hess[1][1]=0;
		hess[0][1]=0;
		hess[1][0]=0;
		return;
	}

	bool converged;
	if (!ellipticity_gradient) {
		LensIntegral<double> lens_integral(this,x,y,lensparams->q);
		double jint0, jint1;
		jint0 = lens_integral.j_integral(0,converged);
		warn_if_not_converged(converged,x,y);
		jint1 = lens_integral.j_integral(1,converged);
		warn_if_not_converged(converged,x,y);
		hess[0][0] = 2*x*x*lens_integral.k_integral(0,converged) + jint0;
		warn_if_not_converged(converged,x,y);
		hess[1][1] = 2*y*y*lens_integral.k_integral(2,converged) + jint1;
		warn_if_not_converged(converged,x,y);
		hess[0][1] = 2*x*y*lens_integral.k_integral(1,converged);
		warn_if_not_converged(converged,x,y);
		//double hess00 = lens_integral.k_integral(0,converged);
		//double hess01 = lens_integral.k_integral(1,converged);
		//double hess11 = lens_integral.k_integral(2,converged);
		/*
		double kap2 = 2*kappa_rsq((x*x + y*y/(lensparams->q*lensparams->q))/(f_major_axis*f_major_axis));
		double laplacian = hess[0][0] + hess[1][1];
		//if (abs(laplacian-kap2) > 1e-3*abs(kap2)) {
			//cout << "Check Laplacian (e=const): " << laplacian << " " << kap2 << " (x=" << x << "," << y << ")" << " errorfrac=" << (abs(laplacian-kap2)/kap2) << endl;
			double oldtol = cc_tolerance;
			set_integral_tolerance(1e-6);
			double jint0check = lens_integral.j_integral(0,converged);
			warn_if_not_converged(converged,x,y);
			double jint1check = lens_integral.j_integral(1,converged);
			warn_if_not_converged(converged,x,y);
			if (abs((jint0-jint0check)/jint0check) > 1e-3)
			cout << "j0: " << jint0 << " " << jint0check << " " << (abs((jint0-jint0check)/jint0check)) << " (x=" << x << "," << y << ")" << endl;
			if (abs((jint1-jint1check)/jint1check) > 1e-3)
			cout << "j1: " << jint1 << " " << jint1check << " " << (abs((jint1-jint1check)/jint1check)) << " (x=" << x << "," << y << ")" << endl;
			set_integral_tolerance(oldtol);
		//}
		*/

		//cout << "HESS: " << hess[0][0] << " " << hess[1][1] << " " << hess[0][1] << endl;
	} else {
		/*
		LensIntegral<double> lens_integral(this,x,y,lensparams->q);
		double jint0, jint1, jint2;
		double kint0, kint1, kint2;
		jint0 = lens_integral.j_integral_egrad(0,converged);
		warn_if_not_converged(converged,x,y);
		jint1 = lens_integral.j_integral_egrad(1,converged);
		warn_if_not_converged(converged,x,y);
		jint2 = lens_integral.j_integral_egrad(2,converged);
		warn_if_not_converged(converged,x,y);

		kint0 = lens_integral.k_integral_egrad(0,converged);
		warn_if_not_converged(converged,x,y);
		kint1 = lens_integral.k_integral_egrad(1,converged);
		warn_if_not_converged(converged,x,y);
		kint2 = lens_integral.k_integral_egrad(2,converged);
		warn_if_not_converged(converged,x,y);

		hess[0][0] = 2*kint0 + jint0;
		warn_if_not_converged(converged,x,y);
		hess[0][1] = 2*kint1 + jint1;
		warn_if_not_converged(converged,x,y);
		hess[1][1] = 2*kint2 + jint2;
		warn_if_not_converged(converged,x,y);
		*/

		//hess[0][0] = 2*lens_integral.k_integral_egrad(0,converged) + jint0;
		//warn_if_not_converged(converged,x,y);
		//hess[0][1] = 2*lens_integral.k_integral_egrad(1,converged) + jint1;
		//warn_if_not_converged(converged,x,y);
		//hess[1][1] = 2*lens_integral.k_integral_egrad(2,converged) + jint2;
		//warn_if_not_converged(converged,x,y);

		LensIntegral<double> lens_integral3(this,x,y,lensparams->q,3);
		double jint[3], kint[3];
		lens_integral3.j_integral_egrad_mult(jint,converged);
		warn_if_not_converged(converged,x,y);
		lens_integral3.k_integral_egrad_mult(kint,converged);
		warn_if_not_converged(converged,x,y);
		hess[0][0] = 2*kint[0] + jint[0];
		hess[0][1] = 2*kint[1] + jint[1];
		hess[1][1] = 2*kint[2] + jint[2];

		//cout << "kint0=" << kint0 << " Kint0=" << kint[0] << ", kint1=" << kint1 << " Kint1=" << kint[1] << " kint2=" << kint2 << " Kint2=" << kint[2] << endl;
		//double hess00 = 2*kint[0] + jint[0];
		//double hess01 = 2*kint[1] + jint[1];
		//double hess11 = 2*kint[2] + jint[2];

		//double hess00 = lens_integral.k_integral_egrad(0,converged);
		//double hess01 = lens_integral.k_integral_egrad(1,converged);
		//double hess11 = lens_integral.k_integral_egrad(2,converged);
		//cout << "hess00=" << hess00 << " hess01=" << hess01 << " hess11=" << hess11 << "Hess00=" << hess[0][0] << " Hess01=" << hess[0][1] << " Hess11=" << hess[1][1] << endl;
		/*
		double kap2 = 2*kappa(x,y);
		double laplacian = hess[0][0] + hess[1][1];
		if (abs(laplacian-kap2) > 1e-3*abs(kap2)) {
			cout << "Check Laplacian: " << laplacian << " " << kap2 << " (x=" << x << "," << y << ")" << " errorfrac=" << (abs(laplacian-kap2)/kap2) << endl;
		}
		*/
		//cout << "j0=" << jint0 << " j1=" << jint1 << " j2=" << jint2 << endl;
		//cout << "HESS: " << hess[0][0] << " " << hess[1][1] << " " << hess[0][1] << endl;
	}
	hess[1][0] = hess[0][1];

}

void LensProfile::deflection_and_hessian_numerical(const double x, const double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	// You should make it save the kappa and kappa' values evaluated during J0 and K0 so it doesn't have to evaluate them again for
	// J1, K1 and K2 (unless higher order quadrature is required for convergence, in which case extra evaluations must be done). This
	// will save a significant amount of time, but might take some doing to implement

	bool converged;
	LensIntegral<double> lens_integral(this,x,y,lensparams->q);
	if (!ellipticity_gradient) {
		//cout << "NOT DOING EGRAD" << endl;
		double jint0, jint1;
		jint0 = lens_integral.j_integral(0,converged);
		warn_if_not_converged(converged,x,y);
		jint1 = lens_integral.j_integral(1,converged);
		warn_if_not_converged(converged,x,y);
		def[0] = x*jint0;
		def[1] = y*jint1;
		hess[0][0] = 2*x*x*lens_integral.k_integral(0,converged) + jint0;
		warn_if_not_converged(converged,x,y);
		hess[1][1] = 2*y*y*lens_integral.k_integral(2,converged) + jint1;
		warn_if_not_converged(converged,x,y);
		hess[0][1] = 2*x*y*lens_integral.k_integral(1,converged);
		warn_if_not_converged(converged,x,y);
		//double hess00 = lens_integral.k_integral(0,converged);
		//double hess01 = lens_integral.k_integral(1,converged);
		//double hess11 = lens_integral.k_integral(2,converged);
		//cout << "CHECK: hess00=" << (hess00*x*x) << " hess01=" << (hess01*x*y) << " hess11=" << (hess11*y*y) << endl;
		/*
		double kap2 = 2*kappa_rsq((x*x + y*y/(lensparams->q*lensparams->q))/(f_major_axis*f_major_axis));
		double laplacian = hess[0][0] + hess[1][1];
		if (abs(laplacian-kap2) > 1e-3*abs(kap2)) cout << "Check Laplacian (e=const): " << laplacian << " " << kap2 << " (x=" << x << "," << y << ")" << " errorfrac=" << (abs(laplacian-kap2)/kap2) << endl;
		double oldtol = cc_tolerance;
		set_integral_tolerance(1e-6);
		double jint0check = lens_integral.j_integral(0,converged);
		warn_if_not_converged(converged,x,y);
		double jint1check = lens_integral.j_integral(1,converged);
		warn_if_not_converged(converged,x,y);
		if (abs((jint0-jint0check)/jint0check) > 1e-3)
		cout << "j0: " << jint0 << " " << jint0check << " " << (abs((jint0-jint0check)/jint0check)) << " (x=" << x << "," << y << ")" << endl;
		if (abs((jint1-jint1check)/jint1check) > 1e-3)
		cout << "j1: " << jint1 << " " << jint1check << " " << (abs((jint1-jint1check)/jint1check)) << " (x=" << x << "," << y << ")" << endl;
		set_integral_tolerance(oldtol);
		*/



	} else {
		//cout << "DOING EGRAD" << endl;
		/*
		double jint0, jint1, jint2;
		jint0 = lens_integral.j_integral_egrad(0,converged);
		warn_if_not_converged(converged,x,y);
		jint1 = lens_integral.j_integral_egrad(1,converged);
		warn_if_not_converged(converged,x,y);
		jint2 = lens_integral.j_integral_egrad(2,converged);
		warn_if_not_converged(converged,x,y);
		def[0] = x*jint0 + y*jint1;
		def[1] = x*jint1 + y*jint2;
		hess[0][0] = 2*lens_integral.k_integral_egrad(0,converged) + jint0;
		warn_if_not_converged(converged,x,y);
		hess[0][1] = 2*lens_integral.k_integral_egrad(1,converged) + jint1;
		warn_if_not_converged(converged,x,y);
		hess[1][1] = 2*lens_integral.k_integral_egrad(2,converged) + jint2;
		warn_if_not_converged(converged,x,y);
*/

		LensIntegral<double> lens_integral3(this,x,y,lensparams->q,3);
		double jint[3], kint[3];
		lens_integral3.j_integral_egrad_mult(jint,converged);
		warn_if_not_converged(converged,x,y);
		def[0] = x*jint[0] + y*jint[1];
		def[1] = x*jint[1] + y*jint[2];
		lens_integral3.k_integral_egrad_mult(kint,converged);
		warn_if_not_converged(converged,x,y);
		hess[0][0] = 2*kint[0] + jint[0];
		hess[0][1] = 2*kint[1] + jint[1];
		hess[1][1] = 2*kint[2] + jint[2];

		//double hess00 = lens_integral.k_integral_egrad(0,converged);
		//double hess01 = lens_integral.k_integral_egrad(1,converged);
		//double hess11 = lens_integral.k_integral_egrad(2,converged);
		//cout << "hess00=" << hess00 << " hess01=" << hess01 << " hess11=" << hess11 << endl;
		/*
		double kap2 = 2*kappa(x,y);
		double laplacian = hess[0][0] + hess[1][1];
		if ((abs(laplacian-kap2) > 1e-3*abs(kap2))) {
			cout << "Check Laplacian: " << laplacian << " " << kap2 << " (x=" << x << "," << y << ")" << " errorfrac=" << (abs(laplacian-kap2)/kap2) << endl;
		}
		*/
	}
	hess[1][0] = hess[0][1];
}

void LensProfile::warn_if_not_converged(const bool& converged, const double &x, const double &y)
{
	if ((!converged) and (integration_warnings)) {
		if ((integral_method==Gauss_Patterson_Quadrature) or (integral_method==Fejer_Quadrature)) {
			if (qlens->mpi_id==0) {
				if (integral_method==Gauss_Patterson_Quadrature) {
					cout << "*WARNING*: Gauss-Patterson did not converge (x=" << x << ",y=" << y << ")";
					if (GaussQuad::numberOfPoints >= 511) cout << "; switched to Gauss-Legendre quadrature              " << endl;
					else cout << endl;	
				} else if (integral_method==Fejer_Quadrature) {
					cout << "*WARNING*: Fejer quadrature did not converge (x=" << x << ",y=" << y << ")" << endl;
				}
				else cout << endl;
				cout << "Lens: " << model_name << ", Params: ";
				for (int i=0; i < n_params; i++) {
					cout << paramnames[i] << "=";
					if (angle_param[i]) cout << radians_to_degrees(*(lensparams->param[i])) << " degrees";
					else cout << *(lensparams->param[i]);
					if (i != n_params-1) cout << ", ";
				}
				cout << "     " << endl;
				if (qlens->use_ansi_characters) {
					cout << "\033[2A";
				}
			}
		}
	}
}

double LensProfile::potential_numerical(const double x, const double y)
{
	if ((!ellipticity_gradient) and (this->kapavgptr_rsq_spherical==NULL)) return 0.0; // for the integral without egrad, cannot calculate potential without a spherical deflection defined
	bool converged;
	double ans;
	LensIntegral<double> lens_integral(this,x,y,lensparams->q);
	if (!ellipticity_gradient) {
		ans = 0.5*lens_integral.i_integral(converged);
		warn_if_not_converged(converged,x,y);
	} else {
		ans = 0.5*lens_integral.i_integral_egrad(converged);
		warn_if_not_converged(converged,x,y);
	}
	return ans;
}

bool LensProfile::core_present() { return false; }


/************************************* Fourier perturbation algorithms *************************************/

double LensProfile::kappa_from_fourier_modes(const double x, const double y)
{
	double fourier_factor = 0;
	double rsq = x*x + y*y;
	double phi;
	if (!ellipticity_gradient) {
		// it is assumed here that coordinates have already been rotated so that major axis is along x
		phi = atan(y/x);
		if (x < 0) phi += M_PI;
		else if (y < 0) phi += M_2PI;

		for (int i=0; i < n_fourier_modes; i++) {
			fourier_factor += fourier_mode_cosamp[i]*cos(fourier_mode_mvals[i]*phi) + fourier_mode_sinamp[i]*sin(fourier_mode_mvals[i]*phi);
		}
	} else {
		double phi0 = (this->*egrad_ptr)(sqrt(rsq),geometric_param[1],1);

		double costh, sinth, xp, yp, phi;
		costh = cos(phi0);
		sinth = sin(phi0);
		xp = x*costh + y*sinth;
		yp = -x*sinth + y*costh;

		phi = atan(yp/xp);
		if (xp < 0) phi += M_PI;
		else if (yp < 0) phi += M_2PI;

		double *cosamps;
		double *sinamps;
		if (fourier_gradient) {
			cosamps = new double[n_fourier_modes];
			sinamps = new double[n_fourier_modes];
			fourier_mode_function(sqrt(rsq),cosamps,sinamps); // lensing multipoles depend on r, not xi, so we follow the same restriction here
		} else {
			// No need to create new arrays, just have them point to fourier_mode_cosamp and fourier_mode_sinamp
			cosamps = fourier_mode_cosamp.array();
			sinamps = fourier_mode_sinamp.array();
		}
		for (int i=0; i < n_fourier_modes; i++) {
			fourier_factor += cosamps[i]*cos(fourier_mode_mvals[i]*phi) + sinamps[i]*sin(fourier_mode_mvals[i]*phi);
			//fourier_factor += (cosamps[i]*cos(fourier_mode_mvals[i]*phi0) - sinamps[i]*sin(fourier_mode_mvals[i]*phi0))*cos(fourier_mode_mvals[i]*phi) + (cosamps[i]*sin(fourier_mode_mvals[i]*phi0) + sinamps[i]*cos(fourier_mode_mvals[i]*phi0))*sin(fourier_mode_mvals[i]*phi); // testing since this is form used to get deflection integrals
		}
		if (fourier_gradient) {
			delete[] cosamps;
			delete[] sinamps;
		}
	}
	//NOTE: this doesn't work for emode=3 (can't use kappa_rsq_deriv). extend later?
	return 2*fourier_factor*kappa_rsq_deriv(rsq)*rsq; // this allows it to approximate perturbing the elliptical radius (via first order term in Taylor expansion in (r + dr))
}

void LensProfile::add_deflection_from_fourier_modes(const double x, const double y, lensvector<double>& def)
{
	if (n_fourier_modes==0) return;
	double r = sqrt(x*x+y*y);
	double ileft_cos, iright_cos, ileft_sin, iright_sin, potc, dpotc_dr, pots, dpots_dr, def_r, def_phi, m, rmfac, cosm, sinm;

	bool converged;
	LensIntegral<double> lens_integral(this,x,y);
	if (fourier_gradient) {
		lens_integral.cosamps = new double[n_fourier_modes];
		lens_integral.sinamps = new double[n_fourier_modes];
		//fourier_mode_function(r,lens_integral.cosamps,lens_integral.sinamps); // lensing multipoles depend on r, not xi, so we follow the same restriction here
	} else {
		// No need to create new arrays, just have them point to fourier_mode_cosamp and fourier_mode_sinamp
		lens_integral.cosamps = fourier_mode_cosamp.array();
		lens_integral.sinamps = fourier_mode_sinamp.array();
	}

	if (ellipticity_gradient) {
		lens_integral.phi0 = (this->*egrad_ptr)(r,geometric_param[1],1);
	}

	double phi; // used for Fourier modes
	phi = atan(y/x);
	if (x < 0) phi += M_PI;
	else if (y < 0) phi += M_2PI;

	for (int i=0; i < n_fourier_modes; i++) {
		m = fourier_mode_mvals[i];
		if ((fourier_integrals_splined) and fourier_integral_left_cos_spline[i].in_range(r)) {
			ileft_cos = fourier_integral_left_cos_spline[i].splint(r);
			iright_cos = fourier_integral_right_cos_spline[i].splint(r);
			ileft_sin = fourier_integral_left_sin_spline[i].splint(r);
			iright_sin = fourier_integral_right_sin_spline[i].splint(r);
			//cout << "USED THE SPLINE! x=" << x << " y=" << y << " r=" << r << endl;
		} else {
			lens_integral.calculate_fourier_integrals(m,i,false,r,ileft_sin,iright_sin,converged);
			lens_integral.calculate_fourier_integrals(m,i,true,r,ileft_cos,iright_cos,converged);
			//if (!converged) warn("FUCK x=%g y=%g r=%g ilc=%g irc=%g ils=%g irs=%g",x,y,r,ileft_cos,iright_cos,ileft_sin,iright_sin);
		}

		rmfac = pow(r,m);
		pots = -(ileft_sin/rmfac + iright_sin*rmfac)/m;
		dpots_dr = (ileft_sin/rmfac - iright_sin*rmfac)/r;
		potc = -(ileft_cos/rmfac + iright_cos*rmfac)/m;
		dpotc_dr = (ileft_cos/rmfac - iright_cos*rmfac)/r;
		cosm = cos(m*phi);
		sinm = sin(m*phi);
		def_r = dpotc_dr*cosm + dpots_dr*sinm;
		def_phi = (-potc*sinm + pots*cosm)*m/r;
		def[0] += (x*def_r - y*def_phi)/r;
		def[1] += (y*def_r + x*def_phi)/r;
	}
	if (fourier_gradient) {
		delete[] lens_integral.cosamps;
		delete[] lens_integral.sinamps;
	}
}

void LensProfile::add_hessian_from_fourier_modes(const double x, const double y, lensmatrix<double>& hess)
{
	if (n_fourier_modes==0) return;
	double r = sqrt(x*x+y*y);
	double cosphi, sinphi, cossq, sinsq, sincos;
	cosphi = x/r;
	sinphi = y/r;
	cossq = cosphi*cosphi;
	sinsq = sinphi*sinphi;
	sincos = cosphi*sinphi;

	bool converged;
	LensIntegral<double> lens_integral(this,x,y);
	if (fourier_gradient) {
		lens_integral.cosamps = new double[n_fourier_modes];
		lens_integral.sinamps = new double[n_fourier_modes];
		//fourier_mode_function(r,lens_integral.cosamps,lens_integral.sinamps); // lensing multipoles depend on r, not xi, so we follow the same restriction here
	} else {
		// No need to create new arrays, just have them point to fourier_mode_cosamp and fourier_mode_sinamp
		lens_integral.cosamps = fourier_mode_cosamp.array();
		lens_integral.sinamps = fourier_mode_sinamp.array();
	}

	if (ellipticity_gradient) {
		lens_integral.phi0 = (this->*egrad_ptr)(r,geometric_param[1],1);
	}

	double ileft_cos, iright_cos, ileft_sin, iright_sin, potc, dpotc_dr, pots, dpots_dr, def_r, def_phi, m, rmfac, cosm, sinm;
	double hess_pp, hess_rr, hess_pr, hess_rp, rpterm, offdiag, kapm;

	double phi; // used for Fourier modes
	phi = atan(y/x);
	if (x < 0) phi += M_PI;
	else if (y < 0) phi += M_2PI;

	for (int i=0; i < n_fourier_modes; i++) {
		m = fourier_mode_mvals[i];
		if ((fourier_integrals_splined) and fourier_integral_left_cos_spline[i].in_range(r)) {
			ileft_cos = fourier_integral_left_cos_spline[i].splint(r);
			iright_cos = fourier_integral_right_cos_spline[i].splint(r);
			ileft_sin = fourier_integral_left_sin_spline[i].splint(r);
			iright_sin = fourier_integral_right_sin_spline[i].splint(r);
		} else {
			lens_integral.calculate_fourier_integrals(m,i,false,r,ileft_sin,iright_sin,converged);
			lens_integral.calculate_fourier_integrals(m,i,true,r,ileft_cos,iright_cos,converged);
		}
		kapm = lens_integral.fourier_kappa_m(r,phi,m,i);

		rmfac = pow(r,m);
		pots = -(ileft_sin/rmfac + iright_sin*rmfac)/m;
		dpots_dr = (ileft_sin/rmfac - iright_sin*rmfac)/r;
		potc = -(ileft_cos/rmfac + iright_cos*rmfac)/m;
		dpotc_dr = (ileft_cos/rmfac - iright_cos*rmfac)/r;
		cosm = cos(m*phi);
		sinm = sin(m*phi);
		def_r = dpotc_dr*cosm + dpots_dr*sinm;
		def_phi = (-potc*sinm + pots*cosm)*m/r;
		hess_pp = -SQR(m/r)*(potc*cosm + pots*sinm);
		hess_rr = 2*kapm - def_r/r - hess_pp;
		hess_rp = (m/r)*(-dpotc_dr*sinm + dpots_dr*cosm);
		hess_pr = hess_rp - def_phi/r;
		rpterm = (hess_rp+hess_pr)*sincos;
		//cout << "mode " << m << " VARS: " << m << " " << ileft_cos << " " << ileft_sin << " " << kapm << " " << pots << " " << dpots_dr << " " << potc << " " << dpotc_dr << endl;
		hess[0][0] += hess_rr*cossq + hess_pp*sinsq - rpterm + (def_r*sinphi+def_phi*cosphi)*sinphi/r;
		hess[1][1] += hess_rr*sinsq + hess_pp*cossq + rpterm + (def_r*cosphi-def_phi*sinphi)*cosphi/r;
		//offdiag = (hess_rr - hess_pp)*sincos - hess_rp*sinsq + hess_pr*cossq - def_r*sincos/r + def_phi*sinsq/r;
		offdiag = (hess_rr - hess_pp)*sincos + hess_pr*(cossq-sinsq) - def_r*sincos/r;
		hess[0][1] += offdiag;
		hess[1][0] += offdiag;
	}
	if (fourier_gradient) {
		delete[] lens_integral.cosamps;
		delete[] lens_integral.sinamps;
	}
}

void LensProfile::spline_fourier_mode_integrals(const double rmin, const double rmax)
{
	if (n_fourier_modes==0) return;
	if (!fourier_integrals_splined) {
		fourier_integral_left_cos_spline = new Spline<double>[n_fourier_modes];
		fourier_integral_right_cos_spline = new Spline<double>[n_fourier_modes];
		fourier_integral_left_sin_spline = new Spline<double>[n_fourier_modes];
		fourier_integral_right_sin_spline = new Spline<double>[n_fourier_modes];
	}

	int i,j,m;

	double **ileft_cos = new double*[n_fourier_modes];
	double **iright_cos = new double*[n_fourier_modes];
	double **ileft_sin = new double*[n_fourier_modes];
	double **iright_sin = new double*[n_fourier_modes];
	for (j=0; j < n_fourier_modes; j++) {
		ileft_cos[j] = new double[fourier_spline_npoints];
		iright_cos[j] = new double[fourier_spline_npoints];
		ileft_sin[j] = new double[fourier_spline_npoints];
		iright_sin[j] = new double[fourier_spline_npoints];
	}
	double *rvals = new double[fourier_spline_npoints];
	double r, rstep = (rmax-rmin)/(fourier_spline_npoints-1);
	for (r=rmin, i=0; i < fourier_spline_npoints; i++, r += rstep) {
		rvals[i] = r;
	}


	int nthreads = 1;
#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		nthreads = omp_get_num_threads();
	}
#endif
	LensIntegral<double> *lens_integral = new LensIntegral<double>[nthreads];

	for (i=0; i < nthreads; i++) {
		lens_integral[i].initialize(this);
		if (fourier_gradient) {
			lens_integral[i].cosamps = new double[n_fourier_modes];
			lens_integral[i].sinamps = new double[n_fourier_modes];
		} else {
			// No need to create new arrays, just have them point to fourier_mode_cosamp and fourier_mode_sinamp
			lens_integral[i].cosamps = fourier_mode_cosamp.array();
			lens_integral[i].sinamps = fourier_mode_sinamp.array();
		}
	}

	#pragma omp parallel
	{
		int thread;
		bool converged;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		#pragma omp for private(i,j) schedule(static)
		for (i=0; i < fourier_spline_npoints; i++) {
			if (ellipticity_gradient) {
				lens_integral[thread].phi0 = (this->*egrad_ptr)(rvals[i],geometric_param[1],1);
			}

			for (j=0; j < n_fourier_modes; j++) {
				lens_integral[thread].calculate_fourier_integrals(fourier_mode_mvals[j],j,false,rvals[i],ileft_sin[j][i],iright_sin[j][i],converged);
				lens_integral[thread].calculate_fourier_integrals(fourier_mode_mvals[j],j,true,rvals[i],ileft_cos[j][i],iright_cos[j][i],converged);
			}
		}
	}
	for (j=0; j < n_fourier_modes; j++) {
		fourier_integral_left_cos_spline[j].input(rvals,ileft_cos[j],fourier_spline_npoints);
		fourier_integral_right_cos_spline[j].input(rvals,iright_cos[j],fourier_spline_npoints);
		fourier_integral_left_sin_spline[j].input(rvals,ileft_sin[j],fourier_spline_npoints);
		fourier_integral_right_sin_spline[j].input(rvals,iright_sin[j],fourier_spline_npoints);
	}
	for (j=0; j < n_fourier_modes; j++) {
		delete[] ileft_cos[j];
		delete[] iright_cos[j];
		delete[] ileft_sin[j];
		delete[] iright_sin[j];
	}
	delete[] ileft_cos;
	delete[] iright_cos;
	delete[] ileft_sin;
	delete[] iright_sin;
	delete[] rvals;

	if (fourier_gradient) {
		for (i=0; i < nthreads; i++) {
			delete[] lens_integral[i].cosamps;
			delete[] lens_integral[i].sinamps;
		}
	}
	delete[] lens_integral;
	fourier_integrals_splined = true;
}

/*********************************** Functions for printing lens/parameter information *************************************/


bool LensProfile::output_plates(const int n_plates)
{
	if (!ellipticity_gradient) return false;
	double xi, xistep, eps, theta, qval, kap;
	int i;
	xistep = pow(xi_final_egrad/xi_initial_egrad,1.0/(n_plates-1));

	ofstream plate_out("plates.dat");
	for (i=0, xi=xi_initial_egrad; i < n_plates; i++, xi *= xistep) {
		ellipticity_function(xi,eps,theta);
		qval = sqrt(1-eps);
		kap = kappa_rsq(xi*xi);
		plate_out << "lens tophat " << kap << " " << xi << " " << qval << " " << radians_to_degrees(theta) << " 0 0" << endl;
	}
	return true;
}

void LensProfile::print_parameters()
{
	ios_base::fmtflags current_flags = cout.flags();
	if (current_flags & ios::scientific) cout << resetiosflags(ios::scientific);
	if (ellipticity_mode==3) cout << "pseudo-";
	cout << model_name << "(";
	if (lens_subclass != -1) cout << subclass_label << "=" << lens_subclass << ",";
	cout << "z=" << zlens << "): ";
	for (int i=0; i < n_params-1; i++) {
		cout << paramnames[i] << "=";
		if (angle_param[i]) cout << radians_to_degrees(*(lensparams->param[i])) << " degrees";
		else cout << *(lensparams->param[i]);
		if (i != n_params-2) cout << ", ";
	}
	//if (!lensed_center_coords) cout << "xc=" << x_center << ", yc=" << y_center;
	//else cout << "xc_l=" << xc_prime << ", yc_l=" << yc_prime << " (xc=" << x_center << ",yc=" << y_center << ")";
	if (center_anchored) cout << " (center anchored to lens " << center_anchor_lens->lens_number << ")";
	if (transform_center_coords_to_pixsrc_frame) cout << " (xc=" << lensparams->x_center << ", yc=" << lensparams->y_center << ")";
	if ((ellipticity_mode != default_ellipticity_mode) and (ellipticity_mode != 3) and (ellipticity_mode != -1)) {
		cout << " (";
		if (ellipticity_gradient) cout << "egrad=on, ";
		if (fourier_gradient) cout << "fgrad=on,";
		cout << "emode=" << ellipticity_mode << ")"; // emode=3 is indicated by "pseudo-" name, not here
	} else {
		if ((ellipticity_mode != -1) and (ellipticity_gradient)) {
			if (fourier_gradient) cout << " (egrad,fgrad=on)";
			else cout << " (egrad=on)";
		}
	}
	double aux_param;
	string aux_paramname;
	get_auxiliary_parameter(aux_paramname,aux_param);
	if (aux_paramname != "") cout << " (" << aux_paramname << "=" << aux_param << ")";
	if (use_concentration_prior) cout << " (c(M,z) prior defined)" << endl;
	cout << endl;
	if (current_flags & ios::scientific) cout << setiosflags(ios::scientific);
}

string LensProfile::mkstring_doub(const double db)
{
	stringstream dstr;
	string dstring;
	dstr << db;
	dstr >> dstring;
	return dstring;
}

string LensProfile::mkstring_int(const int i)
{
	stringstream istr;
	string istring;
	istr << i;
	istr >> istring;
	return istring;
}

// This function is used by the Python wrapper
string LensProfile::get_parameters_string()
{
	string paramstring = "";
	if (lens_number != -1) paramstring += mkstring_int(lens_number) + ". ";
	if (ellipticity_mode==3) paramstring += "pseudo-";
	paramstring += model_name + "(";
	if (lens_subclass != -1) paramstring += subclass_label + "=" + mkstring_int(lens_subclass) + ",";
	paramstring += "z=" + mkstring_doub(zlens) + "): ";
	for (int i=0; i < n_params-1; i++) {
		paramstring += paramnames[i] + "=";
		if (angle_param[i]) paramstring += mkstring_doub(radians_to_degrees(*(lensparams->param[i]))) + " degrees";
		else paramstring += mkstring_doub(*(lensparams->param[i]));
		if (i != n_params-2) paramstring += ", ";
	}
	if (center_anchored) paramstring += " (center anchored to lens " + mkstring_doub(center_anchor_lens->lens_number) + ")";
	if ((ellipticity_mode != default_ellipticity_mode) and (ellipticity_mode != 3) and (ellipticity_mode != -1)) {
		if ((lenstype != SHEAR) and (lenstype != PTMASS) and (lenstype != MULTIPOLE) and (lenstype != SHEET) and (lenstype != TABULATED))   // these models are not elliptical so emode is irrelevant
		paramstring += " (ellipticity mode = " + mkstring_doub(ellipticity_mode) + ")"; // emode=3 is indicated by "pseudo-" name, not here
	}
	double aux_param;
	string aux_paramname;
	get_auxiliary_parameter(aux_paramname,aux_param);
	if (aux_paramname != "") paramstring += " (" + aux_paramname + "=" + mkstring_doub(aux_param) + ")";
	return paramstring;
}

void LensProfile::print_vary_parameters()
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
	if (at_least_one_param_anchored) {
		cout << "   anchored parameters: ";
		int j=0;
		for (int i=0; i < n_params; i++) {
			if (anchor_parameter_to_lens[i]) {
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
			else if (anchor_parameter_to_source[i]) {
				if (j > 0) cout << ", ";
				cout << paramnames[i] << " --> (source " << parameter_anchor_source[i]->sb_number << ": ";
				if ((parameter_anchor_ratio[i] != 1.0) or (parameter_anchor_exponent[i] != 1.0)) {
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


