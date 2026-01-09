#include "qlens.h"
#include "params.h"
#include "pixelgrid.h"
#include "profile.h"
#include "sbprofile.h"
#include "mathexpr.h"
#include "vector.h"
#include "matrix.h"
#include "errors.h"
#include "romberg.h"
#include "spline.h"
#include "mcmchdr.h"
#include "hyp_2F1.h"
#include "cosmo.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <sys/stat.h>
using namespace std;

#if __cplusplus >= 201703L // C++17 standard or later
#include <filesystem>
#endif

#ifdef USE_FITS
#include "fitsio.h"
#endif

#ifdef USE_COOLEST
#include "json/json.h"
#include <CCfits/CCfits>
#endif

#ifdef USE_MKL
#include "mkl.h"
#endif

#ifdef USE_MULTINEST
#include "multinest.h"
#endif

#ifdef USE_POLYCHORD
#include "interfaces.hpp"
#endif

#ifdef USE_MLPACK
#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans.hpp>
using namespace mlpack;
using namespace mlpack::util;
#endif

const int QLens::nmax_lens_planes = 100;
const double QLens::default_autogrid_initial_step = 1.0e-3;
const double QLens::default_autogrid_rmin = 1.0e-5;
const double QLens::default_autogrid_rmax = 1.0e5;
const double QLens::default_autogrid_frac = 2.1; // ****** NOTE: it might be better to make this depend on the axis ratio, since for q=1 you may need larger rfrac
const int QLens::max_cc_search_iterations = 8;
double QLens::galsubgrid_radius_fraction; // radius of perturber subgridding in terms of fraction of Einstein radius
double QLens::galsubgrid_min_cellsize_fraction; // minimum cell size for perturber subgridding in terms of fraction of Einstein radius
int QLens::galsubgrid_cc_splittings;
bool QLens::auto_store_cc_points;
const double QLens::perturber_einstein_radius_fraction = 0.2;
const double QLens::default_rmin_frac = 1e-4;
bool QLens::warnings;
bool QLens::newton_warnings; // newton_warnings: when true, displays warnings when Newton's method fails or returns anomalous results
bool QLens::use_scientific_notation;
bool QLens::use_ansi_output_during_fit;
double QLens::rmin_frac;
string QLens::fit_output_filename;

int QLens::nthreads = 0;
lensvector *QLens::defs = NULL, **QLens::defs_subtot = NULL, *QLens::defs_i = NULL, *QLens::xvals_i = NULL;
lensmatrix *QLens::jacs = NULL, *QLens::hesses = NULL, **QLens::hesses_subtot = NULL, *QLens::hesses_i = NULL, *QLens::Amats_i = NULL;
int *QLens::indxs = NULL;

void QLens::allocate_multithreaded_variables(const int& threads, const bool reallocate)
{
	if (xvals_i != NULL) {
		if (!reallocate) return;
		else deallocate_multithreaded_variables();
	}
	nthreads = threads;
	// Note: the grid construction is not being parallelized any more...if you decide to ditch it for good, then get rid of these multithreaded variables and replace by single-thread version
	xvals_i = new lensvector[nthreads];
	defs = new lensvector[nthreads];
	defs_subtot = new lensvector*[nthreads];
	defs_i = new lensvector[nthreads];
	jacs = new lensmatrix[nthreads];
	hesses = new lensmatrix[nthreads];
	hesses_subtot = new lensmatrix*[nthreads];
	Amats_i = new lensmatrix[nthreads];
	hesses_i = new lensmatrix[nthreads];
	for (int i=0; i < nthreads; i++) {
		defs_subtot[i] = new lensvector[nmax_lens_planes];
		hesses_subtot[i] = new lensmatrix[nmax_lens_planes];
	}
}

void QLens::deallocate_multithreaded_variables()
{
	if (xvals_i != NULL) {
		delete[] xvals_i;
		delete[] defs;
		delete[] defs_i;
		delete[] jacs;
		delete[] hesses;
		delete[] hesses_i;
		delete[] Amats_i;
		for (int i=0; i < nthreads; i++) {
			delete[] defs_subtot[i];
			delete[] hesses_subtot[i];
		}
		delete[] defs_subtot;
		delete[] hesses_subtot;

		xvals_i = NULL;
		defs = NULL;
		defs_i = NULL;
		jacs = NULL;
		hesses = NULL;
		hesses_i = NULL;
		Amats_i = NULL;
		defs_subtot = NULL;
		hesses_subtot = NULL;
	}
}

#ifdef USE_MUMPS
DMUMPS_STRUC_C *QLens::mumps_solver;

void QLens::setup_mumps()
{
	mumps_solver = new DMUMPS_STRUC_C;
	mumps_solver->par = 1; // this tells MUMPS that the host machine participates in calculation
}
#endif

void QLens::delete_mumps()
{
#ifdef USE_MUMPS
	delete mumps_solver;
#endif
}

#ifdef USE_MPI
void QLens::set_mpi_params(const int& mpi_id_in, const int& mpi_np_in, const int& mpi_ngroups_in, const int& group_num_in, const int& group_id_in, const int& group_np_in, int* group_leader_in, MPI_Group* group_in, MPI_Comm* comm, MPI_Group* mygroup, MPI_Comm* mycomm)
{
	mpi_id = mpi_id_in;
	mpi_np = mpi_np_in;
	mpi_ngroups = mpi_ngroups_in;
	group_id = group_id_in;
	group_num = group_num_in;
	group_np = group_np_in;
	group_leader = new int[mpi_ngroups];
	for (int i=0; i < mpi_ngroups; i++) group_leader[i] = group_leader_in[i];
	mpi_group = group_in;
	group_comm = comm;
	my_group = mygroup;
	my_comm = mycomm;
#ifdef USE_MUMPS
	setup_mumps();
#endif
}
#endif

void QLens::set_mpi_params(const int& mpi_id_in, const int& mpi_np_in)
{
	// This assumes only one 'group', so all MPI processes will work together for each likelihood evaluation
	mpi_id = mpi_id_in;
	mpi_np = mpi_np_in;
	mpi_ngroups = 1;
	group_id = mpi_id;
	group_num = 0;
	group_np = mpi_np;
	group_leader = NULL;

#ifdef USE_MPI
	MPI_Comm_group(MPI_COMM_WORLD, mpi_group);
	MPI_Comm_create(MPI_COMM_WORLD, *mpi_group, group_comm);
#ifdef USE_MUMPS
	setup_mumps();
#endif
#endif
}

QLens::QLens(Cosmology* cosmo_in) : UCMC(), ModelParams()
{
	lens_parent = NULL; // this is only set if creating from another lens
	random_seed = 10;
	reinitialize_random_grid = true;
	n_ranchisq = 1;
	mpi_id = 0;
	mpi_np = 1;
	group_np = 1;
	group_id = 0;
	group_num = 0;
	mpi_ngroups = 1;
	group_leader = NULL;
#ifdef USE_MPI
	mpi_group = NULL;
#endif

	int threads = 1;
#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		threads = omp_get_num_threads();
	}
#endif

	allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)
	if (cosmo_in != NULL) {
		cosmo = cosmo_in;
		cosmology_allocated_within_qlens = false;
	}
	else {
		cosmo = new Cosmology();
		cosmology_allocated_within_qlens = true;
		cosmo->set_cosmology(0.3,0.04,0.7,2.215); // defaults: omega_matter = 0.3, hubble = 0.7
	}
	cosmo->set_qlens(this);
	lens_redshift = 0.5;
	source_redshift = 2.0;
	ellipticity_gradient = false;
	contours_overlap = false; // required for ellipticity gradient mode to check that contours don't overlap
	contour_overlap_log_penalty_prior = 0;
	user_changed_zsource = false; // keeps track of whether redshift has been manually changed; if so, then don't change it to redshift from data
	auto_zsource_scaling = true; // this automatically sets the reference source redshift (for kappa scaling) equal to the source redshift being used
	reference_source_redshift = 2.0; // this is the source redshift with respect to which the lens models are defined
	reference_zfactors = NULL;
	default_zsrc_beta_factors = NULL;
	lens_redshifts = NULL;
	lens_redshift_idx = NULL;
	zlens_group_size = NULL;
	zlens_group_lens_indx = NULL;
	n_lens_redshifts = 0;
	extended_src_redshifts = NULL;
	n_assigned_masks = 0;
	assigned_mask = NULL;
	extended_src_zfactors = NULL;
	extended_src_beta_factors = NULL;
	n_extended_src_redshifts = 0;
	sbprofile_redshift_idx = NULL;
	sbprofile_imggrid_idx = NULL;
	sbprofile_band_number = NULL;
	pixellated_src_redshift_idx = NULL;
	pixellated_src_band = NULL;

	chisq_it=0;
	raw_chisq = -1e30;
	calculate_bayes_factor = false;
	reference_lnZ = -1e30; // used to calculate Bayes factors when two different models are run
	chisq_diagnostic = false;
	chisq_bestfit = 1e30;
	bestfit_flux = 0;
	display_chisq_status = false;
	chisq_display_frequency = 100; // Number of chi-square evaluations before displaying chi-square on screen
	show_wtime = false;
	terminal = TEXT;
	suppress_plots = false;
	verbal_mode = true;
	n_infiles = 0;
	infile = infile_list;
	quit_after_reading_file = false;
	quit_after_error = false;
	fitmethod = SIMPLEX;
	fit_output_dir = ".";
	auto_fit_output_dir = true; // name the output directory "chains_<label>" unless manually specified otherwise
	simplex_nmax = 10000;
	simplex_nmax_anneal = 1000;
	simplex_temp_initial = 0; // no simulated annealing by default
	simplex_temp_final = 1;
	simplex_cooling_factor = 0.9; // temperature decrement (multiplicative) for annealing schedule
	simplex_minchisq = -1e30;
	simplex_minchisq_anneal = -1e30;
	simplex_show_bestfit = false;
	n_livepts = 1000; // for nested sampling
	multinest_constant_eff_mode = false;
	multinest_target_efficiency = 0.1;
	multimodal_sampling = false;
	polychord_nrepeats = 5;
	mcmc_threads = 1;
	mcmc_tolerance = 1.01; // Gelman-Rubin statistic for T-Walk sampler
	mcmc_logfile = false;
	open_chisq_logfile = false;
	//psf_convolution_mpi = false;
	//use_input_psf_matrix = false;
	n_psf = 0;
	psf_list = NULL;
	//add_psf(); // always have at least one psf by default
	psf_threshold = 0.0; // no truncation by default
	psf_ptsrc_threshold = 0.0; // no truncation by default
	ignore_foreground_in_chisq = false;
	ptimg_nsplit = 5; // for subpixel evaluation of point source PSF
	fft_convolution = false;
	fgmask_padding = 2; // padding the foreground mask to allow for convolution with neighboring pixels even if they're not included directly in the likelihood function
	n_image_prior = false;
	n_image_threshold = 1.5; // ************THIS SHOULD BE SPECIFIED BY THE USER, AND ONLY GETS USED IF n_image_prior IS SET TO 'TRUE'
	srcpixel_nimg_mag_threshold = 0.1; // this is the minimum magnification an image pixel must have to be counted when calculating source pixel n_images
	n_image_prior_sb_frac = 0.25; // ********ALSO SHOULD BE SPECIFIED BY THE USER, AND ONLY GETS USED IF n_image_prior IS SET TO 'TRUE'
	auxiliary_srcgrid_npixels = 60; // used for the sourcegrid for nimg_prior (unless fitting with a cartesian grid, in which case src_npixels is used)
	outside_sb_prior = false;
	outside_sb_prior_noise_frac = -1e30; // surface brightness threshold is given as multiple of data pixel noise (negative by default so it's effectively not used)
	outside_sb_prior_threshold = 0.3; // surface brightness threshold is given as fraction of max surface brightness
	einstein_radius_prior = false;
	einstein_radius_low_threshold = 0;
	einstein_radius_high_threshold = 1000;
	concentration_prior = false;
	include_fgmask_in_inversion = false;
	zero_sb_fgmask_prior = false;
	include_noise_term_in_loglike = false;
	high_sn_frac = 0.5; // fraction of max SB; used to determine optimal source pixel size based on area the high S/N pixels cover when mapped to source plane
	use_custom_prior = false;
	nlens = 0;
	n_sb = 0;
	n_pixellated_src = 0;
	n_pixellated_lens = 0;
	lensgrids = NULL;
	//sbmin = -1e30;
	//sbmax = 1e30;
	radial_grid = true;
	grid_xlength = 20; // default gridsize
	grid_ylength = 20;
	grid_xcenter = 0;
	grid_ycenter = 0;
	rmin_frac = default_rmin_frac;
	plot_ptsize = 1.2;
	plot_pttype = 7;

	fit_output_filename = "fit";
	auto_save_bestfit = false;
	fitmodel = NULL;
#ifdef USE_FITS
	fits_format = true;
#else
	fits_format = false;
#endif
	default_data_pixel_size = -1; // used for setting a pixel scale for FITS images (only if initialized to a positive number)
	n_ptsrc = 0;
	ptsrc_list = NULL;
	n_ptsrc_redshifts = 0;
	ptsrc_zfactors = NULL;
	ptsrc_beta_factors = NULL;
	borrowed_image_data = false;
	point_image_data = NULL;

	source_fit_mode = Point_Source;
	use_ansi_characters = false;
	chisq_tolerance = 1e-4;
	//chisqtol_lumreg = 1e-3;
	lumreg_max_it = 0;
	//lumreg_max_it_final = 20;
	chisq_magnification_threshold = 0;
	chisq_imgsep_threshold = 0;
	chisq_imgplane_substitute_threshold = -1; // if > 0, will evaluate the source plane chi-square and if above the threshold, use instead of image plane chi-square (if imgplane_chisq is on)
	n_repeats = 1;
	calculate_parameter_errors = true;
	imgplane_chisq = false;
	use_magnification_in_chisq = true;
	use_magnification_in_chisq_during_repeats = true;
	include_central_image = true;
	include_imgpos_chisq = false;
	include_flux_chisq = false;
	include_weak_lensing_chisq = false;
	include_parity_in_chisq = false;
	include_time_delay_chisq = false;
	use_analytic_bestfit_src = false;
	include_ptsrc_shift = false;
	n_images_penalty = false;
	analytic_source_flux = true;
	include_imgfluxes_in_inversion = false;
	include_srcflux_in_inversion = false;

	lenslist = new LensList(this);
	srclist = new SourceList(this);
	pixsrclist = new PixSrcList(this);
	ptsrclist = new PtSrcList(this);
	imgdatalist = new ImgDataList(this);
	param_list = new ParamList(this);
	dparam_list = new DerivedParamList(this);
	sim_err_pos = 0.005;
	sim_err_flux = 0.01;
	sim_err_td = 1;
	sim_err_shear = 0.1;

	n_model_bands = 0; // There can be one or more model bands to generate mock data, even if there are no "data" bands (i.e. no image data has been loaded)
	n_data_bands = 0;
	imgdata_list = NULL;
	n_image_pixel_grids = 0;
	image_pixel_grids = NULL;
	srcgrids = NULL;
	cartesian_srcgrids = NULL;
	delaunay_srcgrids = NULL;
	sourcegrid_xmin = -1;
	sourcegrid_xmax = 1;
	sourcegrid_ymin = -1;
	sourcegrid_ymax = 1;
	sourcegrid_limit_xmin = -1e30;
	sourcegrid_limit_xmax = 1e30;
	sourcegrid_limit_ymin = -1e30;
	sourcegrid_limit_ymax = 1e30;
	redo_lensing_calculations_before_inversion = true;
	save_sbweights_during_inversion = false;
	use_saved_sbweights = false;
	saved_sbweights = NULL;
	n_sbweights = 0;
	auto_sourcegrid = true;
	auto_shapelet_scaling = true;
	auto_shapelet_center = true;
	shapelet_scale_mode = 0;
	shapelet_window_scaling = 0.8;
	shapelet_max_scale = 1.0;
	delaunay_mode = 1;
	ray_tracing_method = Interpolate;
	natural_neighbor_interpolation = true; // if false, uses 3-point interpolation
	inversion_method = DENSE;
	use_non_negative_least_squares = false;
	//use_fnnls = false;
	max_nnls_iterations = 1000;
	nnls_tolerance = 1e-6;
	parallel_mumps = false;
	show_mumps_info = false;

	regularization_method = Curvature;
	regparam_ptr = NULL;

	use_lum_weighted_regularization = false;
	use_distance_weighted_regularization = false;
	use_mag_weighted_regularization = false;
	auto_lumreg_center = true;
	lumreg_center_from_ptsource = false;
	lensed_lumreg_center = false;
	lensed_lumreg_rc = false;
	fix_lumreg_sig = false;
	lumreg_sig = 1.0;

	lum_weight_function = 0;
	use_lum_weighted_srcpixel_clustering = false;
	use_dist_weighted_srcpixel_clustering = false;
	get_lumreg_from_sbweights = false;

	optimize_regparam = false;
	//optimize_regparam_lhi = false;
	optimize_regparam_tol = 0.01; // this is the tolerance on log(regparam)
	optimize_regparam_minlog = -3;
	optimize_regparam_maxlog = 5;
	max_regopt_iterations = 20;

	background_pixel_noise = 0;
	simulate_pixel_noise = false;
	sb_threshold = 0;
	noise_threshold = 0; // when optimizing the source pixel grid size, image pixels whose surface brightness < noise_threshold*pixel_noise are ignored
	n_image_pixels_x = 200;
	n_image_pixels_y = 200;
	source_npixels = 0;
	lensgrid_npixels = 0;
	srcgrid_npixels_x = 50;
	srcgrid_npixels_y = 50;
	auto_srcgrid_npixels = true;
	auto_srcgrid_set_pixel_size = false; // this feature is not working at the moment, so keep it off
	Fmatrix = NULL;
	Fmatrix_copy = NULL;
	Fmatrix_index = NULL;
	Fmatrix_nn = 0;
	use_noise_map = false;
	dense_Rmatrix = false;
	find_covmatrix_inverse = true;
	use_covariance_matrix = false;
	penalize_defective_covmatrix = true;
	covmatrix_epsilon = 1e-9;

	n_src_inv = 0;
	Rmatrix = NULL;
	Rmatrix_index = NULL;
	src_npixels_inv = NULL;
	src_npixel_start = NULL;
	covmatrix_stacked = NULL;
	covmatrix_packed = NULL;
	covmatrix_factored = NULL;
	Rmatrix_packed = NULL;
	Rmatrix_log_determinant = NULL;

	Rmatrix_MGE_packed = NULL;
	Rmatrix_MGE_log_determinants = NULL;
	mge_list = NULL;

	Rmatrix_pot = NULL;
	Rmatrix_pot_index = NULL;
	Dvector = NULL;
	image_surface_brightness = NULL;
	image_surface_brightness_supersampled = NULL;
	imgpixel_covinv_vector = NULL;
	point_image_surface_brightness = NULL;
	sbprofile_surface_brightness = NULL;
	amplitude_vector = NULL;
	reg_weight_factor = NULL;
	image_pixel_location_Lmatrix = NULL;
	source_pixel_location_Lmatrix = NULL;
	Lmatrix = NULL;
	Lmatrix_index = NULL;
	inversion_nthreads = 1;
	include_potential_perturbations = false;
	potential_correction_iterations = 1;
	first_order_sb_correction = false;
	adopt_final_sbgrad = false;
	adaptive_subgrid = false;
	base_srcpixel_imgpixel_ratio = 0.8; // for lowest mag source pixel, this sets fraction of image pixel area covered by it (when mapped to image plane)
	exclude_source_pixels_beyond_fit_window = true;
	activate_unmapped_source_pixels = true;
	delaunay_high_sn_mode = false;
	delaunay_high_sn_sbfrac = 2.0;
	use_srcpixel_clustering = false;
	include_two_pixsrc_in_Lmatrix = false;

	clustering_random_initialization = false;
	weight_initial_centroids = false;
	use_random_delaunay_srcgrid = false;
	use_dualtree_kmeans = true;
	use_f_src_clusters = true;
	f_src_clusters = 0.5;
	n_src_clusters = -1;
	n_cluster_iterations = 20;
	regrid_if_unmapped_source_subpixels = false;
	default_imgpixel_nsplit = 2;
	emask_imgpixel_nsplit = 1;
	split_imgpixels = true;
	split_high_mag_imgpixels = false;
	delaunay_from_pixel_centers = false;
	raytrace_using_pixel_centers = false;
	psf_supersampling = false;
	imgpixel_lomag_threshold = 0;
	imgpixel_himag_threshold = 0;
	imgpixel_sb_threshold = 0.5;

	cc_rmin = default_autogrid_rmin;
	cc_rmax = default_autogrid_rmax;
	cc_thetasteps = 200;
	autogrid_frac = default_autogrid_frac;

	// parameters for the recursive grid
	min_cell_area = 1e-4;
	usplit_initial = 16; // initial number of cell divisions in the r-direction
	wsplit_initial = 24; // initial number of cell divisions in the theta-direction
	splitlevels = 0; // number of times grid squares are recursively split (by default)...setting to zero is best, recursion slows down grid creation & searching
	cc_splitlevels = 2; // number of times grid squares are recursively split when containing a critical curve
	cc_neighbor_splittings = false;
	skip_newtons_method = false;
	use_perturber_flags = false;
	multithread_perturber_deflections = false;
	subgrid_around_perturbers = true;
	subgrid_only_near_data_images = false; // if on, only subgrids around perturber galaxies (during fit) if a data image is within the determined subgridding radius (dangerous if not all images are observed!)
	galsubgrid_radius_fraction = 1.3;
	galsubgrid_min_cellsize_fraction = 0.25;
	galsubgrid_cc_splittings = 1;
	sorted_critical_curves = false;
	n_singular_points = 0;
	auto_store_cc_points = true;
	newton_magnification_threshold = 10000;
	reject_himag_images = true;
	reject_images_found_outside_cell = false;
	redundancy_separation_threshold = 1e-5;

	warnings = true;
	newton_warnings = false;
	set_sci_notation(true);
	use_ansi_output_during_fit = true;
	include_time_delays = false;
	autocenter = true; // this option tells qlens to center the grid on a particular lens (given by primary_lens_number)
	auto_gridsize_from_einstein_radius = true; // this option tells qlens to set the grid size based on the Einstein radius of a particular lens (given by primary_lens_number)
	autogrid_before_grid_creation = false; // this option (if set to true) tells qlens to optimize the grid size & position automatically (using autogrid) when grid is created
	primary_lens_number = 0;
	auto_set_primary_lens = true;
	include_secondary_lens = false; // turn on to use an additional secondary lens to set the grid size (useful if modeling DM halo + BCG)
	secondary_lens_number = 1;
	spline_frac = 1.8;
	tabulate_rmin = 1e-3;
	tabulate_qmin = 0.2;
	tabulate_logr_N = 2000;
	tabulate_phi_N = 200;
	tabulate_q_N = 10;
	grid = NULL;
	default_parameter_mode = 0;
	include_recursive_lensing = true;
	use_mumps_subcomm = true; // this option should probably be removed, but keeping it for now in case a problem with sub_comm turns up
	DerivedParamPtr = static_cast<void (UCMC::*)(double*,double*)> (&QLens::fitmodel_calculate_derived_params);
	setup_parameters(true); // this sets up the generic parameters in the qlens class (not including lens, source or cosmology parameters)
}

QLens::QLens(QLens *lens_in) : UCMC(), ModelParams() // creates lens object with same settings as input lens; does NOT import the lens/source model configurations, however
{
	lens_parent = lens_in;
	verbal_mode = lens_in->verbal_mode;
	random_seed = lens_in->random_seed;
	n_ranchisq = lens_in->n_ranchisq;
	reinitialize_random_grid = lens_in->reinitialize_random_grid;
	if (reinitialize_random_grid) set_random_seed(random_seed);
	else set_random_generator(lens_in);
	chisq_it=0;
	raw_chisq = -1e30;
	calculate_bayes_factor = lens_in->calculate_bayes_factor;
	reference_lnZ = lens_in->reference_lnZ; // used to calculate Bayes factors when two different models are run
	chisq_diagnostic = lens_in->chisq_diagnostic;
	chisq_bestfit = lens_in->chisq_bestfit;
	bestfit_flux = lens_in->bestfit_flux;
	display_chisq_status = lens_in->display_chisq_status;
	chisq_display_frequency = lens_in->chisq_display_frequency; // Number of chi-square evaluations before displaying chi-square on screen
	mpi_id = lens_in->mpi_id;
	mpi_np = lens_in->mpi_np;
	mpi_ngroups = lens_in->mpi_ngroups;
	group_id = lens_in->group_id;
	group_num = lens_in->group_num;
	group_np = lens_in->group_np;
	group_leader = lens_in->group_leader;
	if (lens_in->group_leader==NULL) group_leader = NULL;
	else {
		group_leader = new int[mpi_ngroups];
		for (int i=0; i < mpi_ngroups; i++) group_leader[i] = lens_in->group_leader[i];
	}
#ifdef USE_MPI
	group_comm = lens_in->group_comm;
	mpi_group = lens_in->mpi_group;
	my_comm = lens_in->my_comm;
	my_group = lens_in->my_group;
#endif

	cosmo = new Cosmology();
	cosmology_allocated_within_qlens = true;
	cosmo->copy_cosmo_data(lens_in->cosmo); 
	cosmo->set_qlens(this);

	lens_redshift = lens_in->lens_redshift;
	source_redshift = lens_in->source_redshift;
	ellipticity_gradient = lens_in->ellipticity_gradient;
	contours_overlap = lens_in->contours_overlap; // required for ellipticity gradient mode to check that contours don't overlap
	contour_overlap_log_penalty_prior = lens_in->contour_overlap_log_penalty_prior;
	user_changed_zsource = lens_in->user_changed_zsource; // keeps track of whether redshift has been manually changed; if so, then don't change it to redshift from data
	auto_zsource_scaling = lens_in->auto_zsource_scaling;
	reference_source_redshift = lens_in->reference_source_redshift; // this is the source redshift with respect to which the lens models are defined
	// Dynamically allocated arrays like the ones below should probably just be replaced with container classes, as long as they don't slow
	// down the code significantly (check!)  It would make the code less bug-prone. On the other hand, if no one ever has to look at or mess with the code, then who cares?
	reference_zfactors = NULL; // this is the scaling for lensing quantities if the source redshift is different from the reference value
	default_zsrc_beta_factors = NULL; // this is the scaling for lensing quantities if the source redshift is different from the reference value
	lens_redshifts = NULL;
	n_lens_redshifts = lens_in->n_lens_redshifts;
	lens_redshift_idx = NULL;
	zlens_group_size = NULL;
	zlens_group_lens_indx = NULL;
	extended_src_redshifts = NULL;
	n_assigned_masks = 0;
	assigned_mask = NULL;
	extended_src_zfactors = NULL;
	extended_src_beta_factors = NULL;
	n_extended_src_redshifts = lens_in->n_extended_src_redshifts;
	sbprofile_redshift_idx = NULL;
	sbprofile_imggrid_idx = NULL;
	sbprofile_band_number = NULL;
	pixellated_src_redshift_idx = NULL;
	pixellated_src_band = NULL;

	terminal = lens_in->terminal;
	show_wtime = lens_in->show_wtime;
	fit_output_dir = lens_in->fit_output_dir;
	auto_fit_output_dir = lens_in->auto_fit_output_dir;
	auto_save_bestfit = lens_in->auto_save_bestfit;
	fitmethod = lens_in->fitmethod;
	mcmc_threads = lens_in->mcmc_threads;
	simplex_nmax = lens_in->simplex_nmax;
	simplex_nmax_anneal = lens_in->simplex_nmax_anneal;
	simplex_temp_initial = lens_in->simplex_temp_initial;
	simplex_temp_final = lens_in->simplex_temp_final;
	simplex_cooling_factor = lens_in->simplex_cooling_factor; // temperature decrement (multiplicative) for annealing schedule
	simplex_minchisq = lens_in->simplex_minchisq;
	simplex_minchisq_anneal = lens_in->simplex_minchisq_anneal;
	simplex_show_bestfit = lens_in->simplex_show_bestfit;
	n_livepts = lens_in->n_livepts; // for nested sampling
	multinest_constant_eff_mode = lens_in->multinest_constant_eff_mode;
	multinest_target_efficiency = lens_in->multinest_target_efficiency;
	multimodal_sampling = lens_in->multimodal_sampling;
	polychord_nrepeats = lens_in->polychord_nrepeats;
	mcmc_tolerance = lens_in->mcmc_tolerance; // for T-Walk sampler
	mcmc_logfile = lens_in->mcmc_logfile;
	open_chisq_logfile = lens_in->open_chisq_logfile;
	//psf_convolution_mpi = lens_in->psf_convolution_mpi;
	//use_input_psf_matrix = lens_in->use_input_psf_matrix;
	psf_threshold = lens_in->psf_threshold;
	psf_ptsrc_threshold = lens_in->psf_ptsrc_threshold;
	n_psf = 0;
	psf_list = NULL;
	ignore_foreground_in_chisq = lens_in->ignore_foreground_in_chisq;
	ptimg_nsplit = lens_in->ptimg_nsplit;
	fft_convolution = lens_in->fft_convolution;
	fgmask_padding = lens_in->fgmask_padding; // padding the foreground mask to allow for convolution with neighboring pixels even if they're not included directly in the likelihood function
	n_image_prior = lens_in->n_image_prior;
	n_image_threshold = lens_in->n_image_threshold;
	srcpixel_nimg_mag_threshold = lens_in->srcpixel_nimg_mag_threshold; // this is the minimum magnification an image pixel must have to be counted when calculating source pixel n_images
	n_image_prior_sb_frac = lens_in->n_image_prior_sb_frac;
	auxiliary_srcgrid_npixels = lens_in->auxiliary_srcgrid_npixels;
	outside_sb_prior = lens_in->outside_sb_prior;
	outside_sb_prior_noise_frac = lens_in->outside_sb_prior_noise_frac; // surface brightness threshold is given as multiple of data pixel noise
	outside_sb_prior_threshold = lens_in->outside_sb_prior_threshold; // surface brightness threshold is given as fraction of max surface brightness
	einstein_radius_prior = lens_in->einstein_radius_prior;
	einstein_radius_low_threshold = lens_in->einstein_radius_low_threshold;
	einstein_radius_high_threshold = lens_in->einstein_radius_high_threshold;
	concentration_prior = lens_in->concentration_prior;
	include_fgmask_in_inversion = lens_in->include_fgmask_in_inversion;
	zero_sb_fgmask_prior = lens_in->zero_sb_fgmask_prior;
	include_noise_term_in_loglike = lens_in->include_noise_term_in_loglike;

	high_sn_frac = lens_in->high_sn_frac; // fraction of max SB; used to determine optimal source pixel size based on area the high S/N pixels cover when mapped to source plane
	use_custom_prior = lens_in->use_custom_prior;

	plot_ptsize = lens_in->plot_ptsize;
	plot_pttype = lens_in->plot_pttype;
	linewidth = lens_in->linewidth;
	fontsize = lens_in->fontsize;
	colorbar_min = lens_in->colorbar_min;
	colorbar_max = lens_in->colorbar_max;

	nlens = 0;
	n_sb = 0;
	n_pixellated_src = 0;
	n_pixellated_lens = 0;
	lensgrids = NULL;
	radial_grid = lens_in->radial_grid;
	grid_xlength = lens_in->grid_xlength; // default gridsize
	grid_ylength = lens_in->grid_ylength;
	grid_xcenter = lens_in->grid_xcenter;
	grid_ycenter = lens_in->grid_ycenter;

	LogLikePtr = static_cast<double (UCMC::*)(double *)> (&QLens::fitmodel_loglike_point_source); // unnecessary, but just in case
	source_fit_mode = lens_in->source_fit_mode;
	use_ansi_characters = lens_in->use_ansi_characters;
	chisq_tolerance = lens_in->chisq_tolerance;
	//chisqtol_lumreg = lens_in->chisqtol_lumreg;
	lumreg_max_it = lens_in->lumreg_max_it;
	//lumreg_max_it_final = lens_in->lumreg_max_it_final;
	chisq_magnification_threshold = lens_in->chisq_magnification_threshold;
	chisq_imgsep_threshold = lens_in->chisq_imgsep_threshold;
	chisq_imgplane_substitute_threshold = lens_in->chisq_imgplane_substitute_threshold;
	n_repeats = lens_in->n_repeats;
	calculate_parameter_errors = lens_in->calculate_parameter_errors;
	imgplane_chisq = lens_in->imgplane_chisq;
	use_magnification_in_chisq = lens_in->use_magnification_in_chisq;
	use_magnification_in_chisq_during_repeats = lens_in->use_magnification_in_chisq_during_repeats;
	include_central_image = lens_in->include_central_image;
	include_imgpos_chisq = lens_in->include_imgpos_chisq;
	include_flux_chisq = lens_in->include_flux_chisq;
	include_weak_lensing_chisq = lens_in->include_weak_lensing_chisq;
	include_parity_in_chisq = lens_in->include_parity_in_chisq;
	include_time_delay_chisq = lens_in->include_time_delay_chisq;
	use_analytic_bestfit_src = lens_in->use_analytic_bestfit_src;
	include_ptsrc_shift = lens_in->include_ptsrc_shift;
	n_images_penalty = lens_in->n_images_penalty;
	analytic_source_flux = lens_in->analytic_source_flux;
	include_imgfluxes_in_inversion = lens_in->include_imgfluxes_in_inversion;
	include_srcflux_in_inversion = lens_in->include_srcflux_in_inversion;

	lenslist = new LensList(this);
	srclist = new SourceList(this);
	pixsrclist = new PixSrcList(this);
	ptsrclist = new PtSrcList(this);
	imgdatalist = new ImgDataList(this);
	param_list = new ParamList(*lens_in->param_list,this);
	dparam_list = new DerivedParamList(*lens_in->dparam_list,this);
	sim_err_pos = lens_in->sim_err_pos;
	sim_err_flux = lens_in->sim_err_flux;
	sim_err_td = lens_in->sim_err_td;
	sim_err_shear = lens_in->sim_err_shear;

	fitmodel = NULL;
	fits_format = lens_in->fits_format;
	default_data_pixel_size = lens_in->default_data_pixel_size;
	n_ptsrc = 0;
	ptsrc_list = NULL;
	n_ptsrc_redshifts = lens_in->n_ptsrc_redshifts;
	ptsrc_zfactors = NULL;
	ptsrc_beta_factors = NULL;
	borrowed_image_data = false;
	point_image_data = NULL;
	weak_lensing_data.input(lens_in->weak_lensing_data);

	n_model_bands = 0; // There can be one or more model bands to generate mock data, even if there are no "data" bands (i.e. no image data has been loaded)
	n_data_bands = 0;
	imgdata_list = NULL;
	n_image_pixel_grids = 0;
	image_pixel_grids = NULL;
	srcgrids = NULL;
	cartesian_srcgrids = NULL;
	delaunay_srcgrids = NULL;
	sourcegrid_xmin = lens_in->sourcegrid_xmin;
	sourcegrid_xmax = lens_in->sourcegrid_xmax;
	sourcegrid_ymin = lens_in->sourcegrid_ymin;
	sourcegrid_ymax = lens_in->sourcegrid_ymax;
	sourcegrid_limit_xmin = lens_in->sourcegrid_limit_xmin;
	sourcegrid_limit_xmax = lens_in->sourcegrid_limit_xmax;
	sourcegrid_limit_ymin = lens_in->sourcegrid_limit_ymin;
	sourcegrid_limit_ymax = lens_in->sourcegrid_limit_ymax;
	redo_lensing_calculations_before_inversion = lens_in->redo_lensing_calculations_before_inversion;
	save_sbweights_during_inversion = false;
	use_saved_sbweights = lens_in->use_saved_sbweights;
	n_sbweights = lens_in->n_sbweights;
	if (n_sbweights > 0) {
		saved_sbweights = new double[n_sbweights];
		for (int i=0; i < n_sbweights; i++) saved_sbweights[i] = lens_in->saved_sbweights[i];
	} else saved_sbweights = NULL;
	auto_sourcegrid = lens_in->auto_sourcegrid;
	auto_shapelet_scaling = lens_in->auto_shapelet_scaling;
	auto_shapelet_center = lens_in->auto_shapelet_center;
	shapelet_scale_mode = lens_in->shapelet_scale_mode;
	shapelet_window_scaling = lens_in->shapelet_window_scaling;
	shapelet_max_scale = lens_in->shapelet_max_scale;
	delaunay_mode = lens_in->delaunay_mode;
	natural_neighbor_interpolation = lens_in->natural_neighbor_interpolation;

	regularization_method = lens_in->regularization_method;
	regparam_ptr = NULL;

	use_lum_weighted_regularization = lens_in->use_lum_weighted_regularization;
	use_distance_weighted_regularization = lens_in->use_distance_weighted_regularization;
	use_mag_weighted_regularization = lens_in->use_mag_weighted_regularization;
	auto_lumreg_center = lens_in->auto_lumreg_center;
	lumreg_center_from_ptsource = lens_in->lumreg_center_from_ptsource;
	lensed_lumreg_center = lens_in->lensed_lumreg_center;
	lensed_lumreg_rc = lens_in->lensed_lumreg_rc;
	fix_lumreg_sig = lens_in->fix_lumreg_sig;
	lumreg_sig = lens_in->lumreg_sig;

	lum_weight_function = lens_in->lum_weight_function;
	use_lum_weighted_srcpixel_clustering = lens_in->use_lum_weighted_srcpixel_clustering;
	use_dist_weighted_srcpixel_clustering = lens_in->use_dist_weighted_srcpixel_clustering;
	get_lumreg_from_sbweights = lens_in->get_lumreg_from_sbweights;

	optimize_regparam = lens_in->optimize_regparam;
	//optimize_regparam_lhi = lens_in->optimize_regparam_lhi;
	optimize_regparam_tol = lens_in->optimize_regparam_tol; // this is the tolerance on log(regparam)
	optimize_regparam_minlog = lens_in->optimize_regparam_minlog;
	optimize_regparam_maxlog = lens_in->optimize_regparam_maxlog;
	max_regopt_iterations = lens_in->max_regopt_iterations;

	ray_tracing_method = lens_in->ray_tracing_method;
	inversion_method = lens_in->inversion_method;
	use_non_negative_least_squares = lens_in->use_non_negative_least_squares;
	//use_fnnls = lens_in->use_fnnls;
	max_nnls_iterations = lens_in->max_nnls_iterations;
	nnls_tolerance = lens_in->nnls_tolerance;
	parallel_mumps = lens_in->parallel_mumps;
	show_mumps_info = lens_in->show_mumps_info;

	//psf_width_x = lens_in->psf_width_x;
	//psf_width_y = lens_in->psf_width_y;
	background_pixel_noise = lens_in->background_pixel_noise;
	simulate_pixel_noise = false; // the fit model should never add random noise when generating lensed images
	sb_threshold = lens_in->sb_threshold;
	noise_threshold = lens_in->noise_threshold;
	n_image_pixels_x = lens_in->n_image_pixels_x;
	n_image_pixels_y = lens_in->n_image_pixels_y;
	srcgrid_npixels_x = lens_in->srcgrid_npixels_x;
	srcgrid_npixels_y = lens_in->srcgrid_npixels_y;
	auto_srcgrid_npixels = lens_in->auto_srcgrid_npixels;
	auto_srcgrid_set_pixel_size = lens_in->auto_srcgrid_set_pixel_size;
	source_npixels = 0;
	lensgrid_npixels = 0;

	Dvector = NULL;
	Fmatrix = NULL;
	Fmatrix_copy = NULL;
	Fmatrix_index = NULL;
	Fmatrix_nn = 0;
	use_noise_map = lens_in->use_noise_map;
	dense_Rmatrix = lens_in->dense_Rmatrix;
	find_covmatrix_inverse = lens_in->find_covmatrix_inverse;
	use_covariance_matrix = lens_in->use_covariance_matrix;
	covmatrix_epsilon = lens_in->covmatrix_epsilon;

	n_src_inv = 0;
	Rmatrix = NULL;
	Rmatrix_index = NULL;
	src_npixels_inv = NULL;
	src_npixel_start = NULL;
	covmatrix_stacked = NULL;
	covmatrix_packed = NULL;
	covmatrix_factored = NULL;
	Rmatrix_packed = NULL;
	Rmatrix_log_determinant = NULL;

	Rmatrix_MGE_packed = NULL;
	Rmatrix_MGE_log_determinants = NULL;
	mge_list = NULL;

	penalize_defective_covmatrix = lens_in->penalize_defective_covmatrix;
	Rmatrix = NULL;
	Rmatrix_index = NULL;
	Rmatrix_pot = NULL;
	Rmatrix_pot_index = NULL;
	image_surface_brightness = NULL;
	image_surface_brightness_supersampled = NULL;
	imgpixel_covinv_vector = NULL;
	point_image_surface_brightness = NULL;
	sbprofile_surface_brightness = NULL;
	amplitude_vector = NULL;
	reg_weight_factor = NULL;
	Lmatrix_index = NULL;
	image_pixel_location_Lmatrix = NULL;
	source_pixel_location_Lmatrix = NULL;
	Lmatrix = NULL;
	inversion_nthreads = lens_in->inversion_nthreads;
	include_potential_perturbations = lens_in->include_potential_perturbations;
	potential_correction_iterations = lens_in->potential_correction_iterations;
	first_order_sb_correction = lens_in->first_order_sb_correction;
	adopt_final_sbgrad = lens_in->adopt_final_sbgrad;
	adaptive_subgrid = lens_in->adaptive_subgrid;
	base_srcpixel_imgpixel_ratio = lens_in->base_srcpixel_imgpixel_ratio; // for lowest mag source pixel, this sets fraction of image pixel area covered by it (when mapped to image plane)
	exclude_source_pixels_beyond_fit_window = lens_in->exclude_source_pixels_beyond_fit_window;
	activate_unmapped_source_pixels = lens_in->activate_unmapped_source_pixels;
	delaunay_high_sn_mode = lens_in->delaunay_high_sn_mode;
	delaunay_high_sn_sbfrac = lens_in->delaunay_high_sn_sbfrac;
	use_srcpixel_clustering = lens_in->use_srcpixel_clustering;
	include_two_pixsrc_in_Lmatrix = lens_in->include_two_pixsrc_in_Lmatrix;

	clustering_random_initialization = lens_in->clustering_random_initialization;
	weight_initial_centroids = lens_in->weight_initial_centroids;
	use_random_delaunay_srcgrid = lens_in->use_random_delaunay_srcgrid;
	use_dualtree_kmeans = lens_in->use_dualtree_kmeans;
	use_f_src_clusters = lens_in->use_f_src_clusters;
	f_src_clusters = lens_in->f_src_clusters;
	n_src_clusters = lens_in->n_src_clusters;
	n_cluster_iterations = lens_in->n_cluster_iterations;
	regrid_if_unmapped_source_subpixels = lens_in->regrid_if_unmapped_source_subpixels;
	default_imgpixel_nsplit = lens_in->default_imgpixel_nsplit;
	emask_imgpixel_nsplit = lens_in->emask_imgpixel_nsplit;
	split_imgpixels = lens_in->split_imgpixels;
	split_high_mag_imgpixels = lens_in->split_high_mag_imgpixels;
	delaunay_from_pixel_centers = lens_in->delaunay_from_pixel_centers;
	raytrace_using_pixel_centers = lens_in->raytrace_using_pixel_centers;
	psf_supersampling = lens_in->psf_supersampling;
	imgpixel_lomag_threshold = lens_in->imgpixel_lomag_threshold;
	imgpixel_himag_threshold = lens_in->imgpixel_himag_threshold;
	imgpixel_sb_threshold = lens_in->imgpixel_sb_threshold;

	cc_rmin = lens_in->cc_rmin;
	cc_rmax = lens_in->cc_rmax;
	cc_thetasteps = lens_in->cc_thetasteps;
	autogrid_frac = lens_in->autogrid_frac;

	// parameters for the recursive grid
	min_cell_area = lens_in->min_cell_area;
	usplit_initial = lens_in->usplit_initial; // initial number of cell divisions in the r-direction
	wsplit_initial = lens_in->wsplit_initial; // initial number of cell divisions in the theta-direction
	splitlevels = lens_in->splitlevels; // number of times grid squares are recursively split (by default)...minimum of one splitting is required
	cc_splitlevels = lens_in->cc_splitlevels; // number of times grid squares are recursively split when containing a critical curve
	cc_neighbor_splittings = lens_in->cc_neighbor_splittings;
	skip_newtons_method = lens_in->skip_newtons_method;
	use_perturber_flags = lens_in->use_perturber_flags;
	multithread_perturber_deflections = lens_in->multithread_perturber_deflections;
	subgrid_around_perturbers = lens_in->subgrid_around_perturbers;
	subgrid_only_near_data_images = lens_in->subgrid_only_near_data_images; // if on, only subgrids around perturber galaxies if a data image is within the determined subgridding radius
	galsubgrid_radius_fraction = lens_in->galsubgrid_radius_fraction;
	galsubgrid_min_cellsize_fraction = lens_in->galsubgrid_min_cellsize_fraction;
	galsubgrid_cc_splittings = lens_in->galsubgrid_cc_splittings;
	sorted_critical_curves = false;
	auto_store_cc_points = lens_in->auto_store_cc_points;
	n_singular_points = 0; // the singular points will be recalculated
	newton_magnification_threshold = lens_in->newton_magnification_threshold;
	reject_himag_images = lens_in->reject_himag_images;
	reject_images_found_outside_cell = lens_in->reject_images_found_outside_cell;
	redundancy_separation_threshold = lens_in->redundancy_separation_threshold;

	include_time_delays = lens_in->include_time_delays;
	autocenter = lens_in->autocenter;
	primary_lens_number = lens_in->primary_lens_number;
	auto_set_primary_lens = lens_in->auto_set_primary_lens;
	include_secondary_lens = lens_in->include_secondary_lens; // turn on to use an additional secondary lens to set the grid size (useful if modeling DM halo + BCG)
	secondary_lens_number = lens_in->secondary_lens_number;

	auto_gridsize_from_einstein_radius = lens_in->auto_gridsize_from_einstein_radius;
	autogrid_before_grid_creation = lens_in->autogrid_before_grid_creation; // this option (if set to true) tells qlens to optimize the grid size & position automatically when grid is created
	default_parameter_mode = lens_in->default_parameter_mode;
	spline_frac = lens_in->spline_frac;
	tabulate_rmin = lens_in->tabulate_rmin;
	tabulate_qmin = lens_in->tabulate_qmin;
	tabulate_logr_N = lens_in->tabulate_logr_N;
	tabulate_phi_N = lens_in->tabulate_phi_N;
	tabulate_q_N = lens_in->tabulate_q_N;

	grid = NULL;
	include_recursive_lensing = lens_in->include_recursive_lensing;
	use_mumps_subcomm = lens_in->use_mumps_subcomm;

	setup_parameters(true); // this sets up the generic parameters in the qlens class (not including lens, source or cosmology parameters)
	syserr_pos = lens_in->syserr_pos;
	wl_shear_factor = lens_in->wl_shear_factor;
	copy_param_arrays(lens_in);
	vary_params.input(n_vary_params);
	vary_params[0] = lens_in->vary_params[0];
	vary_params[1] = lens_in->vary_params[1];
}

void QLens::setup_parameters(const bool initial_setup)
{
	// this sets up the generic parameters (besides lens, source, cosmology and psf parameters) that can be varied as model parameters during fitting
	if (initial_setup) {
		// default initial values
		syserr_pos = 0.0;
		wl_shear_factor = 1.0;

		setup_parameter_arrays(2);
	} else {
		// always reset the active parameter flags, since the active ones will be determined below
		// NOTE: if (initial_setup==true), active params are reset in setup_parameter_arrays(..) above
		n_active_params = 0;
		for (int i=0; i < n_params; i++) {
			active_params[i] = false; // default
		}
	}

	int indx = 0;

	if (initial_setup) {
		// systematic error in the position of point images (assumed to be not accounted for in the given position errors)
		param[indx] = &syserr_pos;
		paramnames[indx] = "syserr_pos"; latex_paramnames[indx] = "\\sigma"; latex_param_subscripts[indx] = "sys";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;

	if (initial_setup) {
		param[indx] = &wl_shear_factor;
		paramnames[indx] = "wl_shearfac"; latex_paramnames[indx] = "m"; latex_param_subscripts[indx] = "WL";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;
}

void QLens::create_and_add_lens(LensProfileName name, const int emode, const double zl, const double zs, const double mass_parameter, const double logslope_param, const double scale1, const double scale2, const double eparam, const double theta, const double xc, const double yc, const double special_param1, const double special_param2, const int pmode)
{
	// eparam can be either q (axis ratio) or epsilon (ellipticity) depending on the ellipticity mode
	// if using ellipticity components, (eparam,theta) are actually (e1,e2)
	
	LensProfile* new_lens = NULL;

	int old_emode = LensProfile::default_ellipticity_mode;
	if (emode != -1) LensProfile::default_ellipticity_mode = emode; // set ellipticity mode to user-specified value for this lens

	SPLE_Lens* alphaptr;
	//Shear* shearptr;
	//Truncated_NFW* tnfwptr;
	switch (name) {
		case PTMASS:
			new_lens = new PointMass(zl, zs, mass_parameter, xc, yc, pmode, this->cosmo); break;
		case SHEET:
			new_lens = new MassSheet(zl, zs, mass_parameter, xc, yc, this->cosmo); break;
		case DEFLECTION:
			new_lens = new Deflection(zl, zs, scale1, scale2, this->cosmo); break;
		case sple_LENS:
			//new_lens = new SPLE_Lens(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, this->cosmo); break; // the old way
			
			//alphaptr = new SPLE_Lens();
			//alphaptr->initialize_parameters(mass_parameter, scale1, scale2, eparam, theta, xc, yc);

			new_lens = new SPLE_Lens(zl, zs, mass_parameter, logslope_param, scale1, eparam, theta, xc, yc, pmode, this->cosmo); // an alternative constructor to use; in this->cosmo case you don't need to call initialize_parameters
			break;
		case SHEAR:
			//shearptr = new Shear();
			//shearptr->initialize_parameters(eparam,theta,xc,yc);
			//new_lens = shearptr;
			//break;
			new_lens = new Shear(zl, zs, eparam, theta, xc, yc, this->cosmo); break;
		// Note: the Multipole profile is added using the function add_multipole_lens(..., this->cosmo) because one of the input parameters is an int
		case nfw:
			new_lens = new NFW(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, pmode, this->cosmo); break;
		case TRUNCATED_nfw:
			//tnfwptr = new Truncated_NFW(pmode,special_param1); // this->cosmo doesn't work yet...doesn't load lens redshift
			//cout << "HMM " << mass_parameter << " " << scale1 << " " << scale2 << " " << eparam << " " << theta << " " << xc << " " << yc << endl;
			//tnfwptr->initialize_parameters(mass_parameter, scale1, scale2, eparam, theta, xc, yc);
			//new_lens = tnfwptr;
			//break;
			new_lens = new Truncated_NFW(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, special_param1, pmode, this->cosmo); break;
		case CORED_nfw:
			new_lens = new Cored_NFW(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, pmode, this->cosmo); break;
		case dpie_LENS:
			new_lens = new dPIE_Lens(zl, zs, mass_parameter, scale1, scale2, eparam, theta, xc, yc, pmode, this->cosmo); break;
		case EXPDISK:
			new_lens = new ExpDisk(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, this->cosmo); break;
		case HERNQUIST:
			new_lens = new Hernquist(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, this->cosmo); break;
		case CORECUSP:
			if ((special_param1==-1000) or (special_param2==-1000)) die("special parameters need to be passed to create_and_add_lens(...) function for model CORECUSP");
			new_lens = new CoreCusp(zl, zs, mass_parameter, special_param1, special_param2, scale1, scale2, eparam, theta, xc, yc, pmode, this->cosmo); break;
		case SERSIC_LENS:
			new_lens = new SersicLens(zl, zs, mass_parameter, scale1, logslope_param, eparam, theta, xc, yc, pmode, this->cosmo); break;
		case DOUBLE_SERSIC_LENS:
			new_lens = new DoubleSersicLens(zl, zs, mass_parameter, special_param1, scale1, logslope_param, scale2, special_param2, eparam, theta, xc, yc, pmode, this->cosmo); break;
		case CORED_SERSIC_LENS:
			new_lens = new Cored_SersicLens(zl, zs, mass_parameter, scale1, logslope_param, scale2, eparam, theta, xc, yc, pmode, this->cosmo); break;
		case TOPHAT_LENS:
			new_lens = new TopHatLens(zl, zs, mass_parameter, scale1, eparam, theta, xc, yc, this->cosmo); break;
		case TESTMODEL: // Model for testing purposes
			new_lens = new TestModel(zl, zs, eparam, theta, xc, yc); break;
		default:
			die("Lens type not recognized");
	}
	if (new_lens==NULL) die("new_lens pointer was not set when creating lens");
	if (emode != -1) LensProfile::default_ellipticity_mode = old_emode; // restore ellipticity mode to its default setting
	add_lens(new_lens);
}

bool QLens::spawn_lens_from_source_object(const int src_number, const double zl, const double zs, const int pmode, const bool vary_mass_parameter, const bool include_limits, const double mass_param_lower, const double mass_param_upper)
{
	if ((!SB_Profile::fourier_sb_perturbation) and (sb_list[src_number]->n_fourier_modes > 0)) {
		warn("cannot spawn lens unless 'fourier_sbmode' is turned on");
		return false;
	}
	//if (LensProfile::orient_major_axis_north) {
		//warn("cannot spawn lens unless 'major_axis_along_y' is turned off");
		//return false;
	//}

	if ((SB_Profile::fourier_use_eccentric_anomaly) and (sb_list[src_number]->has_fourier_modes())) warn("spawned lens must use polar angle for Fourier modes; to ensure that angular structure is identical to source model, set 'fourier_ecc_anomaly' off");
	// NOTE: the source object should store its intrinsic redshift, which should be used as the lens redshift here! Implement this soon!
	LensProfile* new_lens;
	bool spawn_lens = true;
	switch (sb_list[src_number]->get_sbtype()) {
		case GAUSSIAN:
			warn("Spawning lens from Gaussian is currently not supported"); spawn_lens = false; break;
		case SERSIC:
			new_lens = new SersicLens((Sersic*) sb_list[src_number], pmode, vary_mass_parameter, include_limits, mass_param_lower, mass_param_upper); break;
		case CORE_SERSIC:
			warn("Spawning lens from Core-Sersic is currently not supported"); spawn_lens = false; break;
		case CORED_SERSIC:
			new_lens = new Cored_SersicLens((Cored_Sersic*) sb_list[src_number], pmode, vary_mass_parameter, include_limits, mass_param_lower, mass_param_upper); break;
		case DOUBLE_SERSIC:
			new_lens = new DoubleSersicLens((DoubleSersic*) sb_list[src_number], pmode, vary_mass_parameter, include_limits, mass_param_lower, mass_param_upper); break;
		case sple:
			new_lens = new SPLE_Lens((SPLE*) sb_list[src_number], pmode, vary_mass_parameter, include_limits, mass_param_lower, mass_param_upper); break;
		case dpie:
			new_lens = new dPIE_Lens((dPIE*) sb_list[src_number], pmode, vary_mass_parameter, include_limits, mass_param_lower, mass_param_upper); break;
		case nfw_SOURCE:
			new_lens = new NFW((NFW_Source*) sb_list[src_number], pmode, vary_mass_parameter, include_limits, mass_param_lower, mass_param_upper); break;
		case SB_MULTIPOLE:
			warn("cannot spawn lens from SB multipole"); spawn_lens = false; break;
		case SHAPELET:
			warn("cannot spawn lens from shapelet"); spawn_lens = false; break;
		case MULTI_GAUSSIAN_EXPANSION:
			warn("cannot spawn lens from MGE"); spawn_lens = false; break;
		case TOPHAT:
			warn("cannot spawn lens from tophat model"); spawn_lens = false; break;
		default:
			die("surface brightness profile type not supported for fitting");
	}
	if (!spawn_lens) return false;
	new_lens->set_zsrc_ref(reference_source_redshift);

	add_lens(new_lens);
	return true;
}

void QLens::create_and_add_lens(const char *splinefile, const int emode, const double zl, const double zs, const double q, const double theta, const double qx, const double f, const double xc, const double yc)
{
	add_new_lens_entry(zl);

	int old_emode = LensProfile::default_ellipticity_mode;
	if (emode != -1) LensProfile::default_ellipticity_mode = emode; // set ellipticity mode to user-specified value for this lens
	if (emode > 3) die("lens emode greater than 3 does not exist");
	lens_list[nlens-1] = new LensProfile(splinefile, zl, zs, q, theta, xc, yc, qx, f, this->cosmo);
	lens_list[nlens-1]->set_qlens_pointer(this);
	if (emode != -1) LensProfile::default_ellipticity_mode = old_emode; // restore ellipticity mode to its default setting

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
}

void QLens::add_shear_lens(const double zl, const double zs, const double shear_p1, const double shear_p2, const double xc, const double yc)
{
	create_and_add_lens(SHEAR,-1,zl,zs,0,0,0,0,shear_p1,shear_p2,xc,yc);
}

void QLens::add_ptmass_lens(const double zl, const double zs, const double mass_parameter, const double xc, const double yc, const int pmode)
{
	create_and_add_lens(PTMASS,-1,zl,zs,mass_parameter,0,0,0,0,0,xc,yc,0,0,pmode);
}

void QLens::add_mass_sheet_lens(const double zl, const double zs, const double mass_parameter, const double xc, const double yc)
{
	create_and_add_lens(SHEET,-1,zl,zs,mass_parameter,0,0,0,0,0,xc,yc);
}

void QLens::add_multipole_lens(const double zl, const double zs, int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool kap, bool sine_term)
{
	add_new_lens_entry(zl);

	lens_list[nlens-1] = new Multipole(zl, zs, a_m, n, m, theta, xc, yc, kap, this->cosmo, sine_term);
	lens_list[nlens-1]->set_qlens_pointer(this);

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
}

void QLens::add_tabulated_lens(const double zl, const double zs, int lnum, const double kscale, const double rscale, const double theta, const double xc, const double yc)
{
	// automatically set gridsize if the appropriate settings are turned on
	if (autogrid_before_grid_creation) autogrid();
	else {
		if (autocenter==true) {
			lens_list[primary_lens_number]->get_center_coords(grid_xcenter,grid_ycenter);
		}
		if (auto_gridsize_from_einstein_radius==true) {
			double re_major, reav;
			re_major = einstein_radius_of_primary_lens(reference_zfactors[lens_redshift_idx[primary_lens_number]],reav);
			if (re_major > 0.0) {
				double rmax = autogrid_frac*re_major;
				grid_xlength = 2*rmax;
				grid_ylength = 2*rmax;
				cc_rmax = rmax;
			}
		}
	}

	add_new_lens_entry(zl);

	lens_list[nlens-1] = new Tabulated_Model(zl, zs, kscale, rscale, theta, xc, yc, lens_list[lnum], tabulate_rmin, dmax(grid_xlength,grid_ylength), tabulate_logr_N, tabulate_phi_N,this->cosmo);
	lens_list[nlens-1]->set_qlens_pointer(this);

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
}

void QLens::add_qtabulated_lens(const double zl, const double zs, int lnum, const double kscale, const double rscale, const double q, const double theta, const double xc, const double yc)
{
	// automatically set gridsize if the appropriate settings are turned on
	if (autogrid_before_grid_creation) autogrid();
	else {
		if (autocenter==true) {
			lens_list[primary_lens_number]->get_center_coords(grid_xcenter,grid_ycenter);
		}
		if (auto_gridsize_from_einstein_radius==true) {
			double re_major, reav;
			re_major = einstein_radius_of_primary_lens(reference_zfactors[lens_redshift_idx[primary_lens_number]],reav);

			if (re_major != 0.0) {
				double rmax = autogrid_frac*re_major;
				grid_xlength = 2*rmax;
				grid_ylength = 2*rmax;
				cc_rmax = rmax;
			}
		}
	}

	add_new_lens_entry(zl);

	lens_list[nlens-1] = new QTabulated_Model(zl, zs, kscale, rscale, q, theta, xc, yc, lens_list[lnum], tabulate_rmin, dmax(grid_xlength,grid_ylength), tabulate_logr_N, tabulate_phi_N, tabulate_qmin, tabulate_q_N, this->cosmo);
	lens_list[nlens-1]->set_qlens_pointer(this);

	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
}

bool QLens::add_tabulated_lens_from_file(const double zl, const double zs, const double kscale, const double rscale, const double theta, const double xc, const double yc, const string tabfileroot)
{
	string tabfilename;
	if (tabfileroot.find(".tab")==string::npos) tabfilename = tabfileroot + ".tab";
	else tabfilename = tabfileroot;
	ifstream tabfile(tabfilename.c_str());
	if (!tabfile.good()) return false;
	if (tabfile.eof()) return false;
	int i, j, k, rN, phiN;
	double dummy;
	string dummyname;
	tabfile >> dummyname;
	tabfile >> rN >> phiN;
	// check that the file length matches the number of fields expected from rN, phiN
	for (i=0; i < rN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < phiN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < rN; i++) {
		for (j=0; j < phiN; j++) {
			for (k=0; k < 7; k++) {
				if (tabfile.eof()) return false;
				tabfile >> dummy;
			}
		}
	}
	tabfile.clear();
	tabfile.seekg(0, ios::beg);

	add_new_lens_entry(zl);

	lens_list[nlens-1] = new Tabulated_Model(zl, zs, kscale, rscale, theta, xc, yc, tabfile, tabfilename, this->cosmo);
	lens_list[nlens-1]->set_qlens_pointer(this);

	for (i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	return true;
}

bool QLens::add_qtabulated_lens_from_file(const double zl, const double zs, const double kscale, const double rscale, const double q, const double theta, const double xc, const double yc, const string tabfileroot)
{
	string tabfilename;
	if (tabfileroot.find(".tab")==string::npos) tabfilename = tabfileroot + ".tab";
	else tabfilename = tabfileroot;
	ifstream tabfile(tabfilename.c_str());
	if (!tabfile.good()) return false;
	if (tabfile.eof()) return false;
	int i, j, k, l, rN, phiN, qN;
	double dummy;
	string dummyname;
	tabfile >> dummyname;
	tabfile >> rN >> phiN >> qN;
	// check that the file length matches the number of fields expected from rN, phiN
	for (i=0; i < rN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < phiN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < qN; i++) {
		if (tabfile.eof()) return false;
		tabfile >> dummy;
	}
	for (i=0; i < rN; i++) {
		for (j=0; j < phiN; j++) {
			for (l=0; l < qN; l++) {
				for (k=0; k < 7; k++) {
					if (tabfile.eof()) return false;
					tabfile >> dummy;
				}
			}
		}
	}
	tabfile.clear();
	tabfile.seekg(0, ios::beg);

	add_new_lens_entry(zl);

	lens_list[nlens-1] = new QTabulated_Model(zl, zs, kscale, rscale, q, theta, xc, yc, tabfile, this->cosmo);
	lens_list[nlens-1]->set_qlens_pointer(this);

	for (i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();
	return true;
}

void QLens::add_lens(LensProfile *new_lens)
{
	new_lens->setup_cosmology(this->cosmo);
	new_lens->set_qlens_pointer(this);

	add_new_lens_entry(new_lens->zlens);

	new_lens->lens_number = nlens-1;
	lens_list[nlens-1] = new_lens;
	register_lens_vary_parameters(nlens-1);

	reset_grid();
	if (auto_zsource_scaling) auto_zsource_scaling = false; // fix zsrc_ref now that a lens has been created, to make sure lens mass scale doesn't change when zsrc is varied
}

void QLens::add_new_lens_entry(const double zl)
{
	LensProfile** newlist = new LensProfile*[nlens+1];
	int* new_lens_redshift_idx = new int[nlens+1];
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			newlist[i] = lens_list[i];
			new_lens_redshift_idx[i] = lens_redshift_idx[i];
		}
		delete[] lens_list;
		delete[] lens_redshift_idx;
	}
	add_new_lens_redshift(zl,nlens,new_lens_redshift_idx);
	lens_redshift_idx = new_lens_redshift_idx;
	lens_list = newlist;
	nlens++;
	lenslist->input_ptr(lens_list,nlens);

	//int j,k;
	//if (n_lens_redshifts > 1) {
		//cout << "Beta matrix:\n";
		//for (j=0; j < n_lens_redshifts-1; j++) {
			//for (k=0; k < j+1; k++) cout << default_zsrc_beta_factors[j][k] << " ";
			//cout << endl;
		//}
		//cout << endl;
	//}
}

void QLens::add_new_lens_redshift(const double zl, const int lens_i, int* zlens_idx)
{
	int i, j, k, znum;
	bool new_redshift = true;
	for (i=0; i < n_lens_redshifts; i++) {
		if (lens_redshifts[i]==zl) { znum = i; new_redshift = false; break; }
	}
	if (new_redshift) {
		znum = n_lens_redshifts;
		double *new_lens_redshifts = new double[n_lens_redshifts+1];
		int *new_zlens_group_size = new int[n_lens_redshifts+1];
		int **new_zlens_group_lens_indx = new int*[n_lens_redshifts+1];
		int *new_zlens_group_lens_indx_col = new int[1];
		double *new_reference_zfactors = new double[n_lens_redshifts+1];
		new_zlens_group_lens_indx_col[0] = lens_i;
		for (i=0; i < n_lens_redshifts; i++) {
			if (zl < lens_redshifts[i]) {
				znum = i;
				break;
			}
		}
		for (i=0; i < znum; i++) {
			new_lens_redshifts[i] = lens_redshifts[i];
			new_zlens_group_lens_indx[i] = zlens_group_lens_indx[i];
			new_zlens_group_size[i] = zlens_group_size[i];
			new_reference_zfactors[i] = reference_zfactors[i];
		}
		new_lens_redshifts[znum] = zl;
		new_zlens_group_lens_indx[znum] = new_zlens_group_lens_indx_col;
		new_zlens_group_size[znum] = 1;
		new_reference_zfactors[znum] = cosmo->kappa_ratio(zl,source_redshift,reference_source_redshift);
		for (i=znum; i < n_lens_redshifts; i++) {
			new_lens_redshifts[i+1] = lens_redshifts[i];
			new_zlens_group_lens_indx[i+1] = zlens_group_lens_indx[i];
			new_zlens_group_size[i+1] = zlens_group_size[i];
			new_reference_zfactors[i+1] = reference_zfactors[i];
		}
		if (n_lens_redshifts > 0) {
			delete[] lens_redshifts;
			delete[] zlens_group_lens_indx;
			delete[] zlens_group_size;
			delete[] reference_zfactors;
		}
		lens_redshifts = new_lens_redshifts;
		zlens_group_lens_indx = new_zlens_group_lens_indx;
		zlens_group_size = new_zlens_group_size;
		reference_zfactors = new_reference_zfactors;

		double **new_default_zsrc_beta_factors;
		if (n_lens_redshifts > 0) {
			// later you can improve on this so it doesn't have to recalculate previous beta matrix elements, but for now I just want to get
			// it up and running quickly
			new_default_zsrc_beta_factors = new double*[n_lens_redshifts];
			for (i=1; i < n_lens_redshifts+1; i++) {
				new_default_zsrc_beta_factors[i-1] = new double[i];
				if (include_recursive_lensing) {
					for (j=0; j < i; j++) new_default_zsrc_beta_factors[i-1][j] = cosmo->calculate_beta_factor(lens_redshifts[j],lens_redshifts[i],source_redshift); // from cosmo->cpp
				} else {
					for (j=0; j < i; j++) new_default_zsrc_beta_factors[i-1][j] = 0;
				}
			}
			if (default_zsrc_beta_factors != NULL) {
				for (i=0; i < n_lens_redshifts-1; i++) {
					delete[] default_zsrc_beta_factors[i];
				}
				delete[] default_zsrc_beta_factors;
			}
			default_zsrc_beta_factors = new_default_zsrc_beta_factors;
		}

		// update extended source redshift z/beta factors
		double **new_extsrc_zfactors;
		double ***new_extsrc_beta_factors;
		if (n_extended_src_redshifts > 0) {
			new_extsrc_zfactors = new double*[n_extended_src_redshifts];
			new_extsrc_beta_factors = new double**[n_extended_src_redshifts];
			for (i=0; i < n_extended_src_redshifts; i++) {
				new_extsrc_zfactors[i] = new double[n_lens_redshifts+1];
				for (j=0; j < znum; j++) {
					new_extsrc_zfactors[i][j] = extended_src_zfactors[i][j];
				}
				new_extsrc_zfactors[i][znum] = cosmo->kappa_ratio(zl,extended_src_redshifts[i],reference_source_redshift);
				for (j=znum; j < n_lens_redshifts; j++) {
					new_extsrc_zfactors[i][j+1] = extended_src_zfactors[i][j];
				}

				if (n_lens_redshifts > 0) {
					new_extsrc_beta_factors[i] = new double*[n_lens_redshifts];
					for (j=1; j < n_lens_redshifts+1; j++) {
						new_extsrc_beta_factors[i][j-1] = new double[j];
						if (include_recursive_lensing) {
							for (k=0; k < j; k++) new_extsrc_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],extended_src_redshifts[i]); // from cosmo->cpp
						} else {
							for (k=0; k < j; k++) new_extsrc_beta_factors[i][j-1][k] = 0;
						}
					}
				} else new_extsrc_beta_factors[i] = NULL;
			}
			if (extended_src_zfactors != NULL) {
				for (i=0; i < n_extended_src_redshifts; i++) delete[] extended_src_zfactors[i];
				delete[] extended_src_zfactors;
			}
			if (extended_src_beta_factors != NULL) {
				for (i=0; i < n_extended_src_redshifts; i++) {
					if (extended_src_beta_factors[i] != NULL) {
						for (j=0; j < n_lens_redshifts-1; j++) {
							delete[] extended_src_beta_factors[i][j];
						}
						if (n_lens_redshifts > 1) delete[] extended_src_beta_factors[i];
					}
				}
				delete[] extended_src_beta_factors;
			}
			extended_src_zfactors = new_extsrc_zfactors;
			extended_src_beta_factors = new_extsrc_beta_factors;
		}

		// update point source redshift z/beta factors
		double **new_ptsrc_zfactors;
		double ***new_ptsrc_beta_factors;
		if (n_ptsrc_redshifts > 0) {
			new_ptsrc_zfactors = new double*[n_ptsrc_redshifts];
			new_ptsrc_beta_factors = new double**[n_ptsrc_redshifts];
			for (i=0; i < n_ptsrc_redshifts; i++) {
				new_ptsrc_zfactors[i] = new double[n_lens_redshifts+1];
				for (j=0; j < znum; j++) {
					new_ptsrc_zfactors[i][j] = ptsrc_zfactors[i][j];
				}
				new_ptsrc_zfactors[i][znum] = cosmo->kappa_ratio(zl,ptsrc_redshifts[i],reference_source_redshift);
				for (j=znum; j < n_lens_redshifts; j++) {
					new_ptsrc_zfactors[i][j+1] = ptsrc_zfactors[i][j];
				}

				if (n_lens_redshifts > 0) {
					new_ptsrc_beta_factors[i] = new double*[n_lens_redshifts];
					for (j=1; j < n_lens_redshifts+1; j++) {
						new_ptsrc_beta_factors[i][j-1] = new double[j];
						if (include_recursive_lensing) {
							for (k=0; k < j; k++) new_ptsrc_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],ptsrc_redshifts[i]); // from cosmo->cpp
						} else {
							for (k=0; k < j; k++) new_ptsrc_beta_factors[i][j-1][k] = 0;
						}
					}
				} else new_ptsrc_beta_factors[i] = NULL;
			}
			if (ptsrc_zfactors != NULL) {
				for (i=0; i < n_ptsrc_redshifts; i++) delete[] ptsrc_zfactors[i];
				delete[] ptsrc_zfactors;
			}
			if (ptsrc_beta_factors != NULL) {
				for (i=0; i < n_ptsrc_redshifts; i++) {
					if (ptsrc_beta_factors[i] != NULL) {
						for (j=0; j < n_lens_redshifts-1; j++) {
							delete[] ptsrc_beta_factors[i][j];
						}
						if (n_lens_redshifts > 1) delete[] ptsrc_beta_factors[i];
					}
				}
				delete[] ptsrc_beta_factors;
			}
			ptsrc_zfactors = new_ptsrc_zfactors;
			ptsrc_beta_factors = new_ptsrc_beta_factors;
		}

		n_lens_redshifts++;
	} else {
		int *new_zlens_group_lens_indx_col = new int[zlens_group_size[znum]+1];
		for (i=0; i < zlens_group_size[znum]; i++) {
			new_zlens_group_lens_indx_col[i] = zlens_group_lens_indx[znum][i];
		}
		new_zlens_group_lens_indx_col[zlens_group_size[znum]] = lens_i;
		delete[] zlens_group_lens_indx[znum];
		zlens_group_lens_indx[znum] = new_zlens_group_lens_indx_col;
		zlens_group_size[znum]++;
	}
	zlens_idx[lens_i] = znum;
	if (new_redshift) {
		// we inserted a new redshift, so higher redshifts get bumped up an index
		for (j=0; j < nlens; j++) {
			if (j==lens_i) continue;
			if (zlens_idx[j] >= zlens_idx[lens_i]) zlens_idx[j]++;
		}
		for (int i=0; i < n_extended_src_redshifts; i++) {
			if ((image_pixel_grids != NULL) and (image_pixel_grids[i])) image_pixel_grids[i]->update_zfactors_and_beta_factors();
		}
	}
}

void QLens::remove_old_lens_redshift(const int znum, const int lens_i, const bool removed_lens)
{
	int i, j, k, nlenses_with_znum=0, idx=-1;
	for (i=0; i < nlens; i++) {
		if (lens_redshift_idx[i]==znum) {
			nlenses_with_znum++;
			idx = i;
			if (nlenses_with_znum > 1) break;
		}
	}
	if (nlenses_with_znum==1) {
		double *new_lens_redshifts = new double[n_lens_redshifts-1];
		int *new_zlens_group_size = new int[n_lens_redshifts-1];
		int **new_zlens_group_lens_indx = new int*[n_lens_redshifts-1];
		for (i=0; i < znum; i++) {
			new_lens_redshifts[i] = lens_redshifts[i];
			new_zlens_group_size[i] = zlens_group_size[i];
			new_zlens_group_lens_indx[i] = zlens_group_lens_indx[i];
		}
		for (i=znum; i < n_lens_redshifts-1; i++) {
			new_lens_redshifts[i] = lens_redshifts[i+1];
			new_zlens_group_size[i] = zlens_group_size[i+1];
			new_zlens_group_lens_indx[i] = zlens_group_lens_indx[i+1];
		}
		if (lens_redshifts != NULL) delete[] lens_redshifts;
		delete[] zlens_group_lens_indx[znum];
		delete[] zlens_group_lens_indx;
		delete[] zlens_group_size;
		zlens_group_lens_indx = new_zlens_group_lens_indx;
		zlens_group_size = new_zlens_group_size;
		lens_redshifts = new_lens_redshifts;
		for (i=0; i < nlens; i++) {
			if (lens_redshift_idx[i] > znum) lens_redshift_idx[i]--;
		}

		double *new_reference_zfactors;
		double **new_default_zsrc_beta_factors;
		if (n_lens_redshifts==1) {
			delete[] reference_zfactors;
			reference_zfactors = NULL;
		} else {
			new_reference_zfactors = new double[n_lens_redshifts-1];
			for (j=0; j < znum; j++) new_reference_zfactors[j] = reference_zfactors[j];
			for (j=znum; j < n_lens_redshifts-1; j++) new_reference_zfactors[j] = reference_zfactors[j+1];
			delete[] reference_zfactors;
			reference_zfactors = new_reference_zfactors;
			if (n_lens_redshifts==2) {
				delete[] default_zsrc_beta_factors[0];
				delete[] default_zsrc_beta_factors;
				default_zsrc_beta_factors = NULL;
			} else {
				new_default_zsrc_beta_factors = new double*[n_lens_redshifts-2];
				for (i=1; i < n_lens_redshifts-1; i++) {
					new_default_zsrc_beta_factors[i-1] = new double[i];
					if (include_recursive_lensing) {
						for (j=0; j < i; j++) new_default_zsrc_beta_factors[i-1][j] = cosmo->calculate_beta_factor(lens_redshifts[j],lens_redshifts[i],source_redshift); // from cosmo->cpp
					} else {
						for (j=0; j < i; j++) new_default_zsrc_beta_factors[i-1][j] = 0;
					}
				}
				if (default_zsrc_beta_factors != NULL) {
					for (i=0; i < n_lens_redshifts-1; i++) {
						delete[] default_zsrc_beta_factors[i];
					}
					delete[] default_zsrc_beta_factors;
				}
				default_zsrc_beta_factors = new_default_zsrc_beta_factors;
			}
		}

		double **new_ptsrc_zfactors;
		if (n_ptsrc_redshifts > 0) {
			if (n_lens_redshifts==1) {
				for (i=0; i < n_ptsrc_redshifts; i++) delete[] ptsrc_zfactors[i];
				delete[] ptsrc_zfactors;
				ptsrc_zfactors = NULL;
			} else {
				new_ptsrc_zfactors = new double*[n_ptsrc_redshifts];
				for (i=0; i < n_ptsrc_redshifts; i++) {
					new_ptsrc_zfactors[i] = new double[n_lens_redshifts-1];
					for (j=0; j < znum; j++) {
						new_ptsrc_zfactors[i][j] = ptsrc_zfactors[i][j];
					}
					for (j=znum; j < n_lens_redshifts-1; j++) {
						new_ptsrc_zfactors[i][j] = ptsrc_zfactors[i][j+1];
					}
				}
				for (i=0; i < n_ptsrc_redshifts; i++) delete[] ptsrc_zfactors[i];
				delete[] ptsrc_zfactors;
				ptsrc_zfactors = new_ptsrc_zfactors;

				double ***new_ptsrc_beta_factors;
				if (n_lens_redshifts==2) {
					for (i=0; i < n_ptsrc_redshifts; i++) {
						delete[] ptsrc_beta_factors[i][0];
						delete[] ptsrc_beta_factors[i];
						ptsrc_beta_factors[i] = NULL;
					}
					default_zsrc_beta_factors = NULL;
				} else {
					new_ptsrc_beta_factors = new double**[n_ptsrc_redshifts];
					for (i=0; i < n_ptsrc_redshifts; i++) {
						new_ptsrc_beta_factors[i] = new double*[n_lens_redshifts-2];
						for (j=1; j < n_lens_redshifts-1; j++) {
							new_ptsrc_beta_factors[i][j-1] = new double[j];
							if (include_recursive_lensing) {
								for (k=0; k < j; k++) new_ptsrc_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],ptsrc_redshifts[i]); // from cosmo->cpp
							} else {
								for (k=0; k < j; k++) new_ptsrc_beta_factors[i][j-1][k] = 0;
							}
						}
					}
					if (ptsrc_beta_factors != NULL) {
						for (i=0; i < n_ptsrc_redshifts; i++) {
							for (j=0; j < n_lens_redshifts-1; j++) {
								delete[] ptsrc_beta_factors[i][j];
							}
							if (n_lens_redshifts > 1) delete[] ptsrc_beta_factors[i];
						}
						delete[] ptsrc_beta_factors;
					}
					ptsrc_beta_factors = new_ptsrc_beta_factors;
				}
			}
		}

		double **new_extended_src_zfactors;
		if (n_extended_src_redshifts > 0) {
			if (n_lens_redshifts==1) {
				for (i=0; i < n_extended_src_redshifts; i++) delete[] extended_src_zfactors[i];
				delete[] extended_src_zfactors;
				extended_src_zfactors = NULL;
			} else {
				new_extended_src_zfactors = new double*[n_extended_src_redshifts];
				for (i=0; i < n_extended_src_redshifts; i++) {
					new_extended_src_zfactors[i] = new double[n_lens_redshifts-1];
					for (j=0; j < znum; j++) {
						new_extended_src_zfactors[i][j] = extended_src_zfactors[i][j];
					}
					for (j=znum; j < n_lens_redshifts-1; j++) {
						new_extended_src_zfactors[i][j] = extended_src_zfactors[i][j+1];
					}
				}
				for (i=0; i < n_extended_src_redshifts; i++) delete[] extended_src_zfactors[i];
				delete[] extended_src_zfactors;
				extended_src_zfactors = new_extended_src_zfactors;

				double ***new_extended_src_beta_factors;
				if (n_lens_redshifts==2) {
					for (i=0; i < n_extended_src_redshifts; i++) {
						delete[] extended_src_beta_factors[i][0];
						delete[] extended_src_beta_factors[i];
						extended_src_beta_factors[i] = NULL;
					}
					default_zsrc_beta_factors = NULL;
				} else {
					new_extended_src_beta_factors = new double**[n_extended_src_redshifts];
					for (i=0; i < n_extended_src_redshifts; i++) {
						new_extended_src_beta_factors[i] = new double*[n_lens_redshifts-2];
						for (j=1; j < n_lens_redshifts-1; j++) {
							new_extended_src_beta_factors[i][j-1] = new double[j];
							if (include_recursive_lensing) {
								for (k=0; k < j; k++) new_extended_src_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],extended_src_redshifts[i]); // from cosmo->cpp
							} else {
								for (k=0; k < j; k++) new_extended_src_beta_factors[i][j-1][k] = 0;
							}
						}
					}
					if (extended_src_beta_factors != NULL) {
						for (i=0; i < n_extended_src_redshifts; i++) {
							for (j=0; j < n_lens_redshifts-1; j++) {
								delete[] extended_src_beta_factors[i][j];
							}
							if (n_lens_redshifts > 1) delete[] extended_src_beta_factors[i];
						}
						delete[] extended_src_beta_factors;
					}
					extended_src_beta_factors = new_extended_src_beta_factors;
				}
			}
		}

		n_lens_redshifts--;
	} else {
		int *new_zlens_group_lens_indx_col = new int[zlens_group_size[znum]-1];
		for (i=0,j=0; i < zlens_group_size[znum]; i++) {
			if (zlens_group_lens_indx[znum][i] != lens_i) {
				new_zlens_group_lens_indx_col[j] = zlens_group_lens_indx[znum][i];
				j++;
			}
		}
		delete[] zlens_group_lens_indx[znum];
		zlens_group_lens_indx[znum] = new_zlens_group_lens_indx_col;
		zlens_group_size[znum]--;
	}
	if (removed_lens) {
		for (i=0; i < n_lens_redshifts; i++) {
			for (j=0; j < zlens_group_size[i]; j++) {
				// move all the lens indices greater than lens_i down by one, since we've removed lens_i
				if (zlens_group_lens_indx[i][j] > lens_i) zlens_group_lens_indx[i][j]--;
			}
		}
	}
}

void QLens::update_lens_redshift_data()
{
	int i;
	for (i=0; i < nlens; i++) {
		if (lens_redshifts[lens_redshift_idx[i]] != lens_list[i]->zlens) {
			double new_zlens = lens_list[i]->zlens;
			remove_old_lens_redshift(lens_redshift_idx[i],i,false); // this will only remove the redshift if there are no other lenses with the old redshift
			add_new_lens_redshift(new_zlens,i,lens_redshift_idx); // this will only add a new redshift if there are no other lenses with new redshift
		}
	}
	for (i=0; i < nlens; i++) {
		if (lens_list[i]->lensed_center_coords) lens_list[i]->set_center_if_lensed_coords(); // for LOS perturbers whose lensed center coordinates are used as parameters (updates true center)
	}
	for (i=0; i < n_sb; i++) {
		if (sb_list[i]->lensed_center_coords) sb_list[i]->set_center_if_lensed_coords(); // for source objects whose lensed center coordinates are used as parameters (updates true center)
	}
}

void QLens::remove_lens(int lensnumber, const bool delete_lens)
{
	int pi, pf;
	get_lens_parameter_numbers(lensnumber,pi,pf);

	if ((lensnumber >= nlens) or (nlens==0)) { warn(warnings,"Specified lens does not exist"); return; }
	LensProfile** newlist = new LensProfile*[nlens-1];
	int* new_lens_redshift_idx;
	if (nlens > 1) new_lens_redshift_idx = new int[nlens-1];
	int i,j;
	for (i=0; i < nlens; i++) {
		if ((i != lensnumber) and (lens_list[i]->center_anchored==true) and (lens_list[i]->get_center_anchor_number()==lensnumber)) lens_list[i]->delete_center_anchor();
		if ((i != lensnumber) and (lens_list[i]->anchor_special_parameter==true) and (lens_list[i]->get_special_parameter_anchor_number()==lensnumber)) lens_list[i]->delete_special_parameter_anchor();
		if (i != lensnumber) lens_list[i]->unanchor_parameter(lens_list[lensnumber]); // this unanchors the lens if any of its parameters are anchored to the lens being deleted
	}
	for (i=0; i < n_sb; i++) {
		// if any source profiles are anchored to the center of this lens (typically to model foreground light), delete the anchor
		if ((sb_list[i]->center_anchored_to_lens==true) and (sb_list[i]->get_center_anchor_number()==lensnumber)) sb_list[i]->delete_center_anchor();
	}
	remove_old_lens_redshift(lens_redshift_idx[lensnumber], lensnumber, true); // removes the lens redshift from the list if no other lenses share that redshift
	for (i=0,j=0; i < nlens; i++) {
		if (i != lensnumber) {
			newlist[j] = lens_list[i];
			new_lens_redshift_idx[j] = lens_redshift_idx[i];
			j++;
		}
	}
	if (delete_lens) delete lens_list[lensnumber];
	delete[] lens_list;
	delete[] lens_redshift_idx;
	nlens--;

	lens_list = newlist;
	lenslist->input_ptr(lens_list,nlens);
	if (nlens > 0) lens_redshift_idx = new_lens_redshift_idx;
	else lens_redshift_idx = NULL;
	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset_grid();

	param_list->remove_params(pi,pf);
	for (int i=dparam_list->n_dparams-1; i >= 0; i--) {
		if (dparam_list->dparams[i]->int_param==lensnumber) { // this is problematic because "int_param" might not be referring to the lens number (some dparams use int_param for pixsrc number, for example). You need to refine the dparams so it says whether it's connected to a lens, pixsrc, ptsrc etc. and then update this code accordingly.
			if (mpi_id==0) cout << "Removing derived param " << i << endl;
			dparam_list->remove_dparam(i);
		}
	}
}

void QLens::clear_lenses()
{
	if (nlens > 0) {
		int pi, pf, pi_min=1000, pf_max=0;
		// since all the lens parameters are blocked together in the param_list list, we just need to find the initial and final parameters to remove
		for (int i=0; i < nlens; i++) {
			get_lens_parameter_numbers(i,pi,pf);
			if (pi < pi_min) pi_min=pi;
			if (pf > pf_max) pf_max=pf;
		}	
		for (int i=0; i < nlens; i++) {
			delete lens_list[i];
		}	
		delete[] lens_list;
		lenslist->clear_ptr();
		param_list->remove_params(pi_min,pf_max);
		nlens = 0;
		if (lens_redshift_idx != NULL) {
			delete[] lens_redshift_idx;
			lens_redshift_idx = NULL;
		}
		if (lens_redshifts != NULL) {
			delete[] lens_redshifts;
			lens_redshifts = NULL;
		}
		if (zlens_group_size != NULL) {
			delete[] zlens_group_size;
			zlens_group_size = NULL;
		}
		if (zlens_group_lens_indx != NULL) {
			for (int i=0; i < n_lens_redshifts; i++) delete[] zlens_group_lens_indx[i];
			delete[] zlens_group_lens_indx;
			zlens_group_lens_indx = NULL;
		}
		if (reference_zfactors != NULL) {
			delete[] reference_zfactors;
			reference_zfactors = NULL;
		}
		if (ptsrc_zfactors != NULL) {
			for (int i=0; i < n_ptsrc_redshifts; i++) delete[] ptsrc_zfactors[i];
			delete[] ptsrc_zfactors;
			ptsrc_zfactors = NULL;
		}
		if (default_zsrc_beta_factors != NULL) {
			for (int i=0; i < n_lens_redshifts-1; i++) delete[] default_zsrc_beta_factors[i];
			delete[] default_zsrc_beta_factors;
			default_zsrc_beta_factors = NULL;
		}
		if (ptsrc_beta_factors != NULL) {
			for (int i=0; i < n_ptsrc_redshifts; i++) {
				for (int j=0; j < n_lens_redshifts-1; j++) delete[] ptsrc_beta_factors[i][j];
				if (n_lens_redshifts > 1) delete[] ptsrc_beta_factors[i];
			}
			delete[] ptsrc_beta_factors;
			ptsrc_beta_factors = NULL;
		}

		n_lens_redshifts = 0;

		reset_grid();
		dparam_list->clear_dparams(); // NOT ALL DPARAMS ARE TIED TO A LENS!!! Should check which dparams are tied to lenses and clear those
	}
}

int QLens::add_new_extended_src_redshift(const double zs, const int src_i, const bool pixellated_src)
{
	int i, j, k, znum;
	bool new_redshift = true;
	if (zs < 0) {
		znum = -1;
		new_redshift = false;
	} else {
		for (i=0; i < n_extended_src_redshifts; i++) {
			if (extended_src_redshifts[i]==zs) { znum = i; new_redshift = false; break; }
		}
	}
	if (new_redshift) {
		znum = n_extended_src_redshifts;
		double *new_extended_src_redshifts = new double[n_extended_src_redshifts+1];
		for (i=0; i < n_extended_src_redshifts; i++) {
			if (zs < extended_src_redshifts[i]) {
				znum = i;
				break;
			}
		}
		for (i=0; i < znum; i++) {
			new_extended_src_redshifts[i] = extended_src_redshifts[i];
		}
		new_extended_src_redshifts[znum] = zs;
		for (i=znum; i < n_extended_src_redshifts; i++) {
			new_extended_src_redshifts[i+1] = extended_src_redshifts[i];
		}
		if (n_extended_src_redshifts > 0) {
			delete[] extended_src_redshifts;
		}
		extended_src_redshifts = new_extended_src_redshifts;

		double **new_zfactors;
		double ***new_beta_factors;
		if (n_lens_redshifts > 0) {
			new_zfactors = new double*[n_extended_src_redshifts+1];
			new_beta_factors = new double**[n_extended_src_redshifts+1];
			for (i=0; i < znum; i++) {
				new_zfactors[i] = new double[n_lens_redshifts];
				for (j=0; j < n_lens_redshifts; j++) {
					new_zfactors[i][j] = extended_src_zfactors[i][j];
				}
			}
			new_zfactors[znum] = new double[n_lens_redshifts];
			for (j=0; j < n_lens_redshifts; j++) {
				new_zfactors[znum][j] = cosmo->kappa_ratio(lens_redshifts[j],zs,reference_source_redshift);
			}
			for (i=znum; i < n_extended_src_redshifts; i++) {
				new_zfactors[i+1] = new double[n_lens_redshifts];
				for (j=0; j < n_lens_redshifts; j++) {
					new_zfactors[i+1][j] = extended_src_zfactors[i][j];
				}
			}
			for (i=0; i < n_extended_src_redshifts+1; i++) {
				new_beta_factors[i] = new double*[n_lens_redshifts-1];
				for (j=1; j < n_lens_redshifts; j++) {
					new_beta_factors[i][j-1] = new double[j];
					if (include_recursive_lensing) {
						// calculating all beta factors again, just to get it working quickly...fix it up later so it doesn't recalculate all of them over again
						for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],extended_src_redshifts[i]); // from cosmo->cpp
					} else {
						for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = 0;
					}
				}
			}
			if (extended_src_zfactors != NULL) {
				for (i=0; i < n_extended_src_redshifts; i++) delete[] extended_src_zfactors[i];
				delete[] extended_src_zfactors;
			}
			if (extended_src_beta_factors != NULL) {
				for (i=0; i < n_extended_src_redshifts; i++) {
					if (extended_src_beta_factors[i] != NULL) {
						for (j=0; j < n_lens_redshifts-1; j++) {
							delete[] extended_src_beta_factors[i][j];
						}
						if (n_lens_redshifts > 1) delete[] extended_src_beta_factors[i];
					}
				}
				delete[] extended_src_beta_factors;
			}
			extended_src_zfactors = new_zfactors;
			extended_src_beta_factors = new_beta_factors;
		}
		n_extended_src_redshifts++;
		//for (i=0; i < n_lens_redshifts; i++) {
			//cout << i << " " << lens_redshifts[i] << " " << reference_zfactors[i] << endl;
		//}
	}
	// Now you need to update sbprofile_redshift_indx and pixellated_src_redshift_indx

	if (src_i >= 0) {
		if (!pixellated_src) sbprofile_redshift_idx[src_i] = znum;
		else pixellated_src_redshift_idx[src_i] = znum;
	}

	if (new_redshift) {
		 //we inserted a new redshift, so higher redshifts get bumped up an index
		for (j=0; j < n_sb; j++) {
			if ((!pixellated_src) and (j==src_i)) continue;
			if (sbprofile_redshift_idx[j] >= znum) sbprofile_redshift_idx[j]++;
		}
		for (j=0; j < n_pixellated_src; j++) {
			if ((pixellated_src) and (j==src_i)) continue;
			if (pixellated_src_redshift_idx[j] >= znum) pixellated_src_redshift_idx[j]++;
		}
		if (n_extended_src_redshifts==1) {
			if (n_image_pixel_grids > 0) die("there shouldn't be any image pixel grids yet!");
			image_pixel_grids = new ImagePixelGrid*[n_model_bands];
			for (i=0; i < n_model_bands; i++) {
				image_pixel_grids[i] = NULL;
			}
			n_image_pixel_grids = n_model_bands;
		} else {
			ImagePixelGrid** new_image_pixel_grid = new ImagePixelGrid*[n_image_pixel_grids+n_model_bands];
			int i_old, istart, istart_old, znum_j, znum_j_old;
			for (j=0; j < n_model_bands; j++) {
				istart = j*n_extended_src_redshifts;
				istart_old = j*(n_extended_src_redshifts-1);
				znum_j = istart+znum;
				znum_j_old = istart_old+znum;
				for (i=istart, i_old=istart_old; i < znum_j; i++, i_old++) new_image_pixel_grid[i] = image_pixel_grids[i_old];
				for (i=znum_j, i_old=znum_j_old; i < istart+n_extended_src_redshifts-1; i++, i_old++) new_image_pixel_grid[i+1] = image_pixel_grids[i_old];
				new_image_pixel_grid[znum_j] = NULL;
			}
			delete[] image_pixel_grids;
			image_pixel_grids = new_image_pixel_grid;
			n_image_pixel_grids = n_model_bands*n_extended_src_redshifts;

			//ImagePixelGrid** new_image_pixel_grid = new ImagePixelGrid*[n_extended_src_redshifts];
			//for (i=0; i < znum; i++) new_image_pixel_grid[i] = image_pixel_grids[i];
			//for (i=znum; i < n_extended_src_redshifts-1; i++) new_image_pixel_grid[i+1] = image_pixel_grids[i];
			//new_image_pixel_grid[znum] = NULL;
			//delete[] image_pixel_grids;
			//image_pixel_grids = new_image_pixel_grid;
		}

		if (n_extended_src_redshifts==1) {
			assigned_mask = new int[n_data_bands];
			for (i=0; i < n_data_bands; i++) {
				assigned_mask[i] = 0;
			}
			n_assigned_masks = n_data_bands;
		} else {
			int *new_assigned_mask = new int[n_image_pixel_grids];
			int i_old, istart, istart_old, znum_j, znum_j_old;
			for (j=0; j < n_data_bands; j++) {
				istart = j*n_extended_src_redshifts;
				istart_old = j*(n_extended_src_redshifts-1);
				znum_j = istart+znum;
				znum_j_old = istart_old+znum;
				for (i=istart, i_old=istart_old; i < znum_j; i++, i_old++) new_assigned_mask[i] = assigned_mask[i_old];
				for (i=znum_j, i_old=znum_j_old; i < istart+n_extended_src_redshifts-1; i++, i_old++) new_assigned_mask[i+1] = assigned_mask[i_old];
				new_assigned_mask[znum_j] = znum;
			}
			delete[] assigned_mask;
			assigned_mask = new_assigned_mask;
			n_assigned_masks = n_data_bands*n_extended_src_redshifts;
		}

		// Now update all the sbprofile_imggrid_idx, since the indices may have shifted due to inserting a new redshift
		for (i=0; i < n_sb; i++) {
			sbprofile_imggrid_idx[i] = sbprofile_band_number[i]*n_extended_src_redshifts + sbprofile_redshift_idx[i];
		}
	}
	return znum;
}

void QLens::remove_old_extended_src_redshift(const int znum, const bool removing_pixellated_src)
{
	int i, j, k, n_pixsrc_with_znum=0, n_sbsrc_with_znum=0;
	bool remove_redshift = false;
	for (i=0; i < n_sb; i++) {
		if (sbprofile_redshift_idx[i]==znum) {
			n_sbsrc_with_znum++;
			if (n_sbsrc_with_znum > 1) break;
		}
	}
	for (i=0; i < n_pixellated_src; i++) {
		if (pixellated_src_redshift_idx[i]==znum) {
			n_pixsrc_with_znum++;
			if (n_pixsrc_with_znum > 1) break;
		}
	}
	if (removing_pixellated_src) {
		if ((n_pixsrc_with_znum <= 1) and (n_sbsrc_with_znum==0)) {
			remove_redshift = true; // the particular source in question might not have been removed yet
			//cout << "removing redshift! znum=" << znum << " npixz=" << n_pixsrc_with_znum << " nsbz=" << n_sbsrc_with_znum << endl;
		//} else {
			//cout << "NOT removing redshift! znum=" << znum << " npixz=" << n_pixsrc_with_znum << " nsbz=" << n_sbsrc_with_znum << endl;
		}
	} else {
		if ((n_sbsrc_with_znum <= 1) and (n_pixsrc_with_znum==0)) remove_redshift = true; // the particular source in question might not have been removed yet
	}

	if (remove_redshift) {
		double *new_extended_src_redshifts = new double[n_extended_src_redshifts-1];
		for (i=0; i < znum; i++) {
			new_extended_src_redshifts[i] = extended_src_redshifts[i];
		}
		for (i=znum; i < n_extended_src_redshifts-1; i++) {
			new_extended_src_redshifts[i] = extended_src_redshifts[i+1];
		}
		if (extended_src_redshifts != NULL) delete[] extended_src_redshifts;
		extended_src_redshifts = new_extended_src_redshifts;
		
		double **new_zfactors;
		if (n_lens_redshifts > 0) {
			if (n_extended_src_redshifts==1) {
				for (i=0; i < n_extended_src_redshifts; i++) delete[] extended_src_zfactors[i];
				delete[] extended_src_zfactors;
				extended_src_zfactors = NULL;
			} else {
				new_zfactors = new double*[n_extended_src_redshifts-1];

				for (i=0; i < znum; i++) {
					new_zfactors[i] = new double[n_lens_redshifts];
					for (j=0; j < n_lens_redshifts; j++) {
						new_zfactors[i][j] = extended_src_zfactors[i][j];
					}
				}
				for (i=znum; i < n_extended_src_redshifts-1; i++) {
					new_zfactors[i] = new double[n_lens_redshifts];
					for (j=0; j < n_lens_redshifts; j++) {
						new_zfactors[i][j] = extended_src_zfactors[i+1][j];
					}
				}
				for (i=0; i < n_extended_src_redshifts; i++) delete[] extended_src_zfactors[i];
				delete[] extended_src_zfactors;
				extended_src_zfactors = new_zfactors;

				double ***new_beta_factors;
				if (n_lens_redshifts > 1) {
					if (n_extended_src_redshifts==1) {
						for (i=0; i < n_lens_redshifts; i++) {
							delete[] extended_src_beta_factors[0][i];
						}
						delete[] extended_src_beta_factors[0];
						delete[] extended_src_beta_factors;
						extended_src_beta_factors = NULL;
					} else {
						new_beta_factors = new double**[n_extended_src_redshifts-1];
						for (i=0; i < n_extended_src_redshifts-1; i++) {
							new_beta_factors[i] = new double*[n_lens_redshifts-1];
							for (j=1; j < n_lens_redshifts; j++) {
								new_beta_factors[i][j-1] = new double[j];
								if (include_recursive_lensing) {
									for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],extended_src_redshifts[i]); // from cosmo->cpp
								} else {
									for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = 0;
								}
							}
						}
						if (extended_src_beta_factors != NULL) {
							for (i=0; i < n_extended_src_redshifts; i++) {
								for (j=0; j < n_lens_redshifts-1; j++) {
									delete[] extended_src_beta_factors[i][j];
								}
								delete[] extended_src_beta_factors[i];
							}
							delete[] extended_src_beta_factors;
						}
						extended_src_beta_factors = new_beta_factors;
					}
				}
			}
		}
		n_extended_src_redshifts--;
		for (i=0; i < n_sb; i++) {
			if (sbprofile_redshift_idx[i] > znum) sbprofile_redshift_idx[i]--;
		}
		for (i=0; i < n_pixellated_src; i++) {
			if (pixellated_src_redshift_idx[i] > znum) pixellated_src_redshift_idx[i]--;
		}
		/*
		if (image_pixel_grids[znum] != NULL) delete image_pixel_grids[znum];
		if (n_extended_src_redshifts==0) {
			delete[] image_pixel_grids;
		} else {
			ImagePixelGrid** new_image_pixel_grids = new ImagePixelGrid*[n_extended_src_redshifts];
			for (i=0; i < znum; i++) new_image_pixel_grids[i] = image_pixel_grids[i];
			for (i=znum; i < n_extended_src_redshifts; i++) {
				new_image_pixel_grids[i] = image_pixel_grids[i+1];
				new_image_pixel_grids[i]->src_redshift_index = i;
				new_image_pixel_grids[i]->update_zfactors_and_beta_factors();
			}
			delete[] image_pixel_grids;
			image_pixel_grids = new_image_pixel_grids;
		}
		*/

		int i_old, istart, istart_old, znum_j, znum_j_old;
		for (j=0; j < n_model_bands; j++) {
			istart_old = j*(n_extended_src_redshifts+1);
			znum_j_old = istart_old+znum;
			if (image_pixel_grids[znum_j_old] != NULL) delete image_pixel_grids[znum_j_old];
		}
		if (n_extended_src_redshifts==0) {
			delete[] image_pixel_grids;
			n_image_pixel_grids = 0;
		} else {
			n_image_pixel_grids = n_model_bands*n_extended_src_redshifts;
			ImagePixelGrid** new_image_pixel_grids = new ImagePixelGrid*[n_image_pixel_grids];
			for (j=0; j < n_model_bands; j++) {
				istart = j*n_extended_src_redshifts;
				istart_old = j*(n_extended_src_redshifts+1);
				znum_j = istart+znum;
				znum_j_old = istart_old+znum;
				for (i=istart, i_old=istart_old; i < znum_j; i++, i_old++) new_image_pixel_grids[i] = image_pixel_grids[i_old];
				for (i=znum_j, i_old=znum_j_old; i < istart+n_extended_src_redshifts; i++, i_old++) {
					new_image_pixel_grids[i] = image_pixel_grids[i_old+1];
					if (new_image_pixel_grids[i] != NULL) {
						new_image_pixel_grids[i]->src_redshift_index--;
						new_image_pixel_grids[i]->update_zfactors_and_beta_factors();
					}
				}
			}
			delete[] image_pixel_grids;
			image_pixel_grids = new_image_pixel_grids;
		}

		if (n_extended_src_redshifts==0) {
			delete[] assigned_mask;
			n_assigned_masks = 0;
		} else {
			n_assigned_masks = n_data_bands*n_extended_src_redshifts;
			int* new_assigned_mask = new int[n_assigned_masks];
			for (j=0; j < n_data_bands; j++) {
				istart = j*n_extended_src_redshifts;
				istart_old = j*(n_extended_src_redshifts+1);
				znum_j = istart+znum;
				znum_j_old = istart_old+znum;
				for (i=istart, i_old=istart_old; i < znum_j; i++, i_old++) new_assigned_mask[i] = assigned_mask[i_old];
				for (i=znum_j, i_old=znum_j_old; i < istart+n_extended_src_redshifts; i++, i_old++) {
					new_assigned_mask[i] = assigned_mask[i_old+1];
				}
			}
			delete[] assigned_mask;
			assigned_mask = new_assigned_mask;
		}

		// Now update all the sbprofile_imggrid_idx, since the indices may have shifted due to removing a redshift
		for (i=0; i < n_sb; i++) {
			sbprofile_imggrid_idx[i] = sbprofile_band_number[i]*n_extended_src_redshifts + sbprofile_redshift_idx[i];
		}
	}
}

int QLens::add_new_ptsrc_redshift(const double zs, const int src_i)
{
	int i, j, k, znum;
	bool new_redshift = true;
	if (zs < 0) {
		znum = -1;
		new_redshift = false;
	} else {
		for (i=0; i < n_ptsrc_redshifts; i++) {
			if (ptsrc_redshifts[i]==zs) { znum = i; new_redshift = false; break; }
		}
	}
	if (new_redshift) {
		znum = n_ptsrc_redshifts;
		double *new_ptsrc_redshifts = new double[n_ptsrc_redshifts+1];
		for (i=0; i < n_ptsrc_redshifts; i++) {
			if (zs < ptsrc_redshifts[i]) {
				znum = i;
				break;
			}
		}
		for (i=0; i < znum; i++) {
			new_ptsrc_redshifts[i] = ptsrc_redshifts[i];
		}
		new_ptsrc_redshifts[znum] = zs;
		for (i=znum; i < n_ptsrc_redshifts; i++) {
			new_ptsrc_redshifts[i+1] = ptsrc_redshifts[i];
		}
		if (n_ptsrc_redshifts > 0) {
			delete[] ptsrc_redshifts;
		}
		ptsrc_redshifts = new_ptsrc_redshifts;

		double **new_zfactors;
		double ***new_beta_factors;
		if (n_lens_redshifts > 0) {
			new_zfactors = new double*[n_ptsrc_redshifts+1];
			new_beta_factors = new double**[n_ptsrc_redshifts+1];
			for (i=0; i < znum; i++) {
				new_zfactors[i] = new double[n_lens_redshifts];
				for (j=0; j < n_lens_redshifts; j++) {
					new_zfactors[i][j] = ptsrc_zfactors[i][j];
				}
			}
			new_zfactors[znum] = new double[n_lens_redshifts];
			for (j=0; j < n_lens_redshifts; j++) {
				new_zfactors[znum][j] = cosmo->kappa_ratio(lens_redshifts[j],zs,reference_source_redshift);
			}
			for (i=znum; i < n_ptsrc_redshifts; i++) {
				new_zfactors[i+1] = new double[n_lens_redshifts];
				for (j=0; j < n_lens_redshifts; j++) {
					new_zfactors[i+1][j] = ptsrc_zfactors[i][j];
				}
			}
			for (i=0; i < n_ptsrc_redshifts+1; i++) {
				new_beta_factors[i] = new double*[n_lens_redshifts-1];
				for (j=1; j < n_lens_redshifts; j++) {
					new_beta_factors[i][j-1] = new double[j];
					if (include_recursive_lensing) {
						// calculating all beta factors again, just to get it working quickly...fix it up later so it doesn't recalculate all of them over again
						for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],ptsrc_redshifts[i]); // from cosmo->cpp
					} else {
						for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = 0;
					}
				}
			}
			if (ptsrc_zfactors != NULL) {
				for (i=0; i < n_ptsrc_redshifts; i++) delete[] ptsrc_zfactors[i];
				delete[] ptsrc_zfactors;
			}
			if (ptsrc_beta_factors != NULL) {
				for (i=0; i < n_ptsrc_redshifts; i++) {
					if (ptsrc_beta_factors[i] != NULL) {
						for (j=0; j < n_lens_redshifts-1; j++) {
							delete[] ptsrc_beta_factors[i][j];
						}
						if (n_lens_redshifts > 1) delete[] ptsrc_beta_factors[i];
					}
				}
				delete[] ptsrc_beta_factors;
			}
			ptsrc_zfactors = new_zfactors;
			ptsrc_beta_factors = new_beta_factors;
		}
		n_ptsrc_redshifts++;
		//for (i=0; i < n_lens_redshifts; i++) {
			//cout << i << " " << lens_redshifts[i] << " " << reference_zfactors[i] << endl;
		//}
	}
	// Now you need to update sbprofile_redshift_indx and ptsrc_redshift_indx

	ptsrc_redshift_idx[src_i] = znum;

	if (new_redshift) {
		 //we inserted a new redshift, so higher redshifts get bumped up an index
		for (j=0; j < n_ptsrc; j++) {
			if (j==src_i) continue;
			if (ptsrc_redshift_idx[j] >= znum) ptsrc_redshift_idx[j]++;
		}
		if (n_ptsrc_redshifts==1) {
			image_pixel_grids = new ImagePixelGrid*[1];
			image_pixel_grids[0] = NULL;
		}
	}
	return znum;
}

void QLens::remove_old_ptsrc_redshift(const int znum)
{
	int i, j, k, n_ptsrc_with_znum=0;
	bool remove_redshift = false;
	for (i=0; i < n_ptsrc; i++) {
		if (ptsrc_redshift_idx[i]==znum) {
			n_ptsrc_with_znum++;
			if (n_ptsrc_with_znum > 1) break;
		}
	}
	if (n_ptsrc_with_znum <= 1) {
		remove_redshift = true; // the particular source in question might not have been removed yet, but if there's only one, that redshift will be removed
	}

	if (remove_redshift) {
		double *new_ptsrc_redshifts = new double[n_ptsrc_redshifts-1];
		for (i=0; i < znum; i++) {
			new_ptsrc_redshifts[i] = ptsrc_redshifts[i];
		}
		for (i=znum; i < n_ptsrc_redshifts-1; i++) {
			new_ptsrc_redshifts[i] = ptsrc_redshifts[i+1];
		}
		if (ptsrc_redshifts != NULL) delete[] ptsrc_redshifts;
		ptsrc_redshifts = new_ptsrc_redshifts;
		
		double **new_zfactors;
		if (n_lens_redshifts > 0) {
			if (n_ptsrc_redshifts==1) {
				for (i=0; i < n_ptsrc_redshifts; i++) delete[] ptsrc_zfactors[i];
				delete[] ptsrc_zfactors;
				ptsrc_zfactors = NULL;
			} else {
				new_zfactors = new double*[n_ptsrc_redshifts-1];

				for (i=0; i < znum; i++) {
					new_zfactors[i] = new double[n_lens_redshifts];
					for (j=0; j < n_lens_redshifts; j++) {
						new_zfactors[i][j] = ptsrc_zfactors[i][j];
					}
				}
				for (i=znum; i < n_ptsrc_redshifts-1; i++) {
					new_zfactors[i] = new double[n_lens_redshifts];
					for (j=0; j < n_lens_redshifts; j++) {
						new_zfactors[i][j] = ptsrc_zfactors[i+1][j];
					}
				}
				for (i=0; i < n_ptsrc_redshifts; i++) delete[] ptsrc_zfactors[i];
				delete[] ptsrc_zfactors;
				ptsrc_zfactors = new_zfactors;

				double ***new_beta_factors;
				if (n_lens_redshifts > 1) {
					if (n_ptsrc_redshifts==1) {
						for (i=0; i < n_lens_redshifts; i++) {
							delete[] ptsrc_beta_factors[0][i];
						}
						delete[] ptsrc_beta_factors[0];
						delete[] ptsrc_beta_factors;
						ptsrc_beta_factors = NULL;
					} else {
						new_beta_factors = new double**[n_ptsrc_redshifts-1];
						for (i=0; i < n_ptsrc_redshifts-1; i++) {
							new_beta_factors[i] = new double*[n_lens_redshifts-1];
							for (j=1; j < n_lens_redshifts; j++) {
								new_beta_factors[i][j-1] = new double[j];
								if (include_recursive_lensing) {
									for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],ptsrc_redshifts[i]); // from cosmo->cpp
								} else {
									for (k=0; k < j; k++) new_beta_factors[i][j-1][k] = 0;
								}
							}
						}
						if (ptsrc_beta_factors != NULL) {
							for (i=0; i < n_ptsrc_redshifts; i++) {
								for (j=0; j < n_lens_redshifts-1; j++) {
									delete[] ptsrc_beta_factors[i][j];
								}
								delete[] ptsrc_beta_factors[i];
							}
							delete[] ptsrc_beta_factors;
						}
						ptsrc_beta_factors = new_beta_factors;
					}
				}
			}
		}
		n_ptsrc_redshifts--;
		for (i=0; i < n_ptsrc; i++) {
			if (ptsrc_redshift_idx[i] > znum) ptsrc_redshift_idx[i]--;
		}
	}
}

void QLens::update_ptsrc_redshift_data()
{
	int i;
	for (i=0; i < n_ptsrc; i++) {
		if (ptsrc_redshifts[ptsrc_redshift_idx[i]] != ptsrc_list[i]->zsrc) {
			double new_zsrc = ptsrc_list[i]->zsrc;
			remove_old_ptsrc_redshift(ptsrc_redshift_idx[i]); // this will only remove the redshift if there are no other point sources with the old redshift
			add_new_ptsrc_redshift(new_zsrc,i); // this will only add a new redshift if there are no other point sources with new redshift
		}
	}
}

bool QLens::assign_mask(const int band, const int znum, const int mask_i)
{
	if (n_extended_src_redshifts == 0) {
		add_new_extended_src_redshift(source_redshift,-1,false);
	}
	if (n_assigned_masks != n_data_bands*n_extended_src_redshifts) die("number of assigned masks is incorrect (%i vs %i)",n_assigned_masks,(n_data_bands*n_extended_src_redshifts));
	if (band >= n_data_bands) { warn("specified data band has not been created"); return false; }
	if (znum >= n_extended_src_redshifts) { warn("source redshift index does not exist"); return false; }
	if (imgdata_list==NULL) { warn("image pixel data has not been loaded"); return false; }
	//int nmasks=0;
	//for (int i=0; i < n_data_bands; i++) nmasks += imgdata_list[band]->n_masks;
	//if (mask_i >= nmasks) { warn("mask index does not exist"); return false; }
	int imggrid_i = band*n_extended_src_redshifts + znum;
	assigned_mask[imggrid_i] = mask_i;
	return true;
}

void QLens::print_mask_assignments()
{
	if (n_extended_src_redshifts==0) cout << "No source redshifts have been created yet" << endl;
	int i,j,imggrid_i=0;
	for (i=0; i < n_data_bands; i++) {
		if (n_data_bands > 1) cout << "Band " << i << ":" << endl;
		for (j=0; j < n_extended_src_redshifts; j++) {
			cout << imggrid_i << ": zsrc=" << extended_src_redshifts[j] << ", mask=" << assigned_mask[imggrid_i] << endl;
			imggrid_i++;
		}
	}
}

void QLens::print_zfactors_and_beta_matrices()
{
	int i,j,k;
	if (n_lens_redshifts > 1) {
		for (i=0; i < nlens; i++) cout << "Lens " << i << " redshift index: " << lens_redshift_idx[i] << endl;
	}
	cout << endl << "reference zfactors: ";
	for (i=0; i < n_lens_redshifts; i++) cout << reference_zfactors[i] << " ";
	cout << endl;
	if (n_lens_redshifts > 1) {
		cout << "zsrc=" << source_redshift << " beta matrix:\n";
		for (j=0; j < n_lens_redshifts-1; j++) {
			cout << "z=" << lens_redshifts[j] << ": ";
			for (k=0; k < j+1; k++) cout << default_zsrc_beta_factors[j][k] << " ";
			cout << endl;
		}
		cout << endl;
	}

	if (n_ptsrc > 0) {
		for (i=0; i < n_ptsrc_redshifts; i++) {
			cout << "ZFACTORS for ptsrc redshift index " << i << " (zs=" << ptsrc_redshifts[i] << "): ";
			for (j=0; j < n_lens_redshifts; j++) cout << ptsrc_zfactors[i][j] << " ";
			cout << endl;
		}
		cout << endl;
		if (n_lens_redshifts > 1) {
			for (i=0; i < n_ptsrc_redshifts; i++) {
				cout << "source " << i << " beta matrix:\n";
				for (j=0; j < n_lens_redshifts-1; j++) {
					for (k=0; k < j+1; k++) cout << ptsrc_beta_factors[i][j][k] << " ";
					cout << endl;
				}
				cout << endl;
			}
		}
	}

	if (n_extended_src_redshifts > 0) {
		cout << "n_extended_src_redshifts = " << n_extended_src_redshifts << endl;	
		for (i=0; i < n_extended_src_redshifts; i++) {
			cout << "ZFACTORS for extended src redshift index " << i << " (zs=" << extended_src_redshifts[i] << "): ";
			for (j=0; j < n_lens_redshifts; j++) cout << extended_src_zfactors[i][j] << " ";
			cout << endl;
		}
		cout << endl;
		if (n_lens_redshifts > 1) {
			for (i=0; i < n_extended_src_redshifts; i++) {
				cout << "source " << i << " beta matrix:\n";
				for (j=0; j < n_lens_redshifts-1; j++) {
					for (k=0; k < j+1; k++) cout << extended_src_beta_factors[i][j][k] << " ";
					cout << endl;
				}
				cout << endl;
			}
		}
	}
}

bool QLens::save_tabulated_lens_to_file(int lnum, const string tabfileroot)
{
	int pos;
	string tabfilename = tabfileroot;
	if ((pos = tabfilename.find(".tab")) != string::npos) {
		tabfilename = tabfilename.substr(0,pos);
	}

	if ((lens_list[lnum]->get_lenstype() != TABULATED) and (lens_list[lnum]->get_lenstype() != QTABULATED)) return false;
	if (lens_list[lnum]->get_lenstype() == TABULATED) {
		Tabulated_Model temp_tablens((Tabulated_Model*) lens_list[lnum]);
		temp_tablens.output_tables(tabfilename);
	} else {
		QTabulated_Model temp_tablens((QTabulated_Model*) lens_list[lnum]);
		temp_tablens.output_tables(tabfilename);
	}
	return true;
}

void QLens::set_source_redshift(const double zsrc)
{
	source_redshift = zsrc;
	int i,j;
	if (auto_zsource_scaling) {
		reference_source_redshift = source_redshift;
		for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = 1.0;
		if (n_ptsrc_redshifts > 1) die("cannot have multiple point source redshifts and zsrc_scaling on--fix so this is forbidden");
		if (n_ptsrc_redshifts==1) {
			for (j=0; j < n_lens_redshifts; j++) {
				ptsrc_zfactors[0][j] = cosmo->kappa_ratio(lens_redshifts[j],ptsrc_redshifts[i],reference_source_redshift);
			}
		}
	} else {
		for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = cosmo->kappa_ratio(lens_redshifts[i],source_redshift,reference_source_redshift);
	}
	recalculate_beta_factors();
	//reset_grid();
}

void QLens::set_reference_source_redshift(const double zsrc)
{
	if (nlens > 0) { warn("zsrc_ref cannot be changed if any lenses have already been created"); return; }
	int i,j;
	reference_source_redshift = zsrc;
	if (auto_zsource_scaling==true) auto_zsource_scaling = false; // Now that zsrc_ref has been set explicitly, don't automatically change it if zsrc is changed
	for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = cosmo->kappa_ratio(lens_redshifts[i],source_redshift,reference_source_redshift);
	reset_grid();
	if (n_extended_src_redshifts > 0) {
		for (i=0; i < n_extended_src_redshifts; i++) {
			for (j=0; j < n_lens_redshifts; j++) {
				extended_src_zfactors[i][j] = cosmo->kappa_ratio(lens_redshifts[j],extended_src_redshifts[i],reference_source_redshift);
			}
		}
	}
	if (n_ptsrc_redshifts > 0) {
		for (i=0; i < n_ptsrc_redshifts; i++) {
			for (j=0; j < n_lens_redshifts; j++) {
				ptsrc_zfactors[i][j] = cosmo->kappa_ratio(lens_redshifts[j],ptsrc_redshifts[i],reference_source_redshift);
			}
		}
	}
}

void QLens::recalculate_beta_factors()
{
	int i,j;
	if (n_lens_redshifts > 1) {
		for (i=1; i < n_lens_redshifts; i++) {
			if (include_recursive_lensing) {
				for (j=0; j < i; j++) default_zsrc_beta_factors[i-1][j] = cosmo->calculate_beta_factor(lens_redshifts[j],lens_redshifts[i],source_redshift);
			} else {
				for (j=0; j < i; j++) default_zsrc_beta_factors[i-1][j] = 0;
			}
		}
	}
}

void QLens::toggle_major_axis_along_y(bool major_axis_along_y)
{
	if (LensProfile::orient_major_axis_north != major_axis_along_y) {
		LensProfile::orient_major_axis_north = major_axis_along_y;
		if (nlens > 0) {
			if (major_axis_along_y) {
				for (int i=0; i < nlens; i++) lens_list[i]->shift_angle_minus_90();
			} else {
				for (int i=0; i < nlens; i++) lens_list[i]->shift_angle_90();
			}
		}
	}
}

void QLens::toggle_major_axis_along_y_src(bool major_axis_along_y)
{
	if (SB_Profile::orient_major_axis_north != major_axis_along_y) {
		SB_Profile::orient_major_axis_north = major_axis_along_y;
		if (n_sb > 0) {
			if (major_axis_along_y) {
				for (int i=0; i < n_sb; i++) sb_list[i]->shift_angle_minus_90();
			} else {
				for (int i=0; i < n_sb; i++) sb_list[i]->shift_angle_90();
			}
		}
	}
}

void QLens::record_singular_points(double *zfacs)
{
	// if kappa goes like r^n near the origin where n <= -1, then a radial critical curve will not form
	// (this is because the deflection, which goes like r^(n+1), must increase as you go outward in order
	// to have a radial critical curve; this will only happen for n>-1). Here we test for this for the
	// relevant models where this can occur
	singular_pts.clear();
	double xc, yc;
	bool singular;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			singular = false;
			if (zfacs[lens_redshift_idx[i]] != 0.0) {
				if ((lens_list[i]->get_lenstype() == dpie_LENS) and (lens_list[i]->core_present()==false)) singular = true;
				else if ((lens_list[i]->get_lenstype() == sple_LENS) and (lens_list[i]->get_inner_logslope() <= -1) and (lens_list[i]->core_present()==false)) singular = true;
				else if (lens_list[i]->get_lenstype() == PTMASS) singular = true;
					// a radial critical curve will occur if a core is present, OR if alpha > 1 (since kappa goes like r^n where n=alpha-2)
				if (singular) {
					lens_list[i]->get_center_coords(xc,yc);
					lensvector singular_pt(xc,yc);
					singular_pts.push_back(singular_pt);
				}
			}
		}
	}
	n_singular_points = singular_pts.size();
}

void QLens::create_and_add_source_object(SB_ProfileName name, const bool is_lensed, const int band_number, const double zsrc_in, const int emode, const double sb_norm, const double scale, const double scale2, const double index_param, const double q, const double theta, const double xc, const double yc, const double special_param1, const double special_param2, const int pmode)
{
	if (band_number > n_model_bands) die("cannot add band with index that is > number of bands");
	int old_emode = SB_Profile::default_ellipticity_mode;
	if (emode > 1) die("SB emode greater than 1 does not exist");
	if (emode != -1) SB_Profile::default_ellipticity_mode = emode; // set ellipticity mode to user-specified value for this lens

	SB_Profile* new_src = NULL;
	switch (name) {
		case GAUSSIAN:
			new_src = new Gaussian(band_number, zsrc_in, sb_norm, scale, q, theta, xc, yc, this); break;
		case SERSIC:
			new_src = new Sersic(band_number, zsrc_in, sb_norm, scale, index_param, q, theta, xc, yc, pmode, this); break;
		case CORE_SERSIC:
			new_src = new CoreSersic(band_number, zsrc_in, sb_norm, scale, index_param, scale2, special_param1, special_param2, q, theta, xc, yc, this); break;
		case CORED_SERSIC:
			new_src = new Cored_Sersic(band_number, zsrc_in, sb_norm, scale, index_param, scale2, q, theta, xc, yc, this); break;
		case DOUBLE_SERSIC:
			new_src = new DoubleSersic(band_number, zsrc_in, sb_norm, index_param, scale, special_param1, scale2, special_param2, q, theta, xc, yc, this); break;
		case sple:
			new_src = new SPLE(band_number, zsrc_in, sb_norm, index_param, scale, q, theta, xc, yc, this); break;
		case dpie:
			new_src = new dPIE(band_number, zsrc_in, sb_norm, scale, scale2, q, theta, xc, yc, this); break;
		case nfw_SOURCE:
			new_src = new NFW_Source(band_number, zsrc_in, sb_norm, scale, q, theta, xc, yc, this); break;
		case TOPHAT:
			new_src = new TopHat(band_number, zsrc_in, sb_norm, scale, q, theta, xc, yc, this); break;
		default:
			die("Surface brightness profile type not recognized");
	}
	if (new_src==NULL) die("new_src pointer was not set when creating source");
	if (emode != -1) SB_Profile::default_ellipticity_mode = old_emode; // restore ellipticity mode to its default setting
	add_source(new_src,is_lensed);
}

void QLens::add_source(SB_Profile* new_src, const bool is_lensed)
{
	new_src->set_qlens_pointer(this);
	int band_number = new_src->band;
	if (band_number > n_model_bands) die("cannot add band with index that is > number of bands");

	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	int* new_sbprofile_redshift_idx = new int[n_sb+1];
	int* new_sbprofile_band_number = new int[n_sb+1];
	int* new_sbprofile_imggrid_idx = new int[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			newlist[i] = sb_list[i];
			new_sbprofile_redshift_idx[i] = sbprofile_redshift_idx[i];
			new_sbprofile_band_number[i] = sbprofile_band_number[i];
			new_sbprofile_imggrid_idx[i] = sbprofile_imggrid_idx[i];
		}
		delete[] sb_list;
		delete[] sbprofile_redshift_idx;
		delete[] sbprofile_band_number;
		delete[] sbprofile_imggrid_idx;
	}

	sbprofile_redshift_idx = new_sbprofile_redshift_idx;
	sbprofile_band_number = new_sbprofile_band_number;
	sbprofile_imggrid_idx = new_sbprofile_imggrid_idx;
	if (band_number==n_model_bands) add_new_model_band();
	sbprofile_band_number[n_sb] = band_number;
	add_new_extended_src_redshift(new_src->zsrc,n_sb,false);
	sbprofile_imggrid_idx[n_sb] = band_number*n_extended_src_redshifts+sbprofile_redshift_idx[n_sb];
	n_sb++;
	sb_list = newlist;
	sb_list[n_sb-1] = new_src;
	srclist->input_ptr(sb_list,n_sb);
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
	register_sb_vary_parameters(n_sb-1);
}

void QLens::create_and_add_shapelet_source(const bool is_lensed, const int band_number, const double zsrc_in, const double amp00, const double sig_x, const double q, const double theta, const double xc, const double yc, const int nmax, const bool truncate, const int pmode)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	int* new_sbprofile_redshift_idx = new int[n_sb+1];
	int* new_sbprofile_band_number = new int[n_sb+1];
	int* new_sbprofile_imggrid_idx = new int[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			newlist[i] = sb_list[i];
			new_sbprofile_redshift_idx[i] = sbprofile_redshift_idx[i];
			new_sbprofile_band_number[i] = sbprofile_band_number[i];
			new_sbprofile_imggrid_idx[i] = sbprofile_imggrid_idx[i];
		}
		delete[] sb_list;
		delete[] sbprofile_redshift_idx;
		delete[] sbprofile_band_number;
		delete[] sbprofile_imggrid_idx;
	}

	newlist[n_sb] = new Shapelet(band_number, zsrc_in, amp00, sig_x, q, theta, xc, yc, nmax, truncate, pmode, this);
	sbprofile_redshift_idx = new_sbprofile_redshift_idx;
	sbprofile_band_number = new_sbprofile_band_number;
	sbprofile_imggrid_idx = new_sbprofile_imggrid_idx;
	//double zsrc = (is_lensed) ? zsrc_in : -1;
	if (band_number==n_model_bands) add_new_model_band();
	sbprofile_band_number[n_sb] = band_number;
	add_new_extended_src_redshift(zsrc_in,n_sb,false);
	sbprofile_imggrid_idx[n_sb] = band_number*n_extended_src_redshifts+sbprofile_redshift_idx[n_sb];
	n_sb++;
	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
	srclist->input_ptr(sb_list,n_sb);
	if (fft_convolution) cleanup_FFT_convolution_arrays(); // since number of shapelet amp's has changed, will need to redo FFT setup
}

void QLens::create_and_add_mge_source(const bool is_lensed, const int band_number, const double zsrc_in, const double reg, const double amp0, const double sig_i, const double sig_f, const double q, const double theta, const double xc, const double yc, const int nmax, const int pmode)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	int* new_sbprofile_redshift_idx = new int[n_sb+1];
	int* new_sbprofile_band_number = new int[n_sb+1];
	int* new_sbprofile_imggrid_idx = new int[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			newlist[i] = sb_list[i];
			new_sbprofile_redshift_idx[i] = sbprofile_redshift_idx[i];
			new_sbprofile_band_number[i] = sbprofile_band_number[i];
			new_sbprofile_imggrid_idx[i] = sbprofile_imggrid_idx[i];
		}
		delete[] sb_list;
		delete[] sbprofile_redshift_idx;
		delete[] sbprofile_band_number;
		delete[] sbprofile_imggrid_idx;
	}

	newlist[n_sb] = new MGE(band_number, zsrc_in, reg, amp0, sig_i, sig_f, q, theta, xc, yc, nmax, pmode, this);
	sbprofile_redshift_idx = new_sbprofile_redshift_idx;
	sbprofile_band_number = new_sbprofile_band_number;
	sbprofile_imggrid_idx = new_sbprofile_imggrid_idx;
	//double zsrc = (is_lensed) ? zsrc_in : -1;
	if (band_number==n_model_bands) add_new_model_band();
	sbprofile_band_number[n_sb] = band_number;
	add_new_extended_src_redshift(zsrc_in,n_sb,false);
	sbprofile_imggrid_idx[n_sb] = band_number*n_extended_src_redshifts+sbprofile_redshift_idx[n_sb];
	n_sb++;
	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
	srclist->input_ptr(sb_list,n_sb);
	if (fft_convolution) cleanup_FFT_convolution_arrays(); // since number of shapelet amp's has changed, will need to redo FFT setup
}

void QLens::create_and_add_multipole_source(const bool is_lensed, const int band_number, const double zsrc_in, int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool sine_term)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];

	int* new_sbprofile_redshift_idx = new int[n_sb+1];
	int* new_sbprofile_band_number = new int[n_sb+1];
	int* new_sbprofile_imggrid_idx = new int[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			newlist[i] = sb_list[i];
			new_sbprofile_redshift_idx[i] = sbprofile_redshift_idx[i];
			new_sbprofile_band_number[i] = sbprofile_band_number[i];
			new_sbprofile_imggrid_idx[i] = sbprofile_imggrid_idx[i];
		}
		delete[] sb_list;
		delete[] sbprofile_redshift_idx;
		delete[] sbprofile_band_number;
		delete[] sbprofile_imggrid_idx;
	}

	newlist[n_sb] = new SB_Multipole(band_number, zsrc_in, a_m, n, m, theta, xc, yc, sine_term, this);
	sbprofile_redshift_idx = new_sbprofile_redshift_idx;
	sbprofile_band_number = new_sbprofile_band_number;
	sbprofile_imggrid_idx = new_sbprofile_imggrid_idx;
	//double zsrc = (is_lensed) ? zsrc_in : -1;
	if (band_number==n_model_bands) add_new_model_band();
	sbprofile_band_number[n_sb] = band_number;
	add_new_extended_src_redshift(zsrc_in,n_sb,false);
	sbprofile_imggrid_idx[n_sb] = band_number*n_extended_src_redshifts+sbprofile_redshift_idx[n_sb];
	n_sb++;
	sb_list = newlist;
	srclist->input_ptr(sb_list,n_sb);
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}

void QLens::create_and_add_splined_source_object(const char *splinefile, const bool is_lensed, const int band_number, const double zsrc_in, const int emode, const double q, const double theta, const double qx, const double f, const double xc, const double yc)
{
	int old_emode = SB_Profile::default_ellipticity_mode;
	if (emode > 1) die("SB emode greater than 1 does not exist");
	if (emode != -1) SB_Profile::default_ellipticity_mode = emode; // set ellipticity mode to user-specified value for this lens

	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	int* new_sbprofile_redshift_idx = new int[n_sb+1];
	int* new_sbprofile_band_number = new int[n_sb+1];
	int* new_sbprofile_imggrid_idx = new int[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			newlist[i] = sb_list[i];
			new_sbprofile_redshift_idx[i] = sbprofile_redshift_idx[i];
			new_sbprofile_band_number[i] = sbprofile_band_number[i];
			new_sbprofile_imggrid_idx[i] = sbprofile_imggrid_idx[i];
		}
		delete[] sb_list;
		delete[] sbprofile_redshift_idx;
		delete[] sbprofile_band_number;
		delete[] sbprofile_imggrid_idx;
	}

	newlist[n_sb] = new SB_Profile(splinefile, band_number, zsrc_in, q, theta, xc, yc, qx, f, this);
	sbprofile_redshift_idx = new_sbprofile_redshift_idx;
	sbprofile_band_number = new_sbprofile_band_number;
	sbprofile_imggrid_idx = new_sbprofile_imggrid_idx;
	//double zsrc = (is_lensed) ? zsrc_in : -1;
	if (band_number==n_model_bands) add_new_model_band();
	sbprofile_band_number[n_sb] = band_number;
	add_new_extended_src_redshift(zsrc_in,n_sb,false);
	sbprofile_imggrid_idx[n_sb] = band_number*n_extended_src_redshifts+sbprofile_redshift_idx[n_sb];
	n_sb++;

	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
	srclist->input_ptr(sb_list,n_sb);
	if (emode != -1) SB_Profile::default_ellipticity_mode = old_emode; // restore ellipticity mode to its default setting
}

void QLens::remove_source_object(int sb_number, const bool delete_src)
{
	int pi, pf;
	get_sb_parameter_numbers(sb_number,pi,pf);

	if ((sb_number >= n_sb) or (n_sb==0)) { warn(warnings,"Specified source object does not exist"); return; }
	SB_Profile** newlist = new SB_Profile*[n_sb-1];
	int* new_sbprofile_redshift_idx;
	int* new_sbprofile_band_number;
	int* new_sbprofile_imggrid_idx;
	if (n_sb > 1) {
		new_sbprofile_redshift_idx = new int[n_sb-1];
		new_sbprofile_band_number = new int[n_sb-1];
		new_sbprofile_imggrid_idx = new int[n_sb-1];
	}

	remove_old_extended_src_redshift(sbprofile_redshift_idx[sb_number],false); // removes the sbprofile redshift from the list if no other sources share that redshift
	int i,j;
	for (i=0, j=0; i < n_sb; i++) {
		if (i != sb_number) {
			newlist[j] = sb_list[i];
			new_sbprofile_redshift_idx[j] = sbprofile_redshift_idx[i];
			new_sbprofile_band_number[j] = sbprofile_band_number[i];
			new_sbprofile_imggrid_idx[j] = sbprofile_imggrid_idx[i];
			j++;
		}
	}

	for (i=0; i < n_sb; i++) {
		// if any source profiles are anchored to the center of this lens (typically to model foreground light), delete the anchor
		if ((i != sb_number) and (sb_list[i]->center_anchored_to_source==true) and (sb_list[i]->get_center_anchor_number()==sb_number)) sb_list[i]->delete_center_anchor();
		if (i != sb_number) sb_list[i]->unanchor_parameter(sb_list[sb_number]); // this unanchors the source if any of its parameters are anchored to the source being deleted
	}

	for (i=0; i < nlens; i++) {
		lens_list[i]->unanchor_parameter(sb_list[sb_number]); // this unanchors the lens if any of its parameters are anchored to the source being deleted
	}

	if (sb_list[sb_number]->sbtype==SHAPELET) {
		if (fft_convolution) cleanup_FFT_convolution_arrays(); // since number of shapelet amp's has changed, will need to redo FFT setup
	}

	if (delete_src) delete sb_list[sb_number];
	delete[] sb_list;
	delete[] sbprofile_redshift_idx;
	delete[] sbprofile_band_number;
	delete[] sbprofile_imggrid_idx;
	n_sb--;
	if (n_sb > 0) {
		sbprofile_redshift_idx = new_sbprofile_redshift_idx;
		sbprofile_band_number = new_sbprofile_band_number;
		sbprofile_imggrid_idx = new_sbprofile_imggrid_idx;
	}
	else {
		sbprofile_redshift_idx = NULL;
		sbprofile_band_number = NULL;
		sbprofile_imggrid_idx = NULL;
	}

	sb_list = newlist;
	srclist->input_ptr(sb_list,n_sb);
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;

	param_list->remove_params(pi,pf);
}

void QLens::clear_source_objects()
{
	if (n_sb > 0) {
		int pi, pf, pi_min=1000, pf_max=0;
		// since all the source parameters are blocked together in the param_list list, we just need to find the initial and final parameters to remove
		for (int i=0; i < n_sb; i++) {
			get_sb_parameter_numbers(i,pi,pf);
			if (pi < pi_min) pi_min=pi;
			if (pf > pf_max) pf_max=pf;
		}	
		int i,j;
		for (i=0; i < n_sb; i++) {
			for (j=0; j < nlens; j++) {
				lens_list[j]->unanchor_parameter(sb_list[i]); // this unanchors the lens if any of its parameters are anchored to the source being deleted
			}
			delete sb_list[i];
		}	
		delete[] sb_list;
		srclist->clear_ptr();
		if (sbprofile_redshift_idx != NULL) delete[] sbprofile_redshift_idx;
		if (sbprofile_band_number != NULL) delete[] sbprofile_band_number;
		if (sbprofile_imggrid_idx != NULL) delete[] sbprofile_imggrid_idx;
		sbprofile_redshift_idx = NULL;
		param_list->remove_params(pi_min,pf_max);
		n_sb = 0;
		int n_old_zsrc = n_extended_src_redshifts;
		if (n_extended_src_redshifts > 0) {
			for (int i=n_old_zsrc-1; i >= 0; i--) {
				remove_old_extended_src_redshift(i,false); // removes the redshift from the list if no pixellated sources share that redshift
			}
		}
		update_parameter_list(); // this is necessary to keep the parameter priors, transforms preserved
		if (fft_convolution) cleanup_FFT_convolution_arrays(); // since number of shapelet amp's may have changed, will need to redo FFT setup
	}
}

void QLens::print_source_list(bool show_vary_params)
{
	bool show_band = false;
	if (n_model_bands > 1) show_band = true;
	//cout << "N_ZSRC: "<< n_extended_src_redshifts << endl;
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			//cout << "IDX=" << sbprofile_redshift_idx[i] << endl;
			//if (sbprofile_redshift_idx[i]==-1) zs = -1;
			//else zs = ((n_extended_src_redshifts > 1) or (extended_src_redshifts[sbprofile_redshift_idx[0]] != source_redshift)) ? extended_src_redshifts[sbprofile_redshift_idx[i]] : -1;
			cout << i << ". ";
			sb_list[i]->print_parameters(show_band,sbprofile_band_number[i]);
			if (show_vary_params)
				sb_list[i]->print_vary_parameters();
		}
	}
	else cout << "No source models have been specified" << endl;
	cout << endl;
}

void QLens::add_new_model_band()
{
	if (n_extended_src_redshifts > 0) {
		if (n_image_pixel_grids != (n_model_bands*n_extended_src_redshifts)) die("number of image pixel grids is not right...FIX");
		int old_n_image_pixel_grids = n_image_pixel_grids;
		n_image_pixel_grids += n_extended_src_redshifts;
		ImagePixelGrid** new_image_pixel_grids = new ImagePixelGrid*[n_image_pixel_grids];
		int i,j,istart;
		for (j=0, istart=0; j < n_model_bands; j++, istart += n_extended_src_redshifts) {
			for (i=istart; i < istart+n_extended_src_redshifts; i++) new_image_pixel_grids[i] = image_pixel_grids[i];
		}
		for (i=old_n_image_pixel_grids; i < n_image_pixel_grids; i++) new_image_pixel_grids[i] = NULL;
		delete[] image_pixel_grids;
		image_pixel_grids = new_image_pixel_grids;
	}
	n_model_bands++;
}

void QLens::remove_model_band(const int band_number, const bool removing_pixellated_src)
{
	if ((n_model_bands==0) or (band_number >= n_model_bands)) return;
	int i, n_pixsrc_with_band=0, n_sbsrc_with_band=0;
	bool remove_band = false;
	for (i=0; i < n_sb; i++) {
		if (sbprofile_band_number[i]==band_number) {
			n_sbsrc_with_band++;
			if (n_sbsrc_with_band > 1) break;
		}
	}
	for (i=0; i < n_pixellated_src; i++) {
		if (pixellated_src_band[i]==band_number) {
			n_pixsrc_with_band++;
			if (n_pixsrc_with_band > 1) break;
		}
	}
	if (removing_pixellated_src) {
		if ((n_pixsrc_with_band <= 1) and (n_sbsrc_with_band==0) and (n_psf <= band_number)) {
			remove_band = true; // the particular source in question might not have been removed yet
			//cout << "removing band! band_number=" << band_number << " npixz=" << n_pixsrc_with_band << " nsbz=" << n_sbsrc_with_band << endl;
		//} else {
			//cout << "NOT removing band! band_number=" << band_number << " npixz=" << n_pixsrc_with_band << " nsbz=" << n_sbsrc_with_band << endl;
		}
	} else {
		if ((n_sbsrc_with_band <= 1) and (n_pixsrc_with_band==0)) remove_band = true; // the particular source in question might not have been removed yet
	}

	if (remove_band) {
		if (n_extended_src_redshifts > 0) {
			ImagePixelGrid** new_image_pixel_grids;
			if (n_image_pixel_grids != (n_model_bands*n_extended_src_redshifts)) die("number of image pixel grids is not right...FIX!");
			if (n_data_bands > 0) {
				new_image_pixel_grids = new ImagePixelGrid*[n_image_pixel_grids-n_extended_src_redshifts];
				int i,j,k,l;
				for (j=0, k=0, l=0; j < n_model_bands; j++) {
					if (j != band_number) {
						for (i=0; i < n_extended_src_redshifts; i++) {
							new_image_pixel_grids[k++] = image_pixel_grids[l++];
						}
					} else {
						for (i=0; i < n_extended_src_redshifts; i++) {
							delete image_pixel_grids[l++];
						}
					}
				}
			} else {
				for (int i=0; i < n_extended_src_redshifts; i++) {
					delete image_pixel_grids[i++];
				}
			}
			n_image_pixel_grids -= n_extended_src_redshifts;
			delete[] image_pixel_grids;
			if (n_data_bands > 0) {
				image_pixel_grids = new_image_pixel_grids;
			} else {
				image_pixel_grids = NULL;
			}

			// Now update all the sbprofile_band_idx and sbprofile_imggrid_idx, since the indices may have shifted due to removing a band
			for (int i=0; i < n_sb; i++) {
				if (sbprofile_band_number[i] > band_number) {
					sbprofile_band_number[i]--;
					sbprofile_imggrid_idx[i] -= n_extended_src_redshifts;
				}
			}

		}
		n_model_bands--;
	}
}

void QLens::add_pixellated_source(const double zsrc, const int band_number)
{
	if (band_number > n_model_bands) die("cannot add band with index that is > number of bands");
	srcgrids = new ModelParams*[n_pixellated_src+1];
	DelaunaySourceGrid** newlist = new DelaunaySourceGrid*[n_pixellated_src+1];
	CartesianSourceGrid** newlist2 = new CartesianSourceGrid*[n_pixellated_src+1];
	int* new_pixellated_src_redshift_idx = new int[n_pixellated_src+1];
	int* new_pixellated_src_band = new int[n_pixellated_src+1];
	if (n_pixellated_src > 0) {
		for (int i=0; i < n_pixellated_src; i++) {
			newlist[i] = delaunay_srcgrids[i];
			newlist2[i] = cartesian_srcgrids[i];
			new_pixellated_src_redshift_idx[i] = pixellated_src_redshift_idx[i];
			new_pixellated_src_band[i] = pixellated_src_band[i];
		}
		delete[] delaunay_srcgrids;
		delete[] cartesian_srcgrids;
		delete[] pixellated_src_redshift_idx;
		delete[] pixellated_src_band;
	}

	pixellated_src_redshift_idx = new_pixellated_src_redshift_idx;
	pixellated_src_band = new_pixellated_src_band;
	if (band_number==n_model_bands) add_new_model_band();
	pixellated_src_band[n_pixellated_src] = band_number;
	add_new_extended_src_redshift(zsrc,n_pixellated_src,true);

	newlist[n_pixellated_src] = new DelaunaySourceGrid(this,band_number,zsrc); 
	newlist2[n_pixellated_src] = new CartesianSourceGrid(this,band_number,zsrc);

	n_pixellated_src++;
	delaunay_srcgrids = newlist;
	cartesian_srcgrids = newlist2;

	for (int i=0; i < n_pixellated_src; i++) {
		if (source_fit_mode==Delaunay_Source) srcgrids[i] = delaunay_srcgrids[i];
		else srcgrids[i] = cartesian_srcgrids[i];
		delaunay_srcgrids[i]->entry_number = i;
		cartesian_srcgrids[i]->entry_number = i;
	}
	delaunay_srcgrids[n_pixellated_src-1]->srcgrid_redshift = zsrc;
	cartesian_srcgrids[n_pixellated_src-1]->srcgrid_redshift = zsrc;
	pixsrclist->input_ptr(srcgrids,n_pixellated_src);
}

void QLens::remove_pixellated_source(int src_number, const bool delete_pixsrc)
{
	if ((n_pixellated_src==0) or (src_number >= n_pixellated_src)) return;
	int pi,pf;
	get_pixsrc_parameter_numbers(src_number,pi,pf);

	DelaunaySourceGrid** newlist;
	CartesianSourceGrid** newlist2;
	if (n_pixellated_src > 1) {
		delete[] srcgrids;
		srcgrids = new ModelParams*[n_pixellated_src-1];
		newlist = new DelaunaySourceGrid*[n_pixellated_src-1];
		newlist2 = new CartesianSourceGrid*[n_pixellated_src-1];
	}
	int* new_pixellated_src_redshift_idx;
	int* new_pixellated_src_band;
	if (n_pixellated_src > 1) {
		new_pixellated_src_redshift_idx = new int[n_pixellated_src-1];
		new_pixellated_src_band = new int[n_pixellated_src-1];
	}

	remove_model_band(pixellated_src_band[src_number],true); // removes the pixellated_src band from the list if no other sources share that band
	remove_old_extended_src_redshift(pixellated_src_redshift_idx[src_number],true); // removes the pixellated_src redshift from the list if no other sources share that redshift
	int i,j;
	for (i=0, j=0; i < n_pixellated_src; i++) {
		if (i != src_number) {
			newlist[j] = delaunay_srcgrids[i];
			newlist2[j] = cartesian_srcgrids[i];
			new_pixellated_src_redshift_idx[j] = pixellated_src_redshift_idx[i];
			new_pixellated_src_band[j] = pixellated_src_band[i];
			j++;
		}
	}

	if (delete_pixsrc) {
		delete delaunay_srcgrids[src_number];
		delete cartesian_srcgrids[src_number];
	}
	delete[] delaunay_srcgrids;
	delete[] cartesian_srcgrids;
	delete[] pixellated_src_redshift_idx;
	n_pixellated_src--;
	if (n_pixellated_src > 0) {
		pixellated_src_redshift_idx = new_pixellated_src_redshift_idx;
		pixellated_src_band = new_pixellated_src_band;
		delaunay_srcgrids = newlist;
		cartesian_srcgrids = newlist2;
		for (int i=0; i < n_pixellated_src; i++) {
			if (source_fit_mode==Delaunay_Source) srcgrids[i] = delaunay_srcgrids[i];
			else srcgrids[i] = cartesian_srcgrids[i];
			delaunay_srcgrids[i]->entry_number = i;
			cartesian_srcgrids[i]->entry_number = i;
		}
	} else {
		pixellated_src_redshift_idx = NULL;
		pixellated_src_band = NULL;
		srcgrids = NULL;
		delaunay_srcgrids = NULL;
		cartesian_srcgrids = NULL;
	}
	if (pf > pi) param_list->remove_params(pi,pf); // eliminate any fit parameters associated with the source being removed
	pixsrclist->input_ptr(srcgrids,n_pixellated_src);
}

void QLens::print_pixellated_source_list(bool show_vary_params)
{
	cout << resetiosflags(ios::scientific);
	double zs;
	//cout << "N_ZSRC: "<< n_extended_src_redshifts << endl;
	if (n_pixellated_src > 0) {
		for (int i=0; i < n_pixellated_src; i++) {
			//cout << "IDX=" << sbprofile_redshift_idx[i] << endl;
			if (pixellated_src_redshift_idx[i]==-1) zs = -1;
			else zs = extended_src_redshifts[pixellated_src_redshift_idx[i]];
			cout << i << ". ";
			if (source_fit_mode==Delaunay_Source) cout << "delaunay(band=";
			else if (source_fit_mode==Cartesian_Source) cout << "cartesian(band=";
			else cout << "(band=";
			cout << pixellated_src_band[i];
			if (zs > 0) cout << ",zsrc=" << zs;
			if (source_fit_mode==Delaunay_Source) {
				if (delaunay_srcgrids[i] == NULL) cout << "), Delaunay grid not created yet" << endl;
				else {
					if (delaunay_srcgrids[i]->n_gridpts > 0) cout << ",npix=" << delaunay_srcgrids[i]->n_gridpts;
					cout << "): ";
					delaunay_srcgrids[i]->print_parameters();
					if (show_vary_params)
						delaunay_srcgrids[i]->print_vary_parameters();
				}
			} else if (source_fit_mode==Cartesian_Source) {
				if (cartesian_srcgrids[i] == NULL) cout << ", Cartesian grid not created yet" << endl;
				else {
					cout << "): ";
					cartesian_srcgrids[i]->print_parameters();
					if (show_vary_params)
						cartesian_srcgrids[i]->print_vary_parameters();
				}
			} else {
				cout << "): grid undefined (set source mode to 'delaunay' or 'cartesian')" << endl;
			}
		}
	}
	else cout << "No pixellated source objects have been specified" << endl;
	cout << endl;
	if (use_scientific_notation) cout << setiosflags(ios::scientific);
}

void QLens::print_pixellated_lens_list(bool show_vary_params)
{
	cout << resetiosflags(ios::scientific);
	double zl;
	//cout << "N_ZSRC: "<< n_lens_redshifts << endl;
	if (n_pixellated_lens > 0) {
		for (int i=0; i < n_pixellated_lens; i++) {
			//cout << "IDX=" << sbprofile_redshift_idx[i] << endl;
			if (pixellated_lens_redshift_idx[i]==-1) zl = -1;
			else zl = lens_redshifts[pixellated_lens_redshift_idx[i]];
			cout << i << ". ";
			if (lensgrids[i]->grid_type==DelaunayPixelGrid) cout << "delaunay(zlens=";
			else if (lensgrids[i]->grid_type==CartesianPixelGrid) cout << "cartesian(zlens=";
			else cout << "(zlens=";
			if (zl < 0) cout << "undefined";
			else cout << zl;
			if (lensgrids[i] == NULL) cout << "), grid not created yet" << endl;
			else {
				if (lensgrids[i]->n_gridpts > 0) cout << ",npix=" << lensgrids[i]->n_gridpts;
				cout << "): ";
				lensgrids[i]->print_parameters();
				if (show_vary_params)
					lensgrids[i]->print_vary_parameters();
			}
		}
	}
	else cout << "No pixellated lens objects have been specified" << endl;
	cout << endl;
	if (use_scientific_notation) cout << setiosflags(ios::scientific);
}

void QLens::print_psf_list(bool show_vary_params)
{
	cout << resetiosflags(ios::scientific);
	//cout << "N_ZSRC: "<< n_lens_redshifts << endl;
	if (n_psf > 0) {
		for (int i=0; i < n_psf; i++) {
			//cout << "IDX=" << sbprofile_redshift_idx[i] << endl;
			cout << i << ". ";
			if (psf_list[i] == NULL) cout << "not created yet" << endl;
			else if (psf_list[i]->use_input_psf_matrix==true) cout << "pixmap(" << psf_list[i]->psf_npixels_x << "," << psf_list[i]->psf_npixels_y;
			else cout << "gaussian(" << psf_list[i]->psf_npixels_x << "," << psf_list[i]->psf_npixels_y;
			if (psf_list[i] != NULL) {
				cout << "): ";
				psf_list[i]->print_parameters();
				if (show_vary_params)
					psf_list[i]->print_vary_parameters();
			}
		}
	}
	else cout << "No PSF objects have been specified" << endl;
	cout << endl;
	if (use_scientific_notation) cout << setiosflags(ios::scientific);
}

void QLens::find_pixellated_source_moments(const int npix, double& qs, double& phi_s, double& sigavg, double& xavg, double& yavg)
{
	if ((source_fit_mode==Delaunay_Source) and (auto_sourcegrid)) {
		for (int imggrid_i=0; imggrid_i < n_image_pixel_grids; imggrid_i++) {
			image_pixel_grids[imggrid_i]->find_optimal_sourcegrid(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,sourcegrid_limit_xmin,sourcegrid_limit_xmax,sourcegrid_limit_ymin,sourcegrid_limit_ymax); // this will just be for plotting purposes
		}
	}

	if ((delaunay_srcgrids) and (delaunay_srcgrids[0])) {
		delaunay_srcgrids[0]->find_source_moments(npix,qs,phi_s,sigavg,xavg,yavg);
	} else {
		qs = 0;
		phi_s = 0;
		xavg = 0;
		yavg = 0;
	}
}

bool QLens::add_pixellated_lens(const double zlens)
{
	// currenly, there must already be analytic lens(es) present with the given redshift, since pixlens provides potential corrections to these lenses
	bool found_matching_lens_redshift = false;
	int znum;
	for (int i=0; i < n_lens_redshifts; i++) {
		if (lens_redshifts[i]==zlens) { znum = i; found_matching_lens_redshift = true; break; }
	}
	if (!found_matching_lens_redshift) return false; // couldn't find any lenses with the redshift given

	LensPixelGrid** newlist = new LensPixelGrid*[n_pixellated_lens+1];
	int* new_pixellated_lens_redshift_idx = new int[n_pixellated_lens+1];
	if (n_pixellated_lens > 0) {
		for (int i=0; i < n_pixellated_lens; i++) {
			newlist[i] = lensgrids[i];
			new_pixellated_lens_redshift_idx[i] = pixellated_lens_redshift_idx[i];
		}
		delete[] lensgrids;
		delete[] pixellated_lens_redshift_idx;
	}

	new_pixellated_lens_redshift_idx[n_pixellated_lens] = znum;
	newlist[n_pixellated_lens] = new LensPixelGrid(this,znum); 

	n_pixellated_lens++;
	pixellated_lens_redshift_idx = new_pixellated_lens_redshift_idx;
	lensgrids = newlist;
	return true;
}

void QLens::remove_pixellated_lens(int pixlens_number)
{
	//cout << "BANDS: " << n_model_bands << endl;
	if ((n_pixellated_lens==0) or (pixlens_number >= n_pixellated_lens)) return;
	int pi,pf;
	get_pixlens_parameter_numbers(pixlens_number,pi,pf);

	int i,j;
	if (image_pixel_grids != NULL) {
		for (i=0; i < n_extended_src_redshifts; i++) {
			// if one of the image pixel grids has a pointer to the lensgrid in question, set the pointer to NULL since we're about to delete the lensgrid
			if ((image_pixel_grids[i] != NULL) and (image_pixel_grids[i]->lensgrid = lensgrids[pixlens_number])) {
				image_pixel_grids[i]->lensgrid = NULL;
				break;
			}
		}
	}
	LensPixelGrid** newlist;
	if (n_pixellated_lens > 1) {
		newlist = new LensPixelGrid*[n_pixellated_lens-1];
	}
	int* new_pixellated_lens_redshift_idx;
	if (n_pixellated_lens > 1) new_pixellated_lens_redshift_idx = new int[n_pixellated_lens-1];

	//remove_old_lens_redshift(pixellated_lens_redshift_idx[pixlens_number],true);
	for (i=0, j=0; i < n_pixellated_lens; i++) {
		if (i != pixlens_number) {
			newlist[j] = lensgrids[i];
			new_pixellated_lens_redshift_idx[j] = pixellated_lens_redshift_idx[i];
			j++;
		}
	}

	delete lensgrids[pixlens_number];
	delete[] lensgrids;
	delete[] pixellated_lens_redshift_idx;
	n_pixellated_lens--;
	if (n_pixellated_lens > 0) {
		pixellated_lens_redshift_idx = new_pixellated_lens_redshift_idx;
		lensgrids = newlist;
	} else {
		pixellated_lens_redshift_idx = NULL;
		lensgrids = NULL;
	}
	if (pf > pi) param_list->remove_params(pi,pf); // eliminate any fit parameters associated with the source being removed
}

void QLens::add_point_source(const double zsrc, const lensvector &sourcept, const bool vary_source_coords)
{
	PointImageData *new_image_data = new PointImageData[n_ptsrc+1];
	for (int i=0; i < n_ptsrc; i++) {
		new_image_data[i].input(point_image_data[i]);
	}
	if (n_ptsrc > 0) {
		delete[] point_image_data;
	}
	point_image_data = new_image_data;

	PointSource** newlist = new PointSource*[n_ptsrc+1];
	int* new_ptsrc_redshift_idx = new int[n_ptsrc+1];
	if (n_ptsrc > 0) {
		for (int i=0; i < n_ptsrc; i++) {
			newlist[i] = ptsrc_list[i];
			new_ptsrc_redshift_idx[i] = ptsrc_redshift_idx[i];
		}
		delete[] ptsrc_list;
		delete[] ptsrc_redshift_idx;
	}

	ptsrc_redshift_idx = new_ptsrc_redshift_idx;
	add_new_ptsrc_redshift(zsrc,n_ptsrc);

	newlist[n_ptsrc] = new PointSource(this,sourcept,zsrc); 
	if (vary_source_coords) newlist[n_ptsrc]->set_vary_source_coords();

	n_ptsrc++;
	ptsrc_list = newlist;
	for (int i=0; i < n_ptsrc; i++) ptsrc_list[i]->entry_number = i;
	register_ptsrc_vary_parameters(n_ptsrc-1);
	ptsrclist->input_ptr(ptsrc_list,n_ptsrc);
}

void QLens::remove_point_source(int ptsrc_number)
{
	if ((n_ptsrc==0) or (ptsrc_number >= n_ptsrc)) return;
	int pi,pf;
	get_ptsrc_parameter_numbers(ptsrc_number,pi,pf);

	PointSource** newlist;
	int* new_ptsrc_redshift_idx;
	PointImageData *new_image_data;
	if (n_ptsrc > 1) {
		newlist = new PointSource*[n_ptsrc-1];
		new_ptsrc_redshift_idx = new int[n_ptsrc-1];
		new_image_data = new PointImageData[n_ptsrc-1];
	}

	remove_old_ptsrc_redshift(ptsrc_redshift_idx[ptsrc_number]); // removes the ptsrc redshift from the list if no other sources share that redshift
	int i,j;
	for (i=0, j=0; i < n_ptsrc; i++) {
		if (i != ptsrc_number) {
			newlist[j] = ptsrc_list[i];
			new_ptsrc_redshift_idx[j] = ptsrc_redshift_idx[i];
			new_image_data[j].input(point_image_data[i]);
			j++;
		}
	}

	for (i=0; i < n_sb; i++) {
		// if any source profiles are anchored to the center of this point source, delete the anchor
		if ((sb_list[i]->center_anchored_to_ptsrc==true) and (sb_list[i]->get_center_anchor_number()==ptsrc_number)) sb_list[i]->delete_center_anchor();
	}

	delete ptsrc_list[ptsrc_number];
	delete[] ptsrc_list;
	delete[] ptsrc_redshift_idx;
	delete[] point_image_data;
	n_ptsrc--;
	if (n_ptsrc > 0) {
		ptsrc_redshift_idx = new_ptsrc_redshift_idx;
		ptsrc_list = newlist;
		point_image_data = new_image_data;
	} else {
		ptsrc_redshift_idx = NULL;
		ptsrc_list = NULL;
		point_image_data = NULL;
	}
	for (int i=0; i < n_ptsrc; i++) ptsrc_list[i]->entry_number = i;

	if (pf > pi) param_list->remove_params(pi,pf); // eliminate any fit parameters associated with the point source being removed
	ptsrclist->input_ptr(ptsrc_list,n_ptsrc);
}

void QLens::print_point_source_list(bool show_vary_params)
{
	cout << resetiosflags(ios::scientific);
	double zs;
	//cout << "N_ZSRC: "<< n_extended_src_redshifts << endl;
	if (n_ptsrc > 0) {
		for (int i=0; i < n_ptsrc; i++) {
			//cout << "IDX=" << sbprofile_redshift_idx[i] << endl;
			if (ptsrc_redshift_idx[i]==-1) zs = -1;
			else zs = ptsrc_redshifts[ptsrc_redshift_idx[i]];
			cout << i << ". ";
			//cout << "(zsrc=";
			//if (zs < 0) cout << "undefined";
			//else cout << zs;
			if (ptsrc_list[i] == NULL) cout << "Point source object not created yet" << endl;
			else {
				//cout << "): ";
				ptsrc_list[i]->print_parameters();
				if (show_vary_params)
					ptsrc_list[i]->print_vary_parameters();
			}
		}
	}
	else cout << "No point sources have been specified" << endl;
	cout << endl;
	if (use_scientific_notation) cout << setiosflags(ios::scientific);
}

void QLens::add_psf()
{
	PSF** newlist = new PSF*[n_psf+1];
	if (n_psf > 0) {
		for (int i=0; i < n_psf; i++) {
			newlist[i] = psf_list[i];
		}
		delete[] psf_list;
	}

	newlist[n_psf] = new PSF(this); 

	n_psf++;
	psf_list = newlist;

	for (int i=0; i < n_psf; i++) psf_list[i]->entry_number = i;
	if (n_psf > n_model_bands) add_new_model_band();
}

void QLens::remove_psf(int psf_number)
{
	if ((n_psf==0) or (psf_number >= n_psf)) return;
	int pi,pf;
	get_psf_parameter_numbers(psf_number,pi,pf);

	PSF** newlist;
	if (n_psf > 1) {
		newlist = new PSF*[n_psf-1];
		int i,j;
		for (i=0, j=0; i < n_psf; i++) {
			if (i != psf_number) {
				newlist[j] = psf_list[i];
				j++;
			}
		}
	}

	delete psf_list[psf_number];
	delete[] psf_list;
	n_psf--;
	if (n_psf > 0) {
		psf_list = newlist;
	} else {
		psf_list = NULL;
	}
	for (int i=0; i < n_psf; i++) psf_list[i]->entry_number = i;

	if (pf > pi) param_list->remove_params(pi,pf); // eliminate any fit parameters associated with the PSF being removed
}

void QLens::add_image_pixel_data()
{
	ImageData** newlist = new ImageData*[n_data_bands+1];
	if (n_data_bands > 0) {
		for (int i=0; i < n_data_bands; i++) {
			newlist[i] = imgdata_list[i];
		}
		delete[] imgdata_list;
	}

	newlist[n_data_bands] = new ImageData(n_data_bands); 
	newlist[n_data_bands]->set_lens(this);

	//if (n_model_bands != n_data_bands) die("number of model bands does not equal number of data bands (%i versus %i)",n_model_bands,n_data_bands);

	if (n_extended_src_redshifts > 0) {
		if (n_assigned_masks != (n_data_bands*n_extended_src_redshifts)) die("number of assignable masks is not right...FIX");
		int old_n_assigned_masks = n_assigned_masks;
		n_assigned_masks += n_extended_src_redshifts;
		int* new_assigned_mask = new int[n_assigned_masks];
		int i,j,istart;
		for (j=0, istart=0; j < n_data_bands; j++, istart += n_extended_src_redshifts) {
			for (i=istart; i < istart+n_extended_src_redshifts; i++) new_assigned_mask[i] = assigned_mask[i];
		}
		for (i=old_n_assigned_masks; i < n_assigned_masks; i++) new_assigned_mask[i] = 0;
		delete[] assigned_mask;
		assigned_mask = new_assigned_mask;
	}

	n_data_bands++;
	imgdata_list = newlist;
	imgdatalist->input_ptr(imgdata_list,n_data_bands);
}

void QLens::remove_image_pixel_data(int band_number)
{
	if ((n_data_bands==0) or (band_number >= n_data_bands)) return;
	ImageData** newlist;
	if (n_data_bands > 1) {
		newlist = new ImageData*[n_data_bands-1];
		int i,j;
		for (i=0, j=0; i < n_data_bands; i++) {
			if (i != band_number) {
				newlist[j] = imgdata_list[i];
				j++;
			}
		}
	}

	delete imgdata_list[band_number];
	delete[] imgdata_list;
	n_data_bands--;
	// NEED TO REDUCE ASSIGNED_MASK? ADD LINES HERE
	if (n_data_bands > 0) {
		imgdata_list = newlist;
	} else {
		imgdata_list = NULL;
	}
	imgdatalist->input_ptr(imgdata_list,n_data_bands);
}

/*********************** Functions for inserting/removing/updating fit parameters in param_list ************************/

bool QLens::get_lens_parameter_numbers(const int lens_i, int& pi, int& pf)
{
	if (lens_i >= nlens) { pf=pi=0; return false; }
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	vector<string> dummy, dummy2, dummy3;
	for (int i=0; i < lens_i; i++) {
		lens_list[i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	}
	pi = dummy.size();
	if (pi == n_fitparams) { pf=pi=0; return false; }
	lens_list[lens_i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	pf = dummy.size();
	if (pf==pi) return false;
	return true;
}

bool QLens::get_sb_parameter_numbers(const int sb_i, int& pi, int& pf)
{
	if (sb_i >= n_sb) { pf=pi=0; return false; }
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	vector<string> dummy, dummy2, dummy3;
	for (int i=0; i < sb_i; i++) {
		sb_list[i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	}
	pi = dummy.size();
	if (pi == n_fitparams) { pf=pi=0; return false; }
	sb_list[sb_i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	pf = dummy.size();
	pi += lensmodel_fit_parameters; // since lens fit parameters come before the source params
	pf += lensmodel_fit_parameters; // since lens fit parameters come before the source params
	if (pf==pi) return false;
	return true;
}

bool QLens::get_pixsrc_parameter_numbers(const int pixsrc_i, int& pi, int& pf)
{
	if (pixsrc_i >= n_pixellated_src) { pf=pi=0; return false; }
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	vector<string> dummy, dummy2, dummy3;
	for (int i=0; i < pixsrc_i; i++) {
		srcgrids[i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	}
	pi = dummy.size();
	if (pi == n_fitparams) { pf=pi=0; return false; }
	srcgrids[pixsrc_i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);

	pf = dummy.size();
	pi += lensmodel_fit_parameters + srcmodel_fit_parameters; // since lens and sb fit parameters come before the source params
	pf += lensmodel_fit_parameters + srcmodel_fit_parameters; // since lens and sb fit parameters come before the source params
	if (pf==pi) return false;
	return true;
}

bool QLens::get_pixlens_parameter_numbers(const int pixlens_i, int& pi, int& pf)
{
	if (pixlens_i >= n_pixellated_lens) { pf=pi=0; return false; }
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	vector<string> dummy, dummy2, dummy3;
	for (int i=0; i < pixlens_i; i++) {
		lensgrids[i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	}
	pi = dummy.size();
	if (pi == n_fitparams) { pf=pi=0; return false; }
	lensgrids[pixlens_i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);

	pf = dummy.size();
	pi += lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters; // since lens, sb and pixsrc fit parameters come before the pixlens params
	pf += lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters; // since lens, sb and pixsrc fit parameters come before the pixlens params
	if (pf==pi) return false;
	return true;
}

bool QLens::get_ptsrc_parameter_numbers(const int ptsrc_i, int& pi, int& pf)
{
	if (ptsrc_i >= n_ptsrc) { pf=pi=0; return false; }
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	vector<string> dummy, dummy2, dummy3;
	for (int i=0; i < ptsrc_i; i++) {
		ptsrc_list[i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	}
	pi = dummy.size();
	if (pi == n_fitparams) { pf=pi=0; return false; }
	ptsrc_list[ptsrc_i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);

	pf = dummy.size();
	pi += lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters + pixlens_fit_parameters; // since lens, sb, ptsrc and pixlens fit parameters come before the ptsrc params
	pf += lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters + pixlens_fit_parameters; // since lens, sb, ptsrc and pixlens fit parameters come before the ptsrc params
	if (pf==pi) return false;
	return true;
}

bool QLens::get_psf_parameter_numbers(const int psf_i, int& pi, int& pf)
{
	if (psf_i >= n_psf) { pf=pi=0; return false; }
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	vector<string> dummy, dummy2, dummy3;
	for (int i=0; i < psf_i; i++) {
		psf_list[i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);
	}
	pi = dummy.size();
	if (pi == n_fitparams) { pf=pi=0; return false; }
	psf_list[psf_i]->get_fit_parameter_names(dummy,&dummy2,&dummy3);

	pf = dummy.size();
	pi += lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters + pixlens_fit_parameters + ptsrc_fit_parameters; // since lens, sb, pixsrc, pixlens and ptsrc fit parameters come before the psf params
	pf += lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters + pixlens_fit_parameters + ptsrc_fit_parameters; // since lens, sb, pixsrc, pixlens and ptsrc fit parameters come before the psf params
	if (pf==pi) return false;
	return true;
}

bool QLens::get_cosmo_parameter_numbers(int& pi, int& pf)
{
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	vector<string> dummy, dummy2, dummy3;
	cosmo->get_fit_parameter_names(dummy,&dummy2,&dummy3);

	pi = lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters + pixlens_fit_parameters + ptsrc_fit_parameters + psf_fit_parameters; // since lens, sb, pixsrc, pixlens, ptsrc and psf fit parameters come before the cosmology params
	pf = pi + dummy.size();
	if (pf==pi) return false;
	return true;
}

bool QLens::get_misc_parameter_numbers(int& pi, int& pf)
{
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	vector<string> dummy, dummy2, dummy3;
	get_fit_parameter_names(dummy,&dummy2,&dummy3);

	pi = lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters + pixlens_fit_parameters + ptsrc_fit_parameters + psf_fit_parameters + cosmo_fit_parameters; // since lens, sb, pixsrc, pixlens, ptsrc, psf and cosmo fit parameters come before the miscellaneous params
	pf = pi + dummy.size();
	if (pf==pi) return false;
	return true;
}

bool QLens::register_lens_vary_parameters(const int lensnumber)
{
	//cout << "registering params for lens " << lensnumber << endl;
	int pi, pf;
	vector<string> fit_parameter_names, latex_parameter_names;
	get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
	if ((lensnumber < 0) or (lensnumber >= nlens)) return false;
	if (get_lens_parameter_numbers(lensnumber,pi,pf) == true) {
		//cout << "pi=" << pi << " pf=" << pf << endl;
		int index=0, index2=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		dvector stepsizes(npar), values(npar);

		//cout << "Inserting parameters " << pi << " to " << pf << endl;
		lens_list[lensnumber]->get_fit_parameters(values.array(),index2);
		lens_list[lensnumber]->get_auto_stepsizes(stepsizes,index);
		param_list->insert_params(pi,pf,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		//cout << "Inserting parameters done " << endl;
		if (lens_list[lensnumber]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
		}
		//cout << "Updating lens prior limits from " << pi << " to " << (pf-1) << endl;
		//param_list->update_untransformed_values(pi,pf,values);
		lens_list[lensnumber]->get_auto_ranges(use_penalty_limits,lower,upper,index=0);
		param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
		//param_list->print_penalty_limits();
	}
	return true;
}

void QLens::register_lens_prior_limits(const int lens_number)
{
	int pi, pf;
	if (get_lens_parameter_numbers(lens_number,pi,pf) == true) {
		int index=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		if (lens_list[lens_number]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
			//lens_list[lens_number]->get_auto_ranges(use_penalty_limits,lower,upper,index);
			//param_list->update_untransformed_prior_limits(pi,pf,use_penalty_limits,lower,upper);
		}
	}
}

void QLens::update_lens_fitparams(const int lens_number)
{
	int pi, pf;
	if (get_lens_parameter_numbers(lens_number,pi,pf) == true) {
		int index=0, npar = pf-pi;
		dvector values(npar);
		lens_list[lens_number]->get_fit_parameters(values.array(),index);
		param_list->update_untransformed_values(pi,pf,values.array());
	}
}

bool QLens::register_sb_vary_parameters(const int sbnumber)
{
	int pi, pf;
	vector<string> fit_parameter_names, latex_parameter_names;
	get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
	if ((sbnumber < 0) or (sbnumber >= n_sb)) return false;
	if (get_sb_parameter_numbers(sbnumber,pi,pf) == true) {
		int index=0, index2=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		dvector stepsizes(npar), values(npar);
		sb_list[sbnumber]->get_fit_parameters(values.array(),index2);
		sb_list[sbnumber]->get_auto_stepsizes(stepsizes,index=0);
		param_list->insert_params(pi,pf,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (sb_list[sbnumber]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
		}
		sb_list[sbnumber]->get_auto_ranges(use_penalty_limits,lower,upper,index=0);
		//cout << "Updating source prior limits from " << pi << " to " << (pf-1) << endl;
		//param_list->update_untransformed_values(pi,pf,values);
		param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
	}
	return true;
}

void QLens::register_sb_prior_limits(const int sb_number)
{
	int pi, pf;
	if (get_sb_parameter_numbers(sb_number,pi,pf) == true) {
		int index=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		if (sb_list[sb_number]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
			//sb_list[sb_number]->get_auto_ranges(use_penalty_limits,lower,upper,index);
			//param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
		}
	}
}

void QLens::update_sb_fitparams(const int sb_number)
{
	int pi, pf;
	if (get_sb_parameter_numbers(sb_number,pi,pf) == true) {
		int index=0, npar = pf-pi;
		dvector values(npar);
		sb_list[sb_number]->get_fit_parameters(values.array(),index);
		param_list->update_untransformed_values(pi,pf,values.array());
	}
}

bool QLens::update_pixellated_src_varyflag(const int src_number, const string name, const bool flag)
{
	// updates one specific parameter
	int pnum, pi, pf;
	srcgrids[src_number]->get_parameter_vary_index(name,pnum);
	get_pixsrc_parameter_numbers(src_number,pi,pf);
	pnum += pi;
	bool flag0;
	srcgrids[src_number]->get_specific_varyflag(name,flag0);
	if (flag==flag0) return true;
	if (srcgrids[src_number]->update_specific_varyflag(name,flag)==false) return false;
	if (flag==false) {
		param_list->remove_params(pnum,pnum+1);
	} else {
		dvector values(1), stepsizes(1);
		double lower, upper;
		vector<string> fit_parameter_names, latex_parameter_names;
		get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
		srcgrids[src_number]->get_specific_parameter(name,values[0]);
		srcgrids[src_number]->get_specific_stepsize(name,stepsizes[0]);
		param_list->insert_params(pnum,pnum+1,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (srcgrids[src_number]->get_specific_limit(name,lower,upper)==true) {
			param_list->set_untransformed_prior_limit(pnum,lower,upper);
		}

	}
	return true;
}

bool QLens::register_pixellated_src_vary_parameters(const int pixsrc_number)
{
	int pi, pf;
	vector<string> fit_parameter_names, latex_parameter_names;
	get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
	if ((pixsrc_number < 0) or (pixsrc_number >= n_pixellated_src)) return false;
	if (get_pixsrc_parameter_numbers(pixsrc_number,pi,pf) == true) {
		int index=0, index2=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		dvector stepsizes(npar), values(npar);
		srcgrids[pixsrc_number]->get_fit_parameters(values.array(),index2);
		srcgrids[pixsrc_number]->get_auto_stepsizes(stepsizes,index=0);
		//cout << "inserting parameters " << pi << " up to " << pf << endl;
		param_list->insert_params(pi,pf,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (srcgrids[pixsrc_number]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
		}
		srcgrids[pixsrc_number]->get_auto_ranges(use_penalty_limits,lower,upper,index=0);
		//cout << "Updating pixellated src prior limits from " << pi << " to " << (pf-1) << endl;
		param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
	}
	return true;
}

void QLens::register_pixellated_src_prior_limits(const int pixsrc_number)
{
	int pi, pf;
	if (get_pixsrc_parameter_numbers(pixsrc_number,pi,pf) == true) {
		int index=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		if (srcgrids[pixsrc_number]->get_limits(lower,upper)==true) {
			//cout << "Setting prior limits in param_list for params " << pi << " up to " << pf << endl;
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
			//sb_list[pixsrc_number]->get_auto_ranges(use_penalty_limits,lower,upper,index);
			//param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
		}
	}
}

void QLens::update_pixellated_src_fitparams(const int pixsrc_number)
{
	int pi, pf;
	if (get_pixsrc_parameter_numbers(pixsrc_number,pi,pf) == true) {
		int index=0, npar = pf-pi;
		dvector values(npar);
		srcgrids[pixsrc_number]->get_fit_parameters(values.array(),index);
		param_list->update_untransformed_values(pi,pf,values.array());
	}
}

bool QLens::update_pixellated_lens_varyflag(const int pixlens_number, const string name, const bool flag)
{
	// updates one specific parameter
	int pnum, pi, pf;
	lensgrids[pixlens_number]->get_parameter_vary_index(name,pnum);
	get_pixlens_parameter_numbers(pixlens_number,pi,pf);
	pnum += pi;
	bool flag0;
	lensgrids[pixlens_number]->get_specific_varyflag(name,flag0);
	if (flag==flag0) return true;
	if (lensgrids[pixlens_number]->update_specific_varyflag(name,flag)==false) return false;
	if (flag==false) {
		param_list->remove_params(pnum,pnum+1);
	} else {
		dvector values(1), stepsizes(1);
		double lower, upper;
		vector<string> fit_parameter_names, latex_parameter_names;
		get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
		lensgrids[pixlens_number]->get_specific_parameter(name,values[0]);
		lensgrids[pixlens_number]->get_specific_stepsize(name,stepsizes[0]);
		param_list->insert_params(pnum,pnum+1,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (lensgrids[pixlens_number]->get_specific_limit(name,lower,upper)==true) {
			param_list->set_untransformed_prior_limit(pnum,lower,upper);
		}
	}
	return true;
}

bool QLens::set_pixellated_lens_vary_parameters(const int pixlens_number, boolvector &vary_flags)
{
	int pi, pf;
	get_pixlens_parameter_numbers(pixlens_number,pi,pf);
	if (lensgrids[pixlens_number]->set_varyflags(vary_flags)==false) return false;
	if (pf > pi) param_list->remove_params(pi,pf);
	vector<string> fit_parameter_names, latex_parameter_names;
	get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
	if (get_pixlens_parameter_numbers(pixlens_number,pi,pf) == true) {
		int index=0, index2=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		dvector stepsizes(npar), values(npar);
		lensgrids[pixlens_number]->get_fit_parameters(values.array(),index2);
		lensgrids[pixlens_number]->get_auto_stepsizes(stepsizes,index2);
		param_list->insert_params(pi,pf,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (lensgrids[pixlens_number]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
		}
		lensgrids[pixlens_number]->get_auto_ranges(use_penalty_limits,lower,upper,index);
		//cout << "Updating pixellated lens prior limits from " << pi << " to " << (pf-1) << endl;
		param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
	}
	return true;
}

bool QLens::update_ptsrc_varyflag(const int src_number, const string name, const bool flag)
{
	// updates one specific parameter
	int pnum, pi, pf;
	ptsrc_list[src_number]->get_parameter_vary_index(name,pnum);
	get_ptsrc_parameter_numbers(src_number,pi,pf);
	pnum += pi;
	bool flag0;
	ptsrc_list[src_number]->get_specific_varyflag(name,flag0);
	if (flag==flag0) return true;
	if (ptsrc_list[src_number]->update_specific_varyflag(name,flag)==false) return false;
	if (flag==false) {
		param_list->remove_params(pnum,pnum+1);
	} else {
		dvector values(1), stepsizes(1);
		double lower, upper;
		vector<string> fit_parameter_names, latex_parameter_names;
		get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
		ptsrc_list[src_number]->get_specific_parameter(name,values[0]);
		ptsrc_list[src_number]->get_specific_stepsize(name,stepsizes[0]);
		param_list->insert_params(pnum,pnum+1,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (ptsrc_list[src_number]->get_specific_limit(name,lower,upper)==true) {
			param_list->set_untransformed_prior_limit(pnum,lower,upper);
		}
	}
	return true;
}

bool QLens::register_ptsrc_vary_parameters(const int src_number)
{
	int pi, pf;
	vector<string> fit_parameter_names, latex_parameter_names;
	get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
	if ((src_number < 0) or (src_number >= n_ptsrc)) return false;
	if (get_ptsrc_parameter_numbers(src_number,pi,pf) == true) {
		int index=0, index2=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		dvector stepsizes(npar), values(npar);
		ptsrc_list[src_number]->get_fit_parameters(values.array(),index2);
		ptsrc_list[src_number]->get_auto_stepsizes(stepsizes,index=0);
		param_list->insert_params(pi,pf,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (ptsrc_list[src_number]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
		}
		ptsrc_list[src_number]->get_auto_ranges(use_penalty_limits,lower,upper,index=0);
		//cout << "Updating ptsrc prior limits from " << pi << " to " << (pf-1) << endl;
		param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
	}
	return true;
}

void QLens::register_ptsrc_prior_limits(const int ptsrc_number)
{
	int pi, pf;
	if (get_ptsrc_parameter_numbers(ptsrc_number,pi,pf) == true) {
		int index=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		if (ptsrc_list[ptsrc_number]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
			//sb_list[ptsrc_number]->get_auto_ranges(use_penalty_limits,lower,upper,index);
			//param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
		}
	}
}

void QLens::update_ptsrc_fitparams(const int ptsrc_number)
{
	int pi, pf;
	if (get_ptsrc_parameter_numbers(ptsrc_number,pi,pf) == true) {
		int index=0, npar = pf-pi;
		dvector values(npar);
		ptsrc_list[ptsrc_number]->get_fit_parameters(values.array(),index);
		param_list->update_untransformed_values(pi,pf,values.array());
	}
}

bool QLens::update_psf_varyflag(const int psf_number, const string name, const bool flag)
{
	// updates one specific parameter
	int pnum, pi, pf;
	psf_list[psf_number]->get_parameter_vary_index(name,pnum);
	get_psf_parameter_numbers(psf_number,pi,pf);
	pnum += pi;
	bool flag0;
	psf_list[psf_number]->get_specific_varyflag(name,flag0);
	if (flag==flag0) return true;
	if (psf_list[psf_number]->update_specific_varyflag(name,flag)==false) return false;
	if (flag==false) {
		param_list->remove_params(pnum,pnum+1);
	} else {
		dvector values(1), stepsizes(1);
		double lower, upper;
		vector<string> fit_parameter_names, latex_parameter_names;
		get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
		psf_list[psf_number]->get_specific_parameter(name,values[0]);
		psf_list[psf_number]->get_specific_stepsize(name,stepsizes[0]);
		param_list->insert_params(pnum,pnum+1,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (psf_list[psf_number]->get_specific_limit(name,lower,upper)==true) {
			param_list->set_untransformed_prior_limit(pnum,lower,upper);
		}
	}
	return true;
}

bool QLens::set_psf_vary_parameters(const int psf_number, boolvector &vary_flags)
{
	int pi, pf;
	get_psf_parameter_numbers(psf_number,pi,pf);
	if (psf_list[psf_number]->set_varyflags(vary_flags)==false) return false;
	if (pf > pi) param_list->remove_params(pi,pf);
	vector<string> fit_parameter_names, latex_parameter_names;
	get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
	if (get_psf_parameter_numbers(psf_number,pi,pf) == true) {
		int index=0, index2=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		dvector stepsizes(npar), values(npar);
		psf_list[psf_number]->get_fit_parameters(values.array(),index);
		psf_list[psf_number]->get_auto_stepsizes(stepsizes,index);
		param_list->insert_params(pi,pf,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (psf_list[psf_number]->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
		}
		psf_list[psf_number]->get_auto_ranges(use_penalty_limits,lower,upper,index);
		//cout << "Updating lens plimits from " << pi << " to " << (pf-1) << endl;
		param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
	}
	return true;
}

bool QLens::update_cosmo_varyflag(const string name, const bool flag)
{
	// updates one specific parameter
	int pnum, pi, pf;
	cosmo->get_parameter_vary_index(name,pnum);
	get_cosmo_parameter_numbers(pi,pf);
	pnum += pi;
	bool flag0;
	cosmo->get_specific_varyflag(name,flag0);
	if (flag==flag0) return true;
	if (cosmo->update_specific_varyflag(name,flag)==false) return false;
	if (flag==false) {
		param_list->remove_params(pnum,pnum+1);
	} else {
		vector<string> fit_parameter_names, latex_parameter_names;
		get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
		dvector values(1), stepsizes(1);
		double lower, upper;
		cosmo->get_specific_parameter(name,values[0]);
		cosmo->get_specific_stepsize(name,stepsizes[0]);
		param_list->insert_params(pnum,pnum+1,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (cosmo->get_specific_limit(name,lower,upper)==true) {
			param_list->set_untransformed_prior_limit(pnum,lower,upper);
		}

	}
	return true;
}

bool QLens::register_cosmo_vary_parameters()
{
	int pi, pf;
	vector<string> fit_parameter_names, latex_parameter_names;
	get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
	if (get_cosmo_parameter_numbers(pi,pf) == true) {
		int index=0, index2=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		dvector stepsizes(npar), values(npar);
		cosmo->get_fit_parameters(values.array(),index2);
		cosmo->get_auto_stepsizes(stepsizes,index=0);
		param_list->insert_params(pi,pf,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (cosmo->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
		}
		cosmo->get_auto_ranges(use_penalty_limits,lower,upper,index=0);
		//cout << "Updating cosmo prior limits from " << pi << " to " << (pf-1) << endl;
		param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
	}
	return true;
}

void QLens::register_cosmo_prior_limits()
{
	int pi, pf;
	if (get_cosmo_parameter_numbers(pi,pf) == true) {
		int index=0, npar = pf-pi;
		boolvector use_penalty_limits(npar);
		dvector lower(npar), upper(npar);
		if (cosmo->get_limits(lower,upper)==true) {
			param_list->set_untransformed_prior_limits(pi,pf,lower,upper);
			//sb_list[cosmo_number]->get_auto_ranges(use_penalty_limits,lower,upper,index);
			//param_list->update_untransformed_prior_limits_from_auto_ranges(pi,pf,use_penalty_limits,lower,upper);
		}
	}
}

void QLens::update_cosmo_fitparams()
{
	int pi, pf;
	if (get_cosmo_parameter_numbers(pi,pf) == true) {
		int index=0, npar = pf-pi;
		dvector values(npar);
		cosmo->get_fit_parameters(values.array(),index);
		param_list->update_untransformed_values(pi,pf,values.array());
	}
}

bool QLens::update_misc_varyflag(const string name, const bool flag)
{
	// updates one specific parameter
	int pnum, pi, pf;
	get_parameter_vary_index(name,pnum);
	get_misc_parameter_numbers(pi,pf);
	pnum += pi;
	bool flag0;
	get_specific_varyflag(name,flag0);
	if (flag==flag0) return true;
	if (update_specific_varyflag(name,flag)==false) return false;
	if (flag==false) {
		param_list->remove_params(pnum,pnum+1);
	} else {
		vector<string> fit_parameter_names, latex_parameter_names;
		get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
		dvector values(1), stepsizes(1);
		double lower, upper;
		get_specific_parameter(name,values[0]);
		get_specific_stepsize(name,stepsizes[0]);
		param_list->insert_params(pnum,pnum+1,fit_parameter_names.data(),latex_parameter_names.data(),values.array(),stepsizes.array());
		if (get_specific_limit(name,lower,upper)==true) {
			param_list->set_untransformed_prior_limit(pnum,lower,upper);
		}

	}
	return true;
}

void QLens::update_pixsrc_active_parameters(const int src_number)
{
	int pi,pf;
	get_pixsrc_parameter_numbers(src_number,pi,pf);
	update_active_parameters(srcgrids[src_number],pi);
}

void QLens::update_pixlens_active_parameters(const int pixlens_number)
{
	int pi,pf;
	get_pixlens_parameter_numbers(pixlens_number,pi,pf);
	update_active_parameters(lensgrids[pixlens_number],pi);
}

void QLens::update_ptsrc_active_parameters(const int src_number)
{
	int pi,pf;
	get_ptsrc_parameter_numbers(src_number,pi,pf);
	update_active_parameters(ptsrc_list[src_number],pi);
}

void QLens::update_active_parameters(ModelParams* param_object, const int pi)
{
	boolvector turned_off_params;
	param_object->update_active_params(mpi_id,turned_off_params);
	int pnum;
	for (int i=turned_off_params.size()-1; i >= 0; i--) {
		if (turned_off_params[i]) {
			pnum = pi+i;
			param_list->remove_params(pnum,pnum+1);
		}
	}
}

void QLens::get_automatic_initial_stepsizes(dvector& stepsizes)
{
	int i, index=0;
	for (i=0; i < nlens; i++) lens_list[i]->get_auto_stepsizes(stepsizes,index);
	for (i=0; i < n_sb; i++) sb_list[i]->get_auto_stepsizes(stepsizes,index);
	for (i=0; i < n_pixellated_src; i++) {
		if (srcgrids[i] != NULL) srcgrids[i]->get_auto_stepsizes(stepsizes,index);
	}
	for (i=0; i < n_pixellated_lens; i++) lensgrids[i]->get_auto_stepsizes(stepsizes,index);
	for (i=0; i < n_ptsrc; i++) ptsrc_list[i]->get_auto_stepsizes(stepsizes,index);
	for (i=0; i < n_psf; i++) psf_list[i]->get_auto_stepsizes(stepsizes,index);
	cosmo->get_auto_stepsizes(stepsizes,index);
	get_auto_stepsizes(stepsizes,index);

	//if (index != n_fit_parameters) die("Index didn't go through all the fit parameters when setting default stepsizes (%i vs %i)",index,n_fit_parameters);
}

void QLens::set_default_plimits()
{
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	boolvector use_penalty_limits(n_fitparams);
	dvector lower(n_fitparams), upper(n_fitparams);
	int i, index=0;
	for (i=0; i < n_fitparams; i++) use_penalty_limits[i] = false; // default

	for (i=0; i < nlens; i++) lens_list[i]->get_auto_ranges(use_penalty_limits,lower,upper,index);
	for (i=0; i < n_sb; i++) sb_list[i]->get_auto_ranges(use_penalty_limits,lower,upper,index);
	for (i=0; i < n_pixellated_src; i++) {
		if (srcgrids[i] != NULL) srcgrids[i]->get_auto_ranges(use_penalty_limits,lower,upper,index);
	}
	for (i=0; i < n_pixellated_lens; i++) lensgrids[i]->get_auto_ranges(use_penalty_limits,lower,upper,index);
	for (i=0; i < n_ptsrc; i++) ptsrc_list[i]->get_auto_ranges(use_penalty_limits,lower,upper,index);
	for (i=0; i < n_psf; i++) psf_list[i]->get_auto_ranges(use_penalty_limits,lower,upper,index);
	cosmo->get_auto_ranges(use_penalty_limits,lower,upper,index);
	get_auto_ranges(use_penalty_limits,lower,upper,index);

	if (index != n_fitparams) die("Index didn't go through all the fit parameters when setting default ranges (%i vs %i)",index,n_fitparams);
	param_list->update_untransformed_prior_limits_from_auto_ranges(0,n_fitparams,use_penalty_limits,lower,upper);
}

void QLens::get_n_fit_parameters(int &nparams)
{
	int i;
	lensmodel_fit_parameters = 0;
	srcmodel_fit_parameters = 0;
	pixsrc_fit_parameters = 0;
	pixlens_fit_parameters = 0;
	ptsrc_fit_parameters = 0;
	psf_fit_parameters = 0;
	cosmo_fit_parameters = 0;
	for (i=0; i < nlens; i++) lensmodel_fit_parameters += lens_list[i]->get_n_vary_params();
	nparams = lensmodel_fit_parameters;
	for (i=0; i < n_sb; i++) srcmodel_fit_parameters += sb_list[i]->get_n_vary_params();
	nparams += srcmodel_fit_parameters;
	for (i=0; i < n_pixellated_src; i++) {
		if (srcgrids[i] != NULL) pixsrc_fit_parameters += srcgrids[i]->get_n_vary_params();
	}
	nparams += pixsrc_fit_parameters;
	for (i=0; i < n_pixellated_lens; i++) pixlens_fit_parameters += lensgrids[i]->get_n_vary_params();
	nparams += pixlens_fit_parameters;
	for (i=0; i < n_ptsrc; i++) ptsrc_fit_parameters += ptsrc_list[i]->get_n_vary_params();
	nparams += ptsrc_fit_parameters;
	for (i=0; i < n_psf; i++) psf_fit_parameters += psf_list[i]->get_n_vary_params();
	nparams += psf_fit_parameters;
	cosmo_fit_parameters = cosmo->get_n_vary_params();
	nparams += cosmo_fit_parameters;
	nparams += n_vary_params; // generic parameters within qlens class
}

bool QLens::update_parameter_list(const bool check_current_params) // if check_current_params=true, tests whether fitparams values match the actual values in the various model objects (and dies if there is a discrepancy)
{
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	if (n_fitparams != param_list->nparams) die("RUHROH, number of parameters didn't match");
	if (n_fitparams==0) { warn("no parameters are being varied"); return false; }
	//fitparams.input(n_fitparams);
	//double *fitparams = param_list->values;
	double *fitparams = new double[n_fitparams];
	int i, index = 0;
	for (i=0; i < nlens; i++) lens_list[i]->get_fit_parameters(fitparams,index);
	if (index != lensmodel_fit_parameters) die("Index didn't go through all the lens model fit parameters (%i vs %i)",index,lensmodel_fit_parameters);
	for (i=0; i < n_sb; i++) sb_list[i]->get_fit_parameters(fitparams,index);
	for (i=0; i < n_pixellated_src; i++) {
		if (srcgrids[i] != NULL) srcgrids[i]->get_fit_parameters(fitparams,index);
	}
	for (i=0; i < n_pixellated_lens; i++) lensgrids[i]->get_fit_parameters(fitparams,index);
	for (i=0; i < n_ptsrc; i++) ptsrc_list[i]->get_fit_parameters(fitparams,index);
	for (i=0; i < n_psf; i++) psf_list[i]->get_fit_parameters(fitparams,index);
	cosmo->get_fit_parameters(fitparams,index); // cosmology parameters
	get_fit_parameters(fitparams,index); // generic parameters in qlens class
	int expected_index = lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters + pixlens_fit_parameters + ptsrc_fit_parameters + psf_fit_parameters + cosmo_fit_parameters + n_vary_params;
	if (index != expected_index) die("Index didn't go through all model fit parameters (%i vs %i)",index,expected_index);

	vector<string> fit_parameter_names, latex_parameter_names;
	get_all_parameter_names(fit_parameter_names,latex_parameter_names); // we have to generate all parameter names so it can add indices to avoid identical parameter names if needed
	dvector stepsizes(n_fitparams);
	get_automatic_initial_stepsizes(stepsizes);
	param_list->update_untransformed_values(fitparams);
	param_list->update_param_list(fit_parameter_names.data(),latex_parameter_names.data(),stepsizes.array(),check_current_params);
	param_list->transform_parameters();
	delete[] fitparams;
	//set_default_plimits();
	//transformed_parameter_names.resize(n_fitparams);
	//transformed_latex_parameter_names.resize(n_fitparams);
	//param_list->transform_parameter_names(fit_parameter_names.data(),transformed_parameter_names.data(),latex_parameter_names.data(),transformed_latex_parameter_names.data());

	//if ((!ignore_limits) and (fitmethod!=POWELL) and (fitmethod!=SIMPLEX)) return setup_limits();
	return true;
}

/*
bool QLens::setup_limits()
{
	param_list->prior_limits_hi.input(n_fit_parameters);
	param_list->prior_limits_lo.input(n_fit_parameters);
	int index=0;
	for (int i=0; i < nlens; i++) {
		if ((lens_list[i]->get_n_vary_params() > 0) and (lens_list[i]->get_limits(param_list->prior_limits_lo,param_list->prior_limits_hi,index)==false)) { warn("limits have not been defined for lens %i",i); return false; }
	}
	if (index != lensmodel_fit_parameters) die("index didn't go through all the lens model fit parameters when setting upper/lower limits");
	//if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
	for (int i=0; i < n_sb; i++) {
		if ((sb_list[i]->get_n_vary_params() > 0) and (sb_list[i]->get_limits(param_list->prior_limits_lo,param_list->prior_limits_hi,index)==false)) { warn("limits have not been defined for source %i",i); return false; }
	}
	for (int i=0; i < n_pixellated_src; i++) {
		if ((srcgrids[i] != NULL) and (srcgrids[i]->get_n_vary_params() > 0) and (srcgrids[i]->get_limits(param_list->prior_limits_lo,param_list->prior_limits_hi,index)==false)) { warn("limits have not been defined for pixellated source %i",i); return false; }
	}
	for (int i=0; i < n_pixellated_lens; i++) {
		if ((lensgrids[i]->get_n_vary_params() > 0) and (lensgrids[i]->get_limits(param_list->prior_limits_lo,param_list->prior_limits_hi,index)==false)) { warn("limits have not been defined for pixellated source %i",i); return false; }
	}
	for (int i=0; i < n_ptsrc; i++) {
		if ((ptsrc_list[i]->get_n_vary_params() > 0) and (ptsrc_list[i]->get_limits(param_list->prior_limits_lo,param_list->prior_limits_hi,index)==false)) { warn("limits have not been defined for point source %i",i); return false; }
	}
	for (int i=0; i < n_psf; i++) {
		if ((psf_list[i]->get_n_vary_params() > 0) and (psf_list[i]->get_limits(param_list->prior_limits_lo,param_list->prior_limits_hi,index)==false)) { warn("limits have not been defined for point source %i",i); return false; }
	}

	int expected_index = lensmodel_fit_parameters + srcmodel_fit_parameters + pixsrc_fit_parameters + pixlens_fit_parameters + ptsrc_fit_parameters + psf_fit_parameters;
	if (index != expected_index) die("index didn't go through all the lens+source model fit parameters when setting upper/lower limits (%i vs %i)", index, expected_index);

	if ((cosmo->get_n_vary_params() > 0) and (cosmo->get_limits(param_list->prior_limits_lo,param_list->prior_limits_hi,index)==false)) { warn("limits have not been defined for cosmology parameters"); return false; }

	if ((n_vary_params > 0) and (get_limits(param_list->prior_limits_lo,param_list->prior_limits_hi,index)==false)) { warn("limits have not been defined for generic parameters"); return false; }

	if (index != n_fit_parameters) die("index didn't go through all the fit parameters when setting upper/lower limits (%i expected, %i found)",n_fit_parameters,index);
	param_list->transform_limits(param_list->prior_limits_lo,param_list->prior_limits_hi);
	param_list->override_limits(param_list->prior_limits_lo,param_list->prior_limits_hi);
	param_list->set_prior_norms(param_list->prior_limits_lo,param_list->prior_limits_hi);
	for (int i=0; i < n_fit_parameters; i++) {
		if (param_list->prior_limits_lo[i] > param_list->prior_limits_hi[i]) {
			double temp = param_list->prior_limits_hi[i]; param_list->prior_limits_hi[i] = param_list->prior_limits_lo[i]; param_list->prior_limits_lo[i] = temp;
		}
	}
	return true;
}
*/

void QLens::get_all_parameter_names(vector<string>& fit_parameter_names, vector<string>& latex_parameter_names)
{
	int n_fitparams;
	get_n_fit_parameters(n_fitparams);
	fit_parameter_names.clear();
	latex_parameter_names.clear();
	vector<string> latex_parameter_subscripts;
	int i,j;
	for (i=0; i < nlens; i++) {
		lens_list[i]->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);
	}
	for (i=0; i < n_sb; i++) {
		sb_list[i]->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts,true);
	}
	for (i=0; i < n_pixellated_src; i++) {
		if (srcgrids[i] != NULL) srcgrids[i]->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);
	}
	for (i=0; i < n_pixellated_lens; i++) {
		lensgrids[i]->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);
	}
	for (i=0; i < n_ptsrc; i++) {
		ptsrc_list[i]->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);
	}
	for (i=0; i < n_psf; i++) {
		psf_list[i]->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);
	}
	cosmo->get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);
	get_fit_parameter_names(fit_parameter_names,&latex_parameter_names,&latex_parameter_subscripts);

	// find any parameters with matching names and number them so they can be distinguished
	int count, n_names;
	n_names = fit_parameter_names.size();
	string *new_parameter_names = new string[n_names];
	for (i=0; i < n_names; i++) {
		count=1;
		new_parameter_names[i] = fit_parameter_names[i];
		for (j=i+1; j < n_names; j++) {
			if (fit_parameter_names[j]==fit_parameter_names[i]) {
				if (count==1) {
					stringstream countstr;
					string countstring;
					countstr << count;
					countstr >> countstring;
					if (isdigit(new_parameter_names[i].at(new_parameter_names[i].length()-1))) new_parameter_names[i] += "_"; // in case parameter name already ends with a number
					new_parameter_names[i] += countstring;
					if (latex_parameter_subscripts[i].empty()) latex_parameter_subscripts[i] = countstring;
					else latex_parameter_subscripts[i] += "," + countstring;
					count++;
				}
				stringstream countstr;
				string countstring;
				countstr << count;
				countstr >> countstring;
				if (isdigit(fit_parameter_names[j].at(fit_parameter_names[j].length()-1))) fit_parameter_names[j] += "_"; // in case parameter name already ends with a number
				fit_parameter_names[j] += countstring;
				if (latex_parameter_subscripts[j].empty()) latex_parameter_subscripts[j] = countstring;
				else latex_parameter_subscripts[j] += "," + countstring;
				count++;
			}
		}
		fit_parameter_names[i] = new_parameter_names[i];
	}
	delete[] new_parameter_names;

	if (fit_parameter_names.size() != n_fitparams) die("get_all_parameter_names(...) did not assign names to all the fit parameters (%i vs %i)",n_fitparams,fit_parameter_names.size());
	for (i=0; i < n_fitparams; i++) {
		if (latex_parameter_subscripts[i] != "") latex_parameter_names[i] += "_{" + latex_parameter_subscripts[i] + "}";
	}
}

bool QLens::lookup_parameter_value(const string pname, double& pval)
{
	bool found_param = false;
	int i;
	for (i=0; i < param_list->nparams; i++) {
		if (param_list->param_names[i]==pname) {
			found_param = true;
			pval = param_list->values[i];
		}
	}
	if (!found_param) {
		for (i=0; i < dparam_list->n_dparams; i++) {
			if (dparam_list->dparams[i]->name==pname) {
				found_param = true;
				pval = dparam_list->dparams[i]->get_derived_param(this);
			}
		}
	}
	return found_param;
}

void QLens::create_parameter_value_string(string &pvals)
{
	pvals = "";
	int i;
	int n_derived_params = dparam_list->n_dparams;
	for (i=0; i < param_list->nparams; i++) {
		stringstream pvalstr;
		string pvalstring;
		pvalstr << param_list->values[i];
		pvalstr >> pvalstring;
		pvals += pvalstring;
		if ((n_derived_params > 0) or (i < param_list->nparams-1)) pvals += " ";
	}
	double pval;
	for (i=0; i < n_derived_params; i++) {
		pval = dparam_list->dparams[i]->get_derived_param(this);
		stringstream pvalstr;
		string pvalstring;
		pvalstr << pval;
		pvalstr >> pvalstring;
		pvals += pvalstring;
		if (i < n_derived_params-1) pvals += " ";
	}
}

/*
bool QLens::output_parameter_values()
{
	int n_fitparams = param_list->nparams;
	if (n_fitparams==0) return false;
	if (mpi_id==0) {
		for (int i=0; i < n_fitparams; i++) {
			cout << i << ". " << param_list->param_names[i] << ": " << param_list->values[i] << endl;
		}
		cout << endl;
	}
	return true;
}
*/

/*********************** Functions for setting up grid for critical curves, image searching ***********************/

void QLens::set_gridcenter(double xc, double yc)
{
	grid_xcenter=xc;
	grid_ycenter=yc;
	if (autocenter) autocenter = false;
}

void QLens::set_gridsize(double xl, double yl)
{
	grid_xlength = xl;
	grid_ylength = yl;
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
	double grid_xmin, grid_xmax, grid_ymin, grid_ymax;
	grid_xmin = grid_xcenter - grid_xlength/2;
	grid_xmax = grid_xcenter + grid_xlength/2;
	grid_ymin = grid_ycenter - grid_ylength/2;
	grid_ymax = grid_ycenter + grid_ylength/2;
	if (n_image_pixel_grids > 0) {
		for (int i=0; i < n_extended_src_redshifts; i++) {
			if ((image_pixel_grids[i]) and (image_pixel_grids[i]->image_data == NULL)) {
				// if an image pixel grid has been created, but it is not tied to a data image, then update the dimensions of this grid
				image_pixel_grids[i]->update_grid_dimensions(grid_xmin,grid_xmax,grid_ymin,grid_ymax);
			}
		}
	}
}

void QLens::set_grid_corners(double xmin, double xmax, double ymin, double ymax)
{
	grid_xcenter = 0.5*(xmax+xmin);
	grid_ycenter = 0.5*(ymax+ymin);
	grid_xlength = xmax-xmin;
	grid_ylength = ymax-ymin;
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
	if (n_image_pixel_grids > 0) {
		for (int i=0; i < n_extended_src_redshifts; i++) {
			if ((image_pixel_grids[i]) and (image_pixel_grids[i]->image_data == NULL)) {
			//if ((n_extended_src_redshifts > 0) and ((!imgdata_list) or (!imgdata_list[0])) and (default_data_pixel_size <= 0)) {
				// if an image pixel grid has been created, but it is not tied to a data image, then update the dimensions of this grid
				image_pixel_grids[i]->update_grid_dimensions(xmin,xmax,ymin,ymax);
			}
		}
	}
}

void QLens::set_grid_from_pixels()
{
	double pixsize_x, pixsize_y;
	if ((n_data_bands > 0) and (imgdata_list[0])) {
		pixsize_x = imgdata_list[0]->pixel_size;
		pixsize_y = imgdata_list[0]->pixel_xy_ratio;
	} else {
		pixsize_x=pixsize_y=default_data_pixel_size;
	}
	if (default_data_pixel_size <= 0) {
		warn("data pixel size <= 0; cannot set grid from pixel size");
		return;
	}
	grid_xlength = n_image_pixels_x * pixsize_x;
	grid_ylength = n_image_pixels_y * pixsize_y;
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
	if (n_image_pixel_grids > 0) {
		for (int i=0; i < n_extended_src_redshifts; i++) {
			if ((image_pixel_grids[i]) and (image_pixel_grids[i]->image_data == NULL)) {
				double grid_xmin, grid_xmax, grid_ymin, grid_ymax;
				grid_xmin = grid_xcenter - grid_xlength/2;
				grid_xmax = grid_xcenter + grid_xlength/2;
				grid_ymin = grid_ycenter - grid_ylength/2;
				grid_ymax = grid_ycenter + grid_ylength/2;
				// if an image pixel grid has been created, but it is not tied to a data image, then update the dimensions of this grid
				image_pixel_grids[i]->update_grid_dimensions(grid_xmin,grid_xmax,grid_ymin,grid_ymax);
			}
		}
	}
}

void QLens::set_img_npixels(const int npix_x, const int npix_y)
{
	n_image_pixels_x = npix_x;
	n_image_pixels_y = npix_y;
	if (n_image_pixel_grids > 0) {
		//if ((n_extended_src_redshifts > 0) and ((!imgdata_list) or (!imgdata_list[0]))) {
		// only remake the image pixel grids if they're not tied to image data
		for (int i=0; i < n_extended_src_redshifts; i++) {
			if ((image_pixel_grids[i]) and (image_pixel_grids[i]->image_data == NULL)) {
				delete image_pixel_grids[i];
				image_pixel_grids[i] = NULL; // this will force qlens to recreate the grid with the new pixel dimensions
			}
		}
	}
}

void QLens::autogrid(double rmin, double rmax, double frac)
{
	cc_rmin = rmin;
	cc_rmax = rmax;
	autogrid_frac = frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

void QLens::autogrid(double rmin, double rmax)
{
	cc_rmin = rmin;
	cc_rmax = rmax;
	autogrid_frac = default_autogrid_frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

void QLens::autogrid() {
	cc_rmin = default_autogrid_rmin;
	cc_rmax = default_autogrid_rmax;
	autogrid_frac = default_autogrid_frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		//else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

bool QLens::create_grid(bool verbal, double *zfacs, double **betafacs, const int redshift_index) // the last (optional) argument indicates which images are being fit to; used to optimize the subgridding
{
	if (nlens==0) { warn(warnings, "no lens model is specified"); return false; }
	double mytime0, mytime;
#ifdef USE_OPENMP
	if (show_wtime) {
		mytime0=omp_get_wtime();
	}
#endif
	lensvector *centers;
	double *einstein_radii;
	int i_primary=0;
	if ((subgrid_around_perturbers) and (nlens > 1)) {
		centers = new lensvector[nlens];
		einstein_radii = new double[nlens];
		find_effective_lens_centers_and_einstein_radii(centers,einstein_radii,i_primary,zfacs,betafacs,verbal);
	}
	if (grid != NULL) {
		int rsp, thetasp;
		grid->get_usplit_initial(rsp);
		grid->get_wsplit_initial(thetasp);
		if ((rsp != usplit_initial) or (thetasp != wsplit_initial)) {
			delete grid;
			grid = NULL;
		}
		if (auto_store_cc_points) {
			critical_curve_pts.clear();
			caustic_pts.clear();
			length_of_cc_cell.clear();
			sorted_critical_curves = false;
			sorted_critical_curve.clear();
		}
	}
	record_singular_points(zfacs); // grid cells will split around singular points (e.g. center of point mass, etc.)

	Grid::set_splitting(usplit_initial, wsplit_initial, splitlevels, cc_splitlevels, min_cell_area, cc_neighbor_splittings);
	Grid::set_enforce_min_area(true);
	Grid::set_lens(this);
	if ((autogrid_before_grid_creation) or (autocenter) or (auto_gridsize_from_einstein_radius)) find_automatic_grid_position_and_size(zfacs);
	double rmax = 0.5*dmax(grid_xlength,grid_ylength);

	if ((verbal) and (mpi_id==0)) cout << "Creating grid..." << flush;
	if (grid != NULL) {
		if (radial_grid)
			grid->redraw_grid(rmin_frac*rmax, rmax, grid_xcenter, grid_ycenter, 1, zfacs, betafacs); // setting grid_q to 1 for the moment...I will play with that later
		else
			grid->redraw_grid(grid_xcenter, grid_ycenter, grid_xlength, grid_ylength, zfacs, betafacs);
	} else {
		if (radial_grid)
			grid = new Grid(rmin_frac*rmax, rmax, grid_xcenter, grid_ycenter, 1, zfacs, betafacs); // setting grid_q to 1 for the moment...I will play with that later
		else
			grid = new Grid(grid_xcenter, grid_ycenter, grid_xlength, grid_ylength, zfacs, betafacs);
	}
	if ((subgrid_around_perturbers) and (nlens > 1)) {
		subgrid_around_perturber_galaxies(centers,einstein_radii,i_primary,zfacs,betafacs,redshift_index);
		delete[] centers;
		delete[] einstein_radii;
	}
	if (auto_store_cc_points==true) grid->store_critical_curve_pts();
	if ((verbal) and (mpi_id==0)) {
		cout << "done" << endl;
#ifdef USE_OPENMP
		if (show_wtime) {
			mytime=omp_get_wtime() - mytime0;
			if (mpi_id==0) cout << "Wall time for creating grid: " << mytime << endl;
		}
#endif
	}

	return true;
}

void QLens::find_automatic_grid_position_and_size(double *zfacs)
{
	if (autogrid_before_grid_creation) autogrid();
	else {
		if (autocenter==true) {
			lens_list[primary_lens_number]->get_center_coords(grid_xcenter,grid_ycenter);
		}
		if (auto_gridsize_from_einstein_radius==true) {
			double re_major, reav;
			re_major = einstein_radius_of_primary_lens(zfacs[lens_redshift_idx[primary_lens_number]],reav);
			if (re_major != 0.0) {
				double rmax = autogrid_frac*re_major;
				grid_xlength = 2*rmax;
				grid_ylength = 2*rmax;
				cc_rmax = rmax;
			}
		}
	}
}

void QLens::set_primary_lens()
{
	double re, re_avg, largest_einstein_radius = 0;
	int i;
	for (i=0; i < nlens; i++) {
		if (reference_zfactors[lens_redshift_idx[i]] != 0.0) {
			lens_list[i]->get_einstein_radius(re,re_avg,reference_zfactors[lens_redshift_idx[i]]);
			if (re > largest_einstein_radius) {
				largest_einstein_radius = re;
				primary_lens_number = i;
			}
		}
	}
}

void QLens::find_effective_lens_centers_and_einstein_radii(lensvector *centers, double *einstein_radii, int& i_primary, double *zfacs, double **betafacs, bool verbal)
{
	double zlprim, zlsub, re_avg;
	double largest_einstein_radius = 0;
	int i;
	i_primary = 0;
	for (i=0; i < nlens; i++) {
		if (zfacs[lens_redshift_idx[i]] != 0.0) {
			lens_list[i]->get_einstein_radius(einstein_radii[i],re_avg,zfacs[lens_redshift_idx[i]]);
			if (einstein_radii[i] > largest_einstein_radius) {
				largest_einstein_radius = einstein_radii[i];
				zlprim = lens_list[i]->zlens;
				i_primary = i;
			}
		}
	}
	if (largest_einstein_radius==0) {
		if ((mpi_id==0) and (verbal)) warn("could not find primary lens; Einstein radii all returned zero, setting primary to lens 0");
		zlprim = lens_list[0]->zlens;
		i_primary = 0;
	}

	for (i=0; i < nlens; i++) {
		if (zfacs[lens_redshift_idx[i]] != 0.0) {
			zlsub = lens_list[i]->zlens;
			if ((zlsub > zlprim) and (include_recursive_lensing)) {
				if ((lens_list[i]->get_specific_parameter("xc_l",centers[i][0])==false) or (lens_list[i]->get_specific_parameter("yc_l",centers[i][1])==false)) {
					if (find_lensed_position_of_background_perturber(verbal,i,centers[i],zfacs,betafacs)==false) {
						if (verbal) warn("cannot find lensed position of background perturber");
						lens_list[i]->get_center_coords(centers[i][0],centers[i][1]);
					}
				}
			} else {
				lens_list[i]->get_center_coords(centers[i][0],centers[i][1]);
			}
		}
	}
}

void QLens::subgrid_around_perturber_galaxies(lensvector *centers, double *einstein_radii, const int ihost, double *zfacs, double **betafacs, const int redshift_index)
{
	if (grid==NULL) {
		if (create_grid(false,zfacs,betafacs)==false) die("Could not create recursive grid");
	}
	if (nlens==0) { warn(warnings,"No galaxies in lens lens_list"); return; }
	double largest_einstein_radius, xch, ych;
	xch = centers[ihost][0];
	ych = centers[ihost][1];
	largest_einstein_radius = einstein_radii[ihost];

	double xc,yc;
	lensvector center;
	int parity, n_perturbers=0;
	double *kappas = new double[nlens];
	double *parities = new double[nlens];
	bool *exclude = new bool[nlens];
	bool *include_as_primary_perturber = new bool[nlens];
	bool *included_as_secondary_perturber = new bool[nlens];
	for (int i=0; i < nlens; i++) {
		include_as_primary_perturber[i] = false;
		included_as_secondary_perturber[i] = false;
		exclude[i] = false;
	}
	vector<int> excluded;
	int i,j,k;
	bool within_grid;
	double grid_xmin, grid_xmax, grid_ymin, grid_ymax;
	grid_xmin = grid_xcenter - grid_xlength/2;
	grid_xmax = grid_xcenter + grid_xlength/2;
	grid_ymin = grid_ycenter - grid_ylength/2;
	grid_ymax = grid_ycenter + grid_ylength/2;
	//for (i=0; i < nlens; i++) {
				//if ((i==primary_lens_number) or ((centers[i][0]==xch) and (centers[i][1]==ych)) or ((!use_perturber_flags) and (einstein_radii[i] >= 0) and (einstein_radii[i] >= perturber_einstein_radius_fraction*largest_einstein_radius))) exclude[i] = false;
//
	//}
	for (i=0; i < nlens; i++) {
		if (!included_as_secondary_perturber[i]) {
			excluded.clear();
			within_grid = false;
			xc = centers[i][0];
			yc = centers[i][1];
			if ((xc >= grid_xmin) and (xc <= grid_xmax) and (yc >= grid_ymin) and (yc <= grid_ymax)) within_grid = true;
			if (zfacs[lens_redshift_idx[i]] != 0.0) {
				// lenses with Einstein radii < some fraction of the largest Einstein radius, and not co-centered with the largest lens, are considered perturbers.

				if ((((!use_perturber_flags) and (lens_list[i]->has_kapavg_profile()) and (einstein_radii[i] < perturber_einstein_radius_fraction*largest_einstein_radius)) or ((use_perturber_flags) and (lens_list[i]->perturber==true))) and (lens_list[i]->has_kapavg_profile()) and (within_grid) and (i != primary_lens_number)) {
					if ((xc != xch) or (yc != ych)) {
						center[0]=xc;
						center[1]=yc;
						exclude[i] = true;
						for (k=i+1; k < nlens; k++) {
							if ((centers[k][0]==xc) and (centers[k][1]==yc)) {
								exclude[k] = true;
								excluded.push_back(k);
							}
						}
						kappas[i] = kappa_exclude(center,exclude,zfacs,betafacs);
						parities[i] = sign(magnification_exclude(center,exclude,zfacs,betafacs)); // use the parity to help determine approx. size of critical curves
						// galaxies in positive-parity regions where kappa > 1 will form no critical curves, so don't subgrid around these
						exclude[i] = false;
						for (k=0; k < excluded.size(); k++) exclude[excluded[k]] = false; // reset the exclude flags
						if ((parities[i]==1) and (kappas[i] >= 1.0)) continue;
						else {
							n_perturbers++;
							include_as_primary_perturber[i] = true;
							for (k=0; k < excluded.size(); k++) included_as_secondary_perturber[excluded[k]] = true; // reset the exclude flags
						}
					}
				}
			}
		}
	}
	lensvector *galcenter = new lensvector[n_perturbers];
	bool *subgrid = new bool[n_perturbers];
	double *subgrid_radius = new double[n_perturbers];
	double *min_galsubgrid_cellsize = new double[n_perturbers];
	
	for (j=0; j < n_perturbers; j++) subgrid[j] = false;
	double rmax, kappa_at_center;
	j=0;
	for (i=0; i < nlens; i++) {
		if (include_as_primary_perturber[i]) {
			excluded.clear();
			within_grid = false;
			xc = centers[i][0];
			yc = centers[i][1];
			if ((xc >= grid_xmin) and (xc <= grid_xmax) and (yc >= grid_ymin) and (yc <= grid_ymax)) within_grid = true;
			if (zfacs[lens_redshift_idx[i]] != 0.0) {
				//cout << "Perturber (lens " << i << ") at " << xc << " " << yc << endl;
				// lenses co-centered with the primary lens, no matter how small, are not considered perturbers unless flagged specifically
				kappa_at_center = kappas[i];
				parity = parities[i]; // use the parity to help determine approx. size of critical curves

				// galaxies in positive-parity regions where kappa > 1 will form no critical curves, so don't subgrid around these
				galcenter[j][0]=xc;
				galcenter[j][1]=yc;

				exclude[i] = true;
				for (k=i+1; k < nlens; k++) {
					if ((centers[k][0]==xc) and (centers[k][1]==yc)) {
						exclude[k] = true;
						excluded.push_back(k);
					}
				}
				if (calculate_perturber_subgridding_scale(i,exclude,ihost,false,centers[i],rmax,zfacs,betafacs)==false) {
					warn("Satellite subgridding failed (NaN shear calculated); this may be because two or more subhalos are at the same position");
					delete[] subgrid;
					delete[] kappas;
					delete[] exclude;
					delete[] include_as_primary_perturber;
					delete[] included_as_secondary_perturber;
					delete[] parities;
					delete[] galcenter;
					delete[] subgrid_radius;
					delete[] min_galsubgrid_cellsize;
					return;
				}
				exclude[i] = false;
				for (k=0; k < excluded.size(); k++) exclude[excluded[k]] = false; // reset the exclude flags

				subgrid_radius[j] = galsubgrid_radius_fraction*rmax;
				min_galsubgrid_cellsize[j] = SQR(galsubgrid_min_cellsize_fraction*rmax);
				if (rmax > 0) subgrid[j] = true;
				//cout << "Nj=" << j << " i=" << i << endl;
				j++;
			}
		}
	}
	if ((subgrid_only_near_data_images) and (ptsrc_redshift_groups.size() > 0)) {
		int zindx = redshift_index;
		if (zindx==-1) zindx = 0;
		int k;
		double distsqr, min_distsqr;
		for (j=0; j < n_perturbers; j++) {
			min_distsqr = 1e30;
			for (i=ptsrc_redshift_groups[zindx]; i < ptsrc_redshift_groups[zindx+1]; i++) {
				for (k=0; k < point_image_data[i].n_images; k++) {
					distsqr = SQR(point_image_data[i].pos[k][0] - galcenter[j][0]) + SQR(point_image_data[i].pos[k][1] - galcenter[j][1]);
					if (distsqr < min_distsqr) min_distsqr = distsqr;
				}
			}
			if (min_distsqr > SQR(subgrid_radius[j])) subgrid[j] = false;
		}
	}

	if (n_perturbers > 0)
		grid->subgrid_around_galaxies(galcenter,n_perturbers,subgrid_radius,min_galsubgrid_cellsize,galsubgrid_cc_splittings,subgrid);

	delete[] subgrid;
	delete[] kappas;
	delete[] exclude;
	delete[] include_as_primary_perturber;
	delete[] included_as_secondary_perturber;

	delete[] parities;
	delete[] galcenter;
	delete[] subgrid_radius;
	delete[] min_galsubgrid_cellsize;
}

bool QLens::calculate_perturber_subgridding_scale(int lens_number, bool* perturber_list, int host_lens_number, bool verbose, lensvector& center, double& rmax_numerical, double *zfacs, double **betafacs)
{
	perturber_lens_number = lens_number;
	linked_perturber_list = perturber_list;
	subgridding_zfacs = zfacs;
	subgridding_betafacs = betafacs;
	perturber_center[0]=center[0]; perturber_center[1]=center[1];

	double zlsub, zlprim;
	zlsub = lens_list[perturber_lens_number]->zlens;
	zlprim = lens_list[0]->zlens;

	double dum, b;
	lens_list[host_lens_number]->get_einstein_radius(dum,b,zfacs[lens_redshift_idx[host_lens_number]]);

	double shear_angle, shear_tot;
	shear_exclude(perturber_center,shear_tot,shear_angle,linked_perturber_list,zfacs,betafacs);
	if (shear_angle*0.0 != 0.0) return false;
	theta_shear = degrees_to_radians(shear_angle);
	theta_shear -= M_PI/2.0;

	double (Brent::*dthetac_eq)(const double);
	dthetac_eq = static_cast<double (Brent::*)(const double)> (&QLens::galaxy_subgridding_scale_equation);
	static const double rmin = 1e-6;
	double rmax_precision = 0.3*sqrt(min_cell_area);
	double bound = 0.4*b;

	bool found_rmax1, found_rmax2;
	double rmax1, rmax2;
	double rmax_pos, rmax_pos_center, rmax_neg, rmax_pos_noperturb;

	subgridding_include_perturber = true;
	subgridding_parity_at_center = 1;
	found_rmax1 = BrentsMethod(dthetac_eq, rmax1, rmin, bound, rmax_precision);
	found_rmax2 = BrentsMethod(dthetac_eq, rmax2, -bound, rmin, rmax_precision);
	if ((found_rmax1) and (found_rmax2)) rmax_pos = dmax(rmax1,rmax2);
	else if (found_rmax1) rmax_pos = rmax1;
	else if (found_rmax2) rmax_pos = rmax2;
	else rmax_pos = 0;

	// Now we compare to where the critical curve is *without* a perturber; if the difference is not great compared to the
	// distance of the perturber from the critical curve, then we don't subgrid around this region (although we might still
	// subgrid if there is a smaller radial critical curve produced, which would give a nonzero rmax_neg).
	subgridding_include_perturber = false;
	found_rmax1 = BrentsMethod(dthetac_eq, rmax1, rmin, bound, rmax_precision);
	found_rmax2 = BrentsMethod(dthetac_eq, rmax2, -bound, -rmin, rmax_precision);
	if ((found_rmax1) and (found_rmax2)) rmax_pos_noperturb = dmax(rmax1,rmax2);
	else if (found_rmax1) rmax_pos_noperturb = rmax1;
	else if (found_rmax2) rmax_pos_noperturb = rmax2;
	else rmax_pos_noperturb = 0;
	if (rmax_pos != 0) {
		double rmax_ratio = abs((rmax_pos-rmax_pos_noperturb)/rmax_pos);
		if (rmax_ratio > 0.5) rmax_pos = abs(rmax_pos);
		else rmax_pos = 0;
	}

	subgridding_parity_at_center = -1;
	subgridding_include_perturber = true;
	found_rmax1 = BrentsMethod(dthetac_eq, rmax1, rmin, bound, rmax_precision);
	found_rmax2 = BrentsMethod(dthetac_eq, rmax2, -bound, -rmin, rmax_precision);
	rmax2 = abs(rmax2);
	if ((found_rmax1) and (found_rmax2)) rmax_neg = dmax(rmax1,rmax2);
	else if (found_rmax1) rmax_neg = rmax1;
	else if (found_rmax2) rmax_neg = rmax2;
	else rmax_neg = 0;
	//cout << "rmax_pos=" << rmax_pos << ", rmax_neg=" << rmax_neg << endl;

	rmax_numerical = dmax(rmax_neg,rmax_pos);
	if (zlsub > zlprim) rmax_numerical *= 1.1; // in this regime, rmax is often a bit underestimated, so this helps counteract that
		//cout << "RMAX: " << rmax_numerical << endl;
	//if (rmax_numerical==0.0) warn("could not find rmax");
	return true;
}

double QLens::galaxy_subgridding_scale_equation(const double r)
{
	double kappa0, shear0, lambda0, shear_angle, perturber_avg_kappa;
	lensvector x;
	x[0] = perturber_center[0] + r*cos(theta_shear);
	x[1] = perturber_center[1] + r*sin(theta_shear);
	if (subgridding_parity_at_center < 0) {
		kappa0 = kappa_exclude(perturber_center,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
		shear_exclude(perturber_center,shear0,shear_angle,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
		lambda0 = 1 - kappa0 + shear0;
	} else {
		kappa0 = kappa_exclude(x,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
		shear_exclude(x,shear0,shear_angle,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
		lambda0 = 1 - kappa0 - shear0;
	}
	double r_eff = r;

	perturber_avg_kappa = 0;
	if (subgridding_include_perturber) {
		double zlsub, zlprim;
		zlsub = lens_list[perturber_lens_number]->zlens;
		zlprim = lens_list[0]->zlens;

		if (zlsub > zlprim) {
			lensvector xp, xpc;
			lens_list[perturber_lens_number]->get_center_coords(xpc[0],xpc[1]);
			double zsrc0 = source_redshift;
			//cout << "ZLSUB ZSRC: " << zlsub << " " << zsrc0 << endl;
			set_source_redshift(zlsub);
			lensvector alpha;
			// BUG!!!!!!! subgridding_zfacs is not updated by set_source_redshift
			deflection(x,alpha,subgridding_zfacs,subgridding_betafacs);
			set_source_redshift(zsrc0);
			xp[0] = x[0] - alpha[0];
			xp[1] = x[1] - alpha[1];
			r_eff = sqrt(SQR(xp[0]-xpc[0])+SQR(xp[1]-xpc[1]));
		} else {
			r_eff = r;
		}
		for (int i=0; i < nlens; i++) {
			if (linked_perturber_list[i]) perturber_avg_kappa += subgridding_zfacs[lens_redshift_idx[i]]*lens_list[i]->kappa_avg_r(r_eff);
		}
		if (subgridding_parity_at_center > 0) {
			if (zlsub < zlprim) {
				int i1,i2;
				i1 = lens_redshift_idx[primary_lens_number];
				i2 = lens_redshift_idx[perturber_lens_number];
				double beta = subgridding_betafacs[i1-1][i2];
				double dr = 1e-5;
				double kappa0_p, shear0_p;
				lensvector xp;
				xp[0] = perturber_center[0] + (r+dr)*cos(theta_shear);
				xp[1] = perturber_center[1] + (r+dr)*sin(theta_shear);
				kappa0_p = kappa_exclude(xp,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
				shear_exclude(xp,shear0_p,shear_angle,linked_perturber_list,subgridding_zfacs,subgridding_betafacs);
				double k0deriv = (kappa0_p+shear0_p-kappa0-shear0)/dr;
				double fac = 1 - beta*(kappa0 + shear0 + r*k0deriv);
				perturber_avg_kappa *= 1 - beta*(kappa0 + shear0 + r*k0deriv);
			} else if (zlsub > zlprim) {
				int i1,i2;
				i1 = lens_redshift_idx[primary_lens_number];
				i2 = lens_redshift_idx[perturber_lens_number];
				double beta = subgridding_betafacs[i2-1][i1];
				perturber_avg_kappa *= 1 - beta*(kappa0 + shear0);
			}
		}
	}

	return (lambda0 - perturber_avg_kappa);
}

void QLens::plot_shear_field(double xmin, double xmax, int nx, double ymin, double ymax, int ny, const string filename)
{
	int i, j, k;
	double x, y;
	double xstep = (xmax-xmin)/(nx-1);
	double ystep = (ymax-ymin)/(ny-1);
	double scale = 0.3*dmin(xstep,ystep);
	int compass_steps = 2;
	double compass_step = scale / (compass_steps-1);
	lensvector pos;
	double kapval,shearval,shear_angle,xp,yp,t;
	ofstream sout;
	open_output_file(sout,filename);
	for (i=0, x=xmin; i < nx; i++, x += xstep) {
		for (j=0, y=ymin; j < ny; j++, y += ystep) {
			pos[0]=x; pos[1]=y;
			shear(pos,shearval,shear_angle,0,reference_zfactors,default_zsrc_beta_factors);
			kapval = kappa(pos,reference_zfactors,default_zsrc_beta_factors);
			shearval /= (1-kapval); // reduced shear
			shear_angle *= M_PI/180.0;
			for (k=-compass_steps+1; k < compass_steps; k++)
			{
				t = k*compass_step;
				xp = x + t*cos(shear_angle);
				yp = y + t*sin(shear_angle);
				sout << xp << " " << yp << endl;
			}
			sout << endl;
		}
	}
	sout.close();
}

void QLens::plot_weak_lensing_shear_data(const bool include_model_shear, const string filename)
{
	double *zfacs = new double[n_lens_redshifts];
	int i, j, k;
	double x, y;
	double shearval,shear_angle,shear1,shear2,xp,yp,t;
	double model_shear1,model_shear2,model_shearval,model_shear_angle,xmp,ymp,tm;
	double xmin=1e30, xmax=-1e30, ymin=1e30, ymax=-1e30, min_shear=1e30, max_shear=-1e30;
	for (i=0; i < weak_lensing_data.n_sources; i++) {
		shear1 = weak_lensing_data.reduced_shear1[i];
		shear2 = weak_lensing_data.reduced_shear2[i];
		shearval = sqrt(shear1*shear1+shear2*shear2);
		x = weak_lensing_data.pos[i][0];
		y = weak_lensing_data.pos[i][1];
		if (x < xmin) xmin = x;
		if (x > xmax) xmax = x;
		if (y < ymin) ymin = y;
		if (y > ymax) ymax = y;
		if (shearval < min_shear) min_shear = shearval;
		if (shearval > max_shear) max_shear = shearval;
	}
	int nsteps_approx = (int) sqrt(weak_lensing_data.n_sources);
	double xstep = (xmax-xmin)/nsteps_approx;
	double ystep = (ymax-ymin)/nsteps_approx;
	double scale_factor = 1.7; // slightly enlarges the "arrows" so they're easier to see on the screen
	double scale = scale_factor*dmin(xstep,ystep)/2.0;
	double zsrc;
	int compass_steps = 2;
	double compass_step, model_compass_step;

	ofstream sout;
	open_output_file(sout,filename);
	for (i=0; i < weak_lensing_data.n_sources; i++) {
		x = weak_lensing_data.pos[i][0];
		y = weak_lensing_data.pos[i][1];
		zsrc = weak_lensing_data.zsrc[i];
		for (int i=0; i < n_lens_redshifts; i++) {
			zfacs[i] = cosmo->kappa_ratio(lens_redshifts[i],zsrc,reference_source_redshift);
		}
		shear1 = weak_lensing_data.reduced_shear1[i];
		shear2 = weak_lensing_data.reduced_shear2[i];
		shearval = sqrt(shear1*shear1+shear2*shear2);
		shear_angle = atan(abs(shear2/shear1));
		compass_step = scale*(shearval/max_shear) / (compass_steps-1);
		if (shear1 < 0) {
			if (shear2 < 0)
				shear_angle = shear_angle - M_PI;
			else
				shear_angle = M_PI - shear_angle;
		} else if (shear2 < 0) {
			shear_angle = -shear_angle;
		}
		shear_angle *= 0.5;

		if (include_model_shear) {
			lensvector xvec(x,y);
			reduced_shear_components(xvec,model_shear1,model_shear2,0,zfacs);
			model_shearval = sqrt(model_shear1*model_shear1 + model_shear2*model_shear2);
			model_shear_angle = atan(abs(model_shear2/model_shear1));
			if (model_shear1 < 0) {
				if (model_shear2 < 0)
					model_shear_angle = model_shear_angle - M_PI;
				else
					model_shear_angle = M_PI - model_shear_angle;
			} else if (model_shear2 < 0) {
				model_shear_angle = -model_shear_angle;
			}
			model_shear_angle *= 0.5;
			model_compass_step = scale*(model_shearval/max_shear) / (compass_steps-1);
		}

		for (k=-compass_steps+1; k < compass_steps; k++)
		{
			t = k*compass_step;
			tm = k*model_compass_step;
			xp = x + t*cos(shear_angle);
			yp = y + t*sin(shear_angle);
			if (include_model_shear) {
				xmp = x + tm*cos(model_shear_angle);
				ymp = y + tm*sin(model_shear_angle);
				sout << xp << " " << yp << " " << xmp << " " << ymp << endl;
			} else {
				sout << xp << " " << yp << endl;
			}
		}
		sout << endl;
	}
	sout.close();
	delete[] zfacs;
}

/*
// The following function uses the series expansions derived in Minor et al. 2017, but it's better to simply use a root finder, so
// this approach is deprecated
void QLens::calculate_critical_curve_perturbation_radius(int lens_number, bool verbose, double &rmax, double& mass_enclosed)
{
	// the analytic formulas require a Pseudo-Jaffe or isothermal profile, and they only work for subhalos in the plane of the lens
	// if one of these conditions isn't satisfied, just use the numerical root-finding version instead
	if (((lens_list[lens_number]->get_lenstype()!=dpie_LENS) and (lens_list[lens_number]->get_lenstype()!=sple_LENS)) or (lens_list[lens_number]->zlens != lens_list[0]->zlens))
	{
		double avg_sigma_enclosed;
		calculate_critical_curve_perturbation_radius_numerical(lens_number,verbose,rmax,avg_sigma_enclosed,mass_enclosed);
		return;
	}
	//this assumes the host halo is lens number 0 (and is centered at the origin), and corresponding external shear (if present) is lens number 1
	double xc, yc, b, alpha, bs, rt, dum, q, shear_ext, phi, phi_0, phi_p, theta_s;
	double host_xc, host_yc;
	double reference_zfactor = reference_zfactors[lens_redshift_idx[lens_number]];
	lens_list[lens_number]->get_center_coords(xc,yc);
	lens_list[0]->get_center_coords(host_xc,host_yc);
	theta_s = sqrt(SQR(xc-host_xc) + SQR(yc-host_yc));
	phi = atan(abs((yc-host_yc)/(xc-host_xc)));
	if ((xc-host_xc) < 0) {
		if ((yc-host_yc) < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if ((yc-host_yc) < 0) {
		phi = -phi;
	}

	bool is_pjaffe;
	if (lens_list[lens_number]->get_lenstype()==dpie_LENS) {
		is_pjaffe = true;
		double params[10];
		lens_list[lens_number]->get_parameters(params);
		bs = params[0];
		rt = params[1];
	} else {
		is_pjaffe = false;
		lens_list[lens_number]->get_einstein_radius(dum,bs,reference_zfactor);
	}
	double host_params[10];
	lens_list[0]->get_parameters(host_params);
	alpha = host_params[1];
	lens_list[0]->get_einstein_radius(dum,b,reference_zfactor);
	lens_list[0]->get_q_theta(q,phi_0);
	double gamma = alpha-1;
	double aprime = alpha;

	if (lens_list[1]->get_lenstype()==SHEAR) lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
	else { shear_ext = 0; phi_p=0; }
	if (LensProfile::orient_major_axis_north==true) {
		phi_0 += M_HALFPI;
		phi_p += M_HALFPI;
	}
	double shear_tot, sigma, eta, mu, delta, epsilon, xi, zeta, rmax_analytic, kappa0_at_sub, phi_normal_to_shear;
	sigma = 1.0/sqrt(q*SQR(cos(phi-phi_0)) + SQR(sin(phi-phi_0))/q);
	kappa0_at_sub = 0.5*(1-gamma)*pow(b*sigma/theta_s,1+gamma);

	double dphi, dphi_prime, gg;
	complex<double> hyp, complex_g;
	dphi = phi-phi_0;
	while (dphi <= -M_HALFPI) dphi += M_2PI;
	while (dphi >= M_2PI) dphi -= M_2PI;
	dphi_prime = atan(abs(tan(dphi)/q));
	if (dphi > M_HALFPI) { // dphi_prime must be in the same quadrant as dphi
		if (dphi <= M_PI) dphi_prime = M_PI - dphi_prime;
		else if (dphi <= 1.5*M_PI) dphi_prime += M_PI;
		else if (dphi <= M_2PI) dphi_prime = -dphi_prime;
	} else if (dphi < 0) dphi_prime = -dphi_prime;
	hyp = hyp_2F1(1.0,aprime/2.0,2.0-aprime/2.0,-(1-q)/(1+q)*polar(1.0,2*dphi_prime));
	complex_g = 1.0 - sqrt(q)*(4.0*(1-aprime)/((1+q)*(2-aprime)))*hyp/sigma*polar(1.0,dphi_prime-dphi);
	gg = sqrt(norm(complex_g));
	double gg_q = alpha/(2-alpha);
	if (verbose) cout << "shear/kappa = " << gg << " q=1 version: " << gg_q << endl;

	double cg = kappa0_at_sub*(1+gg);
	shear_tot = sqrt(SQR(gg*kappa0_at_sub) + shear_ext*shear_ext + 2*gg*kappa0_at_sub*shear_ext*cos(2*(phi-phi_p)));
	eta = 1 + gg*kappa0_at_sub - shear_tot;
	if (shear_ext==0) phi_normal_to_shear = phi;
	else phi_normal_to_shear = asin((gg*kappa0_at_sub*sin(2*phi)+shear_ext*sin(2*phi_p))/shear_tot) / 2;

	if (verbose) cout << "phi_normal = " << radians_to_degrees(phi_normal_to_shear) << ", phi = " << radians_to_degrees(phi) << ", dphi=" << radians_to_degrees(phi_normal_to_shear - phi) << endl;
	if (is_pjaffe) {
		double beta, bsq;
		double y=sqrt(theta_s*bs/(aprime*eta));
		if (verbose) cout << "y=" << y << endl;
		beta = rt/y;
		bsq = beta*beta;
		delta = 1 + 2*beta - 2*sqrt(1+bsq) + 1.0/sqrt(1+bsq);
		xi = beta-sqrt(1+bsq)+1.0/sqrt(1+bsq);
		epsilon = -(bs/y)*xi;
	} else {
		delta = 1;
		epsilon = 0;
	}


	double theta_on_cc = b*sigma*pow((1+gg)*(1-gamma)/(2*eta),1.0/(1+gamma));
	double dtheta = theta_s - theta_on_cc;
	//dtheta=0;
	mu = eta - epsilon + cg*gamma;
	zeta = 0.5*(bs*delta - (eta-epsilon-cg)*theta_s);
	rmax_analytic = (sqrt(bs*theta_s*delta*mu+SQR(zeta))+zeta)/mu;

	// Now for eta, we will use the formula for eta on the critical curve for the isothermal case, which works amazingly well
	double eta_on_cc_iso = (1-shear_ext*shear_ext)/(1 + shear_ext*cos(2*(phi-phi_p))); // isothermal
	//double eta_on_cc_not_iso = (1+gg*shear_ext*cos(2*(phi-phi_p)))*(-1 + sqrt(1 + (gg*gg-1)*(1-shear_ext*shear_ext)/SQR(1+gg*shear_ext*cos(2*(phi-phi_p)))))/(gg-1);
	double dtt = dtheta/theta_on_cc;
	dtt = dtt - (aprime+1)*dtt*dtt/2.0;
	mu = (aprime*eta_on_cc_iso)*(1 + (1-aprime)*dtt + (xi/(aprime*eta_on_cc_iso))*sqrt(bs/b));
	zeta = 0.5*(bs*delta - aprime*eta_on_cc_iso*theta_s*dtt - sqrt(b*bs)*(theta_s/b)*xi);
	double rmax_analytic2 = (1.0/mu)*(sqrt(bs*theta_s*delta*mu+SQR(zeta))+zeta);

	// for the approximate solutions on c.c., expanding subhalo deflection around sqrt(b*bs) seems to work better
	if (is_pjaffe) {
		double beta, bsq;
		beta = rt/sqrt(b*bs);
		bsq = beta*beta;
		delta = 1 + 2*beta - 2*sqrt(1+bsq) + 1.0/sqrt(1+bsq);
		xi = beta-sqrt(1+bsq)+1.0/sqrt(1+bsq);
		epsilon = -sqrt(bs/b)*xi;
	}

	// the next approximation assumes the subhalo is located on the (unperturbed) critical curve
	double eta_on_cc, xx, rmax_on_cc, rmax0;
	double lambda = pow(0.5*(1+gg)*(1-gamma)*pow(eta_on_cc_iso,gamma),1.0/(1+gamma));
	xx = sqrt(theta_s/(b*aprime*delta*eta_on_cc_iso));
	rmax_on_cc = delta*xx*(1-xi*xx/2.0+SQR(xx*xi)/8.0)*sqrt(b*bs) + (delta/(2*eta_on_cc_iso*aprime))*(1-xx*xi)*bs;

	// rough form for non-lens modelers to use
	double xxx = xi*sqrt(theta_s/(b*delta*aprime));
	rmax0 = sqrt(delta*theta_s*bs/aprime)*(1-xxx/2.0 + xxx*xxx/8.0) + delta*bs/(2*aprime)*(1-xxx);
	//double mass_rmax = M_PI*bs*(rmax_analytic - sqrt(bs*b + SQR(rmax_analytic)) + sqrt(bs*b))/aprime;

	double shear_angle, rmax_numerical, totshear;
	perturber_lens_number = lens_number;
	perturber_center[0]=xc; perturber_center[1]=yc;
	shear_exclude(perturber_center,totshear,shear_angle,perturber_lens_number,reference_zfactors,default_zsrc_beta_factors);
	theta_shear = degrees_to_radians(shear_angle);
	theta_shear -= M_PI/2.0;
	double (Brent::*dthetac_eq)(const double);
	dthetac_eq = static_cast<double (Brent::*)(const double)> (&QLens::subhalo_perturbation_radius_equation);
	double bound = 2*sqrt(b*bs);
	rmax_numerical = abs(BrentsMethod_Inclusive(dthetac_eq,-bound,bound,1e-5));
	double avg_kappa = reference_zfactors[lens_redshift_idx[perturber_lens_number]]*lens_list[perturber_lens_number]->kappa_avg_r(rmax_numerical);
	double zlsub, zlprim;
	zlsub = lens_list[perturber_lens_number]->zlens;
	zlprim = lens_list[0]->zlens;
	double menc = avg_kappa*M_PI*SQR(rmax_numerical)*cosmo->sigma_crit_kpc(zlsub,reference_source_redshift);

	if (verbose) {
		cout << "direction of maximum warping = " << radians_to_degrees(theta_shear) << endl;
		cout << "theta_c=" << theta_on_cc << endl;
		cout << "dtheta/theta_c=" << (theta_s-theta_on_cc)/theta_on_cc << endl;
		//cout << "mu=" << mu << endl;
		//cout << "zeta=" << zeta << " zeta2=" << zeta2 << endl;
		//cout << "cosp = " << cos(2*(phi-phi_p)) << endl;
		//cout << "cfactor = " << cg*(1+gamma) << endl;
		cout << "eta = " << eta << ", eta_on_cc = " << eta_on_cc << ", eta_on_cc_iso = " << eta_on_cc_iso << endl;
		cout << "sigma = " << sigma << endl;
		cout << "lambda = " << lambda << endl;
		cout << "zeta = " << zeta << endl;
		cout << "theta_s  = " << theta_s << endl;
		cout << "theta_s (on c.c., approx) = " << theta_on_cc << endl << endl;
		cout << "rmax_numerical = " << rmax_numerical << endl;
		cout << "rmax_analytic = " << rmax_analytic << " (fractional error = " << (rmax_analytic-rmax_numerical)/rmax_numerical << ")" << endl;
		cout << "rmax_analytic_approx = " << rmax_analytic2 << " (fractional error = " << (rmax_analytic2-rmax_numerical)/rmax_numerical << ")" << endl;
		cout << "rmax (if on c.c.) = " << rmax_on_cc << " (fractional error = " << (rmax_on_cc-rmax_numerical)/rmax_numerical << ")" << endl;
		cout << "rmax (rough, if on c.c.) = " << rmax0 << " (fractional error = " << (rmax0-rmax_numerical)/rmax_numerical << ")" << endl;
		cout << "avg_kappa/alpha = " << avg_kappa/alpha << endl;
		cout << "mass_enclosed/alpha = " << menc/alpha << endl;
		cout << "mass_enclosed/alpha/eta = " << menc/alpha/eta << endl;
	}
	mass_enclosed = menc/alpha;
	rmax = rmax_analytic;
}
*/

bool QLens::find_lensed_position_of_background_perturber(bool verbal, int lens_number, lensvector& pos, double *zfacs, double **betafacs)
{
	double zlsub;
	zlsub = lens_list[lens_number]->zlens;
	lensvector perturber_center;
	lens_list[lens_number]->get_center_coords(perturber_center[0],perturber_center[1]);
	double zsrc0 = source_redshift;
	bool subgrid_setting = subgrid_around_perturbers;
	find_automatic_grid_position_and_size(zfacs);
	bool auto0 = auto_gridsize_from_einstein_radius;
	bool auto1 = autogrid_before_grid_creation;
	subgrid_around_perturbers = false;
	auto_gridsize_from_einstein_radius = false;
	autogrid_before_grid_creation = false;
	set_source_redshift(zlsub);
	create_grid(false,zfacs,betafacs);
	int n_images, img_i;
	image *img = get_images(perturber_center, n_images, false);
	if (n_images == 0) {
		reset_grid();
		set_source_redshift(zsrc0);
		return false;
	}
	img_i = 0;
	if (n_images > 1) {
		if ((mpi_id==0) and (verbal)) {
			warn("Well this is interesting. Perturber maps to more than one place in the primary lens plane! Using image furthest from primary lens center");
			cout << "Positions of lensed perturber:\n";
		}
		double rsq, rsqmax=-1e30;
		double xc0, yc0;
		lens_list[primary_lens_number]->get_center_coords(xc0,yc0);
		for (int ii=0; ii < n_images; ii++) {
			rsq = SQR(img[ii].pos[0]-xc0) + SQR(img[ii].pos[1]-yc0);
			if (rsq > rsqmax) {
				rsqmax = rsq;
				img_i = ii;
			}
			if ((mpi_id==0) and (verbal)) cout << img[ii].pos[0] << " " << img[ii].pos[1] << endl;
		}
	}
	set_source_redshift(zsrc0);
	subgrid_around_perturbers = subgrid_setting;
	auto_gridsize_from_einstein_radius = false;
	autogrid_before_grid_creation = false;
	reset_grid();
	pos[0] = img[img_i].pos[0];
	pos[1] = img[img_i].pos[1];
	return true;
}

bool QLens::calculate_critical_curve_perturbation_radius_numerical(int lens_number, bool verbal, double& rmax_numerical, double& avg_sigma_enclosed, double& mass_enclosed, double& rmax_perturber_z, double &avgkap_scaled_to_primary_lensplane, bool subtract_unperturbed)
{
	perturber_lens_number = lens_number;
	bool *perturber_list = new bool[nlens];
	for (int i=0; i < nlens; i++) perturber_list[i] = false;
	perturber_list[lens_number] = true;
	linked_perturber_list = perturber_list;
	double xc, yc, host_xc, host_yc, b, dum, alpha, shear_ext, phi, phi_p;

	double zlsub, zlprim;
	zlsub = lens_list[perturber_lens_number]->zlens;
	zlprim = lens_list[primary_lens_number]->zlens;

	double reference_zfactor = reference_zfactors[lens_redshift_idx[perturber_lens_number]];
	if (zlsub > zlprim) {
		if ((lens_list[lens_number]->get_specific_parameter("xc_l",perturber_center[0])==false) or (lens_list[lens_number]->get_specific_parameter("yc_l",perturber_center[1])==false)) {
			if (find_lensed_position_of_background_perturber(verbal,lens_number,perturber_center,reference_zfactors,default_zsrc_beta_factors)==false) {
				warn("could not find lensed position of background perturber");
				delete[] perturber_list;
				return false;
			}
		}
		xc = perturber_center[0];
		yc = perturber_center[1];
		if ((mpi_id==0) and (verbal)) cout << "Perturber located at (" << xc << "," << yc << ") in primary lens plane\n";
	} else {
		lens_list[perturber_lens_number]->get_center_coords(xc,yc);
		perturber_center[0]=xc; perturber_center[1]=yc;
	}

	lens_list[primary_lens_number]->get_center_coords(host_xc,host_yc);
	phi = atan(abs((yc-host_yc)/(xc-host_xc)));
	if ((xc-host_xc) < 0) {
		if ((yc-host_yc) < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if ((yc-host_yc) < 0) {
		phi = -phi;
	}

	if ((primary_lens_number < (nlens-1)) and (lens_list[primary_lens_number+1]->get_lenstype()==SHEAR)) lens_list[primary_lens_number+1]->get_q_theta(shear_ext,phi_p); // assumes that if there is external shear present, it comes after the primary lens in the lens list
	else { shear_ext = 0; phi_p=0; }
	if (LensProfile::orient_major_axis_north==true) {
		phi_p += M_HALFPI;
	}

	double shear_angle, shear_tot;
	shear_exclude(perturber_center,shear_tot,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
	if (shear_angle*0.0 != 0.0) {
		warn("could not calculate shear at position of perturber");
		rmax_numerical = 0.0;
		mass_enclosed = 0.0;
		delete[] perturber_list;
		return false;
	}

	lens_list[primary_lens_number]->get_einstein_radius(dum,b,reference_zfactors[lens_redshift_idx[primary_lens_number]]);

	theta_shear = degrees_to_radians(shear_angle);
	theta_shear -= M_PI/2.0;
	double (Brent::*dthetac_eq)(const double);
	dthetac_eq = static_cast<double (Brent::*)(const double)> (&QLens::subhalo_perturbation_radius_equation);
	double bound = 0.6*b;
	rmax_numerical = BrentsMethod_Inclusive(dthetac_eq,-bound,bound,1e-5,verbal);
	//if ((rmax_numerical==bound) or (rmax_numerical==-bound)) {
		//rmax_numerical = 0.0; // subhalo too far from critical curve to cause a meaningful "local" perturbation
		//mass_enclosed = 0.0;
		//delete[] perturber_list;
		//return true;
	//}
	if (zlsub > zlprim) {
		lensvector x;
		x[0] = perturber_center[0] + rmax_numerical*cos(theta_shear);
		x[1] = perturber_center[1] + rmax_numerical*sin(theta_shear);
		lensvector xp, xpc;
		lens_list[perturber_lens_number]->get_center_coords(xpc[0],xpc[1]);
		double zsrc0 = source_redshift;
		set_source_redshift(zlsub);
		lensvector defp;
		deflection(x,defp,reference_zfactors,default_zsrc_beta_factors);
		set_source_redshift(zsrc0);
		xp[0] = x[0] - defp[0];
		xp[1] = x[1] - defp[1];
		rmax_perturber_z = sqrt(SQR(xp[0]-xpc[0])+SQR(xp[1]-xpc[1]));
	} else rmax_perturber_z = abs(rmax_numerical);

	double avg_kappa = reference_zfactors[lens_redshift_idx[perturber_lens_number]]*lens_list[perturber_lens_number]->kappa_avg_r(rmax_perturber_z);

	if (lens_list[primary_lens_number]->lenstype==sple_LENS) {
		double host_params[10];
		lens_list[primary_lens_number]->get_parameters(host_params);
		alpha = host_params[1];
	} else {
		alpha = 1.0;
	}

	double subhalo_rc = sqrt(SQR(perturber_center[0]-host_xc)+SQR(perturber_center[1]-host_yc));

	double theta_c_unperturbed, rmax_relative = rmax_numerical;
	if ((subtract_unperturbed) or (verbal)) {
		double (Brent::*dthetac_eq_nosub)(const double);
		dthetac_eq_nosub = static_cast<double (Brent::*)(const double)> (&QLens::perturbation_radius_equation_nosub);
		theta_c_unperturbed = BrentsMethod_Inclusive(dthetac_eq_nosub,-bound,bound,1e-5,verbal);
		rmax_relative = abs(rmax_numerical-theta_c_unperturbed);
		if (subtract_unperturbed) rmax_numerical = rmax_relative;
		theta_c_unperturbed += subhalo_rc; // now it's actually theta_c relative to the primary lens center
	}
	//double delta_s_over_thetac = subhalo_rc/theta_c_unperturbed - 1;
	//double r_over_rc = abs(rmax_numerical)/(subhalo_rc + abs(rmax_numerical));
	//double ktilde_approx = 1 - alpha*(delta_s_over_thetac + r_over_rc);
	//double ktilde_approx2 = 1 - alpha*(r_over_rc);
		//double blergh2_approx = 1 - alpha*(delta_s_over_thetac + 2*r_over_rc);

	double kpc_to_arcsec_sub = 206.264806/cosmo->angular_diameter_distance(zlsub);
	// the following quantities are scaled by 1/alpha
	avg_sigma_enclosed = avg_kappa*cosmo->sigma_crit_kpc(zlsub,reference_source_redshift);
	mass_enclosed = avg_sigma_enclosed*M_PI*SQR(rmax_numerical/kpc_to_arcsec_sub);

	double menc_scaled_to_primary_lensplane = 0;
	avgkap_scaled_to_primary_lensplane = 0;
	double k0deriv=0;
	//double avgkap_scaled2 = 0;
	double kappa0=0;
	if (include_recursive_lensing) {
		lensvector x;
		x[0] = perturber_center[0] + rmax_numerical*cos(theta_shear);
		x[1] = perturber_center[1] + rmax_numerical*sin(theta_shear);
		kappa0 = kappa_exclude(x,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		shear_exclude(x,shear_tot,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		if (zlsub < zlprim) {
			int i1,i2;
			i1 = lens_redshift_idx[primary_lens_number];
			i2 = lens_redshift_idx[perturber_lens_number];
			double beta = default_zsrc_beta_factors[i1-1][i2];
			double dr = 1e-5;
			double kappa0_p, shear_tot_p;
			lensvector xp;
			xp[0] = perturber_center[0] + (rmax_numerical+dr)*cos(theta_shear);
			xp[1] = perturber_center[1] + (rmax_numerical+dr)*sin(theta_shear);
			kappa0_p = kappa_exclude(xp,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(xp,shear_tot_p,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			double kappa0_m, shear_tot_m;
			lensvector xm;
			xm[0] = perturber_center[0] + (rmax_numerical-dr)*cos(theta_shear);
			xm[1] = perturber_center[1] + (rmax_numerical-dr)*sin(theta_shear);
			kappa0_m = kappa_exclude(xm,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(xm,shear_tot_m,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
			k0deriv = (kappa0_p+shear_tot_p-kappa0_m-shear_tot_m)/(2*dr);
			double mass_scale_factor = (cosmo->sigma_crit_kpc(zlprim,reference_source_redshift) / cosmo->sigma_crit_kpc(zlsub,reference_source_redshift))*SQR(rmax_numerical/rmax_perturber_z)*(1 - beta*(kappa0 + shear_tot + rmax_numerical*k0deriv));
			//double fac1 = (cosmo->sigma_crit_kpc(zlprim,reference_source_redshift) / cosmo->sigma_crit_kpc(zlsub,reference_source_redshift));
			//double fac2 = (1 - beta*(kappa0 + shear_tot + rmax_numerical*k0deriv));
			//cout << fac1 << " " << fac2 << " " << mass_scale_factor << endl;
			menc_scaled_to_primary_lensplane = mass_enclosed*mass_scale_factor;
			avgkap_scaled_to_primary_lensplane = avg_kappa*(1-beta*(kappa0+shear_tot+abs(rmax_numerical)*k0deriv));

			//double ktilde = kappa0+shear_tot;
			//double blergh = abs(rmax_numerical)*k0deriv;
			//double blergh2 = ktilde + blergh;
			//double blergh2_approx0 = 1 - alpha*(delta_s_over_thetac + 2*r_over_rc);
			//double blergh2_approx = 1 - alpha*(2*r_over_rc);
			//avgkap_scaled2 = avg_kappa*(1-beta*blergh2_approx);
			//cout << "r*k0deriv=" << blergh << endl;
			//cout << "blergh=" << blergh2 << " approx=" << blergh2_approx << " better_approx=" << blergh2_approx0 << endl;
		} else if (zlsub > zlprim) {
			//double kappa0, shear_tot, shear_angle;
			int i1,i2;
			i1 = lens_redshift_idx[primary_lens_number];
			i2 = lens_redshift_idx[perturber_lens_number];
			double beta = default_zsrc_beta_factors[i2-1][i1];
			double mass_scale_factor = (cosmo->sigma_crit_kpc(zlprim,reference_source_redshift) / cosmo->sigma_crit_kpc(zlsub,reference_source_redshift))*(1 - beta*(kappa0 + shear_tot));
			menc_scaled_to_primary_lensplane = mass_enclosed*mass_scale_factor;
			avgkap_scaled_to_primary_lensplane = avg_kappa*(1-beta*(kappa0+shear_tot));

			//double ktilde = kappa0+shear_tot;
			//double blergh2 = ktilde;
			//double blergh2_approx = 1 - alpha*(r_over_rc);
			//double blergh2_approx0 = 1 - alpha*(delta_s_over_thetac + r_over_rc);
			//avgkap_scaled2 = avg_kappa*(1-beta*blergh2_approx);
			//cout << "blergh=" << blergh2 << " approx=" << blergh2_approx << " better_approx=" << blergh2_approx0 << endl;
		}
	} else {
		double mass_scale_factor = (cosmo->sigma_crit_kpc(zlprim,reference_source_redshift) / cosmo->sigma_crit_kpc(zlsub,reference_source_redshift))*SQR(rmax_numerical/rmax_perturber_z);
		menc_scaled_to_primary_lensplane = mass_enclosed*mass_scale_factor;
		avgkap_scaled_to_primary_lensplane = avg_kappa;
	}
	//if (mpi_id==0) cout << "CHECK0: " << rmax_numerical << " " << rmax_perturber_z << " " << avg_kappa << " " << avgkap_scaled_to_primary_lensplane << " ... " << kappa0 << " " << shear_tot << " " << k0deriv << endl;

	//double avgkap_check,menc_check;
	//get_perturber_avgkappa_scaled(lens_number,rmax_numerical,avgkap_check,menc_check);
	double rmax_kpc = rmax_numerical/kpc_to_arcsec_sub;

	double gg_q = alpha/(2-alpha);
	//if (verbose) cout << "shear/kappa = " << gg << " q=1 version: " << gg_q << endl;
	double eta = 1 + gg_q*kappa0 - shear_tot;
	cout << "eta=" << eta << endl;
	//if (shear_ext==0) phi_normal_to_shear = phi;
	//else phi_normal_to_shear = asin((gg_q*kappa0_at_sub*sin(2*phi)+shear_ext*sin(2*phi_p))/shear_tot) / 2;


	if ((mpi_id==0) and (verbal)) {
		lensvector x;
		x[0] = perturber_center[0] + rmax_numerical*cos(theta_shear);
		x[1] = perturber_center[1] + rmax_numerical*sin(theta_shear);
		cout << "direction of maximum warping = " << radians_to_degrees(theta_shear) << endl;
		cout << "rmax_numerical = " << rmax_numerical << " (rmax_kpc=" << rmax_kpc << ")" << endl;
		cout << "rmax_relative = " << rmax_relative << endl;
		cout << "subhalo_rc = " << subhalo_rc << " (distance from subhalo to primary lens center)" << endl;
		cout << "theta_c_unperturbed = " << theta_c_unperturbed << endl;
		//cout << "delta: " << delta_s_over_thetac << endl;
		//cout << "r_over_rc: " << r_over_rc << endl;
		//cout << "ktilde: " << ktilde << endl;
		cout << "rmax location: (" << x[0] << "," << x[1] << ")\n";
		if (zlsub > zlprim) cout << "rmax_perturber_z = " << rmax_perturber_z << endl;
		cout << "avg_kappa = " << avg_kappa << endl;
		cout << "avg_kappa/alpha = " << avg_kappa/alpha << endl;
		if (avgkap_scaled_to_primary_lensplane != 0) {
			//double avgkaperr = (avgkap_scaled_to_primary_lensplane-avgkap_scaled2)/avgkap_scaled_to_primary_lensplane;
			cout << "avg_kappa(primary_lens_plane)/alpha = " << avgkap_scaled_to_primary_lensplane/alpha << endl;
			//cout << "avg_kappa_approx(primary_lens_plane)/alpha = " << avgkap_scaled2/alpha << " (err=" << avgkaperr << ")" << endl;
		}
		cout << "avg_sigma_enclosed = " << avg_sigma_enclosed << endl;
		cout << "mass_enclosed = " << mass_enclosed << endl;
		if (menc_scaled_to_primary_lensplane != 0) cout << "mass(primary_lens_plane) = " << menc_scaled_to_primary_lensplane << endl;
	}
	//rmax_numerical = abs(rmax_numerical);
	delete[] perturber_list;
	return true;
}

void QLens::get_perturber_avgkappa_scaled(int lens_number, const double r0, double &avgkap_scaled, double &menc_scaled, double &avgkap0, bool verbal)
{
	bool *perturber_list = new bool[nlens];
	for (int i=0; i < nlens; i++) perturber_list[i] = false;
	perturber_list[lens_number] = true;

	double zlsub, zlprim;
	zlsub = lens_list[lens_number]->zlens;
	zlprim = lens_list[primary_lens_number]->zlens;

	if (zlsub > zlprim) {
		if ((lens_list[lens_number]->get_specific_parameter("xc_l",perturber_center[0])==false) or (lens_list[lens_number]->get_specific_parameter("yc_l",perturber_center[1])==false)) {
			if (find_lensed_position_of_background_perturber(verbal,lens_number,perturber_center,reference_zfactors,default_zsrc_beta_factors)==false) {
				warn("could not find lensed position of background perturber");
				delete[] perturber_list;
				die();
			}
		}
	} else {
		lens_list[perturber_lens_number]->get_center_coords(perturber_center[0],perturber_center[1]);
	}


	double kappa0, shear_tot, shear_angle;
	lensvector x;
	x[0] = perturber_center[0] + r0*cos(theta_shear);
	x[1] = perturber_center[1] + r0*sin(theta_shear);
	kappa0 = kappa_exclude(x,perturber_list,reference_zfactors,default_zsrc_beta_factors);
	shear_exclude(x,shear_tot,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);
	avgkap0 = 1 - kappa0 - shear_tot;

	double r;
	if ((zlsub > zlprim) and (include_recursive_lensing)) {
		lensvector x;
		x[0] = perturber_center[0] + r0*cos(theta_shear);
		x[1] = perturber_center[1] + r0*sin(theta_shear);
		lensvector xp, xpc;
		lens_list[lens_number]->get_center_coords(xpc[0],xpc[1]);
		double zsrc0 = source_redshift;
		set_source_redshift(zlsub);
		lensvector defp;
		deflection(x,defp,reference_zfactors,default_zsrc_beta_factors);
		set_source_redshift(zsrc0);
		xp[0] = x[0] - defp[0];
		xp[1] = x[1] - defp[1];
		r = sqrt(SQR(xp[0]-xpc[0])+SQR(xp[1]-xpc[1]));
	} else r = abs(r0);

	double avg_kappa = reference_zfactors[lens_redshift_idx[lens_number]]*lens_list[lens_number]->kappa_avg_r(r);

	double avg_sigma_enclosed = avg_kappa*cosmo->sigma_crit_arcsec(zlsub,reference_source_redshift);
	double mass_enclosed = avg_sigma_enclosed*M_PI*SQR(r0);

	menc_scaled = 0;
	avgkap_scaled = 0;
	double k0deriv=0;
	if (include_recursive_lensing) {
		if (zlsub < zlprim) {
			//double kappa0, shear_tot, shear_angle;
			//lensvector x;
			//x[0] = perturber_center[0] + r0*cos(theta_shear);
			//x[1] = perturber_center[1] + r0*sin(theta_shear);
			//kappa0 = kappa_exclude(x,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			//shear_exclude(x,shear_tot,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);

			int i1,i2;
			i1 = lens_redshift_idx[primary_lens_number];
			i2 = lens_redshift_idx[lens_number];
			double beta = default_zsrc_beta_factors[i1-1][i2];
			double dr = 1e-5;
			double kappa0_p, shear_tot_p;
			lensvector xp;
			xp[0] = perturber_center[0] + (r0+dr)*cos(theta_shear);
			xp[1] = perturber_center[1] + (r0+dr)*sin(theta_shear);
			kappa0_p = kappa_exclude(xp,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(xp,shear_tot_p,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			double kappa0_m, shear_tot_m;
			lensvector xm;
			xm[0] = perturber_center[0] + (r0-dr)*cos(theta_shear);
			xm[1] = perturber_center[1] + (r0-dr)*sin(theta_shear);
			kappa0_m = kappa_exclude(xm,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			shear_exclude(xm,shear_tot_m,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			k0deriv = (kappa0_p+shear_tot_p-kappa0_m-shear_tot_m)/(2*dr);
			double mass_scale_factor = (cosmo->sigma_crit_kpc(zlprim,reference_source_redshift) / cosmo->sigma_crit_kpc(zlsub,reference_source_redshift))*SQR(r0/r)*(1 - beta*(kappa0 + shear_tot + r0*k0deriv));
			//double fac1 = (cosmo->sigma_crit_kpc(zlprim,reference_source_redshift) / cosmo->sigma_crit_kpc(zlsub,reference_source_redshift));
			//double fac2 = (1 - beta*(kappa0 + shear_tot + r0*k0deriv));
			//cout << fac1 << " " << fac2 << " " << mass_scale_factor << endl;
			menc_scaled = mass_enclosed*mass_scale_factor;
			avgkap_scaled = avg_kappa*(1-beta*(kappa0+shear_tot+abs(r0)*k0deriv));
		} else if (zlsub > zlprim) {
			//double kappa0, shear_tot, shear_angle;
			//lensvector x;
			//x[0] = perturber_center[0] + r0*cos(theta_shear);
			//x[1] = perturber_center[1] + r0*sin(theta_shear);
			//kappa0 = kappa_exclude(x,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			//shear_exclude(x,shear_tot,shear_angle,perturber_list,reference_zfactors,default_zsrc_beta_factors);
			int i1,i2;
			i1 = lens_redshift_idx[primary_lens_number];
			i2 = lens_redshift_idx[lens_number];
			double beta = default_zsrc_beta_factors[i2-1][i1];
			double mass_scale_factor = (cosmo->sigma_crit_kpc(zlprim,reference_source_redshift) / cosmo->sigma_crit_kpc(zlsub,reference_source_redshift))*(1 - beta*(kappa0 + shear_tot));
			menc_scaled = mass_enclosed*mass_scale_factor;
			avgkap_scaled = avg_kappa*(1-beta*(kappa0+shear_tot));
		}
	} else {
		double mass_scale_factor = (cosmo->sigma_crit_kpc(zlprim,reference_source_redshift) / cosmo->sigma_crit_kpc(zlsub,reference_source_redshift));
		menc_scaled = mass_enclosed*mass_scale_factor;
		avgkap_scaled = avg_kappa;
	}
	if ((verbal) and (mpi_id==0)) cout << "CHECK: " << r0 << " " << r << " " << avg_kappa << " " << avgkap_scaled << " ... " << kappa0 << " " << shear_tot << " " << k0deriv << endl;

	delete[] perturber_list;
}


double QLens::subhalo_perturbation_radius_equation(const double r)
{
	double kappa0, shear0, shear_angle, subhalo_avg_kappa;
	lensvector x;
	x[0] = perturber_center[0] + r*cos(theta_shear);
	x[1] = perturber_center[1] + r*sin(theta_shear);
	kappa0 = kappa_exclude(x,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
	shear_exclude(x,shear0,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);

	double zlsub, zlprim;
	zlsub = lens_list[perturber_lens_number]->zlens;
	zlprim = lens_list[0]->zlens;

	double r_eff;
	if (zlsub > zlprim) {
		lensvector xp, xpc;
		lens_list[perturber_lens_number]->get_center_coords(xpc[0],xpc[1]);
		double zsrc0 = source_redshift;
		set_source_redshift(zlsub);
		lensvector alpha;
		deflection(x,alpha,reference_zfactors,default_zsrc_beta_factors);
		set_source_redshift(zsrc0);
		xp[0] = x[0] - alpha[0];
		xp[1] = x[1] - alpha[1];
		r_eff = sqrt(SQR(xp[0]-xpc[0])+SQR(xp[1]-xpc[1]));
	} else {
		r_eff = r;
	}
	subhalo_avg_kappa = 0;
	for (int i=0; i < nlens; i++) {
		if (linked_perturber_list[i]) subhalo_avg_kappa += reference_zfactors[lens_redshift_idx[i]]*lens_list[i]->kappa_avg_r(r_eff);
	}
	if (zlsub < zlprim) {
		int i1,i2;
		i1 = lens_redshift_idx[primary_lens_number];
		i2 = lens_redshift_idx[perturber_lens_number];
		double beta = default_zsrc_beta_factors[i1-1][i2];
		double dr = 1e-5;
		double kappa0_p, shear0_p;
		lensvector xp;
		xp[0] = perturber_center[0] + (r+dr)*cos(theta_shear);
		xp[1] = perturber_center[1] + (r+dr)*sin(theta_shear);
		kappa0_p = kappa_exclude(xp,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		shear_exclude(xp,shear0_p,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		double kappa0_m, shear0_m;
		lensvector xm;
		xm[0] = perturber_center[0] + (r-dr)*cos(theta_shear);
		xm[1] = perturber_center[1] + (r-dr)*sin(theta_shear);
		kappa0_m = kappa_exclude(xm,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		shear_exclude(xm,shear0_m,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
		double k0deriv = (kappa0_p+shear0_p-kappa0_m-shear0_m)/(2*dr);
		subhalo_avg_kappa *= 1 - beta*(kappa0 + shear0 + r*k0deriv);
	} else if (zlsub > zlprim) {
		int i1,i2;
		i1 = lens_redshift_idx[primary_lens_number];
		i2 = lens_redshift_idx[perturber_lens_number];
		double beta = default_zsrc_beta_factors[i2-1][i1];
		subhalo_avg_kappa *= 1 - beta*(kappa0 + shear0);
	}
	return (1 - kappa0 - shear0 - subhalo_avg_kappa);
}

double QLens::perturbation_radius_equation_nosub(const double r)
{
	double kappa0, shear0, shear_angle;
	lensvector x;
	x[0] = perturber_center[0] + r*cos(theta_shear);
	x[1] = perturber_center[1] + r*sin(theta_shear);
	kappa0 = kappa_exclude(x,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
	shear_exclude(x,shear0,shear_angle,linked_perturber_list,reference_zfactors,default_zsrc_beta_factors);
	return (1 - kappa0 - shear0);
}

bool QLens::get_einstein_radius(int lens_number, double& re_major_axis, double& re_average)
{
	if (lens_number >= nlens) { warn("lens %i has not been created",lens_number); return false; }
	lens_list[lens_number]->get_einstein_radius(re_major_axis,re_average,reference_zfactors[lens_redshift_idx[lens_number]]);
	return true;
}

double QLens::inverse_magnification_r(const double r)
{
	lensmatrix jac;
	hessian(grid_xcenter + r*cos(theta_crit), grid_ycenter + r*sin(theta_crit), jac, 0, reference_zfactors, default_zsrc_beta_factors);
	jac[0][0] = 1 - jac[0][0];
	jac[1][1] = 1 - jac[1][1];
	jac[0][1] = -jac[0][1];
	jac[1][0] = -jac[1][0];
	return determinant(jac);
}

Vector<dvector> QLens::find_critical_curves(bool &check)
{
	Vector<dvector> rcrit(2);
	rcrit[0].input(cc_thetasteps+1);
	rcrit[1].input(cc_thetasteps+1);

	double (Brent::*mag_r)(const double);
	mag_r = static_cast<double (Brent::*)(const double)> (&QLens::inverse_magnification_r);

	double mag0, mag1;
	bool found_first_root;
	double rstep, thetastep, step_increment, step_increment_change, beginning_increment;
	thetastep = 2*M_PI/cc_thetasteps;
	beginning_increment = 1.2;
	step_increment_change = 0.5;
	bool first_iteration = true;
	double tangential_crit_total = 0;
	int i, iterations;
	double r;
	for (i=0, theta_crit=0; i < cc_thetasteps; i++, theta_crit += thetastep)
	{
		iterations = 0;
		rcrit[0][i] = 0; rcrit[1][i] = 0;
		step_increment = beginning_increment;
		for (;;)
		{
			iterations++;
			rstep = default_autogrid_initial_step;
			mag1 = inverse_magnification_r(cc_rmin);
			found_first_root = false;
			for (r=cc_rmin; r < cc_rmax-rstep; r += ((rstep *= step_increment)/step_increment))
			{
				mag0 = mag1;
				mag1 = inverse_magnification_r(r+rstep);
				if (mag0*mag1 < 0) {
					if (!found_first_root) {
						rcrit[0][i] = BrentsMethod(mag_r, r, r+rstep, 1e-3);
						if (rcrit[0][i] < 1e-6) die("catastrophic failure--critical curves smaller than 1e-6");
						found_first_root = true;
					} else {
						rcrit[1][i] = BrentsMethod(mag_r, r, r+rstep, 1e-3);
						if (rcrit[1][i] < 1e-6) die("catastrophic failure--critical curves smaller than 1e-6");
						tangential_crit_total += rcrit[1][i];
						break;
					}
				}
			}
			if (r+rstep >= cc_rmax) {
				if (iterations >= max_cc_search_iterations)
				{
					check = false;
					if (!found_first_root)
						warn(warnings, "could not find any critical curves along theta = %g after %i iterations",theta_crit,iterations);
					else
						warn(warnings, "could not find two critical curves along theta = %g after %i iterations",theta_crit,iterations);
					return rcrit;
				}
				step_increment = 1 + (step_increment-1)*step_increment_change;
				rstep = default_autogrid_initial_step;
			} else {
				if (first_iteration)	// adjust the scale of rstep if it is too small
				{
					double rmin_frac, cc0_max_rfrac_range;
					rmin_frac = rcrit[0][0] / cc_rmin;
					cc0_max_rfrac_range = 10; // This is the (fractional) margin allowed for the inner cc radius to vary
													  // --must be large, or else it might skip over both curves if they are close!
					while (rmin_frac > 2*cc0_max_rfrac_range) {
						cc_rmin *= 2;
						rmin_frac /= 2;
					}
					if (cc_rmin > rcrit[0][0]/cc0_max_rfrac_range) cc_rmin /= 2;
					first_iteration = false;
				}
				else break;
			}
		}
	}
	rcrit[0][cc_thetasteps] = rcrit[0][0];
	rcrit[1][cc_thetasteps] = rcrit[1][0];

	check = true;
	return rcrit;
}

bool QLens::find_optimal_gridsize()
{
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}

	double (Brent::*mag_r)(const double);
	mag_r = static_cast<double (Brent::*)(const double)> (&QLens::inverse_magnification_r);

	double mag0, mag1;
	bool found_first_root;
	int thetasteps = 40;
	double rstep, thetastep, step_increment;
	thetastep = 2*M_PI/thetasteps;
	step_increment = 1.2;
	bool first_iteration = true;
	int i;
	double r, min_r, max_r, global_rmin=1e30, global_rmax=0;
	double max_x, max_y, global_xmax=0, global_ymax=0;
	for (i=0, theta_crit=0; i < thetasteps; i++, theta_crit += thetastep)
	{
		min_r = 1e30; max_r = 0;
		if (global_rmax > 0) rstep = 0.2*global_rmax;
		else rstep = default_autogrid_initial_step;
		mag1 = inverse_magnification_r(cc_rmin);
		found_first_root = false;
		for (r=cc_rmin; r < cc_rmax-rstep; r += ((rstep *= step_increment)/step_increment))
		{
			mag0 = mag1;
			mag1 = inverse_magnification_r(r+rstep);
			if (mag0*mag1 < 0) {
				if (!found_first_root) {
					min_r = BrentsMethod(mag_r, r, r+rstep, 1e-2*rstep);
					found_first_root = true;
				} else {
					max_r = BrentsMethod(mag_r, r, r+rstep, 1e-2*rstep);
				}
			}
		}
		if (!found_first_root) continue;
		if (min_r > max_r) max_r = min_r;
		max_x = abs(max_r*cos(theta_crit));
		max_y = abs(max_r*sin(theta_crit));
		if (min_r < global_rmin) {
			global_rmin = min_r;
		}
		if (max_r > global_rmax) {
			global_rmax = max_r;
		}
		if (max_x > global_xmax) global_xmax = max_x;
		if (max_y > global_ymax) global_ymax = max_y;
	}
	if ((global_xmax == 0) or (global_ymax == 0)) return false;
	grid_xlength = 2*(global_xmax*autogrid_frac);
	grid_ylength = 2*(global_ymax*autogrid_frac);
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	return true;
}

void QLens::sort_critical_curves()
{
	if (grid==NULL) {
		if (create_grid(false,reference_zfactors,default_zsrc_beta_factors)==false) { warn("Could not create recursive grid"); return; }
	}
	sorted_critical_curve.clear();
	int n_cc_pts = critical_curve_pts.size();
	if (n_cc_pts == 0) return;
	int n_cc = 1;
	double dist_threshold; // this should be defined by the smallest grid cell size
	double dist_threshold_frac = 2;
	vector<lensvector> critical_curves_temp = critical_curve_pts;
	vector<lensvector> caustics_temp = caustic_pts;
	vector<double> length_of_cell_temp = length_of_cc_cell;
	critical_curve new_critical_curve;
	lensvector displacement, last_pt;
	last_pt[0] = critical_curves_temp[0][0];
	last_pt[1] = critical_curves_temp[0][1];
	new_critical_curve.cc_pts.push_back(critical_curves_temp[0]);
	new_critical_curve.caustic_pts.push_back(caustics_temp[0]);
	new_critical_curve.length_of_cell.push_back(length_of_cell_temp[0]);
	dist_threshold = dist_threshold_frac*length_of_cell_temp[0];
	critical_curves_temp.erase(critical_curves_temp.begin());
	caustics_temp.erase(caustics_temp.begin());
	length_of_cell_temp.erase(length_of_cell_temp.begin());
	n_cc_pts--;

	int i, i_closest_pt, i_retry=0;
	double dist, shortest_dist;
	lensvector disp_from_first;
	while (n_cc_pts > 0) {
		shortest_dist = 1e30;
		for (i=0; i < n_cc_pts; i++) {
			displacement[0] = critical_curves_temp[i][0] - last_pt[0];
			displacement[1] = critical_curves_temp[i][1] - last_pt[1];
			dist = displacement.norm();
			if (dist < shortest_dist) {
				shortest_dist = dist;
				i_closest_pt = i;
			}
		}
		if (shortest_dist > dist_threshold) {
			disp_from_first[0] = last_pt[0] - new_critical_curve.cc_pts[0][0];
			disp_from_first[1] = last_pt[1] - new_critical_curve.cc_pts[0][1];
			if (disp_from_first.norm() > dist_threshold) {
				// Since it seems we're not closing the curve, maybe the issue is that the cell size changed as we traversed the critical curve.
				// Let's increase the distance threshold and try again (up to 3 tries).
				if (i_retry < 10) {
					i_retry++;
					dist_threshold *= 1.5;
					continue;
				}
			}
			// store this critical curve, move on to the next one
			sorted_critical_curve.push_back(new_critical_curve);
			new_critical_curve.cc_pts.clear();
			new_critical_curve.caustic_pts.clear();
			n_cc++;
			i_retry=0;
		}
		last_pt[0] = critical_curves_temp[i_closest_pt][0];
		last_pt[1] = critical_curves_temp[i_closest_pt][1];
		new_critical_curve.cc_pts.push_back(critical_curves_temp[i_closest_pt]);
		new_critical_curve.caustic_pts.push_back(caustics_temp[i_closest_pt]);
		new_critical_curve.length_of_cell.push_back(length_of_cell_temp[i_closest_pt]);
		dist_threshold = dist_threshold_frac*length_of_cell_temp[i_closest_pt];
		critical_curves_temp.erase(critical_curves_temp.begin()+i_closest_pt);
		caustics_temp.erase(caustics_temp.begin()+i_closest_pt);
		length_of_cell_temp.erase(length_of_cell_temp.begin()+i_closest_pt);
		n_cc_pts--;
	}
	sorted_critical_curve.push_back(new_critical_curve);
	sorted_critical_curves = true;
}

bool QLens::plot_critical_curves(string critfile)
{
	if (!sorted_critical_curves) sort_critical_curves();

	if (critfile != "") {
		ofstream crit;
		open_output_file(crit,critfile);
		if (use_scientific_notation) crit << setiosflags(ios::scientific);
		int n_cc = sorted_critical_curve.size();
		if (n_cc==0) return false;
		for (int j=0; j < n_cc; j++) {
			for (int k=0; k < sorted_critical_curve[j].cc_pts.size(); k++) {
				crit << sorted_critical_curve[j].cc_pts[k][0] << " " << sorted_critical_curve[j].cc_pts[k][1] << " " << sorted_critical_curve[j].caustic_pts[k][0] << " " << sorted_critical_curve[j].caustic_pts[k][1] << " " << sorted_critical_curve[j].length_of_cell[k] << endl;
			}
			// connect the first and last points to make a closed curve
			crit << sorted_critical_curve[j].cc_pts[0][0] << " " << sorted_critical_curve[j].cc_pts[0][1] << " " << sorted_critical_curve[j].caustic_pts[0][0] << " " << sorted_critical_curve[j].caustic_pts[0][1] << " " << sorted_critical_curve[j].length_of_cell[0] << endl;
			if (j < n_cc-1) crit << endl; // separates the critical curves in the plot
		}
	}
	return true;
}

bool QLens::find_caustic_minmax(double& min, double& max, double& max_minor_axis, const double cc_num)
{
	if (!sorted_critical_curves) sort_critical_curves();

	auto get_angle_from_components = [](const double &comp1, const double &comp2)
	{
		double angle = atan(abs(comp2/comp1));
		if (comp1 < 0) {
			if (comp2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (comp2 < 0) {
			angle = -angle;
		}
		return angle;
	};


	int n_cc = sorted_critical_curve.size();
	if (n_cc==0) return false;
	critical_curve* critical_curve = &sorted_critical_curve[cc_num];
	int npts = critical_curve->cc_pts.size();
	if (npts==0) return false;
	double x_avg=0, y_avg=0;
	for (int k=0; k < npts; k++) {
		x_avg += critical_curve->caustic_pts[k][0];
		y_avg += critical_curve->caustic_pts[k][1];
	}
	x_avg /= npts;
	y_avg /= npts;
	double rsq;
	double rsqmax = 0;
	double rsqmin = 1e30;
	double theta_rmax;
	for (int k=0; k < npts; k++) {
		rsq = SQR(critical_curve->caustic_pts[k][0]-x_avg) + SQR(critical_curve->caustic_pts[k][1]-y_avg);
		if (rsq > rsqmax) {
			rsqmax = rsq;
			theta_rmax = get_angle_from_components(critical_curve->caustic_pts[k][0]-x_avg,critical_curve->caustic_pts[k][1]-y_avg);
		}
		if (rsq < rsqmin) rsqmin = rsq;
	}
	cout << "THETA_RMAX=" << theta_rmax << endl;
	double theta, theta_minor_axis_min, theta_minor_axis_max;
	theta_minor_axis_min = theta_rmax + M_HALFPI - 0.1;
	theta_minor_axis_max = theta_rmax + M_HALFPI + 0.1;
	double rsqmax_minor_axis = 0;
	for (int k=0; k < npts; k++) {
		theta = get_angle_from_components(critical_curve->caustic_pts[k][0]-x_avg,critical_curve->caustic_pts[k][1]-y_avg);
		cout << theta << endl;
		if ((theta > theta_minor_axis_min) and (theta < theta_minor_axis_max)) {
			rsq = SQR(critical_curve->caustic_pts[k][0]-x_avg) + SQR(critical_curve->caustic_pts[k][1]-y_avg);
			cout << "rsq = " << rsq << endl;
			if (rsq > rsqmax_minor_axis) rsqmax_minor_axis = rsq;
		}
	}
	cout << "RSQMAX_MINOR_AXIS=" << rsqmax_minor_axis << endl;
	min = sqrt(rsqmin);
	max = sqrt(rsqmax);
	max_minor_axis = sqrt(rsqmax_minor_axis);
	return true;
}

double QLens::einstein_radius_of_primary_lens(const double zfac, double &reav)
{
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	// this calculates the Einstein radius of the "macro" lens model (treating the lens as spherical), ignoring any lens components that are not centered on the primary lens
	double rmin_einstein_radius = 1e-6;
	double rmax_einstein_radius = 1e4;
	double xc0, yc0, xc1, yc1, xc, yc;
	lens_list[primary_lens_number]->get_center_coords(xc0,yc0);
	centered = new bool[nlens];
	if (lens_list[primary_lens_number]->kapavgptr_rsq_spherical != NULL) centered[primary_lens_number]=true;
	else centered[primary_lens_number] = false;
	bool multiple_lenses = false;
	if (include_secondary_lens) {
		if (lens_list[secondary_lens_number]->kapavgptr_rsq_spherical != NULL) {
			centered[secondary_lens_number]=true;
			lens_list[secondary_lens_number]->get_center_coords(xc1,yc1);
			multiple_lenses = true;
		}
	}
	for (int j=0; j < nlens; j++) {
		if (j==primary_lens_number) continue;
		if ((include_secondary_lens) and (j==secondary_lens_number)) continue;
		lens_list[j]->get_center_coords(xc,yc);
		if ((lens_list[j]->lenstype==SHEET) or ((xc==xc0) and (yc==yc0)) or ((include_secondary_lens) and ((xc==xc1) and (yc==yc1)))) {
			// If a lens is selected as the "secondary" lens (e.g. a BCG), then it will be treated as co-centered with the primary even if there's an offset;
			// the same is true for any other lenses co-centered with the secondary
			if (lens_list[j]->kapavgptr_rsq_spherical != NULL) {
				multiple_lenses = true;
				centered[j]=true;
				if (multiple_lenses==false) multiple_lenses = true;
			} else centered[j] = false;
		}
		else centered[j]=false;
	}
	if (multiple_lenses==false) {
		delete[] centered;
		double re;
		lens_list[primary_lens_number]->get_einstein_radius(re,reav,zfac);
		return re;
	}
	zfac_re = zfac;
	if ((einstein_radius_root(rmin_einstein_radius)*einstein_radius_root(rmax_einstein_radius)) > 0) {
		// multiple imaging does not occur with this lens
		delete[] centered;
		return 0;
	}
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&QLens::einstein_radius_root);
	reav = BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-3);
	double fprim, fprim_max = -1e30;
	for (int j=0; j < nlens; j++) {
		if (centered[j]) {
			fprim = lens_list[primary_lens_number]->get_f_major_axis(); // use the primary lens's axis ratio to approximate the major axis of critical curve
			if (fprim > fprim_max) fprim_max = fprim;
		}
	}
	double re_maj_approx = fprim_max*reav;
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating Einstein radius: " << wtime << endl;
	}
#endif
	delete[] centered;
	return re_maj_approx;
}

double QLens::einstein_radius_root(const double r)
{
	double kapavg=0;
	for (int j=0; j < nlens; j++) {
		if (centered[j]) kapavg += zfac_re*lens_list[j]->kappa_avg_r(r);
	}
	return (kapavg-1);
}

void QLens::plot_total_kappa(const double rmin, const double rmax, const int steps, const char *kname, const char *kdname)
{
	ofstream kout(kname);
	ofstream kdout;
	if (kdname != NULL) kdout.open(kdname);
	if (use_scientific_notation) kout << setiosflags(ios::scientific);
	if (use_scientific_notation) kdout << setiosflags(ios::scientific);
	dvector rvals(steps), kappavals(steps), dkappavals(steps);
	output_total_kappa(rmin, rmax, steps, rvals, kappavals, dkappavals);
	double arcsec_to_kpc = cosmo->angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
	double sigma_cr_kpc = cosmo->sigma_crit_kpc(lens_redshift, reference_source_redshift);
	for (int i=0; i < steps; i++) {
		kout << rvals[i] << " " << kappavals[i] << " " << rvals[i]*arcsec_to_kpc << " " << kappavals[i]*sigma_cr_kpc << endl;
		if (kdname != NULL) kdout << rvals[i] << " " << dkappavals[i] << " " << rvals[i]*arcsec_to_kpc << " " << dkappavals[i]*sigma_cr_kpc/arcsec_to_kpc << endl;
	}
}

void QLens::output_total_kappa(const double rmin, const double rmax, const int steps, dvector& rvals, dvector& kappavals, dvector& dkappavals)
{
	double r, rstep, total_kappa, total_dkappa;
	rstep = pow(rmax/rmin, 1.0/steps);
	int i,j;
	double kap, kap2;
	double theta, thetastep;
	int thetasteps = 200;
	thetastep = 2*M_PI/thetasteps;
	double x, y, x2, y2, dr;
	dr = 1e-1*rmin*(rstep-1);
	
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}
	double *onezfac = new double[n_lens_redshifts];
	for (i=0; i < n_lens_redshifts; i++) onezfac[i] = 1.0;
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		total_kappa = 0;
		total_dkappa = 0;
		for (j=0, theta=0; j < thetasteps; j++, theta += thetastep) {
			x = grid_xcenter + r*cos(theta);
			y = grid_ycenter + r*sin(theta);
			x2 = (r+dr)*cos(theta);
			y2 = (r+dr)*sin(theta);
			kap = kappa(x,y,onezfac,default_zsrc_beta_factors);
			kap2 = kappa(x2,y2,onezfac,default_zsrc_beta_factors);
			total_kappa += kap;
			total_dkappa += (kap2 - kap)/dr;
		}
		total_kappa /= thetasteps;
		total_dkappa /= thetasteps;
		rvals[i] = r;
		kappavals[i] = total_kappa;
		dkappavals[i] = total_dkappa;
	}
	delete[] onezfac;

	/*
	double rsq;
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		rsq = r*r;
		kout << r << " ";
		if (kdname != NULL) kdout << r << " ";
		total_kappa = 0;
		if (kdname != NULL) total_dkappa = 0;
		for (int j=0; j < nlens; j++) {
			if (centered[j]) {
				// this ignores off-center lenses (perturbers) since we are plotting the radial profile; ellipticity is also ignored
				kap = lens_list[j]->kappa_rsq(rsq);
				if (kdname != NULL) dkap = lens_list[j]->kappa_rsq_deriv(rsq);
				total_kappa += kap;
				if (kdname != NULL) total_dkappa += dkap;
			}
		}
		kout << total_kappa << endl;
		if (kdname != NULL) kdout << fabs(total_dkappa) << endl;
	}
	*/
}

double QLens::einstein_radius_single_lens(const double src_redshift, const int lensnum)
{
	double re_avg,re_major,zfac;
	zfac = cosmo->kappa_ratio(lens_list[lensnum]->zlens,src_redshift,reference_source_redshift);
	lens_list[lensnum]->get_einstein_radius(re_major,re_avg,zfac);
	return re_avg;
}

double QLens::get_xi_parameter(const double src_redshift, const int lensnum)
{
	double re_avg,re_major,zfac,xi_param;
	zfac = cosmo->kappa_ratio(lens_list[lensnum]->zlens,src_redshift,reference_source_redshift);
	xi_param = lens_list[lensnum]->get_xi_parameter(zfac);
	//double xitot = get_total_xi_parameter(src_redshift);
	//cout << "CHECK: " << xitot << " " << xi_param << endl;
	return xi_param;
}

double QLens::get_total_xi_parameter(const double src_redshift)
{
	double r_ein,zfac,xi_param;
	zfac = cosmo->kappa_ratio(lens_list[primary_lens_number]->zlens,src_redshift,reference_source_redshift);
	einstein_radius_of_primary_lens(zfac,r_ein);
	//cout << "RE=" << r_ein << endl;
	double xc,yc,xcc,ycc;
	lens_list[primary_lens_number]->get_center_coords(xc,yc);

	int i,j;
	const int n_theta = 100;
	double theta, theta_step = M_2PI/n_theta;

	double xifac_avg = 0;
	double dkappa_e_tot, kappa_e_tot, kap, dkap;

	bool* include_lens = new bool[nlens];
	for (i=0; i < nlens; i++) {
		include_lens[i] = false;
		if (i==primary_lens_number) include_lens[i] = true;
		else if (lens_list[i]->lenstype==SHEET) include_lens[i] = true;
		else {
			if (lens_list[i]->ellipticity_mode != -1) { // this would mean it's an elliptical lens
				lens_list[i]->get_center_coords(xcc,ycc);
				if ((xcc==xc) and (ycc==yc)) include_lens[i] = true; // only include co-centered lenses
			}
		}
	}

	double x,y;
	for (i=0, theta=0.0; i < n_theta; i++, theta += theta_step) {
		x = xc + r_ein*cos(theta);
		y = yc + r_ein*sin(theta);
		kappa_e_tot = 0;
		dkappa_e_tot = 0;
		for (j=0; j < nlens; j++) {
			if (include_lens[j]) {
				lens_list[j]->kappa_and_dkappa_dR(x,y,kap,dkap);
				kappa_e_tot += zfac*kap;
				dkappa_e_tot += zfac*dkap; // we express xi in terms of derivative of kappa, rather than second derivative of the deflection
				//cout << "K=" << kap << " " << kappa_e_tot << endl;
				//cout << "dK=" << dkap << " " << dkappa_e_tot << endl;
			}
		}
		xifac_avg += dkappa_e_tot/(1-kappa_e_tot);
	}
	xifac_avg /= n_theta;
	delete[] include_lens;
	return (2*r_ein*xifac_avg+2);
}

double QLens::total_kappa(const double r, const int lensnum, const bool use_kpc)
{
	// this is used by the DerivedParam class in qlens.h
	double total_kappa;
	int j;
	double kap, kap2;
	double theta, thetastep;
	int thetasteps = 200;
	thetastep = 2*M_PI/thetasteps;
	double x, y;
	double z, r_arcsec = r;
	if (lensnum==-1) z = lens_list[primary_lens_number]->get_redshift();
	else z = lens_list[lensnum]->get_redshift();
	if (use_kpc) r_arcsec *= 206.264806/cosmo->angular_diameter_distance(z);
	
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}
	total_kappa = 0;
	double *onezfac = new double[n_lens_redshifts];
	for (j=0; j < n_lens_redshifts; j++) onezfac[j] = 1.0; // this ensures that the reference source redshift is used, which is appropriate to each lens
	for (j=0, theta=0; j < thetasteps; j++, theta += thetastep) {
		x = grid_xcenter + r_arcsec*cos(theta);
		y = grid_ycenter + r_arcsec*sin(theta);
		if (lensnum==-1) kap = kappa(x,y,onezfac,default_zsrc_beta_factors);
		else kap = lens_list[lensnum]->kappa(x,y);
		total_kappa += kap;
	}
	delete[] onezfac;
	total_kappa /= thetasteps;
	return total_kappa;
}

double QLens::total_dlogkappa(const double r, const int lensnum, const bool use_kpc)
{
	double total_dlogkappa;
	int j;
	double kap, kap2;
	double theta, thetastep;
	int thetasteps = 200;
	thetastep = 2*M_PI/thetasteps;
	double x, y, x2, y2, dlogr, rfac;
	dlogr = 1e-5;
	rfac = exp(dlogr);
	double z, r_arcsec = r;
	if (lensnum==-1) z = lens_list[primary_lens_number]->get_redshift();
	else z = lens_list[lensnum]->get_redshift();
	if (use_kpc) r_arcsec *= 206.264806/cosmo->angular_diameter_distance(z);
	
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}
	total_dlogkappa = 0;
	double *onezfac = new double[n_lens_redshifts];
	for (j=0; j < n_lens_redshifts; j++) onezfac[j] = 1.0;
	for (j=0, theta=0; j < thetasteps; j++, theta += thetastep) {
		x = grid_xcenter + r_arcsec*cos(theta);
		y = grid_ycenter + r_arcsec*sin(theta);
		x2 = (r_arcsec*rfac)*cos(theta);
		y2 = (r_arcsec*rfac)*sin(theta);
		if (lensnum==-1) {
			kap = kappa(x,y,onezfac,default_zsrc_beta_factors);
			kap2 = kappa(x2,y2,onezfac,default_zsrc_beta_factors);
		} else {
			kap = lens_list[lensnum]->kappa(x,y);
			kap2 = lens_list[lensnum]->kappa(x2,y2);
		}
		total_dlogkappa += (log(kap2/kap))/dlogr;
	}
	total_dlogkappa /= thetasteps;
	return total_dlogkappa;
}

void QLens::plot_mass_profile(double rmin, double rmax, int rpts, const char *massname)
{
	double r, rstep, kavg;
	rstep = pow(rmax/rmin, 1.0/(rpts-1));
	int i;
	ofstream mout(massname);
	if (use_scientific_notation) mout << setiosflags(ios::scientific);
	double arcsec_to_kpc = cosmo->angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
	double sigma_cr_arcsec = cosmo->sigma_crit_arcsec(lens_redshift, reference_source_redshift);
	mout << "#radius(arcsec) mass(m_solar) radius(kpc)\n";
	for (i=0, r=rmin; i < rpts; i++, r *= rstep) {
		kavg = 0;
		for (int j=0; j < nlens; j++) {
			kavg += lens_list[j]->kappa_avg_r(r);
		}
		mout << r << " " << sigma_cr_arcsec*M_PI*kavg*r*r << " " << r*arcsec_to_kpc << endl;
	}
}

void QLens::plot_kappa_profile(int l, double rmin, double rmax, int steps, const char *kname, const char *kdname)
{
	if (l >= nlens) { warn("lens %i does not exist", l); return; }
	ofstream kout, kdout;
	open_output_file(kout,kname);
	if (kdname != NULL) open_output_file(kdout,kdname);
	lens_list[l]->plot_kappa_profile(rmin,rmax,steps,kout,kdout);
}

void QLens::plot_sb_profile(int l, double rmin, double rmax, int steps, const char *sname)
{
	if (l >= n_sb) { warn("src %i does not exist", l); return; }
	ofstream sbout;
	open_output_file(sbout,sname);
	sb_list[l]->plot_sb_profile(rmin,rmax,steps,sbout);
}

void QLens::plot_total_sbprofile(double rmin, double rmax, int steps, const char *sbname)
{
	double r, rstep, total_sbprofile;
	rstep = pow(rmax/rmin, 1.0/steps);
	int i,j,k;
	ofstream sbout;
	open_output_file(sbout,sbname);
	if (use_scientific_notation) sbout << setiosflags(ios::scientific);
	double arcsec_to_kpc = cosmo->angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
	double sigma_cr_kpc = cosmo->sigma_crit_kpc(lens_redshift, reference_source_redshift);
	double sb;
	double theta, thetastep;
	int thetasteps = 200;
	thetastep = 2*M_PI/thetasteps;
	double x, y;
	
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==primary_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		total_sbprofile = 0;
		for (j=0, theta=0; j < thetasteps; j++, theta += thetastep) {
			x = grid_xcenter + r*cos(theta);
			y = grid_ycenter + r*sin(theta);
			sb=0;
			for (k=0; k < n_sb; k++)
				sb += sb_list[k]->surface_brightness(x,y);
			total_sbprofile += sb;
		}
		total_sbprofile /= thetasteps;
		sbout << r << " " << total_sbprofile << " " << r*arcsec_to_kpc << endl;
	}
}

bool QLens::isspherical()
{
	bool all_spherical = true;
	for (int i=0; i < nlens; i++)
		if (!(lens_list[i]->isspherical())) { all_spherical = false; break; }
	return all_spherical;
}

void QLens::print_lensing_info_at_point(const double x, const double y)
{
	lensvector point, alpha, beta;
	double sheartot, shear_angle;
	point[0] = x; point[1] = y;
	deflection(point,alpha,reference_zfactors,default_zsrc_beta_factors);
	//lensvector alpha2;
	//custom_deflection(point[0],point[1],alpha2);
	shear(point,sheartot,shear_angle,0,reference_zfactors,default_zsrc_beta_factors);
	beta[0] = point[0] - alpha[0];
	beta[1] = point[1] - alpha[1];
	double kappaval = kappa(point,reference_zfactors,default_zsrc_beta_factors);
	if (mpi_id==0) {
		cout << "kappa = " << kappaval << endl;
		cout << "deflection = (" << alpha[0] << "," << alpha[1] << ")\n";
		//cout << "custom deflection = (" << alpha2[0] << "," << alpha2[1] << ")\n";
		cout << "potential = " << potential(point,reference_zfactors,default_zsrc_beta_factors) << endl;
		cout << "magnification = " << magnification(point,0,reference_zfactors,default_zsrc_beta_factors) << endl;
		cout << "shear = " << sheartot << ", shear_angle=" << shear_angle << endl;
		cout << "reduced_shear1 = " << sheartot*cos(2*shear_angle*M_PI/180.0)/(1-kappaval) << " reduced_shear2 = " << sheartot*sin(2*shear_angle*M_PI/180.0)/(1-kappaval) << endl;
		cout << "sourcept = (" << beta[0] << "," << beta[1] << ")\n";

		/*
		if (n_lens_redshifts > 1) {
			lensvector xl;
			for (int i=1; i < n_lens_redshifts; i++) {
				map_to_lens_plane(i,x,y,xl,0,reference_zfactors,default_zsrc_beta_factors);
				cout << "x(z=" << lens_redshifts[i] << "): (" << xl[0] << "," << xl[1] << ")" << endl;
			}

			int i,j;
			for (i=1; i < n_lens_redshifts; i++) {
				for (j=0; j < i; j++) cout << "beta(" << i << "," << j << "): " << default_zsrc_beta_factors[i-1][j] << endl;
			}
			for (i=0; i < n_lens_redshifts; i++) cout << "zfac(" << i << "): " << reference_zfactors[i] << endl;
		}
		*/
		if (n_sb > 0) {
			double sb = find_sbprofile_surface_brightness(point);
			cout << "surface brightness from analytic sources = " << sb << endl;
		}
		cout << endl;
		//cout << "shear/kappa = " << sheartot/kappa(point) << endl;
	}
}

void QLens::make_source_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename)
{
	ofstream sourcetab(source_filename.c_str());
	int i,j;
	double x,y,xstep,ystep;
	xstep = (xmax-xmin)/(xsteps-1);
	ystep = (ymax-ymin)/(ysteps-1);
	for (i=0, x=xmin; i < xsteps; i++, x += xstep)
		for (j=0, y=ymin; j < ysteps; j++, y += ystep)
			sourcetab << x << " " << y << endl;
}

void QLens::make_source_ellipse(const double xcenter, const double ycenter, const double major_axis, const double q, const double angle_degrees, const int n_subellipses, const int points_per_ellipse, const bool draw_in_imgplane, string source_filename)
{
	ofstream source_file; open_output_file(source_file,source_filename);

	double da, dtheta, angle;
	da = major_axis/n_subellipses;
	dtheta = M_2PI/points_per_ellipse;
	angle = (M_PI/180)*angle_degrees;
	double a, theta, x, y;
	lensvector pt;

	int i,j;
	for (i=1, a=da; i <= n_subellipses; i++, a += da)
	{
		for (j=0, theta=0; j < points_per_ellipse; j++, theta += dtheta)
		{
			x = a*cos(theta); y = a*q*sin(theta);
			pt[0] = xcenter + x*cos(angle) - y*sin(angle);
			pt[1] = ycenter + x*sin(angle) + y*cos(angle);
			if (draw_in_imgplane) {
				find_sourcept(pt,source,0,reference_zfactors,default_zsrc_beta_factors);
			} else {
				source[0] = pt[0];
				source[1] = pt[1];
			}
			source_file << source[0] << " " << source[1] << endl;
		}
	}
}

void QLens::raytrace_image_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename)
{
	ofstream sourcetab(source_filename.c_str());
	int i,j;
	double x,y,xs,ys,xstep,ystep;
	lensvector point, alpha;
	xstep = (xmax-xmin)/(xsteps-1);
	ystep = (ymax-ymin)/(ysteps-1);
	for (i=0, x=xmin; i < xsteps; i++, x += xstep) {
		for (j=0, y=ymin; j < ysteps; j++, y += ystep) {
			point[0] = x; point[1] = y;
			deflection(point,alpha,reference_zfactors,default_zsrc_beta_factors);
			xs = point[0] - alpha[0];
			ys = point[1] - alpha[1];
			sourcetab << xs << " " << ys << endl;
		}
	}
}

/********************************* Functions for point image data (reading, writing, simulating etc.) *********************************/

bool QLens::add_simulated_image_data(const lensvector &sourcept, const double srcflux)
{
	int i,n_images;
	if (nlens==0) { warn("no lens model has been created"); return false; }
	image *imgs = get_images(sourcept, n_images, false);
	if (n_images==0) { warn("could not find any images; no data added"); return false; }

	add_point_source(source_redshift,sourcept,false);
	// replace below lines with function that determines which vary flags to set to true, then set them using set_ptsrc_vary_parameters...DO THIS
	//if (!use_analytic_bestfit_src) {
		//set_sourcept_vary_parameters(n_sourcepts_fit-1,true,true);
	//}

	bool include_image[n_images];
	double err_pos[n_images];
	double err_flux[n_images];

	double min_td=1e30;
	for (i=0; i < n_images; i++) {
		// central maxima images have positive parity and kappa > 1, so use this to exclude them if desired
		if ((include_central_image==false) and (imgs[i].parity == 1) and (kappa(imgs[i].pos,reference_zfactors,default_zsrc_beta_factors) > 1)) include_image[i] = false;
		else include_image[i] = true;
		err_pos[i] = sim_err_pos;
		err_flux[i] = sim_err_flux;
		imgs[i].pos[0] += sim_err_pos*NormalDeviate();
		imgs[i].pos[1] += sim_err_pos*NormalDeviate();
		imgs[i].mag *= srcflux; // now imgs[i].mag is in fact the flux, not just the magnification
		imgs[i].mag += sim_err_flux*NormalDeviate();
		if (include_time_delays) {
			imgs[i].td += sim_err_td*NormalDeviate();
			if (imgs[i].td < min_td) min_td = imgs[i].td;
		}
	}
	if (include_time_delays) {
		for (int i=0; i < n_images; i++) {
			imgs[i].td -= min_td;
		}
	}
	point_image_data[n_ptsrc-1].input(n_images,imgs,err_pos,err_flux,sim_err_td,include_image,include_time_delays);

	sort_image_data_into_redshift_groups();
	include_imgpos_chisq = true;
	return true;
}

bool QLens::add_ptimage_data_from_unlensed_sourcepts(const bool include_errors_from_fisher_matrix, const int param_i, const double scale_errors)
{
	int i,n_images = n_ptsrc;
	if (n_images==0) { warn("could not find any images; no data added"); return false; }
	image imgs[n_images];
	for (i=0; i < n_images; i++) {
		imgs[i].pos[0] = ptsrc_list[i]->pos[0];
		imgs[i].pos[1] = ptsrc_list[i]->pos[1];
		imgs[i].flux = 0; // we don't have a good estimate of the flux
	}
	clear_image_data();

	lensvector sourcept(0,0);
	bool vary_source_coords = (use_analytic_bestfit_src) ? false : true;
	add_point_source(source_redshift,sourcept,vary_source_coords);
	// replace below lines with function that determines which vary flags to set to true, then set them using set_ptsrc_vary_parameters...DO THIS (in above function too)
	//if (!use_analytic_bestfit_src) {
		//set_sourcept_vary_parameters(n_sourcepts_fit-1,true,true);
	//}

	bool include[n_images];
	double err_pos[n_images];
	double err_flux[n_images];

	double err_xsq, err_ysq;
	int indx=0;
	for (i=0; i < n_images; i++) {
		include[i] = true;
		if (include_errors_from_fisher_matrix) {
			err_xsq = abs(fisher_inverse[param_i+indx][param_i+indx]);
			indx++;
			err_ysq = abs(fisher_inverse[param_i+indx][param_i+indx]);
			indx++;
			err_pos[i] = scale_errors*sqrt(dmax(err_xsq,err_ysq)); // right now, imgdata doesn't treat error in x vs. y separately; it uses a common error for both
		} else {
			err_pos[i] = sim_err_pos;
		}
		err_flux[i] = sim_err_flux;
		//point_image_data[0].add_image(sourcepts_fit[i], sim_err_pos, 0, 0, 0, 0);
	}
	point_image_data[0].input(n_images,imgs,err_pos,err_flux,0,include,false);

	sort_image_data_into_redshift_groups();
	include_imgpos_chisq = true;
	return true;
}

void QLens::write_point_image_data(string filename)
{
	ofstream outfile(filename.c_str());
	if (use_scientific_notation==true) outfile << setiosflags(ios::scientific);
	else {
		outfile << setprecision(6);
		outfile << fixed;
	}
	if (data_info != "") outfile << "# data_info: " << data_info << endl;
	outfile << "zlens = " << lens_redshift << endl;
	outfile << n_ptsrc << " # number of source points" << endl;
	for (int i=0; i < n_ptsrc; i++) {
		outfile << point_image_data[i].n_images << " " << ptsrc_redshifts[ptsrc_redshift_idx[i]] << " # number of images, source redshift" << endl;
		point_image_data[i].write_to_file(outfile);
	}
}

bool QLens::load_point_image_data(string filename)
{
	int i,j,k;
	ifstream data_infile(filename.c_str());

	if (!data_infile.is_open()) data_infile.open(("../data/" + filename).c_str());
	if (!data_infile.is_open()) {
		// Now we look for any directories in the PATH variable that have 'qlens' in the name
		size_t pos = 0;
		size_t pos2 = 0;
		int i, ndirs = 1;
		char *env = getenv("PATH");
		string envstring(env);
		while ((pos = envstring.find(':')) != string::npos) {
			ndirs++;
			envstring.replace(pos,1," ");
		}
		istringstream dirstream(envstring);
		string dirstring[ndirs];
		ndirs=0;
		while (dirstream >> dirstring[ndirs]) ndirs++; // it's possible ndirs will be zero, which is why we recount it here
		for (i=0; i < ndirs; i++) {
			pos=pos2=0;
			if (((pos = dirstring[i].find("qlens")) != string::npos) or ((pos2 = dirstring[i].find("kappa")) != string::npos)) {
				data_infile.open((dirstring[i] + "/" + filename).c_str());
				if (!data_infile.is_open()) data_infile.open((dirstring[i] + "/../data/" + filename).c_str());
				if (data_infile.is_open()) break;
			}
		}
	}

	if (!data_infile.is_open()) { warn("Error: input file '%s' could not be opened",filename.c_str()); return false; }

	int n_datawords;
	vector<string> datawords;

	if (read_data_line(data_infile,datawords,n_datawords)==false) return false;
	if ((n_datawords==2) and (datawords[0]=="zlens")) {
		double zlens;
		if (datastring_convert(datawords[1],zlens)==false) { warn("data file has incorrect format; could not read lens redshift"); return false; }
		if (zlens < 0) { warn("invalid redshift; redshift must be greater than zero"); return false; }
		lens_redshift = zlens;
		if (read_data_line(data_infile,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
	}
	if (n_datawords != 1) { warn("input data file has incorrect format; first line should specify number of source points"); return false; }
	int nsrcfit;
	if (datastring_convert(datawords[0],nsrcfit)==false) { warn("data file has incorrect format; could not read number of source points"); return false; }
	if (nsrcfit <= 0) { warn("number of source points must be greater than zero"); return false; }

	bool time_delay_info_included = true;
	int nn;
	bool zsrc_given_in_datafile = false;
	double zsrc;
	lensvector zeros(0,0);
	bool vary_source_coords = (use_analytic_bestfit_src) ? false : true;
	for (i=0; i < nsrcfit; i++) {
		if (read_data_line(data_infile,datawords,n_datawords)==false) { 
			warn("data file could not be read; unexpected end of file"); 
			clear_image_data();
			return false;
		}
		if ((n_datawords != 1) and (n_datawords != 2)) {
			warn("input data file has incorrect format; invalid number of images for source point %i",i);
			clear_image_data();
			return false;
		}
		if (datastring_convert(datawords[0],nn)==false) {
			warn("data file has incorrect format; could not read number of images for source point %i",i);
			clear_image_data();
			return false;
		}
		if (n_datawords==2) {
			if (datastring_convert(datawords[1],zsrc)==false) {
				warn("data file has incorrect format; could not read redshift for source point %i",i);
				clear_image_data();
				return false;
			}
			zsrc_given_in_datafile = true;
		}
		add_point_source(zsrc,zeros,vary_source_coords);
		if (nn==0) warn("no images in data file for source point %i",i);
		point_image_data[i].input(nn);
		for (j=0; j < nn; j++) {
			if (read_data_line(data_infile,datawords,n_datawords)==false) {
				warn("data file could not be read; unexpected end of file"); 
				clear_image_data();
				return false;
			}
			if ((n_datawords != 5) and (n_datawords != 7)) {
				warn("input data file has incorrect format; wrong number of data entries for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[0],point_image_data[i].pos[j][0])==false) {
				warn("image position x-coordinate has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[1],point_image_data[i].pos[j][1])==false) {
				warn("image position y-coordinate has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[2],point_image_data[i].sigma_pos[j])==false) {
				warn("image position measurement error has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[3],point_image_data[i].flux[j])==false) {
				warn("image flux has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[4],point_image_data[i].sigma_f[j])==false) {
				warn("image flux measurement error has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (n_datawords==7) {
				if (datastring_convert(datawords[5],point_image_data[i].time_delays[j])==false) {
					warn("image time delay has incorrect format; could not read entry for source point %i, image number %i",i,j);
					clear_image_data();
					return false;
				}
				if (datastring_convert(datawords[6],point_image_data[i].sigma_t[j])==false) {
					warn("image time delay has incorrect format; could not read entry for source point %i, image number %i",i,j);
					clear_image_data();
					return false;
				}
			} else {
				time_delay_info_included = false;
				point_image_data[i].time_delays[j] = 0;
				point_image_data[i].sigma_t[j] = 0;
			}
		}
	}
	if (zsrc_given_in_datafile) {
		if (!user_changed_zsource) {
			source_redshift = ptsrc_redshifts[0];
			if (auto_zsource_scaling) {
				reference_source_redshift = ptsrc_redshifts[0];
				for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = 1.0;
			}
			else {
				for (i=0; i < n_lens_redshifts; i++) reference_zfactors[i] = cosmo->kappa_ratio(lens_redshifts[i],source_redshift,reference_source_redshift);
			}
		}
		// if source redshifts are given in the datafile, turn off auto scaling of zsrc_ref so user can experiment with different zsrc values if desired (without changing zsrc_ref)
		auto_zsource_scaling = false;
	}

	sort_image_data_into_redshift_groups();

	int ncombs, max_combinations = -1;
	int n;
	for (i=0; i < n_ptsrc; i++) {
		ncombs = point_image_data[i].n_images * (point_image_data[i].n_images-1) / 2;
		if (ncombs > max_combinations) max_combinations = ncombs;
	}
	double *distsqrs = new double[max_combinations];
	for (i=0; i < n_ptsrc; i++) {
		n=0;
		for (k=0; k < point_image_data[i].n_images; k++) {
			for (j=k+1; j < point_image_data[i].n_images; j++) {
				distsqrs[n] = SQR(point_image_data[i].pos[k][0] - point_image_data[i].pos[j][0]) + SQR(point_image_data[i].pos[k][1] - point_image_data[i].pos[j][1]);
				n++;
			}
		}
		sort(n,distsqrs);
		point_image_data[i].max_distsqr = distsqrs[n-1]; // this saves the maximum distance between any pair of images (useful for image chi-square for missing image penalty values)
	}
	delete[] distsqrs;

	if (time_delay_info_included) {
		double *tdsqrs = new double[max_combinations];
		for (i=0; i < n_ptsrc; i++) {
			n=0;
			for (k=0; k < point_image_data[i].n_images; k++) {
				for (j=k+1; j < point_image_data[i].n_images; j++) {
					tdsqrs[n] = SQR(point_image_data[i].time_delays[k] - point_image_data[i].time_delays[j]);
					n++;
				}
			}
			sort(n,tdsqrs);
			point_image_data[i].max_tdsqr = tdsqrs[n-1]; // this saves the maximum distance between any pair of images (useful for image chi-square for missing image penalty values)
		}
		delete[] tdsqrs;
	}

	//cout << "n_redshift_groups=" << ptsrc_redshift_groups.size()-1 << endl;
	//for (i=0; i < ptsrc_redshift_groups.size(); i++) {
		//cout << ptsrc_redshift_groups[i] << endl;
	//}

	include_imgpos_chisq = true;
	return true;
}

void QLens::sort_image_data_into_redshift_groups()
{
	// In this function we reorganize the image data entries, if necessary, so that image sets with the same source
	// redshift are listed together. This makes it easy to assign image sets with different source planes to
	// different MPI processes in the image plane chi-square. We aren't trying to sort the groups from low to high
	// redshift, only to make sure like redshifts occur in groups.

	int i,k,l,j=0;

	PointImageData *sorted_image_data = new PointImageData[n_ptsrc];
	//double *sorted_redshifts = new double[n_ptsrc_redshifts];
	PointSource **sorted_ptsrc_list = new PointSource*[n_ptsrc];
	int *sorted_ptsrc_redshift_idx = new int[n_ptsrc];
	ptsrc_redshift_groups.clear();
	ptsrc_redshift_groups.push_back(0);
	bool *assigned = new bool[n_ptsrc];
	for (i=0; i < n_ptsrc; i++) assigned[i] = false;
	for (i=0; i < n_ptsrc; i++) {
		if (!assigned[i]) {
			sorted_image_data[j].input(point_image_data[i]);
			sorted_ptsrc_list[j] = ptsrc_list[i];
			sorted_ptsrc_redshift_idx[j] = ptsrc_redshift_idx[i];
			//sorted_redshifts[j] = specific_ptsrc_redshifts[i];
			assigned[i] = true;
			j++;
			for (k=i+1; k < n_ptsrc; k++) {
				if (!assigned[k]) {
					if (ptsrc_redshift_idx[k]==ptsrc_redshift_idx[i]) {
						sorted_image_data[j].input(point_image_data[k]);
						sorted_ptsrc_list[j] = ptsrc_list[k];
						sorted_ptsrc_redshift_idx[j] = ptsrc_redshift_idx[k];
						//sorted_redshifts[j] = specific_ptsrc_redshifts[k];
						assigned[k] = true;
						j++;
					}
				}
			}
			ptsrc_redshift_groups.push_back(j); // this stores the last index for each group of image sets with the same redshift
		}
	}
	if (ptsrc_redshift_groups.size() != (n_ptsrc_redshifts+1)) die("number of sorted redshift groups is wrong (%i vs %i)",ptsrc_redshift_groups.size()-1,n_ptsrc_redshifts);
	delete[] assigned;
	delete[] point_image_data;
	delete[] ptsrc_list;
	delete[] ptsrc_redshift_idx;
	point_image_data = sorted_image_data;
	ptsrc_list = sorted_ptsrc_list;
	ptsrc_redshift_idx = sorted_ptsrc_redshift_idx;
	ptsrclist->input_ptr(ptsrc_list,n_ptsrc);
}

/*
void QLens::remove_image_data(int image_set)
{
	if (image_set >= n_ptsrc) { warn(warnings,"Specified image dataset has not been loaded"); return; }
	if (n_ptsrc==1) { clear_image_data(); return; }
	sourcepts_fit.erase(sourcepts_fit.begin()+image_set);
	PointImageData *new_image_data = new PointImageData[n_sourcepts_fit-1];
	int i,j,k;
	double *new_redshifts, **new_zfactors, ***new_beta_factors;
	new_redshifts = new double[n_sourcepts_fit-1];
	if (n_lens_redshifts > 0) {
		new_zfactors = new double*[n_sourcepts_fit-1];
		new_beta_factors = new double**[n_sourcepts_fit-1];
	}
	for (i=0,j=0; i < n_sourcepts_fit; i++) {
		if (i != image_set) {
			new_image_data[j].input(point_image_data[i]);
			new_redshifts[j] = specific_ptsrc_redshifts[i];
			if (n_lens_redshifts > 0) {
				new_zfactors[j] = specific_ptsrc_zfactors[i];
				new_beta_factors[j] = specific_ptsrc_beta_factors[i];
			}
			j++;
		} else {
			if (n_lens_redshifts > 0) {
				delete[] specific_ptsrc_zfactors[i];
				if (specific_ptsrc_beta_factors[i] != NULL) {
					for (k=0; k < n_lens_redshifts-1; k++) delete[] specific_ptsrc_beta_factors[i][k];
					if (n_lens_redshifts > 1) delete[] specific_ptsrc_beta_factors[i];
				}
			}
		}
	}
	delete[] specific_ptsrc_redshifts;
	delete[] point_image_data;

	n_sourcepts_fit--;
	point_image_data = new_image_data;
	specific_ptsrc_redshifts = new_redshifts;
	if (n_lens_redshifts > 0) {
		delete[] specific_ptsrc_zfactors;
		delete[] specific_ptsrc_beta_factors;
		specific_ptsrc_zfactors = new_zfactors;
		specific_ptsrc_beta_factors = new_beta_factors;
	}

	sort_image_data_into_redshift_groups(); // this updates redshift_groups, in case there are no other image sets that shared the redshift of the one being deleted
}
*/

bool QLens::plot_srcpts_from_image_data(int dataset_number, ofstream* srcfile, const double srcpt_x, const double srcpt_y, const double flux)
{
	// flux is an optional argument; if not specified, its default is -1, meaning fluxes will not be calculated or displayed
	if (dataset_number >= n_ptsrc) { warn("specified dataset number does not exist"); return false; }

	int i,n_srcpts = point_image_data[dataset_number].n_images;
	lensvector *srcpts = new lensvector[n_srcpts];
	double *specific_zfacs = ptsrc_zfactors[ptsrc_redshift_idx[dataset_number]];
	double **specific_betafacs = ptsrc_beta_factors[ptsrc_redshift_idx[dataset_number]];
	for (i=0; i < n_srcpts; i++) {
		find_sourcept(point_image_data[dataset_number].pos[i],srcpts[i],0,specific_zfacs,specific_betafacs);
	}

	if (use_scientific_notation==false) {
		cout << setprecision(6);
		cout << fixed;
	}

	double* time_delays_mod;
	if (include_time_delays) {
		double td_factor;
		time_delays_mod = new double[n_srcpts];
		double min_td_obs, min_td_mod;
		double pot;
		td_factor = cosmo->time_delay_factor_arcsec(lens_redshift,ptsrc_redshifts[ptsrc_redshift_idx[dataset_number]]);
		min_td_obs=1e30;
		min_td_mod=1e30;
		for (i=0; i < n_srcpts; i++) {
			pot = potential(point_image_data[dataset_number].pos[i],specific_zfacs,specific_betafacs);
			time_delays_mod[i] = 0.5*(SQR(point_image_data[dataset_number].pos[i][0] - srcpts[i][0]) + SQR(point_image_data[dataset_number].pos[i][1] - srcpts[i][1])) - pot;
			if (time_delays_mod[i] < min_td_mod) min_td_mod = time_delays_mod[i];
		}
		for (i=0; i < n_srcpts; i++) {
			time_delays_mod[i] -= min_td_mod;
			if (time_delays_mod[i] != 0.0) time_delays_mod[i] *= td_factor; // td_factor contains the cosmological factors and is in units of days
		}
	}

	if (mpi_id==0) {
		cout << "# Source " << dataset_number << " from fit: " << srcpt_x << " " << srcpt_y << endl << endl;
		cout << "#imgpos_x\timgpos_y\tsrcpos_x\tsrcpos_y";
		if (flux != -1.0) cout << "\timage flux";
		if (include_time_delays) cout << "\ttime_delay (days)";
		cout << endl;
		double imgflux;
		for (i=0; i < n_srcpts; i++) {
			cout << point_image_data[dataset_number].pos[i][0] << "\t" << point_image_data[dataset_number].pos[i][1] << "\t" << srcpts[i][0] << "\t" << srcpts[i][1];
			if (srcfile != NULL) (*srcfile) << srcpts[i][0] << "\t" << srcpts[i][1];
			if (flux != -1) {
				imgflux = flux/inverse_magnification(point_image_data[dataset_number].pos[i],0,specific_zfacs,specific_betafacs);
				cout << "\t" << imgflux;
			}
			if (include_time_delays) {
				cout << "\t" << time_delays_mod[i];
			}
			cout << endl;
			if (srcfile != NULL) (*srcfile) << endl;
		}
		cout << endl;
	}
	if (use_scientific_notation==false)
		cout.unsetf(ios_base::floatfield);
	if (include_time_delays) delete[] time_delays_mod;

	delete[] srcpts;
	return true;
}

vector<PtImageDataSet> QLens::export_to_ImageDataSet()
{
	vector<PtImageDataSet> ptimage_data_sets;
	ptimage_data_sets.clear();
	ptimage_data_sets.resize(n_ptsrc);
	int i,j;
	for (i=0; i < n_ptsrc; i++) {
		ptimage_data_sets[i].set_n_images(point_image_data[i].n_images);
		ptimage_data_sets[i].zsrc = ptsrc_redshifts[ptsrc_redshift_idx[i]];
		for (j=0; j < point_image_data[i].n_images; j++) {
			ptimage_data_sets[i].images[j].pos[0] = point_image_data[i].pos[j][0];
			ptimage_data_sets[i].images[j].pos[1] = point_image_data[i].pos[j][1];
			ptimage_data_sets[i].images[j].flux = point_image_data[i].flux[j];
			ptimage_data_sets[i].images[j].td = point_image_data[i].time_delays[j];
			ptimage_data_sets[i].images[j].sigma_pos = point_image_data[i].sigma_pos[j];
			ptimage_data_sets[i].images[j].sigma_flux = point_image_data[i].sigma_f[j];
			ptimage_data_sets[i].images[j].sigma_td = point_image_data[i].sigma_t[j];
		}
	}
	return ptimage_data_sets;
}

/********************************* Functions for weak lensing data (reading, writing, simulating etc.) *********************************/

bool QLens::load_weak_lensing_data(string filename)
{
	int i,j,k;
	ifstream data_infile(filename.c_str());
	if (!data_infile.is_open()) { warn("Error: input file '%s' could not be opened",filename.c_str()); return false; }

	int n_datawords;
	vector<string> datawords;
	int n_wl_sources = 0;
	while (read_data_line(data_infile,datawords,n_datawords)) n_wl_sources++;
	data_infile.close();

	if (n_wl_sources==0) return false;
	data_infile.open(filename.c_str());

	weak_lensing_data.input(n_wl_sources);
	for (j=0; j < n_wl_sources; j++) {
		if (read_data_line(data_infile,datawords,n_datawords)==false) { 
			weak_lensing_data.clear();
			return false;
		}
		weak_lensing_data.id[j] = datawords[0];
		if (datastring_convert(datawords[1],weak_lensing_data.pos[j][0])==false) {
			warn("weak lensing source x-coordinate has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[2],weak_lensing_data.pos[j][1])==false) {
			warn("weak lensing source y-coordinate has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[3],weak_lensing_data.reduced_shear1[j])==false) {
			warn("weak lensing source reduced shear1 has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[4],weak_lensing_data.reduced_shear2[j])==false) {
			warn("weak lensing source reduced shear2 has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[5],weak_lensing_data.sigma_shear1[j])==false) {
			warn("source shear1 measurement error has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[6],weak_lensing_data.sigma_shear2[j])==false) {
			warn("source shear2 measurement error has incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
		if (datastring_convert(datawords[7],weak_lensing_data.zsrc[j])==false) {
			warn("source redshift thas incorrect format; could not read entry for source ID %s",weak_lensing_data.id[j].c_str());
			weak_lensing_data.clear();
			return false;
		}
	}

	// For starters, let's not store specific_ptsrc_zfactors, we'll just calculate them when we evaluate the chi-square. We can always put
	// the specific_ptsrc_zfactors in to save time later.
	//if (n_lens_redshifts > 0) {
		//for (i=0; i < n_sourcepts_fit; i++) {
			//for (j=0; j < n_lens_redshifts; j++) {
				//weak_lensing_data.specific_ptsrc_zfactors[i][j] = cosmo->kappa_ratio(lens_redshifts[j],specific_ptsrc_redshifts[i],reference_source_redshift);
			//}
		//}
	//}

	// I don't think beta factors should matter for weak lensing, but you can check this later
	//if (n_lens_redshifts > 1) {
		//for (i=0; i < n_sourcepts_fit; i++) {
			//for (j=1; j < n_lens_redshifts; j++) {
				//specific_ptsrc_beta_factors[i][j-1] = new double[j];
				//if (include_recursive_lensing) {
					//for (k=0; k < j; k++) specific_ptsrc_beta_factors[i][j-1][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j],specific_ptsrc_redshifts[i]); // from cosmo->cpp
				//} else {
					//for (k=0; k < j; k++) specific_ptsrc_beta_factors[i][j-1][k] = 0;
				//}
			//}
		//}
	//}

	include_weak_lensing_chisq = true;
	return true;
}

void QLens::add_simulated_weak_lensing_data(const string id, lensvector &sourcept, const double zsrc)
{
	double *zfacs = new double[n_lens_redshifts];

	for (int i=0; i < n_lens_redshifts; i++) {
		zfacs[i] = cosmo->kappa_ratio(lens_redshifts[i],zsrc,reference_source_redshift);
	}
	double shear1, shear2;
	reduced_shear_components(sourcept,shear1,shear2,0,zfacs);
	shear1 += sim_err_shear*NormalDeviate();
	shear2 += sim_err_shear*NormalDeviate();
	weak_lensing_data.add_source(id,sourcept,shear1,shear2,sim_err_shear,sim_err_shear,zsrc);
	if (!include_weak_lensing_chisq) include_weak_lensing_chisq = true;
	delete[] zfacs;
}

void QLens::add_weak_lensing_data_from_random_sources(const int num_sources, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, const double r_exclude)
{
	int wl_index = weak_lensing_data.n_sources;
	lensvector src;
	string id_string;
	double zsrc;
	for (int i=0; i < num_sources; i++) {
		stringstream idstr;
		idstr << wl_index;
		idstr >> id_string;

		do {
			src[0] = (xmax-xmin)*RandomNumber() + xmin;
			src[1] = (ymax-ymin)*RandomNumber() + ymin;
		} while (sqrt(SQR(src[0]-grid_xcenter)+SQR(src[1]-grid_ycenter)) <= r_exclude);
		zsrc = (zmax-zmin)*RandomNumber() + zmin; // redshift
		add_simulated_weak_lensing_data(id_string,src,zsrc);

		wl_index++;
	}
}

bool QLens::read_data_line(ifstream& data_infile, vector<string>& datawords, int &n_datawords)
{
	static const int n_characters = 512;
	int pos;
	string word;
	n_datawords = 0;
	datawords.clear();
	do {
		char dataline[n_characters];
		data_infile.getline(dataline,n_characters);
		if (data_infile.gcount()==n_characters-1) {
			warn("the number of characters in a single line cannot exceed %i",n_characters);
			return false;
		}
		if ((data_infile.rdstate() & ifstream::eofbit) != 0) {
			return false;
		}
		string linestring(dataline);
		if ((pos = linestring.find("data_info: ")) != string::npos) {
			data_info = linestring.substr(pos+11);
		} else {
			remove_comments(linestring);
			istringstream datastream0(linestring.c_str());
			while (datastream0 >> word) datawords.push_back(word);
			n_datawords = datawords.size();
		}
	} while (n_datawords==0); // skip lines that are blank or only have comments
	remove_equal_sign_datafile(datawords,n_datawords);
	return true;
}

void QLens::remove_comments(string& instring)
{
	string instring_copy(instring);
	instring.clear();
	size_t comment_pos = instring_copy.find("#");
	if (comment_pos != string::npos) {
		instring = instring_copy.substr(0,comment_pos);
	} else instring = instring_copy;
}

void QLens::remove_equal_sign_datafile(vector<string>& datawords, int &n_datawords)
{
	int pos;
	if ((pos = datawords[0].find('=')) != string::npos) {
		// there's an equal sign in the first word, so remove it and separate into two datawords
		datawords.push_back("");
		for (int i=n_datawords-1; i > 0; i--) datawords[i+1] = datawords[i];
		datawords[1] = datawords[0].substr(pos+1);
		datawords[0] = datawords[0].substr(0,pos);
		n_datawords++;
	}
	else if ((n_datawords == 3) and (datawords[1]=="="))
	{
		// there's an equal sign in the second of three datawords (indicating a parameter assignment), so remove it and reduce to two datawords
		string word1,word2;
		word1=datawords[0]; word2=datawords[2];
		datawords.clear();
		datawords.push_back(word1);
		datawords.push_back(word2);
		n_datawords = 2;
	}
}

bool QLens::datastring_convert(const string& instring, int& outvar)
{
	datastream.clear(); // resets the error flags
	datastream.str(string()); // clears the stringstream
	datastream << instring;
	if (datastream >> outvar) return true;
	else return false;
}

bool QLens::datastring_convert(const string& instring, double& outvar)
{
	datastream.clear(); // resets the error flags
	datastream.str(string()); // clears the stringstream
	datastream << instring;
	if (datastream >> outvar) return true;
	else return false;
}

void QLens::clear_sourcepts()
{
	while (n_ptsrc > 0) {
		remove_point_source(n_ptsrc-1);
	}
}

void QLens::clear_image_data()
{
	while (n_ptsrc > 0) {
		remove_point_source(n_ptsrc-1);
	}
}

void QLens::print_image_data(bool include_errors)
{
	if (mpi_id==0) {
		double zsrc;
		for (int i=0; i < n_ptsrc; i++) {
			zsrc = ptsrc_redshifts[ptsrc_redshift_idx[i]];
			cout << "Source " << i << ": zsrc=" << zsrc;
			if ((n_lens_redshifts==0) or ((n_lens_redshifts==1) and (zsrc==lens_redshifts[0]))) cout << " (unlensed)";
			cout << endl;
			point_image_data[i].print_list(include_errors,use_scientific_notation);
		}
	}
}

void PointImageData::input(const int &nn)
{
	n_images = nn;
	pos = new lensvector[n_images];
	flux = new double[n_images];
	time_delays = new double[n_images];
	sigma_pos = new double[n_images];
	sigma_f = new double[n_images];
	sigma_t = new double[n_images];
	use_in_chisq = new bool[n_images];
	max_distsqr = 1e30;
}

void PointImageData::input(const PointImageData& imgs_in)
{
	if (n_images != 0) {
		// delete arrays so we can re-create them
		delete[] pos;
		delete[] flux;
		delete[] time_delays;
		delete[] sigma_pos;
		delete[] sigma_f;
		delete[] sigma_t;
		delete[] use_in_chisq;
	}
	n_images = imgs_in.n_images;
	pos = new lensvector[n_images];
	flux = new double[n_images];
	time_delays = new double[n_images];
	sigma_pos = new double[n_images];
	sigma_f = new double[n_images];
	sigma_t = new double[n_images];
	use_in_chisq = new bool[n_images];
	for (int i=0; i < n_images; i++) {
		pos[i] = imgs_in.pos[i];
		flux[i] = imgs_in.flux[i];
		time_delays[i] = imgs_in.time_delays[i];
		sigma_pos[i] = imgs_in.sigma_pos[i];
		sigma_f[i] = imgs_in.sigma_f[i];
		sigma_t[i] = imgs_in.sigma_t[i];
		use_in_chisq[i] = true;
	}
	max_distsqr = imgs_in.max_distsqr;
}

void PointImageData::input(const int &nn, image* images, double* sigma_pos_in, double* sigma_flux_in, const double sigma_td_in, bool* include, bool include_time_delays)
{
	// this function is used to store simulated data
	int n_images_include=0;
	for (int i=0; i < nn; i++) if (include[i]) n_images_include++;
	n_images = n_images_include;
	pos = new lensvector[n_images];
	flux = new double[n_images];
	time_delays = new double[n_images];
	sigma_pos = new double[n_images];
	sigma_f = new double[n_images];
	sigma_t = new double[n_images];
	use_in_chisq = new bool[n_images];
	int j=0;
	for (int i=0; i < nn; i++) {
		if (!include[i]) continue;
		pos[j] = images[i].pos;
		flux[j] = images[i].flux;
		if (include_time_delays) {
			time_delays[j] = images[i].td;
			sigma_t[j] = sigma_td_in;
		}
		else {
			time_delays[j] = 0;
			sigma_t[j] = 0;
		}
		sigma_pos[j] = sigma_pos_in[j];
		sigma_f[j] = sigma_flux_in[j];
		use_in_chisq[j] = true;
		j++;
	}
	max_distsqr = 1e30;
}


void PointImageData::add_image(lensvector& pos_in, const double sigma_pos_in, const double flux_in, const double sigma_f_in, const double time_delay_in, const double sigma_t_in)
{
	int n_images_new = n_images+1;
	if (n_images != 0) {
		lensvector *new_pos = new lensvector[n_images_new];
		double *new_flux = new double[n_images_new];
		double *new_time_delays = new double[n_images_new];
		double *new_sigma_pos = new double[n_images_new];
		double *new_sigma_f = new double[n_images_new];
		double *new_sigma_t = new double[n_images_new];
		bool *new_use_in_chisq = new bool[n_images_new];
		for (int i=0; i < n_images; i++) {
			new_pos[i][0] = pos[i][0];
			new_pos[i][1] = pos[i][1];
			new_flux[i] = flux[i];
			new_time_delays[i] = time_delays[i];
			new_sigma_pos[i] = sigma_pos[i];
			new_sigma_f[i] = sigma_f[i];
			new_sigma_t[i] = sigma_t[i];
			new_use_in_chisq[i] = use_in_chisq[i];
		}
		delete[] pos;
		delete[] flux;
		delete[] time_delays;
		delete[] sigma_pos;
		delete[] sigma_f;
		delete[] sigma_t;
		delete[] use_in_chisq;
		pos = new_pos;
		flux = new_flux;
		time_delays = new_time_delays;
		sigma_pos = new_sigma_pos;
		sigma_f = new_sigma_f;
		sigma_t = new_sigma_t;
		use_in_chisq = new_use_in_chisq;
		n_images++;
	} else {
		n_images = 1;
		pos = new lensvector[n_images];
		flux = new double[n_images];
		time_delays = new double[n_images];
		sigma_pos = new double[n_images];
		sigma_f = new double[n_images];
		sigma_t = new double[n_images];
		use_in_chisq = new bool[n_images];
	}
	pos[n_images-1][0] = pos_in[0];
	pos[n_images-1][1] = pos_in[1];
	flux[n_images-1] = flux_in;
	time_delays[n_images-1] = time_delay_in;
	sigma_pos[n_images-1] = sigma_pos_in;
	sigma_f[n_images-1] = sigma_f_in;
	sigma_t[n_images-1] = sigma_t_in;
	use_in_chisq[n_images-1] = true;
}

bool PointImageData::set_use_in_chisq(int image_i, bool use_in_chisq_in)
{
	if (image_i >= n_images) return false;
	use_in_chisq[image_i] = use_in_chisq_in;
	return true;
}

void PointImageData::print_list(bool print_errors, bool use_sci)
{
	if (n_images==0) {
		cout << "# no image data available" << endl << endl;
	} else {
		if (use_sci==false) {
			cout << setprecision(6);
			cout << fixed;
		}
		if (print_errors) cout << "#        pos_x(arcsec)\tpos_y(arcsec)\tsig_pos\t\tflux\t\tsig_flux";
		else cout << "#        pos_x\t\tpos_y\t\tflux";
		if (sigma_t[0] != 0) {
			if (print_errors) cout << "\ttime_delay(days)\tsigma_t\n";
			else cout << "\ttime_delay\n";
		}
		else cout << endl;
		for (int i=0; i < n_images; i++) {
			cout << "Image " << i << ": " << pos[i][0] << "\t" << pos[i][1];
			if (print_errors) cout << "\t" << sigma_pos[i];
			cout << "\t" << flux[i];
			if (print_errors) cout << "\t" << sigma_f[i];
			if (sigma_t[0] != 0) {
				cout << "\t" << time_delays[i];
				if (print_errors) cout << "\t\t" << sigma_t[i];
			}
			if (!use_in_chisq[i]) cout << "   (excluded from chisq)";
			cout << endl;
		}
		cout << endl;
		if (use_sci==false)
			cout.unsetf(ios_base::floatfield);
	}
}

void PointImageData::write_to_file(ofstream &outfile)
{
	for (int i=0; i < n_images; i++) {
		outfile << pos[i][0] << " " << pos[i][1];
		outfile << " " << sigma_pos[i];
		outfile << " " << flux[i];
		outfile << " " << sigma_f[i];
		if (sigma_t[0] != 0) {
			outfile << " " << time_delays[i];
			outfile << " " << sigma_t[i];
		}
		outfile << endl;
	}
}

PointImageData::~PointImageData()
{
	if (n_images != 0) {
		delete[] pos;
		delete[] flux;
		delete[] time_delays;
		delete[] sigma_pos;
		delete[] sigma_f;
		delete[] sigma_t;
		delete[] use_in_chisq;
	}
}

void WeakLensingData::input(const int &nn)
{
	if (n_sources != 0) {
		// delete arrays so we can re-create them
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
	}
	n_sources = nn;
	id = new string[n_sources];
	pos = new lensvector[n_sources];
	reduced_shear1 = new double[n_sources];
	reduced_shear2 = new double[n_sources];
	sigma_shear1 = new double[n_sources];
	sigma_shear2 = new double[n_sources];
	zsrc = new double[n_sources];
}

void WeakLensingData::input(const WeakLensingData& wl_in)
{
	if (n_sources != 0) {
		// delete arrays so we can re-create them
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
	}
	n_sources = wl_in.n_sources;
	id = new string[n_sources];
	pos = new lensvector[n_sources];
	reduced_shear1 = new double[n_sources];
	reduced_shear2 = new double[n_sources];
	sigma_shear1 = new double[n_sources];
	sigma_shear2 = new double[n_sources];
	zsrc = new double[n_sources];
	for (int i=0; i < n_sources; i++) {
		id[i] = wl_in.id[i];
		pos[i] = wl_in.pos[i];
		reduced_shear1[i] = wl_in.reduced_shear1[i];
		reduced_shear2[i] = wl_in.reduced_shear2[i];
		sigma_shear1[i] = wl_in.sigma_shear1[i];
		sigma_shear2[i] = wl_in.sigma_shear2[i];
		zsrc[i] = wl_in.zsrc[i];
	}
}

void WeakLensingData::add_source(const string id_in, lensvector& pos_in, const double g1_in, const double g2_in, const double g1_err_in, const double g2_err_in, const double zsrc_in)
{
	int n_sources_new = n_sources+1;
	if (n_sources != 0) {
		string *new_id = new string[n_sources_new];
		lensvector *new_pos = new lensvector[n_sources_new];
		double *new_reduced_shear1 = new double[n_sources_new];
		double *new_reduced_shear2 = new double[n_sources_new];
		double *new_sigma_shear1 = new double[n_sources_new];
		double *new_sigma_shear2 = new double[n_sources_new];
		double *new_zsrc = new double[n_sources_new];
		for (int i=0; i < n_sources; i++) {
			new_id[i] = id[i];
			new_pos[i][0] = pos[i][0];
			new_pos[i][1] = pos[i][1];
			new_reduced_shear1[i] = reduced_shear1[i];
			new_reduced_shear2[i] = reduced_shear2[i];
			new_sigma_shear1[i] = sigma_shear1[i];
			new_sigma_shear2[i] = sigma_shear2[i];
			new_zsrc[i] = zsrc[i];
		}
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
		id = new_id;
		pos = new_pos;
		reduced_shear1 = new_reduced_shear1;
		reduced_shear2 = new_reduced_shear2;
		sigma_shear1 = new_sigma_shear1;
		sigma_shear2 = new_sigma_shear2;
		zsrc = new_zsrc;
		n_sources++;
	} else {
		n_sources = 1;
		id = new string[n_sources];
		pos = new lensvector[n_sources];
		reduced_shear1 = new double[n_sources];
		reduced_shear2 = new double[n_sources];
		sigma_shear1 = new double[n_sources];
		sigma_shear2 = new double[n_sources];
		zsrc = new double[n_sources];
	}
	id[n_sources-1] = id_in;
	pos[n_sources-1][0] = pos_in[0];
	pos[n_sources-1][1] = pos_in[1];
	reduced_shear1[n_sources-1] = g1_in;
	reduced_shear2[n_sources-1] = g2_in;
	sigma_shear1[n_sources-1] = g1_err_in;
	sigma_shear2[n_sources-1] = g2_err_in;
	zsrc[n_sources-1] = zsrc_in;
}

void WeakLensingData::print_list(bool use_sci)
{
	if (use_sci==false) {
		cout << setprecision(6);
		cout << fixed;
	}
	cout << "# id\tpos_x(arcsec)\tpos_y(arcsec)\tg1\t\tg2\t\tsig_g1\t\tsig_g2\t\tzsrc\n";
	for (int i=0; i < n_sources; i++) {
		cout << id[i] << "\t" << pos[i][0] << "\t" << pos[i][1];
		cout << "\t" << reduced_shear1[i];
		cout << "\t" << reduced_shear2[i];
		cout << "\t" << sigma_shear1[i];
		cout << "\t" << sigma_shear2[i];
		cout << "\t" << zsrc[i] << endl;
	}
	cout << endl;
	if (use_sci==false)
		cout.unsetf(ios_base::floatfield);
}

void WeakLensingData::write_to_file(string filename)
{
	ofstream outfile(filename.c_str());
	for (int i=0; i < n_sources; i++) {
		outfile << id[i] << " ";
		outfile << pos[i][0] << " " << pos[i][1];
		outfile << " " << reduced_shear1[i];
		outfile << " " << reduced_shear2[i];
		outfile << " " << sigma_shear1[i];
		outfile << " " << sigma_shear2[i];
		outfile << " " << zsrc[i];
		outfile << endl;
	}
}

WeakLensingData::~WeakLensingData()
{
	if (n_sources != 0) {
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
	}
}

void WeakLensingData::clear()
{
	if (n_sources != 0) {
		delete[] id;
		delete[] pos;
		delete[] reduced_shear1;
		delete[] reduced_shear2;
		delete[] sigma_shear1;
		delete[] sigma_shear2;
		delete[] zsrc;
	}
	n_sources = 0;
}

/******************************************** Functions for lens model fitting ******************************************/

bool QLens::initialize_fitmodel(const bool running_fit_in)
{
	if (source_fit_mode == Point_Source) {
		if (((!include_weak_lensing_chisq) or (weak_lensing_data.n_sources==0)) and ((n_ptsrc==0) or (point_image_data==NULL))) {
			warn("image data points have not been defined");
			return false;
		}
		if ((point_image_data==NULL) and (include_imgpos_chisq)) { warn("cannot evaluate image position chi-square; no image data have been defined"); return false; }
		else if ((point_image_data==NULL) and (include_flux_chisq)) { warn("cannot evaluate image flux chi-square; no image data have been defined"); return false; }
		else if ((point_image_data==NULL) and (include_time_delay_chisq)) { warn("cannot evaluate image time delay chi-square; no image data have been defined"); return false; }
	} else {
		if (imgdata_list==NULL) { warn("image data pixels have not been loaded"); return false; }
		if (source_fit_mode==Shapelet_Source) {
			bool found_invertible_source = false;
			for (int i=0; i < n_sb; i++) {
				if ((sb_list[i]->sbtype==SHAPELET) or (sb_list[i]->sbtype==MULTI_GAUSSIAN_EXPANSION)) {
					found_invertible_source = true;
					break; // currently only one shapelet source supported
				}
			}
			if ((!found_invertible_source) and (n_ptsrc==0)) { warn("no invertible source object or source points found"); return false; }
		}
	}
	if (fitmodel != NULL) delete fitmodel;
	fitmodel = new QLens(this);
	fitmodel->use_ansi_characters = running_fit_in;
	//fitmodel->set_gridcenter(grid_xcenter,grid_ycenter);

	int i,j,k;
	if (n_lens_redshifts > 0) {
		fitmodel->reference_zfactors = new double[n_lens_redshifts];
		fitmodel->default_zsrc_beta_factors = new double*[n_lens_redshifts-1];
		fitmodel->lens_redshifts = new double[n_lens_redshifts];
		fitmodel->lens_redshift_idx = new int[nlens];
		fitmodel->zlens_group_size = new int[n_lens_redshifts];
		fitmodel->zlens_group_lens_indx = new int*[n_lens_redshifts];
		for (i=0; i < n_lens_redshifts; i++) {
			fitmodel->reference_zfactors[i] = reference_zfactors[i];
			fitmodel->lens_redshifts[i] = lens_redshifts[i];
			fitmodel->zlens_group_size[i] = zlens_group_size[i];
			fitmodel->zlens_group_lens_indx[i] = new int[zlens_group_size[i]];
			//cout << "Redshift group " << i << " (z=" << fitmodel->lens_redshifts[i] << ", " << fitmodel->zlens_group_size[i] << " lenses)\n";
			for (j=0; j < zlens_group_size[i]; j++) {
				fitmodel->zlens_group_lens_indx[i][j] = zlens_group_lens_indx[i][j];
				//cout << fitmodel->zlens_group_lens_indx[i][j] << " ";
			}
			//cout << endl;
		}
		for (i=0; i < n_lens_redshifts-1; i++) {
			fitmodel->default_zsrc_beta_factors[i] = new double[i+1];
			for (j=0; j < i+1; j++) fitmodel->default_zsrc_beta_factors[i][j] = default_zsrc_beta_factors[i][j];
		}
		for (j=0; j < nlens; j++) fitmodel->lens_redshift_idx[j] = lens_redshift_idx[j];
	}

	fitmodel->nlens = nlens;
	fitmodel->lens_list = new LensProfile*[nlens];
	for (i=0; i < nlens; i++) {
		switch (lens_list[i]->get_lenstype()) {
			case KSPLINE:
				fitmodel->lens_list[i] = new LensProfile(lens_list[i]); break;
			case sple_LENS:
				fitmodel->lens_list[i] = new SPLE_Lens((SPLE_Lens*) lens_list[i]); break;
			case dpie_LENS:
				fitmodel->lens_list[i] = new dPIE_Lens((dPIE_Lens*) lens_list[i]); break;
			case nfw:
				fitmodel->lens_list[i] = new NFW((NFW*) lens_list[i]); break;
			case TRUNCATED_nfw:
				fitmodel->lens_list[i] = new Truncated_NFW((Truncated_NFW*) lens_list[i]); break;
			case CORED_nfw:
				fitmodel->lens_list[i] = new Cored_NFW((Cored_NFW*) lens_list[i]); break;
			case HERNQUIST:
				fitmodel->lens_list[i] = new Hernquist((Hernquist*) lens_list[i]); break;
			case EXPDISK:
				fitmodel->lens_list[i] = new ExpDisk((ExpDisk*) lens_list[i]); break;
			case SHEAR:
				fitmodel->lens_list[i] = new Shear((Shear*) lens_list[i]); break;
			case MULTIPOLE:
				fitmodel->lens_list[i] = new Multipole((Multipole*) lens_list[i]); break;
			case CORECUSP:
				fitmodel->lens_list[i] = new CoreCusp((CoreCusp*) lens_list[i]); break;
			case SERSIC_LENS:
				fitmodel->lens_list[i] = new SersicLens((SersicLens*) lens_list[i]); break;
			case DOUBLE_SERSIC_LENS:
				fitmodel->lens_list[i] = new DoubleSersicLens((DoubleSersicLens*) lens_list[i]); break;
			case CORED_SERSIC_LENS:
				fitmodel->lens_list[i] = new Cored_SersicLens((Cored_SersicLens*) lens_list[i]); break;
			case TOPHAT_LENS:
				fitmodel->lens_list[i] = new TopHatLens((TopHatLens*) lens_list[i]); break;
			case PTMASS:
				fitmodel->lens_list[i] = new PointMass((PointMass*) lens_list[i]); break;
			case SHEET:
				fitmodel->lens_list[i] = new MassSheet((MassSheet*) lens_list[i]); break;
			case DEFLECTION:
				fitmodel->lens_list[i] = new Deflection((Deflection*) lens_list[i]); break;
			case TABULATED:
				fitmodel->lens_list[i] = new Tabulated_Model((Tabulated_Model*) lens_list[i]); break;
			case QTABULATED:
				fitmodel->lens_list[i] = new QTabulated_Model((QTabulated_Model*) lens_list[i]); break;
			default:
				die("lens type not supported for fitting");
		}
		fitmodel->lens_list[i]->qlens = fitmodel; // point to the fitmodel, since the cosmology may be varied (by varying H0, e.g.)
	}
	fitmodel->lenslist->input_ptr(fitmodel->lens_list,nlens);

	//if ((source_fit_mode != Point_Source) and (n_pixellated_src == 0) and ((n_image_prior) or (n_ptsrc > 0))) {
		 //add_pixellated_source(source_redshift); // THIS IS UGLY. There must be a better way to do this
	//}
	if ((n_pixellated_src==0) and ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source))) add_pixellated_source(source_redshift,0);
	else if (n_extended_src_redshifts == 0) {
		add_new_extended_src_redshift(source_redshift,-1,false);
	}

	if (n_extended_src_redshifts > 0) {
		fitmodel->n_extended_src_redshifts = n_extended_src_redshifts;
		fitmodel->extended_src_redshifts = new double[n_extended_src_redshifts];
		if (n_lens_redshifts > 0) {
			fitmodel->extended_src_zfactors = new double*[n_extended_src_redshifts];
			fitmodel->extended_src_beta_factors = new double**[n_extended_src_redshifts];
		}
		for (i=0; i < n_extended_src_redshifts; i++) {
			fitmodel->extended_src_redshifts[i] = extended_src_redshifts[i];
			if (n_lens_redshifts > 0) {
				fitmodel->extended_src_zfactors[i] = new double[n_lens_redshifts];
				fitmodel->extended_src_beta_factors[i] = new double*[n_lens_redshifts-1];
				for (j=0; j < n_lens_redshifts; j++) fitmodel->extended_src_zfactors[i][j] = extended_src_zfactors[i][j];
				for (j=0; j < n_lens_redshifts-1; j++) {
					fitmodel->extended_src_beta_factors[i][j] = new double[j+1];
					for (k=0; k < j+1; k++) fitmodel->extended_src_beta_factors[i][j][k] = extended_src_beta_factors[i][j][k];
				}
			}
		}
	}
	if (n_image_pixel_grids > 0) {
		fitmodel->n_image_pixel_grids = n_image_pixel_grids;
		fitmodel->image_pixel_grids = new ImagePixelGrid*[n_image_pixel_grids];
		for (i=0; i < n_image_pixel_grids; i++) {
			fitmodel->image_pixel_grids[i] = NULL;
		}
	}
	
	fitmodel->n_assigned_masks = n_assigned_masks;
	if (n_assigned_masks > 0) {
		fitmodel->assigned_mask = new int[n_assigned_masks];
		for (i=0; i < n_assigned_masks; i++) {
			fitmodel->assigned_mask[i] = assigned_mask[i];
		}
	}

	if (n_ptsrc_redshifts > 0) {
		fitmodel->n_ptsrc_redshifts = n_ptsrc_redshifts;
		fitmodel->ptsrc_redshifts = new double[n_ptsrc_redshifts];
		if (n_lens_redshifts > 0) {
			fitmodel->ptsrc_zfactors = new double*[n_ptsrc_redshifts];
			fitmodel->ptsrc_beta_factors = new double**[n_ptsrc_redshifts];
		}
		for (i=0; i < n_ptsrc_redshifts; i++) {
			fitmodel->ptsrc_redshifts[i] = ptsrc_redshifts[i];
			if (n_lens_redshifts > 0) {
				fitmodel->ptsrc_zfactors[i] = new double[n_lens_redshifts];
				fitmodel->ptsrc_beta_factors[i] = new double*[n_lens_redshifts-1];
				for (j=0; j < n_lens_redshifts; j++) fitmodel->ptsrc_zfactors[i][j] = ptsrc_zfactors[i][j];
				for (j=0; j < n_lens_redshifts-1; j++) {
					fitmodel->ptsrc_beta_factors[i][j] = new double[j+1];
					for (k=0; k < j+1; k++) fitmodel->ptsrc_beta_factors[i][j][k] = ptsrc_beta_factors[i][j][k];
				}
			}
		}
		fitmodel->ptsrc_redshift_groups = ptsrc_redshift_groups;
	}
	fitmodel->n_ptsrc = n_ptsrc;
	if (n_ptsrc > 0) {
		fitmodel->point_image_data = point_image_data;
		fitmodel->ptsrc_list = new PointSource*[n_ptsrc];
		fitmodel->ptsrc_redshift_idx = new int[n_ptsrc];
		for (i=0; i < n_ptsrc; i++) {
			fitmodel->ptsrc_redshift_idx[i] = ptsrc_redshift_idx[i];
			fitmodel->ptsrc_list[i] = new PointSource(fitmodel);
			fitmodel->ptsrc_list[i]->copy_ptsrc_data(ptsrc_list[i]);
			fitmodel->ptsrc_list[i]->entry_number = i;
		}
	}
	fitmodel->ptsrclist->input_ptr(fitmodel->ptsrc_list,n_ptsrc);

	fitmodel->n_sb = n_sb;
	if (n_sb > 0) {
		fitmodel->sb_list = new SB_Profile*[n_sb];
		fitmodel->sbprofile_redshift_idx = new int[n_sb];
		fitmodel->sbprofile_imggrid_idx = new int[n_sb];
		fitmodel->sbprofile_band_number = new int[n_sb];
		for (i=0; i < n_sb; i++) {
			switch (sb_list[i]->get_sbtype()) {
				case GAUSSIAN:
					fitmodel->sb_list[i] = new Gaussian((Gaussian*) sb_list[i]); break;
				case SERSIC:
					fitmodel->sb_list[i] = new Sersic((Sersic*) sb_list[i]); break;
				case CORE_SERSIC:
					fitmodel->sb_list[i] = new CoreSersic((CoreSersic*) sb_list[i]); break;
				case CORED_SERSIC:
					fitmodel->sb_list[i] = new Cored_Sersic((Cored_Sersic*) sb_list[i]); break;
				case DOUBLE_SERSIC:
					fitmodel->sb_list[i] = new DoubleSersic((DoubleSersic*) sb_list[i]); break;
				case sple:
					fitmodel->sb_list[i] = new SPLE((SPLE*) sb_list[i]); break;
				case dpie:
					fitmodel->sb_list[i] = new dPIE((dPIE*) sb_list[i]); break;
				case nfw_SOURCE:
					fitmodel->sb_list[i] = new NFW_Source((NFW_Source*) sb_list[i]); break;
				case SB_MULTIPOLE:
					fitmodel->sb_list[i] = new SB_Multipole((SB_Multipole*) sb_list[i]); break;
				case SHAPELET:
					fitmodel->sb_list[i] = new Shapelet((Shapelet*) sb_list[i]); break;
				case MULTI_GAUSSIAN_EXPANSION:
					fitmodel->sb_list[i] = new MGE((MGE*) sb_list[i]); break;
				case TOPHAT:
					fitmodel->sb_list[i] = new TopHat((TopHat*) sb_list[i]); break;
				default:
					die("surface brightness profile type not supported for fitting");
			}
			fitmodel->sbprofile_redshift_idx[i] = sbprofile_redshift_idx[i];
			fitmodel->sbprofile_imggrid_idx[i] = sbprofile_imggrid_idx[i];
			fitmodel->sbprofile_band_number[i] = sbprofile_band_number[i];
			fitmodel->sb_list[i]->qlens = fitmodel; // point to the fitmodel
		}
	}
	fitmodel->srclist->input_ptr(fitmodel->sb_list,n_sb);

	fitmodel->n_pixellated_src = n_pixellated_src;
	if (n_pixellated_src > 0) {
		fitmodel->srcgrids = new ModelParams*[n_pixellated_src];
		fitmodel->delaunay_srcgrids = new DelaunaySourceGrid*[n_pixellated_src];
		fitmodel->cartesian_srcgrids = new CartesianSourceGrid*[n_pixellated_src];
		fitmodel->pixellated_src_redshift_idx = new int[n_pixellated_src];
		fitmodel->pixellated_src_band = new int[n_pixellated_src];
		for (i=0; i < n_pixellated_src; i++) {
			fitmodel->pixellated_src_redshift_idx[i] = pixellated_src_redshift_idx[i];
			fitmodel->pixellated_src_band[i] = pixellated_src_band[i];
			fitmodel->delaunay_srcgrids[i] = new DelaunaySourceGrid(fitmodel,pixellated_src_band[i]);
			fitmodel->cartesian_srcgrids[i] = new CartesianSourceGrid(fitmodel,pixellated_src_band[i]);
			fitmodel->delaunay_srcgrids[i]->entry_number = i;
			fitmodel->cartesian_srcgrids[i]->entry_number = i;
			if (source_fit_mode==Delaunay_Source) {
				fitmodel->delaunay_srcgrids[i]->copy_pixsrc_data(delaunay_srcgrids[i]);
				fitmodel->srcgrids[i] = fitmodel->delaunay_srcgrids[i];
			} else if (source_fit_mode==Cartesian_Source) {
				fitmodel->cartesian_srcgrids[i]->copy_pixsrc_data(cartesian_srcgrids[i]);
				fitmodel->srcgrids[i] = fitmodel->cartesian_srcgrids[i];
			}
		}
	}
	fitmodel->pixsrclist->input_ptr(fitmodel->srcgrids,n_pixellated_src);

	fitmodel->n_pixellated_lens = n_pixellated_lens;
	if (n_pixellated_lens > 0) {
		fitmodel->lensgrids = new LensPixelGrid*[n_pixellated_lens];
		fitmodel->pixellated_lens_redshift_idx = new int[n_pixellated_lens];
		for (i=0; i < n_pixellated_lens; i++) {
			fitmodel->pixellated_lens_redshift_idx[i] = pixellated_lens_redshift_idx[i];
			fitmodel->lensgrids[i] = new LensPixelGrid(fitmodel,pixellated_lens_redshift_idx[i]);
			fitmodel->lensgrids[i]->copy_pixlens_data(lensgrids[i]);
		}
	}

	fitmodel->n_psf = n_psf;
	if (n_psf > 0) {
		fitmodel->psf_list = new PSF*[n_psf];
		for (i=0; i < n_psf; i++) {
			fitmodel->psf_list[i] = new PSF(fitmodel);
			fitmodel->psf_list[i]->copy_psf_data(psf_list[i]);
		}
	}

	fitmodel->borrowed_image_data = true; // this is so we don't have to needlessly copy the data and masks every time we do a fit
	if (source_fit_mode != Point_Source) {
		fitmodel->n_data_bands = n_data_bands;
		fitmodel->n_model_bands = n_model_bands;
		fitmodel->imgdata_list = imgdata_list;
		for (int i=0; i < n_data_bands; i++) fitmodel->load_pixel_grid_from_data(i);
	}
	fitmodel->imgdatalist->input_ptr(fitmodel->imgdata_list,n_data_bands);

	for (i=0; i < nlens; i++) {
		// if the lens is anchored to another lens, re-anchor so that it points to the corresponding
		// lens in fitmodel (the lens whose parameters will be varied)
		if (fitmodel->lens_list[i]->center_anchored==true) fitmodel->lens_list[i]->anchor_center_to_lens(lens_list[i]->get_center_anchor_number());
		if (fitmodel->lens_list[i]->anchor_special_parameter==true) {
			LensProfile *parameter_anchor_lens = fitmodel->lens_list[lens_list[i]->get_special_parameter_anchor_number()];
			fitmodel->lens_list[i]->assign_special_anchored_parameters(parameter_anchor_lens,1,false);
		}
		for (j=0; j < fitmodel->lens_list[i]->get_n_params(); j++) {
			if (fitmodel->lens_list[i]->anchor_parameter_to_lens[j]==true) {
				LensProfile *parameter_anchor_lens = fitmodel->lens_list[lens_list[i]->parameter_anchor_lens[j]->lens_number];
				int paramnum = fitmodel->lens_list[i]->parameter_anchor_paramnum[j];
				fitmodel->lens_list[i]->assign_anchored_parameter(j,paramnum,true,true,lens_list[i]->parameter_anchor_ratio[j],lens_list[i]->parameter_anchor_exponent[j],parameter_anchor_lens);
			} else if (fitmodel->lens_list[i]->anchor_parameter_to_source[j]==true) {
				SB_Profile *parameter_anchor_source = fitmodel->sb_list[lens_list[i]->parameter_anchor_source[j]->sb_number];
				int paramnum = fitmodel->lens_list[i]->parameter_anchor_paramnum[j];
				fitmodel->lens_list[i]->assign_anchored_parameter(j,paramnum,true,true,lens_list[i]->parameter_anchor_ratio[j],lens_list[i]->parameter_anchor_exponent[j],parameter_anchor_source);
			}
		}
	}
	for (i=0; i < n_sb; i++) {
		if (fitmodel->sb_list[i]->center_anchored_to_lens==true) fitmodel->sb_list[i]->anchor_center_to_lens(fitmodel->lens_list, sb_list[i]->get_center_anchor_number());
		if (fitmodel->sb_list[i]->center_anchored_to_source==true) fitmodel->sb_list[i]->anchor_center_to_source(fitmodel->sb_list, sb_list[i]->get_center_anchor_number());
		if (fitmodel->sb_list[i]->center_anchored_to_ptsrc==true) fitmodel->sb_list[i]->anchor_center_to_ptsrc(fitmodel->ptsrc_list, sb_list[i]->get_center_anchor_number());
		for (j=0; j < fitmodel->sb_list[i]->get_n_params(); j++) {
			if (fitmodel->sb_list[i]->anchor_parameter_to_source[j]==true) {
				SB_Profile *parameter_anchor_source = fitmodel->sb_list[sb_list[i]->parameter_anchor_source[j]->sb_number];
				int paramnum = fitmodel->sb_list[i]->parameter_anchor_paramnum[j];
				fitmodel->sb_list[i]->assign_anchored_parameter(j,paramnum,true,true,sb_list[i]->parameter_anchor_ratio[j],sb_list[i]->parameter_anchor_exponent[j],parameter_anchor_source);
			} 
		}
	}

	fitmodel->fitmethod = fitmethod;
	//fitmodel->n_fit_parameters = n_fit_parameters;
	fitmodel->lensmodel_fit_parameters = lensmodel_fit_parameters;
	fitmodel->srcmodel_fit_parameters = srcmodel_fit_parameters;
	fitmodel->pixsrc_fit_parameters = pixsrc_fit_parameters;
	fitmodel->pixlens_fit_parameters = pixlens_fit_parameters;
	fitmodel->ptsrc_fit_parameters = ptsrc_fit_parameters;
	fitmodel->psf_fit_parameters = psf_fit_parameters;
	fitmodel->cosmo_fit_parameters = cosmo_fit_parameters;
	//if ((fitmethod!=POWELL) and (fitmethod!=SIMPLEX)) fitmodel->setup_limits();

	if (open_chisq_logfile) {
		string logfile_str = fit_output_dir + "/" + fit_output_filename + ".log";
		if (group_id==0) {
			if (group_num > 0) {
				// if there is more than one MPI group evaluating the likelihood, output a separate file for each group
				stringstream groupstream;
				string groupstr;
				groupstream << group_num;
				groupstream >> groupstr;
				logfile_str += "." + groupstr;
			}
			fitmodel->logfile.open(logfile_str.c_str());
			fitmodel->logfile << setprecision(10);
		}
	}

	fitmodel->update_parameter_list();
	if ((source_fit_mode != Point_Source) and (!redo_lensing_calculations_before_inversion)) {
		for (int i=0; i < n_extended_src_redshifts; i++) fitmodel->image_pixel_grids[i]->redo_lensing_calculations(); 
	}

	return true;
}

void QLens::update_anchored_parameters_and_redshift_data()
{
	for (int i=0; i < n_sb; i++) {
		if ((sb_list[i]->center_anchored_to_lens) or (sb_list[i]->center_anchored_to_source) or (sb_list[i]->center_anchored_to_ptsrc)) {
			sb_list[i]->update_anchor_center();
		}
		sb_list[i]->update_anchored_parameters();
	}
	for (int i=0; i < nlens; i++) {
		if (lens_list[i]->center_anchored) lens_list[i]->update_anchor_center();
		if (lens_list[i]->anchor_special_parameter) lens_list[i]->update_special_anchored_params();
		if (lens_list[i]->at_least_one_param_anchored) lens_list[i]->update_anchored_parameters();
	}
	update_lens_redshift_data();
	reset_grid();
}

bool QLens::update_lens_centers_from_pixsrc_coords()
{
	bool updated = false;
	for (int i=0; i < nlens; i++) {
		if (lens_list[i]->transform_center_coords_to_pixsrc_frame) {
			lens_list[i]->update_center_from_pixsrc_coords();
			updated = true;
		}
	}
	return updated;
}

void QLens::update_zfactors_and_betafactors()
{
	// This must be done anytime the cosmology has changed
	int i,j,k;
	if (n_lens_redshifts > 0) {
		for (j=0; j < n_lens_redshifts; j++) reference_zfactors[j] = cosmo->kappa_ratio(lens_redshifts[j],source_redshift,reference_source_redshift);
		for (j=0; j < n_lens_redshifts-1; j++) {
			for (k=0; k < j+1; k++) default_zsrc_beta_factors[j][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j+1],source_redshift);
		}
	}

	if (n_extended_src_redshifts > 0) {
		for (i=0; i < n_extended_src_redshifts; i++) {
			if (n_lens_redshifts > 0) {
				for (j=0; j < n_lens_redshifts; j++) extended_src_zfactors[i][j] = cosmo->kappa_ratio(lens_redshifts[j],extended_src_redshifts[i],reference_source_redshift);
				for (j=0; j < n_lens_redshifts-1; j++) {
					for (k=0; k < j+1; k++) extended_src_beta_factors[i][j][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j+1],extended_src_redshifts[i]);
				}
			}
		}
	}

	if (n_ptsrc_redshifts > 0) {
		for (i=0; i < n_ptsrc_redshifts; i++) {
			if (n_lens_redshifts > 0) {
				for (j=0; j < n_lens_redshifts; j++) ptsrc_zfactors[i][j] = cosmo->kappa_ratio(lens_redshifts[j],ptsrc_redshifts[i],reference_source_redshift);
				for (j=0; j < n_lens_redshifts-1; j++) {
					for (k=0; k < j+1; k++) ptsrc_beta_factors[i][j][k] = cosmo->calculate_beta_factor(lens_redshifts[k],lens_redshifts[j+1],ptsrc_redshifts[i]);
				}
			}
		}
	}
}

double QLens::update_model(const double* params)
{
	bool status = true;
	double log_penalty_prior = 0;
	if (ellipticity_gradient) contours_overlap = false; // we will test to see whether new parameters cause density contours to overlap
	int i, index=0;
	for (i=0; i < nlens; i++) {
		lens_list[i]->update_fit_parameters(params,index,status);
	}
	for (i=0; i < n_sb; i++) {
		sb_list[i]->update_fit_parameters(params,index,status);
	}
	for (i=0; i < n_pixellated_src; i++) {
		if (srcgrids[i] != NULL) srcgrids[i]->update_fit_parameters(params,index);
	}
	for (i=0; i < n_pixellated_lens; i++) {
		lensgrids[i]->update_fit_parameters(params,index);
	}
	for (i=0; i < n_ptsrc; i++) {
		ptsrc_list[i]->update_fit_parameters(params,index);
	}
	for (i=0; i < n_psf; i++) {
		psf_list[i]->update_fit_parameters(params,index);
	}

	cosmo->update_fit_parameters(params,index);
	// *NOTE*: Maybe consider putting the cosmological parameters at the very FRONT of the parameter list? Then the cosmology is updated before updating the lenses
	/*
	// the below lines should now be taken care of in the  Cosmology class, since it now has a pointer to qlens (code is in cosmo.cpp under update_meta_parameters)
	if (cosmo->get_n_vary_params() > 0) {
		update_zfactors_and_betafactors();
		for (i=0; i < nlens; i++) {
			if ((!lens_list[i]->at_least_one_param_anchored) and (!lens_list[i]->anchor_special_parameter)) lens_list[i]->update_meta_parameters(); // if the cosmology has changed, update cosmology info and any parameters that depend on them (unless there are anchored parameters, in which case it will be done below)
		}
	}
	*/

	update_fit_parameters(params,index); // this is for the parameters that are part of the QLens class
	if (status==false) log_penalty_prior = 1e30;
	if ((ellipticity_gradient) and (contours_overlap)) {
		log_penalty_prior += contour_overlap_log_penalty_prior;
		//warn("contours overlap in ellipticity gradient model");
	}
	update_anchored_parameters_and_redshift_data();

	if (index != param_list->nparams) die("Index (%i) didn't go through all the fit parameters (ntot=%i), indicating a lens model mismatch",index,param_list->nparams);
	return log_penalty_prior;
}

void QLens::update_prior_limits(const double* lower, const double* upper, const bool* changed_limits)
{
	int i, index=0;
	for (i=0; i < nlens; i++) {
		lens_list[i]->update_limits(lower,upper,changed_limits,index);
	}
	for (i=0; i < n_sb; i++) {
		sb_list[i]->update_limits(lower,upper,changed_limits,index);
	}
	for (i=0; i < n_pixellated_src; i++) {
		if (srcgrids[i] != NULL) srcgrids[i]->update_limits(lower,upper,changed_limits,index);
	}
	for (i=0; i < n_pixellated_lens; i++) {
		lensgrids[i]->update_limits(lower,upper,changed_limits,index);
	}
	for (i=0; i < n_ptsrc; i++) {
		ptsrc_list[i]->update_limits(lower,upper,changed_limits,index);
	}
	for (i=0; i < n_psf; i++) {
		psf_list[i]->update_limits(lower,upper,changed_limits,index);
	}

	cosmo->update_limits(lower,upper,changed_limits,index);

	update_limits(lower,upper,changed_limits,index); // this is for the parameters that are part of the QLens class

	if (index != param_list->nparams) die("Index (%i) didn't go through all the fit parameters (ntot=%i) when updating prior limits, indicating a lens model mismatch",index,param_list->nparams);
}



void QLens::find_analytic_srcpos(lensvector *beta_i)
{
	if (nlens==0) {
		warn("no lens models have been defined; cannot find analytic best-fit source point");
		return;
	}
	// Note: beta_i needs to have the same size as the number of image sets being fit, or else a segmentation fault will occur
	int i,j;
	lensvector beta_ji;
	lensmatrix mag, magsqr;
	lensmatrix amatrix, ainv;
	lensvector bvec;
	lensmatrix jac;

	double siginv, src_norm;
	double *specific_zfacs;
	double **specific_betafacs;
	for (i=0; i < n_ptsrc; i++) {
		amatrix[0][0] = amatrix[0][1] = amatrix[1][0] = amatrix[1][1] = 0;
		bvec[0] = bvec[1] = 0;
		beta_i[i][0] = beta_i[i][1] = 0;
		src_norm=0;
		specific_zfacs = ptsrc_zfactors[ptsrc_redshift_idx[i]];
		specific_betafacs = ptsrc_beta_factors[ptsrc_redshift_idx[i]];
		for (j=0; j < point_image_data[i].n_images; j++) {
			if (point_image_data[i].use_in_chisq[j]) {
				if (use_magnification_in_chisq) {
					sourcept_jacobian(point_image_data[i].pos[j],beta_ji,jac,0,specific_zfacs,specific_betafacs);
					mag = jac.inverse();
					lensmatsqr(mag,magsqr);
					siginv = 1.0/(SQR(point_image_data[i].sigma_pos[j]) + syserr_pos*syserr_pos);
					amatrix[0][0] += magsqr[0][0]*siginv;
					amatrix[1][0] += magsqr[1][0]*siginv;
					amatrix[0][1] += magsqr[0][1]*siginv;
					amatrix[1][1] += magsqr[1][1]*siginv;
					bvec[0] += (magsqr[0][0]*beta_ji[0] + magsqr[0][1]*beta_ji[1])*siginv;
					bvec[1] += (magsqr[1][0]*beta_ji[0] + magsqr[1][1]*beta_ji[1])*siginv;
				} else {
					find_sourcept(point_image_data[i].pos[j],beta_ji,0,specific_zfacs,specific_betafacs);
					siginv = 1.0/(SQR(point_image_data[i].sigma_pos[j]) + syserr_pos*syserr_pos);
					beta_i[i][0] += beta_ji[0]*siginv;
					beta_i[i][1] += beta_ji[1]*siginv;
					src_norm += siginv;
				}
			}
		}
		if (use_magnification_in_chisq) {
			if (amatrix.invert(ainv)==false) {
				warn(warnings,"magnification matrix is singular; cannot use magnification to solve for analytic best-fit source points");
				return;
			}
			beta_i[i] = ainv*bvec;
		} else {
			beta_i[i][0] /= src_norm;
			beta_i[i][1] /= src_norm;
		}
	}
	return;
}

void QLens::set_analytic_sourcepts(const bool verbal)
{
	lensvector *srcpts = new lensvector[n_ptsrc];
	find_analytic_srcpos(srcpts);
	for (int i=0; i < n_ptsrc; i++) {
		ptsrc_list[i]->update_srcpos(srcpts[i]);
		if ((verbal) and (mpi_id==0)) {
			cout << "analytic best-fit source";
			if (n_ptsrc > 1) cout << " " << i;
			cout << ": " << srcpts[i][0] << " " << srcpts[i][1] << endl;
		}
	}
	delete[] srcpts;
}

double QLens::chisq_pos_source_plane()
{
	int i,j;
	double chisq=0;
	int n_images_hi=0;
	lensvector delta_beta, delta_theta;
	lensmatrix mag, magsqr;
	lensmatrix amatrix, ainv;
	lensvector bvec;
	lensmatrix jac;
	lensvector src_bf;
	lensvector *beta;

	for (i=0; i < n_ptsrc; i++) {
		if (point_image_data[i].n_images > n_images_hi) n_images_hi = point_image_data[i].n_images;
	}
	double* mag00 = new double[n_images_hi];
	double* mag11 = new double[n_images_hi];
	double* mag01 = new double[n_images_hi];
	lensvector* beta_ji = new lensvector[n_images_hi];

	double sigsq, signormfac, siginv, src_norm;
	if (syserr_pos == 0.0) signormfac = 0.0; // signormfac is the correction to chi-square to account for unknown systematic error
	int redshift_idx;
	for (i=0; i < n_ptsrc; i++) {
		redshift_idx = ptsrc_redshift_idx[i];
		amatrix[0][0] = amatrix[0][1] = amatrix[1][0] = amatrix[1][1] = 0;
		bvec[0] = bvec[1] = 0;
		src_bf[0] = src_bf[1] = 0;
		src_norm=0;
		for (j=0; j < point_image_data[i].n_images; j++) {
			if (point_image_data[i].use_in_chisq[j]) {
				if (use_magnification_in_chisq) {
					sourcept_jacobian(point_image_data[i].pos[j],beta_ji[j],jac,0,ptsrc_zfactors[redshift_idx],ptsrc_beta_factors[redshift_idx]);
					mag = jac.inverse();
					mag00[j] = mag[0][0];
					mag01[j] = mag[0][1];
					mag11[j] = mag[1][1];

					if (use_analytic_bestfit_src) {
						lensmatsqr(mag,magsqr);
						siginv = 1.0/(SQR(point_image_data[i].sigma_pos[j]) + syserr_pos*syserr_pos);
						amatrix[0][0] += magsqr[0][0]*siginv;
						amatrix[1][0] += magsqr[1][0]*siginv;
						amatrix[0][1] += magsqr[0][1]*siginv;
						amatrix[1][1] += magsqr[1][1]*siginv;
						bvec[0] += (magsqr[0][0]*beta_ji[j][0] + magsqr[0][1]*beta_ji[j][1])*siginv;
						bvec[1] += (magsqr[1][0]*beta_ji[j][0] + magsqr[1][1]*beta_ji[j][1])*siginv;
					}
				} else {
					find_sourcept(point_image_data[i].pos[j],beta_ji[j],0,ptsrc_zfactors[redshift_idx],ptsrc_beta_factors[redshift_idx]);
					if (use_analytic_bestfit_src) {
						siginv = 1.0/(SQR(point_image_data[i].sigma_pos[j]) + syserr_pos*syserr_pos);
						src_bf[0] += beta_ji[j][0]*siginv;
						src_bf[1] += beta_ji[j][1]*siginv;
						src_norm += siginv;
					}
				}
			}
		}
		if (use_analytic_bestfit_src) {
			if (use_magnification_in_chisq) {
				if (amatrix.invert(ainv)==false) return 1e30;
				src_bf = ainv*bvec;
			} else {
				src_bf[0] /= src_norm;
				src_bf[1] /= src_norm;
			}
			beta = &src_bf;
			ptsrc_list[i]->pos[0] = src_bf[0]; // even though it's not being used directly, set the point source object's source position for consistency's sake
			ptsrc_list[i]->pos[1] = src_bf[1];
		} else {
			beta = &ptsrc_list[i]->pos;
		}

		for (j=0; j < point_image_data[i].n_images; j++) {
			if (point_image_data[i].use_in_chisq[j]) {
				delta_beta[0] = (*beta)[0] - beta_ji[j][0];
				delta_beta[1] = (*beta)[1] - beta_ji[j][1];
				sigsq = SQR(point_image_data[i].sigma_pos[j]);
				if (syserr_pos != 0.0) {
					 signormfac = 2*log(1.0 + syserr_pos*syserr_pos/sigsq);
					 sigsq += syserr_pos*syserr_pos;
				}
				if (use_magnification_in_chisq) {
					delta_theta[0] = mag00[j] * delta_beta[0] + mag01[j] * delta_beta[1];
					delta_theta[1] = mag01[j] * delta_beta[0] + mag11[j] * delta_beta[1];
					chisq += delta_theta.sqrnorm() / sigsq + signormfac;
				} else {
					chisq += delta_beta.sqrnorm() / sigsq + signormfac;
				}
			}
		}
	}
	delete[] mag00;
	delete[] mag11;
	delete[] mag01;
	delete[] beta_ji;
	if ((group_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	return chisq;
}

double QLens::chisq_pos_image_plane()
{
	int n_redshift_groups = ptsrc_redshift_groups.size()-1;
	if (n_redshift_groups != n_ptsrc_redshifts) die("wrong number of ptsrc redshift groups");
	int mpi_chunk=n_redshift_groups, mpi_start=0;
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

	if (group_np > 1) {
		if (group_np > n_redshift_groups) die("Number of MPI processes per group cannot be greater than number of source planes in data being fit");
		mpi_chunk = n_redshift_groups / group_np;
		mpi_start = group_id*mpi_chunk;
		if (group_id == group_np-1) mpi_chunk += (n_redshift_groups % group_np); // assign the remainder elements to the last mpi process
	}

	if (use_analytic_bestfit_src) set_analytic_sourcepts();

	double chisq=0, chisq_part=0;

	int n_images, n_tot_images=0, n_tot_images_part=0;
	double sigsq, signormfac, chisq_each_srcpt, dist;
	if (syserr_pos == 0.0) signormfac = 0.0; // signormfac is the correction to chi-square to account for unknown systematic error
	int i,j,k,m,n;
	int redshift_idx;
	for (m=mpi_start; m < mpi_start + mpi_chunk; m++) {
		redshift_idx = ptsrc_redshift_idx[ptsrc_redshift_groups[m]];
		create_grid(false,ptsrc_zfactors[redshift_idx],ptsrc_beta_factors[redshift_idx],m);
		for (i=ptsrc_redshift_groups[m]; i < ptsrc_redshift_groups[m+1]; i++) {
			if (ptsrc_redshift_idx[i] != redshift_idx) die("AWW fuck the redshift groups aren't sorted right");
			chisq_each_srcpt = 0;
			image *img = get_images(ptsrc_list[i]->pos, n_images, false);
			n_visible_images = n_images;
			bool *ignore = new bool[n_images];
			for (j=0; j < n_images; j++) ignore[j] = false;

			for (j=0; j < n_images; j++) {
				if ((!ignore[j]) and (abs(img[j].mag) < chisq_magnification_threshold)) {
					ignore[j] = true;
					n_visible_images--;
				}
				if ((chisq_imgsep_threshold > 0) and (!ignore[j])) {
					for (k=j+1; k < n_images; k++) {
						if (!ignore[k]) {
							dist = sqrt(SQR(img[k].pos[0] - img[j].pos[0]) + SQR(img[k].pos[1] - img[j].pos[1]));
							if (dist < chisq_imgsep_threshold) {
								ignore[k] = true;
								n_visible_images--;
							}
						}
					}
				}
			}

			n_tot_images_part += n_visible_images;
			if ((n_images_penalty==true) and (n_visible_images > point_image_data[i].n_images)) {
				chisq_part += 1e30;
				continue;
			}

			int n_dists = n_visible_images*point_image_data[i].n_images;
			double *distsqrs = new double[n_dists];
			int *data_k = new int[n_dists];
			int *model_j = new int[n_dists];
			n=0;
			for (k=0; k < point_image_data[i].n_images; k++) {
				for (j=0; j < n_images; j++) {
					if (ignore[j]) continue;
					distsqrs[n] = SQR(point_image_data[i].pos[k][0] - img[j].pos[0]) + SQR(point_image_data[i].pos[k][1] - img[j].pos[1]);
					data_k[n] = k;
					model_j[n] = j;
					n++;
				}
			}

			if (n != n_dists) die("count of all data-model image combinations does not equal expected number (%i vs %i)",n,n_dists);
			sort(n_dists,distsqrs,data_k,model_j);
			int *closest_image_j = new int[point_image_data[i].n_images];
			int *closest_image_k = new int[n_images];
			double *closest_distsqrs = new double[point_image_data[i].n_images];
			for (k=0; k < point_image_data[i].n_images; k++) closest_image_j[k] = -1;
			for (j=0; j < n_images; j++) closest_image_k[j] = -1;
			int m=0;
			int mmax = dmin(n_visible_images,point_image_data[i].n_images);
			for (n=0; n < n_dists; n++) {
				if ((closest_image_j[data_k[n]] == -1) and (closest_image_k[model_j[n]] == -1)) {
					closest_image_j[data_k[n]] = model_j[n];
					closest_image_k[model_j[n]] = data_k[n];
					closest_distsqrs[data_k[n]] = distsqrs[n];
					m++;
					if (m==mmax) n = n_dists; // force loop to exit
				}
			}

			for (k=0; k < point_image_data[i].n_images; k++) {
				sigsq = SQR(point_image_data[i].sigma_pos[k]);
				if (syserr_pos != 0.0) {
					 signormfac = 2*log(1.0 + syserr_pos*syserr_pos/sigsq);
					 sigsq += syserr_pos*syserr_pos;
				}
				if (closest_image_j[k] != -1) {
					if (point_image_data[i].use_in_chisq[k]) {
						chisq_each_srcpt += closest_distsqrs[k]/sigsq + signormfac;
					}
				} else {
					// add a penalty value to chi-square for not reproducing this data image; the distance is twice the maximum distance between any pair of images
					chisq_each_srcpt += 4*point_image_data[i].max_distsqr/sigsq + signormfac;
				}
			}
			chisq_part += chisq_each_srcpt;
			delete[] ignore;
			delete[] distsqrs;
			delete[] data_k;
			delete[] model_j;
			delete[] closest_image_j;
			delete[] closest_image_k;
			delete[] closest_distsqrs;
		}
	}
#ifdef USE_MPI
	//cout << "chisq_part=" << chisq_part << ", group_id=" << group_id << endl;
	MPI_Allreduce(&chisq_part, &chisq, 1, MPI_DOUBLE, MPI_SUM, sub_comm);
	MPI_Allreduce(&n_tot_images_part, &n_tot_images, 1, MPI_INT, MPI_SUM, sub_comm);
	MPI_Comm_free(&sub_comm);
#else
	chisq = chisq_part;
	n_tot_images = n_tot_images_part;
#endif

	if ((group_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	n_visible_images = n_tot_images; // save the total number of visible images produced
	return chisq;
}

double QLens::chisq_pos_image_plane_diagnostic(const bool verbose, const bool output_residuals_to_file, double &rms_imgpos_err, int &n_matched_images, const string output_filename)
{
	int n_redshift_groups = ptsrc_redshift_groups.size()-1;
	int mpi_chunk=n_redshift_groups, mpi_start=0;
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

	if (group_np > 1) {
		if (group_np > n_redshift_groups) die("Number of MPI processes per group cannot be greater than number of source planes in data being fit");
		mpi_chunk = n_redshift_groups / group_np;
		mpi_start = group_id*mpi_chunk;
		if (group_id == group_np-1) mpi_chunk += (n_redshift_groups % group_np); // assign the remainder elements to the last mpi process
	}

	if (use_analytic_bestfit_src) set_analytic_sourcepts(verbose);

	double chisq=0, chisq_part=0, rms_part=0;
	int n_images, n_tot_images=0, n_tot_images_part=0, n_matched_images_part=0;
	double sigsq, signormfac, chisq_each_srcpt, n_matched_images_each_srcpt, rms_err_each_srcpt, dist;
	rms_imgpos_err = 0;
	n_matched_images = 0;
	vector<double> closest_chivals, closest_xvals_model, closest_yvals_model, closest_xvals_data, closest_yvals_data;

	if (syserr_pos == 0.0) signormfac = 0.0; // signormfac is the correction to chi-square to account for unknown systematic error
	int i,j,k,m,n;
	int redshift_idx;
	for (m=mpi_start; m < mpi_start + mpi_chunk; m++) {
		redshift_idx = ptsrc_redshift_idx[ptsrc_redshift_groups[m]];
		create_grid(false,ptsrc_zfactors[redshift_idx],ptsrc_beta_factors[redshift_idx],m);
		if ((mpi_id==0) and (verbose)) cout << endl << "zsrc=" << ptsrc_redshifts[redshift_idx] << ": grid = (" << (grid_xcenter-grid_xlength/2) << "," << (grid_xcenter+grid_xlength/2) << ") x (" << (grid_ycenter-grid_ylength/2) << "," << (grid_ycenter+grid_ylength/2) << ")" << endl;
		for (i=ptsrc_redshift_groups[m]; i < ptsrc_redshift_groups[m+1]; i++) {
			chisq_each_srcpt = 0;
			n_matched_images_each_srcpt = 0;
			rms_err_each_srcpt = 0;
			image *img = get_images(ptsrc_list[i]->pos, n_images, false);
			n_visible_images = n_images;
			bool *ignore = new bool[n_images];
			for (j=0; j < n_images; j++) ignore[j] = false;

			for (j=0; j < n_images; j++) {
				if ((!ignore[j]) and (abs(img[j].mag) < chisq_magnification_threshold)) {
					ignore[j] = true;
					n_visible_images--;
				}
				if ((chisq_imgsep_threshold > 0) and (!ignore[j])) {
					for (k=j+1; k < n_images; k++) {
						if (!ignore[k]) {
							dist = sqrt(SQR(img[k].pos[0] - img[j].pos[0]) + SQR(img[k].pos[1] - img[j].pos[1]));
							if (dist < chisq_imgsep_threshold) {
								ignore[k] = true;
								n_visible_images--;
							}
						}
					}
				}
			}

			n_tot_images_part += n_visible_images;
			if ((n_images_penalty==true) and (n_visible_images > point_image_data[i].n_images)) {
				chisq_part += 1e30;
				if ((mpi_id==0) and (verbose)) cout << "nimg_penalty incurred for source " << i << " (# model images = " << n_visible_images << ", # data images = " << point_image_data[i].n_images << ")" << endl;
			}

			int n_dists = n_visible_images*point_image_data[i].n_images;
			double *distsqrs = new double[n_dists];
			int *data_k = new int[n_dists];
			int *model_j = new int[n_dists];
			n=0;
			for (k=0; k < point_image_data[i].n_images; k++) {
				for (j=0; j < n_images; j++) {
					if (ignore[j]) continue;
					distsqrs[n] = SQR(point_image_data[i].pos[k][0] - img[j].pos[0]) + SQR(point_image_data[i].pos[k][1] - img[j].pos[1]);
					data_k[n] = k;
					model_j[n] = j;
					n++;
				}
			}
			if (n != n_dists) die("count of all data-model image combinations does not equal expected number (%i vs %i)",n,n_dists);
			sort(n_dists,distsqrs,data_k,model_j);
			int *closest_image_j = new int[point_image_data[i].n_images];
			int *closest_image_k = new int[n_images];
			double *closest_distsqrs = new double[point_image_data[i].n_images];
			for (k=0; k < point_image_data[i].n_images; k++) closest_image_j[k] = -1;
			for (j=0; j < n_images; j++) closest_image_k[j] = -1;
			int m=0;
			int mmax = dmin(n_visible_images,point_image_data[i].n_images);
			for (n=0; n < n_dists; n++) {
				if ((closest_image_j[data_k[n]] == -1) and (closest_image_k[model_j[n]] == -1)) {
					closest_image_j[data_k[n]] = model_j[n];
					closest_image_k[model_j[n]] = data_k[n];
					closest_distsqrs[data_k[n]] = distsqrs[n];
					m++;
					if (m==mmax) n = n_dists; // force loop to exit
				}
			}

			double chisq_this_img, chi_x, chi_y;
			int this_src_nimgs = point_image_data[i].n_images;
			for (k=0; k < point_image_data[i].n_images; k++) {
				sigsq = SQR(point_image_data[i].sigma_pos[k]);
				if (syserr_pos != 0.0) {
					 signormfac = 2*log(1.0 + syserr_pos*syserr_pos/sigsq);
					 sigsq += syserr_pos*syserr_pos;
				}
				if ((mpi_id==0) and (verbose)) cout << "source " << i << ", image " << k << ": ";
				if (closest_image_j[k] != -1) {
					if (point_image_data[i].use_in_chisq[k]) {
						rms_err_each_srcpt += closest_distsqrs[k];
						n_matched_images_each_srcpt++;
						chisq_this_img = closest_distsqrs[k]/sigsq + signormfac;
						chi_x = (img[closest_image_j[k]].pos[0]-point_image_data[i].pos[k][0])/sqrt(sigsq);
						chi_y = (img[closest_image_j[k]].pos[1]-point_image_data[i].pos[k][1])/sqrt(sigsq);
						closest_chivals.push_back(abs(chi_x));
						closest_xvals_model.push_back(img[closest_image_j[k]].pos[0]);
						closest_yvals_model.push_back(img[closest_image_j[k]].pos[1]);
						closest_xvals_data.push_back(point_image_data[i].pos[k][0]);
						closest_yvals_data.push_back(point_image_data[i].pos[k][1]);
						closest_chivals.push_back(abs(chi_y));
						closest_xvals_model.push_back(img[closest_image_j[k]].pos[0]);
						closest_yvals_model.push_back(img[closest_image_j[k]].pos[1]);
						closest_xvals_data.push_back(point_image_data[i].pos[k][0]);
						closest_yvals_data.push_back(point_image_data[i].pos[k][1]);

						if ((mpi_id==0) and (verbose)) cout << "chi_x=" << chi_x << ", chi_y=" << chi_y << ", chisq=" << chisq_this_img << " matched to (" << img[closest_image_j[k]].pos[0] << "," << img[closest_image_j[k]].pos[1] << ")" << endl << flush;
						chisq_each_srcpt += chisq_this_img;
					}
					else if ((mpi_id==0) and (verbose)) cout << "ignored in chisq,  matched to (" << img[closest_image_j[k]].pos[0] << "," << img[closest_image_j[k]].pos[1] << ")" << endl << flush;
				} else {
					// add a penalty value to chi-square for not reproducing this data image; the distance is twice the maximum distance between any pair of images
					chisq_this_img += 4*point_image_data[i].max_distsqr/sigsq + signormfac;
					if ((mpi_id==0) and (verbose)) cout << "chisq=" << chisq_this_img << " (not matched to model image)" << endl << flush;
					chisq_each_srcpt += chisq_this_img;
				}
			}
			if ((mpi_id==0) and (verbose)) {
				for (k=0; k < n_images; k++) {
					if (closest_image_k[k] == -1) cout << "EXTRA IMAGE: source " << i << ", model image " << k << " (" << img[k].pos[0] << "," << img[k].pos[1] << "), magnification = " << img[k].mag << endl << flush;
				}
			}

			chisq_part += chisq_each_srcpt;
			rms_part += rms_err_each_srcpt;
			n_matched_images_part += n_matched_images_each_srcpt;
			delete[] ignore;
			delete[] distsqrs;
			delete[] data_k;
			delete[] model_j;
			delete[] closest_image_j;
			delete[] closest_image_k;
			delete[] closest_distsqrs;
		}
	}
	if ((mpi_id==0) and (verbose)) cout << endl;
#ifdef USE_MPI
	MPI_Allreduce(&chisq_part, &chisq, 1, MPI_DOUBLE, MPI_SUM, sub_comm);
	MPI_Allreduce(&rms_part, &rms_imgpos_err, 1, MPI_DOUBLE, MPI_SUM, sub_comm);
	MPI_Allreduce(&n_tot_images_part, &n_tot_images, 1, MPI_INT, MPI_SUM, sub_comm);
	MPI_Allreduce(&n_matched_images_part, &n_matched_images, 1, MPI_INT, MPI_SUM, sub_comm);
#else
	chisq = chisq_part;
	n_tot_images = n_tot_images_part;
	n_matched_images = n_matched_images_part;
	rms_imgpos_err = rms_part;
#endif
	rms_imgpos_err = sqrt(rms_imgpos_err/n_matched_images);
	double *chi_all_images = new double[2*n_matched_images];
	double *model_xvals_all_images = new double[2*n_matched_images];
	double *model_yvals_all_images = new double[2*n_matched_images];
	double *data_xvals_all_images = new double[2*n_matched_images];
	double *data_yvals_all_images = new double[2*n_matched_images];
	int *nmatched_parts = new int[group_np];

#ifdef USE_MPI
	int id=0;
	nmatched_parts[group_id] = 2*n_matched_images_part;
	for (id=0; id < group_np; id++) {
		MPI_Bcast(nmatched_parts+id, 1, MPI_INT, id, sub_comm);
	}
	int indx=0;
	for (id=0; id < group_np; id++) {
		if (group_id==id) {
			for (i=0; i < nmatched_parts[id]; i++) {
				chi_all_images[indx+i] = closest_chivals[i];
				model_xvals_all_images[indx+i] = closest_xvals_model[i];
				model_yvals_all_images[indx+i] = closest_yvals_model[i];
				data_xvals_all_images[indx+i] = closest_xvals_data[i];
				data_yvals_all_images[indx+i] = closest_yvals_data[i];
			}
		}

		indx += nmatched_parts[id];
	}
	indx=0;
	for (id=0; id < group_np; id++) {
		MPI_Bcast(chi_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		MPI_Bcast(model_xvals_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		MPI_Bcast(model_yvals_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		MPI_Bcast(data_xvals_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		MPI_Bcast(data_yvals_all_images+indx, nmatched_parts[id], MPI_DOUBLE, id, sub_comm);
		indx += nmatched_parts[id];
	}
	MPI_Comm_free(&sub_comm);
#else
	for (i=0; i < 2*n_matched_images; i++) {
		chi_all_images[i] = closest_chivals[i];
		model_xvals_all_images[i] = closest_xvals_model[i];
		model_yvals_all_images[i] = closest_yvals_model[i];
		data_xvals_all_images[i] = closest_xvals_data[i];
		data_yvals_all_images[i] = closest_yvals_data[i];
	}
#endif
	if ((group_id==0) and (output_residuals_to_file)) {
		sort(2*n_matched_images,chi_all_images,model_xvals_all_images,model_yvals_all_images,data_xvals_all_images,data_yvals_all_images);
		double frac;
		ofstream outfile(output_filename.c_str());
		outfile << "#chi fraction(>chi) model_x model_y data_x data_y" << endl;
		for (i=0; i < 2*n_matched_images; i++) {
			j = 2*n_matched_images-i-1;
			frac = ((double) j)/(2.0*n_matched_images);
			outfile << chi_all_images[i] << " " << frac << " " << model_xvals_all_images[i] << " " << model_yvals_all_images[i] << " " << data_xvals_all_images[i] << " " << data_yvals_all_images[i] << endl;
		}
	}

	if ((mpi_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	n_visible_images = n_tot_images; // save the total number of visible images produced
	if ((mpi_id==0) and (verbose)) cout << "Number of matched image pairs = " << n_matched_images <<", rms_imgpos_error = " << rms_imgpos_err << endl << endl;
	delete[] nmatched_parts;
	delete[] chi_all_images;
	delete[] model_xvals_all_images;
	delete[] model_yvals_all_images;
	delete[] data_xvals_all_images;
	delete[] data_yvals_all_images;
	return chisq;
}

void QLens::find_analytic_srcflux(double *bestfit_flux)
{
	double chisq=0;
	int n_total_images=0;
	int i,j,k=0;

	for (i=0; i < n_ptsrc; i++)
		for (j=0; j < point_image_data[i].n_images; j++) n_total_images++;
	double image_mag;

	lensmatrix jac;
	for (i=0; i < n_ptsrc; i++) {
		double num=0, denom=0;
		for (j=0; j < point_image_data[i].n_images; j++) {
			if (point_image_data[i].sigma_f[j]==0) { k++; continue; }
			hessian(point_image_data[i].pos[j],jac,ptsrc_zfactors[ptsrc_redshift_idx[i]],ptsrc_beta_factors[ptsrc_redshift_idx[i]]);
			jac[0][0] = 1 - jac[0][0];
			jac[1][1] = 1 - jac[1][1];
			jac[0][1] = -jac[0][1];
			jac[1][0] = -jac[1][0];
			image_mag = 1.0/determinant(jac);
			if (include_parity_in_chisq) {
				num += point_image_data[i].flux[j] * image_mag / SQR(point_image_data[i].sigma_f[j]);
			} else {
				num += abs(point_image_data[i].flux[j] * image_mag) / SQR(point_image_data[i].sigma_f[j]);
			}
			denom += SQR(image_mag/point_image_data[i].sigma_f[j]);
			k++;
		}
		if (denom==0) bestfit_flux[i] = -1; // indicates we cannot find the source flux
		else bestfit_flux[i] = num/denom;
	}
}

void QLens::set_analytic_srcflux(const bool verbal)
{
	double *srcflux = new double[n_ptsrc];
	find_analytic_srcflux(srcflux);
	for (int i=0; i < n_ptsrc; i++) {
		ptsrc_list[i]->srcflux = srcflux[i];
		if ((verbal) and (mpi_id==0)) {
			cout << "analytic best-fit srcflux";
			if (n_ptsrc > 1) cout << " " << i;
			cout << ": " << srcflux[i] << endl;
		}
	}
	delete[] srcflux;
}

double QLens::chisq_flux()
{
	double chisq=0;
	int n_images_hi=0;
	int i,j,k;

	for (i=0; i < n_ptsrc; i++) {
		if (point_image_data[i].n_images > n_images_hi) n_images_hi = point_image_data[i].n_images;
	}
	double* image_mags = new double[n_images_hi];

	lensmatrix jac;
	double flux_src, num, denom;
	for (i=0; i < n_ptsrc; i++) {
		k=0; num=0; denom=0;
		for (j=0; j < point_image_data[i].n_images; j++) {
			if (point_image_data[i].sigma_f[j]==0) { k++; continue; }
			hessian(point_image_data[i].pos[j],jac,ptsrc_zfactors[ptsrc_redshift_idx[i]],ptsrc_beta_factors[ptsrc_redshift_idx[i]]);
			jac[0][0] = 1 - jac[0][0];
			jac[1][1] = 1 - jac[1][1];
			jac[0][1] = -jac[0][1];
			jac[1][0] = -jac[1][0];
			image_mags[k] = 1.0/determinant(jac);
			if (include_parity_in_chisq) {
				num += point_image_data[i].flux[j] * image_mags[k] / SQR(point_image_data[i].sigma_f[j]);
			} else {
				num += abs(point_image_data[i].flux[j] * image_mags[k]) / SQR(point_image_data[i].sigma_f[j]);
			}
			denom += SQR(image_mags[k]/point_image_data[i].sigma_f[j]);
			k++;
		}

		if (!analytic_source_flux) {
			flux_src = ptsrc_list[i]->srcflux; // only one source flux value is currently supported; later this should be generalized so that
											// some fluxes can be fixed and others parameterized
		}
		else {
			// the source flux is calculated analytically, rather than including it as a fit parameter (see Keeton 2001, section 4.2)
			flux_src = num / denom;
			ptsrc_list[i]->srcflux = flux_src; // although we're not using it directly, set the source object's flux for consistency
		}

		k=0;
		if (include_parity_in_chisq) {
			for (j=0; j < point_image_data[i].n_images; j++) {
				if (point_image_data[i].sigma_f[j]==0) { k++; continue; }
				chisq += SQR((point_image_data[i].flux[j] - image_mags[k++]*flux_src)/point_image_data[i].sigma_f[j]);
			}
		} else {
			for (j=0; j < point_image_data[i].n_images; j++) {
				if (point_image_data[i].sigma_f[j]==0) { k++; continue; }
				chisq += SQR((abs(point_image_data[i].flux[j]) - abs(image_mags[k++]*flux_src))/point_image_data[i].sigma_f[j]);
			}
		}
	}

	delete[] image_mags;
	return chisq;
}

double QLens::chisq_time_delays()
{
	double chisq=0;
	int n_images_hi=0;
	int i,j,k;

	for (i=0; i < n_ptsrc; i++) {
		if (point_image_data[i].n_images > n_images_hi) n_images_hi = point_image_data[i].n_images;
	}

	double td_factor;
	double* time_delays_obs = new double[n_images_hi];
	double* time_delays_mod = new double[n_images_hi];
	double min_td_obs, min_td_mod;
	double pot;
	lensvector beta_ij;
	double *specific_zfacs;
	double **specific_betafacs;
	for (k=0, i=0; i < n_ptsrc; i++) {
		specific_zfacs = ptsrc_zfactors[ptsrc_redshift_idx[i]];
		specific_betafacs = ptsrc_beta_factors[ptsrc_redshift_idx[i]];
		td_factor = cosmo->time_delay_factor_arcsec(lens_redshift,ptsrc_redshifts[ptsrc_redshift_idx[i]]);
		min_td_obs=1e30;
		min_td_mod=1e30;
		for (j=0; j < point_image_data[i].n_images; j++) {
			if (point_image_data[i].sigma_t[j]==0) continue;
			find_sourcept(point_image_data[i].pos[j],beta_ij,0,specific_zfacs,specific_betafacs);
			pot = potential(point_image_data[i].pos[j],specific_zfacs,specific_betafacs);
			time_delays_mod[j] = 0.5*(SQR(point_image_data[i].pos[j][0] - beta_ij[0]) + SQR(point_image_data[i].pos[j][1] - beta_ij[1])) - pot;
			if (time_delays_mod[j] < min_td_mod) min_td_mod = time_delays_mod[j];

			if (point_image_data[i].time_delays[j] < min_td_obs) min_td_obs = point_image_data[i].time_delays[j];
		}
		for (k=0, j=0; j < point_image_data[i].n_images; j++) {
			if (point_image_data[i].sigma_t[j]==0) { k++; continue; }
			time_delays_mod[k] -= min_td_mod;
			if (time_delays_mod[k] != 0.0) time_delays_mod[k] *= td_factor; // td_factor contains the cosmological factors and is in units of days
			time_delays_obs[k] = point_image_data[i].time_delays[j] - min_td_obs;
			k++;
		}
		for (k=0, j=0; j < point_image_data[i].n_images; j++) {
			if (point_image_data[i].sigma_t[j]==0) { k++; continue; }
			chisq += SQR((time_delays_obs[k] - time_delays_mod[k]) / point_image_data[i].sigma_t[j]);
			k++;
		}
	}
	if (chisq==0) warn("no time delay information has been used for chi-square");

	delete[] time_delays_obs;
	delete[] time_delays_mod;
	return chisq;
}

double QLens::chisq_time_delays_from_model_imgs()
{
	// this version evaluates the time delay at the position of the model images (useful if doing pixel modeling)
	// currently this chisq only works when doing image pixel modeling

	double chisq=0;
	int n_images_hi = 0;
	int i,j,k,n;
	for (i=0; i < n_ptsrc; i++) {
		if (ptsrc_list[i]->images.size() > n_images_hi) n_images_hi = ptsrc_list[i]->images.size();
	}
	double* time_delays_mod = new double[n_images_hi];
	bool zero_td_exists; // if true, then one of the data images has time delay of zero, so model time delays should subtract the TD of the corresponding model image to reproduce this zero point
	int zero_td_indx;
	double td_offset;

	int n_images, n_tot_images=0;
	double sigsq, chisq_each_srcpt, dist;
	bool skip;
	for (i=0; i < n_ptsrc; i++) {
		n_images = ptsrc_list[i]->images.size();
		chisq_each_srcpt = 0;

		/*
		if (n_images<=1) {
			for (k=0; k < point_image_data[i].n_images; k++) {
				sigsq = SQR(point_image_data[i].sigma_t[k]);
				if (point_image_data[i].use_in_chisq[k]) {
					chisq_each_srcpt += 100*point_image_data[i].max_tdsqr/sigsq;
				}
				//chisq_each_srcpt += 100*point_image_data[i].max_tdsqr/sigsq;
			}
			chisq += chisq_each_srcpt;
			continue;
		}
		*/

		zero_td_exists = false;
		td_offset = 0;
		skip = false;

		n_tot_images += n_visible_images;
		if ((n_images_penalty==true) and (n_visible_images > point_image_data[i].n_images)) {
			chisq += 1e30;
			continue;
		}

		int n_dists = n_images*point_image_data[i].n_images;
		double *distsqrs = new double[n_dists];
		int *data_k = new int[n_dists];
		int *model_j = new int[n_dists];
		n=0;
		for (k=0; k < point_image_data[i].n_images; k++) {
			if (point_image_data[i].time_delays[k]==0) {
				zero_td_exists = true;
				zero_td_indx = k;
			}
		}
		for (k=0; k < point_image_data[i].n_images; k++) {
			for (j=0; j < n_images; j++) {
				distsqrs[n] = SQR(point_image_data[i].pos[k][0] - ptsrc_list[i]->images[j].pos[0]) + SQR(point_image_data[i].pos[k][1] - ptsrc_list[i]->images[j].pos[1]);
				data_k[n] = k;
				model_j[n] = j;
				n++;
			}
		}
		if (n != n_dists) die("count of all data-model image combinations does not equal expected number (%i vs %i)",n,n_dists);
		sort(n_dists,distsqrs,data_k,model_j);
		int *closest_image_j = new int[point_image_data[i].n_images];
		int *closest_image_k = new int[n_images];
		for (k=0; k < point_image_data[i].n_images; k++) closest_image_j[k] = -1;
		for (j=0; j < n_images; j++) closest_image_k[j] = -1;
		int m=0;
		int mmax = dmin(n_images,point_image_data[i].n_images);
		for (n=0; n < n_dists; n++) {
			if ((closest_image_j[data_k[n]] == -1) and (closest_image_k[model_j[n]] == -1)) {
				closest_image_j[data_k[n]] = model_j[n];
				closest_image_k[model_j[n]] = data_k[n];
				m++;
				if (m==mmax) n = n_dists; // force loop to exit
			}
		}
		if (zero_td_exists) {
			if (closest_image_j[zero_td_indx] == -1) {
				chisq_each_srcpt = 1e30; // penalty for not even reproducing the image with the zero-point TD
				skip = true;
			} else {
				td_offset = ptsrc_list[i]->images[closest_image_j[zero_td_indx]].td;
				//cout << "zero_td_indx, matched model indx: " << zero_td_indx << " " << closest_image_j[zero_td_indx] << ", offset=" << td_offset << endl;
			}
		}

		if (!skip) {
			for (k=0; k < point_image_data[i].n_images; k++) {
				sigsq = SQR(point_image_data[i].sigma_t[k]);
				j = closest_image_j[k];
				if (j != -1) {
					if (point_image_data[i].use_in_chisq[k]) {
						chisq_each_srcpt += SQR(ptsrc_list[i]->images[j].td - td_offset - point_image_data[i].time_delays[k])/sigsq;
					}
				} else {
					// add a penalty value to chi-square for not reproducing this data image; the effective time delay difference is 10 times the maximum difference between any pair of time delays
					chisq_each_srcpt += 100*point_image_data[i].max_tdsqr/sigsq;
				}
			}
		}
		chisq += chisq_each_srcpt;
		delete[] distsqrs;
		delete[] data_k;
		delete[] model_j;
		delete[] closest_image_j;
		delete[] closest_image_k;
	}

	if (chisq==0) warn("no time delay information has been used for chi-square");
	delete[] time_delays_mod;
	return chisq;
}

double QLens::chisq_weak_lensing()
{
	int i,j,nsrc = weak_lensing_data.n_sources;
	if (nsrc==0) return 0;
	double chisq=0;
	double g1,g2;
	double **zfacs = new double*[nsrc];
	for (i=0; i < nsrc; i++) {
		zfacs[i] = new double[n_lens_redshifts];
	}
	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif

		#pragma omp for private(i,j,g1,g2) schedule(static) reduction(+:chisq)
		for (i=0; i < nsrc; i++) {
			for (j=0; j < n_lens_redshifts; j++) {
				zfacs[i][j] = cosmo->kappa_ratio(lens_redshifts[j],weak_lensing_data.zsrc[i],reference_source_redshift);
			}
			reduced_shear_components(weak_lensing_data.pos[i],g1,g2,thread,zfacs[i]);
			chisq += SQR((wl_shear_factor*g1-weak_lensing_data.reduced_shear1[i])/weak_lensing_data.sigma_shear1[i]) + SQR((wl_shear_factor*g2-weak_lensing_data.reduced_shear2[i])/weak_lensing_data.sigma_shear2[i]);

		}
	}
	for (i=0; i < nsrc; i++) delete[] zfacs[i];
	delete[] zfacs;
	return chisq;
}

bool QLens::output_weak_lensing_chivals(string filename)
{
	int i,j,nsrc = weak_lensing_data.n_sources;
	if (nsrc==0) return false;
	ofstream chifile(filename.c_str());
	double chi1, chi2;
	double g1,g2;
	double **zfacs = new double*[nsrc];
	for (i=0; i < nsrc; i++) {
		zfacs[i] = new double[n_lens_redshifts];
	}
	for (i=0; i < nsrc; i++) {
		for (j=0; j < n_lens_redshifts; j++) {
			zfacs[i][j] = cosmo->kappa_ratio(lens_redshifts[j],weak_lensing_data.zsrc[i],reference_source_redshift);
		}
		reduced_shear_components(weak_lensing_data.pos[i],g1,g2,0,zfacs[i]);
		chi1 = (wl_shear_factor*g1-weak_lensing_data.reduced_shear1[i])/weak_lensing_data.sigma_shear1[i];
		chi2 = (wl_shear_factor*g2-weak_lensing_data.reduced_shear2[i])/weak_lensing_data.sigma_shear2[i];
		chifile << chi1 << " " << chi2 << endl;
	}
	for (i=0; i < nsrc; i++) delete[] zfacs[i];
	delete[] zfacs;
	return true;
}

double QLens::get_avg_ptsrc_dist(const int ptsrc_i)
{
	// is this even useful? maybe get rid of this, since it's better to use analytic_bestfit_src combined with xshift, yshift params
	double avg_srcdist;
	int j,k,n_srcpts,n_src_pairs;
	if (point_image_data[ptsrc_i].n_images > 1) {
		avg_srcdist=0;
		n_src_pairs=0;
		n_srcpts = point_image_data[ptsrc_i].n_images;
		lensvector *srcpts = new lensvector[n_srcpts];
		for (j=0; j < n_srcpts; j++) {
			find_sourcept(point_image_data[ptsrc_i].pos[j],srcpts[j],0,ptsrc_zfactors[ptsrc_redshift_idx[ptsrc_i]],ptsrc_beta_factors[ptsrc_redshift_idx[ptsrc_i]]);
			for (k=0; k < j; k++) {
				avg_srcdist += sqrt(SQR(srcpts[j][0] - srcpts[k][0]) + SQR(srcpts[j][1] - srcpts[k][1]));
				n_src_pairs++;
			}
		}
		avg_srcdist /= n_src_pairs;
		delete[] srcpts;
		return avg_srcdist;
	} else {
		return sqrt(grid_xlength*grid_ylength); // nothing else to use, since there's no lens or image data to model 
	}
}


bool QLens::fit_set_optimizations()
{
	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}
	if (nlens==0) {
		if ((n_sb==0) and (n_ptsrc==0)) {
			warn("no lens or source models have been defined");
			return false;
		} else {
			bool all_unlensed = true;
			for (int i=0; i < n_sb; i++) {
				if (sb_list[i]->is_lensed) all_unlensed = false;
			}
			if (!all_unlensed) {
				warn("background source objects have been defined, but no lens models have been defined");
				return false;
			}
			all_unlensed = true;
			for (int i=0; i < n_ptsrc; i++) {
				if (ptsrc_redshifts[ptsrc_redshift_idx[i]] != lens_redshift) all_unlensed = false;
			}
			if (!all_unlensed) {
				warn("background source points have been defined, but no lens models have been defined");
				return false;
			}
		}
	}

	if ((lensmodel_fit_parameters==0) and (psf_fit_parameters==0)) redo_lensing_calculations_before_inversion = false; // so we don't waste time redoing the ray tracing if lens doesn't change and we're not shifting ray-tracing points (note, the offset in ray-tracing points is in the PSF object)
	else redo_lensing_calculations_before_inversion = true;

	temp_auto_store_cc_points = auto_store_cc_points;
	temp_include_time_delays = include_time_delays;

	// turn the following features off because they add pointless overhead (they will be restored to their
	// former settings after the search is done)
	auto_store_cc_points = false;
	if (source_fit_mode==Point_Source) include_time_delays = false; // calculating time delays from images found not necessary during point source fit, since the chisq_time_delays finds time delays separately

	fisher_inverse.erase(); // reset parameter covariance matrix in case it was used in a previous fit
	return true;
}

void QLens::fit_restore_defaults()
{
	if (!redo_lensing_calculations_before_inversion) redo_lensing_calculations_before_inversion = true;
	auto_store_cc_points = temp_auto_store_cc_points;
	include_time_delays = temp_include_time_delays;
	clear_raw_chisq(); // in case chi-square is being used as a derived parameter
	Grid::set_lens(this); // annoying that the grids can only point to one lens object--it would be better for the pointer to be non-static (implement this later)
}

double QLens::chisq_single_evaluation(bool init_fitmodel, bool show_total_wtime, bool show_diagnostics, bool show_status, bool show_lensinfo)
{
	if (fit_set_optimizations()==false) return -1e30;
	if (fit_output_dir != ".") create_output_directory();
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	if (init_fitmodel) {
		if (!initialize_fitmodel(false)) {
			raw_chisq = -1e30;
			if ((mpi_id==0) and (show_status)) warn(warnings,"Warning: could not evaluate chi-square function");
			return -1e30;
		}
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime = omp_get_wtime() - wtime0;
			if ((mpi_id==0) and (show_status)) cout << "Wall time for initializing fitmodel (not part of likelihood evaluation): " << wtime << endl;
		}
#endif
	} else {
		fitmodel = this;
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime0 = omp_get_wtime();
		}
#endif
		if ((source_fit_mode != Point_Source) and ((image_pixel_grids == NULL) or (image_pixel_grids[0]==NULL))) {
			bool success = true;
			for (int i=0; i < n_data_bands; i++) {
				if (!load_pixel_grid_from_data(i)) {
					success = false;
					break;
				}
			}
			if (success) {
#ifdef USE_OPENMP
				if (show_wtime) {
					wtime = omp_get_wtime() - wtime0;
					if ((mpi_id==0) and (show_status)) cout << "Wall time for initializing image pixel grids (not part of likelihood evaluation): " << wtime << endl;
				}
#endif
			}
		}
	}

	//fitmodel->param_list->print_penalty_limits();

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	display_chisq_status = true;
	fitmodel->chisq_it = 0;
	if (show_diagnostics) chisq_diagnostic = true;
	bool default_display_status = display_chisq_status;
	if (!show_status) display_chisq_status = false;

#ifdef USE_OPENMP
	double chisq_wtime0, chisq_wtime;
	if ((show_wtime) or (show_total_wtime)) {
		chisq_wtime0 = omp_get_wtime();
	}
#endif

	double chisqval = 2 * (this->*LogLikePtr)(param_list->values);
	if (einstein_radius_prior) fitmodel->get_einstein_radius_prior(true); // just to show what the Re prior is returning
	if (!show_status) display_chisq_status = default_display_status;
	if ((chisqval >= 1e30) and (mpi_id==0)) warn(warnings,"Your parameter values are returning a large \"penalty\" chi-square--this likely means one or\nmore parameters have unphysical values or are out of the bounds specified by 'fit plimits'");
#ifdef USE_OPENMP
	if ((show_wtime) or (show_total_wtime)) {
		chisq_wtime = omp_get_wtime() - chisq_wtime0;
		if ((mpi_id==0) and (show_status)) cout << "Wall time for likelihood evaluation: " << chisq_wtime << endl;
	}
#endif
	display_chisq_status = false;
	if (show_diagnostics) chisq_diagnostic = false;

	if ((mpi_id==0) and (show_lensinfo)) {
		cout << "lensing info:" << endl;
		print_lensing_info_at_point(0.05,0.07);
		cout << "fitmodel lensing info:" << endl;
		fitmodel->print_lensing_info_at_point(0.05,0.07);
		cout << "cosmo info:" << endl;
		print_lens_cosmology_info(0,nlens-1);
		cout << "fitmodel cosmo info:" << endl;
		fitmodel->print_lens_cosmology_info(0,nlens-1);
	}

	double rawchisqval = raw_chisq;
	fit_restore_defaults();
	if (init_fitmodel) delete fitmodel;
	fitmodel = NULL;
	return rawchisqval;
}

void QLens::plot_chisq_2d(const int param1, const int param2, const int n1, const double i1, const double f1, const int n2, const double i2, const double f2)
{
	if (fit_set_optimizations()==false) return;
	if (fit_output_dir != ".") create_output_directory();
	if (!initialize_fitmodel(false)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	if (param1 >= param_list->nparams) { warn("Parameter %i does not exist (%i parameters total)",param1,param_list->nparams); return; }
	if (param2 >= param_list->nparams) { warn("Parameter %i does not exist (%i parameters total)",param2,param_list->nparams); return; }

	double (QLens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	double step1 = (f1-i1)/n1;
	double step2 = (f2-i2)/n2;
	int i,j;
	double p1,p2;
	ofstream p1out("chisq2d.x");
	ofstream l1out("like2d.x");
	for (i=0, p1=i1; i <= n1; i++, p1 += step1) {
		p1out << p1 << endl;
		l1out << p1 << endl;
	}
	p1out.close();
	l1out.close();
	ofstream p2out("chisq2d.y");
	ofstream l2out("like2d.y");
	for (i=0, p2=i2; i <= n2; i++, p2 += step2) {
		p2out << p2 << endl;
		l2out << p2 << endl;
	}
	p2out.close();
	l2out.close();

	double chisqmin=1e30;
	dmatrix chisqvals(n1,n2);
	ofstream chisqout("chisq2d.dat");
	double p1min, p2min;
	for (j=0, p2=i2+0.5*step2; j < n2; j++, p2 += step2) {
		for (i=0, p1=i1+0.5*step1; i < n1; i++, p1 += step1) {
			cout << "p1=" << p1 << " p2=" << p2 << endl;
			param_list->values[param1] = p1;
			param_list->values[param2] = p2;
			chisqvals[i][j] = 2.0 * (this->*loglikeptr)(param_list->values);
			if (chisqvals[i][j] < chisqmin) {
				chisqmin = chisqvals[i][j];
				p1min = p1;
				p2min = p2;
			}
			chisqout << chisqvals[i][j] << " ";
		}
		chisqout << endl;
	}
	chisqout.close();
	if (mpi_id==0) cout << "min chisq=" << chisqmin << ", occurs at (" << p1min << "," << p2min << ")\n";

	ofstream likeout("like2d.dat");
	for (i=0; i < n1; i++) {
		for (j=0; j < n2; j++) {
			likeout << exp(-0.5*SQR(chisqvals[i][j]-chisqmin)) << " ";
		}
		likeout << endl;
	}
	likeout.close();

	fit_restore_defaults();
}

void QLens::plot_chisq_1d(const int param, const int n, const double ip, const double fp, string filename)
{
	if (fit_set_optimizations()==false) return;
	if (fit_output_dir != ".") create_output_directory();
	if (!initialize_fitmodel(false)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	if (param >= param_list->nparams) { warn("Parameter %i does not exist (%i parameters total)",param,param_list->nparams); return; }

	double (QLens::*LogLikePtr)(double*);
	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	double step = (fp-ip)/n;
	int i,j;
	double p;

	double chisqmin=1e30;
	dvector chisqvals(n);
	ofstream chisqout(filename.c_str());
	double pmin;
	for (i=0, p=ip; i <= n; i++, p += step) {
		param_list->values[param] = p;
		chisqvals[i] = 2.0 * (this->*LogLikePtr)(param_list->values);
		if (chisqvals[i] < chisqmin) {
			chisqmin = chisqvals[i];
			pmin = p;
		}
		chisqout << p << " " << chisqvals[i] << endl;
	}
	chisqout.close();
	if (mpi_id==0) cout << "min chisq=" << chisqmin << ", occurs at " << pmin << endl;

	fit_restore_defaults();
}

double QLens::chi_square_fit_simplex(const bool show_parameter_errors)
{
	fitmethod = SIMPLEX;
	if (fit_set_optimizations()==false) return -1e30;
	if (!initialize_fitmodel(false)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return 1e30;
	}

	double (Simplex::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Simplex::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (Simplex::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	int n_fitparams = param_list->nparams;
	dvector stepsizes(param_list->stepsizes,n_fitparams);
	if (mpi_id==0) {
		cout << "Initial stepsizes: ";
		for (int i=0; i < n_fitparams; i++) cout << stepsizes[i] << " ";
		cout << endl << endl;
	}

	double *fitparams = param_list->values;
	initialize_simplex(fitparams,n_fitparams,stepsizes.array(),chisq_tolerance);
	simplex_set_display_bfpont(simplex_show_bestfit);
	simplex_set_function(loglikeptr);
	simplex_set_fmin(simplex_minchisq/2);
	simplex_set_fmin_anneal(simplex_minchisq_anneal/2);
	//int iterations = 0;
	//downhill_simplex(iterations,max_iterations,0); // last argument is temperature for simulated annealing, but there is no cooling schedule with this function
	set_annealing_schedule_parameters(simplex_temp_initial,simplex_temp_final,simplex_cooling_factor,simplex_nmax_anneal,simplex_nmax);
	int n_iterations;

	double chisq_initial = (this->*loglikeptr)(fitparams);
	if ((chisq_initial >= 1e30) and (mpi_id==0)) warn(warnings,"Your initial parameter values are returning a large \"penalty\" chi-square--this likely means\none or more parameters have unphysical values or are out of the bounds specified by 'fit plimits'");

	display_chisq_status = true;

	fitmodel->chisq_it = 0;
	bool verbal = (mpi_id==0) ? true : false;
	//if (simplex_show_bestfit) cout << endl; // since we'll need an extra line to display best-fit parameters during annealing
	if (use_ansi_output_during_fit) use_ansi_characters = true;
	else use_ansi_characters = false;
	n_iterations = downhill_simplex_anneal(verbal);
	simplex_minval(fitparams,chisq_bestfit);
	chisq_bestfit *= 2; // since the loglike function actually returns 0.5*chisq
	int chisq_evals = fitmodel->chisq_it;
	fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
	if (display_chisq_status) {
		(this->*loglikeptr)(fitparams);
		if (mpi_id==0) cout << endl << endl;
	}
	//use_ansi_characters = false;

	bool turned_on_chisqmag = false;
	if (n_repeats > 0) {
		if ((source_fit_mode==Point_Source) and (!use_magnification_in_chisq) and (use_magnification_in_chisq_during_repeats) and (!imgplane_chisq)) {
			turned_on_chisqmag = true;
			use_magnification_in_chisq = true;
			fitmodel->use_magnification_in_chisq = true;
			simplex_evaluate_bestfit_point(); // need to re-evaluate and record the chi-square at the best-fit point since we are changing the chi-square function
			cout << "Now using magnification in position chi-square function during repeats...\n";
		}
		set_annealing_schedule_parameters(0,simplex_temp_final,simplex_cooling_factor,simplex_nmax_anneal,simplex_nmax); // repeats have zero temperature (just minimization)
		for (int i=0; i < n_repeats; i++) {
			if (mpi_id==0) cout << "Repeating optimization (trial " << i+1 << ")                                                  \n\n\n" << flush;
			//use_ansi_characters = true;
			n_iterations = downhill_simplex_anneal(verbal);
			//use_ansi_characters = false;
			simplex_minval(fitparams,chisq_bestfit);
			chisq_bestfit *= 2; // since the loglike function actually returns 0.5*chisq
			chisq_evals += fitmodel->chisq_it;
			fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
			if (display_chisq_status) {
				(this->*loglikeptr)(fitparams);
				if (mpi_id==0) cout << endl << endl;
			}
		}
	}
	use_ansi_characters = false;
	bestfitparams.input(fitparams,n_fitparams);

	display_chisq_status = false;
	if (mpi_id==0) {
		if (simplex_exit_status==true) {
			if (simplex_temp_initial==0) cout << "Downhill simplex converged after " << n_iterations << " iterations\n\n";
			else cout << "Downhill simplex converged after " << n_iterations << " iterations at final temperature T=0\n\n";
		} else {
			cout << "Downhill simplex interrupted after " << n_iterations << " iterations\n\n";
		}
	}

	output_fit_results(stepsizes,chisq_bestfit,chisq_evals,show_parameter_errors);

	if (turned_on_chisqmag) use_magnification_in_chisq = false; // restore chisqmag to original setting
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return chisq_bestfit;
}

double QLens::chi_square_fit_powell(const bool show_parameter_errors)
{
	fitmethod = POWELL;
	if (fit_set_optimizations()==false) return -1e30;
	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return 1e30;
	}

	double (Powell::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Powell::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (Powell::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	initialize_powell(loglikeptr,chisq_tolerance);

	int n_fitparams = param_list->nparams;
	dvector stepsizes(param_list->stepsizes,n_fitparams);
	if (mpi_id==0) {
		cout << "Initial stepsizes: ";
		for (int i=0; i < n_fitparams; i++) cout << stepsizes[i] << " ";
		cout << endl << endl;
	}

	double *fitparams = param_list->values;
	double chisq_initial = (this->*loglikeptr)(fitparams);
	if ((chisq_initial >= 1e30) and (mpi_id==0)) warn(warnings,"Your initial parameter values are returning a large \"penalty\" chi-square--this likely means\none or more parameters have unphysical values or are out of the bounds specified by 'fit plimits'");

	display_chisq_status = true;

	fitmodel->chisq_it = 0;
	if (use_ansi_output_during_fit) use_ansi_characters = true;
	else use_ansi_characters = false;
	powell_minimize(fitparams,n_fitparams,stepsizes.array());
	use_ansi_characters = false;
	chisq_bestfit = 2*(this->*loglikeptr)(fitparams);
	int chisq_evals = fitmodel->chisq_it;
	fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
	if (display_chisq_status) {
		(this->*loglikeptr)(fitparams);
		if (mpi_id==0) cout << endl;
	}

	bool turned_on_chisqmag = false;
	if (n_repeats > 0) {
		if ((source_fit_mode==Point_Source) and (!use_magnification_in_chisq) and (use_magnification_in_chisq_during_repeats) and (!imgplane_chisq)) {
			turned_on_chisqmag = true;
			use_magnification_in_chisq = true;
			fitmodel->use_magnification_in_chisq = true;
			cout << "Now using magnification in position chi-square function during repeats...\n";
		}
		for (int i=0; i < n_repeats; i++) {
			if (mpi_id==0) cout << "Repeating optimization (trial " << i+1 << ")\n";
			use_ansi_characters = true;
			powell_minimize(fitparams,n_fitparams,stepsizes.array());
			use_ansi_characters = false;
			chisq_bestfit = 2*(this->*loglikeptr)(fitparams);
			chisq_evals += fitmodel->chisq_it;
			fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
			if (display_chisq_status) {
				(this->*loglikeptr)(fitparams);
				if (mpi_id==0) cout << endl;
			}
		}
	}
	bestfitparams.input(fitparams,n_fitparams);
	display_chisq_status = false;
	if (group_id==0) fitmodel->logfile << "Optimization finished: min chisq = " << chisq_bestfit << endl;

	output_fit_results(stepsizes,chisq_bestfit,chisq_evals,show_parameter_errors);

	if (turned_on_chisqmag) use_magnification_in_chisq = false; // restore chisqmag to original setting
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return chisq_bestfit;
}

void QLens::output_fit_results(dvector &stepsizes, const double chisq_bestfit, const int chisq_evals, const bool show_parameter_errors)
{
	bool fisher_matrix_is_nonsingular;
	double *fitparams = param_list->values;
	if (show_parameter_errors) {
		if (mpi_id==0) cout << "Calculating parameter errors... (press CTRL-C to skip)" << endl;
		fisher_matrix_is_nonsingular = calculate_fisher_matrix(fitparams,stepsizes);
		if (fisher_matrix_is_nonsingular) bestfit_fisher_inverse.input(fisher_inverse);
		else bestfit_fisher_inverse.erase(); // just in case it was defined before
		if (mpi_id==0) cout << endl;
	}
	if (mpi_id==0) {
		if (use_scientific_notation) cout << setiosflags(ios::scientific);
		else {
			cout << resetiosflags(ios::scientific);
			cout.unsetf(ios_base::floatfield);
		}
		cout << "\nBest-fit model: 2*loglike = " << chisq_bestfit << " (after " << chisq_evals << " evals)" << endl << endl;
	}

	int n_fitparams = param_list->nparams;
	double transformed_params[n_fitparams];
	fitmodel->param_list->inverse_transform_parameters(fitparams,transformed_params);
	fitmodel->update_model(transformed_params);
	for (int i=0; i < nlens; i++) {
		fitmodel->lens_list[i]->reset_angle_modulo_2pi();
	}

	if (mpi_id==0) {
		if (nlens > 0) {
			cout << "Lenses:" << endl;
			fitmodel->print_lens_list(false);
		}
		if (n_sb > 0) {
			cout << "Source profiles:" << endl;
			fitmodel->print_source_list(false);
		}
		if (n_ptsrc > 0) {
			cout << "Point sources:" << endl;
			fitmodel->print_point_source_list(false);
		}

		if (show_parameter_errors) {
			if (fisher_matrix_is_nonsingular) {
				cout << "Marginalized 1-sigma errors from Fisher matrix:\n";
				for (int i=0; i < n_fitparams; i++) {
					cout << param_list->param_names[i] << ": " << fitparams[i] << " +/- " << sqrt(abs(fisher_inverse[i][i])) << endl;
				}
			} else {
				cout << "Error: Fisher matrix is singular, marginalized errors cannot be calculated\n";
				for (int i=0; i < n_fitparams; i++)
					cout << param_list->param_names[i] << ": " << fitparams[i] << endl;
			}
		} else {
			for (int i=0; i < n_fitparams; i++)
				cout << param_list->param_names[i] << ": " << fitparams[i] << endl;
		}
		cout << endl;
		if (auto_save_bestfit) output_bestfit_model(show_parameter_errors);
	}
}

int FISHER_KEEP_RUNNING = 1;

void fisher_sighandler(int sig)
{
	FISHER_KEEP_RUNNING = 0;
}

void fisher_quitproc(int sig)
{
	exit(0);
}

bool QLens::calculate_fisher_matrix(double *params, const dvector &stepsizes)
{
	// this function calculates the marginalized error using the Gaussian approximation
	// (only accurate if we are near maximum likelihood point and it is close to Gaussian around this point)
	static const double increment2 = 1e-4;
	if ((mpi_id==0) and (source_fit_mode==Point_Source) and (!imgplane_chisq) and (!use_magnification_in_chisq)) warn("Fisher matrix errors may not be accurate if source plane chi-square is used without magnification");

	int n_fitparams = param_list->nparams;
	dmatrix fisher(n_fitparams,n_fitparams);
	fisher_inverse.erase();
	fisher_inverse.input(n_fitparams,n_fitparams);
	dvector xhi(params,n_fitparams);
	dvector xlo(params,n_fitparams);
	double x0, curvature;
	int i,j;
	double step, derivlo, derivhi;
	for (i=0; i < n_fitparams; i++) {
		x0 = params[i];
		xhi[i] += increment2*stepsizes[i];
		if ((param_list->defined_prior_limits[i]==true) and (xhi[i] > param_list->prior_limits_hi[i])) xhi[i] = x0;
		xlo[i] -= increment2*stepsizes[i];
		if ((param_list->defined_prior_limits[i]==true) and (xlo[i] < param_list->prior_limits_lo[i])) xlo[i] = x0;
		step = xhi[i] - xlo[i];
		for (j=0; j < n_fitparams; j++) {
			derivlo = loglike_deriv(xlo,j,stepsizes[j]);
			derivhi = loglike_deriv(xhi,j,stepsizes[j]);
			fisher[i][j] = (derivhi - derivlo) / step;
			if (fisher[i][j]*0.0) warn(warnings,"Fisher matrix element (%i,%i) calculated as 'nan'",i,j);
			//if (i==j) cout << abs(derivlo+derivhi) << " " << sqrt(abs(fisher[i][j])) << endl;
			if ((mpi_id==0) and (i==j) and (abs(derivlo+derivhi) > sqrt(abs(fisher[i][j])))) warn(warnings,"Derivatives along parameter %i indicate best-fit point may not be at a local minimum of chi-square",i);
			signal(SIGABRT, &fisher_sighandler);
			signal(SIGTERM, &fisher_sighandler);
			signal(SIGINT, &fisher_sighandler);
			signal(SIGUSR1, &fisher_sighandler);
			signal(SIGQUIT, &fisher_quitproc);
			if (!FISHER_KEEP_RUNNING) {
				fisher_inverse.erase();
				return false;
			}
		}
		xhi[i]=xlo[i]=x0;
	}

	double offdiag_avg;
	// average the off-diagonal elements to enforce symmetry
	for (i=1; i < n_fitparams; i++) {
		for (j=0; j < i; j++) {
			offdiag_avg = 0.5*(fisher[i][j]+ fisher[j][i]);
			//if (abs((fisher[i][j]-fisher[j][i])/offdiag_avg) > 0.01) die("Fisher off-diags differ by more than 1%!");
			fisher[i][j] = fisher[j][i] = offdiag_avg;
		}
	}
	bool nonsingular = fisher.check_nan();
	if (nonsingular) fisher.inverse(fisher_inverse,nonsingular);
	if (!nonsingular) {
		if (mpi_id==0) warn(warnings,"Fisher matrix is singular, cannot be inverted\n");
		fisher_inverse.erase();
		return false;
	}
	return true;
}

double QLens::loglike_deriv(const dvector &params, const int index, const double step)
{
	static const double increment = 1e-5;
	dvector xhi(params);
	dvector xlo(params);
	double dif, x0 = xhi[index];
	xhi[index] += increment*step;
	if ((param_list->defined_prior_limits[index]==true) and (xhi[index] > param_list->prior_limits_hi[index])) xhi[index] = x0;
	xlo[index] -= increment*step;
	if ((param_list->defined_prior_limits[index]==true) and (xlo[index] < param_list->prior_limits_lo[index])) xlo[index] = x0;
	dif = xhi[index] - xlo[index];
	double (QLens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		loglikeptr = static_cast<double (QLens::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}
	return (((this->*loglikeptr)(xhi.array()) - (this->*loglikeptr)(xlo.array())) / dif);
}

void QLens::nested_sampling()
{
	fitmethod = NESTED_SAMPLING;
	if (!param_list->all_prior_limits_defined()) { warn("not all prior limits have been defined"); return; }
	if (fit_set_optimizations()==false) return;
	if ((mpi_id==0) and (fit_output_dir != ".")) {
		// I should probably give the nested sampling output a unique extension like ".nest" or something, so that mkdist can't ever confuse it with twalk output in the same dir
		// Do this later...
#if __cplusplus >= 201703L // C++17 standard or later
		if (filesystem::exists(fit_output_dir)) {
			filesystem::remove_all(fit_output_dir);
		}
#else
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		if (system(rmstring.c_str()) != 0) warn("could not delete old output directory for nested sampling results"); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
#endif
		create_output_directory();
	}

	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	double *fitparams = param_list->values;
	int n_fitparams = param_list->nparams;
	int n_derived_params = dparam_list->n_dparams;
	InputPoint(fitparams,param_list->prior_limits_hi,param_list->prior_limits_lo,n_fitparams);
	SetNDerivedParams(n_derived_params);

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	if (mpi_id==0) {
		// This code gets repeated in a few spots and should really be put in a separate function...DO THIS LATER!
		int i;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fitparams << " " << n_derived_params << endl;
		pnumfile.close();
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_fitparams; i++) pnamefile << param_list->param_names[i] << endl;
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list->dparams[i]->name << endl;
		pnamefile.close();
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_fitparams; i++) lpnamefile << param_list->param_names[i] << "\t" << param_list->latex_names[i] << endl;
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list->dparams[i]->name << "\t" << dparam_list->dparams[i]->latex_name << endl;
		lpnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < n_fitparams; i++)
		{
			prangefile << param_list->prior_limits_lo[i] << " " << param_list->prior_limits_hi[i] << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		if (param_markers != "") {
			string marker_str = fit_output_dir + "/" + fit_output_filename + ".markers";
			ofstream markerfile(marker_str.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

	double *param_errors = new double[n_fitparams];
#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	string filename = fit_output_dir + "/" + fit_output_filename;

	display_chisq_status = false; // just in case it was turned on
	double lnZ;

	use_ansi_characters = true;
	MonoSample(filename.c_str(),n_livepts,lnZ,fitparams,param_errors,mcmc_logfile,NULL,chain_info,data_info);
	use_ansi_characters = false;
	bestfitparams.input(fitparams,n_fitparams);
	chisq_bestfit = 2*(this->*LogLikePtr)(fitparams);

#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for nested sampling: " << wt << endl;
	}
#endif

	if (mpi_id==0) {
		cout << endl;
		cout << "\nBest-fit parameters and error estimates (from dispersions of chain output points):\n";
		for (int i=0; i < n_fitparams; i++) {
			cout << param_list->param_names[i] << ": " << fitparams[i] << " +/- " << param_errors[i] << endl;
		}
		cout << endl;
		output_bestfit_model();
	}
	delete[] param_errors;

	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
}

void QLens::multinest(const bool resume_previous, const bool skip_run)
{
	fitmethod = MULTINEST;
	if (!param_list->all_prior_limits_defined()) { warn("not all prior limits have been defined"); return; }
#ifdef USE_MULTINEST
	if (fit_set_optimizations()==false) return;
	if ((mpi_id==0) and (!resume_previous) and (!skip_run) and (fit_output_dir != ".")) {
		// I should probably give the nested sampling output a unique extension like ".nest" or something, so that mkdist can't ever confuse it with twalk output in the same dir
		// Do this later...
#if __cplusplus >= 201703L // C++17 standard or later
		if (filesystem::exists(fit_output_dir)) {
			filesystem::remove_all(fit_output_dir);
		}
#else
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		if (system(rmstring.c_str()) != 0) warn("could not delete old output directory for nested sampling results"); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
#endif
		create_output_directory();
	}

	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	int n_fitparams = param_list->nparams;
	int n_derived_params = dparam_list->n_dparams;
	if (mpi_id==0) {
		// This code gets repeated in a few spots and should really be put in a separate function...DO THIS LATER!
		int i;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fitparams << " " << n_derived_params << endl;
		pnumfile.close();
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_fitparams; i++) pnamefile << param_list->param_names[i] << endl;
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list->dparams[i]->name << endl;
		pnamefile.close();
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_fitparams; i++) lpnamefile << param_list->param_names[i] << "\t" << param_list->latex_names[i] << endl;
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list->dparams[i]->name << "\t" << dparam_list->dparams[i]->latex_name << endl;
		lpnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < n_fitparams; i++)
		{
			prangefile << param_list->prior_limits_lo[i] << " " << param_list->prior_limits_hi[i] << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		if (param_markers != "") {
			string marker_str = fit_output_dir + "/" + fit_output_filename + ".markers";
			ofstream markerfile(marker_str.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	display_chisq_status = false; // just in case it was turned on
	string filename = fit_output_dir + "/" + fit_output_filename;

	 mcsampler_set_lensptr(this);

	int IS, mmodal, ceff, nPar, nClsPar, nlive, updInt, maxModes, seed, fb, resume, outfile, initMPI, maxiter;
	double efr, tol, Ztol, logZero;

	IS = 0;					// do Nested Importance Sampling (bad idea)
	mmodal = (multimodal_sampling) ? 1 : 0;					// do mode separation?
	ceff = (multinest_constant_eff_mode) ? 1 : 0;
	efr = multinest_target_efficiency;				// set the required efficiency
	nlive = n_livepts;
	tol = 0.5;				// tol, defines the stopping criteria
	nPar = n_fitparams+n_derived_params;					// total no. of parameters including free & derived parameters
	nClsPar = n_fitparams;				// no. of parameters to do mode separation on
	updInt = 10;				// after how many iterations feedback is required & the output files should be updated
							// note: posterior files are updated & dumper routine is called after every updInt*10 iterations
	Ztol = -1e90;				// all the modes with logZ < Ztol are ignored
	maxModes = 100;				// expected max no. of modes (used only for memory allocation)
	seed = random_seed+group_num;					// random no. generator seed, if < 0 then take the seed from system clock

	fb = (mpi_id==0) ? 1 : 0;				// need feedback on standard output?
	resume = (resume_previous) ? 1 : 0;				// resume from a previous job?

	outfile = 1;				// write output files?
	initMPI = 0;				// initialize MPI routines?, relevant only if compiling with MPI
							// set it to 0 if you want your main program to handle MPI initialization
	logZero = -1e90;			// points with loglike < logZero will be ignored by MultiNest
	maxiter = 0;				// max no. of iterations, a non-positive value means infinity. MultiNest will terminate if either it 
							// has done max no. of iterations or convergence criterion (defined through tol) has been satisfied
	void *context = 0;				// not required by MultiNest, any additional information user wants to pass
	int pWrap[n_fitparams];				// which parameters to have periodic boundary conditions?
	for (int i = 0; i < n_fitparams; i++) pWrap[i] = 0;
	//MPI_Fint fortran_comm = MPI_Comm_c2f((*group_comm));

	use_ansi_characters = true;

	if (!skip_run) {
#ifdef MULTINEST_MOD
		// This code uses a modified version of MultiNest that allows for the likelihood to be parallelized over a subset of MPI processes
		MPI_Group world_group;
		MPI_Comm world_comm;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);
		MPI_Comm_create(MPI_COMM_WORLD, world_group, &world_comm);

		MPI_Group leader_group;
		MPI_Comm leaders_comm;
		MPI_Group_incl(world_group, mpi_ngroups, group_leader, &leader_group);
		MPI_Comm_create(world_comm, leader_group, &leaders_comm);

		MPI_Fint fortran_comm = MPI_Comm_c2f(leaders_comm);

		int follower_rank = (group_id==0) ? -1 : group_num;

		nested::run(fortran_comm, follower_rank, mpi_ngroups, IS, mmodal, ceff, nlive, tol, efr, n_fitparams, nPar, nClsPar, maxModes, updInt, Ztol, filename.c_str(), seed, pWrap, fb, resume, outfile, initMPI, logZero, maxiter, multinest_loglikelihood, dumper_multinest, context);
#else
		nested::run(IS, mmodal, ceff, nlive, tol, efr, n_fitparams, nPar, nClsPar, maxModes, updInt, Ztol, filename.c_str(), seed, pWrap, fb, resume, outfile, initMPI, logZero, maxiter, multinest_loglikelihood, dumper_multinest, context);
#endif
	}

	bestfitparams.input(n_fitparams);

	use_ansi_characters = false;

#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for nested sampling: " << wt << endl;
	}
#endif

	// Now convert the MultiNest output to a form that mkdist can read
	double lnZ = -1e30;
	double *xparams;
	double *params;
	double *covs;
	double *avgs;
	double minchisq = 1e30;
	int cont = 1;
	bool using_livepts_file = false;
	if (mpi_id==0) {
		cout << endl;
		string stats_filename = filename + "stats.dat";
		ifstream stats_in(stats_filename.c_str());
		if (!(stats_in.is_open())) warn("MultiNest output file %sstats.dat could not be found; evidence undetermined",filename.c_str());
		string dum;
		for (int i=0; i < 5; i++) {
			stats_in >> dum;
		}
		//double area=1.0;
		stats_in >> lnZ;
		stats_in.close();
		//for (int i=0; i < n_fitparams; i++) area *= (param_list->prior_limits_hi[i]-param_list->prior_limits_lo[i]);
		//lnZ += log(area);

		const int n_characters = 16384;
		char line[n_characters];

		string mnin_filename = filename + ".txt";
		ifstream mnin(mnin_filename.c_str());
		if (!(mnin.is_open())) {
			if (!(mnin.is_open())) {
				warn("MultiNest output file %s could not be found; will look for live points log file",mnin_filename.c_str());
			}
			mnin_filename = filename + "live.points";
			mnin.open(mnin_filename.c_str());
			using_livepts_file = true;
		}
		if (!(mnin.is_open())) {
			warn("MultiNest output file %s could not be found; chain cannot be processed",mnin_filename.c_str());
			cont = 0;
		} else {
			ofstream mnout(filename.c_str());
			mnout << setprecision(16);
			if (data_info != "") mnout << "# DATA_INFO: " << data_info << endl;
			if (chain_info != "") mnout << "# CHAIN_INFO: " << chain_info << endl;
			mnout << "# Sampler: MultiNest, n_livepts = " << n_livepts << endl;
			mnout << "# lnZ = " << lnZ << endl;
			if (calculate_bayes_factor) {
				if (reference_lnZ==-1e30) reference_lnZ = lnZ; // first model being fit, so Bayes factor doesn't get calculated yet
				else {
					double log_bayes_factor = lnZ - reference_lnZ;
					mnout << "# Bayes factor: ln(Z/Z_ref) = " << log_bayes_factor << " Z/Z_ref = " << exp(log_bayes_factor) << " (lnZ_ref=" << reference_lnZ << ")" << endl;
					reference_lnZ = lnZ;
				}
			}

			double weight, chi2;
			int n_tot_params = n_fitparams + n_derived_params;
			xparams = new double[n_tot_params];
			params = new double[n_tot_params];
			covs = new double[n_tot_params];
			avgs = new double[n_tot_params];
			int i;
			double weighttot = 0;
			for (int i=0; i < n_tot_params; i++) {
				covs[i] = 0;
				avgs[i] = 0;
			}
			while ((mnin.getline(line,n_characters)) and (!mnin.eof())) {
				istringstream instream(line);
				if (!using_livepts_file) {
					instream >> weight;
					instream >> chi2;
				} else {
					weight = 0.0;
				}
				for (i=0; i < n_tot_params; i++) instream >> xparams[i];
				transform_cube(params,xparams);
				mnout << weight << "   ";
				for (i=0; i < n_fitparams; i++) mnout << params[i] << "   ";
				for (i=0; i < n_derived_params; i++) mnout << xparams[n_fitparams+i] << "   ";
				if (using_livepts_file) {
					double negloglike;
					instream >> negloglike;
					chi2 = -2*negloglike;
				}
				mnout << chi2 << endl;
				if (chi2 < minchisq) {
					minchisq = chi2;
					for (i=0; i < n_fitparams; i++) bestfitparams[i] = params[i];
				}
				for (i=0; i < n_tot_params; i++) {
					avgs[i] += weight*params[i];
					covs[i] += weight*params[i]*params[i];
				}
				weighttot += weight;
			}
			mnin.close();
			for (i=0; i < n_tot_params; i++) {
				if (weighttot==0.0) {
					avgs[i] = 0;
					covs[i] = 0;
				} else {
					avgs[i] /= weighttot;
					covs[i] = covs[i]/weighttot - avgs[i]*avgs[i];
				}
			}
		}
	}

#ifdef USE_MPI
		MPI_Bcast(&cont,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
	if (cont == 0) return;
#ifdef USE_MPI
		MPI_Bcast(bestfitparams.array(),n_fitparams,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	if (group_num==0) {
		chisq_bestfit = 2*(this->*LogLikePtr)(bestfitparams.array());
	}

	if (mpi_id==0) {
		cout << endl << "Log-evidence: ln(Z) = " << lnZ << endl;
		cout << "\nBest-fit parameters and error estimates (from dispersions of chain output points):    (chisq=" << minchisq << ")\n";
		if (using_livepts_file) {
			for (int i=0; i < n_fitparams; i++) {
				cout << param_list->param_names[i] << ": " << bestfitparams[i] << endl;
			}
		} else {
			for (int i=0; i < n_fitparams; i++) {
				cout << param_list->param_names[i] << ": " << bestfitparams[i] << " +/- " << sqrt(covs[i]) << endl;
			}
		}
		cout << endl;
		output_bestfit_model();
		delete[] xparams;
		delete[] params;
		delete[] avgs;
		delete[] covs;
	}
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
#endif
}

void QLens::polychord(const bool resume_previous, const bool skip_run)
{
	fitmethod = POLYCHORD;
	if (!param_list->all_prior_limits_defined()) { warn("not all prior limits have been defined"); return; }
#ifdef USE_POLYCHORD
	if (fit_set_optimizations()==false) return;
	if ((mpi_id==0) and (!resume_previous) and (!skip_run) and (fit_output_dir != ".")) {
		// I should probably give the nested sampling output a unique extension like ".nest" or something, so that mkdist can't ever confuse it with twalk output in the same dir
		// Do this later...

#if __cplusplus >= 201703L // C++17 standard or later
		if (filesystem::exists(fit_output_dir)) {
			filesystem::remove_all(fit_output_dir);
		}
#else
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		if (system(rmstring.c_str()) != 0) warn("could not delete old output directory for nested sampling results"); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
#endif
		create_output_directory();
	}

	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	int n_fitparams = param_list->nparams;
	int n_derived_params = dparam_list->n_dparams;
	if (mpi_id==0) {
		// This code gets repeated in a few spots and should really be put in a separate function...DO THIS LATER!
		int i;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fitparams << " " << n_derived_params << endl;
		pnumfile.close();
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_fitparams; i++) pnamefile << param_list->param_names[i] << endl;
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list->dparams[i]->name << endl;
		pnamefile.close();
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_fitparams; i++) lpnamefile << param_list->param_names[i] << "\t" << param_list->latex_names[i] << endl;
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list->dparams[i]->name << "\t" << dparam_list->dparams[i]->latex_name << endl;
		lpnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < n_fitparams; i++)
		{
			prangefile << param_list->prior_limits_lo[i] << " " << param_list->prior_limits_hi[i] << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		if (param_markers != "") {
			string marker_str = fit_output_dir + "/" + fit_output_filename + ".markers";
			ofstream markerfile(marker_str.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	display_chisq_status = false; // just in case it was turned on

	use_ansi_characters = true;

	mcsampler_set_lensptr(this);
	Settings settings(n_fitparams,n_derived_params);

	settings.nlive         = n_livepts;
	settings.num_repeats   = n_fitparams*polychord_nrepeats;
	settings.do_clustering = (multimodal_sampling) ? true : false;

	settings.precision_criterion = 1e-3;
	settings.logzero = -1e30;

	settings.base_dir      = fit_output_dir.c_str();
	settings.file_root     = fit_output_filename.c_str();

	settings.write_resume  = true;
	settings.read_resume   = resume_previous;
	settings.write_live    = true;
	settings.write_dead    = true;
	settings.write_stats   = true;

	settings.equals        = false;
	settings.posteriors    = true;
	settings.cluster_posteriors = false;

	settings.feedback      = 3;
	settings.compression_factor  = 0.36787944117144233;

	settings.boost_posterior= 1.0;

	if (!skip_run) {
		run_polychord(polychord_loglikelihood,polychord_prior,polychord_dumper,settings);
	}

	use_ansi_characters = false;

#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for nested sampling: " << wt << endl;
	}
#endif

	bestfitparams.input(n_fitparams);
	// Now convert the PolyChord output to a form that mkdist can read
	double *params, *covs, *avgs;
	double lnZ;
	if (mpi_id==0) {
		const int n_characters = 16384;
		char line[n_characters];

		string filename = fit_output_dir + "/" + fit_output_filename;
		string stats_filename = filename + ".stats";
		ifstream stats_in(stats_filename.c_str());
		int i;
		for (i=0; i < 8; i++) stats_in.getline(line,n_characters); // skip past beginning lines
		string dum;
		for (i=0; i < 2; i++) {
			stats_in >> dum;
		}
		//double area=1.0;
		stats_in >> lnZ;
		stats_in.close();
		//for (i=0; i < n_fitparams; i++) area *= (param_list->prior_limits_hi[i]-param_list->prior_limits_lo[i]);
		//lnZ += log(area);

		string polyin_filename = filename + ".txt";
		ifstream polyin(polyin_filename.c_str());
		ofstream polyout(filename.c_str());
		polyout << setprecision(16);
		if (data_info != "") polyout << "# DATA_INFO: " << data_info << endl;
		if (chain_info != "") polyout << "# CHAIN_INFO: " << chain_info << endl;
		polyout << "# Sampler: PolyChord, n_livepts = " << n_livepts << endl;
		polyout << "# lnZ = " << lnZ << endl;
		if (calculate_bayes_factor) {
			if (reference_lnZ==-1e30) reference_lnZ = lnZ; // first model being fit, so Bayes factor doesn't get calculated yet
			else {
				double log_bayes_factor = lnZ - reference_lnZ;
				polyout << "# Bayes factor: ln(Z/Z_ref) = " << log_bayes_factor << " Z/Z_ref = " << exp(log_bayes_factor) << " (lnZ_ref=" << reference_lnZ << ")" << endl;
				reference_lnZ = lnZ;
			}
		}

		double weight, chi2;
		double minchisq = 1e30;
		int n_tot_params = n_fitparams + n_derived_params;
		params = new double[n_tot_params];
		covs = new double[n_tot_params];
		avgs = new double[n_tot_params];
		double weighttot = 0;
		for (int i=0; i < n_tot_params; i++) {
			covs[i] = 0;
			avgs[i] = 0;
		}
		istringstream linestream;
		int ncols = n_tot_params + 2;
		stringstream *colstr = new stringstream[ncols];
		string *colstring = new string[ncols];
		while ((polyin.getline(line,n_characters)) and (!polyin.eof())) {
			linestream.clear();
			linestream.str(line);
			for (i=0; i < ncols; i++) linestream >> colstring[i];
			for (i=0; i < ncols; i++) {
				colstr[i].clear();
				colstr[i].str(colstring[i]);
			}
			colstr[0] >> weight;
			colstr[1] >> chi2;
			for (i=0; i < n_tot_params; i++) colstr[i+2] >> params[i];
			polyout << weight << "   ";
			for (i=0; i < n_tot_params; i++) polyout << params[i] << "   ";
			polyout << chi2 << endl;
			if (chi2 < minchisq) {
				minchisq = chi2;
				for (i=0; i < n_fitparams; i++) bestfitparams[i] = params[i];
			}
			for (i=0; i < n_tot_params; i++) {
				avgs[i] += weight*params[i];
				covs[i] += weight*params[i]*params[i];
			}
			weighttot += weight;
		}
		polyin.close();
		for (i=0; i < n_tot_params; i++) {
			avgs[i] /= weighttot;
			covs[i] = covs[i]/weighttot - avgs[i]*avgs[i];
		}
		delete[] colstr;
		delete[] colstring;
	}

#ifdef USE_MPI
	MPI_Bcast(bestfitparams.array(),n_fitparams,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	if (group_num==0) {
		chisq_bestfit = 2*(this->*LogLikePtr)(bestfitparams.array());
	}

	if (mpi_id==0) {
		cout << endl << "Log-evidence: ln(Z) = " << lnZ << endl;
		cout << "\nBest-fit parameters and error estimates (from dispersions of chain output points):\n";
		for (int i=0; i < n_fitparams; i++) {
			cout << param_list->param_names[i] << ": " << bestfitparams[i] << " +/- " << sqrt(covs[i]) << endl;
		}
		cout << endl;
		output_bestfit_model();
		delete[] params;
		delete[] avgs;
		delete[] covs;
	}

	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
#endif
}

void QLens::chi_square_twalk()
{
	if (fit_set_optimizations()==false) return;
	if (!param_list->all_prior_limits_defined()) { warn("not all prior limits have been defined"); return; }
	if ((mpi_id==0) and (fit_output_dir != ".")) {
#if __cplusplus >= 201703L // C++17 standard or later
		if (filesystem::exists(fit_output_dir)) {
			filesystem::remove_all(fit_output_dir);
		}
#else
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		if (system(rmstring.c_str()) != 0) warn("could not delete old output directory for twalk results"); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
#endif
		create_output_directory();
	}
	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return;
	}

	int n_fitparams = param_list->nparams;
	int n_derived_params = dparam_list->n_dparams;
	double *fitparams = param_list->values;
	InputPoint(fitparams,param_list->prior_limits_hi,param_list->prior_limits_lo,n_fitparams);
	SetNDerivedParams(n_derived_params);

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	if (mpi_id==0) {
		// This code gets repeated in a few spots and should really be put in a separate function...DO THIS LATER!
		int i;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fitparams << " " << n_derived_params << endl;
		pnumfile.close();
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_fitparams; i++) pnamefile << param_list->param_names[i] << endl;
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list->dparams[i]->name << endl;
		pnamefile.close();
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_fitparams; i++) lpnamefile << param_list->param_names[i] << "\t" << param_list->latex_names[i] << endl;
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list->dparams[i]->name << "\t" << dparam_list->dparams[i]->latex_name << endl;
		lpnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (i=0; i < n_fitparams; i++)
		{
			prangefile << param_list->prior_limits_lo[i] << " " << param_list->prior_limits_hi[i] << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		if (param_markers != "") {
			string marker_str = fit_output_dir + "/" + fit_output_filename + ".markers";
			ofstream markerfile(marker_str.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	string filename = fit_output_dir + "/" + fit_output_filename;

	display_chisq_status = false; // just in case it was turned on

	use_ansi_characters = true;
	TWalk(filename.c_str(),0.9836,4,2.4,2.5,6.0,mcmc_tolerance,mcmc_threads,fitparams,mcmc_logfile,NULL,chain_info,data_info);
	use_ansi_characters = false;
	bestfitparams.input(fitparams,n_fitparams);
	chisq_bestfit = 2*(this->*LogLikePtr)(bestfitparams.array());

#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for T-Walk: " << wt << endl;
	}
#endif
	if (mpi_id==0) {
		output_bestfit_model();
	}

	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
}

bool QLens::adopt_model(dvector &fitpars)
{
	if ((nlens==0) and (n_ptsrc==0) and (n_sb==0)) { if (mpi_id==0) warn(warnings,"No lens/source model has been specified"); return false; }
	int n_fitparams = param_list->nparams;
	if (n_fitparams == 0) { if (mpi_id==0) warn(warnings,"No best-fit point has been saved from a previous fit"); return false; }
	if (fitpars.size() != n_fitparams) {
		if (mpi_id==0) {
			if (fitpars.size()==0) warn(warnings,"fit has not been run; best-fit solution is not available");
			else warn(warnings,"Best-fit number of parameters does not match current number; this likely means your current lens/source model does not match the model that was used for fitting.");
		}
		return false;
	}
	double transformed_params[n_fitparams];
	param_list->update_param_values(fitpars.array());
	param_list->inverse_transform_parameters();
	double log_penalty_prior = update_model(param_list->untransformed_values); // the model is adopted here

	int i;
	if ((n_extended_src_redshifts > 1) and (source_fit_mode != Point_Source)) {
		bool transform_center_coords = false;
		for (i=0; i < nlens; i++) {
			if (lens_list[i]->transform_center_coords_to_pixsrc_frame) {
				transform_center_coords = true;
			}
		}
		if (transform_center_coords) {
			if ((image_pixel_grids == NULL) or (image_pixel_grids[0]==NULL)) {
				for (i=0; i < n_data_bands; i++) load_pixel_grid_from_data(i);
				for (i=0; i < n_extended_src_redshifts; i++) {
					image_pixel_grids[i]->redo_lensing_calculations(false);
				}
			}
			update_lens_centers_from_pixsrc_coords();  
		}
	}

	// Since optimizations sometimes result in angles being out of (-2*pi,2*pi) range, reset them if necessary
	for (i=0; i < nlens; i++) {
		lens_list[i]->reset_angle_modulo_2pi();
	}
	for (i=0; i < n_sb; i++) {
		sb_list[i]->reset_angle_modulo_2pi();
	}
	if ((n_ptsrc > 0) and (use_analytic_bestfit_src)) set_analytic_sourcepts(false);
	if ((n_ptsrc > 0) and (include_flux_chisq) and (analytic_source_flux)) set_analytic_srcflux(false);
	reset_grid(); // this will force it to redraw the critical curves if needed
	if (log_penalty_prior > 0) warn(warnings,"adopted parameters are generating a penalty prior; this may be due to parameters being out of plimit ranges");

	return true;
}

void QLens::output_bestfit_model(const bool show_parameter_errors)
{
	if ((nlens == 0) and (n_sb==0)) { warn(warnings,"No fit model has been specified"); return; }
	int n_fitparams = param_list->nparams;
	int n_derived_params = dparam_list->n_dparams;
	if (n_fitparams == 0) { warn(warnings,"No best-fit point has been saved from a previous fit"); return; }
	if (bestfitparams.size() != n_fitparams) { warn(warnings,"Best-fit point number of params does not match current number"); return; }
	if (fit_output_dir != ".") create_output_directory();

	int i;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ofstream pnumfile(pnumfile_str.c_str());
	pnumfile << n_fitparams << " " << n_derived_params << endl;
	pnumfile.close();
	string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
	ofstream pnamefile(pnamefile_str.c_str());
	for (i=0; i < n_fitparams; i++) pnamefile << param_list->param_names[i] << endl;
	for (i=0; i < n_derived_params; i++) pnamefile << dparam_list->dparams[i]->name << endl;
	pnamefile.close();
	string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
	ofstream lpnamefile(lpnamefile_str.c_str());
	for (i=0; i < n_fitparams; i++) lpnamefile << param_list->param_names[i] << "\t" << param_list->latex_names[i] << endl;
	for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list->dparams[i]->name << "\t" << dparam_list->dparams[i]->latex_name << endl;
	lpnamefile.close();

	string bestfit_filename = fit_output_dir + "/" + fit_output_filename + ".bf";
	int n,j;
	ofstream bf_out(bestfit_filename.c_str());
	bf_out << chisq_bestfit << " ";
	for (i=0; i < n_fitparams; i++) bf_out << bestfitparams[i] << " ";
	bf_out << endl;
	bf_out.close();

	string outfile_str = fit_output_dir + "/" + fit_output_filename + ".bestfit";
	ofstream outfile(outfile_str.c_str());
	if ((show_parameter_errors) and (bestfit_fisher_inverse.is_initialized()))
	{
		if (bestfit_fisher_inverse.rows() != n_fitparams) die("dimension of Fisher matrix does not match number of fit parameters (%i vs %i)",bestfit_fisher_inverse.rows(),n_fitparams);
		string fisher_inv_filename = fit_output_dir + "/" + fit_output_filename + ".pcov"; // inverse-fisher matrix is the parameter covariance matrix
		ofstream fisher_inv_out(fisher_inv_filename.c_str());
		for (i=0; i < n_fitparams; i++) {
			for (j=0; j < n_fitparams; j++) {
				fisher_inv_out << bestfit_fisher_inverse[i][j] << " ";
			}
			fisher_inv_out << endl;
		}

		outfile << "Best-fit model: 2*loglike = " << chisq_bestfit << endl;
		if ((include_flux_chisq) and (bestfit_flux != 0)) outfile << "Best-fit source flux = " << bestfit_flux << endl;
		outfile << endl;
		outfile << "Marginalized 1-sigma errors from Fisher matrix:\n";
		for (int i=0; i < n_fitparams; i++) {
			outfile << param_list->param_names[i] << ": " << bestfitparams[i] << " +/- " << sqrt(abs(bestfit_fisher_inverse[i][i])) << endl;
		}
		outfile << endl;
	} else {
		if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
			outfile << "Best-fit model: 2*loglike = " << chisq_bestfit << " (warning: errors omitted here because Fisher matrix was not calculated):\n";
		} else {
			outfile << "Best-fit model: 2*loglike = " << chisq_bestfit << endl;
		}
		if ((include_flux_chisq) and (bestfit_flux != 0)) outfile << "Best-fit source flux = " << bestfit_flux << endl;
		outfile << endl;
		for (int i=0; i < n_fitparams; i++) {
			outfile << param_list->param_names[i] << ": " << bestfitparams[i] << endl;
		}
		outfile << endl;
	}
	string prange_str = fit_output_dir + "/" + fit_output_filename + ".pranges";
	ofstream prangefile(prange_str.c_str());
	for (int i=0; i < n_fitparams; i++)
	{
		if (param_list->defined_prior_limits[i])
			prangefile << param_list->prior_limits_lo[i] << " " << param_list->prior_limits_hi[i] << endl;
		else
			prangefile << "-1e30 1e30" << endl;
	}
	prangefile.close();
	if (lines.size() > 0) {
		string script_str = fit_output_dir + "/" + fit_output_filename + ".commands";
		ofstream scriptfile(script_str.c_str());
		for (int i=0; i < lines.size()-1; i++) {
			scriptfile << lines[i] << endl;
		}
		scriptfile.close();
	}

	QLens* model;
	if (fitmodel != NULL) model = fitmodel;
	else model = this;
	// In order to save the commands for the best-fit model, we adopt the best-fit model in the fitmodel object (if available);
	// that way we're not forced to adopt it in the user-end lens object if the user doesn't want to
	if (model == fitmodel) {
		model->bestfitparams.input(bestfitparams);
		model->adopt_model(bestfitparams);
	}
	bool include_limits;
	if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) include_limits = false;
	else include_limits = true;
	if (include_limits) {
		// save version without limits in case user wants to load best-fit model while in Simplex or Powell mode
		string scriptfile_str2 = fit_output_dir + "/" + fit_output_filename + "_bf_nolimits.in";
	}
}

bool QLens::add_dparams_to_chain(string file_ext)
{
	if (file_ext=="") file_ext = "new"; // default extension
	// Should check whether any new derived parameters have the same name as one of the old derived parameters--this can happen if one accidently adds the
	// same derived parameters they had before (in addition to some new ones). Have it print an error if this is the case. ADD THIS FEATURE!!!!!!
	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_point_source);
	} else {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&QLens::fitmodel_loglike_extended_source);
	}

	int i, n_fitparams, nparams, n_derived_params, n_dparams_old;
	n_fitparams = param_list->nparams;
	n_derived_params = dparam_list->n_dparams;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams_old;
	if (nparams != n_fitparams) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	pnumfile.close();

	if (fit_set_optimizations()==false) return false;
	if (!initialize_fitmodel(true)) {
		if (mpi_id==0) warn(warnings,"Warning: could not evaluate chi-square function");
		return false;
	}

	static const int n_characters = 5000;
	char dataline[n_characters];
	if (mpi_id==0) {
		int i;
		int n_dparams_tot = n_dparams_old + n_derived_params;
		int n_totparams_old = n_fitparams + n_dparams_old;
		string pnumfile_str = fit_output_dir + "/" + fit_output_filename + "." + file_ext + ".nparam";
		ofstream pnumfile(pnumfile_str.c_str());
		pnumfile << n_fitparams << " " << n_dparams_tot << endl;
		pnumfile.close();

		string pnamefile_old_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + "." + file_ext + ".paramnames";
		ifstream pnamefile_old(pnamefile_old_str.c_str());
		ofstream pnamefile(pnamefile_str.c_str());
		for (i=0; i < n_totparams_old; i++) {
			pnamefile_old.getline(dataline,n_characters);
			pnamefile << dataline << endl;
		}
		for (i=0; i < n_derived_params; i++) pnamefile << dparam_list->dparams[i]->name << endl;
		pnamefile.close();
		pnamefile_old.close();

		string lpnamefile_old_str = fit_output_dir + "/" + fit_output_filename + ".latex_paramnames";
		string lpnamefile_str = fit_output_dir + "/" + fit_output_filename + "." + file_ext + ".latex_paramnames";
		ifstream lpnamefile_old(lpnamefile_old_str.c_str());
		ofstream lpnamefile(lpnamefile_str.c_str());
		for (i=0; i < n_totparams_old; i++) {
			lpnamefile_old.getline(dataline,n_characters);
			lpnamefile << dataline << endl;
		}
		for (i=0; i < n_derived_params; i++) lpnamefile << dparam_list->dparams[i]->name << "\t" << dparam_list->dparams[i]->latex_name << endl;
		lpnamefile.close();
		lpnamefile_old.close();

		string prangefile_old_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		string prangefile_str = fit_output_dir + "/" + fit_output_filename + "." + file_ext + ".ranges";
		ifstream prangefile_old(prangefile_old_str.c_str());
		ofstream prangefile(prangefile_str.c_str());
		for (i=0; i < n_totparams_old; i++) {
			prangefile_old.getline(dataline,n_characters);
			prangefile << dataline << endl;
		}
		for (i=0; i < n_derived_params; i++) prangefile << "-1e30 1e30" << endl;
		prangefile.close();
		prangefile_old.close();
	}

	double *params = new double[n_fitparams];
	double *dparams_old = new double[n_dparams_old];
	double weight, chisq;
	string chain_old_str = fit_output_dir + "/" + fit_output_filename;
	string chain_str = fit_output_dir + "/" + fit_output_filename + "." + file_ext;
	ifstream chain_file_old0(chain_old_str.c_str());

	int j,line,nlines=0;
	while (!chain_file_old0.eof()) {
		chain_file_old0.getline(dataline,n_characters);
		if (dataline[0]=='#') continue;
		nlines++;
	}
	double **dparams_new = new double*[nlines];
	for (i=0; i < nlines; i++) dparams_new[i] = new double[n_derived_params];
	char **chain_lines = new char*[nlines];
	for (i=0; i < nlines; i++) chain_lines[i] = new char[5000];
	chain_file_old0.close();

	chain_file_old0.open(chain_old_str.c_str());
	for (line=0; line < nlines; line++) {
		chain_file_old0.getline(chain_lines[line],n_characters);
		if (chain_lines[line][0]=='#') { line--; continue; }
	}

	int nlines_chunk = nlines/20;
	if (mpi_id==0) cout << "Calculating derived parameters: [\033[20C]" << endl << endl << flush;
	int prev_icount, icount = 0;
	for (line=group_num; line < nlines; line += mpi_ngroups) {
		istringstream datastream(chain_lines[line]);
		datastream >> weight;
		for (i=0; i < n_fitparams; i++) {
			datastream >> params[i];
		}
		fitmodel_calculate_derived_params(params, dparams_new[line]);

		prev_icount = icount;
		icount = line/nlines_chunk;
		if ((mpi_id==0) and (prev_icount != icount)) {
			cout << "\033[2ACalculating derived parameters: [" << flush;
			for (j=0; j < icount; j++) cout << "=" << flush;
			cout << "\033[1B" << endl << flush;
		}
	}
	if (mpi_id==0) {
		cout << "\033[2ACalculating derived parameters: [" << flush;
		for (j=0; j < 20; j++) cout << "=" << flush;
		cout << "\033[1B" << endl << flush;
	}
	if (mpi_id==0) cout << endl;
	//cout << "icount=" << icount << " prev=" << prev_icount << "line=" << line << " chunk=" << nlines_chunk << endl;

#ifdef USE_MPI
	int id;
	for (int groupnum=0; groupnum < mpi_ngroups; groupnum++) {
		for (i=groupnum; i < nlines; i += mpi_ngroups) {
			id = group_leader[groupnum];
			MPI_Bcast(dparams_new[i],n_derived_params,MPI_DOUBLE,id,MPI_COMM_WORLD);
		}
	}
#endif

	if (mpi_id==0) {
		ofstream chain_file(chain_str.c_str());
		for (line=0; line < nlines; line++) {
			istringstream datastream(chain_lines[line]);
			datastream >> weight;
			chain_file << weight << "   ";
			for (i=0; i < n_fitparams; i++) {
				datastream >> params[i];
				chain_file << params[i] << "   ";
			}
			for (i=0; i < n_dparams_old; i++) {
				datastream >> dparams_old[i];
				chain_file << dparams_old[i] << "   ";
			}
			datastream >> chisq;
			for (i=0; i < n_derived_params; i++) chain_file << dparams_new[line][i] << "   ";
			chain_file << chisq << endl;
		}
		chain_file.close();
	}

	delete[] params;
	delete[] dparams_old;
	for (i=0; i < nlines; i++) {
		delete[] chain_lines[i];
		delete[] dparams_new[i];
	}
	delete[] dparams_new;
	delete[] chain_lines;
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return true;
}

bool QLens::adopt_bestfit_point_from_chain()
{
	int i, nparams, n_dparams, n_tot_parameters;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams;
	n_tot_parameters = nparams + n_dparams;
	if (nparams != param_list->nparams) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	pnumfile.close();

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[n_tot_parameters];

	string chain_str = fit_output_dir + "/" + fit_output_filename;
	ifstream chain_file(chain_str.c_str());

	unsigned long nline=0, line_num;
	double weight, max_weight = -1e30;
	double chisq, minchisq = 1e30;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		if (dataline[0]=='#') { nline++; continue; }
		istringstream datastream(dataline);
		datastream >> weight;
		// note that we have to go through all the derived parameters before we get to the chisq column, even though we don't actually need the derived parameters here
		for (i=0; i < n_tot_parameters; i++) {
			datastream >> params[i];
		}
		datastream >> chisq;
		if (chisq < minchisq) {
			//max_weight = weight;
			minchisq = chisq;
			line_num = nline;
		}

		nline++;
	}

	chain_file.close();
	chain_file.open(chain_str.c_str());
	for (i=0; i <= line_num; i++) {
		chain_file.getline(dataline,n_characters);
	}
	if (dataline[0]=='#') { warn("line from chain file is a comment line"); delete[] params; return false; }
	istringstream datastream(dataline);
	datastream >> weight;
	for (i=0; i < n_tot_parameters; i++) {
		datastream >> params[i];
	}
	datastream >> chisq;

	if (mpi_id==0) cout << "Line number of point adopted: " << line_num << " (out of " << nline << " total lines); -2*loglike = " << chisq << ")" << endl;
	//if (max_weight==-1e30) { warn("no points from chain fell within range min/max values for specified parameter"); return false; }
	if (minchisq==1e30) { warn("no points from chain fell within range min/max values for specified parameter"); return false; }

	dvector chain_params(params,param_list->nparams);
	adopt_model(chain_params);

	delete[] params;
	return true;
}

bool QLens::load_bestfit_model(const bool custom_filename, string fit_filename)
{
	if (!custom_filename) fit_filename = fit_output_filename;
	int i, nparams;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams;
	int n_fitparams = param_list->nparams;
	if (nparams != n_fitparams) { warn("number of fit parameters in qlens does not match corresponding number in best-fit model file"); return false; }
	pnumfile.close();

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[n_fitparams];

	string bf_str = fit_output_dir + "/" + fit_output_filename + ".bf";
	ifstream bf_file(bf_str.c_str());

	bf_file.getline(dataline,n_characters);
	if (dataline[0]=='#') { warn("encountered a comment line in best-fit parameter file"); delete[] params; return false; }
	istringstream datastream(dataline);
	for (i=0; i < nparams; i++) {
		datastream >> params[i];
	}

	dvector bf_params(params,n_fitparams);
	adopt_model(bf_params);

	delete[] params;
	return true;
}

bool QLens::adopt_point_from_chain(const unsigned long line_num)
{
	int i, nparams, n_dparams_old;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams_old;
	int n_fitparams = param_list->nparams;
	if (nparams != n_fitparams) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	pnumfile.close();

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[n_fitparams];

	string chain_str = fit_output_dir + "/" + fit_output_filename;
	ifstream chain_file(chain_str.c_str());

	unsigned long nlines=0;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		//if (dataline[0]=='#') cout << string(dataline) << endl;
		nlines++;
	}
	if (nlines < line_num) {
		warn("number of points in chain (%i) is less than the point number requested (%i)",nlines,line_num); delete[] params; return false;
	}

	chain_file.close();
	chain_file.open(chain_str.c_str());
	for (i=1; i <= line_num; i++) {
		chain_file.getline(dataline,n_characters);
	}
	if (dataline[0]=='#') { warn("line from chain file is a comment line"); delete[] params; return false; }
	istringstream datastream(dataline);
	double weight;
	datastream >> weight;
	for (i=0; i < n_fitparams; i++) {
		datastream >> params[i];
	}
	dvector chain_params(params,n_fitparams);
	adopt_model(chain_params);

	delete[] params;
	return true;
}

bool QLens::adopt_point_from_chain_paramrange(const int paramnum, const double minval, const double maxval)
{
	int i, nparams, n_dparams, n_tot_parameters, ndparam;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams;
	n_tot_parameters = nparams + n_dparams;
	int n_fitparams = param_list->nparams;
	if (nparams != n_fitparams) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	if (paramnum >= n_tot_parameters) { warn("parameter number is less than number of fit+derived parameters in chain"); return false; }
	pnumfile.close();
	if (paramnum >= n_fitparams) {
		ndparam = paramnum - n_fitparams;
		if (mpi_id==0) cout << "The parameter selected is derived parameter no. " << ndparam << endl;
	}

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[n_tot_parameters];

	string chain_str = fit_output_dir + "/" + fit_output_filename;
	ifstream chain_file(chain_str.c_str());

	unsigned long nline=0, line_num;
	double weight, max_weight = -1e30;
	double chisq, minchisq = 1e30;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		//if (dataline[0]=='#') cout << string(dataline) << endl;
		if (dataline[0]=='#') { nline++; continue; }
		istringstream datastream(dataline);
		datastream >> weight;
		for (i=0; i < n_tot_parameters; i++) {
			datastream >> params[i];
		}
		datastream >> chisq;
		//cout << params[paramnum] << endl;
		if ((params[paramnum] > minval) and (params[paramnum] < maxval) and (chisq < minchisq)) {
			//max_weight = weight;
			minchisq = chisq;
			line_num = nline;
		}

		nline++;
	}
	//if (max_weight==-1e30) { warn("no points from chain fell within range min/max values for specified parameter"); return false; }
	if (minchisq==1e30) { warn("no points from chain fell within range min/max values for specified parameter"); delete[] params; return false; }

	chain_file.close();
	chain_file.open(chain_str.c_str());
	for (i=0; i <= line_num; i++) {
		chain_file.getline(dataline,n_characters);
	}
	if (dataline[0]=='#') { warn("line from chain file is a comment line"); delete[] params; return false; }
	istringstream datastream(dataline);
	datastream >> weight;
	for (i=0; i < n_tot_parameters; i++) {
		datastream >> params[i];
	}
	datastream >> chisq;
	if (mpi_id==0) cout << "Line number of point adopted: " << line_num << " (out of " << nline << " total lines); chisq=" << chisq << endl;
	dvector chain_params(params,n_fitparams);
	adopt_model(chain_params);

	delete[] params;
	return true;
}

bool QLens::plot_kappa_profile_percentiles_from_chain(int lensnum, double rmin, double rmax, int nbins, const string kappa_filename)
{
	double zl = lens_list[lensnum]->get_redshift();
	double r, rstep = pow(rmax/rmin, 1.0/(nbins-1));
	double *rvals = new double[nbins];
	int i;
	for (i=0, r=rmin; i < nbins; i++, r *= rstep) {
		rvals[i] = r;
	}

	int nparams, n_dparams_old;
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	pnumfile >> nparams >> n_dparams_old;
	int n_fitparams = param_list->nparams;
	if (nparams != n_fitparams) { warn("number of fit parameters in qlens does not match corresponding number in chain"); return false; }
	pnumfile.close();

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[n_fitparams];

	string chain_str = fit_output_dir + "/" + fit_output_filename;
	ifstream chain_file(chain_str.c_str());

	unsigned long n_points=0;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		if (dataline[0]=='#') continue;
		n_points++;
	}
	chain_file.close();

	double **weights = new double*[nbins];
	double **weights2 = new double*[nbins];
	double **kappa_r_pts = new double*[nbins];
	double **kappa_avg_pts = new double*[nbins];
	for (i=0; i < nbins; i++) {
		kappa_r_pts[i] = new double[n_points];
		kappa_avg_pts[i] = new double[n_points];
		weights[i] = new double[n_points];
		weights2[i] = new double[n_points];
	}
	double *kappa_r_vals = new double[nbins];
	double *kappa_avg_vals = new double[nbins];

	chain_file.open(chain_str.c_str());
	int j=0;
	double weight, tot=0;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		if (dataline[0]=='#') continue;

		istringstream datastream(dataline);
		datastream >> weight;
		tot += weight;
		for (i=0; i < n_fitparams; i++) {
			datastream >> params[i];
		}
		dvector chain_params(params,n_fitparams);
		adopt_model(chain_params);
		//print_lens_list(false);
		lens_list[lensnum]->plot_kappa_profile(nbins,rvals,kappa_r_vals,kappa_avg_vals);
		for (i=0; i < nbins; i++) {
			kappa_r_pts[i][j] = kappa_r_vals[i];
			kappa_avg_pts[i][j] = kappa_avg_vals[i];
			//cout << "RK: " << rvals[i] << " " << kappa_r_pts[i][j] << endl;
			weights[i][j] = weight;
			weights2[i][j] = weight;
		}

		j++;
	}
	chain_file.close();

	for (i=0; i < nbins; i++) {
		sort(n_points,kappa_r_pts[i],weights[i]);
		sort(n_points,kappa_avg_pts[i],weights2[i]);
	}
	double kaplo1, kaplo2, kaphi1, kaphi2;
	double kaplo1_prev, kaplo2_prev, kaphi1_prev, kaphi2_prev;
	double slope_lo1, slope_lo2, slope_hi1, slope_hi2;
	double kavglo1, kavglo2, kavghi1, kavghi2;
	double mavglo1, mavglo2, mavghi1, mavghi2;
	ofstream outfile(kappa_filename.c_str());
	double sigma_cr_arcsec = cosmo->sigma_crit_arcsec(zl, reference_source_redshift);
	double arcsec_to_kpc = cosmo->angular_diameter_distance(zl)/(1e-3*(180/M_PI)*3600);
	double rval_kpc;
	for (i=0; i < nbins; i++) {
		kaplo1 = find_percentile(n_points, 0.02275, tot, kappa_r_pts[i], weights[i]);
		kaphi1 = find_percentile(n_points, 0.97725, tot, kappa_r_pts[i], weights[i]);
		kaplo2 = find_percentile(n_points, 0.15865, tot, kappa_r_pts[i], weights[i]);
		kaphi2 = find_percentile(n_points, 0.84135, tot, kappa_r_pts[i], weights[i]);
		if (i>0) {
			slope_lo1 = log(kaplo1/kaplo1_prev)/log(rvals[i]/rvals[i-1]);
			slope_hi1 = log(kaphi1/kaphi1_prev)/log(rvals[i]/rvals[i-1]);
			slope_lo2 = log(kaplo2/kaplo2_prev)/log(rvals[i]/rvals[i-1]);
			slope_hi2 = log(kaphi2/kaphi2_prev)/log(rvals[i]/rvals[i-1]);
		}
		kaplo1_prev = kaplo1;
		kaphi1_prev = kaphi1;
		kaplo2_prev = kaplo2;
		kaphi2_prev = kaphi2;
		if (i==0) {
			// stupid hack but it works
			kaplo1 = find_percentile(n_points, 0.02275, tot, kappa_r_pts[1], weights[1]);
			kaphi1 = find_percentile(n_points, 0.97725, tot, kappa_r_pts[1], weights[1]);
			kaplo2 = find_percentile(n_points, 0.15865, tot, kappa_r_pts[1], weights[1]);
			kaphi2 = find_percentile(n_points, 0.84135, tot, kappa_r_pts[1], weights[1]);
			slope_lo1 = log(kaplo1/kaplo1_prev)/log(rvals[i]/rvals[i-1]);
			slope_hi1 = log(kaphi1/kaphi1_prev)/log(rvals[i]/rvals[i-1]);
			slope_lo2 = log(kaplo2/kaplo2_prev)/log(rvals[i]/rvals[i-1]);
			slope_hi2 = log(kaphi2/kaphi2_prev)/log(rvals[i]/rvals[i-1]);
		}
		kavglo1 = find_percentile(n_points, 0.02275, tot, kappa_avg_pts[i], weights2[i]);
		kavghi1 = find_percentile(n_points, 0.97725, tot, kappa_avg_pts[i], weights2[i]);
		kavglo2 = find_percentile(n_points, 0.15865, tot, kappa_avg_pts[i], weights2[i]);
		kavghi2 = find_percentile(n_points, 0.84135, tot, kappa_avg_pts[i], weights2[i]);
		mavglo1 = kavglo1*M_PI*SQR(rvals[i])*sigma_cr_arcsec;
		mavghi1 = kavghi1*M_PI*SQR(rvals[i])*sigma_cr_arcsec;
		mavglo2 = kavglo2*M_PI*SQR(rvals[i])*sigma_cr_arcsec;
		mavghi2 = kavghi2*M_PI*SQR(rvals[i])*sigma_cr_arcsec;
		rval_kpc = rvals[i]*arcsec_to_kpc;
		outfile << rvals[i] << " " << rval_kpc << " " << kaplo1 << " " << kaphi1 << " " << kaplo2 << " " << kaphi2 << " " << kavglo1 << " " << kavghi1 << " " << kavglo2 << " " << kavghi2 << " " << mavglo1 << " " << mavghi1 << " " << mavglo2 << " " << mavghi2 << " " << slope_lo1 << " " << slope_hi1 << " " << slope_lo2 << " " << slope_hi2 << endl;
	}

	delete[] params;
	delete[] rvals;
	delete[] kappa_r_vals;
	delete[] kappa_avg_vals;
	for (i=0; i < nbins; i++) {
		delete[] kappa_r_pts[i];
		delete[] kappa_avg_pts[i];
		delete[] weights[i];
		delete[] weights2[i];
	}
	delete[] kappa_r_pts;
	delete[] kappa_avg_pts;
	delete[] weights;
	delete[] weights2;
	return true;
}

double QLens::find_percentile(const unsigned long npoints, const double pct, const double tot, double *pts, double *weights)
{
	double totsofar = 0;
	for (int j = 0; j < npoints; j++)
	{
		totsofar += weights[j];
		if (totsofar/tot >= pct)
		{
			return pts[j] + (pts[j-1] - pts[j])*(totsofar - pct*tot)/weights[j];
		}
	}
	return 0.0;
}

void QLens::make_histograms(bool copy_post_files, string posts_dirname, const int nbins_1d, const int nbins_2d, bool copy_subplot_only, bool resampled_posts, bool no2dposts, bool nohists, bool use_fisher_matrix, bool run_python_script)
{
	auto file_exists = [](const string &filename)
	{
		ifstream infile(filename.c_str());
		bool exists = infile.good();
		infile.close();
		return exists;
	};

	auto adjust_ranges_to_include_markers = [](double *minvals, double *maxvals, double *markers, const int n_markers)
	{
		const double extra_length_frac = 0.05;
		for (int i=0; i < n_markers; i++) {
			if (minvals[i] > markers[i]) {
				if ((maxvals[i]-markers[i]) > 4*(maxvals[i]-minvals[i])) warn("marker %i is WAY out of range of parameter chain; will not show marker",i);
				else {
					minvals[i] = markers[i];
					minvals[i] -= extra_length_frac*(maxvals[i]-minvals[i]);
				}
			}
			else if (maxvals[i] < markers[i]) {
				if ((markers[i]-minvals[i]) > 4*(maxvals[i]-minvals[i])) warn("marker %i is WAY out of range of parameter chain; will not show marker",i);
				else {
					maxvals[i] = markers[i];
					maxvals[i] += extra_length_frac*(maxvals[i]-minvals[i]);
				}
			}
		}
	};

	int i,j;
	int n_threads=1, n_processes=1;
	bool show_markers = false;
	bool make_subplot = param_list->subplot_params_defined();
	string file_root = fit_output_dir + "/" + fit_output_filename;
	string marker_filename;
	string subplot_paramnames_filename;

	if (resampled_posts) file_root += ".new";
	if (posts_dirname == fit_output_dir) {
		cerr << "Error: directory for storing posteriors cannot be the same as the chains directory" << endl;
		return;
	}
	if (param_markers != "") {
		show_markers = true;
		marker_filename = file_root + ".markers";
		if (mpi_id==0) {
			ofstream markerfile(marker_filename.c_str());
			markerfile << param_markers << endl;
			markerfile.close();
		}
	}

	if (make_subplot) {
		string subplot_paramnames_filename = file_root + ".subplot_params";
		if (mpi_id==0) {
			ofstream subplotfile(subplot_paramnames_filename.c_str());
			//int nparams_tot = param_list->nparams + dparam_list->n_dparams;
			for (i=0; i < param_list->nparams; i++) {
				string pname;
				bool pflag = param_list->subplot_param_flag(i,pname);
				subplotfile << pname << " ";
				if (pflag) subplotfile << "1";
				else subplotfile << "0";
				subplotfile << endl;
			}
			for (i=0; i < dparam_list->n_dparams; i++) {
				string pname;
				bool pflag = dparam_list->subplot_param_flag(i,pname);
				subplotfile << pname << " ";
				if (pflag) subplotfile << "1";
				else subplotfile << "0";
				subplotfile << endl;
			}
			subplotfile.close();
		}
	}

	if (mpi_id==0) {
		if (!no2dposts) {
			string hist2d_str = file_root + ".hist2d_params";
			ofstream hist2dfile(hist2d_str.c_str());
			//int nparams_tot = param_list->nparams + dparam_list->n_dparams;
			for (i=0; i < param_list->nparams; i++) {
				string pname;
				bool pflag = param_list->hist2d_param_flag(i,pname);
				hist2dfile << pname << " ";
				if (pflag) hist2dfile << "1";
				else hist2dfile << "0";
				hist2dfile << endl;
			}
			for (i=0; i < dparam_list->n_dparams; i++) {
				string pname;
				bool pflag = dparam_list->hist2d_param_flag(i,pname);
				hist2dfile << pname << " ";
				if (pflag) hist2dfile << "1";
				else hist2dfile << "0";
				hist2dfile << endl;
			}
			hist2dfile.close();
		}
	}

	i=0;
	j=0;
	int cut = -1;
	bool silent = false;
	bool transform_parameters = false;
	bool importance_sampling = false;
	bool smoothing = false;
	bool exclude_derived_params = false;
	bool latex_table_format = false;
	bool make_1d_posts = true;
	bool make_2d_posts = true;
	int n_markers_allowed = 10000;
	bool use_bestfit_markers = false;
	char param_transform_filename[100] = "";
	char prior_weight_filename[100] = "";
	bool make_derived_posterior = false;
	double threshold = 3e-3;
	bool print_marker_values = false;

	//int nparams_subset = -1;
	string filename, istring, jstring;
	int nparams, nparams_eff, n_fitparams = -1;
	if (!use_fisher_matrix) {
		for(;;)
		{
			stringstream jstream;
			jstream << j;
			jstream >> jstring;
			filename = file_root + "_0." + jstring;
			if (file_exists(filename)) j++;
			else break;
		}
		if (j==0) {
			for(;;) {
				stringstream istream;
				istream << i;
				istream >> istring;
				filename = file_root + "_" + istring;
				if (file_exists(filename)) i++;
				else break;
			}
			if (i==0) {
				if (file_exists(file_root)) {
					// in this case the "chain" data is from nested sampling, and no cut needs to be made
					if (cut == -1) cut = 0;
					i++;
				}
				else die("No data files found");
			}
		} else {
			for (;;) {
				stringstream istream;
				istream << i;
				istream >> istring;
				filename = file_root + "_" + istring + ".0";
				if (file_exists(filename)) i++;
				else break;
			}
			if (i==0) {
				if (file_exists(file_root)) {
					i++;
				}
				else die("No data files found");
			}
		}
		if (cut != 0) {
			if (i > 0) n_threads = i; // set number of chains for MCMC data
			if (j > 0) n_processes = j; // set number of MPI processes that were used to produce MCMC data
		}
	} else {
		filename = file_root + ".pcov";
		if (!file_exists(filename)) die("Inverse-Fisher matrix file not found");
	}

	string nparam_filename = file_root + ".nparam";
	ifstream nparam_file(nparam_filename.c_str());
	if (nparam_file.good()) {
		nparam_file >> n_fitparams;
		nparam_file.close();
	}

	McmcEval Eval;
	FisherEval FEval;
	double logev = 1e30;

	if (use_fisher_matrix) {
		FEval.input(file_root.c_str(),silent);
		FEval.get_nparams(nparams);
	}
	else
	{
		bool mpi_silent = true;
		if (mpi_id==0) mpi_silent = silent;
		Eval.input(file_root.c_str(),-1,n_threads,NULL,NULL,logev,n_processes,cut,MULT|LIKE,mpi_silent,n_fitparams,transform_parameters,param_transform_filename,importance_sampling,prior_weight_filename);
		Eval.get_nparams(nparams);
	}
	if (nparams==0) die();
	nparams_eff = nparams;
	if (n_fitparams==-1) n_fitparams = nparams;
	if (exclude_derived_params) nparams_eff = n_fitparams;
	//if (nparams_subset < nparams) {
		//if (nparams_subset > 0) nparams_eff = nparams_subset;
		//else if (nparams_subset == 0) warn("specified subset number of parameters is equal to or less than zero; using all parameters");
	//}

	// Make it so you can turn parameters on/off in this file! This will require revising nparams_eff after the flags are read in
	string *param_names = new string[nparams];
	string paramnames_filename, dummy;
	paramnames_filename = file_root + ".paramnames";
	ifstream paramnames_file(paramnames_filename.c_str());
	for (i=0; i < nparams; i++) {
		if (!(paramnames_file >> param_names[i])) die("not all parameter names are given in file '%s'",paramnames_filename.c_str());
	}
	paramnames_file.close();

	string *latex_param_names = new string[nparams];
	if ((latex_table_format) or (make_1d_posts) or (make_2d_posts)) {
		string latex_paramnames_filename = file_root + ".latex_paramnames";
		ifstream latex_paramnames_file(latex_paramnames_filename.c_str());
		string dummy;
		const int n_characters = 1024;
		char line[n_characters];
		for (i=0; i < nparams; i++) {
			if (!(latex_paramnames_file.getline(line,n_characters))) die("not all parameter names are given in file '%s'",latex_paramnames_filename.c_str());
			istringstream instream(line);
			if (!(instream >> dummy)) die("not all parameter names are given in file '%s'",latex_paramnames_filename.c_str());
			if (!(instream >> latex_param_names[i])) die("not all latex parameter names are given in file '%s'",latex_paramnames_filename.c_str());
			while (instream >> dummy) latex_param_names[i] += " " + dummy;
		}
		latex_paramnames_file.close();
	}

	double *prior_minvals = new double[nparams];
	double *prior_maxvals = new double[nparams];
	for (i=0; i < nparams; i++) {
		prior_minvals[i] = -1e30;
		prior_maxvals[i] = 1e30;
	}
	string paramranges_filename = file_root + ".ranges";
	ifstream paramranges_file(paramranges_filename.c_str());
	if (paramranges_file.is_open()) {
		for (i=0; i < nparams; i++) {
			if (!(paramranges_file >> prior_minvals[i])) die("not all parameter ranges are given in file '%s'",paramranges_filename.c_str());
			if (!(paramranges_file >> prior_maxvals[i])) die("not all parameter ranges are given in file '%s'",paramranges_filename.c_str());
			if (prior_minvals[i] > prior_maxvals[i]) die("cannot have minimum parameter value greater than maximum parameter value in file '%s'",paramranges_filename.c_str());
		}
		paramranges_file.close();
	} else warn("parameter range file '%s' not found",paramranges_filename.c_str());

	if ((make_1d_posts) or (make_2d_posts)) {
		if (!use_fisher_matrix) Eval.transform_parameter_names(param_names, latex_param_names); // should have this option for the Fisher analysis version too

		if (mpi_id==0) {
			string out_paramnames_filename = file_root + ".py_paramnames";
			ofstream paramnames_out(out_paramnames_filename.c_str());
			for (i=0; i < nparams_eff; i++) {
				paramnames_out << param_names[i] << endl;
			}
			paramnames_out.close();

			string out_latex_paramnames_filename = file_root + ".py_latex_paramnames";
			ofstream latex_paramnames_out(out_latex_paramnames_filename.c_str());
			for (i=0; i < nparams_eff; i++) {
				latex_paramnames_out << param_names[i] << "   " << latex_param_names[i] << endl;
			}
			latex_paramnames_out.close();
		}
	}

	bool *hist2d_active_params = new bool[nparams_eff];
	for (i=0; i < nparams_eff; i++) {
		hist2d_active_params[i] = true;
	}
	if (make_2d_posts) {
		string hist2d_paramnames_filename = file_root + ".hist2d_params";
		if (file_exists(hist2d_paramnames_filename)) {
			string *hist2d_param_names = new string[nparams_eff];
			ifstream hist2d_paramnames_file(hist2d_paramnames_filename.c_str());
			for (i=0; i < nparams_eff; i++) {
				if (!(hist2d_paramnames_file >> hist2d_param_names[i])) die("not all hist2d_parameter names are given in file '%s'",hist2d_paramnames_filename.c_str());
				if (hist2d_param_names[i] != param_names[i]) die("hist2d parameter names do not match names given in paramnames file");
				int pflag;
				if (!(hist2d_paramnames_file >> pflag)) die("hist2d parameter flag not given in file '%s'",hist2d_paramnames_filename.c_str());
				if (pflag == 0) hist2d_active_params[i] = false;
				else if (pflag == 1) hist2d_active_params[i] = true;
				else die("invalid hist2d parameter flag in file '%s'; should either be 0 or 1",hist2d_paramnames_filename.c_str());
			}
			hist2d_paramnames_file.close();
			delete[] hist2d_param_names;
		}
	}
	int nparams_eff_2d = 0;
	for (i=nparams_eff-1; i >= 0; i--) {
		if (hist2d_active_params[i]) {
			nparams_eff_2d = i+1;
			break;
		}
	}

	bool *subplot_active_params = new bool[nparams_eff];
	if (make_subplot) {
		string *subplot_param_names = new string[nparams_eff];
		ifstream subplot_paramnames_file(subplot_paramnames_filename.c_str());
		for (i=0; i < nparams_eff; i++) {
			if (!(subplot_paramnames_file >> subplot_param_names[i])) die("not all subplot_parameter names are given in file '%s'",subplot_paramnames_filename.c_str());
			if (subplot_param_names[i] != param_names[i]) die("subplot parameter names do not match names given in paramnames file");
			int pflag;
			if (!(subplot_paramnames_file >> pflag)) die("subplot parameter flag not given in file '%s'",subplot_paramnames_filename.c_str());
			if (pflag == 0) subplot_active_params[i] = false;
			else if (pflag == 1) subplot_active_params[i] = true;
			else die("invalid subplot parameter flag in file '%s'; should either be 0 or 1",subplot_paramnames_filename.c_str());
			if ((subplot_active_params[i]) and (!hist2d_active_params[i])) die("subplot parameter '%s' must also have the hist2d flag set to 'true' in <label>.hist2d_params",subplot_param_names[i].c_str());
		}
		subplot_paramnames_file.close();
		delete[] subplot_param_names;
	}
	
	double *markers = new double[nparams_eff];
	int n_markers = (n_markers_allowed < nparams_eff ? n_markers_allowed : nparams_eff);
	if (show_markers) {
		if (use_bestfit_markers) {
			double *bestfit = new double[nparams];
			Eval.min_chisq_pt(bestfit);
			for (i=0; i < n_markers; i++) markers[i] = bestfit[i];
			delete[] bestfit;
		} else {
			marker_filename = file_root + ".markers";
			ifstream marker_file(marker_filename.c_str());
			for (i=0; i < n_markers; i++) {
				if (!(marker_file >> markers[i])) {
					if (i==0) {
						if (mpi_id==0) cerr << "marker values could not be read from file '" << marker_filename << "'; will not use markers when plotting" << endl;
						show_markers = false;
						break;
					}
					n_markers = i;
					break;
				}
			}
		}
	}

	if (use_fisher_matrix) {
		if (mpi_id==0) {
			if (make_1d_posts) {
				for (i=0; i < nparams_eff; i++) {
					string dist_out;
					dist_out = file_root + "_p_" + param_names[i] + ".dat";
					FEval.MkDist(201, dist_out.c_str(), i);
				}
			}
			if (make_2d_posts) {
				for (i=0; i < nparams_eff_2d; i++) {
					if (hist2d_active_params[i]) {
						for (j=i+1; j < nparams_eff_2d; j++) {
							if (hist2d_active_params[j]) {
								string dist_out;
								dist_out = file_root + "_2D_" + param_names[j] + "_" + param_names[i];
								FEval.MkDist2D(61,61,dist_out.c_str(),i,j);
							}
						}
					}
				}
			}
		}
	}
	else
	{
		double *minvals = new double[nparams];
		double *maxvals = new double[nparams];
		for (i=0; i < nparams; i++) {
			minvals[i] = -1e30;
			maxvals[i] = 1e30;
		}

		if ((make_1d_posts) and (mpi_id==0)) {
			Eval.FindRanges(minvals,maxvals,nbins_1d,threshold);
			if (show_markers) adjust_ranges_to_include_markers(minvals,maxvals,markers,n_markers);
			double rap[20];
			for (i=0; i < nparams_eff; i++) {
				string hist_out;
				hist_out = file_root + "_p_" + param_names[i] + ".dat";
				if (smoothing) Eval.MkHist(minvals[i], maxvals[i], nbins_1d, hist_out.c_str(), i, HIST|SMOOTH, rap);
				else Eval.MkHist(minvals[i], maxvals[i], nbins_1d, hist_out.c_str(), i, HIST, rap);
			}
		}

		if ((make_derived_posterior) and (mpi_id==0)) {
			double rap[20];
			double mean, sig;
			//Eval.setRadius(radius);
			Eval.calculate_derived_param();
			if (smoothing) Eval.DerivedHist(-1e30, 1e30, nbins_1d, (file_root + "_p_derived.dat").c_str(), mean, sig, HIST|SMOOTH, rap);
			else Eval.DerivedHist(-1e30, 1e30, nbins_1d, (file_root + "_p_derived.dat").c_str(), mean, sig, HIST, rap);
			double cl_l1,cl_l2,cl_h1,cl_h2;
			cl_l1 = Eval.derived_cl(0.02275);
			cl_l2 = Eval.derived_cl(0.15865);
			cl_h1 = Eval.derived_cl(0.84135);
			cl_h2 = Eval.derived_cl(0.97725);
			double center,sigma;
			// NOTE: You need to enforce boundaries in FindDerivedSigs, otherwise outlier points will screw up the derived confidence limits
			//Eval.FindDerivedSigs(center,sigma);
			cout << "Confidence limits: " << cl_l1 << " " << cl_l2 << " " << cl_h1 << " " << cl_h2 << endl;
			//cout << "Sig: " << center << " " << sigma << endl;
		}

		if (make_2d_posts) {
			int omp_nthreads = 1;
#ifdef USE_OPENMP
			double wtime, wtime0;
			#pragma omp parallel
			{
				#pragma omp master
				omp_nthreads = omp_get_num_threads();
			}
			wtime0 = omp_get_wtime();
#endif
			bool derived_param_fail = false; // if contours can't be made for a derived parameter, we'll have it drop the derived parameters and try again
			Eval.FindRanges(minvals,maxvals,nbins_2d,threshold);
			if ((!make_1d_posts) and (show_markers)) adjust_ranges_to_include_markers(minvals,maxvals,markers,n_markers);
			do {
				int k, n_2dposts;
				vector<int> post2d_i, post2d_j;
				for (i=0; i < nparams_eff_2d; i++) {
					if (hist2d_active_params[i]) {
						for (j=i+1; j < nparams_eff_2d; j++) {
							if (hist2d_active_params[j]) {
								post2d_i.push_back(i);
								post2d_j.push_back(j);
							}
						}
					}
				}
				n_2dposts = post2d_i.size();
				if (mpi_id==0) {
					cout << "Generating 2D histograms (total of " << n_2dposts << ") with ";
					if (mpi_np > 1) cout << mpi_np << " processes and ";
					cout << omp_nthreads << " threads..." << endl;
				}
#ifdef USE_OPENMP
				int omp_nthreads0=omp_nthreads;
				while (n_2dposts < mpi_np*omp_nthreads) {
					omp_nthreads--;
				}
				if (omp_nthreads < omp_nthreads0) {
					omp_set_num_threads(omp_nthreads);
					if (mpi_id==0) cout << "Too many threads, reducing number of threads to " << omp_nthreads << "..." << endl;
				}
#endif

				int mpi_chunk, mpi_start, mpi_end;
				mpi_chunk = n_2dposts / mpi_np;
				mpi_start = mpi_id*mpi_chunk;
				if (mpi_id == mpi_np-1) mpi_chunk += (n_2dposts % mpi_np); // assign the remainder elements to the last mpi process
				mpi_end = mpi_start + mpi_chunk;

				if (derived_param_fail) derived_param_fail = false;
				#pragma omp parallel for private(i,j,k) schedule(dynamic)
				for (k=mpi_start; k < mpi_end; k++) {
					i = post2d_i[k];
					j = post2d_j[k];
					string hist_out;
					hist_out = file_root + "_2D_" + param_names[j] + "_" + param_names[i];
					if (!Eval.MkHist2D(minvals[i],maxvals[i],minvals[j],maxvals[j],nbins_2d,nbins_2d,hist_out.c_str(),i,j, SMOOTH)) {
						if ((i>=n_fitparams) or (j>=n_fitparams)) {
							derived_param_fail = true;
							warn("producing contours failed for derived parameter; we will drop the derived parameters and try again");
						}
					}
					if (derived_param_fail) i = nparams_eff_2d + 1; // make sure it exits loop
				}

				if (derived_param_fail) nparams_eff_2d = n_fitparams;
			} while (derived_param_fail);
#ifdef USE_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef USE_OPENMP
			wtime = omp_get_wtime() - wtime0;
			if (mpi_id==0) cout << "Wall time for calculating 2D histograms: " << wtime << endl;
#endif
		}

//		if (mpi_id==0) {
//			if (output_min_chisq_point) {
//				Eval.output_min_chisq_pt(param_names);
//			}
//			if (output_min_chisq_point_format2) {
//				Eval.output_min_chisq_pt2(param_names);
//			}
//
//			if (output_mean_and_errors) {
//				Eval.FindRanges(minvals,maxvals,nbins_1d,threshold);
//				string covar_out = file_root + ".cov";
//				double *centers = new double[nparams];
//				double *sigs = new double[nparams];
//				Eval.FindCoVar(covar_out.c_str(),centers,sigs,minvals,maxvals);
//				for (i=0; i < nparams_eff; i++) {
//					// NOTE: The following errors are from standard deviation, not from CL's 
//					cout << param_names[i] << ": " << centers[i] << " +/- " << sigs[i] << endl;
//				}
//				cout << endl;
//				delete[] centers;
//				delete[] sigs;
//			}
//			if (output_cl) {
//				Eval.FindRanges(minvals,maxvals,nbins_1d,threshold);
//				double *halfpct = new double[nparams];
//				double *lowcl = new double[nparams];
//				double *hicl = new double[nparams];
//				int powers_of_ten;
//				if (!silent) {
//					if (cl_2sigma) cout << "50th percentile values and errors (based on 2.5\% and 97.5\% percentiles of marginalized posteriors):\n\n";
//					else cout << "50th percentile values and errors (based on 15.8\% and 84.1\% percentiles of marginalized posteriors):\n\n";
//				}
//				if (fixed_precision) {
//					//cout << resetiosflags(ios::scientific);
//					cout << setprecision(precision);
//					cout << fixed;
//				}
//				for (i=0; i < nparams_eff; i++) {
//					if (cl_2sigma) {
//						lowcl[i] = Eval.cl(0.025,i,minvals[i],maxvals[i]);
//						hicl[i] = Eval.cl(0.975,i,minvals[i],maxvals[i]);
//					} else {
//						lowcl[i] = Eval.cl(0.15865,i,minvals[i],maxvals[i]);
//						hicl[i] = Eval.cl(0.84135,i,minvals[i],maxvals[i]);
//					}
//					halfpct[i] = Eval.cl(0.5,i,minvals[i],maxvals[i]);
//					if (show_uncertainties_as_percentiles) {
//						double lowerr = pct_scaling*(halfpct[i] - lowcl[i]);
//						double hierr = pct_scaling*(hicl[i] - halfpct[i]);
//						double lowpct = halfpct[i] - lowerr;
//						double hipct = halfpct[i] + hierr;
//						//cout << param_names[i] << ": " << halfpct[i] << " " << lowcl[i] << " " << hicl[i] << endl;
//						cout << param_names[i] << ": " << halfpct[i] << " " << lowpct << " " << hipct << endl;
//					} else {
//						if (!latex_table_format) {
//							cout << param_names[i] << ": " << halfpct[i] << " -" << (halfpct[i]-lowcl[i]) << " / +" << (hicl[i] - halfpct[i]) << endl;
//						} else {
//							bool show_as_powers = false;
//							bool increase_precision = false; // do this if a number is less than 0.1
//							double half, hierr, lowerr;
//							half = halfpct[i];
//							hierr = hicl[i] - halfpct[i];
//							lowerr = halfpct[i] - lowcl[i];
//							if (half > 1e4) {
//								show_as_powers = true;
//								powers_of_ten = 0;
//								do {
//									half /= 10;
//									hierr /= 10;
//									lowerr /= 10;
//									powers_of_ten++;
//								} while (half > 10);
//							}
//							else if (abs(half) < 0.1) {
//								increase_precision = true;
//								powers_of_ten = -1;
//								double halfdup = abs(half);
//								do {
//									halfdup *= 10;
//									powers_of_ten--;
//								} while (halfdup < 0.1);
//								cout << setprecision(precision-powers_of_ten-1);
//								cout << fixed;
//							}
//							if (!suppress_latex_names) {
//								cout << "$" << latex_param_names[i];
//								if (show_as_powers) cout << "(10^" << powers_of_ten << ")";
//								cout << "$ & ";
//								if (show_prior_ranges) {
//									cout << defaultfloat;
//									if ((prior_minvals[i] > -1e30) and (prior_maxvals[i] < 1e30)) {
//										cout << "$(" << prior_minvals[i] << "," << prior_maxvals[i] << ")$ & ";
//									} else {
//										cout << "... & ";
//									}
//									cout << fixed;
//								}
//							}
//							cout << "$" << half << "_{-" << lowerr << "}^{+" << hierr << "}$ & " << endl;
//							if (increase_precision) {
//								cout << setprecision(precision);
//								cout << fixed;
//							}
//						}
//					}
//				}
//				if (add_dummy_params) {
//					for (int i=0; i < ndummy; i++) {
//						if (!suppress_latex_names) {
//							cout << "dummy" << i << " & ";
//							if (show_prior_ranges) cout << "... & ";
//						}
//						cout << "... & " << endl;
//					}
//				}
//				if (latex_table_format) {
//					if (logev != 1e30) {
//						if (!suppress_latex_names) {
//							cout << "$\\ln\\mathcal{E}$ & "; 
//							if (show_prior_ranges) cout << "... & ";
//						}
//						cout << logev << " & " << endl;
//					}
//				}
//				cout << endl;
//				delete[] halfpct;
//				delete[] lowcl;
//				delete[] hicl;
//			}
//			if (output_percentile) {
//				Eval.FindRanges(minvals,maxvals,nbins_1d,threshold);
//				// The following gives the 
//				cout << percentile << " percentile:\n\n";
//				double val;
//				for (i=0; i < nparams_eff; i++) {
//					val = Eval.cl(percentile,i,minvals[i],maxvals[i]);
//					cout << param_names[i] << " = " << val << endl;
//				}
//				cout << endl;
//			}
//			if (print_marker_values) {
//				if (show_markers) {
//					cout << "True parameter values (and bestfit values):\n";
//					for (i=0; i < n_markers; i++) {
//						// NOTE: The following errors are from standard deviation, not from CL's 
//						cout << param_names[i] << ": " << markers[i] << " (" << Eval.output_min_chisq_value(i) << ")" << endl;
//					}
//					cout << endl;
//				}
//			}
//		}

		delete[] minvals;
		delete[] maxvals;
	}

	bool add_title = false;
	bool include_shading = true;
#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	if (mpi_id==0) {
		int system_returnval;
		if (make_1d_posts)
		{
			string pyname = fit_output_filename + ".py";
			ofstream pyscript(pyname.c_str());
			pyscript << "import GetDistPlots, os" << endl;
			pyscript << "g=GetDistPlots.GetDistPlotter('" << fit_output_dir << "/')" << endl;
			pyscript << "g.settings.setSubplotSize(3.0000,width_scale=1.0)  # width_scale scales the width of all lines in the plot" << endl;
			pyscript << "outdir=''" << endl;
			pyscript << "roots=['" << fit_output_filename << "']" << endl;
			if (show_markers) {
				pyscript << "marker_list=[";
				for (i=0; i < n_markers; i++) {
					pyscript << markers[i];
					if (i < nparams_eff-1) pyscript << ",";
				}
				pyscript << "]" << endl;
			} else {
				pyscript << "marker_list=[]   # put parameter values in this list if you want to mark the 'true' or best-fit values on posteriors" << endl;
			}
			pyscript << "g.plots_1d(roots,markers=marker_list,marker_color='orange')" << endl;
			//if (add_title) pyscript << "g.add_title(r'" << plot_title << "')" << endl; // 1d title doesn't look good
			pyscript << "g.export(os.path.join(outdir,'" << fit_output_filename << ".pdf'))" << endl;
			pyscript.close();
			if (run_python_script) {
				string pycommand = "python3 " + pyname;
				if (system(pycommand.c_str()) == 0) {
					cout << "Plot for 1D posteriors saved to '" << fit_output_filename << ".pdf'\n";
					//string rmcommand = "rm " + pyname;
					//system_returnval = system(rmcommand.c_str());
				}
				else cout << "Error: Could not generate PDF file for 1D posteriors\n";
			} else {
				cout << "Plotting script for 1D posteriors saved to '" << pyname << "'\n";
			}
		}

		if (make_2d_posts)
		{
			string pyname = fit_output_filename + "_2D.py";
			ofstream pyscript2d(pyname.c_str());
			pyscript2d << "import GetDistPlots, os" << endl;
			pyscript2d << "g=GetDistPlots.GetDistPlotter('" << fit_output_dir << "/')" << endl;
			pyscript2d << "g.settings.setSubplotSize(3.0000,width_scale=1.0)  # width_scale scales the width of all lines in the plot" << endl;
			pyscript2d << "outdir=''" << endl;
			pyscript2d << "roots=['" << fit_output_filename << "']" << endl;
			pyscript2d << "pairs=[]" << endl;
			for (i=0; i < nparams_eff_2d; i++) {
				for (j=i+1; j < nparams_eff_2d; j++) {
					if ((hist2d_active_params[i]) and (hist2d_active_params[j])) {
						pyscript2d << "pairs.append(['" << param_names[i] << "','" << param_names[j] << "'])\n";
					}
				}
			}
			pyscript2d << "g.plots_2d(roots,param_pairs=pairs,";
			if (include_shading) pyscript2d << "shaded=True";
			else pyscript2d << "shaded=False";
			pyscript2d << ")" << endl;
			if (add_title) pyscript2d << "g.add_title(r'" << plot_title << "')" << endl;
			pyscript2d << "g.export(os.path.join(outdir,'" << fit_output_filename << "_2D.pdf'))" << endl;
			/*
			if (run_python_script) {
				string pycommand = "python " + pyname;
				if (system(pycommand.c_str()) == 0) {
					cout << "Plot for 2D posteriors saved to '" << fit_output_filename << "_2D.pdf'\n";
					//string rmcommand = "rm " + pyname;
					//system_returnval = system(rmcommand.c_str());
				}
				else cout << "Error: Could not generate PDF file for 2D posteriors\n";
			} else {
				cout << "Plotting script for 2D posteriors saved to '" << pyname << "'\n";
			}
			*/


			if (make_1d_posts) {
				// make script for triangle plot
				int n_triplots = 1;
				if (make_subplot) n_triplots++;
				for (int k=0; k < n_triplots; k++) {
					if (k==0) pyname = fit_output_filename + "_tri.py";
					else pyname = fit_output_filename + "_subtri.py";
					ofstream pyscript(pyname.c_str());
					pyscript << "import GetDistPlots, os" << endl;
					pyscript << "g=GetDistPlots.GetDistPlotter('" << fit_output_dir << "/')" << endl;
					pyscript << "g.settings.setSubplotSize(3.0000,width_scale=1.0)  # width_scale scales the width of all lines in the plot" << endl;
					pyscript << "outdir=''" << endl;
					pyscript << "roots=['" << fit_output_filename << "']" << endl;
					if (show_markers) {
						pyscript << "marker_list=[";
						for (i=0; i < n_markers; i++) {
							if ((hist2d_active_params[i]) and ((k==0) or (subplot_active_params[i]))) {
								pyscript << markers[i];
								if ((k==0) and (i != n_markers-1)) pyscript << ",";
								else if (k==1) {
									bool last_param = true;
									for (int ii=i+1; ii < n_markers; ii++) {
										if (subplot_active_params[ii]==true) last_param = false;
									}
									if (!last_param) pyscript << ",";
								}
							}
						}
						pyscript << "]" << endl;
					} else {
						pyscript << "marker_list=[]   # put parameter values in this list if you want to mark the 'true' or best-fit values on posteriors" << endl;
					}
					pyscript << "g.triangle_plot(roots, [";
					for (i=0; i < nparams_eff_2d; i++) {
						if ((hist2d_active_params[i]) and ((k==0) or (subplot_active_params[i]))) {
							pyscript << "'" << param_names[i] << "'";
							if ((k==0) and (i != nparams_eff_2d-1)) pyscript << ",";
							else if (k==1) {
								bool last_param = true;
								for (int ii=i+1; ii < nparams_eff_2d; ii++) {
									if (subplot_active_params[ii]==true) last_param = false;
								}
								if (!last_param) pyscript << ",";
							}
						}
					}
					pyscript << "],markers=marker_list,marker_color='orange',show_marker_2d=";
					if (show_markers) pyscript << "True";
					else pyscript << "False";
					pyscript << ",marker_2d='x',";
					if (include_shading) pyscript << "shaded=True";
					else pyscript << "shaded=False";
					pyscript << ")" << endl;
					if (add_title) pyscript << "g.add_title(r'" << plot_title << "')" << endl;
					pyscript << "g.export(os.path.join(outdir,'" << fit_output_filename;
					if (k==0) pyscript << "_tri.pdf'))" << endl;
					else pyscript << "_subtri.pdf'))" << endl;
					if (run_python_script) {
						string pycommand = "python3 " + pyname;
						if (system(pycommand.c_str()) == 0) {
							if (k==0) cout << "Triangle plot (1D+2D posteriors) saved to '" << fit_output_filename << "_tri.pdf'\n";
							else cout << "Triangle subplot saved to '" << fit_output_filename << "_subtri.pdf'\n";
							//string rmcommand = "rm " + pyname;
							//system_returnval = system(rmcommand.c_str());
						}
						else cout << "Error: Could not generate PDF file for triangle plot (1d + 2d posteriors)\n";
					} else {
						cout << "Plotting script for triangle plot saved to '" << pyname << "'\n";
					}
				}
			}
		}
	}
	delete[] prior_minvals;
	delete[] prior_maxvals;
	delete[] param_names;
	delete[] latex_param_names;
	delete[] markers;
	delete[] subplot_active_params;
	delete[] hist2d_active_params;
}

bool QLens::output_egrad_values_and_knots(const int srcnum, const string suffix)
{
	if (n_sb <= srcnum) return false;
	string scriptfile = fit_output_dir + "/egrad_values_knots";
	if (suffix != "") scriptfile += "_" + suffix;
	scriptfile += ".in";
	ofstream scriptout(scriptfile.c_str());
	sb_list[srcnum]->output_egrad_values_and_knots(scriptout);
	if (mpi_id==0) cout << "egrad values and knots output to '" << scriptfile << "'" << endl;
	return true;
}

bool QLens::output_scaled_percentiles_from_egrad_fits(const int srcnum, const double xcavg, const double ycavg, const double qtheta_pct_scaling, const double fmode_pct_scaling, const bool include_m3_fmode, const bool include_m4_fmode)
{
	if (n_sb <= srcnum) return false;
	string scriptfile = fit_output_dir + "/isofit_knots_limits.in";
	ofstream scriptout(scriptfile.c_str());
	sb_list[srcnum]->output_egrad_values_and_knots(scriptout);
	
	int i,j,k,nparams;
	int n_profile_params = 3;
	if (include_m3_fmode) n_profile_params += 2;
	if (include_m4_fmode) n_profile_params += 2;
	string label[n_profile_params];
	label[0] = "sbprofile"; // SB profile 
	label[1] = "egrad_profile0"; // axis ratio q
	label[2] = "egrad_profile1"; // angle theta
	i = 3;
	if (include_m3_fmode) {
		label[i++] = "egrad_profile4"; // A3 fourier mode
		label[i++] = "egrad_profile5"; // B3 fourier mode
	}
	if (include_m4_fmode) {
		label[i++] = "egrad_profile6"; // A4 fourier mode
		label[i++] = "egrad_profile7"; // B4 fourier mode
	}

	for (k=0; k < n_profile_params; k++) {
		string pnumfile_str = fit_output_dir + "/" + label[k] + ".nparam";
		ifstream pnumfile(pnumfile_str.c_str());
		if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
		pnumfile >> nparams;
		pnumfile.close();

		static const int n_characters = 5000;
		char dataline[n_characters];
		double *params = new double[nparams];
		double *priorlo = new double[nparams];
		double *priorhi = new double[nparams];

		string prangefile_str = fit_output_dir + "/" + label[k] + ".ranges";
		ifstream prangefile(prangefile_str.c_str());
		if (!prangefile.is_open()) { warn("could not open file '%s'",prangefile_str.c_str()); return false; }

		for (i=0; i < nparams; i++) {
			prangefile >> priorlo[i];
			prangefile >> priorhi[i];
		}

		string chain_str = fit_output_dir + "/" + label[k];
		ifstream chain_file(chain_str.c_str());

		unsigned long n_points=0;
		while (!chain_file.eof()) {
			chain_file.getline(dataline,n_characters);
			if (dataline[0]=='#') continue;
			n_points++;
		}
		chain_file.close();

		double **weights = new double*[nparams];
		double **paramvals = new double*[nparams];
		for (i=0; i < nparams; i++) {
			paramvals[i] = new double[n_points];
			weights[i] = new double[n_points]; // each parameter has a copy of all the weights since they'll be sorted differently for each parameter to get percentiles
		}

		chain_file.open(chain_str.c_str());
		j=0;
		double weight, tot=0;
		while (!chain_file.eof()) {
			chain_file.getline(dataline,n_characters);
			if (dataline[0]=='#') continue;

			istringstream datastream(dataline);
			datastream >> weight;
			tot += weight;
			for (i=0; i < nparams; i++) {
				datastream >> paramvals[i][j];
				weights[i][j] = weight;
			}
			j++;
		}
		chain_file.close();

		if (k==0) scriptout << "# SB-profile param limits" << endl;
		else if (k==1) scriptout << "# q-profile param limits" << endl;
		else if (k==2) scriptout << "# theta-profile param limits" << endl;
		else if (label[k]=="egrad_profile4") scriptout << "# A3-profile param limits" << endl;
		else if (label[k]=="egrad_profile5") scriptout << "# B3-profile param limits" << endl;
		else if (label[k]=="egrad_profile6") scriptout << "# A4-profile param limits" << endl;
		else if (label[k]=="egrad_profile7") scriptout << "# B4-profile param limits" << endl;

		double lopct, hipct, medpct, lowerr, hierr, scalefac, scaled_lopct, scaled_hipct;
		if (k==0) scalefac = dmin(5,qtheta_pct_scaling); // SB parameter errors are more trustworthy so they don't need to be scaled as much
		else if ((k==1) or (k==2)) scalefac = qtheta_pct_scaling; // SB parameter errors are more trustworthy so they don't need to be scaled as much
		else scalefac = fmode_pct_scaling;
		for (i=0; i < nparams; i++) {
			sort(n_points,paramvals[i],weights[i]);
			lopct = find_percentile(n_points, 0.02275, tot, paramvals[i], weights[i]);
			hipct = find_percentile(n_points, 0.97725, tot, paramvals[i], weights[i]);
			medpct = find_percentile(n_points, 0.5, tot, paramvals[i], weights[i]);
			lowerr = scalefac*(medpct - lopct);
			hierr = scalefac*(hipct - medpct);
			scaled_lopct = medpct - lowerr;
			scaled_hipct = medpct + hierr;
			if ((k==0) and (i==0) and (scaled_lopct < 0)) scaled_lopct = 0.01; // SB normalization is not allowed to be negative
			if ((k==1) and (scaled_hipct > 1)) scaled_hipct = 1; // q cannot exceed 1
			if ((k==1) and (scaled_lopct < 0.05)) scaled_lopct = 0.05; // q shouldn't get too close to zero
			if (k != 2) {
				// Don't allow scaled posterior ranges to go outside prior ranges (unless these are angle params, which are trickier to deal with)
				if (scaled_lopct < priorlo[i]) scaled_lopct = priorlo[i];
				if (scaled_hipct > priorhi[i]) scaled_hipct = priorhi[i];
			}

			//cout << "Param " << i << ": " << lopct[i] << " " << hipct[i] << endl;
			scriptout << scaled_lopct << " " << scaled_hipct << endl;
		}
		scriptout << endl;
		if ((k==2) and (n_data_bands > 0) and (imgdata_list[0])) {
			// Now output ranges in (xc,yc)
			double pixsize = imgdata_list[0]->pixel_size;
			double xlo, xhi, ylo, yhi;
			xlo = xcavg - pixsize/2;
			xhi = xcavg + pixsize/2;
			ylo = ycavg - pixsize/2;
			yhi = ycavg + pixsize/2;
			scriptout << "# xc, yc limits" << endl;
			scriptout << xlo << " " << xhi << endl;
			scriptout << ylo << " " << yhi << endl;
			scriptout << endl;
		}
		for (i=0; i < nparams; i++) {
			delete[] paramvals[i];
			delete[] weights[i];
		}
		delete[] paramvals;
		delete[] weights;
		delete[] priorlo;
		delete[] priorhi;
	}
	scriptout << "source update 0 xc=" << xcavg << " yc=" << ycavg << endl << endl;

	return true;
}

bool QLens::find_scaled_percentiles_from_chain(const double pct_scaling, double *scaled_lopct, double *scaled_hipct)
{
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	int i,j,nparams;
	pnumfile >> nparams;
	pnumfile.close();

	static const int n_characters = 5000;
	char dataline[n_characters];
	double *params = new double[nparams];
	string *paramnames = new string[nparams];
	string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
	ifstream pnamefile(pnamefile_str.c_str());
	for (i=0; i < nparams; i++) {
		pnamefile >> paramnames[i];
	}

	string chain_str = fit_output_dir + "/" + fit_output_filename;
	ifstream chain_file(chain_str.c_str());

	unsigned long n_points=0;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		if (dataline[0]=='#') continue;
		n_points++;
	}
	chain_file.close();

	double **weights = new double*[nparams];
	double **paramvals = new double*[nparams];
	for (i=0; i < nparams; i++) {
		paramvals[i] = new double[n_points];
		weights[i] = new double[n_points]; // each parameter has a copy of all the weights since they'll be sorted differently for each parameter to get percentiles
	}

	chain_file.open(chain_str.c_str());
	j=0;
	double weight, tot=0;
	while (!chain_file.eof()) {
		chain_file.getline(dataline,n_characters);
		if (dataline[0]=='#') continue;

		istringstream datastream(dataline);
		datastream >> weight;
		tot += weight;
		for (i=0; i < nparams; i++) {
			datastream >> paramvals[i][j];
			weights[i][j] = weight;
		}
		j++;
	}
	chain_file.close();

	double lopct, hipct, medpct, lowerr, hierr;
	for (i=0; i < nparams; i++) {
		sort(n_points,paramvals[i],weights[i]);
		lopct = find_percentile(n_points, 0.02275, tot, paramvals[i], weights[i]);
		hipct = find_percentile(n_points, 0.97725, tot, paramvals[i], weights[i]);
		medpct = find_percentile(n_points, 0.5, tot, paramvals[i], weights[i]);
		lowerr = pct_scaling*(medpct - lopct);
		hierr = pct_scaling*(hipct - medpct);
		scaled_lopct[i] = medpct - lowerr;
		scaled_hipct[i] = medpct + hierr;
	}
	for (i=0; i < nparams; i++) {
		delete[] paramvals[i];
		delete[] weights[i];
	}
	delete[] paramvals;
	delete[] weights;

	return true;
}

bool QLens::get_stepsizes_from_percentiles(const double pct_scaling, dvector& stepsizes)
{
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	int i,j,nparams;
	pnumfile >> nparams;
	pnumfile.close();
	if (nparams != param_list->nparams) { warn("nparams from chains directory does not match n_fit_parameters from current parameter list"); return false; }

	double *scaled_lopct = new double[nparams];
	double *scaled_hipct = new double[nparams];

	bool status = find_scaled_percentiles_from_chain(pct_scaling, scaled_lopct, scaled_hipct);

	if (status==true) {
		for (i=0; i < nparams; i++) {
			stepsizes[i] = (scaled_hipct[i]-scaled_lopct[i])/2;
		}
	}
	delete[] scaled_lopct;
	delete[] scaled_hipct;

	return status;
}

bool QLens::output_scaled_percentiles_from_chain(const double pct_scaling)
{
	string pnumfile_str = fit_output_dir + "/" + fit_output_filename + ".nparam";
	ifstream pnumfile(pnumfile_str.c_str());
	if (!pnumfile.is_open()) { warn("could not open file '%s'",pnumfile_str.c_str()); return false; }
	int i,j,nparams;
	pnumfile >> nparams;
	pnumfile.close();

	double *scaled_lopct = new double[nparams];
	double *scaled_hipct = new double[nparams];

	bool status = find_scaled_percentiles_from_chain(pct_scaling, scaled_lopct, scaled_hipct);

	if (status==true) {
		string scriptfile = fit_output_dir + "/scaled_limits.in";
		ofstream scriptout(scriptfile.c_str());
		scriptout << "fit priors limits" << endl;
		for (i=0; i < nparams; i++) {
			scriptout << scaled_lopct[i] << " " << scaled_hipct[i] << " # " << paramnames[i] << endl;
			cout << scaled_lopct[i] << " " << scaled_hipct[i] << " # " << paramnames[i] << endl;
		}
		scriptout << endl;
		cout << "Limits have been output to file '" << scriptfile << "'" << endl;
		cout << endl;
	}
	delete[] scaled_lopct;
	delete[] scaled_hipct;

	return status;
}

bool QLens::output_coolest_files(const string filename)
{
#ifdef USE_COOLEST
	std::ifstream fin;
	Json::Value coolest;
	fin.open("coolest_fixed_input.json",std::ifstream::in);
	if (!fin.is_open()) {
		warn("could not find file 'coolest_fixed_input.json'; cannot write .json file for coolest");
		return false;
	}
	fin >> coolest;
	fin.close();

	if ((n_data_bands == 0) or (imgdata_list[0]==NULL)) {
		warn("no image data has been loaded; cannot write .json file for coolest");
		return false;
	}
	double pixsize = imgdata_list[0]->pixel_size;
	coolest["instrument"]["pixel_size"] = pixsize;
	//cout << "pixel size: " << coolest["instrument"]["pixel_size"].asDouble() << endl;
	//cout << "standard: " << coolest["standard"].asString() << endl;
	//cout << "H0: " << coolest["cosmology"]["H0"].asDouble() << endl;

	Json::Value pixels_psf;
	pixels_psf["field_of_view_x"] = Json::Value(Json::arrayValue);
	pixels_psf["field_of_view_x"].append(0);
	pixels_psf["field_of_view_y"] = Json::Value(Json::arrayValue);
	pixels_psf["field_of_view_y"].append(0);
	if (psf_supersampling) {
		pixels_psf["field_of_view_x"].append(pixsize*psf_list[0]->supersampled_psf_npixels_x/default_imgpixel_nsplit);
		pixels_psf["field_of_view_y"].append(pixsize*psf_list[0]->supersampled_psf_npixels_y/default_imgpixel_nsplit);
	} else {
		pixels_psf["field_of_view_x"].append(pixsize*psf_list[0]->psf_npixels_x);
		pixels_psf["field_of_view_y"].append(pixsize*psf_list[0]->psf_npixels_y);
	}
	pixels_psf["num_pix_x"] = (psf_supersampling ? psf_list[0]->supersampled_psf_npixels_x : psf_list[0]->psf_npixels_x);
	pixels_psf["num_pix_y"] = (psf_supersampling ? psf_list[0]->supersampled_psf_npixels_y : psf_list[0]->psf_npixels_y);
	pixels_psf["fits_file"] = Json::Value();
	pixels_psf["fits_file"]["path"] = psf_list[0]->psf_filename;
	coolest["instrument"]["psf"]["pixels"] = pixels_psf;

	Json::Value pixels_obs;
	double grid_xmin = grid_xcenter - grid_xlength/2;
	double grid_xmax = grid_xcenter + grid_xlength/2;
	double grid_ymin = grid_ycenter - grid_ylength/2;
	double grid_ymax = grid_ycenter + grid_ylength/2;

	pixels_obs["field_of_view_x"] = Json::Value(Json::arrayValue);
	pixels_obs["field_of_view_x"].append(grid_xmin);
	pixels_obs["field_of_view_x"].append(grid_xmax);
	pixels_obs["field_of_view_y"] = Json::Value(Json::arrayValue);
	pixels_obs["field_of_view_y"].append(grid_ymin);
	pixels_obs["field_of_view_y"].append(grid_ymax);
	pixels_obs["num_pix_x"] = n_image_pixels_x;
	pixels_obs["num_pix_y"] = n_image_pixels_y;
	pixels_obs["fits_file"] = Json::Value();
	pixels_obs["fits_file"]["path"] = imgdata_list[0]->data_fits_filename;
	coolest["observation"]["pixels"] = pixels_obs;

	Json::Value noise;
	if (use_noise_map) {
		Json::Value pixels_noise;
		pixels_noise["field_of_view_x"] = Json::Value(Json::arrayValue);
		pixels_noise["field_of_view_x"].append(grid_xmin);
		pixels_noise["field_of_view_x"].append(grid_xmax);
		pixels_noise["field_of_view_y"] = Json::Value(Json::arrayValue);
		pixels_noise["field_of_view_y"].append(grid_ymin);
		pixels_noise["field_of_view_y"].append(grid_ymax);
		pixels_noise["num_pix_x"] = n_image_pixels_x;
		pixels_noise["num_pix_y"] = n_image_pixels_y;
		pixels_noise["fits_file"] = Json::Value();
		pixels_noise["fits_file"]["path"] = imgdata_list[0]->noise_map_fits_filename;
		noise["type"] = "NoiseMap";
		noise["noise_map"] = pixels_noise;
	} else {
		noise["type"] = "UniformGaussianNoise";
		noise["std_dev"] = background_pixel_noise;
	}
	coolest["observation"]["noise"] = noise;


	//lens["type"] = "Galaxy";
	//lens["name"] = ;
	//lens["redshift"] = lens_redshift;
	//lens["mass_model"] = Json::Value(Json::arrayValue);
	//lens["light_model"] = Json::Value(Json::arrayValue);

	Json::Value posterior_stats;
	posterior_stats["mean"] = Json::Value::null;
	posterior_stats["median"] = Json::Value::null;
	posterior_stats["percentile_16th"] = Json::Value::null;
	posterior_stats["percentile_84th"] = Json::Value::null;

	Json::Value prior;
	prior["type"] = Json::Value::null;

	Json::Value lensing_entities = Json::Value(Json::arrayValue);
	LensProfile* lensptr;
	int i,j;
	double param_val;
	string typestring;
	map<string,string> names_lookup;
	for (i=nlens-1; i >= 0; i--) { // adding lenses in reverse because the other lens modelers put the shear model before PEMD, so just to make it look the same
		Json::Value mass_model;
		lensptr = lens_list[i];
		typestring = "Galaxy";
		string name = lensptr->model_name;
		if (name=="sple") name = "SPEMD";
		else if (name=="shear") {
			name = "ExternalShear";
			typestring = "MassField";
		}
		Json::Value lens;
		lens["type"] = typestring;
		lens["redshift"] = lensptr->zlens;
		lens["mass_model"] = Json::Value(Json::arrayValue);
		lens["light_model"] = Json::Value(Json::arrayValue);

		if (lensptr->model_name=="sple")
		{
			names_lookup = {{"xc","center_x"},{"yc","center_y"},{"alpha","gamma"},{"gamma","gamma"},{"theta","phi"},{"q","q"},{"b","theta_E"},{"s","s"}};
			Json::Value param;
			param["posterior_stats"] = posterior_stats;
			param["prior"] = prior;
			for (j=0; j < lensptr->n_params-1; j++) {
				Json::Value point_estimate;
				param_val = lensptr->get_parameter(j);
				if (lensptr->paramnames[j]=="alpha") param_val += 1; // from 2D power index to 3D power index	
				if (lensptr->paramnames[j]=="theta") {
					while (param_val > 90) param_val -= 180;
					while (param_val < -90) param_val += 180;
				}
				point_estimate["value"] = param_val;
				if ((lensptr->paramnames[j]=="s") and (param_val==0)) {
					name = "PEMD";
					// skip 's' if it is zero, since we will call it a PEMD instead of SPEMD
				} else {
					param["point_estimate"] = point_estimate;
					mass_model["parameters"][names_lookup[lensptr->paramnames[j]]] = param;
				}
			}
			mass_model["type"] = name;
			lens["mass_model"].append(mass_model);

			//cout << "Lens number " << i << " is a SPLE!" << endl;
		}
		else if (lens_list[i]->model_name=="shear")
		{
			names_lookup = {{"xc","center_x"},{"yc","center_y"},{"shear","gamma_ext"},{"theta_shear","phi_ext"},{"theta_pert","phi_ext"}};
			Json::Value param;
			param["posterior_stats"] = posterior_stats;
			param["prior"] = prior;
			mass_model["type"] = typestring;
			mass_model["parameters"] = Json::Value();
			for (j=0; j < 2; j++) {
				Json::Value point_estimate;
				param_val = lensptr->get_parameter(j);
				if (j==1) {
					// shear angle
					if (lensptr->paramnames[j]=="theta_pert") param_val += 90; // from 2D power index to 3D power index	
					while (param_val > 90) param_val -= 180;
					while (param_val < -90) param_val += 180;
				}
				point_estimate["value"] = param_val;
				param["point_estimate"] = point_estimate;
				mass_model["parameters"][names_lookup[lensptr->paramnames[j]]] = param;
			}
			mass_model["type"] = name;
			lens["mass_model"].append(mass_model);

			//cout << "Lens number " << i << " is an external shear!" << endl;
		}
		else
		{
			die("mass model type for lens %i not supported in COOLEST yet",i);
		}
		lens["name"] = name;
		lensing_entities.append(lens);
	}
	coolest["lensing_entities"] = lensing_entities;

	if (source_fit_mode==Delaunay_Source) {
		if ((delaunay_srcgrids) and (delaunay_srcgrids[0])) {
			vector<double> xvals;
			vector<double> yvals;
			vector<double> sbvals;
			delaunay_srcgrids[0]->get_grid_points(xvals,yvals,sbvals);
			int n_srcpts = sbvals.size();

			string src_filename = filename + "_src.fits";
			std::unique_ptr<CCfits::FITS> pFits(nullptr);
			pFits.reset( new CCfits::FITS("!"+src_filename,CCfits::Write) );

			std::string newName("NEW-EXTENSION");
			std::vector<std::string> ColFormats = {"E","E","E"};
			std::vector<std::string> ColNames = {"x","y","z"};
			std::vector<std::string> ColUnits = {"dum","dum","dum"};
			CCfits::Table* newTable = pFits->addTable(newName,n_srcpts,ColNames,ColFormats,ColUnits);
			newTable->column("x").write(xvals,1);  
			newTable->column("y").write(yvals,1);
			newTable->column("z").write(sbvals,1);

			// Then create the remaining json fields
			Json::Value source;
			source["type"] = "Galaxy";
			source["name"] = "qlens Delaunay source";
			source["redshift"] = source_redshift;
			source["mass_model"] = Json::Value(Json::arrayValue);
			source["light_model"] = Json::Value(Json::arrayValue);

			Json::Value light_model;
			Json::Value pixels_irr;
			pixels_irr["field_of_view_x"] = Json::Value(Json::arrayValue);
			pixels_irr["field_of_view_x"].append(0);
			pixels_irr["field_of_view_x"].append(0);
			pixels_irr["field_of_view_y"] = Json::Value(Json::arrayValue);
			pixels_irr["field_of_view_y"].append(0);
			pixels_irr["field_of_view_y"].append(0);
			pixels_irr["num_pix"] = n_srcpts;
			pixels_irr["fits_file"] = Json::Value();
			pixels_irr["fits_file"]["path"] = src_filename;  
			light_model["parameters"] = Json::Value();
			light_model["parameters"]["pixels"] = pixels_irr;
			light_model["type"] = "IrregularGrid";
			source["light_model"].append( light_model );

			lensing_entities.append( source );

			coolest["lensing_entities"] = lensing_entities;

		} else {
			warn("Delaunay source grid has not been constructed, so it cannot be output in FITS table");
		}
	} else if (source_fit_mode==Shapelet_Source) {
		// Implement this when you get time
	} else if (source_fit_mode==Parameterized_Source) {
		// Implement this when you get time
	}

	std::ofstream jsonfile(filename + ".json");
	jsonfile << coolest;
	jsonfile.close();
	return true;
#else
	warn("QLens must be compiled with jsoncpp and ccfits (and -DUSE_COOLEST flag) to output coolest files");
	return false;
#endif
}

//bool QLens::output_coolest_chain_file(const string filename)
//{
//}

double QLens::get_einstein_radius_prior(const bool verbal)
{
	double re, loglike_penalty = 0;
	einstein_radius_of_primary_lens(reference_zfactors[lens_redshift_idx[primary_lens_number]],re);
	//loglike_penalty = SQR((re-einstein_radius_threshold)/0.1);
	if (re < einstein_radius_low_threshold) {
		loglike_penalty = pow(1-re+einstein_radius_low_threshold,40) - 1.0; // constructed so that penalty = 0 if the average n_image = n_image_threshold
		if ((mpi_id==0) and (verbal)) cout << "*NOTE: Einstein radius is below the low prior threshold (" << re << " vs. " << einstein_radius_low_threshold << "), resulting in penalty prior (loglike_penalty=" << loglike_penalty << ")" << endl;
	}
	else if (re > einstein_radius_high_threshold) {
		loglike_penalty = pow(1+re-einstein_radius_high_threshold,40) - 1.0; // constructed so that penalty = 0 if the average n_image = n_image_threshold
		if ((mpi_id==0) and (verbal)) cout << "*NOTE: Einstein radius is above the high prior threshold (" << re << " vs. " << einstein_radius_high_threshold << "), resulting in penalty prior (loglike_penalty=" << loglike_penalty << ")" << endl;
	}
	return loglike_penalty;
}

double QLens::fitmodel_loglike_point_source(double* params)
{
	bool showed_first_chisq = false; // used just to know whether to print a comma before showing the next chisq component
	double loglike=0, chisq_total=0, chisq;
	double log_penalty_prior;
	int n_fitparams = param_list->nparams;
	double transformed_params[n_fitparams];
	if (params != NULL) {
		fitmodel->param_list->inverse_transform_parameters(params,transformed_params);
		//fitmodel->param_list->print_penalty_limits();
		bool penalty_incurred = false;
		for (int i=0; i < n_fitparams; i++) {
			if (fitmodel->param_list->defined_prior_limits[i]==true) {
				//cout << "USE_LIMITS " << i << endl;
				if ((transformed_params[i] < fitmodel->param_list->untransformed_prior_limits_lo[i]) or (transformed_params[i] > fitmodel->param_list->untransformed_prior_limits_hi[i])) penalty_incurred = true;
			}
		}
		//fitmodel->param_list->print_penalty_limits();
		if (penalty_incurred) return 1e30;
		log_penalty_prior = fitmodel->update_model(transformed_params);
		if (log_penalty_prior >= 1e30) return log_penalty_prior; // don't bother to evaluate chi-square if there is huge prior penalty; wastes time
		else if (log_penalty_prior > 0) loglike += log_penalty_prior;

		if (group_id==0) {
			if (fitmodel->logfile.is_open()) {
				for (int i=0; i < n_fitparams; i++) fitmodel->logfile << params[i] << " ";
			}
			fitmodel->logfile << flush;
		}
	}

	if (include_imgpos_chisq) {
		bool used_imgplane_chisq; // keeps track of whether image plane chi-square gets used, since there is an option to switch from srcplane to imgplane below a given threshold
		double rms_err;
		int n_matched_imgs;
		if (imgplane_chisq) {
			used_imgplane_chisq = true;
			double* remember_grid_zfac = Grid::grid_zfactors;
			double** remember_grid_betafac = Grid::grid_betafactors;
			if (chisq_diagnostic) chisq = fitmodel->chisq_pos_image_plane_diagnostic(true,false,rms_err,n_matched_imgs);
			else chisq = fitmodel->chisq_pos_image_plane();
			// THE FOLLOWING IS A HORRIBLE HACK because grid_zfactors, grid_betafactors are static. TO FIX THIS, have a parent Grid that contains these (and other) variables which are no longer static, with children objects called GridCell or something. This is has already been done for CartesianSourceGrid (versus CartesianSourcePixel), so just copy what you did there. DO THIS BEFORE RELEASING PUBLICLY!!!!
			Grid::grid_zfactors = remember_grid_zfac;
			Grid::grid_betafactors = remember_grid_betafac;
		}
		else {
			used_imgplane_chisq = false;
			chisq = fitmodel->chisq_pos_source_plane();
			if (chisq < chisq_imgplane_substitute_threshold) {
				double* remember_grid_zfac = Grid::grid_zfactors;
				double** remember_grid_betafac = Grid::grid_betafactors;
				if (chisq_diagnostic) chisq = fitmodel->chisq_pos_image_plane_diagnostic(true,false,rms_err,n_matched_imgs);
				else chisq = fitmodel->chisq_pos_image_plane();
			// THE FOLLOWING IS A HORRIBLE HACK because grid_zfactors, grid_betafactors are static. TO FIX THIS, have a parent Grid that contains these (and other) variables which are no longer static, with children objects called GridCell or something. This is has already been done for CartesianSourceGrid (versus CartesianSourcePixel), so just copy what you did there. DO THIS BEFORE RELEASING PUBLICLY!!!!
				Grid::grid_zfactors = remember_grid_zfac;
				Grid::grid_betafactors = remember_grid_betafac;
				used_imgplane_chisq = true;
			}
		}
		if ((display_chisq_status) and (mpi_id==0)) {
			if (use_ansi_characters) cout << "\033[2A" << flush;
			if (include_imgpos_chisq) {
				if (used_imgplane_chisq) {
					if (!imgplane_chisq) cout << "imgplane_chisq: "; // so user knows the imgplane chi-square is being used (we're below the threshold to switch from srcplane to imgplane)
					int tot_data_images = 0;
					for (int i=0; i < n_ptsrc; i++) tot_data_images += point_image_data[i].n_images;
					if (use_ansi_characters) cout << "# images: " << fitmodel->n_visible_images << " vs. " << tot_data_images << " data";
					if (fitmodel->chisq_it % chisq_display_frequency == 0) {
						if (!use_ansi_characters) cout << "# images: " << fitmodel->n_visible_images << " vs. " << tot_data_images << " data";
						cout << ", chisq_pos=" << chisq;
						if (syserr_pos != 0.0) {
							double signormfac, chisq_sys = chisq;
							int i,k;
							for (i=0; i < n_ptsrc; i++) {
								for (k=0; k < point_image_data[i].n_images; k++) {
									signormfac = 2*log(1.0 + SQR(fitmodel->syserr_pos/point_image_data[i].sigma_pos[k]));
									chisq_sys -= signormfac;
								}
							}
							cout << ", chisq_pos_sys=" << chisq_sys;
						}
						showed_first_chisq = true;
					}
				} else {
					if (fitmodel->chisq_it % chisq_display_frequency == 0) {
						cout << "chisq_pos=" << chisq;
						// redundant and ugly! make it prettier later
						if (syserr_pos != 0.0) {
							double signormfac, chisq_sys = chisq;
							int i,k;
							for (i=0; i < n_ptsrc; i++) {
								for (k=0; k < point_image_data[i].n_images; k++) {
									signormfac = 2*log(1.0 + SQR(fitmodel->syserr_pos/point_image_data[i].sigma_pos[k]));
									chisq_sys -= signormfac;
								}
							}
							cout << ", chisq_pos_sys=" << chisq_sys;
						}
						showed_first_chisq = true;
					}
				}
			}
		}
	} else {
		chisq=0;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (use_ansi_characters) cout << "\033[2A" << flush;
			if ((fitmodel->chisq_it % chisq_display_frequency == 0) and (include_imgpos_chisq)) {
				cout << "chisq_pos=0";
				showed_first_chisq = true;
			}
		}
	}
	chisq_total += chisq;
	if (include_flux_chisq) {
		chisq = fitmodel->chisq_flux();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (showed_first_chisq) cout << ", ";
			else showed_first_chisq = true;
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_flux=" << chisq;
		}
	}
	if (include_time_delay_chisq) {
		chisq = fitmodel->chisq_time_delays();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (showed_first_chisq) cout << ", ";
			else showed_first_chisq = true;
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_td=" << chisq;
		}
	}
	if (include_weak_lensing_chisq) {
		chisq = fitmodel->chisq_weak_lensing();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (showed_first_chisq) cout << ", ";
			else showed_first_chisq = true;
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_weak_lensing=" << chisq;
		}
	}
	raw_chisq = chisq_total; // in case the chi-square is being used as a derived parameter
	fitmodel->raw_chisq = chisq_total;
	loglike += chisq_total/2;
	if (chisq*0.0 != 0.0) {
		warn("chi-square is returning NaN (%g)",chisq);
	}
 
 	if (params != NULL) {
		fitmodel->param_list->add_prior_terms_to_loglike(params,loglike);
		fitmodel->param_list->add_jacobian_terms_to_loglike(transformed_params,loglike);
		if (use_custom_prior) loglike += fitmodel_custom_prior();
	}
	if ((einstein_radius_prior) and (nlens > 0)) loglike += fitmodel->get_einstein_radius_prior(false);
	if ((display_chisq_status) and (mpi_id==0)) {
		if (fitmodel->chisq_it % chisq_display_frequency == 0) {
			if (chisq_total != (2*loglike)) cout << ", chisq_tot=" << chisq_total;
			cout << ", -2*loglike=" << 2*loglike;
			cout << "                ";
			if (!use_ansi_characters) cout << endl;
		}
		if (use_ansi_characters) cout << endl << endl;
	}

	fitmodel->chisq_it++;
	return loglike;
}

double QLens::fitmodel_loglike_extended_source(double* params)
{
#ifdef USE_OPENMP
	double update_wtime0, update_wtime;
	if (show_wtime) {
		update_wtime0 = omp_get_wtime();
	}
#endif
	double transformed_params[param_list->nparams];
	double loglike=0, chisq=0, chisq0, chisq_td;
	double log_penalty_prior;
	int n_fitparams = param_list->nparams;
	if (params != NULL) {
		fitmodel->param_list->inverse_transform_parameters(params,transformed_params);
		for (int i=0; i < n_fitparams; i++) {
			if (fitmodel->param_list->defined_prior_limits[i]==true) {
				//cout << "parameter " << i << ": plimits " << fitmodel->param_list->penalty_limits_lo[i] << fitmodel->param_list->penalty_limits_hi[i] << endl;
				if ((transformed_params[i] < fitmodel->param_list->untransformed_prior_limits_lo[i]) or (transformed_params[i] > fitmodel->param_list->untransformed_prior_limits_hi[i])) {
					//cout << "RUHROH parameter " << i << ": " << transformed_params[i] << endl;
					return 1e30;
				}
			}
			//else cout << "parameter " << i << ": no plimits " << endl;
		}
		//cout << "updaing model" << endl;
		log_penalty_prior = fitmodel->update_model(transformed_params);
		//cout << "done updating model" << endl;
		if (log_penalty_prior >= 1e30) return log_penalty_prior; // don't bother to evaluate chi-square if there is huge prior penalty; wastes time
		else if (log_penalty_prior > 0) loglike += log_penalty_prior;

		if (group_id==0) {
			if (fitmodel->logfile.is_open()) {
				for (int i=0; i < n_fitparams; i++) fitmodel->logfile << params[i] << " ";
				fitmodel->logfile << flush;
			}
		}
	}
#ifdef USE_OPENMP
	if (show_wtime) {
		update_wtime = omp_get_wtime() - update_wtime0;
		if (mpi_id==0) cout << "wall time for updating parameters: " << update_wtime << endl;
	}
#endif

	if (einstein_radius_prior) {
		loglike += fitmodel->get_einstein_radius_prior(false);
		//if (loglike > 1e10) loglike += 1e5; // in this case, intead of doing inversion we'll just add 1e5 as a stand-in for chi-square to save time
	}
	double chisq00;
	chisq=0,chisq0=0;
	if (loglike < 1e30) {
		for (int i=0; i < n_ranchisq; i++) {
			chisq += fitmodel->pixel_log_evidence_times_two(chisq00,false,i);
			chisq0 += chisq00;
		}
		chisq /= n_ranchisq;
		chisq0 /= n_ranchisq;
	}

	raw_chisq = chisq0; // in case the chi-square is being used as a derived parameter
	fitmodel->raw_chisq = chisq0;
	loglike += chisq/2;
	if (include_time_delay_chisq) {
		chisq_td = fitmodel->chisq_time_delays_from_model_imgs();
		loglike += chisq_td/2;
	}
	if (params != NULL) {
		fitmodel->param_list->add_prior_terms_to_loglike(params,loglike);
		fitmodel->param_list->add_jacobian_terms_to_loglike(transformed_params,loglike);
		if (concentration_prior) {
			for (int i=0; i < nlens; i++) {
				if ((lens_list[i]->lenstype==nfw) and (lens_list[i]->use_concentration_prior)) loglike += lens_list[i]->concentration_prior();
			}
		}
		if (use_custom_prior) loglike += fitmodel_custom_prior();
	}
	if ((display_chisq_status) and (mpi_id==0)) {
		if (use_ansi_characters) cout << "\033[2A" << flush;
		cout << "chisq0=" << chisq0;
		cout << ", chisq_pix=" << chisq;
		if (include_time_delay_chisq) cout << ", chisq_td=" << chisq_td;
		cout << ", -2*loglike=" << 2*loglike;
		cout << "                " << endl;

		//cout << "\033[1A";
		if (use_ansi_characters) cout << endl;
	}

	fitmodel->chisq_it++;
	return loglike;
}

double QLens::loglike_point_source(double* params)
{
	// can use this version for testing purposes in case there is any doubt about whether the fitmodel version is faithfully reproducing the original
	double transformed_params[param_list->nparams];
	param_list->inverse_transform_parameters(params,transformed_params);
	for (int i=0; i < param_list->nparams; i++) {
		if (param_list->defined_prior_limits[i]==true) {
			if ((transformed_params[i] < param_list->untransformed_prior_limits_lo[i]) or (transformed_params[i] > param_list->untransformed_prior_limits_hi[i])) return 1e30;
		}
	}
	//if (update_fitmodel(transformed_params)==false) return 1e30;
	if (fitmodel->update_model(transformed_params) != 0.0) return 1e30;
	if (group_id==0) {
		if (logfile.is_open()) {
			for (int i=0; i < param_list->nparams; i++) logfile << params[i] << " ";
		}
		logfile << flush;
	}

	double loglike, chisq_total=0, chisq;
	if (imgplane_chisq) {
		chisq = chisq_pos_image_plane();
		if ((display_chisq_status) and (mpi_id==0)) {
			int tot_data_images = 0;
			for (int i=0; i < n_ptsrc; i++) tot_data_images += point_image_data[i].n_images;
			cout << "# images: " << n_visible_images << " vs. " << tot_data_images << " data, ";
			if (chisq_it % chisq_display_frequency == 0) cout << "chisq_pos=" << chisq;
		}
	}
	else {
		chisq = chisq_pos_source_plane();
		if ((display_chisq_status) and (mpi_id==0)) {
			if (chisq_it % chisq_display_frequency == 0) cout << "chisq_pos=" << chisq;
		}
	}
	chisq_total += chisq;
	if (include_flux_chisq) {
		chisq = chisq_flux();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (chisq_it % chisq_display_frequency == 0) cout << ", chisq_flux=" << chisq;
		}
	}
	if (include_time_delay_chisq) {
		chisq = chisq_time_delays();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (chisq_it % chisq_display_frequency == 0) cout << ", chisq_td=" << chisq;
		}
	}
	if ((display_chisq_status) and (mpi_id==0)) {
		if (chisq_it % chisq_display_frequency == 0) cout << ", chisq_tot=" << chisq_total << "               ";
		cout << endl;
		//cout << "\033[1A";
	}

	loglike = chisq_total/2.0;

	param_list->add_prior_terms_to_loglike(params,loglike);
	param_list->add_jacobian_terms_to_loglike(transformed_params,loglike);
	if (use_custom_prior) loglike = fitmodel_custom_prior();
	chisq_it++;
	return loglike;
}

void QLens::fitmodel_calculate_derived_params(double* params, double* derived_params)
{
	if (dparam_list->n_dparams==0) return;
	fitmodel->param_list->update_untransformed_values(params);
	fitmodel->dparam_list->get_dparams(derived_params);
	//double transformed_params[param_list->nparams];
	//fitmodel->param_list->inverse_transform_parameters(params,transformed_params);
	//if (fitmodel->update_model(transformed_params) != 0.0) warn("derived params for point incurring penalty chi-square may give absurd results");
	//for (int i=0; i < dparam_list->n_dparams; i++) derived_params[i] = dparam_list->dparams[i]->get_derived_param(fitmodel);
	clear_raw_chisq();
}

double QLens::get_lens_parameter_using_pmode(const int paramnum, const int lensnum, const int pmode_in)
{
	int pmode = pmode_in;
	if (pmode==-1) pmode = default_parameter_mode;
	if (lensnum >= nlens) die("lensnum exceeds number of lenses");
	int lens_nparams = lens_list[lensnum]->get_n_params();
	if (paramnum >= lens_nparams) die("for lensparam, lens parameter number exceeds total number of parameters in lens");
	double lensparam;
	double *lensparams = new double[lens_nparams];
	lens_list[lensnum]->get_parameters_pmode(pmode,lensparams);
	lensparam = lensparams[paramnum];
	//cout << "Number of lens params: " << lens_nparams << endl;
	//for (int i=0; i < lens_nparams; i++) {
		//cout << "param " << i << ": " << lensparams[i] << endl;
	//}
	delete[] lensparams;
	return lensparam;
}

double QLens::fitmodel_custom_prior()
{
	//static const double rcore_threshold = 3.0;
	double cnfw_params[8];
	double rc, rs, rcore;
	if (fitmodel != NULL)
		fitmodel->lens_list[0]->get_parameters_pmode(0,cnfw_params);
	else
		lens_list[0]->get_parameters_pmode(0,cnfw_params); // used for the "test" command"
	rs = cnfw_params[1];
	rc = cnfw_params[2];
	//rcore = rc*(sqrt(1+8*rs/rc)-1)/4.0;
	//if (fitmodel==NULL) cout << "rcore: " << rcore << endl; // for testing purposes, using the "test" command
	//if (rcore < rcore_threshold) return 0.0;
	//else return 1e30+rcore; // penalty function
	if (rc < rs) return 0.0;
	else return 1e30+rc;
}

void QLens::set_Gauss_NN(const int& nn)
{
	LensProfile::Gauss_NN = nn;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->SetGaussLegendre(nn);
		}
	}
}

void QLens::set_integral_tolerance(const double& acc)
{
	LensProfile::integral_tolerance = acc;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->set_integral_tolerance(acc);
		}
	}
}

void QLens::set_integral_convergence_warnings(const bool warn)
{
	LensProfile::integration_warnings = warn;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->set_integral_warnings(); // this is for integrations used for derived parameters etc.
		}
	}
}

void QLens::reassign_lensparam_pointers_and_names(const bool reset_plimits)
{
	// parameter pointers should be reassigned if the parameterization mode has been changed (e.g., shear components turned on/off)
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->calculate_ellipticity_components(); // in case ellipticity components has been turned on
			lens_list[i]->assign_param_pointers();
			lens_list[i]->assign_paramnames();
			lens_list[i]->update_meta_parameters();
		}
		if (reset_plimits) set_default_plimits();
		update_parameter_list();
		if ((reset_plimits) and (mpi_id==0)) cout << "NOTE: plimits have been reset, since lens parameterization has been changed" << endl;
	}
}

void QLens::reassign_sb_param_pointers_and_names()
{
	// parameter pointers should be reassigned if the parameterization mode has been changed
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++) {
			sb_list[i]->calculate_ellipticity_components(); // in case ellipticity components has been turned on
			sb_list[i]->assign_param_pointers();
			sb_list[i]->assign_paramnames();
		}
		set_default_plimits();
		update_parameter_list();
		if (mpi_id==0) cout << "NOTE: plimits have been reset, since source parameterization has been changed" << endl;
	}
}

void QLens::print_lens_cosmology_info(const int lmin, const int lmax)
{
	if (lmax >= nlens) return;
	double sigma_cr = cosmo->sigma_crit_kpc(lens_redshift,reference_source_redshift);
	double dlens = cosmo->angular_diameter_distance(lens_redshift);
	cout << "H0 = " << cosmo->get_hubble()*100 << " km/s/Mpc" << endl;
	cout << "omega_m = " << cosmo->get_omega_m() << endl;
	//cout << "omega_lambda = " << 1-omega_matter << endl;
	cout << "zlens = " << lens_redshift << endl;
	cout << "zsrc = " << source_redshift << endl;
	cout << "D_lens: " << dlens << " Mpc  (angular diameter distance to lens plane)" << endl;
	double rhocrit = 1e-9*cosmo->critical_density(lens_redshift);
	cout << "rho_crit(zlens): " << rhocrit << " M_sol/kpc^3" << endl;
	cout << "Sigma_crit(zlens,zsrc_ref): " << sigma_cr << " M_sol/kpc^2" << endl;
	double kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(lens_redshift);
	cout << "1 arcsec = " << (1.0/kpc_to_arcsec) << " kpc" << endl;
	cout << "sigma8 = " << cosmo->rms_sigma8() << endl;
	cout << endl;
	if (nlens > 0) {
		for (int i=lmin; i <= lmax; i++) {
			lens_list[i]->output_cosmology_info(i);
		}
	}
	else cout << "No lens models have been specified" << endl << endl;
}

bool QLens::output_mass_r(const double r, const int lensnum, const bool use_kpc)
{
	if (lensnum >= nlens) return false;
	double zlens, sigma_cr, kpc_to_arcsec, r_arcsec, r_kpc, mass_r_2d, rho_r_3d, mass_r_3d;
	double zl = lens_list[lensnum]->zlens;
	sigma_cr = cosmo->sigma_crit_arcsec(zl,reference_source_redshift);
	kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(zl);
	if (!use_kpc) {
		r_kpc = r/kpc_to_arcsec;
		r_arcsec = r;
	} else {
		r_kpc = r;
		r_arcsec = r*kpc_to_arcsec;
	}
	cout << "Radius: " << r_kpc << " kpc (" << r_arcsec << " arcsec)\n";
	mass_r_2d = sigma_cr*lens_list[lensnum]->mass_rsq(r_arcsec*r_arcsec);
	cout << "Mass enclosed (2D): " << mass_r_2d << " M_sol" << endl;
	bool converged;
	rho_r_3d = (sigma_cr*CUBE(kpc_to_arcsec))*lens_list[lensnum]->calculate_scaled_density_3d(r_arcsec,1e-4,converged);
	cout << "Density (3D): " << rho_r_3d << " M_sol/kpc^3" << endl;
	mass_r_3d = sigma_cr*lens_list[lensnum]->calculate_scaled_mass_3d(r_arcsec);
	//double mass_r_3d_unscaled = mass_r_3d/sigma_cr;
	//double rho_r_3d_noscale = lens_list[lensnum]->calculate_scaled_density_3d(r_arcsec);
	cout << "Mass enclosed (3D): " << mass_r_3d << " M_sol" << endl;
	//cout << "Mass enclosed (3D) unscaled: " << mass_r_3d_unscaled << " M_sol" << endl;
	//cout << "Density unscaled (3D): " << rho_r_3d_noscale << " arcsec^-1" << endl;
	cout << endl;
	return true;
}

double QLens::mass2d_r(const double r, const int lensnum, const bool use_kpc)
{
	double sigma_cr, mass_r_2d, z;
	z = lens_list[lensnum]->zlens;
	double r_arcsec = (use_kpc) ? r*206.264806/cosmo->angular_diameter_distance(z) : r;

	sigma_cr = cosmo->sigma_crit_arcsec(z,reference_source_redshift);
	mass_r_2d = sigma_cr*lens_list[lensnum]->mass_rsq(r_arcsec*r_arcsec);
	return mass_r_2d;
}

double QLens::mass3d_r(const double r, const int lensnum, const bool use_kpc)
{
	double sigma_cr, mass_r_3d, z;
	z = lens_list[lensnum]->zlens;
	double r_arcsec = (use_kpc) ? r*206.264806/cosmo->angular_diameter_distance(z) : r;
	sigma_cr = cosmo->sigma_crit_arcsec(z,reference_source_redshift);
	mass_r_3d = sigma_cr*lens_list[lensnum]->calculate_scaled_mass_3d(r_arcsec);
	return mass_r_3d;
}

double QLens::calculate_average_log_slope(const int lensnum, const double rmin, const double rmax, const bool use_kpc)
{
	double z = lens_list[lensnum]->zlens;
	double kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(z);
	double rmin_arcsec = rmin, rmax_arcsec = rmax;
	if (use_kpc) {
		rmin_arcsec *= kpc_to_arcsec;
		rmax_arcsec *= kpc_to_arcsec;
	}
	return lens_list[lensnum]->average_log_slope(rmin_arcsec,rmax_arcsec);
}

void QLens::print_lens_list(bool show_vary_params)
{
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			cout << i << ". ";
			lens_list[i]->print_parameters();
			//string lline = lens_list[i]->get_parameters_string();
			//cout << lline << endl;
			if (show_vary_params)
				lens_list[i]->print_vary_parameters();
		}
		if (source_redshift != reference_source_redshift) cout << "NOTE: for all lenses, kappa is defined by zsrc_ref = " << reference_source_redshift << endl;
	}
	else cout << "No lens models have been specified" << endl;
	cout << endl;
}

void QLens::print_fit_model()
{
	if (nlens > 0) {
		cout << "Lenses:" << endl;
		print_lens_list(true);
	}
	if (n_sb > 0) {
		cout << "Source profile list:" << endl;
		print_source_list(true);
	}
	if (n_pixellated_src > 0) {
		cout << "Pixellated source list:" << endl;
		print_pixellated_source_list(true);
	}
	if (n_pixellated_lens > 0) {
		cout << "Pixellated lens list:" << endl;
		print_pixellated_lens_list(true);
	}
	if (n_ptsrc > 0) {
		cout << "Point source list:" << endl;
		print_point_source_list(true);
	}

	if (cosmo->get_n_vary_params() > 0) {
		cout << "Cosmology parameters:" << endl;
		cosmo->print_parameters();
		cosmo->print_vary_parameters();
		cout << endl;
	}
	if (n_vary_params > 0) {
		cout << "Miscellaneous parameters:" << endl;
		print_parameters(true);
		print_vary_parameters();
		cout << endl;
	}
}

void QLens::plot_ray_tracing_grid(double xmin, double xmax, double ymin, double ymax, int x_N, int y_N, string filename)
{

	lensvector **corner_pts = new lensvector*[x_N];
	lensvector **corner_sourcepts = new lensvector*[x_N];
	int i,j;
	for (i=0; i < x_N; i++) {
		corner_pts[i] = new lensvector[y_N];
		corner_sourcepts[i] = new lensvector[y_N];
	}

	double x,y;
	double pixel_xlength = (xmax-xmin)/(x_N-1);
	double pixel_ylength = (ymax-ymin)/(y_N-1);
	for (j=0, y=ymin; j < y_N; j++, y += pixel_ylength) {
		for (i=0, x=xmin; i < x_N; i++, x += pixel_xlength) {
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
			find_sourcept(corner_pts[i][j],corner_sourcepts[i][j],0,reference_zfactors,default_zsrc_beta_factors);
		}
	}
	ofstream outfile(filename.c_str());
	for (j=0, y=ymin; j < y_N-1; j++) {
		for (i=0, x=xmin; i < x_N-1; i++) {
			outfile << corner_sourcepts[i][j][0] << " " << corner_sourcepts[i][j][1] << endl;
			outfile << corner_sourcepts[i+1][j][0] << " " << corner_sourcepts[i+1][j][1] << endl;
			outfile << corner_sourcepts[i+1][j+1][0] << " " << corner_sourcepts[i+1][j+1][1] << endl;
			outfile << corner_sourcepts[i][j+1][0] << " " << corner_sourcepts[i][j+1][1] << endl;
			outfile << corner_sourcepts[i][j][0] << " " << corner_sourcepts[i][j][1] << endl;
			outfile << endl;
		}
	}
	for (i=0; i < x_N; i++) {
		delete[] corner_pts[i];
		delete[] corner_sourcepts[i];
	}
	delete[] corner_pts;
	delete[] corner_sourcepts;
}

void QLens::plot_logkappa_map(const int x_N, const int y_N, const string filename, const bool ignore_mask)
{
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = filename + ".x";
	string y_filename = filename + ".y";
	ofstream pixel_xvals(x_filename.c_str());
	ofstream pixel_yvals(y_filename.c_str());
	int i,j;
	for (i=0, x=xmin; i <= x_N; i++, x += xstep) {
		pixel_xvals << x << endl;
	}
	for (j=0, y=ymin; j <= y_N; j++, y += ystep) {
		pixel_yvals << y << endl;
	}
	pixel_xvals.close();
	pixel_yvals.close();

	string logkapname = filename + ".kappalog";
	ofstream logkapout(logkapname.c_str());

	double kap, mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	bool negkap = false; // Pseudo-elliptical models can produce negative kappa, so produce a warning if so
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			if ((!ignore_mask) and (imgdata_list[0] != NULL) and (!imgdata_list[0]->inside_mask(x,y))) logkapout << "NaN ";
			else {
				kap = kappa(pos,reference_zfactors,default_zsrc_beta_factors);
				//kap = kappa_exclude(pos,0,reference_zfactors,default_zsrc_beta_factors); // for looking at convergence of perturber
				if (kap < 0) {
					negkap = true;
					kap = abs(kap);
				}
				logkapout << log(kap)/log(10) << " ";
				//logkapout << kap << " ";
			}
		}
		logkapout << endl;
	}
	if (negkap==true) warn("kappa has negative values in some locations; plotting abs(kappa)");
}

void QLens::plot_logpot_map(const int x_N, const int y_N, const string filename)
{
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = filename + ".x";
	string y_filename = filename + ".y";
	ofstream pixel_xvals(x_filename.c_str());
	ofstream pixel_yvals(y_filename.c_str());
	int i,j;
	for (i=0, x=xmin; i <= x_N; i++, x += xstep) {
		pixel_xvals << x << endl;
	}
	for (j=0, y=ymin; j <= y_N; j++, y += ystep) {
		pixel_yvals << y << endl;
	}
	pixel_xvals.close();
	pixel_yvals.close();

	string logpotname = filename + ".potlog";
	ofstream logpotout(logpotname.c_str());

	double mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			pot = potential(pos,reference_zfactors,default_zsrc_beta_factors);
			logpotout << log(abs(pot))/log(10) << " ";
		}
		logpotout << endl;
	}
}

void QLens::plot_logmag_map(const int x_N, const int y_N, const string filename)
{
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = filename + ".x";
	string y_filename = filename + ".y";
	ofstream pixel_xvals(x_filename.c_str());
	ofstream pixel_yvals(y_filename.c_str());
	int i,j;
	for (i=0, x=xmin; i <= x_N; i++, x += xstep) {
		pixel_xvals << x << endl;
	}
	for (j=0, y=ymin; j <= y_N; j++, y += ystep) {
		pixel_yvals << y << endl;
	}
	pixel_xvals.close();
	pixel_yvals.close();

	string logmagname = filename + ".maglog";
	ofstream logmagout(logmagname.c_str());

	double mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			mag = magnification(pos,0,reference_zfactors,default_zsrc_beta_factors);
			logmagout << log(abs(mag))/log(10) << " ";
		}
		logmagout << endl;
	}
}

void QLens::plot_lensinfo_maps(string file_root, const int x_N, const int y_N, const int pert_residual)
{
	if (fit_output_dir != ".") create_output_directory();
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = fit_output_dir + "/" + file_root + ".x";
	string y_filename = fit_output_dir + "/" + file_root + ".y";
	ofstream pixel_xvals(x_filename.c_str());
	ofstream pixel_yvals(y_filename.c_str());
	int i,j;
	for (i=0, x=xmin; i <= x_N; i++, x += xstep) {
		pixel_xvals << x << endl;
	}
	for (j=0, y=ymin; j <= y_N; j++, y += ystep) {
		pixel_yvals << y << endl;
	}
	pixel_xvals.close();
	pixel_yvals.close();

	string kapname = fit_output_dir + "/" + file_root + ".kappa";
	string magname = fit_output_dir + "/" + file_root + ".mag";
	string invmagname = fit_output_dir + "/" + file_root + ".invmag";
	string shearname = fit_output_dir + "/" + file_root + ".shear";
	//string potname = fit_output_dir + "/" + file_root + ".pot";
	string defxname = fit_output_dir + "/" + file_root + ".defx";
	string defyname = fit_output_dir + "/" + file_root + ".defy";

	string logkapname = kapname + "log";
	string logmagname = magname + "log";
	string logshearname = shearname + "log";
	ofstream kapout(kapname.c_str());
	ofstream magout(magname.c_str());
	ofstream invmagout(invmagname.c_str());
	ofstream shearout(shearname.c_str());
	//ofstream potout(potname.c_str());
	ofstream defxout(defxname.c_str());
	ofstream defyout(defyname.c_str());
	ofstream logkapout(logkapname.c_str());
	ofstream logmagout(logmagname.c_str());
	ofstream logshearout(logshearname.c_str());

	bool *exclude = new bool[nlens];
	for (int i=0; i < nlens; i++) exclude[i] = false;
	if (pert_residual >= 0) {
		for (int i=0; i < nlens; i++) {
			if (i==pert_residual) exclude[i] = true;
		}
	}

	double kap, mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			kap = kappa(pos,reference_zfactors,default_zsrc_beta_factors);
			mag = magnification(pos,0,reference_zfactors,default_zsrc_beta_factors);
			invmag = inverse_magnification(pos,0,reference_zfactors,default_zsrc_beta_factors);
			shearval = shear(pos,0,reference_zfactors,default_zsrc_beta_factors);
			//pot = lens->potential(pos);
			deflection(pos,alpha,reference_zfactors,default_zsrc_beta_factors);
			if (pert_residual >= 0) {
				kap -= kappa_exclude(pos,exclude,reference_zfactors,default_zsrc_beta_factors);
				mag -= magnification_exclude(pos,exclude,0,reference_zfactors,default_zsrc_beta_factors);
				invmag -= inverse_magnification_exclude(pos,exclude,0,reference_zfactors,default_zsrc_beta_factors);
				shearval -= shear_exclude(pos,exclude,0,reference_zfactors,default_zsrc_beta_factors);
				//pot = lens->potential(pos);
				lensvector alpha_r;
				deflection_exclude(pos,exclude,alpha_r,reference_zfactors,default_zsrc_beta_factors);
				alpha[0] -= alpha_r[0];
				alpha[1] -= alpha_r[1];
			}

			kapout << kap << " ";
			magout << mag << " ";
			invmagout << invmag << " ";
			shearout << shearval << " ";
			//potout << pot << " ";
			defxout << alpha[0] << " ";
			defyout << alpha[1] << " ";
			if (kap==0) logkapout << "NaN ";
			else logkapout << log(kap)/log(10) << " ";
			if (mag==0) logmagout << "NaN ";
			else logmagout << log(abs(mag))/log(10) << " ";
			if (shearval==0) logshearout << "NaN ";
			else logshearout << log(abs(shearval))/log(10) << " ";
		}
		kapout << endl;
		invmagout << endl;
		shearout << endl;
		//potout << endl;
		defxout << endl;
		defyout << endl;
		logkapout << endl;
		logmagout << endl;
		logshearout << endl;
	}
	delete[] exclude;
}

// Pixel grid functions

void QLens::find_optimal_sourcegrid_for_analytic_source()
{
	if (n_sb==0) { warn("no source objects have been specified"); return; }
	sb_list[0]->window_params(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax);
	if (n_sb > 1) {
		double xmin, xmax, ymin, ymax;
		for (int i=1; i < n_sb; i++) {
			if (!sb_list[i]->is_lensed) continue;
			sb_list[i]->window_params(xmin,xmax,ymin,ymax);
			if (xmin < sourcegrid_xmin) {
				if (xmin > sourcegrid_limit_xmin) sourcegrid_xmin = xmin;
				else sourcegrid_xmin = sourcegrid_limit_xmin;
			}
			if (xmax > sourcegrid_xmax) {
				if (xmax < sourcegrid_limit_xmax) sourcegrid_xmax = xmax;
				else sourcegrid_xmax = sourcegrid_limit_xmax;
			}
			if (ymin < sourcegrid_ymin) {
				if (ymin > sourcegrid_limit_ymin) sourcegrid_ymin = ymin;
				else sourcegrid_ymin = sourcegrid_limit_ymin;
			}
			if (ymax > sourcegrid_ymax) {
				if (ymax < sourcegrid_limit_ymax) sourcegrid_ymax = ymax;
				else sourcegrid_ymax = sourcegrid_limit_ymax;
			}
		}
	}
}

bool QLens::create_sourcegrid_cartesian(const int band_number, const int zsrc_i, const bool verbal, const bool use_mask, const bool autogrid_from_analytic_source, const bool image_grid_already_exists, const bool make_auxiliary_srcgrid)
{
	bool use_image_pixelgrid = false;
	if ((adaptive_subgrid) and (nlens==0)) { cerr << "Error: cannot ray trace source for adaptive grid; no lens model has been specified\n"; return false; }
	if ((adaptive_subgrid) or (((auto_sourcegrid) or (auto_srcgrid_npixels)) and (nlens > 0))) use_image_pixelgrid = true;
	//if ((autogrid_from_analytic_source) and (n_sb==0)) { warn("no source objects have been specified"); return false; }
	if ((auto_sourcegrid) and (!autogrid_from_analytic_source) and (!image_grid_already_exists)) { warn("no image data have been generated from which to automatically set source grid dimensions"); return false; }
	if (cartesian_srcgrids == NULL) { warn("no pixellated sources have been created"); return false; }
	int imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;

	int src_i = -1;
	for (int i=0; i < n_pixellated_src; i++) {
		if ((pixellated_src_band[i]==band_number) and (pixellated_src_redshift_idx[i]==zsrc_i)) {
			src_i = i;
			break;
		}
	}
	if (src_i < 0) { warn("no pixellated source corresponding to given redshift has been created"); return false; }

	ImageData *image_data;
	if (n_data_bands > band_number) image_data = imgdata_list[band_number];
	else image_data = NULL;

	bool at_least_one_lensed_src = false;
	for (int k=0; k < n_sb; k++) {
		if (sb_list[k]->is_lensed) { at_least_one_lensed_src = true; break; }
	}
	if ((!image_grid_already_exists) and (use_image_pixelgrid) and (!at_least_one_lensed_src)) die("there are no analytic sources or current pixel grid available to generate source plot");

	if (cartesian_srcgrids[src_i]==NULL) die("cartesian sourcegrid should not be NULL"); // just for debugging purposes; remove this line later

	if (use_image_pixelgrid) {
		if (n_extended_src_redshifts==0) die("no ext src redshift has been created");

		if (!image_grid_already_exists) {
			double xmin,xmax,ymin,ymax;
			xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			xmax += 1e-10;
			ymax += 1e-10;
			if (image_pixel_grids[imggrid_i] == NULL) {
				//image_pixel_grid = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y,0);
				bool raytrace = true;
				if ((use_mask) and (image_data != NULL)) raytrace = false;
				image_pixel_grids[imggrid_i] = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y,raytrace,band_number,zsrc_i,imggrid_i);
				if ((use_mask) and (image_data != NULL)) image_pixel_grids[imggrid_i]->set_fit_window((*image_data),true,assigned_mask[imggrid_i],include_fgmask_in_inversion);
				if (band_number < n_psf) psf_list[band_number]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
			}
		}
		cartesian_srcgrids[src_i]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
		image_pixel_grids[imggrid_i]->set_cartesian_srcgrid(cartesian_srcgrids[src_i]);

		int n_imgpixels;
		if (auto_sourcegrid) {
			if (source_fit_mode != Delaunay_Source) {
				if ((autogrid_from_analytic_source) and (at_least_one_lensed_src)) {
					find_optimal_sourcegrid_for_analytic_source();
				}
				else {
					image_pixel_grids[imggrid_i]->find_optimal_sourcegrid(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,sourcegrid_limit_xmin,sourcegrid_limit_xmax,sourcegrid_limit_ymin,sourcegrid_limit_ymax);
				}
			} else {
				// Use the ray-traced points to define the source grid
				//image_pixel_grids[imggrid_i]->find_optimal_sourcegrid(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,sourcegrid_limit_xmin,sourcegrid_limit_xmax,sourcegrid_limit_ymin,sourcegrid_limit_ymax);
				image_pixel_grids[imggrid_i]->set_sourcegrid_params_from_ray_tracing(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,sourcegrid_limit_xmin,sourcegrid_limit_xmax,sourcegrid_limit_ymin,sourcegrid_limit_ymax);
			}
		}

		if ((auto_srcgrid_npixels) and (!make_auxiliary_srcgrid)) {
			if (auto_srcgrid_set_pixel_size) // this option doesn't work well, DON'T USE RIGHT NOW
				image_pixel_grids[imggrid_i]->find_optimal_firstlevel_sourcegrid_npixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_imgpixels);
			else
				image_pixel_grids[imggrid_i]->find_optimal_sourcegrid_npixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_imgpixels);
			if ((verbal) and (mpi_id==0)) {
				cout << "Optimal sourcegrid number of pixels: " << srcgrid_npixels_x << " " << srcgrid_npixels_y << endl;
				cout << "Sourcegrid dimensions: " << sourcegrid_xmin << " " << sourcegrid_xmax << " " << sourcegrid_ymin << " " << sourcegrid_ymax << endl;
				cout << "Number of active image pixels expected: " << n_imgpixels << endl;
			}
		}

		if ((srcgrid_npixels_x < 2) or (srcgrid_npixels_y < 2)) {
			warn("too few source pixels for ray tracing");
			//if (!image_grid_already_exists) {
				//delete image_pixel_grids[imggrid_i];
				//image_pixel_grids[imggrid_i] = NULL;
			//}
			return false;
		}
	} else {
		if ((auto_sourcegrid) and (autogrid_from_analytic_source) and (n_sb > 0)) find_optimal_sourcegrid_for_analytic_source();
	}

	if (auto_sourcegrid) {
		if (cartesian_srcgrids[src_i]->srcgrid_size_scale != 0) {
			double xwidth_adj = cartesian_srcgrids[src_i]->srcgrid_size_scale*(sourcegrid_xmax-sourcegrid_xmin);
			double ywidth_adj = cartesian_srcgrids[src_i]->srcgrid_size_scale*(sourcegrid_ymax-sourcegrid_ymin);
			double srcgrid_xc, srcgrid_yc;
			srcgrid_xc = (sourcegrid_xmax + sourcegrid_xmin)/2;
			srcgrid_yc = (sourcegrid_ymax + sourcegrid_ymin)/2;
			sourcegrid_xmin = srcgrid_xc - xwidth_adj/2;
			sourcegrid_xmax = srcgrid_xc + xwidth_adj/2;
			sourcegrid_ymin = srcgrid_yc - ywidth_adj/2;
			sourcegrid_ymax = srcgrid_yc + ywidth_adj/2;
		}
	}

	int nsplitx, nsplity;
	if (!make_auxiliary_srcgrid) {
		nsplitx = srcgrid_npixels_x;
		nsplity = srcgrid_npixels_y;
	} else {
		// an "auxiliary" source grid is a grid that is used to find average number of images produced, or source pixel magnifications,
		// but is NOT the grid used to model the source (typically the Delaunay grid is used instead)
		nsplitx = auxiliary_srcgrid_npixels;
		nsplity = auxiliary_srcgrid_npixels;
	}
	cartesian_srcgrids[src_i]->create_pixel_grid(this,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,nsplitx,nsplity);
	if ((mpi_id==0) and (verbal)) {
		cout << "# of Cartesian source pixels: " << cartesian_srcgrids[src_i]->number_of_pixels << endl;
	}

	if (adaptive_subgrid) {
		cartesian_srcgrids[src_i]->adaptive_subgrid();
		if ((mpi_id==0) and (verbal)) {
			cout << "# of source pixels after subgridding: " << cartesian_srcgrids[src_i]->number_of_pixels << endl;
		}
	}
	//if ((use_image_pixelgrid) and (!image_grid_already_exists)) {
		//delete image_pixel_grids[imggrid_i]; // shouldn't have to do this!!!
		//image_pixel_grids[imggrid_i] = NULL;
	//}
	return true;
}

bool QLens::create_sourcegrid_delaunay(const int src_i, const bool use_mask, const bool verbal)
{
	if (nlens==0) { cerr << "Error: cannot ray trace source for adaptive grid; no lens model has been specified\n"; return false; }
	if (n_sb==0) { warn("no source objects have been specified"); return false; }
	if (src_i >= n_pixellated_src) { cerr << "Pixellated source with given index has not been created" << endl; return false; }
	if (n_extended_src_redshifts==0) die("no ext src redshift has been created");
	int zsrc_i = pixellated_src_redshift_idx[src_i];
	int band_number = pixellated_src_band[src_i];
	int imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;

	ImageData *image_data;
	if (n_data_bands > band_number) image_data = imgdata_list[band_number];
	else image_data = NULL;

	double xmin,xmax,ymin,ymax;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xmax += 1e-10;
	ymax += 1e-10;
	if (image_pixel_grids[imggrid_i] == NULL) {
		bool raytrace = true;
		if ((use_mask) and (image_data != NULL)) raytrace = false;
		image_pixel_grids[imggrid_i] = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y,raytrace,band_number,zsrc_i,imggrid_i);
		if ((use_mask) and (image_data != NULL)) image_pixel_grids[imggrid_i]->set_fit_window((*image_data),true,assigned_mask[imggrid_i],include_fgmask_in_inversion); 
		if (band_number < n_psf) psf_list[band_number]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
	}

#ifdef USE_OPENMP
	double srcgrid_wtime0, srcgrid_wtime;
	if (show_wtime) {
		srcgrid_wtime0 = omp_get_wtime();
	}
#endif
	if (auto_sourcegrid) find_optimal_sourcegrid_for_analytic_source(); // this will just be for plotting purposes
	create_sourcegrid_from_imggrid_delaunay(false,band_number,zsrc_i,verbal);
	delaunay_srcgrids[src_i]->assign_surface_brightness_from_analytic_source(zsrc_i);

#ifdef USE_OPENMP
	if (show_wtime) {
		srcgrid_wtime = omp_get_wtime() - srcgrid_wtime0;
		if (mpi_id==0) cout << "wall time for Delaunay grid creation: " << srcgrid_wtime << endl;
	}
#endif
	return true;
}

bool QLens::create_sourcegrid_from_imggrid_delaunay(const bool use_weighted_srcpixel_clustering, const int band_number, const int zsrc_i, const bool verbal)
{
	if (delaunay_srcgrids == NULL) { warn("no pixellated sources have been created"); return false; }
	if (band_number >= n_model_bands) die("specified model band has not been created");
	int imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;

	int src_i = -1;
	for (int i=0; i < n_pixellated_src; i++) {
		if ((pixellated_src_band[i]==band_number) and (pixellated_src_redshift_idx[i]==zsrc_i)) {
			src_i = i;
			break;
		}
	}
	if ((src_i < 0) or (src_i >= n_pixellated_src)) { warn("no pixellated source corresponding to given redshift has been created"); return false; }
	if (delaunay_srcgrids[src_i]==NULL) die("Delaunay source grid should not be NULL!"); // just for debugging; remove this line later

	ImageData *image_data;
	if (n_data_bands > band_number) image_data = imgdata_list[band_number]; // temporary until I sort out the band stuff
	else image_data = NULL;

	double *srcpts_x, *srcpts_y;
	int *ivals, *jvals;
	int *pixptr_i, *pixptr_j;
	int npix_in_mask;
	if (imggrid_i >= n_image_pixel_grids) die("image grid index does not exist");
	//if (include_fgmask_in_inversion) {
		//npix_in_mask = image_pixel_grids[zsrc_i]->ntot_cells_emask;
		//pixptr_i = image_pixel_grids[zsrc_i]->emask_pixels_i;
		//pixptr_j = image_pixel_grids[zsrc_i]->emask_pixels_j;
	//} else {
		npix_in_mask = image_pixel_grids[imggrid_i]->ntot_cells;
		pixptr_i = image_pixel_grids[imggrid_i]->masked_pixels_i;
		pixptr_j = image_pixel_grids[imggrid_i]->masked_pixels_j;
	//}
	double avg_sb = -1e30;
	if (image_data) {
		double bgnoise = (use_noise_map) ? image_data->bg_pixel_noise : background_pixel_noise;
		avg_sb = image_data->find_avg_sb(10*bgnoise);
	}

	int i,j,k,l,n,npix=0,npix_in_lensing_mask=0; // npix_in_lensing_mask will be different from npix_in_mask if include_fgmask_in_inversion is turned on
	bool include;
	double max_sb = -1e30, min_sb = 1e30;
	double sbfrac = delaunay_high_sn_sbfrac;
	if (n_ptsrc > 0) sbfrac = 0; // if there are point sources in the data, then we can't use the peak surface brightness in the data image to help construct the Delaunay grid, since the Delaunay grid is only for the extended source
	bool *include_in_delaunay_grid = new bool[npix_in_mask];
	// if delaunay_high_sn_mode is on, we use sbfrac*avg_sb as the SB threshold to determine the region to have more source pixels;
	// avg_sb is also used to find where to compare grids 1/2
	int nsubpix, nysubpix;
	double sb;
	if (reinitialize_random_grid) reinitialize_random_generator();
	if (!include_fgmask_in_inversion) npix_in_lensing_mask = npix_in_mask;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		if ((include_fgmask_in_inversion) and (!(image_data->in_mask[zsrc_i][i][j]))) {
			include_in_delaunay_grid[n] = false; // if we're including the foreground mask in the inversion, we still only use the pixels within the lensing mask to define our source grid
		} else {
			if (include_fgmask_in_inversion) npix_in_lensing_mask++;
			include = false;
			nysubpix = image_pixel_grids[imggrid_i]->nsplits[i][j]; // why not just store the square and avoid having to always take the square?
			nsubpix = INTSQR(nysubpix); // why not just store the square and avoid having to always take the square?
			if ((use_srcpixel_clustering) or (use_weighted_srcpixel_clustering) or (delaunay_mode==5)) {
				include = true;
				sb = image_data->surface_brightness[i][j];
				if (sb > max_sb) max_sb = sb;
				if (sb < min_sb) min_sb = sb;
			} else {
				if ((delaunay_high_sn_mode) and (image_data->surface_brightness[i][j] > sbfrac*avg_sb)) {
					if ((delaunay_mode==1) or (delaunay_mode==2)) include = true;
					else if ((delaunay_mode==3) and (((i%2==0) and (j%2==0)) or ((i%2==1) and (j%2==1)))) include = true; // switch to mode 1 if S/N high enough
					else if ((delaunay_mode==4) and (((i%2==0) and (j%2==1)) or ((i%2==1) and (j%2==0)))) include = true; // switch to mode 2 if S/N high enough
					else if (image_data->surface_brightness[i][j] > 3*sbfrac*avg_sb) include = true; // if threshold is high enough, just include it
				}
				else if ((delaunay_mode==0) or (delaunay_mode==5)) include = true;
				else if ((delaunay_mode==1) and (((i%2==0) and (j%2==0)) or ((i%2==1) and (j%2==1)))) include = true;
				else if ((delaunay_mode==2) and (((i%2==0) and (j%2==1)) or ((i%2==1) and (j%2==0)))) include = true;
				else if ((delaunay_mode==3) and (((i%3==0) and (j%3==0)) or ((i%3==1) and (j%3==1)) or ((i%3==2) and (j%3==2)))) include = true;
				else if ((delaunay_mode==4) and (((i%4==0) and (j%4==0)) or ((i%4==1) and (j%4==1)) or ((i%4==2) and (j%4==2)) or ((i%4==3) and (j%4==3)))) include = true;
			}
			if ((use_srcpixel_clustering) or (use_weighted_srcpixel_clustering) or (delaunay_mode==5)) {
				npix += nsubpix;
			}
			else if (include) {
				npix++;
				if (split_imgpixels) npix++;
			}
			include_in_delaunay_grid[n] = include;
		}
	}
	if (min_sb < 0) min_sb = 0;

	srcpts_x = new double[npix];
	srcpts_y = new double[npix];
	double *wfactors = new double[npix];
	ivals = new int[npix];
	jvals = new int[npix];

	npix = 0;
	int subcell_i1, subcell_i2;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		if (include_in_delaunay_grid[n]) {
			if ((!split_imgpixels) or ((delaunay_from_pixel_centers) and (!use_srcpixel_clustering) and (!use_weighted_srcpixel_clustering))) {
				srcpts_x[npix] = image_pixel_grids[imggrid_i]->center_sourcepts[i][j][0];
				srcpts_y[npix] = image_pixel_grids[imggrid_i]->center_sourcepts[i][j][1];
			} else {
				nsubpix = INTSQR(image_pixel_grids[imggrid_i]->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
				if ((use_srcpixel_clustering) or (use_weighted_srcpixel_clustering) or (delaunay_mode==5)) {
					for (int k=0; k < nsubpix; k++) {
						srcpts_x[npix] = image_pixel_grids[imggrid_i]->subpixel_center_sourcepts[i][j][nsubpix-1-k][0];
						srcpts_y[npix] = image_pixel_grids[imggrid_i]->subpixel_center_sourcepts[i][j][nsubpix-1-k][1];
						ivals[npix] = i;
						jvals[npix] = j;
						if (use_weighted_srcpixel_clustering) wfactors[npix] = image_pixel_grids[imggrid_i]->subpixel_weights[i][j][nsubpix-1-k];
						npix++;
					}
				} else {
					if (use_random_delaunay_srcgrid) {
						subcell_i1 = (int) (nsubpix*RandomNumber());
						subcell_i2 = (int) (nsubpix*RandomNumber());
					} else {
						subcell_i1 = nsubpix-1 - ((i+2*j) % nsubpix); // this is really only optimized for 2x2 splittings
						subcell_i2 = nsubpix-1 - ((i+2*j+2) % nsubpix); // this is really only optimized for 2x2 splittings
					}

					srcpts_x[npix] = image_pixel_grids[imggrid_i]->subpixel_center_sourcepts[i][j][subcell_i1][0];
					srcpts_y[npix] = image_pixel_grids[imggrid_i]->subpixel_center_sourcepts[i][j][subcell_i1][1];
				}
			}
			//if (srcpts_x[npix]*0.0 != 0.0) die("nonsense source points!");
			if ((!use_srcpixel_clustering) and (!use_weighted_srcpixel_clustering) and (delaunay_mode != 5)) {
				ivals[npix] = i;
				jvals[npix] = j;
				npix++;
			}
		}
	}

	bool find_invmag = ((use_mag_weighted_regularization) and (zsrc_i==0)) ? true : false;
	if ((use_srcpixel_clustering) or (use_weighted_srcpixel_clustering)) {
#ifdef USE_MLPACK
		int *iweights_norm;
		double min_weight = 1e30;
		double *input_data = new double[2*npix];
		double *weights = new double[npix];
		double *initial_centroids;
		int *ivals_centroids;
		int *jvals_centroids;

		if (!use_weighted_srcpixel_clustering) {
			for (i=0; i < npix; i++) weights[i] = 1;
		} else {
			for (i=0; i < npix; i++) {
				//cout << "wfactor " << i << ": " << wfactors[i] << endl;
				weights[i] = pow(wfactors[i]+delaunay_srcgrids[src_i]->alpha_clus,delaunay_srcgrids[src_i]->beta_clus);
				if (weights[i] < min_weight) min_weight = weights[i];
			}
		}
		bool use_weighted_initial_centroids;
		if ((use_weighted_srcpixel_clustering) and (weight_initial_centroids) and (min_weight != 0)) use_weighted_initial_centroids = true;
		else use_weighted_initial_centroids = false;

		int n_src_centroids;
		if (use_f_src_clusters) {
			n_src_centroids = (int) (npix_in_lensing_mask * f_src_clusters);
		} else {
			n_src_centroids = n_src_clusters;	
			if (n_src_centroids < 0) n_src_centroids = npix_in_lensing_mask / 2;
			else if (n_src_centroids == 0) n_src_centroids = npix_in_lensing_mask;
		}

		int data_reduce_factor;
		int icent_offset=0;
		double xrand;
		if (!use_weighted_initial_centroids) {
			int ncorig = n_src_centroids;
			xrand = RandomNumber();
			data_reduce_factor = npix / n_src_centroids;
			icent_offset = (int) (data_reduce_factor*xrand);
			n_src_centroids = npix / data_reduce_factor;
			if (npix % data_reduce_factor > icent_offset) n_src_centroids++;
		} else {
			iweights_norm = new int[npix];
			int totweight=0;
			for (i=0; i < npix; i++) {
				iweights_norm[i] = (int) (pow(weights[i]/min_weight,0.3)); // trying the square root in an attempt to reduce noise in the original centroid assignments
				totweight += iweights_norm[i];
			}
			data_reduce_factor = totweight / n_src_centroids;
			n_src_centroids = totweight / data_reduce_factor;
			if (totweight % data_reduce_factor != 0) n_src_centroids++;
			//cout << "totweight = " << totweight << endl;
		}
		//cout << "n_centroids is " << n_src_centroids << endl;
		initial_centroids = new double[2*n_src_centroids];
		ivals_centroids = new int[n_src_centroids];
		jvals_centroids = new int[n_src_centroids];
		if (icent_offset >= data_reduce_factor) die("FOOK");
		if (!use_weighted_initial_centroids) {
			for (i=0,j=0,k=0,l=0; i < npix; i++) {
				input_data[j++] = srcpts_x[i];
				input_data[j++] = srcpts_y[i];
				if (i%data_reduce_factor==icent_offset) {
					initial_centroids[k++] = srcpts_x[i];
					initial_centroids[k++] = srcpts_y[i];
					ivals_centroids[l] = ivals[i]; // the centroid locations will shift after k-means, but the image-plane pixel ij will still ray-trace to somewhere near the centroid
					jvals_centroids[l] = jvals[i];
					//cout << "l=" << l << " ival=" << ivals_centroids[l] << " jval=" << jvals_centroids[l] << endl;
					l++;
				}
			}
			if (l != n_src_centroids) die("centroid miscount: %i %i",l,n_src_centroids);
		} else {
			int m,n,wnorm;
			for (i=0,j=0,k=0,l=0,n=0; i < npix; i++) {
				input_data[j++] = srcpts_x[i];
				input_data[j++] = srcpts_y[i];
				wnorm = iweights_norm[i];
				if (wnorm >= 2*data_reduce_factor) cout << "RUHROH! Will count a centroid twice due to overweighting" << endl;
				for (m=0; m < wnorm; m++) {
					if (n%data_reduce_factor==0) {
						initial_centroids[k++] = srcpts_x[i];
						initial_centroids[k++] = srcpts_y[i];
						ivals_centroids[l] = ivals[i];
						jvals_centroids[l] = jvals[i];
						l++;
					}
					n++;
				}
			}
			if (l != n_src_centroids) die("miscount of initial centroids");
			delete[] iweights_norm;
		}
		//cout << "Creating source grid for zsrc_i=" << zsrc_i << " with n=" << n_src_centroids << "pixels" << endl;

		arma::mat dataset(input_data, 2, npix);
		arma::Col<double> weightvec(weights, npix);
		arma::mat centroids(initial_centroids, 2, n_src_centroids);

		bool guess_initial_clusters;
		if (!clustering_random_initialization) guess_initial_clusters = true;
		else guess_initial_clusters = false;

		double *src_centroids_x = new double[n_src_centroids];
		double *src_centroids_y = new double[n_src_centroids];

		if (!use_dualtree_kmeans) {
			KMeans<EuclideanDistance, SampleInitialization, MaxVarianceNewCluster, NaiveKMeans> clus(n_cluster_iterations);
			clus.Cluster(dataset, n_src_centroids, centroids, weightvec, use_weighted_srcpixel_clustering, guess_initial_clusters);
			for (i=0; i < n_src_centroids; i++) {
				src_centroids_x[i] = (double) centroids(0,i);
				src_centroids_y[i] = (double) centroids(1,i);
			}

		} else {
			KMeans<EuclideanDistance, SampleInitialization, MaxVarianceNewCluster, DefaultDualTreeKMeans> clus(n_cluster_iterations);
			bool status;
			status = clus.Cluster(dataset, n_src_centroids, centroids, weightvec, use_weighted_srcpixel_clustering, guess_initial_clusters);
			if (status==false) {
				warn("Dual-tree k-means algorithm failed, so using naive k-means instead");
				// Dual Tree didn't work, so let's use naive k-means instead
				arma::mat dataset2(input_data, 2, npix);
				arma::Col<double> weightvec2(weights, npix);
				arma::mat centroids2(initial_centroids, 2, n_src_centroids);
				KMeans<EuclideanDistance, SampleInitialization, MaxVarianceNewCluster, NaiveKMeans> clus_naive(n_cluster_iterations);
				clus_naive.Cluster(dataset2, n_src_centroids, centroids2, weightvec2, use_weighted_srcpixel_clustering, guess_initial_clusters);
				for (i=0; i < n_src_centroids; i++) {
					src_centroids_x[i] = (double) centroids2(0,i);
					src_centroids_y[i] = (double) centroids2(1,i);
				}
			} else {
				for (i=0; i < n_src_centroids; i++) {
					src_centroids_x[i] = (double) centroids(0,i);
					src_centroids_y[i] = (double) centroids(1,i);
				}
			}
		}
		delete[] input_data;
		delete[] initial_centroids;
		delete[] weights;


		if ((mpi_id==0) and (verbal)) cout << "Delaunay grid (with clustering) has n_pixels=" << n_src_centroids << endl;
		//cout << "Source grid = (" << sourcegrid_xmin << "," << sourcegrid_xmax << ") x (" << sourcegrid_ymin << "," << sourcegrid_ymax << ")";
		delaunay_srcgrids[src_i]->create_srcpixel_grid(src_centroids_x,src_centroids_y,n_src_centroids,ivals_centroids,jvals_centroids,n_image_pixels_x,n_image_pixels_y,find_invmag,imggrid_i);
		double edge_sum = delaunay_srcgrids[src_i]->sum_edge_sqrlengths(avg_sb);
		if ((mpi_id==0) and (verbal)) cout << "Delaunay source grid edge_sum: " << edge_sum << endl;
		delete[] src_centroids_x;
		delete[] src_centroids_y;
		delete[] ivals_centroids;
		delete[] jvals_centroids;
#else
		die("Must compile with -DUSE_MLPACK option to use source pixel clustering algorithm with adaptive grid");
#endif
	} else {
		if ((mpi_id==0) and (verbal)) cout << "Delaunay grid has n_pixels=" << npix << endl;
		//for (int i=0; i < npix; i++) {
			//cout << "IJ(" << i << "): " << ivals[i] << " " << jvals[i] << " BAND=" << band_number << endl;
		//}
		delaunay_srcgrids[src_i]->create_srcpixel_grid(srcpts_x,srcpts_y,npix,ivals,jvals,n_image_pixels_x,n_image_pixels_y,find_invmag,imggrid_i);
		double edge_sum = delaunay_srcgrids[src_i]->sum_edge_sqrlengths(avg_sb);
		if ((mpi_id==0) and (verbal)) cout << "Delaunay source grid edge_sum: " << edge_sum << endl;
	}

	delete[] include_in_delaunay_grid;
	delete[] srcpts_x;
	delete[] srcpts_y;
	delete[] wfactors;
	delete[] ivals;
	delete[] jvals;
	return true;
}

bool QLens::create_lensgrid_cartesian(const int band_number, const int zsrc_i, const int pixlens_i, const bool verbal, const bool use_mask)
{
	if (lensgrids == NULL) { warn("no pixellated lenses have been created"); return false; }
	if ((pixlens_i < 0) or (pixlens_i >= n_pixellated_lens)) { warn("the pixellated lens specified has not been created"); return false; }
	if (zsrc_i >= n_extended_src_redshifts) die("image grid index does not exist");
	if (band_number >= n_model_bands) die("specified model band has not been created");
	int imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;

	ImageData *image_data;
	if (n_data_bands > band_number) image_data = imgdata_list[band_number]; // temporary until I sort out the band stuff
	else image_data = NULL;

	double xmin,xmax,ymin,ymax;
	int npx, npy;
	if (image_pixel_grids[imggrid_i] == NULL) {
		xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
		ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
		xmax += 1e-10;
		ymax += 1e-10;
		//image_pixel_grid = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y,0);
		bool raytrace = true;
		if ((use_mask) and (image_data != NULL)) raytrace = false;
		image_pixel_grids[imggrid_i] = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y,raytrace,band_number,zsrc_i,imggrid_i);
		if ((use_mask) and (image_data != NULL)) image_pixel_grids[imggrid_i]->set_fit_window((*image_data),true,assigned_mask[imggrid_i],include_fgmask_in_inversion); 
		if (band_number < n_psf) psf_list[band_number]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
		npx = n_image_pixels_x;
		npy = n_image_pixels_y;
	} else {
		image_pixel_grids[imggrid_i]->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
	}
	lensgrids[pixlens_i]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
	image_pixel_grids[imggrid_i]->set_lensgrid(lensgrids[pixlens_i]);

	lensgrids[pixlens_i]->create_cartesian_pixel_grid(xmin,xmax,ymin,ymax,zsrc_i);

	return true;
}


/*
bool QLens::create_lensgrid_from_imggrid_delaunay(const int zsrc_i, const int pixlens_i, const bool verbal)
{
	if (lensgrids == NULL) { warn("no pixellated lens grids have been created"); return false; }
	if ((pixlens_i < 0) or (pixlens_i >= n_pixellated_lens)) { warn("the pixellated lens specified has not been created"); return false; }
	double *pts_x, *pts_y;
	int *ivals, *jvals;
	int *pixptr_i, *pixptr_j;
	int npix_in_mask;
	if (zsrc_i >= n_extended_src_redshifts) die("image grid index does not exist");
	if (include_fgmask_in_inversion) {
		npix_in_mask = image_pixel_grids[zsrc_i]->ntot_cells_emask;
		pixptr_i = image_pixel_grids[zsrc_i]->emask_pixels_i;
		pixptr_j = image_pixel_grids[zsrc_i]->emask_pixels_j;
	} else {
		npix_in_mask = image_pixel_grids[zsrc_i]->ntot_cells;
		pixptr_i = image_pixel_grids[zsrc_i]->masked_pixels_i;
		pixptr_j = image_pixel_grids[zsrc_i]->masked_pixels_j;
	}
	double avg_sb = -1e30;
	if (image_data) avg_sb = image_data->find_avg_sb(10*background_pixel_noise);

	int i,j,k,l,n,npix=0;
	bool include;
	double max_sb = -1e30, min_sb = 1e30;
	double sbfrac = delaunay_high_sn_sbfrac;
	if (n_ptsrc > 0) sbfrac = 0; // if there are point sources in the data, then we can't use the peak surface brightness in the data image to help construct the Delaunay grid, since the Delaunay grid is only for the extended source
	bool *include_in_delaunay_grid = new bool[npix_in_mask];
	// if delaunay_high_sn_mode is on, we use sbfrac*avg_sb as the SB threshold to determine the region to have more source pixels;
	// avg_sb is also used to find where to compare grids 1/2
	int nsubpix, nysubpix;
	double sb;
	if (reinitialize_random_grid) reinitialize_random_generator();
	for (n=0; n < npix_in_mask; n++) {
		include = false;
		i = pixptr_i[n];
		j = pixptr_j[n];
		nysubpix = image_pixel_grids[zsrc_i]->nsplits[i][j]; // why not just store the square and avoid having to always take the square?
		nsubpix = INTSQR(nysubpix); // why not just store the square and avoid having to always take the square?
		if ((use_lenspixel_clustering) or (delaunay_mode==5)) {
			include = true;
			sb = image_data->surface_brightness[i][j];
			if (sb > max_sb) max_sb = sb;
			if (sb < min_sb) min_sb = sb;
		} else {
			if ((delaunay_high_sn_mode) and (image_data->surface_brightness[i][j] > sbfrac*avg_sb)) {
				if ((delaunay_mode==1) or (delaunay_mode==2)) include = true;
				else if ((delaunay_mode==3) and (((i%2==0) and (j%2==0)) or ((i%2==1) and (j%2==1)))) include = true; // switch to mode 1 if S/N high enough
				else if ((delaunay_mode==4) and (((i%2==0) and (j%2==1)) or ((i%2==1) and (j%2==0)))) include = true; // switch to mode 2 if S/N high enough
				else if (image_data->surface_brightness[i][j] > 3*sbfrac*avg_sb) include = true; // if threshold is high enough, just include it
			}
			else if ((delaunay_mode==0) or (delaunay_mode==5)) include = true;
			else if ((delaunay_mode==1) and (((i%2==0) and (j%2==0)) or ((i%2==1) and (j%2==1)))) include = true;
			else if ((delaunay_mode==2) and (((i%2==0) and (j%2==1)) or ((i%2==1) and (j%2==0)))) include = true;
			else if ((delaunay_mode==3) and (((i%3==0) and (j%3==0)) or ((i%3==1) and (j%3==1)) or ((i%3==2) and (j%3==2)))) include = true;
			else if ((delaunay_mode==4) and (((i%4==0) and (j%4==0)) or ((i%4==1) and (j%4==1)) or ((i%4==2) and (j%4==2)) or ((i%4==3) and (j%4==3)))) include = true;
		}
		if ((use_lenspixel_clustering) or (delaunay_mode==5)) npix += nsubpix;
		else if (include) {
			npix++;
			if (split_imgpixels) npix++;
		}
		include_in_delaunay_grid[n] = include;
	}
	if (min_sb < 0) min_sb = 0;

	pts_x = new double[npix];
	pts_y = new double[npix];
	double *wfactors = new double[npix];
	ivals = new int[npix];
	jvals = new int[npix];

	npix = 0;
	int subcell_i1, subcell_i2;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		if (include_in_delaunay_grid[n]) {
			if ((!split_imgpixels) or ((delaunay_from_pixel_centers) and (!use_lenspixel_clustering))) {
				pts_x[npix] = image_pixel_grids[zsrc_i]->center_pts[i][j][0];
				pts_y[npix] = image_pixel_grids[zsrc_i]->center_pts[i][j][1];
			} else {
				nsubpix = INTSQR(image_pixel_grids[zsrc_i]->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
				if ((use_lenspixel_clustering) or (delaunay_mode==5)) {
					for (int k=0; k < nsubpix; k++) {
						pts_x[npix] = image_pixel_grids[zsrc_i]->subpixel_center_pts[i][j][nsubpix-1-k][0];
						pts_y[npix] = image_pixel_grids[zsrc_i]->subpixel_center_pts[i][j][nsubpix-1-k][1];
						ivals[npix] = i;
						jvals[npix] = j;
						if (use_lenspixel_clustering) wfactors[npix] = image_pixel_grids[zsrc_i]->subpixel_source_gradient[i][j][nsubpix-1-k]; // subpixel weights are determined by gradient of source
						npix++;
					}
				} else {
					if (use_random_delaunay_srcgrid) {
						subcell_i1 = (int) (nsubpix*RandomNumber());
						subcell_i2 = (int) (nsubpix*RandomNumber());
					} else {
						subcell_i1 = nsubpix-1 - ((i+2*j) % nsubpix); // this is really only optimized for 2x2 splittings
						subcell_i2 = nsubpix-1 - ((i+2*j+2) % nsubpix); // this is really only optimized for 2x2 splittings
					}

					pts_x[npix] = image_pixel_grids[zsrc_i]->subpixel_center_pts[i][j][subcell_i1][0];
					pts_y[npix] = image_pixel_grids[zsrc_i]->subpixel_center_pts[i][j][subcell_i1][1];
				}
			}
			//if (pts_x[npix]*0.0 != 0.0) die("nonsense source points!");
			if ((!use_lenspixel_clustering) and (delaunay_mode != 5)) {
				ivals[npix] = i;
				jvals[npix] = j;
				npix++;
			}
		}
	}

	bool find_invmag = ((use_mag_weighted_regularization) and (zsrc_i==0)) ? true : false;
	if (use_lenspixel_clustering) {
#ifdef USE_MLPACK
		int *iweights_norm;
		double min_weight = 1e30;
		double *input_data = new double[2*npix];
		double *weights = new double[npix];
		double *initial_centroids;
		int *ivals_centroids;
		int *jvals_centroids;

		for (i=0; i < npix; i++) {
			//cout << "wfactor " << i << ": " << wfactors[i] << endl;
			weights[i] = pow(wfactors[i]+lensgrids[pixlens_i]->alpha_clus,lensgrids[pixlens_i]->beta_clus);
			if (weights[i] < min_weight) min_weight = weights[i];
		}

		int n_lensgrid_centroids = n_lensgrid_clusters;	
		if (n_lensgrid_centroids < 0) n_lensgrid_centroids = npix_in_mask / 2;
		else if (n_lensgrid_centroids == 0) n_lensgrid_centroids = npix_in_mask;

		int data_reduce_factor;
		int icent_offset;
		double xrand;

		int ncorig = n_lensgrid_centroids;
		xrand = RandomNumber();
		data_reduce_factor = npix / n_lensgrid_centroids;
		icent_offset = (int) (data_reduce_factor*xrand);
		n_lensgrid_centroids = npix / data_reduce_factor;
		if (npix % data_reduce_factor > icent_offset) n_lensgrid_centroids++;

		//cout << "n_centroids is " << n_lensgrid_centroids << endl;
		initial_centroids = new double[2*n_lensgrid_centroids];
		ivals_centroids = new int[n_lensgrid_centroids];
		jvals_centroids = new int[n_lensgrid_centroids];
		if (icent_offset >= data_reduce_factor) die("FOOK");
		for (i=0,j=0,k=0,l=0; i < npix; i++) {
			input_data[j++] = pts_x[i];
			input_data[j++] = pts_y[i];
			if (i%data_reduce_factor==icent_offset) {
				initial_centroids[k++] = pts_x[i];
				initial_centroids[k++] = pts_y[i];
				ivals_centroids[l] = ivals[i]; // the centroid locations will shift after k-means, but the image-plane pixel ij will still ray-trace to somewhere near the centroid
				jvals_centroids[l] = jvals[i];
				//cout << "l=" << l << " ival=" << ivals_centroids[l] << " jval=" << jvals_centroids[l] << endl;
				l++;
			}
		}
		if (l != n_lensgrid_centroids) die("centroid miscount: %i %i",l,n_lensgrid_centroids);

		arma::mat dataset(input_data, 2, npix);
		arma::Col<double> weightvec(weights, npix);
		arma::mat centroids(initial_centroids, 2, n_lensgrid_centroids);

		bool guess_initial_clusters;
		if (!clustering_random_initialization) guess_initial_clusters = true;
		else guess_initial_clusters = false;

		double *lensgrid_centroids_x = new double[n_lensgrid_centroids];
		double *lensgrid_centroids_y = new double[n_lensgrid_centroids];

		if (!use_dualtree_kmeans) {
			KMeans<EuclideanDistance, SampleInitialization, MaxVarianceNewCluster, NaiveKMeans> clus(n_cluster_iterations);
			clus.Cluster(dataset, n_lensgrid_centroids, centroids, weightvec, true, guess_initial_clusters);
			for (i=0; i < n_lensgrid_centroids; i++) {
				lensgrid_centroids_x[i] = (double) centroids(0,i);
				lensgrid_centroids_y[i] = (double) centroids(1,i);
			}

		} else {
			KMeans<EuclideanDistance, SampleInitialization, MaxVarianceNewCluster, DefaultDualTreeKMeans> clus(n_cluster_iterations);
			bool status;
			status = clus.Cluster(dataset, n_lensgrid_centroids, centroids, weightvec, use_weighted_srcpixel_clustering, guess_initial_clusters);
			if (status==false) {
				warn("Dual-tree k-means algorithm failed, so using naive k-means instead");
				// Dual Tree didn't work, so let's use naive k-means instead
				arma::mat dataset2(input_data, 2, npix);
				arma::Col<double> weightvec2(weights, npix);
				arma::mat centroids2(initial_centroids, 2, n_lensgrid_centroids);
				KMeans<EuclideanDistance, SampleInitialization, MaxVarianceNewCluster, NaiveKMeans> clus_naive(n_cluster_iterations);
				clus_naive.Cluster(dataset2, n_lensgrid_centroids, centroids2, weightvec2, use_weighted_srcpixel_clustering, guess_initial_clusters);
				for (i=0; i < n_lensgrid_centroids; i++) {
					lensgrid_centroids_x[i] = (double) centroids2(0,i);
					lensgrid_centroids_y[i] = (double) centroids2(1,i);
				}
			} else {
				for (i=0; i < n_lensgrid_centroids; i++) {
					lensgrid_centroids_x[i] = (double) centroids(0,i);
					lensgrid_centroids_y[i] = (double) centroids(1,i);
				}
			}
		}
		delete[] input_data;
		delete[] initial_centroids;
		delete[] weights;


		if ((mpi_id==0) and (verbal)) cout << "Delaunay grid (with clustering) has n_pixels=" << n_lensgrid_centroids << endl;
		//cout << "Source grid = (" << sourcegrid_xmin << "," << sourcegrid_xmax << ") x (" << sourcegrid_ymin << "," << sourcegrid_ymax << ")";
		lensgrids[pixlens_i]->create_srcpixel_grid(lensgrid_centroids_x,lensgrid_centroids_y,n_lensgrid_centroids,ivals_centroids,jvals_centroids,n_image_pixels_x,n_image_pixels_y,find_invmag,zsrc_i);
		double edge_sum = lensgrids[pixlens_i]->sum_edge_sqrlengths(avg_sb);
		if ((mpi_id==0) and (verbal)) cout << "Delaunay source grid edge_sum: " << edge_sum << endl;
		delete[] lensgrid_centroids_x;
		delete[] lensgrid_centroids_y;
		delete[] ivals_centroids;
		delete[] jvals_centroids;
#else
		die("Must compile with -DUSE_MLPACK option to use source pixel clustering algorithm with adaptive grid");
#endif
	} else {
		if ((mpi_id==0) and (verbal)) cout << "Delaunay grid has n_pixels=" << npix << endl;
		if (lensgrids[pixlens_i]==NULL) lensgrids[pixlens_i] = new LensPixelGrid(this);
		lensgrids[pixlens_i]->create_delaunay_pixel_grid(pts_x,pts_y,npix,ivals,jvals,n_image_pixels_x,n_image_pixels_y,find_invmag,zsrc_i);
	}

	delete[] include_in_delaunay_grid;
	delete[] pts_x;
	delete[] pts_y;
	delete[] wfactors;
	delete[] ivals;
	delete[] jvals;
	return true;
}
*/

void QLens::plot_source_pixel_grid(const int imggrid_i, const char filename[])
{
	if ((imggrid_i >= 0) and (n_extended_src_redshifts==0)) die("no ext src redshift created");
	CartesianSourceGrid *cartesian_srcgrid = image_pixel_grids[imggrid_i]->cartesian_srcgrid;

	if (cartesian_srcgrid==NULL) { warn("No source surface brightness map has been generated"); return; }
	cartesian_srcgrid->xgrid.open(filename, ifstream::out);
	cartesian_srcgrid->plot_corner_coordinates(cartesian_srcgrid->xgrid);
	cartesian_srcgrid->xgrid.close();
}

void QLens::find_source_centroid(const int imggrid_i, double& xc_approx, double& yc_approx, const bool verbal)
{
	// This function is accessed by the LensProfile class when a lens is anchored to a reconstructed source
	if ((image_pixel_grids==NULL) or (image_pixel_grids[imggrid_i]==NULL)) {
		warn("cannot find approximate source size; image pixel grid does not exist");
		xc_approx = 1e30;
		yc_approx = 1e30;
	} else {
		image_pixel_grids[imggrid_i]->find_approx_source_size(xc_approx,yc_approx,verbal);
	}
}

bool QLens::load_image_pixel_data(const int band_i, string image_pixel_filename_root, const double pixsize, const double pix_xy_ratio, const double x_offset, const double y_offset, const int hdu_indx, const bool show_fits_header)
{
	bool first_data_img = false;
	if (band_i > n_data_bands) return false;
	if (band_i==n_data_bands) {
		add_image_pixel_data();
		if (n_data_bands==1) first_data_img = true;
	}
	while (n_model_bands < n_data_bands) add_new_model_band();

	ImageData *image_data = imgdata_list[band_i];

	bool status = true;
	if (fits_format == true) {
		if (pixsize <= 0) { // in this case no pixel scale has been specified, so we simply use the grid that has already been chosen
			double xmin,xmax,ymin,ymax;
			xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			image_data->set_grid_params(xmin,xmax,ymin,ymax); // these functions are defined in the header pixelgrid.h
			// note that if the data pixel size is found in the FITS header, it will override the dimensions set above
		}
		//status = image_data->load_data_fits(xmin,xmax,ymin,ymax,image_pixel_filename_root,hdu_indx,show_fits_header); // these functions are defined in the header pixelgrid.h
		status = image_data->load_data_fits(image_pixel_filename_root,pixsize,pix_xy_ratio,x_offset,y_offset,hdu_indx,show_fits_header);
		// the pixel size may have been specified in the FITS file, in which case data pixel size was just set to something > 0
		if ((status==true) and (pixsize > 0)) {
			double xmin,xmax,ymin,ymax;
			int npx, npy;
			image_data->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
			grid_xlength = xmax-xmin;
			grid_ylength = ymax-ymin;
			set_gridcenter(0.5*(xmin+xmax),0.5*(ymin+ymax));
		}
	} else {
		image_data->load_data(image_pixel_filename_root);
		double xmin,xmax,ymin,ymax;
		int npx, npy;
		image_data->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
		grid_xlength = xmax-xmin;
		grid_ylength = ymax-ymin;
		set_gridcenter(0.5*(xmin+xmax),0.5*(ymin+ymax));
	}
	if (status==false) {
		if (first_data_img) {
			delete image_data;
			image_data = NULL;
		}
		return false;
	}
	image_data->get_npixels(n_image_pixels_x,n_image_pixels_y);
	/*
	if ((status==true) and (image_pixel_grids != NULL)) {
		// delete the image pixel grids so that they will be remade according to the dimensions in the image data
		for (int i=0; i < n_extended_src_redshifts; i++) {
			if (image_pixel_grids[i] != NULL) {
				delete image_pixel_grids[i];
				image_pixel_grids[i] = NULL;
			}
		}
		for (int i=0; i < n_pixellated_src; i++) {
			// make sure the source grids aren't pointing to image pixel grids that don't exist
			if ((delaunay_srcgrids != NULL) and (delaunay_srcgrids[i] != NULL)) delaunay_srcgrids[i]->image_pixel_grid = NULL;
			if ((cartesian_srcgrids != NULL) and (cartesian_srcgrids[i] != NULL)) cartesian_srcgrids[i]->image_pixel_grid = NULL;
		}
		for (int i=0; i < n_pixellated_src; i++) {
			if ((lensgrids != NULL) and (lensgrids[i] != NULL)) lensgrids[i]->image_pixel_grid = NULL;
		}
	}
	*/	
	// Make sure the grid size & center are fixed now
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
	return true;
}

void QLens::update_imggrid_mask_values(const int mask_i)
{
	for (int i=0; i < n_extended_src_redshifts; i++) {
		if ((assigned_mask != NULL) and (assigned_mask[i]==mask_i) and (image_pixel_grids != NULL) and (image_pixel_grids[i] != NULL)) image_pixel_grids[i]->update_mask_values(include_fgmask_in_inversion);
	}
}

void QLens::plot_sbmap(const string filename, dvector& xvals, dvector& yvals, dvector& zvals, const bool plot_fits)
{
	int i,j,k,nx,ny;
	nx = xvals.size()-1; // since xvals contains the corner points, not the center points
	ny = yvals.size()-1;
	if (!plot_fits) {
		string sb_filename = filename + ".dat";
		string x_filename = filename + ".x";
		string y_filename = filename + ".y";

		ofstream pixel_image_file; open_output_file(pixel_image_file,sb_filename);
		ofstream pixel_xvals; open_output_file(pixel_xvals,x_filename);
		ofstream pixel_yvals; open_output_file(pixel_yvals,y_filename);
		pixel_image_file << setiosflags(ios::scientific);
		for (i=0; i <= nx; i++) {
			pixel_xvals << xvals[i] << endl;
		}
		for (j=0; j <= ny; j++) {
			pixel_yvals << yvals[j] << endl;
		}	
		k=0;
		for (j=0; j < ny; j++) {
			for (i=0; i < nx; i++) {
				pixel_image_file << zvals[k++];
				if (i < nx-1) pixel_image_file << " ";
			}
			pixel_image_file << endl;
		}
	} else {
#ifndef USE_FITS
		cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return;
#else
		int k,kk;
		fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
		int status = 0;   // CFITSIO status value MUST be initialized to zero!
		int bitpix = -64, naxis = 2;
		long naxes[2] = {nx,ny};
		double *pixels;
		double pixel_xlength = xvals[1]-xvals[0];
		double pixel_ylength = yvals[1]-yvals[0];
		if (fit_output_dir != ".") create_output_directory(); // in case it hasn't been created already
		string fits_filename = "!" + fit_output_dir + "/" + filename; // ensures that it overwrites an existing file of the same name

		if (!fits_create_file(&outfptr, fits_filename.c_str(), &status))
		{
			if (!fits_create_img(outfptr, bitpix, naxis, naxes, &status))
			{
				if (naxis == 0) {
					die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
				} else {
					kk=0;
					long fpixel[naxis];
					for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
					pixels = new double[nx];

					k=0;
					for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
					{
						for (i=0; i < nx; i++) {
							pixels[i] = zvals[k++];
						}
						fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
					}
					delete[] pixels;
				}
				if (pixel_xlength==pixel_ylength) {
					fits_write_key(outfptr, TDOUBLE, "PXSIZE", &pixel_xlength, "length of square pixels (in arcsec)", &status);
				} else {
					if (mpi_id==0) cout << "NOTE: pixel length not equal in x- versus y-directions; not saving pixel size in FITS file header" << endl;
				}
				if ((simulate_pixel_noise) and (!use_noise_map))
					fits_write_key(outfptr, TDOUBLE, "PXNOISE", &background_pixel_noise, "pixel surface brightness noise", &status);
				fits_write_key(outfptr, TDOUBLE, "ZSRC", &source_redshift, "redshift of source galaxy", &status);
				if (nlens > 0) {
					double zl = lens_list[primary_lens_number]->get_redshift();
					fits_write_key(outfptr, TDOUBLE, "ZLENS", &zl, "redshift of primary qlens", &status);
				}

				if (data_info != "") {
					string comment = "ql: " + data_info;
					fits_write_comment(outfptr, comment.c_str(), &status);
				}
				if (param_markers != "") {
					string param_markers_comma = param_markers;
					// Commas are used as delimeter in FITS file so spaces won't get lost when reading it in
					for (size_t i = 0; i < param_markers_comma.size(); ++i) {
						 if (param_markers_comma[i] == ' ') {
							  param_markers_comma.replace(i, 1, ",");
						 }
					}

					string comment = "mk: " + param_markers_comma;
					fits_write_comment(outfptr, comment.c_str(), &status);
				}
			}
			fits_close_file(outfptr, &status);
		} 

		if (status) fits_report_error(stderr, status); // print any error message
#endif
	}
}

const bool QLens::output_lensed_surface_brightness(dvector& xvals, dvector& yvals, dvector& zvals, const int band_number, const bool output_fits, const bool plot_residual, bool plot_foreground_only, const bool omit_foreground, const bool show_all_pixels, const bool normalize_residuals, const bool offload_to_data, const bool show_extended_mask, const bool show_foreground_mask, const bool show_noise_thresh, const bool exclude_ptimgs, const bool only_ptimgs, int specific_zsrc_i, const bool show_only_first_order_corrections, const bool plot_log, const bool plot_current_sb, const bool verbose)
{
	// Note that if specific_zsrc_i is negative, it will plot images from *all* source redshifts
	// You need to simplify the code in this function. It's too convoluted!!!
	if (source_fit_mode==Cartesian_Source) {
		if ((cartesian_srcgrids==NULL) and (n_ptsrc==0)) { warn("No Cartesian source grid has been generated"); return false; }
	} else if (source_fit_mode==Delaunay_Source) {
		if (n_ptsrc==0) {
			if ((delaunay_srcgrids==NULL) or (delaunay_srcgrids[0]==NULL)) { warn("No Delaunay source grid has been generated"); return false; }
			if ((image_pixel_grids != NULL) and (image_pixel_grids[0] != NULL) and (image_pixel_grids[0]->delaunay_srcgrid == NULL)) { warn("No Delaunay source grid has been generated"); return false; }
		}
	}
	if (image_pixel_grids == NULL) { warn("no extended source redshifts have been setup"); return false; }
	if (((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) and (n_sb==0) and (n_ptsrc==0)) { warn("No surface brightness profiles have been defined"); return false; }
	if ((plot_foreground_only) and (omit_foreground)) { warn("cannot omit both foreground and lensed sources when plotting"); return false; }

	ImageData *image_data;
	if (n_data_bands > band_number) image_data = imgdata_list[band_number];
	else image_data = NULL;

	bool use_data = true;
	if (image_data==NULL) use_data = false;
	if (band_number >= n_model_bands) { warn("specified band number has not been created"); return false; }
	if ((plot_residual==true) and (!image_data)) { warn("cannot plot residual image, pixel data image has been loaded or cannot be used"); return false; }
	double xmin,xmax,ymin,ymax;
	if (use_data) {
		image_data->get_grid_params(xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y);
	} else {
		xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
		ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
		//xmax += 1e-10; // is this still necessary? Check
		//ymax += 1e-10;
	}
	if (n_extended_src_redshifts==0) die("no extended source redshifts have been created");
	bool include_fgmask_in_inversion_orig = include_fgmask_in_inversion;
	if ((show_all_pixels) or (show_extended_mask)) include_fgmask_in_inversion = false;
	//bool raytrace = ((use_data) or (plot_foreground_only)) ? false : true;
	ImagePixelGrid* image_pixel_grid;
	int mask_num;
	int zsrc_i_0 = 0;
	int zsrc_i_f = n_extended_src_redshifts;
	int specific_imggrid_i = -1;
	int primary_imggrid_i = band_number*n_extended_src_redshifts;
	if (specific_zsrc_i >= 0) {
		zsrc_i_0 = specific_zsrc_i;
		zsrc_i_f = specific_zsrc_i+1;
		specific_imggrid_i = primary_imggrid_i + specific_zsrc_i;
		primary_imggrid_i = specific_imggrid_i;
		//cout << "Specific zsrc_i=" << specific_zsrc_i << " specific imggrid=" << specific_imggrid_i << endl;
	}
	bool changed_mask = false;
	int i,j,k,zsrc_i,imggrid_i;
	if (!plot_current_sb) {
		for (zsrc_i=zsrc_i_0, imggrid_i=primary_imggrid_i; zsrc_i < zsrc_i_f; zsrc_i++, imggrid_i++) {
			if (image_pixel_grids[imggrid_i] == NULL) {
				if ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source) or (source_fit_mode==Shapelet_Source)) { warn("No inversion has been performed to reconstruct source"); return false; }
				// if it hasn't been created yet, create now
				if (use_data) {
					image_pixel_grids[imggrid_i] = new ImagePixelGrid(this, source_fit_mode, ray_tracing_method, (*image_data), include_fgmask_in_inversion, band_number, zsrc_i, imggrid_i, assigned_mask[imggrid_i]);
					if (band_number < n_psf) psf_list[band_number]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);

				}
				else {
					image_pixel_grids[imggrid_i] = new ImagePixelGrid(this,source_fit_mode,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y,false,band_number,zsrc_i,imggrid_i);
					image_pixel_grids[imggrid_i]->setup_ray_tracing_arrays(true,verbose);
					if (band_number < n_psf) psf_list[band_number]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
				}
			} else {
				if ((image_data != NULL) and ((image_pixel_grids[imggrid_i]->x_N != image_data->npixels_x) or (image_pixel_grids[imggrid_i]->y_N != image_data->npixels_y))) {
					use_data = false;
					warn("img_npixels does not match number of pixels in data image");
					if (plot_residual==true) { warn("cannot plot residual image, pixel data image has been loaded or cannot be used"); return false; }
				}
			}
			image_pixel_grid = image_pixel_grids[imggrid_i];
			mask_num = assigned_mask[imggrid_i];
			if (use_data) {
				image_pixel_grid->setup_noise_map(this);
				if (show_all_pixels) { 
					image_pixel_grid->include_all_pixels();
					changed_mask = true;
				} else if (show_extended_mask) {
					image_pixel_grid->activate_extended_mask(); 
					changed_mask = true;
				//} else if (include_fgmask_in_inversion) {
					//cout << "ACTIVATING FOREGROUND" << endl;
					//image_pixel_grid->activate_foreground_mask(true,true); // show the foreground mask without the padding
					//changed_mask = true;
				} else if (show_foreground_mask) {
					image_pixel_grid->activate_foreground_mask(); 
					changed_mask = true;
				}
			}
			if (changed_mask) {
				image_pixel_grid->ray_trace_pixels();
			}
			else {
				image_pixel_grid->calculate_sourcepts_and_areas(); // unlike the above line, we don't reinitialize the arrays since mask hasn't changed
			}

			bool at_least_one_inverted_or_lensed_src_object = false; // this could be analytic source or pixellated source grid
			bool at_least_one_noninverted_foreground_src_included = false;
			bool at_least_one_lensed_nonshapelet_src = false;
			bool include_noninverted_src_as_foreground = false;
			for (k=0; k < n_sb; k++) {
				if (((sb_list[k]->is_lensed) and (sbprofile_imggrid_idx[k]==imggrid_i)) or (sb_list[k]->sbtype==MULTI_GAUSSIAN_EXPANSION) or (sb_list[k]->sbtype==SHAPELET)) {
					at_least_one_inverted_or_lensed_src_object = true; 
					if ((sb_list[k]->is_lensed) and (sb_list[k]->sbtype!=SHAPELET)) at_least_one_lensed_nonshapelet_src = true;
				}
				else if ((!sb_list[k]->is_lensed) and (zsrc_i==0)) {
					at_least_one_noninverted_foreground_src_included = true;
				}
			}
			if (source_fit_mode==Cartesian_Source) {
				at_least_one_inverted_or_lensed_src_object = true;
			} else if (source_fit_mode==Delaunay_Source) {
				at_least_one_inverted_or_lensed_src_object = true;
			} else if (source_fit_mode==Shapelet_Source) {
				if (at_least_one_lensed_nonshapelet_src) {
					at_least_one_noninverted_foreground_src_included = true; // non-shapelet analytic sources get included in the "foreground" surface brightness (kinda weird, I know)
					include_noninverted_src_as_foreground = true;
				}
			}
			if ((at_least_one_inverted_or_lensed_src_object) and (at_least_one_noninverted_foreground_src_included)) include_noninverted_src_as_foreground = true;
			if ((!at_least_one_inverted_or_lensed_src_object) and (at_least_one_noninverted_foreground_src_included)) plot_foreground_only = true;

			if (at_least_one_inverted_or_lensed_src_object) {
				bool fg_only = false;
				bool lensed_only = false;
				if (plot_foreground_only) {
					fg_only = true;
				} else if (omit_foreground) {
					lensed_only = true; 
				}
				image_pixel_grid->find_surface_brightness(fg_only,lensed_only,include_potential_perturbations and first_order_sb_correction,show_only_first_order_corrections,include_noninverted_src_as_foreground); // the last argument will cause it to omit lense nonshapelet sources here, since they'll be included in the foreground SB calculation
				vectorize_image_pixel_surface_brightness(imggrid_i,true); // note that in this case, the image pixel vector does NOT contain the foreground; the foreground PSF convolution was done separately above
				PSF_convolution_pixel_vector(imggrid_i,false,verbose,fft_convolution);
				store_image_pixel_surface_brightness(imggrid_i);
			} else {
				image_pixel_grids[imggrid_i]->set_zero_lensed_surface_brightness();
			}

			if ((!omit_foreground) and (at_least_one_noninverted_foreground_src_included)) {
				assign_foreground_mappings(imggrid_i,use_data);
				calculate_foreground_pixel_surface_brightness(imggrid_i,include_noninverted_src_as_foreground); // PSF convolution of foreground is done within this function
				store_foreground_pixel_surface_brightness(imggrid_i);
			} else {
				image_pixel_grids[imggrid_i]->set_zero_foreground_surface_brightness();
			}
				//image_pixel_grids[imggrid_i]->set_zero_foreground_surface_brightness();

			if (only_ptimgs) {
				// this is a hack so that it only shows the point images
				bool single_imggrid = true;
				if (n_extended_src_redshifts > 1) {
					if (specific_zsrc_i >= 0) {
						image_pixel_grid = image_pixel_grids[specific_imggrid_i];
					}
					else {
						single_imggrid = false;
						// If there are multiple extended source redshifts and no specific redshift index is given, combine the surface brightness from the separate image grids
						for (i=0; i < n_image_pixels_x; i++) {
							for (j=0; j < n_image_pixels_y; j++) {
								for (k=0; k < n_extended_src_redshifts; k++) {
									image_pixel_grids[k]->surface_brightness[i][j] = 0;
								}
							}
						}
					}
				} else {
					image_pixel_grid = image_pixel_grids[primary_imggrid_i];
				}
				if (single_imggrid) {
					for (i=0; i < n_image_pixels_x; i++) {
						for (j=0; j < n_image_pixels_y; j++) {
							image_pixel_grid->surface_brightness[i][j] = 0;
						}
					}
				}
			}

			if ((n_ptsrc > 0) and (!exclude_ptimgs)) {
				if (zsrc_i==0) {
					if (use_analytic_bestfit_src) set_analytic_sourcepts(verbose);
					if ((include_flux_chisq) and (analytic_source_flux)) set_analytic_srcflux(verbose);
					bool is_lensed;
					for (i=0; i < n_ptsrc; i++) {
						is_lensed = true;
						if (ptsrc_redshifts[ptsrc_redshift_idx[i]]==lens_redshift) is_lensed = false;
						if (!include_imgfluxes_in_inversion) image_pixel_grid->find_point_images(ptsrc_list[i]->pos[0],ptsrc_list[i]->pos[1],ptsrc_list[i]->images,false,is_lensed,verbose);
						image_pixel_grid->generate_and_add_point_images(ptsrc_list[i]->images, include_imgfluxes_in_inversion, ptsrc_list[i]->srcflux);
					}
				}
			}

			clear_pixel_matrices();
		}
	}

	if (n_extended_src_redshifts > 1) {
		if (specific_zsrc_i >= 0) {
			image_pixel_grid = image_pixel_grids[specific_imggrid_i];
		}
		else {
			// If there are multiple extended source redshifts and no specific redshift index is given, combine the surface brightness from the separate image grids
			if (!plot_current_sb) {
				// if we're showing the current SB, the surface brightness is assumed to already be consolidated into pixel grid 0
				for (i=0; i < n_image_pixels_x; i++) {
					for (j=0; j < n_image_pixels_y; j++) {
						for (k=1; k < n_extended_src_redshifts; k++) {
							image_pixel_grids[primary_imggrid_i]->surface_brightness[i][j] += image_pixel_grids[primary_imggrid_i+k]->surface_brightness[i][j];
							if ((image_pixel_grids[primary_imggrid_i+k]->pixel_in_mask[i][j]) and (!image_pixel_grids[primary_imggrid_i]->pixel_in_mask[i][j])) image_pixel_grids[primary_imggrid_i]->pixel_in_mask[i][j] = true;
						}
					}
				}
			}
			image_pixel_grid = image_pixel_grids[primary_imggrid_i]; // now we can just work with the first image_pixel_grid, which has the combined surface brightness
		}
	} else {
		image_pixel_grid = image_pixel_grids[primary_imggrid_i];
	}

	if ((background_pixel_noise != 0) or (use_noise_map)) {
		if (verbose) {
			double total_signal, noise;
			noise = background_pixel_noise;
			double signal_to_noise = image_pixel_grid->calculate_signal_to_noise(total_signal);
			if (mpi_id==0) {
				cout << "Signal-to-noise ratio = " << signal_to_noise << endl;
				cout << "Total integrated signal = " << total_signal << endl;
			}
		}
		if (simulate_pixel_noise) image_pixel_grid->add_pixel_noise();
	}

	double chisq_from_residuals;
	//if (output_fits==false) {
		if (mpi_id==0)  {
			chisq_from_residuals = image_pixel_grid->output_surface_brightness(xvals,yvals,zvals,plot_residual,normalize_residuals,show_noise_thresh,plot_log,(show_foreground_mask) or (include_fgmask_in_inversion));
		}
	//} else {
		//if (mpi_id==0) image_pixel_grid->output_fits_file(imagefile,plot_residual);
	//}

	if ((show_all_pixels) or (show_extended_mask)) include_fgmask_in_inversion = include_fgmask_in_inversion_orig;

	if (use_data) {
		//if (show_all_pixels) image_pixel_grid->include_all_pixels();
		for (zsrc_i=zsrc_i_0, imggrid_i=primary_imggrid_i; zsrc_i < zsrc_i_f; zsrc_i++, imggrid_i++) {
			if ((changed_mask) or ((n_extended_src_redshifts > 1) and (zsrc_i==0) and (specific_zsrc_i < 0))) // explanation for the latter condition: if all the lensed images were combined in one plot, then masks were combined image_pixel_grid[0], so we should restore the original mask
			{
				if (!image_pixel_grids[imggrid_i]->set_fit_window((*image_data),true,assigned_mask[imggrid_i],false,include_fgmask_in_inversion)) {
					warn("could not reset mask for imggrid index %i",imggrid_i);
					//delete image_pixel_grids; // so when you invert, it will load a new image grid based on the data
					//image_pixel_grids = NULL;
					return false;
				}
			}
		}
	}
	if ((mpi_id==0) and (plot_residual) and (!output_fits)) {
		if ((background_pixel_noise != 0) and (!use_noise_map)) chisq_from_residuals /= background_pixel_noise*background_pixel_noise; // if using noise map, 1/sig^2 factors are included in 'plot_surface_brightness' function above
		cout << "chi-square from residuals = " << chisq_from_residuals << endl;
	}
	//sbmax=-1e30;
	//sbmin=1e30;
	//// store max sb just in case we want to set the color bar scale using it
	//for (i=0; i < n_image_pixels_x; i++) {
		//for (j=0; j < n_image_pixels_y; j++) {
			//if (image_pixel_grids->surface_brightness[i][j] > sbmax) sbmax = image_pixel_grids->surface_brightness[i][j];
			//if (image_pixel_grids->surface_brightness[i][j] < sbmin) sbmin = image_pixel_grids->surface_brightness[i][j];
		//}
	//}
	if (offload_to_data) {
		if ((plot_residual) and (use_data)) {
			for (i=0; i < n_image_pixels_x; i++) {
				for (j=0; j < n_image_pixels_y; j++) {
					image_pixel_grid->surface_brightness[i][j] = image_data->surface_brightness[i][j] - image_pixel_grid->surface_brightness[i][j];
				}
			}
		}
		if (n_data_bands==0) {
			add_image_pixel_data();
		}
		image_data = imgdata_list[band_number]; // temporary until I sort out the band stuff
		image_data->load_from_image_grid(image_pixel_grid);
		if (specific_zsrc_i < 0) {
			specific_zsrc_i = 0;
			specific_imggrid_i = primary_imggrid_i;
		}
		image_pixel_grid->assign_mask_pointers((*image_data),specific_imggrid_i); // this is just to give image_pixel_grid the mask pointers from image_data
		if (n_data_bands==1) {
			// in case an image_pixel_grid has already been created (e.g. when making mock data)
			for (int i=0; i < n_extended_src_redshifts; i++) {
				image_pixel_grids[i]->set_image_pixel_data(imgdata_list[0], 0); // mask index = 0, make this better?
			}
		}

		// are the following lines really necessary??
		double xmin,xmax,ymin,ymax;
		int npx, npy;
		image_data->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
		grid_xlength = xmax - xmin;
		grid_ylength = ymax - ymin;
		set_gridcenter(0.5*(xmin+xmax),0.5*(ymin+ymax));
		image_data->get_npixels(n_image_pixels_x,n_image_pixels_y);

		// Make sure the grid size & center are fixed now
		if (autocenter) autocenter = false;
		if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
		if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
	}

	return true;
}

bool QLens::find_shapelet_scaling_parameters(const int i_shapelet, const int imggrid_i, const bool verbal)
{
	SB_Profile* shapelet = sb_list[i_shapelet];
	double sig,xc,yc,nsplit,sig_src,scaled_maxdist;
	image_pixel_grids[imggrid_i]->find_optimal_shapelet_scale(sig,xc,yc,nsplit,verbal,sig_src,scaled_maxdist);
	//if (auto_shapelet_scaling) shapelet->update_specific_parameter("sigma",sig);
	if (auto_shapelet_scaling) shapelet->update_scale_parameter(sig);
	if (auto_shapelet_center) {
		shapelet->update_specific_parameter("xc",xc);
		shapelet->update_specific_parameter("yc",yc);
	}
	if ((mpi_id==0) and (verbal)) {
		if (auto_shapelet_scaling) cout << "auto shapelet scaling: sig=" << sig << ", xc=" << xc << ", yc=" << yc << endl;
		else if (auto_shapelet_center) cout << "auto shapelet center: xc=" << xc << ", yc=" << yc << endl;
		double scale = shapelet->get_scale_parameter();
		int nn = get_shapelet_nn(imggrid_i);
		double minscale_shapelet = scale/sqrt(nn);
		double maxscale_shapelet = scale*sqrt(nn);
		//cout << "MAXSCALE = " << maxscale << ", MAXDIST = " << scaled_maxdist << endl;
		//if ((downsize_shapelets) and (maxscale_shapelet > scaled_maxdist))
		//if (maxscale_shapelet > scaled_maxdist) {
			//double scale = scaled_maxdist/sqrt(nn);
			//if (!shapelet->update_specific_parameter("sigma",scale)) {
				//if (mpi_id==0) warn("could not downsize shapelets to fit ray-traced mask; make sure shapelet is in pmode=0");
			//}
		//}
		if ((verbal) and (mpi_id==0)) {
			if (maxscale_shapelet < scaled_maxdist) {
			cerr << endl;
			warn("maximum scale of shapelets (%g) is smaller than estimated distance to the outermost ray-traced pixel (%g); this may affect chi-square\n********** to fix this, either reduce the size of the mask (if possible) or increase the shapelet order n_shapelet\n",maxscale_shapelet,scaled_maxdist);
			}
			if (scale > sig_src) {
				cerr << endl;
				warn("scale of shapelets (%g) is larger than dispersion of ray-traced surface brightness (%g); this could potentially affect quality of fit\n",scale,sig_src);
			}
		}
		cout << "shapelet_scale=" << scale << " shapelet_minscale=" << minscale_shapelet << " shapelet_maxscale=" << maxscale_shapelet << " (SCALE_MODE=" << shapelet_scale_mode << ")" << endl;
	}
	// Just in case any other sources are anchored to shapelet scale/center, update the anchored parameters now
	bool anchored_source = false;
	for (int i=0; i < n_sb; i++) {
		if (i != i_shapelet) {
			if (sb_list[i]->update_anchored_parameters_to_source(i_shapelet)==true) anchored_source = true;
		}
	}
	return anchored_source;
}

bool QLens::set_shapelet_imgpixel_nsplit(const int imggrid_i)
{
	if (imgdata_list == NULL) { warn("No image data have been loaded"); return false; }
	ImagePixelGrid *image_pixel_grid = image_pixel_grids[imggrid_i];
	if (image_pixel_grid == NULL) {
		image_pixel_grid = new ImagePixelGrid(this, source_fit_mode, ray_tracing_method, (*imgdata_list[0]), include_fgmask_in_inversion, 0, imggrid_i, imggrid_i, assigned_mask[imggrid_i], true);
		if (n_psf > 0) psf_list[0]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
	}
	image_pixel_grid->redo_lensing_calculations();
	double sig,xc,yc,nsplit,sig_src,maxdist;
	image_pixel_grid->find_optimal_shapelet_scale(sig,xc,yc,nsplit,false,sig_src,maxdist);
	default_imgpixel_nsplit = (((int) nsplit)+3);
	return true;
}

int QLens::get_shapelet_nn(const int imggrid_i)
{
	SB_Profile* shapelet = NULL;
	for (int i=0; i < n_sb; i++) {
		if ((sb_list[i]->sbtype==SHAPELET) and (sbprofile_imggrid_idx[i]==imggrid_i)) {
			shapelet = sb_list[i];
			break; // currently only one shapelet source supported
		}
	}
	if (shapelet==NULL) die("no shapelet object found");
	return *(shapelet->indxptr);
}

bool QLens::load_pixel_grid_from_data(const int band_number)
{
	bool loaded_new_grid = false;
	if (n_data_bands==0) { warn("No image data have been loaded"); return false; }
	if (band_number >= n_data_bands) { warn("Specified band number does not exist"); return false; }
	//if ((n_pixellated_src == 0) and ((n_image_prior) or (n_ptsrc > 0))) {
		//if (mpi_id==0) cout << "NOTE: automatically generating pixellated source object at zsrc=" << source_redshift << endl;
		//add_pixellated_source(source_redshift); // Note, even if in sbprofile or shapelet mode, we'll still need a srcgrid object if we're modeling point images with PSF's
	//}
	if ((n_pixellated_src==0) and ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source))) add_pixellated_source(source_redshift,band_number);
	else if (n_extended_src_redshifts == 0) {
		add_new_extended_src_redshift(source_redshift,-1,false);
	}

	int zsrc_i, imggrid_i;
	if (band_number >= n_model_bands) { warn("Specified model band number should have been created by now"); return false; }
	for (zsrc_i=0, imggrid_i=band_number*n_extended_src_redshifts; zsrc_i < n_extended_src_redshifts; zsrc_i++, imggrid_i++) {
		if (image_pixel_grids[imggrid_i] != NULL) {
			delete image_pixel_grids[imggrid_i];
		}
		image_pixel_grids[imggrid_i] = new ImagePixelGrid(this, source_fit_mode, ray_tracing_method, (*imgdata_list[band_number]), include_fgmask_in_inversion, band_number, zsrc_i, imggrid_i, assigned_mask[imggrid_i]);
		loaded_new_grid = true;
		if (band_number < n_psf) psf_list[band_number]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
	}
	return loaded_new_grid;
}

void QLens::plot_image_pixel_grid(const int band_number, const int zsrc_i)
{
	if (n_data_bands==0) { warn("No image data have been loaded"); return; }
	if (image_pixel_grids == NULL) { warn("No extended sources have been created"); return; }
	//if ((n_pixellated_src == 0) and ((n_image_prior) or (n_ptsrc > 0))) {
		//if (mpi_id==0) cout << "NOTE: automatically generating pixellated source object at zsrc=" << source_redshift << endl;
		//add_pixellated_source(source_redshift); // Note, even if in sbprofile or shapelet mode, we'll still need a srcgrid object if we're modeling point images with PSF's
	//}
	if ((n_pixellated_src==0) and ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source))) add_pixellated_source(source_redshift,band_number);
	else if (n_extended_src_redshifts == 0) {
		add_new_extended_src_redshift(source_redshift,-1,false);
	}

	int imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;
	if (image_pixel_grids[imggrid_i] == NULL) {
		image_pixel_grids[imggrid_i] = new ImagePixelGrid(this, source_fit_mode,ray_tracing_method, (*imgdata_list[band_number]), include_fgmask_in_inversion, zsrc_i, imggrid_i, assigned_mask[imggrid_i]);
		if (band_number < n_psf) psf_list[band_number]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
	}
	image_pixel_grids[imggrid_i]->redo_lensing_calculations();
	image_pixel_grids[imggrid_i]->plot_grid("map",false);
}

double QLens::invert_surface_brightness_map_from_data(double &chisq0, const bool verbal, const bool zero_verbal)
{
	if (n_data_bands==0) { warn("No image data have been loaded"); return -1e30; }
	if (n_model_bands==0) { warn("No model bands have been created"); return -1e30; }
	//if ((n_pixellated_src == 0) and ((n_image_prior) or (n_ptsrc > 0))) {
		//if ((mpi_id==0) and (verbal)) cout << "NOTE: automatically generating pixellated source object at zsrc=" << source_redshift << endl;
		//add_pixellated_source(source_redshift); // Note, even if in sbprofile or shapelet mode, we'll still need a srcgrid object if we're modeling point images with PSF's
	//}

	if ((n_extended_src_redshifts == 0) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Cartesian_Source)) {
		add_new_extended_src_redshift(source_redshift,-1,false);
	}

	int band_number, zsrc_i, imggrid_i=0;
	for (band_number=0; band_number < n_model_bands; band_number++) {
		if ((n_pixellated_src==0) and ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source))) add_pixellated_source(source_redshift,band_number);
		for (zsrc_i=0; zsrc_i < n_extended_src_redshifts; zsrc_i++) {
			if (image_pixel_grids[imggrid_i] == NULL) {
				image_pixel_grids[imggrid_i] = new ImagePixelGrid(this, source_fit_mode, ray_tracing_method, (*imgdata_list[band_number]), include_fgmask_in_inversion, band_number, zsrc_i, imggrid_i, assigned_mask[zsrc_i], true, verbal);
				if (band_number < n_psf) psf_list[band_number]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
			}
			imggrid_i++;
		}
	}
	double chisq=0,chisq00;
	chisq0=0;
#ifdef USE_OPENMP
	double inversion_wtime0, inversion_wtime;
	if (show_wtime) {
		inversion_wtime0 = omp_get_wtime();
	}
#endif
	for (int i=0; i < n_ranchisq; i++) {
		chisq += pixel_log_evidence_times_two(chisq00,verbal,i);
		chisq0 += chisq00;
	}
	chisq /= n_ranchisq;
	chisq0 /= n_ranchisq;
	if ((mpi_id==0) and (!verbal) and (!zero_verbal)) cout << "chisq0=" << chisq0 << ", chisq_pix=" << chisq << endl; // we output here if verbal==false because we still want to see the chisq values at the end (unless zero_verbal is set to true)

#ifdef USE_OPENMP
	if (show_wtime) {
		inversion_wtime = omp_get_wtime() - inversion_wtime0;
		if (mpi_id==0) cout << "Total wall time for lensing reconstruction: " << inversion_wtime << endl;
	}
#endif

	//chisq = pixel_log_evidence_times_two(chisq0,verbal);
	if (chisq == 2e30) {
		// in this case, the inversion didn't work, so we delete the image pixel grids so there is no confusion if the user tries to plot the lensed images
		for (int imggrid_i=0; imggrid_i < n_image_pixel_grids; imggrid_i++) {
			delete image_pixel_grids[imggrid_i];
			image_pixel_grids[imggrid_i] = NULL;
		}
	}

	return chisq;
}

double QLens::pixel_log_evidence_times_two(double &chisq0, const bool verbal, const int ranchisq_i)
{
	// This function is too long, and should be further broken into a bunch of smaller functions.
	if (n_data_bands==0) { warn("No image data have been loaded"); return -1e30; }
	if (n_model_bands < n_data_bands) { warn("Numebr of model bands is not large enough to accommodate number of data bands"); }

	if ((n_extended_src_redshifts == 0) and (source_fit_mode != Delaunay_Source) and (source_fit_mode != Cartesian_Source)) {
		add_new_extended_src_redshift(source_redshift,-1,false);
	}

	if ((n_pixellated_src==0) and ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source))) add_pixellated_source(source_redshift,0);
	else if (n_extended_src_redshifts == 0) {
		add_new_extended_src_redshift(source_redshift,-1,false);
	}
	if (n_model_bands != n_data_bands) die("number of model bands does not equal number of data bands");

	for (int band_number=0; band_number < n_model_bands; band_number++) {
		if ((n_pixellated_src==0) and ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source))) add_pixellated_source(source_redshift,band_number);
	}

	if (image_pixel_grids == NULL) { warn("No image surface brightness grid has been generated"); return -1e30; }
	int imggrid_i, src_i;
	for (imggrid_i=0; imggrid_i < n_image_pixel_grids; imggrid_i++) {
		if (image_pixel_grids[imggrid_i] == NULL) { warn("No image surface brightness grid for imggrid_i=%i has been generated",imggrid_i); return -1e30; }
	}
	if ((source_fit_mode == Parameterized_Source) and (n_sb==0)) {
		warn("no parameterized sources have been defined; cannot evaluate chi-square");
		chisq0=-1e30; return -1e30;
	}
	int *src_i_list = new int[n_image_pixel_grids];
	for (imggrid_i=0; imggrid_i < n_image_pixel_grids; imggrid_i++) {
		src_i_list[imggrid_i] = -1;
		for (int i=0; i < n_pixellated_src; i++) {
			if ((pixellated_src_band[i]==image_pixel_grids[imggrid_i]->band_number) and (pixellated_src_redshift_idx[i]==image_pixel_grids[imggrid_i]->src_redshift_index)) {
				src_i_list[imggrid_i] = i;
				break;
			}
		}
		//if (src_i_list[imggrid_i]==-1) die("src_i did not get defined for imggrid=%i, band_i=%i, zsrc_i=%i",imggrid_i,image_pixel_grids[imggrid_i]->band_number,image_pixel_grids[imggrid_i]->src_redshift_index);
	}

	if (((source_fit_mode == Cartesian_Source) or (source_fit_mode == Delaunay_Source)) and (n_pixellated_src > 1)) {
		set_n_imggrids_to_include_in_inversion();
	}

	if ((mpi_id==0) and (verbal)) cout << "Number of data pixels in mask 0 : " << imgdata_list[0]->n_mask_pixels[0] << endl;
	double tot_wtime0, tot_wtime;
#ifdef USE_OPENMP
	if (show_wtime) {
		tot_wtime0 = omp_get_wtime();
	}
#endif

#ifdef USE_OPENMP
	double fspline_wtime0;
	if (show_wtime) {
		fspline_wtime0 = omp_get_wtime();
	}
#endif

	bool splined_fourier_integrals = false;
	for (int i=0; i < nlens; i++) {
		if (lens_list[i]->n_fourier_modes > 0) {
			lens_list[i]->spline_fourier_mode_integrals(0.01*imgdata_list[0]->emask_rmax,imgdata_list[0]->emask_rmax);
			splined_fourier_integrals = true;
		}
	}
#ifdef USE_OPENMP
	if ((show_wtime) and (splined_fourier_integrals)) {
		double fspline_wtime = omp_get_wtime() - fspline_wtime0;
		if (mpi_id==0) cout << "Wall time for splining Fourier integrals: " << fspline_wtime << endl;
	}
#endif

	if ((redo_lensing_calculations_before_inversion) and (ranchisq_i==0)) {
		for (imggrid_i=0; imggrid_i < n_image_pixel_grids; imggrid_i++) {
			image_pixel_grids[imggrid_i]->redo_lensing_calculations(verbal);
		}
	}
	for (imggrid_i=0; imggrid_i < n_image_pixel_grids; imggrid_i++) {
		if ((n_extended_src_redshifts > 1) and (imggrid_i==0)) {
			update_lens_centers_from_pixsrc_coords();
		}
	}

	int i,j,zsrc_i;
	double logev_times_two = 0;
	chisq0 = 0;
	double logev_times_two_band;
	double loglike_reg;
	double regterms;
	bool skip_inversion = false;

	if ((n_image_prior) or (n_ptsrc > 0)) {
		setup_auxiliary_sourcegrids_and_point_imgs(src_i_list,verbal);
	}

	bool include_foreground_sbmask, include_foreground_sb, at_least_one_noninverted_foreground_src, at_least_one_lensed_src, at_least_one_lensed_nonshapelet_src, at_least_one_shapelet_src, at_least_one_mge_src; 

	ImageData *image_data;
	
	bool sb_outside_window, sb_outside_window_allbands = false;
	for (int band_number = 0; band_number < n_data_bands; band_number++) {
		image_data = imgdata_list[band_number];
		loglike_reg = 0;
		logev_times_two_band = 0;
		skip_inversion = false;
		// the foreground surface brightness includes foreground, but can also include additional (analytic) lensed sources if in pixel mode
		include_foreground_sbmask = false;
		include_foreground_sb = false;
		at_least_one_noninverted_foreground_src = false;
		at_least_one_lensed_src = false;
		at_least_one_lensed_nonshapelet_src = false;
		at_least_one_shapelet_src = false;
		at_least_one_mge_src = false;
		for (int k=0; k < n_sb; k++) {
			if ((!sb_list[k]->is_lensed) and (sb_list[k]->sbtype != MULTI_GAUSSIAN_EXPANSION)) {
				at_least_one_noninverted_foreground_src = true;
			} else {
				at_least_one_lensed_src = true;
				if (sb_list[k]->sbtype!=SHAPELET) at_least_one_lensed_nonshapelet_src = true;
			}
			if (sb_list[k]->sbtype==SHAPELET) at_least_one_shapelet_src = true;
			else if (sb_list[k]->sbtype==MULTI_GAUSSIAN_EXPANSION) at_least_one_mge_src = true;
		}
		if (at_least_one_noninverted_foreground_src) include_foreground_sb = true;
		if ((!ignore_foreground_in_chisq) and (include_fgmask_in_inversion)) { include_foreground_sbmask = true; } 
		else if (((at_least_one_lensed_nonshapelet_src) or ((source_fit_mode != Shapelet_Source) and (at_least_one_lensed_src)))) include_foreground_sb = true; // if doing a pixel inversion, parameterized sources can still be added to the SB by using the "foreground" sb array...it's a bit confusing and convoluted, however
		if ((source_fit_mode==Shapelet_Source) and (!at_least_one_shapelet_src) and (!at_least_one_mge_src) and ((n_ptsrc==0) or ((!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)))) {
			if (verbal) warn("cannot perform inversion because no shapelet/MGE objects and no point sources with invertible amplitudes are defined");
			chisq0=-1e30; return -1e30;
		}
		if (n_ptsrc > 0) {
			if ((source_fit_mode==Shapelet_Source) and (!at_least_one_shapelet_src) and (include_imgfluxes_in_inversion)) {
				// Make sure there is at least one image produced, otherwise there will be no amplitudes to invert! (since no shapelets have been created)
				int nimgs_tot = 0;
				for (i=0; i < n_ptsrc; i++) nimgs_tot += ptsrc_list[i]->images.size(); // in this case, source amplitudes include point image amplitudes as well as pixel values
				if (nimgs_tot==0) {
					if (verbal) warn("cannot perform inversion because no shapelet/MGE objects and no images with invertible amplitudes have been produced");
					skip_inversion = true;
				}
			}
		}

		if (source_fit_mode == Cartesian_Source) {
			for (zsrc_i=0, imggrid_i=band_number*n_extended_src_redshifts; zsrc_i < n_extended_src_redshifts; zsrc_i++, imggrid_i++) {
				int src_i = src_i_list[imggrid_i];
				if (src_i < 0) {
					// no cartesian source at this redshift, so assume there is an analytic source and find/store the corresponding surface brightness
					image_pixel_grids[imggrid_i]->find_surface_brightness();
					vectorize_image_pixel_surface_brightness(imggrid_i,true);
					PSF_convolution_pixel_vector(imggrid_i,false,verbal,fft_convolution);
					store_image_pixel_surface_brightness(imggrid_i);
				} else {
					if (image_pixel_grids[imggrid_i]->n_pixsrc_to_include_in_Lmatrix==0) continue; // that means this pixellated source will be included with the inversion handled from another ImagePixelGrid (with a different imggrid_i index)
					int n_expected_imgpixels;
					if (!setup_cartesian_sourcegrid(imggrid_i,src_i,n_expected_imgpixels,verbal)) return 2e30;

#ifdef USE_OPENMP
					double srcgrid_wtime0, srcgrid_wtime;
					if (show_wtime) {
						srcgrid_wtime0 = omp_get_wtime();
					}
#endif
					cartesian_srcgrids[src_i]->create_pixel_grid(this,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y);
					cartesian_srcgrids[src_i]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);

					if (adaptive_subgrid) {
						cartesian_srcgrids[src_i]->adaptive_subgrid();
						if ((mpi_id==0) and (verbal)) {
							cout << "# of source pixels after subgridding: " << cartesian_srcgrids[src_i]->number_of_pixels;
							if (auto_srcgrid_npixels) {
								double pix_frac = ((double) cartesian_srcgrids[src_i]->number_of_pixels) / n_expected_imgpixels;
								cout << ", f=" << pix_frac;
							}
							cout << endl;
						}
					} else {
						if (n_image_prior) cartesian_srcgrids[src_i]->calculate_pixel_magnifications();
					}

#ifdef USE_OPENMP
					if (show_wtime) {
						srcgrid_wtime = omp_get_wtime() - srcgrid_wtime0;
						if (mpi_id==0) cout << "Wall time for creating source pixel grid: " << srcgrid_wtime << endl;
					}
#endif

					if (!generate_and_invert_lensing_matrix_cartesian(imggrid_i,src_i,tot_wtime,tot_wtime0,verbal)) return 2e30;

					if (inversion_method==DENSE) calculate_image_pixel_surface_brightness_dense();
					else calculate_image_pixel_surface_brightness();
					store_image_pixel_surface_brightness(imggrid_i);
					if ((n_ptsrc > 0) and (!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) {
						image_pixel_grids[imggrid_i]->add_point_images(point_image_surface_brightness,image_pixel_grids[imggrid_i]->n_active_pixels);
					}

					if (regularization_method != None) add_regularization_prior_terms_to_logev(band_number,zsrc_i,logev_times_two_band,loglike_reg,regterms,false,verbal);

#ifdef USE_OPENMP
					if (show_wtime) {
						tot_wtime = omp_get_wtime() - tot_wtime0;
						if (mpi_id==0) cout << "Total wall time for F-matrix construction + inversion: " << tot_wtime << endl;
					}
#endif
					clear_sparse_lensing_matrices();
					clear_pixel_matrices();
				}
			}
		} else if (source_fit_mode == Delaunay_Source) {
			if ((mpi_id==0) and (verbal)) cout << "Assigning foreground pixel mappings..." << endl;
			for (zsrc_i=0, imggrid_i=band_number*n_extended_src_redshifts; zsrc_i < n_extended_src_redshifts; zsrc_i++, imggrid_i++) {
				//cout << "BAND_I=" << band_number << ", ZSRC_I=" << zsrc_i << " imggrid_i=" << imggrid_i << endl;
				src_i = src_i_list[imggrid_i];
				if (src_i >= 0) {
					if (use_dist_weighted_srcpixel_clustering) calculate_subpixel_distweights(imggrid_i);
					else if (use_saved_sbweights) load_pixel_sbweights(imggrid_i);
					if (nlens > 0) {
#ifdef USE_OPENMP
						double srcgrid_wtime0, srcgrid_wtime;
						if (show_wtime) {
							srcgrid_wtime0 = omp_get_wtime();
						}
#endif
						bool use_weighted_clustering = ((use_dist_weighted_srcpixel_clustering) or ((use_lum_weighted_srcpixel_clustering) and (use_saved_sbweights))) ? true : false;

						create_sourcegrid_from_imggrid_delaunay(use_weighted_clustering,band_number,zsrc_i,verbal);
						image_pixel_grids[imggrid_i]->set_delaunay_srcgrid(delaunay_srcgrids[src_i]);
						delaunay_srcgrids[src_i]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
#ifdef USE_OPENMP
						if (show_wtime) {
							srcgrid_wtime = omp_get_wtime() - srcgrid_wtime0;
							if (mpi_id==0) cout << "wall time for Delaunay grid creation: " << srcgrid_wtime << endl;
							srcgrid_wtime0=omp_get_wtime();
						}
#endif
						if ((include_potential_perturbations) and (imggrid_i==0)) {
							if (create_lensgrid_cartesian(band_number,zsrc_i,0,verbal)==false) return 2e30;
							lensgrids[0]->include_in_lensing_calculations = false;
#ifdef USE_OPENMP
							if (show_wtime) {
								srcgrid_wtime = omp_get_wtime() - srcgrid_wtime0;
								if (mpi_id==0) cout << "wall time for pixellated potential grid creation: " << srcgrid_wtime << endl;
							}
#endif
						}
					}
				}
			}
			for (zsrc_i=0, imggrid_i=band_number*n_extended_src_redshifts; zsrc_i < n_extended_src_redshifts; zsrc_i++, imggrid_i++) {
				//cout << "BAND_I=" << band_number << ", ZSRC_I=" << zsrc_i << " imggrid_i=" << imggrid_i << endl;
				src_i = src_i_list[imggrid_i];
				if (src_i < 0) {
					// no Delaunay source at this redshift, so assume there is an analytic source and find/store the corresponding surface brightness
					//image_pixel_grids[imggrid_i]->find_surface_brightness();
					//vectorize_image_pixel_surface_brightness(imggrid_i,true);
					//PSF_convolution_pixel_vector(imggrid_i,false,verbal,fft_convolution);
					//store_image_pixel_surface_brightness(imggrid_i);
				} else {
					assign_foreground_mappings(imggrid_i);
					if (!ignore_foreground_in_chisq) {
						calculate_foreground_pixel_surface_brightness(imggrid_i,true);
						store_foreground_pixel_surface_brightness(imggrid_i);
					}
					if (image_pixel_grids[imggrid_i]->n_pixsrc_to_include_in_Lmatrix==0) continue; // that means this pixellated source will be included with the inversion handled from another ImagePixelGrid (with a different imggrid_i index)

					if (generate_and_invert_lensing_matrix_delaunay(imggrid_i,src_i,false,include_potential_perturbations,tot_wtime,tot_wtime0,verbal)==false) return 2e30;
					if (include_potential_perturbations) {
						clear_sparse_lensing_matrices();
						clear_pixel_matrices(); 
						for (int it=0; it < potential_correction_iterations; it++) {
							if (generate_and_invert_lensing_matrix_delaunay(imggrid_i,src_i,true,true,tot_wtime,tot_wtime0,verbal)==false) return 2e30;
							clear_sparse_lensing_matrices();
							clear_pixel_matrices(); 
						}
						if (generate_and_invert_lensing_matrix_delaunay(imggrid_i,src_i,true,adopt_final_sbgrad,tot_wtime,tot_wtime0,verbal)==false) return 2e30; // choose false for 4th arg if you want to reproduce the same first-order SB corrections (meaning it uses SB_grad from previous iteration in first order term)

					}

					if ((!use_lum_weighted_srcpixel_clustering) and (!use_saved_sbweights) and (save_sbweights_during_inversion)) calculate_subpixel_sbweights(imggrid_i,true,verbal);

					if (inversion_method==DENSE) calculate_image_pixel_surface_brightness_dense();
					else calculate_image_pixel_surface_brightness();
					store_image_pixel_surface_brightness(imggrid_i);
					if ((n_ptsrc > 0) and (!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) {
						image_pixel_grids[imggrid_i]->add_point_images(point_image_surface_brightness,image_pixel_grids[imggrid_i]->n_active_pixels);
					}
					if (regularization_method != None) add_regularization_prior_terms_to_logev(band_number,zsrc_i,logev_times_two_band,loglike_reg,regterms,include_potential_perturbations,verbal);
#ifdef USE_OPENMP
					if (show_wtime) {
						tot_wtime = omp_get_wtime() - tot_wtime0;
						if (mpi_id==0) cout << "Total wall time for F-matrix construction + inversion: " << tot_wtime << endl;
					}
#endif
					clear_sparse_lensing_matrices();
					clear_pixel_matrices();
				}
				if ((include_potential_perturbations) and (!first_order_sb_correction) and (lensgrids != NULL) and (imggrid_i==0)) {
					lensgrids[0]->include_in_lensing_calculations = true;
				}
			}
			//cout << "DONE WITH THE INVERSION PART" << endl;
		} else if (source_fit_mode == Parameterized_Source) {
			for (zsrc_i=0, imggrid_i=band_number*n_extended_src_redshifts; zsrc_i < n_extended_src_redshifts; zsrc_i++, imggrid_i++) {
				if (at_least_one_lensed_src) {
					image_pixel_grids[imggrid_i]->find_surface_brightness(false,true);
					vectorize_image_pixel_surface_brightness(imggrid_i,true);
					PSF_convolution_pixel_vector(imggrid_i,false,verbal,fft_convolution);
					store_image_pixel_surface_brightness(imggrid_i);
				} else {
					image_pixel_grids[imggrid_i]->set_zero_lensed_surface_brightness();
				}
				if (at_least_one_noninverted_foreground_src) {
					assign_foreground_mappings(imggrid_i,true);
					if (!ignore_foreground_in_chisq) {
						calculate_foreground_pixel_surface_brightness(imggrid_i,false);
						store_foreground_pixel_surface_brightness(imggrid_i);
					}
				} else {
					image_pixel_grids[imggrid_i]->set_zero_foreground_surface_brightness();
				}
			}
			if (save_sbweights_during_inversion) calculate_subpixel_sbweights(true,verbal); // these are sb-weights to be used later in Delaunay mode for luminosity weighting
			if (n_ptsrc > 0) {
				if (point_image_surface_brightness != NULL) delete[] point_image_surface_brightness;
				point_image_surface_brightness = new double[image_npixels];
				if ((mpi_id==0) and (verbal)) cout << "Generating point images..." << endl;
				for (i=0; i < n_ptsrc; i++) {
					image_pixel_grids[0]->generate_and_add_point_images(ptsrc_list[i]->images, false, ptsrc_list[i]->srcflux);
				}
			}
		} else {
			// Shapelet_Source mode
			for (zsrc_i=0, imggrid_i=band_number*n_extended_src_redshifts; zsrc_i < n_extended_src_redshifts; zsrc_i++, imggrid_i++) {
				if ((mpi_id==0) and (verbal)) cout << "Assigning foreground pixel mappings... (MAYBE REMOVE THIS FROM CHISQ AND DO AHEAD OF TIME?)\n";
				assign_foreground_mappings(imggrid_i); // note, this is done even if there is no foreground "source" object; foreground sb vector just has all elements equal zero
				if ((at_least_one_noninverted_foreground_src) or (at_least_one_lensed_nonshapelet_src)) {
					if (!ignore_foreground_in_chisq) {
						calculate_foreground_pixel_surface_brightness(imggrid_i);
						store_foreground_pixel_surface_brightness(imggrid_i);
					}
				} else {
					image_pixel_grids[imggrid_i]->set_zero_foreground_surface_brightness();
				}
				int i_shapelet = -1;
				if ((n_sb > 0) and ((auto_shapelet_scaling) or (auto_shapelet_center))) {
					for (int i=0; i < n_sb; i++) {
						if ((sb_list[i]->sbtype==SHAPELET) and (sbprofile_imggrid_idx[i]==imggrid_i)) {
							i_shapelet = i;
							break; // currently only one shapelet source supported
						}
					}
					if (!ignore_foreground_in_chisq) {
						if ((i_shapelet >= 0) and (find_shapelet_scaling_parameters(i_shapelet,imggrid_i,verbal)==true)) {
							// if returned true, then there is a source that is anchored to the shapelet params, so we must rebuild the foreground/sbprofile surface brightness now
							if ((mpi_id==0) and (verbal)) cout << "Recalculating foreground/sbprofile surface brightness (two more iterations)..." << endl;
							calculate_foreground_pixel_surface_brightness(imggrid_i);
							store_foreground_pixel_surface_brightness(imggrid_i);
							// one more iteration for good measure
							find_shapelet_scaling_parameters(i_shapelet,imggrid_i,verbal);
							calculate_foreground_pixel_surface_brightness(imggrid_i);
							store_foreground_pixel_surface_brightness(imggrid_i);
						}
					}
				}
				if (!skip_inversion) {
					initialize_pixel_matrices_shapelets(imggrid_i,verbal);
					if ((mpi_id==0) and (verbal)) {
						cout << "Number of active image pixels: " << image_npixels << endl;
						cout << "Number of shapelet amplitudes: " << source_npixels << endl;
						if (n_amps > source_npixels) cout << "Number of total amplitudes: " << n_amps << endl;
					}

					image_pixel_grids[imggrid_i]->set_surface_brightness_vector_to_data(); // note that image_pixel_grids[0] just has the data pixel values stored in it
					PSF_convolution_Lmatrix_dense(imggrid_i,verbal);
					if (imggrid_i==0) {
						// currently only allowing point sources with first image grid...will extend later
						if ((n_ptsrc > 0) and (!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) {
							if ((mpi_id==0) and (verbal)) cout << "Generating point images..." << endl;
							for (i=0; i < n_ptsrc; i++) {
								image_pixel_grids[imggrid_i]->generate_point_images(ptsrc_list[i]->images, point_image_surface_brightness, false, ptsrc_list[i]->srcflux);
							}
						}
					}

					if ((regularization_method != None) and (i_shapelet >= 0)) create_regularization_matrix_shapelet(imggrid_i);
					if ((mpi_id==0) and (verbal)) cout << "Creating lensing matrices...\n" << flush;
					create_lensing_matrices_from_Lmatrix_dense(imggrid_i,false,verbal);

#ifdef USE_OPENMP
					if (show_wtime) {
						tot_wtime = omp_get_wtime() - tot_wtime0;
						if (mpi_id==0) cout << "Total wall time before F-matrix inversion: " << tot_wtime << endl;
					}
#endif
					if ((mpi_id==0) and (verbal)) cout << "Inverting lens mapping...\n" << flush;
					if ((optimize_regparam) and (regularization_method != None) and (source_npixels > 0)) optimize_regularization_parameter(imggrid_i,true,verbal);
					if ((!optimize_regparam) or (source_npixels==0) or (regularization_method==None)) invert_lens_mapping_dense(imggrid_i,verbal); 
					if (save_sbweights_during_inversion) calculate_subpixel_sbweights(imggrid_i,true,verbal); // these are sb-weights to be used later in Delaunay mode for luminosity weighting
					calculate_image_pixel_surface_brightness_dense();
					store_image_pixel_surface_brightness(imggrid_i);
					if ((n_ptsrc > 0) and (!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) {
						image_pixel_grids[imggrid_i]->add_point_images(point_image_surface_brightness,image_pixel_grids[imggrid_i]->n_active_pixels);
					}

					if (regularization_method != None) add_regularization_prior_terms_to_logev(band_number,zsrc_i,logev_times_two_band,loglike_reg,regterms,false,verbal);
					clear_pixel_matrices();
				} else {
					image_pixel_grids[imggrid_i]->set_zero_lensed_surface_brightness();
				}
			}

#ifdef USE_OPENMP
			if (show_wtime) {
				tot_wtime = omp_get_wtime() - tot_wtime0;
				if (mpi_id==0) cout << "Total wall time for F-matrix construction + inversion: " << tot_wtime << endl;
			}
#endif
		}

		//if (n_extended_src_redshifts > 1) {
			//// If there are multiple extended source redshifts, combine the surface brightness from the separate image grids so it's all in the first image pixel grid
			//for (i=0; i < n_image_pixels_x; i++) {
				//for (j=0; j < n_image_pixels_y; j++) {
					//for (k=1; k < n_extended_src_redshifts; k++) {
						//image_pixel_grids[0]->surface_brightness[i][j] += image_pixel_grids[k]->surface_brightness[i][j];
					//}
				//}
			//}
		//}

		double cov_inverse;
		if (!use_noise_map) {
			if (background_pixel_noise==0) cov_inverse = 1; // background noise hasn't been specified, so just pick an arbitrary value
			else cov_inverse = 1.0/SQR(background_pixel_noise);
		}
		int count, foreground_count;
		int n_data_pixels;
		double chisq0_imggrid;
		double pixel_avg_n_images;
		int chisq0_band = 0;
		foreground_count = 0;
		count = 0;

		// Next we evaluate the chi-square
		for (zsrc_i=0, imggrid_i=band_number*n_extended_src_redshifts; zsrc_i < n_extended_src_redshifts; zsrc_i++, imggrid_i++) {
			ImagePixelGrid* image_pixel_grid = image_pixel_grids[imggrid_i]; 
			if (image_pixel_grids[imggrid_i]->n_pixsrc_to_include_in_Lmatrix==0) continue;
			n_data_pixels = 0;
			chisq0_imggrid = 0;
			for (i=0; i < image_data->npixels_x; i++) {
				for (j=0; j < image_data->npixels_y; j++) {
					if (((!include_fgmask_in_inversion) and (image_pixel_grid->pixel_in_mask[i][j])) or ((include_foreground_sbmask) and (image_data->foreground_mask_data[i][j])))
					{
						n_data_pixels++;
						if (use_noise_map) cov_inverse = image_data->covinv_map[i][j];
						if ((image_pixel_grid->pixel_in_mask[i][j]) and (image_pixel_grid->maps_to_source_pixel[i][j])) {
							if (include_foreground_sb) {
								chisq0_imggrid += SQR(image_pixel_grid->surface_brightness[i][j] + image_pixel_grid->foreground_surface_brightness[i][j] - image_data->surface_brightness[i][j])*cov_inverse; // generalize to full cov_inverse matrix later
								foreground_count++;
							} else {
								chisq0_imggrid += SQR(image_pixel_grid->surface_brightness[i][j] - image_data->surface_brightness[i][j])*cov_inverse; // generalize to full cov_inverse matrix later
							}
							count++;
						} else {
							// NOTE that if a pixel is not in the foreground mask, the foreground_surface_brightness has already been set to zero for that pixel
							if (include_foreground_sb) {
								chisq0_imggrid += SQR(image_pixel_grid->foreground_surface_brightness[i][j] - image_data->surface_brightness[i][j])*cov_inverse;
								foreground_count++;
							}
							else if (image_pixel_grid->pixel_in_mask[i][j]) chisq0_imggrid += SQR(image_data->surface_brightness[i][j])*cov_inverse; // if we're not modeling foreground, then only add to chi-square if it's inside the primary mask
						}
					}
				}
			}
			chisq0_band += chisq0_imggrid;
			logev_times_two_band += chisq0_imggrid; // logev_times_two_band includes the prior terms

			if (group_id==0) {
				if (logfile.is_open()) {
					logfile << "it=" << chisq_it << ": ";
					if (n_extended_src_redshifts > 1) logfile << "imggrid_i=" << imggrid_i;
					logfile << " chisq0_band=" << chisq0_imggrid << " chisq0_per_pixel=" << chisq0_imggrid/n_data_pixels << " (ntot_pixels=" << n_data_pixels << ") regterms=" << regterms << endl;
				}
			}
			if ((mpi_id==0) and (verbal)) {
				if (n_extended_src_redshifts > 1) cout << "imggrid_i=" << imggrid_i << ": ";
				cout << "chisq0=" << chisq0_imggrid << " chisq0_per_pixel=" << chisq0_imggrid/n_data_pixels << " (ntot_pixels=" << n_data_pixels << ")";
				if (regularization_method != None) cout << " regterms=" << loglike_reg;
				cout << endl;
			}
			if (include_noise_term_in_loglike) {
				// Need to improve this when using noise map!
				if (use_noise_map) {
					bool include_pixel;
					for (int img_index=0; img_index < image_npixels; img_index++) {
						include_pixel = true;
						i = image_pixel_grid->active_image_pixel_i[img_index];
						j = image_pixel_grid->active_image_pixel_j[img_index];
						if ((include_fgmask_in_inversion) and (!image_data->foreground_mask_data[i][j])) include_pixel = false;
						if (include_pixel) logev_times_two_band -= log(image_data->covinv_map[i][j]); // if the loglike_reference_noise is equal to sqrt(noise_covariance), then this term becomes zero and it just looks like chi-square (which looks prettier)
					}
				} else {
					logev_times_two_band -= n_data_pixels*log(cov_inverse); // if the loglike_reference_noise is equal to sqrt(noise_covariance), then this term becomes zero and it just looks like chi-square (which looks prettier)
				}
				logev_times_two_band += n_data_pixels*log(M_2PI);
			}

			// Now we evaluate the nimg_prior to penalize the solution if it produces the wrong number of lensed images
			if (src_i_list[imggrid_i] != -1) {
				if ((n_image_prior) and (source_fit_mode != Cartesian_Source) and (source_fit_mode != Parameterized_Source)) {
#ifdef USE_OPENMP
					if (show_wtime) {
						wtime0 = omp_get_wtime();
					}
#endif
					if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) {
						image_pixel_grids[imggrid_i]->cartesian_srcgrid->assign_surface_brightness_from_analytic_source(zsrc_i);
					} else if (source_fit_mode==Delaunay_Source) {
						image_pixel_grids[imggrid_i]->cartesian_srcgrid->assign_surface_brightness_from_delaunay_grid(image_pixel_grids[imggrid_i]->delaunay_srcgrid);
					}
					pixel_avg_n_images = image_pixel_grids[imggrid_i]->cartesian_srcgrid->find_avg_n_images(n_image_prior_sb_frac);
#ifdef USE_OPENMP
					if (show_wtime) {
						wtime = omp_get_wtime() - wtime0;
						if (mpi_id==0) cout << "Wall time for assigning SB for nimg_prior: " << wtime << endl;
					}
#endif
				}
				if ((n_image_prior) and (source_fit_mode != Parameterized_Source)) {
					double chisq_penalty;
					if ((mpi_id==0) and (verbal)) cout << "Average number of images: " << pixel_avg_n_images << endl;
					if (pixel_avg_n_images < n_image_threshold) {
						chisq_penalty = pow(1+n_image_threshold-pixel_avg_n_images,60) - 1.0; // constructed so that penalty = 0 if the average n_image = n_image_threshold
						logev_times_two_band += chisq_penalty;
						if ((mpi_id==0) and (verbal)) cout << "*NOTE: average number of images is below the prior threshold (" << pixel_avg_n_images << " vs. " << n_image_threshold << "), resulting in penalty prior (chisq_penalty=" << chisq_penalty << ")" << endl;
					}
				}
			}
		}
		logev_times_two += logev_times_two_band;
		chisq0 += chisq0_band;
		if ((mpi_id==0) and (verbal)) {
			cout << "total number of image pixels included in loglike within the lensing mask (excluding foreground mask) = " << count << endl;
			if (include_foreground_sb) cout << "total number of foreground image pixels included in loglike = " << foreground_count << endl;
		}
		sb_outside_window = false;
		if ((outside_sb_prior) and (source_fit_mode != Parameterized_Source)) {
			add_outside_sb_prior_penalty(band_number,src_i_list,sb_outside_window,logev_times_two,verbal);
		}
		if (sb_outside_window) sb_outside_window_allbands = true;
	}

	if ((n_extended_src_redshifts > 1) and (mpi_id==0) and (verbal)) cout << "chisq0_tot=" << chisq0 << endl;
	if (((source_fit_mode==Cartesian_Source) or (source_fit_mode==Delaunay_Source) or (source_fit_mode==Shapelet_Source)) and ((source_npixels > 0) or (n_mge_amps > 0)))
	{
		if ((group_id==0) and (logfile.is_open())) {
			if (sb_outside_window_allbands) logfile << " -2*log(ev)=" << logev_times_two << " (no priors; SB produced outside window)" << endl;
			else logfile << " -2*log(ev)=" << logev_times_two << " (no priors)" << endl;
		}
		if ((mpi_id==0) and (verbal)) {
			cout << "-2*log(ev)=" << logev_times_two << " (a.k.a. 'chisq_pix')" << endl;
		}
	}
	if ((!include_noise_term_in_loglike) and (regularization_method != None) and (source_fit_mode != Parameterized_Source)) {
		if ((mpi_id==0) and (verbal)) cout << "NOTE: the noise term(s) in the log(evidence) are NOT being included (to include, set 'include_noise_term_in_loglike' to 'on')" << endl;
	}
	
	chisq_it++;

	delete[] src_i_list;
	return logev_times_two;
}

void QLens::setup_auxiliary_sourcegrids_and_point_imgs(int* src_i_list, const bool verbal)
{
	int i,src_i,band_number,zsrc_i;
	for (int imggrid_i=0; imggrid_i < n_image_pixel_grids; imggrid_i++) {
		src_i = src_i_list[imggrid_i];
		if (src_i == -1) continue;
		band_number = image_pixel_grids[imggrid_i]->band_number;
		zsrc_i = image_pixel_grids[imggrid_i]->src_redshift_index;
		bool source_grid_defined = false;
		if (source_fit_mode != Cartesian_Source) {
			if ((mpi_id==0) and (verbal)) cout << "Trying auxiliary sourcegrid creation..." << endl;
#ifdef USE_OPENMP
			double srcgrid_wtime0, srcgrid_wtime;
			if (show_wtime) {
				srcgrid_wtime0 = omp_get_wtime();
			}
#endif
			if ((source_fit_mode==Cartesian_Source) and (cartesian_srcgrids != NULL) and (cartesian_srcgrids[src_i] != NULL)) source_grid_defined = true;
			if (nlens > 0) {
				// create auxiliary source grid for findimg number of images or finding point images (if not using cartesian source grid already)
				if ((source_fit_mode==Shapelet_Source) and (n_image_prior)) { // note, in shapelet mode, we only need the auxiliary grid if using n_image_prior
					create_sourcegrid_cartesian(band_number,zsrc_i,verbal,true,true,true,true);
					source_grid_defined = true;
				} else if (source_fit_mode==Delaunay_Source) {
					create_sourcegrid_cartesian(band_number,zsrc_i,verbal,true,false,true,true);
					source_grid_defined = true;
				}
			}
			int src_icheck = -1;
			for (int i=0; i < n_pixellated_src; i++) {
				if ((pixellated_src_band[i]==band_number) and (pixellated_src_redshift_idx[i]==zsrc_i)) {
					src_icheck = i;
					break;
				}
			}

			if (cartesian_srcgrids[src_i]->qlens==NULL) die("cartesian source grid does not have pointer to qlens");
#ifdef USE_OPENMP
			if (source_grid_defined) {
				if (show_wtime) {
					srcgrid_wtime = omp_get_wtime() - srcgrid_wtime0;
					if (mpi_id==0) cout << "wall time for auxiliary source grid creation: " << srcgrid_wtime << endl;
				}
			}
#endif
		} else {
			source_grid_defined = true;
		}
		if (source_grid_defined) {
			image_pixel_grids[imggrid_i]->set_cartesian_srcgrid(cartesian_srcgrids[src_i]);
			cartesian_srcgrids[src_i]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
			if (!adaptive_subgrid) cartesian_srcgrids[src_i]->calculate_pixel_magnifications(); // if adaptive_subgrid is off, we still need to get pixel magnifications for nimg_prior
		}
		if (imggrid_i==0) {
			if (n_ptsrc > 0) {
				if (use_analytic_bestfit_src) set_analytic_sourcepts(verbal);
				if ((include_flux_chisq) and (analytic_source_flux)) set_analytic_srcflux(verbal);
				bool is_lensed;
				for (i=0; i < n_ptsrc; i++) {
					is_lensed = true;
					if (ptsrc_redshifts[ptsrc_redshift_idx[i]]==lens_redshift) is_lensed = false;
					if ((is_lensed) and (nlens==0)) die("lensed source point has been defined, but no lens objects have been created");
					image_pixel_grids[imggrid_i]->find_point_images(ptsrc_list[i]->pos[0],ptsrc_list[i]->pos[1],ptsrc_list[i]->images,source_grid_defined,is_lensed,verbal);
				}
			}
			for (i=0; i < n_sb; i++) {
				if (sb_list[i]->center_anchored_to_ptsrc==true) sb_list[i]->update_anchor_center();
			}
		}
	}
}

bool QLens::setup_cartesian_sourcegrid(const int imggrid_i, const int src_i, int& n_expected_imgpixels, const bool verbal)
{
	image_pixel_grids[imggrid_i]->set_cartesian_srcgrid(cartesian_srcgrids[src_i]);
	if (auto_sourcegrid) image_pixel_grids[imggrid_i]->find_optimal_sourcegrid(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,sourcegrid_limit_xmin,sourcegrid_limit_xmax,sourcegrid_limit_ymin,sourcegrid_limit_ymax);
	if (auto_srcgrid_npixels) {
		if (auto_srcgrid_set_pixel_size) {
			image_pixel_grids[imggrid_i]->find_optimal_firstlevel_sourcegrid_npixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_expected_imgpixels);
		} else {
			image_pixel_grids[imggrid_i]->find_optimal_sourcegrid_npixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_expected_imgpixels);
			srcgrid_npixels_x *= 2; // aim high, since many of the source grid pixels may lie outside the mask (we'll refine the # of pixels after drawing the grid once)
			srcgrid_npixels_y *= 2;
		}
	}
	if ((srcgrid_npixels_x < 2) or (srcgrid_npixels_y < 2)) {
		if ((mpi_id==0) and (verbal)) cout << "Source grid has negligible size...cannot invert image\n";
		if (logfile.is_open()) 
			logfile << "it=" << chisq_it << " chisq0=2e30" << endl;
		return false;
	}
	if (auto_sourcegrid) {
		if (image_pixel_grids[imggrid_i]->cartesian_srcgrid->srcgrid_size_scale != 0) {
			double xwidth_adj = image_pixel_grids[imggrid_i]->cartesian_srcgrid->srcgrid_size_scale*(sourcegrid_xmax-sourcegrid_xmin);
			double ywidth_adj = image_pixel_grids[imggrid_i]->cartesian_srcgrid->srcgrid_size_scale*(sourcegrid_ymax-sourcegrid_ymin);
			double srcgrid_xc, srcgrid_yc;
			srcgrid_xc = (sourcegrid_xmax + sourcegrid_xmin)/2;
			srcgrid_yc = (sourcegrid_ymax + sourcegrid_ymin)/2;
			sourcegrid_xmin = srcgrid_xc - xwidth_adj/2;
			sourcegrid_xmax = srcgrid_xc + xwidth_adj/2;
			sourcegrid_ymin = srcgrid_yc - ywidth_adj/2;
			sourcegrid_ymax = srcgrid_yc + ywidth_adj/2;
		}
		if ((mpi_id==0) and (verbal)) {
			cout << "Sourcegrid dimensions: " << sourcegrid_xmin << " " << sourcegrid_xmax << " " << sourcegrid_ymin << " " << sourcegrid_ymax << endl;
		}
	}
	return true;
}


bool QLens::generate_and_invert_lensing_matrix_cartesian(const int imggrid_i, const int src_i, double& tot_wtime, double& tot_wtime0, const bool verbal)
{
	if ((mpi_id==0) and (verbal)) cout << "Assigning pixel mappings...\n";
	if (assign_pixel_mappings(imggrid_i,false,verbal)==false) {
		return false;
	}
	if ((mpi_id==0) and (verbal)) cout << "Assigning foreground pixel mappings... (MAYBE REMOVE THIS FROM CHISQ AND DO AHEAD OF TIME?)\n";
	assign_foreground_mappings(imggrid_i);

	if ((mpi_id==0) and (verbal)) {
		cout << "Number of active image pixels: " << image_npixels << endl;
	}
	//if (mpi_id==0) cout << "****Overlap area: " << total_srcgrid_overlap_area << endl;
	//if (mpi_id==0) cout << "****High S/N Overlap area: " << high_sn_srcgrid_overlap_area << endl;
	double src_pixel_area = ((sourcegrid_xmax-sourcegrid_xmin)*(sourcegrid_ymax-sourcegrid_ymin)) / (srcgrid_npixels_x*srcgrid_npixels_y);
	double est_nmapped = total_srcgrid_overlap_area / src_pixel_area;
	double est_pixfrac = est_nmapped / image_npixels;
	if ((mpi_id==0) and (verbal)) {
		double pixfrac = ((double) n_amps) / image_npixels;
		cout << "Actual f = " << pixfrac << endl;
		if (auto_srcgrid_npixels) {
			double high_sn_pixfrac = ((double) n_amps*high_sn_srcgrid_overlap_area/total_srcgrid_overlap_area) / image_pixel_grids[imggrid_i]->n_high_sn_pixels;
			cout << "Actual high S/N f = " << high_sn_pixfrac << endl;
		}
	}
	if ((mpi_id==0) and (verbal)) cout << "Initializing pixel matrices...\n";
	initialize_pixel_matrices(imggrid_i,false,verbal);
	if (regularization_method != None) create_regularization_matrix(imggrid_i);
	if (inversion_method==DENSE) {
		PSF_convolution_Lmatrix_dense(imggrid_i,verbal);
	} else {
		PSF_convolution_Lmatrix(imggrid_i,verbal);
	}
	image_pixel_grids[imggrid_i]->set_surface_brightness_vector_to_data(); // note that image_pixel_grids[imggrid_i] just has the data pixel values stored in it
	if (!ignore_foreground_in_chisq) {
		calculate_foreground_pixel_surface_brightness(imggrid_i,true);
		store_foreground_pixel_surface_brightness(imggrid_i);
	}
	if ((n_ptsrc > 0) and (!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) {
		if ((mpi_id==0) and (verbal)) cout << "Generating point images..." << endl;
		for (int i=0; i < n_ptsrc; i++) {
			image_pixel_grids[imggrid_i]->generate_point_images(ptsrc_list[i]->images, point_image_surface_brightness, include_imgfluxes_in_inversion, ptsrc_list[i]->srcflux);
		}
	}

	if ((mpi_id==0) and (verbal)) cout << "Creating lensing matrices...\n" << flush;
	bool dense_Fmatrix = ((inversion_method==DENSE) or (inversion_method==DENSE_FMATRIX)) ? true : false;
	if (inversion_method==DENSE) create_lensing_matrices_from_Lmatrix_dense(imggrid_i,false,verbal);
	else create_lensing_matrices_from_Lmatrix(imggrid_i,dense_Fmatrix,false,verbal);
#ifdef USE_OPENMP
	if (show_wtime) {
		tot_wtime = omp_get_wtime() - tot_wtime0;
		if (mpi_id==0) cout << "Total wall time before F-matrix inversion: " << tot_wtime << endl;
	}
#endif

	if ((mpi_id==0) and (verbal)) cout << "Inverting lens mapping...\n" << flush;
	if ((optimize_regparam) and (regularization_method != None)) {
		optimize_regularization_parameter(imggrid_i,dense_Fmatrix,verbal);
	}
	if ((!optimize_regparam)) {
		if (inversion_method==MUMPS) invert_lens_mapping_MUMPS(imggrid_i,Fmatrix_log_determinant,verbal);
		else if (inversion_method==UMFPACK) invert_lens_mapping_UMFPACK(imggrid_i,Fmatrix_log_determinant,verbal);
		else if ((inversion_method==DENSE) or (inversion_method==DENSE_FMATRIX)) invert_lens_mapping_dense(imggrid_i,verbal);
		else invert_lens_mapping_CG_method(imggrid_i,verbal);
	}
	return true;
}

bool QLens::generate_and_invert_lensing_matrix_delaunay(const int imggrid_i, const int src_i, const bool potential_perturbations, const bool save_sb_gradient, double& tot_wtime, double& tot_wtime0, const bool verbal)
{
	if ((mpi_id==0) and (verbal)) cout << "Assigning pixel mappings...\n";
	if (assign_pixel_mappings(imggrid_i,potential_perturbations,verbal)==false) {
		clear_pixel_matrices();
		return false;
	}
	if ((mpi_id==0) and (verbal)) {
		cout << "Number of active image pixels: " << image_npixels << endl;
	}

	//if (potential_perturbations) image_pixel_grids[imggrid_i]->calculate_subpixel_source_gradient();
	if ((mpi_id==0) and (verbal)) cout << "Initializing pixel matrices...\n";
	initialize_pixel_matrices(imggrid_i,potential_perturbations,verbal);
	double xc_approx, yc_approx;

	bool include_lum_weighting = ((use_lum_weighted_regularization) and (get_lumreg_from_sbweights)) ? true : false;
	if ((regularization_method != None) and (image_pixel_grids[imggrid_i]->delaunay_srcgrid != NULL)) {
		if (create_regularization_matrix(imggrid_i,include_lum_weighting,get_lumreg_from_sbweights,false,verbal)==false) { clear_pixel_matrices(); clear_sparse_lensing_matrices(); return false; } // in this case, covariance matrix was not positive definite 
		if ((potential_perturbations) and (create_regularization_matrix(imggrid_i,include_lum_weighting,get_lumreg_from_sbweights,true,verbal)==false)) { clear_pixel_matrices(); clear_sparse_lensing_matrices(); return false; } // in this case, covariance matrix was not positive definite 
	}

	if ((mpi_id==0) and (verbal)) {
		cout << "Number of active image pixels: " << image_npixels << endl;
		cout << "Number of source pixels: " << source_npixels << endl;
		if (n_amps > source_npixels) cout << "Number of total amplitudes: " << n_amps << endl;
	}

	if (inversion_method==DENSE) {
		PSF_convolution_Lmatrix_dense(imggrid_i,verbal);
	} else {
		PSF_convolution_Lmatrix(imggrid_i,verbal);
	}
	image_pixel_grids[imggrid_i]->set_surface_brightness_vector_to_data(); // note that image_pixel_grids[imggrid_i] just has the data pixel values stored in it
	if ((n_ptsrc > 0) and (!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) {
		// Note that if image fluxes are included as linear parameters, we don't need to add point images to the SB separately because
		// they will be included in the Lmatrix. Otherwise, we add them using the code below.
		if ((mpi_id==0) and (verbal)) cout << "Generating point images..." << endl;
		for (int i=0; i < n_ptsrc; i++) {
			image_pixel_grids[imggrid_i]->generate_point_images(ptsrc_list[i]->images, point_image_surface_brightness, include_imgfluxes_in_inversion, ptsrc_list[i]->srcflux);
		}
	}

	if ((mpi_id==0) and (verbal)) cout << "Creating lensing matrices...\n" << flush;
	bool dense_Fmatrix = ((inversion_method==DENSE) or (inversion_method==DENSE_FMATRIX)) ? true : false;
	if (inversion_method==DENSE) create_lensing_matrices_from_Lmatrix_dense(imggrid_i,potential_perturbations,verbal);
	else create_lensing_matrices_from_Lmatrix(imggrid_i,dense_Fmatrix,potential_perturbations,verbal);
#ifdef USE_OPENMP
	if (show_wtime) {
		tot_wtime = omp_get_wtime() - tot_wtime0;
		if (mpi_id==0) cout << "Total wall time before F-matrix inversion: " << tot_wtime << endl;
	}
#endif
	if ((mpi_id==0) and (verbal)) cout << "Inverting lens mapping...\n" << flush;
	//if ((optimize_regparam) and (regularization_method != None) and (image_pixel_grids[imggrid_i]->delaunay_srcgrid != NULL)) 
	if ((optimize_regparam) and (regularization_method != None) and (!potential_perturbations) and (image_pixel_grids[imggrid_i]->delaunay_srcgrid != NULL)) {
		bool pre_srcgrid = ((use_lum_weighted_srcpixel_clustering) and (!use_saved_sbweights)) ? true : false;
		if (optimize_regularization_parameter(imggrid_i,dense_Fmatrix,verbal,pre_srcgrid)==false) { clear_pixel_matrices(); clear_sparse_lensing_matrices(); return false; }
	}
	if ((use_lum_weighted_srcpixel_clustering) and (!use_saved_sbweights)) {
#ifdef USE_OPENMP
		double srcgrid_wtime0, srcgrid_wtime;
		if (show_wtime) {
			srcgrid_wtime0 = omp_get_wtime();
		}
#endif
		create_sourcegrid_from_imggrid_delaunay(true,image_pixel_grids[imggrid_i]->band_number,image_pixel_grids[imggrid_i]->src_redshift_index,verbal);

#ifdef USE_OPENMP
		if (show_wtime) {
			srcgrid_wtime = omp_get_wtime() - srcgrid_wtime0;
			if (mpi_id==0) cout << "wall time for Delaunay grid creation (with lum weighting): " << srcgrid_wtime << endl;
		}
#endif
		image_pixel_grids[imggrid_i]->set_delaunay_srcgrid(delaunay_srcgrids[src_i]);
		delaunay_srcgrids[src_i]->set_image_pixel_grid(image_pixel_grids[imggrid_i]);
		clear_sparse_lensing_matrices();
		clear_pixel_matrices();

		if ((mpi_id==0) and (verbal)) cout << "Assigning pixel mappings (with lum weighting)...\n";
		if (assign_pixel_mappings(imggrid_i,potential_perturbations,verbal)==false) {
			clear_pixel_matrices();
			clear_sparse_lensing_matrices();
			return false;
		}
		if ((mpi_id==0) and (verbal)) cout << "Assigning foreground pixel mappings (with lum weighting)... (MAYBE REMOVE THIS FROM CHISQ AND DO AHEAD OF TIME?)\n";
		assign_foreground_mappings(imggrid_i);

		if ((mpi_id==0) and (verbal)) {
			cout << "Number of active image pixels (with lum weighting): " << image_npixels << endl;
		}

		if ((mpi_id==0) and (verbal)) cout << "Initializing pixel matrices (with lum weighting)...\n";
		initialize_pixel_matrices(imggrid_i,verbal);
		if (regularization_method != None) {
			if (create_regularization_matrix(imggrid_i,include_lum_weighting,get_lumreg_from_sbweights,false,verbal)==false) { clear_pixel_matrices(); clear_sparse_lensing_matrices(); return false; } // in this case, covariance matrix was not positive definite 
			if ((potential_perturbations) and (create_regularization_matrix(imggrid_i,include_lum_weighting,get_lumreg_from_sbweights,true,verbal)==false)) { clear_pixel_matrices(); clear_sparse_lensing_matrices(); return false; } // in this case, covariance matrix was not positive definite 
		}
		if (inversion_method==DENSE) {
			PSF_convolution_Lmatrix_dense(imggrid_i,verbal);
		} else {
			PSF_convolution_Lmatrix(imggrid_i,verbal);
		}
		image_pixel_grids[imggrid_i]->set_surface_brightness_vector_to_data(); // note that image_pixel_grids[imggrid_i] just has the data pixel values stored in it
		if (!ignore_foreground_in_chisq) {
			calculate_foreground_pixel_surface_brightness(imggrid_i,true);
			store_foreground_pixel_surface_brightness(imggrid_i);
		}
		if ((n_ptsrc > 0) and (!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) {
			for (int i=0; i < n_ptsrc; i++) {
				image_pixel_grids[imggrid_i]->generate_point_images(ptsrc_list[i]->images, point_image_surface_brightness, include_imgfluxes_in_inversion, ptsrc_list[i]->srcflux);
			}
		}

		if ((mpi_id==0) and (verbal)) cout << "Creating lensing matrices (with lum weighting)...\n" << flush;
		bool dense_Fmatrix = ((inversion_method==DENSE) or (inversion_method==DENSE_FMATRIX)) ? true : false;
		if (inversion_method==DENSE) create_lensing_matrices_from_Lmatrix_dense(imggrid_i,potential_perturbations,verbal);
		else create_lensing_matrices_from_Lmatrix(imggrid_i,dense_Fmatrix,potential_perturbations,verbal);
#ifdef USE_OPENMP
		if (show_wtime) {
			tot_wtime = omp_get_wtime() - tot_wtime0;
			if (mpi_id==0) cout << "Total wall time before F-matrix inversion (with lum weighting): " << tot_wtime << endl;
		}
#endif
		if ((mpi_id==0) and (verbal)) cout << "Inverting lens mapping...\n" << flush;
		if ((optimize_regparam) and (regularization_method != None) and (!potential_perturbations) and (image_pixel_grids[imggrid_i]->delaunay_srcgrid != NULL)) {
			if (optimize_regularization_parameter(imggrid_i,dense_Fmatrix,verbal)==false) { clear_pixel_matrices(); clear_sparse_lensing_matrices(); return false; }
		}
	}
	if ((!optimize_regparam) or (potential_perturbations)) {
		if (inversion_method==MUMPS) invert_lens_mapping_MUMPS(imggrid_i,Fmatrix_log_determinant,verbal);
		else if (inversion_method==UMFPACK) invert_lens_mapping_UMFPACK(imggrid_i,Fmatrix_log_determinant,verbal);
		else if ((inversion_method==DENSE) or (inversion_method==DENSE_FMATRIX)) invert_lens_mapping_dense(imggrid_i,verbal);
		else invert_lens_mapping_CG_method(imggrid_i,verbal);
	}
	if (save_sb_gradient) image_pixel_grids[imggrid_i]->calculate_subpixel_source_gradient();

	return true;
}

void QLens::add_outside_sb_prior_penalty(const int band_number, int* src_i_list, bool& sb_outside_window, double& logev_times_two, const bool verbal)
{
	bool supersampling_orig = psf_supersampling;
	psf_supersampling = false; // since emask pixels may have fewer or no splittings, we cannot use supersampling for the outside_sb_prior
	int i,j,zsrc_i,imggrid_i;

	ImageData *image_data;
	if (n_data_bands > band_number) image_data = imgdata_list[band_number];
	else image_data = NULL;

	if ((source_fit_mode==Cartesian_Source) or (source_fit_mode==Delaunay_Source)) {
		for (zsrc_i=0; zsrc_i < n_extended_src_redshifts; zsrc_i++) {
			imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;
			if (src_i_list[imggrid_i] == -1) continue;
			delaunay_srcgrids[src_i_list[imggrid_i]]->look_for_starting_point = false; // since we're unmasking, don't use the masked pixels to look for starting point when finding containing triangles
			image_pixel_grids[imggrid_i]->activate_extended_mask(false); 
			image_pixel_grids[imggrid_i]->redo_lensing_calculations(false); // This shouldn't be necessary! FIX!!!
			assign_pixel_mappings(imggrid_i,false,verbal);
			//initialize_pixel_matrices(imggrid_i,false,verbal);
			//if (inversion_method==DENSE) die("need to implement FFT convolution of emask for outside_sb_prior");
			//else PSF_convolution_Lmatrix(imggrid_i,verbal);
			//if (source_fit_mode==Cartesian_Source) image_pixel_grids[imggrid_i]->cartesian_srcgrid->fill_surface_brightness_vector();
			//else image_pixel_grids[imggrid_i]->delaunay_srcgrid->fill_surface_brightness_vector();
			//if (inversion_method==DENSE) calculate_image_pixel_surface_brightness_dense();
			//else calculate_image_pixel_surface_brightness();
			image_pixel_grids[imggrid_i]->find_surface_brightness(false,true);
			vectorize_image_pixel_surface_brightness(imggrid_i,true);
			PSF_convolution_pixel_vector(imggrid_i,false,verbal,false); // no PSF supersampling, no FFT convolution (saves time)
			store_image_pixel_surface_brightness(imggrid_i);
			delaunay_srcgrids[src_i_list[imggrid_i]]->look_for_starting_point = true; // BTW, you should use a better algorithm to look for containing triangles that doesn't rely on ray tracing, but don't worry about it for now
			clear_sparse_lensing_matrices();
			clear_pixel_matrices();
		}
	} else if (source_fit_mode==Shapelet_Source) {
#ifdef USE_OPENMP
		double sbwtime, sbwtime0;
		if (show_wtime) {
			sbwtime0 = omp_get_wtime();
		}
#endif
		for (zsrc_i=0; zsrc_i < n_extended_src_redshifts; zsrc_i++) {
			imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;
			//image_pixel_grids[imggrid_i]->activate_extended_mask(); 
			image_pixel_grids[imggrid_i]->find_surface_brightness(false,true);
			vectorize_image_pixel_surface_brightness(imggrid_i);
#ifdef USE_OPENMP
			if (show_wtime) {
				sbwtime = omp_get_wtime() - sbwtime0;
				if (mpi_id==0) cout << "Wall time for calculating SB outside mask: " << sbwtime << endl;
			}
#endif
			PSF_convolution_pixel_vector(imggrid_i,false,verbal,false); // no supersampling, no fft convolution (saves time)
			store_image_pixel_surface_brightness(imggrid_i);
			//image_pixel_grids[imggrid_i]->load_data((*image_data)); // This restores pixel data values to image_pixel_grids[0] (used for the inversion)
			clear_pixel_matrices();
		}
	} else if (source_fit_mode==Parameterized_Source) {
		for (zsrc_i=0; zsrc_i < n_extended_src_redshifts; zsrc_i++) {
			imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;
			//if (image_data->extended_mask_n_neighbors[assigned_mask[assigned_mask[imggrid_i]]] == -1) image_pixel_grids[imggrid_i]->include_all_pixels();
			image_pixel_grids[imggrid_i]->activate_extended_mask(); 
			image_pixel_grids[imggrid_i]->find_surface_brightness(false,true);
			vectorize_image_pixel_surface_brightness(imggrid_i);
			PSF_convolution_pixel_vector(imggrid_i,false,verbal,false); // no supersampling, no convolution (saves time)
			store_image_pixel_surface_brightness(imggrid_i);
			clear_pixel_matrices();
		}
	}

	bool **mask_for_inversion;
	double bg_noise;
	for (zsrc_i=0; zsrc_i < n_extended_src_redshifts; zsrc_i++) {
		imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;
		if (src_i_list[imggrid_i] == -1) continue;
		if (include_fgmask_in_inversion) mask_for_inversion = image_pixel_grids[imggrid_i]->fgmask;
		else mask_for_inversion = image_pixel_grids[imggrid_i]->mask;
		double max_external_sb = -1e30, max_sb = -1e30;
		for (i=0; i < image_data->npixels_x; i++) {
			for (j=0; j < image_data->npixels_y; j++) {
				if ((mask_for_inversion) and (image_pixel_grids[imggrid_i]->maps_to_source_pixel[i][j])) {
					//img_index = image_pixel_grids[imggrid_i]->pixel_index[i][j];
					if (image_pixel_grids[imggrid_i]->surface_brightness[i][j] > max_sb) {
						 max_sb = image_pixel_grids[imggrid_i]->surface_brightness[i][j];
					}
				}
			}
		}
		 
		// NOTE: by default, outside_sb_prior_noise_frac is a negative number so it isn't used. But it can be changed by the user (useful for low S/N sources)
		bg_noise = (use_noise_map) ? image_data->bg_pixel_noise : background_pixel_noise;
		double outside_sb_threshold = dmax(outside_sb_prior_noise_frac*bg_noise,outside_sb_prior_threshold*max_sb);
		int isb, jsb;
		if (n_image_pixel_grids==1) {
			if ((verbal) and (mpi_id==0)) cout << "OUTSIDE SB THRESHOLD: " << outside_sb_threshold << endl;
		} else {
			if ((verbal) and (mpi_id==0)) cout << "OUTSIDE SB THRESHOLD (imggrid_i=" << imggrid_i << "): " << outside_sb_threshold << endl;
		}
		for (i=0; i < image_data->npixels_x; i++) {
			for (j=0; j < image_data->npixels_y; j++) {
				if ((!mask_for_inversion[i][j]) and ((image_pixel_grids[imggrid_i]->emask[i][j])) and (image_pixel_grids[imggrid_i]->maps_to_source_pixel[i][j])) {
					//img_index = image_pixel_grids[imggrid_i]->pixel_index[i][j];
					//cout << image_surface_brightness[img_index] << endl;
					if (abs(image_pixel_grids[imggrid_i]->surface_brightness[i][j]) >= outside_sb_threshold) {
						if (abs(image_pixel_grids[imggrid_i]->surface_brightness[i][j]) > max_external_sb) {
							 max_external_sb = abs(image_pixel_grids[imggrid_i]->surface_brightness[i][j]);
							 isb=i; jsb=j;
						}
					}
				}
			}
		}
		if (max_external_sb > 0) {
			double chisq_penalty;
			sb_outside_window = true;
			chisq_penalty = pow(1+abs((max_external_sb-outside_sb_threshold)/outside_sb_threshold),60) - 1.0;
			logev_times_two += chisq_penalty;
			if ((mpi_id==0) and (verbal)) cout << "*NOTE: surface brightness above the prior threshold (" << max_external_sb << " vs. " << outside_sb_threshold << ") has been found outside the selected fit region at pixel (" << image_pixel_grids[imggrid_i]->center_pts[isb][jsb][0] << "," << image_pixel_grids[imggrid_i]->center_pts[isb][jsb][1] << "), resulting in penalty prior (chisq_penalty=" << chisq_penalty << ")" << endl;
		}
		image_pixel_grids[imggrid_i]->set_fit_window((*image_data),true,assigned_mask[imggrid_i],false,include_fgmask_in_inversion);
		psf_supersampling = supersampling_orig;
	}
}

void QLens::add_regularization_prior_terms_to_logev(const int band_number, const int zsrc_i, double& logev_times_two, double& loglike_reg, double& regterms, const bool include_potential_perturbations, const bool verbal)
{
	int imggrid_i = band_number*n_extended_src_redshifts + zsrc_i;
	if (source_npixels > 0) {
		source_npixels_ptr = src_npixels_inv;
		src_npixel_start_ptr = src_npixel_start;
		Rmatrix_ptr = Rmatrix;
		Rmatrix_index_ptr = Rmatrix_index;
		covmatrix_stacked_ptr = covmatrix_stacked;
		covmatrix_packed_ptr = covmatrix_packed;
		covmatrix_factored_ptr = covmatrix_factored;
		Rmatrix_packed_ptr = Rmatrix_packed;
		Rmatrix_log_determinant_ptr = Rmatrix_log_determinant;

		for (int i=0; i < n_src_inv; i++) {
			if ((source_fit_mode==Delaunay_Source) or (source_fit_mode==Cartesian_Source)) {
				ImagePixelGrid *imggrid = image_pixel_grids[image_pixel_grids[imggrid_i]->imggrid_indx_to_include_in_Lmatrix[i]];
				if (source_fit_mode==Delaunay_Source) {
					regparam_ptr = &imggrid->delaunay_srcgrid->regparam;
				} else if (source_fit_mode==Cartesian_Source) {
					regparam_ptr = &imggrid->cartesian_srcgrid->regparam;
				}
			}
			else if (source_fit_mode==Shapelet_Source) {
				int shapelet_i = -1;
				for (int i=0; i < n_sb; i++) {
					if ((sb_list[i]->sbtype==SHAPELET) and ((zsrc_i<0) or (sbprofile_imggrid_idx[i]==imggrid_i))) {
						shapelet_i = i;
						break;
					}
				}

				if (shapelet_i >= 0) sb_list[shapelet_i]->get_regularization_param_ptr(regparam_ptr);
				else die("shapelet not found");
			}
			else die("unknown source pixellation mode");

			if ((*regparam_ptr) != 0) {
				regterms = calculate_regularization_prior_term(regparam_ptr);
				logev_times_two += regterms;
				loglike_reg += regterms;
			}
			if ((mpi_id==0) and (verbal)) {
				if ((source_npixels > 0) and (regularization_method != None)) {
					if (n_extended_src_redshifts > 1) cout << "imggrid_i=" << imggrid_i << ": ";
					if (use_covariance_matrix) cout << "logdet(Gmatrix)=" << Gmatrix_log_determinant;
					else cout << "logdet(Fmatrix)=" << Fmatrix_log_determinant;
					if (Rmatrix_log_determinant != NULL) cout << " logdet(Rmatrix)=" << (*Rmatrix_log_determinant_ptr);
					cout << endl;
				}
			}


			source_npixels_ptr++;
			src_npixel_start_ptr++;
			Rmatrix_ptr++;
			Rmatrix_index_ptr++;
			covmatrix_stacked_ptr++;
			covmatrix_packed_ptr++;
			covmatrix_factored_ptr++;
			Rmatrix_packed_ptr++;
			Rmatrix_log_determinant_ptr++;
		}
	}
	if (n_mge_amps > 0)  {
		regterms = calculate_MGE_regularization_prior_term(imggrid_i);
		logev_times_two += regterms;
		loglike_reg += regterms;
	}
	if ((include_potential_perturbations) and (image_pixel_grids[imggrid_i]->lensgrid != NULL)) {
		regparam_ptr = &(image_pixel_grids[imggrid_i]->lensgrid->regparam);
		if ((*regparam_ptr) != 0) {
			Rmatrix_ptr = &Rmatrix_pot;
			Rmatrix_index_ptr = &Rmatrix_pot_index;
			Rmatrix_packed_ptr = &Rmatrix_pot_packed;
			regterms = calculate_regularization_prior_term(regparam_ptr,true);
			logev_times_two += regterms;
			loglike_reg += regterms;
		}
	}

	if ((!use_covariance_matrix) or (source_fit_mode != Delaunay_Source)) logev_times_two += Fmatrix_log_determinant;
	else logev_times_two += Gmatrix_log_determinant;

}

void QLens::set_n_imggrids_to_include_in_inversion()
{
	if (n_extended_src_redshifts < 2) return;
	// right now, only allow two combine zsrc_i=0 and zsrc_i=1 pixellated sources in inversion (can extend later if desired)
	if (include_two_pixsrc_in_Lmatrix) {
		if (image_pixel_grids[0] != NULL) image_pixel_grids[0]->set_include_in_Lmatrix(1);
		if (image_pixel_grids[1] != NULL) image_pixel_grids[1]->n_pixsrc_to_include_in_Lmatrix = 0;
	} else {
		if (image_pixel_grids[0] != NULL) image_pixel_grids[0]->set_include_only_one_pixsrc_in_Lmatrix();
		if (image_pixel_grids[1] != NULL) image_pixel_grids[1]->n_pixsrc_to_include_in_Lmatrix = 1;
	}
}


void QLens::create_output_directory()
{
	if (mpi_id==0) {
		struct stat sb;
		stat(fit_output_dir.c_str(),&sb);
		if (S_ISDIR(sb.st_mode)==false)
			mkdir(fit_output_dir.c_str(),S_IRWXU | S_IRWXG);
	}
}

void QLens::open_output_file(ofstream &outfile, char* filechar_in)
{
	string filename_in(filechar_in);
	if (fit_output_dir != ".") create_output_directory(); // in case it hasn't been created already
	string filename = fit_output_dir + "/" + filename_in;
	//string filename = filename_in;
	outfile.open(filename.c_str());
}

void QLens::open_output_file(ofstream &outfile, string filename_in)
{
	if (fit_output_dir != ".") create_output_directory(); // in case it hasn't been created already
	string filename = fit_output_dir + "/" + filename_in;
	//string filename = filename_in;
	outfile.open(filename.c_str());
}

void QLens::open_input_file(ifstream &infile, string filename_in)
{
	string filename = fit_output_dir + "/" + filename_in;
	infile.open(filename.c_str());
	// should return STATUS! change this
}

void QLens::reset_grid()
{
	if (grid != NULL) {
		delete grid;
		grid = NULL;
	}
	critical_curve_pts.clear();
	caustic_pts.clear();
	length_of_cc_cell.clear();
	sorted_critical_curves = false;
	sorted_critical_curve.clear();
	singular_pts.clear();
}

QLens::~QLens()
{
	clear_pixel_matrices();
	int i,j;
	if (nlens > 0) {
		for (i=0; i < nlens; i++) {
			delete lens_list[i];
		}
		delete[] lens_redshift_idx;
		delete[] lens_list;
	}

	if (n_sb > 0) {
		for (i=0; i < n_sb; i++)
			delete sb_list[i];
		delete[] sbprofile_redshift_idx;
		delete[] sbprofile_band_number;
		delete[] sb_list;
		n_sb = 0;
	}
	if (n_pixellated_src > 0) {
		for (i=0; i < n_pixellated_src; i++) {
			if (delaunay_srcgrids[i] != NULL) delete delaunay_srcgrids[i];
			if (cartesian_srcgrids[i] != NULL) delete cartesian_srcgrids[i];
		}
		delete[] pixellated_src_redshift_idx;
		delete[] pixellated_src_band;
		delete[] delaunay_srcgrids;
		delete[] cartesian_srcgrids;
		delete[] srcgrids;
	}
	if (n_pixellated_lens > 0) {
		for (i=0; i < n_pixellated_lens; i++) {
			if (lensgrids[i] != NULL) delete lensgrids[i];
		}
		delete[] pixellated_lens_redshift_idx;
		delete[] lensgrids;
	}
	if (n_ptsrc > 0) {
		for (i=0; i < n_ptsrc; i++) {
			if (ptsrc_list[i] != NULL) delete ptsrc_list[i];
		}
		delete[] ptsrc_redshift_idx;
		delete[] ptsrc_list;
	}
	if (n_psf > 0) {
		for (i=0; i < n_psf; i++) {
			if (psf_list[i] != NULL) delete psf_list[i];
		}
		delete[] psf_list;
	}

	delete grid;
	delete param_list;
	delete dparam_list;
	if (fitmodel != NULL) delete fitmodel;

	if (n_ptsrc_redshifts > 0) {
		delete[] ptsrc_redshifts;
		for (i=0; i < n_ptsrc_redshifts; i++) {
			if (n_lens_redshifts > 0) delete[] ptsrc_zfactors[i];
		}
		delete[] ptsrc_zfactors;
	}
	if (n_extended_src_redshifts > 0) {
		delete[] extended_src_redshifts;
		delete[] assigned_mask;
		for (i=0; i < n_extended_src_redshifts; i++) {
			if (n_lens_redshifts > 0) delete[] extended_src_zfactors[i];
		}
		delete[] extended_src_zfactors;
	}
	if (image_pixel_grids != NULL) {
		for (i=0; i < n_image_pixel_grids; i++) {
			if (image_pixel_grids[i] != NULL) delete image_pixel_grids[i];
		}
		delete[] image_pixel_grids;
	}

	if (n_lens_redshifts > 0) {
		delete[] lens_redshifts;
		if (default_zsrc_beta_factors != NULL) {
			for (i=0; i < n_lens_redshifts-1; i++) delete[] default_zsrc_beta_factors[i];
			delete[] default_zsrc_beta_factors;
		}
		if (extended_src_beta_factors != NULL) {
			for (i=0; i < n_extended_src_redshifts; i++) {
				for (j=0; j < n_lens_redshifts-1; j++) delete[] extended_src_beta_factors[i][j];
				if (n_lens_redshifts > 1) delete[] extended_src_beta_factors[i];
			}
			delete[] extended_src_beta_factors;
		}
		if (ptsrc_beta_factors != NULL) {
			for (i=0; i < n_ptsrc_redshifts; i++) {
				for (j=0; j < n_lens_redshifts-1; j++) delete[] ptsrc_beta_factors[i][j];
				if (n_lens_redshifts > 1) delete[] ptsrc_beta_factors[i];
			}
			delete[] ptsrc_beta_factors;
		}
	}

	if ((point_image_data != NULL) and (borrowed_image_data==false)) delete[] point_image_data;
	if ((imgdata_list != NULL) and (borrowed_image_data==false)) {
		if (n_data_bands > 0) {
			for (i=0; i < n_data_bands; i++) {
				if (imgdata_list[i] != NULL) delete imgdata_list[i];
			}
			delete[] imgdata_list;
		}
	}
	if (group_leader != NULL) delete[] group_leader;
	if (saved_sbweights != NULL) delete[] saved_sbweights;
	if (cosmology_allocated_within_qlens) delete cosmo;
}

/***********************************************************************************************************************/

// POLYCHORD/MULTINEST FUNCTIONS

void QLens::transform_cube(double* params, double* Cube) {
	double *lower_limits = param_list->prior_limits_lo;
	double *upper_limits = param_list->prior_limits_hi;
	for (int i=0; i < param_list->nparams; i++) {
		params[i] = lower_limits[i] + Cube[i]*(upper_limits[i]-lower_limits[i]);
	}
}

QLens* lensptr;
double mcsampler_set_lensptr(QLens* lens_in)
{
	lensptr = lens_in;
	return 0.0;
}

double polychord_loglikelihood (double theta[], int nDims, double phi[], int nDerived)
{
	double logl = -lensptr->LogLikeFunc(theta);
	lensptr->DerivedParamFunc(theta,phi);
	return logl;
}

void polychord_prior (double cube[], double theta[], int nDims)
{
	lensptr->transform_cube(theta,cube);
}

void polychord_dumper(int ndead,int nlive,int npars,double* live,double* dead,double* logweights,double logZ, double logZerr)
{
}

void multinest_loglikelihood(double *Cube, int &ndim, int &npars, double &lnew, void *context)
{
	double *params = new double[ndim];
	lensptr->transform_cube(params,Cube);
	lnew = -lensptr->LogLikeFunc(params);
	lensptr->DerivedParamFunc(params,Cube+ndim);
	delete[] params;
}

void dumper_multinest(int &nSamples, int &nlive, int &nPar, double **physLive, double **posterior, double **paramConstr, double &maxLogLike, double &logZ, double &INSlogZ, double &logZerr, void *context)
{
	// convert the 2D Fortran arrays to C++ arrays
	
	// the posterior distribution
	// postdist will have nPar parameters in the first nPar columns & loglike value & the posterior probability in the last two columns
	
	int i, j;
	
	double postdist[nSamples][nPar + 2];
	for( i = 0; i < nPar + 2; i++ )
		for( j = 0; j < nSamples; j++ )
			postdist[j][i] = posterior[0][i * nSamples + j];
	
	// last set of live points
	// pLivePts will have nPar parameters in the first nPar columns & loglike value in the last column
	
	double pLivePts[nlive][nPar + 1];
	for( i = 0; i < nPar + 1; i++ )
		for( j = 0; j < nlive; j++ )
			pLivePts[j][i] = physLive[0][i * nlive + j];
}

void QLens::test_lens_functions()
{
	clear_lenses();
	load_point_image_data("alphafit.dat");

	//
	/*
	SPLE_Lens *A = new SPLE_Lens();
	A->initialize_parameters(4.5,1,0,0.8,30,0.7,0.3);
	boolvector flags(7);
	flags[0] = true;
	flags[1] = false;
	flags[2] = false;
	flags[3] = true;
	flags[4] = true;
	flags[5] = true;
	flags[6] = true;
	//flags[7] = true;
	//param_list->print_penalty_limits();
	A->set_vary_flags(flags);
	Shear *S = new Shear();
	S->initialize_parameters(0.02,10,0,0);
	boolvector flag2(4);
	flag2[0] = true;
	flag2[1] = true;
	flag2[2] = false;
	flag2[3] = false;
	S->set_vary_flags(flag2);
	add_lens(A);
	add_lens(S);
	use_analytic_bestfit_src = true;
	include_flux_chisq = true;

	chi_square_fit_simplex();
	adopt_model(bestfitparams);

	bool status;
	vector<PointSource> imgsets = get_fit_imagesets(status);
	*/

	// The following shows how to access the image data in the "imgset" object
	/*
	cout << endl;
	for (int j=0; j < imgsets.size(); j++) {
		cout << "Source " << j << ": redshift = " << imgsets[j].zsrc << endl;
		cout << "Number of images: " << imgsets[j].n_images << endl;
		cout << "Source:  " << imgsets[j].src[0] << " " << imgsets[j].src[1] << endl;
		for (int i=0; i < imgsets[j].n_images; i++) cout << "Image" << i << ": " << imgsets[j].images[i].pos[0] << " " << imgsets[j].images[i].pos[1] << " " << imgsets[j].images[i].mag << " " << imgsets[j].imgflux(i) << endl; 
		cout << endl;
	}

	vector<PtImageDataSet> imgdatasets = export_to_ImageDataSet();
	cout << "Image Data:" << endl;
	for (int j=0; j < imgdatasets.size(); j++) {
		cout << "Source " << j << ": redshift = " << imgdatasets[j].zsrc << endl;
		cout << "Number of images: " << imgdatasets[j].n_images << endl;
		for (int i=0; i < imgdatasets[j].n_images; i++) cout << "Image" << i << ": " << imgdatasets[j].images[i].pos[0] << " " << imgdatasets[j].images[i].pos[1] << " " << imgdatasets[j].images[i].flux << endl; 
		cout << endl;
	}
	*/



	//OR...you can print similar information by calling the following function:
	//imgset.print();

	/*
	cout << "Generating critical curves/caustics and plotting to files 'crit.dat' and 'caust.dat'..." << endl;
	plot_sorted_critical_curves(); // generates critical curves/caustics and stores them
	// The following shows how the critical curves/caustics are accessed using "sorted_critical_curve", which is a std::vector of
	// "critical_curve" objects (in qlens.h you can see what the critical_curve structure looks like). The length of sorted_critical_curve vector
	// tells you how many distinct critical curves there are (sometimes there is one, or two, or more...it depends on the lens).
	ofstream ccfile("crit.dat");
	ofstream caustic_file("caust.dat");
	int i,j;
	for (i=0; i < sorted_critical_curve.size(); i++ ) {
		for (j=0; j < sorted_critical_curve[i].cc_pts.size(); j++) {
			ccfile << sorted_critical_curve[i].cc_pts[j][0] << " " << sorted_critical_curve[i].cc_pts[j][1] << endl; // printing x- and y-values for critical curves
			caustic_file << sorted_critical_curve[i].caustic_pts[j][0] << " " << sorted_critical_curve[i].caustic_pts[j][1] << endl; // same for caustics
		}
		ccfile << endl;
		caustic_file << endl;
	}
	*/

	//delete A;
}
