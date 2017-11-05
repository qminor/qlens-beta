// DOES THIS WORK?
#include "qlens.h"
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

const double Lens::default_autogrid_initial_step = 1.0e-3;
const double Lens::default_autogrid_rmin = 1.0e-5;
const double Lens::default_autogrid_rmax = 1.0e6;
const double Lens::default_autogrid_frac = 1.95; // ****** NOTE: it might be better to make this depend on the axis ratio, since for q=1 you may need larger rfrac
const int Lens::max_cc_search_iterations = 8;
double Lens::galsubgrid_radius_fraction; // radius of satellite subgridding in terms of fraction of Einstein radius
double Lens::galsubgrid_min_cellsize_fraction; // minimum cell size for satellite subgridding in terms of fraction of Einstein radius
int Lens::galsubgrid_cc_splittings;
bool Lens::auto_store_cc_points;
const double Lens::satellite_einstein_radius_fraction = 0.2;
const double Lens::default_rmin_frac = 1e-4;
bool Lens::warnings;
bool Lens::newton_warnings; // newton_warnings: when true, displays warnings when Newton's method fails or returns anomalous results
bool Lens::use_scientific_notation;
double Lens::rmin_frac;
bool Lens::respline_at_end; // for creating deflection spline
int Lens::resplinesteps; // for creating deflection spline
int Lens::ncfound = 0; // for plotting the lens equation x/y components
string Lens::fit_output_filename;

int Lens::nthreads;
lensvector *Lens::defs, *Lens::defs_i;
lensmatrix *Lens::jacs, *Lens::hesses, *Lens::hesses_i;
int *Lens::indxs;

void Lens::allocate_multithreaded_variables(const int& threads)
{
	nthreads = threads;
	defs = new lensvector[nthreads];
	defs_i = new lensvector[nthreads];
	jacs = new lensmatrix[nthreads];
	hesses = new lensmatrix[nthreads];
	hesses_i = new lensmatrix[nthreads];
}

void Lens::deallocate_multithreaded_variables()
{
	delete[] defs;
	delete[] defs_i;
	delete[] jacs;
	delete[] hesses;
	delete[] hesses_i;
}

#ifdef USE_MUMPS
DMUMPS_STRUC_C *Lens::mumps_solver;

void Lens::setup_mumps()
{
	mumps_solver = new DMUMPS_STRUC_C;
	mumps_solver->par = 1; //host machine participates in calculation
}
#endif

void Lens::delete_mumps()
{
#ifdef USE_MUMPS
	delete mumps_solver;
#endif
}

#ifdef USE_MPI
void Lens::set_mpi_params(const int& mpi_id_in, const int& mpi_np_in, const int& mpi_ngroups_in, const int& group_num_in, const int& group_id_in, const int& group_np_in, MPI_Group* group_in, MPI_Comm* comm, MPI_Group* mygroup, MPI_Comm* mycomm)
{
	mpi_id = mpi_id_in;
	mpi_np = mpi_np_in;
	mpi_ngroups = mpi_ngroups_in;
	group_id = group_id_in;
	group_num = group_num_in;
	group_np = group_np_in;
	mpi_group = group_in;
	group_comm = comm;
	my_group = mygroup;
	my_comm = mycomm;
#ifdef USE_MUMPS
	setup_mumps();
#endif
}
#endif

void Lens::set_mpi_params(const int& mpi_id_in, const int& mpi_np_in)
{
	// This assumes only one 'group', so all MPI processes will work together for each likelihood evaluation
	mpi_id = mpi_id_in;
	mpi_np = mpi_np_in;
	mpi_ngroups = 1;
	group_id = mpi_id;
	group_num = 0;
	group_np = mpi_np;

#ifdef USE_MPI
	MPI_Comm_group(MPI_COMM_WORLD, mpi_group);
	MPI_Comm_create(MPI_COMM_WORLD, *mpi_group, group_comm);
#ifdef USE_MUMPS
	setup_mumps();
#endif
#endif
}

Lens::Lens() : UCMC()
{
	mpi_id = 0;
	mpi_np = 1;
	group_np = 1;
	group_id = 0;
#ifdef USE_MPI
	mpi_group = NULL;
#endif

	omega_matter = 0.3;
	hubble = 0.7;
	set_cosmology(omega_matter,0.04,hubble,2.215);
	lens_redshift = 0.5;
	source_redshift = 2.0;
	auto_zsource_scaling = true; // this automatically sets the reference source redshift (for kappa scaling) equal to the source redshift being used
	reference_source_redshift = 2.0; // this is the source redshift with respect to which the lens models are defined
	reference_zfactor = 1.0; // this is the scaling for lensing quantities if the source redshift is different from the reference value
	source_redshifts = NULL;
	zfactors = NULL;
	vary_hubble_parameter = false;
	hubble_lower_limit = 1e30; // These must be specified by user
	hubble_upper_limit = 1e30; // These must be specified by user

	chisq_it=0;
	chisq_bestfit = 1e30;
	bestfit_flux = 0;
	display_chisq_status = false;
	chisq_display_frequency = 100; // Number of chi-square evaluations before displaying chi-square on screen
	show_wtime = false;
	terminal = TEXT;
	verbal_mode = true;
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
	n_mcpoints = 1000; // for nested sampling
	mcmc_threads = 1;
	mcmc_tolerance = 1.01; // Gelman-Rubin statistic for T-Walk sampler
	mcmc_logfile = false;
	open_chisq_logfile = false;
	psf_convolution_mpi = false;
	use_input_psf_matrix = false;
	psf_threshold = 1e-3;
	n_image_prior = false;
	n_image_threshold = 3.5; // ************THIS SHOULD BE SPECIFIED BY THE USER, AND ONLY GETS USED IF n_image_prior IS SET TO 'TRUE'
	max_sb_prior_unselected_pixels = false;
	max_sb_frac_unselected_pixels = 0.3; // ********ALSO SHOULD BE SPECIFIED BY THE USER, AND ONLY GETS USED IF max_sb_prior_unselected_pixels IS SET TO 'TRUE'
	subhalo_prior = false; // if on, this prior constrains any subhalos (with Pseudo-Jaffe profiles) to be positioned within the designated fit area (selected fit pixels only)
	nlens = 0;
	n_sb = 0;
	radial_grid = true;
	grid_xlength = 20; // default gridsize
	grid_ylength = 20;
	grid_xcenter = 0;
	grid_ycenter = 0;
	rmin_frac = default_rmin_frac;
	plot_ptsize = 0.3;
	plot_pttype = 7;

	fit_output_filename = "fit";
	auto_save_bestfit = false;
	fitmodel = NULL;
#ifdef USE_FITS
	fits_format = true;
#else
	fits_format = false;
#endif
	data_pixel_size = -1; // used for setting a pixel scale for FITS images (only if initialized to a positive number)
	n_fit_parameters = 0;
	n_sourcepts_fit = 0;
	sourcepts_fit = NULL;
	sourcepts_lower_limit = NULL;
	sourcepts_upper_limit = NULL;
	vary_sourcepts_x = NULL;
	vary_sourcepts_y = NULL;
	borrowed_image_data = false;
	image_data = NULL;
	defspline = NULL;

	source_fit_mode = Point_Source;
	chisq_tolerance = 1e-3;
	n_repeats = 1;
	calculate_parameter_errors = true;
	use_image_plane_chisq2 = false;
	use_image_plane_chisq = false;
	use_magnification_in_chisq = false;
	use_magnification_in_chisq_during_repeats = true;
	include_central_image = true;
	include_flux_chisq = false;
	include_parity_in_chisq = false;
	include_time_delay_chisq = false;
	use_analytic_bestfit_src = true; // analytic source solution only applicable if source plane chi-square is being used
	n_images_penalty = false;
	fix_source_flux = false;
	source_flux = 1.0;
	param_settings = new ParamSettings;
	sim_err_pos = 0.1;
	sim_err_flux = 0.1;
	sim_err_td = 0.1;

	image_pixel_data = NULL;
	image_pixel_grid = NULL;
	source_pixel_grid = NULL;
	sourcegrid_xmin = -1;
	sourcegrid_xmax = 1;
	sourcegrid_ymin = -1;
	sourcegrid_ymax = 1;
	sourcegrid_limit_xmin = -1e30;
	sourcegrid_limit_xmax = 1e30;
	sourcegrid_limit_ymin = -1e30;
	sourcegrid_limit_ymax = 1e30;
	auto_sourcegrid = true;
	ray_tracing_method = Interpolate;
#ifdef USE_MUMPS
	inversion_method = MUMPS;
#else
#ifdef USE_UMFPACK
	inversion_method = UMFPACK;
#else
	inversion_method = CG_Method;
#endif
#endif
	parallel_mumps = false;
	show_mumps_info = false;
	regularization_method = None;
	regularization_parameter = 0.5;
	regularization_parameter_lower_limit = 1e30; // These must be specified by user
	regularization_parameter_upper_limit = 1e30; // These must be specified by user
	vary_regularization_parameter = false;
	psf_width_x = 0;
	psf_width_y = 0;
	data_pixel_noise = 0;
	sim_pixel_noise = 0;
	sb_threshold = 0;
	noise_threshold = 1.3;
	n_image_pixels_x = 200;
	n_image_pixels_y = 200;
	srcgrid_npixels_x = 50;
	srcgrid_npixels_y = 50;
	auto_srcgrid_npixels = true;
	auto_srcgrid_set_pixel_size = false; // this feature is not working at the moment, so keep it off
	pixel_fraction = 0.3; // this should not be used if adaptive grid is being used
	pixel_fraction_lower_limit = 1e30; // These must be specified by user
	pixel_fraction_upper_limit = 1e30; // These must be specified by user
	vary_pixel_fraction = false; // varying the pixel fraction doesn't work if regularization is also varied (with source pixel regularization)
	Fmatrix = NULL;
	Fmatrix_index = NULL;
	Rmatrix = NULL;
	Rmatrix_index = NULL;
	Dvector = NULL;
	image_surface_brightness = NULL;
	source_surface_brightness = NULL;
	source_pixel_n_images = NULL;
	active_image_pixel_i = NULL;
	active_image_pixel_j = NULL;
	image_pixel_location_Lmatrix = NULL;
	source_pixel_location_Lmatrix = NULL;
	Lmatrix = NULL;
	Lmatrix_index = NULL;
	psf_matrix = NULL;
	inversion_nthreads = 1;
	adaptive_grid = false;
	pixel_magnification_threshold = 6;
	pixel_magnification_threshold_lower_limit = 1e30; // These must be specified by user
	pixel_magnification_threshold_upper_limit = 1e30; // These must be specified by user
	vary_magnification_threshold = false;
	exclude_source_pixels_beyond_fit_window = true;
	activate_unmapped_source_pixels = true;
	regrid_if_unmapped_source_subpixels = false;

	use_cc_spline = false;
	auto_ccspline = false;
	plot_critical_curves = &Lens::plot_sorted_critical_curves;
	cc_rmin = default_autogrid_rmin;
	cc_rmax = default_autogrid_rmax;
	source_plane_rscale = 0; // this must be found by the autogrid
	autogrid_frac = default_autogrid_frac;

	// the following variables are only relevant if use_cc_spline is on (so critical curves are splined)
	cc_thetasteps = 200;
	cc_splined = false;
	effectively_spherical = false;

	// parameters for the recursive grid
	enforce_min_cell_area = true;
	min_cell_area = 1e-6;
	rsplit_initial = 16; // initial number of cell divisions in the r-direction
	thetasplit_initial = 24; // initial number of cell divisions in the theta-direction
	splitlevels = 0; // number of times grid squares are recursively split (by default)...setting to zero is best, recursion slows down grid creation & searching
	cc_splitlevels = 2; // number of times grid squares are recursively split when containing a critical curve
	cc_neighbor_splittings = false;
	subgrid_around_satellites = true;
	galsubgrid_radius_fraction = 1.3;
	galsubgrid_min_cellsize_fraction = 0.1;
	galsubgrid_cc_splittings = 1;
	sorted_critical_curves = false;
	n_singular_points = 0;
	auto_store_cc_points = true;
	newton_magnification_threshold = 1000;

	warnings = true;
	newton_warnings = false;
	use_scientific_notation = true;
	include_time_delays = false;
	autocenter = true; // this option tells qlens to center the grid on a particular lens (given by autocenter_lens_number)
	auto_gridsize_from_einstein_radius = true; // this option tells qlens to set the grid size based on the Einstein radius of a particular lens (given by autocenter_lens_number)
	auto_gridsize_multiple_of_Re = 1.9;
	autogrid_before_grid_creation = false; // this option (if set to true) tells qlens to optimize the grid size & position automatically (using autogrid) when grid is created
	autocenter_lens_number = 0;
	spline_frac = 1.8;
	grid = NULL;
	Gauss_NN = 20;
	romberg_accuracy = 1e-6;
	LensProfile::integral_method = Gaussian_Quadrature;
	LensProfile::orient_major_axis_north = true;
	Shear::use_shear_component_params = false;
	use_mumps_subcomm = false;
}

Lens::Lens(Lens *lens_in) : UCMC() // creates lens object with same settings as input lens; does NOT import the lens/source model configurations, however
{
	verbal_mode = lens_in->verbal_mode;
	chisq_it=0;
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
#ifdef USE_MPI
	group_comm = lens_in->group_comm;
	mpi_group = lens_in->mpi_group;
	my_comm = lens_in->my_comm;
	my_group = lens_in->my_group;
#endif

	omega_matter = lens_in->omega_matter;
	hubble = lens_in->hubble;
	set_cosmology(omega_matter,0.04,hubble,2.215);
	lens_redshift = lens_in->lens_redshift;
	source_redshift = lens_in->source_redshift;
	auto_zsource_scaling = lens_in->auto_zsource_scaling;
	reference_source_redshift = lens_in->reference_source_redshift; // this is the source redshift with respect to which the lens models are defined
	reference_zfactor = lens_in->reference_zfactor; // this is the scaling for lensing quantities if the source redshift is different from the reference value
	source_redshifts = NULL;
	zfactors = NULL;
	vary_hubble_parameter = lens_in->vary_hubble_parameter;
	hubble_lower_limit = lens_in->hubble_lower_limit; // These must be specified by user
	hubble_upper_limit = lens_in->hubble_upper_limit; // These must be specified by user

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
	n_mcpoints = lens_in->n_mcpoints; // for nested sampling
	mcmc_tolerance = lens_in->mcmc_tolerance; // for T-Walk sampler
	mcmc_logfile = lens_in->mcmc_logfile;
	open_chisq_logfile = lens_in->open_chisq_logfile;
	psf_convolution_mpi = lens_in->psf_convolution_mpi;
	use_input_psf_matrix = lens_in->use_input_psf_matrix;
	psf_threshold = lens_in->psf_threshold;
	n_image_prior = lens_in->n_image_prior;
	n_image_threshold = lens_in->n_image_threshold;
	max_sb_prior_unselected_pixels = lens_in->max_sb_prior_unselected_pixels;
	max_sb_frac_unselected_pixels = lens_in->max_sb_frac_unselected_pixels;
	subhalo_prior = lens_in->subhalo_prior;

	plot_ptsize = lens_in->plot_ptsize;
	plot_pttype = lens_in->plot_pttype;

	nlens = 0;
	n_sb = 0;
	radial_grid = lens_in->radial_grid;
	grid_xlength = lens_in->grid_xlength; // default gridsize
	grid_ylength = lens_in->grid_ylength;
	grid_xcenter = lens_in->grid_xcenter;
	grid_ycenter = lens_in->grid_ycenter;

	LogLikePtr = static_cast<double (UCMC::*)(double *)> (&Lens::fitmodel_loglike_point_source); // is this line necessary?
	source_fit_mode = lens_in->source_fit_mode;
	chisq_tolerance = lens_in->chisq_tolerance;
	n_repeats = lens_in->n_repeats;
	calculate_parameter_errors = lens_in->calculate_parameter_errors;
	use_image_plane_chisq2 = lens_in->use_image_plane_chisq;
	use_image_plane_chisq = lens_in->use_image_plane_chisq2;
	use_magnification_in_chisq = lens_in->use_magnification_in_chisq;
	use_magnification_in_chisq_during_repeats = lens_in->use_magnification_in_chisq_during_repeats;
	include_central_image = lens_in->include_central_image;
	include_flux_chisq = lens_in->include_flux_chisq;
	include_parity_in_chisq = lens_in->include_parity_in_chisq;
	include_time_delay_chisq = lens_in->include_time_delay_chisq;
	use_analytic_bestfit_src = lens_in->use_analytic_bestfit_src; // analytic source solution only applicable if source plane chi-square is being used
	n_images_penalty = lens_in->n_images_penalty;
	fix_source_flux = lens_in->fix_source_flux;
	source_flux = lens_in->source_flux;
	param_settings = new ParamSettings(*lens_in->param_settings);
	sim_err_pos = lens_in->sim_err_pos;
	sim_err_flux = lens_in->sim_err_flux;
	sim_err_td = lens_in->sim_err_td;

	fitmodel = NULL;
	fits_format = lens_in->fits_format;
	data_pixel_size = lens_in->data_pixel_size;
	n_fit_parameters = 0;
	n_sourcepts_fit = 0;
	sourcepts_fit = NULL;
	sourcepts_lower_limit = NULL;
	sourcepts_upper_limit = NULL;
	vary_sourcepts_x = NULL;
	vary_sourcepts_y = NULL;
	borrowed_image_data = false;
	image_data = NULL;
	defspline = NULL;

	image_pixel_data = NULL;
	image_pixel_grid = NULL;
	source_pixel_grid = NULL;
	sourcegrid_xmin = lens_in->sourcegrid_xmin;
	sourcegrid_xmax = lens_in->sourcegrid_xmax;
	sourcegrid_ymin = lens_in->sourcegrid_ymin;
	sourcegrid_ymax = lens_in->sourcegrid_ymax;
	sourcegrid_limit_xmin = lens_in->sourcegrid_limit_xmin;
	sourcegrid_limit_xmax = lens_in->sourcegrid_limit_xmax;
	sourcegrid_limit_ymin = lens_in->sourcegrid_limit_ymin;
	sourcegrid_limit_ymax = lens_in->sourcegrid_limit_ymax;
	auto_sourcegrid = lens_in->auto_sourcegrid;
	regularization_method = lens_in->regularization_method;
	regularization_parameter = lens_in->regularization_parameter;
	regularization_parameter_lower_limit = 1e30;
	regularization_parameter_upper_limit = 1e30;
	vary_regularization_parameter = lens_in->vary_regularization_parameter;
	ray_tracing_method = lens_in->ray_tracing_method;
	inversion_method = lens_in->inversion_method;
	parallel_mumps = lens_in->parallel_mumps;
	show_mumps_info = lens_in->show_mumps_info;

	psf_width_x = lens_in->psf_width_x;
	psf_width_y = lens_in->psf_width_y;
	data_pixel_noise = lens_in->data_pixel_noise;
	sim_pixel_noise = lens_in->sim_pixel_noise;
	sb_threshold = lens_in->sb_threshold;
	noise_threshold = lens_in->noise_threshold;
	n_image_pixels_x = lens_in->n_image_pixels_x;
	n_image_pixels_y = lens_in->n_image_pixels_y;
	n_image_pixels_x = lens_in->n_image_pixels_x;
	n_image_pixels_y = lens_in->n_image_pixels_y;
	srcgrid_npixels_x = lens_in->srcgrid_npixels_x;
	srcgrid_npixels_y = lens_in->srcgrid_npixels_y;
	auto_srcgrid_npixels = lens_in->auto_srcgrid_npixels;
	auto_srcgrid_set_pixel_size = lens_in->auto_srcgrid_set_pixel_size;

	pixel_fraction = lens_in->pixel_fraction;
	vary_pixel_fraction = lens_in->vary_pixel_fraction;
	Dvector = NULL;
	Fmatrix = NULL;
	Fmatrix_index = NULL;
	Rmatrix = NULL;
	Rmatrix_index = NULL;
	image_surface_brightness = NULL;
	source_surface_brightness = NULL;
	source_pixel_n_images = NULL;
	active_image_pixel_i = NULL;
	active_image_pixel_j = NULL;
	Lmatrix_index = NULL;
	if (lens_in->psf_matrix==NULL) psf_matrix = NULL;
	else {
		psf_npixels_x = lens_in->psf_npixels_x;
		psf_npixels_y = lens_in->psf_npixels_y;
		psf_matrix = new double*[psf_npixels_x];
		int i,j;
		for (i=0; i < psf_npixels_x; i++) {
			psf_matrix[i] = new double[psf_npixels_y];
			for (j=0; j < psf_npixels_y; j++) psf_matrix[i][j] = lens_in->psf_matrix[i][j];
		}
	}
	image_pixel_location_Lmatrix = NULL;
	source_pixel_location_Lmatrix = NULL;
	Lmatrix = NULL;
	inversion_nthreads = lens_in->inversion_nthreads;
	adaptive_grid = lens_in->adaptive_grid;
	pixel_magnification_threshold = lens_in->pixel_magnification_threshold;
	vary_magnification_threshold = lens_in->vary_magnification_threshold;
	exclude_source_pixels_beyond_fit_window = lens_in->exclude_source_pixels_beyond_fit_window;
	activate_unmapped_source_pixels = lens_in->activate_unmapped_source_pixels;
	regrid_if_unmapped_source_subpixels = lens_in->regrid_if_unmapped_source_subpixels;

	use_cc_spline = lens_in->use_cc_spline;
	auto_ccspline = lens_in->auto_ccspline;
	plot_critical_curves = &Lens::plot_sorted_critical_curves;
	cc_rmin = lens_in->cc_rmin;
	cc_rmax = lens_in->cc_rmax;
	source_plane_rscale = lens_in->source_plane_rscale;
	autogrid_frac = lens_in->autogrid_frac;

	// the following variables are only relevant if use_cc_spline is on (so critical curves are splined)
	cc_thetasteps = lens_in->cc_thetasteps;
	cc_splined = false;
	effectively_spherical = false;

	// parameters for the recursive grid
	enforce_min_cell_area = lens_in->enforce_min_cell_area;
	min_cell_area = lens_in->min_cell_area;
	rsplit_initial = lens_in->rsplit_initial; // initial number of cell divisions in the r-direction
	thetasplit_initial = lens_in->thetasplit_initial; // initial number of cell divisions in the theta-direction
	splitlevels = lens_in->splitlevels; // number of times grid squares are recursively split (by default)...minimum of one splitting is required
	cc_splitlevels = lens_in->cc_splitlevels; // number of times grid squares are recursively split when containing a critical curve
	cc_neighbor_splittings = lens_in->cc_neighbor_splittings;
	subgrid_around_satellites = lens_in->subgrid_around_satellites;
	galsubgrid_radius_fraction = lens_in->galsubgrid_radius_fraction;
	galsubgrid_min_cellsize_fraction = lens_in->galsubgrid_min_cellsize_fraction;
	galsubgrid_cc_splittings = lens_in->galsubgrid_cc_splittings;
	sorted_critical_curves = false;
	auto_store_cc_points = lens_in->auto_store_cc_points;
	n_singular_points = lens_in->n_singular_points;
	newton_magnification_threshold = lens_in->newton_magnification_threshold;

	include_time_delays = lens_in->include_time_delays;
	autocenter = lens_in->autocenter;
	autocenter_lens_number = lens_in->autocenter_lens_number;
	auto_gridsize_from_einstein_radius = lens_in->auto_gridsize_from_einstein_radius;
	auto_gridsize_multiple_of_Re = lens_in->auto_gridsize_multiple_of_Re;
	autogrid_before_grid_creation = lens_in->autogrid_before_grid_creation; // this option (if set to true) tells qlens to optimize the grid size & position automatically when grid is created
	spline_frac = lens_in->spline_frac;
	grid = NULL;
	Gauss_NN = lens_in->Gauss_NN;
	romberg_accuracy = lens_in->romberg_accuracy;
	use_mumps_subcomm = lens_in->use_mumps_subcomm;
}

void Lens::add_lens(LensProfileName name, const double mass_parameter, const double scale1, const double scale2, const double q, const double theta, const double xc, const double yc, const double extra_param1, const double extra_param2, const bool optional_setting)
{
	LensProfile** newlist = new LensProfile*[nlens+1];
	if (nlens > 0) {
		for (int i=0; i < nlens; i++)
			newlist[i] = lens_list[i];
		delete[] lens_list;
	}

	switch (name) {
		case PTMASS:
			newlist[nlens] = new PointMass(mass_parameter, xc, yc); break;
		case SHEET:
			newlist[nlens] = new MassSheet(mass_parameter, xc, yc); break;
		case ALPHA:
			newlist[nlens] = new Alpha(mass_parameter, scale1, scale2, q, theta, xc, yc, Gauss_NN, romberg_accuracy); break;
		case SHEAR:
			newlist[nlens] = new Shear(q, theta, xc, yc); break;
		// Note: the Multipole profile is added using the function add_multipole_lens(...) because one of the input parameters is an int
		case nfw:
			newlist[nlens] = new NFW(mass_parameter, scale1, q, theta, xc, yc, Gauss_NN, romberg_accuracy); break;
		case TRUNCATED_nfw:
			newlist[nlens] = new Truncated_NFW(mass_parameter, scale1, scale2, q, theta, xc, yc, Gauss_NN, romberg_accuracy); break;
		case PJAFFE:
			newlist[nlens] = new PseudoJaffe(mass_parameter, scale1, scale2, q, theta, xc, yc, Gauss_NN, romberg_accuracy); break;
		case EXPDISK:
			newlist[nlens] = new ExpDisk(mass_parameter, scale1, q, theta, xc, yc, Gauss_NN, romberg_accuracy); break;
		case HERNQUIST:
			newlist[nlens] = new Hernquist(mass_parameter, scale1, q, theta, xc, yc, Gauss_NN, romberg_accuracy); break;
		case CORECUSP:
			if ((extra_param1==-1000) or (extra_param2==-1000)) die("extra parameters need to be passed to add_lens(...) function for model CORECUSP");
			newlist[nlens] = new CoreCusp(mass_parameter, extra_param1, extra_param2, scale1, scale2, q, theta, xc, yc, Gauss_NN, romberg_accuracy, optional_setting); break;
		case SERSIC_LENS:
			newlist[nlens] = new SersicLens(mass_parameter, scale1, scale2, q, theta, xc, yc, Gauss_NN, romberg_accuracy); break;
		case TESTMODEL: // Model for testing purposes
			newlist[nlens] = new TestModel(q, theta, xc, yc, Gauss_NN, romberg_accuracy); break;
		default:
			die("Lens type not recognized");
	}
	nlens++;
	lens_list = newlist;
	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset();
	if (auto_ccspline) automatically_determine_ccspline_mode();
}

void Lens::add_lens(const char *splinefile, const double q, const double theta, const double qx, const double f, const double xc, const double yc)
{
	LensProfile** newlist = new LensProfile*[nlens+1];
	if (nlens > 0) {
		for (int i=0; i < nlens; i++)
			newlist[i] = lens_list[i];
		delete[] lens_list;
	}
	newlist[nlens++] = new LensProfile(splinefile, q, theta, xc, yc, Gauss_NN, romberg_accuracy, qx, f);

	lens_list = newlist;
	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset();
	if (auto_ccspline) automatically_determine_ccspline_mode();
}

void Lens::add_shear_lens(const double shear_p1, const double shear_p2, const double xc, const double yc)
{
	add_lens(SHEAR,0,0,0,shear_p1,shear_p2,xc,yc);
}

void Lens::add_ptmass_lens(const double mass_parameter, const double xc, const double yc)
{
	add_lens(PTMASS,mass_parameter,0,0,0,0,xc,yc);
}

void Lens::add_mass_sheet_lens(const double mass_parameter, const double xc, const double yc)
{
	add_lens(SHEET,mass_parameter,0,0,0,0,xc,yc);
}

void Lens::add_multipole_lens(int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool kap, bool sine_term)
{
	LensProfile** newlist = new LensProfile*[nlens+1];
	if (nlens > 0) {
		for (int i=0; i < nlens; i++)
			newlist[i] = lens_list[i];
		delete[] lens_list;
	}
	newlist[nlens++] = new Multipole(a_m, n, m, theta, xc, yc, kap, sine_term);

	lens_list = newlist;
	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset();
	if (auto_ccspline) automatically_determine_ccspline_mode();
}

void Lens::remove_lens(int lensnumber)
{
	if ((lensnumber >= nlens) or (nlens==0)) { warn(warnings,"Specified lens does not exist"); return; }
	LensProfile** newlist = new LensProfile*[nlens-1];
	int i,j;
	for (i=0; i < nlens; i++) {
		if ((i != lensnumber) and (lens_list[i]->anchored==true) and (lens_list[i]->get_anchor_number()==lensnumber)) lens_list[i]->delete_anchor();
		if ((i != lensnumber) and (lens_list[i]->anchor_extra_parameter==true) and (lens_list[i]->get_parameter_anchor_number()==lensnumber)) lens_list[i]->delete_parameter_anchor();
	}
	for (i=0,j=0; i < nlens; i++)
		if (i != lensnumber) { newlist[j] = lens_list[i]; j++; }
	delete[] lens_list;
	nlens--;

	lens_list = newlist;
	for (int i=0; i < nlens; i++) lens_list[i]->lens_number = i;
	reset();
	if (auto_ccspline) automatically_determine_ccspline_mode();
}

void Lens::clear_lenses()
{
	if (nlens > 0) {
		for (int i=0; i < nlens; i++)
			delete lens_list[i];
		delete[] lens_list;
		nlens = 0;
	}
	reset();
}

void Lens::clear()
{
	if (nlens > 0) {
		for (int i=0; i < nlens; i++)
			delete lens_list[i];
		delete[] lens_list;
		nlens = 0;
	}
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			delete sb_list[i];
		delete[] sb_list;
		n_sb = 0;
	}
	reset();
}

void Lens::toggle_major_axis_along_y(bool major_axis_along_y)
{
	if (LensProfile::orient_major_axis_north != major_axis_along_y) {
		LensProfile::orient_major_axis_north = major_axis_along_y;
		if (major_axis_along_y) {
			if (nlens > 0) {
				for (int i=0; i < nlens; i++) lens_list[i]->shift_angle_minus_90();
			}
		} else {
			if (nlens > 0) {
				for (int i=0; i < nlens; i++) lens_list[i]->shift_angle_90();
			}
		}
	}
}

void Lens::automatically_determine_ccspline_mode()
{
	// this feature should be made obsolete. It's really not necessary to spline the critical curves anymore
	bool kappa_present = false;
	for (int i=0; i < nlens; i++) {
		if ((autocenter==true) and (i==autocenter_lens_number)) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
		if ((lens_list[i]->get_lenstype() != SHEAR) and (lens_list[i]->get_lenstype() != MULTIPOLE)) kappa_present = true;
	}
	if (kappa_present==false) set_ccspline_mode(false);
	else if ((test_for_elliptical_symmetry()==true) and (test_for_singularity()==false)) set_ccspline_mode(true);
	else set_ccspline_mode(false);
}

bool Lens::test_for_elliptical_symmetry()
{
	bool elliptical_symmetry_present = true;
	if (nlens > 0) {
		double xc, yc;
		for (int i=0; i < nlens; i++) {
			lens_list[i]->get_center_coords(xc,yc);
			if ((abs(xc-grid_xcenter) > Grid::image_pos_accuracy) or (abs(yc-grid_ycenter) > Grid::image_pos_accuracy))
				elliptical_symmetry_present = false;
		}
	}
	return elliptical_symmetry_present;
}

bool Lens::test_for_singularity()
{
	// if kappa goes like r^n near the origin where n <= -1, then a radial critical curve will not form
	// (this is because the deflection, which goes like r^(n+1), must increase as you go outward in order
	// to have a radial critical curve; this will only happen for n>-1). Here we test for this for the
	// relevant models where this can occur
	bool singular = false;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			if ((lens_list[i]->get_lenstype() == PJAFFE) and (lens_list[i]->core_present()==false)) singular = true;
			else if ((lens_list[i]->get_lenstype() == ALPHA) and (lens_list[i]->get_inner_logslope() <= -1) and (lens_list[i]->core_present()==false)) singular = true;
			else if (lens_list[i]->get_lenstype() == PTMASS) singular = true;
				// a radial critical curve will occur if a core is present, OR if alpha > 1 (since kappa goes like r^n where n=alpha-2)
		}
	}
	return singular;
}

void Lens::record_singular_points()
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
			if ((lens_list[i]->get_lenstype() == PJAFFE) and (lens_list[i]->core_present()==false)) singular = true;
			else if ((lens_list[i]->get_lenstype() == ALPHA) and (lens_list[i]->get_inner_logslope() <= -1) and (lens_list[i]->core_present()==false)) singular = true;
			else if (lens_list[i]->get_lenstype() == PTMASS) singular = true;
				// a radial critical curve will occur if a core is present, OR if alpha > 1 (since kappa goes like r^n where n=alpha-2)
			if (singular) {
				lens_list[i]->get_center_coords(xc,yc);
				lensvector singular_pt(xc,yc);
				singular_pts.push_back(singular_pt);
			}
		}
	}
	n_singular_points = singular_pts.size();
}

void Lens::add_source_object(SB_ProfileName name, double sb_norm, double scale, double logslope_param, double q, double theta, double xc, double yc)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			newlist[i] = sb_list[i];
		delete[] sb_list;
	}

	switch (name) {
		case GAUSSIAN:
			newlist[n_sb] = new Gaussian(sb_norm, scale, q, theta, xc, yc); break;
		case SERSIC:
			newlist[n_sb] = new Sersic(sb_norm, scale, logslope_param, q, theta, xc, yc); break;
		case TOPHAT:
			newlist[n_sb] = new TopHat(sb_norm, scale, q, theta, xc, yc); break;
		default:
			die("Surface brightness profile type not recognized");
	}
	n_sb++;
	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}

void Lens::add_source_object(const char *splinefile, double q, double theta, double qx, double f, double xc, double yc)
{
	SB_Profile** newlist = new SB_Profile*[n_sb+1];
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			newlist[i] = sb_list[i];
		delete[] sb_list;
	}
	newlist[n_sb++] = new SB_Profile(splinefile, q, theta, xc, yc, qx, f);

	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}

void Lens::remove_source_object(int sb_number)
{
	if ((sb_number >= n_sb) or (n_sb == 0)) { warn(warnings,"Specified source object does not exist"); return; }
	SB_Profile** newlist = new SB_Profile*[n_sb-1];
	int i,j;
	for (i=0, j=0; i < n_sb; i++)
		if (i != sb_number) { newlist[j] = sb_list[i]; j++; }
	delete[] sb_list;
	n_sb--;

	sb_list = newlist;
	for (int i=0; i < n_sb; i++) sb_list[i]->sb_number = i;
}

void Lens::clear_source_objects()
{
	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			delete sb_list[i];
		delete[] sb_list;
		n_sb = 0;
	}
}

void Lens::print_source_list()
{
	if (mpi_id==0) {
		cout << resetiosflags(ios::scientific);
		if (n_sb > 0) {
			for (int i=0; i < n_sb; i++) {
				cout << i << ". ";
				sb_list[i]->print_parameters();
			}
		}
		else cout << "No source objects have been specified" << endl;
		if (use_scientific_notation) cout << setiosflags(ios::scientific);
	}
}

void Lens::set_gridcenter(double xc, double yc)
{
	grid_xcenter=xc;
	grid_ycenter=yc;
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
}

void Lens::set_gridsize(double xl, double yl)
{
	grid_xlength = xl;
	grid_ylength = yl;
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
}

void Lens::set_grid_corners(double xmin, double xmax, double ymin, double ymax)
{
	grid_xcenter = 0.5*(xmax+xmin);
	grid_ycenter = 0.5*(ymax+ymin);
	grid_xlength = xmax-xmin;
	grid_ylength = ymax-ymin;
	cc_rmax = 0.5*dmax(grid_xlength, grid_ylength);
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
}

void Lens::autogrid(double rmin, double rmax, double frac)
{
	cc_rmin = rmin;
	cc_rmax = rmax;
	autogrid_frac = frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

void Lens::autogrid(double rmin, double rmax)
{
	cc_rmin = rmin;
	cc_rmax = rmax;
	autogrid_frac = default_autogrid_frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

void Lens::autogrid() {
	cc_rmin = default_autogrid_rmin;
	cc_rmax = default_autogrid_rmax;
	autogrid_frac = default_autogrid_frac;
	if (nlens > 0) {
		if (find_optimal_gridsize()==false) warn(warnings,"could not find any critical curves");
		//else if (grid != NULL) reset_grid(); // if a grid was already in place, then delete the grid
	} else warn("cannot autogrid; no lens model has been specified");
}

bool Lens::create_grid(bool verbal, const double zfac)
{
	if (nlens==0) { warn(warnings, "no lens model is specified"); return false; }
	double mytime0, mytime;
#ifdef USE_OPENMP
	if (show_wtime) {
		mytime0=omp_get_wtime();
	}
#endif
	if (grid != NULL) {
		int rsp, thetasp;
		grid->get_usplit_initial(rsp);
		grid->get_wsplit_initial(thetasp);
		if ((rsp != rsplit_initial) or (thetasp != thetasplit_initial)) {
			delete grid;
			grid = NULL;
		}
		if ((auto_store_cc_points) and (use_cc_spline==false)) {
			critical_curve_pts.clear();
			caustic_pts.clear();
			length_of_cc_cell.clear();
			sorted_critical_curves = false;
			sorted_critical_curve.clear();
		}
	}
	double rmax = 0.5*dmax(grid_xlength,grid_ylength);
	record_singular_points(); // grid cells will split around singular points (e.g. center of point mass, etc.)

	Grid::set_splitting(rsplit_initial, thetasplit_initial, splitlevels, cc_splitlevels, min_cell_area, cc_neighbor_splittings);
	Grid::set_enforce_min_area(enforce_min_cell_area);
	Grid::set_lens(this);

	if (autogrid_before_grid_creation) autogrid();
	else {
		if (autocenter==true) {
			lens_list[autocenter_lens_number]->get_center_coords(grid_xcenter,grid_ycenter);
		}
		if (auto_gridsize_from_einstein_radius==true) {
			double re_major;
			re_major = einstein_radius_of_primary_lens(zfac);
			if (re_major != 0.0) {
				rmax = auto_gridsize_multiple_of_Re*re_major;
				grid_xlength = 2*rmax;
				grid_ylength = 2*rmax;
				cc_rmax = rmax;
			}
		}
	}

	if ((verbal) and (mpi_id==0)) cout << "Creating grid..." << flush;
	if (grid != NULL) {
		if (radial_grid)
			grid->redraw_grid(rmin_frac*rmax, rmax, grid_xcenter, grid_ycenter, 1, zfac); // setting grid_q to 1 for the moment...I will play with that later
		else
			grid->redraw_grid(grid_xcenter, grid_ycenter, grid_xlength, grid_ylength, zfac);
	} else {
		if (radial_grid)
			grid = new Grid(rmin_frac*rmax, rmax, grid_xcenter, grid_ycenter, 1, zfac); // setting grid_q to 1 for the moment...I will play with that later
		else
			grid = new Grid(grid_xcenter, grid_ycenter, grid_xlength, grid_ylength, zfac);
	}
	if (subgrid_around_satellites) subgrid_around_satellite_galaxies(zfac);
	if ((auto_store_cc_points==true) and (use_cc_spline==false)) grid->store_critical_curve_pts();
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

void Lens::subgrid_around_satellite_galaxies(const double zfac)
{
	if (grid==NULL) {
		if (create_grid(true,zfac)==false) die("Could not create recursive grid");
	}
	if (nlens==0) { warn(warnings,"No galaxies in lens lens_list"); return; }
	double largest_einstein_radius = 0, xch, ych;
	dvector einstein_radii(nlens);
	double re_avg; // won't use this
	for (int i=0; i < nlens; i++) {
		lens_list[i]->get_einstein_radius(einstein_radii[i],re_avg,reference_zfactor);
		if (einstein_radii[i] > largest_einstein_radius) {
			largest_einstein_radius = einstein_radii[i];
			lens_list[i]->get_center_coords(xch,ych);
		}
	}
	// lenses with Einstein radii < 0.25 times the largest Einstein radius, and not co-centered with the largest lens, are considered satellites.

	double xc,yc;
	lensvector center;
	int parity, n_satellites=0;
	double *kappas = new double[nlens];
	double *parities = new double[nlens];
	for (int i=0; i < nlens; i++) {
		if ((einstein_radii[i] > 0) and (einstein_radii[i] < satellite_einstein_radius_fraction*largest_einstein_radius)) {
			lens_list[i]->get_center_coords(xc,yc);
			// lenses co-centered with the primary lens, no matter how small, are not considered satellites
			if ((xc != xch) or (yc != ych)) {
				center[0]=xc;
				center[1]=yc;
				kappas[i] = kappa_exclude(center,i,zfac);
				parities[i] = sign(magnification_exclude(center,i,zfac)); // use the parity to help determine approx. size of critical curves
				// galaxies in positive-parity regions where kappa > 1 will form no critical curves, so don't subgrid around these
				if ((parities[i]==1) and (kappas[i] >= 1.0)) continue;
				else n_satellites++;
			}
		}
	}
	lensvector *galcenter = new lensvector[n_satellites];
	double *einstein_radius = new double[n_satellites];
	double *subgrid_radius = new double[n_satellites];
	double *min_galsubgrid_cellsize = new double[n_satellites];

	int j=0;
	double reavg; // won't use this
	double axis1, axis2, ratio;
	double shear_angle;
	double dr, theta, rmax, lambda_minus, dlambda_dr;
	double shear_at_center, kappa_at_center, cc_major_axis_factor;
	lensvector displaced_center;
	dr = 1e-5;
	for (int i=0; i < nlens; i++) {
		if ((einstein_radii[i] > 0) and (einstein_radii[i] < satellite_einstein_radius_fraction*largest_einstein_radius)) {
			lens_list[i]->get_center_coords(xc,yc);
			// lenses co-centered with the primary lens, no matter how small, are not considered satellites
			if ((xc != xch) or (yc != ych)) {
				kappa_at_center = kappas[i];
				parity = parities[i]; // use the parity to help determine approx. size of critical curves

				// galaxies in positive-parity regions where kappa > 1 will form no critical curves, so don't subgrid around these
				if ((parity==1) and (kappa_at_center >= 1.0)) continue;
				galcenter[j][0]=xc;
				galcenter[j][1]=yc;

				lens_list[i]->get_einstein_radius(einstein_radius[j],reavg,reference_zfactor);
				//shear_at_center = shear_exclude(galcenter[j],i,zfac);
				shear_exclude(galcenter[j],shear_at_center,shear_angle,i,zfac);
				shear_angle -= 90;
				shear_angle *= M_PI/180;

				lambda_minus = 1 - kappa_at_center - shear_at_center;
				displaced_center[0] = xc + dr*cos(shear_angle);
				displaced_center[1] = yc + dr*sin(shear_angle);
				dlambda_dr = ((1 - kappa_exclude(displaced_center,i,zfac) - shear_exclude(displaced_center,i,zfac)) - lambda_minus) / dr;
				if (dlambda_dr < 0) {
					shear_angle += M_PI;
					dlambda_dr = -dlambda_dr;
				}

				if (lambda_minus>0)
					rmax = (-lambda_minus + sqrt(lambda_minus*lambda_minus + 4*einstein_radius[j]*dlambda_dr))/(2*dlambda_dr);
				else
					rmax = (lambda_minus + sqrt(lambda_minus*lambda_minus + 4*einstein_radius[j]*dlambda_dr))/(2*dlambda_dr);

				if (parity==1) {
					// if the parity in the region is positive, typically we get two critical curves (or just one tangential critical curve);
					// the outer curve can be enlarged by the shear present, so we take this into account when choosing our subgrid radius
					cc_major_axis_factor = 1.1/abs(1-kappa_at_center-shear_at_center);
				} else {
					// only one (radial) critical curve will form, and its radius is comparable to the Einstein radius. However it can
					// be enlarged if it is near the radial critical curve, which we account for with the following fitting formula
					if (kappa_at_center > 1) {
						axis1 = 1.0/abs(1-kappa_at_center-shear_at_center);
						axis2 = 1.0/abs(1-kappa_at_center+shear_at_center);
						ratio = dmax(axis1,axis2)/dmin(axis1,axis2);
						cc_major_axis_factor = 2.5 + 0.37*(ratio-3.8);
					} else cc_major_axis_factor = 1.8;
				}

				subgrid_radius[j] = galsubgrid_radius_fraction*rmax;
				min_galsubgrid_cellsize[j] = SQR(galsubgrid_min_cellsize_fraction*rmax);
				//cout << "Galaxy " << i << ": kappa: " << kappa_at_center << ", shear: " << shear_at_center << ", axis ratio: " << ratio << ", axis1=" << axis1 << ", axis2=" << axis2 << ", major axis factor: " << cc_major_axis_factor << ", subgrid_radius=" << subgrid_radius[j] << endl;
				j++;
			}
		}
	}

	grid->subgrid_around_galaxies(galcenter,n_satellites,subgrid_radius,min_galsubgrid_cellsize,galsubgrid_cc_splittings);
	delete[] kappas;
	delete[] parities;
	delete[] galcenter;
	delete[] einstein_radius;
	delete[] subgrid_radius;
	delete[] min_galsubgrid_cellsize;
}

void Lens::plot_shear_field(double xmin, double xmax, int nx, double ymin, double ymax, int ny)
{
	int i, j, k;
	double x, y;
	double xstep = (xmax-xmin)/(nx-1);
	double ystep = (ymax-ymin)/(ny-1);
	double scale = 0.3*dmin(xstep,ystep);
	int compass_steps = 2;
	double compass_step = scale / (compass_steps-1);
	lensvector pos;
	double shearval,shear_angle,xp,yp,t;
	ofstream sout("shear.dat");
	for (i=0, x=xmin; i < nx; i++, x += xstep) {
		for (j=0, y=ymin; j < ny; j++, y += ystep) {
			pos[0]=x; pos[1]=y;
			shear(pos,shearval,shear_angle,0,reference_zfactor);
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

void Lens::calculate_critical_curve_deformation_radius(int lens_number, bool verbose, double &rmax, double& mass_enclosed)
{
	if ((lens_list[lens_number]->get_lenstype()!=PJAFFE) and (lens_list[lens_number]->get_lenstype()!=ALPHA))
	{
		calculate_critical_curve_deformation_radius_numerical(lens_number,verbose,rmax,mass_enclosed);
		return;
	}
	//this assumes the host halo is lens number 0 (and is centered at the origin), and corresponding external shear (if present) is lens number 1
	double xc, yc, b, alpha, bs, rt, dum, q, shear_ext, phi, phi_0, phi_p, theta_s;
	double host_xc, host_yc;
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
	if (lens_list[lens_number]->get_lenstype()==PJAFFE) {
		is_pjaffe = true;
		dvector params;
		lens_list[lens_number]->get_extra_params(params);
		bs = params[0];
		rt = params[1];
	} else {
		is_pjaffe = false;
		lens_list[lens_number]->get_einstein_radius(dum,bs,reference_zfactor);
	}
	dvector host_params;
	lens_list[0]->get_extra_params(host_params);
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
	subhalo_lens_number = lens_number;
	subhalo_center[0]=xc; subhalo_center[1]=yc;
	shear_exclude(subhalo_center,totshear,shear_angle,subhalo_lens_number,reference_zfactor);
	theta_shear = degrees_to_radians(shear_angle);
	theta_shear -= M_PI/2.0;
	double (Brent::*dthetac_eq)(const double);
	dthetac_eq = static_cast<double (Brent::*)(const double)> (&Lens::subhalo_perturbation_radius_equation);
	double bound = 2*sqrt(b*bs);
	rmax_numerical = abs(BrentsMethod_Inclusive(dthetac_eq,-bound,bound,1e-5));
	double avg_kappa = reference_zfactor*lens_list[subhalo_lens_number]->kappa_avg_r(rmax_numerical);
	double menc = avg_kappa*M_PI*SQR(rmax_numerical)*4.59888e10;
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

void Lens::calculate_critical_curve_deformation_radius_numerical(int lens_number, bool verbose, double& rmax_numerical, double& mass_enclosed)
{
	//this assumes the host halo is lens number 0 (and is centered at the origin), and corresponding external shear (if present) is lens number 1
	double xc, yc, theta_s, host_xc, host_yc, b, dum, alpha, shear_ext, phi, phi_p, eta;
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

	lens_list[0]->get_einstein_radius(dum,b,reference_zfactor);
	dvector host_params;
	lens_list[0]->get_extra_params(host_params);
	alpha = host_params[1];

	if (lens_list[1]->get_lenstype()==SHEAR) lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
	else { shear_ext = 0; phi_p=0; }
	if (LensProfile::orient_major_axis_north==true) {
		phi_p += M_HALFPI;
	}
	eta = (1-shear_ext*shear_ext)/(1 + shear_ext*cos(2*(phi-phi_p))); // isothermal

	double shear_angle, shear_tot;
	subhalo_lens_number = lens_number;
	subhalo_center[0]=xc; subhalo_center[1]=yc;
	shear_exclude(subhalo_center,shear_tot,shear_angle,subhalo_lens_number,reference_zfactor);
	theta_shear = degrees_to_radians(shear_angle);
	theta_shear -= M_PI/2.0;
	double (Brent::*dthetac_eq)(const double);
	dthetac_eq = static_cast<double (Brent::*)(const double)> (&Lens::subhalo_perturbation_radius_equation);
	double bound = 0.4*b;
	rmax_numerical = abs(BrentsMethod_Inclusive(dthetac_eq,-bound,bound,1e-5));
	double avg_kappa = reference_zfactor*lens_list[subhalo_lens_number]->kappa_avg_r(rmax_numerical);

	double menc = avg_kappa*M_PI*SQR(rmax_numerical)*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
	mass_enclosed = menc/alpha;
	if (verbose) {
		cout << "direction of maximum warping = " << radians_to_degrees(theta_shear) << endl;
		cout << "rmax_numerical = " << rmax_numerical << endl;
		cout << "avg_kappa/alpha = " << avg_kappa/alpha << endl;
		cout << "mass_enclosed/alpha = " << menc/alpha << endl;
		cout << "eta=" << eta << endl;
	}
}

double Lens::subhalo_perturbation_radius_equation(const double r)
{
	double kappa0, shear_tot, shear_angle, subhalo_avg_kappa;
	lensvector x;
	x[0] = subhalo_center[0] + r*cos(theta_shear);
	x[1] = subhalo_center[1] + r*sin(theta_shear);
	kappa0 = kappa_exclude(x,subhalo_lens_number,reference_zfactor);
	shear_exclude(x,shear_tot,shear_angle,subhalo_lens_number,reference_zfactor);
	subhalo_avg_kappa = reference_zfactor*lens_list[subhalo_lens_number]->kappa_avg_r(r);
	return (1 - kappa0 - shear_tot - subhalo_avg_kappa);
}

bool Lens::get_einstein_radius(int lens_number, double& re_major_axis, double& re_average)
{
	if (lens_number >= nlens) { warn("lens %i has not been created",lens_number); return false; }
	lens_list[lens_number]->get_einstein_radius(re_major_axis,re_average,reference_zfactor);
	return true;
}

double Lens::inverse_magnification_r(const double r)
{
	lensmatrix jac;
	hessian(grid_xcenter + r*cos(theta_crit), grid_ycenter + r*sin(theta_crit), jac, 0, reference_zfactor);
	jac[0][0] = 1 - jac[0][0];
	jac[1][1] = 1 - jac[1][1];
	jac[0][1] = -jac[0][1];
	jac[1][0] = -jac[1][0];
	return determinant(jac);
}

double Lens::source_plane_r(const double r)
{
	lensvector x,def;
	x[0] = grid_xcenter + r*cos(theta_crit);
	x[1] = grid_ycenter + r*sin(theta_crit);
	find_sourcept(x,def,0,reference_zfactor);
	def[0] -= grid_xcenter; // this assumes the deflection is approximately zero at the center of the grid (roughly true if any satellite gal's are small)
	def[1] -= grid_ycenter;
	return def.norm();
}

void Lens::create_deflection_spline(int steps)
{
	spline_deflection(0.5*grid_xlength*spline_frac, 0.5*grid_ylength*spline_frac, steps);
}

void Lens::spline_deflection(double xl, double yl, int steps)
{
	dvector xtable(steps+1);
	dvector ytable(steps+1);
	dmatrix defxmatrix(steps+1);
	dmatrix defymatrix(steps+1);
	dmatrix defxxmatrix(steps+1);
	dmatrix defyymatrix(steps+1);
	dmatrix defxymatrix(steps+1);

	double xmin, xmax, ymin, ymax;
	xmin = -xl; xmax = xl;
	ymin = -yl; ymax = yl;
	double x, y, xstep, ystep;
	xstep = (xmax-xmin)/steps;
	ystep = (ymax-ymin)/steps;

	int i, j;
	lensvector def;
	lensmatrix hess;
	for (i=0, x=xmin; i <= steps; i++, x += xstep) {
		xtable[i] = x;
		for (j=0, y=ymin; j <= steps; j++, y += ystep) {
			if (i==0) ytable[j] = y;		// Only needs to be done the first time around (hence "if i==0")
			deflection(x,y,def,0,reference_zfactor);
			hessian(x,y,hess,0,reference_zfactor);
			defxmatrix[i][j] = def[0];
			defymatrix[i][j] = def[1];
			defxxmatrix[i][j] = hess[0][0];
			defyymatrix[i][j] = hess[1][1];
			defxymatrix[i][j] = hess[0][1];
		}
	}

	if (defspline) delete defspline; // delete previous spline
	defspline = new Defspline;
	defspline->ax.input(xtable, ytable, defxmatrix);
	defspline->ay.input(xtable, ytable, defymatrix);
	defspline->axx.input(xtable, ytable, defxxmatrix);
	defspline->ayy.input(xtable, ytable, defyymatrix);
	defspline->axy.input(xtable, ytable, defxymatrix);
}

bool Lens::get_deflection_spline_info(double &xmax, double &ymax, int &nsteps)
{
	if (!defspline) return false;
	xmax = defspline->xmax();
	ymax = defspline->ymax();
	nsteps = defspline->nsteps();
	return true;
}

bool Lens::unspline_deflection()
{
	if (!defspline) return false;
	delete defspline;
	defspline = NULL;
	return true;
}

bool Lens::autospline_deflection(int steps)
{
	double (Brent::*mag_r)(const double);
	mag_r = static_cast<double (Brent::*)(const double)> (&Lens::inverse_magnification_r);

	double mag0, mag1, root0, root1x, root1y;
	bool found_first_root;
	double r, rstep, step_increment, step_increment_change;
	rstep = cc_rmin;
	step_increment = 1.1;
	step_increment_change = 0.5;
	int i;
	for (i=0, theta_crit=0; i < 2; i++, theta_crit += M_PI/2)  // just samples point on x-axis and y-axis
	{
		for (;;)
		{
			mag1 = inverse_magnification_r(rstep);
			found_first_root = false;
			for (r=rstep; r < cc_rmax; r += ((rstep *= step_increment)/step_increment))
			{
				mag0 = mag1;
				mag1 = inverse_magnification_r(r+rstep);
				if (mag0*mag1 < 0) {
					if (!found_first_root) {
						root0 = BrentsMethod(mag_r, r, r+rstep, 1e-3);
							found_first_root = true;
					} else {
						if (i==0) root1x = BrentsMethod(mag_r, r, r+rstep, 1e-3);
						if (i==1) root1y = BrentsMethod(mag_r, r, r+rstep, 1e-3);
						break;
					}
				}
			}
			if (r >= cc_rmax) {
				if (cc_rmin > 1e-5) cc_rmin = ((cc_rmin/10) > 1e-5) ? cc_rmin/10 : 1e-5;
				else step_increment = 1 + (step_increment-1)*step_increment_change;
				cc_rmax *= 1.5;
				rstep = cc_rmin;
			} else {
				if (i==0)	// adjust the scale of rstep if it is too small
				{
					rstep = cc_rmin;
					double rmin_frac, cc0_max_rfrac_range;
					rmin_frac = root0 / cc_rmin;
					cc0_max_rfrac_range = 1.1; // This is the (fractional) margin allowed for the inner cc radius to vary
					while (rmin_frac > 2*cc0_max_rfrac_range) {
						rstep *= 2;
						cc_rmin *= 2;
						rmin_frac /= 2;
					}
				}
				break;
			}
		}
	}
	grid_xlength = spline_frac*autogrid_frac*root1x;
	grid_ylength = spline_frac*autogrid_frac*root1y;
	spline_deflection(grid_xlength,grid_ylength,steps);
	return true;
}

Vector<dvector> Lens::find_critical_curves(bool &check)
{
	Vector<dvector> rcrit(2);
	rcrit[0].input(cc_thetasteps+1);
	rcrit[1].input(cc_thetasteps+1);

	double (Brent::*mag_r)(const double);
	mag_r = static_cast<double (Brent::*)(const double)> (&Lens::inverse_magnification_r);

	respline_at_end = false;
	resplinesteps = 0;
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
				if (defspline) {
					resplinesteps = defspline->nsteps();
					respline_at_end = true;
					unspline_deflection();
					if (cc_rmin > 1e-5) cc_rmin = ((cc_rmin/10) > 1e-5) ? cc_rmin/10 : 1e-5;
					cc_rmax *= 1.5;
					warn(warnings, "could not find critical curves after automatic deflection spline; deleting spline and trying again...");
					i = 0; theta_crit = 0;
				} else {
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
				}
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

bool Lens::spline_critical_curves(bool verbal)
{
	respline_at_end = false;	// this is for deflection spline; will become true if def. needs to be resplined
	if (cc_splined==true) {
		delete[] ccspline;
		delete[] caustic;
	}
	if ((verbal) and (mpi_id==0)) cout << "Splining critical curves..." << endl;
	bool check;
	Vector<dvector> rcrit = find_critical_curves(check);
	if (check==false) {
		cc_splined = false;	// critical curves could not be found
		return false;
	}
	Vector<dvector> rcaust(2);
	double theta, thetastep;
	thetastep = 2*M_PI/cc_thetasteps;
	rcaust[0].input(cc_thetasteps+1);
	rcaust[1].input(cc_thetasteps+1);
	dvector theta_table(cc_thetasteps+1);
	dvector caust0_theta_table(cc_thetasteps+1);
	dvector caust1_theta_table(cc_thetasteps+1);
	double xp, yp;
	lensvector x, caust0, caust1;
	bool seems_spherical = false;
	for (int i=0; i < cc_thetasteps; i++) {
		theta_table[i] = i*thetastep;
		xp = cos(theta_table[i]); yp = sin(theta_table[i]);
		x[0] = grid_xcenter + rcrit[0][i]*xp;
		x[1] = grid_ycenter + rcrit[0][i]*yp;
		find_sourcept(x,caust0,0,reference_zfactor);
		rcaust[0][i] = norm(caust0[0]-grid_xcenter,caust0[1]-grid_ycenter);
		caust0_theta_table[i] = angle(caust0[0]-grid_xcenter,caust0[1]-grid_ycenter);
		if (!(isspherical())) {
			x[0] = grid_xcenter + rcrit[1][i]*xp;
			x[1] = grid_ycenter + rcrit[1][i]*yp;
			find_sourcept(x,caust1,0,reference_zfactor);
			rcaust[1][i] = norm(caust1[0]-grid_xcenter,caust1[1]-grid_ycenter);
			caust1_theta_table[i] = angle(caust1[0]-grid_xcenter,caust1[1]-grid_ycenter);
			if ((i==0) and (rcaust[1][i] < 1e-3))
				seems_spherical = true;
			else if ((seems_spherical == true) and (rcaust[1][i] > 1e-3))
				seems_spherical = false;
		} else {
			rcaust[1][i] = 0;
			caust1_theta_table[i] = theta_table[i];
		}
		if ((rcaust[0][i] < 0) or (rcaust[1][i] < 0)) {
			cc_splined = false;
			warn(warnings, "Cannot spline critical curves; caustics not well-defined");
			return false;
		}
	}
	if (seems_spherical==true) {
		effectively_spherical = true;
		for (int i=0; i < cc_thetasteps; i++)
			rcaust[1][i] = 0;
	}

	// put the (theta,r) points in order for splining
	theta_table[cc_thetasteps] = 2*M_PI;
	sort(cc_thetasteps, caust0_theta_table, rcaust[0]);
	sort(cc_thetasteps, caust1_theta_table, rcaust[1]);
	caust0_theta_table[cc_thetasteps] = 2*M_PI + caust0_theta_table[0];
	caust1_theta_table[cc_thetasteps] = 2*M_PI + caust1_theta_table[0];
	rcaust[0][cc_thetasteps] = rcaust[0][0];
	rcaust[1][cc_thetasteps] = rcaust[1][0];
	
	ccspline = new Spline[2];
	caustic = new Spline[2];
	int c0steps, c1steps;
	ccspline[0].input(theta_table, rcrit[0]);
	ccspline[1].input(theta_table, rcrit[1]);
	caustic[0].input(caust0_theta_table, rcaust[0]);
	caustic[1].input(caust1_theta_table, rcaust[1]);
		
	cc_splined = true;

	return check;
}

bool Lens::plot_splined_critical_curves(const char *critfile)
{
	if (!use_cc_spline) { warn("Cannot plot critical curves from spline, spline option is disabled"); return false; }
	if (!cc_splined) {
		if (spline_critical_curves()==false) return false;
	}
	ofstream crit(critfile);
	if (use_scientific_notation) crit << setiosflags(ios::scientific);
	int nn = 500;
	double thetastep, rcrit0, rcrit1, rcaust0, rcaust1, xp, yp;
	thetastep = 2*M_PI/nn;
	double theta;
	int i;
	for (theta=0, i=0; i <= nn; i++, theta += thetastep) {
		rcrit0 = crit0_interpolate(theta);
		rcaust0 = caust0_interpolate(theta);
		xp = cos(theta); yp = sin(theta);
		crit << grid_xcenter+rcrit0*xp << "\t" << grid_ycenter+rcrit0*yp << "\t" << grid_xcenter+rcaust0*xp << "\t" << grid_ycenter+rcaust0*yp << endl;
	}
	crit << endl;
	for (theta=0, i=0; i <= nn; i++, theta += thetastep) {
		rcrit1 = crit1_interpolate(theta);
		rcaust1 = caust1_interpolate(theta);
		xp = cos(theta); yp = sin(theta);
		crit << grid_xcenter+rcrit1*xp << "\t" << grid_ycenter+rcrit1*yp << "\t" << grid_xcenter+rcaust1*xp << "\t" << grid_ycenter+rcaust1*yp << endl;
	}
	return true;
}

double Lens::caust0_interpolate(double theta)
{
	// sometimes caustic starts at some small angle, so make sure theta is in range
	if ((theta < caustic[0].xmin()) and (theta+2*M_PI < caustic[0].xmax()))
		theta += 2*M_PI;
	return caustic[0].splint(theta);
}
double Lens::caust1_interpolate(double theta)
{
	// sometimes caustic starts at some small angle, so make sure theta is in range
	if (effectively_spherical) return 0.0;
	if ((theta < caustic[1].xmin()) and (theta+2*M_PI < caustic[1].xmax()))
		theta += 2*M_PI;
	return caustic[1].splint(theta);
}

bool Lens::find_optimal_gridsize()
{
	if (autocenter==true) {
	for (int i=0; i < nlens; i++)
		if (i==autocenter_lens_number) { lens_list[i]->get_center_coords(grid_xcenter,grid_ycenter); }
	}

	double (Brent::*mag_r)(const double);
	mag_r = static_cast<double (Brent::*)(const double)> (&Lens::inverse_magnification_r);

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
			source_plane_rscale = source_plane_r(min_r);
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

void Lens::sort_critical_curves()
{
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

bool Lens::plot_sorted_critical_curves(const char *critfile)
{
	if (grid==NULL) {
		if (create_grid(false,reference_zfactor)==false) { warn("Could not create recursive grid"); return false; }
	}
	if (!sorted_critical_curves) sort_critical_curves();

	ofstream crit(critfile);
	if (use_scientific_notation) crit << setiosflags(ios::scientific);
	int n_cc = sorted_critical_curve.size();
	for (int j=0; j < n_cc; j++) {
		for (int k=0; k < sorted_critical_curve[j].cc_pts.size(); k++) {
			crit << sorted_critical_curve[j].cc_pts[k][0] << " " << sorted_critical_curve[j].cc_pts[k][1] << " " << sorted_critical_curve[j].caustic_pts[k][0] << " " << sorted_critical_curve[j].caustic_pts[k][1] << " " << sorted_critical_curve[j].length_of_cell[k] << endl;
		}
		// connect the first and last points to make a closed curve
		crit << sorted_critical_curve[j].cc_pts[0][0] << " " << sorted_critical_curve[j].cc_pts[0][1] << " " << sorted_critical_curve[j].caustic_pts[0][0] << " " << sorted_critical_curve[j].caustic_pts[0][1] << " " << sorted_critical_curve[j].length_of_cell[0] << endl;
		if (j < n_cc-1) crit << endl; // separates the critical curves in the plot
	}
	return true;
}

double Lens::einstein_radius_of_primary_lens(const double zfac)
{
	// this calculates the Einstein radius of the "macro" lens model (treating the lens as spherical), ignoring any lens components that are not centered on the primary lens
	double rmin_einstein_radius = 1e-6;
	double rmax_einstein_radius = 1e4;
	double xc0, yc0, xc, yc;
	lens_list[0]->get_center_coords(xc0,yc0);
	centered = new bool[nlens];
	centered[0]=true;
	bool multiple_lenses = false;
	for (int j=1; j < nlens; j++) {
		lens_list[j]->get_center_coords(xc,yc);
		if ((xc==xc0) and (yc==yc0)) {
			centered[j]=true;
			if ((multiple_lenses==false) and (lens_list[j]->defptr_r_spherical != NULL)) multiple_lenses = true;
		}
		else centered[j]=false;
	}
	if (multiple_lenses==false) {
		delete[] centered;
		double re, reav;
		lens_list[0]->get_einstein_radius(re,reav,zfac);
		return re;
	}
	zfac_re = zfac;
	if ((einstein_radius_root(rmin_einstein_radius)*einstein_radius_root(rmax_einstein_radius)) > 0) {
		// multiple imaging does not occur with this lens
		delete[] centered;
		return 0;
	}
	double re;
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&Lens::einstein_radius_root);
	re = BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-3);
	delete[] centered;
	return re;
}

double Lens::einstein_radius_root(const double r)
{
	double kapavg=0;
	for (int j=0; j < nlens; j++) {
		if ((centered[j]) and (lens_list[j]->defptr_r_spherical != NULL)) kapavg += zfac_re*lens_list[j]->kappa_avg_r(r);
	}
	return (kapavg-1);
}

void Lens::plot_total_kappa(double rmin, double rmax, int steps, const char *kname, const char *kdname)
{
	double r, rstep, total_kappa, total_dkappa;
	rstep = pow(rmax/rmin, 1.0/steps);
	int i;
	ofstream kout(kname);
	ofstream kdout;
	if (kdname != NULL) kdout.open(kdname);
	if (use_scientific_notation) kout << setiosflags(ios::scientific);
	if (use_scientific_notation) kdout << setiosflags(ios::scientific);
	double xc, yc, kap, dkap;
	bool centered[nlens];
	double xc0, yc0;
	lens_list[0]->get_center_coords(xc0,yc0);
	centered[0]=true;
	for (int j=1; j < nlens; j++) {
		lens_list[j]->get_center_coords(xc,yc);
		if ((xc==xc0) and (yc==yc0)) centered[j]=true;
		else centered[j]=false;
	}
	for (i=0, r=rmin; i < steps; i++, r *= rstep) {
		kout << r << " ";
		if (kdname != NULL) kdout << r << " ";
		total_kappa = 0;
		if (kdname != NULL) total_dkappa = 0;
		for (int j=0; j < nlens; j++) {
			if (centered[j]) {
				// this ignores off-center lenses (satellites) since we are plotting the radial profile; ellipticity is also ignored
				kap = lens_list[j]->kappa_r(r);
				if (kdname != NULL) dkap = lens_list[j]->dkappa_rsq(r*r);
				total_kappa += kap;
				if (kdname != NULL) total_dkappa += dkap;
			}
		}
		kout << total_kappa << endl;
		if (kdname != NULL) kdout << fabs(total_dkappa) << endl;
	}
}

void Lens::plot_mass_profile(double rmin, double rmax, int rpts, const char *massname)
{
	double r, rstep, kavg;
	rstep = pow(rmax/rmin, 1.0/(rpts-1));
	int i;
	ofstream mout(massname);
	if (use_scientific_notation) mout << setiosflags(ios::scientific);
	double arcsec_to_kpc = angular_diameter_distance(lens_redshift)/(1e-3*(180/M_PI)*3600);
	double sigma_cr_arcsec = sigma_crit_arcsec(lens_redshift, reference_source_redshift);
	mout << "#radius(arcsec) mass(m_solar) radius(kpc)\n";
	for (i=0, r=rmin; i < rpts; i++, r *= rstep) {
		kavg = 0;
		for (int j=0; j < nlens; j++) {
			kavg += reference_zfactor*lens_list[j]->kappa_avg_r(r);
		}
		mout << r << " " << sigma_cr_arcsec*M_PI*kavg*r*r << " " << r*arcsec_to_kpc << endl;
	}
}

void Lens::plot_kappa_profile(int l, double rmin, double rmax, int steps, const char *kname, const char *kdname)
{
	if (l >= nlens) { warn("lens %i does not exist", l); return; }
	lens_list[l]->plot_kappa_profile(rmin,rmax,steps,kname,kdname);
}

bool Lens::isspherical()
{
	bool all_spherical = true;
	for (int i=0; i < nlens; i++)
		if (!(lens_list[i]->isspherical())) { all_spherical = false; break; }
	return all_spherical;
}

bool Lens::make_random_sources(int nsources, const char *outfilename)
{
	if (!use_cc_spline) { warn(warnings,"Random sources is only supported in critical curve spline mode (ccspline)"); return false; }
	ofstream outfile(outfilename);
	if (!cc_splined)
		if (spline_critical_curves()==false) return false;	// in case critical curves could not be found
	int sources_inside = 0;
	double theta, r, theta_step, rmax;
	int theta_n, theta_count = 400;
	theta_step = 2*M_PI/(theta_count-1);
	rmax = dmax(caust0_interpolate(0), caust1_interpolate(0));
	for (theta_n=0, theta=0; theta_n < theta_count; theta_n++, theta += theta_step) {
		r = dmax(caust0_interpolate(theta), caust1_interpolate(theta));
		if (r > rmax) rmax = r;
	}

	while (sources_inside < nsources)
	{
		theta = RandomNumber2() * 2*M_PI;
		r = sqrt(RandomNumber2()) * rmax;
		if (r < dmax(caust0_interpolate(theta), caust1_interpolate(theta))) {
			sources_inside++;
			outfile << grid_xcenter+r*cos(theta) << "\t" << grid_ycenter+r*sin(theta) << endl;
		}
	}
	return true;
}

void Lens::make_source_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename)
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

void Lens::make_source_ellipse(const double xcenter, const double ycenter, const double major_axis, const double q, const double angle_degrees, const int n_subellipses, const int points_per_ellipse, string source_filename)
{
	ofstream source_file(source_filename.c_str());

	double da, dtheta, angle;
	da = major_axis/(n_subellipses-1);
	dtheta = M_2PI/points_per_ellipse;
	angle = (M_PI/180)*angle_degrees;
	double a, theta, x, y;

	int i,j;
	for (i=1, a=da; i < n_subellipses; i++, a += da)
	{
		for (j=0, theta=0; j < points_per_ellipse; j++, theta += dtheta)
		{
			x = a*cos(theta); y = a*q*sin(theta);
			source[0] = xcenter + x*cos(angle) - y*sin(angle);
			source[1] = ycenter + x*sin(angle) + y*cos(angle);
			source_file << source[0] << " " << source[1] << endl;
		}
	}
}

bool Lens::total_cross_section(double &area)
{
	if (!use_cc_spline) { warn(warnings,"Determining total cross section is only supported when ccspline mode is on"); return false; }
	if (!cc_splined)
		if (spline_critical_curves()==false) return false;	// in case critical curves could not be found
	double (Romberg::*csptr)(const double);
	csptr = static_cast<double (Romberg::*)(const double)> (&Lens::total_cross_section_integrand);
	area = romberg(csptr, 0, 2*M_PI,1e-6,5);

	return true;
}

double Lens::total_cross_section_integrand(const double theta)
{
	return (0.5 * SQR(dmax(caust0_interpolate(theta), caust1_interpolate(theta))));
}

bool Lens::plot_lens_equation(double x_source, double y_source, const char *lxfile, const char *lyfile)
{
	source[0] = x_source; source[1] = y_source;
	if (source[0]==0) source[0] = 1e-6; // the contours look weird for source[0] = 0 or source[1] = 0....
	if (source[1]==0) source[1] = 1e-6; //  no idea why!!
	int nxpoints, nypoints, resolution, ncells;
	double cell_size;
	resolution = 6000;
	cell_size = (dmin(grid_xlength,grid_ylength))/((double) resolution);
	ncells = 100000;
	double max_xy = 0.5*dmax(grid_xlength,grid_ylength);

	Vector<lensvector> eq0_cell(ncells);
	Vector<lensvector> eq1_cell(ncells);
	double (Lens::*xptr)(double,double);
	double (Lens::*yptr)(double,double);
	xptr = (&Lens::lens_equation_x);
	yptr = (&Lens::lens_equation_y);

	ncfound = 0;
	nxpoints = find_zero_curves(eq0_cell, -max_xy, -max_xy, max_xy, max_xy, cell_size, true, xptr);
	ofstream lenseqx(lxfile);
	for (int i = 0; i < nxpoints; i++)
		lenseqx << eq0_cell[i][0] << "\t" << eq0_cell[i][1] << endl;

	ncfound = 0;
	double x_half_length = 0.5*autogrid_frac*grid_xlength;
	double y_half_length = 0.5*autogrid_frac*grid_ylength;
	nypoints = find_zero_curves(eq1_cell, -x_half_length, -y_half_length, x_half_length, y_half_length, cell_size, true, yptr);
	ofstream lenseqy(lyfile);
	for (int i = 0; i < nypoints; i++)
		lenseqy << eq1_cell[i][0] << "\t" << eq1_cell[i][1] << endl;

	return true;
}

double Lens::lens_equation_x(double x, double y)
{
	lensvector def;
	deflection(x,y,def,0,reference_zfactor);
	return (source[0] - x + def[0]);
}

double Lens::lens_equation_y(double x, double y)
{
	lensvector def;
	deflection(x,y,def,0,reference_zfactor);
	return (source[1] - y + def[1]);
}

int Lens::find_zero_curves(Vector<lensvector>& cell, double xmin, double ymin, double xmax, double ymax, double cell_size, bool first_iteration, double (Lens::*func)(double, double))
{
	int steps = (first_iteration==true) ? 80 : 2;   // The gridsearch divides the grid by this number with every iteration
	double x, y, xstep, ystep;
	xstep = (xmax - xmin)/steps;		// The ordering of points in a box is:  3-4
	ystep = (ymax - ymin)/steps;		//                                      | |
	double mag1, mag2, mag3, mag4;	//                                      1-2
	bool zero_curve_inside; // keeps track of whether a given box contains a critical curve
	int i, j;
	for (i = 0, x = xmin; i < steps; i++, x += xstep) {
		for (j = 0, y = ymin; j < steps; j++, y += ystep) {
			if (ncfound >= cell.size()) return ncfound;

			// First, assign the correct (inverse) inverse-magnification values for each point in the box
			mag1 = (this->*func)(x,y);
			mag2 = (this->*func)(x+xstep,y);
			mag3 = (this->*func)(x,y+ystep);
			mag4 = (this->*func)(x+xstep,y+ystep);

			// Now, test to see if there is a critical curve inside the square
			if ((mag1*mag2*mag3*mag4) < 0) zero_curve_inside = true;
			else if (((mag1*mag2) < 0) || (mag1*mag3) < 0) zero_curve_inside = true;
			else zero_curve_inside = false;

			// See if the square is small enough to be counted
			if (zero_curve_inside) {
				if ((xstep < cell_size) && (ystep < cell_size)) {	// found one!!
					cell[ncfound][0] = x + xstep/2.0;
					cell[ncfound][1] = y + ystep/2.0;
					ncfound++;
					if (ncfound >= cell.size()) {
						warn(warnings, "maximum number of cells exceeded in grid search");
						return ncfound;
					}
				} else {
					// There's a zero curve in here, but we need to take a closer look (subgrid)
					find_zero_curves(cell, x, y, x+xstep, y+ystep, cell_size, false, func);
					if (ncfound >= cell.size()) {
						warn(warnings, "maximum number of cells exceeded in grid search");
						return ncfound;
					}
				}
			}
		}
	}

	return ncfound; // Returns number of boxes that contained a critical curve
}

double Lens::make_satellite_population(const double number_density, const double rmax, const double b, const double a)
{
	int N = (int) (number_density*M_PI*rmax*rmax);
	int realizations = 3000;
	double r, theta, alpha_x, alpha_y, defsqr, defnorm, defsqr2;
	double mean_alpha_x=0, mean_alpha_y=0, mean_defsqr=0, mean_defsqr2=0;
	int i,j;
	for (j=0; j < realizations; j++) {
		alpha_x=alpha_y=0;
		defsqr2=0;
		for (i=0; i < N; i++) {
			r = sqrt(RandomNumber2())*rmax;
			theta = RandomNumber2()*2*M_PI;
			defnorm = b*(1+(a-sqrt(r*r+a*a))/r);
			alpha_x += -defnorm*cos(theta);
			alpha_y += -defnorm*sin(theta);
		}
		defsqr = SQR(alpha_x) + SQR(alpha_y);
		mean_defsqr += defsqr;
		mean_alpha_x += alpha_x;
		mean_alpha_y += alpha_y;
	}
	mean_defsqr /= realizations;
	mean_alpha_x /= realizations;
	mean_alpha_y /= realizations;
	//cout << "Root-mean square deflection: " << mean_defsqr << " " << mean_defsqr2 << endl;
	//cout << "Mean deflection: " << mean_alpha_x << " " << mean_alpha_y << endl;
	return mean_defsqr;
}

void Lens::plot_satellite_deflection_vs_area()
{
	int i,nn = 30;
	double r,rmin,rmax,rstep,logrstep,defsqr_avg;
	rmin=5;
	rmax=5000;
	rstep = (rmax-rmin)/nn;
	logrstep = pow(rmax/rmin,1.0/(nn-1));
	for (i=0, r=rmin; i < nn; i++, r *= logrstep) {
		defsqr_avg = make_satellite_population(0.04,r,0.1,0.6);
		cout << r << " " << log(r) << " " << defsqr_avg << endl;
	}
}

/********************************* Functions for point image data (reading, writing, simulating etc.) *********************************/

void Lens::add_simulated_image_data(const lensvector &sourcept)
{
	bool *new_vary_sourcepts_x = new bool[n_sourcepts_fit+1];
	bool *new_vary_sourcepts_y = new bool[n_sourcepts_fit+1];
	lensvector *new_sourcepts_fit = new lensvector[n_sourcepts_fit+1];
	ImageData *new_image_data = new ImageData[n_sourcepts_fit+1];
	lensvector* new_sourcepts_lower_limit;
	lensvector* new_sourcepts_upper_limit;
	if (sourcepts_upper_limit != NULL) {
		new_sourcepts_lower_limit = new lensvector[n_sourcepts_fit+1];
		new_sourcepts_upper_limit = new lensvector[n_sourcepts_fit+1];
	}
	double *new_redshifts, *new_zfactors;
	new_redshifts = new double[n_sourcepts_fit+1];
	new_zfactors = new double[n_sourcepts_fit+1];
	for (int i=0; i < n_sourcepts_fit; i++) {
		new_redshifts[i] = source_redshifts[i];
		new_zfactors[i] = zfactors[i];
		new_sourcepts_fit[i] = sourcepts_fit[i];
		new_vary_sourcepts_x[i] = vary_sourcepts_x[i];
		new_vary_sourcepts_y[i] = vary_sourcepts_y[i];
		new_image_data[i].input(image_data[i]);
		if (sourcepts_upper_limit != NULL) {
			new_sourcepts_upper_limit[i] = sourcepts_upper_limit[i];
			new_sourcepts_lower_limit[i] = sourcepts_lower_limit[i];
		}
	}
	new_redshifts[n_sourcepts_fit] = source_redshift;
	if (n_sourcepts_fit > 0) new_zfactors[n_sourcepts_fit] = kappa_ratio(lens_redshift,source_redshift,reference_source_redshift);
	delete[] image_data;
	delete[] sourcepts_fit;
	delete[] vary_sourcepts_x;
	delete[] vary_sourcepts_y;
	if (source_redshifts != NULL) delete[] source_redshifts;
	if (zfactors != NULL) delete[] zfactors;
	source_redshifts = new_redshifts;
	zfactors = new_zfactors;

	int n_images;
	image *imgs = get_images(sourcept, n_images, false);
	bool include_image[n_images];
	double min_td=1e30;
	for (int i=0; i < n_images; i++) {
		// central maxima images have positive parity and kappa > 1, so use this to exclude them if desired
		if ((include_central_image==false) and (imgs[i].parity == 1) and (kappa(imgs[i].pos,reference_zfactor) > 1)) include_image[i] = false;
		else include_image[i] = true;
		imgs[i].pos[0] += sim_err_pos*NormalDeviate();
		imgs[i].pos[1] += sim_err_pos*NormalDeviate();
		imgs[i].mag *= source_flux; // now imgs[i].mag is in fact the flux, not just the magnification
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
	new_sourcepts_fit[n_sourcepts_fit] = sourcept;
	new_vary_sourcepts_x[n_sourcepts_fit] = true;
	new_vary_sourcepts_y[n_sourcepts_fit] = true;
	new_image_data[n_sourcepts_fit].input(n_images,imgs,sim_err_pos,sim_err_flux,sim_err_td,include_image,include_time_delays);
	n_sourcepts_fit++;
	image_data = new_image_data;
	sourcepts_fit = new_sourcepts_fit;
	vary_sourcepts_x = new_vary_sourcepts_x;
	vary_sourcepts_y = new_vary_sourcepts_y;
	if (sourcepts_upper_limit != NULL) {
		delete[] sourcepts_upper_limit;
		delete[] sourcepts_lower_limit;
		new_sourcepts_lower_limit[n_sourcepts_fit][0] = -1e30;
		new_sourcepts_lower_limit[n_sourcepts_fit][1] = -1e30;
		new_sourcepts_upper_limit[n_sourcepts_fit][0] = 1e30;
		new_sourcepts_upper_limit[n_sourcepts_fit][1] = 1e30;
		sourcepts_upper_limit = new_sourcepts_upper_limit;
		sourcepts_lower_limit = new_sourcepts_lower_limit;
	}
}

void Lens::write_image_data(string filename)
{
	ofstream outfile(filename.c_str());
	if (use_scientific_notation==true) outfile << setiosflags(ios::scientific);
	else {
		outfile << setprecision(6);
		outfile << fixed;
	}
	outfile << "zlens = " << lens_redshift << endl;
	outfile << n_sourcepts_fit << " # number of source points" << endl;
	for (int i=0; i < n_sourcepts_fit; i++) {
		outfile << image_data[i].n_images << " " << source_redshifts[i] << " # number of images, source redshift" << endl;
		image_data[i].write_to_file(outfile);
	}
}

bool Lens::load_image_data(string filename)
{
	ifstream data_infile(filename.c_str());
	if (!data_infile.is_open()) { warn("Error: input file '%s' could not be opened",filename.c_str()); return false; }

	int n_datawords;
	vector<string> datawords;

	if (read_data_line(data_infile,datawords,n_datawords)==false) { warn("data file could not be read; unexpected end of file"); return false; }
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
	n_sourcepts_fit = nsrcfit;

	if (sourcepts_fit != NULL) {
		delete[] sourcepts_fit;
		delete[] vary_sourcepts_x;
		delete[] vary_sourcepts_y;
	}
	if (source_redshifts != NULL) delete[] source_redshifts;
	if (zfactors != NULL) delete[] zfactors;

	sourcepts_fit = new lensvector[n_sourcepts_fit];
	vary_sourcepts_x = new bool[n_sourcepts_fit];
	vary_sourcepts_y = new bool[n_sourcepts_fit];
	source_redshifts = new double[n_sourcepts_fit];
	zfactors = new double[n_sourcepts_fit];
	for (int i=0; i < n_sourcepts_fit; i++) {
		vary_sourcepts_x[i] = true;
		vary_sourcepts_y[i] = true;
		source_redshifts[i] = source_redshift;
	}
	if ((fitmethod != POWELL) and (fitmethod != SIMPLEX)) {
		// You should replace the pointers here with a container class like Vector. This is too bug-prone!
		if (sourcepts_lower_limit!=NULL) delete[] sourcepts_lower_limit;
		sourcepts_lower_limit = new lensvector[n_sourcepts_fit];
		if (sourcepts_upper_limit!=NULL) delete[] sourcepts_upper_limit;
		sourcepts_upper_limit = new lensvector[n_sourcepts_fit];
	}

	if (image_data != NULL) delete[] image_data;
	image_data = new ImageData[n_sourcepts_fit];
	int i, j, nn;
	for (i=0; i < n_sourcepts_fit; i++) {
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
			if (datastring_convert(datawords[1],source_redshifts[i])==false) {
				warn("data file has incorrect format; could not read redshift for source point %i",i);
				clear_image_data();
				return false;
			}
		}
		image_data[i].input(nn);
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
			if (datastring_convert(datawords[0],image_data[i].pos[j][0])==false) {
				warn("image position x-coordinate has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[1],image_data[i].pos[j][1])==false) {
				warn("image position y-coordinate has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[2],image_data[i].sigma_pos[j])==false) {
				warn("image position measurement error has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[3],image_data[i].flux[j])==false) {
				warn("image flux has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (datastring_convert(datawords[4],image_data[i].sigma_f[j])==false) {
				warn("image flux measurement error has incorrect format; could not read entry for source point %i, image number %i",i,j);
				clear_image_data();
				return false;
			}
			if (n_datawords==7) {
				if (datastring_convert(datawords[5],image_data[i].time_delays[j])==false) {
					warn("image time delay has incorrect format; could not read entry for source point %i, image number %i",i,j);
					clear_image_data();
					return false;
				}
				if (datastring_convert(datawords[6],image_data[i].sigma_t[j])==false) {
					warn("image time delay has incorrect format; could not read entry for source point %i, image number %i",i,j);
					n_sourcepts_fit=0; delete[] image_data; image_data = NULL;
					clear_image_data();
					return false;
				}
			} else {
				image_data[i].time_delays[j] = 0;
				image_data[i].sigma_t[j] = 0;
			}
		}
	}
	for (i=0; i < n_sourcepts_fit; i++) {
		zfactors[i] = kappa_ratio(lens_redshift,source_redshifts[i],reference_source_redshift);
	}

	int ncombs, max_combinations = -1;
	int k,n;
	for (i=0; i < n_sourcepts_fit; i++) {
		ncombs = image_data[i].n_images * (image_data[i].n_images-1) / 2;
		if (ncombs > max_combinations) max_combinations = ncombs;
	}
	double *distsqrs = new double[max_combinations];
	for (i=0; i < n_sourcepts_fit; i++) {
		n=0;
		for (k=0; k < image_data[i].n_images; k++) {
			for (j=k+1; j < image_data[i].n_images; j++) {
				distsqrs[n] = SQR(image_data[i].pos[k][0] - image_data[i].pos[j][0]) + SQR(image_data[i].pos[k][1] - image_data[i].pos[j][1]);
				n++;
			}
		}
		sort(n,distsqrs);
		image_data[i].max_distsqr = distsqrs[n-1]; // this saves the maximum distance between any pair of images (useful for image chi-square for missing image penalty values)
	}
	delete[] distsqrs;

	return true;
}

void Lens::remove_image_data(int image_set)
{
	if (image_set >= n_sourcepts_fit) { warn(warnings,"Specified image dataset has not been loaded"); return; }
	bool *new_vary_sourcepts_x = new bool[n_sourcepts_fit-1];
	bool *new_vary_sourcepts_y = new bool[n_sourcepts_fit-1];
	lensvector *new_sourcepts_fit = new lensvector[n_sourcepts_fit-1];
	ImageData *new_image_data = new ImageData[n_sourcepts_fit-1];
	int i,j;
	for (i=0,j=0; i < n_sourcepts_fit; i++) {
		if (i != image_set) {
			new_sourcepts_fit[j] = sourcepts_fit[i];
			new_vary_sourcepts_x[j] = vary_sourcepts_x[i];
			new_vary_sourcepts_y[j] = vary_sourcepts_y[i];
			new_image_data[j].input(image_data[i]);
			j++;
		}
	}
	if (n_sourcepts_fit > 1) {
		// always leave one redshift recorded, which is the generic redshift used when playing around (not necessarily for model fitting)
		double *new_redshifts, *new_zfactors;
		new_redshifts = new double[n_sourcepts_fit-1];
		new_zfactors = new double[n_sourcepts_fit-1];
		for (i=0,j=0; i < n_sourcepts_fit; i++) {
			if (i != image_set) {
				new_redshifts[j] = source_redshifts[i];
				new_zfactors[j] = zfactors[i];
			}
			j++;
		}
		delete[] source_redshifts;
		delete[] zfactors;
		source_redshifts = new_redshifts;
		zfactors = new_zfactors;
	}
	delete[] image_data;
	delete[] sourcepts_fit;
	delete[] vary_sourcepts_x;
	delete[] vary_sourcepts_y;

	n_sourcepts_fit--;
	image_data = new_image_data;
	sourcepts_fit = new_sourcepts_fit;
	vary_sourcepts_x = new_vary_sourcepts_x;
	vary_sourcepts_y = new_vary_sourcepts_y;
}

bool Lens::read_data_line(ifstream& data_infile, vector<string>& datawords, int &n_datawords)
{
	static const int n_characters = 256;
	string word;
	n_datawords = 0;
	datawords.clear();
	do {
		char dataline[n_characters];
		data_infile.getline(dataline,n_characters);
		if ((data_infile.rdstate() & ifstream::eofbit) != 0) return false;
		string linestring(dataline);
		remove_comments(linestring);
		istringstream datastream0(linestring.c_str());
		while (datastream0 >> word) datawords.push_back(word);
		n_datawords = datawords.size();
	} while (n_datawords==0); // skip lines that are blank or only have comments
	remove_equal_sign_datafile(datawords,n_datawords);
	return true;
}

void Lens::remove_comments(string& instring)
{
	string instring_copy(instring);
	instring.clear();
	size_t comment_pos = instring_copy.find("#");
	if (comment_pos != string::npos) {
		instring = instring_copy.substr(0,comment_pos);
	} else instring = instring_copy;
}

void Lens::remove_equal_sign_datafile(vector<string>& datawords, int &n_datawords)
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

bool Lens::datastring_convert(const string& instring, int& outvar)
{
	datastream.clear(); // resets the error flags
	datastream.str(string()); // clears the stringstream
	datastream << instring;
	if (datastream >> outvar) return true;
	else return false;
}

bool Lens::datastring_convert(const string& instring, double& outvar)
{
	datastream.clear(); // resets the error flags
	datastream.str(string()); // clears the stringstream
	datastream << instring;
	if (datastream >> outvar) return true;
	else return false;
}

void Lens::clear_image_data()
{
	if (image_data != NULL) {
		delete[] image_data;
		image_data = NULL;
	}
	if (sourcepts_fit != NULL) {
		delete[] sourcepts_fit;
		delete[] vary_sourcepts_x;
		delete[] vary_sourcepts_y;
		sourcepts_fit = NULL;
	}
	if (sourcepts_lower_limit != NULL) {
		delete[] sourcepts_lower_limit;
		sourcepts_lower_limit = NULL;
	}
	if (sourcepts_upper_limit != NULL) {
		delete[] sourcepts_upper_limit;
		sourcepts_upper_limit = NULL;
	}
	n_sourcepts_fit = 0;
}

void Lens::print_image_data(bool include_errors)
{
	if (mpi_id==0) {
		for (int i=0; i < n_sourcepts_fit; i++) {
			cout << "Source " << i << ": zsrc=" << source_redshifts[i] << endl;
			image_data[i].print_list(include_errors,use_scientific_notation);
		}
	}
}

void ImageData::input(const int &nn)
{
	n_images = nn;
	pos = new lensvector[n_images];
	flux = new double[n_images];
	time_delays = new double[n_images];
	sigma_pos = new double[n_images];
	sigma_f = new double[n_images];
	sigma_t = new double[n_images];
	max_distsqr = 1e30;
}

void ImageData::input(const ImageData& imgs_in)
{
	if (n_images != 0) {
		// delete arrays so we can re-create them
		delete[] pos;
		delete[] flux;
		delete[] time_delays;
		delete[] sigma_pos;
		delete[] sigma_f;
		delete[] sigma_t;
	}
	n_images = imgs_in.n_images;
	pos = new lensvector[n_images];
	flux = new double[n_images];
	time_delays = new double[n_images];
	sigma_pos = new double[n_images];
	sigma_f = new double[n_images];
	sigma_t = new double[n_images];
	for (int i=0; i < n_images; i++) {
		pos[i] = imgs_in.pos[i];
		flux[i] = imgs_in.flux[i];
		time_delays[i] = imgs_in.time_delays[i];
		sigma_pos[i] = imgs_in.sigma_pos[i];
		sigma_f[i] = imgs_in.sigma_f[i];
		sigma_t[i] = imgs_in.sigma_t[i];
	}
	max_distsqr = 1e30;
}

void ImageData::input(const int &nn, image* images, const double sigma_pos_in, const double sigma_flux_in, const double sigma_td_in, bool* include, bool include_time_delays)
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
	int j=0;
	for (int i=0; i < nn; i++) {
		if (!include[i]) continue;
		pos[j] = images[i].pos;
		flux[j] = images[i].mag; // images[i].mag should be the flux, not the magnification
		if (include_time_delays) {
			time_delays[j] = images[i].td;
			sigma_t[j] = sigma_td_in;
		}
		else { time_delays[j] = 0; sigma_t[j] = 0; }
		sigma_pos[j] = sigma_pos_in;
		sigma_f[j] = sigma_flux_in;
		j++;
	}
	max_distsqr = 1e30;
}


void ImageData::add_image(lensvector& pos_in, const double sigma_pos_in, const double flux_in, const double sigma_f_in, const double time_delay_in, const double sigma_t_in)
{
	int n_images_new = n_images+1;
	if (n_images != 0) {
		lensvector *new_pos = new lensvector[n_images_new];
		double *new_flux = new double[n_images_new];
		double *new_time_delays = new double[n_images_new];
		double *new_sigma_pos = new double[n_images_new];
		double *new_sigma_f = new double[n_images_new];
		double *new_sigma_t = new double[n_images_new];
		for (int i=0; i < n_images; i++) {
			new_pos[i][0] = pos[i][0];
			new_pos[i][1] = pos[i][1];
			new_flux[i] = flux[i];
			new_time_delays[i] = time_delays[i];
			new_sigma_pos[i] = sigma_pos[i];
			new_sigma_f[i] = sigma_f[i];
			new_sigma_t[i] = sigma_t[i];
		}
		delete[] pos;
		delete[] flux;
		delete[] time_delays;
		delete[] sigma_pos;
		delete[] sigma_f;
		delete[] sigma_t;
		pos = new_pos;
		flux = new_flux;
		time_delays = new_time_delays;
		sigma_pos = new_sigma_pos;
		sigma_f = new_sigma_f;
		sigma_t = new_sigma_t;
		n_images++;
	} else {
		n_images = 1;
		pos = new lensvector[n_images];
		flux = new double[n_images];
		time_delays = new double[n_images];
		sigma_pos = new double[n_images];
		sigma_f = new double[n_images];
		sigma_t = new double[n_images];
	}
	pos[n_images-1][0] = pos_in[0];
	pos[n_images-1][1] = pos_in[1];
	flux[n_images-1] = flux_in;
	time_delays[n_images-1] = time_delay_in;
	sigma_pos[n_images-1] = sigma_pos_in;
	sigma_f[n_images-1] = sigma_f_in;
	sigma_t[n_images-1] = sigma_t_in;
}

void ImageData::print_list(bool print_errors, bool use_sci)
{
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
		cout << endl;
	}
	cout << endl;
	if (use_sci==false)
		cout.unsetf(ios_base::floatfield);
}

void ImageData::write_to_file(ofstream &outfile)
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

ImageData::~ImageData()
{
	delete[] pos;
	delete[] flux;
	delete[] time_delays;
	delete[] sigma_pos;
	delete[] sigma_f;
	delete[] sigma_t;
}


/******************************************** Functions for lens model fitting ******************************************/

void Lens::initialize_fitmodel()
{
	if (source_fit_mode == Point_Source) {
		if ((sourcepts_fit==NULL) or (image_data==NULL)) { warn("cannot do fit; image data points have not been defined"); return; }
	} else if (source_fit_mode == Pixellated_Source) {
		if (image_pixel_data==NULL) { warn("cannot do fit; image data pixels have not been loaded"); return; }
	}
	if (fitmodel != NULL) delete fitmodel;
	fitmodel = new Lens(this);
	fitmodel->auto_ccspline = false;
	//fitmodel->set_gridcenter(grid_xcenter,grid_ycenter);

	fitmodel->borrowed_image_data = true;
	if (source_fit_mode == Point_Source) {
		fitmodel->image_data = image_data;
		fitmodel->n_sourcepts_fit = n_sourcepts_fit;
		fitmodel->sourcepts_fit = new lensvector[n_sourcepts_fit];
		fitmodel->vary_sourcepts_x = new bool[n_sourcepts_fit];
		fitmodel->vary_sourcepts_y = new bool[n_sourcepts_fit];
		fitmodel->source_redshifts = new double[n_sourcepts_fit];
		fitmodel->zfactors = new double[n_sourcepts_fit];
		for (int i=0; i < n_sourcepts_fit; i++) {
			fitmodel->vary_sourcepts_x[i] = vary_sourcepts_x[i];
			fitmodel->vary_sourcepts_y[i] = vary_sourcepts_y[i];
			fitmodel->sourcepts_fit[i][0] = sourcepts_fit[i][0];
			fitmodel->sourcepts_fit[i][1] = sourcepts_fit[i][1];
			fitmodel->source_redshifts[i] = source_redshifts[i];
			fitmodel->zfactors[i] = zfactors[i];
		}
	} else if (source_fit_mode == Pixellated_Source) {
		fitmodel->image_pixel_data = image_pixel_data;
		fitmodel->load_pixel_grid_from_data();
		delete source_pixel_grid; source_pixel_grid = NULL; // we do this because some of the static source grid parameters will be changed during fit (really should reorganize so this is not an issue)
	}

	fitmodel->nlens = nlens;
	fitmodel->lens_list = new LensProfile*[nlens];
	for (int i=0; i < nlens; i++) {
		switch (lens_list[i]->get_lenstype()) {
			case KSPLINE:
				fitmodel->lens_list[i] = new LensProfile(lens_list[i]); break;
			case ALPHA:
				fitmodel->lens_list[i] = new Alpha((Alpha*) lens_list[i]); break;
			case PJAFFE:
				fitmodel->lens_list[i] = new PseudoJaffe((PseudoJaffe*) lens_list[i]); break;
			case nfw:
				fitmodel->lens_list[i] = new NFW((NFW*) lens_list[i]); break;
			case TRUNCATED_nfw:
				fitmodel->lens_list[i] = new Truncated_NFW((Truncated_NFW*) lens_list[i]); break;
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
			case PTMASS:
				fitmodel->lens_list[i] = new PointMass((PointMass*) lens_list[i]); break;
			case SHEET:
				fitmodel->lens_list[i] = new MassSheet((MassSheet*) lens_list[i]); break;
			default:
				die("lens type not supported for fitting");
		}
	}
	for (int i=0; i < nlens; i++) {
		// if the lens is anchored to another lens, re-anchor so that it points to the corresponding
		// lens in fitmodel (the lens whose parameters will be varied)
		if (fitmodel->lens_list[i]->anchored==true) fitmodel->lens_list[i]->anchor_to_lens(fitmodel->lens_list, lens_list[i]->get_anchor_number());
		if (fitmodel->lens_list[i]->anchor_extra_parameter==true) {
			LensProfile *parameter_anchor_lens = fitmodel->lens_list[lens_list[i]->get_parameter_anchor_number()];
			fitmodel->lens_list[i]->assign_anchored_parameters(parameter_anchor_lens);
		}
	}
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
}

void Lens::update_anchored_parameters()
{
	for (int i=0; i < nlens; i++) {
		if (lens_list[i]->anchored) lens_list[i]->update_anchor_center();
		if (lens_list[i]->anchor_extra_parameter) lens_list[i]->update_extra_anchored_params(); // PJAFFE and CORECUSP are the only lens so far that have a parameter that can be anchored (tidal radius)
	}
}

bool Lens::update_fitmodel(const double* params)
{
	bool status = true;
	int i, index=0;
	for (i=0; i < nlens; i++) {
		fitmodel->lens_list[i]->update_fit_parameters(params,index,status);
	}
	fitmodel->update_anchored_parameters();
	if (source_fit_mode==Point_Source) {
		if ((!use_analytic_bestfit_src) or (use_image_plane_chisq) or (use_image_plane_chisq2)) {
			for (i=0; i < n_sourcepts_fit; i++) {
				if (fitmodel->vary_sourcepts_x[i]) fitmodel->sourcepts_fit[i][0] = params[index++];
				if (fitmodel->vary_sourcepts_y[i]) fitmodel->sourcepts_fit[i][1] = params[index++];
			}
		}
	} else if (source_fit_mode == Pixellated_Source) {
		if ((vary_regularization_parameter) and (regularization_method != None)) fitmodel->regularization_parameter = params[index++];
		if (vary_pixel_fraction) fitmodel->pixel_fraction = params[index++];
		if (vary_magnification_threshold) fitmodel->pixel_magnification_threshold = params[index++];
	}
	if (vary_hubble_parameter) {
		fitmodel->hubble = params[index++];
		if (fitmodel->hubble < 0) status = false; // do not allow negative Hubble parameter
		fitmodel->set_cosmology(fitmodel->omega_matter,0.04,fitmodel->hubble,2.215);
	}
	if (index != n_fit_parameters) die("Index didn't go through all the fit parameters (%i)",n_fit_parameters);
	return status;
}

double Lens::chisq_pos_source_plane()
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

	for (i=0; i < n_sourcepts_fit; i++) {
		if (image_data[i].n_images > n_images_hi) n_images_hi = image_data[i].n_images;
	}
	double* mag00 = new double[n_images_hi];
	double* mag11 = new double[n_images_hi];
	double* mag01 = new double[n_images_hi];
	lensvector* beta_ji = new lensvector[n_images_hi];

	double siginv, src_norm;
	for (i=0; i < n_sourcepts_fit; i++) {
		amatrix[0][0] = amatrix[0][1] = amatrix[1][0] = amatrix[1][1] = 0;
		bvec[0] = bvec[1] = 0;
		src_bf[0] = src_bf[1] = 0;
		src_norm=0;
		for (j=0; j < image_data[i].n_images; j++) {
			find_sourcept(image_data[i].pos[j],beta_ji[j],0,zfactors[i]);
			if (use_magnification_in_chisq) {
				hessian(image_data[i].pos[j],jac,zfactors[i]);
				jac[0][0] = 1 - jac[0][0];
				jac[1][1] = 1 - jac[1][1];
				jac[0][1] = -jac[0][1];
				jac[1][0] = -jac[1][0];
				mag = jac.inverse();
				mag00[j] = mag[0][0];
				mag01[j] = mag[0][1];
				mag11[j] = mag[1][1];

				if (use_analytic_bestfit_src) {
					lensmatsqr(mag,magsqr);
					siginv = 1.0/SQR(image_data[i].sigma_pos[j]);
					amatrix[0][0] += magsqr[0][0]*siginv;
					amatrix[1][0] += magsqr[1][0]*siginv;
					amatrix[0][1] += magsqr[0][1]*siginv;
					amatrix[1][1] += magsqr[1][1]*siginv;
					bvec[0] += (magsqr[0][0]*beta_ji[j][0] + magsqr[0][1]*beta_ji[j][1])*siginv;
					bvec[1] += (magsqr[1][0]*beta_ji[j][0] + magsqr[1][1]*beta_ji[j][1])*siginv;
				}
			} else {
				if (use_analytic_bestfit_src) {
					siginv = 1.0/SQR(image_data[i].sigma_pos[j]);
					src_bf[0] += beta_ji[j][0]*siginv;
					src_bf[1] += beta_ji[j][1]*siginv;
					src_norm += siginv;
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
		} else {
			beta = &sourcepts_fit[i];
		}

		for (j=0; j < image_data[i].n_images; j++) {
			delta_beta[0] = (*beta)[0] - beta_ji[j][0];
			delta_beta[1] = (*beta)[1] - beta_ji[j][1];
			if (use_magnification_in_chisq) {
				delta_theta[0] = mag00[j] * delta_beta[0] + mag01[j] * delta_beta[1];
				delta_theta[1] = mag01[j] * delta_beta[0] + mag11[j] * delta_beta[1];
				chisq += delta_theta.sqrnorm() / SQR(image_data[i].sigma_pos[j]);
			} else {
				chisq += delta_beta.sqrnorm() / SQR(image_data[i].sigma_pos[j]);
			}
		}
		//cout << "source_bf: " << src_bf[0] << " " << src_bf[1] << endl;
	}
	delete[] mag00;
	delete[] mag11;
	delete[] mag01;
	delete[] beta_ji;
	if ((group_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	return chisq;
}

void Lens::output_analytic_srcpos(lensvector *beta_i)
{
	// Note: beta_i needs to have the same size as the number of image sets being fit, or else a seg fault will occur
	int i,j;
	lensvector beta_ji;
	lensmatrix mag, magsqr;
	lensmatrix amatrix, ainv;
	lensvector bvec;
	lensmatrix jac;

	double siginv, src_norm;
	for (i=0; i < n_sourcepts_fit; i++) {
		amatrix[0][0] = amatrix[0][1] = amatrix[1][0] = amatrix[1][1] = 0;
		bvec[0] = bvec[1] = 0;
		beta_i[i][0] = beta_i[i][1] = 0;
		src_norm=0;
		for (j=0; j < image_data[i].n_images; j++) {
			find_sourcept(image_data[i].pos[j],beta_ji,0,zfactors[i]);
			if (use_magnification_in_chisq) {
				hessian(image_data[i].pos[j],jac,zfactors[i]);
				jac[0][0] = 1 - jac[0][0];
				jac[1][1] = 1 - jac[1][1];
				jac[0][1] = -jac[0][1];
				jac[1][0] = -jac[1][0];
				mag = jac.inverse();
				lensmatsqr(mag,magsqr);
				siginv = 1.0/SQR(image_data[i].sigma_pos[j]);
				amatrix[0][0] += magsqr[0][0]*siginv;
				amatrix[1][0] += magsqr[1][0]*siginv;
				amatrix[0][1] += magsqr[0][1]*siginv;
				amatrix[1][1] += magsqr[1][1]*siginv;
				bvec[0] += (magsqr[0][0]*beta_ji[0] + magsqr[0][1]*beta_ji[1])*siginv;
				bvec[1] += (magsqr[1][0]*beta_ji[0] + magsqr[1][1]*beta_ji[1])*siginv;
			} else {
				siginv = 1.0/SQR(image_data[i].sigma_pos[j]);
				beta_i[i][0] += beta_ji[0]*siginv;
				beta_i[i][1] += beta_ji[1]*siginv;
				src_norm += siginv;
			}
		}
		if (use_magnification_in_chisq) {
			if (amatrix.invert(ainv)==false) return;
			beta_i[i] = ainv*bvec;
		} else {
			beta_i[i][0] /= src_norm;
			beta_i[i][1] /= src_norm;
		}
	}
	return;
}

double Lens::chisq_pos_image_plane()
{
	double chisq=0;

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int n_images, n_tot_images=0;
	double chisq_each_srcpt;
	int i,j,k,n;
	for (i=0; i < n_sourcepts_fit; i++) {
		create_grid(false,zfactors[i]);
		chisq_each_srcpt = 0;
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime0 = omp_get_wtime();
		}
#endif
		image *img = get_images(sourcepts_fit[i], n_images, false);
		n_visible_images = n_images;
		n_tot_images += n_visible_images;

		if (!include_central_image) { for (j=0; j < n_images; j++) if ((img[j].parity == 1) and (kappa(img[j].pos,zfactors[i]) > 1)) n_visible_images--; }
		if ((n_images_penalty==true) and (n_visible_images != image_data[i].n_images)) {
			return 1e30;
		}

		int n_dists = n_visible_images*image_data[i].n_images;
		double *distsqrs = new double[n_dists];
		int *data_k = new int[n_dists];
		int *model_j = new int[n_dists];
		n=0;
		for (k=0; k < image_data[i].n_images; k++) {
			for (j=0; j < n_images; j++) {
				distsqrs[n] = SQR(image_data[i].pos[k][0] - img[j].pos[0]) + SQR(image_data[i].pos[k][1] - img[j].pos[1]);
				if ((!include_central_image) and (img[j].parity == 1) and (kappa(img[j].pos,zfactors[i]) > 1)) continue;
				data_k[n] = k;
				model_j[n] = j;
				n++;
			}
		}
		if (n != n_dists) die("count of all data-model image combinations does not equal expected number (%i vs %i)",n,n_dists);
		sort(n_dists,distsqrs,data_k,model_j);
		int *closest_image_j = new int[image_data[i].n_images];
		int *closest_image_k = new int[n_images];
		double *closest_distsqrs = new double[image_data[i].n_images];
		for (k=0; k < image_data[i].n_images; k++) closest_image_j[k] = -1;
		for (j=0; j < n_images; j++) closest_image_k[j] = -1;
		int m=0;
		int mmax = dmin(n_visible_images,image_data[i].n_images);
		for (n=0; n < n_dists; n++) {
			if ((closest_image_j[data_k[n]] == -1) and (closest_image_k[model_j[n]] == -1)) {
				closest_image_j[data_k[n]] = model_j[n];
				closest_image_k[model_j[n]] = data_k[n];
				closest_distsqrs[data_k[n]] = distsqrs[n];
				m++;
				if (m==mmax) n = n_dists; // force loop to exit
			}
		}

		for (k=0; k < image_data[i].n_images; k++) {
			if (closest_image_j[k] != -1) {
				chisq_each_srcpt += closest_distsqrs[k]/SQR(image_data[i].sigma_pos[k]);
			} else {
				// add a penalty value to chi-square for not reproducing this data image; the distance is twice the maximum distance between any pair of images
				chisq_each_srcpt += 4*image_data[i].max_distsqr/SQR(image_data[i].sigma_pos[k]);
			}
		}
		chisq += chisq_each_srcpt;
		delete[] distsqrs;
		delete[] data_k;
		delete[] model_j;
		delete[] closest_image_j;
		delete[] closest_image_k;
		delete[] closest_distsqrs;
	}
	if ((group_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	if (n_sourcepts_fit > 1) n_visible_images = n_tot_images; // save the total number of visible images produced
	return chisq;
}

double Lens::chisq_pos_image_plane2()
{
	double chisq=0;

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int n_images, n_visible_images, n_tot_images=0;
	double chisq_each_srcpt;
	int i,j,k;
	for (i=0; i < n_sourcepts_fit; i++) {
		create_grid(false,zfactors[i]);
		chisq_each_srcpt = 0;
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime0 = omp_get_wtime();
		}
#endif
		image *img = get_images(sourcepts_fit[i], n_images, false);
		n_visible_images = n_images;
		if (!include_central_image) { for (j=0; j < n_images; j++) if ((img[j].parity == 1) and (kappa(img[j].pos,zfactors[i]) > 1)) n_visible_images--; }
		if ((n_images_penalty==true) and (n_visible_images != image_data[i].n_images)) return 1e30;
		n_tot_images += n_visible_images;
		double distsqr, distsqr_min, sig_min;
		for (k=0; k < image_data[i].n_images; k++) {
			distsqr_min=1e30;
			for (j=0; j < n_images; j++) {
				if ((!include_central_image) and (img[j].parity == 1) and (kappa(img[j].pos,zfactors[i]) > 1)) continue;
				distsqr = SQR(image_data[i].pos[k][0] - img[j].pos[0]) + SQR(image_data[i].pos[k][1] - img[j].pos[1]);
				if (distsqr < distsqr_min) { distsqr_min = distsqr; sig_min = image_data[i].sigma_pos[k]; }
			}
			chisq_each_srcpt += distsqr_min/SQR(sig_min);
		}
		chisq += chisq_each_srcpt;
	}
	if ((group_id==0) and (logfile.is_open())) logfile << "it=" << chisq_it << " chisq=" << chisq << endl;
	if (n_sourcepts_fit > 1) n_visible_images = n_tot_images; // save the total number of visible images produced
	return chisq;
}

void Lens::output_model_source_flux(double *bestfit_flux)
{
	double chisq=0;
	int n_total_images=0;
	int i,j,k=0;

	for (i=0; i < n_sourcepts_fit; i++)
		for (j=0; j < image_data[i].n_images; j++) n_total_images++;
	double image_mag;

	lensmatrix jac;
	for (i=0; i < n_sourcepts_fit; i++) {
		double num=0, denom=0;
		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_f[j]==0) { k++; continue; }
			hessian(image_data[i].pos[j],jac,zfactors[i]);
			jac[0][0] = 1 - jac[0][0];
			jac[1][1] = 1 - jac[1][1];
			jac[0][1] = -jac[0][1];
			jac[1][0] = -jac[1][0];
			image_mag = 1.0/determinant(jac);
			if (include_parity_in_chisq) {
				num += image_data[i].flux[j] * image_mag / SQR(image_data[i].sigma_f[j]);
			} else {
				num += abs(image_data[i].flux[j] * image_mag) / SQR(image_data[i].sigma_f[j]);
			}
			denom += SQR(image_mag/image_data[i].sigma_f[j]);
			k++;
		}
		if (denom==0) bestfit_flux[i] = -1; // indicates we cannot find the source flux
		else bestfit_flux[i] = num/denom;
	}
}

double Lens::chisq_flux()
{
	double chisq=0;
	int n_images_hi=0;
	int i,j,k;

	for (i=0; i < n_sourcepts_fit; i++) {
		if (image_data[i].n_images > n_images_hi) n_images_hi = image_data[i].n_images;
	}
	double* image_mags = new double[n_images_hi];

	lensmatrix jac;
	double flux_src, num, denom;
	for (i=0; i < n_sourcepts_fit; i++) {
		k=0; num=0; denom=0;
		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_f[j]==0) { k++; continue; }
			hessian(image_data[i].pos[j],jac,zfactors[i]);
			jac[0][0] = 1 - jac[0][0];
			jac[1][1] = 1 - jac[1][1];
			jac[0][1] = -jac[0][1];
			jac[1][0] = -jac[1][0];
			image_mags[k] = 1.0/determinant(jac);
			if (include_parity_in_chisq) {
				num += image_data[i].flux[j] * image_mags[k] / SQR(image_data[i].sigma_f[j]);
			} else {
				num += abs(image_data[i].flux[j] * image_mags[k]) / SQR(image_data[i].sigma_f[j]);
			}
			denom += SQR(image_mags[k]/image_data[i].sigma_f[j]);
			k++;
		}

		if (fix_source_flux) {
			flux_src = source_flux; // only one source flux value is currently supported; later this should be generalized so that
											// some fluxes can be fixed and others parameterized
		}
		else {
			// the source flux is calculated analytically, rather than including it as a fit parameter (see Keeton 2001, section 4.2)
			flux_src = num / denom;
		}

		k=0;
		if (include_parity_in_chisq) {
			for (j=0; j < image_data[i].n_images; j++) {
				if (image_data[i].sigma_f[j]==0) { k++; continue; }
				chisq += SQR((image_data[i].flux[j] - image_mags[k++]*flux_src)/image_data[i].sigma_f[j]);
			}
		} else {
			for (j=0; j < image_data[i].n_images; j++) {
				if (image_data[i].sigma_f[j]==0) { k++; continue; }
				chisq += SQR((abs(image_data[i].flux[j]) - abs(image_mags[k++]*flux_src))/image_data[i].sigma_f[j]);
			}
		}
	}

	delete[] image_mags;
	return chisq;
}

double Lens::chisq_time_delays()
{
	double chisq=0;
	int n_images_hi=0;
	int i,j,k;

	for (i=0; i < n_sourcepts_fit; i++) {
		if (image_data[i].n_images > n_images_hi) n_images_hi = image_data[i].n_images;
	}

	double td_factor;
	double* time_delays_obs = new double[n_images_hi];
	double* time_delays_mod = new double[n_images_hi];
	double min_td_obs, min_td_mod;
	double pot;
	lensvector beta_ij;
	for (k=0, i=0; i < n_sourcepts_fit; i++) {
		td_factor = time_delay_factor_arcsec(lens_redshift,source_redshifts[i]);
		min_td_obs=1e30;
		min_td_mod=1e30;
		for (j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_t[j]==0) continue;
			find_sourcept(image_data[i].pos[j],beta_ij,0,zfactors[i]);
			pot = potential(image_data[i].pos[j],zfactors[i]);
			time_delays_mod[j] = 0.5*(SQR(image_data[i].pos[j][0] - beta_ij[0]) + SQR(image_data[i].pos[j][1] - beta_ij[1])) - pot;
			if (time_delays_mod[j] < min_td_mod) min_td_mod = time_delays_mod[j];

			if (image_data[i].time_delays[j] < min_td_obs) min_td_obs = image_data[i].time_delays[j];
		}
		for (k=0, j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_t[j]==0) { k++; continue; }
			time_delays_mod[k] -= min_td_mod;
			if (time_delays_mod[k] != 0.0) time_delays_mod[k] *= td_factor; // td_factor contains the cosmological factors and is in units of days
			time_delays_obs[k] = image_data[i].time_delays[j] - min_td_obs;
			k++;
		}
		for (k=0, j=0; j < image_data[i].n_images; j++) {
			if (image_data[i].sigma_t[j]==0) { k++; continue; }
			chisq += SQR((time_delays_obs[k] - time_delays_mod[k]) / image_data[i].sigma_t[j]);
			k++;
		}
	}
	if (chisq==0) warn("no time delay information has been used for chi-square");

	delete[] time_delays_obs;
	delete[] time_delays_mod;
	return chisq;
}

void Lens::get_automatic_initial_stepsizes(dvector& stepsizes)
{
	if (nlens == 0) { warn(warnings,"No fit model has been specified"); return; }
	int i, index=0;
	for (i=0; i < nlens; i++) lens_list[i]->get_auto_stepsizes(stepsizes,index);
	if (source_fit_mode==Point_Source) {
		if (source_plane_rscale==0.0) autogrid(); // autogrid sets source_plane_rscale, which is the scale for the source plane caustics
		if ((!use_analytic_bestfit_src) or (use_image_plane_chisq) or (use_image_plane_chisq2)) {
			for (i=0; i < n_sourcepts_fit; i++) {
				if (vary_sourcepts_x[i]) stepsizes[index++] = 0.33*source_plane_rscale;
				if (vary_sourcepts_y[i]) stepsizes[index++] = 0.33*source_plane_rscale;
			}
		}
	}
	else if ((vary_regularization_parameter) and (source_fit_mode==Pixellated_Source) and (regularization_method != None)) {
		stepsizes[index++] = 0.33*regularization_parameter;
	}
	if (vary_pixel_fraction) stepsizes[index++] = 0.3;
	if (vary_magnification_threshold) stepsizes[index++] = 0.3;
	if (vary_hubble_parameter) stepsizes[index++] = 0.3;
	if (index != n_fit_parameters) die("Index didn't go through all the fit parameters");
}

void Lens::get_n_fit_parameters(int &nparams)
{
	lensmodel_fit_parameters=0;
	for (int i=0; i < nlens; i++)
		lensmodel_fit_parameters += lens_list[i]->get_n_vary_params();
	nparams = lensmodel_fit_parameters;
	if (source_fit_mode==Point_Source) {
		if ((!use_analytic_bestfit_src) or (use_image_plane_chisq) or (use_image_plane_chisq2)) {
			for (int i=0; i < n_sourcepts_fit; i++) {
				if (vary_sourcepts_x[i]) nparams++;
				if (vary_sourcepts_y[i]) nparams++;
			}
		}
	}
	else if ((vary_regularization_parameter) and (source_fit_mode==Pixellated_Source) and (regularization_method != None)) nparams++;
	if (vary_pixel_fraction) nparams++;
	if (vary_magnification_threshold) nparams++;
	if (vary_hubble_parameter) nparams++;
}

bool Lens::setup_fit_parameters(bool include_limits)
{
	if (source_fit_mode==Point_Source) {
		if (image_data==NULL) { warn("cannot do fit; image data points have not been loaded"); return false; }
		if (sourcepts_fit==NULL) { warn("cannot do fit; initial source parameters have not been defined"); return false; }
	} else if (source_fit_mode==Pixellated_Source) {
		if (image_pixel_data==NULL) { warn("cannot do fit; image pixel data has not been loaded"); return false; }
	}
	if (nlens==0) { warn("cannot do fit; no lens models have been defined"); return false; }
	get_n_fit_parameters(n_fit_parameters);
	if (n_fit_parameters==0) { warn("cannot do fit; no parameters are being varied"); return false; }
	fitparams.input(n_fit_parameters);
	int index = 0;
	for (int i=0; i < nlens; i++) lens_list[i]->get_fit_parameters(fitparams,index);
	if (index != lensmodel_fit_parameters) die("Index didn't go through all the lens model fit parameters");
	if (source_fit_mode==Point_Source) {
		if ((!use_analytic_bestfit_src) or (use_image_plane_chisq) or (use_image_plane_chisq2)) {
			for (int i=0; i < n_sourcepts_fit; i++) {
				if (vary_sourcepts_x[i]) fitparams[index++] = sourcepts_fit[i][0];
				if (vary_sourcepts_y[i]) fitparams[index++] = sourcepts_fit[i][1];
			}
		}
	} else if ((vary_regularization_parameter) and (source_fit_mode==Pixellated_Source) and (regularization_method != None)) fitparams[index++] = regularization_parameter;
	if (vary_pixel_fraction) fitparams[index++] = pixel_fraction;
	if (vary_magnification_threshold) fitparams[index++] = pixel_magnification_threshold;
	if (vary_hubble_parameter) fitparams[index++] = hubble;
	get_parameter_names();
	dvector stepsizes(n_fit_parameters);
	get_automatic_initial_stepsizes(stepsizes);
	param_settings->update_params(n_fit_parameters,fit_parameter_names,stepsizes.array());
	param_settings->transform_parameters(fitparams.array());
	transformed_parameter_names.resize(n_fit_parameters);
	param_settings->transform_parameter_names(fit_parameter_names.data(),transformed_parameter_names.data());

	//if (mpi_id==0) {
		//cout << "Initial parameters: ";
		//for (int i=0; i < n_fit_parameters; i++) cout << fitparams[i] << " ";
		//cout << endl;
	//}
	if (include_limits) {
		upper_limits.input(n_fit_parameters);
		lower_limits.input(n_fit_parameters);
		upper_limits_initial.input(n_fit_parameters);
		lower_limits_initial.input(n_fit_parameters);
		index=0;
		for (int i=0; i < nlens; i++) {
			if ((lens_list[i]->get_n_vary_params() > 0) and (lens_list[i]->get_limits(lower_limits,upper_limits,lower_limits_initial,upper_limits_initial,index)==false)) { warn("cannot do fit; limits have not been defined for lens %i",i); return false; }
		}
		if (index != lensmodel_fit_parameters) die("index didn't go through all the lens model fit parameters when setting upper/lower limits");
		if (source_fit_mode==Point_Source) {
			if ((!use_analytic_bestfit_src) or (use_image_plane_chisq) or (use_image_plane_chisq2)) {
				for (int i=0; i < n_sourcepts_fit; i++) {
					if (vary_sourcepts_x[i]) {
						lower_limits[index] = sourcepts_lower_limit[i][0];
						lower_limits_initial[index] = lower_limits[index]; // make it possible to specify initial limits for source point!
						upper_limits[index] = sourcepts_upper_limit[i][0];
						upper_limits_initial[index] = upper_limits[index]; // make it possible to specify initial limits for source point!
						index++;
					}
					if (vary_sourcepts_y[i]) {
						lower_limits[index] = sourcepts_lower_limit[i][1];
						lower_limits_initial[index] = lower_limits[index]; // make it possible to specify initial limits for source point!
						upper_limits[index] = sourcepts_upper_limit[i][1];
						upper_limits_initial[index] = upper_limits[index]; // make it possible to specify initial limits for source point!
						index++;
					}
				}
			}
		} else if (source_fit_mode == Pixellated_Source) {
			if ((vary_regularization_parameter) and (regularization_method != None)) {
				if ((regularization_parameter_lower_limit==1e30) or (regularization_parameter_upper_limit==1e30)) { warn("lower/upper limits must be set for regularization parameter (see 'regparam') before doing fit"); return false; }
				lower_limits[index] = regularization_parameter_lower_limit;
				lower_limits_initial[index] = lower_limits[index];
				upper_limits[index] = regularization_parameter_upper_limit;
				upper_limits_initial[index] = upper_limits[index];
				index++;
			}
			if (vary_pixel_fraction) {
				lower_limits[index] = pixel_fraction_lower_limit;
				lower_limits_initial[index] = lower_limits[index];
				upper_limits[index] = pixel_fraction_upper_limit;
				upper_limits_initial[index] = upper_limits[index];
				index++;
			}
			if (vary_magnification_threshold) {
				lower_limits[index] = pixel_magnification_threshold_lower_limit;
				lower_limits_initial[index] = lower_limits[index];
				upper_limits[index] = pixel_magnification_threshold_upper_limit;
				upper_limits_initial[index] = upper_limits[index];
				index++;
			}
		}
		if (vary_hubble_parameter) {
			lower_limits[index] = hubble_lower_limit;
			lower_limits_initial[index] = lower_limits[index];
			upper_limits[index] = hubble_upper_limit;
			upper_limits_initial[index] = upper_limits[index];
			index++;
		}
		if (index != n_fit_parameters) die("index didn't go through all the fit parameters when setting upper/lower limits (%i expected, %i found)",n_fit_parameters,index);
		param_settings->transform_limits(lower_limits.array(),upper_limits.array());
		param_settings->transform_limits(lower_limits_initial.array(),upper_limits_initial.array());
		for (int i=0; i < n_fit_parameters; i++) {
			if (lower_limits[i] > upper_limits[i]) {
				double temp = upper_limits[i]; upper_limits[i] = lower_limits[i]; lower_limits[i] = temp;
			}
			if (lower_limits_initial[i] > upper_limits_initial[i]) {
				double temp = upper_limits_initial[i]; upper_limits_initial[i] = lower_limits_initial[i]; lower_limits_initial[i] = temp;
			}
		}
	}
	return true;
}

void Lens::get_parameter_names()
{
	get_n_fit_parameters(n_fit_parameters);
	fit_parameter_names.clear();
	int i,j;
	for (i=0; i < nlens; i++) {
		lens_list[i]->get_fit_parameter_names(fit_parameter_names);
	}
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
					new_parameter_names[i] += countstring;
					count++;
				}
				stringstream countstr;
				string countstring;
				countstr << count;
				countstr >> countstring;
				fit_parameter_names[j] += countstring;
				count++;
			}
		}
		fit_parameter_names[i] = new_parameter_names[i];
	}
	delete[] new_parameter_names;
	if (source_fit_mode==Point_Source) {
		if ((!use_analytic_bestfit_src) or (use_image_plane_chisq) or (use_image_plane_chisq2)) {
			if (n_sourcepts_fit==1) {
				if (vary_sourcepts_x[0]) fit_parameter_names.push_back("xsrc");
				if (vary_sourcepts_y[0]) fit_parameter_names.push_back("ysrc");
			} else {
				for (i=0; i < n_sourcepts_fit; i++) {
					stringstream srcpt_num_str;
					string srcpt_num_string;
					srcpt_num_str << i;
					srcpt_num_str >> srcpt_num_string;
					if (vary_sourcepts_x[i]) {
						string beta_x_name = "xsrc" + srcpt_num_string;
						fit_parameter_names.push_back(beta_x_name);
					}
					if (vary_sourcepts_y[i]) {
						string beta_y_name = "ysrc" + srcpt_num_string;
						fit_parameter_names.push_back(beta_y_name);
					}
				}
			}
		}
	}
	else if ((vary_regularization_parameter) and (source_fit_mode==Pixellated_Source) and (regularization_method != None)) {
		fit_parameter_names.push_back("lambda");
	}
	if (vary_pixel_fraction) fit_parameter_names.push_back("pixel_fraction");
	if (vary_magnification_threshold) fit_parameter_names.push_back("mag_threshold");
	if (vary_hubble_parameter) fit_parameter_names.push_back("h0");
	if (fit_parameter_names.size() != n_fit_parameters) die("get_parameter_names() did not assign names to all the fit parameters (%i vs %i)",n_fit_parameters,fit_parameter_names.size());
}

void Lens::fit_set_optimizations()
{
	temp_auto_ccspline = auto_ccspline;
	temp_auto_store_cc_points = auto_store_cc_points;
	temp_enforce_min_cell_area = enforce_min_cell_area;
	temp_include_time_delays = include_time_delays;

	// turn the following features off because they add pointless overhead (they will be restored to their
	// former settings after the search is done)
	auto_ccspline = false;
	auto_store_cc_points = false;
	enforce_min_cell_area = false;
	include_time_delays = false; // calculating time delays from images found not necessary during fit, since the chisq_time_delays finds time delays separately

	fisher_inverse.erase(); // reset parameter covariance matrix in case it was used in a previous fit
}

void Lens::fit_restore_defaults()
{
	auto_ccspline = temp_auto_ccspline;
	auto_store_cc_points = temp_auto_store_cc_points;
	enforce_min_cell_area = temp_enforce_min_cell_area;
	include_time_delays = temp_include_time_delays;
	Grid::set_lens(this); // annoying that the grids can only point to one lens object--it would be better for the pointer to be non-static (implement this later)
}

void Lens::chisq_single_evaluation()
{
	if (setup_fit_parameters(false)==false) return;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	initialize_fitmodel();

	double (Lens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Lens::*)(double*)> (&Lens::fitmodel_loglike_point_source);
	} else if (source_fit_mode==Pixellated_Source) {
		loglikeptr = static_cast<double (Lens::*)(double*)> (&Lens::fitmodel_loglike_pixellated_source);
	}

	double chisq_initial = (this->*loglikeptr)(fitparams.array());
	if (chisq_initial==1e30) warn(warnings,"Your parameter values are returning a large \"penalty\" chi-square--this likely means one or\nmore parameters have unphysical values or are out of the bounds specified by 'fit plimits'");
	if (source_fit_mode==Point_Source) display_chisq_status = true;
	fitmodel->chisq_it = 0;
	double chisqval = 2 * (this->*loglikeptr)(fitparams.array());
	if (mpi_id==0) {
		if (display_chisq_status) cout << endl;
		cout << "loglike: " << chisqval/2 << endl;
	}
	if (source_fit_mode==Point_Source) display_chisq_status = false;

	fit_restore_defaults();
}

void Lens::plot_chisq_2d(const int param1, const int param2, const int n1, const double i1, const double f1, const int n2, const double i2, const double f2)
{
	if (setup_fit_parameters(false)==false) return;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	initialize_fitmodel();

	if (param1 >= n_fit_parameters) { warn("Parameter %i does not exist (%i parameters total)",param1,n_fit_parameters); return; }
	if (param2 >= n_fit_parameters) { warn("Parameter %i does not exist (%i parameters total)",param2,n_fit_parameters); return; }

	double (Lens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Lens::*)(double*)> (&Lens::fitmodel_loglike_point_source);
	} else if (source_fit_mode==Pixellated_Source) {
		loglikeptr = static_cast<double (Lens::*)(double*)> (&Lens::fitmodel_loglike_pixellated_source);
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
			fitparams[param1] = p1;
			fitparams[param2] = p2;
			chisqvals[i][j] = 2.0 * (this->*loglikeptr)(fitparams.array());
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

void Lens::plot_chisq_1d(const int param, const int n, const double ip, const double fp, string filename)
{
	if (setup_fit_parameters(false)==false) return;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	initialize_fitmodel();

	if (param >= n_fit_parameters) { warn("Parameter %i does not exist (%i parameters total)",param,n_fit_parameters); return; }

	double (Lens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Lens::*)(double*)> (&Lens::fitmodel_loglike_point_source);
	} else if (source_fit_mode==Pixellated_Source) {
		loglikeptr = static_cast<double (Lens::*)(double*)> (&Lens::fitmodel_loglike_pixellated_source);
	}

	double step = (fp-ip)/n;
	int i,j;
	double p;

	double chisqmin=1e30;
	dvector chisqvals(n);
	ofstream chisqout(filename.c_str());
	double pmin;
	for (i=0, p=ip+0.5*step; i < n; i++, p += step) {
		fitparams[param] = p;
		chisqvals[i] = 2.0 * (this->*loglikeptr)(fitparams.array());
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

double Lens::chi_square_fit_simplex()
{
	if (setup_fit_parameters(false)==false) return 0.0;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	initialize_fitmodel();

	if (fit_output_dir != ".") create_output_directory();

	double (Simplex::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Simplex::*)(double*)> (&Lens::fitmodel_loglike_point_source);
	} else if (source_fit_mode==Pixellated_Source) {
		loglikeptr = static_cast<double (Simplex::*)(double*)> (&Lens::fitmodel_loglike_pixellated_source);
	}

	dvector stepsizes(param_settings->stepsizes,n_fit_parameters);
	if (mpi_id==0) {
		cout << "Initial stepsizes: ";
		for (int i=0; i < n_fit_parameters; i++) cout << stepsizes[i] << " ";
		cout << endl << endl;
	}

	initialize_simplex(fitparams.array(),n_fit_parameters,stepsizes.array(),chisq_tolerance,-10);
	simplex_set_function(loglikeptr);
	simplex_set_fmin(simplex_minchisq);
	//int iterations = 0;
	//downhill_simplex(iterations,max_iterations,0); // last argument is temperature for simulated annealing, but there is no cooling schedule with this function
	set_annealing_schedule_parameters(simplex_temp_initial,simplex_temp_final,simplex_cooling_factor,simplex_nmax_anneal,simplex_nmax);
	int n_iterations;

	double chisq_initial = (this->*loglikeptr)(fitparams.array());
	if (chisq_initial==1e30) warn(warnings,"Your initial parameter values are returning a large \"penalty\" chi-square--this likely means\none or more parameters have unphysical values or are out of the bounds specified by 'fit plimits'");

	if (source_fit_mode==Point_Source) display_chisq_status = true;
	else display_chisq_status = false;

	fitmodel->chisq_it = 0;
	bool verbal = (mpi_id==0) ? true : false;
	n_iterations = downhill_simplex_anneal(verbal);
	simplex_minval(fitparams.array(),chisq_bestfit);
	chisq_bestfit *= 2; // since the loglike function actually returns 0.5*chisq
	if (display_chisq_status) {
		fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
		(this->*loglikeptr)(fitparams.array());
		if (mpi_id==0) cout << endl;
	}

	bool turned_on_chisqmag = false;
	if (n_repeats > 0) {
		if ((source_fit_mode==Point_Source) and (!use_magnification_in_chisq) and (use_magnification_in_chisq_during_repeats) and (!use_image_plane_chisq) and (!use_image_plane_chisq2)) {
			turned_on_chisqmag = true;
			use_magnification_in_chisq = true;
			fitmodel->use_magnification_in_chisq = true;
			simplex_evaluate_bestfit_point(); // need to re-evaluate and record the chi-square at the best-fit point since we are changing the chi-square function
			cout << "Now using magnification in position chi-square function during repeats...\n";
		}
		set_annealing_schedule_parameters(0,simplex_temp_final,simplex_cooling_factor,simplex_nmax_anneal,simplex_nmax); // repeats have zero temperature (just minimization)
		for (int i=0; i < n_repeats; i++) {
			if (mpi_id==0) cout << "Repeating optimization (trial " << i+1 << ")\n";
			n_iterations = downhill_simplex_anneal(verbal);
			simplex_minval(fitparams.array(),chisq_bestfit);
			chisq_bestfit *= 2; // since the loglike function actually returns 0.5*chisq
			if (display_chisq_status) {
				fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
				(this->*loglikeptr)(fitparams.array());
				if (mpi_id==0) cout << endl;
			}
		}
	}
	bestfitparams.input(fitparams);

	if (display_chisq_status) {
		display_chisq_status = false;
	}
	if (mpi_id==0) {
		if (simplex_exit_status==true) {
			if (simplex_temp_initial==0) cout << "Downhill simplex converged after " << n_iterations << " iterations\n\n";
			else cout << "Downhill simplex converged after " << n_iterations << " iterations at final temperature T=0\n\n";
		} else {
			cout << "Downhill simplex interrupted after " << n_iterations << " iterations\n\n";
		}
	}

	if (source_fit_mode==Pixellated_Source) {
		fitmodel->source_pixel_grid->plot_surface_brightness("src_calc");
		fitmodel->image_pixel_grid->plot_surface_brightness("img_calc");
	}

	bool fisher_matrix_is_nonsingular;
	if (calculate_parameter_errors) {
		if (mpi_id==0) cout << "Calculating parameter errors..." << flush;
		fisher_matrix_is_nonsingular = calculate_fisher_matrix(fitparams,stepsizes);
		if (fisher_matrix_is_nonsingular) bestfit_fisher_inverse.input(fisher_inverse);
		else bestfit_fisher_inverse.erase(); // just in case it was defined before
		cout << "done\n\n";
	}
	if (mpi_id==0) {
		if (use_scientific_notation) cout << setiosflags(ios::scientific);
		else {
			cout << resetiosflags(ios::scientific);
			cout.unsetf(ios_base::floatfield);
		}
		cout << "Best-fit model: chi-square = " << chisq_bestfit << endl;
		//cout << "Number of iterations: " << iterations << endl;
		fitmodel->print_lens_list(false);

		if (source_fit_mode == Point_Source) {
			lensvector *bestfit_src = new lensvector[n_sourcepts_fit];
			double *bestfit_flux;
			if (include_flux_chisq) {
				bestfit_flux = new double[n_sourcepts_fit];
				fitmodel->output_model_source_flux(bestfit_flux);
			};
			if ((use_analytic_bestfit_src) and (!use_image_plane_chisq) and (!use_image_plane_chisq2)) {
				fitmodel->output_analytic_srcpos(bestfit_src);
			} else {
				for (int i=0; i < n_sourcepts_fit; i++) bestfit_src[i] = fitmodel->sourcepts_fit[i];
			}
			for (int i=0; i < n_sourcepts_fit; i++) {
				cout << "src" << i << "_x=" << bestfit_src[i][0] << " src" << i << "_y=" << bestfit_src[i][1];
				if (include_flux_chisq) cout << " src" << i << "_flux=" << bestfit_flux[i];
				cout << endl;
			}
			delete[] bestfit_src;
			if (include_flux_chisq) delete[] bestfit_flux;
		}

		if ((vary_regularization_parameter) and (source_fit_mode == Pixellated_Source) and (regularization_method != None)) {
			cout << "regularization parameter lambda=" << fitmodel->regularization_parameter << endl;
		}
		if (vary_pixel_fraction) cout << "pixel fraction = " << fitmodel->pixel_fraction << endl;
		if (vary_magnification_threshold) cout << "magnification threshold = " << fitmodel->pixel_magnification_threshold << endl;
		if (vary_hubble_parameter) cout << "h0 = " << fitmodel->hubble << endl;

		cout << endl;
		if (calculate_parameter_errors) {
			if (fisher_matrix_is_nonsingular) {
				cout << "Marginalized 1-sigma errors from Fisher matrix:\n";
				for (int i=0; i < n_fit_parameters; i++) {
					cout << transformed_parameter_names[i] << ": " << fitparams[i] << " +/- " << sqrt(abs(fisher_inverse[i][i])) << endl;
				}
			} else {
				cout << "Error: Fisher matrix is singular, marginalized errors cannot be calculated\n";
				for (int i=0; i < n_fit_parameters; i++)
					cout << transformed_parameter_names[i] << ": " << fitparams[i] << endl;
			}
		} else {
			for (int i=0; i < n_fit_parameters; i++)
				cout << transformed_parameter_names[i] << ": " << fitparams[i] << endl;
		}
		cout << endl;
		if (auto_save_bestfit) output_bestfit_model();
	}

	if (turned_on_chisqmag) use_magnification_in_chisq = false; // restore chisqmag to original setting
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return chisq_bestfit;
}

double Lens::chi_square_fit_powell()
{
	if (setup_fit_parameters(false)==false) return 0.0;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	initialize_fitmodel();

	double (Powell::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Powell::*)(double*)> (&Lens::fitmodel_loglike_point_source);
	} else if (source_fit_mode==Pixellated_Source) {
		loglikeptr = static_cast<double (Powell::*)(double*)> (&Lens::fitmodel_loglike_pixellated_source);
	}

	initialize_powell(loglikeptr,chisq_tolerance);

	dvector stepsizes(param_settings->stepsizes,n_fit_parameters);
	if (mpi_id==0) {
		cout << "Initial stepsizes: ";
		for (int i=0; i < n_fit_parameters; i++) cout << stepsizes[i] << " ";
		cout << endl << endl;
	}

	double chisq_initial = (this->*loglikeptr)(fitparams.array());
	if (chisq_initial==1e30) warn(warnings,"Your initial parameter values are returning a large \"penalty\" chi-square--this likely means\none or more parameters have unphysical values or are out of the bounds specified by 'fit plimits'");

	if (source_fit_mode==Point_Source) display_chisq_status = true;
	else display_chisq_status = false;

	fitmodel->chisq_it = 0;
	powell_minimize(fitparams.array(),n_fit_parameters,stepsizes.array());
	chisq_bestfit = 2*(this->*loglikeptr)(fitparams.array());
	if (display_chisq_status) {
		fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
		(this->*loglikeptr)(fitparams.array());
		if (mpi_id==0) cout << endl;
	}

	bool turned_on_chisqmag = false;
	if (n_repeats > 0) {
		if ((source_fit_mode==Point_Source) and (!use_magnification_in_chisq) and (use_magnification_in_chisq_during_repeats) and (!use_image_plane_chisq) and (!use_image_plane_chisq2)) {
			turned_on_chisqmag = true;
			use_magnification_in_chisq = true;
			fitmodel->use_magnification_in_chisq = true;
			cout << "Now using magnification in position chi-square function during repeats...\n";
		}
		for (int i=0; i < n_repeats; i++) {
			if (mpi_id==0) cout << "Repeating optimization (trial " << i+1 << ")\n";
			powell_minimize(fitparams.array(),n_fit_parameters,stepsizes.array());
			chisq_bestfit = 2*(this->*loglikeptr)(fitparams.array());
			if (display_chisq_status) {
				fitmodel->chisq_it = 0; // To ensure it displays the chi-square status
				(this->*loglikeptr)(fitparams.array());
				if (mpi_id==0) cout << endl;
			}
		}
	}
	bestfitparams.input(fitparams);

	if (display_chisq_status) {
		display_chisq_status = false;
	}

	if (group_id==0) fitmodel->logfile << "Optimization finished: min chisq = " << chisq_bestfit << endl;

	if (source_fit_mode==Pixellated_Source) {
		if (mpi_id==0) fitmodel->source_pixel_grid->plot_surface_brightness("src_calc");
		if (mpi_id==0) fitmodel->image_pixel_grid->plot_surface_brightness("img_calc");
	}
	bool fisher_matrix_is_nonsingular;
	if (calculate_parameter_errors) {
		if (mpi_id==0) cout << "Calculating parameter errors..." << flush;
		fisher_matrix_is_nonsingular = calculate_fisher_matrix(fitparams,stepsizes);
		if (fisher_matrix_is_nonsingular) bestfit_fisher_inverse.input(fisher_inverse);
		else bestfit_fisher_inverse.erase(); // just in case it was defined before
		cout << "done\n\n";
	}
	if (mpi_id==0) {
		if (use_scientific_notation) cout << setiosflags(ios::scientific);
		else {
			cout << resetiosflags(ios::scientific);
			cout.unsetf(ios_base::floatfield);
		}

		cout << "\nBest-fit model: chi-square = " << chisq_bestfit << endl;
		update_fitmodel(fitparams.array());
		fitmodel->print_lens_list(false);

		if (source_fit_mode == Point_Source) {
			lensvector *bestfit_src = new lensvector[n_sourcepts_fit];
			double *bestfit_flux;
			if (include_flux_chisq) {
				bestfit_flux = new double[n_sourcepts_fit];
				fitmodel->output_model_source_flux(bestfit_flux);
			};
			if ((use_analytic_bestfit_src) and (!use_image_plane_chisq) and (!use_image_plane_chisq2)) {
				fitmodel->output_analytic_srcpos(bestfit_src);
			} else {
				for (int i=0; i < n_sourcepts_fit; i++) bestfit_src[i] = fitmodel->sourcepts_fit[i];
			}
			for (int i=0; i < n_sourcepts_fit; i++) {
				cout << "src" << i << "_x=" << bestfit_src[i][0] << " src" << i << "_y=" << bestfit_src[i][1];
				if (include_flux_chisq) cout << " src" << i << "_flux=" << bestfit_flux[i];
				cout << endl;
			}
			delete[] bestfit_src;
			if (include_flux_chisq) delete[] bestfit_flux;
		}

		if ((vary_regularization_parameter) and (source_fit_mode == Pixellated_Source) and (regularization_method != None)) {
			cout << "regularization parameter lambda=" << fitmodel->regularization_parameter << endl;
		}
		if (vary_pixel_fraction) cout << "pixel fraction = " << fitmodel->pixel_fraction << endl;
		if (vary_magnification_threshold) cout << "magnification threshold = " << fitmodel->pixel_magnification_threshold << endl;
		if (vary_hubble_parameter) cout << "h0 = " << fitmodel->hubble << endl;
		cout << endl;
		if (calculate_parameter_errors) {
			if (fisher_matrix_is_nonsingular) {
				cout << "Marginalized 1-sigma errors from Fisher matrix:\n";
				for (int i=0; i < n_fit_parameters; i++) {
					cout << transformed_parameter_names[i] << ": " << fitparams[i] << " +/- " << sqrt(abs(fisher_inverse[i][i])) << endl;
				}
			} else {
				cout << "Error: Fisher matrix is singular, marginalized errors cannot be calculated\n";
				for (int i=0; i < n_fit_parameters; i++)
					cout << transformed_parameter_names[i] << ": " << fitparams[i];
			}
		} else {
			for (int i=0; i < n_fit_parameters; i++)
				cout << transformed_parameter_names[i] << ": " << fitparams[i];
		}
		cout << endl;
		if (auto_save_bestfit) output_bestfit_model();
	}

	if (turned_on_chisqmag) use_magnification_in_chisq = false; // restore chisqmag to original setting
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
	return chisq_bestfit;
}

void Lens::chi_square_nested_sampling()
{
	if (setup_fit_parameters(true)==false) return;
	fit_set_optimizations();
	if ((mpi_id==0) and (fit_output_dir != ".")) {
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		system(rmstring.c_str()); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
		// I should probably give the nested sampling output a unique extension like ".nest" or something, so that mkdist can't ever confuse it with twalk output in the same dir
		// Do this later...
		create_output_directory();
	}

	initialize_fitmodel();
	InputPoint(fitparams.array(),upper_limits.array(),lower_limits.array(),upper_limits_initial.array(),lower_limits_initial.array(),n_fit_parameters);

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&Lens::fitmodel_loglike_point_source);
	} else if (source_fit_mode==Pixellated_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&Lens::fitmodel_loglike_pixellated_source);
	}

	string *transformed_names = new string[n_fit_parameters];
	param_settings->transform_parameter_names(fit_parameter_names.data(),transformed_names);
	if (mpi_id==0) {
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (int i=0; i < n_fit_parameters; i++) pnamefile << transformed_names[i] << endl;
		pnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (int i=0; i < n_fit_parameters; i++)
		{
			prangefile << lower_limits[i] << " " << upper_limits[i] << endl;
		}
		prangefile.close();
	}

	double *param_errors = new double[n_fit_parameters];
#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	string filename = fit_output_dir + "/" + fit_output_filename;
	//if (use_image_plane_chisq2) display_chisq_status = true;

	MonoSample(filename.c_str(),n_mcpoints,fitparams.array(),param_errors,mcmc_logfile);
	bestfitparams.input(fitparams);

	//if (display_chisq_status) {
		//for (int i=0; i < n_sourcepts_fit; i++) cout << endl; // to get past the status signs for image position chi-square
		//cout << endl;
		//display_chisq_status = false;
	//}

#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for nested sampling: " << wt << endl;
	}
#endif

	if (mpi_id==0) {
		cout << endl;
		if (source_fit_mode == Point_Source) {
			lensvector *bestfit_src = new lensvector[n_sourcepts_fit];
			double *bestfit_flux;
			if (include_flux_chisq) {
				bestfit_flux = new double[n_sourcepts_fit];
				fitmodel->output_model_source_flux(bestfit_flux);
			};
			if ((use_analytic_bestfit_src) and (!use_image_plane_chisq) and (!use_image_plane_chisq2)) {
				fitmodel->output_analytic_srcpos(bestfit_src);
			} else {
				for (int i=0; i < n_sourcepts_fit; i++) bestfit_src[i] = fitmodel->sourcepts_fit[i];
			}
			for (int i=0; i < n_sourcepts_fit; i++) {
				cout << "src" << i << "_x=" << bestfit_src[i][0] << " src" << i << "_y=" << bestfit_src[i][1];
				if (include_flux_chisq) cout << " src" << i << "_flux=" << bestfit_flux[i];
				cout << endl;
			}
			delete[] bestfit_src;
			if (include_flux_chisq) delete[] bestfit_flux;
		}

		cout << "\nBest-fit parameters and errors:\n";
		for (int i=0; i < n_fit_parameters; i++) {
			cout << transformed_names[i] << ": " << fitparams[i] << " +/- " << param_errors[i] << endl;
		}
		cout << endl;
		if (auto_save_bestfit) output_bestfit_model();
	}
	delete[] param_errors;

	delete[] transformed_names;
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
}

void Lens::test_fitmodel_invert()
{
	if (setup_fit_parameters(false)==false) return;
	fit_set_optimizations();
	if (fit_output_dir != ".") create_output_directory();
	initialize_fitmodel();
	fitmodel_loglike_pixellated_source_test(fitparams.array());
	if (mpi_id==0) {
		cout << endl;
		cout << "Testing inversion again to make sure values are consistent...\n";
	}
	fitmodel_loglike_pixellated_source_test(fitparams.array());
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
}

void Lens::chi_square_twalk()
{
	if (setup_fit_parameters(true)==false) return;
	fit_set_optimizations();
	if ((mpi_id==0) and (fit_output_dir != ".")) {
		string rmstring = "if [ -e " + fit_output_dir + " ]; then rm -r " + fit_output_dir + "; fi";
		system(rmstring.c_str()); // delete the old output directory and remake it, just in case there is old data that might get mixed up when running mkdist
		create_output_directory();
	}
	initialize_fitmodel();
	InputPoint(fitparams.array(),upper_limits.array(),lower_limits.array(),upper_limits_initial.array(),lower_limits_initial.array(),n_fit_parameters);

	if (source_fit_mode==Point_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&Lens::fitmodel_loglike_point_source);
	} else if (source_fit_mode==Pixellated_Source) {
		LogLikePtr = static_cast<double (UCMC::*)(double*)> (&Lens::fitmodel_loglike_pixellated_source);
	}

	string *transformed_names = new string[n_fit_parameters];
	param_settings->transform_parameter_names(fit_parameter_names.data(),transformed_names);
	if (mpi_id==0) {
		string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
		ofstream pnamefile(pnamefile_str.c_str());
		for (int i=0; i < n_fit_parameters; i++) pnamefile << transformed_names[i] << endl;
		pnamefile.close();
		string prange_str = fit_output_dir + "/" + fit_output_filename + ".ranges";
		ofstream prangefile(prange_str.c_str());
		for (int i=0; i < n_fit_parameters; i++)
		{
			prangefile << lower_limits[i] << " " << upper_limits[i] << endl;
		}
		prangefile.close();
	}

#ifdef USE_OPENMP
	double wt0, wt;
	if (show_wtime) {
		wt0 = omp_get_wtime();
	}
#endif
	string filename = fit_output_dir + "/" + fit_output_filename;
	//if (use_image_plane_chisq2) display_chisq_status = true;

	TWalk(filename.c_str(),0.9836,4,2.4,2.5,6.0,mcmc_tolerance,mcmc_threads,fitparams.array(),mcmc_logfile);
	bestfitparams.input(fitparams);

	//if (display_chisq_status) {
		//for (int i=0; i < n_sourcepts_fit; i++) cout << endl; // to get past the status signs for image position chi-square
		//cout << endl;
		//display_chisq_status = false;
	//}
#ifdef USE_OPENMP
	if (show_wtime) {
		wt = omp_get_wtime() - wt0;
		if (mpi_id==0) cout << "Time for T-Walk: " << wt << endl;
	}
#endif
	if (mpi_id==0) {
		if (auto_save_bestfit) output_bestfit_model();
	}

	delete[] transformed_names;
	fit_restore_defaults();
	delete fitmodel;
	fitmodel = NULL;
}

void Lens::use_bestfit_model()
{
	if (nlens == 0) { warn(warnings,"No fit model has been specified"); return; }
	if (n_fit_parameters == 0) { warn(warnings,"No best-fit point has been saved from a previous fit"); return; }
	if (bestfitparams.size() != n_fit_parameters) { warn(warnings,"Best-fit point number of params does not match current number"); return; }
	int i, index=0;
	double transformed_params[n_fit_parameters];
	param_settings->inverse_transform_parameters(bestfitparams.array(),transformed_params);
	bool status;
	for (i=0; i < nlens; i++) {
		lens_list[i]->update_fit_parameters(transformed_params,index,status);
	}
	update_anchored_parameters();
	reset_grid(); // this will force it to redraw the critical curves if needed
	if (source_fit_mode == Point_Source) {
		if ((use_analytic_bestfit_src) and (!use_image_plane_chisq) and (!use_image_plane_chisq2)) {
			output_analytic_srcpos(sourcepts_fit);
		} else {
			for (i=0; i < n_sourcepts_fit; i++) {
				if (vary_sourcepts_x[i]) sourcepts_fit[i][0] = transformed_params[index++];
				if (vary_sourcepts_y[i]) sourcepts_fit[i][1] = transformed_params[index++];
			}
		}
	} else if (source_fit_mode == Pixellated_Source) {
		if ((vary_regularization_parameter) and (regularization_method != None))
			regularization_parameter = transformed_params[index++];
		if (vary_pixel_fraction)
			pixel_fraction = transformed_params[index++];
		if (vary_magnification_threshold)
			pixel_magnification_threshold = transformed_params[index++];
	}
	if (vary_hubble_parameter) {
		hubble = transformed_params[index++];
		set_cosmology(omega_matter,0.04,hubble,2.215);
	}

	//if (index != n_fit_parameters) die("Index didn't go through all the fit parameters (%i)",n_fit_parameters);
	if (index != n_fit_parameters) warn("Index didn't go through all the fit parameters (%i); this likely means your current lens model does not match the lens model that was used for fitting.",n_fit_parameters);
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

bool Lens::calculate_fisher_matrix(const dvector &params, const dvector &stepsizes)
{
	// this function calculates the marginalized error using the Gaussian approximation
	// (only accurate if we are near maximum likelihood point and it is close to Gaussian around this point)
	static const double increment2 = 1e-4;
	if ((mpi_id==0) and (source_fit_mode==Point_Source) and (!use_image_plane_chisq2) and (!use_image_plane_chisq) and (!use_magnification_in_chisq)) warn("Fisher matrix errors may not be accurate if source plane chi-square is used without magnification");

	dmatrix fisher(n_fit_parameters,n_fit_parameters);
	fisher_inverse.erase();
	fisher_inverse.input(n_fit_parameters,n_fit_parameters);
	dvector xhi(params);
	dvector xlo(params);
	double x0, curvature;
	int i,j;
	double step;
	for (i=0; i < n_fit_parameters; i++) {
		x0 = params[i];
		xhi[i] += increment2*stepsizes[i];
		if ((param_settings->use_penalty_limits[i]==true) and (xhi[i] > param_settings->penalty_limits_hi[i])) xhi[i] = x0;
		xlo[i] -= increment2*stepsizes[i];
		if ((param_settings->use_penalty_limits[i]==true) and (xlo[i] < param_settings->penalty_limits_lo[i])) xlo[i] = x0;
		step = xhi[i] - xlo[i];
		for (j=0; j < n_fit_parameters; j++) {
			fisher[i][j] = (loglike_deriv(xhi,j,stepsizes[j]) - loglike_deriv(xlo,j,stepsizes[j])) / step;
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
	for (i=1; i < n_fit_parameters; i++) {
		for (j=0; j < i; j++) {
			offdiag_avg = 0.5*(fisher[i][j]+ fisher[j][i]);
			//cout << i << " " << j << " " << fisher[i][j] << " " << fisher[j][i] << " " << abs((fisher[i][j]-fisher[j][i])/offdiag_avg) << endl;
			//if (abs((fisher[i][j]-fisher[j][i])/offdiag_avg) > 0.01) die("Fisher off-diags differ by more than 1%!");
			fisher[i][j] = fisher[j][i] = offdiag_avg;
		}
	}
	bool nonsingular;
	fisher.inverse(fisher_inverse,nonsingular);
	if (!nonsingular) {
		warn(warnings,"Fisher matrix is singular, cannot be inverted\n");
		fisher_inverse.erase();
		return false;
	}
	return true;
}

double Lens::loglike_deriv(const dvector &params, const int index, const double step)
{
	static const double increment = 1e-5;
	dvector xhi(params);
	dvector xlo(params);
	double dif, x0 = xhi[index];
	xhi[index] += increment*step;
	if ((param_settings->use_penalty_limits[index]==true) and (xhi[index] > param_settings->penalty_limits_hi[index])) xhi[index] = x0;
	xlo[index] -= increment*step;
	if ((param_settings->use_penalty_limits[index]==true) and (xlo[index] < param_settings->penalty_limits_lo[index])) xlo[index] = x0;
	dif = xhi[index] - xlo[index];
	double (Lens::*loglikeptr)(double*);
	if (source_fit_mode==Point_Source) {
		loglikeptr = static_cast<double (Lens::*)(double*)> (&Lens::fitmodel_loglike_point_source);
	} else if (source_fit_mode==Pixellated_Source) {
		loglikeptr = static_cast<double (Lens::*)(double*)> (&Lens::fitmodel_loglike_pixellated_source);
	}
	return (((this->*loglikeptr)(xhi.array()) - (this->*loglikeptr)(xlo.array())) / dif);
}

void Lens::output_bestfit_model()
{
	if (nlens == 0) { warn(warnings,"No fit model has been specified"); return; }
	if (n_fit_parameters == 0) { warn(warnings,"No best-fit point has been saved from a previous fit"); return; }
	if (bestfitparams.size() != n_fit_parameters) { warn(warnings,"Best-fit point number of params does not match current number"); return; }

	string pnamefile_str = fit_output_dir + "/" + fit_output_filename + ".paramnames";
	ofstream pnamefile(pnamefile_str.c_str());
	for (int i=0; i < n_fit_parameters; i++) pnamefile << transformed_parameter_names[i] << endl;
	pnamefile.close();

	string bestfit_filename = fit_output_dir + "/" + fit_output_filename + ".bf";
	int n,i,j;
	ofstream bf_out(bestfit_filename.c_str());
	for (i=0; i < n_fit_parameters; i++) bf_out << bestfitparams[i] << " ";
	bf_out << endl;
	bf_out.close();

	string outfile_str = fit_output_dir + "/" + fit_output_filename + ".bestfit";
	ofstream outfile(outfile_str.c_str());
	if (bestfit_fisher_inverse.is_initialized())
	{
		if (bestfit_fisher_inverse.rows() != n_fit_parameters) die("dimension of Fisher matrix does not match number of fit parameters (%i vs %i)",bestfit_fisher_inverse.rows(),n_fit_parameters);
		string fisher_inv_filename = fit_output_dir + "/" + fit_output_filename + ".pcov"; // inverse-fisher matrix is the parameter covariance matrix
		ofstream fisher_inv_out(fisher_inv_filename.c_str());
		for (i=0; i < n_fit_parameters; i++) {
			for (j=0; j < n_fit_parameters; j++) {
				fisher_inv_out << bestfit_fisher_inverse[i][j] << " ";
			}
			fisher_inv_out << endl;
		}

		outfile << "Best-fit model: chi-square = " << chisq_bestfit << endl;
		if ((include_flux_chisq) and (bestfit_flux != 0)) outfile << "Best-fit source flux = " << bestfit_flux << endl;
		outfile << endl;
		outfile << "Marginalized 1-sigma errors from Fisher matrix:\n";
		for (int i=0; i < n_fit_parameters; i++) {
			outfile << transformed_parameter_names[i] << ": " << bestfitparams[i] << " +/- " << sqrt(abs(bestfit_fisher_inverse[i][i])) << endl;
		}
		outfile << endl;
	} else {
		outfile << "Best-fit parameters (warning: errors are omitted here because Fisher matrix was not calculated):\n";
		for (int i=0; i < n_fit_parameters; i++) {
			outfile << transformed_parameter_names[i] << ": " << bestfitparams[i] << endl;
		}
		outfile << endl;
	}
}

double Lens::fitmodel_loglike_point_source(double* params)
{
	for (int i=0; i < n_fit_parameters; i++) {
		if (fitmodel->param_settings->use_penalty_limits[i]==true) {
			if ((params[i] < fitmodel->param_settings->penalty_limits_lo[i]) or (params[i] > fitmodel->param_settings->penalty_limits_hi[i])) return 1e30;
		}
	}
	double transformed_params[n_fit_parameters];
	fitmodel->param_settings->inverse_transform_parameters(params,transformed_params);
	if (update_fitmodel(transformed_params)==false) return 1e30;
	if (group_id==0) {
		if (fitmodel->logfile.is_open()) {
			for (int i=0; i < n_fit_parameters; i++) fitmodel->logfile << params[i] << " ";
		}
		fitmodel->logfile << flush;
	}

	double loglike, chisq_total=0, chisq;
	if (use_image_plane_chisq) {
		chisq = fitmodel->chisq_pos_image_plane();
		if ((display_chisq_status) and (mpi_id==0)) {
			int tot_data_images = 0;
			for (int i=0; i < n_sourcepts_fit; i++) tot_data_images += image_data[i].n_images;
			cout << "# images found: " << fitmodel->n_visible_images << " vs. " << tot_data_images << " data, ";
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_pos=" << chisq;
		}
	}
	else if (use_image_plane_chisq2) {
		chisq = fitmodel->chisq_pos_image_plane2();
		if ((display_chisq_status) and (mpi_id==0)) {
			cout << "# images found: " << fitmodel->n_visible_images << " vs. " << image_data[0].n_images << " data, ";
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_pos=" << chisq;
		}
	}
	else {
		chisq = fitmodel->chisq_pos_source_plane();
		if ((display_chisq_status) and (mpi_id==0)) {
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << "chisq_pos=" << chisq;
		}
	}
	chisq_total += chisq;
	if (include_flux_chisq) {
		chisq = fitmodel->chisq_flux();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << ", chisq_flux=" << chisq;
		}
	}
	if (include_time_delay_chisq) {
		chisq = fitmodel->chisq_time_delays();
		chisq_total += chisq;
		if ((display_chisq_status) and (mpi_id==0)) {
			if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << ", chisq_td=" << chisq;
		}
	}
	if ((display_chisq_status) and (mpi_id==0)) {
		if (fitmodel->chisq_it % chisq_display_frequency == 0) cout << ", chisq_tot=" << chisq_total << "               ";
		cout << endl;
		cout << "\033[1A";
	}

	loglike = chisq_total/2.0;

	fitmodel->param_settings->add_prior_terms_to_loglike(params,loglike);
	fitmodel->param_settings->add_jacobian_terms_to_loglike(transformed_params,loglike);
	fitmodel->chisq_it++;
	return loglike;
}

double Lens::fitmodel_loglike_pixellated_source(double* params)
{
	for (int i=0; i < n_fit_parameters; i++) {
		if (fitmodel->param_settings->use_penalty_limits[i]==true) {
			if ((params[i] < fitmodel->param_settings->penalty_limits_lo[i]) or (params[i] > fitmodel->param_settings->penalty_limits_hi[i])) return 1e30;
		}
	}
	double transformed_params[n_fit_parameters];
	fitmodel->param_settings->inverse_transform_parameters(params,transformed_params);
	if (update_fitmodel(transformed_params)==false) return 1e30;
	if (group_id==0) {
		if (fitmodel->logfile.is_open()) {
			for (int i=0; i < n_fit_parameters; i++) fitmodel->logfile << params[i] << " ";
			fitmodel->logfile << flush;
		}
	}
	double loglike, chisq=0;
	for (int i=0; i < nlens; i++) {
		if ((lens_list[i]->get_lenstype()==PJAFFE) or (lens_list[i]->get_lenstype()==CORECUSP)) {
			dvector subparams;
			lens_list[i]->get_extra_params(subparams);
			if (subparams[2] > subparams[1]) chisq=2e30; // don't allow s to be larger than a
		}
	}
	if (chisq != 2e30) {
		if (fitmodel->regularization_parameter < 0) chisq = 2e30;
		else if (fitmodel->pixel_fraction <= 0) chisq = 2e30;
		else chisq = fitmodel->invert_image_surface_brightness_map(false);
	}

	loglike = chisq/2.0;
	fitmodel->param_settings->add_prior_terms_to_loglike(params,loglike);
	fitmodel->param_settings->add_jacobian_terms_to_loglike(transformed_params,loglike);
	return loglike;
}

double Lens::fitmodel_loglike_pixellated_source_test(double* params)
{
	for (int i=0; i < n_fit_parameters; i++) {
		if (fitmodel->param_settings->use_penalty_limits[i]==true) {
			if ((params[i] < fitmodel->param_settings->penalty_limits_lo[i]) or (params[i] > fitmodel->param_settings->penalty_limits_hi[i])) return 1e30;
		}
	}
	double transformed_params[n_fit_parameters];
	fitmodel->param_settings->inverse_transform_parameters(params,transformed_params);
	if (update_fitmodel(transformed_params)==false) return 1e30;

	if (group_id==0) {
		if (fitmodel->logfile.is_open()) {
			for (int i=0; i < n_fit_parameters; i++) fitmodel->logfile << params[i] << " ";
			dvector subhalo_params;
		}
	}
	double loglike, chisq;
	if (fitmodel->regularization_parameter < 0) chisq = 2e30;
	else if (fitmodel->pixel_fraction <= 0) chisq = 2e30;
	else chisq = fitmodel->invert_image_surface_brightness_map(true);
	loglike = chisq/2.0;
	fitmodel->param_settings->add_prior_terms_to_loglike(params,loglike);
	fitmodel->param_settings->add_jacobian_terms_to_loglike(transformed_params,loglike);
	return loglike;
}

void Lens::set_Gauss_NN(const int& nn)
{
	Gauss_NN = nn;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->SetGaussLegendre(nn);
		}
	}
}

void Lens::set_romberg_accuracy(const double& acc)
{
	romberg_accuracy = acc;
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			lens_list[i]->set_romberg_accuracy(acc);
		}
	}
}

void Lens::print_lens_list(bool show_vary_params)
{
	cout << resetiosflags(ios::scientific);
	if (nlens > 0) {
		for (int i=0; i < nlens; i++) {
			cout << i << ". ";
			lens_list[i]->print_parameters();
			if (show_vary_params)
				lens_list[i]->print_vary_parameters();
		}
		if (source_redshift != reference_source_redshift) cout << "NOTE: for all lenses, kappa is scaled by zsrc_ref = " << reference_source_redshift << endl;
	}
	else cout << "No lens models have been specified" << endl;
	cout << endl;
	if (use_scientific_notation) cout << setiosflags(ios::scientific);
}

void Lens::print_fit_model()
{
	print_lens_list(true);
	if (source_fit_mode == Point_Source) {
		if (sourcepts_fit != NULL) {
			if ((!use_analytic_bestfit_src) or (use_image_plane_chisq) or (use_image_plane_chisq2)) {
				if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
					cout << "Initial fit coordinates for source points:\n";
					for (int i=0; i < n_sourcepts_fit; i++) cout << "Source point " << i << ": (" << sourcepts_fit[i][0] << "," << sourcepts_fit[i][1] << ")\n";
				} else {
					cout << "Initial fit coordinates and limits for source points:\n";
					for (int i=0; i < n_sourcepts_fit; i++) {
						cout << "Source point " << i << ": (" << sourcepts_fit[i][0] << "," << sourcepts_fit[i][1] << ")\n";
						cout << "x" << i << ": [" << sourcepts_lower_limit[i][0] << ":" << sourcepts_upper_limit[i][0] << "]\n";
						cout << "y" << i << ": [" << sourcepts_lower_limit[i][1] << ":" << sourcepts_upper_limit[i][1] << "]\n";
					}
				}
			}
		} else if (n_sourcepts_fit > 0) cout << "Initial source point parameters not chosen\n";
	}
	else if (source_fit_mode == Pixellated_Source) {
		if (vary_regularization_parameter) {
			if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
				cout << "Regularization parameter: " << regularization_parameter << endl;
			} else {
				if ((regularization_parameter_lower_limit==1e30) or (regularization_parameter_upper_limit==1e30)) cout << "\nRegularization parameter: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
				else cout << "Regularization parameter: [" << regularization_parameter_lower_limit << ":" << regularization_parameter << ":" << regularization_parameter_upper_limit << "]\n";
			}
		}
		if (vary_magnification_threshold) {
			if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
				cout << "Pixel magnification threshold: " << pixel_magnification_threshold << endl;
			} else {
				if ((pixel_magnification_threshold_lower_limit==1e30) or (pixel_magnification_threshold_upper_limit==1e30)) cout << "\nPixel magnification threshold: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
				else cout << "Pixel magnification threshold: [" << pixel_magnification_threshold_lower_limit << ":" << pixel_magnification_threshold << ":" << pixel_magnification_threshold_upper_limit << "]\n";
			}
		}
		if (vary_pixel_fraction) {
			if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
				cout << "Pixel magnification threshold: " << pixel_fraction << endl;
			} else {
				if ((pixel_fraction_lower_limit==1e30) or (pixel_fraction_upper_limit==1e30)) cout << "\nPixel magnification threshold: lower/upper limits not given (these must be set by 'regparam' command before fit)\n";
				else cout << "Pixel magnification threshold: [" << pixel_fraction_lower_limit << ":" << pixel_fraction << ":" << pixel_fraction_upper_limit << "]\n";
			}
		}
	}
	if (vary_hubble_parameter) {
		if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
			cout << "Hubble parameter: " << hubble << endl;
		} else {
			if ((hubble_lower_limit==1e30) or (hubble_upper_limit==1e30)) cout << "\nHubble parameter: lower/upper limits not given (these must be set by 'h0' command before fit)\n";
			else cout << "Hubble parameter: [" << hubble_lower_limit << ":" << hubble << ":" << hubble_upper_limit << "]\n";
		}
	}
}

void Lens::plot_ray_tracing_grid(double xmin, double xmax, double ymin, double ymax, int x_N, int y_N, string filename)
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
			find_sourcept(corner_pts[i][j],corner_sourcepts[i][j],0,reference_zfactor);
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

void Lens::plot_logkappa_map(const int x_N, const int y_N)
{
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = "lensmap.x";
	string y_filename = "lensmap.y";
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

	string logkapname = "lensmap.kappalog";
	ofstream logkapout(logkapname.c_str());

	double kap, mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			kap = kappa(pos,reference_zfactor);
			logkapout << log(kap)/log(10) << " ";
		}
		logkapout << endl;
	}
}

void Lens::plot_logmag_map(const int x_N, const int y_N)
{
	double x,xmin,xmax,xstep,y,ymin,ymax,ystep;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xstep = (xmax-xmin)/x_N;
	ystep = (ymax-ymin)/y_N;
	string x_filename = "lensmap.x";
	string y_filename = "lensmap.y";
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

	string logmagname = "lensmap.maglog";
	ofstream logmagout(logmagname.c_str());

	double mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			mag = magnification(pos,0,reference_zfactor);
			logmagout << log(abs(mag))/log(10) << " ";
		}
		logmagout << endl;
	}
}

void Lens::plot_lensinfo_maps(string file_root, const int x_N, const int y_N)
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

	double kap, mag, invmag, shearval, pot;
	lensvector alpha;
	lensvector pos;
	for (j=0, y=ymin+0.5*ystep; j < y_N; j++, y += ystep) {
		pos[1] = y;
		for (i=0, x=xmin+0.5*xstep; i < x_N; i++, x += xstep) {
			pos[0] = x;
			kap = kappa(pos,reference_zfactor);
			mag = magnification(pos,0,reference_zfactor);
			invmag = inverse_magnification(pos,0,reference_zfactor);
			shearval = shear(pos,0,reference_zfactor);
			//pot = lens->potential(pos);
			deflection(pos,alpha,reference_zfactor);

			kapout << kap << " ";
			magout << mag << " ";
			invmagout << invmag << " ";
			shearout << shearval << " ";
			//potout << pot << " ";
			defxout << alpha[0] << " ";
			defyout << alpha[1] << " ";
			logkapout << log(kap)/log(10) << " ";
			logmagout << log(abs(mag))/log(10) << " ";
			logshearout << log(abs(shearval))/log(10) << " ";

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
}

// Pixel grid functions

void Lens::find_optimal_sourcegrid_for_analytic_source()
{
	if (n_sb==0) { warn("no source objects have been specified"); return; }
	sb_list[0]->window_params(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax);
	if (n_sb > 1) {
		double xmin, xmax, ymin, ymax;
		for (int i=1; i < n_sb; i++) {
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

bool Lens::create_source_surface_brightness_grid(bool verbal)
{
	bool create_image_pixelgrid = false;
	if ((adaptive_grid) and (nlens==0)) { cerr << "Error: cannot ray trace source for adaptive grid; no lens model has been specified\n"; return false; }
	if ((adaptive_grid) or (((auto_sourcegrid) or (auto_srcgrid_npixels)) and (islens()))) create_image_pixelgrid = true;
	if (n_sb==0) { warn("no source objects have been specified"); return false; }

	if (create_image_pixelgrid) {
		double xmin,xmax,ymin,ymax;
		xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
		ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
		xmax += 1e-10;
		ymax += 1e-10;
		if (image_pixel_grid != NULL) delete image_pixel_grid;
		image_pixel_grid = new ImagePixelGrid(this,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y);

		int n_imgpixels;
		if (auto_sourcegrid) find_optimal_sourcegrid_for_analytic_source();
		if (auto_srcgrid_npixels) {
			if (auto_srcgrid_set_pixel_size)
				image_pixel_grid->find_optimal_firstlevel_sourcegrid_npixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_imgpixels);
			else
				image_pixel_grid->find_optimal_sourcegrid_npixels(pixel_fraction,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_imgpixels);
			if ((verbal) and (mpi_id==0)) {
				cout << "Optimal sourcegrid number of pixels: " << srcgrid_npixels_x << " " << srcgrid_npixels_y << endl;
				cout << "Sourcegrid dimensions: " << sourcegrid_xmin << " " << sourcegrid_xmax << " " << sourcegrid_ymin << " " << sourcegrid_ymax << endl;
				cout << "Number of active image pixels expected: " << n_imgpixels << endl;
			}
		}

		if ((srcgrid_npixels_x < 2) or (srcgrid_npixels_y < 2)) {
			warn("too few source pixels for ray tracing");
			delete image_pixel_grid;
			image_pixel_grid = NULL;
			return false;
		}
	}

	SourcePixelGrid::set_splitting(srcgrid_npixels_x,srcgrid_npixels_y,1e-6);
	if (source_pixel_grid != NULL) delete source_pixel_grid;
	source_pixel_grid = new SourcePixelGrid(this,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax);
	if (create_image_pixelgrid) source_pixel_grid->set_image_pixel_grid(image_pixel_grid);
	if ((mpi_id==0) and (verbal)) {
		cout << "# of source pixels: " << source_pixel_grid->number_of_pixels << endl;
	}
	if (adaptive_grid) {
		source_pixel_grid->adaptive_subgrid();
		if ((mpi_id==0) and (verbal)) {
			cout << "# of source pixels after subgridding: " << source_pixel_grid->number_of_pixels << endl;
		}
	}
	source_pixel_grid->assign_surface_brightness();
	if (create_image_pixelgrid) {
		delete image_pixel_grid;
		image_pixel_grid = NULL;
	}
	return true;
}

void Lens::load_source_surface_brightness_grid(string source_inputfile)
{
	if (source_pixel_grid != NULL) delete source_pixel_grid;
	source_pixel_grid = new SourcePixelGrid(this,source_inputfile,1e-6);
}

void Lens::load_image_surface_brightness_grid(string image_pixel_filename_root)
{
	if (image_pixel_data != NULL) delete image_pixel_data;
	image_pixel_data = new ImagePixelData();
	image_pixel_data->set_lens(this);
	if (fits_format == true) {
		if (data_pixel_size < 0) { // in this case no pixel scale has been specified, so we simply use the grid that has already been chosen
			double xmin,xmax,ymin,ymax;
			xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
			ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
			image_pixel_data->load_data_fits(xmin,xmax,ymin,ymax,image_pixel_filename_root);
		} else {
			if (image_pixel_data->load_data_fits(data_pixel_size,image_pixel_filename_root)==true) {
				double xmin,xmax,ymin,ymax;
				int npx, npy;
				image_pixel_data->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
				grid_xlength = xmax-xmin;
				grid_ylength = ymax-ymin;
				set_gridcenter(0.5*(xmin+xmax),0.5*(ymin+ymax));
			}
		}
	} else {
		image_pixel_data->load_data(image_pixel_filename_root);
		double xmin,xmax,ymin,ymax;
		int npx, npy;
		image_pixel_data->get_grid_params(xmin,xmax,ymin,ymax,npx,npy);
		grid_xlength = xmax-xmin;
		grid_ylength = ymax-ymin;
		set_gridcenter(0.5*(xmin+xmax),0.5*(ymin+ymax));
	}
	image_pixel_data->get_npixels(n_image_pixels_x,n_image_pixels_y);
	if (image_pixel_grid != NULL) {
		delete image_pixel_grid; // so when you invert, it will load a new image grid based on the data
		// This should be changed! There should be a separate image_pixel_grid for the data, vs. lensed images. That way, you don't have to do this!
		image_pixel_grid = NULL;
	}
	// Make sure the grid size & center are fixed now
	if (autocenter) autocenter = false;
	if (auto_gridsize_from_einstein_radius) auto_gridsize_from_einstein_radius = false;
	if (autogrid_before_grid_creation) autogrid_before_grid_creation = false;
}

bool Lens::plot_lensed_surface_brightness(string imagefile, bool output_fits, bool plot_residual, bool verbose)
{
	if (source_pixel_grid==NULL) { warn("No source surface brightness map has been generated"); return false; }
	if ((plot_residual==true) and (image_pixel_data==NULL)) { warn("cannot plot residual image, no pixel data image has been loaded"); return false; }
	double xmin,xmax,ymin,ymax;
	xmin = grid_xcenter-0.5*grid_xlength; xmax = grid_xcenter+0.5*grid_xlength;
	ymin = grid_ycenter-0.5*grid_ylength; ymax = grid_ycenter+0.5*grid_ylength;
	xmax += 1e-10;
	ymax += 1e-10;
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(this,ray_tracing_method,xmin,xmax,ymin,ymax,n_image_pixels_x,n_image_pixels_y);
	if (image_pixel_data != NULL) image_pixel_grid->set_fit_window((*image_pixel_data)); // testing

	image_pixel_grid->set_source_pixel_grid(source_pixel_grid);
	source_pixel_grid->set_image_pixel_grid(image_pixel_grid);
	if ((use_input_psf_matrix) or ((psf_width_x != 0) and (psf_width_y != 0))) {
		if (assign_pixel_mappings(verbose)==false) return false;
		initialize_pixel_matrices(verbose);
		//PSF_convolution_Lmatrix(verbose);
		source_pixel_grid->fill_surface_brightness_vector();
		calculate_image_pixel_surface_brightness();
		store_image_pixel_surface_brightness();
		clear_pixel_matrices();
	} else {
		image_pixel_grid->find_surface_brightness(); // no PSF, so direct ray tracing can be used
	}
	if (sim_pixel_noise != 0) {
		if (verbose) {
			double signal_to_noise = image_pixel_grid->calculate_signal_to_noise(sim_pixel_noise);
			if (mpi_id==0) cout << "Signal-to-noise ratio = " << signal_to_noise << endl;
		}
		image_pixel_grid->add_pixel_noise(sim_pixel_noise);
	}
	if (output_fits==false) {
		if (mpi_id==0) image_pixel_grid->plot_surface_brightness(imagefile,plot_residual);
	} else {
		if (mpi_id==0) image_pixel_grid->output_fits_file(imagefile,plot_residual);
	}
	delete image_pixel_grid; // so when you invert, it will load a new image grid based on the data
	image_pixel_grid = NULL;
	return true;
}

double Lens::image_pixel_chi_square()
{
	if (image_pixel_grid==NULL) { warn("No image surface brightness map has been generated"); return -1e30; }
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return -1e30; }
	int i,j;
	if (image_pixel_data->npixels_x != image_pixel_grid->x_N) die("image surface brightness map does not have same dimensions of image pixel data");
	if (image_pixel_data->npixels_y != image_pixel_grid->y_N) die("image surface brightness map does not have same dimensions of image pixel data");
	double chisq=0;
	for (i=0; i < image_pixel_data->npixels_x; i++) {
		for (j=0; j < image_pixel_data->npixels_y; j++) {
			if (image_pixel_grid->maps_to_source_pixel)
				chisq += SQR(image_pixel_grid->surface_brightness[i][j] - image_pixel_data->surface_brightness[i][j]);
		}
	}
	return chisq;
}

void Lens::load_pixel_grid_from_data()
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return; }
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(reference_zfactor, ray_tracing_method, (*image_pixel_data));
	image_pixel_grid->lens = this;
	image_pixel_grid->set_pixel_noise(data_pixel_noise);
}

double Lens::invert_surface_brightness_map_from_data(bool verbal)
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return -1e30; }
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	image_pixel_grid = new ImagePixelGrid(reference_zfactor, ray_tracing_method, (*image_pixel_data));
	image_pixel_grid->lens = this;
	image_pixel_grid->set_pixel_noise(data_pixel_noise);
	double chisq = invert_image_surface_brightness_map(verbal);
	if (chisq == 2e30) {
		delete image_pixel_grid;
		image_pixel_grid = NULL;
	}
	return chisq;
}

double Lens::invert_image_surface_brightness_map(bool verbal)
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return -1e30; }
	if (image_pixel_grid == NULL) { warn("No image surface brightness grid has been loaded"); return -1e30; }

	if (subhalo_prior) {
		double xc, yc;
		for (int i=0; i < nlens; i++) {
			if ((lens_list[i]->get_lenstype()==PJAFFE) or (lens_list[i]->get_lenstype()==CORECUSP)) {
				lens_list[i]->get_center_coords(xc,yc);
				if (!image_pixel_data->test_if_in_fit_region(xc,yc)) {
					if ((mpi_id==0) and (verbal)) cout << "Subhalo outside fit region --> chisq = 2e30, will not invert image\n";
					if (logfile.is_open()) 
						logfile << "it=" << chisq_it << " chisq0=2e30" << endl;
					return 2e30;
				}
			}
		}
	}

	if ((mpi_id==0) and (verbal)) cout << "Number of data pixels in fit window: " << image_pixel_data->n_required_pixels << endl;
	if (pixel_fraction <= 0) die("pixel fraction cannot be less than or equal to zero");
#ifdef USE_OPENMP
	double tot_wtime0, tot_wtime;
	if (show_wtime) {
		tot_wtime0 = omp_get_wtime();
	}
#endif
	image_pixel_grid->redo_lensing_calculations();

	if (auto_sourcegrid) image_pixel_grid->find_optimal_sourcegrid(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,sourcegrid_limit_xmin,sourcegrid_limit_xmax,sourcegrid_limit_ymin,sourcegrid_limit_ymax);
	int n_expected_imgpixels;
	if (auto_srcgrid_npixels) {
		if (auto_srcgrid_set_pixel_size)
			image_pixel_grid->find_optimal_firstlevel_sourcegrid_npixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_expected_imgpixels);
		else
			image_pixel_grid->find_optimal_sourcegrid_npixels(pixel_fraction,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,srcgrid_npixels_x,srcgrid_npixels_y,n_expected_imgpixels);
		if ((mpi_id==0) and (verbal)) {
			cout << "Optimal sourcegrid number of pixels: " << srcgrid_npixels_x << " " << srcgrid_npixels_y << endl;
			cout << "Sourcegrid dimensions: " << sourcegrid_xmin << " " << sourcegrid_xmax << " " << sourcegrid_ymin << " " << sourcegrid_ymax << endl;
			cout << "Number of active image pixels expected: " << n_expected_imgpixels << endl;
		}
	}
	if ((srcgrid_npixels_x < 2) or (srcgrid_npixels_y < 2)) {
		if ((mpi_id==0) and (verbal)) cout << "Source grid has negligible size...cannot invert image\n";
		if (logfile.is_open()) 
			logfile << "it=" << chisq_it << " chisq0=2e30" << endl;
		return 2e30;
	}

	SourcePixelGrid::set_splitting(srcgrid_npixels_x,srcgrid_npixels_y,1e-6);
	if (source_pixel_grid != NULL) delete source_pixel_grid;
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	source_pixel_grid = new SourcePixelGrid(this,sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax);
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for creating source pixel grid: " << wtime << endl;
	}
#endif
	image_pixel_grid->set_source_pixel_grid(source_pixel_grid);
	source_pixel_grid->set_image_pixel_grid(image_pixel_grid);

	if ((mpi_id==0) and (verbal)) {
		cout << "# of source pixels: " << source_pixel_grid->number_of_pixels;
		if (auto_srcgrid_npixels) {
			double pix_frac = ((double) source_pixel_grid->number_of_pixels) / n_expected_imgpixels;
			cout << ", f=" << pix_frac;
		}
		cout << endl;
	}
	if (adaptive_grid) {
		source_pixel_grid->adaptive_subgrid();
		if ((mpi_id==0) and (verbal)) {
			cout << "# of source pixels after subgridding: " << source_pixel_grid->number_of_pixels;
			if (auto_srcgrid_npixels) {
				double pix_frac = ((double) source_pixel_grid->number_of_pixels) / n_expected_imgpixels;
				cout << ", f=" << pix_frac;
			}
			cout << endl;
		}
	} else {
		source_pixel_grid->calculate_pixel_magnifications();
	}

	if ((mpi_id==0) and (verbal)) cout << "Assigning pixel mappings...\n";
	if (assign_pixel_mappings(verbal)==false) {
		return 2e30; // the former argument says to include umapped source pixels, latter argument tells it to redraw the source grid to exclude unmapped pixels (beyond level 1) if necessary
	}
	if ((mpi_id==0) and (verbal)) cout << "Initializing pixel matrices...\n";
	initialize_pixel_matrices(verbal);
	if (regularization_method != None) create_regularization_matrix();
	PSF_convolution_Lmatrix(verbal);
	image_pixel_grid->fill_surface_brightness_vector();

	if ((mpi_id==0) and (verbal)) cout << "Creating lensing matrices...\n" << flush;
	create_lensing_matrices_from_Lmatrix(verbal);
#ifdef USE_OPENMP
	if (show_wtime) {
		tot_wtime = omp_get_wtime() - tot_wtime0;
		if (mpi_id==0) cout << "Total wall time before F-matrix inversion: " << tot_wtime << endl;
	}
#endif

	if ((mpi_id==0) and (verbal)) cout << "Inverting lens mapping...\n" << flush;
	if (inversion_method==MUMPS) invert_lens_mapping_MUMPS(verbal);
	else if (inversion_method==UMFPACK) invert_lens_mapping_UMFPACK(verbal);
	else invert_lens_mapping_CG_method(verbal);

	double chisq;
	if ((n_image_prior) and (n_images_at_sbmax < n_image_threshold)) {
		chisq = 1e30;
		if (group_id==0) {
			if (logfile.is_open()) {
				logfile << "it=" << chisq_it << " chisq0=" << chisq << endl;
				if (vary_pixel_fraction) logfile << "F=" << ((double) source_npixels)/image_npixels << endl;
			}
		}
		if ((mpi_id==0) and (verbal)) cout << "chisq0=" << chisq << endl;
	}
	else
	{
		calculate_image_pixel_surface_brightness();

#ifdef USE_OPENMP
		if (show_wtime) {
			tot_wtime = omp_get_wtime() - tot_wtime0;
			if (mpi_id==0) cout << "Total wall time for F-matrix construction + inversion: " << tot_wtime << endl;
		}
#endif
		chisq=0;
		double chisq_signal=0;
		double covariance; // right now we're using a uniform uncorrelated noise for each pixel
		if (data_pixel_noise==0) covariance = 1; // doesn't matter what covariance is, since we won't be regularizing
		else covariance = SQR(data_pixel_noise);
		int i,j;
		double dchisq, dchisq2;
		int img_index;
		int count=0;
		int n_data_pixels=0;
		for (i=0; i < image_pixel_data->npixels_x; i++) {
			for (j=0; j < image_pixel_data->npixels_y; j++) {
				if (image_pixel_data->require_fit[i][j]) {
					n_data_pixels++;
					if (image_pixel_grid->maps_to_source_pixel[i][j]) {
						img_index = image_pixel_grid->pixel_index[i][j];
						chisq += SQR(image_surface_brightness[img_index] - image_pixel_data->surface_brightness[i][j])/covariance; // generalize to full covariance matrix later
						if (abs(image_pixel_data->surface_brightness[i][j]) > 2*data_pixel_noise)
							chisq_signal += SQR(image_surface_brightness[img_index] - image_pixel_data->surface_brightness[i][j])/covariance; // generalize to full covariance matrix later
						count++;
					} else {
						chisq += SQR(image_pixel_data->surface_brightness[i][j])/covariance;
					}
				}
			}
		}

		if (group_id==0) {
			if (logfile.is_open()) {
				logfile << "it=" << chisq_it << " chisq0=" << chisq << " ";
				if (vary_pixel_fraction) logfile << "F=" << ((double) source_npixels)/image_npixels << " ";
			}
		}
		//cout << "src_npixels=" << source_pixel_grid->number_of_pixels << endl;
		//cout << "pixel fraction = " << pixel_fraction << endl;
		//cout << "mag_threshold = " << pixel_magnification_threshold << endl;
		//cout << "reg = " << regularization_parameter << endl;
		//cout << "chisq0=" << chisq << endl;
		if ((regularization_method != None) and ((vary_regularization_parameter) or (vary_pixel_fraction))) {
			if ((mpi_id==0) and (verbal)) cout << "chisq0=" << chisq << endl;
			// NOTE: technically, you should have these terms even if you do not vary the regularization parameter, since varying
			//       the lens parameters and/or the adaptive grid changes the determinants. However, this probably will not affect
			//       the inferred lens parameters because they are not very sensitive to the regularization. Play with this later!

			double Es=0;
			for (i=0; i < source_npixels; i++) {
				Es += Rmatrix[i]*SQR(source_surface_brightness[i]);
				for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
					Es += 2 * source_surface_brightness[i] * Rmatrix[j] * source_surface_brightness[Rmatrix_index[j]]; // factor of 2 since matrix is symmetric
				}
			}
			// Es here actually differs from its usual definition by a factor of 1/2, so we do not multiply by 2 (as we would normally do for chisq = -2*log(like))
			if (regularization_parameter != 0) {
				double t0, t1, t3, t4, tsum;
				t0 = regularization_parameter*Es;
				t1 = -source_npixels*log(regularization_parameter);
				t3 = -Rmatrix_log_determinant;
				t4 = Fmatrix_log_determinant;
				tsum = t0 + t1 + t3 + t4;

				//cout << "chisq0=" << chisq << " chisq_signal=" << chisq_signal << " lambda*Es=" << t0 << " N_s*log(lambda)=" << t1 << " log_det(R)=" << t3 << " log_det(F)=" << t4 << " sum=" << tsum << endl;
				//cout << "src_npixels=" << source_npixels << endl;
				chisq -= source_npixels*log(regularization_parameter);
				chisq -= Rmatrix_log_determinant;
			}
			chisq += regularization_parameter*Es;
			chisq += Fmatrix_log_determinant;
			if (group_id==0) {
				if (logfile.is_open()) logfile << "reg=" << regularization_parameter << " chisq_reg=" << chisq << " ";
				if (logfile.is_open()) logfile << "logdet=" << Fmatrix_log_determinant << " Rlogdet=" << Rmatrix_log_determinant << " chisq_tot=" << chisq;
			}
		}
		//chisq += n_data_pixels*log(2*M_PI*data_pixel_noise); // this is not very relevant because the data fit window and assumed pixel noise are not varied

		if (max_sb_prior_unselected_pixels) {
			clear_lensing_matrices();
			clear_pixel_matrices();
			image_pixel_grid->include_all_pixels();
			assign_pixel_mappings(verbal);
			initialize_pixel_matrices(verbal);
			PSF_convolution_Lmatrix(verbal);
			source_pixel_grid->fill_surface_brightness_vector();
			calculate_image_pixel_surface_brightness();
			double max_external_sb = -1e30;
			for (i=0; i < image_pixel_data->npixels_x; i++) {
				for (j=0; j < image_pixel_data->npixels_y; j++) {
					if ((!image_pixel_data->require_fit[i][j]) and (image_pixel_grid->maps_to_source_pixel[i][j])) {
						img_index = image_pixel_grid->pixel_index[i][j];
						if (abs(image_surface_brightness[img_index]) >= abs(max_sb_frac_unselected_pixels*max_pixel_sb)) {
							if (image_surface_brightness[img_index] > max_external_sb) {
								 max_external_sb = image_surface_brightness[img_index];
							}
						}
					}
				}
			}
			if (max_external_sb > 0) {
				chisq *= pow(abs(max_external_sb/(max_sb_frac_unselected_pixels*max_pixel_sb)),40);
				if ((mpi_id==0) and (verbal)) cout << "*NOTE: surface brightness above the prior threshold (" << max_external_sb << " vs. " << max_sb_frac_unselected_pixels*max_pixel_sb << ") has been found outside the selected fit region" << endl;
			}
			image_pixel_grid->set_fit_window((*image_pixel_data));
		}

		if ((group_id==0) and (logfile.is_open())) logfile << " chisq_no_priors=" << chisq << endl;
		if ((mpi_id==0) and (verbal)) {
			cout << "chisq=" << chisq << " (without priors)" << endl;
			if ((vary_pixel_fraction) or (vary_regularization_parameter)) cout << " logdet=" << Fmatrix_log_determinant << endl;
		}
		if ((mpi_id==0) and (verbal)) cout << "number of pixels that map to source = " << count << endl;
	}

	chisq_it++;

	clear_lensing_matrices();
	clear_pixel_matrices();
	return chisq;
}

double Lens::set_required_data_pixel_window(bool verbal)
{
	if (image_pixel_data == NULL) { warn("No image surface brightness data has been loaded"); return -1e30; }
	if (image_pixel_grid != NULL) delete image_pixel_grid;
	// NOTE: we should really have a separate ImagePixelGrid object for data inversion, vs. the one we use for plotting images. That way, when you plot, it won't overwrite the
	// data image you are trying to invert. Implement this later!!!!!!!!!
	image_pixel_grid = new ImagePixelGrid(this, ray_tracing_method, (*image_pixel_data));

	int count;
	image_pixel_grid->assign_required_data_pixels(sourcegrid_xmin,sourcegrid_xmax,sourcegrid_ymin,sourcegrid_ymax,count,image_pixel_data);
	if ((mpi_id==0) and (verbal)) cout << "Number of data pixels in window required for fit: " << count << endl;
}

void Lens::generate_solution_chain_sdp81()
{
	double b_true, q_true, theta_true, xc_true, yc_true;
	double shear_x_true, shear_y_true;
	double ks_true, a_true, gamma_true, xs_true, ys_true;
	double b_fit, alpha_fit, q_fit, theta_fit, xc_fit, yc_fit;
	double shear_x_fit, shear_y_fit;
	double bs_fit, a_fit, xs_fit, ys_fit;
	double mtot_true, mtot_fit;

	bool include_subhalo = false;

	b_true = 1.606; q_true = 0.82; theta_true = 8.3; xc_true = -0.019; yc_true = -0.154;
	shear_x_true = 0.035803; shear_y_true = 0.003763;
	ks_true = 0.0089; a_true = 2.595; xs_true = -1.5; ys_true = -0.37; gamma_true = 2;
	mtot_true = M_PI*SQR(a_true)*ks_true*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
	cout << "Total subhalo mass: " << mtot_true << endl;

	double rs, r_initial, r_final, rstep, theta_sub;
	int rpoints = 15;
	r_initial = sqrt(SQR(xs_true)+SQR(ys_true));
	theta_sub = atan(ys_true/xs_true);
	r_final = 2.0;
	rstep = (r_final-r_initial)/(rpoints-1);

	double chisq, rmax, menc;
	warnings = false;
	string temp_data_filename = "tempdata.fits";
	string rmstring = "if [ -f " + temp_data_filename + " ]; then rm " + temp_data_filename + "; fi";
	fits_format = true;
	set_grid_corners(-3,3,-3,3);
	n_image_pixels_x = 500;
	n_image_pixels_y = 500;
	data_pixel_size = 0.012; // 6 arcseconds / 500 pixels long
	data_pixel_noise = 0.3;
	sim_pixel_noise = 0.3;
	psf_width_x = 0.01;
	psf_width_y = 0.01;
	clear_source_objects();
	add_source_object(GAUSSIAN,1.5,0.018,0,0.9,20,0.18,-0.118);
	add_source_object(GAUSSIAN,2,0.021,0,0.7,120,0.18,-0.208);
	sourcegrid_limit_xmin = 0.05; sourcegrid_limit_xmax = 0.3;
	sourcegrid_limit_ymin = -0.33; sourcegrid_limit_ymax = 0.02;
	regularization_method = Curvature;
	regularization_parameter = 31.5085;
	vary_regularization_parameter = true;
	pixel_magnification_threshold = 5;
	Shear::use_shear_component_params = true;
	source_fit_mode = Pixellated_Source;
	adaptive_grid = true;

	int i;

	for (int l=1; l < 2; l++) {
		if (l==1) include_subhalo = true;

		if (!include_subhalo) {
			b_fit = 1.61241; alpha_fit = 1.13264; q_fit = 0.824023; theta_fit = 8.22863; xc_fit = -0.028069; yc_fit = -0.149037;
			shear_x_fit = 0.0503836; shear_y_fit = 0.00956111;
		} else {
			b_fit = 1.607432701; alpha_fit = 1.38045557; q_fit = 0.7201057458; theta_fit = 7.769220326; xc_fit = -0.04439526632; yc_fit = -0.1574250004;
			shear_x_fit = 0.06842352955; shear_y_fit = 0.01062479437;
			bs_fit = 0.04560161822; xs_fit = -1.500479158; ys_fit = -0.3724162998;
		}

		string filename = "rmax_fit_xs";
		if (!include_subhalo) filename += "_nosub";
		else filename += "_sub";
		ofstream fitout((filename + ".dat").c_str());
		ofstream params_out((filename + "_params.dat").c_str());
		for (i=0, rs=r_initial; i < rpoints; i++, rs += rstep)
		{
			clear_lenses();
			auto_srcgrid_npixels = false;
			auto_sourcegrid = false;
			sourcegrid_xmin=-0.4;
			sourcegrid_xmax=0.3;
			sourcegrid_ymin=-0.6;
			sourcegrid_ymax=0.1999;
			srcgrid_npixels_x = 150;
			srcgrid_npixels_y = 150;

			xs_true = -rs*cos(theta_sub);
			ys_true = -rs*sin(theta_sub);
			add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
			add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
			add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
			//calculate_critical_curve_deformation_radius_numerical(2,true,rmax,menc);
			//cout << endl;
			create_source_surface_brightness_grid(false);
			system(rmstring.c_str());
			plot_lensed_surface_brightness(temp_data_filename,true,false);
			load_image_surface_brightness_grid(temp_data_filename);
			image_pixel_data->set_no_required_data_pixels();
			image_pixel_data->set_required_data_annulus(-0.4,-0.2,0.97,1.43,90,270);
			image_pixel_data->set_required_data_pixels(1.75,2.05,-0.5,0.33);
			clear_lenses();

			auto_srcgrid_npixels = true;
			auto_sourcegrid = true;
			add_lens(ALPHA,b_fit,alpha_fit,0,q_fit,theta_fit,xc_fit,yc_fit);
			add_shear_lens(shear_x_fit,shear_y_fit,xc_fit,yc_fit);
			lens_list[1]->anchor_to_lens(lens_list,0);
			boolvector vary_flags(7);
			for (int i=0; i < 7; i++) vary_flags[i] = true;
			vary_flags[2] = false; // not varying core
			lens_list[0]->vary_parameters(vary_flags);
			boolvector shear_vary_flags(2);
			shear_vary_flags[0] = true;
			shear_vary_flags[1] = true;
			lens_list[1]->vary_parameters(shear_vary_flags);
			if (include_subhalo) {
				add_lens(PJAFFE,bs_fit,0,0,1,0,xs_fit,ys_fit);
				lens_list[2]->assign_anchored_parameters(lens_list[0]); // calculates tidal radius
				boolvector pjaffe_vary_flags(7);
				for (int i=0; i < 7; i++) pjaffe_vary_flags[i] = false;
				pjaffe_vary_flags[0] = true;
				pjaffe_vary_flags[5] = true;
				pjaffe_vary_flags[6] = true;
				lens_list[2]->vary_parameters(pjaffe_vary_flags);
			}
			//print_lens_list(true);
			max_sb_prior_unselected_pixels = true;
			chisq = chi_square_fit_simplex();
			use_bestfit_model();

			if (include_subhalo) {
				calculate_critical_curve_deformation_radius_numerical(2,false,rmax,menc);
			}

			dvector host_params;
			double dum;
			lens_list[0]->get_extra_params(host_params);
			b_fit = host_params[0];
			alpha_fit = host_params[1];
			lens_list[0]->get_q_theta(q_fit,theta_fit);
			theta_fit *= 180.0/M_PI;
			lens_list[0]->get_center_coords(xc_fit,yc_fit);
			double shear_ext,phi_p;
			lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
			shear_x_fit = -shear_ext*cos(2*phi_p+M_PI);
			shear_y_fit = -shear_ext*sin(2*phi_p+M_PI);
			if (include_subhalo) {
				dvector sub_params;
				lens_list[2]->get_extra_params(sub_params);
				bs_fit = sub_params[0];
				a_fit = sub_params[1];
				lens_list[2]->get_center_coords(xs_fit,ys_fit);
			}
			//print_lens_list(false);
			cout << endl;

			double avg_kappa, menc_true;
			if (include_subhalo) {
				clear_lenses();
				add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
				add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
				add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
				avg_kappa = reference_zfactor*lens_list[2]->kappa_avg_r(rmax);
				menc_true = avg_kappa*M_PI*SQR(rmax)*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				mtot_fit = M_PI*a_fit*bs_fit*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				cout << "rmax = " << rmax << ", mass_enclosed_fit = " << menc << ", mass_enclosed_true = " << menc_true << endl;
				fitout << xs_true << " " << ys_true << " " << xs_fit << " " << ys_fit << " " << chisq << " " << rmax << " " << menc << " " << menc_true << " " << (menc_true-menc)/menc_true << " " << mtot_fit << " " << (mtot_true-mtot_fit)/mtot_true << endl;
			} else {
				fitout << xs_true << " " << rs << " " << chisq << endl;
			}
			params_out << "xs_true=" << xs_true << " ks_true=" << ks_true << " ";
			if (include_subhalo)
				params_out << bs_fit << " " << xs_fit << " " << ys_fit << " ";
			params_out << b_fit << " " << alpha_fit << " " << q_fit << " " << theta_fit << " " << xc_fit << " " << yc_fit << " " << shear_x_fit << " " << shear_y_fit << " " << endl;
		}

		// Now reset and move subhalo in the other direction...
		if (!include_subhalo) {
			b_fit = 1.61241; alpha_fit = 1.13264; q_fit = 0.824023; theta_fit = 8.22863; xc_fit = -0.028069; yc_fit = -0.149037;
			shear_x_fit = 0.0503836; shear_y_fit = 0.00956111;
		} else {
			b_fit = 1.607432701; alpha_fit = 1.38045557; q_fit = 0.7201057458; theta_fit = 7.769220326; xc_fit = -0.04439526632; yc_fit = -0.1574250004;
			shear_x_fit = 0.06842352955; shear_y_fit = 0.01062479437;
			bs_fit = 0.04560161822; xs_fit = -1.500479158; ys_fit = -0.3724162998;
		}

		for (i=0, rs=r_initial; i < rpoints; i++, rs -= rstep)
		{
			clear_lenses();
			auto_srcgrid_npixels = false;
			auto_sourcegrid = false;
			sourcegrid_xmin=-0.4;
			sourcegrid_xmax=0.3;
			sourcegrid_ymin=-0.6;
			sourcegrid_ymax=0.1999;
			srcgrid_npixels_x = 150;
			srcgrid_npixels_y = 150;

			xs_true = -rs*cos(theta_sub);
			ys_true = -rs*sin(theta_sub);
			add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
			add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
			add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
			//calculate_critical_curve_deformation_radius_numerical(2,true,rmax,menc);
			//cout << endl;
			create_source_surface_brightness_grid(false);
			system(rmstring.c_str());
			plot_lensed_surface_brightness(temp_data_filename,true,false);
			load_image_surface_brightness_grid(temp_data_filename);
			image_pixel_data->set_no_required_data_pixels();
			image_pixel_data->set_required_data_annulus(-0.4,-0.2,0.97,1.43,90,270);
			image_pixel_data->set_required_data_pixels(1.75,2.05,-0.5,0.33);
			clear_lenses();

			auto_srcgrid_npixels = true;
			auto_sourcegrid = true;
			add_lens(ALPHA,b_fit,alpha_fit,0,q_fit,theta_fit,xc_fit,yc_fit);
			add_shear_lens(shear_x_fit,shear_y_fit,xc_fit,yc_fit);
			lens_list[1]->anchor_to_lens(lens_list,0);
			boolvector vary_flags(7);
			for (int i=0; i < 7; i++) vary_flags[i] = true;
			vary_flags[2] = false; // not varying core
			lens_list[0]->vary_parameters(vary_flags);
			boolvector shear_vary_flags(2);
			shear_vary_flags[0] = true;
			shear_vary_flags[1] = true;
			lens_list[1]->vary_parameters(shear_vary_flags);
			if (include_subhalo) {
				add_lens(PJAFFE,bs_fit,0,0,1,0,xs_fit,ys_fit);
				lens_list[2]->assign_anchored_parameters(lens_list[0]); // calculates tidal radius
				boolvector pjaffe_vary_flags(7);
				for (int i=0; i < 7; i++) pjaffe_vary_flags[i] = false;
				pjaffe_vary_flags[0] = true;
				pjaffe_vary_flags[5] = true;
				pjaffe_vary_flags[6] = true;
				lens_list[2]->vary_parameters(pjaffe_vary_flags);
			}
			max_sb_prior_unselected_pixels = true;
			chisq = chi_square_fit_simplex();
			use_bestfit_model();

			if (include_subhalo) {
				calculate_critical_curve_deformation_radius_numerical(2,false,rmax,menc);
			}

			dvector host_params;
			double dum;
			lens_list[0]->get_extra_params(host_params);
			b_fit = host_params[0];
			alpha_fit = host_params[1];
			lens_list[0]->get_q_theta(q_fit,theta_fit);
			theta_fit *= 180.0/M_PI;
			lens_list[0]->get_center_coords(xc_fit,yc_fit);
			double shear_ext,phi_p;
			lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
			shear_x_fit = -shear_ext*cos(2*phi_p+M_PI);
			shear_y_fit = -shear_ext*sin(2*phi_p+M_PI);
			dvector sub_params;
			lens_list[2]->get_extra_params(sub_params);
			bs_fit = sub_params[0];
			a_fit = sub_params[1];
			lens_list[2]->get_center_coords(xs_fit,ys_fit);
			//print_lens_list(false);
			invert_surface_brightness_map_from_data(false);
			plotcrit("crit.dat");
			if (plot_lensed_surface_brightness("img_pixel")==true) {
				if (mpi_id==0) source_pixel_grid->plot_surface_brightness("src_pixel");
				run_plotter_range("srcpixel","");
				run_plotter_range("imgpixel","");
			}
			cout << endl;

			double avg_kappa, menc_true;
			if (include_subhalo) {
				clear_lenses();
				add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
				add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
				add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
				avg_kappa = reference_zfactor*lens_list[2]->kappa_avg_r(rmax);
				menc_true = avg_kappa*M_PI*SQR(rmax)*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				mtot_fit = M_PI*a_fit*bs_fit*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				cout << "rmax = " << rmax << ", mass_enclosed_fit = " << menc << ", mass_enclosed_true = " << menc_true << endl;
				fitout << xs_true << " " << ys_true << " " << xs_fit << " " << ys_fit << " " << chisq << " " << rmax << " " << menc << " " << menc_true << " " << (menc_true-menc)/menc_true << " " << mtot_fit << " " << (mtot_true-mtot_fit)/mtot_true << endl;
			} else {
				fitout << xs_true << " " << rs << " " << chisq << endl;
			}
			params_out << "xs_true=" << xs_true << " ks_true=" << ks_true << " ";
			if (include_subhalo)
				params_out << bs_fit << " " << xs_fit << " " << ys_fit << " ";
			params_out << b_fit << " " << alpha_fit << " " << q_fit << " " << theta_fit << " " << xc_fit << " " << yc_fit << " " << shear_x_fit << " " << shear_y_fit << " " << endl;

		}
		fitout.close();
		params_out.close();
	}

	/*
	// Now re-set and vary ks instead...
	if (!include_subhalo) {
		b_fit = b_true; alpha_fit = 1.0; q_fit = q_true; theta_fit = theta_true; xc_fit = xc_true; yc_fit = yc_true;
		shear_x_fit = shear_x_true; shear_y_fit = shear_y_true;
	} else {
		//b_fit = 1.607432701; alpha_fit = 1.38045557; q_fit = 0.7201057458; theta_fit = 7.769220326; xc_fit = -0.04439526632; yc_fit = -0.1574250004;
		//shear_x_fit = 0.06842352955; shear_y_fit = 0.01062479437;
		//bs_fit = 0.04560161822; xs_fit = -1.500479158; ys_fit = -0.3724162998;
		b_fit = b_true; alpha_fit = 1.0; q_fit = q_true; theta_fit = theta_true; xc_fit = xc_true; yc_fit = yc_true;
		shear_x_fit = shear_x_true; shear_y_fit = shear_y_true;
		bs_fit = 0.001; xs_fit = -1.5; ys_fit = -0.37;
	}

	double ks, ks_initial, ks_final, kstep;
	int kspoints = 30;
	ks_initial = 0.001;
	ks_final = 0.015;
	kstep = (ks_final-ks_initial)/(kspoints-1);

	for (int l=1; l < 2; l++) {
		if (l==1) include_subhalo = true;

		string kfilename = "rmax_fit_ks";
		if (!include_subhalo) kfilename += "_nosub";
		else kfilename += "_sub";
		ofstream kfitout((kfilename + ".dat").c_str());
		ofstream kparams_out((kfilename + "_params.dat").c_str());

		for (i=0, ks=ks_initial; i < kspoints; i++, ks += kstep)
		{
			ks_true = ks;
			clear_lenses();
			auto_srcgrid_npixels = false;
			auto_sourcegrid = false;
			sourcegrid_xmin=-0.4;
			sourcegrid_xmax=0.3;
			sourcegrid_ymin=-0.6;
			sourcegrid_ymax=0.1999;
			srcgrid_npixels_x = 150;
			srcgrid_npixels_y = 150;

			xs_true = -r_initial*cos(theta_sub);
			ys_true = -r_initial*sin(theta_sub);
			a_true = SQR(21.6347)*ks_true/1.606; // from formula for tidal radius of Munoz profile, see Minor et al. (2016)
			add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
			add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
			add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
			//calculate_critical_curve_deformation_radius_numerical(2,true,rmax,menc);
			//cout << endl;
			create_source_surface_brightness_grid(false);
			system(rmstring.c_str());
			plot_lensed_surface_brightness(temp_data_filename,true,false);
			load_image_surface_brightness_grid(temp_data_filename);
			image_pixel_data->set_no_required_data_pixels();
			image_pixel_data->set_required_data_annulus(-0.4,-0.2,0.97,1.43,90,270);
			image_pixel_data->set_required_data_pixels(1.75,2.05,-0.5,0.33);
			clear_lenses();

			auto_srcgrid_npixels = true;
			auto_sourcegrid = true;
			add_lens(ALPHA,b_fit,alpha_fit,0,q_fit,theta_fit,xc_fit,yc_fit);
			add_shear_lens(shear_x_fit,shear_y_fit,xc_fit,yc_fit);
			lens_list[1]->anchor_to_lens(lens_list,0);
			boolvector vary_flags(7);
			for (int i=0; i < 7; i++) vary_flags[i] = true;
			vary_flags[2] = false; // not varying core
			lens_list[0]->vary_parameters(vary_flags);
			boolvector shear_vary_flags(2);
			shear_vary_flags[0] = true;
			shear_vary_flags[1] = true;
			lens_list[1]->vary_parameters(shear_vary_flags);
			if (include_subhalo) {
				add_lens(PJAFFE,bs_fit,0,0,1,0,xs_fit,ys_fit);
				lens_list[2]->assign_anchored_parameters(lens_list[0]); // calculates tidal radius
				boolvector pjaffe_vary_flags(7);
				for (int i=0; i < 7; i++) pjaffe_vary_flags[i] = false;
				pjaffe_vary_flags[0] = true;
				pjaffe_vary_flags[5] = true;
				pjaffe_vary_flags[6] = true;
				lens_list[2]->vary_parameters(pjaffe_vary_flags);
			}
			max_sb_prior_unselected_pixels = true;
			chisq = chi_square_fit_simplex();
			use_bestfit_model();

			if (include_subhalo) {
				calculate_critical_curve_deformation_radius_numerical(2,false,rmax,menc);
			}

			dvector host_params;
			double dum;
			lens_list[0]->get_extra_params(host_params);
			b_fit = host_params[0];
			alpha_fit = host_params[1];
			lens_list[0]->get_q_theta(q_fit,theta_fit);
			theta_fit *= 180.0/M_PI;
			lens_list[0]->get_center_coords(xc_fit,yc_fit);
			double shear_ext,phi_p;
			lens_list[1]->get_q_theta(shear_ext,phi_p); // assumes the host galaxy is lens 0, external shear is lens 1
			shear_x_fit = -shear_ext*cos(2*phi_p+M_PI);
			shear_y_fit = -shear_ext*sin(2*phi_p+M_PI);
			if (include_subhalo) {
				dvector sub_params;
				lens_list[2]->get_extra_params(sub_params);
				bs_fit = sub_params[0];
				a_fit = sub_params[1];
				lens_list[2]->get_center_coords(xs_fit,ys_fit);
			}
			print_lens_list(false);
			cout << endl;
			invert_surface_brightness_map_from_data(false);
			plotcrit("crit.dat");
			if (plot_lensed_surface_brightness("img_pixel")==true) {
				if (mpi_id==0) source_pixel_grid->plot_surface_brightness("src_pixel");
				run_plotter_range("srcpixel","");
				run_plotter_range("imgpixel","");
			}

			double avg_kappa, menc_true;
			if (include_subhalo) {
				clear_lenses();
				add_lens(ALPHA,b_true,1,0,q_true,theta_true,xc_true,yc_true);
				add_shear_lens(shear_x_true, shear_y_true, xc_true, yc_true);
				add_lens(CORECUSP,ks_true,a_true,0,1,0,xs_true,ys_true,gamma_true,4);
				avg_kappa = reference_zfactor*lens_list[2]->kappa_avg_r(rmax);
				menc_true = avg_kappa*M_PI*SQR(rmax)*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				mtot_fit = M_PI*a_fit*bs_fit*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				mtot_true = M_PI*SQR(a_true)*ks_true*4.59888e10; // the last number is sigma_cr for the lens/source redshifts of SDP.81
				cout << "rmax = " << rmax << ", mass_enclosed_fit = " << menc << ", mass_enclosed_true = " << menc_true << endl;
				kfitout << ks_true << " " << xs_fit << " " << bs_fit << " " << chisq << " " << rmax << " " << menc << " " << menc_true << " " << (menc_true-menc)/menc_true << " " << mtot_fit << " " << (mtot_true-mtot_fit)/mtot_true << endl;
			} else {
				kfitout << ks_true << " " << chisq << " " << endl;
			}
			kparams_out << "xs_true=" << xs_true << " ks_true=" << ks_true << " ";
			if (include_subhalo)
				kparams_out << bs_fit << " " << xs_fit << " " << ys_fit << " ";
			kparams_out << b_fit << " " << alpha_fit << " " << q_fit << " " << theta_fit << " " << xc_fit << " " << yc_fit << " " << shear_x_fit << " " << shear_y_fit << endl;

		}
		kfitout.close();
		kparams_out.close();
	}
	*/

}

void Lens::create_output_directory()
{
	if (mpi_id==0) {
		struct stat sb;
		stat(fit_output_dir.c_str(),&sb);
		if (S_ISDIR(sb.st_mode)==false)
			mkdir(fit_output_dir.c_str(),S_IRWXU | S_IRWXG);
	}
}

void Lens::reset()
{
	reset_grid();
	cc_rmin = default_autogrid_rmin;
	cc_rmax = default_autogrid_rmax;
}

void Lens::reset_grid()
{
	if (defspline != NULL) {
		delete defspline;
		defspline = NULL;
	}
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
	if (cc_splined==true) {
		delete[] ccspline;
		delete[] caustic;
		cc_splined = false;
	}
}

void Lens::delete_ccspline()
{
	if (cc_splined==true) {
		delete[] ccspline;
		delete[] caustic;
		cc_splined = false;
	}
}

Lens::~Lens()
{
	int i;
	if (nlens > 0) {
		for (i=0; i < nlens; i++) {
			delete lens_list[i];
		}
		delete[] lens_list;
	}

	if (n_sb > 0) {
		for (int i=0; i < n_sb; i++)
			delete sb_list[i];
		delete[] sb_list;
		n_sb = 0;
	}

	delete grid;
	delete param_settings;
	if (defspline != NULL) delete defspline;
	if (fitmodel != NULL) delete fitmodel;
	if (sourcepts_fit != NULL) delete[] sourcepts_fit;
	if (vary_sourcepts_x != NULL) delete[] vary_sourcepts_x;
	if (vary_sourcepts_y != NULL) delete[] vary_sourcepts_y;
	if (psf_matrix != NULL) {
		for (int i=0; i < psf_npixels_x; i++) delete[] psf_matrix[i];
		delete[] psf_matrix;
	}
	if (source_redshifts != NULL) delete[] source_redshifts;
	if (zfactors != NULL) delete[] zfactors;
	if (sourcepts_upper_limit != NULL) delete[] sourcepts_upper_limit;
	if (sourcepts_lower_limit != NULL) delete[] sourcepts_lower_limit;
	if ((image_data != NULL) and (borrowed_image_data==false)) delete[] image_data;
	if ((image_pixel_data != NULL) and (borrowed_image_data==false)) delete image_pixel_data;
	if (image_surface_brightness != NULL) delete[] image_surface_brightness;
	if (source_surface_brightness != NULL) delete[] source_surface_brightness;
	if (source_pixel_n_images != NULL) delete[] source_pixel_n_images;
	if (active_image_pixel_i != NULL) delete[] active_image_pixel_i;
	if (active_image_pixel_j != NULL) delete[] active_image_pixel_j;
	if (image_pixel_location_Lmatrix != NULL) delete[] image_pixel_location_Lmatrix;
	if (source_pixel_location_Lmatrix != NULL) delete[] source_pixel_location_Lmatrix;
	if (Lmatrix_index != NULL) delete[] Lmatrix_index;
	if (Lmatrix != NULL) delete[] Lmatrix;
	if (Dvector != NULL) delete[] Dvector;
	if (Fmatrix != NULL) delete[] Fmatrix;
	if (Fmatrix_index != NULL) delete[] Fmatrix_index;
	if (Rmatrix != NULL) delete[] Rmatrix;
	if (Rmatrix_index != NULL) delete[] Rmatrix_index;
	if (source_pixel_grid != NULL) delete source_pixel_grid;
	if (image_pixel_grid != NULL) delete image_pixel_grid;
}

