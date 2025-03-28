#ifndef QLENS_H
#define QLENS_H

#include "sort.h"
#include "rand.h"
#include "brent.h"
#include "spline.h"
#include "profile.h"
#include "sbprofile.h"
#include "lensvec.h"
#include "vector.h"
#include "powell.h"
#include "simplex.h"
#include "mcmchdr.h"
#include "cosmo.h"
#ifdef USE_MUMPS
#include "dmumps_c.h"
#endif
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <complex>
#define USE_COMM_WORLD -987654


#ifdef USE_FFTW
#ifdef USE_MKL
#include "fftw/fftw3.h"
#else
#include "fftw3.h"
#endif
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include "mpi.h"
#endif

using namespace std;

enum ImageSystemType { NoImages, Single, Double, Cusp, Quad };
enum inside_cell { Inside, Outside, Edge };
enum edge_sourcept_status { SourceInGap, SourceInOverlap, NoSource };
enum SourceFitMode { Point_Source, Cartesian_Source, Delaunay_Source, Parameterized_Source, Shapelet_Source };
enum Prior { UNIFORM_PRIOR, LOG_PRIOR, GAUSS_PRIOR, GAUSS2_PRIOR, GAUSS2_PRIOR_SECONDARY };
enum Transform { NONE, LOG_TRANSFORM, GAUSS_TRANSFORM, LINEAR_TRANSFORM, RATIO };
enum RayTracingMethod {
	Interpolate,
	Area_Overlap
};
enum DerivedParamType {
	KappaR,
	LambdaR,
	DKappaR,
	Mass2dR,
	Mass3dR,
	Einstein,
	Einstein_Mass,
	Kappa_Re,
	LensParam,
	AvgLogSlope,
	Perturbation_Radius,
	Relative_Perturbation_Radius,
	Robust_Perturbation_Mass,
	Robust_Perturbation_Density,
	Chi_Square,
	UserDefined
};


class QLens;			// Defined after class Grid
class SourcePixelGrid;
class DelaunayGrid;
class ImagePixelGrid;
class Defspline;	// ...
struct ImageData;
struct WeakLensingData;
struct ImagePixelData;
struct ParamSettings;
struct DerivedParam;

struct image {
	lensvector pos;
	double mag, flux, td;
	int parity;
};

struct image_data : public image
{
	double flux;
	double sigma_pos;
	double sigma_flux;
	double sigma_td;
};

// Used for the Python wrapper
struct ImageSet {
	lensvector src;
	double zsrc, srcflux;
	int n_images;
	vector<image> images;

	ImageSet() { }
	ImageSet(lensvector& src_in, double zsrc_in, image* images_in, const int nimg, const double srcflux_in = 1.0) {
		copy_imageset(src_in, zsrc_in, images_in, nimg, srcflux_in);
	}
	void copy_imageset(lensvector& src_in, double zsrc_in, image* images_in, const int nimg, const double srcflux_in = 1.0) {
		n_images = nimg;
		zsrc = zsrc_in;
		srcflux = srcflux_in;
		src[0] = src_in[0];
		src[1] = src_in[1];
		//if (images != NULL) delete[] images;
		images.clear();
		images.resize(n_images);
		for (int i=0; i < n_images; i++) {
			images[i].pos = images_in[i].pos;
			images[i].mag = images_in[i].mag;
			images[i].td = images_in[i].td;
			images[i].parity = images_in[i].parity;
		}
	}
	double imgflux(const int imgnum) { if (imgnum < n_images) return abs(images[imgnum].mag*srcflux); else return -1; }
	void print(bool include_time_delays = false, bool show_labels = true) { print_to_file(include_time_delays,show_labels,NULL,NULL); }
	void print_to_file(bool include_time_delays, bool show_labels, ofstream* srcfile, ofstream* imgfile) {
		cout << "#src_x (arcsec)\tsrc_y (arcsec)\tn_images";
		if (srcflux != -1) cout << "\tsrc_flux";
		cout << endl;
		cout << src[0] << "\t" << src[1] << "\t" << n_images << "\t";
		if (srcflux != -1) cout << "\t" << srcflux;
		cout << endl << endl;

		if (srcfile != NULL) (*srcfile) << src[0] << " " << src[1] << endl;
		//cout << "# " << n_images << " images" << endl;
		if (show_labels) {
			cout << "#pos_x (arcsec)\tpos_y (arcsec)\tmagnification";
			if (srcflux != -1.0) cout << "\tflux\t";
			if (include_time_delays) cout << "\ttime_delay (days)";
			cout << endl;
		}
		if (include_time_delays) {
			for (int i = 0; i < n_images; i++) {
				if (srcflux == -1.0) cout << images[i].pos[0] << "\t" << images[i].pos[1] << "\t" << images[i].mag << "\t" << images[i].td << endl;
				else cout << images[i].pos[0] << "\t" << images[i].pos[1] << "\t" << images[i].mag << "\t" << images[i].mag*srcflux << "\t" << images[i].td << endl;
				if (imgfile != NULL) (*imgfile) << images[i].pos[0] << " " << images[i].pos[1] << endl;
			}
		} else {
			for (int i = 0; i < n_images; i++) {
				if (srcflux == -1.0) cout << images[i].pos[0] << "\t" << images[i].pos[1] << "\t" << images[i].mag << endl;
				else cout << images[i].pos[0] << "\t" << images[i].pos[1] << "\t" << images[i].mag << "\t" << images[i].mag*srcflux << endl;
				if (imgfile != NULL) (*imgfile) << images[i].pos[0] << " " << images[i].pos[1] << endl;
			}
		}

		cout << endl;
	}
	void reset() {
		//if (n_images != 0) delete[] images;
		//images = NULL;
		n_images = 0;
	}
	//~ImageSet() {
		//if (n_images != 0) delete[] images;
	//}
};

struct ImageDataSet {
	double zsrc;
	int n_images;
	vector<image_data> images;

	void set_n_images(const int nimg) {
		n_images = nimg;
		images.clear();
		images.resize(n_images);
	}

};


struct jl_pair {
	int j,l;
};

class Grid : public Brent
{
	private:
	// this constructor is only used by the top-level Grid to initialize the lower-level grids, so it's private
	Grid(lensvector** xij, const int& i, const int& j, const int& level_in, Grid* parent_ptr);

	Grid*** cell;
	static QLens* lens;
	static int nthreads;
	Grid* neighbor[4]; // 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	Grid* parent_cell;

	static double* grid_zfactors; // kappa ratio used for modeling source points at different redshifts
	static double** grid_betafactors; // kappa ratio used for modeling source points at different redshifts
	static const int u_split, w_split;
	static bool radial_grid; // if false, a Cartesian grid is assumed
	static bool enforce_min_area;
	static bool cc_neighbor_splittings;
	static double rmin, rmax;
	static double xcenter, ycenter;
	static double grid_q;
	static double theta_offset;

	int u_N, w_N;
	int level;
	lensvector center_imgplane;
	double cell_area;
	lensvector corner_pt[4];

	// cell lensing properties
	lensvector *corner_sourcept[4];
	double *corner_invmag[4];
	double *corner_kappa[4];
	bool allocated_corner[4];

	// all functions in class Grid are contained in imgsrch.cpp
	bool image_test(const int& thread);
	bool run_newton(const lensvector& xroot_initial, const int& thread);
	inside_cell test_if_inside_sourceplane_cell(lensvector* point, const int& thread);
	bool test_if_sourcept_inside_triangle(lensvector* point1, lensvector* point2, lensvector* point3, const int& thread);
	bool test_if_inside_cell(const lensvector& point, const int& thread);
	bool test_if_galaxy_nearby(const lensvector& point, const double& distsq);

	void assign_lensing_properties(const int& thread);
	void assign_subcell_lensing_properties_firstlevel();
	void reassign_subcell_lensing_properties_firstlevel();
	void assign_subcell_lensing_properties(const int& thread);

	// Used for image searching. If you ever multi-thread the image search, be careful about making these static variables
	static lensvector *d1, *d2, *d3, *d4;
	static double *product1, *product2, *product3;
	static int *maxlevs;
	static lensvector ***xvals_threads;

	// Used for finding critical curves within a grid cell
	static int corner_positive_mag[4], corner_negative_mag[4];
	static lensvector ccsearch_initial_pt, ccsearch_interval;

	bool cc_inside;
	bool singular_pt_inside;
	bool cell_in_central_image_region;
	void check_if_cc_inside();
	void check_if_singular_point_inside(const int& thread);
	void check_if_central_image_region();

	static double ccroot_t;
	static lensvector ccroot;
	static double cclength1, cclength2, long_diagonal_length;
	double invmag_along_diagonal(const double t);

	static int u_split_initial, w_split_initial;
	static const int max_level, max_images;

	static int levels; // keeps track of the total number of grid cell levels
	static int splitlevels; // specifies the number of initial splittings to perform (not counting extra splittings if critical curves present)
	static int cc_splitlevels; // specifies the additional splittings to perform if critical curves are present
	int galsubgrid_cc_splitlevels;
	static double min_cell_area;

	void clear_subcells(int clear_level);
	void split_subcells_firstlevel(int cc_splitlevels, bool cc_neighbor_splitting);
	void split_subcells(int cc_splitlevels, bool cc_neighbor_splitting, const int& thread);
	void assign_neighbors_lensing_subcells(int cc_splitlevel, const int& thread);
	bool split_cells(const int& thread);
	void grid_search(const int& searchlevel, const int& thread);
	void grid_search_firstlevel(const int& searchlevel);
	edge_sourcept_status check_subgrid_neighbor_boundaries(const int& neighbor_direction, Grid* neighbor_subcell, lensvector& centerpt, const int& thread);
	void set_grid_xvals(lensvector** xv, const int& i, const int& j);
	void find_cell_area(const int& thread);
	void assign_firstlevel_neighbors();
	void assign_neighborhood();
	void assign_all_neighbors();
	void assign_level_neighbors(int neighbor_level);

	bool LineSearch(lensvector& xold, double fold, lensvector& g, lensvector& p, lensvector& x, double& f, double stpmax, bool &check, const int& thread);
	bool NewtonsMethod(lensvector& x, bool &check, const int& thread);
	void SolveLinearEqs(lensmatrix&, lensvector&);
	bool redundancy(const lensvector&, double &);
	double max_component(const lensvector&);

	static const int max_iterations, max_step_length;
	static lensvector *fvec;
	static bool *newton_check;

	// make these multithread-safe if you decide to multithread the image searching
	static bool finished_search;
	static int nfound_max, nfound_pos, nfound_neg;
	static image images[];

public:
	Grid(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double* zfactor_in, double** betafactor_in); 
	Grid(double xcenter_in, double ycenter_in, double xlength, double ylength, double* zfactor_in, double** betafactor_in);
	void redraw_grid(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double* zfactor_in, double** betafactor_in);
	void redraw_grid(double xcenter_in, double ycenter_in, double xlength, double ylength, double* zfactor_in, double** betafactor_in);
	void reassign_coordinates(lensvector** xij, const int& i, const int& j, const int& level_in, Grid* parent_ptr);

	static void set_splitting(int rs0, int ts0, int sl, int ccsl, double max_cs, bool neighbor_split);
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	static void reset_search_parameters();
	~Grid();

	static int nfound;
	static double image_pos_accuracy;
	image* tree_search();
	static void set_lens(QLens* lensptr) { lens = lensptr; }
	void subgrid_around_galaxies(lensvector* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_splittings, bool* subgrid);
	void subgrid_around_galaxies_iteration(lensvector* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_split, bool cc_neighbor_splitting, bool *subgrid);

	void galsubgrid();
	void store_critical_curve_pts();
	void find_and_store_critical_curve_pt(const int icorner, const int fcorner, int &added_pts);
	static void set_imagepos_accuracy(const double& setting) {
		image_pos_accuracy = setting;
	}
	static void set_enforce_min_area(const bool& setting) { enforce_min_area = setting; }

	// for plotting the grid to a file:
	static ofstream xgrid;
	void plot_corner_coordinates();
	void get_usplit_initial(int &setting) { setting = u_split_initial; }
	void get_wsplit_initial(int &setting) { setting = w_split_initial; }
};

struct WeakLensingData
{
	int n_sources;
	string *id;
	lensvector *pos;
	double *reduced_shear1;
	double *reduced_shear2;
	double *sigma_shear1, *sigma_shear2;
	double *zsrc;
	WeakLensingData() { n_sources = 0; }
	void input(const int &nn);
	void input(const WeakLensingData& wl_in);
	void add_source(const string id_in, lensvector& pos_in, const double g1_in, const double g2_in, const double g1_err_in, const double g2_err_in, const double zsrc_in);
	void print_list(bool use_sci);
	void write_to_file(string filename);
	void clear();
	~WeakLensingData();
};

struct ParamAnchor {
	bool anchor_param;
	int paramnum;
	int anchor_paramnum;
	bool use_implicit_ratio;
	bool use_exponent;
	int anchor_object_number;
	double ratio;
	double exponent;
	ParamAnchor() {
		anchor_param = false;
		use_implicit_ratio = false;
		use_exponent = false;
		ratio = 1.0;
		exponent = 1.0;
	}
	void shift_down() { paramnum--; }
};


double mcsampler_set_lensptr(QLens* lens_in);
double polychord_loglikelihood (double theta[], int nDims, double phi[], int nDerived);
void polychord_prior (double cube[], double theta[], int nDims);
void polychord_dumper(int ndead,int nlive,int npars,double* live,double* dead,double* logweights,double logZ, double logZerr);
void multinest_loglikelihood(double *Cube, int &ndim, int &npars, double &lnew, void *context);
void dumper_multinest(int &nSamples, int &nlive, int &nPar, double **physLive, double **posterior, double **paramConstr, double &maxLogLike, double &logZ, double &INSlogZ, double &logZerr, void *context);

// There is too much inheritance going on here. Nearly all of these can be changed to simply objects that are created within the QLens
// class; it's more transparent to do so, and more object-oriented.
class QLens : public Cosmology, public Sort, public Powell, public Simplex, public UCMC
{
	private:
	// These are arrays of dummy variables used for lensing calculations, arranged so that each thread gets its own set of dummy variables.
	static lensvector *defs, **defs_subtot, *defs_i, *xvals_i;
	static lensmatrix *jacs, *hesses, **hesses_subtot, *hesses_i, *Amats_i;
	static int *indxs;

	double raw_chisq;
	int chisq_it;
	bool chisq_diagnostic;
	ofstream logfile;

	public:
	int mpi_id, mpi_np, mpi_ngroups, group_id, group_num, group_np;
	int *group_leader;
#ifdef USE_MPI
	MPI_Comm *group_comm;
	MPI_Comm *my_comm;
	MPI_Group *mpi_group;
	MPI_Group *my_group;
#endif
	static int nthreads;
	int inversion_nthreads;
	int simplex_nmax, simplex_nmax_anneal;
	bool simplex_show_bestfit;
	double simplex_temp_initial, simplex_temp_final, simplex_cooling_factor, simplex_minchisq, simplex_minchisq_anneal;
	int n_livepts; // for nested sampling
	int polychord_nrepeats;
	int mcmc_threads;
	double mcmc_tolerance; // for Metropolis-Hastings
	bool mcmc_logfile;
	bool open_chisq_logfile;
	bool psf_convolution_mpi;
	bool fft_convolution;
	bool use_mumps_subcomm;
	bool n_image_prior;
	double n_images_at_sbmax, pixel_avg_n_image;
	int auxiliary_srcgrid_npixels;
	double sbmin, sbmax;
	double n_image_threshold;
	double max_pixel_sb;
	bool outside_sb_prior;
	double outside_sb_prior_noise_frac, n_image_prior_sb_frac;
	double outside_sb_prior_threshold;
	bool einstein_radius_prior;
	bool concentration_prior;
	double einstein_radius_low_threshold;
	double einstein_radius_high_threshold;
	int extended_mask_n_neighbors;
	bool include_extended_mask_in_inversion;
	bool zero_sb_extended_mask_prior;
	bool include_noise_term_in_loglike;
	double loglike_reference_noise;
	double high_sn_frac;
	bool subhalo_prior;
	bool use_custom_prior;
	bool lens_position_gaussian_transformation;
	ParamSettings *param_settings;

	int nlens;
	LensProfile** lens_list;
	vector<LensProfile*> lens_list_vec;

	int n_sb;
	SB_Profile** sb_list;

	int n_derived_params;
	DerivedParam** dparam_list;

	lensvector source;
	image *images_found;
	ImageSystemType system_type;

	double lens_redshift;
	double source_redshift, reference_source_redshift; // reference zsrc is the redshift used to define the lensing quantities (kappa, etc.)
	double *reference_zfactors;
	double **default_zsrc_beta_factors;
	bool user_changed_zsource;
	bool auto_zsource_scaling;
	double *source_redshifts; // used for modeling source points
	vector<int> source_redshift_groups;
	double **zfactors;
	double ***beta_factors;
	double *lens_redshifts;
	int *lens_redshift_idx;
	int *zlens_group_size;
	int **zlens_group_lens_indx;
	int n_lens_redshifts;
	bool vary_hubble_parameter, vary_omega_matter_parameter;
	double hubble, omega_matter;
	double hubble_lower_limit, hubble_upper_limit;
	double omega_matter_lower_limit, omega_matter_upper_limit;
	bool ellipticity_gradient, contours_overlap;
	double contour_overlap_log_penalty_prior;
	bool vary_syserr_pos_parameter;
	double syserr_pos, syserr_pos_lower_limit, syserr_pos_upper_limit;
	bool vary_wl_shear_factor_parameter;
	double wl_shear_factor, wl_shear_factor_lower_limit, wl_shear_factor_upper_limit;

	int Gauss_NN;	// for Gaussian quadrature
	double romberg_accuracy, integral_tolerance; // for Romberg integration, Gauss-Patterson quadrature
	bool include_recursive_lensing; // should only turn off if trying to understand effect of recursive lensing from multiple lens planes

	Grid *grid;
	bool radial_grid;
	double grid_xlength, grid_ylength, grid_xcenter, grid_ycenter;  // for gridsize
	double sourcegrid_xmin, sourcegrid_xmax, sourcegrid_ymin, sourcegrid_ymax;
	double sourcegrid_limit_xmin, sourcegrid_limit_xmax, sourcegrid_limit_ymin, sourcegrid_limit_ymax;
	bool enforce_min_cell_area;
	bool cc_neighbor_splittings;
	double min_cell_area; // area of the smallest allowed cell area
	int usplit_initial, wsplit_initial;
	int splitlevels, cc_splitlevels;

	QLens *fitmodel;
	QLens *lens_parent;
	dvector fitparams, upper_limits, lower_limits, upper_limits_initial, lower_limits_initial, bestfitparams;
	dmatrix bestfit_fisher_inverse;
	dmatrix fisher_inverse;
	double bestfit_flux;
	double chisq_bestfit;
	SourceFitMode source_fit_mode;
	bool use_ansi_characters;
	int lensmodel_fit_parameters, srcmodel_fit_parameters, n_fit_parameters, n_sourcepts_fit;
	vector<string> fit_parameter_names, transformed_parameter_names;
	vector<string> latex_parameter_names, transformed_latex_parameter_names;
	//lensvector *sourcepts_fit;
	//bool *vary_sourcepts_x;
	//bool *vary_sourcepts_y;
	//lensvector *sourcepts_lower_limit;
	//lensvector *sourcepts_upper_limit;
	vector<lensvector> sourcepts_fit;
	vector<lensvector> sourcepts_lower_limit;
	vector<lensvector> sourcepts_upper_limit;

	vector<vector<image>> point_imgs; // this will store the point images from the first source point in sourcepts_fit when doing source pixel modeling, to generate quasar images
	vector<bool> vary_sourcepts_x;
	vector<bool> vary_sourcepts_y;
	double regularization_parameter, regularization_parameter_upper_limit, regularization_parameter_lower_limit;
	double kernel_correlation_length, kernel_correlation_length_upper_limit, kernel_correlation_length_lower_limit;
	double matern_index, matern_index_upper_limit, matern_index_lower_limit;
	bool use_matern_scale_parameter;
	double matern_scale, matern_scale_upper_limit, matern_scale_lower_limit; // can be used in place of correlation length; it's the magnitude of the Matern kernel at the characteristic size of the source (divided by 3)
	double matern_approx_source_size;
	bool vary_regularization_parameter;
	bool vary_correlation_length;
	bool vary_matern_scale;
	bool vary_matern_index;
	bool optimize_regparam;
	bool optimize_regparam_lhi;
	double optimize_regparam_tol, optimize_regparam_minlog, optimize_regparam_maxlog;
	double regopt_chisqmin, regopt_logdet;
	int max_regopt_iterations;

	// the following parameters are used for luminosity-weighted regularization
	bool use_lum_weighted_regularization;
	//double regparam_lhi, regparam_llo, regparam_lum_index; 
	double regparam_lhi, regparam_lum_index; 
	double *lumreg_pixel_weights;
	int lumreg_it;
	//bool vary_regparam_lhi, vary_regparam_llo, vary_regparam_lum_index;
	bool vary_regparam_lhi, vary_regparam_lum_index;
	double regparam_lhi_lower_limit, regparam_lhi_upper_limit;
	//double regparam_llo_lower_limit, regparam_llo_upper_limit;
	double regparam_lum_index_lower_limit, regparam_lum_index_upper_limit;

	bool use_lum_weighted_srcpixel_clustering;
	double alpha_clus, beta_clus;
	bool vary_alpha_clus, vary_beta_clus;
	double alpha_clus_lower_limit, alpha_clus_upper_limit;
	double beta_clus_lower_limit, beta_clus_upper_limit;

	static string fit_output_filename;
	string get_fit_label() { return fit_output_filename; }
	void set_fit_label(const string label_in) {
		fit_output_filename = label_in;
		if (auto_fit_output_dir) fit_output_dir = "chains_" + fit_output_filename;
	}
	bool auto_save_bestfit;
	bool borrowed_image_data; // tells whether image_data is pointing to that of another QLens object (e.g. fitmodel pointing to initial lens object)
	ImageData *image_data;
	WeakLensingData weak_lensing_data;
	double chisq_tolerance;
	double chisqtol_lumreg;
	int lumreg_max_it, lumreg_max_it_final;
	int n_repeats;
	bool display_chisq_status;
	int n_visible_images;
	int chisq_display_frequency;
	double redundancy_separation_threshold;
	double chisq_magnification_threshold, chisq_imgsep_threshold, chisq_imgplane_substitute_threshold;
	bool use_magnification_in_chisq;
	bool use_magnification_in_chisq_during_repeats;
	bool include_parity_in_chisq;
	bool imgplane_chisq;
	bool calculate_parameter_errors;
	bool adaptive_subgrid;
	bool use_average_magnification_for_subgridding;
	int delaunay_mode;
	bool delaunay_high_sn_mode;
	bool use_srcpixel_clustering;
	bool use_random_delaunay_srcgrid;
	bool reinitialize_random_grid;
	int random_seed;
	int n_ranchisq;
	double random_grid_length_factor;
	bool interpolate_random_sourcepts;
	bool clustering_random_initialization;
	bool use_dualtree_kmeans;
	int n_src_clusters;
	int n_cluster_iterations;
	double delaunay_high_sn_sbfrac;
	bool activate_unmapped_source_pixels;
	double total_srcgrid_overlap_area, high_sn_srcgrid_overlap_area;
	bool exclude_source_pixels_beyond_fit_window;
	bool regrid_if_unmapped_source_subpixels;
	bool calculate_bayes_factor;
	double reference_lnZ;
	double pixel_magnification_threshold, pixel_magnification_threshold_lower_limit, pixel_magnification_threshold_upper_limit;
	double base_srcpixel_imgpixel_ratio;
	double sim_err_pos, sim_err_flux, sim_err_td;
	double sim_err_shear; // actually error in reduced shear (for weak lensing data)
	bool split_imgpixels;
	bool split_high_mag_imgpixels;
	int default_imgpixel_nsplit, emask_imgpixel_nsplit;
	double imgpixel_himag_threshold, imgpixel_lomag_threshold, imgpixel_sb_threshold;

	bool fits_format;
	double data_pixel_size;
	bool add_simulated_image_data(const lensvector &sourcept);
	bool add_image_data_from_sourcepts();
	bool add_image_data_from_sourcepts(const bool include_errors_from_fisher_matrix = false, const int param_i = 0, const double scale_errors = 2);
	bool add_fit_sourcept(const lensvector &sourcept, const double zsrc);
	void write_image_data(string filename);
	bool load_image_data(string filename);
	void sort_image_data_into_redshift_groups();
	bool plot_srcpts_from_image_data(int dataset_number, ofstream* srcfile, const double srcpt_x, const double srcpt_y, const double flux = -1);
	void remove_image_data(int image_set);
	vector<ImageDataSet> export_to_ImageDataSet(); // for the Python wrapper

	bool load_weak_lensing_data(string filename);
	void add_simulated_weak_lensing_data(const string id, lensvector &sourcept, const double zsrc);
	void add_weak_lensing_data_from_random_sources(const int num_sources, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, const double r_exclude);

	bool read_data_line(ifstream& infile, vector<string>& datawords, int &n_datawords);
	bool datastring_convert(const string& instring, int& outvar);
	bool datastring_convert(const string& instring, double& outvar);
	void clear_image_data();
	void clear_sourcepts();

	void print_image_data(bool include_errors);

	bool autocenter;
	int primary_lens_number;
	bool auto_set_primary_lens;
	bool include_secondary_lens;
	int secondary_lens_number;
	bool auto_gridsize_from_einstein_radius;
	bool autogrid_before_grid_creation;
	double autogrid_frac, spline_frac;
	double tabulate_rmin, tabulate_qmin;
	int tabulate_logr_N, tabulate_phi_N, tabulate_q_N;
	int default_parameter_mode;

	bool include_time_delays;
	static bool warnings, newton_warnings; // newton_warnings: when true, displays warnings when Newton's method fails or returns anomalous results
	static bool use_scientific_notation;
	static bool use_ansi_output_during_fit;
	string plot_title, post_title;
	bool suppress_plots;
	string data_info; // Allows for a description of data to be saved in chains_* directory and in FITS file header
	string chain_info; // Allows for a description of chain to be saved in chains_* directory
	string param_markers; // Used to create a file with parameter marker values for mkdist; this is used  by the 'mkposts' command o plt markers
	int n_param_markers;
	bool show_plot_key, plot_key_outside;
	double plot_ptsize, fontsize, linewidth;
	bool show_colorbar, plot_square_axes;
	bool show_imgsrch_grid;
	double colorbar_min, colorbar_max;
	int plot_pttype;
	string fit_output_dir;
	bool auto_fit_output_dir;
	enum TerminalType { TEXT, POSTSCRIPT, PDF } terminal; // keeps track of the file format for plotting
	enum FitMethod { POWELL, SIMPLEX, NESTED_SAMPLING, TWALK, POLYCHORD, MULTINEST } fitmethod;
	enum RegularizationMethod { None, Norm, Gradient, Curvature, Matern_Kernel, Exponential_Kernel, Squared_Exponential_Kernel } regularization_method;
	enum InversionMethod { CG_Method, MUMPS, UMFPACK, DENSE, DENSE_FMATRIX } inversion_method;
	RayTracingMethod ray_tracing_method;
	bool interpolate_sb_3pt;
	bool parallel_mumps, show_mumps_info;

	int n_image_pixels_x, n_image_pixels_y; // note that this is the TOTAL number of pixels in the image, as opposed to image_npixels which gives the # of pixels being fit to
	int srcgrid_npixels_x, srcgrid_npixels_y;
	bool auto_srcgrid_npixels;
	bool auto_srcgrid_set_pixel_size;
	double pixel_fraction, pixel_fraction_lower_limit, pixel_fraction_upper_limit;
	double srcpt_xshift, srcpt_xshift_lower_limit, srcpt_xshift_upper_limit;
	double srcpt_yshift, srcpt_yshift_lower_limit, srcpt_yshift_upper_limit;
	double srcgrid_size_scale, srcgrid_size_scale_lower_limit, srcgrid_size_scale_upper_limit;
	bool vary_pixel_fraction, vary_srcpt_xshift, vary_srcpt_yshift, vary_srcgrid_size_scale, vary_magnification_threshold;
	double psf_width_x, psf_width_y, data_pixel_noise, sim_pixel_noise;
	double sb_threshold; // for creating centroid images from pixel maps
	double noise_threshold; // for automatic source grid sizing

	static const int nmax_lens_planes;
	static const double default_autogrid_rmin, default_autogrid_rmax, default_autogrid_frac, default_autogrid_initial_step;
	static const int max_cc_search_iterations;
	static double rmin_frac;
	static const double default_rmin_frac;
	int cc_thetasteps;
	double cc_rmin, cc_rmax;
	double source_plane_rscale;
	Spline *ccspline;
	Spline *caustic;
	bool cc_splined;
	bool effectively_spherical;
	double newton_magnification_threshold;
	bool reject_himag_images;
	bool reject_images_found_outside_cell;

	Defspline *defspline;

	// private functions are all contained in the file lens.cpp
	bool subgrid_around_perturbers; // if on, will always subgrid around perturbers (with pjaffe profile) when new grid is created
	bool subgrid_only_near_data_images; // if on, only subgrids around perturber galaxies if a data image is within the determined subgridding radius (dangerous if not all images are observed!)
	static double galsubgrid_radius_fraction, galsubgrid_min_cellsize_fraction;
	static int galsubgrid_cc_splittings;
	void subgrid_around_perturber_galaxies(lensvector* centers, double *einstein_radii, const int ihost, double* zfacs, double** betafacs, const int redshift_index);
	//void calculate_critical_curve_perturbation_radius(int lens_number, bool verbose, double &rmax, double& mass_enclosed);
	bool calculate_critical_curve_perturbation_radius_numerical(int lens_number, bool verbose, double& rmax_numerical, double& avg_sigma_enclosed, double& mass_enclosed,  double &rmax_perturber_lensplane, double &mass_enclosed_lensing, bool subtract_unperturbed = false);
	void get_perturber_avgkappa_scaled(int lens_number, const double r0, double &avgkap_scaled, double &menc_scaled, double &avgkap0, bool verbal = false);
	bool find_lensed_position_of_background_perturber(bool verbal, int lens_number, lensvector& pos, double *zfacs, double **betafacs);
	void find_effective_lens_centers_and_einstein_radii(lensvector *centers, double *einstein_radii, int& i_primary, double *zfacs, double **betafacs, bool verbal);
	bool calculate_perturber_subgridding_scale(int lens_number, bool* perturber_list, int host_lens_number, bool verbose, lensvector& center, double& rmax_numerical, double *zfacs, double **betafacs);
	double galaxy_subgridding_scale_equation(const double r);

	double subhalo_perturbation_radius_equation(const double r);
	double perturbation_radius_equation_nosub(const double r);

	// needed for calculating the subhalo perturbation radius and scale for perturber subgridding
	bool use_perturber_flags;
	int perturber_lens_number;
	bool* linked_perturber_list;
	double theta_shear;
	lensvector perturber_center;
	int subgridding_parity_at_center;
	bool subgridding_include_perturber;
	double *subgridding_zfacs;
	double **subgridding_betafacs;

	static const double perturber_einstein_radius_fraction;
	void plot_shear_field(double xmin, double xmax, int nx, double ymin, double ymax, int ny, const string filename = "shearfield.dat");
	void plot_weak_lensing_shear_data(const bool include_model_shear, const string filename = "shear.dat");
	void plot_lensinfo_maps(string file_root, const int x_n, const int y_N, const int rpert_residual);
	void plot_logkappa_map(const int x_N, const int y_N, const string filename, const bool ignore_mask);
	void plot_logmag_map(const int x_N, const int y_N, const string filename);
	void plot_logpot_map(const int x_N, const int y_N, const string filename);

	double average_def_residual(const int x_N, const int y_N, const int perturber_lensnum, double z, double mvir); // testing Despali et al (2017) procedure
	void fit_los_despali();

	struct critical_curve {
		vector<lensvector> cc_pts;
		vector<lensvector> caustic_pts;
		vector<double> length_of_cell; // just to make sure the critical curves are being separated out properly
	};

	vector<lensvector> critical_curve_pts;
	vector<lensvector> caustic_pts;
	vector<double> length_of_cc_cell;
	vector<critical_curve> sorted_critical_curve;
	vector<int> npoints_sorted_critical_curve;
	int n_critical_curves;
	void sort_critical_curves();
	bool sorted_critical_curves;
	static bool auto_store_cc_points;
	vector<lensvector> singular_pts;
	int n_singular_points;

	Vector<dvector> find_critical_curves(bool&);
	double theta_crit;
	double inverse_magnification_r(const double);
	double source_plane_r(const double r);
	bool find_optimal_gridsize();

	bool use_cc_spline; // critical curves can be splined when (approximate) elliptical symmetry is present
	bool auto_ccspline;

	bool auto_sourcegrid, auto_shapelet_scaling, auto_shapelet_center;
	int shapelet_scale_mode;
	double shapelet_max_scale;
	double shapelet_window_scaling;
	SourcePixelGrid *source_pixel_grid;
	DelaunayGrid *delaunay_srcgrid;
	void plot_source_pixel_grid(const char filename[]);

	ImagePixelGrid *image_pixel_grid;
	ImagePixelData *image_pixel_data;
	int image_npixels, source_npixels, source_n_amps;
	int *active_image_pixel_i;
	int *active_image_pixel_j;
	int image_npixels_fgmask;
	int *active_image_pixel_i_fgmask;
	int *active_image_pixel_j_fgmask;
	double *image_surface_brightness;
	double *point_image_surface_brightness;
	double *sbprofile_surface_brightness;
	double *img_minus_sbprofile;
	//double *sbprofile_surface_brightness_fgmask;
	double *source_pixel_vector_input_lumreg; // used to store best-fit solution before optimization of regularization parameter using luminosity-weighted regularization
	double *source_pixel_vector_minchisq; // used to store best-fit solution during optimization of regularization parameter
	double *source_pixel_vector;
	double *source_pixel_n_images;

	int *image_pixel_location_Lmatrix;
	int *source_pixel_location_Lmatrix;
	int Lmatrix_n_elements;
	double *Lmatrix;
	int *Lmatrix_index;
	vector<double> *Lmatrix_rows;
	vector<int> *Lmatrix_index_rows;

	bool assign_pixel_mappings(bool verbal);
	void assign_foreground_mappings(const bool use_data = true);
	double *Dvector;
	double *Dvector_cov;
	double *Dvector_cov_copy;
	int Fmatrix_nn;
	double *Fmatrix;
	double *Fmatrix_copy; // used when optimizing the regularization parameter
	int *Fmatrix_index;
	bool dense_Rmatrix;
	bool find_covmatrix_inverse; // set by user (default=false); if true, finds Rmatrix explicitly (usually more computationally intensive)
	bool use_covariance_matrix; // internal bool; set to true if using covariance kernel reg. and if find_covmatrix_inverse is false
	double *Rmatrix;
	int *Rmatrix_index;
	double *Rmatrix_diag_temp;
	vector<double> *Rmatrix_rows;
	vector<int> *Rmatrix_index_rows;
	int *Rmatrix_row_nn;
	int Rmatrix_nn;
#ifdef USE_MUMPS
	static DMUMPS_STRUC_C *mumps_solver;
#endif

	void convert_Lmatrix_to_dense();
	void assign_Lmatrix_shapelets(bool verbal);
	//void get_zeroth_order_shapelet_vector(bool verbal); // used if shapelet amp00 is varied as a nonlinear parameter
	void PSF_convolution_Lmatrix_dense(const bool verbal);
	void PSF_convolution_Lmatrix_dense_emask(const bool verbal);
	void create_lensing_matrices_from_Lmatrix_dense(const bool verbal);
	void generate_Gmatrix();
	void add_regularization_term_to_dense_Fmatrix();
	double calculate_regularization_term(const bool use_lum_weighting);

	void invert_lens_mapping_dense(bool verbal);
	void optimize_regularization_parameter(const bool dense_Fmatrix, const bool verbal, const bool pre_srcgrid = false);
	void chisq_regparam_single_eval(const double regparam, const bool dense_Fmatrix);
	void setup_regparam_optimization(const bool dense_Fmatrix);
	void calculate_pixel_sbweights();
	double chisq_regparam_dense(const double logreg);
	double chisq_regparam(const double logreg);
	double chisq_regparam_it_lumreg_dense(const double logreg);
	double chisq_regparam_it_lumreg_dense_final(const bool verbal);
	double chisq_regparam_lumreg_dense();
	void add_lum_weighted_reg_term(const bool dense_Fmatrix, const bool use_matrix_copies);
	double brents_min_method(double (QLens::*func)(const double), const double ax, const double bx, const double tol, const bool verbal);
	void calculate_image_pixel_surface_brightness_dense(const bool calculate_foreground = true);
	void create_regularization_matrix_shapelet();
	void generate_Rmatrix_shapelet_gradient();
	void generate_Rmatrix_shapelet_curvature();
	void set_corrlength_for_given_matscale();
	double corrlength_eq_matern_factor(const double log_corr_length);

	//bool Cholesky_dcmp(double** a, double &logdet, int n);
	//bool Cholesky_dcmp_upper(double** a, double &logdet, int n);
	bool Cholesky_dcmp_packed(double* a, double &logdet, int n);
	//void Cholesky_solve(double** a, double* b, double* x, int n);
	void Cholesky_solve_lower_packed(double* a, double* b, double* x, int n);
	void LU_logdet_stacked(double* a, double &logdet, int n);
	void Cholesky_logdet_packed(double* a, double &logdet, int n);
	void Cholesky_logdet_lower_packed(double* a, double &logdet, int n);
	//void test_inverts();
	//void Cholesky_invert_upper(double** a, const int n);
	//void Cholesky_invert_lower(double** a, const int n);
	void Cholesky_invert_upper_packed(double* a, const int n);
	void upper_triangular_syrk(double* a, const int n);
	void repack_matrix_lower(dvector& packed_matrix);
	void repack_matrix_upper(dvector& packed_matrix);

	dmatrix Lmatrix_dense;
	dmatrix Lmatrix_transpose_ptimg_amps; // this contains just the part of the Lmatrix_transpose whose columns will multiply the point image amplitudes
	dvector Gmatrix_stacked;
	dvector Gmatrix_stacked_copy;
	dvector Fmatrix_stacked;
	dvector Fmatrix_packed;
	dvector Fmatrix_packed_copy; // used when optimizing the regularization parameter
	dvector covmatrix_stacked;
	dvector covmatrix_stacked_copy; // used when optimizing the regularization parameter with luminosity weighting
	dvector covmatrix_packed;
	dvector covmatrix_factored;
	dvector Rmatrix_packed;
	dvector temp_src; // used when optimizing the regularization parameter

	double *gmatrix[4];
	int *gmatrix_index[4];
	int *gmatrix_row_index[4];
	vector<double> *gmatrix_rows[4];
	vector<int> *gmatrix_index_rows[4];
	int *gmatrix_row_nn[4];
	int gmatrix_nn[4];

	double *hmatrix[2];
	int *hmatrix_index[2];
	int *hmatrix_row_index[2];
	vector<double> *hmatrix_rows[2];
	vector<int> *hmatrix_index_rows[2];
	int *hmatrix_row_nn[2];
	int hmatrix_nn[2];

	bool use_input_psf_matrix;
	bool use_input_psf_ptsrc_matrix;
	double **psf_matrix, **foreground_psf_matrix;
	Spline2D psf_spline;
	bool load_psf_fits(string fits_filename, const bool verbal);
	void setup_foreground_PSF_matrix();
	int psf_npixels_x, psf_npixels_y;
	int foreground_psf_npixels_x, foreground_psf_npixels_y;
	double psf_threshold, psf_ptsrc_threshold, foreground_psf_threshold;
	int psf_ptsrc_nsplit; // allows for subpixel PSF
	static bool setup_fft_convolution;
	static double *psf_zvec; // for convolutions using FFT
	static int fft_imin, fft_jmin, fft_ni, fft_nj;
#ifdef USE_FFTW
	static complex<double> *psf_transform;
	static complex<double> **Lmatrix_transform;
	static double **Lmatrix_imgs_rvec;
	static double *img_rvec;
	static complex<double> *img_transform;
	static fftw_plan fftplan;
	static fftw_plan fftplan_inverse;
	static fftw_plan *fftplans_Lmatrix;
	static fftw_plan *fftplans_Lmatrix_inverse;
#endif
	static bool setup_fft_convolution_emask;
	static double *psf_zvec_emask; // for convolutions using FFT
	static int fft_imin_emask, fft_jmin_emask, fft_ni_emask, fft_nj_emask;
#ifdef USE_FFTW
	static complex<double> *psf_transform_emask;
	static complex<double> **Lmatrix_transform_emask;
	static double **Lmatrix_imgs_rvec_emask;
	//static double *img_rvec_emask;
	//static complex<double> *img_transform_emask;
	//static fftw_plan fftplan_emask;
	//static fftw_plan fftplan_inverse_emask;
	static fftw_plan *fftplans_Lmatrix_emask;
	static fftw_plan *fftplans_Lmatrix_inverse_emask;
#endif

	//double **Lmatrix_imgs_zvec; // has dimensions (src_npixels,imgpixels*2)

	double Fmatrix_log_determinant, Rmatrix_log_determinant;
	double Gmatrix_log_determinant;
	void initialize_pixel_matrices(bool verbal);
	void initialize_pixel_matrices_shapelets(bool verbal);
	void count_shapelet_npixels();
	void clear_pixel_matrices();
	void clear_lensing_matrices();
	double find_surface_brightness(lensvector &pt);
	void assign_Lmatrix(const bool delaunay, const bool verbal);
	void PSF_convolution_Lmatrix(bool verbal = false);
	//void PSF_convolution_image_pixel_vector(bool verbal = false);
	void PSF_convolution_pixel_vector(double *surface_brightness_vector, const bool foreground = false, const bool verbal = false);
	bool setup_convolution_FFT(const bool verbal);
	bool setup_convolution_FFT_emask(const bool verbal);
	void cleanup_FFT_convolution_arrays();
	void copy_FFT_convolution_arrays(QLens* lens_in);
	void fourier_transform(double* data, const int ndim, int* nn, const int isign);
	void fourier_transform_parallel(double** data, const int ndata, const int jstart, const int ndim, int* nn, const int isign);
	bool generate_PSF_matrix(const double pixel_xlength, const double pixel_ylength);
	bool spline_PSF_matrix(const double xstep, const double ystep);
	double interpolate_PSF_matrix(const double x, const double y);

	void create_regularization_matrix(void);
	void generate_Rmatrix_from_gmatrices();
	void generate_Rmatrix_from_hmatrices();
	void generate_Rmatrix_norm();
	void generate_Rmatrix_from_covariance_kernel(const int kernel_type);
	void create_lensing_matrices_from_Lmatrix(const bool dense_Fmatrix, const bool verbal);
	void invert_lens_mapping_MUMPS(bool verbal, bool use_copy = false);
	void invert_lens_mapping_UMFPACK(bool verbal, bool use_copy = false);
	void convert_Rmatrix_to_dense();
	void Rmatrix_determinant_MKL();
	void Rmatrix_determinant_MUMPS();
	void Rmatrix_determinant_UMFPACK();
	void invert_lens_mapping_CG_method(bool verbal);
	void update_source_amplitudes();
	void indexx(int* arr, int* indx, int nn);

	double set_required_data_pixel_window(bool verbal);

	double image_pixel_chi_square();
	void calculate_source_pixel_surface_brightness();
	void calculate_image_pixel_surface_brightness(const bool calculate_foreground = true);
	void calculate_foreground_pixel_surface_brightness(const bool allow_lensed_nonshapelet_sources = true);
	void add_foreground_to_image_pixel_vector();
	void store_image_pixel_surface_brightness();
	void store_foreground_pixel_surface_brightness();
	void vectorize_image_pixel_surface_brightness(bool use_mask = false);
	void plot_image_pixel_surface_brightness(string outfile_root);
	double invert_image_surface_brightness_map(double& chisq0, bool verbal);
	//double calculate_chisq0_from_srcgrid(double &chisq0, bool verbal);

	void load_pixel_grid_from_data();
	double invert_surface_brightness_map_from_data(double& chisq0, bool verbal);
	void plot_image_pixel_grid();
	bool find_shapelet_scaling_parameters(const bool verbal);
	bool set_shapelet_imgpixel_nsplit();

	void update_source_amplitudes_from_shapelets();
	int get_shapelet_nn();

	void find_optimal_sourcegrid_for_analytic_source();
	bool create_sourcegrid_cartesian(const bool verbal, const bool autogrid_from_analytic_source = true, const bool image_grid_already_exists = false, const bool use_nimg_prior_npixels = false);
	bool create_sourcegrid_delaunay(const bool use_mask, const bool verbal);
	void create_sourcegrid_from_imggrid_delaunay(const bool use_weighted_srcpixel_clustering, const bool verbal);
	void create_random_delaunay_sourcegrid(const bool use_weighted_probability, const bool verbal);
	void generate_random_regular_imgpts(double *imgpts_x, double *imgpts_y, double *srcpts_x, double *srcpts_y, int& n_imgpts, int *ivals, int *jvals, const bool use_lum_weighted_number_density, const bool verbal);
	void load_source_surface_brightness_grid(string source_inputfile);
	bool load_image_surface_brightness_grid(string image_pixel_filename_root);
	bool make_image_surface_brightness_data();
	bool plot_lensed_surface_brightness(string imagefile, const int reduce_factor, bool output_fits = false, bool plot_residual = false, bool plot_foreground_only = false, bool omit_foreground = false, bool show_mask_only = true, bool offload_to_data = false, bool show_extended_mask = false, bool show_foreground_mask = false, bool show_noise_thresh = false, bool exclude_ptimgs = false, bool verbose = true);

	void plot_Lmatrix();
	void check_Lmatrix_columns();
	double temp_double;
	void Swap(double& a, double& b) { temp_double = a; a = b; b = temp_double; }

	double wtime0, wtime; // for calculating wall time in parallel calculations
	bool show_wtime;

	friend class Grid;
	friend class SourcePixelGrid;
	friend class ImagePixelGrid;
	friend class ImagePixelData;
	friend struct DerivedParam;
	friend class LensProfile;
	friend class SB_Profile;
	QLens();
	QLens(QLens *lens_in);
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	~QLens();
#ifdef USE_MPI
	void set_mpi_params(const int& mpi_id_in, const int& mpi_np_in, const int& mpi_ngroups_in, const int& group_num_in, const int& group_id_in, const int& group_np_in, int* group_leader_in, MPI_Group* group_in, MPI_Comm* comm, MPI_Group* mygroup, MPI_Comm* mycomm);
#endif
	void set_mpi_params(const int& mpi_id_in, const int& mpi_np_in);
	void set_nthreads(const int& nthreads_in) { nthreads=nthreads_in; }
#ifdef USE_MUMPS
	static void setup_mumps();
#endif
	static void delete_mumps();

	double kappa(const double& x, const double& y, double* zfacs, double** betafacs);
	//double kappa_weak(const double& x, const double& y, double* zfacs);
	double potential(const double&, const double&, double* zfacs, double** betafacs);
	void deflection(const double&, const double&, lensvector&, const int &thread, double* zfacs, double** betafacs);
	void deflection(const double& x, const double& y, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs);
	void custom_deflection(const double& x, const double& y, lensvector& def_tot);
	void map_to_lens_plane(const int& redshift_i, const double& x, const double& y, lensvector& xi, const int &thread, double* zfacs, double** betafacs);
	void hessian(const double&, const double&, lensmatrix&, const int &thread, double* zfacs, double** betafacs);
	void hessian_weak(const double&, const double&, lensmatrix&, const int &thread, double* zfacs);
	void find_sourcept(const lensvector& x, lensvector& srcpt, const int &thread, double* zfacs, double** betafacs);
	void find_sourcept(const lensvector& x, double& srcpt_x, double& srcpt_y, const int &thread, double* zfacs, double** betafacs);
	void kappa_inverse_mag_sourcept(const lensvector& x, lensvector& srcpt, double &kap_tot, double &invmag, const int &thread, double* zfacs, double** betafacs);
	void sourcept_jacobian(const lensvector& xvec, lensvector& srcpt, lensmatrix& jac_tot, const int &thread, double* zfacs, double** betafacs);

	// non-multithreaded versions
	//void deflection(const double& x, const double& y, lensvector &def_in, double* zfacs) { deflection(x,y,def_in,0,zfacs); }
	//void hessian(const double& x, const double& y, lensmatrix &hess_in, double* zfacs) { hessian(x,y,hess_in,0,zfacs); }
	//void find_sourcept(const lensvector& x, lensvector& srcpt, double* zfacs) { find_sourcept(x,srcpt,0,zfacs); }

	// versions of the above functions that use lensvector for (x,y) coordinates
	double kappa(const lensvector &x, double* zfacs, double** betafacs) { return kappa(x[0], x[1], zfacs, betafacs); }
	double potential(const lensvector& x, double* zfacs, double** betafacs) { return potential(x[0],x[1], zfacs, betafacs); }
	void deflection(const lensvector& x, lensvector& def, double* zfacs, double** betafacs) { deflection(x[0], x[1], def, 0, zfacs, betafacs); }
	void hessian(const lensvector& x, lensmatrix& hess, double* zfacs, double** betafacs) { hessian(x[0], x[1], hess, 0, zfacs, betafacs); }

	double inverse_magnification(const lensvector&, const int &thread, double* zfacs, double** betafacs);
	double magnification(const lensvector &x, const int &thread, double* zfacs, double** betafacs);
	double shear(const lensvector &x, const int &thread, double* zfacs, double** betafacs);
	void shear(const lensvector &x, double& shear_tot, double& angle, const int &thread, double* zfacs, double** betafacs);
	void reduced_shear_components(const lensvector &x, double& g1, double& g2, const int &thread, double* zfacs);

	// non-multithreaded versions
	//double inverse_magnification(const lensvector& x, double* zfacs) { return inverse_magnification(x,0,zfacs); }
	//double magnification(const lensvector &x, double* zfacs) { return magnification(x,0,zfacs); }
	//double shear(const lensvector &x, double* zfacs) { return shear(x,0,zfacs); }
	//void shear(const lensvector &x, double& shear_tot, double& angle, double* zfacs) { return shear(x,shear_tot,angle,0,zfacs); }

	void hessian_exclude(const double& x, const double& y, bool* exclude, lensmatrix& hess_tot, const int& thread, double* zfacs, double** betafacs);
	double magnification_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
	double inverse_magnification_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
	double shear_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs);
	void shear_exclude(const lensvector &x, double& shear, double& angle, bool* exclude, const int& thread, double* zfacs, double** betafacs);
	double kappa_exclude(const lensvector &x, bool* exclude, double* zfacs, double** betafacs);
	void deflection_exclude(const double& x, const double& y, bool* exclude, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs);

	// non-multithreaded versions
	void hessian_exclude(const double& x, const double& y, bool* exclude, lensmatrix& hess_tot, double* zfacs, double** betafacs) { hessian_exclude(x,y,exclude,hess_tot,0,zfacs,betafacs); }
	double magnification_exclude(const lensvector &x, bool* exclude, double* zfacs, double** betafacs) { return magnification_exclude(x,exclude,0,zfacs,betafacs); }
	double inverse_magnification_exclude(const lensvector &x, bool* exclude, double* zfacs, double** betafacs) { return inverse_magnification_exclude(x,exclude,0,zfacs,betafacs); }
	double shear_exclude(const lensvector &x, bool* exclude, double* zfacs, double** betafacs) { return shear_exclude(x,exclude,0,zfacs,betafacs); }
	void shear_exclude(const lensvector &x, double &shear, double &angle, bool* exclude, double* zfacs, double** betafacs) { shear_exclude(x,shear,angle,exclude,0,zfacs,betafacs); }
	void deflection_exclude(const lensvector& x, bool* exclude, lensvector& def, double* zfacs, double** betafacs) { deflection_exclude(x[0], x[1], exclude, def[0], def[1], 0, zfacs, betafacs); }

/*
	void hessian_exclude(const double& x, const double& y, const int& exclude_i, lensmatrix& hess_tot, const int& thread, double* zfacs, double** betafacs);
	double magnification_exclude(const lensvector &x, const int& exclude_i, const int& thread, double* zfacs, double** betafacs);
	double shear_exclude(const lensvector &x, const int& exclude_i, const int& thread, double* zfacs, double** betafacs);
	void shear_exclude(const lensvector &x, double& shear, double& angle, const int& exclude_i, const int& thread, double* zfacs, double** betafacs);
	double kappa_exclude(const lensvector &x, const int& exclude_i, double* zfacs, double** betafacs);

	// non-multithreaded versions
	void hessian_exclude(const double& x, const double& y, const int& exclude_i, lensmatrix& hess_tot, double* zfacs, double** betafacs) { hessian_exclude(x,y,exclude_i,hess_tot,0,zfacs,betafacs); }
	double magnification_exclude(const lensvector &x, const int& exclude_i, double* zfacs, double** betafacs) { return magnification_exclude(x,exclude_i,0,zfacs,betafacs); }
	double shear_exclude(const lensvector &x, const int& exclude_i, double* zfacs, double** betafacs) { return shear_exclude(x,exclude_i,0,zfacs,betafacs); }
	void shear_exclude(const lensvector &x, double &shear, double &angle, const int& exclude_i, double* zfacs, double** betafacs) { shear_exclude(x,shear,angle,exclude_i,0,zfacs,betafacs); }
	*/

	bool test_for_elliptical_symmetry();
	bool test_for_singularity();
	void record_singular_points(double *zfacs);

	// the following functions and objects are contained in commands.cpp
	char *buffer;
	int nullflag, buffer_length;
	string line;
	vector<string> lines;
	ifstream infile_list[10];
	ifstream *infile; // used to read commands from an input file
	int n_infiles;
	bool verbal_mode;
	bool quit_after_error;
	int nwords;
	vector<string> words;
	stringstream* ws;
	stringstream datastream;
	bool read_from_file;
	bool paused_while_reading_file;
	bool quit_after_reading_file;
	void process_commands(bool read_file);
	bool read_command(bool show_prompt);
	bool check_vary_z();
	bool read_egrad_params(const bool vary_params, const int egrad_mode, dvector& efunc_params, int& nparams_to_vary, boolvector& varyflags, const int default_nparams, const double xc, const double yc, ParamAnchor* parameter_anchors, int& parameter_anchor_i, int& n_bspline_coefs, dvector& knots, double& ximin, double& ximax, double& xiref, bool& linear_xivals, bool& enter_params_and_varyflags, bool& enter_knots);
	bool read_fgrad_params(const bool vary_params, const int egrad_mode, const int n_fmodes, const vector<int> fourier_mvals, dvector& fgrad_params, int& nparams_to_vary, boolvector& varyflags, const int default_nparams, ParamAnchor* parameter_anchors, int& parameter_anchor_i, int n_bspline_coefs, dvector& knots, const bool enter_params_and_varyflags, const bool enter_knots);
	void run_plotter(string plotcommand, string extra_command = "");
	void run_plotter_file(string plotcommand, string filename, string range = "", string extra_command = "");
	void run_plotter_range(string plotcommand, string range, string extra_command = "");
	void run_mkdist(bool copy_post_files, string posts_dirname, const int nbins_1d, const int nbins_2d, bool copy_subplot_only, bool resampled_posts, bool no2dposts, bool nohists);
	void remove_equal_sign();
	void remove_word(int n_remove);
	void remove_comments(string& instring);
	void remove_equal_sign_datafile(vector<string>& datawords, int& n_datawords);

	void set_show_wtime(bool show_wt) { show_wtime = show_wt; }
	void set_verbal_mode(bool echo) { verbal_mode = echo; }
	bool open_script_file(const string filename);
	void set_quit_after_reading_file(bool setting) { quit_after_reading_file = setting; }
	void set_suppress_plots(bool setting) { suppress_plots = setting; }

	void extract_word_starts_with(const char initial_character, int starting_word, int ending_word, string& extracted_word);
	void extract_word_starts_with(const char initial_character, int starting_word, string& extracted_word) { extract_word_starts_with(initial_character,starting_word,1000,extracted_word); }
	bool extract_word_starts_with(const char initial_character, int starting_word, int ending_word, vector<string>& extracted_words);
	void set_quit_after_error(bool arg) { quit_after_error = arg; }
	void set_plot_title(int starting_word, string& temp_title);

	// the following functions are contained in imgsrch.cpp
	private:
	void find_images();

	public:
	bool plot_recursive_grid(const char filename[]);
	void output_images_single_source(const double &x_source, const double &y_source, bool verbal, const double flux = -1.0, const bool show_labels = false);
	bool plot_images_single_source(const double &x_source, const double &y_source, bool verbal, const double flux = -1.0, const bool show_labels = false, string imgheader = "", string srcheader = "") {
		ofstream imgfile; open_output_file(imgfile,"imgs.dat");
		ofstream srcfile; open_output_file(srcfile,"srcs.dat");
		if (!imgheader.empty()) imgfile << "\"" << imgheader << "\"" << endl;
		if (!srcheader.empty()) srcfile << "\"" << srcheader << "\"" << endl;
		return plot_images_single_source(x_source,y_source,verbal,imgfile,srcfile,flux,show_labels);
	}
	bool plot_images_single_source(const double &x_source, const double &y_source, bool verbal, ofstream& imgfile, ofstream& srcfile, const double flux = -1.0, const bool show_labels = false);
	image* get_images(const lensvector &source_in, int &n_images) { return get_images(source_in, n_images, true); }
	image* get_images(const lensvector &source_in, int &n_images, bool verbal);
	bool get_imageset(const double src_x, const double src_y, ImageSet& image_set, bool verbal = true); // used by Python wrapper
	vector<ImageSet> get_fit_imagesets(bool& status, int min_dataset = 0, int max_dataset = -1, bool verbal = true);
	bool plot_images(const char *sourcefile, const char *imagefile, bool verbal);
	void lens_equation(const lensvector&, lensvector&, const int& thread, double *zfacs, double **betafacs); // Used by Newton's method to find images

	// the remaining functions in this class are all contained in lens.cpp
	void create_and_add_lens(LensProfileName, const int emode, const double zl, const double zs, const double mass_parameter, const double logslope_param, const double scale, const double core, const double q, const double theta, const double xc, const double yc, const double extra_param1 = -1000, const double extra_param2 = -1000, const int parameter_mode = 0);
	void add_shear_lens(const double zl, const double zs, const double shear, const double theta, const double xc, const double yc); // specific version for shear model
	void add_ptmass_lens(const double zl, const double zs, const double mass_parameter, const double xc, const double yc, const int pmode); // specific version for ptmass model
	void add_mass_sheet_lens(const double zl, const double zs, const double mass_parameter, const double xc, const double yc); // specific version for mass sheet
	bool spawn_lens_from_source_object(const int src_number, const double zl, const double zs, const int pmode, const bool vary_mass_parameter, const bool include_limits, const double mass_param_lower, const double mass_param_upper);

	void add_lens(LensProfile *new_lens, const double zl, const double zs);
	void add_new_lens_redshift(const double zl, const int lens_i, int* zlens_idx);
	void remove_old_lens_redshift(const int znum, const int lens_i, const bool removed_lens);
	void update_lens_redshift_data();
	void add_new_lens_entry(const double zl);
	void set_primary_lens();
	void print_beta_matrices();
	void set_source_redshift(const double zsrc);
	double get_source_redshift() { return source_redshift; }
	void set_reference_source_redshift(const double zsrc);
	double get_reference_source_redshift() { return reference_source_redshift; }
	void recalculate_beta_factors();
	void set_sci_notation(const bool scinot) {
		use_scientific_notation = scinot;
		if (use_scientific_notation) cout << setiosflags(ios::scientific);
		else {
			cout << resetiosflags(ios::scientific);
			cout.unsetf(ios_base::floatfield);
		}
	}
	bool get_sci_notation() { return use_scientific_notation; }

	void add_multipole_lens(const double zl, const double zs, int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool kap, bool sine_term);
	void add_tabulated_lens(const double zl, const double zs, int lnum, const double kscale, const double rscale, const double theta, const double xc, const double yc);
	bool add_tabulated_lens_from_file(const double zl, const double zs, const double kscale, const double rscale, const double theta, const double xc, const double yc, const string tabfileroot);
	bool add_qtabulated_lens_from_file(const double zl, const double zs, const double kscale, const double rscale, const double q, const double theta, const double xc, const double yc, const string tabfileroot);
	bool save_tabulated_lens_to_file(int lnum, const string tabfileroot);
	void add_qtabulated_lens(const double zl, const double zs, int lnum, const double kscale, const double rscale, const double q, const double theta, const double xc, const double yc);

	void create_and_add_lens(const char *splinefile, const int emode, const double zl, const double zs, const double q, const double theta, const double qx, const double f, const double xc, const double yc);
	bool set_lens_vary_parameters(const int lensnumber, boolvector &vary_flags);
	bool register_lens_vary_parameters(const int lensnumber);
	bool set_sb_vary_parameters(const int sbnumber, boolvector &vary_flags);
	void update_parameter_list();
	void update_anchored_parameters_and_redshift_data();
	void reassign_lensparam_pointers_and_names();
	void reassign_sb_param_pointers_and_names();
	void print_lens_list(bool show_vary_params);
	LensProfile* get_lens_pointer(const int lensnum) { if (lensnum >= nlens) return NULL; else return lens_list[lensnum]; }
	void output_lens_commands(string filename, const bool use_limits);
	void print_sourcept_list();
	void print_fit_model();
	void print_lens_cosmology_info(const int lmin, const int lmax);
	bool output_mass_r(const double r_arcsec, const int lensnum, const bool use_kpc);
	double mass2d_r(const double r_arcsec, const int lensnum, const bool use_kpc);
	double mass3d_r(const double r_arcsec, const int lensnum, const bool use_kpc);
	double calculate_average_log_slope(const int lensnum, const double rmin, const double rmax, const bool use_kpc);

	void add_source_object(SB_ProfileName name, const int emode, const double sb_norm, const double scale, const double scale2, const double logslope_param, const double q, const double theta, const double xc, const double yc, const double special_param1 = -1, const double special_param2 = -1);
	void add_source_object(const char *splinefile, const int emode, const double q, const double theta, const double qx, const double f, const double xc, const double yc);
	void add_multipole_source(int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool sine_term);
	void add_shapelet_source(const double amp00, const double sig_x, const double q, const double theta, const double xc, const double yc, const int nmax, const bool truncate, const int pmode = 0);

	void remove_source_object(int sb_number);
	void clear_source_objects();
	void print_source_list(bool show_vary_params);

	void add_derived_param(DerivedParamType type_in, double param, int lensnum, double param2 = -1e30, bool use_kpc = false);
	void remove_derived_param(int dparam_number);
	void rename_derived_param(int dparam_number, string newname, string new_latex_name);
	void clear_derived_params();
	void print_derived_param_list();
	void clear_raw_chisq() { raw_chisq = -1e30; if (fitmodel) fitmodel->raw_chisq = -1e30; }

	bool create_grid(bool verbal, double *zfacs, double **betafacs, const int redshift_index = -1); // the last (optional) argument indicates which images are being fit to; used to optimize the subgridding
	void find_automatic_grid_position_and_size(double *zfacs);
	void clear_lenses();
	void clear();
	void reset_grid();
	void remove_lens(int lensnumber);
	void toggle_major_axis_along_y(bool major_axis_along_y);
	void create_output_directory();
	void open_output_file(ofstream &outfile, string filename_in);
	void open_output_file(ofstream &outfile, char* filechar_in);

	private:
	bool temp_auto_ccspline, temp_auto_store_cc_points, temp_include_time_delays;
	void fit_set_optimizations();
	void fit_restore_defaults();
	double zfac_re; // used by einstein_radius_root(...)

	public:
	double chi_square_fit_simplex();
	double chi_square_fit_powell();
	void output_fit_results(dvector& stepsizes, const double chisq_bestfit, const int chisq_evals);
	void nested_sampling();
	void polychord(const bool resume_previous, const bool skip_run);
	void multinest(const bool resume_previous, const bool skip_run);
	void chi_square_twalk();
	bool add_dparams_to_chain();
	bool adopt_bestfit_point_from_chain();
	bool adopt_point_from_chain(const unsigned long point_num);
	bool adopt_point_from_chain_paramrange(const int paramnum, const double minval, const double maxval);
	bool plot_kappa_profile_percentiles_from_chain(int lensnum, double rmin, double rmax, int nbins, const string kappa_filename);
	bool output_scaled_percentiles_from_chain(const double pct_scaling);
	double find_percentile(const unsigned long npoints, const double pct, const double tot, double *pts, double *weights);
	bool output_scaled_percentiles_from_egrad_fits(const double xcavg, const double ycavg, const double qtheta_pct_scaling = 1.0, const double fmode_pct_scaling = 1.0, const bool include_m3_fmode = false, const bool include_m4_fmode = false);

	void plot_chisq_2d(const int param1, const int param2, const int n1, const double i1, const double f1, const int n2, const double i2, const double f2);
	void plot_chisq_1d(const int param, const int n, const double i, const double f, string filename);
	double chisq_single_evaluation(bool showdiag, bool show_status);
	bool setup_fit_parameters(bool include_limits);
	bool setup_limits();
	void get_n_fit_parameters(int &nparams);
	void get_parameter_names();
	bool get_lens_parameter_numbers(const int lens_i, int& pi, int& pf);
	bool get_sb_parameter_numbers(const int lens_i, int& pi, int& pf);
	bool lookup_parameter_value(const string pname, double& pval);
	void create_parameter_value_string(string &pvals);
	bool output_parameter_values();
	bool output_parameter_prior_ranges();
	bool update_parameter_value(const int param_num, const double param_val);

	void get_automatic_initial_stepsizes(dvector& stepsizes);
	void set_default_plimits();
	bool initialize_fitmodel(const bool running_fit_in);
	double update_model(const double* params);
	double fitmodel_loglike_point_source(double* params);
	double fitmodel_loglike_extended_source(double* params);
	double fitmodel_custom_prior();
	double LogLikeFunc(double *params) { return (this->*LogLikePtr)(params); }
	void DerivedParamFunc(double *params, double *dparams) { (this->*DerivedParamPtr)(params,dparams); }
	void fitmodel_calculate_derived_params(double* params, double* derived_params);
	double get_lens_parameter_using_default_pmode(const int lensnum, const int paramnum);
	double loglike_point_source(double* params);
	bool calculate_fisher_matrix(const dvector &params, const dvector &stepsizes);
	double loglike_deriv(const dvector &params, const int index, const double step);
	void output_bestfit_model();
	bool adopt_model(dvector &fitparams);
	bool use_bestfit() { return adopt_model(bestfitparams); }

	bool include_central_image;
	bool include_imgpos_chisq, include_flux_chisq, include_time_delay_chisq;
	bool include_weak_lensing_chisq;
	bool use_analytic_bestfit_src;
	bool n_images_penalty;
	bool analytic_source_flux;
	double source_flux;
	bool vary_srcflux;
	double srcflux_lower_limit, srcflux_upper_limit;
	bool include_imgfluxes_in_inversion;

	bool spline_critical_curves(bool verbal);
	bool spline_critical_curves() { return spline_critical_curves(true); }
	void automatically_determine_ccspline_mode();
	bool plot_splined_critical_curves(string filename = "");
	bool plot_sorted_critical_curves(string filename = "");
	bool (QLens::*plot_critical_curves)(string filename);
	bool plotcrit(string filename) { return (this->*plot_critical_curves)(filename); }
	bool plotcrit_exclude_subhalo(string filename, int exclude_lensnum)
	{
		bool worked = false;
		double mvir;
		if (lens_list[exclude_lensnum]->get_specific_parameter("mvir",mvir)==true) {
			lens_list[exclude_lensnum]->update_specific_parameter("mvir",1e-3);
			worked = (this->*plot_critical_curves)(filename);
			lens_list[exclude_lensnum]->update_specific_parameter("mvir",mvir);
		}
		return worked;
	}
	void plot_ray_tracing_grid(double xmin, double xmax, double ymin, double ymax, int x_N, int y_N, string filename);

	void make_source_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename);
	void make_source_ellipse(const double xcenter, const double ycenter, const double major_axis, const double q, const double angle, const int n_subellipses, const int points_per_ellipse, string source_filename);
	void raytrace_image_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename);

	void plot_kappa_profile(int l, double rmin, double rmax, int steps, const char *kname, const char *kdname = NULL);
	void plot_total_kappa(double rmin, double rmax, int steps, const char *kname, const char *kdname = NULL);
	void plot_sb_profile(int l, double rmin, double rmax, int steps, const char *sname);
	void plot_total_sbprofile(double rmin, double rmax, int steps, const char *sbname);
	double total_kappa(const double r, const int lensnum, const bool use_kpc);
	double total_dkappa(const double r, const int lensnum, const bool use_kpc);
	double einstein_radius_single_lens(const double src_redshift, const int lensnum);
	bool *centered;
	double einstein_radius_of_primary_lens(const double zfac, double& reav);
	double einstein_radius_root(const double r);
	double get_einstein_radius_prior(const bool verbal);
	void plot_mass_profile(double rmin, double rmax, int steps, const char *massname);
	void plot_matern_function(double rmin, double rmax, int rpts, const char *mfilename);
	void print_lensing_info_at_point(const double x, const double y);
	bool make_random_sources(int nsources, const char *outfile);
	bool total_cross_section(double&);
	double total_cross_section_integrand(const double);

	double chisq_pos_source_plane();
	double chisq_pos_image_plane();
	//double chisq_pos_image_plane_diagnostic(const bool verbose, double& rms_imgpos_err, int& n_matched_images);
	double chisq_pos_image_plane_diagnostic(const bool verbose, const bool output_residuals_to_file, double& rms_imgpos_err, int& n_matched_images, const string output_filename = "fit_chivals.dat");

	double chisq_flux();
	double chisq_time_delays();
	double chisq_weak_lensing();
	bool output_weak_lensing_chivals(string filename);
	//void output_imgplane_chisq_vals(); // what was this for?
	void output_model_source_flux(double *bestfit_flux);
	void output_analytic_srcpos(lensvector *beta_i);
	void set_analytic_sourcepts(const bool verbal = false);

	static bool respline_at_end;
	static int resplinesteps;
	void create_deflection_spline(int steps);
	void spline_deflection(double xl, double yl, int steps);
	bool autospline_deflection(int steps);
	bool unspline_deflection();
	bool isspherical();
	bool islens() { return (nlens > 0); }
	void set_grid_corners(double xmin, double xmax, double ymin, double ymax);
	void set_grid_from_pixels();

	void set_gridsize(double xl, double yl);
	void set_gridcenter(double xc, double yc);
	void autogrid(double rmin, double rmax, double frac);
	void autogrid(double rmin, double rmax);
	void autogrid();
	bool get_deflection_spline_info(double &xmax, double &ymax, int &nsteps);
	void delete_ccspline();
	void set_Gauss_NN(const int& nn);
	void set_integral_tolerance(const double& acc);
	void set_integral_convergence_warnings(const bool warn);

	void set_integration_method(IntegrationMethod method) { LensProfile::integral_method = method; }
	void set_analytic_bestfit_src(bool setting) {
		use_analytic_bestfit_src = setting;
		update_parameter_list();
	}
	bool get_analytic_bestfit_src() { return use_analytic_bestfit_src; }

	void set_warnings(bool setting) { warnings = setting; }
	void get_warnings(bool &setting) { setting = warnings; }
	void set_newton_warnings(bool setting) { newton_warnings = setting; }
	void get_newton_warnings(bool &setting) { setting = newton_warnings; }
	void set_ccspline_mode(bool setting) { use_cc_spline = setting; plot_critical_curves = (setting==true) ? &QLens::plot_splined_critical_curves : &QLens::plot_sorted_critical_curves; }
	void get_ccspline_mode(bool &setting) { setting = use_cc_spline; }
	void set_auto_ccspline_mode(bool setting) { auto_ccspline = setting; }
	void get_auto_ccspline_mode(bool &setting) { setting = auto_ccspline; }
	void set_galsubgrid_mode(bool setting) { subgrid_around_perturbers = setting; }
	void get_galsubgrid_mode(bool &setting) { setting = subgrid_around_perturbers; }
	void set_auto_store_cc_points(bool setting) { auto_store_cc_points = setting; }

	void set_usplit_initial(int setting) { usplit_initial = setting; }
	void get_usplit_initial(int &setting) { setting = usplit_initial; }
	void set_wsplit_initial(int setting) { wsplit_initial = setting; }
	void get_wsplit_initial(int &setting) { setting = wsplit_initial; }
	void set_splitlevels(int setting) { splitlevels = setting; }
	void get_splitlevels(int &setting) { setting = splitlevels; }
	void set_cc_splitlevels(int setting) { cc_splitlevels = setting; }
	void get_cc_splitlevels(int &setting) { setting = cc_splitlevels; }
	void set_rminfrac(double setting) { rmin_frac = setting; }
	void get_rminfrac(double &setting) { setting = rmin_frac; }
	void set_imagepos_accuracy(double setting) { Grid::set_imagepos_accuracy(setting); }
	void set_galsubgrid_radius_fraction(double setting) { galsubgrid_radius_fraction = setting; }
	void get_galsubgrid_radius_fraction(double &setting) { setting = galsubgrid_radius_fraction; }
	void set_galsubgrid_min_cellsize_fraction(double setting) { galsubgrid_min_cellsize_fraction = setting; }
	void get_galsubgrid_min_cellsize_fraction(double &setting) { setting = galsubgrid_min_cellsize_fraction; }
	void get_time_delay_setting(bool &setting) { setting = include_time_delays; }
	void set_time_delay_setting(bool setting) { include_time_delays = setting; }
	void set_inversion_nthreads(const int &nt) { inversion_nthreads = nt; }
	void set_mumps_mpi(const bool &setting) { use_mumps_subcomm = setting; }
	void set_fitmethod(FitMethod fitmethod_in)
	{
		fitmethod = fitmethod_in;
		if ((fitmethod==POWELL) or (fitmethod==SIMPLEX)) {
			for (int i=0; i < nlens; i++) lens_list[i]->set_include_limits(false);
			for (int i=0; i < n_sb; i++) sb_list[i]->set_include_limits(false);
		} else {
			for (int i=0; i < nlens; i++) lens_list[i]->set_include_limits(true);
			for (int i=0; i < n_sb; i++) sb_list[i]->set_include_limits(true);
		}
		if ((n_sourcepts_fit > 0) and ((fitmethod != POWELL) and (fitmethod != SIMPLEX))) {
			if (sourcepts_lower_limit.empty()) sourcepts_lower_limit.resize(n_sourcepts_fit);
			if (sourcepts_upper_limit.empty()) sourcepts_upper_limit.resize(n_sourcepts_fit);
			for (int i=0; i < nlens; i++) lens_list[i]->set_include_limits(true);
		}
	}
	void transform_cube(double* params, double* Cube) {
		for (int i=0; i < n_fit_parameters; i++) {
			params[i] = lower_limits[i] + Cube[i]*(upper_limits[i]-lower_limits[i]);
		}
	}
	bool get_einstein_radius(int lens_number, double& re_major_axis, double& re_average);

	double crit0_interpolate(double theta) { return ccspline[0].splint(theta); }
	double crit1_interpolate(double theta) { return ccspline[1].splint(theta); }
	double caust0_interpolate(double theta);
	double caust1_interpolate(double theta);

	//double make_perturber_population(const double number_density, const double rmax, const double a, const double b);
	//void plot_perturber_deflection_vs_area();

	//void generate_solution_chain_sdp81(); // specialty function...probably should put in separate file & header file; do this later
	double rmax_true_mc, menc_true_mc;
	void plot_mc_curve(const int lensnumber, const double logm_min, const double logm_max, const string filename);
	double croot_eq(const double c);
	double rmax_true_mz, menc_true_mz, rmax_z_true_mz, avgkap_scaled_true_mz;
	double NFW_def_function(const double x);
	void plot_mz_curve(const int lensnumber, const double logm_min, const double logm_max, const double yslope_lowz, const double yslope_hiz, const bool keep_dr_const, const string filename);
	double zroot_eq(const double z);
	double muroot_eq(const double mu);
	double mrroot_eq(const double mu);
	double mrroot_eq0(const double mu);
	void plot_mz_bestfit(const int lensnumber, const double zmin, const double zmax, const double zstep, string filename);
	void find_bestfit_smooth_model(const int lensnumber);
	void find_equiv_mvir(const double newc);
	double mroot_eq(const double c);
	void test_lens_functions();
};

struct ImageData
{
	int n_images;
	lensvector *pos;
	double *flux;
	double *time_delays;
	double *sigma_pos, *sigma_f, *sigma_t;
	bool *use_in_chisq;
	double max_distsqr; // maximum squared distance between any pair of images
	ImageData() { n_images = 0; }
	void input(const int &nn);
	void input(const ImageData& imgs_in);
	//void input(const int &nn, image* images, const double sigma_pos_in, const double sigma_flux_in, const double sigma_td_in, bool* include, bool include_time_delays);
	void input(const int &nn, image* images, double* sigma_pos_in, double* sigma_flux_in, const double sigma_td_in, bool* include, bool include_time_delays);
	void add_image(lensvector& pos_in, const double sigma_pos_in, const double flux_in, const double sigma_f_in, const double time_delay_in, const double sigma_t_in);
	void print_list(bool print_errors, bool use_sci);
	void write_to_file(ofstream &outfile);
	bool set_use_in_chisq(int image_i, bool use_in_chisq_in);
	~ImageData();
};

struct ParamPrior
{
	double gaussian_pos, gaussian_sig;
	dmatrix covariance_matrix, inv_covariance_matrix;
	dvector gauss_meanvals;
	ivector gauss_paramnums;
	Prior prior;
	ParamPrior() { prior = UNIFORM_PRIOR; }
	ParamPrior(ParamPrior *prior_in)
	{
		prior = prior_in->prior;
		if (prior==GAUSS_PRIOR) {
			gaussian_pos = prior_in->gaussian_pos;
			gaussian_sig = prior_in->gaussian_sig;
		}
		else if (prior==GAUSS2_PRIOR) {
			gauss_paramnums.input(prior_in->gauss_paramnums);
			gauss_meanvals.input(prior_in->gauss_meanvals);
			inv_covariance_matrix.input(prior_in->inv_covariance_matrix);
		}
	}
	void set_uniform() { prior = UNIFORM_PRIOR; }
	void set_log() { prior = LOG_PRIOR; }
	void set_gaussian(double &pos_in, double &sig_in) { prior = GAUSS_PRIOR; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void set_gauss2(int p1, int p2, double &pos1_in, double &pos2_in, double &sig1_in, double &sig2_in, double &sig12_in) {
		prior = GAUSS2_PRIOR;
		gauss_paramnums.input(2);
		gauss_meanvals.input(2);
		covariance_matrix.input(2,2);
		gauss_paramnums[0] = p1;
		gauss_paramnums[1] = p2;
		gauss_meanvals[0] = pos1_in;
		gauss_meanvals[1] = pos2_in;
		covariance_matrix[0][0] = SQR(sig1_in);
		covariance_matrix[1][1] = SQR(sig2_in);
		covariance_matrix[0][1] = SQR(sig12_in);
		covariance_matrix[1][0] = covariance_matrix[0][1];
		inv_covariance_matrix.input(2,2);
		inv_covariance_matrix = covariance_matrix.inverse();
	}
	void set_gauss2_secondary(int p1, int p2) {
		prior = GAUSS2_PRIOR_SECONDARY;
		gauss_paramnums.input(2);
		gauss_paramnums[0] = p1;
		gauss_paramnums[1] = p2;
	}
};

struct ParamTransform
{
	double gaussian_pos, gaussian_sig;
	double a, b; // for linear transformations
	bool include_jacobian;
	int ratio_paramnum;
	Transform transform;
	ParamTransform() { transform = NONE; include_jacobian = false; }
	ParamTransform(ParamTransform *transform_in)
	{
		transform = transform_in->transform;
		include_jacobian = transform_in->include_jacobian;
		if (transform==GAUSS_TRANSFORM) {
			gaussian_pos = transform_in->gaussian_pos;
			gaussian_sig = transform_in->gaussian_sig;
		} else if (transform==LINEAR_TRANSFORM) {
			a = transform_in->a;
			b = transform_in->b;
		} else if (transform==RATIO) {
			ratio_paramnum = transform_in->ratio_paramnum;
		}
	}
	void set_none() { transform = NONE; }
	void set_log() { transform = LOG_TRANSFORM; }
	void set_linear(double &a_in, double &b_in) { transform = LINEAR_TRANSFORM; a = a_in; b = b_in; }
	void set_gaussian(double &pos_in, double &sig_in) { transform = GAUSS_TRANSFORM; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void set_ratio(int &paramnum_in) { transform = RATIO; ratio_paramnum = paramnum_in; }
	void set_include_jacobian(bool &include) { include_jacobian = include; }
};

struct DerivedParam
{
	DerivedParamType derived_param_type;
	double funcparam; // if funcparam == -1, then there is no parameter required
	double funcparam2;
	bool use_kpc_units;
	int lensnum_param;
	string name, latex_name;
	DerivedParam(DerivedParamType type_in, double param, int lensnum, double param2 = -1, bool usekpc = false) // if lensnum == -1, then it uses *all* the lenses (if possible)
	{
		derived_param_type = type_in;
		funcparam = param;
		funcparam2 = param2;
		lensnum_param = lensnum;
		use_kpc_units = usekpc;
		if (derived_param_type == KappaR) {
			name = "kappa"; latex_name = "\\kappa"; if (lensnum==-1) { name += "_tot"; latex_name += "_{tot}"; }
		} else if (derived_param_type == LambdaR) { // here lambda_R = 1 - <kappa>(R)
			name = "lambdaR"; latex_name = "\\lambda_R";
		} else if (derived_param_type == DKappaR) {
			name = "dkappa"; latex_name = "\\kappa'"; if (lensnum==-1) { name += "_tot"; latex_name += "_{tot}"; }
		} else if (derived_param_type == Mass2dR) {
			name = "mass2d"; latex_name = "M_{2D}";
		} else if (derived_param_type == Mass3dR) {
			name = "mass3d"; latex_name = "M_{3D}";
		} else if (derived_param_type == Einstein) {
			name = "re_zsrc"; latex_name = "R_{e}";
		} else if (derived_param_type == Einstein_Mass) {
			name = "mass_re"; latex_name = "M_{Re}";
		} else if (derived_param_type == Kappa_Re) {
			name = "kappa_re"; latex_name = "\\kappa_{E}";
		} else if (derived_param_type == LensParam) {
			name = "lensparam"; latex_name = "\\lambda";
		} else if (derived_param_type == AvgLogSlope) {
			name = "logslope"; latex_name = "\\gamma_{avg}'";
		} else if (derived_param_type == Relative_Perturbation_Radius) {
			name = "r_perturb_rel"; latex_name = "\\Delta r_{\\delta c}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Perturbation_Radius) {
			name = "r_perturb"; latex_name = "r_{\\delta c}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Robust_Perturbation_Mass) {
			name = "mass_perturb"; latex_name = "M_{\\delta c}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Robust_Perturbation_Density) {
			name = "sigma_perturb"; latex_name = "\\Sigma_{\\delta c}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Chi_Square) {
			name = "raw_chisq"; latex_name = "\\chi^2";
			funcparam = -1e30; // no input parameter for this dparam
		} else die("no user defined function yet");

		if (funcparam != -1e30) {
			if (funcparam2==-1) {
				stringstream paramstr;
				string paramstring;
				paramstr << funcparam;
				paramstr >> paramstring;
				name += "(" + paramstring + ")";
				latex_name += "(" + paramstring + ")";
			} else {
				stringstream paramstr, paramstr2;
				string paramstring, paramstring2;
				paramstr << funcparam;
				paramstr >> paramstring;
				paramstr2 << funcparam2;
				paramstr2 >> paramstring2;
				name += "(" + paramstring + "," + paramstring2 + ")";
				latex_name += "(" + paramstring + "," + paramstring2 + ")";
			}
		}
	}
	double get_derived_param(QLens* lens_in)
	{
		if (derived_param_type == KappaR) return lens_in->total_kappa(funcparam,lensnum_param,use_kpc_units);
		else if (derived_param_type == LambdaR) return (1 - lens_in->total_dkappa(funcparam,-1,use_kpc_units));
		else if (derived_param_type == DKappaR) return lens_in->total_dkappa(funcparam,lensnum_param,use_kpc_units);
		else if (derived_param_type == Mass2dR) return lens_in->mass2d_r(funcparam,lensnum_param,use_kpc_units);
		else if (derived_param_type == Mass3dR) return lens_in->mass3d_r(funcparam,lensnum_param,use_kpc_units);
		else if (derived_param_type == Einstein) return lens_in->einstein_radius_single_lens(funcparam,lensnum_param);
		else if (derived_param_type == AvgLogSlope) return lens_in->calculate_average_log_slope(lensnum_param,funcparam,funcparam2,use_kpc_units);
		else if (derived_param_type == Einstein_Mass) {
			double re = lens_in->einstein_radius_single_lens(funcparam,lensnum_param);
			return lens_in->mass2d_r(re,lensnum_param,false);
		} else if (derived_param_type == Kappa_Re) {
			double reav=0;
			lens_in->einstein_radius_of_primary_lens(lens_in->reference_zfactors[lens_in->lens_redshift_idx[lens_in->primary_lens_number]],reav);
			if (reav <= 0) return 0.0;
			else return lens_in->total_kappa(reav,-1,false);
		} else if (derived_param_type == LensParam) {
			return lens_in->get_lens_parameter_using_default_pmode(funcparam,lensnum_param);
		}
		else if (derived_param_type == Relative_Perturbation_Radius) {
			double rmax,avgsig,menc,rmax_z,avgkap_scaled;
			lens_in->calculate_critical_curve_perturbation_radius_numerical(lensnum_param,false,rmax,avgsig,menc,rmax_z,avgkap_scaled,true);
			return rmax;
		} else if (derived_param_type == Perturbation_Radius) {
			double rmax,avgsig,menc,rmax_z,avgkap_scaled;
			lens_in->calculate_critical_curve_perturbation_radius_numerical(lensnum_param,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
			return rmax;
		} else if (derived_param_type == Robust_Perturbation_Mass) {
			double rmax,avgsig,menc,rmax_z,avgkap_scaled;
			lens_in->calculate_critical_curve_perturbation_radius_numerical(lensnum_param,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
			return menc;
		} else if (derived_param_type == Robust_Perturbation_Density) {
			double rmax,avgsig,menc,rmax_z,avgkap_scaled;
			lens_in->calculate_critical_curve_perturbation_radius_numerical(lensnum_param,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
			return avgsig;
		} else if (derived_param_type == Chi_Square) {
			double chisq_out;
			if (lens_in->raw_chisq==-1e30) {
				if (lens_in->lens_parent != NULL) {
					// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
					lens_in->lens_parent->LogLikeFunc(NULL); // If the chi-square has not already been evaluated, evaluate it here
					chisq_out = lens_in->raw_chisq;
					lens_in->clear_raw_chisq();
				} else {
					chisq_out = lens_in->chisq_single_evaluation(false,false);
				}
			}
			return chisq_out;
		}
		else die("no user defined function yet");
		return 0.0;
	}
	void print_param_description(QLens* lens_in)
	{
		string unitstring = (use_kpc_units) ? " kpc" : " arcsec";
		double dpar = get_derived_param(lens_in);
		//cout << name << ": ";
		if (derived_param_type == KappaR) {
			if (lensnum_param==-1) cout << "Total kappa within r = " << funcparam << unitstring << endl;
			else cout << "kappa for lens " << lensnum_param << " within r = " << funcparam << unitstring << endl;
		} else if (derived_param_type == LambdaR) {
			cout << "One minus average kappa at r = " << funcparam << unitstring << endl;
		} else if (derived_param_type == DKappaR) {
			if (lensnum_param==-1) cout << "Derivative of total kappa within r = " << funcparam << unitstring << endl;
			else cout << "Derivative of kappa for lens " << lensnum_param << " within r = " << funcparam << unitstring << endl;
		} else if (derived_param_type == Mass2dR) {
			cout << "Projected (2D) mass of lens " << lensnum_param << " enclosed within r = " << funcparam << unitstring << endl;
		} else if (derived_param_type == Mass3dR) {
			cout << "Deprojected (3D) mass of lens " << lensnum_param << " enclosed within r = " << funcparam << unitstring << endl;
		} else if (derived_param_type == Einstein) {
			cout << "Einstein radius of lens " << lensnum_param << " for source redshift zsrc = " << funcparam << endl;
		} else if (derived_param_type == Einstein_Mass) {
			cout << "Projected mass within Einstein radius of lens " << lensnum_param << " for source redshift zsrc = " << funcparam << endl;
		} else if (derived_param_type == Kappa_Re) {
			cout << "Kappa at Einstein radius of primary lens (plus other lenses that are co-centered with primary), averaged over all angles" << endl;
		} else if (derived_param_type == LensParam) {
			cout << "Parameter " << ((int) funcparam) << " of lens " << lensnum_param << " using default pmode=" << lens_in->default_parameter_mode << endl;
		} else if (derived_param_type == AvgLogSlope) {
			cout << "Average log-slope of kappa from lens " << lensnum_param << " between r1=" << funcparam << " and r2=" << funcparam2 << endl;
		} else if (derived_param_type == Perturbation_Radius) {
			cout << "Critical curve perturbation radius of lens " << lensnum_param << endl;
		} else if (derived_param_type == Relative_Perturbation_Radius) {
			cout << "Relative critical curve perturbation radius of lens " << lensnum_param << endl;
		} else if (derived_param_type == Robust_Perturbation_Mass) {
			cout << "Projected mass within perturbation radius of lens " << lensnum_param << endl;
		} else if (derived_param_type == Robust_Perturbation_Density) {
			cout << "Average projected density within perturbation radius of lens " << lensnum_param << endl;
		} else if (derived_param_type == Chi_Square) {
			cout << "Raw chi-square value for given set of parameters" << endl;
		} else die("no user defined function yet");
		cout << "   name: '" << name << "', latex_name: '" << latex_name << "'" << endl;
		cout << "   " << name << " = " << dpar << endl;
	}
	void rename(const string new_name, const string new_latex_name)
	{
		name = new_name;
		latex_name = new_latex_name;
	}
};

struct ParamSettings
{
	int nparams;
	ParamPrior **priors;
	ParamTransform **transforms;
	string *param_names;
	string *override_names; // this allows to manually set names even after parameter transformations
	// ParamSettings should handle the latex names too, to simplify things; this would also allow for manual override of the latex names. Implement this!!!!!!
	double *prior_norms;
	double *penalty_limits_lo, *penalty_limits_hi;
	bool *use_penalty_limits;
	// It would be nice if penalty limits and override_limits could be merged. The tricky part is that the penalty limits deal with the	
	// untransformed parameters, while override_limits deal with the transformed parameters. Not sure yet what is the best way to handle this.
	double *override_limits_lo, *override_limits_hi;
	bool *override_prior_limits;
	double *stepsizes;
	bool *auto_stepsize;
	bool *hist2d_param;
	bool *hist2d_dparam;
	bool *subplot_param;
	bool *subplot_dparam;
	string *dparam_names;
	int n_dparams;
	ParamSettings() { priors = NULL; param_names = NULL; transforms = NULL; nparams = 0; stepsizes = NULL; auto_stepsize = NULL; hist2d_param = NULL; hist2d_dparam = NULL; subplot_param = NULL; dparam_names = NULL; subplot_dparam = NULL; nparams = 0; n_dparams = 0; }
	ParamSettings(ParamSettings& param_settings_in) {
		nparams = param_settings_in.nparams;
		n_dparams = param_settings_in.n_dparams;
		param_names = new string[nparams];
		override_names = new string[nparams];
		priors = new ParamPrior*[nparams];
		transforms = new ParamTransform*[nparams];
		stepsizes = new double[nparams];
		auto_stepsize = new bool[nparams];
		hist2d_param = new bool[nparams];
		subplot_param = new bool[nparams];
		prior_norms = new double[nparams];
		penalty_limits_lo = new double[nparams];
		penalty_limits_hi = new double[nparams];
		use_penalty_limits = new bool[nparams];
		override_limits_lo = new double[nparams];
		override_limits_hi = new double[nparams];
		override_prior_limits = new bool[nparams];
		for (int i=0; i < nparams; i++) {
			priors[i] = new ParamPrior(param_settings_in.priors[i]);
			transforms[i] = new ParamTransform(param_settings_in.transforms[i]);
			param_names[i] = param_settings_in.param_names[i];
			override_names[i] = param_settings_in.override_names[i];
			stepsizes[i] = param_settings_in.stepsizes[i];
			auto_stepsize[i] = param_settings_in.auto_stepsize[i];
			hist2d_param[i] = param_settings_in.hist2d_param[i];
			subplot_param[i] = param_settings_in.subplot_param[i];
			prior_norms[i] = param_settings_in.prior_norms[i];
			penalty_limits_lo[i] = param_settings_in.penalty_limits_lo[i];
			penalty_limits_hi[i] = param_settings_in.penalty_limits_hi[i];
			use_penalty_limits[i] = param_settings_in.use_penalty_limits[i];
			override_limits_lo[i] = param_settings_in.override_limits_lo[i];
			override_limits_hi[i] = param_settings_in.override_limits_hi[i];
			override_prior_limits[i] = param_settings_in.override_prior_limits[i];
		}
		if (n_dparams > 0) {
			dparam_names = new string[n_dparams];
			hist2d_dparam = new bool[n_dparams];
			subplot_dparam = new bool[n_dparams];
			for (int i=0; i < n_dparams; i++) {
				dparam_names[i] = param_settings_in.dparam_names[i];
				hist2d_dparam[i] = param_settings_in.hist2d_dparam[i];
				subplot_dparam[i] = param_settings_in.subplot_dparam[i];
			}
		}
	}
	void update_params(const int nparams_in, vector<string>& names, double* stepsizes_in);
	void insert_params(const int pi, const int pf, vector<string>& names, double* stepsizes_in);
	bool remove_params(const int pi, const int pf);
	void add_dparam(string dparam_name);
	void remove_dparam(int dparam_number);
	void rename_dparam(int dparam_number, string newname) { dparam_names[dparam_number] = newname; }
	void clear_dparams()
	{
		if (n_dparams > 0) {
			delete[] dparam_names;
			delete[] hist2d_dparam;
			delete[] subplot_dparam;
			n_dparams = 0;
		}
	}
	int lookup_param_number(const string pname)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		int pnum = -1;
		for (int i=0; i < nparams; i++) {
			if ((transformed_names[i]==pname) or (param_names[i]==pname)) { pnum = i; break; }
		}
		for (int i=0; i < n_dparams; i++) {
			if (dparam_names[i]==pname) pnum = nparams+i;
		}
		delete[] transformed_names;
		return pnum;
	}
	string lookup_param_name(const int i)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		string name = transformed_names[i];
		delete[] transformed_names;
		return name;
	}
	bool exclude_hist2d_param(const string pname)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		bool found_name = false;
		int i;
		for (i=0; i < nparams; i++) {
			if ((param_names[i]==pname) or (transformed_names[i]==pname)) {
				hist2d_param[i] = false;
				found_name = true;
				break;
			}
		}
		if (!found_name) {
			for (i=0; i < n_dparams; i++) {
				if (dparam_names[i]==pname) {
					hist2d_dparam[i] = false;
					found_name = true;
					break;
				}
			}
		}
		delete[] transformed_names;
		return found_name;
	}
	bool hist2d_params_defined()
	{
		bool active_param = false;
		int i;
		for (i=0; i < nparams; i++) {
			if (!hist2d_param[i]) {
				active_param = true;
				break;
			}
		}
		if (!active_param) {
			for (i=0; i < n_dparams; i++) {
				if (!hist2d_dparam[i]) {
					active_param = true;
					break;
				}
			}
		}
		return active_param;
	}
	bool hist2d_param_flag(const int i, string &name)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		bool flag;
		if (i < nparams) {
			name = transformed_names[i];
			flag = hist2d_param[i];
		} else {
			int j = i - nparams;
			name = dparam_names[j];
			flag = hist2d_dparam[j];
		}
		delete[] transformed_names;
		return flag;
	}
	string print_excluded_hist2d_params()
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		string pstring = "";
		int i;
		for (i=0; i < nparams; i++) {
			if (!hist2d_param[i]) pstring += transformed_names[i] + " ";
		}
		for (i=0; i < n_dparams; i++) {
			if (!hist2d_dparam[i]) pstring += dparam_names[i] + " ";
		}
		delete[] transformed_names;
		return pstring;
	}
	void reset_hist2d_params()
	{
		int i;
		for (i=0; i < nparams; i++) hist2d_param[i] = true;
		for (i=0; i < n_dparams; i++) hist2d_dparam[i] = true;
	}
	bool set_subplot_param(const string pname)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		bool found_name = false;
		int i;
		for (i=0; i < nparams; i++) {
			if ((param_names[i]==pname) or (transformed_names[i]==pname)) {
				subplot_param[i] = true;
				found_name = true;
				break;
			}
		}
		if (!found_name) {
			for (i=0; i < n_dparams; i++) {
				if (dparam_names[i]==pname) {
					subplot_dparam[i] = true;
					found_name = true;
					break;
				}
			}
		}
		delete[] transformed_names;
		return found_name;
	}
	bool subplot_params_defined()
	{
		bool active_param = false;
		int i;
		for (i=0; i < nparams; i++) {
			if (subplot_param[i]) {
				active_param = true;
				break;
			}
		}
		if (!active_param) {
			for (i=0; i < n_dparams; i++) {
				if (subplot_dparam[i]) {
					active_param = true;
					break;
				}
			}
		}
		return active_param;
	}
	bool subplot_param_flag(const int i, string &name)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		bool flag;
		if (i < nparams) {
			name = transformed_names[i];
			flag = subplot_param[i];
		} else {
			int j = i - nparams;
			name = dparam_names[j];
			flag = subplot_dparam[j];
		}
		delete[] transformed_names;
		return flag;
	}
	string print_subplot_params()
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		string pstring = "";
		int i;
		for (i=0; i < nparams; i++) {
			if (subplot_param[i]) pstring += transformed_names[i] + " ";
		}
		for (i=0; i < n_dparams; i++) {
			if (subplot_dparam[i]) pstring += dparam_names[i] + " ";
		}
		delete[] transformed_names;
		return pstring;
	}
	void reset_subplot_params()
	{
		int i;
		for (i=0; i < nparams; i++) subplot_param[i] = false;
		for (i=0; i < n_dparams; i++) subplot_dparam[i] = false;
	}
	void clear_penalty_limits()
	{
		for (int i=0; i < nparams; i++) {
			use_penalty_limits[i] = false;
		}
	}
	void print_priors();
	bool output_prior(const int i);
	void print_stepsizes();
	void print_penalty_limits();
	void scale_stepsizes(const double fac)
	{
		for (int i=0; i < nparams; i++) {
			stepsizes[i] *= fac;
			auto_stepsize[i] = false;
		}
	}
	void reset_stepsizes(double *stepsizes_in)
	{
		for (int i=0; i < nparams; i++) {
			stepsizes[i] = stepsizes_in[i];
			auto_stepsize[i] = true;
		}
		transform_stepsizes();
	}
	void set_stepsize(const int i, const double step)
	{
		if (i >= nparams) die("parameter chosen for stepsize is greater than total number of parameters (%i vs %i)",i,nparams);
		auto_stepsize[i] = false;
		stepsizes[i] = step;
	}
	void set_penalty_limit(const int i, const double lo, const double hi)
	{
		if (i >= nparams) die("parameter chosen for penalty limit is greater than total number of parameters (%i vs %i)",i,nparams);
		use_penalty_limits[i] = true;
		penalty_limits_lo[i] = lo;
		penalty_limits_hi[i] = hi;
	}
	void get_penalty_limits(boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		use_plimits.input(nparams);
		lower.input(nparams);
		upper.input(nparams);
		for (int i=0; i < nparams; i++) {
			use_plimits[i] = use_penalty_limits[i];
			lower[i] = penalty_limits_lo[i];
			upper[i] = penalty_limits_hi[i];
		}
	}
	void update_penalty_limits(boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		for (int i=0; i < nparams; i++) {
			use_penalty_limits[i] = use_plimits[i];
			penalty_limits_lo[i] = lower[i];
			penalty_limits_hi[i] = upper[i];
		}
	}
	void update_specific_penalty_limits(const int pi, const int pf, boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		int i, index;
		for (i=0, index=pi; index < pf; i++, index++) {
			use_penalty_limits[index] = use_plimits[i];
			penalty_limits_lo[index] = lower[i];
			penalty_limits_hi[index] = upper[i];
		}
	}
	void clear_penalty_limit(const int i)
	{
		if (i >= nparams) die("parameter chosen for penalty limit is greater than total number of parameters (%i vs %i)",i,nparams);
		use_penalty_limits[i] = true; // this ensures that it won't be overwritten by default values
		penalty_limits_lo[i] = -1e30;
		penalty_limits_hi[i] = 1e30;
	}
	void transform_parameters(double *params)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->transform==LOG_TRANSFORM) params[i] = log(params[i])/M_LN10;
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				params[i] = erff((params[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				params[i] = transforms[i]->a * params[i] + transforms[i]->b;
			} else if (transforms[i]->transform==RATIO) {
				params[i] = params[i]/params[transforms[i]->ratio_paramnum];
			}
		}
	}
	void transform_limits(double *lower, double *upper)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->transform==LOG_TRANSFORM) lower[i] = log(lower[i])/M_LN10;
			if (transforms[i]->transform==LOG_TRANSFORM) upper[i] = log(upper[i])/M_LN10;
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				lower[i] = erff((lower[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
				upper[i] = erff((upper[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				lower[i] = transforms[i]->a * lower[i] + transforms[i]->b;
				upper[i] = transforms[i]->a * upper[i] + transforms[i]->b;
				if (lower[i] > upper[i]) {
					double temp = lower[i]; lower[i] = upper[i]; upper[i] = temp;
				}
			} else if (transforms[i]->transform==RATIO) {
				lower[i] = 0; // these can be manually adjusted using 'fit priors range ...'
				upper[i] = 1; // these can be customized
			}
		}
	}
	void set_override_prior_limit(const int i, const double lo, const double hi)
	{
		if (i >= nparams) die("parameter chosen for prior limit is greater than total number of parameters (%i vs %i)",i,nparams);
		override_prior_limits[i] = true;
		override_limits_lo[i] = lo;
		override_limits_hi[i] = hi;
	}
	void override_limits(double *lower, double *upper)
	{
		for (int i=0; i < nparams; i++) {
			if (override_prior_limits[i]) {
				lower[i] = override_limits_lo[i];
				upper[i] = override_limits_hi[i];
			}
		}
	}
	void inverse_transform_parameters(double *params, double *transformed_params)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->transform==NONE) transformed_params[i] = params[i];
			else if (transforms[i]->transform==LOG_TRANSFORM) transformed_params[i] = pow(10.0,params[i]);
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				transformed_params[i] = transforms[i]->gaussian_pos + M_SQRT2*transforms[i]->gaussian_sig*erfinv(params[i]);
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				transformed_params[i] = (params[i] - transforms[i]->b) / transforms[i]->a;
			} else if (transforms[i]->transform==RATIO) {
				transformed_params[i] = params[i]*params[transforms[i]->ratio_paramnum];
			}
		}
	}
	void inverse_transform_parameters(double *params)
	{
		inverse_transform_parameters(params,params);
	}
	void transform_parameter_names(string *names, string *transformed_names, string *latex_names, string *transformed_latex_names)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->transform==NONE) {
				transformed_names[i] = names[i];
				if (latex_names != NULL) transformed_latex_names[i] = latex_names[i];
			}
			else if (transforms[i]->transform==LOG_TRANSFORM) {
				transformed_names[i] = "log(" + names[i] + ")";
				if (latex_names != NULL) transformed_latex_names[i] = "\\log(" + latex_names[i] + ")";
			}
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				transformed_names[i] = "u{" + names[i] + "}";
				if (latex_names != NULL) transformed_latex_names[i] = "u\\{" + latex_names[i] + "\\}";
			}
			else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				transformed_names[i] = "L{" + names[i] + "}";
				if (latex_names != NULL) transformed_latex_names[i] = "L\\{" + latex_names[i] + "\\}";
			}
			else if (transforms[i]->transform==RATIO) {
				transformed_names[i] = names[i] + "_over_" + names[transforms[i]->ratio_paramnum];
				if (latex_names != NULL) transformed_latex_names[i] = latex_names[i] + "/" + latex_names[transforms[i]->ratio_paramnum];
			}
		}
		override_parameter_names(transformed_names); // allows for manually setting parameter names
	}
	bool set_override_parameter_name(const int i, const string name)
	{
		bool unique_name = true;
		for (int j=0; j < nparams; j++) {
			if ((i != j) and (((override_names[j] != "") and (override_names[j]==name)) or (param_names[j]==name))) unique_name = false;
		}
		if (!unique_name) return false;
		override_names[i] = name;
		return true;
	}
	void override_parameter_names(string* names)
	{
		for (int i=0; i < nparams; i++) {
			if (override_names[i] != "") names[i] = override_names[i];
		}
	}
	void transform_stepsizes()
	{
		// It would be better to have it pass in the current value of the parameters, then use the default
		// (untransformed) stepsize to define the transformed stepsize. For example, the log stepsize would
		// be log((pi+step)/pi). But passing in the parameter values is a bit of a pain...do this later
		for (int i=0; i < nparams; i++) {
			if (auto_stepsize[i]) {
				if (transforms[i]->transform==LOG_TRANSFORM) {
					stepsizes[i] = 0.5; // default for a log transform
				}
				else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				}
				else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				}
				else if (transforms[i]->transform==RATIO) {
					stepsizes[i] = 0.5; // default for a ratio
				}
			}
		}
	}
	void add_prior_terms_to_loglike(double *params, double& loglike)
	{
		for (int i=0; i < nparams; i++) {
			if (priors[i]->prior!=UNIFORM_PRIOR) {
				loglike += log(prior_norms[i]); // Normalize the prior for the bayesian evidence
				if (priors[i]->prior==LOG_PRIOR) loglike += log(params[i]);
				else if (priors[i]->prior==GAUSS_PRIOR) loglike += SQR((params[i] - priors[i]->gaussian_pos)/priors[i]->gaussian_sig)/2.0;
				else if (priors[i]->prior==GAUSS2_PRIOR) {
					int j = priors[i]->gauss_paramnums[1];
					dvector bvec, cvec;
					bvec.input(2);
					cvec.input(2);
					bvec[0] = params[i] - priors[i]->gauss_meanvals[0];
					bvec[1] = params[j] - priors[i]->gauss_meanvals[1];
					cvec = priors[i]->inv_covariance_matrix * bvec;
					loglike += (bvec[0]*cvec[0] + bvec[1]*cvec[1]) / 2.0;
				}
			}
		}
	}
	void update_reference_paramnums(int *new_paramnums)
	{
		// This updates any parameter numbers that are referenced by the priors or transforms; this is done any time the parameter list is changed
		int new_paramnum;
		for (int i=0; i < nparams; i++) {
			if (priors[i]->prior==GAUSS2_PRIOR) {
				new_paramnum = new_paramnums[priors[i]->gauss_paramnums[0]];
				if (new_paramnum==-1) {
					// parameter no longer exists; revert back to uniform prior
					priors[i]->set_uniform();
				} else {
					priors[i]->gauss_paramnums[0] = new_paramnum;
					priors[i]->gauss_paramnums[1] = new_paramnum;
				}
			}
			if (transforms[i]->transform==RATIO) {
				new_paramnum = new_paramnums[transforms[i]->ratio_paramnum];
				if (new_paramnum==-1) {
					// parameter no longer exists; remove transformation
					transforms[i]->set_none();
				} else {
					transforms[i]->ratio_paramnum = new_paramnum;
				}
			}
		}
	}
	void set_prior_norms(double *lower_limit, double* upper_limit)
	{
		// flat priors are automatically given a norm of 1.0, since we'll be transforming to the unit hypercube when doing nested sampling;
		// however a correction is required for other priors
		for (int i=0; i < nparams; i++) {
			if (priors[i]->prior!=UNIFORM_PRIOR) {
				if (priors[i]->prior==LOG_PRIOR) prior_norms[i] = log(upper_limit[i]/lower_limit[i]);
				else if (priors[i]->prior==GAUSS_PRIOR) {
					prior_norms[i] = (erff((upper_limit[i] - priors[i]->gaussian_pos)/(M_SQRT2*priors[i]->gaussian_sig)) - erff((lower_limit[i] - priors[i]->gaussian_pos)/(M_SQRT2*priors[i]->gaussian_sig))) * M_SQRT_HALFPI * priors[i]->gaussian_sig;
				}
				prior_norms[i] /= (upper_limit[i] - lower_limit[i]); // correction since we are transforming to the unit hypercube
			}
		}
	}
	void add_jacobian_terms_to_loglike(double *params, double& loglike)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->include_jacobian==true) {
				if (transforms[i]->transform==LOG_TRANSFORM) loglike -= log(params[i]);
				else if (transforms[i]->transform==GAUSS_TRANSFORM) loglike -= SQR((params[i] - transforms[i]->gaussian_pos)/transforms[i]->gaussian_sig)/2.0;
				else if (transforms[i]->transform==RATIO) loglike += log(params[transforms[i]->ratio_paramnum]);
			}
		}
	}
	void clear_params()
	{
		if (nparams > 0) {
			delete[] param_names;
			delete[] override_names;
			for (int i=0; i < nparams; i++) {
				delete priors[i];
				delete transforms[i];
			}
			delete[] priors;
			delete[] transforms;
			delete[] stepsizes;
			delete[] auto_stepsize;
			delete[] subplot_param;
			delete[] hist2d_param;
			delete[] prior_norms;
			delete[] penalty_limits_lo;
			delete[] penalty_limits_hi;
			delete[] use_penalty_limits;
			delete[] override_limits_lo;
			delete[] override_limits_hi;
			delete[] override_prior_limits;

		}
		priors = NULL;
		param_names = NULL;
		override_names = NULL;
		transforms = NULL;
		nparams = 0;
		stepsizes = NULL;
		auto_stepsize = NULL;
		subplot_param = NULL;
	}
	~ParamSettings()
	{
		if (nparams > 0) {
			delete[] param_names;
			delete[] override_names;
			for (int i=0; i < nparams; i++) {
				delete priors[i];
				delete transforms[i];
			}
			delete[] priors;
			delete[] transforms;
			delete[] stepsizes;
			delete[] auto_stepsize;
			delete[] subplot_param;
			delete[] hist2d_param;
			delete[] prior_norms;
			delete[] penalty_limits_lo;
			delete[] penalty_limits_hi;
			delete[] use_penalty_limits;
			delete[] override_limits_lo;
			delete[] override_limits_hi;
			delete[] override_prior_limits;
		}
		if (n_dparams > 0) {
			delete[] dparam_names;
			delete[] subplot_dparam;
		}
	}
};

class Defspline
{
	Spline2D ax, ay;
	Spline2D axx, ayy, axy;

	public:
	friend void QLens::spline_deflection(double,double,int);
	int nsteps() { return (ax.xlength()-1); }
	double xmax() { return ax.xmax(); }
	double ymax() { return ax.ymax(); }

	lensvector deflection(const double &x, const double &y)
	{
		lensvector ans;
		ans[0] = ax.splint(x,y);
		ans[1] = ay.splint(x,y);
		return ans;
	}

	lensmatrix hessian(const double &x, const double &y)
	{
		lensmatrix ans;
		ans[0][0] = axx.splint(x,y);
		ans[1][1] = ayy.splint(x,y);
		ans[1][0] = axy.splint(x,y);
		ans[0][1] = ans[1][0];
		return ans;
	}
};

inline double QLens::kappa(const double& x, const double& y, double* zfacs, double** betafacs)
{
	double kappa;
	if (n_lens_redshifts==1) {
		int j;
		kappa=0;
		for (j=0; j < nlens; j++) {
			kappa += lens_list[j]->kappa(x,y);
		}
		kappa *= zfacs[0];
	} else {
		lensmatrix *jac = &jacs[0];
		hessian(x,y,(*jac),0,zfacs,betafacs);
		kappa = ((*jac)[0][0] + (*jac)[1][1])/2;
	}

	return kappa;
}

/*
inline double QLens::kappa_weak(const double& x, const double& y, double* zfacs)
{
	double kappa;
	int j;
	kappa=0;
	for (j=0; j < nlens; j++) {
		kappa += lens_list[j]->kappa(x,y);
	}
	kappa *= zfacs[0];

	return kappa;
}
*/

inline double QLens::potential(const double& x, const double& y, double* zfacs, double** betafacs)
{
	double pot=0, pot_subtot;
	// This is not really sensical for multiplane lensing, and time delays need to be treated as in Schneider's textbook. Fix later
	int i,j;
	for (i=0; i < n_lens_redshifts; i++) {
		pot_subtot=0;
		for (j=0; j < zlens_group_size[i]; j++) {
			pot_subtot += lens_list[zlens_group_lens_indx[i][j]]->potential(x,y);
		}
		pot += zfacs[i]*pot_subtot;
	}
	return pot;
}

inline void QLens::deflection(const double& x, const double& y, lensvector& def_tot, const int &thread, double* zfacs, double** betafacs)
{
	if (!defspline)
	{
		lensvector *x_i = &xvals_i[thread];
		lensvector *def = &defs_i[thread];
		lensvector **def_i = &defs_subtot[thread];

		int i,j;
		def_tot[0] = 0;
		def_tot[1] = 0;
		//cout << "n_redshifts=" << n_lens_redshifts << endl;
		for (i=0; i < n_lens_redshifts; i++) {
			//cout << "redshift " << i << ":\n";
			(*def_i)[i][0] = 0;
			(*def_i)[i][1] = 0;
			(*x_i)[0] = x;
			(*x_i)[1] = y;
			for (j=0; j < i; j++) {
				//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
				(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
				(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
			}
			for (j=0; j < zlens_group_size[i]; j++) {
				lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
				//cout << "Lens redshift " << i << ", lens " << zlens_group_lens_indx[i][j] << " def=" << (*def)[0] << " " << (*def)[1] << endl;
				(*def_i)[i][0] += (*def)[0];
				(*def_i)[i][1] += (*def)[1];
			}
			//cout << "Lens redshift" << i << " (z=" << lens_redshifts[i] << "): xi=" << (*x_i)[0] << " " << (*x_i)[1] << endl;
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot[0] += (*def_i)[i][0];
			def_tot[1] += (*def_i)[i][1];
		}
	}
	else {
		def_tot = defspline->deflection(x,y);
	}
}

inline void QLens::custom_deflection(const double& x, const double& y, lensvector& def_tot)
{
	lensvector def;
	lensvector pos, pos_prime;

	def_tot[0] = 0;
	def_tot[1] = 0;
	pos[0] = x;
	pos[1] = y;
	double zlens1 = lens_list[0]->zlens;
	double zlens2 = lens_list[2]->zlens;
	double beta;
	if (zlens1 > zlens2) {
		beta = calculate_beta_factor(zlens2,zlens1,1);
		//cout << "BETA! " << beta << endl;
		lens_list[2]->deflection(pos[0],pos[1],def);
		def_tot[0] += def[0];
		def_tot[1] += def[1];
		pos_prime[0] = pos[0] - beta*def_tot[0];
		pos_prime[1] = pos[1] - beta*def_tot[1];
		//cout << "Lens 0,1 xprime=" << pos_prime[0] << " " << pos_prime[1] << endl;
		lens_list[0]->deflection(pos_prime[0],pos_prime[1],def);
		def_tot[0] += def[0];
		def_tot[1] += def[1];
		lens_list[1]->deflection(pos_prime[0],pos_prime[1],def);
		def_tot[0] += def[0];
		def_tot[1] += def[1];
	} else if (zlens1 < zlens2) {
		beta = calculate_beta_factor(zlens1,zlens2,1);
		//cout << "BETA! " << beta << endl;
		lens_list[0]->deflection(pos[0],pos[1],def);
		//cout << "lens 0 def=" << def[0] << " " << def[1] << endl;
		def_tot[0] += def[0];
		def_tot[1] += def[1];
		lens_list[1]->deflection(pos[0],pos[1],def);
		//cout << "lens 1 def=" << def[0] << " " << def[1] << endl;
		def_tot[0] += def[0];
		def_tot[1] += def[1];
		pos_prime[0] = pos[0] - beta*def_tot[0];
		pos_prime[1] = pos[1] - beta*def_tot[1];
		//cout << "Lens 2 xprime=" << pos_prime[0] << " " << pos_prime[1] << endl;
		lens_list[2]->deflection(pos_prime[0],pos_prime[1],def);
		//cout << "lens 2 def=" << def[0] << " " << def[1] << endl;
		def_tot[0] += def[0];
		def_tot[1] += def[1];
	} else {
		lens_list[0]->deflection(pos[0],pos[1],def);
		def_tot[0] += def[0];
		def_tot[1] += def[1];
		lens_list[1]->deflection(pos[0],pos[1],def);
		def_tot[0] += def[0];
		def_tot[1] += def[1];
		lens_list[2]->deflection(pos[0],pos[1],def);
		def_tot[0] += def[0];
		def_tot[1] += def[1];
	}
}

inline void QLens::deflection(const double& x, const double& y, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs)
{
	if (!defspline)
	{
		lensvector *x_i = &xvals_i[thread];
		lensvector *def = &defs_i[thread];
		lensvector **def_i = &defs_subtot[thread];
		int i,j;
		def_tot_x = 0;
		def_tot_y = 0;
		//cout << "n_redshifts=" << n_lens_redshifts << endl;
		for (i=0; i < n_lens_redshifts; i++) {
			//cout << "redshift " << i << ":\n";
			(*def_i)[i][0] = 0;
			(*def_i)[i][1] = 0;
			(*x_i)[0] = x;
			(*x_i)[1] = y;
			for (j=0; j < i; j++) {
				//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
				(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
				(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
			}
			for (j=0; j < zlens_group_size[i]; j++) {
				lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
				(*def_i)[i][0] += (*def)[0];
				(*def_i)[i][1] += (*def)[1];
			}
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot_x += (*def_i)[i][0];
			def_tot_y += (*def_i)[i][1];
		}
	}
	else {
		lensvector *def = &defs_i[thread];
		(*def) = defspline->deflection(x,y);
		def_tot_x = (*def)[0];
		def_tot_y = (*def)[1];
	}
}

inline void QLens::deflection_exclude(const double& x, const double& y, bool* exclude, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs)
{
	bool skip_lens_plane = false;
	int skip_i = -1;
	lensvector *x_i = &xvals_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];
	int i,j;
	def_tot_x = 0;
	def_tot_y = 0;

	for (i=0; i < n_lens_redshifts; i++) {
		if ((zlens_group_size[i]==1) and (exclude[zlens_group_lens_indx[i][0]])) {
			skip_lens_plane = true;
			skip_i = i;
			// should allow for multiple redshifts to be excluded...fix later
		}
	}

	//cout << "n_redshifts=" << n_lens_redshifts << endl;
	for (i=0; i < n_lens_redshifts; i++) {
		//cout << "redshift " << i << ":\n";
		if ((!skip_lens_plane) or (skip_i != i)) {
			(*def_i)[i][0] = 0;
			(*def_i)[i][1] = 0;
			(*x_i)[0] = x;
			(*x_i)[1] = y;
			for (j=0; j < i; j++) {
				//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
				if ((!skip_lens_plane) or (skip_i != j)) {
					(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
					(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
				}
			}
			for (j=0; j < zlens_group_size[i]; j++) {
				if (exclude[zlens_group_lens_indx[i][j]]) ;
				else {
					lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
			}
			(*def_i)[i][0] *= zfacs[i];
			(*def_i)[i][1] *= zfacs[i];
			def_tot_x += (*def_i)[i][0];
			def_tot_y += (*def_i)[i][1];
		}
	}
}

inline void QLens::lens_equation(const lensvector& x, lensvector& f, const int& thread, double *zfacs, double** betafacs)
{
	deflection(x[0],x[1],f,thread,zfacs,betafacs);
	f[0] = source[0] - x[0] + f[0]; // finding root of lens equation, i.e. f(x) = beta - theta + alpha = 0   (where alpha is the deflection)
	f[1] = source[1] - x[1] + f[1];
}

inline void QLens::map_to_lens_plane(const int& redshift_i, const double& x, const double& y, lensvector& xi, const int &thread, double* zfacs, double** betafacs)
{
	if (redshift_i >= n_lens_redshifts) die("lens redshift index does not exist");
	lensvector *x_i = &xvals_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];

	int i,j;
	//cout << "n_redshifts=" << n_lens_redshifts << endl;
	for (i=0; i <= redshift_i; i++) {
		//cout << "redshift " << i << ":\n";
		(*def_i)[i][0] = 0;
		(*def_i)[i][1] = 0;
		(*x_i)[0] = x;
		(*x_i)[1] = y;
		for (j=0; j < i; j++) {
			//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
			(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
			(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
		}
		if (i==redshift_i) break;
		for (j=0; j < zlens_group_size[i]; j++) {
			lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
			(*def_i)[i][0] += (*def)[0];
			(*def_i)[i][1] += (*def)[1];
		}
		(*def_i)[i][0] *= zfacs[i];
		(*def_i)[i][1] *= zfacs[i];
	}
	xi[0] = (*x_i)[0];
	xi[1] = (*x_i)[1];
}

inline void QLens::hessian(const double& x, const double& y, lensmatrix& hess_tot, const int &thread, double* zfacs, double** betafacs) // calculates the Hessian of the lensing potential
{
	if (!defspline)
	{
		if (n_lens_redshifts > 1) {
			lensvector *x_i = &xvals_i[thread];
			lensmatrix *A_i = &Amats_i[thread];
			lensvector *def = &defs_i[thread];
			lensvector **def_i = &defs_subtot[thread];
			lensmatrix *hess = &hesses_i[thread];
			lensmatrix **hess_i = &hesses_subtot[thread];

			int i,j;
			hess_tot[0][0] = 0;
			hess_tot[1][1] = 0;
			hess_tot[0][1] = 0;
			hess_tot[1][0] = 0;
			for (i=0; i < n_lens_redshifts; i++) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
					(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
					(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
					(*A_i)[0][0] -= (betafacs[i-1][j])*((*hess_i)[j])[0][0];
					(*A_i)[1][1] -= (betafacs[i-1][j])*((*hess_i)[j])[1][1];
					(*A_i)[1][0] -= (betafacs[i-1][j])*((*hess_i)[j])[1][0];
					(*A_i)[0][1] -= (betafacs[i-1][j])*((*hess_i)[j])[0][1];
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					lens_list[zlens_group_lens_indx[i][j]]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess));
					(*hess_i)[i][0][0] += (*hess)[0][0];
					(*hess_i)[i][1][1] += (*hess)[1][1];
					(*hess_i)[i][0][1] += (*hess)[0][1];
					(*hess_i)[i][1][0] += (*hess)[1][0];
					if (i < n_lens_redshifts-1) {
						(*def_i)[i][0] += (*def)[0];
						(*def_i)[i][1] += (*def)[1];
					}
				}
				if (i < n_lens_redshifts-1) {
					(*def_i)[i][0] *= zfacs[i];
					(*def_i)[i][1] *= zfacs[i];
				}
				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];

				hess_tot[0][0] += (*hess_i)[i][0][0];
				hess_tot[1][1] += (*hess_i)[i][1][1];
				hess_tot[1][0] += (*hess_i)[i][1][0];
				hess_tot[0][1] += (*hess_i)[i][0][1];
			}
		} else {
			lensmatrix *hess = &hesses_i[thread];
			int j;
			hess_tot[0][0] = 0;
			hess_tot[1][1] = 0;
			hess_tot[0][1] = 0;
			hess_tot[1][0] = 0;
			for (j=0; j < nlens; j++) {
				lens_list[j]->hessian(x,y,(*hess));
				hess_tot[0][0] += (*hess)[0][0];
				hess_tot[1][1] += (*hess)[1][1];
				hess_tot[0][1] += (*hess)[0][1];
				hess_tot[1][0] += (*hess)[1][0];
			}
			hess_tot[0][0] *= zfacs[0];
			hess_tot[1][1] *= zfacs[0];
			hess_tot[0][1] *= zfacs[0];
			hess_tot[1][0] *= zfacs[0];
		}
	}
	else {
		hess_tot = defspline->hessian(x,y);
	}
}

inline void QLens::hessian_weak(const double& x, const double& y, lensmatrix& hess_tot, const int &thread, double* zfacs) // calculates the Hessian of the lensing potential, but ignores multiplane recursive lensing since it's assumed we're in the weak regime
{
	if (!defspline)
	{
		lensmatrix *hess = &hesses_i[thread];
		int j;
		hess_tot[0][0] = 0;
		hess_tot[1][1] = 0;
		hess_tot[0][1] = 0;
		hess_tot[1][0] = 0;
		for (j=0; j < nlens; j++) {
			lens_list[j]->hessian(x,y,(*hess));
			hess_tot[0][0] += (*hess)[0][0];
			hess_tot[1][1] += (*hess)[1][1];
			hess_tot[0][1] += (*hess)[0][1];
			hess_tot[1][0] += (*hess)[1][0];
		}
		hess_tot[0][0] *= zfacs[0];
		hess_tot[1][1] *= zfacs[0];
		hess_tot[0][1] *= zfacs[0];
		hess_tot[1][0] *= zfacs[0];
	}
	else {
		hess_tot = defspline->hessian(x,y);
	}
}

/*
inline void QLens::kappa_inverse_mag_sourcept(const lensvector& xvec, lensvector& srcpt, double &kap_tot, double &invmag, const int &thread, double* zfacs, double** betafacs)
{
	//cout << "CHECK " << zfacs[0] << " " << betafacs[0][0] << endl;
	double x = xvec[0], y = xvec[1];
	lensmatrix *jac = &jacs[thread];
	lensvector *def_tot = &defs[thread];

	if (!defspline)
	{
		if (n_lens_redshifts > 1) {
			lensvector *x_i = &xvals_i[thread];
			lensmatrix *A_i = &Amats_i[thread];
			lensvector *def = &defs_i[thread];
			lensvector **def_i = &defs_subtot[thread];
			lensmatrix *hess = &hesses_i[thread];
			lensmatrix **hess_i = &hesses_subtot[thread];

			int i,j;
			(*jac)[0][0] = 0;
			(*jac)[1][1] = 0;
			(*jac)[0][1] = 0;
			(*jac)[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			for (i=0; i < n_lens_redshifts; i++) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
					(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
					(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
					(*A_i)[0][0] -= (betafacs[i-1][j])*((*hess_i)[j])[0][0];
					(*A_i)[1][1] -= (betafacs[i-1][j])*((*hess_i)[j])[1][1];
					(*A_i)[1][0] -= (betafacs[i-1][j])*((*hess_i)[j])[1][0];
					(*A_i)[0][1] -= (betafacs[i-1][j])*((*hess_i)[j])[0][1];
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					lens_list[zlens_group_lens_indx[i][j]]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess));
					(*hess_i)[i][0][0] += (*hess)[0][0];
					(*hess_i)[i][1][1] += (*hess)[1][1];
					(*hess_i)[i][0][1] += (*hess)[0][1];
					(*hess_i)[i][1][0] += (*hess)[1][0];
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
				(*def_i)[i][0] *= zfacs[i];
				(*def_i)[i][1] *= zfacs[i];
				(*def_tot)[0] += (*def_i)[i][0];
				(*def_tot)[1] += (*def_i)[i][1];

				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];

				(*jac)[0][0] += (*hess_i)[i][0][0];
				(*jac)[1][1] += (*hess_i)[i][1][1];
				(*jac)[1][0] += (*hess_i)[i][1][0];
				(*jac)[0][1] += (*hess_i)[i][0][1];
			}
			kap_tot = ((*jac)[0][0] + (*jac)[1][1])/2;
		} else {
			(*jac)[0][0] = 0;
			(*jac)[1][1] = 0;
			(*jac)[0][1] = 0;
			(*jac)[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			kap_tot = 0;

			if (nthreads==1) {
				int j;
				double kap;
				(*jac)[0][0] = 0;
				(*jac)[1][1] = 0;
				(*jac)[0][1] = 0;
				(*jac)[1][0] = 0;
				(*def_tot)[0] = 0;
				(*def_tot)[1] = 0;
				kap_tot = 0;
				lensvector *def = &defs_i[0];
				lensmatrix *hess = &hesses_i[0];
				for (j=0; j < nlens; j++) {
					lens_list[j]->kappa_and_potential_derivatives(x,y,kap,(*def),(*hess));
					(*jac)[0][0] += (*hess)[0][0];
					(*jac)[1][1] += (*hess)[1][1];
					(*jac)[0][1] += (*hess)[0][1];
					(*jac)[1][0] += (*hess)[1][0];
					(*def_tot)[0] += (*def)[0];
					(*def_tot)[1] += (*def)[1];
					kap_tot += kap;
				}
			} else {
				// The following parallel scheme is useful for clusters when LOTS of perturbers are present
				#pragma omp parallel
				{
					int thread2;
#ifdef USE_OPENMP
					thread2 = omp_get_thread_num();
#else
					thread2 = 0;
#endif
					lensvector *def = &defs_i[thread2];
					lensmatrix *hess = &hesses_i[thread2];
					double hess00=0, hess11=0, hess01=0, def0=0, def1=0, kapi=0;
					int j;
					double kap;
					#pragma omp for schedule(dynamic)
					for (j=0; j < nlens; j++) {
						lens_list[j]->kappa_and_potential_derivatives(x,y,kap,(*def),(*hess));
						hess00 += (*hess)[0][0];
						hess11 += (*hess)[1][1];
						hess01 += (*hess)[0][1];
						def0 += (*def)[0];
						def1 += (*def)[1];
						kapi += kap;
					}
					#pragma omp critical
					{
						(*jac)[0][0] += hess00;
						(*jac)[1][1] += hess11;
						(*jac)[0][1] += hess01;
						(*jac)[1][0] += hess01;
						(*def_tot)[0] += def0;
						(*def_tot)[1] += def1;
						kap_tot += kapi;
					}
				}
			}
			(*jac)[0][0] *= zfacs[0];
			(*jac)[1][1] *= zfacs[0];
			(*jac)[0][1] *= zfacs[0];
			(*jac)[1][0] *= zfacs[0];
			(*def_tot)[0] *= zfacs[0];
			(*def_tot)[1] *= zfacs[0];
			kap_tot *= zfacs[0];
		}
	}
	else {
		(*def_tot) = defspline->deflection(x,y);
		(*jac) = defspline->hessian(x,y);
		kap_tot = kappa(x,y,zfacs,betafacs);
	}
	srcpt[0] = x - (*def_tot)[0]; // this uses the lens equation, beta = theta - alpha
	srcpt[1] = y - (*def_tot)[1];

	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	invmag = determinant((*jac));
}

inline void QLens::sourcept_jacobian(const lensvector& xvec, lensvector& srcpt, lensmatrix& jac_tot, const int &thread, double* zfacs, double** betafacs)
{
	double x = xvec[0], y = xvec[1];
	lensvector *def_tot = &defs[thread];

	if (!defspline)
	{
		if (n_lens_redshifts > 1) {
			lensvector *x_i = &xvals_i[thread];
			lensmatrix *A_i = &Amats_i[thread];
			lensvector *def = &defs_i[thread];
			lensvector **def_i = &defs_subtot[thread];
			lensmatrix *hess = &hesses_i[thread];
			lensmatrix **hess_i = &hesses_subtot[thread];

			int i,j;
			jac_tot[0][0] = 0;
			jac_tot[1][1] = 0;
			jac_tot[0][1] = 0;
			jac_tot[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;
			for (i=0; i < n_lens_redshifts; i++) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << "...\n";
					(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
					(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
					(*A_i)[0][0] -= (betafacs[i-1][j])*((*hess_i)[j])[0][0];
					(*A_i)[1][1] -= (betafacs[i-1][j])*((*hess_i)[j])[1][1];
					(*A_i)[1][0] -= (betafacs[i-1][j])*((*hess_i)[j])[1][0];
					(*A_i)[0][1] -= (betafacs[i-1][j])*((*hess_i)[j])[0][1];
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					lens_list[zlens_group_lens_indx[i][j]]->potential_derivatives((*x_i)[0],(*x_i)[1],(*def),(*hess));
					(*hess_i)[i][0][0] += (*hess)[0][0];
					(*hess_i)[i][1][1] += (*hess)[1][1];
					(*hess_i)[i][0][1] += (*hess)[0][1];
					(*hess_i)[i][1][0] += (*hess)[1][0];
					(*def_i)[i][0] += (*def)[0];
					(*def_i)[i][1] += (*def)[1];
				}
				(*def_i)[i][0] *= zfacs[i];
				(*def_i)[i][1] *= zfacs[i];
				(*def_tot)[0] += (*def_i)[i][0];
				(*def_tot)[1] += (*def_i)[i][1];

				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];

				jac_tot[0][0] += (*hess_i)[i][0][0];
				jac_tot[1][1] += (*hess_i)[i][1][1];
				jac_tot[1][0] += (*hess_i)[i][1][0];
				jac_tot[0][1] += (*hess_i)[i][0][1];
			}
		} else {
			jac_tot[0][0] = 0;
			jac_tot[1][1] = 0;
			jac_tot[0][1] = 0;
			jac_tot[1][0] = 0;
			(*def_tot)[0] = 0;
			(*def_tot)[1] = 0;

			if (nthreads==1) {
				lensvector *def = &defs_i[0];
				lensmatrix *hess = &hesses_i[0];
				int j;
				jac_tot[0][0] = 0;
				jac_tot[1][1] = 0;
				jac_tot[0][1] = 0;
				jac_tot[1][0] = 0;
				(*def_tot)[0] = 0;
				(*def_tot)[1] = 0;
				for (j=0; j < nlens; j++) {
					lens_list[j]->potential_derivatives(x,y,(*def),(*hess));
					jac_tot[0][0] += (*hess)[0][0];
					jac_tot[1][1] += (*hess)[1][1];
					jac_tot[0][1] += (*hess)[0][1];
					jac_tot[1][0] += (*hess)[1][0];
					(*def_tot)[0] += (*def)[0];
					(*def_tot)[1] += (*def)[1];
				}
			} else {
				// The following parallel scheme is useful for clusters when LOTS of perturbers are present
				#pragma omp parallel
				{
					int thread2;
#ifdef USE_OPENMP
					thread2 = omp_get_thread_num();
#else
					thread2 = 0;
#endif
					lensvector *def = &defs_i[thread2];
					lensmatrix *hess = &hesses_i[thread2];
					double hess00=0, hess11=0, hess01=0, def0=0, def1=0, kapi=0;
					int j;
					double kap;
					#pragma omp for schedule(dynamic)
					for (j=0; j < nlens; j++) {
						lens_list[j]->potential_derivatives(x,y,(*def),(*hess));
						hess00 += (*hess)[0][0];
						hess11 += (*hess)[1][1];
						hess01 += (*hess)[0][1];
						def0 += (*def)[0];
						def1 += (*def)[1];
					}
					#pragma omp critical
					{
						jac_tot[0][0] += hess00;
						jac_tot[1][1] += hess11;
						jac_tot[0][1] += hess01;
						jac_tot[1][0] += hess01;
						(*def_tot)[0] += def0;
						(*def_tot)[1] += def1;
					}
				}
			}
			jac_tot[0][0] *= zfacs[0];
			jac_tot[1][1] *= zfacs[0];
			jac_tot[0][1] *= zfacs[0];
			jac_tot[1][0] *= zfacs[0];
			(*def_tot)[0] *= zfacs[0];
			(*def_tot)[1] *= zfacs[0];
		}
	}
	else {
		(*def_tot) = defspline->deflection(x,y);
		jac_tot = defspline->hessian(x,y);
	}
	srcpt[0] = x - (*def_tot)[0]; // this uses the lens equation, beta = theta - alpha
	srcpt[1] = y - (*def_tot)[1];

	jac_tot[0][0] = 1 - jac_tot[0][0];
	jac_tot[1][1] = 1 - jac_tot[1][1];
	jac_tot[0][1] = -jac_tot[0][1];
	jac_tot[1][0] = -jac_tot[1][0];
}
*/

inline void QLens::find_sourcept(const lensvector& x, lensvector& srcpt, const int& thread, double* zfacs, double** betafacs)
{
	deflection(x[0],x[1],srcpt,thread,zfacs,betafacs);
	srcpt[0] = x[0] - srcpt[0]; // this uses the lens equation, beta = theta - alpha (except without defining an intermediate lensvector alpha, which would be an extra memory operation)
	srcpt[1] = x[1] - srcpt[1];
}

inline void QLens::find_sourcept(const lensvector& x, double& srcpt_x, double& srcpt_y, const int& thread, double* zfacs, double** betafacs)
{
	deflection(x[0],x[1],srcpt_x,srcpt_y,thread,zfacs,betafacs);
	srcpt_x = x[0] - srcpt_x; // this uses the lens equation, beta = theta - alpha (except without defining an intermediate lensvector alpha, which would be an extra memory operation)
	srcpt_y = x[1] - srcpt_y;
}

inline double QLens::inverse_magnification(const lensvector& x, const int &thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian(x[0],x[1],(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	return determinant((*jac));
}

inline double QLens::magnification(const lensvector &x, const int &thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian(x[0],x[1],(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	return 1.0/determinant((*jac));
}

inline double QLens::shear(const lensvector &x, const int &thread, double* zfacs, double** betafacs)
{
	lensmatrix *hess = &hesses[thread];
	hessian(x[0],x[1],(*hess),thread,zfacs,betafacs);
	double shear1, shear2;
	shear1 = 0.5*((*hess)[0][0]-(*hess)[1][1]);
	shear2 = (*hess)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}

inline void QLens::shear(const lensvector &x, double& shear_tot, double& angle, const int &thread, double* zfacs, double** betafacs)
{
	lensmatrix *hess = &hesses[thread];
	hessian(x[0],x[1],(*hess),thread,zfacs,betafacs);
	double shear1, shear2;
	shear1 = 0.5*((*hess)[0][0]-(*hess)[1][1]);
	shear2 = (*hess)[0][1];
	shear_tot = sqrt(shear1*shear1+shear2*shear2);
	if (shear1==0) {
		if (shear2 > 0) angle = M_HALFPI;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(shear2/shear1));
		if (shear1 < 0) {
			if (shear2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (shear2 < 0) {
			angle = -angle;
		}
	}
	angle = 0.5*radians_to_degrees(angle);
}

inline void QLens::reduced_shear_components(const lensvector &x, double& g1, double& g2, const int &thread, double* zfacs)
{
	lensmatrix *hess = &hesses[thread];
	hessian_weak(x[0],x[1],(*hess),thread,zfacs);
	double kap_denom = 1 - ((*hess)[0][0] + (*hess)[1][1])/2;
	g1 = 0.5*((*hess)[0][0]-(*hess)[1][1]) / kap_denom;
	g2 = (*hess)[0][1] / kap_denom;
}

// the following functions find the shear, kappa and magnification at the position where a perturber is placed;
// this information is used to determine the optimal subgrid size and resolution

inline void QLens::hessian_exclude(const double& x, const double& y, bool* exclude, lensmatrix& hess_tot, const int& thread, double* zfacs, double** betafacs)
{
	bool skip_lens_plane = false;
	int skip_i = -1;
	lensvector *x_i = &xvals_i[thread];
	lensmatrix *A_i = &Amats_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];
	lensmatrix *hess = &hesses_i[thread];
	lensmatrix **hess_i = &hesses_subtot[thread];

	int i,j;
	for (i=0; i < n_lens_redshifts; i++) {
		if ((zlens_group_size[i]==1) and (exclude[zlens_group_lens_indx[i][0]])) {
			skip_lens_plane = true;
			skip_i = i;
			// should allow for multiple redshifts to be excluded...fix later
		}
	}
	hess_tot[0][0] = 0;
	hess_tot[1][1] = 0;
	hess_tot[0][1] = 0;
	hess_tot[1][0] = 0;
	if (n_lens_redshifts > 1) {
		for (i=0; i < n_lens_redshifts; i++) {
			if ((!skip_lens_plane) or (skip_i != i)) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << " " << (*def_i)[j][0] << " " << (*def_i)[j][1] << "...\n";
					if ((!skip_lens_plane) or (skip_i != j)) {
						(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
						(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
						(*A_i) -= (betafacs[i-1][j])*((*hess_i)[j]);
					}
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					// if this is only lens in the lens plane, we still want to include in hessian/deflection until the very
					// end when we add up hessian, because we want the nonlinear effects taken into account here
					if (exclude[zlens_group_lens_indx[i][j]]) ;
					else {
						lens_list[zlens_group_lens_indx[i][j]]->hessian((*x_i)[0],(*x_i)[1],(*hess));
						//cout << "lens " << zlens_group_lens_indx[i][j] << ", x=" << (*x_i)[0] << ", y=" << (*x_i)[1] << ", hess: " << (*hess)[0][0] << " " << (*hess)[1][1] << " " << (*hess)[0][1] << endl;
						(*hess_i)[i][0][0] += (*hess)[0][0];
						(*hess_i)[i][1][1] += (*hess)[1][1];
						(*hess_i)[i][0][1] += (*hess)[0][1];
						(*hess_i)[i][1][0] += (*hess)[1][0];
						if (i < n_lens_redshifts-1) {
							lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
							(*def_i)[i][0] += (*def)[0];
							(*def_i)[i][1] += (*def)[1];
						}
					}
				}
				if (i < n_lens_redshifts-1) {
					(*def_i)[i][0] *= zfacs[i];
					(*def_i)[i][1] *= zfacs[i];
				}
				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				//cout << "lens plane " << i << ", hess before: " << (*hess_i)[i][0][0] << " " << (*hess_i)[i][1][1] << " " << (*hess_i)[i][0][1] << endl;
				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];
				//cout << "lens plane " << i << ", hess after: " << (*hess_i)[i][0][0] << " " << (*hess_i)[i][1][1] << " " << (*hess_i)[i][0][1] << endl;

				hess_tot += (*hess_i)[i];
			}
		}
	} else {
		if (use_perturber_flags) {
			for (i=0; i < nlens; i++) {
				if ((!exclude[i]) and (lens_list[i]->perturber==false)) {
					lens_list[i]->hessian(x,y,(*hess));
					hess_tot[0][0] += (*hess)[0][0];
					hess_tot[1][1] += (*hess)[1][1];
					hess_tot[0][1] += (*hess)[0][1];
					hess_tot[1][0] += (*hess)[1][0];
				}
			}
		} else {
			for (i=0; i < nlens; i++) {
				if (!exclude[i]) {
					lens_list[i]->hessian(x,y,(*hess));
					hess_tot[0][0] += (*hess)[0][0];
					hess_tot[1][1] += (*hess)[1][1];
					hess_tot[0][1] += (*hess)[0][1];
					hess_tot[1][0] += (*hess)[1][0];
				}
			}
		}
		hess_tot[0][0] *= zfacs[0];
		hess_tot[1][1] *= zfacs[0];
		hess_tot[0][1] *= zfacs[0];
		hess_tot[1][0] *= zfacs[0];
	}
}

inline double QLens::magnification_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];

	return 1.0/determinant((*jac));
}

inline double QLens::inverse_magnification_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];

	return determinant((*jac));
}

inline double QLens::shear_exclude(const lensvector &x, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	double shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}

inline void QLens::shear_exclude(const lensvector &x, double &shear, double &angle, bool* exclude, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	double shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	shear = sqrt(shear1*shear1+shear2*shear2);
	if (shear1==0) {
		if (shear2 > 0) angle = M_HALFPI;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(shear2/shear1));
		if (shear1 < 0) {
			if (shear2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (shear2 < 0) {
			angle = -angle;
		}
	}
	angle = 0.5*radians_to_degrees(angle);
}

inline double QLens::kappa_exclude(const lensvector &x, bool* exclude, double* zfacs, double** betafacs)
{

	double kappa;
	if (n_lens_redshifts==1) {
		int j;
		kappa=0;
		if (use_perturber_flags) {
			for (j=0; j < nlens; j++) {
				if ((!exclude[j]) and (lens_list[j]->perturber==false))
					kappa += lens_list[j]->kappa(x[0],x[1]);
			}
		} else {
			for (j=0; j < nlens; j++) {
				if (!exclude[j])
					kappa += lens_list[j]->kappa(x[0],x[1]);
			}
		}
		kappa *= zfacs[0];
	} else {
		lensmatrix *jac = &jacs[0];
		hessian_exclude(x[0],x[1],exclude,(*jac),0,zfacs,betafacs);
		kappa = ((*jac)[0][0] + (*jac)[1][1])/2;
	}
	return kappa;
}

/*

inline void QLens::hessian_exclude(const double& x, const double& y, const int& exclude_i, lensmatrix& hess_tot, const int& thread, double* zfacs, double** betafacs)
{
	bool skip_lens_plane = false;
	int skip_i = -1;
	lensvector *x_i = &xvals_i[thread];
	lensmatrix *A_i = &Amats_i[thread];
	lensvector *def = &defs_i[thread];
	lensvector **def_i = &defs_subtot[thread];
	lensmatrix *hess = &hesses_i[thread];
	lensmatrix **hess_i = &hesses_subtot[thread];

	int i,j;
	for (i=0; i < n_lens_redshifts; i++) {
		if ((zlens_group_size[i]==1) and (zlens_group_lens_indx[i][0] == exclude_i)) {
			skip_lens_plane = true;
			skip_i = i;
		}
	}
	hess_tot[0][0] = 0;
	hess_tot[1][1] = 0;
	hess_tot[0][1] = 0;
	hess_tot[1][0] = 0;
	if (n_lens_redshifts > 1) {
		for (i=0; i < n_lens_redshifts; i++) {
			if ((!skip_lens_plane) or (skip_i != i)) {
				(*hess_i)[i][0][0] = 0;
				(*hess_i)[i][1][1] = 0;
				(*hess_i)[i][0][1] = 0;
				(*hess_i)[i][1][0] = 0;
				(*A_i)[0][0] = 1;
				(*A_i)[1][1] = 1;
				(*A_i)[0][1] = 0;
				(*A_i)[1][0] = 0;
				(*def_i)[i][0] = 0;
				(*def_i)[i][1] = 0;
				(*x_i)[0] = x;
				(*x_i)[1] = y;
				for (j=0; j < i; j++) {
					//cout << "Using betafactor " << i-1 << " " << j << " = " << betafacs[i-1][j] << " " << (*def_i)[j][0] << " " << (*def_i)[j][1] << "...\n";
					if ((!skip_lens_plane) or (skip_i != j)) {
						(*x_i)[0] -= betafacs[i-1][j]*(*def_i)[j][0];
						(*x_i)[1] -= betafacs[i-1][j]*(*def_i)[j][1];
						(*A_i) -= (betafacs[i-1][j])*((*hess_i)[j]);
					}
				}
				for (j=0; j < zlens_group_size[i]; j++) {
					// if this is only lens in the lens plane, we still want to include in hessian/deflection until the very
					// end when we add up hessian, because we want the nonlinear effects taken into account here
					if (zlens_group_lens_indx[i][j] == exclude_i) ;
					else {
						lens_list[zlens_group_lens_indx[i][j]]->hessian((*x_i)[0],(*x_i)[1],(*hess));
						//cout << "lens " << zlens_group_lens_indx[i][j] << ", x=" << (*x_i)[0] << ", y=" << (*x_i)[1] << ", hess: " << (*hess)[0][0] << " " << (*hess)[1][1] << " " << (*hess)[0][1] << endl;
						(*hess_i)[i][0][0] += (*hess)[0][0];
						(*hess_i)[i][1][1] += (*hess)[1][1];
						(*hess_i)[i][0][1] += (*hess)[0][1];
						(*hess_i)[i][1][0] += (*hess)[1][0];
						if (i < n_lens_redshifts-1) {
							lens_list[zlens_group_lens_indx[i][j]]->deflection((*x_i)[0],(*x_i)[1],(*def));
							(*def_i)[i][0] += (*def)[0];
							(*def_i)[i][1] += (*def)[1];
						}
					}
				}
				if (i < n_lens_redshifts-1) {
					(*def_i)[i][0] *= zfacs[i];
					(*def_i)[i][1] *= zfacs[i];
				}
				(*hess_i)[i][0][0] *= zfacs[i];
				(*hess_i)[i][1][1] *= zfacs[i];
				(*hess_i)[i][0][1] *= zfacs[i];
				(*hess_i)[i][1][0] *= zfacs[i];

				//cout << "lens plane " << i << ", hess before: " << (*hess_i)[i][0][0] << " " << (*hess_i)[i][1][1] << " " << (*hess_i)[i][0][1] << endl;
				(*hess)[0][0] = (*hess_i)[i][0][0]; // temporary storage for matrix multiplication
				(*hess)[0][1] = (*hess_i)[i][0][1]; // temporary storage for matrix multiplication
				(*hess_i)[i][0][0] = (*hess_i)[i][0][0]*(*A_i)[0][0] + (*hess_i)[i][1][0]*(*A_i)[0][1];
				(*hess_i)[i][1][0] = (*hess)[0][0]*(*A_i)[1][0] + (*hess_i)[i][1][0]*(*A_i)[1][1];
				(*hess_i)[i][0][1] = (*hess_i)[i][0][1]*(*A_i)[0][0] + (*hess_i)[i][1][1]*(*A_i)[0][1];
				(*hess_i)[i][1][1] = (*hess)[0][1]*(*A_i)[1][0] + (*hess_i)[i][1][1]*(*A_i)[1][1];
				//cout << "lens plane " << i << ", hess after: " << (*hess_i)[i][0][0] << " " << (*hess_i)[i][1][1] << " " << (*hess_i)[i][0][1] << endl;

				hess_tot += (*hess_i)[i];
			}
		}
	} else {
		if (use_perturber_flags) {
			for (i=0; i < nlens; i++) {
				if ((i != exclude_i) and (lens_list[i]->perturber==false)) {
					lens_list[i]->hessian(x,y,(*hess));
					hess_tot[0][0] += (*hess)[0][0];
					hess_tot[1][1] += (*hess)[1][1];
					hess_tot[0][1] += (*hess)[0][1];
					hess_tot[1][0] += (*hess)[1][0];
				}
			}
		} else {
			for (i=0; i < nlens; i++) {
				if (i != exclude_i) {
					lens_list[i]->hessian(x,y,(*hess));
					hess_tot[0][0] += (*hess)[0][0];
					hess_tot[1][1] += (*hess)[1][1];
					hess_tot[0][1] += (*hess)[0][1];
					hess_tot[1][0] += (*hess)[1][0];
				}
			}
		}
		hess_tot[0][0] *= zfacs[0];
		hess_tot[1][1] *= zfacs[0];
		hess_tot[0][1] *= zfacs[0];
		hess_tot[1][0] *= zfacs[0];
	}
}

inline double QLens::magnification_exclude(const lensvector &x, const int& exclude_i, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude_i,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];

	return 1.0/determinant((*jac));
}

inline double QLens::shear_exclude(const lensvector &x, const int& exclude_i, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude_i,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	double shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}

inline void QLens::shear_exclude(const lensvector &x, double &shear, double &angle, const int& exclude_i, const int& thread, double* zfacs, double** betafacs)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude_i,(*jac),thread,zfacs,betafacs);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	double shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	shear = sqrt(shear1*shear1+shear2*shear2);
	if (shear1==0) {
		if (shear2 > 0) angle = M_HALFPI;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(shear2/shear1));
		if (shear1 < 0) {
			if (shear2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (shear2 < 0) {
			angle = -angle;
		}
	}
	angle = 0.5*radians_to_degrees(angle);
}

inline double QLens::kappa_exclude(const lensvector &x, const int& exclude_i, double* zfacs, double** betafacs)
{

	double kappa;
	if (n_lens_redshifts==1) {
		int j;
		kappa=0;
		if (use_perturber_flags) {
			for (j=0; j < nlens; j++) {
				if ((j != exclude_i) and (lens_list[j]->perturber==false))
					kappa += lens_list[j]->kappa(x[0],x[1]);
			}
		} else {
			for (j=0; j < nlens; j++) {
				if (j != exclude_i)
					kappa += lens_list[j]->kappa(x[0],x[1]);
			}
		}
		kappa *= zfacs[0];
	} else {
		lensmatrix *jac = &jacs[0];
		hessian_exclude(x[0],x[1],exclude_i,(*jac),0,zfacs,betafacs);
		kappa = ((*jac)[0][0] + (*jac)[1][1])/2;
	}
	return kappa;
}

*/


#endif // QLENS_H
