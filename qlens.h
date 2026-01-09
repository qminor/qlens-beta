#ifndef QLENS_H
#define QLENS_H

#include "modelparams.h"
#include "sort.h"
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
#include "mcmceval.h"
#ifdef USE_MUMPS
#include "dmumps_c.h"
#endif
#include <cstdio>
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

using std::string;

enum ImageSystemType { NoImages, Single, Double, Cusp, Quad };
enum inside_cell { Inside, Outside, Edge };
enum edge_sourcept_status { SourceInGap, SourceInOverlap, NoSource };
enum SourceFitMode { Point_Source, Cartesian_Source, Delaunay_Source, Parameterized_Source, Shapelet_Source };
enum Prior { UNIFORM_PRIOR, LOG_PRIOR, GAUSS_PRIOR, GAUSS2_PRIOR, GAUSS2_PRIOR_SECONDARY };
enum Transform { NONE, LOG_TRANSFORM, GAUSS_TRANSFORM, LINEAR_TRANSFORM, RATIO };
enum RegularizationMethod { None, Norm, Gradient, SmoothGradient, Curvature, SmoothCurvature, Matern_Kernel, Exponential_Kernel, Squared_Exponential_Kernel };
enum RayTracingMethod {
	Interpolate,
	Area_Overlap
};

enum DerivedParamType {
	KappaR,
	LambdaR,
	DlogKappaR,
	Mass2dR,
	Mass3dR,
	Einstein,
	Einstein_Mass,
	Xi_Param,
	Kappa_Re,
	LensParam,
	AvgLogSlope,
	Perturbation_Radius,
	Relative_Perturbation_Radius,
	Robust_Perturbation_Mass,
	Robust_Perturbation_Density,
	Adaptive_Grid_qs,
	Adaptive_Grid_phi_s,
	Adaptive_Grid_sig_s,
	Adaptive_Grid_xavg,
	Adaptive_Grid_yavg,
	Chi_Square,
	UserDefined
};

class QLens;			// Defined after class Grid
class ModelParams;
class CartesianSourceGrid;
class LensPixelGrid;
class DelaunaySourceGrid;
class ImagePixelGrid;
class DelaunaySourceGrid;
class PSF;
struct PointImageData;
struct WeakLensingData;
struct ImageData;
struct ParamList;
struct DerivedParamList;
struct DerivedParam;
class LensList;
class SourceList;
class PixSrcList;
class PtSrcList;
class ImgDataList;

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

struct PtImageDataSet {
	double zsrc;
	int n_images;
	std::vector<image_data> images;

	void set_n_images(const int nimg) {
		n_images = nimg;
		images.clear();
		images.resize(n_images);
	}
};

class PointSource : public ModelParams
{
	friend class QLens;
	friend class SB_Profile;

	private:
	int zsrc_paramnum; // just to keep track of which parameter number is zsrc (set when constructor is called)

	public:
	lensvector pos;
	lensvector shift; // allows for a small correction to the source position estimated using analytic_bestfit_src
	double zsrc, srcflux;
	bool include_shift;
	int n_images;
	std::vector<image> images;

	public:
	PointSource() { qlens = NULL; }
	PointSource(QLens* lens_in);
	PointSource(QLens* lens_in, const lensvector& sourcept, const double zsrc_in);
	PointSource(lensvector& src_in, double zsrc_in, image* images_in, const int nimg, const double srcflux_in = 1.0) {
		copy_imageset(src_in, zsrc_in, images_in, nimg, srcflux_in);
	}
	void setup_parameters(const bool initial_setup);
	void update_meta_parameters(const bool varied_only_fitparams);
	void get_parameter_numbers_from_qlens(int& pi, int& pf);
	bool register_vary_parameters_in_qlens();
	void register_limits_in_qlens();
	void update_fitparams_in_qlens();
	void set_vary_source_coords();
	void copy_ptsrc_data(PointSource* ptsrc_in);
	void copy_imageset(const lensvector& pos_in, const double zsrc_in, image* images_in, const int nimg, const double srcflux_in = 1.0);
	void update_srcpos(const lensvector& srcpt);
	double imgflux(const int imgnum) { if (imgnum < n_images) return abs(images[imgnum].mag*srcflux); else return -1; }
	void print(bool include_time_delays = false, bool show_labels = true) { print_to_file(include_time_delays,show_labels,NULL,NULL); }
	void print_to_file(bool include_time_delays, bool show_labels, std::ofstream* srcfile, std::ofstream* imgfile);
	void reset_images() { n_images = 0; images.clear(); }
};

class Grid : private Brent
{
	private:
	// this constructor is only used by the top-level Grid to initialize the lower-level grids, so it's private
	Grid(lensvector** xij, const int& i, const int& j, const int& level_in, Grid* parent_ptr);

	Grid*** cell;
	static QLens* lens;
	static int nthreads;
	Grid* neighbor[4]; // 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	Grid* parent_cell;

	public:
	// Instead of all these static variables, have a class Grid versus GridCell, where Grid is the parent that contains all these as non-static variables
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
	void add_image_to_list(const lensvector& imgpos);

	bool run_newton(lensvector& xroot, const int& thread);
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
	static std::ofstream xgrid;
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

// There is too much inheritance going on here. Nearly all of these (except ModelParams) can be changed to simply objects that are created within the QLens
// class; it's more transparent to do so, and more object-oriented.
class QLens : public ModelParams, public UCMC, private Brent, private Sort, private Powell, private Simplex
{
	private:
	// These are arrays of dummy variables used for lensing calculations, arranged so that each thread gets its own set of dummy variables.
	static lensvector *defs, **defs_subtot, *defs_i, *xvals_i;
	static lensmatrix *jacs, *hesses, **hesses_subtot, *hesses_i, *Amats_i;
	static int *indxs;

	double raw_chisq;
	int chisq_it;
	bool chisq_diagnostic;
	std::ofstream logfile;

	const int HALF_DATAPIXELS = -1;
	const int ALL_DATAPIXELS = 0;

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
	bool multinest_constant_eff_mode;
	double multinest_target_efficiency;
	bool multimodal_sampling;
	int polychord_nrepeats;
	int mcmc_threads;
	double mcmc_tolerance; // for Metropolis-Hastings
	bool mcmc_logfile;
	bool open_chisq_logfile;
	//bool psf_convolution_mpi;
	bool fft_convolution;
	bool use_mumps_subcomm;
	int fgmask_padding;
	bool n_image_prior;
	int auxiliary_srcgrid_npixels;
	double n_image_threshold;
	double srcpixel_nimg_mag_threshold;
	bool outside_sb_prior;
	double outside_sb_prior_noise_frac, n_image_prior_sb_frac;
	double outside_sb_prior_threshold;
	bool einstein_radius_prior;
	bool concentration_prior;
	double einstein_radius_low_threshold;
	double einstein_radius_high_threshold;
	bool include_fgmask_in_inversion;
	bool zero_sb_fgmask_prior;
	bool include_noise_term_in_loglike;
	double high_sn_frac;
	bool use_custom_prior;
	bool lens_position_gaussian_transformation;
	ParamList *param_list;
	DerivedParamList *dparam_list;

	int nlens;
	LensProfile** lens_list;
	LensList *lenslist;
	int *lens_redshift_idx;

	int n_sb;
	SB_Profile** sb_list;
	SourceList *srclist;
	int* sbprofile_redshift_idx;
	int* sbprofile_band_number;
	int* sbprofile_imggrid_idx;

	int n_pixellated_src;
	int* pixellated_src_redshift_idx;
	int* pixellated_src_band;
	ModelParams **srcgrids; // this is the base class for Delaunay and cartesian source grids
	PixSrcList *pixsrclist;
	DelaunaySourceGrid **delaunay_srcgrids;
	CartesianSourceGrid **cartesian_srcgrids;

	int n_pixellated_lens;
	int* pixellated_lens_redshift_idx;
	LensPixelGrid **lensgrids; // at the moment, the only kind of object is pixellated potential grids (used for first-order potential corrections to lens models)

	Cosmology *cosmo;
	bool cosmology_allocated_within_qlens;

	int n_ptsrc;
	PointSource **ptsrc_list;
	PtSrcList *ptsrclist;
	int* ptsrc_redshift_idx;
	std::vector<int> ptsrc_redshift_groups;

	lensvector source;
	image *images_found;
	ImageSystemType system_type;

	double lens_redshift;
	double source_redshift, reference_source_redshift; // reference zsrc (zsrc_ref) is the redshift used to define scaled lensing quantities (kappa, etc.)

	double *reference_zfactors;
	double **default_zsrc_beta_factors;

	int n_extended_src_redshifts;
	double *extended_src_redshifts; // used for modeling extended sources 
	int n_assigned_masks;
	int *assigned_mask;
	//int *extended_zsrc_group_size;
	//int **extended_zsrc_group_src_index;
	double **extended_src_zfactors;
	double ***extended_src_beta_factors;

	int n_ptsrc_redshifts;
	double *ptsrc_redshifts; // used for modeling point sources 
	double **ptsrc_zfactors;
	double ***ptsrc_beta_factors;

	int n_lens_redshifts;
	double *lens_redshifts;
	int *zlens_group_size;
	int **zlens_group_lens_indx;

	bool user_changed_zsource; // keeps track of whether redshift has been manually changed; if so, then don't change it to redshift from data
	bool auto_zsource_scaling; // this automatically sets zsrc_ref (for kappa scaling) equal to the default source redshift being used

	bool ellipticity_gradient, contours_overlap;
	double contour_overlap_log_penalty_prior;
	double syserr_pos;
	double wl_shear_factor;

	double romberg_accuracy; // for Romberg integration
	bool include_recursive_lensing; // should only turn off if trying to understand effect of recursive lensing from multiple lens planes

	Grid *grid;
	bool radial_grid;
	double grid_xlength, grid_ylength, grid_xcenter, grid_ycenter;  // for gridsize
	double sourcegrid_xmin, sourcegrid_xmax, sourcegrid_ymin, sourcegrid_ymax;
	double sourcegrid_limit_xmin, sourcegrid_limit_xmax, sourcegrid_limit_ymin, sourcegrid_limit_ymax;
	bool cc_neighbor_splittings;
	bool skip_newtons_method;
	double min_cell_area; // area of the smallest allowed cell area
	int usplit_initial, wsplit_initial;
	int splitlevels, cc_splitlevels;

	QLens *fitmodel;
	QLens *lens_parent;
	dvector bestfitparams;
	dmatrix bestfit_fisher_inverse;
	dmatrix fisher_inverse;
	double bestfit_flux;
	double chisq_bestfit;
	SourceFitMode source_fit_mode;
	bool use_ansi_characters;
	int lensmodel_fit_parameters, srcmodel_fit_parameters, pixsrc_fit_parameters, pixlens_fit_parameters, ptsrc_fit_parameters, psf_fit_parameters, cosmo_fit_parameters;

	double *regparam_ptr; // points to regularization parameter for given source pixel grid or shapelet object
	double *regparam_pot_ptr; // points to regularization parameter for potential of given lens pixel grid
	double matern_approx_source_size;
	bool optimize_regparam;
	double optimize_regparam_tol, optimize_regparam_minlog, optimize_regparam_maxlog;
	double regopt_chisqmin, regopt_logdet;
	int max_regopt_iterations;

	// the following parameters are used for luminosity- or distance-weighted regularization
	bool use_lum_weighted_regularization;
	bool use_distance_weighted_regularization;
	bool use_mag_weighted_regularization;
	bool auto_lumreg_center; // if set to true, uses (SB-weighted) centroid of ray-traced points; if false, center coordinates are parameters than can be varied
	bool lumreg_center_from_ptsource; // if true, automatically sets lumreg_center to position of source point (auto_lumreg_center must also be set to 'on')
	bool lensed_lumreg_center; // if true, make lumreg_xcenter and lumreg_ycenter coordinates in the image plane, which are lensed to the source plane
	bool lensed_lumreg_rc; // if true, then lumreg_rc is a distance in the image plane at the position of the (lensed) lumreg center, which is then mapped to rc in source plane
	bool fix_lumreg_sig;
	double lumreg_sig;

	int lum_weight_function;
	bool get_lumreg_from_sbweights;

	double *reg_weight_factor;

	bool use_lum_weighted_srcpixel_clustering;
	bool use_dist_weighted_srcpixel_clustering;

	bool save_sbweights_during_inversion;
	bool use_saved_sbweights;
	double *saved_sbweights;
	int n_sbweights;

	static string fit_output_filename;
	string get_fit_label() { return fit_output_filename; }
	void set_fit_label(const string label_in) {
		fit_output_filename = label_in;
		if (auto_fit_output_dir) fit_output_dir = "chains_" + fit_output_filename;
	}
	bool auto_save_bestfit;
	bool borrowed_image_data; // tells whether image_data is pointing to that of another QLens object (e.g. fitmodel pointing to initial lens object)
	PointImageData *point_image_data;
	WeakLensingData weak_lensing_data;
	double chisq_tolerance;
	int lumreg_max_it;
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
	bool redo_lensing_calculations_before_inversion;
	int delaunay_mode;
	bool delaunay_high_sn_mode;
	bool use_srcpixel_clustering;
	bool include_two_pixsrc_in_Lmatrix;

	bool use_random_delaunay_srcgrid;
	bool reinitialize_random_grid;
	int random_seed;
	int n_ranchisq;
	bool clustering_random_initialization;
	bool weight_initial_centroids;
	bool use_dualtree_kmeans;
	bool use_f_src_clusters;
	double f_src_clusters;
	int n_src_clusters;
	int n_cluster_iterations;
	bool include_potential_perturbations;
	bool first_order_sb_correction;
	bool adopt_final_sbgrad;
	int potential_correction_iterations;
	double delaunay_high_sn_sbfrac;
	bool activate_unmapped_source_pixels;
	double total_srcgrid_overlap_area, high_sn_srcgrid_overlap_area;
	bool exclude_source_pixels_beyond_fit_window;
	bool regrid_if_unmapped_source_subpixels;
	bool calculate_bayes_factor;
	double reference_lnZ;
	double base_srcpixel_imgpixel_ratio;
	double sim_err_pos, sim_err_flux, sim_err_td;
	double sim_err_shear; // actually error in reduced shear (for weak lensing data)
	bool split_imgpixels;
	bool split_high_mag_imgpixels;
	bool delaunay_from_pixel_centers;
	bool raytrace_using_pixel_centers;
	bool psf_supersampling;
	int default_imgpixel_nsplit, emask_imgpixel_nsplit;
	double imgpixel_himag_threshold, imgpixel_lomag_threshold, imgpixel_sb_threshold;

	bool fits_format;
	double default_data_pixel_size;
	bool add_simulated_image_data(const lensvector &sourcept, const double srcflux = 1);
	bool add_ptimage_data_from_unlensed_sourcepts(const bool include_errors_from_fisher_matrix = false, const int param_i = 0, const double scale_errors = 2);
	//bool add_fit_sourcept(const lensvector &sourcept, const double zsrc);
	void write_point_image_data(string filename);
	bool load_point_image_data(string filename);
	void sort_image_data_into_redshift_groups();
	bool plot_srcpts_from_image_data(int dataset_number, std::ofstream* srcfile, const double srcpt_x, const double srcpt_y, const double flux = -1);
	//void remove_image_data(int image_set);
	std::vector<PtImageDataSet> export_to_ImageDataSet(); // for the Python wrapper

	bool load_weak_lensing_data(string filename);
	void add_simulated_weak_lensing_data(const string id, lensvector &sourcept, const double zsrc);
	void add_weak_lensing_data_from_random_sources(const int num_sources, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, const double r_exclude);

	bool read_data_line(std::ifstream& infile, std::vector<string>& datawords, int &n_datawords);
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
	RegularizationMethod regularization_method;
	enum InversionMethod { CG_Method, MUMPS, UMFPACK, DENSE, DENSE_FMATRIX } inversion_method;
	bool use_non_negative_least_squares;
	//bool use_fnnls;
	int max_nnls_iterations;
	double nnls_tolerance;
	RayTracingMethod ray_tracing_method;
	bool natural_neighbor_interpolation;
	bool parallel_mumps, show_mumps_info;

	int n_image_pixels_x, n_image_pixels_y; // note that this is the TOTAL number of pixels in the image, as opposed to image_npixels which gives the # of pixels being fit to
	int srcgrid_npixels_x, srcgrid_npixels_y;
	bool auto_srcgrid_npixels;
	bool auto_srcgrid_set_pixel_size;
	double background_pixel_noise;
	bool simulate_pixel_noise;
	double sb_threshold; // for creating centroid images from pixel maps
	double noise_threshold; // for automatic source grid sizing

	static const int nmax_lens_planes;
	static const double default_autogrid_rmin, default_autogrid_rmax, default_autogrid_frac, default_autogrid_initial_step;
	static const int max_cc_search_iterations;
	static double rmin_frac;
	static const double default_rmin_frac;
	double cc_rmin, cc_rmax, cc_thetasteps;
	bool effectively_spherical;
	double newton_magnification_threshold;
	bool reject_himag_images;
	bool reject_images_found_outside_cell;

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

	bool multithread_perturber_deflections; // provides speedup for large number of perturbers
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

	struct critical_curve {
		std::vector<lensvector> cc_pts;
		std::vector<lensvector> caustic_pts;
		std::vector<double> length_of_cell; // just to make sure the critical curves are being separated out properly
	};

	std::vector<lensvector> critical_curve_pts;
	std::vector<lensvector> caustic_pts;
	std::vector<double> length_of_cc_cell;
	std::vector<critical_curve> sorted_critical_curve;
	std::vector<int> npoints_sorted_critical_curve;
	int n_critical_curves;
	void sort_critical_curves();
	bool sorted_critical_curves;
	static bool auto_store_cc_points;
	std::vector<lensvector> singular_pts;
	int n_singular_points;

	Vector<dvector> find_critical_curves(bool&);
	double theta_crit;
	double inverse_magnification_r(const double);
	//double source_plane_r(const double r);
	bool find_optimal_gridsize();

	bool auto_sourcegrid, auto_shapelet_scaling, auto_shapelet_center;
	int shapelet_scale_mode;
	double shapelet_max_scale;
	double shapelet_window_scaling;
	void plot_source_pixel_grid(const int imggrid_i, const char filename[]);

	int n_image_pixel_grids;
	ImagePixelGrid **image_pixel_grids;
	int* imggrid_zsrc_i;
	int* imggrid_band;

	int n_model_bands, n_data_bands;
	ImageData **imgdata_list;
	ImgDataList *imgdatalist; // this is a container class that is used by the Python wrapper

	int image_npixels, source_npixels, lensgrid_npixels, n_mge_sets, n_mge_amps, source_and_lens_n_amps, n_amps; // note, n_amps can also include point image fluxes
	SB_Profile** mge_list;
	int image_n_subpixels; // for supersampling
	int image_npixels_fgmask;
	int image_npixels_data; // right now, only used during optimization of regparam (and is only different from image_npixels when include_fgmask_in_inversion is used and there is padding of the fgmask)

	double *image_surface_brightness;
	double *image_surface_brightness_supersampled;
	double *imgpixel_covinv_vector;
	double *point_image_surface_brightness;
	double *sbprofile_surface_brightness;
	double *img_minus_sbprofile;
	double *amplitude_vector_minchisq; // used to store best-fit solution during optimization of regularization parameter
	double *amplitude_vector;
	int *img_index_datapixels;

	int *image_pixel_location_Lmatrix;
	int *source_pixel_location_Lmatrix;
	int Lmatrix_n_elements;
	double *Lmatrix;
	int *Lmatrix_index;
	std::vector<double> *Lmatrix_rows;
	std::vector<int> *Lmatrix_index_rows;

	bool assign_pixel_mappings(const int imggrid_i, const bool potential_perturbations=false, const bool verbal=false);
	void assign_foreground_mappings(const int imggrid_i, const bool use_data = true);
	double *Dvector;
	double *Dvector_cov;
	double *Dvector_cov_copy;
	int Fmatrix_nn;
	double *Fmatrix;
	double *Fmatrix_copy; // used when optimizing the regularization parameter
	int *Fmatrix_index;
	bool use_noise_map;
	bool dense_Rmatrix;
	bool find_covmatrix_inverse; // set by user (default=false); if true, finds Rmatrix explicitly (usually more computationally intensive)
	bool use_covariance_matrix; // internal bool; set to true if using covariance kernel reg. and if find_covmatrix_inverse is false
	double covmatrix_epsilon; // fudge factor in covariance matrix diagonal to aid inversion
	bool penalize_defective_covmatrix;

	double **Rmatrix_ptr;
	int **Rmatrix_index_ptr;
	int *source_npixels_ptr;
	int *src_npixel_start_ptr;
	int n_src_inv; // specifies how many pixellated (or shapelet) sources will be included in the Lmatrix
	int* src_npixels_inv;
	int *src_npixel_start;
	double **Rmatrix;
	int **Rmatrix_index;

	double *Rmatrix_pot;
	int *Rmatrix_pot_index;

	// The following are simply used as temporary arrays when constructing Rmatrix
	double *Rmatrix_diag_temp;
	std::vector<double> *Rmatrix_rows;
	std::vector<int> *Rmatrix_index_rows;
	int *Rmatrix_row_nn;

	dmatrix Lmatrix_dense;
	dmatrix Lmatrix_supersampled;
	dmatrix Lmatrix_transpose_ptimg_amps; // this contains just the part of the Lmatrix_transpose whose columns will multiply the point image amplitudes
	dvector Gmatrix_stacked;
	dvector Gmatrix_stacked_copy;
	dvector Fmatrix_stacked;
	dvector Fmatrix_packed;
	dvector Fmatrix_packed_copy; // used when optimizing the regularization parameter

	// these will be arrays of size n_src_inv (number of pixellated sources being included in Lmatrix)
	dvector *covmatrix_stacked;
	dvector *covmatrix_packed;
	dvector *covmatrix_factored;
	dvector *Rmatrix_packed;
	// these will point to the current element in the above arrays
	dvector *covmatrix_stacked_ptr;
	dvector *covmatrix_packed_ptr;
	dvector *covmatrix_factored_ptr;
	dvector *Rmatrix_packed_ptr;

	dvector covmatrix_pot_stacked;
	dvector covmatrix_pot_packed;
	dvector covmatrix_pot_factored;
	dvector Rmatrix_pot_packed;

	dvector *Rmatrix_MGE_packed;
	double *Rmatrix_MGE_log_determinants;

	double* Rmatrix_log_determinant; // array of size n_src_inv (number of pixellated sources being included in Lmatrix)
	double* Rmatrix_log_determinant_ptr;
	double Rmatrix_pot_log_determinant;


	dvector covmatrix_stacked_copy; // used when optimizing the regularization parameter with luminosity weighting
	dvector temp_src; // used when optimizing the regularization parameter

	double *gmatrix[4];
	int *gmatrix_index[4];
	int *gmatrix_row_index[4];
	std::vector<double> *gmatrix_rows[4];
	std::vector<int> *gmatrix_index_rows[4];
	int *gmatrix_row_nn[4];
	int gmatrix_nn[4];

	double *hmatrix[2];
	int *hmatrix_index[2];
	int *hmatrix_row_index[2];
	std::vector<double> *hmatrix_rows[2];
	std::vector<int> *hmatrix_index_rows[2];
	int *hmatrix_row_nn[2];
	int hmatrix_nn[2];

#ifdef USE_MUMPS
	static DMUMPS_STRUC_C *mumps_solver;
#endif

	void convert_Lmatrix_to_dense();
	void construct_Lmatrix_shapelets(const int imggrid_i);
	void add_MGE_amplitudes_to_Lmatrix(const int imggrid_i);
	void PSF_convolution_Lmatrix_dense(const int imggrid_i, const bool verbal=false);
	void create_lensing_matrices_from_Lmatrix_dense(const int imggrid_i, const bool potential_perturbations=false, const bool verbal=false);
	void get_source_regparam_ptr(const int imggrid_i, const int imggrid_include_i, double* &regparam);
	void generate_Gmatrix();
	void add_regularization_term_to_dense_Fmatrix(double *regparam, const bool potential_perturbations=false);
	void add_MGE_regularization_terms_to_dense_Fmatrix(const int imggrid_i);
	double calculate_regularization_prior_term(double *regparam_ptr, const bool potential_perturbations=false);
	double calculate_MGE_regularization_prior_term(const int imggrid_i);

	bool optimize_regularization_parameter(const int imggrid_i, const bool dense_Fmatrix=false, const bool verbal=false, const bool pre_srcgrid = false);
	void setup_regparam_optimization(const int imggrid_i, const bool dense_Fmatrix=false);
	void calculate_subpixel_sbweights(const int imggrid_i, const bool save_sbweights = false, const bool verbal = false);
	void calculate_subpixel_distweights(const int imggrid_i=-1);
	void find_srcpixel_weights(const int imggrid_i=-1);
	void load_pixel_sbweights(const int imggrid_i=-1);
	double chisq_regparam_dense(const double logreg);
	double chisq_regparam(const double logreg);
	void calculate_lumreg_srcpixel_weights(const int imggrid_i, const bool use_sbweights=false);
	void calculate_distreg_srcpixel_weights(const int imggrid_i, const double xc=0, const double yc=0, const double sig=1.0, const bool verbal = false);
	void calculate_srcpixel_scaled_distances(const double xc, const double yc, const double sig, double *dists, lensvector **srcpts, const int nsrcpts, const double e1 = 0, const double e2 = 0);
	void calculate_mag_srcpixel_weights(const int imggrid_i);

	//void add_lum_weighted_reg_term(const bool dense_Fmatrix, const bool use_matrix_copies);
	double brents_min_method(double (QLens::*func)(const double), const double ax, const double bx, const double tol, const bool verbal);
	void create_regularization_matrix_shapelet(const int imggrid_i=-1);
	void create_MGE_regularization_matrices(const int imggrid_i=-1);
	void generate_Rmatrix_shapelet_gradient(const int imggrid_i=-1);
	void generate_Rmatrix_shapelet_curvature(const int imggrid_i=-1);
	//void set_corrlength_for_given_matscale();
	//double corrlength_eq_matern_factor(const double log_corr_length);

	//bool Cholesky_dcmp(double** a, double &logdet, int n);
	//bool Cholesky_dcmp_upper(double** a, double &logdet, int n);
	bool Cholesky_dcmp_packed(double* a, int n);
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
	void repack_matrix_lower(double* packed_matrix, const int nn);
	void repack_matrix_upper(double* packed_matrix, const int nn);

	bool ignore_foreground_in_chisq;
	//bool use_input_psf_matrix;
	//double **psf_matrix;
	//Spline2D psf_spline;
	//double **supersampled_psf_matrix;
	//void generate_supersampled_PSF_matrix(const bool downsample = false, const int downsample_fac = 1);
	//bool load_psf_fits(string fits_filename, const int hdu_indx, const bool supersampled, const bool show_header = false, const bool verbal = false);
	//bool save_psf_fits(string fits_filename, const bool supersampled = false);
	//bool plot_psf(string filename, const bool supersampled);

	//int psf_npixels_x, psf_npixels_y;
	//int supersampled_psf_npixels_x, supersampled_psf_npixels_y;

	int n_psf;
	PSF **psf_list;
	double psf_threshold, psf_ptsrc_threshold;
	int ptimg_nsplit; // allows for subpixel PSF even if supersampling is not being used for all pixels

	double Fmatrix_log_determinant;
	double Gmatrix_log_determinant;
	void initialize_pixel_matrices(const int imggrid_i, const bool potential_perturbations=false, bool verbal=false);
	void initialize_pixel_matrices_shapelets(const int imggrid_i, bool verbal=false);
	void count_shapelet_amplitudes(const int imggrid_i=-1);
	void count_MGE_amplitudes(int& n_mge_objects, int& n_gaussians, const int imggrid_i=-1);
	void clear_pixel_matrices();
	void clear_sparse_lensing_matrices();
	double find_sbprofile_surface_brightness(lensvector &pt);
	void construct_Lmatrix(const int imggrid_i, const bool delaunay=true, const bool potential_perturbations=false, const bool verbal=false);
	void construct_Lmatrix_supersampled(const int imggrid_i, const bool delaunay=true, const bool potential_perturbations=false, const bool verbal=false);
	void PSF_convolution_Lmatrix(const int imggrid_i, bool verbal = false);
	void PSF_convolution_pixel_vector(const int imggrid_i, const bool foreground = false, const bool verbal = false, const bool use_fft = false);
	void average_supersampled_image_surface_brightness(const int imggrid_i=-1);
	void average_supersampled_dense_Lmatrix(const int imggrid_i=-1);
	void cleanup_FFT_convolution_arrays();
	void copy_FFT_convolution_arrays(QLens* lens_in);
	void fourier_transform(double* data, const int ndim, int* nn, const int isign);
	void fourier_transform_parallel(double** data, const int ndata, const int jstart, const int ndim, int* nn, const int isign);

	bool create_regularization_matrix(const int imggrid_i, const bool include_lum_weighting = false, const bool use_sbweights = false, const bool potential_perturbations = false, const bool verbal = false);
	void generate_Rmatrix_from_gmatrices(const int imggrid_i=-1, const bool interpolate = false, const bool potential_perturbations = false);
	void generate_Rmatrix_from_hmatrices(const int imggrid_i=-1, const bool interpolate = false, const bool potential_perturbations = false);
	void generate_Rmatrix_norm(const bool potential_perturbations = false);
	bool generate_Rmatrix_from_covariance_kernel(const int imggrid_i, const int kernel_type=0, const bool include_lum_weighting=false, const bool potential_perturbations = false, const bool verbal = false);
	void find_source_centroid(const int imggrid_i, double& xc_approx, double& yc_approx, const bool verbal);

	void create_lensing_matrices_from_Lmatrix(const int imggrid_i, const bool dense_Fmatrix=false, const bool potential_perturbations=false, const bool verbal=false);
	void invert_lens_mapping_dense(const int imggrid_i, bool verbal=false);
	void invert_lens_mapping_MUMPS(const int imggrid_i, double& logdet, bool verbal, bool use_copy = false);
	void invert_lens_mapping_UMFPACK(const int imggrid_i, double& logdet, bool verbal, bool use_copy = false);
	void convert_Rmatrix_to_dense();
	void convert_Rmatrix_pot_to_dense();
	void Rmatrix_determinant_MUMPS(const bool potential_perturbations);
	void Rmatrix_determinant_UMFPACK(const bool potential_perturbations);
	void matrix_determinant_dense(double& logdet, const dvector& matrix_in, const int npixels);

	void invert_lens_mapping_CG_method(const int imggrid_i, bool verbal);
	void update_source_and_lensgrid_amplitudes(const int imggrid_i, const bool verbal=false);
	void indexx(int* arr, int* indx, int nn);

	double set_required_data_pixel_window(bool verbal);

	void calculate_source_pixel_surface_brightness();
	void calculate_image_pixel_surface_brightness();
	void calculate_image_pixel_surface_brightness_dense();
	void calculate_foreground_pixel_surface_brightness(const int imggrid_i, const bool allow_lensed_nonshapelet_sources = true);
	void add_foreground_to_image_pixel_vector();
	void store_image_pixel_surface_brightness(const int imggrid_i=-1);
	void store_foreground_pixel_surface_brightness(const int imggrid_i=-1);
	void vectorize_image_pixel_surface_brightness(const int imggrid_i, bool use_mask = false);
	void plot_image_pixel_surface_brightness(string outfile_root, const int imggrid_i=-1);
	double pixel_log_evidence_times_two(double& chisq0, const bool verbal = false, const int ranchisq_i = 0);
	void setup_auxiliary_sourcegrids_and_point_imgs(int* src_i_list, const bool verbal);
	bool setup_cartesian_sourcegrid(const int imggrid_i, const int src_i, int& n_expected_imgpixels, const bool verbal);
	bool generate_and_invert_lensing_matrix_cartesian(const int imggrid_i, const int src_i, double& tot_wtime, double& tot_wtime0, const bool verbal);
	bool generate_and_invert_lensing_matrix_delaunay(const int imggrid_i, const int src_i, const bool potential_perturbations, const bool save_sb_gradient, double& tot_wtime, double& tot_wtime0, const bool verbal);
	void add_outside_sb_prior_penalty(const int band_number, int* src_i_list, bool& sb_outside_window, double& logev_times_two, const bool verbal);
	void add_regularization_prior_terms_to_logev(const int band_number, const int zsrc_i, double& logev_times_two, double& loglike_reg, double& regterms, const bool include_potential_perturbations = false, const bool verbal = false);
	void set_n_imggrids_to_include_in_inversion();

	bool load_pixel_grid_from_data(const int band_number);
	double invert_surface_brightness_map_from_data(double& chisq0, const bool verbal, const bool zero_verbal = false);
	void plot_image_pixel_grid(const int band_i, const int zsrc_i=-1);
	bool find_shapelet_scaling_parameters(const int i_shapelet, const int imggrid_i, const bool verbal=false);
	bool set_shapelet_imgpixel_nsplit(const int imggrid_i=-1);

	int get_shapelet_nn(const int imggrid_i=-1);

	void find_optimal_sourcegrid_for_analytic_source();
	bool create_sourcegrid_cartesian(const int band_number, const int zsrc_i, const bool verbal, const bool use_mask, const bool autogrid_from_analytic_source = true, const bool image_grid_already_exists = false, const bool use_auxiliary_srcgrid = false);
	bool create_sourcegrid_delaunay(const int src_i, const bool use_mask, const bool verbal);
	bool create_sourcegrid_from_imggrid_delaunay(const bool use_weighted_srcpixel_clustering, const int band_number, const int zsrc_i, const bool verbal=false);
	bool create_lensgrid_cartesian(const int band_number, const int zsrc_i, const int pixlens_i, const bool verbal, const bool use_mask = true);
	//void load_source_surface_brightness_grid(string source_inputfile);
	bool load_image_pixel_data(const int band_i, string image_pixel_filename_root, const double pixsize, const double pix_xy_ratio = 1.0, const double x_offset = 0.0, const double y_offset = 0.0, const int hdu_indx = 1, const bool show_fits_header = false);
	//bool make_image_surface_brightness_data();
	void plot_sbmap(const string outfile_root, dvector& xvals, dvector& yvals, dvector& zvals, const bool plot_fits = false);
	const bool output_lensed_surface_brightness(dvector& xvals, dvector& yvals, dvector& zvals, const int band_number, const bool output_fits = false, const bool plot_residual = false, bool plot_foreground_only = false, const bool omit_foreground = false, const bool show_mask_only = true, const bool normalize_residuals = false, const bool offload_to_data = false, const bool show_extended_mask = false, const bool show_foreground_mask = false, const bool show_noise_thresh = false, const bool exclude_ptimgs = false, const bool show_only_ptimgs = false, int specific_zsrc_i = -1, const bool show_only_first_order_corrections = false, const bool plot_log = false, const bool plot_current_sb = false, const bool verbose = true);

	//void plot_Lmatrix();
	//void check_Lmatrix_columns();
	//double temp_double;
	//void Swap(double& a, double& b) { temp_double = a; a = b; b = temp_double; }

	double wtime0, wtime; // for calculating wall time in parallel calculations
	bool show_wtime;

	friend class Grid;
	friend class CartesianSourceGrid;
	friend class ImagePixelGrid;
	friend struct ImageData;
	friend struct DerivedParam;
	friend class LensProfile;
	friend class SB_Profile;
	friend class MGE;
	friend class Cosmology;

	QLens(Cosmology* cosmo_in = NULL);
	QLens(QLens *lens_in);
	void setup_parameters(const bool initial_setup); 
	void update_meta_parameters(const bool varied_only_fitparams) {}
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
	double potential(const double&, const double&, double* zfacs, double** betafacs);
	void deflection(const double&, const double&, lensvector&, const int &thread, double* zfacs, double** betafacs);
	void deflection(const double& x, const double& y, double& def_tot_x, double& def_tot_y, const int &thread, double* zfacs, double** betafacs);
	void map_to_lens_plane(const int& redshift_i, const double& x, const double& y, lensvector& xi, const int &thread, double* zfacs, double** betafacs);
	void hessian(const double&, const double&, lensmatrix&, const int &thread, double* zfacs, double** betafacs);
	void hessian_weak(const double&, const double&, lensmatrix&, const int &thread, double* zfacs);
	void find_sourcept(const lensvector& x, lensvector& srcpt, const int &thread, double* zfacs, double** betafacs);
	void find_sourcept(const lensvector& x, double& srcpt_x, double& srcpt_y, const int &thread, double* zfacs, double** betafacs);
	void kappa_inverse_mag_sourcept(const lensvector& x, lensvector& srcpt, double &kap_tot, double &invmag, const int &thread, double* zfacs, double** betafacs);
	void sourcept_jacobian(const lensvector& xvec, lensvector& srcpt, lensmatrix& jac_tot, const int &thread, double* zfacs, double** betafacs);

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

	void record_singular_points(double *zfacs);

	// the following functions and objects are contained in commands.cpp
	char *buffer;
	int nullflag, buffer_length;
	string line;
	std::vector<string> lines;
	std::ifstream infile_list[10];
	std::ifstream *infile; // used to read commands from an input file
	int n_infiles;
	bool verbal_mode;
	bool quit_after_error;
	int nwords;
	std::vector<string> words;
	std::stringstream* ws;
	std::stringstream datastream;
	bool read_from_file;
	bool paused_while_reading_file;
	bool quit_after_reading_file;
	void process_commands(bool read_file);
	bool read_command(bool show_prompt);
	bool check_vary_z();
	bool read_egrad_params(const bool vary_par, const int egrad_mode, dvector& efunc_params, int& nparams_to_vary, boolvector& varyflags, const int default_nparams, const double xc, const double yc, ParamAnchor* parameter_anchors, int& parameter_anchor_i, int& n_bspline_coefs, dvector& knots, double& ximin, double& ximax, double& xiref, bool& linear_xivals, bool& enter_params_and_varyflags, bool& enter_knots);
	bool read_fgrad_params(const bool vary_par, const int egrad_mode, const int n_fmodes, const std::vector<int> fourier_mvals, dvector& fgrad_params, int& nparams_to_vary, boolvector& varyflags, const int default_nparams, ParamAnchor* parameter_anchors, int& parameter_anchor_i, int n_bspline_coefs, dvector& knots, const bool enter_params_and_varyflags, const bool enter_knots);
	void run_plotter(string plotcommand, string extra_command = "");
	void run_plotter_file(string plotcommand, string filename, string range = "", string extra_command = "", string extra_command2 = "");
	void run_plotter_range(string plotcommand, string range, string extra_command = "", string extra_command2 = "");
	void run_mkdist(bool copy_post_files, string posts_dirname, const int nbins_1d, const int nbins_2d, bool copy_subplot_only, bool resampled_posts, bool no2dposts, bool nohists);
	void make_histograms(bool copy_post_files, string posts_dirname, const int nbins_1d, const int nbins_2d, bool copy_subplot_only, bool resampled_posts, bool no2dposts, bool nohists, bool use_fisher_matrix);
	void remove_equal_sign();
	void remove_word(int n_remove);
	void remove_comments(string& instring);
	void remove_equal_sign_datafile(std::vector<string>& datawords, int& n_datawords);

	void set_show_wtime(bool show_wt) { show_wtime = show_wt; }
	void set_verbal_mode(bool echo) { verbal_mode = echo; }
	bool open_script_file(const string filename);
	void set_quit_after_reading_file(bool setting) { quit_after_reading_file = setting; }
	void set_suppress_plots(bool setting) { suppress_plots = setting; }

	void extract_word_starts_with(const char initial_character, int starting_word, int ending_word, string& extracted_word);
	void extract_word_starts_with(const char initial_character, int starting_word, string& extracted_word) { extract_word_starts_with(initial_character,starting_word,1000,extracted_word); }
	bool extract_word_starts_with(const char initial_character, int starting_word, int ending_word, std::vector<string>& extracted_words);
	void set_quit_after_error(bool arg) { quit_after_error = arg; }
	void set_plot_title(int starting_word, string& temp_title);

	// the following functions are contained in imgsrch.cpp
	private:
	void find_images();

	public:
	bool plot_recursive_grid(const char filename[]);
	void output_images_single_source(const double &x_source, const double &y_source, bool verbal, const double flux = -1.0, const bool show_labels = false);
	bool plot_images_single_source(const double &x_source, const double &y_source, bool verbal, const double flux = -1.0, const bool show_labels = false, string imgheader = "", string srcheader = "") {
		std::ofstream imgfile; open_output_file(imgfile,"imgs.dat");
		std::ofstream srcfile; open_output_file(srcfile,"srcs.dat");
		if (!imgheader.empty()) imgfile << "\"" << imgheader << "\"" << std::endl;
		if (!srcheader.empty()) srcfile << "\"" << srcheader << "\"" << std::endl;
		return plot_images_single_source(x_source,y_source,verbal,imgfile,srcfile,flux,show_labels);
	}
	bool plot_images_single_source(const double &x_source, const double &y_source, bool verbal, std::ofstream& imgfile, std::ofstream& srcfile, const double flux = -1.0, const bool show_labels = false);
	image* get_images(const lensvector &source_in, int &n_images) { return get_images(source_in, n_images, true); }
	image* get_images(const lensvector &source_in, int &n_images, bool verbal);
	bool get_imageset(const double src_x, const double src_y, PointSource& image_set, bool verbal = true); // used by Python wrapper
	std::vector<PointSource> get_fit_imagesets(bool& status, int min_dataset = 0, int max_dataset = -1, bool verbal = true); // defined in imgsrch.cpp
	bool plot_images(const char *sourcefile, const char *imagefile, bool color_multiplicities, bool verbal);
	void lens_equation(const lensvector&, lensvector&, const int& thread, double *zfacs, double **betafacs); // Used by Newton's method to find images

	// the remaining functions in this class are all contained in lens.cpp
	void create_and_add_lens(LensProfileName, const int emode, const double zl, const double zs, const double mass_parameter, const double logslope_param, const double scale, const double core, const double q, const double theta, const double xc, const double yc, const double extra_param1 = -1000, const double extra_param2 = -1000, const int parameter_mode = 0);
	void create_and_add_lens(const char *splinefile, const int emode, const double zl, const double zs, const double q, const double theta, const double qx, const double f, const double xc, const double yc);
	void add_shear_lens(const double zl, const double zs, const double shear, const double theta, const double xc, const double yc); // specific version for shear model
	void add_ptmass_lens(const double zl, const double zs, const double mass_parameter, const double xc, const double yc, const int pmode); // specific version for ptmass model
	void add_mass_sheet_lens(const double zl, const double zs, const double mass_parameter, const double xc, const double yc); // specific version for mass sheet
	void add_multipole_lens(const double zl, const double zs, int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool kap, bool sine_term);
	void add_tabulated_lens(const double zl, const double zs, int lnum, const double kscale, const double rscale, const double theta, const double xc, const double yc);
	bool add_tabulated_lens_from_file(const double zl, const double zs, const double kscale, const double rscale, const double theta, const double xc, const double yc, const string tabfileroot);
	bool add_qtabulated_lens_from_file(const double zl, const double zs, const double kscale, const double rscale, const double q, const double theta, const double xc, const double yc, const string tabfileroot);
	bool save_tabulated_lens_to_file(int lnum, const string tabfileroot);
	void add_qtabulated_lens(const double zl, const double zs, int lnum, const double kscale, const double rscale, const double q, const double theta, const double xc, const double yc);
	bool spawn_lens_from_source_object(const int src_number, const double zl, const double zs, const int pmode, const bool vary_mass_parameter, const bool include_limits, const double mass_param_lower, const double mass_param_upper);

	void add_lens(LensProfile *new_lens);
	void add_new_lens_redshift(const double zl, const int lens_i, int* zlens_idx);
	void remove_old_lens_redshift(const int znum, const int lens_i, const bool removed_lens);
	void update_lens_redshift_data();
	int add_new_extended_src_redshift(const double zs, const int src_i, const bool pixellated_src);
	void remove_old_extended_src_redshift(const int znum, const bool removing_pixellated_src);
	bool assign_mask(const int band, const int znum, const int mask_i);
	void print_mask_assignments();
	int add_new_ptsrc_redshift(const double zs, const int src_i);
	void remove_old_ptsrc_redshift(const int znum);
	void update_ptsrc_redshift_data();

	void add_new_lens_entry(const double zl);
	void set_primary_lens();
	bool set_primary_lens_number(const int lensnum) {
		if ((lensnum < 0) or (lensnum >= nlens)) return false;
		if ((include_secondary_lens) and (secondary_lens_number==lensnum)) {
			secondary_lens_number = primary_lens_number; // so we switch primary and secondary
		}
		primary_lens_number = lensnum;
		auto_set_primary_lens = false;
		return true;
	}
	bool set_secondary_lens_number(const int lensnum) {
		if ((lensnum < 0) or (lensnum >= nlens)) return false;
		if (lensnum==primary_lens_number) return false;
		secondary_lens_number = lensnum;
		include_secondary_lens = true;
		return true;
	}
	int get_primary_lens_number() { return primary_lens_number; }
	int get_secondary_lens_number() { return secondary_lens_number; }
	void print_zfactors_and_beta_matrices();
	void set_source_redshift(const double zsrc);
	double get_source_redshift() { return source_redshift; }
	void set_reference_source_redshift(const double zsrc);
	double get_reference_source_redshift() { return reference_source_redshift; }
	void update_zfactors_and_betafactors();
	void recalculate_beta_factors();
	void set_sci_notation(const bool scinot) {
		use_scientific_notation = scinot;
		if (use_scientific_notation) std::cout << std::setiosflags(std::ios::scientific);
		else {
			std::cout << std::resetiosflags(std::ios::scientific);
			std::cout.unsetf(std::ios_base::floatfield);
		}
	}
	bool get_sci_notation() { return use_scientific_notation; }

	bool register_lens_vary_parameters(const int lensnumber);
	void register_lens_prior_limits(const int lens_number);
	void update_lens_fitparams(const int lens_number);

	bool register_sb_vary_parameters(const int sbnumber);
	void register_sb_prior_limits(const int sb_number);
	void update_sb_fitparams(const int sb_number);

	bool register_pixellated_src_vary_parameters(const int pixsrc_number);
	void register_pixellated_src_prior_limits(const int pixsrc_number);
	void update_pixellated_src_fitparams(const int pixsrc_number);

	bool set_pixellated_lens_vary_parameters(const int pixlens_number, boolvector &vary_flags);

	bool register_ptsrc_vary_parameters(const int src_number);
	void register_ptsrc_prior_limits(const int ptsrc_number);
	void update_ptsrc_fitparams(const int ptsrc_number);

	bool set_psf_vary_parameters(const int psf_number, boolvector &vary_flags);

	bool register_cosmo_vary_parameters();
	void register_cosmo_prior_limits();
	void update_cosmo_fitparams();

	bool update_pixellated_src_varyflag(const int src_number, const string name, const bool flag);
	bool update_pixellated_lens_varyflag(const int pixlens_number, const string name, const bool flag);
	bool update_ptsrc_varyflag(const int ptsrc_number, const string name, const bool flag);
	bool update_psf_varyflag(const int psf_number, const string name, const bool flag);

	bool update_cosmo_varyflag(const string name, const bool flag);
	bool update_misc_varyflag(const string name, const bool flag);

	bool update_parameter_list(const bool check_current_params = false);
	void update_anchored_parameters_and_redshift_data();
	bool update_lens_centers_from_pixsrc_coords();

	void reassign_lensparam_pointers_and_names(const bool reset_plimits = true);
	void reassign_sb_param_pointers_and_names();
	void print_lens_list(bool show_vary_params);
	LensProfile* get_lens_pointer(const int lensnum) { if (lensnum >= nlens) return NULL; else return lens_list[lensnum]; }
	void print_fit_model();
	void print_lens_cosmology_info(const int lmin, const int lmax);
	bool output_mass_r(const double r_arcsec, const int lensnum, const bool use_kpc);
	double mass2d_r(const double r_arcsec, const int lensnum, const bool use_kpc);
	double mass3d_r(const double r_arcsec, const int lensnum, const bool use_kpc);
	double calculate_average_log_slope(const int lensnum, const double rmin, const double rmax, const bool use_kpc);

	void add_source(SB_Profile* new_src, const bool is_lensed);
	void create_and_add_source_object(SB_ProfileName name, const bool is_lensed, const int band_number, const double zsrc_in, const int emode, const double sb_norm, const double scale, const double scale2, const double logslope_param, const double q, const double theta, const double xc, const double yc, const double special_param1 = -1, const double special_param2 = -1, const int pmode = 0);
	void create_and_add_splined_source_object(const char *splinefile, const bool is_lensed, const int band_number, const double zsrc_in, const int emode, const double q, const double theta, const double qx, const double f, const double xc, const double yc);
	void create_and_add_multipole_source(const bool is_lensed, const int band_number, const double zsrc_in, int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool sine_term);
	void create_and_add_shapelet_source(const bool is_lensed, const int band_number, const double zsrc_in, const double amp00, const double sig_x, const double q, const double theta, const double xc, const double yc, const int nmax, const bool truncate, const int pmode = 0);
	void create_and_add_mge_source(const bool is_lensed, const int band_number, const double zsrc_in, const double reg, const double amp0, const double sig_i, const double sig_f, const double q, const double theta, const double xc, const double yc, const int nmax, const int pmode = 0);


	void remove_source_object(int sb_number, const bool delete_src = true);
	void clear_source_objects();
	void print_source_list(bool show_vary_params);

	void add_new_model_band();
	void remove_model_band(const int band_number, const bool removing_pixellated_src);

	void add_pixellated_source(const double zsrc, const int band_number = 0);
	void remove_pixellated_source(int src_number, const bool delete_pixsrc = true);
	void print_pixellated_source_list(bool show_vary_params);
	void find_pixellated_source_moments(const int npix, double& qs, double& phi_s, double& sigavg, double& xavg, double& yavg);

	bool add_pixellated_lens(const double zlens);
	void remove_pixellated_lens(int pixlens_number);
	void print_pixellated_lens_list(bool show_vary_params);

	void add_point_source(const double zsrc, const lensvector& sourcept, const bool vary_source_coords = true);
	void remove_point_source(int src_number);
	void print_point_source_list(bool show_vary_params);

	void add_psf();
	void remove_psf(int psf_number);
	void print_psf_list(bool show_vary_params);

	void add_image_pixel_data();
	void remove_image_pixel_data(int band_number);

	//void add_derived_param(DerivedParamType type_in, double param, int lensnum, double param2 = -1e30, bool use_kpc = false);
	//void remove_derived_param(int dparam_number);
	//void rename_derived_param(int dparam_number, string newname, string new_latex_name);
	//void clear_derived_params();
	//void print_derived_param_list();
	void clear_raw_chisq() { raw_chisq = -1e30; if (fitmodel) fitmodel->raw_chisq = -1e30; }

	bool create_grid(bool verbal, double *zfacs, double **betafacs, const int redshift_index = -1); // the last (optional) argument indicates which images are being fit to; used to optimize the subgridding
	void find_automatic_grid_position_and_size(double *zfacs);
	void clear_lenses();
	void clear();
	void reset_grid();
	void remove_lens(int lensnumber, const bool delete_lens = true);
	void toggle_major_axis_along_y(bool major_axis_along_y);
	void toggle_major_axis_along_y_src(bool major_axis_along_y);
	void create_output_directory();
	void open_output_file(std::ofstream &outfile, string filename_in);
	void open_input_file(std::ifstream &infile, string filename_in);
	void open_output_file(std::ofstream &outfile, char* filechar_in);

	private:
	bool temp_auto_store_cc_points, temp_include_time_delays;
	bool fit_set_optimizations();
	void fit_restore_defaults();
	double zfac_re; // used by einstein_radius_root(...)

	public:
	double chi_square_fit_simplex(const bool show_parameter_errors);
	double chi_square_fit_powell(const bool show_parameter_errors);
	void output_fit_results(dvector& stepsizes, const double chisq_bestfit, const int chisq_evals, const bool calculate_parameter_errors);
	void nested_sampling();
	void polychord(const bool resume_previous, const bool skip_run);
	void multinest(const bool resume_previous, const bool skip_run);
	void chi_square_twalk();
	bool add_dparams_to_chain(string file_ext);
	bool adopt_bestfit_point_from_chain();
	bool load_bestfit_model(const bool custom_filename=false, string fit_filename="");

	bool adopt_point_from_chain(const unsigned long point_num);
	bool adopt_point_from_chain_paramrange(const int paramnum, const double minval, const double maxval);
	bool plot_kappa_profile_percentiles_from_chain(int lensnum, double rmin, double rmax, int nbins, const string kappa_filename);
	bool output_scaled_percentiles_from_chain(const double pct_scaling);
	bool get_stepsizes_from_percentiles(const double pct_scaling, dvector& stepsizes);
	bool find_scaled_percentiles_from_chain(const double pct_scaling, double *scaled_lopct, double *scaled_hipct);
	double find_percentile(const unsigned long npoints, const double pct, const double tot, double *pts, double *weights);
	bool output_scaled_percentiles_from_egrad_fits(const int srcnum, const double xcavg, const double ycavg, const double qtheta_pct_scaling = 1.0, const double fmode_pct_scaling = 1.0, const bool include_m3_fmode = false, const bool include_m4_fmode = false);
	bool output_egrad_values_and_knots(const int srcnum,const string suffix);

	bool output_coolest_files(const string filename);

	void plot_chisq_2d(const int param1, const int param2, const int n1, const double i1, const double f1, const int n2, const double i2, const double f2);
	void plot_chisq_1d(const int param, const int n, const double i, const double f, string filename);
	double chisq_single_evaluation(bool init_fitmodel, bool show_total_wtime, bool showdiag, bool show_status, bool show_lensinfo = false);
	//bool setup_fit_parameters(const bool ignore_limits = false);
	//bool setup_limits();
	void get_n_fit_parameters(int &nparams);
	void get_all_parameter_names(vector<string>& fit_parameter_names, vector<string>& latex_parameter_names);
	bool get_lens_parameter_numbers(const int lens_i, int& pi, int& pf);
	bool get_sb_parameter_numbers(const int lens_i, int& pi, int& pf);
	bool get_pixsrc_parameter_numbers(const int pixsrc_i, int& pi, int& pf);
	bool get_pixlens_parameter_numbers(const int pixlens_i, int& pi, int& pf);
	bool get_ptsrc_parameter_numbers(const int ptsrc_i, int& pi, int& pf);
	bool get_psf_parameter_numbers(const int psf_i, int& pi, int& pf);
	bool get_cosmo_parameter_numbers(int& pi, int& pf);
	bool get_misc_parameter_numbers(int& pi, int& pf);
	bool lookup_parameter_value(const string pname, double& pval);
	void create_parameter_value_string(string &pvals);
	//bool output_parameter_values();
	//bool output_parameter_prior_ranges();
	void update_pixsrc_active_parameters(const int src_number);
	void update_pixlens_active_parameters(const int pixlens_number);
	void update_ptsrc_active_parameters(const int src_number);
	void update_active_parameters(ModelParams* param_object, const int pi);

	void get_automatic_initial_stepsizes(dvector& stepsizes);
	void set_default_plimits();
	bool initialize_fitmodel(const bool running_fit_in);
	double update_model(const double* params);
	void update_prior_limits(const double* lower, const double* upper, const bool* changed_limits);
	double fitmodel_loglike_point_source(double* params);
	double fitmodel_loglike_extended_source(double* params);
	double fitmodel_custom_prior();
	double LogLikeFunc(double *params) { return (this->*LogLikePtr)(params); }
	double LogLikeVecFunc(vector<double>& params) { return (this->*LogLikePtr)(params.data()); }
	void DerivedParamFunc(double *params, double *dparams) { (this->*DerivedParamPtr)(params,dparams); }
	void fitmodel_calculate_derived_params(double* params, double* derived_params);
	double get_lens_parameter_using_pmode(const int lensnum, const int paramnum, const int pmode = -1);
	double loglike_point_source(double* params);
	bool calculate_fisher_matrix(double *params, const dvector &stepsizes);
	double loglike_deriv(const dvector &params, const int index, const double step);
	void output_bestfit_model(const bool show_parameter_errors = false);
	bool adopt_model(dvector &fitparams);

	bool include_central_image;
	bool include_imgpos_chisq, include_flux_chisq, include_time_delay_chisq;
	bool include_weak_lensing_chisq;
	bool use_analytic_bestfit_src;
	bool include_ptsrc_shift;
	bool n_images_penalty;
	bool analytic_source_flux;
	bool include_imgfluxes_in_inversion;
	bool include_srcflux_in_inversion;

	bool spline_critical_curves(bool verbal = true);
	bool plot_critical_curves(string filename = "");
	bool find_caustic_minmax(double& min, double& max, double& max_minor_axis, const double cc_num = 0);
	bool plotcrit_exclude_subhalo(string filename, int exclude_lensnum)
	{
		bool worked = false;
		double mvir;
		if (lens_list[exclude_lensnum]->get_specific_parameter("mvir",mvir)==true) {
			lens_list[exclude_lensnum]->update_specific_parameter("mvir",1e-3);
			worked = plot_critical_curves(filename);
			lens_list[exclude_lensnum]->update_specific_parameter("mvir",mvir);
		}
		return worked;
	}
	void plot_ray_tracing_grid(double xmin, double xmax, double ymin, double ymax, int x_N, int y_N, string filename);

	void make_source_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename);
	void make_source_ellipse(const double xcenter, const double ycenter, const double major_axis, const double q, const double angle, const int n_subellipses, const int points_per_ellipse, const bool draw_in_imgplane, string source_filename);
	void raytrace_image_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename);

	void plot_kappa_profile(int l, double rmin, double rmax, int steps, const char *kname, const char *kdname = NULL);
	void plot_total_kappa(const double rmin, const double rmax, const int steps, const char *kname, const char *kdname = NULL);
	void output_total_kappa(const double rmin, const double rmax, const int steps, dvector& rvals, dvector& kappavals, dvector& dkappavals);
	void plot_sb_profile(int l, double rmin, double rmax, int steps, const char *sname);
	void plot_total_sbprofile(double rmin, double rmax, int steps, const char *sbname);
	double total_kappa(const double r, const int lensnum, const bool use_kpc);
	double total_dlogkappa(const double r, const int lensnum, const bool use_kpc);
	double einstein_radius_single_lens(const double src_redshift, const int lensnum);
	double get_xi_parameter(const double src_redshift, const int lensnum);
	double get_total_xi_parameter(const double src_redshift);
	bool *centered;
	double einstein_radius_of_primary_lens(const double zfac, double& reav);
	double einstein_radius_root(const double r);
	double get_einstein_radius_prior(const bool verbal);
	void plot_mass_profile(double rmin, double rmax, int steps, const char *massname);
	void print_lensing_info_at_point(const double x, const double y);

	double chisq_pos_source_plane();
	double chisq_pos_image_plane();
	double chisq_pos_image_plane_diagnostic(const bool verbose, const bool output_residuals_to_file, double& rms_imgpos_err, int& n_matched_images, const string output_filename = "fit_chivals.dat");

	double chisq_flux();
	double chisq_time_delays();
	double chisq_time_delays_from_model_imgs();
	double chisq_weak_lensing();
	bool output_weak_lensing_chivals(string filename);
	void find_analytic_srcflux(double *bestfit_flux);
	void find_analytic_srcpos(lensvector *beta_i);
	void set_analytic_sourcepts(const bool verbal = false);
	void set_analytic_srcflux(const bool verbal = false);
	double get_avg_ptsrc_dist(const int ptsrc_i);

	//static bool respline_at_end;
	//static int resplinesteps;
	//void create_deflection_spline(int steps);
	//void spline_deflection(double xl, double yl, int steps);
	//bool autospline_deflection(int steps);
	//bool unspline_deflection();
	bool isspherical();
	void set_grid_corners(double xmin, double xmax, double ymin, double ymax);
	void set_grid_from_pixels();
	void set_img_npixels(const int npix_x, const int npix_y);

	void set_gridsize(double xl, double yl);
	void set_gridcenter(double xc, double yc);
	void autogrid(double rmin, double rmax, double frac);
	void autogrid(double rmin, double rmax);
	void autogrid();
	bool get_deflection_spline_info(double &xmax, double &ymax, int &nsteps);
	void set_Gauss_NN(const int& nn);
	void set_integral_tolerance(const double& acc);
	void set_integral_convergence_warnings(const bool warn);

	void set_integration_method(IntegrationMethod method) { LensProfile::integral_method = method; }
	void set_analytic_bestfit_src(bool setting) {
		use_analytic_bestfit_src = setting;
		bool vary_source_coords = (use_analytic_bestfit_src) ? false : true;
		// Automatically turn source coordinate parameters on/off accordingly
		for (int i=0; i < n_ptsrc; i++) {
			update_ptsrc_varyflag(i,"xsrc",vary_source_coords);
			update_ptsrc_varyflag(i,"ysrc",vary_source_coords);
		}
		for (int i=0; i < n_ptsrc; i++) {
			update_ptsrc_active_parameters(i);
		}
		update_parameter_list();
	}
	void get_analytic_bestfit_src(bool &setting) { setting = use_analytic_bestfit_src; }

	void update_imggrid_mask_values(const int mask_i);

	void set_warnings(bool setting) { warnings = setting; }
	void get_warnings(bool &setting) { setting = warnings; }
	void set_newton_warnings(bool setting) { newton_warnings = setting; }
	void get_newton_warnings(bool &setting) { setting = newton_warnings; }
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
			for (int i=0; i < n_ptsrc; i++) ptsrc_list[i]->set_include_limits(false);
			for (int i=0; i < n_sb; i++) sb_list[i]->set_include_limits(false);
			for (int i=0; i < n_pixellated_src; i++) srcgrids[i]->set_include_limits(false);
		} else {
			for (int i=0; i < nlens; i++) lens_list[i]->set_include_limits(true);
			for (int i=0; i < n_ptsrc; i++) ptsrc_list[i]->set_include_limits(true);
			for (int i=0; i < n_sb; i++) sb_list[i]->set_include_limits(true);
			for (int i=0; i < n_pixellated_src; i++) srcgrids[i]->set_include_limits(true);
		}
	}
	void transform_cube(double* params, double* Cube);
	bool get_einstein_radius(int lens_number, double& re_major_axis, double& re_average);

	/*
	// specialty functions...now contained in 'specialized.cpp', but not compiled with by default
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
	*/

	void test_lens_functions();
};

class LensList
{
	public:
	QLens* qlens;
	int nlens;
	LensProfile** lenslistptr;
	LensList(QLens* qlens_in) { lenslistptr = NULL; qlens = qlens_in; nlens = 0; }
	void input_ptr(LensProfile** ptr_in, const int nlens_in) { lenslistptr = ptr_in; nlens = nlens_in; }
	void clear_ptr() { lenslistptr = NULL; nlens = 0; }
	void add_lens(LensProfile* lens_in) {
		qlens->add_lens(lens_in);
	}
	void add_lens_extshear(LensProfile* lens_in, Shear* extshear) {
		qlens->add_lens(lens_in);
		qlens->add_lens((LensProfile*) extshear);
		  lenslistptr[nlens-1]->anchor_center_to_lens(nlens-2);
	}
	bool clear(const int min_loc=-1, const int max_loc=-1) {
		if((min_loc == -1) and (max_loc == -1)) {
			for (int i=nlens-1; i >= 0; i--) {
				qlens->remove_lens(i,false);
			}
		} else if ((min_loc != -1) and (max_loc == -1)) {
			if (min_loc >= nlens) { warn("specified lens index does not exist"); return false; }
			qlens->remove_lens(min_loc,false);
		} else {
			if (((min_loc < 0) or (min_loc >= nlens)) or ((max_loc < 0) or (max_loc >= nlens))) { warn("specified lens index does not exist"); return false; }
			if (min_loc > max_loc) { warn("max index must be greater than min index"); return false; }
			for (int i=max_loc; i >= min_loc; i--) {
				qlens->remove_lens(i,false);
			}
		}
		return true;
	}
	bool anchor_lens_center(const int lens_num1, const int lens_num2) {
		if ((lens_num1 >= nlens) or (lens_num2 >= nlens)) return false;
		lenslistptr[lens_num1]->anchor_center_to_lens(lens_num2);
		return true;
	}
};

class SourceList
{
	protected:
	QLens* qlens;

	public:
	int n_sb;
	SB_Profile** srclistptr;
	SourceList(QLens* qlens_in) { srclistptr = NULL; qlens = qlens_in; n_sb = 0; }
	void input_ptr(SB_Profile** ptr_in, const int n_sb_in) { srclistptr = ptr_in; n_sb = n_sb_in; }
	void clear_ptr() { srclistptr = NULL; n_sb = 0; }
	void add_source(SB_Profile* src_in, const bool is_lensed) {
		qlens->add_source(src_in,is_lensed);
	}
	bool clear(const int min_loc=-1, const int max_loc=-1) {
		if((min_loc == -1) and (max_loc == -1)) {
			for (int i=n_sb-1; i >= 0; i--) {
				qlens->remove_source_object(i,false);
			}
		} else if ((min_loc != -1) and (max_loc == -1)) {
			if (min_loc >= n_sb) { warn("specified source index does not exist"); return false; }
			qlens->remove_source_object(min_loc,false);
		} else {
			if (((min_loc < 0) or (min_loc >= n_sb)) or ((max_loc < 0) or (max_loc >= n_sb))) { warn("specified source index does not exist"); return false; }
			if (min_loc > max_loc) { warn("max index must be greater than min index"); return false; }
			for (int i=max_loc; i >= min_loc; i--) {
				qlens->remove_source_object(i,false);
			}
		}
		return true;
	}
	bool anchor_center_to_source(const int src_num1, const int src_num2) {
		if ((src_num1 >= n_sb) or (src_num2 >= n_sb)) return false;
		srclistptr[src_num1]->anchor_center_to_source(srclistptr,src_num2);
		return true;
	}
	bool anchor_center_to_lens(const int src_num, const int lens_num) {
		if ((src_num >= n_sb) or (lens_num >= qlens->nlens)) return false;
		srclistptr[src_num]->anchor_center_to_lens(qlens->lens_list,lens_num);
		return true;
	}
};

class PixSrcList
{
	public:
	QLens* qlens;
	int n_pixsrc;
	ModelParams** pixsrclist_ptr;
	PixSrcList(QLens* qlens_in) { pixsrclist_ptr = NULL; qlens = qlens_in; n_pixsrc = 0; }
	void input_ptr(ModelParams** ptr_in, const int n_pixsrc_in) { pixsrclist_ptr = ptr_in; n_pixsrc = n_pixsrc_in; }
	void clear_ptr() { pixsrclist_ptr = NULL; n_pixsrc = 0; }
	void add_pixsrc(const double zsrc, const int band) {
		qlens->add_pixellated_source(zsrc,band);
	}
	bool clear(const int min_loc=-1, const int max_loc=-1) {
		if((min_loc == -1) and (max_loc == -1)) {
			for (int i=n_pixsrc-1; i >= 0; i--) {
				qlens->remove_pixellated_source(i,false);
			}
		} else if ((min_loc != -1) and (max_loc == -1)) {
			if (min_loc >= n_pixsrc) { warn("specified source index does not exist"); return false; }
			qlens->remove_pixellated_source(min_loc,false);
		} else {
			if (((min_loc < 0) or (min_loc >= n_pixsrc)) or ((max_loc < 0) or (max_loc >= n_pixsrc))) { warn("specified source index does not exist"); return false; }
			if (min_loc > max_loc) { warn("max index must be greater than min index"); return false; }
			for (int i=max_loc; i >= min_loc; i--) {
				qlens->remove_pixellated_source(i,false);
			}
		}
		return true;
	}
};

class PtSrcList
{
	public:
	QLens* qlens;
	int n_ptsrc;
	PointSource** ptsrclist_ptr;
	PtSrcList(QLens* qlens_in) { ptsrclist_ptr = NULL; qlens = qlens_in; n_ptsrc = 0; }
	void input_ptr(PointSource** ptr_in, const int n_ptsrc_in) { ptsrclist_ptr = ptr_in; n_ptsrc = n_ptsrc_in; }
	void clear_ptr() { ptsrclist_ptr = NULL; n_ptsrc = 0; }
	void add_ptsrc(const double zsrc, const lensvector &sourcept, const bool vary_source_coords = false) {
		qlens->add_point_source(zsrc,sourcept,vary_source_coords);
	}
	bool clear(const int min_loc=-1, const int max_loc=-1) {
		if((min_loc == -1) and (max_loc == -1)) {
			for (int i=n_ptsrc-1; i >= 0; i--) {
				qlens->remove_point_source(i);
			}
		} else if ((min_loc != -1) and (max_loc == -1)) {
			if (min_loc >= n_ptsrc) { warn("specified source index does not exist"); return false; }
			qlens->remove_point_source(min_loc);
		} else {
			if (((min_loc < 0) or (min_loc >= n_ptsrc)) or ((max_loc < 0) or (max_loc >= n_ptsrc))) { warn("specified source index does not exist"); return false; }
			if (min_loc > max_loc) { warn("max index must be greater than min index"); return false; }
			for (int i=max_loc; i >= min_loc; i--) {
				qlens->remove_point_source(i);
			}
		}
		return true;
	}
};

class ImgDataList
{
	public:
	QLens* qlens;
	int n_data_bands;
	ImageData** imgdatalist_ptr;
	ImgDataList(QLens* qlens_in) { imgdatalist_ptr = NULL; qlens = qlens_in; n_data_bands = 0; }
	void input_ptr(ImageData** ptr_in, const int n_data_bands_in) { imgdatalist_ptr = ptr_in; n_data_bands = n_data_bands_in; }
	void clear_ptr() { imgdatalist_ptr = NULL; n_data_bands = 0; }
	void add_band() {
		qlens->add_image_pixel_data();
	}
	bool load_imgdata(const int band_i, string filename, const double pixsize, const double pix_xy_ratio=1.0, const double x_offset=0.0, const double y_offset=0.0, const int hdu_indx=1, const bool show_fits_header=false) {
		return (qlens->load_image_pixel_data(band_i,filename,pixsize,pix_xy_ratio,x_offset,y_offset,hdu_indx,show_fits_header));
	}
	bool clear(const int min_loc=-1, const int max_loc=-1) {
		if((min_loc == -1) and (max_loc == -1)) {
			for (int i=n_data_bands-1; i >= 0; i--) {
				qlens->remove_image_pixel_data(i);
			}
		} else if ((min_loc != -1) and (max_loc == -1)) {
			if (min_loc >= n_data_bands) { warn("specified source index does not exist"); return false; }
			qlens->remove_image_pixel_data(min_loc);
		} else {
			if (((min_loc < 0) or (min_loc >= n_data_bands)) or ((max_loc < 0) or (max_loc >= n_data_bands))) { warn("specified source index does not exist"); return false; }
			if (min_loc > max_loc) { warn("max index must be greater than min index"); return false; }
			for (int i=max_loc; i >= min_loc; i--) {
				qlens->remove_image_pixel_data(i);
			}
		}
		return true;
	}
};

struct PointImageData
{
	int n_images;
	lensvector *pos;
	double *flux;
	double *time_delays;
	double *sigma_pos, *sigma_f, *sigma_t;
	bool *use_in_chisq;
	double max_distsqr; // maximum squared distance between any pair of images
	double max_tdsqr; // max squared difference between any pair of time delays
	PointImageData() { n_images = 0; }
	void input(const int &nn);
	void input(const PointImageData& imgs_in);
	//void input(const int &nn, image* images, const double sigma_pos_in, const double sigma_flux_in, const double sigma_td_in, bool* include, bool include_time_delays);
	void input(const int &nn, image* images, double* sigma_pos_in, double* sigma_flux_in, const double sigma_td_in, bool* include, bool include_time_delays);
	void add_image(lensvector& pos_in, const double sigma_pos_in, const double flux_in, const double sigma_f_in, const double time_delay_in, const double sigma_t_in);
	void print_list(bool print_errors, bool use_sci);
	void write_to_file(std::ofstream &outfile);
	bool set_use_in_chisq(int image_i, bool use_in_chisq_in);
	~PointImageData();
};

#endif // QLENS_H
