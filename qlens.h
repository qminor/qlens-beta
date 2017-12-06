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
#define USE_COMM_WORLD -987654

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
enum SourceFitMode { Point_Source, Pixellated_Source, Parameterized_Source };
enum Prior { UNIFORM_PRIOR, LOG_PRIOR, GAUSS_PRIOR };
enum Transform { NONE, LOG_TRANSFORM, GAUSS_TRANSFORM, LINEAR_TRANSFORM };
enum RayTracingMethod {
	Area_Overlap,
	Interpolate,
	Area_Interpolation
};

class Lens;			// Defined after class Grid
class SourcePixelGrid;
class ImagePixelGrid;
class Defspline;	// ...
struct ImageData;
struct ImagePixelData;
struct ParamSettings;

struct image {
	lensvector pos;
	double mag, td;
	int parity;
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
	static Lens* lens;
	static int nthreads;
	Grid* neighbor[4]; // 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	Grid* parent_cell;
	Grid** search_subcells;

	static double zfactor; // kappa ratio used for modeling source points at different redshifts
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
	bool *corner_parity[4];
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

	// Used for image searching. If you ever multi-thread, be careful about making these static variables
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

	static lensvector *fvec;
	bool LineSearch(lensvector& xold, double fold, lensvector& g, lensvector& p, lensvector& x, double& f, double stpmax, bool &check, const int& thread);
	bool NewtonsMethod(lensvector& x, bool &check, const int& thread);
	void SolveLinearEqs(lensmatrix&, lensvector&);
	bool redundancy(const lensvector&);
	double max_component(const lensvector&);

	static const int max_iterations, max_step_length;
	static bool *newton_check;
	// make these multithread-safe when you are ready to multithread the image searching
	static bool finished_search;
	static int nfound_max, nfound_pos, nfound_neg;
	static image images[];

public:
	Grid(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double zfactor_in); 
	Grid(double xcenter_in, double ycenter_in, double xlength, double ylength, double zfactor_in);
	void redraw_grid(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double zfactor_in);
	void redraw_grid(double xcenter_in, double ycenter_in, double xlength, double ylength, double zfactor_in);
	void reassign_coordinates(lensvector** xij, const int& i, const int& j, const int& level_in, Grid* parent_ptr);

	static void set_splitting(int rs0, int ts0, int sl, int ccsl, double max_cs, bool neighbor_split);
	static void allocate_multithreaded_variables(const int& threads);
	static void deallocate_multithreaded_variables();
	static void reset_search_parameters();
	~Grid();

	static int nfound;
	static double image_pos_accuracy;
	static double redundancy_separation_threshold;
	static double warning_magnification_threshold;
	image* tree_search();
	static void set_lens(Lens* lensptr) { lens = lensptr; }
	void subgrid_around_galaxies(lensvector* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_splittings);
	void subgrid_around_galaxies_iteration(lensvector* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_split, bool cc_neighbor_splitting);

	void galsubgrid();
	void store_critical_curve_pts();
	static void set_imagepos_accuracy(const double& setting) {
		image_pos_accuracy = setting;
		redundancy_separation_threshold = 10*setting;
	}
	static void set_enforce_min_area(const bool& setting) { enforce_min_area = setting; }

	// for plotting the grid to a file:
	static ofstream xgrid;
	void plot_corner_coordinates();
	void get_usplit_initial(int &setting) { setting = u_split_initial; }
	void get_wsplit_initial(int &setting) { setting = w_split_initial; }
};

class Lens : public Cosmology, public Brent, public Sort, public Powell, public Simplex, public UCMC
{
	private:
	// These are arrays of dummy variables used for lensing calculations, arranged so that each thread gets its own set of dummy variables.
	static lensvector *defs, *defs_i;
	static lensmatrix *jacs, *hesses, *hesses_i;
	static int *indxs;

	int chisq_it;
	ofstream logfile;
	bool show_wtime;

	protected:
	int mpi_id, mpi_np, mpi_ngroups, group_id, group_num, group_np;
#ifdef USE_MPI
	MPI_Comm *group_comm;
	MPI_Comm *my_comm;
	MPI_Group *mpi_group;
	MPI_Group *my_group;
#endif
	static int nthreads;
	int inversion_nthreads;
	int simplex_nmax, simplex_nmax_anneal;
	double simplex_temp_initial, simplex_temp_final, simplex_cooling_factor, simplex_minchisq, simplex_minchisq_anneal;
	int n_mcpoints; // for nested sampling
	int mcmc_threads;
	double mcmc_tolerance; // for Metropolis-Hastings
	bool mcmc_logfile;
	bool open_chisq_logfile;
	bool psf_convolution_mpi;
	bool use_mumps_subcomm;
	bool n_image_prior;
	double n_images_at_sbmax;
	double n_image_threshold;
	double max_pixel_sb;
	bool max_sb_prior_unselected_pixels;
	double max_sb_frac_unselected_pixels;
	bool subhalo_prior;
	bool lens_position_gaussian_transformation;
	ParamSettings *param_settings;

	int nlens;
	LensProfile** lens_list;

	int n_sb;
	SB_Profile** sb_list;

	lensvector source;
	image *images_found;
	ImageSystemType system_type;

	double lens_redshift;
	double source_redshift, reference_source_redshift, reference_zfactor; // reference zsrc is the redshift used to define the lensing quantities (kappa, etc.)
	bool user_changed_zsource;
	bool auto_zsource_scaling;
	double *source_redshifts; // used for modeling source points
	vector<int> source_redshift_groups;
	double *zfactors;
	bool vary_hubble_parameter;
	double hubble, omega_matter;
	double hubble_lower_limit, hubble_upper_limit;

	int Gauss_NN;	// for Gaussian quadrature
	double romberg_accuracy; // for Romberg integration

	Grid *grid;
	bool radial_grid;
	double grid_xlength, grid_ylength, grid_xcenter, grid_ycenter;  // for gridsize
	double sourcegrid_xmin, sourcegrid_xmax, sourcegrid_ymin, sourcegrid_ymax;
	double sourcegrid_limit_xmin, sourcegrid_limit_xmax, sourcegrid_limit_ymin, sourcegrid_limit_ymax;
	bool enforce_min_cell_area;
	bool cc_neighbor_splittings;
	double min_cell_area; // area of the smallest allowed cell area
	int rsplit_initial, thetasplit_initial;
	int splitlevels, cc_splitlevels;

	Lens *fitmodel;
	dvector fitparams, upper_limits, lower_limits, upper_limits_initial, lower_limits_initial, bestfitparams;
	dmatrix bestfit_fisher_inverse;
	dmatrix fisher_inverse;
	double bestfit_flux;
	double chisq_bestfit;
	SourceFitMode source_fit_mode;
	int lensmodel_fit_parameters, n_fit_parameters, n_sourcepts_fit;
	vector<string> fit_parameter_names, transformed_parameter_names;
	vector<string> latex_parameter_names, transformed_latex_parameter_names;
	lensvector *sourcepts_fit;
	bool *vary_sourcepts_x;
	bool *vary_sourcepts_y;
	lensvector *sourcepts_lower_limit;
	lensvector *sourcepts_upper_limit;
	double regularization_parameter, regularization_parameter_upper_limit, regularization_parameter_lower_limit;
	bool vary_regularization_parameter;
	static string fit_output_filename;
	bool auto_save_bestfit;
	bool borrowed_image_data; // tells whether image_data is pointing to that of another Lens object (e.g. fitmodel pointing to initial lens object)
	ImageData *image_data;
	double chisq_tolerance;
	int n_repeats;
	bool display_chisq_status;
	int n_visible_images;
	int chisq_display_frequency;
	double chisq_magnification_threshold, chisq_imgsep_threshold;
	bool use_magnification_in_chisq;
	bool use_magnification_in_chisq_during_repeats;
	bool include_parity_in_chisq;
	bool use_image_plane_chisq;
	bool calculate_parameter_errors;
	bool adaptive_grid;
	bool use_average_magnification_for_subgridding;
	bool activate_unmapped_source_pixels;
	bool exclude_source_pixels_beyond_fit_window;
	bool regrid_if_unmapped_source_subpixels;
	double pixel_magnification_threshold, pixel_magnification_threshold_lower_limit, pixel_magnification_threshold_upper_limit;
	double sim_err_pos, sim_err_flux, sim_err_td;

	bool fits_format;
	double data_pixel_size;
	void add_simulated_image_data(const lensvector &sourcept);
	void write_image_data(string filename);
	bool load_image_data(string filename);
	void remove_image_data(int image_set);

	bool read_data_line(ifstream& infile, vector<string>& datawords, int &n_datawords);
	bool datastring_convert(const string& instring, int& outvar);
	bool datastring_convert(const string& instring, double& outvar);
	void clear_image_data();
	void print_image_data(bool include_errors);

	bool autocenter;
	int autocenter_lens_number;
	bool auto_gridsize_from_einstein_radius;
	double auto_gridsize_multiple_of_Re;
	bool autogrid_before_grid_creation;
	double autogrid_frac, spline_frac;
	bool include_time_delays;
	static bool warnings, newton_warnings; // newton_warnings: when true, displays warnings when Newton's method fails or returns anomalous results
	static bool use_scientific_notation;
	string plot_title;
	bool plot_key_outside;
	double plot_ptsize, fontsize, linewidth;
	bool show_colorbar;
	int plot_pttype;
	string fit_output_dir;
	bool auto_fit_output_dir;
	enum TerminalType { TEXT, POSTSCRIPT, PDF } terminal; // keeps track of the file format for plotting
	enum FitMethod { POWELL, SIMPLEX, NESTED_SAMPLING, TWALK } fitmethod;
	enum RegularizationMethod { None, Norm, Gradient, Curvature, Image_Plane_Curvature } regularization_method;
	enum InversionMethod { CG_Method, MUMPS, UMFPACK } inversion_method;
	RayTracingMethod ray_tracing_method;
	bool parallel_mumps, show_mumps_info;

	int n_image_pixels_x, n_image_pixels_y; // note that this is the TOTAL number of pixels in the image, as opposed to image_npixels which gives the # of pixels being fit to
	int srcgrid_npixels_x, srcgrid_npixels_y;
	bool auto_srcgrid_npixels;
	bool auto_srcgrid_set_pixel_size;
	double pixel_fraction, pixel_fraction_lower_limit, pixel_fraction_upper_limit;
	bool vary_pixel_fraction, vary_magnification_threshold;
	double psf_width_x, psf_width_y, data_pixel_noise, sim_pixel_noise;
	double sb_threshold; // for creating centroid images from pixel maps
	double noise_threshold; // for automatic source grid sizing

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

	Defspline *defspline;

	// private functions are all contained in the file lens.cpp
	bool subgrid_around_satellites; // if on, will always subgrid around satellites (with pjaffe profile) when new grid is created
	static double galsubgrid_radius_fraction, galsubgrid_min_cellsize_fraction;
	static int galsubgrid_cc_splittings;
	void subgrid_around_satellite_galaxies(const double zfac);
	void calculate_critical_curve_deformation_radius(int lens_number, bool verbose, double &rmax, double& mass_enclosed);
	void calculate_critical_curve_deformation_radius_numerical(int lens_number, bool verbose, double& rmax_numerical, double& mass_enclosed);

	double subhalo_perturbation_radius_equation(const double r);
	// needed for calculating the subhalo perturbation radius
	int subhalo_lens_number;
	double theta_shear;
	lensvector subhalo_center;

	static const double satellite_einstein_radius_fraction;
	void plot_shear_field(double xmin, double xmax, int nx, double ymin, double ymax, int ny);
	void plot_lensinfo_maps(string file_root, const int x_n, const int y_N);
	void plot_logkappa_map(const int x_N, const int y_N);
	void plot_logmag_map(const int x_N, const int y_N);

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
	int find_zero_curves(Vector<lensvector>&, double, double, double, double, double, bool, double (Lens::*)(double, double));
	double lens_equation_x(double x, double y);
	double lens_equation_y(double x, double y);
	static int ncfound;

	bool use_cc_spline; // critical curves can be splined when (approximate) elliptical symmetry is present
	bool auto_ccspline;

	bool auto_sourcegrid;
	SourcePixelGrid *source_pixel_grid;
	void plot_source_pixel_grid(const char filename[]);

	ImagePixelGrid *image_pixel_grid;
	ImagePixelData *image_pixel_data;
	int image_npixels, source_npixels;
	int *active_image_pixel_i;
	int *active_image_pixel_j;
	double *image_surface_brightness;
	double *source_surface_brightness;
	double *source_pixel_n_images;

	int *image_pixel_location_Lmatrix;
	int *source_pixel_location_Lmatrix;
	int Lmatrix_n_elements;
	double *Lmatrix;
	int *Lmatrix_index;
	vector<double> *Lmatrix_rows;
	vector<int> *Lmatrix_index_rows;

	bool assign_pixel_mappings(bool verbal);
	double *Dvector;
	double *Fmatrix;
	int *Fmatrix_index;
	double *Rmatrix;
	int *Rmatrix_index;
	double *Rmatrix_diags;
	vector<double> *Rmatrix_rows;
	vector<int> *Rmatrix_index_rows;
	int *Rmatrix_row_nn;
	int Rmatrix_nn;
#ifdef USE_MUMPS
	static DMUMPS_STRUC_C *mumps_solver;
#endif

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
	double **psf_matrix;
	bool load_psf_fits(string fits_filename);
	int psf_npixels_x, psf_npixels_y;
	double psf_threshold;

	double Fmatrix_log_determinant, Rmatrix_log_determinant;
	void initialize_pixel_matrices(bool verbal);
	void clear_pixel_matrices();
	void clear_lensing_matrices();
	void assign_Lmatrix(bool verbal);
	void PSF_convolution_Lmatrix(bool verbal = false);
	void create_regularization_matrix(void);
	void generate_Rmatrix_from_gmatrices();
	void generate_Rmatrix_from_hmatrices();
	void generate_Rmatrix_norm();
	void generate_Rmatrix_from_image_plane_curvature();
	void create_lensing_matrices_from_Lmatrix(bool verbal);
	void invert_lens_mapping_MUMPS(bool verbal);
	void invert_lens_mapping_UMFPACK(bool verbal);
	void invert_lens_mapping_CG_method(bool verbal);
	void indexx(int* arr, int* indx, int nn);

	double set_required_data_pixel_window(bool verbal);

	double image_pixel_chi_square();
	void calculate_source_pixel_surface_brightness();
	void calculate_image_pixel_surface_brightness();
	void store_image_pixel_surface_brightness();
	void plot_image_pixel_surface_brightness(string outfile_root);
	double invert_image_surface_brightness_map(bool verbal);
	void load_pixel_grid_from_data();
	double invert_surface_brightness_map_from_data(bool verbal);

	void find_optimal_sourcegrid_for_analytic_source();
	bool create_source_surface_brightness_grid(bool verbal);
	void load_source_surface_brightness_grid(string source_inputfile);
	void load_image_surface_brightness_grid(string image_pixel_filename_root);
	bool plot_lensed_surface_brightness(string imagefile, bool output_fits = false, bool plot_residual = false, bool verbose = true);

	void plot_Lmatrix();
	void check_Lmatrix_columns();
	double temp_double;
	void Swap(double& a, double& b) { temp_double = a; a = b; b = temp_double; }

	double wtime0, wtime; // for calculating wall time in parallel calculations

public:

	friend class Grid;
	friend class SourcePixelGrid;
	friend class ImagePixelGrid;
	friend class ImagePixelData;
	Lens();
	Lens(Lens *lens_in);
	static void allocate_multithreaded_variables(const int& threads);
	static void deallocate_multithreaded_variables();
	~Lens();
#ifdef USE_MPI
	void set_mpi_params(const int& mpi_id_in, const int& mpi_np_in, const int& mpi_ngroups_in, const int& group_num_in, const int& group_id_in, const int& group_np_in, MPI_Group* group_in, MPI_Comm* comm, MPI_Group* mygroup, MPI_Comm* mycomm);
#endif
	void set_mpi_params(const int& mpi_id_in, const int& mpi_np_in);
	void set_nthreads(const int& nthreads_in) { nthreads=nthreads_in; }
#ifdef USE_MPI
	static void setup_mumps();
#endif
	static void delete_mumps();

	double kappa(const double& x, const double& y, const double zfactor);
	double potential(const double&, const double&, const double zfactor);
	void deflection(const double&, const double&, lensvector&, const int &thread, const double zfactor);
	void deflection(const double& x, const double& y, double& def_tot_x, double& def_tot_y, const int &thread, const double zfactor);
	void hessian(const double&, const double&, lensmatrix&, const int &thread, const double zfactor);
	void find_sourcept(const lensvector& x, lensvector& srcpt, const int &thread, const double zfactor);
	void find_sourcept(const lensvector& x, double& srcpt_x, double& srcpt_y, const int &thread, const double zfactor);

	// non-multithreaded versions
	//void deflection(const double& x, const double& y, lensvector &def_in, const double zfactor) { deflection(x,y,def_in,0,zfactor); }
	//void hessian(const double& x, const double& y, lensmatrix &hess_in, const double zfactor) { hessian(x,y,hess_in,0,zfactor); }
	//void find_sourcept(const lensvector& x, lensvector& srcpt, const double zfactor) { find_sourcept(x,srcpt,0,zfactor); }

	// versions of the above functions that use lensvector for (x,y) coordinates
	double kappa(const lensvector &x, const double zfactor) { return kappa(x[0], x[1], zfactor); }
	double potential(const lensvector& x, const double zfactor) { return potential(x[0],x[1], zfactor); }
	void deflection(const lensvector& x, lensvector& def, const double zfactor) { deflection(x[0], x[1], def, 0, zfactor); }
	void hessian(const lensvector& x, lensmatrix& hess, const double zfactor) { hessian(x[0], x[1], hess, 0, zfactor); }

	double inverse_magnification(const lensvector&, const int &thread, const double zfactor);
	double magnification(const lensvector &x, const int &thread, const double zfactor);
	double shear(const lensvector &x, const int &thread, const double zfactor);
	void shear(const lensvector &x, double& shear_tot, double& angle, const int &thread, const double zfactor);

	// non-multithreaded versions
	//double inverse_magnification(const lensvector& x, const double zfactor) { return inverse_magnification(x,0,zfactor); }
	//double magnification(const lensvector &x, const double zfactor) { return magnification(x,0,zfactor); }
	//double shear(const lensvector &x, const double zfactor) { return shear(x,0,zfactor); }
	//void shear(const lensvector &x, double& shear_tot, double& angle, const double zfactor) { return shear(x,shear_tot,angle,0,zfactor); }

	void hessian_exclude(const double& x, const double& y, const int& exclude_i, lensmatrix& hess_tot, const int& thread, const double zfactor);
	double magnification_exclude(const lensvector &x, const int& exclude_i, const int& thread, const double zfactor);
	double shear_exclude(const lensvector &x, const int& exclude_i, const int& thread, const double zfactor);
	void shear_exclude(const lensvector &x, double& shear, double& angle, const int& exclude_i, const int& thread, const double zfactor);
	double kappa_exclude(const lensvector &x, const int& exclude_i, const double zfactor);

	// non-multithreaded versions
	void hessian_exclude(const double& x, const double& y, const int& exclude_i, lensmatrix& hess_tot, const double zfactor) { hessian_exclude(x,y,exclude_i,hess_tot,0,zfactor); }
	double magnification_exclude(const lensvector &x, const int& exclude_i, const double zfactor) { return magnification_exclude(x,exclude_i,0,zfactor); }
	double shear_exclude(const lensvector &x, const int& exclude_i, const double zfactor) { return shear_exclude(x,exclude_i,0,zfactor); }
	void shear_exclude(const lensvector &x, double &shear, double &angle, const int& exclude_i, const double zfactor) { shear_exclude(x,shear,angle,exclude_i,0,zfactor); }

	bool test_for_elliptical_symmetry();
	bool test_for_singularity();
	void record_singular_points();

	// the following functions and objects are contained in commands.cpp
	char *buffer;
	int nullflag, buffer_length;
	string line;
	vector<string> lines;
	ifstream infile; // used to read commands from an input file
	bool verbal_mode;
	bool quit_after_error;
	int nwords;
	vector<string> words;
	stringstream* ws;
	stringstream datastream;
	bool read_from_file;
	bool quit_after_reading_file;
	void process_commands(bool read_file);
	bool read_command(bool show_prompt);
	void run_plotter(string plotcommand);
	void run_plotter_file(string plotcommand, string filename);
	void run_plotter_range(string plotcommand, string range);
	void run_plotter(string plotcommand, string filename, string range);
	void remove_equal_sign();
	void remove_word(int n_remove);
	void remove_comments(string& instring);
	void remove_equal_sign_datafile(vector<string>& datawords, int& n_datawords);

	void extract_word_starts_with(const char initial_character, int starting_word, int ending_word, string& extracted_word);
	bool extract_word_starts_with(const char initial_character, int starting_word, int ending_word, vector<string>& extracted_words);
	void set_quit_after_error(bool arg) { quit_after_error = arg; }

	// the following functions are contained in imgsrch.cpp
	private:
	void find_images();

	public:
	bool plot_recursive_grid(const char filename[]);
	void output_images_single_source(const double &x_source, const double &y_source, bool verbal, const double flux = -1.0, const bool show_labels = false);
	bool plot_images_single_source(const double &x_source, const double &y_source, bool verbal, const double flux = -1.0, const bool show_labels = false, string imgheader = "", string srcheader = "") {
		ofstream imgfile("imgs.dat");
		ofstream srcfile("srcs.dat");
		if (!imgheader.empty()) imgfile << "\"" << imgheader << "\"" << endl;
		if (!srcheader.empty()) srcfile << "\"" << srcheader << "\"" << endl;
		return plot_images_single_source(x_source,y_source,verbal,imgfile,srcfile,flux,show_labels);
	}
	bool plot_images_single_source(const double &x_source, const double &y_source, bool verbal, ofstream& imgfile, ofstream& srcfile, const double flux = -1.0, const bool show_labels = false);
	image* get_images(const lensvector &source_in, int &n_images) { return get_images(source_in, n_images, true); }
	image* get_images(const lensvector &source_in, int &n_images, bool verbal);
	bool plot_images(const char *sourcefile, const char *imagefile, bool verbal);
	void lens_equation(const lensvector&, lensvector&, const int& thread, const double zfactor); // Used by Newton's method to find images

	// the remaining functions in this class are all contained in lens.cpp
	void add_lens(LensProfileName, const double mass_parameter, const double scale, const double core, const double q, const double theta, const double xc, const double yc, const double extra_param1 = -1000, const double extra_param2 = -1000, const bool optional_setting = false);
	void add_shear_lens(const double shear, const double theta, const double xc, const double yc); // specific version for shear model
	void add_ptmass_lens(const double mass_parameter, const double xc, const double yc); // specific version for ptmass model
	void add_mass_sheet_lens(const double mass_parameter, const double xc, const double yc); // specific version for mass sheet

	void add_multipole_lens(int m, const double a_m, const double n, const double theta, const double xc, const double yc, bool kap, bool sine_term);
	void add_lens(const char *splinefile, const double q, const double theta, const double qx, const double f, const double xc, const double yc);
	void update_anchored_parameters();
	void print_lens_list(bool show_vary_params);
	void print_fit_model();

	void add_source_object(SB_ProfileName name, double sb_norm, double scale, double logslope_param, double q, double theta, double xc, double yc);
	void add_source_object(const char *splinefile, double q, double theta, double qx, double f, double xc, double yc);
	void remove_source_object(int sb_number);
	void print_source_list();
	void clear_source_objects();

	bool create_grid(bool verbal, const double zfac);
	void clear_lenses();
	void clear();
	void reset();
	void reset_grid();
	void remove_lens(int lensnumber);
	void toggle_major_axis_along_y(bool major_axis_along_y);
	void create_output_directory();

	private:
	bool temp_auto_ccspline, temp_auto_store_cc_points, temp_include_time_delays;
	void fit_set_optimizations();
	void fit_restore_defaults();
	double zfac_re; // used by einstein_radius_root(...)

	public:
	double chi_square_fit_simplex();
	double chi_square_fit_powell();
	void chi_square_nested_sampling();
	//void chi_square_metropolis_hastings();
	void chi_square_twalk();
	void test_fitmodel_invert();
	void plot_chisq_2d(const int param1, const int param2, const int n1, const double i1, const double f1, const int n2, const double i2, const double f2);
	void plot_chisq_1d(const int param, const int n, const double i, const double f, string filename);
	void chisq_single_evaluation();
	bool setup_fit_parameters(bool include_limits);
	void get_n_fit_parameters(int &nparams);
	void get_parameter_names();

	void get_automatic_initial_stepsizes(dvector& stepsizes);
	void set_default_plimits();
	void initialize_fitmodel();
	bool update_fitmodel(const double* params);
	double fitmodel_loglike_point_source(double* params);
	double fitmodel_loglike_pixellated_source(double* params);
	double fitmodel_loglike_pixellated_source_test(double* params);
	double loglike_point_source(double* params);
	bool calculate_fisher_matrix(const dvector &params, const dvector &stepsizes);
	double loglike_deriv(const dvector &params, const int index, const double step);
	void output_bestfit_model();
	bool use_bestfit_model();

	bool include_central_image;
	bool include_flux_chisq, include_time_delay_chisq;
	bool use_analytic_bestfit_src;
	bool n_images_penalty;
	bool fix_source_flux;
	double source_flux;

	bool spline_critical_curves(bool verbal);
	bool spline_critical_curves() { return spline_critical_curves(true); }
	void automatically_determine_ccspline_mode();
	bool plot_splined_critical_curves(const char *);
	bool plot_sorted_critical_curves(const char*);
	bool (Lens::*plot_critical_curves)(const char*);
	bool plotcrit(const char *filename) { return (this->*plot_critical_curves)(filename); }
	void plot_ray_tracing_grid(double xmin, double xmax, double ymin, double ymax, int x_N, int y_N, string filename);

	void make_source_rectangle(const double xmin, const double xmax, const int xsteps, const double ymin, const double ymax, const int ysteps, string source_filename);
	void make_source_ellipse(const double xcenter, const double ycenter, const double major_axis, const double q, const double angle, const int n_subellipses, const int points_per_ellipse, string source_filename);
	void plot_kappa_profile(int l, double rmin, double rmax, int steps, const char *kname, const char *kdname = NULL);
	void plot_total_kappa(double rmin, double rmax, int steps, const char *kname, const char *kdname = NULL);
	bool *centered;
	double einstein_radius_of_primary_lens(const double zfac);
	double einstein_radius_root(const double r);
	void plot_mass_profile(double rmin, double rmax, int steps, const char *massname);
	bool plot_lens_equation(double x_source, double y_source, const char *lxfile, const char *lyfile);
	bool make_random_sources(int nsources, const char *outfile);
	bool total_cross_section(double&);
	double total_cross_section_integrand(const double);

	double chisq_pos_source_plane();
	double chisq_pos_image_plane();
	double chisq_flux();
	double chisq_time_delays();
	void output_model_source_flux(double *bestfit_flux);
	void output_analytic_srcpos(lensvector *beta_i);

	static bool respline_at_end;
	static int resplinesteps;
	void create_deflection_spline(int steps);
	void spline_deflection(double xl, double yl, int steps);
	bool autospline_deflection(int steps);
	bool unspline_deflection();
	bool isspherical();
	bool islens() { return (nlens > 0); }
	void set_grid_corners(double xmin, double xmax, double ymin, double ymax);
	void set_gridsize(double xl, double yl);
	void set_gridcenter(double xc, double yc);
	void autogrid(double rmin, double rmax, double frac);
	void autogrid(double rmin, double rmax);
	void autogrid();
	bool get_deflection_spline_info(double &xmax, double &ymax, int &nsteps);
	void delete_ccspline();
	void set_Gauss_NN(const int& nn);
	void set_romberg_accuracy(const double& acc);

	void set_integration_method(IntegrationMethod method) { LensProfile::integral_method = method; }

	void set_warnings(bool setting) { warnings = setting; }
	void get_warnings(bool &setting) { setting = warnings; }
	void set_newton_warnings(bool setting) { newton_warnings = setting; }
	void get_newton_warnings(bool &setting) { setting = newton_warnings; }
	void set_ccspline_mode(bool setting) { use_cc_spline = setting; plot_critical_curves = (setting==true) ? &Lens::plot_splined_critical_curves : &Lens::plot_sorted_critical_curves; }
	void get_ccspline_mode(bool &setting) { setting = use_cc_spline; }
	void set_auto_ccspline_mode(bool setting) { auto_ccspline = setting; }
	void get_auto_ccspline_mode(bool &setting) { setting = auto_ccspline; }
	void set_galsubgrid_mode(bool setting) { subgrid_around_satellites = setting; }
	void get_galsubgrid_mode(bool &setting) { setting = subgrid_around_satellites; }
	void set_auto_store_cc_points(bool setting) { auto_store_cc_points = setting; }

	void set_rsplit_initial(int setting) { rsplit_initial = setting; }
	void get_rsplit_initial(int &setting) { setting = rsplit_initial; }
	void set_thetasplit_initial(int setting) { thetasplit_initial = setting; }
	void get_thetasplit_initial(int &setting) { setting = thetasplit_initial; }
	void set_splitlevels(int setting) { splitlevels = setting; }
	void get_splitlevels(int &setting) { setting = splitlevels; }
	void set_cc_splitlevels(int setting) { cc_splitlevels = setting; }
	void get_cc_splitlevels(int &setting) { setting = cc_splitlevels; }
	void set_rminfrac(double setting) { rmin_frac = setting; }
	void get_rminfrac(double &setting) { setting = rmin_frac; }
	void set_imagepos_accuracy(double setting) { Grid::set_imagepos_accuracy(setting); }
	void get_imagepos_accuracy(double &setting) { setting = Grid::image_pos_accuracy; }
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
		}
		if ((n_sourcepts_fit > 0) and ((fitmethod == NESTED_SAMPLING) or (fitmethod == TWALK))) {
			if (sourcepts_lower_limit==NULL) sourcepts_lower_limit = new lensvector[n_sourcepts_fit];
			if (sourcepts_upper_limit==NULL) sourcepts_upper_limit = new lensvector[n_sourcepts_fit];
			for (int i=0; i < nlens; i++) lens_list[i]->set_include_limits(true);
		}

	}
	void set_show_wtime(bool show_wt) { show_wtime = show_wt; }
	bool open_command_file(char *filename);
	void set_verbal_mode(bool echo) { verbal_mode = echo; }
	void set_quit_after_reading_file(bool setting) { quit_after_reading_file = setting; }

	bool get_einstein_radius(int lens_number, double& re_major_axis, double& re_average);

	double crit0_interpolate(double theta) { return ccspline[0].splint(theta); }
	double crit1_interpolate(double theta) { return ccspline[1].splint(theta); }
	double caust0_interpolate(double theta);
	double caust1_interpolate(double theta);

	//double make_satellite_population(const double number_density, const double rmax, const double a, const double b);
	//void plot_satellite_deflection_vs_area();

	//void generate_solution_chain_sdp81(); // specialty function...probably should put in separate file & header file; do this later
};

struct ImageData
{
	int n_images;
	lensvector *pos;
	double *flux;
	double *time_delays;
	double *sigma_pos, *sigma_f, *sigma_t;
	double max_distsqr; // maximum squared distance between any pair of images
	ImageData() { n_images = 0; }
	void input(const int &nn);
	void input(const ImageData& imgs_in);
	void input(const int &nn, image* images, const double sigma_pos_in, const double sigma_flux_in, const double sigma_td_in, bool* include, bool include_time_delays);
	void add_image(lensvector& pos_in, const double sigma_pos_in, const double flux_in, const double sigma_f_in, const double time_delay_in, const double sigma_t_in);
	void print_list(bool print_errors, bool use_sci);
	void write_to_file(ofstream &outfile);
	~ImageData();
};

struct ParamPrior
{
	double gaussian_pos, gaussian_sig;
	Prior prior;
	ParamPrior() { prior = UNIFORM_PRIOR; }
	ParamPrior(ParamPrior *prior_in)
	{
		prior = prior_in->prior;
		if (prior==GAUSS_PRIOR) {
			gaussian_pos = prior_in->gaussian_pos;
			gaussian_sig = prior_in->gaussian_sig;
		}
	}
	void set_uniform() { prior = UNIFORM_PRIOR; }
	void set_log() { prior = LOG_PRIOR; }
	void set_gaussian(double &pos_in, double &sig_in) { prior = GAUSS_PRIOR; gaussian_pos = pos_in; gaussian_sig = sig_in; }
};

struct ParamTransform
{
	double gaussian_pos, gaussian_sig;
	double a, b; // for linear transformations
	bool include_jacobian;
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
		}
	}
	void set_none() { transform = NONE; }
	void set_log() { transform = LOG_TRANSFORM; }
	void set_linear(double &a_in, double &b_in) { transform = LINEAR_TRANSFORM; a = a_in; b = b_in; }
	void set_gaussian(double &pos_in, double &sig_in) { transform = GAUSS_TRANSFORM; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void set_include_jacobian(bool &include) { include_jacobian = include; }
};

struct ParamSettings
{
	int nparams;
	ParamPrior **priors;
	ParamTransform ** transforms;
	string *param_names;
	double *penalty_limits_lo, *penalty_limits_hi;
	bool *use_penalty_limits;
	double *stepsizes;
	bool *auto_stepsize;
	ParamSettings() { priors = NULL; param_names = NULL; transforms = NULL; nparams = 0; stepsizes = NULL; auto_stepsize = NULL; }
	ParamSettings(ParamSettings& param_settings_in) {
		nparams = param_settings_in.nparams;
		priors = new ParamPrior*[nparams];
		transforms = new ParamTransform*[nparams];
		stepsizes = new double[nparams];
		auto_stepsize = new bool[nparams];
		penalty_limits_lo = new double[nparams];
		penalty_limits_hi = new double[nparams];
		use_penalty_limits = new bool[nparams];
		for (int i=0; i < nparams; i++) {
			priors[i] = new ParamPrior(param_settings_in.priors[i]);
			transforms[i] = new ParamTransform(param_settings_in.transforms[i]);
			stepsizes[i] = param_settings_in.stepsizes[i];
			auto_stepsize[i] = param_settings_in.auto_stepsize[i];
			penalty_limits_lo[i] = param_settings_in.penalty_limits_lo[i];
			penalty_limits_hi[i] = param_settings_in.penalty_limits_hi[i];
			use_penalty_limits[i] = param_settings_in.use_penalty_limits[i];
		}
	}
	void update_params(int nparams_in, vector<string>& names, double* stepsizes_in)
	{
		int i;
		if (nparams==nparams_in) {
			// update parameter names just in case
			for (i=0; i < nparams_in; i++) {
				param_names[i] = names[i];
			}
			return;
		}
		int newparams = nparams_in - nparams;
		ParamPrior** newpriors = new ParamPrior*[nparams_in];
		ParamTransform** newtransforms = new ParamTransform*[nparams_in];
		double* new_stepsizes = new double[nparams_in];
		bool* new_auto_stepsize = new bool[nparams_in];
		double* new_penalty_limits_lo = new double[nparams_in];
		double* new_penalty_limits_hi = new double[nparams_in];
		bool* new_use_penalty_limits = new bool[nparams_in];
		if (param_names != NULL) delete[] param_names;
		param_names = new string[nparams_in];
		if (nparams_in > nparams) {
			for (i=0; i < nparams; i++) {
				newpriors[i] = new ParamPrior(priors[i]);
				newtransforms[i] = new ParamTransform(transforms[i]);
				new_stepsizes[i] = stepsizes[i];
				new_auto_stepsize[i] = auto_stepsize[i];
				new_penalty_limits_lo[i] = penalty_limits_lo[i];
				new_penalty_limits_hi[i] = penalty_limits_hi[i];
				new_use_penalty_limits[i] = use_penalty_limits[i];
				param_names[i] = names[i];
			}
			for (i=nparams; i < nparams_in; i++) {
				newpriors[i] = new ParamPrior();
				newtransforms[i] = new ParamTransform();
				param_names[i] = names[i];
				new_stepsizes[i] = stepsizes_in[i];
				new_auto_stepsize[i] = true; // stepsizes for newly added parameters are set to 'auto'
				new_penalty_limits_lo[i] = -1e30;
				new_penalty_limits_hi[i] = 1e30;
				new_use_penalty_limits[i] = false;
			}
		} else {
			for (i=0; i < nparams_in; i++) {
				newpriors[i] = new ParamPrior(priors[i]);
				newtransforms[i] = new ParamTransform(transforms[i]);
				new_stepsizes[i] = stepsizes[i];
				new_auto_stepsize[i] = auto_stepsize[i];
				new_penalty_limits_lo[i] = penalty_limits_lo[i];
				new_penalty_limits_hi[i] = penalty_limits_hi[i];
				new_use_penalty_limits[i] = use_penalty_limits[i];
				param_names[i] = names[i];
			}
		}
		if (nparams > 0) {
			for (i=0; i < nparams; i++) {
				delete priors[i];
				delete transforms[i];
			}
			delete[] priors;
			delete[] transforms;
			delete[] stepsizes;
			delete[] auto_stepsize;
			delete[] penalty_limits_lo;
			delete[] penalty_limits_hi;
			delete[] use_penalty_limits;
		}
		priors = newpriors;
		transforms = newtransforms;
		stepsizes = new_stepsizes;
		auto_stepsize = new_auto_stepsize;
		penalty_limits_lo = new_penalty_limits_lo;
		penalty_limits_hi = new_penalty_limits_hi;
		use_penalty_limits = new_use_penalty_limits;
		nparams = nparams_in;
	}
	void clear_penalty_limits()
	{
		for (int i=0; i < nparams; i++) {
			use_penalty_limits[i] = false;
		}
	}
	void print_priors()
	{
		if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
		cout << "Parameter settings:\n";
		int max_length=0;
		for (int i=0; i < nparams; i++) {
			if (param_names[i].length() > max_length) max_length = param_names[i].length();
		}
		int extra_length;
		for (int i=0; i < nparams; i++) {
			cout << i << ". " << param_names[i] << ": ";
			extra_length = max_length - param_names[i].length();
			for (int j=0; j < extra_length; j++) cout << " ";
			if ((nparams > 10) and (i < 10)) cout << " ";
			if (priors[i]->prior==UNIFORM_PRIOR) cout << "uniform prior";
			else if (priors[i]->prior==LOG_PRIOR) cout << "log prior";
			else if (priors[i]->prior==GAUSS_PRIOR) {
				cout << "gaussian prior (mean=" << priors[i]->gaussian_pos << ", sigma=" << priors[i]->gaussian_sig << ")";
			}
			else die("Prior type unknown");
			if (transforms[i]->transform==NONE) ;
			else if (transforms[i]->transform==LOG_TRANSFORM) cout << ", log transformation";
			else if (transforms[i]->transform==GAUSS_TRANSFORM) cout << ", gaussian transformation (mean=" << transforms[i]->gaussian_pos << ", sigma=" << transforms[i]->gaussian_sig << ")";
			else if (transforms[i]->transform==LINEAR_TRANSFORM) cout << ", linear transformation A*" << param_names[i] << " + b (A=" << transforms[i]->a << ", b=" << transforms[i]->b << ")";
			if (transforms[i]->include_jacobian==true) cout << " (include Jacobian in likelihood)";
			cout << endl;
		}
	}
	void print_stepsizes()
	{
		if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
		cout << "Parameter initial stepsizes:\n";
		int max_length=0;
		for (int i=0; i < nparams; i++) {
			if (param_names[i].length() > max_length) max_length = param_names[i].length();
		}
		int extra_length;
		for (int i=0; i < nparams; i++) {
			cout << i << ". " << param_names[i] << ": ";
			extra_length = max_length - param_names[i].length();
			for (int j=0; j < extra_length; j++) cout << " ";
			if ((nparams > 10) and (i < 10)) cout << " ";
			cout << stepsizes[i];
			if (auto_stepsize[i]) cout << " (auto)";
			cout << endl;
		}
	}
	void print_penalty_limits()
	{
		if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
		cout << "Parameter limits imposed on chi-square:\n";
		int max_length=0;
		for (int i=0; i < nparams; i++) {
			if (param_names[i].length() > max_length) max_length = param_names[i].length();
		}
		int extra_length;
		for (int i=0; i < nparams; i++) {
			cout << i << ". " << param_names[i] << ": ";
			extra_length = max_length - param_names[i].length();
			for (int j=0; j < extra_length; j++) cout << " ";
			if ((nparams > 10) and (i < 10)) cout << " ";
			if ((use_penalty_limits[i]==false) or ((penalty_limits_lo[i]==-1e30) and (penalty_limits_hi[i]==1e30))) cout << "none" << endl;
			else {
				cout << "[";
				if (penalty_limits_lo[i]==-1e30) cout << "-inf";
				else cout << penalty_limits_lo[i];
				cout << ":";
				if (penalty_limits_hi[i]==1e30) cout << "inf";
				else cout << penalty_limits_hi[i];
				cout << "]" << endl;
			}
		}
	}
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
			if ((use_penalty_limits[i]==false) and (use_plimits[i]==true))
			{
				use_penalty_limits[i] = true;
				penalty_limits_lo[i] = lower[i];
				penalty_limits_hi[i] = upper[i];
			}
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
			if (transforms[i]->transform==NONE) { transformed_names[i] = names[i]; transformed_latex_names[i] = latex_names[i]; }
			else if (transforms[i]->transform==LOG_TRANSFORM) { transformed_names[i] = "log(" + names[i] + ")"; transformed_latex_names[i] = "\\ln " + latex_names[i]; }
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				transformed_names[i] = "u{" + names[i] + "}";
				transformed_latex_names[i] = "u\\{" + latex_names[i] + "\\}";
			}
			else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				transformed_names[i] = "L{" + names[i] + "}";
				transformed_latex_names[i] = "L\\{" + latex_names[i] + "\\}";
			}
		}
	}
	void add_prior_terms_to_loglike(double *params, double& loglike)
	{
		for (int i=0; i < nparams; i++) {
			if (priors[i]->prior==LOG_PRIOR) loglike += log(params[i]);
			else if (priors[i]->prior==GAUSS_PRIOR) loglike += SQR((params[i] - priors[i]->gaussian_pos)/priors[i]->gaussian_sig)/2.0;
		}
	}
	void add_jacobian_terms_to_loglike(double *params, double& loglike)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->include_jacobian==true) {
				if (transforms[i]->transform==LOG_TRANSFORM) loglike -= log(params[i]);
				else if (transforms[i]->transform==GAUSS_TRANSFORM) loglike -= SQR((params[i] - transforms[i]->gaussian_pos)/transforms[i]->gaussian_sig)/2.0;
			}
		}
	}
	~ParamSettings()
	{
		if (nparams > 0) {
			for (int i=0; i < nparams; i++) {
				delete priors[i];
				delete transforms[i];
			}
			delete[] priors;
			delete[] transforms;
			delete[] stepsizes;
			delete[] auto_stepsize;
			delete[] penalty_limits_lo;
			delete[] penalty_limits_hi;
			delete[] use_penalty_limits;
		}
	}
};

class Defspline
{
	Spline2D ax, ay;
	Spline2D axx, ayy, axy;

public:
	friend void Lens::spline_deflection(double,double,int);
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

inline double Lens::kappa(const double& x, const double& y, const double zfactor)
{
	double kappa=0;
	for (int i=0; i < nlens; i++)
		kappa += lens_list[i]->kappa(x,y);
	return zfactor*kappa;
}

inline double Lens::potential(const double& x, const double& y, const double zfactor)
{
	double pot = 0;
	for (int i=0; i < nlens; i++)
		pot += lens_list[i]->potential(x,y);
	return zfactor*pot;
}

inline void Lens::deflection(const double& x, const double& y, lensvector& def_tot, const int &thread, const double zfactor)
{
	if (!defspline)
	{
		lensvector *def_i = &defs_i[thread];
		lens_list[0]->deflection(x,y,def_tot);
		int indx;
		for (indx=1; indx < nlens; indx++) {
			lens_list[indx]->deflection(x,y,(*def_i));
			def_tot[0] += (*def_i)[0];
			def_tot[1] += (*def_i)[1];
		}
	}
	else {
		def_tot = defspline->deflection(x,y);
	}
	def_tot[0] *= zfactor;
	def_tot[1] *= zfactor;
}

inline void Lens::deflection(const double& x, const double& y, double& def_tot_x, double& def_tot_y, const int &thread, const double zfactor)
{
	if (!defspline)
	{
		lensvector *def_i = &defs_i[thread];
		lens_list[0]->deflection(x,y,(*def_i));
		def_tot_x = (*def_i)[0];
		def_tot_y = (*def_i)[1];
		int indx;
		for (indx=1; indx < nlens; indx++) {
			lens_list[indx]->deflection(x,y,(*def_i));
			def_tot_x += (*def_i)[0];
			def_tot_y += (*def_i)[1];
		}
	}
	else {
		lensvector *def_i = &defs_i[thread];
		(*def_i) = defspline->deflection(x,y);
		def_tot_x = (*def_i)[0];
		def_tot_y = (*def_i)[1];
	}
	def_tot_x *= zfactor;
	def_tot_y *= zfactor;
}

inline void Lens::hessian(const double& x, const double& y, lensmatrix& hess_tot, const int &thread, const double zfactor) // calculates the Hessian of the lensing potential
{
	if (!defspline)
	{
		lensmatrix *hess_i = &hesses_i[thread];
		lens_list[0]->hessian(x,y,hess_tot);
		int indx;
		for (indx=1; indx < nlens; indx++) {
			lens_list[indx]->hessian(x,y,(*hess_i));
			hess_tot[0][0] += (*hess_i)[0][0];
			hess_tot[1][1] += (*hess_i)[1][1];
			hess_tot[0][1] += (*hess_i)[0][1];
			hess_tot[1][0] += (*hess_i)[1][0];
		}
	}
	else {
		hess_tot = defspline->hessian(x,y);
	}
	hess_tot[0][0] *= zfactor;
	hess_tot[1][1] *= zfactor;
	hess_tot[0][1] *= zfactor;
	hess_tot[1][0] *= zfactor;
}

inline void Lens::find_sourcept(const lensvector& x, lensvector& srcpt, const int& thread, const double zfactor)
{
	deflection(x[0],x[1],srcpt,thread,zfactor);
	srcpt[0] = x[0] - srcpt[0]; // this uses the lens equation, beta = theta - alpha (except without defining an intermediate lensvector alpha, which would be an extra memory operation)
	srcpt[1] = x[1] - srcpt[1];
}

inline void Lens::find_sourcept(const lensvector& x, double& srcpt_x, double& srcpt_y, const int& thread, const double zfactor)
{
	deflection(x[0],x[1],srcpt_x,srcpt_y,thread,zfactor);
	srcpt_x = x[0] - srcpt_x; // this uses the lens equation, beta = theta - alpha (except without defining an intermediate lensvector alpha, which would be an extra memory operation)
	srcpt_y = x[1] - srcpt_y;
}

inline double Lens::inverse_magnification(const lensvector& x, const int &thread, const double zfactor)
{
	lensmatrix *jac = &jacs[thread];
	hessian(x[0],x[1],(*jac),thread,zfactor);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	return determinant((*jac));
}

inline double Lens::magnification(const lensvector &x, const int &thread, const double zfactor)
{
	lensmatrix *jac = &jacs[thread];
	hessian(x[0],x[1],(*jac),thread,zfactor);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	return 1.0/determinant((*jac));
}

inline double Lens::shear(const lensvector &x, const int &thread, const double zfactor)
{
	lensmatrix *hess = &hesses[thread];
	hessian(x[0],x[1],(*hess),thread,zfactor);
	double shear1, shear2;
	shear1 = 0.5*((*hess)[0][0]-(*hess)[1][1]);
	shear2 = (*hess)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}

inline void Lens::shear(const lensvector &x, double& shear_tot, double& angle, const int &thread, const double zfactor)
{
	lensmatrix *hess = &hesses[thread];
	hessian(x[0],x[1],(*hess),thread,zfactor);
	double shear1, shear2;
	shear1 = 0.5*((*hess)[0][0]-(*hess)[1][1]);
	shear2 = (*hess)[0][1];
	shear_tot = sqrt(shear1*shear1+shear2*shear2);
	angle = atan(abs(shear2/shear1));
	if (shear1 < 0) {
		if (shear2 < 0)
			angle = angle - M_PI;
		else
			angle = M_PI - angle;
	} else if (shear2 < 0) {
		angle = -angle;
	}
	angle = 0.5*radians_to_degrees(angle);
}

// the following functions find the shear, kappa and magnification at the position where a satellite is placed;
// this information is used to determine the optimal subgrid size and resolution

inline void Lens::hessian_exclude(const double& x, const double& y, const int& exclude_i, lensmatrix& hess_tot, const int& thread, const double zfactor)
{
	lensmatrix *hess = &hesses[thread];
	hess_tot[0][0] = 0;
	hess_tot[1][1] = 0;
	hess_tot[0][1] = 0;
	hess_tot[1][0] = 0;
	int indx;
	for (indx=0; indx < nlens; indx++) {
		if (indx != exclude_i) {
			lens_list[indx]->hessian(x,y,(*hess));
			hess_tot[0][0] += (*hess)[0][0];
			hess_tot[1][1] += (*hess)[1][1];
			hess_tot[0][1] += (*hess)[0][1];
			hess_tot[1][0] += (*hess)[1][0];
		}
	}
	hess_tot[0][0] *= zfactor;
	hess_tot[1][1] *= zfactor;
	hess_tot[0][1] *= zfactor;
	hess_tot[1][0] *= zfactor;
}

inline double Lens::magnification_exclude(const lensvector &x, const int& exclude_i, const int& thread, const double zfactor)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude_i,(*jac),thread,zfactor);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];

	return 1.0/determinant((*jac));
}

inline double Lens::shear_exclude(const lensvector &x, const int& exclude_i, const int& thread, const double zfactor)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude_i,(*jac),thread,zfactor);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	double shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	return sqrt(shear1*shear1+shear2*shear2);
}

inline void Lens::shear_exclude(const lensvector &x, double &shear, double &angle, const int& exclude_i, const int& thread, const double zfactor)
{
	lensmatrix *jac = &jacs[thread];
	hessian_exclude(x[0],x[1],exclude_i,(*jac),thread,zfactor);
	(*jac)[0][0] = 1 - (*jac)[0][0];
	(*jac)[1][1] = 1 - (*jac)[1][1];
	(*jac)[0][1] = -(*jac)[0][1];
	(*jac)[1][0] = -(*jac)[1][0];
	double shear1, shear2;
	shear1 = 0.5*((*jac)[1][1]-(*jac)[0][0]);
	shear2 = -(*jac)[0][1];
	shear = sqrt(shear1*shear1+shear2*shear2);
	angle = atan(abs(shear2/shear1));
	if (shear1 < 0) {
		if (shear2 < 0)
			angle = angle - M_PI;
		else
			angle = M_PI - angle;
	} else if (shear2 < 0) {
		angle = -angle;
	}
	angle = 0.5*radians_to_degrees(angle);
}

inline double Lens::kappa_exclude(const lensvector &x, const int& exclude_i, const double zfactor)
{
	double kappa=0;
	for (int i=0; i < nlens; i++)
		if (i != exclude_i) kappa += lens_list[i]->kappa(x[0],x[1]);

	return zfactor*kappa;
}

#endif // QLENS_H
