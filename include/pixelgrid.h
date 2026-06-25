#ifndef PIXELGRID_H
#define PIXELGRID_H

#include "qlens.h"
#include "modelparams.h"
#include "rand.h"
#include "lensvec.h"
#include "egrad.h" // contains IsophoteData structure used for recording isophote fits
#include "trirectangle.h"
#include "delaunay.h"
#include <vector>
#include <iostream>

class ImagePixelGrid;
class CartesianSourcePixel;
class CartesianSourceGrid;
struct ImageData;
class PSF;

enum PixelGridType { CartesianPixelGrid, DelaunayPixelGrid };

struct InterpolationCells {
	bool found_containing_cell;
	CartesianSourcePixel *pixel[3];
};

template <typename QScalar>
struct PtsWgts {
	int indx;
	QScalar wgt;
	PtsWgts() {}
	PtsWgts(const int indx_in, const QScalar wgt_in) {
		indx = indx_in;
		wgt = wgt_in;
	}
	PtsWgts& assign(const int indx_in, const QScalar wgt_in) {
		indx = indx_in;
		wgt = wgt_in;
		return *this;
	}
};

class CartesianSourcePixel
{
	friend class QLens;
	friend class ImagePixelGrid;
	friend class CartesianSourceGrid;

	protected: 
	QLens *lens;
	ImagePixelGrid *image_pixel_grid;
	CartesianSourceGrid *parent_grid; // this points to the top-level grid
	CartesianSourcePixel ***cell;
	CartesianSourcePixel *neighbor[4]; // 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	int ii, jj; // this is the index assigned to this cell in the grid of the parent cell

	int u_N, w_N;
	int level;
	double cell_area;
	lensvector<double> center_pt;
	lensvector<double> corner_pt[4];
	double surface_brightness;
	int index, active_index;
	bool maps_to_image_pixel;
	bool maps_to_image_window;
	bool active_pixel;
	vector<double> overlaps;
	vector<double> weighted_overlaps;
	vector<int> overlap_pixel_n;
	double total_magnification, n_images, avg_image_pixels_mapped;

	static int max_levels;
	static TriRectangleOverlap *trirec;
	static int nthreads;
	static int *imin, *imax, *jmin, *jmax; // defines "window" within which we will check all the cells for overlap
	static InterpolationCells *nearest_interpolation_cells;
	static lensvector<double> **interpolation_pts[3];
	static int *maxlevs;
	static lensvector<double> ***xvals_threads;
	static lensvector<double> ***corners_threads;
	static lensvector<double> **twistpts_threads;
	static int **twist_status_threads;

	void split_cells(const int usplit, const int wsplit, const int& thread);
	void unsplit();
	void split_subcells(const int splitlevel, const int thread);

	void assign_level_neighbors(int neighbor_level);
	void test_neighbors();
	void assign_indices(int& source_pixel_i);
	void assign_active_indices(int& source_pixel_i);

	public:
	CartesianSourcePixel(QLens* lens_in) { lens = lens_in; image_pixel_grid = NULL; }
	CartesianSourcePixel(QLens* lens_in, lensvector<double>** xij, const int& i, const int& j, const int& level_in, CartesianSourceGrid* parent_ptr);
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	inline bool check_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread);
	inline bool check_if_in_neighborhood(lensvector<double> **input_corner_pts, bool &inside, const int& thread);
	inline double find_rectangle_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread, const int&, const int&);
	inline bool check_triangle1_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread);
	inline bool check_triangle2_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread);
	inline double find_triangle1_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread);
	inline double find_triangle2_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread);

	void generate_gmatrices();
	void generate_hmatrices();

	void subcell_assign_source_mapping_flags_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, const int& thread, bool& image_pixel_maps_to_source_grid);
	void find_lensed_surface_brightness_subcell_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread, double& overlap, double& total_overlap, double& total_weighted_surface_brightness);

	bool subcell_assign_source_mapping_flags_interpolate(lensvector<double> &input_center_pt, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, const int& thread);
	void calculate_Lmatrix_interpolate(const int img_index, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, int& Lmatrix_index, lensvector<double> &input_center_pts, const int& ii, const double weight, const int& thread);
	void find_triangle_weighted_invmag_subcell(lensvector<double>& pt1, lensvector<double>& pt2, lensvector<double>& pt3, double& total_overlap, double& total_weighted_invmag, const int& thread);

	void find_interpolation_cells(lensvector<double> &input_center_pt, const int& thread);
	CartesianSourcePixel* find_nearest_neighbor_cell(lensvector<double> &input_center_pt, const int& side);
	CartesianSourcePixel* find_nearest_neighbor_cell(lensvector<double> &input_center_pt, const int& side, const int tiebreaker_side);
	void find_nearest_two_cells(CartesianSourcePixel* &cellptr1, CartesianSourcePixel* &cellptr2, const int& side);
	CartesianSourcePixel* find_corner_cell(const int i, const int j);

	void assign_surface_brightness_from_analytic_source(const int imggrid_i=-1);
	void assign_surface_brightness_from_delaunay_grid(DelaunaySourceGrid* delaunay_grid, const bool add_sb = false);
	void update_surface_brightness(int& index);
	void fill_surface_brightness_vector();
	void fill_surface_brightness_vector_recursive(int& column_j);
	void fill_n_image_vector();

	void fill_n_image_vector_recursive(int& column_j);
	//void plot_surface_brightness(string root);
	//void output_fits_file(string fits_filename);
	void get_grid_dimensions(double &xmin, double &xmax, double &ymin, double &ymax);
	void output_cell_surface_brightness(int line_number, int pixels_per_cell_x, int pixels_per_cell_y, Vector<double>& sbvals, Vector<double>& maglogvals, Vector<double>& nimgvals, int& indx);
	void plot_cell_surface_brightness(int line_number, int pixels_per_cell_x, int pixels_per_cell_y, std::ofstream& sb_outfile, std::ofstream& mag_outfile, std::ofstream& nimg_outfile);
	void store_surface_brightness_grid_data(string root);
	void write_surface_brightness_to_file(std::ofstream &sb_outfile);
	void read_surface_brightness_data(std::ifstream &sb_infile);

	void clear_subgrids();
	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }
	void plot_corner_coordinates(std::ofstream &gridout);
	void clear(void);
	~CartesianSourcePixel();
};

class CartesianSourceGrid : public CartesianSourcePixel, public Model
{
	friend class QLens;
	friend class ImagePixelGrid;
	friend class CartesianSourcePixel;

	int n_active_pixels;
	double xcenter, ycenter;
	double srcgrid_xmin, srcgrid_xmax, srcgrid_ymin, srcgrid_ymax;
	bool regrid_if_unmapped_source_subcells;
	bool activate_unmapped_source_pixels;
	bool exclude_source_pixels_outside_fit_window;

	int number_of_pixels; // this is the total number of pixels, including all subpixels
	int npixels_x, npixels_y;
	double min_cell_area;
	int levels; // keeps track of the total number of grid cell levels
	bool regrid;

	void assign_firstlevel_neighbors(void);
	void assign_all_neighbors(void);
	int assign_indices_and_count_levels();
	int assign_active_indices_and_count_source_pixels(const int source_pixel_i_initial, const bool activate_unmapped_pixels, const bool exclude_pixels_outside_window);
	void split_subcells_firstlevel(const int splitlevel);

	void print_indices();

	public:
	double srcgrid_redshift;
	CartesianSourceGrid(QLens* lens_in, const int band = 0, const double zsrc_in = -1);
	void copy_pixsrc_data(CartesianSourceGrid* grid_in);
	void update_meta_parameters(const bool varied_only_fitparams);
	void get_parameter_numbers_from_qlens(int& pi, int& pf);
	bool register_vary_parameters_in_qlens();
	void register_limits_in_qlens();
	void update_fitparams_in_qlens();
	void create_pixel_grid(QLens* lens_in, const double x_min, const double x_max, const double y_min, const double y_max, const int usplit0, const int wsplit0);
	void create_pixel_grid(QLens* lens_in, string pixel_data_fileroot, const double minarea_in);
	//void copy_source_pixel_grid(CartesianSourceGrid* input_pixel_grid);
	void setup_parameters(const bool initial_setup);

	double regparam;
	double pixel_fraction, srcgrid_size_scale, pixel_magnification_threshold;

	void calculate_pixel_magnifications(const bool use_emask = false);
	void adaptive_subgrid();
	double get_lowest_mag_sourcept(double &xsrc, double &ysrc);
	void get_highest_mag_sourcept(double &xsrc, double &ysrc);

	bool assign_source_mapping_flags_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, const int& thread);
	void calculate_Lmatrix_overlap(const int &img_index, const int image_pixel_i, const int image_pixel_j, int& Lmatrix_index, lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread);
	double find_lensed_surface_brightness_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread);

	bool bisection_search_overlap(lensvector<double> **input_corner_pts, const int& thread);
	bool bisection_search_overlap(lensvector<double> &a, lensvector<double> &b, lensvector<double> &c, const int& thread);
	bool bisection_search_interpolate(lensvector<double> &input_center_pt, const int& thread);
	bool assign_source_mapping_flags_interpolate(lensvector<double> &input_center_pt, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, const int& thread, const int& image_pixel_i, const int& image_pixel_j);
	void calculate_Lmatrix_interpolate(const int img_index, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, int& Lmatrix_index, lensvector<double> &input_center_pts, const int& ii, const double weight, const int& thread);
	double find_lensed_surface_brightness_interpolate(lensvector<double> &input_center_pt, const int& thread);
	double find_local_inverse_magnification_interpolate(lensvector<double> &input_center_pt, const int& thread);
	double find_triangle_weighted_invmag(lensvector<double>& pt1, lensvector<double>& pt2, lensvector<double>& pt3, double& total_overlap, const int thread);

	double find_avg_n_images(const double sb_threshold_frac);

	void output_surface_brightness(Vector<double>& xvals, Vector<double>& yvals, Vector<double>& sbvals, Vector<double>& maglogvals, Vector<double>& nimgvals);
	//void plot_surface_brightness(string root);
	//void output_fits_file(string fits_filename);
	void get_grid_dimensions(double &xmin, double &xmax, double &ymin, double &ymax);
	void store_surface_brightness_grid_data(string root);

	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }
	~CartesianSourceGrid();

	//static ofstream bad_interps;

	// for plotting the grid to a file:
	std::ifstream sb_infile;
	std::ofstream xgrid;
	//std::ofstream pixel_surface_brightness_file;
	//std::ofstream pixel_magnification_file;
	//std::ofstream pixel_n_image_file;
};

static const int nmax_pts_interp = 120; // I had to take this out of DelaunayGrid so it could be seen by DelaunayGrid_Params

template <typename QScalar>
class DelaunayGrid_Params
{
	public:
	lensvector<QScalar> *interpolation_pts[nmax_pts_interp];
	QScalar interpolation_wgts[nmax_pts_interp];
	lensvector<QScalar> *polygon_vertices[nmax_pts_interp+2]; // the polygon referred to here is the part of the Voronoi cell contained in the Bower-Watson envelope for each vertex in the envelope.
	lensvector<QScalar> new_circumcenter[nmax_pts_interp];
	lensvector<QScalar> *gridpts;
	Triangle<QScalar> *triangle;
	QScalar kernel_correlation_length, matern_index;
};

class DelaunayGrid : private Sort
{
	public:
	DelaunayGrid_Params<double>* delaunay_params; // this will point to the corresponding delaunay_params in the inherited classes
#ifdef USE_STAN
	DelaunayGrid_Params<stan::math::var>* delaunay_params_dif; // this will point to the corresponding delaunay_params in the inherited classes
#endif

	private:
	template <typename QScalar>
	DelaunayGrid_Params<QScalar>& assign_delaunay_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return (*delaunay_params_dif);
		else
#endif
		return (*delaunay_params);
	}

	protected:
	DelaunayGrid_Params<double> delaunaygrid_params;
#ifdef USE_STAN
	DelaunayGrid_Params<stan::math::var> delaunaygrid_params_dif; // autodiff version
#endif

	protected:
	//lensvector<double> *interpolation_pts[nmax_pts_interp];
	//double interpolation_wgts[nmax_pts_interp];
	int interpolation_indx[nmax_pts_interp];
	int triangles_in_envelope[nmax_pts_interp];
	//lensvector<double> *polygon_vertices[nmax_pts_interp+2]; // the polygon referred to here is the part of the Voronoi cell contained in the Bower-Watson envelope for each vertex in the envelope.
	//lensvector<double> new_circumcenter[nmax_pts_interp];

	public:
	static bool zero_outside_border;
	int n_gridpts;
	int n_triangles;
	//lensvector<double> *gridpts;
	//Triangle<double> *triangle;
	int *adj_triangles[4];
	double avg_area;

	protected:
	double** voronoi_boundary_x;
	double** voronoi_boundary_y;
	double *voronoi_area;
	double *voronoi_length;
	int** shared_triangles;
	int* n_shared_triangles;
	// Used for calculating areas and finding whether points are inside a given cell
	//lensvector<double> dt1, dt2, dt3;
	//double prod1, prod2, prod3;

	public:
	DelaunayGrid();
#ifdef USE_STAN
	void sync_delaunaygrid_autodif_parameters();
#endif

	//static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	//static void deallocate_multithreaded_variables();
	template <typename QScalar>
	void create_pixel_grid(QScalar* gridpts_x, QScalar* gridpts_y, const int n_gridpts);

	int search_grid(int initial_srcpixel, const lensvector<double>& pt, bool& inside_triangle);
	bool test_if_inside(int &tri_number, const lensvector<double>& pt, bool& inside_triangle);
	bool test_if_inside(const int tri_number, const lensvector<double>& pt);
	void record_adjacent_triangles_xy();

	int find_closest_vertex(const int tri_number, const lensvector<double>& pt);
	double sum_edge_sqrlengths(const double min_sb);
	void find_containing_triangle(const lensvector<double> &input_pt, int& trinum, bool& inside_triangle, bool& on_vertex, int& kmin);
	//double interpolate(lensvector<double> &input_pt, const bool interp_mag = false, const int thread = 0);

	template <typename QScalar>
	void find_interpolation_weights_3pt(const lensvector<QScalar>& input_pt, const int trinum, int& npts, const int thread);
	template <typename QScalar>
	void find_interpolation_weights_nn(const lensvector<QScalar> &input_pt, const int trinum, int& npts, const int thread); // natural neighbor interpolation
	//void plot_voronoi_grid(string root);

	//void get_grid_points(vector<double>& xvals, vector<double>& yvals, vector<double>& sb_vals);
	//void generate_gmatrices(const bool interpolate);
	//void generate_hmatrices(const bool interpolate);
	//void generate_covariance_matrix_packed(double *cov_matrix_packed, const int kernel_type, const double epsilon, double *wgtfac = NULL, const bool add_to_covmatrix = false, const double amplitude = -1);
	void generate_covariance_matrix(Eigen::MatrixXd& cov_matrix, const int kernel_type, const double epsilon, double *wgtfac = NULL, const bool add_to_covmatrix = false, const double amplitude = -1);
	double modified_bessel_function(const double x, const double nu);
	void beschb(const double x, double& gam1, double& gam2, double& gampl, double& gammi);
	double chebev(const double a, const double b, double* c, const int m, const double x);

	void delete_grid_arrays();
	~DelaunayGrid();
};

template <typename QScalar>
class DelaunaySourceGrid_Params : public DelaunayGrid_Params<QScalar>, public ModelParams<QScalar>
{
	public:
	QScalar *surface_brightness;	
	QScalar regparam;
	QScalar regparam_lsc, regparam_lum_index;
	QScalar distreg_xcenter, distreg_ycenter, distreg_e1, distreg_e2, distreg_rc;
	QScalar mag_weight_sc, mag_weight_index;
	QScalar alpha_clus, beta_clus;
};

class DelaunaySourceGrid : public DelaunayGrid, public Model
{
	public:
	DelaunaySourceGrid_Params<double> delaunay_srcgrid_params;
#ifdef USE_STAN
	DelaunaySourceGrid_Params<stan::math::var> delaunay_srcgrid_params_dif; // autodiff version
#endif
	template <typename QScalar>
	DelaunaySourceGrid_Params<QScalar>& assign_delaunay_srcgrid_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return delaunay_srcgrid_params_dif;
		else
#endif
		return delaunay_srcgrid_params;
	}

	friend class QLens;
	friend class ImagePixelGrid;
	ImagePixelGrid *image_pixel_grid;

	public:
	double srcgrid_redshift;
	bool look_for_starting_point;
	int img_ni, img_nj;
	int n_active_pixels;
	//double *surface_brightness;	
	double *inv_magnification;
	bool *maps_to_image_pixel;
	bool *active_pixel;
	int *active_index;
	int **img_index_ij;
	int *imggrid_ivals;
	int *imggrid_jvals;
	//double srcpixel_xmin, srcpixel_xmax, srcpixel_ymin, srcpixel_ymax;
	double srcgrid_xmin, srcgrid_xmax, srcgrid_ymin, srcgrid_ymax; // for plotting
	int img_imin, img_imax, img_jmin, img_jmax;

	bool activate_unmapped_source_pixels;

	//double regparam;
	//double regparam_lsc, regparam_lum_index, distreg_rc;
	//double distreg_xcenter, distreg_ycenter, distreg_e1, distreg_e2; // for position-weighted regularization
	//double mag_weight_sc, mag_weight_index; // magnification-weighted regularization
	//double alpha_clus, beta_clus;

	public:
	DelaunaySourceGrid(QLens* qlens_in, const int band = 0, const double zsrc_in = -1);
	DelaunaySourceGrid(const DelaunaySourceGrid* pixsrc_in) : Model(), DelaunayGrid() {
		qlens = pixsrc_in->qlens;
		model_name = pixsrc_in->model_name;
		setup_parameters(true);
		copy_pixsrc_data(pixsrc_in);
	}
	void copy_pixsrc_data(const DelaunaySourceGrid* grid_in);
	void update_meta_parameters(const bool varied_only_fitparams);
	void get_parameter_numbers_from_qlens(int& pi, int& pf);
	bool register_vary_parameters_in_qlens();
	void register_limits_in_qlens();
	void update_fitparams_in_qlens();
	template <typename QScalar>
	void create_srcpixel_grid(QScalar* srcpts_x, QScalar* srcpts_y, const int n_srcpts, int* ivals_in = NULL, int* jvals_in = NULL, const int ni=0, const int nj=0, const bool find_pixel_magnification = false, const int imggrid_indx = -1);

	void setup_parameters(const bool initial_setup);
	template <typename QScalar>
	void setup_param_pointers();
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void find_pixel_magnifications();

	template <typename QScalar>
	void assign_surface_brightness_from_analytic_source(const int imggrid_i=-1);
	void fill_surface_brightness_vector();
	void update_surface_brightness(int& index);
	double sum_edge_sqrlengths(const double min_sb);
	template <typename QScalar>
	QScalar find_lensed_surface_brightness(const lensvector<QScalar> &input_pt, const int img_pixel_i, const int img_pixel_j, const int thread);
	bool find_containing_triangle_with_imgpix(const lensvector<double> &input_pt, const int img_pixel_i, const int img_pixel_j, int& trinum, bool& inside_triangle, bool& on_vertex, int& kmin);
	template <typename QScalar>
	QScalar interpolate_surface_brightness(const lensvector<QScalar> &input_pt, const bool interp_mag = false, const int thread = 0);
	double interpolate_voronoi_length(const lensvector<double> &input_pt, const int thread = 0);

	bool assign_source_mapping_flags(lensvector<double> &input_pt, vector<PtsWgts<double>>& mapped_delaunay_srcpixels, int& n_mapped_srcpixels, const int img_pixel_i, const int img_pixel_j, const int thread, bool& trouble_with_starting_vertex);
	void record_srcpixel_mappings();
	void calculate_Lmatrix(const int img_index, PtsWgts<double>* mapped_delaunay_srcpixels, int* n_mapped_srcpixels, int& index, const int& ii, const double weight, const int& thread);
	void calculate_Lmatrix_elements(const int img_index, PtsWgts<double>* mapped_delaunay_srcpixels, int* n_mapped_srcpixels, int& index, const int& subpixel_indx, const double weight, const int& thread);
	int assign_active_indices_and_count_source_pixels(const int source_pixel_i_initial, const bool activate_unmapped_pixels);
	void output_surface_brightness(Vector<double>& xvals, Vector<double>& yvals, Vector<double>& zvals, const int npix = 600, const bool interpolate = false, const bool plot_magnification = false);
	void plot_voronoi_grid(string root);
	double find_moment(const int p, const int q, const int npix, const double xc, const double yc, const double b, const double a, const double phi);
	void find_source_moments(const int npix, double &qs, double &phi_s, double &sigavg, double &xavg, double &yavg);

	void get_grid_points(vector<double>& xvals, vector<double>& yvals, vector<double>& sb_vals);
	void generate_gmatrices(const bool interpolate);
	void generate_hmatrices(const bool interpolate);
	void find_source_gradient(const lensvector<double>& input_pt, lensvector<double>& src_grad_neg, const int thread);

	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }

	int get_n_srcpts() { return n_gridpts; }
	void print_pixel_values();

	void delete_lensing_arrays();
	~DelaunaySourceGrid();
};

template <typename QScalar>
class LensPixelGrid_Params : public DelaunayGrid_Params<QScalar>, public ModelParams<QScalar>
{
	public:
	QScalar regparam;
	QScalar regparam_lsc, regparam_lum_index;
	QScalar distreg_xcenter, distreg_ycenter, distreg_e1, distreg_e2, distreg_rc;
	QScalar mag_weight_sc, mag_weight_index;
	QScalar alpha_clus, beta_clus;
};

class LensPixelGrid : public DelaunayGrid, public Model
{
	friend class QLens;
	friend class ImagePixelGrid;

	public:
	LensPixelGrid_Params<double> lensgrid_params;
#ifdef USE_STAN
	LensPixelGrid_Params<stan::math::var> lensgrid_params_dif; // autodiff version
#endif
	template <typename QScalar>
	LensPixelGrid_Params<QScalar>& assign_lensgrid_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return lensgrid_params_dif;
		else
#endif
		return lensgrid_params;
	}

	QLens *qlens;
	ImagePixelGrid *image_pixel_grid;

	public:
	int lens_redshift_idx;
	PixelGridType grid_type; // options are either CartesianPixelGrid or DelaunayPixelGrid
	bool include_in_lensing_calculations; // if true, include this pixelgrid in general lensing calculations

	// variables for Cartesian grid
	int npix_x, npix_y;
	double *xvals_cartesian;
	double *yvals_cartesian;
	int *cartesian_ivals;
	int *cartesian_jvals;
	int **cartesian_pixel_index;
	double xmin_cartesian, xmax_cartesian, ymin_cartesian, ymax_cartesian; // these give the x/y coordinates of the *center* of the pixels on the edges of the grid (as opposed to lensgrid_xmin etc.)
	double cartesian_pixel_xlength, cartesian_pixel_ylength;

	// variables for Delaunay grid
	bool look_for_starting_point;
	int img_ni, img_nj;
	double *potential;	
	double *def_x;	
	double *def_y;	
	double *hess_xx;	
	double *hess_yy;	
	double *hess_xy;	
	bool *maps_to_image_pixel;
	bool *active_pixel;
	int *active_index;

	double lensgrid_xmin, lensgrid_xmax, lensgrid_ymin, lensgrid_ymax; // for plotting

	bool activate_unmapped_source_pixels;

	//double regparam;
	//double regparam_lsc, regparam_lum_index, distreg_rc;
	//double distreg_xcenter, distreg_ycenter, distreg_e1, distreg_e2; // for position-weighted regularization
	//double mag_weight_sc, mag_weight_index; // magnification-weighted regularization
	//double alpha_clus, beta_clus;

	public:
	LensPixelGrid(QLens* lens_in, const int lens_redshift_indx_in);
	void copy_pixlens_data(LensPixelGrid* grid_in);
	void update_meta_parameters(const bool varied_only_fitparams);
	void create_cartesian_pixel_grid(const double x_min, const double x_max, const double y_min, const double y_max, const int redshift_indx = -1);
	void create_delaunay_pixel_grid(double* pts_x, double* pts_y, const int n_pts, const int redshift_indx = -1);

	void setup_parameters(const bool initial_setup);
	template <typename QScalar>
	void setup_param_pointers();
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif

	void fill_potential_correction_vector();
	void update_potential(int& index);
	void assign_potential_from_analytic_lens(const int imggrid_i, const bool add_potential);
	void assign_potential_from_analytic_lenses();

	double interpolate_potential(const double x, const double y, const int thread = 0);
	void deflection(const double x, const double y, lensvector<double>& def, const int thread);
	void hessian(const double x, const double y, lensmatrix<double>& hess, const int thread);
	double kappa(const double x, const double y, const int thread);
	void kappa_and_potential_derivatives(const double x, const double y, double& kappa, lensvector<double>& def, lensmatrix<double>& hess, const int thread);
	void potential_derivatives(const double x, const double y, lensvector<double>& def, lensmatrix<double>& hess, const int thread);
	bool assign_mapping_flags(lensvector<double> &input_pt, vector<PtsWgts<double>>& mapped_potpixels_ij, int& n_mapped_potpixels, const int img_pixel_i, const int img_pixel_j, const int thread);
	void calculate_Lmatrix(const int img_index, PtsWgts<double>* mapped_potpixels, int* n_mapped_potpixels, lensvector<double>& S0_derivs, int& index, const int& subpixel_indx, const int offset, const double weight, const int& thread);
	double first_order_surface_brightness_correction(const lensvector<double> &input_pt, const lensvector<double>& S0_deriv, const int thread);

	void find_interpolation_weights_cartesian(const lensvector<double> &input_pt, int& npts, const int deriv_type, const int thread);

	bool assign_lens_mapping_flags(lensvector<double> &input_pt, vector<PtsWgts<double>>& mapped_delaunay_lenspixels, int& n_mapped_lenspixels, const int img_pixel_i, const int img_pixel_j, const int thread, bool& trouble_with_starting_vertex);
	void record_lenspixel_mappings();
	int assign_active_indices_and_count_lens_pixels(const bool activate_unmapped_pixels);
	void plot_potential(string root, const int npix = 600, const bool interpolate = false, const bool plot_convergence = false, const bool plot_fits = false);

	void generate_gmatrices(const bool interpolate);
	void generate_hmatrices(const bool interpolate);

	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }
	void set_cartesian_npixels(const int nx, const int ny) { npix_x = nx; npix_y = ny; }

	void delete_lensing_arrays();
	~LensPixelGrid();
};

struct ConvPlan
{
	// This stores the coefficients for doing a direct PSF convolution (without using FFT)
	std::vector<int> offsets; // CSR-style offsets
	std::vector<int> in_idx;
	std::vector<double> weight;

	int out_size = 0;
	int in_size = 0;
	void resize(const int insize, const int outsize) {
		clear();
		in_size = insize;
		out_size = outsize;
		offsets.reserve(out_size+1);
	}
	void clear() {
		offsets.clear();
		in_idx.clear();
		weight.clear();
		out_size = 0;
		in_size = 0;
	}
};

template <typename MathTypes>
class ImgGrid_Params
{
	using QScalar = typename MathTypes::QScalar;
	using VecType = typename MathTypes::VecType;
	using MatType = typename MathTypes::MatType;

	public:
	VecType image_surface_brightness;
	VecType image_surface_brightness_emask;
	VecType image_surface_brightness_supersampled;
	VecType sbprofile_surface_brightness;
	VecType srcpt_x_centers, srcpt_y_centers;
	VecType srcpt_x_subpixel_centers, srcpt_y_subpixel_centers;

	ImgGrid_Params() {}
	void setup_ray_tracing_arrays(const int ntot_corners, const int img_npixels_emask, const int n_imgpixels, const int img_npixels_fgmask) {
		srcpt_x_centers = Eigen::VectorXd::Zero(img_npixels_emask);
		srcpt_y_centers = Eigen::VectorXd::Zero(img_npixels_emask);
		// Note, n_sb_cells could be number of pixels from the primary mask, or it could be from fgmask depending on settings
		image_surface_brightness = Eigen::VectorXd::Zero(n_imgpixels);
		image_surface_brightness_emask = Eigen::VectorXd::Zero(img_npixels_emask);
		sbprofile_surface_brightness = Eigen::VectorXd::Zero(img_npixels_fgmask);
	}
	void setup_subpixel_ray_tracing_arrays(const int n_subpixels_emask) {
		srcpt_x_subpixel_centers = Eigen::VectorXd::Zero(n_subpixels_emask);
		srcpt_y_subpixel_centers = Eigen::VectorXd::Zero(n_subpixels_emask);
		image_surface_brightness_supersampled = Eigen::VectorXd::Zero(n_subpixels_emask);
	}
};

class ImagePixelGrid : private Sort
{
	friend class QLens;
	friend class CartesianSourcePixel;
	friend class CartesianSourceGrid;
	friend class DelaunaySourceGrid;
	friend class LensPixelGrid;
	friend struct ImageData;
	friend class LensProfile;
	friend class PSF;

	private:
	ImgGrid_Params<PlainTypes> imggrid_params;
#ifdef USE_STAN
	ImgGrid_Params<VarmatTypes> imggrid_params_dif; // autodiff version
#endif
	template <typename MathTypes>
	ImgGrid_Params<MathTypes>& assign_imggrid_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<MathTypes, VarmatTypes>)
			return imggrid_params_dif;
		else
#endif
		return imggrid_params;
	}

	double **surface_brightness;
	double **foreground_surface_brightness;
	lensvector<double> **corner_sourcepts;
	lensvector<double> **center_sourcepts;
	lensvector<double> ***subpixel_center_sourcepts;
	double ***subpixel_surface_brightness;
	double *srcpt_x_corners, *srcpt_y_corners;
	double *twistx, *twisty;
	double *area_tri1, *area_tri2;
	Eigen::MatrixXd reduce_matrix;

	QLens *qlens;
	CartesianSourceGrid *cartesian_srcgrid;
	DelaunaySourceGrid *delaunay_srcgrid;
	ImageData *image_data;
	PSF *psf;
	LensPixelGrid *lensgrid;

	lensvector<double> **corner_pts;
	lensvector<double> **center_pts;
	lensvector<double> ***subpixel_center_pts;
	lensvector<double> ***subpixel_source_gradient;

	double* centerpts_x; // this will be turned into autodiff vector
	double* centerpts_y; // this will be turned into autodiff vector

	double **S0_check;
	lensvector<double> **x0_check;
	double ***subpixel_weights;
	int **subpixel_index_ss;

	double **noise_map;
	double **source_plane_triangle1_area; // area of triangle 1 (connecting points 0,1,2) when mapped to the source plane
	double **source_plane_triangle2_area; // area of triangle 2 (connecting points 1,3,2) when mapped to the source plane
	double **pixel_mag; // ratio of sum of source plane triangle areas over the image pixel area
	bool **pixel_in_mask;
	bool **mask;
	bool **emask;
	bool **fgmask;
	double pixel_area, triangle_area; // half of pixel area
	double min_srcplane_area;
	double max_sb;
	bool **maps_to_source_pixel;
	int max_nsplit;
	int imgpixel_nsplit;
	int n_subpix_per_pixel;
	int **nsplits;
	bool ***subpixel_maps_to_srcpixel;
	int **pixel_index;
	int **pixel_index_fgmask;
	int Lmatrix_n_amps;
	int Lmatrix_pot_npixels;

	int *image_pixel_i_from_subcell_ii;
	int *image_pixel_j_from_subcell_jj;

	ConvPlan psfconv_plan;
	ConvPlan psfconv_plan_fg;

	double *psf_zvec; // for convolutions using FFT
	double *psf_zvec_fgmask; // for convolutions using FFT
	int fft_imin, fft_jmin, fft_ni, fft_nj;
	int fft_imin_fgmask, fft_jmin_fgmask, fft_ni_fgmask, fft_nj_fgmask;
#ifdef USE_FFTW
	std::complex<double> *psf_transform;
	std::complex<double> *psf_transform_conj;
	std::complex<double> *psf_transform_fgmask;
	std::complex<double> *psf_transform_conj_fgmask;
	std::complex<double> **Lmatrix_transform;
	double **Lmatrix_imgs_rvec;
	double *single_img_rvec;
	double *single_img_rvec_fgmask;
	std::complex<double> *img_transform;
	std::complex<double> *img_transform_fgmask;
	fftw_plan fftplan;
	fftw_plan fftplan_inverse;
	fftw_plan fftplan_fgmask;
	fftw_plan fftplan_inverse_fgmask;
	fftw_plan *fftplans_Lmatrix;
	fftw_plan *fftplans_Lmatrix_inverse;
#ifdef USE_STAN
	// FFT plans and arrays for the adjoints (derivatives)
	double *adj_single_img_rvec;
	double *adj_single_img_rvec_fgmask;
	std::complex<double> *adj_img_transform;
	std::complex<double> *adj_img_transform_fgmask;
	fftw_plan adj_fftplan;
	fftw_plan adj_fftplan_inverse;
	fftw_plan adj_fftplan_fgmask;
	fftw_plan adj_fftplan_inverse_fgmask;
#endif
#endif
	bool psf_convolution_is_setup;
	bool fg_psf_convolution_is_setup;
	bool fft_convolution_is_setup;
	bool fg_fft_convolution_is_setup;

	int **twist_status;
	lensvector<double> **twist_pts;
	int *twiststat;
	int *mask_pixels_i, *mask_pixels_j, *emask_pixels_i, *emask_pixels_j, *fgmask_pixels_i, *fgmask_pixels_j;
	int *masked_pixel_corner_i, *masked_pixel_corner_j, *masked_pixel_corner, *masked_pixel_corner_up;
	int *extended_mask_subpixel_i, *extended_mask_subpixel_j, *extended_mask_subpixel_index, *emask_subpixels_ii, *emask_subpixels_jj;
	int *mask_subpixel_i, *mask_subpixel_j, *mask_subpixel_index;
	int **ncvals;

	lensvector<double> sourcept;

	bool include_potential_perturbations;

	long int ntot_corners, image_npixels, image_npixels_emask, image_npixels_fgmask;
	long int image_n_subpixels_emask, image_n_subpixels;
	int image_npixels_data; // right now, only used during optimization of regparam (and is only different from image_npixels when include_fgmask_in_inversion is used and there is padding of the fgmask)

	int n_pixsrc_to_include_in_Lmatrix;
	vector<int> imggrid_indx_to_include_in_Lmatrix;

	vector<CartesianSourcePixel*> **mapped_cartesian_srcpixels; // since the Cartesian grid uses recursion (if adaptive_subgrid is on), a pointer to each mapped source pixel is needed
	vector<PtsWgts<double>> **mapped_delaunay_srcpixels; // for the Delaunay grid, it only needs to record the index of each mapped source pixel (no pointer needed)
	vector<PtsWgts<double>> **mapped_potpixels; // for the Delaunay grid, it only needs to record the index of each mapped pixel (no pointer needed)
	int ***n_mapped_srcpixels; // will store how many source pixels map to a given (sub)pixel for Lmatrix
	int ***n_mapped_potpixels; // will store how many potential perturbation pixels map to a given (sub)pixel for Lmatrix
	SourceFitMode source_fit_mode;
	double xmin, xmax, ymin, ymax;
	double src_xmin, src_xmax, src_ymin, src_ymax; // for ray-traced points
	int x_N, y_N; // gives the number of cells in the x- and y- directions (so the number of corner points in each direction is x_N+1, y_N+1)
	int n_active_pixels;
	int n_high_sn_pixels;
	int xy_N; // gives x_N*y_N if the entire pixel grid is used
	double pixel_xlength, pixel_ylength;
	template <typename QScalar>
	bool test_if_between(const QScalar& p, const QScalar& a, const QScalar& b);
	int band_number;
	int src_redshift_index; // each ImagePixelGrid object is associated with a specific redshift
	int imggrid_index;
	double* imggrid_zfactors;
	double** imggrid_betafactors; // kappa ratio used for modeling source points at different redshifts

	int source_npixels, source_npixels_inv, lensgrid_npixels, n_mge_sets, n_mge_amps, source_and_lens_n_amps, n_amps; // note, n_amps can also include point image fluxes
	SB_Profile** mge_list;

	//Eigen::VectorXd image_surface_brightness;
	//Eigen::VectorXd image_surface_brightness_emask;
	//Eigen::VectorXd image_surface_brightness_supersampled;
	Eigen::VectorXd imgpixel_covinv_vector;
	Eigen::VectorXd point_image_surface_brightness;
	Eigen::VectorXd img_minus_sbprofile;
	Eigen::VectorXd amplitude_vector_minchisq; // used to store best-fit solution during optimization of regularization parameter
	Eigen::VectorXd amplitude_vector;
	Eigen::VectorXi img_index_datapixels;

	int *image_pixel_location_Lmatrix;
	int Lmatrix_n_elements;
	double *Lmatrix_sparse;
	int *Lmatrix_index;
	std::vector<double> *Lmatrix_rows;
	std::vector<int> *Lmatrix_index_rows;

	Eigen::VectorXd Dvector;
	Eigen::VectorXd Dvector_cov;
	Eigen::VectorXd Dvector_cov_copy;
	int Fmatrix_nn;
	double *Fmatrix_sparse;
	double *Fmatrix_copy; // used when optimizing the regularization parameter
	int *Fmatrix_index;
	double regopt_chisqmin, regopt_logdet;
	double *reg_weight_factor;

	double *regparam_ptr; // points to regularization parameter for given source pixel grid or shapelet object
	double *regparam_pot_ptr; // points to regularization parameter for potential of given lens pixel grid

	int n_src_inv; // specifies how many pixellated (or shapelet) sources will be included in the Lmatrix
	int src_npixels_inv; // gives # of srcpixels for src associated with this ImagePixelGrid (source_npixels may be larger if other ImagePixelGrid's are included in inversion)
	int src_npixel_start; // gives the source pixel index in Lmatrix/Fmatrix where the source pixels for this source begin (may not be zero if we're including multiple sources)

	double *Rmatrix_sparse;
	int *Rmatrix_index;
	double *Rmatrix_pot;
	int *Rmatrix_pot_index;

	Eigen::MatrixXd covmatrix_dense;
	Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> covmatrix_factored;
	Eigen::MatrixXd Rmatrix_dense;

	double **Rmatrix_ptr; // can either point to Rmatrix for source pixels or Rmatrix for potential corrections
	int **Rmatrix_index_ptr; // can either point to Rmatrix_index for source pixels or Rmatrix_index for potential corrections
	Eigen::MatrixXd *Rmatrix_dense_ptr; // can either point to Rmatrix* for source pixels or Rmatrix for potential corrections

	// The following are simply used as temporary arrays when constructing Rmatrix
	double *Rmatrix_diag_temp;
	std::vector<double> *Rmatrix_rows;
	std::vector<int> *Rmatrix_index_rows;
	int *Rmatrix_row_nn;

	dvector *Rmatrix_MGE_packed;
	double *Rmatrix_MGE_log_determinants;

	Eigen::MatrixXd Lmatrix_trans_dense;
	Eigen::MatrixXd Lmatrix_trans_supersampled;

	Eigen::MatrixXd Fmatrix_dense;
	Eigen::MatrixXd Fmatrix_dense_copy;
	Eigen::MatrixXd Gmatrix;
	Eigen::MatrixXd Gmatrix_copy;

	double Fmatrix_log_determinant;
	double Gmatrix_log_determinant;

	double Rmatrix_log_determinant;
	double Rmatrix_pot_log_determinant;

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

	double *saved_sbweights;
	int n_sbweights;

	bool assign_pixel_mappings(const bool potential_perturbations=false, const bool verbal=false);
	void assign_foreground_mappings(const bool use_data = true);
	void construct_Lmatrix_shapelets();
	void add_MGE_amplitudes_to_Lmatrix();
	void PSF_convolution_Lmatrix_dense(const bool verbal=false);
	void create_lensing_matrices_from_Lmatrix_dense(const bool potential_perturbations=false, const bool verbal=false);
	void get_source_regparam_ptr(const int imggrid_include_i, double* &regparam);
	void generate_Gmatrix();
	void add_regularization_term_to_dense_Fmatrix(ImagePixelGrid* imggrid, double *regparam, const bool potential_perturbations=false);
	void add_MGE_regularization_terms_to_dense_Fmatrix();
	double calculate_regularization_prior_term(double *regparam_ptr, const bool potential_perturbations=false);
	double calculate_MGE_regularization_prior_term();
	void add_regularization_prior_terms_to_logev(double& logev_times_two, double& loglike_reg, double& regterms, const bool include_potential_perturbations = false, const bool verbal = false);

	bool optimize_regularization_parameter(const bool dense_Fmatrix=false, const bool verbal=false, const bool pre_srcgrid = false);
	void setup_regparam_optimization(const bool dense_Fmatrix=false);
	void calculate_subpixel_sbweights(const bool save_sbweights = false, const bool verbal = false);
	void calculate_subpixel_distweights();
	void find_srcpixel_weights();
	void load_pixel_sbweights();
	double chisq_regparam_dense(const double logreg);
	double chisq_regparam(const double logreg);
	void calculate_lumreg_srcpixel_weights(const bool use_sbweights=false);
	void calculate_distreg_srcpixel_weights(const double xc=0, const double yc=0, const double sig=1.0, const bool verbal = false);
	void calculate_srcpixel_scaled_distances(const double xc, const double yc, const double sig, double *dists, lensvector<double> **srcpts, const int nsrcpts, const double e1 = 0, const double e2 = 0);
	void calculate_mag_srcpixel_weights();

	//void add_lum_weighted_reg_term(const bool dense_Fmatrix, const bool use_matrix_copies);
	double brents_min_method(double (ImagePixelGrid::*func)(const double), const double ax, const double bx, const double tol, const bool verbal);
	void create_regularization_matrix_shapelet();
	void create_MGE_regularization_matrices();
	void generate_Rmatrix_shapelet_gradient();
	void generate_Rmatrix_shapelet_curvature();

	void initialize_pixel_matrices(const bool potential_perturbations=false, bool verbal=false);
	void initialize_pixel_matrices_shapelets(bool verbal=false);
	void count_shapelet_amplitudes();
	void count_MGE_amplitudes(int& n_mge_objects, int& n_gaussians);
	void clear_pixel_matrices();
	void clear_sparse_lensing_matrices();
	void construct_Lmatrix(const bool delaunay=true, const bool potential_perturbations=false, const bool verbal=false);
	void construct_Lmatrix_supersampled(const bool delaunay=true, const bool potential_perturbations=false, const bool verbal=false);
	void construct_Lmatrix_dense(const bool delaunay, const bool potential_perturbations, const bool verbal);
	void PSF_convolution_Lmatrix(bool verbal = false);
	void PSF_convolution_pixel_vector(const bool foreground = false, const bool verbal = false, const bool use_fft = false, const bool use_emask = false);
	void average_supersampled_image_surface_brightness();
	void average_supersampled_dense_Lmatrix();
	void copy_FFT_convolution_arrays(QLens* lens_in);
	void fourier_transform(double* data, const int ndim, int* nn, const int isign);
	void fourier_transform_parallel(double** data, const int ndata, const int jstart, const int ndim, int* nn, const int isign);

	bool create_regularization_matrix(const bool include_lum_weighting = false, const bool use_sbweights = false, const bool potential_perturbations = false, const bool verbal = false);
	void generate_Rmatrix_from_gmatrices(const bool interpolate = false, const bool potential_perturbations = false);
	void generate_Rmatrix_from_hmatrices(const bool interpolate = false, const bool potential_perturbations = false);
	void generate_Rmatrix_norm(const bool potential_perturbations = false);
	bool generate_Rmatrix_from_covariance_kernel(const int kernel_type=0, const bool include_lum_weighting=false, const bool potential_perturbations = false, const bool verbal = false);

	void create_lensing_matrices_from_Lmatrix(const bool dense_Fmatrix=false, const bool potential_perturbations=false, const bool verbal=false);
	void invert_lens_mapping_dense(bool verbal=false);
	void invert_lens_mapping_EIGEN_sparse(double& logdet, const bool verbal, const bool use_copy = false);
	void invert_lens_mapping_MUMPS(double& logdet, const bool verbal, const bool use_copy = false);
	void invert_lens_mapping_UMFPACK(double& logdet, const bool verbal, const bool use_copy = false);
	void Rmatrix_determinant_EIGEN(const bool potential_perturbations);

	void invert_lens_mapping_CG_method(bool verbal);
	void update_source_and_lensgrid_amplitudes(const bool verbal=false);
	void indexx(int* arr, int* indx, int nn);

	void calculate_source_pixel_surface_brightness();
	void calculate_image_pixel_surface_brightness();
	void calculate_image_pixel_surface_brightness_dense();
	void calculate_foreground_pixel_surface_brightness(const bool allow_lensed_nonshapelet_sources = true);
	void store_image_pixel_surface_brightness(const bool use_emask = false);
	void store_foreground_pixel_surface_brightness();
	void vectorize_image_pixel_surface_brightness(const bool use_emask = false);

	static int nthreads;
	static const int max_iterations, max_step_length;
	static bool *newton_check;
	static lensvector<double> *fvec;
	static double image_pos_accuracy;

	bool run_newton(lensvector<double>& xroot, double& mag, const int& thread);
	bool LineSearch(lensvector<double>& xold, double fold, lensvector<double>& g, lensvector<double>& p, lensvector<double>& x, double& f, double stpmax, bool &check, const int& thread);
	bool NewtonsMethod(lensvector<double>& x, bool &check, const int& thread);
	void SolveLinearEqs(lensmatrix<double>&, lensvector<double>&);
	bool redundancy(const lensvector<double>&, double &);
	double max_component(const lensvector<double>&);

	// for calculating wall time in parallel calculations
	std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;

	public:
	ImagePixelGrid(QLens* lens_in, SourceFitMode mode, double xmin_in, double xmax_in, double ymin_in, double ymax_in, int x_N_in, int y_N_in, const bool raytrace = false, const int band_number_in = 0, int src_redshift_index = -1, const int imggrid_index_in = -1);
	//ImagePixelGrid(QLens* lens_in, SourceFitMode mode, double** sb_in, const int x_N_in, const int y_N_in, const int reduce_factor, double xmin_in, double xmax_in, double ymin_in, double ymax_in, const int src_redshift_index = -1);
	ImagePixelGrid(QLens* lens_in, SourceFitMode mode, ImageData& pixel_data, const bool include_fgmask = false, const int band_number_in = 0, const int src_redshift_index = -1, const int imggrid_index_in = -1, const int mask_index = 0, const bool setup_mask_and_data = true, const bool verbal = false);
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	void update_zfactors_and_beta_factors();
	void set_include_in_Lmatrix(const int imggrid_i);
	void set_include_only_one_pixsrc_in_Lmatrix();

	//ImagePixelGrid(QLens* lens_in, double* zfactor_in, double** betafactor_in, SourceFitMode mode, ImageData& pixel_data);
	void load_data(ImageData& pixel_data);
	void generate_point_images(const vector<image<double>>& imgs, double* ptimage_surface_brightness, const bool use_img_fluxes, const double srcflux, const int img_num = -1); // -1 means use all images
	void add_point_images(Eigen::VectorXd& ptimage_surface_brightness, const int npix);
	void generate_and_add_point_images(const vector<image<double>>& imgs, const bool include_imgfluxes, const double srcflux);
	void find_point_images(const double src_x, const double src_y, vector<image<double>>& imgs, const bool use_overlap, const bool is_lensed, const bool verbal);
	//bool test_if_inside_cell(const lensvector<double>& point, const int& i, const int& j);
	void assign_mask_pointers(ImageData& pixel_data, const int mask_index);
	bool set_fit_window(ImageData& pixel_data, const bool raytrace = false, const int mask_k = 0, const bool redo_fft = true, const bool use_fgmask = false);
	void include_all_pixels(const bool redo_fft = true);
	void activate_extended_mask(const bool redo_fft = true);
	void activate_foreground_mask(const bool redo_fft = true, const bool datamask = false);
	//void deactivate_extended_mask(const bool redo_fft = true, const bool use_fgmask = false);
	void update_mask_values(const bool use_fgmask = false);

	void setup_pixel_arrays();
	void set_null_ray_tracing_arrays();
	void set_null_subpixel_ray_tracing_arrays();
	void setup_ray_tracing_arrays(const bool include_fft_arrays = true, const bool verbal = false);
	void setup_subpixel_ray_tracing_arrays(const bool verbal = false);
	void delete_ray_tracing_arrays(const bool delete_fft_arrays = true);
	void delete_subpixel_ray_tracing_arrays();
	void update_grid_dimensions(const double xmin, const double xmax, const double ymin, const double ymax);
	template <typename MathTypes>
	void calculate_sourcepts_and_areas(const bool raytrace_pixel_centers = false, const bool verbal = false);
	bool calculate_subpixel_source_gradient();
	void set_nsplits_from_lens_settings();
	void set_nsplits(const bool split_pixels);
	void setup_noise_map(QLens* lens_in);

	void setup_PSF_convolution(const bool foreground = false);
	template <typename VecType>
	VecType PSF_convolution_pixel_vector_stan(const VecType& sbvec, const bool foreground = false);
	void reset_psfconv_plans();

	bool setup_FFT_convolution(const bool supersampling, const bool foreground, const bool include_fgmask_in_inversion, const bool verbal);
	void cleanup_FFT_convolution_arrays();
	void cleanup_foreground_FFT_convolution_arrays();

	template <typename MathTypes>
	void PSF_convolution_pixel_vector_wrapper(const bool foreground = false, const bool verbal = false, const bool use_fft = false, const bool use_extended_mask = false);
	template <typename VecType>
	VecType PSF_convolution_pixel_vector_stan_FFT(const VecType& sbvec, const bool foreground = false, const bool verbal = false);

	~ImagePixelGrid();
	template <typename MathTypes>
	void redo_lensing_calculations(const bool verbal = false);
	void redo_lensing_calculations_corners();
	void assign_mask_pixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& count, ImageData* data_in);

	void find_optimal_sourcegrid(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double &sourcegrid_limit_xmin, const double &sourcegrid_limit_xmax, const double &sourcegrid_limit_ymin, const double& sourcegrid_limit_ymax);
	void set_sourcegrid_params_from_ray_tracing(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double sourcegrid_limit_xmin, const double sourcegrid_limit_xmax, const double sourcegrid_limit_ymin, const double sourcegrid_limit_ymax);


	double find_approx_source_size(double& xcavg, double& ycavg, const bool verbal = false);
	void find_optimal_shapelet_scale(double& scale, double& xcenter, double& ycenter, double& recommended_nsplit, const bool verbal, double& sig, double& scaled_maxdist);
	void set_surface_brightness_vector_to_data();
	void plot_grid(string filename, bool show_inactive_pixels);
	void set_lens(QLens* qlensptr) { qlens = qlensptr; }
	void set_cartesian_srcgrid(CartesianSourceGrid* source_pixel_ptr) { cartesian_srcgrid = source_pixel_ptr; }
	void set_delaunay_srcgrid(DelaunaySourceGrid* delaunayptr) { delaunay_srcgrid = delaunayptr; }
	void set_lensgrid(LensPixelGrid* gridptr) { lensgrid = gridptr; }
	void find_optimal_sourcegrid_npixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels);
	void find_optimal_firstlevel_sourcegrid_npixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels);
	void find_surface_brightness(const bool use_emask = false, const bool foreground_only = false, const bool lensed_sources_only = false, const bool include_first_order_corrections = false, const bool show_only_first_order_corrections = false, const bool omit_noninverted_sources = false);
	template <typename MathTypes>
	void find_surface_brightness_sbprofile(const bool foreground_only = false, const bool lensed_sources_only = false, const bool omit_noninverted_sources = false, const bool psf_convolution = false);

	template <typename QScalar>
	void set_zero_lensed_surface_brightness();
	template <typename QScalar>
	void set_zero_foreground_surface_brightness();

	double output_surface_brightness(Vector<double>& xvals, Vector<double>& yvals, Vector<double>& zvals, bool plot_residual = false, bool normalize_sb = false, bool show_noise_thresh = false, bool plot_log = false, bool show_foreground_mask = false);
	void plot_sourcepts(string outfile_root, const bool show_subpixels = false);
	//void output_fits_file(string fits_filename, bool plot_residual = false);

	void add_pixel_noise();
	void set_image_pixel_data(ImageData* imgdata, const int mask_index);
	void set_uniform_pixel_noise(const double pn)
	{
		if (noise_map != NULL) {
			int i,j;
			for (i=0; i < x_N; i++) {
				for (j=0; j < y_N; j++) {
					noise_map[i][j] = pn;
				}
			}
		}
		if (qlens) qlens->use_noise_map = true;
	}
	double calculate_signal_to_noise(double &total_signal);
	void assign_image_mapping_flags(const bool delaunay, const bool potential_perturbations = false, const bool map_all_imgpixels = false);
	int count_nonzero_source_pixel_mappings_cartesian();
	int count_nonzero_source_pixel_mappings_delaunay();
	int count_nonzero_lensgrid_pixel_mappings();
	void get_grid_params(double& xmin_in, double& xmax_in, double& ymin_in, double& ymax_in, int& npx, int& npy) { xmin_in = xmin; xmax_in = xmax; ymin_in = ymin; ymax_in = ymax; npx = x_N; npy = y_N; }
};

template <typename QScalar>
class PSF_Params : public ModelParams<QScalar>
{
	public:
	//QScalar pos_x, pos_y;
	QScalar psf_width_x, psf_width_y;
	QScalar psf_offset_x, psf_offset_y;

	QScalar zsrc, srcflux;
};

class PSF : public Model
{
	friend class QLens;
	friend class ImagePixelGrid;
	friend struct ImageData;

	public:
	PSF_Params<double> psf_params;
#ifdef USE_STAN
	PSF_Params<stan::math::var> psf_params_dif; // autodiff version
#endif
	template <typename QScalar>
	PSF_Params<QScalar>& assign_psf_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return psf_params_dif;
		else
#endif
		return psf_params;
	}

	QLens *qlens;
	ImagePixelGrid *image_pixel_grid;

	double **psf_matrix;
	double **supersampled_psf_matrix;
	//double psf_width_x, psf_width_y;
	//double psf_offset_x, psf_offset_y;

	bool use_input_psf_matrix;
	Spline2D<double> psf_spline;
	int psf_npixels_x, psf_npixels_y;
	int supersampled_psf_npixels_x, supersampled_psf_npixels_y;
	string psf_filename;

	public:
	PSF(QLens* lens_in);
	void copy_psf_data(PSF* grid_in);
	void get_parameter_numbers_from_qlens(int& pi, int& pf);
	void setup_parameters(const bool initial_setup);
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif
	template <typename QScalar>
	void setup_param_pointers();
	void delete_psf_matrix();
	~PSF();

	bool generate_PSF_matrix(const double pixel_xlength, const double pixel_ylength, const bool supersampling);
	void generate_supersampled_PSF_matrix(const bool downsample = false, const int downsample_fac = 1);
	bool spline_PSF_matrix(const double xstep, const double ystep);
	double interpolate_PSF_matrix(double x, double y, const bool supersampled);
	bool load_psf_fits(string fits_filename, const int hdu_indx, const bool supersampled, const bool show_header = false, const bool verbal = false);
	bool save_psf_fits(string fits_filename, const bool supersampled = false);
	bool plot_psf(string filename, const bool supersampled, const double xstep=1.0, const double ystep=1.0);
	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }
};

class SB_Profile;

struct ImageData : private Sort
{
	friend class QLens;
	QLens *qlens;
	int band_number;
	int npixels_x, npixels_y;
	double **surface_brightness;
	double **noise_map;
	double **covinv_map; // this is simply 1/SQR(noise_map)
	bool **high_sn_pixel; // used to help determine optimal source pixel size based on area the high S/N pixels cover when mapped to source plane
	int n_masks;
	int *n_mask_pixels;
	int *extended_mask_n_neighbors;
	bool ***in_mask;
	bool ***extended_mask;
	bool **foreground_mask;
	bool **foreground_mask_data;
	double *xvals, *yvals, *pixel_xcvals, *pixel_ycvals;
	int n_high_sn_pixels;
	double xmin, xmax, ymin, ymax;
	double pixel_size, pixel_xy_ratio; // pixel_xy_ratio is ratio of pixel y/x lengths
	double emask_rmax; // used only when splining Fourier mode integrals for non-elliptical structure
	double bg_pixel_noise; // set using lowest noise dispersion in noise map
	string data_fits_filename;
	string noise_map_fits_filename;
	std::ostream* isophote_fit_out;
	ImageData(const int band_in = 0)
	{
		band_number = band_in;
		npixels_x = 0;
		npixels_y = 0;
		pixel_size = 0.0;
		xmin=xmax=ymin=ymax=0.0;
		pixel_xy_ratio = 1.0;
		surface_brightness = NULL;
		noise_map = NULL;
		covinv_map = NULL;
		high_sn_pixel = NULL;
		n_masks = 0;
		n_mask_pixels = NULL;
		in_mask = NULL;
		extended_mask = NULL;
		extended_mask_n_neighbors = NULL;
		foreground_mask = NULL;
		foreground_mask_data = NULL;
		xvals = NULL;
		yvals = NULL;
		pixel_xcvals = NULL;
		pixel_ycvals = NULL;
		qlens = NULL;
		isophote_fit_out = &std::cout;
		noise_map_fits_filename = "";
	}
	~ImageData();
	void load_data(string root);
	void load_from_image_grid(ImagePixelGrid* image_pixel_grid);
	bool load_noise_map_fits(string fits_filename, const int hdu_indx = 1, const bool show_header = false);
	bool save_noise_map_fits(string fits_filename, const bool subimage=false, const double xmin_in=-1e30, const double xmax_in=1e30, const double ymin_in=-1e30, const double ymax_in=1e30);
	void unload_noise_map();
	void set_isofit_output_stream(std::ofstream *fitout) { isophote_fit_out = fitout; }
	void set_uniform_pixel_noise(const double noise)
	{
		if (noise_map != NULL) {
			int i,j;
			double covinv = 1.0/(noise*noise);
			for (i=0; i < npixels_x; i++) {
				for (j=0; j < npixels_y; j++) {
					noise_map[i][j] = noise;
					covinv_map[i][j] = covinv;
				}
			}
		}
	}
	bool load_data_fits(string fits_filename, const double pixel_size_in, const double pixel_xy_ratio_in = 1.0, const double x_offset = 0.0, const double y_offset = 0.0, const int hdu_indx = 1, const bool show_header = false);
	void save_data_fits(string fits_filename, const bool subimage=false, const double xmin_in=-1e30, const double xmax_in=1e30, const double ymin_in=-1e30, const double ymax_in=1e30);
	bool load_mask_fits(const int mask_k, const string fits_filename, const bool foreground=false, const bool emask=false, const bool add_mask=false, const bool subtract_mask_pixels=false);
	bool save_mask_fits(string fits_filename, const bool foreground=false, const bool emask=false, const int mask_k=0, const bool subimage=false, const double xmin_in=-1e30, const double xmax_in=1e30, const double ymin_in=-1e30, const double ymax_in=1e30);
	bool copy_mask(ImageData* data, const int mask_k = 0);
	void assign_high_sn_pixels();
	double find_max_sb(const int mask_k = 0);
	double find_avg_sb(const double sb_threshold, const int mask_k = 0);
	bool create_new_mask();
	bool set_no_mask_pixels(const int mask_k = 0);
	bool set_all_mask_pixels(const int mask_k = 0);
	bool set_foreground_mask_to_primary_mask(const int mask_k = 0);
	bool invert_mask(const int mask_k = 0);
	bool inside_mask(const double x, const double y, const int mask_k = 0);
	bool assign_mask_windows(const double sb_noise_threshold, const int threshold_size = 0, const int mask_k = 0);
	bool unset_low_signal_pixels(const double sb_threshold, const int mask_k = 0);
	bool set_positive_radial_gradient_pixels(const int mask_k = 0);
	bool set_neighbor_pixels(const bool only_interior_neighbors, const bool only_exterior_neighbors, const int mask_k = 0);
	bool expand_foreground_mask(const int n_it);
	bool set_mask_window(const double xmin, const double xmax, const double ymin, const double ymax, const bool unset = false, const int mask_k = 0);
	bool set_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1, double theta2, const double xstretch, const double ystretch, const bool unset = false, const bool foreground = false, const int mask_k = 0);
	bool reset_extended_mask(const int mask_k = 0);
	bool set_extended_mask(const int n_neighbors, const bool add_to_emask = false, const bool only_interior_neighbors = false, const int mask_k = 0);
	bool set_extended_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset = false, const int mask_k = 0);
	void remove_overlapping_pixels_from_other_masks(const int mask_k = 0);
	bool activate_partner_image_pixels(const int mask_k = 0, const bool emask = false);
	void find_extended_mask_rmax();
	void set_foreground_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset = false);

	long int get_size_of_extended_mask(const int mask_k = 0);
	long int get_size_of_foreground_mask();
	bool test_if_in_fit_region(const double& x, const double& y, const int mask_k = 0);
	void set_lens(QLens* qlensptr) {
		qlens = qlensptr;
		//pixel_size = qlens->default_data_pixel_size;
	}

	bool estimate_pixel_noise(const double xmin, const double xmax, const double ymin, const double ymax, double &noise, double &mean_sb, const int mask_k = 0);
	//void add_point_image_from_centroid(ImageData* point_image_data, const double xmin_in, const double xmax_in, const double ymin_in, const double ymax_in, const double sb_threshold, const double pixel_error);
	void get_grid_params(double& xmin_in, double& xmax_in, double& ymin_in, double& ymax_in, int& npx, int& npy);
	void get_npixels(int& npx, int& npy)
	{
		npx = npixels_x;
		npy = npixels_y;
	}
	void set_grid_params(double xmin_in, double xmax_in, double ymin_in, double ymax_in)
	{
		xmin=xmin_in;
		xmax=xmax_in;
		ymin=ymin_in;
		ymax=ymax_in;
	}
	void set_pixel_size(const double size) { pixel_size = size; }

	bool fit_isophote(const double xi0, const double xistep, const int emode, const double qi, const double theta_i, const double xc_i, const double yc_i, const int max_it, IsophoteData& isophote_data, const bool use_polar_higher_harmonics = false, const bool verbose = true, SB_Profile* sbprofile = NULL, const int default_sampling_mode = 2, const int n_higher_harmonics = 2, const bool fix_center = false, const int max_xi_it = 10000, const double ximax_in = -1, const double rms_sbgrad_rel_threshold = 1.0, const double npts_frac = 0.5, const double rms_sbgrad_rel_transition = 0.3, const double npts_frac_zeroweight = 0.5);
	double sample_ellipse(const bool show_warnings, const double xi, const double xistep, const double epsilon, const double theta, const double xc, const double yc, int& npts, int& npts_sample, const int emode, int& sampling_mode, const double pixel_noise, double* sbvals = NULL, double* sbgrad_wgts = NULL, SB_Profile* sbprofile = NULL, const bool fill_matrices = false, double* sb_residual = NULL, double* sb_weights = NULL, double** smatrix = NULL, const int ni = 1, const int nf = 2, const bool use_polar_higher_harmonics = false, const bool plot_ellipse = false, std::ofstream* ellout = NULL);
	bool Cholesky_dcmp(double** a, int n);
	void Cholesky_solve(double** a, double* b, double* x, int n);
	void Cholesky_fac_inverse(double** a, int n);
	void output_surface_brightness(Vector<double>& xvals_in, Vector<double>& yvals_in, Vector<double>& sbvals_in, bool show_only_mask, bool show_extended_mask = false, bool show_foreground_mask = false, const int mask_k = 0);
	std::string mkstring_int(const int i);
	std::string mkstring_doub(const double db);
	std::string get_imgdata_info_string();
};

#endif // PIXELGRID_H
