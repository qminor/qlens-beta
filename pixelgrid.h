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
class SourcePixel;
class SourcePixelGrid;
struct ImagePixelData;

enum PixelGridType { CartesianPixelGrid, DelaunayPixelGrid };

struct InterpolationCells {
	bool found_containing_cell;
	SourcePixel *pixel[3];
};

struct PtsWgts {
	int indx;
	double wgt;
	PtsWgts() {}
	PtsWgts(const int indx_in, const double wgt_in) {
		indx = indx_in;
		wgt = wgt_in;
	}
	PtsWgts& assign(const int indx_in, const double wgt_in) {
		indx = indx_in;
		wgt = wgt_in;
		return *this;
	}
};

class SourcePixel
{
	friend class QLens;
	friend class ImagePixelGrid;
	friend class SourcePixelGrid;

	protected: 
	QLens *lens;
	ImagePixelGrid *image_pixel_grid;
	SourcePixelGrid *parent_grid; // this points to the top-level grid
	SourcePixel ***cell;
	SourcePixel *neighbor[4]; // 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	int ii, jj; // this is the index assigned to this cell in the grid of the parent cell

	int u_N, w_N;
	int level;
	double cell_area;
	lensvector center_pt;
	lensvector corner_pt[4];
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
	static lensvector **interpolation_pts[3];
	static int *maxlevs;
	static lensvector ***xvals_threads;
	static lensvector ***corners_threads;
	static lensvector **twistpts_threads;
	static int **twist_status_threads;

	void split_cells(const int usplit, const int wsplit, const int& thread);
	void unsplit();
	void split_subcells(const int splitlevel, const int thread);

	void assign_level_neighbors(int neighbor_level);
	void test_neighbors();
	void assign_indices(int& source_pixel_i);
	void assign_active_indices(int& source_pixel_i);

	public:
	SourcePixel() {}
	SourcePixel(QLens* lens_in, lensvector** xij, const int& i, const int& j, const int& level_in, SourcePixelGrid* parent_ptr);
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	inline bool check_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	inline bool check_if_in_neighborhood(lensvector **input_corner_pts, bool &inside, const int& thread);
	inline double find_rectangle_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread, const int&, const int&);
	inline bool check_triangle1_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	inline bool check_triangle2_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	inline double find_triangle1_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	inline double find_triangle2_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);

	void generate_gmatrices();
	void generate_hmatrices();

	void subcell_assign_source_mapping_flags_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, vector<SourcePixel*>& mapped_cartesian_srcpixels, const int& thread, bool& image_pixel_maps_to_source_grid);
	void find_lensed_surface_brightness_subcell_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread, double& overlap, double& total_overlap, double& total_weighted_surface_brightness);

	bool subcell_assign_source_mapping_flags_interpolate(lensvector &input_center_pt, vector<SourcePixel*>& mapped_cartesian_srcpixels, const int& thread);
	void calculate_Lmatrix_interpolate(const int img_index, vector<SourcePixel*>& mapped_cartesian_srcpixels, int& Lmatrix_index, lensvector &input_center_pts, const int& ii, const double weight, const int& thread);
	void find_triangle_weighted_invmag_subcell(lensvector& pt1, lensvector& pt2, lensvector& pt3, double& total_overlap, double& total_weighted_invmag, const int& thread);

	void find_interpolation_cells(lensvector &input_center_pt, const int& thread);
	SourcePixel* find_nearest_neighbor_cell(lensvector &input_center_pt, const int& side);
	SourcePixel* find_nearest_neighbor_cell(lensvector &input_center_pt, const int& side, const int tiebreaker_side);
	void find_nearest_two_cells(SourcePixel* &cellptr1, SourcePixel* &cellptr2, const int& side);
	SourcePixel* find_corner_cell(const int i, const int j);

	void assign_surface_brightness_from_analytic_source(const int zsrc_i=-1);
	void assign_surface_brightness_from_delaunay_grid(DelaunaySourceGrid* delaunay_grid, const bool add_sb = false);
	void update_surface_brightness(int& index);
	void fill_surface_brightness_vector();
	void fill_surface_brightness_vector_recursive(int& column_j);
	void fill_n_image_vector();

	void fill_n_image_vector_recursive(int& column_j);
	void plot_surface_brightness(string root);
	void output_fits_file(string fits_filename);
	void get_grid_dimensions(double &xmin, double &xmax, double &ymin, double &ymax);
	void plot_cell_surface_brightness(int line_number, int pixels_per_cell_x, int pixels_per_cell_y, std::ofstream& sb_outfile, std::ofstream& mag_outfile, std::ofstream& nimg_outfile);
	void store_surface_brightness_grid_data(string root);
	void write_surface_brightness_to_file(std::ofstream &sb_outfile);
	void read_surface_brightness_data(std::ifstream &sb_infile);

	void clear_subgrids();
	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }
	void plot_corner_coordinates(std::ofstream &gridout);
	void clear(void);
	~SourcePixel();
};

class SourcePixelGrid : public SourcePixel, public ModelParams
{
	friend class QLens;
	friend class ImagePixelGrid;
	friend class SourcePixel;

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
	int assign_active_indices_and_count_source_pixels(bool regrid_if_inactive_cells, bool activate_unmapped_pixels, bool exclude_pixels_outside_window);
	void split_subcells_firstlevel(const int splitlevel);

	void print_indices();

	public:
	SourcePixelGrid(QLens* lens_in);
	void copy_pixsrc_data(SourcePixelGrid* grid_in);
	void update_meta_parameters(const bool varied_only_fitparams);
	void create_pixel_grid(QLens* lens_in, const double x_min, const double x_max, const double y_min, const double y_max, const int usplit0, const int wsplit0);
	void create_pixel_grid(QLens* lens_in, string pixel_data_fileroot, const double minarea_in);
	//void copy_source_pixel_grid(SourcePixelGrid* input_pixel_grid);
	void setup_parameters(const bool initial_setup);

	double regparam;
	double pixel_fraction, srcgrid_size_scale, pixel_magnification_threshold;

	void calculate_pixel_magnifications(const bool use_emask = false);
	void adaptive_subgrid();
	double get_lowest_mag_sourcept(double &xsrc, double &ysrc);
	void get_highest_mag_sourcept(double &xsrc, double &ysrc);

	bool assign_source_mapping_flags_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, vector<SourcePixel*>& mapped_cartesian_srcpixels, const int& thread);
	void calculate_Lmatrix_overlap(const int &img_index, const int image_pixel_i, const int image_pixel_j, int& Lmatrix_index, lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	double find_lensed_surface_brightness_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);

	bool bisection_search_overlap(lensvector **input_corner_pts, const int& thread);
	bool bisection_search_overlap(lensvector &a, lensvector &b, lensvector &c, const int& thread);
	bool bisection_search_interpolate(lensvector &input_center_pt, const int& thread);
	bool assign_source_mapping_flags_interpolate(lensvector &input_center_pt, vector<SourcePixel*>& mapped_cartesian_srcpixels, const int& thread, const int& image_pixel_i, const int& image_pixel_j);
	void calculate_Lmatrix_interpolate(const int img_index, vector<SourcePixel*>& mapped_cartesian_srcpixels, int& Lmatrix_index, lensvector &input_center_pts, const int& ii, const double weight, const int& thread);
	double find_lensed_surface_brightness_interpolate(lensvector &input_center_pt, const int& thread);
	double find_local_inverse_magnification_interpolate(lensvector &input_center_pt, const int& thread);
	double find_triangle_weighted_invmag(lensvector& pt1, lensvector& pt2, lensvector& pt3, double& total_overlap, const int thread);

	double find_avg_n_images(const double sb_threshold_frac);

	void plot_surface_brightness(string root);
	void output_fits_file(string fits_filename);
	void get_grid_dimensions(double &xmin, double &xmax, double &ymin, double &ymax);
	void store_surface_brightness_grid_data(string root);

	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }
	~SourcePixelGrid();

	//static ofstream bad_interps;

	// for plotting the grid to a file:
	std::ifstream sb_infile;
	std::ofstream xgrid;
	std::ofstream pixel_surface_brightness_file;
	std::ofstream pixel_magnification_file;
	std::ofstream pixel_n_image_file;
};

class DelaunayGrid : private Sort
{
	protected:
	static int nthreads;
	static const int nmax_pts_interp = 240;
	static lensvector **interpolation_pts[nmax_pts_interp];
	static double *interpolation_wgts[nmax_pts_interp];
	static int *interpolation_indx[nmax_pts_interp];
	static int *triangles_in_envelope[nmax_pts_interp];
	static lensvector **polygon_vertices[nmax_pts_interp+2]; // the polygon referred to here is the part of the Voronoi cell contained in the Bower-Watson envelope for each vertex in the envelope.
	static lensvector *new_circumcenter[nmax_pts_interp];

	public:
	static bool zero_outside_border;
	//bool look_for_starting_point;
	int n_gridpts;
	int n_triangles;
	//int img_ni, img_nj;
	lensvector *gridpts;
	Triangle *triangle;
	//double *surface_brightness;	
	//bool *active_pixel;
	//int *active_index;
	//int **img_index_ij;
	int *adj_triangles[4];
	//int *imggrid_ivals;
	//int *imggrid_jvals;
	double avg_area;
	//double srcpixel_xmin, srcpixel_xmax, srcpixel_ymin, srcpixel_ymax;
	double kernel_correlation_length, matern_index;

	protected:
	double** voronoi_boundary_x;
	double** voronoi_boundary_y;
	double *voronoi_area;
	double *voronoi_length;
	int** shared_triangles;
	int* n_shared_triangles;
	// Used for calculating areas and finding whether points are inside a given cell
	//lensvector dt1, dt2, dt3;
	//double prod1, prod2, prod3;

	public:
	DelaunayGrid();
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	void create_pixel_grid(double* gridpts_x, double* gridpts_y, const int n_gridpts);

	int search_grid(int initial_srcpixel, const lensvector& pt, bool& inside_triangle);
	bool test_if_inside(int &tri_number, const lensvector& pt, bool& inside_triangle);
	bool test_if_inside(const int tri_number, const lensvector& pt);
	void record_adjacent_triangles_xy();

	int find_closest_vertex(const int tri_number, const lensvector& pt);
	double sum_edge_sqrlengths(const double min_sb);
	void find_containing_triangle(const lensvector &input_pt, int& trinum, bool& inside_triangle, bool& on_vertex, int& kmin);
	//double interpolate(lensvector &input_pt, const bool interp_mag = false, const int thread = 0);

	void find_interpolation_weights_3pt(const lensvector& input_pt, const int trinum, int& npts, const int thread);
	void find_interpolation_weights_nn(const lensvector &input_pt, const int trinum, int& npts, const int thread); // natural neighbor interpolation
	//void plot_voronoi_grid(string root);

	//void get_grid_points(vector<double>& xvals, vector<double>& yvals, vector<double>& sb_vals);
	//void generate_gmatrices(const bool interpolate);
	//void generate_hmatrices(const bool interpolate);
	void generate_covariance_matrix(double *cov_matrix_packed, const int kernel_type, const double epsilon, double *lumfac = NULL, const bool add_to_covmatrix = false, const double amplitude = -1);
	double modified_bessel_function(const double x, const double nu);
	void beschb(const double x, double& gam1, double& gam2, double& gampl, double& gammi);
	double chebev(const double a, const double b, double* c, const int m, const double x);

	void delete_grid_arrays();
	~DelaunayGrid();
};

class DelaunaySourceGrid : public DelaunayGrid, public ModelParams
{
	friend class QLens;
	friend class ImagePixelGrid;
	QLens *lens;
	ImagePixelGrid *image_pixel_grid;

	public:
	bool look_for_starting_point;
	int img_ni, img_nj;
	double *surface_brightness;	
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

	double regparam;
	double regparam_lsc, regparam_lum_index, distreg_rc;
	double distreg_xcenter, distreg_ycenter, distreg_e1, distreg_e2; // for position-weighted regularization
	double mag_weight_sc, mag_weight_index; // magnification-weighted regularization
	double alpha_clus, beta_clus;

	public:
	DelaunaySourceGrid(QLens* lens_in);
	void copy_pixsrc_data(DelaunaySourceGrid* grid_in);
	void update_meta_parameters(const bool varied_only_fitparams);
	void create_srcpixel_grid(double* srcpts_x, double* srcpts_y, const int n_srcpts, int* ivals_in = NULL, int* jvals_in = NULL, const int ni=0, const int nj=0, const bool find_pixel_magnification = false, const int redshift_indx = -1);

	void setup_parameters(const bool initial_setup);

	void find_pixel_magnifications();

	void assign_surface_brightness_from_analytic_source(const int zsrc_i=-1);
	void fill_surface_brightness_vector();
	void update_surface_brightness(int& index);
	double sum_edge_sqrlengths(const double min_sb);
	double find_lensed_surface_brightness(const lensvector &input_pt, const int img_pixel_i, const int img_pixel_j, const int thread);
	bool find_containing_triangle_with_imgpix(const lensvector &input_pt, const int img_pixel_i, const int img_pixel_j, int& trinum, bool& inside_triangle, bool& on_vertex, int& kmin);
	double interpolate_surface_brightness(const lensvector &input_pt, const bool interp_mag = false, const int thread = 0);
	double interpolate_voronoi_length(const lensvector &input_pt, const int thread = 0);

	bool assign_source_mapping_flags(lensvector &input_pt, vector<PtsWgts>& mapped_delaunay_srcpixels, int& n_mapped_srcpixels, const int img_pixel_i, const int img_pixel_j, const int thread, bool& trouble_with_starting_vertex);
	void record_srcpixel_mappings();
	void calculate_Lmatrix(const int img_index, PtsWgts* mapped_delaunay_srcpixels, int* n_mapped_srcpixels, int& index, const int& ii, const double weight, const int& thread);
	int assign_active_indices_and_count_source_pixels(const bool activate_unmapped_pixels);
	void plot_surface_brightness(string root, const int npix = 600, const bool interpolate = false, const bool plot_magnification = false, const bool plot_fits = false);
	void plot_voronoi_grid(string root);
	double find_moment(const int p, const int q, const int npix, const double xc, const double yc, const double b, const double a, const double phi);
	void find_source_moments(const int npix, double &qs, double &phi_s, double &xavg, double &yavg);

	void get_grid_points(vector<double>& xvals, vector<double>& yvals, vector<double>& sb_vals);
	void generate_gmatrices(const bool interpolate);
	void generate_hmatrices(const bool interpolate);
	void find_source_gradient(const lensvector& input_pt, lensvector& src_grad_neg, const int thread);

	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }

	int get_n_srcpts() { return n_gridpts; }
	void print_pixel_values();

	void delete_lensing_arrays();
	~DelaunaySourceGrid();
};

class LensPixelGrid : public DelaunayGrid, public ModelParams
{
	friend class QLens;
	friend class ImagePixelGrid;
	QLens *lens;
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

	double regparam;
	double regparam_lsc, regparam_lum_index, distreg_rc;
	double distreg_xcenter, distreg_ycenter, distreg_e1, distreg_e2; // for position-weighted regularization
	double mag_weight_sc, mag_weight_index; // magnification-weighted regularization
	double alpha_clus, beta_clus;

	public:
	LensPixelGrid(QLens* lens_in, const int lens_redshift_indx_in);
	void copy_pixlens_data(LensPixelGrid* grid_in);
	void update_meta_parameters(const bool varied_only_fitparams);
	void create_cartesian_pixel_grid(const double x_min, const double x_max, const double y_min, const double y_max, const int redshift_indx = -1);
	void create_delaunay_pixel_grid(double* pts_x, double* pts_y, const int n_pts, const int redshift_indx = -1);

	void setup_parameters(const bool initial_setup);

	void fill_potential_correction_vector();
	void update_potential(int& index);
	void assign_potential_from_analytic_lens(const int zsrc_i, const bool add_potential);
	void assign_potential_from_analytic_lenses();

	double interpolate_potential(const double x, const double y, const int thread = 0);
	void deflection(const double x, const double y, lensvector& def, const int thread);
	void hessian(const double x, const double y, lensmatrix& hess, const int thread);
	double kappa(const double x, const double y, const int thread);
	void kappa_and_potential_derivatives(const double x, const double y, double& kappa, lensvector& def, lensmatrix& hess, const int thread);
	void potential_derivatives(const double x, const double y, lensvector& def, lensmatrix& hess, const int thread);
	bool assign_mapping_flags(lensvector &input_pt, vector<PtsWgts>& mapped_potpixels_ij, int& n_mapped_potpixels, const int img_pixel_i, const int img_pixel_j, const int thread);
	void calculate_Lmatrix(const int img_index, PtsWgts* mapped_potpixels, int* n_mapped_potpixels, lensvector& S0_derivs, int& index, const int& subpixel_indx, const int offset, const double weight, const int& thread);
	double first_order_surface_brightness_correction(const lensvector &input_pt, const lensvector& S0_deriv, const int thread);

	void find_interpolation_weights_cartesian(const lensvector &input_pt, int& npts, const int deriv_type, const int thread);

	bool assign_lens_mapping_flags(lensvector &input_pt, vector<PtsWgts>& mapped_delaunay_lenspixels, int& n_mapped_lenspixels, const int img_pixel_i, const int img_pixel_j, const int thread, bool& trouble_with_starting_vertex);
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

class ImagePixelGrid : private Sort
{
	friend class QLens;
	friend class SourcePixel;
	friend class SourcePixelGrid;
	friend class DelaunaySourceGrid;
	friend class ImagePixelData;
	friend class LensProfile;
	QLens *lens;
	SourcePixelGrid *cartesian_srcgrid;
	DelaunaySourceGrid *delaunay_srcgrid;
	LensPixelGrid *lensgrid;
	lensvector **corner_pts;
	lensvector **corner_sourcepts;
	lensvector **center_pts;
	lensvector **center_sourcepts;
	lensvector ***subpixel_center_pts;
	lensvector ***subpixel_center_sourcepts;
	double ***subpixel_surface_brightness;
	lensvector ***subpixel_source_gradient;
	//double **S0_check;
	//lensvector **x0_check;
	double ***subpixel_weights;
	int **subpixel_index;

	double **surface_brightness;
	double **foreground_surface_brightness;
	double **noise_map;
	double **source_plane_triangle1_area; // area of triangle 1 (connecting points 0,1,2) when mapped to the source plane
	double **source_plane_triangle2_area; // area of triangle 2 (connecting points 1,3,2) when mapped to the source plane
	double **pixel_mag; // ratio of sum of source plane triangle areas over the image pixel area
	bool **pixel_in_mask;
	bool **mask;
	bool **emask;
	double pixel_area, triangle_area; // half of pixel area
	double min_srcplane_area;
	double max_sb;
	bool **maps_to_source_pixel;
	int max_nsplit;
	int **nsplits;
	bool ***subpixel_maps_to_srcpixel;
	int **pixel_index;
	int **pixel_index_fgmask;
	int Lmatrix_src_npixels;
	int Lmatrix_pot_npixels, Lmatrix_src_and_pot_npixels;

	int *active_image_pixel_i;
	int *active_image_pixel_j;
	int *active_image_pixel_i_ss;
	int *active_image_pixel_j_ss;
	int *active_image_subpixel_ii;
	int *active_image_subpixel_jj;
	int *active_image_subpixel_ss;
	int *image_pixel_i_from_subcell_ii;
	int *image_pixel_j_from_subcell_jj;
	int *active_image_pixel_i_fgmask;
	int *active_image_pixel_j_fgmask;

	double *psf_zvec; // for convolutions using FFT
	double *psf_zvec_fgmask; // for convolutions using FFT
	int fft_imin, fft_jmin, fft_ni, fft_nj;
	int fft_imin_fgmask, fft_jmin_fgmask, fft_ni_fgmask, fft_nj_fgmask;
#ifdef USE_FFTW
	std::complex<double> *psf_transform;
	std::complex<double> *psf_transform_fgmask;
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
#endif
	bool fft_convolution_is_setup;
	bool fg_fft_convolution_is_setup;

	int **twist_status;
	lensvector **twist_pts;
	double *defx_corners, *defy_corners, *defx_centers, *defy_centers, *area_tri1, *area_tri2;
	double *defx_subpixel_centers, *defy_subpixel_centers;
	double *twistx, *twisty;
	int *twiststat;
	int *masked_pixels_i, *masked_pixels_j, *emask_pixels_i, *emask_pixels_j, *masked_pixel_corner_i, *masked_pixel_corner_j, *masked_pixel_corner, *masked_pixel_corner_up;
	int *extended_mask_subcell_i, *extended_mask_subcell_j, *extended_mask_subcell_index;
	int *mask_subcell_i, *mask_subcell_j, *mask_subcell_index;
	int **ncvals;

	bool include_potential_perturbations;

	long int ntot_corners, ntot_cells, ntot_cells_emask;
	long int ntot_subpixels, ntot_subpixels_in_mask;

	vector<SourcePixel*> **mapped_cartesian_srcpixels; // since the Cartesian grid uses recursion (if adaptive_subgrid is on), a pointer to each mapped source pixel is needed
	vector<PtsWgts> **mapped_delaunay_srcpixels; // for the Delaunay grid, it only needs to record the index of each mapped source pixel (no pointer needed)
	vector<PtsWgts> **mapped_potpixels; // for the Delaunay grid, it only needs to record the index of each mapped pixel (no pointer needed)
	int ***n_mapped_srcpixels; // will store how many source pixels map to a given (sub)pixel for Lmatrix
	int ***n_mapped_potpixels; // will store how many potential perturbation pixels map to a given (sub)pixel for Lmatrix
	RayTracingMethod ray_tracing_method;
	SourceFitMode source_fit_mode;
	double xmin, xmax, ymin, ymax;
	double src_xmin, src_xmax, src_ymin, src_ymax; // for ray-traced points
	int x_N, y_N; // gives the number of cells in the x- and y- directions (so the number of corner points in each direction is x_N+1, y_N+1)
	int n_active_pixels;
	int n_high_sn_pixels;
	int xy_N; // gives x_N*y_N if the entire pixel grid is used
	double pixel_xlength, pixel_ylength;
	inline bool test_if_between(const double& p, const double& a, const double& b);
	int src_redshift_index; // each ImagePixelGrid object is associated with a specific redshift
	double* imggrid_zfactors;
	double** imggrid_betafactors; // kappa ratio used for modeling source points at different redshifts

	static int nthreads;
	static const int max_iterations, max_step_length;
	static bool *newton_check;
	static lensvector *fvec;
	static double image_pos_accuracy;

	bool run_newton(lensvector& xroot, double& mag, const int& thread);
	bool LineSearch(lensvector& xold, double fold, lensvector& g, lensvector& p, lensvector& x, double& f, double stpmax, bool &check, const int& thread);
	bool NewtonsMethod(lensvector& x, bool &check, const int& thread);
	void SolveLinearEqs(lensmatrix&, lensvector&);
	bool redundancy(const lensvector&, double &);
	double max_component(const lensvector&);

	public:
	ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, double xmin_in, double xmax_in, double ymin_in, double ymax_in, int x_N_in, int y_N_in, const bool raytrace = false, int src_redshift_index = -1);
	//ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, double** sb_in, const int x_N_in, const int y_N_in, const int reduce_factor, double xmin_in, double xmax_in, double ymin_in, double ymax_in, const int src_redshift_index = -1);
	ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, ImagePixelData& pixel_data, const bool include_extended_mask = false, const int src_redshift_index = -1, const int mask_index = 0, const bool setup_mask_and_data = true, const bool verbal = false);
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	void update_zfactors_and_beta_factors();

	//ImagePixelGrid(QLens* lens_in, double* zfactor_in, double** betafactor_in, SourceFitMode mode, RayTracingMethod method, ImagePixelData& pixel_data);
	void load_data(ImagePixelData& pixel_data);
	void generate_point_images(const vector<image>& imgs, double *ptimage_surface_brightness, const bool use_img_fluxes, const double srcflux, const int img_num = -1); // -1 means use all images
	void add_point_images(double *ptimage_surface_brightness, const int npix);
	void generate_and_add_point_images(const vector<image>& imgs, const bool include_imgfluxes, const double srcflux);
	void find_point_images(const double src_x, const double src_y, vector<image>& imgs, const bool use_overlap, const bool is_lensed, const bool verbal);
	//bool test_if_inside_cell(const lensvector& point, const int& i, const int& j);
	bool set_fit_window(ImagePixelData& pixel_data, const bool raytrace = false, const int mask_k = 0, const bool redo_fft = true);
	void include_all_pixels(const bool redo_fft = true);
	void activate_extended_mask(const bool redo_fft = true);
	void activate_foreground_mask(const bool redo_fft = true);
	void deactivate_extended_mask(const bool redo_fft = true);
	void update_mask_values();

	void setup_pixel_arrays();
	void set_null_ray_tracing_arrays();
	void set_null_subpixel_ray_tracing_arrays();
	void setup_ray_tracing_arrays(const bool include_fft_arrays = true, const bool verbal = false);
	void setup_subpixel_ray_tracing_arrays(const bool verbal = false);
	void delete_ray_tracing_arrays(const bool delete_fft_arrays = true);
	void delete_subpixel_ray_tracing_arrays();
	void update_grid_dimensions(const double xmin, const double xmax, const double ymin, const double ymax);
	void calculate_sourcepts_and_areas(const bool raytrace_pixel_centers = false, const bool verbal = false);
	bool calculate_subpixel_source_gradient();
	void ray_trace_pixels();
	void set_nsplits_from_lens_settings();
	void set_nsplits(const int default_nsplit, const int emask_nsplit, const bool split_pixels);
	void setup_noise_map(QLens* lens_in);
	bool setup_FFT_convolution(const bool supersampling, const bool foreground, const bool verbal);
	void cleanup_FFT_convolution_arrays();
	void cleanup_foreground_FFT_convolution_arrays();

	~ImagePixelGrid();
	void redo_lensing_calculations(const bool verbal = false);
	void redo_lensing_calculations_corners();
	void assign_mask_pixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& count, ImagePixelData* data_in);

	void find_optimal_sourcegrid(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double &sourcegrid_limit_xmin, const double &sourcegrid_limit_xmax, const double &sourcegrid_limit_ymin, const double& sourcegrid_limit_ymax);
	void set_sourcegrid_params_from_ray_tracing(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double sourcegrid_limit_xmin, const double sourcegrid_limit_xmax, const double sourcegrid_limit_ymin, const double sourcegrid_limit_ymax);


	double find_approx_source_size(double& xcavg, double& ycavg, const bool verbal = false);
	void find_optimal_shapelet_scale(double& scale, double& xcenter, double& ycenter, double& recommended_nsplit, const bool verbal, double& sig, double& scaled_maxdist);
	void fill_surface_brightness_vector();
	void plot_grid(string filename, bool show_inactive_pixels);
	void set_lens(QLens* lensptr) { lens = lensptr; }
	void set_cartesian_srcgrid(SourcePixelGrid* source_pixel_ptr) { cartesian_srcgrid = source_pixel_ptr; }
	void set_delaunay_srcgrid(DelaunaySourceGrid* delaunayptr) { delaunay_srcgrid = delaunayptr; }
	void set_lensgrid(LensPixelGrid* gridptr) { lensgrid = gridptr; }
	void find_optimal_sourcegrid_npixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels);
	void find_optimal_firstlevel_sourcegrid_npixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels);
	void find_surface_brightness(const bool foreground_only = false, const bool lensed_sources_only = false, const bool include_first_order_corrections = false, const bool show_only_first_order_corrections = false, const bool omit_lensed_nonshapelet_sources = false);
	void set_zero_lensed_surface_brightness();
	void set_zero_foreground_surface_brightness();

	double plot_surface_brightness(string outfile_root, bool plot_residual = false, bool normalize_sb = false, bool show_noise_thresh = false, bool plot_log = false);
	void plot_sourcepts(string outfile_root, const bool show_subpixels = false);
	void output_fits_file(string fits_filename, bool plot_residual = false);

	void add_pixel_noise();
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
	}
	double calculate_signal_to_noise(double &total_signal);
	void assign_image_mapping_flags(const bool delaunay, const bool potential_perturbations = false);
	int count_nonzero_source_pixel_mappings_cartesian();
	int count_nonzero_source_pixel_mappings_delaunay();
	int count_nonzero_lensgrid_pixel_mappings();
	void get_grid_params(double& xmin_in, double& xmax_in, double& ymin_in, double& ymax_in, int& npx, int& npy) { xmin_in = xmin; xmax_in = xmax; ymin_in = ymin; ymax_in = ymax; npx = x_N; npy = y_N; }
};

class SB_Profile;

struct ImagePixelData : private Sort
{
	friend class QLens;
	QLens *lens;
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
	double pixel_size;
	double emask_rmax; // used only when splining Fourier mode integrals for non-elliptical structure
	string data_fits_filename;
	string noise_map_fits_filename;
	std::ostream* isophote_fit_out;
	ImagePixelData()
	{
		npixels_x = 0;
		npixels_y = 0;
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
		lens = NULL;
		isophote_fit_out = &std::cout;
		noise_map_fits_filename = "";
	}
	~ImagePixelData();
	void load_data(string root);
	void load_from_image_grid(ImagePixelGrid* image_pixel_grid);
	bool load_data_fits(const double xmin_in, const double xmax_in, const double ymin_in, const double ymax_in, string fits_filename, const int hdu_indx = 1, const bool show_header = false) {
		xmin=xmin_in; xmax=xmax_in; ymin=ymin_in; ymax=ymax_in;
		return load_data_fits(false,fits_filename,hdu_indx,show_header);
	}
	bool load_data_fits(const double pixel_size_in, string fits_filename, const int hdu_indx = 1, const bool show_header = false) {
		pixel_size = pixel_size_in;
		return load_data_fits(true,fits_filename, hdu_indx, show_header);
	}
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
	bool load_data_fits(bool use_pixel_size, string fits_filename, const int hdu_indx, const bool show_header = false);
	void save_data_fits(string fits_filename, const bool subimage=false, const double xmin_in=-1e30, const double xmax_in=1e30, const double ymin_in=-1e30, const double ymax_in=1e30);
	bool load_mask_fits(const int mask_k, string fits_filename, const bool foreground=false, const bool emask=false, const bool add_mask=false);
	bool save_mask_fits(string fits_filename, const bool foreground=false, const bool emask=false, const int mask_k=0, const bool subimage=false, const double xmin_in=-1e30, const double xmax_in=1e30, const double ymin_in=-1e30, const double ymax_in=1e30);
	bool copy_mask(ImagePixelData* data, const int mask_k = 0);
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
	bool set_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1, double theta2, const double xstretch, const double ystretch, const bool unset = false, const int mask_k = 0);
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
	void set_lens(QLens* lensptr) {
		lens = lensptr;
		pixel_size = lens->data_pixel_size;
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
	void plot_surface_brightness(string outfile_root, bool show_only_mask, bool show_extended_mask = false, bool show_foreground_mask = false, const int mask_k = 0);
};

#endif // PIXELGRID_H
