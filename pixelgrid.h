#ifndef PIXELGRID_H
#define PIXELGRID_H

#include "qlens.h"
#include "rand.h"
#include "lensvec.h"
#include "trirectangle.h"
#include <vector>
#include <iostream>
using namespace std;

class ImagePixelGrid;
class SourcePixelGrid;
struct ImagePixelData;

struct InterpolationCells {
	bool found_containing_cell;
	SourcePixelGrid *pixel[3];
};

class SourcePixelGrid
{
	friend class QLens;
	friend class ImagePixelGrid;
	// this constructor is only used by the top-level SourcePixelGrid, so it's private
	SourcePixelGrid(QLens* lens_in, lensvector** xij, const int& i, const int& j, const int& level_in, SourcePixelGrid* parent_ptr);

	SourcePixelGrid ***cell;
	QLens *lens;
	static ImagePixelGrid *image_pixel_grid;
	static TriRectangleOverlap *trirec;
	static int nthreads;
	SourcePixelGrid *neighbor[4]; // 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	SourcePixelGrid *parent_cell;
	int ii, jj; // this is the index assigned to this cell in the grid of the parent cell

	static double* srcgrid_zfactors; // kappa ratio used for modeling source points at different redshifts
	static double** srcgrid_betafactors; // kappa ratio used for modeling source points at different redshifts
	static double xcenter, ycenter;
	static double srcgrid_xmin, srcgrid_xmax, srcgrid_ymin, srcgrid_ymax;
	int u_N, w_N;
	int level;
	static int *imin, *imax, *jmin, *jmax; // defines "window" within which we will check all the cells for overlap
	static int number_of_pixels;
	lensvector center_pt;
	double cell_area;
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
	static InterpolationCells *nearest_interpolation_cells;
	static lensvector **interpolation_pts[3];
	static int *n_interpolation_pts;
	static bool regrid;
	static bool regrid_if_unmapped_source_subcells;
	static bool activate_unmapped_source_pixels;
	static bool exclude_source_pixels_outside_fit_window;

	// Used for calculating areas and finding whether points are inside a given cell
	//static lensvector d1, d2, d3, d4;
	//static double product1, product2, product3;
	static int *maxlevs;
	static lensvector ***xvals_threads;
	static lensvector ***corners_threads;
	static lensvector **twistpts_threads;
	static int **twist_status_threads;

	static int u_split_initial, w_split_initial;
	static const int max_levels;

	static int levels; // keeps track of the total number of grid cell levels
	static int splitlevels; // specifies the number of initial splittings to perform (not counting extra splittings if critical curves present)
	static double min_cell_area;

	void split_cells(const int usplit, const int wsplit, const int& thread);
	void unsplit();
	void split_subcells(const int splitlevel, const int thread);
	void split_subcells_firstlevel(const int splitlevel);

	inline void find_cell_area();
	void assign_firstlevel_neighbors(void);
	void assign_neighborhood();
	void assign_all_neighbors(void);
	void assign_level_neighbors(int neighbor_level);
	void test_neighbors();
	int assign_indices_and_count_levels();
	void assign_indices(int& source_pixel_i);
	int assign_active_indices_and_count_source_pixels(bool regrid_if_inactive_cells, bool activate_unmapped_pixels, bool exclude_pixels_outside_window);
	void assign_active_indices(int& source_pixel_i);

	void print_indices();

	public:
	SourcePixelGrid(QLens* lens_in, double x_min, double x_max, double y_min, double y_max);
	SourcePixelGrid(QLens* lens_in, SourcePixelGrid* input_pixel_grid);
	SourcePixelGrid(QLens* lens_in, string pixel_data_fileroot, const double& minarea_in);
	static void set_splitting(int rs0, int ts0, double min_cs);
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	void copy_source_pixel_grid(SourcePixelGrid* input_pixel_grid);
	inline bool check_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	inline bool check_if_in_neighborhood(lensvector **input_corner_pts, bool &inside, const int& thread);
	inline double find_rectangle_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread, const int&, const int&);
	inline bool check_triangle1_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	inline bool check_triangle2_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	inline double find_triangle1_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	inline double find_triangle2_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	void generate_gmatrices();
	void generate_hmatrices();

	bool bisection_search_overlap(lensvector **input_corner_pts, const int& thread);
	void calculate_pixel_magnifications();
	void adaptive_subgrid();
	double get_lowest_mag_sourcept(double &xsrc, double &ysrc);
	void get_highest_mag_sourcept(double &xsrc, double &ysrc);

	bool assign_source_mapping_flags_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, vector<SourcePixelGrid*>& mapped_source_pixels, const int& thread);
	void subcell_assign_source_mapping_flags_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, vector<SourcePixelGrid*>& mapped_source_pixels, const int& thread, bool& image_pixel_maps_to_source_grid);
	void calculate_Lmatrix_overlap(const int &img_index, const int &image_pixel_i, const int &image_pixel_j, int& Lmatrix_index, lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	double find_lensed_surface_brightness_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread);
	void find_lensed_surface_brightness_subcell_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread, double& overlap, double& total_overlap, double& total_weighted_surface_brightness);

	bool bisection_search_interpolate(lensvector &input_center_pt, const int& thread);
	bool assign_source_mapping_flags_interpolate(lensvector &input_center_pt, vector<SourcePixelGrid*>& mapped_source_pixels, const int& thread, const int& image_pixel_i, const int& image_pixel_j);
	bool subcell_assign_source_mapping_flags_interpolate(lensvector &input_center_pt, vector<SourcePixelGrid*>& mapped_source_pixels, const int& thread);
	void calculate_Lmatrix_interpolate(const int img_index, const int image_pixel_i, const int image_pixel_j, int& Lmatrix_index, lensvector &input_center_pts, const int& ii, const double weight, const int& thread);
	double find_lensed_surface_brightness_interpolate(lensvector &input_center_pt, const int& thread);
	void find_interpolation_cells(lensvector &input_center_pt, const int& thread);
	SourcePixelGrid* find_nearest_neighbor_cell(lensvector &input_center_pt, const int& side);
	SourcePixelGrid* find_nearest_neighbor_cell(lensvector &input_center_pt, const int& side, const int tiebreaker_side);
	void find_nearest_two_cells(SourcePixelGrid* &cellptr1, SourcePixelGrid* &cellptr2, const int& side);
	SourcePixelGrid* find_corner_cell(const int i, const int j);

	void assign_surface_brightness();
	void update_surface_brightness(int& index);
	void fill_surface_brightness_vector();
	void fill_surface_brightness_vector_recursive(int& column_j);
	void fill_n_image_vector();
	void find_avg_n_images();

	void fill_n_image_vector_recursive(int& column_j);
	void plot_surface_brightness(string root);
	void plot_cell_surface_brightness(int line_number, int pixels_per_cell_x, int pixels_per_cell_y);
	void store_surface_brightness_grid_data(string root);
	void write_surface_brightness_to_file();
	void read_surface_brightness_data();

	void clear(void);
	void clear_subgrids();
	void set_image_pixel_grid(ImagePixelGrid* image_pixel_ptr) { image_pixel_grid = image_pixel_ptr; }
	~SourcePixelGrid();

	//static ofstream bad_interps;

	// for plotting the grid to a file:
	static ofstream index_out;
	static ifstream sb_infile;
	static ofstream xgrid;
	static ofstream pixel_surface_brightness_file;
	static ofstream pixel_magnification_file;
	static ofstream pixel_n_image_file;
	static ofstream missed_cells_out;
	void plot_corner_coordinates(void);
};

class ImagePixelGrid : public Sort
{
	// the image pixel grid is simpler because its cells will never be split. So there is no recursion in this grid
	friend class QLens;
	friend class SourcePixelGrid;
	friend class ImagePixelData;
	QLens *lens;
	SourcePixelGrid *source_pixel_grid;
	lensvector **corner_pts;
	lensvector **corner_sourcepts;
	lensvector **center_pts;
	lensvector **center_sourcepts;
	double **surface_brightness;
	double **source_plane_triangle1_area; // area of triangle 1 (connecting points 0,1,2) when mapped to the source plane
	double **source_plane_triangle2_area; // area of triangle 2 (connecting points 1,3,2) when mapped to the source plane
	bool **fit_to_data;
	double pixel_area, triangle_area; // half of pixel area
	double max_sb, pixel_noise;
	bool **maps_to_source_pixel;
	int max_nsplit;
	int **nsplits;
	bool ***subpixel_maps_to_srcpixel;
	int **pixel_index;
	int **twist_status;
	lensvector **twist_pts;
	double *defx_corners, *defy_corners, *defx_centers, *defy_centers, *area_tri1, *area_tri2;
	double *twistx, *twisty;
	int *twiststat;
	int *extended_mask_i, *extended_mask_j, *extended_mask_corner_i, *extended_mask_corner_j, *extended_mask_corner, *extended_mask_corner_up;
	int **nvals, **ncvals;

	long int ntot_corners, ntot_cells;

	vector<SourcePixelGrid*> **mapped_source_pixels;
	RayTracingMethod ray_tracing_method;
	SourceFitMode source_fit_mode;
	double xmin, xmax, ymin, ymax;
	int x_N, y_N; // gives the number of cells in the x- and y- directions (so the number of corner points in each direction is x_N+1, y_N+1)
	int n_active_pixels;
	int n_high_sn_pixels;
	int xy_N; // gives x_N*y_N if the entire pixel grid is used
	double pixel_xlength, pixel_ylength;
	inline bool test_if_between(const double& p, const double& a, const double& b);
	static double* imggrid_zfactors;
	static double** imggrid_betafactors; // kappa ratio used for modeling source points at different redshifts

	public:
	ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, double xmin_in, double xmax_in, double ymin_in, double ymax_in, int x_N_in, int y_N_in);
	ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, ImagePixelData& pixel_data);
	ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, double** sb_in, const int x_N_in, const int y_N_in, const int reduce_factor, double xmin_in, double xmax_in, double ymin_in, double ymax_in);

	ImagePixelGrid(QLens* lens_in, double* zfactor_in, double** betafactor_in, SourceFitMode mode, RayTracingMethod method, ImagePixelData& pixel_data);
	void load_data(ImagePixelData& pixel_data);
	bool test_if_inside_cell(const lensvector& point, const int& i, const int& j);
	void set_fit_window(ImagePixelData& pixel_data);
	void include_all_pixels();
	void activate_extended_mask();
	void deactivate_extended_mask();
	void setup_ray_tracing_arrays();
	void delete_ray_tracing_arrays();
	void reset_nsplit();

	~ImagePixelGrid();
	void redo_lensing_calculations();
	void redo_lensing_calculations_corners();
	void assign_required_data_pixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& count, ImagePixelData* data_in);

	void find_optimal_sourcegrid(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double &sourcegrid_limit_xmin, const double &sourcegrid_limit_xmax, const double &sourcegrid_limit_ymin, const double& sourcegrid_limit_ymax);
	void find_optimal_shapelet_scale(double& scale, double& xcenter, double& ycenter, double& recommended_nsplit, const bool verbal);
	void fill_surface_brightness_vector();
	void plot_grid(string filename, bool show_inactive_pixels);
	void set_lens(QLens* lensptr) { lens = lensptr; }
	void set_source_pixel_grid(SourcePixelGrid* source_pixel_ptr) { source_pixel_grid = source_pixel_ptr; }
	void find_optimal_sourcegrid_npixels(double pixel_fraction, double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels);
	void find_optimal_firstlevel_sourcegrid_npixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels);
	void find_surface_brightness(bool plot_foreground_only = false);
	void plot_surface_brightness(string outfile_root, bool plot_residual = false, bool show_only_mask = true, bool show_extended_mask = false, bool show_noise_thresh = false);
	void output_fits_file(string fits_filename, bool plot_residual = false);

	void add_pixel_noise(const double& pixel_noise_sig);
	void set_pixel_noise(const double& pn) { pixel_noise = pn; }
	double calculate_signal_to_noise(const double& pixel_noise_sig, double &total_signal);
	void assign_image_mapping_flags();
	int count_nonzero_source_pixel_mappings();
};

struct ImagePixelData
{
	friend class QLens;
	QLens *lens;
	int npixels_x, npixels_y;
	int n_required_pixels;
	double **surface_brightness;
	bool **high_sn_pixel; // used to help determine optimal source pixel size based on area the high S/N pixels cover when mapped to source plane
	bool **require_fit;
	bool **extended_mask;
	double *xvals, *yvals;
	int n_high_sn_pixels;
	double xmin, xmax, ymin, ymax;
	double pixel_size;
	double global_max_sb;
	ImagePixelData()
	{
		surface_brightness = NULL;
		high_sn_pixel = NULL;
		require_fit = NULL;
		extended_mask = NULL;
		xvals = NULL;
		yvals = NULL;
		lens = NULL;
	}
	~ImagePixelData();
	void load_data(string root);
	void load_from_image_grid();
	bool load_data_fits(const double xmin_in, const double xmax_in, const double ymin_in, const double ymax_in, string fits_filename) {
		xmin=xmin_in; xmax=xmax_in; ymin=ymin_in; ymax=ymax_in;
		return load_data_fits(false,fits_filename);
	}
	bool load_data_fits(const double pixel_size_in, string fits_filename) {
		pixel_size = pixel_size_in;
		return load_data_fits(true,fits_filename);
	}
	bool load_data_fits(bool use_pixel_size, string fits_filename);
	bool load_mask_fits(string fits_filename);
	bool save_mask_fits(string fits_filename);
	void set_no_required_data_pixels();
	void assign_high_sn_pixels();
	void set_all_required_data_pixels();
	bool inside_mask(const double x, const double y);
	void assign_mask_windows(const double sb_noise_threshold);
	void unset_low_signal_pixels(const double sb_threshold, const bool use_fit);
	void set_nearest_neighbor_pixels();
	void set_required_data_window(const double xmin, const double xmax, const double ymin, const double ymax, const bool unset = false);
	void set_required_data_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1, double theta2, const double xstretch, const double ystretch, const bool unset = false);
	void set_extended_mask(const int n_neighbors);
	long int get_size_of_extended_mask();
	bool test_if_in_fit_region(const double& x, const double& y);
	void set_lens(QLens* lensptr) { lens = lensptr; }

	void estimate_pixel_noise(const double xmin, const double xmax, const double ymin, const double ymax, double &noise, double &mean_sb);
	void add_point_image_from_centroid(ImageData* point_image_data, const double xmin_in, const double xmax_in, const double ymin_in, const double ymax_in, const double sb_threshold, const double pixel_error);
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
	void plot_surface_brightness(string outfile_root, bool show_only_mask, bool show_extended_mask = false);
};



#endif // PIXELGRID_H
