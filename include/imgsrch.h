#ifndef IMGSRCH_H
#define IMGSRCH_H

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

enum ImageSystemType { NoImages, Single, Double, Cusp, Quad };
enum inside_cell { Inside, Outside, Edge };
enum edge_sourcept_status { SourceInGap, SourceInOverlap, NoSource };

class QLens; 
class ImgSrchGrid;

template <typename QScalar>
struct image {
	lensvector<QScalar> pos;
	QScalar mag, flux, td;
	int parity;
	//image() {}
	//image(image& img_in) { pos=img_in.pos; mag=img_in.mag; flux=img_in.flux; td=img_in.td; parity=img_in.parity; }
	//void copy_img(image& img_in) { pos=img_in.pos; mag=img_in.mag; flux=img_in.flux; td=img_in.td; parity=img_in.parity; }
};

struct image_data : public image<double>
{
	double sigma_pos;
	double sigma_flux;
	double sigma_td;
	bool use_in_chisq;
	//image_data() {}
	//image_data(image_data& img_in) { pos=img_in.pos; mag=img_in.mag; flux=img_in.flux; td=img_in.td; parity=img_in.parity; sigma_pos=img_in.sigma_pos; sigma_flux=img_in.sigma_flux; sigma_td=img_in.sigma_td; use_in_chisq=img_in.use_in_chisq; }
	//image_data(image& img_in) { pos=img_in.pos; mag=img_in.mag; flux=img_in.flux; td=img_in.td; parity=img_in.parity; }
};

template <typename QScalar>
class CellStaticParams
{
	friend class GridCell;
	friend class ImgSrchGrid;
	static lensvector<QScalar> *d1, *d2, *d3, *d4;
	static QScalar *product1, *product2, *product3;
	static lensvector<QScalar> *fvec;
	static lensvector<QScalar> ***xvals_threads;
};

class GridCell : private Brent
{
	friend class ImgSrchGrid;

	protected:
	// this constructor is only used by ImgSrchGrid to initialize the lower-level grids, so it's protected
	GridCell(QLens* lens_in, lensvector<double>** xij, const int& i, const int& j, const int& level_in, ImgSrchGrid* parent_grid);

	GridCell*** cell;
	QLens* lens;
	static int nthreads;
	GridCell* neighbor[4]; // 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	ImgSrchGrid* parent_grid; // this points to the top-level grid

	// currently, we don't need any of these to be autodiff variables; autodiff var's are only needed in the top level grid ImgSrchGrid
	lensvector<double> center_imgplane;
	lensvector<double> corner_pt[4];
	// cell lensing properties
	lensvector<double> *corner_sourcept[4];
	double *corner_invmag[4];
	double *corner_kappa[4];

	private: 
	static const int u_split, w_split;
	static bool enforce_min_area;
	static bool cc_neighbor_splittings;
	static int splitlevels; // specifies the number of initial splittings to perform (not counting extra splittings if critical curves present)
	static int cc_splitlevels; // specifies the additional splittings to perform if critical curves are present
	static double min_cell_area;
	static int u_split_initial, w_split_initial;
	static int *maxlevs;

	int u_N, w_N;
	int level;

	bool allocated_corner[4];
	double cell_area;

	// all functions in class GridCell are contained in imgsrch.cpp
	bool image_test(const int& thread);

	template <typename QScalar>
	bool run_newton(lensvector<QScalar>& xroot, const int& thread);
	inside_cell test_if_inside_sourceplane_cell(lensvector<double>* point, const int& thread);
	bool test_if_sourcept_inside_triangle(lensvector<double>* point1, lensvector<double>* point2, lensvector<double>* point3, const int& thread);
	bool test_if_inside_cell(const lensvector<double>& point, const int& thread);
	bool test_if_galaxy_nearby(const lensvector<double>& point, const double& distsq);

	void assign_lensing_properties(const int& thread);
	void assign_subcell_lensing_properties(const int& thread);

	bool cc_inside;
	bool singular_pt_inside;
	bool cell_in_central_image_region;
	void check_if_cc_inside();
	void check_if_singular_point_inside(const int& thread);
	void check_if_central_image_region();

	double invmag_along_diagonal(const double t);
	int galsubgrid_cc_splitlevels;

	void clear_subcells(int clear_level);
	void split_subcells(int cc_splitlevels, bool cc_neighbor_splitting, const int& thread);
	void assign_neighbors_lensing_subcells(int cc_splitlevel, const int& thread);
	bool split_cells(const int& thread);
	template <typename QScalar>
	void grid_search(const int& searchlevel, const int& thread);
	edge_sourcept_status check_subgrid_neighbor_boundaries(const int& neighbor_direction, GridCell* neighbor_subcell, lensvector<double>& centerpt, const int& thread);
	void set_grid_xvals(lensvector<double>** xv, const int& i, const int& j);
	void find_cell_area(const int& thread);
	void assign_neighborhood();
	void assign_all_neighbors();
	void assign_level_neighbors(int neighbor_level);

	template <typename QScalar>
	bool LineSearch(lensvector<QScalar>& xold, QScalar fold, lensvector<QScalar>& g, lensvector<QScalar>& p, lensvector<QScalar>& x, QScalar& f, QScalar stpmax, bool &check, const int& thread);
	template <typename QScalar>
	bool NewtonsMethod(lensvector<QScalar>& x, bool &check, const int& thread);
	template <typename QScalar>
	void SolveLinearEqs(lensmatrix<QScalar>&, lensvector<QScalar>&);
	template <typename QScalar>
	bool redundancy(const lensvector<QScalar>&, QScalar &);
	template <typename QScalar>
	QScalar max_component(const lensvector<QScalar>& x) {
		QScalar a = fabs(x[0]);
		QScalar b = fabs(x[1]);
		return (a > b ? a : b);
	}

	static const int max_iterations, max_step_length;
	static bool *newton_check;

public:
	GridCell() { parent_grid = NULL; }
	GridCell(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double* zfactor_in, double** betafactor_in); 
	GridCell(double xcenter_in, double ycenter_in, double xlength, double ylength, double* zfactor_in, double** betafactor_in);
	void reassign_coordinates(lensvector<double>** xij, const int& i, const int& j, const int& level_in);

	static void set_splitting(int rs0, int ts0, int sl, int ccsl, double max_cs, bool neighbor_split);
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	~GridCell();

	//void set_lens(QLens* lensptr) { lens = lensptr; }
	void subgrid_around_galaxies_iteration(lensvector<double>* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_split, bool cc_neighbor_splitting, bool *subgrid);

	void galsubgrid();
	void store_critical_curve_pts();
	void find_and_store_critical_curve_pt(const int icorner, const int fcorner, int &added_pts);
	static void set_enforce_min_area(const bool& setting) { enforce_min_area = setting; }

	// for plotting the grid to a file:
	//static std::ofstream xgrid;
	//void plot_corner_coordinates();
	void output_corner_coordinates(std::vector<double>& pts_x, std::vector<double>& pts_y, std::vector<double>& srcpts_x, std::vector<double>& srcpts_y);
};

template <typename QScalar>
class GridParams 
{
	public:
	lensvector<QScalar> sourcept;
	image<QScalar> *images;
};

class ImgSrchGrid : public GridCell
{
	friend class GridCell;

	public:
	GridParams<double> gridparams;
#ifdef USE_STAN
	GridParams<stan::math::var> gridparams_dif; // autodiff version
#endif
	template <typename QScalar>
	GridParams<QScalar>& assign_gridparam_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return gridparams_dif;
		else
#endif
		return gridparams;
	}

	public:
	bool radial_grid; // if false, a Cartesian grid is assumed
	double* grid_zfactors; // kappa ratio used for modeling source points at different redshifts
	double** grid_betafactors; // kappa ratio used for modeling source points at different redshifts
	double rmin, rmax;
	double xcenter, ycenter;
	double grid_q;
	double theta_offset;

	// all functions in class ImgSrchGrid are contained in imgsrch.cpp

	void assign_subcell_lensing_properties_firstlevel();
	void reassign_subcell_lensing_properties_firstlevel();

	// Used for finding critical curves within a grid cell
	lensvector<double> ccsearch_initial_pt, ccsearch_interval;

	double ccroot_t;
	lensvector<double> ccroot;
	double cclength1, cclength2, long_diagonal_length;
	double invmag_along_diagonal(const double t);

	int max_level, max_images;

	int levels; // keeps track of the total number of grid cell levels

	void split_subcells_firstlevel(int cc_splitlevels, bool cc_neighbor_splitting);
	template <typename QScalar>
	void grid_search_firstlevel(const int& searchlevel);
	void assign_firstlevel_neighbors();

	// make these multithread-safe if you decide to multithread the image searching
	bool finished_search;
	int nfound_max, nfound_pos, nfound_neg;

public:
	ImgSrchGrid(QLens* lens_in, double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double acc, double* zfactor_in, double** betafactor_in); 
	ImgSrchGrid(QLens* lens_in, double xcenter_in, double ycenter_in, double xlength, double ylength, double acc, double* zfactor_in, double** betafactor_in);
	void redraw_grid(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double* zfactor_in, double** betafactor_in);
	void redraw_grid(double xcenter_in, double ycenter_in, double xlength, double ylength, double* zfactor_in, double** betafactor_in);

	void set_splitting(int rs0, int ts0, int sl, int ccsl, double max_cs, bool neighbor_split);
	void set_default_imgsrch_settings();
	static void allocate_multithreaded_variables(const int& threads, const bool reallocate = true);
	static void deallocate_multithreaded_variables();
	void reset_search_parameters();
	~ImgSrchGrid();

	int nfound;
	double image_pos_accuracy;
	template <typename QScalar>
	void add_image_to_list(const lensvector<QScalar>& imgpos);
	template <typename QScalar>
	image<QScalar>* tree_search(const lensvector<QScalar> source_in);
	void subgrid_around_galaxies(lensvector<double>* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_splittings, bool* subgrid);

	void set_imagepos_accuracy(const double& setting) {
		image_pos_accuracy = setting;
	}

	// for plotting the grid to a file:
	//static std::ofstream xgrid;
	//void plot_corner_coordinates();
	void get_usplit_initial(int &setting) { setting = u_split_initial; }
	void get_wsplit_initial(int &setting) { setting = w_split_initial; }
};

template <typename QScalar>
class PtSrcParams : public ModelParams<QScalar>
{
	public:
	//QScalar pos_x, pos_y;
	lensvector<QScalar> pos;
	lensvector<QScalar> shift; // allows for a small correction to the source position estimated using analytic_bestfit_src
	QScalar zsrc, srcflux;
};

class PointSource : public Model
{
	friend class QLens;
	friend class SB_Profile;

	public:
	PtSrcParams<double> ptsrc_params;
#ifdef USE_STAN
	PtSrcParams<stan::math::var> ptsrc_params_dif; // autodiff version
#endif
	template <typename QScalar>
	PtSrcParams<QScalar>& assign_ptsrc_param_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return ptsrc_params_dif;
		else
#endif
		return ptsrc_params;
	}

	private:
	int zsrc_paramnum; // just to keep track of which parameter number is zsrc (set when constructor is called)

	public:
	bool include_shift;
	int n_images;
	std::vector<image<double>> images;

	public:
	PointSource() { qlens = NULL; }
	PointSource(QLens* lens_in);
	PointSource(QLens* lens_in, const lensvector<double>& sourcept, const double zsrc_in);
	PointSource(lensvector<double>& src_in, double zsrc_in, image<double>* images_in, const int nimg, const double srcflux_in = 1.0) {
		copy_imageset(src_in, zsrc_in, images_in, nimg, srcflux_in);
	}
	void setup_parameters(const bool initial_setup);
	template <typename QScalar>
	void setup_param_pointers();

	void update_meta_parameters(const bool varied_only_fitparams);
	void get_parameter_numbers_from_qlens(int& pi, int& pf);
	bool register_vary_parameters_in_qlens();
	void register_limits_in_qlens();
	void update_fitparams_in_qlens();
	void set_vary_source_coords();
#ifdef USE_STAN
	void sync_autodif_parameters();
#endif
	void copy_ptsrc_data(PointSource* ptsrc_in);
	void copy_imageset(const lensvector<double>& pos_in, const double zsrc_in, image<double>* images_in, const int nimg, const double srcflux_in = 1.0);
	void set_images(image<double>* images_in, const int nimg);
	template <typename QScalar>
	void update_srcpos(const lensvector<QScalar>& srcpt);
	double imgflux(const int imgnum) { if (imgnum < n_images) return abs(images[imgnum].mag*ptsrc_params.srcflux); else return -1; }
	void print(bool include_time_delays = false, bool show_labels = true) { print_to_file(include_time_delays,show_labels,NULL,NULL); }
	void print_to_file(bool include_time_delays, bool show_labels, std::ofstream* srcfile, std::ofstream* imgfile);
	void reset_images() { n_images = 0; images.clear(); }
	double& get_xcenter() { return ptsrc_params.pos[0]; }
	double& get_ycenter() { return ptsrc_params.pos[1]; }
	double& get_zsrc() { return ptsrc_params.zsrc; }
	lensvector<double>& get_pos() { return ptsrc_params.pos; }
#ifdef USE_STAN
	lensvector<stan::math::var>& get_pos_autodif() { return ptsrc_params_dif.pos; }
#endif
	double& get_srcflux() { return ptsrc_params.srcflux; }
	void set_srcflux(const double flux_in) { ptsrc_params.srcflux = flux_in; }
#ifdef USE_STAN
	stan::math::var& get_srcflux_autodif() { return ptsrc_params_dif.srcflux; }
	void set_srcflux_autodif(const stan::math::var flux_in) { ptsrc_params_dif.srcflux = flux_in; }
#endif
};

#endif // IMGSRCH_H
