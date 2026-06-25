#include "trirectangle.h"
#include "cg.h"
#include "pixelgrid.h"
#include "profile.h"
#include "sbprofile.h"
#include "qlens.h"
#include "mathexpr.h"
#include "errors.h"
#include <vector>
#include <complex>
//#include <functional>
#include <limits>
#include <stdio.h>

#ifdef USE_TBB
#include <tbb/enumerable_thread_specific.h>
#endif

#include <Eigen/Core>
#include "Eigen/Cholesky"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#ifdef USE_EIGEN_INV_NNLS
#include "unsupported/Eigen/NNLS"
#endif
//#include "fnnls.hpp"

#ifdef USE_UMFPACK
#include "umfpack.h"
#endif

#ifdef USE_FITS
#include "fitsio.h"
#endif

#ifdef USE_FFTW
#ifdef USE_MKL
#include "fftw/fftw3_mkl.h"
#else
#include "fftw3.h"
#endif
#endif

#if __cplusplus >= 201703L // C++17 standard or later
#include <filesystem>
#endif

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#define USE_COMM_WORLD -987654
#define MUMPS_SILENT -1
#define MUMPS_OUTPUT 6
#define JOB_INIT -1
#define JOB_END -2
using namespace std;

using VectorXd = Eigen::VectorXd;

int CartesianSourcePixel::nthreads = 0;
int CartesianSourcePixel::max_levels = 2;
int *CartesianSourcePixel::imin, *CartesianSourcePixel::imax, *CartesianSourcePixel::jmin, *CartesianSourcePixel::jmax;
TriRectangleOverlap *CartesianSourcePixel::trirec = NULL;
InterpolationCells *CartesianSourcePixel::nearest_interpolation_cells = NULL;
lensvector<double> **CartesianSourcePixel::interpolation_pts[3];
//int *CartesianSourcePixel::n_interpolation_pts = NULL;

//const int DelaunayGrid::nmax_pts_interp; // maximum number of allowed interpolation points; this number is initialized in pixelgrid.h

bool DelaunayGrid::zero_outside_border = true;

// variables for root finding to get point images (for combining with extended pixel images)
int ImagePixelGrid::nthreads = 0;
bool *ImagePixelGrid::newton_check = NULL;
lensvector<double> *ImagePixelGrid::fvec = NULL;
double ImagePixelGrid::image_pos_accuracy = 1e-6; // default

int *CartesianSourcePixel::maxlevs = NULL;
lensvector<double> ***CartesianSourcePixel::xvals_threads = NULL;
lensvector<double> ***CartesianSourcePixel::corners_threads = NULL;
lensvector<double> **CartesianSourcePixel::twistpts_threads = NULL;
int **CartesianSourcePixel::twist_status_threads = NULL;

/***************************************** Multithreaded variables in class ImagePixelGrid ****************************************/

void ImagePixelGrid::allocate_multithreaded_variables(const int& threads, const bool reallocate)
{
	if (newton_check != NULL) {
		if (!reallocate) return;
		else deallocate_multithreaded_variables();
	}
	nthreads = threads;
	newton_check = new bool[threads];
	fvec = new lensvector<double>[threads];
}

void ImagePixelGrid::deallocate_multithreaded_variables()
{
	if (newton_check != NULL) {
		delete[] newton_check;
		delete[] fvec;
		newton_check = NULL;
		fvec = NULL;
	}
}

/***************************************** Functions in class CartesianSourceGrid ****************************************/

void CartesianSourcePixel::allocate_multithreaded_variables(const int& threads, const bool reallocate)
{
	if (trirec != NULL) {
		if (!reallocate) return;
		else deallocate_multithreaded_variables();
	}
	nthreads = threads;
	trirec = new TriRectangleOverlap[nthreads];
	imin = new int[nthreads];
	imax = new int[nthreads];
	jmin = new int[nthreads];
	jmax = new int[nthreads];
	nearest_interpolation_cells = new InterpolationCells[nthreads];
	int i,j;
	for (i=0; i < 3; i++) interpolation_pts[i] = new lensvector<double>*[nthreads];
	//n_interpolation_pts = new int[threads];
	maxlevs = new int[threads];
	xvals_threads = new lensvector<double>**[threads];
	for (j=0; j < threads; j++) {
		xvals_threads[j] = new lensvector<double>*[3];
		for (i=0; i <= 2; i++) xvals_threads[j][i] = new lensvector<double>[3];
	}
	corners_threads = new lensvector<double>**[nthreads];
	for (int i=0; i < nthreads; i++) corners_threads[i] = new lensvector<double>*[4];
	twistpts_threads = new lensvector<double>*[nthreads];
	twist_status_threads = new int*[nthreads];
}

void CartesianSourcePixel::deallocate_multithreaded_variables()
{
	if (trirec != NULL) {
		delete[] trirec;
		delete[] imin;
		delete[] imax;
		delete[] jmin;
		delete[] jmax;
		delete[] nearest_interpolation_cells;
		delete[] maxlevs;
		for (int i=0; i < 3; i++) delete[] interpolation_pts[i];
		//delete[] n_interpolation_pts;
		int i,j;
		for (j=0; j < nthreads; j++) {
			for (i=0; i <= 2; i++) delete[] xvals_threads[j][i];
			delete[] xvals_threads[j];
			delete[] corners_threads[j];
		}
		delete[] xvals_threads;
		delete[] corners_threads;
		delete[] twistpts_threads;
		delete[] twist_status_threads;

		trirec = NULL;
		imin = NULL;
		imax = NULL;
		jmin = NULL;
		jmax = NULL;
		nearest_interpolation_cells = NULL;
		maxlevs = NULL;
		for (int i=0; i < 3; i++) interpolation_pts[i] = NULL;
		//n_interpolation_pts = NULL;
		xvals_threads = NULL;
		corners_threads = NULL;
		twistpts_threads = NULL;
		twist_status_threads = NULL;
	}
}

CartesianSourceGrid::CartesianSourceGrid(QLens* qlens_in, const int band, const double zsrc_in) : Model(), CartesianSourcePixel(qlens_in)
{
	parent_grid = this;
	qlens = qlens_in;
	if (zsrc_in < 0) model_name = "cartesian_srcgrid";
	else model_name = "cartesian_srcgrid(band=" + mkstring_int(band) + ",z=" + mkstring_doub(zsrc_in) + ")";
	cell = NULL;
	levels = 0;
	n_active_pixels = 0;
	srcgrid_redshift = zsrc_in;

	/*
	int threads = 1;
#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		threads = omp_get_num_threads();
	}
#endif
	allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)
	*/
	setup_parameters(true);
}

void CartesianSourceGrid::create_pixel_grid(QLens* qlens_in, double x_min, double x_max, double y_min, double y_max, const int usplit0, const int wsplit0) // use for top-level cell only; subcells use constructor below
{
	parent_grid = this;
	qlens = qlens_in;

	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) delete cell[i][j];
			delete[] cell[i];
		}
		delete[] cell;
		cell = NULL;
	}

	npixels_x = usplit0;
	npixels_y = wsplit0;
	if ((npixels_x < 2) or (npixels_y < 2)) die("source grid dimensions cannot be smaller than 2 along either direction");
	min_cell_area = 1e-6;

// this constructor is used for a Cartesian grid
	center_pt = 0;
	// For the Cartesian grid, u = x, w = y
	u_N = npixels_x;
	w_N = npixels_y;
	level = 0;
	levels = 0;
	ii=jj=0;
	maps_to_image_pixel = false;
	maps_to_image_window = false;
	active_pixel = false;

	for (int i=0; i < 4; i++) {
		corner_pt[i]=0;
		neighbor[i]=NULL;
	}

	xcenter = 0.5*(x_min+x_max);
	ycenter = 0.5*(y_min+y_max);
	srcgrid_xmin = x_min; srcgrid_xmax = x_max;
	srcgrid_ymin = y_min; srcgrid_ymax = y_max;

	double x, y, xstep, ystep;
	xstep = (x_max-x_min)/u_N;
	ystep = (y_max-y_min)/w_N;

	lensvector<double> **firstlevel_xvals = new lensvector<double>*[u_N+1];
	int i,j;
	for (i=0, x=x_min; i <= u_N; i++, x += xstep) {
		firstlevel_xvals[i] = new lensvector<double>[w_N+1];
		for (j=0, y=y_min; j <= w_N; j++, y += ystep) {
			firstlevel_xvals[i][j][0] = x;
			firstlevel_xvals[i][j][1] = y;
		}
	}

	cell = new CartesianSourcePixel**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new CartesianSourcePixel*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new CartesianSourcePixel(qlens,firstlevel_xvals,i,j,1,this);
		}
	}
	levels++;
	assign_firstlevel_neighbors();
	number_of_pixels = u_N*w_N;
	for (int i=0; i < u_N+1; i++)
		delete[] firstlevel_xvals[i];
	delete[] firstlevel_xvals;
}

void CartesianSourceGrid::create_pixel_grid(QLens* qlens_in, string pixel_data_fileroot, const double minarea_in) 	// use for top-level cell only; subcells use constructor below
{
	parent_grid = this;
	qlens = qlens_in;

	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) delete cell[i][j];
			delete[] cell[i];
		}
		delete[] cell;
		cell = NULL;
	}

	min_cell_area = minarea_in;
	string info_filename = pixel_data_fileroot + ".info";
	ifstream infofile(info_filename.c_str());
	double cells_per_pixel;
	infofile >> npixels_x >> npixels_y >> cells_per_pixel;
	infofile >> srcgrid_xmin >> srcgrid_xmax >> srcgrid_ymin >> srcgrid_ymax;
	min_cell_area = 1e-6;

	// this constructor is used for a Cartesian grid
	center_pt = 0;
	// For the Cartesian grid, u = x, w = y
	u_N = npixels_x;
	w_N = npixels_y;
	level = 0;
	levels = 0;
	ii=jj=0;
	cell = NULL;
	maps_to_image_pixel = false;
	maps_to_image_window = false;
	active_pixel = false;

	for (int i=0; i < 4; i++) {
		corner_pt[i]=0;
		neighbor[i]=NULL;
	}

	xcenter = 0.5*(srcgrid_xmin+srcgrid_xmax);
	ycenter = 0.5*(srcgrid_ymin+srcgrid_ymax);

	double x, y, xstep, ystep;
	xstep = (srcgrid_xmax-srcgrid_xmin)/u_N;
	ystep = (srcgrid_ymax-srcgrid_ymin)/w_N;

	lensvector<double> **firstlevel_xvals = new lensvector<double>*[u_N+1];
	int i,j;
	for (i=0, x=srcgrid_xmin; i <= u_N; i++, x += xstep) {
		firstlevel_xvals[i] = new lensvector<double>[w_N+1];
		for (j=0, y=srcgrid_ymin; j <= w_N; j++, y += ystep) {
			firstlevel_xvals[i][j][0] = x;
			firstlevel_xvals[i][j][1] = y;
		}
	}

	cell = new CartesianSourcePixel**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new CartesianSourcePixel*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new CartesianSourcePixel(qlens,firstlevel_xvals,i,j,1,this);
		}
	}
	levels++;
	assign_firstlevel_neighbors();
	number_of_pixels = u_N*w_N;

	string sbfilename = pixel_data_fileroot + ".sb";
	sb_infile.open(sbfilename.c_str());
	read_surface_brightness_data(sb_infile);
	sb_infile.close();
	for (int i=0; i < u_N+1; i++)
		delete[] firstlevel_xvals[i];
	delete[] firstlevel_xvals;
}

void CartesianSourceGrid::setup_parameters(const bool initial_setup)
{
	if (initial_setup) {
		// default initial values
		regparam = 100;
		pixel_fraction = 0.3;
		pixel_magnification_threshold = 7;
		srcgrid_size_scale = 0; // note, the source grid is scaled as xlength*(1+srcgrid_size_scale), etc.

		setup_parameter_arrays(4);
	} else {
		// always reset the active parameter flags, since the active ones will be determined below
		// NOTE: if (initial_setup==true), active params are reset in setup_parameter_arrays(..) above
		n_active_params = 0;
		for (int i=0; i < n_params; i++) {
			active_params[i] = false; // default
		}
	}

	int indx = 0;

	bool regularized_source, auto_srcgrid_npixels, adaptive_splitting, auto_srcgrid;
	if ((qlens->source_fit_mode==Cartesian_Source) and (qlens->regularization_method != None)) regularized_source = true;
	else regularized_source = false;
	auto_srcgrid_npixels = qlens->auto_srcgrid_npixels;
	adaptive_splitting = qlens->adaptive_subgrid;
	auto_srcgrid = qlens->auto_sourcegrid;

	if (initial_setup) {
		param[indx] = &regparam;
		paramnames[indx] = "regparam"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = "";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if (regularized_source) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		param[indx] = &pixel_fraction;
		paramnames[indx] = "pixfrac"; latex_paramnames[indx] = "f"; latex_param_subscripts[indx] = "pixel";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if (auto_srcgrid_npixels) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		param[indx] = &pixel_magnification_threshold;
		paramnames[indx] = "mag_threshold"; latex_paramnames[indx] = "m"; latex_param_subscripts[indx] = "split";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if (adaptive_splitting) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		param[indx] = &srcgrid_size_scale;
		paramnames[indx] = "srcgrid_scale"; latex_paramnames[indx] = "f"; latex_param_subscripts[indx] = "sg";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if (auto_srcgrid) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;
}

void CartesianSourceGrid::copy_pixsrc_data(CartesianSourceGrid* grid_in)
{
	model_name = grid_in->model_name;
	srcgrid_redshift = grid_in->srcgrid_redshift;
	regparam = grid_in->regparam;
	pixel_fraction = grid_in->pixel_fraction;
	pixel_magnification_threshold = grid_in->pixel_magnification_threshold;
	srcgrid_size_scale = grid_in->srcgrid_size_scale; // note, the source grid is scaled as xlength*(1+srcgrid_size_scale), etc.
	copy_param_arrays(grid_in);
}

void CartesianSourceGrid::update_meta_parameters(const bool varied_only_fitparams)
{
	return; // nothing meta to change
}

void CartesianSourceGrid::get_parameter_numbers_from_qlens(int& pi, int& pf)
{
	if (qlens) qlens->get_pixsrc_parameter_numbers(entry_number,pi,pf);
}

bool CartesianSourceGrid::register_vary_parameters_in_qlens()
{
	if (qlens != NULL) {
		return qlens->register_pixellated_src_vary_parameters(entry_number);
	}
	return true;
}

void CartesianSourceGrid::register_limits_in_qlens()
{
	if (qlens != NULL) {
		qlens->register_pixellated_src_prior_limits(entry_number);
	}
}

void CartesianSourceGrid::update_fitparams_in_qlens()
{
	if (qlens != NULL) {
		qlens->update_pixellated_src_fitparams(entry_number);
	}
}

// ***NOTE: the following constructor should NOT be used because there are static variables (e.g. levels), so more than one source grid
// is a bad idea. To make this work, you need to make those variables non-static and contained in the zeroth-level grid (and give subcells
// a pointer to the zeroth-level grid).
/*
CartesianSourceGrid::CartesianSourceGrid(QLens* lens_in, CartesianSourceGrid* input_pixel_grid) : qlens(lens_in)	// use for top-level cell only; subcells use constructor below
{
	int threads = 1;
#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		threads = omp_get_num_threads();
	}
#endif
	allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)

	// these are all static anyway, so this might be superfluous
	min_cell_area = input_pixel_grid->min_cell_area;
	npixels_x = input_pixel_grid->npixels_x;
	npixels_y = input_pixel_grid->npixels_y;
	srcgrid_xmin = input_pixel_grid->srcgrid_xmin;
	srcgrid_xmax = input_pixel_grid->srcgrid_xmax;
	srcgrid_ymin = input_pixel_grid->srcgrid_ymin;
	srcgrid_ymax = input_pixel_grid->srcgrid_ymax;

	// this constructor is used for a Cartesian grid
	center_pt = 0;
	// For the Cartesian grid, u = x, w = y
	u_N = npixels_x;
	w_N = npixels_y;
	level = 0;
	levels = 0;
	ii=jj=0;
	cell = NULL;
	maps_to_image_pixel = false;
	maps_to_image_window = false;
	active_pixel = false;

	for (int i=0; i < 4; i++) {
		corner_pt[i]=0;
		neighbor[i]=NULL;
	}

	xcenter = 0.5*(srcgrid_xmin+srcgrid_xmax);
	ycenter = 0.5*(srcgrid_ymin+srcgrid_ymax);

	double x, y, xstep, ystep;
	xstep = (srcgrid_xmax-srcgrid_xmin)/u_N;
	ystep = (srcgrid_ymax-srcgrid_ymin)/w_N;

	lensvector<double> **firstlevel_xvals = new lensvector<double>*[u_N+1];
	int i,j;
	for (i=0, x=srcgrid_xmin; i <= u_N; i++, x += xstep) {
		firstlevel_xvals[i] = new lensvector<double>[w_N+1];
		for (j=0, y=srcgrid_ymin; j <= w_N; j++, y += ystep) {
			firstlevel_xvals[i][j][0] = x;
			firstlevel_xvals[i][j][1] = y;
		}
	}

	cell = new CartesianSourceGrid**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new CartesianSourceGrid*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new CartesianSourceGrid(qlens,firstlevel_xvals,i,j,1,this);
		}
	}
	levels++;
	assign_firstlevel_neighbors();
	number_of_pixels = u_N*w_N;
	copy_source_pixel_grid(input_pixel_grid); // this copies the surface brightnesses and subpixel_maps_to_srcpixel the source pixels in the same manner as the input grid
	assign_all_neighbors();

	for (int i=0; i < u_N+1; i++)
		delete[] firstlevel_xvals[i];
	delete[] firstlevel_xvals;
}
*/

void CartesianSourcePixel::read_surface_brightness_data(ifstream &sb_infile)
{
	double sb;
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			sb_infile >> sb;
			if (sb==-1e30) // I can't think of a better dividing value to use right now, so -1e30 is what I am using at the moment
			{
				cell[i][j]->split_cells(2,2,0);
				cell[i][j]->read_surface_brightness_data(sb_infile);
			} else {
				cell[i][j]->surface_brightness = sb;
			}
		}
	}
}

/*
void CartesianSourceGrid::copy_source_pixel_grid(CartesianSourceGrid* input_pixel_grid)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (input_pixel_grid->cell[i][j]->cell != NULL) {
				cell[i][j]->split_cells(input_pixel_grid->cell[i][j]->u_N,input_pixel_grid->cell[i][j]->w_N,0);
				cell[i][j]->copy_source_pixel_grid(input_pixel_grid->cell[i][j]);
			} else {
				cell[i][j]->surface_brightness = input_pixel_grid->cell[i][j]->surface_brightness;
			}
		}
	}
}
*/

CartesianSourcePixel::CartesianSourcePixel(QLens* lens_in, lensvector<double>** xij, const int& i, const int& j, const int& level_in, CartesianSourceGrid* parent_ptr)
{
	parent_grid = parent_ptr;
	u_N = 1;
	w_N = 1;
	level = level_in;
	cell = NULL;
	ii=i; jj=j; // store the index carried by this cell in the grid of the parent cell
	maps_to_image_pixel = false;
	maps_to_image_window = false;
	surface_brightness = 0;
	lens = lens_in;

	corner_pt[0] = xij[i][j];
	corner_pt[1] = xij[i][j+1];
	corner_pt[2] = xij[i+1][j];
	corner_pt[3] = xij[i+1][j+1];

	center_pt[0] = (corner_pt[0][0] + corner_pt[1][0] + corner_pt[2][0] + corner_pt[3][0]) / 4.0;
	center_pt[1] = (corner_pt[0][1] + corner_pt[1][1] + corner_pt[2][1] + corner_pt[3][1]) / 4.0;
	cell_area = (corner_pt[2][0] - corner_pt[0][0])*(corner_pt[1][1]-corner_pt[0][1]);
}

void CartesianSourcePixel::assign_surface_brightness_from_analytic_source(const int imggrid_i)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->assign_surface_brightness_from_analytic_source(imggrid_i);
			else {
				cell[i][j]->surface_brightness = 0;
				for (int k=0; k < lens->n_sb; k++) {
					if ((lens->sb_list[k]->is_lensed) and ((imggrid_i < 0) or (lens->sbprofile_imggrid_idx[k]==imggrid_i))) cell[i][j]->surface_brightness += lens->sb_list[k]->surface_brightness(cell[i][j]->center_pt[0],cell[i][j]->center_pt[1]);
				}
			}
		}
	}
}

void CartesianSourcePixel::assign_surface_brightness_from_delaunay_grid(DelaunaySourceGrid* delaunay_grid, const bool add_sb)
{
	int i,j;
	double sb;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->assign_surface_brightness_from_delaunay_grid(delaunay_grid,add_sb);
			else {
				sb = delaunay_grid->find_lensed_surface_brightness(cell[i][j]->center_pt,-1,-1,0); // it would be nice to use Greg's method for searching so it doesn't start from an arbitrary triangle...but it's pretty fast as-is
				if (add_sb) cell[i][j]->surface_brightness += sb;
				else cell[i][j]->surface_brightness = sb;
			}
		}
	}
}

void CartesianSourcePixel::update_surface_brightness(int& index)
{
	if (image_pixel_grid==NULL) warn("cartesian source pixels cannot access image pixel grid; cannot update surface brightness from amplitudes");
	for (int j=0; j < w_N; j++) {
		for (int i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->update_surface_brightness(index);
			else {
				cell[i][j]->surface_brightness = image_pixel_grid->amplitude_vector[index++];
			}
		}
	}
}

void CartesianSourcePixel::fill_surface_brightness_vector()
{
	if (image_pixel_grid==NULL) warn("cartesian source pixels cannot access image pixel grid; cannot fill surface brightness vector");
	int column_j = 0;
	fill_surface_brightness_vector_recursive(column_j);
}

void CartesianSourcePixel::fill_surface_brightness_vector_recursive(int& column_j)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->fill_surface_brightness_vector_recursive(column_j);
			else {
				image_pixel_grid->amplitude_vector[column_j++] = cell[i][j]->surface_brightness;
			}
		}
	}
}

double CartesianSourceGrid::find_avg_n_images(const double sb_threshold_frac)
{
	// no support for adaptive Cartesian grid in this function, which is ok since we're only using this when Cartesian sources are not being used

	double max_pixel_sb=-1e30;
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->surface_brightness > max_pixel_sb) {
				max_pixel_sb = cell[i][j]->surface_brightness;
			}
		}
	}

	double pixel_avg_n_image = 0;
	double sbtot = 0;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->surface_brightness >= max_pixel_sb*sb_threshold_frac) {
				pixel_avg_n_image += cell[i][j]->n_images*cell[i][j]->surface_brightness;
				sbtot += cell[i][j]->surface_brightness;
			}
		}
	}
	if (sbtot != 0) pixel_avg_n_image /= sbtot;
	return pixel_avg_n_image;
}

/*
void CartesianSourceGrid::store_surface_brightness_grid_data(string root)
{
	string img_filename = root + ".sb";
	string info_filename = root + ".info";

	pixel_surface_brightness_file.open(img_filename.c_str());
	write_surface_brightness_to_file(pixel_surface_brightness_file);
	pixel_surface_brightness_file.close();

	ofstream pixel_info; qlens->open_output_file(pixel_info,info_filename);
	pixel_info << npixels_x << " " << npixels_y << " " << levels << endl;
	pixel_info << srcgrid_xmin << " " << srcgrid_xmax << " " << srcgrid_ymin << " " << srcgrid_ymax << endl;
}
*/

void CartesianSourcePixel::write_surface_brightness_to_file(ofstream &sb_outfile)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) {
				sb_outfile << "-1e30\n";
				cell[i][j]->write_surface_brightness_to_file(sb_outfile);
			} else {
				sb_outfile << cell[i][j]->surface_brightness << endl;
			}
		}
	}
}

void CartesianSourceGrid::get_grid_dimensions(double &xmin, double &xmax, double &ymin, double &ymax)
{
	xmin = cell[0][0]->corner_pt[0][0];
	ymin = cell[0][0]->corner_pt[0][1];
	xmax = cell[u_N-1][w_N-1]->corner_pt[3][0];
	ymax = cell[u_N-1][w_N-1]->corner_pt[3][1];
}

void CartesianSourceGrid::output_surface_brightness(Vector<double>& xvals, Vector<double>& yvals, Vector<double>& sbvals, Vector<double>& maglogvals, Vector<double>& nimgvals)
{
	double x, y, cell_xlength, cell_ylength, xmin, ymin;
	int i, j, k, n_plot_xcells, n_plot_ycells, pixels_per_cell_x, pixels_per_cell_y;
	cell_xlength = cell[0][0]->corner_pt[2][0] - cell[0][0]->corner_pt[0][0];
	cell_ylength = cell[0][0]->corner_pt[1][1] - cell[0][0]->corner_pt[0][1];
	n_plot_xcells = u_N;
	n_plot_ycells = w_N;
	pixels_per_cell_x = 1;
	pixels_per_cell_y = 1;
	for (i=0; i < levels-1; i++) {
		cell_xlength /= 2;
		cell_ylength /= 2;
		n_plot_xcells *= 2;
		n_plot_ycells *= 2;
		pixels_per_cell_x *= 2;
		pixels_per_cell_y *= 2;
	}
	xmin = cell[0][0]->corner_pt[0][0];
	ymin = cell[0][0]->corner_pt[0][1];

	xvals.input(n_plot_xcells+1);
	yvals.input(n_plot_ycells+1);
	for (i=0, x=xmin; i <= n_plot_xcells; i++, x += cell_xlength) xvals[i] = x;
	for (i=0, y=ymin; i <= n_plot_ycells; i++, y += cell_ylength) yvals[i] = y;

	sbvals.input(n_plot_xcells*n_plot_ycells);
	maglogvals.input(n_plot_xcells*n_plot_ycells);
	nimgvals.input(n_plot_xcells*n_plot_ycells);

	int line_number;
	int l=0;
	for (j=0; j < w_N; j++) {
		for (line_number=0; line_number < pixels_per_cell_y; line_number++) {
			for (i=0; i < u_N; i++) {
				if (cell[i][j]->cell != NULL) {
					cell[i][j]->output_cell_surface_brightness(line_number,pixels_per_cell_x,pixels_per_cell_y,sbvals,maglogvals,nimgvals,l);
				} else {
					for (k=0; k < pixels_per_cell_x; k++) {
						sbvals[l] = cell[i][j]->surface_brightness;
						maglogvals[l] = log(cell[i][j]->total_magnification)/log(10);
						if (qlens->n_image_prior) nimgvals[l] = cell[i][j]->n_images;
						else nimgvals[l] = 0;
						//pixel_surface_brightness_file << cell[i][j]->surface_brightness << " ";
						//pixel_magnification_file << log(cell[i][j]->total_magnification)/log(10) << " ";
						//if (qlens->n_image_prior) pixel_n_image_file << cell[i][j]->n_images << " ";
						l++;
					}
				}
			}
		}
	}
}

/*
void CartesianSourceGrid::output_fits_file(string fits_filename)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return;
#else
	int i,j,kk;
	fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix = -64, naxis = 2;
	long naxes[2] = {u_N,w_N};
	double *pixels;
	if (qlens->fit_output_dir != ".") qlens->create_output_directory(); // in case it hasn't been created already
	string filename = qlens->fit_output_dir + "/" + fits_filename;

	double cell_xlength, cell_ylength;
	cell_xlength = cell[0][0]->corner_pt[2][0] - cell[0][0]->corner_pt[0][0];
	cell_ylength = cell[0][0]->corner_pt[1][1] - cell[0][0]->corner_pt[0][1];

	if (!fits_create_file(&outfptr, filename.c_str(), &status))
	{
		if (!fits_create_img(outfptr, bitpix, naxis, naxes, &status))
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				fits_write_key(outfptr, TDOUBLE, "PXSIZE_X", &cell_xlength, "length of pixels along the x direction (in arcsec)", &status);
				fits_write_key(outfptr, TDOUBLE, "PXSIZE_Y", &cell_ylength, "length of pixels along the y direction (in arcsec)", &status);

				kk=0;
				long fpixel[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[u_N];

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=0; i < u_N; i++) {
						pixels[i] = cell[i][j]->surface_brightness;
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] pixels;
			}
		}
		fits_close_file(outfptr, &status);
	} 

	if (status) fits_report_error(stderr, status); // print any error message
#endif
}
*/

void CartesianSourcePixel::output_cell_surface_brightness(int line_number, int pixels_per_cell_x, int pixels_per_cell_y, Vector<double>& sbvals, Vector<double>& maglogvals, Vector<double>& nimgvals, int& indx)
{
	int cell_row, subplot_pixels_per_cell_x, subplot_pixels_per_cell_y, subline_number=line_number;
	subplot_pixels_per_cell_x = pixels_per_cell_x/u_N;
	subplot_pixels_per_cell_y = pixels_per_cell_y/w_N;
	cell_row = line_number / subplot_pixels_per_cell_y;
	subline_number -= cell_row*subplot_pixels_per_cell_y;

	int i,j;
	for (i=0; i < u_N; i++) {
		if (cell[i][cell_row]->cell != NULL) {
			cell[i][cell_row]->output_cell_surface_brightness(subline_number,subplot_pixels_per_cell_x,subplot_pixels_per_cell_y,sbvals,maglogvals,nimgvals,indx);
		} else {
			for (j=0; j < subplot_pixels_per_cell_x; j++) {
				sbvals[indx] = cell[i][cell_row]->surface_brightness;
				maglogvals[indx] = log(cell[i][cell_row]->total_magnification)/log(10);
				if (lens->n_image_prior) nimgvals[indx] = cell[i][cell_row]->n_images;
				indx++;
			}
		}
	}
}



void CartesianSourcePixel::plot_cell_surface_brightness(int line_number, int pixels_per_cell_x, int pixels_per_cell_y, ofstream& sb_outfile, ofstream& mag_outfile, ofstream &nimg_outfile)
{
	int cell_row, subplot_pixels_per_cell_x, subplot_pixels_per_cell_y, subline_number=line_number;
	subplot_pixels_per_cell_x = pixels_per_cell_x/u_N;
	subplot_pixels_per_cell_y = pixels_per_cell_y/w_N;
	cell_row = line_number / subplot_pixels_per_cell_y;
	subline_number -= cell_row*subplot_pixels_per_cell_y;

	int i,j;
	for (i=0; i < u_N; i++) {
		if (cell[i][cell_row]->cell != NULL) {
			cell[i][cell_row]->plot_cell_surface_brightness(subline_number,subplot_pixels_per_cell_x,subplot_pixels_per_cell_y,sb_outfile,mag_outfile,nimg_outfile);
		} else {
			for (j=0; j < subplot_pixels_per_cell_x; j++) {
				sb_outfile << cell[i][cell_row]->surface_brightness << " ";
				mag_outfile << log(cell[i][cell_row]->total_magnification)/log(10) << " ";
				if (lens->n_image_prior) nimg_outfile << cell[i][cell_row]->n_images << " ";
			}
		}
	}
}

void CartesianSourceGrid::assign_firstlevel_neighbors()
{
	// neighbor index: 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	if (level != 0) die("assign_firstlevel_neighbors function must be run from grid level 0");
	int i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			if (j < w_N-1)
				cell[i][j]->neighbor[2] = cell[i][j+1];
			else
				cell[i][j]->neighbor[2] = NULL;

			if (j > 0) 
				cell[i][j]->neighbor[3] = cell[i][j-1];
			else
				cell[i][j]->neighbor[3] = NULL;

			if (i > 0) {
				cell[i][j]->neighbor[1] = cell[i-1][j];
				if (i < u_N-1)
					cell[i][j]->neighbor[0] = cell[i+1][j];
				else
					cell[i][j]->neighbor[0] = NULL;
			} else {
				cell[i][j]->neighbor[1] = NULL;
				cell[i][j]->neighbor[0] = cell[i+1][j];
			}
		}
	}
}

/*
void CartesianSourceGrid::assign_neighborhood()
{
	// assign neighbors of this cell, then update neighbors of neighbors of this cell
	// neighbor index: 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	assign_level_neighbors(level);
	int l,k;
	for (l=0; l < 4; l++)
		if ((neighbor[l] != NULL) and (neighbor[l]->cell != NULL)) {
		for (k=level; k <= levels; k++) {
			neighbor[l]->assign_level_neighbors(k);
		}
	}
}
*/

void CartesianSourceGrid::assign_all_neighbors()
{
	if (level!=0) die("assign_all_neighbors should only be run from level 0");

	int k,i,j;
	for (k=1; k < levels; k++) {
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->assign_level_neighbors(k); // we've just created our grid, so we only need to go to level+1
			}
		}
	}
}

void CartesianSourcePixel::test_neighbors() // for testing purposes, to make sure neighbors are assigned correctly
{
	int k,i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->test_neighbors();
			else {
				for (k=0; k < 4; k++) {
					if (cell[i][j]->neighbor[k] == NULL)
						cout << "Level " << cell[i][j]->level << " cell (" << i << "," << j << ") neighbor " << k << ": none\n";
					else
						cout << "Level " << cell[i][j]->level << " cell (" << i << "," << j << ") neighbor " << k << ": level " << cell[i][j]->neighbor[k]->level << " (" << cell[i][j]->neighbor[k]->ii << "," << cell[i][j]->neighbor[k]->jj << ")\n";
				}
			}
		}
	}
}

void CartesianSourcePixel::assign_level_neighbors(int neighbor_level)
{
	if (cell == NULL) return;
	int i,j;
	if (level < neighbor_level) {
		for (i=0; i < u_N; i++)
			for (j=0; j < w_N; j++)
				cell[i][j]->assign_level_neighbors(neighbor_level);
	} else {
		if (cell==NULL) die("cannot find neighbors if no grid has been set up");
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				if (cell[i][j]==NULL) die("a subcell has been erased");
				if (i < u_N-1)
					cell[i][j]->neighbor[0] = cell[i+1][j];
				else {
					if (neighbor[0] == NULL) cell[i][j]->neighbor[0] = NULL;
					else if (neighbor[0]->cell != NULL) {
						if (j >= neighbor[0]->w_N) cell[i][j]->neighbor[0] = neighbor[0]->cell[0][neighbor[0]->w_N-1]; // allows for possibility that neighbor cell may not be split by the same number of cells
						else cell[i][j]->neighbor[0] = neighbor[0]->cell[0][j];
					} else
						cell[i][j]->neighbor[0] = neighbor[0];
				}

				if (i > 0)
					cell[i][j]->neighbor[1] = cell[i-1][j];
				else {
					if (neighbor[1] == NULL) cell[i][j]->neighbor[1] = NULL;
					else if (neighbor[1]->cell != NULL) {
						if (j >= neighbor[1]->w_N) cell[i][j]->neighbor[1] = neighbor[1]->cell[neighbor[1]->u_N-1][neighbor[1]->w_N-1]; // allows for possibility that neighbor cell may not be split by the same number of cells

						else cell[i][j]->neighbor[1] = neighbor[1]->cell[neighbor[1]->u_N-1][j];
					} else
						cell[i][j]->neighbor[1] = neighbor[1];
				}

				if (j < w_N-1)
					cell[i][j]->neighbor[2] = cell[i][j+1];
				else {
					if (neighbor[2] == NULL) cell[i][j]->neighbor[2] = NULL;
					else if (neighbor[2]->cell != NULL) {
						if (i >= neighbor[2]->u_N) cell[i][j]->neighbor[2] = neighbor[2]->cell[neighbor[2]->u_N-1][0];
						else cell[i][j]->neighbor[2] = neighbor[2]->cell[i][0];
					} else
						cell[i][j]->neighbor[2] = neighbor[2];
				}

				if (j > 0)
					cell[i][j]->neighbor[3] = cell[i][j-1];
				else {
					if (neighbor[3] == NULL) cell[i][j]->neighbor[3] = NULL;
					else if (neighbor[3]->cell != NULL) {
						if (i >= neighbor[3]->u_N) cell[i][j]->neighbor[3] = neighbor[3]->cell[neighbor[3]->u_N-1][neighbor[3]->w_N-1];
						else cell[i][j]->neighbor[3] = neighbor[3]->cell[i][neighbor[3]->w_N-1];
					} else
						cell[i][j]->neighbor[3] = neighbor[3];
				}
			}
		}
	}
}

void CartesianSourcePixel::split_cells(const int usplit, const int wsplit, const int& thread)
{
	if (level >= max_levels+1)
		die("maximum number of splittings has been reached (%i)", max_levels);
	if (cell != NULL)
		die("subcells should not already be present in split_cells routine");

	u_N = usplit;
	w_N = wsplit;
	int i,j;
	for (i=0; i <= u_N; i++) {
		for (j=0; j <= w_N; j++) {
			xvals_threads[thread][i][j][0] = ((corner_pt[0][0]*(w_N-j) + corner_pt[1][0]*j)*(u_N-i) + (corner_pt[2][0]*(w_N-j) + corner_pt[3][0]*j)*i)/(u_N*w_N);
			xvals_threads[thread][i][j][1] = ((corner_pt[0][1]*(w_N-j) + corner_pt[1][1]*j)*(u_N-i) + (corner_pt[2][1]*(w_N-j) + corner_pt[3][1]*j)*i)/(u_N*w_N);
		}
	}

	cell = new CartesianSourcePixel**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new CartesianSourcePixel*[w_N];
		for (j=0; j < w_N; j++) {
			cell[i][j] = new CartesianSourcePixel(lens,xvals_threads[thread],i,j,level+1,parent_grid);
			cell[i][j]->total_magnification = 0;
			if (lens->n_image_prior) cell[i][j]->n_images = 0;
		}
	}
	if (level == maxlevs[thread]) {
		maxlevs[thread]++; // our subcells are at the max level, so splitting them increases the number of levels by 1
	}
	parent_grid->number_of_pixels += u_N*w_N - 1; // subtract one because we're not counting the parent cell as a source pixel
}

void CartesianSourcePixel::unsplit()
{
	if (cell==NULL) return;
	surface_brightness = 0;
	int i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->unsplit();
			surface_brightness += cell[i][j]->surface_brightness;
			delete cell[i][j];
		}
		delete[] cell[i];
	}
	delete[] cell;
	parent_grid->number_of_pixels -= (u_N*w_N - 1);
	cell = NULL;
	surface_brightness /= (u_N*w_N);
	u_N=1; w_N = 1;
}

void CartesianSourcePixel::plot_corner_coordinates(ofstream &gridout)
{
	if (level > 0) {
		gridout << corner_pt[1][0] << " " << corner_pt[1][1] << endl;
		gridout << corner_pt[3][0] << " " << corner_pt[3][1] << endl;
		gridout << corner_pt[2][0] << " " << corner_pt[2][1] << endl;
		gridout << corner_pt[0][0] << " " << corner_pt[0][1] << endl;
		gridout << corner_pt[1][0] << " " << corner_pt[1][1] << endl;
		gridout << endl;
	}

	if (cell != NULL) {
		for (int i=0; i < u_N; i++)
			for (int j=0; j < w_N; j++)
				cell[i][j]->plot_corner_coordinates(gridout);
	}
}

double CartesianSourceGrid::find_triangle_weighted_invmag(lensvector<double>& pt1, lensvector<double>& pt2, lensvector<double>& pt3, double& total_overlap, const int thread)
{
	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_overlap(pt1,pt2,pt3,thread)==false) return false; 

	total_overlap = 0;
	double total_weighted_invmag = 0;
	double overlap;
	int i,j;
	lensvector<double> *cornerpt;
	for (j=jmin[thread]; j <= jmax[thread]; j++) {
		for (i=imin[thread]; i <= imax[thread]; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->find_triangle_weighted_invmag_subcell(pt1,pt2,pt3,total_overlap,total_weighted_invmag,thread); // put in recursion later
			else {
				//cout << "before: winvmag=" << total_weighted_invmag << endl;
				cornerpt = cell[i][j]->corner_pt;
				overlap = trirec[thread].find_overlap_area(pt1,pt2,pt3,cornerpt[0][0],cornerpt[2][0],cornerpt[0][1],cornerpt[1][1]);
				if (overlap != 0) {
					total_overlap += overlap;
					if (cell[i][j]->total_magnification != 0) total_weighted_invmag += overlap*(1.0/cell[i][j]->total_magnification);
				}
				//if (overlap != 0) cout << "overlap=" << overlap << " mag=" << cell[i][j]->total_magnification << " wtf=" << (1.0/cell[i][j]->total_magnification) << " winvmag=" << total_weighted_invmag << endl;
			}
		}
	}
	if (total_weighted_invmag*0.0 != 0.0) die("FUCK");
	return total_weighted_invmag;
}

void CartesianSourcePixel::find_triangle_weighted_invmag_subcell(lensvector<double>& pt1, lensvector<double>& pt2, lensvector<double>& pt3, double& total_overlap, double& total_weighted_invmag, const int& thread)
{
	int i,j;
	double overlap;
	lensvector<double> *cornerpt;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->find_triangle_weighted_invmag_subcell(pt1,pt2,pt3,total_overlap,total_weighted_invmag,thread);
			else {
				cornerpt = cell[i][j]->corner_pt;
				overlap = trirec[thread].find_overlap_area(pt1,pt2,pt3,cornerpt[0][0],cornerpt[2][0],cornerpt[0][1],cornerpt[1][1]);
				if (overlap != 0) {
					total_overlap += overlap;
					if (cell[i][j]->total_magnification != 0) total_weighted_invmag += overlap*(1.0/cell[i][j]->total_magnification);
				}
			}
		}
	}
}

inline bool CartesianSourcePixel::check_if_in_neighborhood(lensvector<double> **input_corner_pts, bool& inside, const int& thread)
{
	if (trirec[thread].determine_if_in_neighborhood(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],*input_corner_pts[3],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1],inside)==true) return true;
	return false;
}

inline bool CartesianSourcePixel::check_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		if (trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1])==true) return true;
		if (trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1])==true) return true;
	} else if (twist_status==1) {
		if (trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1])==true) return true;
		if (trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1])==true) return true;
	} else {
		if (trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1])==true) return true;
		if (trirec[thread].determine_if_overlap(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1])==true) return true;
	}
	return false;
}

inline double CartesianSourcePixel::find_rectangle_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread, const int& i, const int& j)
{
	if (twist_status==0) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else if (twist_status==1) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	}
}

inline bool CartesianSourcePixel::check_triangle1_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else if (twist_status==1) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	}
}

inline bool CartesianSourcePixel::check_triangle2_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else if (twist_status==1) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else {
		return trirec[thread].determine_if_overlap(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	}
}

inline double CartesianSourcePixel::find_triangle1_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else if (twist_status==1) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	}
}

inline double CartesianSourcePixel::find_triangle2_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else if (twist_status==1) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else {
		return (trirec[thread].find_overlap_area(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	}
}

bool CartesianSourceGrid::bisection_search_overlap(lensvector<double> **input_corner_pts, const int& thread)
{
	int i, imid, jmid;
	bool inside;
	bool inside_corner[4];
	int n_inside;
	double xmin[4], xmax[4], ymin[4], ymax[4];
	int reduce_mid = 0;

	for (;;) {
		n_inside=0;
		for (i=0; i < 4; i++) inside_corner[i] = false;
		if (reduce_mid==0) {
			imid = (imax[thread] + imin[thread])/2;
			jmid = (jmax[thread] + jmin[thread])/2;
		} else if (reduce_mid==1) {
			imid = (imax[thread] + 2*imin[thread])/3;
			jmid = (jmax[thread] + 2*jmin[thread])/3;
		} else if (reduce_mid==2) {
			imid = (2*imax[thread] + imin[thread])/3;
			jmid = (2*jmax[thread] + jmin[thread])/3;
		} else if (reduce_mid==3) {
			imid = (imax[thread] + 2*imin[thread])/3;
			jmid = (2*jmax[thread] + jmin[thread])/3;
		} else if (reduce_mid==4) {
			imid = (2*imax[thread] + imin[thread])/3;
			jmid = (jmax[thread] + 2*jmin[thread])/3;
		}
		if ((imid==imin[thread]) or ((imid==imax[thread]))) break;
		if ((jmid==jmin[thread]) or ((jmid==jmax[thread]))) break;
		xmin[0] = cell[imin[thread]][jmin[thread]]->corner_pt[0][0];
		ymin[0] = cell[imin[thread]][jmin[thread]]->corner_pt[0][1];
		xmax[0] = cell[imid][jmid]->corner_pt[3][0];
		ymax[0] = cell[imid][jmid]->corner_pt[3][1];

		xmin[1] = cell[imin[thread]][jmid+1]->corner_pt[0][0];
		ymin[1] = cell[imin[thread]][jmid+1]->corner_pt[0][1];
		xmax[1] = cell[imid][jmax[thread]]->corner_pt[3][0];
		ymax[1] = cell[imid][jmax[thread]]->corner_pt[3][1];

		xmin[2] = cell[imid+1][jmin[thread]]->corner_pt[0][0];
		ymin[2] = cell[imid+1][jmin[thread]]->corner_pt[0][1];
		xmax[2] = cell[imax[thread]][jmid]->corner_pt[3][0];
		ymax[2] = cell[imax[thread]][jmid]->corner_pt[3][1];

		xmin[3] = cell[imid+1][jmid+1]->corner_pt[0][0];
		ymin[3] = cell[imid+1][jmid+1]->corner_pt[0][1];
		xmax[3] = cell[imax[thread]][jmax[thread]]->corner_pt[3][0];
		ymax[3] = cell[imax[thread]][jmax[thread]]->corner_pt[3][1];

		for (i=0; i < 4; i++) {
			inside = false;
			if (trirec[thread].determine_if_in_neighborhood(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],*input_corner_pts[3],xmin[i],xmax[i],ymin[i],ymax[i],inside)) {
				if (inside) inside_corner[i] = true;
				else if (trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],xmin[i],xmax[i],ymin[i],ymax[i])) inside_corner[i] = true;
				else if (trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[2],*input_corner_pts[3],xmin[i],xmax[i],ymin[i],ymax[i])) inside_corner[i] = true;
				if (inside_corner[i]) n_inside++;
			}
		}
		if (n_inside==0) return false;
		if (n_inside > 1) {
			if (reduce_mid>0) {
				if (reduce_mid < 4) { reduce_mid++; continue; }
				else break; // tried shifting the dividing lines to 1/3 & 2/3 positions, just in case the cell was straddling the middle, but still didn't contain the cell, so give up
			}
			else {
				reduce_mid = 1;
				continue;
			}
		} else if (reduce_mid>0) {
			reduce_mid = 0;
		}

		if (inside_corner[0]) { imax[thread]=imid; jmax[thread]=jmid; }
		else if (inside_corner[1]) { imax[thread]=imid; jmin[thread]=jmid; }
		else if (inside_corner[2]) { imin[thread]=imid; jmax[thread]=jmid; }
		else if (inside_corner[3]) { imin[thread]=imid; jmin[thread]=jmid; }
		if ((imax[thread] - imin[thread] <= 1) or (jmax[thread] - jmin[thread] <= 1)) break;
	}
	return true;
}

bool CartesianSourceGrid::bisection_search_overlap(lensvector<double> &a, lensvector<double> &b, lensvector<double> &c, const int& thread)
{
	int i, imid, jmid;
	bool inside;
	bool inside_corner[4];
	int n_inside;
	double xmin[4], xmax[4], ymin[4], ymax[4];
	int reduce_mid = 0;

	for (;;) {
		n_inside=0;
		for (i=0; i < 4; i++) inside_corner[i] = false;
		if (reduce_mid==0) {
			imid = (imax[thread] + imin[thread])/2;
			jmid = (jmax[thread] + jmin[thread])/2;
		} else if (reduce_mid==1) {
			imid = (imax[thread] + 2*imin[thread])/3;
			jmid = (jmax[thread] + 2*jmin[thread])/3;
		} else if (reduce_mid==2) {
			imid = (2*imax[thread] + imin[thread])/3;
			jmid = (2*jmax[thread] + jmin[thread])/3;
		} else if (reduce_mid==3) {
			imid = (imax[thread] + 2*imin[thread])/3;
			jmid = (2*jmax[thread] + jmin[thread])/3;
		} else if (reduce_mid==4) {
			imid = (2*imax[thread] + imin[thread])/3;
			jmid = (jmax[thread] + 2*jmin[thread])/3;
		}
		if ((imid==imin[thread]) or ((imid==imax[thread]))) break;
		if ((jmid==jmin[thread]) or ((jmid==jmax[thread]))) break;
		xmin[0] = cell[imin[thread]][jmin[thread]]->corner_pt[0][0];
		ymin[0] = cell[imin[thread]][jmin[thread]]->corner_pt[0][1];
		xmax[0] = cell[imid][jmid]->corner_pt[3][0];
		ymax[0] = cell[imid][jmid]->corner_pt[3][1];

		xmin[1] = cell[imin[thread]][jmid+1]->corner_pt[0][0];
		ymin[1] = cell[imin[thread]][jmid+1]->corner_pt[0][1];
		xmax[1] = cell[imid][jmax[thread]]->corner_pt[3][0];
		ymax[1] = cell[imid][jmax[thread]]->corner_pt[3][1];

		xmin[2] = cell[imid+1][jmin[thread]]->corner_pt[0][0];
		ymin[2] = cell[imid+1][jmin[thread]]->corner_pt[0][1];
		xmax[2] = cell[imax[thread]][jmid]->corner_pt[3][0];
		ymax[2] = cell[imax[thread]][jmid]->corner_pt[3][1];

		xmin[3] = cell[imid+1][jmid+1]->corner_pt[0][0];
		ymin[3] = cell[imid+1][jmid+1]->corner_pt[0][1];
		xmax[3] = cell[imax[thread]][jmax[thread]]->corner_pt[3][0];
		ymax[3] = cell[imax[thread]][jmax[thread]]->corner_pt[3][1];

		for (i=0; i < 4; i++) {
			inside = false;
			if (trirec[thread].determine_if_in_neighborhood(a,b,c,xmin[i],xmax[i],ymin[i],ymax[i],inside)) {
				if (inside) inside_corner[i] = true;
				else if (trirec[thread].determine_if_overlap(a,b,c,xmin[i],xmax[i],ymin[i],ymax[i])) inside_corner[i] = true;
				if (inside_corner[i]) n_inside++;
			}
		}
		if (n_inside==0) return false;
		if (n_inside > 1) {
			if (reduce_mid>0) {
				if (reduce_mid < 4) { reduce_mid++; continue; }
				else break; // tried shifting the dividing lines to 1/3 & 2/3 positions, just in case the cell was straddling the middle, but still didn't contain the cell, so give up
			}
			else {
				reduce_mid = 1;
				continue;
			}
		} else if (reduce_mid>0) {
			reduce_mid = 0;
		}

		if (inside_corner[0]) { imax[thread]=imid; jmax[thread]=jmid; }
		else if (inside_corner[1]) { imax[thread]=imid; jmin[thread]=jmid; }
		else if (inside_corner[2]) { imin[thread]=imid; jmax[thread]=jmid; }
		else if (inside_corner[3]) { imin[thread]=imid; jmin[thread]=jmid; }
		if ((imax[thread] - imin[thread] <= 1) or (jmax[thread] - jmin[thread] <= 1)) break;
	}
	return true;
}

void CartesianSourceGrid::calculate_pixel_magnifications(const bool use_emask)
{
	ImgGrid_Params<PlainTypes>& imggrid_params = image_pixel_grid->assign_imggrid_param_object<PlainTypes>();
	ImageData *imgpixel_data = image_pixel_grid->image_data;

	qlens->total_srcgrid_overlap_area = 0; // Used to find the total coverage of the sourcegrid, which helps determine optimal source pixel size
	qlens->high_sn_srcgrid_overlap_area = 0; // Used to find the total coverage of the sourcegrid, which helps determine optimal source pixel size

	int i,j,k,nsrc;
	double overlap_area, weighted_overlap, triangle1_overlap, triangle2_overlap, triangle1_weight, triangle2_weight;
	bool inside;
	clear_subgrids();
	int ntot_src = u_N*w_N;
	double *area_matrix, *mag_matrix;
	double *high_sn_area_matrix;
	mag_matrix = new double[ntot_src];
	area_matrix = new double[ntot_src];
	high_sn_area_matrix = new double[ntot_src];
	for (i=0; i < ntot_src; i++) {
		area_matrix[i] = 0;
		high_sn_area_matrix[i] = 0;
		mag_matrix[i] = 0;
	}
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			cell[i][j]->overlap_pixel_n.clear();
		}
	}

	std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	//ofstream wout("wout.dat");
	double xstep, ystep;
	xstep = (srcgrid_xmax-srcgrid_xmin)/u_N;
	ystep = (srcgrid_ymax-srcgrid_ymin)/w_N;
	int src_raytrace_i, src_raytrace_j;
	int img_i, img_j;

	long int ntot_cells = (use_emask) ? image_pixel_grid->image_npixels_emask : image_pixel_grid->image_npixels;

	int *overlap_matrix_row_nn = new int[ntot_cells];
	vector<double> *overlap_matrix_rows = new vector<double>[ntot_cells];
	vector<int> *overlap_matrix_index_rows = new vector<int>[ntot_cells];
	vector<double> *overlap_area_matrix_rows;
	overlap_area_matrix_rows = new vector<double>[ntot_cells];

	int overlap_matrix_nn;
	int overlap_matrix_nn_part=0;
	//ofstream wtfout("wtf.dat");
	#pragma omp parallel
	{
		int n, img_i, img_j;
		bool inside;
		int thread;
		int corner_raytrace_i;
		int corner_raytrace_j;
		int min_i, max_i, min_j, max_j;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		#pragma omp for private(i,j,nsrc,overlap_area,weighted_overlap,triangle1_overlap,triangle2_overlap,triangle1_weight,triangle2_weight,inside) schedule(dynamic) reduction(+:overlap_matrix_nn_part)
		for (n=0; n < ntot_cells; n++)
		{
			overlap_matrix_row_nn[n] = 0;
			//img_j = n / image_pixel_grid->x_N;
			//img_i = n % image_pixel_grid->x_N;
			if (use_emask) {
				img_j = image_pixel_grid->emask_pixels_j[n];
				img_i = image_pixel_grid->emask_pixels_i[n];
			} else {
				img_j = image_pixel_grid->mask_pixels_j[n];
				img_i = image_pixel_grid->mask_pixels_i[n];
			}
			if (image_pixel_grid->pixel_mag[img_i][img_j] < qlens->srcpixel_nimg_mag_threshold) continue;
			//wtfout << image_pixel_grid->center_pts[img_i][img_j][0] << " " << image_pixel_grid->center_pts[img_i][img_j][1] << endl;

			corners_threads[thread][0] = &image_pixel_grid->corner_sourcepts[img_i][img_j];
			corners_threads[thread][1] = &image_pixel_grid->corner_sourcepts[img_i][img_j+1];
			corners_threads[thread][2] = &image_pixel_grid->corner_sourcepts[img_i+1][img_j];
			corners_threads[thread][3] = &image_pixel_grid->corner_sourcepts[img_i+1][img_j+1];
			//for (int l=0; l < 4; l++) if ((*corners_threads[thread][l])[0]==-5000) {
				//cout << "WHOOPS! " << l << " " << img_i << " " << img_j << " " << endl;
				//cout << "checking corner 0: " << image_pixel_grid->corner_sourcepts[img_i][img_j][0] << " " << image_pixel_grid->corner_sourcepts[img_i][img_j][1] << endl;
				//cout << "checking corner 1: " << image_pixel_grid->corner_sourcepts[img_i][img_j+1][0] << " " << image_pixel_grid->corner_sourcepts[img_i][img_j+1][1] << endl;
				//cout << "checking corner 2: " << image_pixel_grid->corner_sourcepts[img_i+1][img_j][0] << " " << image_pixel_grid->corner_sourcepts[img_i+1][img_j][1] << endl;
				//cout << "checking corner 3: " << image_pixel_grid->corner_sourcepts[img_i+1][img_j+1][0] << " " << image_pixel_grid->corner_sourcepts[img_i+1][img_j+1][1] << endl;
				//cout << "checking center: " << image_pixel_grid->center_sourcepts[img_i][img_j][0] << " " << image_pixel_grid->center_sourcepts[img_i][img_j][1] << endl;
				////die("OOPSY DOOPSIES!");
			//}
			twistpts_threads[thread] = &image_pixel_grid->twist_pts[img_i][img_j];
			twist_status_threads[thread] = &image_pixel_grid->twist_status[img_i][img_j];

			min_i = (int) (((*corners_threads[thread][0])[0] - srcgrid_xmin) / xstep);
			min_j = (int) (((*corners_threads[thread][0])[1] - srcgrid_ymin) / ystep);
			max_i = min_i;
			max_j = min_j;
			for (i=1; i < 4; i++) {
				corner_raytrace_i = (int) (((*corners_threads[thread][i])[0] - srcgrid_xmin) / xstep);
				corner_raytrace_j = (int) (((*corners_threads[thread][i])[1] - srcgrid_ymin) / ystep);
				if (corner_raytrace_i < min_i) min_i = corner_raytrace_i;
				if (corner_raytrace_i > max_i) max_i = corner_raytrace_i;
				if (corner_raytrace_j < min_j) min_j = corner_raytrace_j;
				if (corner_raytrace_j > max_j) max_j = corner_raytrace_j;
			}
			if ((min_i < 0) or (min_i >= u_N)) continue;
			if ((min_j < 0) or (min_j >= w_N)) continue;
			if ((max_i < 0) or (max_i >= u_N)) continue;
			if ((max_j < 0) or (max_j >= w_N)) continue;

			for (j=min_j; j <= max_j; j++) {
				for (i=min_i; i <= max_i; i++) {
					nsrc = j*u_N + i;
					if (cell[i][j]->check_if_in_neighborhood(corners_threads[thread],inside,thread)) {
						if (inside) {
							triangle1_overlap = cell[i][j]->find_triangle1_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread);
							triangle2_overlap = cell[i][j]->find_triangle2_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread);
							triangle1_weight = triangle1_overlap / image_pixel_grid->source_plane_triangle1_area[img_i][img_j];
							triangle2_weight = triangle2_overlap / image_pixel_grid->source_plane_triangle2_area[img_i][img_j];
							//cout << triangle1_overlap << " " << triangle2_overlap << " " << image_pixel_grid->source_plane_triangle1_area[img_i][img_j] << endl;
						} else {
							if (cell[i][j]->check_triangle1_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread)) {
								triangle1_overlap = cell[i][j]->find_triangle1_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread);
								triangle1_weight = triangle1_overlap / image_pixel_grid->source_plane_triangle1_area[img_i][img_j];
							} else {
								triangle1_overlap = 0;
								triangle1_weight = 0;
							}
							if (cell[i][j]->check_triangle2_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread)) {
								triangle2_overlap = cell[i][j]->find_triangle2_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread);
								triangle2_weight = triangle2_overlap / image_pixel_grid->source_plane_triangle2_area[img_i][img_j];
							} else {
								triangle2_overlap = 0;
								triangle2_weight = 0;
							}
						}
						/*
						if ((nsrc==2251) and ((triangle1_overlap != 0) or (triangle2_overlap != 0))) {
							double mag = image_pixel_grid->pixel_area / (image_pixel_grid->source_plane_triangle1_area[img_i][img_j] + image_pixel_grid->source_plane_triangle2_area[img_i][img_j]);
							wout << "# " << mag << " " << image_pixel_grid->pixel_mag[img_i][img_j] << endl;
							wout << image_pixel_grid->corner_sourcepts[img_i][img_j][0] << " " << image_pixel_grid->corner_sourcepts[img_i][img_j][1] << " " << image_pixel_grid->corner_pts[img_i][img_j][0] << " " << image_pixel_grid->corner_pts[img_i][img_j][1] << endl;
							wout << image_pixel_grid->corner_sourcepts[img_i][img_j+1][0] << " " << image_pixel_grid->corner_sourcepts[img_i][img_j+1][1] << " " << image_pixel_grid->corner_pts[img_i][img_j+1][0] << " " << image_pixel_grid->corner_pts[img_i][img_j+1][1] << endl;
							wout << image_pixel_grid->corner_sourcepts[img_i+1][img_j+1][0] << " " << image_pixel_grid->corner_sourcepts[img_i+1][img_j+1][1] << " " << image_pixel_grid->corner_pts[img_i+1][img_j+1][0] << " " << image_pixel_grid->corner_pts[img_i+1][img_j+1][1] << endl;
							wout << image_pixel_grid->corner_sourcepts[img_i+1][img_j][0] << " " << image_pixel_grid->corner_sourcepts[img_i+1][img_j][1] << " " << image_pixel_grid->corner_pts[img_i+1][img_j][0] << " " << image_pixel_grid->corner_pts[img_i+1][img_j][1] << endl;
							wout << image_pixel_grid->corner_sourcepts[img_i][img_j][0] << " " << image_pixel_grid->corner_sourcepts[img_i][img_j][1] << " " << image_pixel_grid->corner_pts[img_i][img_j][0] << " " << image_pixel_grid->corner_pts[img_i][img_j][1] << endl;
							wout << endl;
						}
						*/
						if ((triangle1_overlap != 0) or (triangle2_overlap != 0)) {
							weighted_overlap = triangle1_weight + triangle2_weight;
							//cout << "WEIGHT: " << weighted_overlap << endl;
							overlap_matrix_rows[n].push_back(weighted_overlap);
							overlap_matrix_index_rows[n].push_back(nsrc);
							overlap_matrix_row_nn[n]++;

							overlap_area = triangle1_overlap + triangle2_overlap;
							if ((image_pixel_grid->pixel_in_mask == NULL) or (!image_pixel_grid->pixel_in_mask[img_i][img_j])) overlap_area = 0;
							overlap_area_matrix_rows[n].push_back(overlap_area);
						}
					}
				}
			}
			overlap_matrix_nn_part += overlap_matrix_row_nn[n];
		}
	}

	overlap_matrix_nn = overlap_matrix_nn_part;

	double *overlap_matrix = new double[overlap_matrix_nn];
	int *overlap_matrix_index = new int[overlap_matrix_nn];
	int *image_pixel_location_overlap = new int[ntot_cells+1];
	double *overlap_area_matrix;
	overlap_area_matrix = new double[overlap_matrix_nn];

	image_pixel_location_overlap[0] = 0;
	int n,l;
	for (n=0; n < ntot_cells; n++) {
		image_pixel_location_overlap[n+1] = image_pixel_location_overlap[n] + overlap_matrix_row_nn[n];
	}

	int indx;
	for (n=0; n < ntot_cells; n++) {
		indx = image_pixel_location_overlap[n];
		for (j=0; j < overlap_matrix_row_nn[n]; j++) {
			overlap_matrix[indx+j] = overlap_matrix_rows[n][j];
			overlap_matrix_index[indx+j] = overlap_matrix_index_rows[n][j];
			overlap_area_matrix[indx+j] = overlap_area_matrix_rows[n][j];
		}
	}

	for (n=0; n < ntot_cells; n++) {
		//img_j = n / image_pixel_grid->x_N;
		//img_i = n % image_pixel_grid->x_N;
		if (use_emask) {
			img_j = image_pixel_grid->emask_pixels_j[n];
			img_i = image_pixel_grid->emask_pixels_i[n];
		} else {
			img_j = image_pixel_grid->mask_pixels_j[n];
			img_i = image_pixel_grid->mask_pixels_i[n];
		}
		for (l=image_pixel_location_overlap[n]; l < image_pixel_location_overlap[n+1]; l++) {
			nsrc = overlap_matrix_index[l];
			j = nsrc / u_N;
			i = nsrc % u_N;
			mag_matrix[nsrc] += overlap_matrix[l];
			area_matrix[nsrc] += overlap_area_matrix[l];
			if ((image_pixel_grid->pixel_in_mask != NULL) and (imgpixel_data->high_sn_pixel[img_i][img_j])) high_sn_area_matrix[nsrc] += overlap_area_matrix[l];
			cell[i][j]->overlap_pixel_n.push_back(n);
			if ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[img_i][img_j]==true)) cell[i][j]->maps_to_image_window = true;
		}
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for finding source cell magnifications: "  << wtime.count() << endl;
	}

	//ofstream nimgout("auxnimg.dat");
	//ofstream nimgout2("auxnimg2.dat");
	for (nsrc=0; nsrc < ntot_src; nsrc++) {
		j = nsrc / u_N;
		i = nsrc % u_N;
		cell[i][j]->total_magnification = mag_matrix[nsrc] * image_pixel_grid->triangle_area / cell[i][j]->cell_area;
		cell[i][j]->avg_image_pixels_mapped = cell[i][j]->total_magnification * cell[i][j]->cell_area / image_pixel_grid->pixel_area;
		if (qlens->n_image_prior) cell[i][j]->n_images = area_matrix[nsrc] / cell[i][j]->cell_area;
		//nimgout << cell[i][j]->center_pt[0] << " " << cell[i][j]->center_pt[1] << " " << cell[i][j]->n_images << endl;
		//nimgout2 << nsrc << " " << cell[i][j]->center_pt[0] << " " << cell[i][j]->center_pt[1] << " " << cell[i][j]->n_images << endl;

		if (area_matrix[nsrc] > cell[i][j]->cell_area) qlens->total_srcgrid_overlap_area += cell[i][j]->cell_area;
		else qlens->total_srcgrid_overlap_area += area_matrix[nsrc];
		if (image_pixel_grid->pixel_in_mask != NULL) {
			if (high_sn_area_matrix[nsrc] > cell[i][j]->cell_area) qlens->high_sn_srcgrid_overlap_area += cell[i][j]->cell_area;
			else qlens->high_sn_srcgrid_overlap_area += high_sn_area_matrix[nsrc];
		}
		if (cell[i][j]->total_magnification*0.0) warn("Nonsensical source cell magnification (mag=%g",cell[i][j]->total_magnification);
	}

	delete[] overlap_matrix;
	delete[] overlap_matrix_index;
	delete[] image_pixel_location_overlap;
	delete[] overlap_matrix_rows;
	delete[] overlap_matrix_index_rows;
	delete[] overlap_matrix_row_nn;
	delete[] mag_matrix;
	delete[] overlap_area_matrix;
	delete[] overlap_area_matrix_rows;
	delete[] area_matrix;
	delete[] high_sn_area_matrix;
}

double CartesianSourceGrid::get_lowest_mag_sourcept(double &xsrc, double &ysrc)
{
	double lowest_mag = 1e30;
	int i, j, i_lowest_mag, j_lowest_mag;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			if (cell[i][j]->maps_to_image_window) {
				if (cell[i][j]->total_magnification < lowest_mag) {
					lowest_mag = cell[i][j]->total_magnification;
					i_lowest_mag = i;
					j_lowest_mag = j;
				}
			}
		}
	}
	xsrc = cell[i_lowest_mag][j_lowest_mag]->center_pt[0];
	ysrc = cell[i_lowest_mag][j_lowest_mag]->center_pt[1];
	return lowest_mag;
}

void CartesianSourceGrid::get_highest_mag_sourcept(double &xsrc, double &ysrc)
{
	double highest_mag = -1e30;
	int i, j, i_highest_mag, j_highest_mag;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			if (cell[i][j]->maps_to_image_window) {
				if (cell[i][j]->total_magnification > highest_mag) {
					highest_mag = cell[i][j]->total_magnification;
					i_highest_mag = i;
					j_highest_mag = j;
				}
			}
		}
	}
	xsrc = cell[i_highest_mag][j_highest_mag]->center_pt[0];
	ysrc = cell[i_highest_mag][j_highest_mag]->center_pt[1];
}

void CartesianSourceGrid::adaptive_subgrid()
{
	calculate_pixel_magnifications();
	std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	int i, prev_levels;
	for (i=0; i < max_levels-1; i++) {
		prev_levels = levels;
		split_subcells_firstlevel(i);
		if (prev_levels==levels) break; // no splitting occurred, so no need to attempt further subgridding
	}
	assign_all_neighbors();

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for adaptive grid splitting: "  << wtime.count() << endl;
	}
}

void CartesianSourceGrid::split_subcells_firstlevel(const int splitlevel)
{
	ImgGrid_Params<PlainTypes>& imggrid_params = image_pixel_grid->assign_imggrid_param_object<PlainTypes>();
	if (level >= max_levels+1)
		die("maximum number of splittings has been reached (%i)", max_levels);

	int ntot = u_N*w_N;
	int i,j,n;
	if (splitlevel > level) {
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			maxlevs[thread] = parent_grid->levels;
			#pragma omp for private(i,j,n) schedule(dynamic)
			for (n=0; n < ntot; n++) {
				j = n / u_N;
				i = n % u_N;
				if (cell[i][j]->cell != NULL) cell[i][j]->split_subcells(splitlevel,thread);
			}
		}
		for (i=0; i < nthreads; i++) if (maxlevs[i] > parent_grid->levels) parent_grid->levels = maxlevs[i];
	} else {
		int k,l,m;
		double overlap_area, weighted_overlap, triangle1_overlap, triangle2_overlap, triangle1_weight, triangle2_weight;
		CartesianSourcePixel *subcell;
		bool subgrid;
		#pragma omp parallel
		{
			int nn, img_i, img_j;
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			maxlevs[thread] = parent_grid->levels;
			double xstep, ystep;
			xstep = (srcgrid_xmax-srcgrid_xmin)/u_N/2.0;
			ystep = (srcgrid_ymax-srcgrid_ymin)/w_N/2.0;
			int min_i,max_i,min_j,max_j;
			int corner_raytrace_i, corner_raytrace_j;
			int ii,lmin,lmax,mmin,mmax;

			#pragma omp for private(i,j,n,k,l,m,overlap_area,weighted_overlap,triangle1_overlap,triangle2_overlap,triangle1_weight,triangle2_weight,subgrid,subcell) schedule(dynamic)
			for (n=0; n < ntot; n++) {
				j = n / u_N;
				i = n % u_N;
				subgrid = false;
				if ((cell[i][j]->total_magnification*cell[i][j]->cell_area/(qlens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) > pixel_magnification_threshold) subgrid = true;
				if (subgrid) {
					//cout << "SPLITTING(FIRST): level=" << cell[i][j]->level << ", mag=" << cell[i][j]->total_magnification << " fac=" << (cell[i][j]->cell_area/(qlens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) << endl;
					cell[i][j]->split_cells(2,2,thread);
					for (k=0; k < cell[i][j]->overlap_pixel_n.size(); k++) {
						nn = cell[i][j]->overlap_pixel_n[k];
						//img_j = nn / image_pixel_grid->x_N;
						//img_i = nn % image_pixel_grid->x_N;
						img_j = image_pixel_grid->mask_pixels_j[nn];
						img_i = image_pixel_grid->mask_pixels_i[nn];

						corners_threads[thread][0] = &image_pixel_grid->corner_sourcepts[img_i][img_j];
						corners_threads[thread][1] = &image_pixel_grid->corner_sourcepts[img_i][img_j+1];
						corners_threads[thread][2] = &image_pixel_grid->corner_sourcepts[img_i+1][img_j];
						corners_threads[thread][3] = &image_pixel_grid->corner_sourcepts[img_i+1][img_j+1];
						twistpts_threads[thread] = &image_pixel_grid->twist_pts[img_i][img_j];
						twist_status_threads[thread] = &image_pixel_grid->twist_status[img_i][img_j];

						min_i = (int) (((*corners_threads[thread][0])[0] - cell[i][j]->corner_pt[0][0]) / xstep);
						min_j = (int) (((*corners_threads[thread][0])[1] - cell[i][j]->corner_pt[0][1]) / ystep);
						max_i = min_i;
						max_j = min_j;
						for (ii=1; ii < 4; ii++) {
							corner_raytrace_i = (int) (((*corners_threads[thread][ii])[0] - cell[i][j]->corner_pt[0][0]) / xstep);
							corner_raytrace_j = (int) (((*corners_threads[thread][ii])[1] - cell[i][j]->corner_pt[0][1]) / ystep);
							if (corner_raytrace_i < min_i) min_i = corner_raytrace_i;
							if (corner_raytrace_i > max_i) max_i = corner_raytrace_i;
							if (corner_raytrace_j < min_j) min_j = corner_raytrace_j;
							if (corner_raytrace_j > max_j) max_j = corner_raytrace_j;
						}
						lmin=0;
						lmax=cell[i][j]->u_N-1;
						mmin=0;
						mmax=cell[i][j]->w_N-1;
						if ((min_i >= 0) and (min_i < cell[i][j]->u_N)) lmin = min_i;
						if ((max_i >= 0) and (max_i < cell[i][j]->u_N)) lmax = max_i;
						if ((min_j >= 0) and (min_j < cell[i][j]->w_N)) mmin = min_j;
						if ((max_j >= 0) and (max_j < cell[i][j]->w_N)) mmax = max_j;

						for (l=lmin; l <= lmax; l++) {
							for (m=mmin; m <= mmax; m++) {
								subcell = cell[i][j]->cell[l][m];
								triangle1_overlap = subcell->find_triangle1_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread);
								triangle2_overlap = subcell->find_triangle2_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread);
								triangle1_weight = triangle1_overlap / image_pixel_grid->source_plane_triangle1_area[img_i][img_j];
								triangle2_weight = triangle2_overlap / image_pixel_grid->source_plane_triangle2_area[img_i][img_j];
								weighted_overlap = triangle1_weight + triangle2_weight;
								if ((triangle2_weight*0.0 != 0.0)) {
									cout << "HMM (" << img_i << "," << img_j << ") " << triangle2_overlap << " " << image_pixel_grid->source_plane_triangle2_area[img_i][img_j] << endl;
									cout << "    .... imgpixel: " << image_pixel_grid->center_pts[img_i][img_j][0] << " " << image_pixel_grid->center_pts[img_i][img_j][1] << endl;
								}

								subcell->total_magnification += weighted_overlap;
								//cout << "MAG: " << triangle1_overlap << " " << triangle2_overlap << " " << image_pixel_grid->source_plane_triangle1_area[img_i][img_j] << " " << image_pixel_grid->source_plane_triangle2_area[img_i][img_j] << endl;
								//cout << "MAG: " << subcell->total_magnification << " " << image_pixel_grid->triangle_area << " " << subcell->cell_area << endl;
								if ((weighted_overlap != 0) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[img_i][img_j]==true))) subcell->maps_to_image_window = true;
								subcell->overlap_pixel_n.push_back(nn);
								if (qlens->n_image_prior) {
									overlap_area = triangle1_overlap + triangle2_overlap;
									subcell->n_images += overlap_area;
								}
							}
						}
						if (subcell->total_magnification*0.0 != 0.0) die("Nonsensical subcell magnification");
					}
					for (l=0; l < cell[i][j]->u_N; l++) {
						for (m=0; m < cell[i][j]->w_N; m++) {
							subcell = cell[i][j]->cell[l][m];
							subcell->total_magnification *= image_pixel_grid->triangle_area / subcell->cell_area;
							//cout << "subcell mag: " << subcell->total_magnification << endl;
							subcell->avg_image_pixels_mapped = subcell->total_magnification * subcell->cell_area / image_pixel_grid->pixel_area;
							if (qlens->n_image_prior) subcell->n_images /= subcell->cell_area;
						}
					}
				}
			}
		}
		for (i=0; i < nthreads; i++) if (maxlevs[i] > parent_grid->levels) parent_grid->levels = maxlevs[i];
	}
}

void CartesianSourcePixel::split_subcells(const int splitlevel, const int thread)
{
	ImgGrid_Params<PlainTypes>& imggrid_params = image_pixel_grid->assign_imggrid_param_object<PlainTypes>();
	if (level >= max_levels+1)
		die("maximum number of splittings has been reached (%i)", max_levels);

	int i,j;
	if (splitlevel > level) {
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				if (cell[i][j]->cell != NULL) cell[i][j]->split_subcells(splitlevel,thread);
			}
		}
	} else {
		double xstep, ystep;
		xstep = (corner_pt[2][0] - corner_pt[0][0])/u_N/2.0;
		ystep = (corner_pt[1][1] - corner_pt[0][1])/w_N/2.0;
		int min_i,max_i,min_j,max_j;
		int corner_raytrace_i, corner_raytrace_j;
		int ii,lmin,lmax,mmin,mmax;

		int k,l,m,nn,img_i,img_j;
		double overlap_area, weighted_overlap, triangle1_overlap, triangle2_overlap, triangle1_weight, triangle2_weight;
		CartesianSourcePixel *subcell;
		bool subgrid;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				subgrid = false;
				if ((cell[i][j]->total_magnification*cell[i][j]->cell_area/(lens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) > parent_grid->pixel_magnification_threshold) subgrid = true;

				if (subgrid) {
					//cout << "SPLITTING: level=" << cell[i][j]->level << ", mag=" << cell[i][j]->total_magnification << " fac=" << (cell[i][j]->cell_area/(lens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) << endl;
					cell[i][j]->split_cells(2,2,thread);
					for (k=0; k < cell[i][j]->overlap_pixel_n.size(); k++) {
						nn = cell[i][j]->overlap_pixel_n[k];
						//img_j = nn / image_pixel_grid->x_N;
						//img_i = nn % image_pixel_grid->x_N;
						img_j = image_pixel_grid->mask_pixels_j[nn];
						img_i = image_pixel_grid->mask_pixels_i[nn];

						corners_threads[thread][0] = &image_pixel_grid->corner_sourcepts[img_i][img_j];
						corners_threads[thread][1] = &image_pixel_grid->corner_sourcepts[img_i][img_j+1];
						corners_threads[thread][2] = &image_pixel_grid->corner_sourcepts[img_i+1][img_j];
						corners_threads[thread][3] = &image_pixel_grid->corner_sourcepts[img_i+1][img_j+1];
						twistpts_threads[thread] = &image_pixel_grid->twist_pts[img_i][img_j];
						twist_status_threads[thread] = &image_pixel_grid->twist_status[img_i][img_j];

						min_i = (int) (((*corners_threads[thread][0])[0] - cell[i][j]->corner_pt[0][0]) / xstep);
						min_j = (int) (((*corners_threads[thread][0])[1] - cell[i][j]->corner_pt[0][1]) / ystep);
						max_i = min_i;
						max_j = min_j;
						for (ii=1; ii < 4; ii++) {
							corner_raytrace_i = (int) (((*corners_threads[thread][ii])[0] - cell[i][j]->corner_pt[0][0]) / xstep);
							corner_raytrace_j = (int) (((*corners_threads[thread][ii])[1] - cell[i][j]->corner_pt[0][1]) / ystep);
							if (corner_raytrace_i < min_i) min_i = corner_raytrace_i;
							if (corner_raytrace_i > max_i) max_i = corner_raytrace_i;
							if (corner_raytrace_j < min_j) min_j = corner_raytrace_j;
							if (corner_raytrace_j > max_j) max_j = corner_raytrace_j;
						}
						lmin=0;
						lmax=cell[i][j]->u_N-1;
						mmin=0;
						mmax=cell[i][j]->w_N-1;
						if ((min_i >= 0) and (min_i < cell[i][j]->u_N)) lmin = min_i;
						if ((max_i >= 0) and (max_i < cell[i][j]->u_N)) lmax = max_i;
						if ((min_j >= 0) and (min_j < cell[i][j]->w_N)) mmin = min_j;
						if ((max_j >= 0) and (max_j < cell[i][j]->w_N)) mmax = max_j;

						for (l=lmin; l <= lmax; l++) {
							for (m=mmin; m <= mmax; m++) {
								subcell = cell[i][j]->cell[l][m];
								triangle1_overlap = subcell->find_triangle1_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread);
								triangle2_overlap = subcell->find_triangle2_overlap(corners_threads[thread],twistpts_threads[thread],*twist_status_threads[thread],thread);
								triangle1_weight = triangle1_overlap / image_pixel_grid->source_plane_triangle1_area[img_i][img_j];
								triangle2_weight = triangle2_overlap / image_pixel_grid->source_plane_triangle2_area[img_i][img_j];
								weighted_overlap = triangle1_weight + triangle2_weight;

								subcell->total_magnification += weighted_overlap;
								subcell->overlap_pixel_n.push_back(nn);
								if ((weighted_overlap != 0) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[img_i][img_j]==true))) subcell->maps_to_image_window = true;
								if (lens->n_image_prior) {
									overlap_area = triangle1_overlap + triangle2_overlap;
									subcell->n_images += overlap_area;
								}
							}
						}
					}
					for (l=0; l < cell[i][j]->u_N; l++) {
						for (m=0; m < cell[i][j]->w_N; m++) {
							subcell = cell[i][j]->cell[l][m];
							subcell->total_magnification *= image_pixel_grid->triangle_area / subcell->cell_area;
							//cout << "subcell mag: " << subcell->total_magnification << endl;
							subcell->avg_image_pixels_mapped = subcell->total_magnification * subcell->cell_area / image_pixel_grid->pixel_area;
							if (lens->n_image_prior) subcell->n_images /= subcell->cell_area;
						}
					}
				}
			}
		}
	}
}

bool CartesianSourceGrid::assign_source_mapping_flags_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, const int& thread)
{
	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_overlap(input_corner_pts,thread)==false) return false;

	bool image_pixel_maps_to_source_grid = false;
	bool inside;
	int i,j;
	for (j=jmin[thread]; j <= jmax[thread]; j++) {
		for (i=imin[thread]; i <= imax[thread]; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->subcell_assign_source_mapping_flags_overlap(input_corner_pts,twist_pt,twist_status,mapped_cartesian_srcpixels,thread,image_pixel_maps_to_source_grid);
			else {
				if (!cell[i][j]->check_if_in_neighborhood(input_corner_pts,inside,thread)) continue;
				if ((inside) or (cell[i][j]->check_overlap(input_corner_pts,twist_pt,twist_status,thread))) {
					cell[i][j]->maps_to_image_pixel = true;
					mapped_cartesian_srcpixels.push_back(cell[i][j]);
					//if ((image_pixel_i==41) and (image_pixel_j==11)) cout << "mapped cell: " << cell[i][j]->center_pt[0] << " " << cell[i][j]->center_pt[1] << endl;
					if (!image_pixel_maps_to_source_grid) image_pixel_maps_to_source_grid = true;
				}
			}
		}
	}
	return image_pixel_maps_to_source_grid;
}

void CartesianSourcePixel::subcell_assign_source_mapping_flags_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, const int& thread, bool& image_pixel_maps_to_source_grid)
{
	bool inside;
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->subcell_assign_source_mapping_flags_overlap(input_corner_pts,twist_pt,twist_status,mapped_cartesian_srcpixels,thread,image_pixel_maps_to_source_grid);
			else {
				if (!cell[i][j]->check_if_in_neighborhood(input_corner_pts,inside,thread)) continue;
				if ((inside) or (cell[i][j]->check_overlap(input_corner_pts,twist_pt,twist_status,thread))) {
					cell[i][j]->maps_to_image_pixel = true;
					mapped_cartesian_srcpixels.push_back(cell[i][j]);
					if (!image_pixel_maps_to_source_grid) image_pixel_maps_to_source_grid = true;
				}
			}
		}
	}
}

void CartesianSourceGrid::calculate_Lmatrix_overlap(const int &img_index, const int image_pixel_i, const int image_pixel_j, int& index, lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread)
{
	if (image_pixel_grid==NULL) warn("cartesian source pixels cannot access image pixel grid; cannot calculate Lmatrix");
	double overlap, total_overlap=0;
	int i,j,k;
	int Lmatrix_index_initial = index;
	CartesianSourcePixel *subcell;

	for (i=0; i < image_pixel_grid->mapped_cartesian_srcpixels[image_pixel_i][image_pixel_j].size(); i++) {
		subcell = image_pixel_grid->mapped_cartesian_srcpixels[image_pixel_i][image_pixel_j][i];
		image_pixel_grid->Lmatrix_index_rows[img_index].push_back(subcell->active_index);
		overlap = subcell->find_rectangle_overlap(input_corner_pts,twist_pt,twist_status,thread,image_pixel_i,image_pixel_j);
		image_pixel_grid->Lmatrix_rows[img_index].push_back(overlap);
		index++;
		total_overlap += overlap;
	}

	if (total_overlap==0) die("image pixel should have mapped to at least one source pixel");
	for (i=Lmatrix_index_initial; i < index; i++)
		image_pixel_grid->Lmatrix_rows[img_index][i] /= total_overlap;
}

double CartesianSourceGrid::find_lensed_surface_brightness_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread)
{
	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_overlap(input_corner_pts,thread)==false) return false;

	double total_overlap = 0;
	double total_weighted_surface_brightness = 0;
	double overlap;
	int i,j;
	for (j=jmin[thread]; j <= jmax[thread]; j++) {
		for (i=imin[thread]; i <= imax[thread]; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->find_lensed_surface_brightness_subcell_overlap(input_corner_pts,twist_pt,twist_status,thread,overlap,total_overlap,total_weighted_surface_brightness);
			else {
				overlap = cell[i][j]->find_rectangle_overlap(input_corner_pts,twist_pt,twist_status,thread,0,0);
				total_overlap += overlap;
				total_weighted_surface_brightness += overlap*cell[i][j]->surface_brightness;
			}
		}
	}
	double lensed_surface_brightness;
	if (total_overlap==0) lensed_surface_brightness = 0;
	else lensed_surface_brightness = total_weighted_surface_brightness/total_overlap;
	return lensed_surface_brightness;
}

void CartesianSourcePixel::find_lensed_surface_brightness_subcell_overlap(lensvector<double> **input_corner_pts, lensvector<double> *twist_pt, int& twist_status, const int& thread, double& overlap, double& total_overlap, double& total_weighted_surface_brightness)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->find_lensed_surface_brightness_subcell_overlap(input_corner_pts,twist_pt,twist_status,thread,overlap,total_overlap,total_weighted_surface_brightness);
			else {
				overlap = cell[i][j]->find_rectangle_overlap(input_corner_pts,twist_pt,twist_status,thread,0,0);
				total_overlap += overlap;
				total_weighted_surface_brightness += overlap*cell[i][j]->surface_brightness;
			}
		}
	}
}

bool CartesianSourceGrid::bisection_search_interpolate(lensvector<double> &input_center_pt, const int& thread)
{
	int i, imid, jmid;
	bool inside;
	bool inside_corner[4];
	int n_inside;
	double xmin[4], xmax[4], ymin[4], ymax[4];

	for (;;) {
		n_inside=0;
		for (i=0; i < 4; i++) inside_corner[i] = false;
		imid = (imax[thread] + imin[thread])/2;
		jmid = (jmax[thread] + jmin[thread])/2;
		xmin[0] = cell[imin[thread]][jmin[thread]]->corner_pt[0][0];
		ymin[0] = cell[imin[thread]][jmin[thread]]->corner_pt[0][1];
		xmax[0] = cell[imid][jmid]->corner_pt[3][0];
		ymax[0] = cell[imid][jmid]->corner_pt[3][1];

		xmin[1] = cell[imin[thread]][jmid+1]->corner_pt[0][0];
		ymin[1] = cell[imin[thread]][jmid+1]->corner_pt[0][1];
		xmax[1] = cell[imid][jmax[thread]]->corner_pt[3][0];
		ymax[1] = cell[imid][jmax[thread]]->corner_pt[3][1];

		xmin[2] = cell[imid+1][jmin[thread]]->corner_pt[0][0];
		ymin[2] = cell[imid+1][jmin[thread]]->corner_pt[0][1];
		xmax[2] = cell[imax[thread]][jmid]->corner_pt[3][0];
		ymax[2] = cell[imax[thread]][jmid]->corner_pt[3][1];

		xmin[3] = cell[imid+1][jmid+1]->corner_pt[0][0];
		ymin[3] = cell[imid+1][jmid+1]->corner_pt[0][1];
		xmax[3] = cell[imax[thread]][jmax[thread]]->corner_pt[3][0];
		ymax[3] = cell[imax[thread]][jmax[thread]]->corner_pt[3][1];

		for (i=0; i < 4; i++) {
			if ((input_center_pt[0] >= xmin[i]) and (input_center_pt[0] < xmax[i]) and (input_center_pt[1] >= ymin[i]) and (input_center_pt[1] < ymax[i])) {
				inside_corner[i] = true;
				n_inside++;
			}
		}
		if (n_inside==0) return false;
		if (n_inside > 1) die("should not be inside more than one rectangle");
		else {
			if (inside_corner[0]) { imax[thread]=imid; jmax[thread]=jmid; }
			else if (inside_corner[1]) { imax[thread]=imid; jmin[thread]=jmid; }
			else if (inside_corner[2]) { imin[thread]=imid; jmax[thread]=jmid; }
			else if (inside_corner[3]) { imin[thread]=imid; jmin[thread]=jmid; }
		}
		if ((imax[thread] - imin[thread] <= 1) or (jmax[thread] - jmin[thread] <= 1)) break;
	}
	return true;
}

bool CartesianSourceGrid::assign_source_mapping_flags_interpolate(lensvector<double> &input_center_pt, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, const int& thread, const int& image_pixel_i, const int& image_pixel_j)
{
	bool image_pixel_maps_to_source_grid = false;
	// when splitting image pixels, there could be multiple entries in the Lmatrix array that belong to the same source pixel; you might save computational time if these can be consolidated (by adding them together). Try this out later
	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_interpolate(input_center_pt,thread)==true) {
		int i,j,side;
		CartesianSourcePixel* cellptr;
		int oldsize = mapped_cartesian_srcpixels.size();
		for (j=jmin[thread]; j <= jmax[thread]; j++) {
			for (i=imin[thread]; i <= imax[thread]; i++) {
				if ((input_center_pt[0] >= cell[i][j]->corner_pt[0][0]) and (input_center_pt[0] < cell[i][j]->corner_pt[2][0]) and (input_center_pt[1] >= cell[i][j]->corner_pt[0][1]) and (input_center_pt[1] < cell[i][j]->corner_pt[3][1])) {
					if (cell[i][j]->cell != NULL) image_pixel_maps_to_source_grid = cell[i][j]->subcell_assign_source_mapping_flags_interpolate(input_center_pt,mapped_cartesian_srcpixels,thread);
					else {
						cell[i][j]->maps_to_image_pixel = true;
						mapped_cartesian_srcpixels.push_back(cell[i][j]);
						if (!image_pixel_maps_to_source_grid) image_pixel_maps_to_source_grid = true;
						if (((input_center_pt[0] > cell[i][j]->center_pt[0]) and (cell[i][j]->neighbor[0] != NULL)) or (cell[i][j]->neighbor[1] == NULL)) {
							if (cell[i][j]->neighbor[0]->cell != NULL) {
								side=0;
								cellptr = cell[i][j]->neighbor[0]->find_nearest_neighbor_cell(input_center_pt,side);
								cellptr->maps_to_image_pixel = true;
								mapped_cartesian_srcpixels.push_back(cellptr);
								//cout << "Adding to maps " << image_pixel_i << " " << image_pixel_j << endl;
							}
							else {
								cell[i][j]->neighbor[0]->maps_to_image_pixel = true;
								mapped_cartesian_srcpixels.push_back(cell[i][j]->neighbor[0]);
								//cout << "Adding to maps " << image_pixel_i << " " << image_pixel_j << endl;
							}
						} else {
							if (cell[i][j]->neighbor[1]->cell != NULL) {
								side=1;
								cellptr = cell[i][j]->neighbor[1]->find_nearest_neighbor_cell(input_center_pt,side);
								cellptr->maps_to_image_pixel = true;
								mapped_cartesian_srcpixels.push_back(cellptr);
								//cout << "Adding to maps " << image_pixel_i << " " << image_pixel_j << endl;
							}
							else {
								cell[i][j]->neighbor[1]->maps_to_image_pixel = true;
								mapped_cartesian_srcpixels.push_back(cell[i][j]->neighbor[1]);
								//cout << "Adding to maps " << image_pixel_i << " " << image_pixel_j << endl;
							}
						}
						if (((input_center_pt[1] > cell[i][j]->center_pt[1]) and (cell[i][j]->neighbor[2] != NULL)) or (cell[i][j]->neighbor[3] == NULL)) {
							if (cell[i][j]->neighbor[2]->cell != NULL) {
								side=2;
								cellptr = cell[i][j]->neighbor[2]->find_nearest_neighbor_cell(input_center_pt,side);
								cellptr->maps_to_image_pixel = true;
								mapped_cartesian_srcpixels.push_back(cellptr);
								//cout << "Adding to maps " << image_pixel_i << " " << image_pixel_j << endl;
							}
							else {
								cell[i][j]->neighbor[2]->maps_to_image_pixel = true;
								mapped_cartesian_srcpixels.push_back(cell[i][j]->neighbor[2]);
								//cout << "Adding to maps " << image_pixel_i << " " << image_pixel_j << endl;
							}
						} else {
							if (cell[i][j]->neighbor[3]->cell != NULL) {
								side=3;
								cellptr = cell[i][j]->neighbor[3]->find_nearest_neighbor_cell(input_center_pt,side);
								cellptr->maps_to_image_pixel = true;
								mapped_cartesian_srcpixels.push_back(cellptr);
								//cout << "Adding to maps " << image_pixel_i << " " << image_pixel_j << endl;
							}
							else {
								cell[i][j]->neighbor[3]->maps_to_image_pixel = true;
								mapped_cartesian_srcpixels.push_back(cell[i][j]->neighbor[3]);
								//cout << "Adding to maps " << image_pixel_i << " " << image_pixel_j << endl;
							}
						}
					}
					break;
				}
			}
		}
		if ((mapped_cartesian_srcpixels.size() - oldsize) != 3) die("Did not assign enough interpolation cells!");
	} else {
		mapped_cartesian_srcpixels.push_back(NULL);
		mapped_cartesian_srcpixels.push_back(NULL);
		mapped_cartesian_srcpixels.push_back(NULL);
	}
	//if ((image_pixel_i==34) and (image_pixel_j==1)) {
		//if (!image_pixel_maps_to_source_grid) cout << "subpixel didn't map!!!" << endl;
		//cout << "SIZE: " << mapped_cartesian_srcpixels.size() << endl;
		//for (int i=0; i < mapped_cartesian_srcpixels.size(); i++) cout << "cell " << i << ": " << mapped_cartesian_srcpixels[i]->active_index << endl;
	//}
	return image_pixel_maps_to_source_grid;
}

bool CartesianSourcePixel::subcell_assign_source_mapping_flags_interpolate(lensvector<double> &input_center_pt, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, const int& thread)
{
	bool image_pixel_maps_to_source_grid = false;
	int i,j,side;
	CartesianSourcePixel* cellptr;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if ((input_center_pt[0] >= cell[i][j]->corner_pt[0][0]) and (input_center_pt[0] < cell[i][j]->corner_pt[2][0]) and (input_center_pt[1] >= cell[i][j]->corner_pt[0][1]) and (input_center_pt[1] < cell[i][j]->corner_pt[3][1])) {
				if (cell[i][j]->cell != NULL) image_pixel_maps_to_source_grid = cell[i][j]->subcell_assign_source_mapping_flags_interpolate(input_center_pt,mapped_cartesian_srcpixels,thread);
				else {
					cell[i][j]->maps_to_image_pixel = true;
					mapped_cartesian_srcpixels.push_back(cell[i][j]);
					if (!image_pixel_maps_to_source_grid) image_pixel_maps_to_source_grid = true;
					if (((input_center_pt[0] > cell[i][j]->center_pt[0]) and (cell[i][j]->neighbor[0] != NULL)) or (cell[i][j]->neighbor[1] == NULL)) {
						if (cell[i][j]->neighbor[0]->cell != NULL) {
							side=0;
							cellptr = cell[i][j]->neighbor[0]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_cartesian_srcpixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[0]->maps_to_image_pixel = true;
							mapped_cartesian_srcpixels.push_back(cell[i][j]->neighbor[0]);
						}
					} else {
						if (cell[i][j]->neighbor[1]->cell != NULL) {
							side=1;
							cellptr = cell[i][j]->neighbor[1]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_cartesian_srcpixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[1]->maps_to_image_pixel = true;
							mapped_cartesian_srcpixels.push_back(cell[i][j]->neighbor[1]);
						}
					}
					if (((input_center_pt[1] > cell[i][j]->center_pt[1]) and (cell[i][j]->neighbor[2] != NULL)) or (cell[i][j]->neighbor[3] == NULL)) {
						if (cell[i][j]->neighbor[2]->cell != NULL) {
							side=2;
							cellptr = cell[i][j]->neighbor[2]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_cartesian_srcpixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[2]->maps_to_image_pixel = true;
							mapped_cartesian_srcpixels.push_back(cell[i][j]->neighbor[2]);
						}
					} else {
						if (cell[i][j]->neighbor[3]->cell != NULL) {
							side=3;
							cellptr = cell[i][j]->neighbor[3]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_cartesian_srcpixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[3]->maps_to_image_pixel = true;
							mapped_cartesian_srcpixels.push_back(cell[i][j]->neighbor[3]);
						}
					}
				}
				break;
			}
		}
	}
	return image_pixel_maps_to_source_grid;
}

void CartesianSourceGrid::calculate_Lmatrix_interpolate(const int img_index, vector<CartesianSourcePixel*>& mapped_cartesian_srcpixels, int& index, lensvector<double> &input_center_pt, const int& ii, const double weight, const int& thread)
{
	if (image_pixel_grid==NULL) warn("cartesian source pixels cannot access image pixel grid; cannot calculate Lmatrix");
	for (int i=0; i < 3; i++) {
		//cout << "What " << i << endl;
		//cout << "ii=" << ii << " trying index " << (3*ii+i) << endl;
		//cout << "imgpix: " << image_pixel_i << " " << image_pixel_j << endl;
		//cout << "size: " << image_pixel_grid->mapped_cartesian_srcpixels[image_pixel_i][image_pixel_j].size() << endl;
		if (mapped_cartesian_srcpixels[3*ii+i] == NULL) return; // in this case, subpixel does not map to anything
		image_pixel_grid->Lmatrix_index_rows[img_index].push_back(mapped_cartesian_srcpixels[3*ii+i]->active_index);
		//cout << "What? " << i << endl;
		interpolation_pts[i][thread] = &mapped_cartesian_srcpixels[3*ii+i]->center_pt;
	}

	//if (qlens->interpolate_sb_3pt) {
		double d = ((*interpolation_pts[0][thread])[0]-(*interpolation_pts[1][thread])[0])*((*interpolation_pts[1][thread])[1]-(*interpolation_pts[2][thread])[1]) - ((*interpolation_pts[1][thread])[0]-(*interpolation_pts[2][thread])[0])*((*interpolation_pts[0][thread])[1]-(*interpolation_pts[1][thread])[1]);
		image_pixel_grid->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[1][thread])[1]-(*interpolation_pts[2][thread])[1]) + input_center_pt[1]*((*interpolation_pts[2][thread])[0]-(*interpolation_pts[1][thread])[0]) + (*interpolation_pts[1][thread])[0]*(*interpolation_pts[2][thread])[1] - (*interpolation_pts[1][thread])[1]*(*interpolation_pts[2][thread])[0])/d);
		image_pixel_grid->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[2][thread])[1]-(*interpolation_pts[0][thread])[1]) + input_center_pt[1]*((*interpolation_pts[0][thread])[0]-(*interpolation_pts[2][thread])[0]) + (*interpolation_pts[0][thread])[1]*(*interpolation_pts[2][thread])[0] - (*interpolation_pts[0][thread])[0]*(*interpolation_pts[2][thread])[1])/d);
		image_pixel_grid->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[0][thread])[1]-(*interpolation_pts[1][thread])[1]) + input_center_pt[1]*((*interpolation_pts[1][thread])[0]-(*interpolation_pts[0][thread])[0]) + (*interpolation_pts[0][thread])[0]*(*interpolation_pts[1][thread])[1] - (*interpolation_pts[0][thread])[1]*(*interpolation_pts[1][thread])[0])/d);
		if (d==0) warn("d is zero!!!");
	//} else {
		//image_pixel_grid->Lmatrix_rows[img_index].push_back(weight);
		//image_pixel_grid->Lmatrix_rows[img_index].push_back(0);
		//image_pixel_grid->Lmatrix_rows[img_index].push_back(0);
	//}

	index += 3;
}

double CartesianSourceGrid::find_lensed_surface_brightness_interpolate(lensvector<double> &input_center_pt, const int& thread)
{
	lensvector<double> *pts[3];
	double *sb[3];
	int indx=0;
	nearest_interpolation_cells[thread].found_containing_cell = false;
	for (int i=0; i < 3; i++) nearest_interpolation_cells[thread].pixel[i] = NULL;

	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_interpolate(input_center_pt,thread)==false) return false;

	bool image_pixel_maps_to_source_grid = false;
	int i,j,side;
	for (j=jmin[thread]; j <= jmax[thread]; j++) {
		for (i=imin[thread]; i <= imax[thread]; i++) {
			if ((input_center_pt[0] >= cell[i][j]->corner_pt[0][0]) and (input_center_pt[0] < cell[i][j]->corner_pt[2][0]) and (input_center_pt[1] >= cell[i][j]->corner_pt[0][1]) and (input_center_pt[1] < cell[i][j]->corner_pt[3][1])) {
				if (cell[i][j]->cell != NULL) cell[i][j]->find_interpolation_cells(input_center_pt,thread);
				else {
					nearest_interpolation_cells[thread].found_containing_cell = true;
					nearest_interpolation_cells[thread].pixel[0] = cell[i][j];
					if (((input_center_pt[0] > cell[i][j]->center_pt[0]) and (cell[i][j]->neighbor[0] != NULL)) or (cell[i][j]->neighbor[1] == NULL)) {
						if (cell[i][j]->neighbor[0]->cell != NULL) {
							side=0;
							nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[0]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[0];
					} else {
						if (cell[i][j]->neighbor[1]->cell != NULL) {
							side=1;
							nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[1]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[1];
					}
					if (((input_center_pt[1] > cell[i][j]->center_pt[1]) and (cell[i][j]->neighbor[2] != NULL)) or (cell[i][j]->neighbor[3] == NULL)) {
						if (cell[i][j]->neighbor[2]->cell != NULL) {
							side=2;
							nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[2]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[2];
					} else {
						if (cell[i][j]->neighbor[3]->cell != NULL) {
							side=3;
							nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[3]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[3];
					}
				}
				break;
			}
		}
	}

	for (i=0; i < 3; i++) {
		pts[i] = &nearest_interpolation_cells[thread].pixel[i]->center_pt;
		sb[i] = &nearest_interpolation_cells[thread].pixel[i]->surface_brightness;
	}

	if (nearest_interpolation_cells[thread].found_containing_cell==false) die("could not find containing cell");
	double d, total_sb = 0;
	d = ((*pts[0])[0]-(*pts[1])[0])*((*pts[1])[1]-(*pts[2])[1]) - ((*pts[1])[0]-(*pts[2])[0])*((*pts[0])[1]-(*pts[1])[1]);
	total_sb += (*sb[0])*(input_center_pt[0]*((*pts[1])[1]-(*pts[2])[1]) + input_center_pt[1]*((*pts[2])[0]-(*pts[1])[0]) + (*pts[1])[0]*(*pts[2])[1] - (*pts[1])[1]*(*pts[2])[0]);
	total_sb += (*sb[1])*(input_center_pt[0]*((*pts[2])[1]-(*pts[0])[1]) + input_center_pt[1]*((*pts[0])[0]-(*pts[2])[0]) + (*pts[0])[1]*(*pts[2])[0] - (*pts[0])[0]*(*pts[2])[1]);
	total_sb += (*sb[2])*(input_center_pt[0]*((*pts[0])[1]-(*pts[1])[1]) + input_center_pt[1]*((*pts[1])[0]-(*pts[0])[0]) + (*pts[0])[0]*(*pts[1])[1] - (*pts[0])[1]*(*pts[1])[0]);
	total_sb /= d;
	return total_sb;
}

void CartesianSourcePixel::find_interpolation_cells(lensvector<double> &input_center_pt, const int& thread)
{
	int i,j,side;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if ((input_center_pt[0] >= cell[i][j]->corner_pt[0][0]) and (input_center_pt[0] < cell[i][j]->corner_pt[2][0]) and (input_center_pt[1] >= cell[i][j]->corner_pt[0][1]) and (input_center_pt[1] < cell[i][j]->corner_pt[3][1])) {
				if (cell[i][j]->cell != NULL) cell[i][j]->find_interpolation_cells(input_center_pt,thread);
				else {
					nearest_interpolation_cells[thread].found_containing_cell = true;
					nearest_interpolation_cells[thread].pixel[0] = cell[i][j];
					if (((input_center_pt[0] > cell[i][j]->center_pt[0]) and (cell[i][j]->neighbor[0] != NULL)) or (cell[i][j]->neighbor[1] == NULL)) {
						if (cell[i][j]->neighbor[0]->cell != NULL) {
							side=0;
							nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[0]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[0];
					} else {
						if (cell[i][j]->neighbor[1]->cell != NULL) {
							side=1;
							nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[1]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[1];
					}
					if (((input_center_pt[1] > cell[i][j]->center_pt[1]) and (cell[i][j]->neighbor[2] != NULL)) or (cell[i][j]->neighbor[3] == NULL)) {
						if (cell[i][j]->neighbor[2]->cell != NULL) {
							side=2;
							nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[2]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[2];
					} else {
						if (cell[i][j]->neighbor[3]->cell != NULL) {
							side=3;
							nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[3]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[3];
					}
				}
				break;
			}
		}
	}
}

CartesianSourcePixel* CartesianSourcePixel::find_nearest_neighbor_cell(lensvector<double> &input_center_pt, const int& side)
{
	int i,ncells;
	CartesianSourcePixel **cells;
	if ((side==0) or (side==1)) ncells = w_N;
	else if ((side==2) or (side==3)) ncells = u_N;
	else die("side number cannot be larger than 3");
	cells = new CartesianSourcePixel*[ncells];

	for (i=0; i < ncells; i++) {
		if (side==0) {
			if (cell[0][i]->cell != NULL) cells[i] = cell[0][i]->find_nearest_neighbor_cell(input_center_pt,side);
			else cells[i] = cell[0][i];
		} else if (side==1) {
			if (cell[u_N-1][i]->cell != NULL) cells[i] = cell[u_N-1][i]->find_nearest_neighbor_cell(input_center_pt,side);
			else cells[i] = cell[u_N-1][i];
		} else if (side==2) {
			if (cell[i][0]->cell != NULL) cells[i] = cell[i][0]->find_nearest_neighbor_cell(input_center_pt,side);
			else cells[i] = cell[i][0];
		} else if (side==3) {
			if (cell[i][w_N-1]->cell != NULL) cells[i] = cell[i][w_N-1]->find_nearest_neighbor_cell(input_center_pt,side);
			else cells[i] = cell[i][w_N-1];
		}
	}
	double sqr_distance, min_sqr_distance = 1e30;
	int i_min;
	for (i=0; i < ncells; i++) {
		sqr_distance = SQR(cells[i]->center_pt[0] - input_center_pt[0]) + SQR(cells[i]->center_pt[1] - input_center_pt[1]);
		if (sqr_distance < min_sqr_distance) {
			min_sqr_distance = sqr_distance;
			i_min = i;
		}
	}
	CartesianSourcePixel *closest_cell = cells[i_min];
	delete[] cells;
	return closest_cell;
}

CartesianSourcePixel* CartesianSourcePixel::find_nearest_neighbor_cell(lensvector<double> &input_center_pt, const int& side, const int tiebreaker_side)
{
	int i,ncells;
	CartesianSourcePixel **cells;
	if ((side==0) or (side==1)) ncells = w_N;
	else if ((side==2) or (side==3)) ncells = u_N;
	else die("side number cannot be larger than 3");
	cells = new CartesianSourcePixel*[ncells];
	double sqr_distance, min_sqr_distance = 1e30;
	CartesianSourcePixel *closest_cell = NULL;
	int it=0, side_try=side;

	while ((closest_cell==NULL) and (it++ < 2))
	{
		for (i=0; i < ncells; i++) {
			if (side_try==0) {
				if (cell[0][i]->cell != NULL) cells[i] = cell[0][i]->find_nearest_neighbor_cell(input_center_pt,side);
				else cells[i] = cell[0][i];
			} else if (side_try==1) {
				if (cell[u_N-1][i]->cell != NULL) cells[i] = cell[u_N-1][i]->find_nearest_neighbor_cell(input_center_pt,side);
				else cells[i] = cell[u_N-1][i];
			} else if (side_try==2) {
				if (cell[i][0]->cell != NULL) cells[i] = cell[i][0]->find_nearest_neighbor_cell(input_center_pt,side);
				else cells[i] = cell[i][0];
			} else if (side_try==3) {
				if (cell[i][w_N-1]->cell != NULL) cells[i] = cell[i][w_N-1]->find_nearest_neighbor_cell(input_center_pt,side);
				else cells[i] = cell[i][w_N-1];
			}
		}
		for (i=0; i < ncells; i++) {
			sqr_distance = SQR(cells[i]->center_pt[0] - input_center_pt[0]) + SQR(cells[i]->center_pt[1] - input_center_pt[1]);
			if ((sqr_distance < min_sqr_distance) or ((sqr_distance==min_sqr_distance) and (i==tiebreaker_side))) {
				min_sqr_distance = sqr_distance;
				closest_cell = cells[i];
			}
		}
		if (closest_cell==NULL) {
			// in this case neither of the subcells in question mapped to the image plane, so we had better try again with the other two subcells.
			if (side_try==0) side_try = 1;
			else if (side_try==1) side_try = 0;
			else if (side_try==2) side_try = 3;
			else if (side_try==3) side_try = 2;
		}
	}
	delete[] cells;
	return closest_cell;
}

void CartesianSourcePixel::find_nearest_two_cells(CartesianSourcePixel* &cellptr1, CartesianSourcePixel* &cellptr2, const int& side)
{
	if ((u_N != 2) or (w_N != 2)) die("cannot find nearest two cells unless splitting is two in either direction");
	if (side==0) {
		if (cell[0][0]->cell == NULL) cellptr1 = cell[0][0];
		else cellptr1 = cell[0][0]->find_corner_cell(0,1);
		if (cell[0][1]->cell == NULL) cellptr2 = cell[0][1];
		else cellptr2 = cell[0][1]->find_corner_cell(0,0);
	} else if (side==1) {
		if (cell[1][0]->cell == NULL) cellptr1 = cell[1][0];
		else cellptr1 = cell[1][0]->find_corner_cell(1,1);
		if (cell[1][1]->cell == NULL) cellptr2 = cell[1][1];
		else cellptr2 = cell[1][1]->find_corner_cell(1,0);
	} else if (side==2) {
		if (cell[0][0]->cell == NULL) cellptr1 = cell[0][0];
		else cellptr1 = cell[0][0]->find_corner_cell(1,0);
		if (cell[1][0]->cell == NULL) cellptr2 = cell[1][0];
		else cellptr2 = cell[1][0]->find_corner_cell(0,0);
	} else if (side==3) {
		if (cell[0][1]->cell == NULL) cellptr1 = cell[0][1];
		else cellptr1 = cell[0][1]->find_corner_cell(1,1);
		if (cell[1][1]->cell == NULL) cellptr2 = cell[1][1];
		else cellptr2 = cell[1][1]->find_corner_cell(0,1);
	}
}

CartesianSourcePixel* CartesianSourcePixel::find_corner_cell(const int i, const int j)
{
	CartesianSourcePixel* cellptr = cell[i][j];
	while (cellptr->cell != NULL)
		cellptr = cellptr->cell[i][j];
	return cellptr;
}

double CartesianSourceGrid::find_local_inverse_magnification_interpolate(lensvector<double> &input_center_pt, const int& thread)
{
	lensvector<double> *pts[3];
	double *mag[3];
	int indx=0;
	nearest_interpolation_cells[thread].found_containing_cell = false;
	for (int i=0; i < 3; i++) nearest_interpolation_cells[thread].pixel[i] = NULL;

	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_interpolate(input_center_pt,thread)==false) return false;

	bool image_pixel_maps_to_source_grid = false;
	int i,j,side;
	for (j=jmin[thread]; j <= jmax[thread]; j++) {
		for (i=imin[thread]; i <= imax[thread]; i++) {
			if ((input_center_pt[0] >= cell[i][j]->corner_pt[0][0]) and (input_center_pt[0] < cell[i][j]->corner_pt[2][0]) and (input_center_pt[1] >= cell[i][j]->corner_pt[0][1]) and (input_center_pt[1] < cell[i][j]->corner_pt[3][1])) {
				if (cell[i][j]->cell != NULL) cell[i][j]->find_interpolation_cells(input_center_pt,thread);
				else {
					nearest_interpolation_cells[thread].found_containing_cell = true;
					nearest_interpolation_cells[thread].pixel[0] = cell[i][j];
					if (((input_center_pt[0] > cell[i][j]->center_pt[0]) and (cell[i][j]->neighbor[0] != NULL)) or (cell[i][j]->neighbor[1] == NULL)) {
						if (cell[i][j]->neighbor[0]->cell != NULL) {
							side=0;
							nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[0]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[0];
					} else {
						if (cell[i][j]->neighbor[1]->cell != NULL) {
							side=1;
							nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[1]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[1] = cell[i][j]->neighbor[1];
					}
					if (((input_center_pt[1] > cell[i][j]->center_pt[1]) and (cell[i][j]->neighbor[2] != NULL)) or (cell[i][j]->neighbor[3] == NULL)) {
						if (cell[i][j]->neighbor[2]->cell != NULL) {
							side=2;
							nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[2]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[2];
					} else {
						if (cell[i][j]->neighbor[3]->cell != NULL) {
							side=3;
							nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[3]->find_nearest_neighbor_cell(input_center_pt,side);
						}
						else nearest_interpolation_cells[thread].pixel[2] = cell[i][j]->neighbor[3];
					}
				}
				break;
			}
		}
	}

	double total_invmag = 0;
	//double lev;
	//cout << "Interpolating pixels for point " << input_center_pt[0] << " " << input_center_pt[1] << endl;
	int missing_mags = 0;
	int missing_mag_i = -1;
	for (i=0; i < 3; i++) {
		pts[i] = &nearest_interpolation_cells[thread].pixel[i]->center_pt;
		mag[i] = &nearest_interpolation_cells[thread].pixel[i]->total_magnification;
		if (*mag[i]==0) {
			// missing magnifiations sometimes occur for a border pixel
			//cout << "UH-OH! zero mag found at " << (*pts[i])[0] << ", " << (*pts[i])[1] << endl;
			missing_mags++;
			missing_mag_i = i;
		}
		if (missing_mags==1) {
			// this only occurs near the boundaries of the source pixel grid
			if (missing_mag_i==0) total_invmag = 1.0/dmax((*mag[1]),(*mag[2]));
			else if (missing_mag_i==1) total_invmag = 1.0/dmax((*mag[0]),(*mag[2]));
			else total_invmag = 1.0/dmax((*mag[0]),(*mag[1]));
			//cout << " GOT HERE" << endl;
			return total_invmag;
		}
		if (missing_mags==2) {
			for (i=0; i < 3; i++) {
				if (*mag[i] != 0) return 1.0/(*mag[i]);
			}
		}
		if (missing_mags==3) die("none of the nearby source pixels have nonzero magnifications");

		//lev = nearest_interpolation_cells[thread].pixel[i]->level;
		//cout << (*pts[i])[0] << " " << (*pts[i])[1] << endl;
		//cout << "LEVEL for pixel " << i << ": " << lev << endl;
	}

	if (nearest_interpolation_cells[thread].found_containing_cell==false) die("could not find containing cell");
	double d;
	// we interpolate in the inverse magnification since this is less likely to blow up
	//cout << 1.0/(*mag[0]) << " " << 1.0/(*mag[1]) << " " << 1.0/(*mag[2]) << endl;
	d = ((*pts[0])[0]-(*pts[1])[0])*((*pts[1])[1]-(*pts[2])[1]) - ((*pts[1])[0]-(*pts[2])[0])*((*pts[0])[1]-(*pts[1])[1]);
	total_invmag += (1.0/(*mag[0]))*(input_center_pt[0]*((*pts[1])[1]-(*pts[2])[1]) + input_center_pt[1]*((*pts[2])[0]-(*pts[1])[0]) + (*pts[1])[0]*(*pts[2])[1] - (*pts[1])[1]*(*pts[2])[0]);
	total_invmag += (1.0/(*mag[1]))*(input_center_pt[0]*((*pts[2])[1]-(*pts[0])[1]) + input_center_pt[1]*((*pts[0])[0]-(*pts[2])[0]) + (*pts[0])[1]*(*pts[2])[0] - (*pts[0])[0]*(*pts[2])[1]);
	total_invmag += (1.0/(*mag[2]))*(input_center_pt[0]*((*pts[0])[1]-(*pts[1])[1]) + input_center_pt[1]*((*pts[1])[0]-(*pts[0])[0]) + (*pts[0])[0]*(*pts[1])[1] - (*pts[0])[1]*(*pts[1])[0]);
	total_invmag /= d;
	return total_invmag;
}

void CartesianSourcePixel::generate_gmatrices()
{
	int i,j,k,l;
	CartesianSourcePixel *cellptr1, *cellptr2;
	double alpha, beta, dxfac;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->generate_gmatrices();
			else {
				//dxfac = pow(1.3,-(cell[i][j]->level)); // seems like there's no real sensible reason to have a scaling factor here; delete this later
				dxfac = 1.0;
				for (k=0; k < 4; k++) {
					image_pixel_grid->gmatrix_rows[k][cell[i][j]->active_index].push_back(1.0/dxfac);
					image_pixel_grid->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cell[i][j]->active_index);
					image_pixel_grid->gmatrix_row_nn[k][cell[i][j]->active_index]++;
					image_pixel_grid->gmatrix_nn[k]++;
					if (cell[i][j]->neighbor[k]) {
						if (cell[i][j]->neighbor[k]->cell != NULL) {
							cell[i][j]->neighbor[k]->find_nearest_two_cells(cellptr1,cellptr2,k);
							if ((cellptr1==NULL) or (cellptr2==NULL)) die("Hmm, not getting back two cells");
							if (k < 2) {
								// interpolating surface brightness along x-direction
								alpha = abs((cell[i][j]->center_pt[1] - cellptr1->center_pt[1]) / (cellptr2->center_pt[1] - cellptr1->center_pt[1]));
							} else {
								// interpolating surface brightness along y-direction
								alpha = abs((cell[i][j]->center_pt[0] - cellptr1->center_pt[0]) / (cellptr2->center_pt[0] - cellptr1->center_pt[0]));
							}
							beta = 1-alpha;
							image_pixel_grid->gmatrix_rows[k][cell[i][j]->active_index].push_back(-beta/dxfac);
							image_pixel_grid->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr1->active_index);
							image_pixel_grid->gmatrix_row_nn[k][cell[i][j]->active_index]++;
							image_pixel_grid->gmatrix_nn[k]++;
							image_pixel_grid->gmatrix_rows[k][cell[i][j]->active_index].push_back(-alpha/dxfac);
							image_pixel_grid->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr2->active_index);
							image_pixel_grid->gmatrix_row_nn[k][cell[i][j]->active_index]++;
							image_pixel_grid->gmatrix_nn[k]++;
						}
						if (cell[i][j]->neighbor[k]->level==cell[i][j]->level) {
							image_pixel_grid->gmatrix_rows[k][cell[i][j]->active_index].push_back(-1.0/dxfac);
							image_pixel_grid->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cell[i][j]->neighbor[k]->active_index);
							image_pixel_grid->gmatrix_row_nn[k][cell[i][j]->active_index]++;
							image_pixel_grid->gmatrix_nn[k]++;
						} else {
							cellptr1 = cell[i][j]->neighbor[k];
							if (k < 2) {
								if (cellptr1->center_pt[1] > cell[i][j]->center_pt[1]) l=3;
								else l=2;
							} else {
								if (cellptr1->center_pt[0] > cell[i][j]->center_pt[0]) l=1;
								else l=0;
							}
							if (cellptr1->neighbor[l]->cell==NULL) cellptr2 = cellptr1->neighbor[l];
							else cellptr2 = cellptr1->neighbor[l]->find_nearest_neighbor_cell(cellptr1->center_pt,l,k%2); // the tiebreaker k%2 ensures that preference goes to cells that are closer to this cell in order to interpolate to find the gradient
							if (cellptr2==NULL) die("Subcell does not map to source pixel; regularization currently cannot handle unmapped subcells");
							if (k < 2) alpha = abs((cell[i][j]->center_pt[1] - cellptr1->center_pt[1]) / (cellptr2->center_pt[1] - cellptr1->center_pt[1]));
							else alpha = abs((cell[i][j]->center_pt[0] - cellptr1->center_pt[0]) / (cellptr2->center_pt[0] - cellptr1->center_pt[0]));
							beta = 1-alpha;
							image_pixel_grid->gmatrix_rows[k][cell[i][j]->active_index].push_back(-beta/dxfac);
							image_pixel_grid->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr1->active_index);
							image_pixel_grid->gmatrix_row_nn[k][cell[i][j]->active_index]++;
							image_pixel_grid->gmatrix_nn[k]++;
							image_pixel_grid->gmatrix_rows[k][cell[i][j]->active_index].push_back(-alpha/dxfac);
							image_pixel_grid->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr2->active_index);
							image_pixel_grid->gmatrix_row_nn[k][cell[i][j]->active_index]++;
							image_pixel_grid->gmatrix_nn[k]++;
						}
					}
				}
			}
		}
	}
}

void CartesianSourcePixel::generate_hmatrices()
{
	int i,j,k,l,m,kmin,kmax;
	CartesianSourcePixel *cellptr1, *cellptr2;
	double alpha, beta;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->generate_hmatrices();
			else {
				for (l=0; l < 2; l++) {
					image_pixel_grid->hmatrix_rows[l][cell[i][j]->active_index].push_back(-2);
					image_pixel_grid->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cell[i][j]->active_index);
					image_pixel_grid->hmatrix_row_nn[l][cell[i][j]->active_index]++;
					image_pixel_grid->hmatrix_nn[l]++;
					if (l==0) {
						kmin=0; kmax=1;
					} else {
						kmin=2; kmax=3;
					}
					for (k=kmin; k <= kmax; k++) {
						if (cell[i][j]->neighbor[k]) {
							if (cell[i][j]->neighbor[k]->cell != NULL) {
								cell[i][j]->neighbor[k]->find_nearest_two_cells(cellptr1,cellptr2,k);
								if ((cellptr1==NULL) or (cellptr2==NULL)) die("Hmm, not getting back two cells");
								if (k < 2) {
									// interpolating surface brightness along x-direction
									alpha = abs((cell[i][j]->center_pt[1] - cellptr1->center_pt[1]) / (cellptr2->center_pt[1] - cellptr1->center_pt[1]));
								} else {
									// interpolating surface brightness along y-direction
									alpha = abs((cell[i][j]->center_pt[0] - cellptr1->center_pt[0]) / (cellptr2->center_pt[0] - cellptr1->center_pt[0]));
								}
								beta = 1-alpha;
								image_pixel_grid->hmatrix_rows[l][cell[i][j]->active_index].push_back(beta);
								image_pixel_grid->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr1->active_index);
								image_pixel_grid->hmatrix_row_nn[l][cell[i][j]->active_index]++;
								image_pixel_grid->hmatrix_nn[l]++;
								image_pixel_grid->hmatrix_rows[l][cell[i][j]->active_index].push_back(alpha);
								image_pixel_grid->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr2->active_index);
								image_pixel_grid->hmatrix_row_nn[l][cell[i][j]->active_index]++;
								image_pixel_grid->hmatrix_nn[l]++;
							}
							else {
								if (cell[i][j]->neighbor[k]->level==cell[i][j]->level) {
									image_pixel_grid->hmatrix_rows[l][cell[i][j]->active_index].push_back(1);
									image_pixel_grid->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cell[i][j]->neighbor[k]->active_index);
									image_pixel_grid->hmatrix_row_nn[l][cell[i][j]->active_index]++;
									image_pixel_grid->hmatrix_nn[l]++;
								} else {
									cellptr1 = cell[i][j]->neighbor[k];
									if (k < 2) {
										if (cellptr1->center_pt[1] > cell[i][j]->center_pt[1]) m=3;
										else m=2;
									} else {
										if (cellptr1->center_pt[0] > cell[i][j]->center_pt[0]) m=1;
										else m=0;
									}
									if (cellptr1->neighbor[m]->cell==NULL) cellptr2 = cellptr1->neighbor[m];
									else cellptr2 = cellptr1->neighbor[m]->find_nearest_neighbor_cell(cellptr1->center_pt,m,k%2); // the tiebreaker k%2 ensures that preference goes to cells that are closer to this cell in order to interpolate to find the curvature
									if (cellptr2==NULL) die("Subcell does not map to source pixel; regularization currently cannot handle unmapped subcells");
									if (k < 2) alpha = abs((cell[i][j]->center_pt[1] - cellptr1->center_pt[1]) / (cellptr2->center_pt[1] - cellptr1->center_pt[1]));
									else alpha = abs((cell[i][j]->center_pt[0] - cellptr1->center_pt[0]) / (cellptr2->center_pt[0] - cellptr1->center_pt[0]));
									beta = 1-alpha;
									//cout << alpha << " " << beta << " " << k << " " << m << " " << ii << " " << jj << " " << i << " " << j << endl;
									//cout << cell[i][j]->center_pt[0] << " " << cellptr1->center_pt[0] << " " << cellptr1->center_pt[1] << " " << cellptr2->center_pt[0] << " " << cellptr2->center_pt[1] << endl;
									image_pixel_grid->hmatrix_rows[l][cell[i][j]->active_index].push_back(beta);
									image_pixel_grid->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr1->active_index);
									image_pixel_grid->hmatrix_row_nn[l][cell[i][j]->active_index]++;
									image_pixel_grid->hmatrix_nn[l]++;
									image_pixel_grid->hmatrix_rows[l][cell[i][j]->active_index].push_back(alpha);
									image_pixel_grid->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr2->active_index);
									image_pixel_grid->hmatrix_row_nn[l][cell[i][j]->active_index]++;
									image_pixel_grid->hmatrix_nn[l]++;
								}
							}
						}
					}
				}
			}
		}
	}
}

int CartesianSourceGrid::assign_indices_and_count_levels()
{
	levels=1; // we are going to recount the number of levels
	int source_pixel_i=0;
	assign_indices(source_pixel_i);
	return source_pixel_i;
}

void CartesianSourcePixel::assign_indices(int& source_pixel_i)
{
	if (parent_grid->levels < level+1) parent_grid->levels=level+1;
	int i, j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->assign_indices(source_pixel_i);
			else {
				cell[i][j]->index = source_pixel_i++;
			}
		}
	}
}

/*
void CartesianSourcePixel::print_indices()
{
	int i, j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->print_indices();
			else {
				parent_grid->index_out << cell[i][j]->index << " " << cell[i][j]->active_index << " level=" << cell[i][j]->level << endl;
			}
		}
	}
}
*/

int CartesianSourceGrid::assign_active_indices_and_count_source_pixels(const int source_pixel_i_initial, const bool activate_unmapped_pixels, const bool exclude_pixels_outside_window)
{
	int source_pixel_i=source_pixel_i_initial;
	parent_grid->activate_unmapped_source_pixels = activate_unmapped_pixels;
	parent_grid->exclude_source_pixels_outside_fit_window = exclude_pixels_outside_window;
	assign_active_indices(source_pixel_i);
	n_active_pixels = source_pixel_i-source_pixel_i_initial;
	return source_pixel_i;
}

void CartesianSourcePixel::assign_active_indices(int& source_pixel_i)
{
	int i, j;
	bool unsplit_cell = false;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->assign_active_indices(source_pixel_i);
			else {
					cell[i][j]->active_index = source_pixel_i++;
					cell[i][j]->active_pixel = true;
				if (!cell[i][j]->maps_to_image_pixel) {
					if ((lens->mpi_id==0) and (lens->regularization_method == 0)) warn(lens->warnings,"A source pixel does not map to any image pixel (for source pixel %i,%i), level %i, center (%g,%g)",i,j,cell[i][j]->level,cell[i][j]->center_pt[0],cell[i][j]->center_pt[1]); // only show warning if no regularization being used, since matrix cannot be inverted in that case
				}
			}
		}
	}
	if (unsplit_cell) unsplit();
}

CartesianSourcePixel::~CartesianSourcePixel()
{
	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) delete cell[i][j];
			delete[] cell[i];
		}
		delete[] cell;
		cell = NULL;
	}
}

CartesianSourceGrid::~CartesianSourceGrid()
{
	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) delete cell[i][j];
			delete[] cell[i];
		}
		delete[] cell;
		cell = NULL;
	}
}

void CartesianSourcePixel::clear()
{
	if (cell == NULL) return;

	int i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) delete cell[i][j];
		delete[] cell[i];
	}
	delete[] cell;
	cell = NULL;
	u_N=1; w_N=1;
}

void CartesianSourcePixel::clear_subgrids()
{
	if (level>0) {
		if (cell == NULL) return;
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				delete cell[i][j];
			}
			delete[] cell[i];
		}
		delete[] cell;
		cell = NULL;
		parent_grid->number_of_pixels -= (u_N*w_N - 1);
		u_N=1; w_N=1;
	} else {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				if (cell[i][j]->cell != NULL) cell[i][j]->clear_subgrids();
			}
		}
	}
}

/********************************************* Functions in class DelaunayGrid ***********************************************/


DelaunayGrid::DelaunayGrid()
{
	delaunay_params = &delaunaygrid_params;
#ifdef USE_STAN
	delaunay_params_dif = &delaunaygrid_params_dif;
#endif

	//allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)
	n_gridpts = 0;
	delaunaygrid_params.triangle = NULL;
	delaunaygrid_params.gridpts = NULL;
#ifdef USE_STAN
	delaunaygrid_params_dif.triangle = NULL;
	delaunaygrid_params_dif.gridpts = NULL;
#endif
}

#ifdef USE_STAN
void DelaunayGrid::sync_delaunaygrid_autodif_parameters()
{
	/*
	// I don't think syncing gridpts and triangle will be necessary?
	if (delaunaygrid_params.gridpts != NULL) {
		for (int i=0; i < n_gridpts; i++) {
			delaunaygrid_params_dif.gridpts[0] = delaunaygrid_params.gridpts[0];
			delaunaygrid_params_dif.gridpts[1] = delaunaygrid_params.gridpts[1];
		}
	}
	if (delaunaygrid_params.triangle != NULL) {
		for (int i=0; i < n_triangles; i++) {
			delaunaygrid_params_dif.triangle[i] = delaunaygrid_params.triangle[i];
		}
	}
	*/
	delaunaygrid_params_dif.kernel_correlation_length = delaunaygrid_params.kernel_correlation_length;
	delaunaygrid_params_dif.matern_index = delaunaygrid_params.matern_index;
}
#endif

template <typename QScalar>
void DelaunayGrid::create_pixel_grid(QScalar* gridpts_x, QScalar* gridpts_y, const int n_gridpts_in)
{
	DelaunayGrid_Params<QScalar>& p = assign_delaunay_param_object<QScalar>();
	if (p.gridpts != NULL) delete_grid_arrays();

	n_gridpts = n_gridpts_in;
	if (p.gridpts != NULL) die("girdpts wasn't NULL");
	p.gridpts = new lensvector<QScalar>[n_gridpts];
	for (int i=0; i < 4; i++) adj_triangles[i] = new int[n_gridpts];
	int n;
	for (n=0; n < n_gridpts; n++) {
		p.gridpts[n][0] = gridpts_x[n];
		p.gridpts[n][1] = gridpts_y[n];
		adj_triangles[0][n] = -1; // +x direction
		adj_triangles[1][n] = -1; // -x direction
		adj_triangles[2][n] = -1; // +y direction
		adj_triangles[3][n] = -1; // -y direction
	}

	vector<int>* shared_triangles_unsorted = new vector<int>[n_gridpts];
	n_shared_triangles = new int[n_gridpts];
	shared_triangles = new int*[n_gridpts];
	voronoi_boundary_x = new double*[n_gridpts];
	voronoi_boundary_y = new double*[n_gridpts];
	voronoi_area = new double[n_gridpts];
	voronoi_length = new double[n_gridpts];

	Delaunay<QScalar> *delaunay_triangles = new Delaunay(gridpts_x, gridpts_y, n_gridpts);
	delaunay_triangles->Process();

	n_triangles = delaunay_triangles->TriNum();
	if (n_triangles==0) die("number of Delaunay triangles is zero; cannot construct Delaunay grid");
	p.triangle = new Triangle<QScalar>[n_triangles];
	delaunay_triangles->store_triangles(p.triangle);
#ifdef USE_STAN
	if constexpr (stan::is_autodiff_v<QScalar>) {
		delaunay_params->gridpts = new lensvector<double>[n_gridpts];
		delaunay_params->triangle = new Triangle<double>[n_triangles];
		for (int i=0; i < n_gridpts; i++) {
			delaunay_params->gridpts[i][0] = stan::math::value_of(p.gridpts[i][0]);
			delaunay_params->gridpts[i][1] = stan::math::value_of(p.gridpts[i][1]);
		}
		for (int i=0; i < n_triangles; i++) {
			p.triangle[i].copy_triangle(&delaunay_params->triangle[i]);
		}
	}
#endif
	avg_area = 0;
	
	for (n=0; n < n_triangles; n++) {
		shared_triangles_unsorted[delaunay_params->triangle[n].vertex_index[0]].push_back(n);
		shared_triangles_unsorted[delaunay_params->triangle[n].vertex_index[1]].push_back(n);
		shared_triangles_unsorted[delaunay_params->triangle[n].vertex_index[2]].push_back(n);
		avg_area += delaunay_params->triangle[n].area;
	}

	avg_area /= n_triangles;
	double avg_tri_length = sqrt(avg_area);
	int n_boundary_pts;
	Triangle<double> *triptr;
	int i;
	lensvector<double> midpoint;
	lensvector<double> vec1,vec2;
	for (n=0; n < n_gridpts; n++) {
		n_boundary_pts = shared_triangles_unsorted[n].size();
		// NOTE: for extreme configurations, occasionally a point gets excluded from the Delaunay triangulation; not sure why this happens,
		//       but as long as the grid is regularized, it shouldn't cause qlens to crash.
		if (n_boundary_pts==0) {
			warn("Point was excluded from Delaunay triangulation (n=%i) located at (%g,%g)",n,delaunay_params->gridpts[n][0],delaunay_params->gridpts[n][1]);
			n_shared_triangles[n] = 0;
			voronoi_boundary_x[n] = NULL;
			voronoi_boundary_y[n] = NULL;
			shared_triangles[n] = NULL;
			voronoi_length[n] = 3*avg_tri_length; // just to avoid possible numerical issues; this is really only a problem for border pixels
			voronoi_area[n] = SQR(voronoi_length[n]);
			continue;
		}
		voronoi_boundary_x[n] = new double[n_boundary_pts];
		voronoi_boundary_y[n] = new double[n_boundary_pts];
		shared_triangles[n] = new int[n_boundary_pts];
		double *angles = new double[n_boundary_pts];
		double *midpt_angles = new double[n_boundary_pts];
		for (i=0; i < n_boundary_pts; i++) {
			shared_triangles[n][i] = shared_triangles_unsorted[n][i];
			triptr = &delaunay_params->triangle[shared_triangles_unsorted[n][i]];
			voronoi_boundary_x[n][i] = triptr->circumcenter[0];
			voronoi_boundary_y[n][i] = triptr->circumcenter[1];
			double comp1,comp2,angle;
			comp1 = voronoi_boundary_x[n][i] - delaunay_params->gridpts[n][0];
			comp2 = voronoi_boundary_y[n][i] - delaunay_params->gridpts[n][1];
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
			while (angle >= M_2PI) angle -= M_2PI;
			while (angle < 0) angle += M_2PI;
			angles[i] = angle;

			midpoint = 0.33333333333333*(triptr->vertex[0] + triptr->vertex[1] + triptr->vertex[2]);
			comp1 = midpoint[0] - delaunay_params->gridpts[n][0];
			comp2 = midpoint[1] - delaunay_params->gridpts[n][1];
			if (comp1==0) {
				if (comp2 > 0) angle = M_HALFPI;
				else if (comp2==0) angle = 0.0;
				else angle = -M_HALFPI;
			} else {
				angle = atan2(comp2,comp1);
			}
			while (angle >= M_2PI) angle -= M_2PI;
			while (angle < 0) angle += M_2PI;
			midpt_angles[i] = angle;
		}
		n_shared_triangles[n] = n_boundary_pts;
		//sort(n_boundary_pts,angles,voronoi_boundary_x[n],voronoi_boundary_y[n],shared_triangles[n]); // I don't think sorting by circumcenters will work well, because circumcenters may lie outside the triangles and orders might get reversed 
		sort(n_boundary_pts,angles,voronoi_boundary_x[n],voronoi_boundary_y[n]);
		sort(n_boundary_pts,midpt_angles,shared_triangles[n]);
		delete[] angles;
		delete[] midpt_angles;
		vec2[0] = voronoi_boundary_x[n][0] - delaunay_params->gridpts[n][0];
		vec2[1] = voronoi_boundary_y[n][0] - delaunay_params->gridpts[n][1];
		voronoi_area[n] = 0;
		for (i=0; i < n_boundary_pts-1; i++) {
			vec1 = vec2;
			vec2[0] = voronoi_boundary_x[n][i+1] - delaunay_params->gridpts[n][0];
			vec2[1] = voronoi_boundary_y[n][i+1] - delaunay_params->gridpts[n][1];
			voronoi_area[n] += abs((vec1[0]*vec2[1] - vec1[1]*vec2[0])/2);
		}
		voronoi_length[n] = sqrt(voronoi_area[n]);
		if (voronoi_length[n] > 100*avg_tri_length) {
			//warn("CRAZY long voronoi length! (length=%g) setting voronoi length to 3*average value to avoid numerical issues",voronoi_length[n]);
			voronoi_length[n] = 3*avg_tri_length; // just to avoid possible numerical issues; this is really only a problem for border pixels
		}
	}

	delete[] shared_triangles_unsorted;
	delete delaunay_triangles;
}
template void DelaunayGrid::create_pixel_grid<double>(double* gridpts_x, double* gridpts_y, const int n_gridpts_in);
#ifdef USE_STAN
template void DelaunayGrid::create_pixel_grid<stan::math::var>(stan::math::var* gridpts_x, stan::math::var* gridpts_y, const int n_gridpts_in);
#endif

void DelaunayGrid::record_adjacent_triangles_xy()
{
	const double increment = 1e-5;
	int i,j;
	bool foundxp, foundyp, foundxm, foundym;
	double x, y, xp, xm, yp, ym;
	lensvector<double> ptx_p, pty_p, ptx_m, pty_m;
	for (i=0; i < n_gridpts; i++) {
		foundxp = false;
		foundxm = false;
		foundyp = false;
		foundym = false;
		x = delaunay_params->gridpts[i][0];
		y = delaunay_params->gridpts[i][1];
		xp = x + increment;
		xm = x - increment;
		yp = y + increment;
		ym = y - increment;
		ptx_p.input(xp,y);
		ptx_m.input(xm,y);
		pty_p.input(x,yp);
		pty_m.input(x,ym);

		for (j=0; j < n_shared_triangles[i]; j++) {
			if ((!foundxp) and (test_if_inside(shared_triangles[i][j],ptx_p)==true)) {
				adj_triangles[0][i] = shared_triangles[i][j];
				foundxp = true;
			}
			if ((!foundxm) and (test_if_inside(shared_triangles[i][j],ptx_m)==true)) {
				adj_triangles[1][i] = shared_triangles[i][j];
				foundxm = true;
			}
			if ((!foundyp) and (test_if_inside(shared_triangles[i][j],pty_p)==true)) {
				adj_triangles[2][i] = shared_triangles[i][j];
				foundyp = true;
			}
			if ((!foundym) and (test_if_inside(shared_triangles[i][j],pty_m)==true)) {
				adj_triangles[3][i] = shared_triangles[i][j];
				foundym = true;
			}
		}
		//if (!foundxp) warn("could not find triangle in +x direction for vertex %i",i);
		//if (!foundxm) warn("could not find triangle in -x direction for vertex %i",i);
		//if (!foundyp) warn("could not find triangle in +y direction for vertex %i",i);
		//if (!foundym) warn("could not find triangle in -y direction for vertex %i",i);
	}
}

int DelaunayGrid::search_grid(int initial_srcpixel, const lensvector<double>& pt, bool& inside_triangle)
{
	DelaunayGrid_Params<double>& p = assign_delaunay_param_object<double>();
	if (n_shared_triangles[initial_srcpixel]==0) {
		if (++initial_srcpixel == n_gridpts) initial_srcpixel = 0;
		//die("something is really wrong! This vertex doesn't share any triangle sides (vertex %i, ntot=%i)",initial_srcpixel,n_gridpts);
	}
	int n, triangle_num = shared_triangles[initial_srcpixel][0]; // there might be a better way to discern which shared triangle to start with, but we can optimize this later
	if ((pt[0]==p.gridpts[initial_srcpixel][0]) and (pt[1]==p.gridpts[initial_srcpixel][1])) {
		inside_triangle = true;
		return triangle_num;
	}

	inside_triangle = false;
	for (n=0; n < n_triangles; n++) {
		// NOTE: bear in mind that if the point is outside the grid, there are multiple border triangles that might accept the point as being on "their" side. This is why having a good starting triangle is valuable
		if (test_if_inside(triangle_num,pt,inside_triangle)==true) break; // note, will return 'true' if the point is outside the grid but closest to that triangle (compared to neighbors); 'inside_triangle' flag reveals if it's actually inside the triangle or not
	}

	/*
	bool ins1 = inside_triangle;
	inside_triangle = false;
	int triangle_num2 = shared_triangles[0][0];
	for (n=0; n < n_triangles; n++) {
		if (test_if_inside(triangle_num2,pt,inside_triangle)==true) break; // note, will return 'true' if the point is outside the grid but closest to that triangle; 'inside_triangle' flag reveals if it's actually inside the triangle or not
	}
	bool ins2 = inside_triangle;
	if (triangle_num != triangle_num2) {
		cout << "OH SHIT: ti=" << triangle_num << " ti2=" << triangle_num2 << " n=" << n << " ntri=" << n_triangles << " ins1=" << ins1 << " ins2=" << ins2 << endl;
		bool in1 = test_if_inside(triangle_num,pt,inside_triangle);
		bool in2 = test_if_inside(triangle_num2,pt,inside_triangle);
		cout << "found1=" << in1 << " found2=" << in2 << endl;
		cout << "pt=" << pt[0] << "," << pt[1] << endl;
		cout << "TRI0:" << endl;
		cout << triangle[triangle_num].vertex[0][0] << " " << triangle[triangle_num].vertex[0][1] << endl;
		cout << triangle[triangle_num].vertex[1][0] << " " << triangle[triangle_num].vertex[1][1] << endl;
		cout << triangle[triangle_num].vertex[2][0] << " " << triangle[triangle_num].vertex[2][1] << endl;
		cout << "TRI1:" << endl;
		cout << triangle[triangle_num2].vertex[0][0] << " " << triangle[triangle_num2].vertex[0][1] << endl;
		cout << triangle[triangle_num2].vertex[1][0] << " " << triangle[triangle_num2].vertex[1][1] << endl;
		cout << triangle[triangle_num2].vertex[2][0] << " " << triangle[triangle_num2].vertex[2][1] << endl;
		cout << endl;
	}
	*/

	if (n > n_triangles) die("searched all triangles (or else searched in a loop), still did not find triangle enclosing point--this shouldn't happen! pt=(%g,%g), pixel0=%i",pt[0],pt[1],initial_srcpixel);
	return triangle_num;
}

#define SAME_SIGN(a,b) (((a>0) and (b>0)) or ((a<0) and (b<0)))
bool DelaunayGrid::test_if_inside(int &tri_number, const lensvector<double>& pt, bool& inside_triangle)
{
	DelaunayGrid_Params<double>& p = assign_delaunay_param_object<double>();
	// To speed things up, these things can be made static and have an array for each (so each thread uses a different set of static elements)
	lensvector<double> dt1, dt2, dt3;
	double cross_prod;

	Triangle<double> *triptr = &p.triangle[tri_number];
	int side, same_sign=0;
	bool prod1_samesign = false;
	bool prod2_samesign = false;
	bool prod3_samesign = false;

	dt1[0] = pt[0] - triptr->vertex[0][0];
	dt1[1] = pt[1] - triptr->vertex[0][1];
	dt2[0] = pt[0] - triptr->vertex[1][0];
	dt2[1] = pt[1] - triptr->vertex[1][1];
	dt3[0] = pt[0] - triptr->vertex[2][0];
	dt3[1] = pt[1] - triptr->vertex[2][1];
	cross_prod = dt1[0]*dt2[1] - dt1[1]*dt2[0];
	if SAME_SIGN(cross_prod,triptr->area) { prod1_samesign = true; same_sign++; }
	cross_prod = dt3[0]*dt1[1] - dt3[1]*dt1[0];
	if SAME_SIGN(cross_prod,triptr->area) { prod2_samesign = true; same_sign++; }
	cross_prod = dt2[0]*dt3[1] - dt2[1]*dt3[0];
	if SAME_SIGN(cross_prod,triptr->area) { prod3_samesign = true; same_sign++; }
	if (same_sign==3) {
		inside_triangle = true;
		return true; // point is inside the triangle
	}
	if (same_sign==0) die("none of cross products have same sign as triangle parity (%g vs %g), pt=(%g,%g); this shouldn't happen",cross_prod,triptr->area,pt[0],pt[1]);
	if (same_sign==2) {
		if (!prod1_samesign) side = 2; // on the side opposite vertex 2
		else if (!prod2_samesign) side = 1; // on the side opposite vertex 1
		else side = 0; // on the side opposite vertex 0
	} else { // same_sign = 1
		if (prod1_samesign) side = 0; // near vertex 2, which is between triangles on sides 0 and 1; we pick triangle 0
		else if (prod2_samesign) side = 2; // near vertex 1, which is between triangles on sides 1 and 2; we pick triangle 2
		else side = 1;  // near vertex 0, which is between triangles on sides 0 and 2; we pick triangle 1
	}
	int new_tri_number = triptr->neighbor_index[side];
	if (new_tri_number < 0) {
		if (same_sign==2) {
			inside_triangle = false;
			return true; // there is no triangle on the given side, so this is the closest triangle
		} else {
			if (prod1_samesign) side = 1; //  we tried triangle 0 and it didn't exist, so try triangle 1 next
			else if (prod2_samesign) side = 0; //  we tried triangle 2 and it didn't exist, so try triangle 0 next
			else side = 2;  //  we tried triangle 1 and it didn't exist, so try triangle 2 next
			new_tri_number = triptr->neighbor_index[side];
			if (new_tri_number < 0) {
				inside_triangle = false;
				return true; // we must be at a corner of the grid, so this is the closest triangle
			}
		}
	}
	tri_number = new_tri_number;
	return false; // if returning 'false', we don't bother to set the 'inside_triangle' flag because it will be ignored anyway
}

bool DelaunayGrid::test_if_inside(const int tri_number, const lensvector<double>& pt)
{
	// To speed things up, these things can be made static and have an array for each (so each thread uses a different set of static elements)
	DelaunayGrid_Params<double>& p = assign_delaunay_param_object<double>();
	lensvector<double> dt1, dt2, dt3;
	double cross_prod;

	//cout << "TRINUM=" << tri_number << endl;
	Triangle<double> *triptr = &p.triangle[tri_number];
	int same_sign=0;
	dt1[0] = pt[0] - triptr->vertex[0][0];
	dt1[1] = pt[1] - triptr->vertex[0][1];
	dt2[0] = pt[0] - triptr->vertex[1][0];
	dt2[1] = pt[1] - triptr->vertex[1][1];
	dt3[0] = pt[0] - triptr->vertex[2][0];
	dt3[1] = pt[1] - triptr->vertex[2][1];
	cross_prod = dt1[0]*dt2[1] - dt1[1]*dt2[0];
	if SAME_SIGN(cross_prod,triptr->area) same_sign++;
	cross_prod = dt3[0]*dt1[1] - dt3[1]*dt1[0];
	if SAME_SIGN(cross_prod,triptr->area) same_sign++;
	cross_prod = dt2[0]*dt3[1] - dt2[1]*dt3[0];
	if SAME_SIGN(cross_prod,triptr->area) same_sign++;
	if (same_sign==3) {
		return true; // point is inside the triangle
	}
	return false; // if returning 'false'
}

#undef SAME_SIGN

int DelaunayGrid::find_closest_vertex(const int tri_number, const lensvector<double>& pt)
{
	// This function effectively allows us to plot the Voronoi cells, since the vertices of the triangles are the "seeds" of the Voronoi cells
	DelaunayGrid_Params<double>& p = assign_delaunay_param_object<double>();
	Triangle<double> *triptr = &p.triangle[tri_number];
	Triangle<double> *neighbor_ptr;
	double distsqr, distsqr_min = 1e30;
	int i,j,k,indx,neighbor_indx;
	for (i=0; i < 3; i++) {
		distsqr = SQR(pt[0] - triptr->vertex[i][0]) + SQR(pt[1] - triptr->vertex[i][1]);
		if (distsqr < distsqr_min) {
			distsqr_min = distsqr;
			indx = triptr->vertex_index[i];
		}
	}
	// Now check neighboring triangles, since it's possible for a point to be closer to a neighboring triangle's vertex,
	// particularly if that point happens to lie close to the edge of the triangle.
	for (i=0; i < 3; i++) {
		if (triptr->neighbor_index[i] != -1) {
			neighbor_ptr = &p.triangle[triptr->neighbor_index[i]];
			for (j=0; j < 3; j++) {
				neighbor_indx = neighbor_ptr->vertex_index[j];
				if ((neighbor_indx != triptr->vertex_index[0]) and (neighbor_indx != triptr->vertex_index[1]) and (neighbor_indx != triptr->vertex_index[2])) {
					distsqr = SQR(pt[0] - neighbor_ptr->vertex[j][0]) + SQR(pt[1] - neighbor_ptr->vertex[j][1]);
					if (distsqr < distsqr_min) {
						distsqr_min = distsqr;
						indx = neighbor_indx;
					}
				}
			}
		}
	}
	return indx;
}


template <typename QScalar>
void DelaunayGrid::find_interpolation_weights_3pt(const lensvector<QScalar>& input_pt, const int trinum, int& npts, const int thread)
{
	DelaunayGrid_Params<QScalar>& p = assign_delaunay_param_object<QScalar>();
	Triangle<QScalar> *triptr = &p.triangle[trinum];
	interpolation_indx[0] = triptr->vertex_index[0];
	interpolation_indx[1] = triptr->vertex_index[1];
	interpolation_indx[2] = triptr->vertex_index[2];
	p.interpolation_pts[0] = &p.gridpts[triptr->vertex_index[0]];
	p.interpolation_pts[1] = &p.gridpts[triptr->vertex_index[1]];
	p.interpolation_pts[2] = &p.gridpts[triptr->vertex_index[2]];
	QScalar d = ((*p.interpolation_pts[0])[0]-(*p.interpolation_pts[1])[0])*((*p.interpolation_pts[1])[1]-(*p.interpolation_pts[2])[1]) - ((*p.interpolation_pts[1])[0]-(*p.interpolation_pts[2])[0])*((*p.interpolation_pts[0])[1]-(*p.interpolation_pts[1])[1]);
	if (d==0) {
		// in this case the points are all the same
		p.interpolation_wgts[0] = 1.0;
		npts = 1;
	} else {
		p.interpolation_wgts[0] = (input_pt[0]*((*p.interpolation_pts[1])[1]-(*p.interpolation_pts[2])[1]) + input_pt[1]*((*p.interpolation_pts[2])[0]-(*p.interpolation_pts[1])[0]) + (*p.interpolation_pts[1])[0]*(*p.interpolation_pts[2])[1] - (*p.interpolation_pts[1])[1]*(*p.interpolation_pts[2])[0])/d;
		p.interpolation_wgts[1] = (input_pt[0]*((*p.interpolation_pts[2])[1]-(*p.interpolation_pts[0])[1]) + input_pt[1]*((*p.interpolation_pts[0])[0]-(*p.interpolation_pts[2])[0]) + (*p.interpolation_pts[0])[1]*(*p.interpolation_pts[2])[0] - (*p.interpolation_pts[0])[0]*(*p.interpolation_pts[2])[1])/d;
		p.interpolation_wgts[2] = (input_pt[0]*((*p.interpolation_pts[0])[1]-(*p.interpolation_pts[1])[1]) + input_pt[1]*((*p.interpolation_pts[1])[0]-(*p.interpolation_pts[0])[0]) + (*p.interpolation_pts[0])[0]*(*p.interpolation_pts[1])[1] - (*p.interpolation_pts[0])[1]*(*p.interpolation_pts[1])[0])/d;
		npts = 3;
	}
}
template void DelaunayGrid::find_interpolation_weights_3pt<double>(const lensvector<double>& input_pt, const int trinum, int& npts, const int thread);
#ifdef USE_STAN
template void DelaunayGrid::find_interpolation_weights_3pt<stan::math::var>(const lensvector<stan::math::var>& input_pt, const int trinum, int& npts, const int thread);
#endif

template <typename QScalar>
void DelaunayGrid::find_interpolation_weights_nn(const lensvector<QScalar> &input_pt, const int trinum, int& npts, const int thread) // natural neighbor interpolation
{
	DelaunayGrid_Params<QScalar>& p = assign_delaunay_param_object<QScalar>();
	npts = 0;
	const int nmax_tri = 60;
	Triangle<QScalar>* adjacent_triangles[nmax_tri];
	int n_adjacent_triangles;

	QScalar area_initial, area_leftover, wgt;
	int ntri_in_envelope = 1;
	Triangle<QScalar> *triptr = &p.triangle[trinum];
	triangles_in_envelope[0] = trinum;
	int k,l,m;

	// recursive lambda function for finding triangles that belong inside the Bowyer-Watson envelope (this will be called below)
	function<bool(Triangle<QScalar>*, const int, const int, int &, int &)> find_triangles_in_envelope = [&](Triangle<QScalar> *neighbor_ptr, const int trinum, const int neighbor_num, int &npt, int &ntri) 
	{
		int idx;
		Triangle<QScalar> *neighbor_ptr2;
		int l, l_new_vertex=-1, l_left, l_right, neighbor_num2;
		QScalar distsq;
		for (l=0; l < 3; l++) {
			if (neighbor_ptr->neighbor_index[l]==trinum) {
				l_new_vertex = l;
				break;
			}
		}
		if (l_new_vertex==-1) {
			warn("could not find vertex within Bowyer-Watson envelope");
			l_new_vertex = 0;
		}
		l_left = l_new_vertex-1;
		if (l_left==-1) l_left = 2;
		l_right = l_new_vertex+1;
		if (l_right==3) l_right = 0;
		neighbor_num2 = neighbor_ptr->neighbor_index[l_right];
		if (neighbor_num2 != -1) {
			neighbor_ptr2 = &p.triangle[neighbor_num2];
			distsq = SQR(input_pt[0]-neighbor_ptr2->circumcenter[0]) + SQR(input_pt[1]-neighbor_ptr2->circumcenter[1]);
			if (distsq < neighbor_ptr2->circumcircle_radsq) {
				if (!find_triangles_in_envelope(neighbor_ptr2,neighbor_num,neighbor_num2,npt,ntri)) return false;
			}
		}
		triangles_in_envelope[ntri++] = neighbor_num;
		if (npt >= nmax_pts_interp) {
			warn("exceeded max number of points (%i versus %i); will use 3-pt interpolation for this point",(npt+1),nmax_pts_interp);
			return false;
		}
		interpolation_indx[npt] = neighbor_ptr->vertex_index[l_new_vertex];
		npt++;
		neighbor_num2 = neighbor_ptr->neighbor_index[l_left];
		if (neighbor_num2 != -1) {
			neighbor_ptr2 = &p.triangle[neighbor_num2];
			distsq = SQR(input_pt[0]-neighbor_ptr2->circumcenter[0]) + SQR(input_pt[1]-neighbor_ptr2->circumcenter[1]);
			if (distsq < neighbor_ptr2->circumcircle_radsq) {
				if (!find_triangles_in_envelope(neighbor_ptr2,neighbor_num,neighbor_num2,npt,ntri)) return false;
			}
		}
		return true;
	};

	Triangle<QScalar> *neighbor_ptr;
	QScalar distsq;
	int kleft,neighbor_num;
	for (k=0; k < 3; k++) {
		if (npts >= nmax_pts_interp) {
			warn("exceeded max number of points (%i versus %i); will use 3-pt interpolation for this point",(npts+1),nmax_pts_interp);
			find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
			return;
		}
		interpolation_indx[npts] = triptr->vertex_index[k];
		npts++;

		kleft = k-1;
		if (kleft==-1) kleft = 2;
		neighbor_num = triptr->neighbor_index[kleft];
		if (neighbor_num != -1) {
			neighbor_ptr = &p.triangle[neighbor_num];
			distsq = SQR(input_pt[0]-neighbor_ptr->circumcenter[0]) + SQR(input_pt[1]-neighbor_ptr->circumcenter[1]);
			if (distsq < neighbor_ptr->circumcircle_radsq) {
				if (!find_triangles_in_envelope(neighbor_ptr,trinum,neighbor_num,npts,ntri_in_envelope)) {
					// exceeded max number of points allowed, so just using 3-pt interpolation instead
					find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
					return;
				}
			}
		}
	}

	int idx,kright;
	QScalar a0,a1,c0,c1,det_inv,asq,csq,ctr0,ctr1;
	for (k=0; k < npts; k++) {
		idx = interpolation_indx[k];
		p.interpolation_pts[k] = &p.gridpts[idx];
	}
	for (k=0; k < npts; k++) {
		kright = k+1;
		if (kright==npts) kright = 0;
		a0 = (*p.interpolation_pts[k])[0]-input_pt[0];
		a1 = (*p.interpolation_pts[k])[1]-input_pt[1];
		c0 = (*p.interpolation_pts[kright])[0]-input_pt[0];
		c1 = (*p.interpolation_pts[kright])[1]-input_pt[1];
		det_inv = 0.5/(a0*c1-c0*a1);
		asq = a0*a0 + a1*a1;
		csq = c0*c0 + c1*c1;
		ctr0 = det_inv*(asq*c1 - csq*a1);
		ctr1 = det_inv*(csq*a0 - asq*c0);
		p.new_circumcenter[k][0] = ctr0 + input_pt[0];
		p.new_circumcenter[k][1] = ctr1 + input_pt[1];
	}

	lensvector<QScalar> midpt_left, midpt_right;
	QScalar totwgt = 0;
	bool first_iteration, fix_mmin, fix_mmax;
	bool mmin_in_envelope, mmax_in_envelope, mmin_in_envelope_prev, mmax_in_envelope_prev;
	int n_polygon_vertices, shared_tri_idx_min, shared_tri_idx_max, mmin_adjacent, mmax_adjacent, mmax, iter;
	for (k=0; k < npts; k++) {
		iter = 0;
		idx = interpolation_indx[k];
		mmin_adjacent = 0;
		mmax_adjacent = n_shared_triangles[idx]-1;
		first_iteration = true;
		mmin_in_envelope_prev = false;
		mmax_in_envelope_prev = false;
		fix_mmin = false;
		fix_mmax = false;

		iter = 0;
		do {
			mmin_in_envelope = false;
			mmax_in_envelope = false;
			shared_tri_idx_min = shared_triangles[idx][mmin_adjacent];
			shared_tri_idx_max = shared_triangles[idx][mmax_adjacent];
			for (m=0; m < ntri_in_envelope; m++) {
				if (shared_tri_idx_min==triangles_in_envelope[m]) mmin_in_envelope = true;
				if (shared_tri_idx_max==triangles_in_envelope[m]) mmax_in_envelope = true;
			}
			if ((mmin_adjacent >= mmax_adjacent) and ((!mmin_in_envelope) and (!mmax_in_envelope) and (!mmin_in_envelope_prev) and (!mmax_in_envelope_prev))) die("there are no shared triangles in envelope!");
			if ((mmin_adjacent <= mmax_adjacent) and ((!first_iteration) and (mmin_in_envelope) and (mmax_in_envelope) and (mmin_in_envelope_prev) and (mmax_in_envelope_prev))) {
				// All the triangles are shared in this case
				mmin_adjacent = 0;
				mmax_adjacent = n_shared_triangles[idx]-1;
				break;
			}
			if (!fix_mmin) {
				if (!mmin_in_envelope_prev) {
					if (!mmin_in_envelope) {
						mmin_adjacent++;
						if (mmin_adjacent==n_shared_triangles[idx]) mmin_adjacent = 0;
					}
					else if ((first_iteration) and (mmin_in_envelope)) {
						mmin_adjacent--;
						if (mmin_adjacent==-1) mmin_adjacent = n_shared_triangles[idx]-1;
					} else {
						fix_mmin = true;
					}
				} else {
					if (mmin_in_envelope) {
						mmin_adjacent--;
						if (mmin_adjacent==-1) mmin_adjacent = n_shared_triangles[idx]-1;
					} else {
						mmin_adjacent++;
						if (mmin_adjacent==n_shared_triangles[idx]) mmin_adjacent = 0;
					}
				}
			}

			if (!fix_mmax) {
				if (!mmax_in_envelope_prev) {
					if (!mmax_in_envelope) {
						mmax_adjacent--;
						if (mmax_adjacent==-1) mmax_adjacent = n_shared_triangles[idx]-1;
					}
					else if ((first_iteration) and (mmax_in_envelope)) {
						mmax_adjacent++;
						if (mmax_adjacent==n_shared_triangles[idx]) mmax_adjacent = 0;
					} else {
						fix_mmax = true;
					}
				} else {
					if (mmax_in_envelope) {
						mmax_adjacent++;
						if (mmax_adjacent==n_shared_triangles[idx]) mmax_adjacent = 0;
					} else {
						mmax_adjacent--;
						if (mmax_adjacent==-1) mmax_adjacent = n_shared_triangles[idx]-1;
					}
				}
			}
			mmin_in_envelope_prev = mmin_in_envelope;
			mmax_in_envelope_prev = mmax_in_envelope;
			if (first_iteration) first_iteration = false;
			iter++;
			if (iter > 100) {
				die("Too many iterations finding ordered list of shared triangles within envelope)");
			}
		} while ((!fix_mmin) or (!fix_mmax));
		if (mmax_adjacent >= mmin_adjacent) n_adjacent_triangles = mmax_adjacent-mmin_adjacent+1;
		else {
			n_adjacent_triangles = n_shared_triangles[idx]-mmin_adjacent+mmax_adjacent+1;
		}
		if (n_adjacent_triangles > nmax_tri) die("number of adjacent triangles exceeded maximum allowed number (%i vs %i)",n_adjacent_triangles,nmax_tri);
		mmax = (mmax_adjacent >= mmin_adjacent) ? mmax_adjacent : n_shared_triangles[idx]-1;
		l=0;
		for (m=mmin_adjacent; m <= mmax; m++) {
			adjacent_triangles[l++] = &p.triangle[shared_triangles[idx][m]];
		}
		if (mmax_adjacent < mmin_adjacent) {
			for (m=0; m <= mmax_adjacent; m++) {
				adjacent_triangles[l++] = &p.triangle[shared_triangles[idx][m]];
			}
		}
		if (l != n_adjacent_triangles) die("number of adjacent triangles didn't add up right (l=%i)",l);

		kleft = k-1;
		kright = k+1;
		if (kleft==-1) kleft = npts-1;
		if (kright==npts) kright = 0;
		midpt_left = ((*p.interpolation_pts[k]) + (*p.interpolation_pts[kleft]))/2;
		midpt_right = ((*p.interpolation_pts[kright]) + (*p.interpolation_pts[k]))/2;
		n_polygon_vertices = n_adjacent_triangles+2;
		p.polygon_vertices[0] = &midpt_left;
		p.polygon_vertices[n_polygon_vertices-1] = &midpt_right;
		l=0;
		for (m=n_adjacent_triangles-1; m >= 0; m--) {
			p.polygon_vertices[l+1] = &(adjacent_triangles[m]->circumcenter);
			l++;
		}

		area_initial = 0;
		for (m=0; m < n_polygon_vertices-1; m++) {
			area_initial += (*p.polygon_vertices[m])[0]*(*p.polygon_vertices[m+1])[1] - (*p.polygon_vertices[m+1])[0]*(*p.polygon_vertices[m])[1];
		}
		area_initial *= -0.5;

		//Now we construct the polygons generated by including the new input point in the grid, creating six new Delaunay triangles
		n_polygon_vertices = 4;
		p.polygon_vertices[1] = &p.new_circumcenter[kleft];
		p.polygon_vertices[2] = &p.new_circumcenter[k];
		p.polygon_vertices[3] = &midpt_right;

		area_leftover = 0;
		for (m=0; m < n_polygon_vertices-1; m++) {
			area_leftover += (*p.polygon_vertices[m])[0]*(*p.polygon_vertices[m+1])[1] - (*p.polygon_vertices[m+1])[0]*(*p.polygon_vertices[m])[1];
		}
		area_leftover *= -0.5;
		wgt = abs(area_initial - area_leftover);

		p.interpolation_wgts[k] = wgt;
		totwgt += wgt;
	}
	for (k=0; k < npts; k++) {
		p.interpolation_wgts[k] /= totwgt;
	}
}
template void DelaunayGrid::find_interpolation_weights_nn<double>(const lensvector<double> &input_pt, const int trinum, int& npts, const int thread); // natural neighbor interpolation;
#ifdef USE_STAN
template void DelaunayGrid::find_interpolation_weights_nn<stan::math::var>(const lensvector<stan::math::var> &input_pt, const int trinum, int& npts, const int thread); // natural neighbor interpolation;
#endif

void DelaunayGrid::find_containing_triangle(const lensvector<double> &input_pt, int& trinum, bool& inside_triangle, bool& on_vertex, int& kmin)
{
	DelaunayGrid_Params<double>& p = assign_delaunay_param_object<double>();
	// this version does not use information from lensing to find a starting triangle during the search; it just starts with triangle zero
	on_vertex = false;
	trinum = search_grid(0,input_pt,inside_triangle);
	Triangle<double> *triptr = &p.triangle[trinum];
	double sqrdist, sqrdistmin=1e30;
	for (int k=0; k < 3; k++) {
		sqrdist = SQR(input_pt[0]-triptr->vertex[k][0]) + SQR(input_pt[1]-triptr->vertex[k][1]);
		if (sqrdist < sqrdistmin) { sqrdistmin = sqrdist; kmin = k; }
	}
	if ((inside_triangle) and (sqrdistmin < 1e-6)) {
		inside_triangle = false;
		on_vertex = true;
	}
}

void DelaunayGrid::generate_covariance_matrix(Eigen::MatrixXd& cov_matrix, const int kernel_type, const double epsilon, double *wgtfac, const bool add_to_covmatrix, const double amplitude)
{
	DelaunayGrid_Params<double>& p = assign_delaunay_param_object<double>();
	bool extra_weighting = (wgtfac==NULL) ? false : true;
	int i,j;
	double sqrdist,x,matern_fac;
	if (kernel_type==0) {
		if (p.matern_index <= 0) die("Matern kernel index nu must be greater than zero");
		matern_fac = pow(2,1-p.matern_index)/Gamma(p.matern_index);
	}

	//double lumreg_rc = qlens->lumreg_rc;
	double wi, wj, fac;
	#pragma omp parallel for private(i,j,sqrdist,x,fac,wi,wj) schedule(dynamic)
	for (i=0; i < n_gridpts; i++) {
		if (extra_weighting) {
			//wi = exp(-wgtfac[i]);
			wi = wgtfac[i];
			//wi = (1-lumreg_rc)*wgtfac[i]+lumreg_rc;
		}
		else wi=1.0;
		fac = wi*wi;
		if (amplitude >= 0) fac *= amplitude;
		if (!add_to_covmatrix) cov_matrix(i,i) = epsilon; // adding epsilon to diagonal reduces numerical error during inversion by increasing the smallest eigenvalues
		cov_matrix(i,i) += fac;
		for (j=i+1; j < n_gridpts; j++) {
			if (!add_to_covmatrix) cov_matrix(i,j) = 0;
			sqrdist = SQR(p.gridpts[i][0]-p.gridpts[j][0]) + SQR(p.gridpts[i][1]-p.gridpts[j][1]);
			double xsig = 0.5;
			if (extra_weighting) {
				//wj = exp(-wgtfac[j]);
				wj = wgtfac[j];
				//wj = (1-lumreg_rc)*wgtfac[j]+lumreg_rc;
				fac = wi*wj;
				//double wj = pow(wgtfac[j],qlens->regparam_lum_index);
				//cout << wi << " " << wj << endl;
			} else {
				fac = 1.0;
			}
			if (amplitude >= 0) fac *= amplitude;
			if (kernel_type==0) {
				x = sqrt(2*p.matern_index*sqrdist)/p.kernel_correlation_length;
				if (x==0) {
					cout << "Got zero distance: x=0... sqrdist=" << sqrdist << " matern_index=" << p.matern_index << " kernel_correlation_length=" << p.kernel_correlation_length << endl;
					cout << "i: " << i << " si_x=" << p.gridpts[i][0] << " si_y=" << p.gridpts[i][1] << endl;
					cout << "j: " << j << " sj_x=" << p.gridpts[j][0] << " sj_y=" << p.gridpts[j][1] << endl;
					die();
				}
				cov_matrix(i,j) += fac*matern_fac*pow(x,p.matern_index)*modified_bessel_function(x,p.matern_index); // Matern kernel
			} else if (kernel_type==1) {
				cov_matrix(i,j) += fac*exp(-sqrt(sqrdist)/p.kernel_correlation_length); // exponential kernel (equal to Matern kernel with matern_index = 0.5)
			} else {
				cov_matrix(i,j) += fac*exp(-sqrdist/(2*p.kernel_correlation_length*p.kernel_correlation_length)); // Gaussian kernel (limit of Matern kernel as matern_index goes to infinity)
			}
		}
	}
}



double DelaunayGrid::modified_bessel_function(const double x, const double nu)
{
	const int MAXIT=10000;
	const double EPS=1e-12;
	const double FPMIN=1e-30;
	const double XMIN=2.0;
	double a,a1,b,c,d,del,del1,delh,dels,e,f,fact,fact2,ff,gam1,gam2,gammi,gampl,h,p,pimu,q,q1,q2,qnew,rk1,rkmu,rkmup,rktemp,s,sum,sum1,x2,xi,xi2,xmu,xmu2;
	int i,l,nl;

	if ((x <= 0) or (nu < 0.0)) die("cannot have x <=0 or nu < 0 for modified Bessel function (x=%g,nu=%g)",x,nu);
	nl = (int) (nu + 0.5);
	xmu = nu-nl;
	xmu2 = xmu*xmu;
	xi = 1.0/x;
	xi2 = 2.0*xi;
	h = nu*xi;
	if (h < FPMIN) h = FPMIN;
	b = xi2*nu;
	d=0.0;
	c=h;
	for (i=0; i < MAXIT; i++) {
		b += xi2;
		d=1.0/(b+d);
		c=b+1.0/c;
		del=c*d;
		h=del*h;
		if (abs(del-1.0) <= EPS) break;
	}
	if (i >= MAXIT) die("x too large for Modified bessel function; try asymptotic expansion");
	fact=nu*xi;
	for (l=nl-1; l >= 0; l--) {
		fact -= xi;
	}
	if (x < XMIN) {
		x2=0.5*x;
		pimu=M_PI*xmu;
		fact = (abs(pimu) < EPS ? 1.0 : pimu/sin(pimu));
		d = -log(x2);
		e=xmu*d;
		fact2 = (abs(e) < EPS ? 1.0 : sinh(e)/e);
		if (abs(xmu) > 1e-8) {
			gampl = 1.0/Gamma(1+xmu);
			gammi = 1.0/Gamma(1-xmu);
			gam2 = (gammi+gampl)/2;
			gam1 = (gammi-gampl)/(2*xmu);
		} else {
			beschb(xmu,gam1,gam2,gampl,gammi); // this is faster but not as accurate as the above four lines...unless xmu is close to zero, in which case it's MORE accurate
		}
		ff=fact*(gam1*cosh(e)+gam2*fact2*d);
		sum=ff;
		e=exp(e);
		p=0.5*e/gampl;
		q=0.5/(e*gammi);
		c=1.0;
		d=x2*x2;
		sum1=p;
		for (i=1; i < MAXIT; i++) {
			ff = (i*ff+p+q)/(i*i-xmu2);
			c *= (d/i);
			p /= (i-xmu);
			q /= (i+xmu);
			del=c*ff;
			sum += del;
			del1 = c*(p-i*ff);
			sum1 += del1;
			if (abs(del) < abs(sum)*EPS) break;
		}
		if (i > MAXIT) die("Modified Bessel series failed to converge");
		rkmu=sum;
		rk1=sum1*xi2;
	} else {
		b=2.0*(1.0+x);
		d=1.0/b;
		h=delh=d;
		q1=0.0;
		q2=1.0;
		a1=0.25-xmu2;
		q=c=a1;
		a = -a1;
		s=1.0+q*delh;
		for (i=1;i < MAXIT; i++) {
			a -= 2*i;
			c = -a*c/(i+1.0);
			qnew=(q1-b*q2)/a;
			q1=q2;
			q2=qnew;
			q += c*qnew;
			b += 2.0;
			d=1.0/(b+a*d);
			delh=(b*d-1.0)*delh;
			h += delh;
			dels=q*delh;
			s += dels;
			if (abs(dels/s) <= EPS) break;
		}
		if (i >= MAXIT) die("Bessel failed to converge in cf2");
		h=a1*h;
		rkmu=sqrt(M_PI/(2.0*x))*exp(-x)/s;
		rk1=rkmu*(xmu+x+0.5-h)*xi;
	}

	for (i=1; i <= nl; i++) {
		rktemp=(xmu+i)*xi2*rk1+rkmu;
		rkmu=rk1;
		rk1=rktemp;
	}
	return rkmu;
}

void DelaunayGrid::beschb(const double x, double& gam1, double& gam2, double& gampl, double& gammi)
{
	const int NUSE1=7, NUSE2=8;
	static double c1[7] = {
		-1.142022680371168e0, 6.5165112670737e-3, 3.087090173086e-4, -3.4706269649e-6, -6.9437664e-9, 3.67795e-11, -1.356e-13 };
	static double c2[8] = {
		1.843740587300905e0, -7.68528408447867e-2, 1.2719271366546e-3, -4.9717367042e-6, -3.31261198e-8, 2.423096e-10, -1.702e-13, -1.49e-15 };
	double xx = 8*x*x-1.0;
	static double *c1p, *c2p;
	c1p = c1;
	c2p = c2;
	gam1=chebev(-1.0,1.0,c1p,NUSE1,xx);
	gam2=chebev(-1.0,1.0,c2p,NUSE2,xx);
	gampl = gam2 - x*gam1;
	gammi = gam2 + x*gam1;
}

double DelaunayGrid::chebev(const double a, const double b, double* c, const int m, const double x)
{
	double d=0.0,dd=0.0,sv,y,y2;
	int j;
	if ((x-a)*(x-b) > 0.0) die("x not in range in function chebev");
	y2 = 2.0*(y=(2.0*x-a-b)/(b-a));
	for (j=m-1; j > 0; j--) {
		sv=d;
		d=y2*d-dd+c[j];
		dd=sv;
	}
	return y*d-dd+0.5*c[0];
}

void DelaunayGrid::delete_grid_arrays()
{
#ifdef USE_STAN
	DelaunayGrid_Params<stan::math::var>& p = assign_delaunay_param_object<stan::math::var>();
	if (p.gridpts != NULL) delete[] p.gridpts;
	if (p.triangle != NULL) delete[] p.triangle;
	p.gridpts = NULL; // just to show that arrays are no longer allocated
	p.triangle = NULL;
#endif

	DelaunayGrid_Params<double>& pd = assign_delaunay_param_object<double>();
	if (pd.gridpts != NULL) delete[] pd.gridpts;
	if (pd.triangle != NULL) {
		delete[] pd.triangle;
		delete[] voronoi_area;
		delete[] voronoi_length;
		for (int i=0; i < n_gridpts; i++) {
			if (n_shared_triangles[i] > 0) {
				delete[] shared_triangles[i];
				delete[] voronoi_boundary_x[i];
				delete[] voronoi_boundary_y[i];
			}
		}
		delete[] voronoi_boundary_x;
		delete[] voronoi_boundary_y;
		delete[] shared_triangles;
		delete[] n_shared_triangles;
		delete[] adj_triangles[0];
		delete[] adj_triangles[1];
		delete[] adj_triangles[2];
		delete[] adj_triangles[3];
	}
	pd.gridpts = NULL; // just to show that arrays are no longer allocated
	pd.triangle = NULL;
	n_gridpts = 0;
}

DelaunayGrid::~DelaunayGrid()
{
	DelaunayGrid_Params<double>& p = assign_delaunay_param_object<double>();
	if (p.gridpts != NULL) delete_grid_arrays();
}

/********************************************* Functions in class DelaunaySourceGrid ***********************************************/

DelaunaySourceGrid::DelaunaySourceGrid(QLens* qlens_in, const int band, const double zsrc_in) : Model()
{
	modelparams = &delaunay_srcgrid_params;
	delaunay_params = &delaunay_srcgrid_params;
#ifdef USE_STAN
	modelparams_dif = &delaunay_srcgrid_params_dif;
	delaunay_params_dif = &delaunay_srcgrid_params_dif;
#endif

	qlens = qlens_in;
	if (zsrc_in < 0) model_name = "delaunay_srcgrid";
	model_name = "delaunay_srcgrid(band=" + mkstring_int(band) + ",z=" + mkstring_doub(zsrc_in) + ")";
	srcgrid_redshift = zsrc_in;

	n_gridpts = 0;
	delaunay_srcgrid_params.triangle = NULL;
	delaunay_srcgrid_params.gridpts = NULL;
	delaunay_srcgrid_params.surface_brightness = NULL;
#ifdef USE_STAN
	delaunay_srcgrid_params_dif.triangle = NULL;
	delaunay_srcgrid_params_dif.gridpts = NULL;
	delaunay_srcgrid_params_dif.surface_brightness = NULL;
#endif

	setup_parameters(true);
	setup_param_pointers<double>();
#ifdef USE_STAN
	setup_param_pointers<stan::math::var>();
#endif
}

template <typename QScalar>
void DelaunaySourceGrid::create_srcpixel_grid(QScalar* srcpts_x, QScalar* srcpts_y, const int n_srcpts, int *ivals_in, int *jvals_in, const int ni, const int nj, const bool find_pixel_magnification, const int imggrid_indx)
{
	DelaunaySourceGrid_Params<QScalar>& p = assign_delaunay_srcgrid_param_object<QScalar>();
	if (p.gridpts != NULL) {
		delete_lensing_arrays();
		delete_grid_arrays();
	}
	create_pixel_grid(srcpts_x,srcpts_y,n_srcpts);

	if ((imggrid_indx >= 0) and (qlens != NULL) and (qlens->image_pixel_grids[imggrid_indx] != NULL)) image_pixel_grid = qlens->image_pixel_grids[imggrid_indx];
	else image_pixel_grid = NULL;

	if (p.surface_brightness != NULL) die("SB wasn't null");
	p.surface_brightness = new QScalar[n_srcpts];
#ifdef USE_STAN
	if constexpr (stan::is_autodiff_v<QScalar>) {
		// since the double version didn't get allocated when create_pixel_grid was called
		delaunay_srcgrid_params.surface_brightness = new double[n_srcpts];
	}
#endif

	inv_magnification = new double[n_srcpts];
	maps_to_image_pixel = new bool[n_srcpts];
	active_pixel = new bool[n_srcpts];
	active_index = new int[n_srcpts];
	if (ivals_in != NULL) {
		imggrid_ivals = new int[n_srcpts];
		imggrid_jvals = new int[n_srcpts];
	}
	int n;
	//srcpixel_xmin=srcpixel_ymin=1e30;
	//srcpixel_xmax=srcpixel_ymax=-1e30;
	n_active_pixels = n_srcpts;
	for (n=0; n < n_srcpts; n++) {
		//cout << "Sourcept " << n << ": " << srcpts_x[n] << " " << srcpts_y[n] << endl;
		//if (srcpts_x[n] > srcpixel_xmax) srcpixel_xmax=srcpts_x[n];
		//if (srcpts_y[n] > srcpixel_ymax) srcpixel_ymax=srcpts_y[n];
		//if (srcpts_x[n] < srcpixel_xmin) srcpixel_xmin=srcpts_x[n];
		//if (srcpts_y[n] < srcpixel_ymin) srcpixel_ymin=srcpts_y[n];
		p.surface_brightness[n] = 0;
		maps_to_image_pixel[n] = false;
		active_pixel[n] = true;
		active_index[n] = -1;
		if (ivals_in != NULL) {
			imggrid_ivals[n] = ivals_in[n];
			imggrid_jvals[n] = jvals_in[n];
		}
	}
	img_imin = 30000;
	img_jmin = 30000;
	img_imax = -30000;
	img_jmax = -30000;

	look_for_starting_point = true; // the "look_for_starting_point" feature for searching triangles doesn't work well with outside_sb_prior
	if ((imggrid_ivals != NULL) and (imggrid_jvals != NULL)) {
		// imggrid_ivals and imggrid_jvals aid the ray-tracing by locating a source point, which has an image point near the point
		// being ray-traced; this gives a starting point when searching through the triangles
		int i,j;
		img_ni = ni+1; // add one since we're doing pixel corners
		img_nj = nj+1; // add one since we're doing pixel corners
		img_index_ij = new int*[img_ni];
		for (i=0; i < img_ni; i++) {
			img_index_ij[i] = new int[img_nj];
			for (j=0; j < img_nj; j++) img_index_ij[i][j] = -1; // for any image pixels that don't have a ray-traced point included in the Delaunay grid, the index will be -1
		}

		for (n=0; n < n_srcpts; n++) {
			img_index_ij[imggrid_ivals[n]][imggrid_jvals[n]] = n;
			//cout << "N=" << n << " " << imggrid_ivals[n] << " " << imggrid_jvals[n] << endl;
			if (imggrid_ivals[n] < img_imin) img_imin = imggrid_ivals[n];
			if (imggrid_ivals[n] > img_imax) img_imax = imggrid_ivals[n];
			if (imggrid_jvals[n] < img_jmin) img_jmin = imggrid_jvals[n];
			if (imggrid_jvals[n] > img_jmax) img_jmax = imggrid_jvals[n];
		}
	} else {
		img_index_ij = NULL;
		look_for_starting_point = false;
	}

	//cout << "TOTAL AREA: " << totarea << endl;
	//delete[] tricheck;
	if ((find_pixel_magnification) and (image_pixel_grid != NULL)) {
		if (image_pixel_grid->cartesian_srcgrid != NULL) find_pixel_magnifications();
	}

	// the following is for plotting purposes
	srcgrid_xmin = 1e30, srcgrid_xmax = -1e30;
	srcgrid_ymin = 1e30, srcgrid_ymax = -1e30;
	for (n=0; n < n_srcpts; n++) {
		if (value_of(srcpts_x[n]) < srcgrid_xmin) srcgrid_xmin = value_of(srcpts_x[n]);
		if (value_of(srcpts_x[n]) > srcgrid_xmax) srcgrid_xmax = value_of(srcpts_x[n]);
		if (value_of(srcpts_y[n]) < srcgrid_ymin) srcgrid_ymin = value_of(srcpts_y[n]);
		if (value_of(srcpts_y[n]) > srcgrid_ymax) srcgrid_ymax = value_of(srcpts_y[n]);
	}
	// add an extra 5% so we can see the outer points on the plots more easily
	double x_extra = 0.05*(srcgrid_xmax-srcgrid_xmax);
	double y_extra = 0.05*(srcgrid_xmax-srcgrid_xmax);
	srcgrid_xmin -= x_extra;
	srcgrid_xmax += x_extra;
	srcgrid_ymin -= y_extra;
	srcgrid_ymax += y_extra;

	/*
	if (qlens != NULL) {
		// This is mainly for plotting purposes
		if (image_pixel_grid != NULL) {
			srcgrid_xmin = image_pixel_grid->src_xmin;
			srcgrid_xmax = image_pixel_grid->src_xmax;
			srcgrid_ymin = image_pixel_grid->src_ymin;
			srcgrid_ymax = image_pixel_grid->src_ymax;
		} else {
			srcgrid_xmin = qlens->sourcegrid_xmin;
			srcgrid_xmax = qlens->sourcegrid_xmax;
			srcgrid_ymin = qlens->sourcegrid_ymin;
			srcgrid_ymax = qlens->sourcegrid_ymax;
		}
	}
	*/
}
template void DelaunaySourceGrid::create_srcpixel_grid<double>(double* srcpts_x, double* srcpts_y, const int n_srcpts, int *ivals_in, int *jvals_in, const int ni, const int nj, const bool find_pixel_magnification, const int imggrid_indx);
#ifdef USE_STAN
template void DelaunaySourceGrid::create_srcpixel_grid<stan::math::var>(stan::math::var* srcpts_x, stan::math::var* srcpts_y, const int n_srcpts, int *ivals_in, int *jvals_in, const int ni, const int nj, const bool find_pixel_magnification, const int imggrid_indx);
#endif

void DelaunaySourceGrid::setup_parameters(const bool initial_setup)
{
	if (initial_setup) {
		setup_parameter_arrays(14);
	} else {
		// always reset the active parameter flags, since the active ones will be determined below
		// NOTE: if (initial_setup==true), active params are reset in setup_parameter_arrays(..) above
		n_active_params = 0;
		for (int i=0; i < n_params; i++) {
			active_params[i] = false; // default
		}
	}

	int indx = 0;

	bool regularized_source, kernel_based_regularization;
	if ((qlens->source_fit_mode==Delaunay_Source) and (qlens->regularization_method != None)) regularized_source = true;
	else regularized_source = false;
	if ((regularized_source) and ((qlens->regularization_method==Exponential_Kernel) or (qlens->regularization_method==Squared_Exponential_Kernel) or (qlens->regularization_method==Matern_Kernel))) kernel_based_regularization = true; 
	else kernel_based_regularization = false;

	if (initial_setup) {
		//param[indx] = &regparam;
		paramnames[indx] = "regparam"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = "";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if (regularized_source) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &kernel_correlation_length;
		paramnames[indx] = "corrlength"; latex_paramnames[indx] = "l"; latex_param_subscripts[indx] = "corr";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if (kernel_based_regularization) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &matern_index;
		paramnames[indx] = "matern_index"; latex_paramnames[indx] = "\\nu"; latex_param_subscripts[indx] = "src";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0.01; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if ((kernel_based_regularization) and (qlens->regularization_method==Matern_Kernel)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &regparam_lsc;
		paramnames[indx] = "regparam_lsc"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = "sc";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if ((qlens->regularization_method != None) and ((qlens->use_lum_weighted_regularization) or (qlens->use_distance_weighted_regularization))) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &regparam_lum_index;
		paramnames[indx] = "regparam_lum_index"; latex_paramnames[indx] = "\\gamma"; latex_param_subscripts[indx] = "reg";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and ((qlens->use_lum_weighted_regularization) or (qlens->use_distance_weighted_regularization))) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_xcenter;
		paramnames[indx] = "distreg_xcenter"; latex_paramnames[indx] = "x"; latex_param_subscripts[indx] = "c,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization) and (!qlens->auto_lumreg_center)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_ycenter;
		paramnames[indx] = "distreg_ycenter"; latex_paramnames[indx] = "y"; latex_param_subscripts[indx] = "c,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization) and (!qlens->auto_lumreg_center)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_e1;
		paramnames[indx] = "distreg_e1"; latex_paramnames[indx] = "e"; latex_param_subscripts[indx] = "1,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_e2;
		paramnames[indx] = "distreg_e2"; latex_paramnames[indx] = "e"; latex_param_subscripts[indx] = "2,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_rc;
		paramnames[indx] = "distreg_rc"; latex_paramnames[indx] = "r"; latex_param_subscripts[indx] = "c,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &mag_weight_sc;
		paramnames[indx] = "mag_weight_sc"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = "\\mu,sc";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if ((qlens->regularization_method != None) and (qlens->use_mag_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &mag_weight_index;
		paramnames[indx] = "mag_weight_index"; latex_paramnames[indx] = "\\gamma"; latex_param_subscripts[indx] = "\\mu";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_mag_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &alpha_clus;
		paramnames[indx] = "alpha_clus"; latex_paramnames[indx] = "\\alpha"; latex_param_subscripts[indx] = "clus";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if ((qlens->use_dist_weighted_srcpixel_clustering) or (qlens->use_lum_weighted_srcpixel_clustering)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &beta_clus;
		paramnames[indx] = "beta_clus"; latex_paramnames[indx] = "\\beta"; latex_param_subscripts[indx] = "clus";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if ((qlens->use_dist_weighted_srcpixel_clustering) or (qlens->use_lum_weighted_srcpixel_clustering)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;
}

template <typename QScalar>
void DelaunaySourceGrid::setup_param_pointers()
{
	DelaunaySourceGrid_Params<QScalar>& p = assign_delaunay_srcgrid_param_object<QScalar>();
	p.regparam = 100;
	p.kernel_correlation_length = 0.1;
	p.matern_index = 0.5;
	p.regparam_lsc = 3;
	p.regparam_lum_index = 1.5;
	p.distreg_xcenter = 0.0;
	p.distreg_ycenter = 0.0;
	p.distreg_e1 = 0.0;
	p.distreg_e2 = 0.0;
	p.distreg_rc = 0.0;
	p.mag_weight_sc = 1.0;
	p.mag_weight_index = 0.3;
	p.alpha_clus = 0.5;
	p.beta_clus = 1.0;

	QScalar** param_ptr = p.param;
	*(param_ptr++) = &p.regparam;
	*(param_ptr++) = &p.kernel_correlation_length;
	*(param_ptr++) = &p.matern_index;
	*(param_ptr++) = &p.regparam_lsc;
	*(param_ptr++) = &p.regparam_lum_index;
	*(param_ptr++) = &p.distreg_xcenter;
	*(param_ptr++) = &p.distreg_ycenter;
	*(param_ptr++) = &p.distreg_e1;
	*(param_ptr++) = &p.distreg_e2;
	*(param_ptr++) = &p.distreg_rc;
	*(param_ptr++) = &p.mag_weight_sc;
	*(param_ptr++) = &p.mag_weight_index;
	*(param_ptr++) = &p.alpha_clus;
	*(param_ptr++) = &p.beta_clus;
}
template void DelaunaySourceGrid::setup_param_pointers<double>();
#ifdef USE_STAN
template void DelaunaySourceGrid::setup_param_pointers<stan::math::var>();
#endif

#ifdef USE_STAN
void DelaunaySourceGrid::sync_autodif_parameters()
{
	delaunay_srcgrid_params_dif.regparam = delaunay_srcgrid_params.regparam;
	delaunay_srcgrid_params_dif.kernel_correlation_length = delaunay_srcgrid_params.kernel_correlation_length;
	delaunay_srcgrid_params_dif.matern_index = delaunay_srcgrid_params.matern_index;
	delaunay_srcgrid_params_dif.regparam_lsc = delaunay_srcgrid_params.regparam_lsc;
	delaunay_srcgrid_params_dif.regparam_lum_index = delaunay_srcgrid_params.regparam_lum_index;
	delaunay_srcgrid_params_dif.distreg_xcenter = delaunay_srcgrid_params.distreg_xcenter;
	delaunay_srcgrid_params_dif.distreg_ycenter = delaunay_srcgrid_params.distreg_ycenter;
	delaunay_srcgrid_params_dif.distreg_e1 = delaunay_srcgrid_params.distreg_e1;
	delaunay_srcgrid_params_dif.distreg_e2 = delaunay_srcgrid_params.distreg_e2;
	delaunay_srcgrid_params_dif.distreg_rc = delaunay_srcgrid_params.distreg_rc;
	delaunay_srcgrid_params_dif.mag_weight_sc = delaunay_srcgrid_params.mag_weight_sc;
	delaunay_srcgrid_params_dif.mag_weight_index = delaunay_srcgrid_params.mag_weight_index;
	delaunay_srcgrid_params_dif.alpha_clus = delaunay_srcgrid_params.alpha_clus;
	delaunay_srcgrid_params_dif.beta_clus = delaunay_srcgrid_params.beta_clus;
}
#endif

void DelaunaySourceGrid::copy_pixsrc_data(const DelaunaySourceGrid* grid_in)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	model_name = grid_in->model_name;
	srcgrid_redshift = grid_in->srcgrid_redshift;
	p.regparam = grid_in->delaunay_srcgrid_params.regparam;
	p.kernel_correlation_length = grid_in->delaunay_srcgrid_params.kernel_correlation_length;
	p.matern_index = grid_in->delaunay_srcgrid_params.matern_index;
	p.regparam_lsc = grid_in->delaunay_srcgrid_params.regparam_lsc;
	p.regparam_lum_index = grid_in->delaunay_srcgrid_params.regparam_lum_index;
	p.distreg_xcenter = grid_in->delaunay_srcgrid_params.distreg_xcenter;
	p.distreg_ycenter = grid_in->delaunay_srcgrid_params.distreg_ycenter;
	p.distreg_e1 = grid_in->delaunay_srcgrid_params.distreg_e1;
	p.distreg_e2 = grid_in->delaunay_srcgrid_params.distreg_e2;
	p.distreg_rc = grid_in->delaunay_srcgrid_params.distreg_rc;
	p.mag_weight_sc = grid_in->delaunay_srcgrid_params.mag_weight_sc;
	p.mag_weight_index = grid_in->delaunay_srcgrid_params.mag_weight_index;
	p.alpha_clus = grid_in->delaunay_srcgrid_params.alpha_clus;
	p.beta_clus = grid_in->delaunay_srcgrid_params.beta_clus;
	copy_param_arrays(grid_in);
#ifdef USE_STAN
	sync_autodif_parameters();
#endif
}

void DelaunaySourceGrid::update_meta_parameters(const bool varied_only_fitparams)
{
	return; // nothing meta to change
}

void DelaunaySourceGrid::get_parameter_numbers_from_qlens(int& pi, int& pf)
{
	if (qlens) {
		qlens->get_pixsrc_parameter_numbers(entry_number,pi,pf);
	}
}

bool DelaunaySourceGrid::register_vary_parameters_in_qlens()
{
	if (qlens != NULL) {
		return qlens->register_pixellated_src_vary_parameters(entry_number);
	}
	return true;
}

void DelaunaySourceGrid::register_limits_in_qlens()
{
	if (qlens != NULL) {
		qlens->register_pixellated_src_prior_limits(entry_number);
	}
}

void DelaunaySourceGrid::update_fitparams_in_qlens()
{
	if (qlens != NULL) {
		qlens->update_pixellated_src_fitparams(entry_number);
	}
}

void DelaunaySourceGrid::find_pixel_magnifications()
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	double area_weighted_invmag, overlap_area, total_overlap_area;
	lensvector<double> pt1,pt2;
	int m,n;
	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif
		//#pragma omp for private(n,m,pt1,pt2,area_weighted_invmag,overlap_area,total_overlap_area) schedule(static)
		for (n=0; n < n_gridpts; n++) {
			if (n_shared_triangles[n] < 3) inv_magnification[n] = 1.0; // make border cells have mag = 1...it's rough, but maybe ok as a first try here
			else {
				inv_magnification[n] = 0.0;
				area_weighted_invmag = 0;
				total_overlap_area = 0;
				for (m=0; m < n_shared_triangles[n]-1; m++) {
					pt1[0] = voronoi_boundary_x[n][m];
					pt1[1] = voronoi_boundary_y[n][m];
					pt2[0] = voronoi_boundary_x[n][m+1];
					pt2[1] = voronoi_boundary_y[n][m+1];
					area_weighted_invmag += image_pixel_grid->cartesian_srcgrid->find_triangle_weighted_invmag(p.gridpts[n],pt1,pt2,overlap_area,thread);
					total_overlap_area += overlap_area;
				}
				//inv_magnification[n] = area_weighted_invmag /= voronoi_area[n];
				if ((total_overlap_area != 0) and (abs(total_overlap_area) >= (0.95*voronoi_area[n]))) { // the latter requirement is a hack to cover the bordering cells that the masked pixels don't completely cover (i.e. pixels outside the mask map to them)
					inv_magnification[n] = area_weighted_invmag /= total_overlap_area;
				}
				else inv_magnification[n] = 1.0;
				if (inv_magnification[n] > 1.0) inv_magnification[n] = 1.0;
				//cout << "srcpt " << n << ": " << inv_magnification[n] << endl;
			}
			//if (qlens->cartesian_srcgrid != NULL) inv_magnification[n] = qlens->cartesian_srcgrid->find_local_inverse_magnification_interpolate(gridpts[n],0);
			//else inv_magnification[n] = 1.0;
			//cout << "mag " << n << ": " << inv_magnification[n] << endl;
		}
	}
}

double DelaunaySourceGrid::sum_edge_sqrlengths(const double min_sb)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	lensvector<double> edge;
	double sum=0;
	bool use_sb = false;
	double** sb;
	int iv[3];
	int jv[3];
	ImageData *image_data = image_pixel_grid->image_data;
	if ((qlens != NULL) and (image_data != NULL)) {
		use_sb = true;
		sb = image_data->surface_brightness;
	}
	int i,j;
	// Note, the inside edges (the majority) will be counted twice, but that's ok
	for (i=0; i < n_triangles; i++) {
		edge = p.triangle[i].vertex[1] - p.triangle[i].vertex[0];
		sum += edge.sqrnorm();
		edge = p.triangle[i].vertex[2] - p.triangle[i].vertex[1];
		sum += edge.sqrnorm();
		edge = p.triangle[i].vertex[0] - p.triangle[i].vertex[2];
		sum += edge.sqrnorm();
	}
	return sum;
}

template <typename QScalar>
void DelaunaySourceGrid::assign_surface_brightness_from_analytic_source(const int imggrid_i)
{
	DelaunaySourceGrid_Params<QScalar>& p = assign_delaunay_srcgrid_param_object<QScalar>();
	//cout << "Sourcepts: " << n_gridpts << endl;
	int i,k;
	for (i=0; i < n_gridpts; i++) {
		//cout << "Assigning SB point " << i << "..." << endl;
		p.surface_brightness[i] = 0;
		for (k=0; k < qlens->n_sb; k++) {
			//cout << "source " << k << endl;
			if ((qlens->sb_list[k]->is_lensed) and ((imggrid_i<0) or (qlens->sbprofile_imggrid_idx[k]==imggrid_i))) {
				p.surface_brightness[i] += qlens->sb_list[k]->surface_brightness(p.gridpts[i][0],p.gridpts[i][1]);
				/*
				QScalar sbval = qlens->sb_list[k]->surface_brightness(p.gridpts[i][0],p.gridpts[i][1]);
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					cout << "sb=" << sbval.val() << endl;
				} else
#endif
				cout << "sb=" << sbval << endl;
				*/
			}
		}
	}
#ifdef USE_STAN
	if constexpr (stan::is_autodiff_v<QScalar>) {
		for (i=0; i < n_gridpts; i++) {
			delaunay_srcgrid_params.surface_brightness[i] = stan::math::value_of(p.surface_brightness[i]);
		}
	}
#endif

}
template void DelaunaySourceGrid::assign_surface_brightness_from_analytic_source<double>(const int imggrid_i);
#ifdef USE_STAN
template void DelaunaySourceGrid::assign_surface_brightness_from_analytic_source<stan::math::var>(const int imggrid_i);
#endif

void DelaunaySourceGrid::fill_surface_brightness_vector()
{
	if (image_pixel_grid==NULL) warn("delaunay source pixels cannot access image pixel grid; cannot fill surface brightness vector");
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	int i,j;
	for (i=0, j=0; i < n_gridpts; i++) {
		image_pixel_grid->amplitude_vector[j++] = p.surface_brightness[i];
	}
}

void DelaunaySourceGrid::update_surface_brightness(int& index)
{
	if (image_pixel_grid==NULL) warn("delaunay source pixels cannot access image pixel grid; cannot update surface brightness from amplitudes");
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	int i;
	//cout << "SOURCE SB: " << endl;
	for (i=0; i < n_gridpts; i++) {
		p.surface_brightness[i] = image_pixel_grid->amplitude_vector[index++];
		//cout << "pixel " << i << ": SB=" << surface_brightness[i] << " (index=" << (index-1) << ")" << endl;
	}
}

bool DelaunaySourceGrid::assign_source_mapping_flags(lensvector<double> &input_pt, vector<PtsWgts<double>>& mapped_delaunay_srcpixels_ij, int& n_mapped_srcpixels, const int img_pixel_i, const int img_pixel_j, const int thread, bool& trouble_with_starting_vertex)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	int trinum,kmin;
	bool inside_triangle, on_vertex;
	if (!find_containing_triangle_with_imgpix(input_pt,img_pixel_i,img_pixel_j,trinum,inside_triangle,on_vertex,kmin)) trouble_with_starting_vertex = true;
	Triangle<double> *triptr = &p.triangle[trinum];

	if (!inside_triangle) {
		// we don't want to extrapolate, because it can lead to crazy results outside the grid. so we find the closest vertex and use that vertex's SB
		if ((zero_outside_border) and (!on_vertex)) {
			n_mapped_srcpixels = 0;
			return true;
		}
		// if we're outside the grid, only attempt to extrapolate if using natural neighbor interpolation; if using 3-pt interpolation, just use closest vertex
		lensvector<double> dist = input_pt - p.gridpts[triptr->vertex_index[kmin]];
		if ((!qlens->natural_neighbor_interpolation) or (dist.norm() < 1e-6)) {
			PtsWgts<double> pt(triptr->vertex_index[kmin],1);
			mapped_delaunay_srcpixels_ij.push_back(pt);
			maps_to_image_pixel[triptr->vertex_index[kmin]] = true;
			n_mapped_srcpixels = 1;
			return true;
		}
	}
	if (qlens->natural_neighbor_interpolation) {
		find_interpolation_weights_nn(input_pt, trinum, n_mapped_srcpixels, thread);
	} else {
		find_interpolation_weights_3pt(input_pt, trinum, n_mapped_srcpixels, thread);
	}
	PtsWgts<double> pt;
	//cout << "n_mapped_srcpixels=" << n_mapped_srcpixels << " (imggrid_i=" << image_pixel_grid->src_redshift_index << ")" << endl;
	for (int i=0; i < n_mapped_srcpixels; i++) {
		maps_to_image_pixel[interpolation_indx[i]] = true;
		mapped_delaunay_srcpixels_ij.push_back(pt.assign(interpolation_indx[i],p.interpolation_wgts[i]));
		//cout << "point: " << interpolation_indx[i] << " active_index=" << active_index[interpolation_indx[i]] << " (imggrid_i=" << image_pixel_grid->src_redshift_index << ")" << endl;
	}
	return true;
}

void DelaunaySourceGrid::calculate_Lmatrix(const int img_index, PtsWgts<double>* mapped_delaunay_srcpixels, int* n_mapped_srcpixels, int& index, const int& subpixel_indx, const double weight, const int& thread)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	int i;
	for (i=0; i < subpixel_indx; i++) mapped_delaunay_srcpixels += (*n_mapped_srcpixels++); // each mapped image subpixel can have its own number of pixels in the Delaunay grid it maps to, so we must skip through these to the requested subpixel index
	//if ((image_pixel_grid->src_redshift_index==1) and (*n_mapped_srcpixels != 0)) cout << "Lmatrix_sparse: n_mapped_srcpixels=" << (*n_mapped_srcpixels) << " (imggrid_i=" << image_pixel_grid->src_redshift_index << ")" << endl;
	for (i=0; i < (*n_mapped_srcpixels); i++) {
		//cout << "Lmatrix_sparse srcpixel " << mapped_delaunay_srcpixels->indx << " active_indx=" << active_index[mapped_delaunay_srcpixels->indx] << " (imggrid_i=" << image_pixel_grid->src_redshift_index << ")" << endl;
		image_pixel_grid->Lmatrix_index_rows[img_index].push_back(active_index[mapped_delaunay_srcpixels->indx]);
		image_pixel_grid->Lmatrix_rows[img_index].push_back(weight*mapped_delaunay_srcpixels->wgt);
		mapped_delaunay_srcpixels++;
	}
	index += (*n_mapped_srcpixels);
}

void DelaunaySourceGrid::calculate_Lmatrix_elements(const int img_index, PtsWgts<double>* mapped_delaunay_srcpixels, int* n_mapped_srcpixels, int& index, const int& subpixel_indx, const double weight, const int& thread)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	int i;
	for (i=0; i < subpixel_indx; i++) mapped_delaunay_srcpixels += (*n_mapped_srcpixels++); // each mapped image subpixel can have its own number of pixels in the Delaunay grid it maps to, so we must skip through these to the requested subpixel index
	//if ((image_pixel_grid->src_redshift_index==1) and (*n_mapped_srcpixels != 0)) cout << "Lmatrix_sparse: n_mapped_srcpixels=" << (*n_mapped_srcpixels) << " (imggrid_i=" << image_pixel_grid->src_redshift_index << ")" << endl;
	for (i=0; i < (*n_mapped_srcpixels); i++) {
		//cout << "Lmatrix_sparse srcpixel " << mapped_delaunay_srcpixels->indx << " active_indx=" << active_index[mapped_delaunay_srcpixels->indx] << " (imggrid_i=" << image_pixel_grid->src_redshift_index << ")" << endl;
		//image_pixel_grid->Lmatrix_dense(img_index,active_index[mapped_delaunay_srcpixels->indx]) += weight*mapped_delaunay_srcpixels->wgt;
		image_pixel_grid->Lmatrix_trans_dense(active_index[mapped_delaunay_srcpixels->indx],img_index) += weight*mapped_delaunay_srcpixels->wgt;
		//image_pixel_grid->Lmatrix_dense0[img_index][active_index[mapped_delaunay_srcpixels->indx]] += weight*mapped_delaunay_srcpixels->wgt;
		mapped_delaunay_srcpixels++;
	}
	index += (*n_mapped_srcpixels);
}

template <typename QScalar>
QScalar DelaunaySourceGrid::find_lensed_surface_brightness(const lensvector<QScalar> &input_pt, const int img_pixel_i, const int img_pixel_j, const int thread)
{
	DelaunaySourceGrid_Params<QScalar>& p = assign_delaunay_srcgrid_param_object<QScalar>();
	bool inside_triangle, on_vertex;
	int trinum,kmin;
#ifdef USE_STAN
	if constexpr (stan::is_autodiff_v<QScalar>) {
		lensvector<double> input_pt_doub;
		// This is inefficient. You should modify the functions involved so you can pass in the components, so you don't have to create another lensvector<double>
		input_pt_doub[0] = stan::math::value_of(input_pt[0]);
		input_pt_doub[1] = stan::math::value_of(input_pt[1]);
		find_containing_triangle_with_imgpix(input_pt_doub,img_pixel_i,img_pixel_j,trinum,inside_triangle,on_vertex,kmin);
	} else
#endif
	{
		find_containing_triangle_with_imgpix(input_pt,img_pixel_i,img_pixel_j,trinum,inside_triangle,on_vertex,kmin);
	}

	if (!inside_triangle) {
		Triangle<QScalar> *triptr = &p.triangle[trinum];
		// we don't want to extrapolate, because it can lead to crazy results outside the grid. so we find the closest vertex and use that vertex's SB
		if ((zero_outside_border) and (!on_vertex)) {
			return 0;
		} else {
			// if we're outside the grid, only attempt to extrapolate if using natural neighbor interpolation; if using 3-pt interpolation, just use closest vertex
			lensvector<QScalar> dist = input_pt - p.gridpts[triptr->vertex_index[kmin]];
			//if ((!qlens->natural_neighbor_interpolation) or (dist.norm() < 1e-6)) return *triptr->sb[kmin];
			double distnorm = value_of(dist.norm());
			if ((!qlens->natural_neighbor_interpolation) or (distnorm < 1e-6)) return p.surface_brightness[triptr->vertex_index[kmin]];

		}
	}
	int npts;
	QScalar sb_interp = 0;
	if (qlens->natural_neighbor_interpolation) {
		find_interpolation_weights_nn(input_pt, trinum, npts, thread);
	} else {
		find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
	}
	for (int i=0; i < npts; i++) {
		sb_interp += p.surface_brightness[interpolation_indx[i]]*p.interpolation_wgts[i];
	}
	return sb_interp;
}
template double DelaunaySourceGrid::find_lensed_surface_brightness<double>(const lensvector<double> &input_pt, const int img_pixel_i, const int img_pixel_j, const int thread);
#ifdef USE_STAN
template stan::math::var DelaunaySourceGrid::find_lensed_surface_brightness<stan::math::var>(const lensvector<stan::math::var> &input_pt, const int img_pixel_i, const int img_pixel_j, const int thread);
#endif

template <typename QScalar>
QScalar DelaunaySourceGrid::interpolate_surface_brightness(const lensvector<QScalar> &input_pt, const bool interp_mag, const int thread)
{
	DelaunaySourceGrid_Params<QScalar>& p = assign_delaunay_srcgrid_param_object<QScalar>();
	bool inside_triangle, on_vertex;
	int trinum,kmin;
#ifdef USE_STAN
	if constexpr (stan::is_autodiff_v<QScalar>) {
		lensvector<double> input_pt_doub;
		// This is inefficient. You should modify the functions involved so you can pass in the components, so you don't have to create another lensvector<double>
		input_pt_doub[0] = stan::math::value_of(input_pt[0]);
		input_pt_doub[1] = stan::math::value_of(input_pt[1]);
		find_containing_triangle(input_pt_doub,trinum,inside_triangle,on_vertex,kmin);
	} else
#endif
	{
		find_containing_triangle(input_pt,trinum,inside_triangle,on_vertex,kmin);
	}

	if (!inside_triangle) {
		//cout << "NOT INSIDE TRIANGLE" << endl;
		Triangle<QScalar> *triptr = &p.triangle[trinum];
		// we don't want to extrapolate, because it can lead to crazy results outside the grid. so we find the closest vertex and use that vertex's SB
		if ((zero_outside_border) and (!on_vertex)) {
			return 0;
		} else {
			// if we're outside the grid, only attempt to extrapolate if using natural neighbor interpolation; if using 3-pt interpolation, just use closest vertex
			lensvector<QScalar> dist = input_pt - p.gridpts[triptr->vertex_index[kmin]];
			//if ((!qlens->natural_neighbor_interpolation) or (dist.norm() < 1e-6)) return *triptr->sb[kmin];
			double distnorm = value_of(dist.norm());
			if ((!qlens->natural_neighbor_interpolation) or (distnorm < 1e-6)) return p.surface_brightness[triptr->vertex_index[kmin]];
		}
	}

	int npts;
	QScalar interp_val = 0;
	if (qlens->natural_neighbor_interpolation) {
		find_interpolation_weights_nn(input_pt, trinum, npts, thread);
	} else {
		find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
	}
	//if (interp_mag) {
		//for (int i=0; i < npts; i++) {
			//interp_val += inv_magnification[interpolation_indx[i]]*p.interpolation_wgts[i];
		//}
		//interp_val = 1.0/interp_val; // this produces the magnification (rather than inverse mag)
	//} else {
		//	cout << "NPTS=" << npts << endl;
		for (int i=0; i < npts; i++) {
			interp_val += p.surface_brightness[interpolation_indx[i]]*p.interpolation_wgts[i];
//#ifdef USE_STAN
			//if constexpr (std::is_same_v<QScalar, stan::math::var>) {
				//cout << "indx=" << interpolation_indx[i] << " sb=" << (p.surface_brightness[interpolation_indx[i]]).val() << " wgt=" << (p.interpolation_wgts[i]).val() << endl;
			//} else
//#endif
			//cout << "indx=" << interpolation_indx[i] << " sb=" << p.surface_brightness[interpolation_indx[i]] << " wgt=" << p.interpolation_wgts[i] << endl;
		}
	//}
	return interp_val;
}
template double DelaunaySourceGrid::interpolate_surface_brightness<double>(const lensvector<double> &input_pt, const bool interp_mag, const int thread);
#ifdef USE_STAN
template stan::math::var DelaunaySourceGrid::interpolate_surface_brightness<stan::math::var>(const lensvector<stan::math::var> &input_pt, const bool interp_mag, const int thread);
#endif

void DelaunaySourceGrid::print_pixel_values()
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	for (int n=0; n < n_gridpts; n++) {
		cout << p.surface_brightness[n] << endl;
	}
}

double DelaunaySourceGrid::interpolate_voronoi_length(const lensvector<double> &input_pt, const int thread)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	bool inside_triangle;
	bool on_vertex;
	int trinum,kmin;
	find_containing_triangle(input_pt,trinum,inside_triangle,on_vertex,kmin);
	//cout << "point: " << input_pt[0] << " " << input_pt[1] << endl;
	if (!inside_triangle) {
		return (voronoi_length[p.triangle[trinum].vertex_index[kmin]]);
	}

	int npts;
	double interp_val = 0;
	if (qlens->natural_neighbor_interpolation) {
		find_interpolation_weights_nn(input_pt, trinum, npts, thread);
	} else {
		find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
	}
	//cout << "NPTS=" << npts << endl;
	for (int i=0; i < npts; i++) {
		//cout << "VORONOI LENGTH: " << voronoi_length[interpolation_indx[i]] << endl;
		interp_val += voronoi_length[interpolation_indx[i]]*p.interpolation_wgts[i];
	}
	//cout << "ITERPOLATED V LENGHT: " << interp_val << endl;
	return interp_val;
}

bool DelaunaySourceGrid::find_containing_triangle_with_imgpix(const lensvector<double> &input_pt, const int img_pixel_i, const int img_pixel_j, int& trinum, bool& inside_triangle, bool& on_vertex, int& kmin)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	bool found_good_starting_vertex = true; // until proven otherwise
	int i,j,k,maxk,n;
	if (look_for_starting_point) {
		i = img_pixel_i;
		j = img_pixel_j;
		if (i < img_imin) i = img_imin;
		else if (i > img_imax) i = img_imax;
		if (j < img_jmin) j = img_jmin;
		else if (j > img_jmax) j = img_jmax;
		maxk = imax(img_imax-img_imin,img_jmax-img_jmin);
		n=img_index_ij[i][j];
		k=0;
		while (n==-1) {
			// looking for a (ray-traced) pixel within the mask, since the closest pixel doesn't seem to be
			k++;
			if ((j-k >= img_jmin) and (n=img_index_ij[i][j-k]) >= 0) break;
			if ((j+k <= img_jmax) and (n=img_index_ij[i][j+k]) >= 0) break;
			if (i-k >= img_imin) {
				if ((n=img_index_ij[i-k][j]) >= 0) break;
				if ((j-k >= img_jmin) and (n=img_index_ij[i-k][j-k]) >= 0) break;
				if ((j+k <= img_jmax) and (n=img_index_ij[i-k][j+k]) >= 0) break;
			}
			if (i+k <= img_imax) {
				if ((n=img_index_ij[i+k][j]) >= 0) break;
				if ((j-k >= img_jmin) and (n=img_index_ij[i+k][j-k]) >= 0) break;
				if ((j+k <= img_jmax) and (n=img_index_ij[i+k][j+k]) >= 0) break;
			}
			if (k > maxk) {
				found_good_starting_vertex = false;
				n=0; // in this case, can't find a good vertex to start with, so we just start with the first one
			}
			//cout << "i=" << img_pixel_i << " j=" << img_pixel_j << " imin=" << img_imin << " imax=" << img_imax << " jmin=" << img_jmin << " jmax=" << img_jmax << " n=" << n << endl;
		}
	} else {
		n = 0;
	}
	//cout << "searching for point (" << input_pt[0] << "," << input_pt[1] << "), starting with pixel " << n << " (" << gridpts[n][0] << " " << gridpts[n][1] << ")" << endl;
	on_vertex = false;
	trinum = search_grid(n,input_pt,inside_triangle);
	//cout << "...found in triangle " << trinum << endl;
	Triangle<double> *triptr = &p.triangle[trinum];
	double sqrdist, sqrdistmin=1e30;
	for (k=0; k < 3; k++) {
		sqrdist = SQR(input_pt[0]-triptr->vertex[k][0]) + SQR(input_pt[1]-triptr->vertex[k][1]);
		if (sqrdist < sqrdistmin) { sqrdistmin = sqrdist; kmin = k; }
	}
	if ((inside_triangle) and (sqrdistmin < 1e-6)) {
		inside_triangle = false;
		on_vertex = true;
	}
	return found_good_starting_vertex;
}

int DelaunaySourceGrid::assign_active_indices_and_count_source_pixels(const int source_pixel_i_initial, const bool activate_unmapped_pixels)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	int source_pixel_i=source_pixel_i_initial;
	for (int i=0; i < n_gridpts; i++) {
		active_pixel[i] = true;
		active_index[i] = source_pixel_i++;
		if (!maps_to_image_pixel[i]) {
			if ((qlens->mpi_id==0) and (qlens->regularization_method == 0)) warn(qlens->warnings,"A source pixel does not map to any image pixel (for source pixel %i), center (%g,%g)",i,p.gridpts[i][0],p.gridpts[i][1]); // only show warning if no regularization being used, since matrix cannot be inverted in that case
		}
	}
	n_active_pixels = source_pixel_i-source_pixel_i_initial;
	return source_pixel_i;
}

void DelaunaySourceGrid::generate_hmatrices(const bool interpolate)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	record_adjacent_triangles_xy();

	auto add_hmatrix_entry = [](ImagePixelGrid *image_pixel_grid, const int l, const int i, const int j, const double entry)
	{
		// l gives index for finite differencing (l=0: +y, l=1: -y, l=2: +x, l=3: -x)
		int dup = false;
		for (int k=0; k < image_pixel_grid->hmatrix_row_nn[l][i]; k++) {
			if (image_pixel_grid->hmatrix_index_rows[l][i][k]==j) {
				image_pixel_grid->hmatrix_rows[l][i][k] += entry;
				dup = true;
				break;
			}
		}
		if (!dup) {
			image_pixel_grid->hmatrix_rows[l][i].push_back(entry);
			image_pixel_grid->hmatrix_index_rows[l][i].push_back(j);
			image_pixel_grid->hmatrix_row_nn[l][i]++;
			image_pixel_grid->hmatrix_nn[l]++;
		}
	};

	int i,j,k,l;
	if (interpolate) {
		int npts;
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		double x,y,xp,xm,yp,ym;
		lensvector<double> interp_pt[4];
		for (i=0; i < n_gridpts; i++) {
			x = p.gridpts[i][0];
			y = p.gridpts[i][1];
			xp = x + voronoi_length[i]/2;
			xm = x - voronoi_length[i]/2;
			yp = y + voronoi_length[i]/2;
			ym = y - voronoi_length[i]/2;
			interp_pt[0].input(xp,y);
			interp_pt[1].input(xm,y);
			interp_pt[2].input(x,yp);
			interp_pt[3].input(x,ym);
			add_hmatrix_entry(image_pixel_grid,0,i,i,-2.0);
			add_hmatrix_entry(image_pixel_grid,1,i,i,-2.0);
			for (j=0; j < 4; j++) {
				if (j > 1) l = 1;
				else l = 0;
				find_containing_triangle(interp_pt[j],trinum,inside_triangle,on_vertex,kmin);
				if (!inside_triangle) {
					if (!on_vertex) continue; // assume SB = 0 outside grid
				}
				if (qlens->natural_neighbor_interpolation) {
					find_interpolation_weights_nn(interp_pt[j], trinum, npts, 0);
				} else {
					find_interpolation_weights_3pt(interp_pt[j], trinum, npts, 0);
				}
				for (k=0; k < npts; k++) {
					add_hmatrix_entry(image_pixel_grid,l,i,interpolation_indx[k],p.interpolation_wgts[k]); 
				}
			}
		}
	} else {
		int vertex_i1, vertex_i2, trinum;
		Triangle<double>* triptr;
		bool found_i1, found_i2;
		double x1, y1, x2, y2, dpt, dpt1, dpt2, dpt12;
		double length, minlength, avg_length;
		avg_length = sqrt(avg_area);
		lensvector<double> pt;
		for (i=0; i < n_gridpts; i++) {
			for (j=0; j < 4; j++) {
				if (j > 1) l = 1;
				else l = 0;
				vertex_i1 = -1;
				vertex_i2 = -1;
				if ((trinum = adj_triangles[j][i]) != -1) {
					triptr = &p.triangle[trinum];
					found_i1 = false;
					found_i2 = false;
					if ((k = triptr->vertex_index[0]) != i) { vertex_i1 = k; found_i1 = true; }
					if ((k = triptr->vertex_index[1]) != i) {
						if (!found_i1) {
							vertex_i1 = k;
							found_i1 = true;
						} else {
							vertex_i2 = k;
							found_i2 = true;
						}
					}
					if (!found_i1) die("WHAT?! couldn't find more than one vertex that isn't the one in question");
					if ((!found_i2) and ((k = triptr->vertex_index[2]) != i)) {
						vertex_i2 = k;
						found_i2 = true;
					}
					if (!found_i2) die("WHAT?! couldn't find both vertices that aren't the one in question");
				}
				if (vertex_i1 != -1) {
					x1 = p.gridpts[vertex_i1][0];
					y1 = p.gridpts[vertex_i1][1];
					x2 = p.gridpts[vertex_i2][0];
					y2 = p.gridpts[vertex_i2][1];
					if (j < 2) {
						pt[1] = p.gridpts[i][1];
						pt[0] = ((x2-x1)/(y2-y1))*(pt[1]-y1) + x1;
						dpt = abs(pt[0]-p.gridpts[i][0]);
					} else {
						pt[0] = p.gridpts[i][0];
						pt[1] = ((y2-y1)/(x2-x1))*(pt[0]-x1) + y1;
						dpt = abs(pt[1]-p.gridpts[i][1]);
					}
					dpt12 = sqrt(SQR(x2-x1) + SQR(y2-y1));
					dpt1 = sqrt(SQR(pt[0]-x1)+SQR(pt[1]-y1));
					dpt2 = sqrt(SQR(pt[0]-x2)+SQR(pt[1]-y2));
					// we scale hmatrix by the average triangle length so the regularization parameter is dimensionless
					add_hmatrix_entry(image_pixel_grid,l,i,i,-avg_length/dpt);
					add_hmatrix_entry(image_pixel_grid,l,i,vertex_i1,avg_length*dpt2/(dpt*dpt12));
					add_hmatrix_entry(image_pixel_grid,l,i,vertex_i2,avg_length*dpt1/(dpt*dpt12));
				} else {
					minlength=1e30;
					for (k=0; k < n_shared_triangles[i]; k++) {
						triptr = &p.triangle[shared_triangles[i][k]];
						length = sqrt(triptr->area);
						if (length < minlength) minlength = length;
					}
					add_hmatrix_entry(image_pixel_grid,l,i,i,-avg_length/minlength);
					//add_hmatrix_entry(image_pixel_grid,l,i,i,sqrt(1/2.0)/2);
				}
			}
		}
	}
}

void DelaunaySourceGrid::generate_gmatrices(const bool interpolate)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	if (!interpolate) record_adjacent_triangles_xy();

	auto add_gmatrix_entry = [](ImagePixelGrid *image_pixel_grid, const int l, const int i, const int j, const double entry)
	{
		// l gives index for finite differencing (l=0: +y, l=1: -y, l=2: +x, l=3: -x)
		int dup = false;
		for (int k=0; k < image_pixel_grid->gmatrix_row_nn[l][i]; k++) {
			if (image_pixel_grid->gmatrix_index_rows[l][i][k]==j) {
				image_pixel_grid->gmatrix_rows[l][i][k] += entry;
				dup = true;
				break;
			}
		}
		if (!dup) {
			image_pixel_grid->gmatrix_rows[l][i].push_back(entry);
			image_pixel_grid->gmatrix_index_rows[l][i].push_back(j);
			image_pixel_grid->gmatrix_row_nn[l][i]++;
			image_pixel_grid->gmatrix_nn[l]++;
		}
	};

	int i,k,l;
	if (interpolate) {
		int npts;
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		double x,y,xp,xm,yp,ym;
		lensvector<double> interp_pt[4];
		for (i=0; i < n_gridpts; i++) {
			//cout << "HARG i=" << i << endl;
			x = p.gridpts[i][0];
			y = p.gridpts[i][1];
			xp = x + voronoi_length[i];
			xm = x - voronoi_length[i];
			yp = y + voronoi_length[i];
			ym = y - voronoi_length[i];
			interp_pt[0].input(xp,y);
			interp_pt[1].input(xm,y);
			interp_pt[2].input(x,yp);
			interp_pt[3].input(x,ym);
			for (l=0; l < 4; l++) {
				//cout << "BLERG l=" << l << endl;
				add_gmatrix_entry(image_pixel_grid,l,i,i,1.0);
				find_containing_triangle(interp_pt[l],trinum,inside_triangle,on_vertex,kmin);
				//cout << "BLERG1 l=" << l << endl;
				if (!inside_triangle) {
					if (!on_vertex) continue; // assume SB = 0 outside grid
				}
				if (qlens->natural_neighbor_interpolation) {
				//cout << "BLERG2 l=" << l << endl;
					find_interpolation_weights_nn(interp_pt[l], trinum, npts, 0);
				//cout << "BLERG3 l=" << l << endl;
				} else {
					find_interpolation_weights_3pt(interp_pt[l], trinum, npts, 0);
				}
				//cout << "BLERG4 l=" << l << endl;
				for (k=0; k < npts; k++) {
					add_gmatrix_entry(image_pixel_grid,l,i,interpolation_indx[k],-p.interpolation_wgts[k]);
				}
				//cout << "BLERG5 l=" << l << endl;
			}
		}
	} else {
		int vertex_i1, vertex_i2, trinum;
		Triangle<double>* triptr;
		bool found_i1, found_i2;
		double length, minlength;
		double x1, y1, x2, y2, dpt, dpt1, dpt2, dpt12;
		lensvector<double> pt;
		for (i=0; i < n_gridpts; i++) {
			for (l=0; l < 4; l++) {
				vertex_i1 = -1;
				vertex_i2 = -1;
				if ((trinum = adj_triangles[l][i]) != -1) {
					triptr = &p.triangle[trinum];
					found_i1 = false;
					found_i2 = false;
					if ((k = triptr->vertex_index[0]) != i) { vertex_i1 = k; found_i1 = true; }
					if ((k = triptr->vertex_index[1]) != i) {
						if (!found_i1) {
							vertex_i1 = k;
							found_i1 = true;
						} else {
							vertex_i2 = k;
							found_i2 = true;
						}
					}
					if (!found_i1) die("WHAT?! couldn't find more than one vertex that isn't the one in question");
					if ((!found_i2) and ((k = triptr->vertex_index[2]) != i)) {
						vertex_i2 = k;
						found_i2 = true;
					}
					if (!found_i2) die("WHAT?! couldn't find both vertices that aren't the one in question");
				}
				if (vertex_i1 != -1) {
					x1 = p.gridpts[vertex_i1][0];
					y1 = p.gridpts[vertex_i1][1];
					x2 = p.gridpts[vertex_i2][0];
					y2 = p.gridpts[vertex_i2][1];
					if (l < 2) {
						pt[1] = p.gridpts[i][1];
						pt[0] = ((x2-x1)/(y2-y1))*(pt[1]-y1) + x1;
						dpt = abs(pt[0]-p.gridpts[i][0]);
					} else {
						pt[0] = p.gridpts[i][0];
						pt[1] = ((y2-y1)/(x2-x1))*(pt[0]-x1) + y1;
						dpt = abs(pt[1]-p.gridpts[i][1]);
					}
					dpt12 = sqrt(SQR(x2-x1) + SQR(y2-y1));
					dpt1 = sqrt(SQR(pt[0]-x1)+SQR(pt[1]-y1));
					dpt2 = sqrt(SQR(pt[0]-x2)+SQR(pt[1]-y2));
					add_gmatrix_entry(image_pixel_grid,l,i,i,1.0);
					add_gmatrix_entry(image_pixel_grid,l,i,vertex_i1,-dpt2/(dpt12));
					add_gmatrix_entry(image_pixel_grid,l,i,vertex_i2,-dpt1/(dpt12));
					//add_gmatrix_entry(image_pixel_grid,l,i,i,sqrt(1/2.0)/2);
				} else {
					minlength=1e30;
					for (k=0; k < n_shared_triangles[i]; k++) {
						triptr = &p.triangle[shared_triangles[i][k]];
						length = sqrt(triptr->area);
						if (length < minlength) minlength = length;
					}
					add_gmatrix_entry(image_pixel_grid,l,i,i,1.0);
					//add_gmatrix_entry(image_pixel_grid,l,i,i,sqrt(1/2.0)/2);
				}
			}
		}
	}
}

void DelaunaySourceGrid::find_source_gradient(const lensvector<double>& input_pt, lensvector<double>& src_grad, const int thread)
{
	double interval;
	lensvector<double> pt_p, pt_m;
	interval = interpolate_voronoi_length(input_pt,thread) / 4.0;
	//interval = 1e-4;
	//interval = interpolate_voronoi_length(input_pt,thread) / 100.0;
	if (interval > 0.5) {
		warn("HUGE INTERVAL! %g (at x=%g,y=%g)",interval,input_pt[0],input_pt[1]); // this is generally only an issue at the borders
		interval = 0.05;
	}
	//cout << "interval=" << interval << endl;
	pt_p[0] = input_pt[0] + interval;
	pt_p[1] = input_pt[1];
	pt_m[0] = input_pt[0] - interval;
	pt_m[1] = input_pt[1];
	src_grad[0] = (interpolate_surface_brightness(pt_p,false,thread) - interpolate_surface_brightness(pt_m,false,thread)) / (2*interval);

	pt_p[0] = input_pt[0];
	pt_p[1] = input_pt[1] + interval;
	pt_m[0] = input_pt[0];
	pt_m[1] = input_pt[1] - interval;
	src_grad[1] = (interpolate_surface_brightness(pt_p,false,thread) - interpolate_surface_brightness(pt_m,false,thread)) / (2*interval);
	//cout << "SOURCE GRAD: " << src_grad[0] << " " << src_grad[1] << endl;
}


void DelaunaySourceGrid::output_surface_brightness(Vector<double>& xvals, Vector<double>& yvals, Vector<double>& sbvals, const int npix, const bool interpolate_sb, const bool plot_magnification)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	double x, y, xlength, ylength, pixel_xlength, pixel_ylength;
	int i, j, k, npts_x, npts_y;
	xlength = srcgrid_xmax-srcgrid_xmin;
	ylength = srcgrid_ymax-srcgrid_ymin;
	npts_x = (int) npix*sqrt(xlength/ylength);
	npts_y = (int) npts_x*ylength/xlength;
	pixel_xlength = xlength/npts_x;
	pixel_ylength = ylength/npts_y;

	xvals.input(npts_x+1);
	yvals.input(npts_y+1);
	for (i=0, x=srcgrid_xmin; i <= npts_x; i++, x += pixel_xlength) xvals[i] = x;
	for (i=0, y=srcgrid_ymin; i <= npts_y; i++, y += pixel_ylength) yvals[i] = y;
	sbvals.input(npts_x*npts_y);

	int srcpt_i, trinum;
	double sb;
	//cout << "npts_x=" << npts_x << " " << xlength << " " << ylength << " " << npix << " " << srcgrid_xmax << " " << srcgrid_xmin << " " << srcgrid_ymax << " " << srcgrid_ymin << endl;
	//cout << "npts_y=" << npts_y << endl;
	lensvector<double> pt;
	k=0;
	for (j=0, y=srcgrid_ymin+pixel_xlength/2; j < npts_y; j++, y += pixel_ylength) {
		pt[1] = y;
		for (i=0, x=srcgrid_xmin+pixel_xlength/2; i < npts_x; i++, x += pixel_xlength) {
			pt[0] = x;
			if (interpolate_sb) {
				sb = interpolate_surface_brightness(pt,plot_magnification);
			} else {
				// The following lines will plot the Voronoi cells that are dual to the Delaunay triangulation. Note however, that when SB interpolation is
				// performed during ray-tracing, we use the vertices of the triangle that a point lands in, which may not include the closest vertex (i.e. the
				// Voronoi cell it lies in). Thus, the Voronoi cells are for visualization only, and do not directly show what the ray-traced SB will look like.
				bool inside_triangle;
				trinum = search_grid(0,pt,inside_triangle); // maybe you can speed this up later by choosing a better initial triangle
				srcpt_i = find_closest_vertex(trinum,pt);
				if (plot_magnification) sb = log(1.0/inv_magnification[srcpt_i])/ln10;
				else sb = p.surface_brightness[srcpt_i];
			}
			sbvals[k++] = sb;
			//cout << x << " " << y << " " << gridpts[srcpt_i][0] << " " << gridpts[srcpt_i][1] << endl;
		}
	}
}

void DelaunaySourceGrid::plot_voronoi_grid(string root)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	string srcpt_filename = root + "_srcpts.dat";
	ofstream srcout; qlens->open_output_file(srcout,srcpt_filename);
	int i,j;
	for (i=0; i < n_gridpts; i++) {
		srcout << p.gridpts[i][0] << " " << p.gridpts[i][1] << endl;
	}
	string voronoi_filename = root + "_voronoi.dat";
	ofstream vout; qlens->open_output_file(vout,voronoi_filename);
	for (i=0; i < n_gridpts; i++) {
		for (j=0; j < n_shared_triangles[i]; j++) {
			vout << voronoi_boundary_x[i][j] << " " << voronoi_boundary_y[i][j] << endl;
		}
		vout << voronoi_boundary_x[i][0] << " " << voronoi_boundary_y[i][0] << endl << endl;
		//cout << "# length = " << voronoi_length[i] << endl;
	}
	string delaunay_filename = root + "_delaunay.dat";
	ofstream delout; qlens->open_output_file(delout,delaunay_filename);
	for (i=0; i < n_triangles; i++) {
		delout << p.triangle[i].vertex[0][0] << " " << p.triangle[i].vertex[0][1] << endl;
		delout << p.triangle[i].vertex[1][0] << " " << p.triangle[i].vertex[1][1] << endl;
		delout << p.triangle[i].vertex[2][0] << " " << p.triangle[i].vertex[2][1] << endl;
		delout << p.triangle[i].vertex[0][0] << " " << p.triangle[i].vertex[0][1] << endl;
		delout << endl;
	}
}

double DelaunaySourceGrid::find_moment(const int p, const int q, const int npix, const double xc, const double yc, const double b, const double a, const double phi)
{
	//DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	double x, y, xlength, ylength, pixel_xlength, pixel_ylength;
	int i, j, k, npts_x, npts_y;
	xlength = srcgrid_xmax-srcgrid_xmin;
	ylength = srcgrid_ymax-srcgrid_ymin;
	npts_x = (int) npix*sqrt(xlength/ylength);
	npts_y = (int) npts_x*ylength/xlength;
	pixel_xlength = xlength/npts_x;
	pixel_ylength = ylength/npts_y;

	double sb,xp,yq,moment=0;
	lensvector<double> pt;
	double y0 = srcgrid_ymin + pixel_ylength/2;
	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif
		double xp,yp;
		double cosphi = cos(phi);
		double sinphi = sin(phi);
		const double sigfac = 2.5; // sigma clipping factor
		double sigsqfac = sigfac*sigfac;
		bool do_sigmaclip = true;
		if ((a >= 1e30) or (b >= 1e30)) do_sigmaclip = false;
		#pragma omp for private(i,j,k,x,y,xp,yp,sb,pt,yq) schedule(static)
		for (j=0; j < npts_y; j++) {
			y = y0 + j*pixel_ylength;
			pt[1] = y;
			for (k=0,yq=1;k<q;k++) yq *= y;
			for (i=0, x=srcgrid_xmin+pixel_xlength/2; i < npts_x; i++, x += pixel_xlength) {
				xp = (x-xc)*cosphi + (y-yc)*sinphi;
				yp = -(x-xc)*sinphi + (y-yc)*cosphi;
				if ((do_sigmaclip) and ((SQR(xp/a)+SQR(yp/b)) > sigsqfac)) continue; // in this case, the point is outside of the 2*sigma range based on the approximate ellipse from covariance matrix

				pt[0] = x;
				for (k=0,xp=1;k<p;k++) xp *= x;
				sb = interpolate_surface_brightness(pt,false,thread);
				#pragma omp atomic
				moment += sb*xp*yq;
			}
		}
	}
	return moment;
}

void DelaunaySourceGrid::find_source_moments(const int npix, double &qs, double &phi_s, double &sigavg, double &xavg, double &yavg)
{
	double M00, M10, M01, M20, M02, M11;
	std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	const int max_it = 10;
	const double qstol = 0.01; // fractional tolerance in accuracy of qs
	int index = 0;
	double mu20,mu02,mu11,sqrdesc,desc,lam1_times_two,lam2_times_two;
	double xc,yc,sig_major=1e30,sig_minor=1e30;
	xc = (srcgrid_xmax+srcgrid_xmin)/2; // initial xc isn't actually getting used
	yc = (srcgrid_ymax+srcgrid_ymin)/2; // initial yc isn't actually getting used

	phi_s = 0;
	qs = 0;
	double qs_old;
	do {
		qs_old = qs;
		M00 = find_moment(0,0,npix,xc,yc,sig_minor,sig_major,phi_s);
		M10 = find_moment(1,0,npix,xc,yc,sig_minor,sig_major,phi_s);
		M01 = find_moment(0,1,npix,xc,yc,sig_minor,sig_major,phi_s);
		xavg = M10/M00;
		yavg = M01/M00;
		//cout << "xavg=" << xavg << " yavg=" << yavg << endl;
		M20 = find_moment(2,0,npix,xc,yc,sig_minor,sig_major,phi_s);
		M02 = find_moment(0,2,npix,xc,yc,sig_minor,sig_major,phi_s);
		M11 = find_moment(1,1,npix,xc,yc,sig_minor,sig_major,phi_s);
		xc = xavg;
		yc = yavg;
		mu20 = M20/M00 - xavg*xavg;
		mu02 = M02/M00 - yavg*yavg;
		mu11 = M11/M00 - xavg*yavg;
		//cout << "mu20=" << mu20 << " mu02=" << mu02 << " mu11=" << mu11 << endl;
		sqrdesc = 4*mu11*mu11 + SQR(mu20-mu02);
		if (sqrdesc < 0) sqrdesc = 0;
		desc = sqrt(sqrdesc);
		lam1_times_two = mu20 + mu02 + desc;
		lam2_times_two = mu20 + mu02 - desc;
		if (lam2_times_two <= 0) lam2_times_two = 0;
		sig_major = sqrt(lam1_times_two/2);
		sig_minor = sqrt(lam2_times_two/2);
		qs = sqrt(lam2_times_two/lam1_times_two);
		if (mu20==mu02) phi_s = 0;
		else {
			phi_s = atan(2*mu11/(mu20-mu02))/2;
			if (mu02 > mu20) phi_s += M_HALFPI;
		}
		//cout << "b=" << sig_minor << " a=" << sig_major << endl;
		//cout << "qs=" << qs << " phi_s=" << phi_s << endl;
	} while ((abs(qs-qs_old) > qstol*qs) and (++index < max_it));
	sigavg = sqrt(sig_major*sig_minor);
	//} while (index++ < 2);
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for finding qs, phi_s for adaptive source grid: "  << wtime.count() << endl;
	}
}

void DelaunaySourceGrid::get_grid_points(vector<double>& xvals, vector<double>& yvals, vector<double>& sb_vals)
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	for (int i=0; i < n_gridpts; i++) {
		xvals.push_back(p.gridpts[i][0]);
		yvals.push_back(p.gridpts[i][1]);
		sb_vals.push_back(p.surface_brightness[i]);
	}
}

void DelaunaySourceGrid::delete_lensing_arrays()
{
#ifdef USE_STAN
	DelaunaySourceGrid_Params<stan::math::var>& p = assign_delaunay_srcgrid_param_object<stan::math::var>();
	if (p.surface_brightness != NULL) delete[] p.surface_brightness;
	p.surface_brightness = NULL;
#endif
	DelaunaySourceGrid_Params<double>& pd = assign_delaunay_srcgrid_param_object<double>();
	if (pd.surface_brightness != NULL) delete[] pd.surface_brightness;
	pd.surface_brightness = NULL;
	delete[] inv_magnification;
	delete[] maps_to_image_pixel;
	delete[] active_pixel;
	delete[] active_index;
	if (imggrid_ivals != NULL) { delete[] imggrid_ivals; imggrid_ivals = NULL; }
	if (imggrid_jvals != NULL) { delete[] imggrid_jvals; imggrid_jvals = NULL; }
	if (img_index_ij != NULL) {
		for (int i=0; i < img_ni; i++) delete[] img_index_ij[i];
		delete[] img_index_ij;
		img_index_ij = NULL;
	}
}

DelaunaySourceGrid::~DelaunaySourceGrid()
{
	DelaunaySourceGrid_Params<double>& p = assign_delaunay_srcgrid_param_object<double>();
	if (p.param != NULL) delete[] p.param;
	if (p.surface_brightness != NULL) {
		delete_lensing_arrays();
		//delete_grid_arrays();
	}
}

/********************************************* Functions in class LensPixelGrid ***********************************************/

LensPixelGrid::LensPixelGrid(QLens* lens_in, const int lens_redshift_indx_in) : Model()
{
	modelparams = &lensgrid_params;
	delaunay_params = &lensgrid_params;
#ifdef USE_STAN
	modelparams_dif = &lensgrid_params_dif;
	delaunay_params_dif = &lensgrid_params_dif;
#endif
	qlens = lens_in;
	cartesian_pixel_index = NULL;
	xvals_cartesian = NULL;
	yvals_cartesian = NULL;
	potential = NULL;
	lens_redshift_idx = lens_redshift_indx_in;
	grid_type = CartesianPixelGrid; // default
	npix_x = 100; npix_y = 100; // default
	include_in_lensing_calculations = false; // no point including until a pixel grid has been created

	n_gridpts = 0;
	lensgrid_params.triangle = NULL;
	lensgrid_params.gridpts = NULL;
#ifdef USE_STAN
	lensgrid_params_dif.triangle = NULL;
	lensgrid_params_dif.gridpts = NULL;
#endif


	setup_parameters(true);
	setup_param_pointers<double>();
#ifdef USE_STAN
	setup_param_pointers<stan::math::var>();
#endif
}

void LensPixelGrid::create_cartesian_pixel_grid(const double xmin, const double xmax, const double ymin, const double ymax, const int src_redshift_indx)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	grid_type = CartesianPixelGrid;
	if (p.gridpts != NULL) {
		delete_lensing_arrays();
		delete_grid_arrays();
	}
	n_gridpts = npix_x*npix_y;
	p.gridpts = new lensvector<double>[n_gridpts];
	cartesian_ivals = new int[n_gridpts];
	cartesian_jvals = new int[n_gridpts];
	xvals_cartesian = new double[npix_x];
	yvals_cartesian = new double[npix_y];

	double x, y;
	cartesian_pixel_xlength = (xmax-xmin)/npix_x;
	cartesian_pixel_ylength = (ymax-ymin)/npix_y;

	int i,j,k;
	k=0;
	cartesian_pixel_index = new int*[npix_x];
	for (i=0, x=xmin + cartesian_pixel_xlength/2; i < npix_x; i++, x += cartesian_pixel_xlength) {
		xvals_cartesian[i] = x;
		cartesian_pixel_index[i] = new int[npix_y];
		for (j=0, y=ymin + cartesian_pixel_ylength/2; j < npix_y; j++, y += cartesian_pixel_ylength) {
			if (i==0) yvals_cartesian[j] = y;
			cartesian_pixel_index[i][j] = k; 
			p.gridpts[k][0] = x;
			p.gridpts[k][1] = y;
			cartesian_ivals[k] = i;
			cartesian_jvals[k] = j;
			k++;
		}
	}

	xmin_cartesian = xmin+cartesian_pixel_xlength/2;
	xmax_cartesian = xmax-cartesian_pixel_xlength/2;
	ymin_cartesian = ymin+cartesian_pixel_ylength/2;
	ymax_cartesian = ymax-cartesian_pixel_ylength/2;

	if ((src_redshift_indx >= 0) and (qlens != NULL) and (qlens->image_pixel_grids != NULL) and (qlens->image_pixel_grids[src_redshift_indx] != NULL)) image_pixel_grid = qlens->image_pixel_grids[src_redshift_indx];
	else image_pixel_grid = NULL;

	potential = new double[n_gridpts];
	maps_to_image_pixel = new bool[n_gridpts];
	active_pixel = new bool[n_gridpts];
	active_index = new int[n_gridpts];
	int n;
	for (n=0; n < n_gridpts; n++) {
		potential[n] = 0;
		maps_to_image_pixel[n] = false;
		active_pixel[n] = true;
		active_index[n] = n; // true if no pixels are inactive
	}

	if (qlens != NULL) {
		// This is mainly for plotting purposes
		lensgrid_xmin = qlens->grid_xcenter - qlens->grid_xlength/2;
		lensgrid_xmax = qlens->grid_xcenter + qlens->grid_xlength/2;
		lensgrid_ymin = qlens->grid_ycenter - qlens->grid_ylength/2;
		lensgrid_ymax = qlens->grid_ycenter + qlens->grid_ylength/2;
	}
	include_in_lensing_calculations = true;
}

void LensPixelGrid::create_delaunay_pixel_grid(double* pts_x, double* pts_y, const int n_pts, const int src_redshift_indx)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	grid_type = DelaunayPixelGrid;
	if (p.gridpts != NULL) {
		delete_lensing_arrays();
		delete_grid_arrays();
	}
	create_pixel_grid<double>(pts_x,pts_y,n_pts);

	if ((src_redshift_indx >= 0) and (qlens != NULL) and (qlens->image_pixel_grids[src_redshift_indx] != NULL)) image_pixel_grid = qlens->image_pixel_grids[src_redshift_indx];
	else image_pixel_grid = NULL;

	potential = new double[n_gridpts];
	maps_to_image_pixel = new bool[n_gridpts];
	active_pixel = new bool[n_gridpts];
	active_index = new int[n_gridpts];
	int n;
	for (n=0; n < n_gridpts; n++) {
		potential[n] = 0;
		maps_to_image_pixel[n] = false;
		active_pixel[n] = true;
		active_index[n] = -1;
	}
	include_in_lensing_calculations = true;
}

void LensPixelGrid::setup_parameters(const bool initial_setup)
{
	if (initial_setup) {
		setup_parameter_arrays(14);
	} else {
		// always reset the active parameter flags, since the active ones will be determined below
		// NOTE: if (initial_setup==true), active params are reset in setup_parameter_arrays(..) above
		n_active_params = 0;
		for (int i=0; i < n_params; i++) {
			active_params[i] = false; // default
		}
	}

	int indx = 0;

	bool regularized_source, kernel_based_regularization;
	if ((qlens->source_fit_mode==Delaunay_Source) and (qlens->regularization_method != None)) regularized_source = true;
	else regularized_source = false;
	if ((regularized_source) and ((qlens->regularization_method==Exponential_Kernel) or (qlens->regularization_method==Squared_Exponential_Kernel) or (qlens->regularization_method==Matern_Kernel))) kernel_based_regularization = true; 
	else kernel_based_regularization = false;

	if (initial_setup) {
		//param[indx] = &regparam;
		paramnames[indx] = "regparam"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = "";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if (regularized_source) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &kernel_correlation_length;
		paramnames[indx] = "corrlength"; latex_paramnames[indx] = "l"; latex_param_subscripts[indx] = "corr";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if (kernel_based_regularization) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &matern_index;
		paramnames[indx] = "matern_index"; latex_paramnames[indx] = "\\nu"; latex_param_subscripts[indx] = "src";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0.01; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if ((kernel_based_regularization) and (qlens->regularization_method==Matern_Kernel)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &regparam_lsc;
		paramnames[indx] = "regparam_lsc"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = "sc";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if ((qlens->regularization_method != None) and ((qlens->use_lum_weighted_regularization) or (qlens->use_distance_weighted_regularization))) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &regparam_lum_index;
		paramnames[indx] = "regparam_lum_index"; latex_paramnames[indx] = "\\gamma"; latex_param_subscripts[indx] = "reg";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and ((qlens->use_lum_weighted_regularization) or (qlens->use_distance_weighted_regularization))) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_xcenter;
		paramnames[indx] = "distreg_xcenter"; latex_paramnames[indx] = "x"; latex_param_subscripts[indx] = "c,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization) and (!qlens->auto_lumreg_center)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_ycenter;
		paramnames[indx] = "distreg_ycenter"; latex_paramnames[indx] = "y"; latex_param_subscripts[indx] = "c,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization) and (!qlens->auto_lumreg_center)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_e1;
		paramnames[indx] = "distreg_e1"; latex_paramnames[indx] = "e"; latex_param_subscripts[indx] = "1,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_e2;
		paramnames[indx] = "distreg_e2"; latex_paramnames[indx] = "e"; latex_param_subscripts[indx] = "2,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &distreg_rc;
		paramnames[indx] = "distreg_rc"; latex_paramnames[indx] = "r"; latex_param_subscripts[indx] = "c,\\lambda";
		set_auto_penalty_limits[indx] = false; 
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_distance_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &mag_weight_sc;
		paramnames[indx] = "mag_weight_sc"; latex_paramnames[indx] = "\\lambda"; latex_param_subscripts[indx] = "\\mu,sc";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if ((qlens->regularization_method != None) and (qlens->use_mag_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &mag_weight_index;
		paramnames[indx] = "mag_weight_index"; latex_paramnames[indx] = "\\gamma"; latex_param_subscripts[indx] = "\\mu";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = false;
	}
	if ((qlens->regularization_method != None) and (qlens->use_mag_weighted_regularization)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &alpha_clus;
		paramnames[indx] = "alpha_clus"; latex_paramnames[indx] = "\\alpha"; latex_param_subscripts[indx] = "clus";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if ((qlens->use_dist_weighted_srcpixel_clustering) or (qlens->use_lum_weighted_srcpixel_clustering)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		//param[indx] = &beta_clus;
		paramnames[indx] = "beta_clus"; latex_paramnames[indx] = "\\beta"; latex_param_subscripts[indx] = "clus";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.3; scale_stepsize_by_param_value[indx] = true;
	}
	if ((qlens->use_dist_weighted_srcpixel_clustering) or (qlens->use_lum_weighted_srcpixel_clustering)) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;
}

template <typename QScalar>
void LensPixelGrid::setup_param_pointers()
{
	LensPixelGrid_Params<QScalar>& p = assign_lensgrid_param_object<QScalar>();
	p.regparam = 100;
	p.kernel_correlation_length = 0.1;
	p.matern_index = 0.5;
	p.regparam_lsc = 3;
	p.regparam_lum_index = 1.5;
	p.distreg_xcenter = 0.0;
	p.distreg_ycenter = 0.0;
	p.distreg_e1 = 0.0;
	p.distreg_e2 = 0.0;
	p.distreg_rc = 0.0;
	p.mag_weight_sc = 1.0;
	p.mag_weight_index = 0.3;
	p.alpha_clus = 0.5;
	p.beta_clus = 1.0;

	QScalar** param_ptr = p.param;
	*(param_ptr++) = &p.regparam;
	*(param_ptr++) = &p.kernel_correlation_length;
	*(param_ptr++) = &p.matern_index;
	*(param_ptr++) = &p.regparam_lsc;
	*(param_ptr++) = &p.regparam_lum_index;
	*(param_ptr++) = &p.distreg_xcenter;
	*(param_ptr++) = &p.distreg_ycenter;
	*(param_ptr++) = &p.distreg_e1;
	*(param_ptr++) = &p.distreg_e2;
	*(param_ptr++) = &p.distreg_rc;
	*(param_ptr++) = &p.mag_weight_sc;
	*(param_ptr++) = &p.mag_weight_index;
	*(param_ptr++) = &p.alpha_clus;
	*(param_ptr++) = &p.beta_clus;
}
template void LensPixelGrid::setup_param_pointers<double>();
#ifdef USE_STAN
template void LensPixelGrid::setup_param_pointers<stan::math::var>();
#endif

#ifdef USE_STAN
void LensPixelGrid::sync_autodif_parameters()
{
	lensgrid_params_dif.regparam = lensgrid_params.regparam;
	lensgrid_params_dif.kernel_correlation_length = lensgrid_params.kernel_correlation_length;
	lensgrid_params_dif.matern_index = lensgrid_params.matern_index;
	lensgrid_params_dif.regparam_lsc = lensgrid_params.regparam_lsc;
	lensgrid_params_dif.regparam_lum_index = lensgrid_params.regparam_lum_index;
	lensgrid_params_dif.distreg_xcenter = lensgrid_params.distreg_xcenter;
	lensgrid_params_dif.distreg_ycenter = lensgrid_params.distreg_ycenter;
	lensgrid_params_dif.distreg_e1 = lensgrid_params.distreg_e1;
	lensgrid_params_dif.distreg_e2 = lensgrid_params.distreg_e2;
	lensgrid_params_dif.distreg_rc = lensgrid_params.distreg_rc;
	lensgrid_params_dif.mag_weight_sc = lensgrid_params.mag_weight_sc;
	lensgrid_params_dif.mag_weight_index = lensgrid_params.mag_weight_index;
	lensgrid_params_dif.alpha_clus = lensgrid_params.alpha_clus;
	lensgrid_params_dif.beta_clus = lensgrid_params.beta_clus;
	sync_delaunaygrid_autodif_parameters();
}
#endif

void LensPixelGrid::copy_pixlens_data(LensPixelGrid* grid_in)
{
	// add in cartesian grid data to be copied
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	npix_x = grid_in->npix_x;
	npix_y = grid_in->npix_y;
	p.regparam = grid_in->lensgrid_params.regparam;
	p.kernel_correlation_length = grid_in->lensgrid_params.kernel_correlation_length;
	p.matern_index = grid_in->lensgrid_params.matern_index;
	p.regparam_lsc = grid_in->lensgrid_params.regparam_lsc;
	p.regparam_lum_index = grid_in->lensgrid_params.regparam_lum_index;
	p.distreg_xcenter = grid_in->lensgrid_params.distreg_xcenter;
	p.distreg_ycenter = grid_in->lensgrid_params.distreg_ycenter;
	p.distreg_e1 = grid_in->lensgrid_params.distreg_e1;
	p.distreg_e2 = grid_in->lensgrid_params.distreg_e2;
	p.distreg_rc = grid_in->lensgrid_params.distreg_rc;
	p.mag_weight_sc = grid_in->lensgrid_params.mag_weight_sc;
	p.mag_weight_index = grid_in->lensgrid_params.mag_weight_index;
	p.alpha_clus = grid_in->lensgrid_params.alpha_clus;
	p.beta_clus = grid_in->lensgrid_params.beta_clus;
	copy_param_arrays(grid_in);
#ifdef USE_STAN
	sync_autodif_parameters();
#endif
}

void LensPixelGrid::update_meta_parameters(const bool varied_only_fitparams)
{
	return; // nothing meta to change
}

void LensPixelGrid::assign_potential_from_analytic_lens(const int lens_indx, const bool add_potential)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	//cout << "Sourcepts: " << n_gridpts << endl;
	if (lens_indx >= qlens->nlens) die("specified qlens number does not exist");
	double pot;
	for (int i=0; i < n_gridpts; i++) {
		pot = qlens->lens_list[lens_indx]->potential(p.gridpts[i][0],p.gridpts[i][1]);
		if (add_potential) potential[i] += pot;
		else potential[i] = pot;
	}
	include_in_lensing_calculations = true;

}

void LensPixelGrid::assign_potential_from_analytic_lenses()
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	//cout << "Sourcepts: " << n_gridpts << endl;
	int i,k;
	for (i=0; i < n_gridpts; i++) {
		potential[i] = 0.0;
	}
	for (k=0; k < qlens->nlens; k++) {
		for (i=0; i < n_gridpts; i++) {
			potential[i] += qlens->lens_list[k]->potential(p.gridpts[i][0],p.gridpts[i][1]);
		}
	}
}

void LensPixelGrid::fill_potential_correction_vector()
{
	if (image_pixel_grid==NULL) warn("lensgrid pixels cannot access image pixel grid; cannot fill potential correction vector");
	int i,j;
	for (i=0, j=image_pixel_grid->n_amps; i < image_pixel_grid->n_amps+n_gridpts; i++) {
		image_pixel_grid->amplitude_vector[j++] = potential[i];
	}
}

void LensPixelGrid::update_potential(int& index)
{
	if (image_pixel_grid==NULL) warn("lensgrid pixels cannot access image pixel grid; cannot update potential corrections");
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	int i;
	for (i=0; i < n_gridpts; i++) {
		potential[i] = image_pixel_grid->amplitude_vector[index++];
	}
}

double LensPixelGrid::interpolate_potential(const double x, const double y, const int thread)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	int npts;
	lensvector<double> input_pt;
	input_pt[0] = x;
	input_pt[1] = y;

	if (grid_type==CartesianPixelGrid) {
		find_interpolation_weights_cartesian(input_pt, npts, 0, thread);
	} else {
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		find_containing_triangle(input_pt,trinum,inside_triangle,on_vertex,kmin);
		if (!inside_triangle) {
			if ((zero_outside_border) and (!on_vertex)) {
				return 0;
			} else {
				//return *p.triangle[trinum].sb[kmin];
				return potential[p.triangle[trinum].vertex_index[kmin]];
			}
		}
		if (qlens->natural_neighbor_interpolation) {
			find_interpolation_weights_nn(input_pt, trinum, npts, thread);
		} else {
			find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
		}
	}

	double interp_val = 0;
	for (int i=0; i < npts; i++) {
		//cout << "indx: " << interpolation_indx[i] << " POTENTIAL: " << potential[interpolation_indx[i]] << " wgt: " << interpolation_wgts[i] << endl;
		interp_val += potential[interpolation_indx[i]]*p.interpolation_wgts[i];
	}
	//cout << "INTERP VAL: " << interp_val << endl;
	return interp_val;
}

void LensPixelGrid::deflection(const double x, const double y, lensvector<double>& def, const int thread)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	int npts, half_npts;
	lensvector<double> input_pt;
	input_pt[0] = x;
	input_pt[1] = y;

	if (grid_type==CartesianPixelGrid) {
		find_interpolation_weights_cartesian(input_pt, npts, 1, thread);
	} else {
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		find_containing_triangle(input_pt,trinum,inside_triangle,on_vertex,kmin);
		//if (!inside_triangle) {
			//if ((zero_outside_border) and (!on_vertex)) {
				//return 0;
			//} else {
				//return *triangle[trinum].sb[kmin];
			//}
		//}
		// need to implement the corresponding delaunay code for getting deflection
		die("haven't implemented this yet");
		if (qlens->natural_neighbor_interpolation) {
			find_interpolation_weights_nn(input_pt, trinum, npts, thread);
		} else {
			find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
		}
	}
	half_npts = npts/2;

	def[0] = 0;
	def[1] = 0;
	int i;
	for (i=0; i < half_npts; i++) {
		def[0] += potential[interpolation_indx[i]]*p.interpolation_wgts[i];
	}
	for (i=half_npts; i < npts; i++) {
		def[1] += potential[interpolation_indx[i]]*p.interpolation_wgts[i];
	}
}

void LensPixelGrid::hessian(const double x, const double y, lensmatrix<double>& hess, const int thread)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	int npts;
	lensvector<double> input_pt;
	input_pt[0] = x;
	input_pt[1] = y;

	if (grid_type==CartesianPixelGrid) {
		find_interpolation_weights_cartesian(input_pt, npts, 2, thread);
	} else {
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		find_containing_triangle(input_pt,trinum,inside_triangle,on_vertex,kmin);
		//if (!inside_triangle) {
			//if ((zero_outside_border) and (!on_vertex)) {
				//return 0;
			//} else {
				//return *triangle[trinum].sb[kmin];
			//}
		//}
		// need to implement the corresponding delaunay code for getting deflection
		die("haven't implemented this yet");
		if (qlens->natural_neighbor_interpolation) {
			find_interpolation_weights_nn(input_pt, trinum, npts, thread);
		} else {
			find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
		}
	}

	hess[0][0] = 0;
	hess[1][0] = 0;
	hess[0][1] = 0;
	hess[1][0] = 0;
	int i;
	if (npts != 32) die("SHIT! WRONG NUMBER OF POInTS");
	// NOTE: the lines below only work for the Cartesian grid
	for (i=0; i < 8; i++) {
		hess[0][0] += potential[interpolation_indx[i]]*p.interpolation_wgts[i];
	}
	for (i=8; i < 16; i++) {
		hess[1][1] += potential[interpolation_indx[i]]*p.interpolation_wgts[i];
	}
	for (i=16; i < 32; i++) {
		hess[0][1] += potential[interpolation_indx[i]]*p.interpolation_wgts[i];
	}
	hess[1][0] = hess[0][1];
}

bool LensPixelGrid::assign_mapping_flags(lensvector<double> &input_pt, vector<PtsWgts<double>>& mapped_potpixels_ij, int& n_mapped_potpixels, const int img_pixel_i, const int img_pixel_j, const int thread)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	if (grid_type==CartesianPixelGrid) {
		find_interpolation_weights_cartesian(input_pt, n_mapped_potpixels, 1, thread);
	} else {
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		find_containing_triangle(input_pt,trinum,inside_triangle,on_vertex,kmin);
		//if (!inside_triangle) {
			//// we don't want to extrapolate, because it can lead to crazy results outside the grid. so we find the closest vertex and use that vertex's SB
			//if ((zero_outside_border) and (!on_vertex)) {
				//n_mapped_srcpixels = 0;
				//return true;
			//}
			//PtsWgts pt(triptr->vertex_index[kmin],1);
			//mapped_potpixels_ij.push_back(pt);
			//maps_to_image_pixel[triptr->vertex_index[kmin]] = true;
			//n_mapped_srcpixels = 1;
		//}
		// need to implement the corresponding delaunay code for getting deflections
		die("haven't implemented this yet");
		if (qlens->natural_neighbor_interpolation) {
			find_interpolation_weights_nn(input_pt, trinum, n_mapped_potpixels, thread);
		} else {
			find_interpolation_weights_3pt(input_pt, trinum, n_mapped_potpixels, thread);
		}
	}

	PtsWgts<double> pt;
	for (int i=0; i < n_mapped_potpixels; i++) {
		maps_to_image_pixel[interpolation_indx[i]] = true;
		mapped_potpixels_ij.push_back(pt.assign(interpolation_indx[i],p.interpolation_wgts[i]));
	}
	return true;
}

void LensPixelGrid::calculate_Lmatrix(const int img_index, PtsWgts<double>* mapped_potpixels, int* n_mapped_potpixels, lensvector<double>& S0_derivs, int& index, const int& subpixel_indx, const int offset, const double weight, const int& thread)
{
	int i,j,n_halfpts;
	for (i=0; i < subpixel_indx; i++) {
		mapped_potpixels += (*n_mapped_potpixels); // each mapped image subpixel can have its own number of pixels in the potential grid it maps to, so we must skip through these to the requested subpixel index
		n_mapped_potpixels++;
	}

	//cout << "S0_deriv: " << S0_derivs[0] << " " << S0_derivs[1] << endl;
	n_halfpts = (*n_mapped_potpixels)/2;
	for (i=0,j=0; i < (*n_mapped_potpixels); i++) {
		if (i==n_halfpts) j++; // switch to y-derivative now
		// (NOTE: when we implement the Delaunay version, it won't always be npts/2. Probably will need to have a separate variable that indicates the index where we should switch to the y-deriative of S0.
		//cout << "potpixel indx: " << (offset+mapped_potpixels->indx) << " OFFSET=" << offset << endl;
		image_pixel_grid->Lmatrix_index_rows[img_index].push_back(offset + mapped_potpixels->indx); // offset because the amplitude vector has source pixels AND potential perturbation pixels
		image_pixel_grid->Lmatrix_rows[img_index].push_back(-S0_derivs[j]*weight*mapped_potpixels->wgt);
		//double harg = -S0_derivs[j]*weight*mapped_potpixels->wgt;
		//cout << "Lmatrix_sparse element (j=" << j << "): " << harg << " " << S0_derivs[j] << " " << " w=" << weight << " W=" << mapped_potpixels->wgt << endl;
		mapped_potpixels++;
	}
	index += (*n_mapped_potpixels);
}

double LensPixelGrid::first_order_surface_brightness_correction(const lensvector<double>& input_pt, const lensvector<double>& S0_deriv, const int thread)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	int npts,n_halfpts;
	if (grid_type==CartesianPixelGrid) {
		find_interpolation_weights_cartesian(input_pt, npts, 1, thread); // the '1' tells it to find the weights for the deflection
	} else {
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		find_containing_triangle(input_pt,trinum,inside_triangle,on_vertex,kmin);
		//if (!inside_triangle) {
			//// we don't want to extrapolate, because it can lead to crazy results outside the grid. so we find the closest vertex and use that vertex's SB
			//if ((zero_outside_border) and (!on_vertex)) {
				// what to do here?
			//}
		//}
		// need to implement the corresponding delaunay code for getting deflections
		die("haven't implemented this yet");
		if (qlens->natural_neighbor_interpolation) {
			find_interpolation_weights_nn(input_pt, trinum, npts, thread);
		} else {
			find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
		}
	}
	n_halfpts = npts/2;
	int i,j;
	double sb_correction=0;
	for (i=0,j=0; i < npts; i++) {
		if (i==n_halfpts) j++; // switch to y-derivative now
		sb_correction -= S0_deriv[j]*p.interpolation_wgts[i]*potential[interpolation_indx[i]];
	}
	//lensvector<double> def;
	//deflection(input_pt[0], input_pt[1], def, thread);
	//double sb_correction_check = -S0_deriv[0]*def[0]-S0_deriv[1]*def[1];
	//cout << "CHECK CORR: " << sb_correction << " " << sb_correction_check << endl;
	return sb_correction;
}

double LensPixelGrid::kappa(const double x, const double y, const int thread)
{
	lensmatrix<double> hess;
	hessian(x,y,hess,thread);

	// check!
	/*
	lensvector<double> def,defp,defm;
	double hesscheck0, hesscheck1, kapcheck;
	double h = 1e-4;
	deflection(x-h,y,defm,thread);
	deflection(x+h,y,defp,thread);
	hesscheck0 = (defp[0] - defm[0])/(2*h);
	deflection(x,y-h,defm,thread);
	deflection(x,y+h,defp,thread);
	hesscheck1 = (defp[1] - defm[1])/(2*h);
	kapcheck = (hesscheck0 + hesscheck1)/2;

	double kap = (hess[0][0] + hess[1][1])/2;
	cout << "KAPCHECK: " << kapcheck << " " << kap << endl;
	//return kapcheck;
*/

	return (hess[0][0] + hess[1][1])/2;
}

void LensPixelGrid::kappa_and_potential_derivatives(const double x, const double y, double& kappa, lensvector<double>& def, lensmatrix<double>& hess, const int thread)
{
	deflection(x,y,def,thread);
	hessian(x,y,hess,thread);
	kappa = (hess[0][0] + hess[1][1])/2;
}

void LensPixelGrid::potential_derivatives(const double x, const double y, lensvector<double>& def, lensmatrix<double>& hess, const int thread)
{
	deflection(x,y,def,thread);
	hessian(x,y,hess,thread);
}

void LensPixelGrid::find_interpolation_weights_cartesian(const lensvector<double> &input_pt, int& npts, const int deriv_type, const int thread) // bilinear interpolation
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	int ii, jj;
	double tt, uu, w00, w01, w10, w11;
	//if (input_pt[0] < lensgrid_xmin) die("input point outside of cartesian grid");
	//if (input_pt[0] > lensgrid_xmax) die("input point outside of cartesian grid");
	//if (input_pt[1] < lensgrid_ymin) die("input point outside of cartesian grid");
	//if (input_pt[1] > lensgrid_ymax) die("input point outside of cartesian grid");
	//double idoub, jdoub;
	//idoub  = ((input_pt[0]-xmin_cartesian) / cartesian_pixel_xlength);
	//jdoub  = ((input_pt[1]-ymin_cartesian) / cartesian_pixel_ylength);
	ii = (int) ((input_pt[0]-xmin_cartesian) / cartesian_pixel_xlength); // NOTE: this does not find the pixel we're in, but rather, the lower left-hand pixel of the interpolation that will be used (note that xmin_cartesian is actually the x-coordinate of the CENTER of the left-most pixel)
	jj = (int) ((input_pt[1]-ymin_cartesian) / cartesian_pixel_ylength); // NOTE: same note as above, but for y-coordinate
	//cout << "ii before fixing: " << ii << endl;
	//cout << "jj before fixing: " << jj << endl;
	//if (ii==-1) die("negative ii");
	if (ii < 0) ii=0;
	if (jj < 0) jj=0;
	if (ii >= (npix_x-1)) ii = npix_x-2;
	if (jj >= (npix_y-1)) jj = npix_y-2;
	tt = (input_pt[0]-xvals_cartesian[ii])/(xvals_cartesian[ii+1]-xvals_cartesian[ii]);
	uu = (input_pt[1]-yvals_cartesian[jj])/(yvals_cartesian[jj+1]-yvals_cartesian[jj]);
	//cout << "PINT: " << idoub << " " << jdoub << " " << ii << " " << jj << " " << input_pt[0] << " " << input_pt[1] << " " << xvals_cartesian[ii] << " " << yvals_cartesian[jj] << endl;
	//cout << "pt_x=" << input_pt[0] << " pt_y=" << input_pt[1] << " ii=" << ii << " jj=" << jj << " indx=" << cartesian_pixel_index[ii][jj] << " indx_a=" << active_index[cartesian_pixel_index[ii][jj]] << " xi=" << xvals_cartesian[ii] << " xip=" << xvals_cartesian[ii+1] << " yj=" << yvals_cartesian[jj] << " yjp=" << yvals_cartesian[jj+1] << " tt=" << tt << " uu=" << uu << endl;
	//if ((tt < 0) or (tt > 1)) die("tt=%g",tt);
	//if ((uu < 0) or (uu > 1)) die("uu=%g",uu);
	// you'll need to put in code to deal with masking, i.e. if a pixel is "inactive" (so maps_to_image_pixel==false)...ignoring this for now
	w00 = (1-tt)*(1-uu);
	w10 = tt*(1-uu);
	w01 = (1-tt)*uu;
	w11 = tt*uu;
	//cout << "WEIGHTS: " << w00 << " " << w10 << " " << w01 << " " << w11 << endl;
	if (deriv_type==0) {
		// no derivatives; just gives potential
		npts = 4;
		interpolation_indx[0] = active_index[cartesian_pixel_index[ii][jj]];
		p.interpolation_wgts[0] = w00;
		interpolation_indx[1] = active_index[cartesian_pixel_index[ii+1][jj]];
		p.interpolation_wgts[1] = w10;
		interpolation_indx[2] = active_index[cartesian_pixel_index[ii][jj+1]];
		p.interpolation_wgts[2] = w01;
		interpolation_indx[3] = active_index[cartesian_pixel_index[ii+1][jj+1]];
		p.interpolation_wgts[3] = w11;
	} else if (deriv_type==1) {
		npts = 16;
		// centered derivative along x
		double W00, W10, W01, W11, interval;
		interval = 2*cartesian_pixel_xlength;
		W00 = w00/interval;
		W10 = w10/interval;
		W01 = w01/interval;
		W11 = w11/interval;
		int ip, im;
		ip = (ii==npix_x-2) ? ii : ii+1;
		im = (ii==0) ? ii : ii-1;
		interpolation_indx[0] = active_index[cartesian_pixel_index[ip][jj]];
		p.interpolation_wgts[0] = W00;
		interpolation_indx[1] = active_index[cartesian_pixel_index[im][jj]];
		p.interpolation_wgts[1] = -W00;
		interpolation_indx[2] = active_index[cartesian_pixel_index[ip+1][jj]];
		p.interpolation_wgts[2] = W10;
		interpolation_indx[3] = active_index[cartesian_pixel_index[im+1][jj]];
		p.interpolation_wgts[3] = -W10;
		interpolation_indx[4] = active_index[cartesian_pixel_index[ip][jj+1]];
		p.interpolation_wgts[4] = W01;
		interpolation_indx[5] = active_index[cartesian_pixel_index[im][jj+1]];
		p.interpolation_wgts[5] = -W01;
		interpolation_indx[6] = active_index[cartesian_pixel_index[ip+1][jj+1]];
		p.interpolation_wgts[6] = W11;
		interpolation_indx[7] = active_index[cartesian_pixel_index[im+1][jj+1]];
		p.interpolation_wgts[7] = -W11;
		// centered derivative along y
		interval = 2*cartesian_pixel_ylength;
		W00 = w00/interval;
		W10 = w10/interval;
		W01 = w01/interval;
		W11 = w11/interval;
		int jp, jm;
		jp = (jj==npix_y-2) ? jj : jj+1;
		jm = (jj==0) ? jj : jj-1;
		interpolation_indx[8] = active_index[cartesian_pixel_index[ii][jp]];
		p.interpolation_wgts[8] = W00;
		interpolation_indx[9] = active_index[cartesian_pixel_index[ii][jm]];
		p.interpolation_wgts[9] = -W00;
		interpolation_indx[12] = active_index[cartesian_pixel_index[ii][jp+1]];
		p.interpolation_wgts[12] = W01;
		interpolation_indx[13] = active_index[cartesian_pixel_index[ii][jm+1]];
		p.interpolation_wgts[13] = -W01;
		interpolation_indx[10] = active_index[cartesian_pixel_index[ii+1][jp]];
		p.interpolation_wgts[10] = W10;
		interpolation_indx[11] = active_index[cartesian_pixel_index[ii+1][jm]];
		p.interpolation_wgts[11] = -W10;
		interpolation_indx[14] = active_index[cartesian_pixel_index[ii+1][jp+1]];
		p.interpolation_wgts[14] = W11;
		interpolation_indx[15] = active_index[cartesian_pixel_index[ii+1][jm+1]];
		p.interpolation_wgts[15] = -W11;
	} else if (deriv_type==2) {
		npts = 32;
		double W00, W10, W01, W11, interval;
		// interpolation weights for hess_xx
		interval = cartesian_pixel_xlength*cartesian_pixel_xlength;
		W00 = w00/interval;
		W10 = w10/interval;
		W01 = w01/interval;
		W11 = w11/interval;
		int ip, im;
		ip = (ii==npix_x-2) ? ii : ii+1;
		im = (ii==0) ? ii : ii-1;
		interpolation_indx[0] = active_index[cartesian_pixel_index[ip][jj]];
		p.interpolation_wgts[0] = W00-2*W10;
		interpolation_indx[1] = active_index[cartesian_pixel_index[ii][jj]];
		p.interpolation_wgts[1] = W10-2*W00;
		interpolation_indx[2] = active_index[cartesian_pixel_index[im][jj]];
		p.interpolation_wgts[2] = W00;
		interpolation_indx[3] = active_index[cartesian_pixel_index[ip+1][jj]];
		p.interpolation_wgts[3] = W10;
		interpolation_indx[4] = active_index[cartesian_pixel_index[ip][jj+1]];
		p.interpolation_wgts[4] = W01-2*W11;
		interpolation_indx[5] = active_index[cartesian_pixel_index[ii][jj+1]];
		p.interpolation_wgts[5] = W11-2*W01;
		interpolation_indx[6] = active_index[cartesian_pixel_index[im][jj+1]];
		p.interpolation_wgts[6] = W01;
		interpolation_indx[7] = active_index[cartesian_pixel_index[ip+1][jj+1]];
		p.interpolation_wgts[7] = W11;

		// interpolation weights for hess_yy
		interval = cartesian_pixel_ylength*cartesian_pixel_ylength;
		W00 = w00/interval;
		W10 = w10/interval;
		W01 = w01/interval;
		W11 = w11/interval;
		int jp, jm;
		jp = (jj==npix_y-2) ? jj : jj+1;
		jm = (jj==0) ? jj : jj-1;
		interpolation_indx[8] = active_index[cartesian_pixel_index[ii][jp]];
		p.interpolation_wgts[8] = W00-2*W01;
		interpolation_indx[9] = active_index[cartesian_pixel_index[ii][jj]];
		p.interpolation_wgts[9] = W01-2*W00;
		interpolation_indx[10] = active_index[cartesian_pixel_index[ii][jm]];
		p.interpolation_wgts[10] = W00;
		interpolation_indx[11] = active_index[cartesian_pixel_index[ii][jp+1]];
		p.interpolation_wgts[11] = W01;
		interpolation_indx[12] = active_index[cartesian_pixel_index[ii+1][jp]];
		p.interpolation_wgts[12] = W10-2*W11;
		interpolation_indx[13] = active_index[cartesian_pixel_index[ii+1][jj]];
		p.interpolation_wgts[13] = W11-2*W10;
		interpolation_indx[14] = active_index[cartesian_pixel_index[ii+1][jm]];
		p.interpolation_wgts[14] = W10;
		interpolation_indx[15] = active_index[cartesian_pixel_index[ii+1][jp+1]];
		p.interpolation_wgts[15] = W11;

		// interpolation weights for hess_xy
		interval = 4*cartesian_pixel_xlength*cartesian_pixel_ylength;
		W00 = w00/interval;
		W10 = w10/interval;
		W01 = w01/interval;
		W11 = w11/interval;
		interpolation_indx[16] = active_index[cartesian_pixel_index[im][jm]];
		p.interpolation_wgts[16] = W00;
		interpolation_indx[17] = active_index[cartesian_pixel_index[ii][jm]];
		p.interpolation_wgts[17] = W10;
		interpolation_indx[18] = active_index[cartesian_pixel_index[ip][jm]];
		p.interpolation_wgts[18] = -W00;
		interpolation_indx[19] = active_index[cartesian_pixel_index[ip+1][jm]];
		p.interpolation_wgts[19] = -W10;
		interpolation_indx[20] = active_index[cartesian_pixel_index[im][jj]];
		p.interpolation_wgts[20] = W01;
		interpolation_indx[21] = active_index[cartesian_pixel_index[ii][jj]];
		p.interpolation_wgts[21] = W11;
		interpolation_indx[22] = active_index[cartesian_pixel_index[ip][jj]];
		p.interpolation_wgts[22] = -W01;
		interpolation_indx[23] = active_index[cartesian_pixel_index[ip+1][jj]];
		p.interpolation_wgts[23] = -W11;
		interpolation_indx[24] = active_index[cartesian_pixel_index[im][jp]];
		p.interpolation_wgts[24] = -W00;
		interpolation_indx[25] = active_index[cartesian_pixel_index[ii][jp]];
		p.interpolation_wgts[25] = -W10;
		interpolation_indx[26] = active_index[cartesian_pixel_index[ip][jp]];
		p.interpolation_wgts[26] = W00;
		interpolation_indx[27] = active_index[cartesian_pixel_index[ip+1][jp]];
		p.interpolation_wgts[27] = W10;
		interpolation_indx[28] = active_index[cartesian_pixel_index[im][jp+1]];
		p.interpolation_wgts[28] = -W01;
		interpolation_indx[29] = active_index[cartesian_pixel_index[ii][jp+1]];
		p.interpolation_wgts[29] = -W11;
		interpolation_indx[30] = active_index[cartesian_pixel_index[ip][jp+1]];
		p.interpolation_wgts[30] = W01;
		interpolation_indx[31] = active_index[cartesian_pixel_index[ip+1][jp+1]];
		p.interpolation_wgts[31] = W11;
		//for (int i=0; i < 32; i++) cout << "INDX: " << interpolation_indx[i] << endl;
	}
}

int LensPixelGrid::assign_active_indices_and_count_lens_pixels(const bool activate_unmapped_pixels)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	int lens_pixel_i=0;
	for (int i=0; i < n_gridpts; i++) {
		active_pixel[i] = false;
		if ((maps_to_image_pixel[i]) or (activate_unmapped_pixels)) {
			active_pixel[i] = true;
			active_index[i] = lens_pixel_i++;
		} else {
			if ((qlens->mpi_id==0) and (qlens->regularization_method == 0)) warn(qlens->warnings,"A lensgrid pixel does not map to any image pixel (for lensgrid pixel %i), center (%g,%g)",i,p.gridpts[i][0],p.gridpts[i][1]); // only show warning if no regularization being used, since matrix cannot be inverted in that case
		}

	}
	return lens_pixel_i;
}

void LensPixelGrid::generate_hmatrices(const bool interpolate)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	if ((grid_type==DelaunayPixelGrid) and (!interpolate)) record_adjacent_triangles_xy();

	auto add_hmatrix_entry = [](ImagePixelGrid *image_pixel_grid, const int l, const int i, const int j, const double entry)
	{
		// l gives index for finite differencing (l=0: +y, l=1: -y, l=2: +x, l=3: -x)
		int dup = false;
		for (int k=0; k < image_pixel_grid->hmatrix_row_nn[l][i]; k++) {
			if (image_pixel_grid->hmatrix_index_rows[l][i][k]==j) {
				image_pixel_grid->hmatrix_rows[l][i][k] += entry;
				dup = true;
				break;
			}
		}
		if (!dup) {
			image_pixel_grid->hmatrix_rows[l][i].push_back(entry);
			image_pixel_grid->hmatrix_index_rows[l][i].push_back(j);
			image_pixel_grid->hmatrix_row_nn[l][i]++;
			image_pixel_grid->hmatrix_nn[l]++;
		}
	};

	int i,j,l;
	if (grid_type==CartesianPixelGrid) {
		int ii,jj;
		int neighbor_index[4];
		for (i=0; i < n_gridpts; i++) {
			ii = cartesian_ivals[i];
			jj = cartesian_jvals[i];
			if (ii < npix_x-1) neighbor_index[0] = cartesian_pixel_index[ii+1][jj];
			else neighbor_index[0] = -1;
			if (ii > 0) neighbor_index[1] = cartesian_pixel_index[ii-1][jj];
			else neighbor_index[1] = -1;
			if (jj < npix_y-1) neighbor_index[2] = cartesian_pixel_index[ii][jj+1];
			else neighbor_index[2] = -1;
			if (jj > 0) neighbor_index[3] = cartesian_pixel_index[ii][jj-1];
			else neighbor_index[3] = -1;
			add_hmatrix_entry(image_pixel_grid,0,i,i,-2.0);
			add_hmatrix_entry(image_pixel_grid,1,i,i,-2.0);
			for (j=0; j < 4; j++) {
				if (j > 1) l = 1;
				else l = 0;
				if (neighbor_index[j] >= 0) add_hmatrix_entry(image_pixel_grid,l,i,neighbor_index[j],1.0);
			}
		}
	} else {
		int k;
		if (interpolate) {
			int npts;
			bool inside_triangle;
			bool on_vertex;
			int trinum,kmin;
			double x,y,xp,xm,yp,ym;
			lensvector<double> interp_pt[4];
			for (i=0; i < n_gridpts; i++) {
				x = p.gridpts[i][0];
				y = p.gridpts[i][1];
				xp = x + voronoi_length[i]/2;
				xm = x - voronoi_length[i]/2;
				yp = y + voronoi_length[i]/2;
				ym = y - voronoi_length[i]/2;
				interp_pt[0].input(xp,y);
				interp_pt[1].input(xm,y);
				interp_pt[2].input(x,yp);
				interp_pt[3].input(x,ym);
				add_hmatrix_entry(image_pixel_grid,0,i,i,-2.0);
				add_hmatrix_entry(image_pixel_grid,1,i,i,-2.0);
				for (j=0; j < 4; j++) {
					if (j > 1) l = 1;
					else l = 0;
					find_containing_triangle(interp_pt[j],trinum,inside_triangle,on_vertex,kmin);
					if (!inside_triangle) {
						if (!on_vertex) continue; // assume SB = 0 outside grid
					}
					if (qlens->natural_neighbor_interpolation) {
						find_interpolation_weights_nn(interp_pt[j], trinum, npts, 0);
					} else {
						find_interpolation_weights_3pt(interp_pt[j], trinum, npts, 0);
					}
					for (k=0; k < npts; k++) {
						add_hmatrix_entry(image_pixel_grid,l,i,interpolation_indx[k],p.interpolation_wgts[k]); 
					}
				}
			}
		} else {
			int vertex_i1, vertex_i2, trinum;
			Triangle<double>* triptr;
			bool found_i1, found_i2;
			double x1, y1, x2, y2, dpt, dpt1, dpt2, dpt12;
			double length, minlength, avg_length;
			avg_length = sqrt(avg_area);
			lensvector<double> pt;
			for (i=0; i < n_gridpts; i++) {
				for (j=0; j < 4; j++) {
					if (j > 1) l = 1;
					else l = 0;
					vertex_i1 = -1;
					vertex_i2 = -1;
					if ((trinum = adj_triangles[j][i]) != -1) {
						triptr = &p.triangle[trinum];
						found_i1 = false;
						found_i2 = false;
						if ((k = triptr->vertex_index[0]) != i) { vertex_i1 = k; found_i1 = true; }
						if ((k = triptr->vertex_index[1]) != i) {
							if (!found_i1) {
								vertex_i1 = k;
								found_i1 = true;
							} else {
								vertex_i2 = k;
								found_i2 = true;
							}
						}
						if (!found_i1) die("WHAT?! couldn't find more than one vertex that isn't the one in question");
						if ((!found_i2) and ((k = triptr->vertex_index[2]) != i)) {
							vertex_i2 = k;
							found_i2 = true;
						}
						if (!found_i2) die("WHAT?! couldn't find both vertices that aren't the one in question");
					}
					if (vertex_i1 != -1) {
						x1 = p.gridpts[vertex_i1][0];
						y1 = p.gridpts[vertex_i1][1];
						x2 = p.gridpts[vertex_i2][0];
						y2 = p.gridpts[vertex_i2][1];
						if (j < 2) {
							pt[1] = p.gridpts[i][1];
							pt[0] = ((x2-x1)/(y2-y1))*(pt[1]-y1) + x1;
							dpt = abs(pt[0]-p.gridpts[i][0]);
						} else {
							pt[0] = p.gridpts[i][0];
							pt[1] = ((y2-y1)/(x2-x1))*(pt[0]-x1) + y1;
							dpt = abs(pt[1]-p.gridpts[i][1]);
						}
						dpt12 = sqrt(SQR(x2-x1) + SQR(y2-y1));
						dpt1 = sqrt(SQR(pt[0]-x1)+SQR(pt[1]-y1));
						dpt2 = sqrt(SQR(pt[0]-x2)+SQR(pt[1]-y2));
						// we scale hmatrix by the average triangle length so the regularization parameter is dimensionless
						add_hmatrix_entry(image_pixel_grid,l,i,i,-avg_length/dpt);
						add_hmatrix_entry(image_pixel_grid,l,i,vertex_i1,avg_length*dpt2/(dpt*dpt12));
						add_hmatrix_entry(image_pixel_grid,l,i,vertex_i2,avg_length*dpt1/(dpt*dpt12));
					} else {
						minlength=1e30;
						for (k=0; k < n_shared_triangles[i]; k++) {
							triptr = &p.triangle[shared_triangles[i][k]];
							length = sqrt(triptr->area);
							if (length < minlength) minlength = length;
						}
						add_hmatrix_entry(image_pixel_grid,l,i,i,-avg_length/minlength);
						//add_hmatrix_entry(image_pixel_grid,l,i,i,sqrt(1/2.0)/2);
					}
				}
			}
		}
	}
}

void LensPixelGrid::generate_gmatrices(const bool interpolate)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	if ((grid_type==DelaunayPixelGrid) and (!interpolate)) record_adjacent_triangles_xy();

	auto add_gmatrix_entry = [](ImagePixelGrid *image_pixel_grid, const int l, const int i, const int j, const double entry)
	{
		// l gives index for finite differencing (l=0: +y, l=1: -y, l=2: +x, l=3: -x)
		int dup = false;
		for (int k=0; k < image_pixel_grid->gmatrix_row_nn[l][i]; k++) {
			if (image_pixel_grid->gmatrix_index_rows[l][i][k]==j) {
				image_pixel_grid->gmatrix_rows[l][i][k] += entry;
				dup = true;
				break;
			}
		}
		if (!dup) {
			image_pixel_grid->gmatrix_rows[l][i].push_back(entry);
			image_pixel_grid->gmatrix_index_rows[l][i].push_back(j);
			image_pixel_grid->gmatrix_row_nn[l][i]++;
			image_pixel_grid->gmatrix_nn[l]++;
		}
	};


	int i,l;
	if (grid_type==CartesianPixelGrid) {
		int ii,jj;
		int neighbor_index[4];
		for (i=0; i < n_gridpts; i++) {
			ii = cartesian_ivals[i];
			jj = cartesian_jvals[i];
			if (ii < npix_x-1) neighbor_index[0] = cartesian_pixel_index[ii+1][jj];
			else neighbor_index[0] = -1;
			if (ii > 0) neighbor_index[1] = cartesian_pixel_index[ii-1][jj];
			else neighbor_index[1] = -1;
			if (jj < npix_y-1) neighbor_index[2] = cartesian_pixel_index[ii][jj+1];
			else neighbor_index[2] = -1;
			if (jj > 0) neighbor_index[3] = cartesian_pixel_index[ii][jj-1];
			else neighbor_index[3] = -1;
			for (l=0; l < 4; l++) {
				add_gmatrix_entry(image_pixel_grid,l,i,i,1.0);
				if (neighbor_index[l] >= 0) add_gmatrix_entry(image_pixel_grid,l,i,neighbor_index[l],-1.0);
			}
		}
	} else {
		int k;
		if (interpolate) {
			int npts;
			bool inside_triangle;
			bool on_vertex;
			int trinum,kmin;
			double x,y,xp,xm,yp,ym;
			lensvector<double> interp_pt[4];
			for (i=0; i < n_gridpts; i++) {
				x = p.gridpts[i][0];
				y = p.gridpts[i][1];
				xp = x + voronoi_length[i];
				xm = x - voronoi_length[i];
				yp = y + voronoi_length[i];
				ym = y - voronoi_length[i];
				interp_pt[0].input(xp,y);
				interp_pt[1].input(xm,y);
				interp_pt[2].input(x,yp);
				interp_pt[3].input(x,ym);
				for (l=0; l < 4; l++) {
					add_gmatrix_entry(image_pixel_grid,l,i,i,1.0);
					find_containing_triangle(interp_pt[l],trinum,inside_triangle,on_vertex,kmin);
					if (!inside_triangle) {
						if (!on_vertex) continue; // assume SB = 0 outside grid
					}
					if (qlens->natural_neighbor_interpolation) {
						find_interpolation_weights_nn(interp_pt[l], trinum, npts, 0);
					} else {
						find_interpolation_weights_3pt(interp_pt[l], trinum, npts, 0);
					}
					for (k=0; k < npts; k++) {
						add_gmatrix_entry(image_pixel_grid,l,i,interpolation_indx[k],-p.interpolation_wgts[k]);
					}
				}
			}
		} else {
			int vertex_i1, vertex_i2, trinum;
			Triangle<double>* triptr;
			bool found_i1, found_i2;
			double length, minlength;
			double x1, y1, x2, y2, dpt, dpt1, dpt2, dpt12;
			lensvector<double> pt;
			for (i=0; i < n_gridpts; i++) {
				for (l=0; l < 4; l++) {
					vertex_i1 = -1;
					vertex_i2 = -1;
					if ((trinum = adj_triangles[l][i]) != -1) {
						triptr = &p.triangle[trinum];
						found_i1 = false;
						found_i2 = false;
						if ((k = triptr->vertex_index[0]) != i) { vertex_i1 = k; found_i1 = true; }
						if ((k = triptr->vertex_index[1]) != i) {
							if (!found_i1) {
								vertex_i1 = k;
								found_i1 = true;
							} else {
								vertex_i2 = k;
								found_i2 = true;
							}
						}
						if (!found_i1) die("WHAT?! couldn't find more than one vertex that isn't the one in question");
						if ((!found_i2) and ((k = triptr->vertex_index[2]) != i)) {
							vertex_i2 = k;
							found_i2 = true;
						}
						if (!found_i2) die("WHAT?! couldn't find both vertices that aren't the one in question");
					}
					if (vertex_i1 != -1) {
						x1 = p.gridpts[vertex_i1][0];
						y1 = p.gridpts[vertex_i1][1];
						x2 = p.gridpts[vertex_i2][0];
						y2 = p.gridpts[vertex_i2][1];
						if (l < 2) {
							pt[1] = p.gridpts[i][1];
							pt[0] = ((x2-x1)/(y2-y1))*(pt[1]-y1) + x1;
							dpt = abs(pt[0]-p.gridpts[i][0]);
						} else {
							pt[0] = p.gridpts[i][0];
							pt[1] = ((y2-y1)/(x2-x1))*(pt[0]-x1) + y1;
							dpt = abs(pt[1]-p.gridpts[i][1]);
						}
						dpt12 = sqrt(SQR(x2-x1) + SQR(y2-y1));
						dpt1 = sqrt(SQR(pt[0]-x1)+SQR(pt[1]-y1));
						dpt2 = sqrt(SQR(pt[0]-x2)+SQR(pt[1]-y2));
						add_gmatrix_entry(image_pixel_grid,l,i,i,1.0);
						add_gmatrix_entry(image_pixel_grid,l,i,vertex_i1,-dpt2/(dpt12));
						add_gmatrix_entry(image_pixel_grid,l,i,vertex_i2,-dpt1/(dpt12));
						//add_gmatrix_entry(image_pixel_grid,l,i,i,sqrt(1/2.0)/2);
					} else {
						minlength=1e30;
						for (k=0; k < n_shared_triangles[i]; k++) {
							triptr = &p.triangle[shared_triangles[i][k]];
							length = sqrt(triptr->area);
							if (length < minlength) minlength = length;
						}
						add_gmatrix_entry(image_pixel_grid,l,i,i,1.0);
						//add_gmatrix_entry(image_pixel_grid,l,i,i,sqrt(1/2.0)/2);
					}
				}
			}
		}
	}
}

void LensPixelGrid::plot_potential(string root, const int npix, const bool interpolate_pot, const bool plot_convergence, const bool plot_fits)
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	double x, y, xlength, ylength, pixel_xlength, pixel_ylength;
	int i, j, npts_x, npts_y;
	xlength = lensgrid_xmax-lensgrid_xmin;
	ylength = lensgrid_ymax-lensgrid_ymin;
	npts_x = (int) npix*sqrt(xlength/ylength);
	npts_y = (int) npts_x*ylength/xlength;
	pixel_xlength = xlength/npts_x;
	pixel_ylength = ylength/npts_y;

	string img_filename;
	string x_filename;
	string y_filename;
	ofstream pixel_output_file;
	if (!plot_fits) {
		string img_filename = root + ".dat";
		string x_filename = root + ".x";
		string y_filename = root + ".y";

		ofstream pixel_xvals; qlens->open_output_file(pixel_xvals,x_filename);
		for (i=0, x=lensgrid_xmin; i <= npts_x; i++, x += pixel_xlength) pixel_xvals << x << endl;

		ofstream pixel_yvals; qlens->open_output_file(pixel_yvals,y_filename);
		for (i=0, y=lensgrid_ymin; i <= npts_y; i++, y += pixel_ylength) pixel_yvals << y << endl;

		qlens->open_output_file(pixel_output_file,img_filename.c_str());
	}
	int pt_i, trinum;
	double pot;
	double **potvals;
	if (plot_fits) {
		potvals = new double*[npts_x];
		for (i=0; i < npts_x; i++) potvals[i] = new double[npts_y];
	}
	//cout << "npts_x=" << npts_x << " " << xlength << " " << ylength << " " << npix << " " << lensgrid_xmax << " " << lensgrid_xmin << " " << lensgrid_ymax << " " << lensgrid_ymin << endl;
	//cout << "npts_y=" << npts_y << endl;
	lensvector<double> pt;
	for (j=0, y=lensgrid_ymin+pixel_xlength/2; j < npts_y; j++, y += pixel_ylength) {
		pt[1] = y;
		for (i=0, x=lensgrid_xmin+pixel_xlength/2; i < npts_x; i++, x += pixel_xlength) {
			pt[0] = x;
			if (interpolate_pot) {
				if (!plot_convergence) pot = interpolate_potential(x,y);
				else pot = kappa(x,y,0);
			} else {
				// The following lines will plot the Voronoi cells that are dual to the Delaunay triangulation. Note however, that when SB interpolation is
				// performed during ray-tracing, we use the vertices of the triangle that a point lands in, which may not include the closest vertex (i.e. the
				// Voronoi cell it lies in). Thus, the Voronoi cells are for visualization only, and do not directly show what the ray-traced SB will look like.
				if (grid_type==CartesianPixelGrid) {
					int ii,jj;
					ii = (int) ((pt[0]-lensgrid_xmin) / cartesian_pixel_xlength);
					jj = (int) ((pt[1]-lensgrid_ymin) / cartesian_pixel_ylength);
					pt_i = cartesian_pixel_index[ii][jj];
				} else {
					bool inside_triangle;
					trinum = search_grid(0,pt,inside_triangle); // maybe you can speed this up later by choosing a better initial triangle
					pt_i = find_closest_vertex(trinum,pt);
				}
				if (!plot_convergence) pot = potential[pt_i];
				else pot = kappa(p.gridpts[pt_i][0],p.gridpts[pt_i][1],0);
			}
			if (plot_fits) potvals[i][j] = pot;
			//cout << x << " " << y << " " << lensgridpts[pt_i][0] << " " << lensgridpts[pt_i][1] << endl;
			if (!plot_fits) pixel_output_file << pot << " ";
		}
		if (!plot_fits) pixel_output_file << endl;
	}
	//if (!plot_fits) {
		//plot_voronoi_grid(root);
	//}

	if (plot_fits) {
#ifndef USE_FITS
		cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return;
#else
		int i,j,kk;
		fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
		int status = 0;   // CFITSIO status value MUST be initialized to zero!
		int bitpix = -64, naxis = 2;
		long naxes[2] = {npts_x,npts_y};
		double *pixels;
		if (qlens->fit_output_dir != ".") qlens->create_output_directory(); // in case it hasn't been created already
		string filename = qlens->fit_output_dir + "/" + root;

		if (!fits_create_file(&outfptr, filename.c_str(), &status))
		{
			if (!fits_create_img(outfptr, bitpix, naxis, naxes, &status))
			{
				if (naxis == 0) {
					die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
				} else {
					fits_write_key(outfptr, TDOUBLE, "PXSIZE_X", &pixel_xlength, "pixel length along the x direction (in arcsec)", &status);
					fits_write_key(outfptr, TDOUBLE, "PXSIZE_Y", &pixel_ylength, "pixel length along the y direction (in arcsec)", &status);

					kk=0;
					long *fpixel = new long[naxis];
					for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
					pixels = new double[npts_x];

					for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
					{
						for (i=0; i < npts_x; i++) {
							pixels[i] = potvals[i][j];
						}
						fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
					}
					delete[] pixels;
					delete[] fpixel;
				}
			}
			fits_close_file(outfptr, &status);
		} 

		if (status) fits_report_error(stderr, status); // print any error message
		for (i=0; i < npts_x; i++) delete[] potvals[i];
		delete[] potvals;
#endif
	}
}

void LensPixelGrid::delete_lensing_arrays()
{
	if (cartesian_pixel_index != NULL) {
		for (int i=0; i < npix_x; i++) {
			delete[] cartesian_pixel_index[i];
		}
		delete[] cartesian_pixel_index;
		delete[] cartesian_ivals;
		delete[] cartesian_jvals;
		delete[] xvals_cartesian;
		delete[] yvals_cartesian;
	}
	if (potential != NULL) {
		delete[] potential;
		delete[] maps_to_image_pixel;
		delete[] active_pixel;
		delete[] active_index;
	}
}

LensPixelGrid::~LensPixelGrid()
{
	LensPixelGrid_Params<double>& p = assign_lensgrid_param_object<double>();
	if (p.param != NULL) delete[] param;
	if (cartesian_pixel_index != NULL) {
		delete_lensing_arrays();
	}
}

/******************************** Functions in class ImageData, and FITS file functions *********************************/

void ImageData::load_data(string root)
{
	string sbfilename = root + ".dat";
	string xfilename = root + ".x";
	string yfilename = root + ".y";

	int i,j,k;
	double dummy;
	if (xvals != NULL) delete[] xvals;
	if (yvals != NULL) delete[] yvals;
	if (surface_brightness != NULL) {
		for (i=0; i < npixels_x; i++) delete[] surface_brightness[i];
		delete[] surface_brightness;
	}
	if (noise_map != NULL) {
		for (i=0; i < npixels_x; i++) delete[] noise_map[i];
		delete[] noise_map;
		noise_map = NULL;
	}
	if (covinv_map != NULL) {
		for (i=0; i < npixels_x; i++) delete[] covinv_map[i];
		delete[] covinv_map;
		covinv_map = NULL;
	}
	if (high_sn_pixel != NULL) {
		for (i=0; i < npixels_x; i++) delete[] high_sn_pixel[i];
		delete[] high_sn_pixel;
	}
	if (n_mask_pixels != NULL) delete[] n_mask_pixels;
	if (extended_mask_n_neighbors != NULL) delete[] extended_mask_n_neighbors;
	if (in_mask != NULL) {
		for (k=0; k < n_masks; k++) {
			for (i=0; i < npixels_x; i++) delete[] in_mask[k][i];
			delete[] in_mask[k];
		}
		delete[] in_mask;
	}
	if (extended_mask != NULL) {
		for (k=0; k < n_masks; k++) {
			for (i=0; i < npixels_x; i++) delete[] extended_mask[k][i];
			delete[] extended_mask[k];
		}
		delete[] extended_mask;
	}
	if (foreground_mask != NULL) {
		for (i=0; i < npixels_x; i++) delete[] foreground_mask[i];
		delete[] foreground_mask;
	}

	ifstream xfile(xfilename.c_str());
	i=0;
	while (xfile >> dummy) i++;
	xfile.close();
	npixels_x = i-1;

	ifstream yfile(yfilename.c_str());
	j=0;
	while (yfile >> dummy) j++;
	yfile.close();
	npixels_y = j-1;

	xvals = new double[npixels_x+1];
	xfile.open(xfilename.c_str());
	for (i=0; i <= npixels_x; i++) xfile >> xvals[i];
	yvals = new double[npixels_y+1];
	yfile.open(yfilename.c_str());
	for (i=0; i <= npixels_y; i++) yfile >> yvals[i];

	pixel_xcvals = new double[npixels_x];
	pixel_ycvals = new double[npixels_y];
	for (i=0; i < npixels_x; i++) {
		pixel_xcvals[i] = (xvals[i]+xvals[i+1])/2;
	}
	for (i=0; i < npixels_y; i++) {
		pixel_ycvals[i] = (yvals[i]+yvals[i+1])/2;
	}

	ifstream sbfile(sbfilename.c_str());
	surface_brightness = new double*[npixels_x];
	high_sn_pixel = new bool*[npixels_x];
	n_masks = 1;
	n_mask_pixels = new int[1];
	extended_mask_n_neighbors = new int[1];
	extended_mask_n_neighbors[0] = -1; // this means all the pixels are included in the extended mask by default
	n_mask_pixels[0] = npixels_x*npixels_y;
	n_high_sn_pixels = n_mask_pixels[0]; // this will be recalculated in assign_high_sn_pixels() function
	in_mask = new bool**[1];
	in_mask[0] = new bool*[npixels_x];
	extended_mask = new bool**[1];
	extended_mask[0] = new bool*[npixels_x];
	foreground_mask = new bool*[npixels_x];
	foreground_mask_data = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) {
		surface_brightness[i] = new double[npixels_y];
		high_sn_pixel[i] = new bool[npixels_y];
		in_mask[0][i] = new bool[npixels_y];
		extended_mask[0][i] = new bool[npixels_y];
		foreground_mask[i] = new bool[npixels_y];
		foreground_mask_data[i] = new bool[npixels_y];
		for (j=0; j < npixels_y; j++) {
			in_mask[0][i][j] = true;
			extended_mask[0][i][j] = true;
			foreground_mask[i][j] = true;
			foreground_mask_data[i][j] = true;
			high_sn_pixel[i][j] = true;
			sbfile >> surface_brightness[i][j];
		}
	}
	find_extended_mask_rmax(); // used when splining integrals for deflection/hessian from Fourier modes
	assign_high_sn_pixels();
}

void ImageData::load_from_image_grid(ImagePixelGrid* image_pixel_grid)
{
	int i,j,k;
	if ((npixels_x != image_pixel_grid->x_N) or (npixels_y != image_pixel_grid->y_N)) {
		if (xvals != NULL) delete[] xvals;
		if (yvals != NULL) delete[] yvals;
		if (pixel_xcvals != NULL) delete[] pixel_xcvals;
		if (pixel_ycvals != NULL) delete[] pixel_ycvals;
		if (surface_brightness != NULL) {
			for (i=0; i < npixels_x; i++) delete[] surface_brightness[i];
			delete[] surface_brightness;
		}
		if (noise_map != NULL) {
			for (i=0; i < npixels_x; i++) delete[] noise_map[i];
			delete[] noise_map;
			noise_map = NULL;
		}
		if (covinv_map != NULL) {
			for (i=0; i < npixels_x; i++) delete[] covinv_map[i];
			delete[] covinv_map;
			covinv_map = NULL;
		}
		if (high_sn_pixel != NULL) {
			for (i=0; i < npixels_x; i++) delete[] high_sn_pixel[i];
			delete[] high_sn_pixel;
		}
		if (n_mask_pixels != NULL) delete[] n_mask_pixels;
		if (extended_mask_n_neighbors != NULL) delete[] extended_mask_n_neighbors;
		if (in_mask != NULL) {
			for (k=0; k < n_masks; k++) {
				for (i=0; i < npixels_x; i++) delete[] in_mask[k][i];
				delete[] in_mask[k];
			}
			delete[] in_mask;
		}
		if (extended_mask != NULL) {
			for (k=0; k < n_masks; k++) {
				for (i=0; i < npixels_x; i++) delete[] extended_mask[k][i];
				delete[] extended_mask[k];
			}
			delete[] extended_mask;
		}
		if (foreground_mask != NULL) {
			for (i=0; i < npixels_x; i++) delete[] foreground_mask[i];
			delete[] foreground_mask;
		}
		if (qlens != NULL) qlens->reset_PSF_convolution_plans(); // since number of image pixels has changed, will need to redo psf convolution plans
		if ((qlens != NULL) and (qlens->fft_convolution)) qlens->cleanup_FFT_convolution_arrays(); // since number of image pixels has changed, will need to redo FFT setup

		npixels_x = image_pixel_grid->x_N;
		npixels_y = image_pixel_grid->y_N;

		xvals = new double[npixels_x+1];
		yvals = new double[npixels_y+1];

		pixel_xcvals = new double[npixels_x];
		pixel_ycvals = new double[npixels_y];

		surface_brightness = new double*[npixels_x];
		high_sn_pixel = new bool*[npixels_x];
		n_masks = 1;
		n_mask_pixels = new int[1];
		extended_mask_n_neighbors = new int[1];

		in_mask = new bool**[1];
		in_mask[0] = new bool*[npixels_x];
		extended_mask = new bool**[1];
		extended_mask[0] = new bool*[npixels_x];
		foreground_mask = new bool*[npixels_x];
		foreground_mask_data = new bool*[npixels_x];
		noise_map = new double*[npixels_x];
		covinv_map = new double*[npixels_x];
		for (i=0; i < npixels_x; i++) {
			surface_brightness[i] = new double[npixels_y];
			high_sn_pixel[i] = new bool[npixels_y];
			in_mask[0][i] = new bool[npixels_y];
			extended_mask[0][i] = new bool[npixels_y];
			foreground_mask[i] = new bool[npixels_y];
			foreground_mask_data[i] = new bool[npixels_y];
			noise_map[i] = new double[npixels_y];
			covinv_map[i] = new double[npixels_y];
			for (j=0; j < npixels_y; j++) {
				in_mask[0][i][j] = true;
				extended_mask[0][i][j] = true;
				foreground_mask[i][j] = true;
				foreground_mask_data[i][j] = true;
				high_sn_pixel[i][j] = true;
				noise_map[i][j] = 1e30; // since we don't have a noise map yet
				covinv_map[i][j] = 0.0; // since we don't have a noise map yet
			}
		}
	}

	xmin = image_pixel_grid->xmin;
	xmax = image_pixel_grid->xmax;
	ymin = image_pixel_grid->ymin;
	ymax = image_pixel_grid->ymax;

	for (i=0; i <= npixels_x; i++) xvals[i] = image_pixel_grid->corner_pts[i][0][0];
	for (i=0; i <= npixels_y; i++) yvals[i] = image_pixel_grid->corner_pts[0][i][1];

	for (i=0; i < npixels_x; i++) {
		pixel_xcvals[i] = (xvals[i]+xvals[i+1])/2;
	}
	for (i=0; i < npixels_y; i++) {
		pixel_ycvals[i] = (yvals[i]+yvals[i+1])/2;
	}

	double xstep = pixel_xcvals[1] - pixel_xcvals[0];
	double ystep = pixel_ycvals[1] - pixel_ycvals[0];
	pixel_size = dmin(xstep,ystep);

	n_mask_pixels[0] = npixels_x*npixels_y;
	extended_mask_n_neighbors[0] = -1; // this means all the pixels are included in the extended mask by default
	n_high_sn_pixels = n_mask_pixels[0]; // this will be recalculated in assign_high_sn_pixels() function
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			high_sn_pixel[i][j] = true;
			surface_brightness[i][j] = image_pixel_grid->foreground_surface_brightness[i][j] + image_pixel_grid->surface_brightness[i][j];
		}
	}
	find_extended_mask_rmax(); // used when splining integrals for deflection/hessian from Fourier modes
	assign_high_sn_pixels();
}

bool ImageData::load_data_fits(string fits_filename, const double pixel_size_in, const double pixel_xy_ratio_in, const double x_offset, const double y_offset, const int hdu_indx, const bool show_header)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else
	pixel_size = pixel_size_in;
	pixel_xy_ratio = pixel_xy_ratio_in;
	bool image_load_status = false;
	int i,j,k,kk;
	fitsfile *fptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix, naxis;
	long naxes[2] = {1,1};
	double *pixels;
	double x, y, xstep, ystep;
	bool pixel_noise_specified = false;
	if ((qlens != NULL) and (qlens->background_pixel_noise > 0)) pixel_noise_specified = true;
	double pnoise;

	int hdutype;
	if (!fits_open_file(&fptr, fits_filename.c_str(), READONLY, &status))
	{
		if (fits_movabs_hdu(fptr, hdu_indx, &hdutype, &status)) // move to HDU given by hdu_indx
			 return false;

		if (xvals != NULL) delete[] xvals;
		if (yvals != NULL) delete[] yvals;
		if (surface_brightness != NULL) {
			for (i=0; i < npixels_x; i++) delete[] surface_brightness[i];
			delete[] surface_brightness;
		}
		if (high_sn_pixel != NULL) {
			for (i=0; i < npixels_x; i++) delete[] high_sn_pixel[i];
			delete[] high_sn_pixel;
		}
		int nkeys;
		fits_get_hdrspace(fptr, &nkeys, NULL, &status); // get # of keywords

		int ii;
		char card[FLEN_CARD];   // Standard string lengths defined in fitsio.h

		bool reading_qlens_comment = false;
		bool reading_markers = false;
		bool pixel_size_found = false;
		int pos, pos1;
		for (ii = 1; ii <= nkeys; ii++) { // Read and print each keywords 
			if (fits_read_record(fptr, ii, card, &status))break;
			string cardstring(card);
			if ((show_header) and ((qlens==NULL) or (qlens->mpi_id==0))) cout << cardstring << endl;
			if (reading_qlens_comment) {
				if ((pos = cardstring.find("COMMENT")) != string::npos) {
					if (((pos1 = cardstring.find("mk: ")) != string::npos) or ((pos1 = cardstring.find("MK: ")) != string::npos)) {
						reading_markers = true;
						reading_qlens_comment = false;
						if (qlens != NULL) qlens->param_markers = cardstring.substr(pos1+4);
					} else {
						if (qlens != NULL) qlens->data_info += cardstring.substr(pos+8);
					}
				} else break;
			} else if (reading_markers) {
				if ((pos = cardstring.find("COMMENT")) != string::npos) {
					if (qlens != NULL) qlens->param_markers += cardstring.substr(pos+8);
					// A potential issue is that if there are enough markers to fill more than one line, there might be an extra space inserted,
					// in which case the markers won't come out properly. No time to deal with this now, but something to look out for.
				} else break;
			} else if (((pos = cardstring.find("ql: ")) != string::npos) or ((pos = cardstring.find("QL: ")) != string::npos)) {
				reading_qlens_comment = true;
				if (qlens != NULL) qlens->data_info = cardstring.substr(pos+4);
			} else if (((pos = cardstring.find("mk: ")) != string::npos) or ((pos = cardstring.find("MK: ")) != string::npos)) {
				reading_markers = true;
				if (qlens != NULL) qlens->param_markers = cardstring.substr(pos+4);
			} else if ((cardstring.find("PXSIZE ") != string::npos) or (cardstring.find("PIXSCL ") != string::npos)) {
				string pxsize_string = cardstring.substr(11);
				stringstream pxsize_str;
				pxsize_str << pxsize_string;
				pxsize_str >> pixel_size;
				pixel_size_found = true;
			} else if (cardstring.find("PXNOISE ") != string::npos) {
				string pxnoise_string = cardstring.substr(11);
				stringstream pxnoise_str;
				pxnoise_str << pxnoise_string;
				pxnoise_str >> pnoise;
				if (qlens != NULL) {
					qlens->background_pixel_noise = pnoise;
					pixel_noise_specified = true;
				}
			//} else if (cardstring.find("PSFSIG ") != string::npos) {
				//string psfwidth_string = cardstring.substr(11);
				//stringstream psfwidth_str;
				//psfwidth_str << psfwidth_string;
				//if (psf != NULL) psfwidth_str >> psf->psf_width_x;
				//if (psf != NULL) psf->psf_width_y = psf->psf_width_x;
			} else if (cardstring.find("ZSRC ") != string::npos) {
				string zsrc_string = cardstring.substr(11);
				stringstream zsrc_str;
				zsrc_str << zsrc_string;
				if (qlens != NULL) {
					zsrc_str >> qlens->source_redshift;
					if (qlens->auto_zsource_scaling) qlens->reference_source_redshift = qlens->source_redshift;
				}
			} else if (cardstring.find("ZLENS ") != string::npos) {
				string zlens_string = cardstring.substr(11);
				stringstream zlens_str;
				zlens_str << zlens_string;
				if (qlens != NULL) zlens_str >> qlens->lens_redshift;
			}
		}
		if ((reading_markers) and (qlens != NULL)) {
			// Commas are used in FITS file as delimeter so spaces don't get lost; now convert to spaces again
			for (size_t i = 0; i < qlens->param_markers.size(); ++i) {
				 if (qlens->param_markers[i] == ',') {
					  qlens->param_markers.replace(i, 1, " ");
				 }
			}
		}

		bool new_dimensions = true;
		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) )
		{
			if (naxis == 0) {
				warn("only 1D or 2D images are supported (dimension is %i for hdu=%i)\n",naxis,hdu_indx);
				surface_brightness = NULL;
				xvals = NULL;
				yvals = NULL;
				high_sn_pixel = NULL;
				return false;
			} else {
				kk=0;
				long *fpixel = new long[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;

				if ((npixels_x != naxes[0]) or (npixels_y != naxes[1])) {
					if (n_mask_pixels != NULL) delete[] n_mask_pixels;
					if (extended_mask_n_neighbors != NULL) delete[] extended_mask_n_neighbors;
					if (in_mask != NULL) {
						for (k=0; k < n_masks; k++) {
							for (i=0; i < npixels_x; i++) delete[] in_mask[k][i];
							delete[] in_mask[k];
						}
						delete[] in_mask;
					}
					if (extended_mask != NULL) {
						for (k=0; k < n_masks; k++) {
							for (i=0; i < npixels_x; i++) delete[] extended_mask[k][i];
							delete[] extended_mask[k];
						}
						delete[] extended_mask;
					}
					if (foreground_mask != NULL) {
						for (i=0; i < npixels_x; i++) delete[] foreground_mask[i];
						delete[] foreground_mask;
					}
					if (foreground_mask_data != NULL) {
						for (i=0; i < npixels_x; i++) delete[] foreground_mask_data[i];
						delete[] foreground_mask_data;
					}
					if (noise_map != NULL) {
						for (i=0; i < npixels_x; i++) delete[] noise_map[i];
						delete[] noise_map;
					}
					if (covinv_map != NULL) {
						for (i=0; i < npixels_x; i++) delete[] covinv_map[i];
						delete[] covinv_map;
					}

					if (qlens != NULL) qlens->reset_PSF_convolution_plans(); // since number of image pixels has changed, will need to redo psf convolution plans
					if ((qlens != NULL) and (qlens->fft_convolution)) qlens->cleanup_FFT_convolution_arrays(); // since number of image pixels has changed, will need to redo FFT setup

					npixels_x = naxes[0];
					npixels_y = naxes[1];

					n_masks = 1;
					n_mask_pixels = new int[1];
					n_mask_pixels[0] = npixels_x*npixels_y;
					extended_mask_n_neighbors = new int[1];
					extended_mask_n_neighbors[0] = -1; // this means all the pixels are included in the extended mask by default
				} else {
					new_dimensions = false;
				}

				n_high_sn_pixels = npixels_x*npixels_y; // this will be recalculated in assign_high_sn_pixels() function

				xvals = new double[npixels_x+1];
				yvals = new double[npixels_y+1];
				if (pixel_size > 0) {
					xstep = pixel_size;
					ystep = pixel_xy_ratio*pixel_size;
					xmax = npixels_x*pixel_size/2;
					ymax = npixels_y*pixel_size/2;
					xmin=-xmax; ymin=-ymax;
				} else {
					if (((xmax-xmin)==0.0) or ((ymax-ymin)==0.0)) die("data pixel size is unspecified, and imgdata grid dimensions have not been set; cannot load data");
					warn("no pixel size found in FITS header, and data pixel size is unspecified; using 'grid' setting to set pixel size");
					xstep = (xmax-xmin)/npixels_x;
					ystep = (ymax-ymin)/npixels_y;
					pixel_size = xstep;
					pixel_xy_ratio = ystep/xstep;
				}
				xmin -= x_offset;
				ymin -= y_offset;
				for (i=0, x=xmin; i <= npixels_x; i++, x += xstep) xvals[i] = x;
				for (i=0, y=ymin; i <= npixels_y; i++, y += ystep) yvals[i] = y;
				xmax = xvals[npixels_x]; // just in case there was any machine error that crept in
				ymax = yvals[npixels_y]; // just in case there was any machine error that crept in
				pixels = new double[npixels_x];
				surface_brightness = new double*[npixels_x];
				high_sn_pixel = new bool*[npixels_x];
				if (new_dimensions) {
					n_masks = 1;
					in_mask = new bool**[1];
					in_mask[0] = new bool*[npixels_x];
					extended_mask = new bool**[1];
					extended_mask[0] = new bool*[npixels_x];

					foreground_mask = new bool*[npixels_x];
					foreground_mask_data = new bool*[npixels_x];
					noise_map = new double*[npixels_x];
					covinv_map = new double*[npixels_x];
				}
				for (i=0; i < npixels_x; i++) {
					surface_brightness[i] = new double[npixels_y];
					high_sn_pixel[i] = new bool[npixels_y];
					if (new_dimensions) {
						in_mask[0][i] = new bool[npixels_y];
						extended_mask[0][i] = new bool[npixels_y];
						foreground_mask[i] = new bool[npixels_y];
						foreground_mask_data[i] = new bool[npixels_y];
						noise_map[i] = new double[npixels_y];
						covinv_map[i] = new double[npixels_y];
						for (j=0; j < npixels_y; j++) {
							in_mask[0][i][j] = true;
							extended_mask[0][i][j] = true;
							foreground_mask[i][j] = true;
							foreground_mask_data[i][j] = true;
						}
					}
					for (j=0; j < npixels_y; j++) {
						high_sn_pixel[i][j] = true;
					}
				}
				if ((new_dimensions) and (pixel_noise_specified)) {
					set_uniform_pixel_noise(pnoise);
				}

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					if (fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0], NULL, pixels, NULL, &status) )  // read row of pixels
						break; // jump out of loop on error

					for (i=0; i < naxes[0]; i++) {
						surface_brightness[i][j] = pixels[i];
					}
				}
				delete[] pixels;
				delete[] fpixel;
				image_load_status = true;
			}
		}
		fits_close_file(fptr, &status);

		pixel_xcvals = new double[npixels_x];
		pixel_ycvals = new double[npixels_y];
		for (i=0; i < npixels_x; i++) {
			pixel_xcvals[i] = (xvals[i]+xvals[i+1])/2;
		}
		for (i=0; i < npixels_y; i++) {
			pixel_ycvals[i] = (yvals[i]+yvals[i+1])/2;
		}
	}

	if (status) fits_report_error(stderr, status); // print any error message
	if (image_load_status) {
		data_fits_filename = fits_filename;
		assign_high_sn_pixels();
	}
	return image_load_status;
#endif
}

void ImageData::save_data_fits(string fits_filename, const bool subimage, const double xmin_in, const double xmax_in, const double ymin_in, const double ymax_in)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return;
#else
	int i,j,kk;
	fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix = -64, naxis = 2;
	int min_i=0, min_j=0, max_i=npixels_x-1, max_j=npixels_y-1;
	if (subimage) {
		min_i = (int) ((xmin_in-xmin) * npixels_x / (xmax-xmin));
		if (min_i < 0) min_i = 0;
		max_i = (int) ((xmax_in-xmin) * npixels_x / (xmax-xmin));
		if (max_i > (npixels_x-1)) max_i = npixels_x-1;
		min_j = (int) ((ymin_in-ymin) * npixels_y / (ymax-ymin));
		if (min_j < 0) min_j = 0;
		max_j = (int) ((ymax_in-ymin) * npixels_y / (ymax-ymin));
		if (max_j > (npixels_y-1)) max_j = npixels_y-1;
	}
	int npix_x = max_i-min_i+1;
	int npix_y = max_j-min_j+1;
	cout << "imin=" << min_i << " imax=" << max_i << " jmin=" << min_j << " jmax=" << max_j << " npix_x=" << npix_x << " npix_y=" << npix_y << endl;
	long naxes[2] = {npix_x,npix_y};
	double *pixels;
	string fits_filename_overwrite = "!" + fits_filename; // ensures that it overwrites an existing file of the same name

	if (!fits_create_file(&outfptr, fits_filename_overwrite.c_str(), &status))
	{
		if (!fits_create_img(outfptr, bitpix, naxis, naxes, &status))
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long *fpixel = new long[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[npix_x];

				for (fpixel[1]=1, j=min_j; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=min_i, kk=0; i <= max_i; i++, kk++) {
						pixels[kk] = surface_brightness[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] fpixel;
				delete[] pixels;
			}
			if (pixel_size > 0)
				fits_write_key(outfptr, TDOUBLE, "PXSIZE", &pixel_size, "length of square pixels (in arcsec)", &status);
			if (qlens->background_pixel_noise != 0)
				fits_write_key(outfptr, TDOUBLE, "PXNOISE", &qlens->background_pixel_noise, "pixel surface brightness noise", &status);
			if (qlens->data_info != "") {
				string comment = "ql: " + qlens->data_info;
				fits_write_comment(outfptr, comment.c_str(), &status);
			}
		}
		fits_close_file(outfptr, &status);
	} 

	if (status) fits_report_error(stderr, status); // print any error message
#endif
}

bool ImageData::load_noise_map_fits(string fits_filename, const int hdu_indx, const bool show_header)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else

	string filename = fits_filename;
#if __cplusplus >= 201703L // C++17 standard or later
		if (!filesystem::exists(filename)) {
			filename = "../data/" + fits_filename; // try the data folder
			if (!filesystem::exists(filename)) {
				if (qlens->fit_output_dir != ".") {
					filename = qlens->fit_output_dir + "/" + fits_filename;  // finally, try the chains folder
					if (!filesystem::exists(filename)) return false;
				} else return false;
			}
		}
#endif
	if ((qlens->mpi_id==0) and (filename != fits_filename)) {
		cout << "Loading file '" << filename << "'" << endl;
	}

	bool image_load_status = false;
	int i,j,kk;
	fitsfile *fptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix, naxis;
	long naxes[2] = {1,1};
	double *pixels;

	if (noise_map == NULL) {
		noise_map = new double*[npixels_x];
		for (i=0; i < npixels_x; i++) noise_map[i] = new double[npixels_y];
	}
	if (covinv_map == NULL) {
		covinv_map = new double*[npixels_x];
		for (i=0; i < npixels_x; i++) covinv_map[i] = new double[npixels_y];
	}

	bg_pixel_noise = 1e30;
	char card[FLEN_CARD];   // Standard string lengths defined in fitsio.h
	int hdutype;
	if (!fits_open_file(&fptr, filename.c_str(), READONLY, &status))
	{
		if (fits_movabs_hdu(fptr, hdu_indx, &hdutype, &status)) // move to HDU given by hdu_indx
			return false;
		int nkeys;
		fits_get_hdrspace(fptr, &nkeys, NULL, &status); // get # of keywords
		if (show_header) {
			for (int ii = 1; ii <= nkeys; ii++) { // Read and print each keywords 
				if (fits_read_record(fptr, ii, card, &status))break;
				string cardstring(card);
				cout << cardstring << endl;
			}
		}

		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) )
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long* fpixel = new long[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				if ((npixels_x == naxes[0]) or (npixels_y == naxes[1])) {
					pixels = new double[npixels_x];
					noise_map = new double*[npixels_x];
					for (i=0; i < npixels_x; i++) {
						noise_map[i] = new double[npixels_y];
					}

					for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
					{
						if (fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0], NULL, pixels, NULL, &status) )  // read row of pixels
							break; // jump out of loop on error

						for (i=0; i < naxes[0]; i++) {
							noise_map[i][j] = pixels[i];
							//cout << "NOISE(" << i << "," << j << ")=" << pixels[i] << endl;
							if (pixels[i] < bg_pixel_noise) bg_pixel_noise = pixels[i];
						}
					}
					delete[] pixels;
					image_load_status = true;
				} else {
					warn("noise map does not have same dimensions as data image");
					image_load_status = false;
				}
				for (i=0; i < npixels_x; i++) {
					for (j=0; j < npixels_y; j++) {
						covinv_map[i][j] = 1.0/SQR(noise_map[i][j]);
					}
				}
				delete[] fpixel;
			}
		}
		fits_close_file(fptr, &status);
	}
	if (qlens != NULL) qlens->background_pixel_noise = bg_pixel_noise; // store the background noise separately

	if (status) fits_report_error(stderr, status); // print any error message
	if (image_load_status) noise_map_fits_filename = filename;
	return image_load_status;
#endif
}

bool ImageData::save_noise_map_fits(string fits_filename, const bool subimage, const double xmin_in, const double xmax_in, const double ymin_in, const double ymax_in)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n";
	return false;
#else
	if (noise_map == NULL) {
		warn("no noise map has been loaded or generated; cannot save noise map to FITS file");
		return false;
	}

	int i,j,kk;
	fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix = -64, naxis = 2;
	int min_i=0, min_j=0, max_i=npixels_x-1, max_j=npixels_y-1;
	if (subimage) {
		min_i = (int) ((xmin_in-xmin) * npixels_x / (xmax-xmin));
		if (min_i < 0) min_i = 0;
		max_i = (int) ((xmax_in-xmin) * npixels_x / (xmax-xmin));
		if (max_i > (npixels_x-1)) max_i = npixels_x-1;
		min_j = (int) ((ymin_in-ymin) * npixels_y / (ymax-ymin));
		if (min_j < 0) min_j = 0;
		max_j = (int) ((ymax_in-ymin) * npixels_y / (ymax-ymin));
		if (max_j > (npixels_y-1)) max_j = npixels_y-1;
	}
	int npix_x = max_i-min_i+1;
	int npix_y = max_j-min_j+1;
	cout << "imin=" << min_i << " imax=" << max_i << " jmin=" << min_j << " jmax=" << max_j << " npix_x=" << npix_x << " npix_y=" << npix_y << endl;

	long naxes[2] = {npixels_x,npixels_y};
	double *pixels;
	string fits_filename_overwrite = "!" + fits_filename; // ensures that it overwrites an existing file of the same name

	if (!fits_create_file(&outfptr, fits_filename_overwrite.c_str(), &status))
	{
		if (!fits_create_img(outfptr, bitpix, naxis, naxes, &status))
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long* fpixel = new long[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[npix_x];

				for (fpixel[1]=1, j=min_j; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=min_i, kk=0; i <= max_i; i++, kk++) {
						pixels[kk] = noise_map[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] fpixel;
				delete[] pixels;
			}
			if (pixel_size > 0)
				fits_write_key(outfptr, TDOUBLE, "PXSIZE", &pixel_size, "length of square pixels (in arcsec)", &status);
		}
		fits_close_file(outfptr, &status);
	} 

	if (status) fits_report_error(stderr, status); // print any error message
	noise_map_fits_filename = fits_filename;
	return true;
#endif
}

void ImageData::unload_noise_map()
{
	int i;
	if (noise_map == NULL) {
		noise_map = new double*[npixels_x];
		for (i=0; i < npixels_x; i++) noise_map[i] = new double[npixels_y];
	}
	if (covinv_map == NULL) {
		covinv_map = new double*[npixels_x];
		for (i=0; i < npixels_x; i++) covinv_map[i] = new double[npixels_y];
	}
	noise_map_fits_filename = "";
}

void ImageData::get_grid_params(double& xmin_in, double& xmax_in, double& ymin_in, double& ymax_in, int& npx, int& npy)
{
	if (xvals==NULL) die("cannot get image pixel data parameters; no data has been loaded");
	xmin_in = xvals[0];
	xmax_in = xvals[npixels_x];
	ymin_in = yvals[0];
	ymax_in = yvals[npixels_y];
	npx = npixels_x;
	npy = npixels_y;
}

void ImageData::assign_high_sn_pixels() // should probably use the foreground mask here, since we might be masking out foreground stars. Implement!!
{
	double global_max_sb = -1e30;
	int i,j;
	for (j=0; j < npixels_y; j++) {
		for (i=0; i < npixels_x; i++) {
			if (surface_brightness[i][j] > global_max_sb) global_max_sb = surface_brightness[i][j];
		}
	}
	if (qlens != NULL) {
		n_high_sn_pixels = 0;
		for (j=0; j < npixels_y; j++) {
			for (i=0; i < npixels_x; i++) {
				if (surface_brightness[i][j] >= qlens->high_sn_frac*global_max_sb) {
					high_sn_pixel[i][j] = true;
					n_high_sn_pixels++;
				}
				else high_sn_pixel[i][j] = false;
			}
		}
	}
}

double ImageData::find_max_sb(const int mask_k)
{
	double max_sb = -1e30;
	int i,j;
	for (j=0; j < npixels_y; j++) {
		for (i=0; i < npixels_x; i++) {
			if ((in_mask[mask_k][i][j]) and (surface_brightness[i][j] > max_sb)) max_sb = surface_brightness[i][j];
		}
	}
	return max_sb;
}

double ImageData::find_avg_sb(const double sb_threshold, const int mask_k)
{
	double avg_sb=0;
	int npix=0;
	int i,j;
	for (j=0; j < npixels_y; j++) {
		for (i=0; i < npixels_x; i++) {
			if ((in_mask[mask_k][i][j]) and (surface_brightness[i][j] > sb_threshold)) {
				avg_sb += surface_brightness[i][j];
				npix++;
			}
		}
	}
	avg_sb /= npix;

	return avg_sb;
}

bool ImageData::load_mask_fits(const int mask_k, const string fits_filename, const bool foreground, const bool emask, const bool add_mask_pixels, const bool subtract_mask_pixels) // if 'add_mask_pixels' is true, then doesn't unmask any pixels that are already masked
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else

	string filename = fits_filename;
#if __cplusplus >= 201703L // C++17 standard or later
		if (!filesystem::exists(filename)) {
			filename = "../data/" + fits_filename; // try the data folder
			if (!filesystem::exists(filename)) {
				if (qlens->fit_output_dir != ".") {
					filename = qlens->fit_output_dir + "/" + fits_filename;  // finally, try the chains folder
					if (!filesystem::exists(filename)) return false;
				} else return false;
			}
		}
#endif
	if ((qlens->mpi_id==0) and (filename != fits_filename)) {
		cout << "Loading file '" << filename << "'" << endl;
	}



	if (n_masks==0) { warn("no mask arrays have been initialized, indicating image data has not been loaded"); return false; }
	if (mask_k > n_masks) die("cannot add mask whose index is greater than the number of masks; to add a new mask, set index = n_masks");
	bool image_load_status = false;
	int i,j,iprime,jprime,k,kk;

	fitsfile *fptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix, naxis;
	long naxes[2] = {1,1};
	double *pixels;
	int n_maskpixels = 0;
	bool new_mask = false;
	int offset_x=0;
	int offset_y=0;

	if (!fits_open_file(&fptr, filename.c_str(), READONLY, &status))
	{
		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) )
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long* fpixel = new long[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				if ((naxes[0] != npixels_x) or (naxes[1] != npixels_y)) {
					if ((naxes[0] < npixels_x) and (naxes[1] < npixels_y)) {
						offset_x = ((npixels_x-naxes[0])/2);
						offset_y = ((npixels_y-naxes[1])/2);
						warn("number of pixels in mask file is less than umber of pixels in load data; padding the mask image accordingly");
					} else {
						cout << "Error: number of pixels in mask file is greater than the number of pixels in loaded data (along x and/or y)\n";
						return false;
					}
				}
				if (mask_k==n_masks) {
					new_mask = true;
					bool ***new_masks = new bool**[n_masks+1];
					bool ***new_extended_masks = new bool**[n_masks+1];
					int *new_n_mask_pixels = new int[n_masks+1];
					int *new_extended_mask_n_neighbors = new int[n_masks+1];
					for (k=0; k < n_masks; k++) {
						new_masks[k] = in_mask[k];
						new_extended_masks[k] = extended_mask[k];
						new_n_mask_pixels[k] = n_mask_pixels[k];
						new_extended_mask_n_neighbors[k] = extended_mask_n_neighbors[k];
					}
					new_masks[n_masks] = new bool*[npixels_x];
					new_extended_masks[n_masks] = new bool*[npixels_x];
					for (i=0; i < npixels_x; i++) {
						new_masks[n_masks][i] = new bool[npixels_y];
						new_extended_masks[n_masks][i] = new bool[npixels_y];
						for (j=0; j < npixels_y; j++) {
							new_masks[n_masks][i][j] = false;
							new_extended_masks[n_masks][i][j] = false;
						}
					}
					delete[] in_mask;
					delete[] extended_mask;
					delete[] n_mask_pixels;
					delete[] extended_mask_n_neighbors;
					in_mask = new_masks;
					extended_mask = new_extended_masks;
					n_mask_pixels = new_n_mask_pixels;
					extended_mask_n_neighbors = new_extended_mask_n_neighbors;
					n_masks++;
				}
				pixels = new double[npixels_x];
				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					if (fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0], NULL, pixels, NULL, &status) )  // read row of pixels
						break; // jump out of loop on error

					jprime = j + offset_y;
					for (i=0; i < naxes[0]; i++) {
						iprime = i + offset_x;
						if (foreground) {
							if (pixels[i] == 0.0) {
								if (!add_mask_pixels) foreground_mask[iprime][jprime] = false;
								else if (foreground_mask[iprime][jprime]==true) n_maskpixels++;
								// Don't allow the primary mask to include pixels that the foreground mask doesn't.
								if (in_mask[mask_k][iprime][jprime] == true) {
									in_mask[mask_k][iprime][jprime] = false;
									n_mask_pixels[mask_k]--;
								}
								if (extended_mask[mask_k][iprime][jprime] == true) {
									extended_mask[mask_k][iprime][jprime] = false;
								}
							}
							else {
								foreground_mask[iprime][jprime] = true;
								n_maskpixels++;
								if (new_mask) {
									in_mask[mask_k][iprime][jprime] = true; // if new mask was created, then mask contains all the pixels by default
									extended_mask[mask_k][iprime][jprime] = true; // if new mask was created, then extended mask contains all the pixels by default
								}
								//cout << pixels[iprime] << endl;
							}
							foreground_mask_data[iprime][jprime] = foreground_mask[iprime][jprime];
						} else if (emask) {
							if (pixels[i] == 0.0) {
								if (!add_mask_pixels) extended_mask[mask_k][iprime][jprime] = false;
								else if (extended_mask[mask_k][iprime][jprime]==true) n_maskpixels++;
							}
							else {
								extended_mask[mask_k][iprime][jprime] = true;
								n_maskpixels++;
								//cout << pixels[iprime] << endl;
							}
							if (new_mask) {
								in_mask[mask_k][iprime][jprime] = true; // if new mask was created, then mask contains all the pixels by default
							}
						} else {
							if (pixels[i] == 0.0) {
								if (!add_mask_pixels) in_mask[mask_k][iprime][jprime] = false;
								else if (in_mask[mask_k][iprime][jprime]==true) n_maskpixels++;
							}
							else {
								if (!subtract_mask_pixels) {
									in_mask[mask_k][iprime][jprime] = true;
									if (!extended_mask[mask_k][iprime][jprime]) extended_mask[mask_k][iprime][jprime] = true; // the extended mask MUST contain all the primary mask pixels
									n_maskpixels++;
								}
							//cout << pixels[iprime] << endl;
							}
							if (new_mask) extended_mask[mask_k][iprime][jprime] = true; // if new mask was created, then extended mask contains all the pixels by default
						}
					}
				}
				delete[] fpixel;
				delete[] pixels;
				image_load_status = true;
			}
		}
		fits_close_file(fptr, &status);
	} 

	if (status) fits_report_error(stderr, status); // print any error message
	if (!foreground) {
		if ((image_load_status) and (!emask)) n_mask_pixels[mask_k] = n_maskpixels;
		if (image_load_status) extended_mask_n_neighbors[mask_k] = -1; // this whole 'emask_n_neighbors' thing is shoddy and should get replaced altogether
		//set_extended_mask(qlens->extended_mask_n_neighbors);
	}
	if (foreground) {
		if ((qlens) and (qlens->fgmask_padding > 0)) {
			expand_foreground_mask(qlens->fgmask_padding);
		}
	}

	return image_load_status;
#endif
}

bool ImageData::copy_mask(ImageData* data, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	if (data->npixels_x != npixels_x) die("cannot copy mask; different number of x-pixels (%i vs %i)",npixels_x,data->npixels_x);
	if (data->npixels_y != npixels_y) die("cannot copy mask; different number of y-pixels (%i vs %i)",npixels_y,data->npixels_y);
	for (j=0; j < npixels_y; j++) {
		for (i=0; i < npixels_x; i++) {
			in_mask[mask_k][i][j] = data->in_mask[mask_k][i][j];
		}
	}
	n_mask_pixels[mask_k] = data->n_mask_pixels[mask_k];
	return true;
}


bool ImageData::save_mask_fits(string fits_filename, const bool foreground, const bool emask, const int mask_k, const bool subimage, const double xmin_in, const double xmax_in, const double ymin_in, const double ymax_in)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return false;
#else
	if (mask_k >= n_masks) { warn("mask with given index has not been created"); return false; }
	int i,j,kk;
	fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix = -64, naxis = 2;
	int min_i=0, min_j=0, max_i=npixels_x-1, max_j=npixels_y-1;
	if (subimage) {
		min_i = (int) ((xmin_in-xmin) * npixels_x / (xmax-xmin));
		if (min_i < 0) min_i = 0;
		max_i = (int) ((xmax_in-xmin) * npixels_x / (xmax-xmin));
		if (max_i > (npixels_x-1)) max_i = npixels_x-1;
		min_j = (int) ((ymin_in-ymin) * npixels_y / (ymax-ymin));
		if (min_j < 0) min_j = 0;
		max_j = (int) ((ymax_in-ymin) * npixels_y / (ymax-ymin));
		if (max_j > (npixels_y-1)) max_j = npixels_y-1;
	}
	int npix_x = max_i-min_i+1;
	int npix_y = max_j-min_j+1;
	cout << "imin=" << min_i << " imax=" << max_i << " jmin=" << min_j << " jmax=" << max_j << " npix_x=" << npix_x << " npix_y=" << npix_y << endl;
	long naxes[2] = {npix_x,npix_y};
	double *pixels;
	string fits_filename_overwrite = "!" + fits_filename; // ensures that it overwrites an existing file of the same name

	if (!fits_create_file(&outfptr, fits_filename_overwrite.c_str(), &status))
	{
		if (!fits_create_img(outfptr, bitpix, naxis, naxes, &status))
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long* fpixel = new long[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[npix_x];

				for (fpixel[1]=1, j=min_j; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=min_i, kk=0; i <= max_i; i++, kk++) {
						if (foreground) {
							if (foreground_mask_data[i][j]) pixels[kk] = 1.0;
							else pixels[kk] = 0.0;
						} else if (emask) {
							if (extended_mask[mask_k][i][j]) pixels[kk] = 1.0;
							else pixels[kk] = 0.0;
						} else {
							if (in_mask[mask_k][i][j]) pixels[kk] = 1.0;
							else pixels[kk] = 0.0;
						}
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] fpixel;
				delete[] pixels;
			}
		}
		fits_close_file(outfptr, &status);
	} 

	if (status) fits_report_error(stderr, status); // print any error message
	return true;
#endif
}

ImageData::~ImageData()
{
	if (xvals != NULL) delete[] xvals;
	if (yvals != NULL) delete[] yvals;
	if (surface_brightness != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] surface_brightness[i];
		delete[] surface_brightness;
	}
	if (noise_map != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] noise_map[i];
		delete[] noise_map;
	}
	if (covinv_map != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] covinv_map[i];
		delete[] covinv_map;
	}
	if (high_sn_pixel != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] high_sn_pixel[i];
		delete[] high_sn_pixel;
	}
	if (in_mask != NULL) {
		int i,k;
		for (k=0; k < n_masks; k++) {
			for (i=0; i < npixels_x; i++) delete[] in_mask[k][i];
			delete[] in_mask[k];
		}
		delete[] in_mask;
	}
	if (extended_mask != NULL) {
		int i,k;
		for (k=0; k < n_masks; k++) {
			for (i=0; i < npixels_x; i++) delete[] extended_mask[k][i];
			delete[] extended_mask[k];
		}
		delete[] extended_mask;
	}
	if (foreground_mask != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] foreground_mask[i];
		delete[] foreground_mask;
	}
	if (foreground_mask_data != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] foreground_mask_data[i];
		delete[] foreground_mask_data;
	}
}

bool ImageData::create_new_mask()
{
	if (n_masks==0) { warn("must load image data before additional masks can be created"); return false; } // remember the first mask is created automatically when data is loaded
	bool ***new_masks = new bool**[n_masks+1];
	bool ***new_extended_masks = new bool**[n_masks+1];
	int *new_n_mask_pixels = new int[n_masks+1];
	int i,j,k;
	for (k=0; k < n_masks; k++) {
		new_masks[k] = in_mask[k];
		new_extended_masks[k] = extended_mask[k];
		new_n_mask_pixels[k] = n_mask_pixels[k];
	}
	new_masks[n_masks] = new bool*[npixels_x];
	new_extended_masks[n_masks] = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) {
		new_masks[n_masks][i] = new bool[npixels_y];
		new_extended_masks[n_masks][i] = new bool[npixels_y];
		for (j=0; j < npixels_y; j++) {
			new_masks[n_masks][i][j] = true;
			new_extended_masks[n_masks][i][j] = true;
		}

	}
	new_n_mask_pixels[n_masks] = npixels_x*npixels_y;

	delete[] in_mask;
	delete[] extended_mask;
	delete[] n_mask_pixels;
	in_mask = new_masks;
	extended_mask = new_extended_masks;
	n_mask_pixels = new_n_mask_pixels;
	n_masks++;
	return true;
}

bool ImageData::set_no_mask_pixels(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			in_mask[mask_k][i][j] = false;
			extended_mask[mask_k][i][j] = false;
		}
	}
	n_mask_pixels[mask_k] = 0;
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

bool ImageData::set_all_mask_pixels(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			in_mask[mask_k][i][j] = true;
			extended_mask[mask_k][i][j] = true;
		}
	}
	n_mask_pixels[mask_k] = npixels_x*npixels_y;
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

bool ImageData::set_foreground_mask_to_primary_mask(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (in_mask[mask_k][i][j]) foreground_mask_data[i][j] = true;
			else foreground_mask_data[i][j] = false;
		}
	}
	return true;
}

bool ImageData::invert_mask(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	n_mask_pixels[mask_k] = 0;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (!in_mask[mask_k][i][j]) {
				in_mask[mask_k][i][j] = true;
				extended_mask[mask_k][i][j] = true;
				n_mask_pixels[mask_k]++;
			}
			else {
				in_mask[mask_k][i][j] = false;
				extended_mask[mask_k][i][j] = false;
			}
		}
	}
	find_extended_mask_rmax(); // used when splining integrals for deflection/hessian from Fourier modes
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

bool ImageData::inside_mask(const double x, const double y, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	if ((x <= xmin) or (x >= xmax) or (y <= ymin) or (y >= ymax)) return false;
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if ((xvals[i] < x) and (xvals[i+1] >= x) and (yvals[j] < y) and (yvals[j+1] > y)) {
				if (in_mask[mask_k][i][j]) return true;
			}
		}
	}
	return false;
}

bool ImageData::assign_mask_windows(const double sb_noise_threshold, const int threshold_size, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	vector<int> mask_window_sizes;
	vector<bool> active_mask_window;
	vector<double> mask_window_max_sb;
	int n_mask_windows = 0;
	int **mask_window_id = new int*[npixels_x];
	int i,j,l;
	for (i=0; i < npixels_x; i++) mask_window_id[i] = new int[npixels_y];
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			mask_window_id[i][j] = -1; // -1 means not belonging to a mask window
		}
	}

	int neighbor_mask, current_mask, this_window_id;
	bool new_mask_member;
	do {
		current_mask = -1;
		do {
			new_mask_member = false;
			for (j=0; j < npixels_y; j++) {
				for (l=0; l < 2*npixels_x; l++) {
					if (l < npixels_x) i=l;
					else i = 2*npixels_x-l-1;
					neighbor_mask = -1;
					if (in_mask[mask_k][i][j]) {
						if ((current_mask == -1) and (mask_window_id[i][j] == -1)) {
							current_mask = n_mask_windows;
							mask_window_sizes.push_back(1);
							active_mask_window.push_back(true);
							mask_window_max_sb.push_back(surface_brightness[i][j]);
							mask_window_id[i][j] = n_mask_windows;
							n_mask_windows++;
							new_mask_member = true;
						} else {
							if (mask_window_id[i][j] == -1) {
								if ((i > 0) and (in_mask[mask_k][i-1][j]) and (mask_window_id[i-1][j] == current_mask)) neighbor_mask = current_mask;
								else if ((i < npixels_x-1) and (in_mask[mask_k][i+1][j]) and (mask_window_id[i+1][j] == current_mask)) neighbor_mask = current_mask;
								else if ((j > 0) and (in_mask[mask_k][i][j-1]) and (mask_window_id[i][j-1] == current_mask)) neighbor_mask = current_mask;
								else if ((j < npixels_y-1) and (in_mask[mask_k][i][j+1]) and (mask_window_id[i][j+1] == current_mask)) neighbor_mask = current_mask;
								if (neighbor_mask == current_mask) {
									mask_window_id[i][j] = neighbor_mask;
									mask_window_sizes[neighbor_mask]++;
									if (surface_brightness[i][j] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i][j];
									new_mask_member = true;
								}
							}
							if (mask_window_id[i][j] == current_mask) {
								this_window_id = mask_window_id[i][j];
								if ((i > 0) and (in_mask[mask_k][i-1][j]) and (mask_window_id[i-1][j] < 0)) {
									mask_window_id[i-1][j] = this_window_id;
									mask_window_sizes[this_window_id]++;
									if (surface_brightness[i-1][j] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i-1][j];
									new_mask_member = true;
								}
								if ((i < npixels_x-1) and (in_mask[mask_k][i+1][j]) and (mask_window_id[i+1][j] < 0)) {
									mask_window_id[i+1][j] = this_window_id;
									mask_window_sizes[this_window_id]++;
									if (surface_brightness[i+1][j] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i+1][j];
									new_mask_member = true;
								}
								if ((j > 0) and (in_mask[mask_k][i][j-1]) and (mask_window_id[i][j-1] < 0)) {
									mask_window_id[i][j-1] = this_window_id;
									mask_window_sizes[this_window_id]++;
									if (surface_brightness[i][j-1] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i][j-1];
									new_mask_member = true;
								}
								if ((j < npixels_y-1) and (in_mask[mask_k][i][j+1]) and (mask_window_id[i][j+1] < 0)) {
									mask_window_id[i][j+1] = this_window_id;
									mask_window_sizes[this_window_id]++;
									if (surface_brightness[i][j+1] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i][j+1];
									new_mask_member = true;
								}
							}
						}
					}
				}
			}
		} while (new_mask_member);
	} while (current_mask != -1);
	int smallest_window_size, smallest_window_id;
	int n_windows_eff = n_mask_windows;
	int old_n_windows = n_windows_eff;
	do {
		old_n_windows = n_windows_eff;
		smallest_window_size = npixels_x*npixels_y;
		smallest_window_id = -1;
		for (l=0; l < n_mask_windows; l++) {
			if (active_mask_window[l]) {
				if ((mask_window_max_sb[l] > sb_noise_threshold*qlens->background_pixel_noise) and (mask_window_sizes[l] > threshold_size)) {
					active_mask_window[l] = false; // ensures it won't get cut
				}
				else if ((mask_window_sizes[l] != 0) and (mask_window_sizes[l] < smallest_window_size)) {
					smallest_window_size = mask_window_sizes[l];
					smallest_window_id = l;
				}
			}
		}
		if ((smallest_window_id != -1) and (smallest_window_size > 0)) {
			for (i=0; i < npixels_x; i++) {
				for (j=0; j < npixels_y; j++) {
					if (mask_window_id[i][j]==smallest_window_id) {
						in_mask[mask_k][i][j] = false;
					}
				}
			}
			active_mask_window[smallest_window_id] = false;
			mask_window_sizes[smallest_window_id] = 0;
			n_windows_eff--;
		}
	}
	while (n_windows_eff < old_n_windows);
	if (qlens->mpi_id == 0) cout << "Trimmed " << n_mask_windows << " windows down to " << n_windows_eff << " windows" << endl;
	j=0;
	for (i=0; i < n_mask_windows; i++) {
		if (mask_window_sizes[i] != 0) {
			j++;
			if (qlens->mpi_id == 0) cout << "Window " << j << " size: " << mask_window_sizes[i] << " max_sb: " << mask_window_max_sb[i] << endl;
		}
	}
	for (i=0; i < npixels_x; i++) delete[] mask_window_id[i];
	delete[] mask_window_id;
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

bool ImageData::unset_low_signal_pixels(const double sb_threshold, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created (mask_i=%i,n_masks=%i)",mask_k,n_masks); return false; }
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (surface_brightness[i][j] < sb_threshold) {
				if (in_mask[mask_k][i][j]) {
					in_mask[mask_k][i][j] = false;
					n_mask_pixels[mask_k]--;
				}
			}
		}
	}

	// now we will deactivate pixels that have 0 or 1 neighboring active pixels (to get rid of isolated bits)
	bool **req = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) req[i] = new bool[npixels_y];
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			req[i][j] = in_mask[mask_k][i][j];
		}
	}
	int n_active_neighbors;
	for (int k=0; k < 3; k++) {
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if (in_mask[mask_k][i][j]) {
					n_active_neighbors = 0;
					if ((i < npixels_x-1) and (in_mask[mask_k][i+1][j])) n_active_neighbors++;
					if ((i > 0) and (in_mask[mask_k][i-1][j])) n_active_neighbors++;
					if ((j < npixels_y-1) and (in_mask[mask_k][i][j+1])) n_active_neighbors++;
					if ((j > 0) and (in_mask[mask_k][i][j-1])) n_active_neighbors++;
					if ((n_active_neighbors < 2) and (req[i][j])) {
						req[i][j] = false;
						n_mask_pixels[mask_k]--;
					}
				}
			}
		}
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				in_mask[mask_k][i][j] = req[i][j];
			}
		}
	}
	// check for any lingering "holes" in the mask and activate them
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (!in_mask[mask_k][i][j]) {
				if (((i < npixels_x-1) and (in_mask[mask_k][i+1][j])) and ((i > 0) and (in_mask[mask_k][i-1][j])) and ((j < npixels_y-1) and (in_mask[mask_k][i][j+1])) and ((j > 0) and (in_mask[mask_k][i][j-1]))) {
					if (!req[i][j]) {
						in_mask[mask_k][i][j] = true;
						n_mask_pixels[mask_k]++;
					}
				}
			}
		}
	}

	for (i=0; i < npixels_x; i++) delete[] req[i];
	delete[] req;
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

bool ImageData::set_positive_radial_gradient_pixels(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return true; }
	int i,j;
	lensvector<double> rhat, grad;
	double rderiv;
	for (i=0; i < npixels_x-1; i++) {
		for (j=0; j < npixels_y-1; j++) {
			if (!in_mask[mask_k][i][j]) {
				rhat[0] = pixel_xcvals[i];
				rhat[1] = pixel_ycvals[j];
				rhat /= rhat.norm();
				grad[0] = surface_brightness[i+1][j] - surface_brightness[i][j];
				grad[1] = surface_brightness[i][j+1] - surface_brightness[i][j];
				rderiv = grad * rhat; // dot product
				if (rderiv > 0) {
					in_mask[mask_k][i][j] = true;
					n_mask_pixels[mask_k]++;
				}
			}
		}
	}
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return false;
}

bool ImageData::set_neighbor_pixels(const bool only_interior_neighbors, const bool only_exterior_neighbors, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	double r0, r;
	bool **req = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) req[i] = new bool[npixels_y];
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			req[i][j] = in_mask[mask_k][i][j];
		}
	}
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if ((in_mask[mask_k][i][j])) {
				if ((only_interior_neighbors) or (only_exterior_neighbors)) r0 = sqrt(SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[j]));
				if ((i < npixels_x-1) and (!in_mask[mask_k][i+1][j])) {
					if (!req[i+1][j]) {
						if ((only_interior_neighbors) or (only_exterior_neighbors)) r = sqrt(SQR(pixel_xcvals[i+1]) + SQR(pixel_ycvals[j]));
						if (((only_interior_neighbors) and (r > r0)) or ((only_exterior_neighbors) and (r < r0))) ;
						else {
							req[i+1][j] = true;
							n_mask_pixels[mask_k]++;
						}
					}
				}
				if ((i > 0) and (!in_mask[mask_k][i-1][j])) {
					if (!req[i-1][j]) {
						if ((only_interior_neighbors) or (only_exterior_neighbors)) r = sqrt(SQR(pixel_xcvals[i-1]) + SQR(pixel_ycvals[j]));
						if (((only_interior_neighbors) and (r > r0)) or ((only_exterior_neighbors) and (r < r0))) ;
						else {
							req[i-1][j] = true;
							n_mask_pixels[mask_k]++;
						}
					}
				}
				if ((j < npixels_y-1) and (!in_mask[mask_k][i][j+1])) {
					if (!req[i][j+1]) {
						if ((only_interior_neighbors) or (only_exterior_neighbors)) r = sqrt(SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[j+1]));
						if (((only_interior_neighbors) and (r > r0)) or ((only_exterior_neighbors) and (r < r0))) ;
						else {
							req[i][j+1] = true;
							n_mask_pixels[mask_k]++;
						}
					}
				}
				if ((j > 0) and (!in_mask[mask_k][i][j-1])) {
					if (!req[i][j-1]) {
						if ((only_interior_neighbors) or (only_exterior_neighbors)) r = sqrt(SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[j-1]));
						if (((only_interior_neighbors) and (r > r0)) or ((only_exterior_neighbors) and (r < r0))) ;
						else {
							req[i][j-1] = true;
							n_mask_pixels[mask_k]++;
						}
					}
				}
			}
		}
	}
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			in_mask[mask_k][i][j] = req[i][j];
		}
	}
	// check for any lingering "holes" in the mask and activate them
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (!in_mask[mask_k][i][j]) {
				if (((i < npixels_x-1) and (in_mask[mask_k][i+1][j])) and ((i > 0) and (in_mask[mask_k][i-1][j])) and ((j < npixels_y-1) and (in_mask[mask_k][i][j+1])) and ((j > 0) and (in_mask[mask_k][i][j-1]))) {
					if (!req[i][j]) {
						req[i][j] = true;
						n_mask_pixels[mask_k]++;
					}
				}
			}
		}
	}
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			in_mask[mask_k][i][j] = req[i][j];
		}
	}
	for (i=0; i < npixels_x; i++) delete[] req[i];
	delete[] req;
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

bool ImageData::expand_foreground_mask(const int n_it)
{
	int ii,i,j;
	double r0, r;
	bool **req = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) req[i] = new bool[npixels_y];
	for (ii=0; ii < n_it; ii++ ) {
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				req[i][j] = foreground_mask[i][j];
			}
		}
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if ((foreground_mask[i][j])) {
					if ((i < npixels_x-1) and (!foreground_mask[i+1][j])) {
						if (!req[i+1][j]) {
							req[i+1][j] = true;
							//n_foreground_mask_pixels++;
						}
					}
					if ((i > 0) and (!foreground_mask[i-1][j])) {
						if (!req[i-1][j]) {
							req[i-1][j] = true;
							//n_foreground_mask_pixels++;
						}
					}
					if ((j < npixels_y-1) and (!foreground_mask[i][j+1])) {
						if (!req[i][j+1]) {
							req[i][j+1] = true;
							//n_foreground_mask_pixels++;
						}
					}
					if ((j > 0) and (!foreground_mask[i][j-1])) {
						if (!req[i][j-1]) {
							req[i][j-1] = true;
							//n_foreground_mask_pixels++;
						}
					}
				}
			}
		}
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				foreground_mask[i][j] = req[i][j];
			}
		}
		// check for any lingering "holes" in the mask and activate them
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if (!foreground_mask[i][j]) {
					if (((i < npixels_x-1) and (foreground_mask[i+1][j])) and ((i > 0) and (foreground_mask[i-1][j])) and ((j < npixels_y-1) and (foreground_mask[i][j+1])) and ((j > 0) and (foreground_mask[i][j-1]))) {
						if (!req[i][j]) {
							req[i][j] = true;
							//n_foreground_mask_pixels++;
						}
					}
				}
			}
		}
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				foreground_mask[i][j] = req[i][j];
			}
		}
	}
	for (i=0; i < npixels_x; i++) delete[] req[i];
	delete[] req;
	return true;
}

bool ImageData::set_mask_window(const double xmin, const double xmax, const double ymin, const double ymax, const bool unset, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if ((xvals[i] > xmin) and (xvals[i+1] < xmax) and (yvals[j] > ymin) and (yvals[j+1] < ymax)) {
				if (!unset) {
					if (in_mask[mask_k][i][j] == false) {
						in_mask[mask_k][i][j] = true;
						n_mask_pixels[mask_k]++;
					}
				} else {
					if (in_mask[mask_k][i][j] == true) {
						in_mask[mask_k][i][j] = false;
						n_mask_pixels[mask_k]--;
					}
				}
			}
		}
	}
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

bool ImageData::set_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset, const bool foreground, const int mask_k)
{
	if ((!foreground) and (mask_k >= n_masks)) { warn("mask with specified index has not been loaded or created"); return false; }
	bool ***mask_ptr = (foreground) ? &foreground_mask_data : &in_mask[mask_k];
	// the angles MUST be between 0 and 360 here, so we enforce this in the following
	while (theta1_deg < 0) theta1_deg += 360;
	while (theta1_deg > 360) theta1_deg -= 360;
	while (theta2_deg < 0) theta2_deg += 360;
	while (theta2_deg > 360) theta2_deg -= 360;
	double x, y, rsq, rminsq, rmaxsq, theta, theta1, theta2;
	rminsq = rmin*rmin;
	rmaxsq = rmax*rmax;
	theta1 = degrees_to_radians(theta1_deg);
	theta2 = degrees_to_radians(theta2_deg);
	int i,j;
	double theta_old;
	for (i=0; i < npixels_x; i++) {
		x = 0.5*(xvals[i] + xvals[i+1]);
		for (j=0; j < npixels_y; j++) {
			y = 0.5*(yvals[j] + yvals[j+1]);
			rsq = SQR((x-xc)/xstretch) + SQR((y-yc)/ystretch);
			theta = atan(abs(((y-yc)/(x-xc))*xstretch/ystretch));
			theta_old=theta;
			if (x < xc) {
				if (y < yc)
					theta = theta + M_PI;
				else
					theta = M_PI - theta;
			} else if (y < yc) {
				theta = M_2PI - theta;
			}
			if ((rsq > rminsq) and (rsq < rmaxsq)) {
				// allow for two possibilities: theta1 < theta2, and theta2 < theta1 (which can happen if, e.g. theta2 is input as negative and theta1 is input as positive)
				if (((theta2 > theta1) and (theta >= theta1) and (theta <= theta2)) or ((theta1 > theta2) and ((theta >= theta1) or (theta <= theta2)))) {
					if (!unset) {
						if ((*mask_ptr)[i][j] == false) {
							(*mask_ptr)[i][j] = true;
							if (!foreground) n_mask_pixels[mask_k]++;
						}
					} else {
						if ((*mask_ptr)[i][j] == true) {
							(*mask_ptr)[i][j] = false;
							if (!foreground) n_mask_pixels[mask_k]--;
						}
					}
				}
			}
			if (foreground) {
				foreground_mask[i][j] = foreground_mask_data[i][j];
				//foreground_mask[i][j] = true;
				//foreground_mask_data[i][j] = true;
			}
		}
	}
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

bool ImageData::reset_extended_mask(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			extended_mask[mask_k][i][j] = in_mask[mask_k][i][j];
		}
	}
	if (mask_k==0) find_extended_mask_rmax(); // used when splining integrals for deflection/hessian from Fourier modes
	return true;
}

bool ImageData::set_extended_mask(const int n_neighbors, const bool add_to_emask_in, const bool only_interior_neighbors, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	// This is very similar to the set_neighbor_pixels() function in ImageData; used here for the outside_sb_prior feature
	int i,j,k;
	bool add_to_emask = add_to_emask_in;
	if (n_neighbors < 0) {
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				extended_mask[mask_k][i][j] = true;
			}
		}
		return true;
	}
	if ((add_to_emask) and (get_size_of_extended_mask(mask_k)<=0)) add_to_emask = false;
	bool **req = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) req[i] = new bool[npixels_y];
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (!add_to_emask) {
				extended_mask[mask_k][i][j] = in_mask[mask_k][i][j];
				req[i][j] = in_mask[mask_k][i][j];
			} else {
				req[i][j] = extended_mask[mask_k][i][j];
			}
		}
	}
	double r0=0, r;
	int emask_npix, emask_npix0;
	for (k=0; k < n_neighbors; k++) {
		emask_npix0 = get_size_of_extended_mask(mask_k);
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if (req[i][j]) {
					if (only_interior_neighbors) r0 = sqrt(SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[j]));
					if ((i < npixels_x-1) and (!req[i+1][j])) {
						if (!extended_mask[mask_k][i+1][j]) {
							if (only_interior_neighbors) r = sqrt(SQR(pixel_xcvals[i+1]) + SQR(pixel_ycvals[j]));
							if ((only_interior_neighbors) and ((r - r0) > 1e-6)) {
								//cout << "NOT adding pixel " << (i+1) << " " << j << " " << r << " vs " << r0 << endl;
							}
							else {
								extended_mask[mask_k][i+1][j] = true;
								//cout << "Adding pixel " << (i+1) << " " << j << " " << r0 << endl;
							}
						}
					}
					if ((i > 0) and (!req[i-1][j])) {
						if (!extended_mask[mask_k][i-1][j]) {
							if (only_interior_neighbors) r = sqrt(SQR(pixel_xcvals[i-1]) + SQR(pixel_ycvals[j]));
							if ((only_interior_neighbors) and ((r - r0) > 1e-6)) {
								//cout << "NOT Adding pixel " << (i-1) << " " << j << " " << r << " vs " << r0 << endl;
							} else {
								extended_mask[mask_k][i-1][j] = true;
								//cout << "Adding pixel " << (i-1) << " " << j << " " << r << " vs " << r0 << endl;
							}
						}
					}
					if ((j < npixels_y-1) and (!req[i][j+1])) {
						if (!extended_mask[mask_k][i][j+1]) {
							if (only_interior_neighbors) r = sqrt(SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[j+1]));
							if ((only_interior_neighbors) and ((r - r0) > 1e-6)) {
								//cout << "NOT Adding pixel " << (i) << " " << (j+1) << " " << r << " vs " << r0 << endl;
							} else {
								extended_mask[mask_k][i][j+1] = true;
								//cout << "Adding pixel " << (i) << " " << (j+1) << " " << r << " vs " << r0 << endl;
							}
						}
					}
					if ((j > 0) and (!req[i][j-1])) {
						if (!extended_mask[mask_k][i][j-1]) {
							if (only_interior_neighbors) r = sqrt(SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[j-1]));
							if ((only_interior_neighbors) and ((r - r0) > 1e-6)) {
								//cout << "NOT Adding pixel " << (i) << " " << (j-1) << " " << r << " vs " << r0 << endl;
							} else {
								extended_mask[mask_k][i][j-1] = true;
								//cout << "Adding pixel " << (i) << " " << (j-1) << " " << r << " vs " << r0 << endl;
							}
						}
					}
				}
			}
		}
		// check for any lingering "holes" in the mask and activate them
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if (!extended_mask[mask_k][i][j]) {
					if (((i < npixels_x-1) and (extended_mask[mask_k][i+1][j])) and ((i > 0) and (extended_mask[mask_k][i-1][j])) and ((j < npixels_y-1) and (extended_mask[mask_k][i][j+1])) and ((j > 0) and (extended_mask[mask_k][i][j-1]))) {
						if (!extended_mask[mask_k][i][j]) {
							extended_mask[mask_k][i][j] = true;
							//cout << "Filling hole " << i << " " << j << endl;
						}
					}
				}
			}
		}
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				req[i][j] = extended_mask[mask_k][i][j];
			}
		}
		emask_npix = get_size_of_extended_mask(mask_k);
		if (emask_npix==emask_npix0) break;

		//long int npix = 0;
		//for (i=0; i < npixels_x; i++) {
			//for (j=0; j < npixels_y; j++) {
				//if (extended_mask[mask_k][i][j]) npix++;
			//}
		//}
		//cout << "iteration " << k << ": npix=" << npix << endl;
	}
	find_extended_mask_rmax(); // used when splining integrals for deflection/hessian from Fourier modes
	for (i=0; i < npixels_x; i++) delete[] req[i];
	delete[] req;
	return true;
}

bool ImageData::activate_partner_image_pixels(const int mask_k, const bool emask)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return true; }
	bool ***maskptr;
	if (emask) maskptr = extended_mask;
	else maskptr = in_mask;
	int i,j,k,ii,jj,n_images;
	bool found_itself;
	double xstep = xvals[1] - xvals[0];
	double ystep = yvals[1] - yvals[0];
	double rsq, outermost_rsq = 0, innermost_rsq = 100000;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			found_itself = false;
			if (maskptr[mask_k][i][j]) {
				rsq = SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[i]);
				if (rsq > outermost_rsq) outermost_rsq = rsq;
				if (rsq < innermost_rsq) innermost_rsq = rsq;
				lensvector<double> pos,src;
				pos[0] = pixel_xcvals[i];
				pos[1] = pixel_ycvals[j];
				qlens->find_sourcept<double>(pos,src,0,qlens->reference_zfactors,qlens->default_zsrc_beta_factors);
				image<double> *img = qlens->get_images<double>(src, n_images, false);
				for (k=0; k < n_images; k++) {
					if ((img[k].mag > 0) and (abs(img[k].mag) < 0.1)) continue; // ignore central images
					ii = (int) ((img[k].pos[0] - xvals[0]) / xstep);
					jj = (int) ((img[k].pos[1] - yvals[0]) / ystep);
					if ((ii < 0) or (jj < 0) or (ii > npixels_x) or (jj > npixels_y)) continue;
					if ((ii==i) and (jj==j)) {
						found_itself = true;
						continue;
					}
					if ((!maskptr[mask_k][ii][jj]) and (foreground_mask[ii][jj])) { // any pixels that are not in the foreground mask shouldn't be in emask either
						maskptr[mask_k][ii][jj] = true;
						if (!emask) n_mask_pixels[mask_k]++;
						rsq = SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[i]);
						if (rsq > outermost_rsq) outermost_rsq = rsq;
						if (rsq < innermost_rsq) innermost_rsq = rsq;
					}
				}
				if (!found_itself) warn("pixel couldn't find itself");
			}
		}
	}

	// Now, we check pixels still not in the mask and see if any of them have partner images inside the mask; if so, activate it
	outermost_rsq = SQR(sqrt(outermost_rsq) + 0.05);
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (!foreground_mask[i][j]) continue;
			rsq = SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[i]);
			if ((rsq < outermost_rsq) and (rsq > innermost_rsq) and (!maskptr[mask_k][i][j])) {
				lensvector<double> pos,src;
				pos[0] = pixel_xcvals[i];
				pos[1] = pixel_ycvals[j];
				qlens->find_sourcept<double>(pos,src,0,qlens->reference_zfactors,qlens->default_zsrc_beta_factors);
				image<double> *img = qlens->get_images<double>(src, n_images, false);
				for (k=0; k < n_images; k++) {
					if ((img[k].mag > 0) and (abs(img[k].mag) < 0.1)) continue; // ignore central images
					ii = (int) ((img[k].pos[0] - xvals[0]) / xstep);
					jj = (int) ((img[k].pos[1] - yvals[0]) / ystep);
					if ((ii < 0) or (jj < 0) or (ii > npixels_x) or (jj > npixels_y)) continue;
					if ((ii==i) and (jj==j)) continue;
					if (maskptr[mask_k][ii][jj]) {
						maskptr[mask_k][i][j] = true;
						if (!emask) n_mask_pixels[mask_k]++;
					}
				}
			}
		}
	}

	// check for any lingering "holes" in the mask and activate them
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (!maskptr[mask_k][i][j]) {
				if (((i < npixels_x-1) and (maskptr[mask_k][i+1][j])) and ((i > 0) and (maskptr[mask_k][i-1][j])) and ((j < npixels_y-1) and (maskptr[mask_k][i][j+1])) and ((j > 0) and (maskptr[mask_k][i][j-1]))) {
					if (!maskptr[mask_k][i][j]) {
						maskptr[mask_k][i][j] = true;
						if (!emask) n_mask_pixels[mask_k]++;
						//cout << "Filling hole " << i << " " << j << endl;
					}
				}
			}
		}
	}
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
	return true;
}

void ImageData::remove_overlapping_pixels_from_other_masks(const int mask_k)
{
	int i,j,mask_j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (in_mask[mask_k][i][j]) {
				for (mask_j=0; mask_j < n_masks; mask_j++) {
					if (mask_j==mask_k) continue;
					if (in_mask[mask_j][i][j]) {
						in_mask[mask_k][i][j] = false;
						n_mask_pixels[mask_k]--;
						break;
					}
				}
			}
		}
	}
	if (qlens->n_extended_src_redshifts > 0) qlens->update_imggrid_mask_values(mask_k);
}

bool ImageData::set_extended_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	// the angles MUST be between 0 and 360 here, so we enforce this in the following
	while (theta1_deg < 0) theta1_deg += 360;
	while (theta1_deg > 360) theta1_deg -= 360;
	while (theta2_deg < 0) theta2_deg += 360;
	while (theta2_deg > 360) theta2_deg -= 360;
	double x, y, rsq, rminsq, rmaxsq, theta, theta1, theta2;
	rminsq = rmin*rmin;
	rmaxsq = rmax*rmax;
	theta1 = degrees_to_radians(theta1_deg);
	theta2 = degrees_to_radians(theta2_deg);
	int i,j;
	double theta_old;
	bool pixels_in_mask = false;
	for (i=0; i < npixels_x; i++) {
		x = 0.5*(xvals[i] + xvals[i+1]);
		for (j=0; j < npixels_y; j++) {
			y = 0.5*(yvals[j] + yvals[j+1]);
			rsq = SQR((x-xc)/xstretch) + SQR((y-yc)/ystretch);
			theta = atan(abs(((y-yc)/(x-xc))*xstretch/ystretch));
			theta_old=theta;
			if (x < xc) {
				if (y < yc)
					theta = theta + M_PI;
				else
					theta = M_PI - theta;
			} else if (y < yc) {
				theta = M_2PI - theta;
			}
			if ((rsq > rminsq) and (rsq < rmaxsq)) {
				// allow for two possibilities: theta1 < theta2, and theta2 < theta1 (which can happen if, e.g. theta1 is input as negative and theta1 is input as positive)
				if (((theta2 > theta1) and (theta >= theta1) and (theta <= theta2)) or ((theta1 > theta2) and ((theta >= theta1) or (theta <= theta2)))) {
					if (!unset) {
						if (extended_mask[mask_k][i][j] == false) {
							extended_mask[mask_k][i][j] = true;
						}
					} else {
						if (extended_mask[mask_k][i][j] == true) {
							if (!in_mask[mask_k][i][j]) extended_mask[mask_k][i][j] = false;
							else pixels_in_mask = true;
						}
					}
				}
			}
		}
	}
	find_extended_mask_rmax(); // used when splining integrals for deflection/hessian from Fourier modes
	if (pixels_in_mask) warn("some pixels in the annulus were in the primary (lensed image) mask, and therefore could not be removed from extended mask");
	return true;
}

void ImageData::find_extended_mask_rmax()
{
	double r, rmax = -1e30;
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (extended_mask[0][i][j]) {
				r = sqrt(SQR(pixel_xcvals[i]) + SQR(pixel_ycvals[j+1]));
				if (r > rmax) rmax = r;
			}
		}
	}
	emask_rmax = rmax;
}


void ImageData::set_foreground_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset)
{
	// the angles MUST be between 0 and 360 here, so we enforce this in the following
	while (theta1_deg < 0) theta1_deg += 360;
	while (theta1_deg > 360) theta1_deg -= 360;
	while (theta2_deg < 0) theta2_deg += 360;
	while (theta2_deg > 360) theta2_deg -= 360;
	double x, y, rsq, rminsq, rmaxsq, theta, theta1, theta2;
	rminsq = rmin*rmin;
	rmaxsq = rmax*rmax;
	theta1 = degrees_to_radians(theta1_deg);
	theta2 = degrees_to_radians(theta2_deg);
	int i,j,k;
	double theta_old;
	bool pixels_in_mask = false;
	for (i=0; i < npixels_x; i++) {
		x = 0.5*(xvals[i] + xvals[i+1]);
		for (j=0; j < npixels_y; j++) {
			y = 0.5*(yvals[j] + yvals[j+1]);
			rsq = SQR((x-xc)/xstretch) + SQR((y-yc)/ystretch);
			theta = atan(abs(((y-yc)/(x-xc))*xstretch/ystretch));
			theta_old=theta;
			if (x < xc) {
				if (y < yc)
					theta = theta + M_PI;
				else
					theta = M_PI - theta;
			} else if (y < yc) {
				theta = M_2PI - theta;
			}
			if ((rsq > rminsq) and (rsq < rmaxsq)) {
				// allow for two possibilities: theta1 < theta2, and theta2 < theta1 (which can happen if, e.g. theta1 is input as negative and theta1 is input as positive)
				if (((theta2 > theta1) and (theta >= theta1) and (theta <= theta2)) or ((theta1 > theta2) and ((theta >= theta1) or (theta <= theta2)))) {
					if (!unset) {
						if (foreground_mask[i][j] == false) {
							foreground_mask[i][j] = true;
						}
					} else {
						if (foreground_mask[i][j] == true) {
							for (k=0; k < n_masks; k++) {
								if (!in_mask[k][i][j]) foreground_mask[i][j] = false;
								else pixels_in_mask = true;
							}
						}
					}
				}
			}
		}
	}
	if (pixels_in_mask) warn("some pixels in the annulus were in the primary (lensed image) mask(s), and therefore could not be removed from foreground mask");
}

long int ImageData::get_size_of_extended_mask(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return -1; }
	int i,j;
	long int npix = 0;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (extended_mask[mask_k][i][j]) npix++;
		}
	}
	return npix;
}

long int ImageData::get_size_of_foreground_mask()
{
	int i,j;
	long int npix = 0;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (foreground_mask[i][j]) npix++;
		}
	}
	return npix;
}



bool ImageData::estimate_pixel_noise(const double xmin, const double xmax, const double ymin, const double ymax, double &noise, double &mean_sb, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	int imin=0, imax=npixels_x-1, jmin=0, jmax=npixels_y-1;
	bool passed_min=false;
	for (i=0; i < npixels_x; i++) {
		if ((passed_min==false) and ((xvals[i+1]+xvals[i]) > 2*xmin)) {
			imin = i;
			passed_min = true;
		} else if (passed_min==true) {
			if ((xvals[i+1]+xvals[i]) > 2*xmax) {
				imax = i-1;
				break;
			}
		}
	}
	passed_min = false;
	for (j=0; j < npixels_y; j++) {
		if ((passed_min==false) and ((yvals[j+1]+yvals[j]) > 2*ymin)) {
			jmin = j;
			passed_min = true;
		} else if (passed_min==true) {
			if ((yvals[j+1]+yvals[j]) > 2*ymax) {
				jmax = j-1;
				break;
			}
		}
	}
	if ((imin==imax) or (jmin==jmax)) die("window for centroid calculation has zero size");
	double sigsq_sb=0, total_flux=0;
	int np=0;
	double xm,ym;
	for (j=jmin; j <= jmax; j++) {
		for (i=imin; i <= imax; i++) {
			if (in_mask[mask_k][i][j]) {
				total_flux += surface_brightness[i][j];
				np++;
			}
		}
	}

	mean_sb = total_flux / np;
	for (j=jmin; j <= jmax; j++) {
		for (i=imin; i <= imax; i++) {
			if (in_mask[mask_k][i][j]) {
				sigsq_sb += SQR(surface_brightness[i][j]-mean_sb);
			}
		}
	}
	double sqrnoise = sigsq_sb/np;
	double sigthreshold = 3.0;
	double sqrthreshold = SQR(sigthreshold)*sqrnoise;
	noise = sqrt(sqrnoise);
	int nclip=0, prev_nclip;
	double difsqr;
	do {
		prev_nclip = nclip;
		nclip = 0;
		sigsq_sb = 0;
		np = 0;
		total_flux = 0;
		for (j=jmin; j <= jmax; j++) {
			for (i=imin; i <= imax; i++) {
				if (in_mask[mask_k][i][j]) {
					difsqr = SQR(surface_brightness[i][j]-mean_sb);
					if (difsqr > sqrthreshold) nclip++;
					else {
						sigsq_sb += difsqr;
						total_flux += surface_brightness[i][j];
						np++;
					}
				}
			}
		}
		sqrnoise = sigsq_sb/np;
		sqrthreshold = SQR(sigthreshold)*sqrnoise;
		noise = sqrt(sqrnoise);
		mean_sb = total_flux / np;
	} while (nclip > prev_nclip);
	return true;
}

bool ImageData::fit_isophote(const double xi0, const double xistep, const int emode, const double qi, const double theta_i_degrees, const double xc_i, const double yc_i, const int max_it, IsophoteData &isophote_data, const bool use_polar_higher_harmonics, const bool verbose, SB_Profile* sbprofile, const int default_sampling_mode_in, const int n_higher_harmonics, const bool fix_center, const int max_xi_it, const double ximax_in, const double rms_sbgrad_rel_threshold, const double npts_frac, const double rms_sbgrad_rel_transition, const double npts_frac_zeroweight)
{
	const int npts_max = 2000;
	const int min_it = 7;
	const double sbfrac = 0.04;
	int default_sampling_mode = default_sampling_mode_in;

	if (max_it < min_it) {
		warn("cannot have less than %i max iterations for isophote fit",min_it);
		return false;
	}

	ofstream ellout;
	if (verbose) ellout.open("ellfit.dat"); // only make the plot if "verbose" is set to true

	if (pixel_size <= 0) {
		pixel_size = dmax(pixel_xcvals[1]-pixel_xcvals[0],pixel_ycvals[1]-pixel_ycvals[0]);
	}
	//double xi_min = 2.5*pixel_size;
	double xi_min = 1.5*pixel_size;
	double xi_max = dmax(pixel_xcvals[npixels_x-1],pixel_ycvals[npixels_y-1]); // NOTE: we may never reach xi_max if S/N falls too low
	if (ximax_in > 0) {
		if (ximax_in > xi_max) {
			warn("cannot do isophote fit with xi_max greater than extend of pixel image");
			return false;
		} else {
			xi_max = ximax_in;
		}
	}
	double xi=xi0;
	double xifac = 1+xistep;
	int i, i_switch, j, k;
	for (i=0, xi=xi0; xi < xi_max; i++, xi *= xifac) ;
	i_switch = i;
	for (xi=xi0/xifac; xi > xi_min; i++, xi /= xifac) ;
	int n_xivals = i;
	if (n_xivals > max_xi_it) n_xivals = max_xi_it;

	int *xi_ivals = new int[n_xivals];
	int *xi_ivals_sorted = new int[n_xivals];
	double *xivals = new double[n_xivals];
	bool *repeat_params = new bool[n_xivals];
	for (i=0; i < n_xivals; i++) repeat_params[i] = false;

	for (i=0, xi=xi0; xi < xi_max; i++, xi *= xifac) { xivals[i] = xi; xi_ivals_sorted[i] = i; if (i==n_xivals-1) { i++; break;} }
	i_switch = i;
	if (i < n_xivals) {
		for (xi=xi0/xifac; xi > xi_min; i++, xi /= xifac) { xivals[i] = xi; xi_ivals_sorted[i] = i; if (i==n_xivals-1) { i++; break;} }
	}
	if (n_xivals > 1) sort(n_xivals,xivals,xi_ivals_sorted);
	for (i=0; i < n_xivals; i++) {
		for (j=0; j < n_xivals; j++) {
			if (xi_ivals_sorted[i] == j) {
				xi_ivals[j] = i;
				break;
			}
		}
	}
	isophote_data.input(n_xivals,xivals);
	if (n_higher_harmonics > 2) isophote_data.use_A56 = true;

	/*************************************** lambda functions ******************************************/
	// We'll use this lambda function to construct the matrices used for least-squares fitting
	auto fill_matrices = [](const int npts, const int nmax_amp, double *sb_residual, double *sb_weights, double **smatrix, double *Dvec, double **Smatrix, const double noise)
	{
		int i,j,k;
		double sqrnoise = noise*noise;
		for (i=0; i < nmax_amp; i++) {
			if ((sb_residual != NULL) and (Dvec != NULL)) {
				Dvec[i] = 0;
				for (k=0; k < npts; k++) {
					Dvec[i] += smatrix[i][k]*sb_residual[k]/sqrnoise;
					//if (sb_weights==NULL) Dvec[i] += smatrix[i][k]*sb_residual[k]/sqrnoise;
					//else Dvec[i] += smatrix[i][k]*sb_residual[k]*sb_weights[k]/sqrnoise;
				}
			}
			for (j=0; j <= i; j++) {
				Smatrix[i][j] = 0;
				for (k=0; k < npts; k++) {
					Smatrix[i][j] += smatrix[i][k]*smatrix[j][k]/sqrnoise;
					//if (sb_weights==NULL) Smatrix[i][j] += smatrix[i][k]*smatrix[j][k]/sqrnoise;
					//else Smatrix[i][j] += smatrix[i][k]*smatrix[j][k]*sb_weights[k]/sqrnoise;
				}
			}
		}
	};

	auto find_sbgrad = [](const int npts_sample, double *sbvals_prev, double *sbvals_next, double *sbgrad_weights_prev, double *sbgrad_weights_next, const double &gradstep, double &sb_grad, double &rms_sbgrad_rel, int &ngrad)
	{
		sb_grad=0;
		int i;
		ngrad=0;
		double sb_grad_sq=0, denom=0, sbgrad_i, sbgrad_wgt;
		for (i=0; i < npts_sample; i++) {
			if (!std::isnan(sbvals_prev[i]) and (!std::isnan(sbvals_next[i]))) {
				sbgrad_wgt = sbgrad_weights_prev[i]+sbgrad_weights_next[i]; // adding uncertainties in quadrature
				sbgrad_i = (sbvals_next[i] - sbvals_prev[i])/gradstep;
				//sb_grad += sbgrad_i;
				//sb_grad_sq += sbgrad_i*sbgrad_i;
				//denom += 1.0;
				sb_grad += sbgrad_i / sbgrad_wgt;
				sb_grad_sq += sbgrad_i*sbgrad_i / sbgrad_wgt;
				denom += 1.0/sbgrad_wgt;
				ngrad++;
			}
		}
		if (ngrad==0) {
			warn("no sampling points could be used for SB gradiant");
			return false;
		}
		sb_grad /= denom;
		sb_grad_sq /= denom;
		rms_sbgrad_rel = sqrt(sb_grad_sq - sb_grad*sb_grad)/abs(sb_grad);
		return true;
	};
	/***************************************************************************************************/

	double epsilon0, theta0, xc0, yc0; // these will be the best-fit params at the first radius xi0, since we'll return to it and step to smaller xi later
	double epsilon_prev, theta_prev, xc_prev, yc_prev; 
	double xc_err, yc_err, epsilon_err, theta_err;
	double theta_i = degrees_to_radians(theta_i_degrees);
	double epsilon, theta, xc, yc;
	double dtheta;
	double sb_avg, next_sb_avg, prev_sb_avg, grad_xistep, sb_grad, prev_sbgrad, rms_sbgrad_rel, prev_rms_sbgrad_rel, max_amp;
	bool abort_isofit = false;
	epsilon = 1 - qi;
	theta = theta_i;
	xc = xc_i;
	yc = yc_i;

	int lowest_harmonic, n_ellipse_amps, xc_ampnum, yc_ampnum, epsilon_ampnum, theta_ampnum;
	if (!fix_center) {
		lowest_harmonic = 1;
		xc_ampnum = 0;
		yc_ampnum = 1;
		epsilon_ampnum = 2;
		theta_ampnum = 3;
	} else {
		lowest_harmonic = 2;
		xc_ampnum = -1;
		yc_ampnum = -1;
		epsilon_ampnum = 0;
		theta_ampnum = 1;
	}
	n_ellipse_amps = 2*(3-lowest_harmonic);

	//const int n_higher_harmonics = 4; //  should be at least 2. Seems to be unstable if n_higher_harmonics > 4; probably matrices becoming singular for some xi vals
	int nmax_amp = n_ellipse_amps + 2*n_higher_harmonics;
	int n_harmonics_it; // will be reduced if singular matrix occurs
	int nmax_amp_it; // will be reduced if singular matrix occurs
	double *sb_residual = new double[npts_max];
	double *sb_weights = new double[npts_max];
	double *sbvals = new double[npts_max];
	double *sbvals_next = new double[npts_max];
	double *sbvals_prev = new double[npts_max];
	double *sbgrad_weights_next = new double[npts_max];
	double *sbgrad_weights_prev = new double[npts_max];
	double *Dvec = new double[nmax_amp];
	double *amp = new double[nmax_amp];
	double *amp_minres = new double[nmax_amp];
	double *amperrs = new double[nmax_amp];
	double **smatrix = new double*[nmax_amp];
	double **Smatrix = new double*[nmax_amp];
	for (i=0; i < nmax_amp; i++) {
		smatrix[i] = new double[npts_max];
		Smatrix[i] = new double[nmax_amp];
	}

	int sampling_mode; // Sampling modes: 0 = interpolation; 1 = pixel_integration; 2 = either 0/1 based on how big xi is; 3 = use sbprofile
	double sampling_noise;
	int it, jmax;
	int npts, npts_sample, next_npts, prev_npts, ngrad;
	double minchisq;
	bool already_switched = false;
	bool failed_isophote_fit;
	bool using_prev_sbgrad;
	bool do_parameter_search;
	bool tried_parameter_search;
	double rms_resid, rms_resid_min;
	double xc_minres, yc_minres, epsilon_minres, theta_minres, sbgrad_minres, rms_sbgrad_rel_minres, maxamp_minres;
	double maxamp_min;
	int it_minres;
	int xi_it, xi_i, xi_i_prev;
	int npts_minres, npts_sample_minres;
	double mean_pixel_noise=0;
	int ntot=0;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			mean_pixel_noise += SQR(noise_map[i][j]);
			ntot++;
		}
	}
	if (ntot != 0) mean_pixel_noise = sqrt(mean_pixel_noise/ntot);
	else {
		mean_pixel_noise = 0.001; // just a hack for cases where there is no noise
		warn("No pixel noise has been defined; choosing sig_noise=0.001");
	}
	for (xi_it=0; xi_it < n_xivals; xi_it++) {
		xi_i = xi_ivals[xi_it];
		if (xi_it==0) xi_i_prev = xi_i;
		else xi_i_prev = xi_ivals[xi_it-1];
		xi = xivals[xi_i];
		grad_xistep = xi*xistep/2;
		if (grad_xistep < pixel_size) {
			grad_xistep = pixel_size;
		}
		if (verbose) (*isophote_fit_out) << "xi_it=" << xi_it << " xi=" << xi << " epsilon=" << epsilon << " theta=" << radians_to_degrees(theta) << " xc=" << xc << " yc=" << yc << endl;
		if (xi_it==i_switch) {
			epsilon = epsilon0;
			theta = theta0;
			xc = xc0;
			yc = yc0;
			default_sampling_mode = default_sampling_mode_in;
			//(*isophote_fit_out) << "sampling mode now: " << default_sampling_mode << endl;
		}
		it = 0;
		minchisq = 1e30;
		rms_resid_min = 1e30;
		maxamp_min = 1e30;
		double hterm;
		epsilon_prev = epsilon;
		theta_prev = theta;
		xc_prev = xc;
		yc_prev = yc;

		sampling_mode = default_sampling_mode;
		n_harmonics_it = n_higher_harmonics;
		do_parameter_search = false;
		nmax_amp_it = nmax_amp;
		//if (xi > 1.0) { // this is just a hack because it's having trouble with higher harmonics above m=4 beyond 1 arcsec or so
			//n_harmonics_it = 2;
			//nmax_amp_it = n_ellipse_amps + 2*n_harmonics_it;
		//}
		if (failed_isophote_fit) do_parameter_search = true; // start out with a parameter search
		tried_parameter_search = false;
		failed_isophote_fit = false;
		while (true) {
			using_prev_sbgrad = false;
			//(*isophote_fit_out) << "Sampling mode: " << sampling_mode << endl;
			npts_sample = -1; // so it will choose automatically
			//(*isophote_fit_out) << "EPSILON=" << epsilon << " THETA=" << theta << endl;
			sb_avg = sample_ellipse(verbose,xi,xistep,epsilon,theta,xc,yc,npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals,NULL,sbprofile,true,sb_residual,sb_weights,smatrix,lowest_harmonic,2+n_harmonics_it,use_polar_higher_harmonics);
			if (npts < npts_sample*npts_frac) {
				// if there aren't enough points/sectors, then it's not worth doing a parameter search (it can even backfire and result in contour crossings etc.); just move on
				do_parameter_search = false;
				tried_parameter_search = true;
				epsilon = epsilon_prev;
				theta = theta_prev;
				xc = xc_prev;
				yc = yc_prev;
				failed_isophote_fit = true;
				double nfrac = npts/((double) npts_sample);
				(*isophote_fit_out) << "WARNING: not enough points being sampled; moving on to next ellipse (npts_frac=" << nfrac << ",npts_frac_threshold=" << npts_frac << ",sb_avg=" << sb_avg << ")" << endl;
				break;
			}
			//if (do_parameter_search) {
				//do_parameter_search = false;
				//tried_parameter_search = true;
			//}
			if (do_parameter_search) {
				double ep, th;
				double epmin = epsilon - 0.1;
				double epmax = epsilon + 0.1;
				if (epmin < 0) epmin = 0.01;
				if (epmax > 1) epmax = 0.9;
				double thmin = theta - M_PI/4;
				double thmax = theta + M_PI/4;
				int parameter_search_nn = 100;
				double epstep = (epmax-epmin)/(parameter_search_nn-1);
				double thstep = (thmax-thmin)/(parameter_search_nn-1);
				int ii,jj;
				double residmin = 1e30;

				for (ii=0, ep=epmin; ii < parameter_search_nn; ii++, ep += epstep) {
					for (jj=0, th=thmin; jj < parameter_search_nn; jj++, th += thstep) {
						sample_ellipse(false,xi,xistep,ep,th,xc,yc,npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals,NULL,sbprofile,true,sb_residual,sb_weights,smatrix,lowest_harmonic,2+n_harmonics_it,use_polar_higher_harmonics);

						// now generate Dvec and Smatrix (s_transpose * s), then do inversion to get amplitudes A.
						fill_matrices(npts,nmax_amp_it,sb_residual,sb_weights,smatrix,Dvec,Smatrix,1.0);
						bool chol_status;
						chol_status = Cholesky_dcmp(Smatrix,nmax_amp_it);
						if (!chol_status) {
							//warn("amplitude matrix is not positive-definite");
							continue;
						}
						Cholesky_solve(Smatrix,Dvec,amp,nmax_amp_it);

						rms_resid = 0;
						for (i=0; i < npts; i++) {
							hterm = sb_residual[i];
							for (j=n_ellipse_amps; j < nmax_amp_it; j++) hterm -= amp[j]*smatrix[j][i];
							rms_resid += hterm*hterm;
						}
						rms_resid = sqrt(rms_resid/npts);

						if (rms_resid < residmin) {
							epsilon = ep;
							theta = th;
							residmin = rms_resid;
						}
					}
				}
				if (residmin==1e30) {
					warn("Cholesky decomposition failed during parameter search; repeating previous isophote parameters");
					failed_isophote_fit = true;
					break;
				}
				(*isophote_fit_out) << "Smallest residuals for epsilon=" << epsilon << ", theta=" << theta << " during parameter search (rms_resid=" << residmin << ")" << endl;
				//sb_avg = sample_ellipse(verbose,xi,xistep,epsilon,theta,xc,yc,npts,npts_sample,emode,sampling_mode,sbvals,NULL,sbprofile,true,sb_residual,sb_weights,smatrix,lowest_harmonic,2+n_harmonics_it,use_polar_higher_harmonics);
				do_parameter_search = false;
				tried_parameter_search = true;
				it = 0; // it's effectively a do-over now
				minchisq = 1e30;
				rms_resid_min = 1e30;
				maxamp_min = 1e30;
				epsilon_prev = epsilon;
				theta_prev = theta;
				xc_prev = xc;
				yc_prev = yc;

				sampling_mode = default_sampling_mode;
				n_harmonics_it = n_higher_harmonics;
				nmax_amp_it = nmax_amp;
				failed_isophote_fit = false;
				continue;
			}

			if (verbose) {
				if (sb_avg*0.0 != 0.0) {
					for (i=0; i < npts; i++) {
						(*isophote_fit_out) << sbvals[i] << " " << sb_residual[i] << endl;
					}
				}
			}

			if ((verbose) and (it==0)) {
				if (sampling_mode==0) (*isophote_fit_out) << "Sampling mode: interpolation" << endl;
				else if (sampling_mode==1) (*isophote_fit_out) << "Sampling mode: sector integration" << endl;
				else if (sampling_mode==3) (*isophote_fit_out) << "Sampling mode: SB profile" << endl;
			}
			if (npts==0) {
				warn("isophote fit failed; no sampling points were accepted on ellipse");
				epsilon = epsilon_prev;
				theta = theta_prev;
				xc = xc_prev;
				yc = yc_prev;
				failed_isophote_fit = true;
				break;
			}
			// now generate Dvec and Smatrix (s_transpose * s), then do inversion to get amplitudes A. pixel errors are ignored here because they just cancel anyway
			fill_matrices(npts,nmax_amp_it,sb_residual,sb_weights,smatrix,Dvec,Smatrix,1.0);
			bool chol_status;
			chol_status = Cholesky_dcmp(Smatrix,nmax_amp_it);
			if (!chol_status) {
				if (n_harmonics_it > 2) {
					n_harmonics_it--;
					nmax_amp_it -= 2;
					continue;
				} else {
					warn("Cholesky decomposition failed, even with only two higher harmonics; repeating previous isophote parameters");
					epsilon = epsilon_prev;
					theta = theta_prev;
					xc = xc_prev;
					yc = yc_prev;
					failed_isophote_fit = true;
					break;
				}
			}
			Cholesky_solve(Smatrix,Dvec,amp,nmax_amp_it);

			rms_resid = 0;
			for (i=0; i < npts; i++) {
				hterm = sb_residual[i];
				for (j=n_ellipse_amps; j < nmax_amp_it; j++) hterm -= amp[j]*smatrix[j][i];
				rms_resid += hterm*hterm;
			}
			rms_resid = sqrt(rms_resid/npts);

			max_amp = -1e30;
			for (j=0; j < n_ellipse_amps; j++) if (abs(amp[j]) > max_amp) { max_amp = abs(amp[j]); jmax = j; }
			if (max_amp < maxamp_min) maxamp_min = max_amp;

			prev_sb_avg = sample_ellipse(verbose,xi-grad_xistep,xistep,epsilon,theta,xc,yc,prev_npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals_prev,sbgrad_weights_prev,sbprofile);
			next_sb_avg = sample_ellipse(verbose,xi+grad_xistep,xistep,epsilon,theta,xc,yc,next_npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals_next,sbgrad_weights_next,sbprofile);
			double nextstep = grad_xistep, prevstep = grad_xistep, gradstep;

			// Now we will see if not enough points are being sampled for the gradient, and will expand the stepsize to see if it helps (this can occur around masks)
			bool not_enough_pts = false;
			if (prev_npts < npts_sample*npts_frac) {
				warn("RUHROH! not enough points when getting gradient (npts_prev). Will increase stepsize (npts_sample=%i,nprev=%i,nnext=%i,npts=%i)",npts_sample,prev_npts,next_npts,npts);
				prev_sb_avg = sample_ellipse(verbose,xi-2*grad_xistep,xistep,epsilon,theta,xc,yc,prev_npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals_prev,sbgrad_weights_prev,sbprofile);
				if (prev_npts < npts_sample*npts_frac) {
					warn("RUHROH! not enough points when getting gradient (npts_next), even after reducing stepsize (npts_sample=%i,nprev=%i,nnext=%i,npts=%i)",npts_sample,prev_npts,next_npts,npts);
					not_enough_pts = true;
					//sb_grad = prev_sbgrad; // hack when all else fails
					//rms_sbgrad_rel = prev_rms_sbgrad_rel;
					//using_prev_sbgrad = true;
				}
				nextstep *= 2;
			}
			else if (next_npts < npts_sample*npts_frac) {
				warn("RUHROH! not enough points when getting gradient (npts_next) Will increase stepsize(npts_sample=%i,nprev=%i,nnext=%i,npts=%i)",npts_sample,prev_npts,next_npts,npts);
				next_sb_avg = sample_ellipse(verbose,xi+2*grad_xistep,xistep,epsilon,theta,xc,yc,next_npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals_next,sbgrad_weights_next,sbprofile);
				if (next_npts < npts_sample*npts_frac) {
					warn("RUHROH! not enough points when getting gradient (npts_next), even after reducing stepsize (npts_sample=%i,nprev=%i,nnext=%i,npts=%i)",npts_sample,prev_npts,next_npts,npts);
					not_enough_pts = true;
					//sb_grad = prev_sbgrad; // hack when all else fails
					//rms_sbgrad_rel = prev_rms_sbgrad_rel;
					//using_prev_sbgrad = true;
				}
				prevstep *= 2;
			}
			if ((not_enough_pts) or (prev_npts==0) or (next_npts==0)) {
				warn("isophote fit failed; not enough sampling points were accepted on ellipse for determining sbgrad");
				epsilon = epsilon_prev;
				theta = theta_prev;
				xc = xc_prev;
				yc = yc_prev;
				failed_isophote_fit = true;
				break;
			}

			if (!using_prev_sbgrad) {
				gradstep = nextstep+prevstep;
				if (!find_sbgrad(npts_sample,sbvals_prev,sbvals_next,sbgrad_weights_prev,sbgrad_weights_next,gradstep,sb_grad,rms_sbgrad_rel,ngrad))
				{
					abort_isofit = true;
					break;
				} else {
					if (verbose) (*isophote_fit_out) << "it=" << it << " sbavg=" << sb_avg << " rms=" << rms_resid << " sbgrad=" << sb_grad << " maxamp=" << max_amp << " eps=" << epsilon << ", theta=" << radians_to_degrees(theta) << ", xc=" << xc << ", yc=" << yc << " (npts=" << npts << ")" << endl;
					if (ngrad < npts_sample*npts_frac) {
						warn("RUHROH! not enough points when getting gradient (ngrad=%i,npts_sample=%i,nprev=%i,nnext=%i,npts=%i)",ngrad,npts_sample,prev_npts,next_npts,npts);
						epsilon = epsilon_prev;
						theta = theta_prev;
						xc = xc_prev;
						yc = yc_prev;
						failed_isophote_fit = true;
						break;
					}
					prev_sbgrad = sb_grad;
					prev_rms_sbgrad_rel = rms_sbgrad_rel;
				}
			}
			//if (prev_npts < npts_sample*npts_frac) die();
			//if (next_npts < npts_sample*npts_frac) die();
			//if (ngrad < npts_sample*npts_frac) die();

			if (rms_resid < rms_resid_min) {
				// save params in case this solution is better than the final one
				for (j=0; j < nmax_amp_it; j++) {
					amp_minres[j] = amp[j];
				}
				rms_resid_min = rms_resid;
				it_minres = it;
				xc_minres = xc;
				yc_minres = yc;
				epsilon_minres = epsilon;
				theta_minres = theta;
				sbgrad_minres = sb_grad;
				rms_sbgrad_rel_minres = rms_sbgrad_rel;
				npts_minres = npts;
				npts_sample_minres = npts_sample;
				maxamp_minres = max_amp;
			}

			if (it >= min_it) {
				if ((max_amp < sbfrac*rms_resid) or (it==max_it)) break; // Jedrzejewsi includes the harmonic corrections when calculating the residuals for this criterion. Is it really necessary?
			}

			if (jmax==xc_ampnum) {
				if (emode==0) xc -= amp[xc_ampnum]/sb_grad;
				else xc -= amp[xc_ampnum]/sqrt(1-epsilon)/sb_grad;
			} else if (jmax==yc_ampnum) {
				if (emode==0) yc -= amp[yc_ampnum]*(1-epsilon)/sb_grad;
				else yc -= amp[yc_ampnum]*sqrt(1-epsilon)/sb_grad;
			} else if (jmax==epsilon_ampnum) {
				epsilon += -2*amp[epsilon_ampnum]*(1-epsilon)/(xi*sb_grad);
			} else if (jmax==theta_ampnum) {
				dtheta = 2*amp[theta_ampnum]*(1-epsilon)/(xi*sb_grad*(SQR(1-epsilon)-1));
				theta += dtheta;
				//if (theta > 0) theta -= M_PI;
			}
			it++;
			if ((xc_ampnum >= 0) and ((xc < pixel_xcvals[0]) or (xc > pixel_xcvals[npixels_x-1]))) {
				warn("isofit failed; ellipse center went outside the image (xc=%g, xamp=%g). Moving on to next isophote (npts=%i, npts_sample=%i, npts/npts_sample=%g)",xc,amp[xc_ampnum],npts,npts_sample,(((double) npts)/npts_sample));
				epsilon = epsilon_prev;
				theta = theta_prev;
				xc = xc_prev;
				yc = yc_prev;
				failed_isophote_fit = true;
				break;
			}
			if ((yc_ampnum >= 0) and ((yc < pixel_ycvals[0]) or (yc > pixel_ycvals[npixels_y-1]))) {
				warn("isofit failed; ellipse center went outside the image (yc=%g, yamp=%g). Moving on to next isophote (npts/npts_sample=%g)",yc,amp[yc_ampnum],(((double) npts)/npts_sample));
				epsilon = epsilon_prev;
				theta = theta_prev;
				xc = xc_prev;
				yc = yc_prev;
				failed_isophote_fit = true;
				break;
			}
			if ((epsilon > 1.0) or (epsilon < 0.0)) {
				double frac = ((double) npts)/npts_sample;
				if ((!tried_parameter_search) and (npts > npts_sample*npts_frac)) {
					if (epsilon > 1.0) warn("isofit failed; ellipticity went above 1.0. Will now try a parameter search in epsilon, theta (npts=%i, npts_sample=%i, npts/npts_sample=%g)",npts,npts_sample,frac);
					else warn("isofit failed; ellipticity went below 0.0 (epsilon=%g). Will now try a parameter search in epsilon, theta (npts=%i, npts_sample=%i, npts/npts_sample=%g)",epsilon,npts,npts_sample,frac);

					epsilon = epsilon_prev;
					theta = theta_prev;
					xc = xc_prev;
					yc = yc_prev;
					do_parameter_search = true;
					continue;
				} else {
					if (epsilon > 1.0) warn("isofit failed; ellipticity went above 1.0. Moving on to next isophote (npts=%i, npts_sample=%i, npts/npts_sample=%g)",npts,npts_sample,frac);
					else warn("isofit failed; ellipticity went below 0.0 (epsilon=%g). Moving on to next isophote (npts=%i, npts_sample=%i, npts/npts_sample=%g)",epsilon,npts,npts_sample,frac);

					epsilon = epsilon_prev;
					theta = theta_prev;
					xc = xc_prev;
					yc = yc_prev;
					if (n_harmonics_it > 2) {
						n_harmonics_it--;
						nmax_amp_it -= 2;
						continue;
					}
					failed_isophote_fit = true;
					break;
				}
			}
			if ((jmax==theta_ampnum) and (abs(dtheta) > M_PI)) {
				if ((!tried_parameter_search) and (npts > npts_sample*npts_frac)) {
					warn("isofit failed; position angle jumped by more than 180 degrees(dtheta=%g, theta_amp=%g). Will now try a parameter search in epsilon, theta (npts/npts_sample=%g)",radians_to_degrees(dtheta),amp[theta_ampnum],(((double) npts)/npts_sample));
					epsilon = epsilon_prev;
					theta = theta_prev;
					xc = xc_prev;
					yc = yc_prev;
					do_parameter_search = true;
					continue;
				} else {
					epsilon = epsilon_prev;
					theta = theta_prev;
					xc = xc_prev;
					yc = yc_prev;
					if (n_harmonics_it > 2) {
						n_harmonics_it--;
						nmax_amp_it -= 2;
						continue;
					}
					warn("isofit failed; position angle jumped by more than 180 degrees(dtheta=%g, theta_amp=%g). Moving on to next isophote (npts/npts_sample=%g)",radians_to_degrees(dtheta),amp[theta_ampnum],(((double) npts)/npts_sample));
					failed_isophote_fit = true;
					break;
				}
			}

			if ((sampling_mode==0) and (default_sampling_mode_in != 0) and (rms_sbgrad_rel > rms_sbgrad_rel_transition)) {
				warn("rms_sbgrad_rel (%g) greater than threshold (%g); switching to sector integration mode",rms_sbgrad_rel,rms_sbgrad_rel_transition);
				default_sampling_mode = 1;
				sampling_mode = default_sampling_mode;
				it=0;
				minchisq = 1e30;
				rms_resid_min = 1e30;
				maxamp_min = 1e30;
				if (already_switched) die("SWITCHING AGAIN");
				else already_switched = true;
				continue;
			}
			if (rms_sbgrad_rel > rms_sbgrad_rel_threshold) {
				warn("rms_sbgrad_rel (%g) greater than threshold; retaining previous isofit fit parameters for xi=%g",rms_sbgrad_rel,xi);
				epsilon = epsilon_prev;
				theta = theta_prev;
				xc = xc_prev;
				yc = yc_prev;
				failed_isophote_fit = true;
				break;
			}
		} // End of while loop

		if ((!failed_isophote_fit) and ((rms_sbgrad_rel_minres > rms_sbgrad_rel_threshold) or (npts_minres < npts_frac*npts_sample_minres))) {
			//warn("rms_sbgrad_rel (%g) greater than threshold; retaining previous isofit fit parameters",rms_sbgrad_rel);
			epsilon = epsilon_prev;
			theta = theta_prev;
			xc = xc_prev;
			yc = yc_prev;
			if (xi_it==0) {
				if (rms_sbgrad_rel_minres > rms_sbgrad_rel_threshold) warn("isophote fit cannot have rms_sbgrad_rel > threshold (%g vs %g) on first iteration",rms_sbgrad_rel_minres,rms_sbgrad_rel_threshold);
				else warn("isophote fit cannot have npts/npts_sample < npts_frac (%g vs %g) on first iteration",(((double) npts_minres)/npts_sample_minres),npts_frac);
				int npts_plot;
				if (verbose) sample_ellipse(verbose,xi,xistep,epsilon,theta,xc,yc,npts_plot,npts_sample,emode,sampling_mode,mean_pixel_noise,NULL,NULL,sbprofile,false,NULL,NULL,NULL,lowest_harmonic,2,false,true,&ellout); // make plot
				abort_isofit = true;
				break;
			}
		}
		if ((failed_isophote_fit) and (xi_it==0)) {
			warn("cannot have failed isophote fit on first iteration; aborting isofit");
			abort_isofit = true;
		}

		if (abort_isofit) break;
		if (npts==0) continue; // don't even record previous isophote parameters because we can't even get an estimate for sb_avg or its uncertainty
		if ((failed_isophote_fit) or (rms_sbgrad_rel_minres > rms_sbgrad_rel_threshold) or (npts_minres < npts_frac*npts_sample_minres)) {
			//(*isophote_fit_out) << "The ellipse parameters are epsilon=" << epsilon << ", theta=" << radians_to_degrees(theta) << ", xc=" << xc << ", yc=" << yc << endl;
			//(*isophote_fit_out) << "USING VERY LARGE ERRORS IN STRUCTURAL PARAMS" << endl;
			repeat_params[xi_i] = true;
			isophote_data.sb_avg_vals[xi_i] = sb_avg;
			isophote_data.sb_errs[xi_i] = rms_resid_min/sqrt(npts); // standard error of the mean
			isophote_data.qvals[xi_i] = isophote_data.qvals[xi_i_prev];
			isophote_data.thetavals[xi_i] = isophote_data.thetavals[xi_i_prev];
			isophote_data.xcvals[xi_i] = isophote_data.xcvals[xi_i_prev];
			isophote_data.ycvals[xi_i] = isophote_data.ycvals[xi_i_prev];
			//if (failed_isophote_fit) {
				isophote_data.q_errs[xi_i] = 1e30;
				isophote_data.theta_errs[xi_i] = 1e30;
				isophote_data.xc_errs[xi_i] = 1e30;
				isophote_data.yc_errs[xi_i] = 1e30;
			//} else {
				//isophote_data.q_errs[xi_i] = isophote_data.q_errs[xi_i_prev];
				//isophote_data.theta_errs[xi_i] = isophote_data.theta_errs[xi_i_prev];
				//isophote_data.xc_errs[xi_i] = isophote_data.xc_errs[xi_i_prev];
				//isophote_data.yc_errs[xi_i] = isophote_data.yc_errs[xi_i_prev];
			//}
			isophote_data.A3vals[xi_i] = isophote_data.A3vals[xi_i_prev];
			isophote_data.B3vals[xi_i] = isophote_data.B3vals[xi_i_prev];
			isophote_data.A4vals[xi_i] = isophote_data.A4vals[xi_i_prev];
			isophote_data.B4vals[xi_i] = isophote_data.B4vals[xi_i_prev];
			//if (failed_isophote_fit) {
				isophote_data.A3_errs[xi_i] = 1e30;
				isophote_data.B3_errs[xi_i] = 1e30;
				isophote_data.A4_errs[xi_i] = 1e30;
				isophote_data.B4_errs[xi_i] = 1e30;
			//} else {
				//isophote_data.A3_errs[xi_i] = isophote_data.A3_errs[xi_i_prev];
				//isophote_data.B3_errs[xi_i] = isophote_data.B3_errs[xi_i_prev];
				//isophote_data.A4_errs[xi_i] = isophote_data.A4_errs[xi_i_prev];
				//isophote_data.B4_errs[xi_i] = isophote_data.B4_errs[xi_i_prev];
			//}
			if (n_higher_harmonics > 2) {
				if (n_harmonics_it > 2) {
					isophote_data.A5vals[xi_i] = isophote_data.A5vals[xi_i_prev];
					isophote_data.B5vals[xi_i] = isophote_data.B5vals[xi_i_prev];
					isophote_data.A5_errs[xi_i] = isophote_data.A5_errs[xi_i_prev];
					isophote_data.B5_errs[xi_i] = isophote_data.B5_errs[xi_i_prev];
				} else {
					isophote_data.A5vals[xi_i] = 0;
					isophote_data.B5vals[xi_i] = 0;
					isophote_data.A5_errs[xi_i] = 1e-6;
					isophote_data.B5_errs[xi_i] = 1e-6;
				}
				if (n_harmonics_it > 3) {
					isophote_data.A6vals[xi_i] = isophote_data.A6vals[xi_i_prev];
					isophote_data.B6vals[xi_i] = isophote_data.B6vals[xi_i_prev];
					isophote_data.A6_errs[xi_i] = isophote_data.A6_errs[xi_i_prev];
					isophote_data.B6_errs[xi_i] = isophote_data.B6_errs[xi_i_prev];
				} else {
					isophote_data.A6vals[xi_i] = 0;
					isophote_data.B6vals[xi_i] = 0;
					isophote_data.A6_errs[xi_i] = 1e-6;
					isophote_data.B6_errs[xi_i] = 1e-6;
				}
			}
			if (verbose) {
				if (rms_sbgrad_rel_minres > rms_sbgrad_rel_threshold) (*isophote_fit_out) << "rms_sbgrad_rel > threshold (" << rms_sbgrad_rel_minres << " vs " << rms_sbgrad_rel_threshold << ") --> repeating ellipse parameters, epsilon=" << epsilon << ", theta=" << radians_to_degrees(theta) << ", xc=" << xc << ", yc=" << yc << endl;
				if (npts_minres < npts_frac*npts_sample_minres) (*isophote_fit_out) << "npts/npts_sample < npts_frac (" << (((double) npts_minres)/npts_sample) << " vs " << npts_frac << ") --> repeating ellipse parameters, epsilon=" << epsilon << ", theta=" << radians_to_degrees(theta) << ", xc=" << xc << ", yc=" << yc << endl;
				int npts_plot;
				sample_ellipse(verbose,xi,xistep,epsilon,theta,xc,yc,npts_plot,npts_sample,emode,sampling_mode,mean_pixel_noise,NULL,NULL,sbprofile,false,NULL,NULL,NULL,lowest_harmonic,2,false,true,&ellout); // make plot
			}

		} else {
			if (rms_resid > rms_resid_min) {
				// in this case the final solution was NOT the one with the smallest rms residuals
				for (j=0; j < nmax_amp_it; j++) {
					amp[j] = amp_minres[j];
				}
				xc = xc_minres;
				yc = yc_minres;
				epsilon = epsilon_minres;
				theta = theta_minres;
				sb_grad = sbgrad_minres;
				rms_sbgrad_rel = rms_sbgrad_rel_minres;
				max_amp = maxamp_minres;
				npts = npts_minres;
				npts_sample = npts_sample_minres;

				prev_sbgrad = sb_grad;
				prev_rms_sbgrad_rel = rms_sbgrad_rel;
			}

			if (verbose) {
				(*isophote_fit_out) << "DONE! The final ellipse parameters are epsilon=" << epsilon << ", theta=" << radians_to_degrees(theta) << ", xc=" << xc << ", yc=" << yc << endl;
				(*isophote_fit_out) << "Performed " << it << " iterations; minimum rms_resid=" << rms_resid_min << " achieved during iteration " << it_minres << endl;
			}

			sb_grad = abs(sb_grad);

			//sampling_noise = (sampling_mode==1) ? 2*rms_resid_min : dmin(rms_resid_min,mean_pixel_noise); // if using pixel integration, noise is reduced due to averaging
			if (sampling_mode==1) {
				sampling_noise = 2*rms_resid_min; // if using pixel integration, noise is reduced due to averaging
			} else {
				if (npts <= 30) sampling_noise = mean_pixel_noise/sqrt(3); // if too few points, rms_resid_min is not well-determined so it shouldn't be used for uncertainties
				else sampling_noise = rms_resid_min;
			}
			//sampling_noise = (sampling_mode==1) ? 2*rms_resid_min : mean_pixel_noise;

			sample_ellipse(verbose,xi,xistep,epsilon,theta,xc,yc,npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals,NULL,sbprofile,true,sb_residual,sb_weights,smatrix,lowest_harmonic,2+n_harmonics_it,use_polar_higher_harmonics); // sample again in case we switched back to previous parameters
			fill_matrices(npts,nmax_amp_it,NULL,sb_weights,smatrix,NULL,Smatrix,sampling_noise);
			if (!Cholesky_dcmp(Smatrix,nmax_amp_it)) {
				warn("unexpected failure of Cholesky decomposition; isofit failed");
				return false;
			} else {
				Cholesky_fac_inverse(Smatrix,nmax_amp_it); // Now the lower triangle of Smatrix gives L_inv

				for (i=0; i < nmax_amp_it; i++) {
					amperrs[i] = 0;
					for (k=0; k <= i; k++) {
						amperrs[i] += SQR(Smatrix[i][k]); // just getting the diagonal elements of S_inverse from L_inv*L_inv^T
					}
					amperrs[i] = sqrt(amperrs[i]);
				}
			}
			//if (verbose) {
				//(*isophote_fit_out) << "Untransformed: A3=" << amp[0] << " B3=" << amp[1] << " A4=" << amp[2] << " B4=" << amp[3] << endl;
				//(*isophote_fit_out) << "Untransformed: A3_err=" << amperrs[0] << " B3_err=" << amperrs[1] << " A4_err=" << amperrs[2] << " B4_err=" << amperrs[3] << endl;
			//}

			for (i=n_ellipse_amps; i < nmax_amp_it; i++) {
				amp[i] = -amp[i]/(sb_grad*xi); // this will relate it to the contour shape amplitudes (when perturbing the elliptical radius)
				amperrs[i] = abs(amperrs[i]/(sb_grad*xi));
			}
			if (emode==0) {
				if (xc_ampnum >= 0) xc_err = amperrs[xc_ampnum]*sqrt(sb_grad);
				if (yc_ampnum >= 0) yc_err = amperrs[yc_ampnum]*sqrt((1-epsilon)/sb_grad);
			} else {
				if (xc_ampnum >= 0) xc_err = amperrs[xc_ampnum]*sqrt(1.0/sqrt(1-epsilon)/sb_grad);
				if (yc_ampnum >= 0) yc_err = amperrs[yc_ampnum]*sqrt(sqrt(1-epsilon)/sb_grad);
			}
			epsilon_err = amperrs[epsilon_ampnum]*sqrt(2*(1-epsilon)/(xi*sb_grad));
			if (epsilon_err > 1e10) die("absurd epsilon error! xi=%g, amperr=%g, sb_grad=%g",xi,amperrs[epsilon_ampnum],sb_grad);
			theta_err = amperrs[theta_ampnum]*sqrt(2*(1-epsilon)/(xi*sb_grad*(1-SQR(1-epsilon))));
			if (theta_err > degrees_to_radians(200)) warn("absurd theta error; amperr=%g,sbgrad=%g,epsilon=%g",amperrs[3],sb_grad,epsilon);

			//(*isophote_fit_out) << "AMPERRS: " << amperrs[0] << " " << amperrs[1] << " " << amperrs[2] << " " << amperrs[3] << endl;
			if (verbose) {
				(*isophote_fit_out) << "epsilon_err=" << epsilon_err << ", theta_err=" << radians_to_degrees(theta_err) << ", xc_err=" << xc_err << ", yc_err=" << yc_err << endl;
				(*isophote_fit_out) << "A3=" << amp[n_ellipse_amps] << " B3=" << amp[n_ellipse_amps+1] << " A4=" << amp[n_ellipse_amps+2] << " B4=" << amp[n_ellipse_amps+3] << endl;
				(*isophote_fit_out) << "A3_err=" << amperrs[n_ellipse_amps] << " B3_err=" << amperrs[n_ellipse_amps+1] << " A4_err=" << amperrs[n_ellipse_amps+2] << " B4_err=" << amperrs[n_ellipse_amps+3] << endl;
				if (n_harmonics_it > 2) {
					(*isophote_fit_out) << "A5=" << amp[n_ellipse_amps+4] << " B5=" << amp[n_ellipse_amps+5] << endl;
					(*isophote_fit_out) << "A5_err=" << amperrs[n_ellipse_amps+4] << " B5_err=" << amperrs[n_ellipse_amps+5] << endl;
				}
				if (n_harmonics_it > 3) {
					(*isophote_fit_out) << "A6=" << amp[n_ellipse_amps+6] << " B6=" << amp[n_ellipse_amps+7] << endl;
					(*isophote_fit_out) << "A6_err=" << amperrs[n_ellipse_amps+6] << " B6_err=" << amperrs[n_ellipse_amps+7] << endl;
				}

			}

			if ((verbose) and (sbprofile==NULL)) (*isophote_fit_out) << "Best-fit rms_resid=" << rms_resid_min << ", sb_avg=" << sb_avg << ", sbgrad=" << sbgrad_minres << ", maxamp=" << maxamp_minres << ", rms_sbgrad_rel=" << rms_sbgrad_rel << endl;

			if (verbose) {
				int npts_plot;
				sample_ellipse(verbose,xi,xistep,epsilon,theta,xc,yc,npts_plot,npts_sample,emode,sampling_mode,mean_pixel_noise,NULL,NULL,sbprofile,false,NULL,NULL,NULL,lowest_harmonic,2,false,true,&ellout); // make plot
			}
			if (xi_it==0) {
				// save the initial best-fit values, since we'll use them again as our initial guess when we switch to stepping to smaller xi
				epsilon0 = epsilon;
				theta0 = theta;
				xc0 = xc;
				yc0 = yc;
			}

			isophote_data.sb_avg_vals[xi_i] = sb_avg;
			if (npts_minres < npts_frac_zeroweight*npts_sample_minres) {
				// Here we blow up the error (giving the data point zero weight) if there weren't enough points to sample the surface brightness well enough
				isophote_data.sb_errs[xi_i] = 1e30;
			} else {
				isophote_data.sb_errs[xi_i] = rms_resid_min/sqrt(npts); // standard error of the mean
			}
			isophote_data.qvals[xi_i] = 1 - epsilon;
			isophote_data.thetavals[xi_i] = theta;
			isophote_data.xcvals[xi_i] = xc;
			isophote_data.ycvals[xi_i] = yc;
			isophote_data.q_errs[xi_i] = epsilon_err;
			isophote_data.theta_errs[xi_i] = theta_err;
			isophote_data.xc_errs[xi_i] = xc_err;
			isophote_data.yc_errs[xi_i] = yc_err;
			isophote_data.A3vals[xi_i] = amp[n_ellipse_amps];
			isophote_data.B3vals[xi_i] = amp[n_ellipse_amps+1];
			isophote_data.A4vals[xi_i] = amp[n_ellipse_amps+2];
			isophote_data.B4vals[xi_i] = amp[n_ellipse_amps+3];
			isophote_data.A3_errs[xi_i] = amperrs[n_ellipse_amps];
			isophote_data.B3_errs[xi_i] = amperrs[n_ellipse_amps+1];
			isophote_data.A4_errs[xi_i] = amperrs[n_ellipse_amps+2];
			isophote_data.B4_errs[xi_i] = amperrs[n_ellipse_amps+3];

			if (n_higher_harmonics > 2) {
				if (n_harmonics_it > 2) {
					isophote_data.A5vals[xi_i] = amp[n_ellipse_amps+4];
					isophote_data.B5vals[xi_i] = amp[n_ellipse_amps+5];
					isophote_data.A5_errs[xi_i] = amperrs[n_ellipse_amps+4];
					isophote_data.B5_errs[xi_i] = amperrs[n_ellipse_amps+5];
				} else {
					isophote_data.A5vals[xi_i] = 0;
					isophote_data.B5vals[xi_i] = 0;
					isophote_data.A5_errs[xi_i] = 1e-6;
					isophote_data.B5_errs[xi_i] = 1e-6;
				}
				if (n_harmonics_it > 3) {
					isophote_data.A6vals[xi_i] = amp[n_ellipse_amps+6];
					isophote_data.B6vals[xi_i] = amp[n_ellipse_amps+7];
					isophote_data.A6_errs[xi_i] = amperrs[n_ellipse_amps+6];
					isophote_data.B6_errs[xi_i] = amperrs[n_ellipse_amps+7];
				} else {
					isophote_data.A6vals[xi_i] = 0;
					isophote_data.B6vals[xi_i] = 0;
					isophote_data.A6_errs[xi_i] = 1e-6;
					isophote_data.B6_errs[xi_i] = 1e-6;
				}
			}

		}
		if (abort_isofit) break;

		if ((sbprofile != NULL) and (verbose)) {
			// Now compare to true values...are uncertainties making sense?
			double true_q, true_epsilon, true_theta, true_xc, true_yc, true_A4, true_rms_sbgrad_rel;
			true_xc = sbprofile->sbparams->x_center;
			true_yc = sbprofile->sbparams->y_center;
			if (sbprofile->ellipticity_gradient==true) {
				double eps;
				sbprofile->ellipticity_function(xi,eps,true_theta);
				true_q = sqrt(1-eps);
				true_epsilon = 1-true_q;
			} else {
				true_q = sbprofile->sbparams->q;
				true_epsilon = 1 - true_q;
				true_theta = sbprofile->sbparams->theta;
			}
			true_A4 = 0.0;
			if (sbprofile->n_fourier_modes > 0) {
				for (int i=0; i < sbprofile->n_fourier_modes; i++) {
					if (sbprofile->fourier_mode_mvals[i]==n_ellipse_amps) true_A4 = sbprofile->fourier_mode_cosamp[i];
				}
			}
			(*isophote_fit_out) << "TRUE MODEL: epsilon_true=" << true_epsilon << ", theta_true=" << radians_to_degrees(true_theta) << ", xc_true=" << true_xc << ", yc_true=" << true_yc << ", A4_true=" << true_A4 << endl;
			if (abs(xc-true_xc) > 3*xc_err) warn("RUHROH! xc off by more than 3*xc_err");
			if (abs(yc-true_yc) > 3*yc_err) warn("RUHROH! yc off by more than 3*yc_err");
			if (abs(epsilon - true_epsilon) > 3*epsilon_err) warn("RUHROH! epsilon off by more than 3*epsilon_err");
			if (abs(theta - true_theta) > 3*theta_err) warn("RUHROH! theta off by more than 3*theta_err");
			if (abs(amp[n_ellipse_amps+1] - true_A4) > 3*amperrs[n_ellipse_amps+1]) warn("RUHROH! A4 off by more than 3*A4_err");

			if ((abs(xc-true_xc) < 0.1*xc_err) and (abs(yc-true_yc) < 0.1*yc_err) and (abs(epsilon - true_epsilon) < 0.1*epsilon_err) and (abs(theta - true_theta) < 0.1*theta_err)) warn("Hmm, parameter residuals are all less than 0.1 times the uncertainties. Perhaps uncertainties are inflated?");

			//sb_avg = sample_ellipse(verbose,xi,xistep,true_epsilon,true_theta,true_xc,true_yc,npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals,sbprofile,true,sb_residual,smatrix,1,2+n_higher_harmonics,use_polar_higher_harmonics);
			//rms_resid = 0;
			//for (i=0; i < npts; i++) rms_resid += SQR(sb_residual[i]);
			//rms_resid = sqrt(rms_resid/npts);
			//fill_matrices(npts,nmax_amp,sb_residual,smatrix,Dvec,Smatrix,1.0);
			//if (!Cholesky_dcmp(Smatrix,nmax_amp)) die("Cholesky decomposition failed");
			//Cholesky_solve(Smatrix,Dvec,amp,nmax_amp);
			//double true_max_amp = -1e30;
			////if (verbose) (*isophote_fit_out) << "AMPS: " << amp[0] << " " << amp[1] << " " << amp[2] << " " << amp[3] << endl;
			//for (j=0; j < 4; j++) if (abs(amp[j]) > true_max_amp) { true_max_amp = abs(amp[j]); }

			prev_sb_avg = sample_ellipse(verbose,xi-grad_xistep,xistep,true_epsilon,true_theta,true_xc,true_yc,prev_npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals_prev,sbgrad_weights_prev,sbprofile);
			next_sb_avg = sample_ellipse(verbose,xi+grad_xistep,xistep,true_epsilon,true_theta,true_xc,true_yc,next_npts,npts_sample,emode,sampling_mode,mean_pixel_noise,sbvals_next,sbgrad_weights_next,sbprofile);
			if ((prev_npts==0) or (next_npts==0)) {
				warn("isophote fit failed for true model; no sampling points were accepted on ellipse for determining sbgrad"); 
				abort_isofit = true;
				break;
			}
			else {
				if (!find_sbgrad(npts_sample,sbvals_prev,sbvals_next,sbgrad_weights_prev,sbgrad_weights_next,grad_xistep,sb_grad,true_rms_sbgrad_rel,ngrad)) { abort_isofit = true; break; }
			}

			// Now see what the rms_residual and amps are for true solution
			int true_npts;
			double true_sb_avg = sample_ellipse(verbose,xi,xistep,true_epsilon,true_theta,true_xc,true_yc,true_npts,npts_sample,emode,sampling_mode,mean_pixel_noise,NULL,NULL,sbprofile,true,sb_residual,sb_weights,smatrix,lowest_harmonic,2+n_higher_harmonics,use_polar_higher_harmonics);
			fill_matrices(npts,nmax_amp,sb_residual,sb_weights,smatrix,Dvec,Smatrix,1.0);
			if (!Cholesky_dcmp(Smatrix,nmax_amp)) die("Cholesky decomposition failed");
			Cholesky_solve(Smatrix,Dvec,amp,nmax_amp);

			double true_max_amp = -1e30;
			for (j=0; j < n_ellipse_amps; j++) if (abs(amp[j]) > true_max_amp) { true_max_amp = abs(amp[j]); }

			double rms_resid_true=0;
			for (i=0; i < npts; i++) {
				hterm = sb_residual[i];
				for (j=n_ellipse_amps; j < nmax_amp; j++) hterm -= amp[j]*smatrix[j][i];
				rms_resid_true += hterm*hterm;

			}
			rms_resid_true = sqrt(rms_resid_true/npts);

			for (i=n_ellipse_amps; i < nmax_amp; i++) amp[i] /= (sb_grad*xi); // this will relate it to the contour shape amplitudes (when perturbing the elliptical radius)

			(*isophote_fit_out) << "Best-fit rms_resid=" << rms_resid_min << ", sb_avg=" << sb_avg << ", sbgrad=" << sbgrad_minres << ", maxamp=" << maxamp_minres << ", rms_sbgrad_rel=" << rms_sbgrad_rel << ", npts=" << npts << endl;
			(*isophote_fit_out) << "True rms_resid=" << rms_resid_true << ", sb_avg=" << true_sb_avg << ", sbgrad=" << sb_grad << ", maxamp=" << true_max_amp << ", rms_sbgrad_rel=" << true_rms_sbgrad_rel << ", npts=" << true_npts << endl;
			(*isophote_fit_out) << "True solution: A3=" << amp[n_ellipse_amps] << " B3=" << amp[n_ellipse_amps+1] << " A4=" << amp[n_ellipse_amps+2] << " B4=" << amp[n_ellipse_amps+3] << endl;
			double sbderiv = (sbprofile->sb_rsq(SQR(xi + grad_xistep)) - sbprofile->sb_rsq(SQR(xi-grad_xistep)))/(2*grad_xistep);

			(*isophote_fit_out) << "sb from model at xi (no PSF or harmonics): " << sbprofile->sb_rsq(xi*xi) << ", sbgrad=" << sbderiv << endl;
			if ((rms_resid_true < rms_resid_min) and (rms_resid_min > 1e-8)) // we don't worry about this if rms_resid_min is super small
			{
				if (rms_sbgrad_rel < 0.5) (*isophote_fit_out) << "WARNING: RUHROH! true solution had smaller rms_resid than best-fit, AND rms_sbgrad_rel < 0.5" << endl;
				else (*isophote_fit_out) << "true solution had smaller rms_resid than the best fit!" << endl;
			}
			if (rms_resid_true > rms_resid_min) (*isophote_fit_out) << "NOTE: YOUR BEST-FIT SOLUTION HAS SMALLER RESIDUALS THAN TRUE SOLUTION" << endl;
		}
		if (verbose) (*isophote_fit_out) << endl;
	}
	if (sbprofile != NULL) {
		int nn_plot = imax(100,6*n_xivals);
		string dir = ".";
		if (qlens != NULL) dir = qlens->fit_output_dir;
		sbprofile->plot_ellipticity_function(xi_min,xi_max,nn_plot,dir);
	}

	/*
	double repeat_errfac = 1e30;
	for (i=0; i < n_xivals; i++) {
		if (repeat_params[i]) {
			isophote_data.q_errs[i] *= repeat_errfac;
			isophote_data.theta_errs[i] *= repeat_errfac;
			isophote_data.xc_errs[i] *= repeat_errfac;
			isophote_data.yc_errs[i] *= repeat_errfac;
			isophote_data.A3_errs[i] *= repeat_errfac;
			isophote_data.B3_errs[i] *= repeat_errfac;
			isophote_data.A4_errs[i] *= repeat_errfac;
			isophote_data.B4_errs[i] *= repeat_errfac;
		}
	}
	*/


	delete[] xivals;
	delete[] xi_ivals;
	delete[] xi_ivals_sorted;

	delete[] sb_residual;
	delete[] sb_weights;
	delete[] sbvals;
	delete[] sbvals_next;
	delete[] sbvals_prev;
	delete[] sbgrad_weights_next;
	delete[] sbgrad_weights_prev;
	delete[] Dvec;
	delete[] amp;
	delete[] amperrs;
	for (i=0; i < nmax_amp; i++) {
		delete[] smatrix[i];
		delete[] Smatrix[i];
	}
	delete[] smatrix;
	delete[] Smatrix;
	if (abort_isofit) return false;
	return true;
}

double ImageData::sample_ellipse(const bool show_warnings, const double xi, const double xistep_in, const double epsilon, const double theta, const double xc, const double yc, int& npts, int& npts_sample, const int emode, int& sampling_mode, const double mean_pixel_noise, double *sbvals, double *sbgrad_wgts, SB_Profile* sbprofile, const bool fill_matrices, double* sb_residual, double *sb_weights, double** smatrix, const int ni, const int nf, const bool use_polar_higher_harmonics, const bool plot_ellipse, ofstream *ellout)
{
	if (epsilon > 1) die("epsilon cannot be greater than 1");
	//cout << xi << " " << epsilon << " " << theta << " " << xc << " " << yc << endl;
	int n_amps = 2*(nf-ni+1); // defaults: ni=1, nf=2
	double q, sqrtq, a0, costh, sinth;
	q = 1 - epsilon;
	sqrtq = sqrt(q);
	costh = cos(theta);
	sinth = sin(theta);
	static const int integration_pix_threshold = 60; // ARBITRARY! Should depend on S/N...figure this out later
	a0 = xi;
	if (emode > 0) a0 /= sqrtq; // semi-major axis

	int i, j, k, ii, jj;
	double x,y,xp,yp;
	double eta, eta_step, phi;
	double xstep = pixel_xcvals[1] - pixel_xcvals[0];
	double ystep = pixel_ycvals[1] - pixel_ycvals[0];
	double a0_npix = a0 / pixel_size;
	bool sector_warning = false;

	bool sector_integration;
	bool use_biweight_avg = false; // make this an option the user can change
	// NOTE: sampling_mode == 3 uses an sbprofile for testing purposes
	if ((sampling_mode==0) or (sampling_mode==3)) sector_integration = false;
	else if (sampling_mode==1) sector_integration = true;
	else {
		if (a0_npix < integration_pix_threshold) {
			sector_integration = false;
			sampling_mode = 0;
		}
		else {
			sector_integration = true;
			sampling_mode = 1;
		}
	}

	const int n_sectors = 36;
	double xistep = xistep_in;
	double xifac = (1+xistep);
	double xisqmax = SQR(xi*xifac);
	double xisqmin = SQR(xi/xifac);

	//double xisqmax = SQR(xi*(1 + xifac)/2);
	//double xisqmin = SQR(xi*(1 + 1.0/xifac)/2);

	double annulus_width = sqrt(xisqmax)-sqrt(xisqmin);
	//if (sbgrad_wgts == NULL) {
		// when getting the SB gradient, we allow a thicker annulus to reduce noise in the gradient; otherwise, reduce thickness if it's greater than 4 pixels across
			//while (annulus_width/pixel_size > 6.001) {
				//xistep *= 0.75;
				//xifac = (1+xistep);
				//xisqmax = SQR(xi*xifac);
				//xisqmin = SQR(xi/xifac);
				//annulus_width = sqrt(xisqmax)-sqrt(xisqmin);
			//}
	//}
	if ((sector_integration) and (sqrt(annulus_width*a0*q*M_2PI/n_sectors)/pixel_size < 2.25)) die("annulus sectors too small for sector integration"); // if the annulus is too thin, it causes problems

	const int max_npix_per_sector = 200;
	double *sb_sector;
	double **sector_sbvals;
	bool *sector_in_bounds;
	int *npixels_in_sector;

	if (!sector_integration) {
		if (npts_sample < 0) {
			if (emode==0) eta_step = pixel_size / xi;
			else eta_step = pixel_size / (xi/sqrtq);
			npts_sample = ((int) (M_2PI / eta_step)) + 1; // Then we will readjust eta_step to match npts_sample exactly
		}
	} else {
		if (npts_sample < 0) npts_sample = n_sectors;
		sb_sector = new double[npts_sample];
		sector_in_bounds = new bool[npts_sample];
		if (use_biweight_avg) sector_sbvals = new double*[npts_sample];
		npixels_in_sector = new int[npts_sample];
		for (i=0; i < npts_sample; i++) {
			sb_sector[i] = 0;
			sector_in_bounds[i] = true;
			npixels_in_sector[i] = 0;
			if (use_biweight_avg) sector_sbvals[i] = new double[max_npix_per_sector];
		}
	}
	eta_step = M_2PI / npts_sample;

	/*
	if ((plot_ellipse) and (sector_integration)) {
		ofstream annout("annulus.dat");
		int an_nn = 200;
		double xixi, etastep = M_2PI/(an_nn-1);

		xixi = sqrt(xisqmin);
		for (i=0, eta=0; i < an_nn; i++, eta += etastep) {
			if (emode==0) {
				xp = xixi*cos(eta);
				yp = xixi*q*sin(eta);
			} else {
				xp = xixi*cos(eta)/sqrtq;
				yp = xixi*sin(eta)*sqrtq;
			}
			x = xc + xp*costh - yp*sinth;
			y = yc + xp*sinth + yp*costh;
			annout << x << " " << y << " " << xp << " " << yp << endl;
		}
		annout << endl;

		xixi = sqrt(xisqmax);
		for (i=0, eta=0; i < an_nn; i++, eta += etastep) {
			if (emode==0) {
				xp = xixi*cos(eta);
				yp = xixi*q*sin(eta);
			} else {
				xp = xixi*cos(eta)/sqrtq;
				yp = xixi*sin(eta)*sqrtq;
			}
			x = xc + xp*costh - yp*sinth;
			y = yc + xp*sinth + yp*costh;
			annout << x << " " << y << " " << xp << " " << yp << endl;
		}
		annout << endl;
		double xi_step = (sqrt(xisqmax)-sqrt(xisqmin)) / (an_nn-1);
		for (i=0, eta=-eta_step/2; i < n_sectors; i++, eta += eta_step) {
			if (emode==0) {
				xp = cos(eta);
				yp = q*sin(eta);
			} else {
				xp = cos(eta)/sqrtq;
				yp = sin(eta)*sqrtq;
			}

			for (j=0, xixi = sqrt(xisqmin); j < an_nn; j++, xixi += xi_step) {
				x = xc + xixi*(xp*costh - yp*sinth);
				y = yc + xixi*(xp*sinth + yp*costh);
				annout << x << " " << y << " " << xixi*xp << " " << xixi*yp << endl;
			}
			annout << endl;
		}
		annout.close();
	}
	*/

	//ofstream secout;
	//if (plot_ellipse) secout.open("sectors.dat");
	//ofstream secout2("sectors2.dat");
	double sb, sb_avg = 0;
	double sbavg=0;
	double tt, uu, idoub, jdoub;
	npts = 0;
	bool out_of_bounds = false;
	int wtf=0;
	for (i=0, eta=0; i < npts_sample; i++, eta += eta_step) {
		if (sbvals != NULL) sbvals[i] = NAN; // if it doesn't get changed, then we know that point wasn't assign an SB
		if (emode==0) {
			xp = xi*cos(eta);
			yp = xi*q*sin(eta);
		} else {
			xp = xi*cos(eta)/sqrtq;
			yp = xi*sin(eta)*sqrtq;
		}

		x = xc + xp*costh - yp*sinth;
		y = yc + xp*sinth + yp*costh;
		if (plot_ellipse) (*ellout) << x << " " << y << " " << xp << " " << yp << endl;

		idoub = ((x - pixel_xcvals[0]) / xstep);
		jdoub = ((y - pixel_ycvals[0]) / ystep);
		if ((idoub < 0) or (idoub > (npixels_x-1))) {
			out_of_bounds = true;
			if (sector_integration) sector_in_bounds[i] = false;
			continue;
		}
		if ((jdoub < 0) or (jdoub > (npixels_y-1))) {
			out_of_bounds = true;
			if (sector_integration) sector_in_bounds[i] = false;
			continue;
		}
		ii = (int) idoub;
		jj = (int) jdoub;
		if ((!in_mask[0][ii][jj]) or (!in_mask[0][ii+1][jj]) or (!in_mask[0][ii][jj+1]) or (!in_mask[0][ii+1][jj+1])) {
			if (sector_integration) sector_in_bounds[i] = false;
			continue;
		}

		if (!sector_integration) {
			tt = (x - pixel_xcvals[ii]) / xstep;
			uu = (y - pixel_ycvals[jj]) / ystep;
			if ((tt < 0) or (tt > 1)) die("invalid interpolation parameters: tt=%g id=%g (ii=%i, npx=%i)",tt,idoub,ii,npixels_x);
			if ((uu < 0) or (uu > 1)) die("invalid interpolation parameters: uu=%g jd=%g (jj=%i, npy=%i)",uu,jdoub,jj,npixels_y);
			if ((sampling_mode==3) and (sbprofile != NULL)) {
				sb = sbprofile->surface_brightness(x,y);
			}
			else sb = (1-tt)*(1-uu)*surface_brightness[ii][jj] + tt*(1-uu)*surface_brightness[ii+1][jj] + (1-tt)*uu*surface_brightness[ii][jj+1] + tt*uu*surface_brightness[ii+1][jj+1];
			//if (plot_ellipse) secout << x << " " << y << endl;
			if ((show_warnings) and (sb*0.0 != 0.0)) {
				cout << "ii=" << ii << " jj=" << jj << endl;
				cout << "SB[ii][jj]=" << surface_brightness[ii][jj] << endl;
				cout << "SB[ii+1][jj]=" << surface_brightness[ii+1][jj] << endl;
				cout << "SB[ii][jj+1]=" << surface_brightness[ii][jj+1] << endl;
				cout << "SB[ii+1][jj+1]=" << surface_brightness[ii+1][jj+1] << endl;
				die("got surface brightness = NAN in data image");
			}

			sb_avg += sb;
			if (sbvals != NULL) sbvals[i] = sb;
			if (sbgrad_wgts != NULL) sbgrad_wgts[i] = SQR(mean_pixel_noise)/4.0; // the same number of pixels is always used in interpolation mode, so weights should all be same
		}
		if (fill_matrices) {
			if (!sector_integration) {
				sb_residual[npts] = sb;
				sb_weights[npts] = 4.0/SQR(mean_pixel_noise); // the same number of pixels is always used in interpolation mode, so weights should all be same
			}
			for (j=0, k=ni; j < n_amps; j += 2, k++) {
				if ((use_polar_higher_harmonics) and (k > 2)) {
					phi = atan(yp/xp);
					if (xp < 0) phi += M_PI;
					else if (yp < 0) phi += M_2PI;

					smatrix[j][npts] = cos(k*phi);
					//if (k==4) {
						//cout << (cos(k*phi)) << " " << amp << endl;
					//}
					smatrix[j+1][npts] = sin(k*phi);
				} else {
					smatrix[j][npts] = cos(k*eta);
					smatrix[j+1][npts] = sin(k*eta);
				}
			}
			if (k != nf+1) die("RUHROH");
		
		}
		npts++;
	}
	if (plot_ellipse) {
		// print initial point again just to close the curve
		if (emode==0) xp = xi;
		else xp = xi/sqrtq;
		yp = 0;

		x = xc + xp*costh - yp*sinth;
		y = yc + xp*sinth + yp*costh;
		(*ellout) << x << " " << y << " " << xp << " " << yp << endl << endl;
	}

	double avg_err=0;
	if ((sector_integration) and (npts > 0)) {
		double xmin, xmax, ymin, ymax;
		//xmin = xc - a0*(1+2*annulus_halfwidth_rel);
		//xmax = xc + a0*(1+2*annulus_halfwidth_rel);
		//ymin = yc - a0*(1+2*annulus_halfwidth_rel);
		//ymax = yc + a0*(1+2*annulus_halfwidth_rel);
		
		double amax = sqrt(xisqmax)/sqrt(q);
		xmin = xc - amax;
		xmax = xc + amax;
		ymin = yc - amax;
		ymax = yc + amax;
		int imin, imax, jmin, jmax;
		imin = (int) ((xmin - pixel_xcvals[0]) / xstep);
		jmin = (int) ((ymin - pixel_ycvals[0]) / ystep);
		imax = (int) ((xmax - pixel_xcvals[0]) / xstep);
		jmax = (int) ((ymax - pixel_ycvals[0]) / ystep);
		if (imax <= 0) die("something's gone horribly wrong");
		if (jmax <= 0) die("something's gone horribly wrong");
		if (imin < 0) imin = 0;
		if (jmin < 0) jmin = 0;
		if (imax >= npixels_x) imax = npixels_x-1;
		if (jmax >= npixels_y) jmax = npixels_y-1;
		double xisqval;
		double eta_i, eta_f, eta_width;
		eta_i = -eta_step/2; // since first sampling point is at y=0, the sector begins at negative eta
		eta_f = M_2PI-eta_step/2; // since first sampling point is at y=0, the sector begins at negative eta
		for (ii=imin; ii < imax; ii++) {
			for (jj=jmin; jj < jmax; jj++) {
				if (!in_mask[0][ii][jj]) continue;
				x = pixel_xcvals[ii] - xc;
				y = pixel_ycvals[jj] - yc;
				xp = x*costh + y*sinth;
				yp = -x*sinth + y*costh;
				if (emode==0) xisqval = xp*xp + SQR(yp/q);
				else xisqval = q*(xp*xp) + (yp*yp)/q;
				if ((xisqval > xisqmax) or (xisqval < xisqmin)) continue;
				eta = atan(yp/(q*xp));
				if (xp < 0) eta += M_PI;
				else if (yp < 0) eta += M_2PI;
				if (eta > eta_f) eta -= M_2PI;
				i = (int) ((eta - eta_i) / eta_step);
				if (sector_in_bounds[i]) {
					//cout << "In sector " << i << ": sb=" << surface_brightness[ii][jj] << " ij: " << ii << " " << jj << " comp to " << iivals[i] << " " << jjvals[i] << endl;
					if (use_biweight_avg) sector_sbvals[i][npixels_in_sector[i]] = surface_brightness[ii][jj];
					else sb_sector[i] += surface_brightness[ii][jj];
					//if (i==44) cout << "sector " << i << ": " << surface_brightness[ii][jj] << " pix=" << npixels_in_sector[i] << endl;
					//if (plot_ellipse) secout << pixel_xcvals[ii] << " " << pixel_ycvals[jj] << " " << xmin << " " << xmax << " " << ymin << " " << ymax << " WTF" << endl;
					//secout2 << ii << " " << jj << " " << imin << " " << imax << " " << jmin << " " << jmax << endl;
					npixels_in_sector[i]++;
				}
			}
		}

		int j;
		for (i=0, j=0; i < npts_sample; i++) {
			if (sector_in_bounds[i]) {
				if (npixels_in_sector[i]==0) {
					//cout << "a: " << a0 << " " << a0_npix << endl;
					if (sbvals != NULL) sbvals[i] = NAN;
					if (sbgrad_wgts != NULL) sbgrad_wgts[i] = NAN;
					npts--;
					sector_warning = true;
					//warn("zero pixels in sector? (i=%i)",i);
					if (fill_matrices) {
						for (ii=j; ii < npts; ii++) {
							for (k=0; k < n_amps; k+=2) {
								// this is really ugly!!!! have to move all remaining elements up because we've shortened the list of accepted points
								smatrix[k][ii] = smatrix[k][ii+1];
								smatrix[k+1][ii] = smatrix[k+1][ii+1];
							}
						}
					}
				} else {
					double pixerr = mean_pixel_noise / sqrt(npixels_in_sector[i]);
					avg_err += pixerr;
					//cout << "pixerr=" << pixerr << endl;
					//	cout << "sector " << i << " error: " << pixerr << " " << npixels_in_sector[i] << " " << mean_pixel_noise << endl;
					if (npixels_in_sector[i] < 5) {
						sector_warning = true;
						//warn("less than 5 pixels in sector (%i)",npixels_in_sector[i]);
					}
					if (!use_biweight_avg) {
						sb_sector[i] /= npixels_in_sector[i];
					} else {
						sort(npixels_in_sector[i],sector_sbvals[i]);

						int median_number = npixels_in_sector[i]/2;
						double median;
						if (npixels_in_sector[i] % 2 == 1) {
							median = sector_sbvals[i][median_number]; // odd number
						} else {
							median = 0.5*(sector_sbvals[i][median_number-1] + sector_sbvals[i][median_number]); // even number
						}

						double *distance_to_median = new double[npixels_in_sector[i]];
						for (k=0; k < npixels_in_sector[i]; k++) distance_to_median[k] = abs(sector_sbvals[i][k]-median);
						sort(npixels_in_sector[i],distance_to_median);

						double median_absolute_deviation;
						if (npixels_in_sector[i] % 2 == 1) median_absolute_deviation = distance_to_median[median_number];
						else median_absolute_deviation = 0.5*(distance_to_median[median_number] + distance_to_median[median_number+1]);

						double uu, uu_loc, biweight_location, location_numsum=0, location_denomsum=0;
						for (k=0; k < npixels_in_sector[i]; k++)
						{
							uu = (sector_sbvals[i][k] - median)/9.0/median_absolute_deviation;
							uu_loc = (sector_sbvals[i][k] - median)/6.0/median_absolute_deviation;
							if (abs(uu) < 1)
							{
								location_numsum += (sector_sbvals[i][k] - median)*SQR(1-uu_loc*uu_loc);
								location_denomsum += SQR(1-uu_loc*uu_loc);
							}
						}
						biweight_location = median + location_numsum/location_denomsum;
						delete[] distance_to_median;

						//sb_sector[i] = biweight_location;
						sb_sector[i] = median;
					}

					sb_avg += sb_sector[i];
					if (fill_matrices) {
						sb_residual[j] = sb_sector[i];
						sb_weights[j] = 1.0/SQR(pixerr);
						j++;
					}
					if (sbvals != NULL) sbvals[i] = sb_sector[i];
					if (sbgrad_wgts != NULL) sbgrad_wgts[i] = SQR(pixerr);
					//if (sbgrad_wgts != NULL) sbgrad_wgts[i] = 1.0;
				}
			}
		}
	}
	if (npts==0) sb_avg = 1e30;
	else {
		sb_avg /= npts;
		avg_err /= npts;
		if ((show_warnings) and (avg_err > (0.5*sb_avg))) warn("Average SB error is greater than 20\% of average isophote SB!! Try increasing annulus width (or fewer sectors) to reduce noise");
	}
	if (show_warnings) {
		if (out_of_bounds) (*isophote_fit_out) << "WARNING: part of ellipse was out of bounds of the image" << endl;
		if (sector_warning) (*isophote_fit_out) << "WARNING: less than 5 pixels in at least one sector" << endl;
	}

	if (fill_matrices) {
		for (i=0; i < npts; i++) sb_residual[i] -= sb_avg;
	}
	if (sector_integration) {
		delete[] sb_sector;
		delete[] sector_in_bounds;
		delete[] npixels_in_sector;
		if (use_biweight_avg) {
			for (i=0; i < npts_sample; i++) {
				delete[] sector_sbvals[i];
			}
			delete[] sector_sbvals;
		}
	}
	return sb_avg;
}

bool ImageData::Cholesky_dcmp(double** a, int n)
{
	int i,j,k;

	a[0][0] = sqrt(a[0][0]);
	for (j=1; j < n; j++) a[j][0] /= a[0][0];

	bool status = true;
	for (i=1; i < n; i++) {
		//#pragma omp parallel for private(j,k) schedule(static)
		for (j=i; j < n; j++) {
			for (k=0; k < i; k++) {
				a[j][i] -= a[i][k]*a[j][k];
			}
		}
		if (a[i][i] < 0) {
			status = false;
		}
		a[i][i] = sqrt(abs(a[i][i]));
		for (j=i+1; j < n; j++) a[j][i] /= a[i][i];
	}
	return status;
}

void ImageData::Cholesky_solve(double** a, double* b, double* x, int n)
{
	int i,k;
	double sum;
	for (i=0; i < n; i++) {
		for (sum=b[i], k=i-1; k >= 0; k--) sum -= a[i][k]*x[k];
		x[i] = sum / a[i][i];
	}
	for (i=n-1; i >= 0; i--) {
		for (sum=x[i], k=i+1; k < n; k++) sum -= a[k][i]*x[k];
		x[i] = sum / a[i][i];
	}	 
}

void ImageData::Cholesky_fac_inverse(double** a, int n)
{
	int i,j,k;
	double sum;
	for (i=0; i < n; i++) {
		a[i][i] = 1.0/a[i][i];
		for (j=i+1; j < n; j++) {
			sum = 0.0;
			for (k=i; k < j; k++) sum -= a[j][k]*a[k][i];
			a[j][i] = sum / a[j][j];
		}
	}
}

void ImageData::output_surface_brightness(Vector<double>& xvals_in, Vector<double>& yvals_in, Vector<double>& sbvals_in, bool show_only_mask, bool show_extended_mask, bool show_foreground_mask, const int mask_k)
{
	int i,j,k,l;
	xvals_in.input(npixels_x+1);
	yvals_in.input(npixels_y+1);
	for (int i=0; i <= npixels_x; i++) {
		xvals_in[i] = xvals[i];
	}
	for (int j=0; j <= npixels_y; j++) {
		yvals_in[j] = yvals[j];
	}	
	sbvals_in.input(npixels_x*npixels_y);
	bool show_sb;
	if (show_extended_mask) {
		l=0;
		for (j=0; j < npixels_y; j++) {
			for (i=0; i < npixels_x; i++) {
				if ((!show_only_mask) or (extended_mask == NULL)) show_sb = true;
				else if (mask_k >= 0) {
					if ((extended_mask[mask_k] == NULL) or (extended_mask[mask_k][i][j])) show_sb = true;
					else show_sb = false;
				} else {
					show_sb = false;
					for (k=0; k < n_masks; k++) {
						if ((extended_mask[k]==NULL) or (extended_mask[k][i][j])) {
							show_sb = true;
							break;
						}
					}
				}
				if (show_sb) {
				//if ((!show_only_mask) or (extended_mask == NULL) or (extended_mask[mask_k][i][j])) {
					sbvals_in[l++] = surface_brightness[i][j];
				} else {
					sbvals_in[l++] = numeric_limits<double>::quiet_NaN();
				}
			}
		}
	} else if (show_foreground_mask) {
		l=0;
		for (j=0; j < npixels_y; j++) {
			for (i=0; i < npixels_x; i++) {
				if ((!show_only_mask) or (foreground_mask_data == NULL) or (foreground_mask_data[i][j])) {
					sbvals_in[l++] = surface_brightness[i][j];
				} else {
					sbvals_in[l++] = numeric_limits<double>::quiet_NaN();
				}
			}
		}
	} else {
		l=0;
		for (j=0; j < npixels_y; j++) {
			for (i=0; i < npixels_x; i++) {
				if ((!show_only_mask) or (in_mask == NULL)) show_sb = true;
				else if (mask_k >= 0) {
					if ((in_mask[mask_k] == NULL) or (in_mask[mask_k][i][j])) show_sb = true;
					else show_sb = false;
				} else {
					show_sb = false;
					for (k=0; k < n_masks; k++) {
						if ((in_mask[k]==NULL) or (in_mask[k][i][j])) {
							show_sb = true;
							break;
						}
					}
				}
				//if ((!show_only_mask) or (in_mask == NULL) or (in_mask[mask_k] == NULL) or (in_mask[mask_k][i][j])) {
				if (show_sb) {
					sbvals_in[l++] = surface_brightness[i][j];
				} else {
					sbvals_in[l++] = numeric_limits<double>::quiet_NaN();
				}
			}
		}
	}
}

string ImageData::mkstring_int(const int i)
{
	stringstream istr;
	string istring;
	istr << i;
	istr >> istring;
	return istring;
}

string ImageData::mkstring_doub(const double db)
{
	stringstream dstr;
	string dstring;
	dstr << db;
	dstr >> dstring;
	return dstring;
}

// This function is used by the Python wrapper
string ImageData::get_imgdata_info_string()
{
	string paramstring = "";
	if (band_number >= 0) paramstring += mkstring_int(band_number) + ": (" + mkstring_int(npixels_x) + "," + mkstring_int(npixels_y) + ") image, pixsize=" + mkstring_doub(pixel_size);

	return paramstring;
}

/***************************************** Functions in class PSF ****************************************/

PSF::PSF(QLens* lens_in) : Model()
{
	modelparams = &psf_params;
#ifdef USE_STAN
	modelparams_dif = &psf_params_dif;
#endif
	qlens = lens_in;
	use_input_psf_matrix = false;
	psf_matrix = NULL;
	supersampled_psf_matrix = NULL;
	image_pixel_grid = NULL;
	setup_parameters(true);
	setup_param_pointers<double>();
#ifdef USE_STAN
	setup_param_pointers<stan::math::var>();
#endif
}

void PSF::copy_psf_data(PSF* psf_in)
{
	PSF_Params<double>& p = assign_psf_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.psf_offset_x = psf_in->psf_params.psf_offset_x;
	p.psf_offset_y = psf_in->psf_params.psf_offset_y;
	p.psf_width_x = psf_in->psf_params.psf_width_x;
	p.psf_width_y = psf_in->psf_params.psf_width_y;
	use_input_psf_matrix = psf_in->use_input_psf_matrix;
	if (psf_in->psf_matrix==NULL) psf_matrix = NULL;
	else {
		psf_npixels_x = psf_in->psf_npixels_x;
		psf_npixels_y = psf_in->psf_npixels_y;
		psf_matrix = new double*[psf_npixels_x];
		int i,j;
		for (i=0; i < psf_npixels_x; i++) {
			psf_matrix[i] = new double[psf_npixels_y];
			for (j=0; j < psf_npixels_y; j++) psf_matrix[i][j] = psf_in->psf_matrix[i][j];
		}
		if (psf_in->psf_spline.is_splined()) psf_spline.input(psf_in->psf_spline);
	}
	if (psf_in->supersampled_psf_matrix==NULL) supersampled_psf_matrix = NULL;
	else {
		supersampled_psf_npixels_x = psf_in->supersampled_psf_npixels_x;
		supersampled_psf_npixels_y = psf_in->supersampled_psf_npixels_y;
		supersampled_psf_matrix = new double*[supersampled_psf_npixels_x];
		int i,j;
		for (i=0; i < supersampled_psf_npixels_x; i++) {
			supersampled_psf_matrix[i] = new double[supersampled_psf_npixels_y];
			for (j=0; j < supersampled_psf_npixels_y; j++) supersampled_psf_matrix[i][j] = psf_in->supersampled_psf_matrix[i][j];
		}
	}
	copy_param_arrays(psf_in);
#ifdef USE_STAN
	sync_autodif_parameters();
#endif
}

#ifdef USE_STAN
void PSF::sync_autodif_parameters()
{
	psf_params_dif.psf_offset_x = psf_params.psf_offset_x;
	psf_params_dif.psf_offset_y = psf_params.psf_offset_y;
	psf_params_dif.psf_width_x = psf_params.psf_width_x;
	psf_params_dif.psf_width_y = psf_params.psf_width_y;
}
#endif

void PSF::setup_parameters(const bool initial_setup)
{
	if (initial_setup) {
		// default initial values
		setup_parameter_arrays(4);
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
		paramnames[indx] = "x_offset"; latex_paramnames[indx] = "\\Delta"; latex_param_subscripts[indx] = "x,PSF";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;

	if (initial_setup) {
		paramnames[indx] = "y_offset"; latex_paramnames[indx] = "\\Delta"; latex_param_subscripts[indx] = "y,PSF";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;

	if (initial_setup) {
		paramnames[indx] = "psf_sigx"; latex_paramnames[indx] = "\\sigma"; latex_param_subscripts[indx] = "PSF,x";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if (!use_input_psf_matrix) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		paramnames[indx] = "psf_sigy"; latex_paramnames[indx] = "\\sigma"; latex_param_subscripts[indx] = "PSF,y";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	if (!use_input_psf_matrix) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;
}

template <typename QScalar>
void PSF::setup_param_pointers()
{
	PSF_Params<QScalar>& p = assign_psf_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	p.psf_offset_x = 0;
	p.psf_offset_y = 0;
	p.psf_width_x = 0;
	p.psf_width_y = 0;

	QScalar** param_ptr = p.param;
	*(param_ptr++) = &p.psf_offset_x;
	*(param_ptr++) = &p.psf_offset_y;
	*(param_ptr++) = &p.psf_width_y;
	*(param_ptr++) = &p.psf_width_y;
}
template void PSF::setup_param_pointers<double>();
#ifdef USE_STAN
template void PSF::setup_param_pointers<stan::math::var>();
#endif

void PSF::get_parameter_numbers_from_qlens(int& pi, int& pf)
{
	if (qlens) qlens->get_psf_parameter_numbers(entry_number,pi,pf);
}

/*
/ implement these when you get time
bool PSF::register_vary_parameters_in_qlens()
{
	if (qlens != NULL) {
		return qlens->register_psf_vary_parameters(entry_number);
	}
	return true;
}

void PSF::register_limits_in_qlens()
{
	if (qlens != NULL) {
		qlens->register_psf_prior_limits(entry_number);
	}
}

void PSF::update_fitparams_in_qlens()
{
	if (qlens != NULL) {
		qlens->update_psf_fitparams(entry_number);
	}
}
*/

bool PSF::generate_PSF_matrix(const double xstep, const double ystep, const bool supersampling)
{
	PSF_Params<double>& p = assign_psf_param_object<double>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	if (qlens->psf_threshold==0) return false; // need a threshold to determine where to truncate the PSF
	double sigma_fraction = sqrt(-2*log(qlens->psf_threshold));
	int i,j;
	int nx_half, ny_half, nx, ny;
	double x, y, xmax, ymax;
	if ((p.psf_width_x==0) or (p.psf_width_y==0)) return false;
	double normalization = 0;
	double nx_half_dec, ny_half_dec;
	int supersampling_fac=1;
	if (supersampling) supersampling_fac = qlens->default_imgpixel_nsplit;
	nx_half_dec = sigma_fraction*p.psf_width_x/xstep;
	ny_half_dec = sigma_fraction*p.psf_width_y/ystep;
	nx_half = ((int) nx_half_dec);
	ny_half = ((int) ny_half_dec);
	if ((nx_half_dec - nx_half) > 0.5) nx_half++;
	if ((ny_half_dec - ny_half) > 0.5) ny_half++;
	xmax = nx_half*xstep;
	ymax = ny_half*ystep;
	//cout << "PIXEL LENGTHS: " << xstep << " " << ystep << endl;
	//cout << "nxhalf=" << nx_half << " nyhalf=" << ny_half << " xmax=" << xmax << " ymax=" << ymax << endl;
	nx = (2*nx_half+1)*supersampling_fac;
	ny = (2*ny_half+1)*supersampling_fac;
	if (psf_matrix != NULL) {
		for (i=0; i < psf_npixels_x; i++) delete[] psf_matrix[i];
		delete[] psf_matrix;
	}
	//cout << "NX=" << nx << " NY=" << ny << endl;
	double **psf;
	if (supersampling) {
		supersampled_psf_matrix = new double*[nx];
		for (i=0; i < nx; i++) supersampled_psf_matrix[i] = new double[ny];
		psf = supersampled_psf_matrix;
		supersampled_psf_npixels_x = nx;
		supersampled_psf_npixels_y = ny;
		//cout << "supersampled psf: NX=" << nx << " NY=" << ny << endl;
	} else {
		psf_matrix = new double*[nx];
		for (i=0; i < nx; i++) psf_matrix[i] = new double[ny];
		psf = psf_matrix;
		psf_npixels_x = nx;
		psf_npixels_y = ny;
	}
	double xstep_sup, ystep_sup, xmax_sup, ymax_sup;
	xstep_sup = xstep/supersampling_fac;
	ystep_sup = ystep/supersampling_fac;
	xmax_sup = ((2*nx_half+1)*xstep-xstep_sup)/2;
	ymax_sup = ((2*ny_half+1)*ystep-ystep_sup)/2;
	//cout << "CHECK xstep: " << xstep_sup << " " << ystep_sup << " " << xstep << " " << ystep << " (xmax=" << xmax << ", xmax_sup=" << xmax_sup << ", nx=" << nx << ", supfac=" << supersampling_fac << ")" << endl;
	//cout << "PSF widths: " << psf_width_x << " " << psf_width_y << endl;
	for (i=0, x=-xmax_sup; i < nx; i++, x += xstep_sup) {
		for (j=0, y=-ymax_sup; j < ny; j++, y += ystep_sup) {
			psf[i][j] = exp(-0.5*(SQR(x/p.psf_width_x) + SQR(y/p.psf_width_y)));
			normalization += psf[i][j];
			//cout << "creating PSF: " << i << " " << j << " x=" << x << " y=" << y << " " << psf[i][j] << endl;
		}
	}
	for (i=0; i < nx; i++) {
		for (j=0; j < ny; j++) {
			psf[i][j] /= normalization;
		}
	}
	use_input_psf_matrix = true;
	return true;
}

bool PSF::spline_PSF_matrix(const double xstep, const double ystep)
{
	if (psf_matrix==NULL) return false;
	int i;
	double nx_half,ny_half;
	double x,y;
	nx_half = psf_npixels_x/2.0;
	ny_half = psf_npixels_y/2.0;
	double xmax = nx_half*xstep;
	double ymax = ny_half*ystep;
	double *xvals = new double[psf_npixels_x];
	double *yvals = new double[psf_npixels_y];
	for (i=0, x=-xmax+xstep/2; i < psf_npixels_x; i++, x += xstep) xvals[i] = x;
	for (i=0, y=-ymax+ystep/2; i < psf_npixels_y; i++, y += ystep) yvals[i] = y;
	psf_spline.input(xvals,yvals,psf_matrix,psf_npixels_x,psf_npixels_y);
	delete[] xvals;
	delete[] yvals;
	return true;
}

double PSF::interpolate_PSF_matrix(double x, double y, const bool supersampled)
{
	//x += 0.046;
	//y += 0.02;
	//x -= psf_offset_x;
	//y -= psf_offset_y;
	double psfint;
	if ((psf_spline.is_splined() and (!supersampled))) {
		psfint = psf_spline.splint(x,y);
	} else {
		double scaled_x, scaled_y;
		int ii,jj;
		double nx_half, ny_half;
		if (!supersampled) {
			nx_half = psf_npixels_x/2.0;
			ny_half = psf_npixels_y/2.0;
		} else {
			nx_half = supersampled_psf_npixels_x/2.0;
			ny_half = supersampled_psf_npixels_y/2.0;
		}

		double pixel_xlength, pixel_ylength;
		if (image_pixel_grid != NULL) {
			pixel_xlength = image_pixel_grid->pixel_xlength;
			pixel_ylength = image_pixel_grid->pixel_ylength;
		} else {
			pixel_xlength = qlens->grid_xlength / qlens->n_image_pixels_x;
			pixel_ylength = qlens->grid_ylength / qlens->n_image_pixels_y;
		}

		if (supersampled) {
			pixel_xlength /= qlens->default_imgpixel_nsplit;
			pixel_ylength /= qlens->default_imgpixel_nsplit;
		}

		scaled_x = (x / pixel_xlength) + nx_half;
		scaled_y = (y / pixel_ylength) + ny_half;
		ii = (int) scaled_x;
		jj = (int) scaled_y;
		//cout << "x=" << x << " y=" << y << " ii=" << ii << " jj=" << jj << endl;
		if (!supersampled) {
			if ((ii < 0) or (jj < 0) or (ii >= psf_npixels_x-1) or (jj >= psf_npixels_y-1)) return 0.0;
		} else {
			if ((ii < 0) or (jj < 0) or (ii >= supersampled_psf_npixels_x-1) or (jj >= supersampled_psf_npixels_y-1)) return 0.0;
		}
		double tt,TT,uu,UU;
		tt = scaled_x - ii;
		TT = 1-tt;
		uu = scaled_y - jj;
		UU = 1-uu;
		if (!supersampled) {
			psfint = TT*UU*psf_matrix[ii][jj] + tt*UU*psf_matrix[ii+1][jj] + TT*uu*psf_matrix[ii][jj+1] + tt*uu*psf_matrix[ii+1][jj+1];
		} else {
			psfint = TT*UU*supersampled_psf_matrix[ii][jj] + tt*UU*supersampled_psf_matrix[ii+1][jj] + TT*uu*supersampled_psf_matrix[ii][jj+1] + tt*uu*supersampled_psf_matrix[ii+1][jj+1];
		}
		//cout << "PSF=" << psfint << endl;
		// zeroth order interpolation
		/*
		int II,JJ;
		if (tt < TT) II = ii;
		else II = ii+1;
		if (uu < UU) JJ = jj;
		else JJ = jj+1;
		psfint = psf_matrix[II][JJ];
		*/
	}
	if (psfint < 0) psfint = 0;
	return psfint;
}

void PSF::generate_supersampled_PSF_matrix(const bool downsample, const int downsample_fac)
{
	int i,j;
	if (supersampled_psf_matrix != NULL) {
		for (i=0; i < supersampled_psf_npixels_x; i++) delete[] supersampled_psf_matrix[i];
		delete[] supersampled_psf_matrix;
	}

	double pixel_xlength, pixel_ylength;
	if (image_pixel_grid != NULL) {
		pixel_xlength = image_pixel_grid->pixel_xlength;
		pixel_ylength = image_pixel_grid->pixel_ylength;
	} else {
		pixel_xlength = qlens->grid_xlength / qlens->n_image_pixels_x;
		pixel_ylength = qlens->grid_ylength / qlens->n_image_pixels_y;
	}

	int nx, ny;
	double x, xmax, xstep, y, ymax, ystep;
	xstep = pixel_xlength / qlens->default_imgpixel_nsplit;
	ystep = pixel_ylength / qlens->default_imgpixel_nsplit;

	if ((psf_spline.is_splined()) and (downsample)) {
		int psf0_nx, psf0_ny;
		psf0_nx = (int) (((psf_spline.xmax()-psf_spline.xmin())/pixel_xlength) + 1.000001);
		psf0_ny = (int) (((psf_spline.ymax()-psf_spline.ymin())/pixel_ylength) + 1.000001);
		nx = psf0_nx * qlens->default_imgpixel_nsplit;
		ny = psf0_ny * qlens->default_imgpixel_nsplit;
		cout << "NX=" << nx << " NY=" << ny << endl;
		double check_nx = psf_npixels_x * qlens->default_imgpixel_nsplit;
		double check_ny = psf_npixels_y * qlens->default_imgpixel_nsplit;
		xmax = ((psf0_nx * pixel_xlength)-xstep) / 2;
		ymax = ((psf0_ny * pixel_ylength)-ystep) / 2;
		//cout << "CHECK: " << check_nx << " " << check_ny << endl;
		//cout << "ARGH: " << ((psf_spline.ymax()-psf_spline.ymin())/pixel_ylength) << endl;
	} else {
		nx = psf_npixels_x * qlens->default_imgpixel_nsplit;
		ny = psf_npixels_y * qlens->default_imgpixel_nsplit;
		xmax = ((psf_npixels_x * pixel_xlength)-xstep) / 2;
		ymax = ((psf_npixels_y * pixel_ylength)-ystep) / 2;
	}
	supersampled_psf_matrix = new double*[nx];
	for (i=0; i < nx; i++) supersampled_psf_matrix[i] = new double[ny];

	//double xstep_check = (psf_npixels_x*pixel_xlength) / nx;
	//cout << "CHECK xstep, ystep: " << xstep << " " << ystep << " (xmax=" << xmax << ", ymax=" << ymax << ", nx=" << nx << ", ny=" << ny << ")" << endl;

	double xs, ys, subpixel_xstep, subpixel_ystep, psf_sum, dx, dy;
	int ii, jj, downsample_npix;
	if (downsample) {
		downsample_npix = downsample_fac*downsample_fac;
		subpixel_xstep = xstep / downsample_fac;
		subpixel_ystep = ystep / downsample_fac;
		dx = (subpixel_xstep - xstep) / 2; // these should be negative, since you're going from pixel center to a subpixel center
		dy = (subpixel_ystep - ystep) / 2; // these should be negative, since you're going from pixel center to a subpixel center
		//cout << "DOWNSAMPLE DX=" << dx << " DY=" << dy << endl;
	}
	//cout << "xmax=" << xmax << " xmax_est=" << (-xmax+(nx-1)*xstep) << endl;
	//cout << "ymax=" << ymax << " ymax_est=" << (-ymax+(ny-1)*ystep) << endl;

	double normalization = 0;
	for (i=0, x=-xmax; i < nx; i++, x += xstep) {
		for (j=0, y=-ymax; j < ny; j++, y += ystep) {
			if (!downsample) {
				supersampled_psf_matrix[i][j] = interpolate_PSF_matrix(x,y,false);
			} else {
				psf_sum = 0;
				for (ii=0, xs=x+dx; ii < downsample_fac; ii++, xs += subpixel_xstep) {
					for (jj=0, ys=y+dy; jj < downsample_fac; jj++, ys += subpixel_ystep) {
						psf_sum += interpolate_PSF_matrix(xs,ys,false);
					}
				}
				supersampled_psf_matrix[i][j] = psf_sum / downsample_npix;
			}
			normalization += supersampled_psf_matrix[i][j];
		}
	}
	x -= xstep;
	y -= ystep;
	//cout << "final_x=" << x << ", final_y=" << y << endl;
	supersampled_psf_npixels_x = nx;
	supersampled_psf_npixels_y = ny;
	for (i=0; i < nx; i++) {
		for (j=0; j < ny; j++) {
			supersampled_psf_matrix[i][j] /= normalization;
		}
	}
	use_input_psf_matrix = true;
}

bool PSF::load_psf_fits(string fits_filename, const int hdu_indx, const bool supersampled, const bool show_header, const bool verbal)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else
	string filename = fits_filename;
#if __cplusplus >= 201703L // C++17 standard or later
		if (!filesystem::exists(filename)) {
			filename = "../data/" + fits_filename; // try the data folder
			if (!filesystem::exists(filename)) {
				if (qlens->fit_output_dir != ".") {
					filename = qlens->fit_output_dir + "/" + fits_filename;  // finally, try the chains folder
					if (!filesystem::exists(filename)) return false;
				} else return false;
			}
		}
#endif
	if ((qlens->mpi_id==0) and (filename != fits_filename)) {
		cout << "Loading file '" << filename << "'" << endl;
	}

	use_input_psf_matrix = true;
	bool image_load_status = false;
	int i,j,kk;
	if ((!supersampled) and (psf_matrix != NULL)) {
		for (i=0; i < psf_npixels_x; i++) delete[] psf_matrix[i];
		delete[] psf_matrix;
		psf_matrix = NULL;
	}
	else if ((supersampled) and (supersampled_psf_matrix != NULL)) {
		for (i=0; i < supersampled_psf_npixels_x; i++) delete[] supersampled_psf_matrix[i];
		delete[] supersampled_psf_matrix;
		supersampled_psf_matrix = NULL;
	}

	double **input_psf_matrix;

	fitsfile *fptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix, naxis;
	int nx=0, ny=0;
	long naxes[2] = {1,1};
	double *pixels;
	double peak_sb = -1e30;

	char card[FLEN_CARD];   // Standard string lengths defined in fitsio.h
	int hdutype;
	if (!fits_open_file(&fptr, filename.c_str(), READONLY, &status))
	{
		if (fits_movabs_hdu(fptr, hdu_indx, &hdutype, &status)) // move to HDU given by hdu_indx
			 return false;

		int nkeys;
		fits_get_hdrspace(fptr, &nkeys, NULL, &status); // get # of keywords
		if ((show_header) and (verbal)) {
			for (int ii = 1; ii <= nkeys; ii++) { // Read and print each keywords 
				if (fits_read_record(fptr, ii, card, &status))break;
				string cardstring(card);
				cout << cardstring << endl;
			}
		}

		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) )
		{
			if (naxis == 0) {
				warn("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
				return false;
			} else {
				kk=0;
				long* fpixel = new long[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				nx = naxes[0];
				ny = naxes[1];
				pixels = new double[nx];
				input_psf_matrix = new double*[nx];
				for (i=0; i < nx; i++) input_psf_matrix[i] = new double[ny];
				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					if (fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0], NULL, pixels, NULL, &status) )  // read row of pixels
						break; // jump out of loop on error

					for (i=0; i < naxes[0]; i++) {
						input_psf_matrix[i][j] = pixels[i];
						if (pixels[i] > peak_sb) peak_sb = pixels[i];
					}
				}
				delete[] pixels;
				delete[] fpixel;
				image_load_status = true;
			}
		} else {
			warn("Error: could not read PSF fits file\n");
			return false;
		}
		fits_close_file(fptr, &status);
	} else {
		return false;
	}
	int imid, jmid, imin, imax, jmin, jmax;
	//if (verbal) cout << "nx=" << nx << ", ny=" << ny << endl;
	bool centered_psf = true;
	int nx_orig = nx;
	if ((nx % 2 == 0) or (ny % 2 == 0)) {
		centered_psf = false;
		warn("PSF dimensions are even (PSF not centered); convolutions will be asymmetric");
	}
	//if (nx % 2 == 0) {
		//nx--;
		//centered_psf = true;
	//}
	//if (ny % 2 == 0) {
		//ny--;
		//centered_psf = true;
	//}
	imid = nx/2;
	jmid = ny/2;
	imin = imid;
	imax = imid;
	jmin = jmid;
	jmax = jmid;
	for (i=0; i < nx; i++) {
		for (j=0; j < ny; j++) {
			if ((input_psf_matrix[i][j] > qlens->psf_threshold*peak_sb) or (supersampled)) {
				if (i < imin) imin=i;
				if (i > imax) imax=i;
				if (j < jmin) jmin=j;
				if (j > jmax) jmax=j;
			}
		}
	}
	int nx_half, ny_half;
	nx_half = (imax-imin+1)/2;
	ny_half = (jmax-jmin+1)/2;
	double ***psf;
	int *npix_x, *npix_y;
	if (!supersampled) {
		psf = &psf_matrix;
		npix_x = &psf_npixels_x;
		npix_y = &psf_npixels_y;
	} else {
		psf = &supersampled_psf_matrix;
		npix_x = &supersampled_psf_npixels_x;
		npix_y = &supersampled_psf_npixels_y;
	}
	if (centered_psf) {
		(*npix_x) = 2*nx_half+1;
		(*npix_y) = 2*ny_half+1;
	} else {
		(*npix_x) = 2*nx_half;
		(*npix_y) = 2*ny_half;
	}
	//if (verbal) cout << "np_x=" << (*npix_x) << " np_y=" << (*npix_y) << endl;
	(*psf) = new double*[(*npix_x)];
	for (i=0; i < (*npix_x); i++) (*psf)[i] = new double[(*npix_y)];
	int ii,jj;
	for (ii=0, i=imid-nx_half; ii < (*npix_x); i++, ii++) {
		for (jj=0, j=jmid-ny_half; jj < (*npix_y); j++, jj++) {
			(*psf)[ii][jj] = input_psf_matrix[i][j];
		}
	}
	double psfmax = -1e30;
	double normalization = 0;
	for (i=0; i < (*npix_x); i++) {
		for (j=0; j < (*npix_y); j++) {
			if ((*psf)[i][j] > psfmax) psfmax = (*psf)[i][j];
			normalization += (*psf)[i][j];
		}
	}
	for (i=0; i < (*npix_x); i++) {
		for (j=0; j < (*npix_y); j++) {
			(*psf)[i][j] /= normalization;
		}
	}

	//for (i=0; i < psf_npixels_x; i++) {
		//for (j=0; j < psf_npixels_y; j++) {
			//cout << psf_matrix[i][j] << " ";
		//}
		//cout << endl;
	//}
	//cout << psf_npixels_x << " " << psf_npixels_y << " " << nx_half << " " << ny_half << endl;

	if (verbal) {
		cout << "PSF matrix dimensions: " << (*npix_x) << " " << (*npix_y) << " (input PSF dimensions: " << nx << " " << ny << ")" << endl;
		//cout << "PSF normalization =" << normalization << endl << endl;
	}
	for (i=0; i < nx_orig; i++) delete[] input_psf_matrix[i];
	delete[] input_psf_matrix;

	if (status) fits_report_error(stderr, status); // print any error message
	if (image_load_status) psf_filename = filename;

	setup_parameters(false); // since now we're using a pixel map instead of analytic PSF
	return image_load_status;
#endif
}

bool PSF::save_psf_fits(string fits_filename, const bool supersampled)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return false;
#else
	int i,j,kk;
	fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix = -64, naxis = 2;
	int nx, ny;
	double **psf;
	if (psf_matrix == NULL) {
		warn("no PSF matrix has been loaded or generated; cannot save PSF to FITS file");
		return false;
	}
	if ((supersampled) and (supersampled_psf_matrix == NULL)) {
		warn("no supersampled PSF matrix has been loaded or generated; cannot save PSF to FITS file");
		return false;
	}
	if (supersampled) {
		nx = supersampled_psf_npixels_x;
		ny = supersampled_psf_npixels_y;
		psf = supersampled_psf_matrix;
	} else {
		nx = psf_npixels_x;
		ny = psf_npixels_y;
		psf = psf_matrix;
	}
	long naxes[2] = {nx,ny};
	double *pixels;
	string fits_filename_overwrite = "!" + fits_filename; // ensures that it overwrites an existing file of the same name

	if (!fits_create_file(&outfptr, fits_filename_overwrite.c_str(), &status))
	{
		if (!fits_create_img(outfptr, bitpix, naxis, naxes, &status))
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long* fpixel = new long[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[nx];

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=0; i < nx; i++) {
						pixels[i] = psf[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] fpixel;
				delete[] pixels;
			}

				/*
				// for playing around with supersampling
				pixels = new double[psf_npixels_x*4];

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; j++)
				{
					int k=0;
					for (i=0; i < psf_npixels_x; i++) {
						pixels[k++] = psf_matrix[i][j];
						pixels[k++] = psf_matrix[i][j];
						pixels[k++] = psf_matrix[i][j];
						pixels[k++] = psf_matrix[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
					fpixel[1]++;
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
					fpixel[1]++;
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
					fpixel[1]++;
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
					fpixel[1]++;
				}
				*/

		}
		fits_close_file(outfptr, &status);
	} 

	if (status) fits_report_error(stderr, status); // print any error message
	psf_filename = fits_filename;
	return true;
#endif
}

bool PSF::plot_psf(string outfile_root, const bool supersampled, const double xstep, const double ystep)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return false;
#else

	int i,j;

	string psf_filename = outfile_root + ".dat";
	string x_filename = outfile_root + ".x";
	string y_filename = outfile_root + ".y";

	ofstream psf_image_file; qlens->open_output_file(psf_image_file,psf_filename);
	ofstream psf_xvals; qlens->open_output_file(psf_xvals,x_filename);
	ofstream psf_yvals; qlens->open_output_file(psf_yvals,y_filename);
	psf_image_file << setiosflags(ios::scientific);

	int nx, ny;
	double **psf;
	double normfac = 1.0;
	if (psf_matrix == NULL) {
		warn("no PSF matrix has been loaded or generated; cannot save PSF to FITS file");
		return false;
	}
	if ((supersampled) and (supersampled_psf_matrix == NULL)) {
		warn("no supersampled PSF matrix has been loaded or generated; cannot save PSF to FITS file");
		return false;
	}
	if (supersampled) {
		nx = supersampled_psf_npixels_x;
		ny = supersampled_psf_npixels_y;
		psf = supersampled_psf_matrix;
		normfac = ((double) (supersampled_psf_npixels_x*supersampled_psf_npixels_y)) / (psf_npixels_x*psf_npixels_y);
	} else {
		nx = psf_npixels_x;
		ny = psf_npixels_y;
		psf = psf_matrix;
	}
	double imid,jmid;
	imid = nx/2.0;
	jmid = ny/2.0;
	//cout << "imid=" << imid << " jmid=" << jmid << endl;
	//cout << "xmin=" << (-imid)*xstep << endl;
	//cout << "xmax=" << -imid*xstep + nx*xstep << endl;
	//cout << "ymin=" << (-jmid)*ystep << endl;
	//cout << "ymax=" << -jmid*ystep + ny*ystep << endl;
	double x,y;
	double maxval = -1e30, max_x, max_y;
	int max_i, max_j;
	for (i=0, x=(-imid)*xstep; i <= nx; i++, x += xstep) {
		psf_xvals << x << endl; 
	}
	for (j=0, y=(-jmid)*ystep; j <= ny; j++, y += ystep) {
		psf_yvals << y << endl;
	}	
	for (j=0, y=(-jmid+0.5)*ystep; j < ny; j++, y += ystep) {
		for (i=0, x=(-imid+0.5)*xstep; i < nx; i++, x += xstep) {
			if (abs(psf[i][j]) > maxval) {
				maxval = abs(psf[i][j]);
				max_x = x;
				max_y = y;
				max_i = i;
				max_j = j;
			}
			psf_image_file << normfac*abs(psf[i][j]) << " "; // the normfac makes it so the supersampled PSF has the same z-scale as the unsupersampled PSF
		}
		psf_image_file << endl;
	}
	cout << "Max PSF value located at x=" << max_x << ", y=" << max_y << ", i=" << max_i << ", j=" << max_j << endl;
	return true;
#endif
}

void PSF::delete_psf_matrix()
{
	use_input_psf_matrix = false;
	if (psf_matrix != NULL) {
		for (int i=0; i < psf_npixels_x; i++) delete[] psf_matrix[i];
		delete[] psf_matrix;
	}
	if (supersampled_psf_matrix != NULL) {
		for (int i=0; i < supersampled_psf_npixels_x; i++) delete[] supersampled_psf_matrix[i];
		delete[] supersampled_psf_matrix;
	}
	if (psf_spline.is_splined()) psf_spline.unspline();
	psf_matrix = NULL;
	supersampled_psf_matrix = NULL;
}

PSF::~PSF()
{
	if (psf_matrix != NULL) {
		for (int i=0; i < psf_npixels_x; i++) delete[] psf_matrix[i];
		delete[] psf_matrix;
	}
	if (supersampled_psf_matrix != NULL) {
		for (int i=0; i < supersampled_psf_npixels_x; i++) delete[] supersampled_psf_matrix[i];
		delete[] supersampled_psf_matrix;
	}
}

/***************************************** Functions in class ImagePixelGrid ****************************************/

ImagePixelGrid::ImagePixelGrid(QLens* lens_in, SourceFitMode mode, double xmin_in, double xmax_in, double ymin_in, double ymax_in, int x_N_in, int y_N_in, const bool raytrace, const int band_number_in, const int src_redshift_index_in, const int imggrid_index_in) : qlens(lens_in), xmin(xmin_in), xmax(xmax_in), ymin(ymin_in), ymax(ymax_in), x_N(x_N_in), y_N(y_N_in), cartesian_srcgrid(NULL), delaunay_srcgrid(NULL), lensgrid(NULL)
{
	source_fit_mode = mode;
	include_potential_perturbations = false;
	imgpixel_nsplit = 1;
	n_subpix_per_pixel = 1;
	setup_pixel_arrays();
	setup_noise_map(lens_in);
	image_data = NULL;
	band_number = band_number_in;
	imggrid_index = imggrid_index_in;
	if (band_number < lens_in->n_psf) psf = lens_in->psf_list[band_number];
	else psf = NULL;

	src_redshift_index = src_redshift_index_in;
	if (src_redshift_index == -1) {
		imggrid_zfactors = qlens->reference_zfactors;
		imggrid_betafactors = qlens->default_zsrc_beta_factors;
	} else {
		if (qlens->extended_src_zfactors != NULL) {
			imggrid_zfactors = qlens->extended_src_zfactors[src_redshift_index];
			imggrid_betafactors = qlens->extended_src_beta_factors[src_redshift_index];
		} else {
			imggrid_zfactors = NULL;
			imggrid_betafactors = NULL;
		}
	}
	n_pixsrc_to_include_in_Lmatrix = 1; // default: one pixellated source (associated with the current ImagePixelGrid) is included in Lmatrix
	imggrid_indx_to_include_in_Lmatrix.push_back(imggrid_index);

	pixel_xlength = (xmax-xmin)/x_N;
	pixel_ylength = (ymax-ymin)/y_N;
	pixel_area = pixel_xlength*pixel_ylength;
	triangle_area = 0.5*pixel_xlength*pixel_ylength;

	double x,y;
	int i,j;
	for (j=0; j <= y_N; j++) {
		y = ymin + j*pixel_ylength;
		for (i=0; i <= x_N; i++) {
			x = xmin + i*pixel_xlength;
			if ((i < x_N) and (j < y_N)) {
				center_pts[i][j][0] = x + 0.5*pixel_xlength;
				center_pts[i][j][1] = y + 0.5*pixel_ylength;
			}
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
		}
	}
	mask = NULL;
	emask = NULL;
	fgmask = NULL;
	if (raytrace) {
		std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;
		if (qlens->show_wtime) {
			wtime0 = std::chrono::steady_clock::now();
		}
		setup_ray_tracing_arrays();
		if ((qlens->nlens > 0) and (imggrid_zfactors != NULL)) calculate_sourcepts_and_areas<PlainTypes>(true);
		if (qlens->show_wtime) {
			wtime = std::chrono::steady_clock::now() - wtime0;
			if (qlens->mpi_id==0) cout << "Wall time for creating and ray-tracing image pixel grid: "  << wtime.count() << endl;
		}
	} else {
		set_nsplits_from_lens_settings();
	}
}

ImagePixelGrid::ImagePixelGrid(QLens* lens_in, SourceFitMode mode, ImageData& pixel_data, const bool include_fgmask, const int band_number_in, const int src_redshift_index_in, const int imggrid_index_in, const int mask_index, const bool setup_mask_and_data, const bool verbal) : cartesian_srcgrid(NULL), delaunay_srcgrid(NULL), lensgrid(NULL)
{
	// with this constructor, we create the arrays but don't actually make any lensing calculations, since these will be done during each likelihood evaluation
	qlens = lens_in;
	source_fit_mode = mode;
	include_potential_perturbations = false;
	pixel_data.get_grid_params(xmin,xmax,ymin,ymax,x_N,y_N);
	src_xmin = -1e30; src_xmax = 1e30;
	src_ymin = -1e30; src_ymax = 1e30;
	image_data = &pixel_data;
	band_number = band_number_in;
	imggrid_index = imggrid_index_in;
	if (band_number < lens_in->n_psf) psf = lens_in->psf_list[band_number];
	else psf = NULL;
	imgpixel_nsplit = 1;
	n_subpix_per_pixel = 1;

	setup_pixel_arrays();

	src_redshift_index = src_redshift_index_in;
	//cout << "src_i=" << src_redshift_index << " n_src_redshifts=" << qlens->n_extended_src_redshifts << endl;
	if ((src_redshift_index == -1) or (qlens->n_lens_redshifts==0)) {
		imggrid_zfactors = qlens->reference_zfactors;
		imggrid_betafactors = qlens->default_zsrc_beta_factors;
	} else {
		if (src_redshift_index >= qlens->n_extended_src_redshifts) die("invalid extended source redshift index (%i)",src_redshift_index);
		if (qlens->extended_src_zfactors == NULL) die("extended_src_zfactors is NULL");
		if (qlens->extended_src_beta_factors == NULL) die("extended_src_beta_factors is NULL");
		imggrid_zfactors = qlens->extended_src_zfactors[src_redshift_index];
		imggrid_betafactors = qlens->extended_src_beta_factors[src_redshift_index];
	}
	n_pixsrc_to_include_in_Lmatrix = 1; // default: one pixellated source (associated with the current ImagePixelGrid) is included in Lmatrix
	imggrid_indx_to_include_in_Lmatrix.push_back(imggrid_index);

	pixel_xlength = (xmax-xmin)/x_N;
	pixel_ylength = (ymax-ymin)/y_N;
	max_sb = -1e30;
	pixel_area = pixel_xlength*pixel_ylength;
	triangle_area = 0.5*pixel_xlength*pixel_ylength;

	if (setup_mask_and_data) {
		if (mask_index >= image_data->n_masks) die("image_pixel_grid initialized with mask index that doesn't correspond to a mask that has been created/loaded (mask_i=%i,n_masks=%i)",mask_index,image_data->n_masks);
		mask = image_data->in_mask[mask_index];
		emask = image_data->extended_mask[mask_index];
		fgmask = image_data->foreground_mask;
	} else {
		mask = NULL;
		emask = NULL;
		fgmask = NULL;
	}

	int i,j;
	double x,y;
	for (j=0; j <= y_N; j++) {
		y = image_data->yvals[j];
		for (i=0; i <= x_N; i++) {
			x = image_data->xvals[i];
			if ((i < x_N) and (j < y_N)) {
				center_pts[i][j][0] = x + 0.5*pixel_xlength;
				center_pts[i][j][1] = y + 0.5*pixel_ylength;
				if (setup_mask_and_data) {
					surface_brightness[i][j] = image_data->surface_brightness[i][j];
					noise_map[i][j] = image_data->noise_map[i][j];
					if (!include_fgmask) pixel_in_mask[i][j] = image_data->in_mask[mask_index][i][j];
					else pixel_in_mask[i][j] = image_data->foreground_mask[i][j];
					if (surface_brightness[i][j] > max_sb) max_sb=surface_brightness[i][j];
				}
			}
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
		}
	}

	if (setup_mask_and_data) setup_ray_tracing_arrays(true,verbal);
	else set_nsplits_from_lens_settings(); // still setup the subpixels, even if the ray tracing arrays aren't being set up
}

void ImagePixelGrid::set_image_pixel_data(ImageData* imgdata, const int mask_index)
{
	image_data = imgdata;
	if (mask_index >= image_data->n_masks) die("image_pixel_grid initialized with mask index that doesn't correspond to a mask that has been created/loaded (mask_i=%i,n_masks=%i)",mask_index,image_data->n_masks);
	mask = image_data->in_mask[mask_index];
	emask = image_data->extended_mask[mask_index];
	fgmask = image_data->foreground_mask;

	int i,j;
	for (j=0; j <= y_N; j++) {
		for (i=0; i <= x_N; i++) {
			if ((i < x_N) and (j < y_N)) {
				surface_brightness[i][j] = image_data->surface_brightness[i][j];
				noise_map[i][j] = image_data->noise_map[i][j];
				if (!qlens->include_fgmask_in_inversion) pixel_in_mask[i][j] = image_data->in_mask[mask_index][i][j];
				else pixel_in_mask[i][j] = image_data->foreground_mask[i][j];
				if (surface_brightness[i][j] > max_sb) max_sb=surface_brightness[i][j];
			}
		}
	}

	setup_ray_tracing_arrays(true,false);
}


void ImagePixelGrid::set_include_in_Lmatrix(const int imggrid_i)
{
	if ((n_pixsrc_to_include_in_Lmatrix==1) and (imggrid_i != src_redshift_index)) {
		n_pixsrc_to_include_in_Lmatrix = 2; 
		imggrid_indx_to_include_in_Lmatrix.push_back(imggrid_i);
	}
}

void ImagePixelGrid::set_include_only_one_pixsrc_in_Lmatrix()
{
	while (n_pixsrc_to_include_in_Lmatrix > 1) {
		n_pixsrc_to_include_in_Lmatrix--; 
		imggrid_indx_to_include_in_Lmatrix.pop_back();
	}
}

void ImagePixelGrid::update_zfactors_and_beta_factors()
{
	// this is necessary if any qlens redshifts are modified or added
	//cout << "INDEX: " << src_redshift_index << endl;
	if (src_redshift_index != -1) {
		if (qlens->extended_src_zfactors != NULL) {
			imggrid_zfactors = qlens->extended_src_zfactors[src_redshift_index];
			imggrid_betafactors = qlens->extended_src_beta_factors[src_redshift_index];
		} else {
			imggrid_zfactors = NULL;
			imggrid_betafactors = NULL;
		}
	}
}

void ImagePixelGrid::setup_noise_map(QLens* lens_in)
{
	int i,j;
	if ((lens_in->use_noise_map) and (image_data)) {
		double **nptr = image_data->noise_map;
		for (i=0; i < x_N; i++) {
			for (j=0; j < y_N; j++) {
				noise_map[i][j] = nptr[i][j];
			}
		}
	} else {
		for (i=0; i < x_N; i++) {
			for (j=0; j < y_N; j++) {
				noise_map[i][j] = lens_in->background_pixel_noise;
			}
		}
	}
}

void ImagePixelGrid::setup_pixel_arrays()
{
	// This function is called by the constructor only. All arrays that are not initialized are set to null at the end of the function
	xy_N = x_N*y_N;

	corner_pts = new lensvector<double>*[x_N+1];
	corner_sourcepts = new lensvector<double>*[x_N+1];
	center_pts = new lensvector<double>*[x_N];
	center_sourcepts = new lensvector<double>*[x_N];
	pixel_in_mask = new bool*[x_N];
	maps_to_source_pixel = new bool*[x_N];
	pixel_index = new int*[x_N];
	pixel_index_fgmask = new int*[x_N];
	mapped_cartesian_srcpixels = new vector<CartesianSourcePixel*>*[x_N];
	mapped_delaunay_srcpixels = new vector<PtsWgts<double>>*[x_N];
	mapped_potpixels = new vector<PtsWgts<double>>*[x_N];
	n_mapped_srcpixels = new int**[x_N];
	n_mapped_potpixels = new int**[x_N];
	surface_brightness = new double*[x_N];
	foreground_surface_brightness = new double*[x_N];
	noise_map = new double*[x_N];
	source_plane_triangle1_area = new double*[x_N];
	source_plane_triangle2_area = new double*[x_N];
	pixel_mag = new double*[x_N];
	max_nsplit = imax(8,qlens->default_imgpixel_nsplit);
	//max_nsplit = qlens->default_imgpixel_nsplit;
	//nsplits = new int*[x_N];
	subpixel_maps_to_srcpixel = new bool**[x_N];
	subpixel_center_pts = new lensvector<double>**[x_N];
	subpixel_center_sourcepts = new lensvector<double>**[x_N];
	subpixel_source_gradient = new lensvector<double>**[x_N]; // used for finding first-order change in surface brightness from potential perturbations
	subpixel_surface_brightness = new double**[x_N];
	//S0_check = new double*[x_N]; // used for finding first-order change in surface brightness from potential perturbations
	//x0_check = new lensvector<double>*[x_N]; // used for finding first-order change in surface brightness from potential perturbations
	subpixel_weights = new double**[x_N];
	subpixel_index_ss = new int*[x_N*max_nsplit];
	twist_pts = new lensvector<double>*[x_N];
	twist_status = new int*[x_N];

	int i,j,k;
	for (i=0; i <= x_N; i++) {
		corner_pts[i] = new lensvector<double>[y_N+1];
		corner_sourcepts[i] = new lensvector<double>[y_N+1];
	}
	for (i=0; i < x_N; i++) {
		center_pts[i] = new lensvector<double>[y_N];
		center_sourcepts[i] = new lensvector<double>[y_N];
		maps_to_source_pixel[i] = new bool[y_N];
		pixel_in_mask[i] = new bool[y_N];
		pixel_index[i] = new int[y_N];
		pixel_index_fgmask[i] = new int[y_N];
		surface_brightness[i] = new double[y_N];
		foreground_surface_brightness[i] = new double[y_N];
		noise_map[i] = new double[y_N];
		source_plane_triangle1_area[i] = new double[y_N];
		source_plane_triangle2_area[i] = new double[y_N];
		pixel_mag[i] = new double[y_N];
		mapped_cartesian_srcpixels[i] = new vector<CartesianSourcePixel*>[y_N];
		mapped_delaunay_srcpixels[i] = new vector<PtsWgts<double>>[y_N];
		mapped_potpixels[i] = new vector<PtsWgts<double>>[y_N];
		n_mapped_srcpixels[i] = new int*[y_N];
		n_mapped_potpixels[i] = new int*[y_N];
		subpixel_maps_to_srcpixel[i] = new bool*[y_N];
		subpixel_center_pts[i] = new lensvector<double>*[y_N];
		subpixel_center_sourcepts[i] = new lensvector<double>*[y_N];
		subpixel_surface_brightness[i] = new double*[y_N];
		subpixel_source_gradient[i] = new lensvector<double>*[y_N];
		//S0_check[i] = new double[y_N];
		//x0_check[i] = new lensvector<double>[y_N];
		subpixel_weights[i] = new double*[y_N];
		twist_pts[i] = new lensvector<double>[y_N];
		twist_status[i] = new int[y_N];
		for (j=0; j < y_N; j++) {
			surface_brightness[i][j] = 0;
			foreground_surface_brightness[i][j] = 0;
			noise_map[i][j] = 0;
			pixel_in_mask[i][j] = true; // default, since no mask has been introduced yet
			maps_to_source_pixel[i][j] = true;
			pixel_index[i][j] = -1;
			//if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) maps_to_source_pixel[i][j] = true; // in this mode you can always get a surface brightness for any image pixel
			//nsplits[i][j] = lens_in->default_imgpixel_nsplit; // default
			subpixel_maps_to_srcpixel[i][j] = new bool[max_nsplit*max_nsplit];
			subpixel_center_pts[i][j] = new lensvector<double>[max_nsplit*max_nsplit];
			subpixel_center_sourcepts[i][j] = new lensvector<double>[max_nsplit*max_nsplit];
			subpixel_surface_brightness[i][j] = new double[max_nsplit*max_nsplit];
			subpixel_source_gradient[i][j] = new lensvector<double>[max_nsplit*max_nsplit];
			subpixel_weights[i][j] = new double[max_nsplit*max_nsplit];
			n_mapped_srcpixels[i][j] = new int[max_nsplit*max_nsplit];
			n_mapped_potpixels[i][j] = new int[max_nsplit*max_nsplit];
			for (k=0; k < max_nsplit*max_nsplit; k++) subpixel_maps_to_srcpixel[i][j][k] = false;
		}
	}
	int max_subpixel_nx = x_N*max_nsplit;
	int max_subpixel_ny = y_N*max_nsplit;
	for (i=0; i < max_subpixel_nx; i++) {
		subpixel_index_ss[i] = new int[max_subpixel_ny];
	}
	set_null_ray_tracing_arrays();
	set_null_subpixel_ray_tracing_arrays();

	image_pixel_i_from_subcell_ii = NULL;
	image_pixel_j_from_subcell_jj = NULL;

	n_sbweights = 0;
	saved_sbweights = NULL;

	psf_convolution_is_setup = false;
	fg_psf_convolution_is_setup = false;
	fft_convolution_is_setup = false;
	fg_fft_convolution_is_setup = false;

	Fmatrix_sparse = NULL;
	Fmatrix_copy = NULL;
	Fmatrix_index = NULL;
	Fmatrix_nn = 0;
	n_src_inv = 0;

	Rmatrix_sparse = NULL;
	Rmatrix_index = NULL;

	Rmatrix_MGE_packed = NULL;
	Rmatrix_MGE_log_determinants = NULL;
	mge_list = NULL;

	Rmatrix_pot = NULL;
	Rmatrix_pot_index = NULL;
	reg_weight_factor = NULL;
	image_pixel_location_Lmatrix = NULL;
	Lmatrix_sparse = NULL;
	Lmatrix_index = NULL;
}

void ImagePixelGrid::set_null_ray_tracing_arrays()
{
	Lmatrix_n_amps = 0;
	twiststat = NULL;
	//imggrid_params.set_null_ray_tracing_arrays();
//#ifdef USE_STAN
	//imggrid_params_dif.set_null_ray_tracing_arrays();
//#endif

	mask_pixels_i = NULL;
	mask_pixels_j = NULL;
	emask_pixels_i = NULL;
	emask_pixels_j = NULL;
	fgmask_pixels_i = NULL;
	fgmask_pixels_j = NULL;
	masked_pixel_corner_i = NULL;
	masked_pixel_corner_j = NULL;
	masked_pixel_corner = NULL;
	masked_pixel_corner_up = NULL;
	ncvals = NULL;
	centerpts_x = NULL;
	centerpts_y = NULL;
	twistx = NULL;
	twisty = NULL;
	srcpt_x_corners = NULL;
	srcpt_y_corners = NULL;
	area_tri1 = NULL;
	area_tri2 = NULL;
}

void ImagePixelGrid::set_null_subpixel_ray_tracing_arrays()
{
	extended_mask_subpixel_i = NULL;
	extended_mask_subpixel_j = NULL;
	extended_mask_subpixel_index = NULL;
	emask_subpixels_ii = NULL;
	emask_subpixels_jj = NULL;
	mask_subpixel_i = NULL;
	mask_subpixel_j = NULL;
	mask_subpixel_index = NULL;
	image_pixel_i_from_subcell_ii = NULL;
	image_pixel_j_from_subcell_jj = NULL;

	//imggrid_params.set_null_subpixel_ray_tracing_arrays();
//#ifdef USE_STAN
	//imggrid_params_dif.set_null_subpixel_ray_tracing_arrays();
//#endif
}

void ImagePixelGrid::setup_ray_tracing_arrays(const bool include_fft_arrays, const bool verbal)
{
	int i,j,k,n,n_cell,n_corner;

	if ((!pixel_in_mask) or (emask == NULL) or (fgmask == NULL)) {
		image_npixels = x_N*y_N;
		image_npixels_emask = x_N*y_N;
		image_npixels_fgmask = x_N*y_N;
		ntot_corners = (x_N+1)*(y_N+1);
	} else {
		image_npixels = 0;
		image_npixels_emask = 0;
		image_npixels_fgmask = 0;
		ntot_corners = 0;
		for (i=0; i < x_N+1; i++) {
			for (j=0; j < y_N+1; j++) {
				if ((i < x_N) and (j < y_N) and (pixel_in_mask[i][j])) image_npixels++;
				if (((i < x_N) and (j < y_N) and (pixel_in_mask[i][j])) or ((j < y_N) and (i > 0) and (pixel_in_mask[i-1][j])) or ((i < x_N) and (j > 0) and (pixel_in_mask[i][j-1])) or ((i > 0) and (j > 0) and (pixel_in_mask[i-1][j-1]))) {
					ntot_corners++;
				}
			}
		}

		for (i=0; i < x_N; i++) {
			for (j=0; j < y_N; j++) {
				if (((pixel_in_mask[i][j]) or (emask[i][j]))) image_npixels_emask++;
				if (fgmask[i][j]) image_npixels_fgmask++;
			}
		}
	}

	if (mask_pixels_i != NULL) {
		delete_ray_tracing_arrays(include_fft_arrays);
	}
	mask_pixels_i = new int[image_npixels];
	mask_pixels_j = new int[image_npixels];
	emask_pixels_i = new int[image_npixels_emask];
	emask_pixels_j = new int[image_npixels_emask];
	fgmask_pixels_i = new int[image_npixels_fgmask];
	fgmask_pixels_j = new int[image_npixels_fgmask];
	masked_pixel_corner_i = new int[ntot_corners];
	masked_pixel_corner_j = new int[ntot_corners];
	masked_pixel_corner = new int[image_npixels];
	masked_pixel_corner_up = new int[image_npixels];
	area_tri1 = new double[image_npixels];
	area_tri2 = new double[image_npixels];
	twistx = new double[image_npixels];
	twisty = new double[image_npixels];
	srcpt_x_corners = new double[ntot_corners];
	srcpt_y_corners = new double[ntot_corners];
	twiststat = new int[image_npixels];
	ncvals = new int*[x_N+1];
	for (i=0; i < x_N+1; i++) ncvals[i] = new int[y_N+1];

	n_cell=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!pixel_in_mask) or (pixel_in_mask[i][j])) {
				mask_pixels_i[n_cell] = i;
				mask_pixels_j[n_cell] = j;
				emask_pixels_i[n_cell] = i; // since emask includes the primary mask, have the emask indices identical to primary mask indices for the pixels that overlap
				emask_pixels_j[n_cell] = j;
				pixel_index[i][j] = n_cell;
				n_cell++;
			}
		}
	}

	//we DON'T reset n_cell, because now we add the emask pixels that aren't in the primary mask
	//n_cell=0;
	if ((pixel_in_mask != NULL) and (emask != NULL)) {
		for (j=0; j < y_N; j++) {
			for (i=0; i < x_N; i++) {
				if ((emask[i][j]) and (!pixel_in_mask[i][j])) {
					emask_pixels_i[n_cell] = i;
					emask_pixels_j[n_cell] = j;
					pixel_index[i][j] = n_cell;
					n_cell++;
				}
			}
		}
	}

	n_cell=0;
	if (fgmask != NULL) {
		for (j=0; j < y_N; j++) {
			for (i=0; i < x_N; i++) {
				if (fgmask[i][j]) {
					fgmask_pixels_i[n_cell] = i;
					fgmask_pixels_j[n_cell] = j;
					pixel_index_fgmask[i][j] = n_cell;
					n_cell++;
				}
			}
		}
	}

	n_corner=0;
	if ((!pixel_in_mask) or (pixel_in_mask == NULL)) {
		for (j=0; j < y_N+1; j++) {
			for (i=0; i < x_N+1; i++) {
				ncvals[i][j] = -1;
				if (((i < x_N) and (j < y_N)) or ((j < y_N) and (i > 0)) or ((i < x_N) and (j > 0)) or ((i > 0) and (j > 0))) {
					masked_pixel_corner_i[n_corner] = i;
					masked_pixel_corner_j[n_corner] = j;
					ncvals[i][j] = n_corner;
					n_corner++;
				}
			}
		}
	} else {
		for (j=0; j < y_N+1; j++) {
			for (i=0; i < x_N+1; i++) {
				ncvals[i][j] = -1;
				if (((i < x_N) and (j < y_N) and (pixel_in_mask[i][j])) or ((j < y_N) and (i > 0) and (pixel_in_mask[i-1][j])) or ((i < x_N) and (j > 0) and (pixel_in_mask[i][j-1])) or ((i > 0) and (j > 0) and (pixel_in_mask[i-1][j-1]))) {
				//if (((i < x_N) and (j < y_N) and (image_data->extended_mask[i][j])) or ((j < y_N) and (i > 0) and (image_data->extended_mask[i-1][j])) or ((i < x_N) and (j > 0) and (image_data->extended_mask[i][j-1])) or ((i > 0) and (j > 0) and (image_data->extended_mask[i-1][j-1]))) {
					masked_pixel_corner_i[n_corner] = i;
					masked_pixel_corner_j[n_corner] = j;
					if (i > (x_N+1)) die("FUCK! corner i is huge from the get-go");
					if (j > (y_N+1)) die("FUCK! corner j is huge from the get-go");
					ncvals[i][j] = n_corner;
					n_corner++;
				}
			}
		}
	}
	//cout << "corner count: " << n_corner << " " << ntot_corners << endl;
	for (int n=0; n < image_npixels; n++) {
		i = mask_pixels_i[n];
		j = mask_pixels_j[n];
		masked_pixel_corner[n] = ncvals[i][j];
		masked_pixel_corner_up[n] = ncvals[i][j+1];
	}
	for (int n=0; n < ntot_corners; n++) {
		i = masked_pixel_corner_i[n];
		j = masked_pixel_corner_j[n];
		if (i > (x_N+1)) die("FUCK! corner i is huge");
		if (j > (y_N+1)) die("FUCK! corner j is huge");
	}

	//double mask_min_r = 1e30;
	//if (image_data) {
		//for (i=0; i < x_N; i++) {
			//for (j=0; j < y_N; j++) {
				//if (image_data->in_mask[i][j]) {
					//double r = sqrt(SQR(center_pts[i][j][0]) + SQR(center_pts[i][j][1]));
					//if (r < mask_min_r) mask_min_r = r;
				//}
			//}
		//}
	//}
	//if (qlens->mpi_id==0) cout << "HACK: mask_min_r=" << mask_min_r << endl;

	set_nsplits_from_lens_settings();

	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			mapped_potpixels[i][j].clear();
			for (k=0; k < n_subpix_per_pixel; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
				n_mapped_potpixels[i][j][k] = 0;
			}
		}
	}

	image_n_subpixels_emask = 0;
	image_n_subpixels = 0;

	if (centerpts_x != NULL) die("shit13");
	centerpts_x = new double[image_npixels_emask];
	centerpts_y = new double[image_npixels_emask];
	for (int n=0; n < image_npixels_emask; n++) {
		i = emask_pixels_i[n]; 
		j = emask_pixels_j[n];
		centerpts_x[n] = center_pts[i][j][0];
		centerpts_y[n] = center_pts[i][j][1];
	}

	source_npixels = 0;
	source_npixels_inv = 0;
	lensgrid_npixels = 0;

	point_image_surface_brightness.resize(image_npixels);
	imgpixel_covinv_vector.resize(image_npixels);
	if (qlens->use_noise_map) {
		int ii,i,j;
		for (ii=0; ii < image_npixels; ii++) {
			i = mask_pixels_i[ii];
			j = mask_pixels_j[ii];
			imgpixel_covinv_vector(ii) = image_data->covinv_map[i][j];
		}
	}

	imggrid_params.setup_ray_tracing_arrays(ntot_corners,image_npixels_emask,image_npixels,image_npixels_fgmask);
#ifdef USE_STAN
	imggrid_params_dif.setup_ray_tracing_arrays(ntot_corners,image_npixels_emask,image_npixels,image_npixels_fgmask);
#endif

	//if ((qlens->split_imgpixels) and (!qlens->split_high_mag_imgpixels)) { 
	if (qlens->split_imgpixels) { 
		// if split_high_mag_imgpixels is on, this part will be deferred until after the ray-traced pixel areas have been
		// calculated (to get magnifications to use as criterion on whether to split or nit)
		setup_subpixel_ray_tracing_arrays(verbal);
	} else {
		set_null_subpixel_ray_tracing_arrays();
	}
}

void ImagePixelGrid::setup_subpixel_ray_tracing_arrays(const bool verbal)
{
	image_n_subpixels = 0;
	int nsplitpix = 0;
	//for (j=0; j < y_N; j++) {
		//for (i=0; i < x_N; i++) {
			//if ((!pixel_in_mask) or (pixel_in_mask[i][j]) or (image_data == NULL)) {
				//image_n_subpixels += SQR(nsplits[i][j]);
				//if ((verbal) and (nsplits[i][j] > 1)) nsplitpix++;
			//}
		//}
	//}

	image_n_subpixels_emask = 0;
	int nsplitpix_emask = 0;
	int i,j;
	nsplitpix = 0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!pixel_in_mask) or (pixel_in_mask[i][j]) or (image_data == NULL) or (emask == NULL) or (emask[i][j])) {
				image_n_subpixels_emask += n_subpix_per_pixel;
				if ((!pixel_in_mask) or (pixel_in_mask[i][j]) or (image_data == NULL)) {
					image_n_subpixels += n_subpix_per_pixel;
				}
			}
		}
	}
	if ((verbal) and (qlens->mpi_id==0)) {
		//cout << "Number of split image pixels in mask: " << nsplitpix << endl;
		cout << "Total number of image pixels/subpixels in mask: " << image_n_subpixels << endl;
		//cout << "Number of split image pixels (including emask): " << nsplitpix_emask << endl;
		cout << "Total number of image pixels/subpixels (including emask): " << image_n_subpixels_emask << endl;
	}

	if (extended_mask_subpixel_i != NULL) delete_subpixel_ray_tracing_arrays();

	mask_subpixel_i = new int[image_n_subpixels];
	mask_subpixel_j = new int[image_n_subpixels];
	mask_subpixel_index = new int[image_n_subpixels];

	extended_mask_subpixel_i = new int[image_n_subpixels_emask];
	extended_mask_subpixel_j = new int[image_n_subpixels_emask];
	extended_mask_subpixel_index = new int[image_n_subpixels_emask];

	emask_subpixels_ii = new int[image_n_subpixels_emask];
	emask_subpixels_jj = new int[image_n_subpixels_emask];

	image_pixel_i_from_subcell_ii = new int[x_N*imgpixel_nsplit];
	image_pixel_j_from_subcell_jj = new int[y_N*imgpixel_nsplit];

	int subpixel_idx = 0;
	int ii,jj,k;
	// We start with pixels that are in the primary mask, and then afterwards we'll proceed to the rest of the pixels in the extended mask
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
				for (k=0; k < n_subpix_per_pixel; k++) {
					mask_subpixel_i[subpixel_idx] = i;
					mask_subpixel_j[subpixel_idx] = j;
					mask_subpixel_index[subpixel_idx] = k;
					extended_mask_subpixel_i[subpixel_idx] = i;
					extended_mask_subpixel_j[subpixel_idx] = j;
					extended_mask_subpixel_index[subpixel_idx] = k;
					ii = i*imgpixel_nsplit + k / imgpixel_nsplit;
					jj = j*imgpixel_nsplit + k % imgpixel_nsplit;
					if (jj==0) image_pixel_i_from_subcell_ii[ii] = i;
					if (ii==0) image_pixel_j_from_subcell_jj[jj] = j;
					subpixel_index_ss[ii][jj] = subpixel_idx;
					emask_subpixels_ii[subpixel_idx] = ii;
					emask_subpixels_jj[subpixel_idx] = jj;
					subpixel_idx++;
				}
			} else if ((i==0) or (j==0)) {
				for (k=0; k < n_subpix_per_pixel; k++) {
					ii = i*imgpixel_nsplit + k / imgpixel_nsplit;
					jj = j*imgpixel_nsplit + k % imgpixel_nsplit;
					if (jj==0) image_pixel_i_from_subcell_ii[ii] = i;
					if (ii==0) image_pixel_j_from_subcell_jj[jj] = j;
				}
			}
		}
	}

	//we DON'T reset subpixel_idx, because now we add the emask pixels that aren't in the primary mask
	if ((pixel_in_mask != NULL) and (emask != NULL)) {
		for (j=0; j < y_N; j++) {
			for (i=0; i < x_N; i++) {
				if ((emask[i][j]) and (!pixel_in_mask[i][j])) {
					for (k=0; k < n_subpix_per_pixel; k++) {
						extended_mask_subpixel_i[subpixel_idx] = i;
						extended_mask_subpixel_j[subpixel_idx] = j;
						extended_mask_subpixel_index[subpixel_idx] = k;
						ii = i*imgpixel_nsplit + k / imgpixel_nsplit;
						jj = j*imgpixel_nsplit + k % imgpixel_nsplit;
						if (jj==0) image_pixel_i_from_subcell_ii[ii] = i;
						if (ii==0) image_pixel_j_from_subcell_jj[jj] = j;
						subpixel_index_ss[ii][jj] = subpixel_idx;
						emask_subpixels_ii[subpixel_idx] = ii;
						emask_subpixels_jj[subpixel_idx] = jj;
						subpixel_idx++;
					}
				} else if ((i==0) or (j==0)) {
					for (k=0; k < n_subpix_per_pixel; k++) {
						ii = i*imgpixel_nsplit + k / imgpixel_nsplit;
						jj = j*imgpixel_nsplit + k % imgpixel_nsplit;
						if (jj==0) image_pixel_i_from_subcell_ii[ii] = i;
						if (ii==0) image_pixel_j_from_subcell_jj[jj] = j;
					}
				}
			}
		}
	}

	imggrid_params.setup_subpixel_ray_tracing_arrays(image_n_subpixels_emask);
#ifdef USE_STAN
	imggrid_params_dif.setup_subpixel_ray_tracing_arrays(image_n_subpixels_emask);
#endif
	//defx_subpixel_centers = new double[image_n_subpixels_emask];
	//defy_subpixel_centers = new double[image_n_subpixels_emask];
}

void ImagePixelGrid::delete_ray_tracing_arrays(const bool reset_psf_arrays)
{
	if (twiststat != NULL) delete[] twiststat;
	if (area_tri1 != NULL) delete[] area_tri1;
	if (area_tri2 != NULL) delete[] area_tri2;
	if (twistx != NULL) delete[] twistx;
	if (twisty != NULL) delete[] twisty;
	if (srcpt_x_corners != NULL) delete[] srcpt_x_corners;
	if (srcpt_y_corners != NULL) delete[] srcpt_y_corners;

	if (mask_pixels_i != NULL) delete[] mask_pixels_i;
	if (mask_pixels_j != NULL) delete[] mask_pixels_j;
	if (emask_pixels_i != NULL) delete[] emask_pixels_i;
	if (emask_pixels_j != NULL) delete[] emask_pixels_j;
	if (fgmask_pixels_i != NULL) delete[] fgmask_pixels_i;
	if (fgmask_pixels_j != NULL) delete[] fgmask_pixels_j;
	if (masked_pixel_corner_i != NULL) delete[] masked_pixel_corner_i;
	if (masked_pixel_corner_j != NULL) delete[] masked_pixel_corner_j;
	if (masked_pixel_corner != NULL) delete[] masked_pixel_corner;
	if (masked_pixel_corner_up != NULL) delete[] masked_pixel_corner_up;
	if (ncvals != NULL) {
		for (int i=0; i < x_N+1; i++) delete[] ncvals[i];
		delete[] ncvals;
	}
	if (centerpts_x != NULL) delete[] centerpts_x;
	if (centerpts_y != NULL) delete[] centerpts_y;

	if (((psf_convolution_is_setup) or (fg_psf_convolution_is_setup)) and (reset_psf_arrays)) reset_psfconv_plans();
	if ((fft_convolution_is_setup) and (reset_psf_arrays)) cleanup_FFT_convolution_arrays();
	if ((fg_fft_convolution_is_setup) and (reset_psf_arrays)) cleanup_foreground_FFT_convolution_arrays();
	set_null_ray_tracing_arrays();
	if (qlens->split_imgpixels) delete_subpixel_ray_tracing_arrays();
}

void ImagePixelGrid::delete_subpixel_ray_tracing_arrays()
{
	if (extended_mask_subpixel_i != NULL) delete[] extended_mask_subpixel_i;
	if (extended_mask_subpixel_j != NULL) delete[] extended_mask_subpixel_j;
	if (extended_mask_subpixel_index != NULL) delete[] extended_mask_subpixel_index;
	if (emask_subpixels_ii != NULL) delete[] emask_subpixels_ii;
	if (emask_subpixels_jj != NULL) delete[] emask_subpixels_jj;
	//if (defx_subpixel_centers != NULL) delete[] defx_subpixel_centers;
	//if (defy_subpixel_centers != NULL) delete[] defy_subpixel_centers;
	if (mask_subpixel_i != NULL) delete[] mask_subpixel_i;
	if (mask_subpixel_j != NULL) delete[] mask_subpixel_j;
	if (mask_subpixel_index != NULL) delete[] mask_subpixel_index;

	if (image_pixel_i_from_subcell_ii != NULL) delete[] image_pixel_i_from_subcell_ii;
	if (image_pixel_j_from_subcell_jj != NULL) delete[] image_pixel_j_from_subcell_jj;

	set_null_subpixel_ray_tracing_arrays();
}

void ImagePixelGrid::update_grid_dimensions(const double xmin, const double xmax, const double ymin, const double ymax)
{
	pixel_xlength = (xmax-xmin)/x_N;
	pixel_ylength = (ymax-ymin)/y_N;
	pixel_area = pixel_xlength*pixel_ylength;
	triangle_area = 0.5*pixel_xlength*pixel_ylength;

	double x,y;
	int i,j;
	for (j=0; j <= y_N; j++) {
		y = ymin + j*pixel_ylength;
		for (i=0; i <= x_N; i++) {
			x = xmin + i*pixel_xlength;
			if ((i < x_N) and (j < y_N)) {
				center_pts[i][j][0] = x + 0.5*pixel_xlength;
				center_pts[i][j][1] = y + 0.5*pixel_ylength;
			}
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
		}
	}
		for (j=0; j < y_N; j++) {
			y = ymin + j*pixel_ylength;
			for (i=0; i < x_N; i++) {
				surface_brightness[i][j] = 0; // since any previous surface brightness was based on different dimensions
			}
		}
	if (qlens) {
		set_nsplits_from_lens_settings();
	}
}

template <typename QScalar>
bool ImagePixelGrid::test_if_between(const QScalar& p, const QScalar& a, const QScalar& b)
{
	if ((b>a) and (p>a) and (p<b)) return true;
	else if ((a>b) and (p>b) and (p<a)) return true;
	return false;
}
template bool ImagePixelGrid::test_if_between<double>(const double& p, const double& a, const double& b);
#ifdef USE_STAN
template bool ImagePixelGrid::test_if_between<stan::math::var>(const stan::math::var& p, const stan::math::var& a, const stan::math::var& b);
#endif

template <typename MathTypes>
void ImagePixelGrid::calculate_sourcepts_and_areas(const bool raytrace_pixel_centers, const bool verbal)
{
	using QScalar = typename MathTypes::QScalar;
	using VecType = typename MathTypes::VecType;
	ImgGrid_Params<MathTypes>& p = assign_imggrid_param_object<MathTypes>();
	int i,j,k,n,n_cell,n_corner,n_yp;

	//ImgGrid_Params<Eigen::VectorXd,Eigen::MatrixXd,double>& pd = assign_imggrid_param_object<Eigen::VectorXd,Eigen::MatrixXd,double>();
	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif
		lensvector<double> d1,d2,d3,d4;
		lensvector<double> offset_pt;
		//int ii,jj;
		//#pragma omp for private(n,i,j) schedule(dynamic)
		for (n=0; n < ntot_corners; n++) {
			//j = n / (x_N+1);
			//i = n % (x_N+1);
			j = masked_pixel_corner_j[n];
			i = masked_pixel_corner_i[n];
			if (psf==NULL) {
				offset_pt = corner_pts[i][j];
			} else {
				offset_pt[0] = corner_pts[i][j][0] - psf->psf_params.psf_offset_x;
				offset_pt[1] = corner_pts[i][j][1] - psf->psf_params.psf_offset_y;
			}
			//cout << i << " " << j << " " << n << " " << ntot_corners << " " << mpi_end << endl;
			qlens->find_sourcept_from_data<double>(offset_pt,srcpt_x_corners[n],srcpt_y_corners[n],thread,imggrid_zfactors,imggrid_betafactors);
		}
		//#pragma omp for private(n_cell,i,j,n,n_yp) schedule(dynamic)
		for (n_cell=0; n_cell < image_npixels; n_cell++) {
			j = mask_pixels_j[n_cell];
			i = mask_pixels_i[n_cell];

			//n = j*(x_N+1)+i;
			//n_yp = (j+1)*(x_N+1)+i;
			n = masked_pixel_corner[n_cell];
			n_yp = masked_pixel_corner_up[n_cell];
			d1[0] = srcpt_x_corners[n] - srcpt_x_corners[n+1];
			d1[1] = srcpt_y_corners[n] - srcpt_y_corners[n+1];
			d2[0] = srcpt_x_corners[n_yp] - srcpt_x_corners[n];
			d2[1] = srcpt_y_corners[n_yp] - srcpt_y_corners[n];
			d3[0] = srcpt_x_corners[n_yp+1] - srcpt_x_corners[n_yp];
			d3[1] = srcpt_y_corners[n_yp+1] - srcpt_y_corners[n_yp];
			d4[0] = srcpt_x_corners[n+1] - srcpt_x_corners[n_yp+1];
			d4[1] = srcpt_y_corners[n+1] - srcpt_y_corners[n_yp+1];

			twiststat[n_cell] = 0;
			double xa,ya,xb,yb,xc,yc,xd,yd,slope1,slope2;
			xa=srcpt_x_corners[n];
			ya=srcpt_y_corners[n];
			xb=srcpt_x_corners[n_yp];
			yb=srcpt_y_corners[n_yp];
			xc=srcpt_x_corners[n_yp+1];
			yc=srcpt_y_corners[n_yp+1];
			xd=srcpt_x_corners[n+1];
			yd=srcpt_y_corners[n+1];
			slope1 = (yb-ya)/(xb-xa);
			slope2 = (yc-yd)/(xc-xd);
			twistx[n_cell] = (yd-ya+xa*slope1-xd*slope2)/(slope1-slope2);
			twisty[n_cell] = (twistx[n_cell]-xa)*slope1+ya;
			if ((test_if_between(twistx[n_cell],xa,xb)) and (test_if_between(twisty[n_cell],ya,yb)) and (test_if_between(twistx[n_cell],xc,xd)) and (test_if_between(twisty[n_cell],yc,yd))) {
				twiststat[n_cell] = 1;
				d2[0] = twistx[n_cell] - srcpt_x_corners[n];
				d2[1] = twisty[n_cell] - srcpt_y_corners[n];
				d4[0] = twistx[n_cell] - srcpt_x_corners[n_yp+1];
				d4[1] = twisty[n_cell] - srcpt_y_corners[n_yp+1];
			} else {
				slope1 = (yd-ya)/(xd-xa);
				slope2 = (yc-yb)/(xc-xb);
				twistx[n_cell] = (yb-ya+xa*slope1-xb*slope2)/(slope1-slope2);
				twisty[n_cell] = (twistx[n_cell]-xa)*slope1+ya;
				if ((test_if_between(twistx[n_cell],xa,xd)) and (test_if_between(twisty[n_cell],ya,yd)) and (test_if_between(twistx[n_cell],xb,xc)) and (test_if_between(twisty[n_cell],yb,yc))) {
					twiststat[n_cell] = 2;
					d1[0] = srcpt_x_corners[n] - twistx[n_cell];
					d1[1] = srcpt_y_corners[n] - twisty[n_cell];
					d3[0] = srcpt_x_corners[n_yp+1] - twistx[n_cell];
					d3[1] = srcpt_y_corners[n_yp+1] - twisty[n_cell];
				}
			}

			area_tri1[n_cell] = 0.5*abs(d1 ^ d2);
			area_tri2[n_cell] = 0.5*abs(d3 ^ d4);
		}

		Eigen::VectorXd offset_pts_x(image_npixels_emask);
		Eigen::VectorXd offset_pts_y(image_npixels_emask);
		if ((!qlens->split_imgpixels) or (raytrace_pixel_centers)) {
			//#pragma omp for private(n_cell,i,j,n,n_yp) schedule(dynamic)
			for (n_cell=0; n_cell < image_npixels_emask; n_cell++) {
				j = emask_pixels_j[n_cell];
				i = emask_pixels_i[n_cell];
				if (psf==NULL) {
					offset_pts_x(n_cell) = center_pts[i][j][0];
					offset_pts_y(n_cell) = center_pts[i][j][1];
				} else {
					offset_pts_x(n_cell) = center_pts[i][j][0] - psf->psf_params.psf_offset_x;
					offset_pts_y(n_cell) = center_pts[i][j][1] - psf->psf_params.psf_offset_y;
				}
				offset_pt[0] = offset_pts_x(n_cell);
				offset_pt[1] = offset_pts_y(n_cell);

				//if constexpr (std::is_same_v<QScalar, double>) {
				//qlens->find_sourcept_from_data<QScalar>(offset_pt,p.srcpt_x_centers(n_cell),p.srcpt_y_centers(n_cell),thread,imggrid_zfactors,imggrid_betafactors);
				//}
			}
			qlens->find_sourcepts_from_data_vec<VecType,QScalar>(offset_pts_x,offset_pts_y,p.srcpt_x_centers,p.srcpt_y_centers,thread,imggrid_zfactors,imggrid_betafactors);

			/*
			for (n_cell=0; n_cell < image_npixels_emask; n_cell++) {
				QScalar srcpt_x_check;
				QScalar srcpt_y_check;
				offset_pt[0] = offset_pts_x(n_cell);
				offset_pt[1] = offset_pts_y(n_cell);
				if constexpr (std::is_same_v<QScalar, double>) {
					qlens->find_sourcept_from_data<QScalar>(offset_pt,srcpt_x_check,srcpt_y_check,thread,imggrid_zfactors,imggrid_betafactors);
					if ((abs(srcpt_x_check-p.srcpt_x_centers(n_cell)) > 1e-6) or (abs(srcpt_y_check-p.srcpt_y_centers(n_cell)) > 1e-6)) {
						cout << "SRCPTS! " << srcpt_x_check << " " << p.srcpt_x_centers(n_cell) << " " << srcpt_y_check << " " << p.srcpt_y_centers(n_cell) << endl;
					}
				}
			}
			*/
		}
	}

	src_xmin = 1e30; src_xmax = -1e30;
	src_ymin = 1e30; src_ymax = -1e30;
	for (n=0; n < ntot_corners; n++) {
		//j = n / (x_N+1);
		//i = n % (x_N+1);
		j = masked_pixel_corner_j[n];
		i = masked_pixel_corner_i[n];
		corner_sourcepts[i][j][0] = srcpt_x_corners[n];
		corner_sourcepts[i][j][1] = srcpt_y_corners[n];
		if (srcpt_x_corners[n] < src_xmin) src_xmin = srcpt_x_corners[n];
		if (srcpt_x_corners[n] > src_xmax) src_xmax = srcpt_x_corners[n];
		if (srcpt_y_corners[n] < src_ymin) src_ymin = srcpt_y_corners[n];
		if (srcpt_y_corners[n] > src_ymax) src_ymax = srcpt_y_corners[n];

		//wtf << corner_pts[i][j][0] << " " << corner_pts[i][j][1] << " " << corner_sourcepts[i][j][0] << " " << corner_sourcepts[i][j][1] << " " << endl;
	}
	//wtf.close();
	int ii,jj;
	double u0,w0,mag;
	int subcell_index;
	min_srcplane_area = 1e30;
	double pixel_srcplane_area;
	for (n=0; n < image_npixels; n++) {
		//n_cell = j*x_N+i;
		j = mask_pixels_j[n];
		i = mask_pixels_i[n];
		source_plane_triangle1_area[i][j] = area_tri1[n];
		source_plane_triangle2_area[i][j] = area_tri2[n];
		pixel_srcplane_area = area_tri1[n] + area_tri2[n];
		pixel_mag[i][j] = pixel_area / pixel_srcplane_area;
		if (pixel_srcplane_area < min_srcplane_area) min_srcplane_area = pixel_srcplane_area;
		//if (i==176) cout << "AREAS (" << i << "," << j << "): " << area_tri1[n] << " " << area_tri2[n] << endl;
		twist_pts[i][j][0] = twistx[n];
		twist_pts[i][j][1] = twisty[n];
		twist_status[i][j] = twiststat[n];
		if (qlens->split_high_mag_imgpixels) {
			mag = pixel_mag[i][j];
			//cout << "TRYING " << mag << " " << image_data->surface_brightness[i][j] << endl;
			//if ((pixel_in_mask[i][j]) and ((mag > qlens->imgpixel_himag_threshold) or (mag < qlens->imgpixel_lomag_threshold)) and ((image_data == NULL) or (image_data->surface_brightness[i][j] > qlens->imgpixel_sb_threshold))) {
			if (pixel_in_mask[i][j]) {
				//nsplits[i][j] = qlens->default_imgpixel_nsplit;
				subcell_index = 0;
				for (ii=0; ii < imgpixel_nsplit; ii++) {
					for (jj=0; jj < nsplits[i][j]; jj++) {
						u0 = ((double) (1+2*ii))/(2*imgpixel_nsplit);
						w0 = ((double) (1+2*jj))/(2*imgpixel_nsplit);
						subpixel_center_pts[i][j][subcell_index][0] = (1-u0)*corner_pts[i][j][0] + u0*corner_pts[i+1][j][0];
						subpixel_center_pts[i][j][subcell_index][1] = (1-w0)*corner_pts[i][j][1] + w0*corner_pts[i][j+1][1];
						//subpixel_center_pts[i][j][subcell_index][0] = u0*corner_pts[i][j][0] + (1-u0)*corner_pts[i+1][j][0];
						//subpixel_center_pts[i][j][subcell_index][1] = w0*corner_pts[i][j][1] + (1-w0)*corner_pts[i][j+1][1];
						subcell_index++;
					}
				}
				//cout << "Setting imgpixel_nsplit=" << qlens->default_imgpixel_nsplit << " for pixel " << i << " " << j << " (sb=" << image_data->surface_brightness[i][j] << ")" << endl;
			//} else {
				//cout << "NOPE, mag=" << mag << " sb=" << image_data->surface_brightness[i][j] << " imgpixel_nsplit=" << imgpixel_nsplit << endl; 
			}
		}
	}
	//if ((qlens->split_imgpixels) and (qlens->split_high_mag_imgpixels)) setup_subpixel_ray_tracing_arrays(verbal);

	if (qlens->split_imgpixels) {
		int n_subcell;
		//#pragma omp parallel
		{
			int thread;
//#ifdef USE_OPENMP
			//thread = omp_get_thread_num();
//#else
			thread = 0;
//#endif
			//lensvector<double> offset_pt;

			Eigen::VectorXd offset_pts_x(image_n_subpixels_emask);
			Eigen::VectorXd offset_pts_y(image_n_subpixels_emask);

			//#pragma omp for private(i,j,k,n_subcell) schedule(dynamic)
			for (n_subcell=0; n_subcell < image_n_subpixels_emask; n_subcell++) {
				j = extended_mask_subpixel_j[n_subcell];
				i = extended_mask_subpixel_i[n_subcell];
				k = extended_mask_subpixel_index[n_subcell];
				if (psf==NULL) {
					offset_pts_x(n_subcell) = subpixel_center_pts[i][j][k][0];
					offset_pts_y(n_subcell) = subpixel_center_pts[i][j][k][1];
				} else {
					offset_pts_x(n_subcell) = subpixel_center_pts[i][j][k][0] - psf->psf_params.psf_offset_x;
					offset_pts_y(n_subcell) = subpixel_center_pts[i][j][k][1] - psf->psf_params.psf_offset_y;
				}
				//offset_pt[0] = offset_pts_x(n_subcell);
				//offset_pt[1] = offset_pts_y(n_subcell);
				//if constexpr (std::is_same_v<QScalar, double>) {
				//qlens->find_sourcept_from_data<QScalar>(offset_pt,p.srcpt_x_subpixel_centers(n_subcell),p.srcpt_y_subpixel_centers(n_subcell),thread,imggrid_zfactors,imggrid_betafactors);
				//}
				//if (p.srcpt_x_subpixel_centers[n_subcell]*0.0 != 0.0) die("nonsense value for deflection (x=%g y=%g srcpt_x=%g srcpt_y=%g)",subpixel_center_pts[i][j][k][0],subpixel_center_pts[i][j][k][1],p.srcpt_x_subpixel_centers[n_subcell],p.srcpt_y_subpixel_centers[n_subcell]);
			}
			qlens->find_sourcepts_from_data_vec<VecType,QScalar>(offset_pts_x,offset_pts_y,p.srcpt_x_subpixel_centers,p.srcpt_y_subpixel_centers,thread,imggrid_zfactors,imggrid_betafactors);
		}
		//if constexpr (std::is_same_v<QScalar, stan::math::var>) {
			//cout << "Finding sourcepts with autodif" << endl;
		//} else {
			//cout << "Finding sourcepts WITHOUT autodif" << endl;
		//}

	}
	if ((!qlens->split_imgpixels) or (raytrace_pixel_centers)) {
		for (n=0; n < image_npixels_emask; n++) {
			//n_cell = j*x_N+i;
			j = emask_pixels_j[n];
			i = emask_pixels_i[n];
			center_sourcepts[i][j][0] = value_of(p.srcpt_x_centers(n));
			center_sourcepts[i][j][1] = value_of(p.srcpt_y_centers(n));
		}
	}
	if (qlens->split_imgpixels) {
		for (n=0; n < image_n_subpixels_emask; n++) {
			j = extended_mask_subpixel_j[n];
			i = extended_mask_subpixel_i[n];
			k = extended_mask_subpixel_index[n];
			subpixel_center_sourcepts[i][j][k][0] = value_of(p.srcpt_x_subpixel_centers(n));
			subpixel_center_sourcepts[i][j][k][1] = value_of(p.srcpt_y_subpixel_centers(n));
		}
	}
}
template void ImagePixelGrid::calculate_sourcepts_and_areas<PlainTypes>(const bool raytrace_pixel_centers, const bool verbal);
#ifdef USE_STAN
template void ImagePixelGrid::calculate_sourcepts_and_areas<VarmatTypes>(const bool raytrace_pixel_centers, const bool verbal);
#endif

bool ImagePixelGrid::calculate_subpixel_source_gradient()
{
	//cout << "GOT HERE?" << endl;
	if (delaunay_srcgrid==NULL) return false;
	//cout << "GOT HERE2?" << endl;
	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif

		int n,i,j,k;
		if (image_n_subpixels_emask != 0) {
			#pragma omp for private(n,i,j,k) schedule(static)
			for (n=0; n < image_n_subpixels; n++) {
				j = mask_subpixel_j[n];
				i = mask_subpixel_i[n];
				k = mask_subpixel_index[n];
				delaunay_srcgrid->find_source_gradient(subpixel_center_sourcepts[i][j][k],subpixel_source_gradient[i][j][k],thread);
			}
		}
	}
	int n_zeros=0, n_not_zeros=0;
	//#pragma omp for private(n,i,j) schedule(static)
	if (image_n_subpixels_emask == 0) {
		int i,j,n;
		//lensvector<double> new_sgrad;
		//double grad_difsq=0;
		for (n=0; n < image_npixels; n++) {
			j = mask_pixels_j[n];
			i = mask_pixels_i[n];
			//delaunay_srcgrid->find_source_gradient(center_sourcepts[i][j],new_sgrad,0);
			//grad_difsq += SQR(new_sgrad[0]-subpixel_source_gradient[i][j][0][0]) + SQR(new_sgrad[1]-subpixel_source_gradient[i][j][0][1]);
			//subpixel_source_gradient[i][j][0] = new_sgrad;
			delaunay_srcgrid->find_source_gradient(center_sourcepts[i][j],subpixel_source_gradient[i][j][0],0);
			//S0_check[i][j] = delaunay_srcgrid->interpolate_surface_brightness(imggrid_params.center_sourcepts[i][j],false,0);
			//x0_check[i][j] = center_sourcepts[i][j];
			//cout << "S0_check original: " << imggrid_params.center_sourcepts[i][j][0] << " " << imggrid_params.center_sourcepts[i][j][1] << " " << S0_check[i][j] << endl;
			//if (S0_check[i][j] != 0.0) {
				//n_not_zeros++;
			//} else {
				//n_zeros++;
			//}
			//cout << "SUBPIXEL SOURCE GRADIENT: " << subpixel_source_gradient[i][j][0][0] << " " << subpixel_source_gradient[i][j][0][1] << endl;
		}
		//grad_difsq = sqrt(grad_difsq/image_npixels);
		//if (qlens->mpi_id==0) cout << "DIFF in sbgrad: " << grad_difsq << endl;
	}
	//cout << "Pixels with zero SB: " << n_zeros << endl;
	//cout << "Pixels with nonzero SB: " << n_not_zeros << endl;
	return true;
}

template <typename MathTypes>
void ImagePixelGrid::redo_lensing_calculations(const bool verbal)
{
	std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	calculate_sourcepts_and_areas<MathTypes>(true,verbal);

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for ray-tracing image pixel grid: "  << wtime.count() << endl;
	}
}
template void ImagePixelGrid::redo_lensing_calculations<PlainTypes>(const bool verbal);
#ifdef USE_STAN
template void ImagePixelGrid::redo_lensing_calculations<VarmatTypes>(const bool verbal);
#endif

void ImagePixelGrid::load_data(ImageData& pixel_data)
{
	max_sb = -1e30;
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			surface_brightness[i][j] = pixel_data.surface_brightness[i][j];
			if (surface_brightness[i][j] > max_sb) max_sb=surface_brightness[i][j];
		}
	}
}

double ImagePixelGrid::output_surface_brightness(Vector<double>& xvals, Vector<double>& yvals, Vector<double>& zvals, bool plot_residual, bool normalize_sb, bool show_noise_thresh, bool plot_log, bool show_foreground_mask)
{
	xvals.input(x_N+1);
	yvals.input(y_N+1);
	for (int i=0; i <= x_N; i++) {
		xvals[i] = corner_pts[i][0][0];
	}
	for (int j=0; j <= y_N; j++) {
		yvals[j] = corner_pts[0][j][1];
	}	
	int i,j,k;
	double residual, tot_residuals = 0;

	//ofstream wtfout("wtf2.dat");
	zvals.input(x_N*y_N);
	k=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (((show_foreground_mask) and (image_data->foreground_mask_data[i][j])) or ((!show_foreground_mask) and ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])))) {
				if (!plot_residual) {
					double sb = surface_brightness[i][j] + foreground_surface_brightness[i][j];
					if (normalize_sb) {
						if (qlens->use_noise_map) {
							if (image_data != NULL) {
								sb /= image_data->noise_map[i][j];
							} else warn("image pixel data not loaded; could not use noise map to normalize plot");
						} else {
							if (qlens->background_pixel_noise > 0) residual /= qlens->background_pixel_noise;
						}
					}
					//if (sb*0.0 != 0.0) die("WTF %g %g",surface_brightness[i][j],foreground_surface_brightness[i][j]);
					if (!plot_log) zvals[k] = sb;
					else zvals[k] = log(abs(sb));
				} else {
					double sb = surface_brightness[i][j] + foreground_surface_brightness[i][j];
					residual = image_data->surface_brightness[i][j] - sb;
					if (normalize_sb) {
						if (qlens->use_noise_map) {
							if (image_data != NULL) {
								residual /= image_data->noise_map[i][j];
							} else warn("image pixel data not loaded; could not use noise map to plot residuals");
						} else {
							if (qlens->background_pixel_noise > 0) residual /= qlens->background_pixel_noise;
						}
					}
					tot_residuals += residual*residual;
					//wtfout << i << " " << j << " " << (residual*residual) << endl;
					if (show_noise_thresh) {
						if (abs(residual) >= qlens->background_pixel_noise) zvals[k] = residual;
						else zvals[k] = numeric_limits<double>::quiet_NaN();
					}
					else zvals[k] = residual;
					//qlens->find_sourcept(center_pts[i][j],center_sourcepts[i][j],0,imggrid_zfactors,imggrid_betafactors);
				}
				//pixel_src_file << center_pts[i][j][0] << " " << center_pts[i][j][1] << " " << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " " << residual << endl;
			} else {
				zvals[k] = numeric_limits<double>::quiet_NaN();
			}
			k++;
		}
	}
	//plot_sourcepts(outfile_root);
	return tot_residuals;
}

void ImagePixelGrid::plot_sourcepts(string outfile_root, const bool show_subpixels)
{
	string sp_filename = outfile_root + "_srcpts.dat";

	ofstream sourcepts_file; qlens->open_output_file(sourcepts_file,sp_filename);
	sourcepts_file << setiosflags(ios::scientific);
	int i,j;
	double residual;

	int k,nsp;
	//cout << "PLOOTTING SOURCE POINTS!" << endl;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
				//if ((qlens->include_fgmask_in_inversion) and (mask[i][j]==false)) {
					//cout << "CONTINUING" << endl;
					//continue; // if including foreground mask in inversion, we still only use the lensing mask to define delaunay source pixels
				//}
				//else if ((qlens->include_fgmask_in_inversion) and (mask[i][j]==true)) {
					//cout << "WTF?" << endl;
				//}
				//else if (!qlens->include_fgmask_in_inversion) cout << "HARG?" << endl;
			//if ((pixel_in_mask==NULL) or ((!qlens->zero_sb_fgmask_prior) and (emask) and (emask[i][j])) or ((qlens->zero_sb_fgmask_prior) and (mask) and (mask[i][j]))) {
				if ((!qlens->split_imgpixels) or (!show_subpixels)) {
					sourcepts_file << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " " << center_pts[i][j][0] << " " << center_pts[i][j][1] << endl;
				} else {
					for (k=0; k < n_subpix_per_pixel; k++) {
						sourcepts_file << subpixel_center_sourcepts[i][j][k][0] << " " << subpixel_center_sourcepts[i][j][k][1] << " " << subpixel_center_pts[i][j][k][0] << " " << subpixel_center_pts[i][j][k][1] << endl;
					}
				}
			}
		}
	}
}

/*
void ImagePixelGrid::output_fits_file(string fits_filename, bool plot_residual)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return;
#else
	int i,j,kk;
	fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix = -64, naxis = 2;
	long naxes[2] = {x_N,y_N};
	double *pixels;
	if (qlens->fit_output_dir != ".") qlens->create_output_directory(); // in case it hasn't been created already
	string filename = "!" + qlens->fit_output_dir + "/" + fits_filename; // ensures that it overwrites an existing file of the same name

	if (!fits_create_file(&outfptr, filename.c_str(), &status))
	{
		if (!fits_create_img(outfptr, bitpix, naxis, naxes, &status))
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long fpixel[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[x_N];

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=0; i < x_N; i++) {
						if (!plot_residual) pixels[i] = surface_brightness[i][j] + foreground_surface_brightness[i][j];
						else pixels[i] = image_data->surface_brightness[i][j] - surface_brightness[i][j] - foreground_surface_brightness[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] pixels;
			}
			if (pixel_xlength==pixel_ylength) {
				fits_write_key(outfptr, TDOUBLE, "PXSIZE", &pixel_xlength, "length of square pixels (in arcsec)", &status);
			} else {
				if (qlens->mpi_id==0) cout << "NOTE: pixel length not equal in x- versus y-directions; not saving pixel size in FITS file header" << endl;
			}
			if ((qlens->simulate_pixel_noise) and (!qlens->use_noise_map))
				fits_write_key(outfptr, TDOUBLE, "PXNOISE", &qlens->background_pixel_noise, "pixel surface brightness noise", &status);
			if ((psf->psf_width_x != 0) and (psf->psf_width_y==psf->psf_width_x) and (!psf->use_input_psf_matrix))
				fits_write_key(outfptr, TDOUBLE, "PSFSIG", &psf->psf_width_x, "Gaussian PSF width (dispersion, not FWHM)", &status);
			fits_write_key(outfptr, TDOUBLE, "ZSRC", &qlens->source_redshift, "redshift of source galaxy", &status);
			if (qlens->nlens > 0) {
				double zl = qlens->lens_list[qlens->primary_lens_number]->get_redshift();
				fits_write_key(outfptr, TDOUBLE, "ZLENS", &zl, "redshift of primary qlens", &status);
			}

			if (qlens->data_info != "") {
				string comment = "ql: " + qlens->data_info;
				fits_write_comment(outfptr, comment.c_str(), &status);
			}
			if (qlens->param_markers != "") {
				string param_markers_comma = qlens->param_markers;
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
*/

void ImagePixelGrid::assign_mask_pointers(ImageData& pixel_data, const int mask_index)
{
	mask = pixel_data.in_mask[mask_index];
	emask = pixel_data.extended_mask[mask_index];
	fgmask = pixel_data.foreground_mask;
}

bool ImagePixelGrid::set_fit_window(ImageData& pixel_data, const bool raytrace, const int mask_index, const bool redo_fft, const bool use_fgmask)
{
	if ((x_N != pixel_data.npixels_x) or (y_N != pixel_data.npixels_y)) {
		warn("Number of data pixels does not match specified number of image pixels; cannot activate fit window");
		return false;
	}
	//cout << "RUNNING SET FIT WINDOW " << endl;
	int i,j,k;
	if (pixel_in_mask==NULL) {
		pixel_in_mask = new bool*[x_N];
		for (i=0; i < x_N; i++) pixel_in_mask[i] = new bool[y_N];
	}
	mask = pixel_data.in_mask[mask_index];
	emask = pixel_data.extended_mask[mask_index];
	fgmask = pixel_data.foreground_mask;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (!use_fgmask) pixel_in_mask[i][j] = pixel_data.in_mask[mask_index][i][j];
			else pixel_in_mask[i][j] = pixel_data.foreground_mask[i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			mapped_potpixels[i][j].clear();
			//pixel_index[i][j] = 0;
			for (k=0; k < n_subpix_per_pixel; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
				n_mapped_potpixels[i][j][k] = 0;
			}
		}
	}
	//double mask_min_r = 1e30;
	//for (i=0; i < x_N; i++) {
		//for (j=0; j < y_N; j++) {
			//if (pixel_data.in_mask[i][j]) {
				//double r = sqrt(SQR(center_pts[i][j][0]) + SQR(center_pts[i][j][1]));
				//if (r < mask_min_r) mask_min_r = r;
			//}
		//}
	//}
	//if ((qlens) and (qlens->mpi_id==0)) cout << "HACK: mask_min_r=" << mask_min_r << endl;

	if (qlens) {
		setup_ray_tracing_arrays(redo_fft);
		if ((raytrace) or (qlens->split_high_mag_imgpixels)) calculate_sourcepts_and_areas<PlainTypes>(true);
	}
	return true;
}

void ImagePixelGrid::include_all_pixels(const bool redo_fft)
{
	int i,j,k;
	if (pixel_in_mask==NULL) {
		pixel_in_mask = new bool*[x_N];
		for (i=0; i < x_N; i++) pixel_in_mask[i] = new bool[y_N];
	}
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			pixel_in_mask[i][j] = true;
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			mapped_potpixels[i][j].clear();
			for (k=0; k < n_subpix_per_pixel; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
				n_mapped_potpixels[i][j][k] = 0;
			}
		}
	}
	if (qlens) setup_ray_tracing_arrays(redo_fft);

	if ((qlens->nlens > 0) and (imggrid_zfactors != NULL)) calculate_sourcepts_and_areas<PlainTypes>(true);
}

void ImagePixelGrid::activate_extended_mask(const bool redo_fft)
{
	if (emask==NULL) { warn("emask pointer set to NULL; could not activate extended mask"); return; }
	int i,j,k;
	if (pixel_in_mask==NULL) {
		pixel_in_mask = new bool*[x_N];
		for (i=0; i < x_N; i++) pixel_in_mask[i] = new bool[y_N];
	}
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			pixel_in_mask[i][j] = emask[i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			mapped_potpixels[i][j].clear();
			for (k=0; k < n_subpix_per_pixel; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
				n_mapped_potpixels[i][j][k] = 0;
			}
		}
	}
	if (qlens) setup_ray_tracing_arrays(redo_fft);
}

void ImagePixelGrid::activate_foreground_mask(const bool redo_fft, const bool datamask)
{
	int i,j,k;
	if (image_data==NULL) { warn("image pixel data set to NULL; could not activate foreground mask"); return; }
	if (pixel_in_mask==NULL) {
		pixel_in_mask = new bool*[x_N];
		for (i=0; i < x_N; i++) pixel_in_mask[i] = new bool[y_N];
	}
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if (datamask) pixel_in_mask[i][j] = image_data->foreground_mask_data[i][j];
			else pixel_in_mask[i][j] = image_data->foreground_mask[i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			mapped_potpixels[i][j].clear();
			for (k=0; k < n_subpix_per_pixel; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
				n_mapped_potpixels[i][j][k] = 0;
			}
		}
	}
	if (qlens) setup_ray_tracing_arrays(redo_fft);
}

/*
void ImagePixelGrid::deactivate_extended_mask(const bool redo_fft, const bool use_fgmask)
{
	if (mask==NULL) { warn("mask pointer set to NULL; could not activate extended mask"); return; }
	int i,j,k;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if (use_fgmask) pixel_in_mask[i][j] = fgmask[i][j];
			else pixel_in_mask[i][j] = mask[i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			mapped_potpixels[i][j].clear();
			for (k=0; k < nsubpix; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
				n_mapped_potpixels[i][j][k] = 0;
			}
		}
	}
	if (qlens) setup_ray_tracing_arrays(redo_fft);
}
*/

void ImagePixelGrid::update_mask_values(const bool use_fgmask)
{
	if (mask==NULL) { warn("mask pointer set to NULL; could not update mask values within imggrid"); return; }
	int i,j,k;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if (use_fgmask) pixel_in_mask[i][j] = fgmask[i][j];
			else pixel_in_mask[i][j] = mask[i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			mapped_potpixels[i][j].clear();
			for (k=0; k < n_subpix_per_pixel; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
				n_mapped_potpixels[i][j][k] = 0;
			}
		}
	}
	if (qlens) setup_ray_tracing_arrays();
}

void ImagePixelGrid::set_nsplits_from_lens_settings()
{
	imgpixel_nsplit = (qlens->split_high_mag_imgpixels) ? 1 : qlens->default_imgpixel_nsplit;
	n_subpix_per_pixel = imgpixel_nsplit*imgpixel_nsplit;
	set_nsplits(qlens->split_imgpixels);
}

void ImagePixelGrid::set_nsplits(const bool split_pixels)
{
	int i,j,ii,jj,subcell_index;
	double u0,w0;

	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if (split_pixels) {
				subcell_index = 0;
				for (ii=0; ii < imgpixel_nsplit; ii++) {
					for (jj=0; jj < imgpixel_nsplit; jj++) {
						u0 = ((double) (1+2*ii))/(2*imgpixel_nsplit);
						w0 = ((double) (1+2*jj))/(2*imgpixel_nsplit);
						subpixel_center_pts[i][j][subcell_index][0] = (1-u0)*corner_pts[i][j][0] + u0*corner_pts[i+1][j][0];
						subpixel_center_pts[i][j][subcell_index][1] = (1-w0)*corner_pts[i][j][1] + w0*corner_pts[i][j+1][1];
						subcell_index++;
					}
				}
			}
		}
	}
}

bool ImageData::test_if_in_fit_region(const double& x, const double& y, const int mask_k)
{
	// it would be faster to just use division to figure out which pixel it's in, but this is good enough
	int i,j;
	for (j=0; j <= npixels_y; j++) {
		if ((yvals[j] <= y) and (yvals[j+1] > y)) {
			for (i=0; i <= npixels_x; i++) {
				if ((xvals[i] <= x) and (xvals[i+1] > x)) {
					if (in_mask[mask_k][i][j] == true) return true;
					else break;
				}
			}
		}
	}
	return false;
}

double ImagePixelGrid::calculate_signal_to_noise(double& total_signal)
{
	// NOTE: Ideally, this function should be called *before* adding noise to the image.
	double sbmax=-1e30;
	static const double signal_threshold_frac = 1e-2;
	int i,j,npixels_above_threshold=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (pixel_in_mask[i][j]) {
				if ((foreground_surface_brightness[i][j] + surface_brightness[i][j]) > sbmax) {
					sbmax = foreground_surface_brightness[i][j] + surface_brightness[i][j];
					npixels_above_threshold++;
				}
			}
		}
	}
	double signal_mean=0,sn_mean=0;
	int npixels=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (pixel_in_mask[i][j]) {
				if ((foreground_surface_brightness[i][j] + surface_brightness[i][j]) > signal_threshold_frac*sbmax) {
					if (noise_map[i][j]==0.0) {
						warn("pixel noise is zero; cannot calculate signal-to-noise");
					}
					signal_mean += foreground_surface_brightness[i][j] + surface_brightness[i][j];
					sn_mean += (foreground_surface_brightness[i][j] + surface_brightness[i][j])/noise_map[i][j];
					npixels++;
				}
			}
		}
	}
	total_signal = signal_mean * pixel_xlength * pixel_ylength;
	if (npixels > 0) {
		sn_mean /= npixels_above_threshold;
		signal_mean /= npixels_above_threshold;
	}
	return sn_mean;
}

void ImagePixelGrid::add_pixel_noise()
{
	if (surface_brightness == NULL) die("surface brightness pixel map has not been loaded");
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			surface_brightness[i][j] += noise_map[i][j]*qlens->NormalDeviate();
		}
	}
}

void ImagePixelGrid::find_optimal_sourcegrid(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double &sourcegrid_limit_xmin, const double &sourcegrid_limit_xmax, const double &sourcegrid_limit_ymin, const double& sourcegrid_limit_ymax)
{
	if (surface_brightness == NULL) die("surface brightness pixel map has not been loaded");
	if (image_data == NULL) die("image pixel data must be loaded to find optimal source grid scale");
	bool use_noise_threshold = true;
	if (qlens->noise_threshold <= 0) use_noise_threshold = false;
	int i,j,k;
	sourcegrid_xmin=1e30;
	sourcegrid_xmax=-1e30;
	sourcegrid_ymin=1e30;
	sourcegrid_ymax=-1e30;
	int ii,jj,il,ih,jl,jh,nn;
	double sbavg;
	double xsavg, ysavg;
	static const int window_size_for_sbavg = 0;
	bool resize_grid;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if ((!mask) or (mask[i][j])) {
				resize_grid = true;
				if (use_noise_threshold) {
					sbavg=0;
					nn=0;
					il = i - window_size_for_sbavg;
					ih = i + window_size_for_sbavg;
					jl = j - window_size_for_sbavg;
					jh = j + window_size_for_sbavg;
					if (il<0) il=0;
					if (ih>x_N-1) ih=x_N-1;
					if (jl<0) jl=0;
					if (jh>y_N-1) jh=y_N-1;
					for (ii=il; ii <= ih; ii++) {
						for (jj=jl; jj <= jh; jj++) {
							sbavg += image_data->surface_brightness[ii][jj];
							nn++;
						}
					}
					sbavg /= nn;
					if (sbavg <= qlens->noise_threshold*noise_map[i][j]) resize_grid = false;
				}
				if (resize_grid) {
					if (!qlens->split_imgpixels) {
						xsavg = center_sourcepts[i][j][0];
						ysavg = center_sourcepts[i][j][1];
					} else {
						xsavg=ysavg=0;
						for (k=0; k < n_subpix_per_pixel; k++) {
							xsavg += subpixel_center_sourcepts[i][j][k][0];
							ysavg += subpixel_center_sourcepts[i][j][k][1];
						}
						xsavg /= n_subpix_per_pixel;
						ysavg /= n_subpix_per_pixel;
					}

					if (xsavg < sourcegrid_xmin) {
						if (xsavg > sourcegrid_limit_xmin) sourcegrid_xmin = xsavg;
						else if (sourcegrid_xmin > sourcegrid_limit_xmin) sourcegrid_xmin = sourcegrid_limit_xmin;
					}
					if (xsavg > sourcegrid_xmax) {
						if (xsavg < sourcegrid_limit_xmax) sourcegrid_xmax = xsavg;
						else if (sourcegrid_xmax < sourcegrid_limit_xmax) sourcegrid_xmax = sourcegrid_limit_xmax;
					}
					if (ysavg < sourcegrid_ymin) {
						if (ysavg > sourcegrid_limit_ymin) sourcegrid_ymin = ysavg;
						else if (sourcegrid_ymin > sourcegrid_limit_ymin) sourcegrid_ymin = sourcegrid_limit_ymin;
					}
					if (ysavg > sourcegrid_ymax) {
						if (ysavg < sourcegrid_limit_ymax) sourcegrid_ymax = ysavg;
						else if (sourcegrid_ymax < sourcegrid_limit_ymax) sourcegrid_ymax = sourcegrid_limit_ymax;
					}
				}
			}
		}
	}
	// Now let's make the box slightly wider just to be safe
	double xwidth_adj = 0.1*(sourcegrid_xmax-sourcegrid_xmin);
	double ywidth_adj = 0.1*(sourcegrid_ymax-sourcegrid_ymin);
	sourcegrid_xmin -= xwidth_adj/2;
	sourcegrid_xmax += xwidth_adj/2;
	sourcegrid_ymin -= ywidth_adj/2;
	sourcegrid_ymax += ywidth_adj/2;
}

void ImagePixelGrid::set_sourcegrid_params_from_ray_tracing(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double sourcegrid_limit_xmin, const double sourcegrid_limit_xmax, const double sourcegrid_limit_ymin, const double sourcegrid_limit_ymax)
{
	const double srcgrid_widening = 1e-2;
	if (src_xmin > sourcegrid_limit_xmin) sourcegrid_xmin = src_xmin - srcgrid_widening;
	else sourcegrid_xmin = sourcegrid_limit_xmin;
	if (src_xmax < sourcegrid_limit_xmax) sourcegrid_xmax = src_xmax + srcgrid_widening;
	else sourcegrid_xmax = sourcegrid_limit_xmax;
	if (src_ymin > sourcegrid_limit_ymin) sourcegrid_ymin = src_ymin - srcgrid_widening;
	else sourcegrid_ymin = sourcegrid_limit_ymin;
	if (src_ymax < sourcegrid_limit_ymax) sourcegrid_ymax = src_ymax + srcgrid_widening;
	else sourcegrid_ymax = sourcegrid_limit_ymax;
}

double ImagePixelGrid::find_approx_source_size(double &xcavg, double &ycavg, const bool verbal)
{
	if (image_data == NULL) die("need to have image pixel data loaded to find optimal shapelet scale");
	static const int nmax_srcsize_it = 8;
	//string sp_filename = "wtf_spt.dat";
	//ofstream sourcepts_file; qlens->open_output_file(sourcepts_file,sp_filename);
	//sourcepts_file << setiosflags(ios::scientific);

	double sig;
	double totsurf;
	double area, min_area = 1e30, max_area = -1e30;
	double xcmin, ycmin, sb;
	double xsavg, ysavg;
	double xcold, ycold;
	int i,j,k;
	double rsq, rsqavg;
	sig = 1e30;
	int npts=10000000, npts_old, iter=0;
	if ((verbal) and (qlens->n_ptsrc > 0) and ((qlens->include_imgfluxes_in_inversion) or (qlens->include_srcflux_in_inversion))) warn("estimated approx extended source size may be biased due to point source when 'invert_imgflux' is on");
	xcavg = 0;
	ycavg = 0;
	do {
		// will use 3-sigma clipping to estimate center and dispersion of source
		npts_old = npts;
		xcold = xcavg; // these are just in case an iteration returns no points, in which case it settles on last useful xcavg, ycavg
		ycold = ycavg;
		xcavg = 0;
		ycavg = 0;
		totsurf = 0;
		npts=0;
		for (i=0; i < x_N; i++) {
			for (j=0; j < y_N; j++) {
				//if (foreground_surface_brightness[i][i] != 0) die("YEAH! %g",foreground_surface_brightness[i][j]);
				if ((mask==NULL) or (mask[i][j])) {
					if (mask==NULL) sb = surface_brightness[i][j] - foreground_surface_brightness[i][j];
					else sb = image_data->surface_brightness[i][j] - foreground_surface_brightness[i][j];
					if ((qlens->n_ptsrc > 0) and (!qlens->include_imgfluxes_in_inversion) and (!qlens->include_srcflux_in_inversion) and (point_image_surface_brightness.size() != 0)) sb -= point_image_surface_brightness[pixel_index[i][j]];
					if (abs(sb) > 5*noise_map[i][j]) {
						//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
						//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
						// You repeat this code three times in this function! Store things in arrays and GET RID OF THE REDUNDANCIES!!!! IT'S UGLY.
						if (!qlens->split_imgpixels) {
							xsavg = center_sourcepts[i][j][0];
							ysavg = center_sourcepts[i][j][1];
						} else {
							xsavg=ysavg=0;
							for (k=0; k < n_subpix_per_pixel; k++) {
								xsavg += subpixel_center_sourcepts[i][j][k][0];
								ysavg += subpixel_center_sourcepts[i][j][k][1];
							}
							xsavg /= n_subpix_per_pixel;
							ysavg /= n_subpix_per_pixel;
						}
						//cout << "HI (" << xsavg << "," << ysavg << ") vs (" << center_sourcepts[i][j][0] << "," << center_sourcepts[i][j][1] << ")" << endl;
						area = (source_plane_triangle1_area[i][j] + source_plane_triangle2_area[i][j]);
						rsq = SQR(xsavg - xcavg) + SQR(ysavg - ycavg);
							//cout << "GOT HERE?0 iter=" << iter << endl;
						//cout << "iter " << iter << " sig=" << sig << endl;
						if ((iter==0) or (sqrt(rsq) < 3*sig)) {
							//cout << "GOT HERE?" << endl;
							xcavg += area*abs(sb)*xsavg;
							ycavg += area*abs(sb)*ysavg;
							totsurf += area*abs(sb);
							//cout << "TOTSURF " << totsurf << " " << abs(sb) << " " << area << endl;
							npts++;
						}
						//wtf << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << endl;
					}
				}
			}
		}
		//wtf.close();
		if (npts==0.0) {
			warn("no ray-traced points with high enough surface brightness found when determining source centroid, scale");
			xcavg = xcold;
			ycavg = ycold;
			break;
		} else if (totsurf != 0.0) {
			xcavg /= totsurf;
			ycavg /= totsurf;
		}
		//cout << "HARG0 " << xcavg << " " << ycavg << endl;
		rsqavg=0;
		// NOTE: the approx. sigma found below will be inflated a bit due to the effect of the PSF (but that's probably ok)
		for (i=0; i < x_N; i++) {
			for (j=0; j < y_N; j++) {
				if (((mask==NULL) or (mask[i][j])) and (abs(sb = image_data->surface_brightness[i][j] - foreground_surface_brightness[i][j]) > 5*noise_map[i][j])) {
					//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
					//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
					if (!qlens->split_imgpixels) {
						xsavg = center_sourcepts[i][j][0];
						ysavg = center_sourcepts[i][j][1];
					} else {
						xsavg=ysavg=0;
						for (k=0; k < n_subpix_per_pixel; k++) {
							xsavg += subpixel_center_sourcepts[i][j][k][0];
							ysavg += subpixel_center_sourcepts[i][j][k][1];
						}
						xsavg /= n_subpix_per_pixel;
						ysavg /= n_subpix_per_pixel;
					}

					area = (source_plane_triangle1_area[i][j] + source_plane_triangle2_area[i][j]);
					rsq = SQR(xsavg - xcavg) + SQR(ysavg - ycavg);
					if ((iter==0) or (sqrt(rsq) < 3*sig)) {
						rsqavg += area*abs(sb)*rsq;
					}
				}
			}
		}
		//cout << "rsqavg=" << rsqavg << " totsurf=" << totsurf << endl;
		rsqavg /= totsurf;
		sig = sqrt(abs(rsqavg));
		//cout << "Iteration " << iter << ": sig=" << sig << ", xc=" << xcavg << ", yc=" << ycavg << ", npts=" << npts << endl;
		iter++;
	} while ((iter < nmax_srcsize_it) and (npts < npts_old));
	if (verbal) cout << "sig=" << sig << ", xc=" << xcavg << ", yc=" << ycavg << ", npts=" << npts << endl;
	if (iter >= nmax_srcsize_it) warn("exceeded max iterations for 3-sigma clipping to determine approx source size");
	//cout << "HARG " << xcavg << " " << ycavg << endl;
	return sig;
}



void ImagePixelGrid::find_optimal_shapelet_scale(double& scale, double& xcenter, double& ycenter, double& recommended_nsplit, const bool verbal, double& sig, double& scaled_maxdist)
{
	//string sp_filename = "wtf_spt.dat";
	//ofstream sourcepts_file; qlens->open_output_file(sourcepts_file,sp_filename);
	//sourcepts_file << setiosflags(ios::scientific);

	if (image_data == NULL) die("need to have image pixel data loaded to find optimal shapelet scale");
	double xcavg, ycavg;
	double totsurf;
	double area, min_area = 1e30, max_area = -1e30;
	double xcmin, ycmin, sb;
	double xsavg, ysavg;
	int i,j,k;
	double rsq, rsqavg;
	sig = 1e30;
	int npts=0, npts_old, iter=0;
	do {
		// will use 3-sigma clipping to estimate center and dispersion of source
		npts_old = npts;
		xcavg = 0;
		ycavg = 0;
		totsurf = 0;
		npts=0;
		for (i=0; i < x_N; i++) {
			for (j=0; j < y_N; j++) {
				//if (foreground_surface_brightness[i][i] != 0) die("YEAH! %g",foreground_surface_brightness[i][j]);
				if (((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) and (abs(sb = image_data->surface_brightness[i][j] - foreground_surface_brightness[i][j]) > 5*noise_map[i][j])) {
					//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
					//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
					// You repeat this code three times in this function! Store things in arrays and GET RID OF THE REDUNDANCIES!!!! IT'S UGLY.
					if (!qlens->split_imgpixels) {
						xsavg = center_sourcepts[i][j][0];
						ysavg = center_sourcepts[i][j][1];
					} else {
						xsavg=ysavg=0;
						for (k=0; k < n_subpix_per_pixel; k++) {
							xsavg += subpixel_center_sourcepts[i][j][k][0];
							ysavg += subpixel_center_sourcepts[i][j][k][1];
						}
						xsavg /= n_subpix_per_pixel;
						ysavg /= n_subpix_per_pixel;
					}
					//cout << "HI (" << xsavg << "," << ysavg << ") vs (" << center_sourcepts[i][j][0] << "," << center_sourcepts[i][j][1] << ")" << endl;
					area = (source_plane_triangle1_area[i][j] + source_plane_triangle2_area[i][j]);
					rsq = SQR(xsavg - xcavg) + SQR(ysavg - ycavg);
					if ((iter==0) or (sqrt(rsq) < 3*sig)) {
						xcavg += area*abs(sb)*xsavg;
						ycavg += area*abs(sb)*ysavg;
						totsurf += area*abs(sb);
						npts++;
					}
					//wtf << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << endl;
				}
			}
		}
		//wtf.close();
		xcavg /= totsurf;
		ycavg /= totsurf;
		rsqavg=0;
		// NOTE: the approx. sigma found below will be inflated a bit due to the effect of the PSF (but that's probably ok)
		for (i=0; i < x_N; i++) {
			for (j=0; j < y_N; j++) {
				if (((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) and (abs(sb = image_data->surface_brightness[i][j] - foreground_surface_brightness[i][j]) > 5*noise_map[i][j])) {
					//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
					//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
					if (!qlens->split_imgpixels) {
						xsavg = center_sourcepts[i][j][0];
						ysavg = center_sourcepts[i][j][1];
					} else {
						xsavg=ysavg=0;
						for (k=0; k < n_subpix_per_pixel; k++) {
							xsavg += subpixel_center_sourcepts[i][j][k][0];
							ysavg += subpixel_center_sourcepts[i][j][k][1];
						}
						xsavg /= n_subpix_per_pixel;
						ysavg /= n_subpix_per_pixel;
					}

					area = (source_plane_triangle1_area[i][j] + source_plane_triangle2_area[i][j]);
					rsq = SQR(xsavg - xcavg) + SQR(ysavg - ycavg);
					if ((iter==0) or (sqrt(rsq) < 3*sig)) {
						rsqavg += area*abs(sb)*rsq;
					}
				}
			}
		}
		//cout << "rsqavg=" << rsqavg << " totsurf=" << totsurf << endl;
		rsqavg /= totsurf;
		sig = sqrt(abs(rsqavg));
		//cout << "Iteration " << iter << ": sig=" << sig << ", xc=" << xcavg << ", yc=" << ycavg << ", npts=" << npts << endl;
		iter++;
	} while ((iter < 6) and (npts != npts_old));
	xcenter = xcavg;
	ycenter = ycavg;

	int ntot=0, nout=0;
	double xd,xmax=-1e30;
	double yd,ymax=-1e30;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			//if (((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) and (abs(sb) > 5*noise_map[i][j])) {
			if ((pixel_in_mask==NULL) or ((!qlens->zero_sb_fgmask_prior) and (emask[i][j])) or ((qlens->zero_sb_fgmask_prior) and (mask[i][j]))) {
				//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
				//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
				if (!qlens->split_imgpixels) {
					xsavg = center_sourcepts[i][j][0];
					ysavg = center_sourcepts[i][j][1];
				} else {
					xsavg=ysavg=0;
					for (k=0; k < n_subpix_per_pixel; k++) {
						xsavg += subpixel_center_sourcepts[i][j][k][0];
						ysavg += subpixel_center_sourcepts[i][j][k][1];
					}
					xsavg /= n_subpix_per_pixel;
					ysavg /= n_subpix_per_pixel;
				}

				xd = abs(xsavg-xcavg);
				yd = abs(ysavg-ycavg);

				//double ri = abs(sqrt(SQR(center_pts[i][j][0]-0.01)+SQR(center_pts[i][j][1])));
				//if (ri > 0.6) {
				if (xd > xmax) xmax = xd;
				if (yd > ymax) ymax = yd;
					//sourcepts_file << xsavg << " " << ysavg << " " << center_pts[i][j][0] << " " << center_pts[i][j][1] << " " << xd << " " << yd << endl;
				//}
				if ((mask[i][j]) and (abs(image_data->surface_brightness[i][j] - foreground_surface_brightness[i][j]) > 5*noise_map[i][j])) {
					ntot++;
					rsq = SQR(xd) + SQR(yd);
					if (sqrt(rsq) > 2*sig) {
						nout++;
					}
				}
			}
		}
	}
	double fout = nout / ((double) ntot);
	if ((verbal) and (qlens->mpi_id==0)) cout << "Fraction of 2-sigma outliers for shapelets: " << fout << endl;
	double maxdist = dmax(xmax,ymax);

	int nn = qlens->get_shapelet_nn(src_redshift_index);
	scaled_maxdist = qlens->shapelet_window_scaling*maxdist;
	if (qlens->shapelet_scale_mode==0) {
		scale = sig; // uses the dispersion of source SB to set scale (WARNING: might not cover all of lensed pixels in mask if n_shapelets is too small!!)
	} else if (qlens->shapelet_scale_mode==1) {
		scale = 1.000001*scaled_maxdist/sqrt(nn); // uses outermost pixel in extended mask to set scale
		if (scale > sig) scale = sig; // if the above scale is bigger than ray-traced source scale (sig), then just set scale = sig (otherwise source will be under-resolved)
	}
	if (scale > qlens->shapelet_max_scale) scale = qlens->shapelet_max_scale;

	int ii, jj, il, ih, jl, jh;
	const double window_size_for_srcarea = 1;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			sb = image_data->surface_brightness[i][j] - foreground_surface_brightness[i][j];
			//if (((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) and (abs(sb) > 5*noise_map[i][j])) {
			if (((pixel_in_mask==NULL) or (mask[i][j])) and (abs(sb) > 5*noise_map[i][j])) {
				il = i - window_size_for_srcarea;
				ih = i + window_size_for_srcarea;
				jl = j - window_size_for_srcarea;
				jh = j + window_size_for_srcarea;
				if (il<0) il=0;
				if (ih>x_N-1) ih=x_N-1;
				if (jl<0) jl=0;
				if (jh>y_N-1) jh=y_N-1;
				area=0;
				for (ii=il; ii <= ih; ii++) {
					for (jj=jl; jj <= jh; jj++) {
						area += (source_plane_triangle1_area[ii][jj] + source_plane_triangle2_area[ii][jj]);
					}
				}
				if (area < min_area) {
					min_area = area;
					xcmin = center_pts[i][j][0];
					ycmin = center_pts[i][j][1];
				}
				if (area > max_area) {
					max_area = area;
				}

			}
		}
	}

	double minscale_res = sqrt(min_area);
	recommended_nsplit = 2*sqrt(max_area*nn)/sig; // this is so the smallest source fluctuations get at least 2x2 ray tracing coverage
	int recommended_nn;
	recommended_nn = ((int) (SQR(sig / minscale_res))) + 1;
	if ((verbal) and (qlens->mpi_id==0)) {
		cout << "expected minscale_res=" << minscale_res << ", found at (x=" << xcmin << ",y=" << ycmin << ") recommended_nn=" << recommended_nn << endl;
		cout << "number of splittings should be at least " << recommended_nsplit << " to capture all source fluctuations" << endl;
		cout << "outermost ray-traced source pixel distance: " << scaled_maxdist << endl;
	}
}

void ImagePixelGrid::set_surface_brightness_vector_to_data()
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	int column_j = 0;
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
				//image_surface_brightness[column_j++] = surface_brightness[i][j];
				p.image_surface_brightness[column_j++] = image_data->surface_brightness[i][j];
			}
		}
	}
}

void ImagePixelGrid::plot_grid(string filename, bool show_inactive_pixels)
{
	int i,j;
	string grid_filename = filename + ".pixgrid";
	string center_filename = filename + ".pixcenters";
	ofstream gridfile;
	qlens->open_output_file(gridfile,grid_filename);
	ofstream centerfile;
	qlens->open_output_file(centerfile,center_filename);
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!pixel_in_mask) or (pixel_in_mask[i][j])) {
				//cout << "WHAZZUP " << i << " " << j << endl;
				//if ((show_inactive_pixels) or (maps_to_source_pixel[i][j])) {
					gridfile << corner_sourcepts[i][j][0] << " " << corner_sourcepts[i][j][1] << " " << corner_pts[i][j][0] << " " << corner_pts[i][j][1] << endl;
					gridfile << corner_sourcepts[i+1][j][0] << " " << corner_sourcepts[i+1][j][1] << " " << corner_pts[i+1][j][0] << " " << corner_pts[i+1][j][1] << endl;
					gridfile << corner_sourcepts[i+1][j+1][0] << " " << corner_sourcepts[i+1][j+1][1] << " " << corner_pts[i+1][j+1][0] << " " << corner_pts[i+1][j+1][1] << endl;
					gridfile << corner_sourcepts[i][j+1][0] << " " << corner_sourcepts[i][j+1][1] << " " << corner_pts[i][j+1][0] << " " << corner_pts[i][j+1][1] << endl;
					gridfile << corner_sourcepts[i][j][0] << " " << corner_sourcepts[i][j][1] << " " << corner_pts[i][j][0] << " " << corner_pts[i][j][1] << endl;
					gridfile << endl;
					centerfile << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " " << center_pts[i][j][0] << " " << center_pts[i][j][1] << endl;
				//}
			}
		}
	}
}

void ImagePixelGrid::find_optimal_sourcegrid_npixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels)
{
	int i,j,count=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
				if ((center_sourcepts[i][j][0] > srcgrid_xmin) and (center_sourcepts[i][j][0] < srcgrid_xmax) and (center_sourcepts[i][j][1] > srcgrid_ymin) and (center_sourcepts[i][j][1] < srcgrid_ymax)) {
					count++;
				}
			}
		}
	}
	double dx = srcgrid_xmax-srcgrid_xmin;
	double dy = srcgrid_ymax-srcgrid_ymin;
	nsrcpixel_x = (int) sqrt(cartesian_srcgrid->pixel_fraction*count*dx/dy);
	nsrcpixel_y = (int) nsrcpixel_x*dy/dx;
	n_expected_active_pixels = count;
}

void ImagePixelGrid::find_optimal_firstlevel_sourcegrid_npixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels)
{
	// this algorithm assumes an adaptive grid, so that higher magnification regions will be subgridded
	// it really doesn't seem to work well though...
	double lowest_magnification = 1e30;
	double average_magnification = 0;
	int i,j,count=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
				if ((center_sourcepts[i][j][0] > srcgrid_xmin) and (center_sourcepts[i][j][0] < srcgrid_xmax) and (center_sourcepts[i][j][1] > srcgrid_ymin) and (center_sourcepts[i][j][1] < srcgrid_ymax)) {
					count++;
				}
			}
		}
	}

	double pixel_area, source_lowlevel_pixel_area, dx, dy, srcgrid_area, srcgrid_firstlevel_npixels;
	pixel_area = pixel_xlength * pixel_ylength;
	source_lowlevel_pixel_area = pixel_area / (1.3*cartesian_srcgrid->pixel_magnification_threshold);
	dx = srcgrid_xmax-srcgrid_xmin;
	dy = srcgrid_ymax-srcgrid_ymin;
	srcgrid_area = dx*dy;
	srcgrid_firstlevel_npixels = dx*dy/source_lowlevel_pixel_area;
	nsrcpixel_x = (int) sqrt(srcgrid_firstlevel_npixels*dx/dy);
	nsrcpixel_y = (int) nsrcpixel_x*dy/dx;
	int srcgrid_npixels = nsrcpixel_x*nsrcpixel_y;
	n_expected_active_pixels = count;
}

/*
void ImagePixelGrid::assign_mask_pixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& count, ImageData *data_in)
{
	int i,j;
	count=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((center_sourcepts[i][j][0] > srcgrid_xmin) and (center_sourcepts[i][j][0] < srcgrid_xmax) and (center_sourcepts[i][j][1] > srcgrid_ymin) and (center_sourcepts[i][j][1] < srcgrid_ymax)) {
				data_in->in_mask[i][j] = true;
				count++;
			}
			else {
				data_in->in_mask[i][j] = false;
			}
		}
	}
}
*/

int ImagePixelGrid::count_nonzero_source_pixel_mappings_cartesian()
{
	int tot=0;
	int i,j,k,img_index;
	//if (qlens->psf_supersampling) nsubpix = SQR(qlens->default_imgpixel_nsplit);
	for (img_index=0; img_index < image_npixels; img_index++) {
		i = mask_pixels_i[img_index];
		j = mask_pixels_j[img_index];
		for (k=0; k < mapped_cartesian_srcpixels[i][j].size(); k++) {
			if (mapped_cartesian_srcpixels[i][j][k] != NULL) {
				 tot++;
				//else tot += nsubpix;
			}
		}
		//tot += mapped_cartesian_srcpixels[i][j].size();
	}
	return tot;
}

int ImagePixelGrid::count_nonzero_source_pixel_mappings_delaunay()
{
	int tot=0;
	int i,j,k,img_index;
	//if (qlens->psf_supersampling) nsubpix = SQR(qlens->default_imgpixel_nsplit);
	//cout << "N_ACTIVE_PIXELS=" << n_active_pixels << endl;
	//if (mask_pixels_i == NULL) die("WHJAT? NULL ACTIVE IMAGE PIXEL_I ARRAY");
	for (img_index=0; img_index < image_npixels; img_index++) {
		//cout << "img_index=" << img_index << endl;
		i = mask_pixels_i[img_index];
		j = mask_pixels_j[img_index];
		for (k=0; k < mapped_delaunay_srcpixels[i][j].size(); k++) {
			 tot++;
			//else tot += nsubpix;
			//if (mapped_delaunay_srcpixels[i][j][k] != -1) tot++;
		}
		//tot += mapped_delaunay_srcpixels[i][j].size();
	}
	return tot;
}

int ImagePixelGrid::count_nonzero_lensgrid_pixel_mappings()
{
	int tot=0;
	int i,j,k,img_index;
	//if (qlens->psf_supersampling) nsubpix = SQR(qlens->default_imgpixel_nsplit);
	for (img_index=0; img_index < image_npixels; img_index++) {
		i = mask_pixels_i[img_index];
		j = mask_pixels_j[img_index];
		for (k=0; k < mapped_potpixels[i][j].size(); k++) {
			 tot++;
			//else tot += nsubpix;
			//if (mapped_delaunay_srcpixels[i][j][k] != -1) tot++;
		}
		//tot += mapped_delaunay_srcpixels[i][j].size();
	}
	return tot;
}

void ImagePixelGrid::assign_image_mapping_flags(const bool delaunay, const bool potential_perturbations, const bool map_all_imgpixels)
{
	//ImgGrid_Params<Eigen::VectorXd,Eigen::MatrixXd,double>& p = assign_imggrid_param_object<Eigen::VectorXd,Eigen::MatrixXd,double>();
	//cout << "ASSIGNING MAPPING FLASGS" << endl;
	int i,j,k;
	n_active_pixels = 0;
	n_high_sn_pixels = 0;
	int *ptr;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (delaunay) mapped_delaunay_srcpixels[i][j].clear();
			else mapped_cartesian_srcpixels[i][j].clear();
			maps_to_source_pixel[i][j] = false;
			if ((map_all_imgpixels) and ((pixel_in_mask == NULL) or (pixel_in_mask[i][j]))) {
				maps_to_source_pixel[i][j] = true;
			}
			ptr = n_mapped_srcpixels[i][j];
			for (k=0; k < n_subpix_per_pixel; k++) {
				(*ptr++) = 0;
			}
			if (potential_perturbations) {
				mapped_potpixels[i][j].clear();
				ptr = n_mapped_potpixels[i][j];
				for (k=0; k < n_subpix_per_pixel; k++) {
					(*ptr++) = 0;
				}
			}
		}
	}
	bool trouble_with_starting_vertex = false;
	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif
		if ((qlens->split_imgpixels) and (!qlens->raytrace_using_pixel_centers)) {
			int subcell_index;
			bool maps_to_something = true;
			//#pragma omp for private(i,j,nsubpix,subcell_index,maps_to_something) schedule(dynamic)
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					if ((pixel_in_mask == NULL) or (pixel_in_mask[i][j])) {
						if (!map_all_imgpixels) maps_to_something = false;
						for (subcell_index=0; subcell_index < n_subpix_per_pixel; subcell_index++)
						{
							if ((delaunay) and ((delaunay_srcgrid == NULL) or (delaunay_srcgrid->assign_source_mapping_flags(subpixel_center_sourcepts[i][j][subcell_index],mapped_delaunay_srcpixels[i][j],n_mapped_srcpixels[i][j][subcell_index],i,j,thread,trouble_with_starting_vertex)==true))) {
								maps_to_something = true;
								subpixel_maps_to_srcpixel[i][j][subcell_index] = true;
							} else if ((!delaunay) and (cartesian_srcgrid->assign_source_mapping_flags_interpolate(subpixel_center_sourcepts[i][j][subcell_index],mapped_cartesian_srcpixels[i][j],thread,i,j)==true)) {
								maps_to_something = true;
								subpixel_maps_to_srcpixel[i][j][subcell_index] = true;
							} else {
								subpixel_maps_to_srcpixel[i][j][subcell_index] = false;
							}
							if (potential_perturbations) lensgrid->assign_mapping_flags(subpixel_center_pts[i][j][subcell_index],mapped_potpixels[i][j],n_mapped_potpixels[i][j][subcell_index],i,j,thread);
							//cout << "n_mapped_srcpixels: " << n_mapped_srcpixels[i][j][subcell_index] << " (imggrid_i=" << src_redshift_index << ")" << endl;
						}
						if (maps_to_something==true) {
							maps_to_source_pixel[i][j] = true;
							#pragma omp atomic
							n_active_pixels++;
							if ((pixel_in_mask != NULL) and (pixel_in_mask[i][j]) and (image_data->high_sn_pixel[i][j])) {
								#pragma omp atomic
								n_high_sn_pixels++;
							}
						}
					}
				}
			}
		} else {
			//#pragma omp for private(i,j) schedule(dynamic)	
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					if ((pixel_in_mask == NULL) or (pixel_in_mask[i][j])) {
						if ((delaunay) and ((delaunay_srcgrid==NULL) or (delaunay_srcgrid->assign_source_mapping_flags(center_sourcepts[i][j],mapped_delaunay_srcpixels[i][j],n_mapped_srcpixels[i][j][0],i,j,thread,trouble_with_starting_vertex)==true))) {
							maps_to_source_pixel[i][j] = true;
							#pragma omp atomic
							n_active_pixels++;
							if ((pixel_in_mask != NULL) and (pixel_in_mask[i][j]) and (image_data->high_sn_pixel[i][j])) {
								#pragma omp atomic
								n_high_sn_pixels++;
							}
						} else if ((!delaunay) and (cartesian_srcgrid->assign_source_mapping_flags_interpolate(center_sourcepts[i][j],mapped_cartesian_srcpixels[i][j],thread,i,j)==true)) {
							maps_to_source_pixel[i][j] = true;
							#pragma omp atomic
							n_active_pixels++;
							if ((pixel_in_mask != NULL) and (pixel_in_mask[i][j]) and (image_data->high_sn_pixel[i][j])) {
								#pragma omp atomic
								n_high_sn_pixels++;
							}
						}
						if (potential_perturbations) {
							lensgrid->assign_mapping_flags(center_pts[i][j],mapped_potpixels[i][j],n_mapped_potpixels[i][j][0],i,j,thread);
						}
					}
				}
			}
		}
	}

	if (trouble_with_starting_vertex) warn(qlens->warnings,"could not find good starting vertices for Delaunay grid; started with vertex 0 when searching for enclosing triangles");


	//int toto=0;
	//for (j=0; j < y_N; j++) {
		//for (i=0; i < x_N; i++) {
			//for (k=0; k < nsubpix; k++) {
				////if (n_mapped_srcpixels[i][j][k] != 0) cout << "YO n_mapped_srcpixels: " << n_mapped_srcpixels[i][j][k] << " (imggrid_i=" << src_redshift_index << ")" << endl;
				//toto += n_mapped_srcpixels[i][j][k];
			//}
		//}
	//}
	//cout << "TOT n_mapped_srcpixels=" << toto << " (imggrid_i=" << src_redshift_index << ")" << endl;
}

template <typename QScalar>
void ImagePixelGrid::set_zero_lensed_surface_brightness()
{
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			surface_brightness[i][j] = 0;
		}
	}
}
template void ImagePixelGrid::set_zero_lensed_surface_brightness<double>();
#ifdef USE_STAN
template void ImagePixelGrid::set_zero_lensed_surface_brightness<stan::math::var>();
#endif

template <typename QScalar>
void ImagePixelGrid::set_zero_foreground_surface_brightness()
{
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			foreground_surface_brightness[i][j] = 0;
		}
	}
}
template void ImagePixelGrid::set_zero_foreground_surface_brightness<double>();
#ifdef USE_STAN
template void ImagePixelGrid::set_zero_foreground_surface_brightness<stan::math::var>();
#endif

void ImagePixelGrid::find_surface_brightness(const bool use_extended_mask, const bool foreground_only, const bool lensed_sources_only, const bool include_first_order_corrections, const bool show_only_first_order_corrections, const bool omit_noninverted_sources)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	if ((source_fit_mode==Delaunay_Source) and (delaunay_srcgrid == NULL)) die("No Delaunay source grid has been created");
	else if ((source_fit_mode==Cartesian_Source) and (cartesian_srcgrid == NULL)) die("No Cartesian source grid has been created");
	bool supersampling = qlens->psf_supersampling;
	double noise;

	bool at_least_one_foreground_src = false;
	bool at_least_one_lensed_or_inverted_src = false;
	for (int k=0; k < qlens->n_sb; k++) {
		if (qlens->sbprofile_band_number[k]==band_number) {
			if (!qlens->sb_list[k]->is_lensed) {
				at_least_one_foreground_src = true;
			} else {
				at_least_one_lensed_or_inverted_src = true;
			}
		}
	}
	if ((foreground_only) and (!at_least_one_foreground_src)) return;

	bool **selected_mask;
	if (use_extended_mask) {
		if (emask==NULL) selected_mask = NULL;
		else selected_mask = emask;
	} else {
		if (pixel_in_mask==NULL) selected_mask = NULL;
		else selected_mask = pixel_in_mask;
	}

	std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			surface_brightness[i][j] = 0;
		}
	}
	double sbmax=0;
	if ((foreground_only) and (src_redshift_index > 0)) return; // only the first image_pixel_grid object will have foreground light included
	if ((source_fit_mode == Cartesian_Source) or (source_fit_mode == Delaunay_Source)) {
		if ((qlens->split_imgpixels) and (!qlens->raytrace_using_pixel_centers)) {
			//#pragma omp parallel
			{
				int thread;
//#ifdef USE_OPENMP
				//thread = omp_get_thread_num();
//#else
				thread = 0;
//#endif
				//int ii,jj,nsplit;
				//double u0, w0, sb;
				double sb, sbtot;

				int subcell_index;
				lensvector<double> *center_srcpt, *center_pt;
				//#pragma omp for private(i,j,ii,jj,nsplit,u0,w0,sb) schedule(dynamic)
				#pragma omp for private(i,j,subcell_index,center_pt,center_srcpt,sb,noise) schedule(dynamic)
				for (j=0; j < y_N; j++) {
					for (i=0; i < x_N; i++) {
						//surface_brightness[i][j] = 0;
						if ((selected_mask == NULL) or (selected_mask[i][j])) {
							sbtot=0;

							center_srcpt = subpixel_center_sourcepts[i][j];
							center_pt = subpixel_center_pts[i][j];
							for (subcell_index=0; subcell_index < n_subpix_per_pixel; subcell_index++) {
								if (!foreground_only) {
									if (source_fit_mode==Delaunay_Source) {
										if (!show_only_first_order_corrections) {
											sb = delaunay_srcgrid->find_lensed_surface_brightness(center_srcpt[subcell_index],i,j,thread);
										} else {
											sb = 0;
										}
										if ((include_first_order_corrections) and (lensgrid)) {
											//lensvector<double> S0_gradient;
											//delaunay_srcgrid->find_source_gradient(center_sourcepts[i][j],S0_gradient,thread);
											//sb += lensgrid->first_order_surface_brightness_correction(center_srcpt[subcell_index],S0_gradient,thread);

											/*
											double unperturbed_sb = sb;
											lensvector<double> defl;
											lensgrid->deflection(center_pt[subcell_index][0],center_pt[subcell_index][1],defl,thread);
											lensvector<double> new_srcpt = center_srcpt[subcell_index] - defl;
											double SB_check = delaunay_srcgrid->find_lensed_surface_brightness(new_srcpt,i,j,thread);
											double sb_pert0 = SB_check - sb;

											double sb_pert1 = lensgrid->first_order_surface_brightness_correction(center_pt[subcell_index],subpixel_source_gradient[i][j][subcell_index],thread);

											double interval = 1e-4;
											lensvector<double> ptp = center_srcpt[subcell_index];
											ptp[0] += interval;
											lensvector<double> ptm = center_srcpt[subcell_index];
											ptm[0] -= interval;
											//double sbgrad_x = (delaunay_srcgrid->find_lensed_surface_brightness(ptp,i,j,thread) - delaunay_srcgrid->find_lensed_surface_brightness(ptm,i,j,thread)) / (2*interval);
											double sbgrad_xp = (delaunay_srcgrid->find_lensed_surface_brightness(ptp,i,j,thread) - delaunay_srcgrid->find_lensed_surface_brightness(center_srcpt[subcell_index],i,j,thread)) / interval;
											double sbgrad_xm = (delaunay_srcgrid->find_lensed_surface_brightness(center_srcpt[subcell_index],i,j,thread) - delaunay_srcgrid->find_lensed_surface_brightness(ptm,i,j,thread)) / interval;
											lensvector<double> ptpy = center_srcpt[subcell_index];
											ptpy[1] += interval;
											lensvector<double> ptmy = center_srcpt[subcell_index];
											ptmy[1] -= interval;
											//double sbgrad_y = (delaunay_srcgrid->find_lensed_surface_brightness(ptpy,i,j,thread) - delaunay_srcgrid->find_lensed_surface_brightness(ptmy,i,j,thread)) / (2*interval);
											double sbgrad_yp = (delaunay_srcgrid->find_lensed_surface_brightness(ptpy,i,j,thread) - delaunay_srcgrid->find_lensed_surface_brightness(center_srcpt[subcell_index],i,j,thread)) / interval;
											double sbgrad_ym = (delaunay_srcgrid->find_lensed_surface_brightness(center_srcpt[subcell_index],i,j,thread) - delaunay_srcgrid->find_lensed_surface_brightness(ptmy,i,j,thread)) / interval;
											double sbgrad_x = (defl[0] < 0) ? sbgrad_xp : sbgrad_xm;
											double sbgrad_y = (defl[1] < 0) ? sbgrad_yp : sbgrad_ym;
											double sbcheck2 = unperturbed_sb - sbgrad_x*defl[0] - sbgrad_y*defl[1];
											double pert = -sbgrad_x*defl[0] - sbgrad_y*defl[1];
											//sb += lensgrid->first_order_surface_brightness_correction(center_pt[subcell_index],S0_gradient,thread);

											if (unperturbed_sb > 0.13) cout << "SB_CHECK: S0=" << unperturbed_sb << " SBpert_1st=" << sb_pert1 << " SBpert_def=" << sb_pert0 << " pertcheck=" << pert << " sbgradc_x=" << sbgrad_x << " sbgradc_y=" << sbgrad_y << " sbgrad_x=" << subpixel_source_gradient[i][j][subcell_index][0] << " sbgrad_y=" << subpixel_source_gradient[i][j][subcell_index][1] << " def: " << defl[0] << " " << defl[1] << endl;
											//lensvector<double> SBgrad(sbgrad_x,sbgrad_y);
											//sb += lensgrid->first_order_surface_brightness_correction(center_pt[subcell_index],SBgrad,thread);
											//sb = SB_check;
											*/

											sb += lensgrid->first_order_surface_brightness_correction(center_pt[subcell_index],subpixel_source_gradient[i][j][subcell_index],thread);
										}
									}
									else if (source_fit_mode==Cartesian_Source) sb = cartesian_srcgrid->find_lensed_surface_brightness_interpolate(center_srcpt[subcell_index],thread);
								}
								if (supersampling) subpixel_surface_brightness[i][j][subcell_index] = sb;
								sbtot += sb;
							}
							surface_brightness[i][j] += sbtot / n_subpix_per_pixel;
						}
					}
				}
			}
		} else {
			//ofstream hergls("hergls.dat");
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					//surface_brightness[i][j] = 0;
					if ((selected_mask == NULL) or (selected_mask[i][j])) {
						if (!foreground_only) {
							if (source_fit_mode==Delaunay_Source) {
								if (!show_only_first_order_corrections) {
									surface_brightness[i][j] = delaunay_srcgrid->find_lensed_surface_brightness(center_sourcepts[i][j],i,j,0);
								} else {
									surface_brightness[i][j] = 0;
								}
								//if ((abs(center_sourcepts[i][j][0]) < 0.3) and (abs(center_sourcepts[i][j][1]) < 0.3)) cout << "HARG " << center_pts[i][j][0] << " " << center_pts[i][j][1] << " " << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " " << surface_brightness[i][j] << endl;
								//hergls << center_pts[i][j][0] << " " << center_pts[i][j][1] << " " << surface_brightness[i][j] << endl;
								if (include_first_order_corrections) {
								//lensvector<double> S0_gradient;
									//delaunay_srcgrid->find_source_gradient(center_sourcepts[i][j],S0_gradient,0);
									//
									//double S0_check2 = delaunay_srcgrid->interpolate_surface_brightness(center_sourcepts[i][j],false,0);
									//cout << "CHECK x0: " << x0_check[i][j][0] << " " << center_sourcepts[i][j][0] << endl;
									//cout << "CHECK y0: " << x0_check[i][j][1] << " " << center_sourcepts[i][j][1] << endl;
									//cout << "CHECK SB0: " << S0_check[i][j] << " " << S0_check2 << endl;
									//cout << "CHECK SBGRAD: " << S0_gradient[0] << " " << subpixel_source_gradient[i][j][0][0] << " " << S0_gradient[1] << " " << subpixel_source_gradient[i][j][0][1] << endl;
									/*
									double unperturbed_sb = surface_brightness[i][j];
									lensvector<double> defl;
									lensgrid->deflection(center_pts[i][j][0],center_pts[i][j][1],defl,0);
									lensvector<double> new_srcpt = center_sourcepts[i][j] - defl;
									double SB_check = delaunay_srcgrid->find_lensed_surface_brightness(new_srcpt,i,j,0);
									double sb_pert0 = SB_check - surface_brightness[i][j];

									double sb_pert1 = lensgrid->first_order_surface_brightness_correction(center_pts[i][j],subpixel_source_gradient[i][j][0],0);

									double interval = 1e-4;
									lensvector<double> ptp = center_sourcepts[i][j];
									ptp[0] += interval;
									lensvector<double> ptm = center_sourcepts[i][j];
									ptm[0] -= interval;
									//double sbgrad_x = (delaunay_srcgrid->find_lensed_surface_brightness(ptp,i,j,0) - delaunay_srcgrid->find_lensed_surface_brightness(ptm,i,j,0)) / (2*interval);
									double sbgrad_xp = (delaunay_srcgrid->find_lensed_surface_brightness(ptp,i,j,0) - delaunay_srcgrid->find_lensed_surface_brightness(center_sourcepts[i][j],i,j,0)) / interval;
									double sbgrad_xm = (delaunay_srcgrid->find_lensed_surface_brightness(center_sourcepts[i][j],i,j,0) - delaunay_srcgrid->find_lensed_surface_brightness(ptm,i,j,0)) / interval;
									lensvector<double> ptpy = center_sourcepts[i][j];
									ptpy[1] += interval;
									lensvector<double> ptmy = center_sourcepts[i][j];
									ptmy[1] -= interval;
									//double sbgrad_y = (delaunay_srcgrid->find_lensed_surface_brightness(ptpy,i,j,0) - delaunay_srcgrid->find_lensed_surface_brightness(ptmy,i,j,0)) / (2*interval);
									double sbgrad_yp = (delaunay_srcgrid->find_lensed_surface_brightness(ptpy,i,j,0) - delaunay_srcgrid->find_lensed_surface_brightness(center_sourcepts[i][j],i,j,0)) / interval;
									double sbgrad_ym = (delaunay_srcgrid->find_lensed_surface_brightness(center_sourcepts[i][j],i,j,0) - delaunay_srcgrid->find_lensed_surface_brightness(ptmy,i,j,0)) / interval;
									double sbgrad_x = (defl[0] < 0) ? sbgrad_xp : sbgrad_xm;
									double sbgrad_y = (defl[1] < 0) ? sbgrad_yp : sbgrad_ym;
									double sbcheck2 = unperturbed_sb - sbgrad_x*defl[0] - sbgrad_y*defl[1];
									double pert = -sbgrad_x*defl[0] - sbgrad_y*defl[1];
									//surface_brightness[i][j] += lensgrid->first_order_surface_brightness_correction(center_pts[i][j],S0_gradient,0);

									cout << "SB_CHECK: S0=" << unperturbed_sb << " SBpert_1st=" << sb_pert1 << " SBpert_def=" << sb_pert0 << " pertcheck=" << pert << " sbgradc_x=" << sbgrad_x << " sbgradc_y=" << sbgrad_y << " sbgrad_x=" << subpixel_source_gradient[i][j][0][0] << " sbgrad_y=" << subpixel_source_gradient[i][j][0][1] << " def: " << defl[0] << " " << defl[1] << endl;
									*/
									surface_brightness[i][j] += lensgrid->first_order_surface_brightness_correction(center_pts[i][j],subpixel_source_gradient[i][j][0],0);
									//lensvector<double> SBgrad(sbgrad_x,sbgrad_y);
									//surface_brightness[i][j] += lensgrid->first_order_surface_brightness_correction(center_pts[i][j],SBgrad,0);
									//surface_brightness[i][j] = unperturbed_sb + pert;

/*
									double smin = -defl.norm()*2;
									double smax = defl.norm()*2;
									int ii, nsteps=600;
									double s, sstep = 2*smax/nsteps;
									lensvector<double> unitvec = defl/defl.norm();
									double SB_checkbla;
									if (surface_brightness[i][j] > 0.5) {
										ofstream sbcheckout("sbcheck.dat");
										for (ii=0,s=smin; ii < nsteps; ii++, s += sstep) {
											new_srcpt = center_sourcepts[i][j] + s*unitvec;
											SB_checkbla = delaunay_srcgrid->find_lensed_surface_brightness(new_srcpt,i,j,0);
											sbcheckout << s << " " << SB_checkbla << endl;
										}
									cout << "SB_CHECK: S0=" << unperturbed_sb << " SB_1st=" << surface_brightness[i][j] << " SB_def=" << SB_check << " SB_check2=" << sbcheck2 << " " << pert << " " << sbgrad_x << " " << sbgrad_y << " " << defl[0] << " " << defl[1] << endl;
									double blerk = abs((surface_brightness[i][j] - SB_check)/SB_check);
									if (blerk > 1.0) die();
									}
									*/
								}
							} else {
								surface_brightness[i][j] = cartesian_srcgrid->find_lensed_surface_brightness_interpolate(center_sourcepts[i][j],0);
							}
						}
					}
					//if ((at_least_one_noninverted_foreground_src) and (!lensed_sources_only) and (src_redshift_index==0)) {
						//for (int k=0; k < qlens->n_sb; k++) {
							//if (!qlens->sb_list[k]->is_lensed) {
								//if (!qlens->sb_list[k]->zoom_subgridding) {
									//surface_brightness[i][j] += qlens->sb_list[k]->surface_brightness(center_pts[i][j][0],center_pts[i][j][1]);
								//} else {
									//noise = (qlens->use_noise_map) ? noise_map[i][j] : qlens->background_pixel_noise;
									//surface_brightness[i][j] += qlens->sb_list[k]->surface_brightness_zoom(center_pts[i][j],corner_pts[i][j],corner_pts[i+1][j],corner_pts[i][j+1],corner_pts[i+1][j+1],noise);
								//}
							//}
						//}
					//}
				}
			}
		}
	}

	// Now we deal with lensed and unlensed source objects, if they exist
	if ((lensed_sources_only) and (!at_least_one_lensed_or_inverted_src)) ;
	else if (qlens->split_imgpixels) {
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			double sb,sbtot;
			lensvector<double> corner1, corner2, corner3, corner4;
			double subpixel_xlength, subpixel_ylength;

			int subcell_index;
			lensvector<double> *center_srcpt, *center_pt;
			#pragma omp for private(i,j,sb,subcell_index,subpixel_xlength,subpixel_ylength,center_pt,center_srcpt,corner1,corner2,corner3,corner4,noise) schedule(dynamic)
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					if ((selected_mask == NULL) or (selected_mask[i][j])) {
						sbtot=0;
						center_srcpt = subpixel_center_sourcepts[i][j];
						center_pt = subpixel_center_pts[i][j];
						for (subcell_index=0; subcell_index < n_subpix_per_pixel; subcell_index++) {
							for (int k=0; k < qlens->n_sb; k++) {
								if (qlens->sbprofile_band_number[k]==band_number) {
									if ((!omit_noninverted_sources) or (qlens->sb_list[k]->sbtype==SHAPELET) or (qlens->sb_list[k]->sbtype==MULTI_GAUSSIAN_EXPANSION)) {
										if ((qlens->sb_list[k]->is_lensed) and (qlens->sbprofile_redshift_idx[k]==src_redshift_index)) {
											if (!foreground_only) {
												sb = qlens->sb_list[k]->surface_brightness(center_srcpt[subcell_index][0],center_srcpt[subcell_index][1]);
											}
										} else if ((!lensed_sources_only) and (!qlens->sb_list[k]->is_lensed) and (src_redshift_index==0)) { // this is ugly. Should just generate a list (near the beginning of this function) of which sources will be used!
											if (!qlens->sb_list[k]->zoom_subgridding) {
												//cout << " center pt: " << center_pt[subcell_index][0] << " " << center_pt[subcell_index][1] << " (should be near " << center_pts[i][j][0] << " " << center_pts[i][j][1] << ")" << endl;
//#ifdef USE_STAN
												sb = qlens->sb_list[k]->surface_brightness(center_pt[subcell_index][0],center_pt[subcell_index][1]);
											}
											else {
												subpixel_xlength = pixel_xlength/imgpixel_nsplit;
												subpixel_ylength = pixel_ylength/imgpixel_nsplit;
												corner1[0] = center_pt[subcell_index][0] - subpixel_xlength/2;
												corner1[1] = center_pt[subcell_index][1] - subpixel_ylength/2;
												corner2[0] = center_pt[subcell_index][0] + subpixel_xlength/2;
												corner2[1] = center_pt[subcell_index][1] - subpixel_ylength/2;
												corner3[0] = center_pt[subcell_index][0] - subpixel_xlength/2;
												corner3[1] = center_pt[subcell_index][1] + subpixel_ylength/2;
												corner4[0] = center_pt[subcell_index][0] + subpixel_xlength/2;
												corner4[1] = center_pt[subcell_index][1] + subpixel_ylength/2;
												noise = (qlens->use_noise_map) ? noise_map[i][j] : qlens->background_pixel_noise;
												sb = qlens->sb_list[k]->surface_brightness_zoom(center_pt[subcell_index],corner1,corner2,corner3,corner4,noise);
											}
										} else {
											sb=0;
										}
										if (sb != 0) {
											if (supersampling) subpixel_surface_brightness[i][j][subcell_index] = sb;
											sbtot += sb;
										}
									}
								}
							}
						}
						surface_brightness[i][j] += sbtot / n_subpix_per_pixel;
					}
				}
			}
		}
	} else {
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			#pragma omp for private(i,j,noise) schedule(dynamic)
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					for (int k=0; k < qlens->n_sb; k++) {
						if (qlens->sbprofile_band_number[k]==band_number) {
							if ((!omit_noninverted_sources) or (qlens->sb_list[k]->sbtype==SHAPELET) or (qlens->sb_list[k]->sbtype==MULTI_GAUSSIAN_EXPANSION)) {
								if ((qlens->sb_list[k]->is_lensed) and (qlens->sbprofile_redshift_idx[k]==src_redshift_index)) {
									if (!foreground_only) {
										if (!qlens->sb_list[k]->zoom_subgridding) surface_brightness[i][j] += qlens->sb_list[k]->surface_brightness(center_sourcepts[i][j][0],center_sourcepts[i][j][1]);
										else {
											noise = (qlens->use_noise_map) ? noise_map[i][j] : qlens->background_pixel_noise;
											surface_brightness[i][j] += qlens->sb_list[k]->surface_brightness_zoom(center_sourcepts[i][j],corner_sourcepts[i][j],corner_sourcepts[i+1][j],corner_sourcepts[i][j+1],corner_sourcepts[i+1][j+1],noise);
										}
									}
								}
								else if ((!lensed_sources_only) and (!qlens->sb_list[k]->is_lensed) and (src_redshift_index==0)) { // this is ugly. Should just generate a list (near the beginning of this function) of which sources will be used!
									if (!qlens->sb_list[k]->zoom_subgridding) {
										surface_brightness[i][j] += qlens->sb_list[k]->surface_brightness(center_pts[i][j][0],center_pts[i][j][1]);
									}
									else {
										noise = (qlens->use_noise_map) ? noise_map[i][j] : qlens->background_pixel_noise;
										surface_brightness[i][j] += qlens->sb_list[k]->surface_brightness_zoom(center_pts[i][j],corner_pts[i][j],corner_pts[i+1][j],corner_pts[i][j+1],corner_pts[i+1][j+1],noise);

									}
								}
							}
						}
					}
				}
			}
		}
	}
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for ray-tracing image surface brightness values: "  << wtime.count() << endl;
	}
}

template <typename MathTypes>
void ImagePixelGrid::find_surface_brightness_sbprofile(const bool foreground_only, const bool lensed_sources_only, const bool omit_noninverted_sources, const bool psf_convolution)
{
	ImgGrid_Params<MathTypes>& p = assign_imggrid_param_object<MathTypes>();
	bool supersampling = qlens->psf_supersampling;

	bool at_least_one_foreground_src = false;
	bool at_least_one_lensed_or_inverted_src = false;
	vector<SB_Profile*> sbprofiles_this_imggrid;
	vector<SB_Profile*> fg_sbprofiles_this_imggrid;
	for (int k=0; k < qlens->n_sb; k++) {
		if (qlens->sbprofile_band_number[k]==band_number) {
			if (!qlens->sb_list[k]->is_lensed) {
				if (src_redshift_index==0) { // foreground sbprofiles are only included for imggrids with src_redshift_index=0
					at_least_one_foreground_src = true;
					fg_sbprofiles_this_imggrid.push_back(qlens->sb_list[k]);
				}
			} else if ((!foreground_only) and (qlens->sbprofile_redshift_idx[k]==src_redshift_index)) {
				if ((!omit_noninverted_sources) or (sbprofiles_this_imggrid[k]->sbtype==SHAPELET) or (sbprofiles_this_imggrid[k]->sbtype==MULTI_GAUSSIAN_EXPANSION)) {
					at_least_one_lensed_or_inverted_src = true;
					sbprofiles_this_imggrid.push_back(qlens->sb_list[k]);
				}
			}
		}
	}
	if ((foreground_only) and (!at_least_one_foreground_src)) return;

	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			surface_brightness[i][j] = 0;
		}
	}
	if ((foreground_only) and (src_redshift_index > 0)) return; // only the first image_pixel_grid object will have foreground light included

	// Now we deal with lensed and unlensed source objects, if they exist
	if ((lensed_sources_only) and (!at_least_one_lensed_or_inverted_src)) return;

	int npix = image_npixels;
	int nsp = SQR(qlens->default_imgpixel_nsplit);
	p.image_surface_brightness = Eigen::VectorXd::Zero(npix);
	if (qlens->split_imgpixels) {
		if (supersampling) {
			int ntot_subpixels = p.srcpt_x_subpixel_centers.size();
			p.image_surface_brightness_supersampled = Eigen::VectorXd::Zero(ntot_subpixels);
			for (int k=0; k < sbprofiles_this_imggrid.size(); k++) {
				sbprofiles_this_imggrid[k]->surface_brightness_vec(p.srcpt_x_subpixel_centers,p.srcpt_y_subpixel_centers,p.image_surface_brightness_supersampled);
			}
		} else {
			for (int k=0; k < sbprofiles_this_imggrid.size(); k++) {
				sbprofiles_this_imggrid[k]->surface_brightness_vec(p.srcpt_x_subpixel_centers,p.srcpt_y_subpixel_centers,p.image_surface_brightness,nsp);
			}
		}
	} else {
		for (int k=0; k < sbprofiles_this_imggrid.size(); k++) {
			sbprofiles_this_imggrid[k]->surface_brightness_vec(p.srcpt_x_centers,p.srcpt_y_centers,p.image_surface_brightness);
		}
	}

	if ((psf_convolution) and (psf != NULL) and (psf->use_input_psf_matrix)) {
		if (!psf_convolution_is_setup) {
			//cout << "checking PSF setup..." << endl;
			setup_PSF_convolution();
		}
		PSF_convolution_pixel_vector_wrapper<MathTypes>(false,false,qlens->fft_convolution);
		//if (!qlens->fft_convolution) {
			//p.image_surface_brightness = PSF_convolution_pixel_vector_stan(p.image_surface_brightness);
		//} else {
			//p.image_surface_brightness = PSF_convolution_pixel_vector_stan_FFT(p.image_surface_brightness);
		//}
	}
	for (int n=0; n < image_npixels; n++) {
		i = emask_pixels_i[n];
		j = emask_pixels_j[n];
		surface_brightness[i][j] = value_of(p.image_surface_brightness(n));
	}
}
template void ImagePixelGrid::find_surface_brightness_sbprofile<PlainTypes>(const bool foreground_only, const bool lensed_sources_only, const bool omit_noninverted_sources, const bool psf_convolution);
#ifdef USE_STAN
template void ImagePixelGrid::find_surface_brightness_sbprofile<VarmatTypes>(const bool foreground_only, const bool lensed_sources_only, const bool omit_noninverted_sources, const bool psf_convolution);
#endif

void ImagePixelGrid::find_point_images(const double src_x, const double src_y, vector<image<double>>& imgs, const bool use_overlap_in, const bool is_lensed, const bool verbal)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	imgs.resize(0);
	if (!is_lensed) {
		image<double> imgpt;
		imgpt.pos[0] = src_x;
		imgpt.pos[1] = src_y;
		imgpt.mag = 1.0;
		imgpt.td = 0;
		imgs.push_back(imgpt);
		return;
	}
	qlens->record_singular_points(imggrid_zfactors);
	static const int max_nimgs = 50;
	sourcept[0] = src_x;
	sourcept[1] = src_y;
	int i,j,npix,cell_i,cell_j,n_candidates = 0;
	CartesianSourcePixel* cellptr;
	bool use_overlap = use_overlap_in;
	if (use_overlap) {
		int srcgrid_nx = cartesian_srcgrid->u_N;
		int srcgrid_ny = cartesian_srcgrid->w_N;
		double xmin, ymin, xmax, ymax;
		xmin = cartesian_srcgrid->cell[0][0]->corner_pt[0][0];
		ymin = cartesian_srcgrid->cell[0][0]->corner_pt[0][1];
		xmax = cartesian_srcgrid->cell[srcgrid_nx-1][srcgrid_ny-1]->corner_pt[3][0];
		ymax = cartesian_srcgrid->cell[srcgrid_nx-1][srcgrid_ny-1]->corner_pt[3][1];
		if ((src_x < xmin) or (src_y < ymin) or (src_x > xmax) or (src_y > ymax)) use_overlap = false;
		else {
			cell_i = (int) (srcgrid_nx * ((src_x - xmin) / (xmax - xmin)));
			cell_j = (int) (srcgrid_ny * ((src_y - ymin) / (ymax - ymin)));
			cellptr = cartesian_srcgrid->cell[cell_i][cell_j];
			npix = cellptr->overlap_pixel_n.size();
		}
	}
	if (!use_overlap) {
		npix = image_npixels;
	}
	//cout << "The source cell containing this source point has " << npix << " overlapping image pixels" << endl;
	int n,imgpt_i,img_i,img_j;
	std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	//cout << "SRC: " << src_x << " " << src_y << endl;
	enum InsideTri { None, Lower, Upper } inside_tri; // inside_tri = 0 if not inside; 1 if inside lower triangle; 2 if inside upper triangle
	struct imgpt_info {
		bool confirmed;
		lensvector<double> pos;
	};
	imgpt_info image_candidates[max_nimgs]; // if there are more than 20 images, we have a truly demonic qlens on our hands
	lensvector<double> d1,d2,d3;
	double product1,product2,product3;
	// No need to parallelize this part--it is very, very fast
	//cout << "NPIX: " << npix << endl;
	lensvector<double> *vertex1,*vertex2,*vertex3,*vertex4,*vertex5;
	lensvector<double> *vertex1_srcplane,*vertex2_srcplane,*vertex3_srcplane,*vertex4_srcplane,*vertex5_srcplane;
	int *twist_type;
	for (i=0; i < npix; i++) {
		if (use_overlap) n = cellptr->overlap_pixel_n[i];
		else n = i;
		img_j = mask_pixels_j[n];
		img_i = mask_pixels_i[n];
		twist_type = &twist_status[img_i][img_j];
		inside_tri = None;

		if ((*twist_type)==0) {
			vertex1 = &corner_pts[img_i][img_j];
			vertex2 = &corner_pts[img_i][img_j+1];
			vertex3 = &corner_pts[img_i+1][img_j];
			vertex4 = &corner_pts[img_i][img_j+1];
			vertex5 = &corner_pts[img_i+1][img_j+1];
			vertex1_srcplane = &corner_sourcepts[img_i][img_j];
			vertex2_srcplane = &corner_sourcepts[img_i][img_j+1];
			vertex3_srcplane = &corner_sourcepts[img_i+1][img_j];
			vertex4_srcplane = &corner_sourcepts[img_i][img_j+1];
			vertex5_srcplane = &corner_sourcepts[img_i+1][img_j+1];
		} else if ((*twist_type)==1) {
			vertex1 = &corner_pts[img_i][img_j];
			vertex2 = &corner_pts[img_i+1][img_j];
			vertex3 = &center_pts[img_i][img_j]; // NOTE, the center point will probably not ray-trace exactly to the "twist point" where the sides cross in the source plane, but hopefully it's not too far off
			vertex4 = &corner_pts[img_i][img_j+1];
			vertex5 = &corner_pts[img_i+1][img_j+1];
			vertex1_srcplane = &corner_sourcepts[img_i][img_j];
			vertex2_srcplane = &corner_sourcepts[img_i+1][img_j];
			vertex3_srcplane = &twist_pts[img_i][img_j];
			//lensvector<double> srcpt;
			//qlens->find_sourcept((*vertex3),srcpt,0,imggrid_zfactors,imggrid_betafactors);
			//cout << "TWISTPT VS CENTER_SRCPT: " << (*vertex3_srcplane)[0] << " " << (*vertex3_srcplane)[1] << " vs " << srcpt[0] << " " << srcpt[1] << endl;
			vertex4_srcplane = &corner_sourcepts[img_i][img_j+1];
			vertex5_srcplane = &corner_sourcepts[img_i+1][img_j+1];
		} else if ((*twist_type)==2) {
			vertex1 = &corner_pts[img_i][img_j];
			vertex2 = &corner_pts[img_i][img_j+1];
			vertex3 = &center_pts[img_i][img_j]; // NOTE, the center point will probably not ray-trace exactly to the "twist point" where the sides cross in the source plane, but hopefully it's not too far off
			vertex4 = &corner_pts[img_i+1][img_j];
			vertex5 = &corner_pts[img_i+1][img_j+1];
			vertex1_srcplane = &corner_sourcepts[img_i][img_j];
			vertex2_srcplane = &corner_sourcepts[img_i][img_j+1];
			vertex3_srcplane = &twist_pts[img_i][img_j];
			//lensvector<double> srcpt;
			//qlens->find_sourcept((*vertex3),srcpt,0,imggrid_zfactors,imggrid_betafactors);
			//cout << "TWISTPT VS CENTER_SRCPT: " << (*vertex3_srcplane)[0] << " " << (*vertex3_srcplane)[1] << " vs " << srcpt[0] << " " << srcpt[1] << endl;
			vertex4_srcplane = &corner_sourcepts[img_i+1][img_j];
			vertex5_srcplane = &corner_sourcepts[img_i+1][img_j+1];
		}

		d1[0] = src_x - (*vertex1_srcplane)[0];
		d1[1] = src_y - (*vertex1_srcplane)[1];
		d2[0] = src_x - (*vertex2_srcplane)[0];
		d2[1] = src_y - (*vertex2_srcplane)[1];
		d3[0] = src_x - (*vertex3_srcplane)[0];
		d3[1] = src_y - (*vertex3_srcplane)[1];
		product1 = d1[0]*d2[1] - d1[1]*d2[0];
		product2 = d3[0]*d1[1] - d3[1]*d1[0];
		product3 = d2[0]*d3[1] - d2[1]*d3[0];
		if ((product1 > 0) and (product2 > 0) and (product3 > 0)) inside_tri = Lower;
		else if ((product1 < 0) and (product2 < 0) and (product3 < 0)) inside_tri = Lower;
		else {
			if (inside_tri==None) {
				d1[0] = src_x - (*vertex5_srcplane)[0];
				d1[1] = src_y - (*vertex5_srcplane)[1];
				d2[0] = src_x - (*vertex4_srcplane)[0];
				d2[1] = src_y - (*vertex4_srcplane)[1];
				product1 = d1[0]*d2[1] - d1[1]*d2[0];
				product2 = d3[0]*d1[1] - d3[1]*d1[0];
				product3 = d2[0]*d3[1] - d2[1]*d3[0];
				if ((product1 > 0) and (product2 > 0) and (product3 > 0)) inside_tri = Upper;
				if ((product1 < 0) and (product2 < 0) and (product3 < 0)) inside_tri = Upper;
			}
		}
		double kap;
		lensvector<double> side1, side2;
		lensvector<double> side1_srcplane, side2_srcplane;
		if ((inside_tri != None) and (*twist_type!=0)) warn(qlens->newton_warnings,"possible image identified in cell with nonzero twist status (%g,%g); status %i",center_pts[img_i][img_j][0],center_pts[img_i][img_j][1],*twist_type);
		if (inside_tri != None) {
			//cout << "i=" << img_i << " j=" << img_j << " twiststat=" << *twist_type << endl;
			imgpt_i = n_candidates++;
			//pixels_with_imgpts[imgpt_i].img_i = img_i;
			//pixels_with_imgpts[imgpt_i].img_j = img_j;
			//pixels_with_imgpts[imgpt_i].upper_tri = (inside_tri==Upper) ? true : false;
			image_candidates[imgpt_i].confirmed = false;
			if (inside_tri==Lower) {
				image_candidates[imgpt_i].pos[0] = ((*vertex1)[0] + (*vertex2)[0] + (*vertex3)[0])/3;
				image_candidates[imgpt_i].pos[1] = ((*vertex1)[1] + (*vertex2)[1] + (*vertex3)[1])/3;
				// For now, just don't bother with central image points when using a pixel image--it only causes trouble in the form of duplicate images
				//if (!qlens->include_central_image) 
				if (*twist_type==0) { // central images are unlikely to be highly magnified near a critical curve, so twisting is unlikely in that case
					kap = qlens->kappa<double>(image_candidates[imgpt_i].pos,imggrid_zfactors,imggrid_betafactors);
					side1 = corner_pts[img_i][img_j+1] - corner_pts[img_i][img_j];
					side2 = corner_pts[img_i][img_j] - corner_pts[img_i+1][img_j+1];
					side1_srcplane = corner_sourcepts[img_i][img_j+1] - corner_sourcepts[img_i][img_j];
					side2_srcplane = corner_sourcepts[img_i][img_j] - corner_sourcepts[img_i+1][img_j+1];
					product1 = side1 ^ side2;
					product2 = side1_srcplane ^ side2_srcplane;
					//if (product1*product2 > 0) cout << "Parity > 0" << endl;
				}

				//cout << "#Lower triangle: " << endl;
				//cout << corner_sourcepts[img_i][img_j][0] << " " << corner_sourcepts[img_i][img_j][1] << endl;
				//cout << corner_sourcepts[img_i][img_j+1][0] << " " << corner_sourcepts[img_i][img_j+1][1] << endl;
				//cout << corner_sourcepts[img_i+1][img_j][0] << " " << corner_sourcepts[img_i+1][img_j][1] << endl;
				//cout << corner_sourcepts[img_i][img_j][0] << " " << corner_sourcepts[img_i][img_j][1] << endl;
				//cout << endl;
			} else {
				image_candidates[imgpt_i].pos[0] = ((*vertex3)[0] + (*vertex4)[0] + (*vertex5)[0])/3;
				image_candidates[imgpt_i].pos[1] = ((*vertex3)[1] + (*vertex4)[1] + (*vertex5)[1])/3;
				//if (!qlens->include_central_image) 
				if (*twist_type==0) { // central images are unlikely to be highly magnified near a critical curve, so twisting is unlikely in that case
					kap = qlens->kappa<double>(image_candidates[imgpt_i].pos,imggrid_zfactors,imggrid_betafactors);
					//if (kap > 1) cout << "CENTRAL IMAGE? candidate pos=" << image_candidates[imgpt_i].pos[0] << "," << image_candidates[imgpt_i].pos[1] << endl;
					side1 = corner_pts[img_i+1][img_j+1] - corner_pts[img_i+1][img_j];
					side2 = corner_pts[img_i][img_j+1] - corner_pts[img_i+1][img_j+1];
					side1_srcplane = corner_sourcepts[img_i+1][img_j+1] - corner_sourcepts[img_i+1][img_j];
					side2_srcplane = corner_sourcepts[img_i][img_j+1] - corner_sourcepts[img_i+1][img_j+1];
					product1 = side1 ^ side2;
					product2 = side1_srcplane ^ side2_srcplane;
					//if (product1*product2 > 0) cout << "Parity > 0" << endl;
				}


				//cout << "#Upper triangle: " << endl;
				//cout << corner_sourcepts[img_i+1][img_j+1][0] << " " << corner_sourcepts[img_i+1][img_j+1][1] << endl;
				//cout << corner_sourcepts[img_i][img_j+1][0] << " " << corner_sourcepts[img_i][img_j+1][1] << endl;
				//cout << corner_sourcepts[img_i+1][img_j][0] << " " << corner_sourcepts[img_i+1][img_j][1] << endl;
				//cout << corner_sourcepts[img_i+1][img_j+1][0] << " " << corner_sourcepts[img_i+1][img_j+1][1] << endl;
			}
			//if ((!qlens->include_central_image) and (kap > 1) and (product1*product2 > 0)) n_candidates--;
			if ((*twist_type==0) and (kap > 1) and (product1*product2 > 0)) n_candidates--; // exclude central image candidate by default
		}

		//cout << "Pixel " << img_i << "," << img_j << endl;
		if (n_candidates == max_nimgs) {
			warn("exceeded max number of images in ImagePixelGrid::find_point_images");
			break;
		}
	}

	LensProfile *lptr;
	bool singular;
	for (i=0; i < qlens->nlens; i++) {
		lptr = qlens->lens_list[i];
		if (lptr->get_lenstype()==SHEAR) continue;
		singular = false;
		if ((lptr->get_lenstype() == dpie_LENS) and (lptr->core_present()==false)) singular = true;
		else if ((lptr->get_lenstype() == sple_LENS) and (lptr->get_inner_logslope() <= -1) and (lptr->core_present()==false)) singular = true;
		else if (lptr->get_lenstype() == PTMASS) singular = true;
		if (!singular) {
			lptr->get_center_coords(image_candidates[n_candidates].pos);
			image_candidates[n_candidates].pos[0] += 1e-6;
			image_candidates[n_candidates].pos[1] += 1e-6;
			n_candidates++;
		}
	}
	image_pos_accuracy = qlens->image_pos_accuracy;
	//if (image_pos_accuracy < 0) image_pos_accuracy = 1e-3; // in case the data pixel size has not been set
	if ((qlens->mpi_id==0) and (verbal)) cout << "Found " << n_candidates << " candidate images" << endl;
	//for (i=0; i < n_candidates; i++) {
		//cout << "candidate " << i << ": " << image_candidates[i].pos[0] << " " << image_candidates[i].pos[1] << endl;
	//}
	//cout << endl;
	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		double td_factor;
		if (qlens->include_time_delays) td_factor = qlens->cosmo->time_delay_factor_arcsec(qlens->lens_redshift,qlens->reference_source_redshift);
		image<double> imgpt;
		double mag;
		#pragma omp for private(i) schedule(static)
		for (i=0; i < n_candidates; i++) {
			//cout << "Trying candidate " << i << ": " << image_candidates[i].pos[0] << " " << image_candidates[i].pos[1] << endl;
			if ((qlens->skip_newtons_method) or (run_newton(image_candidates[i].pos,mag,thread)==true)) {
			//{
				imgpt.pos = image_candidates[i].pos;
				//imgpt.mag = qlens->magnification<double>(image_candidates[i].pos,0,imggrid_zfactors,imggrid_betafactors);
				imgpt.mag = mag;
				imgpt.flux = -1e30;
				//cout << "FOUND IMAGE AT: " << imgpt.pos[0] << " " << imgpt.pos[1] << endl;
				//if ((qlens->mpi_id==0) and (verbal)) cout << "FOUND IMAGE AT: " << imgpt.pos[0] << " " << imgpt.pos[1] << endl;
				if (qlens->include_time_delays) {
					double potential = qlens->potential<double>(imgpt.pos,imggrid_zfactors,imggrid_betafactors);
					imgpt.td = 0.5*(SQR(imgpt.pos[0]-sourcept[0])+SQR(imgpt.pos[1]-sourcept[1])) - potential; // the dimensionless version; it will be converted to days by the QLens class
					imgpt.td *= td_factor;
				} else {
					imgpt.td = 0;
				}
				imgpt.parity = sign(imgpt.mag);
				#pragma omp critical
				{
					imgs.push_back(imgpt);
				}
			}
		}
	}
	bool redundancy;
	vector<image<double>>::iterator it, it2;
	double sep, pixel_size;
	pixel_size = dmax(pixel_xlength,pixel_ylength);
	//int oldsize = imgs.size();
	int i_center, j_center;
	for (it = imgs.begin(); it != imgs.end(); it++) {
		// check to see if the image is inside the pixel grid; if not, discard it since it can't fit any image in the data
		i_center = (it->pos[0] - xmin)/pixel_xlength;
		j_center = (it->pos[1] - ymin)/pixel_ylength;
		//cout << "icenter=" << i_center << " jcenter=" << j_center << endl;
		if ((i_center < 0) or (i_center >= x_N) or (j_center < 0) or (j_center >= y_N)) {
			warn("image point (%g,%g) lies outside image pixel grid; will eliminate from image set",it->pos[0],it->pos[1]);
			if (it == imgs.end()-1) {
				imgs.pop_back();
				break;
			} else {
				imgs.erase(it);
				it--;
			}
		}
	}
	if (imgs.size() > 1) {
		for (it = imgs.begin()+1; it != imgs.end(); it++) {
			redundancy = false;
			for (it2 = imgs.begin(); it2 != it; it2++) {
				sep = sqrt(SQR(it->pos[0] - it2->pos[0]) + SQR(it->pos[1] - it2->pos[1]));
				if (sep < pixel_size)
				{
					redundancy = true;
					warn(qlens->newton_warnings,"rejecting probable duplicate image (imgsep=%g,threshold=%g): src (%g,%g), image (%g,%g), mag %g",sep,pixel_size,sourcept[0],sourcept[1],it->pos[0],it->pos[1],it->mag);
					break;
				}
			}
			if (redundancy) {
				if (it == imgs.end()-1) {
					imgs.pop_back();
					break;
				} else {
					imgs.erase(it);
					it--;
				}
			}
		}
	}
	int max_expected = 4;
	if (qlens->include_central_image) max_expected++;
	if (imgs.size() > max_expected) warn(qlens->newton_warnings,"more than four images after trimming redundancies");

	if ((qlens->mpi_id==0) and (verbal)) {
		cout << "# images found: " << imgs.size() << endl;
		for (i=0; i < imgs.size(); i++) {
			cout << imgs[i].pos[0] << " " << imgs[i].pos[1] << " " << imgs[i].mag;
			if (qlens->include_time_delays) cout << " " << imgs[i].td;
			cout << endl;
		}
	}
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for point image finding: "  << wtime.count() << endl;
	}
}

void ImagePixelGrid::generate_point_images(const vector<image<double>>& imgs, double* ptimage_surface_brightness, const bool use_img_fluxes, const double srcflux, const int img_num)
{
	int nx_half, ny_half;
	double normfac, sigx, sigy;
	int nsplit = qlens->ptimg_nsplit;
	if (psf->use_input_psf_matrix) {
		if (psf->psf_matrix == NULL) return;
		nx_half = psf->psf_npixels_x/2;
		ny_half = psf->psf_npixels_y/2;
	} else {
		double sigma_fraction = sqrt(-2*log(qlens->psf_ptsrc_threshold));
		double nx_half_dec, ny_half_dec;
		sigx = psf->psf_params.psf_width_x;
		sigy = psf->psf_params.psf_width_y;
		nx_half_dec = sigma_fraction*sigx/pixel_xlength;
		ny_half_dec = sigma_fraction*sigy/pixel_ylength;
		//cout << "sigma_frac=" << sigma_fraction << endl;
		//cout << "nxhalfd=" << nx_half_dec << " nyhalfd=" << ny_half_dec << endl;
		nx_half = ((int) nx_half_dec);
		ny_half = ((int) ny_half_dec);
		if ((nx_half_dec - nx_half) > 0.5) nx_half++;
		if ((ny_half_dec - ny_half) > 0.5) ny_half++;
		normfac = 1.0/(M_2PI*sigx*sigy);
	}
	//cout << "nxhalf=" << nx_half << " nyhalf=" << ny_half << endl;
	//int nx = 2*nx_half+1;
	//int ny = 2*ny_half+1;
	//cout << "normfac=" << normfac << endl;

	int n,i,j,ii,jj,img_i;
	int i_center, j_center, imin, imax, jmin, jmax, inn, jnn, nmax;
	//cout << "nsplit=" << nsplit << " nsubpix=" << nsubpix << endl;
	double sb, fluxfac, x, y, x0, y0, u0, w0;
	std::chrono::steady_clock::time_point wtime0;
	std::chrono::duration<double> wtime;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	int img_i0, img_if;
	if (img_num==-1) {
		img_i0 = 0;
		img_if = imgs.size();
	} else {
		if (img_num >= imgs.size()) die("img_num does not exist; cannot generate point image");
		img_i0 = img_num;
		img_if = img_num+1;
	}
	int idx;
	for (idx=0; idx < image_npixels; idx++) ptimage_surface_brightness[idx] = 0;
	for (img_i=img_i0; img_i < img_if; img_i++) {
		//cout << "Generating point image " << imgs[img_i].pos[0] << " " << imgs[img_i].pos[1] << endl;
		if ((use_img_fluxes) and (imgs[img_i].flux != -1e30)) {
			//cout << "USING IMAGE FLUX: " << imgs[img_i].flux << endl;
			fluxfac = imgs[img_i].flux;
		} else {
			//cout << "NOT USING IMAGE FLUX! " << imgs[img_i].flux << " srcflux=" << srcflux << endl;
			if (srcflux >= 0) fluxfac = srcflux*abs(imgs[img_i].mag);
			else fluxfac = 1.0; // if srcflux < 0, then normalize to 1 (this will be multiplied by a flux amplitude afterwards)
		}
		//cout << "flux=" << flux << endl;
		x0 = imgs[img_i].pos[0];
		y0 = imgs[img_i].pos[1];
		i_center = (x0 - xmin)/pixel_xlength;
		j_center = (y0 - ymin)/pixel_ylength;
		//cout << "icenter=" << i_center << " jcenter=" << j_center << endl;
		if ((i_center < 0) or (i_center >= x_N) or (j_center < 0) or (j_center >= y_N)) {
			warn("image point (%g,%g) lies outside image pixel grid",x0,y0);
			continue;
		}
		imin = i_center - nx_half;
		imax = i_center + nx_half;
		jmin = j_center - ny_half;
		jmax = j_center + ny_half;
		if (imin < 0) imin = 0;
		if (imax >= x_N) imax = x_N-1;
		if (jmin < 0) jmin = 0;
		if (jmax >= y_N) jmax = y_N-1;
		inn = imax-imin+1;
		jnn = jmax-jmin+1;
		nmax = inn*jnn;
		//cout << "imin=" << imin << " imax=" << imax << " jmin=" << jmin << " jmax=" << jmax << endl;
		//cout << "xmin=" << corner_pts[imin][jmin][0] << " xmax=" << corner_pts[imax][jmin][0] << endl;
		//cout << "ymin=" << corner_pts[imin][jmin][1] << " ymax=" << corner_pts[imin][jmax][1] << endl;
		//cout << "x0(pixel_center)=" << center_pts[i_center][j_center][0] << "y0(pixel_center)=" << center_pts[i_center][j_center][1] << endl;
		//double norm = 0;
		//for (i=imin; i <= imax; i++) {
			//for (j=jmin; j <= jmax; j++) {
				//sb += flux*exp(-(SQR((center_pts[i][j][0]-x0)/sigx) + SQR((center_pts[i][j][1]-y0)/sigy))/2);
				//surface_brightness[i][j] += sb;
				//norm += sb;
			//}
		//}

		//double tot=0;
		#pragma omp parallel for private(n,i,j,ii,jj,sb,u0,w0,x,y) schedule(static)
		for (n=0; n < nmax; n++) {
			j = jmin + (n / inn);
			i = imin + (n % inn);
			sb=0;
			for (ii=0; ii < nsplit; ii++) {
				u0 = ((double) (1+2*ii))/(2*nsplit);
				x = (1-u0)*corner_pts[i][j][0] + u0*corner_pts[i+1][j][0];
				for (jj=0; jj < nsplit; jj++) {
					w0 = ((double) (1+2*jj))/(2*nsplit);
					y = (1-w0)*corner_pts[i][j][1] + w0*corner_pts[i][j+1][1];
					if (psf->use_input_psf_matrix) {
						sb += fluxfac*psf->interpolate_PSF_matrix(x-x0,y-y0,qlens->psf_supersampling);
					} else {
						sb += fluxfac*normfac*exp(-(SQR((x-x0)/sigx) + SQR((y-y0)/sigy))/2);
					}
				}
			}
			sb /= n_subpix_per_pixel;
			//cout << "sb=" << sb << endl;
			if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
				ptimage_surface_brightness[pixel_index[i][j]] += sb;
				//tot += sb;
			}
		}
		//sigx = qlens->psf_width_x;
		//sigy = qlens->psf_width_y;
		//normfac = 1.0/(M_2PI*sigx*sigy);
		//double sbcheck0 = qlens->psf_matrix[nx_half][ny_half]/(pixel_xlength*pixel_ylength);
		//double sbcheck = qlens->interpolate_PSF_matrix(0,0)/(pixel_xlength*pixel_ylength);
		//double sbcheck2 = normfac;
		////cout << "normfac_inv: " << (1.0/normfac) << endl;
		//cout << "SBCHECKS: " << nx_half << " " << ny_half << " " << sbcheck0 << " " << sbcheck << " " << sbcheck2 << endl;

		//cout << "TOT=" << tot << endl;
		//cout << "Added point image" << endl;
		//cout << "area = " << (nx*ny*pixel_xlength*pixel_ylength) << endl;
	}
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for adding point images: "  << wtime.count() << endl;
	}
}

void ImagePixelGrid::add_point_images(VectorXd& ptimage_surface_brightness, const int npix)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	int i,j,indx;
	for (indx=0; indx < npix; indx++) {
		i = mask_pixels_i[indx];
		j = mask_pixels_j[indx];
		surface_brightness[i][j] += ptimage_surface_brightness[indx];
	}
}

void ImagePixelGrid::generate_and_add_point_images(const vector<image<double>>& imgs, const bool include_imgfluxes, const double srcflux)
{
	Eigen::VectorXd ptimgs(image_npixels);
	generate_point_images(imgs,ptimgs.data(),include_imgfluxes,srcflux);
	add_point_images(ptimgs,image_npixels);
}	

// 2-d Newton's Method w/ backtracking routines
// These functions are redundant with the same ones in the GridCell class, and I *HATE* redundancies like this. But for now, it's
// easiest to just copy it over. Later you can put NewtonsMethod in a separate class and have it inherited by both GridCell and ImagePixelGrid.

const int ImagePixelGrid::max_iterations = 200;
const int ImagePixelGrid::max_step_length = 100;

inline void ImagePixelGrid::SolveLinearEqs(lensmatrix<double>& a, lensvector<double>& b)
{
	double det, temp;
	det = determinant(a);
	temp = (-a[1][0]*b[1]+a[1][1]*b[0]) / det;
	b[1] = (-a[0][1]*b[0]+a[0][0]*b[1]) / det;
	b[0] = temp;
}

inline double ImagePixelGrid::max_component(const lensvector<double>& x) { return dmax(fabs(x[0]),fabs(x[1])); }

bool ImagePixelGrid::run_newton(lensvector<double>& xroot, double& mag, const int& thread)
{
	lensvector<double> xroot_initial = xroot;
	if ((xroot[0]==0) and (xroot[1]==0)) { xroot[0] = xroot[1] = 5e-1*qlens->cc_rmin; }	// Avoiding singularity at center
	if (NewtonsMethod(xroot, newton_check[thread], thread)==false) {
		warn(qlens->newton_warnings,"Newton's method failed for source (%g,%g), initial point (%g,%g)",sourcept[0],sourcept[1],xroot_initial[0],xroot_initial[1]);
		return false;
	}
	//if (qlens->reject_images_found_outside_cell) {
		//if (test_if_inside_cell(xroot,thread)==false) {
			////warn(qlens->warnings,"Rejecting image found outside cell for source (%g,%g), level %i, cell center (%g,%g)",sourcept[0],sourcept[1],level,center_imgplane[0],center_imgplane[1],xroot[0],xroot[1]);
			//return false;
		//}
	//}

	lensvector<double> lens_eq_f;
	qlens->lens_equation<double>(xroot,sourcept,lens_eq_f,thread,imggrid_zfactors,imggrid_betafactors);
	//double lenseq_mag = sqrt(SQR(lens_eq_f[0]) + SQR(lens_eq_f[1]));
	//double tryacc = image_pos_accuracy / sqrt(abs(qlens->magnification<double>(xroot,thread,zfactor)));
	//cout << lenseq_mag << " " << tryacc << " " << sqrt(abs(qlens->magnification<double>(xroot,thread,zfactor))) << endl;
	if (newton_check[thread]==true) { warn(qlens->newton_warnings, "false image--converged to local minimum"); return false; }
	if (qlens->n_singular_points > 0) {
		//cout << "singular point: " << qlens->singular_pts[0][0] << " " << qlens->singular_pts[0][1] << endl;
		double singular_pt_accuracy = 2*image_pos_accuracy;
		for (int i=0; i < qlens->n_singular_points; i++) {
			if ((abs(xroot[0]-qlens->singular_pts[i][0]) < singular_pt_accuracy) and (abs(xroot[1]-qlens->singular_pts[i][1]) < singular_pt_accuracy)) {
				warn(qlens->newton_warnings,"Newton's method converged to singular point (%g,%g) for source (%g,%g)",qlens->singular_pts[i][0],qlens->singular_pts[i][1],sourcept[0],sourcept[1]);
				return false;
			}
		}
	}
	if (((xroot[0]==xroot_initial[0]) and (xroot_initial[0] != 0)) and ((xroot[1]==xroot_initial[1]) and (xroot_initial[1] != 0)))
		warn(qlens->newton_warnings, "Newton's method returned initial point");
	mag = qlens->magnification<double>(xroot,thread,imggrid_zfactors,imggrid_betafactors);
	if ((abs(lens_eq_f[0]) > 1000*image_pos_accuracy) and (abs(lens_eq_f[1]) > 1000*image_pos_accuracy) and (abs(mag) < 1e-3)) {
		if (qlens->newton_warnings==true) {
			warn(qlens->newton_warnings,"Newton's method may have found false root (%g,%g) (within 1000*accuracy) for source (%g,%g), cell center (%g,%g), mag %g",xroot[0],xroot[1],sourcept[0],sourcept[1],xroot_initial[0],xroot_initial[1],xroot[0],xroot[1],mag);
		}
	}
	if ((abs(mag) > qlens->newton_magnification_threshold) or (mag*0.0 != 0.0)) {
		if (qlens->reject_himag_images) {
			if ((qlens->mpi_id==0) and (qlens->newton_warnings)) {
				cout << "*WARNING*: Rejecting image that exceeds imgsrch_mag_threshold (" << abs(mag) << "), src=(" << sourcept[0] << "," << sourcept[1] << "), x=(" << xroot[0] << "," << xroot[1] << ")      " << endl;
				if (qlens->use_ansi_characters) {
					cout << "                                                                                                                            " << endl;
					cout << "\033[2A";
				}
			}
			return false;
		} else {
			if ((qlens->mpi_id==0) and (qlens->warnings)) {
				cout << "*WARNING*: Image exceeds imgsrch_mag_threshold (" << abs(mag) << "); src=(" << sourcept[0] << "," << sourcept[1] << "), x=(" << xroot[0] << "," << xroot[1] << ")        " << endl;
				if (qlens->use_ansi_characters) {
					cout << "                                                                                                                            " << endl;
					cout << "\033[2A";
				}
			}
		}
	}
	if ((qlens->include_central_image==false) and (mag > 0) and (qlens->kappa<double>(xroot,imggrid_zfactors,imggrid_betafactors) > 1)) return false; // discard central image if not desired

	/*
	bool status = true;
	//#pragma omp critical
	//{
			images[nfound].pos[0] = xroot[0];
			images[nfound].pos[1] = xroot[1];
			images[nfound].mag = qlens->magnification<double>(xroot,0,imggrid_zfactors,imggrid_betafactors);
			if (qlens->include_time_delays) {
				double potential = qlens->potential(xroot,imggrid_zfactors,imggrid_betafactors);
				images[nfound].td = 0.5*(SQR(xroot[0]-sourcept[0])+SQR(xroot[1]-sourcept[1])) - potential; // the dimensionless version; it will be converted to days by the QLens class
			} else {
				images[nfound].td = 0;
			}
			images[nfound].parity = sign(images[nfound].mag);

			if (qlens->use_cc_spline) {
				bool found_pos=false, found_neg=false;
				double rroot, thetaroot, cr0, cr1;
				rroot = norm(xroot[0]-qlens->grid_xcenter,xroot[1]-qlens->grid_ycenter);
				thetaroot = angle(xroot[0]-qlens->grid_xcenter,xroot[1]-qlens->grid_ycenter);
				cr0 = qlens->ccspline[0].splint(thetaroot);
				cr1 = qlens->ccspline[1].splint(thetaroot);

				int expected_parity;
				if (rroot < cr0) {
					nfound_max++; expected_parity = 1;
				} else if (rroot > cr1) {
					nfound_pos++; expected_parity = 1;
				} else {
					nfound_neg++; expected_parity = -1;
				}

				if (images[nfound].parity != expected_parity)
					warn(qlens->warnings, "wrong parity found for image from source (%g, %g)", sourcept[0], sourcept[1]);
				
				if ((qlens->system_type==Single) and (nfound_pos >= 1)) finished_search = true;
				else
				{
					if ((qlens->system_type==Double) and (nfound_pos >= 1)) found_pos = true;
					else if (((qlens->system_type==Quad) or (qlens->system_type==Cusp)) and (nfound_pos >= 2)) found_pos = true;

					if (((qlens->system_type==Double) or (qlens->system_type==Cusp)) and (nfound_neg >= 1)) found_neg = true;
					else if ((qlens->system_type==Quad) and (nfound_neg >= 2)) found_neg = true;

					if ((found_pos) and (found_neg)) finished_search = true;
				}
			}

			nfound++;
		}
		*/
	//}
	return true;
}

bool ImagePixelGrid::NewtonsMethod(lensvector<double>& x, bool &check, const int& thread)
{
	check = false;
	lensvector<double> g, p, xold;
	lensmatrix<double> fjac;

	qlens->lens_equation<double>(x, sourcept, fvec[thread], thread, imggrid_zfactors, imggrid_betafactors);
	double f = 0.5*fvec[thread].sqrnorm();
	if (max_component(fvec[thread]) < 0.01*image_pos_accuracy)
		return true; 

	double fold, stpmax, temp, test;
	stpmax = max_step_length * dmax(x.norm(), 2.0); 
	for (int its=0; its < max_iterations; its++) {
		qlens->hessian<double>(x[0],x[1],fjac,thread,imggrid_zfactors,imggrid_betafactors);
		fjac[0][0] = -1 + fjac[0][0];
		fjac[1][1] = -1 + fjac[1][1];
		g[0] = fjac[0][0] * fvec[thread][0] + fjac[0][1]*fvec[thread][1];
		g[1] = fjac[1][0] * fvec[thread][0] + fjac[1][1]*fvec[thread][1];
		xold[0] = x[0];
		xold[1] = x[1];
		fold = f; 
		p[0] = -fvec[thread][0];
		p[1] = -fvec[thread][1];
		SolveLinearEqs(fjac, p);
		if (LineSearch(xold, fold, g, p, x, f, stpmax, check, thread)==false)
			return false;
		if ((x[0] > 1e3*qlens->cc_rmax) or (x[1] > 1e3*qlens->cc_rmax)) {
			warn(qlens->newton_warnings, "Newton blew up!");
			return false;
		}
		/*
		qlens->lens_equation<double>(x, sourcept, fvec[thread], thread, zfactor);
		double magfac = sqrt(abs(qlens->magnification<double>(x,thread,zfactor)));
		double tryacc;
		lensvector<double> dx = x - xold;
		double dxnorm = dx.norm();
		dx[0] /= dxnorm;
		dx[1] /= dxnorm;
		lensmatrix<double> magmat;
		lensvector<double> bb;
		qlens->sourcept_jacobian(x,bb,magmat,thread,zfactor);
		bb = magmat*dx;
		lensvector<double> dy;
		dy[1] = -dx[0];
		dy[0] = dx[1];
		lensvector<double> cc;
		tryacc = image_pos_accuracy * bb.norm();
		*/
		//if (max_component(fvec[thread]) < 4*tryacc) {

		// Maybe someday revisit this and see if you can make it more robust. As it is, it's
		// frustrating that image_pos_accuracy has no simple interpretation, and occasionally
		// spurious images close to critical curves do are found.
		if (max_component(fvec[thread]) < image_pos_accuracy) {
			check = false; 
			return true; 
		}
		if (check) {
			double den = dmax(f, 1.0); 
			temp = fabs(g[0]) * dmax(fabs(x[0]), 1.0)/den; 
			test = fabs(g[1]) * dmax(fabs(x[1]), 1.0)/den; 
			check = (dmax(test,temp) < image_pos_accuracy); 
			return true; 
		}
		test = (fabs(x[0] - xold[0])) / dmax(fabs(x[0]), 1.0); 
		temp = (fabs(x[1] - xold[1])) / dmax(fabs(x[1]), 1.0); 
		if (temp > test) test = temp; 
		if (test < image_pos_accuracy) return true; 
	}

	return false;
}

bool ImagePixelGrid::LineSearch(lensvector<double>& xold, double fold, lensvector<double>& g, lensvector<double>& p, lensvector<double>& x,
	double& f, double stpmax, bool &check, const int& thread)
{
	const double alpha = 1.0e-4;	// Ensures sufficient decrease in function value (see NR Ch. 9.7)

	double a, alam, alam2, alamin, b, disc, f2, rhs1, rhs2, slope, mag, temp, test, tmplam;

	check = false;
	mag = p.norm();
	if (mag > stpmax) {
		double fac = stpmax / mag;
		p[0] *= fac;
		p[1] *= fac;
	}
	slope = g[0]*p[0] + g[1]*p[1];
	if (slope >= 0.0) die("Roundoff problem during line search (g=(%g,%g), p=(%g,%g))",g[0],g[1],p[0],p[1]); 
	test = fabs(p[0]) / dmax(fabs(xold[0]), 1.0); 
	temp = fabs(p[1]) / dmax(fabs(xold[1]), 1.0); 
	alamin = image_pos_accuracy / dmax(temp,test); 
	alam = 1.0; 
	while (true)
	{
		x[0] = xold[0] + alam*p[0];
		x[1] = xold[1] + alam*p[1];
		if ((fabs(x[0]) < 1e6*qlens->cc_rmax) and (fabs(x[1]) < 1e6*qlens->cc_rmax))
			;
		else {
			warn(qlens->newton_warnings, "Newton blew up!");
			return false;
		}
		qlens->lens_equation<double>(x, sourcept, fvec[thread], thread, imggrid_zfactors, imggrid_betafactors);
		f = 0.5 * fvec[thread].sqrnorm();
		if (alam < alamin) {
			x[0] = xold[0];
			x[1] = xold[1];
			check = true; 
			return true; 
		} else if (f <= fold + alpha*alam*slope)
			return true; 
		else
		{
			if (alam == 1.0)
				tmplam = -slope / (2.0*(f-fold-slope));
			else
			{
				rhs1 = f - fold - alam*slope;
				rhs2 = f2 - fold - alam2*slope;
				a = (rhs1/(alam*alam) - rhs2/(alam2*alam2)) / (alam-alam2);
				b = (-alam2*rhs1/(alam*alam) + alam*rhs2/(alam2*alam2)) / (alam-alam2);
				if (a == 0.0) tmplam = -slope / (2.0*b);
				else
				{
					disc = b*b - 3.0*a*slope;
					if (disc < 0.0) tmplam = 0.5*alam;
					else if (b <= 0.0) tmplam = (-b + sqrt(disc)) / (3.0*a);
					else tmplam = -slope / (b + sqrt(disc));
				}
				if (tmplam > 0.5*alam)
					tmplam = 0.5*alam;
			}
		}
		alam2 = alam;
		f2 = f;
		alam = dmax(tmplam, 0.1*alam);
	}
}

/**************** Functions in class ImagePixelGrid for pixel mapping (img <-> src), PSF convolution and inversion *******************/

bool ImagePixelGrid::assign_pixel_mappings(const bool potential_perturbations, const bool verbal)
{
	int i, j, ii, jj, subcell_index, nsubpix, image_pixel_index, image_subpixel_index;
	CartesianSourceGrid *cartesian_srcgrid = cartesian_srcgrid;

	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	count_MGE_amplitudes(n_mge_sets,n_mge_amps);
	if (n_mge_sets > 0) create_MGE_regularization_matrices();
	bool map_all_imgpixels = true; // Now, we include all image pixels in the mask, regardless of whether any source pixels map to them (since regularization matrix will include all image pixels anyway)
	//bool map_all_imgpixels = (n_mge_amps > 0) ? true : false;
	//if (image_pixel_grid->n_pixsrc_to_include_in_Lmatrix > 1) map_all_imgpixels = true; // hack to enforce that image pixel indices are the same for all grids being included
	int tot_npixels_count;
	source_npixels = 0;
	ImagePixelGrid *imggrid;
	for (int imggrid_i_inv=0; imggrid_i_inv < n_pixsrc_to_include_in_Lmatrix; imggrid_i_inv++) {
		imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[imggrid_i_inv]];
		nsubpix = imggrid->n_subpix_per_pixel;
		image_pixel_index=0;
		image_subpixel_index=0;
		if (qlens->source_fit_mode==Delaunay_Source) {
			imggrid->assign_image_mapping_flags(true,potential_perturbations,map_all_imgpixels);
			if (imggrid->delaunay_srcgrid != NULL) { 
				 // note, the following line *adds* to the source_npixels that are already there (if more than one source included in inversion)
				source_npixels = imggrid->delaunay_srcgrid->assign_active_indices_and_count_source_pixels(source_npixels,qlens->activate_unmapped_source_pixels);
			}
		} else {
			tot_npixels_count = imggrid->cartesian_srcgrid->assign_indices_and_count_levels();
			if ((qlens->mpi_id==0) and (qlens->adaptive_subgrid) and (verbal==true)) cout << "Number of source cells: " << tot_npixels_count << endl;
			imggrid->assign_image_mapping_flags(false,potential_perturbations,map_all_imgpixels);

			imggrid->cartesian_srcgrid->regrid = false;
			if (qlens->nlens != 0) {
				source_npixels = imggrid->cartesian_srcgrid->assign_active_indices_and_count_source_pixels(source_npixels,qlens->activate_unmapped_source_pixels,qlens->exclude_source_pixels_beyond_fit_window); // note, this *adds* to the source_npixels that are already there
				if (source_npixels==0) { warn("number of source pixels cannot be zero"); return false; }
			}
		}
		if (qlens->psf_supersampling) {
			image_n_subpixels = imggrid->image_n_subpixels;
			//int nsub=0;
			//for (j=0; j < imggrid->y_N; j++) {
				//for (i=0; i < imggrid->x_N; i++) {
					//if (imggrid->pixel_in_mask[i][j]) {
						////if (imggrid->nsplits[i][j] != default_imgpixel_nsplit) die("nsplit has to be the same for all pixels to use supersampling (pixel (%i,%i), nsplits: %i vs %i)",i,j,imggrid->nsplits[i][j],default_imgpixel_nsplit);
						//nsub += nsubpix;
					//}
				//}
			//}
			//image_n_subpixels = nsub;
		}

		if (n_active_pixels != image_npixels) warn("n_active_pixels does not equal image_npixels! some pixels are not mapped (%i vs %i)",n_active_pixels,image_npixels);
		
		if ((imggrid_i_inv != 0) and (image_npixels != imggrid->image_npixels)) die("secondary imggrid does not have same number of active image pixels as primary imggrid");
	}
	n_amps = source_npixels;
	if (potential_perturbations) {
		lensgrid_npixels = lensgrid->assign_active_indices_and_count_lens_pixels(true);
		if (lensgrid_npixels==0) { warn("number of pixlens pixels cannot be zero if potential perturbations are turned on"); return false; }
		n_amps += lensgrid_npixels;
	} else {
		lensgrid_npixels = 0;
	}
	n_amps += n_mge_amps;
	source_and_lens_n_amps = source_npixels + lensgrid_npixels + n_mge_amps;
	Lmatrix_n_amps = source_and_lens_n_amps; // store the number of source/potential amplitudes for each image pixel grid; useful later for initializing/cleaning up FFT convolution arrays
	Lmatrix_pot_npixels = lensgrid_npixels; // store the number of lensgrid pixels for each image pixel grid; useful later for cleaning up FFT convolution arrays

	if (qlens->include_imgfluxes_in_inversion) {
		for (int i=0; i < qlens->n_ptsrc; i++) {
			n_amps += qlens->ptsrc_list[i]->images.size(); // in this case, source amplitudes include point image amplitudes as well as pixel values
		}
	} else if (qlens->include_srcflux_in_inversion) {
		n_amps += qlens->n_ptsrc;
	}

	//if (image_pixel_index != image_npixels) die("Number of active pixels (%i) doesn't seem to match image_npixels (%i)",image_pixel_index,image_npixels);

	if ((verbal) and (qlens->mpi_id==0)) {
		//cout << "CHECK delaunay srcgrid n_gridpts=" << delaunay_srcgrids[imggrid_i]->n_gridpts << endl;
		if ((qlens->source_fit_mode==Delaunay_Source) and (delaunay_srcgrid != NULL)) cout << "source # of pixels: " << delaunay_srcgrid->n_gridpts << ", # of active pixels: " << source_npixels << endl;
		else cout << "source # of pixels: " << cartesian_srcgrid->number_of_pixels << ", counted up as " << tot_npixels_count << ", # of active pixels: " << source_npixels << endl;
		if ((potential_perturbations) and (lensgrid != NULL)) cout << "pixellated qlens # of pixels: " << lensgrid->n_gridpts << ", # of active pixels: " << lensgrid_npixels << endl;
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for assigning pixel mappings: "  << wtime.count() << endl;
	}

	return true;
}

// this function shouldn't be necessary anymore, this is all done in setup_ray_tracing_arrays
void ImagePixelGrid::assign_foreground_mappings(const bool use_data)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	image_npixels_fgmask = 0;
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!image_data) or (!use_data) or (image_data->foreground_mask[i][j])) {
				image_npixels_fgmask++;
			}
		}
	}
	if (image_npixels_fgmask==0) die("no pixels in foreground mask");

	p.sbprofile_surface_brightness.resize(image_npixels_fgmask);
	for (int i=0; i < image_npixels_fgmask; i++) p.sbprofile_surface_brightness[i] = 0;
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for assigning foreground pixel mappings: "  << wtime.count() << endl;
	}
}

void ImagePixelGrid::initialize_pixel_matrices(const bool potential_perturbations, bool verbal)
{
	CartesianSourceGrid *cartesian_srcgrid = cartesian_srcgrid; // only used here if n_image_prior is on
	SB_Profile** sb_list = qlens->sb_list;

	if (Lmatrix_sparse != NULL) die("Lmatrix_sparse already initialized");
	//if (amplitude_vector != NULL) die("source surface brightness vector already initialized");
	amplitude_vector.resize(n_amps);
	if ((qlens->use_lum_weighted_regularization) or (qlens->use_distance_weighted_regularization) or (qlens->use_mag_weighted_regularization)) {
		if (reg_weight_factor != NULL) die("FUCK reg_ewight_factor");
		reg_weight_factor = new double[source_npixels];
	}

	bool delaunay = false;
	if (qlens->source_fit_mode==Delaunay_Source) delaunay = true;

	if ((qlens->source_fit_mode==Delaunay_Source) or (qlens->source_fit_mode==Cartesian_Source)) n_src_inv = n_pixsrc_to_include_in_Lmatrix;
	else if (qlens->source_fit_mode==Shapelet_Source) n_src_inv = 1; // currently, there is only support for one set of shapelets to invert; can generalize this later
	ImagePixelGrid *imggrid;
	Lmatrix_n_elements = 0;
	for (int i=0; i < n_src_inv; i++) {
		imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
		if (delaunay) {
			Lmatrix_n_elements += imggrid->count_nonzero_source_pixel_mappings_delaunay();
		} else {
			Lmatrix_n_elements += imggrid->count_nonzero_source_pixel_mappings_cartesian();
		}
	}
	if (potential_perturbations) Lmatrix_n_elements += count_nonzero_lensgrid_pixel_mappings();
	if ((qlens->mpi_id==0) and (verbal)) cout << "Expected Lmatrix_n_elements=" << Lmatrix_n_elements << endl << flush;
	//die();
	if (Lmatrix_index != NULL) die("FOOK Lmatrix_index wasn't freed");
	Lmatrix_index = new int[Lmatrix_n_elements];
	if (image_pixel_location_Lmatrix != NULL) die("FOOK image_pixel_location_Lmatrix wasn't freed");
	if (!qlens->psf_supersampling) image_pixel_location_Lmatrix = new int[image_npixels+1];
	else image_pixel_location_Lmatrix = new int[image_n_subpixels+1];
	if (Lmatrix_sparse != NULL) die("FOOK Lmatrix_sparse wasn't freed");
	Lmatrix_sparse = new double[Lmatrix_n_elements];

	src_npixel_start = 0;
	int src_npixels_inv, last_src_npixel_start;

	if ((qlens->source_fit_mode==Delaunay_Source) or (qlens->source_fit_mode==Cartesian_Source)) {
		ImagePixelGrid *imggrid;
		int npixels_check=0;
		for (int i=0; i < n_src_inv; i++) {
			if (i > 0) last_src_npixel_start = imggrid->src_npixel_start;
			imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
			if (i > 0) imggrid->src_npixel_start = last_src_npixel_start + src_npixels_inv;
			if (qlens->source_fit_mode==Delaunay_Source) {
				src_npixels_inv = imggrid->delaunay_srcgrid->n_active_pixels;
				//cout << "src_npixels for pixsrc " << i << " = " << src_npixels_inv[i] << endl;
			} else if (qlens->source_fit_mode==Cartesian_Source) {
				src_npixels_inv = imggrid->cartesian_srcgrid->n_active_pixels;
			}
			npixels_check += src_npixels_inv;
			imggrid->source_npixels_inv = src_npixels_inv;
		}
		if (npixels_check != source_npixels) die("wrong number of total source pixels from total n_active_pixels (%i versus %i)",npixels_check,source_npixels);
	}
	else if (qlens->source_fit_mode==Shapelet_Source) {
		int shapelet_i = -1;
		for (int i=0; i < qlens->n_sb; i++) {
			if ((sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
				shapelet_i = i;
				break;
			}
		}
		src_npixels_inv = *(sb_list[shapelet_i]->indxptr);
		if (src_npixels_inv != source_npixels) die("wrong number of source pixels from shapelet amplitudes");
	}
	else die("unknown source pixellation mode");

	//if (qlens->include_imgfluxes_in_inversion) {
		//int nimgs = 0;
		//for (int i=0; i < qlens->n_ptsrc; i++) nimgs += qlens->ptsrc_list[i]->images.size();
		//Lmatrix_transpose_ptimg_amps.resize(nimgs,image_npixels);
	//} else if (qlens->include_srcflux_in_inversion) {
		//Lmatrix_transpose_ptimg_amps.resize(qlens->n_ptsrc,image_npixels);
	//}

	if ((qlens->mpi_id==0) and (verbal)) cout << "Creating Lmatrix...\n";
	if (!qlens->psf_supersampling) {
		if (qlens->matrix_format==DENSE) construct_Lmatrix_dense(delaunay,potential_perturbations,verbal);
		else construct_Lmatrix(delaunay,potential_perturbations,verbal);
		//construct_Lmatrix(delaunay,potential_perturbations,verbal);
	} else construct_Lmatrix_supersampled(delaunay,potential_perturbations,verbal);
	if (qlens->matrix_format==DENSE) {
		//convert_Lmatrix_to_dense();
		if (n_mge_amps > 0) add_MGE_amplitudes_to_Lmatrix();
	}
}

void ImagePixelGrid::count_shapelet_amplitudes()
{
	double nmax;
	source_npixels = 0;
	for (int i=0; i < qlens->n_sb; i++) {
		if ((qlens->sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
			nmax = *(qlens->sb_list[i]->indxptr);
			source_npixels += nmax*nmax;
			break;
		}
	}
}

void ImagePixelGrid::count_MGE_amplitudes(int& n_mge_objects, int& n_gaussians)
{
	n_mge_objects = 0;
	n_gaussians = 0;
	for (int i=0; i < qlens->n_sb; i++) {
		if ((qlens->sb_list[i]->sbtype==MULTI_GAUSSIAN_EXPANSION) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
			n_mge_objects++;
			n_gaussians += *(qlens->sb_list[i]->indxptr);
		}
	}
}

void ImagePixelGrid::initialize_pixel_matrices_shapelets(bool verbal)
{
	//if (amplitude_vector != NULL) die("source surface brightness vector already initialized");
	count_shapelet_amplitudes();
	count_MGE_amplitudes(n_mge_sets,n_mge_amps);
	if (n_mge_sets > 0) create_MGE_regularization_matrices();
	source_and_lens_n_amps = source_npixels + n_mge_amps; // it's possible one could add shapelet potential corrections later, but probably not worth doing
	n_amps = source_and_lens_n_amps;
	Lmatrix_n_amps = source_and_lens_n_amps; // store the number of source/potential amplitudes for each image pixel grid; useful later for initializing/cleaning up FFT convolution arrays
	if (qlens->include_imgfluxes_in_inversion) {
		for (int i=0; i < qlens->n_ptsrc; i++) {
			n_amps += qlens->ptsrc_list[i]->images.size(); // in this case, source amplitudes include point image amplitudes as well as pixel values
		}
	} else if (qlens->include_srcflux_in_inversion) {
		n_amps += qlens->n_ptsrc;
	}

	if (n_amps <= 0) die("no shapelet or point source amplitude parameters found");
	//amplitude_vector = new double[n_amps];
	amplitude_vector.resize(n_amps);
	if ((qlens->use_lum_weighted_regularization) or (qlens->use_distance_weighted_regularization) or (qlens->use_mag_weighted_regularization)) {
		if (reg_weight_factor != NULL) die("FUCK reg_ewight_factor");
		reg_weight_factor = new double[source_npixels];
		for (int i=0; i < source_npixels; i++) reg_weight_factor[i] = 1.0;
	}

	if (qlens->use_noise_map) {
		int ii,i,j;
		for (ii=0; ii < image_npixels; ii++) {
			i = mask_pixels_i[ii];
			j = mask_pixels_j[ii];
			imgpixel_covinv_vector(ii) = image_data->covinv_map[i][j];
		}
	}
	if ((qlens->mpi_id==0) and (verbal)) cout << "Creating shapelet Lmatrix...\n";
	//Lmatrix_dense0.input(image_npixels,n_amps);
	//Lmatrix_dense0 = 0;
	//Lmatrix_dense = Eigen::MatrixXd::Zero(image_npixels,n_amps);
	Lmatrix_trans_dense = Eigen::MatrixXd::Zero(n_amps,image_npixels);
	if (qlens->include_imgfluxes_in_inversion) {
		int nimgs = 0;
		for (int i=0; i < qlens->n_ptsrc; i++) nimgs += qlens->ptsrc_list[i]->images.size();
		//Lmatrix_transpose_ptimg_amps.resize(nimgs,image_npixels);
	} else if (qlens->include_srcflux_in_inversion) {
		//Lmatrix_transpose_ptimg_amps.resize(qlens->n_ptsrc,image_npixels);
	}

	construct_Lmatrix_shapelets();
	if (n_mge_amps > 0) add_MGE_amplitudes_to_Lmatrix();
}

void ImagePixelGrid::clear_pixel_matrices()
{
	if (reg_weight_factor != NULL) delete[] reg_weight_factor;
	if (image_pixel_location_Lmatrix != NULL) delete[] image_pixel_location_Lmatrix;
	if (Lmatrix_sparse != NULL) delete[] Lmatrix_sparse;
	if (Lmatrix_index != NULL) delete[] Lmatrix_index;

	if (Rmatrix_MGE_packed != NULL) delete[] Rmatrix_MGE_packed;
	if (Rmatrix_MGE_log_determinants != NULL) delete[] Rmatrix_MGE_log_determinants;
	if (mge_list != NULL) delete[] mge_list;

	reg_weight_factor = NULL;
	image_pixel_location_Lmatrix = NULL;
	Lmatrix_sparse = NULL;
	Lmatrix_index = NULL;

	n_src_inv = 0;
	Rmatrix_MGE_packed = NULL;
	Rmatrix_MGE_log_determinants = NULL;
	mge_list = NULL;
}

void ImagePixelGrid::construct_Lmatrix(const bool delaunay, const bool potential_perturbations, const bool verbal)
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();

	int n_imggrids = n_pixsrc_to_include_in_Lmatrix;
	if (n_imggrids==0) die("No image grids are included for constructing Lmatrix_sparse");
	ImagePixelGrid **imggrids;
	imggrids = new ImagePixelGrid*[n_imggrids+1]; // last element will be used to identify end of list
	for (int i=0; i < n_imggrids; i++) imggrids[i] = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
	imggrids[n_imggrids] = NULL;
	ImagePixelGrid **imggrid_end = imggrids+n_imggrids;

	int img_index;
	int index;
	int i,j;
	Lmatrix_rows = new vector<double>[image_npixels];
	Lmatrix_index_rows = new vector<int>[image_npixels];
	int *Lmatrix_row_nn = new int[image_npixels];
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	ImagePixelGrid** imggrid_ptr;
	ImagePixelGrid* imggrid;
	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif
		int nsubpix,subcell_index;
		lensvector<double> *center_srcpt;
		nsubpix = n_subpix_per_pixel;

		if ((qlens->split_imgpixels) and (!qlens->raytrace_using_pixel_centers)) {
			//#pragma omp for private(img_index,i,j,imggrid_ptr,imggrid,index,center_srcpt) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				index = 0;
				i = mask_pixels_i[img_index];
				j = mask_pixels_j[img_index];

				center_srcpt = subpixel_center_sourcepts[i][j];
				if (delaunay) {
					if (delaunay_srcgrid != NULL) {
						for (imggrid_ptr = imggrids; imggrid_ptr != imggrid_end; imggrid_ptr++) {
							imggrid = (*imggrid_ptr);
							//ImgGrid_Params<Eigen::VectorXd,Eigen::MatrixXd,double>& p = imggrid->assign_imggrid_param_object<Eigen::VectorXd,Eigen::MatrixXd,double>();
							for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
								//cout << "source " << kk << ": adding Lmatrix_sparse elements (redshift_index=" << imggrid->src_redshift_index << ")" << endl;
								//if (imggrid->n_mapped_srcpixels[i][j][subcell_index] != 0) cout << "MAPPED SRCPIXELS! (redshift_index=" << imggrid->src_redshift_index << ")" << endl;
								imggrid->delaunay_srcgrid->calculate_Lmatrix(img_index,imggrid->mapped_delaunay_srcpixels[i][j].data(),imggrid->n_mapped_srcpixels[i][j],index,subcell_index,1.0/nsubpix,thread);
								//if (kk==0) ntot_mapped += imggrid->n_mapped_srcpixels[i][j][subcell_index];
								//else ntot_mapped2 += imggrid->n_mapped_srcpixels[i][j][subcell_index];
							}
						}
					}
					if ((potential_perturbations) and (lensgrid != NULL)) {
						for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
							lensgrid->calculate_Lmatrix(img_index,mapped_potpixels[i][j].data(),n_mapped_potpixels[i][j],subpixel_source_gradient[i][j][subcell_index],index,subcell_index,source_npixels,1.0/nsubpix,thread);
						}
					}
				} else {
					for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
						for (imggrid_ptr = imggrids; imggrid_ptr != imggrid_end; imggrid_ptr++) {
							imggrid = (*imggrid_ptr);
							imggrid->cartesian_srcgrid->calculate_Lmatrix_interpolate(img_index,imggrid->mapped_cartesian_srcpixels[i][j],index,center_srcpt[subcell_index],subcell_index,1.0/nsubpix,thread);
						}
					}
				}
				Lmatrix_row_nn[img_index] = index;
			}
		} else {
			//#pragma omp for private(img_index,i,j,imggrid_ptr,imggrid,index) schedule(dynamic)	
			for (img_index=0; img_index < image_npixels; img_index++) {
				index = 0;
				i = mask_pixels_i[img_index];
				j = mask_pixels_j[img_index];
				if (delaunay) {
					if (delaunay_srcgrid != NULL) {
						for (imggrid_ptr = imggrids; imggrid_ptr != imggrid_end; imggrid_ptr++) {
							imggrid = (*imggrid_ptr);
							imggrid->delaunay_srcgrid->calculate_Lmatrix(img_index,imggrid->mapped_delaunay_srcpixels[i][j].data(),imggrid->n_mapped_srcpixels[i][j],index,0,1.0,thread);
						}
					}
				} else {
					for (imggrid_ptr = imggrids; imggrid_ptr != imggrid_end; imggrid_ptr++) {
						imggrid = (*imggrid_ptr);
						imggrid->cartesian_srcgrid->calculate_Lmatrix_interpolate(img_index,imggrid->mapped_cartesian_srcpixels[i][j],index,center_sourcepts[i][j],0,1.0,thread);
					}
				}
				if ((potential_perturbations) and (lensgrid != NULL)) {
					lensgrid->calculate_Lmatrix(img_index,mapped_potpixels[i][j].data(),n_mapped_potpixels[i][j],subpixel_source_gradient[i][j][0],index,0,source_npixels,1.0,thread);
				}

				Lmatrix_row_nn[img_index] = index;
			}
		}
	}

	image_pixel_location_Lmatrix[0] = 0;
	for (img_index=0; img_index < image_npixels; img_index++) {
		image_pixel_location_Lmatrix[img_index+1] = image_pixel_location_Lmatrix[img_index] + Lmatrix_row_nn[img_index];
	}
	if (image_pixel_location_Lmatrix[img_index] != Lmatrix_n_elements) die("Number of Lmatrix elements don't match (%i vs %i)",image_pixel_location_Lmatrix[img_index],Lmatrix_n_elements);

	index=0;
	for (i=0; i < image_npixels; i++) {
		for (j=0; j < Lmatrix_row_nn[i]; j++) {
			Lmatrix_sparse[index] = Lmatrix_rows[i][j];
			Lmatrix_index[index] = Lmatrix_index_rows[i][j];
			index++;
		}
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for constructing Lmatrix: "  << wtime.count() << endl;
	}
	if ((qlens->mpi_id==0) and (verbal)) {
		int Lmatrix_ntot = n_amps*image_npixels;
		double sparseness = ((double) Lmatrix_n_elements)/Lmatrix_ntot;
		cout << "image has " << image_npixels << " pixels in mask, Lmatrix has " << Lmatrix_n_elements << " nonzero elements (sparseness " << sparseness << ")\n";
	}

	delete[] Lmatrix_row_nn;
	delete[] Lmatrix_rows;
	delete[] Lmatrix_index_rows;
	delete[] imggrids;
}

void ImagePixelGrid::construct_Lmatrix_dense(const bool delaunay, const bool potential_perturbations, const bool verbal)
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();

	int n_imggrids = n_pixsrc_to_include_in_Lmatrix;
	if (n_imggrids==0) die("No image grids are included for constructing Lmatrix");
	ImagePixelGrid **imggrids;
	imggrids = new ImagePixelGrid*[n_imggrids+1]; // last element will be used to identify end of list
	for (int i=0; i < n_imggrids; i++) imggrids[i] = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
	imggrids[n_imggrids] = NULL;
	ImagePixelGrid **imggrid_end = imggrids+n_imggrids;

	int img_index;
	int index;
	int i,j;

	//Lmatrix_dense0.input(image_npixels,n_amps);
	//Lmatrix_dense0 = 0;
	//Lmatrix_dense = Eigen::MatrixXd::Zero(image_npixels,n_amps);
	Lmatrix_trans_dense = Eigen::MatrixXd::Zero(n_amps,image_npixels);

	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	ImagePixelGrid** imggrid_ptr;
	ImagePixelGrid* imggrid;
	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif
		int nsubpix,subcell_index;
		lensvector<double> *center_srcpt;
		nsubpix = n_subpix_per_pixel;

		if ((qlens->split_imgpixels) and (!qlens->raytrace_using_pixel_centers)) {
			//#pragma omp for private(img_index,i,j,imggrid_ptr,imggrid,index,center_srcpt) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				index = 0;
				i = mask_pixels_i[img_index];
				j = mask_pixels_j[img_index];

				center_srcpt = subpixel_center_sourcepts[i][j];
				if (delaunay) {
					if (delaunay_srcgrid != NULL) {
						for (imggrid_ptr = imggrids; imggrid_ptr != imggrid_end; imggrid_ptr++) {
							imggrid = (*imggrid_ptr);
							//ImgGrid_Params<Eigen::VectorXd,Eigen::MatrixXd,double>& p = imggrid->assign_imggrid_param_object<Eigen::VectorXd,Eigen::MatrixXd,double>();
							for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
								//cout << "source " << kk << ": adding Lmatrix_sparse elements (redshift_index=" << imggrid->src_redshift_index << ")" << endl;
								//if (imggrid->n_mapped_srcpixels[i][j][subcell_index] != 0) cout << "MAPPED SRCPIXELS! (redshift_index=" << imggrid->src_redshift_index << ")" << endl;
								imggrid->delaunay_srcgrid->calculate_Lmatrix_elements(img_index,imggrid->mapped_delaunay_srcpixels[i][j].data(),imggrid->n_mapped_srcpixels[i][j],index,subcell_index,1.0/nsubpix,thread);
								//if (kk==0) ntot_mapped += imggrid->n_mapped_srcpixels[i][j][subcell_index];
								//else ntot_mapped2 += imggrid->n_mapped_srcpixels[i][j][subcell_index];
							}
						}
					}
					//if ((potential_perturbations) and (lensgrid != NULL)) {
						//for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
							//lensgrid->calculate_Lmatrix(img_index,mapped_potpixels[i][j].data(),n_mapped_potpixels[i][j],subpixel_source_gradient[i][j][subcell_index],index,subcell_index,source_npixels,1.0/nsubpix,thread);
						//}
					//}
				//} else {
					//for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
						//for (imggrid_ptr = imggrids; imggrid_ptr != imggrid_end; imggrid_ptr++) {
							//imggrid = (*imggrid_ptr);
							//imggrid->cartesian_srcgrid->calculate_Lmatrix_interpolate(img_index,imggrid->mapped_cartesian_srcpixels[i][j],index,center_srcpt[subcell_index],subcell_index,1.0/nsubpix,thread);
						//}
					//}
				}
			}
		} else {
			//#pragma omp for private(img_index,i,j,imggrid_ptr,imggrid,index) schedule(dynamic)	
			for (img_index=0; img_index < image_npixels; img_index++) {
				index = 0;
				i = mask_pixels_i[img_index];
				j = mask_pixels_j[img_index];
				if (delaunay) {
					if (delaunay_srcgrid != NULL) {
						for (imggrid_ptr = imggrids; imggrid_ptr != imggrid_end; imggrid_ptr++) {
							imggrid = (*imggrid_ptr);
							imggrid->delaunay_srcgrid->calculate_Lmatrix_elements(img_index,imggrid->mapped_delaunay_srcpixels[i][j].data(),imggrid->n_mapped_srcpixels[i][j],index,0,1.0,thread);
						}
					}
				//} else {
					//for (imggrid_ptr = imggrids; imggrid_ptr != imggrid_end; imggrid_ptr++) {
						//imggrid = (*imggrid_ptr);
						//imggrid->cartesian_srcgrid->calculate_Lmatrix_interpolate(img_index,imggrid->mapped_cartesian_srcpixels[i][j],index,center_sourcepts[i][j],0,1.0,thread);
					//}
				}
				//if ((potential_perturbations) and (lensgrid != NULL)) {
					//lensgrid->calculate_Lmatrix(img_index,mapped_potpixels[i][j].data(),n_mapped_potpixels[i][j],subpixel_source_gradient[i][j][0],index,0,source_npixels,1.0,thread);
				//}
//
				//Lmatrix_row_nn[img_index] = index;
			}
		}
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for constructing Lmatrix: "  << wtime.count() << endl;
	}
	if ((qlens->mpi_id==0) and (verbal)) {
		int Lmatrix_ntot = n_amps*image_npixels;
		double sparseness = ((double) Lmatrix_n_elements)/Lmatrix_ntot;
		cout << "image has " << image_npixels << " pixels in mask, Lmatrix has " << Lmatrix_n_elements << " nonzero elements (sparseness " << sparseness << ")\n";
	}

	delete[] imggrids;
}

void ImagePixelGrid::construct_Lmatrix_supersampled(const bool delaunay, const bool potential_perturbations, const bool verbal)
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();

	int n_imggrids = n_pixsrc_to_include_in_Lmatrix;
	if (n_imggrids==0) die("No image grids are included for constructing Lmatrix");
	ImagePixelGrid **imggrids;
	imggrids = new ImagePixelGrid*[n_imggrids+1]; // last element will be used to identify end of list
	for (int i=0; i < n_imggrids; i++) imggrids[i] = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
	imggrids[n_imggrids] = NULL;
	ImagePixelGrid **imggrid_end = imggrids+n_imggrids;

	int img_index;
	int index;
	int i,j;
	Lmatrix_rows = new vector<double>[image_n_subpixels];
	Lmatrix_index_rows = new vector<int>[image_n_subpixels];
	int *Lmatrix_row_nn = new int[image_n_subpixels];
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	ImagePixelGrid** imggrid_ptr;
	ImagePixelGrid* imggrid;
	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif
		int nsubpix,subcell_index;
		lensvector<double> *center_srcpt;

		nsubpix = n_subpix_per_pixel;

		//#pragma omp for private(img_index,i,j,imggrid_ptr,imggrid,nsubpix,index,center_srcpt) schedule(dynamic)
		for (img_index=0; img_index < image_n_subpixels; img_index++) {
			index = 0;
			i = mask_subpixel_i[img_index];
			j = mask_subpixel_j[img_index];
			subcell_index = mask_subpixel_index[img_index];

			center_srcpt = subpixel_center_sourcepts[i][j];
			if (delaunay) {
				if (delaunay_srcgrid != NULL) { // this might be the case if we're only doing point sources but happen to be in delaunay source mode
					for (imggrid_ptr = imggrids; imggrid_ptr != NULL; imggrid_ptr++) {
						imggrid = (*imggrid_ptr);
						imggrid->delaunay_srcgrid->calculate_Lmatrix(img_index,imggrid->mapped_delaunay_srcpixels[i][j].data(),imggrid->n_mapped_srcpixels[i][j],index,subcell_index,1.0,thread);
					}
				}
			} else {
				for (imggrid_ptr = imggrids; imggrid_ptr != NULL; imggrid_ptr++) {
					imggrid = (*imggrid_ptr);
					imggrid->cartesian_srcgrid->calculate_Lmatrix_interpolate(img_index,imggrid->mapped_cartesian_srcpixels[i][j],index,center_srcpt[subcell_index],subcell_index,1.0,thread);
				}
			}
			if ((potential_perturbations) and (lensgrid != NULL)) {
				lensgrid->calculate_Lmatrix(img_index,mapped_potpixels[i][j].data(),n_mapped_potpixels[i][j],subpixel_source_gradient[i][j][subcell_index],index,subcell_index,source_npixels,1.0,thread);
			}
			Lmatrix_row_nn[img_index] = index;
		}
	}

	image_pixel_location_Lmatrix[0] = 0;
	for (img_index=0; img_index < image_n_subpixels; img_index++) {
		image_pixel_location_Lmatrix[img_index+1] = image_pixel_location_Lmatrix[img_index] + Lmatrix_row_nn[img_index];
	}
	if (image_pixel_location_Lmatrix[img_index] != Lmatrix_n_elements) die("Number of supersampled Lmatrix elements don't match (%i vs %i)",image_pixel_location_Lmatrix[img_index],Lmatrix_n_elements);

	index=0;
	for (i=0; i < image_n_subpixels; i++) {
		for (j=0; j < Lmatrix_row_nn[i]; j++) {
			Lmatrix_sparse[index] = Lmatrix_rows[i][j];
			Lmatrix_index[index] = Lmatrix_index_rows[i][j];
			index++;
		}
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for constructing Lmatrix: "  << wtime.count() << endl;
	}
	if ((qlens->mpi_id==0) and (verbal)) {
		int Lmatrix_ntot = n_amps*image_n_subpixels;
		double sparseness = ((double) Lmatrix_n_elements)/Lmatrix_ntot;
		cout << "image has " << image_npixels << " pixels in mask, Lmatrix has " << Lmatrix_n_elements << " nonzero elements (sparseness " << sparseness << ")\n";
	}

	delete[] Lmatrix_row_nn;
	delete[] Lmatrix_rows;
	delete[] Lmatrix_index_rows;
}

void ImagePixelGrid::construct_Lmatrix_shapelets()
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();
	SB_Profile** sb_list = qlens->sb_list;
	int img_index;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	int i,j,k,n_shapelet_sets = 0;
	for (i=0; i < qlens->n_sb; i++) {
		if ((sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) n_shapelet_sets++;
	}
	if (n_shapelet_sets==0) return;

	SB_Profile** shapelet;
	shapelet = new SB_Profile*[n_shapelet_sets];
	for (i=0,j=0; i < qlens->n_sb; i++) {
		if (sb_list[i]->sbtype==SHAPELET) {
			if ((!sb_list[i]->is_lensed) or (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
				shapelet[j++] = sb_list[i];
			}
		}
	}

	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		int nsubpix,subcell_index;
		lensvector<double> *center_srcpt, *center_pt;

		nsubpix = n_subpix_per_pixel;

		if (qlens->split_imgpixels) {
			#pragma omp for private(img_index,i,j,k,nsubpix,center_srcpt,center_pt) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				i = emask_pixels_i[img_index];
				j = emask_pixels_j[img_index];

				center_srcpt = subpixel_center_sourcepts[i][j];
				center_pt = subpixel_center_pts[i][j];
				for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
					//double *Lmatptr = Lmatrix_dense0.subarray(img_index);
					double *Lmatptr = Lmatrix_trans_dense.col(img_index).data();
					for (k=0; k < n_shapelet_sets; k++) {
						if (shapelet[k]->is_lensed) {
							//Note that calculate_Lmatrix_elements(...) will increment Lmatptr as it goes
							shapelet[k]->calculate_Lmatrix_elements(center_srcpt[subcell_index][0],center_srcpt[subcell_index][1],Lmatptr,1.0/nsubpix);
							//shapelet[k]->calculate_Lmatrix_elements(center_sourcepts[i][j][0],center_sourcepts[i][j][1],Lmatptr,1.0/nsubpix);
							//cout << "cell " << i << "," << j << ": " << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " vs " << center_pt[subcell_index][0] << " " << center_pt[subcell_index][1] << endl;
						} else {
							shapelet[k]->calculate_Lmatrix_elements(center_pt[subcell_index][0],center_pt[subcell_index][1],Lmatptr,1.0/nsubpix);
						}
					}
				}
			}
		} else {
			lensvector<double> center, center_srcpt;
			#pragma omp for private(img_index,i,j,center_srcpt) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				i = emask_pixels_i[img_index];
				j = emask_pixels_j[img_index];

				double *Lmatptr = Lmatrix_trans_dense.col(img_index).data();
				for (k=0; k < n_shapelet_sets; k++) {
					if (shapelet[k]->is_lensed) {
						center_srcpt = center_sourcepts[i][j];
						shapelet[k]->calculate_Lmatrix_elements(center_srcpt[0],center_srcpt[1],Lmatptr,1.0);
					} else {
						center = center_pts[i][j];
						shapelet[k]->calculate_Lmatrix_elements(center[0],center[1],Lmatptr,1.0);
					}
				}
			}
		}
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for constructing shapelet Lmatrix: "  << wtime.count() << endl;
	}
	delete[] shapelet;
}

void ImagePixelGrid::add_MGE_amplitudes_to_Lmatrix()
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();
	int img_index;
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	int i,j,k;
	if (n_mge_sets==0) return;

	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		int nsubpix,subcell_index;
		lensvector<double> *center_srcpt, *center_pt;
		nsubpix = n_subpix_per_pixel;

		if (qlens->split_imgpixels) {
			#pragma omp for private(img_index,i,j,k,nsubpix,center_srcpt,center_pt) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				i = emask_pixels_i[img_index];
				j = emask_pixels_j[img_index];

				center_srcpt = subpixel_center_sourcepts[i][j];
				center_pt = subpixel_center_pts[i][j];
				for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
					double *Lmatptr = Lmatrix_trans_dense.col(img_index).data() + source_npixels + lensgrid_npixels;
					for (k=0; k < n_mge_sets; k++) {
						if (mge_list[k]->is_lensed) {
							//Note that calculate_Lmatrix_elements(...) will increment Lmatptr as it goes
							mge_list[k]->calculate_Lmatrix_elements(center_srcpt[subcell_index][0],center_srcpt[subcell_index][1],Lmatptr,1.0/nsubpix);
							//mge_list[k]->calculate_Lmatrix_elements(center_sourcepts[i][j][0],center_sourcepts[i][j][1],Lmatptr,1.0/nsubpix);
							//cout << "cell " << i << "," << j << ": " << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " vs " << center_pt[subcell_index][0] << " " << center_pt[subcell_index][1] << endl;
						} else {
							mge_list[k]->calculate_Lmatrix_elements(center_pt[subcell_index][0],center_pt[subcell_index][1],Lmatptr,1.0/nsubpix);
						}
					}
				}
			}
		} else {
			lensvector<double> center, center_srcpt;
			#pragma omp for private(img_index,i,j,center_srcpt) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				i = emask_pixels_i[img_index];
				j = emask_pixels_j[img_index];

				double *Lmatptr = Lmatrix_trans_dense.col(img_index).data();
				for (k=0; k < n_mge_sets; k++) {
					if (mge_list[k]->is_lensed) {
						center_srcpt = center_sourcepts[i][j];
						mge_list[k]->calculate_Lmatrix_elements(center_srcpt[0],center_srcpt[1],Lmatptr,1.0);
					} else {
						center = center_pts[i][j];
						mge_list[k]->calculate_Lmatrix_elements(center[0],center[1],Lmatptr,1.0);
					}
				}
			}
		}
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for adding MGE elements to Lmatrix: "  << wtime.count() << endl;
	}
}

void ImagePixelGrid::PSF_convolution_Lmatrix(bool verbal)
{
	if (psf==NULL) return;
	if (qlens->psf_supersampling) die("PSF supersampling has not been implemented with sparse Lmatrix");
	if (psf->use_input_psf_matrix) {
		if (psf->psf_matrix == NULL) return;
	}
	else if (psf->generate_PSF_matrix(pixel_xlength,pixel_ylength,qlens->psf_supersampling)==false) return;
	if ((qlens->mpi_id==0) and (verbal)) cout << "Beginning PSF convolution (sparse)...\n";
	double nx_half, ny_half;
	nx_half = psf->psf_npixels_x/2;
	ny_half = psf->psf_npixels_y/2;

	int *Lmatrix_psf_row_nn = new int[image_npixels];
	vector<double> *Lmatrix_psf_rows = new vector<double>[image_npixels];
	vector<int> *Lmatrix_psf_index_rows = new vector<int>[image_npixels];

	int i,j,k,l,m;
	int Lmatrix_psf_nn=0;
	if (source_and_lens_n_amps > 0) {
		if (qlens->show_wtime) {
			wtime0 = std::chrono::steady_clock::now();
		}
		int psf_k, psf_l;
		int img_index1, img_index2, srcpot_index, col_index;
		int index;
		bool new_entry;
		int Lmatrix_psf_nn_part=0;
		#pragma omp parallel for private(m,k,l,i,j,img_index1,img_index2,srcpot_index,col_index,psf_k,psf_l,index,new_entry) schedule(static) reduction(+:Lmatrix_psf_nn_part)
		for (img_index1=0; img_index1 < image_npixels; img_index1++)
		{ // this loops over columns of the PSF blurring matrix
			int col_i=0;
			Lmatrix_psf_row_nn[img_index1] = 0;
			k = emask_pixels_i[img_index1];
			l = emask_pixels_j[img_index1];
			for (psf_k=0; psf_k < psf->psf_npixels_x; psf_k++) {
				i = k + nx_half - psf_k; // Note, 'k' is the index for the convolved image, so we have k = i - nx_half + psf_k
				if ((i >= 0) and (i < x_N)) {
					for (psf_l=0; psf_l < psf->psf_npixels_y; psf_l++) {
						j = l + ny_half - psf_l; // Note, 'l' is the index for the convolved image, so we have l = j - ny_half + psf_l
						if ((j >= 0) and (j < y_N)) {
							if (maps_to_source_pixel[i][j]) {
								img_index2 = pixel_index[i][j];

								for (index=image_pixel_location_Lmatrix[img_index2]; index < image_pixel_location_Lmatrix[img_index2+1]; index++) {
									if (Lmatrix_sparse[index] != 0) {
										srcpot_index = Lmatrix_index[index];
										new_entry = true;
										for (m=0; m < Lmatrix_psf_row_nn[img_index1]; m++) {
											if (Lmatrix_psf_index_rows[img_index1][m]==srcpot_index) { col_index=m; new_entry=false; }
										}
										if (new_entry) {
											Lmatrix_psf_rows[img_index1].push_back(psf->psf_matrix[psf_k][psf_l]*Lmatrix_sparse[index]);
											Lmatrix_psf_index_rows[img_index1].push_back(srcpot_index);
											Lmatrix_psf_row_nn[img_index1]++;
											col_i++;
										} else {
											Lmatrix_psf_rows[img_index1][col_index] += psf->psf_matrix[psf_k][psf_l]*Lmatrix_sparse[index];
										}
									}
								}
							}
						}
					}
				}
			}
			Lmatrix_psf_nn_part += col_i;
		}
		Lmatrix_psf_nn = Lmatrix_psf_nn_part;
	} else {
		for (int img_index=0; img_index < image_npixels; img_index++) {
			Lmatrix_psf_row_nn[img_index] = 0;
		}
	}


	if (qlens->include_imgfluxes_in_inversion) {
		double *Lmatptr;
		i=0;
		int src_amp_i;
		for (j=0; j < qlens->n_ptsrc; j++) {
			for (k=0; k < qlens->ptsrc_list[j]->images.size(); k++) {
				//Lmatptr = Lmatrix_transpose_ptimg_amps.row(i).data();
				generate_point_images(qlens->ptsrc_list[j]->images, point_image_surface_brightness.data(), false, -1, k);
				src_amp_i = source_and_lens_n_amps + i;
				for (int img_index=0; img_index < image_npixels; img_index++) {
					if (point_image_surface_brightness[img_index] != 0) {
						Lmatrix_psf_rows[img_index].push_back(point_image_surface_brightness[img_index]);
						Lmatrix_psf_index_rows[img_index].push_back(src_amp_i);
						Lmatrix_psf_row_nn[img_index]++;
						Lmatrix_psf_nn++;
					}
				}
				i++;
			}
		}
		//double *Lmatrix_transpose_line;
		//i=0;
		//for (j=0; j < qlens->n_ptsrc; j++) {
			//for (k=0; k < qlens->ptsrc_list[j]->images.size(); k++) {
				//src_amp_i = source_and_lens_n_amps + i;
				//Lmatrix_transpose_line = Lmatrix_transpose_ptimg_amps.row(i).data();
				//for (int img_index=0; img_index < image_npixels; img_index++) {
					//if (Lmatrix_transpose_line[img_index] != 0) {
						//Lmatrix_psf_rows[img_index].push_back(Lmatrix_transpose_line[img_index]);
						//Lmatrix_psf_index_rows[img_index].push_back(src_amp_i);
						//Lmatrix_psf_row_nn[img_index]++;
						//Lmatrix_psf_nn++;
					//}
				//}
				//i++;
			//}
		//}
	} else if (qlens->include_srcflux_in_inversion) {
		//double *Lmatptr;
		int src_amp_i;
		for (j=0; j < qlens->n_ptsrc; j++) {
			//Lmatptr = Lmatrix_transpose_ptimg_amps.row(j).data();
			generate_point_images(qlens->ptsrc_list[j]->images, point_image_surface_brightness.data(), false, 1.0);
			src_amp_i = source_and_lens_n_amps + j;
			for (int img_index=0; img_index < image_npixels; img_index++) {
				if (point_image_surface_brightness[img_index] != 0) {
					Lmatrix_psf_rows[img_index].push_back(point_image_surface_brightness[img_index]);
					Lmatrix_psf_index_rows[img_index].push_back(src_amp_i);
					Lmatrix_psf_row_nn[img_index]++;
					Lmatrix_psf_nn++;
				}
			}

		}
		//double *Lmatrix_transpose_line;
		//for (j=0; j < qlens->n_ptsrc; j++) {
			//src_amp_i = source_and_lens_n_amps + j;
			//Lmatrix_transpose_line = Lmatrix_transpose_ptimg_amps.row(j).data();
			//for (int img_index=0; img_index < image_npixels; img_index++) {
				//if (Lmatrix_transpose_line[img_index] != 0) {
					//Lmatrix_psf_rows[img_index].push_back(Lmatrix_transpose_line[img_index]);
					//Lmatrix_psf_index_rows[img_index].push_back(src_amp_i);
					//Lmatrix_psf_row_nn[img_index]++;
					//Lmatrix_psf_nn++;
				//}
			//}
		//}
	}


	int *image_pixel_location_Lmatrix_psf = new int[image_npixels+1];
	image_pixel_location_Lmatrix_psf[0] = 0;
	for (m=0; m < image_npixels; m++) {
		image_pixel_location_Lmatrix_psf[m+1] = image_pixel_location_Lmatrix_psf[m] + Lmatrix_psf_row_nn[m];
	}

	double *Lmatrix_psf = new double[Lmatrix_psf_nn];
	int *Lmatrix_index_psf = new int[Lmatrix_psf_nn];

	int indx;
	for (m=0; m < image_npixels; m++) {
		indx = image_pixel_location_Lmatrix_psf[m];
		for (j=0; j < Lmatrix_psf_row_nn[m]; j++) {
			Lmatrix_psf[indx+j] = Lmatrix_psf_rows[m][j];
			Lmatrix_index_psf[indx+j] = Lmatrix_psf_index_rows[m][j];
		}
	}

	if ((qlens->mpi_id==0) and (verbal)) cout << "Lmatrix after PSF convolution: Lmatrix now has " << indx << " nonzero elements\n";

	delete[] Lmatrix_sparse;
	delete[] Lmatrix_index;
	delete[] image_pixel_location_Lmatrix;
	Lmatrix_sparse = Lmatrix_psf;
	Lmatrix_index = Lmatrix_index_psf;
	image_pixel_location_Lmatrix = image_pixel_location_Lmatrix_psf;
	Lmatrix_n_elements = Lmatrix_psf_nn;

	delete[] Lmatrix_psf_row_nn;
	delete[] Lmatrix_psf_rows;
	delete[] Lmatrix_psf_index_rows;

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for calculating PSF convolution of Lmatrix: "  << wtime.count() << endl;
	}
}

bool ImagePixelGrid::setup_FFT_convolution(const bool supersampling, const bool noninverted_foreground, const bool include_fgmask_in_inversion, const bool verbal)
{
	bool foreground = noninverted_foreground;
	if ((!foreground) and (include_fgmask_in_inversion)) foreground = true;
	if ((supersampling) and (foreground)) die("supersampling of PSF has not been implemented with foreground mask yet");
	if (qlens->show_wtime) {
		qlens->wtime0 = std::chrono::steady_clock::now();
	}
	int npix;
	int *pixel_map_ii, *pixel_map_jj;
	double **psf_ptr;
	int psf_nx, psf_ny;
	if (!supersampling) {
		psf_ptr = psf->psf_matrix;
		psf_nx = psf->psf_npixels_x;
		psf_ny = psf->psf_npixels_y;
		if (!foreground) {
			npix = image_npixels;
			pixel_map_ii = emask_pixels_i;
			pixel_map_jj = emask_pixels_j;
		} else {
			npix = image_npixels_fgmask;
			pixel_map_ii = fgmask_pixels_i;
			pixel_map_jj = fgmask_pixels_j;
		}
	} else {
		psf_ptr = psf->supersampled_psf_matrix;
		psf_nx = psf->supersampled_psf_npixels_x;
		psf_ny = psf->supersampled_psf_npixels_y;
		npix = image_n_subpixels;
		pixel_map_ii = emask_subpixels_ii;
		pixel_map_jj = emask_subpixels_jj;
		//cout << "PSF_NX=" << psf_nx << " PSF_NY=" << psf_ny << " npix=" << image_n_subpixels << endl;
	}

	if (psf->use_input_psf_matrix) {
		if (psf_ptr == NULL) return false;
	} else {
		if ((psf->psf_params.psf_width_x==0) and (psf->psf_params.psf_width_y==0)) return false;
		else if (psf->generate_PSF_matrix(pixel_xlength,pixel_ylength,supersampling)==false) {
			if (verbal) warn("could not generate_PSF matrix");
			return false;
		}
		if (qlens->mpi_id==0) cout << "generated PSF matrix" << endl;
	}
	int nx_half, ny_half;
	nx_half = psf_nx/2;
	ny_half = psf_ny/2;

	bool **selected_mask;
	if (foreground) {
		if ((image_data==NULL) or (fgmask==NULL)) selected_mask = NULL;
		else selected_mask = fgmask;
	} else {
		if (pixel_in_mask==NULL) selected_mask = NULL;
		else selected_mask = pixel_in_mask;
	}

	int& ni = (noninverted_foreground) ? fft_ni_fgmask : fft_ni;
	int& nj = (noninverted_foreground) ? fft_nj_fgmask : fft_nj;
	int& imin = (noninverted_foreground) ? fft_imin_fgmask : fft_imin;
	int& jmin = (noninverted_foreground) ? fft_jmin_fgmask : fft_jmin;

	int i,j,ii,jj,k,img_index;
	ni = 1;
	nj = 1;
	imin = 50000;
	jmin = 50000;

	int imax=-1,jmax=-1;
	int il0, jl0;

	for (img_index=0; img_index < npix; img_index++)
	{
		ii = pixel_map_ii[img_index];
		jj = pixel_map_jj[img_index];
		if (supersampling) {
			i = image_pixel_i_from_subcell_ii[ii];
			j = image_pixel_j_from_subcell_jj[jj];
		} else {
			i = ii;
			j = jj;
		}
		if ((selected_mask==NULL) or (selected_mask[i][j])) {
			if (ii > imax) imax = ii;
			if (jj > jmax) jmax = jj;
			if (ii < imin) imin = ii;
			if (jj < jmin) jmin = jj;
		}
	}
	il0 = 1+imax-imin + psf_nx; // will pad with extra zeros to avoid edge effects (wraparound of PSF blurring)
	jl0 = 1+jmax-jmin + psf_ny;

#ifdef USE_FFTW
	ni = il0;
	nj = jl0;
	if (ni % 2 != 0) ni++;
	if (nj % 2 != 0) nj++;
	int ncomplex = nj*(ni/2+1);
	int npix_conv = ni*nj;
	if (npix_conv < 0) die("negative ni and/or nk");
	double *psf_rvec = new double[npix_conv];
	if (noninverted_foreground) psf_transform_fgmask = new complex<double>[ncomplex];
	else psf_transform = new complex<double>[ncomplex];
	complex<double> *psf_transform_ptr = (noninverted_foreground) ? psf_transform_fgmask : psf_transform;

	fftw_plan fftplan_psf = fftw_plan_dft_r2c_2d(nj,ni,psf_rvec,reinterpret_cast<fftw_complex*>(psf_transform_ptr),FFTW_MEASURE);
	for (i=0; i < npix_conv; i++) psf_rvec[i] = 0;

	if (noninverted_foreground) single_img_rvec_fgmask = new double[npix_conv];
	else single_img_rvec = new double[npix_conv];
	double *single_img_rvec_ptr = (noninverted_foreground) ? single_img_rvec_fgmask : single_img_rvec;
	fftw_plan& fftplan_ref = (noninverted_foreground) ? fftplan_fgmask : fftplan;
	fftw_plan& fftplan_inverse_ref = (noninverted_foreground) ? fftplan_inverse_fgmask : fftplan_inverse;
	if (noninverted_foreground) img_transform_fgmask = new complex<double>[ncomplex];
	else img_transform = new complex<double>[ncomplex];
	complex<double> *img_transform_ptr = (noninverted_foreground) ? img_transform_fgmask : img_transform;

	fftplan_ref = fftw_plan_dft_r2c_2d(nj,ni,single_img_rvec_ptr,reinterpret_cast<fftw_complex*>(img_transform_ptr),FFTW_MEASURE);
	fftplan_inverse_ref = fftw_plan_dft_c2r_2d(nj,ni,reinterpret_cast<fftw_complex*>(img_transform_ptr),single_img_rvec_ptr,FFTW_MEASURE);
	for (i=0; i < npix_conv; i++) single_img_rvec_ptr[i] = 0;

#ifdef USE_STAN

	if (noninverted_foreground) adj_single_img_rvec_fgmask = new double[npix_conv];
	else adj_single_img_rvec = new double[npix_conv];
	double *adj_single_img_rvec_ptr = (noninverted_foreground) ? adj_single_img_rvec_fgmask : adj_single_img_rvec;
	fftw_plan& adj_fftplan_ref = (noninverted_foreground) ? adj_fftplan_fgmask : adj_fftplan;
	fftw_plan& adj_fftplan_inverse_ref = (noninverted_foreground) ? adj_fftplan_inverse_fgmask : adj_fftplan_inverse;
	if (noninverted_foreground) adj_img_transform_fgmask = new complex<double>[ncomplex];
	else adj_img_transform = new complex<double>[ncomplex];
	complex<double> *adj_img_transform_ptr = (noninverted_foreground) ? adj_img_transform_fgmask : adj_img_transform;

	adj_fftplan_ref = fftw_plan_dft_r2c_2d(nj,ni,adj_single_img_rvec_ptr,reinterpret_cast<fftw_complex*>(adj_img_transform_ptr),FFTW_MEASURE);
	adj_fftplan_inverse_ref = fftw_plan_dft_c2r_2d(nj,ni,reinterpret_cast<fftw_complex*>(adj_img_transform_ptr),adj_single_img_rvec_ptr,FFTW_MEASURE);
	for (i=0; i < npix_conv; i++) adj_single_img_rvec_ptr[i] = 0;
#endif

	if ((!noninverted_foreground) and (Lmatrix_n_amps > 0)) {
		Lmatrix_imgs_rvec = new double*[Lmatrix_n_amps];
		Lmatrix_transform = new complex<double>*[Lmatrix_n_amps];
		fftplans_Lmatrix = new fftw_plan[Lmatrix_n_amps];
		fftplans_Lmatrix_inverse = new fftw_plan[Lmatrix_n_amps];
		for (i=0; i < Lmatrix_n_amps; i++) {
			Lmatrix_imgs_rvec[i] = new double[npix_conv];
			Lmatrix_transform[i] = new complex<double>[ncomplex];
			fftplans_Lmatrix[i] = fftw_plan_dft_r2c_2d(nj,ni,Lmatrix_imgs_rvec[i],reinterpret_cast<fftw_complex*>(Lmatrix_transform[i]),FFTW_MEASURE);
			fftplans_Lmatrix_inverse[i] = fftw_plan_dft_c2r_2d(nj,ni,reinterpret_cast<fftw_complex*>(Lmatrix_transform[i]),Lmatrix_imgs_rvec[i],FFTW_MEASURE);
			for (j=0; j < npix_conv; j++) Lmatrix_imgs_rvec[i][j] = 0;
		}
	}
#else
	while (ni < il0) ni *= 2; // need multiple of 2 to do FFT (note, this is only necessary with native code; it is not necessary with FFTW)
	while (nj < jl0) nj *= 2; // need multiple of 2 to do FFT (note, this is only necessary with native code; it is not necessary with FFTW)
	if (noninverted_foreground) psf_zvec_fgmask = new double[2*ni*nj];
	else psf_zvec = new double[2*ni*nj];
	double *psf_zvec_ptr = (noninverted_foreground) ? psf_zvec_fgmask : psf_zvec;
	for (i=0; i < 2*ni*nj; i++) psf_zvec_ptr[i] = 0;
#endif
	int zpsf_i, zpsf_j;
	int l;
	for (i=-nx_half; i < psf_nx - nx_half; i++) {
		for (j=-ny_half; j < psf_ny - ny_half; j++) {
			zpsf_i=i;
			zpsf_j=j;
			if (zpsf_i < 0) zpsf_i += ni;
			if (zpsf_j < 0) zpsf_j += nj;
#ifdef USE_FFTW
			l = zpsf_j*ni + zpsf_i;
			psf_rvec[l] = psf_ptr[nx_half+i][ny_half+j];
#else
			k = 2*(zpsf_j*ni + zpsf_i);
			psf_zvec_ptr[k] = psf_ptr[nx_half+i][ny_half+j];
#endif
		}
	}

#ifdef USE_FFTW
	fftw_execute(fftplan_psf);
	fftw_destroy_plan(fftplan_psf);
	delete[] psf_rvec;
	
	if (noninverted_foreground) psf_transform_conj_fgmask = new complex<double>[ncomplex];
	else psf_transform_conj = new complex<double>[ncomplex];
	complex<double> *psf_transform_conj_ptr = (noninverted_foreground) ? psf_transform_conj_fgmask : psf_transform_conj;
	for (i=0; i < ncomplex; i++) psf_transform_conj_ptr[i] = std::conj(psf_transform_ptr[i]);
#else
	int nnvec[2];
	nnvec[0] = ni;
	nnvec[1] = nj;
	qlens->fourier_transform(psf_zvec_ptr,2,nnvec,1);
#endif
	if (qlens->show_wtime) {
		qlens->wtime = std::chrono::steady_clock::now() - qlens->wtime0;
		if (qlens->mpi_id==0) {
			cout << "Wall time for setting up FFT for convolutions: " << qlens->wtime.count() << endl;
		}
	}

	if (noninverted_foreground) fg_fft_convolution_is_setup = true;
	else fft_convolution_is_setup = true;
	return true;
}

void ImagePixelGrid::cleanup_FFT_convolution_arrays()
{
#ifdef USE_FFTW
	delete[] psf_transform;
	delete[] psf_transform_conj;
	delete[] img_transform;
	delete[] adj_img_transform;
	delete[] single_img_rvec;
	delete[] adj_single_img_rvec;
	psf_transform = NULL;
	psf_transform_conj = NULL;
	img_transform = NULL;
	adj_img_transform = NULL;
	single_img_rvec = NULL;
	adj_single_img_rvec = NULL;
	if (Lmatrix_n_amps > 0) {
		for (int i=0; i < Lmatrix_n_amps; i++) {
			delete[] Lmatrix_imgs_rvec[i];
			delete[] Lmatrix_transform[i];
			fftw_destroy_plan(fftplans_Lmatrix[i]);
			fftw_destroy_plan(fftplans_Lmatrix_inverse[i]);
		}
		delete[] Lmatrix_imgs_rvec;
		delete[] Lmatrix_transform;
		delete[] fftplans_Lmatrix;
		delete[] fftplans_Lmatrix_inverse;
	}
	fftw_destroy_plan(fftplan);
	fftw_destroy_plan(fftplan_inverse);
#ifdef USE_STAN
	fftw_destroy_plan(adj_fftplan);
	fftw_destroy_plan(adj_fftplan_inverse);
#endif

#else
	delete[] psf_zvec;
#endif
	fft_imin=fft_jmin=fft_ni=fft_nj=0;
	fft_convolution_is_setup = false;
}

void ImagePixelGrid::cleanup_foreground_FFT_convolution_arrays()
{
#ifdef USE_FFTW
	delete[] psf_transform_fgmask;
	delete[] psf_transform_conj_fgmask;
	delete[] img_transform_fgmask;
	delete[] adj_img_transform_fgmask;
	delete[] single_img_rvec_fgmask;
	delete[] adj_single_img_rvec_fgmask;
	psf_transform_fgmask = NULL;
	psf_transform_conj_fgmask = NULL;
	img_transform_fgmask = NULL;
	adj_img_transform_fgmask = NULL;
	single_img_rvec_fgmask = NULL;
	adj_single_img_rvec_fgmask = NULL;

	fftw_destroy_plan(fftplan_fgmask);
	fftw_destroy_plan(fftplan_inverse_fgmask);
#ifdef USE_STAN
	fftw_destroy_plan(adj_fftplan_fgmask);
	fftw_destroy_plan(adj_fftplan_inverse_fgmask);
#endif

#else
	delete[] psf_zvec_fgmask;
#endif
	fft_imin_fgmask=fft_jmin_fgmask=fft_ni_fgmask=fft_nj_fgmask=0;
	fg_fft_convolution_is_setup = false;
}

void ImagePixelGrid::PSF_convolution_Lmatrix_dense(const bool verbal)
{
	if ((source_and_lens_n_amps > 0) and (psf != NULL)) {
		if ((qlens->mpi_id==0) and (verbal)) cout << "Beginning PSF convolution (dense)...\n";

		int nx_half, ny_half;
		int psf_nx, psf_ny;
		int npix;
		int *pixel_map_ii, *pixel_map_jj;

		Eigen::MatrixXd *Lptr;
		if (!qlens->psf_supersampling) {
			Lptr = &Lmatrix_trans_dense;
			npix = image_npixels;
			psf_nx = psf->psf_npixels_x;
			psf_ny = psf->psf_npixels_y;
			pixel_map_ii = emask_pixels_i;
			pixel_map_jj = emask_pixels_j;
		} else {
			Lptr = &Lmatrix_trans_supersampled;
			npix = image_n_subpixels;
			psf_nx = psf->supersampled_psf_npixels_x;
			psf_ny = psf->supersampled_psf_npixels_y;
			pixel_map_ii = emask_subpixels_ii;
			pixel_map_jj = emask_subpixels_jj;
		}
		nx_half = psf_nx/2;
		ny_half = psf_ny/2;
		int i,j,ii,jj,k,l,img_index;

		if (qlens->fft_convolution) {
			if (!fft_convolution_is_setup) {
				if (!setup_FFT_convolution(qlens->psf_supersampling,false,qlens->include_fgmask_in_inversion,verbal)) {
					warn("PSF convolution FFT failed");
					return;	
				}
			}

#ifdef USE_FFTW
			int ncomplex = fft_nj*(fft_ni/2+1);
#else
			int nnvec[2];
			nnvec[0] = fft_ni;
			nnvec[1] = fft_nj;
			int nzvec = 2*fft_ni*fft_nj;
			double **Lmatrix_imgs_zvec = new double*[source_and_lens_n_amps];
			for (i=0; i < source_and_lens_n_amps; i++) {
				Lmatrix_imgs_zvec[i] = new double[nzvec];
			}
#endif


			if (qlens->show_wtime) {
				wtime0 = std::chrono::steady_clock::now();
			}
			double rtemp, itemp;
			int npix_conv = fft_ni*fft_nj;
			double *img_zvec, *img_rvec;
			complex<double> *img_cvec;
			int src_index;
			#pragma omp parallel for private(k,i,j,ii,jj,l,img_index,src_index,img_zvec,img_rvec,img_cvec,rtemp,itemp) schedule(static)
			for (src_index=0; src_index < source_and_lens_n_amps; src_index++) {
#ifdef USE_FFTW
				img_rvec = Lmatrix_imgs_rvec[src_index];
				img_cvec = Lmatrix_transform[src_index];
				for (i=0; i < npix_conv; i++) img_rvec[i] = 0;
#else
				img_zvec = Lmatrix_imgs_zvec[src_index];
				for (j=0; j < nzvec; j++) img_zvec[j] = 0;
#endif
				for (img_index=0; img_index < npix; img_index++)
				{
					ii = pixel_map_ii[img_index];
					jj = pixel_map_jj[img_index];
					if (qlens->psf_supersampling) {
						i = image_pixel_i_from_subcell_ii[ii];
						j = image_pixel_j_from_subcell_jj[jj];
					} else {
						i = ii;
						j = jj;
					}
					if ((maps_to_source_pixel[i][j]) and ((pixel_in_mask==NULL) or (pixel_in_mask[i][j]))) {
						ii -= fft_imin;
						jj -= fft_jmin;
#ifdef USE_FFTW
						l = jj*fft_ni + ii;
						//img_rvec[l] = (*Lptr)[img_index][src_index];
						//img_rvec[l] = (*Lptr)(img_index,src_index);
						img_rvec[l] = (*Lptr)(src_index,img_index);
#else
						k = 2*(jj*fft_ni + ii);
						//img_zvec[k] = (*Lptr)[img_index][src_index];
						//img_zvec[k] = (*Lptr)(img_index,src_index);
						img_zvec[k] = (*Lptr)(src_index,img_index);
#endif
					}
				}

#ifdef USE_FFTW
				fftw_execute(fftplans_Lmatrix[src_index]);
				for (i=0; i < ncomplex; i++) {
					img_cvec[i] = img_cvec[i]*psf_transform[i];
					img_cvec[i] /= npix_conv;
				}
				fftw_execute(fftplans_Lmatrix_inverse[src_index]);

#else
				fourier_transform(img_zvec,2,nnvec,1);
				for (i=0,j=0; i < npix_conv; i++, j += 2) {
					rtemp = (img_zvec[j]*psf_zvec[j] - img_zvec[j+1]*psf_zvec[j+1]) / npix_conv;
					itemp = (img_zvec[j]*psf_zvec[j+1] + img_zvec[j+1]*psf_zvec[j]) / npix_conv;
					img_zvec[j] = rtemp;
					img_zvec[j+1] = itemp;
				}
				fourier_transform(img_zvec,2,nnvec,-1);
#endif

				for (img_index=0; img_index < npix; img_index++)
				{
					ii = pixel_map_ii[img_index];
					jj = pixel_map_jj[img_index];
					if (qlens->psf_supersampling) {
						i = image_pixel_i_from_subcell_ii[ii];
						j = image_pixel_j_from_subcell_jj[jj];
					} else {
						i = ii;
						j = jj;
					}
					if ((maps_to_source_pixel[i][j]) and ((pixel_in_mask==NULL) or (pixel_in_mask[i][j]))) {
						ii -= fft_imin;
						jj -= fft_jmin;
#ifdef USE_FFTW
						l = jj*fft_ni + ii;
						//(*Lptr)(img_index,src_index) = img_rvec[l];
						(*Lptr)(src_index,img_index) = img_rvec[l];
#else
						k = 2*(jj*fft_ni + ii);
						//(*Lptr)(img_index,src_index) = img_zvec[k];
						(*Lptr)(src_index,img_index) = img_zvec[k];
#endif
					}
				}
			}
			if (qlens->show_wtime) {
				wtime = std::chrono::steady_clock::now() - wtime0;
				if (qlens->mpi_id==0) {
					cout << "Wall time for calculating PSF convolution of Lmatrix via FFT: "  << wtime.count() << endl;
				}
			}
#ifndef USE_FFTW
			for (i=0; i < source_and_lens_n_amps; i++) delete[] Lmatrix_imgs_zvec[i];
			delete[] Lmatrix_imgs_zvec;
#endif
		} else {
			if (psf->use_input_psf_matrix) {
				if ((!qlens->psf_supersampling) and (psf->psf_matrix == NULL)) return;
				if ((qlens->psf_supersampling) and (psf->supersampled_psf_matrix == NULL)) return;
			}
			else if (psf->generate_PSF_matrix(pixel_xlength,pixel_ylength,qlens->psf_supersampling)==false) return;

			int **pix_index;
			double **psf_ptr;
			int max_nx, max_ny;
			if (!qlens->psf_supersampling) {
				psf_ptr = psf->psf_matrix;
				pix_index = pixel_index;
				max_nx = x_N;
				max_ny = y_N;
			} else {
				psf_ptr = psf->supersampled_psf_matrix;
				pix_index = subpixel_index_ss;
				max_nx = x_N*imgpixel_nsplit;
				max_ny = y_N*imgpixel_nsplit;
			}

			Eigen::MatrixXd Lmatrix_trans_psf = Eigen::MatrixXd::Zero(n_amps,npix);

			if (qlens->show_wtime) {
				wtime0 = std::chrono::steady_clock::now();
			}
			int psf_k, psf_l;
			int img_index1, img_index2, src_index, col_index;
			int index;
			bool new_entry;
			int Lmatrix_psf_nn=0;
			int Lmatrix_psf_nn_part=0;
			double *lmatptr, *lmatpsfptr, psfval;
			#pragma omp parallel for private(k,l,i,j,ii,jj,img_index1,img_index2,src_index,psf_k,psf_l,lmatptr,lmatpsfptr,psfval) schedule(static) reduction(+:Lmatrix_psf_nn_part)
			for (img_index1=0; img_index1 < npix; img_index1++)
			{ // this loops over columns of the PSF blurring matrix
				int col_i=0;
				k = pixel_map_ii[img_index1];
				l = pixel_map_jj[img_index1];
				for (psf_k=0; psf_k < psf_nx; psf_k++) {
					ii = k + nx_half - psf_k; // Note, 'k' is the index for the convolved image, so we have k = i - nx_half + psf_k
					if ((ii >= 0) and (ii < max_nx)) {
						for (psf_l=0; psf_l < psf_ny; psf_l++) {
							jj = l + ny_half - psf_l; // Note, 'l' is the index for the convolved image, so we have l = j - ny_half + psf_l
							if ((jj >= 0) and (jj < max_ny)) {
								if (qlens->psf_supersampling) {
									i = image_pixel_i_from_subcell_ii[ii];
									j = image_pixel_j_from_subcell_jj[jj];
								} else {
									i = ii;
									j = jj;
								}
								if ((maps_to_source_pixel[i][j]) and ((pixel_in_mask==NULL) or (pixel_in_mask[i][j]))) {
									psfval = psf_ptr[psf_k][psf_l];
									img_index2 = pix_index[ii][jj];
									lmatptr = (*Lptr).col(img_index2).data();
									lmatpsfptr = Lmatrix_trans_psf.col(img_index1).data();
									for (src_index=0; src_index < source_and_lens_n_amps; src_index++) {
										//Lmatrix_psf(img_index1,src_index) += psfval*(*Lptr)(img_index2,src_index);
										//Lmatrix_trans_psf(src_index,img_index1) += psfval*(*Lptr)(src_index,img_index2);
										(*(lmatpsfptr++)) += psfval*(*(lmatptr++));
									}
								}
							}
						}
					}
				}
			}

			// note, the following function sets the pointer in Lmatrix_dense to Lmatrix_trans_psf (and deletes the old memory allocation), so no garbage collection necessary afterwards
			(*Lptr) = std::move(Lmatrix_trans_psf);


			if (qlens->show_wtime) {
				wtime = std::chrono::steady_clock::now() - wtime0;
				if (qlens->mpi_id==0) cout << "Wall time for calculating dense PSF convolution of Lmatrix: "  << wtime.count() << endl;
			}
		}
		if (qlens->psf_supersampling) average_supersampled_dense_Lmatrix();

		if (qlens->include_fgmask_in_inversion) {
			// make sure it doesn't try to fit to the "padded" pixels around the borders of the foreground mask, since those are only there for the PSF convolution
			int i,j,k,img_index;
			for (img_index=0; img_index < image_npixels; img_index++) {
				i = emask_pixels_i[img_index];
				j = emask_pixels_j[img_index];
				if (!image_data->foreground_mask_data[i][j]) {
					for (k=0; k < n_amps; k++) {
						//Lmatrix_dense0[img_index][k] = 0; // pixels that were only used for padding for PSF convolution should not be used for the fit itself
						//Lmatrix_dense(img_index,k) = 0; // pixels that were only used for padding for PSF convolution should not be used for the fit itself
						Lmatrix_trans_dense(k,img_index) = 0; // pixels that were only used for padding for PSF convolution should not be used for the fit itself
					}
				}
			}
		}
	}

	if (qlens->include_imgfluxes_in_inversion) {
		int i,j,k;
		//double *Lmatptr;
		i=0;
		int src_amp_i;
		for (j=0; j < qlens->n_ptsrc; j++) {
			for (k=0; k < qlens->ptsrc_list[j]->images.size(); k++) {
				//Lmatptr = Lmatrix_transpose_ptimg_amps.row(i).data();
				// NOTE: we cannot pass a pointer in directly from Lmatrix_transpose_dense because it is in column-major format (whereas Lmatrix_transpose_ptimg_amps is row-major)
				generate_point_images(qlens->ptsrc_list[j]->images, point_image_surface_brightness.data(), false, -1, k);
				src_amp_i = source_and_lens_n_amps + i;
				for (int img_index=0; img_index < image_npixels; img_index++) {
					Lmatrix_trans_dense(src_amp_i,img_index) = point_image_surface_brightness[img_index];
				}
				i++;
			}
		}
		//double *Lmatrix_transpose_line;
		//i=0;
		//for (j=0; j < qlens->n_ptsrc; j++) {
			//for (k=0; k < qlens->ptsrc_list[j]->images.size(); k++) {
				//src_amp_i = source_and_lens_n_amps + i;
				//Lmatrix_transpose_line = Lmatrix_transpose_ptimg_amps.row(i).data();
				//for (int img_index=0; img_index < image_npixels; img_index++) {
					//Lmatrix_trans_dense(src_amp_i,img_index) = Lmatrix_transpose_line[img_index];
				//}
				//i++;
			//}
		//}
	} else if (qlens->include_srcflux_in_inversion) {
		int j,k;
		//double *Lmatptr;
		int src_amp_i;
		for (j=0; j < qlens->n_ptsrc; j++) {
			//Lmatptr = Lmatrix_transpose_ptimg_amps.row(j).data();
			generate_point_images(qlens->ptsrc_list[j]->images, point_image_surface_brightness.data(), false, 1.0);
			src_amp_i = source_and_lens_n_amps + j;
			for (int img_index=0; img_index < image_npixels; img_index++) {
				Lmatrix_trans_dense(src_amp_i,img_index) = point_image_surface_brightness[img_index];
			}
		}
		//double *Lmatrix_transpose_line;
		//for (j=0; j < qlens->n_ptsrc; j++) {
			//src_amp_i = source_and_lens_n_amps + j;
			//Lmatrix_transpose_line = Lmatrix_transpose_ptimg_amps.row(j).data();
			//for (int img_index=0; img_index < image_npixels; img_index++) {
				//Lmatrix_trans_dense(src_amp_i,img_index) = Lmatrix_transpose_line[img_index];
			//}
		//}
	}
}

template <typename MathTypes>
void ImagePixelGrid::PSF_convolution_pixel_vector_wrapper(const bool foreground, const bool verbal, const bool use_fft, const bool use_extended_mask)
{
	using VecType = typename MathTypes::VecType;
	ImgGrid_Params<MathTypes>& p = assign_imggrid_param_object<MathTypes>();
	if (psf==NULL) return;
	if ((use_fft) and (use_extended_mask)) die("PSF convolution is not setup with FFT for emask");
	if (psf->use_input_psf_matrix) {
		if (!qlens->psf_supersampling) {
			if (psf->psf_matrix == NULL) {
				if (verbal) warn("could not find input PSF matrix");
				return;
			}
		} else {
			if (psf->supersampled_psf_matrix == NULL) {
				if (verbal) warn("could not find input supersampled PSF matrix");
				average_supersampled_image_surface_brightness(); 
				return;
			}
		}
	} else {
		if ((psf->psf_params.psf_width_x==0) and (psf->psf_params.psf_width_y==0)) {
			if (qlens->psf_supersampling) average_supersampled_image_surface_brightness(); // no PSF to convolve
			return;
		}
		else if (psf->generate_PSF_matrix(pixel_xlength,pixel_ylength,qlens->psf_supersampling)==false) {
			if (verbal) warn("could not generate_PSF matrix");
			return;
		}
		if (qlens->mpi_id==0) cout << "generated PSF matrix" << endl;
	}

	if ((qlens->fft_convolution) and (use_fft)) {
		bool fft_is_already_setup = false;
		if ((foreground) and (fg_fft_convolution_is_setup)) fft_is_already_setup = true;
		else if ((!foreground) and (fft_convolution_is_setup)) fft_is_already_setup = true;
		if (!fft_is_already_setup) {
			if (!setup_FFT_convolution(qlens->psf_supersampling,foreground,qlens->include_fgmask_in_inversion,verbal)) {
				warn("PSF convolution FFT setup failed");
				return;	
			}
		}
	}

	if ((qlens->mpi_id==0) and (verbal)) cout << "Beginning PSF convolution...\n";
	VecType *surface_brightness_vector;
	if (foreground) surface_brightness_vector = &p.sbprofile_surface_brightness;
	else {
		if (qlens->psf_supersampling) surface_brightness_vector = &p.image_surface_brightness_supersampled;
		else {
			if (!use_extended_mask) surface_brightness_vector = &p.image_surface_brightness;
			else surface_brightness_vector = &p.image_surface_brightness_emask;
		}	
	}
	if ((qlens->fft_convolution) and (use_fft)) {
		(*surface_brightness_vector) = PSF_convolution_pixel_vector_stan_FFT((*surface_brightness_vector),foreground,verbal);
	} else {
		(*surface_brightness_vector) = PSF_convolution_pixel_vector_stan((*surface_brightness_vector),foreground); // implement emask!
	}
}
template void ImagePixelGrid::PSF_convolution_pixel_vector_wrapper<PlainTypes>(const bool foreground, const bool verbal, const bool use_fft, const bool use_extended_mask);
#ifdef USE_STAN
template void ImagePixelGrid::PSF_convolution_pixel_vector_wrapper<VarmatTypes>(const bool foreground, const bool verbal, const bool use_fft, const bool use_extended_mask);
#endif

void ImagePixelGrid::PSF_convolution_pixel_vector(const bool foreground, const bool verbal, const bool use_fft, const bool use_extended_mask)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	if (psf==NULL) return;
	if ((use_fft) and (use_extended_mask)) die("PSF convolution is not setup with FFT for emask");
	if (psf->use_input_psf_matrix) {
		if (!qlens->psf_supersampling) {
			if (psf->psf_matrix == NULL) {
				if (verbal) warn("could not find input PSF matrix");
				return;
			}
		} else {
			if (psf->supersampled_psf_matrix == NULL) {
				if (verbal) warn("could not find input supersampled PSF matrix");
				average_supersampled_image_surface_brightness(); 
				return;
			}
		}
	} else {
		if ((psf->psf_params.psf_width_x==0) and (psf->psf_params.psf_width_y==0)) {
			if (qlens->psf_supersampling) average_supersampled_image_surface_brightness(); // no PSF to convolve
			return;
		}
		else if (psf->generate_PSF_matrix(pixel_xlength,pixel_ylength,qlens->psf_supersampling)==false) {
			if (verbal) warn("could not generate_PSF matrix");
			return;
		}
		if (qlens->mpi_id==0) cout << "generated PSF matrix" << endl;
	}

	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	if ((qlens->mpi_id==0) and (verbal)) cout << "Beginning PSF convolution...\n";
	double *surface_brightness_vector;
	if (foreground) surface_brightness_vector = p.sbprofile_surface_brightness.data();
	else {
		if (qlens->psf_supersampling) surface_brightness_vector = p.image_surface_brightness_supersampled.data();
		else {
			if (!use_extended_mask) surface_brightness_vector = p.image_surface_brightness.data();
			else surface_brightness_vector = p.image_surface_brightness_emask.data();
		}	
	}

	int nx_half, ny_half;
	int psf_nx, psf_ny;

	if (!qlens->psf_supersampling) {
		psf_nx = psf->psf_npixels_x;
		psf_ny = psf->psf_npixels_y;
	} else {
		psf_nx = psf->supersampled_psf_npixels_x;
		psf_ny = psf->supersampled_psf_npixels_y;
	}
	nx_half = psf_nx/2;
	ny_half = psf_ny/2;

	int npix;
	int *pixel_map_ii, *pixel_map_jj;
	if (foreground) {
		//cout << "using fgmask" << endl;
		npix = image_npixels_fgmask;
		pixel_map_ii = fgmask_pixels_i;
		pixel_map_jj = fgmask_pixels_j;
	} else {
		//cout << "NOT using fgmask" << endl;
		if (!qlens->psf_supersampling) {
			if (!use_extended_mask) npix = image_npixels;
			else npix = image_npixels_emask;
			pixel_map_ii = emask_pixels_i;
			pixel_map_jj = emask_pixels_j;
		} else {
			npix = image_n_subpixels;
			pixel_map_ii = emask_subpixels_ii;
			pixel_map_jj = emask_subpixels_jj;
		}
	}

	bool **selected_mask;
	if (foreground) {
		if ((image_data==NULL) or (fgmask==NULL)) selected_mask = NULL;
		else selected_mask = fgmask;
	} else {
		if (use_extended_mask) {
			if ((image_data==NULL) or (emask==NULL)) selected_mask = NULL;
			else selected_mask = emask;
		} else {
			if (pixel_in_mask==NULL) selected_mask = NULL;
			else selected_mask = pixel_in_mask;
		}
	}

	int i,j,ii,jj,k,img_index;
	if ((qlens->fft_convolution) and (use_fft)) {
		bool fft_is_already_setup = false;
		if ((foreground) and (fg_fft_convolution_is_setup)) fft_is_already_setup = true;
		else if ((!foreground) and (fft_convolution_is_setup)) fft_is_already_setup = true;
		if (!fft_is_already_setup) {
			if (!setup_FFT_convolution(qlens->psf_supersampling,foreground,qlens->include_fgmask_in_inversion,verbal)) {
				warn("PSF convolution FFT failed");
				return;	
			}
		}

		int &ni = (foreground) ? fft_ni_fgmask : fft_ni;
		int &nj = (foreground) ? fft_nj_fgmask : fft_nj;
		int &imin = (foreground) ? fft_imin_fgmask : fft_imin;
		int &jmin = (foreground) ? fft_jmin_fgmask : fft_jmin;

#ifdef USE_FFTW
		complex<double> *psf_transform_ptr = (foreground) ? psf_transform_fgmask : psf_transform;
		double *single_img_rvec_ptr = (foreground) ? single_img_rvec_fgmask : single_img_rvec;
		complex<double> *img_transform_ptr = (foreground) ? img_transform_fgmask : img_transform;

		int ncomplex = nj*(ni/2+1);
		int npix_conv = ni*nj;
		for (i=0; i < npix_conv; i++) single_img_rvec_ptr[i] = 0;
#else
		double *psf_zvec_ptr = (foreground) ? psf_zvec_fgmask : psf_zvec;

		int nzvec = 2*ni*nj;
		double *img_zvec = new double[nzvec];
#endif

		if (qlens->show_wtime) {
			wtime0 = std::chrono::steady_clock::now();
		}

		int l;
#ifndef USE_FFTW
		for (i=0; i < nzvec; i++) img_zvec[i] = 0;
#endif
		for (img_index=0; img_index < npix; img_index++)
		{
			ii = pixel_map_ii[img_index];
			jj = pixel_map_jj[img_index];
			if (qlens->psf_supersampling) {
				i = image_pixel_i_from_subcell_ii[ii];
				j = image_pixel_j_from_subcell_jj[jj];
			} else {
				i = ii;
				j = jj;
			}
			if ((selected_mask==NULL) or (selected_mask[i][j])) {
				ii -= imin;
				jj -= jmin;
#ifdef USE_FFTW
				l = jj*ni + ii;
				single_img_rvec_ptr[l] = surface_brightness_vector[img_index];
#else
				k = 2*(jj*ni + ii);
				img_zvec[k] = surface_brightness_vector[img_index];
#endif
			//} else {
				//cout << "WTF?" << endl;
				//if (!maps_to_source_pixel[i][j]) cout << "DOESNT MAP" << endl;
				//die();
			}
		}
		if (qlens->show_wtime) {
			wtime = std::chrono::steady_clock::now() - wtime0;
			if (qlens->mpi_id==0) {
				cout << "Wall time for PSF convolution via FFT: "  << wtime.count() << endl;
			}
			wtime0 = std::chrono::steady_clock::now();
		}

#ifdef USE_FFTW
		fftw_plan& fftplan_ref = (foreground) ? fftplan_fgmask : fftplan;
		fftw_plan& fftplan_inverse_ref = (foreground) ? fftplan_inverse_fgmask : fftplan_inverse;

		fftw_execute(fftplan_ref);
		for (i=0; i < ncomplex; i++) {
			img_transform_ptr[i] = img_transform_ptr[i]*psf_transform_ptr[i];
			img_transform_ptr[i] /= npix_conv;
		}
		fftw_execute(fftplan_inverse_ref);
#else
		int nnvec[2];
		nnvec[0] = ni;
		nnvec[1] = nj;
		fourier_transform(img_zvec,2,nnvec,1);

		double rtemp, itemp;
		for (i=0,j=0; i < (ni*nj); i++, j += 2) {
			rtemp = (img_zvec[j]*psf_zvec_ptr[j] - img_zvec[j+1]*psf_zvec_ptr[j+1]) / (ni*nj);
			itemp = (img_zvec[j]*psf_zvec_ptr[j+1] + img_zvec[j+1]*psf_zvec_ptr[j]) / (ni*nj);
			img_zvec[j] = rtemp;
			img_zvec[j+1] = itemp;
		}
		fourier_transform(img_zvec,2,nnvec,-1);
#endif

		for (img_index=0; img_index < npix; img_index++)
		{
			ii = pixel_map_ii[img_index];
			jj = pixel_map_jj[img_index];
			if (qlens->psf_supersampling) {
				i = image_pixel_i_from_subcell_ii[ii];
				j = image_pixel_j_from_subcell_jj[jj];
			} else {
				i = ii;
				j = jj;
			}
			if ((selected_mask==NULL) or (selected_mask[i][j])) {
				ii -= imin;
				jj -= jmin;
#ifdef USE_FFTW
				l = jj*ni + ii;
				surface_brightness_vector[img_index] = single_img_rvec_ptr[l];
#else
				k = 2*(jj*ni + ii);
				surface_brightness_vector[img_index] = img_zvec[k];
#endif
			//} else {
				//cout << "HARG?" << endl;
			}
		}
#ifndef USE_FFTW
		delete[] img_zvec;
#endif
	} else {
		int **pix_index;
		double **psf_ptr;
		int max_nx, max_ny;

		if (!qlens->psf_supersampling) {
			psf_ptr = psf->psf_matrix;
			psf_nx = psf->psf_npixels_x;
			psf_ny = psf->psf_npixels_y;
		} else {
			psf_ptr = psf->supersampled_psf_matrix;
			psf_nx = psf->supersampled_psf_npixels_x;
			psf_ny = psf->supersampled_psf_npixels_y;
		}

		if (foreground) {
			pix_index = pixel_index_fgmask;
			max_nx = x_N;
			max_ny = y_N;
		} else {
			if (!qlens->psf_supersampling) {
				pix_index = pixel_index;
				max_nx = x_N;
				max_ny = y_N;
			} else {
				pix_index = subpixel_index_ss;
				max_nx = x_N*imgpixel_nsplit;
				max_ny = y_N*imgpixel_nsplit;
			}
		}
		//cout << "npix=" << npix << " psf_nx=" << psf_nx << " psf_ny=" << psf_ny << " nxhalf=" << nx_half << " nyhalf=" << ny_half << endl;

		double *new_surface_brightness_vector = new double[npix];
		int l;
		int psf_k, psf_l;
		int img_index2;
		#pragma omp parallel for private(k,l,i,j,ii,jj,img_index,img_index2,psf_k,psf_l) schedule(static)
		for (img_index=0; img_index < npix; img_index++)
		{ // this loops over columns of the PSF blurring matrix
			new_surface_brightness_vector[img_index] = 0;
			k = pixel_map_ii[img_index];
			l = pixel_map_jj[img_index];
			for (psf_k=0; psf_k < psf_nx; psf_k++) {
				ii = k + nx_half - psf_k; // Note, 'k' is the index for the convolved image, so we have k = i - nx_half + psf_k
				if ((ii >= 0) and (ii < max_nx)) {
					for (psf_l=0; psf_l < psf_ny; psf_l++) {
						jj = l + ny_half - psf_l; // Note, 'l' is the index for the convolved image, so we have l = j - ny_half + psf_l
						if ((jj >= 0) and (jj < max_ny)) {
							if (qlens->psf_supersampling) {
								i = image_pixel_i_from_subcell_ii[ii];
								j = image_pixel_j_from_subcell_jj[jj];
							} else {
								i = ii;
								j = jj;
							}
							if ((selected_mask==NULL) or (selected_mask[i][j])) {
								img_index2 = pix_index[ii][jj];
								new_surface_brightness_vector[img_index] += psf_ptr[psf_k][psf_l]*surface_brightness_vector[img_index2];
							}
						}
					}
				}
			}
		}

		if (qlens->show_wtime) {
			wtime = std::chrono::steady_clock::now() - wtime0;
			if (qlens->mpi_id==0) {
				if (foreground) cout << "Wall time for calculating PSF convolution of foreground: "  << wtime.count() << endl;
				else cout << "Wall time for calculating PSF convolution of image: "  << wtime.count() << endl;
			}
		}
		for (int i=0; i < npix; i++) {
			surface_brightness_vector[i] = new_surface_brightness_vector[i];
			//cout << surface_brightness_vector[i] << endl;
		}
		delete[] new_surface_brightness_vector;
	}
	if (qlens->psf_supersampling) average_supersampled_image_surface_brightness();
}

template <typename VecType>
VecType ImagePixelGrid::PSF_convolution_pixel_vector_stan_FFT(const VecType& sbvec, const bool foreground, const bool verbal)
{
	int npix = sbvec.size();
#ifdef USE_STAN
	stan::arena_t<Eigen::VectorXd> out(npix);
	const auto& sbvec_val = sbvec.val();
#else
	Eigen::VectorXd out(npix);
	const auto& sbvec_val = sbvec;
#endif

	int *pixel_map_ii, *pixel_map_jj;
	if (foreground) {
		//cout << "using fgmask" << endl;
		npix = image_npixels_fgmask;
		pixel_map_ii = fgmask_pixels_i;
		pixel_map_jj = fgmask_pixels_j;
	} else {
		//cout << "NOT using fgmask" << endl;
		if (!qlens->psf_supersampling) {
			npix = image_npixels_emask;
			pixel_map_ii = emask_pixels_i;
			pixel_map_jj = emask_pixels_j;
		} else {
			npix = image_n_subpixels;
			pixel_map_ii = emask_subpixels_ii;
			pixel_map_jj = emask_subpixels_jj;
		}
	}

	bool **selected_mask;
	if (foreground) {
		if ((image_data==NULL) or (fgmask==NULL)) selected_mask = NULL;
		else selected_mask = fgmask;
	} else {
		if (pixel_in_mask==NULL) selected_mask = NULL;
		else selected_mask = pixel_in_mask;
	}

	int i,j,ii,jj,k,img_index;
	bool fft_is_already_setup = false;
	if ((foreground) and (fg_fft_convolution_is_setup)) fft_is_already_setup = true;
	else if ((!foreground) and (fft_convolution_is_setup)) fft_is_already_setup = true;
	if (!fft_is_already_setup) die("FFT not setup");

	int &ni = (foreground) ? fft_ni_fgmask : fft_ni;
	int &nj = (foreground) ? fft_nj_fgmask : fft_nj;
	int &imin = (foreground) ? fft_imin_fgmask : fft_imin;
	int &jmin = (foreground) ? fft_jmin_fgmask : fft_jmin;

	complex<double> *psf_transform_ptr = (foreground) ? psf_transform_fgmask : psf_transform;
	complex<double> *psf_transform_conj_ptr = (foreground) ? psf_transform_conj_fgmask : psf_transform_conj;
	double *single_img_rvec_ptr = (foreground) ? single_img_rvec_fgmask : single_img_rvec;
	complex<double> *img_transform_ptr = (foreground) ? img_transform_fgmask : img_transform;
#ifdef USE_STAN
	double *adj_single_img_rvec_ptr = (foreground) ? adj_single_img_rvec_fgmask : adj_single_img_rvec;
	complex<double> *adj_img_transform_ptr = (foreground) ? adj_img_transform_fgmask : adj_img_transform;
#endif

	int ncomplex = nj*(ni/2+1);
	int npix_conv = ni*nj;
	for (i=0; i < npix_conv; i++) single_img_rvec_ptr[i] = 0;
#ifdef USE_STAN
	for (i=0; i < npix_conv; i++) adj_single_img_rvec_ptr[i] = 0;
#endif

	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	int l;
	for (img_index=0; img_index < npix; img_index++)
	{
		ii = pixel_map_ii[img_index];
		jj = pixel_map_jj[img_index];
		if (qlens->psf_supersampling) {
			i = image_pixel_i_from_subcell_ii[ii];
			j = image_pixel_j_from_subcell_jj[jj];
		} else {
			i = ii;
			j = jj;
		}
		if ((selected_mask==NULL) or (selected_mask[i][j])) {
			ii -= imin;
			jj -= jmin;
			l = jj*ni + ii;
			single_img_rvec_ptr[l] = sbvec_val[img_index];
		}
	}
	fftw_plan& fftplan_ref = (foreground) ? fftplan_fgmask : fftplan;
	fftw_plan& fftplan_inverse_ref = (foreground) ? fftplan_inverse_fgmask : fftplan_inverse;

	fftw_execute(fftplan_ref);
	for (i=0; i < ncomplex; i++) {
		img_transform_ptr[i] = img_transform_ptr[i]*psf_transform_ptr[i];
		img_transform_ptr[i] /= npix_conv;
	}
	fftw_execute(fftplan_inverse_ref);

	for (img_index=0; img_index < npix; img_index++)
	{
		ii = pixel_map_ii[img_index];
		jj = pixel_map_jj[img_index];
		if (qlens->psf_supersampling) {
			i = image_pixel_i_from_subcell_ii[ii];
			j = image_pixel_j_from_subcell_jj[jj];
		} else {
			i = ii;
			j = jj;
		}
		if ((selected_mask==NULL) or (selected_mask[i][j])) {
			ii -= imin;
			jj -= jmin;
			l = jj*ni + ii;
			out[img_index] = single_img_rvec_ptr[l];
		}
	}
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) {
			cout << "Wall time for PSF convolution via FFT: "  << wtime.count() << endl;
		}
	}

	if (qlens->psf_supersampling) average_supersampled_image_surface_brightness();

#ifdef USE_STAN
	if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>)
	{
		return stan::math::make_callback_var(out, [this,sbvec,npix,npix_conv,ni,nj,imin,jmin,ncomplex,pixel_map_ii,pixel_map_jj,selected_mask,psf_transform_conj_ptr,adj_single_img_rvec_ptr,adj_img_transform_ptr](const auto& res) mutable {
			const auto& out_adj = res.adj();
			auto& sb_adj = sbvec.adj();

			std::fill(adj_single_img_rvec_ptr, adj_single_img_rvec_ptr + npix_conv, 0.0);

			for (int img_index = 0; img_index < npix; ++img_index)
			{
				int ii = pixel_map_ii[img_index];
				int jj = pixel_map_jj[img_index];

				int i, j;

				if (qlens->psf_supersampling) {
					i = image_pixel_i_from_subcell_ii[ii];
					j = image_pixel_j_from_subcell_jj[jj];
				} else {
					i = ii;
					j = jj;
				}

				if ((selected_mask == NULL) || selected_mask[i][j])
				{
					int ii0 = ii - imin;
					int jj0 = jj - jmin;

					int l = jj0 * ni + ii0;

					adj_single_img_rvec_ptr[l] = out_adj[img_index];
				}
			}

			fftw_execute(adj_fftplan); // FFT of adjoint image

			for (int k = 0; k < ncomplex; ++k)
			{
				adj_img_transform_ptr[k] *= psf_transform_conj_ptr[k]; // multiply by complex conjugate of PSF transform
				adj_img_transform_ptr[k] /= npix_conv;
			}

			fftw_execute(adj_fftplan_inverse); // inverse FFT -> back to pixel domain

			for (int img_index = 0; img_index < npix; ++img_index)
			{
				int ii = pixel_map_ii[img_index];
				int jj = pixel_map_jj[img_index];

				int i, j;

				if (qlens->psf_supersampling) {
					i = image_pixel_i_from_subcell_ii[ii];
					j = image_pixel_j_from_subcell_jj[jj];
				} else {
					i = ii;
					j = jj;
				}

				if ((selected_mask == NULL) || selected_mask[i][j])
				{
					int ii0 = ii - imin;
					int jj0 = jj - jmin;

					int l = jj0 * ni + ii0;

					sb_adj[img_index] += adj_single_img_rvec_ptr[l];
				}
			}
		});
	} else
#endif
	return out;
}
template Eigen::VectorXd ImagePixelGrid::PSF_convolution_pixel_vector_stan_FFT<Eigen::VectorXd>(const Eigen::VectorXd& sbvec, const bool foreground, const bool verbal);
#ifdef USE_STAN
template stan::math::var_value<Eigen::VectorXd> ImagePixelGrid::PSF_convolution_pixel_vector_stan_FFT<stan::math::var_value<Eigen::VectorXd>>(const stan::math::var_value<Eigen::VectorXd>& sbvec, const bool foreground, const bool verbal);
#endif

void ImagePixelGrid::reset_psfconv_plans()
{
	psfconv_plan.clear();
	psfconv_plan_fg.clear();
	psf_convolution_is_setup = false;
	fg_psf_convolution_is_setup = false;
}

void ImagePixelGrid::setup_PSF_convolution(const bool foreground)
{
	if ((!foreground) and (psf_convolution_is_setup)) return; // convolution already setup
	if ((foreground) and (fg_psf_convolution_is_setup)) return; // fg convolution already setup
	//cout << "Setting up PSF convolution..." << endl;

	int **pix_index;
	double **psf_ptr;
	ConvPlan* conv_plan;
	int max_nx, max_ny;

	if (foreground) conv_plan = &this->psfconv_plan_fg;
	else conv_plan = &this->psfconv_plan;

	int psf_nx, psf_ny;
	if ((qlens==NULL) or (!qlens->psf_supersampling)) {
		psf_ptr = psf->psf_matrix;
		psf_nx = psf->psf_npixels_x;
		psf_ny = psf->psf_npixels_y;
	} else {
		psf_ptr = psf->supersampled_psf_matrix;
		psf_nx = psf->supersampled_psf_npixels_x;
		psf_ny = psf->supersampled_psf_npixels_y;
	}
	int nx_half, ny_half;
	nx_half = psf_nx/2;
	ny_half = psf_ny/2;

	if (foreground) {
		pix_index = pixel_index_fgmask;
		max_nx = x_N;
		max_ny = y_N;
	} else {
		if ((qlens==NULL) or (!qlens->psf_supersampling)) {
			pix_index = pixel_index;
			max_nx = x_N;
			max_ny = y_N;
		} else {
			pix_index = subpixel_index_ss;
			max_nx = x_N*imgpixel_nsplit;
			max_ny = y_N*imgpixel_nsplit;
		}
	}

	int npix;
	int *pixel_map_ii, *pixel_map_jj;
	if (foreground) {
		npix = image_npixels_fgmask; // change it so this gets stored in image_pixel_grid
		pixel_map_ii = fgmask_pixels_i;
		pixel_map_jj = fgmask_pixels_j;
	} else {
		//cout << "NOT using fgmask" << endl;
		if ((qlens==NULL) or (!qlens->psf_supersampling)) {
			npix = image_npixels; // change it so this gets stored in image_pixel_grid
			pixel_map_ii = emask_pixels_i;
			pixel_map_jj = emask_pixels_j;
		} else {
			npix = image_n_subpixels; // change it so this gets stored in image_pixel_grid->image_pixel_grid
			pixel_map_ii = emask_subpixels_ii;
			pixel_map_jj = emask_subpixels_jj;
		}
	}

	conv_plan->clear();
	conv_plan->resize(npix,npix); // eventually, allow for option where input mask may be "padded", i.e. bigger from output mask

	int psf_i,psf_j,k,l;
	int i,j;

	conv_plan->offsets.push_back(0);
	int img_index2; // this is the index for the unconvolved surface_brightness
	double w;
	for (int img_index=0; img_index < npix; img_index++)
	{ // this loops over columns of the PSF blurring matrix
		k = pixel_map_ii[img_index];
		l = pixel_map_jj[img_index];
		for (psf_j=0; psf_j < psf_ny; psf_j++) {
			j = l + ny_half - psf_j;
			if ((j < 0) or (j >= max_ny)) continue;
			for (psf_i=0; psf_i < psf_nx; psf_i++) {
				i = k + nx_half - psf_i;
				if ((i < 0) or (i >= max_nx)) continue;
				if ((pixel_in_mask != NULL) and (!pixel_in_mask[i][j])) continue;
				//if ((i == k) and (j == l)) w = 1; // should amount to no convolution at all
				//else w = 0;
				w = psf_ptr[psf_i][psf_j];
				//cout << "PSF weight at (" << psf_i << " " << psf_j << "): " << w << endl;

				img_index2 = pix_index[i][j];
				if (img_index2 > npix) die("invalid index at (%i,%i) gives idx=%i",i,j,img_index2);
				if (img_index2 < 0) die("undefined pixel index at (%i,%i)",i,j);
				conv_plan->in_idx.push_back(img_index2);
				//psfconv_plan.weight.push_back(psf_ptr[psf_i][psf_j]);
				conv_plan->weight.push_back(w);
			}
		}
		conv_plan->offsets.push_back(conv_plan->in_idx.size());
	}
	if (foreground) fg_psf_convolution_is_setup = true;
	else psf_convolution_is_setup = true;
}

template <typename VecType>
VecType ImagePixelGrid::PSF_convolution_pixel_vector_stan(const VecType& sbvec, const bool foreground)
{
	ConvPlan* conv_plan;
	if (foreground) conv_plan = &this->psfconv_plan_fg;
	else conv_plan = &this->psfconv_plan;
#ifdef USE_STAN
	stan::arena_t<Eigen::VectorXd> out(conv_plan->out_size);
	const auto& sbvec_val = sbvec.val();
#else
	Eigen::VectorXd out(conv_plan->out_size);
	const auto& sbvec_val = sbvec;
#endif

	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}

	// Forward pass
#ifdef USE_TBB
	tbb::parallel_for(tbb::blocked_range<int>(0, conv_plan->out_size), [&](const auto& r)
	{
		for (int o = r.begin(); o != r.end(); o++) {
			double sum = 0.0;
			const int begin = conv_plan->offsets[o];
			const int end   = conv_plan->offsets[o + 1];

			for (int e = begin; e < end; ++e) {
				sum += sbvec_val(conv_plan->in_idx[e]) * conv_plan->weight[e];
			}
			out[o] = sum;
		}
	});
#else
	for (int o = 0; o < conv_plan->out_size; o++) {
		double sum = 0.0;
		const int begin = conv_plan->offsets[o];
		const int end   = conv_plan->offsets[o + 1];

		for (int e = begin; e < end; ++e) {
			//cout << "idx=" << conv_plan->in_idx[e] << ", w=" << conv_plan->weight[e] << "..." << endl;
			if (conv_plan->in_idx[e] > sbvec_val.size()) die("invalid index! sbvec size=%i",sbvec_val.size());
			sum += sbvec_val[conv_plan->in_idx[e]] * conv_plan->weight[e];
			//cout << "psf_weight=" << conv_plan->weight[e] << " sbval=" << sbvec_val[conv_plan->in_idx[e]] << " idx=" << conv_plan->in_idx[e] << endl;
		}
		out[o] = sum;
	}
#endif

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) {
			cout << "Wall time for PSF convolution: "  << wtime.count() << endl;
		}
	}

#ifdef USE_STAN
	if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
		return stan::math::make_callback_var(out, [this, foreground, sbvec](const auto& res) mutable {
			const auto& plan = (foreground) ? this->psfconv_plan_fg : this->psfconv_plan;
			const auto& out_adj = res.adj();
			const int a_size = plan.in_size;

#ifdef USE_TBB
			// thread-local adjoint buffers
			tbb::enumerable_thread_specific<Eigen::VectorXd> local_adj([&] {
				return Eigen::VectorXd::Zero(a_size);
			});

			// parallel reverse pass
			tbb::parallel_for(tbb::blocked_range<int>(0, plan.out_size), [&](const auto& r)
			{
				auto& adj = local_adj.local();

				for (int o = r.begin(); o != r.end(); o++) {
					const double chain = out_adj[o];
					const int begin = plan.offsets[o];
					const int end   = plan.offsets[o + 1];

					for (int e = begin; e < end; ++e) {
						adj[plan.in_idx[e]] += chain * plan.weight[e];
					}
				}
			});

			// reduction step
			auto& sb_adj = sbvec.adj();

			for (auto& local : local_adj) {
				sb_adj += local;
			}
#else
			// serial reverse pass
			auto& sb_adj = sbvec.adj();

			for (int o = 0; o < plan.out_size; o++) {
				const double chain = out_adj[o];
				const int begin = plan.offsets[o];
				const int end   = plan.offsets[o + 1];

				for (int e = begin; e < end; ++e) {
					sb_adj[plan.in_idx[e]] += chain * plan.weight[e];
				}
			}
#endif
		});
	} else
#endif
	return out;
}
template Eigen::VectorXd ImagePixelGrid::PSF_convolution_pixel_vector_stan<Eigen::VectorXd>(const Eigen::VectorXd& sbvec, const bool foreground);
#ifdef USE_STAN
template stan::math::var_value<Eigen::VectorXd> ImagePixelGrid::PSF_convolution_pixel_vector_stan<stan::math::var_value<Eigen::VectorXd>>(const stan::math::var_value<Eigen::VectorXd>& sbvec, const bool foreground);
#endif

void ImagePixelGrid::average_supersampled_image_surface_brightness()
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	// now average the subpixel surface brightnesses to get the new image surface brightness
	int nsubpix = imgpixel_nsplit*imgpixel_nsplit;
	int i, j, img_index;
	p.image_surface_brightness = Eigen::VectorXd::Zero(image_npixels);
	for (img_index=0; img_index < image_n_subpixels; img_index++) {
		i = mask_subpixel_i[img_index];
		j = mask_subpixel_j[img_index];
		p.image_surface_brightness[pixel_index[i][j]] += p.image_surface_brightness_supersampled[img_index];
	}
	for (i=0; i < image_npixels; i++) p.image_surface_brightness[i] /= nsubpix;
}

void ImagePixelGrid::average_supersampled_dense_Lmatrix()
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	int i,j,k;

	int nsubpix = imgpixel_nsplit*imgpixel_nsplit;
	int img_index;
	double *lmatptr, *lmatsup_ptr;
	for (i=0; i < image_npixels; i++) p.image_surface_brightness[i] = 0;
	for (img_index=0; img_index < image_n_subpixels; img_index++) {
		i = mask_subpixel_i[img_index];
		j = mask_subpixel_j[img_index];
		lmatptr = Lmatrix_trans_dense.col(pixel_index[i][j]).data();
		lmatsup_ptr = Lmatrix_trans_supersampled.col(img_index).data();
		for (k=0; k < source_and_lens_n_amps; k++) {
			(*(lmatptr++)) += (*(lmatsup_ptr++))/nsubpix;
		}
	}


	//double *lmatptr, *lmatsup_ptr;
	//for (i=0; i < image_npixels; i++) p.image_surface_brightness[i] = 0;
	//for (img_index=0; img_index < image_n_subpixels; img_index++) {
		//i = mask_subpixel_i[img_index];
		//j = mask_subpixel_j[img_index];
		//lmatptr = Lmatrix_dense0[pixel_index[i][j]];
		//lmatsup_ptr = Lmatrix_supersampled0[img_index];
		//for (k=0; k < source_and_lens_n_amps; k++) {
			//(*(lmatptr++)) += (*(lmatsup_ptr++))/nsubpix;
		//}
	//}
}

#define DSWAP(a,b) dtemp=(a);(a)=(b);(b)=dtemp;
void ImagePixelGrid::fourier_transform(double* data, const int ndim, int* nn, const int isign)
{
	int idim,i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
	int ibit,k1,k2,n,nprev,nrem,ntot=1;
	double tempi,tempr,theta,wi,wpi,wpr,wr,wtemp,dtemp;
	for (i1=0; i1 < ndim; i1++) ntot *= nn[i1];

	nprev=1;
	for (idim=ndim-1;idim>=0;idim--) {
		n=nn[idim];
		nrem=ntot/(n*nprev);
		ip1=nprev << 1;
		ip2=ip1*n;
		ip3=ip2*nrem;
		i2rev=0;
		for (i2=0;i2<ip2;i2+=ip1) {
			if (i2 < i2rev) {
				for (i1=i2;i1<i2+ip1-1;i1+=2) {
					for (i3=i1;i3<ip3;i3+=ip2) {
						i3rev=i2rev+i3-i2;
						DSWAP(data[i3],data[i3rev]);
						DSWAP(data[i3+1],data[i3rev+1]);

					}
				}
			}
			ibit=ip2 >> 1;
			while ((ibit >= ip1) and ((i2rev+1) > ibit)) {
				i2rev -= ibit;
				ibit >>= 1;
			}
			i2rev += ibit;
		}
		ifp1=ip1;
		while (ifp1 < ip2) {
			ifp2=ifp1 << 1;
			theta=isign*M_2PI/(ifp2/ip1);
			wtemp=sin(theta/2);
			wpr = -2.0*wtemp*wtemp;
			wpi=sin(theta);
			wr=1.0;
			wi=0.0;
			for (i3=0;i3<ifp1;i3+=ip1) {
				for (i1=i3;i1<i3+ip1-1;i1+=2) {
					for (i2=i1;i2<ip3;i2+=ifp2) {
						k1=i2;
						k2=k1+ifp1;
						tempr=wr*data[k2]-wi*data[k2+1];
						tempi=wr*data[k2+1]+wi*data[k2];
						data[k2]=data[k1]-tempr;
						data[k2+1]=data[k1+1]-tempi;
						data[k1] += tempr;
						data[k1+1] += tempi;
					}
				}
				wr=(wtemp=wr)*wpr-wi*wpi+wr;
				wi=wi*wpr+wtemp*wpi+wi;
			}
			ifp1=ifp2;
		}
		nprev *= n;
	}
}
#undef DSWAP

bool ImagePixelGrid::create_regularization_matrix(const bool allow_reg_weighting, const bool use_sbweights, const bool potential_perturbations, const bool verbal)
{
	RegularizationMethod reg_method = qlens->regularization_method;
	if (!potential_perturbations) {
		if (Rmatrix_sparse != NULL) die("Rmatrix_sparse is not NULL");
		if (Rmatrix_index != NULL) die("Rmatrix_index is not NULL");
		if (n_src_inv==0) die("no sources have been selected to invert");
		if ((qlens->use_lum_weighted_regularization) and (!allow_reg_weighting)) reg_method = Curvature;
	} else {
		if (Rmatrix_pot != NULL) { delete[] Rmatrix_pot; Rmatrix_pot = NULL; }
		if (Rmatrix_pot_index != NULL) { delete[] Rmatrix_pot_index; Rmatrix_pot_index = NULL; }
	}

	qlens->dense_Rmatrix = false; // assume sparse unless a dense regularization is chosen
	qlens->use_covariance_matrix = false; // if true, will use covariance matrix directly instead of Rmatrix
	bool successful_Rmatrix = true;
	if ((!qlens->find_covmatrix_inverse) and (qlens->n_ptsrc > 0)) die("modeling point images is not currently compatible with 'find_cov_inverse off' setting"); // see notes in generate_Gmatrix function (this is where the problem is, I believe)...FIX LATER!!

	ImagePixelGrid *imggrid;
	for (int i=0; i < n_src_inv; i++) {
		imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
		if (allow_reg_weighting) calculate_lumreg_srcpixel_weights(use_sbweights);
		switch (reg_method) {
			case Norm:
				imggrid->generate_Rmatrix_norm(potential_perturbations); break;
			case Gradient:
				imggrid->generate_Rmatrix_from_gmatrices(potential_perturbations); break;
			case SmoothGradient:
				imggrid->generate_Rmatrix_from_gmatrices(true,potential_perturbations); break;
			case Curvature:
				imggrid->generate_Rmatrix_from_hmatrices(potential_perturbations); break;
			case SmoothCurvature:
				imggrid->generate_Rmatrix_from_hmatrices(true,potential_perturbations); break;
			case Matern_Kernel:
				qlens->dense_Rmatrix = true;
				if (!qlens->find_covmatrix_inverse) qlens->use_covariance_matrix = true;
				successful_Rmatrix = imggrid->generate_Rmatrix_from_covariance_kernel(0,allow_reg_weighting,potential_perturbations,verbal);
				break;
			case Exponential_Kernel:
				qlens->dense_Rmatrix = true;
				if (!qlens->find_covmatrix_inverse) qlens->use_covariance_matrix = true;
				successful_Rmatrix = imggrid->generate_Rmatrix_from_covariance_kernel(1,allow_reg_weighting,potential_perturbations,verbal);
				break;
			case Squared_Exponential_Kernel:
				qlens->dense_Rmatrix = true;
				if (!qlens->find_covmatrix_inverse) qlens->use_covariance_matrix = true;
				successful_Rmatrix = imggrid->generate_Rmatrix_from_covariance_kernel(2,allow_reg_weighting,potential_perturbations,verbal);
				break;
			default:
				die("Regularization method not recognized");
		}
		if (!successful_Rmatrix) return false;
		if ((qlens->dense_Rmatrix) and (qlens->matrix_format!=DENSE) and (qlens->matrix_format!=DENSE_FMATRIX)) die("inversion method must be set to 'dense' or 'fdense' if a dense regularization matrix is used");
		if (!qlens->dense_Rmatrix) {
			// If doing a sparse inversion, the determinant of R-matrix will be calculated when doing the inversion; otherwise, must be done here
			// unless R-matrix is dense (as in the covariance kernel reg.), in which case determinant is found during its construction
			imggrid->Rmatrix_determinant_EIGEN(false);
		}
	}

	//cout << "Printing Rmatrix..." << endl;
	//int indx;	
	//for (i=0; i < source_npixels; i++) {
		//indx = Rmatrix_index[i];
		//int nn = Rmatrix_index[i+1]-Rmatrix_index[i];
		//cout << "Row " << i << ": " << nn << " entries (starts at index " << indx << ")" << endl;
		//cout << "diag: " << Rmatrix_sparse[i] << endl;
		//for (j=0; j < nn; j++) {
			//cout << i << " " << Rmatrix_index[indx+j] << " " << Rmatrix_sparse[indx+j] << endl;
		//}
	//}
	return true;
}

void ImagePixelGrid::create_regularization_matrix_shapelet()
{
	if (source_npixels==0) return;
	if (Rmatrix_sparse != NULL) die("Rmatrix_sparse is not NULL");
	if (Rmatrix_index != NULL) die("Rmatrix_index is not NULL");
	if (n_src_inv==0) die("no sources have been selected to invert");

	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	qlens->dense_Rmatrix = false;
	qlens->use_covariance_matrix = false; // if true, will use covariance matrix directly instead of Rmatrix
	ImagePixelGrid *imggrid;
	for (int i=0; i < n_src_inv; i++) {
		imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
		switch (qlens->regularization_method) {
			case Norm:
				imggrid->generate_Rmatrix_norm(); break;
			case Gradient:
				imggrid->generate_Rmatrix_shapelet_gradient(); break;
			case Curvature:
				imggrid->generate_Rmatrix_shapelet_curvature(); break;
			default:
				die("Regularization method not recognized for dense matrices");
		}
		imggrid->Rmatrix_determinant_EIGEN(false);
	}
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for calculating Rmatrix: "  << wtime.count() << endl;
		wtime0 = std::chrono::steady_clock::now();
	}
}

void ImagePixelGrid::create_MGE_regularization_matrices()
{
	mge_list = new SB_Profile*[n_mge_sets];
	Rmatrix_MGE_packed = new Vector<double>[n_mge_sets];
	Rmatrix_MGE_log_determinants = new double[n_mge_sets];
	int i,j,k;
	for (i=0,j=0; i < qlens->n_sb; i++) {
		if (qlens->sb_list[i]->sbtype==MULTI_GAUSSIAN_EXPANSION) {
			if (qlens->sbprofile_imggrid_idx[i]==imggrid_index) {
				mge_list[j++] = qlens->sb_list[i];
			}
		}
	}
	if (j != n_mge_sets) die("number of MGE's don't add up");

	int n_rvals = 20;
	double *rvalsq = new double[n_rvals];
	double rmin=0.03, rmax=3;
	double r,rstep = pow(rmax/rmin,1.0/(n_rvals-1));
	for (i=0,r=rmin; i < n_rvals; i++, r *= rstep) rvalsq[i] = r*r;

	int n_amps_this_mge;
	SB_Profile** mge = mge_list;
	for (i=0; i < n_mge_sets; i++) {
		n_amps_this_mge = *((*mge)->indxptr);
		Rmatrix_MGE_packed[i].input(n_amps_this_mge*(n_amps_this_mge+1)/2);
		Rmatrix_MGE_packed[i] = 0;
		// Just going with identity matrix (norm regularization) because curvature regularization doesn't seem helpful.
		// It does seem that using NNLS is necessary when fitting multiple MGE components; regularization can't seem to fix the issue
		for (j=0,k=0; j < n_amps_this_mge; j++) {
			Rmatrix_MGE_packed[i][k] = 1;
			k += n_amps_this_mge - j; // gets to the next diagonal
		}
		Rmatrix_MGE_log_determinants[i] = 0; // for identity matrix
		//(*mge)->calculate_curvature_Rmatrix_elements_rvals(rvalsq,n_rvals,Rmatrix_MGE_packed[i].array());
		//matrix_determinant_dense(Rmatrix_MGE_log_determinants[i], Rmatrix_MGE_packed[i], n_amps_this_mge);
		//cout << "Rmatrix log-det for mge " << i << ": " << Rmatrix_MGE_log_determinants[i] << endl;
		mge++;
	}
	delete[] rvalsq;
}

void ImagePixelGrid::generate_Rmatrix_norm(const bool potential_perturbations)
{
	int Rmatrix_nn, npixels;
	double *new_Rmatrix_sparse;
	int *new_Rmatrix_index;
	if (!potential_perturbations) {
		npixels = source_npixels_inv;
	} else {
		npixels = lensgrid_npixels;
	}
	Rmatrix_nn = npixels+1;

	new_Rmatrix_sparse = new double[Rmatrix_nn];
	new_Rmatrix_index = new int[Rmatrix_nn];

	for (int i=0; i < npixels; i++) {
		new_Rmatrix_sparse[i] = 1;
		new_Rmatrix_index[i] = npixels+1;
	}
	new_Rmatrix_index[npixels] = npixels+1;

	if (!potential_perturbations) {
		if (Rmatrix_sparse != NULL) die("Rmatrix wasn't NULL");
		Rmatrix_sparse = new_Rmatrix_sparse;
		Rmatrix_index = new_Rmatrix_index;
		Rmatrix_log_determinant = 0;
	} else {
		Rmatrix_pot = new_Rmatrix_sparse;
		Rmatrix_pot_index = new_Rmatrix_index;
		Rmatrix_pot_log_determinant = 0;
	}
}

void ImagePixelGrid::generate_Rmatrix_from_hmatrices(const bool interpolate, const bool potential_perturbations)
{
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	int npixels;
	if (!potential_perturbations) {
		npixels = source_npixels_inv;
	} else {
		npixels = lensgrid_npixels;
	}

	int i,j,k,l,m,n,indx;

	vector<int> *jvals[2];
	vector<int> *lvals[2];
	for (i=0; i < 2; i++) {
		jvals[i] = new vector<int>[npixels];
		lvals[i] = new vector<int>[npixels];
	}

	Rmatrix_diag_temp = new double[npixels];
	Rmatrix_rows = new vector<double>[npixels];
	Rmatrix_index_rows = new vector<int>[npixels];
	Rmatrix_row_nn = new int[npixels];
	int Rmatrix_nn = 0;
	int Rmatrix_nn_part = 0;
	for (j=0; j < npixels; j++) {
		Rmatrix_diag_temp[j] = 0;
		Rmatrix_row_nn[j] = 0;
	}

	bool new_entry;
	int srcpot_index1, srcpot_index2, col_index, col_i;
	double tmp, element;

	for (k=0; k < 2; k++) {
		hmatrix_rows[k] = new vector<double>[npixels];
		hmatrix_index_rows[k] = new vector<int>[npixels];
		hmatrix_row_nn[k] = new int[npixels];
		hmatrix_nn[k] = 0;
		for (j=0; j < npixels; j++) {
			hmatrix_row_nn[k][j] = 0;
		}
	}
	if (!potential_perturbations) {
		if (qlens->source_fit_mode==Delaunay_Source) {
			delaunay_srcgrid->generate_hmatrices(interpolate);
		}
		else if (qlens->source_fit_mode==Cartesian_Source) {
			cartesian_srcgrid->generate_hmatrices();
		}
		else die("hmatrix not supported for sources other than Delaunay or Cartesian");
	} else {
		lensgrid->generate_hmatrices(interpolate); // in LensPixelGrid, the same function handles a Cartesian versus Delaunay grid
	}

	for (k=0; k < 2; k++) {
		hmatrix[k] = new double[hmatrix_nn[k]];
		hmatrix_index[k] = new int[hmatrix_nn[k]];
		hmatrix_row_index[k] = new int[npixels+1];

		hmatrix_row_index[k][0] = 0;
		for (i=0; i < npixels; i++)
			hmatrix_row_index[k][i+1] = hmatrix_row_index[k][i] + hmatrix_row_nn[k][i];
		if (hmatrix_row_index[k][npixels] != hmatrix_nn[k]) die("the number of elements don't match up for hmatrix %i",k);

		for (i=0; i < npixels; i++) {
			indx = hmatrix_row_index[k][i];
			for (j=0; j < hmatrix_row_nn[k][i]; j++) {
				hmatrix[k][indx+j] = hmatrix_rows[k][i][j];
				hmatrix_index[k][indx+j] = hmatrix_index_rows[k][i][j];
			}
		}
		delete[] hmatrix_rows[k];
		delete[] hmatrix_index_rows[k];
		delete[] hmatrix_row_nn[k];

		for (i=0; i < npixels; i++) {
			for (j=hmatrix_row_index[k][i]; j < hmatrix_row_index[k][i+1]; j++) {
				for (l=j; l < hmatrix_row_index[k][i+1]; l++) {
					srcpot_index1 = hmatrix_index[k][j];
					srcpot_index2 = hmatrix_index[k][l];
					if (srcpot_index1 > srcpot_index2) {
						tmp=srcpot_index1;
						srcpot_index1=srcpot_index2;
						srcpot_index2=tmp;
						jvals[k][srcpot_index1].push_back(l);
						lvals[k][srcpot_index1].push_back(j);
					} else {
						jvals[k][srcpot_index1].push_back(j);
						lvals[k][srcpot_index1].push_back(l);
					}
				}
			}
		}
	}

	#pragma omp parallel for private(i,j,k,l,m,n,srcpot_index1,srcpot_index2,new_entry,col_index,col_i,element) schedule(static) reduction(+:Rmatrix_nn_part)
	for (srcpot_index1=0; srcpot_index1 < npixels; srcpot_index1++) {
		for (k=0; k < 2; k++) {
			col_i=0;
			for (n=0; n < jvals[k][srcpot_index1].size(); n++) {
				j = jvals[k][srcpot_index1][n];
				l = lvals[k][srcpot_index1][n];
				srcpot_index2 = hmatrix_index[k][l];
				new_entry = true;
				element = hmatrix[k][j]*hmatrix[k][l];
				if (srcpot_index1==srcpot_index2) Rmatrix_diag_temp[srcpot_index1] += element;
				else {
					m=0;
					while ((m < Rmatrix_row_nn[srcpot_index1]) and (new_entry==true)) {
						if (Rmatrix_index_rows[srcpot_index1][m]==srcpot_index2) {
							new_entry = false;
							col_index = m;
						}
						m++;
					}
					if (new_entry) {
						Rmatrix_rows[srcpot_index1].push_back(element);
						Rmatrix_index_rows[srcpot_index1].push_back(srcpot_index2);
						Rmatrix_row_nn[srcpot_index1]++;
						col_i++;
					}
					else Rmatrix_rows[srcpot_index1][col_index] += element;
				}
			}
			Rmatrix_nn_part += col_i;
		}
	}

	for (k=0; k < 2; k++) {
		delete[] hmatrix[k];
		delete[] hmatrix_index[k];
		delete[] hmatrix_row_index[k];
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for calculating Rmatrix: "  << wtime.count() << endl;
	}

	Rmatrix_nn = Rmatrix_nn_part;
	Rmatrix_nn += npixels+1;

	double *new_Rmatrix_sparse;
	int *new_Rmatrix_index;

	new_Rmatrix_sparse = new double[Rmatrix_nn];
	new_Rmatrix_index = new int[Rmatrix_nn];

	for (i=0; i < npixels; i++)
		new_Rmatrix_sparse[i] = Rmatrix_diag_temp[i];

	new_Rmatrix_index[0] = npixels+1;
	for (i=0; i < npixels; i++) {
		new_Rmatrix_index[i+1] = new_Rmatrix_index[i] + Rmatrix_row_nn[i];
	}

	for (i=0; i < npixels; i++) {
		indx = new_Rmatrix_index[i];
		for (j=0; j < Rmatrix_row_nn[i]; j++) {
			new_Rmatrix_sparse[indx+j] = Rmatrix_rows[i][j];
			new_Rmatrix_index[indx+j] = Rmatrix_index_rows[i][j];
		}
	}

	if (!potential_perturbations) {
		if (Rmatrix_sparse != NULL) die("Rmatrix wasn't NULL");
		Rmatrix_sparse = new_Rmatrix_sparse;
		Rmatrix_index = new_Rmatrix_index;
	} else {
		Rmatrix_pot = new_Rmatrix_sparse;
		Rmatrix_pot_index = new_Rmatrix_index;
	}

	delete[] Rmatrix_row_nn;
	delete[] Rmatrix_diag_temp;
	delete[] Rmatrix_rows;
	delete[] Rmatrix_index_rows;

	for (i=0; i < 2; i++) {
		delete[] jvals[i];
		delete[] lvals[i];
	}
}

void ImagePixelGrid::generate_Rmatrix_from_gmatrices(const bool interpolate, const bool potential_perturbations)
{
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	int npixels;
	if (!potential_perturbations) {
		npixels = source_npixels_inv;
	} else {
		npixels = lensgrid_npixels;
	}

	int i,j,k,l,m,n,indx;

	vector<int> *jvals[4];
	vector<int> *lvals[4];
	for (i=0; i < 4; i++) {
		jvals[i] = new vector<int>[npixels];
		lvals[i] = new vector<int>[npixels];
	}

	Rmatrix_diag_temp = new double[npixels];
	Rmatrix_rows = new vector<double>[npixels];
	Rmatrix_index_rows = new vector<int>[npixels];
	Rmatrix_row_nn = new int[npixels];
	int Rmatrix_nn = 0;
	int Rmatrix_nn_part = 0;
	for (j=0; j < npixels; j++) {
		Rmatrix_diag_temp[j] = 0;
		Rmatrix_row_nn[j] = 0;
	}

	bool new_entry;
	int src_index1, src_index2, col_index, col_i;
	double tmp, element;

	for (k=0; k < 4; k++) {
		gmatrix_rows[k] = new vector<double>[npixels];
		gmatrix_index_rows[k] = new vector<int>[npixels];
		gmatrix_row_nn[k] = new int[npixels];
		gmatrix_nn[k] = 0;
		for (j=0; j < npixels; j++) {
			gmatrix_row_nn[k][j] = 0;
		}
	}
	if (!potential_perturbations) {
		if (qlens->source_fit_mode==Delaunay_Source) {
			delaunay_srcgrid->generate_gmatrices(interpolate);
		}
		else if (qlens->source_fit_mode==Cartesian_Source) {
			cartesian_srcgrid->generate_gmatrices();
		}
		else die("gmatrix not supported for sources other than Delaunay or Cartesian");
	} else {
		lensgrid->generate_gmatrices(false); // in LensPixelGrid, the same function handles a Cartesian versus Delaunay grid
	}

	for (k=0; k < 4; k++) {
		gmatrix[k] = new double[gmatrix_nn[k]];
		gmatrix_index[k] = new int[gmatrix_nn[k]];
		gmatrix_row_index[k] = new int[npixels+1];

		gmatrix_row_index[k][0] = 0;
		for (i=0; i < npixels; i++)
			gmatrix_row_index[k][i+1] = gmatrix_row_index[k][i] + gmatrix_row_nn[k][i];
		if (gmatrix_row_index[k][npixels] != gmatrix_nn[k]) die("the number of elements don't match up for gmatrix %i",k);

		for (i=0; i < npixels; i++) {
			indx = gmatrix_row_index[k][i];
			for (j=0; j < gmatrix_row_nn[k][i]; j++) {
				gmatrix[k][indx+j] = gmatrix_rows[k][i][j];
				gmatrix_index[k][indx+j] = gmatrix_index_rows[k][i][j];
			}
		}
		delete[] gmatrix_rows[k];
		delete[] gmatrix_index_rows[k];
		delete[] gmatrix_row_nn[k];

		for (i=0; i < npixels; i++) {
			for (j=gmatrix_row_index[k][i]; j < gmatrix_row_index[k][i+1]; j++) {
				for (l=j; l < gmatrix_row_index[k][i+1]; l++) {
					src_index1 = gmatrix_index[k][j];
					src_index2 = gmatrix_index[k][l];
					if (src_index1 > src_index2) {
						tmp=src_index1;
						src_index1=src_index2;
						src_index2=tmp;
						jvals[k][src_index1].push_back(l);
						lvals[k][src_index1].push_back(j);
					} else {
						jvals[k][src_index1].push_back(j);
						lvals[k][src_index1].push_back(l);
					}
				}
			}
		}
	}

	#pragma omp parallel for private(i,j,k,l,m,n,src_index1,src_index2,new_entry,col_index,col_i,element) schedule(static) reduction(+:Rmatrix_nn_part)
	for (src_index1=0; src_index1 < npixels; src_index1++) {
		for (k=0; k < 4; k++) {
			col_i=0;
			for (n=0; n < jvals[k][src_index1].size(); n++) {
				j = jvals[k][src_index1][n];
				l = lvals[k][src_index1][n];
				src_index2 = gmatrix_index[k][l];
				new_entry = true;
				element = gmatrix[k][j]*gmatrix[k][l];
				if (src_index1==src_index2) Rmatrix_diag_temp[src_index1] += element;
				else {
					m=0;
					while ((m < Rmatrix_row_nn[src_index1]) and (new_entry==true)) {
						if (Rmatrix_index_rows[src_index1][m]==src_index2) {
							new_entry = false;
							col_index = m;
						}
						m++;
					}
					if (new_entry) {
						Rmatrix_rows[src_index1].push_back(element);
						Rmatrix_index_rows[src_index1].push_back(src_index2);
						Rmatrix_row_nn[src_index1]++;
						col_i++;
					}
					else Rmatrix_rows[src_index1][col_index] += element;
				}
			}
			Rmatrix_nn_part += col_i;
		}
	}

	for (k=0; k < 4; k++) {
		delete[] gmatrix[k];
		delete[] gmatrix_index[k];
		delete[] gmatrix_row_index[k];
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for calculating Rmatrix: "  << wtime.count() << endl;
	}

	Rmatrix_nn = Rmatrix_nn_part;
	Rmatrix_nn += npixels+1;

	double *new_Rmatrix_sparse;
	int *new_Rmatrix_index;

	new_Rmatrix_sparse = new double[Rmatrix_nn];
	new_Rmatrix_index = new int[Rmatrix_nn];

	for (i=0; i < npixels; i++)
		new_Rmatrix_sparse[i] = Rmatrix_diag_temp[i];

	new_Rmatrix_index[0] = npixels+1;
	for (i=0; i < npixels; i++) {
		new_Rmatrix_index[i+1] = new_Rmatrix_index[i] + Rmatrix_row_nn[i];
	}

	for (i=0; i < npixels; i++) {
		indx = new_Rmatrix_index[i];
		for (j=0; j < Rmatrix_row_nn[i]; j++) {
			new_Rmatrix_sparse[indx+j] = Rmatrix_rows[i][j];
			new_Rmatrix_index[indx+j] = Rmatrix_index_rows[i][j];
		}
	}

	if (!potential_perturbations) {
		if (Rmatrix_sparse != NULL) die("Rmatrix wasn't NULL");
		Rmatrix_sparse = new_Rmatrix_sparse;
		if (Rmatrix_index != NULL) die("Rmatrix_index wasn't NULL");
		Rmatrix_index = new_Rmatrix_index;
	} else {
		Rmatrix_pot = new_Rmatrix_sparse;
		Rmatrix_pot_index = new_Rmatrix_index;
	}

	delete[] Rmatrix_row_nn;
	delete[] Rmatrix_diag_temp;
	delete[] Rmatrix_rows;
	delete[] Rmatrix_index_rows;

	for (i=0; i < 4; i++) {
		delete[] jvals[i];
		delete[] lvals[i];
	}
}

bool ImagePixelGrid::generate_Rmatrix_from_covariance_kernel(const int kernel_type, const bool allow_reg_weighting, const bool potential_perturbations, const bool verbal)
{
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	int npixels;
	double *Rmatrix_logdet_ptr;
	if (!potential_perturbations) {
		npixels = source_npixels_inv;
		Rmatrix_logdet_ptr = &Rmatrix_log_determinant;
	} else {
		npixels = lensgrid_npixels;
		Rmatrix_logdet_ptr = &Rmatrix_pot_log_determinant;
	}

	int ntot = npixels*(npixels+1)/2;
	double xc_approx, yc_approx, sig;
	covmatrix_dense.resize(npixels,npixels);
	Rmatrix_dense.resize(npixels,npixels);
	if (qlens->source_fit_mode==Delaunay_Source) {
		if (qlens->use_distance_weighted_regularization) {
			sig = find_approx_source_size(xc_approx,yc_approx,verbal);
			if ((verbal) and (qlens->mpi_id==0)) cout << "approx source size=" << sig << ", src_xc_approx=" << xc_approx << " src_yc_approx=" << yc_approx << endl;
			if (qlens->fix_lumreg_sig) sig = qlens->lumreg_sig;
			calculate_distreg_srcpixel_weights(xc_approx,yc_approx,sig,verbal);
		}
		if (qlens->use_mag_weighted_regularization) calculate_mag_srcpixel_weights();

		double *wgtfac = ((qlens->use_distance_weighted_regularization) or (qlens->use_mag_weighted_regularization) or ((allow_reg_weighting) and (qlens->use_lum_weighted_regularization))) ? reg_weight_factor : NULL;
		delaunay_srcgrid->generate_covariance_matrix(covmatrix_dense,kernel_type,qlens->covmatrix_epsilon,wgtfac);
	}
	else die("covariance kernel regularization requires source mode to be 'delaunay'");
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for calculating covariance matrix: "  << wtime.count() << endl;
	}

	covmatrix_factored.compute(covmatrix_dense);
	if(covmatrix_factored.info() != Eigen::Success) {
		if (verbal) warn("cholesky decomposition of covmatrix was not successful; covmatrix is not positive definite");
		if (qlens->penalize_defective_covmatrix) return false;
	}
	Eigen::MatrixXd U = covmatrix_factored.matrixU();
	(*Rmatrix_logdet_ptr) = -2.0*U.diagonal().array().log().sum(); // since this was the (log-)determinant of the inverse of the Rmatrix (i.e. using det(cov) = 1/det(cov_inverse))
	if (!qlens->use_covariance_matrix) {
		// Since we're going to use R-matrix explicitly, we must find it by taking cov_inverse
		Rmatrix_dense = covmatrix_factored.solve(Eigen::MatrixXd::Identity(npixels, npixels));
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for calculating covariance kernel Rmatrix: "  << wtime.count() << endl;
	}
	return true;
}

void ImagePixelGrid::generate_Rmatrix_shapelet_gradient()
{
	bool at_least_one_shapelet = false;
	int Rmatrix_nn = 3*source_npixels_inv+1; // actually it will be slightly less than this due to truncation at shapelets with i=n_shapelets-1 or j=n_shapelets-1

	Rmatrix_sparse = new double[Rmatrix_nn];
	Rmatrix_index = new int[Rmatrix_nn];

	for (int i=0; i < qlens->n_sb; i++) {
		if ((qlens->sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
			qlens->sb_list[i]->calculate_gradient_Rmatrix_elements(Rmatrix_sparse, Rmatrix_index);
			at_least_one_shapelet = true;
			break;
		}
	}
	if (!at_least_one_shapelet) die("No shapelet profile has been created; cannot calculate regularization matrix");
	//Rmatrix_nn = Rmatrix_index[source_npixels_inv];
	//for (int i=0; i <= source_npixels_inv; i++) cout << Rmatrix_sparse[i] << " " << Rmatrix_index[i] << endl;
	//cout << "Rmatrix_nn=" << Rmatrix_nn << " source_npixels_inv=" << source_npixels_inv << endl;
}

void ImagePixelGrid::generate_Rmatrix_shapelet_curvature()
{
	int Rmatrix_nn = source_npixels_inv+1;

	Rmatrix_sparse = new double[Rmatrix_nn];
	Rmatrix_index = new int[Rmatrix_nn];

	bool at_least_one_shapelet = false;
	for (int i=0; i < qlens->n_sb; i++) {
		if ((qlens->sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
			qlens->sb_list[i]->calculate_curvature_Rmatrix_elements(Rmatrix_sparse, Rmatrix_index);
			at_least_one_shapelet = true;
		}
	}
	if (!at_least_one_shapelet) die("No shapelet profile has been created; cannot calculate regularization matrix");
	//Rmatrix_nn = Rmatrix_index[source_npixels_inv];
}

/*
void ImagePixelGrid::generate_Rmatrix_MGE_curvature()
{
	bool at_least_one_mge = false;

	for (int i=0; i < qlens->n_sb; i++) {
		if ((qlens->sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
			qlens->sb_list[i]->calculate_gradient_Rmatrix_elements(Rmatrix_sparse, Rmatrix_index);
			at_least_one_shapelet = true;
			break;
		}
	}
	if (!at_least_one_shapelet) die("No shapelet profile has been created; cannot calculate regularization matrix");
	//Rmatrix_nn = Rmatrix_index[source_npixels_inv];
	//for (int i=0; i <= source_npixels_inv; i++) cout << Rmatrix_sparse[i] << " " << Rmatrix_index[i] << endl;
	//cout << "Rmatrix_nn=" << Rmatrix_nn << " source_npixels_inv=" << source_npixels_inv << endl;
}
*/

void ImagePixelGrid::get_source_regparam_ptr(const int imggrid_include_i, double* &regparam)
{
	if ((qlens->source_fit_mode==Delaunay_Source) or (qlens->source_fit_mode==Cartesian_Source)) {
		ImagePixelGrid *imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[imggrid_include_i]];
		if (qlens->source_fit_mode==Delaunay_Source) {
			regparam = &imggrid->delaunay_srcgrid->delaunay_srcgrid_params.regparam;
		} else if (qlens->source_fit_mode==Cartesian_Source) {
			regparam = &imggrid->cartesian_srcgrid->regparam;
		}
	}
	else if (qlens->source_fit_mode==Shapelet_Source) {
		int shapelet_i = -1;
		for (int j=0; j < qlens->n_sb; j++) {
			if ((qlens->sb_list[j]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[j]==imggrid_index)) {
				shapelet_i = j;
				break;
			}
		}

		if (shapelet_i >= 0) qlens->sb_list[shapelet_i]->get_regularization_param_ptr(regparam);
		else die("shapelet not found");
	}
	else die("unknown source pixellation mode");
}

void ImagePixelGrid::create_lensing_matrices_from_Lmatrix(const bool dense_Fmatrix, const bool potential_perturbations, const bool verbal)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	double cov_inverse; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (!qlens->use_noise_map) {
		if (qlens->background_pixel_noise==0) cov_inverse = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
		else cov_inverse = 1.0/SQR(qlens->background_pixel_noise);
	}

	int i,j,k,l,m,t;

	vector<int> *Fmatrix_index_rows;
	vector<double> *Fmatrix_rows;
	double *Fmatrix_diags;
	int *Fmatrix_row_nn;
	Fmatrix_nn = 0;
	int Fmatrix_nn_part = 0;
	for (j=0; j < n_amps; j++) {
		Fmatrix_diags[j] = 0;
		Fmatrix_row_nn[j] = 0;
	}
	int ntot = n_amps*n_amps;

	bool new_entry;
	int index1, index2, col_index, col_i;
	double tmp, element;
	Dvector = Eigen::VectorXd::Zero(n_amps);

	int pix_i, pix_j, img_index_fgmask;
	double sbcov;
	//double *Lmatrix_eff;
	//Lmatrix_eff = new double[Lmatrix_n_elements];
	for (i=0; i < image_npixels; i++) {
		if (qlens->use_noise_map) cov_inverse = imgpixel_covinv_vector[i];
		pix_i = emask_pixels_i[i];
		pix_j = emask_pixels_j[i];
		img_index_fgmask = pixel_index_fgmask[pix_i][pix_j];
		sbcov = p.image_surface_brightness[i] - p.sbprofile_surface_brightness[img_index_fgmask];
		if (((!qlens->include_imgfluxes_in_inversion) and (!qlens->include_srcflux_in_inversion)) and (qlens->n_ptsrc > 0)) sbcov -= point_image_surface_brightness[i];
		sbcov *= cov_inverse;
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			Dvector[Lmatrix_index[j]] += Lmatrix_sparse[j]*sbcov;
		}
	}
	for (i=0; i < image_npixels; i++) {
		if (qlens->use_noise_map) cov_inverse = imgpixel_covinv_vector[i];
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			Lmatrix_sparse[j] *= sqrt(cov_inverse); // this is a trick so that we can use the 'syrk' routine in MKL to create the Fmatrix_sparse
		}
	}

	int *srcpixel_location_Fmatrix, *srcpixel_end_Fmatrix, *Fmatrix_csr_index;
	double *Fmatrix_csr;
	using SparseRMd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
	SparseRMd L(image_npixels, n_amps);
	std::vector<Eigen::Triplet<double>> triplets;

	for(int i = 0; i < image_npixels; i++) {
		 for(int k = image_pixel_location_Lmatrix[i]; k < image_pixel_location_Lmatrix[i+1]; k++) {
			  triplets.emplace_back(i, Lmatrix_index[k], Lmatrix_sparse[k]);
		 }
	}
	L.setFromTriplets(triplets.begin(), triplets.end());
	if (!dense_Fmatrix) {
		Fmatrix_index_rows = new vector<int>[n_amps];
		Fmatrix_rows = new vector<double>[n_amps];
		Fmatrix_diags = new double[n_amps];
		Fmatrix_row_nn = new int[n_amps];
		SparseRMd Ffull = L.transpose() * L;
		SparseRMd F = Ffull.triangularView<Eigen::Upper>();
		F.makeCompressed();
		int nsrc1 = F.rows();
		//int nsrc2 = F.cols();

		srcpixel_location_Fmatrix = F.outerIndexPtr();  // row pointers (starts of rows)
		srcpixel_end_Fmatrix = F.outerIndexPtr() + 1; // you can access row ends similarly
		Fmatrix_csr_index = F.innerIndexPtr();  // column indices
		Fmatrix_csr = F.valuePtr();       // nonzero values
		//srcpixel_end_Fmatrix = new int[nsrc1];
		//for(int i=0;i<nsrc1;i++) {
			 //srcpixel_end_Fmatrix[i] = srcpixel_location_Fmatrix[i+1];
		//}
		if ((verbal) and (qlens->mpi_id==0)) cout << "Fmatrix_sparse has " << srcpixel_end_Fmatrix[n_amps-1] << " elements" << endl;
		bool duplicate_column;
		int dup_k;
		for (i=0; i < n_amps; i++) {
			for (j=srcpixel_location_Fmatrix[i]; j < srcpixel_end_Fmatrix[i]; j++) {
				duplicate_column = false;
				if (Fmatrix_csr_index[j]==i) {
					Fmatrix_diags[i] += Fmatrix_csr[j];
					//cout << "Adding " << Fmatrix_csr[j] << " to diag " << i << endl;
				}
				else if (Fmatrix_csr[j] != 0) {
					for (k=0; k < Fmatrix_index_rows[i].size(); k++) if (Fmatrix_csr_index[j]==Fmatrix_index_rows[i][k]) {
						duplicate_column = true;
						dup_k = k;
					}
					if (duplicate_column) {
						Fmatrix_rows[i][k] += Fmatrix_csr[j];
						die("duplicate!"); // this is not a big deal, but if duplicates never happen, then you might want to redo this part so it allocates memory in one go for each row instead of a bunch of push_back's
					} else {
						Fmatrix_rows[i].push_back(Fmatrix_csr[j]);
						Fmatrix_index_rows[i].push_back(Fmatrix_csr_index[j]);
						Fmatrix_row_nn[i]++;
						Fmatrix_nn_part++;
					}
				}
			}
		}
		//cout << "Done!" << endl;
		//cout << "LMATRIX:" << endl;
		//for (i=0; i < image_npixels; i++) {
			//cout << "Row " << i << ":" << endl;
			//for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
				//cout << Lmatrix_index[j] << " " << Lmatrix_sparse[j] << endl;
			//}
		//}
		//cout << endl << "FMATRIX:" << endl;
		//double Fsum=0;
		//int Fisum=0;
		//for (i=0; i < n_amps; i++) {
			////cout << "Row " << i << ":" << endl;
			//for (j=srcpixel_location_Fmatrix[i]; j < srcpixel_end_Fmatrix[i]; j++) {
				//Fsum += Fmatrix_csr[j];
				//Fisum += Fmatrix_csr_index[j];
				////cout << Fmatrix_csr_index[j] << " " << Fmatrix_csr[j] << endl;
			//}
		//}
		//cout << "Fsum=" << Fsum << " Fisum=" << Fisum << endl;
	} else {
		Fmatrix_dense.resize(n_amps,n_amps);
		Fmatrix_dense = (L.transpose() * L).toDense();
		if (qlens->use_covariance_matrix) generate_Gmatrix();
	}

	bool optimize_regparam_this_time = qlens->optimize_regparam;
	if (potential_perturbations) optimize_regparam_this_time = false;
	double *regparam, *regparam_pot;
	if (qlens->source_fit_mode==Delaunay_Source) regparam = &(delaunay_srcgrid->delaunay_srcgrid_params.regparam);
	else if (qlens->source_fit_mode==Cartesian_Source) regparam = &(cartesian_srcgrid->regparam);
	else die("unknown source pixellation mode");
	if ((potential_perturbations) and (lensgrid != NULL)) regparam_pot = &(lensgrid->lensgrid_params.regparam);
	ImagePixelGrid *imggrid;
	if (!dense_Fmatrix) {
		if ((qlens->regularization_method != None) and (source_npixels > 0)) {
			for (int i=0; i < n_src_inv; i++) {
				imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
				get_source_regparam_ptr(i,regparam);
				for (index1=imggrid->src_npixel_start, index2=0; index2 < imggrid->source_npixels_inv; index1++, index2++) {
					if ((!optimize_regparam_this_time) or (i>0)) Fmatrix_diags[index1] += (*regparam)*imggrid->Rmatrix_sparse[index2];
					col_i=0;
					for (j=imggrid->Rmatrix_index[index2]; j < imggrid->Rmatrix_index[index2+1]; j++) {
						new_entry = true;
						k=0;
						while ((k < Fmatrix_row_nn[index1]) and (new_entry==true)) {
							if (imggrid->Rmatrix_index[j]==Fmatrix_index_rows[index1][k]) {
								new_entry = false;
								col_index = k;
							}
							k++;
						}
						if (new_entry) {
							if ((!optimize_regparam_this_time) or (i>0)) {
							//cout << "Fmat row " << index1 << ", col " << (imggrid->Rmatrix_index[j]) << ": was 0, now adding " << ((*regparam)*imggrid->Rmatrix_sparse[j]) << endl;
								Fmatrix_rows[index1].push_back((*regparam)*imggrid->Rmatrix_sparse[j]);
							} else {
								Fmatrix_rows[index1].push_back(0);
								// This way, when we're optimizing the regularization parameter, the needed entries are already there to add to
							}
							Fmatrix_index_rows[index1].push_back(imggrid->Rmatrix_index[j]);
							Fmatrix_row_nn[index1]++;
							col_i++;
						} else {
							if ((!optimize_regparam_this_time) or (i>0)) {
							//cout << "Fmat row " << index1 << ", col " << (imggrid->Rmatrix_index[j]) << ": was " << Fmatrix_rows[index1][col_index] << ", now adding " << ((*regparam)*imggrid->Rmatrix[j]) << endl;
								Fmatrix_rows[index1][col_index] += (*regparam)*imggrid->Rmatrix_sparse[j];
							}

						}
					}
					Fmatrix_nn_part += col_i;
				}
			}

			if (potential_perturbations) {
				regparam_pot = &(lensgrid->lensgrid_params.regparam);
				for (index1=source_npixels, index2=0; index2 < lensgrid_npixels; index1++, index2++) {
					if ((!optimize_regparam_this_time) or (i>0)) Fmatrix_diags[index1] += (*regparam_pot)*Rmatrix_pot[index2];
					col_i=0;
					for (j=Rmatrix_pot_index[index2]; j < Rmatrix_pot_index[index2+1]; j++) {
						new_entry = true;
						k=0;
						while ((k < Fmatrix_row_nn[index1]) and (new_entry==true)) {
							if (Rmatrix_pot_index[j]==Fmatrix_index_rows[index1][k]) {
								new_entry = false;
								col_index = k;
							}
							k++;
						}
						if (new_entry) {
							if ((!optimize_regparam_this_time) or (i>0)) {
							//cout << "Fmat row " << index1 << ", col " << (Rmatrix_pot_index[j]) << ": was 0, now adding " << ((*regparam)*Rmatrix_pot[j]) << endl;
								Fmatrix_rows[index1].push_back((*regparam_pot)*Rmatrix_pot[j]);
							} else {
								Fmatrix_rows[index1].push_back(0);
								// This way, when we're optimizing the regularization parameter, the needed entries are already there to add to
							}
							Fmatrix_index_rows[index1].push_back(Rmatrix_pot_index[j]);
							Fmatrix_row_nn[index1]++;
							col_i++;
						} else {
							if ((!optimize_regparam_this_time) or (i>0)) {
							//cout << "Fmat row " << index1 << ", col " << (Rmatrix_pot_index[j]) << ": was " << Fmatrix_rows[index1][col_index] << ", now adding " << ((*regparam_pot)*Rmatrix_pot[j]) << endl;
								Fmatrix_rows[index1][col_index] += (*regparam_pot)*Rmatrix_pot[j];
							}

						}
					}
					Fmatrix_nn_part += col_i;
				}
			}
		}

		Fmatrix_nn = Fmatrix_nn_part;
		Fmatrix_nn += n_amps+1;

		Fmatrix_sparse = new double[Fmatrix_nn];
		Fmatrix_index = new int[Fmatrix_nn];

		Fmatrix_index[0] = n_amps+1;
		for (i=0; i < n_amps; i++) {
			Fmatrix_index[i+1] = Fmatrix_index[i] + Fmatrix_row_nn[i];
		}
		if (Fmatrix_index[n_amps] != Fmatrix_nn) die("Fmatrix_sparse # of elements don't match up (%i vs %i), process %i",Fmatrix_index[n_amps],Fmatrix_nn,qlens->mpi_id);

		for (i=0; i < n_amps; i++)
			Fmatrix_sparse[i] = Fmatrix_diags[i];

		int indx;
		for (i=0; i < n_amps; i++) {
			indx = Fmatrix_index[i];
			for (j=0; j < Fmatrix_row_nn[i]; j++) {
				Fmatrix_sparse[indx+j] = Fmatrix_rows[i][j];
				Fmatrix_index[indx+j] = Fmatrix_index_rows[i][j];
			}
		}

		if (qlens->show_wtime) {
			wtime = std::chrono::steady_clock::now() - wtime0;
			if (qlens->mpi_id==0) cout << "Wall time for Fmatrix_sparse construction: "  << wtime.count() << endl;
		}
		if ((qlens->mpi_id==0) and (verbal)) cout << "Fmatrix_sparse now has " << Fmatrix_nn << " elements\n";

		if ((qlens->mpi_id==0) and (verbal)) {
			int Fmatrix_ntot = n_amps*(n_amps+1)/2;
			double sparseness = ((double) Fmatrix_nn)/Fmatrix_ntot;
			cout << "src_npixels = " << n_amps << endl;
			cout << "Fmatrix_sparse ntot = " << Fmatrix_ntot << endl;
			cout << "Fmatrix_sparse sparseness = " << sparseness << endl;
		}
	} else {
		if ((qlens->regularization_method != None) and (source_npixels > 0)) {
			for (int i=0; i < n_src_inv; i++) {
				get_source_regparam_ptr(i,regparam);
				imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
				if ((!optimize_regparam_this_time) or (i > 0)) imggrid->add_regularization_term_to_dense_Fmatrix(this,regparam,false);
			}

			if ((potential_perturbations) and (lensgrid != NULL)) {
				regparam_pot = &(lensgrid->lensgrid_params.regparam);
				add_regularization_term_to_dense_Fmatrix(this,regparam_pot,true);
			}
		}
	}
		if (qlens->show_wtime) {
			wtime = std::chrono::steady_clock::now() - wtime0;
			if (qlens->mpi_id==0) cout << "Wall time for calculating Fmatrix_sparse elements: "  << wtime.count() << endl;
			wtime0 = std::chrono::steady_clock::now();
		}
	for (i=0; i < image_npixels; i++) {
		if (qlens->use_noise_map) cov_inverse = imgpixel_covinv_vector[i];
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			Lmatrix_sparse[j] /= sqrt(cov_inverse);
		}
	}

	//cout << "FMATRIX (SPARSE):" << endl;
	//for (i=0; i < n_amps; i++) {
		//cout << i << "," << i << ": " << Fmatrix_sparse[i] << endl;
		//for (j=Fmatrix_index[i]; j < Fmatrix_index[i+1]; j++) {
			//cout << i << "," << Fmatrix_index[j] << ": " << Fmatrix_sparse[j] << endl;
		//}
		//cout << endl;
	//}

/*
	bool found;
	cout << "LMATRIX:" << endl;
	for (i=0; i < image_npixels; i++) {
		for (j=0; j < n_amps; j++) {
			found = false;
			for (k=image_pixel_location_Lmatrix[i]; k < image_pixel_location_Lmatrix[i+1]; k++) {
				if (Lmatrix_index[k]==j) {
					found = true;
					cout << Lmatrix_sparse[k] << " ";
				}
			}
			if (!found) cout << "0 ";
		}
		cout << endl;
	}
	*/	

	if (!dense_Fmatrix) {
		delete[] Fmatrix_index_rows;
		delete[] Fmatrix_rows;
		delete[] Fmatrix_diags;
		delete[] Fmatrix_row_nn;
	}
}

void ImagePixelGrid::create_lensing_matrices_from_Lmatrix_dense(const bool potential_perturbations, const bool verbal)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	bool at_least_one_shapelet = false;
	for (int i=0; i < qlens->n_sb; i++) {
		if ((qlens->sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
			at_least_one_shapelet = true;
			break;
		}
	}

	double cov_inverse; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (!qlens->use_noise_map) {
		if (qlens->background_pixel_noise==0) cov_inverse = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
		else cov_inverse = 1.0/SQR(qlens->background_pixel_noise);
	}

	int i,j,l,n;

	bool new_entry;
	Dvector = Eigen::VectorXd::Zero(n_amps);
	Eigen::MatrixXd Lmatrix_trans_scaled(n_amps,image_npixels);
	Lmatrix_trans_scaled = Lmatrix_trans_dense.array().rowwise() * imgpixel_covinv_vector.cwiseSqrt().transpose().array();
	Eigen::VectorXd sb_adj(image_npixels);

	int pix_i, pix_j;
	int img_index_fgmask;
	double covinv = cov_inverse;
	for (j=0; j < image_npixels; j++) {
		if (qlens->use_noise_map) covinv = imgpixel_covinv_vector[j];
		pix_i = emask_pixels_i[j];
		pix_j = emask_pixels_j[j];
		img_index_fgmask = pixel_index_fgmask[pix_i][pix_j];
		sb_adj[j] = p.image_surface_brightness[j] - p.sbprofile_surface_brightness[img_index_fgmask];
		if (((!qlens->include_imgfluxes_in_inversion) and (!qlens->include_srcflux_in_inversion)) and (qlens->n_ptsrc > 0)) sb_adj[j] -= point_image_surface_brightness[j];
		sb_adj[j] *= covinv;
	}
	Dvector = Lmatrix_trans_dense*sb_adj;

	Fmatrix_dense = Eigen::MatrixXd::Zero(n_amps,n_amps);
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for initializing Fmatrix and Dvector: "  << wtime.count() << endl;
		wtime0 = std::chrono::steady_clock::now();
	}

	Fmatrix_dense.selfadjointView<Eigen::Upper>().rankUpdate(Lmatrix_trans_scaled);
	//for (i=0; i < n_amps; i++) {
		//for (j=i+1; j < n_amps; j++) {
			//Fmatrix_dense(j,i) = Fmatrix_dense(i,j);
		//}
	//}

	ImagePixelGrid* imggrid;
	if (qlens->regularization_method != None) {
		if (source_npixels > 0) {
			double *regparam, *regparam_pot;
			if (qlens->use_covariance_matrix) generate_Gmatrix();

			for (int i=0; i < n_src_inv; i++) {
				get_source_regparam_ptr(i,regparam);
				imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
				if ((!qlens->optimize_regparam) or (i > 0) or (potential_perturbations)) imggrid->add_regularization_term_to_dense_Fmatrix(this,regparam,false);
			}

			if ((potential_perturbations) and (lensgrid != NULL)) {
				regparam_pot = &(lensgrid->lensgrid_params.regparam);
				add_regularization_term_to_dense_Fmatrix(this,regparam_pot,true);
			}
		}
		if (n_mge_amps > 0) add_MGE_regularization_terms_to_dense_Fmatrix();
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for calculating Fmatrix dense elements: "  << wtime.count() << endl;
		wtime0 = std::chrono::steady_clock::now();
	}
}

void ImagePixelGrid::generate_Gmatrix()
{
	Dvector_cov = Eigen::VectorXd::Zero(n_amps);
	Gmatrix.resize(n_amps,n_amps);
	// NOTE: right now, this doesn't work if image fluxes are included (so n_amps > source_npixels)
	// You need to explicitly make elements zero that are beyond source_npixels*source_npixels (i.e. if there are other amplitudes)
	std::chrono::steady_clock::time_point wtime_gmat0;
	std::chrono::duration<double> wtime_gmat;
	if (qlens->show_wtime) {
		wtime_gmat0 = std::chrono::steady_clock::now();
	}

	int i,j;
	//Must fill in all elements of Fmatrix_dense before multiplying to get Gmatrix
	for (i=0; i < n_amps; i++) {
		for (j=i+1; j < n_amps; j++) {
			Fmatrix_dense(j,i) = Fmatrix_dense(i,j);
		}
	}
	Gmatrix.noalias() = covmatrix_dense.selfadjointView<Eigen::Upper>() * Fmatrix_dense;


	Dvector_cov.noalias() = covmatrix_dense.selfadjointView<Eigen::Upper>() * Dvector;
	if (qlens->show_wtime) {
		wtime_gmat = std::chrono::steady_clock::now() - wtime_gmat0;
		if (qlens->mpi_id==0) cout << "Wall time for generating Gmatrix and D_cov: "  << wtime_gmat.count() << endl;
	}
}

void ImagePixelGrid::add_regularization_term_to_dense_Fmatrix(ImagePixelGrid *imggrid, double *regparam, const bool potential_perturbations)
{
	int npixels, start_indx, end_indx;
	if (!potential_perturbations) {
		npixels = source_npixels_inv;
		start_indx = src_npixel_start;
		Rmatrix_ptr = &Rmatrix_sparse;
		Rmatrix_index_ptr = &Rmatrix_index;
		Rmatrix_dense_ptr = &Rmatrix_dense;
	} else {
		npixels = lensgrid_npixels;
		start_indx = source_npixels;
		Rmatrix_ptr = &Rmatrix_pot;
		Rmatrix_index_ptr = &Rmatrix_pot_index;
	}

	int row_indx_offset;
	if (start_indx==0) row_indx_offset = 0;
	else {
		row_indx_offset = start_indx*n_amps - start_indx*(start_indx-1)/2; // the sum of elements of rows prior to the row=start_indx
	}

	int i,j;
	if (qlens->dense_Rmatrix) {
		end_indx = start_indx + npixels;
		if (!qlens->use_covariance_matrix) {
			imggrid->Fmatrix_dense += (*regparam)*(*Rmatrix_dense_ptr);  // Is this really the most computationally efficient way to do this?
		} else {
			// You'll have to carefully work out how the Gmatrix will work when pot perturbations are included. Leaving this for now
			imggrid->Gmatrix /= (*regparam);
			imggrid->Dvector_cov /= (*regparam);
			for (i=0; i < source_npixels; i++) { // additional source amplitudes (beyond source_npixels) are not regularized
				imggrid->Gmatrix(i,i) += 1.0;
			}
		}
	} else {
		//if (potential_perturbations) cout << "We are doing pot perturbations" << endl;
		//else cout << "We are NOT doing pot perturbations" << endl;
		//cout << "Basic Rmatrix check:" << endl;
		//cout << Rmatrix_ptr[0] << endl;
		//cout << Rmatrix_index_ptr[0] << endl;
		for (i=0; i < npixels; i++) { // additional source amplitudes (beyond source_npixels) are not regularized
			//	cout << "row " << i << "... offset=" << row_indx_offset << endl;
			imggrid->Fmatrix_dense(start_indx+i,start_indx+i) += (*regparam)*(*Rmatrix_ptr)[i];
			for (j=(*Rmatrix_index_ptr)[i]; j < (*Rmatrix_index_ptr)[i+1]; j++) {
				imggrid->Fmatrix_dense(start_indx+i,start_indx+(*Rmatrix_index_ptr)[j]) += (*regparam)*(*Rmatrix_ptr)[j];
				//Fmatrix_dense(start_indx+(*Rmatrix_index_ptr)[j],start_indx+i) += (*regparam)*(*Rmatrix_ptr)[j];
			}
			//cout << "setting new offset for row " << i << endl;
			row_indx_offset += n_amps-start_indx-i;
		}
	}
}

void ImagePixelGrid::add_MGE_regularization_terms_to_dense_Fmatrix()
{
	//int i,j;
	//int start_indx, row_indx_offset;
	//int end_indx = start_indx+n_mge_amps;
	//int n_extra_amps = n_amps - end_indx;

	//Fptr = Fmatrix_packed.array_offset(start_indx);

	//j=0,k=0;
	//cout << "START_i=" << start_indx << " N_AMPS=" << n_mge_amps << " N_SETS=" << n_mge_sets << " n_amps_this=" << n_amps_this_mge << endl;
	/*
	for (i=start_indx, j=0; i < end_indx; i++, j++) {
		//Fptr += i;
		*(Fptr) += (*regparam);

		Fptr += n_extra_amps+n_mge_amps-j;
		if ((k++ == n_amps_this_mge) and (mge != mge_end)) {
			mge++;
			mge->get_regularization_param_ptr(regparam);
			n_amps_this_mge = *(mge->indxptr);
			k=0;
		}
	}
	*/

	/*
	for (i=0; i < n_mge_amps; i++) { // additional source amplitudes (beyond source_npixels) are not regularized
		//	cout << "row " << i << "... offset=" << row_indx_offset << endl;
		//cout << "i=" << i << ", k=" << k << " reg=" << (*regparam) << endl;
		Fmatrix_packed[row_indx_offset] += (*regparam);
		row_indx_offset += n_amps-start_indx-i;
		if ((++k == n_amps_this_mge) and (mge != mge_end)) {
			mge++;
			(*mge)->get_regularization_param_ptr(regparam);
			n_amps_this_mge = *((*mge)->indxptr);
			//cout << "REACHED END OF MGE: " << k << " amps, n_amps_next=" << n_amps_this_mge << endl;
			k=0;
		}
	}
	*/

	int i,j;
	int start_indx, row_indx_offset, n_amps_this_mge, n_extra_amps;
	SB_Profile** mge = mge_list;
	SB_Profile** mge_end = &mge_list[n_mge_sets-1] + 1;
	double *regparam;
	Vector<double> *Rmatrix_ptr;
	double *Fptr, *Rptr;

	start_indx = source_npixels+lensgrid_npixels;
	//int indx_crap;
	for (mge=mge_list, Rmatrix_ptr=Rmatrix_MGE_packed; mge != mge_end; mge++, Rmatrix_ptr++) {
		n_amps_this_mge = *((*mge)->indxptr);
		(*mge)->get_regularization_param_ptr(regparam);
		if (start_indx==0) row_indx_offset = 0;
		else {
			row_indx_offset = start_indx*n_amps - start_indx*(start_indx-1)/2; // the sum of elements of rows prior to the row=start_indx
		}
		//indx_crap=row_indx_offset;
		n_extra_amps = n_amps - start_indx - n_amps_this_mge;
		Rptr = Rmatrix_ptr->array();
		//Fptr = Fmatrix_packed.array_offset(row_indx_offset);

		for (i=0; i < n_amps_this_mge; i++) {
			for (j=i; j < n_amps_this_mge; j++) {
				*(Fptr++) += (*regparam)*(*(Rptr++));
				//*(Fptr) += (*regparam)*(*(Rptr));
				//if ((*Rptr) != 0.0) cout << "Adding " << (*Rptr) << " to Fmatrix element " << indx_crap << " and kerg=" << kerg << endl;
				//indx_crap++;
				//kerg++;
				//Fptr++;
				//Rptr++;
			}
			Fptr += n_extra_amps;
		}
		start_indx += n_amps_this_mge;
	}
}

double ImagePixelGrid::calculate_regularization_prior_term(double *regparam, const bool potential_perturbations)
{
	int npixels, start_indx;
	double *Rmatrix_logdet_ptr;
	if (!potential_perturbations) {
		npixels = source_npixels_inv;
		start_indx = src_npixel_start;
		Rmatrix_ptr = &Rmatrix_sparse;
		Rmatrix_index_ptr = &Rmatrix_index;
		Rmatrix_dense_ptr = &Rmatrix_dense;
	} else {
		npixels = lensgrid_npixels;
		start_indx = source_npixels;
		Rmatrix_ptr = &Rmatrix_pot;
		Rmatrix_index_ptr = &Rmatrix_pot_index;
	}

	int i,j;
	double loglike_reg,Es_times_two=0;
	if (qlens->dense_Rmatrix) {
		if (qlens->use_covariance_matrix) {
			// Need to expand this to work for potential perturbations
			Eigen::VectorXd b(npixels);
			for (int i=0; i < npixels; i++) b[i] = amplitude_vector[i];
			b = covmatrix_factored.solve(b);
			for (i=0; i < npixels; i++) Es_times_two += amplitude_vector[i]*b[i];
			loglike_reg = (*regparam)*Es_times_two;
		} else {
			Eigen::Map<Eigen::VectorXd> s(amplitude_vector.data(), npixels);
			Es_times_two = s.transpose()*(*Rmatrix_dense_ptr)*s;
			loglike_reg = (*regparam)*Es_times_two - npixels*log((*regparam)) - Rmatrix_log_determinant;
				//cout << "regparam=" << (*regparam) << " Es_times_two=" << Es_times_two << " Flogdet=" << Fmatrix_log_determinant << " logreg0=" << loglike_reg << " loglike_reg=" << (loglike_reg+Fmatrix_log_determinant) << " Rlogdet=" << Rmatrix_log_determinant << endl;
		}
	} else {
		for (i=0; i < npixels; i++) {
			Es_times_two += (*Rmatrix_ptr)[i]*SQR(amplitude_vector[start_indx+i]);
			for (j=(*Rmatrix_index_ptr)[i]; j < (*Rmatrix_index_ptr)[i+1]; j++) {
				Es_times_two += 2 * amplitude_vector[start_indx+i] * (*Rmatrix_ptr)[j] * amplitude_vector[start_indx+(*Rmatrix_index_ptr)[j]]; // factor of 2 since matrix is symmetric
			}
		}
		loglike_reg = (*regparam)*Es_times_two - npixels*log((*regparam)) - Rmatrix_log_determinant;
		//cout << "regparam=" << (*regparam) << " Es_times_two=" << Es_times_two << " Flogdet=" << Fmatrix_log_determinant << " logreg0=" << loglike_reg << " loglike_reg=" << (loglike_reg+Fmatrix_log_determinant) << " Rlogdet=" << Rmatrix_log_determinant << endl;
	}
	return loglike_reg;
}

void ImagePixelGrid::add_regularization_prior_terms_to_logev(double& logev_times_two, double& loglike_reg, double& regterms, const bool include_potential_perturbations, const bool verbal)
{
	ImagePixelGrid *imggrid;
	if (source_npixels > 0) {
		for (int i=0; i < n_src_inv; i++) {
			imggrid = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];
			if ((qlens->source_fit_mode==Delaunay_Source) or (qlens->source_fit_mode==Cartesian_Source)) {
				if (qlens->source_fit_mode==Delaunay_Source) {
					regparam_ptr = &imggrid->delaunay_srcgrid->delaunay_srcgrid_params.regparam;
				} else if (qlens->source_fit_mode==Cartesian_Source) {
					regparam_ptr = &imggrid->cartesian_srcgrid->regparam;
				}
			}
			else if (qlens->source_fit_mode==Shapelet_Source) {
				int shapelet_i = -1;
				for (int i=0; i < qlens->n_sb; i++) {
					if ((qlens->sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
						shapelet_i = i;
						break;
					}
				}

				if (shapelet_i >= 0) qlens->sb_list[shapelet_i]->get_regularization_param_ptr(regparam_ptr);
				else die("shapelet not found");
			}
			else die("unknown source pixellation mode");

			if ((*regparam_ptr) != 0) {
				regterms = imggrid->calculate_regularization_prior_term(regparam_ptr);
				logev_times_two += regterms;
				loglike_reg += regterms;
			}
			if ((qlens->mpi_id==0) and (verbal)) {
				if ((source_npixels > 0) and (qlens->regularization_method != None)) {
					if (qlens->n_extended_src_redshifts > 1) cout << "imggrid_i=" << imggrid_indx_to_include_in_Lmatrix[i] << ": ";
					if (qlens->use_covariance_matrix) cout << "logdet(Gmatrix)=" << Gmatrix_log_determinant;
					else cout << "logdet(Fmatrix)=" << Fmatrix_log_determinant;
					cout << " logdet(Rmatrix)=" << Rmatrix_log_determinant;
					cout << endl;
				}
			}
		}
	}
	if (n_mge_amps > 0)  {
		regterms = calculate_MGE_regularization_prior_term();
		logev_times_two += regterms;
		loglike_reg += regterms;
	}
	if ((include_potential_perturbations) and (lensgrid != NULL)) {
		regparam_ptr = &(lensgrid->lensgrid_params.regparam);
		if ((*regparam_ptr) != 0) {
			regterms = calculate_regularization_prior_term(regparam_ptr,true);
			logev_times_two += regterms;
			loglike_reg += regterms;
		}
	}

	if ((!qlens->use_covariance_matrix) or (qlens->source_fit_mode != Delaunay_Source)) logev_times_two += Fmatrix_log_determinant;
	else logev_times_two += Gmatrix_log_determinant;
}

double ImagePixelGrid::calculate_MGE_regularization_prior_term()
{
	if (n_mge_sets==0) return 0.0;

	SB_Profile** mge = mge_list;
	SB_Profile** mge_end = &mge_list[n_mge_sets-1]+1;
	int start_indx, n_amps_this_mge;
	double loglike_reg,Es_times_two;

	double *regparam;
	Vector<double> *Rmatrix_ptr;
	double *Rmatrix_logdet_ptr;
	double *Rptr, *sptr_i, *sptr_j, *s_end;

	loglike_reg = 0;
	start_indx = source_npixels+lensgrid_npixels;
	for (mge=mge_list, Rmatrix_ptr=Rmatrix_MGE_packed, Rmatrix_logdet_ptr=Rmatrix_MGE_log_determinants; mge != mge_end; mge++, Rmatrix_ptr++, Rmatrix_logdet_ptr++) {
		n_amps_this_mge = *((*mge)->indxptr);
		(*mge)->get_regularization_param_ptr(regparam);
		Rptr = Rmatrix_ptr->array();
		s_end = amplitude_vector.data() + start_indx + n_amps_this_mge;
		Es_times_two = 0;
		for (sptr_i=amplitude_vector.data() + start_indx; sptr_i != s_end; sptr_i++) {
			sptr_j = sptr_i;
			Es_times_two += (*sptr_i)*(*(Rptr++))*(*(sptr_j++)); // diagonal element only gets counted once (no factor of 2 here)
			while (sptr_j != s_end) {
				Es_times_two += 2*(*sptr_i)*(*(Rptr++))*(*(sptr_j++));
			}
		}
		loglike_reg += (*regparam)*Es_times_two - n_amps_this_mge*log((*regparam)) - (*Rmatrix_logdet_ptr);
		start_indx += n_amps_this_mge;
	}

	return loglike_reg;
}

bool ImagePixelGrid::optimize_regularization_parameter(const bool dense_Fmatrix, const bool verbal, const bool pre_srcgrid)
{
	std::chrono::steady_clock::time_point wtime_opt0;
	std::chrono::duration<double> wtime_opt;
	if (qlens->show_wtime) {
		wtime_opt0 = std::chrono::steady_clock::now();
	}
	setup_regparam_optimization(dense_Fmatrix);
	int i;
	double logreg_min;
	double (ImagePixelGrid::*chisqreg)(const double);
	if (dense_Fmatrix) chisqreg = &ImagePixelGrid::chisq_regparam_dense;
	else chisqreg = &ImagePixelGrid::chisq_regparam;
	logreg_min = brents_min_method(chisqreg,qlens->optimize_regparam_minlog,qlens->optimize_regparam_maxlog,qlens->optimize_regparam_tol,verbal);
	//(this->*chisqreg)(log((*regparam_ptr))/ln10); // used for testing purposes
	(*regparam_ptr) = pow(10,logreg_min);
	if ((verbal) and (qlens->mpi_id==0)) cout << "regparam after optimizing: " << (*regparam_ptr) << endl;

	if (qlens->use_covariance_matrix) Gmatrix_log_determinant = regopt_logdet;
	else Fmatrix_log_determinant = regopt_logdet;
	for (i=0; i < n_amps; i++) amplitude_vector[i] = amplitude_vector_minchisq[i];
	if (qlens->show_wtime) {
		wtime_opt = std::chrono::steady_clock::now() - wtime_opt0;
		if (qlens->mpi_id==0) cout << "Wall time for optimizing regularization parameter: "  << wtime_opt.count() << endl;
		wtime_opt0 = std::chrono::steady_clock::now();
	}
	if ((qlens->use_lum_weighted_regularization) and (!pre_srcgrid)) {
		if (!qlens->get_lumreg_from_sbweights) {
			if (!qlens->use_covariance_matrix) {
				// This means we started with a non-covmatrix based regularization (e.g. curvature) to get the initial luminosity.
				// We'll switch to covariance matrix shortly, so initialize the copies here
				Gmatrix_copy.resize(n_amps,n_amps);
				Dvector_cov_copy.resize(n_amps);
			}
			if (Rmatrix_sparse != NULL) { delete[] Rmatrix_sparse; Rmatrix_sparse = NULL; }
			if (Rmatrix_index != NULL) { delete[] Rmatrix_index; Rmatrix_index = NULL; }
			if (create_regularization_matrix(true)==false) return false; // must re-generate covariance matrix with updated correlation lengths (from new pixel sb-weights)
			if (qlens->use_covariance_matrix) generate_Gmatrix();
			regopt_chisqmin = 1e30;
			if (qlens->show_wtime) {
				wtime_opt0 = std::chrono::steady_clock::now();
			}
			logreg_min = brents_min_method(chisqreg,qlens->optimize_regparam_minlog,qlens->optimize_regparam_maxlog,qlens->optimize_regparam_tol,verbal);
			(*regparam_ptr) = pow(10,logreg_min);
			if ((verbal) and (qlens->mpi_id==0)) cout << "regparam after optimizing with luminosity-weighted regularization: " << (*regparam_ptr) << endl;
			if (qlens->use_covariance_matrix) Gmatrix_log_determinant = regopt_logdet;
			else Fmatrix_log_determinant = regopt_logdet;
			for (i=0; i < n_amps; i++) amplitude_vector[i] = amplitude_vector_minchisq[i];
			if (verbal) if (qlens->mpi_id==0) cout << "loglike=" << regopt_chisqmin << endl;
		}

		for (int j=0; j < qlens->lumreg_max_it; j++) {
			if (Rmatrix_sparse != NULL) { delete[] Rmatrix_sparse; Rmatrix_sparse = NULL; }
			if (Rmatrix_index != NULL) { delete[] Rmatrix_index; Rmatrix_index = NULL; }
			if (create_regularization_matrix(true)==false) return false; // must re-generate covariance matrix with updated correlation lengths (from new pixel sb-weights)
			if (qlens->use_covariance_matrix) generate_Gmatrix();
			regopt_chisqmin = 1e30;
			logreg_min = brents_min_method(chisqreg,qlens->optimize_regparam_minlog,qlens->optimize_regparam_maxlog,qlens->optimize_regparam_tol,verbal);
			(*regparam_ptr) = pow(10,logreg_min);
			if ((verbal) and (qlens->mpi_id==0)) cout << "regparam after optimizing with luminosity-weighted regularization: " << (*regparam_ptr) << endl;
			if (qlens->use_covariance_matrix) Gmatrix_log_determinant = regopt_logdet;
			else Fmatrix_log_determinant = regopt_logdet;
			for (i=0; i < n_amps; i++) amplitude_vector[i] = amplitude_vector_minchisq[i];
			if (verbal) if (qlens->mpi_id==0) cout << "lumreg_it=" << j << " loglike=" << regopt_chisqmin << endl;
		}

		if (qlens->show_wtime) {
			if ((qlens->lumreg_max_it > 0) or (!qlens->get_lumreg_from_sbweights)) {
				wtime_opt = std::chrono::steady_clock::now() - wtime_opt0;
				if (qlens->mpi_id==0) cout << "Wall time for optimizing regparam with lum-weighted regularization: "  << wtime_opt.count() << endl;
				wtime_opt0 = std::chrono::steady_clock::now();
			}
		}
	}

	update_source_and_lensgrid_amplitudes(verbal);
	if ((qlens->use_lum_weighted_srcpixel_clustering) and (pre_srcgrid)) {
		if (qlens->show_wtime) {
			wtime_opt0 = std::chrono::steady_clock::now();
		}
		if (!qlens->use_saved_sbweights) calculate_subpixel_sbweights(qlens->save_sbweights_during_inversion,verbal); // only need to calculate sb weights for the initial grid, to be used to construct the final pixellation
		if (qlens->show_wtime) {
			wtime_opt = std::chrono::steady_clock::now() - wtime_opt0;
			if (qlens->mpi_id==0) cout << "Wall time for calculating pixel sbweights: "  << wtime_opt.count() << endl;
			wtime_opt0 = std::chrono::steady_clock::now();
		}
	}

	if (!dense_Fmatrix) {
		delete[] Fmatrix_copy;
		Fmatrix_copy = NULL;
	}

	//delete[] img_minus_sbprofile;
	//delete[] amplitude_vector_minchisq;
	//delete[] img_index_datapixels;
	return true;
}

void ImagePixelGrid::setup_regparam_optimization(const bool dense_Fmatrix)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	get_source_regparam_ptr(0,regparam_ptr);
	img_minus_sbprofile.resize(image_npixels);
	int i, pix_i, pix_j, img_index_fgmask;
	image_npixels_data = image_npixels;
	if (qlens->include_fgmask_in_inversion) {
		image_npixels_data = 0;
		for (i=0; i < image_npixels; i++) {
			pix_i = emask_pixels_i[i];
			pix_j = emask_pixels_j[i];
			if (image_data->foreground_mask_data[pix_i][pix_j]) image_npixels_data++;
		}
	}
	int j;
	img_index_datapixels.resize(image_npixels_data);
	for (i=0,j=0; i < image_npixels; i++) {
		pix_i = emask_pixels_i[i];
		pix_j = emask_pixels_j[i];
		img_index_fgmask = pixel_index_fgmask[pix_i][pix_j];
		img_minus_sbprofile[i] = p.image_surface_brightness[i] - p.sbprofile_surface_brightness[img_index_fgmask];
		if ((!qlens->include_fgmask_in_inversion) or (image_data->foreground_mask_data[pix_i][pix_j])) {
			img_index_datapixels[j++] = i;
		}
		if (((!qlens->include_imgfluxes_in_inversion) and (!qlens->include_srcflux_in_inversion)) and (qlens->n_ptsrc > 0)) img_minus_sbprofile[i] -= point_image_surface_brightness[i];
	}

	//amplitude_vector_minchisq = new double[n_amps];
	amplitude_vector_minchisq.resize(n_amps);
	regopt_chisqmin = 1e30;
	regopt_logdet = 1e30; // this will be changed during optimization

	if (dense_Fmatrix) {
		if (qlens->use_covariance_matrix) {
			Gmatrix_copy.resize(n_amps,n_amps);
			Dvector_cov_copy.resize(n_amps);
		} else {
			int ntot = n_amps*(n_amps+1)/2;
			Fmatrix_dense_copy.resize(n_amps,n_amps);
		}
	} else {
		if (Fmatrix_nn==0) die("Fmatrix_sparse length has not been set");
		Fmatrix_copy = new double[Fmatrix_nn];
	}

	Rmatrix_ptr = &Rmatrix_sparse;
	Rmatrix_index_ptr = &Rmatrix_index;
	Rmatrix_dense_ptr = &Rmatrix_dense;
}

void ImagePixelGrid::calculate_subpixel_sbweights(const bool save_sbweights, const bool verbal)
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();
	SB_Profile** sb_list = qlens->sb_list;
	int npix_in_mask;
	int *pixptr_i, *pixptr_j;
	if (qlens->include_fgmask_in_inversion) {
		npix_in_mask = image_npixels_emask;
		pixptr_i = emask_pixels_i;
		pixptr_j = emask_pixels_j;
	} else {
		npix_in_mask = image_npixels;
		pixptr_i = mask_pixels_i;
		pixptr_j = mask_pixels_j;
	}
	int i,j,k,n,l,m,nsubpix;
	double sb, max_sb = 1e-30;
	nsubpix = n_subpix_per_pixel;

	if (save_sbweights) {
		n_sbweights = 0;
		if (saved_sbweights != NULL) delete[] saved_sbweights;
		for (n=0; n < npix_in_mask; n++) {
			i = pixptr_i[n];
			j = pixptr_j[n];
			n_sbweights += nsubpix;
		}
		if (qlens->mpi_id==0) cout << "Saving " << n_sbweights << " sbweights" << endl;
		saved_sbweights = new double[n_sbweights];
		l=0;
	}

	bool at_least_one_lensed_src = false;
	for (k=0; k < qlens->n_sb; k++) {
		if (sb_list[k]->is_lensed) {
			at_least_one_lensed_src = true;
			break;
		}
	}

	if ((qlens->source_fit_mode==Delaunay_Source) and (delaunay_srcgrid == NULL)) die("delaunay_srcgrid has not been created");
	if ((qlens->source_fit_mode==Cartesian_Source) and (cartesian_srcgrid == NULL)) die("cartesian_srcgrid has not been created");

	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		#pragma omp for private(n,i,j,k,m,nsubpix,sb) schedule(static) 
		for (n=0; n < npix_in_mask; n++) {
			i = pixptr_i[n];
			j = pixptr_j[n];
			for (k=0; k < nsubpix; k++) {
				// This needs to be generalized so the weights can be created using different source modes (shapelet, sbprofile, etc.)...did I already accomplish this? (check)
				sb = 0;
				if (qlens->source_fit_mode==Delaunay_Source) sb += delaunay_srcgrid->interpolate_surface_brightness(subpixel_center_sourcepts[i][j][k],false,thread);
				else if (qlens->source_fit_mode==Cartesian_Source) sb += cartesian_srcgrid->find_lensed_surface_brightness_interpolate(subpixel_center_sourcepts[i][j][k],thread);
				else if (at_least_one_lensed_src) {
					for (m=0; m < qlens->n_sb; m++) {
						if (sb_list[m]->is_lensed) {
							sb += sb_list[m]->surface_brightness(subpixel_center_sourcepts[i][j][k][0],subpixel_center_sourcepts[i][j][k][1]);
						}
					}
				}
				if (sb < 0) sb = 0;

				if (sb > max_sb) {
					#pragma omp critical
					max_sb = sb;
				}
				subpixel_weights[i][j][k] = sb;
			}
		}
	}
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		for (k=0; k < nsubpix; k++) {
			subpixel_weights[i][j][k] /= max_sb;
			if (save_sbweights) saved_sbweights[l++] = subpixel_weights[i][j][k];
		}
	}
	if ((save_sbweights) and (qlens->mpi_id==0)) cout << "Pixel sb-weights saved" << endl;
}

void ImagePixelGrid::calculate_subpixel_distweights()
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();
	DelaunaySourceGrid_Params<double>& p = delaunay_srcgrid->assign_delaunay_srcgrid_param_object<double>();
	if (psf==NULL) return;
	double xc, yc, xc_approx, yc_approx, sig, rc;
	rc = p.distreg_rc;
	sig = find_approx_source_size(xc_approx,yc_approx);
	if (qlens->fix_lumreg_sig) sig = qlens->lumreg_sig;
	if (qlens->auto_lumreg_center) {
		xc = xc_approx;
		yc = yc_approx;
	} else {
		if (qlens->lensed_lumreg_center) {
			lensvector<double> xl;
			xl[0] = p.distreg_xcenter;
			xl[1] = p.distreg_ycenter;
			if (psf != 0) {
				xl[0] -= psf->psf_params.psf_offset_x;
				xl[1] -= psf->psf_params.psf_offset_y;
			}
			qlens->find_sourcept<double>(xl,xc,yc,0,imggrid_zfactors,imggrid_betafactors);
			//if ((verbal) and (qlens->mpi_id==0)) cout << "center coordinates in source plane: xc=" << xc << ", yc=" << yc << endl;
			if (qlens->lensed_lumreg_rc) {
				int i, phi_nn = 24;
				double phi, phi_step = M_2PI/(phi_nn-1);
				double xc2, yc2;
				rc = 0;
				for (i=0, phi=0; i < phi_nn; i++, phi += phi_step) {
					xl[0] = p.distreg_xcenter + p.distreg_rc*cos(phi);
					xl[1] = p.distreg_ycenter + p.distreg_rc*sin(phi);
					if (psf != 0) {
						xl[0] -= psf->psf_params.psf_offset_x;
						xl[1] -= psf->psf_params.psf_offset_y;
					}
					qlens->find_sourcept<double>(xl,xc2,yc2,0,imggrid_zfactors,imggrid_betafactors);
					rc += SQR(xc2-xc)+SQR(yc2-yc);
				}
				rc = sqrt(rc/phi_nn);
			}
			// this is all repeated in function calculate_distreg_srcpixel_weights(...), which is not great...find a way to consolidate?
			//if ((verbal) and (qlens->mpi_id==0)) cout << "estimated lumreg_rc mapped to source plane: " << rc << endl;

		} else {
			xc = p.distreg_xcenter; yc = p.distreg_ycenter;
		}
	}

	int npix_in_mask;
	int *pixptr_i, *pixptr_j;
	if (qlens->include_fgmask_in_inversion) {
		npix_in_mask = image_npixels_emask;
		pixptr_i = emask_pixels_i;
		pixptr_j = emask_pixels_j;
	} else {
		npix_in_mask = image_npixels;
		pixptr_i = mask_pixels_i;
		pixptr_j = mask_pixels_j;
	}
	int i,j,k,n,l,nsubpix,n_weights;
	nsubpix = n_subpix_per_pixel;

	n_weights = 0;
	if (saved_sbweights != NULL) delete[] saved_sbweights;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		n_weights += nsubpix;
	}
	double *scaled_dists = new double[n_weights];
	lensvector<double> **srcpts = new lensvector<double>*[n_weights];
	l=0;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		for (k=0; k < nsubpix; k++) {
			srcpts[l++] = &subpixel_center_sourcepts[i][j][k];
		}
	}
	calculate_srcpixel_scaled_distances(xc,yc,sig,scaled_dists,srcpts,n_weights,p.distreg_e1,p.distreg_e2);

	l=0;
	double scaled_rcsq = SQR(rc/sig);
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		for (k=0; k < nsubpix; k++) {
			subpixel_weights[i][j][k] = exp(-pow(sqrt(SQR(scaled_dists[l++]) + scaled_rcsq),p.regparam_lum_index));
			//cout << "WEIGHT " << (l-1) << ": " << subpixel_weights[i][j][k] << " " << scaled_dists[l-1] << " " << (*srcpts[l-1])[0] << " " << (*srcpts[l-1])[1] << " " << xc << " " << yc << " " << sig << endl;
		}
	}
	delete[] scaled_dists;
	delete[] srcpts;
}

void ImagePixelGrid::calculate_lumreg_srcpixel_weights(const bool use_sbweights)
{
	double lumfac, max_sb=-1e30;
	int i;
	DelaunaySourceGrid_Params<double>& p = delaunay_srcgrid->assign_delaunay_srcgrid_param_object<double>();
	if (use_sbweights) find_srcpixel_weights();
	for (i=0; i < source_npixels; i++) {
		if (amplitude_vector[i] > max_sb) max_sb = amplitude_vector[i];
	}
	if (qlens->use_lum_weighted_regularization) {
		for (i=0; i < source_npixels; i++) {
			if (qlens->lum_weight_function==0) {
				if (amplitude_vector[i]==max_sb) reg_weight_factor[i] = 1;
				else {
					lumfac = (amplitude_vector[i] > 0) ? pow(1 - amplitude_vector[i]/max_sb,p.regparam_lum_index) : 1;
					reg_weight_factor[i] = exp(-p.regparam_lsc*lumfac);
				}
			} else if (qlens->lum_weight_function==1) {
				lumfac = (amplitude_vector[i] > 0) ? 1 - pow(amplitude_vector[i]/max_sb,p.regparam_lum_index) : 1;
				reg_weight_factor[i] = exp(-p.regparam_lsc*lumfac);
			} else {
				if (p.regparam_lum_index==0) {
					reg_weight_factor[i] = exp(-p.regparam_lsc);
				} else {
					lumfac = (amplitude_vector[i] > 0) ? pow(1-pow(amplitude_vector[i]/max_sb,1.0/p.regparam_lum_index),p.regparam_lum_index) : 1;
					reg_weight_factor[i] = exp(-p.regparam_lsc*lumfac);
				}
			}
		}
	}
}

void ImagePixelGrid::calculate_distreg_srcpixel_weights(const double xc_in, const double yc_in, const double sig, const bool verbal)
{
	if (delaunay_srcgrid == NULL) die("Delaunay source grid has not been created");
	DelaunaySourceGrid_Params<double>& p = delaunay_srcgrid->assign_delaunay_srcgrid_param_object<double>();
	if (psf==NULL) return;
	double xc, yc, rc;
	rc = p.distreg_rc;
	if (qlens->auto_lumreg_center) {
		if (qlens->lumreg_center_from_ptsource) {
			if (qlens->n_ptsrc==0) die("no source points have been defined");
			xc = qlens->ptsrc_list[0]->ptsrc_params.pos[0];
			yc = qlens->ptsrc_list[0]->ptsrc_params.pos[1];
		} else {
			xc = xc_in;
			yc = yc_in;
		}
	} else {
		if (qlens->lensed_lumreg_center) {
			lensvector<double> xl;
			xl[0] = p.distreg_xcenter;
			xl[1] = p.distreg_ycenter;
			if (psf != 0) {
				xl[0] -= psf->psf_params.psf_offset_x;
				xl[1] -= psf->psf_params.psf_offset_y;
			}
			qlens->find_sourcept<double>(xl,xc,yc,0,imggrid_zfactors,imggrid_betafactors);
			if ((verbal) and (qlens->mpi_id==0)) cout << "center coordinates in source plane: xc=" << xc << ", yc=" << yc << endl;
			if ((qlens->lensed_lumreg_rc) and (p.distreg_rc > 0)) {
				int i, phi_nn = 24;
				double phi, phi_step = M_2PI/(phi_nn-1);
				double xc2, yc2;
				rc = 0;
				for (i=0, phi=0; i < phi_nn; i++, phi += phi_step) {
					xl[0] = p.distreg_xcenter + p.distreg_rc*cos(phi);
					xl[1] = p.distreg_ycenter + p.distreg_rc*sin(phi);
					if (psf != 0) {
						xl[0] -= psf->psf_params.psf_offset_x;
						xl[1] -= psf->psf_params.psf_offset_y;
					}
					qlens->find_sourcept<double>(xl,xc2,yc2,0,imggrid_zfactors,imggrid_betafactors);
					rc += SQR(xc2-xc)+SQR(yc2-yc);
				}
				rc = sqrt(rc/phi_nn);
				if ((verbal) and (qlens->mpi_id==0)) cout << "estimated lumreg_rc mapped to source plane: " << rc << endl;
			}
		} else {
			xc = p.distreg_xcenter; yc = p.distreg_ycenter;
		}
	}
	double *scaled_dists = new double[source_npixels];
	lensvector<double> **srcpts = new lensvector<double>*[source_npixels];
	for (int i=0; i < source_npixels; i++) srcpts[i] = &p.gridpts[i];
	calculate_srcpixel_scaled_distances(xc,yc,sig,scaled_dists,srcpts,source_npixels,p.distreg_e1,p.distreg_e2);
	double scaled_rcsq = SQR(rc/sig);
	for (int i=0; i < source_npixels; i++) {
		if (qlens->lum_weight_function==0) {
			if (Rmatrix_sparse != NULL) { delete[] Rmatrix_sparse; Rmatrix_sparse = NULL; }
			if (Rmatrix_index != NULL) { delete[] Rmatrix_index; Rmatrix_index = NULL; }
			reg_weight_factor[i] = exp(-p.regparam_lsc*pow(sqrt(SQR(scaled_dists[i]) + scaled_rcsq),p.regparam_lum_index));
		} else {
			die("lumweight_func greater than 0 not supported in dist-weighted regularization");
		}
	}

	delete[] scaled_dists;
	delete[] srcpts;
}

void ImagePixelGrid::calculate_srcpixel_scaled_distances(const double xc, const double yc, const double sig, double *dists, lensvector<double> **srcpts, const int nsrcpts, const double e1, const double e2)
{
	double angle;
	if (e1==0) {
		if (e2 > 0) angle = M_HALFPI;
		else if (e2==0) angle = 0.0;
		else angle = -M_HALFPI;
	} else {
		angle = atan2(e2,e1);
	}
	angle = 0.5*angle;
	double q = 1 - sqrt(e1*e1 + e2*e2);
	if (q < 0.01) q = 0.01; // in case e1, e2 too large and you get a negative q

	double costh, sinth;
	costh=cos(angle);
	sinth=sin(angle);
	double xval, yval, xprime, yprime;
	for (int i=0; i < nsrcpts; i++) {
		xval = (*srcpts[i])[0]-xc;
		yval = (*srcpts[i])[1]-yc;
		xprime = xval*costh + yval*sinth;
		yprime = -xval*sinth + yval*costh;

		dists[i] = sqrt(q*xprime*xprime+yprime*yprime/q)/sig;
		//cout << "HUH? " << xval << " " << yval << " " << xprime << " " << yprime << " " << dists[i] << endl;
	}
}

void ImagePixelGrid::calculate_mag_srcpixel_weights()
{
	int i;
	double logmag,logmag_max = -1e30;
	if (imggrid_index==0) // at the moment, this is only set up for the first source being modeled
	{
		DelaunaySourceGrid_Params<double>& p = delaunay_srcgrid->assign_delaunay_srcgrid_param_object<double>();
		for (i=0; i < delaunay_srcgrid->n_gridpts; i++) {
			logmag = -log(delaunay_srcgrid->inv_magnification[i])/ln10;
			if (logmag > logmag_max) logmag_max = logmag;
		}
		for (i=0; i < delaunay_srcgrid->n_gridpts; i++) {
			if (delaunay_srcgrid->active_pixel[i]) {
				logmag = -log(delaunay_srcgrid->inv_magnification[i])/ln10;
				if (p.mag_weight_index != 0) {
					reg_weight_factor[i] *= exp(-p.mag_weight_sc*pow((logmag_max-logmag),p.mag_weight_index));
				}
			}
		}
	}
}


void ImagePixelGrid::find_srcpixel_weights()
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();
	int npix_in_mask;
	int *pixptr_i, *pixptr_j;
	if (qlens->include_fgmask_in_inversion) {
		npix_in_mask = image_npixels_emask;
		pixptr_i = emask_pixels_i;
		pixptr_j = emask_pixels_j;
	} else {
		npix_in_mask = image_npixels;
		pixptr_i = mask_pixels_i;
		pixptr_j = mask_pixels_j;
	}
	int i,j,k,n,indx,trinum,nsubpix;
	nsubpix = n_subpix_per_pixel;

	int nsrcpix = delaunay_srcgrid->n_gridpts; // note, this might not be the same as source_npixels if there are inactive source pixels
	double *srcpixel_weights = new double[nsrcpix];
	int *srcpixel_nimgpts = new int[nsrcpix];
	for (i=0; i < nsrcpix; i++) {
		srcpixel_weights[i] = 0;
		srcpixel_nimgpts[i] = 0;
	}

	bool inside_triangle;
	lensvector<double> *pt;
	#pragma omp parallel for private(n,i,j,k,indx,trinum,inside_triangle,pt) schedule(static) 
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		for (k=0; k < nsubpix; k++) {
			pt = &subpixel_center_sourcepts[i][j][k];
			// This needs to be generalized so the weights can be created using different source modes (shapelet, sbprofile, etc.)
			inside_triangle = false;
			trinum = delaunay_srcgrid->search_grid(0,*pt,inside_triangle); // maybe you can speed this up later by choosing a better initial triangle
			indx = delaunay_srcgrid->find_closest_vertex(trinum,*pt);
			#pragma omp critical
			{
				srcpixel_weights[indx] += subpixel_weights[i][j][k];
				srcpixel_nimgpts[indx]++;
			}
		}
	}

	indx=0;
	for (i=0; i < nsrcpix; i++) {
		if (delaunay_srcgrid->active_pixel[i]) amplitude_vector[indx++] = srcpixel_weights[i] / srcpixel_nimgpts[i];
	}
}

void ImagePixelGrid::load_pixel_sbweights()
{
	int npix_in_mask;
	int *pixptr_i, *pixptr_j;
	if (qlens->include_fgmask_in_inversion) {
		npix_in_mask = image_npixels_emask;
		pixptr_i = emask_pixels_i;
		pixptr_j = emask_pixels_j;
	} else {
		npix_in_mask = image_npixels;
		pixptr_i = mask_pixels_i;
		pixptr_j = mask_pixels_j;
	}
	int i,j,k,n,l,nsubpix;
	nsubpix = n_subpix_per_pixel;

	int nweights=0;
	for (int n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		nweights += nsubpix;
	}
	if (nweights != n_sbweights) die("number of subpixels (%i) doesn't match number of saved sb-weights (%i)",nweights,n_sbweights);

	l=0;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		for (k=0; k < nsubpix; k++) {
			subpixel_weights[i][j][k] = saved_sbweights[l++];
		}
	}
}

double ImagePixelGrid::chisq_regparam(const double logreg)
{
	double cov_inverse, cov_inverse_bg, chisq; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (!qlens->use_noise_map) {
		if (qlens->background_pixel_noise==0) cov_inverse_bg = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
		else cov_inverse_bg = 1.0/SQR(qlens->background_pixel_noise);
	}

	(*regparam_ptr) = pow(10,logreg);
	int i,j,k,index2;

	for (i=0; i < Fmatrix_nn; i++) {
		Fmatrix_copy[i] = Fmatrix_sparse[i];
	}

	for (i=src_npixel_start, index2=0; index2 < source_npixels_inv; i++, index2++) {
		Fmatrix_copy[i] += (*regparam_ptr)*(*Rmatrix_ptr)[index2];
		for (j=(*Rmatrix_index_ptr)[index2]; j < (*Rmatrix_index_ptr)[index2+1]; j++) {
			for (k=Fmatrix_index[i]; k < Fmatrix_index[i+1]; k++) {
				if ((*Rmatrix_index_ptr)[j]==Fmatrix_index[k]) {
					Fmatrix_copy[k] += (*regparam_ptr)*(*Rmatrix_ptr)[j];
				}
			}
		}
	}

	double Fmatrix_logdet;

	if (qlens->sparse_solver==MUMPS) invert_lens_mapping_MUMPS(Fmatrix_logdet,false,true);
	else if (qlens->sparse_solver==UMFPACK) invert_lens_mapping_UMFPACK(Fmatrix_logdet,false,true);
	else if (qlens->sparse_solver==EIGEN_SPARSE) invert_lens_mapping_EIGEN_sparse(Fmatrix_logdet,false,true);
	else die("can only use MUMPS, UMFPACK or Eigen for sparse inversions with optimize_regparam on");

	//double temp_img, Ed_times_two=0,Es_times_two=0;
	double temp_img, Ed_times_two=0;

	#pragma omp parallel for private(temp_img,i,j,cov_inverse) schedule(static) reduction(+:Ed_times_two)
	for (i=0; i < image_npixels; i++) {
		if (qlens->use_noise_map) cov_inverse = imgpixel_covinv_vector[i];
		else cov_inverse = cov_inverse_bg;
		temp_img = 0;
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			temp_img += Lmatrix_sparse[j]*amplitude_vector[Lmatrix_index[j]];
		}

		// NOTE: this chisq does not include foreground mask pixels that lie outside the primary mask, since those pixels don't contribute to determining the regularization
		Ed_times_two += SQR(temp_img - img_minus_sbprofile[i])*cov_inverse;
	}
	//for (i=0; i < source_npixels; i++) {
		//Es_times_two += (*Rmatrix_ptr)[i]*SQR(amplitude_vector[i]);
		//for (j=(*Rmatrix_index_ptr)[i]; j < (*Rmatrix_index_ptr)[i+1]; j++) {
			//Es_times_two += 2 * amplitude_vector[i] * (*Rmatrix_ptr)[j] * amplitude_vector[(*Rmatrix_index_ptr)[j]]; // factor of 2 since matrix is symmetric
		//}
	//}
	//cout << "chisqreg: " << (Ed_times_two + (*regparam_ptr)*Es_times_two + Fmatrix_log_determinant - n_amps*log((*regparam_ptr)) - Rmatrix_log_determinant) << endl;
	//cout << "reg*Es_times_two=" << ((*regparam_ptr)*Es_times_two) << " n_shapelets*log(regparam)=" << (-n_amps*log((*regparam_ptr))) << " -det(Rmatrix)=" << (-Rmatrix_log_determinant) << " log(Fmatrix_sparse)=" << Fmatrix_logdet << endl;

	//chisq = (Ed_times_two + (*regparam_ptr)*Es_times_two + Fmatrix_logdet - source_npixels*log((*regparam_ptr)) - Rmatrix_log_determinant);

	double loglike_reg = calculate_regularization_prior_term(regparam_ptr,false);
	//cout << "regparam: " << (*regparam_ptr) << " loglike_reg=" << loglike_reg << " Flogdet=" << Fmatrix_logdet << endl;
	chisq = Ed_times_two + loglike_reg;
	chisq += Fmatrix_logdet;

	if (chisq < regopt_chisqmin) {
		regopt_chisqmin = chisq;
		for (i=0; i < n_amps; i++) amplitude_vector_minchisq[i] = amplitude_vector[i];
		regopt_logdet = Fmatrix_logdet;
	}
	return chisq;
}

double ImagePixelGrid::chisq_regparam_dense(const double logreg)
{
	double chisq, logdet, cov_inverse, cov_inverse_bg; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (qlens->background_pixel_noise==0) cov_inverse_bg = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
	else cov_inverse_bg = 1.0/SQR(qlens->background_pixel_noise);

	(*regparam_ptr) = pow(10,logreg);
	int i,j;
	int npixels, start_indx, end_indx;
	npixels = source_npixels_inv;
	start_indx = src_npixel_start;

	int row_indx_offset;
	if (start_indx==0) row_indx_offset = 0;
	else {
		row_indx_offset = start_indx*n_amps - start_indx*(start_indx-1)/2; // the sum of elements of rows prior to the row=start_indx
	}

	if (qlens->dense_Rmatrix) {
		end_indx = start_indx + npixels;
		if (!qlens->use_covariance_matrix) {
			Fmatrix_dense_copy = Fmatrix_dense;
			Fmatrix_dense_copy += (*regparam_ptr)*(*Rmatrix_dense_ptr);
			int n_extra_amps = n_amps - npixels;
			double *Fptr, *Rptr;
		} else {
			Gmatrix_copy = Gmatrix;
			Gmatrix_copy /= (*regparam_ptr);
			Dvector_cov_copy = Dvector_cov / (*regparam_ptr);
			for (i=0; i < npixels; i++) { // additional source amplitudes (beyond npixels) are not regularized
				Gmatrix_copy(i,i) += 1.0;
			}
		}
	} else {
		Fmatrix_dense_copy = Fmatrix_dense;
		int k;
		for (i=0; i < npixels; i++) {
			Fmatrix_dense_copy(start_indx+i,start_indx+i) += (*regparam_ptr)*(*Rmatrix_ptr)[i];
			for (k=(*Rmatrix_index_ptr)[i]; k < (*Rmatrix_index_ptr)[i+1]; k++) {
				Fmatrix_dense_copy(start_indx+i,start_indx+(*Rmatrix_index_ptr)[k]) += (*regparam_ptr)*(*Rmatrix_ptr)[k];
				//Fmatrix_dense_copy(start_indx+(*Rmatrix_index_ptr)[k],start_indx+i) += (*regparam_ptr)*(*Rmatrix_ptr)[k];
			}
			row_indx_offset += n_amps-start_indx-i;
		}
	}

	double Fmatrix_logdet, Gmatrix_logdet;
	if (!qlens->use_covariance_matrix) {
		Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> Fmatrix_llt(Fmatrix_dense_copy);
		Eigen::MatrixXd lltmat(n_amps,n_amps);
		lltmat = Fmatrix_llt.matrixL();
		amplitude_vector = Fmatrix_llt.solve(Dvector);
		Fmatrix_logdet = 0;
		for (i=0; i < n_amps; i++) Fmatrix_logdet += log(abs(lltmat(i,i)));
		Fmatrix_logdet *= 2;
	} else {
		for (i=0; i < n_amps; i++) amplitude_vector[i] = Dvector_cov_copy[i];
		Eigen::PartialPivLU<Eigen::MatrixXd> lu(Gmatrix_copy);
		if(lu.determinant()==0.0) warn("Matrix was not invertible");
		amplitude_vector = lu.solve(amplitude_vector);
		const auto& LU = lu.matrixLU();
		Gmatrix_logdet = 0;
		for (int i=0; i < n_amps; i++) {
			Gmatrix_logdet += log(abs(LU(i,i)));
		}
	}

	double temp_img, Ed_times_two=0;
	//double *Lmatptr;
	//double *tempsrcptr = amplitude_vector;
	//double *tempsrc_end = amplitude_vector + n_amps;
	int img_index;

	#pragma omp parallel for private(temp_img,img_index,i,j,cov_inverse) schedule(static) reduction(+:Ed_times_two)
	for (img_index=0; img_index < image_npixels_data; img_index++) {
		i = img_index_datapixels[img_index];
		temp_img = 0;
		if (qlens->use_noise_map) {
			cov_inverse = imgpixel_covinv_vector[i];
		} else {
			cov_inverse = cov_inverse_bg;
		}
		if ((qlens->source_fit_mode==Shapelet_Source) or (qlens->matrix_format==DENSE)) {
			// even if using a pixellated source, if matrix_format is set to DENSE, only the dense form of the Lmatrix_sparse has been convolved with the PSF, so this form must be used
			//Lmatptr = (Lmatrix_dense0.pointer())[i];
			//Lmatptr = Lmatrix_dense.row(i).data();
			//tempsrcptr = amplitude_vector;
			//while (tempsrcptr != tempsrc_end) {
				//temp_img += (*(Lmatptr++))*(*(tempsrcptr++));
			//}
			//temp_img += Lmatrix_dense.row(i).dot(amplitude_vector);
			temp_img += Lmatrix_trans_dense.col(i).dot(amplitude_vector);
		} else {
			for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
				temp_img += Lmatrix_sparse[j]*amplitude_vector[Lmatrix_index[j]];
			}
		}
		// NOTE: this chisq does not include foreground mask pixels that lie outside the primary mask, since those pixels don't contribute to determining the regularization
		Ed_times_two += SQR(temp_img - img_minus_sbprofile[i])*cov_inverse;
	}

	double loglike_reg = calculate_regularization_prior_term(regparam_ptr,false);
	//cout << "regparam: " << (*regparam_ptr) << " loglike_reg=" << loglike_reg << " Flogdet=" << Fmatrix_logdet << endl;
	chisq = Ed_times_two + loglike_reg;
	logdet = (qlens->use_covariance_matrix) ? Gmatrix_logdet : Fmatrix_logdet;
	chisq += logdet;
	//cout << "chisq0=" << Ed_times_two << " regterms=" << loglike_reg << " F_logdet=" << Fmatrix_logdet << " logev=" << chisq << endl;

	if (chisq < regopt_chisqmin) {
		regopt_chisqmin = chisq;
		for (i=0; i < n_amps; i++) amplitude_vector_minchisq[i] = amplitude_vector[i];
		regopt_logdet = logdet;
	}
	return chisq;
}

double ImagePixelGrid::brents_min_method(double (ImagePixelGrid::*func)(const double), const double ax, const double bx, const double tol, const bool verbal)
{
	// (NOTE: I've found that with optimizing the regularization, it always seems to converge even if we ONLY do parabolic
	// interpolation after the first two iterations. But leaving Brent's method as-is, just to be safe)
	double a,b,xstep=0.0,etemp,fu,fwprev,fw,fx;
	double p,q,r,tol1,tol2,u,wprev,w,x,xmid;
	double e=0.0;

	const double CGOLD = 0.3819660; // golden ratio
	const double ZEPS = 1.0e-10;
	const double ROOTPREC = 1.0e-8; // square root of machine precision for double floating points

	a = ax;
	b = bx;
	// in what follows, x is the point with the least function value thus far, while w is the point with the second least function value thus far
	x=w=wprev=bx-CGOLD*(bx-ax); // start with point closer to the higher regularization (sometimes seems to converge better), using golden ratio
	fw=fwprev=fx=(this->*func)(x);
	//cout << "Just evaluated f(" << x << ")=" << fx << end;
	for (int iter=0; iter < qlens->max_regopt_iterations; iter++)
	{
		xmid=0.5*(a+b);
		tol2 = 2.0 * ((tol1=tol*abs(x)) + ZEPS);
		if (abs(x-xmid) <= (tol2-0.5*(b-a))) {
			if ((verbal) and (qlens->mpi_id==0)) {
				cout << "Number of regparam optimizing log(L) evaluations: " << (iter+1) << endl;
				if ((x-ax < tol2) or (bx-x < tol2)) cout << "NOTE: Brent's method converged to edge of bracket in log(regparam), indicating a minimum was not bracketed" << endl;
			}
			return x;
		}
		if ((w != wprev) and (abs(e) > tol1)) {
			// try (inverse) parabolic interpolation 
			r = (x-w)*(fx-fwprev);
			q = (x-wprev)*(fx-fw);
			p = (x-wprev)*q - (x-w)*r;
			q = 2.0*(q-r);
			if (q > 0.0) p = -p;
			q = abs(q);
			etemp = e;
			e = xstep;
			if ((abs(p) >= abs(0.5*q*etemp)) or (p <= q*(a-x)) or (p >= q*(b-x))) {
				// parabolic step either went out of the bounding interval, OR it was greater than half the previous step,
				// so we'll switch to a golden section step instead
				xstep = CGOLD*(e=(x >= xmid ? a-x : b-x));
			} else {
				// parabolic fit looked good, so take a parabolic step
				xstep = p/q;
				u = x + xstep;
				if ((u-a < tol2) or (b-u < tol2))
					xstep = ((xmid-x) >= 0 ? (tol1 >= 0 ? tol1 : -tol1) : (tol1 >= 0 ? -tol1 : tol1));
			}
		} else {
			// take golden section step
			xstep = CGOLD*(e=(x >= xmid ? a-x : b-x));
		}
		if (abs(xstep) >= ROOTPREC) {
			u = x + xstep;
		} else {
			// It is pointless to have steps smaller than the sqrt of machine precision, so just have a step equal to sqrt(prec)
			u = x + (xstep >= 0 ? ROOTPREC : (-ROOTPREC));
		}
		fu = (this->*func)(u);
		//cout << "Just evaluated f(" << u << ")=" << fu << end;
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			wprev=w; w=x; x=u;
			fwprev=fw; fw=fx; fx=fu;
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw or w == x) {
				wprev = w;
				fwprev = fw;
				w = u;
				fw = fu;
			} else if ((fu <= fwprev) or (wprev == x) or (wprev == w)) {
				wprev = u;
				fwprev = fu;
			}
		}
	}
	if ((verbal) and (qlens->mpi_id==0)) {
		warn("Brent's Method reached maximum number of iterations for optimizing regparam");
	}
	return x;
}

void ImagePixelGrid::invert_lens_mapping_dense(bool verbal)
{
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	int i,j;
	if (!qlens->use_covariance_matrix) {
		int i,j,idx = 0;
		Eigen::MatrixXd lltmat(n_amps,n_amps);
		if (qlens->use_non_negative_least_squares) {
#ifdef USE_EIGEN_INV_NNLS
			Eigen::MatrixXd lnnlsmat(n_amps,n_amps);

			Eigen::NNLS<Eigen::MatrixXd> Fmatrix_nnls(Fmatrix_dense,max_nnls_iterations,nnls_tolerance);
			Fmatrix_nnls.setTolerance(nnls_tolerance);
			//amplitude_vector = Fmatrix_nnls.solve(dvector_eigen);
			amplitude_vector = Fmatrix_nnls.solve(Dvector);
			if ((qlens->mpi_id==0) and (verbal)) {
				cout << "Number of iterations required: " << Fmatrix_nnls.iterations() << endl;
				cout << "Tolerance: " << Fmatrix_nnls.tolerance() << endl;
			}
			Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> Fmatrix_llt(Fmatrix_dense);
			lltmat = Fmatrix_llt.matrixL();
#else
			die("qlens must be compiled with Eigen NNLS support to use non-negative least squares fitting");
#endif
		} else {
			Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> Fmatrix_llt(Fmatrix_dense);
			lltmat = Fmatrix_llt.matrixL();
			amplitude_vector = Fmatrix_llt.solve(Dvector);
		}
		Fmatrix_log_determinant = 0;
		for (int i=0; i < n_amps; i++) Fmatrix_log_determinant += log(abs(lltmat(i,i)));
		Fmatrix_log_determinant *= 2;
	} else {
		amplitude_vector = Dvector_cov;
		Eigen::PartialPivLU<Eigen::MatrixXd> lu(Gmatrix);
		if(lu.determinant()==0.0) warn("Matrix was not invertible");
		amplitude_vector = lu.solve(amplitude_vector);
		const auto& LU = lu.matrixLU();
		Gmatrix_log_determinant = 0;
		for (int i=0; i < n_amps; i++) {
			Gmatrix_log_determinant += log(abs(LU(i,i)));
		}

		//cout << "DETERMINANTS: " << Fmatrix_log_determinant << " " << Gmatrix_log_determinant << " " << (Gmatrix_log_determinant+Rmatrix_log_determinant) << endl;
	}
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for inverting Fmatrix_sparse: "  << wtime.count() << endl;
		wtime0 = std::chrono::steady_clock::now();
	}
	update_source_and_lensgrid_amplitudes(verbal);
}

void ImagePixelGrid::invert_lens_mapping_CG_method(bool verbal) {
	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	int i,j,k;
	double *temp = new double[n_amps];
	// it would be prettier to just pass the MPI communicator in, and have CG_sparse figure out the rank and # of processes internally--implement this later
	CG_sparse cg_method(Fmatrix_sparse,Fmatrix_index,1e-4,100000,qlens->inversion_nthreads,qlens->group_np,qlens->group_id);
	for (int i=0; i < n_amps; i++) temp[i] = 0;
	if ((qlens->regularization_method != None) and (source_npixels > 0))
		cg_method.set_determinant_mode(true);
	else cg_method.set_determinant_mode(false);
	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for setting up CG method: "  << wtime.count() << endl;
		wtime0 = std::chrono::steady_clock::now();
	}
	cg_method.solve(Dvector.data(),temp);

	for (int i=0; i < n_amps; i++) {
		if ((qlens->background_pixel_noise==0) and (temp[i] < 0)) temp[i] = 0; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
		amplitude_vector[i] = temp[i];
	}

	if ((qlens->regularization_method != None) and (source_npixels > 0)) {
		cg_method.get_log_determinant(Fmatrix_log_determinant);
		if ((qlens->mpi_id==0) and (verbal)) cout << "log determinant = " << Fmatrix_log_determinant << endl;

		Rmatrix_ptr = &Rmatrix_sparse;
		Rmatrix_index_ptr = &Rmatrix_index;

		for (int i=0; i < n_src_inv; i++) {
			CG_sparse cg_det((*Rmatrix_ptr),(*Rmatrix_index_ptr),3e-4,100000,qlens->inversion_nthreads,qlens->group_np,qlens->group_id);
			Rmatrix_log_determinant = cg_det.calculate_log_determinant();
			if ((qlens->mpi_id==0) and (verbal)) cout << "Rmatrix log determinant = " << Rmatrix_log_determinant << endl;
		}
	}

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for inverting Fmatrix_sparse: "  << wtime.count() << endl;
	}

	int iterations;
	double error;
	cg_method.get_error(iterations,error);
	if ((qlens->mpi_id==0) and (verbal)) cout << iterations << " iterations, error=" << error << endl << endl;

	delete[] temp;
	update_source_and_lensgrid_amplitudes(verbal);
}

void ImagePixelGrid::invert_lens_mapping_EIGEN_sparse(double& logdet, const bool verbal, const bool use_copy)
{
	double *Fmatptr = (use_copy==true) ? Fmatrix_copy : Fmatrix_sparse;

	std::vector<Eigen::Triplet<double>> triplets;

	int i,k;
	for (i=0; i < n_amps; i++) {
	  triplets.emplace_back(i, i, Fmatptr[i]);
	}

	for (i=0; i < n_amps; i++) {
		 for (k=Fmatrix_index[i]; k < Fmatrix_index[i+1]; k++) {
			  triplets.emplace_back(i, Fmatrix_index[k], Fmatptr[k]);
			  triplets.emplace_back(Fmatrix_index[k], i, Fmatptr[k]);
		 }
	}
	Eigen::SparseMatrix<double> F(n_amps,n_amps);
	F.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	using VectorXd  = Eigen::VectorXd;
	solver.compute(F);
	if(solver.info() != Eigen::Success) warn("Sparse Cholesky factorization failed");
	VectorXd s = solver.solve(Dvector);
	for (i=0; i < n_amps; i++) {
		amplitude_vector[i] = s(i);
	}
	const auto& Lview = solver.matrixL();
	const auto& L = Lview.nestedExpression(); // underlying sparse matrix

	logdet = 0.0;
	for (i=0; i < L.rows(); i++) {
		logdet += std::log(L.coeff(i,i));
	}
	logdet *= 2;

	update_source_and_lensgrid_amplitudes(verbal);
}

void ImagePixelGrid::invert_lens_mapping_UMFPACK(double& logdet, const bool verbal, const bool use_copy)
{
#ifndef USE_UMFPACK
	die("QLens requires compilation with UMFPACK for factorization");
#else

	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	int i,j;
	double *Fmatptr = (use_copy==true) ? Fmatrix_copy : Fmatrix_sparse;

   double *null = (double *) NULL ;
	double *temp = new double[n_amps];
   void *Symbolic, *Numeric ;
	double Control [UMFPACK_CONTROL];
	double Info [UMFPACK_INFO];
    umfpack_di_defaults (Control) ;
	 Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;

	int Fmatrix_nonzero_elements = Fmatrix_index[n_amps]-1;
	int Fmatrix_offdiags = Fmatrix_index[n_amps]-1-n_amps;
	int Fmatrix_unsymmetric_nonzero_elements = n_amps + 2*Fmatrix_offdiags;
	if (Fmatrix_nonzero_elements==0) {
		cout << "nsource_pixels=" << n_amps << endl;
		die("Fmatrix has zero size");
	}

	// Now we construct the transpose of Fmatrix so we can cast it into "unsymmetric" format for UMFPACK (by including offdiagonals on either side of diagonal elements)

	double *Fmatrix_transpose = new double[Fmatrix_nonzero_elements+1];
	int *Fmatrix_transpose_index = new int[Fmatrix_nonzero_elements+1];

	int k,jl,jm,jp,ju,m,n2,noff,inc,iv;
	double v;

	n2=Fmatrix_index[0];
	for (j=0; j < n2-1; j++) Fmatrix_transpose[j] = Fmatptr[j];
	int n_offdiag = Fmatrix_index[n2-1] - Fmatrix_index[0];
	int *offdiag_indx = new int[n_offdiag];
	int *offdiag_indx_transpose = new int[n_offdiag];
	for (i=0; i < n_offdiag; i++) offdiag_indx[i] = Fmatrix_index[n2+i];
	indexx(offdiag_indx,offdiag_indx_transpose,n_offdiag);
	for (j=n2, k=0; j < Fmatrix_index[n2-1]; j++, k++) {
		Fmatrix_transpose_index[j] = offdiag_indx_transpose[k];
	}
	jp=0;
	for (k=Fmatrix_index[0]; k < Fmatrix_index[n2-1]; k++) {
		m = Fmatrix_transpose_index[k] + n2;
		Fmatrix_transpose[k] = Fmatptr[m];
		for (j=jp; j < Fmatrix_index[m]+1; j++)
			Fmatrix_transpose_index[j]=k;
		jp = Fmatrix_index[m] + 1;
		jl=0;
		ju=n2-1;
		while (ju-jl > 1) {
			jm = (ju+jl)/2;
			if (Fmatrix_index[jm] > m) ju=jm; else jl=jm;
		}
		Fmatrix_transpose_index[k]=jl;
	}
	for (j=jp; j < n2; j++) Fmatrix_transpose_index[j] = Fmatrix_index[n2-1];
	for (j=0; j < n2-1; j++) {
		jl = Fmatrix_transpose_index[j+1] - Fmatrix_transpose_index[j];
		noff=Fmatrix_transpose_index[j];
		inc=1;
		do {
			inc *= 3;
			inc++;
		} while (inc <= jl);
		do {
			inc /= 3;
			for (k=noff+inc; k < noff+jl; k++) {
				iv = Fmatrix_transpose_index[k];
				v = Fmatrix_transpose[k];
				m=k;
				while (Fmatrix_transpose_index[m-inc] > iv) {
					Fmatrix_transpose_index[m] = Fmatrix_transpose_index[m-inc];
					Fmatrix_transpose[m] = Fmatrix_transpose[m-inc];
					m -= inc;
					if (m-noff+1 <= inc) break;
				}
				Fmatrix_transpose_index[m] = iv;
				Fmatrix_transpose[m] = v;
			}
		} while (inc > 1);
	}
	delete[] offdiag_indx;
	delete[] offdiag_indx_transpose;

	int *Fmatrix_unsymmetric_cols = new int[n_amps+1];
	int *Fmatrix_unsymmetric_indices = new int[Fmatrix_unsymmetric_nonzero_elements];
	double *Fmatrix_unsymmetric = new double[Fmatrix_unsymmetric_nonzero_elements];

	int indx=0;
	Fmatrix_unsymmetric_cols[0] = 0;
	for (i=0; i < n_amps; i++) {
		for (j=Fmatrix_transpose_index[i]; j < Fmatrix_transpose_index[i+1]; j++) {
			Fmatrix_unsymmetric[indx] = Fmatrix_transpose[j];
			Fmatrix_unsymmetric_indices[indx] = Fmatrix_transpose_index[j];
			indx++;
		}
		Fmatrix_unsymmetric_indices[indx] = i;
		Fmatrix_unsymmetric[indx] = Fmatptr[i];
		indx++;
		for (j=Fmatrix_index[i]; j < Fmatrix_index[i+1]; j++) {
			Fmatrix_unsymmetric[indx] = Fmatptr[j];
			Fmatrix_unsymmetric_indices[indx] = Fmatrix_index[j];
			indx++;
		}
		Fmatrix_unsymmetric_cols[i+1] = indx;
	}

	for (i=0; i < n_amps; i++) {
		sort(Fmatrix_unsymmetric_cols[i+1]-Fmatrix_unsymmetric_cols[i],Fmatrix_unsymmetric_indices+Fmatrix_unsymmetric_cols[i],Fmatrix_unsymmetric+Fmatrix_unsymmetric_cols[i]);
		//cout << "Row " << i << ": " << endl;
		//cout << Fmatrix_unsymmetric_cols[i] << " ";
		//for (j=Fmatrix_unsymmetric_cols[i]; j < Fmatrix_unsymmetric_cols[i+1]; j++) {
			//cout << Fmatrix_unsymmetric_indices[j] << " ";
		//}
		//cout << endl;
		//for (j=Fmatrix_unsymmetric_cols[i]; j < Fmatrix_unsymmetric_cols[i+1]; j++) {
			//cout << "j=" << j << " " << Fmatrix_unsymmetric[j] << " ";
		//}
		//cout << endl;
	}
	//cout << endl;

	if (indx != Fmatrix_unsymmetric_nonzero_elements) die("WTF! Wrong number of nonzero elements");

	int status;
   status = umfpack_di_symbolic(n_amps, n_amps, Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, &Symbolic, Control, Info);
	if (status < 0) {
		umfpack_di_report_info (Control, Info) ;
		umfpack_di_report_status (Control, status) ;
		die("Error inputting matrix");
	}
   status = umfpack_di_numeric(Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, Symbolic, &Numeric, Control, Info);
   umfpack_di_free_symbolic(&Symbolic);

   status = umfpack_di_solve(UMFPACK_A, Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, temp, Dvector.data(), Numeric, Control, Info);

	for (int i=0; i < n_amps; i++) {
		amplitude_vector[i] = temp[i];
	}

	double mantissa, exponent;
	status = umfpack_di_get_determinant (&mantissa, &exponent, Numeric, Info) ;
	if (status < 0) {
		die("Could not get determinant using UMFPACK");
	}
	umfpack_di_free_numeric(&Numeric);
	logdet = log(mantissa) + exponent*log(10);

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for inverting Fmatrix_sparse: "  << wtime.count() << endl;
	}

	delete[] temp;
	delete[] Fmatrix_transpose;
	delete[] Fmatrix_transpose_index;
	delete[] Fmatrix_unsymmetric_cols;
	delete[] Fmatrix_unsymmetric_indices;
	delete[] Fmatrix_unsymmetric;
	update_source_and_lensgrid_amplitudes(verbal);
#endif
}

void ImagePixelGrid::invert_lens_mapping_MUMPS(double& logdet, const bool verbal, const bool use_copy)
{
#ifndef USE_MUMPS
	die("QLens requires compilation with MUMPS for Cholesky factorization");
#else

	int default_nthreads=1;

#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		default_nthreads = omp_get_num_threads();
	}
	omp_set_num_threads(qlens->inversion_nthreads);
#endif

	if (qlens->show_wtime) {
		wtime0 = std::chrono::steady_clock::now();
	}
	int i,j;
	double *Fmatptr = (use_copy==true) ? Fmatrix_copy : Fmatrix_sparse;

	double *temp = new double[n_amps];
	MUMPS_INT Fmatrix_nonzero_elements = Fmatrix_index[n_amps]-1;
	if (Fmatrix_nonzero_elements==0) {
		cout << "nsource_pixels=" << n_amps << endl;
		die("Fmatrix has zero size");
	}
	MUMPS_INT *irn = new MUMPS_INT[Fmatrix_nonzero_elements];
	MUMPS_INT *jcn = new MUMPS_INT[Fmatrix_nonzero_elements];
	double *Fmatrix_elements = new double[Fmatrix_nonzero_elements];
	for (i=0; i < n_amps; i++) {
		Fmatrix_elements[i] = Fmatptr[i];
		irn[i] = i+1;
		jcn[i] = i+1;
		temp[i] = Dvector[i];
	}
	int indx=n_amps;
	for (i=0; i < n_amps; i++) {
		for (j=Fmatrix_index[i]; j < Fmatrix_index[i+1]; j++) {
			Fmatrix_elements[indx] = Fmatptr[j];
			irn[indx] = i+1;
			jcn[indx] = Fmatrix_index[j]+1;
			indx++;
		}
	}

	mumps_solver->job = JOB_INIT; // initialize
	mumps_solver->sym = 2; // specifies that matrix is symmetric and positive-definite
	//cout << "ICNTL = " << mumps_solver->icntl[13] << endl;
	dmumps_c(mumps_solver);
	mumps_solver->n = n_amps; mumps_solver->nz = Fmatrix_nonzero_elements; mumps_solver->irn=irn; mumps_solver->jcn=jcn;
	mumps_solver->a = Fmatrix_elements; mumps_solver->rhs = temp;
	if (show_mumps_info) {
		mumps_solver->icntl[0] = MUMPS_OUTPUT;
		mumps_solver->icntl[1] = MUMPS_OUTPUT;
		mumps_solver->icntl[2] = MUMPS_OUTPUT;
		mumps_solver->icntl[3] = MUMPS_OUTPUT;
	} else {
		mumps_solver->icntl[0] = MUMPS_SILENT;
		mumps_solver->icntl[1] = MUMPS_SILENT;
		mumps_solver->icntl[2] = MUMPS_SILENT;
		mumps_solver->icntl[3] = MUMPS_SILENT;
	}
	if (parallel_mumps) {
		mumps_solver->icntl[27]=2; // parallel analysis phase
		mumps_solver->icntl[28]=2; // parallel analysis phase
	}
	mumps_solver->job = 6; // specifies to factorize and solve linear equation
	dmumps_c(mumps_solver);

	if (mumps_solver->info[0] < 0) {
		if (mumps_solver->info[0]==-10) die("Singular matrix, cannot invert");
		else warn("Error occurred during matrix inversion; MUMPS error code %i (n_amps=%i)",mumps_solver->info[0],n_amps);
	}

	for (int i=0; i < n_amps; i++) {
		if ((qlens->background_pixel_noise==0) and (temp[i] < 0)) temp[i] = 0; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
		amplitude_vector[i] = temp[i];
	}

	/*
	if ((qlens->regularization_method != None) and (source_npixels > 0))
	{
		logdet = log(mumps_solver->rinfog[11]) + mumps_solver->infog[33]*log(2);
		//cout << "Fmatrix log determinant = " << logdet << endl;
		if ((qlens->mpi_id==0) and (verbal)) cout << "log determinant = " << logdet << endl;

		mumps_solver->job=JOB_END; dmumps_c(mumps_solver); //Terminate instance

		MUMPS_INT Rmatrix_nonzero_elements = Rmatrix_index[n_amps]-1;
		MUMPS_INT *irn_reg = new MUMPS_INT[Rmatrix_nonzero_elements];
		MUMPS_INT *jcn_reg = new MUMPS_INT[Rmatrix_nonzero_elements];
		double *Rmatrix_elements = new double[Rmatrix_nonzero_elements];
		for (i=0; i < n_amps; i++) {
			Rmatrix_elements[i] = Rmatrix_sparse[i];
			irn_reg[i] = i+1;
			jcn_reg[i] = i+1;
		}
		indx=n_amps;
		for (i=0; i < n_amps; i++) {
			//cout << "Row " << i << ": diag=" << Rmatrix_sparse[i] << endl;
			//for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
				//cout << Rmatrix_index[j] << " ";
			//}
			//cout << endl;
			for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
				//cout << Rmatrix_sparse[j] << " ";
				Rmatrix_elements[indx] = Rmatrix_sparse[j];
				irn_reg[indx] = i+1;
				jcn_reg[indx] = Rmatrix_index[j]+1;
				indx++;
			}
		}

		mumps_solver->job=JOB_INIT; mumps_solver->sym=2;
		dmumps_c(mumps_solver);
		mumps_solver->n = n_amps; mumps_solver->nz = Rmatrix_nonzero_elements; mumps_solver->irn=irn_reg; mumps_solver->jcn=jcn_reg;
		mumps_solver->a = Rmatrix_elements;
		mumps_solver->icntl[0]=MUMPS_SILENT;
		mumps_solver->icntl[1]=MUMPS_SILENT;
		mumps_solver->icntl[2]=MUMPS_SILENT;
		mumps_solver->icntl[3]=MUMPS_SILENT;
		mumps_solver->icntl[32]=1; // calculate determinant
		mumps_solver->icntl[30]=1; // discard factorized matrices
		if (parallel_mumps) {
			mumps_solver->icntl[27]=2; // parallel analysis phase
			mumps_solver->icntl[28]=2; // parallel analysis phase
		}
		mumps_solver->job=4;
		dmumps_c(mumps_solver);
		if (mumps_solver->rinfog[11]==0) Rmatrix_log_determinant = -1e20;
		else Rmatrix_log_determinant = log(mumps_solver->rinfog[11]) + mumps_solver->infog[33]*log(2);
		//cout << "Rmatrix log determinant = " << Rmatrix_log_determinant << " " << mumps_solver->rinfog[11] << " " << mumps_solver->infog[33] << endl;
		if ((qlens->mpi_id==0) and (verbal)) cout << "Rmatrix log determinant = " << Rmatrix_log_determinant << " " << mumps_solver->rinfog[11] << " " << mumps_solver->infog[33] << endl;

		delete[] irn_reg;
		delete[] jcn_reg;
		delete[] Rmatrix_elements;
	}
	*/
	mumps_solver->job=JOB_END;
	dmumps_c(mumps_solver); //Terminate instance

	if (qlens->show_wtime) {
		wtime = std::chrono::steady_clock::now() - wtime0;
		if (qlens->mpi_id==0) cout << "Wall time for inverting Fmatrix_sparse: "  << wtime.count() << endl;
	}

#ifdef USE_OPENMP
	omp_set_num_threads(default_nthreads);
#endif

	delete[] temp;
	delete[] irn;
	delete[] jcn;
	delete[] Fmatrix_elements;
	update_source_and_lensgrid_amplitudes(imggrid_i,verbal);
#endif

}

void ImagePixelGrid::update_source_and_lensgrid_amplitudes(const bool verbal)
{
	SB_Profile** sb_list = qlens->sb_list;
	int i,j,index=0;

	int n_imggrids = n_pixsrc_to_include_in_Lmatrix;
	if (n_imggrids==0) die("No image grids are included for constructing Lmatrix");
	ImagePixelGrid **imggrids;
	imggrids = new ImagePixelGrid*[n_imggrids]; // last element will be used to identify end of list
	for (i=0; i < n_imggrids; i++) imggrids[i] = qlens->image_pixel_grids[imggrid_indx_to_include_in_Lmatrix[i]];

	if ((qlens->source_fit_mode==Delaunay_Source) and (delaunay_srcgrid != NULL)) {
		for (i=0; i < n_imggrids; i++) imggrids[i]->delaunay_srcgrid->update_surface_brightness(index);
	}
	else if (qlens->source_fit_mode==Cartesian_Source) {
		for (i=0; i < n_imggrids; i++) imggrids[i]->cartesian_srcgrid->update_surface_brightness(index);
	}
	else if (qlens->source_fit_mode==Shapelet_Source) {
		double* srcpix = amplitude_vector.data();
		for (i=0; i < qlens->n_sb; i++) {
			if ((qlens->sb_list[i]->sbtype==SHAPELET) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
				qlens->sb_list[i]->update_amplitudes(srcpix);
			}
		}
		index = source_npixels;
	}
	if (index != source_npixels) die("WTF? did not go through all the source pixels (index=%i)",index);
	if ((include_potential_perturbations) and (lensgrid_npixels > 0)) lensgrid->update_potential(index);
	if (n_mge_amps > 0) {
		double* srcpix = amplitude_vector.data() + source_npixels + lensgrid_npixels;
		//for (i=0; i < n_mge_amps; i++) cout << srcpix[i] << endl;
		for (i=0; i < qlens->n_sb; i++) {
			if ((qlens->sb_list[i]->sbtype==MULTI_GAUSSIAN_EXPANSION) and (qlens->sbprofile_imggrid_idx[i]==imggrid_index)) {
				qlens->sb_list[i]->update_amplitudes(srcpix);
			}
		}
		index += n_mge_amps;
	}
	if (qlens->include_imgfluxes_in_inversion) {
		for (j=0; j < qlens->n_ptsrc; j++) {
			for (i=0; i < qlens->ptsrc_list[j]->images.size(); i++) {
				qlens->ptsrc_list[j]->images[i].flux = amplitude_vector[index++];
				if ((qlens->mpi_id==0) and (verbal)) cout << "srcpt " << j << " (img " << i << "): flux=" << qlens->ptsrc_list[j]->images[i].flux << endl;
			}
		}
	} else if (qlens->include_srcflux_in_inversion) {
		for (j=0; j < qlens->n_ptsrc; j++) {
			qlens->ptsrc_list[j]->get_srcflux() = amplitude_vector[index++];
			if ((qlens->mpi_id==0) and (verbal)) cout << "srcpt " << j << ": srcflux=" << qlens->ptsrc_list[j]->get_srcflux() << endl;
		}
	}
	delete[] imggrids;
}

void ImagePixelGrid::Rmatrix_determinant_EIGEN(const bool potential_perturbations)
{
	int npixels;
	if (!potential_perturbations) {
		npixels = source_npixels_inv;
		Rmatrix_ptr = &Rmatrix_sparse;
		Rmatrix_index_ptr = &Rmatrix_index;
		Rmatrix_dense_ptr = &Rmatrix_dense;
	} else {
		npixels = lensgrid_npixels;
		Rmatrix_ptr = &Rmatrix_pot;
		Rmatrix_index_ptr = &Rmatrix_pot_index;
	}

	std::vector<Eigen::Triplet<double>> triplets;

	int i,k;
	for (i=0; i < npixels; i++) {
	  triplets.emplace_back(i, i, (*Rmatrix_ptr)[i]);
	}

	for (i=0; i < npixels; i++) {
		 for (k=Rmatrix_index[i]; k < (*Rmatrix_index_ptr)[i+1]; k++) {
			  triplets.emplace_back(i, (*Rmatrix_index_ptr)[k], (*Rmatrix_ptr)[k]);
			  triplets.emplace_back((*Rmatrix_index_ptr)[k], i, (*Rmatrix_ptr)[k]);
		 }
	}
	Eigen::SparseMatrix<double> R(npixels,npixels);
	R.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	solver.compute(R);
	Rmatrix_log_determinant = solver.logAbsDeterminant();
}

#define ISWAP(a,b) temp=(a);(a)=(b);(b)=temp;
void ImagePixelGrid::indexx(int* arr, int* indx, int nn)
{
	const int M=7, NSTACK=50;
	int i,indxt,ir,j,k,jstack=-1,l=0;
	double a,temp;
	int *istack = new int[NSTACK];
	ir = nn - 1;
	for (j=0; j < nn; j++) indx[j] = j;
	for (;;) {
		if (ir-l < M) {
			for (j=l+1; j <= ir; j++) {
				indxt=indx[j];
				a=arr[indxt];
				for (i=j-1; i >=l; i--) {
					if (arr[indx[i]] <= a) break;
					indx[i+1]=indx[i];
				}
				indx[i+1]=indxt;
			}
			if (jstack < 0) break;
			ir=istack[jstack--];
			l=istack[jstack--];
		} else {
			k=(l+ir) >> 1;
			ISWAP(indx[k],indx[l+1]);
			if (arr[indx[l]] > arr[indx[ir]]) {
				ISWAP(indx[l],indx[ir]);
			}
			if (arr[indx[l+1]] > arr[indx[ir]]) {
				ISWAP(indx[l+1],indx[ir]);
			}
			if (arr[indx[l]] > arr[indx[l+1]]) {
				ISWAP(indx[l],indx[l+1]);
			}
			i=l+1;
			j=ir;
			indxt=indx[l+1];
			a=arr[indxt];
			for (;;) {
				do i++; while (arr[indx[i]] < a);
				do j--; while (arr[indx[j]] > a);
				if (j < i) break;
				ISWAP(indx[i],indx[j]);
			}
			indx[l+1]=indx[j];
			indx[j]=indxt;
			jstack += 2;
			if (jstack >= NSTACK) die("NSTACK too small in indexx");
			if (ir-i+1 >= j-l) {
				istack[jstack]=ir;
				istack[jstack-1]=i;
				ir=j-1;
			} else {
				istack[jstack]=j-1;
				istack[jstack-1]=l;
				l=i;
			}
		}
	}
	delete[] istack;
}
#undef ISWAP

void ImagePixelGrid::clear_sparse_lensing_matrices()
{
	if (Fmatrix_sparse != NULL) delete[] Fmatrix_sparse;
	if (Fmatrix_index != NULL) delete[] Fmatrix_index;
	if (Rmatrix_sparse != NULL) delete[] Rmatrix_sparse;
	if (Rmatrix_index != NULL) delete[] Rmatrix_index;
	if (Rmatrix_pot != NULL) delete[] Rmatrix_pot;
	if (Rmatrix_pot_index != NULL) delete[] Rmatrix_pot_index;
	Fmatrix_sparse = NULL;
	Fmatrix_index = NULL;
	Rmatrix_sparse = NULL;
	Rmatrix_index = NULL;
	Rmatrix_pot = NULL;
	Rmatrix_pot_index = NULL;
}

void ImagePixelGrid::calculate_image_pixel_surface_brightness()
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	int img_index_j;
	int i,j,k;

	for (int img_index=0; img_index < image_npixels; img_index++) {
		p.image_surface_brightness[img_index] = 0;
		for (img_index_j=image_pixel_location_Lmatrix[img_index]; img_index_j < image_pixel_location_Lmatrix[img_index+1]; img_index_j++) {
			p.image_surface_brightness[img_index] += Lmatrix_sparse[img_index_j]*amplitude_vector[Lmatrix_index[img_index_j]];
		}
		//if (p.image_surface_brightness[i] < 0) p.image_surface_brightness[i] = 0;
	}

	/*
	if (calculate_foreground) {
		bool at_least_one_foreground_src = false;

		for (k=0; k < qlens->n_sb; k++) {
			if ((!qlens->sb_list[k]->is_lensed) and (imggrid_i==0)) {
				at_least_one_foreground_src = true;
				break;
			}
		}
		if ((at_least_one_foreground_src) and (!ignore_foreground_in_chisq)) {
			calculate_foreground_pixel_surface_brightness();
			//add_foreground_to_image_pixel_vector();
			store_foreground_pixel_surface_brightness(); // this stores it in foreground_surface_brightness[i][j]
		} else {
			for (int img_index=0; img_index < image_npixels_fgmask; img_index++) {
				sbprofile_surface_brightness[img_index] = 0;
			}
		}
	}
	*/
}

void ImagePixelGrid::calculate_image_pixel_surface_brightness_dense()
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	int i,j;
	if (lensgrid_npixels > 0) {
		for (j=0; j < source_and_lens_n_amps; j++) {
			//if (j < source_npixels) cout << "SURF " << j << ": " << amplitude_vector[j] << endl;
			if ((j >= source_npixels) and (j < source_npixels + lensgrid_npixels)) {
				if (abs(amplitude_vector[j]) > 0.08) cout << "LARGE POT " << (j-source_npixels) << ": " << amplitude_vector[j] << endl;
			}
		}
	}

	p.image_surface_brightness = Lmatrix_trans_dense.transpose()*amplitude_vector;
	/*
	for (i=0; i < image_npixels; i++) {
		p.image_surface_brightness[i] = Lmatrix_trans_dense.col(i).dot(amplitude_vector);
	}
	*/

	/*
	if (calculate_foreground) {
		bool at_least_one_foreground_src = false;
		for (k=0; k < qlens->n_sb; k++) {
			if (!qlens->sb_list[k]->is_lensed) {
				at_least_one_foreground_src = true;
			}
		}
		if ((at_least_one_foreground_src) and (!ignore_foreground_in_chisq)) {
			calculate_foreground_pixel_surface_brightness();
			store_foreground_pixel_surface_brightness(); // this stores it in image_pixel_grid->sbprofile_surface_brightness[i][j]
			//add_foreground_to_image_pixel_vector();
		} else {
			for (int img_index=0; img_index < image_npixels_fgmask; img_index++) sbprofile_surface_brightness[img_index] = 0;
		}
	}
	if (!calculate_foreground) {
		for (int img_index=0; img_index < image_npixels_fgmask; img_index++) sbprofile_surface_brightness[img_index] = 0;
	}
	*/
}

void ImagePixelGrid::calculate_foreground_pixel_surface_brightness(const bool allow_lensed_noninverted_sources)
{
	ImgGrid_Params<PlainTypes>& imggrid_params = assign_imggrid_param_object<PlainTypes>();
	SB_Profile** sb_list = qlens->sb_list;
	bool subgridded;
	int img_index;
	int i,j,k;
	bool at_least_one_foreground_noninverted_src = false;
	bool at_least_one_lensed_src = false;
	for (k=0; k < qlens->n_sb; k++) {
		if (!sb_list[k]->is_lensed) {
			if ((imggrid_index == 0) and (sb_list[k]->sbtype != MULTI_GAUSSIAN_EXPANSION)) at_least_one_foreground_noninverted_src = true;
		} else {
			if (qlens->sbprofile_imggrid_idx[k]==imggrid_index) at_least_one_lensed_src = true;
		}
	}

	// here, we are adding together SB of foreground sources, but also lensed non-shapelet and non-MGE sources
	// If none of those conditions are true, then we skip everything.
	if ((!at_least_one_foreground_noninverted_src) and ((!allow_lensed_noninverted_sources) or (!at_least_one_lensed_src))) {
		for (img_index=0; img_index < image_npixels_fgmask; img_index++) imggrid_params.sbprofile_surface_brightness[img_index] = 0;
		return;
	} else {
		if (qlens->show_wtime) {
			wtime0 = std::chrono::steady_clock::now();
		}

		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif

			int ii, jj, nsplit;
			double u0, w0, sb;
			//double U0, W0, U1, W1;
			lensvector<double> center_pt, center_srcpt;
			lensvector<double> corner1, corner2, corner3, corner4;
			double subpixel_xlength, subpixel_ylength;
			double noise;
			int subcell_index;
			#pragma omp for private(img_index,i,j,ii,jj,nsplit,u0,w0,sb,subpixel_xlength,subpixel_ylength,center_pt,center_srcpt,corner1,corner2,corner3,corner4,subcell_index,noise) schedule(dynamic)
			for (img_index=0; img_index < image_npixels_fgmask; img_index++) {
				imggrid_params.sbprofile_surface_brightness[img_index] = 0;

				i = fgmask_pixels_i[img_index];
				j = fgmask_pixels_j[img_index];

				sb = 0;

				if (qlens->split_imgpixels) nsplit = imgpixel_nsplit;
				else nsplit = 1;
				// Now check to see if center of foreground galaxy is in or next to the pixel; if so, make sure it has at least four splittings so its
				// surface brightness is well-reproduced
				if ((at_least_one_foreground_noninverted_src) and (nsplit < 4) and (i > 0) and (i < x_N-1) and (j > 0) and (j < y_N)) {
					for (k=0; k < qlens->n_sb; k++) {
						if ((!sb_list[k]->is_lensed) and (sb_list[k]->sbtype != MULTI_GAUSSIAN_EXPANSION)) {
							double xc, yc;
							sb_list[k]->get_center_coords(xc,yc);
							if ((xc > corner_pts[i-1][j][0]) and (xc < corner_pts[i+2][j][0]) and (yc > corner_pts[i][j-1][1]) and (yc < corner_pts[i][j+2][1])) nsplit = 4;
						} 
					}
				}

				subpixel_xlength = pixel_xlength/nsplit;
				subpixel_ylength = pixel_ylength/nsplit;
				subcell_index = 0;
				for (ii=0; ii < nsplit; ii++) {
					u0 = ((double) (1+2*ii))/(2*nsplit);
					//center_pt[0] = u0*corner_pts[i][j][0] + (1-u0)*corner_pts[i+1][j][0];
					center_pt[0] = (1-u0)*corner_pts[i][j][0] + u0*corner_pts[i+1][j][0];
					for (jj=0; jj < nsplit; jj++) {
						w0 = ((double) (1+2*jj))/(2*nsplit);
						//center_pt[1] = w0*corner_pts[i][j][1] + (1-w0)*corner_pts[i][j+1][1];
						center_pt[1] = (1-w0)*corner_pts[i][j][1] + w0*corner_pts[i][j+1][1];
						//center_pt = subpixel_center_pts[i][j][subcell_index]; 
						//cout << "CHECK: " << subpixel_center_pts[i][j][subcell_index][0] << " " << center_pt[0] << " and " << subpixel_center_pts[i][j][subcell_index][1] << " " << center_pt[1] << endl;
						for (int k=0; k < qlens->n_sb; k++) {
							//if ((at_least_one_foreground_noninverted_src) and (!sb_list[k]->is_lensed) and ((qlens->source_fit_mode != Shapelet_Source) or (sb_list[k]->sbtype != SHAPELET)) and (sb_list[k]->sbtype != MULTI_GAUSSIAN_EXPANSION)) {
							if ((at_least_one_foreground_noninverted_src) and (!sb_list[k]->is_lensed) and (sb_list[k]->sbtype != MULTI_GAUSSIAN_EXPANSION)) {
								if (!sb_list[k]->zoom_subgridding) {
									sb += sb_list[k]->surface_brightness(center_pt[0],center_pt[1]);
								}
								else {
									corner1[0] = center_pt[0] - subpixel_xlength/2;
									corner1[1] = center_pt[1] - subpixel_ylength/2;
									corner2[0] = center_pt[0] + subpixel_xlength/2;
									corner2[1] = center_pt[1] - subpixel_ylength/2;
									corner3[0] = center_pt[0] - subpixel_xlength/2;
									corner3[1] = center_pt[1] + subpixel_ylength/2;
									corner4[0] = center_pt[0] + subpixel_xlength/2;
									corner4[1] = center_pt[1] + subpixel_ylength/2;
									noise = (qlens->use_noise_map) ? noise_map[i][j] : qlens->background_pixel_noise;
									sb += sb_list[k]->surface_brightness_zoom(center_pt,corner1,corner2,corner3,corner4,noise);
								}
							}
							else if ((allow_lensed_noninverted_sources) and (sb_list[k]->is_lensed) and (qlens->sbprofile_imggrid_idx[k]==imggrid_index) and (sb_list[k]->sbtype != SHAPELET) and (sb_list[k]->sbtype != MULTI_GAUSSIAN_EXPANSION) and ((image_data == NULL) or (image_data->foreground_mask[i][j]))) { // if source mode is shapelet and sbprofile is shapelet, will include in inversion
								//center_srcpt = subpixel_center_sourcepts[i][j][subcell_index];
								//center_srcpt = subpixel_center_sourcepts[i][j][subcell_index];
								//qlens->find_sourcept<double>(center_pt,center_srcpt,thread,reference_zfactors,default_zsrc_beta_factors);
								//sb += sb_list[k]->surface_brightness(center_srcpt[0],center_srcpt[1]);
								if (qlens->split_imgpixels) sb += sb_list[k]->surface_brightness(subpixel_center_sourcepts[i][j][subcell_index][0],subpixel_center_sourcepts[i][j][subcell_index][1]);
								else sb += sb_list[k]->surface_brightness(center_sourcepts[i][j][0],center_sourcepts[i][j][1]);
								//if (!sb_list[k]->zoom_subgridding) sb += sb_list[k]->surface_brightness(center_srcpt[0],center_srcpt[1]);
								//else {
									//corner1[0] = center_srcpt[0] - subpixel_xlength/2;
									//corner1[1] = center_srcpt[1] - subpixel_ylength/2;
									//corner2[0] = center_srcpt[0] + subpixel_xlength/2;
									//corner2[1] = center_srcpt[1] - subpixel_ylength/2;
									//corner3[0] = center_srcpt[0] - subpixel_xlength/2;
									//corner3[1] = center_srcpt[1] + subpixel_ylength/2;
									//corner4[0] = center_srcpt[0] + subpixel_xlength/2;
									//corner4[1] = center_srcpt[1] + subpixel_ylength/2;
									//sb += sb_list[k]->surface_brightness_zoom(center_srcpt,corner1,corner2,corner3,corner4,noise_map[i][j]);
								//}
							}
						}
						subcell_index++;
					}
				}
				imggrid_params.sbprofile_surface_brightness[img_index] += sb / (nsplit*nsplit);
			}
		}
		if (qlens->show_wtime) {
			wtime = std::chrono::steady_clock::now() - wtime0;
			if (qlens->mpi_id==0) cout << "Wall time for calculating foreground SB: "  << wtime.count() << endl;
		}
	}
	//for (int img_index=0; img_index < image_npixels_fgmask; img_index++) {
		//cout << sbprofile_surface_brightness[img_index] << endl;
	//}
	//die();
	PSF_convolution_pixel_vector(true,false,true);
}

void ImagePixelGrid::store_image_pixel_surface_brightness(const bool use_emask)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	int i,j;
	int npix;
	double *surface_brightness_vector;
	if (!use_emask) {
		npix = image_npixels;
		surface_brightness_vector = p.image_surface_brightness.data();
	} else {
		npix = image_npixels_emask;
		surface_brightness_vector = p.image_surface_brightness_emask.data();
	}
	for (i=0; i < x_N; i++)
		for (j=0; j < y_N; j++)
			surface_brightness[i][j] = 0;

	for (int img_index=0; img_index < npix; img_index++) {
		i = emask_pixels_i[img_index];
		j = emask_pixels_j[img_index];
		surface_brightness[i][j] = surface_brightness_vector[img_index];
	}
}

void ImagePixelGrid::store_foreground_pixel_surface_brightness() // note, foreground_surface_brightness could also include source objects that aren't shapelets (if in shapelet mode)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	int i,j;
	for (int img_index=0; img_index < image_npixels_fgmask; img_index++) {
		i = fgmask_pixels_i[img_index];
		j = fgmask_pixels_j[img_index];
		//if (sbprofile_surface_brightness[img_index] != 0.0) cout << "NONZERO FG SB = " << sbprofile_surface_brightness[img_index] << endl;
		foreground_surface_brightness[i][j] = p.sbprofile_surface_brightness[img_index];
	}
}

void ImagePixelGrid::vectorize_image_pixel_surface_brightness(const bool use_emask)
{
	ImgGrid_Params<PlainTypes>& p = assign_imggrid_param_object<PlainTypes>();
	int i,j,img_index;
	if (use_emask) {
		for (img_index=0; img_index < image_npixels_emask; img_index++) {
			i = emask_pixels_i[img_index];
			j = emask_pixels_j[img_index];
			p.image_surface_brightness_emask[img_index] = surface_brightness[i][j];
		}
	} else {
		for (img_index=0; img_index < image_npixels; img_index++) {
			i = emask_pixels_i[img_index];
			j = emask_pixels_j[img_index];
			p.image_surface_brightness[img_index] = surface_brightness[i][j];
		}
		if (qlens->psf_supersampling) {
			int subcell_index;
			for (img_index=0; img_index < image_n_subpixels; img_index++) {
				i = extended_mask_subpixel_i[img_index];
				j = extended_mask_subpixel_j[img_index];
				subcell_index = extended_mask_subpixel_index[img_index];
				p.image_surface_brightness_supersampled[img_index] = subpixel_surface_brightness[i][j][subcell_index];
			}
		}
	}
}

ImagePixelGrid::~ImagePixelGrid()
{
	clear_sparse_lensing_matrices();
	clear_pixel_matrices();
	for (int i=0; i <= x_N; i++) {
		delete[] corner_pts[i];
		delete[] corner_sourcepts[i];
	}
	delete[] corner_pts;
	delete[] corner_sourcepts;
	for (int i=0; i < x_N; i++) {
		delete[] center_pts[i];
		delete[] center_sourcepts[i];
		delete[] maps_to_source_pixel[i];
		delete[] pixel_index[i];
		delete[] pixel_index_fgmask[i];
		delete[] mapped_cartesian_srcpixels[i];
		delete[] mapped_delaunay_srcpixels[i];
		delete[] mapped_potpixels[i];
		delete[] foreground_surface_brightness[i];
		delete[] surface_brightness[i];
		delete[] noise_map[i];
		delete[] source_plane_triangle1_area[i];
		delete[] source_plane_triangle2_area[i];
		delete[] pixel_mag[i];
		delete[] twist_status[i];
		delete[] twist_pts[i];
		//delete[] nsplits[i];
		for (int j=0; j < y_N; j++) {
			delete[] subpixel_maps_to_srcpixel[i][j];
			delete[] subpixel_center_pts[i][j];
			delete[] subpixel_center_sourcepts[i][j];
			delete[] subpixel_surface_brightness[i][j];
			delete[] subpixel_weights[i][j];
			delete[] subpixel_source_gradient[i][j];
			delete[] n_mapped_srcpixels[i][j];
			delete[] n_mapped_potpixels[i][j];
		}
		delete[] subpixel_maps_to_srcpixel[i];
		delete[] subpixel_center_pts[i];
		delete[] subpixel_center_sourcepts[i];
		delete[] subpixel_surface_brightness[i];
		delete[] subpixel_weights[i];
		delete[] subpixel_source_gradient[i];
		delete[] n_mapped_srcpixels[i];
		delete[] n_mapped_potpixels[i];
	}
	int max_subpixel_nx = x_N*max_nsplit;
	for (int i=0; i < max_subpixel_nx; i++) {
		delete[] subpixel_index_ss[i];
	}
	delete[] center_pts;
	delete[] center_sourcepts;
	delete[] maps_to_source_pixel;
	delete[] pixel_index;
	delete[] pixel_index_fgmask;
	delete[] subpixel_index_ss;
	delete[] mapped_cartesian_srcpixels;
	delete[] mapped_delaunay_srcpixels;
	delete[] mapped_potpixels;
	delete[] foreground_surface_brightness;
	delete[] surface_brightness;
	delete[] noise_map;
	delete[] source_plane_triangle1_area;
	delete[] source_plane_triangle2_area;
	delete[] pixel_mag;
	delete[] subpixel_maps_to_srcpixel;
	delete[] subpixel_center_pts;
	delete[] subpixel_center_sourcepts;
	delete[] subpixel_surface_brightness;
	delete[] subpixel_weights;
	delete[] subpixel_source_gradient;
	delete[] n_mapped_srcpixels;
	delete[] n_mapped_potpixels;
	//delete[] nsplits;
	delete[] twist_status;
	delete[] twist_pts;
	if (pixel_in_mask != NULL) {
		for (int i=0; i < x_N; i++) delete[] pixel_in_mask[i];
		delete[] pixel_in_mask;
	}
	if (saved_sbweights != NULL) delete[] saved_sbweights;
	delete_ray_tracing_arrays();
	delete_subpixel_ray_tracing_arrays();
}


