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
#include <functional>
#include <stdio.h>

#ifdef USE_MKL
#include "mkl.h"
#include "mkl_spblas.h"
#endif

#ifdef USE_UMFPACK
#include "umfpack.h"
#endif

#ifdef USE_FITS
#include "fitsio.h"
#endif

#ifdef USE_FFTW
#ifdef USE_MKL
#include "fftw/fftw3.h"
#else
#include "fftw3.h"
#endif
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

int SourcePixel::nthreads = 0;
int SourcePixel::max_levels = 2;
int *SourcePixel::imin, *SourcePixel::imax, *SourcePixel::jmin, *SourcePixel::jmax;
TriRectangleOverlap *SourcePixel::trirec = NULL;
InterpolationCells *SourcePixel::nearest_interpolation_cells = NULL;
lensvector **SourcePixel::interpolation_pts[3];
//int *SourcePixel::n_interpolation_pts = NULL;

int DelaunayGrid::nthreads = 0;
const int DelaunayGrid::nmax_pts_interp; // maximum number of allowed interpolation points; this number is initialized in pixelgrid.h

lensvector **DelaunayGrid::interpolation_pts[nmax_pts_interp];
double *DelaunayGrid::interpolation_wgts[nmax_pts_interp];
int *DelaunayGrid::interpolation_indx[nmax_pts_interp];
int *DelaunayGrid::triangles_in_envelope[nmax_pts_interp];
lensvector **DelaunayGrid::polygon_vertices[nmax_pts_interp+2];
lensvector *DelaunayGrid::new_circumcenter[nmax_pts_interp];

bool DelaunayGrid::zero_outside_border = false;
//ImagePixelGrid* DelaunayGrid::image_pixel_grid = NULL;

// variables for root finding to get point images (for combining with extended pixel images)
int ImagePixelGrid::nthreads = 0;
bool *ImagePixelGrid::newton_check = NULL;
lensvector *ImagePixelGrid::fvec = NULL;
double ImagePixelGrid::image_pos_accuracy = 1e-6; // default

int *SourcePixel::maxlevs = NULL;
lensvector ***SourcePixel::xvals_threads = NULL;
lensvector ***SourcePixel::corners_threads = NULL;
lensvector **SourcePixel::twistpts_threads = NULL;
int **SourcePixel::twist_status_threads = NULL;

//bool QLens::fft_convolution_is_setup;
//double *QLens::psf_zvec;
//int QLens::fft_imin, QLens::fft_jmin, QLens::fft_ni, QLens::fft_nj;
//#ifdef USE_FFTW
//fftw_plan QLens::fftplan, QLens::fftplan_inverse;
//fftw_plan *QLens::fftplans_Lmatrix, *QLens::fftplans_Lmatrix_inverse;
//complex<double> *QLens::psf_transform, *QLens::img_transform;
//complex<double> **QLens::Lmatrix_transform;
//double *QLens::img_rvec;
//double **QLens::Lmatrix_imgs_rvec;
//#endif

/***************************************** Multithreaded variables in class ImagePixelGrid ****************************************/

void ImagePixelGrid::allocate_multithreaded_variables(const int& threads, const bool reallocate)
{
	if (newton_check != NULL) {
		if (!reallocate) return;
		else deallocate_multithreaded_variables();
	}
	nthreads = threads;
	newton_check = new bool[threads];
	fvec = new lensvector[threads];
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

/***************************************** Functions in class SourcePixelGrid ****************************************/

void SourcePixel::allocate_multithreaded_variables(const int& threads, const bool reallocate)
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
	for (i=0; i < 3; i++) interpolation_pts[i] = new lensvector*[nthreads];
	//n_interpolation_pts = new int[threads];
	maxlevs = new int[threads];
	xvals_threads = new lensvector**[threads];
	for (j=0; j < threads; j++) {
		xvals_threads[j] = new lensvector*[3];
		for (i=0; i <= 2; i++) xvals_threads[j][i] = new lensvector[3];
	}
	corners_threads = new lensvector**[nthreads];
	for (int i=0; i < nthreads; i++) corners_threads[i] = new lensvector*[4];
	twistpts_threads = new lensvector*[nthreads];
	twist_status_threads = new int*[nthreads];
}

void SourcePixel::deallocate_multithreaded_variables()
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

SourcePixelGrid::SourcePixelGrid(QLens* lens_in, double x_min, double x_max, double y_min, double y_max, const int usplit0, const int wsplit0) : SourcePixel()	// use for top-level cell only; subcells use constructor below
{
	parent_grid = this;
	lens = lens_in;
	int threads = 1;
#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		threads = omp_get_num_threads();
	}
#endif

	allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)
	u_split_initial = usplit0;
	w_split_initial = wsplit0;
	if ((u_split_initial < 2) or (w_split_initial < 2)) die("source grid dimensions cannot be smaller than 2 along either direction");
	min_cell_area = 1e-6;

// this constructor is used for a Cartesian grid
	center_pt = 0;
	// For the Cartesian grid, u = x, w = y
	u_N = u_split_initial;
	w_N = w_split_initial;
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

	xcenter = 0.5*(x_min+x_max);
	ycenter = 0.5*(y_min+y_max);
	srcgrid_xmin = x_min; srcgrid_xmax = x_max;
	srcgrid_ymin = y_min; srcgrid_ymax = y_max;

	double x, y, xstep, ystep;
	xstep = (x_max-x_min)/u_N;
	ystep = (y_max-y_min)/w_N;

	lensvector **firstlevel_xvals = new lensvector*[u_N+1];
	int i,j;
	for (i=0, x=x_min; i <= u_N; i++, x += xstep) {
		firstlevel_xvals[i] = new lensvector[w_N+1];
		for (j=0, y=y_min; j <= w_N; j++, y += ystep) {
			firstlevel_xvals[i][j][0] = x;
			firstlevel_xvals[i][j][1] = y;
		}
	}

	cell = new SourcePixel**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new SourcePixel*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new SourcePixel(lens,firstlevel_xvals,i,j,1,this);
		}
	}
	levels++;
	assign_firstlevel_neighbors();
	number_of_pixels = u_N*w_N;
	for (int i=0; i < u_N+1; i++)
		delete[] firstlevel_xvals[i];
	delete[] firstlevel_xvals;
}

SourcePixelGrid::SourcePixelGrid(QLens* lens_in, string pixel_data_fileroot, const double minarea_in) 	// use for top-level cell only; subcells use constructor below
{
	parent_grid = this;
	lens = lens_in;
	int threads = 1;
#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		threads = omp_get_num_threads();
	}
#endif
	allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)

	min_cell_area = minarea_in;
	string info_filename = pixel_data_fileroot + ".info";
	ifstream infofile(info_filename.c_str());
	double cells_per_pixel;
	infofile >> u_split_initial >> w_split_initial >> cells_per_pixel;
	infofile >> srcgrid_xmin >> srcgrid_xmax >> srcgrid_ymin >> srcgrid_ymax;
	min_cell_area = 1e-6;

	// this constructor is used for a Cartesian grid
	center_pt = 0;
	// For the Cartesian grid, u = x, w = y
	u_N = u_split_initial;
	w_N = w_split_initial;
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

	lensvector **firstlevel_xvals = new lensvector*[u_N+1];
	int i,j;
	for (i=0, x=srcgrid_xmin; i <= u_N; i++, x += xstep) {
		firstlevel_xvals[i] = new lensvector[w_N+1];
		for (j=0, y=srcgrid_ymin; j <= w_N; j++, y += ystep) {
			firstlevel_xvals[i][j][0] = x;
			firstlevel_xvals[i][j][1] = y;
		}
	}

	cell = new SourcePixel**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new SourcePixel*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new SourcePixel(lens,firstlevel_xvals,i,j,1,this);
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

// ***NOTE: the following constructor should NOT be used because there are static variables (e.g. levels), so more than one source grid
// is a bad idea. To make this work, you need to make those variables non-static and contained in the zeroth-level grid (and give subcells
// a pointer to the zeroth-level grid).
/*
SourcePixelGrid::SourcePixelGrid(QLens* lens_in, SourcePixelGrid* input_pixel_grid) : lens(lens_in)	// use for top-level cell only; subcells use constructor below
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
	u_split_initial = input_pixel_grid->u_split_initial;
	w_split_initial = input_pixel_grid->w_split_initial;
	srcgrid_xmin = input_pixel_grid->srcgrid_xmin;
	srcgrid_xmax = input_pixel_grid->srcgrid_xmax;
	srcgrid_ymin = input_pixel_grid->srcgrid_ymin;
	srcgrid_ymax = input_pixel_grid->srcgrid_ymax;

	// this constructor is used for a Cartesian grid
	center_pt = 0;
	// For the Cartesian grid, u = x, w = y
	u_N = u_split_initial;
	w_N = w_split_initial;
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

	lensvector **firstlevel_xvals = new lensvector*[u_N+1];
	int i,j;
	for (i=0, x=srcgrid_xmin; i <= u_N; i++, x += xstep) {
		firstlevel_xvals[i] = new lensvector[w_N+1];
		for (j=0, y=srcgrid_ymin; j <= w_N; j++, y += ystep) {
			firstlevel_xvals[i][j][0] = x;
			firstlevel_xvals[i][j][1] = y;
		}
	}

	cell = new SourcePixelGrid**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new SourcePixelGrid*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new SourcePixelGrid(lens,firstlevel_xvals,i,j,1,this);
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

void SourcePixel::read_surface_brightness_data(ifstream &sb_infile)
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
void SourcePixelGrid::copy_source_pixel_grid(SourcePixelGrid* input_pixel_grid)
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

SourcePixel::SourcePixel(QLens* lens_in, lensvector** xij, const int& i, const int& j, const int& level_in, SourcePixelGrid* parent_ptr)
{
	parent_grid = parent_ptr;
	u_N = 1;
	w_N = 1;
	level = level_in;
	cell = NULL;
	ii=i; jj=j; // store the index carried by this cell in the grid of the parent cell
	maps_to_image_pixel = false;
	maps_to_image_window = false;
	active_pixel = false;
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

void SourcePixel::assign_surface_brightness_from_analytic_source(const int zsrc_i)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->assign_surface_brightness_from_analytic_source(zsrc_i);
			else {
				cell[i][j]->surface_brightness = 0;
				for (int k=0; k < lens->n_sb; k++) {
					if ((lens->sb_list[k]->is_lensed) and ((zsrc_i < 0) or (lens->sbprofile_redshift_idx[k]==zsrc_i))) cell[i][j]->surface_brightness += lens->sb_list[k]->surface_brightness(cell[i][j]->center_pt[0],cell[i][j]->center_pt[1]);
				}
			}
		}
	}
}

void SourcePixel::assign_surface_brightness_from_delaunay_grid(DelaunayGrid* delaunay_grid, const bool add_sb)
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

void SourcePixel::update_surface_brightness(int& index)
{
	for (int j=0; j < w_N; j++) {
		for (int i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->update_surface_brightness(index);
			else {
				if (cell[i][j]->active_pixel) {
					cell[i][j]->surface_brightness = lens->source_pixel_vector[index++];
				} else {
					cell[i][j]->surface_brightness = 0;
				}
			}
		}
	}
}

void SourcePixel::fill_surface_brightness_vector()
{
	int column_j = 0;
	fill_surface_brightness_vector_recursive(column_j);
}

void SourcePixel::fill_surface_brightness_vector_recursive(int& column_j)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->fill_surface_brightness_vector_recursive(column_j);
			else {
				if (cell[i][j]->active_pixel) {
					lens->source_pixel_vector[column_j++] = cell[i][j]->surface_brightness;
				}
			}
		}
	}
}

void SourcePixel::fill_n_image_vector()
{
	int column_j = 0;
	fill_n_image_vector_recursive(column_j);
}

void SourcePixel::fill_n_image_vector_recursive(int& column_j)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->fill_n_image_vector_recursive(column_j);
			else {
				if (cell[i][j]->active_pixel) {
					lens->source_pixel_n_images[column_j++] = cell[i][j]->n_images;
				}
			}
		}
	}
}

double SourcePixelGrid::find_avg_n_images(const double sb_threshold_frac)
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

void SourcePixelGrid::store_surface_brightness_grid_data(string root)
{
	string img_filename = root + ".sb";
	string info_filename = root + ".info";

	pixel_surface_brightness_file.open(img_filename.c_str());
	write_surface_brightness_to_file(pixel_surface_brightness_file);
	pixel_surface_brightness_file.close();

	ofstream pixel_info; lens->open_output_file(pixel_info,info_filename);
	pixel_info << u_split_initial << " " << w_split_initial << " " << levels << endl;
	pixel_info << srcgrid_xmin << " " << srcgrid_xmax << " " << srcgrid_ymin << " " << srcgrid_ymax << endl;
}

void SourcePixel::write_surface_brightness_to_file(ofstream &sb_outfile)
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

void SourcePixelGrid::get_grid_dimensions(double &xmin, double &xmax, double &ymin, double &ymax)
{
	xmin = cell[0][0]->corner_pt[0][0];
	ymin = cell[0][0]->corner_pt[0][1];
	xmax = cell[u_N-1][w_N-1]->corner_pt[3][0];
	ymax = cell[u_N-1][w_N-1]->corner_pt[3][1];
}

void SourcePixelGrid::plot_surface_brightness(string root)
{
	string img_filename = root + ".dat";
	string x_filename = root + ".x";
	string y_filename = root + ".y";
	string info_filename = root + ".info";
	string mag_filename = root + ".maglog";
	string n_image_filename = root + ".nimg";

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

	ofstream pixel_xvals; lens->open_output_file(pixel_xvals,x_filename);
	for (i=0, x=xmin; i <= n_plot_xcells; i++, x += cell_xlength) pixel_xvals << x << endl;

	ofstream pixel_yvals; lens->open_output_file(pixel_yvals,y_filename);
	for (i=0, y=ymin; i <= n_plot_ycells; i++, y += cell_ylength) pixel_yvals << y << endl;

	lens->open_output_file(pixel_surface_brightness_file,img_filename.c_str());
	lens->open_output_file(pixel_magnification_file,mag_filename.c_str());
	if (lens->n_image_prior) lens->open_output_file(pixel_n_image_file,n_image_filename.c_str());
	int line_number;
	for (j=0; j < w_N; j++) {
		for (line_number=0; line_number < pixels_per_cell_y; line_number++) {
			for (i=0; i < u_N; i++) {
				if (cell[i][j]->cell != NULL) {
					cell[i][j]->plot_cell_surface_brightness(line_number,pixels_per_cell_x,pixels_per_cell_y,pixel_surface_brightness_file,pixel_magnification_file,pixel_n_image_file);
				} else {
					for (k=0; k < pixels_per_cell_x; k++) {
						pixel_surface_brightness_file << cell[i][j]->surface_brightness << " ";
						pixel_magnification_file << log(cell[i][j]->total_magnification)/log(10) << " ";
						if (lens->n_image_prior) pixel_n_image_file << cell[i][j]->n_images << " ";
					}
				}
			}
			pixel_surface_brightness_file << endl;
			pixel_magnification_file << endl;
			if (lens->n_image_prior) pixel_n_image_file << endl;
		}
	}
	pixel_surface_brightness_file.close();
	pixel_magnification_file.close();
	pixel_n_image_file.close();

	ofstream pixel_info; lens->open_output_file(pixel_info,info_filename);
	pixel_info << u_split_initial << " " << w_split_initial << " " << levels << endl;
	pixel_info << srcgrid_xmin << " " << srcgrid_xmax << " " << srcgrid_ymin << " " << srcgrid_ymax << endl;
}

void SourcePixelGrid::output_fits_file(string fits_filename)
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
	if (lens->fit_output_dir != ".") lens->create_output_directory(); // in case it hasn't been created already
	string filename = lens->fit_output_dir + "/" + fits_filename;

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

void SourcePixel::plot_cell_surface_brightness(int line_number, int pixels_per_cell_x, int pixels_per_cell_y, ofstream& sb_outfile, ofstream& mag_outfile, ofstream &nimg_outfile)
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

void SourcePixelGrid::assign_firstlevel_neighbors()
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
void SourcePixelGrid::assign_neighborhood()
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

void SourcePixelGrid::assign_all_neighbors()
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

void SourcePixel::test_neighbors() // for testing purposes, to make sure neighbors are assigned correctly
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

void SourcePixel::assign_level_neighbors(int neighbor_level)
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

void SourcePixel::split_cells(const int usplit, const int wsplit, const int& thread)
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

	cell = new SourcePixel**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new SourcePixel*[w_N];
		for (j=0; j < w_N; j++) {
			cell[i][j] = new SourcePixel(lens,xvals_threads[thread],i,j,level+1,parent_grid);
			cell[i][j]->total_magnification = 0;
			if (lens->n_image_prior) cell[i][j]->n_images = 0;
		}
	}
	if (level == maxlevs[thread]) {
		maxlevs[thread]++; // our subcells are at the max level, so splitting them increases the number of levels by 1
	}
	parent_grid->number_of_pixels += u_N*w_N - 1; // subtract one because we're not counting the parent cell as a source pixel
}

void SourcePixel::unsplit()
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

void QLens::plot_source_pixel_grid(const int zsrc_i, const char filename[])
{
	if ((zsrc_i >= 0) and (n_extended_src_redshifts==0)) die("no ext src redshift created");
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	SourcePixelGrid *cartesian_srcgrid = image_pixel_grid->cartesian_srcgrid;

	if (cartesian_srcgrid==NULL) { warn("No source surface brightness map has been generated"); return; }
	cartesian_srcgrid->xgrid.open(filename, ifstream::out);
	cartesian_srcgrid->plot_corner_coordinates(cartesian_srcgrid->xgrid);
	cartesian_srcgrid->xgrid.close();
}

void SourcePixel::plot_corner_coordinates(ofstream &gridout)
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

double SourcePixelGrid::find_triangle_weighted_invmag(lensvector& pt1, lensvector& pt2, lensvector& pt3, double& total_overlap, const int thread)
{
	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_overlap(pt1,pt2,pt3,thread)==false) return false; 

	total_overlap = 0;
	double total_weighted_invmag = 0;
	double overlap;
	int i,j;
	lensvector *cornerpt;
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

void SourcePixel::find_triangle_weighted_invmag_subcell(lensvector& pt1, lensvector& pt2, lensvector& pt3, double& total_overlap, double& total_weighted_invmag, const int& thread)
{
	int i,j;
	double overlap;
	lensvector *cornerpt;
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

inline bool SourcePixel::check_if_in_neighborhood(lensvector **input_corner_pts, bool& inside, const int& thread)
{
	if (trirec[thread].determine_if_in_neighborhood(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],*input_corner_pts[3],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1],inside)==true) return true;
	return false;
}

inline bool SourcePixel::check_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
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

inline double SourcePixel::find_rectangle_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread, const int& i, const int& j)
{
	if (twist_status==0) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else if (twist_status==1) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	}
}

inline bool SourcePixel::check_triangle1_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else if (twist_status==1) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	}
}

inline bool SourcePixel::check_triangle2_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else if (twist_status==1) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else {
		return trirec[thread].determine_if_overlap(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	}
}

inline double SourcePixel::find_triangle1_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else if (twist_status==1) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	}
}

inline double SourcePixel::find_triangle2_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else if (twist_status==1) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else {
		return (trirec[thread].find_overlap_area(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	}
}

bool SourcePixelGrid::bisection_search_overlap(lensvector **input_corner_pts, const int& thread)
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

bool SourcePixelGrid::bisection_search_overlap(lensvector &a, lensvector &b, lensvector &c, const int& thread)
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

void SourcePixelGrid::calculate_pixel_magnifications(const bool use_emask)
{
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*(lens->group_comm), *(lens->mpi_group), &sub_comm);
#endif

	lens->total_srcgrid_overlap_area = 0; // Used to find the total coverage of the sourcegrid, which helps determine optimal source pixel size
	lens->high_sn_srcgrid_overlap_area = 0; // Used to find the total coverage of the sourcegrid, which helps determine optimal source pixel size

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

#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	//ofstream wout("wout.dat");
	double xstep, ystep;
	xstep = (srcgrid_xmax-srcgrid_xmin)/u_N;
	ystep = (srcgrid_ymax-srcgrid_ymin)/w_N;
	int src_raytrace_i, src_raytrace_j;
	int img_i, img_j;

	long int ntot_cells = (use_emask) ? image_pixel_grid->ntot_cells_emask : image_pixel_grid->ntot_cells;

	int *overlap_matrix_row_nn = new int[ntot_cells];
	vector<double> *overlap_matrix_rows = new vector<double>[ntot_cells];
	vector<int> *overlap_matrix_index_rows = new vector<int>[ntot_cells];
	vector<double> *overlap_area_matrix_rows;
	overlap_area_matrix_rows = new vector<double>[ntot_cells];

	int mpi_chunk, mpi_start, mpi_end;
	mpi_chunk = ntot_cells / lens->group_np;
	mpi_start = lens->group_id*mpi_chunk;
	if (lens->group_id == lens->group_np-1) mpi_chunk += (ntot_cells % lens->group_np); // assign the remainder elements to the last mpi process
	mpi_end = mpi_start + mpi_chunk;

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
		for (n=mpi_start; n < mpi_end; n++)
		{
			overlap_matrix_row_nn[n] = 0;
			//img_j = n / image_pixel_grid->x_N;
			//img_i = n % image_pixel_grid->x_N;
			if (use_emask) {
				img_j = image_pixel_grid->emask_pixels_j[n];
				img_i = image_pixel_grid->emask_pixels_i[n];
			} else {
				img_j = image_pixel_grid->masked_pixels_j[n];
				img_i = image_pixel_grid->masked_pixels_i[n];
			}
			if (image_pixel_grid->pixel_mag[img_i][img_j] < lens->srcpixel_nimg_mag_threshold) continue;
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

#ifdef USE_MPI
	MPI_Allreduce(&overlap_matrix_nn_part, &overlap_matrix_nn, 1, MPI_INT, MPI_SUM, sub_comm);
#else
	overlap_matrix_nn = overlap_matrix_nn_part;
#endif

	double *overlap_matrix = new double[overlap_matrix_nn];
	int *overlap_matrix_index = new int[overlap_matrix_nn];
	int *image_pixel_location_overlap = new int[ntot_cells+1];
	double *overlap_area_matrix;
	overlap_area_matrix = new double[overlap_matrix_nn];

#ifdef USE_MPI
	int id, chunk, start, end, length;
	for (id=0; id < lens->group_np; id++) {
		chunk = ntot_cells / lens->group_np;
		start = id*chunk;
		if (id == lens->group_np-1) chunk += (ntot_cells % lens->group_np); // assign the remainder elements to the last mpi process
		MPI_Bcast(overlap_matrix_row_nn + start,chunk,MPI_INT,id,sub_comm);
	}
#endif

	image_pixel_location_overlap[0] = 0;
	int n,l;
	for (n=0; n < ntot_cells; n++) {
		image_pixel_location_overlap[n+1] = image_pixel_location_overlap[n] + overlap_matrix_row_nn[n];
	}

	int indx;
	for (n=mpi_start; n < mpi_end; n++) {
		indx = image_pixel_location_overlap[n];
		for (j=0; j < overlap_matrix_row_nn[n]; j++) {
			overlap_matrix[indx+j] = overlap_matrix_rows[n][j];
			overlap_matrix_index[indx+j] = overlap_matrix_index_rows[n][j];
			overlap_area_matrix[indx+j] = overlap_area_matrix_rows[n][j];
		}
	}

#ifdef USE_MPI
	for (id=0; id < lens->group_np; id++) {
		chunk = ntot_cells / lens->group_np;
		start = id*chunk;
		if (id == lens->group_np-1) chunk += (ntot_cells % lens->group_np); // assign the remainder elements to the last mpi process
		end = start + chunk;
		length = image_pixel_location_overlap[end] - image_pixel_location_overlap[start];
		MPI_Bcast(overlap_matrix + image_pixel_location_overlap[start],length,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(overlap_matrix_index + image_pixel_location_overlap[start],length,MPI_INT,id,sub_comm);
		MPI_Bcast(overlap_area_matrix + image_pixel_location_overlap[start],length,MPI_DOUBLE,id,sub_comm);
	}
	MPI_Comm_free(&sub_comm);
#endif

	for (n=0; n < ntot_cells; n++) {
		//img_j = n / image_pixel_grid->x_N;
		//img_i = n % image_pixel_grid->x_N;
		if (use_emask) {
			img_j = image_pixel_grid->emask_pixels_j[n];
			img_i = image_pixel_grid->emask_pixels_i[n];
		} else {
			img_j = image_pixel_grid->masked_pixels_j[n];
			img_i = image_pixel_grid->masked_pixels_i[n];
		}
		for (l=image_pixel_location_overlap[n]; l < image_pixel_location_overlap[n+1]; l++) {
			nsrc = overlap_matrix_index[l];
			j = nsrc / u_N;
			i = nsrc % u_N;
			mag_matrix[nsrc] += overlap_matrix[l];
			area_matrix[nsrc] += overlap_area_matrix[l];
			if ((image_pixel_grid->pixel_in_mask != NULL) and (lens->image_pixel_data->high_sn_pixel[img_i][img_j])) high_sn_area_matrix[nsrc] += overlap_area_matrix[l];
			cell[i][j]->overlap_pixel_n.push_back(n);
			if ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[img_i][img_j]==true)) cell[i][j]->maps_to_image_window = true;
		}
	}

#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for finding source cell magnifications: " << wtime << endl;
	}
#endif

	//ofstream nimgout("auxnimg.dat");
	//ofstream nimgout2("auxnimg2.dat");
	for (nsrc=0; nsrc < ntot_src; nsrc++) {
		j = nsrc / u_N;
		i = nsrc % u_N;
		cell[i][j]->total_magnification = mag_matrix[nsrc] * image_pixel_grid->triangle_area / cell[i][j]->cell_area;
		cell[i][j]->avg_image_pixels_mapped = cell[i][j]->total_magnification * cell[i][j]->cell_area / image_pixel_grid->pixel_area;
		if (lens->n_image_prior) cell[i][j]->n_images = area_matrix[nsrc] / cell[i][j]->cell_area;
		//nimgout << cell[i][j]->center_pt[0] << " " << cell[i][j]->center_pt[1] << " " << cell[i][j]->n_images << endl;
		//nimgout2 << nsrc << " " << cell[i][j]->center_pt[0] << " " << cell[i][j]->center_pt[1] << " " << cell[i][j]->n_images << endl;

		if (area_matrix[nsrc] > cell[i][j]->cell_area) lens->total_srcgrid_overlap_area += cell[i][j]->cell_area;
		else lens->total_srcgrid_overlap_area += area_matrix[nsrc];
		if (image_pixel_grid->pixel_in_mask != NULL) {
			if (high_sn_area_matrix[nsrc] > cell[i][j]->cell_area) lens->high_sn_srcgrid_overlap_area += cell[i][j]->cell_area;
			else lens->high_sn_srcgrid_overlap_area += high_sn_area_matrix[nsrc];
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

double SourcePixelGrid::get_lowest_mag_sourcept(double &xsrc, double &ysrc)
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

void SourcePixelGrid::get_highest_mag_sourcept(double &xsrc, double &ysrc)
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

void SourcePixelGrid::adaptive_subgrid()
{
	calculate_pixel_magnifications();
#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int i, prev_levels;
	for (i=0; i < max_levels-1; i++) {
		prev_levels = levels;
		split_subcells_firstlevel(i);
		if (prev_levels==levels) break; // no splitting occurred, so no need to attempt further subgridding
	}
	assign_all_neighbors();

#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for adaptive grid splitting: " << wtime << endl;
	}
#endif
}

void SourcePixelGrid::split_subcells_firstlevel(const int splitlevel)
{
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
		SourcePixel *subcell;
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
				if ((cell[i][j]->total_magnification*cell[i][j]->cell_area/(lens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) > lens->pixel_magnification_threshold) subgrid = true;
				if (subgrid) {
					//cout << "SPLITTING(FIRST): level=" << cell[i][j]->level << ", mag=" << cell[i][j]->total_magnification << " fac=" << (cell[i][j]->cell_area/(lens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) << endl;
					cell[i][j]->split_cells(2,2,thread);
					for (k=0; k < cell[i][j]->overlap_pixel_n.size(); k++) {
						nn = cell[i][j]->overlap_pixel_n[k];
						//img_j = nn / image_pixel_grid->x_N;
						//img_i = nn % image_pixel_grid->x_N;
						img_j = image_pixel_grid->masked_pixels_j[nn];
						img_i = image_pixel_grid->masked_pixels_i[nn];

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
								if (lens->n_image_prior) {
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
							if (lens->n_image_prior) subcell->n_images /= subcell->cell_area;
						}
					}
				}
			}
		}
		for (i=0; i < nthreads; i++) if (maxlevs[i] > parent_grid->levels) parent_grid->levels = maxlevs[i];
	}
}

void SourcePixel::split_subcells(const int splitlevel, const int thread)
{
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
		SourcePixel *subcell;
		bool subgrid;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				subgrid = false;
				if ((cell[i][j]->total_magnification*cell[i][j]->cell_area/(lens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) > lens->pixel_magnification_threshold) subgrid = true;

				if (subgrid) {
					//cout << "SPLITTING: level=" << cell[i][j]->level << ", mag=" << cell[i][j]->total_magnification << " fac=" << (cell[i][j]->cell_area/(lens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) << endl;
					cell[i][j]->split_cells(2,2,thread);
					for (k=0; k < cell[i][j]->overlap_pixel_n.size(); k++) {
						nn = cell[i][j]->overlap_pixel_n[k];
						//img_j = nn / image_pixel_grid->x_N;
						//img_i = nn % image_pixel_grid->x_N;
						img_j = image_pixel_grid->masked_pixels_j[nn];
						img_i = image_pixel_grid->masked_pixels_i[nn];

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

bool SourcePixelGrid::assign_source_mapping_flags_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, vector<SourcePixel*>& mapped_cartesian_srcpixels, const int& thread)
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

void SourcePixel::subcell_assign_source_mapping_flags_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, vector<SourcePixel*>& mapped_cartesian_srcpixels, const int& thread, bool& image_pixel_maps_to_source_grid)
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

void SourcePixelGrid::calculate_Lmatrix_overlap(const int &img_index, const int image_pixel_i, const int image_pixel_j, int& index, lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	double overlap, total_overlap=0;
	int i,j,k;
	int Lmatrix_index_initial = index;
	SourcePixel *subcell;

	for (i=0; i < image_pixel_grid->mapped_cartesian_srcpixels[image_pixel_i][image_pixel_j].size(); i++) {
		subcell = image_pixel_grid->mapped_cartesian_srcpixels[image_pixel_i][image_pixel_j][i];
		lens->Lmatrix_index_rows[img_index].push_back(subcell->active_index);
		overlap = subcell->find_rectangle_overlap(input_corner_pts,twist_pt,twist_status,thread,image_pixel_i,image_pixel_j);
		lens->Lmatrix_rows[img_index].push_back(overlap);
		index++;
		total_overlap += overlap;
	}

	if (total_overlap==0) die("image pixel should have mapped to at least one source pixel");
	for (i=Lmatrix_index_initial; i < index; i++)
		lens->Lmatrix_rows[img_index][i] /= total_overlap;
}

double SourcePixelGrid::find_lensed_surface_brightness_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
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

void SourcePixel::find_lensed_surface_brightness_subcell_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread, double& overlap, double& total_overlap, double& total_weighted_surface_brightness)
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

bool SourcePixelGrid::bisection_search_interpolate(lensvector &input_center_pt, const int& thread)
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

bool SourcePixelGrid::assign_source_mapping_flags_interpolate(lensvector &input_center_pt, vector<SourcePixel*>& mapped_cartesian_srcpixels, const int& thread, const int& image_pixel_i, const int& image_pixel_j)
{
	bool image_pixel_maps_to_source_grid = false;
	// when splitting image pixels, there could be multiple entries in the Lmatrix array that belong to the same source pixel; you might save computational time if these can be consolidated (by adding them together). Try this out later
	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_interpolate(input_center_pt,thread)==true) {
		int i,j,side;
		SourcePixel* cellptr;
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

bool SourcePixel::subcell_assign_source_mapping_flags_interpolate(lensvector &input_center_pt, vector<SourcePixel*>& mapped_cartesian_srcpixels, const int& thread)
{
	bool image_pixel_maps_to_source_grid = false;
	int i,j,side;
	SourcePixel* cellptr;
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

void SourcePixelGrid::calculate_Lmatrix_interpolate(const int img_index, vector<SourcePixel*>& mapped_cartesian_srcpixels, int& index, lensvector &input_center_pt, const int& ii, const double weight, const int& thread)
{
	for (int i=0; i < 3; i++) {
		//cout << "What " << i << endl;
		//cout << "ii=" << ii << " trying index " << (3*ii+i) << endl;
		//cout << "imgpix: " << image_pixel_i << " " << image_pixel_j << endl;
		//cout << "size: " << image_pixel_grid->mapped_cartesian_srcpixels[image_pixel_i][image_pixel_j].size() << endl;
		if (mapped_cartesian_srcpixels[3*ii+i] == NULL) return; // in this case, subpixel does not map to anything
		lens->Lmatrix_index_rows[img_index].push_back(mapped_cartesian_srcpixels[3*ii+i]->active_index);
		//cout << "What? " << i << endl;
		interpolation_pts[i][thread] = &mapped_cartesian_srcpixels[3*ii+i]->center_pt;
	}

	//if (lens->interpolate_sb_3pt) {
		double d = ((*interpolation_pts[0][thread])[0]-(*interpolation_pts[1][thread])[0])*((*interpolation_pts[1][thread])[1]-(*interpolation_pts[2][thread])[1]) - ((*interpolation_pts[1][thread])[0]-(*interpolation_pts[2][thread])[0])*((*interpolation_pts[0][thread])[1]-(*interpolation_pts[1][thread])[1]);
		lens->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[1][thread])[1]-(*interpolation_pts[2][thread])[1]) + input_center_pt[1]*((*interpolation_pts[2][thread])[0]-(*interpolation_pts[1][thread])[0]) + (*interpolation_pts[1][thread])[0]*(*interpolation_pts[2][thread])[1] - (*interpolation_pts[1][thread])[1]*(*interpolation_pts[2][thread])[0])/d);
		lens->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[2][thread])[1]-(*interpolation_pts[0][thread])[1]) + input_center_pt[1]*((*interpolation_pts[0][thread])[0]-(*interpolation_pts[2][thread])[0]) + (*interpolation_pts[0][thread])[1]*(*interpolation_pts[2][thread])[0] - (*interpolation_pts[0][thread])[0]*(*interpolation_pts[2][thread])[1])/d);
		lens->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[0][thread])[1]-(*interpolation_pts[1][thread])[1]) + input_center_pt[1]*((*interpolation_pts[1][thread])[0]-(*interpolation_pts[0][thread])[0]) + (*interpolation_pts[0][thread])[0]*(*interpolation_pts[1][thread])[1] - (*interpolation_pts[0][thread])[1]*(*interpolation_pts[1][thread])[0])/d);
		if (d==0) warn("d is zero!!!");
	//} else {
		//lens->Lmatrix_rows[img_index].push_back(weight);
		//lens->Lmatrix_rows[img_index].push_back(0);
		//lens->Lmatrix_rows[img_index].push_back(0);
	//}

	index += 3;
}

double SourcePixelGrid::find_lensed_surface_brightness_interpolate(lensvector &input_center_pt, const int& thread)
{
	lensvector *pts[3];
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

void SourcePixel::find_interpolation_cells(lensvector &input_center_pt, const int& thread)
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

SourcePixel* SourcePixel::find_nearest_neighbor_cell(lensvector &input_center_pt, const int& side)
{
	int i,ncells;
	SourcePixel **cells;
	if ((side==0) or (side==1)) ncells = w_N;
	else if ((side==2) or (side==3)) ncells = u_N;
	else die("side number cannot be larger than 3");
	cells = new SourcePixel*[ncells];

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
	SourcePixel *closest_cell = cells[i_min];
	delete[] cells;
	return closest_cell;
}

SourcePixel* SourcePixel::find_nearest_neighbor_cell(lensvector &input_center_pt, const int& side, const int tiebreaker_side)
{
	int i,ncells;
	SourcePixel **cells;
	if ((side==0) or (side==1)) ncells = w_N;
	else if ((side==2) or (side==3)) ncells = u_N;
	else die("side number cannot be larger than 3");
	cells = new SourcePixel*[ncells];
	double sqr_distance, min_sqr_distance = 1e30;
	SourcePixel *closest_cell = NULL;
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

void SourcePixel::find_nearest_two_cells(SourcePixel* &cellptr1, SourcePixel* &cellptr2, const int& side)
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

SourcePixel* SourcePixel::find_corner_cell(const int i, const int j)
{
	SourcePixel* cellptr = cell[i][j];
	while (cellptr->cell != NULL)
		cellptr = cellptr->cell[i][j];
	return cellptr;
}

double SourcePixelGrid::find_local_inverse_magnification_interpolate(lensvector &input_center_pt, const int& thread)
{
	lensvector *pts[3];
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
			//cout << "HI!" << endl;
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

void SourcePixel::generate_gmatrices()
{
	int i,j,k,l;
	SourcePixel *cellptr1, *cellptr2;
	double alpha, beta, dxfac;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->generate_gmatrices();
			else {
				if (cell[i][j]->active_pixel) {
					//dxfac = pow(1.3,-(cell[i][j]->level)); // seems like there's no real sensible reason to have a scaling factor here; delete this later
					dxfac = 1.0;
					for (k=0; k < 4; k++) {
						lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(1.0/dxfac);
						lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cell[i][j]->active_index);
						lens->gmatrix_row_nn[k][cell[i][j]->active_index]++;
						lens->gmatrix_nn[k]++;
						if (cell[i][j]->neighbor[k]) {
							if (cell[i][j]->neighbor[k]->cell != NULL) {
								cell[i][j]->neighbor[k]->find_nearest_two_cells(cellptr1,cellptr2,k);
								//cout << "cell 1: " << cellptr1->center_pt[0] << " " << cellptr1->center_pt[1] << endl;
								//cout << "cell 2: " << cellptr2->center_pt[0] << " " << cellptr2->center_pt[1] << endl;
								if ((cellptr1==NULL) or (cellptr2==NULL)) die("Hmm, not getting back two cells");
								if (k < 2) {
									// interpolating surface brightness along x-direction
									alpha = abs((cell[i][j]->center_pt[1] - cellptr1->center_pt[1]) / (cellptr2->center_pt[1] - cellptr1->center_pt[1]));
								} else {
									// interpolating surface brightness along y-direction
									alpha = abs((cell[i][j]->center_pt[0] - cellptr1->center_pt[0]) / (cellptr2->center_pt[0] - cellptr1->center_pt[0]));
								}
								beta = 1-alpha;
								if (cellptr1->active_pixel) {
									if (!cellptr2->active_pixel) beta=1; // just in case the other point is no good
									lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(-beta/dxfac);
									lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr1->active_index);
									lens->gmatrix_row_nn[k][cell[i][j]->active_index]++;
									lens->gmatrix_nn[k]++;
								}
								if (cellptr2->active_pixel) {
									if (!cellptr1->active_pixel) alpha=1; // just in case the other point is no good
									lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(-alpha/dxfac);
									lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr2->active_index);
									lens->gmatrix_row_nn[k][cell[i][j]->active_index]++;
									lens->gmatrix_nn[k]++;
								}
							}
							else if (cell[i][j]->neighbor[k]->active_pixel) {
								if (cell[i][j]->neighbor[k]->level==cell[i][j]->level) {
									lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(-1.0/dxfac);
									lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cell[i][j]->neighbor[k]->active_index);
									lens->gmatrix_row_nn[k][cell[i][j]->active_index]++;
									lens->gmatrix_nn[k]++;
								} else {
									cellptr1 = cell[i][j]->neighbor[k];
									if (k < 2) {
										if (cellptr1->center_pt[1] > cell[i][j]->center_pt[1]) l=3;
										else l=2;
									} else {
										if (cellptr1->center_pt[0] > cell[i][j]->center_pt[0]) l=1;
										else l=0;
									}
									if ((cellptr1->neighbor[l]==NULL) or ((cellptr1->neighbor[l]->cell==NULL) and (!cellptr1->neighbor[l]->active_pixel))) {
										// There is no useful nearby neighbor to interpolate with, so just use the single neighbor pixel
										lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(-1.0/dxfac);
										lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr1->active_index);
										lens->gmatrix_row_nn[k][cell[i][j]->active_index]++;
										lens->gmatrix_nn[k]++;
									} else {
										if (cellptr1->neighbor[l]->cell==NULL) cellptr2 = cellptr1->neighbor[l];
										else cellptr2 = cellptr1->neighbor[l]->find_nearest_neighbor_cell(cellptr1->center_pt,l,k%2); // the tiebreaker k%2 ensures that preference goes to cells that are closer to this cell in order to interpolate to find the gradient
										if (cellptr2==NULL) die("Subcell does not map to source pixel; regularization currently cannot handle unmapped subcells");
										if (k < 2) alpha = abs((cell[i][j]->center_pt[1] - cellptr1->center_pt[1]) / (cellptr2->center_pt[1] - cellptr1->center_pt[1]));
										else alpha = abs((cell[i][j]->center_pt[0] - cellptr1->center_pt[0]) / (cellptr2->center_pt[0] - cellptr1->center_pt[0]));
										beta = 1-alpha;
										//cout << alpha << " " << beta << " " << k << " " << l << " " << ii << " " << jj << " " << i << " " << j << endl;
										//cout << cell[i][j]->center_pt[0] << " " << cellptr1->center_pt[0] << " " << cellptr1->center_pt[1] << " " << cellptr2->center_pt[0] << " " << cellptr2->center_pt[1] << endl;
										if (cellptr1->active_pixel) {
											if (!cellptr2->active_pixel) beta=1; // just in case the other point is no good
											lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(-beta/dxfac);
											lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr1->active_index);
											lens->gmatrix_row_nn[k][cell[i][j]->active_index]++;
											lens->gmatrix_nn[k]++;
										}
										if (cellptr2->active_pixel) {
											if (!cellptr1->active_pixel) alpha=1; // just in case the other point is no good
											lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(-alpha/dxfac);
											lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr2->active_index);
											lens->gmatrix_row_nn[k][cell[i][j]->active_index]++;
											lens->gmatrix_nn[k]++;
										}

										//lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(-beta);
										//lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr1->active_index);
										//lens->gmatrix_rows[k][cell[i][j]->active_index].push_back(-alpha);
										//lens->gmatrix_index_rows[k][cell[i][j]->active_index].push_back(cellptr2->active_index);
										//lens->gmatrix_row_nn[k][cell[i][j]->active_index] += 2;
										//lens->gmatrix_nn[k] += 2;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void SourcePixel::generate_hmatrices()
{
	int i,j,k,l,m,kmin,kmax;
	SourcePixel *cellptr1, *cellptr2;
	double alpha, beta;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->generate_hmatrices();
			else {
				for (l=0; l < 2; l++) {
					if (cell[i][j]->active_pixel) {
						lens->hmatrix_rows[l][cell[i][j]->active_index].push_back(-2);
						lens->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cell[i][j]->active_index);
						lens->hmatrix_row_nn[l][cell[i][j]->active_index]++;
						lens->hmatrix_nn[l]++;
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
									if (!cellptr1->active_pixel) alpha=1;
									if (!cellptr2->active_pixel) beta=1;
									if (cellptr1->active_pixel) {
										lens->hmatrix_rows[l][cell[i][j]->active_index].push_back(beta);
										lens->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr1->active_index);
										lens->hmatrix_row_nn[l][cell[i][j]->active_index]++;
										lens->hmatrix_nn[l]++;
									}
									if (cellptr2->active_pixel) {
										lens->hmatrix_rows[l][cell[i][j]->active_index].push_back(alpha);
										lens->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr2->active_index);
										lens->hmatrix_row_nn[l][cell[i][j]->active_index]++;
										lens->hmatrix_nn[l]++;
									}

								}
								else if (cell[i][j]->neighbor[k]->active_pixel) {
									if (cell[i][j]->neighbor[k]->level==cell[i][j]->level) {
										lens->hmatrix_rows[l][cell[i][j]->active_index].push_back(1);
										lens->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cell[i][j]->neighbor[k]->active_index);
										lens->hmatrix_row_nn[l][cell[i][j]->active_index]++;
										lens->hmatrix_nn[l]++;
									} else {
										cellptr1 = cell[i][j]->neighbor[k];
										if (k < 2) {
											if (cellptr1->center_pt[1] > cell[i][j]->center_pt[1]) m=3;
											else m=2;
										} else {
											if (cellptr1->center_pt[0] > cell[i][j]->center_pt[0]) m=1;
											else m=0;
										}
										if ((cellptr1->neighbor[m]==NULL) or ((cellptr1->neighbor[m]->cell==NULL) and (!cellptr1->neighbor[m]->active_pixel))) {
											// There is no useful nearby neighbor to interpolate with, so just use the single neighbor pixel
											lens->hmatrix_rows[l][cell[i][j]->active_index].push_back(1);
											lens->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr1->active_index);
											lens->hmatrix_row_nn[l][cell[i][j]->active_index]++;
											lens->hmatrix_nn[l]++;
										} else {
											if (cellptr1->neighbor[m]->cell==NULL) cellptr2 = cellptr1->neighbor[m];
											else cellptr2 = cellptr1->neighbor[m]->find_nearest_neighbor_cell(cellptr1->center_pt,m,k%2); // the tiebreaker k%2 ensures that preference goes to cells that are closer to this cell in order to interpolate to find the curvature
											if (cellptr2==NULL) die("Subcell does not map to source pixel; regularization currently cannot handle unmapped subcells");
											if (k < 2) alpha = abs((cell[i][j]->center_pt[1] - cellptr1->center_pt[1]) / (cellptr2->center_pt[1] - cellptr1->center_pt[1]));
											else alpha = abs((cell[i][j]->center_pt[0] - cellptr1->center_pt[0]) / (cellptr2->center_pt[0] - cellptr1->center_pt[0]));
											beta = 1-alpha;
											//cout << alpha << " " << beta << " " << k << " " << m << " " << ii << " " << jj << " " << i << " " << j << endl;
											//cout << cell[i][j]->center_pt[0] << " " << cellptr1->center_pt[0] << " " << cellptr1->center_pt[1] << " " << cellptr2->center_pt[0] << " " << cellptr2->center_pt[1] << endl;
											if (!cellptr1->active_pixel) alpha=1;
											if (!cellptr2->active_pixel) beta=1;
											if (cellptr1->active_pixel) {
												lens->hmatrix_rows[l][cell[i][j]->active_index].push_back(beta);
												lens->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr1->active_index);
												lens->hmatrix_row_nn[l][cell[i][j]->active_index]++;
												lens->hmatrix_nn[l]++;
											}
											if (cellptr2->active_pixel) {
												lens->hmatrix_rows[l][cell[i][j]->active_index].push_back(alpha);
												lens->hmatrix_index_rows[l][cell[i][j]->active_index].push_back(cellptr2->active_index);
												lens->hmatrix_row_nn[l][cell[i][j]->active_index]++;
												lens->hmatrix_nn[l]++;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void QLens::generate_Rmatrix_from_hmatrices(const int zsrc_i, const bool interpolate)
{
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int i,j,k,l,m,n,indx;

	vector<int> *jvals[2];
	vector<int> *lvals[2];
	for (i=0; i < 2; i++) {
		jvals[i] = new vector<int>[source_npixels];
		lvals[i] = new vector<int>[source_npixels];
	}

	Rmatrix_diag_temp = new double[source_npixels];
	Rmatrix_rows = new vector<double>[source_npixels];
	Rmatrix_index_rows = new vector<int>[source_npixels];
	Rmatrix_row_nn = new int[source_npixels];
	Rmatrix_nn = 0;
	int Rmatrix_nn_part = 0;
	for (j=0; j < source_npixels; j++) {
		Rmatrix_diag_temp[j] = 0;
		Rmatrix_row_nn[j] = 0;
	}

	bool new_entry;
	int src_index1, src_index2, col_index, col_i;
	double tmp, element;

	for (k=0; k < 2; k++) {
		hmatrix_rows[k] = new vector<double>[source_npixels];
		hmatrix_index_rows[k] = new vector<int>[source_npixels];
		hmatrix_row_nn[k] = new int[source_npixels];
		hmatrix_nn[k] = 0;
		for (j=0; j < source_npixels; j++) {
			hmatrix_row_nn[k][j] = 0;
		}
	}
	if (source_fit_mode==Delaunay_Source) {
		if (zsrc_i < 0) image_pixel_grid0->delaunay_srcgrid->generate_hmatrices(interpolate);
		else image_pixel_grids[zsrc_i]->delaunay_srcgrid->generate_hmatrices(interpolate);
	}
	else if (source_fit_mode==Cartesian_Source) {
		if (zsrc_i < 0) image_pixel_grid0->cartesian_srcgrid->generate_hmatrices();
		else image_pixel_grids[zsrc_i]->cartesian_srcgrid->generate_hmatrices();
	}
	else die("hmatrix not supported for sources other than Delaunay or Cartesian");

	for (k=0; k < 2; k++) {
		hmatrix[k] = new double[hmatrix_nn[k]];
		hmatrix_index[k] = new int[hmatrix_nn[k]];
		hmatrix_row_index[k] = new int[source_npixels+1];

		hmatrix_row_index[k][0] = 0;
		for (i=0; i < source_npixels; i++)
			hmatrix_row_index[k][i+1] = hmatrix_row_index[k][i] + hmatrix_row_nn[k][i];
		if (hmatrix_row_index[k][source_npixels] != hmatrix_nn[k]) die("the number of elements don't match up for hmatrix %i",k);

		for (i=0; i < source_npixels; i++) {
			indx = hmatrix_row_index[k][i];
			for (j=0; j < hmatrix_row_nn[k][i]; j++) {
				hmatrix[k][indx+j] = hmatrix_rows[k][i][j];
				hmatrix_index[k][indx+j] = hmatrix_index_rows[k][i][j];
			}
		}
		delete[] hmatrix_rows[k];
		delete[] hmatrix_index_rows[k];
		delete[] hmatrix_row_nn[k];

		for (i=0; i < source_npixels; i++) {
			for (j=hmatrix_row_index[k][i]; j < hmatrix_row_index[k][i+1]; j++) {
				for (l=j; l < hmatrix_row_index[k][i+1]; l++) {
					src_index1 = hmatrix_index[k][j];
					src_index2 = hmatrix_index[k][l];
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
	for (src_index1=0; src_index1 < source_npixels; src_index1++) {
		for (k=0; k < 2; k++) {
			col_i=0;
			for (n=0; n < jvals[k][src_index1].size(); n++) {
				j = jvals[k][src_index1][n];
				l = lvals[k][src_index1][n];
				src_index2 = hmatrix_index[k][l];
				new_entry = true;
				element = hmatrix[k][j]*hmatrix[k][l];
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

	for (k=0; k < 2; k++) {
		delete[] hmatrix[k];
		delete[] hmatrix_index[k];
		delete[] hmatrix_row_index[k];
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating Rmatrix: " << wtime << endl;
	}
#endif

	Rmatrix_nn = Rmatrix_nn_part;
	Rmatrix_nn += source_npixels+1;

	Rmatrix = new double[Rmatrix_nn];
	Rmatrix_index = new int[Rmatrix_nn];

	for (i=0; i < source_npixels; i++)
		Rmatrix[i] = Rmatrix_diag_temp[i];

	Rmatrix_index[0] = source_npixels+1;
	for (i=0; i < source_npixels; i++) {
		Rmatrix_index[i+1] = Rmatrix_index[i] + Rmatrix_row_nn[i];
	}

	for (i=0; i < source_npixels; i++) {
		indx = Rmatrix_index[i];
		for (j=0; j < Rmatrix_row_nn[i]; j++) {
			Rmatrix[indx+j] = Rmatrix_rows[i][j];
			Rmatrix_index[indx+j] = Rmatrix_index_rows[i][j];
		}
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

void QLens::generate_Rmatrix_from_gmatrices(const int zsrc_i, const bool interpolate)
{
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int i,j,k,l,m,n,indx;

	vector<int> *jvals[4];
	vector<int> *lvals[4];
	for (i=0; i < 4; i++) {
		jvals[i] = new vector<int>[source_npixels];
		lvals[i] = new vector<int>[source_npixels];
	}

	Rmatrix_diag_temp = new double[source_npixels];
	Rmatrix_rows = new vector<double>[source_npixels];
	Rmatrix_index_rows = new vector<int>[source_npixels];
	Rmatrix_row_nn = new int[source_npixels];
	Rmatrix_nn = 0;
	int Rmatrix_nn_part = 0;
	for (j=0; j < source_npixels; j++) {
		Rmatrix_diag_temp[j] = 0;
		Rmatrix_row_nn[j] = 0;
	}

	bool new_entry;
	int src_index1, src_index2, col_index, col_i;
	double tmp, element;

	for (k=0; k < 4; k++) {
		gmatrix_rows[k] = new vector<double>[source_npixels];
		gmatrix_index_rows[k] = new vector<int>[source_npixels];
		gmatrix_row_nn[k] = new int[source_npixels];
		gmatrix_nn[k] = 0;
		for (j=0; j < source_npixels; j++) {
			gmatrix_row_nn[k][j] = 0;
		}
	}
	if (source_fit_mode==Delaunay_Source) {
		if (zsrc_i < 0) image_pixel_grid0->delaunay_srcgrid->generate_gmatrices(interpolate);
		else image_pixel_grids[zsrc_i]->delaunay_srcgrid->generate_gmatrices(interpolate);
	}
	else if (source_fit_mode==Cartesian_Source) {
		if (zsrc_i < 0) image_pixel_grid0->cartesian_srcgrid->generate_gmatrices();
		else image_pixel_grids[zsrc_i]->cartesian_srcgrid->generate_gmatrices();
	}
	else die("gmatrix not supported for sources other than Delaunay or Cartesian");

	for (k=0; k < 4; k++) {
		gmatrix[k] = new double[gmatrix_nn[k]];
		gmatrix_index[k] = new int[gmatrix_nn[k]];
		gmatrix_row_index[k] = new int[source_npixels+1];

		gmatrix_row_index[k][0] = 0;
		for (i=0; i < source_npixels; i++)
			gmatrix_row_index[k][i+1] = gmatrix_row_index[k][i] + gmatrix_row_nn[k][i];
		if (gmatrix_row_index[k][source_npixels] != gmatrix_nn[k]) die("the number of elements don't match up for gmatrix %i",k);

		for (i=0; i < source_npixels; i++) {
			indx = gmatrix_row_index[k][i];
			for (j=0; j < gmatrix_row_nn[k][i]; j++) {
				gmatrix[k][indx+j] = gmatrix_rows[k][i][j];
				gmatrix_index[k][indx+j] = gmatrix_index_rows[k][i][j];
			}
		}
		delete[] gmatrix_rows[k];
		delete[] gmatrix_index_rows[k];
		delete[] gmatrix_row_nn[k];

		for (i=0; i < source_npixels; i++) {
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
	for (src_index1=0; src_index1 < source_npixels; src_index1++) {
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

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating Rmatrix: " << wtime << endl;
	}
#endif

	Rmatrix_nn = Rmatrix_nn_part;
	Rmatrix_nn += source_npixels+1;

	Rmatrix = new double[Rmatrix_nn];
	Rmatrix_index = new int[Rmatrix_nn];

	for (i=0; i < source_npixels; i++)
		Rmatrix[i] = Rmatrix_diag_temp[i];

	Rmatrix_index[0] = source_npixels+1;
	for (i=0; i < source_npixels; i++) {
		Rmatrix_index[i+1] = Rmatrix_index[i] + Rmatrix_row_nn[i];
	}

	for (i=0; i < source_npixels; i++) {
		indx = Rmatrix_index[i];
		for (j=0; j < Rmatrix_row_nn[i]; j++) {
			Rmatrix[indx+j] = Rmatrix_rows[i][j];
			Rmatrix_index[indx+j] = Rmatrix_index_rows[i][j];
		}
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

bool QLens::generate_Rmatrix_from_covariance_kernel(const int zsrc_i, const int kernel_type, const bool allow_lum_weighting, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	int ntot = source_npixels*(source_npixels+1)/2;
	double xc_approx, yc_approx, sig;
	covmatrix_packed.input(ntot);
	covmatrix_factored.input(ntot);
	Rmatrix_packed.input(ntot);
	if (source_fit_mode==Delaunay_Source) {
		//if ((use_distance_weighted_regularization) or ((kernel_type==0) and (use_matern_scale_parameter))) {
		if (use_distance_weighted_regularization) {
			sig = image_pixel_grid->find_approx_source_size(xc_approx,yc_approx,verbal);
			if ((verbal) and (mpi_id==0)) cout << "approx source size=" << sig << ", src_xc_approx=" << xc_approx << " src_yc_approx=" << yc_approx << endl;
			//if (use_matern_scale_parameter) {
				//matern_approx_source_size = sig/3;
				//wtime0 = omp_get_wtime();
				//set_corrlength_for_given_matscale();
				//wtime = omp_get_wtime() - wtime0;
				//if (mpi_id==0) cout << "Wall time for calculating corrlength: " << wtime << endl;
			//}
			if (fix_lumreg_sig) sig = lumreg_sig;
			calculate_distreg_srcpixel_weights(zsrc_i,xc_approx,yc_approx,sig,verbal);
		}
		if (use_mag_weighted_regularization) calculate_mag_srcpixel_weights(zsrc_i);

		double *wgtfac = ((use_distance_weighted_regularization) or (use_mag_weighted_regularization) or ((allow_lum_weighting) and (use_lum_weighted_regularization))) ? reg_weight_factor : NULL;
		image_pixel_grid->delaunay_srcgrid->generate_covariance_matrix(covmatrix_packed.array(),kernel_correlation_length,kernel_type,matern_index,wgtfac);
		//if (((allow_lum_weighting) or (use_distance_weighted_regularization)) and (use_second_covariance_kernel)) {
			//delaunay_srcgrid->generate_covariance_matrix(covmatrix_packed.array(),kernel2_correlation_length,kernel_type,matern_index,reg_weight_factor2,true,kernel2_amplitude_ratio); // uses exponential kernel
		//}
	}
	else die("covariance kernel regularization requires source mode to be 'delaunay'");
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating covariance matrix: " << wtime << endl;
	}
#endif

	for (int i=0; i < ntot; i++) covmatrix_factored[i] = covmatrix_packed[i];
#ifdef USE_MKL
	lapack_int status;
   status = LAPACKE_dpptrf(LAPACK_ROW_MAJOR,'U',source_npixels,covmatrix_factored.array()); // Cholesky decomposition
	if ((status != 0) and (penalize_defective_covmatrix)) return false;
	//if (status > 0) warn("cholesky decomposition of covmatrix was not successful; covmatrix is not positive definite");
	Cholesky_logdet_packed(covmatrix_factored.array(),Rmatrix_log_determinant,source_npixels);
	Rmatrix_log_determinant = -Rmatrix_log_determinant; // since this was the (log-)determinant of the inverse of the Rmatrix (i.e. using det(cov) = 1/det(cov_inverse))
	if (!use_covariance_matrix) {
		// Since we're going to use R-matrix explicitly, we must find it by taking cov_inverse
		for (int i=0; i < ntot; i++) Rmatrix_packed[i] = covmatrix_factored[i];
		lapack_int status;
		status = LAPACKE_dpptri(LAPACK_ROW_MAJOR,'U',source_npixels,Rmatrix_packed.array()); // computes inverse; this is where most of the computational burden is
		//if (status != 0) { warn("covmatrix inversion was not successful"); return false; }

		//Gmatrix_stacked.input(source_n_amps*source_n_amps); // just used here to check if inverse is correct
		//dvector Rmatrix_stacked(source_n_amps*source_n_amps);
		//covmatrix_stacked.input(source_n_amps*source_n_amps);
		//Rmatrix_stacked = 0;
		//covmatrix_stacked = 0;
		//Gmatrix_stacked = 0;
		//LAPACKE_mkl_dtpunpack(LAPACK_ROW_MAJOR,'U','N',source_npixels,covmatrix_packed.array(),1,1,source_n_amps,source_npixels,covmatrix_stacked.array(),source_n_amps);
		//LAPACKE_mkl_dtpunpack(LAPACK_ROW_MAJOR,'U','T',source_npixels,covmatrix_packed.array(),1,1,source_n_amps,source_npixels,covmatrix_stacked.array(),source_n_amps);
		//LAPACKE_mkl_dtpunpack(LAPACK_ROW_MAJOR,'U','N',source_npixels,Rmatrix_packed.array(),1,1,source_n_amps,source_npixels,Rmatrix_stacked.array(),source_n_amps);
		//LAPACKE_mkl_dtpunpack(LAPACK_ROW_MAJOR,'U','T',source_npixels,Rmatrix_packed.array(),1,1,source_n_amps,source_npixels,Rmatrix_stacked.array(),source_n_amps);
		//cblas_dsymm(CblasRowMajor,CblasLeft,CblasUpper,source_n_amps,source_n_amps,1.0,covmatrix_stacked.array(),source_n_amps,Rmatrix_stacked.array(),source_n_amps,0,Gmatrix_stacked.array(),source_n_amps);
		//int i,j;
		//for (i=0,j=0; i < source_n_amps; i++) {
			//if (abs(Gmatrix_stacked[j]-1.0) > 1e-2) {
				//warn("Rmatrix times covmatrix does not produce identity matrix");
				//break;
			//}
			//j += source_n_amps+1;
		//}
		//int k=0;
		//for (i=0; i < 3; i++) {
			//for (j=0; j < source_n_amps; j++) {
				//cout << Gmatrix_stacked[k++] << " ";
			//}
			//cout << endl;
		//}
	}
#else
	// Doing this without MKL, using the following functions, is MUCH slower (and might be broken right now?)
	repack_matrix_lower(covmatrix_factored);
	Cholesky_dcmp_packed(covmatrix_factored.array(),source_n_amps);
	Cholesky_logdet_lower_packed(covmatrix_factored.array(),Rmatrix_log_determinant,source_n_amps);
	Rmatrix_log_determinant = -Rmatrix_log_determinant; // since this was the (log-)determinant of the inverse of the Rmatrix (i.e. using det(cov) = 1/det(cov_inverse))
	repack_matrix_upper(covmatrix_factored);
	//for (int i=0; i < ntot; i++) Rmatrix_packed[i] = covmatrix_factored[i];
	//Cholesky_invert_upper_packed(Rmatrix_packed.array(),source_npixels); // invert the triangular matrix to get U_inverse
	//upper_triangular_syrk(Rmatrix_packed.array(),source_npixels); // Now take U_inverse * U_inverse_transpose to get C_inverse (the regularization matrix)
#endif

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating covariance kernel Rmatrix: " << wtime << endl;
	}
#endif
	return true;
}

double QLens::find_approx_source_size(const int zsrc_i, double& xc_approx, double& yc_approx, const bool verbal)
{
	double sig;
	if ((image_pixel_grids==NULL) or (image_pixel_grids[zsrc_i]==NULL)) {
		warn("cannot find approximate source size; image pixel grid does not exist");
		xc_approx = 1e30;
		yc_approx = 1e30;
		sig = 1.0;
	} else {
		sig = image_pixel_grids[zsrc_i]->find_approx_source_size(xc_approx,yc_approx,verbal);
	}
	return sig;
}


int SourcePixelGrid::assign_indices_and_count_levels()
{
	levels=1; // we are going to recount the number of levels
	int source_pixel_i=0;
	assign_indices(source_pixel_i);
	return source_pixel_i;
}

void SourcePixel::assign_indices(int& source_pixel_i)
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
void SourcePixel::print_indices()
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

//bool SourcePixelGrid::regrid_if_unmapped_source_subcells;
//bool SourcePixelGrid::activate_unmapped_source_pixels;
//bool SourcePixelGrid::exclude_source_pixels_outside_fit_window;

int SourcePixelGrid::assign_active_indices_and_count_source_pixels(bool regrid_if_inactive_cells, bool activate_unmapped_pixels, bool exclude_pixels_outside_window)
{
	parent_grid->regrid_if_unmapped_source_subcells = regrid_if_inactive_cells;
	parent_grid->activate_unmapped_source_pixels = activate_unmapped_pixels;
	parent_grid->exclude_source_pixels_outside_fit_window = exclude_pixels_outside_window;
	int source_pixel_i=0;
	assign_active_indices(source_pixel_i);
	return source_pixel_i;
}

void SourcePixel::assign_active_indices(int& source_pixel_i)
{
	int i, j;
	bool unsplit_cell = false;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->assign_active_indices(source_pixel_i);
			else {
					//cell[i][j]->active_index = source_pixel_i++;
					//cell[i][j]->active_pixel = true;
				if (cell[i][j]->maps_to_image_pixel) {
					cell[i][j]->active_index = source_pixel_i++;
					cell[i][j]->active_pixel = true;
				} else {
					if ((lens->mpi_id==0) and (lens->regularization_method == 0)) warn(lens->warnings,"A source pixel does not map to any image pixel (for source pixel %i,%i), level %i, center (%g,%g)",i,j,cell[i][j]->level,cell[i][j]->center_pt[0],cell[i][j]->center_pt[1]); // only show warning if no regularization being used, since matrix cannot be inverted in that case
					if ((parent_grid->activate_unmapped_source_pixels) and ((!parent_grid->regrid_if_unmapped_source_subcells) or (level==0))) { // if we are removing unmapped subpixels, we may still want to activate first-level unmapped pixels
						if ((parent_grid->exclude_source_pixels_outside_fit_window) and (cell[i][j]->maps_to_image_window==false)) ;
						else {
							cell[i][j]->active_index = source_pixel_i++;
							cell[i][j]->active_pixel = true;
						}
					} else {
						cell[i][j]->active_pixel = false;
						if ((parent_grid->regrid_if_unmapped_source_subcells) and (level >= 1)) {
							if (!parent_grid->regrid) parent_grid->regrid = true;
							unsplit_cell = true;
						}
					}
					//if ((exclude_source_pixels_outside_fit_window) and (cell[i][j]->maps_to_image_window==false)) {
						//if (cell[i][j]->active_pixel==true) {
							//source_pixel_i--;
							//if (!regrid) regrid = true;
							//cell[i][j]->active_pixel = false;
						//}
					//}
				}
			}
		}
	}
	if (unsplit_cell) unsplit();
}

SourcePixel::~SourcePixel()
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

SourcePixelGrid::~SourcePixelGrid()
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



void SourcePixel::clear()
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

void SourcePixel::clear_subgrids()
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

void DelaunayGrid::allocate_multithreaded_variables(const int& threads, const bool reallocate)
{
	if ((nthreads != 0) and (interpolation_pts[0] != NULL)) {
		if (!reallocate) return;
		else deallocate_multithreaded_variables();
	}
	nthreads = threads;
	int i;
	for (i=0; i < nmax_pts_interp; i++) {
		interpolation_pts[i] = new lensvector*[nthreads];
		interpolation_wgts[i] = new double[nthreads];
		interpolation_indx[i] = new int[nthreads];
		triangles_in_envelope[i] = new int[nthreads];
		new_circumcenter[i] = new lensvector[nthreads];
	}
	for (i=0; i < nmax_pts_interp+2; i++) {
		polygon_vertices[i] = new lensvector*[nthreads];
	}
}

void DelaunayGrid::deallocate_multithreaded_variables()
{
	if (interpolation_pts[0] != NULL) {
		int i;
		for (i=0; i < nmax_pts_interp; i++) {
			delete[] interpolation_pts[i];
			delete[] interpolation_wgts[i];
			delete[] interpolation_indx[i];
			delete[] triangles_in_envelope[i];
			delete[] new_circumcenter[i];
			interpolation_pts[i] = NULL;
			interpolation_wgts[i] = NULL;
			interpolation_indx[i] = NULL;
		}
		for (i=0; i < nmax_pts_interp+2; i++) {
			delete[] polygon_vertices[i];
		}
	}
}

DelaunayGrid::DelaunayGrid(QLens* lens_in, const int redshift_indx, double* srcpts_x, double* srcpts_y, const int n_srcpts_in, int *ivals_in, int *jvals_in, const int ni, const int nj, const bool find_pixel_magnification) : lens(lens_in)
{
	int threads = 1;
#ifdef USE_OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		threads = omp_get_num_threads();
	}
#endif
	//herg = 0;
	//allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)
	lens = lens_in;
	if ((redshift_indx < 0) and (lens->image_pixel_grid0 != NULL)) image_pixel_grid = lens->image_pixel_grid0;
	else if ((lens != NULL) and (lens->image_pixel_grids[redshift_indx] != NULL)) image_pixel_grid = lens->image_pixel_grids[redshift_indx];
	else image_pixel_grid = NULL;
	if (lens != NULL) {
		// This is mainly for plotting purposes
		if (image_pixel_grid != NULL) {
			srcgrid_xmin = image_pixel_grid->src_xmin;
			srcgrid_xmax = image_pixel_grid->src_xmax;
			srcgrid_ymin = image_pixel_grid->src_ymin;
			srcgrid_ymax = image_pixel_grid->src_ymax;
		} else {
			srcgrid_xmin = lens->sourcegrid_xmin;
			srcgrid_xmax = lens->sourcegrid_xmax;
			srcgrid_ymin = lens->sourcegrid_ymin;
			srcgrid_ymax = lens->sourcegrid_ymax;
		}
	}

	n_srcpts = n_srcpts_in;
	srcpts = new lensvector[n_srcpts];
	surface_brightness = new double[n_srcpts];
	maps_to_image_pixel = new bool[n_srcpts];
	active_pixel = new bool[n_srcpts];
	active_index = new int[n_srcpts];
	if (ivals_in != NULL) {
		imggrid_ivals = new int[n_srcpts];
		imggrid_jvals = new int[n_srcpts];
	}
	for (int i=0; i < 4; i++) adj_triangles[i] = new int[n_srcpts];
	int n;
	srcpixel_xmin=srcpixel_ymin=1e30;
	srcpixel_xmax=srcpixel_ymax=-1e30;
	for (n=0; n < n_srcpts; n++) {
		srcpts[n][0] = srcpts_x[n];
		srcpts[n][1] = srcpts_y[n];
		//cout << "Sourcept " << n << ": " << srcpts_x[n] << " " << srcpts_y[n] << endl;
		if (srcpts_x[n] > srcpixel_xmax) srcpixel_xmax=srcpts_x[n];
		if (srcpts_y[n] > srcpixel_ymax) srcpixel_ymax=srcpts_y[n];
		if (srcpts_x[n] < srcpixel_xmin) srcpixel_xmin=srcpts_x[n];
		if (srcpts_y[n] < srcpixel_ymin) srcpixel_ymin=srcpts_y[n];
		surface_brightness[n] = 0;
		maps_to_image_pixel[n] = false;
		active_pixel[n] = true;
		active_index[n] = -1;
		if (ivals_in != NULL) {
			imggrid_ivals[n] = ivals_in[n];
			imggrid_jvals[n] = jvals_in[n];
		}
		adj_triangles[0][n] = -1; // +x direction
		adj_triangles[1][n] = -1; // -x direction
		adj_triangles[2][n] = -1; // +y direction
		adj_triangles[3][n] = -1; // -y direction
	}
	img_imin = 30000;
	img_jmin = 30000;
	img_imax = -30000;
	img_jmax = -30000;

	look_for_starting_point = true; // the "look_for_starting_point" feature for searching triangles doesn't work well with outside_sb_prior, so stashing it for now.
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
			if (imggrid_ivals[n] < img_imin) img_imin = imggrid_ivals[n];
			if (imggrid_ivals[n] > img_imax) img_imax = imggrid_ivals[n];
			if (imggrid_jvals[n] < img_jmin) img_jmin = imggrid_jvals[n];
			if (imggrid_jvals[n] > img_jmax) img_jmax = imggrid_jvals[n];
		}
	} else {
		img_index_ij = NULL;
	}

	vector<int>* shared_triangles_unsorted = new vector<int>[n_srcpts];
	n_shared_triangles = new int[n_srcpts];
	shared_triangles = new int*[n_srcpts];
	voronoi_boundary_x = new double*[n_srcpts];
	voronoi_boundary_y = new double*[n_srcpts];
	voronoi_area = new double[n_srcpts];
	voronoi_length = new double[n_srcpts];
	inv_magnification = new double[n_srcpts];

	Delaunay *delaunay_triangles = new Delaunay(srcpts_x, srcpts_y, n_srcpts);
	delaunay_triangles->Process();
	n_triangles = delaunay_triangles->TriNum();
	if (n_triangles==0) die("number of Delaunay triangles is zero; cannot construct Delaunay grid");
	triangle = new Triangle[n_triangles];
	delaunay_triangles->store_triangles(triangle);
	//cout << "THERE ARE " << n_triangles << " TRIANGLES " << endl;

	string srcpt_filename = "test_srcpts.dat";
	ofstream srcout; lens->open_output_file(srcout,srcpt_filename);
	for (int i=0; i < n_srcpts; i++) {
		srcout << srcpts[i][0] << " " << srcpts[i][1] << endl;
	}

	string delaunay_filename = "test_delaunay.dat";
	ofstream delout; lens->open_output_file(delout,delaunay_filename);
	for (int i=0; i < n_triangles; i++) {
		delout << triangle[i].vertex[0][0] << " " << triangle[i].vertex[0][1] << endl;
		delout << triangle[i].vertex[1][0] << " " << triangle[i].vertex[1][1] << endl;
		delout << triangle[i].vertex[2][0] << " " << triangle[i].vertex[2][1] << endl;
		delout << triangle[i].vertex[0][0] << " " << triangle[i].vertex[0][1] << endl;
		delout << endl;
	}


	avg_area = 0;
	for (n=0; n < n_triangles; n++) {
		shared_triangles_unsorted[triangle[n].vertex_index[0]].push_back(n);
		shared_triangles_unsorted[triangle[n].vertex_index[1]].push_back(n);
		shared_triangles_unsorted[triangle[n].vertex_index[2]].push_back(n);
		avg_area += triangle[n].area;
	}
	avg_area /= n_triangles;
	int n_boundary_pts;
	Triangle *triptr;
	int i;
	lensvector midpoint;
	//int** tricheck = new int*[n_srcpts];
	lensvector vec1,vec2;
	//double totarea = 0;
	for (n=0; n < n_srcpts; n++) {
		n_boundary_pts = shared_triangles_unsorted[n].size();
		voronoi_boundary_x[n] = new double[n_boundary_pts];
		voronoi_boundary_y[n] = new double[n_boundary_pts];
		shared_triangles[n] = new int[n_boundary_pts];
		//tricheck[n] = new int[n_boundary_pts];
		double *angles = new double[n_boundary_pts];
		double *midpt_angles = new double[n_boundary_pts];
		for (i=0; i < n_boundary_pts; i++) {
			shared_triangles[n][i] = shared_triangles_unsorted[n][i];
			//tricheck[n][i] = shared_triangles_unsorted[n][i];
			triptr = &triangle[shared_triangles_unsorted[n][i]];
			voronoi_boundary_x[n][i] = triptr->circumcenter[0];
			voronoi_boundary_y[n][i] = triptr->circumcenter[1];
			double comp1,comp2,angle;
			comp1 = voronoi_boundary_x[n][i] - srcpts[n][0];
			comp2 = voronoi_boundary_y[n][i] - srcpts[n][1];
			//cout << "COMPS: " << comp1 << " " << comp2 << endl;
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
			comp1 = midpoint[0] - srcpts[n][0];
			comp2 = midpoint[1] - srcpts[n][1];
			//cout << "COMPS: " << comp1 << " " << comp2 << endl;
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
			midpt_angles[i] = angle;
		}
		n_shared_triangles[n] = n_boundary_pts;
		//sort(n_boundary_pts,angles,voronoi_boundary_x[n],voronoi_boundary_y[n],shared_triangles[n]); // I don't think sorting by circumcenters will work well, because circumcenters may lie outside the triangles and orders might get reversed 
		sort(n_boundary_pts,angles,voronoi_boundary_x[n],voronoi_boundary_y[n]);
		sort(n_boundary_pts,midpt_angles,shared_triangles[n]);
		//delete[] tricheck[n];
		delete[] angles;
		delete[] midpt_angles;
		vec2[0] = voronoi_boundary_x[n][0] - srcpts[n][0];
		vec2[1] = voronoi_boundary_y[n][0] - srcpts[n][1];
		voronoi_area[n] = 0;
		for (i=0; i < n_boundary_pts-1; i++) {
			vec1 = vec2;
			vec2[0] = voronoi_boundary_x[n][i+1] - srcpts[n][0];
			vec2[1] = voronoi_boundary_y[n][i+1] - srcpts[n][1];
			voronoi_area[n] += abs((vec1[0]*vec2[1] - vec1[1]*vec2[0])/2);
		}
		//totarea += voronoi_length[n];
		voronoi_length[n] = sqrt(voronoi_area[n]);
	}



	//cout << "TOTAL AREA: " << totarea << endl;
	//delete[] tricheck;
	if (find_pixel_magnification) {
		if (image_pixel_grid->cartesian_srcgrid != NULL) find_pixel_magnifications();
	}

	delete[] shared_triangles_unsorted;
	delete delaunay_triangles;
}

void DelaunayGrid::find_pixel_magnifications()
{
	double area_weighted_invmag, overlap_area, total_overlap_area;
	lensvector pt1,pt2;
	int m,n;
	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		#pragma omp for private(n,m,pt1,pt2,area_weighted_invmag,overlap_area,total_overlap_area) schedule(static)
		for (n=0; n < n_srcpts; n++) {
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
					area_weighted_invmag += image_pixel_grid->cartesian_srcgrid->find_triangle_weighted_invmag(srcpts[n],pt1,pt2,overlap_area,thread);
					total_overlap_area += overlap_area;
				}
				//inv_magnification[n] = area_weighted_invmag /= voronoi_area[n];
				if ((total_overlap_area != 0) and (abs(total_overlap_area) >= (0.95*voronoi_area[n]))) { // the latter requirement is a hack to cover the bordering cells for whom the masked pixels don't completely cover (i.e. pixels outside the mask map to them)
					inv_magnification[n] = area_weighted_invmag /= total_overlap_area;
				}
				else inv_magnification[n] = 1.0;
				if (inv_magnification[n] > 1.0) inv_magnification[n] = 1.0;
				//cout << "srcpt " << n << ": " << inv_magnification[n] << endl;
			}
			//if (lens->cartesian_srcgrid != NULL) inv_magnification[n] = lens->cartesian_srcgrid->find_local_inverse_magnification_interpolate(srcpts[n],0);
			//else inv_magnification[n] = 1.0;
			//cout << "mag " << n << ": " << inv_magnification[n] << endl;
		}
	}
}

void DelaunayGrid::record_adjacent_triangles_xy()
{
	const double increment = 1e-5;
	int i,j;
	bool foundxp, foundyp, foundxm, foundym;
	double x, y, xp, xm, yp, ym;
	lensvector ptx_p, pty_p, ptx_m, pty_m;
	for (i=0; i < n_srcpts; i++) {
		foundxp = false;
		foundxm = false;
		foundyp = false;
		foundym = false;
		x = srcpts[i][0];
		y = srcpts[i][1];
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

int DelaunayGrid::search_grid(const int initial_srcpixel, const lensvector& pt, bool& inside_triangle)
{
	if (n_shared_triangles[initial_srcpixel]==0) die("something is really wrong! This vertex doesn't share any triangle sides (vertex %i, ntot=%i)",initial_srcpixel,n_srcpts);
	int n, triangle_num = shared_triangles[initial_srcpixel][0]; // there might be a better way to discern which shared triangle to start with, but we can optimize this later
	if ((pt[0]==srcpts[initial_srcpixel][0]) and (pt[1]==srcpts[initial_srcpixel][1])) {
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
bool DelaunayGrid::test_if_inside(int &tri_number, const lensvector& pt, bool& inside_triangle)
{
	// To speed things up, these things can be made static and have an array for each (so each thread uses a different set of static elements)
	lensvector dt1, dt2, dt3;
	double cross_prod;

	Triangle *triptr = &triangle[tri_number];
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

bool DelaunayGrid::test_if_inside(const int tri_number, const lensvector& pt)
{
	// To speed things up, these things can be made static and have an array for each (so each thread uses a different set of static elements)
	lensvector dt1, dt2, dt3;
	double cross_prod;

	//cout << "TRINUM=" << tri_number << endl;
	Triangle *triptr = &triangle[tri_number];
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

int DelaunayGrid::find_closest_vertex(const int tri_number, const lensvector& pt)
{
	// This function effectively allows us to plot the Voronoi cells, since the vertices of the triangles are the "seeds" of the Voronoi cells
	Triangle *triptr = &triangle[tri_number];
	Triangle *neighbor_ptr;
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
			neighbor_ptr = &triangle[triptr->neighbor_index[i]];
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

double DelaunayGrid::sum_edge_sqrlengths(const double min_sb)
{
	lensvector edge;
	double sum=0;
	bool use_sb = false;
	double** sb;
	int iv[3];
	int jv[3];
	if ((lens != NULL) and (lens->image_pixel_data != NULL)) {
		use_sb = true;
		sb = lens->image_pixel_data->surface_brightness;
	}
	int i,j;
	// Note, the inside edges (the majority) will be counted twice, but that's ok
	for (i=0; i < n_triangles; i++) {
		edge = triangle[i].vertex[1] - triangle[i].vertex[0];
		sum += edge.sqrnorm();
		edge = triangle[i].vertex[2] - triangle[i].vertex[1];
		sum += edge.sqrnorm();
		edge = triangle[i].vertex[0] - triangle[i].vertex[2];
		sum += edge.sqrnorm();
	}
	return sum;
}

void DelaunayGrid::assign_surface_brightness_from_analytic_source(const int zsrc_i)
{
	//cout << "Sourcepts: " << n_srcpts << endl;
	int i,k;
	for (i=0; i < n_srcpts; i++) {
		//cout << "Assigning SB point " << i << "..." << endl;
		surface_brightness[i] = 0;
		for (k=0; k < lens->n_sb; k++) {
			//cout << "source " << k << endl;
			if ((lens->sb_list[k]->is_lensed) and ((zsrc_i<0) or (lens->sbprofile_redshift_idx[k]==zsrc_i))) surface_brightness[i] += lens->sb_list[k]->surface_brightness(srcpts[i][0],srcpts[i][1]);
		}
	}
	for (i=0; i < n_triangles; i++) {
		triangle[i].sb[0] = &surface_brightness[triangle[i].vertex_index[0]];
		triangle[i].sb[1] = &surface_brightness[triangle[i].vertex_index[1]];
		triangle[i].sb[2] = &surface_brightness[triangle[i].vertex_index[2]];
	}
}

void DelaunayGrid::fill_surface_brightness_vector()
{
	int i,j;
	for (i=0, j=0; i < n_srcpts; i++) {
		if (active_pixel[i]) {
			lens->source_pixel_vector[j++] = surface_brightness[i];
		}
	}
}

void DelaunayGrid::update_surface_brightness(int& index)
{
	int i;
	for (i=0; i < n_srcpts; i++) {
		if (active_pixel[i]) {
			surface_brightness[i] = lens->source_pixel_vector[index++];
		} else {
			surface_brightness[i] = 0;
		}
	}
	for (i=0; i < n_triangles; i++) {
		triangle[i].sb[0] = &surface_brightness[triangle[i].vertex_index[0]];
		triangle[i].sb[1] = &surface_brightness[triangle[i].vertex_index[1]];
		triangle[i].sb[2] = &surface_brightness[triangle[i].vertex_index[2]];
	}
}

bool DelaunayGrid::assign_source_mapping_flags(lensvector &input_pt, vector<PtsWgts>& mapped_delaunay_srcpixels_ij, int& n_mapped_srcpixels, const int img_pixel_i, const int img_pixel_j, const int thread, bool& trouble_with_starting_vertex)
{
	int trinum,kmin;
	bool inside_triangle, on_vertex;
	if (!find_containing_triangle(input_pt,img_pixel_i,img_pixel_j,trinum,inside_triangle,on_vertex,kmin)) trouble_with_starting_vertex = true;
	Triangle *triptr = &triangle[trinum];

	if (!inside_triangle) {
		// we don't want to extrapolate, because it can lead to crazy results outside the grid. so we find the closest vertex and use that vertex's SB
		if ((zero_outside_border) and (!on_vertex)) {
			n_mapped_srcpixels = 0;
			return true;
		}
		PtsWgts pt(triptr->vertex_index[kmin],1);
		mapped_delaunay_srcpixels_ij.push_back(pt);
		maps_to_image_pixel[triptr->vertex_index[kmin]] = true;
		n_mapped_srcpixels = 1;
	} else {
		if (lens->natural_neighbor_interpolation) {
			find_interpolation_weights_nn(input_pt, trinum, n_mapped_srcpixels, thread);
		} else {
			find_interpolation_weights_3pt(input_pt, trinum, n_mapped_srcpixels, thread);
		}
		PtsWgts pt;
		for (int i=0; i < n_mapped_srcpixels; i++) {
			maps_to_image_pixel[interpolation_indx[i][thread]] = true;
			mapped_delaunay_srcpixels_ij.push_back(pt.assign(interpolation_indx[i][thread],interpolation_wgts[i][thread]));
		}
	}
	return true;
}

void DelaunayGrid::calculate_Lmatrix(const int img_index, PtsWgts* mapped_delaunay_srcpixels, int* n_mapped_subpixels, int& index, lensvector &input_pt, const int& subpixel_indx, const double weight, const int& thread)
{
	int i;
	for (i=0; i < subpixel_indx; i++) mapped_delaunay_srcpixels += (*n_mapped_subpixels++);
	for (i=0; i < (*n_mapped_subpixels); i++) {
		lens->Lmatrix_index_rows[img_index].push_back(mapped_delaunay_srcpixels->indx);
		lens->Lmatrix_rows[img_index].push_back(weight*mapped_delaunay_srcpixels->wgt);
		mapped_delaunay_srcpixels++;
	}
	index += (*n_mapped_subpixels);
}

double DelaunayGrid::find_lensed_surface_brightness(lensvector &input_pt, const int img_pixel_i, const int img_pixel_j, const int thread)
{
	int trinum;
	bool inside_triangle;
	bool on_vertex = false;
	int kmin;
	find_containing_triangle(input_pt,img_pixel_i,img_pixel_j,trinum,inside_triangle,on_vertex,kmin);
	Triangle *triptr = &triangle[trinum];

	if (!inside_triangle) {
		// we don't want to extrapolate, because it can lead to crazy results outside the grid. so we find the closest vertex and use that vertex's SB
		if ((zero_outside_border) and (!on_vertex)) {
			return 0;
		}
		return *triptr->sb[kmin];
	}
	int npts;
	double sb_interp = 0;
	if (lens->natural_neighbor_interpolation) {
		find_interpolation_weights_nn(input_pt, trinum, npts, thread);
	} else {
		find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
	}
	for (int i=0; i < npts; i++) {
		sb_interp += surface_brightness[interpolation_indx[i][thread]]*interpolation_wgts[i][thread];
	}
	return sb_interp;
}

double DelaunayGrid::interpolate_surface_brightness(lensvector &input_pt, const bool interp_mag, const int thread)
{
	bool inside_triangle;
	bool on_vertex;
	int trinum,kmin;
	find_containing_triangle(input_pt,trinum,inside_triangle,on_vertex,kmin);
	if (!inside_triangle) {
		if ((zero_outside_border) and (!on_vertex)) {
			return 0;
		} else {
			return *triangle[trinum].sb[kmin];
		}
	}

	int npts;
	double interp_val = 0;
	if (lens->natural_neighbor_interpolation) {
		find_interpolation_weights_nn(input_pt, trinum, npts, thread);
	} else {
		find_interpolation_weights_3pt(input_pt, trinum, npts, thread);
	}
	if (interp_mag) {
		for (int i=0; i < npts; i++) {
			interp_val += inv_magnification[interpolation_indx[i][thread]]*interpolation_wgts[i][thread];
		}
		interp_val = 1.0/interp_val; // this produces the magnification (rather than inverse mag)
	} else {
		for (int i=0; i < npts; i++) {
			interp_val += surface_brightness[interpolation_indx[i][thread]]*interpolation_wgts[i][thread];
		}
	}
	return interp_val;
}

void DelaunayGrid::find_interpolation_weights_3pt(lensvector& input_pt, const int trinum, int& npts, const int thread)
{
	Triangle *triptr = &triangle[trinum];
	interpolation_indx[0][thread] = triptr->vertex_index[0];
	interpolation_indx[1][thread] = triptr->vertex_index[1];
	interpolation_indx[2][thread] = triptr->vertex_index[2];
	interpolation_pts[0][thread] = &srcpts[triptr->vertex_index[0]];
	interpolation_pts[1][thread] = &srcpts[triptr->vertex_index[1]];
	interpolation_pts[2][thread] = &srcpts[triptr->vertex_index[2]];
	double d = ((*interpolation_pts[0][thread])[0]-(*interpolation_pts[1][thread])[0])*((*interpolation_pts[1][thread])[1]-(*interpolation_pts[2][thread])[1]) - ((*interpolation_pts[1][thread])[0]-(*interpolation_pts[2][thread])[0])*((*interpolation_pts[0][thread])[1]-(*interpolation_pts[1][thread])[1]);
	if (d==0) {
		// in this case the points are all the same
		interpolation_wgts[0][thread] = 1.0;
		npts = 1;
	} else {
		interpolation_wgts[0][thread] = (input_pt[0]*((*interpolation_pts[1][thread])[1]-(*interpolation_pts[2][thread])[1]) + input_pt[1]*((*interpolation_pts[2][thread])[0]-(*interpolation_pts[1][thread])[0]) + (*interpolation_pts[1][thread])[0]*(*interpolation_pts[2][thread])[1] - (*interpolation_pts[1][thread])[1]*(*interpolation_pts[2][thread])[0])/d;
		interpolation_wgts[1][thread] = (input_pt[0]*((*interpolation_pts[2][thread])[1]-(*interpolation_pts[0][thread])[1]) + input_pt[1]*((*interpolation_pts[0][thread])[0]-(*interpolation_pts[2][thread])[0]) + (*interpolation_pts[0][thread])[1]*(*interpolation_pts[2][thread])[0] - (*interpolation_pts[0][thread])[0]*(*interpolation_pts[2][thread])[1])/d;
		interpolation_wgts[2][thread] = (input_pt[0]*((*interpolation_pts[0][thread])[1]-(*interpolation_pts[1][thread])[1]) + input_pt[1]*((*interpolation_pts[1][thread])[0]-(*interpolation_pts[0][thread])[0]) + (*interpolation_pts[0][thread])[0]*(*interpolation_pts[1][thread])[1] - (*interpolation_pts[0][thread])[1]*(*interpolation_pts[1][thread])[0])/d;
		npts = 3;
	}
}

void DelaunayGrid::find_interpolation_weights_nn(lensvector &input_pt, const int trinum, int& npts, const int thread) // natural neighbor interpolation
{
	npts = 0;
	const int nmax_tri = 60;
	Triangle* adjacent_triangles[nmax_tri];
	int n_adjacent_triangles;

	double area_initial, area_leftover, wgt;
	int ntri_in_envelope = 1;
	Triangle *triptr = &triangle[trinum];
	triangles_in_envelope[0][thread] = trinum;
	int k,l,m;

	// recursive lambda function for finding triangles that belong inside the Bowyer-Watson envelope (this will be called below)
	function<void(Triangle*, const int, const int, int &, int &)> find_triangles_in_envelope = [&](Triangle *neighbor_ptr, const int trinum, const int neighbor_num, int &npt, int &ntri) 
	{
		int idx;
		Triangle *neighbor_ptr2;
		int l, l_new_vertex, l_left, l_right, neighbor_num2;
		double distsq;
		for (l=0; l < 3; l++) {
			if (neighbor_ptr->neighbor_index[l]==trinum) {
				l_new_vertex = l;
				break;
			}
		}
		l_left = l_new_vertex-1;
		if (l_left==-1) l_left = 2;
		l_right = l_new_vertex+1;
		if (l_right==3) l_right = 0;
		neighbor_num2 = neighbor_ptr->neighbor_index[l_right];
		if (neighbor_num2 != -1) {
			neighbor_ptr2 = &triangle[neighbor_num2];
			distsq = SQR(input_pt[0]-neighbor_ptr2->circumcenter[0]) + SQR(input_pt[1]-neighbor_ptr2->circumcenter[1]);
			if (distsq < neighbor_ptr2->circumcircle_radsq) {
				find_triangles_in_envelope(neighbor_ptr2,neighbor_num,neighbor_num2,npt,ntri);
			}
		}
		triangles_in_envelope[ntri++][thread] = neighbor_num;
		interpolation_indx[npt][thread] = neighbor_ptr->vertex_index[l_new_vertex];
		npt++;
		if (npt > nmax_pts_interp) die("exceeded max number of points (%i versus %i)",npt,nmax_pts_interp);
		neighbor_num2 = neighbor_ptr->neighbor_index[l_left];
		if (neighbor_num2 != -1) {
			neighbor_ptr2 = &triangle[neighbor_num2];
			distsq = SQR(input_pt[0]-neighbor_ptr2->circumcenter[0]) + SQR(input_pt[1]-neighbor_ptr2->circumcenter[1]);
			if (distsq < neighbor_ptr2->circumcircle_radsq) {
				find_triangles_in_envelope(neighbor_ptr2,neighbor_num,neighbor_num2,npt,ntri);
			}
		}
	};

	Triangle *neighbor_ptr;
	double distsq;
	int kleft,neighbor_num;
	for (k=0; k < 3; k++) {
		interpolation_indx[npts][thread] = triptr->vertex_index[k];
		npts++;
		if (npts > nmax_pts_interp) die("exceeded max number of points");
		kleft = k-1;
		if (kleft==-1) kleft = 2;
		neighbor_num = triptr->neighbor_index[kleft];
		if (neighbor_num != -1) {
			neighbor_ptr = &triangle[neighbor_num];
			distsq = SQR(input_pt[0]-neighbor_ptr->circumcenter[0]) + SQR(input_pt[1]-neighbor_ptr->circumcenter[1]);
			if (distsq < neighbor_ptr->circumcircle_radsq) {
				find_triangles_in_envelope(neighbor_ptr,trinum,neighbor_num,npts,ntri_in_envelope);
			}
		}
	}

	/*
	if (herg==301) {
		for (k=0; k < npts; k++) {
			idx = interpolation_indx[k][thread];
			//sb[k] = &surface_brightness[interpolation_indx[k][thread]];
			interpolation_pts[k][thread] = &srcpts[idx];
		}
	
		cout << "# npts=" << npts << endl;
		cout << "# srcpts=" << n_srcpts << endl;
		for (k=0; k < npts; k++)  {
			cout << "Point "  << k << ": " << interpolation_indx[k][thread] << endl;
		}
		ofstream triout("inttri.dat");
		for (m=0; m < ntri_in_envelope; m++) {
			neighbor_ptr = &triangle[triangles_in_envelope[m][thread]];
			for (k=0; k < 3; k++) {
				triout << neighbor_ptr->vertex[k][0] << " " << neighbor_ptr->vertex[k][1] << endl;
			}
			triout << neighbor_ptr->vertex[0][0] << " " << neighbor_ptr->vertex[0][1] << endl << endl;
		}
		ofstream envout("intenv.dat");
		envout << "# npts=" << npts << endl;
		for (k=0; k < npts; k++)  {
			envout << (*interpolation_pts[k][thread])[0] << " " << (*interpolation_pts[k][thread])[1] << " k=" << k << endl;
		}
		envout << (*interpolation_pts[0][thread])[0] << " " << (*interpolation_pts[0][thread])[1] << " k=" << k << endl;

		ofstream newccircout("newccircs.dat");
		for (k=0; k < npts; k++) {
			newccircout << new_circumcenter[k][0] << " " << new_circumcenter[k][1] << endl;
		}

		ofstream ccircout("ccircs.dat");
		ccircout << triptr->circumcenter[0] << " " << triptr->circumcenter[1] << endl;
		for (k=0; k < ntri_in_envelope; k++) {
				ccircout << triangle[triangles_in_envelope[k][thread]].circumcenter[0] << " " << triangle[triangles_in_envelope[k][thread]].circumcenter[1] << endl;
		}
	}
	*/



	/*
	int tricheck=0;
	for (k=0; k < n_triangles; k++) {
		neighbor_ptr = &triangle[k];
		distsq = SQR(input_pt[0]-neighbor_ptr->circumcenter[0]) + SQR(input_pt[1]-neighbor_ptr->circumcenter[1]);
		if (distsq < neighbor_ptr->circumcircle_radsq) {
			tricheck++;
			//cout << "WITHIN CIRCUMCIRCLE FOR TRIANGLE " << k << endl;
		}
		//cout << endl;
	}
	//cout << "Within circumcenter of " << tricheck << " triangles" << endl;
	if (tricheck != ntri_in_envelope) {
		cout << "WARNING! number of triangles in envelope (" << ntri_in_envelope << ") does not match number triangles whose circumcircles enclose the interpolating point (" << tricheck << ")" << endl;
		die();
	}
	//else cout << "YAY! number of triangles in envelope (" << ntri_in_envelope << ") matched # triangles whose circumcircles enclose interpolating point (" << tricheck << ")" << endl;
		//}
*/

	int idx,kright;
	double a0,a1,c0,c1,det_inv,asq,csq,ctr0,ctr1;
	for (k=0; k < npts; k++) {
		idx = interpolation_indx[k][thread];
		interpolation_pts[k][thread] = &srcpts[idx];
	}
	for (k=0; k < npts; k++) {
		kright = k+1;
		if (kright==npts) kright = 0;
		a0 = (*interpolation_pts[k][thread])[0]-input_pt[0];
		a1 = (*interpolation_pts[k][thread])[1]-input_pt[1];
		c0 = (*interpolation_pts[kright][thread])[0]-input_pt[0];
		c1 = (*interpolation_pts[kright][thread])[1]-input_pt[1];
		det_inv = 0.5/(a0*c1-c0*a1);
		asq = a0*a0 + a1*a1;
		csq = c0*c0 + c1*c1;
		ctr0 = det_inv*(asq*c1 - csq*a1);
		ctr1 = det_inv*(csq*a0 - asq*c0);
		new_circumcenter[k][thread][0] = ctr0 + input_pt[0];
		new_circumcenter[k][thread][1] = ctr1 + input_pt[1];
	}

	//cout << "Triangle circumcenters:" << endl;
	//for (k=0; k < ntri_in_envelope; k++) {
		//cout << triangle[triangles_in_envelope[k][thread]].circumcenter[0] << " " << triangle[triangles_in_envelope[k][thread]].circumcenter[1] << endl;
	//}
	//cout << endl;

	lensvector midpt_left, midpt_right;
	double totwgt = 0;
	bool first_iteration, fix_mmin, fix_mmax;
	bool mmin_in_envelope, mmax_in_envelope, mmin_in_envelope_prev, mmax_in_envelope_prev;
	int n_polygon_vertices, shared_tri_idx_min, shared_tri_idx_max, mmin_adjacent, mmax_adjacent, mmax, iter;
	for (k=0; k < npts; k++) {
		iter = 0;
		idx = interpolation_indx[k][thread];
		//sb[k] = &surface_brightness[interpolation_indx[k][thread]];
		//interpolation_pts[k][thread] = &srcpts[idx];
		//cout << "pp" << k << ": idx=" << idx << " " << (*interpolation_pts[k][thread])[0] << " " << (*interpolation_pts[k][thread])[1] << endl;
		//cout << "Finding shared triangles for (" << (*interpolation_pts[k])[0] << "," << (*interpolation_pts[k])[1] << ")" << endl;
		mmin_adjacent = 0;
		mmax_adjacent = n_shared_triangles[idx]-1;
		first_iteration = true;
		mmin_in_envelope_prev = false;
		mmax_in_envelope_prev = false;
		fix_mmin = false;
		fix_mmax = false;

		//for (m=0; m < ntri_in_envelope; m++) {
			//if (shared_tri_idx_min==triangles_in_envelope[m][thread]) mmin_in_envelope = true;
			//if (shared_tri_idx_max==triangles_in_envelope[m][thread]) mmax_in_envelope = true;
		//}

		iter = 0;
		//if (herg==301) cout << "Point " << k << ":" << endl;
		do {
			mmin_in_envelope = false;
			mmax_in_envelope = false;
			shared_tri_idx_min = shared_triangles[idx][mmin_adjacent];
			shared_tri_idx_max = shared_triangles[idx][mmax_adjacent];
			for (m=0; m < ntri_in_envelope; m++) {
				if (shared_tri_idx_min==triangles_in_envelope[m][thread]) mmin_in_envelope = true;
				if (shared_tri_idx_max==triangles_in_envelope[m][thread]) mmax_in_envelope = true;
			}
			//if (herg==301) cout << "iter=" << iter << ": mmin=" << mmin_adjacent << " mmax=" << mmax_adjacent << " min_in_env=" << mmin_in_envelope << " max_in_env=" << mmax_in_envelope << " nshared=" << n_shared_triangles[idx] << endl;
			if ((mmin_adjacent >= mmax_adjacent) and ((!mmin_in_envelope) and (!mmax_in_envelope) and (!mmin_in_envelope_prev) and (!mmax_in_envelope_prev))) die("shit, there are no shared triangles in envelope!");
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
		//if (herg==301) cout << "final iter=" << iter << ": mmin=" << mmin_adjacent << " mmax=" << mmax_adjacent << " nshared=" << n_shared_triangles[idx] << endl << endl;
		if (mmax_adjacent >= mmin_adjacent) n_adjacent_triangles = mmax_adjacent-mmin_adjacent+1;
		else {
			n_adjacent_triangles = n_shared_triangles[idx]-mmin_adjacent+mmax_adjacent+1;
		}
		if (n_adjacent_triangles > nmax_tri) die("number of adjacent triangles exceeded maximum allowed number (%i vs %i)",n_adjacent_triangles,nmax_tri);
		//cout << "Found " << n_adjacent_triangles << " adjacent triangles" << endl;
		//adjacent_triangles[k] = new Triangle*[n_adjacent_triangles];
		mmax = (mmax_adjacent >= mmin_adjacent) ? mmax_adjacent : n_shared_triangles[idx]-1;
		l=0;
		for (m=mmin_adjacent; m <= mmax; m++) {
			adjacent_triangles[l++] = &triangle[shared_triangles[idx][m]];
		}
		if (mmax_adjacent < mmin_adjacent) {
			for (m=0; m <= mmax_adjacent; m++) {
				adjacent_triangles[l++] = &triangle[shared_triangles[idx][m]];
			}
		}
		if (l != n_adjacent_triangles) die("number of adjacent triangles didn't add up right (l=%i)",l);

		kleft = k-1;
		kright = k+1;
		if (kleft==-1) kleft = npts-1;
		if (kright==npts) kright = 0;
		midpt_left = ((*interpolation_pts[k][thread]) + (*interpolation_pts[kleft][thread]))/2;
		midpt_right = ((*interpolation_pts[kright][thread]) + (*interpolation_pts[k][thread]))/2;
		n_polygon_vertices = n_adjacent_triangles+2;
		polygon_vertices[0][thread] = &midpt_left;
		polygon_vertices[n_polygon_vertices-1][thread] = &midpt_right;
		l=0;
		for (m=n_adjacent_triangles-1; m >= 0; m--) {
			polygon_vertices[l+1][thread] = &(adjacent_triangles[m]->circumcenter);
			l++;
		}

		area_initial = 0;
		for (m=0; m < n_polygon_vertices-1; m++) {
			area_initial += (*polygon_vertices[m][thread])[0]*(*polygon_vertices[m+1][thread])[1] - (*polygon_vertices[m+1][thread])[0]*(*polygon_vertices[m][thread])[1];
		}
		area_initial *= -0.5;
		/*
		if (herg==301) {
		stringstream pstr;
		pstr << k;
		string pstring;
		pstr >> pstring;
		ofstream pgonout(("pgon" + pstring + ".dat").c_str());
		//cout << "Polygon vertices for point " << k << ": area0=" << area_initial << " area_f=" << area_leftover << " area_dif=" << wgt << endl;
		for (m=0; m < n_polygon_vertices; m++) {
			pgonout << (*polygon_vertices[m][thread])[0] << " " << (*polygon_vertices[m][thread])[1] << endl;
		}
		*/

		//Now we construct the polygons generated by including the new input point in the grid, creating six new Delaunay triangles
		n_polygon_vertices = 4;
		polygon_vertices[1][thread] = &new_circumcenter[kleft][thread];
		polygon_vertices[2][thread] = &new_circumcenter[k][thread];
		polygon_vertices[3][thread] = &midpt_right;

		area_leftover = 0;
		for (m=0; m < n_polygon_vertices-1; m++) {
			area_leftover += (*polygon_vertices[m][thread])[0]*(*polygon_vertices[m+1][thread])[1] - (*polygon_vertices[m+1][thread])[0]*(*polygon_vertices[m][thread])[1];
		}
		area_leftover *= -0.5;
		wgt = abs(area_initial - area_leftover);

		interpolation_wgts[k][thread] = wgt;
		totwgt += wgt;
	}
	for (k=0; k < npts; k++) {
		interpolation_wgts[k][thread] /= totwgt;
	}

	//if ((hergerr > 0) and (herg==hergerr)) {
	/*
	if (herg==301) {
	double sb_interp = 0;
	for (k=0; k < npts; k++) {
		sb_interp += surface_brightness[interpolation_indx[k][thread]]*interpolation_wgts[k][thread];
	}
	cout << "SB_INTERP: " << sb_interp << endl;
	cout << "SBVALS: " << endl;
	for (k=0; k < npts; k++) {
		cout << (*interpolation_pts[k][thread])[0] << " " << (*interpolation_pts[k][thread])[1] << " " << surface_brightness[interpolation_indx[k][thread]] << " " << interpolation_wgts[k][thread] << endl;
	}
	cout << endl;
	ofstream ptout("intpt.dat");
	ofstream tri0out("inttri0.dat");
	ofstream tri1out("inttri1.dat");
	ofstream tri2out("inttri2.dat");
	ofstream circout("circs.dat");
	ofstream newcircout("newcircs.dat");
	ofstream vout("vcells.dat");
	ofstream nvout("nvcells.dat");
	ofstream ntout("newtri.dat");
	ptout << input_pt[0] << " " << input_pt[1] << endl << endl;
	ptout.close();

	ofstream triout("inttri.dat");
	for (m=0; m < ntri_in_envelope; m++) {
		neighbor_ptr = &triangle[triangles_in_envelope[m][thread]];
		for (k=0; k < 3; k++) {
			triout << neighbor_ptr->vertex[k][0] << " " << neighbor_ptr->vertex[k][1] << endl;
		}
		triout << neighbor_ptr->vertex[0][0] << " " << neighbor_ptr->vertex[0][1] << endl << endl;
	}
	ofstream envout("intenv.dat");
	for (k=0; k < npts; k++) 
		envout << (*interpolation_pts[k][thread])[0] << " " << (*interpolation_pts[k][thread])[1] << endl;
	envout << (*interpolation_pts[0][thread])[0] << " " << (*interpolation_pts[0][thread])[1] << endl;

	ofstream newccircout("newccircs.dat");
	for (k=0; k < npts; k++) {
		newccircout << new_circumcenter[k][thread][0] << " " << new_circumcenter[k][thread][1] << endl;
	}

	//for (m=0; m < n_srcpts; m++) {
	for (k=0; k < npts; k++) {
		m = interpolation_indx[k][thread];
		for (int j=0; j < n_shared_triangles[m]; j++) {
			vout << voronoi_boundary_x[m][j] << " " << voronoi_boundary_y[m][j] << endl;
		}
		vout << voronoi_boundary_x[m][0] << " " << voronoi_boundary_y[m][0] << endl << endl;
	}
	double xc, yc, r;
	int i, n_circpts = 600;
	double tstep = M_2PI/(n_circpts-1);
	double x,y,t;

		xc = triptr->circumcenter[0];
		yc = triptr->circumcenter[1];
		r = sqrt(SQR(triptr->vertex[0][0] - xc) + SQR(triptr->vertex[0][1] - yc));
		for (i=0, t=0; i < n_circpts; i++, t += tstep) {
			x = xc + r*cos(t);
			y = yc + r*sin(t);
			circout << x << " " << y << endl;
		}
		circout << endl;

	for (k=0; k < 3; k++) {
		if (triptr->neighbor_index[k] != -1) {
			xc = triangle[triptr->neighbor_index[k]].circumcenter[0];
			yc = triangle[triptr->neighbor_index[k]].circumcenter[1];
			r = sqrt(SQR(triangle[triptr->neighbor_index[k]].vertex[0][0] - xc) + SQR(triangle[triptr->neighbor_index[k]].vertex[0][1] - yc));
			for (i=0, t=0; i < n_circpts; i++, t += tstep) {
				x = xc + r*cos(t);
				y = yc + r*sin(t);
				circout << x << " " << y << endl;
			}
			circout << endl;
		}
	}

	for (k=0; k < npts; k++) {
			xc = new_circumcenter[k][thread][0];
			yc = new_circumcenter[k][thread][1];
			r = sqrt(SQR(input_pt[0] - xc) + SQR(input_pt[1] - yc));
			for (i=0, t=0; i < n_circpts; i++, t += tstep) {
				x = xc + r*cos(t);
				y = yc + r*sin(t);
				newcircout << x << " " << y << endl;
			}
			newcircout << endl;
	}


	if (triptr->neighbor_index[0] != -1) {
		for (k=0; k < 3; k++) {
			tri0out << triangle[triptr->neighbor_index[0]].vertex[k][0] << " " << triangle[triptr->neighbor_index[0]].vertex[k][1] << endl;
		}
		tri0out << triangle[triptr->neighbor_index[0]].vertex[0][0] << " " << triangle[triptr->neighbor_index[0]].vertex[0][1] << endl;
	}

	if (triptr->neighbor_index[1] != -1) {
		for (k=0; k < 3; k++) {
			tri1out << triangle[triptr->neighbor_index[1]].vertex[k][0] << " " << triangle[triptr->neighbor_index[1]].vertex[k][1] << endl;
		}
		tri1out << triangle[triptr->neighbor_index[1]].vertex[0][0] << " " << triangle[triptr->neighbor_index[1]].vertex[0][1] << endl;
	}

	if (triptr->neighbor_index[2] != -1) {
		for (k=0; k < 3; k++) {
			tri2out << triangle[triptr->neighbor_index[2]].vertex[k][0] << " " << triangle[triptr->neighbor_index[2]].vertex[k][1] << endl;
		}
		tri2out << triangle[triptr->neighbor_index[2]].vertex[0][0] << " " << triangle[triptr->neighbor_index[2]].vertex[0][1] << endl;
	}

	vector<int>* shared_triangles_mod = new vector<int>[n_srcpts+1];
	double **voronoi_boundary_mod_x = new double*[n_srcpts+1];
	double **voronoi_boundary_mod_y = new double*[n_srcpts+1];

	double *srcpts_mod_x = new double[n_srcpts+1];
	double *srcpts_mod_y = new double[n_srcpts+1];
	for (i=0; i < n_srcpts; i++) {
		srcpts_mod_x[i] = srcpts[i][0];
		srcpts_mod_y[i] = srcpts[i][1];
	}
	srcpts_mod_x[n_srcpts] = input_pt[0];
	srcpts_mod_y[n_srcpts] = input_pt[1];
	Delaunay *delaunay_triangles_mod = new Delaunay(srcpts_mod_x, srcpts_mod_y, n_srcpts+1);
	delaunay_triangles_mod->Process();
	int n_triangles_mod = delaunay_triangles_mod->TriNum();
	Triangle *triangle_mod = new Triangle[n_triangles_mod];
	delaunay_triangles_mod->store_triangles(triangle_mod);
	for (int n=0; n < n_triangles_mod; n++) {
		shared_triangles_mod[triangle_mod[n].vertex_index[0]].push_back(n);
		shared_triangles_mod[triangle_mod[n].vertex_index[1]].push_back(n);
		shared_triangles_mod[triangle_mod[n].vertex_index[2]].push_back(n);
	}
	int n_boundary_mod_pts;
	for (int n=0; n < n_srcpts+1; n++) {
		n_boundary_mod_pts = shared_triangles_mod[n].size();
		voronoi_boundary_mod_x[n] = new double[n_boundary_mod_pts];
		voronoi_boundary_mod_y[n] = new double[n_boundary_mod_pts];
		double *angles = new double[n_boundary_mod_pts];
		for (i=0; i < n_boundary_mod_pts; i++) {
			triptr = &triangle_mod[shared_triangles_mod[n][i]];
			voronoi_boundary_mod_x[n][i] = triptr->circumcenter[0];
			voronoi_boundary_mod_y[n][i] = triptr->circumcenter[1];
			double comp1,comp2,angle;
			comp1 = voronoi_boundary_mod_x[n][i] - srcpts_mod_x[n];
			comp2 = voronoi_boundary_mod_y[n][i] - srcpts_mod_y[n];
			//cout << "COMPS: " << comp1 << " " << comp2 << endl;
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
		}
		sort(n_boundary_mod_pts,angles,voronoi_boundary_mod_x[n],voronoi_boundary_mod_y[n]);
		delete[] angles;
	}
	for (k=0; k < npts; k++) {
		m = interpolation_indx[k][thread];
	//for (m=0; m < n_srcpts; m++) {
		for (int j=0; j < shared_triangles_mod[m].size(); j++) {
			nvout << voronoi_boundary_mod_x[m][j] << " " << voronoi_boundary_mod_y[m][j] << endl;
		}
		nvout << voronoi_boundary_mod_x[m][0] << " " << voronoi_boundary_mod_y[m][0] << endl << endl;
	}
	m = n_srcpts;
	for (int j=0; j < shared_triangles_mod[m].size(); j++) {
		nvout << voronoi_boundary_mod_x[m][j] << " " << voronoi_boundary_mod_y[m][j] << endl;
	}
	nvout << voronoi_boundary_mod_x[m][0] << " " << voronoi_boundary_mod_y[m][0] << endl << endl;

	for (int j=0; j < shared_triangles_mod[m].size(); j++) {
		triptr = &triangle_mod[shared_triangles_mod[m][j]];
		ntout << triptr->vertex[0][0] << " " << triptr->vertex[0][1] << endl;
		ntout << triptr->vertex[1][0] << " " << triptr->vertex[1][1] << endl;
		ntout << triptr->vertex[2][0] << " " << triptr->vertex[2][1] << endl;
		ntout << triptr->vertex[0][0] << " " << triptr->vertex[0][1] << endl << endl;
	}

	delete delaunay_triangles_mod;

	//}

	//die();
	}
	*/
	//herg++;

	//for (int i=0; i < npts; i++) delete[] adjacent_triangles[i];
}

bool DelaunayGrid::find_containing_triangle(lensvector &input_pt, const int img_pixel_i, const int img_pixel_j, int& trinum, bool& inside_triangle, bool& on_vertex, int& kmin)
{
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
		}
	} else {
		n = 0;
	}
	//cout << "searching for point (" << input_pt[0] << "," << input_pt[1] << "), starting with pixel " << n << " (" << srcpts[n][0] << " " << srcpts[n][1] << ")" << endl;
	on_vertex = false;
	trinum = search_grid(n,input_pt,inside_triangle);
	//cout << "...found in triangle " << trinum << endl;
	Triangle *triptr = &triangle[trinum];
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

void DelaunayGrid::find_containing_triangle(lensvector &input_pt, int& trinum, bool& inside_triangle, bool& on_vertex, int& kmin)
{
	// this version does not use information from lensing to find a starting triangle during the search; it just starts with triangle zero
	on_vertex = false;
	trinum = search_grid(0,input_pt,inside_triangle);
	Triangle *triptr = &triangle[trinum];
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

int DelaunayGrid::assign_active_indices_and_count_source_pixels(const bool activate_unmapped_pixels)
{
	int source_pixel_i=0;
	for (int i=0; i < n_srcpts; i++) {
		active_pixel[i] = false;
		if ((maps_to_image_pixel[i]) or (activate_unmapped_pixels)) {
			active_pixel[i] = true;
			active_index[i] = source_pixel_i++;
		} else {
			if ((lens->mpi_id==0) and (lens->regularization_method == 0)) warn(lens->warnings,"A source pixel does not map to any image pixel (for source pixel %i), center (%g,%g)",i,srcpts[i][0],srcpts[i][1]); // only show warning if no regularization being used, since matrix cannot be inverted in that case
		}

	}
	return source_pixel_i;
}

void DelaunayGrid::generate_hmatrices(const bool interpolate)
{
	// NOTE: for the moment, we are assuming all the source pixels are 'active', i.e. will be used in the inversion
	record_adjacent_triangles_xy();

	auto add_hmatrix_entry = [](QLens *lens, const int l, const int i, const int j, const double entry)
	{
		int dup = false;
		for (int k=0; k < lens->hmatrix_row_nn[l][i]; k++) {
			if (lens->hmatrix_index_rows[l][i][k]==j) {
				lens->hmatrix_rows[l][i][k] += entry;
				dup = true;
				break;
			}
		}
		if (!dup) {
			lens->hmatrix_rows[l][i].push_back(entry);
			lens->hmatrix_index_rows[l][i].push_back(j);
			lens->hmatrix_row_nn[l][i]++;
			lens->hmatrix_nn[l]++;
		}
	};

	int i,j,k,l;
	if (interpolate) {
		int npts;
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		double x,y,xp,xm,yp,ym;
		lensvector interp_pt[4];
		for (i=0; i < n_srcpts; i++) {
			x = srcpts[i][0];
			y = srcpts[i][1];
			xp = x + voronoi_length[i]/2;
			xm = x - voronoi_length[i]/2;
			yp = y + voronoi_length[i]/2;
			ym = y - voronoi_length[i]/2;
			interp_pt[0].input(xp,y);
			interp_pt[1].input(xm,y);
			interp_pt[2].input(x,yp);
			interp_pt[3].input(x,ym);
			add_hmatrix_entry(lens,0,i,i,-2.0);
			add_hmatrix_entry(lens,1,i,i,-2.0);
			for (j=0; j < 4; j++) {
				if (j > 1) l = 1;
				else l = 0;
				find_containing_triangle(interp_pt[j],trinum,inside_triangle,on_vertex,kmin);
				if (!inside_triangle) {
					if (!on_vertex) continue; // assume SB = 0 outside grid
				}
				if (lens->natural_neighbor_interpolation) {
					find_interpolation_weights_nn(interp_pt[j], trinum, npts, 0);
				} else {
					find_interpolation_weights_3pt(interp_pt[j], trinum, npts, 0);
				}
				for (k=0; k < npts; k++) {
					add_hmatrix_entry(lens,l,i,interpolation_indx[k][0],interpolation_wgts[k][0]); 
				}
			}
		}
	} else {
		int vertex_i1, vertex_i2, trinum;
		Triangle* triptr;
		bool found_i1, found_i2;
		double x1, y1, x2, y2, dpt, dpt1, dpt2, dpt12;
		double length, minlength, avg_length;
		avg_length = sqrt(avg_area);
		lensvector pt;
		for (i=0; i < n_srcpts; i++) {
			for (j=0; j < 4; j++) {
				if (j > 1) l = 1;
				else l = 0;
				vertex_i1 = -1;
				vertex_i2 = -1;
				if ((trinum = adj_triangles[j][i]) != -1) {
					triptr = &triangle[trinum];
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
					x1 = srcpts[vertex_i1][0];
					y1 = srcpts[vertex_i1][1];
					x2 = srcpts[vertex_i2][0];
					y2 = srcpts[vertex_i2][1];
					if (j < 2) {
						pt[1] = srcpts[i][1];
						pt[0] = ((x2-x1)/(y2-y1))*(pt[1]-y1) + x1;
						dpt = abs(pt[0]-srcpts[i][0]);
					} else {
						pt[0] = srcpts[i][0];
						pt[1] = ((y2-y1)/(x2-x1))*(pt[0]-x1) + y1;
						dpt = abs(pt[1]-srcpts[i][1]);
					}
					dpt12 = sqrt(SQR(x2-x1) + SQR(y2-y1));
					dpt1 = sqrt(SQR(pt[0]-x1)+SQR(pt[1]-y1));
					dpt2 = sqrt(SQR(pt[0]-x2)+SQR(pt[1]-y2));
					// we scale hmatrix by the average triangle length so the regularization parameter is dimensionless
					add_hmatrix_entry(lens,l,i,i,-avg_length/dpt);
					add_hmatrix_entry(lens,l,i,vertex_i1,avg_length*dpt2/(dpt*dpt12));
					add_hmatrix_entry(lens,l,i,vertex_i2,avg_length*dpt1/(dpt*dpt12));
				} else {
					minlength=1e30;
					for (k=0; k < n_shared_triangles[i]; k++) {
						triptr = &triangle[shared_triangles[i][k]];
						length = sqrt(triptr->area);
						if (length < minlength) minlength = length;
					}
					add_hmatrix_entry(lens,l,i,i,-avg_length/minlength);
					//add_hmatrix_entry(lens,l,i,i,sqrt(1/2.0)/2);
				}
			}
		}
	}
}

void DelaunayGrid::generate_gmatrices(const bool interpolate)
{
	// NOTE: for the moment, we are assuming all the source pixels are 'active', i.e. will be used in the inversion
	if (!interpolate) record_adjacent_triangles_xy();

	auto add_gmatrix_entry = [](QLens *lens, const int l, const int i, const int j, const double entry)
	{
		int dup = false;
		for (int k=0; k < lens->gmatrix_row_nn[l][i]; k++) {
			if (lens->gmatrix_index_rows[l][i][k]==j) {
				lens->gmatrix_rows[l][i][k] += entry;
				dup = true;
				break;
			}
		}
		if (!dup) {
			lens->gmatrix_rows[l][i].push_back(entry);
			lens->gmatrix_index_rows[l][i].push_back(j);
			lens->gmatrix_row_nn[l][i]++;
			lens->gmatrix_nn[l]++;
		}
	};


	int i,k,l;
	if (interpolate) {
		int npts;
		bool inside_triangle;
		bool on_vertex;
		int trinum,kmin;
		double x,y,xp,xm,yp,ym;
		lensvector interp_pt[4];
		for (i=0; i < n_srcpts; i++) {
			x = srcpts[i][0];
			y = srcpts[i][1];
			xp = x + voronoi_length[i];
			xm = x - voronoi_length[i];
			yp = y + voronoi_length[i];
			ym = y - voronoi_length[i];
			interp_pt[0].input(xp,y);
			interp_pt[1].input(xm,y);
			interp_pt[2].input(x,yp);
			interp_pt[3].input(x,ym);
			for (l=0; l < 4; l++) {
				add_gmatrix_entry(lens,l,i,i,1.0);
				find_containing_triangle(interp_pt[l],trinum,inside_triangle,on_vertex,kmin);
				if (!inside_triangle) {
					if (!on_vertex) continue; // assume SB = 0 outside grid
				}
				if (lens->natural_neighbor_interpolation) {
					find_interpolation_weights_nn(interp_pt[l], trinum, npts, 0);
				} else {
					find_interpolation_weights_3pt(interp_pt[l], trinum, npts, 0);
				}
				for (k=0; k < npts; k++) {
					add_gmatrix_entry(lens,l,i,interpolation_indx[k][0],-interpolation_wgts[k][0]);
				}
			}
		}
	} else {
		int vertex_i1, vertex_i2, trinum;
		Triangle* triptr;
		bool found_i1, found_i2;
		double length, minlength;
		double x1, y1, x2, y2, dpt, dpt1, dpt2, dpt12;
		lensvector pt;
		for (i=0; i < n_srcpts; i++) {
			for (l=0; l < 4; l++) {
				vertex_i1 = -1;
				vertex_i2 = -1;
				if ((trinum = adj_triangles[l][i]) != -1) {
					triptr = &triangle[trinum];
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
					x1 = srcpts[vertex_i1][0];
					y1 = srcpts[vertex_i1][1];
					x2 = srcpts[vertex_i2][0];
					y2 = srcpts[vertex_i2][1];
					if (l < 2) {
						pt[1] = srcpts[i][1];
						pt[0] = ((x2-x1)/(y2-y1))*(pt[1]-y1) + x1;
						dpt = abs(pt[0]-srcpts[i][0]);
					} else {
						pt[0] = srcpts[i][0];
						pt[1] = ((y2-y1)/(x2-x1))*(pt[0]-x1) + y1;
						dpt = abs(pt[1]-srcpts[i][1]);
					}
					dpt12 = sqrt(SQR(x2-x1) + SQR(y2-y1));
					dpt1 = sqrt(SQR(pt[0]-x1)+SQR(pt[1]-y1));
					dpt2 = sqrt(SQR(pt[0]-x2)+SQR(pt[1]-y2));
					add_gmatrix_entry(lens,l,i,i,1.0);
					add_gmatrix_entry(lens,l,i,vertex_i1,-dpt2/(dpt12));
					add_gmatrix_entry(lens,l,i,vertex_i2,-dpt1/(dpt12));
					//add_gmatrix_entry(lens,l,i,i,sqrt(1/2.0)/2);
				} else {
					minlength=1e30;
					for (k=0; k < n_shared_triangles[i]; k++) {
						triptr = &triangle[shared_triangles[i][k]];
						length = sqrt(triptr->area);
						if (length < minlength) minlength = length;
					}
					add_gmatrix_entry(lens,l,i,i,1.0);
					//add_gmatrix_entry(lens,l,i,i,sqrt(1/2.0)/2);
				}
			}
		}
	}
}

void DelaunayGrid::generate_covariance_matrix(double *cov_matrix_packed, const double input_corr_length, const int kernel_type, const double matern_index, double *wgtfac, const bool add_to_covmatrix, const double amplitude)
{
	bool extra_weighting = (wgtfac==NULL) ? false : true;
	double corrlength;
	int i,j;
	double sqrdist,x,matern_fac;
	double epsilon = lens->covmatrix_epsilon;
	if (kernel_type==0) {
		if (matern_index <= 0) die("Matern kernel index nu must be greater than zero");
		matern_fac = pow(2,1-matern_index)/Gamma(matern_index);
	}
	double *covptr;
	int *indx = new int[n_srcpts];
	indx[0] = 0;
	for (i=1; i < n_srcpts; i++) indx[i] = indx[i-1] + n_srcpts-i+1; // allows us to find first nonzero column with packed storage

	//double lumreg_rc = lens->lumreg_rc;
	double wi, wj, fac;
	#pragma omp parallel for private(i,j,sqrdist,x,covptr,corrlength,fac,wi,wj) schedule(dynamic)
	for (i=0; i < n_srcpts; i++) {
		covptr = cov_matrix_packed+indx[i];
		if (extra_weighting) {
			//wi = exp(-wgtfac[i]);
			wi = wgtfac[i];
			//wi = (1-lumreg_rc)*wgtfac[i]+lumreg_rc;
		}
		else wi=1.0;
		fac = wi*wi;
		if (amplitude >= 0) fac *= amplitude;
		if (!add_to_covmatrix) *covptr = epsilon; // adding epsilon to diagonal reduces numerical error during inversion by increasing the smallest eigenvalues
		*(covptr++) += fac;
		//*(covptr++) = 1.0 + epsilon; // adding epsilon to diagonal reduces numerical error during inversion by increasing the smallest eigenvalues
		for (j=i+1; j < n_srcpts; j++) {
			if (!add_to_covmatrix) *covptr = 0;
			sqrdist = SQR(srcpts[i][0]-srcpts[j][0]) + SQR(srcpts[i][1]-srcpts[j][1]);
			corrlength = input_corr_length;
			double xsig = 0.5;
			if (extra_weighting) {
				//wj = exp(-wgtfac[j]);
				wj = wgtfac[j];
				//wj = (1-lumreg_rc)*wgtfac[j]+lumreg_rc;
				fac = wi*wj;
				//double wj = pow(wgtfac[j],lens->regparam_lum_index);
				//cout << wi << " " << wj << endl;
			} else {
				fac = 1.0;
			}
			if (amplitude >= 0) fac *= amplitude;
			if (kernel_type==0) {
				x = sqrt(2*matern_index*sqrdist)/corrlength;
				if (x==0) {
					cout << "WHAT THE FUCK? x=0... sqrdist=" << sqrdist << " matern_index=" << matern_index << " corrlength=" << corrlength << endl;
					cout << "i: " << i << " si_x=" << srcpts[i][0] << " si_y=" << srcpts[i][1] << endl;
					cout << "j: " << j << " sj_x=" << srcpts[j][0] << " sj_y=" << srcpts[j][1] << endl;
					die();
				}
				*(covptr++) += fac*matern_fac*pow(x,matern_index)*modified_bessel_function(x,matern_index); // Matern kernel
			} else if (kernel_type==1) {
				*(covptr++) += fac*exp(-sqrt(sqrdist)/corrlength); // exponential kernel (equal to Matern kernel with matern_index = 0.5)
			} else {
				*(covptr++) += fac*exp(-sqrdist/(2*corrlength*corrlength)); // Gaussian kernel (limit of Matern kernel as matern_index goes to infinity)
			}
		}
	}
	delete[] indx;
}

/*
void QLens::set_corrlength_for_given_matscale()
{
	double (Brent::*corrlength_eq)(const double);
	corrlength_eq = static_cast<double (Brent::*)(const double)> (&QLens::corrlength_eq_matern_factor);
	double logcorr = BrentsMethod_Inclusive(corrlength_eq,-3,40,1e-6,true);
	kernel_correlation_length = pow(10,logcorr);
	//double optimal_corrlength = pow(10,logcorr);
	//cout << "Optimal correlation length: " << optimal_corrlength << endl;
}

double QLens::corrlength_eq_matern_factor(const double log_corr_length)
{
	double matern_fac, xsc;
	matern_fac = pow(2,1-matern_index)/Gamma(matern_index);
	xsc = sqrt(2*matern_index)*matern_approx_source_size/pow(10,log_corr_length);
	return (matern_scale - (1-matern_fac*pow(xsc,matern_index)*delaunay_srcgrid->modified_bessel_function(xsc,matern_index)));
}
*/

/*
void QLens::plot_matern_function(double rmin, double rmax, int rpts, const char *mfilename)
{
	double r, rstep, x, matern_fac, mfunc;
	rstep = pow(rmax/rmin, 1.0/(rpts-1));
	matern_fac = pow(2,1-matern_index)/Gamma(matern_index);
	int i;
	ofstream mout(mfilename);
	if (use_scientific_notation) mout << setiosflags(ios::scientific);
	for (i=0, r=rmin; i < rpts; i++, r *= rstep) {
		x = sqrt(2*matern_index)*r/kernel_correlation_length;
		mfunc = matern_fac*pow(x,matern_index)*delaunay_srcgrid->modified_bessel_function(x,matern_index);
		mout << r << " " << mfunc << endl;
	}
}
*/

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

void DelaunayGrid::plot_surface_brightness(string root, const int npix, const bool interpolate_sb, const bool plot_magnification, const bool plot_fits)
{
	double x, y, xlength, ylength, pixel_xlength, pixel_ylength;
	int i, j, npts_x, npts_y;
	xlength = srcgrid_xmax-srcgrid_xmin;
	ylength = srcgrid_ymax-srcgrid_ymin;
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

		ofstream pixel_xvals; lens->open_output_file(pixel_xvals,x_filename);
		for (i=0, x=srcgrid_xmin; i <= npts_x; i++, x += pixel_xlength) pixel_xvals << x << endl;

		ofstream pixel_yvals; lens->open_output_file(pixel_yvals,y_filename);
		for (i=0, y=srcgrid_ymin; i <= npts_y; i++, y += pixel_ylength) pixel_yvals << y << endl;

		lens->open_output_file(pixel_output_file,img_filename.c_str());
	}
	int srcpt_i, trinum;
	double sb;
	double **sbvals;
	if (plot_fits) {
		sbvals = new double*[npts_x];
		for (i=0; i < npts_x; i++) sbvals[i] = new double[npts_y];
	}
	//cout << "npts_x=" << npts_x << " " << xlength << " " << ylength << " " << npix << " " << srcgrid_xmax << " " << srcgrid_xmin << " " << srcgrid_ymax << " " << srcgrid_ymin << endl;
	//cout << "npts_y=" << npts_y << endl;
	lensvector pt;
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
				else sb = surface_brightness[srcpt_i];
			}
			if (plot_fits) sbvals[i][j] = sb;
			//cout << x << " " << y << " " << srcpts[srcpt_i][0] << " " << srcpts[srcpt_i][1] << endl;
			if (!plot_fits) pixel_output_file << sb << " ";
		}
		if (!plot_fits) pixel_output_file << endl;
	}
	if (!plot_fits) {
		string srcpt_filename = root + "_srcpts.dat";
		ofstream srcout; lens->open_output_file(srcout,srcpt_filename);
		for (i=0; i < n_srcpts; i++) {
			srcout << srcpts[i][0] << " " << srcpts[i][1] << endl;
		}
		string voronoi_filename = root + "_voronoi.dat";
		ofstream vout; lens->open_output_file(vout,voronoi_filename);
		for (i=0; i < n_srcpts; i++) {
			for (j=0; j < n_shared_triangles[i]; j++) {
				vout << voronoi_boundary_x[i][j] << " " << voronoi_boundary_y[i][j] << endl;
			}
			vout << voronoi_boundary_x[i][0] << " " << voronoi_boundary_y[i][0] << endl << endl;
		}
		string delaunay_filename = root + "_delaunay.dat";
		ofstream delout; lens->open_output_file(delout,delaunay_filename);
		for (i=0; i < n_triangles; i++) {
			delout << triangle[i].vertex[0][0] << " " << triangle[i].vertex[0][1] << endl;
			delout << triangle[i].vertex[1][0] << " " << triangle[i].vertex[1][1] << endl;
			delout << triangle[i].vertex[2][0] << " " << triangle[i].vertex[2][1] << endl;
			delout << triangle[i].vertex[0][0] << " " << triangle[i].vertex[0][1] << endl;
			delout << endl;
		}

	}

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
		if (lens->fit_output_dir != ".") lens->create_output_directory(); // in case it hasn't been created already
		string filename = lens->fit_output_dir + "/" + root;

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
					long fpixel[naxis];
					for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
					pixels = new double[npts_x];

					for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
					{
						for (i=0; i < npts_x; i++) {
							pixels[i] = sbvals[i][j];
						}
						fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
					}
					delete[] pixels;
				}
			}
			fits_close_file(outfptr, &status);
		} 

		if (status) fits_report_error(stderr, status); // print any error message
		for (i=0; i < npts_x; i++) delete[] sbvals[i];
		delete[] sbvals;
#endif
	}

}

double DelaunayGrid::find_moment(const int p, const int q, const int npix, const double xc, const double yc, const double b, const double a, const double phi)
{
	double x, y, xlength, ylength, pixel_xlength, pixel_ylength;
	int i, j, k, npts_x, npts_y;
	xlength = srcgrid_xmax-srcgrid_xmin;
	ylength = srcgrid_ymax-srcgrid_ymin;
	npts_x = (int) npix*sqrt(xlength/ylength);
	npts_y = (int) npts_x*ylength/xlength;
	pixel_xlength = xlength/npts_x;
	pixel_ylength = ylength/npts_y;

	double sb,xp,yq,moment=0;
	lensvector pt;
	double y0 = srcgrid_ymin + pixel_ylength/2;
	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
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

void DelaunayGrid::find_source_moments(const int npix, double &qs, double &phi_s, double &xavg, double &yavg)
{
	double M00, M10, M01, M20, M02, M11;
#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
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
		cout << "b=" << sig_minor << " a=" << sig_major << endl;
		cout << "qs=" << qs << " phi_s=" << phi_s << endl;
	} while ((abs(qs-qs_old) > qstol*qs) and (++index < max_it));
	//} while (index++ < 2);
#ifdef USE_OPENMP
		if (lens->show_wtime) {
			wtime = omp_get_wtime() - wtime0;
			if (lens->mpi_id==0) cout << "Wall time for finding qs, phi_s for adaptive source grid: " << wtime << endl;
		}
#endif
}

void DelaunayGrid::get_grid_points(vector<double>& xvals, vector<double>& yvals, vector<double>& sb_vals)
{
	for (int i=0; i < n_srcpts; i++) {
		xvals.push_back(srcpts[i][0]);
		yvals.push_back(srcpts[i][1]);
		sb_vals.push_back(surface_brightness[i]);
	}
}

DelaunayGrid::~DelaunayGrid()
{
	delete[] srcpts;
	delete[] triangle;
	delete[] surface_brightness;
	delete[] inv_magnification;
	delete[] maps_to_image_pixel;
	delete[] active_pixel;
	delete[] active_index;
	delete[] n_shared_triangles;
	delete[] voronoi_area;
	delete[] voronoi_length;
	for (int i=0; i < n_srcpts; i++) {
		delete[] voronoi_boundary_x[i];
		delete[] voronoi_boundary_y[i];
		delete[] shared_triangles[i];
	}
	delete[] voronoi_boundary_x;
	delete[] voronoi_boundary_y;
	delete[] shared_triangles;
	if (imggrid_ivals != NULL) delete[] imggrid_ivals;
	if (imggrid_jvals != NULL) delete[] imggrid_jvals;
	delete[] adj_triangles[0];
	delete[] adj_triangles[1];
	delete[] adj_triangles[2];
	delete[] adj_triangles[3];
	if (img_index_ij != NULL) {
		for (int i=0; i < img_ni; i++) delete[] img_index_ij[i];
		delete[] img_index_ij;
	}
}

/******************************** Functions in class ImagePixelData, and FITS file functions *********************************/

void ImagePixelData::load_data(string root)
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
	for (i=0; i < npixels_x; i++) {
		surface_brightness[i] = new double[npixels_y];
		high_sn_pixel[i] = new bool[npixels_y];
		in_mask[0][i] = new bool[npixels_y];
		extended_mask[0][i] = new bool[npixels_y];
		foreground_mask[i] = new bool[npixels_y];
		for (j=0; j < npixels_y; j++) {
			in_mask[0][i][j] = true;
			extended_mask[0][i][j] = true;
			foreground_mask[i][j] = true;
			high_sn_pixel[i][j] = true;
			sbfile >> surface_brightness[i][j];
		}
	}
	find_extended_mask_rmax(); // used when splining integrals for deflection/hessian from Fourier modes
	assign_high_sn_pixels();
}

void ImagePixelData::load_from_image_grid(ImagePixelGrid* image_pixel_grid, const double noise_in)
{
	int i,j,k;
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
	if ((lens != NULL) and (lens->fft_convolution)) lens->cleanup_FFT_convolution_arrays(); // since number of image pixels has changed, will need to redo FFT setup

	npixels_x = image_pixel_grid->x_N;
	npixels_y = image_pixel_grid->y_N;
	xmin = image_pixel_grid->xmin;
	xmax = image_pixel_grid->xmax;
	ymin = image_pixel_grid->ymin;
	ymax = image_pixel_grid->ymax;

	xvals = new double[npixels_x+1];
	for (i=0; i <= npixels_x; i++) xvals[i] = image_pixel_grid->corner_pts[i][0][0];
	yvals = new double[npixels_y+1];
	for (i=0; i <= npixels_y; i++) yvals[i] = image_pixel_grid->corner_pts[0][i][1];

	pixel_xcvals = new double[npixels_x];
	pixel_ycvals = new double[npixels_y];
	for (i=0; i < npixels_x; i++) {
		pixel_xcvals[i] = (xvals[i]+xvals[i+1])/2;
	}
	for (i=0; i < npixels_y; i++) {
		pixel_ycvals[i] = (yvals[i]+yvals[i+1])/2;
	}

	double xstep = pixel_xcvals[1] - pixel_xcvals[0];
	double ystep = pixel_ycvals[1] - pixel_ycvals[0];
	pixel_size = dmin(xstep,ystep);

	surface_brightness = new double*[npixels_x];
	high_sn_pixel = new bool*[npixels_x];
	n_masks = 1;
	n_mask_pixels = new int[1];
	n_mask_pixels[0] = npixels_x*npixels_y;
	extended_mask_n_neighbors = new int[1];
	extended_mask_n_neighbors[0] = -1; // this means all the pixels are included in the extended mask by default
	n_high_sn_pixels = n_mask_pixels[0]; // this will be recalculated in assign_high_sn_pixels() function
	in_mask = new bool**[1];
	in_mask[0] = new bool*[npixels_x];
	extended_mask = new bool**[1];
	extended_mask[0] = new bool*[npixels_x];
	foreground_mask = new bool*[npixels_x];
	noise_map = new double*[npixels_x];
	covinv_map = new double*[npixels_x];
	double covinv = 1.0/(noise_in*noise_in);
	for (i=0; i < npixels_x; i++) {
		surface_brightness[i] = new double[npixels_y];
		high_sn_pixel[i] = new bool[npixels_y];
		in_mask[0][i] = new bool[npixels_y];
		extended_mask[0][i] = new bool[npixels_y];
		foreground_mask[i] = new bool[npixels_y];
		noise_map[i] = new double[npixels_y];
		covinv_map[i] = new double[npixels_y];
		for (j=0; j < npixels_y; j++) {
			in_mask[0][i][j] = true;
			extended_mask[0][i][j] = true;
			foreground_mask[i][j] = true;
			high_sn_pixel[i][j] = true;
			surface_brightness[i][j] = image_pixel_grid->surface_brightness[i][j];
			noise_map[i][j] = noise_in;
			covinv_map[i][j] = covinv;
		}
	}
	find_extended_mask_rmax(); // used when splining integrals for deflection/hessian from Fourier modes
	assign_high_sn_pixels();
}

bool ImagePixelData::load_data_fits(bool use_pixel_size, string fits_filename, const int hdu_indx, const bool show_header)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else
	bool image_load_status = false;
	int i,j,k,kk;
	fitsfile *fptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix, naxis;
	long naxes[2] = {1,1};
	double *pixels;
	double x, y, xstep, ystep;
	bool pixel_noise_specified = false;
	if ((lens != NULL) and (lens->background_pixel_noise > 0)) pixel_noise_specified = true;
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
			if ((show_header) and ((lens==NULL) or (lens->mpi_id==0))) cout << cardstring << endl;
			if (reading_qlens_comment) {
				if ((pos = cardstring.find("COMMENT")) != string::npos) {
					if (((pos1 = cardstring.find("mk: ")) != string::npos) or ((pos1 = cardstring.find("MK: ")) != string::npos)) {
						reading_markers = true;
						reading_qlens_comment = false;
						if (lens != NULL) lens->param_markers = cardstring.substr(pos1+4);
					} else {
						if (lens != NULL) lens->data_info += cardstring.substr(pos+8);
					}
				} else break;
			} else if (reading_markers) {
				if ((pos = cardstring.find("COMMENT")) != string::npos) {
					if (lens != NULL) lens->param_markers += cardstring.substr(pos+8);
					// A potential issue is that if there are enough markers to fill more than one line, there might be an extra space inserted,
					// in which case the markers won't come out properly. No time to deal with this now, but something to look out for.
				} else break;
			} else if (((pos = cardstring.find("ql: ")) != string::npos) or ((pos = cardstring.find("QL: ")) != string::npos)) {
				reading_qlens_comment = true;
				if (lens != NULL) lens->data_info = cardstring.substr(pos+4);
			} else if (((pos = cardstring.find("mk: ")) != string::npos) or ((pos = cardstring.find("MK: ")) != string::npos)) {
				reading_markers = true;
				if (lens != NULL) lens->param_markers = cardstring.substr(pos+4);
			} else if (cardstring.find("PXSIZE ") != string::npos) {
				string pxsize_string = cardstring.substr(11);
				stringstream pxsize_str;
				pxsize_str << pxsize_string;
				pxsize_str >> pixel_size;
				if (lens != NULL) lens->data_pixel_size = pixel_size;
				pixel_size_found = true;
				if (!use_pixel_size) use_pixel_size = true;
			} else if (cardstring.find("PXNOISE ") != string::npos) {
				string pxnoise_string = cardstring.substr(11);
				stringstream pxnoise_str;
				pxnoise_str << pxnoise_string;
				pxnoise_str >> pnoise;
				if (lens != NULL) {
					lens->background_pixel_noise = pnoise;
					pixel_noise_specified = true;
				}
			} else if (cardstring.find("PSFSIG ") != string::npos) {
				string psfwidth_string = cardstring.substr(11);
				stringstream psfwidth_str;
				psfwidth_str << psfwidth_string;
				if (lens != NULL) psfwidth_str >> lens->psf_width_x;
				if (lens != NULL) lens->psf_width_y = lens->psf_width_x;
			} else if (cardstring.find("ZSRC ") != string::npos) {
				string zsrc_string = cardstring.substr(11);
				stringstream zsrc_str;
				zsrc_str << zsrc_string;
				if (lens != NULL) {
					zsrc_str >> lens->source_redshift;
					if (lens->auto_zsource_scaling) lens->reference_source_redshift = lens->source_redshift;
				}
			} else if (cardstring.find("ZLENS ") != string::npos) {
				string zlens_string = cardstring.substr(11);
				stringstream zlens_str;
				zlens_str << zlens_string;
				if (lens != NULL) zlens_str >> lens->lens_redshift;
			}
		}
		if ((reading_markers) and (lens != NULL)) {
			// Commas are used in FITS file as delimeter so spaces don't get lost; now convert to spaces again
			for (size_t i = 0; i < lens->param_markers.size(); ++i) {
				 if (lens->param_markers[i] == ',') {
					  lens->param_markers.replace(i, 1, " ");
				 }
			}
		}

		bool new_dimensions = true;
		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) )
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long fpixel[naxis];
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
					if (noise_map != NULL) {
						for (i=0; i < npixels_x; i++) delete[] noise_map[i];
						delete[] noise_map;
					}
					if (covinv_map != NULL) {
						for (i=0; i < npixels_x; i++) delete[] covinv_map[i];
						delete[] covinv_map;
					}

					if ((lens != NULL) and (lens->fft_convolution)) lens->cleanup_FFT_convolution_arrays(); // since number of image pixels has changed, will need to redo FFT setup

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
				if (use_pixel_size) {
					if (!pixel_size_found) {
						if (lens==NULL) use_pixel_size = false; // couldn't find pixel size in file, and there is no lens object to find it from either
						else pixel_size = lens->data_pixel_size;
					}
				}
				if (use_pixel_size) {
					xstep = ystep = pixel_size;
					xmax = 0.5*npixels_x*pixel_size;
					ymax = 0.5*npixels_y*pixel_size;
					xmin=-xmax; ymin=-ymax;
				} else {
					xstep = (xmax-xmin)/npixels_x;
					ystep = (ymax-ymin)/npixels_y;
				}
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
						noise_map[i] = new double[npixels_y];
						covinv_map[i] = new double[npixels_y];
						for (j=0; j < npixels_y; j++) {
							in_mask[0][i][j] = true;
							extended_mask[0][i][j] = true;
							foreground_mask[i][j] = true;
						}
					}
					for (j=0; j < npixels_y; j++) {
						high_sn_pixel[i][j] = true;
					}
				}
				if (pixel_noise_specified) {
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

void ImagePixelData::save_data_fits(string fits_filename, const bool subimage, const double xmin_in, const double xmax_in, const double ymin_in, const double ymax_in)
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
				long fpixel[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[npix_x];

				for (fpixel[1]=1, j=min_j; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=min_i, kk=0; i <= max_i; i++, kk++) {
						pixels[kk] = surface_brightness[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] pixels;
			}
			if (lens->data_pixel_size > 0)
				fits_write_key(outfptr, TDOUBLE, "PXSIZE", &lens->data_pixel_size, "length of square pixels (in arcsec)", &status);
			if (lens->background_pixel_noise != 0)
				fits_write_key(outfptr, TDOUBLE, "PXNOISE", &lens->background_pixel_noise, "pixel surface brightness noise", &status);
			if (lens->data_info != "") {
				string comment = "ql: " + lens->data_info;
				fits_write_comment(outfptr, comment.c_str(), &status);
			}
		}
		fits_close_file(outfptr, &status);
	} 

	if (status) fits_report_error(stderr, status); // print any error message
#endif
}

bool ImagePixelData::load_noise_map_fits(string fits_filename, const int hdu_indx, const bool show_header)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else
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

	double noise_bg = 1e30;
	char card[FLEN_CARD];   // Standard string lengths defined in fitsio.h
	int hdutype;
	if (!fits_open_file(&fptr, fits_filename.c_str(), READONLY, &status))
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
				long fpixel[naxis];
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
							if (pixels[i] < noise_bg) noise_bg = pixels[i];
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

			}
		}
		fits_close_file(fptr, &status);
	}
	if (lens != NULL) lens->background_pixel_noise = noise_bg; // store the background noise separately

	if (status) fits_report_error(stderr, status); // print any error message
	if (image_load_status) noise_map_fits_filename = fits_filename;
	return image_load_status;
#endif
}

void ImagePixelData::unload_noise_map()
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



void ImagePixelData::get_grid_params(double& xmin_in, double& xmax_in, double& ymin_in, double& ymax_in, int& npx, int& npy)
{
	if (xvals==NULL) die("cannot get image pixel data parameters; no data has been loaded");
	xmin_in = xvals[0];
	xmax_in = xvals[npixels_x];
	ymin_in = yvals[0];
	ymax_in = yvals[npixels_y];
	npx = npixels_x;
	npy = npixels_y;
}

void ImagePixelData::assign_high_sn_pixels() // should probably use the foreground mask here, since we might be masking out foreground stars. Implement!!
{
	double global_max_sb = -1e30;
	int i,j;
	for (j=0; j < npixels_y; j++) {
		for (i=0; i < npixels_x; i++) {
			if (surface_brightness[i][j] > global_max_sb) global_max_sb = surface_brightness[i][j];
		}
	}
	if (lens != NULL) {
		n_high_sn_pixels = 0;
		for (j=0; j < npixels_y; j++) {
			for (i=0; i < npixels_x; i++) {
				if (surface_brightness[i][j] >= lens->high_sn_frac*global_max_sb) {
					high_sn_pixel[i][j] = true;
					n_high_sn_pixels++;
				}
				else high_sn_pixel[i][j] = false;
			}
		}
	}
}

double ImagePixelData::find_max_sb(const int mask_k)
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

double ImagePixelData::find_avg_sb(const double sb_threshold, const int mask_k)
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

bool ImagePixelData::load_mask_fits(const int mask_k, string fits_filename, const bool foreground, const bool emask, const bool add_mask_pixels) // if 'add_mask_pixels' is true, then doesn't unmask any pixels that are already masked
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else
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

	if (!fits_open_file(&fptr, fits_filename.c_str(), READONLY, &status))
	{
		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) )
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long fpixel[naxis];
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
							}
							else {
								foreground_mask[iprime][jprime] = true;
								n_maskpixels++;
								//cout << pixels[iprime] << endl;
							}
							if (new_mask) {
								in_mask[mask_k][iprime][jprime] = true; // if new mask was created, then mask contains all the pixels by default
								extended_mask[mask_k][iprime][jprime] = true; // if new mask was created, then extended mask contains all the pixels by default
							}
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
								in_mask[mask_k][iprime][jprime] = true;
								if (!extended_mask[mask_k][iprime][jprime]) extended_mask[mask_k][iprime][jprime] = true; // the extended mask MUST contain all the primary mask pixels
								n_maskpixels++;
							//cout << pixels[iprime] << endl;
							}
							if (new_mask) extended_mask[mask_k][iprime][jprime] = true; // if new mask was created, then extended mask contains all the pixels by default
						}
					}
				}
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
		//set_extended_mask(lens->extended_mask_n_neighbors);
	}
	return image_load_status;
#endif
}

bool ImagePixelData::copy_mask(ImagePixelData* data, const int mask_k)
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


bool ImagePixelData::save_mask_fits(string fits_filename, const bool foreground, const bool emask, const int mask_k, const int reduce_nx, const int reduce_ny)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return false;
#else
	if (mask_k >= n_masks) { warn("mask with given index has not been created"); return false; }
	int i,j,iprime,jprime,kk;
	fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix = -64, naxis = 2;
	int offset_x=0, offset_y=0;
	int npix_x=npixels_x, npix_y=npixels_y;
	if (reduce_nx > 0) {
		if (reduce_nx > npixels_x) { warn("cannot reduce number of max pixels; reduced nx must be smaller than current number of pixels along x"); return false; }
		npix_x = reduce_nx;
		offset_x = (npixels_x-reduce_nx)/2;
		offset_x--;
	}
	if (reduce_ny > 0) {
		if (reduce_ny > npixels_y) { warn("cannot reduce number of max pixels; reduced nx must be smaller than current number of pixels along x"); return false; }
		npix_y = reduce_ny;
		offset_y = (npixels_y-reduce_ny)/2;
		offset_y--;
	}
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
				long fpixel[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[npix_x];

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					jprime = j + offset_y;
					for (i=0; i < npix_x; i++) {
						iprime = i + offset_x;
						if (foreground) {
							if (foreground_mask[iprime][jprime]) pixels[i] = 1.0;
							else pixels[i] = 0.0;
						} else if (emask) {
							if (extended_mask[mask_k][iprime][jprime]) pixels[i] = 1.0;
							else pixels[i] = 0.0;
						} else {
							if (in_mask[mask_k][iprime][jprime]) pixels[i] = 1.0;
							else pixels[i] = 0.0;
						}
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] pixels;
			}
		}
		fits_close_file(outfptr, &status);
	} 

	if (status) fits_report_error(stderr, status); // print any error message
	return true;
#endif
}

bool QLens::load_psf_fits(string fits_filename, const bool supersampled, const bool verbal)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else
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
	int nx, ny;
	long naxes[2] = {1,1};
	double *pixels;
	double peak_sb = -1e30;

	if (!fits_open_file(&fptr, fits_filename.c_str(), READONLY, &status))
	{
		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) )
		{
			if (naxis == 0) {
				die("Error: only 1D or 2D images are supported (dimension is %i)\n",naxis);
			} else {
				kk=0;
				long fpixel[naxis];
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
				image_load_status = true;
			}
		}
		fits_close_file(fptr, &status);
	} else {
		return false;
	}
	int imid, jmid, imin, imax, jmin, jmax;
	imid = nx/2;
	jmid = ny/2;
	imin = imid;
	imax = imid;
	jmin = jmid;
	jmax = jmid;
	for (i=0; i < nx; i++) {
		for (j=0; j < ny; j++) {
			if ((input_psf_matrix[i][j] > psf_threshold*peak_sb) or (supersampled)) {
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
	(*npix_x) = 2*nx_half+1;
	(*npix_y) = 2*ny_half+1;
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

	if ((verbal) and (mpi_id==0)) {
		cout << "PSF matrix dimensions: " << (*npix_x) << " " << (*npix_y) << " (input PSF dimensions: " << nx << " " << ny << ")" << endl;
		//cout << "PSF normalization =" << normalization << endl << endl;
	}
	for (i=0; i < nx; i++) delete[] input_psf_matrix[i];
	delete[] input_psf_matrix;

	if (status) fits_report_error(stderr, status); // print any error message
	if (image_load_status) psf_filename = fits_filename;

	return image_load_status;
#endif
}

bool QLens::save_psf_fits(string fits_filename, const bool supersampled)
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
				long fpixel[naxis];
				for (kk=0; kk < naxis; kk++) fpixel[kk] = 1;
				pixels = new double[nx];

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=0; i < nx; i++) {
						pixels[i] = psf[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
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

ImagePixelData::~ImagePixelData()
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
}

bool ImagePixelData::create_new_mask()
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

bool ImagePixelData::set_no_mask_pixels(const int mask_k)
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
	return true;
}

bool ImagePixelData::set_all_mask_pixels(const int mask_k)
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
	return true;
}

bool ImagePixelData::set_foreground_mask_to_primary_mask(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (in_mask[mask_k][i][j]) foreground_mask[i][j] = true;
			else foreground_mask[i][j] = false;
		}
	}
	return true;
}

bool ImagePixelData::invert_mask(const int mask_k)
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
	return true;
}

bool ImagePixelData::inside_mask(const double x, const double y, const int mask_k)
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

bool ImagePixelData::assign_mask_windows(const double sb_noise_threshold, const int threshold_size, const int mask_k)
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
				if ((mask_window_max_sb[l] > sb_noise_threshold*lens->background_pixel_noise) and (mask_window_sizes[l] > threshold_size)) {
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
	if (lens->mpi_id == 0) cout << "Trimmed " << n_mask_windows << " windows down to " << n_windows_eff << " windows" << endl;
	j=0;
	for (i=0; i < n_mask_windows; i++) {
		if (mask_window_sizes[i] != 0) {
			j++;
			if (lens->mpi_id == 0) cout << "Window " << j << " size: " << mask_window_sizes[i] << " max_sb: " << mask_window_max_sb[i] << endl;
		}
	}
	for (i=0; i < npixels_x; i++) delete[] mask_window_id[i];
	delete[] mask_window_id;
	return true;
}

bool ImagePixelData::unset_low_signal_pixels(const double sb_threshold, const int mask_k)
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
	return true;
}

bool ImagePixelData::set_positive_radial_gradient_pixels(const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return true; }
	int i,j;
	lensvector rhat, grad;
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
	return false;
}

bool ImagePixelData::set_neighbor_pixels(const bool only_interior_neighbors, const bool only_exterior_neighbors, const int mask_k)
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
	return true;
}

bool ImagePixelData::set_mask_window(const double xmin, const double xmax, const double ymin, const double ymax, const bool unset, const int mask_k)
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
	return true;
}

bool ImagePixelData::set_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset, const int mask_k)
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
	}
	return true;
}

bool ImagePixelData::reset_extended_mask(const int mask_k)
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

bool ImagePixelData::set_extended_mask(const int n_neighbors, const bool add_to_emask_in, const bool only_interior_neighbors, const int mask_k)
{
	if (mask_k >= n_masks) { warn("mask with specified index has not been loaded or created"); return false; }
	// This is very similar to the set_neighbor_pixels() function in ImagePixelData; used here for the outside_sb_prior feature
	int i,j,k;
	bool add_to_emask = add_to_emask_in;
	if (n_neighbors < 0) {
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if (foreground_mask[i][j]) extended_mask[mask_k][i][j] = true;
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
								if (foreground_mask[i+1][j]) extended_mask[mask_k][i+1][j] = true;
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
								if (foreground_mask[i-1][j]) extended_mask[mask_k][i-1][j] = true;
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
								if (foreground_mask[i][j+1]) extended_mask[mask_k][i][j+1] = true;
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
								if (foreground_mask[i][j-1]) extended_mask[mask_k][i][j-1] = true;
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
							if (foreground_mask[i][j]) extended_mask[mask_k][i][j] = true;
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
	find_extended_mask_rmax();
	for (i=0; i < npixels_x; i++) delete[] req[i];
	delete[] req;
	return true;
}

bool ImagePixelData::activate_partner_image_pixels(const int mask_k, const bool emask)
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
				lensvector pos,src;
				pos[0] = pixel_xcvals[i];
				pos[1] = pixel_ycvals[j];
				lens->find_sourcept(pos,src,0,lens->reference_zfactors,lens->default_zsrc_beta_factors);
				image *img = lens->get_images(src, n_images, false);
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
				if (found_itself) cout << "Found itself! Yay!" << endl;
				else cout << "NOTE: pixel couldn't find itself" << endl;
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
				lensvector pos,src;
				pos[0] = pixel_xcvals[i];
				pos[1] = pixel_ycvals[j];
				lens->find_sourcept(pos,src,0,lens->reference_zfactors,lens->default_zsrc_beta_factors);
				image *img = lens->get_images(src, n_images, false);
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
	return true;
}

void ImagePixelData::remove_overlapping_pixels_from_other_masks(const int mask_k)
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
}

bool ImagePixelData::set_extended_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset, const int mask_k)
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
	find_extended_mask_rmax();
	if (pixels_in_mask) warn("some pixels in the annulus were in the primary (lensed image) mask, and therefore could not be removed from extended mask");
	return true;
}

void ImagePixelData::find_extended_mask_rmax()
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


void ImagePixelData::set_foreground_mask_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset)
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

long int ImagePixelData::get_size_of_extended_mask(const int mask_k)
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

long int ImagePixelData::get_size_of_foreground_mask()
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



bool ImagePixelData::estimate_pixel_noise(const double xmin, const double xmax, const double ymin, const double ymax, double &noise, double &mean_sb, const int mask_k)
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

bool ImagePixelData::fit_isophote(const double xi0, const double xistep, const int emode, const double qi, const double theta_i_degrees, const double xc_i, const double yc_i, const int max_it, IsophoteData &isophote_data, const bool use_polar_higher_harmonics, const bool verbose, SB_Profile* sbprofile, const int default_sampling_mode_in, const int n_higher_harmonics, const bool fix_center, const int max_xi_it, const double ximax_in, const double rms_sbgrad_rel_threshold, const double npts_frac, const double rms_sbgrad_rel_transition, const double npts_frac_zeroweight)
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
			true_xc = sbprofile->x_center;
			true_yc = sbprofile->y_center;
			if (sbprofile->ellipticity_gradient==true) {
				double eps;
				sbprofile->ellipticity_function(xi,eps,true_theta);
				true_q = sqrt(1-eps);
				true_epsilon = 1-true_q;
			} else {
				true_q = sbprofile->q;
				true_epsilon = 1 - true_q;
				true_theta = sbprofile->theta;
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
		if (lens != NULL) dir = lens->fit_output_dir;
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

double ImagePixelData::sample_ellipse(const bool show_warnings, const double xi, const double xistep_in, const double epsilon, const double theta, const double xc, const double yc, int& npts, int& npts_sample, const int emode, int& sampling_mode, const double mean_pixel_noise, double *sbvals, double *sbgrad_wgts, SB_Profile* sbprofile, const bool fill_matrices, double* sb_residual, double *sb_weights, double** smatrix, const int ni, const int nf, const bool use_polar_higher_harmonics, const bool plot_ellipse, ofstream *ellout)
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

bool ImagePixelData::Cholesky_dcmp(double** a, int n)
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

void ImagePixelData::Cholesky_solve(double** a, double* b, double* x, int n)
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

void ImagePixelData::Cholesky_fac_inverse(double** a, int n)
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

/*
void ImagePixelData::add_point_image_from_centroid(ImageData* point_image_data, const double xmin, const double xmax, const double ymin, const double ymax, const double sb_threshold, const double pixel_error)
{
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
	double centroid_x=0, centroid_y=0, centroid_err_x=0, centroid_err_y=0, total_flux=0;
	int np=0;
	double xm,ym;
	for (j=jmin; j <= jmax; j++) {
		for (i=imin; i <= imax; i++) {
			if (surface_brightness[i][j] > sb_threshold) {
				xm = 0.5*(xvals[i+1]+xvals[i]);
				ym = 0.5*(yvals[j+1]+yvals[j]);
				centroid_x += xm*surface_brightness[i][j];
				centroid_y += ym*surface_brightness[i][j];
				//centroid_err_x += xm*xm*surface_brightness[i][j];
				//centroid_err_y += ym*ym*surface_brightness[i][j];
				total_flux += surface_brightness[i][j];
				np++;
			}
		}
	}
	if (total_flux==0) die("Zero pixels are above the stated surface brightness threshold for calculating image centroid");
	//double avg_signal = total_flux / np;
	//cout << "Average signal: " << avg_signal << endl;
	centroid_x /= total_flux;
	centroid_y /= total_flux;
	double centroid_err_x_sb=0, centroid_err_y_sb=0;
	for (j=jmin; j <= jmax; j++) {
		for (i=imin; i <= imax; i++) {
			if (surface_brightness[i][j] > sb_threshold) {
				xm = 0.5*(xvals[i+1]+xvals[i]);
				ym = 0.5*(yvals[j+1]+yvals[j]);
				centroid_err_x_sb += SQR(xm-centroid_x);
				centroid_err_y_sb += SQR(ym-centroid_y);
			}
		}
	}

	// Finding an error based on the second moment seems flawed, since with enough pixels the centroid should be known very well.
	// For now, we choose the pixel size to give the error.
	//centroid_err_x /= total_flux;
	//centroid_err_y /= total_flux;
	//centroid_err_x = sqrt(centroid_err_x - SQR(centroid_x));
	//centroid_err_y = sqrt(centroid_err_y - SQR(centroid_y));
	centroid_err_x_sb = sqrt(centroid_err_x_sb)*pixel_error/total_flux;
	centroid_err_y_sb = sqrt(centroid_err_y_sb)*pixel_error/total_flux;
	centroid_err_x = xvals[1] - xvals[0];
	centroid_err_y = yvals[1] - yvals[0];
	centroid_err_x = sqrt(SQR(centroid_err_x) + SQR(centroid_err_x_sb));
	centroid_err_y = sqrt(SQR(centroid_err_y) + SQR(centroid_err_y_sb));
	//cout << "err_x_sb=" << centroid_err_x_sb << ", err_x=" << centroid_err_x << ", err_y_sb=" << centroid_err_y_sb << ", err_y=" << centroid_err_y << endl;
	double centroid_err = dmax(centroid_err_x,centroid_err_y);
	double flux_err = pixel_error / sqrt(np);
	//cout << "centroid = (" << centroid_x << "," << centroid_y << "), err=(" << centroid_err_x << "," << centroid_err_y << "), flux = " << total_flux << ", flux_err = " << flux_err << endl;
	lensvector pos; pos[0] = centroid_x; pos[1] = centroid_y;
	point_image_data->add_image(pos,centroid_err,total_flux,flux_err,0,0);
}
*/

void ImagePixelData::plot_surface_brightness(string outfile_root, bool show_only_mask, bool show_extended_mask, bool show_foreground_mask, const int mask_k)
{
	string sb_filename = outfile_root + ".dat";
	string x_filename = outfile_root + ".x";
	string y_filename = outfile_root + ".y";

	ofstream pixel_image_file; lens->open_output_file(pixel_image_file,sb_filename);
	ofstream pixel_xvals; lens->open_output_file(pixel_xvals,x_filename);
	ofstream pixel_yvals; lens->open_output_file(pixel_yvals,y_filename);
	pixel_image_file << setiosflags(ios::scientific);
	int i,j,k;
	for (int i=0; i <= npixels_x; i++) {
		pixel_xvals << xvals[i] << endl;
	}
	for (int j=0; j <= npixels_y; j++) {
		pixel_yvals << yvals[j] << endl;
	}	
	bool show_sb;
	if (show_extended_mask) {
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
					pixel_image_file << surface_brightness[i][j];
				} else {
					pixel_image_file << "NaN";
				}
				if (i < npixels_x-1) pixel_image_file << " ";
			}
			pixel_image_file << endl;
		}
	} else if (show_foreground_mask) {
		for (j=0; j < npixels_y; j++) {
			for (i=0; i < npixels_x; i++) {
				if ((!show_only_mask) or (foreground_mask == NULL) or (foreground_mask[i][j])) {
					pixel_image_file << surface_brightness[i][j];
				} else {
					pixel_image_file << "NaN";
				}
				if (i < npixels_x-1) pixel_image_file << " ";
			}
			pixel_image_file << endl;
		}
	} else {
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
					pixel_image_file << surface_brightness[i][j];
				} else {
					pixel_image_file << "NaN";
				}
				if (i < npixels_x-1) pixel_image_file << " ";
			}
			pixel_image_file << endl;
		}
	}
}

/***************************************** Functions in class ImagePixelGrid ****************************************/

ImagePixelGrid::ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, double xmin_in, double xmax_in, double ymin_in, double ymax_in, int x_N_in, int y_N_in, const bool raytrace, const int src_redshift_index_in) : lens(lens_in), xmin(xmin_in), xmax(xmax_in), ymin(ymin_in), ymax(ymax_in), x_N(x_N_in), y_N(y_N_in), cartesian_srcgrid(NULL), delaunay_srcgrid(NULL)
{
	source_fit_mode = mode;
	ray_tracing_method = method;
	setup_pixel_arrays();
	setup_noise_map(lens_in);

	src_redshift_index = src_redshift_index_in;
	if (src_redshift_index == -1) {
		imggrid_zfactors = lens->reference_zfactors;
		imggrid_betafactors = lens->default_zsrc_beta_factors;
	} else {
		if (lens->extended_src_zfactors != NULL) {
			imggrid_zfactors = lens->extended_src_zfactors[src_redshift_index];
			imggrid_betafactors = lens->extended_src_beta_factors[src_redshift_index];
		} else {
			imggrid_zfactors = NULL;
			imggrid_betafactors = NULL;
		}
	}

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
	if (raytrace) {
#ifdef USE_OPENMP
		double wtime0, wtime;
		if (lens->show_wtime) {
			wtime0 = omp_get_wtime();
		}
#endif
		setup_ray_tracing_arrays();
		if ((lens->nlens > 0) and (imggrid_zfactors != NULL)) calculate_sourcepts_and_areas(true);
#ifdef USE_OPENMP
		if (lens->show_wtime) {
			wtime = omp_get_wtime() - wtime0;
			if (lens->mpi_id==0) cout << "Wall time for creating and ray-tracing image pixel grid: " << wtime << endl;
		}
#endif
	}
}

/*
ImagePixelGrid::ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, double** sb_in, const int x_N_in, const int y_N_in, const int reduce_factor, double xmin_in, double xmax_in, double ymin_in, double ymax_in, const int src_redshift_index_in) : lens(lens_in), xmin(xmin_in), xmax(xmax_in), ymin(ymin_in), ymax(ymax_in), cartesian_srcgrid(NULL), delaunay_srcgrid(NULL)
{
	// I think this constructor is only used for the "reduce" option, which I never use anymore. Get rid of this option (and constructor) altogether?
	lens = lens_in;
	source_fit_mode = mode;
	ray_tracing_method = method;

	x_N = x_N_in / reduce_factor;
	y_N = y_N_in / reduce_factor;

	setup_pixel_arrays();
	setup_noise_map(lens_in);

	src_redshift_index = src_redshift_index_in;
	if (src_redshift_index == -1) {
		imggrid_zfactors = lens->reference_zfactors;
		imggrid_betafactors = lens->default_zsrc_beta_factors;
	} else {
		if (lens->extended_src_zfactors != NULL) {
			imggrid_zfactors = lens->extended_src_zfactors[src_redshift_index];
			imggrid_betafactors = lens->extended_src_beta_factors[src_redshift_index];
		} else {
			imggrid_zfactors = NULL;
			imggrid_betafactors = NULL;
		}
	}

	pixel_xlength = (xmax-xmin)/x_N;
	pixel_ylength = (ymax-ymin)/y_N;
	pixel_area = pixel_xlength*pixel_ylength;
	triangle_area = 0.5*pixel_xlength*pixel_ylength;

	int i,j;
	double x,y;
	int ii,jj;
	for (j=0; j <= y_N; j++) {
		y = ymin + j*pixel_ylength;
		for (i=0; i <= x_N; i++) {
			x = xmin + i*pixel_xlength;
			if ((i < x_N) and (j < y_N)) {
				center_pts[i][j][0] = x + 0.5*pixel_xlength;
				center_pts[i][j][1] = y + 0.5*pixel_ylength;
				//if (lens->nlens > 0) lens->find_sourcept(center_pts[i][j],center_sourcepts[i][j],0,imggrid_zfactors,imggrid_betafactors);
			}
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
			//if (lens->nlens > 0) lens->find_sourcept(corner_pts[i][j],corner_sourcepts[i][j],0,imggrid_zfactors,imggrid_betafactors);
			if ((i < x_N) and (j < y_N)) {
				surface_brightness[i][j] = 0;
				for (ii=0; ii < reduce_factor; ii++) {
					for (jj=0; jj < reduce_factor; jj++) {
						surface_brightness[i][j] += sb_in[i*reduce_factor+ii][j*reduce_factor+jj];
					}
				}
				surface_brightness[i][j] /= SQR(reduce_factor);
			}
		}
	}
	mask = NULL;
	emask = NULL;
	setup_ray_tracing_arrays();
	if ((lens->nlens > 0) and (imggrid_zfactors != NULL)) calculate_sourcepts_and_areas(true);
}
*/

ImagePixelGrid::ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, ImagePixelData& pixel_data, const bool include_extended_mask, const int src_redshift_index_in, const int mask_index, const bool setup_mask_and_data, const bool verbal) : cartesian_srcgrid(NULL), delaunay_srcgrid(NULL)
{
	// with this constructor, we create the arrays but don't actually make any lensing calculations, since these will be done during each likelihood evaluation
	lens = lens_in;
	source_fit_mode = mode;
	ray_tracing_method = method;
	pixel_data.get_grid_params(xmin,xmax,ymin,ymax,x_N,y_N);
	src_xmin = -1e30; src_xmax = 1e30;
	src_ymin = -1e30; src_ymax = 1e30;

	setup_pixel_arrays();

	src_redshift_index = src_redshift_index_in;
	//cout << "src_i=" << src_redshift_index << " n_src_redshifts=" << lens->n_extended_src_redshifts << endl;
	if ((src_redshift_index == -1) or (lens->n_lens_redshifts==0)) {
		imggrid_zfactors = lens->reference_zfactors;
		imggrid_betafactors = lens->default_zsrc_beta_factors;
	} else {
		if (src_redshift_index >= lens->n_extended_src_redshifts) die("invalid extended source redshift index (%i)",src_redshift_index);
		if (lens->extended_src_zfactors == NULL) die("extended_src_zfactors is NULL");
		if (lens->extended_src_beta_factors == NULL) die("extended_src_beta_factors is NULL");
		imggrid_zfactors = lens->extended_src_zfactors[src_redshift_index];
		imggrid_betafactors = lens->extended_src_beta_factors[src_redshift_index];
	}

	pixel_xlength = (xmax-xmin)/x_N;
	pixel_ylength = (ymax-ymin)/y_N;
	max_sb = -1e30;
	pixel_area = pixel_xlength*pixel_ylength;
	triangle_area = 0.5*pixel_xlength*pixel_ylength;

	if (setup_mask_and_data) {
		if (mask_index >= pixel_data.n_masks) die("image_pixel_grid initialized with mask index that doesn't correspond to a mask that has been created/loaded (mask_i=%i,n_masks=%i)",mask_index,pixel_data.n_masks);
		mask = pixel_data.in_mask[mask_index];
		emask = pixel_data.extended_mask[mask_index];
	} else {
		mask = NULL;
		emask = NULL;
	}

	int i,j;
	double x,y;
	for (j=0; j <= y_N; j++) {
		y = pixel_data.yvals[j];
		for (i=0; i <= x_N; i++) {
			x = pixel_data.xvals[i];
			if ((i < x_N) and (j < y_N)) {
				center_pts[i][j][0] = x + 0.5*pixel_xlength;
				center_pts[i][j][1] = y + 0.5*pixel_ylength;
				if (setup_mask_and_data) {
					surface_brightness[i][j] = pixel_data.surface_brightness[i][j];
					noise_map[i][j] = pixel_data.noise_map[i][j];
					if (!include_extended_mask) pixel_in_mask[i][j] = pixel_data.in_mask[mask_index][i][j];
					else pixel_in_mask[i][j] = pixel_data.extended_mask[mask_index][i][j];
					if (surface_brightness[i][j] > max_sb) max_sb=surface_brightness[i][j];
				}
			}
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
		}
	}
	if (setup_mask_and_data) setup_ray_tracing_arrays(verbal);
}

void ImagePixelGrid::setup_noise_map(QLens* lens_in)
{
	int i,j;
	if ((lens_in->use_noise_map) and (lens_in->image_pixel_data)) {
		double **nptr = lens_in->image_pixel_data->noise_map;
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
	if ((source_fit_mode==Cartesian_Source) or (source_fit_mode==Delaunay_Source)) n_active_pixels = 0;
	else n_active_pixels = xy_N;

	corner_pts = new lensvector*[x_N+1];
	corner_sourcepts = new lensvector*[x_N+1];
	center_pts = new lensvector*[x_N];
	center_sourcepts = new lensvector*[x_N];
	pixel_in_mask = new bool*[x_N];
	maps_to_source_pixel = new bool*[x_N];
	pixel_index = new int*[x_N];
	pixel_index_fgmask = new int*[x_N];
	mapped_cartesian_srcpixels = new vector<SourcePixel*>*[x_N];
	mapped_delaunay_srcpixels = new vector<PtsWgts>*[x_N];
	n_mapped_srcpixels = new int**[x_N];
	surface_brightness = new double*[x_N];
	foreground_surface_brightness = new double*[x_N];
	noise_map = new double*[x_N];
	source_plane_triangle1_area = new double*[x_N];
	source_plane_triangle2_area = new double*[x_N];
	pixel_mag = new double*[x_N];
	max_nsplit = imax(8,lens->default_imgpixel_nsplit);
	//max_nsplit = lens->default_imgpixel_nsplit;
	nsplits = new int*[x_N];
	subpixel_maps_to_srcpixel = new bool**[x_N];
	subpixel_center_pts = new lensvector**[x_N];
	subpixel_center_sourcepts = new lensvector**[x_N];
	subpixel_surface_brightness = new double**[x_N];
	subpixel_weights = new double**[x_N];
	subpixel_index = new int*[x_N*max_nsplit];
	twist_pts = new lensvector*[x_N];
	twist_status = new int*[x_N];

	int i,j,k;
	for (i=0; i <= x_N; i++) {
		corner_pts[i] = new lensvector[y_N+1];
		corner_sourcepts[i] = new lensvector[y_N+1];
	}
	for (i=0; i < x_N; i++) {
		center_pts[i] = new lensvector[y_N];
		center_sourcepts[i] = new lensvector[y_N];
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
		mapped_cartesian_srcpixels[i] = new vector<SourcePixel*>[y_N];
		mapped_delaunay_srcpixels[i] = new vector<PtsWgts>[y_N];
		n_mapped_srcpixels[i] = new int*[y_N];
		subpixel_maps_to_srcpixel[i] = new bool*[y_N];
		subpixel_center_pts[i] = new lensvector*[y_N];
		subpixel_center_sourcepts[i] = new lensvector*[y_N];
		subpixel_surface_brightness[i] = new double*[y_N];
		subpixel_weights[i] = new double*[y_N];
		nsplits[i] = new int[y_N];
		twist_pts[i] = new lensvector[y_N];
		twist_status[i] = new int[y_N];
		for (j=0; j < y_N; j++) {
			surface_brightness[i][j] = 0;
			foreground_surface_brightness[i][j] = 0;
			noise_map[i][j] = 0;
			pixel_in_mask[i][j] = true; // default, since no mask has been introduced yet
			if ((source_fit_mode==Parameterized_Source) or (source_fit_mode==Shapelet_Source)) maps_to_source_pixel[i][j] = true; // in this mode you can always get a surface brightness for any image pixel
			else if (source_fit_mode==Delaunay_Source) maps_to_source_pixel[i][j] = true; // JUST A HACK FOR THE MOMENT--DELETE THIS ONCE THE IMAGE MAPPING FLAGS ARE WORKED OUT FOR THE DELAUNAY GRID
			//nsplits[i][j] = lens_in->default_imgpixel_nsplit; // default
			subpixel_maps_to_srcpixel[i][j] = new bool[max_nsplit*max_nsplit];
			subpixel_center_pts[i][j] = new lensvector[max_nsplit*max_nsplit];
			subpixel_center_sourcepts[i][j] = new lensvector[max_nsplit*max_nsplit];
			subpixel_surface_brightness[i][j] = new double[max_nsplit*max_nsplit];
			subpixel_weights[i][j] = new double[max_nsplit*max_nsplit];
			n_mapped_srcpixels[i][j] = new int[max_nsplit*max_nsplit];
			for (k=0; k < max_nsplit*max_nsplit; k++) subpixel_maps_to_srcpixel[i][j][k] = false;
			nsplits[i][j] = 1.0;
		}
	}
	int max_subpixel_nx = x_N*max_nsplit;
	int max_subpixel_ny = y_N*max_nsplit;
	for (i=0; i < max_subpixel_nx; i++) {
		subpixel_index[i] = new int[max_subpixel_ny];
	}
	set_null_ray_tracing_arrays();

	extended_mask_subcell_i = NULL;
	extended_mask_subcell_j = NULL;
	extended_mask_subcell_index = NULL;
	defx_subpixel_centers = NULL;
	defy_subpixel_centers = NULL;

	active_image_pixel_i = NULL;
	active_image_pixel_j = NULL;
	active_image_subpixel_ii = NULL;
	active_image_subpixel_jj = NULL;
	active_image_pixel_i_ss = NULL;
	active_image_pixel_j_ss = NULL;
	active_image_subpixel_ss = NULL;
	image_pixel_i_from_subcell_ii = NULL;
	image_pixel_j_from_subcell_jj = NULL;
	active_image_pixel_i_fgmask = NULL;
	active_image_pixel_j_fgmask= NULL;

	fft_convolution_is_setup = false;
}

void ImagePixelGrid::set_null_ray_tracing_arrays()
{
	Lmatrix_src_npixels = 0;
	defx_corners = NULL;
	defy_corners = NULL;
	defx_centers = NULL;
	defy_centers = NULL;
	area_tri1 = NULL;
	area_tri2 = NULL;
	twistx = NULL;
	twisty = NULL;
	twiststat = NULL;

	masked_pixels_i = NULL;
	masked_pixels_j = NULL;
	emask_pixels_i = NULL;
	emask_pixels_j = NULL;
	masked_pixel_corner_i = NULL;
	masked_pixel_corner_j = NULL;
	masked_pixel_corner = NULL;
	masked_pixel_corner_up = NULL;
	if (lens->split_imgpixels) {
		extended_mask_subcell_i = NULL;
		extended_mask_subcell_j = NULL;
		extended_mask_subcell_index = NULL;
		defx_subpixel_centers = NULL;
		defy_subpixel_centers = NULL;
	}
	ncvals = NULL;
}

void ImagePixelGrid::setup_ray_tracing_arrays(const bool verbal)
{
	int i,j,k,n,n_cell,n_corner;

	if ((!pixel_in_mask) or (emask == NULL)) {
		ntot_cells = x_N*y_N;
		ntot_cells_emask = x_N*y_N;
		ntot_corners = (x_N+1)*(y_N+1);
	} else {
		ntot_cells = 0;
		ntot_cells_emask = 0;
		ntot_corners = 0;
		for (i=0; i < x_N+1; i++) {
			for (j=0; j < y_N+1; j++) {
				if ((i < x_N) and (j < y_N) and (pixel_in_mask[i][j])) ntot_cells++;
				if (((i < x_N) and (j < y_N) and (pixel_in_mask[i][j])) or ((j < y_N) and (i > 0) and (pixel_in_mask[i-1][j])) or ((i < x_N) and (j > 0) and (pixel_in_mask[i][j-1])) or ((i > 0) and (j > 0) and (pixel_in_mask[i-1][j-1]))) {
					ntot_corners++;
				}
			}
		}

		for (i=0; i < x_N+1; i++) {
			for (j=0; j < y_N+1; j++) {
				if ((i < x_N) and (j < y_N) and ((pixel_in_mask[i][j]) or (emask[i][j]))) ntot_cells_emask++;
			}
		}

	}

	if (defx_corners != NULL) delete_ray_tracing_arrays();
	// The following is used for the ray tracing
	defx_corners = new double[ntot_corners];
	defy_corners = new double[ntot_corners];
	defx_centers = new double[ntot_cells_emask];
	defy_centers = new double[ntot_cells_emask];
	area_tri1 = new double[ntot_cells];
	area_tri2 = new double[ntot_cells];
	twistx = new double[ntot_cells];
	twisty = new double[ntot_cells];
	twiststat = new int[ntot_cells];

	masked_pixels_i = new int[ntot_cells];
	masked_pixels_j = new int[ntot_cells];
	emask_pixels_i = new int[ntot_cells_emask];
	emask_pixels_j = new int[ntot_cells_emask];
	masked_pixel_corner_i = new int[ntot_corners];
	masked_pixel_corner_j = new int[ntot_corners];
	masked_pixel_corner = new int[ntot_cells];
	masked_pixel_corner_up = new int[ntot_cells];
	ncvals = new int*[x_N+1];
	for (i=0; i < x_N+1; i++) ncvals[i] = new int[y_N+1];

	
	n_cell=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!pixel_in_mask) or (pixel_in_mask[i][j])) {
				masked_pixels_i[n_cell] = i;
				masked_pixels_j[n_cell] = j;
				n_cell++;
			}
		}
	}

	n_cell=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!pixel_in_mask) or (emask == NULL) or (pixel_in_mask[i][j]) or (emask[i][j])) {
				emask_pixels_i[n_cell] = i;
				emask_pixels_j[n_cell] = j;
				n_cell++;
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
				//if (((i < x_N) and (j < y_N) and (lens->image_pixel_data->extended_mask[i][j])) or ((j < y_N) and (i > 0) and (lens->image_pixel_data->extended_mask[i-1][j])) or ((i < x_N) and (j > 0) and (lens->image_pixel_data->extended_mask[i][j-1])) or ((i > 0) and (j > 0) and (lens->image_pixel_data->extended_mask[i-1][j-1]))) {
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
	for (int n=0; n < ntot_cells; n++) {
		i = masked_pixels_i[n];
		j = masked_pixels_j[n];
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
	//if (lens->image_pixel_data) {
		//for (i=0; i < x_N; i++) {
			//for (j=0; j < y_N; j++) {
				//if (lens->image_pixel_data->in_mask[i][j]) {
					//double r = sqrt(SQR(center_pts[i][j][0]) + SQR(center_pts[i][j][1]));
					//if (r < mask_min_r) mask_min_r = r;
				//}
			//}
		//}
	//}
	//if (lens->mpi_id==0) cout << "HACK: mask_min_r=" << mask_min_r << endl;

	int nsubpix = INTSQR(lens->default_imgpixel_nsplit);
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			for (k=0; k < nsubpix; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
			}
		}
	}
	int nsplit = (lens->split_high_mag_imgpixels) ? 1 : lens->default_imgpixel_nsplit;
	int emask_nsplit = (lens->split_high_mag_imgpixels) ? 1 : lens->emask_imgpixel_nsplit;
	set_nsplits(nsplit,emask_nsplit,lens->split_imgpixels);

	ntot_subpixels = 0;

	extended_mask_subcell_i = NULL;
	extended_mask_subcell_j = NULL;
	extended_mask_subcell_index = NULL;
	defx_subpixel_centers = NULL;
	defy_subpixel_centers = NULL;

	//if ((lens->split_imgpixels) and (!lens->split_high_mag_imgpixels)) { 
	if (lens->split_imgpixels) { 
		// if split_high_mag_imgpixels is on, this part will be deferred until after the ray-traced pixel areas have been
		// calculated (to get magnifications to use as criterion on whether to split or nit)
		setup_subpixel_ray_tracing_arrays(verbal);
	}
}

void ImagePixelGrid::setup_subpixel_ray_tracing_arrays(const bool verbal)
{
	ntot_subpixels = 0;
	int i,j;
	int nsplitpix = 0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!pixel_in_mask) or (pixel_in_mask[i][j]) or (lens->image_pixel_data == NULL) or (emask == NULL) or (emask[i][j])) {
				ntot_subpixels += INTSQR(nsplits[i][j]);
				if ((verbal) and (nsplits[i][j] > 1)) nsplitpix++;
			}
		}
	}
	if ((verbal) and (lens->mpi_id==0)) {
		cout << "Number of split image pixels (including emask): " << nsplitpix << endl;
		cout << "Total number of image pixels/subpixels (including emask): " << ntot_subpixels << endl;
	}

	if (extended_mask_subcell_i != NULL) delete[] extended_mask_subcell_i;
	if (extended_mask_subcell_j != NULL) delete[] extended_mask_subcell_j;
	if (extended_mask_subcell_index != NULL) delete[] extended_mask_subcell_index;

	extended_mask_subcell_i = new int[ntot_subpixels];
	extended_mask_subcell_j = new int[ntot_subpixels];
	extended_mask_subcell_index = new int[ntot_subpixels];

	int n_subpixel = 0;
	int k;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!pixel_in_mask) or (pixel_in_mask[i][j]) or (lens->image_pixel_data == NULL) or ((emask==NULL) or (emask[i][j]))) {
				for (k=0; k < INTSQR(nsplits[i][j]); k++) {
					extended_mask_subcell_i[n_subpixel] = i;
					extended_mask_subcell_j[n_subpixel] = j;
					extended_mask_subcell_index[n_subpixel] = k;
					n_subpixel++;
				}
			}
		}
	}

	if (defx_subpixel_centers != NULL) delete[] defx_subpixel_centers;
	if (defy_subpixel_centers != NULL) delete[] defy_subpixel_centers;
	defx_subpixel_centers = new double[ntot_subpixels];
	defy_subpixel_centers = new double[ntot_subpixels];
}

void ImagePixelGrid::delete_ray_tracing_arrays()
{
	if (defx_corners != NULL) delete[] defx_corners;
	if (defy_corners != NULL) delete[] defy_corners;
	if (defx_centers != NULL) delete[] defx_centers;
	if (defy_centers != NULL) delete[] defy_centers;
	if (area_tri1 != NULL) delete[] area_tri1;
	if (area_tri2 != NULL) delete[] area_tri2;
	if (twistx != NULL) delete[] twistx;
	if (twisty != NULL) delete[] twisty;
	if (twiststat != NULL) delete[] twiststat;

	if (masked_pixels_i != NULL) delete[] masked_pixels_i;
	if (masked_pixels_j != NULL) delete[] masked_pixels_j;
	if (emask_pixels_i != NULL) delete[] emask_pixels_i;
	if (emask_pixels_j != NULL) delete[] emask_pixels_j;
	if (masked_pixel_corner_i != NULL) delete[] masked_pixel_corner_i;
	if (masked_pixel_corner_j != NULL) delete[] masked_pixel_corner_j;
	if (masked_pixel_corner != NULL) delete[] masked_pixel_corner;
	if (masked_pixel_corner_up != NULL) delete[] masked_pixel_corner_up;
	if (lens->split_imgpixels) {
		if (extended_mask_subcell_i != NULL) delete[] extended_mask_subcell_i;
		if (extended_mask_subcell_j != NULL) delete[] extended_mask_subcell_j;
		if (extended_mask_subcell_index != NULL) delete[] extended_mask_subcell_index;
		if (defx_subpixel_centers != NULL) delete[] defx_subpixel_centers;
		if (defx_subpixel_centers != NULL) delete[] defy_subpixel_centers;
	}
	if (ncvals != NULL) {
		for (int i=0; i < x_N+1; i++) delete[] ncvals[i];
		delete[] ncvals;
	}
	if (fft_convolution_is_setup) cleanup_FFT_convolution_arrays();
	set_null_ray_tracing_arrays();
}

inline bool ImagePixelGrid::test_if_between(const double& p, const double& a, const double& b)
{
	if ((b>a) and (p>a) and (p<b)) return true;
	else if ((a>b) and (p>b) and (p<a)) return true;
	return false;
}

void ImagePixelGrid::calculate_sourcepts_and_areas(const bool raytrace_pixel_centers, const bool verbal)
{
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*(lens->group_comm), *(lens->mpi_group), &sub_comm);
#endif

	//long int ntot_cells_check = 0;
	//long int ntot_corners_check = 0;
	//for (i=0; i < x_N+1; i++) {
		//for (j=0; j < y_N+1; j++) {
			//if ((i < x_N) and (j < y_N) and (lens->image_pixel_data->in_mask[i][j])) ntot_cells_check++;
			//if (((i < x_N) and (j < y_N) and (lens->image_pixel_data->in_mask[i][j])) or ((j < y_N) and (i > 0) and (lens->image_pixel_data->in_mask[i-1][j])) or ((i < x_N) and (j > 0) and (lens->image_pixel_data->in_mask[i][j-1])) or ((i > 0) and (j > 0) and (lens->image_pixel_data->in_mask[i-1][j-1]))) {
			////if ((i < x_N) and (j < y_N) and (lens->image_pixel_data->extended_mask[i][j])) ntot_cells_check++;
			////if (((i < x_N) and (j < y_N) and (lens->image_pixel_data->extended_mask[i][j])) or ((j < y_N) and (i > 0) and (lens->image_pixel_data->extended_mask[i-1][j])) or ((i < x_N) and (j > 0) and (lens->image_pixel_data->extended_mask[i][j-1])) or ((i > 0) and (j > 0) and (lens->image_pixel_data->extended_mask[i-1][j-1]))) {
				//ntot_corners_check++;
			//}
		//}
	//}
	//if (ntot_cells_check != ntot_cells) die("ntot_cells does not equal the value assigned when image grid created");
	//if (ntot_corners_check != ntot_corners) die("ntot_corners does not equal the value assigned when image grid created");

	int i,j,k,n,n_cell,n_corner,n_yp;

	int mpi_chunk, mpi_start, mpi_end;
	mpi_chunk = ntot_corners / lens->group_np;
	mpi_start = lens->group_id*mpi_chunk;
	if (lens->group_id == lens->group_np-1) mpi_chunk += (ntot_corners % lens->group_np); // assign the remainder elements to the last mpi process
	mpi_end = mpi_start + mpi_chunk;

	int mpi_chunk2, mpi_start2, mpi_end2;
	mpi_chunk2 = ntot_cells / lens->group_np;
	mpi_start2 = lens->group_id*mpi_chunk2;
	if (lens->group_id == lens->group_np-1) mpi_chunk2 += (ntot_cells % lens->group_np); // assign the remainder elements to the last mpi process
	mpi_end2 = mpi_start2 + mpi_chunk2;

	int mpi_chunk4, mpi_start4, mpi_end4;
	mpi_chunk4 = ntot_cells_emask / lens->group_np;
	mpi_start4 = lens->group_id*mpi_chunk4;
	if (lens->group_id == lens->group_np-1) mpi_chunk4 += (ntot_cells_emask % lens->group_np); // assign the remainder elements to the last mpi process
	mpi_end4 = mpi_start4 + mpi_chunk4;

	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		lensvector d1,d2,d3,d4;
		//int ii,jj;
		#pragma omp for private(n,i,j) schedule(dynamic)
		for (n=mpi_start; n < mpi_end; n++) {
			//j = n / (x_N+1);
			//i = n % (x_N+1);
			j = masked_pixel_corner_j[n];
			i = masked_pixel_corner_i[n];
			//cout << i << " " << j << " " << n << " " << ntot_corners << " " << mpi_end << endl;
			lens->find_sourcept(corner_pts[i][j],defx_corners[n],defy_corners[n],thread,imggrid_zfactors,imggrid_betafactors);
		}
#ifdef USE_MPI
		#pragma omp master
		{
			if (lens->group_np > 1) {
				int id, chunk, start;
				for (id=0; id < lens->group_np; id++) {
					chunk = ntot_corners / lens->group_np;
					start = id*chunk;
					if (id == lens->group_np-1) chunk += (ntot_corners % lens->group_np); // assign the remainder elements to the last mpi process
					MPI_Bcast(defx_corners+start,chunk,MPI_DOUBLE,id,sub_comm);
					MPI_Bcast(defy_corners+start,chunk,MPI_DOUBLE,id,sub_comm);
				}
			}
		}
		#pragma omp barrier
#endif
		#pragma omp for private(n_cell,i,j,n,n_yp) schedule(dynamic)
		for (n_cell=mpi_start2; n_cell < mpi_end2; n_cell++) {
			j = masked_pixels_j[n_cell];
			i = masked_pixels_i[n_cell];

			//n = j*(x_N+1)+i;
			//n_yp = (j+1)*(x_N+1)+i;
			n = masked_pixel_corner[n_cell];
			n_yp = masked_pixel_corner_up[n_cell];
			d1[0] = defx_corners[n] - defx_corners[n+1];
			d1[1] = defy_corners[n] - defy_corners[n+1];
			d2[0] = defx_corners[n_yp] - defx_corners[n];
			d2[1] = defy_corners[n_yp] - defy_corners[n];
			d3[0] = defx_corners[n_yp+1] - defx_corners[n_yp];
			d3[1] = defy_corners[n_yp+1] - defy_corners[n_yp];
			d4[0] = defx_corners[n+1] - defx_corners[n_yp+1];
			d4[1] = defy_corners[n+1] - defy_corners[n_yp+1];

			twiststat[n_cell] = 0;
			double xa,ya,xb,yb,xc,yc,xd,yd,slope1,slope2;
			xa=defx_corners[n];
			ya=defy_corners[n];
			xb=defx_corners[n_yp];
			yb=defy_corners[n_yp];
			xc=defx_corners[n_yp+1];
			yc=defy_corners[n_yp+1];
			xd=defx_corners[n+1];
			yd=defy_corners[n+1];
			slope1 = (yb-ya)/(xb-xa);
			slope2 = (yc-yd)/(xc-xd);
			twistx[n_cell] = (yd-ya+xa*slope1-xd*slope2)/(slope1-slope2);
			twisty[n_cell] = (twistx[n_cell]-xa)*slope1+ya;
			if ((test_if_between(twistx[n_cell],xa,xb)) and (test_if_between(twisty[n_cell],ya,yb)) and (test_if_between(twistx[n_cell],xc,xd)) and (test_if_between(twisty[n_cell],yc,yd))) {
				twiststat[n_cell] = 1;
				d2[0] = twistx[n_cell] - defx_corners[n];
				d2[1] = twisty[n_cell] - defy_corners[n];
				d4[0] = twistx[n_cell] - defx_corners[n_yp+1];
				d4[1] = twisty[n_cell] - defy_corners[n_yp+1];
			} else {
				slope1 = (yd-ya)/(xd-xa);
				slope2 = (yc-yb)/(xc-xb);
				twistx[n_cell] = (yb-ya+xa*slope1-xb*slope2)/(slope1-slope2);
				twisty[n_cell] = (twistx[n_cell]-xa)*slope1+ya;
				if ((test_if_between(twistx[n_cell],xa,xd)) and (test_if_between(twisty[n_cell],ya,yd)) and (test_if_between(twistx[n_cell],xb,xc)) and (test_if_between(twisty[n_cell],yb,yc))) {
					twiststat[n_cell] = 2;
					d1[0] = defx_corners[n] - twistx[n_cell];
					d1[1] = defy_corners[n] - twisty[n_cell];
					d3[0] = defx_corners[n_yp+1] - twistx[n_cell];
					d3[1] = defy_corners[n_yp+1] - twisty[n_cell];
				}
			}

			area_tri1[n_cell] = 0.5*abs(d1 ^ d2);
			area_tri2[n_cell] = 0.5*abs(d3 ^ d4);
		}

		if ((!lens->split_imgpixels) or (raytrace_pixel_centers)) {
			#pragma omp for private(n_cell,i,j,n,n_yp) schedule(dynamic)
			for (n_cell=mpi_start4; n_cell < mpi_end4; n_cell++) {
				j = emask_pixels_j[n_cell];
				i = emask_pixels_i[n_cell];
				lens->find_sourcept(center_pts[i][j],defx_centers[n_cell],defy_centers[n_cell],thread,imggrid_zfactors,imggrid_betafactors);
			}
		}
	}

#ifdef USE_MPI
	if (lens->group_np > 1) {
		int id, chunk, start;
		int id2, chunk2, start2;
		for (id=0; id < lens->group_np; id++) {
			chunk = ntot_cells / lens->group_np;
			start = id*chunk;
			if (id == lens->group_np-1) chunk += (ntot_cells % lens->group_np); // assign the remainder elements to the last mpi process
			if ((!lens->split_imgpixels) or (raytrace_pixel_centers)) {
				chunk2 = ntot_cells_emask / lens->group_np;
				start2 = id*chunk2;
				MPI_Bcast(defx_centers+start2,chunk2,MPI_DOUBLE,id,sub_comm);
				MPI_Bcast(defy_centers+start2,chunk2,MPI_DOUBLE,id,sub_comm);
			}
			MPI_Bcast(area_tri1+start,chunk,MPI_DOUBLE,id,sub_comm);
			MPI_Bcast(area_tri2+start,chunk,MPI_DOUBLE,id,sub_comm);
			MPI_Bcast(twistx+start,chunk,MPI_DOUBLE,id,sub_comm);
			MPI_Bcast(twisty+start,chunk,MPI_DOUBLE,id,sub_comm);
			MPI_Bcast(twiststat+start,chunk,MPI_INT,id,sub_comm);
		}
	}
#endif
	src_xmin = 1e30; src_xmax = -1e30;
	src_ymin = 1e30; src_ymax = -1e30;
	for (n=0; n < ntot_corners; n++) {
		//j = n / (x_N+1);
		//i = n % (x_N+1);
		j = masked_pixel_corner_j[n];
		i = masked_pixel_corner_i[n];
		corner_sourcepts[i][j][0] = defx_corners[n];
		corner_sourcepts[i][j][1] = defy_corners[n];
		if (defx_corners[n] < src_xmin) src_xmin = defx_corners[n];
		if (defx_corners[n] > src_xmax) src_xmax = defx_corners[n];
		if (defy_corners[n] < src_ymin) src_ymin = defy_corners[n];
		if (defy_corners[n] > src_ymax) src_ymax = defy_corners[n];

		//wtf << corner_pts[i][j][0] << " " << corner_pts[i][j][1] << " " << corner_sourcepts[i][j][0] << " " << corner_sourcepts[i][j][1] << " " << endl;
	}
	//wtf.close();
	int ii,jj;
	double u0,w0,mag;
	int subcell_index;
	min_srcplane_area = 1e30;
	double pixel_srcplane_area;
	for (n=0; n < ntot_cells; n++) {
		//n_cell = j*x_N+i;
		j = masked_pixels_j[n];
		i = masked_pixels_i[n];
		source_plane_triangle1_area[i][j] = area_tri1[n];
		source_plane_triangle2_area[i][j] = area_tri2[n];
		pixel_srcplane_area = area_tri1[n] + area_tri2[n];
		pixel_mag[i][j] = pixel_area / pixel_srcplane_area;
		if (pixel_srcplane_area < min_srcplane_area) min_srcplane_area = pixel_srcplane_area;
		//if (i==176) cout << "AREAS (" << i << "," << j << "): " << area_tri1[n] << " " << area_tri2[n] << endl;
		twist_pts[i][j][0] = twistx[n];
		twist_pts[i][j][1] = twisty[n];
		twist_status[i][j] = twiststat[n];
		if (lens->split_high_mag_imgpixels) {
			mag = pixel_area/(area_tri1[n]+area_tri2[n]);
			//cout << "TRYING " << mag << " " << lens->image_pixel_data->surface_brightness[i][j] << endl;
			if ((mask[i][j]) and ((mag > lens->imgpixel_himag_threshold) or (mag < lens->imgpixel_lomag_threshold)) and ((lens->image_pixel_data == NULL) or (lens->image_pixel_data->surface_brightness[i][j] > lens->imgpixel_sb_threshold))) {
				nsplits[i][j] = lens->default_imgpixel_nsplit;
				subcell_index = 0;
				for (ii=0; ii < nsplits[i][j]; ii++) {
					for (jj=0; jj < nsplits[i][j]; jj++) {
						u0 = ((double) (1+2*ii))/(2*nsplits[i][j]);
						w0 = ((double) (1+2*jj))/(2*nsplits[i][j]);
						subpixel_center_pts[i][j][subcell_index][0] = (1-u0)*corner_pts[i][j][0] + u0*corner_pts[i+1][j][0];
						subpixel_center_pts[i][j][subcell_index][1] = (1-w0)*corner_pts[i][j][1] + w0*corner_pts[i][j+1][1];
						//subpixel_center_pts[i][j][subcell_index][0] = u0*corner_pts[i][j][0] + (1-u0)*corner_pts[i+1][j][0];
						//subpixel_center_pts[i][j][subcell_index][1] = w0*corner_pts[i][j][1] + (1-w0)*corner_pts[i][j+1][1];
						subcell_index++;
					}
				}
				//cout << "Setting nsplit=" << lens->default_imgpixel_nsplit << " for pixel " << i << " " << j << " (sb=" << lens->image_pixel_data->surface_brightness[i][j] << ")" << endl;
			//} else {
				//cout << "NOPE, mag=" << mag << " sb=" << lens->image_pixel_data->surface_brightness[i][j] << " nsplit=" << nsplits[i][j] << endl; 
			} else {
				nsplits[i][j] = 1;
			}
		}
	}
	if ((lens->split_imgpixels) and (lens->split_high_mag_imgpixels)) setup_subpixel_ray_tracing_arrays(verbal);

	int mpi_chunk3, mpi_start3, mpi_end3;
	mpi_chunk3 = ntot_subpixels / lens->group_np;
	mpi_start3 = lens->group_id*mpi_chunk3;
	if (lens->group_id == lens->group_np-1) mpi_chunk3 += (ntot_subpixels % lens->group_np); // assign the remainder elements to the last mpi process
	mpi_end3 = mpi_start3 + mpi_chunk3;

	if (lens->split_imgpixels) {
		int n_subcell;
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif

			#pragma omp for private(i,j,k,n_subcell) schedule(dynamic)
			for (n_subcell=mpi_start3; n_subcell < mpi_end3; n_subcell++) {
				j = extended_mask_subcell_j[n_subcell];
				i = extended_mask_subcell_i[n_subcell];
				k = extended_mask_subcell_index[n_subcell];
				lens->find_sourcept(subpixel_center_pts[i][j][k],defx_subpixel_centers[n_subcell],defy_subpixel_centers[n_subcell],thread,imggrid_zfactors,imggrid_betafactors);
				//if (defx_subpixel_centers[n_subcell]*0.0 != 0.0) die("nonsense value for deflection (x=%g y=%g defx=%g defy=%g)",subpixel_center_pts[i][j][k][0],subpixel_center_pts[i][j][k][1],defx_subpixel_centers[n_subcell],defy_subpixel_centers[n_subcell]);
			}
		}
	}
#ifdef USE_MPI
	if ((lens->group_np > 1) and (lens->split_imgpixels)) {
		int id, chunk, start;
		for (id=0; id < lens->group_np; id++) {
			chunk = ntot_subpixels / lens->group_np;
			start = id*chunk;
			if (id == lens->group_np-1) chunk += (ntot_subpixels % lens->group_np); // assign the remainder elements to the last mpi process
			MPI_Bcast(defx_subpixel_centers+start,chunk,MPI_DOUBLE,id,sub_comm);
			MPI_Bcast(defy_subpixel_centers+start,chunk,MPI_DOUBLE,id,sub_comm);
		}
	}
	MPI_Comm_free(&sub_comm);
#endif
	if ((!lens->split_imgpixels) or (raytrace_pixel_centers)) {
		for (n=0; n < ntot_cells_emask; n++) {
			//n_cell = j*x_N+i;
			j = emask_pixels_j[n];
			i = emask_pixels_i[n];
			center_sourcepts[i][j][0] = defx_centers[n];
			center_sourcepts[i][j][1] = defy_centers[n];
		}
	}
	if (lens->split_imgpixels) {
		for (n=0; n < ntot_subpixels; n++) {
			j = extended_mask_subcell_j[n];
			i = extended_mask_subcell_i[n];
			k = extended_mask_subcell_index[n];
			//cout << "CHECKING2: " << defy_subpixel_centers[n] << " " << subpixel_center_sourcepts[i][j][k][1] << endl;
			//if (subpixel_center_sourcepts[i][j][k][0] != defx_subpixel_centers[n]) cout << "wrong defx: " << defx_subpixel_centers[n] << " " << subpixel_center_sourcepts[i][j][k][0] << endl;
			//if (subpixel_center_sourcepts[i][j][k][1] != defy_subpixel_centers[n]) cout << "wrong defy: " << defx_subpixel_centers[n] << " " << subpixel_center_sourcepts[i][j][k][1] << endl;
			subpixel_center_sourcepts[i][j][k][0] = defx_subpixel_centers[n];
			subpixel_center_sourcepts[i][j][k][1] = defy_subpixel_centers[n];
			//cout << "SRCPT: " << subpixel_center_sourcepts[i][j][k][0] << " " << subpixel_center_sourcepts[i][j][k][1] << endl;
		}
	}
}

void ImagePixelGrid::ray_trace_pixels()
{
	if (lens) {
		setup_ray_tracing_arrays();
		calculate_sourcepts_and_areas(true);
	}
}

void ImagePixelGrid::redo_lensing_calculations(const bool verbal)
{
#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	if ((source_fit_mode==Cartesian_Source) or (source_fit_mode==Delaunay_Source)) n_active_pixels = 0;
	//delete_ray_tracing_arrays();
	//setup_pixel_arrays();
	//setup_ray_tracing_arrays();
	calculate_sourcepts_and_areas(true,verbal);

#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for ray-tracing image pixel grid: " << wtime << endl;
	}
#endif
}

void ImagePixelGrid::redo_lensing_calculations_corners() // this is used for analytic source mode with zooming when not using pixellated or shapelet sources
{
	// Update this so it uses the extended mask!! DO THIS!!!!!!!!
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*(lens->group_comm), *(lens->mpi_group), &sub_comm);
#endif

#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int i,j,n,n_cell,n_yp;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if (lens->split_imgpixels) nsplits[i][j] = lens->default_imgpixel_nsplit; // default
		}
	}
	long int ntot_corners = (x_N+1)*(y_N+1);
	long int ntot_cells = x_N*y_N;
	double *defx_corners, *defy_corners;
	defx_corners = new double[ntot_corners];
	defy_corners = new double[ntot_corners];

	int mpi_chunk, mpi_start, mpi_end;
	mpi_chunk = ntot_corners / lens->group_np;
	mpi_start = lens->group_id*mpi_chunk;
	if (lens->group_id == lens->group_np-1) mpi_chunk += (ntot_corners % lens->group_np); // assign the remainder elements to the last mpi process
	mpi_end = mpi_start + mpi_chunk;

	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		lensvector d1,d2,d3,d4;
		#pragma omp for private(n,i,j) schedule(dynamic)
		for (n=mpi_start; n < mpi_end; n++) {
			j = n / (x_N+1);
			i = n % (x_N+1);
			lens->find_sourcept(corner_pts[i][j],defx_corners[n],defy_corners[n],thread,imggrid_zfactors,imggrid_betafactors);
		}
	}
#ifdef USE_MPI
		#pragma omp master
		{
			int id, chunk, start;
			for (id=0; id < lens->group_np; id++) {
				chunk = ntot_corners / lens->group_np;
				start = id*chunk;
				if (id == lens->group_np-1) chunk += (ntot_corners % lens->group_np); // assign the remainder elements to the last mpi process
				MPI_Bcast(defx_corners+start,chunk,MPI_DOUBLE,id,sub_comm);
				MPI_Bcast(defy_corners+start,chunk,MPI_DOUBLE,id,sub_comm);
			}
		}
		#pragma omp barrier
#endif
	for (n=0; n < ntot_corners; n++) {
		j = n / (x_N+1);
		i = n % (x_N+1);
		corner_sourcepts[i][j][0] = defx_corners[n];
		corner_sourcepts[i][j][1] = defy_corners[n];
	}

#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for ray-tracing image pixel grid: " << wtime << endl;
	}
#endif
	delete[] defx_corners;
	delete[] defy_corners;
}

void ImagePixelGrid::load_data(ImagePixelData& pixel_data)
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

double ImagePixelGrid::plot_surface_brightness(string outfile_root, bool plot_residual, bool normalize_residuals, bool show_noise_thresh, bool plot_log)
{
	string sb_filename = outfile_root + ".dat";
	string x_filename = outfile_root + ".x";
	string y_filename = outfile_root + ".y";
	string src_filename = outfile_root + "_srcpts.dat";

	ofstream pixel_image_file; lens->open_output_file(pixel_image_file,sb_filename);
	ofstream pixel_xvals; lens->open_output_file(pixel_xvals,x_filename);
	ofstream pixel_yvals; lens->open_output_file(pixel_yvals,y_filename);
	ofstream pixel_src_file; lens->open_output_file(pixel_src_file,src_filename);
	pixel_image_file << setiosflags(ios::scientific);
	for (int i=0; i <= x_N; i++) {
		pixel_xvals << corner_pts[i][0][0] << endl;
	}
	for (int j=0; j <= y_N; j++) {
		pixel_yvals << corner_pts[0][j][1] << endl;
	}	
	int i,j;
	double residual, tot_residuals = 0;

	//ofstream wtfout("wtf2.dat");
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
				if (!plot_residual) {
					double sb = surface_brightness[i][j] + foreground_surface_brightness[i][j];
					//if (sb*0.0 != 0.0) die("WTF %g %g",surface_brightness[i][j],foreground_surface_brightness[i][j]);
					if (!plot_log) pixel_image_file << sb;
					else pixel_image_file << log(abs(sb));
				} else {
					double sb = surface_brightness[i][j] + foreground_surface_brightness[i][j];
					residual = lens->image_pixel_data->surface_brightness[i][j] - sb;
					if (normalize_residuals) {
						if (lens->use_noise_map) {
							if (lens->image_pixel_data != NULL) {
								residual /= lens->image_pixel_data->noise_map[i][j];
							} else warn("image pixel data not loaded; could not use noise map to plot residuals");
						} else {
							if (lens->background_pixel_noise > 0) residual /= lens->background_pixel_noise;
						}
					}
					tot_residuals += residual*residual;
					//wtfout << i << " " << j << " " << (residual*residual) << endl;
					if (show_noise_thresh) {
						if (abs(residual) >= lens->background_pixel_noise) pixel_image_file << residual;
						else pixel_image_file << "NaN";
					}
					else pixel_image_file << residual;
					//lens->find_sourcept(center_pts[i][j],center_sourcepts[i][j],0,imggrid_zfactors,imggrid_betafactors);
				}
				pixel_src_file << center_pts[i][j][0] << " " << center_pts[i][j][1] << " " << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " " << residual << endl;
			} else {
				pixel_image_file << "NaN";
			}
			if (i < x_N-1) pixel_image_file << " ";
		}
		pixel_image_file << endl;
	}
	//plot_sourcepts(outfile_root);
	return tot_residuals;
}

void ImagePixelGrid::plot_sourcepts(string outfile_root, const bool show_subpixels)
{
	string sp_filename = outfile_root + "_srcpts.dat";

	ofstream sourcepts_file; lens->open_output_file(sourcepts_file,sp_filename);
	sourcepts_file << setiosflags(ios::scientific);
	int i,j;
	double residual;

	int k,nsp;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			//if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
			if ((pixel_in_mask==NULL) or ((!lens->zero_sb_extended_mask_prior) and (emask) and (emask[i][j])) or ((lens->zero_sb_extended_mask_prior) and (mask) and (mask[i][j]))) {
				if ((!lens->split_imgpixels) or (!show_subpixels)) {
					sourcepts_file << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " " << center_pts[i][j][0] << " " << center_pts[i][j][1] << endl;
				} else {
					nsp = INTSQR(nsplits[i][j]);
					for (k=0; k < nsp; k++) {
						sourcepts_file << subpixel_center_sourcepts[i][j][k][0] << " " << subpixel_center_sourcepts[i][j][k][1] << " " << subpixel_center_pts[i][j][k][0] << " " << subpixel_center_pts[i][j][k][1] << endl;
					}
				}
			}
		}
	}
}

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
	if (lens->fit_output_dir != ".") lens->create_output_directory(); // in case it hasn't been created already
	string filename = lens->fit_output_dir + "/" + fits_filename;

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
						else pixels[i] = lens->image_pixel_data->surface_brightness[i][j] - surface_brightness[i][j] - foreground_surface_brightness[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] pixels;
			}
			if (pixel_xlength==pixel_ylength) {
				fits_write_key(outfptr, TDOUBLE, "PXSIZE", &pixel_xlength, "length of square pixels (in arcsec)", &status);
			} else {
				if (lens->mpi_id==0) cout << "NOTE: pixel length not equal in x- versus y-directions; not saving pixel size in FITS file header" << endl;
			}
			if ((lens->simulate_pixel_noise) and (!lens->use_noise_map))
				fits_write_key(outfptr, TDOUBLE, "PXNOISE", &lens->background_pixel_noise, "pixel surface brightness noise", &status);
			if ((lens->psf_width_x != 0) and (lens->psf_width_y==lens->psf_width_x) and (!lens->use_input_psf_matrix))
				fits_write_key(outfptr, TDOUBLE, "PSFSIG", &lens->psf_width_x, "Gaussian PSF width (dispersion, not FWHM)", &status);
			fits_write_key(outfptr, TDOUBLE, "ZSRC", &lens->source_redshift, "redshift of source galaxy", &status);
			if (lens->nlens > 0) {
				double zl = lens->lens_list[lens->primary_lens_number]->get_redshift();
				fits_write_key(outfptr, TDOUBLE, "ZLENS", &zl, "redshift of primary lens", &status);
			}

			if (lens->data_info != "") {
				string comment = "ql: " + lens->data_info;
				fits_write_comment(outfptr, comment.c_str(), &status);
			}
			if (lens->param_markers != "") {
				string param_markers_comma = lens->param_markers;
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

bool ImagePixelGrid::set_fit_window(ImagePixelData& pixel_data, const bool raytrace, const int mask_index)
{
	if ((x_N != pixel_data.npixels_x) or (y_N != pixel_data.npixels_y)) {
		warn("Number of data pixels does not match specified number of image pixels; cannot activate fit window");
		return false;
	}
	int i,j,k;
	if (pixel_in_mask==NULL) {
		pixel_in_mask = new bool*[x_N];
		for (i=0; i < x_N; i++) pixel_in_mask[i] = new bool[y_N];
	}
	int nsubpix = INTSQR(lens->default_imgpixel_nsplit);
	mask = pixel_data.in_mask[mask_index];
	emask = pixel_data.extended_mask[mask_index];
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			pixel_in_mask[i][j] = pixel_data.in_mask[mask_index][i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			for (k=0; k < nsubpix; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
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
	//if ((lens) and (lens->mpi_id==0)) cout << "HACK: mask_min_r=" << mask_min_r << endl;

	if (lens) {
		setup_ray_tracing_arrays();
		if ((raytrace) or (lens->split_high_mag_imgpixels)) calculate_sourcepts_and_areas(true);
	}
	return true;
}

void ImagePixelGrid::include_all_pixels()
{
	int i,j,k;
	if (pixel_in_mask==NULL) {
		pixel_in_mask = new bool*[x_N];
		for (i=0; i < x_N; i++) pixel_in_mask[i] = new bool[y_N];
	}
	int nsubpix = INTSQR(lens->default_imgpixel_nsplit);
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			pixel_in_mask[i][j] = true;
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			for (k=0; k < nsubpix; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
			}
		}
	}
	//mask = NULL;
	//emask = NULL;
	if (lens) setup_ray_tracing_arrays();
}

void ImagePixelGrid::activate_extended_mask()
{
	int i,j,k;
	if (pixel_in_mask==NULL) {
		pixel_in_mask = new bool*[x_N];
		for (i=0; i < x_N; i++) pixel_in_mask[i] = new bool[y_N];
	}
	int nsubpix = INTSQR(lens->default_imgpixel_nsplit);
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			pixel_in_mask[i][j] = emask[i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			for (k=0; k < nsubpix; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
			}
		}
	}
	if (lens) setup_ray_tracing_arrays();
}

void ImagePixelGrid::activate_foreground_mask()
{
	int i,j,k;
	if (pixel_in_mask==NULL) {
		pixel_in_mask = new bool*[x_N];
		for (i=0; i < x_N; i++) pixel_in_mask[i] = new bool[y_N];
	}
	int nsubpix = INTSQR(lens->default_imgpixel_nsplit);
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			pixel_in_mask[i][j] = lens->image_pixel_data->foreground_mask[i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			for (k=0; k < nsubpix; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
			}
		}
	}
	if (lens) setup_ray_tracing_arrays();
}

void ImagePixelGrid::deactivate_extended_mask()
{
	int i,j,k;
	//int n=0, m=0;
	int nsubpix = INTSQR(lens->default_imgpixel_nsplit);
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			pixel_in_mask[i][j] = mask[i][j];
			mapped_cartesian_srcpixels[i][j].clear();
			mapped_delaunay_srcpixels[i][j].clear();
			for (k=0; k < nsubpix; k++) {
				n_mapped_srcpixels[i][j][k] = 0;
			}

			//if (pixel_in_mask[i][j]) n++;
			//if (lens->image_pixel_data->extended_mask[i][j]) m++;
		}
	}
	//cout << "NFIT: " << n << endl;
	//cout << "NEXT: " << m << endl;
	if (lens) setup_ray_tracing_arrays();
}

void ImagePixelGrid::set_nsplits(const int default_nsplit, const int emask_nsplit, const bool split_pixels)
{
	int i,j,ii,jj,nsplit,subcell_index;
	double u0,w0;

	//int ii_check, jj_check;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if (split_pixels) {
				if (mask != NULL) {
					if (mask[i][j]) nsplit = default_nsplit;
					else nsplit = emask_nsplit;
				} else {
					nsplit = default_nsplit;
				}
				nsplits[i][j] = nsplit;
				subcell_index = 0;
				for (ii=0; ii < nsplit; ii++) {
					for (jj=0; jj < nsplit; jj++) {
						//ii_check = subcell_index / nsplit;
						//jj_check = subcell_index % nsplit;
						//cout << "IIJJ: " << ii << " " << ii_check << " " << jj << " " << jj_check << endl;
						//if (ii != ii_check) die("FUCK ii");
						//if (jj != jj_check) die("FUCK jj");

						u0 = ((double) (1+2*ii))/(2*nsplit);
						w0 = ((double) (1+2*jj))/(2*nsplit);
						subpixel_center_pts[i][j][subcell_index][0] = (1-u0)*corner_pts[i][j][0] + u0*corner_pts[i+1][j][0];
						subpixel_center_pts[i][j][subcell_index][1] = (1-w0)*corner_pts[i][j][1] + w0*corner_pts[i][j+1][1];

						//subpixel_center_pts[i][j][subcell_index][0] = u0*corner_pts[i][j][0] + (1-u0)*corner_pts[i+1][j][0];
						//subpixel_center_pts[i][j][subcell_index][1] = w0*corner_pts[i][j][1] + (1-w0)*corner_pts[i][j+1][1];
						subcell_index++;
					}
				}
			} else {
				nsplits[i][j] = 1;
			}
		}
	}
}

bool ImagePixelData::test_if_in_fit_region(const double& x, const double& y, const int mask_k)
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
	// NOTE: This function should be called *before* adding noise to the image.
	double sbmax=-1e30;
	static const double signal_threshold_frac = 1e-1;
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (surface_brightness[i][j] > sbmax) sbmax = surface_brightness[i][j];
		}
	}
	double signal_mean=0,sn_mean=0;
	int npixels=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (surface_brightness[i][j] > signal_threshold_frac*sbmax) {
				signal_mean += surface_brightness[i][j];
				sn_mean += surface_brightness[i][j]/noise_map[i][j];
				npixels++;
			}
		}
	}
	total_signal = signal_mean * pixel_xlength * pixel_ylength;
	if (npixels > 0) {
		sn_mean /= npixels;
		signal_mean /= npixels;
	}
	return sn_mean;
}

void ImagePixelGrid::add_pixel_noise()
{
	if (surface_brightness == NULL) die("surface brightness pixel map has not been loaded");
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			surface_brightness[i][j] += noise_map[i][j]*lens->NormalDeviate();
		}
	}
}

void ImagePixelGrid::find_optimal_sourcegrid(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double &sourcegrid_limit_xmin, const double &sourcegrid_limit_xmax, const double &sourcegrid_limit_ymin, const double& sourcegrid_limit_ymax)
{
	if (surface_brightness == NULL) die("surface brightness pixel map has not been loaded");
	if (lens->image_pixel_data == NULL) die("image pixel data must be loaded to find optimal source grid scale");
	bool use_noise_threshold = true;
	if (lens->noise_threshold <= 0) use_noise_threshold = false;
	int i,j,k,nsp;
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
			if (pixel_in_mask[i][j]) {
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
							sbavg += lens->image_pixel_data->surface_brightness[ii][jj];
							nn++;
						}
					}
					sbavg /= nn;
					if (sbavg <= lens->noise_threshold*noise_map[i][j]) resize_grid = false;
				}
				if (resize_grid) {
					if (!lens->split_imgpixels) {
						xsavg = center_sourcepts[i][j][0];
						ysavg = center_sourcepts[i][j][1];
					} else {
						xsavg=ysavg=0;
						nsp = INTSQR(nsplits[i][j]);
						for (k=0; k < nsp; k++) {
							xsavg += subpixel_center_sourcepts[i][j][k][0];
							ysavg += subpixel_center_sourcepts[i][j][k][1];
						}
						xsavg /= nsp;
						ysavg /= nsp;
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
	// Now let's make the box slightly wider just to be safe
	//double xwidth_adj = 0.1*(sourcegrid_xmax-sourcegrid_xmin);
	//double ywidth_adj = 0.1*(sourcegrid_ymax-sourcegrid_ymin);
	//sourcegrid_xmin -= xwidth_adj/2;
	//sourcegrid_xmax += xwidth_adj/2;
	//sourcegrid_ymin -= ywidth_adj/2;
	//sourcegrid_ymax += ywidth_adj/2;
}

double ImagePixelGrid::find_approx_source_size(double &xcavg, double &ycavg, const bool verbal)
{
	if (lens->image_pixel_data == NULL) die("need to have image pixel data loaded to find optimal shapelet scale");
	static const int nmax_srcsize_it = 8;
	//string sp_filename = "wtf_spt.dat";
	//ofstream sourcepts_file; lens->open_output_file(sourcepts_file,sp_filename);
	//sourcepts_file << setiosflags(ios::scientific);

	double sig;
	double totsurf;
	double area, min_area = 1e30, max_area = -1e30;
	double xcmin, ycmin, sb;
	double xsavg, ysavg;
	double xcold, ycold;
	int i,j,k,nsp;
	double rsq, rsqavg;
	sig = 1e30;
	int npts=10000000, npts_old, iter=0;
	//ofstream wtf("wtf.dat");
	if ((verbal) and (lens->n_sourcepts_fit > 0) and ((lens->include_imgfluxes_in_inversion) or (lens->include_srcflux_in_inversion))) warn("estimated approx extended source size may be biased due to point source when 'invert_imgflux' is on");
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
				if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
					if (pixel_in_mask==NULL) sb = surface_brightness[i][j] - foreground_surface_brightness[i][j];
					else sb = lens->image_pixel_data->surface_brightness[i][j] - foreground_surface_brightness[i][j];
					if ((lens->n_sourcepts_fit > 0) and (!lens->include_imgfluxes_in_inversion) and (!lens->include_srcflux_in_inversion) and (lens->point_image_surface_brightness != NULL)) sb -= lens->point_image_surface_brightness[pixel_index[i][j]];
					if (abs(sb) > 5*noise_map[i][j]) {
						//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
						//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
						// You repeat this code three times in this function! Store things in arrays and GET RID OF THE REDUNDANCIES!!!! IT'S UGLY.
						if (!lens->split_imgpixels) {
							xsavg = center_sourcepts[i][j][0];
							ysavg = center_sourcepts[i][j][1];
						} else {
							xsavg=ysavg=0;
							nsp = INTSQR(nsplits[i][j]);
							for (k=0; k < nsp; k++) {
								xsavg += subpixel_center_sourcepts[i][j][k][0];
								ysavg += subpixel_center_sourcepts[i][j][k][1];
							}
							xsavg /= nsp;
							ysavg /= nsp;
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
				if (((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) and (abs(sb = lens->image_pixel_data->surface_brightness[i][j] - foreground_surface_brightness[i][j]) > 5*noise_map[i][j])) {
					//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
					//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
					if (!lens->split_imgpixels) {
						xsavg = center_sourcepts[i][j][0];
						ysavg = center_sourcepts[i][j][1];
					} else {
						xsavg=ysavg=0;
						nsp = INTSQR(nsplits[i][j]);
						for (k=0; k < nsp; k++) {
							xsavg += subpixel_center_sourcepts[i][j][k][0];
							ysavg += subpixel_center_sourcepts[i][j][k][1];
						}
						xsavg /= nsp;
						ysavg /= nsp;
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
	//ofstream sourcepts_file; lens->open_output_file(sourcepts_file,sp_filename);
	//sourcepts_file << setiosflags(ios::scientific);

	if (lens->image_pixel_data == NULL) die("need to have image pixel data loaded to find optimal shapelet scale");
	double xcavg, ycavg;
	double totsurf;
	double area, min_area = 1e30, max_area = -1e30;
	double xcmin, ycmin, sb;
	double xsavg, ysavg;
	int i,j,k,nsp;
	double rsq, rsqavg;
	sig = 1e30;
	int npts=0, npts_old, iter=0;
	//ofstream wtf("wtf.dat");
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
				if (((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) and (abs(sb = lens->image_pixel_data->surface_brightness[i][j] - foreground_surface_brightness[i][j]) > 5*noise_map[i][j])) {
					//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
					//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
					// You repeat this code three times in this function! Store things in arrays and GET RID OF THE REDUNDANCIES!!!! IT'S UGLY.
					if (!lens->split_imgpixels) {
						xsavg = center_sourcepts[i][j][0];
						ysavg = center_sourcepts[i][j][1];
					} else {
						xsavg=ysavg=0;
						nsp = INTSQR(nsplits[i][j]);
						for (k=0; k < nsp; k++) {
							xsavg += subpixel_center_sourcepts[i][j][k][0];
							ysavg += subpixel_center_sourcepts[i][j][k][1];
						}
						xsavg /= nsp;
						ysavg /= nsp;
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
				if (((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) and (abs(sb = lens->image_pixel_data->surface_brightness[i][j] - foreground_surface_brightness[i][j]) > 5*noise_map[i][j])) {
					//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
					//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
					if (!lens->split_imgpixels) {
						xsavg = center_sourcepts[i][j][0];
						ysavg = center_sourcepts[i][j][1];
					} else {
						xsavg=ysavg=0;
						nsp = INTSQR(nsplits[i][j]);
						for (k=0; k < nsp; k++) {
							xsavg += subpixel_center_sourcepts[i][j][k][0];
							ysavg += subpixel_center_sourcepts[i][j][k][1];
						}
						xsavg /= nsp;
						ysavg /= nsp;
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
			if ((pixel_in_mask==NULL) or ((!lens->zero_sb_extended_mask_prior) and (emask[i][j])) or ((lens->zero_sb_extended_mask_prior) and (mask[i][j]))) {
				//xsavg = (corner_sourcepts[i][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j][0] + corner_sourcepts[i+1][j+1][0]) / 4;
				//ysavg = (corner_sourcepts[i][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j][1] + corner_sourcepts[i+1][j+1][1]) / 4;
				if (!lens->split_imgpixels) {
					xsavg = center_sourcepts[i][j][0];
					ysavg = center_sourcepts[i][j][1];
				} else {
					xsavg=ysavg=0;
					nsp = INTSQR(nsplits[i][j]);
					for (k=0; k < nsp; k++) {
						xsavg += subpixel_center_sourcepts[i][j][k][0];
						ysavg += subpixel_center_sourcepts[i][j][k][1];
					}
					xsavg /= nsp;
					ysavg /= nsp;
				}

				xd = abs(xsavg-xcavg);
				yd = abs(ysavg-ycavg);

				//double ri = abs(sqrt(SQR(center_pts[i][j][0]-0.01)+SQR(center_pts[i][j][1])));
				//if (ri > 0.6) {
				if (xd > xmax) xmax = xd;
				if (yd > ymax) ymax = yd;
					//sourcepts_file << xsavg << " " << ysavg << " " << center_pts[i][j][0] << " " << center_pts[i][j][1] << " " << xd << " " << yd << endl;
				//}
				if ((mask[i][j]) and (abs(lens->image_pixel_data->surface_brightness[i][j] - foreground_surface_brightness[i][j]) > 5*noise_map[i][j])) {
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
	if ((verbal) and (lens->mpi_id==0)) cout << "Fraction of 2-sigma outliers for shapelets: " << fout << endl;
	double maxdist = dmax(xmax,ymax);

	int nn = lens->get_shapelet_nn(src_redshift_index);
	scaled_maxdist = lens->shapelet_window_scaling*maxdist;
	if (lens->shapelet_scale_mode==0) {
		scale = sig; // uses the dispersion of source SB to set scale (WARNING: might not cover all of lensed pixels in mask if n_shapelets is too small!!)
	} else if (lens->shapelet_scale_mode==1) {
		scale = 1.000001*scaled_maxdist/sqrt(nn); // uses outermost pixel in extended mask to set scale
		if (scale > sig) scale = sig; // if the above scale is bigger than ray-traced source scale (sig), then just set scale = sig (otherwise source will be under-resolved)
	}
	if (scale > lens->shapelet_max_scale) scale = lens->shapelet_max_scale;

	int ii, jj, il, ih, jl, jh;
	const double window_size_for_srcarea = 1;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			sb = lens->image_pixel_data->surface_brightness[i][j] - foreground_surface_brightness[i][j];
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
	if ((verbal) and (lens->mpi_id==0)) {
		cout << "expected minscale_res=" << minscale_res << ", found at (x=" << xcmin << ",y=" << ycmin << ") recommended_nn=" << recommended_nn << endl;
		cout << "number of splittings should be at least " << recommended_nsplit << " to capture all source fluctuations" << endl;
		cout << "outermost ray-traced source pixel distance: " << scaled_maxdist << endl;
	}
}

void ImagePixelGrid::fill_surface_brightness_vector()
{
	int column_j = 0;
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((maps_to_source_pixel[i][j]) and ((pixel_in_mask==NULL) or (pixel_in_mask[i][j]))) {
				//lens->image_surface_brightness[column_j++] = surface_brightness[i][j];
				lens->image_surface_brightness[column_j++] = lens->image_pixel_data->surface_brightness[i][j];
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
	lens->open_output_file(gridfile,grid_filename);
	ofstream centerfile;
	lens->open_output_file(centerfile,center_filename);
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

void ImagePixelGrid::find_optimal_sourcegrid_npixels(double pixel_fraction, double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& nsrcpixel_x, int& nsrcpixel_y, int& n_expected_active_pixels)
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
	nsrcpixel_x = (int) sqrt(pixel_fraction*count*dx/dy);
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
	source_lowlevel_pixel_area = pixel_area / (1.3*lens->pixel_magnification_threshold);
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
void ImagePixelGrid::assign_mask_pixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& count, ImagePixelData *data_in)
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
	//if (lens->psf_supersampling) nsubpix = SQR(lens->default_imgpixel_nsplit);
	for (img_index=0; img_index < n_active_pixels; img_index++) {
		i = active_image_pixel_i[img_index];
		j = active_image_pixel_j[img_index];
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
	//if (lens->psf_supersampling) nsubpix = SQR(lens->default_imgpixel_nsplit);
	for (img_index=0; img_index < n_active_pixels; img_index++) {
		i = active_image_pixel_i[img_index];
		j = active_image_pixel_j[img_index];
		for (k=0; k < mapped_delaunay_srcpixels[i][j].size(); k++) {
			 tot++;
			//else tot += nsubpix;
			//if (mapped_delaunay_srcpixels[i][j][k] != -1) tot++;
		}
		//tot += mapped_delaunay_srcpixels[i][j].size();
	}
	return tot;
}

void ImagePixelGrid::assign_image_mapping_flags(const bool delaunay)
{
	int i,j,k;
	n_active_pixels = 0;
	n_high_sn_pixels = 0;
	int nsubpix = INTSQR(lens->default_imgpixel_nsplit);
	int *ptr;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (delaunay) mapped_delaunay_srcpixels[i][j].clear();
			else mapped_cartesian_srcpixels[i][j].clear();
			maps_to_source_pixel[i][j] = false;
			ptr = n_mapped_srcpixels[i][j];
			for (k=0; k < nsubpix; k++) {
				(*ptr++) = 0;
			}
		}
	}
	if ((!delaunay) and (ray_tracing_method == Area_Overlap)) // Delaunay grid does not support overlap ray tracing
	{
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			lensvector *corners[4];
			#pragma omp for private(i,j,corners) schedule(dynamic)
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					if ((pixel_in_mask == NULL) or (pixel_in_mask[i][j])) {
						corners[0] = &corner_sourcepts[i][j];
						corners[1] = &corner_sourcepts[i][j+1];
						corners[2] = &corner_sourcepts[i+1][j];
						corners[3] = &corner_sourcepts[i+1][j+1];
						if (cartesian_srcgrid->assign_source_mapping_flags_overlap(corners,&twist_pts[i][j],twist_status[i][j],mapped_cartesian_srcpixels[i][j],thread)==true) {
							maps_to_source_pixel[i][j] = true;
							#pragma omp atomic
							n_active_pixels++;
							if ((pixel_in_mask != NULL) and (pixel_in_mask[i][j]) and (lens->image_pixel_data->high_sn_pixel[i][j])) n_high_sn_pixels++;
						} else
							maps_to_source_pixel[i][j] = false;
					}
				}
			}
		}
	}
	else if (ray_tracing_method == Interpolate)
	{
		bool trouble_with_starting_vertex = false;
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			if ((lens->split_imgpixels) and (!lens->raytrace_using_pixel_centers)) {
				int nsubpix,subcell_index;
				bool maps_to_something;
				#pragma omp for private(i,j,nsubpix,subcell_index,maps_to_something) schedule(dynamic)
				for (j=0; j < y_N; j++) {
					for (i=0; i < x_N; i++) {
						if ((pixel_in_mask == NULL) or (pixel_in_mask[i][j])) {
							nsubpix = INTSQR(nsplits[i][j]);
							maps_to_something = false;
							for (subcell_index=0; subcell_index < nsubpix; subcell_index++)
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
							}
							if (maps_to_something==true) {
								maps_to_source_pixel[i][j] = true;
								#pragma omp atomic
								n_active_pixels++;
								if ((pixel_in_mask != NULL) and (pixel_in_mask[i][j]) and (lens->image_pixel_data->high_sn_pixel[i][j])) {
									#pragma omp atomic
									n_high_sn_pixels++;
								}
							} else maps_to_source_pixel[i][j] = false;
						}
					}
				}
			} else {
				#pragma omp for private(i,j) schedule(dynamic)	
				for (j=0; j < y_N; j++) {
					for (i=0; i < x_N; i++) {
						if ((pixel_in_mask == NULL) or (pixel_in_mask[i][j])) {
							if ((delaunay) and ((delaunay_srcgrid==NULL) or (delaunay_srcgrid->assign_source_mapping_flags(center_sourcepts[i][j],mapped_delaunay_srcpixels[i][j],n_mapped_srcpixels[i][j][0],i,j,thread,trouble_with_starting_vertex)==true))) {
								maps_to_source_pixel[i][j] = true;
								#pragma omp atomic
								n_active_pixels++;
								if ((pixel_in_mask != NULL) and (pixel_in_mask[i][j]) and (lens->image_pixel_data->high_sn_pixel[i][j])) {
									#pragma omp atomic
									n_high_sn_pixels++;
								}
							} else if ((!delaunay) and (cartesian_srcgrid->assign_source_mapping_flags_interpolate(center_sourcepts[i][j],mapped_cartesian_srcpixels[i][j],thread,i,j)==true)) {
								maps_to_source_pixel[i][j] = true;
								#pragma omp atomic
								n_active_pixels++;
								if ((pixel_in_mask != NULL) and (pixel_in_mask[i][j]) and (lens->image_pixel_data->high_sn_pixel[i][j])) {
									#pragma omp atomic
									n_high_sn_pixels++;
								}
							} else {
								maps_to_source_pixel[i][j] = false;
							}
						}
					}
				}
			}
		}
		if (trouble_with_starting_vertex) warn(lens->warnings,"could not find good starting vertices for Delaunay grid; started with vertex 0 when searching for enclosing triangles");
	}
}

void ImagePixelGrid::find_surface_brightness(const bool foreground_only, const bool lensed_sources_only)
{
	bool supersampling = lens->psf_supersampling;
	double noise;
#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			surface_brightness[i][j] = 0;
		}
	}
	double sbmax=0;
	if ((foreground_only) and (src_redshift_index > 0)) return; // only the first image_pixel_grid object will have foreground light included
	if ((source_fit_mode == Cartesian_Source) or (source_fit_mode == Delaunay_Source)) {
		bool at_least_one_foreground_src = false;
		bool at_least_one_lensed_src = false;
		for (int k=0; k < lens->n_sb; k++) {
			if (!lens->sb_list[k]->is_lensed) {
				at_least_one_foreground_src = true;
			} else {
				at_least_one_lensed_src = true;
			}
		}
		if ((foreground_only) and (!at_least_one_foreground_src)) return;

		if ((source_fit_mode == Cartesian_Source) and (ray_tracing_method == Area_Overlap)) {
			lensvector **corners = new lensvector*[4];
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					//surface_brightness[i][j] = 0;
					corners[0] = &corner_sourcepts[i][j];
					corners[1] = &corner_sourcepts[i][j+1];
					corners[2] = &corner_sourcepts[i+1][j];
					corners[3] = &corner_sourcepts[i+1][j+1];
					if (!foreground_only) surface_brightness[i][j] = cartesian_srcgrid->find_lensed_surface_brightness_overlap(corners,&twist_pts[i][j],twist_status[i][j],0);
					if ((at_least_one_foreground_src) and (!lensed_sources_only) and (src_redshift_index==0)) {
						for (int k=0; k < lens->n_sb; k++) {
							if (!lens->sb_list[k]->is_lensed) {
								if (!lens->sb_list[k]->zoom_subgridding) {
									surface_brightness[i][j] += lens->sb_list[k]->surface_brightness(center_pts[i][j][0],center_pts[i][j][1]);
								} else {
									noise = (lens->use_noise_map) ? noise_map[i][j] : lens->background_pixel_noise;
									surface_brightness[i][j] += lens->sb_list[k]->surface_brightness_zoom(center_pts[i][j],corner_pts[i][j],corner_pts[i+1][j],corner_pts[i][j+1],corner_pts[i+1][j+1],noise);
								}
							}
						}
					}
				}
			}
			delete[] corners;
		} else { // use interpolation to get surface brightness
			if ((lens->split_imgpixels) and (!lens->raytrace_using_pixel_centers)) {
				#pragma omp parallel
				{
					int thread;
#ifdef USE_OPENMP
					thread = omp_get_thread_num();
#else
					thread = 0;
#endif
					//int ii,jj,nsplit;
					//double u0, w0, sb;
					lensvector corner1, corner2, corner3, corner4;
					double subpixel_xlength, subpixel_ylength;
					double sb, sbtot;

					int nsubpix, subcell_index;
					lensvector *center_srcpt, *center_pt;
					//#pragma omp for private(i,j,ii,jj,nsplit,u0,w0,sb) schedule(dynamic)
					#pragma omp for private(i,j,nsubpix,subcell_index,center_pt,center_srcpt,subpixel_xlength,subpixel_ylength,corner1,corner2,corner3,corner4,sb,noise) schedule(dynamic)
					for (j=0; j < y_N; j++) {
						for (i=0; i < x_N; i++) {
							//surface_brightness[i][j] = 0;
							if ((pixel_in_mask == NULL) or (pixel_in_mask[i][j])) {
								sbtot=0;

								nsubpix = INTSQR(nsplits[i][j]);
								center_srcpt = subpixel_center_sourcepts[i][j];
								center_pt = subpixel_center_pts[i][j];
								for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
									if (!foreground_only) {
										if (source_fit_mode==Delaunay_Source) sb = delaunay_srcgrid->find_lensed_surface_brightness(center_srcpt[subcell_index],i,j,thread);
										else if (source_fit_mode==Cartesian_Source) sb = cartesian_srcgrid->find_lensed_surface_brightness_interpolate(center_srcpt[subcell_index],thread);
									}
									if (supersampling) subpixel_surface_brightness[i][j][subcell_index] = sb;
									sbtot += sb;
								}
								surface_brightness[i][j] += sbtot / nsubpix;
							}
						}
					}
				}
			} else {
				for (j=0; j < y_N; j++) {
					for (i=0; i < x_N; i++) {
						//surface_brightness[i][j] = 0;
						if ((pixel_in_mask == NULL) or (pixel_in_mask[i][j])) {
							if (!foreground_only) {
								if (source_fit_mode==Delaunay_Source) surface_brightness[i][j] = delaunay_srcgrid->find_lensed_surface_brightness(center_sourcepts[i][j],i,j,0);
								else {
									surface_brightness[i][j] = cartesian_srcgrid->find_lensed_surface_brightness_interpolate(center_sourcepts[i][j],0);
								}
							}
						}
						if ((at_least_one_foreground_src) and (!lensed_sources_only) and (src_redshift_index==0)) {
							for (int k=0; k < lens->n_sb; k++) {
								if (!lens->sb_list[k]->is_lensed) {
									if (!lens->sb_list[k]->zoom_subgridding) {
										surface_brightness[i][j] += lens->sb_list[k]->surface_brightness(center_pts[i][j][0],center_pts[i][j][1]);
									} else {
										noise = (lens->use_noise_map) ? noise_map[i][j] : lens->background_pixel_noise;
										surface_brightness[i][j] += lens->sb_list[k]->surface_brightness_zoom(center_pts[i][j],corner_pts[i][j],corner_pts[i+1][j],corner_pts[i][j+1],corner_pts[i+1][j+1],noise);
									}
								}
							}
						}
					}

				}
			}
		}
	}

	// Now we deal with lensed and unlensed source objects, if they exist
	bool at_least_one_lensed_src = false;
	for (int k=0; k < lens->n_sb; k++) {
		if ((lens->sb_list[k]->is_lensed) and (lens->sbprofile_redshift_idx[k]==src_redshift_index)) {
			at_least_one_lensed_src = true;
		}
	}
	if (lens->split_imgpixels) {
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			double sb,sbtot;
			lensvector corner1, corner2, corner3, corner4;
			double subpixel_xlength, subpixel_ylength;

			int nsubpix,subcell_index;
			lensvector *center_srcpt, *center_pt;
			#pragma omp for private(i,j,sb,subcell_index,nsubpix,subpixel_xlength,subpixel_ylength,center_pt,center_srcpt,corner1,corner2,corner3,corner4,noise) schedule(dynamic)
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					if ((pixel_in_mask == NULL) or (pixel_in_mask[i][j])) {
						sbtot=0;
						nsubpix = INTSQR(nsplits[i][j]);
						center_srcpt = subpixel_center_sourcepts[i][j];
						center_pt = subpixel_center_pts[i][j];
						for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
							for (int k=0; k < lens->n_sb; k++) {
								if ((lens->sb_list[k]->is_lensed) and (lens->sbprofile_redshift_idx[k]==src_redshift_index)) {
									if (!foreground_only) {
										sb = lens->sb_list[k]->surface_brightness(center_srcpt[subcell_index][0],center_srcpt[subcell_index][1]);
									}
								} else if ((!lensed_sources_only) and (!lens->sb_list[k]->is_lensed) and (src_redshift_index==0)) { // this is ugly. Should just generate a list (near the beginning of this function) of which sources will be used!
									if (!lens->sb_list[k]->zoom_subgridding) {
										//cout << " center pt: " << center_pt[subcell_index][0] << " " << center_pt[subcell_index][1] << " (should be near " << center_pts[i][j][0] << " " << center_pts[i][j][1] << ")" << endl;
										sb = lens->sb_list[k]->surface_brightness(center_pt[subcell_index][0],center_pt[subcell_index][1]);
									}
									else {
										subpixel_xlength = pixel_xlength/nsplits[i][j];
										subpixel_ylength = pixel_ylength/nsplits[i][j];
										corner1[0] = center_pt[subcell_index][0] - subpixel_xlength/2;
										corner1[1] = center_pt[subcell_index][1] - subpixel_ylength/2;
										corner2[0] = center_pt[subcell_index][0] + subpixel_xlength/2;
										corner2[1] = center_pt[subcell_index][1] - subpixel_ylength/2;
										corner3[0] = center_pt[subcell_index][0] - subpixel_xlength/2;
										corner3[1] = center_pt[subcell_index][1] + subpixel_ylength/2;
										corner4[0] = center_pt[subcell_index][0] + subpixel_xlength/2;
										corner4[1] = center_pt[subcell_index][1] + subpixel_ylength/2;
										noise = (lens->use_noise_map) ? noise_map[i][j] : lens->background_pixel_noise;
										sb = lens->sb_list[k]->surface_brightness_zoom(center_pt[subcell_index],corner1,corner2,corner3,corner4,noise);
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
						surface_brightness[i][j] += sbtot / nsubpix;
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
					for (int k=0; k < lens->n_sb; k++) {
						if ((lens->sb_list[k]->is_lensed) and (lens->sbprofile_redshift_idx[k]==src_redshift_index)) { // this is ugly. Should just generate a list (near the beginning of this function) of which sources will be used!
							if (!foreground_only) {
								if (!lens->sb_list[k]->zoom_subgridding) surface_brightness[i][j] += lens->sb_list[k]->surface_brightness(center_sourcepts[i][j][0],center_sourcepts[i][j][1]);
								else {
									noise = (lens->use_noise_map) ? noise_map[i][j] : lens->background_pixel_noise;
									surface_brightness[i][j] += lens->sb_list[k]->surface_brightness_zoom(center_sourcepts[i][j],corner_sourcepts[i][j],corner_sourcepts[i+1][j],corner_sourcepts[i][j+1],corner_sourcepts[i+1][j+1],noise);
								}
							}
						}
						else if ((!lensed_sources_only) and (!lens->sb_list[k]->is_lensed) and (src_redshift_index==0)) { // this is ugly. Should just generate a list (near the beginning of this function) of which sources will be used!
							if (!lens->sb_list[k]->zoom_subgridding) {
								surface_brightness[i][j] += lens->sb_list[k]->surface_brightness(center_pts[i][j][0],center_pts[i][j][1]);
							}
							else {
								noise = (lens->use_noise_map) ? noise_map[i][j] : lens->background_pixel_noise;
								surface_brightness[i][j] += lens->sb_list[k]->surface_brightness_zoom(center_pts[i][j],corner_pts[i][j],corner_pts[i+1][j],corner_pts[i][j+1],corner_pts[i+1][j+1],noise);

							}
						}
					}
				}
			}
		}
	}
#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for ray-tracing image surface brightness values: " << wtime << endl;
	}
#endif
}



void ImagePixelGrid::find_point_images(const double src_x, const double src_y, vector<image>& imgs, const bool use_overlap_in, const bool is_lensed, const bool verbal)
{
	imgs.resize(0);
	if (!is_lensed) {
		image imgpt;
		imgpt.pos[0] = src_x;
		imgpt.pos[1] = src_y;
		imgpt.mag = 1.0;
		imgpt.td = 0;
		imgs.push_back(imgpt);
		return;
	}
	lens->record_singular_points(imggrid_zfactors);
	static const int max_nimgs = 50;
	lens->source[0] = src_x;
	lens->source[1] = src_y;
	int i,j,npix,cell_i,cell_j,n_candidates = 0;
	SourcePixel* cellptr;
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
		npix = ntot_cells;
	}
	//cout << "The source cell containing this source point has " << npix << " overlapping image pixels" << endl;
	int n,imgpt_i,img_i,img_j;
#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	//cout << "SRC: " << src_x << " " << src_y << endl;
	enum InsideTri { None, Lower, Upper } inside_tri; // inside_tri = 0 if not inside; 1 if inside lower triangle; 2 if inside upper triangle
	struct imgpt_info {
		bool confirmed;
		lensvector pos;
	};
	imgpt_info image_candidates[max_nimgs]; // if there are more than 20 images, we have a truly demonic lens on our hands
	lensvector d1,d2,d3;
	double product1,product2,product3;
	// No need to parallelize this part--it is very, very fast
	//cout << "NPIX: " << npix << endl;
	lensvector *vertex1,*vertex2,*vertex3,*vertex4,*vertex5;
	lensvector *vertex1_srcplane,*vertex2_srcplane,*vertex3_srcplane,*vertex4_srcplane,*vertex5_srcplane;
	int *twist_type;
	for (i=0; i < npix; i++) {
		if (use_overlap) n = cellptr->overlap_pixel_n[i];
		else n = i;
		img_j = masked_pixels_j[n];
		img_i = masked_pixels_i[n];
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
			//lensvector srcpt;
			//lens->find_sourcept((*vertex3),srcpt,0,imggrid_zfactors,imggrid_betafactors);
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
			//lensvector srcpt;
			//lens->find_sourcept((*vertex3),srcpt,0,imggrid_zfactors,imggrid_betafactors);
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
		lensvector side1, side2;
		lensvector side1_srcplane, side2_srcplane;
		if ((inside_tri != None) and (*twist_type!=0)) warn(lens->newton_warnings,"possible image identified in cell with nonzero twist status (%g,%g); status %i",center_pts[img_i][img_j][0],center_pts[img_i][img_j][1],*twist_type);
		if (inside_tri != None) {
			//cout << "i=" << img_i << " j=" << img_j << " twiststat=" << *twist_type << endl;
			imgpt_i = n_candidates++;
			if (n_candidates > max_nimgs) die("exceeded max number of images in ImagePixelGrid::find_point_images");
			//pixels_with_imgpts[imgpt_i].img_i = img_i;
			//pixels_with_imgpts[imgpt_i].img_j = img_j;
			//pixels_with_imgpts[imgpt_i].upper_tri = (inside_tri==Upper) ? true : false;
			image_candidates[imgpt_i].confirmed = false;
			if (inside_tri==Lower) {
				image_candidates[imgpt_i].pos[0] = ((*vertex1)[0] + (*vertex2)[0] + (*vertex3)[0])/3;
				image_candidates[imgpt_i].pos[1] = ((*vertex1)[1] + (*vertex2)[1] + (*vertex3)[1])/3;
				// For now, just don't bother with central image points when using a pixel image--it only causes trouble in the form of duplicate images
				//if (!lens->include_central_image) {
				if (*twist_type==0) { // central images are unlikely to be highly magnified near a critical curve, so twisting is unlikely in that case
					kap = lens->kappa(image_candidates[imgpt_i].pos,imggrid_zfactors,imggrid_betafactors);
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
				//if (!lens->include_central_image) {
				if (*twist_type==0) { // central images are unlikely to be highly magnified near a critical curve, so twisting is unlikely in that case
					kap = lens->kappa(image_candidates[imgpt_i].pos,imggrid_zfactors,imggrid_betafactors);
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
			//if ((!lens->include_central_image) and (kap > 1) and (product1*product2 > 0)) n_candidates--;
			if ((*twist_type==0) and (kap > 1) and (product1*product2 > 0)) n_candidates--; // exclude central image candidate by default
		}

		//cout << "Pixel " << img_i << "," << img_j << endl;
	}

	LensProfile *lptr;
	bool singular;
	for (i=0; i < lens->nlens; i++) {
		lptr = lens->lens_list[i];
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
	image_pos_accuracy = Grid::image_pos_accuracy;
	//image_pos_accuracy = 0.05*lens->data_pixel_size;
	//if (image_pos_accuracy < 0) image_pos_accuracy = 1e-3; // in case the data pixel size has not been set
	if ((lens->mpi_id==0) and (verbal)) cout << "Found " << n_candidates << " candidate images" << endl;
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
		if (lens->include_time_delays) td_factor = lens->time_delay_factor_arcsec(lens->lens_redshift,lens->reference_source_redshift);
		image imgpt;
		double mag;
		#pragma omp for private(i) schedule(static)
		for (i=0; i < n_candidates; i++) {
			//cout << "Trying candidate " << i << ": " << image_candidates[i].pos[0] << " " << image_candidates[i].pos[1] << endl;
			if ((lens->skip_newtons_method) or (run_newton(image_candidates[i].pos,mag,thread)==true)) {
			//{
				imgpt.pos = image_candidates[i].pos;
				//imgpt.mag = lens->magnification(image_candidates[i].pos,0,imggrid_zfactors,imggrid_betafactors);
				imgpt.mag = mag;
				imgpt.flux = -1e30;
				//cout << "FOUND IMAGE AT: " << imgpt.pos[0] << " " << imgpt.pos[1] << endl;
				//if ((lens->mpi_id==0) and (verbal)) cout << "FOUND IMAGE AT: " << imgpt.pos[0] << " " << imgpt.pos[1] << endl;
				if (lens->include_time_delays) {
					double potential = lens->potential(imgpt.pos,imggrid_zfactors,imggrid_betafactors);
					imgpt.td = 0.5*(SQR(imgpt.pos[0]-lens->source[0])+SQR(imgpt.pos[1]-lens->source[1])) - potential; // the dimensionless version; it will be converted to days by the QLens class
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
	vector<image>::iterator it, it2;
	double sep, pixel_size;
	pixel_size = dmax(pixel_xlength,pixel_ylength);
	//int oldsize = imgs.size();
	if (imgs.size() > 1) {
		for (it = imgs.begin()+1; it != imgs.end(); it++) {
			redundancy = false;
			for (it2 = imgs.begin(); it2 != it; it2++) {
				sep = sqrt(SQR(it->pos[0] - it2->pos[0]) + SQR(it->pos[1] - it2->pos[1]));
				if (sep < pixel_size)
				{
					redundancy = true;
					warn(lens->newton_warnings,"rejecting probable duplicate image (imgsep=%g,threshold=%g): src (%g,%g), image (%g,%g), mag %g",sep,pixel_size,lens->source[0],lens->source[1],it->pos[0],it->pos[1],it->mag);
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
	if (lens->include_central_image) max_expected++;
	if (imgs.size() > max_expected) warn(lens->newton_warnings,"more than four images after trimming redundancies");

	if ((lens->mpi_id==0) and (verbal)) {
		cout << "# images found: " << imgs.size() << endl;
		for (i=0; i < imgs.size(); i++) {
			cout << imgs[i].pos[0] << " " << imgs[i].pos[1] << " " << imgs[i].mag;
			if (lens->include_time_delays) cout << " " << imgs[i].td;
			cout << endl;
		}
	}
#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for point image finding: " << wtime << endl;
	}
#endif
}

void ImagePixelGrid::generate_point_images(const vector<image>& imgs, double *ptimage_surface_brightness, const bool use_img_fluxes, const double srcflux, const int img_num)
{
	int nx_half, ny_half;
	double normfac, sigx, sigy;
	int nsplit = lens->psf_ptsrc_nsplit;
	int nsubpix = nsplit*nsplit;
	if (lens->use_input_psf_matrix) {
		if (lens->psf_matrix == NULL) return;
		nx_half = lens->psf_npixels_x/2;
		ny_half = lens->psf_npixels_y/2;
	} else {
		double sigma_fraction = sqrt(-2*log(lens->psf_ptsrc_threshold));
		double nx_half_dec, ny_half_dec;
		sigx = lens->psf_width_x;
		sigy = lens->psf_width_y;
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
#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int img_i0, img_if;
	if (img_num==-1) {
		img_i0 = 0;
		img_if = imgs.size();
	} else {
		if (img_num >= imgs.size()) die("img_num does not exist; cannot generate point image");
		img_i0 = img_num;
		img_if = img_num+1;
	}
	if (ptimage_surface_brightness==NULL) die("RUHROH ptimgae_surface_brihgtness is null");
	int idx;
	for (idx=0; idx < lens->image_npixels; idx++) ptimage_surface_brightness[idx] = 0;
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
					if (lens->use_input_psf_matrix) {
						sb += fluxfac*lens->interpolate_PSF_matrix(x-x0,y-y0,lens->psf_supersampling);
					} else {
						sb += fluxfac*normfac*exp(-(SQR((x-x0)/sigx) + SQR((y-y0)/sigy))/2);
					}
				}
			}
			sb /= nsubpix;
			//cout << "sb=" << sb << endl;
			if ((pixel_in_mask==NULL) or (pixel_in_mask[i][j])) {
				ptimage_surface_brightness[pixel_index[i][j]] += sb;
				//tot += sb;
			}
		}
		//sigx = lens->psf_width_x;
		//sigy = lens->psf_width_y;
		//normfac = 1.0/(M_2PI*sigx*sigy);
		//double sbcheck0 = lens->psf_matrix[nx_half][ny_half]/(pixel_xlength*pixel_ylength);
		//double sbcheck = lens->interpolate_PSF_matrix(0,0)/(pixel_xlength*pixel_ylength);
		//double sbcheck2 = normfac;
		////cout << "normfac_inv: " << (1.0/normfac) << endl;
		//cout << "SBCHECKS: " << nx_half << " " << ny_half << " " << sbcheck0 << " " << sbcheck << " " << sbcheck2 << endl;

		//cout << "TOT=" << tot << endl;
		//cout << "Added point image" << endl;
		//cout << "area = " << (nx*ny*pixel_xlength*pixel_ylength) << endl;
	}
#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for adding point images: " << wtime << endl;
	}
#endif
}

void ImagePixelGrid::add_point_images(double *ptimage_surface_brightness, const int npix)
{
	int i,j,indx;
	for (indx=0; indx < npix; indx++) {
		i = active_image_pixel_i[indx];
		j = active_image_pixel_j[indx];
		surface_brightness[i][j] += ptimage_surface_brightness[indx];
	}
}

void ImagePixelGrid::generate_and_add_point_images(const vector<image>& imgs, const bool include_imgfluxes, const double srcflux)
{
	double *ptimgs = new double[n_active_pixels];
	generate_point_images(imgs,ptimgs,include_imgfluxes,srcflux);
	add_point_images(ptimgs,n_active_pixels);

	delete[] ptimgs;
}	

// 2-d Newton's Method w/ backtracking routines
// These functions are redundant with the same ones in the Grid class, and I *HATE* redundancies like this. But for now, it's
// easiest to just copy it over. Later you can put NewtonsMethod in a separate class and have it inherited by both Grid and ImagePixelGrid.

const int ImagePixelGrid::max_iterations = 200;
const int ImagePixelGrid::max_step_length = 100;

inline void ImagePixelGrid::SolveLinearEqs(lensmatrix& a, lensvector& b)
{
	double det, temp;
	det = determinant(a);
	temp = (-a[1][0]*b[1]+a[1][1]*b[0]) / det;
	b[1] = (-a[0][1]*b[0]+a[0][0]*b[1]) / det;
	b[0] = temp;
}

inline double ImagePixelGrid::max_component(const lensvector& x) { return dmax(fabs(x[0]),fabs(x[1])); }

bool ImagePixelGrid::run_newton(lensvector& xroot, double& mag, const int& thread)
{
	lensvector xroot_initial = xroot;
	if ((xroot[0]==0) and (xroot[1]==0)) { xroot[0] = xroot[1] = 5e-1*lens->cc_rmin; }	// Avoiding singularity at center
	if (NewtonsMethod(xroot, newton_check[thread], thread)==false) {
		warn(lens->newton_warnings,"Newton's method failed for source (%g,%g), initial point (%g,%g)",lens->source[0],lens->source[1],xroot_initial[0],xroot_initial[1]);
		return false;
	}
	//if (lens->reject_images_found_outside_cell) {
		//if (test_if_inside_cell(xroot,thread)==false) {
			////warn(lens->warnings,"Rejecting image found outside cell for source (%g,%g), level %i, cell center (%g,%g)",lens->source[0],lens->source[1],level,center_imgplane[0],center_imgplane[1],xroot[0],xroot[1]);
			//return false;
		//}
	//}

	lensvector lens_eq_f;
	lens->lens_equation(xroot,lens_eq_f,thread,imggrid_zfactors,imggrid_betafactors);
	//double lenseq_mag = sqrt(SQR(lens_eq_f[0]) + SQR(lens_eq_f[1]));
	//double tryacc = image_pos_accuracy / sqrt(abs(lens->magnification(xroot,thread,zfactor)));
	//cout << lenseq_mag << " " << tryacc << " " << sqrt(abs(lens->magnification(xroot,thread,zfactor))) << endl;
	if (newton_check[thread]==true) { warn(lens->newton_warnings, "false image--converged to local minimum"); return false; }
	if (lens->n_singular_points > 0) {
		//cout << "singular point: " << lens->singular_pts[0][0] << " " << lens->singular_pts[0][1] << endl;
		double singular_pt_accuracy = 2*image_pos_accuracy;
		for (int i=0; i < lens->n_singular_points; i++) {
			if ((abs(xroot[0]-lens->singular_pts[i][0]) < singular_pt_accuracy) and (abs(xroot[1]-lens->singular_pts[i][1]) < singular_pt_accuracy)) {
				warn(lens->newton_warnings,"Newton's method converged to singular point (%g,%g) for source (%g,%g)",lens->singular_pts[i][0],lens->singular_pts[i][1],lens->source[0],lens->source[1]);
				return false;
			}
		}
	}
	if (((xroot[0]==xroot_initial[0]) and (xroot_initial[0] != 0)) and ((xroot[1]==xroot_initial[1]) and (xroot_initial[1] != 0)))
		warn(lens->newton_warnings, "Newton's method returned initial point");
	mag = lens->magnification(xroot,thread,imggrid_zfactors,imggrid_betafactors);
	if ((abs(lens_eq_f[0]) > 1000*image_pos_accuracy) and (abs(lens_eq_f[1]) > 1000*image_pos_accuracy) and (abs(mag) < 1e-3)) {
		if (lens->newton_warnings==true) {
			warn(lens->newton_warnings,"Newton's method may have found false root (%g,%g) (within 1000*accuracy) for source (%g,%g), cell center (%g,%g), mag %g",xroot[0],xroot[1],lens->source[0],lens->source[1],xroot_initial[0],xroot_initial[1],xroot[0],xroot[1],mag);
		}
	}
	if ((abs(mag) > lens->newton_magnification_threshold) or (mag*0.0 != 0.0)) {
		if (lens->reject_himag_images) {
			if ((lens->mpi_id==0) and (lens->newton_warnings)) {
				cout << "*WARNING*: Rejecting image that exceeds imgsrch_mag_threshold (" << abs(mag) << "), src=(" << lens->source[0] << "," << lens->source[1] << "), x=(" << xroot[0] << "," << xroot[1] << ")      " << endl;
				if (lens->use_ansi_characters) {
					cout << "                                                                                                                            " << endl;
					cout << "\033[2A";
				}
			}
			return false;
		} else {
			if ((lens->mpi_id==0) and (lens->warnings)) {
				cout << "*WARNING*: Image exceeds imgsrch_mag_threshold (" << abs(mag) << "); src=(" << lens->source[0] << "," << lens->source[1] << "), x=(" << xroot[0] << "," << xroot[1] << ")        " << endl;
				if (lens->use_ansi_characters) {
					cout << "                                                                                                                            " << endl;
					cout << "\033[2A";
				}
			}
		}
	}
	if ((lens->include_central_image==false) and (mag > 0) and (lens->kappa(xroot,imggrid_zfactors,imggrid_betafactors) > 1)) return false; // discard central image if not desired

	/*
	bool status = true;
	//#pragma omp critical
	//{
			images[nfound].pos[0] = xroot[0];
			images[nfound].pos[1] = xroot[1];
			images[nfound].mag = lens->magnification(xroot,0,imggrid_zfactors,imggrid_betafactors);
			if (lens->include_time_delays) {
				double potential = lens->potential(xroot,imggrid_zfactors,imggrid_betafactors);
				images[nfound].td = 0.5*(SQR(xroot[0]-lens->source[0])+SQR(xroot[1]-lens->source[1])) - potential; // the dimensionless version; it will be converted to days by the QLens class
			} else {
				images[nfound].td = 0;
			}
			images[nfound].parity = sign(images[nfound].mag);

			if (lens->use_cc_spline) {
				bool found_pos=false, found_neg=false;
				double rroot, thetaroot, cr0, cr1;
				rroot = norm(xroot[0]-lens->grid_xcenter,xroot[1]-lens->grid_ycenter);
				thetaroot = angle(xroot[0]-lens->grid_xcenter,xroot[1]-lens->grid_ycenter);
				cr0 = lens->ccspline[0].splint(thetaroot);
				cr1 = lens->ccspline[1].splint(thetaroot);

				int expected_parity;
				if (rroot < cr0) {
					nfound_max++; expected_parity = 1;
				} else if (rroot > cr1) {
					nfound_pos++; expected_parity = 1;
				} else {
					nfound_neg++; expected_parity = -1;
				}

				if (images[nfound].parity != expected_parity)
					warn(lens->warnings, "wrong parity found for image from source (%g, %g)", lens->source[0], lens->source[1]);
				
				if ((lens->system_type==Single) and (nfound_pos >= 1)) finished_search = true;
				else
				{
					if ((lens->system_type==Double) and (nfound_pos >= 1)) found_pos = true;
					else if (((lens->system_type==Quad) or (lens->system_type==Cusp)) and (nfound_pos >= 2)) found_pos = true;

					if (((lens->system_type==Double) or (lens->system_type==Cusp)) and (nfound_neg >= 1)) found_neg = true;
					else if ((lens->system_type==Quad) and (nfound_neg >= 2)) found_neg = true;

					if ((found_pos) and (found_neg)) finished_search = true;
				}
			}

			nfound++;
		}
		*/
	//}
	return true;
}

bool ImagePixelGrid::NewtonsMethod(lensvector& x, bool &check, const int& thread)
{
	check = false;
	lensvector g, p, xold;
	lensmatrix fjac;

	lens->lens_equation(x, fvec[thread], thread, imggrid_zfactors, imggrid_betafactors);
	double f = 0.5*fvec[thread].sqrnorm();
	if (max_component(fvec[thread]) < 0.01*image_pos_accuracy)
		return true; 

	double fold, stpmax, temp, test;
	stpmax = max_step_length * dmax(x.norm(), 2.0); 
	for (int its=0; its < max_iterations; its++) {
		lens->hessian(x[0],x[1],fjac,thread,imggrid_zfactors,imggrid_betafactors);
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
		if ((x[0] > 1e3*lens->cc_rmax) or (x[1] > 1e3*lens->cc_rmax)) {
			warn(lens->newton_warnings, "Newton blew up!");
			return false;
		}
		/*
		lens->lens_equation(x, fvec[thread], thread, zfactor);
		double magfac = sqrt(abs(lens->magnification(x,thread,zfactor)));
		double tryacc;
		lensvector dx = x - xold;
		double dxnorm = dx.norm();
		dx[0] /= dxnorm;
		dx[1] /= dxnorm;
		lensmatrix magmat;
		lensvector bb;
		lens->sourcept_jacobian(x,bb,magmat,thread,zfactor);
		bb = magmat*dx;
		lensvector dy;
		dy[1] = -dx[0];
		dy[0] = dx[1];
		lensvector cc;
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

bool ImagePixelGrid::LineSearch(lensvector& xold, double fold, lensvector& g, lensvector& p, lensvector& x,
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
		if ((fabs(x[0]) < 1e6*lens->cc_rmax) and (fabs(x[1]) < 1e6*lens->cc_rmax))
			;
		else {
			warn(lens->newton_warnings, "Newton blew up!");
			return false;
		}
		lens->lens_equation(x, fvec[thread], thread, imggrid_zfactors, imggrid_betafactors);
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

ImagePixelGrid::~ImagePixelGrid()
{
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
		delete[] foreground_surface_brightness[i];
		delete[] surface_brightness[i];
		delete[] noise_map[i];
		delete[] source_plane_triangle1_area[i];
		delete[] source_plane_triangle2_area[i];
		delete[] pixel_mag[i];
		delete[] twist_status[i];
		delete[] twist_pts[i];
		delete[] nsplits[i];
		for (int j=0; j < y_N; j++) {
			delete[] subpixel_maps_to_srcpixel[i][j];
			delete[] subpixel_center_pts[i][j];
			delete[] subpixel_center_sourcepts[i][j];
			delete[] subpixel_surface_brightness[i][j];
			delete[] subpixel_weights[i][j];
			delete[] n_mapped_srcpixels[i][j];
		}
		delete[] subpixel_maps_to_srcpixel[i];
		delete[] subpixel_center_pts[i];
		delete[] subpixel_center_sourcepts[i];
		delete[] subpixel_surface_brightness[i];
		delete[] subpixel_weights[i];
		delete[] n_mapped_srcpixels[i];
	}
	int max_subpixel_nx = x_N*max_nsplit;
	for (int i=0; i < max_subpixel_nx; i++) {
		delete[] subpixel_index[i];
	}
	delete[] center_pts;
	delete[] center_sourcepts;
	delete[] maps_to_source_pixel;
	delete[] pixel_index;
	delete[] pixel_index_fgmask;
	delete[] subpixel_index;
	delete[] mapped_cartesian_srcpixels;
	delete[] mapped_delaunay_srcpixels;
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
	delete[] n_mapped_srcpixels;
	delete[] nsplits;
	delete[] twist_status;
	delete[] twist_pts;
	if (pixel_in_mask != NULL) {
		for (int i=0; i < x_N; i++) delete[] pixel_in_mask[i];
		delete[] pixel_in_mask;
	}
	delete_ray_tracing_arrays();

	if (active_image_pixel_i != NULL) delete[] active_image_pixel_i;
	if (active_image_pixel_j != NULL) delete[] active_image_pixel_j;
	if (active_image_pixel_i_ss != NULL) delete[] active_image_pixel_i_ss;
	if (active_image_pixel_j_ss != NULL) delete[] active_image_pixel_j_ss;
	if (active_image_subpixel_ss != NULL) delete[] active_image_subpixel_ss;
	if (image_pixel_i_from_subcell_ii != NULL) delete[] image_pixel_i_from_subcell_ii;
	if (image_pixel_j_from_subcell_jj != NULL) delete[] image_pixel_j_from_subcell_jj;
	if (active_image_pixel_i_fgmask != NULL) delete[] active_image_pixel_i_fgmask;
	if (active_image_pixel_j_fgmask != NULL) delete[] active_image_pixel_j_fgmask;

}

/************************** Functions in class QLens that pertain to pixel mapping and inversion ****************************/

bool QLens::assign_pixel_mappings(const int zsrc_i, const bool verbal)
{
	int i, j, ii, jj, subcell_index, nsubpix, image_pixel_index=0, image_subpixel_index=0;
	if ((zsrc_i >= 0) and (n_extended_src_redshifts==0)) die("no ext src redshift created");
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	SourcePixelGrid *cartesian_srcgrid = image_pixel_grid->cartesian_srcgrid;

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	int tot_npixels_count;
	if (source_fit_mode==Delaunay_Source) {
			image_pixel_grid->assign_image_mapping_flags(true);
			if (image_pixel_grid->delaunay_srcgrid != NULL) { 
				source_npixels = image_pixel_grid->delaunay_srcgrid->assign_active_indices_and_count_source_pixels(activate_unmapped_source_pixels);
			} else {
				source_npixels = 0;
			}
	} else {
		tot_npixels_count = cartesian_srcgrid->assign_indices_and_count_levels();
		if ((mpi_id==0) and (adaptive_subgrid) and (verbal==true)) cout << "Number of source cells: " << tot_npixels_count << endl;
		image_pixel_grid->assign_image_mapping_flags(false);

		cartesian_srcgrid->regrid = false;
		if (nlens==0) source_npixels = 0;
		else {
			source_npixels = cartesian_srcgrid->assign_active_indices_and_count_source_pixels(regrid_if_unmapped_source_subpixels,activate_unmapped_source_pixels,exclude_source_pixels_beyond_fit_window);
			if (source_npixels==0) { warn("number of source pixels cannot be zero"); return false; }
		}
		while (cartesian_srcgrid->regrid) {
			if ((mpi_id==0) and (verbal==true)) cout << "Redrawing the source grid after reverse-splitting unmapped source pixels...\n";
			cartesian_srcgrid->regrid = false;
			cartesian_srcgrid->assign_all_neighbors();
			tot_npixels_count = cartesian_srcgrid->assign_indices_and_count_levels();
			if ((mpi_id==0) and (verbal==true)) cout << "Number of source cells after re-gridding: " << tot_npixels_count << endl;
			image_pixel_grid->assign_image_mapping_flags(false);
			//cartesian_srcgrid->print_indices();
			source_npixels = cartesian_srcgrid->assign_active_indices_and_count_source_pixels(regrid_if_unmapped_source_subpixels,activate_unmapped_source_pixels,exclude_source_pixels_beyond_fit_window);
		}
	}
	image_pixel_grid->Lmatrix_src_npixels = source_npixels; // store the number of source pixels for each image pixel grid; useful later for cleaning up FFT convolution arrays
	source_n_amps = source_npixels;
	if (include_imgfluxes_in_inversion) {
		for (int i=0; i < point_imgs.size(); i++) {
			source_n_amps += point_imgs[i].size(); // in this case, source amplitudes include point image amplitudes as well as pixel values
		}
	} else if (include_srcflux_in_inversion) {
		source_n_amps += point_imgs.size();
	}

	if (psf_supersampling) {
		int nsub=0;
		for (j=0; j < image_pixel_grid->y_N; j++) {
			for (i=0; i < image_pixel_grid->x_N; i++) {
				if (image_pixel_grid->maps_to_source_pixel[i][j]) {
					if (image_pixel_grid->nsplits[i][j] != default_imgpixel_nsplit) die("nsplit has to be the same for all pixels to use supersampling (pixel (%i,%i), nsplits: %i vs %i)",i,j,image_pixel_grid->nsplits[i][j],default_imgpixel_nsplit);
					nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]);
					nsub += nsubpix;
				}
			}
		}
		image_n_subpixels = nsub;
	}

	image_npixels = image_pixel_grid->n_active_pixels;
	if (image_pixel_grid->active_image_pixel_i != NULL) delete[] image_pixel_grid->active_image_pixel_i;
	if (image_pixel_grid->active_image_pixel_j != NULL) delete[] image_pixel_grid->active_image_pixel_j;
	image_pixel_grid->active_image_pixel_i = new int[image_npixels];
	image_pixel_grid->active_image_pixel_j = new int[image_npixels];

	if (psf_supersampling) {
		if (image_pixel_grid->active_image_pixel_i_ss != NULL) delete[] image_pixel_grid->active_image_pixel_i_ss;
		if (image_pixel_grid->active_image_pixel_j_ss != NULL) delete[] image_pixel_grid->active_image_pixel_j_ss;
		if (image_pixel_grid->active_image_subpixel_ss != NULL) delete[] image_pixel_grid->active_image_subpixel_ss;
		if (image_pixel_grid->active_image_subpixel_ii != NULL) delete[] image_pixel_grid->active_image_subpixel_ii;
		if (image_pixel_grid->active_image_subpixel_jj != NULL) delete[] image_pixel_grid->active_image_subpixel_jj;
		if (image_pixel_grid->image_pixel_i_from_subcell_ii != NULL) delete[] image_pixel_grid->image_pixel_i_from_subcell_ii;
		if (image_pixel_grid->image_pixel_j_from_subcell_jj != NULL) delete[] image_pixel_grid->image_pixel_j_from_subcell_jj;


		image_pixel_grid->active_image_pixel_i_ss = new int[image_n_subpixels];
		image_pixel_grid->active_image_pixel_j_ss = new int[image_n_subpixels];
		image_pixel_grid->active_image_subpixel_ss = new int[image_n_subpixels];
		image_pixel_grid->active_image_subpixel_ii = new int[image_n_subpixels];
		image_pixel_grid->active_image_subpixel_jj = new int[image_n_subpixels];
		image_pixel_grid->image_pixel_i_from_subcell_ii = new int[image_pixel_grid->x_N*default_imgpixel_nsplit];
		image_pixel_grid->image_pixel_j_from_subcell_jj = new int[image_pixel_grid->y_N*default_imgpixel_nsplit];
		for (j=0; j < image_pixel_grid->y_N; j++) {
			for (i=0; i < image_pixel_grid->x_N; i++) {
				for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
					ii = i*default_imgpixel_nsplit + subcell_index / default_imgpixel_nsplit;
					jj = j*default_imgpixel_nsplit + subcell_index % default_imgpixel_nsplit;
					image_pixel_grid->image_pixel_i_from_subcell_ii[ii] = i;
					image_pixel_grid->image_pixel_j_from_subcell_jj[jj] = j;
				}
			}
		}
	}

	for (j=0; j < image_pixel_grid->y_N; j++) {
		for (i=0; i < image_pixel_grid->x_N; i++) {
			if (image_pixel_grid->maps_to_source_pixel[i][j]) {
				image_pixel_grid->active_image_pixel_i[image_pixel_index] = i;
				image_pixel_grid->active_image_pixel_j[image_pixel_index] = j;
				image_pixel_grid->pixel_index[i][j] = image_pixel_index++;
				if (psf_supersampling) {
					nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]);
					if (image_pixel_grid->nsplits[i][j] != default_imgpixel_nsplit) die("nsplit has to be the same for all pixels to use supersampling (pixel (%i,%i), nsplits: %i vs %i)",i,j,image_pixel_grid->nsplits[i][j],default_imgpixel_nsplit);
					for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
						image_pixel_grid->active_image_pixel_i_ss[image_subpixel_index] = i;
						image_pixel_grid->active_image_pixel_j_ss[image_subpixel_index] = j;
						ii = i*default_imgpixel_nsplit + subcell_index / default_imgpixel_nsplit;
						jj = j*default_imgpixel_nsplit + subcell_index % default_imgpixel_nsplit;
						image_pixel_grid->active_image_subpixel_ii[image_subpixel_index] = ii;
						image_pixel_grid->active_image_subpixel_jj[image_subpixel_index] = jj;

						//cout << "SUBCELL: " << image_pixel_grid->active_image_subpixel_ii[image_subpixel_index] << " " << image_pixel_grid->active_image_subpixel_jj[image_subpixel_index] << " " << image_pixel_grid->subpixel_center_pts[i][j][subcell_index][0] << " " << image_pixel_grid->subpixel_center_pts[i][j][subcell_index][1] << endl;

						image_pixel_grid->active_image_subpixel_ss[image_subpixel_index] = subcell_index;
						image_pixel_grid->subpixel_index[ii][jj] = image_subpixel_index++;
						//cout << image_subpixel_index << " (total=" << image_n_subpixels << ")" << endl;
					}
				}
			} else {
				image_pixel_grid->pixel_index[i][j] = -1;
				if (psf_supersampling) {
					for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
						ii = i*default_imgpixel_nsplit + subcell_index / default_imgpixel_nsplit;
						jj = j*default_imgpixel_nsplit + subcell_index % default_imgpixel_nsplit;
						image_pixel_grid->subpixel_index[ii][jj] = -1;
					}
				}
			}
		}
	}
	if (image_pixel_index != image_npixels) die("Number of active pixels (%i) doesn't seem to match image_npixels (%i)",image_pixel_index,image_npixels);

	if ((verbal) and (mpi_id==0)) {
		if ((source_fit_mode==Delaunay_Source) and (image_pixel_grid->delaunay_srcgrid != NULL)) cout << "source # of pixels: " << image_pixel_grid->delaunay_srcgrid->n_srcpts << ", # of active pixels: " << source_npixels << endl;
		else cout << "source # of pixels: " << cartesian_srcgrid->number_of_pixels << ", counted up as " << tot_npixels_count << ", # of active pixels: " << source_npixels << endl;
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for assigning pixel mappings: " << wtime << endl;
	}
#endif

	return true;
}

void QLens::assign_foreground_mappings(const int zsrc_i, const bool use_data)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	image_npixels_fgmask = 0;
	int i,j;
	for (j=0; j < image_pixel_grid->y_N; j++) {
		for (i=0; i < image_pixel_grid->x_N; i++) {
			if ((!image_pixel_data) or (!use_data) or (image_pixel_data->foreground_mask[i][j])) {
				image_npixels_fgmask++;
			}
		}
	}
	if (image_npixels_fgmask==0) die("no pixels in foreground mask");

	if (image_pixel_grid->active_image_pixel_i_fgmask != NULL) delete[] image_pixel_grid->active_image_pixel_i_fgmask;
	if (image_pixel_grid->active_image_pixel_j_fgmask != NULL) delete[] image_pixel_grid->active_image_pixel_j_fgmask;
	image_pixel_grid->active_image_pixel_i_fgmask = new int[image_npixels_fgmask];
	image_pixel_grid->active_image_pixel_j_fgmask = new int[image_npixels_fgmask];
	int image_pixel_index=0;
	for (j=0; j < image_pixel_grid->y_N; j++) {
		for (i=0; i < image_pixel_grid->x_N; i++) {
			if ((!image_pixel_data) or (!use_data) or (image_pixel_data->foreground_mask[i][j])) {
				image_pixel_grid->active_image_pixel_i_fgmask[image_pixel_index] = i;
				image_pixel_grid->active_image_pixel_j_fgmask[image_pixel_index] = j;
				//cout << "Assigining " << image_pixel_index << " to (" << i << "," << j << ")" << endl;
				//image_pixel_index++;
				image_pixel_grid->pixel_index_fgmask[i][j] = image_pixel_index++;
			} else image_pixel_grid->pixel_index_fgmask[i][j] = -1;
		}
	}
	sbprofile_surface_brightness = new double[image_npixels_fgmask];
	for (int i=0; i < image_npixels_fgmask; i++) sbprofile_surface_brightness[i] = 0;
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for assigning foreground pixel mappings: " << wtime << endl;
	}
#endif
}

void QLens::initialize_pixel_matrices(const int zsrc_i, bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	SourcePixelGrid *cartesian_srcgrid = image_pixel_grid->cartesian_srcgrid;
	if (Lmatrix != NULL) die("Lmatrix already initialized");
	if (source_pixel_vector != NULL) die("source surface brightness vector already initialized");
	if (image_surface_brightness != NULL) die("image surface brightness vector already initialized");
	if (image_pixel_grid->active_image_pixel_i == NULL) die("Need to assign pixel mappings before initializing pixel matrices");
	image_surface_brightness = new double[image_npixels];
	if (psf_supersampling) image_surface_brightness_supersampled = new double[image_n_subpixels];
	imgpixel_covinv_vector = new double[image_npixels];
	source_pixel_vector = new double[source_n_amps];
	point_image_surface_brightness = new double[image_npixels];
	if ((use_lum_weighted_regularization) or (use_distance_weighted_regularization) or (use_mag_weighted_regularization)) {
		reg_weight_factor = new double[source_npixels];
		//lumreg_pixel_weights = new double[source_npixels];
	}
	//if (use_second_covariance_kernel) {
		//reg_weight_factor2 = new double[source_npixels];
	//}

	if (use_noise_map) {
		int ii,i,j;
		for (ii=0; ii < image_npixels; ii++) {
			i = image_pixel_grid->active_image_pixel_i[ii];
			j = image_pixel_grid->active_image_pixel_j[ii];
			imgpixel_covinv_vector[ii] = image_pixel_data->covinv_map[i][j];
		}
	}

	bool delaunay = false;
	if (source_fit_mode==Delaunay_Source) delaunay = true;

	if (delaunay) {
		Lmatrix_n_elements = image_pixel_grid->count_nonzero_source_pixel_mappings_delaunay();
	} else {
		if (n_image_prior) {
			source_pixel_n_images = new double[source_n_amps];
			cartesian_srcgrid->fill_n_image_vector();
		}
		Lmatrix_n_elements = image_pixel_grid->count_nonzero_source_pixel_mappings_cartesian();
	}
	if ((mpi_id==0) and (verbal)) cout << "Expected Lmatrix_n_elements=" << Lmatrix_n_elements << endl << flush;
	Lmatrix_index = new int[Lmatrix_n_elements];
	if (!psf_supersampling) image_pixel_location_Lmatrix = new int[image_npixels+1];
	else image_pixel_location_Lmatrix = new int[image_n_subpixels+1];
	Lmatrix = new double[Lmatrix_n_elements];
	if (include_imgfluxes_in_inversion) {
		int nimgs = 0;
		for (int i=0; i < point_imgs.size(); i++) nimgs += point_imgs[i].size();
		Lmatrix_transpose_ptimg_amps.input(nimgs,image_npixels);
	} else if (include_srcflux_in_inversion) {
		Lmatrix_transpose_ptimg_amps.input(point_imgs.size(),image_npixels);
	}

	if ((mpi_id==0) and (verbal)) cout << "Creating Lmatrix...\n";
	if (!psf_supersampling) assign_Lmatrix(zsrc_i,delaunay,verbal);
	else assign_Lmatrix_supersampled(zsrc_i,delaunay,verbal);
}

void QLens::count_shapelet_npixels(const int zsrc_i)
{
	double nmax;
	source_npixels = 0;
	for (int i=0; i < n_sb; i++) {
		if ((sb_list[i]->sbtype==SHAPELET) and ((zsrc_i < 0) or (sbprofile_redshift_idx[i]==zsrc_i))) {
			nmax = *(sb_list[i]->indxptr);
			source_npixels += nmax*nmax;
			break;
		}
	}
}

void QLens::initialize_pixel_matrices_shapelets(const int zsrc_i, bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	//if (source_pixel_vector != NULL) die("source surface brightness vector already initialized");
	vectorize_image_pixel_surface_brightness(zsrc_i, true);
	count_shapelet_npixels(zsrc_i);
	source_n_amps = source_npixels;
	if (include_imgfluxes_in_inversion) {
		for (int i=0; i < point_imgs.size(); i++) {
			source_n_amps += point_imgs[i].size(); // in this case, source amplitudes include point image amplitudes as well as pixel values
		}
	} else if (include_srcflux_in_inversion) {
		source_n_amps += point_imgs.size();
	}

	point_image_surface_brightness = new double[image_npixels];

	if (source_n_amps <= 0) die("no shapelet or point source amplitude parameters found");
	source_pixel_vector = new double[source_n_amps];
	imgpixel_covinv_vector = new double[image_npixels];
	if ((use_lum_weighted_regularization) or (use_distance_weighted_regularization) or (use_mag_weighted_regularization)) {
		reg_weight_factor = new double[source_npixels];
		for (int i=0; i < source_npixels; i++) reg_weight_factor[i] = 1.0;
		//lumreg_pixel_weights = new double[source_npixels];
	}
	//if (use_second_covariance_kernel) {
		//reg_weight_factor2 = new double[source_npixels];
	//}

	if (use_noise_map) {
		int ii,i,j;
		for (ii=0; ii < image_npixels; ii++) {
			i = image_pixel_grid->active_image_pixel_i[ii];
			j = image_pixel_grid->active_image_pixel_j[ii];
			imgpixel_covinv_vector[ii] = image_pixel_data->covinv_map[i][j];
		}
	}
	if ((mpi_id==0) and (verbal)) cout << "Creating shapelet Lmatrix...\n";
	Lmatrix_dense.input(image_npixels,source_n_amps);
	Lmatrix_dense = 0;
	if (include_imgfluxes_in_inversion) {
		int nimgs = 0;
		for (int i=0; i < point_imgs.size(); i++) nimgs += point_imgs[i].size();
		Lmatrix_transpose_ptimg_amps.input(nimgs,image_npixels);
	} else if (include_srcflux_in_inversion) {
		Lmatrix_transpose_ptimg_amps.input(point_imgs.size(),image_npixels);
	}

	assign_Lmatrix_shapelets(zsrc_i,verbal);
}

void QLens::clear_pixel_matrices(const int zsrc_i)
{
	if (image_surface_brightness != NULL) delete[] image_surface_brightness;
	if (imgpixel_covinv_vector != NULL) delete[] imgpixel_covinv_vector;
	if (point_image_surface_brightness != NULL) delete[] point_image_surface_brightness;
	if (sbprofile_surface_brightness != NULL) delete[] sbprofile_surface_brightness;
	if (source_pixel_vector != NULL) delete[] source_pixel_vector;
	if (reg_weight_factor != NULL) delete[] reg_weight_factor;
	//if (reg_weight_factor2 != NULL) delete[] reg_weight_factor2;
	//if (lumreg_pixel_weights != NULL) delete[] lumreg_pixel_weights;
	if (image_pixel_location_Lmatrix != NULL) delete[] image_pixel_location_Lmatrix;
	if (source_pixel_location_Lmatrix != NULL) delete[] source_pixel_location_Lmatrix;
	if (Lmatrix_index != NULL) delete[] Lmatrix_index;
	if (Lmatrix != NULL) delete[] Lmatrix;
	image_surface_brightness = NULL;
	imgpixel_covinv_vector = NULL;
	point_image_surface_brightness = NULL;
	sbprofile_surface_brightness = NULL;
	source_pixel_vector = NULL;
	reg_weight_factor = NULL;
	//reg_weight_factor2 = NULL;
	image_pixel_location_Lmatrix = NULL;
	source_pixel_location_Lmatrix = NULL;
	Lmatrix = NULL;
	Lmatrix_index = NULL;
	//lumreg_pixel_weights = NULL;
	/*
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	if (image_pixel_grid->active_image_pixel_i != NULL) delete[] image_pixel_grid->active_image_pixel_i;
	if (image_pixel_grid->active_image_pixel_j != NULL) delete[] image_pixel_grid->active_image_pixel_j;
	if (image_pixel_grid->active_image_pixel_i_ss != NULL) delete[] image_pixel_grid->active_image_pixel_i_ss;
	if (image_pixel_grid->active_image_pixel_j_ss != NULL) delete[] image_pixel_grid->active_image_pixel_j_ss;
	if (image_pixel_grid->active_image_subpixel_ss != NULL) delete[] image_pixel_grid->active_image_subpixel_ss;
	if (image_pixel_grid->active_image_subpixel_ii != NULL) delete[] image_pixel_grid->active_image_subpixel_ii;
	if (image_pixel_grid->active_image_subpixel_jj != NULL) delete[] image_pixel_grid->active_image_subpixel_jj;
	if (image_pixel_grid->image_pixel_i_from_subcell_ii != NULL) delete[] image_pixel_grid->image_pixel_i_from_subcell_ii;
	if (image_pixel_grid->image_pixel_j_from_subcell_jj != NULL) delete[] image_pixel_grid->image_pixel_j_from_subcell_jj;
	if (image_pixel_grid->active_image_pixel_i_fgmask != NULL) delete[] image_pixel_grid->active_image_pixel_i_fgmask;
	if (image_pixel_grid->active_image_pixel_j_fgmask != NULL) delete[] image_pixel_grid->active_image_pixel_j_fgmask;
	image_pixel_grid->active_image_pixel_i = NULL;
	image_pixel_grid->active_image_pixel_j = NULL;
	image_pixel_grid->active_image_subpixel_ii = NULL;
	image_pixel_grid->active_image_subpixel_jj = NULL;
	image_pixel_grid->active_image_pixel_i_ss = NULL;
	image_pixel_grid->active_image_pixel_j_ss = NULL;
	image_pixel_grid->active_image_subpixel_ss = NULL;
	image_pixel_grid->image_pixel_i_from_subcell_ii = NULL;
	image_pixel_grid->image_pixel_j_from_subcell_jj = NULL;
	image_pixel_grid->active_image_pixel_i_fgmask = NULL;
	image_pixel_grid->active_image_pixel_j_fgmask = NULL;
	*/
	/*
	// I don't think this is necessary, so commented out...these will be cleared when pixel mappings are assigned
	int nsubpix = INTSQR(default_imgpixel_nsplit);
	int i,j,k;
	int *ptr;
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[0];
	for (i=0; i < image_pixel_grid->x_N; i++) {
		for (j=0; j < image_pixel_grid->y_N; j++) {
			image_pixel_grid->mapped_cartesian_srcpixels[i][j].clear();
			image_pixel_grid->mapped_delaunay_srcpixels[i][j].clear();
			ptr = image_pixel_grid->n_mapped_srcpixels[i][j];
			for (k=0; k < nsubpix; k++) {
				(*ptr++) = 0;
			}
		}
	}
	*/
	if ((n_image_prior) and (source_fit_mode==Cartesian_Source)) {
		if (source_pixel_n_images != NULL) delete[] source_pixel_n_images;
		source_pixel_n_images = NULL;
	}
}

/*
void QLens::clear_pixel_matrices_dense()
{
	if (image_surface_brightness != NULL) delete[] image_surface_brightness;
	if (sbprofile_surface_brightness != NULL) delete[] sbprofile_surface_brightness;
	if (source_pixel_vector != NULL) delete[] source_pixel_vector;
	if (active_image_pixel_i != NULL) delete[] active_image_pixel_i;
	if (active_image_pixel_j != NULL) delete[] active_image_pixel_j;
	image_surface_brightness = NULL;
	sbprofile_surface_brightness = NULL;
	source_pixel_vector = NULL;
	active_image_pixel_i = NULL;
	active_image_pixel_j = NULL;
}
*/

void QLens::assign_Lmatrix(const int zsrc_i, const bool delaunay, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	SourcePixelGrid *cartesian_srcgrid = image_pixel_grid->cartesian_srcgrid;
	int img_index;
	int index;
	int i,j;
	Lmatrix_rows = new vector<double>[image_npixels];
	Lmatrix_index_rows = new vector<int>[image_npixels];
	int *Lmatrix_row_nn = new int[image_npixels];
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	if ((!delaunay) and (image_pixel_grid->ray_tracing_method == Area_Overlap))
	{
		lensvector *corners[4];
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			#pragma omp for private(img_index,i,j,index,corners) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				index=0;
				i = image_pixel_grid->active_image_pixel_i[img_index];
				j = image_pixel_grid->active_image_pixel_j[img_index];
				corners[0] = &image_pixel_grid->corner_sourcepts[i][j];
				corners[1] = &image_pixel_grid->corner_sourcepts[i][j+1];
				corners[2] = &image_pixel_grid->corner_sourcepts[i+1][j];
				corners[3] = &image_pixel_grid->corner_sourcepts[i+1][j+1];
				cartesian_srcgrid->calculate_Lmatrix_overlap(img_index,i,j,index,corners,&image_pixel_grid->twist_pts[i][j],image_pixel_grid->twist_status[i][j],thread);
				Lmatrix_row_nn[img_index] = index;
			}
		}
	}
	else // interpolate
	{
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			int nsubpix,subcell_index;
			lensvector *center_srcpt;

			if ((split_imgpixels) and (!raytrace_using_pixel_centers)) {
				#pragma omp for private(img_index,i,j,nsubpix,index,center_srcpt) schedule(dynamic)
				for (img_index=0; img_index < image_npixels; img_index++) {
					index = 0;
					i = image_pixel_grid->active_image_pixel_i[img_index];
					j = image_pixel_grid->active_image_pixel_j[img_index];

					nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]);
					center_srcpt = image_pixel_grid->subpixel_center_sourcepts[i][j];
					if (delaunay) {
						if (image_pixel_grid->delaunay_srcgrid != NULL) {
							for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
								image_pixel_grid->delaunay_srcgrid->calculate_Lmatrix(img_index,image_pixel_grid->mapped_delaunay_srcpixels[i][j].data(),image_pixel_grid->n_mapped_srcpixels[i][j],index,center_srcpt[subcell_index],subcell_index,1.0/nsubpix,thread);
							}
						}
					} else {
						for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
							cartesian_srcgrid->calculate_Lmatrix_interpolate(img_index,image_pixel_grid->mapped_cartesian_srcpixels[i][j],index,center_srcpt[subcell_index],subcell_index,1.0/nsubpix,thread);
						}
					}
					Lmatrix_row_nn[img_index] = index;
				}
			} else {
				#pragma omp for private(img_index,i,j,index) schedule(dynamic)	
				for (img_index=0; img_index < image_npixels; img_index++) {
					index = 0;
					i = image_pixel_grid->active_image_pixel_i[img_index];
					j = image_pixel_grid->active_image_pixel_j[img_index];
					if (delaunay) {
						if (image_pixel_grid->delaunay_srcgrid != NULL) {
							image_pixel_grid->delaunay_srcgrid->calculate_Lmatrix(img_index,image_pixel_grid->mapped_delaunay_srcpixels[i][j].data(),image_pixel_grid->n_mapped_srcpixels[i][j],index,image_pixel_grid->center_sourcepts[i][j],0,1.0,thread);
						}
					} else {
						cartesian_srcgrid->calculate_Lmatrix_interpolate(img_index,image_pixel_grid->mapped_cartesian_srcpixels[i][j],index,image_pixel_grid->center_sourcepts[i][j],0,1.0,thread);
					}
					Lmatrix_row_nn[img_index] = index;
				}
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
			Lmatrix[index] = Lmatrix_rows[i][j];
			Lmatrix_index[index] = Lmatrix_index_rows[i][j];
			index++;
		}
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for constructing Lmatrix: " << wtime << endl;
	}
#endif
	if ((mpi_id==0) and (verbal)) {
		int Lmatrix_ntot = source_n_amps*image_npixels;
		double sparseness = ((double) Lmatrix_n_elements)/Lmatrix_ntot;
		cout << "image has " << image_pixel_grid->n_active_pixels << " active pixels, Lmatrix has " << Lmatrix_n_elements << " nonzero elements (sparseness " << sparseness << ")\n";
	}

	delete[] Lmatrix_row_nn;
	delete[] Lmatrix_rows;
	delete[] Lmatrix_index_rows;
}

void QLens::assign_Lmatrix_supersampled(const int zsrc_i, const bool delaunay, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	SourcePixelGrid *cartesian_srcgrid = image_pixel_grid->cartesian_srcgrid;
	int img_index;
	int index;
	int i,j;
	Lmatrix_rows = new vector<double>[image_n_subpixels];
	Lmatrix_index_rows = new vector<int>[image_n_subpixels];
	int *Lmatrix_row_nn = new int[image_n_subpixels];
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		int nsubpix,subcell_index;
		lensvector *center_srcpt;

		#pragma omp for private(img_index,i,j,nsubpix,index,center_srcpt) schedule(dynamic)
		for (img_index=0; img_index < image_n_subpixels; img_index++) {
			index = 0;
			i = image_pixel_grid->active_image_pixel_i_ss[img_index];
			j = image_pixel_grid->active_image_pixel_j_ss[img_index];
			subcell_index = image_pixel_grid->active_image_subpixel_ss[img_index];

			center_srcpt = image_pixel_grid->subpixel_center_sourcepts[i][j];
			if (delaunay) {
				if (image_pixel_grid->delaunay_srcgrid != NULL) { // this might be the case if we're only doing point sources but happen to be in delaunay source mode
						image_pixel_grid->delaunay_srcgrid->calculate_Lmatrix(img_index,image_pixel_grid->mapped_delaunay_srcpixels[i][j].data(),image_pixel_grid->n_mapped_srcpixels[i][j],index,center_srcpt[subcell_index],subcell_index,1.0,thread);
				}
			} else {
				cartesian_srcgrid->calculate_Lmatrix_interpolate(img_index,image_pixel_grid->mapped_cartesian_srcpixels[i][j],index,center_srcpt[subcell_index],subcell_index,1.0,thread);
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
			Lmatrix[index] = Lmatrix_rows[i][j];
			Lmatrix_index[index] = Lmatrix_index_rows[i][j];
			index++;
		}
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for constructing Lmatrix: " << wtime << endl;
	}
#endif
	if ((mpi_id==0) and (verbal)) {
		int Lmatrix_ntot = source_n_amps*image_n_subpixels;
		double sparseness = ((double) Lmatrix_n_elements)/Lmatrix_ntot;
		cout << "image has " << image_pixel_grid->n_active_pixels << " active pixels, Lmatrix has " << Lmatrix_n_elements << " nonzero elements (sparseness " << sparseness << ")\n";
	}

	delete[] Lmatrix_row_nn;
	delete[] Lmatrix_rows;
	delete[] Lmatrix_index_rows;
}

void QLens::assign_Lmatrix_shapelets(const int zsrc_i, bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	int img_index;
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int i,j,k,n_shapelet_sets = 0;
	for (i=0; i < n_sb; i++) {
		if ((sb_list[i]->sbtype==SHAPELET) and (((!sb_list[i]->is_lensed) and (zsrc_i<0)) or (sbprofile_redshift_idx[i]==zsrc_i))) n_shapelet_sets++;
	}
	if (n_shapelet_sets==0) return;

	SB_Profile** shapelet;
	shapelet = new SB_Profile*[n_shapelet_sets];
	bool at_least_one_lensed_shapelet = false;
	for (i=0,j=0; i < n_sb; i++) {
		if (sb_list[i]->sbtype==SHAPELET) {
			if (((!sb_list[i]->is_lensed) and (zsrc_i<0)) or (sbprofile_redshift_idx[i]==zsrc_i)) {
				shapelet[j++] = sb_list[i];
				if (sb_list[i]->is_lensed) at_least_one_lensed_shapelet = true;
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
		lensvector *center_srcpt, *center_pt;

		if (split_imgpixels) {
			#pragma omp for private(img_index,i,j,k,nsubpix,center_srcpt,center_pt) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				i = image_pixel_grid->active_image_pixel_i[img_index];
				j = image_pixel_grid->active_image_pixel_j[img_index];

				nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]);
				center_srcpt = image_pixel_grid->subpixel_center_sourcepts[i][j];
				center_pt = image_pixel_grid->subpixel_center_pts[i][j];
				for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
					double *Lmatptr = Lmatrix_dense.subarray(img_index);
					for (k=0; k < n_shapelet_sets; k++) {
						if (shapelet[k]->is_lensed) {
							//Note that calculate_Lmatrix_elements(...) will increment Lmatptr as it goes
							shapelet[k]->calculate_Lmatrix_elements(center_srcpt[subcell_index][0],center_srcpt[subcell_index][1],Lmatptr,1.0/nsubpix);
							//shapelet[k]->calculate_Lmatrix_elements(image_pixel_grid->center_sourcepts[i][j][0],image_pixel_grid->center_sourcepts[i][j][1],Lmatptr,1.0/nsubpix);
							//cout << "cell " << i << "," << j << ": " << image_pixel_grid->center_sourcepts[i][j][0] << " " << image_pixel_grid->center_sourcepts[i][j][1] << " vs " << center_pt[subcell_index][0] << " " << center_pt[subcell_index][1] << endl;
						} else {
							shapelet[k]->calculate_Lmatrix_elements(center_pt[subcell_index][0],center_pt[subcell_index][1],Lmatptr,1.0/nsubpix);
						}
					}
				}
			}
		} else {
			lensvector center, center_srcpt;
			#pragma omp for private(img_index,i,j,center_srcpt) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				i = image_pixel_grid->active_image_pixel_i[img_index];
				j = image_pixel_grid->active_image_pixel_j[img_index];

				double *Lmatptr = Lmatrix_dense.subarray(img_index);
				for (k=0; k < n_shapelet_sets; k++) {
					if (shapelet[k]->is_lensed) {
						center_srcpt = image_pixel_grid->center_sourcepts[i][j];
						shapelet[k]->calculate_Lmatrix_elements(center_srcpt[0],center_srcpt[1],Lmatptr,1.0);
					} else {
						center = image_pixel_grid->center_pts[i][j];
						shapelet[k]->calculate_Lmatrix_elements(center[0],center[1],Lmatptr,1.0);
					}
				}
			}
		}
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for constructing shapelet Lmatrix: " << wtime << endl;
	}
#endif
	delete[] shapelet;
}

void QLens::PSF_convolution_Lmatrix(const int zsrc_i, bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
#ifdef USE_MPI
	MPI_Comm sub_comm;
	if (psf_convolution_mpi) {
		MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
	}
#endif
	if (psf_supersampling) die("PSF supersampling has not been implemented with sparse Lmatrix");
	if (use_input_psf_matrix) {
		if (psf_matrix == NULL) return;
	}
	else if (generate_PSF_matrix(image_pixel_grid->pixel_xlength,image_pixel_grid->pixel_ylength,psf_supersampling)==false) return;
	if ((mpi_id==0) and (verbal)) cout << "Beginning PSF convolution (sparse)...\n";
	double nx_half, ny_half;
	nx_half = psf_npixels_x/2;
	ny_half = psf_npixels_y/2;

	int *Lmatrix_psf_row_nn = new int[image_npixels];
	vector<double> *Lmatrix_psf_rows = new vector<double>[image_npixels];
	vector<int> *Lmatrix_psf_index_rows = new vector<int>[image_npixels];

	// If the PSF is sufficiently wide, it may save time to MPI the PSF convolution by setting psf_convolution_mpi to 'true'. This option is off by default.
	int mpi_chunk, mpi_start, mpi_end;
	if (psf_convolution_mpi) {
		mpi_chunk = image_npixels / group_np;
		mpi_start = group_id*mpi_chunk;
		if (group_id == group_np-1) mpi_chunk += (image_npixels % group_np); // assign the remainder elements to the last mpi process
		mpi_end = mpi_start + mpi_chunk;
	} else {
		mpi_start = 0; mpi_end = image_npixels;
	}

	int i,j,k,l,m;
	int Lmatrix_psf_nn=0;
	if (source_npixels > 0) {
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime0 = omp_get_wtime();
		}
#endif
		int psf_k, psf_l;
		int img_index1, img_index2, src_index, col_index;
		int index;
		bool new_entry;
		int Lmatrix_psf_nn_part=0;
		#pragma omp parallel for private(m,k,l,i,j,img_index1,img_index2,src_index,col_index,psf_k,psf_l,index,new_entry) schedule(static) reduction(+:Lmatrix_psf_nn_part)
		for (img_index1=mpi_start; img_index1 < mpi_end; img_index1++)
		{ // this loops over columns of the PSF blurring matrix
			int col_i=0;
			Lmatrix_psf_row_nn[img_index1] = 0;
			k = image_pixel_grid->active_image_pixel_i[img_index1];
			l = image_pixel_grid->active_image_pixel_j[img_index1];
			for (psf_k=0; psf_k < psf_npixels_x; psf_k++) {
				i = k + nx_half - psf_k; // Note, 'k' is the index for the convolved image, so we have k = i - nx_half + psf_k
				if ((i >= 0) and (i < image_pixel_grid->x_N)) {
					for (psf_l=0; psf_l < psf_npixels_y; psf_l++) {
						j = l + ny_half - psf_l; // Note, 'l' is the index for the convolved image, so we have l = j - ny_half + psf_l
						if ((j >= 0) and (j < image_pixel_grid->y_N)) {
							if (image_pixel_grid->maps_to_source_pixel[i][j]) {
								img_index2 = image_pixel_grid->pixel_index[i][j];

								for (index=image_pixel_location_Lmatrix[img_index2]; index < image_pixel_location_Lmatrix[img_index2+1]; index++) {
									if (Lmatrix[index] != 0) {
										src_index = Lmatrix_index[index];
										new_entry = true;
										for (m=0; m < Lmatrix_psf_row_nn[img_index1]; m++) {
											if (Lmatrix_psf_index_rows[img_index1][m]==src_index) { col_index=m; new_entry=false; }
										}
										if (new_entry) {
											Lmatrix_psf_rows[img_index1].push_back(psf_matrix[psf_k][psf_l]*Lmatrix[index]);
											Lmatrix_psf_index_rows[img_index1].push_back(src_index);
											Lmatrix_psf_row_nn[img_index1]++;
											col_i++;
										} else {
											Lmatrix_psf_rows[img_index1][col_index] += psf_matrix[psf_k][psf_l]*Lmatrix[index];
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

#ifdef USE_MPI
		if (psf_convolution_mpi)
			MPI_Allreduce(&Lmatrix_psf_nn_part, &Lmatrix_psf_nn, 1, MPI_INT, MPI_SUM, sub_comm);
		else
			Lmatrix_psf_nn = Lmatrix_psf_nn_part;
#else
		Lmatrix_psf_nn = Lmatrix_psf_nn_part;
#endif

#ifdef USE_MPI
		if (psf_convolution_mpi) {
			int id, chunk, start, end, length;
			for (id=0; id < group_np; id++) {
				chunk = image_npixels / group_np;
				start = id*chunk;
				if (id == group_np-1) chunk += (image_npixels % group_np); // assign the remainder elements to the last mpi process
				MPI_Bcast(Lmatrix_psf_row_nn + start,chunk,MPI_INT,id,sub_comm);
			}
		}
#endif
	} else {
		for (int img_index=0; img_index < image_npixels; img_index++) {
			Lmatrix_psf_row_nn[img_index] = 0;
		}
	}


	if (include_imgfluxes_in_inversion) {
		double *Lmatptr;
		i=0;
		for (j=0; j < point_imgs.size(); j++) {
			for (k=0; k < point_imgs[j].size(); k++) {
				Lmatptr = Lmatrix_transpose_ptimg_amps.subarray(i);
				image_pixel_grid->generate_point_images(point_imgs[j], Lmatptr, false, -1, k);
				i++;
			}
		}
		int src_amp_i;
		double *Lmatrix_transpose_line;
		i=0;
		for (j=0; j < point_imgs.size(); j++) {
			for (k=0; k < point_imgs[j].size(); k++) {
				src_amp_i = source_npixels + i;
				Lmatrix_transpose_line = Lmatrix_transpose_ptimg_amps[i];
				for (int img_index=0; img_index < image_npixels; img_index++) {
					if (Lmatrix_transpose_line[img_index] != 0) {
						Lmatrix_psf_rows[img_index].push_back(Lmatrix_transpose_line[img_index]);
						Lmatrix_psf_index_rows[img_index].push_back(src_amp_i);
						Lmatrix_psf_row_nn[img_index]++;
						Lmatrix_psf_nn++;
					}
				}
				i++;
			}
		}
	} else if (include_srcflux_in_inversion) {
		double *Lmatptr;
		for (j=0; j < point_imgs.size(); j++) {
			Lmatptr = Lmatrix_transpose_ptimg_amps.subarray(j);
			image_pixel_grid->generate_point_images(point_imgs[j], Lmatptr, false, 1.0);
		}
		int src_amp_i;
		double *Lmatrix_transpose_line;
		for (j=0; j < point_imgs.size(); j++) {
			src_amp_i = source_npixels + j;
			Lmatrix_transpose_line = Lmatrix_transpose_ptimg_amps[j];
			for (int img_index=0; img_index < image_npixels; img_index++) {
				if (Lmatrix_transpose_line[img_index] != 0) {
					Lmatrix_psf_rows[img_index].push_back(Lmatrix_transpose_line[img_index]);
					Lmatrix_psf_index_rows[img_index].push_back(src_amp_i);
					Lmatrix_psf_row_nn[img_index]++;
					Lmatrix_psf_nn++;
				}
			}
		}
	}


	int *image_pixel_location_Lmatrix_psf = new int[image_npixels+1];
	image_pixel_location_Lmatrix_psf[0] = 0;
	for (m=0; m < image_npixels; m++) {
		image_pixel_location_Lmatrix_psf[m+1] = image_pixel_location_Lmatrix_psf[m] + Lmatrix_psf_row_nn[m];
	}

	double *Lmatrix_psf = new double[Lmatrix_psf_nn];
	int *Lmatrix_index_psf = new int[Lmatrix_psf_nn];

	int indx;
	for (m=mpi_start; m < mpi_end; m++) {
		indx = image_pixel_location_Lmatrix_psf[m];
		for (j=0; j < Lmatrix_psf_row_nn[m]; j++) {
			Lmatrix_psf[indx+j] = Lmatrix_psf_rows[m][j];
			Lmatrix_index_psf[indx+j] = Lmatrix_psf_index_rows[m][j];
		}
	}

#ifdef USE_MPI
	if (psf_convolution_mpi) {
		int id, chunk, start, end, length;
		for (id=0; id < group_np; id++) {
			chunk = image_npixels / group_np;
			start = id*chunk;
			if (id == group_np-1) chunk += (image_npixels % group_np); // assign the remainder elements to the last mpi process
			end = start + chunk;
			length = image_pixel_location_Lmatrix_psf[end] - image_pixel_location_Lmatrix_psf[start];
			MPI_Bcast(Lmatrix_psf + image_pixel_location_Lmatrix_psf[start],length,MPI_DOUBLE,id,sub_comm);
			MPI_Bcast(Lmatrix_index_psf + image_pixel_location_Lmatrix_psf[start],length,MPI_INT,id,sub_comm);
		}
		MPI_Comm_free(&sub_comm);
	}
#endif

	if ((mpi_id==0) and (verbal)) cout << "Lmatrix after PSF convolution: Lmatrix now has " << indx << " nonzero elements\n";

	delete[] Lmatrix;
	delete[] Lmatrix_index;
	delete[] image_pixel_location_Lmatrix;
	Lmatrix = Lmatrix_psf;
	Lmatrix_index = Lmatrix_index_psf;
	image_pixel_location_Lmatrix = image_pixel_location_Lmatrix_psf;
	Lmatrix_n_elements = Lmatrix_psf_nn;

	delete[] Lmatrix_psf_row_nn;
	delete[] Lmatrix_psf_rows;
	delete[] Lmatrix_psf_index_rows;

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating PSF convolution of Lmatrix: " << wtime << endl;
	}
#endif
}

void QLens::convert_Lmatrix_to_dense()
{
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime0 = omp_get_wtime();
		}
#endif
	int i,j,npix;
	dmatrix *Lptr;
	if (!psf_supersampling) {
		npix = image_npixels;
		Lptr = &Lmatrix_dense;
	}
	else {
		npix = image_n_subpixels;
		Lptr = &Lmatrix_supersampled;
	}
	(*Lptr).input(npix,source_n_amps);
	(*Lptr) = 0;
	for (i=0; i < npix; i++) {
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			(*Lptr)[i][Lmatrix_index[j]] += Lmatrix[j];
		}
	}
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for converting Lmatrix to dense form: " << wtime << endl;
	}
#endif
}

bool ImagePixelGrid::setup_FFT_convolution(const bool supersampling, const bool verbal)
{
#ifdef USE_OPENMP
	if (lens->show_wtime) {
		lens->wtime0 = omp_get_wtime();
	}
#endif
	int npix;
	int *pixel_map_ii, *pixel_map_jj;
	double **psf;
	int psf_nx, psf_ny;
	if (!supersampling) {
		psf = lens->psf_matrix;
		psf_nx = lens->psf_npixels_x;
		psf_ny = lens->psf_npixels_y;
		npix = lens->image_npixels;
		pixel_map_ii = active_image_pixel_i;
		pixel_map_jj = active_image_pixel_j;
	} else {
		psf = lens->supersampled_psf_matrix;
		psf_nx = lens->supersampled_psf_npixels_x;
		psf_ny = lens->supersampled_psf_npixels_y;
		npix = lens->image_n_subpixels;
		pixel_map_ii = active_image_subpixel_ii;
		pixel_map_jj = active_image_subpixel_jj;
		//cout << "PSF_NX=" << psf_nx << " PSF_NY=" << psf_ny << " npix=" << image_n_subpixels << endl;
	}

	if (lens->use_input_psf_matrix) {
		if (psf == NULL) return false;
	} else {
		if ((lens->psf_width_x==0) and (lens->psf_width_y==0)) return false;
		else if (lens->generate_PSF_matrix(pixel_xlength,pixel_ylength,supersampling)==false) {
			if (verbal) warn("could not generate_PSF matrix");
			return false;
		}
		if (lens->mpi_id==0) cout << "generated PSF matrix" << endl;
	}
	int nx_half, ny_half;
	nx_half = psf_nx/2;
	ny_half = psf_ny/2;

	int i,j,ii,jj,k,img_index;
	fft_ni = 1;
	fft_nj = 1;
	fft_imin = 50000;
	fft_jmin = 50000;
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
		if ((maps_to_source_pixel[i][j]) and ((pixel_in_mask==NULL) or (pixel_in_mask[i][j]))) {
			if (ii > imax) imax = ii;
			if (jj > jmax) jmax = jj;
			if (ii < fft_imin) fft_imin = ii;
			if (jj < fft_jmin) fft_jmin = jj;
		}
	}
	il0 = 1+imax-fft_imin + psf_nx; // will pad with extra zeros to avoid edge effects (wraparound of PSF blurring)
	jl0 = 1+jmax-fft_jmin + psf_ny;

#ifdef USE_FFTW
	fft_ni = il0;
	fft_nj = jl0;
	if (fft_ni % 2 != 0) fft_ni++;
	if (fft_nj % 2 != 0) fft_nj++;
	int ncomplex = fft_nj*(fft_ni/2+1);
	int npix_conv = fft_ni*fft_nj;
	double *psf_rvec = new double[npix_conv];
	psf_transform = new complex<double>[ncomplex];
	fftw_plan fftplan_psf = fftw_plan_dft_r2c_2d(fft_nj,fft_ni,psf_rvec,reinterpret_cast<fftw_complex*>(psf_transform),FFTW_MEASURE);
	for (i=0; i < npix_conv; i++) psf_rvec[i] = 0;
	single_img_rvec = new double[npix_conv];
	img_transform = new complex<double>[ncomplex];
	fftplan = fftw_plan_dft_r2c_2d(fft_nj,fft_ni,single_img_rvec,reinterpret_cast<fftw_complex*>(img_transform),FFTW_MEASURE);
	fftplan_inverse = fftw_plan_dft_c2r_2d(fft_nj,fft_ni,reinterpret_cast<fftw_complex*>(img_transform),single_img_rvec,FFTW_MEASURE);
	for (i=0; i < npix_conv; i++) single_img_rvec[i] = 0;

	if (Lmatrix_src_npixels > 0) {
		Lmatrix_imgs_rvec = new double*[Lmatrix_src_npixels];
		Lmatrix_transform = new complex<double>*[Lmatrix_src_npixels];
		fftplans_Lmatrix = new fftw_plan[Lmatrix_src_npixels];
		fftplans_Lmatrix_inverse = new fftw_plan[Lmatrix_src_npixels];
		for (i=0; i < Lmatrix_src_npixels; i++) {
			Lmatrix_imgs_rvec[i] = new double[npix_conv];
			Lmatrix_transform[i] = new complex<double>[ncomplex];
			fftplans_Lmatrix[i] = fftw_plan_dft_r2c_2d(fft_nj,fft_ni,Lmatrix_imgs_rvec[i],reinterpret_cast<fftw_complex*>(Lmatrix_transform[i]),FFTW_MEASURE);
			fftplans_Lmatrix_inverse[i] = fftw_plan_dft_c2r_2d(fft_nj,fft_ni,reinterpret_cast<fftw_complex*>(Lmatrix_transform[i]),Lmatrix_imgs_rvec[i],FFTW_MEASURE);
			for (j=0; j < npix_conv; j++) Lmatrix_imgs_rvec[i][j] = 0;
		}
	}
#else
	while (fft_ni < il0) fft_ni *= 2; // need multiple of 2 to do FFT (note, this is only necessary with native code; it is not necessary with FFTW)
	while (fft_nj < jl0) fft_nj *= 2; // need multiple of 2 to do FFT (note, this is only necessary with native code; it is not necessary with FFTW)
	psf_zvec = new double[2*fft_ni*fft_nj];
	for (i=0; i < 2*fft_ni*fft_nj; i++) psf_zvec[i] = 0;
#endif
	int zpsf_i, zpsf_j;
	int l;
	for (i=-nx_half; i < psf_nx - nx_half; i++) {
		for (j=-ny_half; j < psf_ny - ny_half; j++) {
			zpsf_i=i;
			zpsf_j=j;
			if (zpsf_i < 0) zpsf_i += fft_ni;
			if (zpsf_j < 0) zpsf_j += fft_nj;
#ifdef USE_FFTW
			l = zpsf_j*fft_ni + zpsf_i;
			psf_rvec[l] = psf[nx_half+i][ny_half+j];
#else
			k = 2*(zpsf_j*fft_ni + zpsf_i);
			psf_zvec[k] = psf[nx_half+i][ny_half+j];
#endif
		}
	}

#ifdef USE_FFTW
	fftw_execute(fftplan_psf);
	fftw_destroy_plan(fftplan_psf);
	delete[] psf_rvec;
#else
	int nnvec[2];
	nnvec[0] = fft_ni;
	nnvec[1] = fft_nj;
	lens->fourier_transform(psf_zvec,2,nnvec,1);
#endif
#ifdef USE_OPENMP
	if (lens->show_wtime) {
		lens->wtime = omp_get_wtime() - lens->wtime0;
		if (lens->mpi_id==0) {
			cout << "Wall time for setting up FFT for convolutions: " << lens->wtime << endl;
		}
	}
#endif

	fft_convolution_is_setup = true;
	return true;
}

void QLens::cleanup_FFT_convolution_arrays()
{
	if (!use_old_pixelgrids) {
		if (image_pixel_grids == NULL) return;
		for (int zsrc_i=0; zsrc_i < n_extended_src_redshifts; zsrc_i++) {
			if (image_pixel_grids[zsrc_i]->fft_convolution_is_setup) image_pixel_grids[zsrc_i]->cleanup_FFT_convolution_arrays();
		}
	} else {
		if (image_pixel_grid0->fft_convolution_is_setup) image_pixel_grid0->cleanup_FFT_convolution_arrays();
	}
}

void ImagePixelGrid::cleanup_FFT_convolution_arrays()
{
#ifdef USE_FFTW
	delete[] psf_transform;
	psf_transform = NULL;
	if (Lmatrix_src_npixels > 0) {
		for (int i=0; i < Lmatrix_src_npixels; i++) {
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
#else
	delete[] psf_zvec;
#endif
	fft_imin=fft_jmin=fft_ni=fft_nj=0;
	fft_convolution_is_setup = false;
}

void QLens::PSF_convolution_Lmatrix_dense(const int zsrc_i, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
#ifdef USE_MPI
	MPI_Comm sub_comm;
	if (psf_convolution_mpi) {
		MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
	}
#endif

	if (source_npixels > 0) {
		if ((mpi_id==0) and (verbal)) cout << "Beginning PSF convolution (dense)...\n";

		int nx_half, ny_half;
		int psf_nx, psf_ny;
		int npix;
		int *pixel_map_ii, *pixel_map_jj;

		dmatrix *Lptr;
		if (!psf_supersampling) {
			Lptr = &Lmatrix_dense;
			npix = image_npixels;
			psf_nx = psf_npixels_x;
			psf_ny = psf_npixels_y;
			pixel_map_ii = image_pixel_grid->active_image_pixel_i;
			pixel_map_jj = image_pixel_grid->active_image_pixel_j;
		} else {
			Lptr = &Lmatrix_supersampled;
			npix = image_n_subpixels;
			psf_nx = supersampled_psf_npixels_x;
			psf_ny = supersampled_psf_npixels_y;
			pixel_map_ii = image_pixel_grid->active_image_subpixel_ii;
			pixel_map_jj = image_pixel_grid->active_image_subpixel_jj;
		}
		nx_half = psf_nx/2;
		ny_half = psf_ny/2;
		int i,j,ii,jj,k,l,img_index;

		if (fft_convolution) {
			if (!image_pixel_grid->fft_convolution_is_setup) {
				if (!image_pixel_grid->setup_FFT_convolution(psf_supersampling,verbal)) {
					warn("PSF convolution FFT failed");
					return;	
				}
			}

#ifdef USE_FFTW
			int ncomplex = image_pixel_grid->fft_nj*(image_pixel_grid->fft_ni/2+1);
#else
			int nnvec[2];
			nnvec[0] = image_pixel_grid->fft_ni;
			nnvec[1] = image_pixel_grid->fft_nj;
			int nzvec = 2*image_pixel_grid->fft_ni*image_pixel_grid->fft_nj;
			double **Lmatrix_imgs_zvec = new double*[source_npixels];
			for (i=0; i < source_npixels; i++) {
				Lmatrix_imgs_zvec[i] = new double[nzvec];
			}
#endif


#ifdef USE_OPENMP
			if (show_wtime) {
				wtime0 = omp_get_wtime();
			}
#endif
			double fwtime0, fwtime;
			double rtemp, itemp;
			int npix_conv = image_pixel_grid->fft_ni*image_pixel_grid->fft_nj;
			double *img_zvec, *img_rvec;
			complex<double> *img_cvec;
			int src_index;
			#pragma omp parallel for private(k,i,j,ii,jj,l,img_index,src_index,img_zvec,img_rvec,img_cvec,rtemp,itemp) schedule(static)
			for (src_index=0; src_index < source_npixels; src_index++) {
#ifdef USE_FFTW
				img_rvec = image_pixel_grid->Lmatrix_imgs_rvec[src_index];
				img_cvec = image_pixel_grid->Lmatrix_transform[src_index];
				for (i=0; i < npix_conv; i++) img_rvec[i] = 0;
#else
				img_zvec = Lmatrix_imgs_zvec[src_index];
				for (j=0; j < nzvec; j++) img_zvec[j] = 0;
#endif
				for (img_index=0; img_index < npix; img_index++)
				{
					ii = pixel_map_ii[img_index];
					jj = pixel_map_jj[img_index];
					if (psf_supersampling) {
						i = image_pixel_grid->image_pixel_i_from_subcell_ii[ii];
						j = image_pixel_grid->image_pixel_j_from_subcell_jj[jj];
					} else {
						i = ii;
						j = jj;
					}
					if ((image_pixel_grid->maps_to_source_pixel[i][j]) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j]))) {
						ii -= image_pixel_grid->fft_imin;
						jj -= image_pixel_grid->fft_jmin;
#ifdef USE_FFTW
						l = jj*image_pixel_grid->fft_ni + ii;
						img_rvec[l] = (*Lptr)[img_index][src_index];
#else
						k = 2*(jj*image_pixel_grid->fft_ni + ii);
						img_zvec[k] = (*Lptr)[img_index][src_index];
#endif
					}
				}

#ifdef USE_FFTW
				fftw_execute(image_pixel_grid->fftplans_Lmatrix[src_index]);
				for (i=0; i < ncomplex; i++) {
					img_cvec[i] = img_cvec[i]*image_pixel_grid->psf_transform[i];
					img_cvec[i] /= npix_conv;
				}
				fftw_execute(image_pixel_grid->fftplans_Lmatrix_inverse[src_index]);

#else
				fourier_transform(img_zvec,2,nnvec,1);
				for (i=0,j=0; i < npix_conv; i++, j += 2) {
					rtemp = (img_zvec[j]*image_pixel_grid->psf_zvec[j] - img_zvec[j+1]*image_pixel_grid->psf_zvec[j+1]) / npix_conv;
					itemp = (img_zvec[j]*image_pixel_grid->psf_zvec[j+1] + img_zvec[j+1]*image_pixel_grid->psf_zvec[j]) / npix_conv;
					img_zvec[j] = rtemp;
					img_zvec[j+1] = itemp;
				}
				fourier_transform(img_zvec,2,nnvec,-1);
#endif

				for (img_index=0; img_index < npix; img_index++)
				{
					ii = pixel_map_ii[img_index];
					jj = pixel_map_jj[img_index];
					if (psf_supersampling) {
						i = image_pixel_grid->image_pixel_i_from_subcell_ii[ii];
						j = image_pixel_grid->image_pixel_j_from_subcell_jj[jj];
					} else {
						i = ii;
						j = jj;
					}
					if ((image_pixel_grid->maps_to_source_pixel[i][j]) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j]))) {
						ii -= image_pixel_grid->fft_imin;
						jj -= image_pixel_grid->fft_jmin;
#ifdef USE_FFTW
						l = jj*image_pixel_grid->fft_ni + ii;
						(*Lptr)[img_index][src_index] = img_rvec[l];
#else
						k = 2*(jj*image_pixel_grid->fft_ni + ii);
						(*Lptr)[img_index][src_index] = img_zvec[k];
#endif
					}
				}
			}
#ifdef USE_OPENMP
			if (show_wtime) {
				wtime = omp_get_wtime() - wtime0;
				if (mpi_id==0) {
					cout << "Wall time for calculating PSF convolution of Lmatrix via FFT: " << wtime << endl;
				}
			}
#endif
#ifndef USE_FFTW
			for (i=0; i < source_npixels; i++) delete[] Lmatrix_imgs_zvec[i];
			delete[] Lmatrix_imgs_zvec;
#endif
		} else {
			if (use_input_psf_matrix) {
				if ((!psf_supersampling) and (psf_matrix == NULL)) return;
				if ((psf_supersampling) and (supersampled_psf_matrix == NULL)) return;
			}
			else if (generate_PSF_matrix(image_pixel_grid->pixel_xlength,image_pixel_grid->pixel_ylength,psf_supersampling)==false) return;

			int **pix_index;
			double **psf;
			int max_nx, max_ny;
			if (!psf_supersampling) {
				psf = psf_matrix;
				pix_index = image_pixel_grid->pixel_index;
				max_nx = image_pixel_grid->x_N;
				max_ny = image_pixel_grid->y_N;
			} else {
				psf = supersampled_psf_matrix;
				pix_index = image_pixel_grid->subpixel_index;
				max_nx = image_pixel_grid->x_N*default_imgpixel_nsplit;
				max_ny = image_pixel_grid->y_N*default_imgpixel_nsplit;
			}

			double **Lmatrix_psf = new double*[npix];
			for (i=0; i < npix; i++) {
				Lmatrix_psf[i] = new double[source_n_amps];
				for (j=0; j < source_n_amps; j++) Lmatrix_psf[i][j] = 0;
			}

#ifdef USE_OPENMP
			if (show_wtime) {
				wtime0 = omp_get_wtime();
			}
#endif
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
								if (psf_supersampling) {
									i = image_pixel_grid->image_pixel_i_from_subcell_ii[ii];
									j = image_pixel_grid->image_pixel_j_from_subcell_jj[jj];
								} else {
									i = ii;
									j = jj;
								}
								if ((image_pixel_grid->maps_to_source_pixel[i][j]) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j]))) {
									psfval = psf[psf_k][psf_l];
									img_index2 = pix_index[ii][jj];
									lmatptr = (*Lptr).subarray(img_index2);
									lmatpsfptr = Lmatrix_psf[img_index1];
									for (src_index=0; src_index < source_npixels; src_index++) {
										(*(lmatpsfptr++)) += psfval*(*(lmatptr++));
									}
								}
							}
						}
					}
				}
			}

			// note, the following function sets the pointer in Lmatrix dense to Lmatrix_psf (and deletes the old pointer), so no garbage collection necessary afterwards
			(*Lptr).input(Lmatrix_psf);


#ifdef USE_OPENMP
			if (show_wtime) {
				wtime = omp_get_wtime() - wtime0;
				if (mpi_id==0) cout << "Wall time for calculating dense PSF convolution of Lmatrix: " << wtime << endl;
			}
#endif
		}
		if (psf_supersampling) average_supersampled_dense_Lmatrix(zsrc_i);
	}

	if (include_imgfluxes_in_inversion) {
		int i,j,k;
		double *Lmatptr;
		i=0;
		for (j=0; j < point_imgs.size(); j++) {
			for (k=0; k < point_imgs[j].size(); k++) {
				Lmatptr = Lmatrix_transpose_ptimg_amps.subarray(i);
				image_pixel_grid->generate_point_images(point_imgs[j], Lmatptr, false, -1, k);
				i++;
			}
		}
		int src_amp_i;
		double *Lmatrix_transpose_line;
		i=0;
		for (j=0; j < point_imgs.size(); j++) {
			for (k=0; k < point_imgs[j].size(); k++) {
				src_amp_i = source_npixels + i;
				Lmatrix_transpose_line = Lmatrix_transpose_ptimg_amps[i];
				for (int img_index=0; img_index < image_npixels; img_index++) {
					Lmatrix_dense[img_index][src_amp_i] = Lmatrix_transpose_line[img_index];
				}
				i++;
			}
		}
	} else if (include_srcflux_in_inversion) {
		int j,k;
		double *Lmatptr;
		for (j=0; j < point_imgs.size(); j++) {
			Lmatptr = Lmatrix_transpose_ptimg_amps.subarray(j);
			image_pixel_grid->generate_point_images(point_imgs[j], Lmatptr, false, 1.0);
		}
		int src_amp_i;
		double *Lmatrix_transpose_line;
		for (j=0; j < point_imgs.size(); j++) {
			src_amp_i = source_npixels + j;
			Lmatrix_transpose_line = Lmatrix_transpose_ptimg_amps[j];
			for (int img_index=0; img_index < image_npixels; img_index++) {
				Lmatrix_dense[img_index][src_amp_i] = Lmatrix_transpose_line[img_index];
			}
		}
	}

}

void QLens::PSF_convolution_pixel_vector(const int zsrc_i, const bool foreground, const bool verbal, const bool use_fft)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	if (use_input_psf_matrix) {
		if (!psf_supersampling) {
			if (psf_matrix == NULL) {
				if (verbal) warn("could not find input PSF matrix");
				return;
			}
		} else {
			if (supersampled_psf_matrix == NULL) {
				if (verbal) warn("could not find input supersampled PSF matrix");
				average_supersampled_image_surface_brightness(zsrc_i); 
				return;
			}
		}
	} else {
		if ((psf_width_x==0) and (psf_width_y==0)) {
			if (psf_supersampling) average_supersampled_image_surface_brightness(zsrc_i); // no PSF to convolve
			return;
		}
		else if (generate_PSF_matrix(image_pixel_grid->pixel_xlength,image_pixel_grid->pixel_ylength,psf_supersampling)==false) {
			if (verbal) warn("could not generate_PSF matrix");
			return;
		}
		if (mpi_id==0) cout << "generated PSF matrix" << endl;
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	if ((mpi_id==0) and (verbal)) cout << "Beginning PSF convolution...\n";
	double *surface_brightness_vector;
	if (foreground) surface_brightness_vector = sbprofile_surface_brightness;
	else {
		if (psf_supersampling) surface_brightness_vector = image_surface_brightness_supersampled;
		else surface_brightness_vector = image_surface_brightness;
	}

	int nx_half, ny_half;
	int psf_nx, psf_ny;

	if (!psf_supersampling) {
		psf_nx = psf_npixels_x;
		psf_ny = psf_npixels_y;
	} else {
		psf_nx = supersampled_psf_npixels_x;
		psf_ny = supersampled_psf_npixels_y;
	}
	nx_half = psf_nx/2;
	ny_half = psf_ny/2;

	int npix;
	int *pixel_map_ii, *pixel_map_jj;
	if (foreground) {
		npix = image_npixels_fgmask;
		pixel_map_ii = image_pixel_grid->active_image_pixel_i_fgmask;
		pixel_map_jj = image_pixel_grid->active_image_pixel_j_fgmask;
	} else {
		if (!psf_supersampling) {
			npix = image_npixels;
			pixel_map_ii = image_pixel_grid->active_image_pixel_i;
			pixel_map_jj = image_pixel_grid->active_image_pixel_j;
		} else {
			npix = image_n_subpixels;
			pixel_map_ii = image_pixel_grid->active_image_subpixel_ii;
			pixel_map_jj = image_pixel_grid->active_image_subpixel_jj;
		}
	}

	int i,j,ii,jj,k,img_index;
	if ((!foreground) and (fft_convolution) and (use_fft)) {
		if (!image_pixel_grid->fft_convolution_is_setup) {
			if (!image_pixel_grid->setup_FFT_convolution(psf_supersampling,verbal)) {
				warn("PSF convolution FFT failed");
				return;	
			}
		}

		//int *pixel_map_i, *pixel_map_j;
		//pixel_map_i = image_pixel_grid->active_image_pixel_i;
		//pixel_map_j = image_pixel_grid->active_image_pixel_j;

#ifdef USE_FFTW
		int ncomplex = image_pixel_grid->fft_nj*(image_pixel_grid->fft_ni/2+1);
#else
		int nzvec = 2*image_pixel_grid->fft_ni*image_pixel_grid->fft_nj;
		double *img_zvec = new double[nzvec];
#endif

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

		int l;
#ifndef USE_FFTW
		for (i=0; i < nzvec; i++) img_zvec[i] = 0;
#endif
		for (img_index=0; img_index < npix; img_index++)
		{
			ii = pixel_map_ii[img_index];
			jj = pixel_map_jj[img_index];
			if (psf_supersampling) {
				i = image_pixel_grid->image_pixel_i_from_subcell_ii[ii];
				j = image_pixel_grid->image_pixel_j_from_subcell_jj[jj];
			} else {
				i = ii;
				j = jj;
			}
			if ((image_pixel_grid->maps_to_source_pixel[i][j]) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j]))) {
				ii -= image_pixel_grid->fft_imin;
				jj -= image_pixel_grid->fft_jmin;
#ifdef USE_FFTW
				l = jj*image_pixel_grid->fft_ni + ii;
				image_pixel_grid->single_img_rvec[l] = surface_brightness_vector[img_index];
#else
				k = 2*(jj*image_pixel_grid->fft_ni + ii);
				img_zvec[k] = surface_brightness_vector[img_index];
#endif
			}
		}
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime = omp_get_wtime() - wtime0;
			if (mpi_id==0) {
				cout << "Wall time for setting up PSF convolution via FFT: " << wtime << endl;
			}
			wtime0 = omp_get_wtime();
		}
#endif

#ifdef USE_FFTW
		fftw_execute(image_pixel_grid->fftplan);
		for (i=0; i < ncomplex; i++) {
			image_pixel_grid->img_transform[i] = image_pixel_grid->img_transform[i]*image_pixel_grid->psf_transform[i];
			image_pixel_grid->img_transform[i] /= (image_pixel_grid->fft_ni*image_pixel_grid->fft_nj);
		}
		fftw_execute(image_pixel_grid->fftplan_inverse);
#else
		int nnvec[2];
		nnvec[0] = image_pixel_grid->fft_ni;
		nnvec[1] = image_pixel_grid->fft_nj;
		fourier_transform(img_zvec,2,nnvec,1);

		double rtemp, itemp;
		for (i=0,j=0; i < (image_pixel_grid->fft_ni*image_pixel_grid->fft_nj); i++, j += 2) {
			rtemp = (img_zvec[j]*image_pixel_grid->psf_zvec[j] - img_zvec[j+1]*image_pixel_grid->psf_zvec[j+1]) / (image_pixel_grid->fft_ni*image_pixel_grid->fft_nj);
			itemp = (img_zvec[j]*image_pixel_grid->psf_zvec[j+1] + img_zvec[j+1]*image_pixel_grid->psf_zvec[j]) / (image_pixel_grid->fft_ni*image_pixel_grid->fft_nj);
			img_zvec[j] = rtemp;
			img_zvec[j+1] = itemp;
		}
		fourier_transform(img_zvec,2,nnvec,-1);
#endif

		for (img_index=0; img_index < npix; img_index++)
		{
			ii = pixel_map_ii[img_index];
			jj = pixel_map_jj[img_index];
			if (psf_supersampling) {
				i = image_pixel_grid->image_pixel_i_from_subcell_ii[ii];
				j = image_pixel_grid->image_pixel_j_from_subcell_jj[jj];
			} else {
				i = ii;
				j = jj;
			}
			if ((image_pixel_grid->maps_to_source_pixel[i][j]) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j]))) {
				ii -= image_pixel_grid->fft_imin;
				jj -= image_pixel_grid->fft_jmin;
#ifdef USE_FFTW
				l = jj*image_pixel_grid->fft_ni + ii;
				surface_brightness_vector[img_index] = image_pixel_grid->single_img_rvec[l];
#else
				k = 2*(jj*image_pixel_grid->fft_ni + ii);
				surface_brightness_vector[img_index] = img_zvec[k];
#endif
			}
		}
#ifndef USE_FFTW
		delete[] img_zvec;
#endif
	} else {
		int **pix_index;
		double **psf;
		int max_nx, max_ny;

		if (!psf_supersampling) {
			psf = psf_matrix;
			psf_nx = psf_npixels_x;
			psf_ny = psf_npixels_y;
		} else {
			psf = supersampled_psf_matrix;
			psf_nx = supersampled_psf_npixels_x;
			psf_ny = supersampled_psf_npixels_y;
		}

		if (foreground) {
			pix_index = image_pixel_grid->pixel_index_fgmask;
			max_nx = image_pixel_grid->x_N;
			max_ny = image_pixel_grid->y_N;
		} else {
			if (!psf_supersampling) {
				pix_index = image_pixel_grid->pixel_index;
				max_nx = image_pixel_grid->x_N;
				max_ny = image_pixel_grid->y_N;
			} else {
				pix_index = image_pixel_grid->subpixel_index;
				max_nx = image_pixel_grid->x_N*default_imgpixel_nsplit;
				max_ny = image_pixel_grid->y_N*default_imgpixel_nsplit;
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
					//cout << "ii=" << ii << " versus max_nx=" << max_nx << endl;
					for (psf_l=0; psf_l < psf_ny; psf_l++) {
						jj = l + ny_half - psf_l; // Note, 'l' is the index for the convolved image, so we have l = j - ny_half + psf_l
						if ((jj >= 0) and (jj < max_ny)) {
							//cout << "k-ii=" << (k-ii) << " l-jj=" << (l-jj) << endl;
							if (psf_supersampling) {
								i = image_pixel_grid->image_pixel_i_from_subcell_ii[ii];
								j = image_pixel_grid->image_pixel_j_from_subcell_jj[jj];
							} else {
								i = ii;
								j = jj;
							}
							//cout << "i=" << i << " ii=" << ii << ", j=" << j << " jj=" << jj << endl;
							// THIS IS VERY CLUMSY! RE-IMPLEMENT IN A MORE ELEGANT WAY?
							if (((foreground) and ((image_pixel_grid->pixel_in_mask==NULL) or (!image_pixel_data) or (image_pixel_data->foreground_mask[i][j]))) or  
							((!foreground) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j])))) {
								img_index2 = pix_index[ii][jj];
								new_surface_brightness_vector[img_index] += psf[psf_k][psf_l]*surface_brightness_vector[img_index2];
								//cout << "PSF: " << psf_k << " " << psf_l << " " << psf[psf_k][psf_l] << " " << surface_brightness_vector[img_index2] << " " << new_surface_brightness_vector[img_index] << endl;
							}
						}
					}
				}
			}
		}

#ifdef USE_OPENMP
		if (show_wtime) {
			wtime = omp_get_wtime() - wtime0;
			if (mpi_id==0) {
				if (foreground) cout << "Wall time for calculating PSF convolution of foreground: " << wtime << endl;
				else cout << "Wall time for calculating PSF convolution of image: " << wtime << endl;
			}
		}
#endif
		for (int i=0; i < npix; i++) {
			surface_brightness_vector[i] = new_surface_brightness_vector[i];
			//cout << surface_brightness_vector[i] << endl;
		}
		delete[] new_surface_brightness_vector;
	}
	if (psf_supersampling) average_supersampled_image_surface_brightness(zsrc_i);
}

void QLens::average_supersampled_image_surface_brightness(const int zsrc_i)
{
	// now average the subpixel surface brightnesses to get the new image surface brightness
	int nsubpix = default_imgpixel_nsplit*default_imgpixel_nsplit;
	int i, j, img_index;
	for (i=0; i < image_npixels; i++) image_surface_brightness[i] = 0;
	for (img_index=0; img_index < image_n_subpixels; img_index++) {
		i = image_pixel_grids[zsrc_i]->active_image_pixel_i_ss[img_index];
		j = image_pixel_grids[zsrc_i]->active_image_pixel_j_ss[img_index];
		image_surface_brightness[image_pixel_grids[zsrc_i]->pixel_index[i][j]] += image_surface_brightness_supersampled[img_index];
	}
	for (i=0; i < image_npixels; i++) image_surface_brightness[i] /= nsubpix;
}

void QLens::average_supersampled_dense_Lmatrix(const int zsrc_i)
{
	Lmatrix_dense.input(image_npixels,source_n_amps);
	Lmatrix_dense = 0;
	int i,j,k;

	// now average the subpixel surface brightnesses to get the new image surface brightness
	int nsubpix = default_imgpixel_nsplit*default_imgpixel_nsplit;
	int img_index;
	double *lmatptr, *lmatsup_ptr;
	for (i=0; i < image_npixels; i++) image_surface_brightness[i] = 0;
	for (img_index=0; img_index < image_n_subpixels; img_index++) {
		i = image_pixel_grids[zsrc_i]->active_image_pixel_i_ss[img_index];
		j = image_pixel_grids[zsrc_i]->active_image_pixel_j_ss[img_index];
		lmatptr = Lmatrix_dense[image_pixel_grids[zsrc_i]->pixel_index[i][j]];
		lmatsup_ptr = Lmatrix_supersampled.subarray(img_index);
		for (k=0; k < source_npixels; k++) {
			//Lmatrix_reduced[image_pixel_grids[0]->pixel_index[i][j]][k] += Lmatrix_dense[img_index][k]/nsubpix;
			(*(lmatptr++)) += (*(lmatsup_ptr++))/nsubpix;
		}
	}
}

#define DSWAP(a,b) dtemp=(a);(a)=(b);(b)=dtemp;
void QLens::fourier_transform(double* data, const int ndim, int* nn, const int isign)
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

bool QLens::generate_PSF_matrix(const double xstep, const double ystep, const bool supersampling)
{
	//static const double sigma_fraction = 1.6; // the bigger you make this, the less sparse the matrix will become (more pixel-pixel correlations)
	if (psf_threshold==0) return false; // need a threshold to determine where to truncate the PSF
	double sigma_fraction = sqrt(-2*log(psf_threshold));
	int i,j;
	int nx_half, ny_half, nx, ny;
	double x, y, xmax, ymax;
	if ((psf_width_x==0) or (psf_width_y==0)) return false;
	double normalization = 0;
	double nx_half_dec, ny_half_dec;
	int supersampling_fac=1;
	if (supersampling) supersampling_fac = default_imgpixel_nsplit;
	//xstep = image_pixel_grids[0]->pixel_xlength;
	//ystep = image_pixel_grids[0]->pixel_ylength;
	nx_half_dec = sigma_fraction*psf_width_x/xstep;
	ny_half_dec = sigma_fraction*psf_width_y/ystep;
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
			psf[i][j] = exp(-0.5*(SQR(x/psf_width_x) + SQR(y/psf_width_y)));
			normalization += psf[i][j];
			//cout << "creating PSF: " << i << " " << j << " x=" << x << " y=" << y << " " << psf[i][j] << endl;
		}
	}
	for (i=0; i < nx; i++) {
		for (j=0; j < ny; j++) {
			psf[i][j] /= normalization;
		}
	}
	return true;
}

bool QLens::spline_PSF_matrix(const double xstep, const double ystep)
{
	if (psf_matrix==NULL) return false;
	int i,nx_half,ny_half;
	double x,y;
	nx_half = psf_npixels_x/2;
	ny_half = psf_npixels_y/2;
	double xmax = nx_half*xstep;
	double ymax = ny_half*ystep;
	double *xvals = new double[psf_npixels_x];
	double *yvals = new double[psf_npixels_y];
	for (i=0, x=-xmax; i < psf_npixels_x; i++, x += xstep) xvals[i] = x;
	for (i=0, y=-ymax; i < psf_npixels_y; i++, y += ystep) yvals[i] = y;
	psf_spline.input(xvals,yvals,psf_matrix,psf_npixels_x,psf_npixels_y);
	delete[] xvals;
	delete[] yvals;
	return true;
}

double QLens::interpolate_PSF_matrix(const double x, const double y, const bool supersampled)
{
	double psfint;
	if ((psf_spline.is_splined() and (!supersampled))) {
		psfint = psf_spline.splint(x,y);
	} else {
		double scaled_x, scaled_y;
		int ii,jj;
		double nx_half, ny_half;
		if (!supersampled) {
			nx_half = psf_npixels_x/2;
			ny_half = psf_npixels_y/2;
		} else {
			nx_half = supersampled_psf_npixels_x/2;
			ny_half = supersampled_psf_npixels_y/2;
		}

		// Each image_pixel_grid should have its own separate stored PSF; that way, you can have different PSF's for different bands.
		// It would also be much less awkward than the code below. IMPLEMENT THIS!!
		double pixel_xlength, pixel_ylength;
		if ((!use_old_pixelgrids) and (image_pixel_grids != NULL) and (image_pixel_grids[0] != NULL)) {
			pixel_xlength = image_pixel_grids[0]->pixel_xlength;
			pixel_ylength = image_pixel_grids[0]->pixel_ylength;
		} else if ((use_old_pixelgrids) and (image_pixel_grid0 != NULL)) {
			pixel_xlength = image_pixel_grid0->pixel_xlength;
			pixel_ylength = image_pixel_grid0->pixel_ylength;
		} else {
			pixel_xlength = grid_xlength / n_image_pixels_x;
			pixel_ylength = grid_ylength / n_image_pixels_y;
		}

		if (supersampled) {
			pixel_xlength /= default_imgpixel_nsplit;
			pixel_ylength /= default_imgpixel_nsplit;
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

void QLens::generate_supersampled_PSF_matrix(const bool downsample, const int downsample_fac)
{
	int i,j;
	if (supersampled_psf_matrix != NULL) {
		for (i=0; i < supersampled_psf_npixels_x; i++) delete[] supersampled_psf_matrix[i];
		delete[] supersampled_psf_matrix;
	}


	// Each image_pixel_grid should have its own separate stored PSF; that way, you can have different PSF's for different bands.
	// It would also be much less awkward than the code below. IMPLEMENT THIS!!
	double pixel_xlength, pixel_ylength;
	if ((!use_old_pixelgrids) and (image_pixel_grids != NULL) and (image_pixel_grids[0] != NULL)) {
		pixel_xlength = image_pixel_grids[0]->pixel_xlength;
		pixel_ylength = image_pixel_grids[0]->pixel_ylength;
	} else if ((use_old_pixelgrids) and (image_pixel_grid0 != NULL)) {
		pixel_xlength = image_pixel_grid0->pixel_xlength;
		pixel_ylength = image_pixel_grid0->pixel_ylength;
	} else {
		pixel_xlength = grid_xlength / n_image_pixels_x;
		pixel_ylength = grid_ylength / n_image_pixels_y;
	}

	int nx, ny;
	double x, xmax, xstep, y, ymax, ystep;
	xstep = pixel_xlength / default_imgpixel_nsplit;
	ystep = pixel_ylength / default_imgpixel_nsplit;

	if ((psf_spline.is_splined()) and (downsample)) {
		int psf0_nx, psf0_ny;
		psf0_nx = (int) (((psf_spline.xmax()-psf_spline.xmin())/pixel_xlength) + 1.000001);
		psf0_ny = (int) (((psf_spline.ymax()-psf_spline.ymin())/pixel_ylength) + 1.000001);
		nx = psf0_nx * default_imgpixel_nsplit;
		ny = psf0_ny * default_imgpixel_nsplit;
		cout << "NX=" << nx << " NY=" << ny << endl;
		double check_nx = psf_npixels_x * default_imgpixel_nsplit;
		double check_ny = psf_npixels_y * default_imgpixel_nsplit;
		xmax = ((psf0_nx * pixel_xlength)-xstep) / 2;
		ymax = ((psf0_ny * pixel_ylength)-ystep) / 2;
		cout << "CHECK: " << check_nx << " " << check_ny << endl;
		cout << "ARGH: " << ((psf_spline.ymax()-psf_spline.ymin())/pixel_ylength) << endl;
	} else {
		nx = psf_npixels_x * default_imgpixel_nsplit;
		ny = psf_npixels_y * default_imgpixel_nsplit;
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
		cout << "DOWNSAMPLE DX=" << dx << " DY=" << dy << endl;
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

}

bool QLens::create_regularization_matrix(const int zsrc_i, const bool allow_lum_weighting, const bool use_sbweights, const bool verbal)
{
	RegularizationMethod reg_method = regularization_method;
	if ((use_lum_weighted_regularization) and (!allow_lum_weighting)) reg_method = Curvature;
	if (zsrc_i > 0) reg_method = SmoothGradient; // HACK, since we have no way of assigning different correlation length, matern index, etc to different zsrc_i fits
	if (Rmatrix != NULL) { delete[] Rmatrix; Rmatrix = NULL; }
	if (Rmatrix_index != NULL) { delete[] Rmatrix_index; Rmatrix_index = NULL; }
	if (allow_lum_weighting) calculate_lumreg_srcpixel_weights(zsrc_i,use_sbweights);

	dense_Rmatrix = false; // assume sparse unless a dense regularization is chosen
	bool covariance_kernel_regularization = false;
	use_covariance_matrix = false; // if true, will use covariance matrix directly instead of Rmatrix
	bool successful_Rmatrix = true;
	if ((!find_covmatrix_inverse) and (n_sourcepts_fit > 0)) die("modeling point images is not currently compatible with 'find_cov_inverse off' setting"); // see notes in generate_Gmatrix function (this is where the problem is, I believe)...FIX LATER!!
	switch (reg_method) {
		case Norm:
			generate_Rmatrix_norm(); break;
		case Gradient:
			generate_Rmatrix_from_gmatrices(zsrc_i); break;
		case SmoothGradient:
			generate_Rmatrix_from_gmatrices(zsrc_i,true); break;
		case Curvature:
			generate_Rmatrix_from_hmatrices(zsrc_i); break;
		case SmoothCurvature:
			generate_Rmatrix_from_hmatrices(zsrc_i,true); break;
		case Matern_Kernel:
			dense_Rmatrix = true;
			covariance_kernel_regularization = true;
			if (!find_covmatrix_inverse) use_covariance_matrix = true;
			successful_Rmatrix = generate_Rmatrix_from_covariance_kernel(zsrc_i,0,allow_lum_weighting,verbal);
			break;
		case Exponential_Kernel:
			dense_Rmatrix = true;
			covariance_kernel_regularization = true;
			if (!find_covmatrix_inverse) use_covariance_matrix = true;
			successful_Rmatrix = generate_Rmatrix_from_covariance_kernel(zsrc_i,1,allow_lum_weighting,verbal);
			break;
		case Squared_Exponential_Kernel:
			dense_Rmatrix = true;
			covariance_kernel_regularization = true;
			if (!find_covmatrix_inverse) use_covariance_matrix = true;
			successful_Rmatrix = generate_Rmatrix_from_covariance_kernel(zsrc_i,2,allow_lum_weighting,verbal);
			break;
		default:
			die("Regularization method not recognized");
	}
	if (!successful_Rmatrix) return false;
	if ((dense_Rmatrix) and (inversion_method!=DENSE) and (inversion_method!=DENSE_FMATRIX)) die("inversion method must be set to 'dense' or 'fdense' if a dense regularization matrix is used");
	if ((!dense_Rmatrix) and ((inversion_method==DENSE) or (inversion_method==DENSE_FMATRIX))) {
		// If doing a sparse inversion, the determinant of R-matrix will be calculated when doing the inversion; otherwise, must be done here
		// unless R-matrix is dense (as in the covariance kernel reg.), in which case determinant is found during its construction

#ifdef USE_UMFPACK
		Rmatrix_determinant_UMFPACK();
#else
#ifdef USE_MUMPS
		Rmatrix_determinant_MUMPS();
#else
#ifdef USE_MKL
		Rmatrix_determinant_MKL();
#else
		warn("Converting Rmatrix to dense, since MUMPS, UMFPACK, or MKL is required to calculate sparse R-matrix determinants");
		Rmatrix_determinant_dense();
#endif
#endif
#endif
	}

	//cout << "Printing Rmatrix..." << endl;
	//int indx;	
	//for (i=0; i < source_npixels; i++) {
		//indx = Rmatrix_index[i];
		//int nn = Rmatrix_index[i+1]-Rmatrix_index[i];
		//cout << "Row " << i << ": " << nn << " entries (starts at index " << indx << ")" << endl;
		//cout << "diag: " << Rmatrix[i] << endl;
		//for (j=0; j < nn; j++) {
			//cout << i << " " << Rmatrix_index[indx+j] << " " << Rmatrix[indx+j] << endl;
		//}
	//}
	return true;
}

void QLens::create_regularization_matrix_shapelet(const int zsrc_i)
{
	if (source_npixels==0) return;
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	dense_Rmatrix = false;
	use_covariance_matrix = false; // if true, will use covariance matrix directly instead of Rmatrix
	switch (regularization_method) {
		case Norm:
			generate_Rmatrix_norm(); break;
		case Gradient:
			generate_Rmatrix_shapelet_gradient(zsrc_i); break;
		case Curvature:
			generate_Rmatrix_shapelet_curvature(zsrc_i); break;
		default:
			die("Regularization method not recognized for dense matrices");
	}
#ifdef USE_UMFPACK
	Rmatrix_determinant_UMFPACK();
#else
#ifdef USE_MUMPS
	Rmatrix_determinant_MUMPS();
#else
#ifdef USE_MKL
	Rmatrix_determinant_MKL();
#else
	warn("Converting Rmatrix to dense, since MUMPS, UMFPACK, or MKL is required to calculate sparse R-matrix determinants");
	Rmatrix_determinant_dense();
#endif
#endif
#endif
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating Rmatrix: " << wtime << endl;
		wtime0 = omp_get_wtime();
	}
#endif

}

void QLens::generate_Rmatrix_norm()
{
	Rmatrix_nn = source_npixels+1;
	Rmatrix = new double[Rmatrix_nn];
	Rmatrix_index = new int[Rmatrix_nn];

	for (int i=0; i < source_npixels; i++) {
		Rmatrix[i] = 1;
		Rmatrix_index[i] = source_npixels+1;
	}
	Rmatrix_index[source_npixels] = source_npixels+1;

	Rmatrix_log_determinant = 0;
}

void QLens::generate_Rmatrix_shapelet_gradient(const int zsrc_i)
{
	bool at_least_one_shapelet = false;
	Rmatrix_nn = 3*source_npixels+1; // actually it will be slightly less than this due to truncation at shapelets with i=n_shapelets-1 or j=n_shapelets-1

	Rmatrix = new double[Rmatrix_nn];
	Rmatrix_index = new int[Rmatrix_nn];

	for (int i=0; i < n_sb; i++) {
		if ((sb_list[i]->sbtype==SHAPELET) and ((zsrc_i<0) or (sbprofile_redshift_idx[i]==zsrc_i))) {
			sb_list[i]->calculate_gradient_Rmatrix_elements(Rmatrix, Rmatrix_index);
			at_least_one_shapelet = true;
			break;
		}
	}
	if (!at_least_one_shapelet) die("No shapelet profile has been created; cannot calculate regularization matrix");
	Rmatrix_nn = Rmatrix_index[source_npixels];
	//for (int i=0; i <= source_npixels; i++) cout << Rmatrix[i] << " " << Rmatrix_index[i] << endl;
	//cout << "Rmatrix_nn=" << Rmatrix_nn << " source_npixels=" << source_npixels << endl;
}

void QLens::generate_Rmatrix_shapelet_curvature(const int zsrc_i)
{
	Rmatrix_nn = source_npixels+1;

	Rmatrix = new double[Rmatrix_nn];
	Rmatrix_index = new int[Rmatrix_nn];

	bool at_least_one_shapelet = false;
	for (int i=0; i < n_sb; i++) {
		if ((sb_list[i]->sbtype==SHAPELET) and ((zsrc_i<0) or (sbprofile_redshift_idx[i]==zsrc_i))) {
			sb_list[i]->calculate_curvature_Rmatrix_elements(Rmatrix, Rmatrix_index);
			at_least_one_shapelet = true;
		}
	}
	if (!at_least_one_shapelet) die("No shapelet profile has been created; cannot calculate regularization matrix");
	Rmatrix_nn = Rmatrix_index[source_npixels];
}

void QLens::create_lensing_matrices_from_Lmatrix(const int zsrc_i, const bool dense_Fmatrix, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	double cov_inverse; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (!use_noise_map) {
		if (background_pixel_noise==0) cov_inverse = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
		else cov_inverse = 1.0/SQR(background_pixel_noise);
	}

	int i,j,k,l,m,t;

	vector<int> *Fmatrix_index_rows = new vector<int>[source_n_amps];
	vector<double> *Fmatrix_rows = new vector<double>[source_n_amps];
	double *Fmatrix_diags = new double[source_n_amps];
	int *Fmatrix_row_nn = new int[source_n_amps];
	Fmatrix_nn = 0;
	int Fmatrix_nn_part = 0;
	for (j=0; j < source_n_amps; j++) {
		Fmatrix_diags[j] = 0;
		Fmatrix_row_nn[j] = 0;
	}
	int ntot = source_n_amps*source_n_amps;
	int ntot_packed = source_n_amps*(source_n_amps+1)/2;

	bool new_entry;
	int src_index1, src_index2, col_index, col_i;
	double tmp, element;
	Dvector = new double[source_n_amps];
	for (i=0; i < source_n_amps; i++) Dvector[i] = 0;

	int pix_i, pix_j, img_index_fgmask;
	double sbcov;
	//double *Lmatrix_eff;
	//Lmatrix_eff = new double[Lmatrix_n_elements];
	for (i=0; i < image_npixels; i++) {
		if (use_noise_map) cov_inverse = imgpixel_covinv_vector[i];
		pix_i = image_pixel_grid->active_image_pixel_i[i];
		pix_j = image_pixel_grid->active_image_pixel_j[i];
		img_index_fgmask = image_pixel_grid->pixel_index_fgmask[pix_i][pix_j];
		sbcov = image_surface_brightness[i] - sbprofile_surface_brightness[img_index_fgmask];
		if (((!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) and (n_sourcepts_fit > 0)) sbcov -= point_image_surface_brightness[i];
		sbcov *= cov_inverse;
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			//Dvector[Lmatrix_index[j]] += Lmatrix[j]*(image_surface_brightness[i] - sbprofile_surface_brightness[i])/cov_inverse;
			//Dvector[Lmatrix_index[j]] += Lmatrix[j]*(image_surface_brightness[i] - image_pixel_grid->foreground_surface_brightness[pix_i][pix_j])/cov_inverse;
			Dvector[Lmatrix_index[j]] += Lmatrix[j]*sbcov;
			//Lmatrix_eff[j] = Lmatrix[j]*sqrt(cov_inverse);
		}
	}
	for (i=0; i < image_npixels; i++) {
		if (use_noise_map) cov_inverse = imgpixel_covinv_vector[i];
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			Lmatrix[j] *= sqrt(cov_inverse);
		}
	}

	int mpi_chunk, mpi_start, mpi_end;
	mpi_chunk = source_n_amps / group_np;
	mpi_start = group_id*mpi_chunk;
	if (group_id == group_np-1) mpi_chunk += (source_n_amps % group_np); // assign the remainder elements to the last mpi process
	mpi_end = mpi_start + mpi_chunk;

#ifdef USE_MKL
	int *srcpixel_location_Fmatrix, *srcpixel_end_Fmatrix, *Fmatrix_csr_index;
	double *Fmatrix_csr;
	int nsrc1, nsrc2;
	sparse_index_base_t indxing;
	sparse_matrix_t Lsparse;
	sparse_matrix_t Fsparse;
	int *image_pixel_end_Lmatrix = new int[image_npixels];
	for (i=0; i < image_npixels; i++) image_pixel_end_Lmatrix[i] = image_pixel_location_Lmatrix[i+1];
	//cout << "Creating CSR matrix..." << endl;
	mkl_sparse_d_create_csr(&Lsparse, SPARSE_INDEX_BASE_ZERO, image_npixels, source_n_amps, image_pixel_location_Lmatrix, image_pixel_end_Lmatrix, Lmatrix_index, Lmatrix);
	mkl_sparse_order(Lsparse);
	sparse_status_t status;
	if (!dense_Fmatrix) {
		status = mkl_sparse_syrk(SPARSE_OPERATION_TRANSPOSE, Lsparse, &Fsparse);
		mkl_sparse_d_export_csr(Fsparse, &indxing, &nsrc1, &nsrc2, &srcpixel_location_Fmatrix, &srcpixel_end_Fmatrix, &Fmatrix_csr_index, &Fmatrix_csr);

		if ((verbal) and (mpi_id==0)) cout << "Fmatrix_sparse has " << srcpixel_end_Fmatrix[source_n_amps-1] << " elements" << endl;
		bool duplicate_column;
		int dup_k;
		for (i=0; i < source_n_amps; i++) {
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
				//cout << Lmatrix_index[j] << " " << Lmatrix[j] << endl;
			//}
		//}

		//cout << endl << "FMATRIX:" << endl;

		//for (i=0; i < source_n_amps; i++) {
			//cout << "Row " << i << ":" << endl;
			//for (j=srcpixel_location_Fmatrix[i]; j < srcpixel_end_Fmatrix[i]; j++) {
				//cout << Fmatrix_csr_index[j] << " " << Fmatrix_csr[j] << endl;
			//}
		//}
	} else {
		Fmatrix_packed.input(ntot_packed);
		Fmatrix_stacked.input(ntot);
		for (i=0; i < ntot; i++) Fmatrix_stacked[i] = 0;
		mkl_sparse_d_syrkd(SPARSE_OPERATION_TRANSPOSE,Lsparse,1.0,0.0,Fmatrix_stacked.array(),SPARSE_LAYOUT_ROW_MAJOR,source_n_amps);
		int nf=0;
		for (i=0; i < ntot; i++) {
			if (Fmatrix_stacked[i] != 0) {
				//Fmatrix_stacked[i] *= cov_inverse; // not necessary since the noise (inverse) covariance was put into Lmatrix_eff
				nf++;
			}
		}
		LAPACKE_dtrttp(LAPACK_ROW_MAJOR,'U',source_n_amps,Fmatrix_stacked.array(),source_n_amps,Fmatrix_packed.array());
		if ((verbal) and (mpi_id==0)) cout << "Fmatrix_dense has " << nf << " nonzero elements" << endl;
		if (use_covariance_matrix) generate_Gmatrix();
	}
#else
	// the non-MKL version (considerably slower)
	vector<jl_pair> **jlvals = new vector<jl_pair>*[nthreads];
	for (i=0; i < nthreads; i++) {
		jlvals[i] = new vector<jl_pair>[source_n_amps];
	}

	jl_pair jl;
	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
	// idea: just store j and l, so that all the calculating can be done in the loop below (which can be made parallel much more easily)
		#pragma omp for private(i,j,l,jl,src_index1,src_index2,tmp) schedule(dynamic)
		for (i=0; i < image_npixels; i++) {
			for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
				for (l=j; l < image_pixel_location_Lmatrix[i+1]; l++) {
					src_index1 = Lmatrix_index[j];
					src_index2 = Lmatrix_index[l];
					if (src_index1 > src_index2) {
						jl.l=j; jl.j=l;
						jlvals[thread][src_index2].push_back(jl);
					} else {
						jl.j=j; jl.l=l;
						jlvals[thread][src_index1].push_back(jl);
					}
				}
			}
		}

#ifdef USE_OPENMP
		#pragma omp barrier
		#pragma omp master
		{
			if (show_wtime) {
				wtime = omp_get_wtime() - wtime0;
				if (mpi_id==0) cout << "Wall time for calculating Fmatrix (storing jvals,lvals): " << wtime << endl;
				wtime0 = omp_get_wtime();
			}
		}
#endif

		#pragma omp for private(i,j,k,l,m,t,src_index1,src_index2,new_entry,col_index,col_i,element) schedule(static) reduction(+:Fmatrix_nn_part)
		for (src_index1=mpi_start; src_index1 < mpi_end; src_index1++) {
			col_i=0;
			for (t=0; t < nthreads; t++) {
				for (k=0; k < jlvals[t][src_index1].size(); k++) {
					j = jlvals[t][src_index1][k].j;
					l = jlvals[t][src_index1][k].l;
					src_index2 = Lmatrix_index[l];
					new_entry = true;
					element = Lmatrix[j]*Lmatrix[l];
					if (src_index1==src_index2) Fmatrix_diags[src_index1] += element;
					else {
						m=0;
						while ((m < Fmatrix_row_nn[src_index1]) and (new_entry==true))
						{
							if (Fmatrix_index_rows[src_index1][m]==src_index2) {
								new_entry = false;
								col_index = m;
							}
							m++;
						}
						if (new_entry) {
							Fmatrix_rows[src_index1].push_back(element);
							Fmatrix_index_rows[src_index1].push_back(src_index2);
							Fmatrix_row_nn[src_index1]++;
							col_i++;
						}
						else Fmatrix_rows[src_index1][col_index] += element;
					}
				}
			}
			Fmatrix_nn_part += col_i;
		}
	}
#endif

	if (!dense_Fmatrix) {
		if ((regularization_method != None) and (source_npixels > 0)) {
			for (src_index1=mpi_start; src_index1 < mpi_end; src_index1++) {
				if (src_index1 < source_npixels) { // additional source amplitudes are not regularized
					if ((!optimize_regparam) and (zsrc_i==0)) Fmatrix_diags[src_index1] += regularization_parameter*Rmatrix[src_index1];
					col_i=0;
					for (j=Rmatrix_index[src_index1]; j < Rmatrix_index[src_index1+1]; j++) {
						new_entry = true;
						k=0;
						while ((k < Fmatrix_row_nn[src_index1]) and (new_entry==true)) {
							if (Rmatrix_index[j]==Fmatrix_index_rows[src_index1][k]) {
								new_entry = false;
								col_index = k;
							}
							k++;
						}
						if (new_entry) {
							if ((!optimize_regparam) and (zsrc_i==0)) {
							//cout << "Fmat row " << src_index1 << ", col " << (Rmatrix_index[j]) << ": was 0, now adding " << (regularization_parameter*Rmatrix[j]) << endl;
								Fmatrix_rows[src_index1].push_back(regularization_parameter*Rmatrix[j]);
							} else {
								Fmatrix_rows[src_index1].push_back(0);
								// This way, when we're optimizing the regularization parameter, the needed entries are already there to add to
							}
							Fmatrix_index_rows[src_index1].push_back(Rmatrix_index[j]);
							Fmatrix_row_nn[src_index1]++;
							col_i++;
						} else {
							if ((!optimize_regparam) and (zsrc_i==0)) {
							//cout << "Fmat row " << src_index1 << ", col " << (Rmatrix_index[j]) << ": was " << Fmatrix_rows[src_index1][col_index] << ", now adding " << (regularization_parameter*Rmatrix[j]) << endl;
								Fmatrix_rows[src_index1][col_index] += regularization_parameter*Rmatrix[j];
							}

						}
					}
					Fmatrix_nn_part += col_i;
				}
			}
		}

#ifdef USE_MPI
		MPI_Allreduce(&Fmatrix_nn_part, &Fmatrix_nn, 1, MPI_INT, MPI_SUM, sub_comm);
#else
		Fmatrix_nn = Fmatrix_nn_part;
#endif
		Fmatrix_nn += source_n_amps+1;

		Fmatrix = new double[Fmatrix_nn];
		Fmatrix_index = new int[Fmatrix_nn];

#ifdef USE_MPI
		int id, chunk, start, end, length;
		for (id=0; id < group_np; id++) {
			chunk = source_n_amps / group_np;
			start = id*chunk;
			if (id == group_np-1) chunk += (source_n_amps % group_np); // assign the remainder elements to the last mpi process
			MPI_Bcast(Fmatrix_row_nn + start,chunk,MPI_INT,id,sub_comm);
			MPI_Bcast(Fmatrix_diags + start,chunk,MPI_DOUBLE,id,sub_comm);
		}
#endif

		Fmatrix_index[0] = source_n_amps+1;
		for (i=0; i < source_n_amps; i++) {
			Fmatrix_index[i+1] = Fmatrix_index[i] + Fmatrix_row_nn[i];
		}
		if (Fmatrix_index[source_n_amps] != Fmatrix_nn) die("Fmatrix # of elements don't match up (%i vs %i), process %i",Fmatrix_index[source_n_amps],Fmatrix_nn,mpi_id);

		for (i=0; i < source_n_amps; i++)
			Fmatrix[i] = Fmatrix_diags[i];

		int indx;
		for (i=mpi_start; i < mpi_end; i++) {
			indx = Fmatrix_index[i];
			for (j=0; j < Fmatrix_row_nn[i]; j++) {
				Fmatrix[indx+j] = Fmatrix_rows[i][j];
				Fmatrix_index[indx+j] = Fmatrix_index_rows[i][j];
			}
		}

#ifdef USE_MPI
		for (id=0; id < group_np; id++) {
			chunk = source_n_amps / group_np;
			start = id*chunk;
			if (id == group_np-1) chunk += (source_n_amps % group_np); // assign the remainder elements to the last mpi process
			end = start + chunk;
			length = Fmatrix_index[end] - Fmatrix_index[start];
			MPI_Bcast(Fmatrix + Fmatrix_index[start],length,MPI_DOUBLE,id,sub_comm);
			MPI_Bcast(Fmatrix_index + Fmatrix_index[start],length,MPI_INT,id,sub_comm);
		}
#endif

#ifdef USE_OPENMP
		if (show_wtime) {
			wtime = omp_get_wtime() - wtime0;
			if (mpi_id==0) cout << "Wall time for Fmatrix MPI communication + construction: " << wtime << endl;
		}
#endif
		if ((mpi_id==0) and (verbal)) cout << "Fmatrix now has " << Fmatrix_nn << " elements\n";

		if ((mpi_id==0) and (verbal)) {
			int Fmatrix_ntot = source_n_amps*(source_n_amps+1)/2;
			double sparseness = ((double) Fmatrix_nn)/Fmatrix_ntot;
			cout << "src_npixels = " << source_n_amps << endl;
			cout << "Fmatrix ntot = " << Fmatrix_ntot << endl;
			cout << "Fmatrix sparseness = " << sparseness << endl;
		}
	} else {
		if ((regularization_method != None) and (source_npixels > 0) and ((!optimize_regparam) and (zsrc_i==0))) add_regularization_term_to_dense_Fmatrix();
	}
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime = omp_get_wtime() - wtime0;
			if (mpi_id==0) cout << "Wall time for calculating Fmatrix elements: " << wtime << endl;
			wtime0 = omp_get_wtime();
		}
#endif
	for (i=0; i < image_npixels; i++) {
		if (use_noise_map) cov_inverse = imgpixel_covinv_vector[i];
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			Lmatrix[j] /= sqrt(cov_inverse);
		}
	}

	//cout << "FMATRIX (SPARSE):" << endl;
	//for (i=0; i < source_n_amps; i++) {
		//cout << i << "," << i << ": " << Fmatrix[i] << endl;
		//for (j=Fmatrix_index[i]; j < Fmatrix_index[i+1]; j++) {
			//cout << i << "," << Fmatrix_index[j] << ": " << Fmatrix[j] << endl;
		//}
		//cout << endl;
	//}

/*
	bool found;
	cout << "LMATRIX:" << endl;
	for (i=0; i < image_npixels; i++) {
		for (j=0; j < source_n_amps; j++) {
			found = false;
			for (k=image_pixel_location_Lmatrix[i]; k < image_pixel_location_Lmatrix[i+1]; k++) {
				if (Lmatrix_index[k]==j) {
					found = true;
					cout << Lmatrix[k] << " ";
				}
			}
			if (!found) cout << "0 ";
		}
		cout << endl;
	}
	*/	

#ifdef USE_MKL
	mkl_sparse_destroy(Lsparse);
	if (!dense_Fmatrix) mkl_sparse_destroy(Fsparse);
	delete[] image_pixel_end_Lmatrix;
	//delete[] Lmatrix_eff;
#else
	for (i=0; i < nthreads; i++) {
		delete[] jlvals[i];
	}
	delete[] jlvals;
#endif
	delete[] Fmatrix_index_rows;
	delete[] Fmatrix_rows;
	delete[] Fmatrix_diags;
	delete[] Fmatrix_row_nn;
#ifdef USE_MPI
	MPI_Comm_free(&sub_comm);
#endif
}

/*
void QLens::create_lensing_matrices_from_Lmatrix_dense(const bool verbal)
{
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	double cov_inverse; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (!use_noise_map) {
		if (background_pixel_noise==0) cov_inverse = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
		else cov_inverse = 1.0/SQR(background_pixel_noise);
	}

	int i,j,l,n;

	bool new_entry;
	Dvector = new double[source_n_amps];
	for (i=0; i < source_n_amps; i++) Dvector[i] = 0;
	int ntot_packed = source_n_amps*(source_n_amps+1)/2;
	Fmatrix_packed.input(ntot_packed);
#ifdef USE_MKL
   double *Ltrans_stacked = new double[source_n_amps*image_npixels];
	Fmatrix_stacked.input(source_n_amps*source_n_amps);
#else
	double *i_n = new double[ntot_packed];
	double *j_n = new double[ntot_packed];
	double **Ltrans = new double*[source_n_amps];
	n=0;
	for (i=0; i < source_n_amps; i++) {
		Ltrans[i] = new double[image_npixels];
		for (j=i; j < source_n_amps; j++) {
			i_n[n] = i;
			j_n[n] = j;
			n++;
		}
	}
#endif

	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		int row;
		int pix_i, pix_j;
		int img_index_fgmask;
		double sb_adj;
		#pragma omp for private(i,j,pix_i,pix_j,img_index_fgmask,row,sb_adj) schedule(static)
		for (i=0; i < source_n_amps; i++) {
			row = i*image_npixels;
			for (j=0; j < image_npixels; j++) {
				//if (use_noise_map) cov_inverse = imgpixel_covinv_vector[j];
				pix_i = image_pixel_grid->active_image_pixel_i[j];
				pix_j = image_pixel_grid->active_image_pixel_j[j];
				img_index_fgmask = image_pixel_grid->pixel_index_fgmask[pix_i][pix_j];
				//Dvector[i] += Lmatrix_dense[j][i]*(image_surface_brightness[j] - sbprofile_surface_brightness[j])*cov_inverse;
				//Dvector[i] += Lmatrix_dense[j][i]*(image_surface_brightness[j] - image_pixel_grid->foreground_surface_brightness[pix_i][pix_j])*cov_inverse;
				if ((zero_sb_extended_mask_prior) and (include_extended_mask_in_inversion) and (image_pixel_data->extended_mask[pix_i][pix_j]) and (!image_pixel_data->in_mask[pix_i][pix_j])) ; 
				else {
					sb_adj = image_surface_brightness[j] - sbprofile_surface_brightness[img_index_fgmask];
					if (((vary_srcflux) and (!include_imgfluxes_in_inversion)) and (n_sourcepts_fit > 0)) sb_adj -= point_image_surface_brightness[j];
					Dvector[i] += Lmatrix_dense[j][i]*sb_adj*cov_inverse;
					if (sbprofile_surface_brightness[img_index_fgmask]*0.0 != 0.0) die("FUCK");
				}
#ifdef USE_MKL
				Ltrans_stacked[row+j] = Lmatrix_dense[j][i]*sqrt(cov_inverse); // hack to get the cov_inverse in there
#else
				Ltrans[i][j] = Lmatrix_dense[j][i];
#endif
			}
		}

#ifdef USE_OPENMP
		#pragma omp master
		{
			if (show_wtime) {
				wtime = omp_get_wtime() - wtime0;
				if (mpi_id==0) cout << "Wall time for initializing Fmatrix and Dvector: " << wtime << endl;
				wtime0 = omp_get_wtime();
			}
		}
#endif

#ifndef USE_MKL
		// The following is not as fast as the Blas function dsyrk (below), but it still gets the job done
		double *fpmatptr;
		double *lmatptr1, *lmatptr2;
		double *covinvptr;
		#pragma omp for private(n,i,j,l,lmatptr1,lmatptr2,fpmatptr) schedule(static)
		for (n=0; n < ntot_packed; n++) {
			i = i_n[n];
			j = j_n[n];
			fpmatptr = Fmatrix_packed.array()+n;
			lmatptr1 = Ltrans[i];
			lmatptr2 = Ltrans[j];
			(*fpmatptr) = 0;
			if (use_noise_map) {
				covinvptr = imgpixel_covinv_vector;
				for (l=0; l < image_npixels; l++) {
					(*fpmatptr) += (*(lmatptr1++))*(*(lmatptr2++))*(*(imgpixel_covinv_vector++));
				}
			} else {
				for (l=0; l < image_npixels; l++) {
					(*fpmatptr) += (*(lmatptr1++))*(*(lmatptr2++));
				}
				(*fpmatptr) *= cov_inverse;
			}
		}
#endif
	}

#ifdef USE_MKL
   cblas_dsyrk(CblasRowMajor,CblasUpper,CblasNoTrans,source_n_amps,image_npixels,1,Ltrans_stacked,image_npixels,0,Fmatrix_stacked.array(),source_n_amps); // Note: this only fills the upper triangular half of the stacked matrix
	LAPACKE_dtrttp(LAPACK_ROW_MAJOR,'U',source_n_amps,Fmatrix_stacked.array(),source_n_amps,Fmatrix_packed.array());
#endif
	if (use_covariance_matrix) generate_Gmatrix();

	if ((regularization_method != None) and (source_npixels > 0) and (!optimize_regparam)) add_regularization_term_to_dense_Fmatrix();
	//double Ftot = 0;
	//for (i=0; i < ntot_packed; i++) Ftot += Fmatrix_packed[i];
	//double ltot = 0;
	//for (i=0; i < source_n_amps; i++) {
		//for (j=0; j < image_npixels; j++) {
			//ltot += Lmatrix_dense[i][j];
		//}
	//}
	//cout << "Ltot, Ftot: " << ltot << " " << Ftot << endl;

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating Fmatrix dense elements: " << wtime << endl;
		wtime0 = omp_get_wtime();
	}
#endif
#ifdef USE_MKL
	delete[] Ltrans_stacked;
#else
	for (i=0; i < source_n_amps; i++) delete[] Ltrans[i];
	delete[] Ltrans;
	delete[] i_n;
	delete[] j_n;
#endif
}
*/

void QLens::create_lensing_matrices_from_Lmatrix_dense(const int zsrc_i, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	double cov_inverse; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (!use_noise_map) {
		if (background_pixel_noise==0) cov_inverse = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
		else cov_inverse = 1.0/SQR(background_pixel_noise);
	}

	int i,j,l,n;

	bool new_entry;
	Dvector = new double[source_n_amps];
	for (i=0; i < source_n_amps; i++) Dvector[i] = 0;
	int ntot_packed = source_n_amps*(source_n_amps+1)/2;
	Fmatrix_packed.input(ntot_packed);
#ifdef USE_MKL
   double *Ltrans_stacked = new double[source_n_amps*image_npixels];
	Fmatrix_stacked.input(source_n_amps*source_n_amps);
#else
	double *i_n = new double[ntot_packed];
	double *j_n = new double[ntot_packed];
	double **Ltrans = new double*[source_n_amps];
	n=0;
	for (i=0; i < source_n_amps; i++) {
		Ltrans[i] = new double[image_npixels];
		for (j=i; j < source_n_amps; j++) {
			i_n[n] = i;
			j_n[n] = j;
			n++;
		}
	}
#endif

	#pragma omp parallel
	{
		int thread;
#ifdef USE_OPENMP
		thread = omp_get_thread_num();
#else
		thread = 0;
#endif
		int row;
		int pix_i, pix_j;
		int img_index_fgmask;
		double sb_adj;
		double covinv = cov_inverse;
		//#pragma omp master
			// Parallelizing this part was causing problems previously, and I don't know why!!!
		#pragma omp for private(i,j,pix_i,pix_j,img_index_fgmask,row,sb_adj,covinv) schedule(static)
		for (i=0; i < source_n_amps; i++) {
			row = i*image_npixels;
			for (j=0; j < image_npixels; j++) {
				if (use_noise_map) covinv = imgpixel_covinv_vector[j];
				pix_i = image_pixel_grid->active_image_pixel_i[j];
				pix_j = image_pixel_grid->active_image_pixel_j[j];
				img_index_fgmask = image_pixel_grid->pixel_index_fgmask[pix_i][pix_j];
				//Dvector[i] += Lmatrix_dense[j][i]*(image_surface_brightness[j] - sbprofile_surface_brightness[j])/cov_inverse;
				//Dvector[i] += Lmatrix_dense[j][i]*(image_surface_brightness[j] - image_pixel_grid->foreground_surface_brightness[pix_i][pix_j])/cov_inverse;
				if ((zero_sb_extended_mask_prior) and (include_extended_mask_in_inversion) and (image_pixel_data->extended_mask[assigned_mask[zsrc_i]][pix_i][pix_j]) and (!image_pixel_data->in_mask[assigned_mask[zsrc_i]][pix_i][pix_j])) ; 
				else {
					sb_adj = image_surface_brightness[j] - sbprofile_surface_brightness[img_index_fgmask];
					if (((!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) and (n_sourcepts_fit > 0)) sb_adj -= point_image_surface_brightness[j];
					Dvector[i] += Lmatrix_dense[j][i]*sb_adj*covinv;
					//if (sbprofile_surface_brightness[img_index_fgmask]*0.0 != 0.0) die("FUCK");
				}
#ifdef USE_MKL
				Ltrans_stacked[row+j] = Lmatrix_dense[j][i]*sqrt(covinv); // hack to get the cov_inverse in there
#else
				Ltrans[i][j] = Lmatrix_dense[j][i];
#endif
			}
		}

#ifdef USE_OPENMP
		#pragma omp master
		{
			if (show_wtime) {
				wtime = omp_get_wtime() - wtime0;
				if (mpi_id==0) cout << "Wall time for initializing Fmatrix and Dvector: " << wtime << endl;
				wtime0 = omp_get_wtime();
			}
		}
#endif

#ifndef USE_MKL
		// The following is not as fast as the Blas function dsyrk (below), but it still gets the job done

		double *fpmatptr;
		double *lmatptr1, *lmatptr2;
		double *covinvptr;
		#pragma omp for private(n,i,j,l,lmatptr1,lmatptr2,fpmatptr) schedule(static)
		for (n=0; n < ntot_packed; n++) {
			i = i_n[n];
			j = j_n[n];
			fpmatptr = Fmatrix_packed.array()+n;
			lmatptr1 = Ltrans[i];
			lmatptr2 = Ltrans[j];
			(*fpmatptr) = 0;
			if (use_noise_map) {
				covinvptr = imgpixel_covinv_vector;
				for (l=0; l < image_npixels; l++) {
					(*fpmatptr) += (*(lmatptr1++))*(*(lmatptr2++))*(*(covinvptr++));
				}
			} else {
				for (l=0; l < image_npixels; l++) {
					(*fpmatptr) += (*(lmatptr1++))*(*(lmatptr2++));
				}
				(*fpmatptr) *= cov_inverse;
			}
		}
#endif
	}

#ifdef USE_MKL
   cblas_dsyrk(CblasRowMajor,CblasUpper,CblasNoTrans,source_n_amps,image_npixels,1,Ltrans_stacked,image_npixels,0,Fmatrix_stacked.array(),source_n_amps); // Note: this only fills the upper triangular half of the stacked matrix
	LAPACKE_dtrttp(LAPACK_ROW_MAJOR,'U',source_n_amps,Fmatrix_stacked.array(),source_n_amps,Fmatrix_packed.array());
#endif
	if (use_covariance_matrix) generate_Gmatrix();


#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating Fmatrix dense elements: " << wtime << endl;
		wtime0 = omp_get_wtime();
	}
#endif
#ifdef USE_MKL
	delete[] Ltrans_stacked;
#else
	for (i=0; i < source_n_amps; i++) delete[] Ltrans[i];
	delete[] Ltrans;
	delete[] i_n;
	delete[] j_n;
#endif
}



void QLens::generate_Gmatrix()
{
	Dvector_cov = new double[source_n_amps];
	for (int i=0; i < source_n_amps; i++) Dvector_cov[i] = 0;
	Gmatrix_stacked.input(source_n_amps*source_n_amps);
	covmatrix_stacked.input(source_n_amps*source_n_amps);
	// NOTE: right now, this doesn't work if image fluxes are included (so source_n_amps > source_npixels)
	// You need to explicitly make elements zero that are beyond source_npixels*source_npixels (i.e. if there are other amplitudes)
#ifdef USE_OPENMP
	double wtime_gmat0, wtime_gmat;
	if (show_wtime) {
		wtime_gmat0 = omp_get_wtime();
	}
#endif

#ifdef USE_MKL
	LAPACKE_mkl_dtpunpack(LAPACK_ROW_MAJOR,'U','T',source_n_amps,Fmatrix_packed.array(),1,1,source_n_amps,source_n_amps,Fmatrix_stacked.array(),source_n_amps); // fill the lower half of Fmatrix_stacked
	LAPACKE_mkl_dtpunpack(LAPACK_ROW_MAJOR,'U','N',source_npixels,covmatrix_packed.array(),1,1,source_n_amps,source_npixels,covmatrix_stacked.array(),source_n_amps);
	LAPACKE_mkl_dtpunpack(LAPACK_ROW_MAJOR,'U','T',source_npixels,covmatrix_packed.array(),1,1,source_n_amps,source_npixels,covmatrix_stacked.array(),source_n_amps);
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime_gmat = omp_get_wtime() - wtime_gmat0;
		if (mpi_id==0) cout << "Wall time for unpacking F- and cov-matrices: " << wtime_gmat << endl;
		wtime_gmat0 = omp_get_wtime();
	}
#endif

	cblas_dsymm(CblasRowMajor,CblasLeft,CblasUpper,source_n_amps,source_n_amps,1.0,covmatrix_stacked.array(),source_n_amps,Fmatrix_stacked.array(),source_n_amps,0,Gmatrix_stacked.array(),source_n_amps);
	cblas_dsymv(CblasRowMajor,CblasUpper,source_n_amps,1.0,covmatrix_stacked.array(),source_n_amps,Dvector,1,0,Dvector_cov,1);
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime_gmat = omp_get_wtime() - wtime_gmat0;
		if (mpi_id==0) cout << "Wall time for generating Gmatrix and D_cov: " << wtime_gmat << endl;
	}
#endif

#else
	die("MKL is currently required for covariance kernel regularization");
#endif
}

void QLens::add_regularization_term_to_dense_Fmatrix()
{
	int i,j;
	if (dense_Rmatrix) {
		if (!use_covariance_matrix) {
			int n_extra_amps = source_n_amps - source_npixels;
			double *Fptr, *Rptr;
			Fptr = Fmatrix_packed.array();
			Rptr = Rmatrix_packed.array();
			for (i=0; i < source_npixels; i++) {
				for (j=i; j < source_npixels; j++) {
					*(Fptr++) += regularization_parameter*(*(Rptr++));
				}
				Fptr += n_extra_amps;
			}
		} else {
			for (i=0; i < (source_n_amps*source_n_amps); i++) Gmatrix_stacked[i] /= regularization_parameter;
			for (i=0; i < source_n_amps; i++) Dvector_cov[i] /= regularization_parameter;
			int indx_start = 0;
			for (i=0; i < source_npixels; i++) { // additional source amplitudes (beyond source_npixels) are not regularized
				//for (j=0; j < source_npixels; j++) Gmatrix_stacked[i+j] /= regularization_parameter;
				//Dvector_cov[i] /= regularization_parameter;
				//Gmatrix_stacked[indx_start] += regularization_parameter;
				Gmatrix_stacked[indx_start] += 1.0;
				indx_start += source_n_amps+1;
			}
		}
	} else {
		int j,indx_start=0;
		for (i=0; i < source_npixels; i++) { // additional source amplitudes (beyond source_npixels) are not regularized
			Fmatrix_packed[indx_start] += regularization_parameter*Rmatrix[i];
			for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
				//cout << "Fmat row " << i << ", col " << (Rmatrix_index[j]) << ": was " << Fmatrix_packed[indx_start+Rmatrix_index[j]-i] << ", now adding " << (regularization_parameter*Rmatrix[j]) << endl;
				Fmatrix_packed[indx_start+Rmatrix_index[j]-i] += regularization_parameter*Rmatrix[j];
			}
			indx_start += source_n_amps-i;
		}
	}
}

double QLens::calculate_regularization_prior_term()
{
	int i,j;
	double loglike_reg,Es_times_two=0;
	if (dense_Rmatrix) {
		if (use_covariance_matrix) {
			double* sprime = new double[source_npixels];
			for (int i=0; i < source_npixels; i++) sprime[i] = source_pixel_vector[i];
#ifdef USE_MKL
			LAPACKE_dpptrs(LAPACK_ROW_MAJOR,'U',source_npixels,1,covmatrix_factored.array(),sprime,1);
#else
			die("Compiling with MKL is currently required for covariance kernel regularization");
#endif
			for (i=0; i < source_npixels; i++) Es_times_two += source_pixel_vector[i]*sprime[i];
			delete[] sprime;
			//loglike_reg = regularization_parameter*Es_times_two - source_npixels*log(regularization_parameter);
			loglike_reg = regularization_parameter*Es_times_two;
		} else {
			double *Rptr, *sptr_i, *sptr_j, *s_end;
			Rptr = Rmatrix_packed.array();
			s_end = source_pixel_vector + source_npixels;
			for (sptr_i=source_pixel_vector; sptr_i != s_end; sptr_i++) {
				sptr_j = sptr_i;
				Es_times_two += (*sptr_i)*(*(Rptr++))*(*(sptr_j++)); // diagonal element only gets counted once (no factor of 2 here)
				while (sptr_j != s_end) {
					Es_times_two += 2*(*sptr_i)*(*(Rptr++))*(*(sptr_j++));
				}
			}
			loglike_reg = regularization_parameter*Es_times_two - source_npixels*log(regularization_parameter) - Rmatrix_log_determinant;
				//cout << "regparam=" << regularization_parameter << " Es_times_two=" << Es_times_two << " Flogdet=" << Fmatrix_log_determinant << " logreg0=" << loglike_reg << " loglike_reg=" << (loglike_reg+Fmatrix_log_determinant) << " Rlogdet=" << Rmatrix_log_determinant << endl;
		}
	} else {
		for (i=0; i < source_npixels; i++) {
			Es_times_two += Rmatrix[i]*SQR(source_pixel_vector[i]);
			for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
				Es_times_two += 2 * source_pixel_vector[i] * Rmatrix[j] * source_pixel_vector[Rmatrix_index[j]]; // factor of 2 since matrix is symmetric
			}
		}
		loglike_reg = regularization_parameter*Es_times_two - source_npixels*log(regularization_parameter) - Rmatrix_log_determinant;
		//cout << "regparam=" << regularization_parameter << " Es_times_two=" << Es_times_two << " Flogdet=" << Fmatrix_log_determinant << " logreg0=" << loglike_reg << " loglike_reg=" << (loglike_reg+Fmatrix_log_determinant) << " Rlogdet=" << Rmatrix_log_determinant << endl;
	}
	return loglike_reg;
}

bool QLens::optimize_regularization_parameter(const int zsrc_i, const bool dense_Fmatrix, const bool verbal, const bool pre_srcgrid)
{
#ifdef USE_OPENMP
	double wtime_opt0, wtime_opt;
	if (show_wtime) {
		wtime_opt0 = omp_get_wtime();
	}
#endif
	setup_regparam_optimization(zsrc_i,dense_Fmatrix);
	int i;
	double logreg_min;
	double (QLens::*chisqreg)(const double);
	if (dense_Fmatrix) chisqreg = &QLens::chisq_regparam_dense;
	else chisqreg = &QLens::chisq_regparam;
	logreg_min = brents_min_method(chisqreg,optimize_regparam_minlog,optimize_regparam_maxlog,optimize_regparam_tol,verbal);
	//(this->*chisqreg)(log(regularization_parameter)/ln10); // used for testing purposes
	regularization_parameter = pow(10,logreg_min);
	if ((verbal) and (mpi_id==0)) cout << "regparam after optimizing: " << regularization_parameter << endl;

	if (use_covariance_matrix) Gmatrix_log_determinant = regopt_logdet;
	else Fmatrix_log_determinant = regopt_logdet;
	for (i=0; i < source_n_amps; i++) source_pixel_vector[i] = source_pixel_vector_minchisq[i];
	//if (use_lum_weighted_regularization) {
		//for (i=0; i < source_n_amps; i++) source_pixel_vector_input_lumreg[i] = source_pixel_vector_minchisq[i];
	//}
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime_opt = omp_get_wtime() - wtime_opt0;
		if (mpi_id==0) cout << "Wall time for optimizing regularization parameter: " << wtime_opt << endl;
		wtime_opt0 = omp_get_wtime();
	}
#endif
	if ((use_lum_weighted_regularization) and (!pre_srcgrid)) {
		if (!get_lumreg_from_sbweights) {
			if (!use_covariance_matrix) {
				// This means we started with a non-covmatrix based regularization (e.g. curvature) to get the initial luminosity.
				// We'll switch to covariance matrix shortly, so initialize the copies here
				Gmatrix_stacked_copy.input(source_n_amps*source_n_amps);
				covmatrix_stacked_copy.input(source_n_amps*source_n_amps);
				Dvector_cov_copy = new double[source_n_amps];
			}
			if (create_regularization_matrix(zsrc_i,true)==false) return false; // must re-generate covariance matrix with updated correlation lengths (from new pixel sb-weights)
			if (use_covariance_matrix) generate_Gmatrix();
			regopt_chisqmin = 1e30;
#ifdef USE_OPENMP
			if (show_wtime) {
				wtime_opt0 = omp_get_wtime();
			}
#endif
			logreg_min = brents_min_method(chisqreg,optimize_regparam_minlog,optimize_regparam_maxlog,optimize_regparam_tol,verbal);
			//(this->*chisqreg)(log(regparam_lhi)/ln10); // used for testing purposes
			regularization_parameter = pow(10,logreg_min);
			if ((verbal) and (mpi_id==0)) cout << "regparam after optimizing with luminosity-weighted regularization: " << regularization_parameter << endl;
			if (use_covariance_matrix) Gmatrix_log_determinant = regopt_logdet;
			else Fmatrix_log_determinant = regopt_logdet;
			for (i=0; i < source_n_amps; i++) source_pixel_vector[i] = source_pixel_vector_minchisq[i];
			if (verbal) if (mpi_id==0) cout << "loglike=" << regopt_chisqmin << endl;
		}

		for (int j=0; j < lumreg_max_it; j++) {
			if (create_regularization_matrix(zsrc_i,true)==false) return false; // must re-generate covariance matrix with updated correlation lengths (from new pixel sb-weights)
			if (use_covariance_matrix) generate_Gmatrix();
			regopt_chisqmin = 1e30;
			logreg_min = brents_min_method(chisqreg,optimize_regparam_minlog,optimize_regparam_maxlog,optimize_regparam_tol,verbal);
			//(this->*chisqreg)(log(regparam_lhi)/ln10); // used for testing purposes
			regularization_parameter = pow(10,logreg_min);
			if ((verbal) and (mpi_id==0)) cout << "regparam after optimizing with luminosity-weighted regularization: " << regularization_parameter << endl;
			if (use_covariance_matrix) Gmatrix_log_determinant = regopt_logdet;
			else Fmatrix_log_determinant = regopt_logdet;
			for (i=0; i < source_n_amps; i++) source_pixel_vector[i] = source_pixel_vector_minchisq[i];
			if (verbal) if (mpi_id==0) cout << "lumreg_it=" << j << " loglike=" << regopt_chisqmin << endl;
		}

#ifdef USE_OPENMP
		if (show_wtime) {
			if ((lumreg_max_it > 0) or (!get_lumreg_from_sbweights)) {
				wtime_opt = omp_get_wtime() - wtime_opt0;
				if (mpi_id==0) cout << "Wall time for optimizing regparam with lum-weighted regularization: " << wtime_opt << endl;
				wtime_opt0 = omp_get_wtime();
			}
		}
#endif
	}

	update_source_amplitudes(zsrc_i,verbal);
	if ((use_lum_weighted_srcpixel_clustering) and (pre_srcgrid)) {
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime_opt0 = omp_get_wtime();
		}
#endif
		if (!use_saved_sbweights) calculate_subpixel_sbweights(zsrc_i,save_sbweights_during_inversion,verbal); // only need to calculate sb weights for the initial grid, to be used to construct the final pixellation
#ifdef USE_OPENMP
		if (show_wtime) {
			wtime_opt = omp_get_wtime() - wtime_opt0;
			if (mpi_id==0) cout << "Wall time for calculating pixel sbweights: " << wtime_opt << endl;
			wtime_opt0 = omp_get_wtime();
		}
#endif
	}

	if (!dense_Fmatrix) {
		delete[] Fmatrix_copy;
		Fmatrix_copy = NULL;
	}
	if (use_covariance_matrix) {
		delete[] Dvector_cov_copy;
		Dvector_cov_copy = NULL;
	}

	delete[] img_minus_sbprofile;
	delete[] source_pixel_vector_minchisq;
	//if (use_lum_weighted_regularization) delete[] source_pixel_vector_input_lumreg;
	return true;
}

void QLens::setup_regparam_optimization(const int zsrc_i, const bool dense_Fmatrix)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	img_minus_sbprofile = new double[image_npixels];
	int i, pix_i, pix_j, img_index_fgmask;
	if (image_pixel_grid->active_image_pixel_i==NULL) die("did not allocate memory to active_image_pixel_i array");
	for (i=0; i < image_npixels; i++) {
		pix_i = image_pixel_grid->active_image_pixel_i[i];
		pix_j = image_pixel_grid->active_image_pixel_j[i];
		img_index_fgmask = image_pixel_grid->pixel_index_fgmask[pix_i][pix_j];
		img_minus_sbprofile[i] = image_surface_brightness[i] - sbprofile_surface_brightness[img_index_fgmask];
		if (((!include_imgfluxes_in_inversion) and (!include_srcflux_in_inversion)) and (n_sourcepts_fit > 0)) img_minus_sbprofile[i] -= point_image_surface_brightness[i];
	}

	source_pixel_vector_minchisq = new double[source_n_amps];
	//if (use_lum_weighted_regularization) source_pixel_vector_input_lumreg = new double[source_npixels];
	regopt_chisqmin = 1e30;
	regopt_logdet = 1e30; // this will be changed during optimization

	if (dense_Fmatrix) {
		if (use_covariance_matrix) {
			Gmatrix_stacked_copy.input(source_n_amps*source_n_amps);
			covmatrix_stacked_copy.input(source_n_amps*source_n_amps);
			Dvector_cov_copy = new double[source_n_amps];
		} else {
			int ntot = source_n_amps*(source_n_amps+1)/2;
			Fmatrix_packed_copy.input(ntot);
		}
	} else {
		if (Fmatrix_nn==0) die("Fmatrix length has not been set");
		Fmatrix_copy = new double[Fmatrix_nn];
	}
	temp_src.input(source_n_amps);
}

void QLens::calculate_subpixel_sbweights(const int zsrc_i, const bool save_sbweights, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	SourcePixelGrid *cartesian_srcgrid = image_pixel_grid->cartesian_srcgrid;
	int npix_in_mask;
	int *pixptr_i, *pixptr_j;
	if (include_extended_mask_in_inversion) {
		npix_in_mask = image_pixel_grid->ntot_cells_emask;
		pixptr_i = image_pixel_grid->emask_pixels_i;
		pixptr_j = image_pixel_grid->emask_pixels_j;
	} else {
		npix_in_mask = image_pixel_grid->ntot_cells;
		pixptr_i = image_pixel_grid->masked_pixels_i;
		pixptr_j = image_pixel_grid->masked_pixels_j;
	}
	int i,j,k,n,l,m,nsubpix;
	double sb, max_sb = 1e-30;

	if (save_sbweights) {
		n_sbweights = 0;
		if (saved_sbweights != NULL) delete[] saved_sbweights;
		for (n=0; n < npix_in_mask; n++) {
			i = pixptr_i[n];
			j = pixptr_j[n];
			nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
			n_sbweights += nsubpix;
		}
		if (mpi_id==0) cout << "Saving " << n_sbweights << " sbweights" << endl;
		saved_sbweights = new double[n_sbweights];
		l=0;
	}

	bool at_least_one_lensed_src = false;
	for (k=0; k < n_sb; k++) {
		if (sb_list[k]->is_lensed) {
			at_least_one_lensed_src = true;
			break;
		}
	}

	if ((source_fit_mode==Delaunay_Source) and (image_pixel_grid->delaunay_srcgrid == NULL)) die("delaunay_srcgrid has not been created");
	if ((source_fit_mode==Cartesian_Source) and (cartesian_srcgrid == NULL)) die("cartesian_srcgrid has not been created");

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
			nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
			for (k=0; k < nsubpix; k++) {
				// This needs to be generalized so the weights can be created using different source modes (shapelet, sbprofile, etc.)...did I already accomplish this? (check)
				sb = 0;
				if (source_fit_mode==Delaunay_Source) sb += image_pixel_grid->delaunay_srcgrid->interpolate_surface_brightness(image_pixel_grid->subpixel_center_sourcepts[i][j][k],false,thread);
				else if (source_fit_mode==Cartesian_Source) sb += cartesian_srcgrid->find_lensed_surface_brightness_interpolate(image_pixel_grid->subpixel_center_sourcepts[i][j][k],thread);
				else if (at_least_one_lensed_src) {
					for (m=0; m < n_sb; m++) {
						if (sb_list[m]->is_lensed) {
							sb += sb_list[m]->surface_brightness(image_pixel_grid->subpixel_center_sourcepts[i][j][k][0],image_pixel_grid->subpixel_center_sourcepts[i][j][k][1]);
						}
					}
				}
				if (sb < 0) sb = 0;

				if (sb > max_sb) {
					#pragma omp critical
					max_sb = sb;
				}
				image_pixel_grid->subpixel_weights[i][j][k] = sb;
			}
		}
	}
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
		for (k=0; k < nsubpix; k++) {
			image_pixel_grid->subpixel_weights[i][j][k] /= max_sb;
			if (save_sbweights) saved_sbweights[l++] = image_pixel_grid->subpixel_weights[i][j][k];
		}
	}
	if ((save_sbweights) and (mpi_id==0)) cout << "Pixel sb-weights saved" << endl;
}

void QLens::calculate_subpixel_distweights(const int zsrc_i)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	double xc, yc, xc_approx, yc_approx, sig, rc;
	rc = lumreg_rc;
	sig = image_pixel_grid->find_approx_source_size(xc_approx,yc_approx);
	if (fix_lumreg_sig) sig = lumreg_sig;
	if (auto_lumreg_center) {
		xc = xc_approx;
		yc = yc_approx;
	} else {
		if (lensed_lumreg_center) {
			lensvector xl;
			xl[0] = lumreg_xcenter;
			xl[1] = lumreg_ycenter;
			find_sourcept(xl,xc,yc,0,reference_zfactors,default_zsrc_beta_factors);
			//if ((verbal) and (mpi_id==0)) cout << "center coordinates in source plane: xc=" << xc << ", yc=" << yc << endl;
			if (lensed_lumreg_rc) {
				int i, phi_nn = 24;
				double phi, phi_step = M_2PI/(phi_nn-1);
				double xc2, yc2;
				rc = 0;
				for (i=0, phi=0; i < phi_nn; i++, phi += phi_step) {
					xl[0] = lumreg_xcenter + lumreg_rc*cos(phi);
					xl[1] = lumreg_ycenter + lumreg_rc*sin(phi);
					find_sourcept(xl,xc2,yc2,0,reference_zfactors,default_zsrc_beta_factors);
					rc += SQR(xc2-xc)+SQR(yc2-yc);
				}
				rc = sqrt(rc/phi_nn);
			}
			// this is all repeated in function calculate_distreg_srcpixel_weights(...), which is not great...find a way to consolidate?
			//if ((verbal) and (mpi_id==0)) cout << "estimated lumreg_rc mapped to source plane: " << rc << endl;

		} else {
			xc = lumreg_xcenter; yc = lumreg_ycenter;
		}
	}

	int npix_in_mask;
	int *pixptr_i, *pixptr_j;
	if (include_extended_mask_in_inversion) {
		npix_in_mask = image_pixel_grid->ntot_cells_emask;
		pixptr_i = image_pixel_grid->emask_pixels_i;
		pixptr_j = image_pixel_grid->emask_pixels_j;
	} else {
		npix_in_mask = image_pixel_grid->ntot_cells;
		pixptr_i = image_pixel_grid->masked_pixels_i;
		pixptr_j = image_pixel_grid->masked_pixels_j;
	}
	int i,j,k,n,l,nsubpix,n_weights;

	n_weights = 0;
	if (saved_sbweights != NULL) delete[] saved_sbweights;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
		n_weights += nsubpix;
	}
	double *scaled_dists = new double[n_weights];
	lensvector **srcpts = new lensvector*[n_weights];
	l=0;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		for (k=0; k < nsubpix; k++) {
			srcpts[l++] = &image_pixel_grid->subpixel_center_sourcepts[i][j][k];
		}
	}
	calculate_srcpixel_scaled_distances(xc,yc,sig,scaled_dists,srcpts,n_weights,lumreg_e1,lumreg_e2);

	l=0;
	double scaled_rcsq = SQR(rc/sig);
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
		for (k=0; k < nsubpix; k++) {
			image_pixel_grid->subpixel_weights[i][j][k] = exp(-pow(sqrt(SQR(scaled_dists[l++]) + scaled_rcsq),regparam_lum_index));
			//cout << "WEIGHT " << (l-1) << ": " << image_pixel_grid->subpixel_weights[i][j][k] << " " << scaled_dists[l-1] << " " << (*srcpts[l-1])[0] << " " << (*srcpts[l-1])[1] << " " << xc << " " << yc << " " << sig << endl;
		}
	}
	delete[] scaled_dists;
	delete[] srcpts;
}

void QLens::calculate_lumreg_srcpixel_weights(const int zsrc_i, const bool use_sbweights)
{
	double lumfac, max_sb=-1e30;
	int i;
	if (use_sbweights) find_srcpixel_weights(zsrc_i);
	for (i=0; i < source_npixels; i++) {
		if (source_pixel_vector[i] > max_sb) max_sb = source_pixel_vector[i];
	}
	if (use_lum_weighted_regularization) {
		for (i=0; i < source_npixels; i++) {
			if (lum_weight_function==0) {
				if (source_pixel_vector[i]==max_sb) reg_weight_factor[i] = 1;
				else {
					lumfac = (source_pixel_vector[i] > 0) ? pow(1 - source_pixel_vector[i]/max_sb,regparam_lum_index) : 1;
					reg_weight_factor[i] = exp(-regparam_lsc*lumfac);
				}
			} else if (lum_weight_function==1) {
				lumfac = (source_pixel_vector[i] > 0) ? 1 - pow(source_pixel_vector[i]/max_sb,regparam_lum_index) : 1;
				reg_weight_factor[i] = exp(-regparam_lsc*lumfac);
			} else {
				if (regparam_lum_index==0) {
					reg_weight_factor[i] = exp(-regparam_lsc);
				} else {
					lumfac = (source_pixel_vector[i] > 0) ? pow(1-pow(source_pixel_vector[i]/max_sb,1.0/regparam_lum_index),regparam_lum_index) : 1;
					reg_weight_factor[i] = exp(-regparam_lsc*lumfac);
				}
			}
		}
	}
	/*
	if (use_second_covariance_kernel) {
		for (i=0; i < source_npixels; i++) {
			//if (regparam_lum_index==0) lumfac2 = 1;
			lumfac2 = (source_pixel_vector[i] > 0) ? pow(1 - source_pixel_vector[i]/max_sb,2) : 1;
			//reg_weight_factor2[i] = exp(-pow(regparam_lsc,2)*lumfac2);
			//reg_weight_factor2[i] = 1.0;
		}
	}
	*/
}

void QLens::calculate_distreg_srcpixel_weights(const int zsrc_i, const double xc_in, const double yc_in, const double sig, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	double xc, yc, rc;
	rc = lumreg_rc;
	if (auto_lumreg_center) {
		if (lumreg_center_from_ptsource) {
			if (n_sourcepts_fit==0) die("no source points have been defined");
			xc = sourcepts_fit[0][0];
			yc = sourcepts_fit[0][1];
		} else {
			xc = xc_in;
			yc = yc_in;
		}
	} else {
		if (lensed_lumreg_center) {
			lensvector xl;
			xl[0] = lumreg_xcenter;
			xl[1] = lumreg_ycenter;
			find_sourcept(xl,xc,yc,0,extended_src_zfactors[zsrc_i],extended_src_beta_factors[zsrc_i]);
			if ((verbal) and (mpi_id==0)) cout << "center coordinates in source plane: xc=" << xc << ", yc=" << yc << endl;
			if ((lensed_lumreg_rc) and (lumreg_rc > 0)) {
				int i, phi_nn = 24;
				double phi, phi_step = M_2PI/(phi_nn-1);
				double xc2, yc2;
				rc = 0;
				for (i=0, phi=0; i < phi_nn; i++, phi += phi_step) {
					xl[0] = lumreg_xcenter + lumreg_rc*cos(phi);
					xl[1] = lumreg_ycenter + lumreg_rc*sin(phi);
					find_sourcept(xl,xc2,yc2,0,extended_src_zfactors[zsrc_i],extended_src_beta_factors[zsrc_i]);
					rc += SQR(xc2-xc)+SQR(yc2-yc);
				}
				rc = sqrt(rc/phi_nn);
				if ((verbal) and (mpi_id==0)) cout << "estimated lumreg_rc mapped to source plane: " << rc << endl;
			}
		} else {
			xc = lumreg_xcenter; yc = lumreg_ycenter;
		}
	}
	double *scaled_dists = new double[source_npixels];
	lensvector **srcpts = new lensvector*[source_npixels];
	if (image_pixel_grid->delaunay_srcgrid == NULL) die("Delaunay source grid has not been created");
	for (int i=0; i < source_npixels; i++) srcpts[i] = &(image_pixel_grid->delaunay_srcgrid->srcpts[i]);
	calculate_srcpixel_scaled_distances(xc,yc,sig,scaled_dists,srcpts,source_npixels,lumreg_e1,lumreg_e2);
	double scaled_rcsq = SQR(rc/sig);
	for (int i=0; i < source_npixels; i++) {
		if (lum_weight_function==0) {
			reg_weight_factor[i] = exp(-regparam_lsc*pow(sqrt(SQR(scaled_dists[i]) + scaled_rcsq),regparam_lum_index));
			//reg_weight_factor[i] = (exp(-regparam_lsc*pow(scaled_dists[i],regparam_lum_index)) + lumreg_rc*exp(-regparam_lsc2*pow(scaled_dists[i],regparam_lum_index2)))/(1+lumreg_rc);
		} else {
			die("lumweight_func greater than 0 not supported in dist-weighted regularization");
		}
	}

	/*
	if (use_second_covariance_kernel) {
		for (int i=0; i < source_npixels; i++) {
			if (lum_weight_function==0) {
				//reg_weight_factor2[i] = exp(-regparam_lsc2*pow(scaled_dists[i],regparam_lum_index2));
				reg_weight_factor2[i] = 1.0;
			} else {
				die("lumweight_func greater than 0 not supported in dist-weighted regularization");
			}
		}
	}
	*/


	delete[] scaled_dists;
	delete[] srcpts;
}

void QLens::calculate_srcpixel_scaled_distances(const double xc, const double yc, const double sig, double *dists, lensvector **srcpts, const int nsrcpts, const double e1, const double e2)
{
	double angle;
	if (e1==0) {
		if (e2 > 0) angle = M_HALFPI;
		else if (e2==0) angle = 0.0;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(e2/e1));
		if (e1 < 0) {
			if (e2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (e2 < 0) {
			angle = -angle;
		}
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

void QLens::calculate_mag_srcpixel_weights(const int zsrc_i)
{
	int i;
	double logmag,logmag_max = -1e30;
	if (zsrc_i==0) // at the moment, this is only set up for the first source being modeled
	{
		DelaunayGrid *srcgrid = image_pixel_grids[zsrc_i]->delaunay_srcgrid;
		for (i=0; i < srcgrid->n_srcpts; i++) {
			logmag = -log(srcgrid->inv_magnification[i])/ln10;
			if (logmag > logmag_max) logmag_max = logmag;
		}
		for (i=0; i < srcgrid->n_srcpts; i++) {
			if (srcgrid->active_pixel[i]) {
				logmag = -log(srcgrid->inv_magnification[i])/ln10;
				if (mag_weight_index != 0) {
					reg_weight_factor[i] *= exp(-mag_weight_sc*pow((logmag_max-logmag),mag_weight_index));
				}
			}
		}
	}
}


void QLens::find_srcpixel_weights(const int zsrc_i)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	int npix_in_mask;
	int *pixptr_i, *pixptr_j;
	if (include_extended_mask_in_inversion) {
		npix_in_mask = image_pixel_grid->ntot_cells_emask;
		pixptr_i = image_pixel_grid->emask_pixels_i;
		pixptr_j = image_pixel_grid->emask_pixels_j;
	} else {
		npix_in_mask = image_pixel_grid->ntot_cells;
		pixptr_i = image_pixel_grid->masked_pixels_i;
		pixptr_j = image_pixel_grid->masked_pixels_j;
	}
	int i,j,k,n,indx,trinum,nsubpix;

	int nsrcpix = image_pixel_grid->delaunay_srcgrid->n_srcpts; // note, this might not be the same as source_npixels if there are inactive source pixels
	double *srcpixel_weights = new double[nsrcpix];
	int *srcpixel_nimgpts = new int[nsrcpix];
	for (i=0; i < nsrcpix; i++) {
		srcpixel_weights[i] = 0;
		srcpixel_nimgpts[i] = 0;
	}

	bool inside_triangle;
	lensvector *pt;
	#pragma omp parallel for private(n,i,j,k,nsubpix,indx,trinum,inside_triangle,pt) schedule(static) 
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
		for (k=0; k < nsubpix; k++) {
			pt = &image_pixel_grid->subpixel_center_sourcepts[i][j][k];
			// This needs to be generalized so the weights can be created using different source modes (shapelet, sbprofile, etc.)
			inside_triangle = false;
			trinum = image_pixel_grid->delaunay_srcgrid->search_grid(0,*pt,inside_triangle); // maybe you can speed this up later by choosing a better initial triangle
			indx = image_pixel_grid->delaunay_srcgrid->find_closest_vertex(trinum,*pt);
			#pragma omp critical
			{
				srcpixel_weights[indx] += image_pixel_grid->subpixel_weights[i][j][k];
				srcpixel_nimgpts[indx]++;
			}
		}
	}

	indx=0;
	for (i=0; i < nsrcpix; i++) {
		if (image_pixel_grid->delaunay_srcgrid->active_pixel[i]) source_pixel_vector[indx++] = srcpixel_weights[i] / srcpixel_nimgpts[i];
	}
}

void QLens::load_pixel_sbweights(const int zsrc_i)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	int npix_in_mask;
	int *pixptr_i, *pixptr_j;
	if (include_extended_mask_in_inversion) {
		npix_in_mask = image_pixel_grid->ntot_cells_emask;
		pixptr_i = image_pixel_grid->emask_pixels_i;
		pixptr_j = image_pixel_grid->emask_pixels_j;
	} else {
		npix_in_mask = image_pixel_grid->ntot_cells;
		pixptr_i = image_pixel_grid->masked_pixels_i;
		pixptr_j = image_pixel_grid->masked_pixels_j;
	}
	int i,j,k,n,l,nsubpix;

	int nweights=0;
	for (int n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
		nweights += nsubpix;
	}
	if (nweights != n_sbweights) die("number of subpixels (%i) doesn't match number of saved sb-weights (%i)",nweights,n_sbweights);

	l=0;
	for (n=0; n < npix_in_mask; n++) {
		i = pixptr_i[n];
		j = pixptr_j[n];
		nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]); // why not just store the square and avoid having to always take the square?
		for (k=0; k < nsubpix; k++) {
			image_pixel_grid->subpixel_weights[i][j][k] = saved_sbweights[l++];
		}
	}
}

double QLens::chisq_regparam(const double logreg)
{
	double cov_inverse, cov_inverse_bg, chisq; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (!use_noise_map) {
		if (background_pixel_noise==0) cov_inverse_bg = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
		else cov_inverse_bg = 1.0/SQR(background_pixel_noise);
	}

	regularization_parameter = pow(10,logreg);
	int i,j,k;

	for (i=0; i < Fmatrix_nn; i++) {
		Fmatrix_copy[i] = Fmatrix[i];
	}

	for (i=0; i < source_npixels; i++) {
		Fmatrix_copy[i] += regularization_parameter*Rmatrix[i];
		for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
			for (k=Fmatrix_index[i]; k < Fmatrix_index[i+1]; k++) {
				if (Rmatrix_index[j]==Fmatrix_index[k]) {
					Fmatrix_copy[k] += regularization_parameter*Rmatrix[j];
				}
			}
		}
	}

	double Fmatrix_logdet;

	if (inversion_method==MUMPS) invert_lens_mapping_MUMPS(false,true);
	else if (inversion_method==UMFPACK) invert_lens_mapping_UMFPACK(false,true);
	else die("can only use MUMPS or UMFPACK for sparse inversions with optimize_regparam on");

	double temp_img, Ed_times_two=0,Es_times_two=0;

	#pragma omp parallel for private(temp_img,i,j,cov_inverse) schedule(static) reduction(+:Ed_times_two)
	for (i=0; i < image_npixels; i++) {
		if (use_noise_map) cov_inverse = imgpixel_covinv_vector[i];
		else cov_inverse = cov_inverse_bg;
		temp_img = 0;
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			temp_img += Lmatrix[j]*source_pixel_vector[Lmatrix_index[j]];
		}

		// NOTE: this chisq does not include foreground mask pixels that lie outside the primary mask, since those pixels don't contribute to determining the regularization
		Ed_times_two += SQR(temp_img - img_minus_sbprofile[i])*cov_inverse;
	}
	for (i=0; i < source_npixels; i++) {
		Es_times_two += Rmatrix[i]*SQR(source_pixel_vector[i]);
		for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
			Es_times_two += 2 * source_pixel_vector[i] * Rmatrix[j] * source_pixel_vector[Rmatrix_index[j]]; // factor of 2 since matrix is symmetric
		}
	}
	//cout << "regparam: "<< regularization_parameter << endl;
	//cout << "chisqreg: " << (Ed_times_two + regularization_parameter*Es_times_two + Fmatrix_log_determinant - source_n_amps*log(regularization_parameter) - Rmatrix_log_determinant) << endl;
	//cout << "reg*Es_times_two=" << (regularization_parameter*Es_times_two) << " n_shapelets*log(regparam)=" << (-source_n_amps*log(regularization_parameter)) << " -det(Rmatrix)=" << (-Rmatrix_log_determinant) << " log(Fmatrix)=" << Fmatrix_logdet << endl;

	chisq = (Ed_times_two + regularization_parameter*Es_times_two + Fmatrix_log_determinant - source_npixels*log(regularization_parameter) - Rmatrix_log_determinant);
	if (chisq < regopt_chisqmin) {
		regopt_chisqmin = chisq;
		for (i=0; i < source_n_amps; i++) source_pixel_vector_minchisq[i] = source_pixel_vector[i];
		regopt_logdet = Fmatrix_logdet;
	}
	return chisq;
}

double QLens::chisq_regparam_dense(const double logreg)
{
	double chisq, logdet, cov_inverse, cov_inverse_bg; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (background_pixel_noise==0) cov_inverse_bg = 1; // if there is no noise it doesn't matter what the cov_inverse is, since we won't be regularizing
	else cov_inverse_bg = 1.0/SQR(background_pixel_noise);

	regularization_parameter = pow(10,logreg);
	int i,j;

	if (dense_Rmatrix) {
		if (!use_covariance_matrix) {
			for (i=0; i < Fmatrix_packed.size(); i++) {
				Fmatrix_packed_copy[i] = Fmatrix_packed[i];
			}
			int n_extra_amps = source_n_amps - source_npixels;
			double *Fptr, *Rptr;
			Fptr = Fmatrix_packed_copy.array();
			Rptr = Rmatrix_packed.array();
			for (i=0; i < source_npixels; i++) {
				for (j=i; j < source_npixels; j++) {
					*(Fptr++) += regularization_parameter*(*(Rptr++));
				}
				Fptr += n_extra_amps;
			}
		} else {
			for (i=0; i < Gmatrix_stacked.size(); i++) {
				Gmatrix_stacked_copy[i] = Gmatrix_stacked[i];
			}
			for (i=0; i < source_n_amps; i++) {
				//Dvector_cov_copy[i] = Dvector_cov[i]/regularization_parameter;
				Dvector_cov_copy[i] = Dvector_cov[i];
			}
			for (i=0; i < (source_n_amps*source_n_amps); i++) Gmatrix_stacked_copy[i] /= regularization_parameter;
			for (i=0; i < source_n_amps; i++) Dvector_cov_copy[i] /= regularization_parameter;
			int indx_start = 0;
			for (i=0; i < source_npixels; i++) { // additional source amplitudes (beyond source_npixels) are not regularized
				//Dvector_cov_copy[i] /= regularization_parameter;
				//Gmatrix_stacked[indx_start] += regularization_parameter;
				Gmatrix_stacked_copy[indx_start] += 1.0;
				indx_start += source_n_amps+1;
			}
		}
	} else {
		for (i=0; i < Fmatrix_packed.size(); i++) {
			Fmatrix_packed_copy[i] = Fmatrix_packed[i];
		}

		int k,indx_start=0;
		for (i=0; i < source_npixels; i++) {
			Fmatrix_packed_copy[indx_start] += regularization_parameter*Rmatrix[i];
			for (k=Rmatrix_index[i]; k < Rmatrix_index[i+1]; k++) {
				Fmatrix_packed_copy[indx_start+Rmatrix_index[k]-i] += regularization_parameter*Rmatrix[k];
			}
			indx_start += source_n_amps-i;
		}
	}

	double Fmatrix_logdet, Gmatrix_logdet;
	if (!use_covariance_matrix) {
#ifdef USE_MKL
		lapack_int status;
		status = LAPACKE_dpptrf(LAPACK_ROW_MAJOR,'U',source_n_amps,Fmatrix_packed_copy.array());
		if (status != 0) warn("Matrix was not invertible and/or positive definite");
		for (int i=0; i < source_n_amps; i++) source_pixel_vector[i] = Dvector[i];
		LAPACKE_dpptrs(LAPACK_ROW_MAJOR,'U',source_n_amps,1,Fmatrix_packed_copy.array(),source_pixel_vector,1);
		Cholesky_logdet_packed(Fmatrix_packed_copy.array(),Fmatrix_logdet,source_n_amps);
#else
		// At the moment, the native (non-MKL) Cholesky decomposition code does a lower triangular decomposition; since Fmatrix/Rmatrix stores the upper
		// triangular part, we have to switch Fmatrix to a lower triangular version here. Fix later so it uses the upper triangular Cholesky version!!!
		repack_matrix_lower(Fmatrix_packed_copy);

		bool status = Cholesky_dcmp_packed(Fmatrix_packed_copy.array(),source_n_amps);
		if (!status) die("Cholesky decomposition failed");
		Cholesky_solve_lower_packed(Fmatrix_packed_copy.array(),Dvector,source_pixel_vector,source_n_amps);
		Cholesky_logdet_lower_packed(Fmatrix_packed_copy.array(),Fmatrix_logdet,source_n_amps);
#endif
	} else {
#ifdef USE_MKL
		int *ipiv = new int[source_npixels];
		lapack_int status;
		status = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, source_n_amps, source_n_amps, Gmatrix_stacked_copy.array(), source_n_amps, ipiv);
		if (status != 0) warn("Matrix was not invertible");
		for (int i=0; i < source_n_amps; i++) source_pixel_vector[i] = Dvector_cov_copy[i];
		LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', source_n_amps, 1, Gmatrix_stacked_copy.array(), source_n_amps, ipiv, source_pixel_vector, 1);
		LU_logdet_stacked(Gmatrix_stacked_copy.array(),Gmatrix_logdet,source_n_amps);
		delete[] ipiv;
#else
		die("Currently MKL is required to do cov_inverse kernel regularization");
#endif
	}

	double temp_img, Ed_times_two=0,Es_times_two=0;
	double *Lmatptr;
	double *tempsrcptr = source_pixel_vector;
	double *tempsrc_end = source_pixel_vector + source_n_amps;

	#pragma omp parallel for private(temp_img,i,j,Lmatptr,tempsrcptr,cov_inverse) schedule(static) reduction(+:Ed_times_two)
	for (i=0; i < image_npixels; i++) {
		temp_img = 0;
		if (use_noise_map) {
			cov_inverse = imgpixel_covinv_vector[i];
		} else {
			cov_inverse = cov_inverse_bg;
		}
		if ((source_fit_mode==Shapelet_Source) or (inversion_method==DENSE)) {
			// even if using a pixellated source, if inversion_method is set to DENSE, only the dense form of the Lmatrix has been convolved with the PSF, so this form must be used
			Lmatptr = (Lmatrix_dense.pointer())[i];
			tempsrcptr = source_pixel_vector;
			while (tempsrcptr != tempsrc_end) {
				temp_img += (*(Lmatptr++))*(*(tempsrcptr++));
			}
		} else {
			for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
				temp_img += Lmatrix[j]*source_pixel_vector[Lmatrix_index[j]];
			}
		}
		// NOTE: this chisq does not include foreground mask pixels that lie outside the primary mask, since those pixels don't contribute to determining the regularization
		Ed_times_two += SQR(temp_img - img_minus_sbprofile[i])*cov_inverse;
	}

/*
	if (dense_Rmatrix) {
		if (use_covariance_matrix) {
			double* sprime = new double[source_npixels];
			for (int i=0; i < source_npixels; i++) sprime[i] = source_pixel_vector[i];
#ifdef USE_MKL
			LAPACKE_dpptrs(LAPACK_ROW_MAJOR,'U',source_npixels,1,covmatrix_factored.array(),sprime,1);
#else
			die("Compiling with MKL is currently required for covariance kernel regularization");
#endif
			for (i=0; i < source_npixels; i++) Es_times_two += source_pixel_vector[i]*sprime[i];
			delete[] sprime;
		} else {
			double *Rptr, *sptr_i, *sptr_j, *s_end;
			Rptr = Rmatrix_packed.array();
			s_end = source_pixel_vector + source_npixels;
			for (sptr_i=source_pixel_vector; sptr_i != s_end; sptr_i++) {
				sptr_j = sptr_i;
				Es_times_two += (*sptr_i)*(*(Rptr++))*(*sptr_j++);
				while (sptr_j != s_end) {
					Es_times_two += 2*(*sptr_i)*(*(Rptr++))*(*(sptr_j++)); // factor of 2 since matrix is symmetric
				}
			}
		}
	} else {
		for (i=0; i < source_npixels; i++) {
			Es_times_two += Rmatrix[i]*SQR(source_pixel_vector[i]);
			for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
				Es_times_two += 2 * source_pixel_vector[i] * Rmatrix[j] * source_pixel_vector[Rmatrix_index[j]]; // factor of 2 since matrix is symmetric
			}
		}
	}
	//cout << "regparam: "<< regularization_parameter << endl;
	//cout << "chisqreg: " << (Ed_times_two + regularization_parameter*Es_times_two + Fmatrix_logdet - source_n_amps*log(regularization_parameter) - Rmatrix_log_determinant) << endl;
	//cout << "reg*Es_times_two=" << (regularization_parameter*Es_times_two) << " n_shapelets*log(regparam)=" << (-source_n_amps*log(regularization_parameter)) << " -det(Rmatrix)=" << (-Rmatrix_log_determinant) << " log(Fmatrix)=" << Fmatrix_logdet << endl;

	if (use_covariance_matrix)
		chisq = (Ed_times_two + regularization_parameter*Es_times_two + Gmatrix_logdet);
	else
		chisq = (Ed_times_two + regularization_parameter*Es_times_two + Fmatrix_logdet - source_npixels*log(regularization_parameter) - Rmatrix_log_determinant);
		*/

	double loglike_reg = calculate_regularization_prior_term();
	chisq = Ed_times_two + loglike_reg;
	logdet = (use_covariance_matrix) ? Gmatrix_logdet : Fmatrix_logdet;
	chisq += logdet;

	if (chisq < regopt_chisqmin) {
		regopt_chisqmin = chisq;
		for (i=0; i < source_n_amps; i++) source_pixel_vector_minchisq[i] = source_pixel_vector[i];
		regopt_logdet = logdet;
	}
	return chisq;
}

/*
double QLens::chisq_regparam_it_lumreg_dense(const double logreg)
{
	regparam_lhi = pow(10,logreg);
	int i;
	for (i=0; i < source_n_amps; i++) source_pixel_vector[i] = source_pixel_vector_input_lumreg[i];

	// Note: if we're also lum_weighted_srcpixel_clustering, we don't use luminosity-weighted regularization
	// with the initial grid (with no lum-weighted clustering), since there's too much overhead and it doesn't make a big difference
	lumreg_it = 0;
	double chisq, chisqprev;
	chisq = chisq_regparam_lumreg_dense();
	//if ((verbal) and (mpi_id==0)) cout << "lumreg_it=" << lumreg_it << " loglike=" << chisq << endl;
	lumreg_it++;
	do {
		chisqprev = chisq;
		chisq = chisq_regparam_lumreg_dense();
		//if ((verbal) and (mpi_id==0)) cout << "lumreg_it=" << lumreg_it << " loglike=" << chisq << endl;

		if (chisq > chisqprev) {
			//if (verbal) warn("chi-square became worse during iterations of luminosity-weighted regularization");
			chisq = chisqprev;
			break;
		}
	} while ((++lumreg_it < lumreg_max_it) and (abs(chisq-chisqprev) > chisqtol_lumreg*chisq));
	return chisq;
}

double QLens::chisq_regparam_it_lumreg_dense_final(const bool verbal)
{
	lumreg_it = 0;
	double chisq, chisqprev;
	chisq = chisq_regparam_lumreg_dense();
	if ((verbal) and (mpi_id==0)) cout << "lumreg_it=" << lumreg_it << " loglike=" << chisq << endl;
	if (lumreg_max_it_final > 0) {
		lumreg_it++;
		do {
			chisqprev = chisq;
			chisq = chisq_regparam_lumreg_dense();
			if ((verbal) and (mpi_id==0)) cout << "lumreg_it=" << lumreg_it << " loglike=" << chisq << endl;

			if (chisq > chisqprev) {
				if (verbal) warn("chi-square became worse during iterations of luminosity-weighted regularization");
				chisq = chisqprev;
				break;
			}
		} while ((++lumreg_it < lumreg_max_it_final) and (abs(chisq-chisqprev) > chisqtol_lumreg*chisq));
	}
	return chisq;
}

double QLens::chisq_regparam_lumreg_dense()
{
	double chisq, logdet, covariance; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (background_pixel_noise==0) covariance = 1; // if there is no noise it doesn't matter what the covariance is, since we won't be regularizing
	else covariance = SQR(background_pixel_noise);

	int i,j;
	if (dense_Rmatrix) {
		if (!use_covariance_matrix) {
			for (i=0; i < Fmatrix_packed.size(); i++) {
				Fmatrix_packed_copy[i] = Fmatrix_packed[i];
			}
		} else {
			for (i=0; i < Gmatrix_stacked.size(); i++) {
				Gmatrix_stacked_copy[i] = Gmatrix_stacked[i];
				covmatrix_stacked_copy[i] = covmatrix_stacked[i];
			}
			for (i=0; i < source_n_amps; i++) {
				//Dvector_cov_copy[i] = Dvector_cov[i]/regularization_parameter;
				Dvector_cov_copy[i] = Dvector_cov[i];
			}
		}
	} else {
		for (i=0; i < Fmatrix_packed.size(); i++) {
			Fmatrix_packed_copy[i] = Fmatrix_packed[i];
		}
	}
	add_lum_weighted_reg_term(true,true);

	double Fmatrix_logdet, Gmatrix_logdet;
	if (!use_covariance_matrix) {
#ifdef USE_MKL
		LAPACKE_dpptrf(LAPACK_ROW_MAJOR,'U',source_n_amps,Fmatrix_packed_copy.array());
		for (int i=0; i < source_n_amps; i++) source_pixel_vector[i] = Dvector[i];
		LAPACKE_dpptrs(LAPACK_ROW_MAJOR,'U',source_n_amps,1,Fmatrix_packed_copy.array(),source_pixel_vector,1);
		Cholesky_logdet_packed(Fmatrix_packed_copy.array(),Fmatrix_logdet,source_n_amps);
#else
		// At the moment, the native (non-MKL) Cholesky decomposition code does a lower triangular decomposition; since Fmatrix/Rmatrix stores the upper
		// triangular part, we have to switch Fmatrix to a lower triangular version here. Fix later so it uses the upper triangular Cholesky version!!!
		repack_matrix_lower(Fmatrix_dense);

		bool status = Cholesky_dcmp_packed(Fmatrix_packed_copy.array(),source_n_amps);
		if (!status) die("Cholesky decomposition failed");
		Cholesky_solve_lower_packed(Fmatrix_packed_copy.array(),Dvector,source_pixel_vector,source_n_amps);
		Cholesky_logdet_lower_packed(Fmatrix_packed_copy.array(),Fmatrix_logdet,source_n_amps);
#endif
	} else {
#ifdef USE_MKL
		int *ipiv = new int[source_npixels];
		LAPACKE_dgetrf(LAPACK_ROW_MAJOR, source_n_amps, source_n_amps, Gmatrix_stacked_copy.array(), source_n_amps, ipiv);
		for (int i=0; i < source_n_amps; i++) source_pixel_vector[i] = Dvector_cov_copy[i];
		LAPACKE_dgetrs (LAPACK_ROW_MAJOR, 'N', source_n_amps, 1, Gmatrix_stacked_copy.array(), source_n_amps, ipiv, source_pixel_vector, 1);
		LU_logdet_stacked(Gmatrix_stacked_copy.array(),Gmatrix_logdet,source_n_amps);
		delete[] ipiv;
#else
		die("Currently MKL is required to do covariance kernel regularization");
#endif
	}

	double temp_img, Ed_times_two=0,Es_times_two=0;
	double *Lmatptr;
	double *tempsrcptr = source_pixel_vector;
	double *tempsrc_end = source_pixel_vector + source_n_amps;

	#pragma omp parallel for private(temp_img,i,j,Lmatptr,tempsrcptr) schedule(static) reduction(+:Ed_times_two)
	for (i=0; i < image_npixels; i++) {
		temp_img = 0;
		if ((source_fit_mode==Shapelet_Source) or (inversion_method==DENSE)) {
			// even if using a pixellated source, if inversion_method is set to DENSE, only the dense form of the Lmatrix has been convolved with the PSF, so this form must be used
			Lmatptr = (Lmatrix_dense.pointer())[i];
			tempsrcptr = source_pixel_vector;
			while (tempsrcptr != tempsrc_end) {
				temp_img += (*(Lmatptr++))*(*(tempsrcptr++));
			}
		} else {
			for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
				temp_img += Lmatrix[j]*source_pixel_vector[Lmatrix_index[j]];
			}
		}
		// NOTE: this chisq does not include foreground mask pixels that lie outside the primary mask, since those pixels don't contribute to determining the regularization
		Ed_times_two += SQR(temp_img - img_minus_sbprofile[i])/covariance;
	}
	//cout << "Ed_times_two=" << Ed_times_two << endl;
	double loglike_reg = calculate_regularization_prior_term(true);
	chisq = Ed_times_two + loglike_reg;
	logdet = (use_covariance_matrix) ? Gmatrix_logdet : Fmatrix_logdet;
	chisq += logdet;

	if (chisq < regopt_chisqmin) {
		regopt_chisqmin = chisq;
		for (i=0; i < source_n_amps; i++) source_pixel_vector_minchisq[i] = source_pixel_vector[i];
		regopt_logdet = logdet;
	}
	return chisq;
}
*/

double QLens::brents_min_method(double (QLens::*func)(const double), const double ax, const double bx, const double tol, const bool verbal)
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
	for (int iter=0; iter < max_regopt_iterations; iter++)
	{
		xmid=0.5*(a+b);
		tol2 = 2.0 * ((tol1=tol*abs(x)) + ZEPS);
		if (abs(x-xmid) <= (tol2-0.5*(b-a))) {
			if ((verbal) and (mpi_id==0)) {
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
	if ((verbal) and (mpi_id==0)) {
		warn("Brent's Method reached maximum number of iterations for optimizing regparam");
	}
	return x;
}

/*
bool QLens::Cholesky_dcmp(double** a, double &logdet, int n)
{
	int i,j,k;

	logdet = log(abs(a[0][0]));
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
			warn("matrix is not positive-definite (row %i)",i);
			status = false;
		}
		logdet += log(abs(a[i][i]));
		a[i][i] = sqrt(abs(a[i][i]));
		for (j=i+1; j < n; j++) a[j][i] /= a[i][i];
	}
	// switch to upper triangular (annoying, shouldn't have to!)
	for (i=0; i < n; i++) {
		for (j=0; j < i; j++) {
			a[j][i] = a[i][j];
			a[i][j] = 0;
		}
	}
	
	return status;
}
*/

/*
// Not sure why this upper version doesn't work...trying to start from bottom-right and go upwards from there
bool QLens::Cholesky_dcmp_upper(double** a, double &logdet, int n)
{
	int i,j,k;

	logdet = log(abs(a[n-1][n-1]));
	a[n-1][n-1] = sqrt(a[n-1][n-1]);
	for (j=0; j < n-1; j++) a[j][n-1] /= a[n-1][n-1];

	bool status = true;
	for (i=n-2; i >= 0; i--) {
		//#pragma omp parallel for private(j,k) schedule(static)
		for (j=i; j >= 0; j--) {
			for (k=n-1; k >= i; k--) {
				a[j][i] -= a[i][k]*a[j][k];
			}
		}
		if (a[i][i] < 0) {
			warn("matrix is not positive-definite (row %i)",i);
			status = false;
		}
		logdet += log(abs(a[i][i]));
		a[i][i] = sqrt(abs(a[i][i]));
		for (j=0; j < i; j++) a[j][i] /= a[i][i];
	}
	
	return status;
}
*/

/*
bool QLens::Cholesky_dcmp_upper(double** a, double &logdet, int n)
{
	int i,j,k;

	logdet = log(abs(a[0][0]));
	a[0][0] = sqrt(a[0][0]);
	for (j=1; j < n; j++) a[0][j] /= a[0][0];

	bool status = true;
	for (i=1; i < n; i++) {
		#pragma omp parallel for private(j,k) schedule(static)
		for (j=i; j < n; j++) {
			for (k=0; k < i; k++) {
				a[i][j] -= a[k][i]*a[k][j];
			}
		}
		if (a[i][i] < 0) {
			warn("matrix is not positive-definite (row %i)",i);
			status = false;
		}
		logdet += log(abs(a[i][i]));
		a[i][i] = sqrt(abs(a[i][i]));
		for (j=i+1; j < n; j++) a[i][j] /= a[i][i];
	}
	
	return status;
}
*/

/*
bool QLens::Cholesky_dcmp_upper_packed(double* a, double &logdet, int n)
{
	int i,j,k;

	int *indx = new int[n];
	indx[0] = 0;
	for (j=1; j < n; j++) indx[j] = indx[j-1] + n-j+1;

	a[0] = sqrt(a[0]);
	for (j=1; j < n; j++) a[j] /= a[0];

	bool status = true;
	double *aptr1, *aptr2, *aptr3;
	for (i=1; i < n; i++) {
		#pragma omp parallel for private(j,k,aptr1,aptr2,aptr3) schedule(static)
		for (j=i; j < n; j++) {
			aptr1 = a+indx[i];
			aptr2 = a+indx[j];
			aptr3 = aptr2+i;
			for (k=0; k < i; k++) {
				*(aptr3) -= (*(aptr1++))*(*(aptr2++));
			}
		}
		aptr1 = a+indx[i]+i;
		if ((*aptr1) < 0) {
			warn("matrix is not positive-definite (row %i)",i);
			status = false;
		}
		(*aptr1) = sqrt(abs((*aptr1)));
		for (j=0; j < i; j++) a[indx[j]+i-j] /= (*aptr1);
	}
	delete[] indx;
	
	return status;
}
*/

// This does a lower triangular Cholesky decomposition
bool QLens::Cholesky_dcmp_packed(double* a, int n)
{
	int i,j,k;

	int *indx = new int[n];
	indx[0] = 0;
	for (j=0; j < n; j++) if (j > 0) indx[j] = indx[j-1] + j;

	a[0] = sqrt(a[0]);
	for (j=1; j < n; j++) a[indx[j]] /= a[0];

	bool status = true;
	double *aptr1, *aptr2, *aptr3;
	for (i=1; i < n; i++) {
		#pragma omp parallel for private(j,k,aptr1,aptr2,aptr3) schedule(static)
		for (j=i; j < n; j++) {
			aptr1 = a+indx[i];
			aptr2 = a+indx[j];
			aptr3 = aptr2+i;
			for (k=0; k < i; k++) {
				*(aptr3) -= (*(aptr1++))*(*(aptr2++));
			}
		}
		aptr1 = a+indx[i]+i;
		if ((*aptr1) < 0) {
			warn("matrix is not positive-definite (row %i)",i);
			status = false;
		}
		(*aptr1) = sqrt(abs((*aptr1)));
		for (j=i+1; j < n; j++) a[indx[j]+i] /= (*aptr1);
	}
	delete[] indx;
	
	return status;
}

/*
void QLens::Cholesky_invert_lower(double** a, const int n)
{
	double sum;
	int i,j,k;
	for (i=0; i < n; i++) {
		a[i][i] = 1.0/a[i][i];
		for (j=i+1; j < n; j++) {
			sum=0.0;
			for (k=i; k < j; k++) sum -= a[j][k]*a[k][i];
			a[j][i]=sum/a[j][j];
		}
	}
}
*/

/*
void QLens::Cholesky_invert_upper(double** a, const int n)
{
	double sum;
	int i,j,k;
	for (i=0; i < n; i++) {
		a[i][i] = 1.0/a[i][i];
		for (j=i+1; j < n; j++) {
			sum=0.0;
			for (k=i; k < j; k++) sum -= a[k][j]*a[i][k];
			a[i][j]=sum/a[j][j];
		}
	}
}
*/

void QLens::Cholesky_invert_upper_packed(double* a, const int n)
{
	double sum;
	int i,j,k;
	int indx=0, indx2;
	// Replace indx, indx2 with pointers, as in upper_triangular_syrk
	for (i=0; i < n; i++) {
		a[indx] = 1.0/a[indx];
		for (j=i+1; j < n; j++) {
			sum=0.0;
			indx2=indx;
			for (k=i; k < j; k++) {
				sum -= a[indx2+j-k]*a[indx+k-i];
				indx2 += n-k;
			}
			a[indx+j-i]=sum/a[indx2];
		}
		indx += n-i;
	}
}

/*
void QLens::upper_triangular_syrk(double* a, const int n)
{
	double sum;
	int i,j,k;
	int indx=0, indx2;
	for (i=0; i < n; i++) {
		indx2=indx;
		for (j=i; j < n; j++) {
			sum=0.0;
			for (k=j; k < n; k++) {
				sum += a[indx+k-i]*a[indx2+k-j];
			}
			a[indx+j-i] = sum;
			indx2 += n-j;
		}
		indx += n-i;
	}
}
*/	

void QLens::upper_triangular_syrk(double* a, const int n)
{
	double sum;
	int i,j,k;
	double *aptr, *aptr2;
	aptr=aptr2=a;
	for (i=0; i < n; i++) {
		aptr2 = aptr;
		for (j=i; j < n; j++) {
			sum=0.0;
			for (k=j; k < n; k++) {
				sum += (*(aptr++))*(*(aptr2++));
			}
			*a = sum;
			aptr = ++a;
		}
	}
}

// This is for the determinant from the lower triangular version of the decomposition
void QLens::Cholesky_logdet_lower_packed(double* a, double &logdet, int n)
{
	logdet = 0;
	int indx = 0;
	for (int i=0; i < n; i++) {
		logdet += log(abs(a[indx]));
		indx += i+2;
	}
	logdet *= 2;
}

/*
// This function is kept for reference so you can convert to an upper triangular version (not done yet)...after that you can get rid of this
void QLens::Cholesky_solve(double** a, double* b, double* x, int n)
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
*/

// This is the lower triangular version
void QLens::Cholesky_solve_lower_packed(double* a, double* b, double* x, int n)
{
	int i,k;
	double sum;
	int *indx = new int[n];
	indx[0] = 0;
	for (i=0; i < n; i++) {
		if (i > 0) indx[i] = indx[i-1] + i;
		for (sum=b[i], k=i-1; k >= 0; k--) sum -= a[indx[i]+k]*x[k];
		x[i] = sum / a[indx[i]+i];
	}
	for (i=n-1; i >= 0; i--) {
		for (sum=x[i], k=i+1; k < n; k++) sum -= a[indx[k]+i]*x[k];
		x[i] = sum / a[indx[i]+i];
	}	 
	delete[] indx;
}

// This is for the determinant from the upper triangular version of the decomposition
void QLens::Cholesky_logdet_packed(double* a, double &logdet, int n)
{
	logdet = 0;
	int indx = 0;
	for (int i=0; i < n; i++) {
		logdet += log(abs(a[indx]));
		indx += n-i;
	}
	logdet *= 2;
}

// This is for the determinant from the upper triangular version of the decomposition
void QLens::LU_logdet_stacked(double* a, double &logdet, int n)
{
	logdet = 0;
	int indx = 0;
	for (int i=0; i < n; i++) {
		logdet += log(abs(a[indx]));
		indx += n+1;
	}
}



/*
// This is an attempt at an upper triangular version of Cholesky solve (not working yet), but you need to make the Cholesky decomp upper triangular as well...fix later
void QLens::Cholesky_solve_packed(double* a, double* b, double* x, int n)
{
	int i,k;
	double sum;
	int *indx = new int[n];
	cout << "HI0" << endl;
	indx[n-1] = (n*(n+1))/2 - 1;
	cout << "HI1" << endl;
	for (i=n-1; i >= 0; i--) {
		if (i < n-1) indx[i] = indx[i+1] - n + i;
		for (sum=b[i], k=1; k < n-1-i; k++) sum -= a[indx[i]+k]*x[k+i];
		x[i] = sum / a[indx[i]];
		cout << "Setting y[" << i << "]" << endl;
	}
	cout << "HI2" << endl;
	for (i=0; i < n; i++) {
		for (sum=x[i], k=i-1; k >= 0; k--) sum -= a[indx[k]+i-k]*x[k];
		x[i] = sum / a[indx[i]];
		cout << "Setting x[" << i << "]" << endl;
	}	 
	cout << "HI3" << endl;
	delete[] indx;
}
*/

void QLens::repack_matrix_lower(dvector& packed_matrix)
{
	// At the moment, the native Cholesky decomposition code does a lower triangular decomposition; since Fmatrix/Rmatrix stores the upper triangular part,
	// we have to switch Fmatrix to a lower triangular version here
	double **Fmat = new double*[source_n_amps];
	int i,j,k;
	for (i=0; i < source_n_amps; i++) {
		Fmat[i] = new double[i+1];
	}
	for (k=0,j=0; j < source_n_amps; j++) {
		for (i=j; i < source_n_amps; i++) {
			Fmat[i][j] = packed_matrix[k++];
		}
	}
	for (k=0,i=0; i < source_n_amps; i++) {
		for (j=0; j <= i; j++) {
			packed_matrix[k++] = Fmat[i][j];
		}
		delete[] Fmat[i];
	}
	delete[] Fmat;
}

void QLens::repack_matrix_upper(dvector& packed_matrix)
{
	// At the moment, the native Cholesky decomposition code does a lower triangular decomposition; since Fmatrix/Rmatrix stores the upper triangular part,
	// we have to switch Fmatrix to a lower triangular version here
	double **Fmat = new double*[source_n_amps];
	int i,j,k;
	for (i=0; i < source_n_amps; i++) {
		Fmat[i] = new double[i+1];
	}
	for (k=0,i=0; i < source_n_amps; i++) {
		for (j=0; j <= i; j++) {
			Fmat[i][j] = packed_matrix[k++];
		}
	}
	for (k=0,j=0; j < source_n_amps; j++) {
		for (i=j; i < source_n_amps; i++) {
			packed_matrix[k++] = Fmat[i][j];
		}
	}
	for (i=0; i < source_n_amps; i++) delete[] Fmat[i];
	delete[] Fmat;
}

void QLens::invert_lens_mapping_dense(const int zsrc_i, bool verbal)
{
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	int i,j;
#ifdef USE_MKL
	if (!use_covariance_matrix) {
		lapack_int status;
		status = LAPACKE_dpptrf(LAPACK_ROW_MAJOR,'U',source_n_amps,Fmatrix_packed.array());
		if (status != 0) warn("Matrix was not invertible and/or positive definite");
		for (int i=0; i < source_n_amps; i++) source_pixel_vector[i] = Dvector[i];
		LAPACKE_dpptrs(LAPACK_ROW_MAJOR,'U',source_n_amps,1,Fmatrix_packed.array(),source_pixel_vector,1);
		Cholesky_logdet_packed(Fmatrix_packed.array(),Fmatrix_log_determinant,source_n_amps);
	} else {
		lapack_int status;
		int *ipiv = new int[source_npixels];
		status = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, source_n_amps, source_n_amps, Gmatrix_stacked.array(), source_n_amps, ipiv);
		if (status != 0) warn("Matrix was not invertible");
		//LAPACKE_dsptrf(LAPACK_ROW_MAJOR,'U', source_n_amps, Gmatrix_packed.array(), ipiv);
		for (int i=0; i < source_n_amps; i++) source_pixel_vector[i] = Dvector_cov[i];
		//LAPACKE_dsptrs(LAPACK_ROW_MAJOR,'U',source_n_amps,1,Gmatrix_packed.array(),ipiv,source_pixel_vector,1);
		LAPACKE_dgetrs (LAPACK_ROW_MAJOR, 'N', source_n_amps, 1, Gmatrix_stacked.array(), source_n_amps, ipiv, source_pixel_vector, 1);
		LU_logdet_stacked(Gmatrix_stacked.array(),Gmatrix_log_determinant,source_n_amps);
		delete[] ipiv;
		//cout << "DETERMINANTS: " << Fmatrix_log_determinant << " " << Gmatrix_log_determinant << " " << (Gmatrix_log_determinant+Rmatrix_log_determinant) << endl;
	}
#else
	if (use_covariance_matrix) {
		die("Compiling with MKL is currently required for covariance kernel regularization");
	}
	// At the moment, the native Cholesky decomposition code does a lower triangular decomposition; since Fmatrix/Rmatrix stores the upper triangular part,
	// we have to switch Fmatrix to a lower triangular version here
	repack_matrix_lower(Fmatrix_packed);

	bool status = Cholesky_dcmp_packed(Fmatrix_packed.array(),source_n_amps);
	if (!status) die("Cholesky decomposition failed");
	Cholesky_solve_lower_packed(Fmatrix_packed.array(),Dvector,source_pixel_vector,source_n_amps);
	Cholesky_logdet_lower_packed(Fmatrix_packed.array(),Fmatrix_log_determinant,source_n_amps);
#endif
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for inverting Fmatrix: " << wtime << endl;
		wtime0 = omp_get_wtime();
	}
#endif
	update_source_amplitudes(zsrc_i,verbal);
}

void QLens::invert_lens_mapping_CG_method(const int zsrc_i, bool verbal)
{
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

#ifdef USE_MPI
	MPI_Barrier(sub_comm);
#endif

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	int i,j,k;
	double *temp = new double[source_n_amps];
	// it would be prettier to just pass the MPI communicator in, and have CG_sparse figure out the rank and # of processes internally--implement this later
	CG_sparse cg_method(Fmatrix,Fmatrix_index,1e-4,100000,inversion_nthreads,group_np,group_id);
#ifdef USE_MPI
	cg_method.set_MPI_comm(&sub_comm);
#endif
	for (int i=0; i < source_n_amps; i++) temp[i] = 0;
	if ((regularization_method != None) and (source_npixels > 0))
		cg_method.set_determinant_mode(true);
	else cg_method.set_determinant_mode(false);
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for setting up CG method: " << wtime << endl;
		wtime0 = omp_get_wtime();
	}
#endif
	cg_method.solve(Dvector,temp);

	for (int i=0; i < source_n_amps; i++) {
		if ((background_pixel_noise==0) and (temp[i] < 0)) temp[i] = 0; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
		source_pixel_vector[i] = temp[i];
	}

	if ((regularization_method != None) and (source_npixels > 0)) {
		cg_method.get_log_determinant(Fmatrix_log_determinant);
		if ((mpi_id==0) and (verbal)) cout << "log determinant = " << Fmatrix_log_determinant << endl;
		CG_sparse cg_det(Rmatrix,Rmatrix_index,3e-4,100000,inversion_nthreads,group_np,group_id);
#ifdef USE_MPI
		cg_det.set_MPI_comm(&sub_comm);
#endif
		Rmatrix_log_determinant = cg_det.calculate_log_determinant();
		if ((mpi_id==0) and (verbal)) cout << "Rmatrix log determinant = " << Rmatrix_log_determinant << endl;
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for inverting Fmatrix: " << wtime << endl;
	}
#endif

	int iterations;
	double error;
	cg_method.get_error(iterations,error);
	if ((mpi_id==0) and (verbal)) cout << iterations << " iterations, error=" << error << endl << endl;

	delete[] temp;
	update_source_amplitudes(zsrc_i,verbal);
#ifdef USE_MPI
	MPI_Comm_free(&sub_comm);
#endif
}

void QLens::invert_lens_mapping_UMFPACK(const int zsrc_i, bool verbal, bool use_copy)
{
#ifndef USE_UMFPACK
	die("QLens requires compilation with UMFPACK for factorization");
#else
	bool calculate_determinant = false;
	int default_nthreads=1;

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	int i,j;
	double *Fmatptr = (use_copy==true) ? Fmatrix_copy : Fmatrix;

   double *null = (double *) NULL ;
	double *temp = new double[source_n_amps];
   void *Symbolic, *Numeric ;
	double Control [UMFPACK_CONTROL];
	double Info [UMFPACK_INFO];
    umfpack_di_defaults (Control) ;
	 Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;

	int Fmatrix_nonzero_elements = Fmatrix_index[source_n_amps]-1;
	int Fmatrix_offdiags = Fmatrix_index[source_n_amps]-1-source_n_amps;
	int Fmatrix_unsymmetric_nonzero_elements = source_n_amps + 2*Fmatrix_offdiags;
	if (Fmatrix_nonzero_elements==0) {
		cout << "nsource_pixels=" << source_n_amps << endl;
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

	int *Fmatrix_unsymmetric_cols = new int[source_n_amps+1];
	int *Fmatrix_unsymmetric_indices = new int[Fmatrix_unsymmetric_nonzero_elements];
	double *Fmatrix_unsymmetric = new double[Fmatrix_unsymmetric_nonzero_elements];

	int indx=0;
	Fmatrix_unsymmetric_cols[0] = 0;
	for (i=0; i < source_n_amps; i++) {
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

	//cout << "Dvector: " << endl;
	//for (i=0; i < source_n_amps; i++) {
		//cout << Dvector[i] << " ";
	//}
	//cout << endl;

	for (i=0; i < source_n_amps; i++) {
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
   status = umfpack_di_symbolic(source_n_amps, source_n_amps, Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, &Symbolic, Control, Info);
	if (status < 0) {
		umfpack_di_report_info (Control, Info) ;
		umfpack_di_report_status (Control, status) ;
		die("Error inputting matrix");
	}
   status = umfpack_di_numeric(Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, Symbolic, &Numeric, Control, Info);
   umfpack_di_free_symbolic(&Symbolic);

   status = umfpack_di_solve(UMFPACK_A, Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, temp, Dvector, Numeric, Control, Info);

	if ((regularization_method != None) and (source_npixels > 0)) calculate_determinant = true; // specifies to calculate determinant

	for (int i=0; i < source_n_amps; i++) {
		source_pixel_vector[i] = temp[i];
	}

	double mantissa, exponent;
	status = umfpack_di_get_determinant (&mantissa, &exponent, Numeric, Info) ;
	if (status < 0) {
		die("Could not get determinant using UMFPACK");
	}
	umfpack_di_free_numeric(&Numeric);
	Fmatrix_log_determinant = log(mantissa) + exponent*log(10);

	if (calculate_determinant) Rmatrix_determinant_UMFPACK();
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for inverting Fmatrix: " << wtime << endl;
	}
#endif

	delete[] temp;
	delete[] Fmatrix_transpose;
	delete[] Fmatrix_transpose_index;
	delete[] Fmatrix_unsymmetric_cols;
	delete[] Fmatrix_unsymmetric_indices;
	delete[] Fmatrix_unsymmetric;
	update_source_amplitudes(zsrc_i,verbal);
#endif
}

void QLens::invert_lens_mapping_MUMPS(const int zsrc_i, bool verbal, bool use_copy)
{
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

#ifdef USE_MPI
	MPI_Comm this_comm;
	MPI_Comm_create(*my_comm, *my_group, &this_comm);
#endif

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
	omp_set_num_threads(inversion_nthreads);
#endif

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	int i,j;
	double *Fmatptr = (use_copy==true) ? Fmatrix_copy : Fmatrix;

	double *temp = new double[source_n_amps];
	MUMPS_INT Fmatrix_nonzero_elements = Fmatrix_index[source_n_amps]-1;
	if (Fmatrix_nonzero_elements==0) {
		cout << "nsource_pixels=" << source_n_amps << endl;
		die("Fmatrix has zero size");
	}
	MUMPS_INT *irn = new MUMPS_INT[Fmatrix_nonzero_elements];
	MUMPS_INT *jcn = new MUMPS_INT[Fmatrix_nonzero_elements];
	double *Fmatrix_elements = new double[Fmatrix_nonzero_elements];
	for (i=0; i < source_n_amps; i++) {
		Fmatrix_elements[i] = Fmatptr[i];
		irn[i] = i+1;
		jcn[i] = i+1;
		temp[i] = Dvector[i];
	}
	int indx=source_n_amps;
	for (i=0; i < source_n_amps; i++) {
		for (j=Fmatrix_index[i]; j < Fmatrix_index[i+1]; j++) {
			Fmatrix_elements[indx] = Fmatptr[j];
			irn[indx] = i+1;
			jcn[indx] = Fmatrix_index[j]+1;
			indx++;
		}
	}

#ifdef USE_MPI
	if (use_mumps_subcomm) {
		mumps_solver->comm_fortran=(MUMPS_INT) MPI_Comm_c2f(sub_comm);
	} else {
		mumps_solver->comm_fortran=(MUMPS_INT) MPI_Comm_c2f(this_comm);
	}
#endif
	mumps_solver->job = JOB_INIT; // initialize
	mumps_solver->sym = 2; // specifies that matrix is symmetric and positive-definite
	//cout << "ICNTL = " << mumps_solver->icntl[13] << endl;
	dmumps_c(mumps_solver);
	mumps_solver->n = source_n_amps; mumps_solver->nz = Fmatrix_nonzero_elements; mumps_solver->irn=irn; mumps_solver->jcn=jcn;
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
	if ((regularization_method != None) and (source_npixels > 0)) mumps_solver->icntl[32]=1; // specifies to calculate determinant
	else mumps_solver->icntl[32] = 0;
	if (parallel_mumps) {
		mumps_solver->icntl[27]=2; // parallel analysis phase
		mumps_solver->icntl[28]=2; // parallel analysis phase
	}
	mumps_solver->job = 6; // specifies to factorize and solve linear equation
#ifdef USE_MPI
	MPI_Barrier(sub_comm);
#endif
	dmumps_c(mumps_solver);
#ifdef USE_MPI
	if (use_mumps_subcomm) {
		MPI_Bcast(temp,source_n_amps,MPI_DOUBLE,0,sub_comm);
		MPI_Barrier(sub_comm);
	}
#endif

	if (mumps_solver->info[0] < 0) {
		if (mumps_solver->info[0]==-10) die("Singular matrix, cannot invert");
		else warn("Error occurred during matrix inversion; MUMPS error code %i (source_n_amps=%i)",mumps_solver->info[0],source_n_amps);
	}

	for (int i=0; i < source_n_amps; i++) {
		if ((background_pixel_noise==0) and (temp[i] < 0)) temp[i] = 0; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
		source_pixel_vector[i] = temp[i];
	}

	if ((regularization_method != None) and (source_npixels > 0))
	{
		Fmatrix_log_determinant = log(mumps_solver->rinfog[11]) + mumps_solver->infog[33]*log(2);
		//cout << "Fmatrix log determinant = " << Fmatrix_log_determinant << endl;
		if ((mpi_id==0) and (verbal)) cout << "log determinant = " << Fmatrix_log_determinant << endl;

		mumps_solver->job=JOB_END; dmumps_c(mumps_solver); //Terminate instance

		MUMPS_INT Rmatrix_nonzero_elements = Rmatrix_index[source_n_amps]-1;
		MUMPS_INT *irn_reg = new MUMPS_INT[Rmatrix_nonzero_elements];
		MUMPS_INT *jcn_reg = new MUMPS_INT[Rmatrix_nonzero_elements];
		double *Rmatrix_elements = new double[Rmatrix_nonzero_elements];
		for (i=0; i < source_n_amps; i++) {
			Rmatrix_elements[i] = Rmatrix[i];
			irn_reg[i] = i+1;
			jcn_reg[i] = i+1;
		}
		indx=source_n_amps;
		for (i=0; i < source_n_amps; i++) {
			//cout << "Row " << i << ": diag=" << Rmatrix[i] << endl;
			//for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
				//cout << Rmatrix_index[j] << " ";
			//}
			//cout << endl;
			for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
				//cout << Rmatrix[j] << " ";
				Rmatrix_elements[indx] = Rmatrix[j];
				irn_reg[indx] = i+1;
				jcn_reg[indx] = Rmatrix_index[j]+1;
				indx++;
			}
		}

		mumps_solver->job=JOB_INIT; mumps_solver->sym=2;
		dmumps_c(mumps_solver);
		mumps_solver->n = source_n_amps; mumps_solver->nz = Rmatrix_nonzero_elements; mumps_solver->irn=irn_reg; mumps_solver->jcn=jcn_reg;
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
		if ((mpi_id==0) and (verbal)) cout << "Rmatrix log determinant = " << Rmatrix_log_determinant << " " << mumps_solver->rinfog[11] << " " << mumps_solver->infog[33] << endl;

		delete[] irn_reg;
		delete[] jcn_reg;
		delete[] Rmatrix_elements;
	}
	mumps_solver->job=JOB_END;
	dmumps_c(mumps_solver); //Terminate instance

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for inverting Fmatrix: " << wtime << endl;
	}
#endif

#ifdef USE_OPENMP
	omp_set_num_threads(default_nthreads);
#endif

	delete[] temp;
	delete[] irn;
	delete[] jcn;
	delete[] Fmatrix_elements;
	update_source_amplitudes(zsrc_i,verbal);
#endif
#ifdef USE_MPI
	MPI_Comm_free(&sub_comm);
	MPI_Comm_free(&this_comm);
#endif

}

void QLens::update_source_amplitudes(const int zsrc_i, const bool verbal)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	SourcePixelGrid *cartesian_srcgrid = image_pixel_grid->cartesian_srcgrid;
	int i,j,index=0;
	if ((source_fit_mode==Delaunay_Source) and (image_pixel_grid->delaunay_srcgrid != NULL)) image_pixel_grid->delaunay_srcgrid->update_surface_brightness(index);
	else if (source_fit_mode==Cartesian_Source) cartesian_srcgrid->update_surface_brightness(index);
	else if (source_fit_mode==Shapelet_Source) {
		double* srcpix = source_pixel_vector;
		for (i=0; i < n_sb; i++) {
			if ((sb_list[i]->sbtype==SHAPELET) and ((zsrc_i < 0) or (sbprofile_redshift_idx[i]==zsrc_i))) {
				sb_list[i]->update_amplitudes(srcpix);
			}
		}
	}
	if (include_imgfluxes_in_inversion) {
		index = source_npixels;
		for (j=0; j < point_imgs.size(); j++) {
			for (i=0; i < point_imgs[j].size(); i++) {
				point_imgs[j][i].flux = source_pixel_vector[index++];
				if ((mpi_id==0) and (verbal)) cout << "srcpt " << j << " (img " << i << "): flux=" << point_imgs[j][i].flux << endl;
			}
		}
	} else if (include_srcflux_in_inversion) {
		index = source_npixels;
		for (j=0; j < point_imgs.size(); j++) {
			source_flux = source_pixel_vector[index++]; // need to have more than one srcflux parameter!!!!!!!! UPGRADE THIS
			if ((mpi_id==0) and (verbal)) cout << "srcpt " << j << ": srcflux=" << source_flux << endl;
		}
	}
}


void QLens::Rmatrix_determinant_UMFPACK()
{
#ifndef USE_UMFPACK
	die("QLens requires compilation with UMFPACK (or MUMPS) for determinants of sparse matrices");
#else
	double mantissa, exponent;
	int i,j,status;
   void *Symbolic, *Numeric;
	double Control [UMFPACK_CONTROL];
	double Info [UMFPACK_INFO];
   umfpack_di_defaults (Control);
	Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
	int Rmatrix_nonzero_elements = Rmatrix_index[source_npixels]-1;
	int Rmatrix_n_offdiags = Rmatrix_index[source_npixels]-1-source_npixels;
	int Rmatrix_unsymmetric_nonzero_elements = source_npixels + 2*Rmatrix_n_offdiags;
	if (Rmatrix_nonzero_elements==0) {
		cout << "nsource_pixels=" << source_npixels << endl;
		die("Rmatrix has zero size");
	}

	int k,jl,jm,jp,ju,m,n2,noff,inc,iv;
	double v;

	// Now we construct the transpose of Rmatrix so we can cast it into "unsymmetric" format for UMFPACK (by including offdiagonals on either side of diagonal elements)
	double *Rmatrix_transpose = new double[Rmatrix_nonzero_elements+1];
	int *Rmatrix_transpose_index = new int[Rmatrix_nonzero_elements+1];

	n2=Rmatrix_index[0];
	for (j=0; j < n2-1; j++) Rmatrix_transpose[j] = Rmatrix[j];
	int n_offdiag = Rmatrix_index[n2-1] - Rmatrix_index[0];
	int *offdiag_indx = new int[n_offdiag];
	int *offdiag_indx_transpose = new int[n_offdiag];
	for (i=0; i < n_offdiag; i++) offdiag_indx[i] = Rmatrix_index[n2+i];
	indexx(offdiag_indx,offdiag_indx_transpose,n_offdiag);
	for (j=n2, k=0; j < Rmatrix_index[n2-1]; j++, k++) {
		Rmatrix_transpose_index[j] = offdiag_indx_transpose[k];
	}
	jp=0;
	for (k=Rmatrix_index[0]; k < Rmatrix_index[n2-1]; k++) {
		m = Rmatrix_transpose_index[k] + n2;
		Rmatrix_transpose[k] = Rmatrix[m];
		for (j=jp; j < Rmatrix_index[m]+1; j++)
			Rmatrix_transpose_index[j]=k;
		jp = Rmatrix_index[m] + 1;
		jl=0;
		ju=n2-1;
		while (ju-jl > 1) {
			jm = (ju+jl)/2;
			if (Rmatrix_index[jm] > m) ju=jm; else jl=jm;
		}
		Rmatrix_transpose_index[k]=jl;
	}
	for (j=jp; j < n2; j++) Rmatrix_transpose_index[j] = Rmatrix_index[n2-1];
	for (j=0; j < n2-1; j++) {
		jl = Rmatrix_transpose_index[j+1] - Rmatrix_transpose_index[j];
		noff=Rmatrix_transpose_index[j];
		inc=1;
		do {
			inc *= 3;
			inc++;
		} while (inc <= jl);
		do {
			inc /= 3;
			for (k=noff+inc; k < noff+jl; k++) {
				iv = Rmatrix_transpose_index[k];
				v = Rmatrix_transpose[k];
				m=k;
				while (Rmatrix_transpose_index[m-inc] > iv) {
					Rmatrix_transpose_index[m] = Rmatrix_transpose_index[m-inc];
					Rmatrix_transpose[m] = Rmatrix_transpose[m-inc];
					m -= inc;
					if (m-noff+1 <= inc) break;
				}
				Rmatrix_transpose_index[m] = iv;
				Rmatrix_transpose[m] = v;
			}
		} while (inc > 1);
	}
	delete[] offdiag_indx;
	delete[] offdiag_indx_transpose;

	int *Rmatrix_unsymmetric_cols = new int[source_npixels+1];
	int *Rmatrix_unsymmetric_indices = new int[Rmatrix_unsymmetric_nonzero_elements];
	double *Rmatrix_unsymmetric = new double[Rmatrix_unsymmetric_nonzero_elements];
	int indx=0;
	Rmatrix_unsymmetric_cols[0] = 0;
	for (i=0; i < source_npixels; i++) {
		for (j=Rmatrix_transpose_index[i]; j < Rmatrix_transpose_index[i+1]; j++) {
			Rmatrix_unsymmetric[indx] = Rmatrix_transpose[j];
			Rmatrix_unsymmetric_indices[indx] = Rmatrix_transpose_index[j];
			indx++;
		}
		Rmatrix_unsymmetric_indices[indx] = i;
		Rmatrix_unsymmetric[indx] = Rmatrix[i];
		indx++;
		for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
			Rmatrix_unsymmetric[indx] = Rmatrix[j];
			//cout << "Row " << i << ", column " << Rmatrix_index[j] << ": " << Rmatrix[j] << " " << Rmatrix_unsymmetric[indx] << " (element " << indx << ")" << endl;
			Rmatrix_unsymmetric_indices[indx] = Rmatrix_index[j];
			indx++;
		}
		Rmatrix_unsymmetric_cols[i+1] = indx;
	}

	for (i=0; i < source_npixels; i++) {
		sort(Rmatrix_unsymmetric_cols[i+1]-Rmatrix_unsymmetric_cols[i],Rmatrix_unsymmetric_indices+Rmatrix_unsymmetric_cols[i],Rmatrix_unsymmetric+Rmatrix_unsymmetric_cols[i]);
		//cout << "Row " << i << ": " << endl;
		//cout << Rmatrix_unsymmetric_cols[i] << " ";
		//for (j=Rmatrix_unsymmetric_cols[i]; j < Rmatrix_unsymmetric_cols[i+1]; j++) {
			//cout << Rmatrix_unsymmetric_indices[j] << " ";
		//}
		//cout << endl;
		//for (j=Rmatrix_unsymmetric_cols[i]; j < Rmatrix_unsymmetric_cols[i+1]; j++) {
			//cout << Rmatrix_unsymmetric[j] << " ";
		//}
		//cout << endl;
	}
	//cout << endl;

	if (indx != Rmatrix_unsymmetric_nonzero_elements) die("WTF! Wrong number of nonzero elements");

	status = umfpack_di_symbolic(source_npixels, source_npixels, Rmatrix_unsymmetric_cols, Rmatrix_unsymmetric_indices, Rmatrix_unsymmetric, &Symbolic, Control, Info);
	if (status < 0) {
		umfpack_di_report_info (Control, Info) ;
		umfpack_di_report_status (Control, status) ;
		die("Error inputting matrix");
	}
	status = umfpack_di_numeric(Rmatrix_unsymmetric_cols, Rmatrix_unsymmetric_indices, Rmatrix_unsymmetric, Symbolic, &Numeric, Control, Info);
	if (status < 0) {
		umfpack_di_report_info (Control, Info) ;
		umfpack_di_report_status (Control, status) ;
		die("Error inputting matrix");
	}
	umfpack_di_free_symbolic(&Symbolic);

	status = umfpack_di_get_determinant (&mantissa, &exponent, Numeric, Info) ;
	//cout << "Rmatrix mantissa=" << mantissa << ", exponent=" << exponent << endl;
	if (status < 0) {
		die("Could not calculate determinant");
	}
	Rmatrix_log_determinant = log(mantissa) + exponent*log(10);
	//cout << "Rmatrix_logdet=" << Rmatrix_log_determinant << endl;
	delete[] Rmatrix_transpose;
	delete[] Rmatrix_transpose_index;
	delete[] Rmatrix_unsymmetric_cols;
	delete[] Rmatrix_unsymmetric_indices;
	delete[] Rmatrix_unsymmetric;
	umfpack_di_free_numeric(&Numeric);
#endif
}

void QLens::Rmatrix_determinant_MUMPS()
{
#ifndef USE_MUMPS
	die("QLens requires compilation with UMFPACK (or MUMPS) for determinants of sparse matrices");
#else
	int i,j;
	MUMPS_INT Rmatrix_nonzero_elements = Rmatrix_index[source_npixels]-1;
	MUMPS_INT *irn_reg = new MUMPS_INT[Rmatrix_nonzero_elements];
	MUMPS_INT *jcn_reg = new MUMPS_INT[Rmatrix_nonzero_elements];
	double *Rmatrix_elements = new double[Rmatrix_nonzero_elements];
	for (i=0; i < source_npixels; i++) {
		Rmatrix_elements[i] = Rmatrix[i];
		irn_reg[i] = i+1;
		jcn_reg[i] = i+1;
	}
	int indx=source_npixels;
	for (i=0; i < source_npixels; i++) {
		//cout << "Row " << i << ": diag=" << Rmatrix[i] << endl;
		//for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
			//cout << Rmatrix_index[j] << " ";
		//}
		//cout << endl;
		for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
			//cout << Rmatrix[j] << " ";
			Rmatrix_elements[indx] = Rmatrix[j];
			irn_reg[indx] = i+1;
			jcn_reg[indx] = Rmatrix_index[j]+1;
			indx++;
		}
	}

	mumps_solver->job=JOB_INIT; mumps_solver->sym=2;
	dmumps_c(mumps_solver);
	mumps_solver->n = source_npixels; mumps_solver->nz = Rmatrix_nonzero_elements; mumps_solver->irn=irn_reg; mumps_solver->jcn=jcn_reg;
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
	//if (mpi_id==0) cout << "Rmatrix log determinant = " << Rmatrix_log_determinant << " " << mumps_solver->rinfog[11] << " " << mumps_solver->infog[33] << endl;

	delete[] irn_reg;
	delete[] jcn_reg;
	delete[] Rmatrix_elements;
	mumps_solver->job=JOB_END;
	dmumps_c(mumps_solver); //Terminate instance
#endif
}

#define ISWAP(a,b) temp=(a);(a)=(b);(b)=temp;
void QLens::indexx(int* arr, int* indx, int nn)
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

void QLens::Rmatrix_determinant_MKL()
{
#ifndef USE_MKL
	die("QLens requires compilation with MKL (or UMFPACK or MUMPS) for determinants of sparse matrices");
#else
	// MKL should use Pardiso to get the Cholesky decomposition, but for the moment, I will just convert to dense matrix and do it that way
	if (!dense_Rmatrix) convert_Rmatrix_to_dense();
	int ntot = Rmatrix_packed.size();
	if (ntot != (source_npixels*(source_npixels+1)/2)) die("Rmatrix packed does not have correct number of elements");
	double *Rmatrix_packed_copy = new double[ntot];
	for (int i=0; i < ntot; i++) Rmatrix_packed_copy[i] = Rmatrix_packed[i];
   LAPACKE_dpptrf(LAPACK_ROW_MAJOR,'U',source_npixels,Rmatrix_packed_copy); // Cholesky decomposition
	Cholesky_logdet_packed(Rmatrix_packed_copy,Rmatrix_log_determinant,source_npixels);
	delete[] Rmatrix_packed_copy;
#endif
}

void QLens::Rmatrix_determinant_dense()
{
	if (!dense_Rmatrix) convert_Rmatrix_to_dense();
	int ntot = Rmatrix_packed.size();
	if (ntot != (source_npixels*(source_npixels+1)/2)) die("Rmatrix packed does not have correct number of elements");
	dvector Rmatrix_packed_copy(Rmatrix_packed.size()); 
	for (int i=0; i < Rmatrix_packed.size(); i++) {
		Rmatrix_packed_copy[i] = Rmatrix_packed[i];
	}

	repack_matrix_lower(Rmatrix_packed_copy);

	bool status = Cholesky_dcmp_packed(Rmatrix_packed_copy.array(),source_n_amps);
	if (!status) die("Cholesky decomposition failed");
	Cholesky_logdet_lower_packed(Rmatrix_packed_copy.array(),Rmatrix_log_determinant,source_n_amps);
}

void QLens::convert_Rmatrix_to_dense()
{
	int i,j,indx;
	int ntot = source_npixels*(source_npixels+1)/2;
	Rmatrix_packed.input_zero(ntot);
	indx=0;
	for (i=0; i < source_npixels; i++) {
		Rmatrix_packed[indx] = Rmatrix[i];
		//cout << "Rmat: " << Rmatrix[i] << endl;
		for (j=Rmatrix_index[i]; j < Rmatrix_index[i+1]; j++) {
			Rmatrix_packed[indx+Rmatrix_index[j]-i] = Rmatrix[j];
		}
		indx += source_npixels-i;
	}
}

void QLens::clear_sparse_lensing_matrices()
{
	if (Dvector != NULL) delete[] Dvector;
	if (Fmatrix != NULL) delete[] Fmatrix;
	if (Fmatrix_index != NULL) delete[] Fmatrix_index;
	if (Rmatrix != NULL) delete[] Rmatrix;
	if (Rmatrix_index != NULL) delete[] Rmatrix_index;
	Dvector = NULL;
	Fmatrix = NULL;
	Fmatrix_index = NULL;
	Rmatrix = NULL;
	Rmatrix_index = NULL;
}

void QLens::calculate_image_pixel_surface_brightness()
{
	int img_index_j;
	int i,j,k;

	//cout << "SPARSE SB:" << endl;
	//for (j=0; j < source_n_amps; j++) {
		//cout << source_pixel_vector[j] << " ";
	//}
	//cout << endl;


	for (int img_index=0; img_index < image_npixels; img_index++) {
		image_surface_brightness[img_index] = 0;
		for (img_index_j=image_pixel_location_Lmatrix[img_index]; img_index_j < image_pixel_location_Lmatrix[img_index+1]; img_index_j++) {
			image_surface_brightness[img_index] += Lmatrix[img_index_j]*source_pixel_vector[Lmatrix_index[img_index_j]];
		}
		//if (image_surface_brightness[i] < 0) image_surface_brightness[i] = 0;
	}

	/*
	if (calculate_foreground) {
		bool at_least_one_foreground_src = false;

		for (k=0; k < n_sb; k++) {
			if ((!sb_list[k]->is_lensed) and (zsrc_i==0)) {
				at_least_one_foreground_src = true;
				break;
			}
		}
		if ((at_least_one_foreground_src) and (!ignore_foreground_in_chisq)) {
			if (image_pixel_grids[0]->active_image_pixel_i_fgmask==NULL) die("Need to assign foreground pixel mappings before calculating foreground surface brightness");
j
			calculate_foreground_pixel_surface_brightness();
			//add_foreground_to_image_pixel_vector();
			store_foreground_pixel_surface_brightness(); // this stores it in image_pixel_grid->foreground_surface_brightness[i][j]
		} else {
			for (int img_index=0; img_index < image_npixels_fgmask; img_index++) {
				sbprofile_surface_brightness[img_index] = 0;
			}
		}
	}
	*/
}

void QLens::calculate_image_pixel_surface_brightness_dense()
{
	int i,j,k;

	//cout << "DENSE SB:" << endl;
	//for (j=0; j < source_n_amps; j++) {
		//cout << source_pixel_vector[j] << " ";
	//}
	//cout << endl;
	//double maxsb = -1e30;
	for (int i=0; i < image_npixels; i++) {
		image_surface_brightness[i] = 0;
		for (j=0; j < source_n_amps; j++) {
			image_surface_brightness[i] += Lmatrix_dense[i][j]*source_pixel_vector[j];
		}
		//if (image_surface_brightness[i] < 0) image_surface_brightness[i] = 0;
			//if (image_surface_brightness[i] > maxsb) maxsb=image_surface_brightness[i];
	}

	/*
	if (calculate_foreground) {
		bool at_least_one_foreground_src = false;
		for (k=0; k < n_sb; k++) {
			if (!sb_list[k]->is_lensed) {
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

void QLens::calculate_foreground_pixel_surface_brightness(const int zsrc_i, const bool allow_lensed_nonshapelet_sources)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	bool subgridded;
	int img_index;
	int i,j,k;
	bool at_least_one_foreground_src = false;
	bool at_least_one_lensed_src = false;
	for (k=0; k < n_sb; k++) {
		if (!sb_list[k]->is_lensed) {
			if (zsrc_i<=0) at_least_one_foreground_src = true;
		} else {
			if ((zsrc_i < 0) or (sbprofile_redshift_idx[k]==zsrc_i)) at_least_one_lensed_src = true;
		}
	}

	/*	
	for (int img_index=0; img_index < image_npixels_fgmask; img_index++) {
		//cout << img_index << endl;
		sbprofile_surface_brightness[img_index] = 0;

		i = image_pixel_grid->active_image_pixel_i_fgmask[img_index];
		j = image_pixel_grid->active_image_pixel_j_fgmask[img_index];

		cout << i << " " << j << " " << image_pixel_grid->x_N << " " << image_pixel_grid->y_N << endl;
		cout << image_pixel_grid->center_pts[i][j][0] << " " << image_pixel_grid->center_pts[i][j][1] << endl;
	}
	die();
	*/


	//for (i=0; i < image_pixel_grid->x_N; i++) {
		//for (j=0; j < image_pixel_grid->y_N; j++) {
			//cout << i << " " << j << " " << image_pixel_grid->x_N << " " << image_pixel_grid->y_N << endl;
			//cout << image_pixel_grid->center_pts[i][j][0] << " " << image_pixel_grid->center_pts[i][j][1] << endl;
		//}
	//}
	//cout << "DONE" << endl;
	//die();

	// here, we are adding together SB of foreground sources, but also lensed non-shapelet sources if we're in shapelet mode.
	// If none of those conditions are true, then we skip everything.
	if ((!at_least_one_foreground_src) and ((!allow_lensed_nonshapelet_sources) or (!at_least_one_lensed_src))) {
		for (img_index=0; img_index < image_npixels_fgmask; img_index++) sbprofile_surface_brightness[img_index] = 0;
		return;
	} else {
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

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
		lensvector center_pt, center_srcpt;
		lensvector corner1, corner2, corner3, corner4;
		lensvector corner1_src, corner2_src, corner3_src, corner4_src;
		double subpixel_xlength, subpixel_ylength;
		double noise;
		int subcell_index;
		#pragma omp for private(img_index,i,j,ii,jj,nsplit,u0,w0,sb,subpixel_xlength,subpixel_ylength,center_pt,center_srcpt,corner1,corner2,corner3,corner4,corner1_src,corner2_src,corner3_src,corner4_src,subcell_index,noise) schedule(dynamic)
		for (img_index=0; img_index < image_npixels_fgmask; img_index++) {
			sbprofile_surface_brightness[img_index] = 0;

			i = image_pixel_grid->active_image_pixel_i_fgmask[img_index];
			j = image_pixel_grid->active_image_pixel_j_fgmask[img_index];

			sb = 0;

			if (split_imgpixels) nsplit = image_pixel_grid->nsplits[i][j];
			else nsplit = 1;
			// Now check to see if center of foreground galaxy is in or next to the pixel; if so, make sure it has at least four splittings so its
			// surface brightness is well-reproduced
			if ((at_least_one_foreground_src) and (nsplit < 4) and (i > 0) and (i < image_pixel_grid->x_N-1) and (j > 0) and (j < image_pixel_grid->y_N)) {
				for (k=0; k < n_sb; k++) {
					if (!sb_list[k]->is_lensed) {
						double xc, yc;
						sb_list[k]->get_center_coords(xc,yc);
						if ((xc > image_pixel_grid->corner_pts[i-1][j][0]) and (xc < image_pixel_grid->corner_pts[i+2][j][0]) and (yc > image_pixel_grid->corner_pts[i][j-1][1]) and (yc < image_pixel_grid->corner_pts[i][j+2][1])) nsplit = 4;
					} 
				}
			}

			subpixel_xlength = image_pixel_grid->pixel_xlength/nsplit;
			subpixel_ylength = image_pixel_grid->pixel_ylength/nsplit;
			subcell_index = 0;
			for (ii=0; ii < nsplit; ii++) {
				u0 = ((double) (1+2*ii))/(2*nsplit);
				//center_pt[0] = u0*image_pixel_grid->corner_pts[i][j][0] + (1-u0)*image_pixel_grid->corner_pts[i+1][j][0];
				center_pt[0] = (1-u0)*image_pixel_grid->corner_pts[i][j][0] + u0*image_pixel_grid->corner_pts[i+1][j][0];
				for (jj=0; jj < nsplit; jj++) {
					w0 = ((double) (1+2*jj))/(2*nsplit);
					//center_pt[1] = w0*image_pixel_grid->corner_pts[i][j][1] + (1-w0)*image_pixel_grid->corner_pts[i][j+1][1];
					center_pt[1] = (1-w0)*image_pixel_grid->corner_pts[i][j][1] + w0*image_pixel_grid->corner_pts[i][j+1][1];
					//center_pt = image_pixel_grid->subpixel_center_pts[i][j][subcell_index]; 
					//cout << "CHECK: " << image_pixel_grid->subpixel_center_pts[i][j][subcell_index][0] << " " << center_pt[0] << " and " << image_pixel_grid->subpixel_center_pts[i][j][subcell_index][1] << " " << center_pt[1] << endl;
					for (int k=0; k < n_sb; k++) {
						if ((at_least_one_foreground_src) and (!sb_list[k]->is_lensed) and ((source_fit_mode != Shapelet_Source) or (sb_list[k]->sbtype != SHAPELET))) {
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
								noise = (use_noise_map) ? image_pixel_grid->noise_map[i][j] : background_pixel_noise;
								sb += sb_list[k]->surface_brightness_zoom(center_pt,corner1,corner2,corner3,corner4,noise);
							}
						}
						else if ((allow_lensed_nonshapelet_sources) and (sb_list[k]->is_lensed) and ((zsrc_i<0) or (sbprofile_redshift_idx[k]==zsrc_i)) and (sb_list[k]->sbtype != SHAPELET) and (image_pixel_data->foreground_mask[i][j])) { // if source mode is shapelet and sbprofile is shapelet, will include in inversion
							//center_srcpt = image_pixel_grid->subpixel_center_sourcepts[i][j][subcell_index];
							//center_srcpt = image_pixel_grid->subpixel_center_sourcepts[i][j][subcell_index];
							//find_sourcept(center_pt,center_srcpt,thread,reference_zfactors,default_zsrc_beta_factors);
							//sb += sb_list[k]->surface_brightness(center_srcpt[0],center_srcpt[1]);
							if (split_imgpixels) sb += sb_list[k]->surface_brightness(image_pixel_grid->subpixel_center_sourcepts[i][j][subcell_index][0],image_pixel_grid->subpixel_center_sourcepts[i][j][subcell_index][1]);
							else sb += sb_list[k]->surface_brightness(image_pixel_grid->center_sourcepts[i][j][0],image_pixel_grid->center_sourcepts[i][j][1]);
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
			sbprofile_surface_brightness[img_index] += sb / (nsplit*nsplit);
		}
	}
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating foreground SB: " << wtime << endl;
	}
#endif

	}
	//for (int img_index=0; img_index < image_npixels_fgmask; img_index++) {
		//cout << sbprofile_surface_brightness[img_index] << endl;
	//}
	//die();
	PSF_convolution_pixel_vector(zsrc_i,true,false);
}

void QLens::add_foreground_to_image_pixel_vector()
{
	for (int img_index=0; img_index < image_npixels; img_index++) {
		image_surface_brightness[img_index] += sbprofile_surface_brightness[img_index];
	}
}

void QLens::store_image_pixel_surface_brightness(const int zsrc_i)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	int i,j;
	for (i=0; i < image_pixel_grid->x_N; i++)
		for (j=0; j < image_pixel_grid->y_N; j++)
			image_pixel_grid->surface_brightness[i][j] = 0;

	for (int img_index=0; img_index < image_npixels; img_index++) {
		i = image_pixel_grid->active_image_pixel_i[img_index];
		j = image_pixel_grid->active_image_pixel_j[img_index];
		image_pixel_grid->surface_brightness[i][j] = image_surface_brightness[img_index];
		//cout << image_surface_brightness[img_index] << endl;;
	}
}

void QLens::store_foreground_pixel_surface_brightness(const int zsrc_i) // note, foreground_surface_brightness could also include source objects that aren't shapelets (if in shapelet mode)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	int i,j;
	for (int img_index=0; img_index < image_npixels_fgmask; img_index++) {
		i = image_pixel_grid->active_image_pixel_i_fgmask[img_index];
		j = image_pixel_grid->active_image_pixel_j_fgmask[img_index];
		image_pixel_grid->foreground_surface_brightness[i][j] = sbprofile_surface_brightness[img_index];
	}
}

void QLens::vectorize_image_pixel_surface_brightness(const int zsrc_i, bool use_mask)
{
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	int i,j,k=0;
	int subcell_index, nsubpix, image_subpixel_index=0;
	if (use_mask) {
		int n=0, nsub=0;
		for (j=0; j < image_pixel_grid->y_N; j++) {
			for (i=0; i < image_pixel_grid->x_N; i++) {
				if ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j])) {
					n++;
					if (psf_supersampling) {
						if (image_pixel_grid->nsplits[i][j] != default_imgpixel_nsplit) die("nsplit has to be the same for all pixels to use supersampling (pixel (%i,%i), nsplits: %i vs %i)",i,j,image_pixel_grid->nsplits[i][j],default_imgpixel_nsplit);
						nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]);
						nsub += nsubpix;
					}
				}
			}
		}
		image_npixels = n;
		image_pixel_grid->n_active_pixels = n;
		if (psf_supersampling) {
			if (nsub != image_npixels*default_imgpixel_nsplit*default_imgpixel_nsplit) die("number of counted subpixels (%i) does not match image_npixels*imgpixel_nsplit (%i)",nsub,(image_npixels*default_imgpixel_nsplit*default_imgpixel_nsplit));
			image_n_subpixels = nsub;
		}
	} else {
		image_npixels = image_pixel_grid->x_N*image_pixel_grid->y_N;
		image_pixel_grid->n_active_pixels = image_npixels;
		if (psf_supersampling) image_n_subpixels = image_npixels*default_imgpixel_nsplit*default_imgpixel_nsplit;
	}
	if (image_pixel_grid->active_image_pixel_i == NULL) {
		if (image_pixel_grid->active_image_pixel_i != NULL) delete[] image_pixel_grid->active_image_pixel_i;
		if (image_pixel_grid->active_image_pixel_j != NULL) delete[] image_pixel_grid->active_image_pixel_j;
		image_pixel_grid->active_image_pixel_i = new int[image_npixels];
		image_pixel_grid->active_image_pixel_j = new int[image_npixels];
	}
	int ii,jj;
	if (psf_supersampling) {
		if (image_pixel_grid->active_image_pixel_i_ss == NULL) {
			if (image_pixel_grid->active_image_pixel_i_ss != NULL) delete[] image_pixel_grid->active_image_pixel_i_ss;
			if (image_pixel_grid->active_image_pixel_j_ss != NULL) delete[] image_pixel_grid->active_image_pixel_j_ss;
			if (image_pixel_grid->active_image_subpixel_ss != NULL) delete[] image_pixel_grid->active_image_subpixel_ss;
			if (image_pixel_grid->active_image_subpixel_ii != NULL) delete[] image_pixel_grid->active_image_subpixel_ii;
			if (image_pixel_grid->active_image_subpixel_jj != NULL) delete[] image_pixel_grid->active_image_subpixel_jj;
			if (image_pixel_grid->image_pixel_i_from_subcell_ii != NULL) delete[] image_pixel_grid->image_pixel_i_from_subcell_ii;
			if (image_pixel_grid->image_pixel_j_from_subcell_jj != NULL) delete[] image_pixel_grid->image_pixel_j_from_subcell_jj;

			image_pixel_grid->active_image_pixel_i_ss = new int[image_n_subpixels];
			image_pixel_grid->active_image_pixel_j_ss = new int[image_n_subpixels];
			image_pixel_grid->active_image_subpixel_ss = new int[image_n_subpixels];
			image_pixel_grid->active_image_subpixel_ii = new int[image_n_subpixels];
			image_pixel_grid->active_image_subpixel_jj = new int[image_n_subpixels];
			image_pixel_grid->image_pixel_i_from_subcell_ii = new int[image_pixel_grid->x_N*default_imgpixel_nsplit];
			image_pixel_grid->image_pixel_j_from_subcell_jj = new int[image_pixel_grid->y_N*default_imgpixel_nsplit];
		}
		for (j=0; j < image_pixel_grid->y_N; j++) {
			for (i=0; i < image_pixel_grid->x_N; i++) {
				for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
					ii = i*default_imgpixel_nsplit + subcell_index / default_imgpixel_nsplit;
					jj = j*default_imgpixel_nsplit + subcell_index % default_imgpixel_nsplit;
					image_pixel_grid->image_pixel_i_from_subcell_ii[ii] = i;
					image_pixel_grid->image_pixel_j_from_subcell_jj[jj] = j;
				}
			}
		}
	}

	for (j=0; j < image_pixel_grid->y_N; j++) {
		for (i=0; i < image_pixel_grid->x_N; i++) {
			if ((!use_mask) or (image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j])) {
				//if (image_pixel_grid->active_image_pixel_i[k] != i) cout << "ACTIVE IM I ARRAY FUCKED UP (" << image_pixel_grid->active_image_pixel_i[k] << " vs " << i << ")"  << endl;
				//if (image_pixel_grid->active_image_pixel_j[k] != j) cout << "ACTIVE IM J ARRAY FUCKED UP" << image_pixel_grid->active_image_pixel_j[k] << " vs " << j << ")"   << endl;
				image_pixel_grid->active_image_pixel_i[k] = i;
				image_pixel_grid->active_image_pixel_j[k] = j;
				//if (image_pixel_grid->pixel_index[i][j] != k) cout << "PIXEL_INDEX ARRAYS FUCKED UP" << endl;
				image_pixel_grid->pixel_index[i][j] = k++;
				if (psf_supersampling) {
					nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]);
					if (image_pixel_grid->nsplits[i][j] != default_imgpixel_nsplit) die("nsplit has to be the same for all pixels to use supersampling (pixel (%i,%i), nsplits: %i vs %i)",i,j,image_pixel_grid->nsplits[i][j],default_imgpixel_nsplit);
					for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
						image_pixel_grid->active_image_pixel_i_ss[image_subpixel_index] = i;
						image_pixel_grid->active_image_pixel_j_ss[image_subpixel_index] = j;
						ii = i*default_imgpixel_nsplit + subcell_index / default_imgpixel_nsplit;
						jj = j*default_imgpixel_nsplit + subcell_index % default_imgpixel_nsplit;
						image_pixel_grid->active_image_subpixel_ii[image_subpixel_index] = ii;
						image_pixel_grid->active_image_subpixel_jj[image_subpixel_index] = jj;

						//cout << "SUBCELL: " << image_pixel_grid->active_image_subpixel_ii[image_subpixel_index] << " " << image_pixel_grid->active_image_subpixel_jj[image_subpixel_index] << " " << image_pixel_grid->subpixel_center_pts[i][j][subcell_index][0] << " " << image_pixel_grid->subpixel_center_pts[i][j][subcell_index][1] << endl;

						image_pixel_grid->active_image_subpixel_ss[image_subpixel_index] = subcell_index;
						image_pixel_grid->subpixel_index[ii][jj] = image_subpixel_index++;
						//cout << image_subpixel_index << " (total=" << image_n_subpixels << ")" << endl;
					}
				}
			} else {
				image_pixel_grid->pixel_index[i][j] = -1;
				if (psf_supersampling) {
					for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
						ii = i*default_imgpixel_nsplit + subcell_index / default_imgpixel_nsplit;
						jj = j*default_imgpixel_nsplit + subcell_index % default_imgpixel_nsplit;
						image_pixel_grid->subpixel_index[ii][jj] = -1;
					}
				}
			}
		}
	}
	//} else {
		//image_npixels = image_pixel_grid->n_active_pixels;
		//// THERE SHOULD ALSO BE N_ACTIVE_SUBPIXELS, RIGHT? FOR PSF_SUPERSAMPLING
	//}
	if (image_surface_brightness != NULL) delete[] image_surface_brightness;
	image_surface_brightness = new double[image_npixels];
	imgpixel_covinv_vector = new double[image_npixels];

	for (k=0; k < image_npixels; k++) {
		i = image_pixel_grid->active_image_pixel_i[k];
		j = image_pixel_grid->active_image_pixel_j[k];
		image_surface_brightness[k] = image_pixel_grid->surface_brightness[i][j];
	}
	if (use_noise_map) {
		int ii,i,j;
		for (ii=0; ii < image_npixels; ii++) {
			i = image_pixel_grid->active_image_pixel_i[ii];
			j = image_pixel_grid->active_image_pixel_j[ii];
			imgpixel_covinv_vector[ii] = image_pixel_data->covinv_map[i][j];
		}
	}
	
	if (psf_supersampling) {
		if (image_surface_brightness_supersampled != NULL) delete[] image_surface_brightness_supersampled;
		image_surface_brightness_supersampled = new double[image_n_subpixels];
		//image_subpixel_index = 0;
		//for (k=0; k < image_npixels; k++) {
			//i = image_pixel_grid->active_image_pixel_i[k];
			//j = image_pixel_grid->active_image_pixel_j[k];
			//nsubpix = INTSQR(image_pixel_grid->nsplits[i][j]);
			//for (subcell_index=0; subcell_index < nsubpix; subcell_index++) {
				//image_surface_brightness_supersampled[image_subpixel_index++] = image_pixel_grid->subpixel_surface_brightness[i][j][subcell_index];
			//}
		//}
		for (int img_index=0; img_index < image_n_subpixels; img_index++) {
			i = image_pixel_grid->active_image_pixel_i_ss[img_index];
			j = image_pixel_grid->active_image_pixel_j_ss[img_index];
			subcell_index = image_pixel_grid->active_image_subpixel_ss[img_index];
			image_surface_brightness_supersampled[img_index] = image_pixel_grid->subpixel_surface_brightness[i][j][subcell_index];
		}
		//average_supersampled_image_surface_brightness(zsrc_i);
	}
}

double QLens::find_sbprofile_surface_brightness(lensvector &pt)
{
	double sb = 0;
	lensvector srcpt;
	
	find_sourcept(pt,srcpt,0,reference_zfactors,default_zsrc_beta_factors);
	//cout << "src=" << srcpt[0] << "," << srcpt[1] << " pt=" << pt[0] << "," << pt[1] << endl;
	for (int k=0; k < n_sb; k++) {
		if (sb_list[k]->is_lensed) {
			sb += sb_list[k]->surface_brightness(srcpt[0],srcpt[1]);
			//if (!sb_list[k]->zoom_subgridding) sb += sb_list[k]->surface_brightness(srcpt[0],srcpt[1]);
			//else {
				//lensvector srcpt1,srcpt2,srcpt3,srcpt4;
				//find_sourcept(pt1,srcpt1,0,reference_zfactors,default_zsrc_beta_factors);
				//find_sourcept(pt2,srcpt2,0,reference_zfactors,default_zsrc_beta_factors);
				//find_sourcept(pt3,srcpt3,0,reference_zfactors,default_zsrc_beta_factors);
				//find_sourcept(pt4,srcpt4,0,reference_zfactors,default_zsrc_beta_factors);
				//sb += sb_list[k]->surface_brightness_zoom(pt,srcpt1,srcpt2,srcpt3,srcpt4);
			//}
		} else {
			sb += sb_list[k]->surface_brightness(pt[0],pt[1]);
			//if (!sb_list[k]->zoom_subgridding) sb += sb_list[k]->surface_brightness(pt[0],pt[1]);
			//else sb += sb_list[k]->surface_brightness_zoom(pt,pt1,pt2,pt3,pt4);
		}
		//cout << "object " << k << ": sb=" << sb << endl;
	}
	return sb;
}

/*
void QLens::plot_image_pixel_surface_brightness(string outfile_root, const int zsrc_i)
{
	cout << "WHAT??" << endl;
	ImagePixelGrid *image_pixel_grid;
	if (zsrc_i < 0) image_pixel_grid = image_pixel_grid0;
	else image_pixel_grid = image_pixel_grids[zsrc_i];
	string sb_filename = outfile_root + ".dat";
	string x_filename = outfile_root + ".x";
	string y_filename = outfile_root + ".y";
	string pts_filename = outfile_root + "_pts.dat";

	ofstream xfile; open_output_file(xfile,x_filename);
	for (int i=0; i <= image_pixel_grid->x_N; i++) {
		xfile << image_pixel_grid->corner_pts[i][0][0] << endl;
	}

	ofstream yfile; open_output_file(yfile,y_filename);
	for (int i=0; i <= image_pixel_grid->y_N; i++) {
		yfile << image_pixel_grid->corner_pts[0][i][1] << endl;
	}

	ofstream surface_brightness_file; open_output_file(surface_brightness_file,sb_filename);
	ofstream pts_file; open_output_file(pts_file,pts_filename);
	int index=0;
	for (int j=0; j < image_pixel_grid->y_N; j++) {
		for (int i=0; i < image_pixel_grid->x_N; i++) {
			if ((image_pixel_grid->maps_to_source_pixel[i][j]) and ((image_pixel_grid->pixel_in_mask==NULL) or (image_pixel_grid->pixel_in_mask[i][j]))) {
				surface_brightness_file << image_surface_brightness[index++] << " ";
				pts_file << image_pixel_grid->center_pts[i][j][0] << " " << image_pixel_grid->center_pts[i][j][1] << " " << image_pixel_grid->center_sourcepts[i][j][0] << " " << image_pixel_grid->center_sourcepts[i][j][1] << endl;
			} else surface_brightness_file << "0 ";
		}
		surface_brightness_file << endl;
	}
}
*/

