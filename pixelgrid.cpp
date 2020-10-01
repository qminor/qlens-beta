#include "trirectangle.h"
#include "cg.h"
#include "pixelgrid.h"
#include "profile.h"
#include "qlens.h"
#include "mathexpr.h"
#include "errors.h"
#include <stdio.h>

#ifdef USE_UMFPACK
#include "umfpack.h"
#endif

#ifdef USE_FITS
#include "fitsio.h"
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

int SourcePixelGrid::nthreads = 0;
const int SourcePixelGrid::max_levels = 6;
int SourcePixelGrid::number_of_pixels;
int *SourcePixelGrid::imin, *SourcePixelGrid::imax, *SourcePixelGrid::jmin, *SourcePixelGrid::jmax;
TriRectangleOverlap *SourcePixelGrid::trirec = NULL;
InterpolationCells *SourcePixelGrid::nearest_interpolation_cells = NULL;
lensvector **SourcePixelGrid::interpolation_pts[3];
int *SourcePixelGrid::n_interpolation_pts = NULL;

// The following should probably just be private, local variables in the relevant functions, that have to keep getting set from the lens pointers.
// Otherwise it will be bug prone whenever changes are made, since the zfactors/betafactors pointers may be deleted and reassigned
double *SourcePixelGrid::srcgrid_zfactors = NULL;
double *ImagePixelGrid::imggrid_zfactors = NULL;
double **SourcePixelGrid::srcgrid_betafactors = NULL;
double **ImagePixelGrid::imggrid_betafactors = NULL;

// parameters for creating the recursive grid
double SourcePixelGrid::xcenter, SourcePixelGrid::ycenter;
double SourcePixelGrid::srcgrid_xmin, SourcePixelGrid::srcgrid_xmax, SourcePixelGrid::srcgrid_ymin, SourcePixelGrid::srcgrid_ymax;
int SourcePixelGrid::u_split_initial, SourcePixelGrid::w_split_initial;
double SourcePixelGrid::min_cell_area;

// NOTE!!! It would be better to make a few of these (e.g. levels) non-static and contained in the zeroth-level grid, and just give all the subcells a pointer to the zeroth-level grid.
// That way, you can create multiple source grids and they won't interfere with each other.
int SourcePixelGrid::levels, SourcePixelGrid::splitlevels;
//lensvector SourcePixelGrid::d1, SourcePixelGrid::d2, SourcePixelGrid::d3, SourcePixelGrid::d4;
//double SourcePixelGrid::product1, SourcePixelGrid::product2, SourcePixelGrid::product3;
ImagePixelGrid* SourcePixelGrid::image_pixel_grid = NULL;
bool SourcePixelGrid::regrid;
int *SourcePixelGrid::maxlevs = NULL;
lensvector ***SourcePixelGrid::xvals_threads = NULL;
lensvector ***SourcePixelGrid::corners_threads = NULL;
lensvector **SourcePixelGrid::twistpts_threads = NULL;
int **SourcePixelGrid::twist_status_threads = NULL;

ifstream SourcePixelGrid::sb_infile;

/***************************************** Functions in class SourcePixelGrid ****************************************/

void SourcePixelGrid::set_splitting(int usplit0, int wsplit0, double min_cs)
{
	u_split_initial = usplit0;
	w_split_initial = wsplit0;
	if ((u_split_initial < 2) or (w_split_initial < 2)) die("source grid dimensions cannot be smaller than 2 along either direction");
	min_cell_area = min_cs;
}

void SourcePixelGrid::allocate_multithreaded_variables(const int& threads, const bool reallocate)
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
	n_interpolation_pts = new int[threads];
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

void SourcePixelGrid::deallocate_multithreaded_variables()
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
		delete[] n_interpolation_pts;
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
		n_interpolation_pts = NULL;
		xvals_threads = NULL;
		corners_threads = NULL;
		twistpts_threads = NULL;
		twist_status_threads = NULL;
	}
}

SourcePixelGrid::SourcePixelGrid(QLens* lens_in, double x_min, double x_max, double y_min, double y_max) : lens(lens_in)	// use for top-level cell only; subcells use constructor below
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

// this constructor is used for a Cartesian grid
	center_pt = 0;
	// For the Cartesian grid, u = x, w = y
	u_N = u_split_initial;
	w_N = w_split_initial;
	level = 0;
	levels = 0;
	ii=jj=0;
	cell = NULL;
	parent_cell = NULL;
	maps_to_image_pixel = false;
	maps_to_image_window = false;
	active_pixel = false;
	srcgrid_zfactors = lens->reference_zfactors;
	srcgrid_betafactors = lens->default_zsrc_beta_factors;

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
	for (int i=0; i < u_N+1; i++)
		delete[] firstlevel_xvals[i];
	delete[] firstlevel_xvals;
}

SourcePixelGrid::SourcePixelGrid(QLens* lens_in, string pixel_data_fileroot, const double& minarea_in) : lens(lens_in)	// use for top-level cell only; subcells use constructor below
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

	min_cell_area = minarea_in;
	string info_filename = pixel_data_fileroot + ".info";
	ifstream infofile(info_filename.c_str());
	double cells_per_pixel;
	infofile >> u_split_initial >> w_split_initial >> cells_per_pixel;
	infofile >> srcgrid_xmin >> srcgrid_xmax >> srcgrid_ymin >> srcgrid_ymax;

	// this constructor is used for a Cartesian grid
	center_pt = 0;
	// For the Cartesian grid, u = x, w = y
	u_N = u_split_initial;
	w_N = w_split_initial;
	level = 0;
	levels = 0;
	ii=jj=0;
	cell = NULL;
	parent_cell = NULL;
	maps_to_image_pixel = false;
	maps_to_image_window = false;
	active_pixel = false;
	srcgrid_zfactors = lens->reference_zfactors;
	srcgrid_betafactors = lens->default_zsrc_beta_factors;

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

	string sbfilename = pixel_data_fileroot + ".sb";
	sb_infile.open(sbfilename.c_str());
	read_surface_brightness_data();
	sb_infile.close();
	for (int i=0; i < u_N+1; i++)
		delete[] firstlevel_xvals[i];
	delete[] firstlevel_xvals;
}

// ***NOTE: the following constructor should NOT be used because there are static variables (e.g. levels), so more than one source grid
// is a bad idea. To make this work, you need to make those variables non-static and contained in the zeroth-level grid (and give subcells
// a pointer to the zeroth-level grid).
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
	parent_cell = NULL;
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

void SourcePixelGrid::read_surface_brightness_data()
{
	double sb;
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			sb_infile >> sb;
			if (sb==-1e30) // I can't think of a better dividing value to use right now, so -1e30 is what I am using at the moment
			{
				cell[i][j]->split_cells(2,2,0);
				cell[i][j]->read_surface_brightness_data();
			} else {
				cell[i][j]->surface_brightness = sb;
			}
		}
	}
}

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

SourcePixelGrid::SourcePixelGrid(QLens* lens_in, lensvector** xij, const int& i, const int& j, const int& level_in, SourcePixelGrid* parent_ptr)
{
	u_N = 1;
	w_N = 1;
	level = level_in;
	cell = NULL;
	ii=i; jj=j; // store the index carried by this cell in the grid of the parent cell
	parent_cell = parent_ptr;
	maps_to_image_pixel = false;
	maps_to_image_window = false;
	active_pixel = false;
	lens = lens_in;

	corner_pt[0] = xij[i][j];
	corner_pt[1] = xij[i][j+1];
	corner_pt[2] = xij[i+1][j];
	corner_pt[3] = xij[i+1][j+1];

	center_pt[0] = (corner_pt[0][0] + corner_pt[1][0] + corner_pt[2][0] + corner_pt[3][0]) / 4.0;
	center_pt[1] = (corner_pt[0][1] + corner_pt[1][1] + corner_pt[2][1] + corner_pt[3][1]) / 4.0;
	find_cell_area();
}

void SourcePixelGrid::assign_surface_brightness()
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->assign_surface_brightness();
			else {
				cell[i][j]->surface_brightness = 0;
				for (int k=0; k < lens->n_sb; k++) {
					if (lens->sb_list[k]->is_lensed) cell[i][j]->surface_brightness += lens->sb_list[k]->surface_brightness(cell[i][j]->center_pt[0],cell[i][j]->center_pt[1]);
				}
			}
		}
	}
}

void SourcePixelGrid::update_surface_brightness(int& index)
{
	for (int j=0; j < w_N; j++) {
		for (int i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->update_surface_brightness(index);
			else {
				if (cell[i][j]->active_pixel) {
					cell[i][j]->surface_brightness = lens->source_surface_brightness[index++];
				} else {
					cell[i][j]->surface_brightness = 0;
				}
			}
		}
	}
}

void SourcePixelGrid::fill_surface_brightness_vector()
{
	int column_j = 0;
	fill_surface_brightness_vector_recursive(column_j);
}

void SourcePixelGrid::fill_surface_brightness_vector_recursive(int& column_j)
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->fill_surface_brightness_vector_recursive(column_j);
			else {
				if (cell[i][j]->active_pixel) {
					lens->source_surface_brightness[column_j++] = cell[i][j]->surface_brightness;
				}
			}
		}
	}
}

void SourcePixelGrid::fill_n_image_vector()
{
	int column_j = 0;
	fill_n_image_vector_recursive(column_j);
}

void SourcePixelGrid::fill_n_image_vector_recursive(int& column_j)
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

void SourcePixelGrid::find_avg_n_images()
{
	// no support for adaptive grid in this function, since we're only using it when parameterized sources are being used

	lens->max_pixel_sb=-1e30;
	int max_sb_i, max_sb_j;
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->surface_brightness > lens->max_pixel_sb) {
				lens->max_pixel_sb = cell[i][j]->surface_brightness;
				max_sb_i = i;
				max_sb_j = j;
			}
		}
	}

	lens->n_images_at_sbmax = cell[max_sb_i][max_sb_j]->n_images;
	lens->pixel_avg_n_image = 0;
	double sbtot = 0;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->surface_brightness >= lens->max_pixel_sb*lens->n_image_prior_sb_frac) {
				lens->pixel_avg_n_image += cell[i][j]->n_images*cell[i][j]->surface_brightness;
				sbtot += cell[i][j]->surface_brightness;
			}
		}
	}
	if (sbtot != 0) lens->pixel_avg_n_image /= sbtot;
}

ofstream SourcePixelGrid::pixel_surface_brightness_file;
ofstream SourcePixelGrid::pixel_magnification_file;
ofstream SourcePixelGrid::pixel_n_image_file;

void SourcePixelGrid::store_surface_brightness_grid_data(string root)
{
	string img_filename = root + ".sb";
	string info_filename = root + ".info";

	pixel_surface_brightness_file.open(img_filename.c_str());
	write_surface_brightness_to_file();
	pixel_surface_brightness_file.close();

	ofstream pixel_info; lens->open_output_file(pixel_info,info_filename);
	pixel_info << u_split_initial << " " << w_split_initial << " " << levels << endl;
	pixel_info << srcgrid_xmin << " " << srcgrid_xmax << " " << srcgrid_ymin << " " << srcgrid_ymax << endl;
}

void SourcePixelGrid::write_surface_brightness_to_file()
{
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) {
				pixel_surface_brightness_file << "-1e30\n";
				cell[i][j]->write_surface_brightness_to_file();
			} else {
				pixel_surface_brightness_file << cell[i][j]->surface_brightness << endl;
			}
		}
	}
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
					cell[i][j]->plot_cell_surface_brightness(line_number,pixels_per_cell_x,pixels_per_cell_y);
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

void SourcePixelGrid::plot_cell_surface_brightness(int line_number, int pixels_per_cell_x, int pixels_per_cell_y)
{
	int cell_row, subplot_pixels_per_cell_x, subplot_pixels_per_cell_y, subline_number=line_number;
	subplot_pixels_per_cell_x = pixels_per_cell_x/u_N;
	subplot_pixels_per_cell_y = pixels_per_cell_y/w_N;
	cell_row = line_number / subplot_pixels_per_cell_y;
	subline_number -= cell_row*subplot_pixels_per_cell_y;

	int i,j;
	for (i=0; i < u_N; i++) {
		if (cell[i][cell_row]->cell != NULL) {
			cell[i][cell_row]->plot_cell_surface_brightness(subline_number,subplot_pixels_per_cell_x,subplot_pixels_per_cell_y);
		} else {
			for (j=0; j < subplot_pixels_per_cell_x; j++) {
				pixel_surface_brightness_file << cell[i][cell_row]->surface_brightness << " ";
				pixel_magnification_file << log(cell[i][cell_row]->total_magnification)/log(10) << " ";
				if (lens->n_image_prior) pixel_n_image_file << cell[i][cell_row]->n_images << " ";
			}
		}
	}
}

inline void SourcePixelGrid::find_cell_area()
{
	//d1[0] = corner_pt[2][0] - corner_pt[0][0]; d1[1] = corner_pt[2][1] - corner_pt[0][1];
	//d2[0] = corner_pt[1][0] - corner_pt[0][0]; d2[1] = corner_pt[1][1] - corner_pt[0][1];
	//d3[0] = corner_pt[2][0] - corner_pt[3][0]; d3[1] = corner_pt[2][1] - corner_pt[3][1];
	//d4[0] = corner_pt[1][0] - corner_pt[3][0]; d4[1] = corner_pt[1][1] - corner_pt[3][1];
	// split cell into two triangles; cross product of the vectors forming the legs gives area of each triangle, so their sum gives area of cell
	//cell_area = 0.5 * (abs(d1 ^ d2) + abs(d3 ^ d4)); // overkill since the cells are just square
	cell_area = (corner_pt[2][0] - corner_pt[0][0])*(corner_pt[1][1]-corner_pt[0][1]);
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

void SourcePixelGrid::test_neighbors() // for testing purposes, to make sure neighbors are assigned correctly
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

void SourcePixelGrid::assign_level_neighbors(int neighbor_level)
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

void SourcePixelGrid::split_cells(const int usplit, const int wsplit, const int& thread)
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

	cell = new SourcePixelGrid**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new SourcePixelGrid*[w_N];
		for (j=0; j < w_N; j++) {
			cell[i][j] = new SourcePixelGrid(lens,xvals_threads[thread],i,j,level+1,this);
			cell[i][j]->total_magnification = 0;
			if (lens->n_image_prior) cell[i][j]->n_images = 0;
		}
	}
	if (level == maxlevs[thread]) {
		maxlevs[thread]++; // our subcells are at the max level, so splitting them increases the number of levels by 1
	}
	number_of_pixels += u_N*w_N - 1; // subtract one because we're not counting the parent cell as a source pixel
}

void SourcePixelGrid::unsplit()
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
	number_of_pixels -= (u_N*w_N - 1);
	cell = NULL;
	surface_brightness /= (u_N*w_N);
	u_N=1; w_N = 1;
}

ofstream SourcePixelGrid::xgrid;

void QLens::plot_source_pixel_grid(const char filename[])
{
	if (source_pixel_grid==NULL) { warn("No source surface brightness map has been generated"); return; }
	SourcePixelGrid::xgrid.open(filename, ifstream::out);
	source_pixel_grid->plot_corner_coordinates();
	SourcePixelGrid::xgrid.close();
}

void SourcePixelGrid::plot_corner_coordinates()
{
	if (level > 0) {
		xgrid << corner_pt[1][0] << " " << corner_pt[1][1] << endl;
		xgrid << corner_pt[3][0] << " " << corner_pt[3][1] << endl;
		xgrid << corner_pt[2][0] << " " << corner_pt[2][1] << endl;
		xgrid << corner_pt[0][0] << " " << corner_pt[0][1] << endl;
		xgrid << corner_pt[1][0] << " " << corner_pt[1][1] << endl;
		xgrid << endl;
	}

	if (cell != NULL)
		for (int i=0; i < u_N; i++)
			for (int j=0; j < w_N; j++)
				cell[i][j]->plot_corner_coordinates();
}

inline bool SourcePixelGrid::check_if_in_neighborhood(lensvector **input_corner_pts, bool& inside, const int& thread)
{
	if (trirec[thread].determine_if_in_neighborhood(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],*input_corner_pts[3],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1],inside)==true) return true;
	return false;
}

inline bool SourcePixelGrid::check_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
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

inline double SourcePixelGrid::find_rectangle_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread, const int& i, const int& j)
{
	if (twist_status==0) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else if (twist_status==1) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]) + trirec[thread].find_overlap_area(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	}
}

inline bool SourcePixelGrid::check_triangle1_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else if (twist_status==1) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else {
		return trirec[thread].determine_if_overlap(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	}
}

inline bool SourcePixelGrid::check_triangle2_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else if (twist_status==1) {
		return trirec[thread].determine_if_overlap(*input_corner_pts[1],*input_corner_pts[3],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	} else {
		return trirec[thread].determine_if_overlap(*twist_pt,*input_corner_pts[3],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]);
	}
}

inline double SourcePixelGrid::find_triangle1_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	if (twist_status==0) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*input_corner_pts[2],corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else if (twist_status==1) {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[2],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	} else {
		return (trirec[thread].find_overlap_area(*input_corner_pts[0],*input_corner_pts[1],*twist_pt,corner_pt[0][0],corner_pt[2][0],corner_pt[0][1],corner_pt[1][1]));
	}
}

inline double SourcePixelGrid::find_triangle2_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
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

void SourcePixelGrid::calculate_pixel_magnifications()
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

	double xstep, ystep;
	xstep = (srcgrid_xmax-srcgrid_xmin)/u_N;
	ystep = (srcgrid_ymax-srcgrid_ymin)/w_N;
	int src_raytrace_i, src_raytrace_j;
	int img_i, img_j;

	//int ntot = image_pixel_grid->x_N * image_pixel_grid->y_N;

	long int ntot = 0;
	for (i=0; i < image_pixel_grid->x_N; i++) {
		for (j=0; j < image_pixel_grid->y_N; j++) {
			if (lens->image_pixel_data->extended_mask[i][j]) ntot++;
		}
	}
	int *extended_mask_i = new int[ntot];
	int *extended_mask_j = new int[ntot];
	int n_cell=0;
	// you shouldn't have to calculate this again...this was already calculated in redo_lensing_calculations. Save extended_mask_i arrays?
	for (j=0; j < image_pixel_grid->y_N; j++) {
		for (i=0; i < image_pixel_grid->x_N; i++) {
			if (lens->image_pixel_data->extended_mask[i][j]) {
				extended_mask_i[n_cell] = i;
				extended_mask_j[n_cell] = j;
				n_cell++;
			}
		}
	}


	int *overlap_matrix_row_nn = new int[ntot];
	vector<double> *overlap_matrix_rows = new vector<double>[ntot];
	vector<int> *overlap_matrix_index_rows = new vector<int>[ntot];
	vector<double> *overlap_area_matrix_rows;
	overlap_area_matrix_rows = new vector<double>[ntot];

	int mpi_chunk, mpi_start, mpi_end;
	mpi_chunk = ntot / lens->group_np;
	mpi_start = lens->group_id*mpi_chunk;
	if (lens->group_id == lens->group_np-1) mpi_chunk += (ntot % lens->group_np); // assign the remainder elements to the last mpi process
	mpi_end = mpi_start + mpi_chunk;

	int overlap_matrix_nn;
	int overlap_matrix_nn_part=0;
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
		//int img_i2,img_j2;
					//cout << "TRYING HERE..." << endl;
		#pragma omp for private(i,j,nsrc,overlap_area,weighted_overlap,triangle1_overlap,triangle2_overlap,triangle1_weight,triangle2_weight,inside) schedule(dynamic) reduction(+:overlap_matrix_nn_part)
		for (n=mpi_start; n < mpi_end; n++)
		{
			overlap_matrix_row_nn[n] = 0;
			//img_j = n / image_pixel_grid->x_N;
			//img_i = n % image_pixel_grid->x_N;
			img_j = extended_mask_j[n];
			img_i = extended_mask_i[n];
			//cout << "WOAH " << img_i << " " << img_j << " " << img_i2 << " " << img_j2 << endl;

			corners_threads[thread][0] = &image_pixel_grid->corner_sourcepts[img_i][img_j];
			corners_threads[thread][1] = &image_pixel_grid->corner_sourcepts[img_i][img_j+1];
			corners_threads[thread][2] = &image_pixel_grid->corner_sourcepts[img_i+1][img_j];
			corners_threads[thread][3] = &image_pixel_grid->corner_sourcepts[img_i+1][img_j+1];
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
			//cout << "BLERGH " << min_i << " " << max_i << " " << min_j << " " << max_j << endl;
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
						if ((triangle1_overlap != 0) or (triangle2_overlap != 0)) {
							weighted_overlap = triangle1_weight + triangle2_weight;
							cout << "WEIGHT: " << weighted_overlap << endl;
							overlap_matrix_rows[n].push_back(weighted_overlap);
							overlap_matrix_index_rows[n].push_back(nsrc);
							overlap_matrix_row_nn[n]++;

							overlap_area = triangle1_overlap + triangle2_overlap;
							if ((image_pixel_grid->fit_to_data == NULL) or (!image_pixel_grid->fit_to_data[img_i][img_j])) overlap_area = 0;
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
	int *image_pixel_location_overlap = new int[ntot+1];
	double *overlap_area_matrix;
	overlap_area_matrix = new double[overlap_matrix_nn];

#ifdef USE_MPI
	int id, chunk, start, end, length;
	for (id=0; id < lens->group_np; id++) {
		chunk = ntot / lens->group_np;
		start = id*chunk;
		if (id == lens->group_np-1) chunk += (ntot % lens->group_np); // assign the remainder elements to the last mpi process
		MPI_Bcast(overlap_matrix_row_nn + start,chunk,MPI_INT,id,sub_comm);
	}
#endif

	image_pixel_location_overlap[0] = 0;
	int n,l;
	for (n=0; n < ntot; n++) {
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
		chunk = ntot / lens->group_np;
		start = id*chunk;
		if (id == lens->group_np-1) chunk += (ntot % lens->group_np); // assign the remainder elements to the last mpi process
		end = start + chunk;
		length = image_pixel_location_overlap[end] - image_pixel_location_overlap[start];
		MPI_Bcast(overlap_matrix + image_pixel_location_overlap[start],length,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(overlap_matrix_index + image_pixel_location_overlap[start],length,MPI_INT,id,sub_comm);
		MPI_Bcast(overlap_area_matrix + image_pixel_location_overlap[start],length,MPI_DOUBLE,id,sub_comm);
	}
	MPI_Comm_free(&sub_comm);
#endif

	for (n=0; n < ntot; n++) {
		//img_j = n / image_pixel_grid->x_N;
		//img_i = n % image_pixel_grid->x_N;
		img_j = extended_mask_j[n];
		img_i = extended_mask_i[n];
		for (l=image_pixel_location_overlap[n]; l < image_pixel_location_overlap[n+1]; l++) {
			nsrc = overlap_matrix_index[l];
			j = nsrc / u_N;
			i = nsrc % u_N;
			mag_matrix[nsrc] += overlap_matrix[l];
			area_matrix[nsrc] += overlap_area_matrix[l];
			if ((image_pixel_grid->fit_to_data != NULL) and (lens->image_pixel_data->high_sn_pixel[img_i][img_j])) high_sn_area_matrix[nsrc] += overlap_area_matrix[l];
			cell[i][j]->overlap_pixel_n.push_back(n);
			if ((image_pixel_grid->fit_to_data==NULL) or (image_pixel_grid->fit_to_data[img_i][img_j]==true)) cell[i][j]->maps_to_image_window = true;
		}
	}

#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for finding source cell magnifications: " << wtime << endl;
	}
#endif

	for (nsrc=0; nsrc < ntot_src; nsrc++) {
		j = nsrc / u_N;
		i = nsrc % u_N;
		cell[i][j]->total_magnification = mag_matrix[nsrc] * image_pixel_grid->triangle_area / cell[i][j]->cell_area;
		cell[i][j]->avg_image_pixels_mapped = cell[i][j]->total_magnification * cell[i][j]->cell_area / image_pixel_grid->pixel_area;
		if (lens->n_image_prior) cell[i][j]->n_images = area_matrix[nsrc] / cell[i][j]->cell_area;

		if (area_matrix[nsrc] > cell[i][j]->cell_area) lens->total_srcgrid_overlap_area += cell[i][j]->cell_area;
		else lens->total_srcgrid_overlap_area += area_matrix[nsrc];
		if (image_pixel_grid->fit_to_data != NULL) {
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
	delete[] extended_mask_i;
	delete[] extended_mask_j;
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
			maxlevs[thread] = levels;
			#pragma omp for private(i,j,n) schedule(dynamic)
			for (n=0; n < ntot; n++) {
				j = n / u_N;
				i = n % u_N;
				if (cell[i][j]->cell != NULL) cell[i][j]->split_subcells(splitlevel,thread);
			}
		}
		for (i=0; i < nthreads; i++) if (maxlevs[i] > levels) levels = maxlevs[i];
	} else {
		int k,l,m;
		double overlap_area, weighted_overlap, triangle1_overlap, triangle2_overlap, triangle1_weight, triangle2_weight;
		SourcePixelGrid *subcell;
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
			maxlevs[thread] = levels;
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
					cell[i][j]->split_cells(2,2,thread);
					for (k=0; k < cell[i][j]->overlap_pixel_n.size(); k++) {
						nn = cell[i][j]->overlap_pixel_n[k];
						img_j = nn / image_pixel_grid->x_N;
						img_i = nn % image_pixel_grid->x_N;
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
								if ((weighted_overlap != 0) and ((image_pixel_grid->fit_to_data==NULL) or (image_pixel_grid->fit_to_data[img_i][img_j]==true))) subcell->maps_to_image_window = true;
								subcell->overlap_pixel_n.push_back(nn);
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
							subcell->avg_image_pixels_mapped = subcell->total_magnification * subcell->cell_area / image_pixel_grid->pixel_area;
							if (lens->n_image_prior) subcell->n_images /= subcell->cell_area;
						}
					}
				}
			}
		}
		for (i=0; i < nthreads; i++) if (maxlevs[i] > levels) levels = maxlevs[i];
	}
}

void SourcePixelGrid::split_subcells(const int splitlevel, const int thread)
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
		SourcePixelGrid *subcell;
		bool subgrid;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				subgrid = false;
				if ((cell[i][j]->total_magnification*cell[i][j]->cell_area/(lens->base_srcpixel_imgpixel_ratio*image_pixel_grid->pixel_area)) > lens->pixel_magnification_threshold) subgrid = true;

				if (subgrid) {
					cell[i][j]->split_cells(2,2,thread);
					for (k=0; k < cell[i][j]->overlap_pixel_n.size(); k++) {
						nn = cell[i][j]->overlap_pixel_n[k];
						img_j = nn / image_pixel_grid->x_N;
						img_i = nn % image_pixel_grid->x_N;
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
								if ((weighted_overlap != 0) and ((image_pixel_grid->fit_to_data==NULL) or (image_pixel_grid->fit_to_data[img_i][img_j]==true))) subcell->maps_to_image_window = true;
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
							subcell->avg_image_pixels_mapped = subcell->total_magnification * subcell->cell_area / image_pixel_grid->pixel_area;
							if (lens->n_image_prior) subcell->n_images /= subcell->cell_area;
						}
					}
				}
			}
		}
	}
}

bool SourcePixelGrid::assign_source_mapping_flags_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, vector<SourcePixelGrid*>& mapped_source_pixels, const int& thread)
{
	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_overlap(input_corner_pts,thread)==false) return false;

	bool image_pixel_maps_to_source_grid = false;
	bool inside;
	int i,j;
	for (j=jmin[thread]; j <= jmax[thread]; j++) {
		for (i=imin[thread]; i <= imax[thread]; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->subcell_assign_source_mapping_flags_overlap(input_corner_pts,twist_pt,twist_status,mapped_source_pixels,thread,image_pixel_maps_to_source_grid);
			else {
				if (!cell[i][j]->check_if_in_neighborhood(input_corner_pts,inside,thread)) continue;
				if ((inside) or (cell[i][j]->check_overlap(input_corner_pts,twist_pt,twist_status,thread))) {
					cell[i][j]->maps_to_image_pixel = true;
					mapped_source_pixels.push_back(cell[i][j]);
					//if ((image_pixel_i==41) and (image_pixel_j==11)) cout << "mapped cell: " << cell[i][j]->center_pt[0] << " " << cell[i][j]->center_pt[1] << endl;
					if (!image_pixel_maps_to_source_grid) image_pixel_maps_to_source_grid = true;
				}
			}
		}
	}
	return image_pixel_maps_to_source_grid;
}

void SourcePixelGrid::subcell_assign_source_mapping_flags_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, vector<SourcePixelGrid*>& mapped_source_pixels, const int& thread, bool& image_pixel_maps_to_source_grid)
{
	bool inside;
	int i,j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->subcell_assign_source_mapping_flags_overlap(input_corner_pts,twist_pt,twist_status,mapped_source_pixels,thread,image_pixel_maps_to_source_grid);
			else {
				if (!cell[i][j]->check_if_in_neighborhood(input_corner_pts,inside,thread)) continue;
				if ((inside) or (cell[i][j]->check_overlap(input_corner_pts,twist_pt,twist_status,thread))) {
					cell[i][j]->maps_to_image_pixel = true;
					mapped_source_pixels.push_back(cell[i][j]);
					if (!image_pixel_maps_to_source_grid) image_pixel_maps_to_source_grid = true;
				}
			}
		}
	}
}

void SourcePixelGrid::calculate_Lmatrix_overlap(const int &img_index, const int &image_pixel_i, const int &image_pixel_j, int& index, lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread)
{
	double overlap, total_overlap=0;
	int i,j,k;
	int Lmatrix_index_initial = index;
	SourcePixelGrid *subcell;

	for (i=0; i < image_pixel_grid->mapped_source_pixels[image_pixel_i][image_pixel_j].size(); i++) {
		subcell = image_pixel_grid->mapped_source_pixels[image_pixel_i][image_pixel_j][i];
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

void SourcePixelGrid::find_lensed_surface_brightness_subcell_overlap(lensvector **input_corner_pts, lensvector *twist_pt, int& twist_status, const int& thread, double& overlap, double& total_overlap, double& total_weighted_surface_brightness)
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

bool SourcePixelGrid::assign_source_mapping_flags_interpolate(lensvector &input_center_pt, vector<SourcePixelGrid*>& mapped_source_pixels, const int& thread, const int& image_pixel_i, const int& image_pixel_j)
{
	// when splitting image pixels, there could be multiple entries in the Lmatrix array that belong to the same source pixel; you might save computational time if these can be consolidated (by adding them together). Try this out later
	imin[thread]=0; imax[thread]=u_N-1;
	jmin[thread]=0; jmax[thread]=w_N-1;
	if (bisection_search_interpolate(input_center_pt,thread)==false) return false;

	bool image_pixel_maps_to_source_grid = false;
	int i,j,side;
	SourcePixelGrid* cellptr;
	int oldsize = mapped_source_pixels.size();
	for (j=jmin[thread]; j <= jmax[thread]; j++) {
		for (i=imin[thread]; i <= imax[thread]; i++) {
			if ((input_center_pt[0] >= cell[i][j]->corner_pt[0][0]) and (input_center_pt[0] < cell[i][j]->corner_pt[2][0]) and (input_center_pt[1] >= cell[i][j]->corner_pt[0][1]) and (input_center_pt[1] < cell[i][j]->corner_pt[3][1])) {
				if (cell[i][j]->cell != NULL) image_pixel_maps_to_source_grid = cell[i][j]->subcell_assign_source_mapping_flags_interpolate(input_center_pt,mapped_source_pixels,thread);
				else {
					cell[i][j]->maps_to_image_pixel = true;
					mapped_source_pixels.push_back(cell[i][j]);
					if (!image_pixel_maps_to_source_grid) image_pixel_maps_to_source_grid = true;
					if (((input_center_pt[0] > cell[i][j]->center_pt[0]) and (cell[i][j]->neighbor[0] != NULL)) or (cell[i][j]->neighbor[1] == NULL)) {
						if (cell[i][j]->neighbor[0]->cell != NULL) {
							side=0;
							cellptr = cell[i][j]->neighbor[0]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[0]->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cell[i][j]->neighbor[0]);
						}
					} else {
						if (cell[i][j]->neighbor[1]->cell != NULL) {
							side=1;
							cellptr = cell[i][j]->neighbor[1]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[1]->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cell[i][j]->neighbor[1]);
						}
					}
					if (((input_center_pt[1] > cell[i][j]->center_pt[1]) and (cell[i][j]->neighbor[2] != NULL)) or (cell[i][j]->neighbor[3] == NULL)) {
						if (cell[i][j]->neighbor[2]->cell != NULL) {
							side=2;
							cellptr = cell[i][j]->neighbor[2]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[2]->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cell[i][j]->neighbor[2]);
						}
					} else {
						if (cell[i][j]->neighbor[3]->cell != NULL) {
							side=3;
							cellptr = cell[i][j]->neighbor[3]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[3]->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cell[i][j]->neighbor[3]);
						}
					}
				}
				break;
			}
		}
	}
	if ((mapped_source_pixels.size() - oldsize) != 3) die("Did not assign enough interpolation cells!");
	return image_pixel_maps_to_source_grid;
}

bool SourcePixelGrid::subcell_assign_source_mapping_flags_interpolate(lensvector &input_center_pt, vector<SourcePixelGrid*>& mapped_source_pixels, const int& thread)
{
	bool image_pixel_maps_to_source_grid = false;
	int i,j,side;
	SourcePixelGrid* cellptr;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if ((input_center_pt[0] >= cell[i][j]->corner_pt[0][0]) and (input_center_pt[0] < cell[i][j]->corner_pt[2][0]) and (input_center_pt[1] >= cell[i][j]->corner_pt[0][1]) and (input_center_pt[1] < cell[i][j]->corner_pt[3][1])) {
				if (cell[i][j]->cell != NULL) image_pixel_maps_to_source_grid = cell[i][j]->subcell_assign_source_mapping_flags_interpolate(input_center_pt,mapped_source_pixels,thread);
				else {
					cell[i][j]->maps_to_image_pixel = true;
					mapped_source_pixels.push_back(cell[i][j]);
					if (!image_pixel_maps_to_source_grid) image_pixel_maps_to_source_grid = true;
					if (((input_center_pt[0] > cell[i][j]->center_pt[0]) and (cell[i][j]->neighbor[0] != NULL)) or (cell[i][j]->neighbor[1] == NULL)) {
						if (cell[i][j]->neighbor[0]->cell != NULL) {
							side=0;
							cellptr = cell[i][j]->neighbor[0]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[0]->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cell[i][j]->neighbor[0]);
						}
					} else {
						if (cell[i][j]->neighbor[1]->cell != NULL) {
							side=1;
							cellptr = cell[i][j]->neighbor[1]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[1]->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cell[i][j]->neighbor[1]);
						}
					}
					if (((input_center_pt[1] > cell[i][j]->center_pt[1]) and (cell[i][j]->neighbor[2] != NULL)) or (cell[i][j]->neighbor[3] == NULL)) {
						if (cell[i][j]->neighbor[2]->cell != NULL) {
							side=2;
							cellptr = cell[i][j]->neighbor[2]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[2]->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cell[i][j]->neighbor[2]);
						}
					} else {
						if (cell[i][j]->neighbor[3]->cell != NULL) {
							side=3;
							cellptr = cell[i][j]->neighbor[3]->find_nearest_neighbor_cell(input_center_pt,side);
							cellptr->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cellptr);
						}
						else {
							cell[i][j]->neighbor[3]->maps_to_image_pixel = true;
							mapped_source_pixels.push_back(cell[i][j]->neighbor[3]);
						}
					}
				}
				break;
			}
		}
	}
	return image_pixel_maps_to_source_grid;
}

void SourcePixelGrid::calculate_Lmatrix_interpolate(const int img_index, const int image_pixel_i, const int image_pixel_j, int& index, lensvector &input_center_pt, const int& ii, const double weight, const int& thread)
{
	for (int i=0; i < 3; i++) {
		lens->Lmatrix_index_rows[img_index].push_back(image_pixel_grid->mapped_source_pixels[image_pixel_i][image_pixel_j][3*ii+i]->active_index);
		interpolation_pts[i][thread] = &image_pixel_grid->mapped_source_pixels[image_pixel_i][image_pixel_j][3*ii+i]->center_pt;
	}

	if (lens->interpolate_sb_3pt) {
		double d = ((*interpolation_pts[0][thread])[0]-(*interpolation_pts[1][thread])[0])*((*interpolation_pts[1][thread])[1]-(*interpolation_pts[2][thread])[1]) - ((*interpolation_pts[1][thread])[0]-(*interpolation_pts[2][thread])[0])*((*interpolation_pts[0][thread])[1]-(*interpolation_pts[1][thread])[1]);
		lens->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[1][thread])[1]-(*interpolation_pts[2][thread])[1]) + input_center_pt[1]*((*interpolation_pts[2][thread])[0]-(*interpolation_pts[1][thread])[0]) + (*interpolation_pts[1][thread])[0]*(*interpolation_pts[2][thread])[1] - (*interpolation_pts[1][thread])[1]*(*interpolation_pts[2][thread])[0])/d);
		lens->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[2][thread])[1]-(*interpolation_pts[0][thread])[1]) + input_center_pt[1]*((*interpolation_pts[0][thread])[0]-(*interpolation_pts[2][thread])[0]) + (*interpolation_pts[0][thread])[1]*(*interpolation_pts[2][thread])[0] - (*interpolation_pts[0][thread])[0]*(*interpolation_pts[2][thread])[1])/d);
		lens->Lmatrix_rows[img_index].push_back(weight*(input_center_pt[0]*((*interpolation_pts[0][thread])[1]-(*interpolation_pts[1][thread])[1]) + input_center_pt[1]*((*interpolation_pts[1][thread])[0]-(*interpolation_pts[0][thread])[0]) + (*interpolation_pts[0][thread])[0]*(*interpolation_pts[1][thread])[1] - (*interpolation_pts[0][thread])[1]*(*interpolation_pts[1][thread])[0])/d);
		if (d==0) warn("d is zero!!!");
	} else {
		lens->Lmatrix_rows[img_index].push_back(weight);
		lens->Lmatrix_rows[img_index].push_back(0);
		lens->Lmatrix_rows[img_index].push_back(0);
	}

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

void SourcePixelGrid::find_interpolation_cells(lensvector &input_center_pt, const int& thread)
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

SourcePixelGrid* SourcePixelGrid::find_nearest_neighbor_cell(lensvector &input_center_pt, const int& side)
{
	int i,ncells;
	SourcePixelGrid **cells;
	if ((side==0) or (side==1)) ncells = w_N;
	else if ((side==2) or (side==3)) ncells = u_N;
	else die("side number cannot be larger than 3");
	cells = new SourcePixelGrid*[ncells];

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
	SourcePixelGrid *closest_cell = cells[i_min];
	delete[] cells;
	return closest_cell;
}

SourcePixelGrid* SourcePixelGrid::find_nearest_neighbor_cell(lensvector &input_center_pt, const int& side, const int tiebreaker_side)
{
	int i,ncells;
	SourcePixelGrid **cells;
	if ((side==0) or (side==1)) ncells = w_N;
	else if ((side==2) or (side==3)) ncells = u_N;
	else die("side number cannot be larger than 3");
	cells = new SourcePixelGrid*[ncells];
	double sqr_distance, min_sqr_distance = 1e30;
	SourcePixelGrid *closest_cell = NULL;
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

void SourcePixelGrid::find_nearest_two_cells(SourcePixelGrid* &cellptr1, SourcePixelGrid* &cellptr2, const int& side)
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

SourcePixelGrid* SourcePixelGrid::find_corner_cell(const int i, const int j)
{
	SourcePixelGrid* cellptr = cell[i][j];
	while (cellptr->cell != NULL)
		cellptr = cellptr->cell[i][j];
	return cellptr;
}

void SourcePixelGrid::generate_gmatrices()
{
	int i,j,k,l;
	SourcePixelGrid *cellptr1, *cellptr2;
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

void SourcePixelGrid::generate_hmatrices()
{
	int i,j,k,l,m,kmin,kmax;
	SourcePixelGrid *cellptr1, *cellptr2;
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

void QLens::generate_Rmatrix_from_hmatrices()
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

	Rmatrix_diags = new double[source_npixels];
	Rmatrix_rows = new vector<double>[source_npixels];
	Rmatrix_index_rows = new vector<int>[source_npixels];
	Rmatrix_row_nn = new int[source_npixels];
	Rmatrix_nn = 0;
	int Rmatrix_nn_part = 0;
	for (j=0; j < source_npixels; j++) {
		Rmatrix_diags[j] = 0;
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
	source_pixel_grid->generate_hmatrices();

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
				element = hmatrix[k][j]*hmatrix[k][l]; // generalize this to full covariance matrix later
				if (src_index1==src_index2) Rmatrix_diags[src_index1] += element;
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
		Rmatrix[i] = Rmatrix_diags[i];

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
	delete[] Rmatrix_diags;
	delete[] Rmatrix_rows;
	delete[] Rmatrix_index_rows;

	for (i=0; i < 2; i++) {
		delete[] jvals[i];
		delete[] lvals[i];
	}
}

void QLens::generate_Rmatrix_from_gmatrices()
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

	Rmatrix_diags = new double[source_npixels];
	Rmatrix_rows = new vector<double>[source_npixels];
	Rmatrix_index_rows = new vector<int>[source_npixels];
	Rmatrix_row_nn = new int[source_npixels];
	Rmatrix_nn = 0;
	int Rmatrix_nn_part = 0;
	for (j=0; j < source_npixels; j++) {
		Rmatrix_diags[j] = 0;
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
	source_pixel_grid->generate_gmatrices();

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
				element = gmatrix[k][j]*gmatrix[k][l]; // generalize this to full covariance matrix later
				if (src_index1==src_index2) Rmatrix_diags[src_index1] += element;
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
		Rmatrix[i] = Rmatrix_diags[i];

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
	delete[] Rmatrix_diags;
	delete[] Rmatrix_rows;
	delete[] Rmatrix_index_rows;

	for (i=0; i < 4; i++) {
		delete[] jvals[i];
		delete[] lvals[i];
	}
}

int SourcePixelGrid::assign_indices_and_count_levels()
{
	levels=1; // we are going to recount the number of levels
	int source_pixel_i=0;
	assign_indices(source_pixel_i);
	return source_pixel_i;
}

void SourcePixelGrid::assign_indices(int& source_pixel_i)
{
	if (levels < level+1) levels=level+1;
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

ofstream SourcePixelGrid::index_out;

void SourcePixelGrid::print_indices()
{
	int i, j;
	for (j=0; j < w_N; j++) {
		for (i=0; i < u_N; i++) {
			if (cell[i][j]->cell != NULL) cell[i][j]->print_indices();
			else {
				index_out << cell[i][j]->index << " " << cell[i][j]->active_index << " level=" << cell[i][j]->level << endl;
			}
		}
	}
}

bool SourcePixelGrid::regrid_if_unmapped_source_subcells;
bool SourcePixelGrid::activate_unmapped_source_pixels;
bool SourcePixelGrid::exclude_source_pixels_outside_fit_window;

int SourcePixelGrid::assign_active_indices_and_count_source_pixels(bool regrid_if_inactive_cells, bool activate_unmapped_pixels, bool exclude_pixels_outside_window)
{
	regrid_if_unmapped_source_subcells = regrid_if_inactive_cells;
	activate_unmapped_source_pixels = activate_unmapped_pixels;
	exclude_source_pixels_outside_fit_window = exclude_pixels_outside_window;
	int source_pixel_i=0;
	assign_active_indices(source_pixel_i);
	return source_pixel_i;
}

ofstream SourcePixelGrid::missed_cells_out; // remove later?

void SourcePixelGrid::assign_active_indices(int& source_pixel_i)
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
					if ((activate_unmapped_source_pixels) and ((!regrid_if_unmapped_source_subcells) or (level==0))) { // if we are removing unmapped subpixels, we may still want to activate first-level unmapped pixels
						if ((exclude_source_pixels_outside_fit_window) and (cell[i][j]->maps_to_image_window==false)) ;
						else {
							cell[i][j]->active_index = source_pixel_i++;
							cell[i][j]->active_pixel = true;
						}
					} else {
						cell[i][j]->active_pixel = false;
						if ((regrid_if_unmapped_source_subcells) and (level >= 1)) {
							if (!regrid) regrid = true;
							unsplit_cell = true;
						}
						//missed_cells_out << cell[i][j]->center_pt[0] << " " << cell[i][j]->center_pt[1] << endl;
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

void SourcePixelGrid::clear()
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

void SourcePixelGrid::clear_subgrids()
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
		number_of_pixels -= (u_N*w_N - 1);
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

/******************************** Functions in class ImagePixelData, and FITS file functions *********************************/

void ImagePixelData::load_data(string root)
{
	string sbfilename = root + ".dat";
	string xfilename = root + ".x";
	string yfilename = root + ".y";

	int i,j;
	double dummy;
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
	if (require_fit != NULL) {
		for (i=0; i < npixels_x; i++) delete[] require_fit[i];
		delete[] require_fit;
	}
	if (extended_mask != NULL) {
		for (i=0; i < npixels_x; i++) delete[] extended_mask[i];
		delete[] extended_mask;
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

	n_required_pixels = npixels_x*npixels_y;
	n_high_sn_pixels = n_required_pixels; // this will be recalculated in assign_high_sn_pixels() function
	xvals = new double[npixels_x+1];
	xfile.open(xfilename.c_str());
	for (i=0; i <= npixels_x; i++) xfile >> xvals[i];
	yvals = new double[npixels_y+1];
	yfile.open(yfilename.c_str());
	for (i=0; i <= npixels_y; i++) yfile >> yvals[i];

	ifstream sbfile(sbfilename.c_str());
	surface_brightness = new double*[npixels_x];
	high_sn_pixel = new bool*[npixels_x];
	require_fit = new bool*[npixels_x];
	extended_mask = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) {
		surface_brightness[i] = new double[npixels_y];
		high_sn_pixel[i] = new bool[npixels_y];
		require_fit[i] = new bool[npixels_y];
		extended_mask[i] = new bool[npixels_y];
		for (j=0; j < npixels_y; j++) {
			require_fit[i][j] = true;
			extended_mask[i][j] = true;
			high_sn_pixel[i][j] = true;
			sbfile >> surface_brightness[i][j];
		}
	}
	assign_high_sn_pixels();
}

void ImagePixelData::load_from_image_grid()
{
	int i,j;
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
	if (require_fit != NULL) {
		for (i=0; i < npixels_x; i++) delete[] require_fit[i];
		delete[] require_fit;
	}
	if (extended_mask != NULL) {
		for (i=0; i < npixels_x; i++) delete[] extended_mask[i];
		delete[] extended_mask;
	}

	npixels_x = lens->image_pixel_grid->x_N;
	npixels_y = lens->image_pixel_grid->y_N;

	n_required_pixels = npixels_x*npixels_y;
	n_high_sn_pixels = n_required_pixels; // this will be recalculated in assign_high_sn_pixels() function
	xvals = new double[npixels_x+1];
	for (i=0; i <= npixels_x; i++) xvals[i] = lens->image_pixel_grid->corner_pts[i][0][0];
	yvals = new double[npixels_y+1];
	for (i=0; i <= npixels_y; i++) yvals[i] = lens->image_pixel_grid->corner_pts[0][i][1];

	surface_brightness = new double*[npixels_x];
	high_sn_pixel = new bool*[npixels_x];
	require_fit = new bool*[npixels_x];
	extended_mask = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) {
		surface_brightness[i] = new double[npixels_y];
		high_sn_pixel[i] = new bool[npixels_y];
		require_fit[i] = new bool[npixels_y];
		extended_mask[i] = new bool[npixels_y];
		for (j=0; j < npixels_y; j++) {
			require_fit[i][j] = true;
			extended_mask[i][j] = true;
			high_sn_pixel[i][j] = true;
			surface_brightness[i][j] = lens->image_pixel_grid->surface_brightness[i][j];
		}
	}
	assign_high_sn_pixels();
}

bool ImagePixelData::load_data_fits(bool use_pixel_size, string fits_filename)
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
	double x, y, xstep, ystep;

	int hdutype;
	if (!fits_open_file(&fptr, fits_filename.c_str(), READONLY, &status))
	{
		 //if ( fits_movabs_hdu(fptr, 2, &hdutype, &status) ) /* move to 2nd HDU */
			//die("fuck");

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
		if (require_fit != NULL) {
			for (i=0; i < npixels_x; i++) delete[] require_fit[i];
			delete[] require_fit;
		}
		if (extended_mask != NULL) {
			for (i=0; i < npixels_x; i++) delete[] extended_mask[i];
			delete[] extended_mask;
		}

		int nkeys;
		fits_get_hdrspace(fptr, &nkeys, NULL, &status); // get # of keywords

		int ii;
		char card[FLEN_CARD];   // Standard string lengths defined in fitsio.h

		bool reading_qlens_comment = false;
		bool reading_markers = false;
		int pos, pos1;
		for (ii = 1; ii <= nkeys; ii++) { // Read and print each keywords 
			if (fits_read_record(fptr, ii, card, &status))break;
			// When you get time: put pixel size and pixel noise as lines in the FITS file comment! Then have it load them here so you don't need to specify them as separate lines.
			string cardstring(card);
			if (reading_qlens_comment) {
				if ((pos = cardstring.find("COMMENT")) != string::npos) {
					if (((pos1 = cardstring.find("mk: ")) != string::npos) or ((pos1 = cardstring.find("MK: ")) != string::npos)) {
						reading_markers = true;
						reading_qlens_comment = false;
						lens->param_markers = cardstring.substr(pos1+4);
					} else lens->data_info += cardstring.substr(pos+8);
				} else break;
			} else if (reading_markers) {
				if ((pos = cardstring.find("COMMENT")) != string::npos) {
					lens->param_markers += cardstring.substr(pos+8);
					// A potential issue is that if there are enough markers to fill more than one line, there might be an extra space inserted,
					// in which case the markers won't come out properly. No time to deal with this now, but something to look out for.
				} else break;
			} else if (((pos = cardstring.find("ql: ")) != string::npos) or ((pos = cardstring.find("QL: ")) != string::npos)) {
				reading_qlens_comment = true;
				lens->data_info = cardstring.substr(pos+4);
			} else if (((pos = cardstring.find("mk: ")) != string::npos) or ((pos = cardstring.find("MK: ")) != string::npos)) {
				reading_markers = true;
				lens->param_markers = cardstring.substr(pos+4);
			} else if (cardstring.find("PXSIZE ") != string::npos) {
				string pxsize_string = cardstring.substr(11);
				stringstream pxsize_str;
				pxsize_str << pxsize_string;
				pxsize_str >> pixel_size;
				lens->data_pixel_size = pixel_size;
				if (!use_pixel_size) use_pixel_size = true;
			} else if (cardstring.find("PXNOISE ") != string::npos) {
				string pxnoise_string = cardstring.substr(11);
				stringstream pxnoise_str;
				pxnoise_str << pxnoise_string;
				pxnoise_str >> lens->data_pixel_noise;
			} else if (cardstring.find("PSFSIG ") != string::npos) {
				string psfwidth_string = cardstring.substr(11);
				stringstream psfwidth_str;
				psfwidth_str << psfwidth_string;
				psfwidth_str >> lens->psf_width_x;
				lens->psf_width_y = lens->psf_width_x;
			} else if (cardstring.find("ZSRC ") != string::npos) {
				string zsrc_string = cardstring.substr(11);
				stringstream zsrc_str;
				zsrc_str << zsrc_string;
				zsrc_str >> lens->source_redshift;
			} else if (cardstring.find("ZLENS ") != string::npos) {
				string zlens_string = cardstring.substr(11);
				stringstream zlens_str;
				zlens_str << zlens_string;
				zlens_str >> lens->lens_redshift;
			}
		}
		if (reading_markers) {
			// Commas are used in FITS file as delimeter so spaces don't get lost; now convert to spaces again
			for (size_t i = 0; i < lens->param_markers.size(); ++i) {
				 if (lens->param_markers[i] == ',') {
					  lens->param_markers.replace(i, 1, " ");
				 }
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
				npixels_x = naxes[0];
				npixels_y = naxes[1];
				n_required_pixels = npixels_x*npixels_y;
				n_high_sn_pixels = n_required_pixels; // this will be recalculated in assign_high_sn_pixels() function
				xvals = new double[npixels_x+1];
				yvals = new double[npixels_y+1];
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
				pixels = new double[npixels_x];
				surface_brightness = new double*[npixels_x];
				high_sn_pixel = new bool*[npixels_x];
				require_fit = new bool*[npixels_x];
				extended_mask = new bool*[npixels_x];
				for (i=0; i < npixels_x; i++) {
					surface_brightness[i] = new double[npixels_y];
					high_sn_pixel[i] = new bool[npixels_y];
					require_fit[i] = new bool[npixels_y];
					extended_mask[i] = new bool[npixels_y];
					for (j=0; j < npixels_y; j++) {
						require_fit[i][j] = true;
						extended_mask[i][j] = true;
						high_sn_pixel[i][j] = true;
					}
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
	}

	if (status) fits_report_error(stderr, status); // print any error message
	if (image_load_status) assign_high_sn_pixels();
	return image_load_status;
#endif
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

void ImagePixelData::assign_high_sn_pixels()
{
	global_max_sb = -1e30;
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


bool ImagePixelData::load_mask_fits(string fits_filename)
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
	double x, y, xstep, ystep;
	int n_maskpixels = 0;

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
				if ((naxes[0] != npixels_x) or (naxes[1] != npixels_y)) { cout << "Error: number of pixels in mask file does not match number of pixels in loaded data\n"; return false; }
				pixels = new double[npixels_x];
				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					if (fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0], NULL, pixels, NULL, &status) )  // read row of pixels
						break; // jump out of loop on error

					for (i=0; i < naxes[0]; i++) {
						if (pixels[i] == 0.0) require_fit[i][j] = false;
						else {
							require_fit[i][j] = true;
							n_maskpixels++;
						//cout << pixels[i] << endl;
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
	if (image_load_status) n_required_pixels = n_maskpixels;
	set_extended_mask(lens->extended_mask_n_neighbors);
	return image_load_status;
#endif
}

bool ImagePixelData::save_mask_fits(string fits_filename)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to write FITS files\n"; return false;
#else
	int i,j,kk;
	fitsfile *outfptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix = -64, naxis = 2;
	long naxes[2] = {npixels_x,npixels_y};
	double *pixels;
	double x, y, xstep, ystep;
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
				pixels = new double[npixels_x];

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=0; i < npixels_x; i++) {
						if (require_fit[i][j]) pixels[i] = 1.0;
						else pixels[i] = 0.0;
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

bool QLens::load_psf_fits(string fits_filename, const bool verbal)
{
#ifndef USE_FITS
	cout << "FITS capability disabled; QLens must be compiled with the CFITSIO library to read FITS files\n"; return false;
#else
	use_input_psf_matrix = true;
	bool image_load_status = false;
	int i,j,kk;
	if (psf_matrix != NULL) {
		for (i=0; i < psf_npixels_x; i++) delete[] psf_matrix[i];
		delete[] psf_matrix;
		psf_matrix = NULL;
	}
	double **input_psf_matrix;

	fitsfile *fptr;   // FITS file pointer, defined in fitsio.h
	int status = 0;   // CFITSIO status value MUST be initialized to zero!
	int bitpix, naxis;
	int nx, ny;
	long naxes[2] = {1,1};
	double *pixels;
	double x, y, xstep, ystep;
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
			if (input_psf_matrix[i][j] > psf_threshold*peak_sb) {
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
	psf_npixels_x = 2*nx_half+1;
	psf_npixels_y = 2*ny_half+1;
	psf_matrix = new double*[psf_npixels_x];
	for (i=0; i < psf_npixels_x; i++) psf_matrix[i] = new double[psf_npixels_y];
	int ii,jj;
	for (ii=0, i=imid-nx_half; ii < psf_npixels_x; i++, ii++) {
		for (jj=0, j=jmid-ny_half; jj < psf_npixels_y; j++, jj++) {
			psf_matrix[ii][jj] = input_psf_matrix[i][j];
		}
	}
	double normalization = 0;
	for (i=0; i < psf_npixels_x; i++) {
		for (j=0; j < psf_npixels_y; j++) {
			normalization += psf_matrix[i][j];
		}
	}
	for (i=0; i < psf_npixels_x; i++) {
		for (j=0; j < psf_npixels_y; j++) {
			psf_matrix[i][j] /= normalization;
		}
	}
	//for (i=0; i < psf_npixels_x; i++) {
		//for (j=0; j < psf_npixels_y; j++) {
			//cout << psf_matrix[i][j] << " ";
		//}
		//cout << endl;
	//}
	//cout << psf_npixels_x << " " << psf_npixels_y << " " << nx_half << " " << ny_half << endl;

	if ((verbal) and (mpi_id==0)) cout << "PSF matrix dimensions: " << psf_npixels_x << " " << psf_npixels_y << " (input PSF dimensions: " << nx << " " << ny << ")" << endl << endl;
	for (i=0; i < nx; i++) delete[] input_psf_matrix[i];
	delete[] input_psf_matrix;

	if (status) fits_report_error(stderr, status); // print any error message
	return image_load_status;
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
	if (high_sn_pixel != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] high_sn_pixel[i];
		delete[] high_sn_pixel;
	}
	if (require_fit != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] require_fit[i];
		delete[] require_fit;
	}
	if (extended_mask != NULL) {
		for (int i=0; i < npixels_x; i++) delete[] extended_mask[i];
		delete[] extended_mask;
	}
}

void ImagePixelData::set_no_required_data_pixels()
{
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			require_fit[i][j] = false;
			extended_mask[i][j] = false;
		}
	}
	n_required_pixels = 0;
}

void ImagePixelData::set_all_required_data_pixels()
{
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			require_fit[i][j] = true;
			extended_mask[i][j] = true;
		}
	}
	n_required_pixels = npixels_x*npixels_y;
}

bool ImagePixelData::inside_mask(const double x, const double y)
{
	if ((x <= xmin) or (x >= xmax) or (y <= ymin) or (y >= ymax)) return false;
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if ((xvals[i] < x) and (xvals[i+1] >= x) and (yvals[j] < y) and (yvals[j+1] > y)) {
				if (require_fit[i][j]) return true;
			}
		}
	}
	return false;
}

void ImagePixelData::assign_mask_windows(const double sb_noise_threshold)
{
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
					if (require_fit[i][j]) {
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
								if ((i > 0) and (require_fit[i-1][j]) and (mask_window_id[i-1][j] == current_mask)) neighbor_mask = current_mask;
								else if ((i < npixels_x-1) and (require_fit[i+1][j]) and (mask_window_id[i+1][j] == current_mask)) neighbor_mask = current_mask;
								else if ((j > 0) and (require_fit[i][j-1]) and (mask_window_id[i][j-1] == current_mask)) neighbor_mask = current_mask;
								else if ((j < npixels_y-1) and (require_fit[i][j+1]) and (mask_window_id[i][j+1] == current_mask)) neighbor_mask = current_mask;
								if (neighbor_mask == current_mask) {
									mask_window_id[i][j] = neighbor_mask;
									mask_window_sizes[neighbor_mask]++;
									if (surface_brightness[i][j] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i][j];
									new_mask_member = true;
								}
							}
							if (mask_window_id[i][j] == current_mask) {
								this_window_id = mask_window_id[i][j];
								if ((i > 0) and (require_fit[i-1][j]) and (mask_window_id[i-1][j] < 0)) {
									mask_window_id[i-1][j] = this_window_id;
									mask_window_sizes[this_window_id]++;
									if (surface_brightness[i-1][j] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i-1][j];
									new_mask_member = true;
								}
								if ((i < npixels_x-1) and (require_fit[i+1][j]) and (mask_window_id[i+1][j] < 0)) {
									mask_window_id[i+1][j] = this_window_id;
									mask_window_sizes[this_window_id]++;
									if (surface_brightness[i+1][j] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i+1][j];
									new_mask_member = true;
								}
								if ((j > 0) and (require_fit[i][j-1]) and (mask_window_id[i][j-1] < 0)) {
									mask_window_id[i][j-1] = this_window_id;
									mask_window_sizes[this_window_id]++;
									if (surface_brightness[i][j-1] > mask_window_max_sb[current_mask]) mask_window_max_sb[current_mask] = surface_brightness[i][j-1];
									new_mask_member = true;
								}
								if ((j < npixels_y-1) and (require_fit[i][j+1]) and (mask_window_id[i][j+1] < 0)) {
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
				if (mask_window_max_sb[l] > sb_noise_threshold*lens->data_pixel_noise) {
					active_mask_window[l] = false; // ensures it won't get cut
					//n_windows_eff--;
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
						require_fit[i][j] = false;
					}
				}
			}
			active_mask_window[smallest_window_id] = false;
			mask_window_sizes[smallest_window_id] = 0;
			n_windows_eff--;
		}
		//cout << "HI " << n_windows_eff << endl;
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
	set_extended_mask(lens->extended_mask_n_neighbors);
}

void ImagePixelData::unset_low_signal_pixels(const double sb_threshold, const bool use_fit)
{
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (use_fit) {
				if ((lens->image_pixel_grid->maps_to_source_pixel[i][j]) and (lens->image_pixel_grid->surface_brightness[i][j] < sb_threshold)) {
					if (require_fit[i][j]) {
						require_fit[i][j] = false;
						n_required_pixels--;
					}
				}
			} else {
				if (surface_brightness[i][j] < sb_threshold) {
					if (require_fit[i][j]) {
						require_fit[i][j] = false;
						n_required_pixels--;
					}
				}
			}
		}
	}

	// now we will deactivate pixels that have 0 or 1 neighboring active pixels (to get rid of isolated bits)
	bool **req = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) req[i] = new bool[npixels_y];
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			req[i][j] = require_fit[i][j];
		}
	}
	int n_active_neighbors;
	for (int k=0; k < 3; k++) {
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if (require_fit[i][j]) {
					n_active_neighbors = 0;
					if ((i < npixels_x-1) and (require_fit[i+1][j])) n_active_neighbors++;
					if ((i > 0) and (require_fit[i-1][j])) n_active_neighbors++;
					if ((j < npixels_y-1) and (require_fit[i][j+1])) n_active_neighbors++;
					if ((j > 0) and (require_fit[i][j-1])) n_active_neighbors++;
					if ((n_active_neighbors < 2) and (req[i][j])) {
						req[i][j] = false;
						n_required_pixels--;
					}
				}
			}
		}
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				require_fit[i][j] = req[i][j];
			}
		}
	}
	for (i=0; i < npixels_x; i++) delete[] req[i];
	delete[] req;
	set_extended_mask(lens->extended_mask_n_neighbors);
}

void ImagePixelData::set_nearest_neighbor_pixels()
{
	int i,j;
	bool **req = new bool*[npixels_x];
	for (i=0; i < npixels_x; i++) req[i] = new bool[npixels_y];
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			req[i][j] = require_fit[i][j];
		}
	}
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (require_fit[i][j]) {
				if ((i < npixels_x-1) and (!require_fit[i+1][j])) {
					if (!req[i+1][j]) {
						req[i+1][j] = true;
						n_required_pixels++;
					}
				}
				if ((i > 0) and (!require_fit[i-1][j])) {
					if (!req[i-1][j]) {
						req[i-1][j] = true;
						n_required_pixels++;
					}
				}
				if ((j < npixels_y-1) and (!require_fit[i][j+1])) {
					if (!req[i][j+1]) {
						req[i][j+1] = true;
						n_required_pixels++;
					}
				}
				if ((j > 0) and (!require_fit[i][j-1])) {
					if (!req[i][j-1]) {
						req[i][j-1] = true;
						n_required_pixels++;
					}
				}
			}
		}
	}
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			require_fit[i][j] = req[i][j];
		}
	}
	// check for any lingering "holes" in the mask and activate them
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if (!require_fit[i][j]) {
				if (((i < npixels_x-1) and (require_fit[i+1][j])) and ((i > 0) and (require_fit[i-1][j])) and ((j < npixels_y-1) and (require_fit[i][j+1])) and ((j > 0) and (require_fit[i][j-1]))) {
					if (!req[i][j]) {
						req[i][j] = true;
						n_required_pixels++;
					}
				}
			}
		}
	}
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			require_fit[i][j] = req[i][j];
		}
	}
	for (i=0; i < npixels_x; i++) delete[] req[i];
	delete[] req;
	set_extended_mask(lens->extended_mask_n_neighbors);
}

void ImagePixelData::set_required_data_window(const double xmin, const double xmax, const double ymin, const double ymax, const bool unset)
{
	int i,j;
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			if ((xvals[i] > xmin) and (xvals[i+1] < xmax) and (yvals[j] > ymin) and (yvals[j+1] < ymax)) {
				if (!unset) {
					if (require_fit[i][j] == false) {
						require_fit[i][j] = true;
						n_required_pixels++;
					}
				} else {
					if (require_fit[i][j] == true) {
						require_fit[i][j] = false;
						n_required_pixels--;
					}
				}
			}
		}
	}
	set_extended_mask(lens->extended_mask_n_neighbors);
}

void ImagePixelData::set_required_data_annulus(const double xc, const double yc, const double rmin, const double rmax, double theta1_deg, double theta2_deg, const double xstretch, const double ystretch, const bool unset)
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
				// allow for two possibilities: theta1 < theta2, and theta2 < theta1 (which can happen if, e.g. theta1 is input as negative and theta1 is input as positive)
				if (((theta2 > theta1) and (theta >= theta1) and (theta <= theta2)) or ((theta1 > theta2) and ((theta >= theta1) or (theta <= theta2)))) {
					if (!unset) {
						if (require_fit[i][j] == false) {
							require_fit[i][j] = true;
							n_required_pixels++;
						}
					} else {
						if (require_fit[i][j] == true) {
							require_fit[i][j] = false;
							n_required_pixels--;
						}
					}
				}
			}
		}
	}
	set_extended_mask(lens->extended_mask_n_neighbors);
}

void ImagePixelData::set_extended_mask(const int n_neighbors)
{
	// This is very similar to the set_neighbor_pixels() function in ImagePixelData; used here for the outside_sb_prior feature
	int i,j,k;
	//bool **req = new bool*[npixels_x];
	//for (i=0; i < npixels_x; i++) req[i] = new bool[npixels_y];
	if (n_neighbors < 0) {
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				extended_mask[i][j] = true;
			}
		}
		return;
	}
	for (i=0; i < npixels_x; i++) {
		for (j=0; j < npixels_y; j++) {
			extended_mask[i][j] = require_fit[i][j];
		}
	}
	for (k=0; k < n_neighbors; k++) {
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if (require_fit[i][j]) {
					if ((i < npixels_x-1) and (!require_fit[i+1][j])) {
						if (!extended_mask[i+1][j]) {
							extended_mask[i+1][j] = true;
							//n_required_pixels++;
						}
					}
					if ((i > 0) and (!require_fit[i-1][j])) {
						if (!extended_mask[i-1][j]) {
							extended_mask[i-1][j] = true;
							//n_required_pixels++;
						}
					}
					if ((j < npixels_y-1) and (!require_fit[i][j+1])) {
						if (!extended_mask[i][j+1]) {
							extended_mask[i][j+1] = true;
							//n_required_pixels++;
						}
					}
					if ((j > 0) and (!require_fit[i][j-1])) {
						if (!extended_mask[i][j-1]) {
							extended_mask[i][j-1] = true;
							//n_required_pixels++;
						}
					}
				}
			}
		}
		/*
		// check for any lingering "holes" in the mask and activate them
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				if (!require_fit[i][j]) {
					if (((i < npixels_x-1) and (require_fit[i+1][j])) and ((i > 0) and (require_fit[i-1][j])) and ((j < npixels_y-1) and (require_fit[i][j+1])) and ((j > 0) and (require_fit[i][j-1]))) {
						if (!extended_mask[i][j]) {
							extended_mask[i][j] = true;
							//n_required_pixels++;
						}
					}
				}
			}
		}
		for (i=0; i < npixels_x; i++) {
			for (j=0; j < npixels_y; j++) {
				require_fit[i][j] = extended_mask[i][j];
			}
		}
		*/
	}
	//for (i=0; i < npixels_x; i++) delete[] req[i];
	//delete[] req;
}

void ImagePixelData::estimate_pixel_noise(const double xmin, const double xmax, const double ymin, const double ymax, double &noise, double &mean_sb)
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
	double sigsq_sb=0, total_flux=0;
	int np=0;
	double xm,ym;
	for (j=jmin; j <= jmax; j++) {
		for (i=imin; i <= imax; i++) {
			if (require_fit[i][j]) {
				total_flux += surface_brightness[i][j];
				np++;
			}
		}
	}

	mean_sb = total_flux / np;
	for (j=jmin; j <= jmax; j++) {
		for (i=imin; i <= imax; i++) {
			if (require_fit[i][j]) {
				sigsq_sb += SQR(surface_brightness[i][j]-mean_sb);
			}
		}
	}
	double sqrnoise = sigsq_sb/np;
	double sigthreshold = 3.0;
	double sqrthreshold = SQR(sigthreshold)*sqrnoise;
	noise = sqrt(sqrnoise);
	cout << "noise=" << noise << endl;
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
				if (require_fit[i][j]) {
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
		cout << "noise=" << noise << ", mean=" << mean_sb << ", nclip=" << nclip << endl;
	} while (nclip > prev_nclip);

}

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

void ImagePixelData::plot_surface_brightness(string outfile_root, bool show_only_mask)
{
	string sb_filename = outfile_root + ".dat";
	string x_filename = outfile_root + ".x";
	string y_filename = outfile_root + ".y";

	ofstream pixel_image_file; lens->open_output_file(pixel_image_file,sb_filename);
	ofstream pixel_xvals; lens->open_output_file(pixel_xvals,x_filename);
	ofstream pixel_yvals; lens->open_output_file(pixel_yvals,y_filename);
	pixel_image_file << setiosflags(ios::scientific);
	int i,j;
	for (int i=0; i <= npixels_x; i++) {
		pixel_xvals << xvals[i] << endl;
	}
	for (int j=0; j <= npixels_y; j++) {
		pixel_yvals << yvals[j] << endl;
	}	
	for (j=0; j < npixels_y; j++) {
		for (i=0; i < npixels_x; i++) {
			if ((!show_only_mask) or (require_fit == NULL) or (require_fit[i][j])) {
				pixel_image_file << surface_brightness[i][j];
			} else {
				pixel_image_file << "NaN";
			}
			if (i < npixels_x-1) pixel_image_file << " ";
		}
		pixel_image_file << endl;
	}
}

/***************************************** Functions in class ImagePixelGrid ****************************************/

ImagePixelGrid::ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, double xmin_in, double xmax_in, double ymin_in, double ymax_in, int x_N_in, int y_N_in) : lens(lens_in), xmin(xmin_in), xmax(xmax_in), ymin(ymin_in), ymax(ymax_in), x_N(x_N_in), y_N(y_N_in)
{
	source_fit_mode = mode;
	ray_tracing_method = method;
	xy_N = x_N*y_N;
	if (source_fit_mode==Pixellated_Source) n_active_pixels = 0;
	else n_active_pixels = xy_N;
	corner_pts = new lensvector*[x_N+1];
	corner_sourcepts = new lensvector*[x_N+1];
	center_pts = new lensvector*[x_N];
	center_sourcepts = new lensvector*[x_N];
	maps_to_source_pixel = new bool*[x_N];
	pixel_index = new int*[x_N];
	mapped_source_pixels = new vector<SourcePixelGrid*>*[x_N];
	surface_brightness = new double*[x_N];
	source_plane_triangle1_area = new double*[x_N];
	source_plane_triangle2_area = new double*[x_N];
	max_nsplit = imax(24,lens_in->default_imgpixel_nsplit);
	nsplits = new int*[x_N];
	subpixel_maps_to_srcpixel = new bool**[x_N];
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
		pixel_index[i] = new int[y_N];
		surface_brightness[i] = new double[y_N];
		source_plane_triangle1_area[i] = new double[y_N];
		source_plane_triangle2_area[i] = new double[y_N];
		mapped_source_pixels[i] = new vector<SourcePixelGrid*>[y_N];
		subpixel_maps_to_srcpixel[i] = new bool*[y_N];
		nsplits[i] = new int[y_N];
		twist_pts[i] = new lensvector[y_N];
		twist_status[i] = new int[y_N];
		for (j=0; j < y_N; j++) {
			surface_brightness[i][j] = 0;
			if (source_fit_mode==Parameterized_Source) maps_to_source_pixel[i][j] = true; // in this mode you can always get a surface brightness for any image pixel
			if (lens_in->split_imgpixels) nsplits[i][j] = lens_in->default_imgpixel_nsplit; // default
			else nsplits[i][j] = 1;
			subpixel_maps_to_srcpixel[i][j] = new bool[max_nsplit*max_nsplit];
			for (k=0; k < max_nsplit*max_nsplit; k++) subpixel_maps_to_srcpixel[i][j][k] = false;
		}
	}
	imggrid_zfactors = lens->reference_zfactors;
	imggrid_betafactors = lens->default_zsrc_beta_factors;

	pixel_xlength = (xmax-xmin)/x_N;
	pixel_ylength = (ymax-ymin)/y_N;
	pixel_area = pixel_xlength*pixel_ylength;
	triangle_area = 0.5*pixel_xlength*pixel_ylength;

#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
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
		double x,y;
		lensvector d1,d2,d3,d4;
		#pragma omp for private(i,j) schedule(dynamic)
		for (j=0; j <= y_N; j++) {
			y = ymin + j*pixel_ylength;
			for (i=0; i <= x_N; i++) {
				x = xmin + i*pixel_xlength;
				if ((i < x_N) and (j < y_N)) {
					center_pts[i][j][0] = x + 0.5*pixel_xlength;
					center_pts[i][j][1] = y + 0.5*pixel_ylength;
					lens->find_sourcept(center_pts[i][j],center_sourcepts[i][j],thread,imggrid_zfactors,imggrid_betafactors);
				}
				corner_pts[i][j][0] = x;
				corner_pts[i][j][1] = y;
				lens->find_sourcept(corner_pts[i][j],corner_sourcepts[i][j],thread,imggrid_zfactors,imggrid_betafactors);
			}
		}
		//if (source_fit_mode==Pixellated_Source) {
			#pragma omp for private(i,j) schedule(dynamic)
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					d1[0] = corner_sourcepts[i][j][0] - corner_sourcepts[i+1][j][0];
					d1[1] = corner_sourcepts[i][j][1] - corner_sourcepts[i+1][j][1];
					d2[0] = corner_sourcepts[i][j+1][0] - corner_sourcepts[i][j][0];
					d2[1] = corner_sourcepts[i][j+1][1] - corner_sourcepts[i][j][1];
					d3[0] = corner_sourcepts[i+1][j+1][0] - corner_sourcepts[i][j+1][0];
					d3[1] = corner_sourcepts[i+1][j+1][1] - corner_sourcepts[i][j+1][1];
					d4[0] = corner_sourcepts[i+1][j][0] - corner_sourcepts[i+1][j+1][0];
					d4[1] = corner_sourcepts[i+1][j][1] - corner_sourcepts[i+1][j+1][1];
					// Now check whether the cell is twisted by mapping around a critical curve. We do this by extending opposite sides along lines
					// and finding where they intersect; if the intersection point lies within the cell, the pixel is twisted
					twist_status[i][j] = 0;
					double xa,ya,xb,yb,xc,yc,xd,yd,slope1,slope2;
					xa=corner_sourcepts[i][j][0];
					ya=corner_sourcepts[i][j][1];
					xb=corner_sourcepts[i][j+1][0];
					yb=corner_sourcepts[i][j+1][1];
					xc=corner_sourcepts[i+1][j+1][0];
					yc=corner_sourcepts[i+1][j+1][1];
					xd=corner_sourcepts[i+1][j][0];
					yd=corner_sourcepts[i+1][j][1];
					slope1 = (yb-ya)/(xb-xa);
					slope2 = (yc-yd)/(xc-xd);
					twist_pts[i][j][0] = (yd-ya+xa*slope1-xd*slope2)/(slope1-slope2);
					twist_pts[i][j][1] = (twist_pts[i][j][0]-xa)*slope1+ya;
					if ((test_if_between(twist_pts[i][j][0],xa,xb)) and (test_if_between(twist_pts[i][j][1],ya,yb)) and (test_if_between(twist_pts[i][j][0],xc,xd)) and (test_if_between(twist_pts[i][j][1],yc,yd))) {
						twist_status[i][j] = 1;
						d2[0] = twist_pts[i][j][0] - corner_sourcepts[i][j][0];
						d2[1] = twist_pts[i][j][1] - corner_sourcepts[i][j][1];
						d4[0] = twist_pts[i][j][0] - corner_sourcepts[i+1][j+1][0];
						d4[1] = twist_pts[i][j][1] - corner_sourcepts[i+1][j+1][1];
					} else {
						slope1 = (yd-ya)/(xd-xa);
						slope2 = (yc-yb)/(xc-xb);
						twist_pts[i][j][0] = (yb-ya+xa*slope1-xb*slope2)/(slope1-slope2);
						twist_pts[i][j][1] = (twist_pts[i][j][0]-xa)*slope1+ya;
						if ((test_if_between(twist_pts[i][j][0],xa,xd)) and (test_if_between(twist_pts[i][j][1],ya,yd)) and (test_if_between(twist_pts[i][j][0],xb,xc)) and (test_if_between(twist_pts[i][j][1],yb,yc))) {
							twist_status[i][j] = 2;
							d1[0] = corner_sourcepts[i][j][0] - twist_pts[i][j][0];
							d1[1] = corner_sourcepts[i][j][1] - twist_pts[i][j][1];
							d3[0] = corner_sourcepts[i+1][j+1][0] - twist_pts[i][j][0];
							d3[1] = corner_sourcepts[i+1][j+1][1] - twist_pts[i][j][1];
						}
					}

					source_plane_triangle1_area[i][j] = 0.5*abs(d1 ^ d2);
					source_plane_triangle2_area[i][j] = 0.5*abs(d3 ^ d4);
					//cout << source_plane_triangle1_area[i][j] << endl;
				}
			}
		//}
	}
#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for creating and ray-tracing image pixel grid: " << wtime << endl;
	}
#endif
	fit_to_data = NULL;
}

inline bool ImagePixelGrid::test_if_between(const double& p, const double& a, const double& b)
{
	if ((b>a) and (p>a) and (p<b)) return true;
	else if ((a>b) and (p>b) and (p<a)) return true;
	return false;
}

ImagePixelGrid::ImagePixelGrid(QLens* lens_in, SourceFitMode mode, RayTracingMethod method, double** sb_in, const int x_N_in, const int y_N_in, const int reduce_factor, double xmin_in, double xmax_in, double ymin_in, double ymax_in) : lens(lens_in), xmin(xmin_in), xmax(xmax_in), ymin(ymin_in), ymax(ymax_in)
{
	source_fit_mode = mode;
	ray_tracing_method = method;

	x_N = x_N_in / reduce_factor;
	y_N = y_N_in / reduce_factor;

	xy_N = x_N*y_N;
	if (source_fit_mode==Pixellated_Source) n_active_pixels = 0;
	else n_active_pixels = xy_N;

	corner_pts = new lensvector*[x_N+1];
	corner_sourcepts = new lensvector*[x_N+1];
	center_pts = new lensvector*[x_N];
	center_sourcepts = new lensvector*[x_N];
	maps_to_source_pixel = new bool*[x_N];
	pixel_index = new int*[x_N];
	mapped_source_pixels = new vector<SourcePixelGrid*>*[x_N];
	surface_brightness = new double*[x_N];
	source_plane_triangle1_area = new double*[x_N];
	source_plane_triangle2_area = new double*[x_N];
	max_nsplit = imax(24,lens_in->default_imgpixel_nsplit);
	nsplits = new int*[x_N];
	subpixel_maps_to_srcpixel = new bool**[x_N];
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
		pixel_index[i] = new int[y_N];
		surface_brightness[i] = new double[y_N];
		source_plane_triangle1_area[i] = new double[y_N];
		source_plane_triangle2_area[i] = new double[y_N];
		mapped_source_pixels[i] = new vector<SourcePixelGrid*>[y_N];
		subpixel_maps_to_srcpixel[i] = new bool*[y_N];
		nsplits[i] = new int[y_N];
		twist_pts[i] = new lensvector[y_N];
		twist_status[i] = new int[y_N];
		for (j=0; j < y_N; j++) {
			maps_to_source_pixel[i][j] = true;
			if (lens_in->split_imgpixels) nsplits[i][j] = lens_in->default_imgpixel_nsplit; // default
			else nsplits[i][j] = 1;
			subpixel_maps_to_srcpixel[i][j] = new bool[max_nsplit*max_nsplit];
			for (k=0; k < max_nsplit*max_nsplit; k++) subpixel_maps_to_srcpixel[i][j][k] = false;
		}
	}
	//if ((lens->active_image_pixel_i != NULL) and (lens->active_image_pixel_j != NULL)) {
		//for (i=0; i < x_N; i++) {
			//for (j=0; j < y_N; j++) {
				//maps_to_source_pixel[i][j] = false;
			//}
		//}
		//for (k=0; k < lens->image_npixels; k++) {
			//i = lens->active_image_pixel_i[k];
			//j = lens->active_image_pixel_j[k];
			//pixel_index[i][j] = k;
			//maps_to_source_pixel[i][j] = true;
		//}
	//}

	imggrid_zfactors = lens->reference_zfactors;
	imggrid_betafactors = lens->default_zsrc_beta_factors;
	pixel_xlength = (xmax-xmin)/x_N;
	pixel_ylength = (ymax-ymin)/y_N;
	pixel_area = pixel_xlength*pixel_ylength;
	triangle_area = 0.5*pixel_xlength*pixel_ylength;

	double x,y;
	int ii,jj;
	lensvector d1,d2,d3,d4;
	for (j=0; j <= y_N; j++) {
		y = ymin + j*pixel_ylength;
		for (i=0; i <= x_N; i++) {
			x = xmin + i*pixel_xlength;
			if ((i < x_N) and (j < y_N)) {
				center_pts[i][j][0] = x + 0.5*pixel_xlength;
				center_pts[i][j][1] = y + 0.5*pixel_ylength;
				lens->find_sourcept(center_pts[i][j],center_sourcepts[i][j],0,imggrid_zfactors,imggrid_betafactors);
			}
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
			lens->find_sourcept(corner_pts[i][j],corner_sourcepts[i][j],0,imggrid_zfactors,imggrid_betafactors);
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

	if (source_fit_mode==Pixellated_Source) {
		for (j=0; j < y_N; j++) {
			for (i=0; i < x_N; i++) {
				d1[0] = corner_sourcepts[i][j][0] - corner_sourcepts[i+1][j][0];
				d1[1] = corner_sourcepts[i][j][1] - corner_sourcepts[i+1][j][1];
				d2[0] = corner_sourcepts[i][j+1][0] - corner_sourcepts[i][j][0];
				d2[1] = corner_sourcepts[i][j+1][1] - corner_sourcepts[i][j][1];
				d3[0] = corner_sourcepts[i+1][j+1][0] - corner_sourcepts[i][j+1][0];
				d3[1] = corner_sourcepts[i+1][j+1][1] - corner_sourcepts[i][j+1][1];
				d4[0] = corner_sourcepts[i+1][j][0] - corner_sourcepts[i+1][j+1][0];
				d4[1] = corner_sourcepts[i+1][j][1] - corner_sourcepts[i+1][j+1][1];

				// Now check whether the cell is twisted by mapping around a critical curve. We do this by extending opposite sides along lines
				// and finding where they intersect; if the intersection point lies within the cell, the pixel is twisted
				twist_status[i][j] = 0;
				double xa,ya,xb,yb,xc,yc,xd,yd,slope1,slope2;
				xa=corner_sourcepts[i][j][0];
				ya=corner_sourcepts[i][j][1];
				xb=corner_sourcepts[i][j+1][0];
				yb=corner_sourcepts[i][j+1][1];
				xc=corner_sourcepts[i+1][j+1][0];
				yc=corner_sourcepts[i+1][j+1][1];
				xd=corner_sourcepts[i+1][j][0];
				yd=corner_sourcepts[i+1][j][1];
				slope1 = (yb-ya)/(xb-xa);
				slope2 = (yc-yd)/(xc-xd);
				twist_pts[i][j][0] = (yd-ya+xa*slope1-xd*slope2)/(slope1-slope2);
				twist_pts[i][j][1] = (twist_pts[i][j][0]-xa)*slope1+ya;
				if ((test_if_between(twist_pts[i][j][0],xa,xb)) and (test_if_between(twist_pts[i][j][1],ya,yb)) and (test_if_between(twist_pts[i][j][0],xc,xd)) and (test_if_between(twist_pts[i][j][1],yc,yd))) {
					twist_status[i][j] = 1;
					d2[0] = twist_pts[i][j][0] - corner_sourcepts[i][j][0];
					d2[1] = twist_pts[i][j][1] - corner_sourcepts[i][j][1];
					d4[0] = twist_pts[i][j][0] - corner_sourcepts[i+1][j+1][0];
					d4[1] = twist_pts[i][j][1] - corner_sourcepts[i+1][j+1][1];
				} else {
					slope1 = (yd-ya)/(xd-xa);
					slope2 = (yc-yb)/(xc-xb);
					twist_pts[i][j][0] = (yb-ya+xa*slope1-xb*slope2)/(slope1-slope2);
					twist_pts[i][j][1] = (twist_pts[i][j][0]-xa)*slope1+ya;
					if ((test_if_between(twist_pts[i][j][0],xa,xd)) and (test_if_between(twist_pts[i][j][1],ya,yd)) and (test_if_between(twist_pts[i][j][0],xb,xc)) and (test_if_between(twist_pts[i][j][1],yb,yc))) {
						twist_status[i][j] = 2;
						d1[0] = corner_sourcepts[i][j][0] - twist_pts[i][j][0];
						d1[1] = corner_sourcepts[i][j][1] - twist_pts[i][j][1];
						d3[0] = corner_sourcepts[i+1][j+1][0] - twist_pts[i][j][0];
						d3[1] = corner_sourcepts[i+1][j+1][1] - twist_pts[i][j][1];
					}
				}

				source_plane_triangle1_area[i][j] = 0.5*abs(d1 ^ d2);
				source_plane_triangle2_area[i][j] = 0.5*abs(d3 ^ d4);
			}
		}
	}

	fit_to_data = NULL;
}

ImagePixelGrid::ImagePixelGrid(QLens* lens_in, double* zfactor_in, double** betafactor_in, SourceFitMode mode, RayTracingMethod method, ImagePixelData& pixel_data)
{
	lens = lens_in;
	source_fit_mode = mode;
	ray_tracing_method = method;
	pixel_data.get_grid_params(xmin,xmax,ymin,ymax,x_N,y_N);

	xy_N = x_N*y_N;
	if (source_fit_mode==Pixellated_Source) n_active_pixels = 0;
	else n_active_pixels = xy_N;

	corner_pts = new lensvector*[x_N+1];
	corner_sourcepts = new lensvector*[x_N+1];
	center_pts = new lensvector*[x_N];
	center_sourcepts = new lensvector*[x_N];
	fit_to_data = new bool*[x_N];
	maps_to_source_pixel = new bool*[x_N];
	pixel_index = new int*[x_N];
	mapped_source_pixels = new vector<SourcePixelGrid*>*[x_N];
	surface_brightness = new double*[x_N];
	source_plane_triangle1_area = new double*[x_N];
	source_plane_triangle2_area = new double*[x_N];
	max_nsplit = imax(24,lens_in->default_imgpixel_nsplit);
	nsplits = new int*[x_N];
	subpixel_maps_to_srcpixel = new bool**[x_N];
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
		fit_to_data[i] = new bool[y_N];
		pixel_index[i] = new int[y_N];
		surface_brightness[i] = new double[y_N];
		source_plane_triangle1_area[i] = new double[y_N];
		source_plane_triangle2_area[i] = new double[y_N];
		mapped_source_pixels[i] = new vector<SourcePixelGrid*>[y_N];
		subpixel_maps_to_srcpixel[i] = new bool*[y_N];
		nsplits[i] = new int[y_N];
		twist_pts[i] = new lensvector[y_N];
		twist_status[i] = new int[y_N];
		for (j=0; j < y_N; j++) {
			surface_brightness[i][j] = 0;
			if (source_fit_mode==Parameterized_Source) maps_to_source_pixel[i][j] = true; // in this mode you can always get a surface brightness for any image pixel
			if (lens_in->split_imgpixels) nsplits[i][j] = lens_in->default_imgpixel_nsplit; // default
			else nsplits[i][j] = 1;
			subpixel_maps_to_srcpixel[i][j] = new bool[max_nsplit*max_nsplit];
			for (k=0; k < max_nsplit*max_nsplit; k++) subpixel_maps_to_srcpixel[i][j][k] = false;
		}
	}
	imggrid_zfactors = zfactor_in;
	imggrid_betafactors = betafactor_in;

	pixel_xlength = (xmax-xmin)/x_N;
	pixel_ylength = (ymax-ymin)/y_N;
	max_sb = -1e30;
	pixel_area = pixel_xlength*pixel_ylength;
	triangle_area = 0.5*pixel_xlength*pixel_ylength;

	double x,y;
	for (j=0; j <= y_N; j++) {
		y = pixel_data.yvals[j];
		for (i=0; i <= x_N; i++) {
			x = pixel_data.xvals[i];
			if ((i < x_N) and (j < y_N)) {
				center_pts[i][j][0] = x + 0.5*pixel_xlength;
				center_pts[i][j][1] = y + 0.5*pixel_ylength;
				surface_brightness[i][j] = pixel_data.surface_brightness[i][j];
				fit_to_data[i][j] = pixel_data.require_fit[i][j];
				if (surface_brightness[i][j] > max_sb) max_sb=surface_brightness[i][j];
			}
			corner_pts[i][j][0] = x;
			corner_pts[i][j][1] = y;
		}
	}
}

void ImagePixelGrid::plot_surface_brightness(string outfile_root, bool plot_residual, bool show_only_mask)
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
	double residual;

	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((!show_only_mask) or (fit_to_data==NULL) or (fit_to_data[i][j])) {
				if (!plot_residual) pixel_image_file << surface_brightness[i][j];
				else {
					residual = lens->image_pixel_data->surface_brightness[i][j] - surface_brightness[i][j];
					pixel_image_file << residual;
					lens->find_sourcept(center_pts[i][j],center_sourcepts[i][j],0,imggrid_zfactors,imggrid_betafactors);
					if (abs(residual) > 0.02) pixel_src_file << center_sourcepts[i][j][0] << " " << center_sourcepts[i][j][1] << " " << residual << endl;
				}
			} else {
				if (plot_residual) {
					if (show_only_mask) pixel_image_file << "NaN";
					else pixel_image_file << lens->image_pixel_data->surface_brightness[i][j];
				} else {
					pixel_image_file << "NaN";
				}
			}
			if (i < x_N-1) pixel_image_file << " ";
		}
		pixel_image_file << endl;
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
	double x, y, xstep, ystep;
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
				pixels = new double[x_N];

				for (fpixel[1]=1, j=0; fpixel[1] <= naxes[1]; fpixel[1]++, j++)
				{
					for (i=0; i < x_N; i++) {
						if (!plot_residual) pixels[i] = surface_brightness[i][j];
						else pixels[i] = lens->image_pixel_data->surface_brightness[i][j] - surface_brightness[i][j];
					}
					fits_write_pix(outfptr, TDOUBLE, fpixel, naxes[0], pixels, &status);
				}
				delete[] pixels;
			}
			if (pixel_xlength==pixel_ylength)
				fits_write_key(outfptr, TDOUBLE, "PXSIZE", &pixel_xlength, "length of square pixels (in arcsec)", &status);
			if (lens->sim_pixel_noise != 0)
				fits_write_key(outfptr, TDOUBLE, "PXNOISE", &lens->sim_pixel_noise, "pixel surface brightness noise", &status);
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

void ImagePixelGrid::set_fit_window(ImagePixelData& pixel_data)
{
	if ((x_N != pixel_data.npixels_x) or (y_N != pixel_data.npixels_y)) {
		warn("Number of data pixels does not match specified number of image pixels; cannot activate fit window");
		return;
	}
	int i,j;
	if (fit_to_data==NULL) {
		fit_to_data = new bool*[x_N];
		for (i=0; i < x_N; i++) fit_to_data[i] = new bool[y_N];
	}
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			fit_to_data[i][j] = pixel_data.require_fit[i][j];
		}
	}
}

void ImagePixelGrid::include_all_pixels()
{
	int i,j;
	if (fit_to_data==NULL) {
		fit_to_data = new bool*[x_N];
		for (i=0; i < x_N; i++) fit_to_data[i] = new bool[y_N];
	}
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			fit_to_data[i][j] = true;
		}
	}
}

void ImagePixelGrid::activate_extended_mask()
{
	int i,j;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			fit_to_data[i][j] = lens->image_pixel_data->extended_mask[i][j];
		}
	}
}


void ImagePixelGrid::reset_nsplit()
{
	int i,j;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if (lens->split_imgpixels) nsplits[i][j] = lens->default_imgpixel_nsplit; // default
			else nsplits[i][j] = 1;
		}
	}
}

void ImagePixelGrid::redo_lensing_calculations()
{
	imggrid_zfactors = lens->reference_zfactors;
	imggrid_betafactors = lens->default_zsrc_beta_factors;
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
	if (source_fit_mode==Pixellated_Source) n_active_pixels = 0;

	int i,j,n,n_cell,n_corner,n_yp;

	long int ntot_cells = 0;
	long int ntot_corners = 0;
	for (i=0; i < x_N+1; i++) {
		for (j=0; j < y_N+1; j++) {
			if ((i < x_N) and (j < y_N) and (lens->image_pixel_data->extended_mask[i][j])) ntot_cells++;
			if (((i < x_N) and (j < y_N) and (lens->image_pixel_data->extended_mask[i][j])) or ((j < y_N) and (i > 0) and (lens->image_pixel_data->extended_mask[i-1][j])) or ((i < x_N) and (j > 0) and (lens->image_pixel_data->extended_mask[i][j-1])) or ((i > 0) and (j > 0) and (lens->image_pixel_data->extended_mask[i-1][j-1]))) {
				ntot_corners++;
			}
		}
	}

	//cout << ntot_corners << " " << ntot_cells << endl;
	int *extended_mask_i = new int[ntot_cells];
	int *extended_mask_j = new int[ntot_cells];
	int *extended_mask_corner_i = new int[ntot_corners];
	int *extended_mask_corner_j = new int[ntot_corners];
	int *extended_mask_corner = new int[ntot_cells];
	int *extended_mask_corner_up = new int[ntot_cells];
	int **nvals = new int*[x_N];
	for (i=0; i < x_N; i++) nvals[i] = new int[y_N];
	int **ncvals = new int*[x_N+1];
	for (i=0; i < x_N+1; i++) ncvals[i] = new int[y_N+1];
	
	n_cell=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (lens->image_pixel_data->extended_mask[i][j]) {
				extended_mask_i[n_cell] = i;
				extended_mask_j[n_cell] = j;
				nvals[i][j] = n_cell;
				n_cell++;
			} else {
				nvals[i][j] = -1;
			}
		}
	}

	//cout << "HI1" << endl;
	n_corner=0;
	for (j=0; j < y_N+1; j++) {
		for (i=0; i < x_N+1; i++) {
			ncvals[i][j] = -1;
			if (((i < x_N) and (j < y_N) and (lens->image_pixel_data->extended_mask[i][j])) or ((j < y_N) and (i > 0) and (lens->image_pixel_data->extended_mask[i-1][j])) or ((i < x_N) and (j > 0) and (lens->image_pixel_data->extended_mask[i][j-1])) or ((i > 0) and (j > 0) and (lens->image_pixel_data->extended_mask[i-1][j-1]))) {
			//if (((i < x_N) and (j < y_N) and (lens->image_pixel_data->extended_mask[i][j])) or ((j < y_N) and (lens->image_pixel_data->extended_mask[i-1][j])) or ((i < x_N) and (lens->image_pixel_data->extended_mask[i][j-1])) or (lens->image_pixel_data->extended_mask[i-1][j-1])) {
				extended_mask_corner_i[n_corner] = i;
				extended_mask_corner_j[n_corner] = j;
				if (i > (x_N+1)) die("FUCK! corner i is huge from the get-go");
				if (j > (y_N+1)) die("FUCK! corner j is huge from the get-go");
				//if ((i < x_N) and (j < y_N)) extended_mask_corner[nvals[i][j]] = n_corner;
				ncvals[i][j] = n_corner;
				n_corner++;
			}
		}
	}
	//cout << "corner count: " << n_corner << " " << ntot_corners << endl;
	//cout << "HI2" << endl;
	for (int n=0; n < ntot_cells; n++) {
		i = extended_mask_i[n];
		j = extended_mask_j[n];
		extended_mask_corner[n] = ncvals[i][j];
		extended_mask_corner_up[n] = ncvals[i][j+1];
	}
	for (int n=0; n < ntot_corners; n++) {
		i = extended_mask_corner_i[n];
		j = extended_mask_corner_j[n];
		if (i > (x_N+1)) die("FUCK! corner i is huge");
		if (j > (y_N+1)) die("FUCK! corner j is huge");
	}

	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			mapped_source_pixels[i][j].clear();
			if (lens->split_imgpixels) nsplits[i][j] = lens->default_imgpixel_nsplit; // default
			else nsplits[i][j] = 1;
		}
	}
	//cout << "cells: " << ntot_cells << " tot: " << (x_N*y_N) << endl;
	//cout << "corners: " << ntot_corners << " tot: " << ((x_N+1)*(y_N+1)) << endl;
	//ntot_corners = (x_N+1)*(y_N+1);
	//ntot_cells = x_N*y_N;
	double *defx_corners, *defy_corners, *defx_centers, *defy_centers, *area_tri1, *area_tri2;
	double *twistx, *twisty;
	int *twiststat;
	defx_corners = new double[ntot_corners];
	defy_corners = new double[ntot_corners];
	defx_centers = new double[ntot_cells];
	defy_centers = new double[ntot_cells];
	area_tri1 = new double[ntot_cells];
	area_tri2 = new double[ntot_cells];
	twistx = new double[ntot_cells];
	twisty = new double[ntot_cells];
	twiststat = new int[ntot_cells];

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

	//cout << "HI3" << endl;
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
			j = extended_mask_corner_j[n];
			i = extended_mask_corner_i[n];
			//cout << i << " " << j << " " << n << " " << ntot_corners << " " << mpi_end << endl;
			if (i > x_N+1) die("FUCK! i is huge");
			if (j > y_N+1) die("FUCK! j is huge");
			//lens->find_sourcept(corner_pts[i][j],defx_corners[n],defy_corners[n],thread,imggrid_zfactors,imggrid_betafactors);
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
		int nn, nn_yp;
	//cout << "HI4" << endl;
		#pragma omp for private(n_cell,i,j,n,n_yp) schedule(dynamic)
		for (n_cell=mpi_start2; n_cell < mpi_end2; n_cell++) {
			//j = n_cell / x_N;
			//i = n_cell % x_N;
			j = extended_mask_j[n_cell];
			i = extended_mask_i[n_cell];
			//cout << "TEST: " << i << " " << j << " " << ii << "  " << jj << endl;
			lens->find_sourcept(center_pts[i][j],defx_centers[n_cell],defy_centers[n_cell],thread,imggrid_zfactors,imggrid_betafactors);

			//n = j*(x_N+1)+i;
			//n_yp = (j+1)*(x_N+1)+i;
			n = extended_mask_corner[n_cell];
			n_yp = extended_mask_corner_up[n_cell];
			//cout << "TEST2: " << n << " " << n_yp << " " << nn << " " << nn_yp << endl;
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
	}
	//cout << "HI5" << endl;
#ifdef USE_MPI
	int id, chunk, start;
	for (id=0; id < lens->group_np; id++) {
		chunk = ntot_cells / lens->group_np;
		start = id*chunk;
		if (id == lens->group_np-1) chunk += (ntot_cells % lens->group_np); // assign the remainder elements to the last mpi process
		MPI_Bcast(defx_centers+start,chunk,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(defy_centers+start,chunk,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(area_tri1+start,chunk,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(area_tri2+start,chunk,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(twistx+start,chunk,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(twisty+start,chunk,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(twiststat+start,chunk,MPI_INT,id,sub_comm);
	}
	MPI_Comm_free(&sub_comm);
#endif
	for (n=0; n < ntot_corners; n++) {
		//j = n / (x_N+1);
		//i = n % (x_N+1);
		j = extended_mask_corner_j[n];
		i = extended_mask_corner_i[n];
		corner_sourcepts[i][j][0] = defx_corners[n];
		corner_sourcepts[i][j][1] = defy_corners[n];
		if ((i < x_N) and (j < y_N)) {
			//n_cell = j*x_N+i;
			n_cell = nvals[i][j];
			//cout << "BLA: " << n_cell << " " << n_cell2 << endl;
			source_plane_triangle1_area[i][j] = area_tri1[n_cell];
			source_plane_triangle2_area[i][j] = area_tri2[n_cell];
			center_sourcepts[i][j][0] = defx_centers[n_cell];
			center_sourcepts[i][j][1] = defy_centers[n_cell];
			twist_pts[i][j][0] = twistx[n_cell];
			twist_pts[i][j][1] = twisty[n_cell];
			twist_status[i][j] = twiststat[n_cell];
		}
	}

	//cout << "HI6" << endl;
#ifdef USE_OPENMP
	if (lens->show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (lens->mpi_id==0) cout << "Wall time for ray-tracing image pixel grid: " << wtime << endl;
	}
#endif
	delete[] defx_corners;
	delete[] defy_corners;
	delete[] defx_centers;
	delete[] defy_centers;
	delete[] area_tri1;
	delete[] area_tri2;
	delete[] twistx;
	delete[] twisty;
	delete[] twiststat;

	delete[] extended_mask_i;
	delete[] extended_mask_j;
	delete[] extended_mask_corner_i;
	delete[] extended_mask_corner_j;
	delete[] extended_mask_corner;
	delete[] extended_mask_corner_up;
	for (i=0; i < x_N; i++) delete[] nvals[i];
	delete[] nvals;
	for (i=0; i < x_N+1; i++) delete[] ncvals[i];
	delete[] ncvals;
	
}

void ImagePixelGrid::redo_lensing_calculations_corners()
{
	imggrid_zfactors = lens->reference_zfactors;
	imggrid_betafactors = lens->default_zsrc_beta_factors;
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
			else nsplits[i][j] = 1;
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

bool ImagePixelData::test_if_in_fit_region(const double& x, const double& y)
{
	// it would be faster to just use division to figure out which pixel it's in, but this is good enough
	int i,j;
	for (j=0; j <= npixels_y; j++) {
		if ((yvals[j] <= y) and (yvals[j+1] > y)) {
			for (i=0; i <= npixels_x; i++) {
				if ((xvals[i] <= x) and (xvals[i+1] > x)) {
					if (require_fit[i][j] == true) return true;
					else break;
				}
			}
		}
	}
	return false;
}

double ImagePixelGrid::calculate_signal_to_noise(const double& pixel_noise_sig, double& total_signal)
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
	double signal_mean=0;
	int npixels=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (surface_brightness[i][j] > signal_threshold_frac*sbmax) {
				signal_mean += surface_brightness[i][j];
				npixels++;
			}
		}
	}
	total_signal = signal_mean * pixel_xlength * pixel_ylength;
	if (npixels > 0) signal_mean /= npixels;
	return signal_mean / pixel_noise_sig;
}

void ImagePixelGrid::add_pixel_noise(const double& pixel_noise_sig)
{
	if (surface_brightness == NULL) die("surface brightness pixel map has not been loaded");
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			surface_brightness[i][j] += pixel_noise_sig*lens->NormalDeviate();
		}
	}
	pixel_noise = pixel_noise_sig;
}

void ImagePixelGrid::find_optimal_sourcegrid(double& sourcegrid_xmin, double& sourcegrid_xmax, double& sourcegrid_ymin, double& sourcegrid_ymax, const double &sourcegrid_limit_xmin, const double &sourcegrid_limit_xmax, const double &sourcegrid_limit_ymin, const double& sourcegrid_limit_ymax)
{
	if (surface_brightness == NULL) die("surface brightness pixel map has not been loaded");
	bool use_noise_threshold = true;
	if (lens->noise_threshold <= 0) use_noise_threshold = false;
	double threshold = lens->noise_threshold*pixel_noise;
	int i,j;
	sourcegrid_xmin=1e30;
	sourcegrid_xmax=-1e30;
	sourcegrid_ymin=1e30;
	sourcegrid_ymax=-1e30;
	int ii,jj,il,ih,jl,jh,nn;
	double sbavg;
	static const int window_size_for_sbavg = 0;
	bool resize_grid;
	for (i=0; i < x_N; i++) {
		for (j=0; j < y_N; j++) {
			if (fit_to_data[i][j]) {
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
							sbavg += surface_brightness[ii][jj];
							nn++;
						}
					}
					sbavg /= nn;
					if (sbavg <= threshold) resize_grid = false;
				}
				if (resize_grid) {
					if (center_sourcepts[i][j][0] < sourcegrid_xmin) {
						if (center_sourcepts[i][j][0] > sourcegrid_limit_xmin) sourcegrid_xmin = center_sourcepts[i][j][0];
						else if (sourcegrid_xmin > sourcegrid_limit_xmin) sourcegrid_xmin = sourcegrid_limit_xmin;
					}
					if (center_sourcepts[i][j][0] > sourcegrid_xmax) {
						if (center_sourcepts[i][j][0] < sourcegrid_limit_xmax) sourcegrid_xmax = center_sourcepts[i][j][0];
						else if (sourcegrid_xmax < sourcegrid_limit_xmax) sourcegrid_xmax = sourcegrid_limit_xmax;
					}
					if (center_sourcepts[i][j][1] < sourcegrid_ymin) {
						if (center_sourcepts[i][j][1] > sourcegrid_limit_ymin) sourcegrid_ymin = center_sourcepts[i][j][1];
						else if (sourcegrid_ymin > sourcegrid_limit_ymin) sourcegrid_ymin = sourcegrid_limit_ymin;
					}
					if (center_sourcepts[i][j][1] > sourcegrid_ymax) {
						if (center_sourcepts[i][j][1] < sourcegrid_limit_ymax) sourcegrid_ymax = center_sourcepts[i][j][1];
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

void ImagePixelGrid::fill_surface_brightness_vector()
{
	int column_j = 0;
	int i,j;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if (maps_to_source_pixel[i][j]) {
				lens->image_surface_brightness[column_j++] = surface_brightness[i][j];
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
			if ((!fit_to_data) or (fit_to_data[i][j])) {
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
			if ((fit_to_data==NULL) or (fit_to_data[i][j])) {
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
			if ((fit_to_data==NULL) or (fit_to_data[i][j])) {
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
void ImagePixelGrid::assign_required_data_pixels(double srcgrid_xmin, double srcgrid_xmax, double srcgrid_ymin, double srcgrid_ymax, int& count, ImagePixelData *data_in)
{
	int i,j;
	count=0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			if ((center_sourcepts[i][j][0] > srcgrid_xmin) and (center_sourcepts[i][j][0] < srcgrid_xmax) and (center_sourcepts[i][j][1] > srcgrid_ymin) and (center_sourcepts[i][j][1] < srcgrid_ymax)) {
				data_in->require_fit[i][j] = true;
				count++;
			}
			else {
				data_in->require_fit[i][j] = false;
			}
		}
	}
}
*/

int ImagePixelGrid::count_nonzero_source_pixel_mappings()
{
	int tot=0;
	int i,j,img_index;
	for (img_index=0; img_index < lens->image_npixels; img_index++) {
		i = lens->active_image_pixel_i[img_index];
		j = lens->active_image_pixel_j[img_index];
		tot += mapped_source_pixels[i][j].size();
	}
	return tot;
}

void ImagePixelGrid::assign_image_mapping_flags()
{
	int i,j;
	n_active_pixels = 0;
	n_high_sn_pixels = 0;
	for (j=0; j < y_N; j++) {
		for (i=0; i < x_N; i++) {
			mapped_source_pixels[i][j].clear();
			maps_to_source_pixel[i][j] = false;
		}
	}
	if (ray_tracing_method == Area_Overlap)
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
					if ((fit_to_data == NULL) or (fit_to_data[i][j])) {
						corners[0] = &corner_sourcepts[i][j];
						corners[1] = &corner_sourcepts[i][j+1];
						corners[2] = &corner_sourcepts[i+1][j];
						corners[3] = &corner_sourcepts[i+1][j+1];
						if (source_pixel_grid->assign_source_mapping_flags_overlap(corners,&twist_pts[i][j],twist_status[i][j],mapped_source_pixels[i][j],thread)==true) {
							maps_to_source_pixel[i][j] = true;
							#pragma omp atomic
							n_active_pixels++;
							if ((fit_to_data != NULL) and (fit_to_data[i][j]) and (lens->image_pixel_data->high_sn_pixel[i][j])) n_high_sn_pixels++;
						} else
							maps_to_source_pixel[i][j] = false;
					}
				}
			}
		}
	}
	else if (ray_tracing_method == Interpolate)
	{
		#pragma omp parallel
		{
			int thread;
#ifdef USE_OPENMP
			thread = omp_get_thread_num();
#else
			thread = 0;
#endif
			int ii,jj,kk,nsplit;
			double u0, w0;
			#pragma omp for private(i,j,ii,jj,kk,nsplit,u0,w0) schedule(dynamic)
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					if ((fit_to_data == NULL) or (fit_to_data[i][j])) {
						bool maps_to_something = false;
						kk=0;
						nsplit = nsplits[i][j];
						for (ii=0; ii < nsplit; ii++) {
							for (jj=0; jj < nsplit; jj++) {
								// The image pixel subgridding may result in multiple Lmatrix entries that map to the same source pixel, but these
								// will be consolidated when the PSF convolution is carried out
								subpixel_maps_to_srcpixel[i][j][kk] = false;
								lensvector center1, center1_srcpt;
								u0 = ((double) (1+2*ii))/(2*nsplit);
								w0 = ((double) (1+2*jj))/(2*nsplit);
								center1[0] = u0*corner_pts[i][j][0] + (1-u0)*corner_pts[i+1][j][0];
								center1[1] = w0*corner_pts[i][j][1] + (1-w0)*corner_pts[i][j+1][1];
								lens->find_sourcept(center1,center1_srcpt,thread,imggrid_zfactors,imggrid_betafactors);
								if (source_pixel_grid->assign_source_mapping_flags_interpolate(center1_srcpt,mapped_source_pixels[i][j],thread,i,j)==true) {
									maps_to_something = true;
									subpixel_maps_to_srcpixel[i][j][kk] = true;
								}
								kk++;
							}
						}

						//if (source_pixel_grid->assign_source_mapping_flags_interpolate(center_sourcepts[i][j],mapped_source_pixels[i][j],thread,i,j)==true) {
						if (maps_to_something==true) {
							maps_to_source_pixel[i][j] = true;
							#pragma omp atomic
							n_active_pixels++;
							if ((fit_to_data != NULL) and (fit_to_data[i][j]) and (lens->image_pixel_data->high_sn_pixel[i][j])) n_high_sn_pixels++;
						} else maps_to_source_pixel[i][j] = false;
					}
				}
			}
		}
	}
}

void ImagePixelGrid::find_surface_brightness(bool plot_foreground_only)
{
	//ofstream blergh("blergh.dat");
	imggrid_zfactors = lens->reference_zfactors;
	imggrid_betafactors = lens->default_zsrc_beta_factors;
#ifdef USE_OPENMP
	double wtime0, wtime;
	if (lens->show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	if (source_fit_mode == Pixellated_Source) {
		bool at_least_one_foreground_src = false;
		for (int k=0; k < lens->n_sb; k++) {
			if (!lens->sb_list[k]->is_lensed) {
				at_least_one_foreground_src = true;
				break;
			}
		}

		if (ray_tracing_method == Area_Overlap) {
			lensvector **corners = new lensvector*[4];
			int i,j;
			for (j=0; j < y_N; j++) {
				for (i=0; i < x_N; i++) {
					surface_brightness[i][j] = 0;
					corners[0] = &corner_sourcepts[i][j];
					corners[1] = &corner_sourcepts[i][j+1];
					corners[2] = &corner_sourcepts[i+1][j];
					corners[3] = &corner_sourcepts[i+1][j+1];
					if (!plot_foreground_only) surface_brightness[i][j] = source_pixel_grid->find_lensed_surface_brightness_overlap(corners,&twist_pts[i][j],twist_status[i][j],0);
					if (at_least_one_foreground_src) {
						for (int k=0; k < lens->n_sb; k++) {
							if (!lens->sb_list[k]->is_lensed) {
								if (!lens->sb_list[k]->zoom_subgridding) surface_brightness[i][j] += lens->sb_list[k]->surface_brightness(center_pts[i][j][0],center_pts[i][j][1]);
								else surface_brightness[i][j] += lens->sb_list[k]->surface_brightness_zoom(center_pts[i][j],corner_pts[i][j],corner_pts[i+1][j],corner_pts[i][j+1],corner_pts[i+1][j+1]);
							}
						}
					}
				}
			}
			delete[] corners;
		}
		else if (ray_tracing_method == Interpolate) {
			int i,j;
			if (lens->split_imgpixels) {
				#pragma omp parallel
				{
					int thread;
#ifdef USE_OPENMP
					thread = omp_get_thread_num();
#else
					thread = 0;
#endif
					int ii,jj,nsplit;
					double u0, w0, sb;
					lensvector corner1, corner2, corner3, corner4;
					double subpixel_xlength, subpixel_ylength;
					subpixel_xlength = pixel_xlength/nsplit;
					subpixel_ylength = pixel_ylength/nsplit;
					#pragma omp for private(i,j,ii,jj,nsplit,u0,w0,sb) schedule(dynamic)
					for (j=0; j < y_N; j++) {
						for (i=0; i < x_N; i++) {
							surface_brightness[i][j] = 0;
							if ((fit_to_data == NULL) or (fit_to_data[i][j])) {
								sb=0;
								nsplit = nsplits[i][j];
								for (ii=0; ii < nsplit; ii++) {
									for (jj=0; jj < nsplit; jj++) {
										lensvector center_pt, center_srcpt;
										u0 = ((double) (1+2*ii))/(2*nsplit);
										w0 = ((double) (1+2*jj))/(2*nsplit);
										center_pt[0] = u0*corner_pts[i][j][0] + (1-u0)*corner_pts[i+1][j][0];
										center_pt[1] = w0*corner_pts[i][j][1] + (1-w0)*corner_pts[i][j+1][1];
										if (!plot_foreground_only) {
											lens->find_sourcept(center_pt,center_srcpt,thread,imggrid_zfactors,imggrid_betafactors);
											sb += source_pixel_grid->find_lensed_surface_brightness_interpolate(center_srcpt,thread);
										}
										if (at_least_one_foreground_src) {
											for (int k=0; k < lens->n_sb; k++) {
												if (!lens->sb_list[k]->is_lensed) {
													if (!lens->sb_list[k]->zoom_subgridding) sb += lens->sb_list[k]->surface_brightness(center_pt[0],center_pt[1]);
													else {
														corner1[0] = center_pt[0] - subpixel_xlength/2;
														corner1[1] = center_pt[1] - subpixel_ylength/2;
														corner2[0] = center_pt[0] + subpixel_xlength/2;
														corner2[1] = center_pt[1] - subpixel_ylength/2;
														corner3[0] = center_pt[0] - subpixel_xlength/2;
														corner3[1] = center_pt[1] + subpixel_ylength/2;
														corner4[0] = center_pt[0] + subpixel_xlength/2;
														corner4[1] = center_pt[1] + subpixel_ylength/2;
														sb += lens->sb_list[k]->surface_brightness_zoom(center_pt,corner1,corner2,corner3,corner4);
													}
												}
											}
										}

									}
								}
								surface_brightness[i][j] = sb / (nsplit*nsplit);
							}
						}
					}
				}
			} else {
				int i,j;
				for (j=0; j < y_N; j++) {
					for (i=0; i < x_N; i++) {
						surface_brightness[i][j] = 0;
						if ((fit_to_data == NULL) or (fit_to_data[i][j])) {
							if (!plot_foreground_only) surface_brightness[i][j] = source_pixel_grid->find_lensed_surface_brightness_interpolate(center_sourcepts[i][j],0);
							if (at_least_one_foreground_src) {
								for (int k=0; k < lens->n_sb; k++) {
									if (!lens->sb_list[k]->is_lensed) {
										if (!lens->sb_list[k]->zoom_subgridding) surface_brightness[i][j] += lens->sb_list[k]->surface_brightness(center_pts[i][j][0],center_pts[i][j][1]);
										else surface_brightness[i][j] += lens->sb_list[k]->surface_brightness_zoom(center_pts[i][j],corner_pts[i][j],corner_pts[i+1][j],corner_pts[i][j+1],corner_pts[i+1][j+1]);
									}
								}
							}
						}

					}

				}
			}
		}
	}
	else
	{
		bool at_least_one_lensed_src = false;
		bool at_least_one_zoom_lensed_src = false;
		for (int k=0; k < lens->n_sb; k++) {
			if (lens->sb_list[k]->is_lensed) {
				at_least_one_lensed_src = true;
				if (lens->sb_list[k]->zoom_subgridding) at_least_one_zoom_lensed_src = true;
			}
		}
		int i,j;
		if (lens->split_imgpixels) {
			#pragma omp parallel
			{
				int thread;
#ifdef USE_OPENMP
				thread = omp_get_thread_num();
#else
				thread = 0;
#endif
				int ii,jj,nsplit;
				double u0, w0, sb;
				double U0, W0, U1, W1;
				lensvector center_pt, center_srcpt;
				lensvector corner1, corner2, corner3, corner4;
				lensvector corner1_src, corner2_src, corner3_src, corner4_src;
				double subpixel_xlength, subpixel_ylength;
				#pragma omp for private(i,j,ii,jj,nsplit,u0,w0,U0,W0,U1,W1,sb,subpixel_xlength,subpixel_ylength,center_pt,center_srcpt,corner1,corner2,corner3,corner4,corner1_src,corner2_src,corner3_src,corner4_src) schedule(dynamic)
				for (j=0; j < y_N; j++) {
					for (i=0; i < x_N; i++) {
						surface_brightness[i][j] = 0;
						if ((fit_to_data == NULL) or (fit_to_data[i][j])) {
							sb=0;
							nsplit = nsplits[i][j];
							subpixel_xlength = pixel_xlength/nsplit;
							subpixel_ylength = pixel_ylength/nsplit;
							for (ii=0; ii < nsplit; ii++) {
								u0 = ((double) (1+2*ii))/(2*nsplit);
								center_pt[0] = u0*corner_pts[i][j][0] + (1-u0)*corner_pts[i+1][j][0];
								for (jj=0; jj < nsplit; jj++) {
									w0 = ((double) (1+2*jj))/(2*nsplit);
									center_pt[1] = w0*corner_pts[i][j][1] + (1-w0)*corner_pts[i][j+1][1];
									if (at_least_one_zoom_lensed_src) {
										U0 = ((double) (2*ii))/(2*nsplit);
										W0 = ((double) (2*jj))/(2*nsplit);
										U1 = ((double) (2+2*ii))/(2*nsplit);
										W1 = ((double) (2+2*jj))/(2*nsplit);

										corner1_src[0] = (U1*corner_sourcepts[i][j][0] + (1-U1)*corner_sourcepts[i+1][j][0])*W1 + (U1*corner_sourcepts[i][j+1][0] + (1-U1)*corner_sourcepts[i+1][j+1][0])*(1-W1);
										corner1_src[1] = (U1*corner_sourcepts[i][j][1] + (1-U1)*corner_sourcepts[i+1][j][1])*W1 + (U1*corner_sourcepts[i][j+1][1] + (1-U1)*corner_sourcepts[i+1][j+1][1])*(1-W1);
										corner2_src[0] = (U0*corner_sourcepts[i][j][0] + (1-U0)*corner_sourcepts[i+1][j][0])*W1 + (U0*corner_sourcepts[i][j+1][0] + (1-U0)*corner_sourcepts[i+1][j+1][0])*(1-W1);
										corner2_src[1] = (U0*corner_sourcepts[i][j][1] + (1-U0)*corner_sourcepts[i+1][j][1])*W1 + (U0*corner_sourcepts[i][j+1][1] + (1-U0)*corner_sourcepts[i+1][j+1][1])*(1-W1);
										corner3_src[0] = (U1*corner_sourcepts[i][j][0] + (1-U1)*corner_sourcepts[i+1][j][0])*W0 + (U1*corner_sourcepts[i][j+1][0] + (1-U1)*corner_sourcepts[i+1][j+1][0])*(1-W0);
										corner3_src[1] = (U1*corner_sourcepts[i][j][1] + (1-U1)*corner_sourcepts[i+1][j][1])*W0 + (U1*corner_sourcepts[i][j+1][1] + (1-U1)*corner_sourcepts[i+1][j+1][1])*(1-W0);
										corner4_src[0] = (U0*corner_sourcepts[i][j][0] + (1-U0)*corner_sourcepts[i+1][j][0])*W0 + (U0*corner_sourcepts[i][j+1][0] + (1-U0)*corner_sourcepts[i+1][j+1][0])*(1-W0);
										corner4_src[1] = (U0*corner_sourcepts[i][j][1] + (1-U0)*corner_sourcepts[i+1][j][1])*W0 + (U0*corner_sourcepts[i][j+1][1] + (1-U0)*corner_sourcepts[i+1][j+1][1])*(1-W0);

									}
									if ((!plot_foreground_only) and (at_least_one_lensed_src)) {
										lens->find_sourcept(center_pt,center_srcpt,thread,imggrid_zfactors,imggrid_betafactors);
									}
									for (int k=0; k < lens->n_sb; k++) {
										if ((lens->sb_list[k]->is_lensed) and (!plot_foreground_only)) {
											if (!lens->sb_list[k]->zoom_subgridding) sb += lens->sb_list[k]->surface_brightness(center_srcpt[0],center_srcpt[1]);
											else {
												sb += lens->sb_list[k]->surface_brightness_zoom(center_srcpt,corner1_src,corner2_src,corner3_src,corner4_src);
											}
										}
										else {
											if (!lens->sb_list[k]->zoom_subgridding) sb += lens->sb_list[k]->surface_brightness(center_pt[0],center_pt[1]);
											else {
												corner1[0] = center_pt[0] - subpixel_xlength/2;
												corner1[1] = center_pt[1] - subpixel_ylength/2;
												corner2[0] = center_pt[0] + subpixel_xlength/2;
												corner2[1] = center_pt[1] - subpixel_ylength/2;
												corner3[0] = center_pt[0] - subpixel_xlength/2;
												corner3[1] = center_pt[1] + subpixel_ylength/2;
												corner4[0] = center_pt[0] + subpixel_xlength/2;
												corner4[1] = center_pt[1] + subpixel_ylength/2;
												sb += lens->sb_list[k]->surface_brightness_zoom(center_pt,corner1,corner2,corner3,corner4);
											}
										}
									}
								}
							}
							surface_brightness[i][j] = sb / (nsplit*nsplit);
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
				#pragma omp for private(i,j) schedule(dynamic)
				for (j=0; j < y_N; j++) {
					for (i=0; i < x_N; i++) {
						surface_brightness[i][j] = 0;
						if ((!plot_foreground_only) and (at_least_one_lensed_src)) {
							lens->find_sourcept(center_pts[i][j],center_sourcepts[i][j],thread,imggrid_zfactors,imggrid_betafactors);
						}
						for (int k=0; k < lens->n_sb; k++) {
							if ((lens->sb_list[k]->is_lensed) and (!plot_foreground_only)) {
								if (!lens->sb_list[k]->zoom_subgridding) surface_brightness[i][j] += lens->sb_list[k]->surface_brightness(center_sourcepts[i][j][0],center_sourcepts[i][j][1]);
								else {
									surface_brightness[i][j] += lens->sb_list[k]->surface_brightness_zoom(center_sourcepts[i][j],corner_sourcepts[i][j],corner_sourcepts[i+1][j],corner_sourcepts[i][j+1],corner_sourcepts[i+1][j+1]);
									//if (!subgridded) blergh << center_pts[i][j][0] << " " << center_pts[i][j][1] << " " << i << " " << j << endl;
									//else blergh << "subgrid: " << center_pt[0] << " " << center_pt[1] << " " << i << " " << j << endl;
								}
							}
							else {
								if (!lens->sb_list[k]->zoom_subgridding) surface_brightness[i][j] += lens->sb_list[k]->surface_brightness(center_pts[i][j][0],center_pts[i][j][1]);
								else {
									surface_brightness[i][j] += lens->sb_list[k]->surface_brightness_zoom(center_pts[i][j],corner_pts[i][j],corner_pts[i+1][j],corner_pts[i][j+1],corner_pts[i+1][j+1]);

								}
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

double QLens::find_surface_brightness(lensvector &pt)
{
	//double xl=0.01, yl=0.01;
	//lensvector pt1,pt2,pt3,pt4;
	//pt1[0] = pt[0] - xl/2;
	//pt1[1] = pt[1] - yl/2;

	//pt2[0] = pt[0] + xl/2;
	//pt2[1] = pt[1] - yl/2;

	//pt3[0] = pt[0] - xl/2;
	//pt3[1] = pt[1] + yl/2;

	//pt4[0] = pt[0] + xl/2;
	//pt4[1] = pt[1] + yl/2;
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
		delete[] mapped_source_pixels[i];
		delete[] surface_brightness[i];
		delete[] source_plane_triangle1_area[i];
		delete[] source_plane_triangle2_area[i];
		delete[] nsplits[i];
		delete[] twist_status[i];
		delete[] twist_pts[i];
		for (int j=0; j < y_N; j++) delete[] subpixel_maps_to_srcpixel[i][j];
		delete[] subpixel_maps_to_srcpixel[i];
	}
	delete[] center_pts;
	delete[] center_sourcepts;
	delete[] maps_to_source_pixel;
	delete[] pixel_index;
	delete[] mapped_source_pixels;
	delete[] surface_brightness;
	delete[] source_plane_triangle1_area;
	delete[] source_plane_triangle2_area;
	delete[] subpixel_maps_to_srcpixel;
	delete[] nsplits;
	delete[] twist_status;
	delete[] twist_pts;
	if (fit_to_data != NULL) {
		for (int i=0; i < x_N; i++) delete[] fit_to_data[i];
		delete[] fit_to_data;
	}
}

/************************** Functions in class QLens that pertain to pixel mapping and inversion ****************************/

bool QLens::assign_pixel_mappings(bool verbal)
{

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	int tot_npixels_count;
	tot_npixels_count = source_pixel_grid->assign_indices_and_count_levels();
	if ((mpi_id==0) and (adaptive_grid) and (verbal==true)) cout << "Number of source cells: " << tot_npixels_count << endl;
	image_pixel_grid->assign_image_mapping_flags();

	//source_pixel_grid->missed_cells_out.open("missed_cells.dat");
	source_pixel_grid->regrid = false;
	source_npixels = source_pixel_grid->assign_active_indices_and_count_source_pixels(regrid_if_unmapped_source_subpixels,activate_unmapped_source_pixels,exclude_source_pixels_beyond_fit_window);
	if (source_npixels==0) { warn("number of source pixels cannot be zero"); return false; }
	//source_pixel_grid->missed_cells_out.close();
	while (source_pixel_grid->regrid) {
		if ((mpi_id==0) and (verbal==true)) cout << "Redrawing the source grid after reverse-splitting unmapped source pixels...\n";
		source_pixel_grid->regrid = false;
		source_pixel_grid->assign_all_neighbors();
		tot_npixels_count = source_pixel_grid->assign_indices_and_count_levels();
		if ((mpi_id==0) and (verbal==true)) cout << "Number of source cells after re-gridding: " << tot_npixels_count << endl;
		image_pixel_grid->assign_image_mapping_flags();
		//source_pixel_grid->print_indices();
		source_npixels = source_pixel_grid->assign_active_indices_and_count_source_pixels(regrid_if_unmapped_source_subpixels,activate_unmapped_source_pixels,exclude_source_pixels_beyond_fit_window);
	}

	image_npixels = image_pixel_grid->n_active_pixels;
	if (active_image_pixel_i != NULL) delete[] active_image_pixel_i;
	if (active_image_pixel_j != NULL) delete[] active_image_pixel_j;
	active_image_pixel_i = new int[image_npixels];
	active_image_pixel_j = new int[image_npixels];
	int i, j, image_pixel_index=0;
	for (j=0; j < image_pixel_grid->y_N; j++) {
		for (i=0; i < image_pixel_grid->x_N; i++) {
			if (image_pixel_grid->maps_to_source_pixel[i][j]) {
				active_image_pixel_i[image_pixel_index] = i;
				active_image_pixel_j[image_pixel_index] = j;
				image_pixel_grid->pixel_index[i][j] = image_pixel_index++;
			} else image_pixel_grid->pixel_index[i][j] = -1;
		}
	}
	if (image_pixel_index != image_npixels) die("Number of active pixels (%i) doesn't seem to match image_npixels (%i)",image_pixel_index,image_npixels);

	if ((verbal) and (mpi_id==0)) cout << "source # of pixels: " << source_pixel_grid->number_of_pixels << ", counted up as " << tot_npixels_count << ", # of active pixels: " << source_npixels << endl;
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for assigning pixel mappings: " << wtime << endl;
	}
#endif

	return true;
}

void QLens::initialize_pixel_matrices(bool verbal)
{
	if (Lmatrix != NULL) die("Lmatrix already initialized");
	if (source_surface_brightness != NULL) die("source surface brightness vector already initialized");
	if (image_surface_brightness != NULL) die("image surface brightness vector already initialized");
	image_surface_brightness = new double[image_npixels];
	foreground_surface_brightness = new double[image_npixels];
	source_surface_brightness = new double[source_npixels];
	if (n_image_prior) {
		source_pixel_n_images = new double[source_npixels];
		source_pixel_grid->fill_n_image_vector();
	}

	Lmatrix_n_elements = image_pixel_grid->count_nonzero_source_pixel_mappings();
	if ((mpi_id==0) and (verbal)) cout << "Expected Lmatrix_n_elements=" << Lmatrix_n_elements << endl << flush;
	Lmatrix_index = new int[Lmatrix_n_elements];
	image_pixel_location_Lmatrix = new int[image_npixels+1];
	Lmatrix = new double[Lmatrix_n_elements];

	if ((mpi_id==0) and (verbal)) cout << "Creating Lmatrix...\n";
	assign_Lmatrix(verbal);
}

void QLens::clear_pixel_matrices()
{
	if (image_surface_brightness != NULL) delete[] image_surface_brightness;
	if (foreground_surface_brightness != NULL) delete[] foreground_surface_brightness;
	if (source_surface_brightness != NULL) delete[] source_surface_brightness;
	if (active_image_pixel_i != NULL) delete[] active_image_pixel_i;
	if (active_image_pixel_j != NULL) delete[] active_image_pixel_j;
	if (image_pixel_location_Lmatrix != NULL) delete[] image_pixel_location_Lmatrix;
	if (Lmatrix_index != NULL) delete[] Lmatrix_index;
	if (Lmatrix != NULL) delete[] Lmatrix;
	if (source_pixel_location_Lmatrix != NULL) delete[] source_pixel_location_Lmatrix;
	image_surface_brightness = NULL;
	foreground_surface_brightness = NULL;
	source_surface_brightness = NULL;
	active_image_pixel_i = NULL;
	active_image_pixel_j = NULL;
	image_pixel_location_Lmatrix = NULL;
	source_pixel_location_Lmatrix = NULL;
	Lmatrix = NULL;
	Lmatrix_index = NULL;
	for (int i=0; i < image_pixel_grid->x_N; i++) {
		for (int j=0; j < image_pixel_grid->y_N; j++) {
			image_pixel_grid->mapped_source_pixels[i][j].clear();
		}
	}
	if (n_image_prior) {
		if (source_pixel_n_images != NULL) delete[] source_pixel_n_images;
		source_pixel_n_images = NULL;
	}
}

void QLens::assign_Lmatrix(bool verbal)
{
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
	if (image_pixel_grid->ray_tracing_method == Area_Overlap)
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
				i = active_image_pixel_i[img_index];
				j = active_image_pixel_j[img_index];
				corners[0] = &image_pixel_grid->corner_sourcepts[i][j];
				corners[1] = &image_pixel_grid->corner_sourcepts[i][j+1];
				corners[2] = &image_pixel_grid->corner_sourcepts[i+1][j];
				corners[3] = &image_pixel_grid->corner_sourcepts[i+1][j+1];
				source_pixel_grid->calculate_Lmatrix_overlap(img_index,i,j,index,corners,&image_pixel_grid->twist_pts[i][j],image_pixel_grid->twist_status[i][j],thread);
				Lmatrix_row_nn[img_index] = index;
			}
		}
	}
	else if (image_pixel_grid->ray_tracing_method == Interpolate)
	{
		int max_nsplit = image_pixel_grid->max_nsplit;
		lensvector ***corner_srcpts;
		double ***subpixel_area;
		if (!weight_interpolation_by_imgplane_area) {
			corner_srcpts = new lensvector**[max_nsplit+1];
			subpixel_area = new double**[max_nsplit+1];
			for (i=0; i < max_nsplit+1; i++) {
				corner_srcpts[i] = new lensvector*[max_nsplit+1];
				subpixel_area[i] = new double*[max_nsplit+1];
				for (j=0; j < max_nsplit+1; j++) {
					corner_srcpts[i][j] = new lensvector[nthreads];
					subpixel_area[i][j] = new double[nthreads];
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
			int ii,jj,kk,ll,nsplit;
			double u0, w0, cell, atot;
			int nmaps;

			#pragma omp for private(img_index,i,j,ii,jj,kk,ll,nsplit,index,u0,w0,cell,atot,nmaps) schedule(dynamic)
			for (img_index=0; img_index < image_npixels; img_index++) {
				index=0;
				i = active_image_pixel_i[img_index];
				j = active_image_pixel_j[img_index];
				//source_pixel_grid->calculate_Lmatrix_interpolate(img_index,i,j,index,image_pixel_grid->center_sourcepts[i][j],thread);

				kk=0;
				ll=0;
				nsplit = image_pixel_grid->nsplits[i][j];
				cell = 1.0/nsplit;
				lensvector center, center_srcpt, corner;

				if (!weight_interpolation_by_imgplane_area) {
					atot=0;
					nmaps=0;
					lensvector d1, d2, d3, d4;
					ofstream wtfout;
					for (ii=0; ii < nsplit; ii++) {
						for (jj=0; jj < nsplit; jj++) {
							u0 = ((double) ii)/nsplit;
							w0 = ((double) jj)/nsplit;
							corner[0] = (u0+cell)*image_pixel_grid->corner_pts[i][j][0] + (1-(u0+cell))*image_pixel_grid->corner_pts[i+1][j][0];
							corner[1] = (w0+cell)*image_pixel_grid->corner_pts[i][j][1] + (1-(w0+cell))*image_pixel_grid->corner_pts[i][j+1][1];
							find_sourcept(corner,corner_srcpts[ii+1][jj+1][thread],thread,reference_zfactors,default_zsrc_beta_factors);

							if (ii==0) {
								corner[0] = (u0)*image_pixel_grid->corner_pts[i][j][0] + (1-(u0))*image_pixel_grid->corner_pts[i+1][j][0];
								corner[1] = (w0+cell)*image_pixel_grid->corner_pts[i][j][1] + (1-(w0+cell))*image_pixel_grid->corner_pts[i][j+1][1];
								find_sourcept(corner,corner_srcpts[ii][jj+1][thread],thread,reference_zfactors,default_zsrc_beta_factors);
							} else {
								corner[0] = (u0)*image_pixel_grid->corner_pts[i][j][0] + (1-(u0))*image_pixel_grid->corner_pts[i+1][j][0];
								corner[1] = (w0+cell)*image_pixel_grid->corner_pts[i][j][1] + (1-(w0+cell))*image_pixel_grid->corner_pts[i][j+1][1];
								lensvector check;
								find_sourcept(corner,check,thread,reference_zfactors,default_zsrc_beta_factors);
							}
							if (jj==0) {
								corner[0] = (u0+cell)*image_pixel_grid->corner_pts[i][j][0] + (1-(u0+cell))*image_pixel_grid->corner_pts[i+1][j][0];
								corner[1] = (w0)*image_pixel_grid->corner_pts[i][j][1] + (1-(w0))*image_pixel_grid->corner_pts[i][j+1][1];
								find_sourcept(corner,corner_srcpts[ii+1][jj][thread],thread,reference_zfactors,default_zsrc_beta_factors);
							}
							if ((ii==0) and (jj==0)) {
								corner[0] = (u0)*image_pixel_grid->corner_pts[i][j][0] + (1-(u0))*image_pixel_grid->corner_pts[i+1][j][0];
								corner[1] = (w0)*image_pixel_grid->corner_pts[i][j][1] + (1-(w0))*image_pixel_grid->corner_pts[i][j+1][1];
								find_sourcept(corner,corner_srcpts[ii][jj][thread],thread,reference_zfactors,default_zsrc_beta_factors);
							}
							d1[0] = corner_srcpts[ii][jj][thread][0] - corner_srcpts[ii+1][jj][thread][0];
							d1[1] = corner_srcpts[ii][jj][thread][1] - corner_srcpts[ii+1][jj][thread][1];
							d2[0] = corner_srcpts[ii][jj+1][thread][0] - corner_srcpts[ii][jj][thread][0];
							d2[1] = corner_srcpts[ii][jj+1][thread][1] - corner_srcpts[ii][jj][thread][1];
							d3[0] = corner_srcpts[ii+1][jj+1][thread][0] - corner_srcpts[ii][jj+1][thread][0];
							d3[1] = corner_srcpts[ii+1][jj+1][thread][1] - corner_srcpts[ii][jj+1][thread][1];
							d4[0] = corner_srcpts[ii+1][jj][thread][0] - corner_srcpts[ii+1][jj+1][thread][0];
							d4[1] = corner_srcpts[ii+1][jj][thread][1] - corner_srcpts[ii+1][jj+1][thread][1];

							subpixel_area[ii][jj][thread] = (0.5*(abs(d1 ^ d2) + abs (d3 ^ d4)));
							nmaps++;
							atot += subpixel_area[ii][jj][thread];
						}
					}
				}
				//if ((i==9) and (j==46)) {
					//cout << "afrac: " << afrac << " " << i << " " << j;
					//if (image_pixel_grid->twist_status[i][j]==1) cout << " (twist)";
					//else if (image_pixel_grid->twist_status[i][j]==2) cout << " (TWIST)";
					//cout << endl;
				//}

				//if ((i==15) and (j==24)) {
					//double afrac = atot / (image_pixel_grid->source_plane_triangle1_area[i][j] + image_pixel_grid->source_plane_triangle2_area[i][j]);
					//cout << "AFRAC: " << afrac << endl;
				//}

				for (ii=0; ii < nsplit; ii++) {
					for (jj=0; jj < nsplit; jj++) {
						if (image_pixel_grid->subpixel_maps_to_srcpixel[i][j][kk]) {
							u0 = ((double) (1+2*ii))*cell/2;
							w0 = ((double) (1+2*jj))*cell/2;
							center[0] = u0*image_pixel_grid->corner_pts[i][j][0] + (1-u0)*image_pixel_grid->corner_pts[i+1][j][0];
							center[1] = w0*image_pixel_grid->corner_pts[i][j][1] + (1-w0)*image_pixel_grid->corner_pts[i][j+1][1];
							find_sourcept(center,center_srcpt,thread,reference_zfactors,default_zsrc_beta_factors);
							if (weight_interpolation_by_imgplane_area) {
								source_pixel_grid->calculate_Lmatrix_interpolate(img_index,i,j,index,center_srcpt,ll,1.0/SQR(nsplit),thread); // weights by area in image plane
							} else {
								source_pixel_grid->calculate_Lmatrix_interpolate(img_index,i,j,index,center_srcpt,ll,subpixel_area[ii][jj][thread]/atot,thread); // weights by area in source plane
							}
							ll++;
						}
						kk++;
					}
				}
				Lmatrix_row_nn[img_index] = index;
			}

		}
		if (!weight_interpolation_by_imgplane_area) {
			for (i=0; i < max_nsplit+1; i++) {
				for (j=0; j < max_nsplit+1; j++) {
					delete[] corner_srcpts[i][j];
					delete[] subpixel_area[i][j];
				}
				delete[] corner_srcpts[i];
				delete[] subpixel_area[i];
			}
			delete[] corner_srcpts;
			delete[] subpixel_area;
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
		int Lmatrix_ntot = source_npixels*image_npixels;
		double sparseness = ((double) Lmatrix_n_elements)/Lmatrix_ntot;
		cout << "image has " << image_pixel_grid->n_active_pixels << " active pixels, Lmatrix has " << Lmatrix_n_elements << " nonzero elements (sparseness " << sparseness << ")\n";
	}

	delete[] Lmatrix_row_nn;
	delete[] Lmatrix_rows;
	delete[] Lmatrix_index_rows;
}

void QLens::PSF_convolution_Lmatrix(bool verbal)
{
#ifdef USE_MPI
	MPI_Comm sub_comm;
	if (psf_convolution_mpi) {
		MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
	}
#endif

	if (use_input_psf_matrix) {
		if (psf_matrix == NULL) return;
	}
	else if (generate_PSF_matrix()==false) return;
	if ((mpi_id==0) and (verbal)) cout << "Beginning PSF convolution...\n";
	double nx_half, ny_half;
	nx_half = psf_npixels_x/2;
	ny_half = psf_npixels_y/2;

	int *Lmatrix_psf_row_nn = new int[image_npixels];
	vector<double> *Lmatrix_psf_rows = new vector<double>[image_npixels];
	vector<int> *Lmatrix_psf_index_rows = new vector<int>[image_npixels];

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

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
	int psf_k, psf_l;
	int img_index1, img_index2, src_index, col_index;
	int index;
	bool new_entry;
	int Lmatrix_psf_nn=0;
	int Lmatrix_psf_nn_part=0;
	int src_index1, src_index2;
	#pragma omp parallel for private(m,k,l,i,j,img_index1,img_index2,src_index,src_index1,src_index2,col_index,psf_k,psf_l,index,new_entry) schedule(static) reduction(+:Lmatrix_psf_nn_part)
	for (img_index1=mpi_start; img_index1 < mpi_end; img_index1++)
	{ // this loops over columns of the PSF blurring matrix
		int col_i=0;
		Lmatrix_psf_row_nn[img_index1] = 0;
		k = active_image_pixel_i[img_index1];
		l = active_image_pixel_j[img_index1];
		for (psf_k=0; psf_k < psf_npixels_y; psf_k++) {
			i = k + ny_half - psf_k;
			if ((i >= 0) and (i < image_pixel_grid->x_N)) {
				for (psf_l=0; psf_l < psf_npixels_x; psf_l++) {
					j = l + nx_half - psf_l;
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
										Lmatrix_psf_rows[img_index1].push_back(psf_matrix[psf_l][psf_k]*Lmatrix[index]);
										Lmatrix_psf_index_rows[img_index1].push_back(src_index);
										Lmatrix_psf_row_nn[img_index1]++;
										col_i++;
									} else {
										Lmatrix_psf_rows[img_index1][col_index] += psf_matrix[psf_l][psf_k]*Lmatrix[index];
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

	double *Lmatrix_psf = new double[Lmatrix_psf_nn];
	int *Lmatrix_index_psf = new int[Lmatrix_psf_nn];
	int *image_pixel_location_Lmatrix_psf = new int[image_npixels+1];

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

	image_pixel_location_Lmatrix_psf[0] = 0;
	for (m=0; m < image_npixels; m++) {
		image_pixel_location_Lmatrix_psf[m+1] = image_pixel_location_Lmatrix_psf[m] + Lmatrix_psf_row_nn[m];
	}

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

void QLens::PSF_convolution_image_pixel_vector(bool verbal)
{
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	double *new_image_surface_brightness = new double[image_npixels];
	if (use_input_psf_matrix) {
		if (psf_matrix == NULL) return;
	}
	else {
		if ((psf_width_x==0) and (psf_width_y==0)) return;
		else if (generate_PSF_matrix()==false) {
			if (verbal) warn("could not generate_PSF matrix");
			return;
		}
	}
	if ((mpi_id==0) and (verbal)) cout << "Beginning PSF convolution...\n";
	double nx_half, ny_half;
	nx_half = psf_npixels_x/2;
	ny_half = psf_npixels_y/2;

	int i,j,k,l;
	int psf_k, psf_l;
	int img_index1, img_index2;
	#pragma omp parallel for private(k,l,i,j,img_index1,img_index2,psf_k,psf_l) schedule(static)
	for (img_index1=0; img_index1 < image_npixels; img_index1++)
	{ // this loops over columns of the PSF blurring matrix
		new_image_surface_brightness[img_index1] = 0;
		k = active_image_pixel_i[img_index1];
		l = active_image_pixel_j[img_index1];
		for (psf_k=0; psf_k < psf_npixels_y; psf_k++) {
			i = k + ny_half - psf_k;
			if ((i >= 0) and (i < image_pixel_grid->x_N)) {
				for (psf_l=0; psf_l < psf_npixels_x; psf_l++) {
					j = l + nx_half - psf_l;
					if ((j >= 0) and (j < image_pixel_grid->y_N)) {
						if (image_pixel_grid->maps_to_source_pixel[i][j]) {
							img_index2 = image_pixel_grid->pixel_index[i][j];
							new_image_surface_brightness[img_index1] += psf_matrix[psf_l][psf_k]*image_surface_brightness[img_index2];
						}
					}
				}
			}
		}
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating PSF convolution of image: " << wtime << endl;
	}
#endif

	delete[] image_surface_brightness;
	image_surface_brightness = new_image_surface_brightness;
}

void QLens::PSF_convolution_foreground_pixel_vector(bool verbal)
{
#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	double *new_foreground_surface_brightness = new double[image_npixels];
	if (use_input_psf_matrix) {
		if (psf_matrix == NULL) return;
	}
	else {
		if ((psf_width_x==0) and (psf_width_y==0)) return;
		else if (generate_PSF_matrix()==false) {
			if (verbal) warn("could not generate_PSF matrix");
			return;
		}
	}
	if ((mpi_id==0) and (verbal)) cout << "Beginning PSF convolution...\n";
	double nx_half, ny_half;
	nx_half = psf_npixels_x/2;
	ny_half = psf_npixels_y/2;

	int i,j,k,l;
	int psf_k, psf_l;
	int img_index1, img_index2;

	#pragma omp parallel for private(k,l,i,j,img_index1,img_index2,psf_k,psf_l) schedule(static)
	for (img_index1=0; img_index1 < image_npixels; img_index1++)
	{ // this loops over columns of the PSF blurring matrix
		new_foreground_surface_brightness[img_index1] = 0;
		k = active_image_pixel_i[img_index1];
		l = active_image_pixel_j[img_index1];
		for (psf_k=0; psf_k < psf_npixels_y; psf_k++) {
			i = k + ny_half - psf_k;
			if ((i >= 0) and (i < image_pixel_grid->x_N)) {
				for (psf_l=0; psf_l < psf_npixels_x; psf_l++) {
					j = l + nx_half - psf_l;
					if ((j >= 0) and (j < image_pixel_grid->y_N)) {
						if (image_pixel_grid->maps_to_source_pixel[i][j]) {
							img_index2 = image_pixel_grid->pixel_index[i][j];
							new_foreground_surface_brightness[img_index1] += psf_matrix[psf_l][psf_k]*foreground_surface_brightness[img_index2];
						}
					}
				}
			}
		}
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating PSF convolution of image: " << wtime << endl;
	}
#endif

	delete[] foreground_surface_brightness;
	foreground_surface_brightness = new_foreground_surface_brightness;
}

bool QLens::generate_PSF_matrix()
{
	//static const double sigma_fraction = 1.6; // the bigger you make this, the less sparse the matrix will become (more pixel-pixel correlations)
	double sigma_fraction = sqrt(-2*log(psf_threshold));
	int i,j;
	int nx_half, ny_half, nx, ny;
	double x, y, xmax, ymax;
	if ((psf_width_x==0) or (psf_width_y==0)) return false;
	double normalization = 0;
	double xstep, ystep, nx_half_dec, ny_half_dec;
	xstep = image_pixel_grid->pixel_xlength;
	ystep = image_pixel_grid->pixel_ylength;
	nx_half_dec = sigma_fraction*psf_width_x/xstep;
	ny_half_dec = sigma_fraction*psf_width_y/ystep;
	nx_half = ((int) nx_half_dec);
	ny_half = ((int) ny_half_dec);
	if ((nx_half_dec - nx_half) > 0.5) nx_half++;
	if ((ny_half_dec - ny_half) > 0.5) ny_half++;
	xmax = nx_half*xstep;
	ymax = ny_half*ystep;
	nx = 2*nx_half+1;
	ny = 2*ny_half+1;
	if (psf_matrix != NULL) {
		for (i=0; i < psf_npixels_x; i++) delete[] psf_matrix[i];
		delete[] psf_matrix;
	}
	psf_matrix = new double*[nx];
	for (i=0; i < nx; i++) psf_matrix[i] = new double[ny];
	psf_npixels_x = nx;
	psf_npixels_y = ny;
	for (i=0, x=-xmax; i < nx; i++, x += xstep) {
		for (j=0, y=-ymax; j < ny; j++, y += ystep) {
			psf_matrix[i][j] = exp(-0.5*(SQR(x/psf_width_x) + SQR(y/psf_width_y)));
			normalization += psf_matrix[i][j];
		}
	}
	for (i=0; i < nx; i++) {
		for (j=0; j < ny; j++) {
			psf_matrix[i][j] /= normalization;
		}
	}
	return true;
}

void QLens::generate_Rmatrix_from_image_plane_curvature()
{
	cout << "Generating Rmatrix from image plane curvature...\n";
	int i,j,k,l,m,n,indx;

	double curvature_submatrix[3][3] = {{0,1,0},{1,-4,1},{0,1,0}};

	int *curvature_matrix_row_nn = new int[image_npixels];
	vector<double> *curvature_matrix_rows = new vector<double>[image_npixels];
	vector<int> *curvature_matrix_index_rows = new vector<int>[image_npixels];

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif

	int curv_k, curv_l;
	int img_index1, img_index2, src_index, col_index;
	int index;
	bool new_entry;
	int curvature_matrix_nn=0;
	int curvature_matrix_nn_part=0;
	#pragma omp parallel for private(m,k,l,i,j,img_index1,img_index2,src_index,col_index,curv_k,curv_l,index,new_entry) schedule(static) reduction(+:curvature_matrix_nn_part)
	for (img_index1=0; img_index1 < image_npixels; img_index1++) {
		int col_i=0;
		curvature_matrix_row_nn[img_index1] = 0;
		k = active_image_pixel_i[img_index1];
		l = active_image_pixel_j[img_index1];
		for (curv_k=0; curv_k < 3; curv_k++) {
			i = k + 1 - curv_k;
			if ((i >= 0) and (i < image_pixel_grid->x_N)) {
				for (curv_l=0; curv_l < 3; curv_l++) {
					j = l + 1 - curv_l;
					if ((j >= 0) and (j < image_pixel_grid->y_N)) {
						if (image_pixel_grid->maps_to_source_pixel[i][j]) {
							img_index2 = image_pixel_grid->pixel_index[i][j];

							for (index=image_pixel_location_Lmatrix[img_index2]; index < image_pixel_location_Lmatrix[img_index2+1]; index++) {
								src_index = Lmatrix_index[index];
								if (curvature_submatrix[curv_l][curv_k] != 0) {
									new_entry = true;
									for (m=0; m < curvature_matrix_row_nn[img_index1]; m++) {
										if (curvature_matrix_index_rows[img_index1][m]==src_index) { col_index=m; new_entry=false; }
									}
									if (new_entry) {
										curvature_matrix_rows[img_index1].push_back(curvature_submatrix[curv_l][curv_k]*Lmatrix[index]);
										curvature_matrix_index_rows[img_index1].push_back(src_index);
										curvature_matrix_row_nn[img_index1]++;
										col_i++;
									} else {
										curvature_matrix_rows[img_index1][col_index] += curvature_submatrix[curv_l][curv_k]*Lmatrix[index];
									}
								}
							}
						}
					}
				}
			}
		}
		curvature_matrix_nn_part += col_i;
	}

	curvature_matrix_nn = curvature_matrix_nn_part;

	double *curvature_matrix;
	int *curvature_index;
	int *curvature_row_index;
	curvature_matrix = new double[curvature_matrix_nn];
	curvature_index = new int[curvature_matrix_nn];
	curvature_row_index = new int[image_npixels+1];

	curvature_row_index[0] = 0;
	for (m=0; m < image_npixels; m++) {
		curvature_row_index[m+1] = curvature_row_index[m] + curvature_matrix_row_nn[m];
	}

	for (m=0; m < image_npixels; m++) {
		indx = curvature_row_index[m];
		for (j=0; j < curvature_matrix_row_nn[m]; j++) {
			curvature_matrix[indx+j] = curvature_matrix_rows[m][j];
			curvature_index[indx+j] = curvature_matrix_index_rows[m][j];
		}
	}
	delete[] curvature_matrix_row_nn;

	vector<int> *jvals = new vector<int>[source_npixels];
	vector<int> *lvals = new vector<int>[source_npixels];

	Rmatrix_diags = new double[source_npixels];
	Rmatrix_rows = new vector<double>[source_npixels];
	Rmatrix_index_rows = new vector<int>[source_npixels];
	Rmatrix_row_nn = new int[source_npixels];
	Rmatrix_nn = 0;
	int Rmatrix_nn_part = 0;
	for (j=0; j < source_npixels; j++) {
		Rmatrix_diags[j] = 0;
		Rmatrix_row_nn[j] = 0;
	}

	int src_index1, src_index2, col_i;
	double tmp, element;

	for (i=0; i < image_npixels; i++) {
		for (j=curvature_row_index[i]; j < curvature_row_index[i+1]; j++) {
			for (l=j; l < curvature_row_index[i+1]; l++) {
				src_index1 = curvature_index[j];
				src_index2 = curvature_index[l];
				if (src_index1 > src_index2) {
					tmp=src_index1;
					src_index1=src_index2;
					src_index2=tmp;
					jvals[src_index1].push_back(l);
					lvals[src_index1].push_back(j);
				} else {
					jvals[src_index1].push_back(j);
					lvals[src_index1].push_back(l);
				}
			}
		}
	}

	#pragma omp parallel for private(i,j,k,l,m,n,src_index1,src_index2,new_entry,col_index,col_i,element) schedule(static) reduction(+:Rmatrix_nn_part)
	for (src_index1=0; src_index1 < source_npixels; src_index1++) {
		col_i=0;
		for (n=0; n < jvals[src_index1].size(); n++) {
			j = jvals[src_index1][n];
			l = lvals[src_index1][n];
			src_index2 = curvature_index[l];
			new_entry = true;
			element = curvature_matrix[j]*curvature_matrix[l]; // generalize this to full covariance matrix later
			if (src_index1==src_index2) Rmatrix_diags[src_index1] += element;
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

	delete[] curvature_matrix;
	delete[] curvature_index;
	delete[] curvature_row_index;

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
		Rmatrix[i] = Rmatrix_diags[i];

	Rmatrix_index[0] = source_npixels+1;
	for (i=0; i < source_npixels; i++) {
		Rmatrix_index[i+1] = Rmatrix_index[i] + Rmatrix_row_nn[i];
	}

	for (i=mpi_id; i < source_npixels; i += mpi_np) {
		indx = Rmatrix_index[i];
		for (j=0; j < Rmatrix_row_nn[i]; j++) {
			Rmatrix[indx+j] = Rmatrix_rows[i][j];
			Rmatrix_index[indx+j] = Rmatrix_index_rows[i][j];
		}
	}

	delete[] Rmatrix_row_nn;
	delete[] Rmatrix_diags;
	delete[] Rmatrix_rows;
	delete[] Rmatrix_index_rows;

	delete[] curvature_matrix_row_nn;
	delete[] curvature_matrix_rows;
	delete[] curvature_matrix_index_rows;

	delete[] jvals;
	delete[] lvals;

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating image plane curvature matrices: " << wtime << endl;
	}
#endif
}

void QLens::generate_Rmatrix_norm()
{
	int i,j;
	Rmatrix_diags = new double[source_npixels];
	Rmatrix_rows = new vector<double>[source_npixels];
	Rmatrix_index_rows = new vector<int>[source_npixels];
	Rmatrix_row_nn = new int[source_npixels];

	Rmatrix_nn = source_npixels+1;
	for (i=0; i < source_npixels; i++) {
		Rmatrix_row_nn[i] = 0;
		Rmatrix_diags[i] = 1;
	}

	Rmatrix = new double[Rmatrix_nn];
	Rmatrix_index = new int[Rmatrix_nn];
	for (i=0; i < source_npixels; i++) Rmatrix[i] = Rmatrix_diags[i];

	Rmatrix_index[0] = source_npixels+1;
	for (i=0; i < source_npixels; i++)
		Rmatrix_index[i+1] = Rmatrix_index[i] + Rmatrix_row_nn[i];

	int indx;
	for (i=0; i < source_npixels; i++) {
		indx = Rmatrix_index[i];
		for (j=0; j < Rmatrix_row_nn[i]; j++) {
			Rmatrix[indx+j] = Rmatrix_rows[i][j];
			Rmatrix_index[indx+j] = Rmatrix_index_rows[i][j];
			if (Rmatrix_index[indx+j] <= i) die("off-diagonal indices should be greater than i! %i vs %i",i,Rmatrix_index[indx+j]);
		}
	}
	
	delete[] Rmatrix_row_nn;
	delete[] Rmatrix_diags;
	delete[] Rmatrix_rows;
	delete[] Rmatrix_index_rows;
}

void QLens::create_regularization_matrix()
{
	if (Rmatrix != NULL) delete[] Rmatrix;
	if (Rmatrix_index != NULL) delete[] Rmatrix_index;

	int i,j;

	switch (regularization_method) {
		case Norm:
			generate_Rmatrix_norm(); break;
		case Gradient:
			generate_Rmatrix_from_gmatrices(); break;
		case Curvature:
			generate_Rmatrix_from_hmatrices(); break;
		case Image_Plane_Curvature:
			generate_Rmatrix_from_image_plane_curvature(); break;
		default:
			die("Regularization method not recognized");
	}
}

void QLens::create_lensing_matrices_from_Lmatrix(bool verbal)
{
#ifdef USE_MPI
	MPI_Comm sub_comm;
	MPI_Comm_create(*group_comm, *mpi_group, &sub_comm);
#endif

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime0 = omp_get_wtime();
	}
#endif
	//double effective_reg_parameter = regularization_parameter * (1000.0/source_npixels);
	double effective_reg_parameter = regularization_parameter;

	double covariance; // right now we're using a uniform uncorrelated noise for each pixel; will generalize this later
	if (data_pixel_noise==0) covariance = 1; // if there is no noise it doesn't matter what the covariance is, since we won't be regularizing
	else covariance = SQR(data_pixel_noise);

	int i,j,k,l,m,t;

	vector<jl_pair> **jlvals = new vector<jl_pair>*[nthreads];
	for (i=0; i < nthreads; i++) {
		jlvals[i] = new vector<jl_pair>[source_npixels];
	}

	vector<int> *Fmatrix_index_rows = new vector<int>[source_npixels];
	vector<double> *Fmatrix_rows = new vector<double>[source_npixels];
	double *Fmatrix_diags = new double[source_npixels];
	int *Fmatrix_row_nn = new int[source_npixels];
	int Fmatrix_nn = 0;
	int Fmatrix_nn_part = 0;
	for (j=0; j < source_npixels; j++) {
		Fmatrix_diags[j] = 0;
		Fmatrix_row_nn[j] = 0;
	}

	bool new_entry;
	int src_index1, src_index2, col_index, col_i;
	double tmp, element;
	Dvector = new double[source_npixels];
	for (i=0; i < source_npixels; i++) Dvector[i] = 0;

	for (i=0; i < image_npixels; i++) {
		for (j=image_pixel_location_Lmatrix[i]; j < image_pixel_location_Lmatrix[i+1]; j++) {
			Dvector[Lmatrix_index[j]] += Lmatrix[j]*(image_surface_brightness[i] - foreground_surface_brightness[i])/covariance;
		}
	}

	int mpi_chunk, mpi_start, mpi_end;
	mpi_chunk = source_npixels / group_np;
	mpi_start = group_id*mpi_chunk;
	if (group_id == group_np-1) mpi_chunk += (source_npixels % group_np); // assign the remainder elements to the last mpi process
	mpi_end = mpi_start + mpi_chunk;

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
					element = Lmatrix[j]*Lmatrix[l]/covariance; // generalize this to full covariance matrix later
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

			if (regularization_method != None) {
				Fmatrix_diags[src_index1] += effective_reg_parameter*Rmatrix[src_index1];
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
						Fmatrix_rows[src_index1].push_back(effective_reg_parameter*Rmatrix[j]);
						Fmatrix_index_rows[src_index1].push_back(Rmatrix_index[j]);
						Fmatrix_row_nn[src_index1]++;
						col_i++;
					} else {
						Fmatrix_rows[src_index1][col_index] += effective_reg_parameter*Rmatrix[j];
					}
				}
				Fmatrix_nn_part += col_i;
			}
		}
	}

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for calculating Fmatrix elements: " << wtime << endl;
		wtime0 = omp_get_wtime();
	}
#endif

#ifdef USE_MPI
	MPI_Allreduce(&Fmatrix_nn_part, &Fmatrix_nn, 1, MPI_INT, MPI_SUM, sub_comm);
#else
	Fmatrix_nn = Fmatrix_nn_part;
#endif
	Fmatrix_nn += source_npixels+1;

	Fmatrix = new double[Fmatrix_nn];
	Fmatrix_index = new int[Fmatrix_nn];

#ifdef USE_MPI
	int id, chunk, start, end, length;
	for (id=0; id < group_np; id++) {
		chunk = source_npixels / group_np;
		start = id*chunk;
		if (id == group_np-1) chunk += (source_npixels % group_np); // assign the remainder elements to the last mpi process
		MPI_Bcast(Fmatrix_row_nn + start,chunk,MPI_INT,id,sub_comm);
		MPI_Bcast(Fmatrix_diags + start,chunk,MPI_DOUBLE,id,sub_comm);
	}
#endif

	Fmatrix_index[0] = source_npixels+1;
	for (i=0; i < source_npixels; i++) {
		Fmatrix_index[i+1] = Fmatrix_index[i] + Fmatrix_row_nn[i];
	}
	if (Fmatrix_index[source_npixels] != Fmatrix_nn) die("Fmatrix # of elements don't match up (%i vs %i), process %i",Fmatrix_index[source_npixels],Fmatrix_nn,mpi_id);

	for (i=0; i < source_npixels; i++)
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
		chunk = source_npixels / group_np;
		start = id*chunk;
		if (id == group_np-1) chunk += (source_npixels % group_np); // assign the remainder elements to the last mpi process
		end = start + chunk;
		length = Fmatrix_index[end] - Fmatrix_index[start];
		MPI_Bcast(Fmatrix + Fmatrix_index[start],length,MPI_DOUBLE,id,sub_comm);
		MPI_Bcast(Fmatrix_index + Fmatrix_index[start],length,MPI_INT,id,sub_comm);
	}
	MPI_Comm_free(&sub_comm);
#endif

#ifdef USE_OPENMP
	if (show_wtime) {
		wtime = omp_get_wtime() - wtime0;
		if (mpi_id==0) cout << "Wall time for Fmatrix MPI communication + construction: " << wtime << endl;
	}
#endif

	if ((mpi_id==0) and (verbal)) cout << "Fmatrix now has " << Fmatrix_nn << " elements\n";

	if ((mpi_id==0) and (verbal)) {
		int Fmatrix_ntot = source_npixels*(source_npixels+1)/2;
		double sparseness = ((double) Fmatrix_nn)/Fmatrix_ntot;
		cout << "Fmatrix sparseness = " << sparseness << endl;
	}

	for (i=0; i < nthreads; i++) {
		delete[] jlvals[i];
	}
	delete[] jlvals;
	delete[] Fmatrix_index_rows;
	delete[] Fmatrix_rows;
	delete[] Fmatrix_diags;
	delete[] Fmatrix_row_nn;
}

void QLens::invert_lens_mapping_CG_method(bool verbal)
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
	int i,k;
	double *temp = new double[source_npixels];
	// it would be prettier to just pass the MPI communicator in, and have CG_sparse figure out the rank and # of processes internally--implement this later
	CG_sparse cg_method(Fmatrix,Fmatrix_index,1e-4,100000,inversion_nthreads,group_np,group_id);
#ifdef USE_MPI
	cg_method.set_MPI_comm(&sub_comm);
#endif
	for (int i=0; i < source_npixels; i++) temp[i] = 0;
	if (regularization_method != None)
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

	if ((n_image_prior) or (outside_sb_prior)) {
		max_pixel_sb=-1e30;
		int max_sb_i;
		for (int i=0; i < source_npixels; i++) {
			if ((data_pixel_noise==0) and (temp[i] < 0)) temp[i] = 0; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
			source_surface_brightness[i] = temp[i];
			if (source_surface_brightness[i] > max_pixel_sb) {
				max_pixel_sb = source_surface_brightness[i];
				max_sb_i = i;
			}
		}
		if (n_image_prior) {
			n_images_at_sbmax = source_pixel_n_images[max_sb_i];
			pixel_avg_n_image = 0;
			double sbtot = 0;
			for (int i=0; i < source_npixels; i++) {
				if (source_surface_brightness[i] >= max_pixel_sb*n_image_prior_sb_frac) {
					pixel_avg_n_image += source_pixel_n_images[i]*source_surface_brightness[i];
					sbtot += source_surface_brightness[i];
				}
			}
			if (sbtot != 0) pixel_avg_n_image /= sbtot;
		}
	} else {
		for (int i=0; i < source_npixels; i++) {
			if ((data_pixel_noise==0) and (temp[i] < 0)) temp[i] = 0; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
			source_surface_brightness[i] = temp[i];
		}
	}

	if (regularization_method != None) {
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
	int index=0;
	source_pixel_grid->update_surface_brightness(index);
#ifdef USE_MPI
	MPI_Comm_free(&sub_comm);
#endif
}

void QLens::invert_lens_mapping_UMFPACK(bool verbal)
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

   double *null = (double *) NULL ;
	double *temp = new double[source_npixels];
   void *Symbolic, *Numeric ;
	double Control [UMFPACK_CONTROL];
	double Info [UMFPACK_INFO];
    umfpack_di_defaults (Control) ;
	 Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;

	int Fmatrix_nonzero_elements = Fmatrix_index[source_npixels]-1;
	int Fmatrix_offdiags = Fmatrix_index[source_npixels]-1-source_npixels;
	int Fmatrix_unsymmetric_nonzero_elements = source_npixels + 2*Fmatrix_offdiags;
	if (Fmatrix_nonzero_elements==0) {
		cout << "nsource_pixels=" << source_npixels << endl;
		die("Fmatrix has zero size");
	}

	// Now we construct the transpose of Fmatrix so we can cast it into "unsymmetric" format for UMFPACK (by including offdiagonals on either side of diagonal elements)

	double *Fmatrix_transpose = new double[Fmatrix_nonzero_elements+1];
	int *Fmatrix_transpose_index = new int[Fmatrix_nonzero_elements+1];

	int k,jl,jm,jp,ju,m,n2,noff,inc,iv;
	double v;

	n2=Fmatrix_index[0];
	for (j=0; j < n2-1; j++) Fmatrix_transpose[j] = Fmatrix[j];
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
		Fmatrix_transpose[k] = Fmatrix[m];
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

	int *Fmatrix_unsymmetric_cols = new int[source_npixels+1];
	int *Fmatrix_unsymmetric_indices = new int[Fmatrix_unsymmetric_nonzero_elements];
	double *Fmatrix_unsymmetric = new double[Fmatrix_unsymmetric_nonzero_elements];

	int indx=0;
	Fmatrix_unsymmetric_cols[0] = 0;
	for (i=0; i < source_npixels; i++) {
		for (j=Fmatrix_transpose_index[i]; j < Fmatrix_transpose_index[i+1]; j++) {
			Fmatrix_unsymmetric[indx] = Fmatrix_transpose[j];
			Fmatrix_unsymmetric_indices[indx] = Fmatrix_transpose_index[j];
			indx++;
		}
		Fmatrix_unsymmetric_indices[indx] = i;
		Fmatrix_unsymmetric[indx] = Fmatrix[i];
		indx++;
		for (j=Fmatrix_index[i]; j < Fmatrix_index[i+1]; j++) {
			Fmatrix_unsymmetric[indx] = Fmatrix[j];
			Fmatrix_unsymmetric_indices[indx] = Fmatrix_index[j];
			indx++;
		}
		Fmatrix_unsymmetric_cols[i+1] = indx;
	}

	//cout << "Dvector: " << endl;
	//for (i=0; i < source_npixels; i++) {
		//cout << Dvector[i] << " ";
	//}
	//cout << endl;

	for (i=0; i < source_npixels; i++) {
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
   status = umfpack_di_symbolic(source_npixels, source_npixels, Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, &Symbolic, Control, Info);
	if (status < 0) {
		umfpack_di_report_info (Control, Info) ;
		umfpack_di_report_status (Control, status) ;
		die("Error inputting matrix");
	}
   status = umfpack_di_numeric(Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, Symbolic, &Numeric, Control, Info);
   umfpack_di_free_symbolic(&Symbolic);

   status = umfpack_di_solve(UMFPACK_A, Fmatrix_unsymmetric_cols, Fmatrix_unsymmetric_indices, Fmatrix_unsymmetric, temp, Dvector, Numeric, Control, Info);

	if (regularization_method != None) calculate_determinant = true; // specifies to calculate determinant

	if ((n_image_prior) or (outside_sb_prior)) {
		max_pixel_sb=-1e30;
		int max_sb_i;
		for (int i=0; i < source_npixels; i++) {
			source_surface_brightness[i] = temp[i];
			if (source_surface_brightness[i] > max_pixel_sb) {
				max_pixel_sb = source_surface_brightness[i];
				max_sb_i = i;
			}
		}
		if (n_image_prior) {
			n_images_at_sbmax = source_pixel_n_images[max_sb_i];
			pixel_avg_n_image = 0;
			double sbtot = 0;
			for (int i=0; i < source_npixels; i++) {
				if (source_surface_brightness[i] >= max_pixel_sb*n_image_prior_sb_frac) {
					pixel_avg_n_image += source_pixel_n_images[i]*source_surface_brightness[i];
					sbtot += source_surface_brightness[i];
				}
			}
			if (sbtot != 0) pixel_avg_n_image /= sbtot;
		}
	} else {
		for (int i=0; i < source_npixels; i++) {
			source_surface_brightness[i] = temp[i];
		}
	}
	if (calculate_determinant) {
		double mantissa, exponent;
		status = umfpack_di_get_determinant (&mantissa, &exponent, Numeric, Info) ;
		if (status < 0) {
			die("WTF!");
		}
		umfpack_di_free_numeric(&Numeric);
		Fmatrix_log_determinant = log(mantissa) + exponent*log(10);

		int Rmatrix_nonzero_elements = Rmatrix_index[source_npixels]-1;
		int Rmatrix_offdiags = Rmatrix_index[source_npixels]-1-source_npixels;
		int Rmatrix_unsymmetric_nonzero_elements = source_npixels + 2*Rmatrix_offdiags;
		if (Rmatrix_nonzero_elements==0) {
			cout << "nsource_pixels=" << source_npixels << endl;
			die("Rmatrix has zero size");
		}

		// Now we construct the transpose of Rmatrix so we can cast it into "unsymmetric" format for UMFPACK (by including offdiagonals on either side of diagonal elements)
		double *Rmatrix_transpose = new double[Rmatrix_nonzero_elements+1];
		int *Rmatrix_transpose_index = new int[Rmatrix_nonzero_elements+1];

		n2=Rmatrix_index[0];
		for (j=0; j < n2-1; j++) Rmatrix_transpose[j] = Rmatrix[j];
		n_offdiag = Rmatrix_index[n2-1] - Rmatrix_index[0];
		offdiag_indx = new int[n_offdiag];
		offdiag_indx_transpose = new int[n_offdiag];
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
		indx=0;
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
	}
	umfpack_di_free_numeric(&Numeric);

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
	int index=0;
	source_pixel_grid->update_surface_brightness(index);
#endif
}

void QLens::invert_lens_mapping_MUMPS(bool verbal)
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

	double *temp = new double[source_npixels];
	MUMPS_INT Fmatrix_nonzero_elements = Fmatrix_index[source_npixels]-1;
	if (Fmatrix_nonzero_elements==0) {
		cout << "nsource_pixels=" << source_npixels << endl;
		die("Fmatrix has zero size");
	}
	MUMPS_INT *irn = new MUMPS_INT[Fmatrix_nonzero_elements];
	MUMPS_INT *jcn = new MUMPS_INT[Fmatrix_nonzero_elements];
	double *Fmatrix_elements = new double[Fmatrix_nonzero_elements];
	for (i=0; i < source_npixels; i++) {
		Fmatrix_elements[i] = Fmatrix[i];
		irn[i] = i+1;
		jcn[i] = i+1;
		temp[i] = Dvector[i];
	}
	int indx=source_npixels;
	for (i=0; i < source_npixels; i++) {
		for (j=Fmatrix_index[i]; j < Fmatrix_index[i+1]; j++) {
			Fmatrix_elements[indx] = Fmatrix[j];
			irn[indx] = i+1;
			jcn[indx] = Fmatrix_index[j]+1;
			indx++;
		}
	}

	if (use_mumps_subcomm) {
		mumps_solver->comm_fortran=(MUMPS_INT) MPI_Comm_c2f(sub_comm);
	} else {
		mumps_solver->comm_fortran=(MUMPS_INT) MPI_Comm_c2f(this_comm);
	}
	mumps_solver->job = JOB_INIT; // initialize
	mumps_solver->sym = 2; // specifies that matrix is symmetric and positive-definite
	//cout << "ICNTL = " << mumps_solver->icntl[13] << endl;
	dmumps_c(mumps_solver);
	mumps_solver->n = source_npixels; mumps_solver->nz = Fmatrix_nonzero_elements; mumps_solver->irn=irn; mumps_solver->jcn=jcn;
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
	if (regularization_method != None) mumps_solver->icntl[32]=1; // specifies to calculate determinant
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
		MPI_Bcast(temp,source_npixels,MPI_DOUBLE,0,sub_comm);
		MPI_Barrier(sub_comm);
	}
#endif

	if (mumps_solver->info[0] < 0) {
		if (mumps_solver->info[0]==-10) die("Singular matrix, cannot invert");
		else warn("Error occurred during matrix inversion; MUMPS error code %i (source_npixels=%i)",mumps_solver->info[0],source_npixels);
	}

	if ((n_image_prior) or (outside_sb_prior)) {
		max_pixel_sb=-1e30;
		int max_sb_i;
		for (int i=0; i < source_npixels; i++) {
			//if ((data_pixel_noise==0) and (temp[i] < 0)) temp[i] = 0; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
			//if (temp[i] < -0.05) temp[i] = -0.05; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
			source_surface_brightness[i] = temp[i];
			if (source_surface_brightness[i] > max_pixel_sb) {
				max_pixel_sb = source_surface_brightness[i];
				max_sb_i = i;
			}
		}
		if (n_image_prior) {
			n_images_at_sbmax = source_pixel_n_images[max_sb_i];
			pixel_avg_n_image = 0;
			double sbtot = 0;
			for (int i=0; i < source_npixels; i++) {
				if (source_surface_brightness[i] >= max_pixel_sb*n_image_prior_sb_frac) {
					pixel_avg_n_image += source_pixel_n_images[i]*source_surface_brightness[i];
					sbtot += source_surface_brightness[i];
				}
			}
			if (sbtot != 0) pixel_avg_n_image /= sbtot;
		}
	} else {
		for (int i=0; i < source_npixels; i++) {
			if ((data_pixel_noise==0) and (temp[i] < 0)) temp[i] = 0; // This might be a bad idea, but with zero noise there should be no negatives, and they annoy me when plotted
			source_surface_brightness[i] = temp[i];
		}
	}

	if (regularization_method != None)
	{
		Fmatrix_log_determinant = log(mumps_solver->rinfog[11]) + mumps_solver->infog[33]*log(2);
		//cout << "Fmatrix log determinant = " << Fmatrix_log_determinant << endl;
		if ((mpi_id==0) and (verbal)) cout << "log determinant = " << Fmatrix_log_determinant << endl;

		mumps_solver->job=JOB_END; dmumps_c(mumps_solver); //Terminate instance

		MUMPS_INT Rmatrix_nonzero_elements = Rmatrix_index[source_npixels]-1;
		MUMPS_INT *irn_reg = new MUMPS_INT[Rmatrix_nonzero_elements];
		MUMPS_INT *jcn_reg = new MUMPS_INT[Rmatrix_nonzero_elements];
		double *Rmatrix_elements = new double[Rmatrix_nonzero_elements];
		for (i=0; i < source_npixels; i++) {
			Rmatrix_elements[i] = Rmatrix[i];
			irn_reg[i] = i+1;
			jcn_reg[i] = i+1;
		}
		indx=source_npixels;
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
	int index=0;
	source_pixel_grid->update_surface_brightness(index);
#endif
#ifdef USE_MPI
	MPI_Comm_free(&sub_comm);
	MPI_Comm_free(&this_comm);
#endif

}

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
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
			SWAP(indx[k],indx[l+1]);
			if (arr[indx[l]] > arr[indx[ir]]) {
				SWAP(indx[l],indx[ir]);
			}
			if (arr[indx[l+1]] > arr[indx[ir]]) {
				SWAP(indx[l+1],indx[ir]);
			}
			if (arr[indx[l]] > arr[indx[l+1]]) {
				SWAP(indx[l],indx[l+1]);
			}
			i=l+1;
			j=ir;
			indxt=indx[l+1];
			a=arr[indxt];
			for (;;) {
				do i++; while (arr[indx[i]] < a);
				do j--; while (arr[indx[j]] > a);
				if (j < i) break;
				SWAP(indx[i],indx[j]);
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
#undef SWAP

void QLens::clear_lensing_matrices()
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

	for (int img_index=0; img_index < image_npixels; img_index++) {
		image_surface_brightness[img_index] = 0;
		for (img_index_j=image_pixel_location_Lmatrix[img_index]; img_index_j < image_pixel_location_Lmatrix[img_index+1]; img_index_j++) {
			image_surface_brightness[img_index] += Lmatrix[img_index_j]*source_surface_brightness[Lmatrix_index[img_index_j]];
		}
		//if (image_surface_brightness[i] < 0) image_surface_brightness[i] = 0;
	}

	bool at_least_one_foreground_src = false;
	for (k=0; k < n_sb; k++) {
		if (!sb_list[k]->is_lensed) {
			at_least_one_foreground_src = true;
			break;
		}
	}
	if (at_least_one_foreground_src) {
		calculate_foreground_pixel_surface_brightness();
		add_foreground_to_image_pixel_vector();
	}

}

void QLens::calculate_foreground_pixel_surface_brightness()
{
	bool subgridded;
	int img_index;
	int i,j,k;
	bool at_least_one_foreground_src = false;
	for (k=0; k < n_sb; k++) {
		if (!sb_list[k]->is_lensed) {
			at_least_one_foreground_src = true;
			break;
		}
	}
	if (!at_least_one_foreground_src) {
		for (img_index=0; img_index < image_npixels; img_index++) foreground_surface_brightness[img_index] = 0;
		return;
	}

	for (img_index=0; img_index < image_npixels; img_index++) {
		foreground_surface_brightness[img_index] = 0;

		i = active_image_pixel_i[img_index];
		j = active_image_pixel_j[img_index];

		int ii,jj,nsplit;
		double u0, w0, sb;
		sb = 0;
		nsplit = image_pixel_grid->nsplits[i][j];
		lensvector corner1, corner2, corner3, corner4;
		double subpixel_xlength, subpixel_ylength;
		subpixel_xlength = image_pixel_grid->pixel_xlength/nsplit;
		subpixel_ylength = image_pixel_grid->pixel_ylength/nsplit;
		for (ii=0; ii < nsplit; ii++) {
			for (jj=0; jj < nsplit; jj++) {
				lensvector center_pt, center_srcpt;
				u0 = ((double) (1+2*ii))/(2*nsplit);
				w0 = ((double) (1+2*jj))/(2*nsplit);
				center_pt[0] = u0*image_pixel_grid->corner_pts[i][j][0] + (1-u0)*image_pixel_grid->corner_pts[i+1][j][0];
				center_pt[1] = w0*image_pixel_grid->corner_pts[i][j][1] + (1-w0)*image_pixel_grid->corner_pts[i][j+1][1];
				for (int k=0; k < n_sb; k++) {
					if (!sb_list[k]->is_lensed) {
						if (!sb_list[k]->zoom_subgridding) sb += sb_list[k]->surface_brightness(center_pt[0],center_pt[1]);
						else {
							corner1[0] = center_pt[0] - subpixel_xlength;
							corner1[1] = center_pt[1] - subpixel_ylength;
							corner2[0] = center_pt[0] + subpixel_xlength;
							corner2[1] = center_pt[1] - subpixel_ylength;
							corner3[0] = center_pt[0] - subpixel_xlength;
							corner3[1] = center_pt[1] + subpixel_ylength;
							corner4[0] = center_pt[0] + subpixel_xlength;
							corner4[1] = center_pt[1] + subpixel_ylength;
							sb += sb_list[k]->surface_brightness_zoom(center_pt,corner1,corner2,corner3,corner4);
						}
					}
				}
			}
		}
		foreground_surface_brightness[img_index] += sb / (nsplit*nsplit);
	}
	PSF_convolution_foreground_pixel_vector(false);
}

void QLens::add_foreground_to_image_pixel_vector()
{
	for (int img_index=0; img_index < image_npixels; img_index++) {
		image_surface_brightness[img_index] += foreground_surface_brightness[img_index];
	}
}

void QLens::store_image_pixel_surface_brightness()
{
	int i,j;
	for (i=0; i < image_pixel_grid->x_N; i++)
		for (j=0; j < image_pixel_grid->y_N; j++)
			image_pixel_grid->surface_brightness[i][j] = 0;

	for (int img_index=0; img_index < image_npixels; img_index++) {
		i = active_image_pixel_i[img_index];
		j = active_image_pixel_j[img_index];
		image_pixel_grid->surface_brightness[i][j] = image_surface_brightness[img_index];
	}
}

void QLens::vectorize_image_pixel_surface_brightness()
{
	int i,j,k=0;
	if (active_image_pixel_i == NULL) {
		delete[] active_image_pixel_i;
		if (active_image_pixel_j == NULL) delete[] active_image_pixel_j;
		image_npixels = image_pixel_grid->x_N*image_pixel_grid->y_N;
		active_image_pixel_i = new int[image_npixels];
		active_image_pixel_j = new int[image_npixels];
		for (i=0; i < image_pixel_grid->x_N; i++) {
			for (j=0; j < image_pixel_grid->y_N; j++) {
				active_image_pixel_i[k] = i;
				active_image_pixel_j[k] = j;
				image_pixel_grid->pixel_index[i][j] = k++;
			}
		}
	}
	if (image_surface_brightness == NULL) delete[] image_surface_brightness;
	image_surface_brightness = new double[image_npixels];

	for (k=0; k < image_npixels; k++) {
		i = active_image_pixel_i[k];
		j = active_image_pixel_j[k];
		image_surface_brightness[k] = image_pixel_grid->surface_brightness[i][j];
	}
}

void QLens::plot_image_pixel_surface_brightness(string outfile_root)
{
	string sb_filename = outfile_root + ".dat";
	string x_filename = outfile_root + ".x";
	string y_filename = outfile_root + ".y";

	ofstream xfile; open_output_file(xfile,x_filename);
	for (int i=0; i <= image_pixel_grid->x_N; i++) {
		xfile << image_pixel_grid->corner_pts[i][0][0] << endl;
	}

	ofstream yfile; open_output_file(yfile,y_filename);
	for (int i=0; i <= image_pixel_grid->y_N; i++) {
		yfile << image_pixel_grid->corner_pts[0][i][1] << endl;
	}

	ofstream surface_brightness_file; open_output_file(surface_brightness_file,sb_filename);
	int index=0;
	index=0;
	for (int j=0; j < image_pixel_grid->y_N; j++) {
		for (int i=0; i < image_pixel_grid->x_N; i++) {
			if (image_pixel_grid->maps_to_source_pixel[i][j])
				surface_brightness_file << image_surface_brightness[index++] << " ";
			else surface_brightness_file << "0 ";
		}
		surface_brightness_file << endl;
	}
}

