#include "profile.h"
#include "qlens.h"
#include "mathexpr.h"
#include "errors.h"
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

int Grid::nthreads = 0;
double Grid::image_pos_accuracy = 1e-6; // default
const int Grid::max_images = 50;
const int Grid::max_level = 10;
double Grid::theta_offset = 0; // slight offset in the initial angle for creating the grid; obsolete, but keeping it here just in case
double Grid::ccroot_t;
lensvector Grid::ccroot;
double Grid::cclength1, Grid::cclength2, Grid::long_diagonal_length;
bool Grid::enforce_min_area;
bool Grid::cc_neighbor_splittings;
double *Grid::grid_zfactors;
double **Grid::grid_betafactors;

// parameters for creating the recursive grid
const int Grid::u_split = 2;
const int Grid::w_split = 2;
bool Grid::radial_grid;
double Grid::rmin, Grid::rmax, Grid::xcenter, Grid::ycenter, Grid::grid_q;
int Grid::u_split_initial, Grid::w_split_initial;
int Grid::levels, Grid::splitlevels, Grid::cc_splitlevels;
double Grid::min_cell_area;

// multithreaded variables
lensvector *Grid::d1 = NULL, *Grid::d2 = NULL, *Grid::d3 = NULL, *Grid::d4 = NULL;
double *Grid::product1 = NULL, *Grid::product2 = NULL, *Grid::product3 = NULL;
bool *Grid::newton_check = NULL;
lensvector *Grid::fvec = NULL;
int *Grid::maxlevs = NULL;
lensvector ***Grid::xvals_threads = NULL;

int Grid::nfound, Grid::nfound_max, Grid::nfound_pos, Grid::nfound_neg;
bool Grid::finished_search;

int Grid::corner_positive_mag[4], Grid::corner_negative_mag[4];
lensvector Grid::ccsearch_initial_pt, Grid::ccsearch_interval;

image Grid::images[Grid::max_images];
QLens* Grid::lens = NULL;

void Grid::set_splitting(int rs0, int ts0, int sl, int ccsl, double min_cs, bool neighbor_split)
{
	u_split_initial = rs0;
	w_split_initial = ts0;
	splitlevels = sl;
	cc_splitlevels = ccsl;
	min_cell_area = min_cs;
	cc_neighbor_splittings = neighbor_split;
}

void Grid::allocate_multithreaded_variables(const int& threads, const bool reallocate)
{
	if (d1 != NULL) {
		if (!reallocate) return;
		else deallocate_multithreaded_variables();
	}
	nthreads = threads;
	d1 = new lensvector[threads];
	d2 = new lensvector[threads];
	d3 = new lensvector[threads];
	d4 = new lensvector[threads];
	product1 = new double[threads];
	product2 = new double[threads];
	product3 = new double[threads];
	newton_check = new bool[threads];
	fvec = new lensvector[threads];
	maxlevs = new int[threads];
	xvals_threads = new lensvector**[threads];
	int i,j;
	for (j=0; j < threads; j++) {
		xvals_threads[j] = new lensvector*[u_split+1];
		for (i=0; i <= u_split; i++) xvals_threads[j][i] = new lensvector[w_split+1];
	}
}

void Grid::deallocate_multithreaded_variables()
{
	if (d1 != NULL) {
		delete[] d1;
		delete[] d2;
		delete[] d3;
		delete[] d4;
		delete[] product1;
		delete[] product2;
		delete[] product3;
		delete[] newton_check;
		delete[] fvec;
		delete[] maxlevs;
		int i,j;
		for (j=0; j < nthreads; j++) {
			for (i=0; i <= u_split; i++) delete[] xvals_threads[j][i];
			delete[] xvals_threads[j];
		}
		delete[] xvals_threads;

		d1 = NULL;
		d2 = NULL;
		d3 = NULL;
		d4 = NULL;
		product1 = NULL;
		product2 = NULL;
		product3 = NULL;
		newton_check = NULL;
		fvec = NULL;
		maxlevs = NULL;
		xvals_threads = NULL;
	}
}

Grid::Grid(double xcenter_in, double ycenter_in, double xlength, double ylength, double *zfactor_in, double **betafactor_in)	// use for top-level cell only; subcells use constructor below
{
	int threads = 1;
#ifdef USE_OPENMP
	threads = omp_get_num_threads();
#endif
	allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)

	// this constructor is used for a Cartesian grid
	radial_grid = false;
	center_imgplane[0] = 0; // these should not be used for the top-level grid
	center_imgplane[1] = 0; // these should not be used for the top-level grid
	// For the Cartesian grid, u = x, w = y
	u_N = u_split_initial;
	w_N = w_split_initial;
	level = 0;
	levels = 0;
	cell = NULL;
	parent_cell = NULL;
	singular_pt_inside = false;
	cell_in_central_image_region = false;
	grid_zfactors = zfactor_in;
	grid_betafactors = betafactor_in;

	for (int i=0; i < 4; i++) {
		corner_pt[i][0]=0;
		corner_pt[i][1]=0;
		corner_sourcept[i]=NULL;
		corner_invmag[i]=NULL;
		corner_kappa[i]=NULL;
		neighbor[i]=NULL;
		allocated_corner[i]=false;
	}

	xcenter = xcenter_in; ycenter = ycenter_in;
	double x_min, x_max, y_min, y_max;
	x_min = xcenter - 0.5*xlength;
	x_max = xcenter + 0.5*xlength;
	y_min = ycenter - 0.5*ylength;
	y_max = ycenter + 0.5*ylength;

	double x, y, xstep, ystep;
	xstep = (x_max-x_min)/u_N;
	ystep = (y_max-y_min)/w_N;

	lensvector** xvals = new lensvector*[u_N+1];
	int i,j;
	for (i=0, x=x_min; i <= u_N; i++, x += xstep) {
		xvals[i] = new lensvector[w_N+1];
		for (j=0, y=y_min; j <= w_N; j++, y += ystep) {
			xvals[i][j][0] = x;
			xvals[i][j][1] = y;
		}
	}

	cell = new Grid**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new Grid*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new Grid(xvals,i,j,1,this);
		}
	}

	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			cell[i][j]->assign_lensing_properties(0);
		}
	}

	for (i=0; i < u_N+1; i++)
		delete[] xvals[i];
	delete[] xvals;

	levels++;
	assign_firstlevel_neighbors();

	assign_subcell_lensing_properties_firstlevel();

	for (i=0; i < splitlevels + cc_splitlevels - 1; i++) {
		// the second argument here, set to 'true', says to subgrid around neighbors of critical curves (this allows us to catch
		// cells that might have a curve piercing in and out of one side only; we can only detect this by breaking into smaller cells)
		split_subcells_firstlevel(i,cc_neighbor_splittings);
	}
	// don't subgrid around neighbors of critical curves for last iteration, since it's not necessary and the extra cells add overhead
	if (splitlevels + cc_splitlevels > 0) {
		if (splitlevels + cc_splitlevels==1) split_subcells_firstlevel(splitlevels + cc_splitlevels-1,cc_neighbor_splittings);
		else split_subcells_firstlevel(splitlevels + cc_splitlevels-1,false); // if more than one level of splitting, then don't split neighbors on last level (it's wasteful)
	}
}

Grid::Grid(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double *zfactor_in, double **betafactor_in) // use for top-level cell only; subcells use constructor below
{
	int threads = 1;
#ifdef USE_OPENMP
	threads = omp_get_num_threads();
#endif
	allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)

	// this constructor is used for a radial grid
	radial_grid = true;
	center_imgplane[0] = 0; // these should not be used for the top-level grid
	center_imgplane[1] = 0;
	// For the radial grid, u = r, w = theta
	u_N = u_split_initial;
	w_N = w_split_initial;
	level = 0;
	levels = 0;
	cell = NULL;
	parent_cell = NULL;
	singular_pt_inside = false;
	cell_in_central_image_region = false;
	grid_zfactors = zfactor_in;
	grid_betafactors = betafactor_in;

	int i,j;
	for (i=0; i < 4; i++) {
		corner_pt[i][0]=0;
		corner_pt[i][1]=0;
		corner_sourcept[i]=NULL;
		corner_invmag[i]=NULL;
		corner_kappa[i]=NULL;
		neighbor[i]=NULL;
		allocated_corner[i]=false;
	}

	rmin = r_min; rmax = r_max;
	xcenter = xcenter_in;
	ycenter = ycenter_in;
	grid_q = grid_q_in;

	double r, theta, rstep, thetastep;
	rstep = (rmax-rmin)/u_N;
	thetastep = 2*M_PI/w_N;

	lensvector** xvals = new lensvector*[u_N+1];
	r = rmin;
	for (i=0; i <= u_N; i++, r += rstep) {
		xvals[i] = new lensvector[w_N+1];
		theta = theta_offset;
		for (j=0; j <= w_N; j++, theta += thetastep) {
			xvals[i][j][0] = xcenter + r*cos(theta);
			xvals[i][j][1] = ycenter + grid_q*r*sin(theta);
		}
	}

	cell = new Grid**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new Grid*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new Grid(xvals,i,j,1,this);
		}
	}

	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			cell[i][j]->assign_lensing_properties(0);
		}
	}

	for (i=0; i < u_N+1; i++)
		delete[] xvals[i];
	delete[] xvals;

	levels++;
	assign_firstlevel_neighbors();
	assign_subcell_lensing_properties_firstlevel();

	for (i=0; i < splitlevels + cc_splitlevels - 1; i++) {
		// the second argument here, set to 'true', says to subgrid around neighbors of critical curves (this allows us to catch
		// cells that might have a curve piercing in and out of one side only; we can only detect this by breaking into smaller cells)
		split_subcells_firstlevel(i,cc_neighbor_splittings);
	}
	// don't subgrid around neighbors of critical curves for last iteration, since it's not necessary and the extra cells add overhead
	if (splitlevels + cc_splitlevels > 0) {
		if (splitlevels + cc_splitlevels==1) split_subcells_firstlevel(splitlevels + cc_splitlevels-1,cc_neighbor_splittings);
		else split_subcells_firstlevel(splitlevels + cc_splitlevels-1,false); // if more than one level of splitting, then don't split neighbors on last level (it's wasteful)
	}
}

Grid::Grid(lensvector** xij, const int& i, const int& j, const int& level_in, Grid* parent_ptr)
{
	u_N = 1;
	w_N = 1;
	level = level_in;
	cell = NULL;
	cc_inside = false;
	singular_pt_inside = false;
	cell_in_central_image_region = false;
	galsubgrid_cc_splitlevels = 0;
	parent_cell = parent_ptr;

	for (int k=0; k < 2; k++) {
		corner_pt[0][k] = xij[i][j][k];
		corner_pt[1][k] = xij[i][j+1][k];
		corner_pt[2][k] = xij[i+1][j][k];
		corner_pt[3][k] = xij[i+1][j+1][k];
	}

	center_imgplane[0] = (corner_pt[0][0] + corner_pt[1][0] + corner_pt[2][0] + corner_pt[3][0]) / 4.0;
	center_imgplane[1] = (corner_pt[0][1] + corner_pt[1][1] + corner_pt[2][1] + corner_pt[3][1]) / 4.0;

	corner_invmag[0] = new double;
	corner_invmag[1] = corner_invmag[2] = corner_invmag[3] = NULL;

	corner_sourcept[0] = new lensvector;
	corner_sourcept[1] = corner_sourcept[2] = corner_sourcept[3] = NULL;

	corner_kappa[0] = new double;
	corner_kappa[1] = corner_kappa[2] = corner_kappa[3] = NULL;

	allocated_corner[0] = true;
	allocated_corner[1] = allocated_corner[2] = allocated_corner[3] = false;
}

void Grid::redraw_grid(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double *zfactor_in, double **betafactor_in)  // for radial grid
{
	if (radial_grid==false) radial_grid = true;
	rmin = r_min; rmax = r_max;
	xcenter = xcenter_in;
	ycenter = ycenter_in;
	grid_q = grid_q_in;
	grid_zfactors = zfactor_in;
	grid_betafactors = betafactor_in;

	double r, theta, rstep, thetastep;
	rstep = (rmax-rmin)/u_N;
	thetastep = 2*M_PI/w_N;

	lensvector** xvals = new lensvector*[u_N+1];
	r = rmin;
	int i, j;
	for (i=0; i <= u_N; i++, r += rstep) {
		xvals[i] = new lensvector[w_N+1];
		theta = theta_offset;
		for (j=0; j <= w_N; j++, theta += thetastep) {
			xvals[i][j][0] = xcenter + r*cos(theta);
			xvals[i][j][1] = ycenter + grid_q*r*sin(theta);
		}
	}

	clear_subcells(splitlevels);
	levels = splitlevels+1;

	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif

		//#pragma omp for private(i,j) schedule(static)
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->reassign_coordinates(xvals,i,j,1,this);
				cell[i][j]->assign_lensing_properties(thread);
			}
		}
	}

	for (i=0; i < u_N+1; i++)
		delete[] xvals[i];
	delete[] xvals;

	reassign_subcell_lensing_properties_firstlevel();

//#ifdef USE_OPENMP
	//double wtime, wtime0;
//#endif
	for (i=0; i < splitlevels + cc_splitlevels - 1; i++) {
		// the second argument here, set to 'true', says to subgrid around neighbors of critical curves (this allows us to catch
		// cells that might have a curve piercing in and out of one side only; we can only detect this by breaking into smaller cells)
		split_subcells_firstlevel(i,cc_neighbor_splittings);
	}
	// don't subgrid around neighbors of critical curves for last iteration, since it's not necessary and the extra cells add overhead
	if (splitlevels + cc_splitlevels > 0) {
		if (splitlevels + cc_splitlevels==1) split_subcells_firstlevel(splitlevels + cc_splitlevels-1,cc_neighbor_splittings);
		else split_subcells_firstlevel(splitlevels + cc_splitlevels-1,false); // if more than one level of splitting, then don't split neighbors on last level (it's wasteful)
	}
}

void Grid::redraw_grid(double xcenter_in, double ycenter_in, double xlength, double ylength, double *zfactor_in, double **betafactor_in)  // for Cartesian grid
{
	if (radial_grid==true) radial_grid = false;
	xcenter = xcenter_in;
	ycenter = ycenter_in;
	grid_zfactors = zfactor_in;
	grid_betafactors = betafactor_in;

	double x_min, x_max, y_min, y_max;
	x_min = xcenter - 0.5*xlength;
	x_max = xcenter + 0.5*xlength;
	y_min = ycenter - 0.5*ylength;
	y_max = ycenter + 0.5*ylength;

	double x, y, xstep, ystep;
	xstep = (x_max-x_min)/u_N;
	ystep = (y_max-y_min)/w_N;

	lensvector** xvals = new lensvector*[u_N+1];
	int i,j;
	for (i=0, x=x_min; i <= u_N; i++, x += xstep) {
		xvals[i] = new lensvector[w_N+1];
		for (j=0, y=y_min; j <= w_N; j++, y += ystep) {
			xvals[i][j][0] = x;
			xvals[i][j][1] = y;
		}
	}

	clear_subcells(splitlevels);
	levels = splitlevels+1;

	//#pragma omp parallel
	{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif

		//#pragma omp for private(i,j) schedule(static)
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->reassign_coordinates(xvals,i,j,1,this);
				cell[i][j]->assign_lensing_properties(thread);
			}
		}
	}

	for (i=0; i < u_N+1; i++)
		delete[] xvals[i];
	delete[] xvals;

	reassign_subcell_lensing_properties_firstlevel();

//#ifdef USE_OPENMP
	//double wtime, wtime0;
//#endif
	for (i=0; i < splitlevels + cc_splitlevels - 1; i++) {
		// the second argument here, set to 'true', says to subgrid around neighbors of critical curves (this allows us to catch
		// cells that might have a curve piercing in and out of one side only; we can only detect this by breaking into smaller cells)
		split_subcells_firstlevel(i,cc_neighbor_splittings);
	}
	// don't subgrid around neighbors of critical curves for last iteration, since it's not necessary and the extra cells add overhead
	if (splitlevels + cc_splitlevels > 0) {
		if (splitlevels + cc_splitlevels==1) split_subcells_firstlevel(splitlevels + cc_splitlevels-1,cc_neighbor_splittings);
		else split_subcells_firstlevel(splitlevels + cc_splitlevels-1,false); // if more than one level of splitting, then don't split neighbors on last level (it's wasteful)
	}
}

void Grid::reassign_coordinates(lensvector** xij, const int& i, const int& j, const int& level_in, Grid* parent_ptr)
{
	u_N = 1;
	w_N = 1;
	cell = NULL;
	cc_inside = false;
	singular_pt_inside = false;
	cell_in_central_image_region = false;
	galsubgrid_cc_splitlevels = 0;


	for (int k=0; k < 2; k++) {
		corner_pt[0][k] = xij[i][j][k];
		corner_pt[1][k] = xij[i][j+1][k];
		corner_pt[2][k] = xij[i+1][j][k];
		corner_pt[3][k] = xij[i+1][j+1][k];
	}

	center_imgplane[0] = (corner_pt[0][0] + corner_pt[1][0] + corner_pt[2][0] + corner_pt[3][0]) / 4.0;
	center_imgplane[1] = (corner_pt[0][1] + corner_pt[1][1] + corner_pt[2][1] + corner_pt[3][1]) / 4.0;
}

void Grid::assign_lensing_properties(const int& thread)
{
	if (enforce_min_area) find_cell_area(thread);
	else cell_area=0;

	lens->kappa_inverse_mag_sourcept(corner_pt[0],(*corner_sourcept[0]),(*corner_kappa[0]),(*corner_invmag[0]),thread,grid_zfactors,grid_betafactors);
}

inline void Grid::set_grid_xvals(lensvector** xv, const int& i, const int& j)
{
	xv[i][j][0] = ((corner_pt[0][0]*(w_N-j) + corner_pt[1][0]*j)*(u_N-i) + (corner_pt[2][0]*(w_N-j) + corner_pt[3][0]*j)*i)/(u_N*w_N);
	xv[i][j][1] = ((corner_pt[0][1]*(w_N-j) + corner_pt[1][1]*j)*(u_N-i) + (corner_pt[2][1]*(w_N-j) + corner_pt[3][1]*j)*i)/(u_N*w_N);
}

inline void Grid::find_cell_area(const int& thread)
{
	d1[thread][0] = corner_pt[2][0] - corner_pt[0][0]; d1[thread][1] = corner_pt[2][1] - corner_pt[0][1];
	d2[thread][0] = corner_pt[1][0] - corner_pt[0][0]; d2[thread][1] = corner_pt[1][1] - corner_pt[0][1];
	d3[thread][0] = corner_pt[2][0] - corner_pt[3][0]; d3[thread][1] = corner_pt[2][1] - corner_pt[3][1];
	d4[thread][0] = corner_pt[1][0] - corner_pt[3][0]; d4[thread][1] = corner_pt[1][1] - corner_pt[3][1];
	// split cell into two triangles; cross product of the vectors forming the legs gives area of each triangle, so their sum gives area of cell
	cell_area = 0.5 * (abs(d1[thread] ^ d2[thread]) + abs(d3[thread] ^ d4[thread]));
}

void Grid::assign_firstlevel_neighbors()
{
	// neighbor index: 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	if (level != 0) die("assign_firstlevel_neighbors function must be run from grid level 0");
	int i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			if (j < w_N-1)
				cell[i][j]->neighbor[2] = cell[i][j+1];
			else {
				if (radial_grid)
					cell[i][j]->neighbor[2] = cell[i][0];
				else
					cell[i][j]->neighbor[2] = NULL;
			}

			if (j > 0) 
				cell[i][j]->neighbor[3] = cell[i][j-1];
			else {
				if (radial_grid)
					cell[i][j]->neighbor[3] = cell[i][w_N-1];
				else
					cell[i][j]->neighbor[3] = NULL;
			}

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

void Grid::assign_neighborhood()
{
	// assign neighbors of this cell, then update neighbors of neighbors of this cell
	// neighbor index: 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	int k,l;
	assign_level_neighbors(level);
	for (l=0; l < 4; l++)
		if ((neighbor[l] != NULL) and (neighbor[l]->cell != NULL)) {
		for (k=level; k <= levels; k++) {
			neighbor[l]->assign_level_neighbors(k);
		}
	}
}

void Grid::assign_all_neighbors()
{
	if (level!=0) die("assign_all_neighbors should only be run from level 0");

	int i,j,k;
	for (k=1; k < levels; k++) {
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->assign_level_neighbors(k); // we've just created our grid, so we only need to go to level+1
			}
		}
	}
}

void Grid::assign_level_neighbors(int neighbor_level)
{
	if (cell == NULL) return;
	if (level < neighbor_level) {
		int i,j;
		for (i=0; i < u_N; i++)
			for (j=0; j < w_N; j++)
				cell[i][j]->assign_level_neighbors(neighbor_level);
	} else {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				if (cell==NULL) die("cannot find neighbors if no grid has been set up");
				if (i < u_N-1)
					cell[i][j]->neighbor[0] = cell[i+1][j];
				else {
					if (neighbor[0] == NULL) cell[i][j]->neighbor[0] = NULL;
					else if (neighbor[0]->cell != NULL)
						cell[i][j]->neighbor[0] = neighbor[0]->cell[0][j];
					else
						cell[i][j]->neighbor[0] = neighbor[0];
				}

				if (i > 0)
					cell[i][j]->neighbor[1] = cell[i-1][j];
				else {
					if (neighbor[1] == NULL) cell[i][j]->neighbor[1] = NULL;
					else if (neighbor[1]->cell != NULL)
						cell[i][j]->neighbor[1] = neighbor[1]->cell[neighbor[1]->u_N-1][j];
					else
						cell[i][j]->neighbor[1] = neighbor[1];
				}

				if (j < w_N-1)
					cell[i][j]->neighbor[2] = cell[i][j+1];
				else {
					if (neighbor[2] == NULL) cell[i][j]->neighbor[2] = NULL;
					else if (neighbor[2]->cell != NULL)
						cell[i][j]->neighbor[2] = neighbor[2]->cell[i][0];
					else
						cell[i][j]->neighbor[2] = neighbor[2];
				}

				if (j > 0)
					cell[i][j]->neighbor[3] = cell[i][j-1];
				else {
					if (neighbor[3] == NULL) cell[i][j]->neighbor[3] = NULL;
					else if (neighbor[3]->cell != NULL)
						cell[i][j]->neighbor[3] = neighbor[3]->cell[i][neighbor[3]->w_N-1];
					else
						cell[i][j]->neighbor[3] = neighbor[3];
				}
			}
		}
	}
}

inline void Grid::check_if_singular_point_inside(const int& thread)
{
	// if a singular point is inside this cell, treat it as though it were a critical curve so we split around it.
	for (int i=0; i < lens->n_singular_points; i++) {
		if (test_if_inside_cell(lens->singular_pts[i],thread)) { singular_pt_inside = true; break; }
	}
}

inline void Grid::check_if_cc_inside()
{
	if (((*corner_invmag[0]) * (*corner_invmag[1]) * (*corner_invmag[2]) * (*corner_invmag[3])) < 0) cc_inside = true;
	else if ((((*corner_invmag[0]) * (*corner_invmag[1])) < 0) or ((*corner_invmag[0]) * (*corner_invmag[2])) < 0) cc_inside = true;
	else cc_inside = false;
}

inline void Grid::check_if_central_image_region()
{
	// establish whether this cell only contains central images. If it does, then we can exclude them
	// from searches if no central image is observed in the data
	cell_in_central_image_region = true;
	for (int k=0; k < 4; k++)
		if ((*corner_invmag[k] < 0) or (*corner_kappa[k] < 1)) { cell_in_central_image_region = false; break; }
}

bool Grid::split_cells(const int& thread)
{
	if (level >= max_level)
		die("maximum number of splittings has been reached (%i)", max_level);

	bool subgridded = false;
	if (cell==NULL) {
		subgridded = true;
		u_N = u_split;
		w_N = w_split;

		int i,j;
		for (i=0; i <= u_N; i++) {
			for (j=0; j <= w_N; j++) {
				xvals_threads[thread][i][j][0] = ((corner_pt[0][0]*(w_N-j) + corner_pt[1][0]*j)*(u_N-i) + (corner_pt[2][0]*(w_N-j) + corner_pt[3][0]*j)*i)/(u_N*w_N);
				xvals_threads[thread][i][j][1] = ((corner_pt[0][1]*(w_N-j) + corner_pt[1][1]*j)*(u_N-i) + (corner_pt[2][1]*(w_N-j) + corner_pt[3][1]*j)*i)/(u_N*w_N);
			}
		}

		cell = new Grid**[u_N];
		for (i=0; i < u_N; i++)
		{
			cell[i] = new Grid*[w_N];
			for (j=0; j < w_N; j++)
			{
				cell[i][j] = new Grid(xvals_threads[thread],i,j,level+1,this);
			}
		}

		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->assign_lensing_properties(thread);
			}
		}
	} else die("subcells should not already be present in split_cells routine");

	return subgridded;
}

void Grid::split_subcells_firstlevel(int cc_splitlevel, bool cc_neighbor_splitting)
{
	int i,j;
	for (i=0; i < nthreads; i++) maxlevs[i] = levels;
	//#pragma omp parallel
	//{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif

		if (cc_splitlevel > level) {
			int i,j;
			//#pragma omp for private(i,j) schedule(dynamic)
			for (i=0; i < u_N; i++) {
				for (j=0; j < w_N; j++) {
					if (cell[i][j]->cell != NULL) cell[i][j]->split_subcells(cc_splitlevel,cc_neighbor_splitting,thread);
				}
			}
		} else {
			int i,j;
			if (level >= splitlevels)
			{
				if (level < splitlevels + cc_splitlevels) {
					// check for critical curves in each grid cell, and subgrid each cell that contains a critical curve
					// (provided the subgridded cells won't be smaller than the specified min_cell_area limit)
					bool recurse = false;
					//#pragma omp for private(i,j) schedule(dynamic)
					for (i=0; i < u_N; i++) {
						for (j=0; j < w_N; j++) {
							if ((!enforce_min_area) or (cell[i][j]->cell_area > min_cell_area)) {
								if (recurse) recurse = false;
								// check to see if critical curve goes through the grid cell (or its neighbors if cc_neighbor_splitting is turned on); if so, subgrid...
								if ((cell[i][j]->cc_inside) or (cell[i][j]->singular_pt_inside)) recurse = true;
								else if (cc_neighbor_splitting) {
									for (int k=0; k < 4; k++) {
										if (cell[i][j]->neighbor[k] != NULL)
											if ((cell[i][j]->neighbor[k]->level==cell[i][j]->level) and (cell[i][j]->neighbor[k]->cc_inside)) recurse = true;
									}
								}
								if (recurse) {
									cell[i][j]->split_cells(thread);
									if (level == maxlevs[thread]-1) {
										maxlevs[thread]++; // our subcells are at the max level, so splitting them increases the number of levels by 1
									}
									//cell[i][j]->assign_neighborhood();
								}
							}
						}
					}
				}
			}
			else
			{
				// in this case we're going to subgrid regardless
					if (level == maxlevs[thread]-1) {
						maxlevs[thread]++; // our subcells are at the max level, so splitting them increases the number of levels by 1
					}
				//#pragma omp for private(i,j) schedule(dynamic)
				for (i=0; i < u_N; i++) {
					for (j=0; j < w_N; j++) {
						cell[i][j]->split_cells(thread);
					}
				}
			}
		}
	//}
	assign_neighbors_lensing_subcells(cc_splitlevel,0);
	for (i=0; i < nthreads; i++) if (maxlevs[i] > levels) levels = maxlevs[i];
}

void Grid::split_subcells(int cc_splitlevel, bool cc_neighbor_splitting, const int& thread)
{
	if (level >= max_level)
		die("maximum number of splittings has been reached (%i)", max_level);

	if (cc_splitlevel > level) {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				if (cell[i][j]->cell != NULL) cell[i][j]->split_subcells(cc_splitlevel,cc_neighbor_splitting,thread);
			}
		}
	} else {
		int i,j;
		if (level >= splitlevels)
		{
			if (level < splitlevels + cc_splitlevels) {
				// check for critical curves in each grid cell, and subgrid each cell that contains a critical curve
				// (provided the subgridded cells won't be smaller than the specified min_cell_area limit)
				bool recurse = false;
				for (i=0; i < u_N; i++) {
					for (j=0; j < w_N; j++) {
						if ((!enforce_min_area) or (cell[i][j]->cell_area > min_cell_area)) {
							if (recurse) recurse = false;
							// check to see if critical curve goes through the grid cell (or its neighbors if cc_neighbor_splitting is turned on); if so, subgrid...
							if ((cell[i][j]->cc_inside) or (cell[i][j]->singular_pt_inside)) recurse = true;
							else if (cc_neighbor_splitting) {
								for (int k=0; k < 4; k++) {
									if (cell[i][j]->neighbor[k] != NULL)
										if ((cell[i][j]->neighbor[k]->level==cell[i][j]->level) and (cell[i][j]->neighbor[k]->cc_inside)) recurse = true;
								}
							}
							if (recurse) {
								cell[i][j]->split_cells(thread);
								if (level == maxlevs[thread]-1) {
									maxlevs[thread]++; // our subcells are at the max level, so splitting them increases the number of levels by 1
								}
								//cell[i][j]->assign_neighborhood();
							}
						}
					}
				}
			}
		}
		else
		{
			// in this case we're going to subgrid regardless
				if (level == maxlevs[thread]-1) {
					maxlevs[thread]++; // our subcells are at the max level, so splitting them increases the number of levels by 1
				}
			for (i=0; i < u_N; i++) {
				for (j=0; j < w_N; j++) {
					cell[i][j]->split_cells(thread);
					//#pragma omp critical
					//{
						//cell[i][j]->assign_neighborhood();
					//}
				}
			}
		}
	}

	return;
}

void Grid::assign_neighbors_lensing_subcells(int cc_splitlevel, const int& thread)
{
	if (level >= max_level)
		die("maximum number of splittings has been reached (%i)", max_level);

	if (cc_splitlevel > level) {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				if (cell[i][j]->cell != NULL) cell[i][j]->assign_neighbors_lensing_subcells(cc_splitlevel,thread);
			}
		}
	} else {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				if (cell[i][j]->cell != NULL) {
					cell[i][j]->assign_neighborhood();
					cell[i][j]->assign_subcell_lensing_properties(thread);
				}
			}
		}
	}

	return;
}

void Grid::galsubgrid()
{
	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->galsubgrid();
			}
		}
	} else {
		u_N = u_split;
		w_N = w_split;

		lensvector** xvals = new lensvector*[u_N+1];
		int i,j;
		for (i=0; i <= u_N; i++) {
			xvals[i] = new lensvector[w_N+1];
			for (j=0; j <= w_N; j++) {
				set_grid_xvals(xvals,i,j);
			}
		}
		cell = new Grid**[u_N];
		for (i=0; i < u_N; i++) {
			cell[i] = new Grid*[w_N];
			for (j=0; j < w_N; j++) {
				cell[i][j] = new Grid(xvals,i,j,level+1,this);
				cell[i][j]->galsubgrid_cc_splitlevels = galsubgrid_cc_splitlevels;
			}
		}
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->assign_lensing_properties(0);
			}
		}
		assign_neighborhood();
		assign_subcell_lensing_properties(0);
		if (level == levels-1) {
			levels++; // our subcells are at the max level, so splitting them increases the number of levels by 1
		}
		for (i=0; i <= u_N; i++)
			delete[] xvals[i];
		delete[] xvals;
	}

	return;
}

void Grid::assign_subcell_lensing_properties_firstlevel()
{
	int i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			// only lower left-hand corner of each cell has source pt. and magnification stored in memory;
			// other corners point to the lower-left hand corner of adjacent cells, as defined below
			// unless we're at the rightmost or bottommost cell, or both (see else cases below)
			if (cell[i][j]->neighbor[2] != NULL) {
				cell[i][j]->corner_invmag[1] = cell[i][j]->neighbor[2]->corner_invmag[0];
				cell[i][j]->corner_sourcept[1] = cell[i][j]->neighbor[2]->corner_sourcept[0];
				cell[i][j]->corner_kappa[1] = cell[i][j]->neighbor[2]->corner_kappa[0];
			} else {
				cell[i][j]->corner_invmag[1] = new double;
				cell[i][j]->corner_sourcept[1] = new lensvector;
				cell[i][j]->corner_kappa[1] = new double;
				lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[1],(*cell[i][j]->corner_sourcept[1]),(*cell[i][j]->corner_kappa[1]),(*cell[i][j]->corner_invmag[1]),0,grid_zfactors,grid_betafactors);

				cell[i][j]->allocated_corner[1] = true;
			}

			if (cell[i][j]->neighbor[0] != NULL) {
				cell[i][j]->corner_invmag[2] = cell[i][j]->neighbor[0]->corner_invmag[0];
				cell[i][j]->corner_sourcept[2] = cell[i][j]->neighbor[0]->corner_sourcept[0];
				cell[i][j]->corner_kappa[2] = cell[i][j]->neighbor[0]->corner_kappa[0];
				if (cell[i][j]->neighbor[0]->neighbor[2] != NULL) {
					cell[i][j]->corner_invmag[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_invmag[0];
					cell[i][j]->corner_sourcept[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_sourcept[0];
					cell[i][j]->corner_kappa[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_kappa[0];
				} else {
					cell[i][j]->corner_invmag[3] = new double;
					cell[i][j]->corner_sourcept[3] = new lensvector;
					cell[i][j]->corner_kappa[3] = new double;
					lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[3],(*cell[i][j]->corner_sourcept[3]),(*cell[i][j]->corner_kappa[3]),(*cell[i][j]->corner_invmag[3]),0,grid_zfactors,grid_betafactors);

					cell[i][j]->allocated_corner[3] = true;
				}
			} else {
				cell[i][j]->corner_invmag[2] = new double;
				cell[i][j]->corner_sourcept[2] = new lensvector;
				cell[i][j]->corner_kappa[2] = new double;
				lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[2],(*cell[i][j]->corner_sourcept[2]),(*cell[i][j]->corner_kappa[2]),(*cell[i][j]->corner_invmag[2]),0,grid_zfactors,grid_betafactors);
				cell[i][j]->allocated_corner[2] = true;

				cell[i][j]->corner_invmag[3] = new double;
				cell[i][j]->corner_sourcept[3] = new lensvector;
				cell[i][j]->corner_kappa[3] = new double;
				lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[3],(*cell[i][j]->corner_sourcept[3]),(*cell[i][j]->corner_kappa[3]),(*cell[i][j]->corner_invmag[3]),0,grid_zfactors,grid_betafactors);
				cell[i][j]->allocated_corner[3] = true;
			}
			cell[i][j]->check_if_cc_inside();
			cell[i][j]->check_if_singular_point_inside(0);
			cell[i][j]->check_if_central_image_region();
			// Just in case we missed the critical curve when searching the larger cell...
			if (cc_inside==false)
				if (cell[i][j]->cc_inside==true) cc_inside = true;
		}
	}
}

void Grid::reassign_subcell_lensing_properties_firstlevel()
{
	int i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			// only lower left-hand corner of each cell has source pt. and magnification stored in memory;
			// other corners point to the lower-left hand corner of adjacent cells, as defined below
			// unless we're at the rightmost or bottommost cell, or both (see else cases below)
			if (cell[i][j]->neighbor[2] != NULL) {
				cell[i][j]->corner_invmag[1] = cell[i][j]->neighbor[2]->corner_invmag[0];
				cell[i][j]->corner_sourcept[1] = cell[i][j]->neighbor[2]->corner_sourcept[0];
				cell[i][j]->corner_kappa[1] = cell[i][j]->neighbor[2]->corner_kappa[0];
			} else {
				lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[1],(*cell[i][j]->corner_sourcept[1]),(*cell[i][j]->corner_kappa[1]),(*cell[i][j]->corner_invmag[1]),0,grid_zfactors,grid_betafactors);

				cell[i][j]->allocated_corner[1] = true;
			}

			if (cell[i][j]->neighbor[0] != NULL) {
				cell[i][j]->corner_invmag[2] = cell[i][j]->neighbor[0]->corner_invmag[0];
				cell[i][j]->corner_sourcept[2] = cell[i][j]->neighbor[0]->corner_sourcept[0];
				cell[i][j]->corner_kappa[2] = cell[i][j]->neighbor[0]->corner_kappa[0];
				if (cell[i][j]->neighbor[0]->neighbor[2] != NULL) {
					cell[i][j]->corner_invmag[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_invmag[0];
					cell[i][j]->corner_sourcept[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_sourcept[0];
					cell[i][j]->corner_kappa[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_kappa[0];
				} else {
					lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[3],(*cell[i][j]->corner_sourcept[3]),(*cell[i][j]->corner_kappa[3]),(*cell[i][j]->corner_invmag[3]),0,grid_zfactors,grid_betafactors);

					cell[i][j]->allocated_corner[3] = true;
				}
			} else {
				lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[2],(*cell[i][j]->corner_sourcept[2]),(*cell[i][j]->corner_kappa[2]),(*cell[i][j]->corner_invmag[2]),0,grid_zfactors,grid_betafactors);

				cell[i][j]->allocated_corner[2] = true;

				lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[3],(*cell[i][j]->corner_sourcept[3]),(*cell[i][j]->corner_kappa[3]),(*cell[i][j]->corner_invmag[3]),0,grid_zfactors,grid_betafactors);

				cell[i][j]->allocated_corner[3] = true;
			}
			cell[i][j]->check_if_cc_inside();
			cell[i][j]->check_if_singular_point_inside(0);
			cell[i][j]->check_if_central_image_region();
			// Just in case we missed the critical curve when searching the larger cell...
			if (cc_inside==false)
				if (cell[i][j]->cc_inside==true) cc_inside = true;
		}
	}
}

void Grid::assign_subcell_lensing_properties(const int& thread)
{
	cell[0][w_N-1]->corner_invmag[1] = corner_invmag[1];
	cell[u_N-1][0]->corner_invmag[2] = corner_invmag[2];
	cell[u_N-1][w_N-1]->corner_invmag[3] = corner_invmag[3];

	cell[0][w_N-1]->corner_sourcept[1] = corner_sourcept[1];
	cell[u_N-1][0]->corner_sourcept[2] = corner_sourcept[2];
	cell[u_N-1][w_N-1]->corner_sourcept[3] = corner_sourcept[3];

	cell[0][w_N-1]->corner_kappa[1] = corner_kappa[1];
	cell[u_N-1][0]->corner_kappa[2] = corner_kappa[2];
	cell[u_N-1][w_N-1]->corner_kappa[3] = corner_kappa[3];

	int i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			// only lower left-hand corner of each cell has source pt. and magnification stored in memory;
			// other corners point to the lower-left hand corner of adjacent cells, unless we're at
			// the inner or outer edges of the grid or neighboring cells are larger than our own
			if (cell[i][j]->corner_invmag[1]==NULL) {
				if ((cell[i][j]->neighbor[2] != NULL) and (cell[i][j]->neighbor[2]->level == cell[i][j]->level)) {
					cell[i][j]->corner_invmag[1] = cell[i][j]->neighbor[2]->corner_invmag[0];
					cell[i][j]->corner_sourcept[1] = cell[i][j]->neighbor[2]->corner_sourcept[0];
					cell[i][j]->corner_kappa[1] = cell[i][j]->neighbor[2]->corner_kappa[0];
				} else {
					cell[i][j]->corner_invmag[1] = new double;
					cell[i][j]->corner_sourcept[1] = new lensvector;
					cell[i][j]->corner_kappa[1] = new double;
					lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[1],(*cell[i][j]->corner_sourcept[1]),(*cell[i][j]->corner_kappa[1]),(*cell[i][j]->corner_invmag[1]),thread,grid_zfactors,grid_betafactors);

					cell[i][j]->allocated_corner[1] = true;
				}
			}

			if ((cell[i][j]->neighbor[0] != NULL) and (cell[i][j]->neighbor[0]->level == cell[i][j]->level)) {
				if (cell[i][j]->corner_invmag[2]==NULL) {
					cell[i][j]->corner_invmag[2] = cell[i][j]->neighbor[0]->corner_invmag[0];
					cell[i][j]->corner_sourcept[2] = cell[i][j]->neighbor[0]->corner_sourcept[0];
					cell[i][j]->corner_kappa[2] = cell[i][j]->neighbor[0]->corner_kappa[0];
				}
				if ((cell[i][j]->neighbor[0]->neighbor[2] != NULL) and (cell[i][j]->neighbor[0]->neighbor[2]->level == cell[i][j]->level)) {
					if (cell[i][j]->corner_invmag[3]==NULL) {
						cell[i][j]->corner_invmag[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_invmag[0];
						cell[i][j]->corner_sourcept[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_sourcept[0];
						cell[i][j]->corner_kappa[3] = cell[i][j]->neighbor[0]->neighbor[2]->corner_kappa[0];
					}
				} else {
					if (cell[i][j]->corner_invmag[3]==NULL) {
						cell[i][j]->corner_invmag[3] = new double;
						cell[i][j]->corner_sourcept[3] = new lensvector;
						cell[i][j]->corner_kappa[3] = new double;
						lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[3],(*cell[i][j]->corner_sourcept[3]),(*cell[i][j]->corner_kappa[3]),(*cell[i][j]->corner_invmag[3]),thread,grid_zfactors,grid_betafactors);

						cell[i][j]->allocated_corner[3] = true;
					}
				}
			} else {
				if (cell[i][j]->corner_invmag[2]==NULL) {
					cell[i][j]->corner_invmag[2] = new double;
					cell[i][j]->corner_sourcept[2] = new lensvector;
					cell[i][j]->corner_kappa[2] = new double;
					lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[2],(*cell[i][j]->corner_sourcept[2]),(*cell[i][j]->corner_kappa[2]),(*cell[i][j]->corner_invmag[2]),thread,grid_zfactors,grid_betafactors);
					cell[i][j]->allocated_corner[2] = true;
				}
					if (cell[i][j]->corner_invmag[3]==NULL) {
					cell[i][j]->corner_invmag[3] = new double;
					cell[i][j]->corner_sourcept[3] = new lensvector;
					cell[i][j]->corner_kappa[3] = new double;
					lens->kappa_inverse_mag_sourcept(cell[i][j]->corner_pt[3],(*cell[i][j]->corner_sourcept[3]),(*cell[i][j]->corner_kappa[3]),(*cell[i][j]->corner_invmag[3]),thread,grid_zfactors,grid_betafactors);

					cell[i][j]->allocated_corner[3] = true;
				}
			}
			cell[i][j]->check_if_cc_inside();
			if (singular_pt_inside) cell[i][j]->check_if_singular_point_inside(thread);
			cell[i][j]->check_if_central_image_region();
			// Just in case we missed the critical curve when searching the larger cell...
			if (cc_inside==false)
				if (cell[i][j]->cc_inside==true) cc_inside = true;
		}
	}
}

ofstream Grid::xgrid;

bool QLens::plot_recursive_grid(const char filename[])
{
	plot_critical_curves("crit.dat");
	if ((grid==NULL) and (create_grid(true,reference_zfactors,default_zsrc_beta_factors)==false)) return false;

	if (mpi_id==0) {
		string filelabel(filename);
		string filename_string = fit_output_dir + "/" + filelabel;
		Grid::xgrid.open(filename_string.c_str(), ifstream::out);
		grid->plot_corner_coordinates();
		Grid::xgrid.close();
	}
	return true;
}

void Grid::plot_corner_coordinates()
{
	if (level > 0) {
			xgrid << corner_pt[1][0] << " " << corner_pt[1][1] << " " << (*corner_sourcept[1])[0] << " " << (*corner_sourcept[1])[1] << endl;
			xgrid << corner_pt[3][0] << " " << corner_pt[3][1] << " " << (*corner_sourcept[3])[0] << " " << (*corner_sourcept[3])[1] << endl;
			xgrid << corner_pt[2][0] << " " << corner_pt[2][1] << " " << (*corner_sourcept[2])[0] << " " << (*corner_sourcept[2])[1] << endl;
			xgrid << corner_pt[0][0] << " " << corner_pt[0][1] << " " << (*corner_sourcept[0])[0] << " " << (*corner_sourcept[0])[1] << endl;
			xgrid << corner_pt[1][0] << " " << corner_pt[1][1] << " " << (*corner_sourcept[1])[0] << " " << (*corner_sourcept[1])[1] << endl;
			xgrid << endl;
	}

	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++)
			for (j=0; j < w_N; j++)
				cell[i][j]->plot_corner_coordinates();
	}
}

void Grid::store_critical_curve_pts()
{
	if (lens->sorted_critical_curves==true) lens->sorted_critical_curves = false;
	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++)
			for (j=0; j < w_N; j++)
				cell[i][j]->store_critical_curve_pts();
	} else if (cc_inside) {
		int added_pts = 0;
		// Try finding roots along both diagonals
		if (((*corner_invmag[0]) * (*corner_invmag[3])) < 0) find_and_store_critical_curve_pt(0,3,added_pts);
		if (((*corner_invmag[1]) * (*corner_invmag[2])) < 0) find_and_store_critical_curve_pt(1,2,added_pts);
		// Now we try the edges 0-1, 0-2; no need to check the other edges, because they will be searched from neighboring cells
		if (((*corner_invmag[0]) * (*corner_invmag[1])) < 0) find_and_store_critical_curve_pt(0,1,added_pts);
		if (((*corner_invmag[0]) * (*corner_invmag[2])) < 0) find_and_store_critical_curve_pt(0,2,added_pts);
		if (added_pts==0) warn("cell flagged erroneously as containing critical curve");
		lensvector diagonal1, diagonal2;
		diagonal1[0] = corner_pt[3][0] - corner_pt[0][0];
		diagonal1[1] = corner_pt[3][1] - corner_pt[0][1];
		diagonal2[0] = corner_pt[2][0] - corner_pt[1][0];
		diagonal2[1] = corner_pt[2][1] - corner_pt[1][1];
		cclength1 = diagonal1.norm();
		cclength2 = diagonal2.norm();
		long_diagonal_length = dmax(cclength1,cclength2);
		while (added_pts-- > 0) lens->length_of_cc_cell.push_back(long_diagonal_length);
	}
}

void Grid::find_and_store_critical_curve_pt(const int icorner, const int fcorner, int& added_pts)
{
	ccsearch_initial_pt[0] = corner_pt[icorner][0];
	ccsearch_initial_pt[1] = corner_pt[icorner][1];
	ccsearch_interval[0] = corner_pt[fcorner][0] - corner_pt[icorner][0];
	ccsearch_interval[1] = corner_pt[fcorner][1] - corner_pt[icorner][1];

	double (Brent::*invmag)(const double);
	invmag = static_cast<double (Brent::*)(const double)> (&Grid::invmag_along_diagonal);
	if ((invmag_along_diagonal(0)*invmag_along_diagonal(1)) > 0) {
		double inv0=invmag_along_diagonal(0);
		double inv1=invmag_along_diagonal(1);
		warn("critical curve root not bracketed within diagonal: invmag0=%g, invmag1=%g, invmag2=%g, invmag3=%g, (cell corner=%g,%g)",*corner_invmag[0],*corner_invmag[1],*corner_invmag[2],*corner_invmag[3],corner_pt[0][0],corner_pt[0][1]);
		return;
	}
	ccroot_t = BrentsMethod(invmag,0,1,1e-6);
	ccroot[0] = ccsearch_initial_pt[0] + ccroot_t*ccsearch_interval[0];
	ccroot[1] = ccsearch_initial_pt[1] + ccroot_t*ccsearch_interval[1];
	lens->critical_curve_pts.push_back(ccroot);
	lensvector new_srcpt;
	lens->find_sourcept(ccroot,new_srcpt,0,grid_zfactors,grid_betafactors);
	lens->caustic_pts.push_back(new_srcpt);
	added_pts++;
}

double Grid::invmag_along_diagonal(const double t)
{
	return lens->inverse_magnification(ccsearch_initial_pt + t*ccsearch_interval,0,grid_zfactors,grid_betafactors);
}

inline bool Grid::image_test(const int& thread)
{
	// This function is similar to test_if_inside_sourceplane_cell(...), except
	// it explicitly uses the source point whose images are being searched for.
	// We check to see if the given cell, when mapped to the source plane, contains 
	// the source point (in which case an image is inside the cell).
	// The method: split the cell into two triangles. For each triangle, define 
	// vectors from the source point to each corner of the triangle. If the 
	// source point is within one of the triangles, then the cross products of 
	// the vectors will all have the same sign (provided the order of the cross 
	// products is cyclic: 1x2, 2x3, 3x1).

	d1[thread][0] = lens->source[0] - (*corner_sourcept[1])[0];
	d1[thread][1] = lens->source[1] - (*corner_sourcept[1])[1];
	d2[thread][0] = lens->source[0] - (*corner_sourcept[2])[0];
	d2[thread][1] = lens->source[1] - (*corner_sourcept[2])[1];
	d3[thread][0] = lens->source[0] - (*corner_sourcept[0])[0];
	d3[thread][1] = lens->source[1] - (*corner_sourcept[0])[1];
	product1[thread] = d1[thread] ^ d2[thread];
	product2[thread] = d3[thread] ^ d1[thread];
	product3[thread] = d2[thread] ^ d3[thread];
	if ((product1[thread] > 0) and (product2[thread] > 0) and (product3[thread] > 0)) return true;
	if ((product1[thread] < 0) and (product2[thread] < 0) and (product3[thread] < 0)) return true;

	if (product1[thread] > 0) {
		if ((abs(product2[thread])==0) and (product3[thread] > 0)) return true;
		if ((product2[thread] > 0) and (abs(product3[thread])==0)) return true;
	} else if (product1[thread] < 0) {
		if ((abs(product2[thread])==0) and (product3[thread] < 0)) return true;
		if ((product2[thread] < 0) and (abs(product3[thread])==0)) return true;
	}

	d3[thread][0] = lens->source[0] - (*corner_sourcept[3])[0];
	d3[thread][1] = lens->source[1] - (*corner_sourcept[3])[1];
	product2[thread] = d3[thread] ^ d1[thread];
	product3[thread] = d2[thread] ^ d3[thread];
	if ((product1[thread] > 0) and (product2[thread] > 0) and (product3[thread] > 0)) return true;
	if ((product1[thread] < 0) and (product2[thread] < 0) and (product3[thread] < 0)) return true;

	return false;	// source not enclosed, therefore no images in this cell
}

inline bool Grid::test_if_sourcept_inside_triangle(lensvector* point1, lensvector* point2, lensvector* point3, const int& thread)
{
	// Check to see if the given cell, when mapped to the source plane, contains 
	// the point in question.
	// The method: split the cell into two triangles. For each triangle, define 
	// vectors from the source point to each corner of the triangle. If the 
	// point is within one of the triangles, then the cross products of 
	// the vectors will all have the same sign (provided the order of the cross 
	// products is cyclic: 1x2, 2x3, 3x1).

	d1[thread][0] = lens->source[0] - (*point1)[0];
	d1[thread][1] = lens->source[1] - (*point1)[1];
	d2[thread][0] = lens->source[0] - (*point2)[0];
	d2[thread][1] = lens->source[1] - (*point2)[1];
	d3[thread][0] = lens->source[0] - (*point3)[0];
	d3[thread][1] = lens->source[1] - (*point3)[1];
	product1[thread] = d1[thread] ^ d2[thread];
	product2[thread] = d3[thread] ^ d1[thread];
	product3[thread] = d2[thread] ^ d3[thread];
	if ((product1[thread] > 0) and (product2[thread] > 0) and (product3[thread] > 0)) return true;
	if ((product1[thread] < 0) and (product2[thread] < 0) and (product3[thread] < 0)) return true;

	return false;	// point not enclosed
}

inside_cell Grid::test_if_inside_sourceplane_cell(lensvector* point, const int& thread)
{
	// Check to see if the given cell, when mapped to the source plane, contains 
	// the point in question.
	// The method: split the cell into two triangles. For each triangle, define 
	// vectors from the source point to each corner of the triangle. If the 
	// point is within one of the triangles, then the cross products of 
	// the vectors will all have the same sign (provided the order of the cross 
	// products is cyclic: 1x2, 2x3, 3x1).

	d1[thread][0] = (*point)[0] - (*corner_sourcept[1])[0];
	d1[thread][1] = (*point)[1] - (*corner_sourcept[1])[1];
	d2[thread][0] = (*point)[0] - (*corner_sourcept[2])[0];
	d2[thread][1] = (*point)[1] - (*corner_sourcept[2])[1];
	d3[thread][0] = (*point)[0] - (*corner_sourcept[0])[0];
	d3[thread][1] = (*point)[1] - (*corner_sourcept[0])[1];
	product1[thread] = d1[thread] ^ d2[thread];
	product2[thread] = d3[thread] ^ d1[thread];
	product3[thread] = d2[thread] ^ d3[thread];
	if ((product1[thread] > 0) and (product2[thread] > 0) and (product3[thread] > 0)) return Inside;
	if ((product1[thread] < 0) and (product2[thread] < 0) and (product3[thread] < 0)) return Inside;

	// if the point is on the "low r" or "low theta" edge of the cell, count it as inside
	if (product1[thread] > 0) {
		if ((abs(product2[thread])==0) and (product3[thread] > 0)) return Edge;
		if ((product2[thread] > 0) and (abs(product3[thread])==0)) return Edge;
	} else if (product1[thread] < 0) {
		if ((abs(product2[thread])==0) and (product3[thread] < 0)) return Edge;
		if ((product2[thread] < 0) and (abs(product3[thread])==0)) return Edge;
	}

	d3[thread][0] = (*point)[0] - (*corner_sourcept[3])[0];
	d3[thread][1] = (*point)[1] - (*corner_sourcept[3])[1];
	product2[thread] = d3[thread] ^ d1[thread];
	product3[thread] = d2[thread] ^ d3[thread];
	if ((product1[thread] > 0) and (product2[thread] > 0) and (product3[thread] > 0)) return Inside;
	if ((product1[thread] < 0) and (product2[thread] < 0) and (product3[thread] < 0)) return Inside;

	return Outside;	// point not enclosed
}

inline bool Grid::test_if_inside_cell(const lensvector& point, const int& thread)
{
	// The method: split the cell into two triangles. For each triangle, define 
	// vectors from the point to each corner of the triangle. If the 
	// point is within one of the triangles, then the cross products of 
	// the vectors will all have the same sign (provided the order of the cross 
	// products is cyclic: 1x2, 2x3, 3x1).

	d1[thread][0] = point[0] - corner_pt[1][0];
	d1[thread][1] = point[1] - corner_pt[1][1];
	d2[thread][0] = point[0] - corner_pt[2][0];
	d2[thread][1] = point[1] - corner_pt[2][1];
	d3[thread][0] = point[0] - corner_pt[0][0];
	d3[thread][1] = point[1] - corner_pt[0][1];
	product1[thread] = d1[thread] ^ d2[thread];
	product2[thread] = d3[thread] ^ d1[thread];
	product3[thread] = d2[thread] ^ d3[thread];
	if ((product1[thread] > 0) and (product2[thread] > 0) and (product3[thread] > 0)) return true;
	if ((product1[thread] < 0) and (product2[thread] < 0) and (product3[thread] < 0)) return true;

	// check to see whether point is just outside the edges of the cell, within the accuracy set
	// for image searching
	lensvector sidevec;
	if (product1[thread] > 0) {
		if (product3[thread] > 0) {
			sidevec = d3[thread] - d1[thread];
			if (abs(product2[thread])/sidevec.norm() < 2*image_pos_accuracy) return true;
		}
		if (product2[thread] > 0) {
			sidevec = d2[thread] - d3[thread];
			if (abs(product3[thread])/sidevec.norm() < 2*image_pos_accuracy) return true;
		}
	} else if (product1[thread] < 0) {
		if (product3[thread] < 0) {
			sidevec = d3[thread] - d1[thread];
			if (abs(product2[thread])/sidevec.norm() < 2*image_pos_accuracy) return true;
		}
		if (product2[thread] < 0) {
			sidevec = d2[thread] - d3[thread];
			if (abs(product3[thread])/sidevec.norm() < 2*image_pos_accuracy) return true;
		}
	}

	d3[thread][0] = point[0] - corner_pt[3][0];
	d3[thread][1] = point[1] - corner_pt[3][1];
	product2[thread] = d3[thread] ^ d1[thread];
	product3[thread] = d2[thread] ^ d3[thread];

	// check to see whether point is just outside the edges of the cell, within the accuracy set
	// for image searching
	if (product1[thread] > 0) {
		if (product3[thread] > 0) {
			sidevec = d3[thread] - d1[thread];
			if (abs(product2[thread])/sidevec.norm() < 2*image_pos_accuracy) return true;
		}
		if (product2[thread] > 0) {
			sidevec = d2[thread] - d3[thread];
			if (abs(product3[thread])/sidevec.norm() < 2*image_pos_accuracy) return true;
		}
	} else if (product1[thread] < 0) {
		if (product3[thread] < 0) {
			sidevec = d3[thread] - d1[thread];
			if (abs(product2[thread])/sidevec.norm() < 2*image_pos_accuracy) return true;
		}
		if (product2[thread] < 0) {
			sidevec = d2[thread] - d3[thread];
			if (abs(product3[thread])/sidevec.norm() < 2*image_pos_accuracy) return true;
		}
	}

	if ((product1[thread] > 0) and (product2[thread] > 0) and (product3[thread] > 0)) return true;
	if ((product1[thread] < 0) and (product2[thread] < 0) and (product3[thread] < 0)) return true;

	return false;	// point not enclosed within cell
}

inline bool Grid::test_if_galaxy_nearby(const lensvector& point, const double& distsq)
{
	lensvector d_corner;
	for (int i=0; i < 4; i++) {
		d_corner[0] = point[0] - corner_pt[i][0];
		d_corner[1] = point[1] - corner_pt[i][1];
		if (d_corner.sqrnorm() < distsq) return true;
	}
	lensvector disp;
	int k;
	for (k=0; k < 2; k++) disp[k] = point[k] - 0.5*(corner_pt[0][k]+corner_pt[1][k]);
	if (disp.sqrnorm() < distsq) return true;
	for (k=0; k < 2; k++) disp[k] = point[k] - 0.5*(corner_pt[0][k]+corner_pt[2][k]);
	if (disp.sqrnorm() < distsq) return true;
	for (k=0; k < 2; k++) disp[k] = point[k] - 0.5*(corner_pt[1][k]+corner_pt[3][k]);
	if (disp.sqrnorm() < distsq) return true;
	for (k=0; k < 2; k++) disp[k] = point[k] - 0.5*(corner_pt[2][k]+corner_pt[3][k]);
	if (disp.sqrnorm() < distsq) return true;

	// final check: test four points on the edges of the galaxy region to see if they are inside this cell
	lensvector nearby_pt;
	disp[0] = sqrt(distsq);
	disp[1] = 0;
	nearby_pt[0] = point[0] + disp[0];
	nearby_pt[1] = point[1] + disp[1];
	if (test_if_inside_cell(nearby_pt,0)==true) return true;
	nearby_pt[0] = point[0] - disp[0];
	nearby_pt[1] = point[1] - disp[1];
	if (test_if_inside_cell(nearby_pt,0)==true) return true;
	disp[0] = 0;
	disp[1] = sqrt(distsq);
	nearby_pt[0] = point[0] + disp[0];
	nearby_pt[1] = point[1] + disp[1];
	if (test_if_inside_cell(nearby_pt,0)==true) return true;
	nearby_pt[0] = point[0] - disp[0];
	nearby_pt[1] = point[1] - disp[1];
	if (test_if_inside_cell(nearby_pt,0)==true) return true;

	return false;
}

edge_sourcept_status Grid::check_subgrid_neighbor_boundaries(const int& neighbor_direction, Grid* neighbor_subcell, lensvector& centerpt, const int& thread)
{
	edge_sourcept_status status = NoSource;
	inside_cell inside_sourceplane_cell;
	lensvector *interior_edge_point, *edgept1, *edgept2;
	lensvector *interior_edge_point_src, *edgept1_src, *edgept2_src;
	bool edgept1_parity, edgept2_parity; // make static multithreaded variables?
	if (neighbor_direction==0) {
		interior_edge_point_src = neighbor_subcell->cell[0][0]->corner_sourcept[1];
		edgept1_src = neighbor_subcell->corner_sourcept[0];
		edgept2_src = neighbor_subcell->corner_sourcept[1];
		edgept1_parity = sign_bool(*neighbor_subcell->cell[0][0]->corner_invmag[0]);
		edgept2_parity = sign_bool(*neighbor_subcell->cell[0][0]->corner_invmag[1]);
		d1[thread][0] = (*edgept1_src)[0] - (*edgept2_src)[0];
		d1[thread][1] = (*edgept1_src)[1] - (*edgept2_src)[1];
		d2[thread][0] = (*interior_edge_point_src)[0] - (*edgept2_src)[0];
		d2[thread][1] = (*interior_edge_point_src)[1] - (*edgept2_src)[1];
	} else if (neighbor_direction==1) {
		interior_edge_point_src = neighbor_subcell->cell[1][0]->corner_sourcept[3];
		edgept1_src = neighbor_subcell->corner_sourcept[2];
		edgept2_src = neighbor_subcell->corner_sourcept[3];
		edgept1_parity = sign_bool(*neighbor_subcell->cell[1][0]->corner_invmag[2]);
		edgept2_parity = sign_bool(*neighbor_subcell->cell[1][0]->corner_invmag[3]);
		d1[thread][0] = (*edgept2_src)[0] - (*edgept1_src)[0];
		d1[thread][1] = (*edgept2_src)[1] - (*edgept1_src)[1];
		d2[thread][0] = (*interior_edge_point_src)[0] - (*edgept1_src)[0];
		d2[thread][1] = (*interior_edge_point_src)[1] - (*edgept1_src)[1];
	} else if (neighbor_direction==2) {
		interior_edge_point_src = neighbor_subcell->cell[0][0]->corner_sourcept[2];
		edgept1_src = neighbor_subcell->corner_sourcept[0];
		edgept2_src = neighbor_subcell->corner_sourcept[2];
		edgept1_parity = sign_bool(*neighbor_subcell->cell[0][0]->corner_invmag[0]);
		edgept2_parity = sign_bool(*neighbor_subcell->cell[0][0]->corner_invmag[2]);
		d1[thread][0] = (*edgept2_src)[0] - (*edgept1_src)[0];
		d1[thread][1] = (*edgept2_src)[1] - (*edgept1_src)[1];
		d2[thread][0] = (*interior_edge_point_src)[0] - (*edgept1_src)[0];
		d2[thread][1] = (*interior_edge_point_src)[1] - (*edgept1_src)[1];
	} else if (neighbor_direction==3) {
		interior_edge_point_src = neighbor_subcell->cell[0][1]->corner_sourcept[3];
		edgept1_src = neighbor_subcell->corner_sourcept[1];
		edgept2_src = neighbor_subcell->corner_sourcept[3];
		edgept1_parity = sign_bool(*neighbor_subcell->cell[0][1]->corner_invmag[1]);
		edgept2_parity = sign_bool(*neighbor_subcell->cell[0][1]->corner_invmag[3]);
		d1[thread][0] = (*edgept1_src)[0] - (*edgept2_src)[0];
		d1[thread][1] = (*edgept1_src)[1] - (*edgept2_src)[1];
		d2[thread][0] = (*interior_edge_point_src)[0] - (*edgept2_src)[0];
		d2[thread][1] = (*interior_edge_point_src)[1] - (*edgept2_src)[1];
	}
	if (edgept1_parity == edgept2_parity) {
		product1[thread] = d1[thread] ^ d2[thread];
		if (edgept1_parity == true) inside_sourceplane_cell = (product1[thread] < 0) ? Inside : (product1[thread] > 0) ? Outside : Edge; // positive parity
		else inside_sourceplane_cell = (product1[thread] > 0) ? Inside : (product1[thread] < 0) ? Outside : Edge; // negative parity
	} else {
		inside_sourceplane_cell = test_if_inside_sourceplane_cell(interior_edge_point_src,thread);
	}
	if (inside_sourceplane_cell==Outside) {
		if (test_if_sourcept_inside_triangle(edgept1_src,edgept2_src,interior_edge_point_src,thread)==true) {
			if (neighbor_direction==0) {
				interior_edge_point = &neighbor_subcell->cell[0][0]->corner_pt[1];
				edgept1 = &neighbor_subcell->corner_pt[0];
				edgept2 = &neighbor_subcell->corner_pt[1];
			}
			else if (neighbor_direction==1) {
				interior_edge_point = &neighbor_subcell->cell[1][0]->corner_pt[3];
				edgept1 = &neighbor_subcell->corner_pt[2];
				edgept2 = &neighbor_subcell->corner_pt[3];
			}
			else if (neighbor_direction==2) {
				interior_edge_point = &neighbor_subcell->cell[0][0]->corner_pt[2];
				edgept1 = &neighbor_subcell->corner_pt[0];
				edgept2 = &neighbor_subcell->corner_pt[2];
			}
			else if (neighbor_direction==3) {
				interior_edge_point = &neighbor_subcell->cell[0][1]->corner_pt[3];
				edgept1 = &neighbor_subcell->corner_pt[1];
				edgept2 = &neighbor_subcell->corner_pt[3];
			}
			centerpt[0] = ((*edgept1)[0] + (*edgept2)[0] + (*interior_edge_point)[0])/3.0;
			centerpt[1] = ((*edgept1)[1] + (*edgept2)[1] + (*interior_edge_point)[1])/3.0;
			status = SourceInGap;
		}
	}
	else if (inside_sourceplane_cell==Inside)
	{
		if (test_if_sourcept_inside_triangle(edgept1_src,edgept2_src,interior_edge_point_src,thread)==true)
			status = SourceInOverlap;
	}
	edge_sourcept_status parent_edge_sourcept_status;
	if (neighbor_subcell == neighbor[neighbor_direction]) parent_edge_sourcept_status = status;
	edge_sourcept_status substatus1 = NoSource, substatus2 = NoSource;
	if (neighbor_direction==0) {
		if (neighbor_subcell->cell[0][0]->cell != NULL)
			substatus1 = check_subgrid_neighbor_boundaries(neighbor_direction, neighbor_subcell->cell[0][0], centerpt, thread);
		if (neighbor_subcell->cell[0][1]->cell != NULL)
			substatus2 = check_subgrid_neighbor_boundaries(neighbor_direction, neighbor_subcell->cell[0][1], centerpt, thread);
	}
	else if (neighbor_direction==1) {
		if (neighbor_subcell->cell[1][0]->cell != NULL)
			substatus1 = check_subgrid_neighbor_boundaries(neighbor_direction, neighbor_subcell->cell[1][0], centerpt, thread);
		if (neighbor_subcell->cell[1][1]->cell != NULL)
			substatus2 = check_subgrid_neighbor_boundaries(neighbor_direction, neighbor_subcell->cell[1][1], centerpt, thread);
	}
	else if (neighbor_direction==2) {
		if (neighbor_subcell->cell[0][0]->cell != NULL)
			substatus1 = check_subgrid_neighbor_boundaries(neighbor_direction, neighbor_subcell->cell[0][0], centerpt, thread);
		if (neighbor_subcell->cell[1][0]->cell != NULL)
			substatus2 = check_subgrid_neighbor_boundaries(neighbor_direction, neighbor_subcell->cell[1][0], centerpt, thread);
	}
	else if (neighbor_direction==3) {
		if (neighbor_subcell->cell[0][1]->cell != NULL)
			substatus1 = check_subgrid_neighbor_boundaries(neighbor_direction, neighbor_subcell->cell[0][1], centerpt, thread);
		if (neighbor_subcell->cell[1][1]->cell != NULL)
			substatus2 = check_subgrid_neighbor_boundaries(neighbor_direction, neighbor_subcell->cell[1][1], centerpt, thread);
	}
	
	if (status==SourceInGap) {
		if ((substatus1==SourceInOverlap) or (substatus2==SourceInOverlap)) {
			status = SourceInOverlap; // the smallest neighboring subcells actually contain the source point (even though the larger subcells didn't)
		}
	} else if (status==SourceInOverlap) {
		if ((substatus1==SourceInGap) or (substatus2==SourceInGap)) {
			 // in this case, the smallest neighboring subcells don't actually contain the source point
			if (parent_edge_sourcept_status==SourceInOverlap) {
				status = NoSource; // since the source point is contained in our cell, we can just search for it in our cell as usual
			} else {
				status = SourceInGap; // since the source point is not containe in our cell, we must search the triangle gap containing the source point
			}
		}
	} else if (status==NoSource) {
		if ((substatus1==SourceInGap) or (substatus2==SourceInGap)) status = SourceInGap;
		else if ((substatus1==SourceInOverlap) or (substatus2==SourceInOverlap)) status = SourceInOverlap;
	}

	return status;
}

void Grid::grid_search_firstlevel(const int& searchlevel)
{
	// There seems to be negligible time savings in multi-threading the grid search, and it doesn't seem to be working right, so I've commented it out for now.
	//#pragma omp parallel
	//{
		int thread;
//#ifdef USE_OPENMP
		//thread = omp_get_thread_num();
//#else
		thread = 0;
//#endif

		int ntot = u_N*w_N;
		int k,i,j;
		//#pragma omp for schedule(dynamic)
		for (k=0; k < ntot; k++) {
			j = k / u_N;
			i = k % u_N;
			cell[i][j]->grid_search(searchlevel,thread);
		}
	//}
}

void Grid::grid_search(const int& searchlevel, const int& thread)
{
	if (finished_search) return;
	if ((lens->include_central_image==false) and (cell_in_central_image_region==true)) return;
	// 'searchlevel' specifies level at which we should start hunting for images.
	// If the level is at or above the searchlevel, start searching for images;
	// otherwise, descend into higher level until we reach the specified searchlevel

	if ((cell != NULL) and (level < searchlevel))
	{
		int i,j;
		for (j=0; j < w_N; j++) {
			for (i=0; i < u_N; i++) {
				if (finished_search) break;
				cell[i][j]->grid_search(searchlevel,thread);
			}
		}
	}
	else if (!singular_pt_inside)
	{
		bool cell_maps_around_sourcept = image_test(thread);
		lensvector imgpos;
		if (cell==NULL) {
			for (int i=0; i < 4; i++) {
				if ((neighbor[i] != NULL) and (neighbor[i]->cell != NULL)) {
					edge_sourcept_status status = check_subgrid_neighbor_boundaries(i, neighbor[i], imgpos, thread); // imgpos is set to the center of the triangle if source is found in the corresponding ray-traced triangle
					if (status==SourceInGap) {
						if ((lens->skip_newtons_method) or (run_newton(imgpos,thread)==true))
							add_image_to_list(imgpos);
					} else if (status==SourceInOverlap) {
						cell_maps_around_sourcept = false; // even if this cell maps around the source, don't search if it overlaps with neighboring subcells
					}
				}
			}
		}
		if (cell_maps_around_sourcept) {
			imgpos[0] = center_imgplane[0];
			imgpos[1] = center_imgplane[1];
			if ((lens->skip_newtons_method) or (run_newton(imgpos,thread)==true)) {
				add_image_to_list(imgpos);
			}
		}
	}
}
void Grid::add_image_to_list(const lensvector& imgpos)
{
	images[nfound].pos[0] = imgpos[0];
	images[nfound].pos[1] = imgpos[1];
	images[nfound].mag = lens->magnification(imgpos,0,grid_zfactors,grid_betafactors);
	if (lens->include_time_delays) {
		double potential = lens->potential(imgpos,grid_zfactors,grid_betafactors);
		images[nfound].td = 0.5*(SQR(imgpos[0]-lens->source[0])+SQR(imgpos[1]-lens->source[1])) - potential; // the dimensionless version; it will be converted to days by the QLens class
	} else {
		images[nfound].td = 0;
	}
	images[nfound].parity = sign(images[nfound].mag);

	nfound++;
	if (nfound >= max_images) finished_search = true;
}

void Grid::subgrid_around_galaxies(lensvector* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_splittings, bool* subgrid)
{
	for (int i=0; i < n_cc_splittings; i++)
		subgrid_around_galaxies_iteration(galaxy_centers,ngal,subgrid_radius,min_galsubgrid_cellsize,i,false,subgrid);
	subgrid_around_galaxies_iteration(galaxy_centers,ngal,subgrid_radius,min_galsubgrid_cellsize,n_cc_splittings,false,subgrid);
}

void Grid::subgrid_around_galaxies_iteration(lensvector* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_splittings, bool cc_neighbor_splitting, bool *subgrid)
{
	bool galaxy_nearby;
	if (level < splitlevels+1)
	{
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->subgrid_around_galaxies_iteration(galaxy_centers, ngal, subgrid_radius, min_galsubgrid_cellsize,n_cc_splittings,cc_neighbor_splitting,subgrid);
			}
		}
	}
	else if ((!enforce_min_area) or (cell_area > min_cell_area))
	{
		if ((!enforce_min_area) and (cell_area==0)) find_cell_area(0);
		int i,j,k;
		for (k=0; k < ngal; k++) {
			galaxy_nearby = false;
			if ((subgrid[k]) and ((cell_area > min_galsubgrid_cellsize[k]) or ((n_cc_splittings > 0) and (cc_inside)))) {
				if (test_if_inside_cell(galaxy_centers[k],0)==true) galaxy_nearby = true;
				else galaxy_nearby = test_if_galaxy_nearby(galaxy_centers[k],SQR(subgrid_radius[k]));
				if (galaxy_nearby==true) {
					if (cell != NULL) {
						// We're going to assume that the cells all have nearly the same area, so we can just check the first cell area
						if (cell[0][0]->cell_area > min_cell_area) {
							if (cell[0][0]->cell_area > min_galsubgrid_cellsize[k]) {
								for (i=0; i < u_N; i++)
									for (j=0; j < w_N; j++)
										cell[i][j]->subgrid_around_galaxies_iteration(galaxy_centers,ngal,subgrid_radius,min_galsubgrid_cellsize,n_cc_splittings,cc_neighbor_splitting,subgrid);
							} else if (n_cc_splittings > 0) {
								bool recurse;
								for (i=0; i < u_N; i++) {
									for (j=0; j < w_N; j++) {
										if (cell[i][j]->cell != NULL) {
											cell[i][j]->subgrid_around_galaxies_iteration(galaxy_centers,ngal,subgrid_radius,min_galsubgrid_cellsize,n_cc_splittings,cc_neighbor_splitting,subgrid);
										} else {
											if (recurse==true) recurse = false;
											if (cell[i][j]->galsubgrid_cc_splitlevels < n_cc_splittings) {
												if (cell[i][j]->cc_inside) recurse = true;
												else if (cc_neighbor_splitting==true) {
													for (k=0; k < 4; k++)
														if ((cell[i][j]->neighbor[k] != NULL) and (cell[i][j]->neighbor[k]->cc_inside)) recurse = true;
												}
											}
											if (recurse==true) {
												cell[i][j]->galsubgrid_cc_splitlevels++;
												cell[i][j]->galsubgrid();
											}
										}
									}
								}
							}
						}
					} else {
						galsubgrid();
						if (cell[0][0]->cell_area > dmax(min_cell_area,min_galsubgrid_cellsize[k])) {
							for (i=0; i < u_N; i++) {
								for (j=0; j < w_N; j++) {
									cell[i][j]->subgrid_around_galaxies_iteration(galaxy_centers, ngal, subgrid_radius, min_galsubgrid_cellsize, n_cc_splittings, cc_neighbor_splitting,subgrid);
								}
							}
						}
					}
				}
			}
		}
	}
	return;
}

image* Grid::tree_search()
{
   finished_search = false;
	grid_search_firstlevel(levels);

   return images;
}

inline bool Grid::redundancy(const lensvector& xroot, double &sep)
{
	bool redundancy = false;
	for (int k = 0; k < nfound; k++)
	{
		sep = sqrt(SQR(xroot[0]-images[k].pos[0]) + SQR(xroot[1]-images[k].pos[1]));
		if (sep < lens->redundancy_separation_threshold)
		{
			redundancy = true;

			break;
		}
	}
	return redundancy;
}

void QLens::find_images()
{
	// called by plot_images(...)

	Grid::reset_search_parameters();
	images_found = grid->tree_search();

	if (include_time_delays) {
		double td_factor = time_delay_factor_arcsec(lens_redshift,reference_source_redshift);
		double min_td=1e30;
		int i;
		for (i = 0; i < Grid::nfound; i++)
			if (images_found[i].td < min_td) min_td = images_found[i].td;
		for (i = 0; i < Grid::nfound; i++) {
			images_found[i].td -= min_td;
			if (images_found[i].td != 0.0) images_found[i].td *= td_factor;
		}
	}
}

void QLens::output_images_single_source(const double &x_source, const double &y_source, bool verbal, const double flux, const bool show_labels)
{
	if (grid==NULL) {
		if (create_grid(verbal,reference_zfactors,default_zsrc_beta_factors)==false) {
			return;
		}
	}
	//cout << Grid::image_pos_accuracy << endl; // default
	//cout << Grid::theta_offset << endl; // slight offset in the initial angle for creating the grid; obsolete, but keeping it here just in case
	//cout << Grid::ccroot_t << endl;
	//cout << Grid::enforce_min_area << endl;
	//cout << Grid::cc_neighbor_splittings << endl;
	//double *Grid::grid_zfactors;
	//double **Grid::grid_betafactors;
	//cout << "zfacs: " << Grid::grid_zfactors[0] << " " << reference_zfactors[0] << " " << specific_ptsrc_zfactors[0][0] << endl;

	// parameters for creating the recursive grid
	//cout << Grid::rmin << endl;
	//cout << Grid::rmax << endl;
	//cout << Grid::xcenter << endl;
	//cout << Grid::ycenter << endl;
	//cout << Grid::grid_q << endl;
	//cout << Grid::u_split_initial << endl; cout << Grid::w_split_initial << endl;
	//cout << Grid::levels << endl;
	//cout << Grid::splitlevels << endl;
	//cout << Grid::cc_splitlevels << endl;
	//cout << Grid::min_cell_area << endl;


	source[0] = x_source; source[1] = y_source;

	find_images();

	if (mpi_id==0) {
		if (use_scientific_notation==false) {
			cout << setprecision(6);
			cout << fixed;
		}
		cout << "#src_x (arcsec)\tsrc_y (arcsec)\tn_images";
		if (flux != -1) cout << "\tsrc_flux";
		cout << endl;
		cout << source[0] << "\t" << source[1] << "\t" << Grid::nfound << "\t";
		if (flux != -1) cout << "\t" << flux;
		cout << endl << endl;

		//cout << "# " << Grid::nfound << " images" << endl;
		if (show_labels) {
			cout << "#pos_x (arcsec)\tpos_y (arcsec)\tmagnification";
			if (flux != -1.0) cout << "\tflux\t";
			if (include_time_delays) cout << "\ttime_delay (days)";
			cout << endl;
		}
		if (include_time_delays) {
			for (int i = 0; i < Grid::nfound; i++) {
				if (flux == -1.0) cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].td << endl;
				else cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].mag*flux << "\t" << images_found[i].td << endl;
			}
		} else {
			for (int i = 0; i < Grid::nfound; i++) {
				if (flux == -1.0) cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << endl;
				else cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].mag*flux << endl;
			}
		}

		if (use_scientific_notation==false)
			cout.unsetf(ios_base::floatfield);

		cout << endl;
	}
}

bool QLens::plot_images_single_source(const double &x_source, const double &y_source, bool verbal, ofstream& imgfile, ofstream& srcfile, const double flux, const bool show_labels)
{
	// flux is an optional argument; if not specified, its default is -1, meaning fluxes will not be calculated or displayed
	if ((grid==NULL) and (create_grid(verbal,reference_zfactors,default_zsrc_beta_factors)==false)) return false;

	if (use_scientific_notation==false) {
		cout << setprecision(6);
		cout << fixed;
	}
	source[0] = x_source; source[1] = y_source;

	find_images();

	if (mpi_id==0) {
		cout << "#src_x (arcsec)\tsrc_y (arcsec)\tn_images";
		if (flux != -1) cout << "\tsrc_flux";
		cout << endl;
		cout << source[0] << "\t" << source[1] << "\t" << Grid::nfound << "\t";
		if (flux != -1) cout << "\t" << flux;
		cout << endl << endl;

		srcfile << x_source << " " << y_source << endl;
		//cout << "# " << Grid::nfound << " images" << endl;
		if (show_labels) {
			cout << "#pos_x (arcsec)\tpos_y (arcsec)\tmagnification";
			if (flux != -1.0) cout << "\tflux\t";
			if (include_time_delays) cout << "\ttime_delay (days)";
			cout << endl;
		}
		if (include_time_delays) {
			for (int i = 0; i < Grid::nfound; i++) {
				if (flux == -1.0) cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].td << endl;
				else cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].mag*flux << "\t" << images_found[i].td << endl;
				imgfile << images_found[i].pos[0] << " " << images_found[i].pos[1] << endl;
			}
		} else {
			for (int i = 0; i < Grid::nfound; i++) {
				if (flux == -1.0) cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << endl;
				else cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].mag*flux << endl;
				imgfile << images_found[i].pos[0] << " " << images_found[i].pos[1] << endl;
			}
		}

		cout << endl;
	}
	if (use_scientific_notation==false)
		cout.unsetf(ios_base::floatfield);

	return true;
}

image* QLens::get_images(const lensvector &source_in, int &n_images, bool verbal)
{
	if (grid==NULL) {
		if (create_grid(verbal,reference_zfactors,default_zsrc_beta_factors)==false) return NULL;
	}
	source[0] = source_in[0];
	source[1] = source_in[1];

	find_images();
	n_images = Grid::nfound;
	return images_found;
}

// this is for the Python wrapper, but I would like to replace the above functions with this in qlens anyway (DO LATER)
bool QLens::get_imageset(const double src_x, const double src_y, PointSource& image_set, bool verbal)
{
	if (grid==NULL) {
		if (create_grid(verbal,reference_zfactors,default_zsrc_beta_factors)==false) return false;
	}
	source[0] = src_x;
	source[1] = src_y;

	find_images();
	image_set.copy_imageset(source,source_redshift,images_found,Grid::nfound);
	return true;
}

vector<PointSource> QLens::get_fit_imagesets(bool &status, int min_dataset, int max_dataset, bool verbal)
{
	status = true;
	if (n_ptsrc==0) status = false;
	if (max_dataset < 0) max_dataset = n_ptsrc - 1;
	if ((min_dataset < 0) or (min_dataset > max_dataset)) status = false;
	vector<PointSource> image_sets;
	image_sets.clear();
	if (!status) return image_sets;
	image_sets.resize(n_ptsrc);

	if (analytic_source_flux) set_analytic_srcflux(false);
	if (use_analytic_bestfit_src) set_analytic_sourcepts(false);

	int redshift_idx;
	for (int i=min_dataset; i <= max_dataset; i++) {
		redshift_idx = ptsrc_redshift_idx[i];
		if ((i == min_dataset) or (redshift_idx != ptsrc_redshift_idx[i-1])) {
			create_grid(false,ptsrc_zfactors[redshift_idx],ptsrc_beta_factors[redshift_idx]);
		}

		source[0] = ptsrc_list[i]->pos[0];
		source[1] = ptsrc_list[i]->pos[1];

		find_images();
		image_sets[i].copy_imageset(source,ptsrc_redshifts[redshift_idx],images_found,Grid::nfound,ptsrc_list[i]->srcflux);
	}
	reset_grid();
	return image_sets;
}

bool QLens::plot_images(const char *sourcefile, const char *imagefile, bool color_multiplicities, bool verbal)
{
	if (plot_critical_curves("crit.dat")==false) warn(warnings,"no critical curves found");
	if ((grid==NULL) and (create_grid(verbal,reference_zfactors,default_zsrc_beta_factors)==false)) return false;

	//ifstream sources(sourcefile);
	ifstream sources;
	open_input_file(sources,sourcefile);
	ofstream imagedat;
	ofstream srcdat; // This is somewhat redundant, but the graphical plotter prefers to have a standard filename for the source

	ofstream quads;
	ofstream doubles;
	ofstream cusps;
	ofstream singles;
	ofstream weird;

	ofstream srcquads;
	ofstream srcdoubles;
	ofstream srccusps;
	ofstream srcsingles;
	ofstream srcweird;
	if (mpi_id==0) {
		open_output_file(imagedat,imagefile);
		open_output_file(srcdat,"src.dat"); // This is somewhat redundant, but the graphical plotter prefers to have a standard filename for the source

		if (color_multiplicities) {
			open_output_file(quads,"images.quads");
			open_output_file(doubles,"images.doubles");
			open_output_file(cusps,"images.cusps");
			open_output_file(singles,"images.singles");
			open_output_file(weird,"images.weird");

			open_output_file(srcquads,"sources.quads");
			open_output_file(srcdoubles,"sources.doubles");
			open_output_file(srccusps,"sources.cusps");
			open_output_file(srcsingles,"sources.singles");
			open_output_file(srcweird,"sources.weird");
			
			quads << setiosflags(ios::scientific);
			cusps << setiosflags(ios::scientific);
			doubles << setiosflags(ios::scientific);
			singles << setiosflags(ios::scientific);
			weird << setiosflags(ios::scientific);

			srcquads << setiosflags(ios::scientific);
			srccusps << setiosflags(ios::scientific);
			srcdoubles << setiosflags(ios::scientific);
			srcsingles << setiosflags(ios::scientific);
			srcweird << setiosflags(ios::scientific);

			imagedat << setiosflags(ios::scientific);
			srcdat << setiosflags(ios::scientific);
		}
	}

	int nsources = 0;
	while (sources >> source[0] >> source[1])
	{
		nsources++;
		if (mpi_id==0) srcdat << source[0] << " " << source[1] << endl;
		find_images();

		if (mpi_id==0) {
			imagedat << "# " << Grid::nfound << " images" << endl;

			for (int i = 0; i < Grid::nfound; i++)
			{
				if (include_time_delays)
					imagedat << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].td << " " << images_found[i].parity << endl;
				else
					imagedat << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
				if (color_multiplicities) {
					if (Grid::nfound==5) {
						quads << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
					}
					else if (Grid::nfound==3) {
						// this will count doubles and cusps
						doubles << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
					}
					else if (Grid::nfound==1) {
						singles << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
					} else {
						weird << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
					}
				}
			}
			if (color_multiplicities) {
				if (Grid::nfound==5) {
					srcquads << source[0] << " " << source[1] << endl;
				}
				else if (Grid::nfound==3) {
					srcdoubles << source[0] << " " << source[1] << endl;
				}
				else if (Grid::nfound==1) {
					srcsingles << source[0] << " " << source[1] << endl;
				} else {
					srcweird << source[0] << " " << source[1] << endl;
				}
			}

			imagedat << endl;
		}
	}

	return true;
}

// 2-d Newton's Method w/ backtracking routines

const int Grid::max_iterations = 200;
const int Grid::max_step_length = 100;

inline void Grid::SolveLinearEqs(lensmatrix& a, lensvector& b)
{
	double det, temp;
	det = determinant(a);
	temp = (-a[1][0]*b[1]+a[1][1]*b[0]) / det;
	b[1] = (-a[0][1]*b[0]+a[0][0]*b[1]) / det;
	b[0] = temp;
}

inline double Grid::max_component(const lensvector& x) { return dmax(fabs(x[0]),fabs(x[1])); }

bool Grid::run_newton(lensvector& xroot, const int& thread)
{
	if ((enforce_min_area) and (image_pos_accuracy > 0.2*sqrt(cell_area))) warn(lens->newton_warnings,"image position accuracy comparable to or larger than cell size");
	if ((xroot[0]==0) and (xroot[1]==0)) { xroot[0] = xroot[1] = 5e-1*lens->cc_rmin; }	// Avoiding singularity at center
	if (NewtonsMethod(xroot, newton_check[thread], thread)==false) {
		warn(lens->newton_warnings,"Newton's method failed for source (%g,%g), level %i, cell center (%g,%g)",lens->source[0],lens->source[1],level,center_imgplane[0],center_imgplane[1],xroot[0],xroot[1]);
		return false;
	}
	if (lens->reject_images_found_outside_cell) {
		if (test_if_inside_cell(xroot,thread)==false) {
			warn(lens->warnings,"Rejecting image found outside cell for source (%g,%g), level %i, cell center (%g,%g)",lens->source[0],lens->source[1],level,center_imgplane[0],center_imgplane[1],xroot[0],xroot[1]);
			return false;
		}
	}

	lensvector lens_eq_f;
	lens->lens_equation(xroot,lens_eq_f,thread,grid_zfactors,grid_betafactors);
	//double lenseq_mag = sqrt(SQR(lens_eq_f[0]) + SQR(lens_eq_f[1]));
	//double tryacc = image_pos_accuracy / sqrt(abs(lens->magnification(xroot,thread,zfactor)));
	//cout << lenseq_mag << " " << tryacc << " " << sqrt(abs(lens->magnification(xroot,thread,zfactor))) << endl;
	if (newton_check[thread]==true) { warn(lens->newton_warnings, "false image--converged to local minimum"); return false; }
	if (lens->n_singular_points > 0) {
		double singular_pt_accuracy = 2*image_pos_accuracy;
		for (int i=0; i < lens->n_singular_points; i++) {
			if ((abs(xroot[0]-lens->singular_pts[i][0]) < singular_pt_accuracy) and (abs(xroot[1]-lens->singular_pts[i][1]) < singular_pt_accuracy)) {
				warn(lens->newton_warnings,"Newton's method converged to singular point (%g,%g) for source (%g,%g)",lens->singular_pts[i][0],lens->singular_pts[i][1],lens->source[0],lens->source[1]);
				return false;
			}
		}
	}
	if (((xroot[0]==center_imgplane[0]) and (center_imgplane[0] != 0)) and ((xroot[1]==center_imgplane[1]) and (center_imgplane[1] != 0)))
		warn(lens->newton_warnings, "Newton's method returned center of grid cell");
	double mag = lens->magnification(xroot,thread,grid_zfactors,grid_betafactors);
	if ((abs(lens_eq_f[0]) > 1000*image_pos_accuracy) and (abs(lens_eq_f[1]) > 1000*image_pos_accuracy) and (abs(mag) < 1e-3)) {
		if (lens->newton_warnings==true) {
			warn(lens->newton_warnings,"Newton's method may have found false root (%g,%g) (within 1000*accuracy) for source (%g,%g), level %i, cell center (%g,%g), mag %g",xroot[0],xroot[1],lens->source[0],lens->source[1],level,center_imgplane[0],center_imgplane[1],xroot[0],xroot[1],mag);
		}
	}
	if (abs(mag) > lens->newton_magnification_threshold) {
		if (lens->reject_himag_images) {
			if ((lens->mpi_id==0) and (lens->warnings)) {
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
	if ((lens->include_central_image==false) and (mag > 0) and (lens->kappa(xroot,grid_zfactors,grid_betafactors) > 1)) return false; // discard central image if not desired
	bool status = true;
	//#pragma omp critical
	//{
		double sep;
		if (redundancy(xroot,sep)) {
			// generally, this only occurs very close to critical curves and is best solved by further cell splittings
			// around said curves. However for extreme magnifications (near Einstein-ring images), even cell splittings
			// does not solve the issue.
			if (lens->newton_warnings==true) {
				warn(lens->newton_warnings,"rejecting probable duplicate image (imgsep=%g): src (%g,%g), level %i, image (%g,%g), mag %g",sep,lens->source[0],lens->source[1],level,xroot[0],xroot[1],mag);
			}
			status = false;
		}
		else if (nfound >= max_images) status = false;
	//}
	return status;
}

bool Grid::NewtonsMethod(lensvector& x, bool &check, const int& thread)
{
	check = false;
	lensvector g, p, xold;
	lensmatrix fjac;

	lens->lens_equation(x, fvec[thread], thread, grid_zfactors, grid_betafactors);
	double f = 0.5*fvec[thread].sqrnorm();
	if (max_component(fvec[thread]) < 0.01*image_pos_accuracy)
		return true; 

	double fold, stpmax, temp, test;
	stpmax = max_step_length * dmax(x.norm(), 2.0); 
	for (int its=0; its < max_iterations; its++) {
		lens->hessian(x[0],x[1],fjac,thread,grid_zfactors,grid_betafactors);
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

bool Grid::LineSearch(lensvector& xold, double fold, lensvector& g, lensvector& p, lensvector& x,
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
		lens->lens_equation(x, fvec[thread], thread, grid_zfactors, grid_betafactors);
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

void Grid::reset_search_parameters()
{
	nfound = 0;
	nfound_max = 0; nfound_pos = 0; nfound_neg = 0;
}

void Grid::clear_subcells(int clear_level)
{
	if (cell != NULL) {
		int i,j;
		if (level <= clear_level) {
			for (i=0; i < u_N; i++) {
				for (j=0; j < w_N; j++) {
					cell[i][j]->clear_subcells(clear_level);
				}
			}
		}
		else
		{
			for (i=0; i < u_N; i++) {
				for (j=0; j < w_N; j++) {
					delete cell[i][j];
				}
				delete[] cell[i];
			}
			delete[] cell;
			cell = NULL;
		}
	}
}

Grid::~Grid()
{
	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				delete cell[i][j];
			}
			delete[] cell[i];
		}
		delete[] cell;
		cell = NULL;
	}
	if (level > 0) {
		delete corner_invmag[0];
		delete corner_sourcept[0];
		delete corner_kappa[0];
		for (int k=1; k < 4; k++) {
			if (allocated_corner[k]) {
				delete corner_invmag[k];
				delete corner_sourcept[k];
				delete corner_kappa[k];
			}
		}
	}
}

