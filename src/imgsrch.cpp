#include "profile.h"
#include "qlens.h"
#include "imgsrch.h"
#include "mathexpr.h"
#include "errors.h"
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
using namespace std;

// parameters for creating the recursive grid
const int GridCell::u_split = 2;
const int GridCell::w_split = 2;

int GridCell::u_split_initial;
int GridCell::w_split_initial;
int GridCell::splitlevels;
int GridCell::cc_splitlevels;
double GridCell::min_cell_area;
bool GridCell::cc_neighbor_splittings;
bool GridCell::enforce_min_area;

bool *GridCell::newton_check = NULL;
int *GridCell::maxlevs = NULL;
int GridCell::nthreads=0;

// multithreaded variables
template <>
lensvector<double> *CellStaticParams<double>::d1 = NULL;
template <>
lensvector<double> *CellStaticParams<double>::d2 = NULL;
template <>
lensvector<double> *CellStaticParams<double>::d3 = NULL;
template <>
lensvector<double> *CellStaticParams<double>::d4 = NULL;
template <>
double *CellStaticParams<double>::product1 = NULL;
template <>
double *CellStaticParams<double>::product2 = NULL;
template <>
double *CellStaticParams<double>::product3 = NULL;
template <>
lensvector<double> *CellStaticParams<double>::fvec = NULL;
template <>
lensvector<double> ***CellStaticParams<double>::xvals_threads = NULL;

#ifdef USE_STAN
template <>
lensvector<stan::math::var> *CellStaticParams<stan::math::var>::d1 = NULL;
template <>
lensvector<stan::math::var> *CellStaticParams<stan::math::var>::d2 = NULL;
template <>
lensvector<stan::math::var> *CellStaticParams<stan::math::var>::d3 = NULL;
template <>
lensvector<stan::math::var> *CellStaticParams<stan::math::var>::d4 = NULL;
template <>
stan::math::var *CellStaticParams<stan::math::var>::product1 = NULL;
template <>
stan::math::var *CellStaticParams<stan::math::var>::product2 = NULL;
template <>
stan::math::var *CellStaticParams<stan::math::var>::product3 = NULL;
template <>
lensvector<stan::math::var> *CellStaticParams<stan::math::var>::fvec = NULL;
template <>
lensvector<stan::math::var> ***CellStaticParams<stan::math::var>::xvals_threads = NULL;
#endif

void GridCell::allocate_multithreaded_variables(const int& threads, const bool reallocate)
{
	if (maxlevs != NULL) {
		if (!reallocate) return;
		else deallocate_multithreaded_variables();
	}
	int i,j;
	nthreads = threads;
	newton_check = new bool[threads];
	maxlevs = new int[threads];
	CellStaticParams<double>::d1 = new lensvector<double>[threads];
	CellStaticParams<double>::d2 = new lensvector<double>[threads];
	CellStaticParams<double>::d3 = new lensvector<double>[threads];
	CellStaticParams<double>::d4 = new lensvector<double>[threads];
	CellStaticParams<double>::product1 = new double[threads];
	CellStaticParams<double>::product2 = new double[threads];
	CellStaticParams<double>::product3 = new double[threads];
	CellStaticParams<double>::fvec = new lensvector<double>[threads];
	CellStaticParams<double>::xvals_threads = new lensvector<double>**[threads];
	for (j=0; j < threads; j++) {
		CellStaticParams<double>::xvals_threads[j] = new lensvector<double>*[u_split+1];
		for (i=0; i <= u_split; i++) CellStaticParams<double>::xvals_threads[j][i] = new lensvector<double>[w_split+1];
	}
#ifdef USE_STAN
	CellStaticParams<stan::math::var>::d1 = new lensvector<stan::math::var>[threads];
	CellStaticParams<stan::math::var>::d2 = new lensvector<stan::math::var>[threads];
	CellStaticParams<stan::math::var>::d3 = new lensvector<stan::math::var>[threads];
	CellStaticParams<stan::math::var>::d4 = new lensvector<stan::math::var>[threads];
	CellStaticParams<stan::math::var>::product1 = new stan::math::var[threads];
	CellStaticParams<stan::math::var>::product2 = new stan::math::var[threads];
	CellStaticParams<stan::math::var>::product3 = new stan::math::var[threads];
	CellStaticParams<stan::math::var>::fvec = new lensvector<stan::math::var>[threads];
	CellStaticParams<stan::math::var>::xvals_threads = new lensvector<stan::math::var>**[threads];
	for (j=0; j < threads; j++) {
		CellStaticParams<stan::math::var>::xvals_threads[j] = new lensvector<stan::math::var>*[u_split+1];
		for (i=0; i <= u_split; i++) CellStaticParams<stan::math::var>::xvals_threads[j][i] = new lensvector<stan::math::var>[w_split+1];
	}
#endif
}

void GridCell::deallocate_multithreaded_variables()
{
	if (maxlevs != NULL) {
		delete[] maxlevs;
		delete[] newton_check;
		maxlevs = NULL;
		newton_check = NULL;

		delete[] CellStaticParams<double>::d1;
		delete[] CellStaticParams<double>::d2;
		delete[] CellStaticParams<double>::d3;
		delete[] CellStaticParams<double>::d4;
		delete[] CellStaticParams<double>::product1;
		delete[] CellStaticParams<double>::product2;
		delete[] CellStaticParams<double>::product3;
		delete[] CellStaticParams<double>::fvec;
		int i,j;
		for (j=0; j < nthreads; j++) {
			for (i=0; i <= u_split; i++) delete[] CellStaticParams<double>::xvals_threads[j][i];
			delete[] CellStaticParams<double>::xvals_threads[j];
		}
		delete[] CellStaticParams<double>::xvals_threads;
		CellStaticParams<double>::d1 = NULL;
		CellStaticParams<double>::d2 = NULL;
		CellStaticParams<double>::d3 = NULL;
		CellStaticParams<double>::d4 = NULL;
		CellStaticParams<double>::product1 = NULL;
		CellStaticParams<double>::product2 = NULL;
		CellStaticParams<double>::product3 = NULL;
		CellStaticParams<double>::fvec = NULL;
		CellStaticParams<double>::xvals_threads = NULL;
#ifdef USE_STAN
		delete[] CellStaticParams<stan::math::var>::d1;
		delete[] CellStaticParams<stan::math::var>::d2;
		delete[] CellStaticParams<stan::math::var>::d3;
		delete[] CellStaticParams<stan::math::var>::d4;
		delete[] CellStaticParams<stan::math::var>::product1;
		delete[] CellStaticParams<stan::math::var>::product2;
		delete[] CellStaticParams<stan::math::var>::product3;
		delete[] CellStaticParams<stan::math::var>::fvec;
		for (j=0; j < nthreads; j++) {
			for (i=0; i <= u_split; i++) delete[] CellStaticParams<stan::math::var>::xvals_threads[j][i];
			delete[] CellStaticParams<stan::math::var>::xvals_threads[j];
		}
		delete[] CellStaticParams<stan::math::var>::xvals_threads;
		CellStaticParams<stan::math::var>::d1 = NULL;
		CellStaticParams<stan::math::var>::d2 = NULL;
		CellStaticParams<stan::math::var>::d3 = NULL;
		CellStaticParams<stan::math::var>::d4 = NULL;
		CellStaticParams<stan::math::var>::product1 = NULL;
		CellStaticParams<stan::math::var>::product2 = NULL;
		CellStaticParams<stan::math::var>::product3 = NULL;
		CellStaticParams<stan::math::var>::fvec = NULL;
		CellStaticParams<stan::math::var>::xvals_threads = NULL;
#endif
	}
}

void GridCell::set_splitting(int rs0, int ts0, int sl, int ccsl, double min_cs, bool neighbor_split)
{
	u_split_initial = rs0;
	w_split_initial = ts0;
	splitlevels = sl;
	cc_splitlevels = ccsl;
	min_cell_area = min_cs;
	cc_neighbor_splittings = neighbor_split;
}

void ImgSrchGrid::set_default_imgsrch_settings()
{
	image_pos_accuracy = 1e-6;
	max_images = 50;
	max_level = 10;
	theta_offset = 0; // slight offset in the initial angle for creating the grid; obsolete, but keeping it here just in case
	gridparams.images = new image<double>[max_images];
#ifdef USE_STAN
	gridparams_dif.images = new image<stan::math::var>[max_images];
#endif
}

ImgSrchGrid::ImgSrchGrid(QLens* lens_in, double xcenter_in, double ycenter_in, double xlength, double ylength, double acc, double *zfactor_in, double **betafactor_in) // use for top-level cell only; subcells use constructor below
{
	int threads = 1;
#ifdef USE_OPENMP
	threads = omp_get_num_threads();
#endif
	GridCell::allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)
	lens = lens_in;

	set_default_imgsrch_settings();
	image_pos_accuracy = acc;

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
	parent_grid = this;
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

	lensvector<double>** xvals = new lensvector<double>*[u_N+1];
	int i,j;
	for (i=0, x=x_min; i <= u_N; i++, x += xstep) {
		xvals[i] = new lensvector<double>[w_N+1];
		for (j=0, y=y_min; j <= w_N; j++, y += ystep) {
			xvals[i][j][0] = x;
			xvals[i][j][1] = y;
		}
	}

	cell = new GridCell**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new GridCell*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new GridCell(lens,xvals,i,j,1,this);
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

ImgSrchGrid::ImgSrchGrid(QLens* lens_in, double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double acc, double *zfactor_in, double **betafactor_in) // use for top-level cell only; subcells use constructor below
{
	int threads = 1;
#ifdef USE_OPENMP
	threads = omp_get_num_threads();
#endif
	GridCell::allocate_multithreaded_variables(threads,false); // allocate multithreading arrays ONLY if it hasn't been allocated already (avoids seg faults)
	lens = lens_in;

	set_default_imgsrch_settings();
	image_pos_accuracy = acc;

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
	parent_grid = this;
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

	lensvector<double>** xvals = new lensvector<double>*[u_N+1];
	r = rmin;
	for (i=0; i <= u_N; i++, r += rstep) {
		xvals[i] = new lensvector<double>[w_N+1];
		theta = theta_offset;
		for (j=0; j <= w_N; j++, theta += thetastep) {
			xvals[i][j][0] = xcenter + r*cos(theta);
			xvals[i][j][1] = ycenter + grid_q*r*sin(theta);
		}
	}

	cell = new GridCell**[u_N];
	for (i=0; i < u_N; i++)
	{
		cell[i] = new GridCell*[w_N];
		for (j=0; j < w_N; j++)
		{
			cell[i][j] = new GridCell(lens,xvals,i,j,1,this);
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

GridCell::GridCell(QLens* lens_in, lensvector<double>** xij, const int& i, const int& j, const int& level_in, ImgSrchGrid* parent_ptr)
{
	lens = lens_in;
	u_N = 1;
	w_N = 1;
	level = level_in;
	cell = NULL;
	cc_inside = false;
	singular_pt_inside = false;
	cell_in_central_image_region = false;
	galsubgrid_cc_splitlevels = 0;
	parent_grid = parent_ptr;

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

	corner_sourcept[0] = new lensvector<double>;
	corner_sourcept[1] = corner_sourcept[2] = corner_sourcept[3] = NULL;

	corner_kappa[0] = new double;
	corner_kappa[1] = corner_kappa[2] = corner_kappa[3] = NULL;

	allocated_corner[0] = true;
	allocated_corner[1] = allocated_corner[2] = allocated_corner[3] = false;
}

void ImgSrchGrid::redraw_grid(double r_min, double r_max, double xcenter_in, double ycenter_in, double grid_q_in, double *zfactor_in, double **betafactor_in)  // for radial grid
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

	lensvector<double>** xvals = new lensvector<double>*[u_N+1];
	r = rmin;
	int i, j;
	for (i=0; i <= u_N; i++, r += rstep) {
		xvals[i] = new lensvector<double>[w_N+1];
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
				cell[i][j]->reassign_coordinates(xvals,i,j,1);
				cell[i][j]->assign_lensing_properties(thread);
			}
		}
	}

	for (i=0; i < u_N+1; i++)
		delete[] xvals[i];
	delete[] xvals;

	reassign_subcell_lensing_properties_firstlevel();

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

void ImgSrchGrid::redraw_grid(double xcenter_in, double ycenter_in, double xlength, double ylength, double *zfactor_in, double **betafactor_in)  // for Cartesian grid
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

	lensvector<double>** xvals = new lensvector<double>*[u_N+1];
	int i,j;
	for (i=0, x=x_min; i <= u_N; i++, x += xstep) {
		xvals[i] = new lensvector<double>[w_N+1];
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
				cell[i][j]->reassign_coordinates(xvals,i,j,1);
				cell[i][j]->assign_lensing_properties(thread);
			}
		}
	}

	for (i=0; i < u_N+1; i++)
		delete[] xvals[i];
	delete[] xvals;

	reassign_subcell_lensing_properties_firstlevel();

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

void GridCell::reassign_coordinates(lensvector<double>** xij, const int& i, const int& j, const int& level_in)
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

void GridCell::assign_lensing_properties(const int& thread)
{
	if (enforce_min_area) find_cell_area(thread);
	else cell_area=0;

	lens->kappa_inverse_mag_sourcept<double>(corner_pt[0],(*(corner_sourcept[0])),(*(corner_kappa[0])),(*(corner_invmag[0])),thread,parent_grid->grid_zfactors,parent_grid->grid_betafactors);
}

inline void GridCell::set_grid_xvals(lensvector<double>** xv, const int& i, const int& j)
{
	xv[i][j][0] = ((corner_pt[0][0]*(w_N-j) + corner_pt[1][0]*j)*(u_N-i) + (corner_pt[2][0]*(w_N-j) + corner_pt[3][0]*j)*i)/(u_N*w_N);
	xv[i][j][1] = ((corner_pt[0][1]*(w_N-j) + corner_pt[1][1]*j)*(u_N-i) + (corner_pt[2][1]*(w_N-j) + corner_pt[3][1]*j)*i)/(u_N*w_N);
}

inline void GridCell::find_cell_area(const int& thread)
{
	CellStaticParams<double>::d1[thread][0] = corner_pt[2][0] - corner_pt[0][0]; CellStaticParams<double>::d1[thread][1] = corner_pt[2][1] - corner_pt[0][1];
	CellStaticParams<double>::d2[thread][0] = corner_pt[1][0] - corner_pt[0][0]; CellStaticParams<double>::d2[thread][1] = corner_pt[1][1] - corner_pt[0][1];
	CellStaticParams<double>::d3[thread][0] = corner_pt[2][0] - corner_pt[3][0]; CellStaticParams<double>::d3[thread][1] = corner_pt[2][1] - corner_pt[3][1];
	CellStaticParams<double>::d4[thread][0] = corner_pt[1][0] - corner_pt[3][0]; CellStaticParams<double>::d4[thread][1] = corner_pt[1][1] - corner_pt[3][1];
	// split cell into two triangles; cross product of the vectors forming the legs gives area of each triangle, so their sum gives area of cell
	cell_area = 0.5 * (abs(CellStaticParams<double>::d1[thread] ^ CellStaticParams<double>::d2[thread]) + abs(CellStaticParams<double>::d3[thread] ^ CellStaticParams<double>::d4[thread]));
}

void ImgSrchGrid::assign_firstlevel_neighbors()
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

void GridCell::assign_neighborhood()
{
	// assign neighbors of this cell, then update neighbors of neighbors of this cell
	// neighbor index: 0 = i+1 neighbor, 1 = i-1 neighbor, 2 = j+1 neighbor, 3 = j-1 neighbor
	int k,l;
	assign_level_neighbors(level);
	for (l=0; l < 4; l++)
		if ((neighbor[l] != NULL) and (neighbor[l]->cell != NULL)) {
		for (k=level; k <= parent_grid->levels; k++) {
			neighbor[l]->assign_level_neighbors(k);
		}
	}
}

void GridCell::assign_all_neighbors()
{
	if (level!=0) die("assign_all_neighbors should only be run from level 0");

	int i,j,k;
	for (k=1; k < parent_grid->levels; k++) {
		for (i=0; i < u_N; i++) {
			for (j=0; j < w_N; j++) {
				cell[i][j]->assign_level_neighbors(k); // we've just created our grid, so we only need to go to level+1
			}
		}
	}
}

void GridCell::assign_level_neighbors(int neighbor_level)
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

inline void GridCell::check_if_singular_point_inside(const int& thread)
{
	// if a singular point is inside this cell, treat it as though it were a critical curve so we split around it.
	for (int i=0; i < lens->n_singular_points; i++) {
		if (test_if_inside_cell(lens->singular_pts[i],thread)) { singular_pt_inside = true; break; }
	}
}

inline void GridCell::check_if_cc_inside()
{
	if (((*(corner_invmag[0])) * (*(corner_invmag[1])) * (*(corner_invmag[2])) * (*(corner_invmag[3]))) < 0) cc_inside = true;
	else if ((((*(corner_invmag[0])) * (*(corner_invmag[1]))) < 0) or ((*(corner_invmag[0])) * (*(corner_invmag[2]))) < 0) cc_inside = true;
	else cc_inside = false;
}

inline void GridCell::check_if_central_image_region()
{
	// establish whether this cell only contains central images. If it does, then we can exclude them
	// from searches if no central image is observed in the data
	cell_in_central_image_region = true;
	for (int k=0; k < 4; k++)
		if ((*(corner_invmag[k]) < 0) or (*(corner_kappa[k]) < 1)) { cell_in_central_image_region = false; break; }
}

bool GridCell::split_cells(const int& thread)
{
	if (level >= parent_grid->max_level)
		die("maximum number of splittings has been reached (%i)", parent_grid->max_level);

	bool subgridded = false;
	if (cell==NULL) {
		subgridded = true;
		u_N = u_split;
		w_N = w_split;

		int i,j;
		for (i=0; i <= u_N; i++) {
			for (j=0; j <= w_N; j++) {
				CellStaticParams<double>::xvals_threads[thread][i][j][0] = ((corner_pt[0][0]*(w_N-j) + corner_pt[1][0]*j)*(u_N-i) + (corner_pt[2][0]*(w_N-j) + corner_pt[3][0]*j)*i)/(u_N*w_N);
				CellStaticParams<double>::xvals_threads[thread][i][j][1] = ((corner_pt[0][1]*(w_N-j) + corner_pt[1][1]*j)*(u_N-i) + (corner_pt[2][1]*(w_N-j) + corner_pt[3][1]*j)*i)/(u_N*w_N);
			}
		}

		cell = new GridCell**[u_N];
		for (i=0; i < u_N; i++)
		{
			cell[i] = new GridCell*[w_N];
			for (j=0; j < w_N; j++)
			{
				cell[i][j] = new GridCell(lens,CellStaticParams<double>::xvals_threads[thread],i,j,level+1,parent_grid);
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

void ImgSrchGrid::split_subcells_firstlevel(int cc_splitlevel, bool cc_neighbor_splitting)
{
	int i,j;
	for (i=0; i < GridCell::nthreads; i++) maxlevs[i] = levels;
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
							if ((!GridCell::enforce_min_area) or (cell[i][j]->cell_area > GridCell::min_cell_area)) {
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
	for (i=0; i < GridCell::nthreads; i++) if (maxlevs[i] > levels) levels = maxlevs[i];
}

void GridCell::split_subcells(int cc_splitlevel, bool cc_neighbor_splitting, const int& thread)
{
	if (level >= parent_grid->max_level)
		die("maximum number of splittings has been reached (%i)", parent_grid->max_level);

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

void GridCell::assign_neighbors_lensing_subcells(int cc_splitlevel, const int& thread)
{
	if (level >= parent_grid->max_level)
		die("maximum number of splittings has been reached (%i)", parent_grid->max_level);

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

void GridCell::galsubgrid()
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

		lensvector<double>** xvals = new lensvector<double>*[u_N+1];
		int i,j;
		for (i=0; i <= u_N; i++) {
			xvals[i] = new lensvector<double>[w_N+1];
			for (j=0; j <= w_N; j++) {
				set_grid_xvals(xvals,i,j);
			}
		}
		cell = new GridCell**[u_N];
		for (i=0; i < u_N; i++) {
			cell[i] = new GridCell*[w_N];
			for (j=0; j < w_N; j++) {
				cell[i][j] = new GridCell(lens,xvals,i,j,level+1,parent_grid);
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
		if (level == parent_grid->levels-1) {
			parent_grid->levels++; // our subcells are at the max level, so splitting them increases the number of levels by 1
		}
		for (i=0; i <= u_N; i++)
			delete[] xvals[i];
		delete[] xvals;
	}

	return;
}

void ImgSrchGrid::assign_subcell_lensing_properties_firstlevel()
{
	int i,j;
	GridCell* gridcell;
	GridCell* neighbor_gridcell;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			gridcell = cell[i][j];
			// only lower left-hand corner of each cell has source pt. and magnification stored in memory;
			// other corners point to the lower-left hand corner of adjacent cells, as defined below
			// unless we're at the rightmost or bottommost cell, or both (see else cases below)
			if (cell[i][j]->neighbor[2] != NULL) {
				neighbor_gridcell = cell[i][j]->neighbor[2];
				gridcell->corner_invmag[1] = neighbor_gridcell->corner_invmag[0];
				gridcell->corner_sourcept[1] = neighbor_gridcell->corner_sourcept[0];
				gridcell->corner_kappa[1] = neighbor_gridcell->corner_kappa[0];
			} else {
				gridcell->corner_invmag[1] = new double;
				gridcell->corner_sourcept[1] = new lensvector<double>;
				gridcell->corner_kappa[1] = new double;
				lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[1],(*gridcell->corner_sourcept[1]),(*gridcell->corner_kappa[1]),(*gridcell->corner_invmag[1]),0,grid_zfactors,grid_betafactors);

				cell[i][j]->allocated_corner[1] = true;
			}

			if (cell[i][j]->neighbor[0] != NULL) {
				neighbor_gridcell = cell[i][j]->neighbor[0];
				gridcell->corner_invmag[2] = neighbor_gridcell->corner_invmag[0];
				gridcell->corner_sourcept[2] = neighbor_gridcell->corner_sourcept[0];
				gridcell->corner_kappa[2] = neighbor_gridcell->corner_kappa[0];
				if (cell[i][j]->neighbor[0]->neighbor[2] != NULL) {
					neighbor_gridcell = cell[i][j]->neighbor[0]->neighbor[2];
					gridcell->corner_invmag[3] = neighbor_gridcell->corner_invmag[0];
					gridcell->corner_sourcept[3] = neighbor_gridcell->corner_sourcept[0];
					gridcell->corner_kappa[3] = neighbor_gridcell->corner_kappa[0];
				} else {
					gridcell->corner_invmag[3] = new double;
					gridcell->corner_sourcept[3] = new lensvector<double>;
					gridcell->corner_kappa[3] = new double;
					lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[3],(*gridcell->corner_sourcept[3]),(*gridcell->corner_kappa[3]),(*gridcell->corner_invmag[3]),0,grid_zfactors,grid_betafactors);

					cell[i][j]->allocated_corner[3] = true;
				}
			} else {
				gridcell->corner_invmag[2] = new double;
				gridcell->corner_sourcept[2] = new lensvector<double>;
				gridcell->corner_kappa[2] = new double;
				lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[2],(*gridcell->corner_sourcept[2]),(*gridcell->corner_kappa[2]),(*gridcell->corner_invmag[2]),0,grid_zfactors,grid_betafactors);
				cell[i][j]->allocated_corner[2] = true;

				gridcell->corner_invmag[3] = new double;
				gridcell->corner_sourcept[3] = new lensvector<double>;
				gridcell->corner_kappa[3] = new double;
				lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[3],(*gridcell->corner_sourcept[3]),(*gridcell->corner_kappa[3]),(*gridcell->corner_invmag[3]),0,grid_zfactors,grid_betafactors);
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

void ImgSrchGrid::reassign_subcell_lensing_properties_firstlevel()
{
	GridCell* gridcell;
	GridCell* neighbor_gridcell;
	int i,j;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			gridcell = cell[i][j];
			// only lower left-hand corner of each cell has source pt. and magnification stored in memory;
			// other corners point to the lower-left hand corner of adjacent cells, as defined below
			// unless we're at the rightmost or bottommost cell, or both (see else cases below)
			if (cell[i][j]->neighbor[2] != NULL) {
				neighbor_gridcell = cell[i][j]->neighbor[2];
				gridcell->corner_invmag[1] = neighbor_gridcell->corner_invmag[0];
				gridcell->corner_sourcept[1] = neighbor_gridcell->corner_sourcept[0];
				gridcell->corner_kappa[1] = neighbor_gridcell->corner_kappa[0];
			} else {
				lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[1],(*gridcell->corner_sourcept[1]),(*gridcell->corner_kappa[1]),(*gridcell->corner_invmag[1]),0,grid_zfactors,grid_betafactors);

				cell[i][j]->allocated_corner[1] = true;
			}

			if (cell[i][j]->neighbor[0] != NULL) {
				neighbor_gridcell = cell[i][j]->neighbor[0];
				gridcell->corner_invmag[2] = neighbor_gridcell->corner_invmag[0];
				gridcell->corner_sourcept[2] = neighbor_gridcell->corner_sourcept[0];
				gridcell->corner_kappa[2] = neighbor_gridcell->corner_kappa[0];
				if (cell[i][j]->neighbor[0]->neighbor[2] != NULL) {
					neighbor_gridcell = cell[i][j]->neighbor[0]->neighbor[2];
					gridcell->corner_invmag[3] = neighbor_gridcell->corner_invmag[0];
					gridcell->corner_sourcept[3] = neighbor_gridcell->corner_sourcept[0];
					gridcell->corner_kappa[3] = neighbor_gridcell->corner_kappa[0];
				} else {
					lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[3],(*gridcell->corner_sourcept[3]),(*gridcell->corner_kappa[3]),(*gridcell->corner_invmag[3]),0,grid_zfactors,grid_betafactors);

					cell[i][j]->allocated_corner[3] = true;
				}
			} else {
				lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[2],(*gridcell->corner_sourcept[2]),(*gridcell->corner_kappa[2]),(*gridcell->corner_invmag[2]),0,grid_zfactors,grid_betafactors);

				cell[i][j]->allocated_corner[2] = true;

				lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[3],(*gridcell->corner_sourcept[3]),(*gridcell->corner_kappa[3]),(*gridcell->corner_invmag[3]),0,grid_zfactors,grid_betafactors);

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

void GridCell::assign_subcell_lensing_properties(const int& thread)
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
	GridCell* gridcell;
	GridCell* neighbor_gridcell;
	for (i=0; i < u_N; i++) {
		for (j=0; j < w_N; j++) {
			gridcell = cell[i][j];
			// only lower left-hand corner of each cell has source pt. and magnification stored in memory;
			// other corners point to the lower-left hand corner of adjacent cells, unless we're at
			// the inner or outer edges of the grid or neighboring cells are larger than our own
			if (gridcell->corner_invmag[1]==NULL) {
				if ((cell[i][j]->neighbor[2] != NULL) and (cell[i][j]->neighbor[2]->level == cell[i][j]->level)) {
					neighbor_gridcell = cell[i][j]->neighbor[2];
					gridcell->corner_invmag[1] = neighbor_gridcell->corner_invmag[0];
					gridcell->corner_sourcept[1] = neighbor_gridcell->corner_sourcept[0];
					gridcell->corner_kappa[1] = neighbor_gridcell->corner_kappa[0];
				} else {
					gridcell->corner_invmag[1] = new double;
					gridcell->corner_sourcept[1] = new lensvector<double>;
					gridcell->corner_kappa[1] = new double;
					lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[1],(*gridcell->corner_sourcept[1]),(*gridcell->corner_kappa[1]),(*gridcell->corner_invmag[1]),thread,parent_grid->grid_zfactors,parent_grid->grid_betafactors);

					cell[i][j]->allocated_corner[1] = true;
				}
			}

			if ((cell[i][j]->neighbor[0] != NULL) and (cell[i][j]->neighbor[0]->level == cell[i][j]->level)) {
				if (gridcell->corner_invmag[2]==NULL) {
					neighbor_gridcell = cell[i][j]->neighbor[0];
					gridcell->corner_invmag[2] = neighbor_gridcell->corner_invmag[0];
					gridcell->corner_sourcept[2] = neighbor_gridcell->corner_sourcept[0];
					gridcell->corner_kappa[2] = neighbor_gridcell->corner_kappa[0];
				}
				if ((cell[i][j]->neighbor[0]->neighbor[2] != NULL) and (cell[i][j]->neighbor[0]->neighbor[2]->level == cell[i][j]->level)) {
					if (gridcell->corner_invmag[3]==NULL) {
						neighbor_gridcell = cell[i][j]->neighbor[0]->neighbor[2];
						gridcell->corner_invmag[3] = neighbor_gridcell->corner_invmag[0];
						gridcell->corner_sourcept[3] = neighbor_gridcell->corner_sourcept[0];
						gridcell->corner_kappa[3] = neighbor_gridcell->corner_kappa[0];
					}
				} else {
					if (gridcell->corner_invmag[3]==NULL) {
						gridcell->corner_invmag[3] = new double;
						gridcell->corner_sourcept[3] = new lensvector<double>;
						gridcell->corner_kappa[3] = new double;
						lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[3],(*gridcell->corner_sourcept[3]),(*gridcell->corner_kappa[3]),(*gridcell->corner_invmag[3]),thread,parent_grid->grid_zfactors,parent_grid->grid_betafactors);

						cell[i][j]->allocated_corner[3] = true;
					}
				}
			} else {
				if (gridcell->corner_invmag[2]==NULL) {
					gridcell->corner_invmag[2] = new double;
					gridcell->corner_sourcept[2] = new lensvector<double>;
					gridcell->corner_kappa[2] = new double;
					lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[2],(*gridcell->corner_sourcept[2]),(*gridcell->corner_kappa[2]),(*gridcell->corner_invmag[2]),thread,parent_grid->grid_zfactors,parent_grid->grid_betafactors);
					cell[i][j]->allocated_corner[2] = true;
				}
					if (gridcell->corner_invmag[3]==NULL) {
					gridcell->corner_invmag[3] = new double;
					gridcell->corner_sourcept[3] = new lensvector<double>;
					gridcell->corner_kappa[3] = new double;
					lens->kappa_inverse_mag_sourcept<double>(gridcell->corner_pt[3],(*gridcell->corner_sourcept[3]),(*gridcell->corner_kappa[3]),(*gridcell->corner_invmag[3]),thread,parent_grid->grid_zfactors,parent_grid->grid_betafactors);

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

/*
ofstream GridCell::xgrid;

bool QLens::plot_recursive_grid(const char filename[])
{
	plot_critical_curves("crit.dat");
	if ((grid==NULL) and (create_grid(true,reference_zfactors,default_zsrc_beta_factors)==false)) return false;

	if (mpi_id==0) {
		string filelabel(filename);
		string filename_string = fit_output_dir + "/" + filelabel;
		GridCell::xgrid.open(filename_string.c_str(), ifstream::out);
		grid->plot_corner_coordinates();
		GridCell::xgrid.close();
	}
	return true;
}

void GridCell::plot_corner_coordinates()
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
*/

bool QLens::plot_recursive_grid(const char filename[])
{
	vector<double> pts_x, pts_y, srcpts_x, srcpts_y;
	if (!output_recursive_grid(pts_x,pts_y,srcpts_x,srcpts_y)) return false;
	string filelabel(filename);
	string filename_string = fit_output_dir + "/" + filelabel;
	if (mpi_id==0) {
		ofstream xgrid(filename_string.c_str());
		int npts = pts_x.size();
		for (int i=0; i < npts; i++) {
			if (pts_x[i] != numeric_limits<double>::quiet_NaN()) xgrid << pts_x[i] << " " << pts_y[i] << " " << srcpts_x[i] << " " << srcpts_y[i] << endl;
			else xgrid << endl;
		}
	}

	return true;
}

bool QLens::output_recursive_grid(vector<double>& pts_x, vector<double>& pts_y, vector<double>& srcpts_x, vector<double>& srcpts_y)
{
	sort_critical_curves();

	// clear the vectors just in case this function has been called before with the same vector objects
	pts_x.clear();
	pts_y.clear();
	srcpts_x.clear();
	srcpts_y.clear();

	if ((grid==NULL) and (create_grid(true,reference_zfactors,default_zsrc_beta_factors)==false)) return false;

	if (mpi_id==0) {
		grid->output_corner_coordinates(pts_x,pts_y,srcpts_x,srcpts_y);
	}
	return true;
}

void GridCell::output_corner_coordinates(vector<double>& pts_x, vector<double>& pts_y, vector<double>& srcpts_x, vector<double>& srcpts_y)
{
	if (level > 0) {
		pts_x.push_back(corner_pt[1][0]);
		pts_y.push_back(corner_pt[1][1]);
		pts_x.push_back(corner_pt[3][0]);
		pts_y.push_back(corner_pt[3][1]);
		pts_x.push_back(corner_pt[2][0]);
		pts_y.push_back(corner_pt[2][1]);
		pts_x.push_back(corner_pt[0][0]);
		pts_y.push_back(corner_pt[0][1]);
		pts_x.push_back(numeric_limits<double>::quiet_NaN());
		pts_y.push_back(numeric_limits<double>::quiet_NaN());

		srcpts_x.push_back((*(corner_sourcept[1]))[0]);
		srcpts_y.push_back((*(corner_sourcept[1]))[1]);
		srcpts_x.push_back((*(corner_sourcept[3]))[0]);
		srcpts_y.push_back((*(corner_sourcept[3]))[1]);
		srcpts_x.push_back((*(corner_sourcept[2]))[0]);
		srcpts_y.push_back((*(corner_sourcept[2]))[1]);
		srcpts_x.push_back((*(corner_sourcept[0]))[0]);
		srcpts_y.push_back((*(corner_sourcept[0]))[1]);
		srcpts_x.push_back(numeric_limits<double>::quiet_NaN());
		srcpts_y.push_back(numeric_limits<double>::quiet_NaN());
	}

	if (cell != NULL) {
		int i,j;
		for (i=0; i < u_N; i++)
			for (j=0; j < w_N; j++)
				cell[i][j]->output_corner_coordinates(pts_x,pts_y,srcpts_x,srcpts_y);
	}
}

void GridCell::store_critical_curve_pts()
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
		if (((*(corner_invmag[0])) * (*(corner_invmag[3]))) < 0) find_and_store_critical_curve_pt(0,3,added_pts);
		if (((*(corner_invmag[1])) * (*(corner_invmag[2]))) < 0) find_and_store_critical_curve_pt(1,2,added_pts);
		// Now we try the edges 0-1, 0-2; no need to check the other edges, because they will be searched from neighboring cells
		if (((*(corner_invmag[0])) * (*(corner_invmag[1]))) < 0) find_and_store_critical_curve_pt(0,1,added_pts);
		if (((*(corner_invmag[0])) * (*(corner_invmag[2]))) < 0) find_and_store_critical_curve_pt(0,2,added_pts);
		if (added_pts==0) warn("cell flagged erroneously as containing critical curve");
		lensvector<double> diagonal1, diagonal2;
		diagonal1[0] = corner_pt[3][0] - corner_pt[0][0];
		diagonal1[1] = corner_pt[3][1] - corner_pt[0][1];
		diagonal2[0] = corner_pt[2][0] - corner_pt[1][0];
		diagonal2[1] = corner_pt[2][1] - corner_pt[1][1];
		parent_grid->cclength1 = diagonal1.norm();
		parent_grid->cclength2 = diagonal2.norm();
		parent_grid->long_diagonal_length = dmax(parent_grid->cclength1,parent_grid->cclength2);
		while (added_pts-- > 0) lens->length_of_cc_cell.push_back(parent_grid->long_diagonal_length);
	}
}

void GridCell::find_and_store_critical_curve_pt(const int icorner, const int fcorner, int& added_pts)
{
	parent_grid->ccsearch_initial_pt[0] = corner_pt[icorner][0];
	parent_grid->ccsearch_initial_pt[1] = corner_pt[icorner][1];
	parent_grid->ccsearch_interval[0] = corner_pt[fcorner][0] - corner_pt[icorner][0];
	parent_grid->ccsearch_interval[1] = corner_pt[fcorner][1] - corner_pt[icorner][1];

	double (Brent::*invmag)(const double);
	invmag = static_cast<double (Brent::*)(const double)> (&GridCell::invmag_along_diagonal);
	if ((invmag_along_diagonal(0)*invmag_along_diagonal(1)) > 0) {
		double inv0=invmag_along_diagonal(0);
		double inv1=invmag_along_diagonal(1);
		warn("critical curve root not bracketed within diagonal: invmag0=%g, invmag1=%g, invmag2=%g, invmag3=%g, (cell corner=%g,%g)",*(corner_invmag[0]),*(corner_invmag[1]),*(corner_invmag[2]),*(corner_invmag[3]),corner_pt[0][0],corner_pt[0][1]);
		return;
	}
	parent_grid->ccroot_t = BrentsMethod(invmag,0,1,1e-6);
	parent_grid->ccroot[0] = parent_grid->ccsearch_initial_pt[0] + parent_grid->ccroot_t*parent_grid->ccsearch_interval[0];
	parent_grid->ccroot[1] = parent_grid->ccsearch_initial_pt[1] + parent_grid->ccroot_t*parent_grid->ccsearch_interval[1];
	lens->critical_curve_pts.push_back(parent_grid->ccroot);
	lensvector<double> new_srcpt;
	lens->find_sourcept<double>(parent_grid->ccroot,new_srcpt,0,parent_grid->grid_zfactors,parent_grid->grid_betafactors);
	lens->caustic_pts.push_back(new_srcpt);
	added_pts++;
}

double GridCell::invmag_along_diagonal(const double t)
{
	return lens->inverse_magnification<double>(parent_grid->ccsearch_initial_pt + t*parent_grid->ccsearch_interval,0,parent_grid->grid_zfactors,parent_grid->grid_betafactors);
}

inline bool GridCell::image_test(const int& thread)
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

	CellStaticParams<double>::d1[thread][0] = parent_grid->gridparams.sourcept[0] - (*(corner_sourcept[1]))[0];
	CellStaticParams<double>::d1[thread][1] = parent_grid->gridparams.sourcept[1] - (*(corner_sourcept[1]))[1];
	CellStaticParams<double>::d2[thread][0] = parent_grid->gridparams.sourcept[0] - (*(corner_sourcept[2]))[0];
	CellStaticParams<double>::d2[thread][1] = parent_grid->gridparams.sourcept[1] - (*(corner_sourcept[2]))[1];
	CellStaticParams<double>::d3[thread][0] = parent_grid->gridparams.sourcept[0] - (*(corner_sourcept[0]))[0];
	CellStaticParams<double>::d3[thread][1] = parent_grid->gridparams.sourcept[1] - (*(corner_sourcept[0]))[1];
	CellStaticParams<double>::product1[thread] = CellStaticParams<double>::d1[thread] ^ CellStaticParams<double>::d2[thread];
	CellStaticParams<double>::product2[thread] = CellStaticParams<double>::d3[thread] ^ CellStaticParams<double>::d1[thread];
	CellStaticParams<double>::product3[thread] = CellStaticParams<double>::d2[thread] ^ CellStaticParams<double>::d3[thread];
	if ((CellStaticParams<double>::product1[thread] > 0) and (CellStaticParams<double>::product2[thread] > 0) and (CellStaticParams<double>::product3[thread] > 0)) return true;
	if ((CellStaticParams<double>::product1[thread] < 0) and (CellStaticParams<double>::product2[thread] < 0) and (CellStaticParams<double>::product3[thread] < 0)) return true;

	if (CellStaticParams<double>::product1[thread] > 0) {
		if ((abs(CellStaticParams<double>::product2[thread])==0) and (CellStaticParams<double>::product3[thread] > 0)) return true;
		if ((CellStaticParams<double>::product2[thread] > 0) and (abs(CellStaticParams<double>::product3[thread])==0)) return true;
	} else if (CellStaticParams<double>::product1[thread] < 0) {
		if ((abs(CellStaticParams<double>::product2[thread])==0) and (CellStaticParams<double>::product3[thread] < 0)) return true;
		if ((CellStaticParams<double>::product2[thread] < 0) and (abs(CellStaticParams<double>::product3[thread])==0)) return true;
	}

	CellStaticParams<double>::d3[thread][0] = parent_grid->gridparams.sourcept[0] - (*(corner_sourcept[3]))[0];
	CellStaticParams<double>::d3[thread][1] = parent_grid->gridparams.sourcept[1] - (*(corner_sourcept[3]))[1];
	CellStaticParams<double>::product2[thread] = CellStaticParams<double>::d3[thread] ^ CellStaticParams<double>::d1[thread];
	CellStaticParams<double>::product3[thread] = CellStaticParams<double>::d2[thread] ^ CellStaticParams<double>::d3[thread];
	if ((CellStaticParams<double>::product1[thread] > 0) and (CellStaticParams<double>::product2[thread] > 0) and (CellStaticParams<double>::product3[thread] > 0)) return true;
	if ((CellStaticParams<double>::product1[thread] < 0) and (CellStaticParams<double>::product2[thread] < 0) and (CellStaticParams<double>::product3[thread] < 0)) return true;

	return false;	// source not enclosed, therefore no images in this cell
}

inline bool GridCell::test_if_sourcept_inside_triangle(lensvector<double>* point1, lensvector<double>* point2, lensvector<double>* point3, const int& thread)
{
	// Check to see if the given cell, when mapped to the source plane, contains 
	// the point in question.
	// The method: split the cell into two triangles. For each triangle, define 
	// vectors from the source point to each corner of the triangle. If the 
	// point is within one of the triangles, then the cross products of 
	// the vectors will all have the same sign (provided the order of the cross 
	// products is cyclic: 1x2, 2x3, 3x1).

	CellStaticParams<double>::d1[thread][0] = parent_grid->gridparams.sourcept[0] - (*point1)[0];
	CellStaticParams<double>::d1[thread][1] = parent_grid->gridparams.sourcept[1] - (*point1)[1];
	CellStaticParams<double>::d2[thread][0] = parent_grid->gridparams.sourcept[0] - (*point2)[0];
	CellStaticParams<double>::d2[thread][1] = parent_grid->gridparams.sourcept[1] - (*point2)[1];
	CellStaticParams<double>::d3[thread][0] = parent_grid->gridparams.sourcept[0] - (*point3)[0];
	CellStaticParams<double>::d3[thread][1] = parent_grid->gridparams.sourcept[1] - (*point3)[1];
	CellStaticParams<double>::product1[thread] = CellStaticParams<double>::d1[thread] ^ CellStaticParams<double>::d2[thread];
	CellStaticParams<double>::product2[thread] = CellStaticParams<double>::d3[thread] ^ CellStaticParams<double>::d1[thread];
	CellStaticParams<double>::product3[thread] = CellStaticParams<double>::d2[thread] ^ CellStaticParams<double>::d3[thread];
	if ((CellStaticParams<double>::product1[thread] > 0) and (CellStaticParams<double>::product2[thread] > 0) and (CellStaticParams<double>::product3[thread] > 0)) return true;
	if ((CellStaticParams<double>::product1[thread] < 0) and (CellStaticParams<double>::product2[thread] < 0) and (CellStaticParams<double>::product3[thread] < 0)) return true;

	return false;	// point not enclosed
}

inside_cell GridCell::test_if_inside_sourceplane_cell(lensvector<double>* point, const int& thread)
{
	// Check to see if the given cell, when mapped to the source plane, contains 
	// the point in question.
	// The method: split the cell into two triangles. For each triangle, define 
	// vectors from the source point to each corner of the triangle. If the 
	// point is within one of the triangles, then the cross products of 
	// the vectors will all have the same sign (provided the order of the cross 
	// products is cyclic: 1x2, 2x3, 3x1).

	CellStaticParams<double>::d1[thread][0] = (*point)[0] - (*(corner_sourcept[1]))[0];
	CellStaticParams<double>::d1[thread][1] = (*point)[1] - (*(corner_sourcept[1]))[1];
	CellStaticParams<double>::d2[thread][0] = (*point)[0] - (*(corner_sourcept[2]))[0];
	CellStaticParams<double>::d2[thread][1] = (*point)[1] - (*(corner_sourcept[2]))[1];
	CellStaticParams<double>::d3[thread][0] = (*point)[0] - (*(corner_sourcept[0]))[0];
	CellStaticParams<double>::d3[thread][1] = (*point)[1] - (*(corner_sourcept[0]))[1];
	CellStaticParams<double>::product1[thread] = CellStaticParams<double>::d1[thread] ^ CellStaticParams<double>::d2[thread];
	CellStaticParams<double>::product2[thread] = CellStaticParams<double>::d3[thread] ^ CellStaticParams<double>::d1[thread];
	CellStaticParams<double>::product3[thread] = CellStaticParams<double>::d2[thread] ^ CellStaticParams<double>::d3[thread];
	if ((CellStaticParams<double>::product1[thread] > 0) and (CellStaticParams<double>::product2[thread] > 0) and (CellStaticParams<double>::product3[thread] > 0)) return Inside;
	if ((CellStaticParams<double>::product1[thread] < 0) and (CellStaticParams<double>::product2[thread] < 0) and (CellStaticParams<double>::product3[thread] < 0)) return Inside;

	// if the point is on the "low r" or "low theta" edge of the cell, count it as inside
	if (CellStaticParams<double>::product1[thread] > 0) {
		if ((abs(CellStaticParams<double>::product2[thread])==0) and (CellStaticParams<double>::product3[thread] > 0)) return Edge;
		if ((CellStaticParams<double>::product2[thread] > 0) and (abs(CellStaticParams<double>::product3[thread])==0)) return Edge;
	} else if (CellStaticParams<double>::product1[thread] < 0) {
		if ((abs(CellStaticParams<double>::product2[thread])==0) and (CellStaticParams<double>::product3[thread] < 0)) return Edge;
		if ((CellStaticParams<double>::product2[thread] < 0) and (abs(CellStaticParams<double>::product3[thread])==0)) return Edge;
	}

	CellStaticParams<double>::d3[thread][0] = (*point)[0] - (*(corner_sourcept[3]))[0];
	CellStaticParams<double>::d3[thread][1] = (*point)[1] - (*(corner_sourcept[3]))[1];
	CellStaticParams<double>::product2[thread] = CellStaticParams<double>::d3[thread] ^ CellStaticParams<double>::d1[thread];
	CellStaticParams<double>::product3[thread] = CellStaticParams<double>::d2[thread] ^ CellStaticParams<double>::d3[thread];
	if ((CellStaticParams<double>::product1[thread] > 0) and (CellStaticParams<double>::product2[thread] > 0) and (CellStaticParams<double>::product3[thread] > 0)) return Inside;
	if ((CellStaticParams<double>::product1[thread] < 0) and (CellStaticParams<double>::product2[thread] < 0) and (CellStaticParams<double>::product3[thread] < 0)) return Inside;

	return Outside;	// point not enclosed
}

inline bool GridCell::test_if_inside_cell(const lensvector<double>& point, const int& thread)
{
	// The method: split the cell into two triangles. For each triangle, define 
	// vectors from the point to each corner of the triangle. If the 
	// point is within one of the triangles, then the cross products of 
	// the vectors will all have the same sign (provided the order of the cross 
	// products is cyclic: 1x2, 2x3, 3x1).

	CellStaticParams<double>::d1[thread][0] = point[0] - corner_pt[1][0];
	CellStaticParams<double>::d1[thread][1] = point[1] - corner_pt[1][1];
	CellStaticParams<double>::d2[thread][0] = point[0] - corner_pt[2][0];
	CellStaticParams<double>::d2[thread][1] = point[1] - corner_pt[2][1];
	CellStaticParams<double>::d3[thread][0] = point[0] - corner_pt[0][0];
	CellStaticParams<double>::d3[thread][1] = point[1] - corner_pt[0][1];
	CellStaticParams<double>::product1[thread] = CellStaticParams<double>::d1[thread] ^ CellStaticParams<double>::d2[thread];
	CellStaticParams<double>::product2[thread] = CellStaticParams<double>::d3[thread] ^ CellStaticParams<double>::d1[thread];
	CellStaticParams<double>::product3[thread] = CellStaticParams<double>::d2[thread] ^ CellStaticParams<double>::d3[thread];
	if ((CellStaticParams<double>::product1[thread] > 0) and (CellStaticParams<double>::product2[thread] > 0) and (CellStaticParams<double>::product3[thread] > 0)) return true;
	if ((CellStaticParams<double>::product1[thread] < 0) and (CellStaticParams<double>::product2[thread] < 0) and (CellStaticParams<double>::product3[thread] < 0)) return true;

	// check to see whether point is just outside the edges of the cell, within the accuracy set
	// for image searching
	lensvector<double> sidevec;
	if (CellStaticParams<double>::product1[thread] > 0) {
		if (CellStaticParams<double>::product3[thread] > 0) {
			sidevec = CellStaticParams<double>::d3[thread] - CellStaticParams<double>::d1[thread];
			if (abs(CellStaticParams<double>::product2[thread])/sidevec.norm() < 2*parent_grid->image_pos_accuracy) return true;
		}
		if (CellStaticParams<double>::product2[thread] > 0) {
			sidevec = CellStaticParams<double>::d2[thread] - CellStaticParams<double>::d3[thread];
			if (abs(CellStaticParams<double>::product3[thread])/sidevec.norm() < 2*parent_grid->image_pos_accuracy) return true;
		}
	} else if (CellStaticParams<double>::product1[thread] < 0) {
		if (CellStaticParams<double>::product3[thread] < 0) {
			sidevec = CellStaticParams<double>::d3[thread] - CellStaticParams<double>::d1[thread];
			if (abs(CellStaticParams<double>::product2[thread])/sidevec.norm() < 2*parent_grid->image_pos_accuracy) return true;
		}
		if (CellStaticParams<double>::product2[thread] < 0) {
			sidevec = CellStaticParams<double>::d2[thread] - CellStaticParams<double>::d3[thread];
			if (abs(CellStaticParams<double>::product3[thread])/sidevec.norm() < 2*parent_grid->image_pos_accuracy) return true;
		}
	}

	CellStaticParams<double>::d3[thread][0] = point[0] - corner_pt[3][0];
	CellStaticParams<double>::d3[thread][1] = point[1] - corner_pt[3][1];
	CellStaticParams<double>::product2[thread] = CellStaticParams<double>::d3[thread] ^ CellStaticParams<double>::d1[thread];
	CellStaticParams<double>::product3[thread] = CellStaticParams<double>::d2[thread] ^ CellStaticParams<double>::d3[thread];

	// check to see whether point is just outside the edges of the cell, within the accuracy set
	// for image searching
	if (CellStaticParams<double>::product1[thread] > 0) {
		if (CellStaticParams<double>::product3[thread] > 0) {
			sidevec = CellStaticParams<double>::d3[thread] - CellStaticParams<double>::d1[thread];
			if (abs(CellStaticParams<double>::product2[thread])/sidevec.norm() < 2*parent_grid->image_pos_accuracy) return true;
		}
		if (CellStaticParams<double>::product2[thread] > 0) {
			sidevec = CellStaticParams<double>::d2[thread] - CellStaticParams<double>::d3[thread];
			if (abs(CellStaticParams<double>::product3[thread])/sidevec.norm() < 2*parent_grid->image_pos_accuracy) return true;
		}
	} else if (CellStaticParams<double>::product1[thread] < 0) {
		if (CellStaticParams<double>::product3[thread] < 0) {
			sidevec = CellStaticParams<double>::d3[thread] - CellStaticParams<double>::d1[thread];
			if (abs(CellStaticParams<double>::product2[thread])/sidevec.norm() < 2*parent_grid->image_pos_accuracy) return true;
		}
		if (CellStaticParams<double>::product2[thread] < 0) {
			sidevec = CellStaticParams<double>::d2[thread] - CellStaticParams<double>::d3[thread];
			if (abs(CellStaticParams<double>::product3[thread])/sidevec.norm() < 2*parent_grid->image_pos_accuracy) return true;
		}
	}

	if ((CellStaticParams<double>::product1[thread] > 0) and (CellStaticParams<double>::product2[thread] > 0) and (CellStaticParams<double>::product3[thread] > 0)) return true;
	if ((CellStaticParams<double>::product1[thread] < 0) and (CellStaticParams<double>::product2[thread] < 0) and (CellStaticParams<double>::product3[thread] < 0)) return true;

	return false;	// point not enclosed within cell
}

inline bool GridCell::test_if_galaxy_nearby(const lensvector<double>& point, const double& distsq)
{
	lensvector<double> d_corner;
	for (int i=0; i < 4; i++) {
		d_corner[0] = point[0] - corner_pt[i][0];
		d_corner[1] = point[1] - corner_pt[i][1];
		if (d_corner.sqrnorm() < distsq) return true;
	}
	lensvector<double> disp;
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
	lensvector<double> nearby_pt;
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

edge_sourcept_status GridCell::check_subgrid_neighbor_boundaries(const int& neighbor_direction, GridCell* neighbor_subcell, lensvector<double>& centerpt, const int& thread)
{
	edge_sourcept_status status = NoSource;
	inside_cell inside_sourceplane_cell;
	lensvector<double> *interior_edge_point, *edgept1, *edgept2;
	lensvector<double> *interior_edge_point_src, *edgept1_src, *edgept2_src;
	bool edgept1_parity, edgept2_parity; // make static multithreaded variables?
	GridCell *gridcell;
	if (neighbor_direction==0) {
		gridcell = neighbor_subcell->cell[0][0];
		interior_edge_point_src = gridcell->corner_sourcept[1];
		edgept1_src = neighbor_subcell->corner_sourcept[0];
		edgept2_src = neighbor_subcell->corner_sourcept[1];
		edgept1_parity = sign_bool(*gridcell->corner_invmag[0]);
		edgept2_parity = sign_bool(*gridcell->corner_invmag[1]);
		CellStaticParams<double>::d1[thread][0] = (*edgept1_src)[0] - (*edgept2_src)[0];
		CellStaticParams<double>::d1[thread][1] = (*edgept1_src)[1] - (*edgept2_src)[1];
		CellStaticParams<double>::d2[thread][0] = (*interior_edge_point_src)[0] - (*edgept2_src)[0];
		CellStaticParams<double>::d2[thread][1] = (*interior_edge_point_src)[1] - (*edgept2_src)[1];
	} else if (neighbor_direction==1) {
		gridcell = neighbor_subcell->cell[1][0];
		interior_edge_point_src = gridcell->corner_sourcept[3];
		edgept1_src = neighbor_subcell->corner_sourcept[2];
		edgept2_src = neighbor_subcell->corner_sourcept[3];
		edgept1_parity = sign_bool(*gridcell->corner_invmag[2]);
		edgept2_parity = sign_bool(*gridcell->corner_invmag[3]);
		CellStaticParams<double>::d1[thread][0] = (*edgept2_src)[0] - (*edgept1_src)[0];
		CellStaticParams<double>::d1[thread][1] = (*edgept2_src)[1] - (*edgept1_src)[1];
		CellStaticParams<double>::d2[thread][0] = (*interior_edge_point_src)[0] - (*edgept1_src)[0];
		CellStaticParams<double>::d2[thread][1] = (*interior_edge_point_src)[1] - (*edgept1_src)[1];
	} else if (neighbor_direction==2) {
		gridcell = neighbor_subcell->cell[0][0];
		interior_edge_point_src = gridcell->corner_sourcept[2];
		edgept1_src = neighbor_subcell->corner_sourcept[0];
		edgept2_src = neighbor_subcell->corner_sourcept[2];
		edgept1_parity = sign_bool(*gridcell->corner_invmag[0]);
		edgept2_parity = sign_bool(*gridcell->corner_invmag[2]);
		CellStaticParams<double>::d1[thread][0] = (*edgept2_src)[0] - (*edgept1_src)[0];
		CellStaticParams<double>::d1[thread][1] = (*edgept2_src)[1] - (*edgept1_src)[1];
		CellStaticParams<double>::d2[thread][0] = (*interior_edge_point_src)[0] - (*edgept1_src)[0];
		CellStaticParams<double>::d2[thread][1] = (*interior_edge_point_src)[1] - (*edgept1_src)[1];
	} else if (neighbor_direction==3) {
		gridcell = neighbor_subcell->cell[0][1];
		interior_edge_point_src = gridcell->corner_sourcept[3];
		edgept1_src = neighbor_subcell->corner_sourcept[1];
		edgept2_src = neighbor_subcell->corner_sourcept[3];
		edgept1_parity = sign_bool(*gridcell->corner_invmag[1]);
		edgept2_parity = sign_bool(*gridcell->corner_invmag[3]);
		CellStaticParams<double>::d1[thread][0] = (*edgept1_src)[0] - (*edgept2_src)[0];
		CellStaticParams<double>::d1[thread][1] = (*edgept1_src)[1] - (*edgept2_src)[1];
		CellStaticParams<double>::d2[thread][0] = (*interior_edge_point_src)[0] - (*edgept2_src)[0];
		CellStaticParams<double>::d2[thread][1] = (*interior_edge_point_src)[1] - (*edgept2_src)[1];
	}
	if (edgept1_parity == edgept2_parity) {
		CellStaticParams<double>::product1[thread] = CellStaticParams<double>::d1[thread] ^ CellStaticParams<double>::d2[thread];
		if (edgept1_parity == true) inside_sourceplane_cell = (CellStaticParams<double>::product1[thread] < 0) ? Inside : (CellStaticParams<double>::product1[thread] > 0) ? Outside : Edge; // positive parity
		else inside_sourceplane_cell = (CellStaticParams<double>::product1[thread] > 0) ? Inside : (CellStaticParams<double>::product1[thread] < 0) ? Outside : Edge; // negative parity
	} else {
		inside_sourceplane_cell = test_if_inside_sourceplane_cell(interior_edge_point_src,thread);
	}
	if (inside_sourceplane_cell==Outside) {
		if (test_if_sourcept_inside_triangle(edgept1_src,edgept2_src,interior_edge_point_src,thread)==true) {
			gridcell = neighbor_subcell;
			if (neighbor_direction==0) {
				interior_edge_point = &neighbor_subcell->cell[0][0]->corner_pt[1];
				edgept1 = &gridcell->corner_pt[0];
				edgept2 = &gridcell->corner_pt[1];
			}
			else if (neighbor_direction==1) {
				interior_edge_point = &neighbor_subcell->cell[1][0]->corner_pt[3];
				edgept1 = &gridcell->corner_pt[2];
				edgept2 = &gridcell->corner_pt[3];
			}
			else if (neighbor_direction==2) {
				interior_edge_point = &neighbor_subcell->cell[0][0]->corner_pt[2];
				edgept1 = &gridcell->corner_pt[0];
				edgept2 = &gridcell->corner_pt[2];
			}
			else if (neighbor_direction==3) {
				interior_edge_point = &neighbor_subcell->cell[0][1]->corner_pt[3];
				edgept1 = &gridcell->corner_pt[1];
				edgept2 = &gridcell->corner_pt[3];
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

template <typename QScalar>
void ImgSrchGrid::grid_search_firstlevel(const int& searchlevel)
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

//cout << "HI0" << endl;
		int ntot = u_N*w_N;
		int k,i,j;
		//#pragma omp for schedule(dynamic)
		for (k=0; k < ntot; k++) {
			j = k / u_N;
			i = k % u_N;
			cell[i][j]->grid_search<QScalar>(searchlevel,thread);
		}
//cout << "HIf" << endl;
	//}
}
template void ImgSrchGrid::grid_search_firstlevel<double>(const int& searchlevel);
#ifdef USE_STAN
template void ImgSrchGrid::grid_search_firstlevel<stan::math::var>(const int& searchlevel);
#endif

template <typename QScalar>
void GridCell::grid_search(const int& searchlevel, const int& thread)
{
	if (parent_grid->finished_search) return;
	if ((lens->include_central_image==false) and (cell_in_central_image_region==true)) return;
	// 'searchlevel' specifies level at which we should start hunting for images.
	// If the level is at or above the searchlevel, start searching for images;
	// otherwise, descend into higher level until we reach the specified searchlevel

	if ((cell != NULL) and (level < searchlevel))
	{
		int i,j;
		for (j=0; j < w_N; j++) {
			for (i=0; i < u_N; i++) {
				if (parent_grid->finished_search) break;
				cell[i][j]->grid_search<QScalar>(searchlevel,thread);
			}
		}
	}
	else if (!singular_pt_inside)
	{
		bool cell_maps_around_sourcept = image_test(thread);
		lensvector<QScalar> imgpos;
		if (cell==NULL) {
			lensvector<double> imgpos_doub;
			for (int i=0; i < 4; i++) {
				if ((neighbor[i] != NULL) and (neighbor[i]->cell != NULL)) {
					edge_sourcept_status status = check_subgrid_neighbor_boundaries(i, neighbor[i], imgpos_doub, thread); // imgpos is set to the center of the triangle if source is found in the corresponding ray-traced triangle
					if (status==SourceInGap) {
						imgpos[0] = imgpos_doub[0];
						imgpos[1] = imgpos_doub[1];
						if ((lens->skip_newtons_method) or (run_newton<QScalar>(imgpos,thread)==true))
							parent_grid->add_image_to_list(imgpos);
					} else if (status==SourceInOverlap) {
						cell_maps_around_sourcept = false; // even if this cell maps around the source, don't search if it overlaps with neighboring subcells
					}
				}
			}
		}
		if (cell_maps_around_sourcept) {
			imgpos[0] = center_imgplane[0];
			imgpos[1] = center_imgplane[1];
			if ((lens->skip_newtons_method) or (run_newton<QScalar>(imgpos,thread)==true)) {
				parent_grid->add_image_to_list(imgpos);
			}
		}
	}
}
template void GridCell::grid_search<double>(const int& searchlevel, const int& thread);
#ifdef USE_STAN
template void GridCell::grid_search<stan::math::var>(const int& searchlevel, const int& thread);
#endif

template <typename QScalar>
void ImgSrchGrid::add_image_to_list(const lensvector<QScalar>& imgpos)
{
	GridParams<QScalar>& p = assign_gridparam_object<QScalar>();
	p.images[nfound].pos[0] = imgpos[0];
	p.images[nfound].pos[1] = imgpos[1];
	p.images[nfound].mag = lens->magnification<QScalar>(imgpos,0,grid_zfactors,grid_betafactors);
	if (lens->include_time_delays) {
		QScalar potential = lens->potential<QScalar>(imgpos,grid_zfactors,grid_betafactors);
		p.images[nfound].td = 0.5*(SQR(imgpos[0]-p.sourcept[0])+SQR(imgpos[1]-p.sourcept[1])) - potential; // the dimensionless version; it will be converted to days by the QLens class
	} else {
		p.images[nfound].td = 0;
	}
	p.images[nfound].parity = sign(p.images[nfound].mag);

	nfound++;
	if (nfound >= parent_grid->max_images) finished_search = true;
}
template void ImgSrchGrid::add_image_to_list<double>(const lensvector<double>& imgpos);
#ifdef USE_STAN
template void ImgSrchGrid::add_image_to_list<stan::math::var>(const lensvector<stan::math::var>& imgpos);
#endif

void ImgSrchGrid::subgrid_around_galaxies(lensvector<double>* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_splittings, bool* subgrid)
{
	for (int i=0; i < n_cc_splittings; i++)
		subgrid_around_galaxies_iteration(galaxy_centers,ngal,subgrid_radius,min_galsubgrid_cellsize,i,false,subgrid);
	subgrid_around_galaxies_iteration(galaxy_centers,ngal,subgrid_radius,min_galsubgrid_cellsize,n_cc_splittings,false,subgrid);
}

void GridCell::subgrid_around_galaxies_iteration(lensvector<double>* galaxy_centers, const int& ngal, double* subgrid_radius, double* min_galsubgrid_cellsize, const int& n_cc_splittings, bool cc_neighbor_splitting, bool *subgrid)
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

template <typename QScalar>
image<QScalar>* ImgSrchGrid::tree_search(const lensvector<QScalar> source_in)
{
	GridParams<QScalar>& p = assign_gridparam_object<QScalar>();
	p.sourcept[0] = source_in[0];
	p.sourcept[1] = source_in[1];
#ifdef USE_STAN
	if constexpr (std::is_same_v<QScalar, stan::math::var>) {
		// we want the "double" version which will be used for grid search stuff etc.
		gridparams.sourcept[0] = source_in[0].val();
		gridparams.sourcept[1] = source_in[1].val();
	} 
#endif

   finished_search = false;
	grid_search_firstlevel<QScalar>(levels);

   return p.images;
}
template image<double>* ImgSrchGrid::tree_search<double>(const lensvector<double> source_in);
#ifdef USE_STAN
template image<stan::math::var>* ImgSrchGrid::tree_search<stan::math::var>(const lensvector<stan::math::var> source_in);
#endif

template <typename QScalar>
bool GridCell::redundancy(const lensvector<QScalar>& xroot, QScalar &sep)
{
	GridParams<QScalar>& p = parent_grid->assign_gridparam_object<QScalar>();
	bool redundancy = false;
	for (int k = 0; k < parent_grid->nfound; k++)
	{
		sep = sqrt(SQR(xroot[0]-p.images[k].pos[0]) + SQR(xroot[1]-p.images[k].pos[1]));
		if (sep < lens->redundancy_separation_threshold)
		{
			redundancy = true;

			break;
		}
	}
	return redundancy;
}
template bool GridCell::redundancy<double>(const lensvector<double>& xroot, double &sep);
#ifdef USE_STAN
template bool GridCell::redundancy<stan::math::var>(const lensvector<stan::math::var>& xroot, stan::math::var &sep);
#endif

template <typename QScalar>
void QLens::find_images(image<QScalar>*& images_found, const lensvector<QScalar> source_in)
{
	// called by plot_images(...)

	grid->reset_search_parameters();
	images_found = grid->tree_search<QScalar>(source_in);

	if (include_time_delays) {
		double td_factor = cosmo->time_delay_factor_arcsec(lens_redshift,reference_source_redshift);
		double td, min_td=1e30;
		int i;
		for (i = 0; i < grid->nfound; i++) {
#ifdef USE_STAN
			if constexpr (std::is_same_v<QScalar, stan::math::var>) {
				td = images_found[i].td.val();
			} else
#endif
			td = images_found[i].td;
			if (td < min_td) {
#ifdef USE_STAN
				if constexpr (std::is_same_v<QScalar, stan::math::var>) {
					min_td = images_found[i].td.val();
				} else
#endif
				min_td = images_found[i].td;
			}
		}
		for (i = 0; i < grid->nfound; i++) {
			images_found[i].td -= min_td;
			if (images_found[i].td != 0.0) images_found[i].td *= td_factor;
		}
	}
}
template void QLens::find_images<double>(image<double>*& images_found, const lensvector<double> source_in);
#ifdef USE_STAN
template void QLens::find_images<stan::math::var>(image<stan::math::var>*& images_found, const lensvector<stan::math::var> source_in);
#endif

void QLens::output_images_single_source(const double &x_source, const double &y_source, bool verbal, const double flux, const bool show_labels)
{
	if (grid==NULL) {
		if (create_grid(verbal,reference_zfactors,default_zsrc_beta_factors)==false) {
			return;
		}
	}

	lensvector<double> source;
	source[0] = x_source; source[1] = y_source;

	image<double> *images_found;
	find_images(images_found,source);

	if (mpi_id==0) {
		if (use_scientific_notation==false) {
			cout << setprecision(6);
			cout << fixed;
		}
		cout << "#src_x (arcsec)\tsrc_y (arcsec)\tn_images";
		if (flux != -1) cout << "\tsrc_flux";
		cout << endl;
		cout << source[0] << "\t" << source[1] << "\t" << grid->nfound << "\t";
		if (flux != -1) cout << "\t" << flux;
		cout << endl << endl;

		//cout << "# " << grid->nfound << " images" << endl;
		if (show_labels) {
			cout << "#pos_x (arcsec)\tpos_y (arcsec)\tmagnification";
			if (flux != -1.0) cout << "\tflux\t";
			if (include_time_delays) cout << "\ttime_delay (days)";
			cout << endl;
		}
		if (include_time_delays) {
			for (int i = 0; i < grid->nfound; i++) {
				if (flux == -1.0) cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].td << endl;
				else cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].mag*flux << "\t" << images_found[i].td << endl;
			}
		} else {
			for (int i = 0; i < grid->nfound; i++) {
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
	lensvector<double> source;
	source[0] = x_source; source[1] = y_source;

	image<double> *images_found;
	find_images(images_found,source);

	if (mpi_id==0) {
		cout << "#src_x (arcsec)\tsrc_y (arcsec)\tn_images";
		if (flux != -1) cout << "\tsrc_flux";
		cout << endl;
		cout << source[0] << "\t" << source[1] << "\t" << grid->nfound << "\t";
		if (flux != -1) cout << "\t" << flux;
		cout << endl << endl;

		srcfile << x_source << " " << y_source << endl;
		//cout << "# " << grid->nfound << " images" << endl;
		if (show_labels) {
			cout << "#pos_x (arcsec)\tpos_y (arcsec)\tmagnification";
			if (flux != -1.0) cout << "\tflux\t";
			if (include_time_delays) cout << "\ttime_delay (days)";
			cout << endl;
		}
		if (include_time_delays) {
			for (int i = 0; i < grid->nfound; i++) {
				if (flux == -1.0) cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].td << endl;
				else cout << images_found[i].pos[0] << "\t" << images_found[i].pos[1] << "\t" << images_found[i].mag << "\t" << images_found[i].mag*flux << "\t" << images_found[i].td << endl;
				imgfile << images_found[i].pos[0] << " " << images_found[i].pos[1] << endl;
			}
		} else {
			for (int i = 0; i < grid->nfound; i++) {
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

void PtImageSet::copy_imageset(const lensvector<double>& srcpos_in, const double zsrc_in, image<double>* images_in, const int nimg)
{
	n_images = nimg;
	zsrc = zsrc_in;
	srcpos[0] = srcpos_in[0];
	srcpos[1] = srcpos_in[1];
	images.clear();
	images.resize(n_images);
	for (int i=0; i < n_images; i++) {
		images[i].pos = images_in[i].pos;
		images[i].mag = images_in[i].mag;
		images[i].td = images_in[i].td;
		images[i].parity = images_in[i].parity;
	}
}

string PtImageSet::output_images_string(const bool use_sci)
{
	auto mkstring_doub = [](const double db, const bool use_sci) {
		stringstream dstr;
		if (use_sci==false) {
			dstr << setprecision(6);
			dstr << fixed;
		} else {
			dstr << setiosflags(ios::scientific);
		}
		string dstring;
		dstr << db;
		dstr >> dstring;
		return dstring;
	};

	auto mkstring_int = [](const int i) {
		stringstream istr;
		string istring;
		istr << i;
		istr >> istring;
		return istring;
	};

	string imgstring = "#src_x (arcsec)\tsrc_y (arcsec)\tn_images";
	//if (srcflux != -1) imgstring += "\tsrc_flux";
	imgstring += "\n";
	imgstring += mkstring_doub(srcpos[0],use_sci) + "\t" + mkstring_doub(srcpos[1],use_sci) + "\t" + mkstring_int(n_images) + "\t";
	//if (srcflux != -1) imgstring += "\t" + srcflux;
	imgstring += "\n\n";

	if (n_images==0) {
		imgstring += "# no images were generated\n\n";
	} else {
		imgstring += "#pos_x(arcsec)\tpos_y(arcsec)\tmagnification\n";
		for (int i=0; i < n_images; i++) {
			imgstring += mkstring_doub(images[i].pos[0],use_sci) + "\t" + mkstring_doub(images[i].pos[1],use_sci) + "\t" + mkstring_doub(images[i].mag,use_sci);
			if (include_time_delays) imgstring += "\t" + mkstring_doub(images[i].td,use_sci);
			imgstring += "\n";
		}
	}
	return imgstring;
}

template <typename QScalar>
image<QScalar>* QLens::get_images(const lensvector<QScalar> &source_in, int &n_images, bool verbal)
{
	if (grid==NULL) {
		if (create_grid(verbal,reference_zfactors,default_zsrc_beta_factors)==false) return NULL;
	}

	image<QScalar> *images_found;
	find_images(images_found,source_in);
	n_images = grid->nfound;
	return images_found;
}
template image<double>* QLens::get_images<double>(const lensvector<double> &source_in, int &n_images, bool verbal);
#ifdef USE_STAN
template image<stan::math::var>* QLens::get_images<stan::math::var>(const lensvector<stan::math::var> &source_in, int &n_images, bool verbal);
#endif

// this is for the Python wrapper, but I would like to replace the above functions with this in qlens anyway (DO LATER)
bool QLens::get_imageset(const double src_x, const double src_y, PtImageSet& image_set, bool verbal)
{
	if (grid==NULL) {
		if (create_grid(verbal,reference_zfactors,default_zsrc_beta_factors)==false) return false;
	}
	lensvector<double> source;
	source[0] = src_x;
	source[1] = src_y;

	image<double> *images_found;
	find_images(images_found,source);
	image_set.copy_imageset(source,source_redshift,images_found,grid->nfound);
	image_set.include_time_delays = include_time_delays;
	return true;
}

bool QLens::get_fit_imagesets(int min_dataset, int max_dataset, bool verbal)
{
	if (n_ptsrc==0) return false;
	if (max_dataset < 0) max_dataset = n_ptsrc - 1;
	if ((min_dataset < 0) or (min_dataset > max_dataset)) return false;

	if (analytic_source_flux) set_analytic_srcflux(false);
	if (use_analytic_bestfit_src) set_analytic_sourcepts(false);

	int redshift_idx;
	for (int i=min_dataset; i <= max_dataset; i++) {
		redshift_idx = ptsrc_redshift_idx[i];
		if ((i == min_dataset) or (redshift_idx != ptsrc_redshift_idx[i-1])) {
			if (!create_grid(false,ptsrc_zfactors[redshift_idx],ptsrc_beta_factors[redshift_idx])) return false;
		}

		//source[0] = ptsrc_list[i]->pos[0];
		//source[1] = ptsrc_list[i]->pos[1];
		lensvector<double> source;
		source = ptsrc_list[i]->get_pos();

		image<double> *images_found;
		find_images(images_found,source);
		ptsrc_list[i]->set_images(images_found,grid->nfound);
	}
	reset_grid();
	return true;
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
	lensvector<double> source;
	while (sources >> source[0] >> source[1])
	{
		nsources++;
		if (mpi_id==0) srcdat << source[0] << " " << source[1] << endl;
		image<double> *images_found;
		find_images(images_found,source);

		if (mpi_id==0) {
			imagedat << "# " << grid->nfound << " images" << endl;

			for (int i = 0; i < grid->nfound; i++)
			{
				if (include_time_delays)
					imagedat << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].td << " " << images_found[i].parity << endl;
				else
					imagedat << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
				if (color_multiplicities) {
					if (grid->nfound==5) {
						quads << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
					}
					else if (grid->nfound==3) {
						// this will count doubles and cusps
						doubles << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
					}
					else if (grid->nfound==1) {
						singles << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
					} else {
						weird << images_found[i].pos[0] << " " << images_found[i].pos[1] << " " << images_found[i].mag << " " << images_found[i].parity << endl;
					}
				}
			}
			if (color_multiplicities) {
				if (grid->nfound==5) {
					srcquads << source[0] << " " << source[1] << endl;
				}
				else if (grid->nfound==3) {
					srcdoubles << source[0] << " " << source[1] << endl;
				}
				else if (grid->nfound==1) {
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

const int GridCell::max_iterations = 200;
const int GridCell::max_step_length = 100;

template <typename QScalar>
void GridCell::SolveLinearEqs(lensmatrix<QScalar>& a, lensvector<QScalar>& b)
{
	QScalar det, temp;
	det = determinant(a);
	temp = (-a[1][0]*b[1]+a[1][1]*b[0]) / det;
	b[1] = (-a[0][1]*b[0]+a[0][0]*b[1]) / det;
	b[0] = temp;
}
template void GridCell::SolveLinearEqs<double>(lensmatrix<double>& a, lensvector<double>& b);
#ifdef USE_STAN
template void GridCell::SolveLinearEqs<stan::math::var>(lensmatrix<stan::math::var>& a, lensvector<stan::math::var>& b);
#endif

template <typename QScalar>
bool GridCell::run_newton(lensvector<QScalar>& xroot, const int& thread)
{
	GridParams<QScalar>& p = parent_grid->assign_gridparam_object<QScalar>();
#ifdef USE_STAN
	using stan::math::abs;
#endif
	if ((enforce_min_area) and (parent_grid->image_pos_accuracy > 0.2*sqrt(cell_area))) warn(lens->newton_warnings,"image position accuracy comparable to or larger than cell size");
	if ((xroot[0]==0) and (xroot[1]==0)) { xroot[0] = xroot[1] = 5e-1*lens->cc_rmin; }	// Avoiding singularity at center
	if (NewtonsMethod(xroot, newton_check[thread], thread)==false) {
		warn(lens->newton_warnings,"Newton's method failed for source (%g,%g), level %i, cell center (%g,%g)",parent_grid->gridparams.sourcept[0],parent_grid->gridparams.sourcept[1],level,center_imgplane[0],center_imgplane[1],xroot[0],xroot[1]);
		return false;
	}
	if (lens->reject_images_found_outside_cell) {
		lensvector<double> xroot_doub;
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>) {
			xroot_doub[0] = xroot[0].val();
			xroot_doub[1] = xroot[1].val();
		} else
#endif
		{
			xroot_doub[0] = xroot[0];
			xroot_doub[1] = xroot[1];
		}
		if (test_if_inside_cell(xroot_doub,thread)==false) {
			warn(lens->warnings,"Rejecting image found outside cell for source (%g,%g), level %i, cell center (%g,%g)",parent_grid->gridparams.sourcept[0],parent_grid->gridparams.sourcept[1],level,center_imgplane[0],center_imgplane[1],xroot_doub[0],xroot_doub[1]);
			return false;
		}
	}

	lensvector<QScalar> lens_eq_f;
	lens->lens_equation<QScalar>(xroot,p.sourcept,lens_eq_f,thread,parent_grid->grid_zfactors,parent_grid->grid_betafactors);
	lensvector<double> xroot_doub;
#ifdef USE_STAN
	if constexpr (std::is_same_v<QScalar, stan::math::var>) {
		xroot_doub[0] = xroot[0].val();
		xroot_doub[1] = xroot[1].val();
	} else
#endif
	{
		xroot_doub[0] = xroot[0];
		xroot_doub[1] = xroot[1];
	}

	//QScalar lenseq_mag = sqrt(SQR(lens_eq_f[0]) + SQR(lens_eq_f[1]));
	//QScalar tryacc = parent_grid->image_pos_accuracy / sqrt(abs(lens->magnification(xroot,thread,zfactor)));
	//cout << lenseq_mag << " " << tryacc << " " << sqrt(abs(lens->magnification(xroot,thread,zfactor))) << endl;
	if (newton_check[thread]==true) { warn(lens->newton_warnings, "false image--converged to local minimum"); return false; }
	if (lens->n_singular_points > 0) {
		QScalar singular_pt_accuracy = 2*parent_grid->image_pos_accuracy;
		for (int i=0; i < lens->n_singular_points; i++) {
			if ((abs(xroot_doub[0]-lens->singular_pts[i][0]) < singular_pt_accuracy) and (abs(xroot_doub[1]-lens->singular_pts[i][1]) < singular_pt_accuracy)) {
				warn(lens->newton_warnings,"Newton's method converged to singular point (%g,%g) for source (%g,%g)",lens->singular_pts[i][0],lens->singular_pts[i][1],parent_grid->gridparams.sourcept[0],parent_grid->gridparams.sourcept[1]);
				return false;
			}
		}
	}
	if (((xroot_doub[0]==center_imgplane[0]) and (center_imgplane[0] != 0)) and ((xroot_doub[1]==center_imgplane[1]) and (center_imgplane[1] != 0)))
		warn(lens->newton_warnings, "Newton's method returned center of grid cell");
	double mag = lens->magnification<double>(xroot_doub,thread,parent_grid->grid_zfactors,parent_grid->grid_betafactors);
	if ((abs(lens_eq_f[0]) > 1000*parent_grid->image_pos_accuracy) and (abs(lens_eq_f[1]) > 1000*parent_grid->image_pos_accuracy) and (abs(mag) < 1e-3)) {
		if (lens->newton_warnings==true) {
			warn(lens->newton_warnings,"Newton's method may have found false root (%g,%g) (within 1000*accuracy) for source (%g,%g), level %i, cell center (%g,%g), mag %g",xroot_doub[0],xroot_doub[1],parent_grid->gridparams.sourcept[0],parent_grid->gridparams.sourcept[1],level,center_imgplane[0],center_imgplane[1],xroot_doub[0],xroot_doub[1],mag);
		}
	}
	if (abs(mag) > lens->newton_magnification_threshold) {
		if (lens->reject_himag_images) {
			if ((lens->mpi_id==0) and (lens->warnings)) {
				cout << "*WARNING*: Rejecting image that exceeds imgsrch_mag_threshold (" << abs(mag) << "), src=(" << parent_grid->gridparams.sourcept[0] << "," << parent_grid->gridparams.sourcept[1] << "), x=(" << xroot_doub[0] << "," << xroot_doub[1] << ")      " << endl;
				if (lens->use_ansi_characters) {
					cout << "                                                                                                                            " << endl;
					cout << "\033[2A";
				}
			}
			return false;
		} else {
			if ((lens->mpi_id==0) and (lens->warnings)) {
				cout << "*WARNING*: Image exceeds imgsrch_mag_threshold (" << abs(mag) << "); src=(" << parent_grid->gridparams.sourcept[0] << "," << parent_grid->gridparams.sourcept[1] << "), x=(" << xroot_doub[0] << "," << xroot_doub[1] << ")        " << endl;
				if (lens->use_ansi_characters) {
					cout << "                                                                                                                            " << endl;
					cout << "\033[2A";
				}
			}
		}
	}
	if ((lens->include_central_image==false) and (mag > 0) and (lens->kappa<QScalar>(xroot,parent_grid->grid_zfactors,parent_grid->grid_betafactors) > 1)) return false; // discard central image if not desired
	bool status = true;
	//#pragma omp critical
	//{
		QScalar sep;
		if (redundancy(xroot,sep)) {
			// generally, this only occurs very close to critical curves and is best solved by further cell splittings
			// around said curves. However for extreme magnifications (near Einstein-ring images), even cell splittings
			// does not solve the issue.
			if (lens->newton_warnings==true) {
				warn(lens->newton_warnings,"rejecting probable duplicate image (imgsep=%g): src (%g,%g), level %i, image (%g,%g), mag %g",sep,parent_grid->gridparams.sourcept[0],parent_grid->gridparams.sourcept[1],level,xroot_doub[0],xroot_doub[1],mag);
			}
			status = false;
		}
		else if (parent_grid->nfound >= parent_grid->max_images) status = false;
	//}
	return status;
}
template bool GridCell::run_newton<double>(lensvector<double>& xroot, const int& thread);
#ifdef USE_STAN
template bool GridCell::run_newton<stan::math::var>(lensvector<stan::math::var>& xroot, const int& thread);
#endif

template <typename QScalar>
bool GridCell::NewtonsMethod(lensvector<QScalar>& x, bool &check, const int& thread)
{
	GridParams<QScalar>& p = parent_grid->assign_gridparam_object<QScalar>();
#ifdef USE_STAN
	using stan::math::fabs;
#endif
	check = false;
	lensvector<QScalar> g, pp, xold;
	lensmatrix<QScalar> fjac;

	lens->lens_equation<QScalar>(x, p.sourcept, CellStaticParams<QScalar>::fvec[thread], thread, parent_grid->grid_zfactors, parent_grid->grid_betafactors);
	QScalar f = 0.5*CellStaticParams<QScalar>::fvec[thread].sqrnorm();
	if (max_component(CellStaticParams<QScalar>::fvec[thread]) < 0.01*parent_grid->image_pos_accuracy)
		return true; 

	QScalar fold, stpmax, temp, test;
#ifdef USE_STAN
	if constexpr (std::is_same_v<QScalar, stan::math::var>) {
		double stpmax_doub = max_step_length * maxval(x.norm().val(), 2.0); 
		stpmax = stpmax_doub;
	} else
#endif
	stpmax = max_step_length * maxval(x.norm(), 2.0); 
	for (int its=0; its < max_iterations; its++) {
		lens->hessian<QScalar>(x[0],x[1],fjac,thread,parent_grid->grid_zfactors,parent_grid->grid_betafactors);
		fjac[0][0] = -1 + fjac[0][0];
		fjac[1][1] = -1 + fjac[1][1];
		g[0] = fjac[0][0] * CellStaticParams<QScalar>::fvec[thread][0] + fjac[0][1]*CellStaticParams<QScalar>::fvec[thread][1];
		g[1] = fjac[1][0] * CellStaticParams<QScalar>::fvec[thread][0] + fjac[1][1]*CellStaticParams<QScalar>::fvec[thread][1];
		xold[0] = x[0];
		xold[1] = x[1];
		fold = f; 
		pp[0] = -CellStaticParams<QScalar>::fvec[thread][0];
		pp[1] = -CellStaticParams<QScalar>::fvec[thread][1];
		SolveLinearEqs(fjac, pp);
		if (LineSearch(xold, fold, g, pp, x, f, stpmax, check, thread)==false)
			return false;
		if ((x[0] > 1e3*lens->cc_rmax) or (x[1] > 1e3*lens->cc_rmax)) {
			warn(lens->newton_warnings, "Newton blew up!");
			return false;
		}
		/*
		lens->lens_equation(x, CellStaticParams<QScalar>::fvec[thread], thread, zfactor);
		QScalar magfac = sqrt(abs(lens->magnification(x,thread,zfactor)));
		QScalar tryacc;
		lensvector<QScalar> dx = x - xold;
		QScalar dxnorm = dx.norm();
		dx[0] /= dxnorm;
		dx[1] /= dxnorm;
		lensmatrix<QScalar> magmat;
		lensvector<QScalar> bb;
		lens->sourcept_jacobian(x,bb,magmat,thread,zfactor);
		bb = magmat*dx;
		lensvector<QScalar> dy;
		dy[1] = -dx[0];
		dy[0] = dx[1];
		lensvector<QScalar> cc;
		tryacc = parent_grid->image_pos_accuracy * bb.norm();
		*/
		//if (max_component(CellStaticParams<QScalar>::fvec[thread]) < 4*tryacc) {

		// Maybe someday revisit this and see if you can make it more robust. As it is, it's
		// frustrating that image_pos_accuracy has no simple interpretation, and occasionally
		// spurious images close to critical curves are found.
		if (max_component(CellStaticParams<QScalar>::fvec[thread]) < parent_grid->image_pos_accuracy) {
			check = false; 
			return true; 
		}
		QScalar one = 1.0;
		if (check) {
			QScalar den = maxval(f, one); 
			temp = fabs(g[0]) * maxval(fabs(x[0]), one)/den; 
			test = fabs(g[1]) * maxval(fabs(x[1]), one)/den; 
			check = (maxval(test,temp) < parent_grid->image_pos_accuracy); 
			return true; 
		}
		test = (fabs(x[0] - xold[0])) / maxval(fabs(x[0]), one); 
		temp = (fabs(x[1] - xold[1])) / maxval(fabs(x[1]), one); 
		if (temp > test) test = temp; 
		if (test < parent_grid->image_pos_accuracy) return true; 
	}

	return false;
}
template bool GridCell::NewtonsMethod<double>(lensvector<double>& x, bool &check, const int& thread);
#ifdef USE_STAN
template bool GridCell::NewtonsMethod<stan::math::var>(lensvector<stan::math::var>& x, bool &check, const int& thread);
#endif

template <typename QScalar>
bool GridCell::LineSearch(lensvector<QScalar>& xold, QScalar fold, lensvector<QScalar>& g, lensvector<QScalar>& pp, lensvector<QScalar>& x, QScalar& f, QScalar stpmax, bool &check, const int& thread)
{
	GridParams<QScalar>& p = parent_grid->assign_gridparam_object<QScalar>();
#ifdef USE_STAN
	using stan::math::fabs;
#endif
	const QScalar alpha = 1.0e-4;	// Ensures sufficient decrease in function value (see NR Ch. 9.7)

	QScalar a, alam, alam2, alamin, b, disc, f2, rhs1, rhs2, slope, mag, temp, test, tmplam;

	check = false;
	mag = pp.norm();
	if (mag > stpmax) {
		QScalar fac = stpmax / mag;
		pp[0] *= fac;
		pp[1] *= fac;
	}
	slope = g[0]*pp[0] + g[1]*pp[1];
	if (slope >= 0.0) die("Roundoff problem during line search (g=(%g,%g), pp=(%g,%g))",g[0],g[1],pp[0],pp[1]); 
	QScalar one = 1.0;
	test = fabs(pp[0]) / maxval(fabs(xold[0]), one); 
	temp = fabs(pp[1]) / maxval(fabs(xold[1]), one); 
	alamin = parent_grid->image_pos_accuracy / maxval(temp,test); 
	alam = 1.0; 
	while (true)
	{
		x[0] = xold[0] + alam*pp[0];
		x[1] = xold[1] + alam*pp[1];
		if ((fabs(x[0]) < 1e6*lens->cc_rmax) and (fabs(x[1]) < 1e6*lens->cc_rmax))
			;
		else {
			warn(lens->newton_warnings, "Newton blew up!");
			return false;
		}
		lens->lens_equation<QScalar>(x, p.sourcept, CellStaticParams<QScalar>::fvec[thread], thread, parent_grid->grid_zfactors, parent_grid->grid_betafactors);
		f = 0.5 * CellStaticParams<QScalar>::fvec[thread].sqrnorm();
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
		alam = maxval(tmplam, 0.1*alam);
	}
}
template bool GridCell::LineSearch<double>(lensvector<double>& xold, double fold, lensvector<double>& g, lensvector<double>& pp, lensvector<double>& x, double& f, double stpmax, bool &check, const int& thread);
#ifdef USE_STAN
template bool GridCell::LineSearch<stan::math::var>(lensvector<stan::math::var>& xold, stan::math::var fold, lensvector<stan::math::var>& g, lensvector<stan::math::var>& pp, lensvector<stan::math::var>& x, stan::math::var& f, stan::math::var stpmax, bool &check, const int& thread);
#endif

void ImgSrchGrid::reset_search_parameters()
{
	nfound = 0;
	nfound_max = 0; nfound_pos = 0; nfound_neg = 0;
}

void GridCell::clear_subcells(int clear_level)
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

GridCell::~GridCell()
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

ImgSrchGrid::~ImgSrchGrid()
{
	delete[] gridparams.images;
#ifdef USE_STAN
	delete[] gridparams_dif.images;
#endif
}

/************************ Functions in class PointSource (perhaps put in different file?) ***************************/

PointSource::PointSource(QLens* lens_in) : Model()
{
	modelparams = &ptsrc_params;
#ifdef USE_STAN
	modelparams_dif = &ptsrc_params_dif;
#endif
	model_name = "ptsrc";
	qlens = lens_in;
	setup_parameters(true);
	setup_param_pointers<double>();
#ifdef USE_STAN
	setup_param_pointers<stan::math::var>();
#endif

}

PointSource::PointSource(QLens* lens_in, const lensvector<double>& sourcept, const double zsrc_in) : Model()
{
	modelparams = &ptsrc_params;
#ifdef USE_STAN
	modelparams_dif = &ptsrc_params_dif;
#endif
	qlens = lens_in;
	setup_parameters(true);
	setup_param_pointers<double>();
#ifdef USE_STAN
	setup_param_pointers<stan::math::var>();
#endif

	PtSrcParams<double>& p = assign_ptsrc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.zsrc = zsrc_in;
	p.pos[0] = sourcept[0];
	p.pos[1] = sourcept[1];
#ifdef USE_STAN
	sync_autodif_parameters();
#endif
}

void PointSource::setup_parameters(const bool initial_setup)
{
	if (initial_setup) {
		setup_parameter_arrays(6);
	} else {
		// always reset the active parameter flags, since the active ones will be determined below
		// NOTE: if (initial_setup==true), active params are reset in setup_parameter_arrays(..) above
		n_active_params = 0;
		for (int i=0; i < n_params; i++) {
			active_params[i] = false; // default
		}
	}

	int indx = 0;

	include_shift = false;
	if ((qlens) and (qlens->include_ptsrc_shift)) include_shift = true;

	if (initial_setup) {
		paramnames[indx] = "xsrc"; latex_paramnames[indx] = "x"; latex_param_subscripts[indx] = "src";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;

	if (initial_setup) {
		paramnames[indx] = "ysrc"; latex_paramnames[indx] = "y"; latex_param_subscripts[indx] = "src";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;

	if (initial_setup) {
		paramnames[indx] = "srcflux"; latex_paramnames[indx] = "f"; latex_param_subscripts[indx] = "src";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0.01; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = true;
	}
	active_params[indx] = true; 
	n_active_params++;
	indx++;

	if (initial_setup) {
		paramnames[indx] = "zsrc"; latex_paramnames[indx] = "z"; latex_param_subscripts[indx] = "src";
		set_auto_penalty_limits[indx] = true; penalty_lower_limits[indx] = 0; penalty_upper_limits[indx] = 1e30;
		stepsizes[indx] = 0.1; scale_stepsize_by_param_value[indx] = false;
	}
	zsrc_paramnum = indx;
	active_params[indx] = true; 
	n_active_params++;
	indx++;

	if (initial_setup) {
		paramnames[indx] = "xshift"; latex_paramnames[indx] = "\\delta x"; latex_param_subscripts[indx] = "s";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.01; scale_stepsize_by_param_value[indx] = false;
	}
	if (include_shift) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;

	if (initial_setup) {
		paramnames[indx] = "yshift"; latex_paramnames[indx] = "\\delta y"; latex_param_subscripts[indx] = "s";
		set_auto_penalty_limits[indx] = false;
		stepsizes[indx] = 0.01; scale_stepsize_by_param_value[indx] = false;
	}
	if (include_shift) {
		active_params[indx] = true; 
		n_active_params++;
	}
	indx++;
}

template <typename QScalar>
void PointSource::setup_param_pointers()
{
	PtSrcParams<QScalar>& p = assign_ptsrc_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	p.pos[0] = 0.0;
	p.pos[1] = 0.0;
	p.srcflux = 1.0;
	p.zsrc=2.0;
	p.shift[0] = 0.0;
	p.shift[1] = 0.0;

	QScalar** param_ptr = p.param;
	*(param_ptr++) = &p.pos[0];
	*(param_ptr++) = &p.pos[1];
	*(param_ptr++) = &p.srcflux;
	*(param_ptr++) = &p.zsrc;
	*(param_ptr++) = &p.shift[0];
	*(param_ptr++) = &p.shift[1];
}
template void PointSource::setup_param_pointers<double>();
#ifdef USE_STAN
template void PointSource::setup_param_pointers<stan::math::var>();
#endif

void PointSource::copy_ptsrc_data(PointSource* ptsrc_in)
{
	PtSrcParams<double>& p = assign_ptsrc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.pos[0] = ptsrc_in->ptsrc_params.pos[0];
	p.pos[1] = ptsrc_in->ptsrc_params.pos[1];
	p.srcflux = ptsrc_in->ptsrc_params.srcflux;
	p.zsrc=ptsrc_in->ptsrc_params.zsrc;
	p.shift[0] = ptsrc_in->ptsrc_params.shift[0];
	p.shift[1] = ptsrc_in->ptsrc_params.shift[1];
	include_shift = ptsrc_in->include_shift;
	copy_param_arrays(ptsrc_in);
#ifdef USE_STAN
	sync_autodif_parameters();
#endif
}

#ifdef USE_STAN
void PointSource::sync_autodif_parameters()
{
	ptsrc_params_dif.pos[0] = ptsrc_params.pos[0];
	ptsrc_params_dif.pos[1] = ptsrc_params.pos[1];
	ptsrc_params_dif.srcflux = ptsrc_params.srcflux;
	ptsrc_params_dif.zsrc = ptsrc_params.zsrc;
	ptsrc_params_dif.shift[0] = ptsrc_params.shift[0];
	ptsrc_params_dif.shift[1] = ptsrc_params.shift[1];
}
#endif

void PointSource::update_meta_parameters(const bool varied_only_fitparams)
{
	if ((qlens != NULL) and ((vary_params[zsrc_paramnum]) or (!varied_only_fitparams))) qlens->update_ptsrc_redshift_data(); // just in case the source redshift was changed
}

void PointSource::get_parameter_numbers_from_qlens(int& pi, int& pf)
{
	if (qlens) qlens->get_ptsrc_parameter_numbers(entry_number,pi,pf);
}

bool PointSource::register_vary_parameters_in_qlens()
{
	if (qlens != NULL) {
		return qlens->register_ptsrc_vary_parameters(entry_number);
	}
	return true;
}

void PointSource::register_limits_in_qlens()
{
	if (qlens != NULL) {
		qlens->register_ptsrc_prior_limits(entry_number);
	}
}

void PointSource::update_fitparams_in_qlens()
{
	if (qlens != NULL) {
		qlens->update_ptsrc_fitparams(entry_number);
	}
}

void PointSource::set_vary_source_coords()
{
	// parameters 0 and 1 are the source position coordinates
	if (!vary_params[0]) n_vary_params++;
	vary_params[0] = true;
	if (!vary_params[1]) n_vary_params++;
	vary_params[1] = true;
}

void PointSource::copy_imageset(const lensvector<double>& pos_in, const double zsrc_in, image<double>* images_in, const int nimg, const double srcflux_in)
{
	PtSrcParams<double>& p = assign_ptsrc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	n_images = nimg;
	p.zsrc = zsrc_in;
	p.srcflux = srcflux_in;
	p.pos[0] = pos_in[0];
	p.pos[1] = pos_in[1];
	images.clear();
	images.resize(n_images);
	for (int i=0; i < n_images; i++) {
		images[i].pos = images_in[i].pos;
		images[i].mag = images_in[i].mag;
		images[i].td = images_in[i].td;
		images[i].parity = images_in[i].parity;
	}
#ifdef USE_STAN
	sync_autodif_parameters();
#endif
}

void PointSource::set_images(image<double>* images_in, const int nimg)
{
	n_images = nimg;
	images.clear();
	images.resize(n_images);
	for (int i=0; i < n_images; i++) {
		images[i].pos = images_in[i].pos;
		images[i].mag = images_in[i].mag;
		images[i].td = images_in[i].td;
		images[i].parity = images_in[i].parity;
	}
}

template <typename QScalar>
void PointSource::update_srcpos(const lensvector<QScalar>& srcpt)
{
	PtSrcParams<QScalar>& p = assign_ptsrc_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	p.pos[0] = srcpt[0];
	p.pos[1] = srcpt[1];
	if (include_shift) {
		p.pos[0] += p.shift[0];
		p.pos[1] += p.shift[1];
	}
#ifdef USE_STAN
		// if using autodif params, let's update the non-autodiff params too (or vice versa) for consistency. Maybe revisit this later? Might not be necessary
		if constexpr (std::is_same_v<QScalar, stan::math::var>) {
			ptsrc_params.pos[0] = (ptsrc_params_dif.pos[0]).val();
			ptsrc_params.pos[1] = (ptsrc_params_dif.pos[1]).val();
		} else {
			ptsrc_params_dif.pos[0] = ptsrc_params.pos[0];
			ptsrc_params_dif.pos[1] = ptsrc_params.pos[1];
		}
#endif
}
template void PointSource::update_srcpos<double>(const lensvector<double>& srcpt);
#ifdef USE_STAN
template void PointSource::update_srcpos<stan::math::var>(const lensvector<stan::math::var>& srcpt);
#endif

void PointSource::print_to_file(bool include_time_delays, bool show_labels, ofstream* srcfile, ofstream* imgfile)
{
	PtSrcParams<double>& p = assign_ptsrc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	cout << "#src_x (arcsec)\tsrc_y (arcsec)\tn_images";
	if (p.srcflux != -1) cout << "\tsrc_flux";
	cout << endl;
	cout << p.pos[0] << "\t" << p.pos[1] << "\t" << n_images << "\t";
	if (p.srcflux != -1) cout << "\t" << p.srcflux;
	cout << endl << endl;

	if (srcfile != NULL) (*srcfile) << p.pos[0] << " " << p.pos[1] << endl;
	//cout << "# " << n_images << " images" << endl;
	if (show_labels) {
		cout << "#pos_x (arcsec)\tpos_y (arcsec)\tmagnification";
		if (p.srcflux != -1.0) cout << "\tflux\t";
		if (include_time_delays) cout << "\ttime_delay (days)";
		cout << endl;
	}
	if (include_time_delays) {
		for (int i = 0; i < n_images; i++) {
			if (p.srcflux == -1.0) cout << images[i].pos[0] << "\t" << images[i].pos[1] << "\t" << images[i].mag << "\t" << images[i].td << endl;
			else cout << images[i].pos[0] << "\t" << images[i].pos[1] << "\t" << images[i].mag << "\t" << images[i].mag*p.srcflux << "\t" << images[i].td << endl;
			if (imgfile != NULL) (*imgfile) << images[i].pos[0] << " " << images[i].pos[1] << endl;
		}
	} else {
		for (int i = 0; i < n_images; i++) {
			if (p.srcflux == -1.0) cout << images[i].pos[0] << "\t" << images[i].pos[1] << "\t" << images[i].mag << endl;
			else cout << images[i].pos[0] << "\t" << images[i].pos[1] << "\t" << images[i].mag << "\t" << images[i].mag*p.srcflux << endl;
			if (imgfile != NULL) (*imgfile) << images[i].pos[0] << " " << images[i].pos[1] << endl;
		}
	}

	cout << endl;
}


