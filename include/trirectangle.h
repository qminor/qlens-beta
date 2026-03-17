#ifndef TRIRECTANGLE_H
#define TRIRECTANGLE_H

#include "lensvec.h"
//using namespace std;

class TriRectangleOverlap
{
	private:
	lensvector<double> *vertex[4];
	lensvector<double> overlap_pts[7];
	lensvector<double> **overlap_pts_list;
	int i,j,i_plus_one;
	double triangle_area;
	double x,y;

	protected:
	int n_overlap_pts, n_rectangle_overlap_pts;
	bool in_side_region[4][4];
	bool outside_rectangle;

	private:
	lensvector<double> ab, bc;
	double slope;
	int temp_index;
	lensvector<double> *temp_vector;

	double overlap_area;
	double subtriangle_area;
	lensvector<double> *r01, *r02;
	lensvector<double> aa, bb;
	bool swapped1, swapped2;
	lensvector<double> **newlist;

	double calculate_polygon_area();
	inline bool test_if_inside(const double& x, const double& y);
	inline double dif_cross_product(const double& x, const double& y, const lensvector<double>* A, const lensvector<double>* B);
	inline bool test_if_in_xrange(const double& x, const double& y, const int& i_xmin, const int& i_xmax);
	inline bool test_if_in_yrange(const double& x, const double& y, const int& i_ymin, const int& i_ymax);

	public:
	TriRectangleOverlap();
	~TriRectangleOverlap();
	double find_overlap_area(lensvector<double>& a, lensvector<double>& b, lensvector<double>& c, double xmin, double xmax, double ymin, double ymax);
	bool determine_if_overlap(lensvector<double>& a, lensvector<double>& b, lensvector<double>& c, const double& xmin, const double& xmax, const double& ymin, const double& ymax);
	bool determine_if_overlap_rough(lensvector<double>& a, lensvector<double>& b, lensvector<double>& c, const double& xmin, const double& xmax, const double& ymin, const double& ymax);
	bool determine_if_in_neighborhood(lensvector<double>& a, lensvector<double>& b, lensvector<double>& c, lensvector<double>& d, const double& xmin, const double& xmax, const double& ymin, const double& ymax, bool &inside);
	bool determine_if_in_neighborhood(lensvector<double>& a, lensvector<double>& b, lensvector<double>& c, const double& xmin, const double& xmax, const double& ymin, const double& ymax, bool &inside);
};

#endif // TRIRECTANGLE_H
