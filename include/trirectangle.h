#ifndef TRIRECTANGLE_H
#define TRIRECTANGLE_H

#include "lensvec.h"

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

// this function is for assigning value that sorts out whether it's autodiff or not. (This should not add overhead as long as compiling with -O2 or -O3)
template <typename T>
inline auto get_value_of(const T& x)
{
#ifdef USE_STAN
	 if constexpr (stan::is_autodiff_v<T>) {
		 return stan::math::value_of(x);
	 } else
#endif
    return x;
}

template <typename QScalar>
class TriRectangleOverlap
{
	private:
	lensvector<QScalar> overlap_pts[7];
	lensvector<QScalar> **overlap_pts_list;

	const QScalar *vertex_x[4];
	const QScalar *vertex_y[4];

	int i,j,i_plus_one;
	QScalar triangle_area;
	QScalar x,y;

	protected:
	int n_overlap_pts, n_rectangle_overlap_pts;
	bool in_side_region[4][4];
	bool outside_rectangle;

	private:
	lensvector<QScalar> ab, bc;
	QScalar slope;
	int temp_index;
	lensvector<QScalar> *temp_vector;

	QScalar overlap_area;
	QScalar subtriangle_area;
	lensvector<QScalar> *r01, *r02;
	lensvector<QScalar> aa, bb;
	bool swapped1, swapped2;
	lensvector<QScalar> **newlist;

	QScalar calculate_polygon_area();
	inline bool test_if_inside(const double x, const double y);
	inline double dif_cross_product(const double x, const double y, const QScalar* A_x, const QScalar* A_y, const QScalar* B_x, const QScalar* B_y);
	inline bool test_if_in_xrange(const double x, const double y, const int i_xmin, const int i_xmax);
	inline bool test_if_in_yrange(const double x, const double y, const int i_ymin, const int i_ymax);

	public:
	TriRectangleOverlap();
	~TriRectangleOverlap();
	QScalar find_overlap_area(const QScalar& ax, const QScalar& ay, const QScalar& bx, const QScalar& by, const QScalar& cx, const QScalar& cy, QScalar xmin, QScalar xmax, QScalar ymin, QScalar ymax);
	bool determine_if_overlap(const QScalar& ax, const QScalar& ay, const QScalar& bx, const QScalar& by, const QScalar& cx, const QScalar& cy, const double xmin, const double xmax, const double ymin, const double ymax);
	bool determine_if_in_neighborhood(const QScalar& ax, const QScalar& ay, const QScalar& bx, const QScalar& by, const QScalar& cx, const QScalar& cy, const QScalar& dx, const QScalar& dy, const double xmin, const double xmax, const double ymin, const double ymax, bool &inside);
	bool determine_if_in_neighborhood(const QScalar& ax, const QScalar& ay, const QScalar& bx, const QScalar& by, const QScalar& cx, const QScalar& cy, const double xmin, const double xmax, const double ymin, const double ymax, bool &inside);
};

#define plus_one(i) (((temp_index = (i)+1) < 3) ? temp_index : 0)
#define minus_one(i) (((temp_index = (i)-1) >= 0) ? temp_index : 2)
#define swap_index(i1,i2) temp_index = (i2); (i2) = (i1); (i1) = temp_index;
#define swap_vectors(v1,v2) temp_vector = (v2); (v2) = (v1); (v1) = temp_vector;

template <typename QScalar>
TriRectangleOverlap<QScalar>::TriRectangleOverlap()
{
	r01 = new lensvector<QScalar>;
	r02 = new lensvector<QScalar>;
	n_overlap_pts = 0;
	for (int i=0; i < 7; i++) { overlap_pts[i][0] = 0; overlap_pts[i][1] = 0; }
}

template <typename QScalar>
QScalar TriRectangleOverlap<QScalar>::find_overlap_area(const QScalar& ax, const QScalar& ay, const QScalar& bx, const QScalar& by, const QScalar& cx, const QScalar& cy, QScalar xmin, QScalar xmax, QScalar ymin, QScalar ymax)
{
	// The assignment of the rectangle corner and side indices will be as follows:
	// 3-2-2
	// 3   1
	// 0-0-1

	// this is a bit ugly, but seg faults occur if the rectangle shares any x/y values with the square and this is a quick fix
	if (ax==xmin) xmin -= 1e-10;
	if (bx==xmin) xmin -= 1e-10;
	if (ax==xmax) xmax += 1e-10;
	if (bx==xmax) xmax += 1e-10;

	if (ay==ymin) ymin -= 1e-10;
	if (by==ymin) ymin -= 1e-10;
	if (ay==ymax) ymax += 1e-10;
	if (by==ymax) ymax += 1e-10;

	vertex_x[0] = &ax;
	vertex_y[0] = &ay;
	vertex_x[1] = &bx;
	vertex_y[1] = &by;
	vertex_x[2] = &cx;
	vertex_y[2] = &cy;
	ab[0] = (*vertex_x[1]) - (*vertex_x[0]);
	ab[1] = (*vertex_y[1]) - (*vertex_y[0]);
	bc[0] = (*vertex_x[2]) - (*vertex_x[1]);
	bc[1] = (*vertex_y[2]) - (*vertex_y[1]);
	triangle_area = 0.5 * (ab ^ bc);
	if (get_value_of(triangle_area) < 0) {
		vertex_x[1] = &cx;
		vertex_y[1] = &cy;
		vertex_x[2] = &bx;
		vertex_y[2] = &by;
		triangle_area = -triangle_area;
	}

	double xmin_doub, xmax_doub, ymin_doub, ymax_doub;
	xmin_doub = get_value_of(xmin);
	xmax_doub = get_value_of(xmax);
	ymin_doub = get_value_of(ymin);
	ymax_doub = get_value_of(ymax);

	for (i=0; i < 3; i++) {
		for (j=0; j < 4; j++) {
			in_side_region[i][j] = false;
		}
	}
	n_overlap_pts = 0;
	n_rectangle_overlap_pts = 0;

	for (i=0; i < 3; i++) {
		outside_rectangle = false;
		if (get_value_of(*vertex_x[i]) < xmin_doub) {
			in_side_region[i][3] = true;
			outside_rectangle = true;
		} else if (get_value_of(*vertex_x[i]) > xmax_doub) {
			in_side_region[i][1] = true;
			outside_rectangle = true;
		}
		if (get_value_of(*vertex_y[i]) < ymin_doub) {
			in_side_region[i][0] = true;
			outside_rectangle = true;
		} else if (get_value_of(*vertex_y[i]) > ymax_doub) {
			in_side_region[i][2] = true;
			outside_rectangle = true;
		}
		if (outside_rectangle==false) {
			overlap_pts[n_overlap_pts][0] = (*vertex_x[i]);
			overlap_pts[n_overlap_pts][1] = (*vertex_y[i]);
			n_overlap_pts++;
		}
	}
	if (n_overlap_pts==3) return triangle_area;
	if (n_overlap_pts==0) {
		for (j=0; j < 4; j++) {
			if ((in_side_region[0][j]) and (in_side_region[1][j]) and (in_side_region[2][j])) return 0; // all on same side of rectangle, so there's no overlap
		}
	}
	if (test_if_inside(xmin_doub,ymin_doub)) {
		overlap_pts[n_overlap_pts][0] = xmin;
		overlap_pts[n_overlap_pts][1] = ymin;
		n_rectangle_overlap_pts++;
		n_overlap_pts++;
	}
	if (test_if_inside(xmax_doub,ymin_doub)) {
		overlap_pts[n_overlap_pts][0] = xmax;
		overlap_pts[n_overlap_pts][1] = ymin;
		n_rectangle_overlap_pts++;
		n_overlap_pts++;
	}
	if (test_if_inside(xmin_doub,ymax_doub)) {
		overlap_pts[n_overlap_pts][0] = xmin;
		overlap_pts[n_overlap_pts][1] = ymax;
		n_rectangle_overlap_pts++;
		n_overlap_pts++;
	}
	if (test_if_inside(xmax_doub,ymax_doub)) {
		overlap_pts[n_overlap_pts][0] = xmax;
		overlap_pts[n_overlap_pts][1] = ymax;
		n_rectangle_overlap_pts++;
		n_overlap_pts++;
	}
	if (n_rectangle_overlap_pts==4) return ((xmax-xmin)*(ymax-ymin)); // the rectangle is completely inside the triangle
	int i_xmax, i_xmin, i_ymax, i_ymin;
	for (i=0; i < 3; i++) {
		i_plus_one = plus_one(i);
		if (get_value_of(*vertex_x[i_plus_one]) != get_value_of(*vertex_x[i])) {
			slope = ((*vertex_y[i_plus_one]) - (*vertex_y[i])) / ((*vertex_x[i_plus_one]) - (*vertex_x[i]));
			if (get_value_of((*vertex_x[i_plus_one])) > get_value_of((*vertex_x[i]))) { i_xmin = i; i_xmax = i_plus_one; }
			else { i_xmin = i_plus_one; i_xmax = i; }
			y = slope*(xmin-(*vertex_x[i])) + (*vertex_y[i]);
			if ((get_value_of(y) > ymin) and (get_value_of(y) < ymax)) {
				if (test_if_in_xrange(xmin_doub,get_value_of(y),i_xmin,i_xmax)) {
					overlap_pts[n_overlap_pts][0] = xmin;
					overlap_pts[n_overlap_pts][1] = y;
					n_overlap_pts++;
				}
			}
			y = slope*(xmax-(*vertex_x[i])) + (*vertex_y[i]);
			if ((get_value_of(y) > ymin_doub) and (get_value_of(y) < ymax_doub)) {
				if (test_if_in_xrange(xmax_doub,get_value_of(y),i_xmin,i_xmax)) {
					overlap_pts[n_overlap_pts][0] = xmax;
					overlap_pts[n_overlap_pts][1] = y;
					n_overlap_pts++;
				}
			}
			x = (ymin-(*vertex_y[i]))/slope + (*vertex_x[i]);
			if ((get_value_of(x) > xmin_doub) and (get_value_of(x) < xmax_doub)) {
				if (test_if_in_xrange(get_value_of(x),ymin_doub,i_xmin,i_xmax)) {
					overlap_pts[n_overlap_pts][0] = x;
					overlap_pts[n_overlap_pts][1] = ymin;
					n_overlap_pts++;
				}
			}
			x = (ymax-(*vertex_y[i]))/slope + (*vertex_x[i]);
			if ((get_value_of(x) > xmin_doub) and (get_value_of(x) < xmax_doub)) {
				if (test_if_in_xrange(get_value_of(x),ymax_doub,i_xmin,i_xmax)) {
					overlap_pts[n_overlap_pts][0] = x;
					overlap_pts[n_overlap_pts][1] = ymax;
					n_overlap_pts++;
				}
			}
		}
		else {
			if (get_value_of(*vertex_y[i_plus_one]) > get_value_of(*vertex_y[i])) { i_ymin = i; i_ymax = i_plus_one; }
			else { i_ymin = i_plus_one; i_ymax = i; }
			if ((get_value_of(*vertex_x[i]) > xmin_doub) and (get_value_of(*vertex_x[i]) < xmax_doub)) {
				if (test_if_in_yrange(get_value_of(*vertex_x[i]),ymin_doub,i_ymin,i_ymax)) {
					overlap_pts[n_overlap_pts][0] = (*vertex_x[i]);
					overlap_pts[n_overlap_pts][1] = ymin;
					n_overlap_pts++;
				}
				if (test_if_in_yrange(get_value_of(*vertex_x[i]),ymax_doub,i_ymin,i_ymax)) {
					overlap_pts[n_overlap_pts][0] = (*vertex_x[i]);
					overlap_pts[n_overlap_pts][1] = ymax;
					n_overlap_pts++;
				}
			}
		}
	}

	return calculate_polygon_area();
}

template <typename QScalar>
inline bool TriRectangleOverlap<QScalar>::test_if_inside(const double x, const double y)
{
	if ((dif_cross_product(x,y,vertex_x[0],vertex_y[0],vertex_x[1],vertex_y[1]) > 0) and (dif_cross_product(x,y,vertex_x[1],vertex_y[1],vertex_x[2],vertex_y[2]) > 0) and (dif_cross_product(x,y,vertex_x[2],vertex_y[2],vertex_x[0],vertex_y[0]) > 0)) return true;
	else return false;
}

template <typename QScalar>
inline bool TriRectangleOverlap<QScalar>::test_if_in_xrange(const double x, const double y, const int i_xmin, const int i_xmax)
{
	if ((x >= (*vertex_x[i_xmin])) and (x <= (*vertex_x[i_xmax]))) return true;
	else return false;
}

template <typename QScalar>
inline bool TriRectangleOverlap<QScalar>::test_if_in_yrange(const double x, const double y, const int i_ymin, const int i_ymax)
{
	if ((y >= (*vertex_y[i_ymin])) and (y <= (*vertex_y[i_ymax]))) return true;
	else return false;
}


template <typename QScalar>
inline double TriRectangleOverlap<QScalar>::dif_cross_product(const double x, const double y, const QScalar* A_x, const QScalar* A_y, const QScalar* B_x, const QScalar* B_y)
{
	return ((get_value_of(*A_x)-x)*(get_value_of(*B_y)-y) - (get_value_of(*A_y)-y)*(get_value_of(*B_x)-x));
}

template <typename QScalar>
QScalar TriRectangleOverlap<QScalar>::calculate_polygon_area(void)
{
#ifdef USE_STAN
	using stan::math::abs;
#endif
	overlap_area = 0;
	overlap_pts_list = new lensvector<QScalar>*[n_overlap_pts];
	for (i=0; i < n_overlap_pts; i++) overlap_pts_list[i] = &overlap_pts[i];
	bool duplicate_pts;
	int k;
	do {
		// Duplicate points really should not occur, but very rarely they do...and I don't have time to figure out why, so here they will just be removed when they occur
		duplicate_pts = false;
		for (i=0; i < n_overlap_pts; i++) {
			for (j=i+1; j < n_overlap_pts; j++) {
				if ((get_value_of(overlap_pts[i][0]) == get_value_of(overlap_pts[j][0])) and (get_value_of(overlap_pts[i][1]) == get_value_of(overlap_pts[j][1]))) {
					k = j;
					duplicate_pts = true;
				}
			}
		}
		if (duplicate_pts) {
			newlist = new lensvector<QScalar>*[n_overlap_pts-1];
			for (i=0; i < k; i++) {
				newlist[i] = overlap_pts_list[i];
			}
			for (i=k+1; i < n_overlap_pts; i++) {
				newlist[i-1] = overlap_pts_list[i];
			}
			delete[] overlap_pts_list;
			overlap_pts_list = newlist;
			n_overlap_pts--;
		}
	} while (duplicate_pts);

	//for (i=0; i < n_overlap_pts; i++) {
		//for (j=i+1; j < n_overlap_pts; j++) {
			//if ((get_value_of(overlap_pts[i][0]) == get_value_of(overlap_pts[j][0])) and (get_value_of(overlap_pts[i][1]) == get_value_of(overlap_pts[j][1]))) {
				//std::cout << "Duplicate points (THIS SHOULD NOT HAPPEN!)" << i << " and " << j << ": (" << get_value_of(overlap_pts[i][0]) << "," << get_value_of(overlap_pts[i][1]) << ") and (" << get_value_of(overlap_pts[j][0]) << "," << get_value_of(overlap_pts[j][1]) << ")\n";
			//}
		//}
	//}

	while (n_overlap_pts > 2)
	{
		swapped1 = false;
		swapped2 = false;
		(*r01)[0] = (*overlap_pts_list[1])[0] - (*overlap_pts_list[0])[0];
		(*r01)[1] = (*overlap_pts_list[1])[1] - (*overlap_pts_list[0])[1];
		(*r02)[0] = (*overlap_pts_list[2])[0] - (*overlap_pts_list[0])[0];
		(*r02)[1] = (*overlap_pts_list[2])[1] - (*overlap_pts_list[0])[1];
		subtriangle_area = (*r01) ^ (*r02);
#ifdef USE_STAN
		if constexpr (stan::is_autodiff_v<QScalar>) {
			if (n_overlap_pts==3) { overlap_area += stan::math::abs(subtriangle_area); break; }
		} else
#endif
		if (n_overlap_pts==3) { overlap_area += abs(subtriangle_area); break; }
		if (subtriangle_area < 0) {
			swap_vectors(overlap_pts_list[1],overlap_pts_list[2]);
			swap_vectors(r01,r02);
			subtriangle_area = -subtriangle_area;
		}
		for (i=3; i < n_overlap_pts; i++) {
			aa[0] = (*overlap_pts_list[0])[0] - (*overlap_pts_list[i])[0];
			aa[1] = (*overlap_pts_list[0])[1] - (*overlap_pts_list[i])[1];
			bb[0] = (*overlap_pts_list[1])[0] - (*overlap_pts_list[i])[0];
			bb[1] = (*overlap_pts_list[1])[1] - (*overlap_pts_list[i])[1];
			if (get_value_of(aa ^ bb) < 0) { swap_vectors(overlap_pts_list[1],overlap_pts_list[i]); swapped1 = true; continue; }
			bb[0] = (*overlap_pts_list[2])[0] - (*overlap_pts_list[i])[0];
			bb[1] = (*overlap_pts_list[2])[1] - (*overlap_pts_list[i])[1];
			if (get_value_of(aa ^ bb) > 0) { swap_vectors(overlap_pts_list[2],overlap_pts_list[i]); swapped2 = true; continue; }
		}
		if (swapped1) {
			(*r01)[0] = (*overlap_pts_list[1])[0] - (*overlap_pts_list[0])[0];
			(*r01)[1] = (*overlap_pts_list[1])[1] - (*overlap_pts_list[0])[1];
			if (swapped2) {
				(*r02)[0] = (*overlap_pts_list[2])[0] - (*overlap_pts_list[0])[0];
				(*r02)[1] = (*overlap_pts_list[2])[1] - (*overlap_pts_list[0])[1];
			}
			subtriangle_area = (*r01) ^ (*r02);
		} else if (swapped2) {
			(*r02)[0] = (*overlap_pts_list[2])[0] - (*overlap_pts_list[0])[0];
			(*r02)[1] = (*overlap_pts_list[2])[1] - (*overlap_pts_list[0])[1];
			subtriangle_area = (*r01) ^ (*r02);
			if (subtriangle_area < 0) die("shouldn't have the wrong orientation!"); // remove this later after testing
		}
#ifdef USE_STAN
		if constexpr (stan::is_autodiff_v<QScalar>) {
			overlap_area += stan::math::abs(subtriangle_area);
		} else
#endif
		overlap_area += abs(subtriangle_area);
		newlist = new lensvector<QScalar>*[n_overlap_pts-1];
		for (i=0; i < n_overlap_pts-1; i++) {
			newlist[i] = overlap_pts_list[i+1];
		}
		delete[] overlap_pts_list;
		overlap_pts_list = newlist;
		n_overlap_pts--;
	}

	delete[] overlap_pts_list;
	return 0.5*overlap_area;
}

template <typename QScalar>
bool TriRectangleOverlap<QScalar>::determine_if_in_neighborhood(const QScalar& ax, const QScalar& ay, const QScalar& bx, const QScalar& by, const QScalar& cx, const QScalar& cy, const QScalar& dx, const QScalar& dy, const double xmin, const double xmax, const double ymin, const double ymax, bool &inside)
{
	inside = false;
	vertex_x[0] = &ax;
	vertex_y[0] = &ay;
	vertex_x[1] = &bx;
	vertex_y[1] = &by;
	vertex_x[2] = &cx;
	vertex_y[2] = &cy;
	vertex_x[3] = &dx;
	vertex_y[3] = &dy;

	for (i=0; i < 4; i++) {
		for (j=0; j < 4; j++) {
			in_side_region[i][j] = false;
		}
	}

	for (i=0; i < 4; i++) {
		outside_rectangle = false;
		if (get_value_of(*vertex_x[i]) < xmin) {
			in_side_region[i][3] = true;
			if (outside_rectangle==false) outside_rectangle = true;
		} else if (get_value_of(*vertex_x[i]) > xmax) {
			in_side_region[i][1] = true;
			if (outside_rectangle==false) outside_rectangle = true;
		}
		if (get_value_of(*vertex_y[i]) < ymin) {
			in_side_region[i][0] = true;
			if (outside_rectangle==false) outside_rectangle = true;
		} else if (get_value_of(*vertex_y)[1] > ymax) {
			in_side_region[i][2] = true;
			if (outside_rectangle==false) outside_rectangle = true;
		}
		if (outside_rectangle==false) { inside = true; return true; }
	}
	for (j=0; j < 4; j++) {
		if ((in_side_region[0][j]) and (in_side_region[1][j]) and (in_side_region[2][j]) and (in_side_region[3][j])) return false; // all on same side of rectangle, so there's no overlap
	}
	return true;
}


template <typename QScalar>
bool TriRectangleOverlap<QScalar>::determine_if_overlap(const QScalar& ax, const QScalar& ay, const QScalar& bx, const QScalar& by, const QScalar& cx, const QScalar& cy, const double xmin, const double xmax, const double ymin, const double ymax)
{
	// The assignment of the rectangle corner and side indices will be as follows:
	// 3-2-2
	// 3   1
	// 0-0-1

	vertex_x[0] = &ax;
	vertex_y[0] = &ay;
	vertex_x[1] = &bx;
	vertex_y[1] = &by;
	vertex_x[2] = &cx;
	vertex_y[2] = &cy;

	// Here we assume that the 'determine_if_in_neighborhood' function has already been run
	ab[0] = (*vertex_x[1]) - (*vertex_x[0]);
	ab[1] = (*vertex_y[1]) - (*vertex_y[0]);
	bc[0] = (*vertex_x[2]) - (*vertex_x[1]);
	bc[1] = (*vertex_y[2]) - (*vertex_y[1]);
	if ((ab ^ bc) < 0) {
		vertex_x[1] = &cx;
		vertex_y[1] = &cy;
		vertex_x[2] = &bx; 
		vertex_y[2] = &by; 
	}

	if (test_if_inside(xmin,ymin)) return true;
	if (test_if_inside(xmax,ymin)) return true;
	if (test_if_inside(xmin,ymax)) return true;
	if (test_if_inside(xmax,ymax)) return true;
	int i_xmax, i_xmin, i_ymax, i_ymin;
	for (i=0; i < 3; i++) {
		i_plus_one = plus_one(i);
		if ((*vertex_x[i_plus_one]) != (*vertex_x[i])) {
			slope = ((*vertex_y[i_plus_one]) - (*vertex_y[i])) / ((*vertex_x[i_plus_one]) - (*vertex_x[i]));
			if ((*vertex_x[i_plus_one]) > (*vertex_x[i])) { i_xmin = i; i_xmax = i_plus_one; }
			else { i_xmin = i_plus_one; i_xmax = i; }
			y = slope*(xmin-(*vertex_x[i])) + (*vertex_y[i]);
			if ((y > ymin) and (y < ymax)) {
				if (test_if_in_xrange(xmin,get_value_of(y),i_xmin,i_xmax)) return true;
			}
			y = slope*(xmax-(*vertex_x[i])) + (*vertex_y[i]);
			if ((get_value_of(y) > ymin) and (get_value_of(y) < ymax)) {
				if (test_if_in_xrange(xmax,get_value_of(y),i_xmin,i_xmax)) return true;
			}
			x = (ymin-(*vertex_y[i]))/slope + (*vertex_y[i]);
			if ((get_value_of(x) > xmin) and (get_value_of(x) < xmax)) {
				if (test_if_in_xrange(get_value_of(x),ymin,i_xmin,i_xmax)) return true;
			}
			x = (ymax-(*vertex_y[i]))/slope + (*vertex_x[i]);
			if ((get_value_of(x) > xmin) and (get_value_of(x) < xmax)) {
				if (test_if_in_xrange(get_value_of(x),ymax,i_xmin,i_xmax)) return true;
			}
		}
		else {
			if (get_value_of(*vertex_y[i_plus_one]) > get_value_of(*vertex_y[i])) { i_ymin = i; i_ymax = i_plus_one; }
			else { i_ymin = i_plus_one; i_ymax = i; }
			if ((get_value_of(*vertex_x[i]) > xmin) and (get_value_of(*vertex_x[i]) < xmax)) {
				if (test_if_in_yrange(get_value_of(*vertex_x[i]),ymin,i_ymin,i_ymax)) return true;
				if (test_if_in_yrange(get_value_of(*vertex_x[i]),ymax,i_ymin,i_ymax)) return true;
			}
		}
	}
	return false;
}

template <typename QScalar>
bool TriRectangleOverlap<QScalar>::determine_if_in_neighborhood(const QScalar& ax, const QScalar& ay, const QScalar& bx, const QScalar& by, const QScalar& cx, const QScalar& cy, const double xmin, const double xmax, const double ymin, const double ymax, bool &inside)
{
	inside = false;
	vertex_x[0] = &ax;
	vertex_y[0] = &ay;
	vertex_x[1] = &bx;
	vertex_y[1] = &by;
	vertex_x[2] = &cx; 
	vertex_y[2] = &cy; 

	for (i=0; i < 3; i++) {
		for (j=0; j < 4; j++) {
			in_side_region[i][j] = false;
		}
	}

	for (i=0; i < 3; i++) {
		outside_rectangle = false;
		if (get_value_of(*vertex_x[i]) < xmin) {
			in_side_region[i][3] = true;
			if (outside_rectangle==false) outside_rectangle = true;
		} else if (get_value_of(*vertex_x[i]) > xmax) {
			in_side_region[i][1] = true;
			if (outside_rectangle==false) outside_rectangle = true;
		}
		if (get_value_of(*vertex_y[i]) < ymin) {
			in_side_region[i][0] = true;
			if (outside_rectangle==false) outside_rectangle = true;
		} else if (get_value_of(*vertex_y[i]) > ymax) {
			in_side_region[i][2] = true;
			if (outside_rectangle==false) outside_rectangle = true;
		}
		if (outside_rectangle==false) { inside = true; return true; }
	}
	for (j=0; j < 4; j++) {
		if ((in_side_region[0][j]) and (in_side_region[1][j]) and (in_side_region[2][j])) return false; // all on same side of rectangle, so there's no overlap
	}
	return true;
}

template <typename QScalar>
TriRectangleOverlap<QScalar>::~TriRectangleOverlap()
{
	delete r01;
	delete r02;
}

#endif // TRIRECTANGLE_H
