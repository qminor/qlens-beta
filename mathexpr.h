#ifndef MATHEXPR_H
#define MATHEXPR_H

#include <cmath>

#define EULER 2.71828182846
#define ln10 2.30258509299

const double M_2PI = 6.28318530718;
const double M_HALFPI = 1.57079632679;
const double M_SQRT_PI = 1.77245385091;

inline double dmin(const double &a, const double &b) { return (a < b ? a : b); }
inline double dmax(const double &a, const double &b) { return (a > b ? a : b); }
inline int imin(const int &a, const int &b) { return (a < b ? a : b); }
inline int imax(const int &a, const int &b) { return (a > b ? a : b); }
inline double SQR(const double s) { return s*s; }
inline double CUBE(const double s) { return s*s*s; }
inline double QUARTIC(const double s) { return s*s*s*s; }
inline int sign(const double &a) { return (a < 0 ? -1 : a > 0 ? 1 : 0); }
inline bool sign_bool(const double &a) { return (a < 0 ? false : a > 0 ? true : true); }
inline double norm(const double a, const double b) { return sqrt(a*a+b*b); }
inline double degrees_to_radians(const double theta) { return (0.0174532925199*theta); }
inline double radians_to_degrees(const double theta) { return (57.2957795131*theta); }

inline double angle(double a, double b) {
	double c = (a==0.0) ? 0.5*M_PI
		: (a > 0.0) ? atan(b/a) : M_PI+atan(b/a);
	if (c < 0) c = 2*M_PI + c;
	return c;
}

#endif // MATHEXPR_H
