#ifndef MATHEXPR_H
#define MATHEXPR_H

#include <cmath>

//#define EULER 2.71828182846
//#define ln10 2.30258509299

const double M_2PI = 6.283185307179586;
const double M_HALFPI = 1.5707963267948966;
const double M_SQRT_PI = 1.7724538509055159;
const double M_SQRT_2PI = 2.5066282746310005;
const double M_SQRT_HALFPI = 1.2533141373155001;
const double M_4PI = 12.566370614359172;
const double ln10 = 2.302585092994046;
const double EULER = 2.718281828459045;

template <typename T>
inline T minval(const T &a, const T &b) { return (a < b ? a : b); }
template <typename T>
inline T maxval(const T &a, const T &b) { return (a > b ? a : b); }

inline double dmin(const double &a, const double &b) { return (a < b ? a : b); }
inline double dmax(const double &a, const double &b) { return (a > b ? a : b); }
inline int imin(const int &a, const int &b) { return (a < b ? a : b); }
inline int imax(const int &a, const int &b) { return (a > b ? a : b); }
template <typename T>
inline T SQR(const T s) { return s*s; }
template <typename T>
T CUBE(const T s) { return s*s*s; }
template <typename T>
T QUARTIC(const T s) { return s*s*s*s; }

inline int sign(const double &a) { return (a < 0 ? -1 : a > 0 ? 1 : 0); }
inline bool sign_bool(const double &a) { return (a < 0 ? false : a > 0 ? true : true); }
inline double norm(const double a, const double b) { return std::sqrt(a*a+b*b); }
template <typename T>
inline T degrees_to_radians(const T theta) { return (0.017453292519943295*theta); }
template <typename T>
inline T radians_to_degrees(const T theta) { return (57.29577951308232*theta); }

inline double get_angle(double a, double b) {
	double angle = std::atan(std::abs(b/a));
	if (a < 0) {
		if (b < 0)
			angle = angle - M_PI;
		else
			angle = M_PI - angle;
	} else if (b < 0) {
		angle = -angle;
	}
	return angle;

}

#endif // MATHEXPR_H
