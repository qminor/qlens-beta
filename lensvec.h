#ifndef LENSVEC_H
#define LENSVEC_H

#include <cmath>
#include "errors.h"

class lensvector
{
	double *v;

public:
	lensvector() : v(new double[2]) {}
	lensvector(const double z) : v(new double[2]) { v[0] = v[1] = z; }
	lensvector(const double &x, const double &y) : v(new double[2]) { v[0] = x; v[1] = y; }
	lensvector(const lensvector& b) : v(new double[2]) { v[0] = b[0]; v[1] = b[1]; }
	void input(const double &x, const double &y) { v[0] = x; v[1] = y; }
	~lensvector() { delete[] v; }

	lensvector& operator = (const lensvector& b) { v[0] = b[0]; v[1] = b[1]; return *this; }
	lensvector& operator = (const double b) { v[0] = b; v[1] = b; return *this; }
	double& operator [] (const int n) { return v[n]; }
	double& operator [] (const int n) const { return v[n]; }

	lensvector operator + (const lensvector& b) {
		lensvector ans;
		ans[0] = v[0] + b[0];
		ans[1] = v[1] + b[1];
		return ans;
	}
	lensvector operator - (const lensvector& b) {
		lensvector ans;
		ans[0] = v[0] - b[0];
		ans[1] = v[1] - b[1];
		return ans;
	}
	lensvector& operator += (const lensvector& b) {
		v[0] += b[0];
		v[1] += b[1];
		return *this;
	}
	lensvector operator + (const lensvector& b) const {
		lensvector ans;
		ans[0] = v[0] + b[0];
		ans[1] = v[1] + b[1];
		return ans;
	}
	lensvector operator - (const lensvector& b) const {
		lensvector ans;
		ans[0] = v[0] - b[0];
		ans[1] = v[1] - b[1];
		return ans;
	}
	lensvector operator - (void) {
		lensvector ans;
		ans[0] = -v[0];
		ans[1] = -v[1];
		return ans;
	}
	lensvector& operator -= (const lensvector& b) {
		v[0] -= b[0];
		v[1] -= b[1];
		return *this;
	}
	lensvector& operator *= (const double num) {
		v[0] *= num;
		v[1] *= num;
		return *this;
	}
	double operator * (const lensvector& b) { return (v[0]*b[0] + v[1]*b[1]); }
	double operator ^ (const lensvector& b) { return (v[0]*b[1] - v[1]*b[0]); }
	double norm(void) { return sqrt(v[0]*v[0]+v[1]*v[1]); }
	double sqrnorm(void) { return (v[0]*v[0]+v[1]*v[1]); }
	double angle(void) { return atan(v[1]/v[0]); }
	void rotate(const double theta) {
		double cs=cos(theta), ss=sin(theta);
		double x_prime = v[0]*cs + v[1]*ss;
		v[1] = -v[0]*ss + v[1]*cs;
		v[0] = x_prime;
		return;
	}
	// counter-clockwise rotation
	void rotate(const double cs, const double ss) {
		double x_prime = v[0]*cs + v[1]*ss;
		v[1] = -v[0]*ss + v[1]*cs;
		v[0] = x_prime;
		return;
	}
	// clockwise rotation
	void rotate_back(const double cs, const double ss) {
		double x_prime = v[0]*cs - v[1]*ss;
		v[1] = v[0]*ss + v[1]*cs;
		v[0] = x_prime;
		return;
	}
};

// NOTE: the following operations cost a bit of overhead since an intermediate object (ans) is created; for better
// (though less pretty) efficiency, do the operations on each component explicitly rather than calling these functions.
inline lensvector operator + (const double a, const lensvector b)
{
	lensvector ans;
	ans[0] = a + b[0];
	ans[1] = a + b[1];
	return ans;
}

inline lensvector operator - (const double a, const lensvector b)
{
	lensvector ans;
	ans[0] = a - b[0];
	ans[1] = a - b[1];
	return ans;
}

inline lensvector operator * (const double num, const lensvector a)
{
	lensvector ans;
	ans[0] = num * a[0];
	ans[1] = num * a[1];
	return ans;
}

inline lensvector operator / (const lensvector a, const double num)
{
	lensvector ans;
	ans[0] = a[0] / num;
	ans[1] = a[1] / num;
	return ans;
}

class lensmatrix
{
	double **j;

public:
	lensmatrix() : j(new double*[2]) { j[0] = new double[2]; j[1] = new double[2]; }
	~lensmatrix() { delete[] j[0]; delete[] j[1]; delete[] j; }
	lensmatrix(const double z) : j(new double*[2]) {
		j[0] = new double[2];
		j[1] = new double[2];
		j[0][0] = j[1][1] = z;
		j[1][0] = j[0][1] = 0;
	}
	lensmatrix(const lensmatrix& b) : j(new double*[2]) {
		j[0] = new double[2];
		j[1] = new double[2];
		j[0][0] = b[0][0]; j[0][1] = b[0][1];
		j[1][0] = b[1][0]; j[1][1] = b[1][1];
		return;
	}

	lensmatrix& operator = (const lensmatrix& b) {
		j[0][0] = b[0][0]; j[0][1] = b[0][1];
		j[1][0] = b[1][0]; j[1][1] = b[1][1];
		return *this;
	}
	lensmatrix& operator = (const double b) {
		j[0][0] = b; j[0][1] = b;
		j[1][0] = b; j[1][1] = b;
		return *this;
	}
	double* operator [] (const int n) { return j[n]; }
	double* operator [] (const int n) const { return j[n]; }

	lensmatrix& operator += (const lensmatrix& b) {
		j[0][0] += b[0][0]; j[1][0] += b[1][0];
		j[0][1] += b[0][1]; j[1][1] += b[1][1];
		return *this;
	}
	lensmatrix operator + (const lensmatrix& b) {
		lensmatrix ans;
		ans[0][0] = j[0][0] + b[0][0]; ans[1][0] = j[1][0] + b[1][0];
		ans[0][1] = j[0][1] + b[0][1]; ans[1][1] = j[1][1] + b[1][1];
		return ans;
	}
	lensmatrix operator + (const double z) {
		lensmatrix ans;
		ans[0][0] = j[0][0] + z;
		ans[1][1] = j[1][1] + z;
		ans[1][0] = j[1][0];
		ans[0][1] = j[0][1];
		return ans;
	}
	lensmatrix operator - (const double z) {
		lensmatrix ans;
		ans[0][0] = j[0][0] - z;
		ans[1][1] = j[1][1] - z;
		return ans;
	}
	lensmatrix& operator -= (const lensmatrix& b) {
		j[0][0] -= b[0][0]; j[1][0] -= b[1][0];
		j[0][1] -= b[0][1]; j[1][1] -= b[1][1];
		return *this;
	}
	lensmatrix operator - (const lensmatrix& b) {
		lensmatrix ans;
		ans[0][0] = j[0][0] - b[0][0]; ans[1][0] = j[1][0] - b[1][0];
		ans[0][1] = j[0][1] - b[0][1]; ans[1][1] = j[1][1] - b[1][1];
		return ans;
	}
	lensmatrix operator - (void) {
		lensmatrix ans;
		ans[0][0] = -j[0][0]; ans[1][0] = -j[1][0];
		ans[0][1] = -j[0][1]; ans[1][1] = -j[1][1];
		return ans;
	}
	lensvector operator * (const lensvector& b) {
		lensvector ans;
		ans[0] = j[0][0] * b[0] + j[1][0] * b[1];
		ans[1] = j[0][1] * b[0] + j[1][1] * b[1];
		return ans;
	}

	void rotate(const double theta)
	{
		double x_prime, cs, ss;
		cs=cos(theta); ss=sin(theta);

		// Similarity transformation: J' = R*J*R^(-1) (where R = rotation matrix, J = jacobian)
		x_prime = j[0][0]*cs + j[0][1]*ss;
		j[0][1] = -j[0][0]*ss + j[0][1]*cs;
		j[0][0] = x_prime;

		x_prime = j[1][0]*cs + j[1][1]*ss;
		j[1][1] = -j[1][0]*ss + j[1][1]*cs;
		j[1][0] = x_prime;

		x_prime = j[0][0]*cs + j[1][0]*ss;
		j[1][0] = -j[0][0]*ss + j[1][0]*cs;
		j[0][0] = x_prime;

		x_prime = j[0][1]*cs + j[1][1]*ss;
		j[1][1] = -j[0][1]*ss + j[1][1]*cs;
		j[0][1] = x_prime;

		return;
	}
	void rotate(const double cs, const double ss)
	{
		double x_prime;

		// Similarity transformation: J' = R*J*R^(-1) (where R = rotation matrix, J = jacobian)
		x_prime = j[0][0]*cs + j[0][1]*ss;
		j[0][1] = -j[0][0]*ss + j[0][1]*cs;
		j[0][0] = x_prime;

		x_prime = j[1][0]*cs + j[1][1]*ss;
		j[1][1] = -j[1][0]*ss + j[1][1]*cs;
		j[1][0] = x_prime;

		x_prime = j[0][0]*cs + j[1][0]*ss;
		j[1][0] = -j[0][0]*ss + j[1][0]*cs;
		j[0][0] = x_prime;

		x_prime = j[0][1]*cs + j[1][1]*ss;
		j[1][1] = -j[0][1]*ss + j[1][1]*cs;
		j[0][1] = x_prime;

		return;
	}
	void rotate_back(const double cs, const double ss)
	{
		double x_prime;
		if (cs==0) {
			if (ss==1) {
				x_prime = j[1][1];
				j[1][1] = j[0][0];
				j[0][0] = x_prime;
				x_prime = j[1][0];
				j[1][0] = -j[0][1];
				j[0][1] = -x_prime;
			} else if (ss==-1) {
				x_prime = j[1][1];
				j[1][1] = -j[0][0];
				j[0][0] = -x_prime;
				x_prime = j[1][0];
				j[1][0] = j[0][1];
				j[0][1] = x_prime;
			} else die("sin(theta) value not consistent with cos(theta) = 0");
			return;
		}

		// Similarity transformation: J' = R*J*R^(-1) (where R = rotation matrix, J = jacobian)
		x_prime = j[0][0]*cs - j[0][1]*ss;
		j[0][1] = j[0][0]*ss + j[0][1]*cs;
		j[0][0] = x_prime;

		x_prime = j[1][0]*cs - j[1][1]*ss;
		j[1][1] = j[1][0]*ss + j[1][1]*cs;
		j[1][0] = x_prime;

		x_prime = j[0][0]*cs - j[1][0]*ss;
		j[1][0] = j[0][0]*ss + j[1][0]*cs;
		j[0][0] = x_prime;

		x_prime = j[0][1]*cs - j[1][1]*ss;
		j[1][1] = j[0][1]*ss + j[1][1]*cs;
		j[0][1] = x_prime;

		return;
	}
	lensmatrix inverse(void)
	{
		lensmatrix ans;
		double det = j[0][0]*j[1][1] - j[0][1]*j[1][0];
		if (det==0.0) die("singular matrix--cannot invert Jacobian matrix");
		ans[0][0] = j[1][1]/det;
		ans[1][0] = -j[1][0]/det;
		ans[0][1] = -j[0][1]/det;
		ans[1][1] = j[0][0]/det;
		return ans;
	}
	bool invert(lensmatrix& ans)
	{
		double det = j[0][0]*j[1][1] - j[0][1]*j[1][0];
		if (det==0.0) { warn("singular matrix--cannot invert Jacobian matrix"); return false; }
		ans[0][0] = j[1][1]/det;
		ans[1][0] = -j[1][0]/det;
		ans[0][1] = -j[0][1]/det;
		ans[1][1] = j[0][0]/det;
		return true;
	}

};

inline double determinant(const lensmatrix b) { return (b[0][0]*b[1][1] - b[1][0]*b[0][1]); }
inline lensmatrix operator + (const double a, const lensmatrix b)
{
	lensmatrix ans;
	ans[0][0] = a + b[0][0];
	ans[1][1] = a + b[1][1];
	ans[1][0] = b[1][0];
	ans[0][1] = b[0][1];
	return ans;
}
inline lensmatrix operator - (const double a, const lensmatrix b)
{
	lensmatrix ans;
	ans[0][0] = a - b[0][0];
	ans[1][1] = a - b[1][1];
	ans[1][0] = -b[1][0];
	ans[0][1] = -b[0][1];
	return ans;
}
inline void lensmatsqr(const lensmatrix a, lensmatrix& b)
{
	b[0][0] = a[0][0]*a[0][0] + a[0][1]*a[0][1];
	b[0][1] = a[0][0]*a[1][0] + a[0][1]*a[1][1];
	b[1][0] = b[0][1];
	b[1][1] = a[1][1]*a[1][1] + a[1][0]*a[1][0];
}
inline lensmatrix operator * (const double num, const lensmatrix a)
{
	lensmatrix ans;
	ans[0][0] = num * a[0][0];
	ans[1][1] = num * a[1][1];
	ans[1][0] = num * a[1][0];
	ans[0][1] = num * a[0][1];
	return ans;
}

inline lensmatrix operator * (const lensmatrix a, const lensmatrix b)
{
	lensmatrix ans;
	ans[0][0] = a[0][0]*b[0][0] + a[1][0]*b[0][1];
	ans[1][0] = a[0][0]*b[1][0] + a[1][0]*b[1][1];
	ans[0][1] = a[0][1]*b[0][0] + a[1][1]*b[0][1];
	ans[1][1] = a[0][1]*b[1][0] + a[1][1]*b[1][1];
	return ans;
}

#endif // LENSVEC_H
