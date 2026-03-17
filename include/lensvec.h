#ifndef LENSVEC_H
#define LENSVEC_H

#include <cmath>
#include "errors.h"

template <typename QScalar>
class lensvector
{
public:
	QScalar *v;
	lensvector() : v(new QScalar[2]) {}
	lensvector(const QScalar z) : v(new QScalar[2]) { v[0] = v[1] = z; }
	lensvector(const QScalar &x, const QScalar &y) : v(new QScalar[2]) { v[0] = x; v[1] = y; }
	lensvector(const lensvector<QScalar>& b) : v(new QScalar[2]) { v[0] = b[0]; v[1] = b[1]; }
	void input(const QScalar &x, const QScalar &y) { v[0] = x; v[1] = y; }
	QScalar* array(void) { return v; }
	QScalar* array(void) const { return v; }
	QScalar& xval() { return v[0]; }
	QScalar& yval() { return v[1]; }
	void set_xval(const QScalar xv) { v[0] = xv; }
	void set_yval(const QScalar yv) { v[1] = yv; }
	~lensvector() { delete[] v; }

	lensvector<QScalar>& operator = (const lensvector<QScalar>& b) { v[0] = b[0]; v[1] = b[1]; return *this; }
	lensvector<QScalar>& operator = (const QScalar b) { v[0] = b; v[1] = b; return *this; }
	QScalar& operator [] (const int n) { return v[n]; }
	QScalar& operator [] (const int n) const { return v[n]; }

	lensvector<QScalar> operator + (const lensvector<QScalar>& b) {
		lensvector<QScalar> ans;
		ans[0] = v[0] + b[0];
		ans[1] = v[1] + b[1];
		return ans;
	}
	lensvector<QScalar> operator - (const lensvector<QScalar>& b) {
		lensvector<QScalar> ans;
		ans[0] = v[0] - b[0];
		ans[1] = v[1] - b[1];
		return ans;
	}
	lensvector<QScalar>& operator += (const lensvector<QScalar>& b) {
		v[0] += b[0];
		v[1] += b[1];
		return *this;
	}
	lensvector<QScalar> operator + (const lensvector<QScalar>& b) const {
		lensvector<QScalar> ans;
		ans[0] = v[0] + b[0];
		ans[1] = v[1] + b[1];
		return ans;
	}
	lensvector<QScalar> operator - (const lensvector<QScalar>& b) const {
		lensvector<QScalar> ans;
		ans[0] = v[0] - b[0];
		ans[1] = v[1] - b[1];
		return ans;
	}
	lensvector<QScalar> operator - (void) {
		lensvector<QScalar> ans;
		ans[0] = -v[0];
		ans[1] = -v[1];
		return ans;
	}
	lensvector<QScalar>& operator -= (const lensvector<QScalar>& b) {
		v[0] -= b[0];
		v[1] -= b[1];
		return *this;
	}
	lensvector<QScalar>& operator *= (const double num) {
		v[0] *= num;
		v[1] *= num;
		return *this;
	}
	lensvector<QScalar>& operator /= (const double num) {
		v[0] /= num;
		v[1] /= num;
		return *this;
	}
	QScalar operator * (const lensvector<QScalar>& b) { return (v[0]*b[0] + v[1]*b[1]); }
	QScalar operator ^ (const lensvector<QScalar>& b) { return (v[0]*b[1] - v[1]*b[0]); }
	QScalar norm(void) { return sqrt(v[0]*v[0]+v[1]*v[1]); }
	QScalar sqrnorm(void) { return (v[0]*v[0]+v[1]*v[1]); }
	QScalar angle(void) { return atan(v[1]/v[0]); }
	void rotate(const QScalar theta) {
		QScalar cs=cos(theta), ss=sin(theta);
		QScalar x_prime = v[0]*cs + v[1]*ss;
		v[1] = -v[0]*ss + v[1]*cs;
		v[0] = x_prime;
		return;
	}
	// counter-clockwise rotation
	void rotate(const QScalar cs, const QScalar ss) {
		QScalar x_prime = v[0]*cs + v[1]*ss;
		v[1] = -v[0]*ss + v[1]*cs;
		v[0] = x_prime;
		return;
	}
	// clockwise rotation
	void rotate_back(const QScalar cs, const QScalar ss) {
		QScalar x_prime = v[0]*cs - v[1]*ss;
		v[1] = v[0]*ss + v[1]*cs;
		v[0] = x_prime;
		return;
	}
};

// NOTE: the following operations cost a bit of overhead since an intermediate object (ans) is created; for better
// (though less pretty) efficiency, do the operations on each component explicitly rather than calling these functions.
template <typename QScalar>
lensvector<QScalar> operator + (const QScalar a, const lensvector<QScalar> b)
{
	lensvector<QScalar> ans;
	ans[0] = a + b[0];
	ans[1] = a + b[1];
	return ans;
}

template <typename QScalar>
inline lensvector<QScalar> operator - (const QScalar a, const lensvector<QScalar> b)
{
	lensvector<QScalar> ans;
	ans[0] = a - b[0];
	ans[1] = a - b[1];
	return ans;
}

template <typename QScalar>
inline lensvector<QScalar> operator * (const double num, const lensvector<QScalar> a)
{
	lensvector<QScalar> ans;
	ans[0] = num * a[0];
	ans[1] = num * a[1];
	return ans;
}

template <typename QScalar>
inline lensvector<QScalar> operator / (const lensvector<QScalar> a, const double num)
{
	lensvector<QScalar> ans;
	ans[0] = a[0] / num;
	ans[1] = a[1] / num;
	return ans;
}

template <typename QScalar>
class lensmatrix
{
	QScalar **j;

	public:
	lensmatrix() : j(new QScalar*[2]) { j[0] = new QScalar[2]; j[1] = new QScalar[2]; }
	~lensmatrix() { delete[] j[0]; delete[] j[1]; delete[] j; }
	lensmatrix(const QScalar z) : j(new QScalar*[2]) {
		j[0] = new QScalar[2];
		j[1] = new QScalar[2];
		j[0][0] = j[1][1] = z;
		j[1][0] = j[0][1] = 0;
	}
	lensmatrix(const lensmatrix<QScalar>& b) : j(new QScalar*[2]) {
		j[0] = new QScalar[2];
		j[1] = new QScalar[2];
		j[0][0] = b[0][0]; j[0][1] = b[0][1];
		j[1][0] = b[1][0]; j[1][1] = b[1][1];
		return;
	}

	lensmatrix<QScalar>& operator = (const lensmatrix<QScalar>& b) {
		j[0][0] = b[0][0]; j[0][1] = b[0][1];
		j[1][0] = b[1][0]; j[1][1] = b[1][1];
		return *this;
	}
	lensmatrix<QScalar>& operator = (const QScalar b) {
		j[0][0] = b; j[0][1] = b;
		j[1][0] = b; j[1][1] = b;
		return *this;
	}
	QScalar* operator [] (const int n) { return j[n]; }
	QScalar* operator [] (const int n) const { return j[n]; }

	lensmatrix<QScalar>& operator += (const lensmatrix<QScalar>& b) {
		j[0][0] += b[0][0]; j[1][0] += b[1][0];
		j[0][1] += b[0][1]; j[1][1] += b[1][1];
		return *this;
	}
	lensmatrix<QScalar> operator + (const lensmatrix<QScalar>& b) {
		lensmatrix<QScalar> ans;
		ans[0][0] = j[0][0] + b[0][0]; ans[1][0] = j[1][0] + b[1][0];
		ans[0][1] = j[0][1] + b[0][1]; ans[1][1] = j[1][1] + b[1][1];
		return ans;
	}
	lensmatrix<QScalar> operator + (const QScalar z) {
		lensmatrix<QScalar> ans;
		ans[0][0] = j[0][0] + z;
		ans[1][1] = j[1][1] + z;
		ans[1][0] = j[1][0];
		ans[0][1] = j[0][1];
		return ans;
	}
	lensmatrix<QScalar> operator - (const QScalar z) {
		lensmatrix<QScalar> ans;
		ans[0][0] = j[0][0] - z;
		ans[1][1] = j[1][1] - z;
		return ans;
	}
	lensmatrix<QScalar>& operator -= (const lensmatrix<QScalar>& b) {
		j[0][0] -= b[0][0]; j[1][0] -= b[1][0];
		j[0][1] -= b[0][1]; j[1][1] -= b[1][1];
		return *this;
	}
	lensmatrix<QScalar> operator - (const lensmatrix<QScalar>& b) {
		lensmatrix<QScalar> ans;
		ans[0][0] = j[0][0] - b[0][0]; ans[1][0] = j[1][0] - b[1][0];
		ans[0][1] = j[0][1] - b[0][1]; ans[1][1] = j[1][1] - b[1][1];
		return ans;
	}
	lensmatrix<QScalar> operator - (void) {
		lensmatrix<QScalar> ans;
		ans[0][0] = -j[0][0]; ans[1][0] = -j[1][0];
		ans[0][1] = -j[0][1]; ans[1][1] = -j[1][1];
		return ans;
	}
	lensvector<QScalar> operator * (const lensvector<QScalar>& b) {
		lensvector<QScalar> ans;
		ans[0] = j[0][0] * b[0] + j[1][0] * b[1];
		ans[1] = j[0][1] * b[0] + j[1][1] * b[1];
		return ans;
	}

	void rotate(const QScalar theta)
	{
		QScalar x_prime, cs, ss;
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
		QScalar x_prime;

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
		QScalar x_prime;
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
	lensmatrix<QScalar> inverse(void)
	{
		lensmatrix<QScalar> ans;
		QScalar det = j[0][0]*j[1][1] - j[0][1]*j[1][0];
		if (det==0.0) die("singular matrix--cannot invert Jacobian matrix");
		ans[0][0] = j[1][1]/det;
		ans[1][0] = -j[1][0]/det;
		ans[0][1] = -j[0][1]/det;
		ans[1][1] = j[0][0]/det;
		return ans;
	}
	bool invert(lensmatrix<QScalar>& ans)
	{
		QScalar det = j[0][0]*j[1][1] - j[0][1]*j[1][0];
		if (det==0.0) { warn("singular matrix--cannot invert Jacobian matrix"); return false; }
		ans[0][0] = j[1][1]/det;
		ans[1][0] = -j[1][0]/det;
		ans[0][1] = -j[0][1]/det;
		ans[1][1] = j[0][0]/det;
		return true;
	}

};

template <typename QScalar>
inline QScalar determinant(const lensmatrix<QScalar> b) { return (b[0][0]*b[1][1] - b[1][0]*b[0][1]); }
template <typename QScalar>
inline lensmatrix<QScalar> operator + (const QScalar a, const lensmatrix<QScalar> b)
{
	lensmatrix<QScalar> ans;
	ans[0][0] = a + b[0][0];
	ans[1][1] = a + b[1][1];
	ans[1][0] = b[1][0];
	ans[0][1] = b[0][1];
	return ans;
}

template <typename QScalar>
inline lensmatrix<QScalar> operator - (const QScalar a, const lensmatrix<QScalar> b)
{
	lensmatrix<QScalar> ans;
	ans[0][0] = a - b[0][0];
	ans[1][1] = a - b[1][1];
	ans[1][0] = -b[1][0];
	ans[0][1] = -b[0][1];
	return ans;
}

template <typename QScalar>
inline void lensmatsqr(const lensmatrix<QScalar> a, lensmatrix<QScalar>& b)
{
	b[0][0] = a[0][0]*a[0][0] + a[0][1]*a[0][1];
	b[0][1] = a[0][0]*a[1][0] + a[0][1]*a[1][1];
	b[1][0] = b[0][1];
	b[1][1] = a[1][1]*a[1][1] + a[1][0]*a[1][0];
}

template <typename QScalar>
inline lensmatrix<QScalar> operator * (const double num, const lensmatrix<QScalar> a)
{
	lensmatrix<QScalar> ans;
	ans[0][0] = num * a[0][0];
	ans[1][1] = num * a[1][1];
	ans[1][0] = num * a[1][0];
	ans[0][1] = num * a[0][1];
	return ans;
}

template <typename QScalar>
inline lensmatrix<QScalar> operator * (const lensmatrix<QScalar> a, const lensmatrix<QScalar> b)
{
	lensmatrix<QScalar> ans;
	ans[0][0] = a[0][0]*b[0][0] + a[1][0]*b[0][1];
	ans[1][0] = a[0][0]*b[1][0] + a[1][0]*b[1][1];
	ans[0][1] = a[0][1]*b[0][0] + a[1][1]*b[0][1];
	ans[1][1] = a[0][1]*b[1][0] + a[1][1]*b[1][1];
	return ans;
}

#endif // LENSVEC_H
