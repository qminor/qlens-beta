// MATRIX.H: Matrix template

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "vector.h"
#include "errors.h"
using namespace std;

template <class T>
class Matrix
{
	T **a;
	int nrows, ncolumns;

public:
	Matrix() : nrows(0), ncolumns(0) { a = NULL; }
	Matrix(const int &n) : a(nullptr) { input(n,n); }
	Matrix(const int &m, const int &n) : a(nullptr) { input(m,n); }
	Matrix(const int &m, const int &n, const char filename[]) : a(nullptr) { input(m,n,filename); }
	Matrix(T **inmatrix, const int &m, const int &n) : a(nullptr) { input(inmatrix, m, n); }
	Matrix(const Matrix&); // copy-constructor
	Matrix& operator = (const Matrix&);
	Matrix& operator = (const T&);
	void input(const int&, const int&);
	void input(T**, const int&, const int&);
	void input(T**); // this assumes the input dimensions are the same as already set
	void input(const Matrix&);
	void input(const int &m, const int &n, const char filename[]);
	void output(const char filename[]);
	void open(const char filename[]);
	void erase();
	void lu_decomposition(int* indx);
	void lu_solve(int *indx, T* b);
	double **pointer() { return a; }
	double *subarray(const int i) { return a[i]; }
	bool is_initialized() { return (a != NULL); }
	double bicubic_interpolate(const double &t, const double &u); // can use if this is a 4x4 matrix with the coefficients (c_ij).
	//Matrix<T>* inverse_ptr(void);
	~Matrix();

	T* operator [] (const int &n) { return a[n]; }
	T* operator [] (const int &n) const { return a[n]; }
		// Note: the trailing 'const' above overloads [] so that it can be
		// used in functions where 'const Matrix' are taken as arguments

	T** ptr(void) { return a; }
	T** ptr(void) const { return a; }
	int rows(void) const { return nrows; }
	int columns(void) const { return ncolumns; }
	bool check_nan(void);
	void print(void);

	inline Matrix operator + (const Matrix&);
	inline Matrix operator - (const Matrix&);
	inline Matrix operator * (const Matrix&);
	inline Matrix& operator += (const Matrix&);
	inline Matrix& operator -= (const Matrix&);
	inline Matrix& operator *= (const Matrix&);
	Matrix operator ~ (void);     // inverse operator
	inline Matrix inverse(void);         // same as ~ above
	void invert (void);    // Takes the inverse of a matrix
	Matrix operator / (const Matrix&);  // same as *~
	void inverse(Matrix<T> &minv, bool &nonsingular_matrix);    // Takes the inverse of a matrix

	inline Matrix operator + (const T&);
	inline Matrix operator - (const T&);
	inline Matrix operator * (const T&);
	inline Vector<T> operator* (const Vector<T>&);
	inline Matrix& operator += (const T&);
	inline Matrix& operator -= (const T&);
	inline Matrix& operator *= (const T&);
	inline Matrix operator - (void);
	Matrix& equate_submatrix(int, int, int, int, Matrix&, int, int, int, int);
	Matrix& equate_submatrix(int, int, int, int, T**, int, int, int, int);
	Matrix& equate_submatrix_transpose(int, int, int, int, Matrix&, int, int, int, int);
	void Swap(T& a, T& b) { T temp = a; a = b; b = temp; }
};

typedef Matrix<float> fmatrix;
typedef Matrix<double> dmatrix;
typedef Matrix<int> imatrix;
typedef Matrix<long double> lmatrix;
typedef Matrix<long long int> limatrix;

template <class T>
Matrix<T>::Matrix(const Matrix<T>& b)
{
	nrows = b.nrows;
	ncolumns = b.ncolumns;
	a = new T*[nrows];
	
	for (int i = 0; i < nrows; i++)
	{
		a[i] = new T[ncolumns];
		for (int j = 0; j < ncolumns; j++)
			a[i][j] = b[i][j];
	}
	return;
}

template <class T>
void Matrix<T>::input(const Matrix<T>& b)
{
	if (a != NULL) {
		for (int i = 0; i < nrows; i++)
			delete[] a[i];
		delete[] a;
	}

	nrows = b.nrows;
	ncolumns = b.ncolumns;
	a = new T*[nrows];
	
	for (int i = 0; i < nrows; i++)
	{
		a[i] = new T[ncolumns];
		for (int j = 0; j < ncolumns; j++)
			a[i][j] = b[i][j];
	}
	return;
}

template <class T>
void Matrix<T>::input(const int &m, const int &n)
{
	if (a != NULL) {
		for (int i = 0; i < nrows; i++)
			delete[] a[i];
		delete[] a;
	}

	nrows = m; ncolumns = n;
	a = new T*[nrows];
	for (int i = 0; i < nrows; i++) {
		a[i] = new T[ncolumns];
	}
	return;
}

template <class T>
void Matrix<T>::input(T **inmatrix, const int &m, const int &n)
{
	if (a != NULL) {
		for (int i = 0; i < nrows; i++)
			delete[] a[i];
		delete[] a;
	}

	nrows = m; ncolumns = n;
	a = new T*[nrows];
	for (int i = 0; i < nrows; i++) {
		a[i] = new T[ncolumns];
		for (int j = 0; j < ncolumns; j++) {
			a[i][j] = inmatrix[i][j];
		}
	}
	return;
}

template <class T>
void Matrix<T>::input(T **inmatrix)
{
	if (a != NULL) {
		for (int i = 0; i < nrows; i++)
			delete[] a[i];
		delete[] a;
	}

	a = inmatrix; // NOTE: if nrows or ncolumns does not match the inmatrix, you'll get a seg fault!
}

template <class T>
void Matrix<T>::input(const int &m, const int &n, const char filename[])
{
	if (a != NULL) {
		for (int i = 0; i < nrows; i++)
			delete[] a[i];
		delete[] a;
	}

	ifstream infile(filename);

	nrows = m; ncolumns = n;
	a = new T*[nrows];
	for (int i = 0; i < nrows; i++) {
		a[i] = new T[ncolumns];
		for (int j = 0; j < ncolumns; j++) {
			if (!(infile >> a[i][j])) die("could not load matrix from file '%s'; wrong number of matrix elements?",filename);
		}
	}
	return;
}

template <class T>
void Matrix<T>::open(const char filename[])
{
	ifstream infile(filename);

	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncolumns; j++) {
			if (!(infile >> a[i][j])) die("could not load matrix from file '%s'; wrong number of matrix elements?",filename);
		}
	}
	return;
}

template <class T>
double Matrix<T>::bicubic_interpolate(const double &t, const double &u) // can use if this is a 4x4 matrix with the coefficients (c_ij).
{
	static double rfactor;
	rfactor = 0;
	for (int i=3; i >= 0; i--)
		rfactor = t*rfactor + ((a[i][3]*u + a[i][2])*u + a[i][1])*u + a[i][0];
	return rfactor;
}

template <class T>
void Matrix<T>::output(const char filename[])
{
	ofstream outfile(filename);

	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncolumns; j++) {
			outfile << a[i][j] << " ";
		}
		outfile << endl;
	}
	return;
}

template <class T>
Matrix<T>::~Matrix()
{
	if (a != NULL) {
		for (int i = 0; i < nrows; i++)
			delete[] a[i];
		delete[] a;
	}
}

template <class T>
void Matrix<T>::erase()
{
	if (a != NULL) {
		for (int i = 0; i < nrows; i++)
			delete[] a[i];
		delete[] a;
		a = NULL;
	}
	nrows = 0;
	ncolumns = 0;
}

template <class T>
inline Matrix<T> Matrix<T>::operator+ (const Matrix<T> &b)
{
	if ((b.nrows) != nrows) die("Addition of two matrices with different row number");
	if ((b.ncolumns) != ncolumns) die("Addition of two matrices with different column number");
	
	Matrix<T> c(nrows, ncolumns);
	
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			c[i][j] = a[i][j] + b[i][j];
	
	return c;
}

template <class T>
inline Matrix<T> Matrix<T>::operator* (const T &number)
{
	Matrix<T> c(nrows, ncolumns);
	
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			c[i][j] =  a[i][j] * number;
	
	return c;
}

template <class T>
inline Matrix<T> Matrix<T>::operator+ (const T &number)
{
	
	Matrix<T> c(nrows, ncolumns);
	
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			c[i][j] = (i==j) ? a[i][j] + number : a[i][j];
	
	return c;
}

template <class T>
inline Matrix<T> Matrix<T>::operator- (const Matrix<T> &b)
{
	if ((b.nrows) != nrows) die("Addition of two matrices with different row number");
	if ((b.ncolumns) != ncolumns) die("Addition of two matrices with different column number");
	
	Matrix<T> c(nrows, ncolumns);
	
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			c[i][j] = a[i][j] - b[i][j];
	
	return c;
}

template <class T>
inline Matrix<T> Matrix<T>::operator- (const T &number)
{
	Matrix<T> c(nrows, ncolumns);
	
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
				c[i][j] = (i==j) ? a[i][j] - number : a[i][j];
	
	return c;
}

template <class T>
inline Matrix<T> &Matrix<T>::operator= (const Matrix<T> &b)
{
	if ((b.nrows) != nrows) die("Equating two matrices with different row number");
	if ((b.ncolumns) != ncolumns) die("Equating two matrices with different column number");
	
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			a[i][j] = b.a[i][j];
	
	return *this;
}

template <class T>
Matrix<T> &Matrix<T>::equate_submatrix(int ri, int ci, int rf, int cf, Matrix<T> &b, int bri, int bci, int brf, int bcf)
{
	if (rf-ri != brf-bri) die("equating submatrices of unequal row number");
	if (cf-ci != bcf-bci) die("equating submatrices of unequal column number");
	if ((rf >= nrows) || (cf >= ncolumns)) die("element (%i,%i) not contained in matrix", rf, cf);
	if ((brf >= b.nrows) || (bcf >= b.ncolumns)) die("element (%i,%i) not contained in matrix", brf, bcf);

	int i, j, bi, bj;
	for (i = ri, bi = bri; i <= rf; i++, bi++)
		for (j = ci, bj = bci; j <= cf; j++, bj++)
		a[i][j] = b[bi][bj];

	return *this;
}

template <class T>
Matrix<T> &Matrix<T>::equate_submatrix(int ri, int ci, int rf, int cf, T **b, int bri, int bci, int brf, int bcf)
{
	if (rf-ri != brf-bri) die("equating submatrices of unequal row number");
	if (cf-ci != bcf-bci) die("equating submatrices of unequal column number");
	if ((rf >= nrows) || (cf >= ncolumns)) die("element (%i,%i) not contained in matrix", rf, cf);

	int i, j, bi, bj;
	for (i = ri, bi = bri; i <= rf; i++, bi++)
		for (j = ci, bj = bci; j <= cf; j++, bj++)
		a[i][j] = b[bi][bj];

	return *this;
}

template <class T>
Matrix<T> &Matrix<T>::equate_submatrix_transpose(int ri, int ci, int rf, int cf, Matrix<T> &b, int bri, int bci, int brf, int bcf)
{
	if (rf-ri != bcf-bci) die("equating transposed submatrices of unequal row number");
	if (cf-ci != brf-bri) die("equating transposed submatrices of unequal column number");
	if ((rf >= nrows) || (cf >= ncolumns)) die("element (%i,%i) not contained in matrix", rf, cf);

	int i, j, bi, bj;
	for (i = ri, bj = bci; i <= rf; i++, bj++)
		for (j = ci, bi = bri; j <= cf; j++, bi++)
		a[i][j] = b[bi][bj];

	return *this;
}

template <class T>
inline Matrix<T> &Matrix<T>::operator= (const T &number)
{
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			a[i][j] = (i==j) ? number : 0.0;
	
	return *this;
}

template <class T>
inline Matrix<T> &Matrix<T>::operator+= (const Matrix<T> &b)
{
	if ((b.nrows) != nrows) die("Equating two matrices with different row number");
	if ((b.ncolumns) != ncolumns) die("Equating two matrices with different column number");
	
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			a[i][j] += b[i][j];
	
	return *this;
}

template <class T>
inline Matrix<T> &Matrix<T>::operator*= (const T &number)
{
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			a[i][j] *= number;
	
	return *this;
}

template <class T>
inline Matrix<T> &Matrix<T>::operator+= (const T &number)
{
	for (int i = 0; i < ((nrows < ncolumns) ? nrows : ncolumns); i++)
			a[i][i] += number;
	
	return *this;
}

template <class T>
inline Matrix<T> &Matrix<T>::operator-= (const Matrix<T> &b)
{
	if ((b.nrows) != nrows) die("Equating two matrices with different row number");
	if ((b.ncolumns) != ncolumns) die("Equating two matrices with different column number");
	
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			a[i][j] -= b[i][j];
	
	return *this;
}

template <class T>
inline Matrix<T> &Matrix<T>::operator-= (const T &number)
{
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			if (i==j) a[i][j] -= number;
	
	return *this;
}

template <class T>
inline Matrix<T> Matrix<T>::operator* (const Matrix<T> &b)
{
	int brows, bcolumns;
	brows = b.nrows;
	bcolumns = b.ncolumns;

	if (brows != ncolumns) die("Multiplication of two matrices with row# != column#");

	Matrix<T> c(nrows, bcolumns);

	for (int j = 0; j < bcolumns; j++) {
		for (int i = 0; i < nrows; i++) {
			c[i][j] = 0;
			for (int k = 0; k < ncolumns; k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	}

	return c;
}

template <class T>
inline Vector<T> Matrix<T>::operator* (const Vector<T>& b)
{
	if (b.size() != ncolumns) die("multiplication of a matrix and a vector with column# != vector length");
	Vector<T> c(nrows);

	for (int i = 0; i < nrows; i++) {
		c[i] = 0;
		for (int j = 0; j < ncolumns; j++)
			c[i] += a[i][j] * b[j];
	}

	return c;
}

template <class T>
inline Matrix<T> &Matrix<T>::operator*= (const Matrix<T>& b)
{
	int brows, bcolumns;
	brows = b.nrows;
	bcolumns = b.ncolumns;

	if (brows != ncolumns) die("Multiplication of two matrices with row# != column#");
	if (nrows != ncolumns) die("Matrix must be square to use '*=' operator");
	if (brows != bcolumns) die("Matrix must be square to use '*=' operator");

	Matrix<T> c(nrows, bcolumns);

	for (int j = 0; j < ncolumns; j++) {
		for (int i = 0; i < nrows; i++) {
			c[i][j] = 0;
			for (int k = 0; k < ncolumns; k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	}

	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncolumns; j++)
			a[i][j] = c[i][j];
	
	return *this;
}

template <class T>
void Matrix<T>::lu_decomposition(int* indx)
{
	if (nrows != ncolumns) die("cannot perform LU decomposition of a non-square matrix (%ix%i)", nrows, ncolumns);
	static const T TINY=1e-20;
	int i,imax,j,k;
	T big,dum,sum,temp;

	T *vv = new T[nrows];
	for (i=0; i < nrows; i++) {
		big=0;
		for (j=0; j < nrows; j++)
			if ((temp=abs(a[i][j])) > big) big=temp;
		if (big == 0) die("Singular matrix in routine lu_decomposition");
		vv[i] = 1.0/big;
	}
	for (j=0; j < nrows; j++) {
		for (i=0; i < j; i++) {
			sum = a[i][j];
			for (k=0; k < i; k++) sum -= a[i][k]*a[k][j];
			a[i][j] = sum;
		}
		big=0;
		for (i=j; i < nrows; i++) {
			sum = a[i][j];
			for (k=0; k < j; k++) sum -= a[i][k]*a[k][j];
			a[i][j] = sum;
			if ((dum=vv[i]*abs(sum)) >= big) {
				big=dum;
				imax=i;
			}
		}
		if (j != imax) {
			for (k=0; k < nrows; k++) {
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (a[j][j] == 0) { cout << "Singular matrix!\n"; a[j][j] = TINY; }
		if (j != nrows-1) {
			dum = 1.0/(a[j][j]);
			for (i=j+1; i < nrows; i++) a[i][j] *= dum;
		}
	}
	delete[] vv;
}

template <class T>
void Matrix<T>::lu_solve(int* indx, T* b)
{
	int i,ii=0,ip,j;
	T sum;

	for (i=0; i < nrows; i++) {
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii != 0)
			for (j=ii-1; j < i; j++) sum -= a[i][j]*b[j];
		else if (sum != 0)
			ii=i+1;
		b[i] = sum;
	}
	for (i=nrows-1; i >= 0; i--) {
		sum = b[i];
		for (j=i+1; j < nrows; j++) sum -= a[i][j]*b[j];
		b[i] = sum / a[i][i];
	}
}

template <class T>
bool Matrix<T>::check_nan(void)
{
	int i,j;
	for (i=0; i < nrows; i++) {
		for (j=0; j < ncolumns; j++) {
			if (std::isnan(a[i][j])) return false;
		}
	}
	return true;
}

template <class T>
Matrix<T> Matrix<T>::operator~ (void)    // Takes the inverse of a matrix
{
	if (nrows != ncolumns) die("cannot take inverse of a non-square matrix (%ix%i)", nrows, ncolumns);
	int n = nrows;

	T **inv;
	inv = new T*[n];
	for (int i = 0; i < n; i++) {
		inv[i] = new T[n];
		for (int j = 0; j < n; j++)
			inv[i][j] = a[i][j];
	}

	int *indxc,*indxr,*ipiv;
	int i,icol,irow,j,k,l,ll;
	T big,dum,pivinv;

	indxc = new int[n];
	indxr = new int[n];
	ipiv = new int[n];
	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) {
		big=0;
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) {
					if (ipiv[k] == 0) {
						if (fabs(inv[j][k]) >= big) {
							big=(T) fabs(inv[j][k]);
							irow=j;
							icol=k;
						}
					} else if (ipiv[k] > 1) die("Singular Matrix-1, could not take inverse");
				}
		++(ipiv[icol]);
		if (irow != icol) {
			for (l=0;l<n;l++) Swap(inv[irow][l],inv[icol][l]);
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if (inv[icol][icol] == 0) die("Singular Matrix-2, could not take inverse");
		pivinv=(T) 1.0/inv[icol][icol];
		inv[icol][icol]=1;
		for (l=0;l<n;l++) inv[icol][l] *= pivinv;
		for (ll=0;ll<n;ll++)
			if (ll != icol) {
				dum=inv[ll][icol];
				inv[ll][icol]=0;
				for (l=0;l<n;l++) inv[ll][l] -= inv[icol][l]*dum;
			}
	}
	for (l=n-1;l>=0;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				Swap(inv[k][indxr[l]],inv[k][indxc[l]]);
	}
	delete[] ipiv;
	delete[] indxr;
	delete[] indxc;

	Matrix<T> minv(inv, n, n);

	for (int i = 0; i < n; i++)
		delete[] inv[i];
	delete[] inv;

	return minv;
}

template <class T>
void Matrix<T>::invert (void)    // Takes the inverse of a matrix
{
	if (nrows != ncolumns) die("cannot take inverse of a non-square matrix (%ix%i)", nrows, ncolumns);
	int n = nrows;

	T **inv;
	inv = new T*[n];
	for (int i = 0; i < n; i++) {
		inv[i] = new T[n];
		for (int j = 0; j < n; j++)
			inv[i][j] = a[i][j];
	}

	int *indxc,*indxr,*ipiv;
	int i,icol,irow,j,k,l,ll;
	T big,dum,pivinv;

	indxc = new int[n];
	indxr = new int[n];
	ipiv = new int[n];
	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) {
		big=0;
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) {
					if (ipiv[k] == 0) {
						if (fabs(inv[j][k]) >= big) {
							big=(T) fabs(inv[j][k]);
							irow=j;
							icol=k;
						}
					} else if (ipiv[k] > 1) die("Singular Matrix-1, could not take inverse");
				}
		++(ipiv[icol]);
		if (irow != icol) {
			for (l=0;l<n;l++) Swap(inv[irow][l],inv[icol][l]);
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if (inv[icol][icol] == 0) die("Singular Matrix-2, could not take inverse");
		pivinv=(T) 1.0/inv[icol][icol];
		inv[icol][icol]=1;
		for (l=0;l<n;l++) inv[icol][l] *= pivinv;
		for (ll=0;ll<n;ll++)
			if (ll != icol) {
				dum=inv[ll][icol];
				inv[ll][icol]=0;
				for (l=0;l<n;l++) inv[ll][l] -= inv[icol][l]*dum;
			}
	}
	for (l=n-1;l>=0;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				Swap(inv[k][indxr[l]],inv[k][indxc[l]]);
	}
	delete[] ipiv;
	delete[] indxr;
	delete[] indxc;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			a[i][j] = inv[i][j];
	}

	for (int i = 0; i < n; i++)
		delete[] inv[i];
	delete[] inv;
}

template <class T>
Matrix<T> Matrix<T>::operator/ (const Matrix& b)    // Takes the inverse of a matrix
{
	int brows, bcolumns;
	brows = b.nrows;
	bcolumns = b.ncolumns;

	if (brows != bcolumns) die("cannot take inverse of a non-square matrix (%ix%i)", nrows, ncolumns);
	if (brows != ncolumns) die("Multiplication of two matrices with row# != column#");
	int n = nrows;

	T **inv;
	inv = new T*[n];
	for (int i = 0; i < n; i++) {
		inv[i] = new T[n];
		for (int j = 0; j < n; j++)
			inv[i][j] = b[i][j];
	}

	int *indxc,*indxr,*ipiv;
	int i,icol,irow,j,k,l,ll;
	T big,dum,pivinv,temp;

	indxc = new int[n];
	indxr = new int[n];
	ipiv = new int[n];
	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) {
		big=0.0;
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) {
					if (ipiv[k] == 0) {
						if (fabs(inv[j][k]) >= big) {
							big=fabs(inv[j][k]);
							irow=j;
							icol=k;
						}
					} else if (ipiv[k] > 1) die("Singular Matrix-1, could not take inverse");
				}
		++(ipiv[icol]);
		if (irow != icol) {
			for (l=0;l<n;l++) Swap(inv[irow][l],inv[icol][l]);
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if (inv[icol][icol] == 0.0) die("Singular Matrix-2, could not take inverse");
		pivinv=1.0/inv[icol][icol];
		inv[icol][icol]=1.0;
		for (l=0;l<n;l++) inv[icol][l] *= pivinv;
		for (ll=0;ll<n;ll++)
			if (ll != icol) {
				dum=inv[ll][icol];
				inv[ll][icol]=0.0;
				for (l=0;l<n;l++) inv[ll][l] -= inv[icol][l]*dum;
			}
	}
	for (l=n-1;l>=0;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				Swap(inv[k][indxr[l]],inv[k][indxc[l]]);
	}
	delete[] ipiv;
	delete[] indxr;
	delete[] indxc;

	Matrix<T> c(nrows, bcolumns);

	for (int j = 0; j < bcolumns; j++) {
		for (int i = 0; i < nrows; i++) {
			temp = 0;
			for (int k = 0; k < ncolumns; k++)
				temp += a[i][k] * inv[k][j];
			c[i][j] = temp;
		}
	}
	for (int i = 0; i < n; i++)
		delete[] inv[i];
	delete[] inv;

	return c;
}

template <class T>
inline Matrix<T> operator + (const T a, Matrix<T> b)
{
	int brows, bcolumns;
	brows = b.rows(); bcolumns = b.columns();
	Matrix<T> c(brows,bcolumns);
	for (int j = 0; j < bcolumns; j++)
		for (int i = 0; i < brows; i++)
			c[i][j] = (i==j) ? a + b[i][j] : b[i][j];
	return c;
}

template <class T>
inline Matrix<T> operator - (const T a, Matrix<T> b)
{
	int brows, bcolumns;
	brows = b.rows(); bcolumns = b.columns();
	Matrix<T> c(brows,bcolumns);
	for (int j = 0; j < bcolumns; j++)
		for (int i = 0; i < brows; i++)
			c[i][j] = (i==j) ? a - b[i][j] : -b[i][j];
	return c;
}

template <class T>
inline Matrix<T> Matrix<T>::operator - (void)
{
	Matrix c(nrows, ncolumns);
	for (int j = 0; j < ncolumns; j++)
		for (int i = 0; i < nrows; i++)
			c[i][j] = -a[i][j];
	return c;
}

template <class T>
inline Matrix<T> Matrix<T>::inverse(void) { return ~(*this); }

template <class T>
void Matrix<T>::inverse(Matrix<T>& minv, bool &nonsingular_matrix)    // Takes the inverse of a matrix
{
	nonsingular_matrix = false; // singular until proven otherwise!
	if (nrows != ncolumns) die("cannot take inverse of a non-square matrix (%ix%i)", nrows, ncolumns);
	int n = nrows;

	T **inv;
	inv = new T*[n];
	for (int i = 0; i < n; i++) {
		inv[i] = new T[n];
		for (int j = 0; j < n; j++)
			inv[i][j] = a[i][j];
	}

	int *indxc,*indxr,*ipiv;
	int i,icol,irow,j,k,l,ll;
	T big,dum,pivinv,temp;

	indxc = new int[n];
	indxr = new int[n];
	ipiv = new int[n];
	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) {
		big=0;
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) {
					if (ipiv[k] == 0) {
						if (fabs(inv[j][k]) >= big) {
							big=(T) fabs(inv[j][k]);
							irow=j;
							icol=k;
						}
					} else if (ipiv[k] > 1) { nonsingular_matrix = false; return; }
				}
		++(ipiv[icol]);
		if (irow != icol) {
			for (l=0;l<n;l++) Swap(inv[irow][l],inv[icol][l]);
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if (inv[icol][icol] == 0) { nonsingular_matrix = false; return; }
		pivinv=(T) 1.0/inv[icol][icol];
		inv[icol][icol]=1;
		for (l=0;l<n;l++) inv[icol][l] *= pivinv;
		for (ll=0;ll<n;ll++)
			if (ll != icol) {
				dum=inv[ll][icol];
				inv[ll][icol]=0;
				for (l=0;l<n;l++) inv[ll][l] -= inv[icol][l]*dum;
			}
	}
	for (l=n-1;l>=0;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				Swap(inv[k][indxr[l]],inv[k][indxc[l]]);
	}
	delete[] ipiv;
	delete[] indxr;
	delete[] indxc;

	minv.input(inv, n, n);

	for (int i = 0; i < n; i++)
		delete[] inv[i];
	delete[] inv;

	nonsingular_matrix = true;
}



/*
template <class T>
Matrix<T>* Matrix<T>::inverse_ptr(void)
{
	if (nrows != ncolumns) die("cannot take inverse of a non-square matrix (%ix%i)", nrows, ncolumns);
	int i, j, n = nrows;

	//T **inv;
	Matrix<T> inv(n,n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			inv[i][j] = a[i][j];
	}

	int *indxc,*indxr,*ipiv;
	int icol,irow,k,l,ll;
	T big,dum,pivinv,temp;

	indxc = new int[n];
	indxr = new int[n];
	ipiv = new int[n];
	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) {
		big=0;
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) {
					if (ipiv[k] == 0) {
						if (fabs(inv[j][k]) >= big) {
							big=(T) fabs(inv[j][k]);
							irow=j;
							icol=k;
						}
					} else if (ipiv[k] > 1) die("Singular Matrix-1, could not take inverse");
				}
		++(ipiv[icol]);
		if (irow != icol) {
			for (l=0;l<n;l++) Swap(inv[irow][l],inv[icol][l]);
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if (inv[icol][icol] == 0) die("Singular Matrix-2, could not take inverse");
		pivinv=(T) 1.0/inv[icol][icol];
		inv[icol][icol]=1;
		for (l=0;l<n;l++) inv[icol][l] *= pivinv;
		for (ll=0;ll<n;ll++)
			if (ll != icol) {
				dum=inv[ll][icol];
				inv[ll][icol]=0;
				for (l=0;l<n;l++) inv[ll][l] -= inv[icol][l]*dum;
			}
	}
	for (l=n-1;l>=0;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				Swap(inv[k][indxr[l]],inv[k][indxc[l]]);
	}
	delete[] ipiv;
	delete[] indxr;
	delete[] indxc;

	return &inv;
}
*/

template <class T>
void Matrix<T>::print(void)
{
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncolumns; j++)
			cout << a[i][j] << "\t";
		cout << endl;
	}
}

inline Matrix<double> operator* (const double num, const Matrix<double>& a)
{
	Matrix<double> c(a.rows(), a.columns());
	
	for (int i = 0; i < a.rows(); i++)
		for (int j = 0; j < a.columns(); j++)
			c[i][j] =  a[i][j] * num;
	
	return c;
}

inline Matrix<float> operator* (const float num, const Matrix<float>& a)
{
	Matrix<float> c(a.rows(), a.columns());
	
	for (int i = 0; i < a.rows(); i++)
		for (int j = 0; j < a.columns(); j++)
			c[i][j] =  a[i][j] * num;
	
	return c;
}

inline Matrix<int> operator* (const int num, const Matrix<int>& a)
{
	Matrix<int> c(a.rows(), a.columns());
	
	for (int i = 0; i < a.rows(); i++)
		for (int j = 0; j < a.columns(); j++)
			c[i][j] =  a[i][j] * num;
	
	return c;
}

#endif // MATRIX_H
