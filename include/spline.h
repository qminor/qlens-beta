#ifndef SPLINE_H
#define SPLINE_H

#include "string.h"
#include "vector.h"
#include "matrix.h"
#include "errors.h"

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

template <typename T>
class Spline
{
	private:
		T *xarray, *yarray, *yspline;
		int nn;

	public:
		int length(void) { return nn; }
		T xmin(void) { return xarray[0]; }
		T xmax(void) { return xarray[nn-1]; }
		bool in_range(const T x) { return ((x >= xarray[0]) and (x <= xarray[nn-1])); }
		T y_at_xmin(void) { return yarray[0]; }
		T y_at_xmax(void) { return yarray[nn-1]; }
		T ymax(void) { T ymax_val=-1e30; for (int i=0; i < nn; i++) if (yarray[i] > ymax_val) ymax_val = yarray[i]; return ymax_val; }

		T yval(int n) { return yarray[n]; }

		Spline(T x[], T y[], const int n) { xarray = yarray = yspline = NULL; input(x, y, n); }
		Spline(T x[], T y[], const int n, const T yp1, const T ypn) { xarray=yarray=yspline=NULL; input(x,y,n,yp1,ypn); }
		Spline(const dvector& x, const dvector& y) { xarray = yarray = yspline = NULL; input(x, y); }
		Spline(const char filename[]) { xarray = yarray = yspline = NULL; input(filename); }
		Spline();
		~Spline();
		void input(T x[], T y[], const int n);
		void input(T x[], T y[], const int n, const T yp1, const T ypn);
		void input(const dvector& x, const dvector& y);
		void input(const dvector& x, const dvector& y, const T yp1, const T ypn);
		void input(const char filename[]);
		void output(const char filename[]);
		void output(const std::string filename) { output(filename.c_str()); }
		void input(const std::string filename) { input(filename.c_str()); }
		void input(const Spline& spline_in);
		void natural_spline(void);
		void unnatural_spline(T yp1, T ypn);
		T splint(const T x);
		T splint_linear(const T x);
		T extend_inner_logslope(const T& x);
		T extend_outer_logslope(const T& x);
		T extend_outer_line(const T& x);
		T extend_inner_line(const T& x);
		T dsplint(const T& x);  // interpolates the derivative of the spline

		//void print(double, double, long);
		//void logprint(double, double, long);
		//void printall(long);
		//void printall(long steps, std::string filename);
		//void logprintall(long);

		//void dprint(double, double, long);
		//void dlogprint(double, double, long);
		//void dprintall(long);
		//void dlogprintall(long);
};

template <typename T>
class Spline2D
{
	private:
		T *xarray, *yarray;
	  	T **zmatrix, **z2matrix;
		T **zspline, **z2spline;
		int nn, mm;
		bool invert_y;

	public:
		int xlength(void) { return mm; }
		int ylength(void) { return nn; }
		T xmin(void) { return xarray[0]; }
		T xmax(void) { return xarray[mm-1]; }
		T ymin(void) { return yarray[0]; }
		T ymax(void) { return yarray[nn-1]; }
		void get_xvals(dvector &xv) { xv.input(mm); for (int i=0; i < mm; i++) xv[i] = xarray[i]; }
		void get_yvals(dvector &yv) { yv.input(nn); for (int j=0; j < nn; j++) yv[j] = yarray[j]; }

		Spline2D(T x[], T y[], T** z, const int m, const int n) { invert_y = false; input(x,y,z,m,n); }
		Spline2D(char xyfilename[], char zfilename[]) { invert_y = false; input(xyfilename, zfilename); }
		Spline2D(const dvector& x, const dvector& y, const dmatrix& z) { invert_y = false; input(x,y,z); }
		Spline2D();
		~Spline2D();
		void input(T x[], T y[], T** z, const int m, const int n);
		void input(const dvector& x, const dvector& y, const dmatrix& z);
		void input(Spline2D &spline_in);
		//void input(const char xyfilename[], const char zfilename[]);
		//void input_3column(const char xyfilename[], const char zfilename[]);
		void output(const char xyfilename[], const char zfilename[]);
		void input(const std::string xyfilename, const std::string zfilename) { input(xyfilename.c_str(),zfilename.c_str()); }
		void output(const std::string xyfilename, const std::string zfilename) { output(xyfilename.c_str(),zfilename.c_str()); }
		void spline(void);
		void spline_invert_y(void);
		void spline1D(T[], T[], int, T, T, T[]);
		void splint1D(T xa[], T ya[], T y2a[], int n, T x, T *y);

		T splint(const T x, const T y);
		T splint_invert_y(const T x, const T y);
		void unspline();
		//void print(double, double, long, double, double, long);
		//void logprint(double, double, long, double, double, long);
		//void printall(long, long);
		//void logprintall(long, long);
		void set_invert_y(void) { invert_y = true; }
		T zval(int i, int j) { return zmatrix[i][j]; }
		bool is_splined() { return (zmatrix != NULL); }

};

template <typename T>
Spline<T>::Spline()
{
	xarray = NULL;
	yarray = NULL;
	yspline = NULL;
	nn = 0;
	return;
}

template <typename T>
Spline<T>::~Spline()
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;
	return;
}

template <typename T>
void Spline<T>::input(T x[], T y[], const int n)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = n;
	xarray = new T[nn];
	yarray = new T[nn];
	yspline = new T[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = x[i];
		yarray[i] = y[i];
	}

	natural_spline();
}

template <typename T>
void Spline<T>::input(T x[], T y[], const int n, const T yp1, const T ypn)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = n;
	xarray = new T[nn];
	yarray = new T[nn];
	yspline = new T[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = x[i];
		yarray[i] = y[i];
	}

	unnatural_spline(yp1, ypn);
}

template <typename T>
void Spline<T>::input(const Spline& spline_in)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = spline_in.nn;
	xarray = new T[nn];
	yarray = new T[nn];
	yspline = new T[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = spline_in.xarray[i];
		yarray[i] = spline_in.yarray[i];
		yspline[i] = spline_in.yspline[i];
	}
}

template <typename T>
void Spline<T>::input(const dvector& x, const dvector& y)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = x.size();
	xarray = new T[nn];
	yarray = new T[nn];
	yspline = new T[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = x[i];
		yarray[i] = y[i];
	}

	natural_spline();
}

template <typename T>
void Spline<T>::input(const dvector& x, const dvector& y, const T yp1, const T ypn)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = x.size();
	xarray = new T[nn];
	yarray = new T[nn];
	yspline = new T[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = x[i];
		yarray[i] = y[i];
	}

	unnatural_spline(yp1, ypn);
}

template <typename T>
void Spline<T>::output(const char *filename)
{
	FILE *outfile;
	outfile = fopen(filename, "w");
	double xval, yval;
	for (int i = 0; i < nn; i++) {
#ifdef USE_STAN
		xval = xarray[i].val();
		yval = yarray[i].val();
#else
		xval = xarray[i];
		yval = yarray[i];
#endif
		fprintf(outfile, "%le %le\n", xval, yval);
	}
	fclose(outfile);
}

template <typename T>
void Spline<T>::input(const char *filename)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	FILE *splinefile;
	if ((splinefile = fopen(filename, "r"))==NULL)
		die("could not open file '%s' for splining\n", filename);

	double xval, yval, yspval;
	int i = 0;

	// Count the number of lines (i.e., points to spline)
	while ((fscanf(splinefile, "%le %le\n", &xval, &yval))==2) { i++; }
	nn = i;
	fclose(splinefile);

	xarray = new T[nn];
	yarray = new T[nn];
	yspline = new T[nn];

	splinefile = fopen(filename, "r");
	i = 0;
	while ((fscanf(splinefile, "%le %le\n", &xval, &yval))==2) {
		xarray[i] = xval;
		yarray[i] = yval;
		i++;
	}
	fclose(splinefile);

	natural_spline();
	return;
}

template <typename T>
void Spline<T>::natural_spline(void)
{
	T p, qn, sig, un;

	T *u = new T[nn];
	yspline[0] = u[0] = 0.0;

	for (int i=1; i < nn-1; i++) {
		sig = (xarray[i]-xarray[i-1]) / (xarray[i+1]-xarray[i-1]);
		p = sig*yspline[i-1] + 2.0;
		yspline[i] = (sig-1.0)/p;
		u[i] = (yarray[i+1]-yarray[i])/(xarray[i+1]-xarray[i]) - (yarray[i]-yarray[i-1])/(xarray[i]-xarray[i-1]);
		u[i] = (6.0*u[i]/(xarray[i+1]-xarray[i-1]) - sig*u[i-1]) / p;
	}
	qn = un = 0.0;
	yspline[nn-1] = (un-qn*u[nn-2])/(qn*yspline[nn-2]+1.0);
	for (int k = nn-2; k >= 0; k--)
		yspline[k] = yspline[k]*yspline[k+1] + u[k];

	delete[] u;
}

template <typename T>
void Spline<T>::unnatural_spline(T yp1, T ypn)
{
	T p, qn, sig, un;

	T *u = new T[nn];
	yspline[0] = -0.5;
	u[0]=(3.0/(xarray[1]-xarray[0])) * ((yarray[1]-yarray[0])/(xarray[1]-xarray[0]) - yp1);

	for (int i=1; i < nn-1; i++) {
		sig = (xarray[i]-xarray[i-1]) / (xarray[i+1]-xarray[i-1]);
		p = sig*yspline[i-1] + 2.0;
		yspline[i] = (sig-1.0)/p;
		u[i] = (yarray[i+1]-yarray[i])/(xarray[i+1]-xarray[i]) - (yarray[i]-yarray[i-1])/(xarray[i]-xarray[i-1]);
		u[i] = (6.0*u[i]/(xarray[i+1]-xarray[i-1]) - sig*u[i-1]) / p;
	}

	qn = 0.5;
	un = (3.0/(xarray[nn-1]-xarray[nn-2])) * (ypn - (yarray[nn-1]-yarray[nn-2])/(xarray[nn-1]-xarray[nn-2]));
	yspline[nn-1] = (un-qn*u[nn-2])/(qn*yspline[nn-2]+1.0);
	for (int k = nn-2; k >= 0; k--)
		yspline[k] = yspline[k]*yspline[k+1] + u[k];

	delete[] u;
}

template <typename T>
T Spline<T>::splint(const T x)
{
	int klo, khi, k;
	T h, b, a;
	T yi;

	klo = 0;
	khi = nn - 1;
	while (khi-klo > 1) {
		k = (khi+klo) >> 1;
		if (xarray[k] > x) khi = k;
		else klo = k;
	}
	h = xarray[khi] - xarray[klo];
	if (h == 0.0) die("Bad xarray input to routine splint");
	a = (xarray[khi] - x)/h;
	b = (x - xarray[klo])/h;
	yi = a*yarray[klo] + b*yarray[khi] + ((a*a*a-a)*yspline[klo] + (b*b*b-b)*yspline[khi]) * (h*h)/6.0;

	return yi;
}

template <typename T>
T Spline<T>::splint_linear(const T x)
{
	int klo, khi, k;
	T h, b, a;
	T yi;

	klo = 0;
	khi = nn - 1;
	while (khi-klo > 1) {
		k = (khi+klo) >> 1;
		if (xarray[k] > x) khi = k;
		else klo = k;
	}
	h = xarray[khi] - xarray[klo];
	if (h == 0.0) die("Bad xarray input to routine splint");
	a = (xarray[khi] - x)/h;
	b = (x - xarray[klo])/h;
	yi = a*yarray[klo] + b*yarray[khi];

	return yi;
}

template <typename T>
T Spline<T>::dsplint(const T &x)
{
	int klo, khi, k;
	T h, b, a;
	T yi;

	klo = 0;
	khi = nn - 1;
	while (khi-klo > 1) {
		k = (khi+klo) >> 1;
		if (xarray[k] > x) khi = k;
		else klo = k;
	}
	h = xarray[khi] - xarray[klo];
	if (h == 0.0) die("Bad xarray input to routine dsplint");
	a = (xarray[khi] - x) / h;
	b = (x - xarray[klo]) / h;
	yi = (yarray[khi]-yarray[klo])/h + (-(3*a*a-1.0)*yspline[klo] + (3*b*b-1.0)*yspline[khi])*h/6.0;

	return yi;
}

template <typename T>
T Spline<T>::extend_inner_logslope(const T& x)
{
	T n = (log(yarray[1])-log(yarray[0]))/(log(xarray[1])-log(xarray[0]));
	return (yarray[0] * pow(x/xarray[0], n));
}

template <typename T>
T Spline<T>::extend_outer_logslope(const T& x)
{
	T n = (log(yarray[nn-1])-log(yarray[nn-2]))/(log(xarray[nn-1])-log(xarray[nn-2]));
	return (yarray[nn-1] * pow(x/xarray[nn-1], n));
}

template <typename T>
T Spline<T>::extend_outer_line(const T& x)
{
	T slope, yval;
	slope = (yarray[nn-1]-yarray[nn-2])/(xarray[nn-1]-xarray[nn-2]);
	yval = yarray[nn-1] + slope*(x-xarray[nn-1]);
	return yval;
}

template <typename T>
T Spline<T>::extend_inner_line(const T& x)
{
	T slope, yval;
	slope = (yarray[1]-yarray[0])/(xarray[1]-xarray[0]);
	yval = yarray[0] + slope*(x-xarray[0]);
	return yval;
}

/*
template <typename T>
void Spline<T>::print(double min, double max, long steps)
{
	double x, xstep;
	xstep = (max-min)/steps;
	long i;
	for (i = 0, x = min; i < steps; i++, x += xstep)
		printf("%le\t%le\n", x, splint(x));
}

template <typename T>
void Spline<T>::logprint(double min, double max, long steps)
{
	double x, xstep;
	xstep = pow(max/min, 1.0/steps);
	long i;
	for (i = 0, x = min; i < steps; i++, x *= xstep)
		printf("%le\t%le\n", x, splint(x));
}

template <typename T>
void Spline<T>::dprint(double min, double max, long steps)
{
	double x, xstep;
	xstep = (max-min)/steps;
	long i;
	for (i = 0, x = min; i < steps; i++, x += xstep)
		printf("%le\t%le\n", x, dsplint(x));
}

void Spline<T>::dlogprint(double min, double max, long steps)
template <typename T>
template <typename T>
{
	double x, xstep;
	xstep = pow(max/min, 1.0/steps);
	long i;
	for (i = 0, x = min; i < steps; i++, x *= xstep)
		printf("%le\t%le\n", x, dsplint(x));
}

template <typename T>
void Spline<T>::printall(long steps)
{
	double x, xstep;
	xstep = (xarray[nn-1]-xarray[0])/steps;
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x += xstep)
		printf("%le\t%le\n", x, splint(x));
}

template <typename T>
void Spline<T>::printall(long steps, string filename)
{
	ofstream file(filename.c_str());
	double x, xstep;
	xstep = (xarray[nn-1]-xarray[0])/steps;
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x += xstep)
		file << x << " " << splint(x) << endl;
}

template <typename T>
void Spline<T>::logprintall(long steps)
{
	double x, xstep;
	xstep = pow(xarray[nn-1]/xarray[0], 1.0/steps);
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x *= xstep)
		printf("%le\t%le\n", x, splint(x));
}

template <typename T>
void Spline<T>::dprintall(long steps)
{
	double x, xstep;
	xstep = (xarray[nn-1]-xarray[0])/steps;
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x += xstep)
		printf("%le\t%le\n", x, dsplint(x));
}

template <typename T>
void Spline<T>::dlogprintall(long steps)
{
	double x, xstep;
	xstep = pow(xarray[nn-1]/xarray[0], 1.0/steps);
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x *= xstep)
		printf("%le\t%le\n", x, dsplint(x));
}
*/

// Spline2D: 2-dimensional bicubic spline

template <typename T>
Spline2D<T>::Spline2D()
{
	xarray = NULL;
	yarray = NULL;
	zmatrix = NULL;
	zspline = NULL;
	mm = 0; nn = 0;
	invert_y = false;
}

template <typename T>
Spline2D<T>::~Spline2D()
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if ((zmatrix) and (zspline)) {
		for (int i=0; i < mm; i++) {
			delete[] zmatrix[i];
			delete[] zspline[i];
		}
	delete[] zmatrix;
	delete[] zspline;
	}
	/*
	if ((z2matrix) and (z2spline)) {
		for (int i=0; i < nn; i++) {
			delete[] z2matrix[i];
			delete[] z2spline[i];
		}
	delete[] z2matrix;
	delete[] z2spline;
	}
	*/
}

template <typename T>
void Spline2D<T>::unspline()
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if ((zmatrix) and (zspline)) {
		for (int i=0; i < mm; i++) {
			delete[] zmatrix[i];
			delete[] zspline[i];
		}
	delete[] zmatrix;
	delete[] zspline;
	}
	xarray = NULL;
	yarray = NULL;
	zmatrix = NULL;
	zspline = NULL;
	mm = 0; nn = 0;
	invert_y = false;
}



template <typename T>
void Spline2D<T>::input(T x[], T y[], T **z, const int m, const int n)
{
	nn = n; mm = m;
	xarray = new T[mm];
	yarray = new T[nn];
	zmatrix = new T*[mm];
	zspline = new T*[mm];
	//z2matrix = new T*[nn];
	//z2spline = new T*[nn];

	for (int i = 0; i < mm; i++) {
		xarray[i] = x[i];
		zmatrix[i] = new T[nn];
		zspline[i] = new T[nn];
	}

	for (int j = 0; j < nn; j++) {
		yarray[j] = y[j];
		//z2matrix[j] = new T[mm];
		//z2spline[j] = new T[mm];
	}

	for (int i = 0; i < mm; i++) {
		for (int j = 0; j < nn; j++) {
			zmatrix[i][j] = z[i][j];
			//z2matrix[j][i] = z[i][j];
		}
	}

	spline();
}

template <typename T>
void Spline2D<T>::input(const dvector& x, const dvector& y, const dmatrix& z)
{
	nn = y.size(); mm = x.size();
	xarray = new T[mm];
	yarray = new T[nn];
	zmatrix = new T*[mm];
	zspline = new T*[mm];
	//z2matrix = new T*[nn];
	//z2spline = new T*[nn];

	for (int i = 0; i < mm; i++) {
		xarray[i] = x[i];
		zmatrix[i] = new T[nn];
		zspline[i] = new T[nn];
	}

	for (int j = 0; j < nn; j++) {
		yarray[j] = y[j];
		//z2matrix[j] = new T[mm];
		//z2spline[j] = new T[mm];
	}

	for (int i = 0; i < mm; i++) {
		for (int j = 0; j < nn; j++) {
			zmatrix[i][j] = z[i][j];
			//z2matrix[j][i] = z[i][j];
		}
	}

	spline();
}

template <typename T>
void Spline2D<T>::input(Spline2D &spline_in)
{
	nn = spline_in.nn;
	mm = spline_in.mm;
	xarray = new T[mm];
	yarray = new T[nn];
	zmatrix = new T*[mm];
	zspline = new T*[mm];
	//z2matrix = new T*[nn];
	//z2spline = new T*[nn];

	for (int i = 0; i < mm; i++) {
		xarray[i] = spline_in.xarray[i];
		zmatrix[i] = new T[nn];
		zspline[i] = new T[nn];
	}

	for (int j = 0; j < nn; j++) {
		yarray[j] = spline_in.yarray[j];
	}

	for (int i = 0; i < mm; i++) {
		for (int j = 0; j < nn; j++) {
			zmatrix[i][j] = spline_in.zmatrix[i][j];
		}
	}

	for (int i = 0; i < mm; i++) {
		for (int j = 0; j < nn; j++) {
			zspline[i][j] = spline_in.zspline[i][j];
		}
	}
}

/*
template <typename T>
void Spline2D<T>::input(const char *xyfilename, const char *zfilename)
{
	FILE *xyfile;
	if ((xyfile = fopen(xyfilename, "r"))==NULL)
		die("could not open file '%s' for splining\n", xyfilename);

	T test;
	int i=0;
	while ((fscanf(xyfile, "%le", &test))==1) { i++; } // Count the number of x values
	getc(xyfile); // Skip over divider (#)
	mm = i;

	int j=0;
	while ((fscanf(xyfile, "%le", &test))==1) { j++; } // Count the number of y values
	nn = j;

	fclose(xyfile);
	xarray = new T[mm];
	yarray = new T[nn];
	zmatrix = new T*[mm];
	zspline = new T*[mm];
	//z2matrix = new T*[nn];
	//z2spline = new T*[nn];

	xyfile = fopen(xyfilename, "r");
	i=0; j=0;
	while ((fscanf(xyfile, "%le", &xarray[i++]))==1)
		;
	char cc;
	cc = getc(xyfile); // Skip over divider (#)
	while ((fscanf(xyfile, "%le", &yarray[j++]))==1)
		;

	fclose(xyfile);
	FILE *zfile;
	if ((zfile = fopen(zfilename, "r"))==NULL)
		die("could not open file '%s' for splining\n", zfilename);

	for (i=0; i < mm; i++) {
		zmatrix[i] = new T[nn];
		zspline[i] = new T[nn];
		for (j=0; j < nn; j++) {
			if (fscanf(zfile, "%le", &zmatrix[i][j]) != 1)
				die("unexpected end of file in %s\n", zfilename);
			//z2matrix[j][i] = zmatrix[i][j];
		}
	}
	fclose(zfile);

	if (invert_y)
		spline_invert_y();
	else
		spline();
}

template <typename T>
void Spline2D<T>::input_3column(const char *xyfilename, const char *zfilename)
{
	FILE *xyfile;
	if ((xyfile = fopen(xyfilename, "r"))==NULL)
		die("could not open file '%s' for splining\n", xyfilename);

	T test;
	int i=0;
	while ((fscanf(xyfile, "%le", &test))==1) { i++; } // Count the number of x values
	getc(xyfile); // Skip over divider (#)
	mm = i;

	int j=0;
	while ((fscanf(xyfile, "%le", &test))==1) { j++; } // Count the number of y values
	nn = j;

	fclose(xyfile);
	xarray = new T[mm];
	yarray = new T[nn];
	zmatrix = new T*[mm];
	zspline = new T*[mm];
	//z2matrix = new T*[nn];
	//z2spline = new T*[nn];

	xyfile = fopen(xyfilename, "r");
	i=0; j=0;
	while ((fscanf(xyfile, "%le", &xarray[i++]))==1)
		;
	char cc;
	cc = getc(xyfile); // Skip over divider (#)
	while ((fscanf(xyfile, "%le", &yarray[j++]))==1)
		;

	fclose(xyfile);
	std::ifstream zin(zfilename);
	//FILE *zfile;
	//if ((zfile = fopen(zfilename, "r"))==NULL)
		//die("could not open file '%s' for splining\n", zfilename);

	//for (i=0; i < nn; i++) {
		//z2matrix[i] = new T[mm];
		//z2spline[i] = new T[mm];
	//}
	T crap1, crap2;
	for (i=0; i < mm; i++) {
		zmatrix[i] = new T[nn];
		zspline[i] = new T[nn];
		for (j=0; j < nn; j++) {
			zin >> crap1 >> crap2 >> zmatrix[i][j];
			//if (fscanf(zfile, "%le", &zmatrix[i][j]) != 1)
				//die("unexpected end of file in %s\n", zfilename);
			//z2matrix[j][i] = zmatrix[i][j];
		}
	}
	//fclose(zfile);

	if (invert_y)
		spline_invert_y();
	else
		spline();
}
*/

template <typename T>
void Spline2D<T>::output(const char *xyfilename, const char *zfilename)
{
	FILE *xyfile, *zfile;
	xyfile = fopen(xyfilename, "w");
	double xval, yval;
	for (int i = 0; i < mm; i++) {
#ifdef USE_STAN
		xval = xarray[i].val();
#else
		xval = xarray[i];
#endif
		fprintf(xyfile, "%le\n", xval);
	}
	fprintf(xyfile, "#\n");
	for (int i = 0; i < nn; i++) {
#ifdef USE_STAN
		yval = yarray[i].val();
#else
		yval = yarray[i];
#endif

		fprintf(xyfile, "%le\n", yval);
	}
	fclose(xyfile);

	zfile = fopen(zfilename, "w");
	double zval;
	for (int i = 0; i < mm; i++) {
		for (int j = 0; j < nn; j++) {
#ifdef USE_STAN
			zval = zmatrix[i][j].val();
#else
			zval = zmatrix[i][j];
#endif
			fprintf(zfile, "%le\n", zval);
		}
	}
	fclose(zfile);
}

/*
template <typename T>
void Spline2D<T>::print(double x_min, double x_max, long xsteps, double y_min, double y_max, long ysteps)
{
	double x, xstep, y, ystep;
	xstep = (x_max-x_min)/xsteps;
	ystep = (y_max-y_min)/ysteps;
	long i, j;
	for (i = 0, x = x_min; i < xsteps; i++, x += xstep)
		for (j = 0, y = y_min; j < ysteps; j++, y += ystep)
			printf("%le\t%le\t%le\n", x, y, splint(x,y));
}

template <typename T>
void Spline2D<T>::logprint(double x_min, double x_max, long xsteps, double y_min, double y_max, long ysteps)
{
	double x, xstep, y, ystep;
	xstep = pow(x_max/x_min, 1.0/xsteps);
	ystep = pow(y_max/y_min, 1.0/ysteps);
	long i, j;
	for (i = 0, x = x_min; i < xsteps; i++, x *= xstep)
		for (j = 0, y = y_min; j < ysteps; j++, y *= ystep)
			printf("%le\t%le\t%le\n", x, y, splint(x,y));
}

template <typename T>
void Spline2D<T>::printall(long xsteps, long ysteps)
{
	double x, xstep, y, ystep;
	xstep = (xarray[mm-1]-xarray[0])/xsteps;
	ystep = (yarray[nn-1]-yarray[0])/ysteps;
	long i, j;
	for (i = 0, x = xarray[0]; i <= xsteps; i++, x += xstep)
		for (j = 0, y = yarray[0]; j <= ysteps; j++, y += ystep)
			printf("%le\t%le\t%le\n", x, y, splint(x,y));
}

template <typename T>
void Spline2D<T>::logprintall(long xsteps, long ysteps)
{
	double x, xstep, y, ystep;
	xstep = pow(xarray[0]/xarray[0], 1.0/xsteps);
	ystep = pow(yarray[mm-1]/yarray[0], 1.0/ysteps);
	long i, j;
	for (i = 0, x = xarray[0]; i <= xsteps; i++, x *= xstep)
		for (j = 0, y = yarray[0]; j <= ysteps; j++, y *= ystep)
			printf("%le\t%le\t%le\n", x, y, splint(x,y));
}
*/

template <typename T>
void Spline2D<T>::spline(void)
{
	for (int k=0; k < mm; k++)
		spline1D(yarray, zmatrix[k], nn, 1.0e30, 1.0e30, zspline[k]);

	//for (int j=0; j < nn; j++)
		//spline1D(xarray, z2matrix[j], mm, 1.0e30, 1.0e30, z2spline[j]);

	return;
}

template <typename T>
void Spline2D<T>::spline_invert_y(void)
{
	for (int k=0; k < mm; k++)
		spline1D(zmatrix[k], yarray, nn, 1.0e30, 1.0e30, zspline[k]);

	return;
}

template <typename T>
T Spline2D<T>::splint_invert_y(const T x, const T z)
{
	T *ztmp,*zztmp;
	ztmp = new T[mm];
	zztmp = new T[mm];
	T y_interp;

	for (int j=0; j < mm; j++)
		splint1D(zmatrix[j],yarray,zspline[j],nn,z,&zztmp[j]);

	spline1D(xarray,zztmp,mm,1.0e30,1.0e30,ztmp);
	splint1D(xarray,zztmp,ztmp,mm,x,&y_interp);

	delete[] ztmp;
	delete[] zztmp;
	return y_interp;
}

template <typename T>
T Spline2D<T>::splint(const T x, const T y)
{
	T *ytmp,*yytmp;
	ytmp = new T[mm];
	yytmp = new T[mm];
	T z_interp;

	for (int j=0; j < mm; j++)
		splint1D(yarray,zmatrix[j],zspline[j],nn,y,&yytmp[j]);

	spline1D(xarray,yytmp,mm,1.0e30,1.0e30,ytmp);
	splint1D(xarray,yytmp,ytmp,mm,x,&z_interp);

	delete[] ytmp;
	delete[] yytmp;
	return z_interp;
}

template <typename T>
void Spline2D<T>::spline1D(T x[], T y[], int n, T yp1, T ypn, T y2[])
{
	T p, qn, sig, un;

	T *u = new T[n];
	if (yp1 > 0.99e30)
		y2[0] = u[0] = 0.0;
	else {
		y2[0] = -0.5;
		u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0])/(x[1]-x[0]) - yp1);
	}
	for (int i=1; i < n-1; i++) {
		sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
		p = sig * y2[i-1] + 2.0;
		y2[i] = (sig-1.0)/p;
		u[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
		u[i] = (6.0*u[i]/(x[i+1]-x[i-1]) - sig*u[i-1]) / p;
	}
	if (ypn > 0.99e30)
		qn = un = 0.0;
	else {
		qn = 0.5;
		un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]));
	}
	y2[n-1] = (un - qn*u[n-2]) / (qn*y2[n-2] + 1.0);

	for (int k = n-2; k >= 0; k--)
		y2[k] = y2[k]*y2[k+1] + u[k];

	delete[] u;
	return;
}

template <typename T>
void Spline2D<T>::splint1D(T xa[], T ya[], T y2a[], int n, T x, T *y)
{
	int klo,khi,k;
	T h,b,a;

	klo=0;
	khi=n-1;
	while (khi-klo > 1) {
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	if (h == 0.0) die("Bad xa input to routine splint");
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	*y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

#endif // SPLINE_H
