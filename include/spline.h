#ifndef SPLINE_H
#define SPLINE_H

#include "string.h"
#include "vector.h"
#include "matrix.h"
#include "errors.h"

class Spline
{
	private:
		double *xarray, *yarray, *yspline;
		int nn;

	public:
		int length(void) { return nn; }
		double xmin(void) { return xarray[0]; }
		double xmax(void) { return xarray[nn-1]; }
		bool in_range(const double x) { return ((x >= xarray[0]) and (x <= xarray[nn-1])); }
		double y_at_xmin(void) { return yarray[0]; }
		double y_at_xmax(void) { return yarray[nn-1]; }
		double ymax(void) { double ymax_val=-1e30; for (int i=0; i < nn; i++) if (yarray[i] > ymax_val) ymax_val = yarray[i]; return ymax_val; }

		double yval(int n) { return yarray[n]; }

		Spline(double x[], double y[], const int n) { xarray = yarray = yspline = NULL; input(x, y, n); }
		Spline(double x[], double y[], const int n, const double yp1, const double ypn) { xarray=yarray=yspline=NULL; input(x,y,n,yp1,ypn); }
		Spline(const dvector& x, const dvector& y) { xarray = yarray = yspline = NULL; input(x, y); }
		Spline(const char filename[]) { xarray = yarray = yspline = NULL; input(filename); }
		Spline();
		~Spline();
		void input(double x[], double y[], const int n);
		void input(double x[], double y[], const int n, const double yp1, const double ypn);
		void input(const dvector& x, const dvector& y);
		void input(const dvector& x, const dvector& y, const double yp1, const double ypn);
		void input(const char filename[]);
		void output(const char filename[]);
		void output(const std::string filename) { output(filename.c_str()); }
		void input(const std::string filename) { input(filename.c_str()); }
		void input(const Spline& spline_in);
		void natural_spline(void);
		void unnatural_spline(double yp1, double ypn);
		double splint(const double x);
		double splint_linear(const double x);
		double extend_inner_logslope(const double& x);
		double extend_outer_logslope(const double& x);
		double extend_outer_line(const double& x);
		double extend_inner_line(const double& x);
		double dsplint(const double& x);  // interpolates the derivative of the spline

		void print(double, double, long);
		void logprint(double, double, long);
		void printall(long);
		void printall(long steps, std::string filename);
		void logprintall(long);

		void dprint(double, double, long);
		void dlogprint(double, double, long);
		void dprintall(long);
		void dlogprintall(long);
};

class Spline2D
{
	private:
		double *xarray, *yarray;
	  	double **zmatrix, **z2matrix;
		double **zspline, **z2spline;
		int nn, mm;
		bool invert_y;

	public:
		int xlength(void) { return mm; }
		int ylength(void) { return nn; }
		double xmin(void) { return xarray[0]; }
		double xmax(void) { return xarray[mm-1]; }
		double ymin(void) { return yarray[0]; }
		double ymax(void) { return yarray[nn-1]; }
		void get_xvals(dvector &xv) { xv.input(mm); for (int i=0; i < mm; i++) xv[i] = xarray[i]; }
		void get_yvals(dvector &yv) { yv.input(nn); for (int j=0; j < nn; j++) yv[j] = yarray[j]; }

		Spline2D(double x[], double y[], double** z, const int m, const int n) { invert_y = false; input(x,y,z,m,n); }
		Spline2D(char xyfilename[], char zfilename[]) { invert_y = false; input(xyfilename, zfilename); }
		Spline2D(const dvector& x, const dvector& y, const dmatrix& z) { invert_y = false; input(x,y,z); }
		Spline2D();
		~Spline2D();
		void input(double x[], double y[], double** z, const int m, const int n);
		void input(const dvector& x, const dvector& y, const dmatrix& z);
		void input(const char xyfilename[], const char zfilename[]);
		void input(Spline2D &spline_in);
		void input_3column(const char xyfilename[], const char zfilename[]);
		void output(const char xyfilename[], const char zfilename[]);
		void input(const std::string xyfilename, const std::string zfilename) { input(xyfilename.c_str(),zfilename.c_str()); }
		void output(const std::string xyfilename, const std::string zfilename) { output(xyfilename.c_str(),zfilename.c_str()); }
		void spline(void);
		void spline_invert_y(void);
		void spline1D(double[], double[], int, double, double, double[]);
		void splint1D(double xa[], double ya[], double y2a[], int n, double x, double *y);

		double splint(const double x, const double y);
		double splint_invert_y(const double x, const double y);
		void unspline();
		void print(double, double, long, double, double, long);
		void logprint(double, double, long, double, double, long);
		void printall(long, long);
		void logprintall(long, long);
		void set_invert_y(void) { invert_y = true; }
		double zval(int i, int j) { return zmatrix[i][j]; }
		bool is_splined() { return (zmatrix != NULL); }

};

#endif // SPLINE_H
