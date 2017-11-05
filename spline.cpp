#include "spline.h"
#include "errors.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

// Spline: 1-dimensional cubic spline

Spline::Spline()
{
	xarray = NULL;
	yarray = NULL;
	yspline = NULL;
	nn = 0;
	return;
}

Spline::~Spline()
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;
	return;
}

void Spline::input(double x[], double y[], const int n)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = n;
	xarray = new double[nn];
	yarray = new double[nn];
	yspline = new double[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = x[i];
		yarray[i] = y[i];
	}

	natural_spline();
}

void Spline::input(double x[], double y[], const int n, const double yp1, const double ypn)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = n;
	xarray = new double[nn];
	yarray = new double[nn];
	yspline = new double[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = x[i];
		yarray[i] = y[i];
	}

	unnatural_spline(yp1, ypn);
}

void Spline::input(const Spline& spline_in)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = spline_in.nn;
	xarray = new double[nn];
	yarray = new double[nn];
	yspline = new double[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = spline_in.xarray[i];
		yarray[i] = spline_in.yarray[i];
		yspline[i] = spline_in.yspline[i];
	}
}

void Spline::input(const dvector& x, const dvector& y)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = x.size();
	xarray = new double[nn];
	yarray = new double[nn];
	yspline = new double[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = x[i];
		yarray[i] = y[i];
	}

	natural_spline();
}

void Spline::input(const dvector& x, const dvector& y, const double yp1, const double ypn)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	nn = x.size();
	xarray = new double[nn];
	yarray = new double[nn];
	yspline = new double[nn];

	for (int i = 0; i < nn; i++) {
		xarray[i] = x[i];
		yarray[i] = y[i];
	}

	unnatural_spline(yp1, ypn);
}

void Spline::output(const char *filename)
{
	FILE *outfile;
	outfile = fopen(filename, "w");
	for (int i = 0; i < nn; i++)
		fprintf(outfile, "%le %le\n", xarray[i], yarray[i]);
	fclose(outfile);
}

void Spline::input(const char *filename)
{
	if (xarray) delete[] xarray;
	if (yarray) delete[] yarray;
	if (yspline) delete[] yspline;

	FILE *splinefile;
	if ((splinefile = fopen(filename, "r"))==NULL)
		die("could not open file '%s' for splining\n", filename);

	double xtest, ytest, ysptest;
	int i = 0;

	// Count the number of lines (i.e., points to spline)
	while ((fscanf(splinefile, "%le %le\n", &xtest, &ytest))==2) { i++; }
	nn = i;
	fclose(splinefile);

	xarray = new double[nn];
	yarray = new double[nn];
	yspline = new double[nn];

	splinefile = fopen(filename, "r");
	i = 0;
	while ((fscanf(splinefile, "%le %le\n", &xarray[i], &yarray[i]))==2) { i++; }
	fclose(splinefile);

	natural_spline();
	return;
}

void Spline::natural_spline(void)
{
	double p, qn, sig, un;

	double *u = new double[nn];
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

void Spline::unnatural_spline(double yp1, double ypn)
{
	double p, qn, sig, un;

	double *u = new double[nn];
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

double Spline::splint(const double x)
{
	int klo, khi, k;
	double h, b, a;
	double yi;

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

double Spline::dsplint(const double &x)
{
	int klo, khi, k;
	double h, b, a;
	double yi;

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

double Spline::extend_inner_logslope(const double& x)
{
	double n = (log(yarray[1])-log(yarray[0]))/(log(xarray[1])-log(xarray[0]));
	return (yarray[0] * pow(x/xarray[0], n));
}

double Spline::extend_outer_logslope(const double& x)
{
	double n = (log(yarray[nn-1])-log(yarray[nn-2]))/(log(xarray[nn-1])-log(xarray[nn-2]));
	return (yarray[nn-1] * pow(x/xarray[nn-1], n));
}

double Spline::extend_outer_line(const double& x)
{
	double slope, yval;
	slope = (yarray[nn-1]-yarray[nn-2])/(xarray[nn-1]-xarray[nn-2]);
	yval = yarray[nn-1] + slope*(x-xarray[nn-1]);
	return yval;
}

double Spline::extend_inner_line(const double& x)
{
	double slope, yval;
	slope = (yarray[1]-yarray[0])/(xarray[1]-xarray[0]);
	yval = yarray[0] + slope*(x-xarray[0]);
	return yval;
}

void Spline::print(double min, double max, long steps)
{
	double x, xstep;
	xstep = (max-min)/steps;
	long i;
	for (i = 0, x = min; i < steps; i++, x += xstep)
		printf("%le\t%le\n", x, splint(x));
}

void Spline::logprint(double min, double max, long steps)
{
	double x, xstep;
	xstep = pow(max/min, 1.0/steps);
	long i;
	for (i = 0, x = min; i < steps; i++, x *= xstep)
		printf("%le\t%le\n", x, splint(x));
}

void Spline::dprint(double min, double max, long steps)
{
	double x, xstep;
	xstep = (max-min)/steps;
	long i;
	for (i = 0, x = min; i < steps; i++, x += xstep)
		printf("%le\t%le\n", x, dsplint(x));
}

void Spline::dlogprint(double min, double max, long steps)
{
	double x, xstep;
	xstep = pow(max/min, 1.0/steps);
	long i;
	for (i = 0, x = min; i < steps; i++, x *= xstep)
		printf("%le\t%le\n", x, dsplint(x));
}

void Spline::printall(long steps)
{
	double x, xstep;
	xstep = (xarray[nn-1]-xarray[0])/steps;
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x += xstep)
		printf("%le\t%le\n", x, splint(x));
}

void Spline::printall(long steps, string filename)
{
	ofstream file(filename.c_str());
	double x, xstep;
	xstep = (xarray[nn-1]-xarray[0])/steps;
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x += xstep)
		file << x << " " << splint(x) << endl;
}

void Spline::logprintall(long steps)
{
	double x, xstep;
	xstep = pow(xarray[nn-1]/xarray[0], 1.0/steps);
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x *= xstep)
		printf("%le\t%le\n", x, splint(x));
}

void Spline::dprintall(long steps)
{
	double x, xstep;
	xstep = (xarray[nn-1]-xarray[0])/steps;
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x += xstep)
		printf("%le\t%le\n", x, dsplint(x));
}

void Spline::dlogprintall(long steps)
{
	double x, xstep;
	xstep = pow(xarray[nn-1]/xarray[0], 1.0/steps);
	long i;
	for (i = 0, x = xarray[0]; i <= steps; i++, x *= xstep)
		printf("%le\t%le\n", x, dsplint(x));
}

// Spline2D: 2-dimensional bicubic spline

Spline2D::Spline2D()
{
	xarray = NULL;
	yarray = NULL;
	zmatrix = NULL;
	zspline = NULL;
	mm = 0; nn = 0;
	invert_y = false;
}

Spline2D::~Spline2D()
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

void Spline2D::input(double x[], double y[], double **z, const int m, const int n)
{
	nn = n; mm = m;
	xarray = new double[mm];
	yarray = new double[nn];
	zmatrix = new double*[mm];
	zspline = new double*[mm];
	//z2matrix = new double*[nn];
	//z2spline = new double*[nn];

	for (int i = 0; i < mm; i++) {
		xarray[i] = x[i];
		zmatrix[i] = new double[nn];
		zspline[i] = new double[nn];
	}

	for (int j = 0; j < nn; j++) {
		yarray[j] = y[j];
		//z2matrix[j] = new double[mm];
		//z2spline[j] = new double[mm];
	}

	for (int i = 0; i < mm; i++) {
		for (int j = 0; j < nn; j++) {
			zmatrix[i][j] = z[i][j];
			//z2matrix[j][i] = z[i][j];
		}
	}

	spline();
}

void Spline2D::input(const dvector& x, const dvector& y, const dmatrix& z)
{
	nn = y.size(); mm = x.size();
	xarray = new double[mm];
	yarray = new double[nn];
	zmatrix = new double*[mm];
	zspline = new double*[mm];
	//z2matrix = new double*[nn];
	//z2spline = new double*[nn];

	for (int i = 0; i < mm; i++) {
		xarray[i] = x[i];
		zmatrix[i] = new double[nn];
		zspline[i] = new double[nn];
	}

	for (int j = 0; j < nn; j++) {
		yarray[j] = y[j];
		//z2matrix[j] = new double[mm];
		//z2spline[j] = new double[mm];
	}

	for (int i = 0; i < mm; i++) {
		for (int j = 0; j < nn; j++) {
			zmatrix[i][j] = z[i][j];
			//z2matrix[j][i] = z[i][j];
		}
	}

	spline();
}

void Spline2D::input(const char *xyfilename, const char *zfilename)
{
	FILE *xyfile;
	if ((xyfile = fopen(xyfilename, "r"))==NULL)
		die("could not open file '%s' for splining\n", xyfilename);

	double test;
	int i=0;
	while ((fscanf(xyfile, "%le", &test))==1) { i++; } // Count the number of x values
	getc(xyfile); // Skip over divider (#)
	mm = i;

	int j=0;
	while ((fscanf(xyfile, "%le", &test))==1) { j++; } // Count the number of y values
	nn = j;

	fclose(xyfile);
	xarray = new double[mm];
	yarray = new double[nn];
	zmatrix = new double*[mm];
	zspline = new double*[mm];
	//z2matrix = new double*[nn];
	//z2spline = new double*[nn];

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

/*
	for (i=0; i < nn; i++) {
		z2matrix[i] = new double[mm];
		z2spline[i] = new double[mm];
	}
*/
	for (i=0; i < mm; i++) {
		zmatrix[i] = new double[nn];
		zspline[i] = new double[nn];
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

void Spline2D::input_3column(const char *xyfilename, const char *zfilename)
{
	FILE *xyfile;
	if ((xyfile = fopen(xyfilename, "r"))==NULL)
		die("could not open file '%s' for splining\n", xyfilename);

	double test;
	int i=0;
	while ((fscanf(xyfile, "%le", &test))==1) { i++; } // Count the number of x values
	getc(xyfile); // Skip over divider (#)
	mm = i;

	int j=0;
	while ((fscanf(xyfile, "%le", &test))==1) { j++; } // Count the number of y values
	nn = j;

	fclose(xyfile);
	xarray = new double[mm];
	yarray = new double[nn];
	zmatrix = new double*[mm];
	zspline = new double*[mm];
	//z2matrix = new double*[nn];
	//z2spline = new double*[nn];

	xyfile = fopen(xyfilename, "r");
	i=0; j=0;
	while ((fscanf(xyfile, "%le", &xarray[i++]))==1)
		;
	char cc;
	cc = getc(xyfile); // Skip over divider (#)
	while ((fscanf(xyfile, "%le", &yarray[j++]))==1)
		;

	fclose(xyfile);
	ifstream zin(zfilename);
	//FILE *zfile;
	//if ((zfile = fopen(zfilename, "r"))==NULL)
		//die("could not open file '%s' for splining\n", zfilename);

/*
	for (i=0; i < nn; i++) {
		z2matrix[i] = new double[mm];
		z2spline[i] = new double[mm];
	}
*/
	double crap1, crap2;
	for (i=0; i < mm; i++) {
		zmatrix[i] = new double[nn];
		zspline[i] = new double[nn];
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

void Spline2D::output(const char *xyfilename, const char *zfilename)
{
	FILE *xyfile, *zfile;
	xyfile = fopen(xyfilename, "w");
	for (int i = 0; i < mm; i++)
		fprintf(xyfile, "%le\n", xarray[i]);
	fprintf(xyfile, "#\n");
	for (int i = 0; i < nn; i++)
		fprintf(xyfile, "%le\n", yarray[i]);
	fclose(xyfile);

	zfile = fopen(zfilename, "w");
	for (int i = 0; i < mm; i++)
		for (int j = 0; j < nn; j++)
			fprintf(zfile, "%le\n", zmatrix[i][j]);
	fclose(zfile);
}

void Spline2D::print(double x_min, double x_max, long xsteps, double y_min, double y_max, long ysteps)
{
	double x, xstep, y, ystep;
	xstep = (x_max-x_min)/xsteps;
	ystep = (y_max-y_min)/ysteps;
	long i, j;
	for (i = 0, x = x_min; i < xsteps; i++, x += xstep)
		for (j = 0, y = y_min; j < ysteps; j++, y += ystep)
			printf("%le\t%le\t%le\n", x, y, splint(x,y));
}

void Spline2D::logprint(double x_min, double x_max, long xsteps, double y_min, double y_max, long ysteps)
{
	double x, xstep, y, ystep;
	xstep = pow(x_max/x_min, 1.0/xsteps);
	ystep = pow(y_max/y_min, 1.0/ysteps);
	long i, j;
	for (i = 0, x = x_min; i < xsteps; i++, x *= xstep)
		for (j = 0, y = y_min; j < ysteps; j++, y *= ystep)
			printf("%le\t%le\t%le\n", x, y, splint(x,y));
}

void Spline2D::printall(long xsteps, long ysteps)
{
	double x, xstep, y, ystep;
	xstep = (xarray[mm-1]-xarray[0])/xsteps;
	ystep = (yarray[nn-1]-yarray[0])/ysteps;
	long i, j;
	for (i = 0, x = xarray[0]; i <= xsteps; i++, x += xstep)
		for (j = 0, y = yarray[0]; j <= ysteps; j++, y += ystep)
			printf("%le\t%le\t%le\n", x, y, splint(x,y));
}

void Spline2D::logprintall(long xsteps, long ysteps)
{
	double x, xstep, y, ystep;
	xstep = pow(xarray[0]/xarray[0], 1.0/xsteps);
	ystep = pow(yarray[mm-1]/yarray[0], 1.0/ysteps);
	long i, j;
	for (i = 0, x = xarray[0]; i <= xsteps; i++, x *= xstep)
		for (j = 0, y = yarray[0]; j <= ysteps; j++, y *= ystep)
			printf("%le\t%le\t%le\n", x, y, splint(x,y));
}

void Spline2D::spline(void)
{
	for (int k=0; k < mm; k++)
		spline1D(yarray, zmatrix[k], nn, 1.0e30, 1.0e30, zspline[k]);

	//for (int j=0; j < nn; j++)
		//spline1D(xarray, z2matrix[j], mm, 1.0e30, 1.0e30, z2spline[j]);

	return;
}

void Spline2D::spline_invert_y(void)
{
	for (int k=0; k < mm; k++)
		spline1D(zmatrix[k], yarray, nn, 1.0e30, 1.0e30, zspline[k]);

	return;
}

double Spline2D::splint_invert_y(const double x, const double z)
{
	double *ztmp,*zztmp;
	ztmp = new double[mm];
	zztmp = new double[mm];
	double y_interp;

	for (int j=0; j < mm; j++)
		splint1D(zmatrix[j],yarray,zspline[j],nn,z,&zztmp[j]);

	spline1D(xarray,zztmp,mm,1.0e30,1.0e30,ztmp);
	splint1D(xarray,zztmp,ztmp,mm,x,&y_interp);

	delete[] ztmp;
	delete[] zztmp;
	return y_interp;
}

double Spline2D::splint(const double x, const double y)
{
	double *ytmp,*yytmp;
	ytmp = new double[mm];
	yytmp = new double[mm];
	double z_interp;

	for (int j=0; j < mm; j++)
		splint1D(yarray,zmatrix[j],zspline[j],nn,y,&yytmp[j]);

	spline1D(xarray,yytmp,mm,1.0e30,1.0e30,ytmp);
	splint1D(xarray,yytmp,ytmp,mm,x,&z_interp);

	delete[] ytmp;
	delete[] yytmp;
	return z_interp;
}

void Spline2D::spline1D(double x[], double y[], int n, double yp1, double ypn, double y2[])
{
	double p, qn, sig, un;

	double *u = new double[n];
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

void Spline2D::splint1D(double xa[], double ya[], double y2a[], int n, double x, double *y)
{
	int klo,khi,k;
	double h,b,a;

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
