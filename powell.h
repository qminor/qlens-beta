#ifndef POWELL_H
#define POWELL_H

#include "errors.h"
#include <iostream>

class Powell
{
	private:
	double *p;
	double *xi;
	int iter;
	double fret;
	double ftol;
	int n;
	double *xt;
	double (Powell::*func)(double*);
	static const int ITMAX;
	static const double CGOLD;
	static const double ZEPS;
	double xmin,fmin;
	static const double tol;
	double ax,bx,cx,fa,fb,fc;
	inline void shft3(double &a, double &b, double &c, const double d) { a=b; b=c; c=d; }

	public:
	Powell() { ftol = 3.0e-8; }
	Powell(double (Powell::*funcc)(double*), const double ftoll=3.0e-8) : ftol(ftoll) { func = funcc; }
	void initialize_powell(double (Powell::*funcc)(double*), const double ftoll)
	{
		func = funcc;
		ftol = ftoll;
	}
	void set_precision(const double ftoll) { ftol = ftoll; }
	~Powell() { func = NULL; }

	void powell_minimize(double* pp, const int nn);
	void powell_minimize(double* pp, const int nn, double* initial_stepsizes);

	private:
	void minimize(double* pp, double** ximat);
	double f1dim(const double x);
	void bracket(const double a, const double b);
	double minimize(void);
	double linemin();
};

#endif // POWELL_H
