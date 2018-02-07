#include "romberg.h"
#include "errors.h"
#include <cmath>

#define FUNC(x) ((this->*func)(x))

double Romberg::romberg(double (Romberg::*func)(const double), const double a, const double b, const double eps, const int k)
{
	const int jmax = 20;
	if (k > jmax) die("k must be less than or equal to max iterations (%i) in romberg_open", jmax);
	double ss, dss;
	double *s, *hsq;

	// hsq[j] is the (squared) stepsize of the j'th iteration, since the error is a function of h^2
	hsq = new double[jmax+1];
	s = new double[jmax];

	hsq[0] = 1.0;
	for (int j=1; j <= jmax; j++)
	{
		s[j-1] = trapzd(func,a,b,j);
		if (j >= k) {
			RombergPolyExtrapolate(hsq+j-k, s+j-k, k, 0.0, ss, dss);
			if (fabs(dss) <= eps*fabs(ss)) {
				delete[] s;
				delete[] hsq;
				return ss;
			}
		}
		hsq[j] = 0.25 * hsq[j-1]; // number of steps is doubled with each iteration, so h_new = h_old / 2
	}
	die("Too many iterations in routine romberg\n\ncalculated error = %g\nrequired accuracy = %g\nmax iterations = %i", fabs(dss), eps*fabs(ss), jmax);
	return 0.0;
}

double Romberg::romberg(double (Romberg::*func)(const double), const double a, const double b, const double eps, const int k, const double min_error)
{
	const int jmax = 20;
	if (k > jmax) die("k must be less than or equal to max iterations (%i) in romberg_open", jmax);
	double ss, dss;
	double *s, *hsq;
	double err, ferr;

	// hsq[j] is the (squared) stepsize of the j'th iteration, since the error is a function of h^2
	hsq = new double[jmax+1];
	s = new double[jmax];

	hsq[0] = 1.0;
	for (int j=1; j <= jmax; j++)
	{
		s[j-1] = trapzd(func,a,b,j);
		if (j >= k) {
			RombergPolyExtrapolate(hsq+j-k, s+j-k, k, 0.0, ss, dss);
			//err = ((ferr=eps*fabs(ss)) > min_error) ? min_error : ferr;
			err = (fabs(ss) < min_error) ? eps*fabs(ss) : min_error;
			if (fabs(dss) <= err) {
				delete[] s;
				delete[] hsq;
				return ss;
			}
		}
		hsq[j] = 0.25 * hsq[j-1]; // number of steps is doubled with each iteration, so h_new = h_old / 2
	}
	die("Too many iterations in routine romberg\n\ncalculated error = %g\nrequired accuracy = %g\nmax iterations = %i", fabs(dss), eps*fabs(ss), jmax);
	return 0.0;
}

double Romberg::romberg_open(double (Romberg::*func)(const double), const double a, const double b, const double eps, const int k)
{
	const int jmax = 20;
	if (k > jmax) die("k must be less than or equal to max iterations (%i) in romberg_open", jmax);
	double ss, dss;
	double *s, *hsq;

	// hsq[j] is the (squared) stepsize of the j'th iteration, since the error goes like h^2
	hsq = new double[jmax+1];
	s = new double[jmax];

	hsq[0] = 1.0;
	for (int j=1; j <= jmax; j++)
	{
		s[j-1] = midpnt(func,a,b,j);
		if (j >= k) {
			RombergPolyExtrapolate(hsq+j-k, s+j-k, k, 0.0, ss, dss);
			if (fabs(dss) <= eps*fabs(ss)) {
				delete[] s;
				delete[] hsq;
				return ss;
			}
		}
		hsq[j] = hsq[j-1]/9.0; // number of steps is tripled with each iteration, so h_new = h_old / 3
	}
	die("Too many iterations in routine romberg_open\n\ncalculated error = %g\nrequired accuracy = %g\nmax iterations = %i", fabs(dss), eps*fabs(ss), jmax);
	return 0.0;
}

double Romberg::romberg_improper(double (Romberg::*func)(const double), const double a, const double b, const double eps, const int k)
{
	const int jmax = 20;
	if (k > jmax) die("k must be less than or equal to max iterations (%i) in romberg_open", jmax);
	double ss, dss;
	double *s, *hsq;

	// hsq[j] is the (squared) stepsize of the j'th iteration, since the error goes like h^2
	hsq = new double[jmax+1];
	s = new double[jmax];

	hsq[0] = 1.0;
	for (int j=1; j <= jmax; j++)
	{
		s[j-1] = midinf(func,a,b,j);
		if (j >= k) {
			RombergPolyExtrapolate(hsq+j-k, s+j-k, k, 0.0, ss, dss);
			if (fabs(dss) <= eps*fabs(ss)) {
				delete[] s;
				delete[] hsq;
				return ss;
			}
		}
		hsq[j] = hsq[j-1]/9.0; // number of steps is tripled with each iteration, so h_new = h_old / 3
	}
	die("Too many iterations in routine romberg_open\n\ncalculated error = %g\nrequired accuracy = %g\nmax iterations = %i", fabs(dss), eps*fabs(ss), jmax);
	return 0.0;
}

double Romberg::trapzd(double (Romberg::*func)(double), const double a, const double b, const int n)
{
	double x, tnm, sum, del;
	static double s;
	int it, j;

	if (n == 1) {
		return (s = 0.5 * (b-a) * (FUNC(a)+FUNC(b)));
	} else {
		for (it=1, j=1; j < n-1; j++) it <<= 1;
		tnm = it;
		del = (b-a)/tnm;
		x = a + 0.5*del;
		for (sum=0.0, j=0; j < it; j++, x += del) sum += FUNC(x);
		s = 0.5*(s + (b-a)*sum/tnm);
		return s;
	}
}

double Romberg::midpnt(double (Romberg::*func)(double), const double a, const double b, const int n)
{
	double x,tnm,sum,del,ddel;
	static double s;
	int it,j;

	if (n == 1) {
		return (s = (b-a)*FUNC(0.5*(a+b)));
	} else {
		for (it=1, j=1; j < n-1; j++) it *= 3;
		tnm = it;
		del = (b-a)/(3.0*tnm);
		ddel = del + del;
		x = a + 0.5*del;
		sum = 0.0;
		for (j=0; j < it; j++) {
			sum += FUNC(x);
			x += ddel;
			sum += FUNC(x);
			x += del;
		}
		s = (s + (b-a)*sum/tnm)/3.0;
		return s;
	}
}

double Romberg::midinf(double (Romberg::*func)(double), const double aa, const double bb, const int n)
{
	double a,b,x,tnm,sum,del,ddel,mid;
	static double s;
	int it,j;

	b=1.0/aa;
	a=1.0/bb;
	if (n == 1) {
		mid = 0.5*(a+b);
		return (s = (b-a)*(this->*func)(1.0/mid)/mid*mid);
	} else {
		for (it=1, j=1; j < n-1; j++) it *= 3;
		tnm = it;
		del = (b-a)/(3.0*tnm);
		ddel = del + del;
		x = a + 0.5*del;
		sum = 0.0;
		for (j=0; j < it; j++) {
			sum += (this->*func)(1.0/x)/(x*x);
			x += ddel;
			sum += (this->*func)(1.0/x)/(x*x);
			x += del;
		}
		s = (s + (b-a)*sum/tnm)/3.0;
		return s;
	}
}

void Romberg::RombergPolyExtrapolate(double xa[], double ya[], const int n, const double x, double &y, double &dy)
{
	double *c, *d;
	c = new double[n];
	d = new double[n];
	for (int i=0; i < n; i++) {
		c[i] = d[i] = ya[i];
	}

	int m,i;
	double w;
	y = ya[n-1];
	for (m=0; m < n-1; m++)
	{
		for (i=0; i < n-m-1; i++)
		{
			w = (c[i+1] - d[i]) / (xa[i] - xa[i+m+1]);
			c[i] = w * (xa[i]-x);
			d[i] = w * (xa[i+m+1]-x);
		}
		y += d[n-m-2];
	}
	dy = d[0];
	delete[] c;
	delete[] d;
	return;
}


