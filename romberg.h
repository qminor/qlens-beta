
#ifndef ROMBERG_H
#define ROMBERG_H

class Romberg
{
	double trapzd(double (Romberg::*func)(const double), const double a, const double b, const int n);
	double midinf(double (Romberg::*func)(const double), const double aa, const double bb, const int n);
	double midpnt(double (Romberg::*func)(const double), const double a, const double b, const int n);
	void RombergPolyExtrapolate(double xa[], double ya[], const int n, const double x, double &y, double &dy);

	public:
	Romberg() {}
	double romberg(double (Romberg::*func)(const double), const double a, const double b, const double eps, const int k);
	double romberg(double (Romberg::*func)(const double), const double a, const double b, const double eps, const int k, const double min_error);
	double romberg_open(double (Romberg::*func)(const double), const double a, const double b, const double eps, const int k);
	double romberg_improper(double (Romberg::*func)(const double), const double a, const double b, const double eps, const int k);
};

#endif // ROMBERG_H
