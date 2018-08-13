#ifndef BRENT_H
#define BRENT_H

class Brent
{
	static const int itmax;
	static const double eps;

	public:
	double BrentsMethod(double (Brent::*func)(const double), const double x1, const double x2, const double tol);
	double BrentsMethod_Inclusive(double (Brent::*func)(const double), const double x1, const double x2, const double tol);
	bool BrentsMethod(double (Brent::*func)(const double), double& root, const double x1, const double x2, const double tol);
};

double BrentsMethod(double (*func)(const double), const double x1, const double x2, const double tol);
double BrentsMethod_Inclusive(double (*func)(const double), const double x1, const double x2, const double tol);

#endif // BRENT_H
