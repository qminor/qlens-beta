#include "brent.h"
#include "errors.h"
#include <cmath>

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

const int Brent::itmax = 100;
const double Brent::eps = 3.0e-8;

const int itmax = 100;
const double eps = 3.0e-8;

double Brent::BrentsMethod(double (Brent::*func)(const double), const double x1, const double x2, const double tol)
{
	double a = x1, b = x2, c = x2, d, e, min1, min2;
	double fa = (this->*func)(a), fb = (this->*func)(b);
	double fc, p, q, r, s, tol1, xm;

	if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
		die("root must be bracketed in Brent's Method");

	fc = fb;
	for (int it=0; it < itmax; it++) {
		if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
			c = a;
			fc = fa;
			e = d = b-a;
		}
		if (fabs(fc) < fabs(fb)) {
			a = b;
			b = c;
			c = a;
			fa = fb;
			fb = fc;
			fc = fa;
		}
		tol1 = 2.0*eps*fabs(b) + 0.5*tol;
		xm = 0.5*(c-b);
		if (fabs(xm) <= tol1 || fb == 0.0) return b;

		if (fabs(e) >= tol1 && fabs(fa) > fabs(fb))
		{
			s = fb / fa;
			if (a == c) {
				p = 2.0*xm*s;
				q = 1.0 - s;
			} else {
				q = fa/fc;
				r = fb/fc;
				p = s * (2.0*xm*q*(q-r) - (b-a)*(r-1.0));
				q = (q-1.0)*(r-1.0)*(s-1.0);
			}
			if (p > 0.0) q = -q;
			p = fabs(p);
			min1 = 3.0*xm*q - fabs(tol1*q);
			min2 = fabs(e*q);
			if (2.0*p < (min1 < min2 ? min1 : min2)) {
				e = d;
				d = p/q;
			} else {
				d = xm;
				e = d;
			}
		} else {
			d = xm;
			e = d;
		}
		a = b;
		fa = fb;
		if (fabs(d) > tol1)
			b += d;
		else
			b += SIGN(tol1,xm);
		fb = (this->*func)(b);
	}
	die("maximum number of iterations exceeded in Brent's Method");
	return 0.0;
}

double Brent::BrentsMethod(double (Brent::*func)(const double, const double&, const double&), const double &param1, const double &param2, const double x1, const double x2, const double tol)
{
	double a = x1, b = x2, c = x2, d, e, min1, min2;
	double fa = (this->*func)(a,param1,param2), fb = (this->*func)(b,param1,param2);
	double fc, p, q, r, s, tol1, xm;

	if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
		die("root must be bracketed in Brent's Method");

	fc = fb;
	for (int it=0; it < itmax; it++) {
		if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
			c = a;
			fc = fa;
			e = d = b-a;
		}
		if (fabs(fc) < fabs(fb)) {
			a = b;
			b = c;
			c = a;
			fa = fb;
			fb = fc;
			fc = fa;
		}
		tol1 = 2.0*eps*fabs(b) + 0.5*tol;
		xm = 0.5*(c-b);
		if (fabs(xm) <= tol1 || fb == 0.0) return b;

		if (fabs(e) >= tol1 && fabs(fa) > fabs(fb))
		{
			s = fb / fa;
			if (a == c) {
				p = 2.0*xm*s;
				q = 1.0 - s;
			} else {
				q = fa/fc;
				r = fb/fc;
				p = s * (2.0*xm*q*(q-r) - (b-a)*(r-1.0));
				q = (q-1.0)*(r-1.0)*(s-1.0);
			}
			if (p > 0.0) q = -q;
			p = fabs(p);
			min1 = 3.0*xm*q - fabs(tol1*q);
			min2 = fabs(e*q);
			if (2.0*p < (min1 < min2 ? min1 : min2)) {
				e = d;
				d = p/q;
			} else {
				d = xm;
				e = d;
			}
		} else {
			d = xm;
			e = d;
		}
		a = b;
		fa = fb;
		if (fabs(d) > tol1)
			b += d;
		else
			b += SIGN(tol1,xm);
		fb = (this->*func)(b,param1,param2);
	}
	die("maximum number of iterations exceeded in Brent's Method");
	return 0.0;
}

bool Brent::BrentsMethod(double (Brent::*func)(const double), double &root, const double x1, const double x2, const double tol)
{
	double a = x1, b = x2, c = x2, d, e, min1, min2;
	double fa = (this->*func)(a), fb = (this->*func)(b);
	double fc, p, q, r, s, tol1, xm;

	if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0)) {
		if (fabs(fa) < fabs(fb)) root = a;
		else root = b;
		return false;
	}


	fc = fb;
	for (int it=0; it < itmax; it++) {
		if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
			c = a;
			fc = fa;
			e = d = b-a;
		}
		if (fabs(fc) < fabs(fb)) {
			a = b;
			b = c;
			c = a;
			fa = fb;
			fb = fc;
			fc = fa;
		}
		tol1 = 2.0*eps*fabs(b) + 0.5*tol;
		xm = 0.5*(c-b);
		if (fabs(xm) <= tol1 || fb == 0.0) {
			root = b;
			return true;
		}

		if (fabs(e) >= tol1 && fabs(fa) > fabs(fb))
		{
			s = fb / fa;
			if (a == c) {
				p = 2.0*xm*s;
				q = 1.0 - s;
			} else {
				q = fa/fc;
				r = fb/fc;
				p = s * (2.0*xm*q*(q-r) - (b-a)*(r-1.0));
				q = (q-1.0)*(r-1.0)*(s-1.0);
			}
			if (p > 0.0) q = -q;
			p = fabs(p);
			min1 = 3.0*xm*q - fabs(tol1*q);
			min2 = fabs(e*q);
			if (2.0*p < (min1 < min2 ? min1 : min2)) {
				e = d;
				d = p/q;
			} else {
				d = xm;
				e = d;
			}
		} else {
			d = xm;
			e = d;
		}
		a = b;
		fa = fb;
		if (fabs(d) > tol1)
			b += d;
		else
			b += SIGN(tol1,xm);
		fb = (this->*func)(b);
	}
	die("maximum number of iterations exceeded in Brent's Method");
	return false;
}

double Brent::BrentsMethod_Inclusive(double (Brent::*func)(const double), const double x1, const double x2, const double tol, const bool verbose)
{
	double a = x1, b = x2, c = x2, d, e, min1, min2;
	double fa = (this->*func)(a), fb = (this->*func)(b);
	double fc, p, q, r, s, tol1, xm;

	if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0)) {
		if (verbose) warn("root must be bracketed in Brent's Method");
		if (fabs(fa) < fabs(fb)) return a;
		else return b;
	}

	fc = fb;
	for (int it=0; it < itmax; it++) {
		if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
			c = a;
			fc = fa;
			e = d = b-a;
		}
		if (fabs(fc) < fabs(fb)) {
			a = b;
			b = c;
			c = a;
			fa = fb;
			fb = fc;
			fc = fa;
		}
		tol1 = 2.0*eps*fabs(b) + 0.5*tol;
		xm = 0.5*(c-b);
		if (fabs(xm) <= tol1 || fb == 0.0) return b;

		if (fabs(e) >= tol1 && fabs(fa) > fabs(fb))
		{
			s = fb / fa;
			if (a == c) {
				p = 2.0*xm*s;
				q = 1.0 - s;
			} else {
				q = fa/fc;
				r = fb/fc;
				p = s * (2.0*xm*q*(q-r) - (b-a)*(r-1.0));
				q = (q-1.0)*(r-1.0)*(s-1.0);
			}
			if (p > 0.0) q = -q;
			p = fabs(p);
			min1 = 3.0*xm*q - fabs(tol1*q);
			min2 = fabs(e*q);
			if (2.0*p < (min1 < min2 ? min1 : min2)) {
				e = d;
				d = p/q;
			} else {
				d = xm;
				e = d;
			}
		} else {
			d = xm;
			e = d;
		}
		a = b;
		fa = fb;
		if (fabs(d) > tol1)
			b += d;
		else
			b += SIGN(tol1,xm);
		fb = (this->*func)(b);
	}
	die("maximum number of iterations exceeded in Brent's Method");
	return 0.0;
}

double BrentsMethod(double (*func)(const double), const double x1, const double x2, const double tol)
{
	double a = x1, b = x2, c = x2, d, e, min1, min2;
	double fa = (*func)(a), fb = (*func)(b);
	double fc, p, q, r, s, tol1, xm;

	if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
		die("root must be bracketed in zbrent");

	fc = fb;
	for (int it=0; it < itmax; it++) {
		if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
			c = a;
			fc = fa;
			e = d = b-a;
		}
		if (fabs(fc) < fabs(fb)) {
			a = b;
			b = c;
			c = a;
			fa = fb;
			fb = fc;
			fc = fa;
		}
		tol1 = 2.0*eps*fabs(b) + 0.5*tol;
		xm = 0.5*(c-b);
		if (fabs(xm) <= tol1 || fb == 0.0) return b;

		if (fabs(e) >= tol1 && fabs(fa) > fabs(fb))
		{
			s = fb / fa;
			if (a == c) {
				p = 2.0*xm*s;
				q = 1.0 - s;
			} else {
				q = fa/fc;
				r = fb/fc;
				p = s * (2.0*xm*q*(q-r) - (b-a)*(r-1.0));
				q = (q-1.0)*(r-1.0)*(s-1.0);
			}
			if (p > 0.0) q = -q;
			p = fabs(p);
			min1 = 3.0*xm*q - fabs(tol1*q);
			min2 = fabs(e*q);
			if (2.0*p < (min1 < min2 ? min1 : min2)) {
				e = d;
				d = p/q;
			} else {
				d = xm;
				e = d;
			}
		} else {
			d = xm;
			e = d;
		}
		a = b;
		fa = fb;
		if (fabs(d) > tol1)
			b += d;
		else
			b += SIGN(tol1,xm);
		fb = (*func)(b);
	}
	die("maximum number of iterations exceeded in zbrent");
	return 0.0;
}

double BrentsMethod_Inclusive(double (*func)(const double), const double x1, const double x2, const double tol)
{
	const int itmax = 100;
	const double eps = 3.0e-8;

	double a = x1,b = x2,c = x2, d, e, min1, min2;
	double fa = (*func)(a), fb = (*func)(b);
	double fc, p, q, r, s, tol1, xm;

	if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0)) {
		warn("Root not bracketed in Brent's method");
		if (fabs(fa) < fabs(fb)) return a;
		else return b;
	}
	fc = fb;
	for (int it=1; it <= itmax; it++) {
		if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
			c = a;
			fc = fa;
			e = d = b-a;
		}
		if (fabs(fc) < fabs(fb)) {
			a = b;
			b = c;
			c = a;
			fa = fb;
			fb = fc;
			fc = fa;
		}
		tol1 = 2.0*eps*fabs(b)+0.5*tol;
		xm = 0.5*(c-b);
		if (fabs(xm) <= tol1 || fb == 0.0) return b;
		if (fabs(e) >= tol1 && fabs(fa) > fabs(fb)) {
			s = fb/fa;
			if (a == c) {
				p = 2.0*xm*s;
				q = 1.0 - s;
			} else {
				q = fa/fc;
				r = fb/fc;
				p = s * (2.0*xm*q*(q-r) - (b-a)*(r-1.0));
				q = (q-1.0)*(r-1.0)*(s-1.0);
			}
			if (p > 0.0) q = -q;
			p = fabs(p);
			min1 = 3.0*xm*q - fabs(tol1*q);
			min2 = fabs(e*q);
			if (2.0*p < (min1 < min2 ? min1 : min2)) {
				e = d;
				d = p/q;
			} else {
				d = xm;
				e = d;
			}
		} else {
			d = xm;
			e = d;
		}
		a = b;
		fa = fb;
		if (fabs(d) > tol1)
			b += d;
		else
			b += (xm >= 0.0 ? fabs(tol1) : -fabs(tol1));
		fb = (*func)(b);
	}
	die("Maximum number of iterations exceeded in Brent's Method");
	return 0.0;
}
