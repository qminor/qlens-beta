#include "powell.h"
#include "errors.h"
#include <iostream>
#include <csignal>
#include <cmath>
using namespace std;

const int Powell::ITMAX = 100;
const double Powell::CGOLD = 0.3819660;
const double Powell::ZEPS = 1.0e-10;
const double Powell::tol = 3e-4;

inline void SWAP(double &a, double &b)
	{double dum=a; a=b; b=dum;}
inline double SIGN(const double &a, const double &b)
	{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}
inline double MAX(const double &a, const double &b) { return (a > b ? a : b); }
inline double MIN(const double &a, const double &b) { return (a < b ? a : b); }


int POWELL_KEEP_RUNNING = 1;

void powell_sighandler(int sig)
{
	POWELL_KEEP_RUNNING = 0;
}

void powell_quitproc(int sig)
{
	exit(0);
}

void Powell::powell_minimize(double* pp, const int nn)
{
	n = nn;
	double **ximat;
	ximat = new double*[n];
	for (int i=0; i < n; i++) {
		ximat[i] = new double[n];
		for (int j=0; j < n; j++) {
			ximat[i][j] = 0;
		}
		ximat[i][i] = 1.0;
	}
	minimize(pp,ximat);
	for (int i=0; i < n; i++) delete[] ximat[i];
	delete[] ximat;
}

void Powell::powell_minimize(double* pp, const int nn, double* initial_stepsizes)
{
	POWELL_KEEP_RUNNING = 1;
	n = nn;
	double **ximat;
	ximat = new double*[n];
	for (int i=0; i < n; i++) {
		ximat[i] = new double[n];
		for (int j=0; j < n; j++) {
			ximat[i][j] = 0;
		}
		ximat[i][i] = initial_stepsizes[i];
	}
	minimize(pp,ximat);
	for (int i=0; i < n; i++) delete[] ximat[i];
	delete[] ximat;
}

void Powell::minimize(double* pp, double** ximat)
{
	POWELL_KEEP_RUNNING = 1;
	powell_exit_status = true;
	const int max_iterations = 200;
	const double TINY = 1.0e-25;
	double fptt;
	xt = new double[n];
	xi = new double[n];
	p = new double[n];
	for (int j=0; j < n; j++) p[j] = pp[j];
	double* ptt = new double[n];
	double* pt = new double[n];
	fret = (this->*func)(p);
	for (int j=0; j < n; j++) pt[j] = p[j];
	for (iter=0;; ++iter)
	{
		if (powell_exit_status==false) break;
		double fp = fret;
		int ibig = 0;
		double del = 0.0;
		for (int i=0; i < n; i++)
		{
			for (int j=0; j < n; j++) xi[j] = ximat[j][i];
			fptt = fret;
			fret = linemin();
			if (fptt-fret > del) {
				del = fptt - fret;
				ibig = i + 1;
			}
			if (powell_exit_status==false) break;
		}
		double hi = 2.0*(fp-fret);
		double hi2 = ftol*(abs(fp) + abs(fret)) + TINY;
		if ((2.0*(fp-fret) <= ftol*(abs(fp) + abs(fret)) + TINY) or (powell_exit_status==false)) {
			for (int j=0; j < n; j++) pp[j] = p[j];
			return;
		}
		if (iter == max_iterations) die("powell exceeding maximum iterations.");
		if (powell_exit_status==false) break;
		for (int j=0; j < n; j++)
		{
			ptt[j] = 2.0*p[j] - pt[j];
			xi[j] = p[j] - pt[j];
			pt[j] = p[j];
		}
		fptt = (this->*func)(ptt);
		signal(SIGABRT, &powell_sighandler);
		signal(SIGTERM, &powell_sighandler);
		signal(SIGINT, &powell_sighandler);
		signal(SIGUSR1, &powell_sighandler);
		signal(SIGQUIT, &powell_quitproc);
		if (!POWELL_KEEP_RUNNING) { powell_exit_status = false; break; }
		if (fptt < fp)
		{
			double tmp=fp-fret-del, tmp2=fp-fptt;
			double t = 2.0*(fp-2.0*fret+fptt)*tmp*tmp - del*tmp2*tmp2;
			if (t < 0.0)
			{
				fret = linemin();
				for (int j=0; j < n; j++) {
					ximat[j][ibig-1] = ximat[j][n-1];
					ximat[j][n-1] = xi[j];
				}
			}
		}
	}
	delete[] xt;
	delete[] xi;
	delete[] p;
	delete[] ptt;
	delete[] pt;
}

double Powell::f1dim(const double x)
{
	for (int j=0; j<n; j++) {
		xt[j] = p[j] + x*xi[j];
	}
	double ans = (this->*func)(xt);
	signal(SIGABRT, &powell_sighandler);
	signal(SIGTERM, &powell_sighandler);
	signal(SIGINT, &powell_sighandler);
	signal(SIGUSR1, &powell_sighandler);
	signal(SIGQUIT, &powell_quitproc);
	if (!POWELL_KEEP_RUNNING) powell_exit_status = false;
	return ans;
}

void Powell::bracket(const double a, const double b)
{
	const double GOLD=1.618034, GLIMIT=100.0, TINY=1.0e-20;
	ax = a; bx = b;
	double fu;
	fa = f1dim(ax);
	fb = f1dim(bx);
	if (fb > fa) {
		SWAP(ax,bx);
		SWAP(fb,fa);
	}
	cx = bx + GOLD*(bx-ax);
	fc = f1dim(cx);

	while (fb > fc)
	{
		double r = (bx-ax)*(fb-fc);
		double q = (bx-cx)*(fb-fa);
		double u = bx - ((bx-cx)*q-(bx-ax)*r)/(2.0*SIGN(MAX(abs(q-r),TINY),q-r));
		double ulim = bx + GLIMIT*(cx-bx);
		if ((bx-u)*(u-cx) > 0.0) {
			fu = f1dim(u);
			if (fu < fc) {
				ax = bx;
				bx = u;
				fa = fb;
				fb = fu;
				return;
			} else if (fu > fb) {
				cx = u;
				fc = fu;
				return;
			}
			u = cx + GOLD*(cx-bx);
			fu = f1dim(u);
		} else if ((cx-u)*(u-ulim) > 0.0) {
			fu = f1dim(u);
			if (fu < fc) {
				shft3(bx,cx,u,u+GOLD*(u-cx));
				shft3(fb,fc,fu,f1dim(u));
			}
		} else if ((u-ulim)*(ulim-cx) >= 0.0) {
			u = ulim;
			fu = f1dim(u);
		} else {
			u = cx + GOLD*(cx-bx);
			fu = f1dim(u);
		}
		shft3(ax,bx,cx,u);
		shft3(fa,fb,fc,fu);
	}
}

double Powell::minimize(void)
{
	double a,b,d=0.0,etemp,fu,fv,fw,fx;
	double p,q,r,tol1,tol2,u,v,w,x,xm;
	double e=0.0;
	double fx_old=1e30;
	const double TINY=1.0e-25;
	
	a = MIN(ax,cx);
	b = MAX(ax,cx);
	x=w=v=bx;
	fw=fv=fx=f1dim(x);
	for (int iter=0; iter < ITMAX; iter++)
	{
		if (powell_exit_status==false) return xmin=x;
		xm=0.5*(a+b);
		tol1 = tol*abs(x) + ZEPS;
		tol2 = 2.0 * tol1;
		// The following modification allows it to converge based on the function value criterion ftol
		// (although if a better function value cannot be found, it can still converge based on x as well).
		if ((2.0*abs(fx-fx_old) <= ftol*(abs(fx) + abs(fx_old) + TINY)) or (abs(x-xm) <= (tol2-0.5*(b-a)))) {
		//if (abs(x-xm) <= (tol2-0.5*(b-a))) open bracket here (this is the original version)
			fmin = fx;
			return xmin = x;
		}
		if (abs(e) > tol1) {
			r = (x-w)*(fx-fv);
			q = (x-v)*(fx-fw);
			p = (x-v)*q - (x-w)*r;
			q = 2.0*(q-r);
			if (q > 0.0) p = -p;
			q = abs(q);
			etemp = e;
			e = d;
			if (abs(p) >= abs(0.5*q*etemp) or p <= q*(a-x) or p >= q*(b-x))
				d = CGOLD*(e=(x >= xm ? a-x : b-x));
			else {
				d = p/q;
				u = x + d;
				if (u-a < tol2 or b-u < tol2)
					d = SIGN(tol1,xm-x);
			}
		} else {
			d = CGOLD*(e=(x >= xm ? a-x : b-x));
		}
		u = (abs(d) >= tol1 ? x+d : x + SIGN(tol1,d));
		fu = f1dim(u);
		if (fu <= fx) {
			fx_old = fx;
			if (u >= x) a=x; else b=x;
			shft3(v,w,x,u);
			shft3(fv,fw,fx,fu);
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw or w == x) {
				v = w;
				w = u;
				fv = fw;
				fw = fu;
			} else if (fu <= fv or v == x or v == w) {
				v = u;
				fv = fu;
			}
		}
	}
	die("Too many iterations in Brent's Method (for minimizing)");
}

double Powell::linemin()
{
	double xmin;
	bracket(0.0,1.0);
	xmin=minimize();
	for (int j=0; j < n; j++)
	{
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	return fmin;
}


