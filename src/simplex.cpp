#include "simplex.h"
#include "rand.h"
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace std;

const double Simplex::t0_default = 1.0;
const double Simplex::tfinal_default = 1e-10;
const double Simplex::tinc_default = 0.9;
const int Simplex::max_iterations_default = 1000;
const int Simplex::max_iterations_anneal_default = 1000;
bool Simplex::simplex_display_bestfit_point;

int SIMPLEX_KEEP_RUNNING = 1;

void simplex_sighandler(int sig)
{
	SIMPLEX_KEEP_RUNNING = 0;
}

void simplex_quitproc(int sig)
{
	exit(0);
}

void Simplex::initialize_simplex(double* point, const int& ndim_in, double* vertex_displacements, const double& ftol_in)
{
	if (initialized) {
		delete[] y;
		delete[] pb;
		for (int i=0; i < mpts; i++) delete[] p[i];
		delete[] p;
		delete[] disps;
	}
	ftol = ftol_in;
	yb = 1e30;
	ndim = ndim_in;
	mpts = ndim+1;
	max_iterations = max_iterations_default;
	max_iterations_anneal = max_iterations_anneal_default;
	fmin = -1e30;
	pb = new double[ndim];
	y = new double[mpts];
	p = new double*[mpts];
	disps = new double[ndim];
	
	int i,j;
	for (j=0; j < ndim; j++) disps[j] = vertex_displacements[j];
	for (i=0; i < mpts; i++) {
		p[i] = new double[ndim];
		for (j=0; j < ndim; j++)
			p[i][j]=point[j];
		if (i != 0) p[i][i-1] += vertex_displacements[i-1];
	}
	if (!initialized) initialized = true;
	simplex_exit_status = true;
}

void Simplex::downhill_simplex(int &nfunk, const int& nmax, const double& temperature)
{
	// the following code is from Numerical Recipes in C (slightly modified)
	SIMPLEX_KEEP_RUNNING = 1;
	const double tiny=1e-10;
	int i,j,ihi,ilo,inhi;
	double rtol,ysave,yt,ytry;
	double ylo,yhi,ynhi;
	int itmax = (temperature==0) ? nmax : max_iterations_anneal;

	if (yb < 1e30) {
		// reset simplex to start from the best-fit point from previous iterations
		for (i=0; i < mpts; i++) {
			for (j=0; j < ndim; j++) p[i][j] = pb[j];
			if (i != 0) p[i][i-1] += disps[i-1];
		}
	}

	for (i=0;i<mpts;i++) {
		y[i]=(this->*func)(p[i]);
		signal(SIGABRT, &simplex_sighandler);
		signal(SIGTERM, &simplex_sighandler);
		signal(SIGINT, &simplex_sighandler);
		signal(SIGUSR1, &simplex_sighandler);
		signal(SIGQUIT, &simplex_quitproc);
		if (!SIMPLEX_KEEP_RUNNING) { simplex_exit_status = false; break; }
	}

	if (yb == 1e30) {
		// record initial point as best-fit point thus far, just in case the random-walk takes us too far away
		for (j=0;j<ndim;j++) pb[j]=p[0][j];
		yb=y[0];
	}

	nfunk=0;
	tt = -temperature;
	double *psum = new double[ndim];
	get_psum(psum);
	for (;;) {
		if (!SIMPLEX_KEEP_RUNNING) { simplex_exit_status = false; break; }
		ilo=0;
		ihi = y[0] > y[1] ? (inhi=1,0) : (inhi=0,1);
		ynhi=y[inhi];
		ylo=y[ilo];
		yhi=y[ihi];
		for (i=0; i < mpts; i++) {
			yt = y[i];
			if (tt != 0.0) yt += tt*log(RandomNumber());
			if (yt <= y[ilo]) {
				ilo=i;
				ylo=yt;
			}
			if (yt > yhi) {
				inhi=ihi;
				ynhi=yhi;
				ihi=i;
				ynhi=yhi;
				yhi=yt;
			} else if ((yt > ynhi) and (i != ihi)) {
				inhi=i;
				ynhi=yt;
			}
		}
		rtol = 2.0*abs(yhi-ylo)/(abs(yhi)+abs(ylo)+tiny);
		if (rtol < ftol) {
			temp=y[0]; y[0]=y[ilo]; y[ilo]=temp;
			for (i=0; i < ndim; i++) {
				temp=p[0][i]; p[0][i]=p[ilo][i]; p[ilo][i]=temp;
			}
			break;
		}
		if (nfunk >= itmax) {
			delete[] psum;
			if (temperature==0) cout << "\n*WARNING*: Exceeded maximum number of iterations (" << itmax << ")" << endl;
			return;
		}
		nfunk += 2;
		ytry=amotry(psum,ihi,yhi,-1.0);
		if (!SIMPLEX_KEEP_RUNNING) { simplex_exit_status = false; break; }
		if (ytry <= ylo) {
			ytry=amotry(psum,ihi,yhi,2.0);
			if (!SIMPLEX_KEEP_RUNNING) { simplex_exit_status = false; break; }
		} else if (ytry >= ynhi) {
			ysave=yhi;
			ytry=amotry(psum,ihi,yhi,0.5);
			if (!SIMPLEX_KEEP_RUNNING) { simplex_exit_status = false; break; }
			if (ytry >= ysave) {
				for (i=0; i < mpts; i++) {
					if (!SIMPLEX_KEEP_RUNNING) { simplex_exit_status = false; break; }
					if (i != ilo) {
						for (j=0; j < ndim; j++)
							p[i][j] = psum[j] = 0.5*(p[i][j]+p[ilo][j]);
						y[i] = (this->*func)(psum);
						signal(SIGABRT, &simplex_sighandler);
						signal(SIGTERM, &simplex_sighandler);
						signal(SIGINT, &simplex_sighandler);
						signal(SIGUSR1, &simplex_sighandler);
						signal(SIGQUIT, &simplex_quitproc);
					}
				}
				nfunk += ndim;
				get_psum(psum);
			}
		} else nfunk--;

		signal(SIGABRT, &simplex_sighandler);
		signal(SIGTERM, &simplex_sighandler);
		signal(SIGINT, &simplex_sighandler);
		signal(SIGUSR1, &simplex_sighandler);
		signal(SIGQUIT, &simplex_quitproc);

		if (!SIMPLEX_KEEP_RUNNING) {
			simplex_exit_status = false;
			break;
		}
	}
	delete[] psum;
}

double Simplex::amotry(double* psum, const int &ihi, double& yhi, const double& fac)
{
	// the following code is from Numerical Recipes in C
	int j;
	double fac1,fac2,yflu,ytry;
	double ptry[ndim];
	fac1=(1.0-fac)/ndim;
	fac2=fac1-fac;
	for (j=0; j < ndim; j++) {
		ptry[j]=psum[j]*fac1 - p[ihi][j]*fac2;
	}
	ytry=(this->*func)(ptry);
	if (ytry <= yb) {
		if (ytry > fmin) {
			for (j=0;j<ndim;j++) pb[j]=ptry[j];
			yb=ytry;
		} else {
			SIMPLEX_KEEP_RUNNING = 0;
		}
	}
	yflu=ytry;
	if (tt != 0.0) yflu -= tt*log(RandomNumber());
	if (yflu < yhi) {
		y[ihi] = ytry;
		yhi = yflu;
		for (j=0; j < ndim; j++) {
			psum[j] += ptry[j] - p[ihi][j];
			p[ihi][j] = ptry[j];
		}
	}
	signal(SIGABRT, &simplex_sighandler);
	signal(SIGTERM, &simplex_sighandler);
	signal(SIGINT, &simplex_sighandler);
	signal(SIGUSR1, &simplex_sighandler);
	signal(SIGQUIT, &simplex_quitproc);
	return yflu;
}


int Simplex::downhill_simplex_anneal(bool verbal)
{
	simplex_exit_status = true;
	double t = t0;
	int iterations;
	if (t > 0) {
		if ((verbal) and (t >= tfinal)) {
			cout << "\033[1A" << "temperature=" << t << endl;
		}
		while (t >= tfinal) {
			if (simplex_exit_status==false) break;
			iterations=0;
			downhill_simplex(iterations,max_iterations,t);
			if (simplex_exit_status==false) break;

			if (yb < fmin_anneal) t = 0;
			else t *= tinc;
			if (verbal) {
				if (simplex_display_bestfit_point) {
					cout << "\033[1A" << "temperature=" << t << ", best(-2*loglike)=" << (2*yb) << "       " << endl;
					cout << "best-fit point: (";
					for (int i=0; i < ndim; i++) {
						cout << pb[i];
						if (i < ndim-1) cout << ",";
					}
					cout << ")      " << endl;
					cout << "\033[1A" << flush;
				} else {
					cout << "\033[1A" << "temperature=" << t << ", best(-2*loglike)=" << yb << "       " << endl;
				}
			}
		}
		if (simplex_exit_status==false) return iterations;
	}
	downhill_simplex(iterations,max_iterations,0); // do final run with zero temperature
	return iterations;
}

Simplex::~Simplex()
{
	if (initialized) {
		delete[] y;
		delete[] pb;
		for (int i=0; i < mpts; i++) delete[] p[i];
		delete[] p;
		delete[] disps;
	}
}

