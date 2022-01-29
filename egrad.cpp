#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "errors.h"
#include "vector.h"
#include "egrad.h"
using namespace std;

#ifdef USE_FITPACK
extern "C" {
	void curfit_(int *iopt, int *m, double *x, double *y, double *w, double *xb, double *xe, int *order, double *s, int *bspline_nmax, int *n_knots, double *knots, double *coefs, double *minchisq, double *work, int *lwork, int *iwork, int *ier);
	void splev_(double *knots, int *n_knots, double *coefs, int *order, double *x, double *y, int *m, int *e, int *ier);
}
#endif

EllipticityGradient::EllipticityGradient()
{
	bspline_order = 3;
	ellipticity_gradient = false;
	fourier_gradient = false;
	egrad_mode = 0;
	egrad_ellipticity_mode = -1; // ellipticity gradient is off by default
	center_gradient = false;
	use_linear_xivals = false;
	xi_initial_egrad = 0.0;
	xi_final_egrad = 0.0;
	for (int i=0; i < 4; i++) {
		n_egrad_params[i] = 0;
		geometric_param[i] = NULL;
		geometric_knots[i] = NULL;
		geometric_param_ref[i] = 0;
		geometric_param_dif[i] = 0;
	}
	n_bspline_knots_tot = 0;
	angle_param_egrad = NULL;
	n_fourier_grad_modes = 0;
	fourier_grad_mvals = NULL;
	n_fourier_grad_params = NULL;
	fourier_param = NULL;
	fourier_knots = NULL;
	egrad_minq = 1e30;
}

EllipticityGradient::~EllipticityGradient()
{
	for (int i=0; i < 4; i++) {
		if (geometric_param[i] != NULL) delete[] geometric_param[i];
		if (geometric_knots[i] != NULL) delete[] geometric_knots[i];
	}
	if (angle_param_egrad != NULL) delete[] angle_param_egrad;
	if (fourier_gradient) {
		int i,j;
		for (i=0,j=0; i < n_fourier_grad_modes; i++) {
			delete[] fourier_param[j++];
			delete[] fourier_param[j++];
		}
		delete[] fourier_param;
		delete[] n_fourier_grad_params;
		if (fourier_knots != NULL) {
			for (i=0,j=0; i < n_fourier_grad_modes; i++) {
				delete[] fourier_knots[j++];
				delete[] fourier_knots[j++];
			}
			delete[] fourier_knots;
		}
	}
}

bool EllipticityGradient::setup_egrad_params(const int egrad_mode_in, const int ellipticity_mode_in, const dvector& egrad_params, int& n_egrad_params_tot, const int n_bspline_coefs, const dvector& knots, const double ximin, const double ximax, const double xiref, const bool linear_xivals)
{
#ifndef USE_FITPACK
	if (egrad_mode_in==0) {
		warn("qlens must be compiled with FITPACK in order to use B-spline mode");
		return false;
	}
#endif
	for (int i=0; i < 4; i++) {
		if (geometric_param[i] != NULL) delete[] geometric_param[i];
		if (geometric_knots[i] != NULL) delete[] geometric_knots[i];
	}
	if (angle_param_egrad != NULL) delete[] angle_param_egrad;
	ellipticity_gradient = true;
	egrad_mode = egrad_mode_in;
	egrad_ellipticity_mode = ellipticity_mode_in;
	use_linear_xivals = linear_xivals;
	int i,j,ip=0;
	if (egrad_mode==0) {
		n_egrad_params_tot=2*n_bspline_coefs+2;
		if (egrad_params.size() != n_egrad_params_tot) {
			warn("incorrect number of efunc_params");
			ellipticity_gradient = false;
			return false;
		}
		n_egrad_params[0] = n_bspline_coefs; // q
		n_egrad_params[1] = n_bspline_coefs; // theta
		n_egrad_params[2] = 1; // xc
		n_egrad_params[3] = 1; // yc
		angle_param_egrad = new bool[n_egrad_params[1]]; // keeps track of which parameters are angles, so they can be converted to degrees when displayed
		for (i=0; i < n_egrad_params[1]; i++) angle_param_egrad[i] = true; // all theta B-spline coefficients will have units of angle

		for (i=0; i < 4; i++) {
			geometric_param[i] = new double[n_egrad_params[i]];
			for (j=0; j < n_egrad_params[i]; j++) {
				if ((i==1) and (angle_param_egrad[j])) geometric_param[i][j] = degrees_to_radians(egrad_params[ip++]);
				else geometric_param[i][j] = egrad_params[ip++];
			}
		}
		if (ip != egrad_params.size()) die("we fucked up the egrad_params, wrong number of arguments");
		n_bspline_knots_tot = n_bspline_coefs + bspline_order + 1;
		int n_unique_knots = n_bspline_coefs - bspline_order + 1;
		double bspline_logximin = log(ximin)/ln10;
		double bspline_logximax = log(ximax)/ln10;
		double logxi, logxistep = (bspline_logximax-bspline_logximin)/(n_unique_knots-1);
		double xi, xistep = (ximax-ximin)/(n_unique_knots-1);
		if (knots[0]==-1e30) {
			// Set up the knot vectors
			for (i=0; i < 4; i++) {
				geometric_knots[i] = new double[n_bspline_knots_tot];
				for (j=0; j < bspline_order; j++) geometric_knots[i][j] = bspline_logximin;
				if (!use_linear_xivals) {
					for (j=0, logxi=bspline_logximin; j < n_unique_knots; j++, logxi += logxistep) geometric_knots[i][j+bspline_order] = logxi;
				} else {
					for (j=0, xi=ximin; j < n_unique_knots; j++, xi += xistep) {
						geometric_knots[i][j+bspline_order] = log(xi)/ln10;
						//cout << pow(10,geometric_knots[i][j+bspline_order]) << endl;
					}
				}
				for (j=0; j < bspline_order; j++) geometric_knots[i][n_bspline_knots_tot-bspline_order+j] = bspline_logximax;
			}
		} else {
			ip=0;
			for (i=0; i < 4; i++) {
				geometric_knots[i] = new double[n_bspline_knots_tot];
				if ((i < 2) or (center_gradient)) {
					for (j=0; j < bspline_order; j++) geometric_knots[i][j] = log(knots[ip])/ln10;
					for (j=0; j < n_unique_knots; j++) geometric_knots[i][j+bspline_order] = log(knots[ip++])/ln10;
					for (j=0; j < bspline_order; j++) geometric_knots[i][n_bspline_knots_tot-bspline_order+j] = log(knots[ip-1])/ln10;
				}
			}
			if (ip != knots.size()) die("we fucked up the knot vector, wrong number of arguments");
		}

		xi_initial_egrad = ximin;
		xi_final_egrad = ximax;
		egrad_minq = 1e30;
		int n_xisteps_qmin = 50;
		//for (int i=0; i < n_egrad_params[0]; i++) if (geometric_param[0][i] < egrad_minq) egrad_minq = geometric_param[0][i];
		double qq, logxistep_q = (bspline_logximax-bspline_logximin)/(n_xisteps_qmin-1);
		set_egrad_ptr();
		for (i=0, logxi = bspline_logximin; i < n_xisteps_qmin; i++, logxi += logxistep_q) {
			xi = pow(10.0,logxi);
			qq = (this->*egrad_ptr)(xi,geometric_param[0],0);
			if (qq < egrad_minq) egrad_minq = qq;
		}
	} else if ((egrad_mode==1) or (egrad_mode==2)) {
		n_egrad_params_tot=10;
		if (egrad_params.size() != n_egrad_params_tot) {
			warn("incorrect number of efunc_params");
			return false;
		}
		n_egrad_params[0] = 4; // q
		n_egrad_params[1] = 4; // theta
		n_egrad_params[2] = 1; // xc
		n_egrad_params[3] = 1; // yc
		angle_param_egrad = new bool[n_egrad_params[1]]; // keeps track of which parameters are angles, so they can be converted to degrees when displayed
		angle_param_egrad[0] = true; // theta_i
		angle_param_egrad[1] = true; // theta_f
		angle_param_egrad[2] = false;
		angle_param_egrad[3] = false;
		for (i=0; i < 4; i++) {
			geometric_param[i] = new double[n_egrad_params[i]];
		}

		if (egrad_mode==1) {
			for (i=0; i < 4; i++) {
				for (j=0; j < n_egrad_params[i]; j++) {
					if ((i==1) and (angle_param_egrad[j])) geometric_param[i][j] = degrees_to_radians(egrad_params[ip++]);
					else geometric_param[i][j] = egrad_params[ip++];
				}
			}
			if ((geometric_param[0][0] <= 0) or (geometric_param[0][0] > 1) or (geometric_param[0][1] <= 0) or (geometric_param[0][1] > 1)) die("qi and qf must be between 0 and 1 in egrad mode 1");
		} else { // egrad_mode==2
			for (i=0; i < 4; i++) {
				for (j=0; j < n_egrad_params[i]; j++) {
					if (j==0) {
						if (i==1) geometric_param_ref[i] = degrees_to_radians(egrad_params[ip++]);
						else geometric_param_ref[i] = egrad_params[ip++];
					} else if (j==1) { 
						if (i==1) geometric_param_dif[i] = degrees_to_radians(egrad_params[ip++]);
						else geometric_param_dif[i] = egrad_params[ip++];
					} else {
						geometric_param[i][j] = egrad_params[ip++];
					}
				}
			}
			xi_ref_egrad = xiref;
			double step_ref;
			for (i=0; i < 4; i++) {
				if (n_egrad_params[i]==1) geometric_param[i][0] = geometric_param_ref[i];
				else {
					step_ref = tanh((xi_ref_egrad-geometric_param[i][2])/geometric_param[i][3]);
					geometric_param[i][0] = geometric_param_ref[i] - (1+step_ref)*geometric_param_dif[i]/2;
					geometric_param[i][1] = geometric_param_ref[i] + (1-step_ref)*geometric_param_dif[i]/2;
				}
			}
			if (geometric_param[0][0] > 1.0) geometric_param[0][0] = 1.0; // q cannot be greater than 1
			if (geometric_param[0][1] > 1.0) geometric_param[0][1] = 1.0; // q cannot be greater than 1
			if (geometric_param[0][0] < 0.001) geometric_param[0][0] = 0.001;
			if (geometric_param[0][1] < 0.001) geometric_param[0][1] = 0.001;
			//cout << "qi=" << geometric_param[0][0] << ", qf=" << geometric_param[0][1] << endl;
			//cout << "theta_i=" << radians_to_degrees(geometric_param[1][0]) << ", theta_f=" << radians_to_degrees(geometric_param[1][1]) << endl;
			//cout << "xc=" << geometric_param[2][0] << endl;
			//cout << "yc=" << geometric_param[3][0] << endl;
		}
		if (ip != egrad_params.size()) die("we fucked up the egrad_params, wrong number of arguments");

		double xi0_q, xif_q, xi0_th, xif_th;
		xi0_q = geometric_param[0][2] - 3*geometric_param[0][3];
		if (xi0_q < 0) xi0_q = 0;
		xif_q = geometric_param[0][2] + 3*geometric_param[0][3];

		xi0_th = geometric_param[1][2] - 3*geometric_param[1][3];
		if (xi0_th < 0) xi0_th = 0;
		xif_th = geometric_param[1][2] + 3*geometric_param[1][3];
		xi_initial_egrad = dmin(xi0_q,xi0_th);
		if (xi_initial_egrad < 0) xi_initial_egrad = 0;
		xi_final_egrad = dmax(xif_q,xif_th);
		egrad_minq = (geometric_param[0][1] < geometric_param[0][0]) ? geometric_param[0][1] : geometric_param[0][0];
	} else {
		warn("only egrad_mode=0,1,2 currently supported");
		ellipticity_gradient = false;
		return false;
	}
	set_egrad_ptr();
	return true;
}

bool EllipticityGradient::setup_fourier_grad_params(const int n_modes, const ivector& mvals, const dvector& fourier_grad_params, int& n_fourier_grad_params_tot)
{
	if (ellipticity_gradient==false) {
		// Is this truly necessary? Might be nice to make them independent
		warn("egrad mode must be on in order to include Fourier mode gradients");
		return false;
	}
	if (n_modes==0) {
		warn("Fourier modes must already be present to enable Fourier gradient mode");
		return false;
	}
	int i,j;
	if (fourier_param != NULL) {
		for (i=0,j=0; i < n_fourier_grad_modes; i++) {
			delete[] fourier_param[j++];
			delete[] fourier_param[j++];
		}
		delete[] fourier_param;
	}
	if (n_fourier_grad_params != NULL) delete[] n_fourier_grad_params;
	if (fourier_knots != NULL) {
		for (i=0,j=0; i < n_fourier_grad_modes; i++) {
			delete[] fourier_knots[j++];
			delete[] fourier_knots[j++];
		}
		delete[] fourier_knots;
	}

	fourier_gradient = true;
	n_fourier_grad_modes = n_modes;
	n_fourier_grad_params = new int[n_modes];
	int n_amps = n_modes*2;
	int k,ip=0;
	if (egrad_mode==0) {
		int n_bspline_coefs = n_bspline_knots_tot - bspline_order - 1;
		n_fourier_grad_params_tot = n_amps*n_bspline_coefs;
		for (i=0; i < n_modes; i++) {
			n_fourier_grad_params[i] = n_bspline_coefs;
		}
	} else if (egrad_mode==1) {
		n_fourier_grad_params_tot = n_amps*4;
		n_fourier_grad_params = new int[n_modes];
		for (i=0; i < n_modes; i++) {
			n_fourier_grad_params[i] = 4;
		}
	} else {
		warn("only egrad_mode=0 or 1 currently supported");
		delete[] n_fourier_grad_params;
		n_fourier_grad_params = NULL;
		fourier_gradient = false;
		return false;
	}

	if (fourier_grad_params.size() != n_fourier_grad_params_tot) {
		warn("incorrect number of fourier_grad_params");
		delete[] n_fourier_grad_params;
		n_fourier_grad_params = NULL;
		fourier_gradient = false;
		return false;
	}
	fourier_grad_mvals = mvals.array();
	fourier_param = new double*[n_amps];
	for (i=0,k=0; i < n_modes; i++, k += 2) {
		fourier_param[k] = new double[n_fourier_grad_params[i]];
		fourier_param[k+1] = new double[n_fourier_grad_params[i]];
		for (j=0; j < n_fourier_grad_params[i]; j++) {
			fourier_param[k][j] = fourier_grad_params[ip++];
		}
		for (j=0; j < n_fourier_grad_params[i]; j++) {
			fourier_param[k+1][j] = fourier_grad_params[ip++];
		}
	}
	if (ip != fourier_grad_params.size()) die("we fucked up size of fourier_grad_params");

	if (egrad_mode==0) {
		// Set up the knot vectors
		int n_unique_knots = n_bspline_knots_tot - 2*bspline_order;
		double bspline_logximin = log(xi_initial_egrad)/ln10;
		double bspline_logximax = log(xi_final_egrad)/ln10;
		double logxi, logxistep = (bspline_logximax-bspline_logximin)/(n_unique_knots-1);
		fourier_knots = new double*[n_amps];
		for (i=0; i < n_amps; i++) {
			fourier_knots[i] = new double[n_bspline_knots_tot];
			for (j=0; j < bspline_order; j++) fourier_knots[i][j] = bspline_logximin;
			for (j=0, logxi=bspline_logximin; j < n_unique_knots; j++, logxi += logxistep) fourier_knots[i][j+bspline_order] = logxi;
			for (j=0; j < bspline_order; j++) fourier_knots[i][n_bspline_knots_tot-bspline_order+j] = bspline_logximax;
		}
	}
	return true;
}

void EllipticityGradient::check_for_overlapping_contours()
{
	int i, j, n_contours = 100, npoints = 100;
	double xi, ximin, ximax, xistep;
	double qq, th, qprev, thprev, ep, costh, sinth;
	double phi, x, y;
	double phistep = M_2PI/(npoints-1);
	double xisqtest, xisqprev;
	double xprimesq, yprimesq;

	ximin = xi_initial_egrad;
	ximax = xi_final_egrad;
	xistep = (ximax-ximin)/(n_contours-1);
	if (ximin==0) {
		ximin = xistep;
		n_contours--;
	}

	qq = geometric_param[0][0];
	th = geometric_param[1][0];
	for (i=0, xi=ximin; i < n_contours; i++, xi += xistep) {
		qprev = qq;
		thprev = th;
		xisqprev = SQR(xi-xistep);
		ellipticity_function(xi,ep,th);
		qq = sqrt(1-ep);
		if (i==0) continue;
		costh = cos(th-thprev);
		sinth = sin(th-thprev);
		for (j=0, phi=0; j < npoints-1; j++, phi += phistep) {
			double xisqcheck;
			if (egrad_ellipticity_mode==0) {
				x = xi*cos(phi);
				y = xi*qq*sin(phi);
			} else {
				x = xi/sqrt(qq)*cos(phi);
				y = xi*sqrt(qq)*sin(phi);
			}
			// Now we rotate the coordinates so we can 
			xprimesq = SQR(x*costh - y*sinth);
			yprimesq = SQR(x*sinth + y*costh);
			if (egrad_ellipticity_mode==0) {
				xisqtest = xprimesq + yprimesq/(qprev*qprev);
			} else {
				xisqtest = qprev*xprimesq + yprimesq/(qprev);
			}
			if (xisqtest < xisqprev) {
				// a point in the contour is inside the previous ellipse (at smaller xi), indicating contours have crossed
				contours_overlap = true;
				return;
			}
		}
	}
	contours_overlap = false;
}

void EllipticityGradient::ellipticity_function(const double xi, double& ep, double& angle)
{
	ep = (this->*egrad_ptr)(xi,geometric_param[0],0); // NOTE!!! ep is actually the axis ratio q on this line!!!
	angle = (this->*egrad_ptr)(xi,geometric_param[1],1);
	ep = 1 - ep*ep; // this gets it in the epsilon form required for deflection formulas (remember 'ep' is the axis ratio before this line)
}

void EllipticityGradient::fourier_mode_function(const double xi, double* cosamp, double* sinamp)
{
	int i,j;
	for (i=0, j=0; i < n_fourier_grad_modes; i++, j += 2) {
		cosamp[i] = (this->*egrad_ptr)(xi,fourier_param[j],4+j);
		sinamp[i] = (this->*egrad_ptr)(xi,fourier_param[j+1],4+j+1);
	}
}

void EllipticityGradient::fourier_mode_function(const double xi, const int mval, double& cosamp, double& sinamp)
{
	int i,j;
	bool mval_found = false;
	for (i=0, j=0; i < n_fourier_grad_modes; i++, j += 2) {
		if (fourier_grad_mvals[i]==mval) {
			cosamp = (this->*egrad_ptr)(xi,fourier_param[j],4+j);
			sinamp = (this->*egrad_ptr)(xi,fourier_param[j+1],4+j+1);
			mval_found = true;
			break;
		}
	}
	if (!mval_found) die("mval not found");
}

double EllipticityGradient::egrad_tanh_function(const double xi, double *paramvals, const int param_index)
{
	//Note: param_index is not needed for this function
	double stepf = tanh((xi-paramvals[2])/paramvals[3]);
	return (paramvals[0]*(1-stepf) + paramvals[1]*(1+stepf))/2;
}

double EllipticityGradient::egrad_bspline_function(const double xi, double *paramvals, const int param_index)
{
	double ans = 0.0;
#ifdef USE_FITPACK
	int m = 1; // Evaluate a single point
	int e = 0;
	int ier = 0;

	double logxi;
	if (xi < xi_initial_egrad) logxi = log(xi_initial_egrad)/ln10;
	else if (xi > xi_final_egrad) logxi = log(xi_final_egrad)/ln10;
	else logxi = log(xi)/ln10;

	double *knots;
	if (param_index < 4) knots = geometric_knots[param_index];
	else knots = fourier_knots[param_index-4];

	splev_(knots, &n_bspline_knots_tot, paramvals, &bspline_order, &logxi, &ans, &m, &e, &ier);
	if (ier > 0) {
		stringstream s;
		s << "Error evaluating B-Spline curve using splev() at point " << xi << ": " << ier;
		throw runtime_error(s.str());
	}
	if ((param_index==0) and (ans > 1)) return 1.0; // in case something greater than q=1 is returned
#endif

	return ans;
}

void EllipticityGradient::allocate_bspline_work_arrays(const int n_data)
{
	bspline_nmax = n_data + bspline_order + 1;
	int lwork = (n_data * (bspline_order + 1) + bspline_nmax * (7 + 3*bspline_order));
	bspline_work  = new double[lwork];
	bspline_iwork = new int[bspline_nmax];
}

void EllipticityGradient::free_bspline_work_arrays()
{
	delete[] bspline_work;
	delete[] bspline_iwork;
}

double EllipticityGradient::fit_bspline_curve(double *knots, double *coefs)
{
	double minchisq = 0.0; // optimal chi-square that will be returned
#ifdef USE_FITPACK
	double logxi_initial = log(xi_initial_egrad)/ln10;
	double logxi_final = log(xi_final_egrad)/ln10;
	int lwork = (n_isophote_datapts * (bspline_order + 1) + bspline_nmax * (7 + 3 * bspline_order));
	int iopt = -1;                       // Least-squares fitting mode
	int ier = 0;
	double smoothing = 0;
	curfit_(&iopt, &n_isophote_datapts, profile_fit_logxivals, profile_fit_data, profile_fit_weights, &logxi_initial, &logxi_final, &bspline_order, &smoothing, &bspline_nmax, &n_bspline_knots_tot, knots, coefs, &minchisq, bspline_work, &lwork, bspline_iwork, &ier);
	if (ier > 0) {
		if (ier >= 10) {
			//cout << "KNOTS: " << endl;
			//for (int i=0; i < n_bspline_knots_tot; i++) cout << knots[i] << endl;
			//cout << "DATA: " << endl;
			//for (int i=0; i < n_isophote_datapts; i++) cout << pow(10,profile_fit_logxivals[i]) << " " << profile_fit_data[i] << " " << profile_fit_weights[i] << endl;
			//for (int i=0; i < n_bspline_knots_tot - bspline_order - 1; i++) cout << coefs[i] << endl;
			stringstream s;
			s << "Error fitting B-Spline curve using curfit(): " << ier;
			throw runtime_error(s.str());
		} else {
			cerr << "WARNING:  Non-fatal error while fitting B-Spline curve using curfit(): " << ier << endl;
		}
	}
#endif
	return minchisq/2;
}

void EllipticityGradient::plot_ellipticity_function(const double ximin, const double ximax, const int nn, const string suffix)
{
	double xi, xistep = pow(ximax/ximin,1.0/(nn-1));
	int i;
	string qname = "qfunc_" + suffix + ".dat";
	string thetaname = "thetafunc_" + suffix + ".dat";
	ofstream qout(qname.c_str());
	ofstream thetaout(thetaname.c_str());
	double ep, angle, q;
	for (i=0, xi=ximin; i < nn; i++, xi *= xistep) {
		ellipticity_function(xi,ep,angle);
		q = sqrt(1 - ep);
		qout << xi << " " << q << endl;
		thetaout << xi << " " << radians_to_degrees(angle) << endl;
	}
}

void EllipticityGradient::plot_fourier_functions(const double ximin, const double ximax, const int nn, const string suffix)
{
	double xi, xistep = pow(ximax/ximin,1.0/(nn-1));
	int i,j;

	for (j=0; j < n_fourier_grad_modes; j++) {
		double cosamp, sinamp;
		string mstring, cosampname, sinampname, cos_filename, sin_filename;
		stringstream mstr;
		mstr << fourier_grad_mvals[j];
		mstr >> mstring;
		cosampname = "A" + mstring;
		sinampname = "B" + mstring;
		cos_filename = cosampname + "func_" + suffix + ".dat";
		sin_filename = sinampname + "func_" + suffix + ".dat";
		ofstream cosout(cos_filename.c_str());
		ofstream sinout(sin_filename.c_str());
		for (i=0, xi=ximin; i < nn; i++, xi *= xistep) {
			fourier_mode_function(xi, fourier_grad_mvals[j], cosamp, sinamp);
			cosout << xi << " " << cosamp << endl;
			sinout << xi << " " << sinamp << endl;
		}
	}
}

double EllipticityGradient::elliptical_radius_root(const double x, const double y)
{
	double (Brent::*xiptr)(const double, const double&, const double&);
	xiptr = static_cast<double (Brent::*)(const double, const double&, const double&)> (&EllipticityGradient::elliptical_radius_root_eq);
	double xisqrmax, xisqrmin;
	xisqrmax = xisqrmin = (x*x+y*y);
	if (egrad_minq > 1.0) die("egrad_minq has not been set (egrad_minq=%g)",egrad_minq);
	if (egrad_minq==0.0) die("egrad_minq is equal to zero");
	if (egrad_ellipticity_mode==0) {
		xisqrmax /= egrad_minq*egrad_minq; // greatest possible xi value is if efunc_qi is at its minimum value, and (x,y) are on the minor axis
	}
	else {
		xisqrmax /= egrad_minq; // greatest possible xi value is if efunc_qi is at its minimum value, and (x,y) are on the minor axis
		xisqrmin *= egrad_minq; // smallest possible xi value is if efunc_qi is at its minimum value, and (x,y) are on the major axis
	}
	double ximax = sqrt(xisqrmax);
	double ximin = sqrt(xisqrmin);
	if ((ximax < xi_initial_egrad) or (ximin > xi_final_egrad)) {
		double ep, th, xisq;
		if (ximax < xi_initial_egrad) {
			ellipticity_function(xi_initial_egrad,ep,th);
			//qq = (this->*egrad_ptr)(xi_initial_egrad,geometric_param[0],0);
		} else if (ximin > xi_final_egrad) {
			ellipticity_function(xi_final_egrad,ep,th);
			//qq = (this->*egrad_ptr)(xi_final_egrad,geometric_param[0],0);
		}
		double costh, sinth, xprime, yprime;
		double fsqinv = (egrad_ellipticity_mode==0) ? 1 : sqrt(1-ep);
		costh = cos(th);
		sinth = sin(th);
		xprime = x*costh + y*sinth;
		yprime = -x*sinth + y*costh;
		return sqrt(fsqinv*(xprime*xprime + (yprime*yprime)/(1-ep)));
	}
	//cout << "minq=" << egrad_minq << " ximin=" << ximin << " ximax=" << ximax << endl;
	//double xi = BrentsMethod(xiptr,x,y,0.9*ximin,1.1*ximax,1e-4);
	//cout << "Trying x=" << x << ", y=" << y << ", minq=" << egrad_minq << ", ximin=" << ximin << ", ximax=" << ximax << endl;
	double xi = BrentsMethod(xiptr,x,y,0.4*ximin,1.6*ximax,1e-7);
	if ((xi > (1.1*ximax)) or (xi < (0.9*ximin))) {
		cout << "WARNING: xi out of expected range (xi=" << xi << ", ximin=" << ximin << ", ximax=" << ximax << ")" << endl;
		double ep,th;
		ellipticity_function(xi,ep,th);
		double qq = sqrt(1 - ep);
		cout << "x=" << x << ", y=" << y << ", xi0=" << xi_initial_egrad << ", xif=" << xi_final_egrad << ", q=" << qq << ", theta=" << radians_to_degrees(th) << ", minq=" << egrad_minq << endl;

		if (egrad_mode==0) {
			double bspline_logximin = geometric_knots[0][0];
			double bspline_logximax = geometric_knots[0][n_bspline_knots_tot-1];
			cout << "xi=" << xi << " xmin_spline=" << pow(10,bspline_logximin) << " xmax_spline=" << pow(10,bspline_logximax) << endl;

			cout << "BSPLINE: " << endl;
			double logxi;
			int n_xisteps_qmin = 120;
			double logxistep_q = (bspline_logximax-bspline_logximin)/(n_xisteps_qmin-1);
			int i;
			for (i=0, logxi = bspline_logximin; i < n_xisteps_qmin; i++, logxi += logxistep_q) {
				xi = pow(10.0,logxi);
				qq = (this->*egrad_ptr)(xi,geometric_param[0],0);
				cout << xi << " " << qq << endl;
			}
			for (i=0, logxi = bspline_logximin; i < n_xisteps_qmin; i++, logxi += logxistep_q) {
				xi = pow(10.0,logxi);
				qq = (this->*egrad_ptr)(xi,geometric_param[0],0);
				if (qq < egrad_minq) egrad_minq = qq;
				//cout << xi << " " << qq << " " << egrad_minq << endl;
			}
			cout << "CHECKING EGRAD_MINQ: " << egrad_minq << endl;	
		}
		die();
		//cout << "qi=" << geometric_param[0][0] << " qf=" << geometric_param[0][1] << " xi_q=" << geometric_param[0][2] << " dxi_q=" << geometric_param[0][3] << endl;
		//die();
		//cout << geometric_param[1][0] << " " << geometric_param[1][1] << " " << geometric_param[1][2] << " " << geometric_param[1][3] << endl;
	}
	return xi;
}

double EllipticityGradient::elliptical_radius_root_eq(const double xi, const double &xi_root_x, const double &xi_root_y)
{
	double ep, efunc_theta_i;
	double costh, sinth, xprime, yprime;
	ellipticity_function(xi,ep,efunc_theta_i);
	double fsqinv = (egrad_ellipticity_mode==0) ? 1 : sqrt(1-ep);
	costh = cos(efunc_theta_i);
	sinth = sin(efunc_theta_i);
	xprime = xi_root_x*costh + xi_root_y*sinth;
	yprime = -xi_root_x*sinth + xi_root_y*costh;
	return (xi*xi - fsqinv*(xprime*xprime + (yprime*yprime)/(1-ep)));
}

void EllipticityGradient::set_egrad_ptr()
{
	if (egrad_mode==0) {
		egrad_ptr = &EllipticityGradient::egrad_bspline_function;
	} else if ((egrad_mode==1) or (egrad_mode==2)) {
		egrad_ptr = &EllipticityGradient::egrad_tanh_function;
	} else {
		egrad_ptr = NULL;
	}
}

void EllipticityGradient::disable_egrad_mode(int& n_tot_egrad_params)
{
	n_tot_egrad_params = 0;
	for (int i=0; i < 4; i++) n_tot_egrad_params += n_egrad_params[i];

	ellipticity_gradient = false;
	n_egrad_params[0] = 0; // q
	n_egrad_params[1] = 0; // theta
	n_egrad_params[2] = 0; // xc
	n_egrad_params[3] = 0; // yc
}

void EllipticityGradient::set_geometric_param_pointers_egrad(double **param, boolvector& angle_param, int& qi)
{
	int i,j;
	if ((egrad_mode==0) or (egrad_mode==1)) {
		for (i=0; i < 4; i++) {
			for (j=0; j < n_egrad_params[i]; j++) {
				if ((i==1) and (angle_param_egrad[j])) angle_param[qi] = true;
				param[qi++] = &geometric_param[i][j];
			}
		}
	} else if (egrad_mode==2) {
		for (i=0; i < 4; i++) {
			for (j=0; j < n_egrad_params[i]; j++) {
				if ((i==1) and (angle_param_egrad[j])) angle_param[qi] = true;
				if (j==0) param[qi++] = &geometric_param_ref[i];
				else if (j==1) param[qi++] = &geometric_param_dif[i];
				else param[qi++] = &geometric_param[i][j];
			}
		}
	} else die("only egrad_mode=0,1,2 supported");
	if (fourier_gradient) {
		int k=0;
		for (i=0, k=0; i < n_fourier_grad_modes; i++, k += 2) {
			for (j=0; j < n_fourier_grad_params[i]; j++) {
				param[qi++] = &fourier_param[k][j];
			}
			for (j=0; j < n_fourier_grad_params[i]; j++) {
				param[qi++] = &fourier_param[k+1][j];
			}
		}
	}
}

void EllipticityGradient::get_egrad_params(dvector& egrad_params)
{
	int n_tot_egrad_params = 0;
	for (int i=0; i < 4; i++) n_tot_egrad_params += n_egrad_params[i];
	egrad_params.input(n_tot_egrad_params);
	int i,j,qi=0;
	for (i=0; i < 4; i++) {
		for (j=0; j < n_egrad_params[i]; j++) {
			if ((i==1) and (angle_param_egrad[j]))
				egrad_params[qi++] = radians_to_degrees(geometric_param[i][j]);
			else
				egrad_params[qi++] = geometric_param[i][j];
		}
	}
}

int EllipticityGradient::get_egrad_nparams()
{
	int n_tot_egrad_params = 0;
	for (int i=0; i < 4; i++) n_tot_egrad_params += n_egrad_params[i];
	return n_tot_egrad_params;
}

void EllipticityGradient::update_egrad_meta_parameters()
{
	if (egrad_mode==0) {
		egrad_minq = 1e30;
		int i, n_xisteps_qmin = 120;
		//for (int i=0; i < n_egrad_params[0]; i++) if (geometric_param[0][i] < egrad_minq) egrad_minq = geometric_param[0][i];
		double bspline_logximin = geometric_knots[0][0];
		double bspline_logximax = geometric_knots[0][n_bspline_knots_tot-1];
		//for (i=0; i < n_bspline_knots_tot; i++) cout << geometric_knots[0][i] << endl;
		//cout << "MINMAX: " << bspline_logximin << " " << bspline_logximax << endl;
		//die();
		double qq, xi, logxi;
		double logxistep_q = (bspline_logximax-bspline_logximin)/(n_xisteps_qmin-1);
		for (i=0, logxi = bspline_logximin; i < n_xisteps_qmin; i++, logxi += logxistep_q) {
			xi = pow(10.0,logxi);
			qq = (this->*egrad_ptr)(xi,geometric_param[0],0);
			if (qq < egrad_minq) egrad_minq = qq;
			//cout << xi << " " << qq << " " << egrad_minq << endl;
		}
		if ((egrad_minq <= 0) or (egrad_minq > 1)) {
			for (i=0, logxi = bspline_logximin; i < n_xisteps_qmin; i++, logxi += logxistep_q) {
				xi = pow(10.0,logxi);
				qq = (this->*egrad_ptr)(xi,geometric_param[0],0);
				cout << xi << " " << qq << endl;
			}
			die("absurd minimum q value from B-spline");
		}
		//cout << "updated egrad_minq=" << egrad_minq << endl;
	} else if ((egrad_mode==1) or (egrad_mode==2)) {
		if (egrad_mode==2) {
			double step_ref;
			for (int i=0; i < 4; i++) {
				if (n_egrad_params[i]==1) geometric_param[i][0] = geometric_param_ref[i];
				else {
					step_ref = tanh((xi_ref_egrad-geometric_param[i][2])/geometric_param[i][3]);
					geometric_param[i][0] = geometric_param_ref[i] - (1+step_ref)*geometric_param_dif[i]/2;
					geometric_param[i][1] = geometric_param_ref[i] + (1-step_ref)*geometric_param_dif[i]/2;
				}
			}
			if (geometric_param[0][0] > 1.0) geometric_param[0][0] = 1.0; // q cannot be greater than 1
			if (geometric_param[0][1] > 1.0) geometric_param[0][1] = 1.0; // q cannot be greater than 1
			if (geometric_param[0][0] < 0.001) geometric_param[0][0] = 0.001;
			if (geometric_param[0][1] < 0.001) geometric_param[0][1] = 0.001;
		}
		double xi0_q, xif_q, xi0_th, xif_th;
		xi0_q = geometric_param[0][2] - 3*geometric_param[0][3];
		if (xi0_q < 0) xi0_q = 0;
		xif_q = geometric_param[0][2] + 3*geometric_param[0][3];

		xi0_th = geometric_param[1][2] - 3*geometric_param[1][3];
		if (xi0_th < 0) xi0_th = 0;
		xif_th = geometric_param[1][2] + 3*geometric_param[1][3];
		xi_initial_egrad = dmin(xi0_q,xi0_th);
		if (xi_initial_egrad < 0) xi_initial_egrad = 0;
		xi_final_egrad = dmax(xif_q,xif_th);
		egrad_minq = (geometric_param[0][1] < geometric_param[0][0]) ? geometric_param[0][1] : geometric_param[0][0];
		//cout << "qi=" << geometric_param[0][0] << ", qf=" << geometric_param[0][1] << endl;
		//cout << "theta_i=" << radians_to_degrees(geometric_param[1][0]) << ", theta_f=" << radians_to_degrees(geometric_param[1][1]) << endl;
		//cout << "xc=" << geometric_param[2][0] << endl;
		//cout << "yc=" << geometric_param[3][0] << endl;

	} else die("only egrad_mode=0 or 1 currently supported");
}

void EllipticityGradient::set_geometric_paramnames_egrad(vector<string>& paramnames, vector<string>& latex_paramnames, vector<string>& latex_param_subscripts, int &qi, string latex_suffix)
{
	if (egrad_mode==0) {
		int i,j,k;
		string name, lname;
		for (i=0; i < 2; i++) {
			if (i==0) { name = "q"; lname = "q"; }
			else { name = "theta"; lname = "\\theta"; }
			for (j=0; j < n_egrad_params[i]; j++) {
				stringstream jstr;
				string jstring;
				jstr << j;
				jstr >> jstring;
				paramnames[qi] = name + "_spl" + jstring;
				latex_paramnames[qi] = lname;
				latex_param_subscripts[qi] = "sp," + jstring + latex_suffix;
				qi++;
			}
		}
		paramnames[qi] = "xc"; latex_paramnames[qi] = "x"; latex_param_subscripts[qi] = "c" + latex_suffix; qi++;
		paramnames[qi] = "yc"; latex_paramnames[qi] = "y"; latex_param_subscripts[qi] = "c" + latex_suffix; qi++;
		for (i=0; i < n_fourier_grad_modes; i++) {
			string mstring, ampname, latex_ampname;
			stringstream mstr;
			mstr << fourier_grad_mvals[i];
			mstr >> mstring;
			for (k=0; k < 2; k++) {
				if (k==0) {
					ampname = "A" + mstring;
					latex_ampname = "A";
				} else {
					ampname = "B" + mstring;
					latex_ampname = "B";
				}

				for (j=0; j < n_fourier_grad_params[i]; j++) {
					stringstream jstr;
					string jstring;
					jstr << j;
					jstr >> jstring;
					paramnames[qi] = ampname + "_spl" + jstring;
					latex_paramnames[qi] = latex_ampname;
					latex_param_subscripts[qi] = mstring + "sp," + jstring + latex_suffix;
					qi++;
				}
			}
		}
	} else if (egrad_mode==1) {
		paramnames[qi] = "qi"; latex_paramnames[qi] = "q"; latex_param_subscripts[qi] = "i" + latex_suffix; qi++;
		paramnames[qi] = "qf"; latex_paramnames[qi] = "q"; latex_param_subscripts[qi] = "f" + latex_suffix; qi++;
		paramnames[qi] = "xi0_q"; latex_paramnames[qi] = "\\xi"; latex_param_subscripts[qi] = "0,q" + latex_suffix; qi++;
		paramnames[qi] = "dxi_q"; latex_paramnames[qi] = "\\Delta\\xi"; latex_param_subscripts[qi] = "q" + latex_suffix; qi++;
		paramnames[qi] = "theta_i"; latex_paramnames[qi] = "\\theta"; latex_param_subscripts[qi] = "i" + latex_suffix; qi++;
		paramnames[qi] = "theta_f"; latex_paramnames[qi] = "\\theta"; latex_param_subscripts[qi] = "f" + latex_suffix; qi++;
		paramnames[qi] = "xi0_theta"; latex_paramnames[qi] = "\\xi"; latex_param_subscripts[qi] = "0,\\theta" + latex_suffix; qi++;
		paramnames[qi] = "dxi_theta"; latex_paramnames[qi] = "\\Delta\\xi"; latex_param_subscripts[qi] = "\\theta" + latex_suffix; qi++;
		paramnames[qi] = "xc"; latex_paramnames[qi] = "x"; latex_param_subscripts[qi] = "c" + latex_suffix; qi++;
		paramnames[qi] = "yc"; latex_paramnames[qi] = "y"; latex_param_subscripts[qi] = "c" + latex_suffix; qi++;
		for (int i=0; i < n_fourier_grad_modes; i++) {
			string mstring, ampname, latex_ampname;
			stringstream mstr;
			mstr << fourier_grad_mvals[i];
			mstr >> mstring;
			ampname = "A" + mstring;
			latex_ampname = "A";
			paramnames[qi] = ampname + "_i"; latex_paramnames[qi] = latex_ampname; latex_param_subscripts[qi] = mstring + "i" + latex_suffix; qi++;
			paramnames[qi] = ampname + "_f"; latex_paramnames[qi] = latex_ampname; latex_param_subscripts[qi] = mstring + "f" + latex_suffix; qi++;
			paramnames[qi] = "xi0_" + ampname; latex_paramnames[qi] = "\\xi"; latex_param_subscripts[qi] = ampname + latex_suffix; qi++;
			paramnames[qi] = "dxi_" + ampname; latex_paramnames[qi] = "\\Delta\\xi"; latex_param_subscripts[qi] = ampname + latex_suffix; qi++;

			ampname = "B" + mstring;
			latex_ampname = "B";
			paramnames[qi] = ampname + "_i"; latex_paramnames[qi] = latex_ampname; latex_param_subscripts[qi] = mstring + "i" + latex_suffix; qi++;
			paramnames[qi] = ampname + "_f"; latex_paramnames[qi] = latex_ampname; latex_param_subscripts[qi] = mstring + "f" + latex_suffix; qi++;
			paramnames[qi] = "xi0_" + ampname; latex_paramnames[qi] = "\\xi"; latex_param_subscripts[qi] = ampname + latex_suffix; qi++;
			paramnames[qi] = "dxi_" + ampname; latex_paramnames[qi] = "\\Delta\\xi"; latex_param_subscripts[qi] = ampname + latex_suffix; qi++;
		}
	} else if (egrad_mode==2) {
		paramnames[qi] = "qref"; latex_paramnames[qi] = "q"; latex_param_subscripts[qi] = "ref" + latex_suffix; qi++;
		paramnames[qi] = "delta_q"; latex_paramnames[qi] = "\\Delta"; latex_param_subscripts[qi] = "q" + latex_suffix; qi++;
		paramnames[qi] = "xi0_q"; latex_paramnames[qi] = "\\xi"; latex_param_subscripts[qi] = "0,q" + latex_suffix; qi++;
		paramnames[qi] = "dxi_q"; latex_paramnames[qi] = "\\Delta\\xi"; latex_param_subscripts[qi] = "q" + latex_suffix; qi++;
		paramnames[qi] = "theta_ref"; latex_paramnames[qi] = "\\theta"; latex_param_subscripts[qi] = "ref" + latex_suffix; qi++;
		paramnames[qi] = "delta_theta"; latex_paramnames[qi] = "\\Delta"; latex_param_subscripts[qi] = "\\theta" + latex_suffix; qi++;
		paramnames[qi] = "xi0_theta"; latex_paramnames[qi] = "\\xi"; latex_param_subscripts[qi] = "0,\\theta" + latex_suffix; qi++;
		paramnames[qi] = "dxi_theta"; latex_paramnames[qi] = "\\Delta\\xi"; latex_param_subscripts[qi] = "\\theta" + latex_suffix; qi++;
		paramnames[qi] = "xc"; latex_paramnames[qi] = "x"; latex_param_subscripts[qi] = "c" + latex_suffix; qi++;
		paramnames[qi] = "yc"; latex_paramnames[qi] = "y"; latex_param_subscripts[qi] = "c" + latex_suffix; qi++;
		for (int i=0; i < n_fourier_grad_modes; i++) {
			string mstring, ampname, latex_ampname;
			stringstream mstr;
			mstr << fourier_grad_mvals[i];
			mstr >> mstring;
			ampname = "A" + mstring;
			latex_ampname = "A";
			paramnames[qi] = ampname + "_i"; latex_paramnames[qi] = latex_ampname; latex_param_subscripts[qi] = mstring + "i" + latex_suffix; qi++;
			paramnames[qi] = ampname + "_f"; latex_paramnames[qi] = latex_ampname; latex_param_subscripts[qi] = mstring + "f" + latex_suffix; qi++;
			paramnames[qi] = "xi0_" + ampname; latex_paramnames[qi] = "\\xi"; latex_param_subscripts[qi] = ampname + latex_suffix; qi++;
			paramnames[qi] = "dxi_" + ampname; latex_paramnames[qi] = "\\Delta\\xi"; latex_param_subscripts[qi] = ampname + latex_suffix; qi++;

			ampname = "B" + mstring;
			latex_ampname = "B";
			paramnames[qi] = ampname + "_i"; latex_paramnames[qi] = latex_ampname; latex_param_subscripts[qi] = mstring + "i" + latex_suffix; qi++;
			paramnames[qi] = ampname + "_f"; latex_paramnames[qi] = latex_ampname; latex_param_subscripts[qi] = mstring + "f" + latex_suffix; qi++;
			paramnames[qi] = "xi0_" + ampname; latex_paramnames[qi] = "\\xi"; latex_param_subscripts[qi] = ampname + latex_suffix; qi++;
			paramnames[qi] = "dxi_" + ampname; latex_paramnames[qi] = "\\Delta\\xi"; latex_param_subscripts[qi] = ampname + latex_suffix; qi++;
		}
	} else die("only egrad_mode=0, 1, or 2 currently supported");
}

void EllipticityGradient::set_geometric_param_ranges_egrad(boolvector& set_auto_penalty_limits, dvector& penalty_lower_limits, dvector& penalty_upper_limits, int &param_i)
{
	if (egrad_mode==0) {
		int i,j,k;
		for (i=0; i < 2; i++) {
			for (j=0; j < n_egrad_params[i]; j++) {
				if (i==0) {
					// for q params, note that they can technically be above 1 since they're B-spline coefficients
					set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 5e-3; penalty_upper_limits[param_i] = 2; param_i++; // q params
				} else {
					set_auto_penalty_limits[param_i++] = false; // theta params
				}
			}
		}
		set_auto_penalty_limits[param_i++] = false; // xc
		set_auto_penalty_limits[param_i++] = false; // yc

		for (i=0; i < n_fourier_grad_modes; i++) {
			for (k=0; k < 2; k++) {
				for (j=0; j < n_fourier_grad_params[i]; j++) {
					set_auto_penalty_limits[param_i++] = false; // Fourier amplitudes
				}
			}
		}
	} else if ((egrad_mode==1) or (egrad_mode==2)) {
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 5e-3; penalty_upper_limits[param_i] = 1; param_i++; // qi
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 5e-3; penalty_upper_limits[param_i] = 1; param_i++; // qf
		set_auto_penalty_limits[param_i++] = false; // xi0_q
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 1e-3; penalty_upper_limits[param_i] = 1e30; param_i++; // dxi_q
		set_auto_penalty_limits[param_i++] = false; // theta_i
		set_auto_penalty_limits[param_i++] = false; // theta_f
		set_auto_penalty_limits[param_i++] = false; // xi0_theta
		set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 1e-3; penalty_upper_limits[param_i] = 1e30; param_i++; // dxi_theta
		set_auto_penalty_limits[param_i++] = false; // xc
		set_auto_penalty_limits[param_i++] = false; // yc
		for (int i=0; i < (2*n_fourier_grad_modes); i++) {
			set_auto_penalty_limits[param_i++] = false; // amp_i
			set_auto_penalty_limits[param_i++] = false; // amp_f
			set_auto_penalty_limits[param_i++] = false; // xi0_amp
			set_auto_penalty_limits[param_i] = true; penalty_lower_limits[param_i] = 1e-3; penalty_upper_limits[param_i] = 1e30; param_i++; // dxi_amp
		}
	} else die("only egrad_mode=0, 1, or 2 currently supported");
}

void EllipticityGradient::set_geometric_stepsizes_egrad(dvector& stepsizes, int &index)
{
	if (egrad_mode==0) {
		int i,j,k;
		for (i=0; i < 2; i++) {
			for (j=0; j < n_egrad_params[i]; j++) {
				if (i==0) {
					stepsizes[index++] = 0.05; // q params
				} else {
					stepsizes[index++] = 5; // theta params
				}
			}
		}
		stepsizes[index++] = 0.1; // xc
		stepsizes[index++] = 0.1; // yc
		for (i=0; i < n_fourier_grad_modes; i++) {
			for (k=0; k < 2; k++) {
				// k=0 and k=1 correspond to the cos and sin modes
				for (j=0; j < n_fourier_grad_params[i]; j++) {
					stepsizes[index++] = 0.01; // Fourier amplitudes
				}
			}
		}
	} else if ((egrad_mode==1) or (egrad_mode==2)) {
		stepsizes[index++] = 0.1; // qi (or qref)
		stepsizes[index++] = 0.1; // qf (or delta_q)
		stepsizes[index++] = 0.3; // xi0_q
		stepsizes[index++] = 0.3; // dxi_q
		stepsizes[index++] = 5; // theta_i (or theta_ref)
		stepsizes[index++] = 5; // theta_f (or delta_theta)
		stepsizes[index++] = 0.3; // xi0_theta
		stepsizes[index++] = 0.3; // xi0_theta
		stepsizes[index++] = 0.1; // xc
		stepsizes[index++] = 0.1; // yc
		for (int i=0; i < (2*n_fourier_grad_modes); i++) {
			stepsizes[index++] = 0.01; // amp_i
			stepsizes[index++] = 0.01; // amp_f
			stepsizes[index++] = 0.3; // xi0_q
			stepsizes[index++] = 0.3; // dxi_q
		}
	} else die("only egrad_mode=0, 1, or 2 currently supported");
}

void EllipticityGradient::set_fourier_paramnums(int *paramnum, int paramnum0)
{
	if (fourier_gradient) {
		paramnum[0] = paramnum0;
		for (int i=0; i < n_fourier_grad_modes-1; i++) {
			paramnum[i+1] = paramnum[i] + 2*n_fourier_grad_params[i];
		}
	}
}

