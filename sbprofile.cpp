#include "sbprofile.h"
#include "qlens.h"
#include "mathexpr.h"
#include "errors.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

SB_Profile::SB_Profile(const char *splinefile, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in)
{
	sbtype = SB_SPLINE;
	qx_parameter = qx_in;
	f_parameter = f_in;
	sb_spline.input(splinefile);
}

SB_Profile::SB_Profile(const SB_Profile* sb_in)
{
	sbtype = sb_in->sbtype;
	sb_number = sb_in->sb_number;

	set_geometric_parameters_radians(sb_in->q,sb_in->theta,sb_in->x_center,sb_in->y_center);
	qx_parameter = sb_in->qx_parameter;
	f_parameter = sb_in->f_parameter;
	sb_spline.input(sb_in->sb_spline);
}

void SB_Profile::set_geometric_parameters(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	qx_parameter = 1.0;
	q=q_in;
	if (q < 0) q = -q; // don't allow negative axis ratios
	if (q > 1) q = 1.0; // don't allow q>1
	set_angle(theta_degrees);
	x_center = xc_in;
	y_center = yc_in;
}

void SB_Profile::set_geometric_parameters_radians(const double &q_in, const double &theta_in, const double &xc_in, const double &yc_in)
{
	qx_parameter = 1.0;
	q=q_in;
	if (q < 0) q = -q; // don't allow negative axis ratios
	if (q > 1) q = 1.0; // don't allow q>1
	set_angle_radians(theta_in);
	x_center = xc_in;
	y_center = yc_in;
}

double SB_Profile::sb_rsq(const double rsq) // this function should be redefined in all derived classes
{
	double r = sqrt(rsq);
	if (r < qx_parameter*sb_spline.xmin()) return (f_parameter*sb_spline.extend_inner_logslope(r/qx_parameter));
	if (r > qx_parameter*sb_spline.xmax()) return (f_parameter*sb_spline.extend_outer_logslope(r/qx_parameter));
	return (f_parameter*sb_spline.splint(r/qx_parameter));
}

void SB_Profile::set_angle(const double &theta_degrees)
{
	theta = degrees_to_radians(theta_degrees); costheta = cos(theta); sintheta = sin(theta); // trig functions are stored to save computation time later
}

void SB_Profile::set_angle_radians(const double &theta_in)
{
	theta = theta_in; costheta = cos(theta); sintheta = sin(theta); // trig functions are stored to save computation time later
}

inline void SB_Profile::rotate(double &x, double &y)
{
	// perform a counter-clockwise rotation of the coordinate system to match the coordinate system of the rotated galaxy
	double xp = x*costheta + y*sintheta;
	y = -x*sintheta + y*costheta;
	x = xp;
}

inline void SB_Profile::rotate_back(double &x, double &y)
{
	// perform a counter-clockwise rotation of the coordinate system to match the coordinate system of the rotated galaxy
	double xp = x*costheta - y*sintheta;
	y = x*sintheta + y*costheta;
	x = xp;
}

double SB_Profile::surface_brightness(double x, double y)
{
	// switch to coordinate system centered on surface brightness profile
	x -= x_center;
	y -= y_center;
	if (theta != 0) rotate(x,y);
	return sb_rsq(x*x + y*y/(q*q));
}

double SB_Profile::surface_brightness_r(const double r)
{
	return sb_rsq(r*r);
}

void SB_Profile::print_parameters()
{
	cout << "sbspline: q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")\n";
}

void SB_Profile::window_params(double& xmin, double& xmax, double& ymin, double& ymax)
{
	double rmax = window_rmax();
	xmin = -rmax;
	xmax = rmax;
	ymin = -q*rmax;
	ymax = q*rmax;
	if (theta != 0) {
		double xx[4], yy[4];
		xx[0]=xmin; yy[0]=ymin;
		xx[1]=xmax; yy[1]=ymin;
		xx[2]=xmax; yy[2]=ymax;
		xx[3]=xmin; yy[3]=ymax;
		xmin=1e30; xmax=-1e30; ymin=1e30; ymax=-1e30;
		for (int i=0; i < 4; i++) {
			rotate_back(xx[i],yy[i]);
			if (xx[i] < xmin) xmin=xx[i];
			if (xx[i] > xmax) xmax=xx[i];
			if (yy[i] < ymin) ymin=yy[i];
			if (yy[i] > ymax) ymax=yy[i];
		}
	}
	xmin += x_center;
	xmax += x_center;
	ymin += y_center;
	ymax += y_center;
}

double SB_Profile::window_rmax()
{
	return qx_parameter*sb_spline.xmax();
}

Gaussian::Gaussian(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	sbtype=GAUSSIAN;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	max_sb = max_sb_in; sig_x = sig_x_in;
}

Gaussian::Gaussian(const Gaussian* sb_in)
{
	sbtype = sb_in->sbtype;
	sb_number = sb_in->sb_number;

	max_sb = sb_in->max_sb;
	sig_x = sb_in->sig_x;
	set_geometric_parameters_radians(sb_in->q,sb_in->theta,sb_in->x_center,sb_in->y_center);
}

double Gaussian::sb_rsq(const double rsq)
{
	return max_sb*exp(-0.5*rsq/(sig_x*sig_x));
}

void Gaussian::print_parameters()
{
	cout << "gaussian: max_sb=" << max_sb << ", sig_x=" << sig_x << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")\n";
}

double Gaussian::window_rmax()
{
	return 5*sig_x;
}

Sersic::Sersic(const double &s0_in, const double &Re_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	sbtype=SERSIC;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	n = n_in;
	re = Re_in;
	double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	k = b*pow(sqrt(q)/Re_in,1.0/n);
	s0 = s0_in;
	//s0 = L0_in/(M_PI*re*re*2*n*Gamma(2*n)/pow(b,2*n));
}

Sersic::Sersic(const Sersic* sb_in)
{
	sbtype = sb_in->sbtype;
	sb_number = sb_in->sb_number;

	s0 = sb_in->s0;
	n = sb_in->n;
	k = sb_in->k;
	set_geometric_parameters_radians(sb_in->q,sb_in->theta,sb_in->x_center,sb_in->y_center);
}

double Sersic::sb_rsq(const double rsq)
{
	return s0*exp(-k*pow(rsq,0.5/n));
}

void Sersic::print_parameters()
{
	cout << "sersic: s0=" << s0 << ", R_eff=" << re << ", n=" << n << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")\n";
}

double Sersic::window_rmax()
{
	return pow(3.0/k,n);
}

TopHat::TopHat(const double &sb_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	sbtype=TOPHAT;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	sb = sb_in; rad = rad_in;
}

TopHat::TopHat(const TopHat* sb_in)
{
	sbtype = sb_in->sbtype;
	sb_number = sb_in->sb_number;

	sb = sb_in->sb;
	rad = sb_in->rad;
	set_geometric_parameters_radians(sb_in->q,sb_in->theta,sb_in->x_center,sb_in->y_center);
}

double TopHat::sb_rsq(const double rsq)
{
	return (rsq < rad*rad) ? sb : 0.0;
}

void TopHat::print_parameters()
{
	cout << "tophat: sb=" << sb << ", radius=" << rad << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")\n";
}

double TopHat::window_rmax()
{
	return 2*rad;
}


