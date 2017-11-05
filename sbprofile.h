#ifndef SBPROFILE_H
#define SBPROFILE_H

#include "mathexpr.h"
#include "spline.h"
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

enum SB_ProfileName { SB_SPLINE, GAUSSIAN, SERSIC, TOPHAT };

class SB_Profile
{
	private:
	Spline sb_spline;
	double sb_splint(double);
	double qx_parameter, f_parameter;

	protected:
	SB_ProfileName sbtype;
	double q, theta, x_center, y_center; // four base parameters, which can be added to in derived surface brightness models
	double costheta, sintheta;

	// in all derived classes, each of the following function pointers MUST be set in the constructor.
	// If the integrals are to be done numerically, they should point to the appropriate function using
	// a static cast (see the constructor for any of the derived classes for examples)

	void set_angle(const double &theta_degrees);
	void set_angle_radians(const double &theta_in);
	void rotate(double&, double&);
	void rotate_back(double&, double&);

	public:
	int sb_number;

	SB_Profile() : qx_parameter(1) {}
	SB_Profile(const char *splinefile, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const double &qx_in, const double &f_in);
	SB_Profile(const SB_Profile* sb_in);
	~SB_Profile() {}

	void set_geometric_parameters(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	void set_geometric_parameters_radians(const double &q_in, const double &theta_in, const double &xc_in, const double &yc_in);

	// the following items MUST be redefined in all derived classes
	virtual double sb_rsq(const double rsq); // we use the r^2 version in the integrations rather than r because it is most directly used in cored models
	virtual void window_params(double& xmin, double& xmax, double& ymin, double& ymax);
	virtual double window_rmax();
	virtual void print_parameters();

	// these functions can be redefined in the derived classes, but don't have to be
	virtual double surface_brightness_r(const double r);
	virtual double surface_brightness(double x, double y);

	SB_ProfileName get_sbtype() { return sbtype; }
	void get_center_coords(double &xc, double &yc) { xc=x_center; yc=y_center; }
};

class Gaussian : public SB_Profile
{
	private:
	double max_sb, sig_x; // sig_x is the dispersion along the major axis

	double sb_rsq(const double);

	public:
	Gaussian() : SB_Profile() {}
	Gaussian(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Gaussian(const Gaussian* sb_in);
	~Gaussian() {}

	void print_parameters();
	double window_rmax();
};

class Sersic : public SB_Profile
{
	private:
	double s0, k, n; // sig_x is the dispersion along the major axis
	double re; // effective radius

	double sb_rsq(const double);

	public:
	Sersic() : SB_Profile() {}
	Sersic(const double &s0_in, const double &k_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	Sersic(const Sersic* sb_in);
	~Sersic() {}

	void print_parameters();
	double window_rmax();
};

class TopHat : public SB_Profile
{
	private:
	double sb, rad; // sig_x is the dispersion along the major axis

	double sb_rsq(const double);

	public:
	TopHat() : SB_Profile() {}
	TopHat(const double &max_sb_in, const double &sig_x_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in);
	TopHat(const TopHat* sb_in);
	~TopHat() {}

	void print_parameters();
	double window_rmax();
};


#endif // SBPROFILE_H
