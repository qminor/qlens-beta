#include <cmath>
#include <iostream>
#include <sstream>
#include "profile.h"
#include "mathexpr.h"
#include "errors.h"
#include "GregsMathHdr.h"
#include "hyp_2F1.h"
#include <complex>
using namespace std;

bool Shear::use_shear_component_params;

/***************************** Generalized Isothermal Ellipsoid (alpha) *****************************/

Alpha::Alpha(const double &bb_prime, const double &aa, const double &ss_prime, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype=ALPHA;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(7);
	assign_param_pointers();
	assign_paramnames();
	set_default_base_values(nn,acc);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	if (q > 1) q = 1.0; // don't allow q>1
	b = bb_prime/sqrt(q);
	s = ss_prime/sqrt(q);
	alpha = aa;
	if (s < 0) s = -s; // don't allow negative core radii
	qsq=q*q; ssq=s*s;

	set_integration_pointers();
	set_model_specific_integration_pointers();
}

Alpha::Alpha(const Alpha* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	center_anchor_lens = lens_in->center_anchor_lens;
	n_params = lens_in->n_params;
	assign_paramnames();
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	b = lens_in->b;
	alpha = lens_in->alpha;
	s = lens_in->s;
	if (s < 0) s = -s; // don't allow negative core radii
	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (q > 1) q = 1.0; // don't allow q>1
	qsq=q*q; ssq=s*s;
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);

	set_integration_pointers();
	set_model_specific_integration_pointers();
}

void Alpha::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "b"; latex_paramnames[0] = "b"; latex_param_subscripts[0] = "";
	paramnames[1] = "alpha"; latex_paramnames[1] = "\\alpha"; latex_param_subscripts[1] = "";
	paramnames[2] = "s"; latex_paramnames[2] = "s"; latex_param_subscripts[2] = "";
	if (use_ellipticity_components) {
		paramnames[3] = "e1"; latex_paramnames[3] = "e"; latex_param_subscripts[3] = "1";
		paramnames[4] = "e2"; latex_paramnames[4] = "e"; latex_param_subscripts[4] = "2";
	} else {
		paramnames[3] = "q"; latex_paramnames[3] = "q"; latex_param_subscripts[3] = "";
		paramnames[4] = "theta"; latex_paramnames[4] = "\\theta"; latex_param_subscripts[4] = "";
	}
	if (!center_anchored) {
		paramnames[5] = "xc"; latex_paramnames[5] = "x"; latex_param_subscripts[5] = "c";
		paramnames[6] = "yc"; latex_paramnames[6] = "y"; latex_param_subscripts[6] = "c";
	}
}

void Alpha::assign_param_pointers()
{
	param[0] = &b;
	param[1] = &alpha;
	param[2] = &s;
	param[3] = &q;
	param[4] = &theta;
	if (!center_anchored) {
		param[5] = &x_center;
		param[6] = &y_center;
	}
}

void Alpha::get_parameters(double* params)
{
	params[0] = b*sqrt(q);
	params[1] = alpha;
	params[2] = s*sqrt(q);
	if (use_ellipticity_components) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[3] = (1-q)*cos(2*theta_eff);
		params[4] = (1-q)*sin(2*theta_eff);
	} else {
		params[3] = q;
		params[4] = radians_to_degrees(theta);
	}
	params[5] = x_center;
	params[6] = y_center;
}

void Alpha::update_parameters(const double* params)
{
	alpha=params[1];
	if (use_ellipticity_components) {
		q = 1 - sqrt(SQR(params[3]) + SQR(params[4]));
		set_angle_from_components(params[3],params[4]);
	} else {
		q=params[3];
		set_angle(params[4]);
	}
	b = params[0]/sqrt(q);
	s = params[2]/sqrt(q);
	if (!center_anchored) {
		x_center = params[5];
		y_center = params[6];
	}
	qsq=q*q; ssq=s*s;

	set_integration_pointers();
	set_model_specific_integration_pointers();
}

void Alpha::update_fit_parameters(const double* fitparams, int &index, bool &status)
{
	if (n_vary_params > 0) {
		// note, the actual mass and core parameters are b_prime, s_prime, even though their spherical counterparts b, s are used in the calculations
		// (b_prime = b*sqrt(q) is favored because it is much less degenerate with q; same for s_prime).
		double b_prime, s_prime;
		double old_q = q;
		if (vary_params[0]) b_prime = fitparams[index++];
		else b_prime = b*sqrt(q);

		if (vary_params[1]) {
			if (fitparams[index] <= 0) status = false; // alpha <= 0 is not a physically acceptable value, so report that we're out of bounds
			alpha = fitparams[index++];
		}
		if (vary_params[2]) s_prime = fitparams[index++];
		else {
			s_prime = s*sqrt(q);
		}

		if (use_ellipticity_components) {
			if ((vary_params[3]) or (vary_params[4])) {
				double e_1, e_2;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[3]) e_1 = fitparams[index++];
				else e_1 = (1-q)*cos(2*theta_eff);
				if (vary_params[4]) e_2 = fitparams[index++];
				else e_2 = (1-q)*sin(2*theta_eff);
				q = 1 - sqrt(SQR(e_1) + SQR(e_2));
				set_angle_from_components(e_1,e_2);
				if ((q <= 0) or (q > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
				if (q==0) q = 0.01;
				qsq=q*q;
			}
		} else {
			if (vary_params[3]) {
				if ((fitparams[index] <= 0) or (fitparams[index] > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
				if (q==0) q = 0.01;
				qsq=q*q;
			}
			if (vary_params[4]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[5]) x_center = fitparams[index++];
			if (vary_params[6]) y_center = fitparams[index++];
		}

		if ((vary_params[0]) or (vary_params[3]) or ((use_ellipticity_components) and (vary_params[4]))) b = b_prime/sqrt(q);
		if ((vary_params[2]) or (vary_params[3]) or ((use_ellipticity_components) and (vary_params[4]))) {
			s = s_prime/sqrt(q);
			if (s < 0) s = -s; // don't allow negative core radii
			ssq=s*s;
		}

		set_integration_pointers();
		set_model_specific_integration_pointers();
	}
}

void Alpha::set_model_specific_integration_pointers()
{
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Alpha::deflection_spherical_r);
	if (alpha==1.0) {
		defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Alpha::deflection_spherical_r_iso);
		if (q==1.0) {
			potptr = static_cast<double (LensProfile::*)(const double,const double)> (&Alpha::potential_spherical_iso);
		} else {
			defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&Alpha::deflection_elliptical_iso);
			hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&Alpha::hessian_elliptical_iso);
			potptr = static_cast<double (LensProfile::*)(const double,const double)> (&Alpha::potential_elliptical_iso);
		}
	} else if (s==0.0) {
		defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&Alpha::deflection_elliptical_nocore);
		hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&Alpha::hessian_elliptical_nocore);
		potptr = static_cast<double (LensProfile::*)(const double,const double)> (&Alpha::potential_elliptical_nocore);
	}
}

void Alpha::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = b*sqrt(q);
	if (vary_params[1]) fitparams[index++] = alpha;
	if (vary_params[2]) fitparams[index++] = s*sqrt(q);
	if (use_ellipticity_components) {
		if (vary_params[3]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*cos(2*theta_eff);
		}
		if (vary_params[4]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*sin(2*theta_eff);
		}
	} else {
		if (vary_params[3]) fitparams[index++] = q;
		if (vary_params[4]) fitparams[index++] = radians_to_degrees(theta);
	}
	if (!center_anchored) {
		if (vary_params[5]) fitparams[index++] = x_center;
		if (vary_params[6]) fitparams[index++] = y_center;
	}
}

void Alpha::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.1*b*sqrt(q);
	if (vary_params[1]) stepsizes[index++] = 0.1;
	if (vary_params[2]) stepsizes[index++] = 0.02*b*sqrt(q); // this one is a bit arbitrary, but hopefully reasonable enough
	if (use_ellipticity_components) {
		if (vary_params[3]==true) stepsizes[index++] = 0.1;
		if (vary_params[4]==true) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[3]) stepsizes[index++] = 0.1;
		if (vary_params[4]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[5]) stepsizes[index++] = 0.1*b*sqrt(q);
		if (vary_params[6]) stepsizes[index++] = 0.1*b*sqrt(q);
	}
}

double Alpha::kappa_rsq(const double rsq)
{
	return (0.5 * (2-alpha) * pow(b*b/(ssq+rsq), alpha/2));
}

double Alpha::kappa_rsq_deriv(const double rsq)
{
	return (-0.25 * alpha * (2-alpha) * pow(b*b/(ssq+rsq), alpha/2 + 1)) / (b*b);
}

double Alpha::deflection_spherical_r(const double r)
{
	return (pow(b,alpha)*(pow(r*r+ssq,1-alpha/2) - pow(s,2-alpha)))/r;
}

double Alpha::deflection_spherical_r_iso(const double r) // only for alpha=1
{
	return b*(sqrt(ssq+r*r)-s)/r; // now, tmp = kappa_average
}

void Alpha::deflection_elliptical_iso(const double x, const double y, lensvector& def) // only for alpha=1
{
	double u, psi;
	psi = sqrt(qsq*(ssq+x*x)+y*y);
	u = sqrt(1-qsq);

	def[0] = (b*q/u)*atan(u*x/(psi+s));
	def[1] = (b*q/u)*atanh(u*y/(psi+qsq*s));
}

void Alpha::hessian_elliptical_iso(const double x, const double y, lensmatrix& hess) // only for alpha=1
{
	double xsq, ysq, psi, tmp;
	xsq=x*x; ysq=y*y;

	psi = sqrt(qsq*(ssq+xsq)+ysq);
	tmp = ((b*q)/psi)/(xsq+ysq+2*psi*s+ssq*(1+qsq));

	hess[0][0] = tmp*(ysq+s*psi+ssq*qsq);
	hess[1][1] = tmp*(xsq+s*psi+ssq);
	hess[0][1] = -tmp*x*y;
	hess[1][0] = hess[0][1];
}

double Alpha::potential_spherical_iso(const double x, const double y) // only for alpha=1
{
	double rsq, tmp;
	rsq = x*x+y*y;
	tmp = b*(sqrt(ssq+rsq)-s); // now, tmp = kappa_average*rsq
	if (s != 0) tmp -= b*s*log((s + sqrt(ssq+rsq))/(2.0*s));
	return tmp;
}

double Alpha::potential_elliptical_iso(const double x, const double y) // only for alpha=1
{
	double u, tmp;
	tmp = sqrt(qsq*(ssq+x*x)+y*y);
	u = sqrt(1-qsq);

	tmp = (b*q/u)*(x*atan(u*x/(tmp+s)) + y*atanh(u*y/(tmp+qsq*s)));
	if (s != 0) tmp += b*q*s*(-log(SQR(tmp+s) + SQR(u*x))/2 + log((1.0+q)*s));
	return tmp;
}

void Alpha::deflection_elliptical_nocore(const double x, const double y, lensvector& def)
{
	double phi, R = sqrt(x*x+y*y/(q*q));
	phi = atan(abs(y/(q*x)));

	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	complex<double> def_complex = 2*b*q/(1+q)*pow(b/R,alpha-1)*polar(1.0,phi)*hyp_2F1(1.0,alpha/2.0,2.0-alpha/2.0,-(1-q)/(1+q)*polar(1.0,2*phi));
	//complex<double> z(x,y);
	//complex<double> zconj(x,-y);
	//complex<double> def_complex = (b*b*q)*pow(b/R,-alpha)*hyp_2F1(0.5,alpha/2,1.0+alpha/2,(1-q*q)*R*R/zconj/zconj)/zconj;
	//complex<double> def_complex = (b*b*q)*pow(b/R,-alpha)*hyp_2F1(0.5,alpha/2,1.0+alpha/2,(1-q*q)*R*R/z/z)/z;
	//def_complex = conj(def_complex);

	def[0] = real(def_complex);
	def[1] = imag(def_complex);
}

void Alpha::hessian_elliptical_nocore(const double x, const double y, lensmatrix& hess)
{
	double xi, phi, kap;
	xi = sqrt(q*x*x+y*y/q);
	kap = 0.5 * (2-alpha) * pow(b*sqrt(q)/xi, alpha);
	phi = atan(abs(y/(q*x)));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}

	complex<double> hess_complex, zstar(x,-y);
	// The following is the *deflection*, not the shear, but it will be transformed to shear in the following line
	hess_complex = 2*b*q/(1+q)*pow(b*sqrt(q)/xi,alpha-1)*polar(1.0,phi)*hyp_2F1(1.0,alpha/2.0,2.0-alpha/2.0,-(1-q)/(1+q)*polar(1.0,2*phi));
	hess_complex = -kap*conj(zstar)/zstar + (1-alpha)*hess_complex/zstar; // this is the complex shear

	hess_complex = kap + hess_complex; // this is now (kappa+shear)
	hess[0][0] = real(hess_complex);
	hess[0][1] = imag(hess_complex);
	hess[1][0] = hess[0][1];
	hess_complex = 2*kap - hess_complex; // now we have transformed to (kappa-shear)
	hess[1][1] = real(hess_complex);
}

double Alpha::potential_elliptical_nocore(const double x, const double y) // only for alpha=1
{
	double phi, R = sqrt(x*x+y*y/(q*q));
	phi = atan(abs(y/(q*x)));

	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	complex<double> def_complex = 2*b*q/(1+q)*pow(b/R,alpha-1)*polar(1.0,phi)*hyp_2F1(1.0,alpha/2.0,2.0-alpha/2.0,-(1-q)/(1+q)*polar(1.0,2*phi));
	return (x*real(def_complex) + y*imag(def_complex))/(2-alpha);
}

void Alpha::get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor)
{
	if (s==0.0) {
		re_major_axis = b*pow(zfactor,1.0/alpha);
		re_average = re_major_axis*sqrt(q);
	} else if (alpha==1.0) {
		if (s < b/2.0) {
			re_major_axis = b*sqrt(1-2*s/b/zfactor)*zfactor;
			re_average = re_major_axis*sqrt(q);
		} else {
			re_major_axis = 0;
			re_average = 0;
		}
	} else {
		rmin_einstein_radius = 0.01*b;
		rmax_einstein_radius = 100*b;
		LensProfile::get_einstein_radius(re_major_axis,re_average,zfactor);
	}
}

void Alpha::print_parameters()
{
	double b_prime, s_prime;
	b_prime = b*sqrt(q);
	s_prime = s*sqrt(q);
	if (use_ellipticity_components) {
		double e_1, e_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		e_1 = (1-q)*cos(2*theta_eff);
		e_2 = (1-q)*sin(2*theta_eff);
		cout << "alpha: b=" << b_prime << ", alpha=" << alpha << ", s=" << s_prime << ", e1=" << e_1 << ", e2=" << e_2 << ", center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "alpha: b=" << b_prime << ", alpha=" << alpha << ", s=" << s_prime << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/********************************** PseudoJaffe **********************************/

PseudoJaffe::PseudoJaffe(const double &bb_prime, const double &aa_prime, const double &ss_prime, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = PJAFFE;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(7);
	assign_param_pointers();
	set_default_base_values(nn,acc);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_paramnames();
	if (q > 1) q = 1.0; // don't allow q>1
	b = bb_prime/sqrt(q);
	s = ss_prime/sqrt(q);
	a = aa_prime/sqrt(q);
	if (s < 0) s = -s;
	qsq=q*q; ssq=s*s; asq=a*a;

	set_integration_pointers();
	set_model_specific_integration_pointers();
}

PseudoJaffe::PseudoJaffe(const PseudoJaffe* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	tidal_host = lens_in->tidal_host;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	assign_paramnames();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	b = lens_in->b;
	s = lens_in->s;
	a = lens_in->a;
	if (s < 0) s = -s; // don't allow negative core radii
	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (q > 1) q = 1.0; // don't allow q>1
	qsq=q*q; ssq=s*s; asq=a*a;
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

void PseudoJaffe::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "b"; latex_paramnames[0] = "b"; latex_param_subscripts[0] = "";
	paramnames[1] = "a"; latex_paramnames[1] = "a"; latex_param_subscripts[1] = "";
	paramnames[2] = "s"; latex_paramnames[2] = "s"; latex_param_subscripts[2] = "";

	if (use_ellipticity_components) {
		paramnames[3] = "e1"; latex_paramnames[3] = "e"; latex_param_subscripts[3] = "1";
		paramnames[4] = "e2"; latex_paramnames[4] = "e"; latex_param_subscripts[4] = "2";
	} else {
		paramnames[3] = "q"; latex_paramnames[3] = "q"; latex_param_subscripts[3] = "";
		paramnames[4] = "theta"; latex_paramnames[4] = "\\theta"; latex_param_subscripts[4] = "";
	}
	if (!center_anchored) {
		paramnames[5] = "xc"; latex_paramnames[5] = "x"; latex_param_subscripts[5] = "c";
		paramnames[6] = "yc"; latex_paramnames[6] = "y"; latex_param_subscripts[6] = "c";
	}
}

void PseudoJaffe::assign_param_pointers()
{
	param[0] = &b;
	param[1] = &a;
	param[2] = &s;
	param[3] = &q;
	param[4] = &theta;
	if (!center_anchored) {
		param[5] = &x_center;
		param[6] = &y_center;
	}
}

void PseudoJaffe::assign_special_anchored_parameters(LensProfile *host_in)
{
	anchor_special_parameter = true;
	tidal_host = host_in;
	double rm, ravg;
	tidal_host->get_einstein_radius(rm,ravg,1.0);
	a = sqrt(ravg*b/sqrt(q)); // this is an approximate formula (a' = sqrt(b'*Re_halo)) and assumes the subhalo is found roughly near the Einstein radius of the halo
	asq = a*a;
}

void PseudoJaffe::update_special_anchored_params()
{
	if (anchor_special_parameter) {
		double rm, ravg;
		tidal_host->get_einstein_radius(rm,ravg,1.0);
		a = sqrt(ravg*b/sqrt(q)); // this is an approximate formula (a' = sqrt(b'*Re_halo)) and assumes the subhalo is found roughly near the Einstein radius of the halo
		asq = a*a;
	}
}

void PseudoJaffe::get_parameters(double* params)
{
	params[0] = b*sqrt(q);
	params[1] = a*sqrt(q);
	params[2] = s*sqrt(q);
	if (use_ellipticity_components) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[3] = (1-q)*cos(2*theta_eff);
		params[4] = (1-q)*sin(2*theta_eff);
	} else {
		params[3] = q;
		params[4] = radians_to_degrees(theta);
	}
	params[5] = x_center;
	params[6] = y_center;
}

void PseudoJaffe::update_parameters(const double* params)
{
	if (use_ellipticity_components) {
		q = 1 - sqrt(SQR(params[3]) + SQR(params[4]));
		set_angle_from_components(params[3],params[4]);
	} else {
		q=params[3];
		set_angle(params[4]);
	}
	b = params[0]/sqrt(q);
	if (!anchor_special_parameter) a = params[1]/sqrt(q);
	else a = a/sqrt(q); // the average tidal radius (a') may not have changed, but q has changed, so update a
	s = params[2]/sqrt(q);
	if (!center_anchored) {
		x_center = params[5];
		y_center = params[6];
	}
	qsq=q*q; ssq=s*s; asq=a*a;
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

void PseudoJaffe::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		double b_prime, a_prime, s_prime;
		if (vary_params[0]) b_prime = fitparams[index++];
		else b_prime = b*sqrt(q);
		if (vary_params[1]) a_prime = fitparams[index++];
		else a_prime = a*sqrt(q);
		if (vary_params[2]) s_prime = fitparams[index++];
		else s_prime = s*sqrt(q);

		if (use_ellipticity_components) {
			if ((vary_params[3]) or (vary_params[4])) {
				double e_1, e_2;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[3]) e_1 = fitparams[index++];
				else e_1 = (1-q)*cos(2*theta_eff);
				if (vary_params[4]) e_2 = fitparams[index++];
				else e_2 = (1-q)*sin(2*theta_eff);
				q = 1 - sqrt(SQR(e_1) + SQR(e_2));
				set_angle_from_components(e_1,e_2);
				qsq=q*q;
				if ((q <= 0) or (q > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
		} else {
			if (vary_params[3]) {
				if ((fitparams[index] <= 0) or (fitparams[index] > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
				qsq=q*q;
			}
			if (vary_params[4]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[5]) x_center = fitparams[index++];
			if (vary_params[6]) y_center = fitparams[index++];
		}

		if ((vary_params[0]) or (vary_params[3]) or ((use_ellipticity_components) and (vary_params[4]))) b = b_prime/sqrt(q);
		if ((vary_params[1]) or (vary_params[3]) or ((use_ellipticity_components) and (vary_params[4]))) {
			a = a_prime/sqrt(q);
			if (a < 0) a = -a; // don't allow negative tidal radii
			asq=a*a;
		}
		if ((vary_params[2]) or (vary_params[3]) or ((use_ellipticity_components) and (vary_params[4]))) {
			s = s_prime/sqrt(q);
			if (s < 0) s = -s; // don't allow negative core radii
			ssq=s*s;
		}

		set_integration_pointers(); // we do this because the potential is still calculated using integration, although analytic formulas can be used--add this in later
		set_model_specific_integration_pointers();
	}
}

void PseudoJaffe::set_model_specific_integration_pointers()
{
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&PseudoJaffe::deflection_spherical_r);
	if (q==1.0) {
		potptr = static_cast<double (LensProfile::*)(const double,const double)> (&PseudoJaffe::potential_spherical);
	} else {
		defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&PseudoJaffe::deflection_elliptical);
		hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&PseudoJaffe::hessian_elliptical);
		potptr = static_cast<double (LensProfile::*)(const double,const double)> (&PseudoJaffe::potential_elliptical);
	}
}

void PseudoJaffe::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = b*sqrt(q);
	if (vary_params[1]) fitparams[index++] = a*sqrt(q);
	if (vary_params[2]) fitparams[index++] = s*sqrt(q);
	if (use_ellipticity_components) {
		if (vary_params[3]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*cos(2*theta_eff);
		}
		if (vary_params[4]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*sin(2*theta_eff);
		}
	} else {
		if (vary_params[3]) fitparams[index++] = q;
		if (vary_params[4]) fitparams[index++] = radians_to_degrees(theta);
		}
	if (!center_anchored) {
		if (vary_params[5]) fitparams[index++] = x_center;
		if (vary_params[6]) fitparams[index++] = y_center;
	}
}

void PseudoJaffe::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.5*b*sqrt(q);
	if (vary_params[1]) stepsizes[index++] = 0.2*b*sqrt(q);
	if (vary_params[2]) stepsizes[index++] = 0.02*b*sqrt(q); // this one is a bit arbitrary, but hopefully reasonable enough
	if (use_ellipticity_components) {
		if (vary_params[3]==true) stepsizes[index++] = 0.1;
		if (vary_params[4]==true) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[3]) stepsizes[index++] = 0.2;
		if (vary_params[4]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[5]) stepsizes[index++] = 0.5*b*sqrt(q);
		if (vary_params[6]) stepsizes[index++] = 0.5*b*sqrt(q);
	}
}

double PseudoJaffe::kappa_rsq(const double rsq)
{
	return (0.5 * b * (pow(ssq+rsq, -0.5) - pow(asq+rsq,-0.5)));
}

double PseudoJaffe::kappa_rsq_deriv(const double rsq)
{
	return (-0.25 * b * (pow(ssq+rsq, -1.5) - pow(asq+rsq,-1.5)));
}

double PseudoJaffe::deflection_spherical_r(const double r)
{
	double rsq = r*r;
	return b*((sqrt(ssq+rsq)-s) - (sqrt(asq+rsq)-a))/r;
}

void PseudoJaffe::deflection_elliptical(const double x, const double y, lensvector& def)
{
	double psi, psi2, u;
	psi = sqrt(qsq*(ssq+x*x)+y*y);
	psi2 = sqrt(qsq*(asq+x*x)+y*y);
	u = sqrt(1-qsq);

	def[0] = (b*q/u)*(atan(u*x/(psi+s)) - atan(u*x/(psi2+a)));
	def[1] = (b*q/u)*(atanh(u*y/(psi+qsq*s)) - atanh(u*y/(psi2+qsq*a)));
}

void PseudoJaffe::hessian_elliptical(const double x, const double y, lensmatrix& hess)
{
	double xsq, ysq, psi, tmp1, psi2, tmp2;
	xsq=x*x; ysq=y*y;
	psi = sqrt(qsq*(ssq+xsq)+ysq);
	tmp1 = ((b*q)/psi)/(xsq+ysq+2*psi*s+ssq*(1+qsq));

	psi2 = sqrt(qsq*(asq+xsq)+ysq);
	tmp2 = ((b*q)/psi2)/(xsq+ysq+2*psi2*a+asq*(1+qsq));

	hess[0][0] = tmp1*(ysq+s*psi+ssq*qsq) - tmp2*(ysq+a*psi2+asq*qsq);
	hess[1][1] = tmp1*(xsq+s*psi+ssq) - tmp2*(xsq+a*psi2+asq);
	hess[0][1] = (-tmp1+tmp2)*x*y;
	hess[1][0] = hess[0][1];
}

double PseudoJaffe::potential_spherical(const double x, const double y)
{
	double rsq, tmp;
	rsq = x*x+y*y;
	tmp = b*(sqrt(ssq+rsq) - s - sqrt(asq+rsq) + a); // now, tmp = kappa_average*rsq
	tmp += b*(a*log((a + sqrt(asq+rsq))/(2.0*a)) - s*log((s + sqrt(ssq+rsq))/(2.0*s)));
	return tmp;
}

double PseudoJaffe::potential_elliptical(const double x, const double y)
{
	double psi, psi2, u;
	psi = sqrt(qsq*(ssq+x*x)+y*y);
	psi2 = sqrt(qsq*(asq+x*x)+y*y);
	u = sqrt(1-qsq);

	return (b*q/u)*(x*(atan(u*x/(psi+s)) - atan(u*x/(psi2+a)))+ y*(atanh(u*y/(psi+qsq*s)) - atanh(u*y/(psi2+qsq*a)))) + b*q*(s*(-log(SQR(psi+s) + SQR(u*x))/2 + log((1.0+q)*s)) - a*(-log(SQR(psi2+a) + SQR(u*x))/2 + log((1.0+q)*a)));
}

void PseudoJaffe::print_parameters()
{
	double b_prime, s_prime, a_prime;
	b_prime = b*sqrt(q);
	s_prime = s*sqrt(q);
	a_prime = a*sqrt(q);
	if (use_ellipticity_components) {
		double e_1, e_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		e_1 = (1-q)*cos(2*theta_eff);
		e_2 = (1-q)*sin(2*theta_eff);
		cout << "pjaffe: b=" << b_prime << ", a=" << a_prime << ", s=" << s_prime << ", e1=" << e_1 << ", e2=" << e_2 << ", center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "pjaffe: b=" << b_prime << ", a=" << a_prime << ", s=" << s_prime << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (anchor_special_parameter) cout << " (tidal radius a set by lens " << tidal_host->lens_number << ")";
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/********************************** NFW **********************************/

NFW::NFW(const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = nfw;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(6);
	assign_param_pointers();
	ks = ks_in; rs = rs_in;
	set_default_base_values(nn,acc);
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under deflection_spherical(...) function)
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_paramnames();
	if (q > 1) q = 1.0; // don't allow q>1
	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::deflection_spherical_r);
}

NFW::NFW(const NFW* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	assign_paramnames();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	ks = lens_in->ks;
	rs = lens_in->rs;
	if (rs < 0) rs = -rs; // don't allow negative core radii
	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (q > 1) q = 1.0; // don't allow q>1
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under deflection_spherical(...) function)
	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::deflection_spherical_r);
}

void NFW::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	if (use_ellipticity_components) {
		paramnames[2] = "e1"; latex_paramnames[2] = "e"; latex_param_subscripts[2] = "1";
		paramnames[3] = "e2"; latex_paramnames[3] = "e"; latex_param_subscripts[3] = "2";
	} else {
		paramnames[2] = "q"; latex_paramnames[2] = "q"; latex_param_subscripts[2] = "";
		paramnames[3] = "theta"; latex_paramnames[3] = "\\theta"; latex_param_subscripts[3] = "";
	}
	if (!center_anchored) {
		paramnames[4] = "xc"; latex_paramnames[4] = "x"; latex_param_subscripts[4] = "c";
		paramnames[5] = "yc"; latex_paramnames[5] = "y"; latex_param_subscripts[5] = "c";
	}
}

void NFW::assign_param_pointers()
{
	param[0] = &ks;
	param[1] = &rs;
	param[2] = &q;
	param[3] = &theta;
	if (!center_anchored) {
		param[4] = &x_center;
		param[5] = &y_center;
	}
}

void NFW::get_parameters(double* params)
{
	params[0] = ks;
	params[1] = rs;
	if (use_ellipticity_components) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[2] = (1-q)*cos(2*theta_eff);
		params[3] = (1-q)*sin(2*theta_eff);
	} else {
		params[2] = q;
		params[3] = radians_to_degrees(theta);
	}
	params[4] = x_center;
	params[5] = y_center;
}

void NFW::update_parameters(const double* params)
{
	ks=params[0];
	rs=params[1];
	if (use_ellipticity_components) {
		q = 1 - sqrt(SQR(params[2]) + SQR(params[3]));
		set_angle_from_components(params[2],params[3]);
	} else {
		q=params[2];
		set_angle(params[3]);
	}
	if (!center_anchored) {
		x_center = params[4];
		y_center = params[5];
	}
	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::deflection_spherical_r);
}

void NFW::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (vary_params[0]) {
			ks = fitparams[index++];
			if (ks < 0) {
				status = false;
				ks = -ks; // don't allow negative kappa
			}
		}
		if (vary_params[1]) {
			rs = fitparams[index++];
			if (rs < 0) {
				status = false;
				rs = -rs; // don't allow negative core radii
			}
		}
		if (use_ellipticity_components) {
			if ((vary_params[2]) or (vary_params[3])) {
				double e_1, e_2;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[2]) e_1 = fitparams[index++];
				else e_1 = (1-q)*cos(2*theta_eff);
				if (vary_params[3]) e_2 = fitparams[index++];
				else e_2 = (1-q)*sin(2*theta_eff);
				q = 1 - sqrt(SQR(e_1) + SQR(e_2));
				set_angle_from_components(e_1,e_2);
				if ((q <= 0) or (q > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
		} else {
			if (vary_params[2]) {
				if ((fitparams[index] <= 0) or (fitparams[index] > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
			if (vary_params[3]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[4]) x_center = fitparams[index++];
			if (vary_params[5]) y_center = fitparams[index++];
		}

		set_integration_pointers();
		defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::deflection_spherical_r);
	}
}

void NFW::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = ks;
	if (vary_params[1]) fitparams[index++] = rs;
	if (use_ellipticity_components) {
		if (vary_params[2]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*cos(2*theta_eff);
		}
		if (vary_params[3]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*sin(2*theta_eff);
		}
	} else {
		if (vary_params[2]) fitparams[index++] = q;
		if (vary_params[3]) fitparams[index++] = radians_to_degrees(theta);
	}
	if (!center_anchored) {
		if (vary_params[4]) fitparams[index++] = x_center;
		if (vary_params[5]) fitparams[index++] = y_center;
	}
}

void NFW::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.2*ks;
	if (vary_params[1]) stepsizes[index++] = 0.2*rs;
	if (use_ellipticity_components) {
		if (vary_params[2]) stepsizes[index++] = 0.1;
		if (vary_params[3]) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[2]) stepsizes[index++] = 0.2;
		if (vary_params[3]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[4]) stepsizes[index++] = 1.0; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
		if (vary_params[5]) stepsizes[index++] = 1.0;
	}
}

double NFW::kappa_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (xsq==1) return (2*ks/3.0); // note, ks is defined as ks = rho_s * r_s / sigma_crit
	else return (2*ks*(1 - lens_function_xsq(xsq))/(xsq - 1));
}

double NFW::kappa_rsq_deriv(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (abs(xsq-1.0) < 1e-5) return -0.4*sqrt(xsq); // derivative function on next line becomes unstable for x very close to 1, this fixes the instability
	else return -(ks/rsq)*((xsq*(2.0-3*lens_function_xsq(xsq)) + 1)/((xsq-1)*(xsq-1)));
}

inline double NFW::lens_function_xsq(const double &xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}

double NFW::deflection_spherical_r(const double r)
{
	double tmp = SQR(r/rs);
	// warning: below tmp ~ 10^-6 or so, this becomes inaccurate due to fine cancellations; a series expansion should be done for tmp smaller than this (do later)
	return 2*ks*r*(2*lens_function_xsq(tmp) + log(tmp/4))/tmp;
}

void NFW::print_parameters()
{
	if (use_ellipticity_components) {
		double e_1, e_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		e_1 = (1-q)*cos(2*theta_eff);
		e_2 = (1-q)*sin(2*theta_eff);
		cout << "nfw: ks=" << ks << ", rs=" << rs << ", e1=" << e_1 << ", e2=" << e_2 << ", center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "nfw: ks=" << ks << ", rs=" << rs << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/********************************** Truncated_NFW **********************************/

Truncated_NFW::Truncated_NFW(const double &ks_in, const double &rs_in, const double &rt_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = TRUNCATED_nfw;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(7);
	assign_param_pointers();
	ks = ks_in; rs = rs_in; rt = rt_in;
	set_default_base_values(nn,acc);
	assign_paramnames();
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under deflection_spherical(...) function)
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	if (q > 1) q = 1.0; // don't allow q>1
	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Truncated_NFW::deflection_spherical_r);
}

Truncated_NFW::Truncated_NFW(const Truncated_NFW* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	assign_paramnames();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	ks = lens_in->ks;
	rs = lens_in->rs;
	rt = lens_in->rt;
	if (rs < 0) rs = -rs; // don't allow negative core radii
	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (q > 1) q = 1.0; // don't allow q>1
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under NFW deflection_spherical(...) function)
	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Truncated_NFW::deflection_spherical_r);
}

void Truncated_NFW::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	paramnames[2] = "rt"; latex_paramnames[2] = "r"; latex_param_subscripts[2] = "t";
	if (use_ellipticity_components) {
		paramnames[3] = "e1"; latex_paramnames[3] = "e"; latex_param_subscripts[3] = "1";
		paramnames[4] = "e2"; latex_paramnames[4] = "e"; latex_param_subscripts[4] = "2";
	} else {
		paramnames[3] = "q"; latex_paramnames[3] = "q"; latex_param_subscripts[3] = "";
		paramnames[4] = "theta"; latex_paramnames[4] = "\\theta"; latex_param_subscripts[4] = "";
	}
	if (!center_anchored) {
		paramnames[5] = "xc"; latex_paramnames[5] = "x"; latex_param_subscripts[5] = "c";
		paramnames[6] = "yc"; latex_paramnames[6] = "y"; latex_param_subscripts[6] = "c";
	}

}

void Truncated_NFW::assign_param_pointers()
{
	param[0] = &ks;
	param[1] = &rs;
	param[2] = &rt;
	param[3] = &q;
	param[4] = &theta;
	if (!center_anchored) {
		param[5] = &x_center;
		param[6] = &y_center;
	}
}

void Truncated_NFW::get_parameters(double* params)
{
	params[0] = ks;
	params[1] = rs;
	params[2] = rt;
	if (use_ellipticity_components) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[3] = (1-q)*cos(2*theta_eff);
		params[4] = (1-q)*sin(2*theta_eff);
	} else {
		params[3] = q;
		params[4] = radians_to_degrees(theta);
	}
	params[5] = x_center;
	params[6] = y_center;
}

void Truncated_NFW::update_parameters(const double* params)
{
	ks=params[0];
	rs=params[1];
	rt=params[2];
	if (use_ellipticity_components) {
		q = 1 - sqrt(SQR(params[3]) + SQR(params[4]));
		set_angle_from_components(params[3],params[4]);
	} else {
		q=params[3];
		set_angle(params[4]);
	}
	if (!center_anchored) {
		x_center = params[5];
		y_center = params[6];
	}
	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Truncated_NFW::deflection_spherical_r);
}

void Truncated_NFW::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (vary_params[0]) {
			ks = fitparams[index++];
			if (ks < 0) ks = -ks; // don't allow negative kappa
		}
		if (vary_params[1]) {
			rs = fitparams[index++];
			if (rs < 0) rs = -rs; // don't allow negative core radii
		}
		if (vary_params[2]) {
			rt = fitparams[index++];
			if (rt < 0) rt = -rt; // don't allow negative tidal radii
		}
		if (use_ellipticity_components) {
			if ((vary_params[3]) or (vary_params[4])) {
				double e_1, e_2;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[3]) e_1 = fitparams[index++];
				else e_1 = (1-q)*cos(2*theta_eff);
				if (vary_params[4]) e_2 = fitparams[index++];
				else e_2 = (1-q)*sin(2*theta_eff);
				q = 1 - sqrt(SQR(e_1) + SQR(e_2));
				set_angle_from_components(e_1,e_2);
				if ((q <= 0) or (q > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
		} else {
			if (vary_params[3]) {
				if ((fitparams[index] <= 0) or (fitparams[index] > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
			if (vary_params[4]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[5]) x_center = fitparams[index++];
			if (vary_params[6]) y_center = fitparams[index++];
		}

		set_integration_pointers();
		defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Truncated_NFW::deflection_spherical_r);
	}
}

void Truncated_NFW::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = ks;
	if (vary_params[1]) fitparams[index++] = rs;
	if (vary_params[2]) fitparams[index++] = rt;
	if (use_ellipticity_components) {
		if (vary_params[3]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*cos(2*theta_eff);
		}
		if (vary_params[4]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*sin(2*theta_eff);
		}
	} else {
		if (vary_params[3]) fitparams[index++] = q;
		if (vary_params[4]) fitparams[index++] = radians_to_degrees(theta);
	}
	if (!center_anchored) {
		if (vary_params[5]) fitparams[index++] = x_center;
		if (vary_params[6]) fitparams[index++] = y_center;
	}
}

void Truncated_NFW::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.2*ks;
	if (vary_params[1]) stepsizes[index++] = 0.2*rs;
	if (vary_params[2]) stepsizes[index++] = 0.2*rt;
	if (use_ellipticity_components) {
		if (vary_params[3]==true) stepsizes[index++] = 0.1;
		if (vary_params[4]==true) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[3]) stepsizes[index++] = 0.1;
		if (vary_params[4]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[5]) stepsizes[index++] = 1.0; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
		if (vary_params[6]) stepsizes[index++] = 1.0;
	}
}

double Truncated_NFW::kappa_rsq(const double rsq)
{
	double xsq, tsq, sqrttx, lx, tmp;
	xsq = rsq/(rs*rs);
	tsq = SQR(rt/rs);
	sqrttx = sqrt(tsq+xsq);
	lx = log(sqrt(xsq)/(sqrttx+sqrt(tsq)));
	if (xsq==1) tmp = 2*(tsq+1)/3.0 + 8.0 + (tsq*tsq-1)/tsq/(tsq+1) + (-M_PI*(4*(tsq+1)+tsq+1) + (tsq*(tsq*tsq-1) + (tsq+1)*(3*tsq*tsq-6*tsq-1))*lx/CUBE(rt/rs))/CUBE(sqrttx);
	else tmp = 2*(tsq+1)/(xsq-1)*(1-lens_function_xsq(xsq)) + 8*lens_function_xsq(xsq) + (tsq*tsq-1)/tsq/(tsq+xsq) + (-M_PI*(4*(tsq+xsq)+tsq+1) + (tsq*(tsq*tsq-1) + (tsq+xsq)*(3*tsq*tsq-6*tsq-1))*lx/CUBE(rt/rs))/CUBE(sqrttx);
	return ks*tsq*tsq/CUBE(tsq+1)*tmp;
}

inline double Truncated_NFW::lens_function_xsq(const double &xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}

double Truncated_NFW::deflection_spherical_r(const double r)
{
	double xsq, tau, tsq, sqrttx, lx, tmp;
	xsq = SQR(r/rs);
	tau = rt/rs;
	tsq = tau*tau;
	sqrttx = sqrt(tsq+xsq);
	lx = log(sqrt(xsq)/(sqrttx+sqrt(tsq)));
	tmp = 2*(tsq+1+4*(xsq-1))*lens_function_xsq(xsq) + (M_PI*(3*tsq-1) + 2*tau*(tsq-3)*log(tau))/tau + (-CUBE(tau)*M_PI*(4*(tsq+xsq)-tsq-1) + (-tsq*(tsq*tsq-1) + (tsq+xsq)*(3*tsq*tsq-6*tsq-1))*lx)/CUBE(tau)/sqrttx;
	return 2*r*ks*tsq*tsq/CUBE(tsq+1)/xsq*tmp; // now, tmp = kappa_average
}

/*
void Truncated_NFW::deflection_spherical(const double x, const double y, lensvector& def)
{
	double xsq, tau, tsq, sqrttx, lx, tmp;
	xsq = (x*x+y*y)/(rs*rs);
	tau = rt/rs;
	tsq = tau*tau;
	sqrttx = sqrt(tsq+xsq);
	lx = log(sqrt(xsq)/(sqrttx+sqrt(tsq)));
	tmp = 2*(tsq+1+4*(xsq-1))*lens_function_xsq(xsq) + (M_PI*(3*tsq-1) + 2*tau*(tsq-3)*log(tau))/tau + (-CUBE(tau)*M_PI*(4*(tsq+xsq)-tsq-1) + (-tsq*(tsq*tsq-1) + (tsq+xsq)*(3*tsq*tsq-6*tsq-1))*lx)/CUBE(tau)/sqrttx;
	tmp = 2*ks*tsq*tsq/CUBE(tsq+1)/xsq*tmp; // now, tmp = kappa_average

	def[0] = tmp*x;
	def[1] = tmp*y;
}
*/

void Truncated_NFW::hessian_spherical(const double x, const double y, lensmatrix& hess)
{
	double rsq, xsq, tau, tsq, sqrttx, lx, kappa_avg, r_dfdr;
	rsq = x*x+y*y;
	xsq = rsq/(rs*rs);
	tau = rt/rs;
	tsq = tau*tau;
	sqrttx = sqrt(tsq+xsq);
	lx = log(sqrt(xsq)/(sqrttx+sqrt(tsq)));
	kappa_avg = 2*(tsq+1+4*(xsq-1))*lens_function_xsq(xsq) + (M_PI*(3*tsq-1) + 2*tau*(tsq-3)*log(tau))/tau + (-CUBE(tau)*M_PI*(4*(tsq+xsq)-tsq-1) + (-tsq*(tsq*tsq-1) + (tsq+xsq)*(3*tsq*tsq-6*tsq-1))*lx)/CUBE(tau)/sqrttx;
	kappa_avg = 2*ks*tsq*tsq/CUBE(tsq+1)/xsq*kappa_avg; // kappa_avg = deflection(r)/r
	r_dfdr = 2*(kappa_rsq(rsq) - kappa_avg)/rsq; // Here, r_dfdr = (1/r)*d/dr(kappa_avg)

	hess[0][0] = kappa_avg + x*x*r_dfdr;
	hess[1][1] = kappa_avg + y*y*r_dfdr;
	hess[0][1] = x*y*r_dfdr;
	hess[1][0] = hess[0][1];
}

void Truncated_NFW::print_parameters()
{
	if (use_ellipticity_components) {
		double e_1, e_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		e_1 = (1-q)*cos(2*theta_eff);
		e_2 = (1-q)*sin(2*theta_eff);
		cout << "tnfw: ks=" << ks << ", rs=" << rs << ", rt=" << rt << ", e1=" << e_1 << ", e2=" << e_2 << ", center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "tnfw: ks=" << ks << ", rs=" << rs << ", rt=" << rt << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/********************************** Hernquist **********************************/

Hernquist::Hernquist(const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = HERNQUIST;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(6);
	assign_param_pointers();
	ks = ks_in; rs = rs_in;
	set_default_base_values(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_paramnames();
	if (q > 1) q = 1.0; // don't allow q>1
	set_integration_pointers();
	// NOTE: for q=1, the deflection has an analytic formula. Implement this later!
}

Hernquist::Hernquist(const Hernquist* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	assign_paramnames();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	ks = lens_in->ks;
	rs = lens_in->rs;
	if (rs < 0) rs = -rs; // don't allow negative core radii
	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (q > 1) q = 1.0; // don't allow q>1
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);
	set_integration_pointers();
}

void Hernquist::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	if (use_ellipticity_components) {
		paramnames[2] = "e1"; latex_paramnames[2] = "e"; latex_param_subscripts[2] = "1";
		paramnames[3] = "e2"; latex_paramnames[3] = "e"; latex_param_subscripts[3] = "2";
	} else {
		paramnames[2] = "q"; latex_paramnames[2] = "q"; latex_param_subscripts[2] = "";
		paramnames[3] = "theta"; latex_paramnames[3] = "\\theta"; latex_param_subscripts[3] = "";
	}
	if (!center_anchored) {
		paramnames[4] = "xc"; latex_paramnames[4] = "x"; latex_param_subscripts[4] = "c";
		paramnames[5] = "yc"; latex_paramnames[5] = "y"; latex_param_subscripts[5] = "c";
	}
}

void Hernquist::assign_param_pointers()
{
	param[0] = &ks;
	param[1] = &rs;
	param[2] = &q;
	param[3] = &theta;
	if (!center_anchored) {
		param[4] = &x_center;
		param[5] = &y_center;
	}
}

void Hernquist::get_parameters(double* params)
{
	params[0] = ks;
	params[1] = rs;
	if (use_ellipticity_components) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[2] = (1-q)*cos(2*theta_eff);
		params[3] = (1-q)*sin(2*theta_eff);
	} else {
		params[2] = q;
		params[3] = radians_to_degrees(theta);
	}
	params[4] = x_center;
	params[5] = y_center;
}

void Hernquist::update_parameters(const double* params)
{
	ks=params[0];
	rs=params[1];
	if (use_ellipticity_components) {
		q = 1 - sqrt(SQR(params[2]) + SQR(params[3]));
		set_angle_from_components(params[2],params[3]);
	} else {
		q=params[2];
		set_angle(params[3]);
	}
	if (!center_anchored) {
		x_center = params[4];
		y_center = params[5];
	}
	set_integration_pointers();
}

void Hernquist::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (vary_params[0]) ks = fitparams[index++];
		if (vary_params[1]) {
			rs = fitparams[index++];
			if (rs < 0) rs = -rs; // don't allow negative core radii
		}
		if (use_ellipticity_components) {
			if ((vary_params[2]) or (vary_params[3])) {
				double e_1, e_2;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[2]) e_1 = fitparams[index++];
				else e_1 = (1-q)*cos(2*theta_eff);
				if (vary_params[3]) e_2 = fitparams[index++];
				else e_2 = (1-q)*sin(2*theta_eff);
				q = 1 - sqrt(SQR(e_1) + SQR(e_2));
				set_angle_from_components(e_1,e_2);
				if ((q <= 0) or (q > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
		} else {
			if (vary_params[2]) {
				if ((fitparams[index] <= 0) or (fitparams[index] > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
			if (vary_params[3]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[4]) x_center = fitparams[index++];
			if (vary_params[5]) y_center = fitparams[index++];
		}

		set_integration_pointers();
	}
}

void Hernquist::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = ks;
	if (vary_params[1]) fitparams[index++] = rs;
	if (use_ellipticity_components) {
		if (vary_params[2]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*cos(2*theta_eff);
		}
		if (vary_params[3]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*sin(2*theta_eff);
		}
	} else {
		if (vary_params[2]) fitparams[index++] = q;
		if (vary_params[3]) fitparams[index++] = radians_to_degrees(theta);
	}
	if (!center_anchored) {
		if (vary_params[4]) fitparams[index++] = x_center;
		if (vary_params[5]) fitparams[index++] = y_center;
	}
}

void Hernquist::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.2*ks;
	if (vary_params[1]) stepsizes[index++] = 0.2*rs;
	if (use_ellipticity_components) {
		if (vary_params[2]) stepsizes[index++] = 0.1;
		if (vary_params[3]) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[2]) stepsizes[index++] = 0.2;
		if (vary_params[3]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[4]) stepsizes[index++] = 1.0; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
		if (vary_params[5]) stepsizes[index++] = 1.0;
	}
}

double Hernquist::kappa_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (abs(xsq-1.0) < 1e-4) return 0.4*(0.666666666667 - 0.571428571429*(xsq-1)); // function on next line becomes unstable for x very close to 1; this fixes the instability
	return (ks*(-3 + (2+xsq)*lens_function_xsq(xsq))/((xsq-1)*(xsq-1)));
}

double Hernquist::kappa_rsq_deriv(const double rsq)
{
	double xsq = rsq/(rs*rs);
	return ((ks/((2*rsq)*CUBE(xsq-1))) * (-3*xsq*lens_function_xsq(xsq)*(xsq+4) + 13*xsq + 2));
}

inline double Hernquist::lens_function_xsq(const double xsq)
{
	return ((sqrt(xsq) > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (sqrt(xsq) < 1.0) ? (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}

void Hernquist::print_parameters()
{
	if (use_ellipticity_components) {
		double e_1, e_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		e_1 = (1-q)*cos(2*theta_eff);
		e_2 = (1-q)*sin(2*theta_eff);
		cout << "hern: ks=" << ks << ", rs=" << rs << ", e1=" << e_1 << ", e2=" << e_2 << ", center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "hern: ks=" << ks << ", rs=" << rs << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/********************************** Exponential Disk **********************************/

ExpDisk::ExpDisk(const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = EXPDISK;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(6);
	assign_param_pointers();
	k0 = k0_in; R_d = R_d_in;
	set_default_base_values(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_paramnames();
	if (q > 1) q = 1.0; // don't allow q>1
	set_integration_pointers();
}

ExpDisk::ExpDisk(const ExpDisk* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	assign_paramnames();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	k0 = lens_in->k0;
	R_d = lens_in->R_d;
	if (R_d < 0) R_d = -R_d; // don't allow negative core radii
	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (q > 1) q = 1.0; // don't allow q>1
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);
	set_integration_pointers();
}

void ExpDisk::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "k0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	paramnames[1] = "R_d"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "d";
	if (use_ellipticity_components) {
		paramnames[2] = "e1"; latex_paramnames[2] = "e"; latex_param_subscripts[2] = "1";
		paramnames[3] = "e2"; latex_paramnames[3] = "e"; latex_param_subscripts[3] = "2";
	} else {
		paramnames[2] = "q"; latex_paramnames[2] = "q"; latex_param_subscripts[2] = "";
		paramnames[3] = "theta"; latex_paramnames[3] = "\\theta"; latex_param_subscripts[3] = "";
	}
	if (!center_anchored) {
		paramnames[4] = "xc"; latex_paramnames[4] = "x"; latex_param_subscripts[4] = "c";
		paramnames[5] = "yc"; latex_paramnames[5] = "y"; latex_param_subscripts[5] = "c";
	}

}

void ExpDisk::assign_param_pointers()
{
	param[0] = &k0;
	param[1] = &R_d;
	param[2] = &q;
	param[3] = &theta;
	if (!center_anchored) {
		param[4] = &x_center;
		param[5] = &y_center;
	}
}

void ExpDisk::get_parameters(double* params)
{
	params[0] = k0;
	params[1] = R_d;
	if (use_ellipticity_components) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[2] = (1-q)*cos(2*theta_eff);
		params[3] = (1-q)*sin(2*theta_eff);
	} else {
		params[2] = q;
		params[3] = radians_to_degrees(theta);
	}
	params[4] = x_center;
	params[5] = y_center;
}

void ExpDisk::update_parameters(const double* params)
{
	k0=params[0];
	R_d=params[1];
	if (use_ellipticity_components) {
		q = 1 - sqrt(SQR(params[2]) + SQR(params[3]));
		set_angle_from_components(params[2],params[3]);
	} else {
		q=params[2];
		set_angle(params[3]);
	}
	if (!center_anchored) {
		x_center = params[4];
		y_center = params[5];
	}
	set_integration_pointers();
}

void ExpDisk::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (vary_params[0]) {
			k0 = fitparams[index++];
			if (k0 < 0) k0 = -k0; // don't allow negative kappa
		}
		if (vary_params[1]) {
			R_d = fitparams[index++];
			if (R_d < 0) R_d = -R_d; // don't allow negative scale radii
		}
		if (use_ellipticity_components) {
			if ((vary_params[2]) or (vary_params[3])) {
				double e_1, e_2;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[2]) e_1 = fitparams[index++];
				else e_1 = (1-q)*cos(2*theta_eff);
				if (vary_params[3]) e_2 = fitparams[index++];
				else e_2 = (1-q)*sin(2*theta_eff);
				q = 1 - sqrt(SQR(e_1) + SQR(e_2));
				set_angle_from_components(e_1,e_2);
				if ((q <= 0) or (q > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
		} else {
			if (vary_params[2]) {
				if ((fitparams[index] <= 0) or (fitparams[index] > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
			if (vary_params[3]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[4]) x_center = fitparams[index++];
			if (vary_params[5]) y_center = fitparams[index++];
		}

		set_integration_pointers();
	}
}

void ExpDisk::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = k0;
	if (vary_params[1]) fitparams[index++] = R_d;
	if (use_ellipticity_components) {
		if (vary_params[2]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*cos(2*theta_eff);
		}
		if (vary_params[3]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*sin(2*theta_eff);
		}
	} else {
		if (vary_params[2]) fitparams[index++] = q;
		if (vary_params[3]) fitparams[index++] = radians_to_degrees(theta);
	}
	if (!center_anchored) {
		if (vary_params[4]) fitparams[index++] = x_center;
		if (vary_params[5]) fitparams[index++] = y_center;
	}
}

void ExpDisk::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.2*k0;
	if (vary_params[1]) stepsizes[index++] = 0.2*R_d;
	if (use_ellipticity_components) {
		if (vary_params[2]) stepsizes[index++] = 0.1;
		if (vary_params[3]) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[2]) stepsizes[index++] = 0.2;
		if (vary_params[3]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[4]) stepsizes[index++] = 1.0; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
		if (vary_params[5]) stepsizes[index++] = 1.0;
	}
}

double ExpDisk::kappa_rsq(const double rsq)
{
	return (k0*exp(-sqrt(rsq)/R_d)/q);
}

double ExpDisk::kappa_rsq_deriv(const double rsq)
{
	double r = sqrt(rsq);
	return (-k0*exp(-r/R_d)/(q*R_d*2*r));
}

void ExpDisk::print_parameters()
{
	if (use_ellipticity_components) {
		double e_1, e_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		e_1 = (1-q)*cos(2*theta_eff);
		e_2 = (1-q)*sin(2*theta_eff);
		cout << "expdisk: k0=" << k0 << ", R_d=" << R_d << ", e1=" << e_1 << ", e2=" << e_2 << ", center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "expdisk: k0=" << k0 << ", R_d=" << R_d << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/***************************** External shear *****************************/

Shear::Shear(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in)
{
	lenstype=SHEAR;
	defined_spherical_kappa_profile = false;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(4);
	assign_param_pointers();
	assign_paramnames();
	if (use_shear_component_params) {
		double shear, angle;
		shear = sqrt(SQR(shear_p1_in) + SQR(shear_p2_in));
		if (shear_p1_in==0) {
			if (shear_p2_in > 0) angle = M_HALFPI;
			else angle = -M_HALFPI;
		} else {
			angle = atan(abs(shear_p2_in/shear_p1_in));
			if (shear_p1_in < 0) {
				if (shear_p2_in < 0)
					angle = angle - M_PI;
				else
					angle = M_PI - angle;
			} else if (shear_p2_in < 0) {
				angle = -angle;
			}
		}
		angle = 0.5*(angle+M_PI); // the pi/2 phase shift is because the angle is the direction of the perturber, NOT the shear angle
		if (orient_major_axis_north) angle -= M_HALFPI;
		while (angle > M_HALFPI) angle -= M_PI;
		while (angle <= -M_HALFPI) angle += M_PI;
		q=shear;
		set_angle_radians(angle);
		x_center = xc_in;
		y_center = yc_in;
	} else {
		q=shear_p1_in;
		set_angle(shear_p2_in);
		x_center = xc_in;
		y_center = yc_in;
	}
}

Shear::Shear(const Shear* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	assign_paramnames();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
}

void Shear::get_parameters(double* params)
{
	if (use_shear_component_params) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[0] = -q*cos(2*theta_eff);
		params[1] = -q*sin(2*theta_eff);
	} else {
		params[0] = q;
		params[1] = radians_to_degrees(theta);
	}
	params[2] = x_center;
	params[3] = y_center;
}

void Shear::update_parameters(const double* params)
{
	if (use_shear_component_params) {
		q = sqrt(SQR(params[0]) + SQR(params[1]));
		set_angle_from_components(params[0],params[1]);
	} else {
		q=params[0];
		set_angle(params[1]);
	}
	if (!center_anchored) {
		x_center = params[2];
		y_center = params[3];
	}
}

void Shear::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	if (use_shear_component_params) {
		paramnames[0] = "shear1"; latex_paramnames[0] = "\\gamma"; latex_param_subscripts[0] = "1";
		paramnames[1] = "shear2"; latex_paramnames[1] = "\\gamma"; latex_param_subscripts[1] = "2";
	} else {
		paramnames[0] = "shear"; latex_paramnames[0] = "\\gamma"; latex_param_subscripts[0] = "";
		paramnames[1] = "theta_shear"; latex_paramnames[1] = "\\theta"; latex_param_subscripts[1] = "\\gamma";
	}
	if (!center_anchored) {
		paramnames[2] = "xc"; latex_paramnames[2] = "x"; latex_param_subscripts[2] = "c";
		paramnames[3] = "yc"; latex_paramnames[3] = "y"; latex_param_subscripts[3] = "c";
	}
}

void Shear::assign_param_pointers()
{
	param[0] = &q;
	param[1] = &theta;
	if (!center_anchored) {
		param[2] = &x_center;
		param[3] = &y_center;
	}
}

void Shear::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (use_shear_component_params) {
			if ((vary_params[0]) or (vary_params[1])) {
				double shear_x, shear_y;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[0]) shear_x = fitparams[index++];
				else shear_x = -q*cos(2*theta_eff);
				if (vary_params[1]) shear_y = fitparams[index++];
				else shear_y = -q*sin(2*theta_eff);
				q = sqrt(SQR(shear_x) + SQR(shear_y));
				set_angle_from_components(shear_x,shear_y);
			}
		} else {
			if (vary_params[0]) {
				if (fitparams[index] < 0) status = false; // shear < 0 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative magnitude for shear
			}
			if (vary_params[1]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[2]) x_center = fitparams[index++];
			if (vary_params[3]) y_center = fitparams[index++];
		}
	}
}

void Shear::get_fit_parameters(dvector& fitparams, int &index)
{
	if (use_shear_component_params) {
		if (vary_params[0]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			double shear_x = -q*cos(2*theta_eff); // the negative is because theta is the angle of the perturber, NOT the shear angle
			fitparams[index++] = shear_x;
		}
		if (vary_params[1]) {
			double shear_y = -q*sin(2*theta_eff); // the negative is because theta is the angle of the perturber, NOT the shear angle
			fitparams[index++] = shear_y;
		}
	} else {
		if (vary_params[0]) fitparams[index++] = q;
		if (vary_params[1]) fitparams[index++] = radians_to_degrees(theta);
	}
	if (!center_anchored) {
		if (vary_params[2]) fitparams[index++] = x_center;
		if (vary_params[3]) fitparams[index++] = y_center;
	}
}

void Shear::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (use_shear_component_params) {
		if (vary_params[0]) stepsizes[index++] = 0.035;
		if (vary_params[1]) stepsizes[index++] = 0.035;
	} else {
		if (vary_params[0]) stepsizes[index++] = 0.05;
		if (vary_params[1]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[2]) stepsizes[index++] = 0.1; // very arbitrary, but shear is usually center_anchored anyway
		if (vary_params[3]) stepsizes[index++] = 0.1; // very arbitrary, but shear is usually center_anchored anyway
	}
}

double Shear::potential(double x, double y)
{
	x -= x_center;
	y -= y_center;
	theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
	return (0.5*q*(y*y-x*x)*cos(2*theta_eff) - x*y*q*sin(2*theta_eff));
}

void Shear::deflection(double x, double y, lensvector& def)
{
	x -= x_center;
	y -= y_center;
	theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
	def[0] = -q*(x*cos(2*theta_eff) + y*sin(2*theta_eff));
	def[1] = q*(y*cos(2*theta_eff) - x*sin(2*theta_eff));
}

void Shear::hessian(double x, double y, lensmatrix& hess)
{
	// Hessian does not depend on x or y
	theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
	hess[0][0] = -q*cos(2*theta_eff);
	hess[1][1] = -hess[0][0];
	hess[0][1] = -q*sin(2*theta_eff);
	hess[1][0] = hess[0][1];
}

void Shear::set_angle_from_components(const double &shear1, const double &shear2)
{
	double angle;
	if (shear1==0) {
		if (shear2 > 0) angle = M_HALFPI;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(shear2/shear1));
		if (shear1 < 0) {
			if (shear2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (shear2 < 0) {
			angle = -angle;
		}
	}
	angle = 0.5*(angle+M_PI); // the phase shift is because the angle is the direction of the perturber, NOT the shear angle
	if (orient_major_axis_north) angle -= M_HALFPI;
	while (angle > M_HALFPI) angle -= M_PI;
	while (angle <= -M_HALFPI) angle += M_PI;
	set_angle_radians(angle);
}

void Shear::print_parameters()
{
	if (use_shear_component_params) {
		double shear_1, shear_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		shear_1 = -q*cos(2*theta_eff);
		shear_2 = -q*sin(2*theta_eff);
		cout << "external shear: shear_1=" << shear_1 << ", shear_2=" << shear_2 << ", center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "external shear: shear=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/***************************** Multipole term *******************************/

Multipole::Multipole(const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, const bool sine)
{
	lenstype=MULTIPOLE;
	defined_spherical_kappa_profile = false;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(5); // Note, m cannot be varied since it must be an integer, so it is not counted as a parameter here
	assign_param_pointers();
	n = n_in; m = m_in;
	q=A_m_in;
	set_angle(theta_degrees);
	x_center = xc_in;
	y_center = yc_in;
	kappa_multipole = kap; // specifies whether it is a multipole in the potential or in kappa
	sine_term = sine;
	assign_paramnames();
	defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&Multipole::deflection);
	hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&Multipole::hessian);
	defptr_r_spherical = NULL;
	if (m==0) defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Multipole::deflection_m0_spherical_r);
}

Multipole::Multipole(const Multipole* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	n = lens_in->n;
	m = lens_in->m;
	q=lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	kappa_multipole = lens_in->kappa_multipole;
	sine_term = lens_in->sine_term;
	assign_paramnames();
}

void Multipole::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	string mstring;
	stringstream mstr;
	mstr << m;
	mstr >> mstring;
	if (sine_term) {
		paramnames[0] = "B_" + mstring; latex_paramnames[0] = "B"; latex_param_subscripts[0] = mstring;
	} else {
		paramnames[0] =  "A_" + mstring; latex_paramnames[0] = "A"; latex_param_subscripts[0] = mstring;
	}
	if (kappa_multipole) {
		paramnames[1] = "beta"; latex_paramnames[1] = "\\beta"; latex_param_subscripts[1] = "";
	} else {
		paramnames[1] = "n"; latex_paramnames[1] = "n"; latex_param_subscripts[1] = "";
	}
	paramnames[2] = "theta"; latex_paramnames[2] = "\\theta"; latex_param_subscripts[2] = "";
	if (!center_anchored) {
		paramnames[3] = "xc"; latex_paramnames[3] = "x"; latex_param_subscripts[3] = "c";
		paramnames[4] = "yc"; latex_paramnames[4] = "y"; latex_param_subscripts[4] = "c";
	}
}

void Multipole::assign_param_pointers()
{
	param[0] = &q;
	param[1] = &n;
	param[2] = &theta;
	if (!center_anchored) {
		param[3] = &x_center;
		param[4] = &y_center;
	}
}

void Multipole::get_parameters(double* params)
{
	params[0] = q;
	params[1] = n;
	params[2] = radians_to_degrees(theta);
	params[3] = x_center;
	params[4] = y_center;
}

void Multipole::update_parameters(const double* params)
{
	q=params[0];
	n=params[1];
	set_angle(params[2]);
	if (!center_anchored) {
		x_center = params[3];
		y_center = params[4];
	}
}

void Multipole::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (vary_params[0]) q = fitparams[index++];
		if (vary_params[1]) n = fitparams[index++];
		if (vary_params[2]) set_angle(fitparams[index++]);
		if (!center_anchored) {
			if (vary_params[3]) x_center = fitparams[index++];
			if (vary_params[4]) y_center = fitparams[index++];
		}
	}
}

void Multipole::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = q;
	if (vary_params[1]) fitparams[index++] = n;
	if (vary_params[2]) fitparams[index++] = radians_to_degrees(theta);
	if (!center_anchored) {
		if (vary_params[3]) fitparams[index++] = x_center;
		if (vary_params[4]) fitparams[index++] = y_center;
	}
}

void Multipole::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.05;
	if (vary_params[1]) stepsizes[index++] = 0.1;
	if (vary_params[2]) stepsizes[index++] = 20;
	if (!center_anchored) {
		if (vary_params[3]) stepsizes[index++] = 0.1; // very arbitrary, but a multipole term is usually center_anchored anyway
		if (vary_params[4]) stepsizes[index++] = 0.1; // very arbitrary, but a multipole term is usually center_anchored anyway
	}
}

double Multipole::kappa(double x, double y)
{
	x -= x_center;
	y -= y_center;
	theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
	if (sine_term) theta_eff += M_HALFPI/m;
	double phi = atan(abs(y/x));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	if (kappa_multipole) {
		return q*pow(x*x+y*y,-n/2) * cos(m*(phi-theta_eff));
	} else {
		if (n==m) return 0;
		else {
			if (m==0)
				return -q*pow(x*x+y*y,n/2-1)*(0.5*(n*n-m*m)) * cos(m*(phi-theta_eff));
			else
				return -q*pow(x*x+y*y,n/2-1)*(0.5*(n*n-m*m)/m) * cos(m*(phi-theta_eff));
		}
	}
}

double Multipole::kappa_rsq(const double rsq)
{
	if (kappa_multipole) {
		if (m==0) return q*pow(rsq,-n/2);
		else return 0; // this model does not have a radial profile, unless n=0
	} else
		return 0;
}

double Multipole::kappa_rsq_deriv(const double rsq)
{
	if (kappa_multipole) {
		if (m==0) return -(n/2)*q*pow(rsq,-n/2-1);
		else return 0; // this model does not have a radial profile, unless n=0
	} else
		return 0;
}

double Multipole::potential(double x, double y)
{
	x -= x_center;
	y -= y_center;
	double phi = atan(abs(y/x));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	if (kappa_multipole) {
		return (2*q*pow(x*x+y*y,1-n/2)/(SQR(2-n)-m*m)) * cos(m*(phi-theta_eff));
	} else {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		if (sine_term) theta_eff += M_HALFPI/m;
		if (m==0)
			return -(q*pow(x*x+y*y,n/2)) * cos(m*(phi-theta_eff));
		else
			return -(q*pow(x*x+y*y,n/2)/m) * cos(m*(phi-theta_eff));
	}
}

void Multipole::deflection(double x, double y, lensvector& def)
{
	x -= x_center;
	y -= y_center;
	theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
	if (sine_term) theta_eff += M_HALFPI/m;
	double r, phi, psi, dpsi, cs, ss;
	r = sqrt(x*x+y*y);
	phi = atan(abs(y/x));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}

	if (kappa_multipole) {
		psi = 2*q*pow(r,2-n)/(SQR(2-n)-m*m);
		dpsi = (2-n)*psi/r;
	} else {
		if (m==0)
			psi = -q*pow(r,n);
		else
			psi = -q*pow(r,n)/m;
		dpsi = n*psi/r;
	}

	cs = cos(m*(phi-theta_eff));
	ss = sin(m*(phi-theta_eff));
	def[0] = dpsi*cs*x/r + psi*m*ss*y/r/r;
	def[1] = dpsi*cs*y/r - psi*m*ss*x/r/r;
}

double Multipole::deflection_m0_spherical_r(const double r)
{
	double ans;
	if (kappa_multipole) {
		ans = 2*q*pow(r,1-n)/(2-n);
	} else {
		ans = -q*pow(r,n-1);
	}
	return ans;
}

void Multipole::hessian(double x, double y, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
	if ((sine_term) and (m != 0)) theta_eff += M_HALFPI/m;
	int mm = m*m;
	double r, rsq, rcube, xy, xx, yy, phi, psi, dpsi, ddpsi, cs, ss;
	xx = x*x;
	yy = y*y;
	rsq = xx+yy;
	r = sqrt(rsq);
	rcube = rsq*r;
	xy = x*y;
	phi = atan(abs(y/x));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}

	if (kappa_multipole) {
		psi = 2*q*pow(r,2-n)/(SQR(2-n)-mm);
		dpsi = (2-n)*psi/r;
		ddpsi = (1-n)*dpsi/r;
	} else {
		if (m==0)
			psi = -q*pow(r,n);
		else
			psi = -q*pow(r,n)/m;
		dpsi = n*psi/r;
		ddpsi = (n-1)*dpsi/r;
	}

	cs = cos(m*(phi-theta_eff));
	ss = sin(m*(phi-theta_eff));
	hess[0][0] = (ddpsi*xx + dpsi*yy/r - psi*mm*yy/rsq)*cs/rsq + (dpsi - psi/r)*2*m*xy*ss/rcube;
	hess[1][1] = (ddpsi*yy + dpsi*xx/r - psi*mm*xx/rsq)*cs/rsq + (-dpsi + psi/r)*2*m*xy*ss/rcube;
	hess[0][1] = (ddpsi - dpsi/r + psi*mm/rsq)*xy*cs/rsq + (dpsi - psi/r)*(yy-xx)*m*ss/rcube;
	hess[1][0] = hess[0][1];
}

void Multipole::get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor)
{
	// this gives the spherically averaged Einstein radius
	if (n==0.0) {
		re_major_axis = 0;
		re_average = 0;
		return;
	}
	double b;
	if (kappa_multipole) {
		if (q < 0) b = 0;
		else b = pow(2*q*zfactor/(2-n),1.0/n);
	} else {
		if (q > 0) b = 0;
		else {
			if (m==0) {
				b = pow(-q*zfactor*n,1.0/(2-n));
			} else {
				b = pow(-(q*zfactor*n)/m,1.0/(2-n));
			}
		}
	}
	re_major_axis = re_average = b;
}

void Multipole::print_parameters()
{
	string normstring, sintype;
	if (sine_term) {
		normstring = "B_m";
		sintype = "sine";
	} else {
		normstring = "A_m";
		sintype = "cosine";
	}
	if (kappa_multipole) {
		cout << "kappa multipole (" << sintype << ", m=" << m << "): " << normstring << "=" << q << ", beta=" << n << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "potential multipole (" << sintype << ", m=" << m << "): " << normstring << "=" << q << ", n=" << n << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/***************************** Point mass *****************************/

PointMass::PointMass(const double &bb, const double &xc_in, const double &yc_in)
{
	lenstype = PTMASS;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(3);
	assign_param_pointers();
	b = bb;
	x_center = xc_in; y_center = yc_in;
	assign_paramnames();
}

PointMass::PointMass(const PointMass* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	b = lens_in->b;
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	assign_paramnames();
}

void PointMass::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "b"; latex_paramnames[0] = "b"; latex_param_subscripts[0] = "";
	if (!center_anchored) {
		paramnames[1] = "xc"; latex_paramnames[1] = "x"; latex_param_subscripts[1] = "c";
		paramnames[2] = "yc"; latex_paramnames[2] = "y"; latex_param_subscripts[2] = "c";
	}
}

void PointMass::assign_param_pointers()
{
	param[0] = &b;
	if (!center_anchored) {
		param[1] = &x_center;
		param[2] = &y_center;
	}
}

void PointMass::get_parameters(double* params)
{
	params[0] = b;
	params[1] = x_center;
	params[2] = y_center;
}

void PointMass::update_parameters(const double* params)
{
	b=params[0];
	if (b < 0) b=-b;
	if (!center_anchored) {
		x_center = params[1];
		y_center = params[2];
	}
}

void PointMass::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (vary_params[0]) b = fitparams[index++];
		if (!center_anchored) {
			if (vary_params[1]) x_center = fitparams[index++];
			if (vary_params[2]) y_center = fitparams[index++];
		}
	}
}

void PointMass::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = b;
	if (!center_anchored) {
		if (vary_params[1]) fitparams[index++] = x_center;
		if (vary_params[2]) fitparams[index++] = y_center;
	}
}

void PointMass::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.1*b;
	if (!center_anchored) {
		if (vary_params[1]) stepsizes[index++] = 0.1*b;
		if (vary_params[2]) stepsizes[index++] = 0.1*b;
	}
}

double PointMass::potential(double x, double y)
{
	x -= x_center;
	y -= y_center;
	return (b*b*log(sqrt(x*x+y*y)));
}

double PointMass::kappa(double x, double y)
{
	return 0; // really it's a delta function, but effectively zero for our purposes here
}

void PointMass::deflection(double x, double y, lensvector& def)
{
	x -= x_center;
	y -= y_center;
	double rsq = x*x + y*y;
	def[0] = b*b*x/rsq;
	def[1] = b*b*y/rsq;
}

void PointMass::hessian(double x, double y, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	double bsq = b*b, xsq = x*x, ysq = y*y, r4 = SQR(xsq + ysq);
	hess[0][0] = bsq*(ysq-xsq)/r4;
	hess[1][1] = -hess[0][0];
	hess[1][0] = -2*bsq*x*y/r4;
	hess[0][1] = hess[1][0];
}

void PointMass::print_parameters()
{
	cout << "point mass: b=" << b << ", center=(" << x_center << "," << y_center << ")";
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/***************************** Core/Cusp Model *****************************/

CoreCusp::CoreCusp(const double &mass_param_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, bool parametrize_einstein_radius)
{
	lenstype=CORECUSP;
	set_n_params(9);
	assign_param_pointers();
	center_anchored = false;
	anchor_special_parameter = false;
	set_default_base_values(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	set_k0_by_einstein_radius = parametrize_einstein_radius;
	gamma = gamma_in;
	n = n_in;
	a = a_in;
	s = s_in;
	if (s < 0) s = -s; // don't allow negative core radii
	if (set_k0_by_einstein_radius) {
		einstein_radius = mass_param_in;
		if (einstein_radius < 0) einstein_radius = -einstein_radius; // don't allow negative einstein radius
		k0 = 1.0; // needed since kappa_avg_spherical includes k0
		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
		if (a != 0) k0 = 1.0 / kappa_avg_spherical_rsq(einstein_radius*einstein_radius);
	}
	else k0 = mass_param_in;

	if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;

	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&CoreCusp::deflection_spherical_r);
	assign_paramnames();
}

CoreCusp::CoreCusp(const CoreCusp* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	tidal_host = lens_in->tidal_host;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	set_k0_by_einstein_radius = lens_in->set_k0_by_einstein_radius;
	if (set_k0_by_einstein_radius) einstein_radius = lens_in->einstein_radius;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	k0 = lens_in->k0;
	gamma = lens_in->gamma;
	n = lens_in->n;
	a = lens_in->a;
	s = lens_in->s;
	if (s < 0) s = -s; // don't allow negative core radii
	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (q > 1) q = 1.0; // don't allow q>1
	if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);

	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&CoreCusp::deflection_spherical_r);
	assign_paramnames();
}

void CoreCusp::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	if (set_k0_by_einstein_radius) { paramnames[0] = "Re"; latex_paramnames[0] = "R"; latex_param_subscripts[0] = "e"; }
	else { paramnames[0] = "k0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0"; }
	paramnames[1] = "gamma"; latex_paramnames[1] = "\\gamma"; latex_param_subscripts[1] = "";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	paramnames[3] = "a"; latex_paramnames[3] = "a"; latex_param_subscripts[3] = "";
	paramnames[4] = "s"; latex_paramnames[4] = "s"; latex_param_subscripts[4] = "";
	if (use_ellipticity_components) {
		paramnames[5] = "e1"; latex_paramnames[5] = "e"; latex_param_subscripts[5] = "1";
		paramnames[6] = "e2"; latex_paramnames[6] = "e"; latex_param_subscripts[6] = "2";
	} else {
		paramnames[5] = "q"; latex_paramnames[5] = "q"; latex_param_subscripts[5] = "";
		paramnames[6] = "theta"; latex_paramnames[6] = "\\theta"; latex_param_subscripts[6] = "";
	}
	if (!center_anchored) {
		paramnames[7] = "xc"; latex_paramnames[7] = "x"; latex_param_subscripts[7] = "c";
		paramnames[8] = "yc"; latex_paramnames[8] = "y"; latex_param_subscripts[8] = "c";
	}
}

void CoreCusp::assign_param_pointers()
{
	param[0] = &k0;
	param[1] = &gamma;
	param[2] = &n;
	param[3] = &a;
	param[4] = &s;
	param[5] = &q;
	param[6] = &theta;
	if (!center_anchored) {
		param[7] = &x_center;
		param[8] = &y_center;
	}
}

void CoreCusp::assign_special_anchored_parameters(LensProfile *host_in)
{
	anchor_special_parameter = true;
	tidal_host = host_in;
	double rm, ravg;
	tidal_host->get_einstein_radius(rm,ravg,1.0);
	if (set_k0_by_einstein_radius) {
		a = sqrt(ravg*einstein_radius); // Not good! Only true for Pseudo-Jaffe subhalo. Fix this later (if it can be fixed)
		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
		k0 = k0 / kappa_avg_spherical_rsq(einstein_radius*einstein_radius);
		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
	} else {
		if (gamma >= 3) a=1e30; // effectively infinite in this case (not physical however, because the mass diverges at the center of the subhalo)
		else a = ravg*k0/(3-gamma); // we have ignored the core in this formulat, but should be reasonable as long as a >> s
	}
	if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
}

void CoreCusp::update_special_anchored_params()
{
	if (anchor_special_parameter) {
		double rm, ravg;
		tidal_host->get_einstein_radius(rm,ravg,1.0);
		if (set_k0_by_einstein_radius) {
			a = sqrt(ravg*einstein_radius);
			if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
			k0 = k0 / kappa_avg_spherical_rsq(einstein_radius*einstein_radius);
			if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
		}
		else {
			if (gamma >= 3) a=1e30; // effectively infinite in this case (not physical however, because the mass diverges at the center of the subhalo)
			else a = ravg*k0/(3-gamma); // we have ignored the core in this formula, but should be reasonable as long as a >> s
		}
		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
	}
}

void CoreCusp::get_parameters(double* params)
{
	if (set_k0_by_einstein_radius) params[0] = einstein_radius;
	else params[0] = k0;
	params[1] = gamma;
	params[2] = n;
	params[3] = a;
	params[4] = s;
	if (use_ellipticity_components) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[5] = (1-q)*cos(2*theta_eff);
		params[6] = (1-q)*sin(2*theta_eff);
	} else {
		params[5] = q;
		params[6] = theta;
	}
	params[7] = x_center;
	params[8] = y_center;
}

void CoreCusp::update_parameters(const double* params)
{
	double a_old;
	gamma = params[1];
	n = params[2];
	a_old = a;
	a = params[3];
	s = params[4];

	if (use_ellipticity_components) {
		q = 1 - sqrt(SQR(params[5]) + SQR(params[6]));
		set_angle_from_components(params[5],params[6]);
	} else {
		q = params[5];
		set_angle(params[6]);
	}
	if (!center_anchored) {
		x_center = params[7];
		y_center = params[8];
	}

	if (set_k0_by_einstein_radius) {
		einstein_radius = params[0];
		double a_new = a;
		if ((anchor_special_parameter) and (a==0)) a = a_old; // the tidal radius will be updated (assuming it is set by host, otherwise a should not be set to zero!)
		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
		k0 = k0 / kappa_avg_spherical_rsq(einstein_radius*einstein_radius);
		a = a_new;
	}
	else k0 = params[0];
	if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;

	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&CoreCusp::deflection_spherical_r);
}

void CoreCusp::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		double mass_param;
		if (vary_params[0]) mass_param = fitparams[index++];
		if (vary_params[1]) gamma = fitparams[index++];
		if (vary_params[2]) n = fitparams[index++];
		if (vary_params[3]) a = fitparams[index++];
		if (vary_params[4]) {
			s = fitparams[index++];
			if (s < 0) s = -s; // don't allow negative core radii
		}

		if (use_ellipticity_components) {
			if ((vary_params[5]) or (vary_params[6])) {
				double e_1, e_2;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[5]) e_1 = fitparams[index++];
				else e_1 = (1-q)*cos(2*theta_eff);
				if (vary_params[6]) e_2 = fitparams[index++];
				else e_2 = (1-q)*sin(2*theta_eff);
				q = 1 - sqrt(SQR(e_1) + SQR(e_2));
				set_angle_from_components(e_1,e_2);
				if ((q <= 0) or (q > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
		} else {
			if (vary_params[5]) {
				if ((fitparams[index] <= 0) or (fitparams[index] > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
			if (vary_params[6]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[7]) x_center = fitparams[index++];
			if (vary_params[8]) y_center = fitparams[index++];
		}

		if (set_k0_by_einstein_radius) {
			if (vary_params[0]) einstein_radius = mass_param;
			if (einstein_radius < 0) einstein_radius = -einstein_radius; // don't allow negative einstein radius
			if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
			k0 = k0 / kappa_avg_spherical_rsq(einstein_radius*einstein_radius); // if the tidal radius is being set by a host galaxy, k0 will be reset once a is updated (a bit redundant)
		}
		else if (vary_params[0]) k0 = mass_param;

		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;

		set_integration_pointers();
		defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&CoreCusp::deflection_spherical_r);
	}
}

void CoreCusp::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) {
		if (set_k0_by_einstein_radius) fitparams[index++] = einstein_radius;
		else fitparams[index++] = k0;
	}
	if (vary_params[1]) fitparams[index++] = gamma;
	if (vary_params[2]) fitparams[index++] = n;
	if (vary_params[3]) fitparams[index++] = a;
	if (vary_params[4]) fitparams[index++] = s;
	if (use_ellipticity_components) {
		if (vary_params[5]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*cos(2*theta_eff);
		}
		if (vary_params[6]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*sin(2*theta_eff);
		}
	} else {
		if (vary_params[5]) fitparams[index++] = q;
		if (vary_params[6]) fitparams[index++] = radians_to_degrees(theta);
	}
	if (!center_anchored) {
		if (vary_params[7]) fitparams[index++] = x_center;
		if (vary_params[8]) fitparams[index++] = y_center;
	}
}

void CoreCusp::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) {
		if (set_k0_by_einstein_radius) stepsizes[index++] = 0.1*einstein_radius;
		else stepsizes[index++] = 0.1*k0;
	}
	if (vary_params[1]) stepsizes[index++] = 0.1;
	if (vary_params[2]) stepsizes[index++] = 0.1;
	if (vary_params[3]) stepsizes[index++] = 0.1*a;
	if (vary_params[4]) stepsizes[index++] = 0.01*a; // this one is a bit arbitrary, but hopefully reasonable enough
	if (use_ellipticity_components) {
		if (vary_params[5]==true) stepsizes[index++] = 0.1;
		if (vary_params[6]==true) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[5]) stepsizes[index++] = 0.1;
		if (vary_params[6]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[7]) stepsizes[index++] = 0.1*k0;
		if (vary_params[8]) stepsizes[index++] = 0.1*k0;
	}
}

double CoreCusp::kappa_rsq(const double rsq)
{
	double aprime = sqrt(a*a-s*s);
	return pow(a/aprime,n) * kappa_rsq_nocore(rsq+s*s,aprime);
}

double CoreCusp::kappa_rsq_nocore(const double rsq_prime, const double aprime)
{
	double ks, xisq, p, hyp, ans;
	p = (n-1.0)/2;
	ks = k0*aprime/(a*M_2PI);
	xisq = rsq_prime/(aprime*aprime);
	hyp = real(hyp_2F1(p,gamma/2,n/2,1/(1+xisq)));
	ans = ks*Beta(p,0.5)*pow(1+xisq,-p)*hyp;
	return ans;
}

double CoreCusp::kappa_rsq_deriv(const double rsq)
{
	double aprime = sqrt(a*a-s*s);
	return pow(a/aprime,n) * kappa_rsq_deriv_nocore(rsq+s*s,aprime);
}

double CoreCusp::kappa_rsq_deriv_nocore(const double rsq_prime, const double aprime)
{
	double ks, xisq, hyp, ans;
	ks = k0*aprime/(a*M_2PI);
	xisq = rsq_prime/(aprime*aprime);
	hyp = n*(1+xisq)*real(hyp_2F1((n-1.0)/2,gamma/2,n/2,1/(1+xisq))) + gamma*real(hyp_2F1((n+1.0)/2,(gamma+2.0)/2,(n+2.0)/2,1/(1+xisq)));
	ans = -(ks/(2*aprime*aprime))*Beta((n+1.0)/2,0.5)*pow(1+xisq,-(n+3.0)/2)*hyp;
	return ans;
}

void CoreCusp::set_core_enclosed_mass()
{
	if (n==3) core_enclosed_mass = enclosed_mass_spherical_nocore_limit(s*s,sqrt(a*a-s*s),nstep);
	else core_enclosed_mass = enclosed_mass_spherical_nocore(s*s,sqrt(a*a-s*s));
}

double CoreCusp::kappa_avg_spherical_rsq(const double rsq)
{
	double r = sqrt(rsq);
	return deflection_spherical_r(r)/r;
}

double CoreCusp::deflection_spherical_r(const double r)
{
	double aprime, def_r, rsq = r*r;
	if (s != 0) {
		aprime = sqrt(a*a-s*s);
		if (n==3) {
			def_r = pow(1-SQR(s/a),-n/2) * (enclosed_mass_spherical_nocore_limit(rsq+s*s,aprime,nstep) - core_enclosed_mass) / r;
		} else {
			def_r = pow(1-SQR(s/a),-n/2) * (enclosed_mass_spherical_nocore(rsq+s*s,aprime) - core_enclosed_mass) / r;
		}
	} else {
		if (n==3) def_r = enclosed_mass_spherical_nocore_limit(rsq,a,nstep) / r;
		else def_r = enclosed_mass_spherical_nocore(rsq,a) / r;
	}
	return def_r;
}

double CoreCusp::enclosed_mass_spherical_nocore(const double rsq_prime, const double aprime, const double nprime) // actually mass_enclosed/(pi*sigma_crit)
{
	double xisq, p, hyp;
	xisq = rsq_prime/(aprime*aprime);
	p = (nprime-3.0)/2;
	hyp = pow(1+xisq,-p) * real(hyp_2F1(p,gamma/2,nprime/2,1/(1+xisq)));

	return 2*k0*CUBE(aprime)/(a*M_2PI) * (Beta(p,(3-gamma)/2) - Beta(p,1.5)*hyp);
}

double CoreCusp::enclosed_mass_spherical_nocore_limit(const double rsq, const double aprime, const double n_stepsize)
{
	// This uses Richardson specialpolation to calculate the enclosed mass, required for the n=3 case
	const double CON=1.4, CON2=(CON*CON);
	const double BIG=1.0e100;
	const double SAFE=2.0;
	const int NTAB=100;
	int i,j;
	double errt,fac,hh,ans=0.0;
	double **a = new double*[NTAB];
	for (i=0; i < NTAB; i++) a[i] = new double[NTAB];

	hh=n_stepsize;
	a[0][0] = 0.5*(enclosed_mass_spherical_nocore(rsq,aprime,n+hh) + enclosed_mass_spherical_nocore(rsq,aprime,n-hh));
	double err=BIG;
	for (i=1;i<NTAB;i++) {
		hh /= CON;
		a[0][i] = 0.5*(enclosed_mass_spherical_nocore(rsq,aprime,n+hh) + enclosed_mass_spherical_nocore(rsq,aprime,n-hh));

		fac=CON2;
		for (j=1;j<=i;j++) {
			a[j][i]=(a[j-1][i]*fac-a[j-1][i-1])/(fac-1.0);
			fac=CON2*fac;
			errt=dmax(abs(a[j][i]-a[j-1][i]),abs(a[j][i]-a[j-1][i-1]));
			if (errt <= err) {
				err=errt;
				ans=a[j][i];
			}
		}
		if (abs(a[i][i]-a[i-1][i-1]) >= SAFE*err) break;
	}
	for (i=0; i < NTAB; i++) delete[] a[i];
	delete[] a;
	return ans;
}

/*
double CoreCusp::kappa_rsq(const double rsq)
{
	Rsq = rsq;
	double (Romberg::*kapptr)(const double);
	kapptr = static_cast<double (Romberg::*)(const double)> (&CoreCusp::kappa_integrand_z);
	double ans = romberg_open(kapptr, 0, a, 1.0e-6, 5) + romberg_improper(kapptr, a, 1e30, 1.0e-6, 5);
	ans *= 2*k0/(a*M_2PI)*pow(a,n);
	return ans;
}

double CoreCusp::kappa_integrand_z(const double z)
{
	return pow(Rsq+s*s+z*z,-gamma/2)*pow(Rsq+a*a+z*z,(gamma-n)/2);
}
*/

void CoreCusp::print_parameters()
{
	cout << "corecusp: ";
	if (set_k0_by_einstein_radius) cout << "Re=" << einstein_radius << " (k0=" << k0 << ")";
	else cout << "k0=" << k0;
	if (use_ellipticity_components) {
		double e_1, e_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		e_1 = (1-q)*cos(2*theta_eff);
		e_2 = (1-q)*sin(2*theta_eff);
		cout << ", gamma=" << gamma << ", n=" << n << ", a=" << a << ", s=" << s << ", e1=" << e_1 << ", e2=" << e_2 << ", center=(" << x_center << "," << y_center << ")";
	} else {
		cout << ", gamma=" << gamma << ", n=" << n << ", a=" << a << ", s=" << s << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

/***************************** SersicLens profile *****************************/


SersicLens::SersicLens(const double &kappa0_in, const double &Re_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype=SERSIC_LENS;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(7);
	assign_param_pointers();
	assign_paramnames();
	set_default_base_values(nn,acc);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	n = n_in;
	re = Re_in;
	double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	k = b*pow(sqrt(q)/re,1.0/n);
	kappa0 = kappa0_in;
	set_default_base_values(nn,acc);
	set_integration_pointers();
}

SersicLens::SersicLens(const SersicLens* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	center_anchor_lens = lens_in->center_anchor_lens;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	assign_paramnames();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	kappa0 = lens_in->kappa0;
	n = lens_in->n;
	re = lens_in->re;
	k = lens_in->k;
	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (q > 1) q = 1.0; // don't allow q>1
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);
	set_integration_pointers();
}

void SersicLens::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "kappa0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	paramnames[1] = "R_eff"; latex_paramnames[1] = "R"; latex_param_subscripts[1] = "eff";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	if (use_ellipticity_components) {
		paramnames[3] = "e1"; latex_paramnames[3] = "e"; latex_param_subscripts[3] = "1";
		paramnames[4] = "e2"; latex_paramnames[4] = "e"; latex_param_subscripts[4] = "2";
	} else {
		paramnames[3] = "q"; latex_paramnames[3] = "q"; latex_param_subscripts[3] = "";
		paramnames[4] = "theta"; latex_paramnames[4] = "\\theta"; latex_param_subscripts[4] = "";
	}
	if (!center_anchored) {
		paramnames[5] = "xc"; latex_paramnames[5] = "x"; latex_param_subscripts[5] = "c";
		paramnames[6] = "yc"; latex_paramnames[6] = "y"; latex_param_subscripts[6] = "c";
	}
}

void SersicLens::assign_param_pointers()
{
	param[0] = &kappa0;
	param[1] = &re;
	param[2] = &n;
	param[3] = &q;
	param[4] = &theta;
	if (!center_anchored) {
		param[5] = &x_center;
		param[6] = &y_center;
	}
}

void SersicLens::get_parameters(double* params)
{
	params[0] = kappa0;
	params[1] = re;
	params[2] = n;
	if (use_ellipticity_components) {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		params[3] = (1-q)*cos(2*theta_eff);
		params[4] = (1-q)*sin(2*theta_eff);
	} else {
		params[3] = q;
		params[4] = radians_to_degrees(theta);
	}
	params[5] = x_center;
	params[6] = y_center;
}

void SersicLens::update_parameters(const double* params)
{
	kappa0=params[0];
	re=params[1];
	n=params[2];
	if (use_ellipticity_components) {
		q = 1 - sqrt(SQR(params[3]) + SQR(params[4]));
		set_angle_from_components(params[3],params[4]);
	} else {
		q=params[3];
		set_angle(params[4]);
	}
	if (!center_anchored) {
		x_center = params[5];
		y_center = params[6];
	}
	double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	k = b*pow(sqrt(q)/re,1.0/n);
	set_integration_pointers();
	//defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&SersicLens::deflection_spherical_r);
}

void SersicLens::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (vary_params[0]) {
			kappa0 = fitparams[index++];
			if (kappa0 < 0) {
				status = false; // kappa < 0 is not a physically acceptable value, so report that we're out of bounds
				kappa0 = -kappa0; // don't allow negative kappa
			}
		}
		if (vary_params[1]) {
			re = fitparams[index++];
			if (re < 0) {
				status = false; // kappa < 0 is not a physically acceptable value, so report that we're out of bounds
				re = -re; // don't allow negative effective radii
			}
		}
		if (vary_params[2]) n = fitparams[index++];
		if (use_ellipticity_components) {
			if ((vary_params[3]) or (vary_params[4])) {
				double e_1, e_2;
				theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
				if (vary_params[3]) e_1 = fitparams[index++];
				else e_1 = (1-q)*cos(2*theta_eff);
				if (vary_params[4]) e_2 = fitparams[index++];
				else e_2 = (1-q)*sin(2*theta_eff);
				q = 1 - sqrt(SQR(e_1) + SQR(e_2));
				set_angle_from_components(e_1,e_2);
				if ((q <= 0) or (q > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
		} else {
			if (vary_params[3]) {
				if ((fitparams[index] <= 0) or (fitparams[index] > 1)) status = false; // q <= 0 or q > 1 is not a physically acceptable value, so report that we're out of bounds
				q = fitparams[index++];
				if (q < 0) q = -q; // don't allow negative axis ratios
				if (q > 1) q = 1.0; // don't allow q>1
			}
			if (vary_params[4]) set_angle(fitparams[index++]);
		}
		if (!center_anchored) {
			if (vary_params[5]) x_center = fitparams[index++];
			if (vary_params[6]) y_center = fitparams[index++];
		}

		double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
		k = b*pow(sqrt(q)/re,1.0/n);
		set_integration_pointers();
		//defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&SersicLens::deflection_spherical_r);
	}
}

void SersicLens::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = kappa0;
	if (vary_params[1]) fitparams[index++] = re;
	if (vary_params[2]) fitparams[index++] = n;
	if (use_ellipticity_components) {
		if (vary_params[3]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*cos(2*theta_eff);
		}
		if (vary_params[4]) {
			theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
			fitparams[index++] = (1-q)*sin(2*theta_eff);
		}
	} else {
		if (vary_params[3]) fitparams[index++] = q;
		if (vary_params[4]) fitparams[index++] = radians_to_degrees(theta);
	}
	if (!center_anchored) {
		if (vary_params[5]) fitparams[index++] = x_center;
		if (vary_params[6]) fitparams[index++] = y_center;
	}
}

void SersicLens::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.2*kappa0;
	if (vary_params[1]) stepsizes[index++] = 0.2*re;
	if (vary_params[2]) stepsizes[index++] = 0.2;
	if (use_ellipticity_components) {
		if (vary_params[3]) stepsizes[index++] = 0.1;
		if (vary_params[4]) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[3]) stepsizes[index++] = 0.2;
		if (vary_params[4]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[5]) stepsizes[index++] = 0.3; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
		if (vary_params[6]) stepsizes[index++] = 0.3;
	}
}

double SersicLens::kappa_rsq(const double rsq)
{
	return kappa0*exp(-k*pow(rsq,0.5/n));
}

double SersicLens::kappa_rsq_deriv(const double rsq)
{
	return -kappa0*exp(-k*pow(rsq,0.5/n))*(0.5*k/n)*pow(rsq,0.5/n-1);
}

void SersicLens::print_parameters()
{
	if (use_ellipticity_components) {
		double e_1, e_2;
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		e_1 = (1-q)*cos(2*theta_eff);
		e_2 = (1-q)*sin(2*theta_eff);
		cout << "sersic: kappa0=" << kappa0 << ", R_eff=" << re << ", n=" << n << ", e1=" << e_1 << ", e2=" << e_2 << " degrees, center=(" << x_center << "," << y_center << ")";
	} else {
		cout << "sersic: kappa0=" << kappa0 << ", R_eff=" << re << ", n=" << n << ", q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	}
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;

}

/***************************** Mass sheet *****************************/

MassSheet::MassSheet(const double &kext_in, const double &xc_in, const double &yc_in)
{
	lenstype = SHEET;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(3);
	assign_param_pointers();
	kext = kext_in;
	x_center = xc_in; y_center = yc_in;
	assign_paramnames();
}

MassSheet::MassSheet(const MassSheet* lens_in)
{
	lenstype = lens_in->lenstype;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);
	param_number_to_vary.input(n_vary_params);

	kext = lens_in->kext;
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	assign_paramnames();
}

void MassSheet::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "kext"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "ext";
	if (!center_anchored) {
		paramnames[1] = "xc"; latex_paramnames[1] = "x"; latex_param_subscripts[1] = "c";
		paramnames[2] = "yc"; latex_paramnames[2] = "y"; latex_param_subscripts[2] = "c";
	}
}

void MassSheet::assign_param_pointers()
{
	param[0] = &kext;
	if (!center_anchored) {
		param[1] = &x_center;
		param[2] = &y_center;
	}
}

void MassSheet::get_parameters(double* params)
{
	params[0] = kext;
	params[1] = x_center;
	params[2] = y_center;
}

void MassSheet::update_parameters(const double* params)
{
	kext=params[0];
	if (kext < 0) kext=-kext;
	if (!center_anchored) {
		x_center = params[1];
		y_center = params[2];
	}
}

void MassSheet::update_fit_parameters(const double* fitparams, int &index, bool& status)
{
	if (n_vary_params > 0) {
		if (vary_params[0]) kext = fitparams[index++];
		if (!center_anchored) {
			if (vary_params[1]) x_center = fitparams[index++];
			if (vary_params[2]) y_center = fitparams[index++];
		}
	}
}

void MassSheet::get_fit_parameters(dvector& fitparams, int &index)
{
	if (vary_params[0]) fitparams[index++] = kext;
	if (!center_anchored) {
		if (vary_params[1]) fitparams[index++] = x_center;
		if (vary_params[2]) fitparams[index++] = y_center;
	}
}

void MassSheet::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.1*kext;
	if (!center_anchored) {
		if (vary_params[1]) stepsizes[index++] = 0.1; // arbitrary! really, the center should never be independently varied
		if (vary_params[2]) stepsizes[index++] = 0.1;
	}
}

double MassSheet::potential(double x, double y)
{
	x -= x_center;
	y -= y_center;
	return (kext*(x*x+y*y)/2.0);
}

double MassSheet::kappa(double x, double y)
{
	return kext; // really it's a delta function, but effectively zero for our purposes here
}

void MassSheet::deflection(double x, double y, lensvector& def)
{
	x -= x_center;
	y -= y_center;
	def[0] = kext*x;
	def[1] = kext*y;
}

void MassSheet::hessian(double x, double y, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	hess[0][0] = kext;
	hess[1][1] = kext;
	hess[1][0] = 0;
	hess[0][1] = 0;
}

void MassSheet::print_parameters()
{
	cout << "mass sheet: kext=" << kext << ", center=(" << x_center << "," << y_center << ")";
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}



/***************************** Test Model (for testing purposes only) *****************************/

TestModel::TestModel(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype=TESTMODEL;
	center_anchored = false;
	set_default_base_values(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	set_integration_pointers();
}

/*
void TestModel::deflection(double x, double y, lensvector& def)
{
	def[0] = 0;
	def[1] = 0;
}

void TestModel::hessian(double x, double y, lensmatrix& hess)
{
	hess[0][0] = 0;
	hess[1][1] = 0;
	hess[1][0] = 0;
	hess[0][1] = 0;
}
*/

double TestModel::kappa_rsq(const double rsq)
{
	double ans;
	static const double cutoff = 0.173;
	double cutoffsq = cutoff*cutoff;
	if (rsq > cutoffsq) return 0;
	else
		return SQR(3.0/cutoff);
		//return (0.5 * 0.0216 * (1/sqrt(rsq) - pow(0.173*0.173+rsq,-0.5))); // PJaffe
}

void TestModel::print_parameters()
{
	cout << "test: q=" << q << ", theta=" << radians_to_degrees(theta) << " degrees, center=(" << x_center << "," << y_center << ")";
	if (center_anchored) cout << " (center_anchored to lens " << center_anchor_lens->lens_number << ")";
	cout << endl;
}

