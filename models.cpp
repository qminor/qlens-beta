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
const double CoreCusp::nstep = 0.2;

/***************************** Ellipsoidal power law model with core (alpha) *****************************/

Alpha::Alpha(const double &bb_prime, const double &aa, const double &ss_prime, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = ALPHA;
	model_name = "alpha";
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(7);
	ellipticity_mode = default_ellipticity_mode;
	assign_param_pointers();
	assign_paramnames();
	set_default_base_values(nn,acc);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	bprime = bb_prime;
	sprime = ss_prime;
	alpha = aa;
	if (sprime < 0) sprime = -sprime; // don't allow negative core radii

	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

Alpha::Alpha(const Alpha* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	center_anchor_lens = lens_in->center_anchor_lens;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	bprime = lens_in->bprime;
	alpha = lens_in->alpha;
	sprime = lens_in->sprime;
	q = lens_in->q;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	update_meta_parameters();
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
	set_geometric_paramnames(3);
}

void Alpha::assign_param_pointers()
{
	param[0] = &bprime;
	param[1] = &alpha;
	param[2] = &sprime;
	set_geometric_param_pointers(3);
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

void Alpha::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.1*bprime;
	if (vary_params[1]) stepsizes[index++] = 0.1;
	if (vary_params[2]) stepsizes[index++] = 0.02*bprime; // this one is a bit arbitrary, but hopefully reasonable enough
	if (use_ellipticity_components) {
		if (vary_params[3]) stepsizes[index++] = 0.1;
		if (vary_params[4]) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[3]) stepsizes[index++] = 0.1;
		if (vary_params[4]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[5]) stepsizes[index++] = 0.1*bprime;
		if (vary_params[6]) stepsizes[index++] = 0.1*bprime;
	}
}

void Alpha::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 2;    } index++; }
	if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
		
	if (use_ellipticity_components) {
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[4]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[4]) index++;
	}
	if (!center_anchored) {
		if (vary_params[5]) index++;
		if (vary_params[6]) index++;
	}
}

double Alpha::kappa_rsq(const double rsq)
{
	return (0.5 * (2-alpha) * pow(bprime*bprime/(sprime*sprime+rsq), alpha/2));
}

double Alpha::kappa_rsq_deriv(const double rsq)
{
	return (-0.25 * alpha * (2-alpha) * pow(bprime*bprime/(sprime*sprime+rsq), alpha/2 + 1)) / (bprime*bprime);
}

double Alpha::deflection_spherical_r(const double r)
{
	return (pow(bprime,alpha)*(pow(r*r+sprime*sprime,1-alpha/2) - pow(sprime,2-alpha)))/r;
}

double Alpha::deflection_spherical_r_iso(const double r) // only for alpha=1
{
	return bprime*(sqrt(sprime*sprime+r*r)-sprime)/r; // now, tmp = kappa_average
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

	def[0] = real(def_complex);
	def[1] = imag(def_complex);
}

void Alpha::hessian_elliptical_nocore(const double x, const double y, lensmatrix& hess)
{
	double xi, phi, kap;
	xi = sqrt(q*x*x+y*y/q);
	kap = 0.5 * (2-alpha) * pow(bprime/xi, alpha);
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
	hess_complex = 2*b*q/(1+q)*pow(bprime/xi,alpha-1)*polar(1.0,phi)*hyp_2F1(1.0,alpha/2.0,2.0-alpha/2.0,-(1-q)/(1+q)*polar(1.0,2*phi));
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
		re_average = re_major_axis/f_major_axis;
	} else if (alpha==1.0) {
		if (s < b/2.0) {
			re_major_axis = b*sqrt(1-2*s/b/zfactor)*zfactor;
			re_average = re_major_axis/f_major_axis;
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

/********************************** PseudoJaffe **********************************/

PseudoJaffe::PseudoJaffe(const double &bb_prime, const double &aa_prime, const double &ss_prime, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = PJAFFE;
	model_name = "pjaffe";
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(7);
	ellipticity_mode = default_ellipticity_mode;
	assign_param_pointers();
	set_default_base_values(nn,acc);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_paramnames();
	bprime = bb_prime;
	sprime = ss_prime;
	aprime = aa_prime;
	if (sprime < 0) sprime = -sprime;

	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

PseudoJaffe::PseudoJaffe(const PseudoJaffe* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	tidal_host = lens_in->tidal_host;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	bprime = lens_in->bprime;
	sprime = lens_in->sprime;
	aprime = lens_in->aprime;
	if (sprime < 0) sprime = -sprime; // don't allow negative core radii
	q = lens_in->q;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;

	update_meta_parameters();
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
	set_geometric_paramnames(3);
}

void PseudoJaffe::assign_param_pointers()
{
	param[0] = &bprime;
	param[1] = &aprime;
	param[2] = &sprime;
	set_geometric_param_pointers(3);
}

void PseudoJaffe::assign_special_anchored_parameters(LensProfile *host_in)
{
	anchor_special_parameter = true;
	tidal_host = host_in;
	double rm, ravg;
	tidal_host->get_einstein_radius(rm,ravg,1.0);
	aprime = sqrt(ravg*bprime); // this is an approximate formula (a' = sqrt(b'*Re_halo)) and assumes the subhalo is found roughly near the Einstein radius of the halo
	a = aprime/f_major_axis;
	asq = a*a;
}

void PseudoJaffe::update_special_anchored_params()
{
	if (anchor_special_parameter) {
		double rm, ravg;
		tidal_host->get_einstein_radius(rm,ravg,1.0);
		aprime = sqrt(ravg*bprime); // this is an approximate formula (a' = sqrt(b'*Re_halo)) and assumes the subhalo is found roughly near the Einstein radius of the halo
		a = aprime/f_major_axis;
		asq = a*a;
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

void PseudoJaffe::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.5*bprime;
	if (vary_params[1]) stepsizes[index++] = 0.2*bprime;
	if (vary_params[2]) stepsizes[index++] = 0.02*bprime; // this one is a bit arbitrary, but hopefully reasonable enough
	if (use_ellipticity_components) {
		if (vary_params[3]) stepsizes[index++] = 0.1;
		if (vary_params[4]) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[3]) stepsizes[index++] = 0.2;
		if (vary_params[4]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[5]) stepsizes[index++] = 0.5*bprime;
		if (vary_params[6]) stepsizes[index++] = 0.5*bprime;
	}
}

void PseudoJaffe::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }

	if (use_ellipticity_components) {
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[4]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[4]) index++;
	}
	if (!center_anchored) {
		if (vary_params[5]) index++;
		if (vary_params[6]) index++;
	}
}

double PseudoJaffe::kappa_rsq(const double rsq)
{
	return (0.5 * bprime * (pow(sprime*sprime+rsq, -0.5) - pow(aprime*aprime+rsq,-0.5)));
}

double PseudoJaffe::kappa_rsq_deriv(const double rsq)
{
	return (-0.25 * bprime * (pow(sprime*sprime+rsq, -1.5) - pow(aprime*aprime+rsq,-1.5)));
}

double PseudoJaffe::deflection_spherical_r(const double r)
{
	double rsq = r*r;
	return bprime*((sqrt(sprime*sprime+rsq)-sprime) - (sqrt(aprime*aprime+rsq)-aprime))/r;
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
	tmp1 = (b*q/psi)/(xsq+ysq+2*psi*s+ssq*(1+qsq));

	psi2 = sqrt(qsq*(asq+xsq)+ysq);
	tmp2 = (b*q/psi2)/(xsq+ysq+2*psi2*a+asq*(1+qsq));

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

bool PseudoJaffe::output_cosmology_info(const double zlens, const double zsrc, Cosmology* cosmo, const int lens_number)
{
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr = cosmo->sigma_crit_kpc(zlens,zsrc);
	double kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(zlens);
	double kpc_to_km = 3.086e16;
	double Rs_sun_km = 2.953; // Schwarzchild radius of the Sun in km
	double c = 2.998e5;
	double b_kpc, sigma, r_tidal, r_core;
	b_kpc = bprime/kpc_to_arcsec;
	sigma = c * sqrt(b_kpc*(1-sprime/aprime)*(Rs_sun_km/kpc_to_km)*sigma_cr/2);
	cout << "sigma = " << sigma << " km/s  (velocity dispersion)\n";
	r_tidal = aprime/kpc_to_arcsec;
	r_core = sprime/kpc_to_arcsec;
	cout << "r_tidal = " << r_tidal << " kpc" << endl;
	cout << "r_core = " << r_core << " kpc" << endl;
	cout << endl;
	//double tsig = 297.2, tb;
	//tb = kpc_to_arcsec*2*SQR(tsig/c)/((Rs_sun_km/kpc_to_km)*sigma_cr*(1-sprime/aprime));
	//cout << "Test b = " << tb << endl;
}

/********************************** NFW **********************************/

NFW::NFW(const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = nfw;
	model_name = "nfw";
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(6);
	ellipticity_mode = default_ellipticity_mode;
	assign_param_pointers();
	ks = ks_in; rs = rs_in;
	set_default_base_values(nn,acc);
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under deflection_spherical(...) function)
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_paramnames();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

NFW::NFW(const NFW* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	ks = lens_in->ks;
	rs = lens_in->rs;
	if (rs < 0) rs = -rs; // don't allow negative scale radii
	q = lens_in->q;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under deflection_spherical(...) function)
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

void NFW::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	set_geometric_paramnames(2);
}

void NFW::assign_param_pointers()
{
	param[0] = &ks;
	param[1] = &rs;
	set_geometric_param_pointers(2);
}

void NFW::set_model_specific_integration_pointers()
{
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::deflection_spherical_r);
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
		if (vary_params[4]) stepsizes[index++] = 0.5; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
		if (vary_params[5]) stepsizes[index++] = 0.5;
	}
}

void NFW::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (use_ellipticity_components) {
		if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[3]) index++;
	}
	if (!center_anchored) {
		if (vary_params[4]) index++;
		if (vary_params[5]) index++;
	}
}

double NFW::kappa_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (xsq==1) return (2*ks/3.0); // note, ks is defined as ks = rho_s * r_s / sigma_crit
	else if (xsq < 1e-6) return -ks*(2+log(xsq/4));
	else return 2*ks*(1 - lens_function_xsq(xsq))/(xsq - 1);
}

double NFW::kappa_rsq_deriv(const double rsq)
{
	double xsq = rsq/(rs*rs);
	// below xsq ~ 1e-6 or so, kappa formula becomes inaccurate due to fine cancellations; a series expansion is done for xsq smaller than this
	if (xsq < 1e-6) return -ks/rsq;
	else if (abs(xsq-1.0) < 1e-5) return -0.4*sqrt(xsq); // kappa formula on next line becomes unstable for x very close to 1, this fixes the instability
	else return -(ks/rsq)*((xsq*(2.0-3*lens_function_xsq(xsq)) + 1)/((xsq-1)*(xsq-1)));
}

inline double NFW::lens_function_xsq(const double &xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}

double NFW::deflection_spherical_r(const double r)
{
	double xsq = SQR(r/rs);
	// below xsq ~ 1e-6 or so, this becomes inaccurate due to fine cancellations; a series expansion is done for xsq smaller than this
	if (xsq > 1e-6)
		return 2*ks*r*(2*lens_function_xsq(xsq) + log(xsq/4))/xsq;
	else
		return -ks*r*(1+log(xsq/4));
}

bool NFW::output_cosmology_info(const double zlens, const double zsrc, Cosmology* cosmo, const int lens_number)
{
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr = cosmo->sigma_crit_kpc(zlens,zsrc);
	double kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(zlens);
	double rs_kpc, ds, c, m200, r200;
	rs_kpc = rs / kpc_to_arcsec;
	ds = ks * sigma_cr / rs_kpc;
	cosmo->get_halo_parameters_from_rs_ds(zlens,rs_kpc,ds,m200,r200);
	c = r200/rs_kpc;
	cout << "rho_s = " << ds << " M_sol/kpc^3  (density at scale radius)" << endl;
	cout << "r_s = " << rs_kpc << " kpc  (scale radius)" << endl;
	cout << "c = " << c << endl;
	cout << "M_200 = " << m200 << " M_sol\n";
	cout << "r_200 = " << r200 << " kpc\n";
	cout << endl;
	return true;
}

/********************************** NFW with elliptic potential *************************************/

Pseudo_Elliptical_NFW::Pseudo_Elliptical_NFW(const double &ks_in, const double &rs_in, const double &e_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = pnfw;
	model_name = "pnfw";
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(6);
	ellipticity_mode = 2;
	assign_param_pointers();
	ks = ks_in; rs = rs_in;
	set_default_base_values(nn,acc);
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under deflection_spherical(...) function)
	epsilon = e_in;
	set_angle(theta_degrees);
	x_center = xc_in;
	y_center = yc_in;
	assign_paramnames();
	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

Pseudo_Elliptical_NFW::Pseudo_Elliptical_NFW(const Pseudo_Elliptical_NFW* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	ks = lens_in->ks;
	rs = lens_in->rs;
	if (rs < 0) rs = -rs; // don't allow negative scale radii
	epsilon = lens_in->epsilon;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under deflection_spherical(...) function)
	update_meta_parameters();
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

void Pseudo_Elliptical_NFW::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	set_geometric_paramnames(2);
}

void Pseudo_Elliptical_NFW::assign_param_pointers()
{
	param[0] = &ks;
	param[1] = &rs;
	param[2] = &epsilon;
	param[3] = &theta; angle_paramnum = 3;
	if (!center_anchored) {
		param[4] = &x_center;
		param[5] = &y_center;
	}
}

void Pseudo_Elliptical_NFW::set_model_specific_integration_pointers()
{
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Pseudo_Elliptical_NFW::deflection_spherical_r);
	defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&Pseudo_Elliptical_NFW::deflection_elliptical);
	hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&Pseudo_Elliptical_NFW::hessian_elliptical);
	potptr = static_cast<double (LensProfile::*)(const double,const double)> (&Pseudo_Elliptical_NFW::potential_elliptical);
}


void Pseudo_Elliptical_NFW::get_auto_stepsizes(dvector& stepsizes, int &index)
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
		if (vary_params[4]) stepsizes[index++] = 0.5; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
		if (vary_params[5]) stepsizes[index++] = 0.5;
	}
}

void Pseudo_Elliptical_NFW::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (use_ellipticity_components) {
		if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[3]) index++;
	}
	if (!center_anchored) {
		if (vary_params[4]) index++;
		if (vary_params[5]) index++;
	}
}

double Pseudo_Elliptical_NFW::kappa_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (xsq==1) return (2*ks/3.0); // note, ks is defined as ks = rho_s * r_s / sigma_crit
	// below xsq ~ 1e-6 or so, this becomes inaccurate due to fine cancellations; a series expansion is done for xsq smaller than this
	else if (xsq < 1e-6)
		return -ks*(2+log(xsq/4));
	else
		return (2*ks*(1 - lens_function_xsq(xsq))/(xsq - 1));
}

double Pseudo_Elliptical_NFW::kappa_rsq_deriv(const double rsq)
{
	double xsq = rsq/(rs*rs);
	// below xsq ~ 1e-6 or so, kappa formula becomes inaccurate due to fine cancellations; a series expansion is done for xsq smaller than this
	if (xsq < 1e-6) return -ks/rsq;
	else if (abs(xsq-1.0) < 1e-5) return -0.4*sqrt(xsq); // kappa formula on next line becomes unstable for x very close to 1, this fixes the instability
	else return -(ks/rsq)*((xsq*(2.0-3*lens_function_xsq(xsq)) + 1)/((xsq-1)*(xsq-1)));
}

inline double Pseudo_Elliptical_NFW::lens_function_xsq(const double &xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}

double Pseudo_Elliptical_NFW::deflection_spherical_r(const double r)
{
	double xsq = SQR(r/rs);
	// below xsq ~ 1e-6 or so, this becomes inaccurate due to fine cancellations; a series expansion is done for xsq smaller than this
	if (xsq > 1e-6)
		return 2*ks*r*(2*lens_function_xsq(xsq) + log(xsq/4))/xsq;
	else
		return -ks*r*(1+log(xsq/4));
}

double Pseudo_Elliptical_NFW::shear_magnitude(const double rsq)
{
	// This is the shear of the *spherical* model
	double xsq, kapavg, kappa;
	xsq = rsq/(rs*rs);
	// warning: below xsq ~ 10^-6 or so, this becomes inaccurate due to fine cancellations; a series expansion should be done for xsq smaller than this (do later)

	if (xsq==1) {
		kappa = (2*ks/3.0); // note, ks is defined as ks = rho_s * r_s / sigma_crit
		kapavg = 2*ks*(2*lens_function_xsq(xsq) + log(xsq/4))/xsq;
	}
	else if (xsq > 1e-6) {
		kapavg = 2*ks*(2*lens_function_xsq(xsq) + log(xsq/4))/xsq;
		kappa = 2*ks*(1 - lens_function_xsq(xsq))/(xsq - 1);
	} else {
		kapavg = -ks*(1+log(xsq/4));
		kappa = -ks*(2+log(xsq/4));
	}
	return (kapavg - kappa);
}

double Pseudo_Elliptical_NFW::kappa(double x, double y)
{
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	double phi = atan(abs(y/(q*x)));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	double rsq = (1-epsilon)*x*x + (1+epsilon)*y*y;
	return (kappa_rsq(rsq) + epsilon*shear_magnitude(rsq)*cos(2*phi));
}

double Pseudo_Elliptical_NFW::potential_elliptical(const double x, const double y)
{
	double xsq = ((1-epsilon)*x*x + (1+epsilon)*y*y)/(rs*rs); // just r_ell for the moment
	if (xsq < 1)
		return 2*ks*rs*rs*(-SQR(atanh(sqrt(1-xsq))) + SQR(log(xsq/4)/2));
	else 
		return 2*ks*rs*rs*(-SQR(atan(sqrt(xsq-1))) + SQR(log(xsq/4)/2));
}

void Pseudo_Elliptical_NFW::deflection_elliptical(const double x, const double y, lensvector& def)
{
	double defmag_over_r_ell = sqrt((1-epsilon)*x*x + (1+epsilon)*y*y); // just r_ell for the moment
	defmag_over_r_ell = deflection_spherical_r(defmag_over_r_ell)/defmag_over_r_ell;
	def[0] = defmag_over_r_ell*(1-epsilon)*x;
	def[1] = defmag_over_r_ell*(1+epsilon)*y;
}

void Pseudo_Elliptical_NFW::hessian_elliptical(const double x, const double y, lensmatrix& hess)
{
	double temp, gamma1, gamma2, kap_r, shearmag, kap, phi;
	if (x==0) {
		if (y > 0) phi = M_PI/2;
		else phi = -M_PI/2;
	} else {
		phi = atan(abs(y/(q*x)));
		if (x < 0) {
			if (y < 0)
				phi = phi - M_PI;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = -phi;
		}
	}
	temp = (1-epsilon)*x*x + (1+epsilon)*y*y; // elliptical r^2
	kap_r = kappa_rsq(temp);
	shearmag = shear_magnitude(temp);
	temp = cos(2*phi);
	kap = kap_r + epsilon*shearmag*temp;
	gamma1 = -epsilon*kap_r - shearmag*temp;
	gamma2 = -sqrt(1-epsilon*epsilon)*shearmag*sin(2*phi);
	hess[0][0] = kap + gamma1;
	hess[1][1] = kap - gamma1;
	hess[0][1] = gamma2;
	hess[1][0] = gamma2;
}

void Pseudo_Elliptical_NFW::get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor)
{
	zfac = zfactor;
	if ((einstein_radius_root(rmin_einstein_radius)*einstein_radius_root(rmax_einstein_radius)) > 0) {
		// multiple imaging does not occur with this lens
		re_major_axis = 0;
		re_average = 0;
		return;
	}
	double (Brent::*bptr)(const double);
	bptr = static_cast<double (Brent::*)(const double)> (&Pseudo_Elliptical_NFW::einstein_radius_root);
	// for this lens, the elliptical radius is such that the scale rs gives the average, rather than major axis value (thus, same true for Einstein radius)
	re_average = BrentsMethod(bptr,rmin_einstein_radius,rmax_einstein_radius,1e-3);
	re_major_axis = re_average / sqrt(1-epsilon);
	zfac = 1.0;
}

bool Pseudo_Elliptical_NFW::output_cosmology_info(const double zlens, const double zsrc, Cosmology* cosmo, const int lens_number)
{
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr = cosmo->sigma_crit_kpc(zlens,zsrc);
	double kpc_to_arcsec = 206.264806/cosmo->angular_diameter_distance(zlens);
	double rs_kpc, ds, c, m200, r200;
	rs_kpc = rs / kpc_to_arcsec;
	ds = ks * sigma_cr / rs_kpc;
	cosmo->get_halo_parameters_from_rs_ds(zlens,rs_kpc,ds,m200,r200);
	c = r200/rs_kpc;
	cout << "rho_s = " << ds << " M_sol/kpc^3  (density at scale radius)" << endl;
	cout << "r_s = " << rs_kpc << " kpc  (scale radius)" << endl;
	cout << "c = " << c << endl;
	cout << "M_200 = " << m200 << " M_sol\n";
	cout << "r_200 = " << r200 << " kpc\n";
	cout << endl;
	return true;
}

/********************************** Truncated_NFW **********************************/

Truncated_NFW::Truncated_NFW(const double &ks_in, const double &rs_in, const double &rt_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = TRUNCATED_nfw;
	model_name = "tnfw";
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(7);
	ellipticity_mode = default_ellipticity_mode;
	assign_param_pointers();
	ks = ks_in; rs = rs_in; rt = rt_in;
	set_default_base_values(nn,acc);
	assign_paramnames();
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under deflection_spherical(...) function)
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

Truncated_NFW::Truncated_NFW(const Truncated_NFW* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;

	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	ks = lens_in->ks;
	rs = lens_in->rs;
	rt = lens_in->rt;
	if (rs < 0) rs = -rs; // don't allow negative scale radii
	q = lens_in->q;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);
	rmin_einstein_radius = 1e-3*rs; // at the moment, kappa_average is not reliable below this value (see note under NFW deflection_spherical(...) function)
	set_integration_pointers();
	set_model_specific_integration_pointers();
}

void Truncated_NFW::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	paramnames[2] = "rt"; latex_paramnames[2] = "r"; latex_param_subscripts[2] = "t";
	set_geometric_paramnames(3);
}

void Truncated_NFW::assign_param_pointers()
{
	param[0] = &ks;
	param[1] = &rs;
	param[2] = &rt;
	set_geometric_param_pointers(3);
}

void Truncated_NFW::set_model_specific_integration_pointers()
{
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&Truncated_NFW::deflection_spherical_r);
}

void Truncated_NFW::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.2*ks;
	if (vary_params[1]) stepsizes[index++] = 0.2*rs;
	if (vary_params[2]) stepsizes[index++] = 0.2*rt;
	if (use_ellipticity_components) {
		if (vary_params[3]) stepsizes[index++] = 0.1;
		if (vary_params[4]) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[3]) stepsizes[index++] = 0.1;
		if (vary_params[4]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[5]) stepsizes[index++] = 0.5; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
		if (vary_params[6]) stepsizes[index++] = 0.5;
	}
}

void Truncated_NFW::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (use_ellipticity_components) {
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[4]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[4]) index++;
	}
	if (!center_anchored) {
		if (vary_params[5]) index++;
		if (vary_params[6]) index++;
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

/********************************** Hernquist **********************************/

Hernquist::Hernquist(const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = HERNQUIST;
	model_name = "hern";
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(6);
	ellipticity_mode = default_ellipticity_mode;
	assign_param_pointers();
	ks = ks_in; rs = rs_in;
	set_default_base_values(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_paramnames();
	set_integration_pointers();
	// NOTE: for q=1, the deflection has an analytic formula. Implement this later!
}

Hernquist::Hernquist(const Hernquist* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	ks = lens_in->ks;
	rs = lens_in->rs;
	if (rs < 0) rs = -rs; // don't allow negative core radii
	q = lens_in->q;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
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
	set_geometric_paramnames(2);
}

void Hernquist::assign_param_pointers()
{
	param[0] = &ks;
	param[1] = &rs;
	set_geometric_param_pointers(2);
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

void Hernquist::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (use_ellipticity_components) {
		if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[3]) index++;
	}
	if (!center_anchored) {
		if (vary_params[4]) index++;
		if (vary_params[5]) index++;
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

/********************************** Exponential Disk **********************************/

ExpDisk::ExpDisk(const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = EXPDISK;
	model_name = "expdisk";
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(6);
	ellipticity_mode = default_ellipticity_mode;
	assign_param_pointers();
	k0 = k0_in; R_d = R_d_in;
	set_default_base_values(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	assign_paramnames();
	set_integration_pointers();
}

ExpDisk::ExpDisk(const ExpDisk* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	k0 = lens_in->k0;
	R_d = lens_in->R_d;
	if (R_d < 0) R_d = -R_d; // don't allow negative core radii
	q = lens_in->q;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
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
	set_geometric_paramnames(2);
}

void ExpDisk::assign_param_pointers()
{
	param[0] = &k0;
	param[1] = &R_d;
	set_geometric_param_pointers(2);
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

void ExpDisk::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (use_ellipticity_components) {
		if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[2]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[3]) index++;
	}
	if (!center_anchored) {
		if (vary_params[4]) index++;
		if (vary_params[5]) index++;
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

/***************************** External shear *****************************/

Shear::Shear(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in)
{
	lenstype = SHEAR;
	model_name = "shear";
	defined_spherical_kappa_profile = false;
	center_anchored = false;
	anchor_special_parameter = false;
	set_n_params(4);
	assign_param_pointers();
	assign_paramnames();

	if (use_shear_component_params) {
		shear1 = shear_p1_in;
		shear2 = shear_p2_in;
		q = sqrt(SQR(shear_p1_in) + SQR(shear_p2_in)); // shear magnitude
		set_angle_from_components(shear_p1_in,shear_p2_in);
	} else {
		q=shear_p1_in;
		set_angle(shear_p2_in);
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		shear1 = -q*cos(2*theta_eff);
		shear2 = -q*sin(2*theta_eff);
	}
	x_center = xc_in;
	y_center = yc_in;
}

Shear::Shear(const Shear* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	q = lens_in->q;
	set_angle_radians(lens_in->theta);
	shear1 = lens_in->shear1;
	shear2 = lens_in->shear2;
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
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
	if (use_shear_component_params) {
		param[0] = &shear1;
		param[1] = &shear2;
		angle_paramnum = -1; // since there is no angle parameter in this mode
	} else {
		param[0] = &q; // here, q is actually the shear magnitude
		param[1] = &theta; angle_paramnum = 1;
	}
	if (!center_anchored) {
		param[2] = &x_center;
		param[3] = &y_center;
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

void Shear::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (use_shear_component_params) {
		if (vary_params[0]) index++;
		if (vary_params[1]) index++;
	} else {
		if (vary_params[0]) {
			if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; }
			index++;
		}
		if (vary_params[1]) index++;
	}
	if (!center_anchored) {
		if (vary_params[2]) index++;
		if (vary_params[3]) index++;
	}
}

double Shear::potential(double x, double y)
{
	x -= x_center;
	y -= y_center;
	return -0.5*(y*y-x*x)*shear1 + x*y*shear2;
}

void Shear::deflection(double x, double y, lensvector& def)
{
	x -= x_center;
	y -= y_center;
	def[0] = x*shear1 + y*shear2;
	def[1] = -y*shear1 + x*shear2;
}

void Shear::hessian(double x, double y, lensmatrix& hess)
{
	// Hessian does not depend on x or y
	hess[0][0] = shear1;
	hess[1][1] = -hess[0][0];
	hess[0][1] = shear2;
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

/***************************** Multipole term *******************************/

Multipole::Multipole(const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, const bool sine)
{
	lenstype = MULTIPOLE;
	model_name = (kap==true) ? "kmpole" : "mpole";
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
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	n = lens_in->n;
	m = lens_in->m;
	q=lens_in->q;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	kappa_multipole = lens_in->kappa_multipole;
	sine_term = lens_in->sine_term;
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;

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
	param[0] = &q; // here, q is actually the shear magnitude
	param[1] = &n;
	param[2] = &theta; angle_paramnum = 2;
	if (!center_anchored) {
		param[3] = &x_center;
		param[4] = &y_center;
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

void Multipole::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) index++;
	if (vary_params[1]) index++;
	if (vary_params[2]) index++;
	if (!center_anchored) {
		if (vary_params[3]) index++;
		if (vary_params[4]) index++;
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

/***************************** Point mass *****************************/

PointMass::PointMass(const double &bb, const double &xc_in, const double &yc_in)
{
	lenstype = PTMASS;
	model_name = "ptmass";
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
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	b = lens_in->b;
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
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
	angle_paramnum = -1; // since there is no angle parameter
}

void PointMass::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.1*b;
	if (!center_anchored) {
		if (vary_params[1]) stepsizes[index++] = 0.1*b;
		if (vary_params[2]) stepsizes[index++] = 0.1*b;
	}
}

void PointMass::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) {
		if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; }
		index++;
	}
	if (!center_anchored) {
		if (vary_params[1]) index++;
		if (vary_params[2]) index++;
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

/***************************** Core/Cusp Model *****************************/

CoreCusp::CoreCusp(const double &mass_param_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, bool parametrize_einstein_radius)
{
	lenstype = CORECUSP;
	model_name = "corecusp";
	set_n_params(9);
	ellipticity_mode = default_ellipticity_mode;
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
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	tidal_host = lens_in->tidal_host;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	set_k0_by_einstein_radius = lens_in->set_k0_by_einstein_radius;
	if (set_k0_by_einstein_radius) einstein_radius = lens_in->einstein_radius;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	k0 = lens_in->k0;
	gamma = lens_in->gamma;
	n = lens_in->n;
	a = lens_in->a;
	s = lens_in->s;
	if (s < 0) s = -s; // don't allow negative core radii
	q = lens_in->q;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
	set_default_base_values(lens_in->numberOfPoints,lens_in->romberg_accuracy);

	set_integration_pointers();
	defptr_r_spherical = static_cast<double (LensProfile::*)(const double)> (&CoreCusp::deflection_spherical_r);
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
}

void CoreCusp::assign_paramnames()
{
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	if (set_k0_by_einstein_radius) { paramnames[0] = "Re"; latex_paramnames[0] = "R"; latex_param_subscripts[0] = "\\epsilon"; }
	else { paramnames[0] = "k0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0"; }
	paramnames[1] = "gamma"; latex_paramnames[1] = "\\gamma"; latex_param_subscripts[1] = "";
	paramnames[2] = "n"; latex_paramnames[2] = "n"; latex_param_subscripts[2] = "";
	paramnames[3] = "a"; latex_paramnames[3] = "a"; latex_param_subscripts[3] = "";
	paramnames[4] = "s"; latex_paramnames[4] = "s"; latex_param_subscripts[4] = "";
	set_geometric_paramnames(5);
}

void CoreCusp::assign_param_pointers()
{
	if (set_k0_by_einstein_radius) param[0] = &einstein_radius;
	else param[0] = &k0;
	param[1] = &gamma;
	param[2] = &n;
	param[3] = &a;
	param[4] = &s;
	set_geometric_param_pointers(5);
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
		else a = ravg*k0/(3-gamma); // we have ignored the core in this formula, but should be reasonable as long as a >> s
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
		if (vary_params[5]) stepsizes[index++] = 0.1;
		if (vary_params[6]) stepsizes[index++] = 0.1;
	} else {
		if (vary_params[5]) stepsizes[index++] = 0.1;
		if (vary_params[6]) stepsizes[index++] = 20;
	}
	if (!center_anchored) {
		if (vary_params[7]) stepsizes[index++] = 0.1*k0;
		if (vary_params[8]) stepsizes[index++] = 0.1*k0;
	}
}

void CoreCusp::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) index++;
	if (vary_params[2]) index++;
	if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[4]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (use_ellipticity_components) {
		if (vary_params[5]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[6]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[5]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[6]) index++;
	}
	if (!center_anchored) {
		if (vary_params[7]) index++;
		if (vary_params[8]) index++;
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
	// This uses Richardson extrapolation to calculate the enclosed mass, required for the n=3 case
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

/***************************** SersicLens profile *****************************/


SersicLens::SersicLens(const double &kappa0_in, const double &Re_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = SERSIC_LENS;
	model_name = "sersic";
	center_anchored = false;
	anchor_special_parameter = false;
	ellipticity_mode = default_ellipticity_mode;
	set_n_params(7);
	assign_param_pointers();
	assign_paramnames();
	set_default_base_values(nn,acc);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	n = n_in;
	re = Re_in;
	double b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	k = b*pow(1.0/re,1.0/n);
	kappa0 = kappa0_in;
	set_default_base_values(nn,acc);
	set_integration_pointers();
}

SersicLens::SersicLens(const SersicLens* lens_in)
{
	lenstype = lens_in->lenstype;
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	center_anchor_lens = lens_in->center_anchor_lens;
	n_params = lens_in->n_params;
	ellipticity_mode = lens_in->ellipticity_mode;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	kappa0 = lens_in->kappa0;
	n = lens_in->n;
	re = lens_in->re;
	k = lens_in->k;
	q = lens_in->q;
	f_major_axis = lens_in->f_major_axis;
	set_angle_radians(lens_in->theta);
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
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
	set_geometric_paramnames(3);
}

void SersicLens::assign_param_pointers()
{
	param[0] = &kappa0;
	param[1] = &re;
	param[2] = &n;
	set_geometric_param_pointers(3);
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

void SersicLens::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[1]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1e30; } index++; }
	if (vary_params[2]) index++;
	if (use_ellipticity_components) {
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
		if (vary_params[4]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = -1; upper[index] = 1; } index++; }
	} else {
		if (vary_params[3]) { if (use_penalty_limits[index]==false) { use_penalty_limits[index] = true; lower[index] = 0; upper[index] = 1; } index++; }
		if (vary_params[4]) index++;
	}
	if (!center_anchored) {
		if (vary_params[5]) index++;
		if (vary_params[6]) index++;
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

/***************************** Mass sheet *****************************/

MassSheet::MassSheet(const double &kext_in, const double &xc_in, const double &yc_in)
{
	lenstype = SHEET;
	model_name = "sheet";
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
	model_name = lens_in->model_name;
	lens_number = lens_in->lens_number;
	center_anchored = lens_in->center_anchored;
	center_anchor_lens = lens_in->center_anchor_lens;
	anchor_special_parameter = lens_in->anchor_special_parameter;
	n_params = lens_in->n_params;
	copy_parameter_anchors(lens_in);
	assign_param_pointers();
	n_vary_params = lens_in->n_vary_params;
	vary_params.input(lens_in->vary_params);

	kext = lens_in->kext;
	x_center = lens_in->x_center;
	y_center = lens_in->y_center;
	paramnames = lens_in->paramnames;
	latex_paramnames = lens_in->latex_paramnames;
	latex_param_subscripts = lens_in->latex_param_subscripts;
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
	angle_paramnum = -1; // since there is no angle parameter
}

void MassSheet::get_auto_stepsizes(dvector& stepsizes, int &index)
{
	if (vary_params[0]) stepsizes[index++] = 0.1*kext;
	if (!center_anchored) {
		if (vary_params[1]) stepsizes[index++] = 0.1; // arbitrary! really, the center should never be independently varied
		if (vary_params[2]) stepsizes[index++] = 0.1;
	}
}

void MassSheet::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	if (vary_params[0]) index++;
	if (!center_anchored) {
		if (vary_params[1]) index++;
		if (vary_params[2]) index++;
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

/***************************** Test Model (for testing purposes only) *****************************/

TestModel::TestModel(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = TESTMODEL;
	model_name = "test";
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

