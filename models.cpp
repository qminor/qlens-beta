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
const double CoreCusp::digamma_three_halves = 0.036489973978435;
const double Alpha::euler_mascheroni = 0.57721566490153286060;
const double Alpha::def_tolerance = 1e-16;

/*************************** Softened power law model (alpha) *****************************/

Alpha::Alpha(const double &bb, const double &aa, const double &ss, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = ALPHA;
	model_name = "alpha";
	special_parameter_command = "";
	setup_base_lens(7,true); // number of parameters = 7, is_elliptical_lens = true

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	b = bb;
	s = ss;
	alpha = aa;
	if (s < 0) s = -s; // don't allow negative core radii

	update_meta_parameters_and_pointers();
}

Alpha::Alpha(const Alpha* lens_in)
{
	b = lens_in->b;
	alpha = lens_in->alpha;
	s = lens_in->s;

	copy_base_lensdata(lens_in);
	update_meta_parameters_and_pointers();
}

void Alpha::assign_paramnames()
{
	paramnames[0] = "b";     latex_paramnames[0] = "b";       latex_param_subscripts[0] = "";
	paramnames[1] = "alpha"; latex_paramnames[1] = "\\alpha"; latex_param_subscripts[1] = "";
	paramnames[2] = "s";     latex_paramnames[2] = "s";       latex_param_subscripts[2] = "";
	set_geometric_paramnames(3);
}

void Alpha::assign_param_pointers()
{
	param[0] = &b;
	param[1] = &alpha;
	param[2] = &s;
	set_geometric_param_pointers(3);
}

void Alpha::update_meta_parameters()
{
	update_ellipticity_meta_parameters();
	// these meta-parameters are used in analytic formulas for deflection, potential, etc.
	bprime = b*f_major_axis;
	sprime = s*f_major_axis;
	qsq = q*q; ssq = sprime*sprime;
}

void Alpha::set_auto_stepsizes()
{
	stepsizes[0] = 0.1*b;
	stepsizes[1] = 0.1;
	stepsizes[2] = 0.02*b; // this one is a bit arbitrary, but hopefully reasonable enough
	set_auto_eparam_stepsizes(3,4);
	stepsizes[5] = 0.1*b;
	stepsizes[6] = 0.1*b;
}

void Alpha::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 2;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(3);
}

void Alpha::set_model_specific_integration_pointers()
{
	// Here, we direct the integration pointers to analytic formulas in special cases where analytic solutions are possible
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Alpha::kapavg_spherical_rsq);
	potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Alpha::potential_spherical_rsq);
	if (alpha==1.0) {
		kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Alpha::kapavg_spherical_rsq_iso);
		potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Alpha::potential_spherical_rsq_iso);
		if (q != 1.0) {
			defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&Alpha::deflection_elliptical_iso);
			hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&Alpha::hessian_elliptical_iso);
			potptr = static_cast<double (LensProfile::*)(const double,const double)> (&Alpha::potential_elliptical_iso);
		}
	} else if (s==0.0) {
		potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Alpha::potential_spherical_rsq_nocore);
		if (q != 1.0) {
			defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&Alpha::deflection_elliptical_nocore);
			hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&Alpha::hessian_elliptical_nocore);
			potptr = static_cast<double (LensProfile::*)(const double,const double)> (&Alpha::potential_elliptical_nocore);
			def_and_hess_ptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&,lensmatrix&)> (&Alpha::deflection_and_hessian_elliptical_nocore);
		}
	}
}

double Alpha::kappa_rsq(const double rsq)
{
	return (0.5 * (2-alpha) * pow(b*b/(s*s+rsq), alpha/2));
}

double Alpha::kappa_rsq_deriv(const double rsq)
{
	return (-0.25 * alpha * (2-alpha) * pow(b*b/(s*s+rsq), alpha/2 + 1)) / (b*b);
}

double Alpha::kapavg_spherical_rsq(const double rsq)
{
	return (pow(b,alpha)*(pow(rsq+s*s,1-alpha/2) - pow(s,2-alpha)))/rsq;
}

double Alpha::kapavg_spherical_rsq_iso(const double rsq) // only for alpha=1
{
	return b*(sqrt(s*s+rsq)-s)/rsq; // now, tmp = kappa_average
}

double Alpha::potential_spherical_rsq(const double rsq)
{
	// Formula from Keeton (2002), w/ typo corrected (sign in front of the DiGamma() term)
	double bpow, bs, p, tmp;
	bpow = pow(b,alpha);
	bs = bpow*pow(s,2-alpha);
	p = alpha/2-1;
	tmp = bpow*pow(rsq,-p)*real(hyp_2F1(p,p,1+p,-s*s/rsq))/(2-alpha);
	tmp += -bs*log(rsq/(s*s))/2 - bs*(euler_mascheroni + DiGamma(p))/2;
	return tmp;
}

double Alpha::potential_spherical_rsq_iso(const double rsq) // only for alpha=1
{
	double tmp;
	tmp = b*(sqrt(s*s+rsq)-s); // now, tmp = kappa_average*rsq
	if (s != 0) tmp -= b*s*log((s + sqrt(s*s+rsq))/(2.0*s));
	return tmp;
}

double Alpha::potential_spherical_rsq_nocore(const double rsq) // only for sprime=0
{
	return pow(b*b/rsq,alpha/2)*rsq/(2-alpha);
}

//  Note: although the elliptical formulas are expressed in terms of ellipticity mode 0, they use parameters
//  (the prime versions b', a', etc.) transformed from the correct emode

void Alpha::deflection_elliptical_iso(const double x, const double y, lensvector& def) // only for alpha=1
{
	double u, psi;
	psi = sqrt(qsq*(ssq+x*x)+y*y);
	u = sqrt(1-qsq);

	def[0] = (bprime*q/u)*atan(u*x/(psi+sprime));
	def[1] = (bprime*q/u)*atanh(u*y/(psi+qsq*sprime));
}

void Alpha::hessian_elliptical_iso(const double x, const double y, lensmatrix& hess) // only for alpha=1
{
	double xsq, ysq, psi, tmp;
	xsq=x*x; ysq=y*y;

	psi = sqrt(qsq*(ssq+xsq)+ysq);
	tmp = ((bprime*q)/psi)/(xsq+ysq+2*psi*sprime+ssq*(1+qsq));

	hess[0][0] = tmp*(ysq+sprime*psi+ssq*qsq);
	hess[1][1] = tmp*(xsq+sprime*psi+ssq);
	hess[0][1] = -tmp*x*y;
	hess[1][0] = hess[0][1];
}

double Alpha::potential_elliptical_iso(const double x, const double y) // only for alpha=1
{
	double u, tmp, psi;
	psi = sqrt(qsq*(ssq+x*x)+y*y);
	u = sqrt(1-qsq);

	tmp = (bprime*q/u)*(x*atan(u*x/(psi+sprime)) + y*atanh(u*y/(psi+qsq*sprime)));
	if (sprime != 0) tmp += bprime*q*sprime*(-log(SQR(psi+sprime) + SQR(u*x))/2 + log((1.0+q)*sprime));

	return tmp;
}

void Alpha::deflection_elliptical_nocore(const double x, const double y, lensvector& def)
{
	// Formulas from Tessore et al. 2015
	double phi, R = sqrt(x*x+y*y/qsq);
	phi = atan(abs(y/(q*x)));

	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	complex<double> def_complex = 2*bprime*q/(1+q)*pow(bprime/R,alpha-1)*deflection_angular_factor(phi);

	def[0] = real(def_complex);
	def[1] = imag(def_complex);
}

complex<double> Alpha::deflection_angular_factor(const double &phi)
{
	// Formulas from Tessore et al. 2015
	double beta, ff;
	beta = 2.0/(2-alpha);
	ff = (1-q)/(1+q);
	complex<double> fac = polar(1.0,phi);
	complex<double> omega = fac;
	int i=1;
	do {
		omega = polar(-ff*(beta*i - 1)/(beta*i + 1),2*phi)*omega;
		fac += omega;
		i++;
	} while (norm(omega) > def_tolerance*norm(fac));
	return fac;
}

void Alpha::hessian_elliptical_nocore(const double x, const double y, lensmatrix& hess)
{
	double R, phi, kap;
	R = sqrt(x*x+y*y/qsq);
	kap = 0.5 * (2-alpha) * pow(bprime/R, alpha);
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
	hess_complex = 2*bprime*q/(1+q)*pow(bprime/R,alpha-1)*deflection_angular_factor(phi);
	hess_complex = -kap*conj(zstar)/zstar + (1-alpha)*hess_complex/zstar; // this is the complex shear

	hess_complex = kap + hess_complex; // this is now (kappa+shear)
	hess[0][0] = real(hess_complex);
	hess[0][1] = imag(hess_complex);
	hess[1][0] = hess[0][1];
	hess_complex = 2*kap - hess_complex; // now we have transformed to (kappa-shear)
	hess[1][1] = real(hess_complex);
}

void Alpha::deflection_and_hessian_elliptical_nocore(const double x, const double y, lensvector& def, lensmatrix& hess)
{
	double R, phi, kap;
	R = sqrt(x*x+y*y/qsq);
	kap = 0.5 * (2-alpha) * pow(bprime/R, alpha);
	phi = atan(abs(y/(q*x)));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	complex<double> def_complex = 2*bprime*q/(1+q)*pow(bprime/R,alpha-1)*deflection_angular_factor(phi);
	def[0] = real(def_complex);
	def[1] = imag(def_complex);
	complex<double> hess_complex, zstar(x,-y);
	hess_complex = -kap*conj(zstar)/zstar + (1-alpha)*def_complex/zstar; // this is the complex shear
	hess_complex = kap + hess_complex; // this is now (kappa+shear)
	hess[0][0] = real(hess_complex);
	hess[0][1] = imag(hess_complex);
	hess[1][0] = hess[0][1];
	hess_complex = 2*kap - hess_complex; // now we have transformed to (kappa-shear)
	hess[1][1] = real(hess_complex);
}

double Alpha::potential_elliptical_nocore(const double x, const double y) // only for sprime=0
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
	complex<double> def_complex = 2*bprime*q/(1+q)*pow(bprime/R,alpha-1)*deflection_angular_factor(phi);
	return (x*real(def_complex) + y*imag(def_complex))/(2-alpha);
}

void Alpha::get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor)
{
	if (sprime==0.0) {
		re_major_axis = bprime*pow(zfactor,1.0/alpha);
		re_average = re_major_axis * sqrt(q);
	} else if (alpha==1.0) {
		if (sprime < bprime/2.0) {
			re_major_axis = bprime*sqrt(1-2*sprime/bprime/zfactor)*zfactor;
			re_average = re_major_axis * sqrt(q);
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

PseudoJaffe::PseudoJaffe(const double &bb, const double &aa, const double &ss, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = PJAFFE;
	model_name = "pjaffe";
	special_parameter_command = "";
	setup_base_lens(7,true); // number of parameters = 7, is_elliptical_lens = true

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	b = bb;
	s = ss;
	a = aa;
	if (s < 0) s = -s;

	update_meta_parameters_and_pointers();
}

PseudoJaffe::PseudoJaffe(const PseudoJaffe* lens_in)
{
	b = lens_in->b;
	s = lens_in->s;
	a = lens_in->a;

	copy_base_lensdata(lens_in);
	update_meta_parameters_and_pointers();
}

void PseudoJaffe::assign_paramnames()
{
	paramnames[0] = "b"; latex_paramnames[0] = "b"; latex_param_subscripts[0] = "";
	paramnames[1] = "a"; latex_paramnames[1] = "a"; latex_param_subscripts[1] = "";
	paramnames[2] = "s"; latex_paramnames[2] = "s"; latex_param_subscripts[2] = "";
	set_geometric_paramnames(3);
}

void PseudoJaffe::assign_param_pointers()
{
	param[0] = &b;
	param[1] = &a;
	param[2] = &s;
	set_geometric_param_pointers(3);
}

void PseudoJaffe::update_meta_parameters()
{
	update_ellipticity_meta_parameters();
	bprime = b*f_major_axis;
	sprime = s*f_major_axis;
	aprime = a*f_major_axis;
	qsq = q*q; ssq = sprime*sprime; asq = aprime*aprime;
}

void PseudoJaffe::assign_special_anchored_parameters(LensProfile *host_in)
{
	anchor_special_parameter = true;
	special_anchor_lens = host_in;
	double rm, ravg;
	special_anchor_lens->get_einstein_radius(rm,ravg,1.0);
	a = sqrt(ravg*b); // this is an approximate formula (a = sqrt(b*Re_halo)) and assumes the subhalo is found roughly near the Einstein radius of the halo
	update_meta_parameters();
}

void PseudoJaffe::update_special_anchored_params()
{
	if (anchor_special_parameter) {
		double rm, ravg;
		special_anchor_lens->get_einstein_radius(rm,ravg,1.0);
		a = sqrt(ravg*b); // this is an approximate formula (a = sqrt(b*Re_halo)) and assumes the subhalo is found roughly near the Einstein radius of the halo
		aprime = a/f_major_axis;
		asq = aprime*aprime;
	}
}

void PseudoJaffe::set_auto_stepsizes()
{
	stepsizes[0] = 0.2*b;
	stepsizes[1] = 0.2*b;
	stepsizes[2] = 0.02*b; // this one is a bit arbitrary, but hopefully reasonable enough
	set_auto_eparam_stepsizes(3,4);
	stepsizes[5] = 0.2*b;
	stepsizes[6] = 0.2*b;
}

void PseudoJaffe::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(3);
}

void PseudoJaffe::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&PseudoJaffe::kapavg_spherical_rsq);
	potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&PseudoJaffe::potential_spherical_rsq);
	if (q != 1.0) {
		defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&PseudoJaffe::deflection_elliptical);
		hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&PseudoJaffe::hessian_elliptical);
		potptr = static_cast<double (LensProfile::*)(const double,const double)> (&PseudoJaffe::potential_elliptical);
	}
}

double PseudoJaffe::kappa_rsq(const double rsq)
{
	return (0.5 * b * (pow(s*s+rsq, -0.5) - pow(a*a+rsq,-0.5)));
}

double PseudoJaffe::kappa_rsq_deriv(const double rsq)
{
	return (-0.25 * b * (pow(s*s+rsq, -1.5) - pow(a*a+rsq,-1.5)));
}

double PseudoJaffe::kapavg_spherical_rsq(const double rsq)
{
	return b*((sqrt(s*s+rsq)-s) - (sqrt(a*a+rsq)-a))/rsq;
}

double PseudoJaffe::potential_spherical_rsq(const double rsq)
{
	double tmp;
	tmp = b*(sqrt(s*s+rsq) - s - sqrt(a*a+rsq) + a); // now, tmp = kappa_average*rsq
	tmp += b*(a*log((a + sqrt(a*a+rsq))/(2.0*a)) - s*log((s + sqrt(s*s+rsq))/(2.0*s)));
	return tmp;
}

//  Note: although the elliptical formulas are expressed in terms of ellipticity mode 0, they use parameters
//  (the prime versions b', a', etc.) transformed from the correct emode

void PseudoJaffe::deflection_elliptical(const double x, const double y, lensvector& def)
{
	double psi, psi2, u;
	psi = sqrt(qsq*(ssq+x*x)+y*y);
	psi2 = sqrt(qsq*(asq+x*x)+y*y);
	u = sqrt(1-qsq);

	def[0] = (bprime*q/u)*(atan(u*x/(psi+sprime)) - atan(u*x/(psi2+aprime)));
	def[1] = (bprime*q/u)*(atanh(u*y/(psi+qsq*sprime)) - atanh(u*y/(psi2+qsq*aprime)));
}

void PseudoJaffe::hessian_elliptical(const double x, const double y, lensmatrix& hess)
{
	double xsq, ysq, psi, tmp1, psi2, tmp2;
	xsq=x*x; ysq=y*y;
	psi = sqrt(qsq*(ssq+xsq)+ysq);
	tmp1 = (bprime*q/psi)/(xsq+ysq+2*psi*sprime+ssq*(1+qsq));

	psi2 = sqrt(qsq*(asq+xsq)+ysq);
	tmp2 = (bprime*q/psi2)/(xsq+ysq+2*psi2*aprime+asq*(1+qsq));

	hess[0][0] = tmp1*(ysq+sprime*psi+ssq*qsq) - tmp2*(ysq+aprime*psi2+asq*qsq);
	hess[1][1] = tmp1*(xsq+sprime*psi+ssq) - tmp2*(xsq+aprime*psi2+asq);
	hess[0][1] = (-tmp1+tmp2)*x*y;
	hess[1][0] = hess[0][1];
}

double PseudoJaffe::potential_elliptical(const double x, const double y)
{
	double psi, psi2, u;
	psi = sqrt(qsq*(ssq+x*x)+y*y);
	psi2 = sqrt(qsq*(asq+x*x)+y*y);
	u = sqrt(1-qsq);

	return (bprime*q/u)*(x*(atan(u*x/(psi+sprime)) - atan(u*x/(psi2+aprime)))+ y*(atanh(u*y/(psi+qsq*sprime))
		- atanh(u*y/(psi2+qsq*aprime)))) + bprime*q*(sprime*(-log(SQR(psi+sprime) + SQR(u*x))/2 + log((1.0+q)*sprime))
		- aprime*(-log(SQR(psi2+aprime) + SQR(u*x))/2 + log((1.0+q)*aprime)));
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
	b_kpc = b / kpc_to_arcsec;
	sigma = c * sqrt(b_kpc*(1-s/a)*(Rs_sun_km/kpc_to_km)*sigma_cr/2);
	cout << "sigma = " << sigma << " km/sprime  (velocity dispersion)\n";
	r_tidal = a / kpc_to_arcsec;
	r_core = s / kpc_to_arcsec;
	cout << "r_tidal = " << r_tidal << " kpc" << endl;
	cout << "r_core = " << r_core << " kpc" << endl;
	cout << endl;
}

/************************************* NFW *************************************/

NFW::NFW(const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = nfw;
	model_name = "nfw";
	special_parameter_command = "";
	setup_base_lens(6,true); // number of parameters = 6, is_elliptical_lens = true
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	ks = ks_in;
	rs = rs_in;

	update_meta_parameters_and_pointers();
}

NFW::NFW(const NFW* lens_in)
{
	ks = lens_in->ks;
	rs = lens_in->rs;

	copy_base_lensdata(lens_in);
	update_meta_parameters_and_pointers();
}

void NFW::assign_paramnames()
{
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

void NFW::update_meta_parameters()
{
	rmin_einstein_radius = 1e-6*rs; // for determining the Einstein radius (sets lower bound of root finder)
}

void NFW::set_auto_stepsizes()
{
	stepsizes[0] = 0.2*ks;
	stepsizes[1] = 0.2*rs;
	set_auto_eparam_stepsizes(2,3);
	stepsizes[4] = 0.5; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
	stepsizes[5] = 0.5;
}

void NFW::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(2);
}

void NFW::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::kapavg_spherical_rsq);
	potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::potential_spherical_rsq);
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
	// below xsq ~ 1e-6 or so, kappa formula becomes inaccurate due to fine cancellations; a series expansion
	// is done for xsq smaller than this
	if (xsq < 1e-6) return -ks/rsq;
	else if (abs(xsq-1.0) < 1e-5) return -0.4*sqrt(xsq); // formula on next line becomes unstable for x close to 1, this fixes it
	else return -(ks/rsq)*((xsq*(2.0-3*lens_function_xsq(xsq)) + 1)/((xsq-1)*(xsq-1)));
}

inline double NFW::lens_function_xsq(const double &xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}

double NFW::kapavg_spherical_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	// below xsq ~ 1e-6 or so, this becomes inaccurate due to fine cancellations; a series expansion
	// is done for xsq smaller than this
	if (xsq > 1e-6)
		return 2*ks*(2*lens_function_xsq(xsq) + log(xsq/4))/xsq;
	else
		return -ks*(1+log(xsq/4));
}

double NFW::potential_spherical_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (xsq < 1) {
		if (xsq > 1e-4)
			return 2*ks*rs*rs*(-SQR(atanh(sqrt(1-xsq))) + SQR(log(xsq/4)/2));
		else
			return -ks*rsq*log(xsq/4)/2;
	}
	else {
		return 2*ks*rs*rs*(SQR(atan(sqrt(xsq-1))) + SQR(log(xsq/4)/2));
	}
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

/********************************** Truncated_NFW **********************************/

Truncated_NFW::Truncated_NFW(const double &ks_in, const double &rs_in, const double &rt_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = TRUNCATED_nfw;
	model_name = "tnfw";
	special_parameter_command = "";
	setup_base_lens(7,true); // number of parameters = 7, is_elliptical_lens = true
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	ks = ks_in;
	rs = rs_in;
	rt = rt_in;

	update_meta_parameters_and_pointers();
}

Truncated_NFW::Truncated_NFW(const Truncated_NFW* lens_in)
{
	ks = lens_in->ks;
	rs = lens_in->rs;
	rt = lens_in->rt;

	copy_base_lensdata(lens_in);
	update_meta_parameters_and_pointers();
}

void Truncated_NFW::assign_paramnames()
{
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

void Truncated_NFW::update_meta_parameters()
{
	rmin_einstein_radius = 1e-6*rs;
}

void Truncated_NFW::set_auto_stepsizes()
{
	stepsizes[0] = 0.2*ks;
	stepsizes[1] = 0.2*rs;
	stepsizes[2] = 0.2*rt;
	set_auto_eparam_stepsizes(3,4);
	stepsizes[5] = 0.5; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
	stepsizes[6] = 0.5;
}

void Truncated_NFW::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(3);
}

void Truncated_NFW::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Truncated_NFW::kapavg_spherical_rsq);
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

double Truncated_NFW::kapavg_spherical_rsq(const double rsq)
{
	double xsq, tau, tsq, sqrttx, lx, tmp;
	xsq = rsq/(rs*rs);
	tau = rt/rs;
	tsq = tau*tau;
	sqrttx = sqrt(tsq+xsq);
	lx = log(sqrt(xsq)/(sqrttx+sqrt(tsq)));
	tmp = 2*(tsq+1+4*(xsq-1))*lens_function_xsq(xsq) + (M_PI*(3*tsq-1) + 2*tau*(tsq-3)*log(tau))/tau + (-CUBE(tau)*M_PI*(4*(tsq+xsq)-tsq-1) + (-tsq*(tsq*tsq-1) + (tsq+xsq)*(3*tsq*tsq-6*tsq-1))*lx)/CUBE(tau)/sqrttx;
	return 2*ks*tsq*tsq/CUBE(tsq+1)/xsq*tmp; // now, tmp = kappa_average
}

/********************************** Hernquist **********************************/

Hernquist::Hernquist(const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = HERNQUIST;
	model_name = "hern";
	special_parameter_command = "";
	setup_base_lens(6,true); // number of parameters = 6, is_elliptical_lens = true
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	ks = ks_in;
	rs = rs_in;

	update_meta_parameters_and_pointers();
}

Hernquist::Hernquist(const Hernquist* lens_in)
{
	ks = lens_in->ks;
	rs = lens_in->rs;

	copy_base_lensdata(lens_in);
	update_meta_parameters_and_pointers();
}

void Hernquist::assign_paramnames()
{
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

void Hernquist::update_meta_parameters()
{
	rmin_einstein_radius = 1e-6*rs;
}

void Hernquist::set_auto_stepsizes()
{
	stepsizes[0] = 0.2*ks;
	stepsizes[1] = 0.2*rs;
	set_auto_eparam_stepsizes(2,3);
	stepsizes[4] = 0.5; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
	stepsizes[5] = 0.5;
}

void Hernquist::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(2);
}

void Hernquist::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Hernquist::kapavg_spherical_rsq);
	potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Hernquist::potential_spherical_rsq);
}

double Hernquist::kappa_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (abs(xsq-1.0) < 1e-4) return 0.4*ks*(0.666666666667 - 0.571428571429*(xsq-1)); // function on next line becomes unstable for x close to 1, this fixes it
	return (ks*(-3 + (2+xsq)*lens_function_xsq(xsq))/((xsq-1)*(xsq-1)));
}

double Hernquist::kappa_rsq_deriv(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (abs(xsq-1.0) < 1e-4) return -0.4*ks*(0.571428571429)/(rs*rs); // function on next line becomes unstable for x close to 1, this fixes it
	return ((ks/((2*rsq)*CUBE(xsq-1))) * (-3*xsq*lens_function_xsq(xsq)*(xsq+4) + 13*xsq + 2));
}

inline double Hernquist::lens_function_xsq(const double xsq)
{
	return ((sqrt(xsq) > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (sqrt(xsq) < 1.0) ? (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}

double Hernquist::kapavg_spherical_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	if (xsq==1) return (2*ks/3.0);
	else return 2*ks*(1 - lens_function_xsq(xsq))/(xsq - 1);
}

double Hernquist::potential_spherical_rsq(const double rsq)
{
	double xsq = rsq/(rs*rs);
	return ks*rs*rs*(log(xsq/4) + 2*lens_function_xsq(xsq));
}

/********************************** Exponential Disk **********************************/

ExpDisk::ExpDisk(const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = EXPDISK;
	model_name = "expdisk";
	special_parameter_command = "";
	setup_base_lens(6,true); // number of parameters = 6, is_elliptical_lens = true
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	k0 = k0_in;
	R_d = R_d_in;

	update_meta_parameters_and_pointers();
}

ExpDisk::ExpDisk(const ExpDisk* lens_in)
{
	k0 = lens_in->k0;
	R_d = lens_in->R_d;

	copy_base_lensdata(lens_in);
	update_meta_parameters_and_pointers();
}

void ExpDisk::assign_paramnames()
{
	paramnames[0] = "k0";  latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	paramnames[1] = "R_d"; latex_paramnames[1] = "R";       latex_param_subscripts[1] = "d";
	set_geometric_paramnames(2);
}

void ExpDisk::assign_param_pointers()
{
	param[0] = &k0;
	param[1] = &R_d;
	set_geometric_param_pointers(2);
}

void ExpDisk::update_meta_parameters()
{
	rmin_einstein_radius = 1e-6*R_d;
}

void ExpDisk::set_auto_stepsizes()
{
	stepsizes[0] = 0.2*k0;
	stepsizes[1] = 0.2*R_d;
	set_auto_eparam_stepsizes(2,3);
	stepsizes[4] = 0.5; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
	stepsizes[5] = 0.5;
}

void ExpDisk::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(2);
}

void ExpDisk::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&ExpDisk::kapavg_spherical_rsq);
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

double ExpDisk::kapavg_spherical_rsq(const double rsq)
{
	double x = sqrt(rsq)/R_d;
	return 2*k0*(1 - (1+x)*exp(-x))/(x*x);
}

/***************************** External shear *****************************/

Shear::Shear(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in)
{
	lenstype = SHEAR;
	model_name = "shear";
	special_parameter_command = "";
	setup_base_lens(4,false); // number of parameters = 4, is_elliptical_lens = false

	if (use_shear_component_params) {
		shear1 = shear_p1_in;
		shear2 = shear_p2_in;
	} else {
		q = shear_p1_in;
		set_angle(shear_p2_in);
	}
	x_center = xc_in;
	y_center = yc_in;
	update_meta_parameters();
}

Shear::Shear(const Shear* lens_in)
{
	shear1 = lens_in->shear1;
	shear2 = lens_in->shear2;
	copy_base_lensdata(lens_in);
	update_meta_parameters();
}

void Shear::assign_paramnames()
{
	if (use_shear_component_params) {
		paramnames[0] = "shear1";      latex_paramnames[0] = "\\gamma"; latex_param_subscripts[0] = "1";
		paramnames[1] = "shear2";      latex_paramnames[1] = "\\gamma"; latex_param_subscripts[1] = "2";
	} else {
		paramnames[0] = "shear";       latex_paramnames[0] = "\\gamma"; latex_param_subscripts[0] = "";
		paramnames[1] = "theta_shear"; latex_paramnames[1] = "\\theta"; latex_param_subscripts[1] = "\\gamma";
	}
	paramnames[2] = "xc"; latex_paramnames[2] = "x"; latex_param_subscripts[2] = "c";
	paramnames[3] = "yc"; latex_paramnames[3] = "y"; latex_param_subscripts[3] = "c";
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
	param[2] = &x_center;
	param[3] = &y_center;
}

void Shear::update_meta_parameters()
{
	if (use_shear_component_params) {
		q = sqrt(SQR(shear1) + SQR(shear2));
		set_angle_from_components(shear1,shear2);
	} else {
		theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
		shear1 = -q*cos(2*theta_eff);
		shear2 = -q*sin(2*theta_eff);
	}
}

void Shear::set_auto_stepsizes()
{
	if (use_shear_component_params) {
		stepsizes[0] = 0.035;
		stepsizes[1] = 0.035;
	} else {
		stepsizes[0] = 0.05;
		stepsizes[1] = 20;
	}
	stepsizes[2] = 0.1; // very arbitrary, but shear is usually center_anchored anyway
	stepsizes[3] = 0.1;
}

void Shear::set_auto_ranges()
{
	if (use_shear_component_params) {
		set_auto_penalty_limits[0] = false;
		set_auto_penalty_limits[1] = false;
	} else {
		set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
		set_auto_penalty_limits[1] = false;
	}
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = false;
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

void Shear::potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	def[0] = x*shear1 + y*shear2;
	def[1] = -y*shear1 + x*shear2;
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
	stringstream mstr;
	string mstring;
	mstr << m_in;
	mstr >> mstring;
	special_parameter_command = "m=" + mstring;
	kappa_multipole = kap; // specifies whether it is a multipole in the potential or in kappa
	sine_term = sine;
	setup_base_lens(5,false); // number of parameters = 5, is_elliptical_lens = false

	n = n_in;
	m = m_in;
	q = A_m_in;
	set_angle(theta_degrees);
	x_center = xc_in;
	y_center = yc_in;

	update_meta_parameters();
	set_model_specific_integration_pointers();
}

Multipole::Multipole(const Multipole* lens_in)
{
	n = lens_in->n;
	m = lens_in->m;
	kappa_multipole = lens_in->kappa_multipole;
	sine_term = lens_in->sine_term;

	copy_base_lensdata(lens_in);
	update_meta_parameters();
	set_model_specific_integration_pointers();
}

void Multipole::assign_paramnames()
{
	string mstring;
	stringstream mstr;
	mstr << m;
	mstr >> mstring;
	if (sine_term) {
		paramnames[0] = "B_" + mstring;  latex_paramnames[0] = "B"; latex_param_subscripts[0] = mstring;
	} else {
		paramnames[0] =  "A_" + mstring; latex_paramnames[0] = "A"; latex_param_subscripts[0] = mstring;
	}
	if (kappa_multipole) {
		paramnames[1] = "beta"; latex_paramnames[1] = "\\beta"; latex_param_subscripts[1] = "";
	} else {
		paramnames[1] = "n";    latex_paramnames[1] = "n"; latex_param_subscripts[1] = "";
	}
	paramnames[2] = "theta"; latex_paramnames[2] = "\\theta"; latex_param_subscripts[2] = "";
	paramnames[3] = "xc";    latex_paramnames[3] = "x";       latex_param_subscripts[3] = "c";
	paramnames[4] = "yc";    latex_paramnames[4] = "y";       latex_param_subscripts[4] = "c";
}

void Multipole::assign_param_pointers()
{
	param[0] = &q; // here, q is actually the shear magnitude
	param[1] = &n;
	param[2] = &theta; angle_paramnum = 2;
	param[3] = &x_center;
	param[4] = &y_center;
}

void Multipole::update_meta_parameters()
{
	theta_eff = (orient_major_axis_north) ? theta + M_HALFPI : theta;
	if (sine_term) theta_eff += M_HALFPI/m;
}

void Multipole::set_auto_stepsizes()
{
	stepsizes[0] = 0.05;
	stepsizes[1] = 0.1;
	stepsizes[2] = 20;
	stepsizes[3] = 0.1; // very arbitrary, but a multipole term is usually center_anchored anyway
	stepsizes[4] = 0.1;
}

void Multipole::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = false;
	set_auto_penalty_limits[4] = false;
}

void Multipole::set_model_specific_integration_pointers()
{
	defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector&)> (&Multipole::deflection);
	hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix&)> (&Multipole::hessian);
	kapavgptr_rsq_spherical = NULL;
	if (m==0) kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Multipole::deflection_m0_spherical_r);
}

double Multipole::kappa(double x, double y)
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

void Multipole::potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;

	double r, rsq, rcube, xy, xx, yy, phi, ddpsi, psi, dpsi, cs, ss;
	int mm = m*m;
	phi = atan(abs(y/x));
	xx = x*x;
	yy = y*y;
	rsq = xx+yy;
	r = sqrt(rsq);
	rcube = rsq*r;
	xy = x*y;

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

	if (kappa_multipole) {
		ddpsi = (1-n)*dpsi/r;
	} else {
		ddpsi = (n-1)*dpsi/r;
	}

	hess[0][0] = (ddpsi*xx + dpsi*yy/r - psi*mm*yy/rsq)*cs/rsq + (dpsi - psi/r)*2*m*xy*ss/rcube;
	hess[1][1] = (ddpsi*yy + dpsi*xx/r - psi*mm*xx/rsq)*cs/rsq + (-dpsi + psi/r)*2*m*xy*ss/rcube;
	hess[0][1] = (ddpsi - dpsi/r + psi*mm/rsq)*xy*cs/rsq + (dpsi - psi/r)*(yy-xx)*m*ss/rcube;
	hess[1][0] = hess[0][1];
}

void Multipole::kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess)
{
	potential_derivatives(x,y,def,hess);
	kap = kappa(x,y);
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
			if (m==0) b = pow(-q*zfactor*n,1.0/(2-n));
			else b = pow(-(q*zfactor*n)/m,1.0/(2-n));
		}
	}
	re_major_axis = re_average = b;
}

/***************************** Point mass *****************************/

PointMass::PointMass(const double &bb, const double &xc_in, const double &yc_in)
{
	lenstype = PTMASS;
	model_name = "ptmass";
	special_parameter_command = "";
	setup_base_lens(3,false); // number of parameters = 3, is_elliptical_lens = false
	b = bb;
	x_center = xc_in;
	y_center = yc_in;
	assign_param_pointers();
	assign_paramnames();
}

PointMass::PointMass(const PointMass* lens_in)
{
	b = lens_in->b;
	// the base class copies q and theta, which are useless here, but it's simpler to just call it
	copy_base_lensdata(lens_in);
}

void PointMass::assign_paramnames()
{
	paramnames[0] = "b";  latex_paramnames[0] = "b"; latex_param_subscripts[0] = "";
	paramnames[1] = "xc"; latex_paramnames[1] = "x"; latex_param_subscripts[1] = "c";
	paramnames[2] = "yc"; latex_paramnames[2] = "y"; latex_param_subscripts[2] = "c";
}

void PointMass::assign_param_pointers()
{
	param[0] = &b;
	param[1] = &x_center;
	param[2] = &y_center;
	angle_paramnum = -1; // since there is no angle parameter
}

void PointMass::set_auto_stepsizes()
{
	stepsizes[0] = 0.1*b;
	stepsizes[1] = 0.1*b;
	stepsizes[2] = 0.1*b;
}

void PointMass::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
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
	double bsq = b*b, rsq = x*x + y*y;
	def[0] = bsq*x/rsq;
	def[1] = bsq*y/rsq;
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

void PointMass::potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	double bsq = b*b, xsq = x*x, ysq = y*y, rsq = xsq+ysq, r4 = SQR(rsq);
	def[0] = bsq*x/rsq;
	def[1] = bsq*y/rsq;
	hess[0][0] = bsq*(ysq-xsq)/r4;
	hess[1][1] = -hess[0][0];
	hess[1][0] = -2*bsq*x*y/r4;
	hess[0][1] = hess[1][0];
}

void PointMass::get_einstein_radius(double& r1, double& r2, const double zfactor)
{
	r1 = b*sqrt(zfactor);
	r2 = r1;
}

/***************************** Core/Cusp Model *****************************/

CoreCusp::CoreCusp(const double &mass_param_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, bool parametrize_einstein_radius)
{
	lenstype = CORECUSP;
	model_name = "corecusp";
	special_parameter_command = ((parametrize_einstein_radius==true) ? "re_param" : "");
	set_k0_by_einstein_radius = parametrize_einstein_radius;
	setup_base_lens(9,true); // number of parameters = 9, is_elliptical_lens = true
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	gamma = gamma_in;
	n = n_in;
	a = a_in;
	s = s_in;
	if (s < 0) s = -s; // don't allow negative core radii
	if (a < 0) a = -a; // don't allow negative scale radii
	if (a < s) die("scale radius a cannot be less than core radius s for corecusp model");
	if (gamma >= n) die("inner slope cannot be equal to or greater than than outer slope for corecusp model");
	if (gamma >= 3) die("inner slope cannot be equal to or greater than 3 for corecusp model (mass diverges at r=0)");
	if (n <= 1) die("outer slope cannot be equal to or less than 1 for corecusp model");
	if (set_k0_by_einstein_radius) {
		einstein_radius = mass_param_in;
		if (einstein_radius < 0) einstein_radius = -einstein_radius; // don't allow negative einstein radius
		k0 = 1.0; // This will be reset when update_meta_parameters() is called
	}
	else k0 = mass_param_in;

	update_meta_parameters_and_pointers();
}

CoreCusp::CoreCusp(const CoreCusp* lens_in)
{
	k0 = lens_in->k0;
	gamma = lens_in->gamma;
	n = lens_in->n;
	a = lens_in->a;
	s = lens_in->s;
	set_k0_by_einstein_radius = lens_in->set_k0_by_einstein_radius;
	if (set_k0_by_einstein_radius) einstein_radius = lens_in->einstein_radius;

	copy_base_lensdata(lens_in);
	update_meta_parameters_and_pointers();
}

void CoreCusp::assign_paramnames()
{
	if (set_k0_by_einstein_radius) {
		paramnames[0] = "Re"; latex_paramnames[0] = "R";       latex_param_subscripts[0] = "e";
	} else {
		paramnames[0] = "k0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	}
	paramnames[1] = "gamma"; latex_paramnames[1] = "\\gamma"; latex_param_subscripts[1] = "";
	paramnames[2] = "n";     latex_paramnames[2] = "n";       latex_param_subscripts[2] = "";
	paramnames[3] = "a";     latex_paramnames[3] = "a";       latex_param_subscripts[3] = "";
	paramnames[4] = "s";     latex_paramnames[4] = "s";       latex_param_subscripts[4] = "";
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

void CoreCusp::update_meta_parameters()
{
	update_ellipticity_meta_parameters();
	if (a < s) die("scale radius a cannot be less than core radius s for corecusp model");
	if (gamma >= n) die("inner slope cannot be equal to or greater than than outer slope for corecusp model");
	if (gamma >= 3) die("inner slope cannot be equal to or greater than 3 for corecusp model (mass diverges at r=0)");
	if (n <= 1) die("outer slope cannot be equal to or less than 1 for corecusp model");
	digamma_term = DiGamma(1.5-gamma/2);
	double p = (n-1.0)/2;
	beta_p1 = Beta(p,0.5);
	beta_p2 = beta_p1/(1+1.0/(2*p)); // Beta(p,1.5)
	if (set_k0_by_einstein_radius) {
		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
		k0 = k0 / kapavg_spherical_rsq(einstein_radius*einstein_radius);
	}
	if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
}


void CoreCusp::assign_special_anchored_parameters(LensProfile *host_in)
{
	anchor_special_parameter = true;
	special_anchor_lens = host_in;
	double rm, ravg;
	special_anchor_lens->get_einstein_radius(rm,ravg,1.0);
	if (set_k0_by_einstein_radius) {
		a = sqrt(ravg*einstein_radius); // Not good! Only true for Pseudo-Jaffe subhalo. Fix this later (if it can be fixed)
		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
		k0 = k0 / kapavg_spherical_rsq(einstein_radius*einstein_radius);
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
		special_anchor_lens->get_einstein_radius(rm,ravg,1.0);
		if (set_k0_by_einstein_radius) {
			a = sqrt(ravg*einstein_radius);
			if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
			k0 = k0 / kapavg_spherical_rsq(einstein_radius*einstein_radius);
			if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
		}
		else {
			if (gamma >= 3) a=1e30; // effectively infinite in this case (not physical however, because the mass diverges at the center of the subhalo)
			else a = ravg*k0/(3-gamma); // we have ignored the core in this formula, but should be reasonable as long as a >> s
		}
		if (s != 0) set_core_enclosed_mass(); else core_enclosed_mass = 0;
	}
}

void CoreCusp::set_auto_stepsizes()
{
	if (set_k0_by_einstein_radius) stepsizes[0] = 0.1*einstein_radius;
	else stepsizes[0] = 0.1*k0;
	stepsizes[1] = 0.1;
	stepsizes[2] = 0.1;
	stepsizes[3] = 0.1*a;
	stepsizes[4] = 0.02*a;
	set_auto_eparam_stepsizes(5,6);
	if (set_k0_by_einstein_radius) {
		// take advantage of the fact that we're keeping track of the Einstein radius
		stepsizes[7] = 0.1*einstein_radius;
		stepsizes[8] = 0.1*einstein_radius;
	} else {
		stepsizes[7] = 0.1; // arbitrary...maybe we should just calculate Einstein radius before determining stepsizes anyway?
		stepsizes[8] = 0.1;
	}
}

void CoreCusp::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = true; penalty_lower_limits[3] = 0; penalty_upper_limits[3] = 1e30;
	set_auto_penalty_limits[4] = true; penalty_lower_limits[4] = 0; penalty_upper_limits[4] = 1e30;
	set_geometric_param_auto_ranges(5);
}

void CoreCusp::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&CoreCusp::kapavg_spherical_rsq);
}

double CoreCusp::kappa_rsq(const double rsq)
{
	double atilde = sqrt(a*a-s*s);
	return pow(a/atilde,n) * kappa_rsq_nocore(rsq+s*s,atilde);
}

double CoreCusp::kappa_rsq_nocore(const double rsq_prime, const double atilde)
{
	// Formulas for the non-cored profile are from Munoz et al. 2001
	double ks, xisq, p, hyp, ans;
	p = (n-1.0)/2;
	ks = k0*atilde/(a*M_2PI);
	xisq = rsq_prime/(atilde*atilde);
	hyp = real(hyp_2F1(p,gamma/2,n/2,1/(1+xisq)));
	ans = ks*beta_p1*pow(1+xisq,-p)*hyp;
	return ans;
}

double CoreCusp::kappa_rsq_deriv(const double rsq)
{
	double atilde = sqrt(a*a-s*s);
	return pow(a/atilde,n) * kappa_rsq_deriv_nocore(rsq+s*s,atilde);
}

double CoreCusp::kappa_rsq_deriv_nocore(const double rsq_prime, const double atilde)
{
	double ks, xisq, hyp, ans;
	ks = k0*atilde/(a*M_2PI);
	xisq = rsq_prime/(atilde*atilde);
	hyp = n*(1+xisq)*real(hyp_2F1((n-1.0)/2,gamma/2,n/2,1/(1+xisq))) + gamma*real(hyp_2F1((n+1.0)/2,(gamma+2.0)/2,(n+2.0)/2,1/(1+xisq)));
	return -(ks/(2*atilde*atilde))*beta_p2*pow(1+xisq,-(n+3.0)/2)*hyp;
}

void CoreCusp::set_core_enclosed_mass()
{
	if (n==3) core_enclosed_mass = enclosed_mass_spherical_nocore_n3(s*s,sqrt(a*a-s*s),nstep);
	else core_enclosed_mass = enclosed_mass_spherical_nocore(s*s,sqrt(a*a-s*s));
}

double CoreCusp::kapavg_spherical_rsq(const double rsq)
{
	double atilde, kapavg;
	if (s != 0) {
		atilde = sqrt(a*a-s*s);
		if (n==3) {
			kapavg = pow(1-SQR(s/a),-n/2) * (enclosed_mass_spherical_nocore_n3(rsq+s*s,atilde,nstep) - core_enclosed_mass) / rsq;
		} else {
			kapavg = pow(1-SQR(s/a),-n/2) * (enclosed_mass_spherical_nocore(rsq+s*s,atilde) - core_enclosed_mass) / rsq;
		}
	} else {
		if (n==3) {
			kapavg = enclosed_mass_spherical_nocore_n3(rsq,a,nstep) / rsq;
		}
		else {
			kapavg = enclosed_mass_spherical_nocore(rsq,a) / rsq;
		}
	}
	return kapavg;
}

double CoreCusp::enclosed_mass_spherical_nocore(const double rsq_prime, const double atilde, const double nprime) // actually mass_enclosed/(pi*sigma_crit)
{
	double xisq, p, hyp;
	xisq = rsq_prime/(atilde*atilde);
	p = (nprime-3.0)/2;
	hyp = pow(1+xisq,-p) * real(hyp_2F1(p,gamma/2,nprime/2,1.0/(1+xisq)));

	//double ans = 2*k0*CUBE(atilde)/(a*M_2PI) * (Beta(p,(3-gamma)/2) - Beta(p,1.5)*hyp);
		//if (ans*0.0 != 0.0) {
			//cout << "deflection: a=" << atilde << " gamma=" << gamma << " xisq=" << xisq << " def=" << ans << " hyp=" << hyp << " B1=" << Beta(p,(3-gamma)/2) << " B2=" << Beta(p,1.5)*hyp << " " << endl;
			//print_parameters();
			//die();
		//}
	return 2*k0*CUBE(atilde)/(a*M_2PI) * (Beta(p,(3-gamma)/2) - Beta(p,1.5)*hyp);
}

double CoreCusp::enclosed_mass_spherical_nocore_n3(const double rsq_prime, const double atilde, const double n_stepsize) // actually mass_enclosed/(pi*sigma_crit)
{
	// if gamma = 1, use GFunction2 (eq. 67 of Keeton 2001), but for very small r/a, use Richardson extrapolation which requires fewer iterations
	// for other values of gamma, use Gfunction1 (eq. 66) if r < a, and GFunction2 (eq. 67) if r >= a

	double xisq = rsq_prime/(atilde*atilde);
	if ((gamma == 1.0) and (xisq < 0.01)) return enclosed_mass_spherical_nocore_limit(rsq_prime,atilde,n_stepsize); // in this regime, Richardson extrapolation is faster
	double x, p, fac;
	p = (3-gamma)/2;
	if ((xisq < 1) and ((gamma-1.0)/2 > 1e-12)) {
		x=xisq/(1+xisq);
		fac = log(1+xisq) - G_Function(gamma/2,(gamma-1)/2,x) - Beta(-p,1.5)*real(hyp_2F1(1.5,p,1+p,x))*pow(x,p); // uses Gfunction1
	} else {
		x=1.0/(1+xisq);
		fac = log(1+xisq) - G_Function(gamma/2,1.5,x) + digamma_three_halves - digamma_term; // uses Gfunction2
	}

	if (fac*0.0 != 0.0) {
		cout << "NaN deflection: a=" << atilde << " gamma=" << gamma << " xisq=" << xisq << " dig=" << digamma_term << " gf=" << (gamma-1)/2 << " " << G_Function(gamma/2,0.001,x) << " " << Beta(-p,1.5) << " " << real(hyp_2F1(1.5,p,1+p,x)) << endl;
		print_parameters();
		die();
	}

	return 2*k0*CUBE(atilde)/(a*M_2PI) * fac;
}

double CoreCusp::enclosed_mass_spherical_nocore_limit(const double rsq, const double atilde, const double n_stepsize)
{
	// This uses Richardson extrapolation to calculate the enclosed mass, which can be used for the n=3 case
	const double CON=1.4, CON2=(CON*CON);
	const double BIG=1.0e100;
	const double SAFE=2.0;
	const int NTAB=100;
	int i,j;
	double errt,fac,hh,ans=0.0;
	double **a = new double*[NTAB];
	for (i=0; i < NTAB; i++) a[i] = new double[NTAB];

	hh=n_stepsize;
	a[0][0] = 0.5*(enclosed_mass_spherical_nocore(rsq,atilde,n+hh) + enclosed_mass_spherical_nocore(rsq,atilde,n-hh));
	double err=BIG;
	for (i=1;i<NTAB;i++) {
		hh /= CON;
		a[0][i] = 0.5*(enclosed_mass_spherical_nocore(rsq,atilde,n+hh) + enclosed_mass_spherical_nocore(rsq,atilde,n-hh));

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


SersicLens::SersicLens(const double &kappa_e_in, const double &Re_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = SERSIC_LENS;
	model_name = "sersic";
	special_parameter_command = "";
	setup_base_lens(7,true); // number of parameters = 7, is_elliptical_lens = true

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_default_base_settings(nn,acc);
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	n = n_in;
	re = Re_in;
	kappa_e = kappa_e_in;

	update_meta_parameters_and_pointers();
}

SersicLens::SersicLens(const SersicLens* lens_in)
{
	kappa_e = lens_in->kappa_e;
	n = lens_in->n;
	re = lens_in->re;
	b = lens_in->b;

	copy_base_lensdata(lens_in);
	update_meta_parameters_and_pointers();
}

void SersicLens::assign_paramnames()
{
	paramnames[0] = "kappa_e"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "e";
	paramnames[1] = "R_eff";   latex_paramnames[1] = "R";       latex_param_subscripts[1] = "eff";
	paramnames[2] = "n";       latex_paramnames[2] = "n";       latex_param_subscripts[2] = "";
	set_geometric_paramnames(3);
}

void SersicLens::assign_param_pointers()
{
	param[0] = &kappa_e;
	param[1] = &re;
	param[2] = &n;
	set_geometric_param_pointers(3);
}

void SersicLens::update_meta_parameters()
{
	update_ellipticity_meta_parameters();
	b = 2*n - 0.33333333333333 + 4.0/(405*n) + 46.0/(25515*n*n) + 131.0/(1148175*n*n*n);
	def_factor = 2*n*re*re*kappa_e*pow(b,-2*n)*exp(b);
}

void SersicLens::set_auto_stepsizes()
{
	stepsizes[0] = 0.2*kappa_e;
	stepsizes[1] = 0.2*re;
	stepsizes[2] = 0.2;
	set_auto_eparam_stepsizes(3,4);
	stepsizes[5] = 0.3; // these are quite arbitrary--should calculate Einstein radius and use 0.05*r_ein
	stepsizes[6] = 0.3;
}

void SersicLens::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = false;
	set_geometric_param_auto_ranges(3);
}

void SersicLens::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&SersicLens::kapavg_spherical_rsq);
}

double SersicLens::kappa_rsq(const double rsq)
{
	return kappa_e*exp(-b*(pow(rsq/(re*re),0.5/n)-1));
}

double SersicLens::kappa_rsq_deriv(const double rsq)
{
	return -kappa_e*exp(-b*(pow(rsq/(re*re),0.5/n)-1))*b*pow(re,-1.0/n)*pow(rsq,0.5/n-1)/(2*n);
}

double SersicLens::kapavg_spherical_rsq(const double rsq)
{
	// Formula from Cardone et al. 2003
	double x, alpha_e_times_2re, gamm2n, incgam2n;
	x = pow(rsq/(re*re),1.0/(2*n));
	IncGammaP_and_Gamma(2*n,b*x,incgam2n,gamm2n);
	return def_factor*gamm2n*incgam2n/rsq;  // def_factor is equal to 2*re*alpha_re/Gamma(2n), where alpha_re is the deflection at re
}


/***************************** Mass sheet *****************************/

MassSheet::MassSheet(const double &kext_in, const double &xc_in, const double &yc_in)
{
	lenstype = SHEET;
	model_name = "sheet";
	special_parameter_command = "";
	setup_base_lens(3,false); // number of parameters = 3, is_elliptical_lens = false

	kext = kext_in;
	x_center = xc_in;
	y_center = yc_in;
}

MassSheet::MassSheet(const MassSheet* lens_in)
{
	kext = lens_in->kext;
	copy_base_lensdata(lens_in);
}

void MassSheet::assign_paramnames()
{
	paramnames[0] = "kext"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "ext";
	paramnames[1] = "xc";   latex_paramnames[1] = "x";       latex_param_subscripts[1] = "c";
	paramnames[2] = "yc";   latex_paramnames[2] = "y";       latex_param_subscripts[2] = "c";
}

void MassSheet::assign_param_pointers()
{
	param[0] = &kext;
	param[1] = &x_center;
	param[2] = &y_center;
	angle_paramnum = -1; // since there is no angle parameter
}

void MassSheet::set_auto_stepsizes()
{
	stepsizes[0] = 0.3*kext;
	stepsizes[1] = 0.1; // arbitrary! really, the center should never be independently varied
	stepsizes[2] = 0.1;
}

void MassSheet::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
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

void MassSheet::potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	def[0] = kext*x;
	def[1] = kext*y;
	hess[0][0] = kext;
	hess[1][1] = kext;
	hess[1][0] = 0;
	hess[0][1] = 0;
}

/******************************* Tabulated Model *******************************/

Tabulated_Model::Tabulated_Model(const double &kscale_in, const double &theta_in, const double &xc_in, const double &yc_in, LensProfile* lens_in, const double xmin, const double xmax, const int x_N, const double ymin, const double ymax, const int y_N)
{
	//cout << "HI " << x_N << " " << y_N << " " << xmin << " " << xmax << " " << ymin << " " << ymax << endl;
	lenstype = TABULATED;
	model_name = "tab";
	special_parameter_command = "";
	setup_base_lens(4,false); // number of parameters = 3, is_elliptical_lens = false

	kscale = kscale_in;
	theta = degrees_to_radians(theta_in);
	x_center = xc_in;
	y_center = yc_in;
	update_meta_parameters();

	grid_x_N = x_N;
	grid_y_N = y_N;
	grid_xlength = xmax-xmin;
	grid_ylength = ymax-ymin;
	grid_xvals = new double[x_N];
	grid_yvals = new double[y_N];

	kappa_vals = new double*[x_N];
	pot_vals = new double*[x_N];
	defx = new double*[x_N];
	defy = new double*[x_N];
	hess_xx = new double*[x_N];
	hess_yy = new double*[x_N];
	hess_xy = new double*[x_N];
	int i,j;
	for (i=0; i < x_N; i++) {
		kappa_vals[i] = new double[y_N];
		pot_vals[i] = new double[y_N];
		defx[i] = new double[y_N];
		defy[i] = new double[y_N];
		hess_xx[i] = new double[y_N];
		hess_yy[i] = new double[y_N];
		hess_xy[i] = new double[y_N];
	}

	lensvector def_in;
	lensmatrix hess_in;
	double x,y;
	double xstep = grid_xlength/(x_N-1);
	double ystep = grid_ylength/(y_N-1);
	for (i=0, x=xmin; i < x_N; i++, x += xstep) grid_xvals[i] = x;
	for (j=0, y=ymin; j < y_N; j++, y += ystep) grid_yvals[j] = y;

	for (i=0, x=xmin; i < x_N; i++, x += xstep) {
		for (j=0, y=ymin; j < y_N; j++, y += ystep) {
			kappa_vals[i][j] = lens_in->kappa(x,y);
			pot_vals[i][j] = lens_in->potential(x,y);
			lens_in->deflection(x,y,def_in);
			defx[i][j] = def_in[0];
			defy[i][j] = def_in[1];
			lens_in->hessian(x,y,hess_in);
			hess_xx[i][j] = hess_in[0][0];
			hess_yy[i][j] = hess_in[1][1];
			hess_xy[i][j] = hess_in[0][1];
		}
	}
}

Tabulated_Model::Tabulated_Model(const Tabulated_Model* lens_in)
{
	kscale = lens_in->kscale;
	copy_base_lensdata(lens_in);
	grid_xlength = lens_in->grid_xlength;
	grid_ylength = lens_in->grid_ylength;

	grid_x_N = lens_in->grid_x_N;
	grid_y_N = lens_in->grid_y_N;
	grid_xvals = new double[grid_x_N];
	grid_yvals = new double[grid_y_N];

	kappa_vals = new double*[grid_x_N];
	pot_vals = new double*[grid_x_N];
	defx = new double*[grid_x_N];
	defy = new double*[grid_x_N];
	hess_xx = new double*[grid_x_N];
	hess_yy = new double*[grid_x_N];
	hess_xy = new double*[grid_x_N];
	int i,j;
	for (i=0; i < grid_x_N; i++) {
		kappa_vals[i] = new double[grid_y_N];
		pot_vals[i] = new double[grid_y_N];
		defx[i] = new double[grid_y_N];
		defy[i] = new double[grid_y_N];
		hess_xx[i] = new double[grid_y_N];
		hess_yy[i] = new double[grid_y_N];
		hess_xy[i] = new double[grid_y_N];
	}

	lensvector def_in;
	lensmatrix hess_in;
	double x,y;
	double xstep = grid_xlength/(grid_x_N-1);
	double ystep = grid_ylength/(grid_y_N-1);
	for (i=0; i < grid_x_N; i++) grid_xvals[i] = lens_in->grid_xvals[i];
	for (j=0; j < grid_y_N; j++) grid_yvals[j] = lens_in->grid_yvals[j];

	for (i=0; i < grid_x_N; i++) {
		for (j=0; j < grid_y_N; j++) {
			kappa_vals[i][j] = lens_in->kappa_vals[i][j];
			pot_vals[i][j] = lens_in->pot_vals[i][j];
			defx[i][j] = lens_in->defx[i][j];
			defy[i][j] = lens_in->defy[i][j];
			hess_xx[i][j] = lens_in->hess_xx[i][j];
			hess_yy[i][j] = lens_in->hess_yy[i][j];
			hess_xy[i][j] = lens_in->hess_xy[i][j];
		}
	}
}

void Tabulated_Model::assign_paramnames()
{
	paramnames[0] = "kscale"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "ext";
	paramnames[1] = "theta";  latex_paramnames[1] = "\\theta"; latex_param_subscripts[1] = "";
	paramnames[2] = "xc";     latex_paramnames[2] = "x";       latex_param_subscripts[2] = "c";
	paramnames[3] = "yc";     latex_paramnames[3] = "y";       latex_param_subscripts[3] = "c";
}

void Tabulated_Model::assign_param_pointers()
{
	param[0] = &kscale;
	param[1] = &theta; angle_paramnum = 1;
	param[2] = &x_center;
	param[3] = &y_center;
}

void Tabulated_Model::update_meta_parameters()
{
	// We don't use orient_major_axis_north because this is meaningless for the tabulated model
	costheta = cos(theta);
	sintheta = sin(theta);
}

void Tabulated_Model::set_auto_stepsizes()
{
	stepsizes[0] = 0.3*kscale;
	stepsizes[1] = 20;
	stepsizes[2] = x_center; // arbitrary! really, the center should never be independently varied
	stepsizes[3] = y_center;
}

void Tabulated_Model::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = false;
}

double Tabulated_Model::potential(double x, double y)
{
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	ival = (int) ((x - grid_xvals[0]) * grid_x_N / grid_xlength);
	jval = (int) ((y - grid_yvals[0]) * grid_y_N / grid_ylength);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_x_N-1) ival=grid_x_N-2;
	if (jval >= grid_y_N-1) jval=grid_y_N-2;
	tt = (x - grid_xvals[ival]) / (grid_xvals[ival+1] - grid_xvals[ival]);
	uu = (y - grid_yvals[jval]) / (grid_yvals[jval+1] - grid_yvals[jval]);
	TT = 1-tt;
	UU = 1-uu;
	interp = TT*UU*pot_vals[ival][jval] + tt*UU*pot_vals[ival+1][jval]
						+ TT*uu*pot_vals[ival][jval+1] + tt*uu*pot_vals[ival+1][jval+1];
	return kscale*interp;
}

double Tabulated_Model::kappa(double x, double y)
{
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	ival = (int) ((x - grid_xvals[0]) * grid_x_N / grid_xlength);
	jval = (int) ((y - grid_yvals[0]) * grid_y_N / grid_ylength);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_x_N-1) ival=grid_x_N-2;
	if (jval >= grid_y_N-1) jval=grid_y_N-2;
	tt = (x - grid_xvals[ival]) / (grid_xvals[ival+1] - grid_xvals[ival]);
	uu = (y - grid_yvals[jval]) / (grid_yvals[jval+1] - grid_yvals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	//cout << ival << " " << jval << " " << tt << " " << uu << endl;
	//cout << kappa_vals[ival][jval] << endl;
	interp = TT*UU*kappa_vals[ival][jval] + tt*UU*kappa_vals[ival+1][jval]
						+ TT*uu*kappa_vals[ival][jval+1] + tt*uu*kappa_vals[ival+1][jval+1];
	return kscale*interp;
}

void Tabulated_Model::deflection(double x, double y, lensvector& def)
{
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	ival = (int) ((x - grid_xvals[0]) * grid_x_N / grid_xlength);
	jval = (int) ((y - grid_yvals[0]) * grid_y_N / grid_ylength);
	//cout << x << " " << grid_xvals[0] << " " << grid_xlength << " " << grid_x_N << endl;
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_x_N-1) ival=grid_x_N-2;
	if (jval >= grid_y_N-1) jval=grid_y_N-2;
	tt = (x - grid_xvals[ival]) / (grid_xvals[ival+1] - grid_xvals[ival]);
	uu = (y - grid_yvals[jval]) / (grid_yvals[jval+1] - grid_yvals[jval]);
	TT = 1-tt;
	UU = 1-uu;
	interp = TT*UU*defx[ival][jval] + tt*UU*defx[ival+1][jval] + TT*uu*defx[ival][jval+1] + tt*uu*defx[ival+1][jval+1];
	//cout << ival << " " << jval << " " << uu << " " << tt << endl;
	def[0] = kscale*interp;
	interp = TT*UU*defy[ival][jval] + tt*UU*defy[ival+1][jval] + TT*uu*defy[ival][jval+1] + tt*uu*defy[ival+1][jval+1];
	def[1] = kscale*interp;
	if (sintheta != 0) def.rotate_back(costheta,sintheta);
}

void Tabulated_Model::hessian(double x, double y, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);
	ival = (int) ((x - grid_xvals[0]) * grid_x_N / grid_xlength);
	jval = (int) ((y - grid_yvals[0]) * grid_y_N / grid_ylength);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_x_N-1) ival=grid_x_N-2;
	if (jval >= grid_y_N-1) jval=grid_y_N-2;
	tt = (x - grid_xvals[ival]) / (grid_xvals[ival+1] - grid_xvals[ival]);
	uu = (y - grid_yvals[jval]) / (grid_yvals[jval+1] - grid_yvals[jval]);
	TT = 1-tt;
	UU = 1-uu;
	interp = TT*UU*hess_xx[ival][jval] + tt*UU*hess_xx[ival+1][jval]
						+ TT*uu*hess_xx[ival][jval+1] + tt*uu*hess_xx[ival+1][jval+1];
	hess[0][0] = kscale*interp;
	interp = TT*UU*hess_yy[ival][jval] + tt*UU*hess_yy[ival+1][jval]
						+ TT*uu*hess_yy[ival][jval+1] + tt*uu*hess_yy[ival+1][jval+1];
	hess[1][1] = kscale*interp;
	interp = TT*UU*hess_xy[ival][jval] + tt*UU*hess_xy[ival+1][jval]
						+ TT*uu*hess_xy[ival][jval+1] + tt*uu*hess_xy[ival+1][jval+1];
	hess[0][1] = kscale*interp;
	hess[1][0] = hess[0][1];
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

void Tabulated_Model::kappa_and_potential_derivatives(double x, double y, double& kap, lensvector& def, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);

	if (sintheta != 0) rotate(x,y);
	ival = (int) ((x - grid_xvals[0]) * grid_x_N / grid_xlength);
	jval = (int) ((y - grid_yvals[0]) * grid_y_N / grid_ylength);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_x_N-1) ival=grid_x_N-2;
	if (jval >= grid_y_N-1) jval=grid_y_N-2;
	tt = (x - grid_xvals[ival]) / (grid_xvals[ival+1] - grid_xvals[ival]);
	uu = (y - grid_yvals[jval]) / (grid_yvals[jval+1] - grid_yvals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	interp = TT*UU*kappa_vals[ival][jval] + tt*UU*kappa_vals[ival+1][jval]
						+ TT*uu*kappa_vals[ival][jval+1] + tt*uu*kappa_vals[ival+1][jval+1];
	kap = kscale*interp;

	interp = TT*UU*defx[ival][jval] + tt*UU*defx[ival+1][jval] + TT*uu*defx[ival][jval+1] + tt*uu*defx[ival+1][jval+1];
	//cout << ival << " " << jval << " " << uu << " " << tt << endl;
	def[0] = kscale*interp;
	interp = TT*UU*defy[ival][jval] + tt*UU*defy[ival+1][jval] + TT*uu*defy[ival][jval+1] + tt*uu*defy[ival+1][jval+1];
	def[1] = kscale*interp;

	interp = TT*UU*hess_xx[ival][jval] + tt*UU*hess_xx[ival+1][jval]
						+ TT*uu*hess_xx[ival][jval+1] + tt*uu*hess_xx[ival+1][jval+1];
	hess[0][0] = kscale*interp;
	interp = TT*UU*hess_yy[ival][jval] + tt*UU*hess_yy[ival+1][jval]
						+ TT*uu*hess_yy[ival][jval+1] + tt*uu*hess_yy[ival+1][jval+1];
	hess[1][1] = kscale*interp;
	interp = TT*UU*hess_xy[ival][jval] + tt*UU*hess_xy[ival+1][jval]
						+ TT*uu*hess_xy[ival][jval+1] + tt*uu*hess_xy[ival+1][jval+1];
	hess[0][1] = kscale*interp;
	hess[1][0] = hess[0][1];

	if (sintheta != 0) def.rotate_back(costheta,sintheta);
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

void Tabulated_Model::potential_derivatives(double x, double y, lensvector& def, lensmatrix& hess)
{
	x -= x_center;
	y -= y_center;
	if (sintheta != 0) rotate(x,y);

	if (sintheta != 0) rotate(x,y);
	ival = (int) ((x - grid_xvals[0]) * grid_x_N / grid_xlength);
	jval = (int) ((y - grid_yvals[0]) * grid_y_N / grid_ylength);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_x_N-1) ival=grid_x_N-2;
	if (jval >= grid_y_N-1) jval=grid_y_N-2;
	tt = (x - grid_xvals[ival]) / (grid_xvals[ival+1] - grid_xvals[ival]);
	uu = (y - grid_yvals[jval]) / (grid_yvals[jval+1] - grid_yvals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	interp = TT*UU*defx[ival][jval] + tt*UU*defx[ival+1][jval] + TT*uu*defx[ival][jval+1] + tt*uu*defx[ival+1][jval+1];
	//cout << ival << " " << jval << " " << uu << " " << tt << endl;
	def[0] = kscale*interp;
	interp = TT*UU*defy[ival][jval] + tt*UU*defy[ival+1][jval] + TT*uu*defy[ival][jval+1] + tt*uu*defy[ival+1][jval+1];
	def[1] = kscale*interp;

	interp = TT*UU*hess_xx[ival][jval] + tt*UU*hess_xx[ival+1][jval]
						+ TT*uu*hess_xx[ival][jval+1] + tt*uu*hess_xx[ival+1][jval+1];
	hess[0][0] = kscale*interp;
	interp = TT*UU*hess_yy[ival][jval] + tt*UU*hess_yy[ival+1][jval]
						+ TT*uu*hess_yy[ival][jval+1] + tt*uu*hess_yy[ival+1][jval+1];
	hess[1][1] = kscale*interp;
	interp = TT*UU*hess_xy[ival][jval] + tt*UU*hess_xy[ival+1][jval]
						+ TT*uu*hess_xy[ival][jval+1] + tt*uu*hess_xy[ival+1][jval+1];
	hess[0][1] = kscale*interp;
	hess[1][0] = hess[0][1];

	if (sintheta != 0) def.rotate_back(costheta,sintheta);
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

Tabulated_Model::~Tabulated_Model() {
	if (grid_xvals != NULL) {
		delete[] grid_xvals;
		delete[] grid_yvals;
		for (int i=0; i < grid_x_N; i++) {
			delete[] kappa_vals[i];
			delete[] pot_vals[i];
			delete[] defx[i];
			delete[] defy[i];
			delete[] hess_xx[i];
			delete[] hess_yy[i];
			delete[] hess_xy[i];
		}
		delete[] kappa_vals;
		delete[] pot_vals;
		delete[] defx;
		delete[] defy;
		delete[] hess_xx;
		delete[] hess_yy;
		delete[] hess_xy;
	}
	if (param != NULL) delete[] param;
	if (anchor_parameter != NULL) delete[] anchor_parameter;
	if (parameter_anchor_lens != NULL) delete[] parameter_anchor_lens;
	if (parameter_anchor_paramnum != NULL) delete[] parameter_anchor_paramnum;
	if (parameter_anchor_ratio != NULL) delete[] parameter_anchor_ratio;
}



/***************************** Test Model (for testing purposes only) *****************************/

TestModel::TestModel(const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc)
{
	lenstype = TESTMODEL;
	model_name = "test";
	special_parameter_command = "";
	//setup_base_lens(X,false); // number of parameters = X, is_elliptical_lens = false
	set_default_base_settings(nn,acc);
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

