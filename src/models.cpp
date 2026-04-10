#include <cmath>
#include <iostream>
#include <sstream>
#include "profile.h"
#include "sbprofile.h"
#include "mathexpr.h"
#include "errors.h"
#include "mathfuncs.h"
#include "hyp_2F1.h"
#include "qlens.h"
#include <complex>
using namespace std;

/*************************** Softened power law model (alpha) *****************************/

//SPLE_Lens::SPLE_Lens(const double zlens_in, const double zsrc_in, const double &bb, const double &slope, const double &ss, const double &q_in, const double &theta_degrees,
		//const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
//{
	//setup_lens_properties(parameter_mode_in);
	//setup_cosmology(cosmo_in,zlens_in,zsrc_in);
	//initialize_parameters(bb,slope,ss,q_in,theta_degrees,xc_in,yc_in);
//}

SPLE_Lens::SPLE_Lens(const double zlens_in, const double zsrc_in, const double &bb, const double &slope, const double &ss, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_sple;
	setup_lens_properties(parameter_mode_in);
	this->set_redshifts(zlens_in,zsrc_in);
	this->setup_cosmology(cosmo_in);
	this->initialize_parameters(bb,slope,ss,q_in,theta_degrees,xc_in,yc_in);
}

void SPLE_Lens::setup_lens_properties(const int parameter_mode_in, const int subclass)
{
	lenstype = sple_LENS;
	model_name = "sple";
	this->setup_base_lens_properties(8,3,true,parameter_mode_in); // number of parameters = 7, is_elliptical_lens = true
	analytic_3d_density = true;
}

void SPLE_Lens::initialize_parameters(const double &bb, const double &slope, const double &ss, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	this->set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	lensparams_sple.b = bb;
	lensparams_sple.s = ss;
	if (parameter_mode==0) {
		lensparams_sple.alpha = slope;
	} else {
		lensparams_sple.gamma = slope;
	}
	if (lensparams_sple.s < 0) lensparams_sple.s = -lensparams_sple.s; // don't allow negative core radii

	this->update_meta_parameters_and_pointers();
}

SPLE_Lens::SPLE_Lens(const SPLE_Lens* lens_in)
{
	lensparams = &lensparams_sple;
	this->copy_base_lensdata(lens_in);
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	//SPLE_Params<double>& p_in = lens_in->assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.b = lens_in->lensparams_sple.b;
	p.alpha = lens_in->lensparams_sple.alpha;
	if (parameter_mode==1) p.gamma = lens_in->lensparams_sple.gamma;
	p.s = lens_in->lensparams_sple.s;

	this->update_meta_parameters_and_pointers();
}

SPLE_Lens::SPLE_Lens(SPLE* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper)
{
	lensparams = &lensparams_sple;
	this->setup_lens_properties(parameter_mode_in);
	this->copy_source_data_to_lens(sb_in);
	lensparams_sple.b = sb_in->bs;
	lensparams_sple.alpha = sb_in->alpha;
	lensparams_sple.s = sb_in->s;
	this->set_spawned_mass_and_anchor_parameters((SB_Profile*) sb_in, vary_mass_parameter, include_limits_in, mass_param_lower,mass_param_upper);
}

void SPLE_Lens::assign_paramnames()
{
	paramnames[0] = "b";     latex_paramnames[0] = "b";       latex_param_subscripts[0] = "";
	if (parameter_mode==0) {
		paramnames[1] = "alpha"; latex_paramnames[1] = "\\alpha"; latex_param_subscripts[1] = "";
	} else {
		paramnames[1] = "gamma"; latex_paramnames[1] = "\\gamma"; latex_param_subscripts[1] = "";
	}
	paramnames[2] = "s";     latex_paramnames[2] = "s";       latex_param_subscripts[2] = "";
	this->set_geometric_paramnames(lensprofile_nparams);
}

void SPLE_Lens::assign_param_pointers()
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.param[0] = &p.b;
	if (parameter_mode==0) {
		p.param[1] = &p.alpha;
	} else {
		p.param[1] = &p.gamma;
	}
	p.param[2] = &p.s;
	this->set_geometric_param_pointers(lensprofile_nparams);
}

void SPLE_Lens::update_meta_parameters()
{
	this->update_cosmology_meta_parameters();
	this->update_ellipticity_meta_parameters();
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// these meta-parameters are used in analytic formulas for deflection, potential, etc.
	p.bprime = p.b*f_major_axis;
	p.sprime = p.s*f_major_axis;
	p.qsq = p.q*p.q; p.ssq_prime = p.sprime*p.sprime;
	if (parameter_mode==1) p.alpha = p.gamma-1;
}

void SPLE_Lens::set_auto_stepsizes()
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	stepsizes[index++] = 0.1*p.b;
	stepsizes[index++] = 0.1;
	stepsizes[index++] = 0.02*p.b; // this one is a bit arbitrary, but hopefully reasonable enough
	this->set_geometric_param_auto_stepsizes(index);
}

void SPLE_Lens::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	if (parameter_mode==0) {
		set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 2; // for 2D log-slope alpha
	} else {
		set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 1; penalty_upper_limits[1] = 3; // for 3D log-slope gamma
	}
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	this->set_geometric_param_auto_ranges(lensprofile_nparams);
}

void SPLE_Lens::set_model_specific_integration_pointers()
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// Here, we direct the integration pointers to analytic formulas in special cases where analytic solutions are possible
	this->kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&SPLE_Lens::kapavg_spherical_rsq);
	this->potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&SPLE_Lens::potential_spherical_rsq);
	if (!this->ellipticity_gradient) {
		if (p.alpha==1.0) {
			this->kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&SPLE_Lens::kapavg_spherical_rsq_iso);
			this->potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&SPLE_Lens::potential_spherical_rsq_iso);
			if (p.q != 1.0) {
				this->defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector<double>&)> (&SPLE_Lens::deflection_elliptical_iso);
				this->hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix<double>&)> (&SPLE_Lens::hessian_elliptical_iso);
				this->potptr = static_cast<double (LensProfile::*)(const double,const double)> (&SPLE_Lens::potential_elliptical_iso);
			}
		} else if (p.s==0.0) {
			this->potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&SPLE_Lens::potential_spherical_rsq_nocore);
			if (p.q != 1.0) {
				this->defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector<double>&)> (&SPLE_Lens::deflection_elliptical_nocore);
				this->hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix<double>&)> (&SPLE_Lens::hessian_elliptical_nocore);
				this->potptr = static_cast<double (LensProfile::*)(const double,const double)> (&SPLE_Lens::potential_elliptical_nocore);
				this->def_and_hess_ptr = static_cast<void (LensProfile::*)(const double,const double,lensvector<double>&,lensmatrix<double>&)> (&SPLE_Lens::deflection_and_hessian_elliptical_nocore);
			}
		}
	}
}

template <typename QScalar>
QScalar SPLE_Lens::kappa_rsq_impl(const QScalar rsq)
{
	SPLE_Params<QScalar>& p = assign_sple_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return ((2-p.alpha) * pow(p.b*p.b/(p.s*p.s+rsq), p.alpha/2) / 2);
}
template double SPLE_Lens::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var SPLE_Lens::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar SPLE_Lens::kappa_rsq_deriv_impl(const QScalar rsq)
{
	SPLE_Params<QScalar>& p = assign_sple_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar bsqr = p.b*p.b;
	return (-p.alpha * (2-p.alpha) * pow(bsqr/(p.s*p.s+rsq), p.alpha/2 + 1)) / (4*bsqr);
}
template double SPLE_Lens::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var SPLE_Lens::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

double SPLE_Lens::kapavg_spherical_rsq(const double rsq)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return (pow(p.b,p.alpha)*(pow(rsq+p.s*p.s,1-p.alpha/2) - pow(p.s,2-p.alpha)))/rsq;
}

double SPLE_Lens::kapavg_spherical_rsq_iso(const double rsq) // only for alpha=1
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.b*(sqrt(p.s*p.s+rsq)-p.s)/rsq; // now, tmp = kappa_average
}

double SPLE_Lens::potential_spherical_rsq(const double rsq)
{
	//using std::real;
	//using stan::math::real;
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// Formula from Keeton (2002), w/ typo corrected (sign in front of the DiGamma() term)
	double bpow, bs, ssq, pp, tmp;
	bpow = pow(p.b,p.alpha);
	bs = bpow*pow(p.s,2-p.alpha);
	pp = p.alpha/2-1;
	ssq = p.s*p.s;
	tmp = bpow*pow(rsq,-pp)*real(hyp_2F1<double>(pp,pp,1+pp,-ssq/rsq))/(2-p.alpha);
	/*
	stan::math::var pps = p.alpha/2-1;
	stan::math::var herg = -ssq/rsq;
	stan::math::var hyp = real(hyp_2F1<stan::math::var>(pps,pps,1+pps,herg));
	hyp.grad();
    cout << "hyp is: " << hyp.val() << endl;
    cout << "Derivative of hyp with respect to pps is: " << pps.adj() << endl; 
    cout << "Derivative of hyp with respect to herg is: " << herg.adj() << endl; 
	pps += 1e-5;
	stan::math::var hyp2 = real(hyp_2F1<stan::math::var>(pps,pps,1+pps,herg));
	stan::math::var hypder = (hyp2-hyp)/1e-5;
	cout << "finite dif deriv = " << hypder.val() << endl;
	*/

	tmp += -bs*log(rsq/(ssq))/2 - bs*(euler_mascheroni + DiGamma(pp))/2;
	return tmp;
}

double SPLE_Lens::potential_spherical_rsq_iso(const double rsq) // only for alpha=1
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double tmp, sqrtterm;
	sqrtterm = sqrt(p.s*p.s+rsq);
	tmp = p.b*(sqrtterm-p.s); // now, tmp = kappa_average*rsq
	if (p.s != 0) tmp -= p.b*p.s*log((p.s + sqrtterm)/(2.0*p.s));
	return tmp;
}

double SPLE_Lens::potential_spherical_rsq_nocore(const double rsq)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return pow(p.b*p.b/rsq,p.alpha/2)*rsq/(2-p.alpha);
}

//  Note: although the elliptical formulas are expressed in terms of ellipticity mode 0, they use parameters
//  (the prime versions b', a', etc.) transformed from the correct emode

void SPLE_Lens::deflection_elliptical_iso(const double x, const double y, lensvector<double>& def) // only for alpha=1
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double u, psi;
	psi = sqrt(p.qsq*(p.ssq_prime+x*x)+y*y);
	u = sqrt(1-p.qsq);

	def[0] = (p.bprime*p.q/u)*atan(u*x/(psi+p.sprime));
	def[1] = (p.bprime*p.q/u)*atanh(u*y/(psi+p.qsq*p.sprime));
}

void SPLE_Lens::hessian_elliptical_iso(const double x, const double y, lensmatrix<double>& hess) // only for alpha=1
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double xsq, ysq, psi, tmp;
	xsq=x*x; ysq=y*y;

	psi = sqrt(p.qsq*(p.ssq_prime+xsq)+ysq);
	tmp = ((p.bprime*p.q)/psi)/(xsq+ysq+2*psi*p.sprime+p.ssq_prime*(1+p.qsq));

	hess[0][0] = tmp*(ysq+p.sprime*psi+p.ssq_prime*p.qsq);
	hess[1][1] = tmp*(xsq+p.sprime*psi+p.ssq_prime);
	hess[0][1] = -tmp*x*y;
	hess[1][0] = hess[0][1];
}

double SPLE_Lens::potential_elliptical_iso(const double x, const double y) // only for alpha=1
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double u, tmp, psi;
	psi = sqrt(p.qsq*(p.ssq_prime+x*x)+y*y);
	u = sqrt(1-p.qsq);

	tmp = (p.bprime*p.q/u)*(x*atan(u*x/(psi+p.sprime)) + y*atanh(u*y/(psi+p.qsq*p.sprime)));
	if (p.sprime != 0) tmp += p.bprime*p.q*p.sprime*(-log(SQR(psi+p.sprime) + SQR(u*x))/2 + log((1.0+p.q)*p.sprime));

	return tmp;
}

void SPLE_Lens::deflection_elliptical_nocore(const double x, const double y, lensvector<double>& def)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// Formulas from Tessore et al. 2015
	double phi, R = sqrt(x*x+y*y/p.qsq);
	phi = atan(abs(y/(p.q*x)));

	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	complex<double> def_complex = 2*p.bprime*p.q/(1+p.q)*pow(p.bprime/R,p.alpha-1)*deflection_angular_factor(phi);

	def[0] = real(def_complex);
	def[1] = imag(def_complex);
}

void SPLE_Lens::hessian_elliptical_nocore(const double x, const double y, lensmatrix<double>& hess)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double R, phi, kap;
	R = sqrt(x*x+y*y/p.qsq);
	kap = 0.5 * (2-p.alpha) * pow(p.bprime/R, p.alpha);
	phi = atan(abs(y/(p.q*x)));
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
	hess_complex = 2*p.bprime*p.q/(1+p.q)*pow(p.bprime/R,p.alpha-1)*deflection_angular_factor(phi);
	hess_complex = -kap*conj(zstar)/zstar + (1-p.alpha)*hess_complex/zstar; // this is the complex shear

	hess_complex = kap + hess_complex; // this is now (kappa+shear)
	hess[0][0] = real(hess_complex);
	hess[0][1] = imag(hess_complex);
	hess[1][0] = hess[0][1];
	hess_complex = 2*kap - hess_complex; // now we have transformed to (kappa-shear)
	hess[1][1] = real(hess_complex);
}

void SPLE_Lens::deflection_and_hessian_elliptical_nocore(const double x, const double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double R, phi, kap;
	R = sqrt(x*x+y*y/p.qsq);
	kap = 0.5 * (2-p.alpha) * pow(p.bprime/R, p.alpha);
	phi = atan(abs(y/(p.q*x)));
	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	complex<double> def_complex = 2*p.bprime*p.q/(1+p.q)*pow(p.bprime/R,p.alpha-1)*deflection_angular_factor(phi);
	def[0] = real(def_complex);
	def[1] = imag(def_complex);
	complex<double> hess_complex, zstar(x,-y);
	hess_complex = -kap*conj(zstar)/zstar + (1-p.alpha)*def_complex/zstar; // this is the complex shear
	hess_complex = kap + hess_complex; // this is now (kappa+shear)
	hess[0][0] = real(hess_complex);
	hess[0][1] = imag(hess_complex);
	hess[1][0] = hess[0][1];
	hess_complex = 2*kap - hess_complex; // now we have transformed to (kappa-shear)
	hess[1][1] = real(hess_complex);
}

double SPLE_Lens::potential_elliptical_nocore(const double x, const double y) // only for sprime=0
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double phi, R = sqrt(x*x+y*y/(p.q*p.q));
	phi = atan(abs(y/(p.q*x)));

	if (x < 0) {
		if (y < 0)
			phi = phi - M_PI;
		else
			phi = M_PI - phi;
	} else if (y < 0) {
		phi = -phi;
	}
	complex<double> def_complex = 2*p.bprime*p.q/(1+p.q)*pow(p.bprime/R,p.alpha-1)*deflection_angular_factor(phi);
	return (x*real(def_complex) + y*imag(def_complex))/(2-p.alpha);
}

complex<double> SPLE_Lens::deflection_angular_factor(const double &phi)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// Formulas from Tessore et al. 2015
	double beta, ff;
	beta = 2.0/(2-p.alpha);
	ff = (1-p.q)/(1+p.q);
	complex<double> fac = polar(1.0,phi);
	complex<double> omega = fac;
	int i=1;
	do {
		omega = -polar(ff*(beta*i - 1)/(beta*i + 1),2*phi)*omega;
		fac += omega;
		i++;
	} while (norm(omega) > def_tolerance*norm(fac));
	return fac;
}

void SPLE_Lens::get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (p.s==0.0) {
		re_major_axis = p.bprime*pow(zfactor,1.0/p.alpha);
		re_average = re_major_axis * sqrt(p.q);
	} else if (p.alpha==1.0) {
		if (p.sprime < p.bprime/2.0) {
			re_major_axis = p.bprime*sqrt(1-2*p.sprime/p.bprime/zfactor)*zfactor;
			re_average = re_major_axis * sqrt(p.q);
		} else {
			re_major_axis = 0;
			re_average = 0;
		}
	} else {
		rmin_einstein_radius = 0.01*p.b;
		rmax_einstein_radius = 100*p.b;
		LensProfile::get_einstein_radius(re_major_axis,re_average,zfactor);
	}
}

double SPLE_Lens::calculate_scaled_mass_3d(const double r)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (p.s==0.0) {
		double a2, B;
		a2 = (1+p.alpha)/2;
		B = (1.5-a2)*pow(p.b,p.alpha)*Gamma(a2)/(M_SQRT_PI*Gamma(p.alpha/2));
		return 4*M_PI*B*pow(r,2-p.alpha)/(2-p.alpha);
	} else {
		return this->calculate_scaled_mass_3d_from_analytic_rho3d(r);
	}
}

double SPLE_Lens::rho3d_r_integrand_analytic(const double r)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rsq, a2, B;
	rsq = r*r;
	a2 = (1+p.alpha)/2;
	B = (1.5-a2)*pow(p.b,p.alpha)*Gamma(a2)/(M_SQRT_PI*Gamma(p.alpha/2));
	return B/pow(rsq+p.s*p.s,a2);
}

bool SPLE_Lens::output_cosmology_info(const int lens_number)
{
	SPLE_Params<double>& p = assign_sple_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (p.alpha != 1.0) return false;
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}
	double kpc_to_km = 3.086e16;
	double Rs_sun_km = 2.953; // Schwarzchild radius of the Sun in km
	double c = 2.998e5;
	double b_kpc, sigma, r_tidal, r_core, mtot, rhalf;
	b_kpc = p.b / kpc_to_arcsec;
	sigma = c * sqrt(b_kpc*(Rs_sun_km/kpc_to_km)*sigma_cr_kpc/2);
	cout << "sigma = " << sigma << " km/s  (velocity dispersion)\n";
	return true;
}

/********************************** dPIE_Lens **********************************/

dPIE_Lens::dPIE_Lens(const double zlens_in, const double zsrc_in, const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_dpie;
	this->setup_lens_properties(parameter_mode_in);
	this->set_redshifts(zlens_in,zsrc_in);
	this->setup_cosmology(cosmo_in);
	this->initialize_parameters(p1_in,p2_in,p3_in,q_in,theta_degrees,xc_in,yc_in);
}

void dPIE_Lens::initialize_parameters(const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	this->set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.b = p1_in;
		p.a = p2_in;
		p.s = p3_in;
	} else if (parameter_mode==1) {
		p.sigma0 = p1_in;
		p.a_kpc = p2_in;
		p.s_kpc = p3_in;
	} else {
		p.mtot = p1_in;
		p.a_kpc = p2_in;
		p.s_kpc = p3_in;
	}

	this->update_meta_parameters_and_pointers();
}

void dPIE_Lens::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = dpie_LENS;
	model_name = "dpie";
	this->setup_base_lens_properties(8,3,true,parameter_mode); // number of parameters = 7, is_elliptical_lens = true
	analytic_3d_density = true;
}

dPIE_Lens::dPIE_Lens(const dPIE_Lens* lens_in)
{
	lensparams = &lensparams_dpie;
	this->copy_base_lensdata(lens_in);
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.b = lens_in->lensparams_dpie.b;
	p.a = lens_in->lensparams_dpie.a;
	p.s = lens_in->lensparams_dpie.s;
	if (parameter_mode==1) {
		p.sigma0 = lens_in->lensparams_dpie.sigma0;
		p.a_kpc = lens_in->lensparams_dpie.a_kpc;
		p.s_kpc = lens_in->lensparams_dpie.s_kpc;
	} else if (parameter_mode==2) {
		p.mtot = lens_in->lensparams_dpie.mtot;
		p.a_kpc = lens_in->lensparams_dpie.a_kpc;
		p.s_kpc = lens_in->lensparams_dpie.s_kpc;
	}

	this->update_meta_parameters_and_pointers();
}

dPIE_Lens::dPIE_Lens(dPIE* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper)
{
	lensparams = &lensparams_dpie;
	this->setup_lens_properties(parameter_mode_in);
	this->copy_source_data_to_lens(sb_in);
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.b = sb_in->bs;
	p.a = sb_in->a;
	p.s = sb_in->s;
	this->set_spawned_mass_and_anchor_parameters((SB_Profile*) sb_in, vary_mass_parameter, include_limits_in, mass_param_lower,mass_param_upper);
}

void dPIE_Lens::assign_paramnames()
{
	if (parameter_mode==0) {
		paramnames[0] = "b"; latex_paramnames[0] = "b"; latex_param_subscripts[0] = "";
		paramnames[1] = "a"; latex_paramnames[1] = "a"; latex_param_subscripts[1] = "";
		paramnames[2] = "s"; latex_paramnames[2] = "s"; latex_param_subscripts[2] = "";
	} else if (parameter_mode==1) {
		paramnames[0] = "sigma0"; latex_paramnames[0] = "\\sigma"; latex_param_subscripts[0] = "0";
		paramnames[1] = "a_kpc"; latex_paramnames[1] = "a"; latex_param_subscripts[1] = "kpc";
		paramnames[2] = "s_kpc"; latex_paramnames[2] = "s"; latex_param_subscripts[2] = "kpc";
	} else {
		paramnames[0] = "mtot"; latex_paramnames[0] = "M"; latex_param_subscripts[0] = "tot";
		paramnames[1] = "a_kpc"; latex_paramnames[1] = "a"; latex_param_subscripts[1] = "kpc";
		paramnames[2] = "s_kpc"; latex_paramnames[2] = "s"; latex_param_subscripts[2] = "kpc";
	}
	this->set_geometric_paramnames(lensprofile_nparams);
}

void dPIE_Lens::assign_param_pointers()
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.param[0] = &p.b;
		p.param[1] = &p.a;
		p.param[2] = &p.s;
	} else if (parameter_mode==1) {
		p.param[0] = &p.sigma0;
		p.param[1] = &p.a_kpc;
		p.param[2] = &p.s_kpc;
	} else {
		p.param[0] = &p.mtot;
		p.param[1] = &p.a_kpc;
		p.param[2] = &p.s_kpc;
	}
	this->set_geometric_param_pointers(lensprofile_nparams);
}

void dPIE_Lens::update_meta_parameters()
{
	this->update_cosmology_meta_parameters();
	this->update_ellipticity_meta_parameters();
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (this->cosmo != NULL) {
		if (parameter_mode==1) set_abs_params_from_sigma0();
		else if (parameter_mode==2) set_abs_params_from_mtot();
	}
	p.bprime = p.b*f_major_axis;
	p.aprime = p.a*f_major_axis;
	p.sprime = p.s*f_major_axis;
	p.qsq = p.q*p.q; p.asq = p.aprime*p.aprime; p.ssq_prime = p.sprime*p.sprime;
}

void dPIE_Lens::assign_special_anchored_parameters(LensProfile *host_in, const double factor, const bool just_created)
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	anchor_special_parameter = true;
	special_anchor_lens = host_in;
	double rm, ravg;
	special_anchor_lens->get_einstein_radius(rm,ravg,1.0);
	if (parameter_mode==2) {
		p.a = pow(ravg*p.mtot/(M_PI*sigma_cr),0.3333333333333333);
		p.a_kpc = p.a/kpc_to_arcsec;
		p.b = p.mtot/(M_PI*sigma_cr*(p.a-p.s));
		set_abs_params_from_mtot();
	}
	else p.a = sqrt(ravg*p.b); // this is an approximate formula (a = sqrt(b*Re_halo)) and assumes the subhalo is found roughly near the Einstein radius of the halo
	if (parameter_mode==1) p.a_kpc = p.a/kpc_to_arcsec;
	this->update_meta_parameters();
}

void dPIE_Lens::update_special_anchored_params()
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (anchor_special_parameter) {
		double rm, ravg;
		special_anchor_lens->get_einstein_radius(rm,ravg,1.0);
		if (parameter_mode==2) {
			p.a = pow(ravg*p.mtot/(M_PI*sigma_cr),0.3333333333333333);
			p.a_kpc = p.a/kpc_to_arcsec;
			p.b = p.mtot/(M_PI*sigma_cr*(p.a-p.s));
			set_abs_params_from_mtot();
		}
		else p.a = sqrt(ravg*p.b); // this is an approximate formula (a = sqrt(b*Re_halo)) and assumes the subhalo is found roughly near the Einstein radius of the halo

		if (parameter_mode==1) p.a_kpc = p.a/kpc_to_arcsec;
		p.aprime = p.a/f_major_axis;
		p.asq = p.aprime*p.aprime;
	}
}

void dPIE_Lens::get_parameters_pmode(const int pmode, double* params)
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (pmode==2) {
		params[0] = p.mtot;
		params[1] = p.a_kpc;
		params[2] = p.s_kpc;
	} else if (pmode==1) {
		params[0] = p.sigma0;
		params[1] = p.a_kpc;
		params[2] = p.s_kpc;
	} else {
		params[0] = p.b;
		params[1] = p.a;
		params[2] = p.s;
	}
	for (int i=lensprofile_nparams; i < n_params; i++) {
		if (angle_param[i]) params[i] = radians_to_degrees(*(p.param[i]));
		else params[i] = *(p.param[i]);
	}
	if (lensed_center_coords) {
		params[n_params-3] = p.x_center;
		params[n_params-2] = p.y_center;
	}
	if (lensed_center_coords) {
		params[n_params-3] = p.x_center;
		params[n_params-2] = p.y_center;
	}
}

void dPIE_Lens::set_auto_stepsizes()
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	if (parameter_mode==0) {
		stepsizes[index++] = 0.2*p.b;
		stepsizes[index++] = 0.2*p.b;
		stepsizes[index++] = 0.02*p.b; // this one is a bit arbitrary, but hopefully reasonable enough
	} else if (parameter_mode==1) {
		stepsizes[index++] = 0.2*p.sigma0;
		stepsizes[index++] = 0.2*p.b/kpc_to_arcsec;
		stepsizes[index++] = 0.02*p.b/kpc_to_arcsec; // this one is a bit arbitrary, but hopefully reasonable enough
	} else {
		stepsizes[index++] = 0.2*p.mtot;
		stepsizes[index++] = 0.2*p.b/kpc_to_arcsec;
		stepsizes[index++] = 0.02*p.b/kpc_to_arcsec; // this one is a bit arbitrary, but hopefully reasonable enough
	}
	this->set_geometric_param_auto_stepsizes(index);
}

void dPIE_Lens::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	this->set_geometric_param_auto_ranges(lensprofile_nparams);
}

void dPIE_Lens::set_model_specific_integration_pointers()
{
	this->kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&dPIE_Lens::kapavg_spherical_rsq);
	this->potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&dPIE_Lens::potential_spherical_rsq);
	if (!ellipticity_gradient) {
		if (lensparams->q != 1.0) {
			this->defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector<double>&)> (&dPIE_Lens::deflection_elliptical);
			this->hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix<double>&)> (&dPIE_Lens::hessian_elliptical);
			this->potptr = static_cast<double (LensProfile::*)(const double,const double)> (&dPIE_Lens::potential_elliptical);
		}
	}
}

template <typename QScalar>
QScalar dPIE_Lens::kappa_rsq_impl(const QScalar rsq)
{
	dPIE_Params<QScalar>& p = assign_dpie_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return (0.5 * p.b * (pow(p.s*p.s+rsq, -0.5) - pow(p.a*p.a+rsq,-0.5)));
}
template double dPIE_Lens::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var dPIE_Lens::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar dPIE_Lens::kappa_rsq_deriv_impl(const QScalar rsq)
{
	dPIE_Params<QScalar>& p = assign_dpie_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return (-0.25 * p.b * (pow(p.s*p.s+rsq, -1.5) - pow(p.a*p.a+rsq,-1.5)));
}
template double dPIE_Lens::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var dPIE_Lens::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

double dPIE_Lens::kapavg_spherical_rsq(const double rsq)
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.b*((sqrt(p.s*p.s+rsq)-p.s) - (sqrt(p.a*p.a+rsq)-p.a))/rsq;
}

double dPIE_Lens::potential_spherical_rsq(const double rsq)
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double tmp, asq;
	// might need to have a first order expansion for small s values
	asq = p.a*p.a;
	tmp = p.b*(sqrt(p.s*p.s+rsq) - p.s - sqrt(asq+rsq) + p.a + p.a*log((p.a + sqrt(asq+rsq))/(2.0*p.a)));
	if (p.s != 0.0) tmp -= p.s*log((p.s + sqrt(p.s*p.s+rsq))/(2.0*p.s));
	return tmp;
}

//  Note: although the elliptical formulas are expressed in terms of ellipticity mode 0, they use parameters
//  (the prime versions b', a', etc.) transformed from the correct emode

void dPIE_Lens::deflection_elliptical(const double x, const double y, lensvector<double>& def)
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double psi, psi2, u;
	psi = sqrt(p.qsq*(p.ssq_prime+x*x)+y*y);
	psi2 = sqrt(p.qsq*(p.asq+x*x)+y*y);
	u = sqrt(1-p.qsq);

	def[0] = (p.bprime*p.q/u)*(atan(u*x/(psi+p.sprime)) - atan(u*x/(psi2+p.aprime)));
	def[1] = (p.bprime*p.q/u)*(atanh(u*y/(psi+p.qsq*p.sprime)) - atanh(u*y/(psi2+p.qsq*p.aprime)));
}

void dPIE_Lens::hessian_elliptical(const double x, const double y, lensmatrix<double>& hess)
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double xsq, ysq, psi, tmp1, psi2, tmp2;
	xsq=x*x; ysq=y*y;
	psi = sqrt(p.qsq*(p.ssq_prime+xsq)+ysq);
	tmp1 = (p.bprime*p.q/psi)/(xsq+ysq+2*psi*p.sprime+p.ssq_prime*(1+p.qsq));

	psi2 = sqrt(p.qsq*(p.asq+xsq)+ysq);
	tmp2 = (p.bprime*p.q/psi2)/(xsq+ysq+2*psi2*p.aprime+p.asq*(1+p.qsq));

	hess[0][0] = tmp1*(ysq+p.sprime*psi+p.ssq_prime*p.qsq) - tmp2*(ysq+p.aprime*psi2+p.asq*p.qsq);
	hess[1][1] = tmp1*(xsq+p.sprime*psi+p.ssq_prime) - tmp2*(xsq+p.aprime*psi2+p.asq);
	hess[0][1] = (-tmp1+tmp2)*x*y;
	hess[1][0] = hess[0][1];
}

double dPIE_Lens::potential_elliptical(const double x, const double y)
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double psi, psi2, u;
	psi = sqrt(p.qsq*(p.ssq_prime+x*x)+y*y);
	psi2 = sqrt(p.qsq*(p.asq+x*x)+y*y);
	u = sqrt(1-p.qsq);

	// might need to have p.a first order expansion for small p.s values
	double ans = (p.bprime*p.q/u)*(x*(atan(u*x/(psi+p.sprime)) - atan(u*x/(psi2+p.aprime)))+ y*(atanh(u*y/(psi+p.qsq*p.sprime))
		- atanh(u*y/(psi2+p.qsq*p.aprime)))) + p.bprime*p.q*(-p.aprime*(-log(SQR(psi2+p.aprime) + SQR(u*x))/2 + log((1.0+p.q)*p.aprime)));
	if (p.sprime != 0) ans += p.bprime*p.q*p.sprime*(-log(SQR(psi+p.sprime) + SQR(u*x))/2 + log((1.0+p.q)*p.sprime));
	return ans;
}

void dPIE_Lens::set_abs_params_from_sigma0()
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.b = 2.325092515e5*p.sigma0*p.sigma0/((1-p.s_kpc/p.a_kpc)*kpc_to_arcsec*sigma_cr);
	p.a = p.a_kpc * kpc_to_arcsec;
	p.s = p.s_kpc * kpc_to_arcsec;
}

void dPIE_Lens::set_abs_params_from_mtot()
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.a = p.a_kpc * kpc_to_arcsec;
	p.s = p.s_kpc * kpc_to_arcsec;
	p.b = p.mtot/(M_PI*sigma_cr*(p.a-p.s));
}

bool dPIE_Lens::output_cosmology_info(const int lens_number)
{
	dPIE_Params<double>& p = assign_dpie_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}
	double sigma, r_tidal, r_core, mtot, rhalf;
	sigma = 2.07386213e-3*sqrt(p.b*(1-p.s/p.a)*kpc_to_arcsec*sigma_cr); // this is = c*sqrt(b*(1-s/a)*D_s/D_ls/M_4PI), expressed in terms of kpc_to_arcsec and sigma_cr
	if ((parameter_mode==0) or (parameter_mode==2)) {
		cout << "sigma = " << sigma << " km/s  (velocity dispersion)\n";
	}
	if ((parameter_mode==1) or (parameter_mode==2)) {
		cout << "b = " << p.b << " arcsec" << endl;
	}
	calculate_total_scaled_mass(mtot);
	bool rhalf_converged = this->calculate_half_mass_radius(rhalf,mtot);
	mtot *= sigma_cr;
	cout << "total mass = " << mtot << " M_sun" << endl;
	if (rhalf_converged) cout << "half-mass radius: " << rhalf/kpc_to_arcsec << " kpc (" << rhalf << " arcsec)" << endl;

	r_tidal = p.a / kpc_to_arcsec;
	r_core = p.s / kpc_to_arcsec;
	cout << "r_tidal = " << r_tidal << " kpc" << endl;
	if (r_core > 0) cout << "r_core = " << r_core << " kpc" << endl;
	cout << endl;
	return true;
}

bool dPIE_Lens::calculate_total_scaled_mass(double& total_mass)
{
	total_mass = M_PI*lensparams_dpie.b*(lensparams_dpie.a-lensparams_dpie.s);
	return true;
}

double dPIE_Lens::calculate_scaled_mass_3d(const double r)
{
	double ans = lensparams_dpie.a*atan(r/lensparams_dpie.a);
	if (lensparams_dpie.s != 0.0) ans -= lensparams_dpie.s*atan(r/lensparams_dpie.s);
	return 2*lensparams_dpie.b*ans;
}

double dPIE_Lens::rho3d_r_integrand_analytic(const double r)
{
	double rsq = r*r;
	return (lensparams_dpie.b/M_2PI)*(SQR(lensparams_dpie.a)-SQR(lensparams_dpie.s))/(rsq+SQR(lensparams_dpie.a))/(rsq+SQR(lensparams_dpie.s));
}

/************************************* NFW *************************************/

NFW::NFW(const double zlens_in, const double zsrc_in, const double &p1_in, const double &p2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_nfw;
	setup_lens_properties(parameter_mode_in);
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(p1_in,p2_in,q_in,theta_degrees,xc_in,yc_in);
}

void NFW::initialize_parameters(const double &p1_in, const double &p2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==2) {
		p.m200 = p1_in;
		p.rs_kpc = p2_in;
	} else if (parameter_mode==1) {
		p.m200 = p1_in;
		p.c200 = p2_in;
	} else {
		p.ks = p1_in;
		p.rs = p2_in;
	}

	update_meta_parameters_and_pointers();
}

void NFW::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = nfw;
	model_name = "nfw";
	setup_base_lens_properties(7,2,true,parameter_mode); // number of parameters = 6, is_elliptical_lens = true
	analytic_3d_density = true;
}

NFW::NFW(const NFW* lens_in)
{
	lensparams = &lensparams_nfw;
	copy_base_lensdata(lens_in);
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.ks = lens_in->lensparams_nfw.ks;
	p.rs = lens_in->lensparams_nfw.rs;
	if (parameter_mode==2) {
		p.m200 = lens_in->lensparams_nfw.m200;
		p.rs_kpc = lens_in->lensparams_nfw.rs_kpc;
	} else if (parameter_mode==1) {
		p.m200 = lens_in->lensparams_nfw.m200;
		p.c200 = lens_in->lensparams_nfw.c200;
	}

	special_anchor_factor = lens_in->special_anchor_factor;
	update_meta_parameters_and_pointers();
}

NFW::NFW(NFW_Source* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper)
{
	lensparams = &lensparams_nfw;
	setup_lens_properties(parameter_mode_in);
	copy_source_data_to_lens(sb_in);
	lensparams_nfw.ks = sb_in->s0;
	lensparams_nfw.rs = sb_in->rs;
	set_spawned_mass_and_anchor_parameters((SB_Profile*) sb_in, vary_mass_parameter, include_limits_in, mass_param_lower,mass_param_upper);
}

void NFW::assign_paramnames()
{
	if (parameter_mode==2) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "rs_kpc"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	} else if (parameter_mode==1) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "c"; latex_paramnames[1] = "c"; latex_param_subscripts[1] = "";
	} else {
		paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
		paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	}
	set_geometric_paramnames(lensprofile_nparams);
}

void NFW::assign_param_pointers()
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==2) {
		p.param[0] = &p.m200;
		p.param[1] = &p.rs_kpc;
	} else if (parameter_mode==1) {
		p.param[0] = &p.m200;
		p.param[1] = &p.c200;
	} else {
		p.param[0] = &p.ks;
		p.param[1] = &p.rs;
	}
	set_geometric_param_pointers(lensprofile_nparams);
}

void NFW::get_parameters_pmode(const int pmode, double* params)
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		double ds, r200;
		double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
		p.rs_kpc = p.rs / kpc_to_arcsec;
		ds = p.ks * sigma_cr_kpc / p.rs_kpc;
		// Using a root-finder to solve for c, then p.m200 can be solved for
		cosmo->get_halo_parameters_from_rs_ds(zlens,p.rs_kpc,ds,p.m200,r200);
	}

	if (pmode==2) {
		params[0] = p.m200;
		params[1] = p.rs_kpc;
	} else if (pmode==1) {
		params[0] = p.m200;
		params[1] = p.c200;
	} else {
		params[0] = p.ks;
		params[1] = p.rs;
	}
	for (int i=lensprofile_nparams; i < n_params; i++) {
		if (angle_param[i]) params[i] = radians_to_degrees(*(p.param[i]));
		else params[i] = *(p.param[i]);
	}
	if (lensed_center_coords) {
		params[n_params-3] = p.x_center;
		params[n_params-2] = p.y_center;
	}
}

void NFW::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (cosmo != NULL) {
		if (parameter_mode==2) set_ks_c200_from_m200_rs();
		else if (parameter_mode==1) set_ks_rs_from_m200_c200();
		else {
			double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
			double ds, r200;
			if (parameter_mode != 2) p.rs_kpc = p.rs / kpc_to_arcsec;
			ds = p.ks * sigma_cr_kpc / p.rs_kpc;
			//cosmo->get_halo_parameters_from_rs_ds(zlens,p.rs_kpc,ds,p.m200,r200);
			//p.c200 = r200/p.rs_kpc;
		}
	}
	rmin_einstein_radius = 1e-6*p.rs; // for determining the Einstein radius (sets lower bound of root finder)
}

void NFW::assign_special_anchored_parameters(LensProfile *host_in, const double factor, const bool just_created)
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// the following special anchoring is to enforce a mass-concentration relation
	anchor_special_parameter = true;
	special_anchor_lens = this; // not actually used anyway, since we're not anchoring to another lens at all
	//p.c200 = factor*cosmo->median_concentration_bullock(p.m200,zlens);
	if (just_created) special_anchor_factor = factor;
	p.c200 = special_anchor_factor*cosmo->median_concentration_dutton(p.m200,zlens);
	update_meta_parameters();
}

void NFW::update_special_anchored_params()
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (anchor_special_parameter) {
		//p.c200 = cosmo->median_concentration_bullock(p.m200,zlens);
		p.c200 = special_anchor_factor * cosmo->median_concentration_dutton(p.m200,zlens);
		update_meta_parameters();
	}
}

void NFW::set_auto_stepsizes()
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	if (parameter_mode==2) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.rs_kpc;
	} else if (parameter_mode==1) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.c200;
	} else {
		stepsizes[index++] = 0.2*p.ks;
		stepsizes[index++] = 0.2*p.rs;
	}
	set_geometric_param_auto_stepsizes(index);
}

void NFW::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void NFW::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::kapavg_spherical_rsq);
	potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&NFW::potential_spherical_rsq);
}

void NFW::set_ks_c200_from_m200_rs()
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rvir_kpc;
	rvir_kpc = pow(p.m200/(200.0*M_4PI/3.0*1e-9*cosmo->critical_density(zlens)),0.333333333333);
	p.rs = p.rs_kpc * kpc_to_arcsec;
	p.c200 = rvir_kpc / p.rs_kpc;
	p.ks = p.m200 / (M_4PI*p.rs*p.rs*sigma_cr*(log(1+p.c200) - p.c200/(1+p.c200)));
}

void NFW::set_ks_rs_from_m200_c200()
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rvir_kpc;
	rvir_kpc = pow(p.m200/(200.0*M_4PI/3.0*1e-9*cosmo->critical_density(zlens)),0.333333333333);
	p.rs_kpc = rvir_kpc / p.c200;
	p.rs = p.rs_kpc * kpc_to_arcsec;
	p.ks = p.m200 / (M_4PI*p.rs*p.rs*sigma_cr*(log(1+p.c200) - p.c200/(1+p.c200)));
	//cout << "NFW: dcrit=" << cosmo->critical_density(zlens) << " lenfac=" << kpc_to_arcsec << " p.rs_kpc=" << p.rs_kpc << " p.rs=" << p.rs << " p.ks=" << p.ks << " p.c200=" << p.c200 << " p.m200=" << p.m200 << endl;
}

template <typename QScalar>
QScalar NFW::kappa_rsq_impl(const QScalar rsq)
{
	NFW_Params<QScalar>& p = assign_nfw_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar xsq = rsq/(p.rs*p.rs);
	if (xsq < 1e-6) return -p.ks*(2+log(xsq/4));
	else if (abs(xsq-1) < 1e-5) return 2*p.ks*(0.3333333333333333 - (xsq-1)/5.0); // formula on next line becomes unstable for x close to 1, this fixes it
	else return 2*p.ks*(1 - lens_function_xsq(xsq))/(xsq - 1);
}
template double NFW::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var NFW::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar NFW::kappa_rsq_deriv_impl(const QScalar rsq)
{
	NFW_Params<QScalar>& p = assign_nfw_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar rssq, xsq;
	rssq = p.rs*p.rs;
	xsq = rsq/rssq;
	// below xsq ~ 1e-6 or so, kappa formula becomes inaccurate due to fine cancellations; a series expansion
	// is done for xsq smaller than this
	if (xsq < 1e-6) return -p.ks/rsq;
	else if (abs(xsq-1.0) < 1e-5) return 2*p.ks/rssq*(-0.2 + 2*(xsq-1)/7.0); // formula on next line becomes unstable for x close to 1, this fixes it
	else return -(p.ks/rsq)*((xsq*(2.0-3*lens_function_xsq(xsq)) + 1)/((xsq-1)*(xsq-1)));
}
template double NFW::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var NFW::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar NFW::lens_function_xsq(const QScalar xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}
template double NFW::lens_function_xsq<double>(const double xsq);
#ifdef USE_STAN
template stan::math::var NFW::lens_function_xsq<stan::math::var>(const stan::math::var xsq);
#endif

double NFW::kapavg_spherical_rsq(const double rsq)
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double xsq = rsq/(p.rs*p.rs);
	// below xsq ~ 1e-6 or so, this becomes inaccurate due to fine cancellations; a series expansion
	// is done for xsq smaller than this
	if (xsq > 1e-5)
		return 2*p.ks*(2*lens_function_xsq<double>(xsq) + log(xsq/4))/xsq;
	else
		return -p.ks*(1+log(xsq/4));
}

double NFW::potential_spherical_rsq(const double rsq)
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double xsq = rsq/(p.rs*p.rs);
	if (xsq < 1) {
		if (xsq > 1e-4)
			return 2*p.ks*p.rs*p.rs*(-SQR(atanh(sqrt(1-xsq))) + SQR(log(xsq/4)/2));
		else
			return -p.ks*rsq*log(xsq/4)/2;
	}
	else {
		return 2*p.ks*p.rs*p.rs*(SQR(atan(sqrt(xsq-1))) + SQR(log(xsq/4)/2));
	}
}

double NFW::rho3d_r_integrand_analytic(const double r)
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.ks/(r*SQR(1+r/p.rs));
}

double NFW::calculate_scaled_mass_3d(const double r)
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return 4*M_PI*p.ks*p.rs*p.rs*(log(1+r/p.rs) - r/(r+p.rs));
}

double NFW::concentration_prior()
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double ds, r200;
	ds = p.ks * sigma_cr / p.rs;
	cosmo->get_halo_parameters_from_rs_ds(zlens,p.rs_kpc,ds,p.m200,r200);
	p.c200 = r200/p.rs_kpc;
	double log_medc = log(cosmo->median_concentration_dutton(p.m200,zlens));
	const double sig_logc = 0.110; // mass-concentration scatter of 0.110 dex (Dutton et al 2014)
	//return (exp(-SQR((log(p.c200)-log_medc)/(ln10*sig_logc))/2)/(sig_logc*M_SQRT_2PI));
	return (SQR((log(p.c200)-log_medc)/(ln10*sig_logc))/2 + (sig_logc*M_SQRT_2PI)); // returning -log(prior)
}

bool NFW::output_cosmology_info(const int lens_number)
{
	NFW_Params<double>& p = assign_nfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}

	double ds, r200;
	if (parameter_mode != 2) p.rs_kpc = p.rs / kpc_to_arcsec;
	ds = p.ks * sigma_cr_kpc / p.rs_kpc;
	if (parameter_mode > 0) {
		r200 = p.c200 * p.rs_kpc;
	} else {
		cosmo->get_halo_parameters_from_rs_ds(zlens,p.rs_kpc,ds,p.m200,r200);
		p.c200 = r200/p.rs_kpc;
	}

	cout << "rho_s = " << ds << " M_sun/kpc^3  (density at scale radius)" << endl;
	cout << "r_s = " << p.rs_kpc << " kpc  (" << (p.rs_kpc*kpc_to_arcsec) << " arcsec)" << endl;
	cout << "c = " << p.c200 << endl;
	if (parameter_mode > 0) {
		cout << "ks = " << p.ks << endl;
	} else {
		cout << "M_200 = " << p.m200 << " M_sun\n";
	}
	cout << "r_200 = " << r200 << " kpc  (" << (r200*kpc_to_arcsec) << " arcsec)" << endl;
	//cosmo->get_halo_parameters_from_rs_ds(5,p.rs_kpc,ds,p.m200,r200);
	//p.c200 = r200/p.rs_kpc;
	//cout << "M_200(z=5) = " << p.m200 << " M_sun\n";
	//cout << "r_200(z=5) = " << r200 << " kpc\n";
	//cout << "c(z=5) = " << p.c200 << endl;
	if (use_concentration_prior) {
		double cmprior = exp(-concentration_prior());
		cout << "concentration-mass prior P(logc|M,z) = " << cmprior << endl;
	}

	cout << endl;
	return true;
}

/********************************** Truncated_NFW **********************************/

Truncated_NFW::Truncated_NFW(const double zlens_in, const double zsrc_in, const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int truncation_mode_in, const int parameter_mode_in, Cosmology* cosmo_in)
			//cosmo->get_halo_parameters_from_rs_ds(zlens,rs_kpc,ds,m200,r200);
			//c200 = r200/rs_kpc;
{
	lensparams = &lensparams_tnfw;
	setup_lens_properties(parameter_mode_in,truncation_mode_in);
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(p1_in,p2_in,p3_in,q_in,theta_degrees,xc_in,yc_in);
}

void Truncated_NFW::initialize_parameters(const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==4) {
		p.m200 = p1_in;
		p.rs_kpc = p2_in;
		p.tau_s = p3_in;
	} else if (parameter_mode==3) {
		p.m200 = p1_in;
		p.rs_kpc = p2_in;
		p.rt_kpc = p3_in;
	} else if (parameter_mode==2) {
		p.m200 = p1_in;
		p.c200 = p2_in;
		p.tau200 = p3_in;
	} else if (parameter_mode==1) {
		p.m200 = p1_in;
		p.c200 = p2_in;
		p.rt_kpc = p3_in;
	} else {
		p.ks = p1_in;
		p.rs = p2_in;
		p.rt = p3_in;
	}

	update_meta_parameters_and_pointers();
}

void Truncated_NFW::setup_lens_properties(const int parameter_mode, const int subclass)
{
	// here "subclass" gives the truncation mode
	lenstype = TRUNCATED_nfw;
	model_name = "tnfw";
	subclass_label = "t";
	stringstream tstr;
	string tstring;
	tstr << subclass;
	tstr >> tstring;
	setup_base_lens_properties(8,3,true,parameter_mode,subclass); // number of parameters = 7, is_elliptical_lens = true
	analytic_3d_density = true;
}

Truncated_NFW::Truncated_NFW(const Truncated_NFW* lens_in)
{
	lensparams = &lensparams_tnfw;
	copy_base_lensdata(lens_in);
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.ks = lens_in->lensparams_tnfw.ks;
	p.rs = lens_in->lensparams_tnfw.rs;
	p.rt = lens_in->lensparams_tnfw.rt;
	if (parameter_mode==4) {
		p.m200 = lens_in->lensparams_tnfw.m200;
		p.rs_kpc = lens_in->lensparams_tnfw.rs_kpc;
		p.tau_s = lens_in->lensparams_tnfw.tau_s;
	} else if (parameter_mode==3) {
		p.m200 = lens_in->lensparams_tnfw.m200;
		p.rs_kpc = lens_in->lensparams_tnfw.rs_kpc;
		p.rt_kpc = lens_in->lensparams_tnfw.rt_kpc;
	} else if (parameter_mode==2) {
		p.m200 = lens_in->lensparams_tnfw.m200;
		p.c200 = lens_in->lensparams_tnfw.c200;
		p.tau200 = lens_in->lensparams_tnfw.tau200;
	} else if (parameter_mode==1) {
		p.m200 = lens_in->lensparams_tnfw.m200;
		p.c200 = lens_in->lensparams_tnfw.c200;
		p.rt_kpc = lens_in->lensparams_tnfw.rt_kpc;
	}

	special_anchor_factor = lens_in->special_anchor_factor;
	update_meta_parameters_and_pointers();
}

void Truncated_NFW::assign_paramnames()
{
	if (parameter_mode==4) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "rs_kpc"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
		paramnames[2] = "tau_s"; latex_paramnames[2] = "\\tau"; latex_param_subscripts[2] = "s";
	} else if (parameter_mode==3) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "rs_kpc"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
		paramnames[2] = "rt_kpc"; latex_paramnames[2] = "r"; latex_param_subscripts[2] = "t";
	} else if (parameter_mode==2) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "c"; latex_paramnames[1] = "c"; latex_param_subscripts[1] = "";
		paramnames[2] = "tau"; latex_paramnames[2] = "\\tau"; latex_param_subscripts[2] = "200";
	} else if (parameter_mode==1) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "c"; latex_paramnames[1] = "c"; latex_param_subscripts[1] = "";
		paramnames[2] = "rt_kpc"; latex_paramnames[2] = "r"; latex_param_subscripts[2] = "t";
	} else {
		paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
		paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
		paramnames[2] = "rt"; latex_paramnames[2] = "r"; latex_param_subscripts[2] = "t";
	}
	set_geometric_paramnames(lensprofile_nparams);
}

void Truncated_NFW::assign_param_pointers()
{
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==4) {
		p.param[0] = &p.m200;
		p.param[1] = &p.rs_kpc;
		p.param[2] = &p.tau_s;
	} else if (parameter_mode==3) {
		p.param[0] = &p.m200;
		p.param[1] = &p.rs_kpc;
		p.param[2] = &p.rt_kpc;
	} else if (parameter_mode==2) {
		p.param[0] = &p.m200;
		p.param[1] = &p.c200;
		p.param[2] = &p.tau200;
	} else if (parameter_mode==1) {
		p.param[0] = &p.m200;
		p.param[1] = &p.c200;
		p.param[2] = &p.rt_kpc;
	} else {
		p.param[0] = &p.ks;
		p.param[1] = &p.rs;
		p.param[2] = &p.rt;
	}
	set_geometric_param_pointers(lensprofile_nparams);
}

void Truncated_NFW::get_parameters_pmode(const int pmode, double* params)
{
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		// For parameter mode 0, you need to use a root-finder to solve for c, and then you can find p.m200 easily
		// This should be done here because unless you need it here, it would waste CPU time to do this every
		// time the parameters are varied
	}

	if (pmode==4) {
		p.tau_s = p.rt/p.rs;
		p.rs_kpc = p.rs / kpc_to_arcsec;
		params[0] = p.m200;
		params[1] = p.rs_kpc;
		params[2] = p.tau_s;
	} else if (pmode==3) {
		params[0] = p.m200;
		params[1] = p.rs_kpc;
		params[2] = p.rt_kpc;
	} else if (pmode==2) {
		double rvir_kpc;
		// the mvir, rvir formulas ignore the truncation, referring to the values before the NFW was tidally stripped
		rvir_kpc = pow(p.m200/(200.0*M_4PI/3.0*1e-9*cosmo->critical_density(zlens)),0.333333333333);
		p.rs_kpc = rvir_kpc / p.c200;
		p.tau200 = p.rt_kpc/rvir_kpc;
		params[0] = p.m200;
		params[1] = p.c200;
		params[2] = p.tau200;
	} else if (pmode==1) {
		params[0] = p.m200;
		params[1] = p.c200;
		params[2] = p.rt_kpc;
	} else {
		params[0] = p.ks;
		params[1] = p.rs;
		params[2] = p.rt;
	}
	for (int i=lensprofile_nparams; i < n_params; i++) {
		if (angle_param[i]) params[i] = radians_to_degrees(*(p.param[i]));
		else params[i] = *(p.param[i]);
	}
	if (lensed_center_coords) {
		params[n_params-3] = p.x_center;
		params[n_params-2] = p.y_center;
	}
	if (lensed_center_coords) {
		params[n_params-3] = p.x_center;
		params[n_params-2] = p.y_center;
	}
}

void Truncated_NFW::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	if (cosmo != NULL) {
		if ((parameter_mode==3) or (parameter_mode==4)) set_ks_c200_from_m200_rs();
		else if ((parameter_mode==1) or (parameter_mode==2)) set_ks_rs_from_m200_c200();
	}
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	rmin_einstein_radius = 1e-6*p.rs;
}

void Truncated_NFW::assign_special_anchored_parameters(LensProfile *host_in, const double factor, const bool just_created)
{
	// the following special anchoring is to enforce a mass-concentration relation
	anchor_special_parameter = true;
	special_anchor_lens = this; // not actually used anyway, since we're not anchoring to another qlens at all
	//p.c200 = factor*cosmo->median_concentration_bullock(p.m200,zlens);
	if (just_created) special_anchor_factor = factor;
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.c200 = special_anchor_factor*cosmo->median_concentration_dutton(p.m200,zlens);
	update_meta_parameters();
}

void Truncated_NFW::update_special_anchored_params()
{
	if (anchor_special_parameter) {
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
		//p.c200 = cosmo->median_concentration_bullock(p.m200,zlens);
		p.c200 = special_anchor_factor * cosmo->median_concentration_dutton(p.m200,zlens);
		update_meta_parameters();
	}
}

void Truncated_NFW::set_auto_stepsizes()
{
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	if (parameter_mode==4) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.rs_kpc;
		stepsizes[index++] = 0.2*p.tau_s;
	} else if (parameter_mode==3) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.rs_kpc;
		stepsizes[index++] = 0.2*p.rt_kpc;
	} else if (parameter_mode==2) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.c200;
		stepsizes[index++] = 0.2*p.tau200;
	} else if (parameter_mode==1) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.c200;
		stepsizes[index++] = 0.2*p.rt_kpc;
	} else {
		stepsizes[index++] = 0.2*p.ks;
		stepsizes[index++] = 0.2*p.rs;
		stepsizes[index++] = 0.2*p.rt;
	}

	set_geometric_param_auto_stepsizes(index);
}

void Truncated_NFW::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void Truncated_NFW::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Truncated_NFW::kapavg_spherical_rsq);
}

void Truncated_NFW::set_ks_c200_from_m200_rs()
{
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rvir_kpc;
	// the mvir, rvir formulas ignore the truncation, referring to the values before the NFW was tidally stripped
	rvir_kpc = pow(p.m200/(200.0*M_4PI/3.0*1e-9*cosmo->critical_density(zlens)),0.333333333333);
	p.rs = p.rs_kpc * kpc_to_arcsec;
	if (parameter_mode==4) p.rt_kpc = p.tau_s * p.rs_kpc;
	p.rt = p.rt_kpc * kpc_to_arcsec;
	p.c200 = rvir_kpc / p.rs_kpc;
	p.ks = p.m200 / (M_4PI*p.rs*p.rs*sigma_cr*(log(1+p.c200) - p.c200/(1+p.c200)));
}

void Truncated_NFW::set_ks_rs_from_m200_c200()
{
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rvir_kpc;
	// the mvir, rvir formulas ignore the truncation, referring to the values before the NFW was tidally stripped
	rvir_kpc = pow(p.m200/(200.0*M_4PI/3.0*1e-9*cosmo->critical_density(zlens)),0.333333333333);
	p.rs_kpc = rvir_kpc / p.c200;
	p.rs = p.rs_kpc * kpc_to_arcsec;
	if (parameter_mode==2) p.rt_kpc = p.tau200 * rvir_kpc;
	p.rt = p.rt_kpc * kpc_to_arcsec;
	p.ks = p.m200 / (M_4PI*p.rs*p.rs*sigma_cr*(log(1+p.c200) - p.c200/(1+p.c200)));
}

template <typename QScalar>
QScalar Truncated_NFW::kappa_rsq_impl(const QScalar rsq)
{
	Truncated_NFW_Params<QScalar>& p = assign_tnfw_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar xsq, tsq, sqrttx, lx, lf, tmp, ans;
	xsq = rsq/(p.rs*p.rs);
	tsq = SQR(p.rt/p.rs);
	sqrttx = sqrt(tsq+xsq);
	lx = log(sqrt(xsq)/(sqrttx+sqrt(tsq)));
	if (xsq < 1e-6) lf = -log(xsq/4)/2;
	else lf = lens_function_xsq(xsq);
	if (lens_subclass==0) {
		if (xsq==1) tmp = (tsq+1)/3.0 + 2*lf - M_PI/sqrttx + (tsq-1)*lx/(sqrttx*p.rt/p.rs);
		else tmp = ((tsq+1)/(xsq-1))*(1-lf) + 2*lf - M_PI/sqrttx + (tsq-1)*lx/(sqrttx*p.rt/p.rs);
		ans = 2*p.ks*tsq/SQR(tsq+1)*tmp;
	} else {

		if (xsq==1) tmp = 2*(tsq+1)/3.0 + 8.0 + (tsq*tsq-1)/tsq/(tsq+1) + (-M_PI*(4*(tsq+1)+tsq+1) + (tsq*(tsq*tsq-1) + (tsq+1)*(3*tsq*tsq-6*tsq-1))*lx/CUBE(p.rt/p.rs))/CUBE(sqrttx);
		else tmp = 2*(tsq+1)/(xsq-1)*(1-lf) + 8*lf + (tsq*tsq-1)/tsq/(tsq+xsq) + (-M_PI*(4*(tsq+xsq)+tsq+1) + (tsq*(tsq*tsq-1) + (tsq+xsq)*(3*tsq*tsq-6*tsq-1))*lx/CUBE(p.rt/p.rs))/CUBE(sqrttx);
		ans = p.ks*tsq*tsq/CUBE(tsq+1)*tmp;
	}
	return ans;
}
template double Truncated_NFW::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Truncated_NFW::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
inline QScalar Truncated_NFW::lens_function_xsq(const QScalar xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}
template double Truncated_NFW::lens_function_xsq<double>(const double xsq);
#ifdef USE_STAN
template stan::math::var Truncated_NFW::lens_function_xsq<stan::math::var>(const stan::math::var xsq);
#endif

double Truncated_NFW::kapavg_spherical_rsq(const double rsq)
{
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double xsq, tau, tsq, sqrttx, lx, tmp, ans;
	xsq = rsq/(p.rs*p.rs);
	tau = p.rt/p.rs;
	tsq = tau*tau;
	if ((xsq < 1e-6) and (xsq/tsq < 1e-6)) return -p.ks*(1+log(xsq/4)); // fixes numerical instability in limit of small r
	sqrttx = sqrt(tsq+xsq);
	lx = log(sqrt(xsq)/(sqrttx+sqrt(tsq)));
	if (lens_subclass==0) {
		tmp = (tsq + 1 + 2*(xsq-1))*lens_function_xsq<double>(xsq) + M_PI*tau + (tsq-1)*log(tau) + sqrttx*(-M_PI + (tsq-1)*lx/tau);
		ans = 4*p.ks*tsq/SQR(tsq+1)/xsq*tmp; // now, tmp = kappa_average
	} else {
		tmp = 2*(tsq+1+4*(xsq-1))*lens_function_xsq<double>(xsq) + (M_PI*(3*tsq-1) + 2*tau*(tsq-3)*log(tau))/tau + (-CUBE(tau)*M_PI*(4*(tsq+xsq)-tsq-1) + (-tsq*(tsq*tsq-1) + (tsq+xsq)*(3*tsq*tsq-6*tsq-1))*lx)/CUBE(tau)/sqrttx;
		ans = 2*p.ks*tsq*tsq/CUBE(tsq+1)/xsq*tmp; // now, tmp = kappa_average
	}
	return ans;
}

double Truncated_NFW::rho3d_r_integrand_analytic(const double r)
{
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_subclass==0) {
		return (p.ks/r/SQR(1+r/p.rs)/(1+SQR(r/p.rt)));
	} else {
		return (p.ks/r/SQR(1+r/p.rs)/SQR(1+SQR(r/p.rt)));
	}
}

bool Truncated_NFW::output_cosmology_info(const int lens_number)
{
	Truncated_NFW_Params<double>& p = assign_tnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}

	double ds, r200;
	if ((parameter_mode != 3) and (parameter_mode != 4)) p.rs_kpc = p.rs / kpc_to_arcsec;
	if (parameter_mode == 0) p.rt_kpc = p.rt / kpc_to_arcsec;
	ds = p.ks * sigma_cr_kpc / p.rs_kpc;
	if (parameter_mode > 0) {
		r200 = p.c200 * p.rs_kpc;
		if (parameter_mode == 2) p.rt_kpc = p.tau200 * r200;
		if (parameter_mode == 4) p.rt_kpc = p.tau_s * p.rs_kpc;
	} else {
		cosmo->get_halo_parameters_from_rs_ds(zlens,p.rs_kpc,ds,p.m200,r200);
		p.c200 = r200/p.rs_kpc;
	}

	cout << "rho_s = " << ds << " M_sun/kpc^3  (density at scale radius)" << endl;
	cout << "r_s = " << p.rs_kpc << " kpc  (" << (p.rs_kpc*kpc_to_arcsec) << " arcsec)" << endl;
	cout << "r_t = " << p.rt_kpc << " kpc  (truncation radius)" << endl;
	cout << "c = " << p.c200 << endl;
	if (parameter_mode > 0) {
		cout << "ks = " << p.ks << endl;
	} else {
		cout << "M_200 = " << p.m200 << " M_sun (ignores truncation)\n";
	}
	cout << "r_200 = " << r200 << " kpc  (" << (r200*kpc_to_arcsec) << " arcsec) (NOTE: ignores truncation)" << endl;

	//cosmo->get_halo_parameters_from_rs_ds(5,p.rs_kpc,ds,p.m200,r200);
	//zlens = 5;
	//update_cosmology_meta_parameters();
	//p.c200 = r200/p.rs_kpc;
	//set_ks_rs_from_m200_c200();
	//cout << "M_200(z=5) = " << p.m200 << " M_sun\n";
	//cout << "r_200(z=5) = " << r200 << " kpc\n";
	//cout << "c(z=5) = " << p.c200 << endl;


	cout << endl;
	return true;
}

/********************************** Cored_NFW **********************************/

Cored_NFW::Cored_NFW(const double zlens_in, const double zsrc_in, const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_cnfw;
	setup_lens_properties(parameter_mode_in);
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(p1_in,p2_in,p3_in,q_in,theta_degrees,xc_in,yc_in);
}

void Cored_NFW::initialize_parameters(const double &p1_in, const double &p2_in, const double &p3_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==3) {
		p.m200 = p1_in;
		p.c200 = p2_in;
		p.rc_kpc = p3_in;
	} else if (parameter_mode==2) {
		p.m200 = p1_in;
		p.rs_kpc = p2_in;
		p.beta = p3_in;
	} else if (parameter_mode==1) {
		p.m200 = p1_in;
		p.c200 = p2_in;
		p.beta = p3_in;
	} else {
		p.ks = p1_in;
		p.rs = p2_in;
		p.rc = p3_in;
	}

	update_meta_parameters_and_pointers();
}


void Cored_NFW::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = CORED_nfw;
	model_name = "cnfw";
	setup_base_lens_properties(8,3,true,parameter_mode); // number of parameters = 7, is_elliptical_lens = true
	analytic_3d_density = true;
}

Cored_NFW::Cored_NFW(const Cored_NFW* lens_in)
{
	lensparams = &lensparams_cnfw;
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	copy_base_lensdata(lens_in);
	p.ks = lens_in->lensparams_cnfw.ks;
	p.rs = lens_in->lensparams_cnfw.rs;
	p.rc = lens_in->lensparams_cnfw.rc;
	p.beta = lens_in->lensparams_cnfw.beta;
	if (parameter_mode==3) {
		p.m200 = lens_in->lensparams_cnfw.m200;
		p.c200 = lens_in->lensparams_cnfw.c200;
		p.rc_kpc = lens_in->lensparams_cnfw.rc_kpc;
	} else if (parameter_mode==2) {
		p.m200 = lens_in->lensparams_cnfw.m200;
		p.rs_kpc = lens_in->lensparams_cnfw.rs_kpc;
	} else if (parameter_mode==1) {
		p.m200 = lens_in->lensparams_cnfw.m200;
		p.c200 = lens_in->lensparams_cnfw.c200;
	}

	special_anchor_factor = lens_in->special_anchor_factor;
	update_meta_parameters_and_pointers();
}

void Cored_NFW::assign_paramnames()
{
	if (parameter_mode==3) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "c"; latex_paramnames[1] = "c"; latex_param_subscripts[1] = "";
		paramnames[2] = "rc_kpc"; latex_paramnames[2] = "r"; latex_param_subscripts[2] = "c";
	} else if (parameter_mode==2) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "rs_kpc"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
		paramnames[2] = "beta"; latex_paramnames[2] = "\\beta"; latex_param_subscripts[2] = "c";
	} else if (parameter_mode==1) {
		paramnames[0] = "mvir"; latex_paramnames[0] = "m"; latex_param_subscripts[0] = "vir";
		paramnames[1] = "c"; latex_paramnames[1] = "c"; latex_param_subscripts[1] = "";
		paramnames[2] = "beta"; latex_paramnames[2] = "\\beta"; latex_param_subscripts[2] = "c";
	} else {
		paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
		paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
		paramnames[2] = "rc"; latex_paramnames[2] = "r"; latex_param_subscripts[2] = "c";
	}
	set_geometric_paramnames(lensprofile_nparams);
}

void Cored_NFW::assign_param_pointers()
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==3) {
		p.param[0] = &p.m200;
		p.param[1] = &p.c200;
		p.param[2] = &p.rc_kpc;
	} else if (parameter_mode==2) {
		p.param[0] = &p.m200;
		p.param[1] = &p.rs_kpc;
		p.param[2] = &p.beta;
	} else if (parameter_mode==1) {
		p.param[0] = &p.m200;
		p.param[1] = &p.c200;
		p.param[2] = &p.beta;
	} else {
		p.param[0] = &p.ks;
		p.param[1] = &p.rs;
		p.param[2] = &p.rc;
	}
	set_geometric_param_pointers(lensprofile_nparams);
}

void Cored_NFW::get_parameters_pmode(const int pmode, double* params)
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (pmode==3) {
		params[0] = p.m200;
		params[1] = p.c200;
		params[2] = p.rc_kpc;
	} else if (pmode==2) {
		params[0] = p.m200;
		params[1] = p.rs_kpc;
		params[2] = p.beta;
	} else if (pmode==1) {
		params[0] = p.m200;
		params[1] = p.c200;
		params[2] = p.beta;
	} else {
		params[0] = p.ks;
		params[1] = p.rs;
		params[2] = p.rc;
	}
	for (int i=lensprofile_nparams; i < n_params; i++) {
		if (angle_param[i]) params[i] = radians_to_degrees(*(p.param[i]));
		else params[i] = *(p.param[i]);
	}
	if (lensed_center_coords) {
		params[n_params-3] = p.x_center;
		params[n_params-2] = p.y_center;
	}
}

void Cored_NFW::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (cosmo != NULL) {
		if (parameter_mode==3) {
			set_ks_rs_from_m200_c200_rckpc();
		} else if (parameter_mode==2) {
			set_ks_c200_from_m200_rs();
			p.rc = p.beta*p.rs;
		} else if (parameter_mode==1) {
			set_ks_rs_from_m200_c200_beta();
			p.rc = p.beta*p.rs;
		} else {
			p.beta = p.rc/p.rs;
		}
	}
	rmin_einstein_radius = 1e-6*p.rs;
	//if (p.rs <= p.rc) die("scale radius a cannot be equal to or less than core radius s for Cored NFW model");
}

void Cored_NFW::assign_special_anchored_parameters(LensProfile *host_in, const double factor, const bool just_created)
{
	// the following special anchoring is to enforce a mass-concentration relation
	anchor_special_parameter = true;
	special_anchor_lens = this; // not actually used anyway, since we're not anchoring to another lens at all
	if (just_created) special_anchor_factor = factor;
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.c200 = special_anchor_factor*cosmo->median_concentration_bullock(p.m200,zlens);
	update_meta_parameters();
}

void Cored_NFW::update_special_anchored_params()
{
	if (anchor_special_parameter) {
		Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
		p.c200 = cosmo->median_concentration_bullock(p.m200,zlens);
		update_meta_parameters();
	}
}

void Cored_NFW::set_auto_stepsizes()
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	if (parameter_mode==3) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.c200;
		stepsizes[index++] = 0.05*p.rs_kpc;
	} else if (parameter_mode==2) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.rs_kpc;
		stepsizes[index++] = 0.2*p.beta;
	} else if (parameter_mode==1) {
		stepsizes[index++] = 0.2*p.m200;
		stepsizes[index++] = 0.2*p.c200;
		stepsizes[index++] = 0.2*p.beta;
	} else {
		stepsizes[index++] = 0.2*p.ks;
		stepsizes[index++] = 0.2*p.rs;
		stepsizes[index++] = 0.05*p.rs;
	}
	set_geometric_param_auto_stepsizes(index);
}

void Cored_NFW::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void Cored_NFW::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Cored_NFW::kapavg_spherical_rsq);
	//potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Cored_NFW::potential_spherical_rsq);
}

void Cored_NFW::set_ks_rs_from_m200_c200_beta()
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rvir_kpc;
	rvir_kpc = pow(p.m200/(200.0*M_4PI/3.0*1e-9*cosmo->critical_density(zlens)),0.333333333333);
	p.rs_kpc = rvir_kpc / p.c200;
	p.rs = p.rs_kpc * kpc_to_arcsec;
	double rcterm;
	if (p.beta==0.0) rcterm = 0;
	else rcterm = p.beta*p.beta*log(1+p.c200/p.beta);
	p.ks = p.m200 / (M_4PI*p.rs*p.rs*sigma_cr*((1-2*p.beta)*log(1+p.c200) + rcterm - (1-p.beta)*p.c200/(1+p.c200))/SQR(1-p.beta));
}

void Cored_NFW::set_ks_rs_from_m200_c200_rckpc()
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rvir_kpc;
	rvir_kpc = pow(p.m200/(200.0*M_4PI/3.0*1e-9*cosmo->critical_density(zlens)),0.333333333333);
	p.rs_kpc = rvir_kpc / p.c200;
	p.rs = p.rs_kpc * kpc_to_arcsec;
	p.rc = p.rc_kpc * kpc_to_arcsec;
	p.beta = p.rc/p.rs;
	double rcterm;
	if (p.beta==0.0) rcterm = 0;
	else rcterm = p.beta*p.beta*log(1+p.c200/p.beta);
	p.ks = p.m200 / (M_4PI*p.rs*p.rs*sigma_cr*((1-2*p.beta)*log(1+p.c200) + rcterm - (1-p.beta)*p.c200/(1+p.c200))/SQR(1-p.beta));
}

void Cored_NFW::set_ks_c200_from_m200_rs()
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rvir_kpc;
	rvir_kpc = pow(p.m200/(200.0*M_4PI/3.0*1e-9*cosmo->critical_density(zlens)),0.333333333333);
	p.rs = p.rs_kpc * kpc_to_arcsec;
	p.c200 = rvir_kpc / p.rs_kpc;
	double rcterm;
	if (p.beta==0.0) rcterm = 0;
	else rcterm = p.beta*p.beta*log(1+p.c200/p.beta);
	p.ks = p.m200 / (M_4PI*p.rs*p.rs*sigma_cr*((1-2*p.beta)*log(1+p.c200) + rcterm - (1-p.beta)*p.c200/(1+p.c200))/SQR(1-p.beta));
}

template <typename QScalar>
QScalar Cored_NFW::kappa_rsq_impl(const QScalar rsq)
{
	Cored_NFW_Params<QScalar>& p = assign_cnfw_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar xsq, rsterm, rcterm;
	xsq = rsq/(p.rs*p.rs);
	if (p.rc < 1e-120) {
		if (xsq < 1e-6) return -p.ks*(2+log(xsq/4));
		rcterm = 0;
	}
	else {
		if (abs(1-p.beta) < 5e-4) {
			// formulae are unstable near p.beta=1, so we use a series expansion here
			if (abs(xsq-1) < 6e-5) {
				QScalar ans1, ans2;
				// the following is a quick and dirty way to avoid singularity which is very close to x=1
				xsq += 1.0e-4;
				ans1 = p.ks*(1 + 2*xsq - 3*xsq*lens_function_xsq(xsq)) / SQR(xsq-1);
				xsq -= 2.0e-4;
				ans2 = p.ks*(1 + 2*xsq - 3*xsq*lens_function_xsq(xsq)) / SQR(xsq-1);
				return (ans1 + ans2)/2;
			} else {
				if (xsq < 1e-5)
					return p.ks*(1 + 2*xsq - 1.5*xsq)/SQR(xsq-1);
				else
					return p.ks*(1 + 2*xsq - 3*xsq*lens_function_xsq(xsq)) / SQR(xsq-1);
			}
		}
		QScalar xcsq = rsq/(p.rc*p.rc);
		if (xcsq < 1e-8) {
			if (xsq < 1e-14) return -p.ks*(log(p.beta*p.beta) + 2*(1-p.beta)) / SQR(1-p.beta);
			rcterm = -log(xcsq/4)/2;
		} else rcterm = lens_function_xsq(xcsq);
	}
	if (xsq < 1e-8) rsterm = -(1 - p.beta + log(xsq/4)/2);
	else if (abs(xsq-1) < 1e-5) rsterm = (1+2*p.beta)*0.3333333333333333 - (1-p.beta)*(0.2 + 2*p.beta/15.0)*(xsq-1); // formula on next line becomes unstable for x close to 1, this fixes it
	else rsterm = (1 - p.beta - (1-xsq*p.beta)*lens_function_xsq(xsq))/(xsq-1);
	return 2*p.ks/SQR(1-p.beta)*(rsterm - rcterm);
}
template double Cored_NFW::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Cored_NFW::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar Cored_NFW::kappa_rsq_deriv_impl(const QScalar rsq)
{
	Cored_NFW_Params<QScalar>& p = assign_cnfw_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar xsq, xcsq, rsterm, rcterm;
	xsq = rsq/(p.rs*p.rs);
	xcsq = rsq/(p.rc*p.rc);
	if (p.rc < 1e-120) rcterm = 0;
	else if ((xcsq < 1e-1) and (xsq < 1e-14)) return 0; // this could be improved on for a more seamless transition, but it's at such a small r it really doesn't matter
	else if (abs(1-p.beta) < 5e-4) {
		// formulae are unstable near p.beta=1, so we use a series expansion here
		if (abs(xsq-1) < 1.2e-3) {
			// the following is a quick and dirty way to avoid singularity which is very close to x=1
			QScalar ans1, ans2;
			xsq += 2.4e-3;
			ans1 = p.ks/SQR(p.rs)*(-5.5 - 2*xsq + 3*(1 + 1.5*xsq)*lens_function_xsq(xsq))/CUBE(xsq-1);
			xsq -= 4.8e-3;
			ans2 = p.ks/SQR(p.rs)*(-5.5 - 2*xsq + 3*(1 + 1.5*xsq)*lens_function_xsq(xsq))/CUBE(xsq-1);
			return (ans1+ans2)/2;
		} else {
			if (xsq < 1e-10) return 0;
			else return p.ks/SQR(p.rs)*(-5.5 - 2*xsq + 3*(1 + 1.5*xsq)*lens_function_xsq(xsq))/CUBE(xsq-1);
		}
	}
	else rcterm = (lens_function_xsq(xcsq) - 1.0/xcsq) / (xsq - p.beta*p.beta);

	if (xsq < 1e-10) rsterm = -1.0/xsq;
	else if (abs(xsq-1) < 1e-5) rsterm = -2*(0.2 + 2*p.beta/15.0); // formula on next line becomes unstable for x close to 1, this fixes it
	else rsterm = (-2 + 3*p.beta + (3-2*p.beta-xsq*p.beta)*lens_function_xsq(xsq) - 1.0/xsq)/SQR(xsq-1);

	return p.ks/SQR(p.rs*(1-p.beta))*(rsterm + rcterm);
}
template double Cored_NFW::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Cored_NFW::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
inline QScalar Cored_NFW::lens_function_xsq(const QScalar xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ?  (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}
template double Cored_NFW::lens_function_xsq<double>(const double xsq);
#ifdef USE_STAN
template stan::math::var Cored_NFW::lens_function_xsq<stan::math::var>(const stan::math::var xsq);
#endif

/*
inline double Cored_NFW::potential_lens_function_xsq(const double &xsq)
{
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1))) : (xsq < 1.0) ?  (-atanh(sqrt(1-xsq))) : 0.0);
}

double Cored_NFW::potential_spherical_rsq(const double rsq)
{
	// Something is wrong with these formulae, but I don't have time to fix now. Figure out later
	double betasq, xsq, xcsq, rsfac, rcfac, logbterm;
	betasq = p.beta*p.beta;
	xsq = rsq/(p.rs*p.rs);
	if (xsq < 1e-6) rsfac = log(xsq/4)/2;
	else rsfac = potential_lens_function_xsq(xsq);
	if (p.rc==0.0) {
		rcfac = 0;
		xcsq = 0;
	} else {
		xcsq = rsq/(p.rc*p.rc);
		if (xcsq < 1e-6) rsfac = log(xcsq/4)/2;
		else rcfac = potential_lens_function_xsq(xcsq);
	}
	if (p.beta==0.0) logbterm = 0;
	else logbterm = p.beta*log(betasq)/2;
	double rsfacsq = (xsq >= 1.0) ? rsfac*rsfac : -rsfac*rsfac;
	double rcfacsq = (xcsq >= 1.0) ? rcfac*rcfac : -rcfac*rcfac;
	double ans =  2*p.ks*p.rs*p.rs/SQR(1-p.beta)*(betasq*rcfacsq - 2*p.beta*sqrt(abs(xsq-betasq))*rcfac - p.beta*(logbterm - p.beta + 1)*log(xsq) + SQR((betasq-1)*log(xsq/4))/4 + 2*p.beta*sqrt(abs(xsq-1))*rsfac + (1-2*p.beta)*rsfacsq);
	ans -= betasq*log(4*betasq) - betasq*SQR(log(betasq))/4 - p.beta*log(4); // this should be the limit as xsq --> 0, but something is wrong here
	return ans;
}
*/

double Cored_NFW::kapavg_spherical_rsq(const double rsq)
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double betasq, xsq, rsterm, rcterm;
	betasq = p.beta*p.beta;
	xsq = rsq/(p.rs*p.rs);
	if (p.rc < 1e-120) {
		if (xsq < 1e-6) return -p.ks*(1+log(xsq/4));
		rcterm = 0;
	}
	else {
		if (abs(1-p.beta) < 5e-4) {
			// formulae are unstable near p.beta=1, so we use a series expansion here
			if (abs(xsq-1) < 1e-9) {
				double ans1, ans2;
				// the following is a quick and dirty way to avoid singularity which is very close to x=1
				xsq += 1.0e-8;
				ans1 = 2*(p.ks/xsq)*(log(xsq/4) + (-xsq + (3*xsq-2)*lens_function_xsq<double>(xsq))/(xsq-1));
				xsq -= 2.0e-8;
				ans2 = 2*(p.ks/xsq)*(log(xsq/4) + (-xsq + (3*xsq-2)*lens_function_xsq<double>(xsq))/(xsq-1));
				return (ans1+ans2)/2;
			} else {
				return 2*(p.ks/xsq)*(log(xsq/4) + (-xsq + (3*xsq-2)*lens_function_xsq<double>(xsq))/(xsq-1));
			}
		}
		double xcsq = rsq/(p.rc*p.rc);
		if ((xcsq < 1e-5) and (xsq < 1e-5)) return -p.ks*(log(p.beta*p.beta) + 2*(1-p.beta))/SQR(1-p.beta); // inside the core, kappa_avg = kappa (constant density)
		rcterm = 2*(betasq - xsq)*lens_function_xsq<double>(xcsq) - betasq*log(betasq);
	}
	if (xsq > 1e-5)
		rsterm = SQR(1-p.beta)*log(xsq/4) + 2*(1+p.beta*(xsq-2))*lens_function_xsq<double>(xsq);
	else
		rsterm = (p.beta-0.5)*xsq + (betasq-xsq/2)*log(xsq/4);
	return 2*p.ks*(rsterm + rcterm)/(xsq*SQR(1-p.beta));
}

double Cored_NFW::rho3d_r_integrand_analytic(const double r)
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return (p.ks/(r+p.rc)/SQR(1+r/p.rs));
}

double Cored_NFW::calculate_scaled_mass_3d(const double r)
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rcterm;
	if (p.rc < 1e-120) rcterm = 0;
	else rcterm = p.beta*p.beta*log(1+r/p.rc);
	return 4*M_PI*p.ks*p.rs*p.rs*((1-2*p.beta)*log(1+r/p.rs) + rcterm - (1-p.beta)*r/(r+p.rs))/SQR(1-p.beta);
}

bool Cored_NFW::output_cosmology_info(const int lens_number)
{
	Cored_NFW_Params<double>& p = assign_cnfw_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}

	double ds, r200;
	if (parameter_mode != 2) p.rs_kpc = p.rs / kpc_to_arcsec;
	if (parameter_mode != 3) p.rc_kpc = p.beta * p.rs_kpc;
	ds = p.ks * sigma_cr_kpc / p.rs_kpc;
	if (parameter_mode > 0) {
		r200 = p.c200 * p.rs_kpc;
	} else {
		cosmo->get_cored_halo_parameters_from_rs_ds(zlens,p.rs_kpc,ds,p.beta,p.m200,r200);
		p.c200 = r200/p.rs_kpc;
	}

	cout << "rho_s = " << ds << " M_sun/kpc^3 (density at scale radius)" << endl;
	cout << "r_s = " << p.rs_kpc << " kpc  (" << (p.rs_kpc*kpc_to_arcsec) << " arcsec)" << endl;
	cout << "r_c = " << p.rc_kpc << " kpc (core radius)" << endl;
	cout << "c = " << p.c200 << endl;
	if (parameter_mode > 0) {
		cout << "ks = " << p.ks << endl;
		cout << "r_200 = " << r200 << " kpc\n";
	} else {
		cout << "M_200 = " << p.m200 << " M_sun\n";
		cout << "r_200 = " << r200 << " kpc\n";
	}
	cout << endl;
	return true;
}

/********************************** Hernquist **********************************/

Hernquist::Hernquist(const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_hernquist;
	setup_lens_properties();
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(ks_in,rs_in,q_in,theta_degrees,xc_in,yc_in);
}

void Hernquist::initialize_parameters(const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	Hernquist_Params<double>& p = assign_hernquist_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.ks = ks_in;
	p.rs = rs_in;

	update_meta_parameters_and_pointers();
}

void Hernquist::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = HERNQUIST;
	model_name = "hern";
	setup_base_lens_properties(7,2,true); // number of parameters = 6, is_elliptical_lens = true
	analytic_3d_density = true;
}

Hernquist::Hernquist(const Hernquist* lens_in)
{
	lensparams = &lensparams_hernquist;
	copy_base_lensdata(lens_in);
	Hernquist_Params<double>& p = assign_hernquist_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.ks = lens_in->lensparams_hernquist.ks;
	p.rs = lens_in->lensparams_hernquist.rs;

	update_meta_parameters_and_pointers();
}

void Hernquist::assign_paramnames()
{
	paramnames[0] = "ks"; latex_paramnames[0] = "k"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rs"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	set_geometric_paramnames(lensprofile_nparams);
}

void Hernquist::assign_param_pointers()
{
	Hernquist_Params<double>& p = assign_hernquist_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.param[0] = &p.ks;
	p.param[1] = &p.rs;
	set_geometric_param_pointers(lensprofile_nparams);
}

void Hernquist::update_meta_parameters()
{
	Hernquist_Params<double>& p = assign_hernquist_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	rmin_einstein_radius = 1e-6*p.rs;
}

void Hernquist::set_auto_stepsizes()
{
	Hernquist_Params<double>& p = assign_hernquist_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	stepsizes[index++] = 0.2*p.ks;
	stepsizes[index++] = 0.2*p.rs;
	set_geometric_param_auto_stepsizes(index);
}

void Hernquist::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void Hernquist::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Hernquist::kapavg_spherical_rsq);
	potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Hernquist::potential_spherical_rsq);
}

template <typename QScalar>
QScalar Hernquist::kappa_rsq_impl(const QScalar rsq)
{
	Hernquist_Params<QScalar>& p = assign_hernquist_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar xsq = rsq/(p.rs*p.rs);
	if (abs(xsq-1.0) < 1e-4) return 0.4*p.ks*(0.666666666667 - 0.571428571429*(xsq-1)); // function on next line becomes unstable for x close to 1, this fixes it
	return (p.ks*(-3 + (2+xsq)*lens_function_xsq(xsq))/((xsq-1)*(xsq-1)));
}
template double Hernquist::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Hernquist::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar Hernquist::kappa_rsq_deriv_impl(const QScalar rsq)
{
	Hernquist_Params<QScalar>& p = assign_hernquist_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar xsq = rsq/(p.rs*p.rs);
	if (abs(xsq-1.0) < 1e-4) return -0.4*p.ks*(0.571428571429)/(p.rs*p.rs); // function on next line becomes unstable for x close to 1, this fixes it
	return ((p.ks/((2*rsq)*CUBE(xsq-1))) * (-3*xsq*lens_function_xsq(xsq)*(xsq+4) + 13*xsq + 2));
}
template double Hernquist::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Hernquist::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
inline QScalar Hernquist::lens_function_xsq(const QScalar xsq)
{
	Hernquist_Params<QScalar>& p = assign_hernquist_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return ((xsq > 1.0) ? (atan(sqrt(xsq-1)) / sqrt(xsq-1)) : (xsq < 1.0) ? (atanh(sqrt(1-xsq)) / sqrt(1-xsq)) : 1.0);
}
template double Hernquist::lens_function_xsq<double>(const double xsq);
#ifdef USE_STAN
template stan::math::var Hernquist::lens_function_xsq<stan::math::var>(const stan::math::var xsq);
#endif

double Hernquist::kapavg_spherical_rsq(const double rsq)
{
	Hernquist_Params<double>& p = assign_hernquist_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double xsq = rsq/(p.rs*p.rs);
	if (xsq==1) return (2*p.ks/3.0);
	else return 2*p.ks*(1 - lens_function_xsq<double>(xsq))/(xsq - 1);
}

double Hernquist::potential_spherical_rsq(const double rsq)
{
	Hernquist_Params<double>& p = assign_hernquist_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double xsq = rsq/(p.rs*p.rs);
	return p.ks*p.rs*p.rs*(log(xsq/4) + 2*lens_function_xsq<double>(xsq));
}

double Hernquist::rho3d_r_integrand_analytic(const double r)
{
	Hernquist_Params<double>& p = assign_hernquist_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return (p.ks/r/CUBE(1+r/p.rs));
}

/********************************** Exponential Disk **********************************/

ExpDisk::ExpDisk(const double zlens_in, const double zsrc_in, const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees,
		const double &xc_in, const double &yc_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_expdisk;
	setup_lens_properties();
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(k0_in,R_d_in,q_in,theta_degrees,xc_in,yc_in);
}

void ExpDisk::initialize_parameters(const double &k0_in, const double &R_d_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	ExpDisk_Params<double>& p = assign_expdisk_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.k0 = k0_in;
	p.R_d = R_d_in;

	update_meta_parameters_and_pointers();
}

void ExpDisk::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = EXPDISK;
	model_name = "expdisk";
	setup_base_lens_properties(7,2,true); // number of parameters = 6, is_elliptical_lens = true
}

ExpDisk::ExpDisk(const ExpDisk* lens_in)
{
	lensparams = &lensparams_expdisk;
	copy_base_lensdata(lens_in);
	ExpDisk_Params<double>& p = assign_expdisk_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.k0 = lens_in->lensparams_expdisk.k0;
	p.R_d = lens_in->lensparams_expdisk.R_d;

	update_meta_parameters_and_pointers();
}

void ExpDisk::assign_paramnames()
{
	paramnames[0] = "k0";  latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	paramnames[1] = "R_d"; latex_paramnames[1] = "R";       latex_param_subscripts[1] = "d";
	set_geometric_paramnames(lensprofile_nparams);
}

void ExpDisk::assign_param_pointers()
{
	ExpDisk_Params<double>& p = assign_expdisk_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.param[0] = &p.k0;
	p.param[1] = &p.R_d;
	set_geometric_param_pointers(lensprofile_nparams);
}

void ExpDisk::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	ExpDisk_Params<double>& p = assign_expdisk_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	rmin_einstein_radius = 1e-6*p.R_d;
}

void ExpDisk::set_auto_stepsizes()
{
	ExpDisk_Params<double>& p = assign_expdisk_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	stepsizes[index++] = 0.2*p.k0;
	stepsizes[index++] = 0.2*p.R_d;
	set_geometric_param_auto_stepsizes(index);
}

void ExpDisk::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void ExpDisk::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&ExpDisk::kapavg_spherical_rsq);
}

template <typename QScalar>
QScalar ExpDisk::kappa_rsq_impl(const QScalar rsq)
{
	ExpDisk_Params<QScalar>& p = assign_expdisk_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return (p.k0*exp(-sqrt(rsq)/p.R_d));
}
template double ExpDisk::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var ExpDisk::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar ExpDisk::kappa_rsq_deriv_impl(const QScalar rsq)
{
	ExpDisk_Params<QScalar>& p = assign_expdisk_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar r = sqrt(rsq);
	return (-p.k0*exp(-r/p.R_d)/(p.R_d*2*r));
}
template double ExpDisk::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var ExpDisk::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

double ExpDisk::kapavg_spherical_rsq(const double rsq)
{
	ExpDisk_Params<double>& p = assign_expdisk_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double x = sqrt(rsq)/p.R_d;
	return 2*p.k0*(1 - (1+x)*exp(-x))/(x*x);
}

bool ExpDisk::calculate_total_scaled_mass(double& total_mass)
{
	ExpDisk_Params<double>& p = assign_expdisk_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	total_mass = 2*M_PI*p.k0*p.R_d*p.R_d;
	return true;
}

/***************************** External shear *****************************/

Shear::Shear(const double zlens_in, const double zsrc_in, const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_shear;
	this->setup_lens_properties();
	this->set_redshifts(zlens_in,zsrc_in);
	this->setup_cosmology(cosmo_in);
	this->initialize_parameters(shear_p1_in,shear_p2_in,xc_in,yc_in);
}

void Shear::initialize_parameters(const double &shear_p1_in, const double &shear_p2_in, const double &xc_in, const double &yc_in)
{
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (use_shear_component_params) {
		p.shear1 = shear_p1_in;
		p.shear2 = shear_p2_in;
	} else {
		p.shear = shear_p1_in;
		this->set_angle(shear_p2_in);
	}
	p.x_center = xc_in;
	p.y_center = yc_in;
	this->update_meta_parameters();
}

void Shear::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = SHEAR;
	model_name = "shear";
	this->setup_base_lens_properties(5,-1,false); // number of parameters = 4, is_elliptical_lens = false
}

Shear::Shear(const Shear* lens_in)
{
	lensparams = &lensparams_shear;
	this->copy_base_lensdata(lens_in);
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.shear1 = lens_in->lensparams_shear.shear1;
	p.shear2 = lens_in->lensparams_shear.shear2;
	p.shear = lens_in->lensparams_shear.shear;
	this->update_meta_parameters();
}

void Shear::assign_paramnames()
{
	if (use_shear_component_params) {
		paramnames[0] = "shear1";      latex_paramnames[0] = "\\gamma"; latex_param_subscripts[0] = "1";
		paramnames[1] = "shear2";      latex_paramnames[1] = "\\gamma"; latex_param_subscripts[1] = "2";
	} else {
		paramnames[0] = "shear";       latex_paramnames[0] = "\\gamma"; latex_param_subscripts[0] = "ext";
		if (angle_points_towards_perturber) {
			paramnames[1] = "theta_pert"; latex_paramnames[1] = "\\theta"; latex_param_subscripts[1] = "pert";
		} else {
			paramnames[1] = "theta_shear"; latex_paramnames[1] = "\\theta"; latex_param_subscripts[1] = "\\gamma";
		}
	}
	paramnames[2] = "xc"; latex_paramnames[2] = "x"; latex_param_subscripts[2] = "c";
	paramnames[3] = "yc"; latex_paramnames[3] = "y"; latex_param_subscripts[3] = "c";
	if (lensed_center_coords) {
		paramnames[2] += "_l"; latex_param_subscripts[2] += ",l";
		paramnames[3] += "_l"; latex_param_subscripts[3] += ",l";
	}
}

void Shear::assign_param_pointers()
{
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	ellipticity_paramnum = -1; // no ellipticity parameter here
	if (use_shear_component_params) {
		p.param[0] = &p.shear1;
		p.param[1] = &p.shear2; angle_param[1] = false; angle_param_exists = false;
		angle_param_exists = false; // since there is no angle parameter in this mode
	} else {
		p.param[0] = &p.shear; // here, p.shear is actually the p.shear magnitude
		p.param[1] = &p.theta; angle_param[1] = true; angle_param_exists = true;
	}
	if (!lensed_center_coords) {
		p.param[2] = &p.x_center;
		p.param[3] = &p.y_center;
	} else {
		p.param[2] = &p.xc_prime;
		p.param[3] = &p.yc_prime;
	}
	p.param[4] = &zlens;
}

void Shear::update_meta_parameters()
{
	this->update_cosmology_meta_parameters();
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (use_shear_component_params) {
		p.shear = sqrt(SQR(p.shear1) + SQR(p.shear2));
		set_angle_from_components(p.shear1,p.shear2);
	} else {
		p.theta_eff = (orient_major_axis_north) ? p.theta + M_HALFPI : p.theta;
		if (angle_points_towards_perturber) p.theta_eff -= M_HALFPI; // the phase shift is because the angle is the direction of the perturber, NOT the p.shear angle
		p.shear1 = p.shear*cos(2*p.theta_eff);
		p.shear2 = p.shear*sin(2*p.theta_eff);
	}
}

void Shear::set_auto_stepsizes()
{
	if (use_shear_component_params) {
		stepsizes[0] = 0.035;
		stepsizes[1] = 0.035;
	} else {
		stepsizes[0] = 0.03;
		stepsizes[1] = 20;
	}
	stepsizes[2] = 0.1; // very arbitrary, but p.shear is usually center_anchored anyway
	stepsizes[3] = 0.1;
	stepsizes[4] = 0.1;
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

void Shear::set_model_specific_integration_pointers()
{
	this->kapavgptr_rsq_spherical = NULL;
	this->potptr_rsq_spherical = NULL;
}

double Shear::potential(double x, double y)
{
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	return -0.5*(y*y-x*x)*p.shear1 + x*y*p.shear2;
}

void Shear::deflection(double x, double y, lensvector<double>& def)
{
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	def[0] = x*p.shear1 + y*p.shear2;
	def[1] = -y*p.shear1 + x*p.shear2;
}

void Shear::hessian(double x, double y, lensmatrix<double>& hess)
{
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// Hessian does not depend on x or y
	hess[0][0] = p.shear1;
	hess[1][1] = -hess[0][0];
	hess[0][1] = p.shear2;
	hess[1][0] = hess[0][1];
}

void Shear::potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	def[0] = x*p.shear1 + y*p.shear2;
	def[1] = -y*p.shear1 + x*p.shear2;
	hess[0][0] = p.shear1;
	hess[1][1] = -hess[0][0];
	hess[0][1] = p.shear2;
	hess[1][0] = hess[0][1];
}

void Shear::set_angle_from_components(const double &comp1, const double &comp2)
{
	Shear_Params<double>& p = assign_shear_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double angle;
	if (comp1==0) {
		if (comp2 > 0) angle = M_HALFPI;
		else angle = -M_HALFPI;
	} else {
		angle = atan(abs(comp2/comp1));
		if (comp1 < 0) {
			if (comp2 < 0)
				angle = angle - M_PI;
			else
				angle = M_PI - angle;
		} else if (comp2 < 0) {
			angle = -angle;
		}
	}
	angle /= 2;
	if (angle_points_towards_perturber) angle += M_HALFPI; // the phase shift is because the angle is the direction of the perturber, NOT the p.shear angle
	if (orient_major_axis_north) angle -= M_HALFPI;
	while (angle > M_HALFPI) angle -= M_PI;
	while (angle <= -M_HALFPI) angle += M_PI;
	this->set_angle_radians(angle);
}

/***************************** Multipole term *******************************/

Multipole::Multipole(const double zlens_in, const double zsrc_in, const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, Cosmology* cosmo_in, const bool sine)
{
	lensparams = &lensparams_mpole;
	sine_term = sine;
	setup_lens_properties(0,m_in);
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(A_m_in,n_in,m_in,theta_degrees,xc_in,yc_in,kap,sine);
}

void Multipole::initialize_parameters(const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const bool kap, const bool sine)
{
	kappa_multipole = kap; // specifies whether it is a multipole in the potential or in kappa
	sine_term = sine;
	model_name = (kap==true) ? "kmpole" : "mpole"; // rename if necessary

	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.n = n_in;
	p.A_n = A_m_in;
	set_angle(theta_degrees);
	p.x_center = xc_in;
	p.y_center = yc_in;

	update_meta_parameters();
	set_model_specific_integration_pointers();
}


void Multipole::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = MULTIPOLE;
	kappa_multipole = false; // default; specifies it is a multipole in the potential
	model_name = "mpole";
	m = subclass; // m will be used when assigning the amplitude parameter name (A_m or B_m)
	setup_base_lens_properties(6,-1,false,0,m); // number of parameters = 5, is_elliptical_lens = false
}

Multipole::Multipole(const Multipole* lens_in)
{
	lensparams = &lensparams_mpole;
	copy_base_lensdata(lens_in);
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.n = lens_in->lensparams_mpole.n;
	p.A_n = lens_in->lensparams_mpole.A_n;
	p.theta_eff = lens_in->lensparams_mpole.theta_eff;
	m = lens_in->m;
	kappa_multipole = lens_in->kappa_multipole;
	sine_term = lens_in->sine_term;

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
	if (lensed_center_coords) {
		paramnames[3] += "_l"; latex_param_subscripts[3] += ",l";
		paramnames[4] += "_l"; latex_param_subscripts[4] += ",l";
	}
}

void Multipole::assign_param_pointers()
{
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	ellipticity_paramnum = -1; // no ellipticity parameter here
	p.param[0] = &p.A_n; // here, p.A_n is actually the shear magnitude
	p.param[1] = &p.n;
	p.param[2] = &p.theta; angle_param[2] = true; angle_param_exists = true;
	if (!lensed_center_coords) {
		p.param[3] = &p.x_center;
		p.param[4] = &p.y_center;
	} else {
		p.param[3] = &p.xc_prime;
		p.param[4] = &p.yc_prime;
	}
	p.param[5] = &zlens;
}

void Multipole::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.theta_eff = (orient_major_axis_north) ? p.theta + M_HALFPI : p.theta;
	if (sine_term) p.theta_eff += M_HALFPI/m;
}

void Multipole::set_auto_stepsizes()
{
	stepsizes[0] = 0.05;
	stepsizes[1] = 0.1;
	stepsizes[2] = 20;
	stepsizes[3] = 0.1; // very arbitrary, but a multipole term is usually center_anchored anyway
	stepsizes[4] = 0.1;
	stepsizes[5] = 0.1;
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
	defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector<double>&)> (&Multipole::deflection);
	hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix<double>&)> (&Multipole::hessian);
	kapavgptr_rsq_spherical = NULL;
	potptr_rsq_spherical = NULL;
	if (m==0) kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Multipole::deflection_m0_spherical_r);
}

double Multipole::kappa(double x, double y)
{
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
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
		return p.A_n*pow(x*x+y*y,-p.n/2) * cos(m*(phi-p.theta_eff));
	} else {
		if (p.n==m) return 0;
		else {
			if (m==0)
				return -p.A_n*pow(x*x+y*y,p.n/2-1)*(0.5*(p.n*p.n-m*m)) * cos(m*(phi-p.theta_eff));
			else
				return -p.A_n*pow(x*x+y*y,p.n/2-1)*(0.5*(p.n*p.n-m*m)/m) * cos(m*(phi-p.theta_eff));
		}
	}
}

template <typename QScalar>
QScalar Multipole::kappa_rsq_impl(const QScalar rsq)
{
	Multipole_Params<QScalar>& p = assign_mpole_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	if (kappa_multipole) {
		if (m==0) return p.A_n*pow(rsq,-p.n/2);
		else return 0; // this model does not have a radial profile, unless p.n=0
	} else
		return 0;
}
template double Multipole::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Multipole::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar Multipole::kappa_rsq_deriv_impl(const QScalar rsq)
{
	Multipole_Params<QScalar>& p = assign_mpole_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	if (kappa_multipole) {
		if (m==0) return -(p.n/2)*p.A_n*pow(rsq,-p.n/2-1);
		else return 0; // this model does not have a radial profile, unless p.n=0
	} else
		return 0;
}
template double Multipole::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Multipole::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

double Multipole::potential(double x, double y)
{
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
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
		return (2*p.A_n*pow(x*x+y*y,1-p.n/2)/(SQR(2-p.n)-m*m)) * cos(m*(phi-p.theta_eff));
	} else {
		if (m==0)
			return -(p.A_n*pow(x*x+y*y,p.n/2)) * cos(m*(phi-p.theta_eff));
		else
			return -(p.A_n*pow(x*x+y*y,p.n/2)/m) * cos(m*(phi-p.theta_eff));
	}
}

void Multipole::deflection(double x, double y, lensvector<double>& def)
{
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
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
		psi = 2*p.A_n*pow(r,2-p.n)/(SQR(2-p.n)-m*m);
		dpsi = (2-p.n)*psi/r;
	} else {
		if (m==0)
			psi = -p.A_n*pow(r,p.n);
		else
			psi = -p.A_n*pow(r,p.n)/m;
		dpsi = p.n*psi/r;
	}

	cs = cos(m*(phi-p.theta_eff));
	ss = sin(m*(phi-p.theta_eff));
	def[0] = dpsi*cs*x/r + psi*m*ss*y/r/r;
	def[1] = dpsi*cs*y/r - psi*m*ss*x/r/r;
}

double Multipole::deflection_m0_spherical_r(const double r)
{
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double ans;
	if (kappa_multipole) {
		ans = 2*p.A_n*pow(r,1-p.n)/(2-p.n);
	} else {
		ans = -p.A_n*pow(r,p.n-1);
	}
	return ans;
}

void Multipole::hessian(double x, double y, lensmatrix<double>& hess)
{
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
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
		psi = 2*p.A_n*pow(r,2-p.n)/(SQR(2-p.n)-mm);
		dpsi = (2-p.n)*psi/r;
		ddpsi = (1-p.n)*dpsi/r;
	} else {
		if (m==0)
			psi = -p.A_n*pow(r,p.n);
		else
			psi = -p.A_n*pow(r,p.n)/m;
		dpsi = p.n*psi/r;
		ddpsi = (p.n-1)*dpsi/r;
	}

	cs = cos(m*(phi-p.theta_eff));
	ss = sin(m*(phi-p.theta_eff));
	hess[0][0] = (ddpsi*xx + dpsi*yy/r - psi*mm*yy/rsq)*cs/rsq + (dpsi - psi/r)*2*m*xy*ss/rcube;
	hess[1][1] = (ddpsi*yy + dpsi*xx/r - psi*mm*xx/rsq)*cs/rsq + (-dpsi + psi/r)*2*m*xy*ss/rcube;
	hess[0][1] = (ddpsi - dpsi/r + psi*mm/rsq)*xy*cs/rsq + (dpsi - psi/r)*(yy-xx)*m*ss/rcube;
	hess[1][0] = hess[0][1];
}

void Multipole::potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;

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
		psi = 2*p.A_n*pow(r,2-p.n)/(SQR(2-p.n)-mm);
		dpsi = (2-p.n)*psi/r;
	} else {
		if (m==0)
			psi = -p.A_n*pow(r,p.n);
		else
			psi = -p.A_n*pow(r,p.n)/m;
		dpsi = p.n*psi/r;
	}

	cs = cos(m*(phi-p.theta_eff));
	ss = sin(m*(phi-p.theta_eff));
	def[0] = dpsi*cs*x/r + psi*m*ss*y/r/r;
	def[1] = dpsi*cs*y/r - psi*m*ss*x/r/r;

	if (kappa_multipole) {
		ddpsi = (1-p.n)*dpsi/r;
	} else {
		ddpsi = (p.n-1)*dpsi/r;
	}

	hess[0][0] = (ddpsi*xx + dpsi*yy/r - psi*mm*yy/rsq)*cs/rsq + (dpsi - psi/r)*2*m*xy*ss/rcube;
	hess[1][1] = (ddpsi*yy + dpsi*xx/r - psi*mm*xx/rsq)*cs/rsq + (-dpsi + psi/r)*2*m*xy*ss/rcube;
	hess[0][1] = (ddpsi - dpsi/r + psi*mm/rsq)*xy*cs/rsq + (dpsi - psi/r)*(yy-xx)*m*ss/rcube;
	hess[1][0] = hess[0][1];
}

void Multipole::kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
{
	potential_derivatives(x,y,def,hess);
	kap = kappa(x,y);
}

void Multipole::get_einstein_radius(double& re_major_axis, double& re_average, const double zfactor)
{
	Multipole_Params<double>& p = assign_mpole_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// this gives the spherically averaged Einstein radius
	if (p.n==0.0) {
		re_major_axis = 0;
		re_average = 0;
		return;
	}
	double b;
	if (kappa_multipole) {
		if (p.A_n < 0) b = 0;
		else b = pow(2*p.A_n*zfactor/(2-p.n),1.0/p.n);
	} else {
		if (p.A_n > 0) b = 0;
		else {
			if (m==0) b = pow(-p.A_n*zfactor*p.n,1.0/(2-p.n));
			else b = pow(-(p.A_n*zfactor*p.n)/m,1.0/(2-p.n));
		}
	}
	re_major_axis = re_average = b;
}

/***************************** Point mass *****************************/

PointMass::PointMass(const double zlens_in, const double zsrc_in, const double &p_in, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_ptmass;
	setup_lens_properties(parameter_mode_in);
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(p_in,xc_in,yc_in);
}

void PointMass::initialize_parameters(const double &p_in, const double &xc_in, const double &yc_in)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==1) {
		p.mtot = p_in;
	} else {
		p.b = p_in;
	}
	p.x_center = xc_in;
	p.y_center = yc_in;
	update_meta_parameters();
	set_model_specific_integration_pointers();
}

void PointMass::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = PTMASS;
	model_name = "ptmass";
	setup_base_lens_properties(4,-1,false,parameter_mode); // number of parameters = 3, is_elliptical_lens = false
}

PointMass::PointMass(const PointMass* lens_in)
{
	lensparams = &lensparams_ptmass;
	copy_base_lensdata(lens_in);
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.b = lens_in->lensparams_ptmass.b;
	if (parameter_mode==1) p.mtot = lens_in->lensparams_ptmass.mtot;
	update_meta_parameters();
}

void PointMass::assign_paramnames()
{
	if (parameter_mode==1) {
		paramnames[0] = "mtot";  latex_paramnames[0] = "M"; latex_param_subscripts[0] = "";
	} else {
		paramnames[0] = "b";  latex_paramnames[0] = "b"; latex_param_subscripts[0] = "";
	}
	paramnames[1] = "xc"; latex_paramnames[1] = "x"; latex_param_subscripts[1] = "c";
	paramnames[2] = "yc"; latex_paramnames[2] = "y"; latex_param_subscripts[2] = "c";
	if (lensed_center_coords) {
		paramnames[1] += "_l"; latex_param_subscripts[1] += ",l";
		paramnames[2] += "_l"; latex_param_subscripts[2] += ",l";
	}

	paramnames[3] = "z"; latex_paramnames[3] = "z"; latex_param_subscripts[3] = "l";
}

void PointMass::assign_param_pointers()
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==1) {
		p.param[0] = &p.mtot;
	} else {
		p.param[0] = &p.b;
	}
	if (!lensed_center_coords) {
		p.param[1] = &p.x_center;
		p.param[2] = &p.y_center;
	} else {
		p.param[1] = &p.xc_prime;
		p.param[2] = &p.yc_prime;
	}
	p.param[3] = &zlens;
	ellipticity_paramnum = -1; // no ellipticity parameter here
	angle_param_exists = false; // since there is no angle parameter
}

void PointMass::set_auto_stepsizes()
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==1) {
		stepsizes[0] = 0.1*p.mtot;
	} else {
		stepsizes[0] = 0.1*p.b;
	}
	stepsizes[1] = 0.1*p.b;
	stepsizes[2] = 0.1*p.b;
	stepsizes[3] = 0.1;
}

void PointMass::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
}

void PointMass::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==1) p.b = sqrt(p.mtot/(M_PI*sigma_cr));
}

void PointMass::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&PointMass::kapavg_spherical_rsq);
	potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&PointMass::potential_spherical_rsq);
}

double PointMass::potential(double x, double y)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	return (p.b*p.b*log(sqrt(x*x+y*y)));
}

double PointMass::kappa(double x, double y)
{
	return 0; // really it's a delta function, but effectively zero for our purposes here
}

double PointMass::kapavg_spherical_rsq(const double rsq)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.b*p.b/rsq;
}

double PointMass::potential_spherical_rsq(const double rsq)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.b*p.b*log(sqrt(rsq));
}

void PointMass::deflection(double x, double y, lensvector<double>& def)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	double bsq = p.b*p.b, rsq = x*x + y*y;
	def[0] = bsq*x/rsq;
	def[1] = bsq*y/rsq;
}

void PointMass::hessian(double x, double y, lensmatrix<double>& hess)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	double bsq = p.b*p.b, xsq = x*x, ysq = y*y, r4 = SQR(xsq + ysq);
	hess[0][0] = bsq*(ysq-xsq)/r4;
	hess[1][1] = -hess[0][0];
	hess[1][0] = -2*bsq*x*y/r4;
	hess[0][1] = hess[1][0];
}

void PointMass::potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	double bsq = p.b*p.b, xsq = x*x, ysq = y*y, rsq = xsq+ysq, r4 = SQR(rsq);
	def[0] = bsq*x/rsq;
	def[1] = bsq*y/rsq;
	hess[0][0] = bsq*(ysq-xsq)/r4;
	hess[1][1] = -hess[0][0];
	hess[1][0] = -2*bsq*x*y/r4;
	hess[0][1] = hess[1][0];
}

void PointMass::get_einstein_radius(double& r1, double& r2, const double zfactor)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	r1 = p.b*sqrt(zfactor);
	r2 = r1;
}

bool PointMass::calculate_total_scaled_mass(double& total_scaled_mass)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	total_scaled_mass = M_PI*p.b*p.b;
	return true;
}

double PointMass::calculate_scaled_mass_3d(const double r)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return M_PI*p.b*p.b;
}

double PointMass::kappa_avg_r(const double r)
{
	PointMass_Params<double>& p = assign_ptmass_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return SQR(p.b/r);
}

/***************************** Core/Cusp Model *****************************/

CoreCusp::CoreCusp(const double zlens_in, const double zsrc_in, const double &mass_param_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_cc;
	setup_lens_properties(parameter_mode_in);
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(mass_param_in,gamma_in,n_in,a_in,s_in,q_in,theta_degrees,xc_in,yc_in);
}

void CoreCusp::initialize_parameters(const double &mass_param_in, const double &gamma_in, const double &n_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.gamma = gamma_in;
	p.n = n_in;
	p.a = a_in;
	p.s = s_in;
	if (p.s < 0) p.s = -p.s; // don't allow negative core radii
	if (p.a < 0) p.a = -p.a; // don't allow negative scale radii
	if (p.a < p.s) die("scale radius p.a cannot be less than core radius p.s for corecusp model");
	if (p.gamma >= p.n) die("inner slope cannot be equal to or greater than than outer slope for corecusp model");
	if (p.gamma >= 3) die("inner slope cannot be equal to or greater than 3 for corecusp model (mass diverges at r=0)");
	if (p.n <= 1) die("outer slope cannot be equal to or less than 1 for corecusp model");
	if (parameter_mode==1) {
		p.einstein_radius = mass_param_in;
		if (p.einstein_radius < 0) p.einstein_radius = -p.einstein_radius; // don't allow negative einstein radius
		p.k0 = 1.0; // This will be reset when update_meta_parameters() is called
	}
	else p.k0 = mass_param_in;
	//cout << "s=" << p.s << " " << "a=" << p.a << " " << p.gamma << " " << p.k0 << endl;

	update_meta_parameters_and_pointers();
}

void CoreCusp::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = CORECUSP;
	model_name = "corecusp";
	setup_base_lens_properties(10,5,true,parameter_mode); // number of parameters = 9, is_elliptical_lens = true
	analytic_3d_density = true;
}

CoreCusp::CoreCusp(const CoreCusp* lens_in)
{
	lensparams = &lensparams_cc;
	copy_base_lensdata(lens_in);
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.k0 = lens_in->lensparams_cc.k0;
	p.gamma = lens_in->lensparams_cc.gamma;
	p.n = lens_in->lensparams_cc.n;
	p.a = lens_in->lensparams_cc.a;
	p.s = lens_in->lensparams_cc.s;
	if (parameter_mode==1) p.einstein_radius = lens_in->lensparams_cc.einstein_radius;

	update_meta_parameters_and_pointers();
}

void CoreCusp::assign_paramnames()
{
	if (parameter_mode==1) {
		paramnames[0] = "Re"; latex_paramnames[0] = "R";       latex_param_subscripts[0] = "e";
	} else {
		paramnames[0] = "k0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	}
	paramnames[1] = "gamma"; latex_paramnames[1] = "\\gamma"; latex_param_subscripts[1] = "";
	paramnames[2] = "n";     latex_paramnames[2] = "n";       latex_param_subscripts[2] = "";
	paramnames[3] = "a";     latex_paramnames[3] = "a";       latex_param_subscripts[3] = "";
	paramnames[4] = "s";     latex_paramnames[4] = "s";       latex_param_subscripts[4] = "";
	set_geometric_paramnames(lensprofile_nparams);
}

void CoreCusp::assign_param_pointers()
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==1) p.param[0] = &p.einstein_radius;
	else p.param[0] = &p.k0;
	p.param[1] = &p.gamma;
	p.param[2] = &p.n;
	p.param[3] = &p.a;
	p.param[4] = &p.s;
	set_geometric_param_pointers(lensprofile_nparams);
}

void CoreCusp::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (p.a < p.s) die("scale radius p.a cannot be less than core radius p.s for corecusp model");
	if (p.gamma >= p.n) die("inner slope cannot be equal to or greater than than outer slope for corecusp model");
	if (p.gamma >= 3) die("inner slope cannot be equal to or greater than 3 for corecusp model (mass diverges at r=0)");
	if (p.n <= 1) die("outer slope cannot be equal to or less than 1 for corecusp model");
	p.digamma_term = DiGamma(1.5-p.gamma/2);
	double pp = (p.n-1.0)/2;
	p.beta_p1 = Beta(pp,0.5);
	p.beta_p2 = p.beta_p1/(1+1.0/(2*pp)); // Beta(p,1.5)
	if (parameter_mode==1) {
		if (p.s != 0) set_core_enclosed_mass(); else p.core_enclosed_mass = 0;
		p.k0 = p.k0 / kapavg_spherical_rsq(p.einstein_radius*p.einstein_radius);
	}
	if (p.s != 0) set_core_enclosed_mass(); else p.core_enclosed_mass = 0;
}


void CoreCusp::assign_special_anchored_parameters(LensProfile *host_in, const double factor, const bool just_created)
{
	anchor_special_parameter = true;
	special_anchor_lens = host_in;
	double rm, ravg;
	special_anchor_lens->get_einstein_radius(rm,ravg,1.0);
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==1) {
		p.a = sqrt(ravg*p.einstein_radius); // Not good! Only true for Pseudo-Jaffe subhalo. Fix this later (if it can be fixed)
		if (p.s != 0) set_core_enclosed_mass(); else p.core_enclosed_mass = 0;
		p.k0 = p.k0 / kapavg_spherical_rsq(p.einstein_radius*p.einstein_radius);
		if (p.s != 0) set_core_enclosed_mass(); else p.core_enclosed_mass = 0;
	} else {
		if (p.gamma >= 3) p.a=1e30; // effectively infinite in this case (not physical however, because the mass diverges at the center of the subhalo)
		else p.a = ravg*p.k0/(3-p.gamma); // we have ignored the core in this formula, but should be reasonable as long as p.a >> p.s
	}
	if (p.s != 0) set_core_enclosed_mass(); else p.core_enclosed_mass = 0;
}

void CoreCusp::update_special_anchored_params()
{
	if (anchor_special_parameter) {
		CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
		double rm, ravg;
		special_anchor_lens->get_einstein_radius(rm,ravg,1.0);
		if (parameter_mode==1) {
			p.a = sqrt(ravg*p.einstein_radius);
			if (p.s != 0) set_core_enclosed_mass(); else p.core_enclosed_mass = 0;
			p.k0 = p.k0 / kapavg_spherical_rsq(p.einstein_radius*p.einstein_radius);
			if (p.s != 0) set_core_enclosed_mass(); else p.core_enclosed_mass = 0;
		}
		else {
			if (p.gamma >= 3) p.a=1e30; // effectively infinite in this case (not physical however, because the mass diverges at the center of the subhalo)
			else p.a = ravg*p.k0/(3-p.gamma); // we have ignored the core in this formula, but should be reasonable as long as p.a >> p.s
		}
		if (p.s != 0) set_core_enclosed_mass(); else p.core_enclosed_mass = 0;
	}
}

void CoreCusp::set_auto_stepsizes()
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	if (parameter_mode==1) stepsizes[index++] = 0.1*p.einstein_radius;
	else stepsizes[index++] = 0.1*p.k0;
	stepsizes[index++] = 0.1;
	stepsizes[index++] = 0.1;
	stepsizes[index++] = 0.1*p.a;
	stepsizes[index++] = 0.02*p.a;
	set_geometric_param_auto_stepsizes(index);
}

void CoreCusp::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 2.99999;
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = true; penalty_lower_limits[3] = 0; penalty_upper_limits[3] = 1e30;
	set_auto_penalty_limits[4] = true; penalty_lower_limits[4] = 0; penalty_upper_limits[4] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void CoreCusp::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&CoreCusp::kapavg_spherical_rsq);
}

template <typename QScalar>
QScalar CoreCusp::kappa_rsq_impl(const QScalar rsq)
{
	CoreCusp_Params<QScalar>& p = assign_cc_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar atilde = sqrt(p.a*p.a-p.s*p.s);
	return pow(p.a/atilde,p.n) * kappa_rsq_nocore(rsq+p.s*p.s,atilde);
}
template double CoreCusp::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var CoreCusp::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar CoreCusp::kappa_rsq_nocore(const QScalar rsq_prime, const QScalar atilde)
{
	CoreCusp_Params<QScalar>& p = assign_cc_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	// Formulas for the non-cored profile are from Munoz et al. 2001
	QScalar ks, xisq, pp, hyp, ans;
	pp = (p.n-1.0)/2;
	ks = p.k0*atilde/(p.a*M_2PI);
	xisq = rsq_prime/(atilde*atilde);
#ifdef USE_STAN
	if constexpr (std::is_same_v<QScalar, stan::math::var>)
		hyp = real(hyp_2F1<double>(pp.val(),(p.gamma/2).val(),(p.n/2).val(),(1/(1+xisq)).val()));
	else
#endif
	hyp = real(hyp_2F1<double>(pp,p.gamma/2,p.n/2,1/(1+xisq)));
	ans = ks*p.beta_p1*pow(1+xisq,-pp)*hyp;
	return ans;
}
template double CoreCusp::kappa_rsq_nocore<double>(const double rsq_prime, const double atilde);
#ifdef USE_STAN
template stan::math::var CoreCusp::kappa_rsq_nocore<stan::math::var>(const stan::math::var rsq_prime, const stan::math::var atilde);
#endif

template <typename QScalar>
QScalar CoreCusp::kappa_rsq_deriv_impl(const QScalar rsq)
{
	CoreCusp_Params<QScalar>& p = assign_cc_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar atilde = sqrt(p.a*p.a-p.s*p.s);
	return pow(p.a/atilde,p.n) * kappa_rsq_deriv_nocore(rsq+p.s*p.s,atilde);
}
template double CoreCusp::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var CoreCusp::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar CoreCusp::kappa_rsq_deriv_nocore(const QScalar rsq_prime, const QScalar atilde)
{
	CoreCusp_Params<QScalar>& p = assign_cc_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	QScalar ks, xisq, hyp, ans;
	ks = p.k0*atilde/(p.a*M_2PI);
	xisq = rsq_prime/(atilde*atilde);
#ifdef USE_STAN
	if constexpr (std::is_same_v<QScalar, stan::math::var>)
		hyp = p.n*(1+xisq)*real(hyp_2F1<double>(((p.n-1.0)/2).val(),(p.gamma/2).val(),(p.n/2).val(),(1/(1+xisq)).val())) + p.gamma*real(hyp_2F1<double>(((p.n+1.0)/2).val(),((p.gamma+2.0)/2).val(),((p.n+2.0)/2).val(),(1/(1+xisq).val()))); 
	else
#endif
	hyp = p.n*(1+xisq)*real(hyp_2F1<double>((p.n-1.0)/2,p.gamma/2,p.n/2,1/(1+xisq))) + p.gamma*real(hyp_2F1<double>((p.n+1.0)/2,(p.gamma+2.0)/2,(p.n+2.0)/2,1/(1+xisq)));
	return -(ks/(2*atilde*atilde))*p.beta_p2*pow(1+xisq,-(p.n+3.0)/2)*hyp;
}
template double CoreCusp::kappa_rsq_deriv_nocore<double>(const double rsq_prime, const double atilde);
#ifdef USE_STAN
template stan::math::var CoreCusp::kappa_rsq_deriv_nocore<stan::math::var>(const stan::math::var rsq_prime, const stan::math::var atilde);
#endif

void CoreCusp::set_core_enclosed_mass()
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (p.n==3) p.core_enclosed_mass = enclosed_mass_spherical_nocore_n3(p.s*p.s,sqrt(p.a*p.a-p.s*p.s),nstep);
	else p.core_enclosed_mass = enclosed_mass_spherical_nocore(p.s*p.s,sqrt(p.a*p.a-p.s*p.s));
}

double CoreCusp::kapavg_spherical_rsq(const double rsq)
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double atilde, kapavg;
	if (p.s != 0) {
		atilde = sqrt(p.a*p.a-p.s*p.s);
		if (p.n==3) {
			kapavg = pow(1-SQR(p.s/p.a),-p.n/2) * (enclosed_mass_spherical_nocore_n3(rsq+p.s*p.s,atilde,nstep) - p.core_enclosed_mass) / rsq;
		} else {
			kapavg = pow(1-SQR(p.s/p.a),-p.n/2) * (enclosed_mass_spherical_nocore(rsq+p.s*p.s,atilde) - p.core_enclosed_mass) / rsq;
		}
	} else {
		if (p.n==3) {
			kapavg = enclosed_mass_spherical_nocore_n3(rsq,p.a,nstep) / rsq;
		}
		else {
			kapavg = enclosed_mass_spherical_nocore(rsq,p.a) / rsq;
		}
	}
	return kapavg;
}

double CoreCusp::enclosed_mass_spherical_nocore(const double rsq_prime, const double atilde, const double nprime) // actually mass_enclosed/(pi*sigma_crit)
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double xisq, pp, hyp;
	xisq = rsq_prime/(atilde*atilde);
	pp = (nprime-3.0)/2;
	hyp = pow(1+xisq,-pp) * real(hyp_2F1<double>(pp,p.gamma/2,nprime/2,1.0/(1+xisq)));

	return 2*p.k0*CUBE(atilde)/(p.a*M_2PI) * (Beta(pp,(3-p.gamma)/2) - Beta(pp,1.5)*hyp);
}

double CoreCusp::enclosed_mass_spherical_nocore_n3(const double rsq_prime, const double atilde, const double n_stepsize) // actually mass_enclosed/(pi*sigma_crit)
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// if p.gamma = 1, use GFunction2 (eq. 67 of Keeton 2001), but for very small r/p.a, use Richardson extrapolation which requires fewer iterations
	// for other values of p.gamma, use Gfunction1 (eq. 66) if r < p.a, and GFunction2 (eq. 67) if r >= p.a

	double xisq = rsq_prime/(atilde*atilde);
	if ((p.gamma == 1.0) and (xisq < 0.01)) return enclosed_mass_spherical_nocore_limit(rsq_prime,atilde,n_stepsize); // in this regime, Richardson extrapolation is faster
	double x, pp, fac;
	pp = (3-p.gamma)/2;
	if ((xisq < 1) and ((p.gamma-1.0)/2 > 1e-12)) {
		x=xisq/(1+xisq);
		fac = log(1+xisq) - G_Function(p.gamma/2,(p.gamma-1)/2,x) - Beta(-pp,1.5)*real(hyp_2F1<double>(1.5,pp,1+pp,x))*pow(x,pp); // uses Gfunction1
	} else {
		x=1.0/(1+xisq);
		fac = log(1+xisq) - G_Function(p.gamma/2,1.5,x) + digamma_three_halves - p.digamma_term; // uses Gfunction2
	}

	if (fac*0.0 != 0.0) {
		cout << "NaN deflection: a=" << atilde << " s=" << p.s << " gamma=" << p.gamma << " xisq=" << xisq << " rsq=" << rsq_prime << " atilde=" << atilde << " dig=" << p.digamma_term << " gf=" << (p.gamma-1)/2 << " " << G_Function(p.gamma/2,0.001,x) << " " << Beta(-pp,1.5) << " " << real(hyp_2F1<double>(1.5,pp,1+pp,x)) << endl;
		//print_parameters();
		//die();
	}

	return 2*p.k0*CUBE(atilde)/(p.a*M_2PI) * fac;
}

double CoreCusp::enclosed_mass_spherical_nocore_limit(const double rsq, const double atilde, const double n_stepsize)
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// This uses Richardson extrapolation to calculate the enclosed mass, which can be used for the p.n=3 case
	const double CON=1.4, CON2=(CON*CON);
	const double BIG=1.0e100;
	const double SAFE=2.0;
	const int NTAB=100;
	int i,j;
	double errt,fac,hh,ans=0.0;
	double **amat = new double*[NTAB];
	for (i=0; i < NTAB; i++) amat[i] = new double[NTAB];

	hh=n_stepsize;
	amat[0][0] = 0.5*(enclosed_mass_spherical_nocore(rsq,atilde,p.n+hh) + enclosed_mass_spherical_nocore(rsq,atilde,p.n-hh));
	double err=BIG;
	for (i=1;i<NTAB;i++) {
		hh /= CON;
		amat[0][i] = 0.5*(enclosed_mass_spherical_nocore(rsq,atilde,p.n+hh) + enclosed_mass_spherical_nocore(rsq,atilde,p.n-hh));

		fac=CON2;
		for (j=1;j<=i;j++) {
			amat[j][i]=(amat[j-1][i]*fac - amat[j-1][i-1])/(fac-1.0);
			fac=CON2*fac;
			errt=dmax(abs(amat[j][i]-amat[j-1][i]),abs(amat[j][i]-amat[j-1][i-1]));
			if (errt <= err) {
				err=errt;
				ans=amat[j][i];
			}
		}
		if (abs(amat[i][i]-amat[i-1][i-1]) >= SAFE*err) break;
	}
	for (i=0; i < NTAB; i++) delete[] amat[i];
	delete[] amat;
	return ans;
}

bool CoreCusp::output_cosmology_info(const int lens_number)
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}

	double rs_kpc, ds, m200, r200, r200_arcsec;
	rs_kpc = p.a / kpc_to_arcsec;
	ds = p.k0 * sigma_cr_kpc / rs_kpc;
	p.r200_const = 200.0*cosmo->critical_density(zlens)*1e-9/CUBE(kpc_to_arcsec)*4*M_PI/3.0;
	double (Brent::*r200root)(const double);
	r200root = static_cast<double (Brent::*)(const double)> (&CoreCusp::r200_root_eq);
	r200_arcsec = BrentsMethod(r200root, 0.1, 10000, 1e-4);
	m200 = sigma_cr*calculate_scaled_mass_3d_from_analytic_rho3d(r200_arcsec);
	r200 = r200_arcsec/kpc_to_arcsec;

	cout << "rho_0 = " << ds << " M_sun/kpc^3  (density at scale radius)" << endl;
	cout << "a = " << rs_kpc << " kpc  (scale radius)" << endl;
	cout << "M_200 = " << m200 << " M_sun\n";
	cout << "r_200 = " << r200 << " kpc\n";
	cout << endl;
	return true;
}

double CoreCusp::r200_root_eq(const double r)
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.r200_const*r*r*r - sigma_cr*calculate_scaled_mass_3d_from_analytic_rho3d(r);
}

double CoreCusp::rho3d_r_integrand_analytic(const double r)
{
	CoreCusp_Params<double>& p = assign_cc_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double rsq = r*r;
	return (p.k0/M_2PI)*pow(p.a,p.n-1)*pow(rsq+p.s*p.s,-p.gamma/2)*pow(rsq+p.a*p.a,-(p.n-p.gamma)/2);
}

/***************************** Sersic profile *****************************/

SersicLens::SersicLens(const double zlens_in, const double zsrc_in, const double &p1_in, const double &Re_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_sersic;
#ifdef USE_STAN
	lensparams_dif = &lensparams_sersic_dif;
#endif
	setup_lens_properties(parameter_mode_in);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(p1_in,Re_in,n_in,q_in,theta_degrees,xc_in,yc_in);
}

void SersicLens::initialize_parameters(const double &p1_in, const double &Re_in, const double &n_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.kappa0 = p1_in;
	} else {
		p.mstar = p1_in;
	}
	p.n = n_in;
	p.re = Re_in;

	/*
	Sersic_Params<stan::math::var>& p2 = assign_sersic_param_object<stan::math::var>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p2.kappa0 = p1_in;
	} else {
		p2.mstar = p1_in;
	}
	p2.n = n_in;
	p2.re = Re_in;
	*/




	update_meta_parameters_and_pointers();
}

void SersicLens::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = SERSIC_LENS;
	model_name = "sersic";
	setup_base_lens_properties(8,3,true,parameter_mode); // number of parameters = 7, is_elliptical_lens = true
}

SersicLens::SersicLens(const SersicLens* lens_in)
{
	lensparams = &lensparams_sersic;
	copy_base_lensdata(lens_in);
	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.kappa0 = lens_in->lensparams_sersic.kappa0;
	} else {
		p.mstar = lens_in->lensparams_sersic.mstar;
	}
	p.n = lens_in->lensparams_sersic.n;
	p.re = lens_in->lensparams_sersic.re;
	p.b = lens_in->lensparams_sersic.b;

	update_meta_parameters_and_pointers();
}

SersicLens::SersicLens(Sersic* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper)
{
	lensparams = &lensparams_sersic;
	setup_lens_properties(parameter_mode_in);
	copy_source_data_to_lens(sb_in);
	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.n = sb_in->n;
	p.re = sb_in->Reff;
	p.b = sb_in->b;
	p.kappa0 = 3; // arbitrary
	p.mstar = 1e12; // arbitrary
	set_spawned_mass_and_anchor_parameters((SB_Profile*) sb_in, vary_mass_parameter, include_limits_in, mass_param_lower,mass_param_upper);

/*
	if (vary_mass_parameter) {
		vary_params[0] = true;
		n_vary_params = 1;
		include_limits = include_limits_in;
		if (include_limits) {
			lower_limits.input(n_vary_params);
			upper_limits.input(n_vary_params);
			lower_limits[0] = mass_param_lower;
			upper_limits[0] = mass_param_upper;
			lower_limits_initial.input(lower_limits);
			upper_limits_initial.input(upper_limits);
		}
	}

	set_integration_pointers();
	set_model_specific_integration_pointers();
	// We don't update meta parameters yet because we still need to initialize the cosmology (since cosmology info couldn't be retrieved from source object)

	for (int i=1; i < n_params-1; i++) {
		// anchoring every parameter except the mass parameter (since stellar mass-to-light ratio is not known), and the redshift (since that's not a parameter in SB_Profile yet)
		anchor_parameter_to_source[i] = true;
		parameter_anchor_source[i] = (SB_Profile*) sb_in;
		parameter_anchor_paramnum[i] = i;
		parameter_anchor_ratio[i] = 1.0;
		(*p.param[i]) = *(parameter_anchor_source[i]->p.param[i]);
		at_least_one_param_anchored = true;
	}
	update_anchored_parameters();
	*/
}

void SersicLens::assign_paramnames()
{
	if (parameter_mode==0) {
		paramnames[0] = "kappa0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	} else {
		paramnames[0] = "mstar"; latex_paramnames[0] = "M"; latex_param_subscripts[0] = "*";
	}
	paramnames[1] = "Reff";   latex_paramnames[1] = "R";       latex_param_subscripts[1] = "eff";
	paramnames[2] = "n";       latex_paramnames[2] = "n";       latex_param_subscripts[2] = "";
	set_geometric_paramnames(lensprofile_nparams);
}

void SersicLens::assign_param_pointers()
{
	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.param[0] = &p.kappa0;
	} else {
		p.param[0] = &p.mstar;
	}

	p.param[1] = &p.re;
	p.param[2] = &p.n;
	set_geometric_param_pointers(lensprofile_nparams);
}

void SersicLens::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.b = 2*p.n - 0.33333333333333 + 4.0/(405*p.n) + 46.0/(25515*p.n*p.n) + 131.0/(1148175*p.n*p.n*p.n); // from Cardone 2003 (or Ciotti 1999)
	if (parameter_mode==0) {
		p.mstar = (p.kappa0*sigma_cr*p.re*p.re*M_2PI*p.n*Gamma(2*p.n)) / pow(p.b,2*p.n);
	} else {
		p.kappa0 = (p.mstar*pow(p.b,2*p.n))/(sigma_cr*p.re*p.re*M_2PI*p.n*Gamma(2*p.n));
	}

	p.def_factor = 2*p.n*p.re*p.re*p.kappa0*pow(p.b,-2*p.n);

	/*
	Sersic_Params<stan::math::var>& p2 = assign_sersic_param_object<stan::math::var>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p2.b = 2*p2.n - 0.33333333333333 + 4.0/(405*p2.n) + 46.0/(25515*p2.n*p2.n) + 131.0/(1148175*p2.n*p2.n*p2.n); // from Cardone 2003 (or Ciotti 1999)
	if (parameter_mode==0) {
		p2.mstar = (p2.kappa0*sigma_cr*p2.re*p2.re*M_2PI*p2.n*Gamma(2*p2.n.val())) / pow(p2.b,2*p2.n);
	} else {
		p2.kappa0 = (p2.mstar*pow(p2.b,2*p2.n))/(sigma_cr*p2.re*p2.re*M_2PI*p2.n*Gamma(2*p2.n.val()));
	}

	p2.def_factor = 2*p2.n*p2.re*p2.re*p2.kappa0*pow(p2.b,-2*p2.n);
	*/


}

void SersicLens::set_auto_stepsizes()
{
	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	if (parameter_mode==0) {
		stepsizes[index++] = 0.2*p.kappa0;
	} else {
		stepsizes[index++] = 0.2*p.mstar;
	}
	stepsizes[index++] = 0.2*p.re;
	stepsizes[index++] = 0.2;
	set_geometric_param_auto_stepsizes(index);
}

void SersicLens::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = false;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void SersicLens::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&SersicLens::kapavg_spherical_rsq);
}

template <typename QScalar>
QScalar SersicLens::kappa_rsq_impl(const QScalar rsq)
{
	using std::exp;
	using std::pow;
#ifdef USE_STAN
	using stan::math::exp;
	using stan::math::pow;
#endif
#ifdef USE_STAN
	if constexpr (std::is_same_v<QScalar, stan::math::var>) cout << "SersBARG0" << endl;
#endif
	Sersic_Params<QScalar>& p = assign_sersic_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
#ifdef USE_STAN
	if constexpr (std::is_same_v<QScalar, stan::math::var>) cout << "SersBARG1" << endl;
	QScalar dump = p.kappa0;
	if constexpr (std::is_same_v<QScalar, stan::math::var>) cout << "SersBARG2" << endl;
	QScalar dump2;
	if constexpr (std::is_same_v<QScalar, stan::math::var>) dump2 = stan::math::exp(5.0);
	if constexpr (std::is_same_v<QScalar, stan::math::var>) cout << "SersBARG2b" << endl;

	if constexpr (std::is_same_v<QScalar, stan::math::var>) {
		cout << "BVALUE: " << lensparams_sersic_dif.b << endl;
		dump2 = stan::math::exp(lensparams_sersic_dif.b);
	}
	else dump2 = std::exp(p.b);
	if constexpr (std::is_same_v<QScalar, stan::math::var>) cout << "SersBARG3" << endl;
	QScalar dump3 = pow(rsq/(p.re*p.re),p.n);
	if constexpr (std::is_same_v<QScalar, stan::math::var>) cout << "SersBARG4" << endl;
	QScalar dump4 = pow(rsq/(p.re*p.re),0.5/p.n);
	if constexpr (std::is_same_v<QScalar, stan::math::var>) cout << "SersBARG5" << endl;
#endif
	return p.kappa0*exp(-p.b*pow(rsq/(p.re*p.re),0.5/p.n));
	//p.kappa0 = kappa_e*exp(p.b);
}
template double SersicLens::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var SersicLens::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar SersicLens::kappa_rsq_deriv_impl(const QScalar rsq)
{
	Sersic_Params<QScalar>& p = assign_sersic_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return -p.kappa0*exp(-p.b*pow(rsq/(p.re*p.re),0.5/p.n))*p.b*pow(p.re,-1.0/p.n)*pow(rsq,0.5/p.n-1)/(2*p.n);
}
template double SersicLens::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var SersicLens::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

double SersicLens::kapavg_spherical_rsq(const double rsq)
{
	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// Formula from Cardone et al. 2003
	double x, alpha_e_times_2re, gamm2n, incgam2n;
	x = pow(rsq/(p.re*p.re),1.0/(2*p.n));
	IncGammaP_and_Gamma(2*p.n,p.b*x,incgam2n,gamm2n);
	return p.def_factor*gamm2n*incgam2n/rsq;  // p.def_factor is equal to 2*p.re*alpha_re/Gamma(2n), where alpha_re is the deflection at p.re
}

bool SersicLens::output_cosmology_info(const int lens_number)
{
	Sersic_Params<double>& p = assign_sersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}

	if (parameter_mode==0) {
		cout << "Total stellar mass = " << p.mstar << endl;
	} else {
		cout << "kappa(r=0) = " << p.kappa0 << endl;
	}
	double Reff_kpc = p.re/kpc_to_arcsec;
	cout << "R_eff = " << Reff_kpc << " kpc" << endl;
	cout << endl;
	return true;
}

/***************************** Double Sersic profile *****************************/

DoubleSersicLens::DoubleSersicLens(const double zlens_in, const double zsrc_in, const double &p1_in, const double &delta_k_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_dsersic;
	setup_lens_properties(parameter_mode_in);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(p1_in,delta_k_in,Reff1_in,n1_in,Reff2_in,n2_in,q_in,theta_degrees,xc_in,yc_in);
}

void DoubleSersicLens::initialize_parameters(const double &p1_in, const double &delta_k_in, const double &Reff1_in, const double &n1_in, const double &Reff2_in, const double &n2_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	DoubleSersic_Params<double>& p = assign_dsersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.kappa0 = p1_in;
	} else {
		p.mstar = p1_in;
	}
	p.delta_k = delta_k_in;
	p.n1 = n1_in;
	p.Reff1 = Reff1_in;
	p.n2 = n2_in;
	p.Reff2 = Reff2_in;

	update_meta_parameters_and_pointers();
}

void DoubleSersicLens::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = DOUBLE_SERSIC_LENS;
	model_name = "dsersic";
	setup_base_lens_properties(11,6,true,parameter_mode); // number of parameters = 10 (not including redshift), is_elliptical_lens = true
}

DoubleSersicLens::DoubleSersicLens(const DoubleSersicLens* lens_in)
{
	lensparams = &lensparams_dsersic;
	copy_base_lensdata(lens_in);
	DoubleSersic_Params<double>& p = assign_dsersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.kappa0 = lens_in->lensparams_dsersic.kappa0;
	} else {
		p.mstar = lens_in->lensparams_dsersic.mstar;
	}
	p.delta_k = lens_in->lensparams_dsersic.delta_k;
	p.n1 = lens_in->lensparams_dsersic.n1;
	p.Reff1 = lens_in->lensparams_dsersic.Reff1;
	p.n2 = lens_in->lensparams_dsersic.n2;
	p.Reff2 = lens_in->lensparams_dsersic.Reff2;

	update_meta_parameters_and_pointers();
}

DoubleSersicLens::DoubleSersicLens(DoubleSersic* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper)
{
	lensparams = &lensparams_dsersic;
	setup_lens_properties(parameter_mode_in);
	copy_source_data_to_lens(sb_in);
	DoubleSersic_Params<double>& p = assign_dsersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.delta_k = sb_in->delta_s;
	p.n1 = sb_in->n1;
	p.n2 = sb_in->n2;
	p.Reff1 = sb_in->Reff1;
	p.Reff2 = sb_in->Reff2;
	p.b1 = sb_in->b1;
	p.b2 = sb_in->b2;
	p.kappa0 = 3; // arbitrary
	p.mstar = 1e12; // arbitrary
	set_spawned_mass_and_anchor_parameters((SB_Profile*) sb_in, vary_mass_parameter, include_limits_in, mass_param_lower,mass_param_upper);

	/*
	if (vary_mass_parameter) {
		vary_params[0] = true;
		n_vary_params = 1;
		include_limits = include_limits_in;
		if (include_limits) {
			lower_limits.input(n_vary_params);
			upper_limits.input(n_vary_params);
			lower_limits[0] = mass_param_lower;
			upper_limits[0] = mass_param_upper;
			lower_limits_initial.input(lower_limits);
			upper_limits_initial.input(upper_limits);
		}
	}

	set_integration_pointers();
	set_model_specific_integration_pointers();
	// We don't update meta parameters yet because we still need to initialize the cosmology (since cosmology info couldn't be retrieved from source object)

	for (int i=1; i < n_params-1; i++) {
		// anchoring every parameter except the mass parameter (since stellar mass-to-light ratio is not known), and the redshift (since that's not a parameter in SB_Profile yet)
		anchor_parameter_to_source[i] = true;
		parameter_anchor_source[i] = (SB_Profile*) sb_in;
		parameter_anchor_paramnum[i] = i;
		parameter_anchor_ratio[i] = 1.0;
		(*p.param[i]) = *(parameter_anchor_source[i]->p.param[i]);
		at_least_one_param_anchored = true;
	}
	update_anchored_parameters();
	*/	
}

void DoubleSersicLens::assign_paramnames()
{
	if (parameter_mode==0) {
		paramnames[0] = "kappa0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	} else {
		paramnames[0] = "mstar"; latex_paramnames[0] = "M"; latex_param_subscripts[0] = "*";
	}
	paramnames[1] = "delta_k"; latex_paramnames[1] = "\\Delta"; latex_param_subscripts[1] = "\\kappa";
	paramnames[2] = "Reff1"; latex_paramnames[2] = "R"; latex_param_subscripts[2] = "eff,1";
	paramnames[3] = "n1"; latex_paramnames[3] = "n"; latex_param_subscripts[3] = "1";
	paramnames[4] = "Reff2"; latex_paramnames[4] = "R"; latex_param_subscripts[4] = "eff,2";
	paramnames[5] = "n2"; latex_paramnames[5] = "n"; latex_param_subscripts[5] = "2";

	set_geometric_paramnames(lensprofile_nparams);
}

void DoubleSersicLens::assign_param_pointers()
{
	DoubleSersic_Params<double>& p = assign_dsersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.param[0] = &p.kappa0;
	} else {
		p.param[0] = &p.mstar;
	}
	p.param[1] = &p.delta_k;
	p.param[2] = &p.Reff1;
	p.param[3] = &p.n1;
	p.param[4] = &p.Reff2;
	p.param[5] = &p.n2;
	set_geometric_param_pointers(lensprofile_nparams);
}

void DoubleSersicLens::update_meta_parameters()
{
	DoubleSersic_Params<double>& p = assign_dsersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	p.b1 = 2*p.n1 - 0.33333333333333 + 4.0/(405*p.n1) + 46.0/(25515*p.n1*p.n1) + 131.0/(1148175*p.n1*p.n1*p.n1);
	p.b2 = 2*p.n2 - 0.33333333333333 + 4.0/(405*p.n2) + 46.0/(25515*p.n2*p.n2) + 131.0/(1148175*p.n2*p.n2*p.n2);
	if (parameter_mode==0) {
		p.mstar = M_PI*sigma_cr*p.kappa0*((1+p.delta_k)*p.Reff1*p.Reff1*p.n1*Gamma(2*p.n1)*pow(p.b1,-2*p.n1) + (1-p.delta_k)*p.Reff2*p.Reff2*p.n2*Gamma(2*p.n2)*pow(p.b2,-2*p.n2));
	} else {
		p.kappa0 = p.mstar / (M_PI*sigma_cr*((1+p.delta_k)*p.Reff1*p.Reff1*p.n1*Gamma(2*p.n1)*pow(p.b1,-2*p.n1) + (1-p.delta_k)*p.Reff2*p.Reff2*p.n2*Gamma(2*p.n2)*pow(p.b2,-2*p.n2)));
	}
	p.kappa0_1 = p.kappa0*(1+p.delta_k)/2;
	p.kappa0_2 = p.kappa0*(1-p.delta_k)/2;
	update_ellipticity_meta_parameters();
}

void DoubleSersicLens::set_auto_stepsizes()
{
	DoubleSersic_Params<double>& p = assign_dsersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	if (parameter_mode==0) {
		stepsizes[index++] = 0.2*p.kappa0;
	} else {
		stepsizes[index++] = 0.2*p.mstar;
	}
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.2*p.Reff1; // arbitrary
	stepsizes[index++] = 0.2; // arbitrary
	stepsizes[index++] = 0.2*p.Reff2; // arbitrary
	stepsizes[index++] = 0.2; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void DoubleSersicLens::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = false; penalty_lower_limits[1] = -1e30; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = 0; penalty_upper_limits[2] = 1e30;
	set_auto_penalty_limits[3] = true; penalty_lower_limits[3] = 0; penalty_upper_limits[3] = 1e30;
	set_auto_penalty_limits[4] = true; penalty_lower_limits[4] = 0; penalty_upper_limits[4] = 1e30;
	set_auto_penalty_limits[5] = true; penalty_lower_limits[5] = 0; penalty_upper_limits[5] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

template <typename QScalar>
QScalar DoubleSersicLens::kappa_rsq_impl(const QScalar rsq)
{
	DoubleSersic_Params<QScalar>& p = assign_dsersic_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return (p.kappa0_1*exp(-p.b1*pow(rsq/(p.Reff1*p.Reff1),0.5/p.n1)) + p.kappa0_2*exp(-p.b2*pow(rsq/(p.Reff2*p.Reff2),0.5/p.n2)));
}
template double DoubleSersicLens::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var DoubleSersicLens::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar DoubleSersicLens::kappa_rsq_deriv_impl(const QScalar rsq)
{
	DoubleSersic_Params<QScalar>& p = assign_dsersic_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return -(p.kappa0_1*exp(-p.b1*pow(rsq/(p.Reff1*p.Reff1),0.5/p.n1))*p.b1*pow(p.Reff1,-1.0/p.n1)*pow(rsq,0.5/p.n1-1)/(2*p.n1) + p.kappa0_2*exp(-p.b2*pow(rsq/(p.Reff2*p.Reff2),0.5/p.n2))*p.b2*pow(p.Reff2,-1.0/p.n2)*pow(rsq,0.5/p.n2-1)/(2*p.n2));
}
template double DoubleSersicLens::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var DoubleSersicLens::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

bool DoubleSersicLens::output_cosmology_info(const int lens_number)
{
	DoubleSersic_Params<double>& p = assign_dsersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}

	if (parameter_mode==0) {
		cout << "Total stellar mass = " << p.mstar << endl;
	} else {
		cout << "kappa(r=0) = " << p.kappa0 << endl;
	}
	cout << "kappa1(r=0) = " << p.kappa0_1 << endl;
	cout << "kappa2(r=0) = " << p.kappa0_2 << endl;
	double mstar1 = M_PI*sigma_cr*p.kappa0*(1+p.delta_k)*p.Reff1*p.Reff1*p.n1*Gamma(2*p.n1)*pow(p.b1,-2*p.n1);
	double mstar2 = M_PI*sigma_cr*p.kappa0*(1-p.delta_k)*p.Reff2*p.Reff2*p.n2*Gamma(2*p.n2)*pow(p.b2,-2*p.n2);
	cout << "mstar1 = " << mstar1 << endl;
	cout << "mstar2 = " << mstar2 << endl;

	double Reff1_kpc = p.Reff1/kpc_to_arcsec;
	double Reff2_kpc = p.Reff2/kpc_to_arcsec;
	cout << "R_eff1 = " << Reff1_kpc << " kpc" << endl;
	cout << "R_eff2 = " << Reff2_kpc << " kpc" << endl;
	cout << endl;
	return true;
}

/***************************** Cored Sersic profile *****************************/

Cored_SersicLens::Cored_SersicLens(const double zlens_in, const double zsrc_in, const double &p1_in, const double &Re_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int parameter_mode_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_csersic;
	setup_lens_properties(parameter_mode_in);

	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(p1_in,Re_in,n_in,rc_in,q_in,theta_degrees,xc_in,yc_in);
}

void Cored_SersicLens::initialize_parameters(const double &p1_in, const double &Re_in, const double &n_in, const double &rc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);

	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.kappa0 = p1_in;
	} else {
		p.mstar = p1_in;
	}
	p.n = n_in;
	p.re = Re_in;
	p.rc = rc_in;

	update_meta_parameters_and_pointers();
}

void Cored_SersicLens::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = CORED_SERSIC_LENS;
	model_name = "csersic";
	setup_base_lens_properties(9,4,true,parameter_mode); // number of parameters = 7, is_elliptical_lens = true
}

Cored_SersicLens::Cored_SersicLens(const Cored_SersicLens* lens_in)
{
	lensparams = &lensparams_csersic;
	copy_base_lensdata(lens_in);
	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.kappa0 = lens_in->lensparams_csersic.kappa0;
	} else {
		p.mstar = lens_in->lensparams_csersic.mstar;
	}
	p.n = lens_in->lensparams_csersic.n;
	p.re = lens_in->lensparams_csersic.re;
	p.b = lens_in->lensparams_csersic.b;
	p.rc = lens_in->lensparams_csersic.rc;

	update_meta_parameters_and_pointers();
}

Cored_SersicLens::Cored_SersicLens(Cored_Sersic* sb_in, const int parameter_mode_in, const bool vary_mass_parameter, const bool include_limits_in, const double mass_param_lower, const double mass_param_upper)
{
	lensparams = &lensparams_csersic;
	setup_lens_properties(parameter_mode_in);
	copy_source_data_to_lens(sb_in);
	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.n = sb_in->n;
	p.re = sb_in->Reff;
	p.b = sb_in->b;
	p.rc = sb_in->rc;
	p.kappa0 = 3; // arbitrary
	p.mstar = 1e12; // arbitrary
	set_spawned_mass_and_anchor_parameters((SB_Profile*) sb_in, vary_mass_parameter, include_limits_in, mass_param_lower,mass_param_upper);

	/*
	if (vary_mass_parameter) {
		vary_params[0] = true;
		n_vary_params = 1;
		include_limits = include_limits_in;
		if (include_limits) {
			lower_limits.input(n_vary_params);
			upper_limits.input(n_vary_params);
			lower_limits[0] = mass_param_lower;
			upper_limits[0] = mass_param_upper;
			lower_limits_initial.input(lower_limits);
			upper_limits_initial.input(upper_limits);
		}
	}

	set_integration_pointers();
	set_model_specific_integration_pointers();
	// We don't update meta parameters yet because we still need to initialize the cosmology (since cosmology info couldn't be retrieved from source object)

	for (int i=1; i < n_params-1; i++) {
		// anchoring every parameter except the mass parameter (since stellar mass-to-light ratio is not known), and the redshift (since that's not a parameter in SB_Profile yet)
		anchor_parameter_to_source[i] = true;
		parameter_anchor_source[i] = (SB_Profile*) sb_in;
		parameter_anchor_paramnum[i] = i;
		parameter_anchor_ratio[i] = 1.0;
		(*p.param[i]) = *(parameter_anchor_source[i]->p.param[i]);
		at_least_one_param_anchored = true;
	}
	update_anchored_parameters();
	*/
}

void Cored_SersicLens::assign_paramnames()
{
	if (parameter_mode==0) {
		paramnames[0] = "kappa0"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "0";
	} else {
		paramnames[0] = "mstar"; latex_paramnames[0] = "M"; latex_param_subscripts[0] = "*";
	}
	paramnames[1] = "Reff";   latex_paramnames[1] = "R";       latex_param_subscripts[1] = "eff";
	paramnames[2] = "n";       latex_paramnames[2] = "n";       latex_param_subscripts[2] = "";
	paramnames[3] = "rc";       latex_paramnames[3] = "r";       latex_param_subscripts[3] = "c";
	set_geometric_paramnames(lensprofile_nparams);
}

void Cored_SersicLens::assign_param_pointers()
{
	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (parameter_mode==0) {
		p.param[0] = &p.kappa0;
	} else {
		p.param[0] = &p.mstar;
	}

	p.param[1] = &p.re;
	p.param[2] = &p.n;
	p.param[3] = &p.rc;
	set_geometric_param_pointers(lensprofile_nparams);
}

void Cored_SersicLens::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.b = 2*p.n - 0.33333333333333 + 4.0/(405*p.n) + 46.0/(25515*p.n*p.n) + 131.0/(1148175*p.n*p.n*p.n);
	if (parameter_mode==0) {
		p.mstar = (p.kappa0*sigma_cr*p.re*p.re*M_2PI*p.n*Gamma(2*p.n)) / pow(p.b,2*p.n);
	} else {
		p.kappa0 = (p.mstar*pow(p.b,2*p.n))/(sigma_cr*p.re*p.re*M_2PI*p.n*Gamma(2*p.n));
	}
	p.def_factor = 2*p.n*p.re*p.re*p.kappa0*pow(p.b,-2*p.n);
}

void Cored_SersicLens::set_auto_stepsizes()
{
	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	int index = 0;
	if (parameter_mode==0) {
		stepsizes[index++] = 0.2*p.kappa0;
	} else {
		stepsizes[index++] = 0.2*p.mstar;
	}
	stepsizes[index++] = 0.2*p.re;
	stepsizes[index++] = 0.2;
	stepsizes[index++] = 0.05*p.re;
	set_geometric_param_auto_stepsizes(index);
}

void Cored_SersicLens::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = true; penalty_lower_limits[3] = 0; penalty_upper_limits[3] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void Cored_SersicLens::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&Cored_SersicLens::kapavg_spherical_rsq);
}

template <typename QScalar>
QScalar Cored_SersicLens::kappa_rsq_impl(const QScalar rsq)
{
	Cored_Sersic_Params<QScalar>& p = assign_csersic_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return p.kappa0*exp(-p.b*pow((rsq+p.rc*p.rc)/(p.re*p.re),0.5/p.n));
}
template double Cored_SersicLens::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Cored_SersicLens::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar Cored_SersicLens::kappa_rsq_deriv_impl(const QScalar rsq)
{
	Cored_Sersic_Params<QScalar>& p = assign_csersic_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return -p.kappa0*exp(-p.b*pow((rsq+p.rc*p.rc)/(p.re*p.re),0.5/p.n))*p.b*pow(p.re,-1.0/p.n)*pow((rsq+p.rc*p.rc),0.5/p.n-1)/(2*p.n);
}
template double Cored_SersicLens::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var Cored_SersicLens::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

double Cored_SersicLens::kapavg_spherical_rsq(const double rsq)
{
	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	// WARNING! This is a hack and incorrect for p.rc not equal to zero! At some point, derive the correct formula!
	// Formula from Cardone et al. 2003
	double x, alpha_e_times_2re, gamm2n, incgam2n;
	x = pow((rsq+p.rc*p.rc)/(p.re*p.re),1.0/(2*p.n));
	IncGammaP_and_Gamma(2*p.n,p.b*x,incgam2n,gamm2n);
	return p.def_factor*gamm2n*incgam2n/rsq;  // def_factor is equal to 2*p.re*alpha_re/Gamma(2n), where alpha_re is the deflection at p.re
}

bool Cored_SersicLens::output_cosmology_info(const int lens_number)
{
	Cored_Sersic_Params<double>& p = assign_csersic_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (lens_number != -1) cout << "Lens " << lens_number << ":\n";
	double sigma_cr_kpc = sigma_cr*SQR(kpc_to_arcsec);
	if (zlens != qlens->lens_redshift) {
		cout << resetiosflags(ios::scientific);
		cout << "sigma_crit(z=" << zlens << "): ";
		if (qlens->use_scientific_notation) cout << setiosflags(ios::scientific);
		cout << sigma_cr_kpc << " Msun/kpc^2 (" << sigma_cr << " Msun/arcsec^2)" << endl;
	}

	if (parameter_mode==0) {
		cout << "Total stellar mass = " << p.mstar << endl;
	} else {
		cout << "kappa(r=0) = " << p.kappa0 << endl;
	}
	double Reff_kpc = p.re/kpc_to_arcsec;
	double rc_kpc = p.rc/kpc_to_arcsec;
	cout << "R_eff = " << Reff_kpc << " kpc" << endl;
	cout << "p.rc = " << rc_kpc << " kpc" << endl;
	cout << endl;
	return true;
}

/***************************** Mass sheet *****************************/

MassSheet::MassSheet(const double zlens_in, const double zsrc_in, const double &kext_in, const double &xc_in, const double &yc_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_sheet;
	setup_lens_properties();
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(kext_in,xc_in,yc_in);
}

void MassSheet::initialize_parameters(const double &kext_in, const double &xc_in, const double &yc_in)
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.kext = kext_in;
	p.x_center = xc_in;
	p.y_center = yc_in;
	update_meta_parameters_and_pointers();
}

void MassSheet::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = SHEET;
	model_name = "sheet";
	setup_base_lens_properties(4,-1,false); // number of parameters = 3, is_elliptical_lens = false
}

MassSheet::MassSheet(const MassSheet* lens_in)
{
	lensparams = &lensparams_sheet;
	copy_base_lensdata(lens_in);
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.kext = lens_in->lensparams_sheet.kext;
	update_meta_parameters_and_pointers();
}

void MassSheet::assign_paramnames()
{
	paramnames[0] = "kext"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "ext";
	paramnames[1] = "xc";   latex_paramnames[1] = "x";       latex_param_subscripts[1] = "c";
	paramnames[2] = "yc";   latex_paramnames[2] = "y";       latex_param_subscripts[2] = "c";
	paramnames[3] = "z";   latex_paramnames[3] = "z";       latex_param_subscripts[3] = "l";
	if (lensed_center_coords) {
		paramnames[1] += "_l"; latex_param_subscripts[1] += ",l";
		paramnames[2] += "_l"; latex_param_subscripts[2] += ",l";
	}
}

void MassSheet::assign_param_pointers()
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.param[0] = &p.kext;
	if (!lensed_center_coords) {
		p.param[1] = &p.x_center;
		p.param[2] = &p.y_center;
	} else {
		p.param[1] = &p.xc_prime;
		p.param[2] = &p.yc_prime;
	}
	p.param[3] = &zlens;
	ellipticity_paramnum = -1; // no ellipticity parameter here
	angle_param_exists = false; // since there is no angle parameter
}

void MassSheet::set_auto_stepsizes()
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	stepsizes[0] = (p.kext==0) ? 0.1 : 0.3*p.kext;
	stepsizes[1] = 0.1; // arbitrary! really, the center should never be independently varied
	stepsizes[2] = 0.1;
	stepsizes[3] = 0.1;
}

void MassSheet::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = false;
}

void MassSheet::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&MassSheet::kapavg_spherical_rsq);
	potptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&MassSheet::potential_spherical_rsq);
}

double MassSheet::potential(double x, double y)
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	return (p.kext*(x*x+y*y)/2.0);
}

double MassSheet::kappa(double x, double y)
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.kext;
}

template <typename QScalar>
QScalar MassSheet::kappa_rsq_impl(const QScalar rsq)
{
	MassSheet_Params<QScalar>& p = assign_sheet_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return p.kext;
}
template double MassSheet::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var MassSheet::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar MassSheet::kappa_rsq_deriv_impl(const QScalar rsq)
{
	MassSheet_Params<QScalar>& p = assign_sheet_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	return 0.0;
}
template double MassSheet::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var MassSheet::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

double MassSheet::kapavg_spherical_rsq(const double rsq)
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.kext;
}

double MassSheet::potential_spherical_rsq(const double rsq)
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.kext*rsq/2.0;
}

void MassSheet::deflection(double x, double y, lensvector<double>& def)
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	def[0] = p.kext*x;
	def[1] = p.kext*y;
}

void MassSheet::hessian(double x, double y, lensmatrix<double>& hess)
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	hess[0][0] = p.kext;
	hess[1][1] = p.kext;
	hess[1][0] = 0;
	hess[0][1] = 0;
}

void MassSheet::potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	MassSheet_Params<double>& p = assign_sheet_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	x -= p.x_center;
	y -= p.y_center;
	def[0] = p.kext*x;
	def[1] = p.kext*y;
	hess[0][0] = p.kext;
	hess[1][1] = p.kext;
	hess[1][0] = 0;
	hess[0][1] = 0;
}

/***************** External deflection (only relevant if multiple source redshifts) *****************/

Deflection::Deflection(const double zlens_in, const double zsrc_in, const double &defx_in, const double &defy_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_defl;
	setup_lens_properties();
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(defx_in,defy_in);
}

void Deflection::initialize_parameters(const double &defx_in, const double &defy_in)
{
	Deflection_Params<double>& p = assign_defl_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.def_x = defx_in;
	p.def_y = defy_in;
	p.x_center = 0; p.y_center = 0; // will not be used anyway
	update_meta_parameters_and_pointers();
}

void Deflection::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = DEFLECTION;
	model_name = "deflection";
	setup_base_lens_properties(3,-1,false); // number of parameters = 2, is_elliptical_lens = false
	center_defined = false;
}

Deflection::Deflection(const Deflection* lens_in)
{
	lensparams = &lensparams_defl;
	copy_base_lensdata(lens_in);
	Deflection_Params<double>& p = assign_defl_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.def_x = lens_in->lensparams_defl.def_x;
	p.def_y = lens_in->lensparams_defl.def_y;
	update_meta_parameters_and_pointers();
}

void Deflection::assign_paramnames()
{
	paramnames[0] = "def_x"; latex_paramnames[0] = "\\alpha"; latex_param_subscripts[0] = "x";
	paramnames[1] = "def_y"; latex_paramnames[1] = "\\alpha"; latex_param_subscripts[1] = "y";
	paramnames[2] = "z";   latex_paramnames[2] = "z";       latex_param_subscripts[2] = "l";
}

void Deflection::set_model_specific_integration_pointers()
{
	defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector<double>&)> (&Deflection::deflection);
	hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix<double>&)> (&Deflection::hessian);
	kapavgptr_rsq_spherical = NULL;
	potptr_rsq_spherical = NULL;
}

void Deflection::assign_param_pointers()
{
	Deflection_Params<double>& p = assign_defl_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.param[0] = &p.def_x;
	p.param[1] = &p.def_y;
	p.param[2] = &zlens;
	ellipticity_paramnum = -1; // no ellipticity parameter here
	angle_param_exists = false; // since there is no angle parameter
}

void Deflection::set_auto_stepsizes()
{
	Deflection_Params<double>& p = assign_defl_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	stepsizes[0] = (p.def_x==0) ? 0.1: 0.3*p.def_x;
	stepsizes[1] = (p.def_y==0) ? 0.1: 0.3*p.def_y;
	stepsizes[2] = 0.1;
}

void Deflection::set_auto_ranges()
{
	set_auto_penalty_limits[0] = false;
	set_auto_penalty_limits[1] = false;
	set_auto_penalty_limits[2] = false;
}

double Deflection::potential(double x, double y)
{
	Deflection_Params<double>& p = assign_defl_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	return p.def_x*x + p.def_y*y;
}

double Deflection::kappa(double x, double y)
{
	return 0; // really it's a delta function, but effectively zero for our purposes here
}

void Deflection::deflection(double x, double y, lensvector<double>& def)
{
	Deflection_Params<double>& p = assign_defl_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	def[0] = p.def_x;
	def[1] = p.def_y;
}

void Deflection::hessian(double x, double y, lensmatrix<double>& hess)
{
	hess[0][0] = 0;
	hess[1][1] = 0;
	hess[1][0] = 0;
	hess[0][1] = 0;
}

void Deflection::potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	Deflection_Params<double>& p = assign_defl_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	def[0] = p.def_x;
	def[1] = p.def_y;
	hess[0][0] = 0;
	hess[1][1] = 0;
	hess[1][0] = 0;
	hess[0][1] = 0;
}

/******************************* Tabulated Model *******************************/

/*
Tabulated_Model::Tabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, Cosmology* cosmo_in)
{
	lenstype = TABULATED;
	model_name = "tab(" + lens_in->get_model_name() + ")";
	setup_base_lens_properties(6,-1,false); // number of parameters = 3, is_elliptical_lens = false

	kscale = kscale_in;
	rscale = rscale0 = rscale_in;
	p.x_center = xc;
	p.y_center = yc;
	theta = degrees_to_radians(theta_in);
	update_meta_parameters_and_pointers();
	lens_in->set_theta(0);
	lens_in->set_center(0,0); // we're going to delete the original lens anyway, so it doesn't matter what its original angle and center position was

	double logrmin, logrmax;
	logrmin = log(rmin);
	logrmax = log(rmax);

	grid_logr_N = logr_N;
	grid_phi_N = phi_N;
	grid_logrlength = logrmax-logrmin;
	grid_logrvals = new double[logr_N];
	grid_phivals = new double[phi_N];

	kappa_vals = new double*[logr_N];
	pot_vals = new double*[logr_N];
	defx = new double*[logr_N];
	defy = new double*[logr_N];
	hess_xx = new double*[logr_N];
	hess_yy = new double*[logr_N];
	hess_xy = new double*[logr_N];
	int i,j;
	for (i=0; i < logr_N; i++) {
		kappa_vals[i] = new double[phi_N];
		pot_vals[i] = new double[phi_N];
		defx[i] = new double[phi_N];
		defy[i] = new double[phi_N];
		hess_xx[i] = new double[phi_N];
		hess_yy[i] = new double[phi_N];
		hess_xy[i] = new double[phi_N];
	}

	lensvector<double> def_in;
	lensmatrix<double> hess_in;
	double r,x,y;
	double logrstep = grid_logrlength/(logr_N-1);
	double phistep = M_2PI/(phi_N-1); // the final phi value will be 2*pi, which is redundant (since it's equivalent to phi=0) but it's much simpler to do it this way
	double logr, phi;
	for (i=0, logr=logrmin; i < logr_N; i++, logr += logrstep) grid_logrvals[i] = logr;
	for (j=0, phi=0; j < phi_N; j++, phi += phistep) grid_phivals[j] = phi;

	rmin_einstein_radius = exp(logrmin);
	rmax_einstein_radius = exp(logrmax);

	for (i=0, logr=logrmin; i < logr_N; i++, logr += logrstep) {
		for (j=0, phi=0; j < phi_N; j++, phi += phistep) {
			r = exp(logr);
			x = r*cos(phi);
			y = r*sin(phi);
			pot_vals[i][j] = lens_in->potential(x,y) / kscale;
			lens_in->kappa_and_potential_derivatives(x, y, kappa_vals[i][j], def_in, hess_in);
			kappa_vals[i][j] /= kscale;
			defx[i][j] = def_in[0] / kscale;
			defy[i][j] = def_in[1] / kscale;
			hess_xx[i][j] = hess_in[0][0] / kscale;
			hess_yy[i][j] = hess_in[1][1] / kscale;
			hess_xy[i][j] = hess_in[0][1] / kscale;
		}
	}

	// the following data are stored so this model can be reproduced later if needed
	original_kscale = kscale;
	original_rscale = rscale;
	loaded_from_file = false;
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
}

Tabulated_Model::Tabulated_Model(const Tabulated_Model* lens_in)
{
	copy_base_lensdata(lens_in);
	kscale = lens_in->kscale;
	rscale = lens_in->rscale;
	rscale0 = lens_in->rscale0;
	grid_logrlength = lens_in->grid_logrlength;
	grid_logr_N = lens_in->grid_logr_N;
	grid_phi_N = lens_in->grid_phi_N;
	grid_logrvals = new double[grid_logr_N];
	grid_phivals = new double[grid_phi_N];

	kappa_vals = new double*[grid_logr_N];
	pot_vals = new double*[grid_logr_N];
	defx = new double*[grid_logr_N];
	defy = new double*[grid_logr_N];
	hess_xx = new double*[grid_logr_N];
	hess_yy = new double*[grid_logr_N];
	hess_xy = new double*[grid_logr_N];
	int i,j;
	for (i=0; i < grid_logr_N; i++) {
		kappa_vals[i] = new double[grid_phi_N];
		pot_vals[i] = new double[grid_phi_N];
		defx[i] = new double[grid_phi_N];
		defy[i] = new double[grid_phi_N];
		hess_xx[i] = new double[grid_phi_N];
		hess_yy[i] = new double[grid_phi_N];
		hess_xy[i] = new double[grid_phi_N];
	}

	for (i=0; i < grid_logr_N; i++) grid_logrvals[i] = lens_in->grid_logrvals[i];
	for (j=0; j < grid_phi_N; j++) grid_phivals[j] = lens_in->grid_phivals[j];

	rmin_einstein_radius = lens_in->rmin_einstein_radius;
	rmax_einstein_radius = lens_in->rmax_einstein_radius;

	for (i=0; i < grid_logr_N; i++) {
		for (j=0; j < grid_phi_N; j++) {
			kappa_vals[i][j] = lens_in->kappa_vals[i][j];
			pot_vals[i][j] = lens_in->pot_vals[i][j];
			defx[i][j] = lens_in->defx[i][j];
			defy[i][j] = lens_in->defy[i][j];
			hess_xx[i][j] = lens_in->hess_xx[i][j];
			hess_yy[i][j] = lens_in->hess_yy[i][j];
			hess_xy[i][j] = lens_in->hess_xy[i][j];
		}
	}
	update_meta_parameters_and_pointers();

	loaded_from_file = lens_in->loaded_from_file;
	if (!loaded_from_file) {
		original_kscale = lens_in->original_kscale;
		original_rscale = lens_in->original_rscale;
	}
}

Tabulated_Model::Tabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double &xc, const double &yc, ifstream& tabfile, const string& tab_filename, Cosmology* cosmo_in)
{
	lenstype = TABULATED;
	setup_base_lens_properties(6,-1,false); // number of parameters = 3, is_elliptical_lens = false
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);

	kscale = kscale_in;
	rscale = rscale_in;
	theta = degrees_to_radians(theta_in);
	p.x_center = xc;
	p.y_center = yc;

	tabfile >> model_name;
	tabfile >> rscale0;
	update_meta_parameters_and_pointers();

	tabfile >> grid_logr_N;
	tabfile >> grid_phi_N;

	grid_logrvals = new double[grid_logr_N];
	grid_phivals = new double[grid_phi_N];
	kappa_vals = new double*[grid_logr_N];
	pot_vals = new double*[grid_logr_N];
	defx = new double*[grid_logr_N];
	defy = new double*[grid_logr_N];
	hess_xx = new double*[grid_logr_N];
	hess_yy = new double*[grid_logr_N];
	hess_xy = new double*[grid_logr_N];
	int i,j;
	for (i=0; i < grid_logr_N; i++) {
		kappa_vals[i] = new double[grid_phi_N];
		pot_vals[i] = new double[grid_phi_N];
		defx[i] = new double[grid_phi_N];
		defy[i] = new double[grid_phi_N];
		hess_xx[i] = new double[grid_phi_N];
		hess_yy[i] = new double[grid_phi_N];
		hess_xy[i] = new double[grid_phi_N];
	}

	for (i=0; i < grid_logr_N; i++) tabfile >> grid_logrvals[i];
	for (j=0; j < grid_phi_N; j++) tabfile >> grid_phivals[j];

	rmin_einstein_radius = exp(2*grid_logrvals[0]);
	rmax_einstein_radius = exp(grid_logrvals[grid_logr_N-1]);

	for (i=0; i < grid_logr_N; i++) {
		for (j=0; j < grid_phi_N; j++) {
			tabfile >> kappa_vals[i][j] >> pot_vals[i][j] >> defx[i][j] >> defy[i][j] >> hess_xx[i][j] >> hess_yy[i][j] >> hess_xy[i][j];
		}
	}
	grid_logrlength = grid_logrvals[grid_logr_N-1] - grid_logrvals[0];
	loaded_from_file = true;
}

void Tabulated_Model::output_tables(const string tabfile_root)
{
	string tabfilename = tabfile_root + ".tab";
	ofstream tabfile(tabfilename.c_str());
	tabfile << model_name << " " << rscale0 << endl;
	tabfile << grid_logr_N << " " << grid_phi_N << endl << endl;
	int i,j;
	for (i=0; i < grid_logr_N; i++) tabfile << grid_logrvals[i] << " ";
	tabfile << endl;
	for (j=0; j < grid_phi_N; j++) tabfile << grid_phivals[j] << " ";
	tabfile << endl << endl;
	for (i=0; i < grid_logr_N; i++) {
		for (j=0; j < grid_phi_N; j++) {
			tabfile << kappa_vals[i][j] << " ";
			tabfile << pot_vals[i][j] << " ";
			tabfile << defx[i][j] << " ";
			tabfile << defy[i][j] << " ";
			tabfile << hess_xx[i][j] << " ";
			tabfile << hess_yy[i][j] << " ";
			tabfile << hess_xy[i][j];
			tabfile << endl;
		}
	}
	tabfile.close();
}

void Tabulated_Model::assign_paramnames()
{
	paramnames[0] = "kscale"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rscale";  latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	paramnames[2] = "theta";  latex_paramnames[2] = "\\theta"; latex_param_subscripts[2] = "";
	paramnames[3] = "xc";     latex_paramnames[3] = "x";       latex_param_subscripts[3] = "c";
	paramnames[4] = "yc";     latex_paramnames[4] = "y";       latex_param_subscripts[4] = "c";
	if (lensed_center_coords) {
		paramnames[3] += "_l"; latex_param_subscripts[3] += ",l";
		paramnames[4] += "_l"; latex_param_subscripts[4] += ",l";
	}
}

void Tabulated_Model::assign_param_pointers()
{
	ellipticity_paramnum = -1; // no ellipticity parameter here
	p.param[0] = &kscale;
	p.param[1] = &rscale;
	p.param[2] = &theta; angle_param[2] = true; angle_param_exists = true;
	if (!lensed_center_coords) {
		p.param[3] = &p.x_center;
		p.param[4] = &p.y_center;
	} else {
		p.param[3] = &xc_prime;
		p.param[4] = &yc_prime;
	}
}

void Tabulated_Model::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	// We don't use orient_major_axis_north because this is meaningless for the tabulated model
	costheta = cos(theta);
	sintheta = sin(theta);
	rscale_factor = rscale / rscale0;
	//cout << rscale << " " << rscale0 << " " << rscale_factor << endl;
}

void Tabulated_Model::set_auto_stepsizes()
{
	stepsizes[0] = 0.3*kscale;
	stepsizes[1] = 0.3*rscale;
	stepsizes[2] = 20;
	stepsizes[3] = p.x_center;
	stepsizes[4] = p.y_center;
	stepsizes[5] = 0.1;
}

void Tabulated_Model::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0.01*kscale; penalty_upper_limits[0] = 100*kscale;  // limits are somewhat arbitrary, but we shouldn't allow k --> 0 or k very large
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = exp(grid_logrvals[0]); penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = false;
	set_auto_penalty_limits[3] = false;
	set_auto_penalty_limits[4] = false;
}

double Tabulated_Model::potential(double x, double y)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	int ival, jval;
	double tt, uu, TT, UU, interp;
	double logr, phi;
	logr = log(sqrt(x*x+y*y)/rscale_factor); // scaling r is easier than scaling the table itself
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	interp = TT*UU*pot_vals[ival][jval] + tt*UU*pot_vals[ival+1][jval]
						+ TT*uu*pot_vals[ival][jval+1] + tt*uu*pot_vals[ival+1][jval+1];
	return kscale*rscale_factor*rscale_factor*interp;
}

double Tabulated_Model::kappa(double x, double y)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	double logr, phi;
	int ival, jval;
	double tt, uu, TT, UU, interp;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}
	//cout << "logr=" << logr << ", r=" << exp(logr) << ", phi=" << phi << endl;
	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	//cout << ival << " " << jval << " " << tt << " " << uu << endl;
	//cout << kappa_vals[ival][jval] << endl;
	interp = TT*UU*kappa_vals[ival][jval] + tt*UU*kappa_vals[ival+1][jval]
						+ TT*uu*kappa_vals[ival][jval+1] + tt*uu*kappa_vals[ival+1][jval+1];
	return kscale*interp;
}

double Tabulated_Model::kappa_rsq_impl(const double rsq)
{
	double logr, phi;
	logr = log(sqrt(rsq)/rscale_factor);
	static const int phi_N = 100;
	double phistep = M_2PI/phi_N;
	int i, ival, jval;
	double tt, uu, TT, UU, interp;
	double kappa_angular_avg = 0;
	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	if (ival < 0) ival=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	TT = 1-tt;
	for (i=0, phi=0; i < phi_N; i++, phi += phistep) {
		jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
		if (jval < 0) jval=0;
		if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
		uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
		UU = 1-uu;
		kappa_angular_avg += kscale*(TT*UU*kappa_vals[ival][jval] + tt*UU*kappa_vals[ival+1][jval]
							+ TT*uu*kappa_vals[ival][jval+1] + tt*uu*kappa_vals[ival+1][jval+1]);
	}
	kappa_angular_avg /= phi_N;
	return kappa_angular_avg;
}

void Tabulated_Model::deflection(double x, double y, lensvector<double>& def)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	double logr, phi;
	int ival, jval;
	double tt, uu, TT, UU, interp;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	//cout << logr << " " << grid_logrvals[0] << " " << grid_logrlength << " " << grid_logr_N << endl;
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;
	interp = TT*UU*defx[ival][jval] + tt*UU*defx[ival+1][jval] + TT*uu*defx[ival][jval+1] + tt*uu*defx[ival+1][jval+1];
	//cout << ival << " " << jval << " " << uu << " " << tt << endl;
	def[0] = kscale*rscale_factor*interp;
	interp = TT*UU*defy[ival][jval] + tt*UU*defy[ival+1][jval] + TT*uu*defy[ival][jval+1] + tt*uu*defy[ival+1][jval+1];
	def[1] = kscale*rscale_factor*interp;
	if (sintheta != 0) def.rotate_back(costheta,sintheta);
}

void Tabulated_Model::hessian(double x, double y, lensmatrix<double>& hess)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	double logr, phi;
	int ival, jval;
	double tt, uu, TT, UU, interp;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
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

void Tabulated_Model::kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	int ival, jval;
	double tt, uu, TT, UU, interp;
	double logr, phi;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	interp = TT*UU*kappa_vals[ival][jval] + tt*UU*kappa_vals[ival+1][jval]
						+ TT*uu*kappa_vals[ival][jval+1] + tt*uu*kappa_vals[ival+1][jval+1];
	kap = kscale*interp;

	interp = TT*UU*defx[ival][jval] + tt*UU*defx[ival+1][jval] + TT*uu*defx[ival][jval+1] + tt*uu*defx[ival+1][jval+1];
	def[0] = kscale*rscale_factor*interp;
	interp = TT*UU*defy[ival][jval] + tt*UU*defy[ival+1][jval] + TT*uu*defy[ival][jval+1] + tt*uu*defy[ival+1][jval+1];
	def[1] = kscale*rscale_factor*interp;

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

void Tabulated_Model::potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	int ival, jval;
	double tt, uu, TT, UU, interp;
	double logr, phi;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	interp = TT*UU*defx[ival][jval] + tt*UU*defx[ival+1][jval] + TT*uu*defx[ival][jval+1] + tt*uu*defx[ival+1][jval+1];
	//cout << ival << " " << jval << " " << uu << " " << tt << endl;
	def[0] = kscale*rscale_factor*interp;
	interp = TT*UU*defy[ival][jval] + tt*UU*defy[ival+1][jval] + TT*uu*defy[ival][jval+1] + tt*uu*defy[ival+1][jval+1];
	def[1] = kscale*rscale_factor*interp;

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
	if (grid_logrvals != NULL) {
		delete[] grid_logrvals;
		delete[] grid_phivals;
		for (int i=0; i < grid_logr_N; i++) {
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
}
*/


/***************************** Tabulated Model that interpolates in q *****************************/
/*
QTabulated_Model::QTabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, const double qmin_in, const int q_N, Cosmology* cosmo_in)
{
	lenstype = QTABULATED;
	model_name = "qtab(" + lens_in->get_model_name() + ")";
	setup_base_lens_properties(7,-1,false); // number of parameters = 3, is_elliptical_lens = false
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	ellipticity_mode = -1;
	original_emode = lens_in->ellipticity_mode;
	// I wanted to allow q or e to be a parameter, but at present only q is allowed...fix this later

	kscale = kscale_in;
	rscale = rscale0 = rscale_in;
	q = q_in;
	p.x_center = xc;
	p.y_center = yc;
	theta = degrees_to_radians(theta_in);
	lens_in->set_theta(0);
	lens_in->set_center(0,0); // we're going to delete the original lens anyway, so it doesn't matter what its original angle and center position was

	double logrmin, logrmax;
	logrmin = log(rmin);
	logrmax = log(rmax);
	double qmin, qmax;
	if ((original_emode==0) or (original_emode==1)) {
		qmin = qmin_in; qmax = 1.0;
	} else {
		qmin = 0; qmax = 1 - qmin_in; // 'qq' here is actually epsilon = 1-qq
	}

	grid_logr_N = logr_N;
	grid_phi_N = phi_N;
	grid_q_N = q_N;
	grid_logrlength = logrmax-logrmin;
	grid_qlength = qmax-qmin;
	grid_logrvals = new double[logr_N];
	grid_phivals = new double[phi_N];
	grid_qvals = new double[q_N];

	kappa_vals = new double**[logr_N];
	pot_vals = new double**[logr_N];
	defx = new double**[logr_N];
	defy = new double**[logr_N];
	hess_xx = new double**[logr_N];
	hess_yy = new double**[logr_N];
	hess_xy = new double**[logr_N];
	int i,j,k;
	for (i=0; i < logr_N; i++) {
		kappa_vals[i] = new double*[phi_N];
		pot_vals[i] = new double*[phi_N];
		defx[i] = new double*[phi_N];
		defy[i] = new double*[phi_N];
		hess_xx[i] = new double*[phi_N];
		hess_yy[i] = new double*[phi_N];
		hess_xy[i] = new double*[phi_N];
		for (j=0; j < phi_N; j++) {
			kappa_vals[i][j] = new double[q_N];
			pot_vals[i][j] = new double[q_N];
			defx[i][j] = new double[q_N];
			defy[i][j] = new double[q_N];
			hess_xx[i][j] = new double[q_N];
			hess_yy[i][j] = new double[q_N];
			hess_xy[i][j] = new double[q_N];
		}
	}

	lensvector<double> def_in;
	lensmatrix<double> hess_in;
	double r,x,y;
	double logrstep = grid_logrlength/(logr_N-1);
	double phistep = M_2PI/(phi_N-1); // the final phi value will be 2*pi, which is redundant (since it's equivalent to phi=0) but it's much simpler to do it this way
	double qstep = (qmax-qmin)/(q_N-1);
	double logr, phi, qq;
	for (i=0, logr=logrmin; i < logr_N; i++, logr += logrstep) grid_logrvals[i] = logr;
	for (j=0, phi=0; j < phi_N; j++, phi += phistep) grid_phivals[j] = phi;
	for (k=0, qq=qmin; k < q_N; k++, qq += qstep) grid_qvals[k] = qq;

	rmin_einstein_radius = exp(logrmin);
	rmax_einstein_radius = exp(logrmax);

	for (k=0, qq=qmin; k < q_N; k++, qq += qstep) {
		if (k==q_N-1) qq=qmax; // just to enforce q=1 at the end without any machine error
		lens_in->update_ellipticity_parameter(qq);
		for (i=0, logr=logrmin; i < logr_N; i++, logr += logrstep) {
			r = exp(logr);
			for (j=0, phi=0; j < phi_N; j++, phi += phistep) {
				x = r*cos(phi);
				y = r*sin(phi);
				pot_vals[i][j][k] = lens_in->potential(x,y) / kscale;
				lens_in->kappa_and_potential_derivatives(x, y, kappa_vals[i][j][k], def_in, hess_in);
				kappa_vals[i][j][k] /= kscale;
				defx[i][j][k] = def_in[0] / kscale;
				defy[i][j][k] = def_in[1] / kscale;
				hess_xx[i][j][k] = hess_in[0][0] / kscale;
				hess_yy[i][j][k] = hess_in[1][1] / kscale;
				hess_xy[i][j][k] = hess_in[0][1] / kscale;
			}
		}
		cout << "Row " << k << " (q=" << qq << ") done...\n" << flush;
	}
	update_meta_parameters_and_pointers();
}

QTabulated_Model::QTabulated_Model(const QTabulated_Model* lens_in)
{
	copy_base_lensdata(lens_in);
	original_emode = lens_in->original_emode;
	kscale = lens_in->kscale;
	rscale = lens_in->rscale;
	rscale0 = lens_in->rscale0;
	if ((original_emode==0) or (original_emode==1))
		q = lens_in->q;
	else
		q = lens_in->epsilon;
	grid_logrlength = lens_in->grid_logrlength;
	grid_qlength = lens_in->grid_qlength;
	grid_logr_N = lens_in->grid_logr_N;
	grid_phi_N = lens_in->grid_phi_N;
	grid_q_N = lens_in->grid_q_N;
	grid_logrvals = new double[grid_logr_N];
	grid_phivals = new double[grid_phi_N];
	grid_qvals = new double[grid_q_N];

	kappa_vals = new double**[grid_logr_N];
	pot_vals = new double**[grid_logr_N];
	defx = new double**[grid_logr_N];
	defy = new double**[grid_logr_N];
	hess_xx = new double**[grid_logr_N];
	hess_yy = new double**[grid_logr_N];
	hess_xy = new double**[grid_logr_N];
	int i,j,k;
	for (i=0; i < grid_logr_N; i++) {
		kappa_vals[i] = new double*[grid_phi_N];
		pot_vals[i] = new double*[grid_phi_N];
		defx[i] = new double*[grid_phi_N];
		defy[i] = new double*[grid_phi_N];
		hess_xx[i] = new double*[grid_phi_N];
		hess_yy[i] = new double*[grid_phi_N];
		hess_xy[i] = new double*[grid_phi_N];
		for (j=0; j < grid_phi_N; j++) {
			kappa_vals[i][j] = new double[grid_q_N];
			pot_vals[i][j] = new double[grid_q_N];
			defx[i][j] = new double[grid_q_N];
			defy[i][j] = new double[grid_q_N];
			hess_xx[i][j] = new double[grid_q_N];
			hess_yy[i][j] = new double[grid_q_N];
			hess_xy[i][j] = new double[grid_q_N];
		}
	}

	for (i=0; i < grid_logr_N; i++) grid_logrvals[i] = lens_in->grid_logrvals[i];
	for (j=0; j < grid_phi_N; j++) grid_phivals[j] = lens_in->grid_phivals[j];
	for (k=0; k < grid_q_N; k++) grid_qvals[k] = lens_in->grid_qvals[k];

	rmin_einstein_radius = lens_in->rmin_einstein_radius;
	rmax_einstein_radius = lens_in->rmax_einstein_radius;

	for (i=0; i < grid_logr_N; i++) {
		for (j=0; j < grid_phi_N; j++) {
			for (k=0; k < grid_q_N; k++) {
				kappa_vals[i][j][k] = lens_in->kappa_vals[i][j][k];
				pot_vals[i][j][k] = lens_in->pot_vals[i][j][k];
				defx[i][j][k] = lens_in->defx[i][j][k];
				defy[i][j][k] = lens_in->defy[i][j][k];
				hess_xx[i][j][k] = lens_in->hess_xx[i][j][k];
				hess_yy[i][j][k] = lens_in->hess_yy[i][j][k];
				hess_xy[i][j][k] = lens_in->hess_xy[i][j][k];
			}
		}
	}
	update_meta_parameters_and_pointers();
}

QTabulated_Model::QTabulated_Model(const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, const double &xc, const double &yc, ifstream& tabfile, Cosmology* cosmo_in)
{
	lenstype = QTABULATED;
	setup_base_lens_properties(7,-1,false); // number of parameters = 5, is_elliptical_lens = false
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);

	kscale = kscale_in;
	rscale = rscale_in;
	q = q_in;
	f_major_axis = 1.0;
	theta = degrees_to_radians(theta_in);
	p.x_center = xc;
	p.y_center = yc;

	tabfile >> model_name;
	tabfile >> rscale0;

	tabfile >> grid_logr_N;
	tabfile >> grid_phi_N;
	tabfile >> grid_q_N;

	grid_logrvals = new double[grid_logr_N];
	grid_phivals = new double[grid_phi_N];
	grid_qvals = new double[grid_q_N];

	kappa_vals = new double**[grid_logr_N];
	pot_vals = new double**[grid_logr_N];
	defx = new double**[grid_logr_N];
	defy = new double**[grid_logr_N];
	hess_xx = new double**[grid_logr_N];
	hess_yy = new double**[grid_logr_N];
	hess_xy = new double**[grid_logr_N];
	int i,j,k;
	for (i=0; i < grid_logr_N; i++) {
		kappa_vals[i] = new double*[grid_phi_N];
		pot_vals[i] = new double*[grid_phi_N];
		defx[i] = new double*[grid_phi_N];
		defy[i] = new double*[grid_phi_N];
		hess_xx[i] = new double*[grid_phi_N];
		hess_yy[i] = new double*[grid_phi_N];
		hess_xy[i] = new double*[grid_phi_N];
		for (j=0; j < grid_phi_N; j++) {
			kappa_vals[i][j] = new double[grid_q_N];
			pot_vals[i][j] = new double[grid_q_N];
			defx[i][j] = new double[grid_q_N];
			defy[i][j] = new double[grid_q_N];
			hess_xx[i][j] = new double[grid_q_N];
			hess_yy[i][j] = new double[grid_q_N];
			hess_xy[i][j] = new double[grid_q_N];
		}
	}

	for (i=0; i < grid_logr_N; i++) tabfile >> grid_logrvals[i];
	for (j=0; j < grid_phi_N; j++) tabfile >> grid_phivals[j];
	for (k=0; k < grid_q_N; k++) tabfile >> grid_qvals[k];
	grid_logrlength = grid_logrvals[grid_logr_N-1] - grid_logrvals[0];
	grid_qlength = grid_qvals[grid_q_N-1] - grid_qvals[0];
	update_meta_parameters_and_pointers();

	rmin_einstein_radius = exp(grid_logrvals[0]);
	rmax_einstein_radius = exp(grid_logrvals[grid_logr_N-1]);

	for (i=0; i < grid_logr_N; i++) {
		for (j=0; j < grid_phi_N; j++) {
			for (k=0; k < grid_q_N; k++) {
				tabfile >> kappa_vals[i][j][k] >> pot_vals[i][j][k] >> defx[i][j][k] >> defy[i][j][k] >> hess_xx[i][j][k] >> hess_yy[i][j][k] >> hess_xy[i][j][k];
			}
		}
		cout << "Row " << i << " done...\n" << flush;
	}
}

void QTabulated_Model::output_tables(const string tabfile_root)
{
	string tabfilename = tabfile_root + ".tab";
	ofstream tabfile(tabfilename.c_str());
	tabfile << model_name << " " << rscale0 << endl;
	tabfile << grid_logr_N << " " << grid_phi_N << " " << grid_q_N << endl << endl;
	int i,j,k;
	for (i=0; i < grid_logr_N; i++) tabfile << grid_logrvals[i] << " ";
	tabfile << endl;
	for (j=0; j < grid_phi_N; j++) tabfile << grid_phivals[j] << " ";
	tabfile << endl << endl;
	for (k=0; k < grid_q_N; k++) tabfile << grid_qvals[k] << " ";
	tabfile << endl << endl;
	for (i=0; i < grid_logr_N; i++) {
		for (j=0; j < grid_phi_N; j++) {
			for (k=0; k < grid_q_N; k++) {
				tabfile << kappa_vals[i][j][k] << " ";
				tabfile << pot_vals[i][j][k] << " ";
				tabfile << defx[i][j][k] << " ";
				tabfile << defy[i][j][k] << " ";
				tabfile << hess_xx[i][j][k] << " ";
				tabfile << hess_yy[i][j][k] << " ";
				tabfile << hess_xy[i][j][k];
				tabfile << endl;
			}
		}
	}
	tabfile.close();
}

void QTabulated_Model::assign_paramnames()
{
	paramnames[0] = "kscale"; latex_paramnames[0] = "\\kappa"; latex_param_subscripts[0] = "s";
	paramnames[1] = "rscale";  latex_paramnames[1] = "r"; latex_param_subscripts[1] = "s";
	paramnames[2] = "q";  latex_paramnames[2] = "q"; latex_param_subscripts[2] = "";
	paramnames[3] = "theta";  latex_paramnames[3] = "\\theta"; latex_param_subscripts[3] = "";
	paramnames[4] = "xc";     latex_paramnames[4] = "x";       latex_param_subscripts[4] = "c";
	paramnames[5] = "yc";     latex_paramnames[5] = "y";       latex_param_subscripts[5] = "c";
	if (lensed_center_coords) {
		paramnames[4] += "_l"; latex_param_subscripts[4] += ",l";
		paramnames[5] += "_l"; latex_param_subscripts[5] += ",l";
	}
}

void QTabulated_Model::assign_param_pointers()
{
	p.param[0] = &kscale;
	p.param[1] = &rscale;
	p.param[2] = &q;
	p.param[3] = &theta; angle_param[3] = true; angle_param_exists = true;
	if (!lensed_center_coords) {
		p.param[4] = &p.x_center;
		p.param[5] = &p.y_center;
	} else {
		p.param[4] = &xc_prime;
		p.param[5] = &yc_prime;
	}
}

void QTabulated_Model::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	// We don't use orient_major_axis_north because this is meaningless for the tabulated model
	costheta = cos(theta);
	sintheta = sin(theta);
	rscale_factor = rscale / rscale0;
	kval = (int) ((q - grid_qvals[0]) * grid_q_N / grid_qlength);
	if (kval < 0) kval=0;
	if (kval >= grid_q_N-1) kval=grid_q_N-2;
	ww = (q - grid_qvals[kval]) / (grid_qvals[kval+1] - grid_qvals[kval]);
	WW = 1-ww;
}

void QTabulated_Model::set_auto_stepsizes()
{
	stepsizes[0] = 0.3*kscale;
	stepsizes[1] = 0.3*rscale;
	stepsizes[2] = 0.1;
	stepsizes[3] = 20;
	stepsizes[4] = p.x_center;
	stepsizes[5] = p.y_center;
	stepsizes[6] = 0.1;
}

void QTabulated_Model::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = 0.01*kscale; penalty_upper_limits[0] = 100*kscale;  // limits are somewhat arbitrary, but we shouldn't allow k --> 0 or k very large
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = exp(grid_logrvals[0]); penalty_upper_limits[1] = 1e30;
	set_auto_penalty_limits[2] = true; penalty_lower_limits[2] = grid_qvals[0]; penalty_upper_limits[2] = grid_qvals[grid_q_N-1];
	set_auto_penalty_limits[3] = false;
	set_auto_penalty_limits[4] = false;
	set_auto_penalty_limits[5] = false;
}

double QTabulated_Model::potential(double x, double y)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	int ival, jval;
	double tt, uu, TT, UU, interp;
	double logr, phi;
	logr = log(sqrt(x*x+y*y)/rscale_factor); // scaling r is easier than scaling the table itself
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;
	//cout << kval << " " << pot_vals[ival][jval][kval] << " " << pot_vals[ival][jval][kval+1] << endl;

	interp = WW*(TT*UU*pot_vals[ival][jval][kval] + tt*UU*pot_vals[ival+1][jval][kval]
						+ TT*uu*pot_vals[ival][jval+1][kval] + tt*uu*pot_vals[ival+1][jval+1][kval])
				+ ww*(TT*UU*pot_vals[ival][jval][kval+1] + tt*UU*pot_vals[ival+1][jval][kval+1]
						+ TT*uu*pot_vals[ival][jval+1][kval+1] + tt*uu*pot_vals[ival+1][jval+1][kval+1]);

	return kscale*rscale_factor*rscale_factor*interp;
}

double QTabulated_Model::kappa(double x, double y)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	double logr, phi;
	int ival, jval;
	double tt, uu, TT, UU, interp;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}
	//cout << "logr=" << logr << ", r=" << exp(logr) << ", phi=" << phi << endl;
	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	//cout << ival << " " << jval << " " << tt << " " << uu << endl;
	//cout << kappa_vals[ival][jval] << endl;

	interp = WW*(TT*UU*kappa_vals[ival][jval][kval] + tt*UU*kappa_vals[ival+1][jval][kval]
						+ TT*uu*kappa_vals[ival][jval+1][kval] + tt*uu*kappa_vals[ival+1][jval+1][kval])
				+ ww*(TT*UU*kappa_vals[ival][jval][kval+1] + tt*UU*kappa_vals[ival+1][jval][kval+1]
						+ TT*uu*kappa_vals[ival][jval+1][kval+1] + tt*uu*kappa_vals[ival+1][jval+1][kval+1]);

	return kscale*interp;
}

double QTabulated_Model::kappa_rsq_impl(const double rsq)
{
	// probably should change this so it just sets q=1 and doesn't require an angle average (perhaps?)
	double logr, phi;
	logr = log(sqrt(rsq)/rscale_factor);
	static const int phi_N = 100;
	double phistep = M_2PI/phi_N;
	int i, ival, jval;
	double tt, uu, TT, UU, interp;
	double kappa_angular_avg = 0;
	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	if (ival < 0) ival=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	TT = 1-tt;
	for (i=0, phi=0; i < phi_N; i++, phi += phistep) {
		jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
		if (jval < 0) jval=0;
		if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
		uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
		UU = 1-uu;

		kappa_angular_avg += kscale*(WW*(TT*UU*kappa_vals[ival][jval][kval] + tt*UU*kappa_vals[ival+1][jval][kval]
						+ TT*uu*kappa_vals[ival][jval+1][kval] + tt*uu*kappa_vals[ival+1][jval+1][kval])
				+ ww*(TT*UU*kappa_vals[ival][jval][kval+1] + tt*UU*kappa_vals[ival+1][jval][kval+1]
						+ TT*uu*kappa_vals[ival][jval+1][kval+1] + tt*uu*kappa_vals[ival+1][jval+1][kval+1]));

	}
	kappa_angular_avg /= phi_N;
	return kappa_angular_avg;
}

void QTabulated_Model::deflection(double x, double y, lensvector<double>& def)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	double logr, phi;
	int ival, jval;
	double tt, uu, TT, UU, interp;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	//cout << logr << " " << grid_logrvals[0] << " " << grid_logrlength << " " << grid_logr_N << endl;
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;
	interp = WW*(TT*UU*defx[ival][jval][kval] + tt*UU*defx[ival+1][jval][kval] + TT*uu*defx[ival][jval+1][kval] + tt*uu*defx[ival+1][jval+1][kval])
				+ ww*(TT*UU*defx[ival][jval][kval+1] + tt*UU*defx[ival+1][jval][kval+1] + TT*uu*defx[ival][jval+1][kval+1] + tt*uu*defx[ival+1][jval+1][kval+1]);
	def[0] = kscale*rscale_factor*interp;
	interp = WW*(TT*UU*defy[ival][jval][kval] + tt*UU*defy[ival+1][jval][kval] + TT*uu*defy[ival][jval+1][kval] + tt*uu*defy[ival+1][jval+1][kval])
				+ ww*(TT*UU*defy[ival][jval][kval+1] + tt*UU*defy[ival+1][jval][kval+1] + TT*uu*defy[ival][jval+1][kval+1] + tt*uu*defy[ival+1][jval+1][kval+1]);
	def[1] = kscale*rscale_factor*interp;
	if (sintheta != 0) def.rotate_back(costheta,sintheta);
}

void QTabulated_Model::hessian(double x, double y, lensmatrix<double>& hess)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	double logr, phi;
	int ival, jval;
	double tt, uu, TT, UU, interp;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;
	interp = WW*(TT*UU*hess_xx[ival][jval][kval] + tt*UU*hess_xx[ival+1][jval][kval] + TT*uu*hess_xx[ival][jval+1][kval] + tt*uu*hess_xx[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_xx[ival][jval][kval+1] + tt*UU*hess_xx[ival+1][jval][kval+1] + TT*uu*hess_xx[ival][jval+1][kval+1] + tt*uu*hess_xx[ival+1][jval+1][kval+1]);

	hess[0][0] = kscale*interp;
	interp = WW*(TT*UU*hess_yy[ival][jval][kval] + tt*UU*hess_yy[ival+1][jval][kval] + TT*uu*hess_yy[ival][jval+1][kval] + tt*uu*hess_yy[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_yy[ival][jval][kval+1] + tt*UU*hess_yy[ival+1][jval][kval+1] + TT*uu*hess_yy[ival][jval+1][kval+1] + tt*uu*hess_yy[ival+1][jval+1][kval+1]);

	hess[1][1] = kscale*interp;
	interp = WW*(TT*UU*hess_xy[ival][jval][kval] + tt*UU*hess_xy[ival+1][jval][kval] + TT*uu*hess_xy[ival][jval+1][kval] + tt*uu*hess_xy[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_xy[ival][jval][kval+1] + tt*UU*hess_xy[ival+1][jval][kval+1] + TT*uu*hess_xy[ival][jval+1][kval+1] + tt*uu*hess_xy[ival+1][jval+1][kval+1]);

	hess[0][1] = kscale*interp;
	hess[1][0] = hess[0][1];
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

void QTabulated_Model::kappa_and_potential_derivatives(double x, double y, double& kap, lensvector<double>& def, lensmatrix<double>& hess)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	int ival, jval;
	double tt, uu, TT, UU, interp;
	double logr, phi;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	interp = WW*(TT*UU*kappa_vals[ival][jval][kval] + tt*UU*kappa_vals[ival+1][jval][kval]
						+ TT*uu*kappa_vals[ival][jval+1][kval] + tt*uu*kappa_vals[ival+1][jval+1][kval])
				+ ww*(TT*UU*kappa_vals[ival][jval][kval+1] + tt*UU*kappa_vals[ival+1][jval][kval+1]
						+ TT*uu*kappa_vals[ival][jval+1][kval+1] + tt*uu*kappa_vals[ival+1][jval+1][kval+1]);
	kap = kscale*interp;

	interp = WW*(TT*UU*defx[ival][jval][kval] + tt*UU*defx[ival+1][jval][kval] + TT*uu*defx[ival][jval+1][kval] + tt*uu*defx[ival+1][jval+1][kval])
				+ ww*(TT*UU*defx[ival][jval][kval+1] + tt*UU*defx[ival+1][jval][kval+1] + TT*uu*defx[ival][jval+1][kval+1] + tt*uu*defx[ival+1][jval+1][kval+1]);
	def[0] = kscale*rscale_factor*interp;
	interp = WW*(TT*UU*defy[ival][jval][kval] + tt*UU*defy[ival+1][jval][kval] + TT*uu*defy[ival][jval+1][kval] + tt*uu*defy[ival+1][jval+1][kval])
				+ ww*(TT*UU*defy[ival][jval][kval+1] + tt*UU*defy[ival+1][jval][kval+1] + TT*uu*defy[ival][jval+1][kval+1] + tt*uu*defy[ival+1][jval+1][kval+1]);
	def[1] = kscale*rscale_factor*interp;

	interp = WW*(TT*UU*hess_xx[ival][jval][kval] + tt*UU*hess_xx[ival+1][jval][kval] + TT*uu*hess_xx[ival][jval+1][kval] + tt*uu*hess_xx[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_xx[ival][jval][kval+1] + tt*UU*hess_xx[ival+1][jval][kval+1] + TT*uu*hess_xx[ival][jval+1][kval+1] + tt*uu*hess_xx[ival+1][jval+1][kval+1]);
	hess[0][0] = kscale*interp;

	interp = WW*(TT*UU*hess_yy[ival][jval][kval] + tt*UU*hess_yy[ival+1][jval][kval] + TT*uu*hess_yy[ival][jval+1][kval] + tt*uu*hess_yy[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_yy[ival][jval][kval+1] + tt*UU*hess_yy[ival+1][jval][kval+1] + TT*uu*hess_yy[ival][jval+1][kval+1] + tt*uu*hess_yy[ival+1][jval+1][kval+1]);
	hess[1][1] = kscale*interp;

	interp = WW*(TT*UU*hess_xy[ival][jval][kval] + tt*UU*hess_xy[ival+1][jval][kval] + TT*uu*hess_xy[ival][jval+1][kval] + tt*uu*hess_xy[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_xy[ival][jval][kval+1] + tt*UU*hess_xy[ival+1][jval][kval+1] + TT*uu*hess_xy[ival][jval+1][kval+1] + tt*uu*hess_xy[ival+1][jval+1][kval+1]);
	hess[0][1] = kscale*interp;
	hess[1][0] = hess[0][1];

	if (sintheta != 0) def.rotate_back(costheta,sintheta);
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

void QTabulated_Model::potential_derivatives(double x, double y, lensvector<double>& def, lensmatrix<double>& hess)
{
	x -= p.x_center;
	y -= p.y_center;
	if (sintheta != 0) rotate(x,y);
	int ival, jval;
	double tt, uu, TT, UU, interp;
	double logr, phi;
	logr = log(sqrt(x*x+y*y)/rscale_factor);
	if (x==0.0) phi = 0;
	else {
		phi = atan(abs(y/x));
		if (x < 0) {
			if (y < 0)
				phi = M_PI + phi;
			else
				phi = M_PI - phi;
		} else if (y < 0) {
			phi = M_2PI - phi;
		}
	}

	ival = (int) ((logr - grid_logrvals[0]) * grid_logr_N / grid_logrlength);
	jval = (int) ((phi - grid_phivals[0]) * grid_phi_N / M_2PI);
	if (ival < 0) ival=0;
	if (jval < 0) jval=0;
	if (ival >= grid_logr_N-1) ival=grid_logr_N-2;
	if (jval >= grid_phi_N-1) jval=grid_phi_N-2;
	tt = (logr - grid_logrvals[ival]) / (grid_logrvals[ival+1] - grid_logrvals[ival]);
	uu = (phi - grid_phivals[jval]) / (grid_phivals[jval+1] - grid_phivals[jval]);
	TT = 1-tt;
	UU = 1-uu;

	interp = WW*(TT*UU*defx[ival][jval][kval] + tt*UU*defx[ival+1][jval][kval] + TT*uu*defx[ival][jval+1][kval] + tt*uu*defx[ival+1][jval+1][kval])
				+ ww*(TT*UU*defx[ival][jval][kval+1] + tt*UU*defx[ival+1][jval][kval+1] + TT*uu*defx[ival][jval+1][kval+1] + tt*uu*defx[ival+1][jval+1][kval+1]);
	def[0] = kscale*rscale_factor*interp;
	interp = WW*(TT*UU*defy[ival][jval][kval] + tt*UU*defy[ival+1][jval][kval] + TT*uu*defy[ival][jval+1][kval] + tt*uu*defy[ival+1][jval+1][kval])
				+ ww*(TT*UU*defy[ival][jval][kval+1] + tt*UU*defy[ival+1][jval][kval+1] + TT*uu*defy[ival][jval+1][kval+1] + tt*uu*defy[ival+1][jval+1][kval+1]);
	def[1] = kscale*rscale_factor*interp;

	interp = WW*(TT*UU*hess_xx[ival][jval][kval] + tt*UU*hess_xx[ival+1][jval][kval] + TT*uu*hess_xx[ival][jval+1][kval] + tt*uu*hess_xx[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_xx[ival][jval][kval+1] + tt*UU*hess_xx[ival+1][jval][kval+1] + TT*uu*hess_xx[ival][jval+1][kval+1] + tt*uu*hess_xx[ival+1][jval+1][kval+1]);

	hess[0][0] = kscale*interp;
	interp = WW*(TT*UU*hess_yy[ival][jval][kval] + tt*UU*hess_yy[ival+1][jval][kval] + TT*uu*hess_yy[ival][jval+1][kval] + tt*uu*hess_yy[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_yy[ival][jval][kval+1] + tt*UU*hess_yy[ival+1][jval][kval+1] + TT*uu*hess_yy[ival][jval+1][kval+1] + tt*uu*hess_yy[ival+1][jval+1][kval+1]);

	hess[1][1] = kscale*interp;
	interp = WW*(TT*UU*hess_xy[ival][jval][kval] + tt*UU*hess_xy[ival+1][jval][kval] + TT*uu*hess_xy[ival][jval+1][kval] + tt*uu*hess_xy[ival+1][jval+1][kval])
				+ ww*(TT*UU*hess_xy[ival][jval][kval+1] + tt*UU*hess_xy[ival+1][jval][kval+1] + TT*uu*hess_xy[ival][jval+1][kval+1] + tt*uu*hess_xy[ival+1][jval+1][kval+1]);
	hess[0][1] = kscale*interp;
	hess[1][0] = hess[0][1];

	if (sintheta != 0) def.rotate_back(costheta,sintheta);
	if (sintheta != 0) hess.rotate_back(costheta,sintheta);
}

QTabulated_Model::~QTabulated_Model() {
	if (grid_logrvals != NULL) {
		delete[] grid_logrvals;
		delete[] grid_phivals;
		delete[] grid_qvals;
		for (int i=0; i < grid_logr_N; i++) {
			for (int j=0; j < grid_phi_N; j++) {
				delete[] kappa_vals[i][j];
				delete[] pot_vals[i][j];
				delete[] defx[i][j];
				delete[] defy[i][j];
				delete[] hess_xx[i][j];
				delete[] hess_yy[i][j];
				delete[] hess_xy[i][j];
			}
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
}
*/

TopHatLens::TopHatLens(const double zlens_in, const double zsrc_in, const double &kap0_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, Cosmology* cosmo_in)
{
	lensparams = &lensparams_tophat;
	setup_lens_properties();
	set_redshifts(zlens_in,zsrc_in);
	setup_cosmology(cosmo_in);
	initialize_parameters(kap0_in,rad_in,q_in,theta_degrees,xc_in,yc_in);
}

void TopHatLens::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = TOPHAT_LENS;
	model_name = "tophat";
	setup_base_lens_properties(7,2,true);
	analytic_3d_density = false;
}

void TopHatLens::initialize_parameters(const double &kap0_in, const double &rad_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	// if use_ellipticity_components is on, q_in and theta_in are actually e1, e2, but this is taken care of in set_geometric_parameters
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	TopHat_Params<double>& p = assign_tophat_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.kap0 = kap0_in;
	p.xi0 = rad_in;
	update_meta_parameters_and_pointers();
}

TopHatLens::TopHatLens(const TopHatLens* lens_in)
{
	lensparams = &lensparams_tophat;
	copy_base_lensdata(lens_in);
	TopHat_Params<double>& p = assign_tophat_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.kap0 = lens_in->lensparams_tophat.kap0;
	p.xi0 = lens_in->lensparams_tophat.xi0;
	update_meta_parameters_and_pointers();
}

void TopHatLens::assign_paramnames()
{
	paramnames[0] = "kap0";     latex_paramnames[0] = "\\kappa";       latex_param_subscripts[0] = "top";
	paramnames[1] = "xi0"; latex_paramnames[1] = "r"; latex_param_subscripts[1] = "top";
	set_geometric_paramnames(lensprofile_nparams);
}

void TopHatLens::assign_param_pointers()
{
	TopHat_Params<double>& p = assign_tophat_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	p.param[0] = &p.kap0;
	p.param[1] = &p.xi0;
	set_geometric_param_pointers(lensprofile_nparams);
}

void TopHatLens::update_meta_parameters()
{
	update_cosmology_meta_parameters();
	update_ellipticity_meta_parameters();
}

void TopHatLens::set_auto_stepsizes()
{
	int index = 0;
	stepsizes[index++] = 0.1; // arbitrary
	stepsizes[index++] = 0.1; // arbitrary
	set_geometric_param_auto_stepsizes(index);
}

void TopHatLens::set_auto_ranges()
{
	set_auto_penalty_limits[0] = true; penalty_lower_limits[0] = -1e30; penalty_upper_limits[0] = 1e30;
	set_auto_penalty_limits[1] = true; penalty_lower_limits[1] = 0; penalty_upper_limits[1] = 1e30;
	set_geometric_param_auto_ranges(lensprofile_nparams);
}

void TopHatLens::set_model_specific_integration_pointers()
{
	kapavgptr_rsq_spherical = static_cast<double (LensProfile::*)(const double)> (&TopHatLens::kapavg_spherical_rsq);
	if (!ellipticity_gradient) {
		defptr = static_cast<void (LensProfile::*)(const double,const double,lensvector<double>&)> (&TopHatLens::deflection_analytic);
		hessptr = static_cast<void (LensProfile::*)(const double,const double,lensmatrix<double>&)> (&TopHatLens::hessian_analytic);
		potptr = static_cast<double (LensProfile::*)(const double,const double)> (&TopHatLens::potential_analytic);
	}
}

template <typename QScalar>
QScalar TopHatLens::kappa_rsq_impl(const QScalar rsq)
{
	TopHat_Params<QScalar>& p = assign_tophat_param_object<QScalar>(); // this reference will point to either the <QScalar> lensparams or <stan::math::var> lensparams for autodiff
	if (rsq > p.xi0*p.xi0) return 0;
	else return p.kap0;
}
template double TopHatLens::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var TopHatLens::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

template <typename QScalar>
QScalar TopHatLens::kappa_rsq_deriv_impl(const QScalar rsq)
{
	return 0;
}
template double TopHatLens::kappa_rsq_deriv_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var TopHatLens::kappa_rsq_deriv_impl<stan::math::var>(const stan::math::var rsq);
#endif

double TopHatLens::kapavg_spherical_rsq(const double rsq)
{
	TopHat_Params<double>& p = assign_tophat_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	if (rsq < p.xi0*p.xi0) return p.kap0;
	else return p.kap0*rsq/(p.xi0*p.xi0);
}

void TopHatLens::deflection_analytic(const double x, const double y, lensvector<double>& def)
{
	TopHat_Params<double>& p = assign_tophat_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double eps, xsqval, ysqval, xisq, qfac, u, qufactor, def_fac;
	double fsqinv = 1/SQR(f_major_axis);
	eps = 1 - p.q*p.q;
	xsqval = x*x;
	ysqval = y*y;
	xisq = p.xi0*p.xi0;
	qfac = xsqval + ysqval + eps*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-eps))) u = 1.0;
	else if ((eps*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*eps*xsqval*xisq/fsqinv)) / (2*eps*xsqval);
	
	qufactor = sqrt(1-eps*u);
	def_fac = (2*p.kap0*p.q);

	if (eps > 1e-9) {
		def[0] = x*def_fac*(1-qufactor)/eps;
		def[1] = -y*def_fac*(1-1.0/qufactor)/eps;
	} else {
		def[0] = x*def_fac*u/2;
		def[1] = -y*def_fac*u/2;
	}
}

void TopHatLens::hessian_analytic(const double x, const double y, lensmatrix<double>& hess)
{
	TopHat_Params<double>& p = assign_tophat_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double eps, xsqval, ysqval, xisq, qfac, u, qufactor, dxisq, hessfac, def_fac, def0, def1;
	double fsqinv = 1.0/SQR(f_major_axis);
	eps = 1 - p.q*p.q;
	xsqval = x*x;
	ysqval = y*y;
	xisq = p.xi0*p.xi0;
	qfac = xsqval + ysqval + eps*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-eps))) u = 1.0;
	else if ((eps*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*eps*xsqval*xisq/fsqinv)) / (2*eps*xsqval);

	qufactor = sqrt(1-eps*u);
	//dxisq = xsqval + ysqval/(qufactor*qufactor);
	//hessfac = -2*(2*p.xi0*p.kap0*p.q*u/sqrt(qufactor))/dxisq;
	def_fac = (2*p.kap0*p.q);

	if (eps > 1e-9) {
		def0 = def_fac*(1-qufactor)/eps;
		def1 = -def_fac*(1-1.0/qufactor)/eps;
	} else {
		def0 = def_fac*u/2;
		def1 = -def_fac*u/2;
	}

	hess[0][0] = def0;
	hess[1][1] = def1;
	hess[1][0] = 0;
	hess[0][1] = 0;
}

double TopHatLens::potential_analytic(const double x, const double y)
{
	TopHat_Params<double>& p = assign_tophat_param_object<double>(); // this reference will point to either the <double> lensparams or <stan::math::var> lensparams for autodiff
	double eps, xsqval, ysqval, xisq, qfac, u, qufactor, dxisq, hessfac, def_fac, def0, def1;
	double fsqinv = 1.0/SQR(f_major_axis);
	eps = 1 - p.q*p.q;
	xsqval = x*x;
	ysqval = y*y;
	xisq = p.xi0*p.xi0;
	qfac = xsqval + ysqval + eps*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-eps))) u = 1.0;
	else if ((eps*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*eps*xsqval*xisq/fsqinv)) / (2*eps*xsqval);

	qufactor = sqrt(1-eps*u);
	//dxisq = xsqval + ysqval/(qufactor*qufactor);
	//hessfac = -2*(2*p.xi0*p.kap0*p.q*u/sqrt(qufactor))/dxisq;

	double inside_fac, outside_fac;
	if (eps > 1e-4) {
		inside_fac = (2/eps)*(xsqval*(1-qufactor) - ysqval*(1-1.0/qufactor)); // this is the contribution from work done moving through the inside of the plates
		outside_fac = (u==1.0) ? 0.0 : xisq*log((1-p.q)/(1-qufactor)*(1+qufactor)/(1+p.q))/fsqinv; // contribution from work done moving outside the plates
	} else {
		inside_fac = u*(xsqval + ysqval);
		outside_fac = (u==1.0) ? 0.0 : xisq*log((1-eps*u/4)/(1-eps/4)/u);
	}
	
	return (p.kap0*p.q*(inside_fac+outside_fac)/2);
}

/***************************** Test Model (for testing purposes only) *****************************/

TestModel::TestModel(const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in)
{
	cout << "HARG" << endl;
	setup_lens_properties();
	//setup_base_lens_properties(X,false); // number of parameters = X, is_elliptical_lens = false
	cout << "HARG2" << endl;
	set_geometric_parameters(q_in,theta_degrees,xc_in,yc_in);
	cout << "HARG3" << endl;
	set_integration_pointers();
	cout << "HARG4" << endl;
}

/*
void TestModel::deflection(double x, double y, lensvector<double>& def)
{
	def[0] = 0;
	def[1] = 0;
}

void TestModel::hessian(double x, double y, lensmatrix<double>& hess)
{
	hess[0][0] = 0;
	hess[1][1] = 0;
	hess[1][0] = 0;
	hess[0][1] = 0;
}
*/

void TestModel::setup_lens_properties(const int parameter_mode, const int subclass)
{
	lenstype = TESTMODEL;
	model_name = "test";
}

template <typename QScalar>
QScalar TestModel::kappa_rsq_impl(const QScalar rsq)
{
	QScalar ans;
	static const QScalar cutoff = 0.173;
	QScalar cutoffsq = cutoff*cutoff;
	if (rsq > cutoffsq) return 0;
	else
		return SQR(3.0/cutoff);
		//return (0.5 * 0.0216 * (1/sqrt(rsq) - pow(0.173*0.173+rsq,-0.5))); // PJaffe
}
template double TestModel::kappa_rsq_impl<double>(const double rsq);
#ifdef USE_STAN
template stan::math::var TestModel::kappa_rsq_impl<stan::math::var>(const stan::math::var rsq);
#endif

