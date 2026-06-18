#include "gauss.h"
#include "profile.h"
#include "mathexpr.h"
#include "errors.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

class LensProfile;

template <typename VecType, typename QScalar>
struct LensIntegral 
{
	using GaussQuad = GaussLegendre<std::function<QScalar(const QScalar)>,QScalar>;
	using Patterson = GaussPatterson<std::function<QScalar(const QScalar)>,QScalar>;
	using Fejer = ClenshawCurtis<std::function<QScalar(const QScalar)>,QScalar>;

	LensProfile *profile;
	QScalar xval, yval, xsqval, ysqval;
	double nval_plus_half, mnval_plus_half;
	QScalar fsqinv, xisq, u, epsilon, qfac;
	int nval, emode;
	int mval, fourier_ival; // mval, fourier_ival are used for the Fourier mode integrals
	QScalar phi0; // phi0 is used for Fourier mode integrals if ellipticity gradient is used
	bool cosmode;
	QScalar *cosamps, *sinamps; // used for Fourier modes
	double *gausspoints, *gaussweights;
	//double *pat_points, **pat_weights;
	QScalar *pat_funcs;
	VecType *pat_funcs_vec;
	QScalar **pat_funcs_mult;
	//double *cc_points, **cc_weights;
	QScalar *cc_funcs;
	VecType *cc_funcs_vec;
	QScalar **cc_funcs_mult;
	int n_mult;

	LensIntegral()
	{
		cosamps=sinamps=NULL;
		pat_funcs = NULL;
		pat_funcs_mult = NULL;
		cc_funcs = NULL;
		cc_funcs_mult = NULL;
		pat_funcs_vec = NULL;
		cc_funcs_vec = NULL;
		n_mult = 0;
	}
	LensIntegral(LensProfile *profile_in, const QScalar xval_in, const QScalar yval_in, const QScalar q = 1, const int n_mult_in = 0, const bool use_vecs = false) : xval(xval_in), yval(yval_in)
	{
		cosamps=sinamps=NULL;
		pat_funcs_vec = NULL;
		cc_funcs_vec = NULL;
		initialize(profile_in,q,n_mult_in,use_vecs);
	}
	void initialize(LensProfile *profile_in, const QScalar q = 1, const int n_mult_in = 0, const bool use_vecs = false)
	{
		n_mult = n_mult_in;
		profile = profile_in;
		xsqval = xval*xval;
		ysqval = yval*yval;
		epsilon = 1 - q*q;
		emode = profile->ellipticity_mode;
		fsqinv = (emode==0) ? 1 : (emode==1) ? q : (emode==2) ? q : q*q/((1+q*q)/2); 
		phi0 = 0;
		gausspoints = GaussLegendre<std::function<QScalar(const QScalar)>,QScalar>::points;
		gaussweights = GaussLegendre<std::function<QScalar(const QScalar)>,QScalar>::weights;
		if (profile->integral_method==Gauss_Patterson_Quadrature) {
			if (n_mult > 0) {
				pat_funcs_mult = new QScalar*[511];
				for (int i=0; i < 511; i++) pat_funcs_mult[i] = new QScalar[n_mult];
			} else {
				pat_funcs = new QScalar[511];
				if (use_vecs) {
					pat_funcs_vec = new VecType[511];
				}
			}

		} else if (profile->integral_method==Fejer_Quadrature) {
			if (n_mult > 0) {
				cc_funcs_mult = new QScalar*[Fejer::cc_N];
				for (int i=0; i < Fejer::cc_N; i++) cc_funcs_mult[i] = new QScalar[n_mult];
			} else {
				cc_funcs = new QScalar[Fejer::cc_N];
				if (use_vecs) {
					cc_funcs_vec = new VecType[Fejer::cc_N];
				}
			}
		}
	}
	~LensIntegral() {
		if (profile->integral_method==Gauss_Patterson_Quadrature) {
			if (n_mult > 0) {
				for (int i=0; i < 511; i++) delete[] pat_funcs_mult[i];
				delete[] pat_funcs_mult;
			} else {
				delete[] pat_funcs;
				if (pat_funcs_vec != NULL) delete[] pat_funcs_vec;
			}
		} else if (profile->integral_method==Fejer_Quadrature) {
			if (n_mult > 0) {
				for (int i=0; i < Fejer::cc_N; i++) delete[] cc_funcs_mult[i];
				delete[] cc_funcs_mult;
			} else {
				delete[] cc_funcs;
				if (cc_funcs_vec != NULL) delete[] pat_funcs_vec;
			}
		}
	}
	QScalar GaussIntegrate(QScalar (LensIntegral::*func)(const QScalar), const QScalar a, const QScalar b);
	QScalar PattersonIntegrate(QScalar (LensIntegral::*func)(const QScalar), const QScalar a, const QScalar b, bool &converged);
	QScalar FejerIntegrate(QScalar (LensIntegral::*func)(QScalar), QScalar a, QScalar b, bool &converged);
	void GaussIntegrateVec(void (LensIntegral::*func)(const QScalar, const VecType&, const VecType&, VecType&), const VecType& xsq, const VecType& ysq, VecType& sum, const QScalar a, const QScalar b);
	void PattersonIntegrateVec(void (LensIntegral::*func)(const QScalar, const VecType&, const VecType&, VecType&), const VecType& xsq, const VecType& ysq, VecType& result, const QScalar a, const QScalar b, bool& converged);
	void FejerIntegrateVec(void (LensIntegral::*func)(const QScalar, const VecType&, const VecType&, VecType&), const VecType& xsq, const VecType& ysq, VecType& result, const QScalar a, const QScalar b, bool& converged);


	// Functions for doing multiple integrals simultaneously
	void GaussIntegrate(void (LensIntegral::*func)(const QScalar, QScalar*), const QScalar a, const QScalar b, QScalar* results, const int n_funcs);
	void PattersonIntegrate(void (LensIntegral::*func)(const QScalar, QScalar*), const QScalar a, const QScalar b, QScalar* results, const int n_funcs, bool& converged);
	void FejerIntegrate(void (LensIntegral::*func)(const QScalar, QScalar*), const QScalar a, const QScalar b, QScalar* results, const int n_funcs, bool& converged);

	QScalar i_integrand_prime(const QScalar w);
	QScalar j_integrand_prime(const QScalar w);
	QScalar k_integrand_prime(const QScalar w);
	void j_integrand_vec(const QScalar w, const VecType& xsqval, const VecType& ysqval, VecType& ans);

	//QScalar i_integrand_v2(const QScalar w);
	//QScalar j_integrand_v2(const QScalar w);
	//QScalar k_integrand_v2(const QScalar w);
	QScalar i_integral(bool &converged);
	QScalar j_integral(const int nval_in, bool &converged);
	QScalar k_integral(const int nval_in, bool &converged);
	void j_integral_vec(const int nval_in, const VecType& x, const VecType& y, VecType& ans, bool &converged);

	QScalar i_integrand_egrad(const QScalar w);
	//QScalar j_integrand_egrad(const QScalar w);
	//QScalar k_integrand_egrad(const QScalar w);
	//QScalar jprime_integrand_egrad(const QScalar w);

	QScalar i_integral_egrad(bool &converged);
	//QScalar j_integral_egrad(const int nval_in, bool &converged);
	//QScalar k_integral_egrad(const int nval_in, bool &converged);
	//QScalar jprime_integral_egrad(const int nval_in, bool &converged);

	void j_integrand_egrad_mult(const QScalar w, QScalar* jint);
	void k_integrand_egrad_mult(const QScalar w, QScalar* kint);
	void jprime_integrand_egrad_mult(const QScalar w, QScalar* jint);
	void j_integral_egrad_mult(QScalar *jint, bool &converged);
	void k_integral_egrad_mult(QScalar *kint, bool &converged);
	void jprime_integral_egrad_mult(QScalar *jint, bool &converged);

	void calculate_fourier_integrals(const int mval_in, const int fourier_ival_in, const bool cosmode_in, const QScalar rval, QScalar& ileft, QScalar& iright, bool &converged);
	QScalar fourier_kappa_perturbation(const QScalar r);
	QScalar ileft_integrand(const QScalar r);
	QScalar iright_integrand(const QScalar u); // here, u = 1/r
	QScalar fourier_kappa_m(const QScalar r, const QScalar phi, const int mval_in, const int fourier_ival_in);
};


/*************************** Integrals when ellipticity is constant ***************************/

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::i_integral(bool &converged)
{
	using std::sqrt;
#ifdef USE_STAN
	using stan::math::sqrt;
#endif
	converged = true; // will change if convergence not achieved
	QScalar ans;
	if (profile->integral_method == Romberg_Integration)
	{
		Romberg<std::function<QScalar(const QScalar)>,QScalar> romberg;
		ans = sqrt(1-epsilon)*romberg.integrate_open([this](auto x){return i_integrand_prime(x);}, 0, 1, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		QScalar (LensIntegral::*iptr)(QScalar) = &LensIntegral::i_integrand_prime;
		ans = sqrt(1-epsilon)*GaussIntegrate(iptr,0,1);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		QScalar (LensIntegral::*iptr)(QScalar) = &LensIntegral::i_integrand_prime;
		ans = sqrt(1-epsilon)*PattersonIntegrate(iptr,0,1,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		QScalar (LensIntegral::*iptr)(QScalar) = &LensIntegral::i_integrand_prime;
		ans = sqrt(1-epsilon)*FejerIntegrate(iptr,0,1,converged);
	}
	else die("unknown integral method");
	return ans;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::j_integral(const int nval_in, bool &converged)
{
	using std::sqrt;
#ifdef USE_STAN
	using stan::math::sqrt;
#endif
	nval_plus_half = nval_in + 0.5;
	mnval_plus_half = -nval_in + 0.5;
	converged = true; // will change if convergence not achieved
	QScalar ans;
	if (profile->integral_method == Romberg_Integration)
	{
		Romberg<std::function<QScalar(const QScalar)>,QScalar> romberg;
		ans = sqrt(1-epsilon)*romberg.integrate_open([this](auto x){return j_integrand_prime(x);}, 0, 1, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::j_integrand_prime;
		ans = sqrt(1-epsilon)*GaussIntegrate(jptr,0,1);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::j_integrand_prime;
		ans = sqrt(1-epsilon)*PattersonIntegrate(jptr,0,1,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::j_integrand_prime;
		ans = sqrt(1-epsilon)*FejerIntegrate(jptr,0,1,converged);

	}
	else die("unknown integral method");
	return ans;
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::j_integral_vec(const int nval_in, const VecType& x, const VecType& y, VecType& ans, bool &converged)
{
	//std::cout << "HERG?" << std::endl;
	using std::sqrt;
#ifdef USE_STAN
	using stan::math::sqrt;
#endif
	nval_plus_half = nval_in + 0.5;
	mnval_plus_half = -nval_in + 0.5;
	converged = true; // will change if convergence not achieved

	VecType xsq,ysq;
#ifdef USE_STAN
	if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
		xsq = stan::math::elt_multiply(x,x);
		ysq = stan::math::elt_multiply(y,y);
	} else
#endif
	{
		xsq = (x.array().square()).matrix();
		ysq = (y.array().square()).matrix();
	}

	if (profile->integral_method == Romberg_Integration)
	{
		// Not supported with vec
		ans = Eigen::VectorXd::Zero(x.size());
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		void (LensIntegral::*jptr)(const QScalar,const VecType&,const VecType&,VecType&) = &LensIntegral::j_integrand_vec;
		GaussIntegrateVec(jptr,xsq,ysq,ans,0,1);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		void (LensIntegral::*jptr)(const QScalar,const VecType&,const VecType&,VecType&) = &LensIntegral::j_integrand_vec;
		PattersonIntegrateVec(jptr,xsq,ysq,ans,0,1,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		void (LensIntegral::*jptr)(const QScalar,const VecType&,const VecType&,VecType&) = &LensIntegral::j_integrand_vec;
		FejerIntegrateVec(jptr,xsq,ysq,ans,0,1,converged);
	}
	else die("unknown integral method");
	ans = ans * sqrt(1-epsilon);
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::k_integral(const int nval_in, bool &converged)
{
	using std::sqrt;
#ifdef USE_STAN
	using stan::math::sqrt;
#endif
	nval_plus_half = nval_in + 0.5;
	converged = true; // will change if convergence not achieved
	QScalar ans;
	if (profile->integral_method == Romberg_Integration)
	{
		Romberg<std::function<QScalar(const QScalar)>,QScalar> romberg;
		ans = sqrt(1-epsilon)*romberg.integrate_open([this](auto x){return k_integrand_prime(x);}, 0, 1, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		QScalar (LensIntegral::*kptr)(QScalar) = &LensIntegral::k_integrand_prime;
		ans = sqrt(1-epsilon)*GaussIntegrate(kptr,0,1);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		QScalar (LensIntegral::*kptr)(QScalar) = &LensIntegral::k_integrand_prime;
		ans = sqrt(1-epsilon)*PattersonIntegrate(kptr,0,1,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		QScalar (LensIntegral::*kptr)(QScalar) = &LensIntegral::k_integrand_prime;
		ans = sqrt(1-epsilon)*FejerIntegrate(kptr,0,1,converged);
	}

	else die("unknown integral method");
	return ans;
}

// i,j,k integrals are in form similar to Keeton (2001), but generalized to allow for different
// definitions of the elliptical radius. I have also made the substitution
// u=w*w (easier for Gaussian quadrature; makes kappa singularity more manageable)

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::i_integrand_prime(const QScalar w)
{
	using std::sqrt;
#ifdef USE_STAN
	using stan::math::sqrt;
#endif
	u = w*w;
	qfac = 1 - epsilon*u;
	xisq = u*(xsqval + ysqval/qfac)*fsqinv;
	return (2*w*(xisq/u)*(profile->kapavg_spherical_generic)(xisq) / sqrt(qfac))/fsqinv;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::j_integrand_prime(const QScalar w)
{
	//using std::pow;
#ifdef USE_STAN
	using stan::math::pow;
#endif

	u = w*w;
	qfac = 1 - epsilon*u;
	return (2*w*profile->kappa_rsq(u*(xsqval + ysqval/qfac)*fsqinv) / pow(qfac, nval_plus_half));
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::j_integrand_vec(const QScalar w, const VecType& xsqval, const VecType& ysqval, VecType& ans)
{
#ifdef USE_STAN
	using stan::math::pow;
#endif

	QScalar u, qfac;
	VecType uxisq, kap;

	u = w*w;
	qfac = 1 - epsilon*u;

#ifdef USE_STAN
	if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
		uxisq = u*(xsqval + ysqval/qfac)*fsqinv;
		profile->kappa_rsq_vec(uxisq,kap);
		ans = 2*w*kap/pow(qfac, nval_plus_half);
	} else
#endif
	{
		uxisq = (u*(xsqval.array() + ysqval.array()/qfac)*fsqinv).matrix();
		profile->kappa_rsq_vec(uxisq,kap);
		ans = (2*w*kap.array()/pow(qfac, nval_plus_half)).matrix();
	}
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::k_integrand_prime(const QScalar w)
{
	//using std::pow;
#ifdef USE_STAN
	using stan::math::pow;
#endif

	u = w*w;
	qfac = 1 - epsilon*u;
	return fsqinv*(2*w*u*profile->kappa_rsq_deriv(u*(xsqval + ysqval/qfac)*fsqinv) / pow(qfac, nval_plus_half));
}

/*
// This version of i_integrand might still be useful in cases where we have no formula for the spherical deflection, since it only requires kappa_rsq_deriv. Keep in mind for later!
template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::i_integrand_v2(const QScalar xi)
{
	xisq = xi*xi;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-5) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);

	QScalar inside_fac = (2/epsilon)*(xsqval*(1-pow(1-epsilon*u,0.5)) - ysqval*(1-pow(1-epsilon*u,-0.5))); // this is the contribution from work done moving through the inside of the plates
	QScalar outside_fac = xisq*log((1-sqrt(1-epsilon))/(1-sqrt(1-epsilon*u))*(1+sqrt(1-epsilon*u))/(1+sqrt(1-epsilon)))/fsqinv; // contribution from work done moving outside the plates
	
	return (2*xi*(-profile->kappa_rsq_deriv(xisq))*(inside_fac+outside_fac));
}
*/

/*
template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::j_integrand_v2(const QScalar w)
{
	xisq = w*w;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	
	return (2*w*(-profile->kappa_rsq_deriv(xisq))*(1-pow(1-epsilon*u,mnval_plus_half)));
}
*/

/*
template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::k_integrand_v2(const QScalar w)
{
	xisq = w*w;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if ((epsilon*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	QScalar dxisq = (xsqval + ysqval/SQR(1-epsilon*u));

	return 2*w*profile->kappa_rsq_deriv(xisq)*u*pow(1-epsilon*u,-nval_plus_half)/dxisq;
}
*/

/******************** Integrals required when ellipticity gradient is present *********************/

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::i_integral_egrad(bool &converged)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
	using stan::math::sqrt;
#endif
	converged = true; // will change if convergence not achieved
	QScalar ans;

	QScalar xi = profile->elliptical_radius_root(xval,yval);
	QScalar xif = (xi < profile->xi_final_egrad) ? profile->xi_final_egrad : xi;

	QScalar costh, sinth;
	QScalar epf, thetaf;
	profile->ellipticity_function(xif,epf,thetaf);
	costh=cos(thetaf);
	sinth=sin(thetaf);
	QScalar xprime, yprime;
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;
	xsqval = xprime*xprime;
	ysqval = yprime*yprime;

	QScalar qfactor = sqrt(1 - epf);
	if (epf > 1e-4)
		ans = (2*qfactor/epf)*profile->kappa_rsq(xif*xif)*(xsqval*(1-qfactor) - ysqval*(1-1.0/qfactor)); // This is the boundary term
	else
		ans = qfactor*profile->kappa_rsq(xif*xif)*(xsqval + ysqval); // This is the boundary term

	if (profile->integral_method == Romberg_Integration)
	{
		Romberg<std::function<QScalar(const QScalar)>,QScalar> romberg;
		ans += romberg.integrate_open([this](auto x){return i_integrand_egrad(x);}, 0, xif, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		QScalar (LensIntegral::*iptr)(QScalar) = &LensIntegral::i_integrand_egrad;
		ans += GaussIntegrate(iptr,0,xif);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		QScalar (LensIntegral::*iptr)(QScalar) = &LensIntegral::i_integrand_egrad;
		ans += PattersonIntegrate(iptr,0,xif,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		QScalar (LensIntegral::*iptr)(QScalar) = &LensIntegral::i_integrand_egrad;
		ans += FejerIntegrate(iptr,0,xif,converged);

	}
	else die("unknown integral method");
	return ans;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::i_integrand_egrad(const QScalar xi)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
	using stan::math::sqrt;
#endif

	xisq = xi*xi;
	QScalar theta, costh, sinth, xprime, yprime, qufactor, qval;
	profile->ellipticity_function(xi,epsilon,theta);
	qval = sqrt(1-epsilon);
	fsqinv = (emode==0) ? 1 : qval;

	costh=cos(theta);
	sinth=sin(theta); // later, use trig identity to get sinth (don't forget to put in sign based on quadrant) to save a little time
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	xsqval = xprime*xprime;
	ysqval = yprime*yprime;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-4) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	qufactor = sqrt(1-epsilon*u);

	QScalar inside_fac, outside_fac;
	if (epsilon > 1e-4) {
		inside_fac = (2/epsilon)*(xsqval*(1-qufactor) - ysqval*(1-1.0/qufactor)); // this is the contribution from work done moving through the inside of the plates
		outside_fac = (u==1.0) ? 0.0 : xisq*log((1-qval)/(1-qufactor)*(1+qufactor)/(1+qval))/fsqinv; // contribution from work done moving outside the plates
	} else {
		inside_fac = u*(xsqval + ysqval);
		outside_fac = (u==1.0) ? 0.0 : xisq*log((1-epsilon*u/4)/(1-epsilon/4)/u);
	}
	
	return (2*xi*(-profile->kappa_rsq_deriv(xisq))*qval*(inside_fac+outside_fac));
}

/*
template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::j_integral_egrad(const int nval_in, bool &converged)
{
	converged = true; // will change if convergence not achieved
	QScalar ans;

	nval = nval_in;
	QScalar xi = profile->elliptical_radius_root(xval,yval);
	QScalar xif = (xi < profile->xi_final_egrad) ? profile->xi_final_egrad : xi;

	QScalar costh, sinth;
	QScalar epf, thetaf;
	profile->ellipticity_function(xif,epf,thetaf);
	costh=cos(thetaf);
	sinth=sin(thetaf);
	QScalar fac0, fac1;
	if (nval==0) { // Jxx integral
		fac0 = costh*costh;
		fac1 = -sinth*sinth;
	} else if (nval==1) { // Jxy integral
		fac0 = costh*sinth;
		fac1 = fac0;
	} else if (nval==2) { // Jyy integral
		fac0 = sinth*sinth;
		fac1 = -costh*costh;
	}
	if (epf > 1e-9)
		ans = 2*profile->kappa_rsq(xif*xif)*(sqrt(1-epf)/epf)*(fac0*(1-sqrt(1-epf)) + fac1*(1-1.0/sqrt(1-epf))); // This is the boundary term
	else
		ans = profile->kappa_rsq(xif*xif)*sqrt(1-epf)*(fac0 - fac1); // This is the boundary term

	if (profile->integral_method == Romberg_Integration)
	{
		QScalar (Romberg::*jptr)(const QScalar);
		jptr = static_cast<QScalar (Romberg::*)(const QScalar)> (&LensIntegral::j_integrand_egrad);
		ans += 2*romberg_open(jptr, 0, xif, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::j_integrand_egrad;
		ans += 2*GaussIntegrate(jptr,0,xif);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::j_integrand_egrad;
		ans += 2*PattersonIntegrate(jptr,0,xif,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::j_integrand_egrad;
		ans += 2*FejerIntegrate(jptr,0,xif,converged);

	}
	else die("unknown integral method");
	return ans;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::k_integral_egrad(const int nval_in, bool &converged)
{
	converged = true; // will change if convergence not achieved
	QScalar ans;

	nval = nval_in;
	QScalar xi = profile->elliptical_radius_root(xval,yval);
	if (profile->integral_method == Romberg_Integration)
	{
		QScalar (Romberg::*kptr)(const QScalar);
		kptr = static_cast<QScalar (Romberg::*)(const QScalar)> (&LensIntegral::k_integrand_egrad);
		ans = romberg_open(kptr, 0, xi, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		QScalar (LensIntegral::*kptr)(QScalar) = &LensIntegral::k_integrand_egrad;
		ans = GaussIntegrate(kptr,0,xi);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		QScalar (LensIntegral::*kptr)(QScalar) = &LensIntegral::k_integrand_egrad;
		ans = PattersonIntegrate(kptr,0,xi,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		QScalar (LensIntegral::*kptr)(QScalar) = &LensIntegral::k_integrand_egrad;
		ans = FejerIntegrate(kptr,0,xi,converged);

	}
	else die("unknown integral method");
	return ans;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::jprime_integral_egrad(const int nval_in, bool &converged)
{
	// If we only need the deflection, and not the hessian, these integrals are faster because it's just two integrals J_0' and J_1'
	converged = true; // will change if convergence not achieved
	QScalar ans;

	nval = nval_in;
	QScalar xi = profile->elliptical_radius_root(xval,yval);
	QScalar xif = (xi < profile->xi_final_egrad) ? profile->xi_final_egrad : xi;

	QScalar costh, sinth;
	QScalar epf, thetaf;
	profile->ellipticity_function(xif,epf,thetaf);
	costh=cos(thetaf);
	sinth=sin(thetaf);
	QScalar xprime, yprime;
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	QScalar fac0, fac1;
	if (nval==0) { // Jxx integral
		fac0 = xprime*costh;
		fac1 = yprime*sinth;
	} else if (nval==1) { // Jxy integral
		fac0 = xprime*sinth;
		fac1 = -yprime*costh;
	}
	if (epf > 1e-3)
		ans = 2*profile->kappa_rsq(xif*xif)*(sqrt(1-epf)/epf)*(fac0*(1-sqrt(1-epf)) + fac1*(1-1.0/sqrt(1-epf))); // This is the boundary term
	else
		ans = profile->kappa_rsq(xif*xif)*sqrt(1-epf)*(fac0 - fac1); // This is the boundary term

	if (profile->integral_method == Romberg_Integration)
	{
		QScalar (Romberg::*jptr)(const QScalar);
		jptr = static_cast<QScalar (Romberg::*)(const QScalar)> (&LensIntegral::jprime_integrand_egrad);
		ans += 2*romberg_open(jptr, 0, xif, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::jprime_integrand_egrad;
		ans += 2*GaussIntegrate(jptr,0,xif);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::jprime_integrand_egrad;
		ans += 2*PattersonIntegrate(jptr,0,xif,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		QScalar (LensIntegral::*jptr)(QScalar) = &LensIntegral::jprime_integrand_egrad;
		ans += 2*FejerIntegrate(jptr,0,xif,converged);

	}
	else die("unknown integral method");
	return ans;
}
*/

/*
template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::j_integrand_egrad(const QScalar xi)
{
	xisq = xi*xi;
	QScalar theta, costh, sinth, xprime, yprime, qufactor, qval;
	profile->ellipticity_function(xi,epsilon,theta);
	qval = sqrt(1-epsilon);
	fsqinv = (emode==0) ? 1 : qval;
	if ((nval==1) and ((theta==0.0) or (theta==M_PI))) return 0.0;

	costh=cos(theta);
	sinth=sin(theta); // later, use trig identity to get sinth (don't forget to put in sign based on quadrant) to save a little time
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	xsqval = xprime*xprime;
	ysqval = yprime*yprime;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	
	QScalar fac0, fac1, gfac;
	if (nval==0) { // Jxx integral
		fac0 = -costh*costh;
		fac1 = sinth*sinth;
	} else if (nval==1) { // Jxy integral
		fac0 = -costh*sinth;
		fac1 = fac0;
	} else if (nval==2) { // Jyy integral
		fac0 = -sinth*sinth;
		fac1 = costh*costh;
	}
	qufactor = sqrt(1-epsilon*u);
	if (epsilon > 1e-9)
		gfac = (fac0*(1-qufactor) + fac1*(1-1.0/qufactor))/epsilon; // I moved epsilon into the gfac so it can have a first order expansion in the limit of small epsilon
	else
		gfac = (fac0 - fac1)*u/2; // I moved epsilon into the gfac so it can have a first order expansion in the limit of small epsilon
	return (2*xi*(profile->kappa_rsq_deriv(xisq))*qval*gfac);
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::k_integrand_egrad(const QScalar xi)
{
	xisq = xi*xi;
	QScalar theta, costh, sinth, xprime, yprime, qval;
	profile->ellipticity_function(xi,epsilon,theta);
	qval = sqrt(1-epsilon);
	fsqinv = (emode==0) ? 1 : qval;
	costh=cos(theta);
	sinth=sin(theta); // later, use trig identity to get sinth (don't forget to put in sign based on quadrant) to save a little time
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	xsqval = xprime*xprime;
	ysqval = yprime*yprime;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	
	QScalar fac0, fac1, qufactor;
	qufactor = 1-epsilon*u;
	if (nval==0) { // Jxx integral
		fac0 = xprime*costh - yprime*sinth/qufactor;
		fac1 = fac0;
	} else if (nval==1) { // Jxy integral
		fac0 = xprime*costh - yprime*sinth/qufactor;
		fac1 = xprime*sinth + yprime*costh/qufactor;
	} else if (nval==2) { // Jyy integral
		fac0 = xprime*sinth + yprime*costh/qufactor;
		fac1 = fac0;
	}
	QScalar dxisq = xsqval + ysqval/(qufactor*qufactor);
	return (2*xi*(profile->kappa_rsq_deriv(xisq))*qval*fac0*fac1*u/sqrt(qufactor))/dxisq;
	//return 2*xi*profile->kappa_rsq_deriv(xisq)*u*qval*pow(1-epsilon*u,-(nval+0.5))/dxisq;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::jprime_integrand_egrad(const QScalar xi)
{
	xisq = xi*xi;
	QScalar theta, costh, sinth, xprime, yprime, qufactor, qval;
	profile->ellipticity_function(xi,epsilon,theta);
	qval = sqrt(1-epsilon);
	fsqinv = (emode==0) ? 1 : qval;

	costh=cos(theta);
	sinth=sin(theta); // later, use trig identity to get sinth (don't forget to put in sign based on quadrant) to save a little time
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	xsqval = xprime*xprime;
	ysqval = yprime*yprime;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	
	QScalar fac0, fac1, gfac;
	if (nval==0) { // Jxx integral
		fac0 = -xprime*costh;
		fac1 = -yprime*sinth;
	} else if (nval==1) { // Jxy integral
		fac0 = -xprime*sinth;
		fac1 = yprime*costh;
	}
	qufactor = sqrt(1-epsilon*u);
	if (epsilon > 1e-9)
		gfac = (fac0*(1-qufactor) + fac1*(1-1.0/qufactor))/epsilon; // I moved epsilon into the gfac so it can have a first order expansion in the limit of small epsilon
	else
		gfac = (fac0 - fac1)*u/2; // I moved epsilon into the gfac so it can have a first order expansion in the limit of small epsilon

	return (2*xi*(profile->kappa_rsq_deriv(xisq))*qval*gfac);
}
*/

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::j_integral_egrad_mult(QScalar *jint, bool &converged)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
	using stan::math::sqrt;
#endif

	if (n_mult < 3) die("n_mult must be at least 3 to use j_integral_egrad_mult");
	converged = true; // will change if convergence not achieved
	QScalar ans;

	QScalar xi = profile->elliptical_radius_root(xval,yval);
	QScalar xif = (xi < profile->xi_final_egrad) ? profile->xi_final_egrad : xi;

	QScalar epf, thetaf;
	profile->ellipticity_function(xif,epf,thetaf);
	void (LensIntegral::*jptr)(const QScalar, QScalar*) = &LensIntegral::j_integrand_egrad_mult;
	if (profile->integral_method == Gaussian_Quadrature)
	{
		GaussIntegrate(jptr,0,xif,jint,3);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		PattersonIntegrate(jptr,0,xif,jint,3,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		FejerIntegrate(jptr,0,xif,jint,3,converged);
	}
	else die("cannot use chosen integral method with egrad_mult");
	for (int i=0; i < 3; i++) jint[i] *= 2;

	QScalar costh, sinth;
	costh=cos(thetaf);
	sinth=sin(thetaf);
	QScalar fac0[3], fac1[3];
	fac0[0] = costh*costh;
	fac1[0] = -sinth*sinth;
	fac0[1] = costh*sinth;
	fac1[1] = fac0[1];
	fac0[2] = sinth*sinth;
	fac1[2] = -costh*costh;
	QScalar jfac = profile->kappa_rsq(xif*xif);
	if (epf > 1e-9) {
		for (int i=0; i < 3; i++) jint[i] += 2*jfac*(sqrt(1-epf)/epf)*(fac0[i]*(1-sqrt(1-epf)) + fac1[i]*(1-1.0/sqrt(1-epf))); // This is the boundary term
	} else {
		for (int i=0; i < 3; i++) jint[i] += jfac*sqrt(1-epf)*(fac0[i] - fac1[i]); // This is the boundary term
	}
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::k_integral_egrad_mult(QScalar *kint, bool &converged)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
	using stan::math::sqrt;
#endif

	if (n_mult < 3) die("n_mult must be at least 3 to use k_integral_egrad_mult");
	converged = true; // will change if convergence not achieved
	QScalar ans;

	QScalar xi = profile->elliptical_radius_root(xval,yval);
	void (LensIntegral::*kptr)(const QScalar, QScalar*) = &LensIntegral::k_integrand_egrad_mult;
	if (profile->integral_method == Gaussian_Quadrature)
	{
		GaussIntegrate(kptr,0,xi,kint,3);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		PattersonIntegrate(kptr,0,xi,kint,3,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		FejerIntegrate(kptr,0,xi,kint,3,converged);
	}
	else die("cannot use chosen integral method with egrad_mult");
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::jprime_integral_egrad_mult(QScalar *jint, bool &converged)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
	using stan::math::sqrt;
#endif

	if (n_mult < 2) die("n_mult must be at least 2 to use jprime_integral_egrad_mult");
	// If we only need the deflection, and not the hessian, these integrals are faster because it's just two integrals J_0' and J_1'
	converged = true; // will change if convergence not achieved
	QScalar ans;

	QScalar xi = profile->elliptical_radius_root(xval,yval);
	QScalar xif = (xi < profile->xi_final_egrad) ? profile->xi_final_egrad : xi;

	QScalar epf, thetaf;
	profile->ellipticity_function(xif,epf,thetaf);
	void (LensIntegral::*jptr)(const QScalar, QScalar*) = &LensIntegral::jprime_integrand_egrad_mult;
	if (profile->integral_method == Gaussian_Quadrature)
	{
		GaussIntegrate(jptr,0,xif,jint,2);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		PattersonIntegrate(jptr,0,xif,jint,2,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		FejerIntegrate(jptr,0,xif,jint,2,converged);
	}
	else die("cannot use chosen integral method with egrad_mult");
	jint[0] *= 2;
	jint[1] *= 2;

	QScalar costh, sinth;
	costh=cos(thetaf);
	sinth=sin(thetaf);
	QScalar xprime, yprime;
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	QScalar fac0[2], fac1[2];
	fac0[0] = xprime*costh;
	fac1[0] = yprime*sinth;
	fac0[1] = xprime*sinth;
	fac1[1] = -yprime*costh;
	QScalar jfac = profile->kappa_rsq(xif*xif);

	if (epf > 1e-3) {
		for (int i=0; i < 2; i++) jint[i] += 2*jfac*(sqrt(1-epf)/epf)*(fac0[i]*(1-sqrt(1-epf)) + fac1[i]*(1-1.0/sqrt(1-epf))); // This is the boundary term
	} else {
		for (int i=0; i < 2; i++) jint[i] += jfac*sqrt(1-epf)*(fac0[i] - fac1[i]); // This is the boundary term
	}
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::j_integrand_egrad_mult(const QScalar xi, QScalar* jint)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
	using stan::math::sqrt;
#endif

	xisq = xi*xi;
	QScalar theta, costh, sinth, xprime, yprime, qufactor, qval, jfac;
	profile->ellipticity_function(xi,epsilon,theta);
	qval = sqrt(1-epsilon);
	fsqinv = (emode==0) ? 1 : qval;
	//if ((nval==1) and ((theta==0.0) or (theta==M_PI))) return 0.0;
	jfac = 2*xi*(profile->kappa_rsq_deriv(xisq))*qval;

	costh=cos(theta);
	sinth=sin(theta); // later, use trig identity to get sinth (don't forget to put in sign based on quadrant) to save a little time
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	xsqval = xprime*xprime;
	ysqval = yprime*yprime;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	qufactor = sqrt(1-epsilon*u);
	
	QScalar fac0[3], fac1[3];
	fac0[0] = -costh*costh; // Jxx integral
	fac1[0] = sinth*sinth; // Jxx integral
	fac0[1] = -costh*sinth; // Jxy integral
	fac1[1] = fac0[1]; // Jxy integral
	fac0[2] = -sinth*sinth; // Jyy integral
	fac1[2] = costh*costh; // Jyy integral
	if (epsilon > 1e-9) {
		for (int i=0; i < 3; i++) jint[i] = jfac*(fac0[i]*(1-qufactor) + fac1[i]*(1-1.0/qufactor))/epsilon; // I moved epsilon into the gfac so it can have a first order expansion in the limit of small epsilon
	} else {
		for (int i=0; i < 3; i++) jint[i] = jfac*(fac0[i] - fac1[i])*u/2; // I moved epsilon into the gfac so it can have a first order expansion in the limit of small epsilon
	}
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::k_integrand_egrad_mult(const QScalar xi, QScalar* kint)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
	using stan::math::sqrt;
#endif

	xisq = xi*xi;
	QScalar theta, costh, sinth, xprime, yprime, qval, kfac, qufactor, dxisq;
	profile->ellipticity_function(xi,epsilon,theta);
	qval = sqrt(1-epsilon);
	fsqinv = (emode==0) ? 1 : qval;
	costh=cos(theta);
	sinth=sin(theta); // later, use trig identity to get sinth (don't forget to put in sign based on quadrant) to save a little time
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	xsqval = xprime*xprime;
	ysqval = yprime*yprime;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	qufactor = 1-epsilon*u;
	dxisq = xsqval + ysqval/(qufactor*qufactor);
	kfac = 2*xi*(profile->kappa_rsq_deriv(xisq))*qval*u/sqrt(qufactor)/dxisq;
	
	QScalar fac0[3], fac1[3];
	fac0[0] = xprime*costh - yprime*sinth/qufactor;
	fac1[0] = fac0[0];
	fac0[1] = fac0[0];
	fac1[1] = xprime*sinth + yprime*costh/qufactor;
	fac0[2] = fac1[1];
	fac1[2] = fac1[1];
	for (int i=0; i < 3; i++) kint[i] = kfac*fac0[i]*fac1[i];

	//return (2*xi*profile->kappa_rsq_deriv(xisq)*fac0*fac1*u/sqrt(qufactor))/dxisq;
	//return 2*xi*profile->kappa_rsq_deriv(xisq)*u*qval*pow(1-epsilon*u,-(nval+0.5))/dxisq;
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::jprime_integrand_egrad_mult(const QScalar xi, QScalar* jint)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
	using stan::math::sqrt;
#endif

	xisq = xi*xi;
	QScalar theta, costh, sinth, xprime, yprime, qufactor, qval, jfac;
	profile->ellipticity_function(xi,epsilon,theta);
	qval = sqrt(1-epsilon);
	fsqinv = (emode==0) ? 1 : qval;
	jfac = 2*xi*(profile->kappa_rsq_deriv(xisq))*qval;

	costh=cos(theta);
	sinth=sin(theta); // later, use trig identity to get sinth (don't forget to put in sign based on quadrant) to save a little time
	xprime = xval*costh + yval*sinth;
	yprime = -xval*sinth + yval*costh;

	xsqval = xprime*xprime;
	ysqval = yprime*yprime;
	qfac = xsqval + ysqval + epsilon*xisq/fsqinv;
	if (xisq > fsqinv*(xsqval+ysqval/(1-epsilon))) u = 1.0;
	else if ((epsilon*xsqval) < 1e-9) u = xisq/qfac/fsqinv;
	else u = (qfac - sqrt(qfac*qfac - 4*epsilon*xsqval*xisq/fsqinv)) / (2*epsilon*xsqval);
	qufactor = sqrt(1-epsilon*u);
	
	QScalar fac0[2], fac1[2];
	fac0[0] = -xprime*costh;
	fac1[0] = -yprime*sinth;
	fac0[1] = -xprime*sinth;
	fac1[1] = yprime*costh;

	if (epsilon > 1e-9) {
		for (int i=0; i < 2; i++) jint[i] = jfac*(fac0[i]*(1-qufactor) + fac1[i]*(1-1.0/qufactor))/epsilon; // I moved epsilon into the gfac so it can have a first order expansion in the limit of small epsilon
	} else {
		for (int i=0; i < 2; i++) jint[i] = jfac*(fac0[i] - fac1[i])*u/2; // I moved epsilon into the gfac so it can have a first order expansion in the limit of small epsilon
	}
}


template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::calculate_fourier_integrals(const int mval_in, const int fourier_ival_in, const bool cosmode_in, const QScalar rval, QScalar& ileft, QScalar& iright, bool &converged)
{
	mval = mval_in;
	fourier_ival = fourier_ival_in;
	cosmode = cosmode_in;
	converged = true; // will change if convergence not achieved
	QScalar ans;

	if (profile->integral_method == Romberg_Integration)
	{
		Romberg<std::function<QScalar(const QScalar)>,QScalar> romberg;
		ileft = romberg.integrate_open([this](auto x){return ileft_integrand(x);}, 0, rval, profile->integral_tolerance, 5);
		iright = romberg.integrate_open([this](auto x){return iright_integrand(x);}, 0, 1.0/rval, profile->integral_tolerance, 5);
	}
	else if (profile->integral_method == Gaussian_Quadrature)
	{
		QScalar (LensIntegral::*fptr)(QScalar) = &LensIntegral::ileft_integrand;
		ileft = GaussIntegrate(fptr,0,rval);
		fptr = &LensIntegral::iright_integrand;
		iright = GaussIntegrate(fptr,0,1.0/rval);
	}
	else if (profile->integral_method == Gauss_Patterson_Quadrature)
	{
		QScalar (LensIntegral::*fptr)(QScalar) = &LensIntegral::ileft_integrand;
		ileft = PattersonIntegrate(fptr,0,rval,converged);
		fptr = &LensIntegral::iright_integrand;
		iright = PattersonIntegrate(fptr,0,1.0/rval,converged);
	}
	else if (profile->integral_method == Fejer_Quadrature)
	{
		QScalar (LensIntegral::*fptr)(QScalar) = &LensIntegral::ileft_integrand;
		ileft = FejerIntegrate(fptr,0,rval,converged);
		fptr = &LensIntegral::iright_integrand;
		iright = FejerIntegrate(fptr,0,1.0/rval,converged);
	}
	else die("unknown integral method");

	/*
	cc_points = profile->cc_points;
	cc_weights = profile->cc_weights;
	cc_funcs = new QScalar[profile->cc_N];
	QScalar (LensIntegral::*fptr)(QScalar) = &LensIntegral::ileft_integrand;
	profile->cc_tolerance = 1e-5;
	QScalar ileftcheck = FejerIntegrate(fptr,0,rval,converged);
	fptr = &LensIntegral::iright_integrand;
	QScalar irightcheck = FejerIntegrate(fptr,0,1.0/rval,converged);
	delete[] cc_funcs;
	if ((abs(ileft-ileftcheck) > 1e-3*abs(ileftcheck)) or (abs(iright-irightcheck) > 1e-3*abs(irightcheck))) {
		cout << "CHECK(r=" << rval << "): " << ileftcheck << " " << ileft << " " << (abs((ileft-ileftcheck)/ileftcheck)) << " " << irightcheck << " " << iright << " " << (abs((iright-irightcheck)/irightcheck)) << endl;
	}
	*/
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::fourier_kappa_m(const QScalar r, const QScalar phi, const int mval_in, const int fourier_ival_in)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
#endif

	QScalar ans, mphi;
	mval = mval_in;
	fourier_ival = fourier_ival_in;
	mphi = mval*phi;
	cosmode = true;
	ans = fourier_kappa_perturbation(r)*cos(mphi);
	cosmode = false;
	ans += fourier_kappa_perturbation(r)*sin(mphi);
	return ans;
}


template <typename VecType, typename QScalar>
inline QScalar LensIntegral<VecType,QScalar>::fourier_kappa_perturbation(const QScalar r)
{
#ifdef USE_STAN
	using stan::math::cos;
	using stan::math::sin;
#endif

	if (profile->ellipticity_gradient) {
		//phi0 = profile->angle_function(r);
	}
	if (profile->fourier_gradient) {
		//profile->fourier_mode_function(r,cosamps,sinamps); // lensing multipoles depend on r, not xi, so we follow the same restriction here
	}
	QScalar kapm;
	if (phi0==0) {
		if (cosmode) kapm = cosamps[fourier_ival];
		else kapm = sinamps[fourier_ival];
	} else {
		if (cosmode) kapm = cosamps[fourier_ival]*cos(mval*phi0) - sinamps[fourier_ival]*sin(mval*phi0);
		else kapm = cosamps[fourier_ival]*sin(mval*phi0) + sinamps[fourier_ival]*cos(mval*phi0);
	}
	QScalar rsq = r*r;
	//NOTE: this doesn't work for emode=3 (can't use kappa_rsq_deriv). extend later?
	kapm *= 2*profile->kappa_rsq_deriv(rsq)*rsq; // this allows it to approximate perturbing the elliptical radius (via first order term in Taylor expansion in (r + dr))
	return kapm;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::ileft_integrand(const QScalar r)
{
#ifdef USE_STAN
	using stan::math::pow;
#endif
	return pow(r,mval+1)*fourier_kappa_perturbation(r);
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::iright_integrand(const QScalar u) // here, u = 1/r
		//ans = sqrt(1-epsilon)*PattersonIntegrate(jptr,0,1,converged);
		//ans = sqrt(1-epsilon)*PattersonIntegrate(jptr,0,1,converged);
{
#ifdef USE_STAN
	using stan::math::pow;
#endif
	return pow(u,mval-3)*fourier_kappa_perturbation(1.0/u);
}

/************************************* Integration algorithms *************************************/

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::GaussIntegrate(QScalar (LensIntegral::*func)(const QScalar), const QScalar a, const QScalar b)
{
	QScalar result = 0;

	for (int i = 0; i < GaussQuad::numberOfPoints; i++)
		result += gaussweights[i]*(this->*func)(((a+b) + (b-a)*gausspoints[i])/2.0);

	return (b-a)*result/2.0;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::PattersonIntegrate(QScalar (LensIntegral::*func)(const QScalar), const QScalar a, const QScalar b, bool& converged)
{
	using std::abs;
	QScalar result=0, result_old;
	int i, level=0, istep, istart;
	QScalar absum = (a+b)/2, abdif = (b-a)/2;
	double *weightptr;
	converged = true; // will change to false if convergence not achieved

	int order, j;
	do {
		weightptr = Patterson::pat_weights[level];
		result_old = result;
		order = Patterson::pat_orders[level];
		istep = 512 / (order+1);
		istart = istep - 1;
		istep *= 2;
		result = 0;
		// Note, the pat_funcs[i] is not a problem for multiple OpenMP threads because a separate LensIntegral object was
		// created for each thread
		for (j=0, i=istart; j < order; j += 2, i += istep) {
			pat_funcs[i] = (this->*func)(absum + abdif*Patterson::pat_points[i]);
			result += weightptr[j]*pat_funcs[i];
		}
		for (j=1, i=istep-1; j < order; j += 2, i += istep) {
			result += weightptr[j]*pat_funcs[i];
		}
		//if (level > 1) cout << "n=" << nval << ", level " << level << ": j=" << result << ", j_old=" << result_old << ", diff=" << abs(result-result_old) << " tol=" << (profile->pat_tolerance*abs(result)) << endl;

		if ((level > 1) and (abs(result-result_old) < profile->integral_tolerance*abs(result))) {
			//cout << "patterson converged at level " << level << endl;
			break;
		}
		//if ((level > 1) and (abs(result-result_old) < profile->integral_tolerance)) break;
	} while (++level < 9);

	if (level == 9) {
		if ((result*0.0 != 0.0) or (result > 1e100)) warn("integration gave absurdly large or infinite number; suggests numerical problems in evaluating the integrand");
		// If Gauss-Legendre is set up with at least 1023 points, then switch to this to get a (hopefully) more accurate value
		if (GaussQuad::numberOfPoints >= 511) {
			gausspoints = GaussQuad::points;
			gaussweights = GaussQuad::weights;

			result = 0;
			for (int i = 0; i < GaussQuad::numberOfPoints; i++)
				result += gaussweights[i]*(this->*func)(absum + abdif*gausspoints[i]);
		}
		converged = false;
	}

	//cout << "INTEGRAL=" << (abdif*result) << endl;
	return abdif*result;
}

template <typename VecType, typename QScalar>
QScalar LensIntegral<VecType,QScalar>::FejerIntegrate(QScalar (LensIntegral::*func)(QScalar), QScalar a, QScalar b, bool &converged)
{
	using std::abs;
	// Fejer's quadrature rule--seems to be require slightly more function eval's than Patterson quadrature, but can allow for more
	// points in case integrand doesn't converge easily
	QScalar result = 0, result_old;
	int i, level = 0, istep, istart;
	QScalar abavg = (a+b)/2, abdif = (b-a)/2;
	converged = true; // until proven otherwise
	double *weightptr;
	level = 1;
	cc_funcs[0] = 0;
	cc_funcs[Fejer::cc_N-1] = (this->*func)(abavg);

	int lval, j;
	do {
		weightptr = Fejer::cc_weights[level];
		result_old = result;
		lval = Fejer::cc_lvals[level];
		istart = (Fejer::cc_N-1) / lval;
		istep = istart*2;
		result = 0;
		for (j=1, i=istart; j < lval; j += 2, i += istep) {
			cc_funcs[i] = (this->*func)(abavg + abdif*Fejer::cc_points[i]) + (this->*func)(abavg - abdif*Fejer::cc_points[i]);
			result += weightptr[j]*cc_funcs[i];
		}
		for (j=2, i=istep; j <= lval; j += 2, i += istep) {
			result += weightptr[j]*cc_funcs[i];
		}
		if ((level > 1) and (abs(result-result_old) < profile->integral_tolerance*abs(result))) break;
	} while (++level < Fejer::cc_nlevels);

	if (level==Fejer::cc_nlevels) {
		if ((result*0.0 != 0.0) or (result > 1e100)) warn("integration gave absurdly large or infinite number; suggests numerical problems in evaluating the integrand");
		converged = false;
		// If Gauss-Legendre is set up with at least cc_N points, then switch to this to get a (hopefully) more accurate value
		if (GaussQuad::numberOfPoints >= Fejer::cc_N) {
			gausspoints = GaussQuad::points;
			gaussweights = GaussQuad::weights;

			result = 0;
			for (int i = 0; i < GaussQuad::numberOfPoints; i++)
				result += gaussweights[i]*(this->*func)(abavg + abdif*gausspoints[i]);
		}
		converged = false;
	}
	return abdif*result;
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::GaussIntegrateVec(void (LensIntegral::*func)(const QScalar, const VecType&, const VecType&, VecType&), const VecType& xsq, const VecType& ysq, VecType& result, const QScalar a, const QScalar b)
{
	result = Eigen::VectorXd::Zero(xsq.size());
	VecType ans;
	for (int i = 0; i < GaussQuad::numberOfPoints; i++) {
		(this->*func)(((a+b) + (b-a)*gausspoints[i])/2.0,xsq,ysq,ans);
		result = result + gaussweights[i]*ans;
	}
#ifdef USE_STAN
	if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
		result = ((b-a)/2)*result;
	} else
#endif
	{
		result = (((b-a)/2)*result.array()).matrix();
	}
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::PattersonIntegrateVec(void (LensIntegral::*func)(const QScalar, const VecType&, const VecType&, VecType&), const VecType& xsq, const VecType& ysq, VecType& result, const QScalar a, const QScalar b, bool& converged)
{
	using std::abs;

	result = Eigen::VectorXd::Zero(xsq.size());

	VecType result_old, ans;
	int i, level=0, istep, istart;
	QScalar absum = (a+b)/2, abdif = (b-a)/2;
	double *weightptr;
	converged = true;

	int order, j;

	do {
		weightptr = Patterson::pat_weights[level];
		result_old = result;
		order = Patterson::pat_orders[level];
		istep = 512 / (order+1);
		istart = istep - 1;
		istep *= 2;
		result = Eigen::VectorXd::Zero(xsq.size());
		for (j=0, i=istart; j < order; j += 2, i += istep) {
			(this->*func)(absum + abdif*Patterson::pat_points[i],xsq,ysq,ans);
			pat_funcs_vec[i] = ans;
			result = result + weightptr[j]*pat_funcs_vec[i];
		}
		for (j=1, i=istep-1; j < order; j += 2, i += istep) {
			result = result + weightptr[j]*pat_funcs_vec[i];
		}

		Eigen::Index imax;
#ifdef USE_STAN
		if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
			double maxdiff = (result - result_old).val().cwiseAbs().maxCoeff(&imax);
			if ((level > 1) and (maxdiff < profile->integral_tolerance*abs(result.val()(imax)))) {
				break;
			}
		} else
#endif
		{
			double maxdiff = (result - result_old).cwiseAbs().maxCoeff(&imax);
			if ((level > 1) and (maxdiff < profile->integral_tolerance*abs(result(imax)))) {
				break;
			}
		}
	} while (++level < 9);

	if (level == 9) {
#ifdef USE_STAN
		if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
			if (((result.val().array()*0.0 != 0.0).any()) or ((result.val().array() > 1e100).any()))
				warn("integration gave absurdly large or infinite number; suggests numerical problems in evaluating the integrand");
		} else
#endif
		{
			if (((result.array()*0.0 != 0.0).any()) or ((result.array() > 1e100).any()))
				warn("integration gave absurdly large or infinite number; suggests numerical problems in evaluating the integrand");
		}

		if (GaussQuad::numberOfPoints >= 511) {
			gausspoints = GaussQuad::points;
			gaussweights = GaussQuad::weights;
			result = Eigen::VectorXd::Zero(xsq.size());
			for (int i = 0; i < GaussQuad::numberOfPoints; i++) {
				(this->*func)(absum + abdif*gausspoints[i],xsq,ysq,ans);
				result = result + gaussweights[i]*ans;
			}
		}

		converged = false;
	}
	//if (converged) std::cout << "Converged after " << level << " levels" << std::endl;

#ifdef USE_STAN
	if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
		result = abdif*result;
	} else
#endif
	{
		result = (abdif*result.array()).matrix();
	}
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::FejerIntegrateVec(void (LensIntegral::*func)(const QScalar, const VecType&, const VecType&, VecType&), const VecType& xsq, const VecType& ysq, VecType& result, const QScalar a, const QScalar b, bool& converged)
{
	using std::abs;

	int i, j, level = 0, istep, istart;
	QScalar abavg = (a+b)/2, abdif = (b-a)/2;
	double *weightptr;

	converged = true;

	VecType result_old, tmp, fplus, fminus;

	level = 1;

	// initialize endpoints
	(this->*func)(abavg,xsq,ysq,tmp);
	cc_funcs_vec[Fejer::cc_N - 1] = tmp;

	result = Eigen::VectorXd::Zero(xsq.size());
	cc_funcs_vec[0] = result;

	int lval;

	do {
		weightptr = Fejer::cc_weights[level];
		result_old = result;
		lval = Fejer::cc_lvals[level];
		istart = (Fejer::cc_N - 1) / lval;
		istep = istart * 2;

		for (j = 1, i = istart; j < lval; j += 2, i += istep) {
			(this->*func)(abavg + abdif*Fejer::cc_points[i],xsq,ysq,fplus);
			(this->*func)(abavg - abdif*Fejer::cc_points[i],xsq,ysq,fminus);
			cc_funcs_vec[i] = fplus + fminus;
			result = result + weightptr[j] * cc_funcs_vec[i];
		}

		for (j = 2, i = istep; j <= lval; j += 2, i += istep) {
			result = result + weightptr[j] * cc_funcs_vec[i];
		}

		if (level > 1) {
			Eigen::Index imax;
#ifdef USE_STAN
			if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
				double maxdiff = (result - result_old).val().cwiseAbs().maxCoeff(&imax);
				if (maxdiff < profile->integral_tolerance * abs(result.val()(imax))) {
					converged = true;
					break;
				}
			} else
#endif
			{
				double maxdiff = (result - result_old).cwiseAbs().maxCoeff(&imax);
				if (maxdiff < profile->integral_tolerance * abs(result(imax))) {
					converged = true;
					break;
				}
			}
		}
	} while (++level < Fejer::cc_nlevels);

	if (level == Fejer::cc_nlevels) {
#ifdef USE_STAN
		if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
			if (((result.val().array()*0.0 != 0.0).any()) ||
			    ((result.val().array() > 1e100).any()))
				warn("integration gave absurdly large or infinite number; suggests numerical problems in evaluating the integrand");
		} else
#endif
		{
			if (((result.array()*0.0 != 0.0).any()) ||
			    ((result.array() > 1e100).any()))
				warn("integration gave absurdly large or infinite number; suggests numerical problems in evaluating the integrand");
		}

		converged = false;

		if (GaussQuad::numberOfPoints >= Fejer::cc_N) {
			gausspoints = GaussQuad::points;
			gaussweights = GaussQuad::weights;
			result = Eigen::VectorXd::Zero(xsq.size());
			for (int k = 0; k < GaussQuad::numberOfPoints; k++) {
				VecType f;
				(this->*func)(abavg + abdif*gausspoints[k], xsq, ysq, f);
				result = result + gaussweights[k] * f;
			}
		}
	}

#ifdef USE_STAN
	if constexpr (std::is_same_v<VecType, stan::math::var_value<Eigen::VectorXd>>) {
		result = abdif*result;
	} else
#endif
	{
		result = (abdif*result.array()).matrix();
	}
}

/***************************** Integration algorithms (for simulateneous integrations) *******************************/

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::GaussIntegrate(void (LensIntegral::*func)(const QScalar, QScalar*), const QScalar a, const QScalar b, QScalar* results, const int n_funcs)
{
	int i,j;
	for (j=0; j < n_funcs; j++) results[j] = 0;
	QScalar *funcs = new QScalar[n_funcs];

	for (i = 0; i < GaussQuad::numberOfPoints; i++) {
		(this->*func)(((a+b) + (b-a)*gausspoints[i])/2.0,funcs);
		for (j=0; j < n_funcs; j++) results[j] += gaussweights[i]*funcs[j];
	}
	for (j=0; j < n_funcs; j++) results[j] = (b-a)*results[j]/2.0;
	delete[] funcs;
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::PattersonIntegrate(void (LensIntegral::*func)(const QScalar, QScalar*), const QScalar a, const QScalar b, QScalar* results, const int n_funcs, bool& converged)
{
	using std::abs;
	int i,j,k;
	bool at_least_one_converged;
	QScalar *results_old = new QScalar[n_funcs];
	bool *func_converged = new bool[n_funcs];
	for (k=0; k < n_funcs; k++) {
		results[k] = 0;
		func_converged[k] = false;
	}

	int level=0, istep, istart;
	QScalar absum = (a+b)/2, abdif = (b-a)/2;
	double *weightptr;
	converged = false;

	int order;

	do {
		weightptr = Patterson::pat_weights[level];
		for (k=0; k < n_funcs; k++) {
			results_old[k] = results[k];
			results[k] = 0;
		}
		order = Patterson::pat_orders[level];
		istep = 512 / (order+1);
		istart = istep - 1;
		istep *= 2;
		// Note, the pat_funcs_mult[i][k] is not a problem for multiple OpenMP threads because a separate LensIntegral object was
		// created for each thread
		for (j=0, i=istart; j < order; j += 2, i += istep) {
			(this->*func)(absum + abdif*Patterson::pat_points[i],pat_funcs_mult[i]);
			for (k=0; k < n_funcs; k++) results[k] += weightptr[j]*pat_funcs_mult[i][k];
		}
		for (j=1, i=istep-1; j < order; j += 2, i += istep) {
			for (k=0; k < n_funcs; k++) results[k] += weightptr[j]*pat_funcs_mult[i][k];
		}
		at_least_one_converged = false;
		if (level > 1) {
			//cout << "level " << level << ": j0=" << results[0] << ", j0_old=" << results_old[0] << ".... j1=" << results[1] << ", j1_old=" << results_old[1] << endl;
			for (k=0; k < n_funcs; k++) {
				if (!func_converged[k]) {
					if (abs(results[k]-results_old[k]) < profile->integral_tolerance*abs(results[k])) {
						//cout << "j[" << k << "] converged because diff=" << abs(results[k]-results_old[k]) << " is less than " << (profile->integral_tolerance*abs(results[k]))  << endl;
						func_converged[k] = true;
						at_least_one_converged = true;
					}
				}
			}
			if (at_least_one_converged) { // we won't bother to check if they're all converged, if we haven't had at least one integral converge
				converged = true;
				for (k=0; k < n_funcs; k++) {
					if (!func_converged[k]) converged = false;
				}
			}
			if (converged) {
				//cout << "patterson_mult converged after level " << level << endl;
				break;
			}
		}
		//if ((level > 1) and (abs(result-result_old) < profile->integral_tolerance)) break;
	} while (++level < 9);

	if (level == 9) {
		if (converged) die("converged should not be true if we reached level 9!");
		for (k=0; k < n_funcs; k++) {
			if ((results[k]*0.0 != 0.0) or (results[k] > 1e100)) warn("integration gave absurdly large or infinite number; suggests numerical problems in evaluating the integrand");
		}
		// If Gauss-Legendre is set up with at least 1023 points, then switch to this to get a (hopefully) more accurate value
		if (GaussQuad::numberOfPoints >= 511) {
			gausspoints = GaussQuad::points;
			gaussweights = GaussQuad::weights;

			for (k=0; k < n_funcs; k++) {
				results[k] = 0;
			}

			QScalar *funcs = new QScalar[n_funcs];
			for (i = 0; i < GaussQuad::numberOfPoints; i++) {
				(this->*func)(((a+b) + (b-a)*gausspoints[i])/2.0,funcs);
				for (k=0; k < n_funcs; k++) results[k] += gaussweights[i]*funcs[k];
			}
			delete[] funcs;
		}
		converged = false;
	}

	for (k=0; k < n_funcs; k++) results[k] *= abdif;
	delete[] func_converged;
	delete[] results_old;
}

template <typename VecType, typename QScalar>
void LensIntegral<VecType,QScalar>::FejerIntegrate(void (LensIntegral::*func)(const QScalar, QScalar*), const QScalar a, const QScalar b, QScalar* results, const int n_funcs, bool& converged)
{
	using std::abs;
	// Fejer's quadrature rule--seems to be require slightly more function eval's than Patterson quadrature, but can allow for more
	// points in case integrand doesn't converge easily
	int i, j, k, level = 0, istep, istart;
	QScalar abavg = (a+b)/2, abdif = (b-a)/2;
	converged = false;
	double *weightptr;

	bool at_least_one_converged;
	QScalar *results_old = new QScalar[n_funcs];
	QScalar *funcs = new QScalar[n_funcs];
	bool *func_converged = new bool[n_funcs];
	for (k=0; k < n_funcs; k++) {
		results[k] = 0;
		func_converged[k] = false;
	}

	level = 1;
	for (k=0; k < n_funcs; k++) cc_funcs_mult[k][0] = 0;
	(this->*func)(abavg,cc_funcs_mult[Fejer::cc_N-1]);

	int lval;
	do {
		//cout << "Level " << level << " (versus lmax=" << profile->cc_nlevels << ")" << endl;
		weightptr = Fejer::cc_weights[level];
		for (k=0; k < n_funcs; k++) {
			results_old[k] = results[k];
			//cout << "integral " << k << ": " << results[k] << endl;
			results[k] = 0;
		}
		lval = Fejer::cc_lvals[level];
		istart = (Fejer::cc_N-1) / lval;
		istep = istart*2;
		for (j=1, i=istart; j < lval; j += 2, i += istep) {
			(this->*func)(abavg + abdif*Fejer::cc_points[i],funcs);
			for (k=0; k < n_funcs; k++) cc_funcs_mult[i][k] = funcs[k];
			(this->*func)(abavg - abdif*Fejer::cc_points[i],funcs);
			for (k=0; k < n_funcs; k++) {
				cc_funcs_mult[i][k] += funcs[k];
				results[k] += weightptr[j]*cc_funcs_mult[i][k];
			}
		}
		for (j=2, i=istep; j <= lval; j += 2, i += istep) {
			for (k=0; k < n_funcs; k++) results[k] += weightptr[j]*cc_funcs_mult[i][k];
		}
		at_least_one_converged = false;
		if (level > 1) {
			for (k=0; k < n_funcs; k++) {
				if (!func_converged[k]) {
					if (abs(results[k]-results_old[k]) < profile->integral_tolerance*abs(results[k])) {
						func_converged[k] = true;
						at_least_one_converged = true;
					}
				}
			}
			if (at_least_one_converged) { // we won't bother to check if they're all converged, if we haven't had at least one integral converge
				converged = true;
				for (k=0; k < n_funcs; k++) {
					if (!func_converged[k]) converged = false;
				}
			}
			if (converged) {
				break;
			}
		}
	} while (++level < Fejer::cc_nlevels);

	if (level==Fejer::cc_nlevels) {
		for (k=0; k < n_funcs; k++) {
			if ((results[k]*0.0 != 0.0) or (results[k] > 1e100)) warn("integration gave absurdly large or infinite number; suggests numerical problems in evaluating the integrand");
		}
		converged = false;
		//cout << "result=" << result << endl;
		//int npoints = 2*profile->cc_lvals[profile->cc_nlevels-1] + 1;
		// If Gauss-Legendre is set up with at least cc_N points, then switch to this to get a (hopefully) more accurate value
	}

	for (k=0; k < n_funcs; k++) results[k] *= abdif;
	delete[] funcs;
	delete[] func_converged;
	delete[] results_old;
}


