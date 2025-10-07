#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "gauss.h"
#include "errors.h"
using namespace std;

const double GaussianIntegral::EPS = 1e-6;
const double GaussianIntegral::RT4M_PI_INV = 0.7511255444649425;

GaussianIntegral::GaussianIntegral(int N)
{
     numberOfPoints = N;
     weights = new double[N];
     points = new double[N];
}

GaussianIntegral::GaussianIntegral()
{
	numberOfPoints = 0;
	weights = NULL;
	points = NULL;
}

GaussianIntegral::~GaussianIntegral()
{
   if (points != NULL) delete[] points;
	if (weights != NULL) delete[] weights;
}

GaussLegendre::GaussLegendre(int N) : GaussianIntegral(N)
{
	SetGaussLegendre(N);
}

void GaussLegendre::SetGaussLegendre(int N)
{
	numberOfPoints = N;
	if (points != NULL) delete[] points;
	if (weights != NULL) delete[] weights;
	weights = new double[N];
	points = new double[N];

	int max = (numberOfPoints + 1)/2;
	double z1, z, pp, p3, p2, p1;
	 
	for (int i = 0; i < max; i++)
	{
		z = cos(M_PI*(i + 0.75)/(N + 0.5));
		 
		do
		{
			p1 = 1.0;
			p2 = 0.0;
			for (int j = 0; j < numberOfPoints; j++)
			{
				p3 = p2;
				p2 = p1;
				p1 = ((2.0*j + 1.0)*z*p2-j*p3)/(j + 1.0);
			}
			pp = N*(z*p1 - p2)/(z*z - 1.0);
			z1 = z;
			z = z1 - p1/pp;
		}
		while(fabs(z-z1) > EPS);
		 
		points[i] = -z;
		points[numberOfPoints - 1 - i] = z;
		weights[i] = 2.0/((1.0 - z*z)*pp*pp);
		weights[numberOfPoints - 1 - i] = weights[i];
	}
}

void GaussLegendre::SetGaussLegendre(int N, double *points_in, double *weights_in)
{
	numberOfPoints = N;
	if (points != NULL) delete[] points;
	if (weights != NULL) delete[] weights;
	weights = new double[N];
	points = new double[N];
	for (int i=0; i < N; i++) {
		points[i] = points_in[i];
		weights[i] = weights_in[i];
	}
}


double GaussianIntegral::NIntegrate(double (GaussianIntegral::*func)(double), double a, double b)
{
	double result = 0;

	for (int i = 0; i < numberOfPoints; i++)
		result += weights[i]*(this->*func)(((a+b) + (b-a)*points[i])/2.0);

	return (b-a)*result/2.0;
}

double GaussianIntegral::NIntegrateInf(double (GaussianIntegral::*func)(double))
{
	double result = 0;

	for (int i = 0; i < numberOfPoints; i++)
	    result += 2.0/(1.0-points[i])/(1.0-points[i])*weights[i]*(this->*func)((1.0+points[i])/(1.0-points[i]));
	
	return result;
}

GaussLaguerre::GaussLaguerre(const double alpha, int N) : GaussianIntegral(N)
{
     double z1, z, pp, p3, p2, p1, x1, x2;
     
     for (int i = 0; i < numberOfPoints; i++)
     {
          if(i == 0)
               z = (1.0 + alpha)*(3.0 + 0.92*alpha)/(1.0 + 2.4*N + 1.8*alpha);
          else if(i == 1)
               z += (15.0 + 6.25*alpha)/(1.0 +9.0*alpha + 2.5 + 2.5*N);
          else
               z += ((1.0 + 2.55*(i-1))/(1.9*(i-1)) + 1.26*(i-1)*alpha/(1.0 + 3.5*(i-1)))*(z - points[i-2])/(1.0+0.3*alpha);
       
          do
          {
               p1 = 1.0;
               p2 = 0.0;
               for (int j = 0; j < N; j++)
               {
                    p3 = p2;
                    p2 = p1;
                    p1 = ((2.0*j + 1.0 + alpha - z)*p2-(j + alpha)*p3)/(j + 1.0);
               }
               pp = (N*p1-(N+alpha)*p2)/z;
               z1 = z;
               z = z1 - p1/pp;
          }
          while(fabs(z - z1) > EPS);
          points[i] = z;
          weights[i] = -GammaFunction(alpha + N)/GammaFunction(double(N))/N/pp/p2;
     }
}

GaussHermite::GaussHermite(int N) : GaussianIntegral(N)
{
	double z1, z, pp, p3, p2, p1;
	int max = (N + 1)/2;

	for (int i = 0; i < max; i++)
	{
		if (i == 0)
			z = sqrt(double(2*N+1)) - 1.85575*pow(double(2*N+1), -0.16667);
		else if (i == 1)
			z -= 1.14*pow(double(N), 0.426)/z;
		else if (i == 2)
			z = 1.86*z - 0.86*points[0];
		else if (i == 3)
			z = 1.91*z - 0.91*points[1];
		else
			z = 2.0*z - points[i - 2];
          
		do
		{
			p1 = RT4M_PI_INV;
			p2 = 0.0;
			for (int j = 0; j < numberOfPoints; j++)
			{
				p3 = p2;
				p2 = p1;
				p1 = z*sqrt(2.0/(j+1))*p2 - sqrt(double(j)/(j+1))*p3;
			}
			pp = sqrt(double(2*N))*p2;
			z1 = z;
			z = z1 - p1/pp;
		}
		while(fabs(z-z1) > EPS);
          
		points[i] = z;
		points[numberOfPoints - 1 - i] = -z;
		weights[i] = 2.0/(pp*pp);
		weights[numberOfPoints - 1 - i] = weights[i];
	}
}

void GaussHermite::SetGaussHermite(int N)
{
	numberOfPoints = N;
	if (points != NULL) delete[] points;
	if (weights != NULL) delete[] weights;
	weights = new double[N];
	points = new double[N];

	double z1, z, pp, p3, p2, p1;
	int max = (N + 1)/2;

	for (int i = 0; i < max; i++)
	{
		if (i == 0)
			z = sqrt(double(2*N+1)) - 1.85575*pow(double(2*N+1), -0.16667);
		else if (i == 1)
			z -= 1.14*pow(double(N), 0.426)/z;
		else if (i == 2)
			z = 1.86*z - 0.86*points[0];
		else if (i == 3)
			z = 1.91*z - 0.91*points[1];
		else
			z = 2.0*z - points[i - 2];
          
		do
		{
			p1 = RT4M_PI_INV;
			p2 = 0.0;
			for (int j = 0; j < numberOfPoints; j++)
			{
				p3 = p2;
				p2 = p1;
				p1 = z*sqrt(2.0/(j+1))*p2 - sqrt(double(j)/(j+1))*p3;
			}
			pp = sqrt(double(2*N))*p2;
			z1 = z;
			z = z1 - p1/pp;
		}
		while(fabs(z-z1) > EPS);
          
		points[i] = z;
		points[numberOfPoints - 1 - i] = -z;
		weights[i] = 2.0/(pp*pp);
		weights[numberOfPoints - 1 - i] = weights[i];
	}
}

GaussJacobi::GaussJacobi(const double alpha, const double beta, int N) : GaussianIntegral(N)
{
	double alfbet,an,bn,r1,r2,r3;
	double a,b,c,p1,p2,p3,pp,temp,z,z1;
     
     for (int i = 0; i < N; i++)
     {
          if (i == 0) 
          {
               an=alpha/N;
               bn=beta/N;
               r1=(1.0 + alpha)*(2.78/(4.0 + N*N) + 0.768*an/N);
               r2=1.0 + 1.48*an + 0.96*bn + 0.452*an*an + 0.83*an*bn;
               z=1.0 - r1/r2;
          } 
          else if (i == 1) 
          {
               r1=(4.1 + alpha)/((1.0 + alpha)*(1.0 + 0.156*alpha));
               r2=1.0 + 0.06*(N-8.0)*(1.0 + 0.12*alpha)/N;
               r3=1.0 + 0.012*beta*(1.0 + 0.25*fabs(alpha))/N;
               z -= (1.0 - z)*r1*r2*r3;
          } 
          else if (i == 2) 
          {
               r1=(1.67 + 0.28*alpha)/(1.0+0.37*alpha);
               r2=1.0 + 0.22*(N-8.0)/N;
               r3=1.0 + 8.0*beta/((6.28+beta)*N*N);
               z -= (points[0] - z)*r1*r2*r3;
          } 
          else if (i == N-2) 
          {
               r1=(1.0+0.235*beta)/(0.766+0.119*beta);
               r2=1.0/(1.0+0.639*(N - 4.0)/(1.0+0.71*(N - 4.0)));
               r3=1.0/(1.0+20.0*alpha/((7.5+alpha)*N*N));
               z += (z - points[N-4])*r1*r2*r3;
          } 
          else if (i == N-1) 
          {
               r1=(1.0 + 0.37*beta)/(1.67 + 0.28*beta);
               r2=1.0/(1.0+0.22*(N - 8.0)/N);
               r3=1.0/(1.0+8.0*alpha/((6.28+alpha)*N*N));
               z += (z - points[N-3])*r1*r2*r3;
          } 
          else 
          {
               z=3.0*points[i-1]-3.0*points[i-2]+points[i-3];
          }
          alfbet = alpha + beta;
          do
          {
               temp=2.0+alfbet;
               p1=(alpha - beta+temp*z)/2.0;
               p2=1.0;
               for (int j=2; j<=N; j++) 
               {
                    p3=p2;
                    p2=p1;
                    temp=2*j+alfbet;
                    a=2*j*(j+alfbet)*(temp-2.0);
                    b=(temp-1.0)*(alpha*alpha-beta*beta+temp*(temp-2.0)*z);
                    c=2.0*(j-1+alpha)*(j-1+beta)*temp;
                    p1=(b*p2-c*p3)/a;
               }
               pp = (N*(alpha-beta-temp*z)*p1+2.0*(N + alpha)*(N + beta)*p2)/(temp*(1.0 - z*z));
               z1 = z;
               z = z1-p1/pp;
          }
          while(fabs(z-z1) > EPS);
          points[i] = z;
          weights[i] = GammaFunction(alpha + N)*GammaFunction(beta + N)/GammaFunction(N + 1.0)/GammaFunction(N + alfbet + 1.0)*temp*pow(2.0, alfbet)/(pp*p2);
     }
}

GaussChebyshev::GaussChebyshev(int N) : GaussianIntegral(N)
{
     for (int i = 0; i < N; i++)
     {
          points[i] = cos(M_PI*(i + 0.5)/N);
          weights[i] = M_PI/N;
     }
}

double GaussianIntegral::GammaFunction(const double xx)
{
	double x,y,tmp,ser;
	static double cof[6]={76.18009172947146,-86.50532032941677,
		24.01409824083091,-1.231739572450155,
		0.1208650973866179e-2,-0.5395239384953e-5};
	int j;

	y=x=xx;
	tmp=x+5.5;
	tmp -= (x+0.5)*log(tmp);
	ser=1.000000000190015;
	for (j=0;j<=5;j++) ser += cof[j]/++y;
	return exp(-tmp)*(2.5066282746310005*ser/x);
}

ClenshawCurtis::ClenshawCurtis()
{
	cc_points = NULL;
	cc_weights = NULL;
	cc_lvals = NULL;
	cc_funcs = NULL;
	cc_funcs2 = NULL;
	show_convergence_warning = true;
	//SetClenshawCurtis(12,1e-6,true);
}

void ClenshawCurtis::SetClenshawCurtis(const int nlevels_in, const double tol_in, const bool include_endpoints_in, const bool show_warnings)
{
	// Computing weights is very fast for nlevels up to 12 or 13; beyond that, it gets dramatically slower as nlevels is increased
	// If you really need nlevels > 13, the weights should be computed via the fast Fourier transform technique which is much faster.
	// However I have never needed such a high number of levels, and if I did, it's not clear that Romberg integration wouldn't be
	// just as good. But it's something to keep in mind down the road.
	
	cc_tolerance = tol_in;
	cc_tolerance_outer = tol_in;
	include_endpoints = include_endpoints_in;
	show_convergence_warning = show_warnings;
	if (cc_points != NULL) delete[] cc_points;
	if (cc_funcs != NULL) delete[] cc_funcs;
	if (cc_funcs2 != NULL) delete[] cc_funcs2; // useful if doing nested integrals (albeit an ugly solution)
	if (cc_lvals != NULL) delete[] cc_lvals;
	if (cc_weights != NULL) {
		for (int i=0; i < cc_nlevels; i++) delete[] cc_weights[i];
		delete[] cc_weights;
	}
	cc_nlevels = nlevels_in;
	cc_N = pow(2,cc_nlevels-1)+1; // this gives the number of points for the max level (not counting the negative points here)
	cc_lvals = new int [cc_nlevels];
	cc_points = new double[cc_N];
	cc_funcs = new double[cc_N];
	cc_funcs2 = new double[cc_N];
	cc_weights = new double*[cc_nlevels];
	int i,j,k,l = 1;
	for (int i=0; i < cc_nlevels; i++) {
		cc_lvals[i] = l;
		cc_weights[i] = new double[l+1];
		l *= 2;
	}
	l=1;
	int kterm;
	if (include_endpoints) {
		for (i=0; i < cc_nlevels; i++) {
			cc_weights[i][0] = 1.0;
			cc_weights[i][l] = 1.0;
			for (j=1; j < l; j++) {
				cc_weights[i][j] = 1.0;
			}
			for (k=1; k <= l; k++) {
				kterm = 1 - 4*k*k;
				if (k==l) kterm *= 2;
				cc_weights[i][0] += 2.0/kterm;
				if (k % 2 == 0) cc_weights[i][l] += 2.0/kterm;
				else cc_weights[i][l] -= 2.0/kterm;
				for (j=1; j < l; j++) {
					cc_weights[i][j] += (2*cos((j*k*M_PI)/l))/kterm;
				}
			}
			for (j=0; j <= l; j++) {
				cc_weights[i][j] /= l;
			}
			cc_weights[i][0] /= 2; // for the endpoints, to avoid double-counting (as in trapezoid rule)
			l *= 2;
		}
	} else {
		// Here we use Fejer's rule type 2, which excludes endpoints
		for (i=0; i < cc_nlevels; i++) {
			cc_weights[i][0] = 0.0;
			cc_weights[i][l] = 1.0;
			for (j=1; j < l; j++) {
				cc_weights[i][j] = 1.0;
			}
			for (k=1; k < l; k++) {
				kterm = 1 - 4*k*k;
				if (k % 2 == 0) cc_weights[i][l] += 2.0/kterm;
				else cc_weights[i][l] -= 2.0/kterm;
				for (j=1; j < l; j++) {
					cc_weights[i][j] += (2*cos((j*k*M_PI)/l))/kterm;
				}
			}
			double p;
			for (j=1; j <= l; j++) {
				p = 2*l-1;
				if (j % 2 == 0) cc_weights[i][j] -= 1.0/p;
				else cc_weights[i][j] += 1.0/p;
				cc_weights[i][j] /= l;
			}
			l *= 2;
		}
	}

	l = cc_N-1;
	cc_points[0] = 1;
	cc_points[l] = 0;
	for (j=1; j < l; j++) {
		cc_points[j] = cos(j*M_PI/(2*l));
	}
}

ClenshawCurtis::~ClenshawCurtis()
{
	if (cc_points != NULL) delete[] cc_points;
	if (cc_funcs != NULL) delete[] cc_funcs;
	if (cc_funcs2 != NULL) delete[] cc_funcs2;
	if (cc_lvals != NULL) delete[] cc_lvals;
	if (cc_weights != NULL) {
		for (int i=0; i < cc_nlevels; i++) delete[] cc_weights[i];
		delete[] cc_weights;
	}
}

void ClenshawCurtis::set_cc_tolerance(const double tol_in)
{
	cc_tolerance = tol_in;
}

void ClenshawCurtis::set_cc_tolerance_outer(const double tol_in)
{
	cc_tolerance_outer = tol_in;
}

double ClenshawCurtis::AdaptiveQuadCC(double (ClenshawCurtis::*func)(double), double a, double b, bool &converged, bool outer)
{
	double result = 0, result_old;
	double tolerance = (outer) ? cc_tolerance_outer : cc_tolerance;
	int i, level = 0, istep, istart;
	double abavg = (a+b)/2, abdif = (b-a)/2;
	converged = true; // until proven otherwise
	double *funcptr = cc_funcs;
	if (outer) funcptr = cc_funcs2;
	if (!include_endpoints) {
		level = 1;
		funcptr[0] = 0;
		funcptr[cc_N-1] = (this->*func)(abavg);
	}

	int lval, j;
	do {
		result_old = result;
		lval = cc_lvals[level];
		istart = (cc_N-1) / lval;
		istep = istart*2;
		result = 0;
		if (level==0) {
			funcptr[0] = (this->*func)(abavg + abdif*cc_points[0]) + (this->*func)(abavg - abdif*cc_points[0]);
			funcptr[cc_N-1] = (this->*func)(abavg);
			result += cc_weights[0][1]*funcptr[cc_N-1];
		}
		for (j=1, i=istart; j < lval; j += 2, i += istep) {
			funcptr[i] = (this->*func)(abavg + abdif*cc_points[i]) + (this->*func)(abavg - abdif*cc_points[i]);
			result += cc_weights[level][j]*funcptr[i];
		}
		if (include_endpoints) {
			for (j=0, i=0; j <= lval; j += 2, i += istep) {
				result += cc_weights[level][j]*funcptr[i];
			}
		} else {
			for (j=2, i=istep; j <= lval; j += 2, i += istep) {
				result += cc_weights[level][j]*funcptr[i];
			}
		}
		if ((level > 1) and (abs(result-result_old) < tolerance*abs(result))) break;
	} while (++level < cc_nlevels);

	if (level==cc_nlevels) {
		converged = false;
		int npoints = 2*cc_lvals[cc_nlevels-1] + 1;
		if (show_convergence_warning) warn("Clenshaw-Curtis quadrature did not achieve desired tolerance (%g) after NMAX=%i points",tolerance,npoints);
	}
	//else {
	//int npoints = 2*cc_lvals[level] - 1;
	//cout << "Final level: " << (level) << " npoints=" << npoints << endl;
	//}

	return abdif*result;
}

GaussPatterson::GaussPatterson()
{
	//pat_points = NULL;
	pat_weights = NULL;
	pat_orders = NULL;
	//pat_funcs = NULL;
	//pat_funcs2 = NULL;
	//pat_funcs_mult = NULL;
	show_convergence_warning = true;
	//SetGaussPatterson(1e-6,true);
}

void GaussPatterson::SetGaussPatterson(const double tol_in, const bool show_warnings)
{
	pat_tolerance = tol_in;
	pat_tolerance_outer = tol_in;
	show_convergence_warning = show_warnings;
	pat_N = 511;
	//if (pat_points != NULL) delete[] pat_points;
	//if (pat_funcs != NULL) delete[] pat_funcs;
	//if (pat_funcs_mult != NULL) {
		//for (int i=0; i < 6; i++) delete[] pat_funcs_mult[i];
		//delete[] pat_funcs_mult;
	//}
	//if (pat_funcs2 != NULL) delete[] pat_funcs2; // useful if doing nested integrals (albeit an ugly solution)
	if (pat_orders != NULL) delete[] pat_orders;
	if (pat_weights != NULL) {
		for (int i=0; i < 9; i++) delete[] pat_weights[i];
		delete[] pat_weights;
	}
	pat_orders = new int [9];
	//pat_points = new double[511];
	//pat_funcs = new double[511];
	//pat_funcs2 = new double[511];
	//pat_funcs_mult = new double*[6];
	//for (int i=0; i < 6; i++) pat_funcs_mult[i] = new double[511];
	pat_weights = new double*[9];
	pat_weights[0] = new double[1];
	pat_weights[1] = new double[3];
	pat_weights[2] = new double[7];
	pat_weights[3] = new double[15];
	pat_weights[4] = new double[31];
	pat_weights[5] = new double[63];
	pat_weights[6] = new double[127];
	pat_weights[7] = new double[255];
	pat_weights[8] = new double[511];

	pat_orders[0] = 1;
	pat_orders[1] = 3;
	pat_orders[2] = 7;
	pat_orders[3] = 15;
	pat_orders[4] = 31;
	pat_orders[5] = 63;
	pat_orders[6] = 127;
	pat_orders[7] = 255;
	pat_orders[8] = 511;

    pat_points[  0] = -0.999999672956734384381;
    pat_points[  1] = -0.999997596379748464620;
    pat_points[  2] = -0.999992298136257588028;
    pat_points[  3] = -0.999982430354891598580;
    pat_points[  4] = -0.999966730098486276883;
    pat_points[  5] = -0.999943996207054375764;
    pat_points[  6] = -0.999913081144678282800;
    pat_points[  7] = -0.999872888120357611938;
    pat_points[  8] = -0.999822363679787739196;
    pat_points[  9] = -0.999760490924432047330;
    pat_points[ 10] = -0.999686286448317731776;
    pat_points[ 11] = -0.999598799671910683252;
    pat_points[ 12] = -0.999497112467187190535;
    pat_points[ 13] = -0.999380338025023581928;
    pat_points[ 14] = -0.999247618943342473599;
    pat_points[ 15] = -0.999098124967667597662;
    pat_points[ 16] = -0.998931050830810562236;
    pat_points[ 17] = -0.998745614468095114704;
    pat_points[ 18] = -0.998541055697167906027;
    pat_points[ 19] = -0.998316635318407392531;
    pat_points[ 20] = -0.998071634524930323302;
    pat_points[ 21] = -0.997805354495957274562;
    pat_points[ 22] = -0.997517116063472399965;
    pat_points[ 23] = -0.997206259372221959076;
    pat_points[ 24] = -0.996872143485260161299;
    pat_points[ 25] = -0.996514145914890273849;
    pat_points[ 26] = -0.996131662079315037786;
    pat_points[ 27] = -0.995724104698407188509;
    pat_points[ 28] = -0.995290903148810302261;
    pat_points[ 29] = -0.994831502800621000519;
    pat_points[ 30] = -0.994345364356723405931;
    pat_points[ 31] = -0.993831963212755022209;
    pat_points[ 32] = -0.993290788851684966211;
    pat_points[ 33] = -0.992721344282788615328;
    pat_points[ 34] = -0.992123145530863117683;
    pat_points[ 35] = -0.991495721178106132399;
    pat_points[ 36] = -0.990838611958294243677;
    pat_points[ 37] = -0.990151370400770159181;
    pat_points[ 38] = -0.989433560520240838716;
    pat_points[ 39] = -0.988684757547429479939;
    pat_points[ 40] = -0.987904547695124280467;
    pat_points[ 41] = -0.987092527954034067190;
    pat_points[ 42] = -0.986248305913007552681;
    pat_points[ 43] = -0.985371499598520371114;
    pat_points[ 44] = -0.984461737328814534596;
    pat_points[ 45] = -0.983518657578632728762;
    pat_points[ 46] = -0.982541908851080604251;
    pat_points[ 47] = -0.981531149553740106867;
    pat_points[ 48] = -0.980486047876721339416;
    pat_points[ 49] = -0.979406281670862683806;
    pat_points[ 50] = -0.978291538324758539526;
    pat_points[ 51] = -0.977141514639705714156;
    pat_points[ 52] = -0.975955916702011753129;
    pat_points[ 53] = -0.974734459752402667761;
    pat_points[ 54] = -0.973476868052506926773;
    pat_points[ 55] = -0.972182874748581796578;
    pat_points[ 56] = -0.970852221732792443256;
    pat_points[ 57] = -0.969484659502459231771;
    pat_points[ 58] = -0.968079947017759947964;
    pat_points[ 59] = -0.966637851558416567092;
    pat_points[ 60] = -0.965158148579915665979;
    pat_points[ 61] = -0.963640621569812132521;
    pat_points[ 62] = -0.962085061904651475741;
    pat_points[ 63] = -0.960491268708020283423;
    pat_points[ 64] = -0.958859048710200221356;
    pat_points[ 65] = -0.957188216109860962736;
    pat_points[ 66] = -0.955478592438183697574;
    pat_points[ 67] = -0.953730006425761136415;
    pat_points[ 68] = -0.951942293872573589498;
    pat_points[ 69] = -0.950115297521294876558;
    pat_points[ 70] = -0.948248866934137357063;
    pat_points[ 71] = -0.946342858373402905148;
    pat_points[ 72] = -0.944397134685866648591;
    pat_points[ 73] = -0.942411565191083059813;
    pat_points[ 74] = -0.940386025573669721370;
    pat_points[ 75] = -0.938320397779592883655;
    pat_points[ 76] = -0.936214569916450806625;
    pat_points[ 77] = -0.934068436157725787999;
    pat_points[ 78] = -0.931881896650953639345;
    pat_points[ 79] = -0.929654857429740056670;
    pat_points[ 80] = -0.927387230329536696843;
    pat_points[ 81] = -0.925078932907075652364;
    pat_points[ 82] = -0.922729888363349241523;
    pat_points[ 83] = -0.920340025470012420730;
    pat_points[ 84] = -0.917909278499077501636;
    pat_points[ 85] = -0.915437587155765040644;
    pat_points[ 86] = -0.912924896514370590080;
    pat_points[ 87] = -0.910371156957004292498;
    pat_points[ 88] = -0.907776324115058903624;
    pat_points[ 89] = -0.905140358813261595189;
    pat_points[ 90] = -0.902463227016165675048;
    pat_points[ 91] = -0.899744899776940036639;
    pat_points[ 92] = -0.896985353188316590376;
    pat_points[ 93] = -0.894184568335559022859;
    pat_points[ 94] = -0.891342531251319871666;
    pat_points[ 95] = -0.888459232872256998890;
    pat_points[ 96] = -0.885534668997285008926;
    pat_points[ 97] = -0.882568840247341906842;
    pat_points[ 98] = -0.879561752026556262568;
    pat_points[ 99] = -0.876513414484705269742;
    pat_points[100] = -0.873423842480859310192;
    pat_points[101] = -0.870293055548113905851;
    pat_points[102] = -0.867121077859315215614;
    pat_points[103] = -0.863907938193690477146;
    pat_points[104] = -0.860653669904299969802;
    pat_points[105] = -0.857358310886232156525;
    pat_points[106] = -0.854021903545468625813;
    pat_points[107] = -0.850644494768350279758;
    pat_points[108] = -0.847226135891580884381;
    pat_points[109] = -0.843766882672708601038;
    pat_points[110] = -0.840266795261030442350;
    pat_points[111] = -0.836725938168868735503;
    pat_points[112] = -0.833144380243172624728;
    pat_points[113] = -0.829522194637401400178;
    pat_points[114] = -0.825859458783650001088;
    pat_points[115] = -0.822156254364980407373;
    pat_points[116] = -0.818412667287925807395;
    pat_points[117] = -0.814628787655137413436;
    pat_points[118] = -0.810804709738146594361;
    pat_points[119] = -0.806940531950217611856;
    pat_points[120] = -0.803036356819268687782;
    pat_points[121] = -0.799092290960841401800;
    pat_points[122] = -0.795108445051100526780;
    pat_points[123] = -0.791084933799848361435;
    pat_points[124] = -0.787021875923539422170;
    pat_points[125] = -0.782919394118283016385;
    pat_points[126] = -0.778777615032822744702;
    pat_points[127] = -0.774596669241483377036;
    pat_points[128] = -0.770376691217076824278;
    pat_points[129] = -0.766117819303760090717;
    pat_points[130] = -0.761820195689839149173;
    pat_points[131] = -0.757483966380513637926;
    pat_points[132] = -0.753109281170558142523;
    pat_points[133] = -0.748696293616936602823;
    pat_points[134] = -0.744245161011347082309;
    pat_points[135] = -0.739756044352694758677;
    pat_points[136] = -0.735229108319491547663;
    pat_points[137] = -0.730664521242181261329;
    pat_points[138] = -0.726062455075389632685;
    pat_points[139] = -0.721423085370098915485;
    pat_points[140] = -0.716746591245747095767;
    pat_points[141] = -0.712033155362252034587;
    pat_points[142] = -0.707282963891961103412;
    pat_points[143] = -0.702496206491527078610;
    pat_points[144] = -0.697673076273711232906;
    pat_points[145] = -0.692813769779114702895;
    pat_points[146] = -0.687918486947839325756;
    pat_points[147] = -0.682987431091079228087;
    pat_points[148] = -0.678020808862644517838;
    pat_points[149] = -0.673018830230418479199;
    pat_points[150] = -0.667981708447749702165;
    pat_points[151] = -0.662909660024780595461;
    pat_points[152] = -0.657802904699713735422;
    pat_points[153] = -0.652661665410017496101;
    pat_points[154] = -0.647486168263572388782;
    pat_points[155] = -0.642276642509759513774;
    pat_points[156] = -0.637033320510492495071;
    pat_points[157] = -0.631756437711194230414;
    pat_points[158] = -0.626446232611719746542;
    pat_points[159] = -0.621102946737226402941;
    pat_points[160] = -0.615726824608992638014;
    pat_points[161] = -0.610318113715186400156;
    pat_points[162] = -0.604877064481584353319;
    pat_points[163] = -0.599403930242242892974;
    pat_points[164] = -0.593898967210121954393;
    pat_points[165] = -0.588362434447662541434;
    pat_points[166] = -0.582794593837318850840;
    pat_points[167] = -0.577195710052045814844;
    pat_points[168] = -0.571566050525742833992;
    pat_points[169] = -0.565905885423654422623;
    pat_points[170] = -0.560215487612728441818;
    pat_points[171] = -0.554495132631932548866;
    pat_points[172] = -0.548745098662529448608;
    pat_points[173] = -0.542965666498311490492;
    pat_points[174] = -0.537157119515795115982;
    pat_points[175] = -0.531319743644375623972;
    pat_points[176] = -0.525453827336442687395;
    pat_points[177] = -0.519559661537457021993;
    pat_points[178] = -0.513637539655988578507;
    pat_points[179] = -0.507687757533716602155;
    pat_points[180] = -0.501710613415391878251;
    pat_points[181] = -0.495706407918761460170;
    pat_points[182] = -0.489675444004456155436;
    pat_points[183] = -0.483618026945841027562;
    pat_points[184] = -0.477534464298829155284;
    pat_points[185] = -0.471425065871658876934;
    pat_points[186] = -0.465290143694634735858;
    pat_points[187] = -0.459130011989832332874;
    pat_points[188] = -0.452944987140767283784;
    pat_points[189] = -0.446735387662028473742;
    pat_points[190] = -0.440501534168875795783;
    pat_points[191] = -0.434243749346802558002;
    pat_points[192] = -0.427962357921062742583;
    pat_points[193] = -0.421657686626163300056;
    pat_points[194] = -0.415330064175321663764;
    pat_points[195] = -0.408979821229888672409;
    pat_points[196] = -0.402607290368737092671;
    pat_points[197] = -0.396212806057615939183;
    pat_points[198] = -0.389796704618470795479;
    pat_points[199] = -0.383359324198730346916;
    pat_points[200] = -0.376901004740559344802;
    pat_points[201] = -0.370422087950078230138;
    pat_points[202] = -0.363922917266549655269;
    pat_points[203] = -0.357403837831532152376;
    pat_points[204] = -0.350865196458001209011;
    pat_points[205] = -0.344307341599438022777;
    pat_points[206] = -0.337730623318886219621;
    pat_points[207] = -0.331135393257976833093;
    pat_points[208] = -0.324522004605921855207;
    pat_points[209] = -0.317890812068476683182;
    pat_points[210] = -0.311242171836871800300;
    pat_points[211] = -0.304576441556714043335;
    pat_points[212] = -0.297893980296857823437;
    pat_points[213] = -0.291195148518246681964;
    pat_points[214] = -0.284480308042725577496;
    pat_points[215] = -0.277749822021824315065;
    pat_points[216] = -0.271004054905512543536;
    pat_points[217] = -0.264243372410926761945;
    pat_points[218] = -0.257468141491069790481;
    pat_points[219] = -0.250678730303483176613;
    pat_points[220] = -0.243875508178893021593;
    pat_points[221] = -0.237058845589829727213;
    pat_points[222] = -0.230229114119222177156;
    pat_points[223] = -0.223386686428966881628;
    pat_points[224] = -0.216531936228472628081;
    pat_points[225] = -0.209665238243181194766;
    pat_points[226] = -0.202786968183064697557;
    pat_points[227] = -0.195897502711100153915;
    pat_points[228] = -0.188997219411721861059;
    pat_points[229] = -0.182086496759252198246;
    pat_points[230] = -0.175165714086311475707;
    pat_points[231] = -0.168235251552207464982;
    pat_points[232] = -0.161295490111305257361;
    pat_points[233] = -0.154346811481378108692;
    pat_points[234] = -0.147389598111939940054;
    pat_points[235] = -0.140424233152560174594;
    pat_points[236] = -0.133451100421161601344;
    pat_points[237] = -0.126470584372301966851;
    pat_points[238] = -0.119483070065440005133;
    pat_points[239] = -0.112488943133186625746;
    pat_points[240] = -0.105488589749541988533;
    pat_points[241] = -0.984823965981192020903E-01;
    pat_points[242] = -0.914707508403553909095E-01;
    pat_points[243] = -0.844540400837108837102E-01;
    pat_points[244] = -0.774326523498572825675E-01;
    pat_points[245] = -0.704069760428551790633E-01;
    pat_points[246] = -0.633773999173222898797E-01;
    pat_points[247] = -0.563443130465927899720E-01;
    pat_points[248] = -0.493081047908686267156E-01;
    pat_points[249] = -0.422691647653636032124E-01;
    pat_points[250] = -0.352278828084410232603E-01;
    pat_points[251] = -0.281846489497456943394E-01;
    pat_points[252] = -0.211398533783310883350E-01;
    pat_points[253] = -0.140938864107824626142E-01;
    pat_points[254] = -0.704713845933674648514E-02;
    pat_points[255] = +0.000000000000000000000;
    pat_points[256] = +0.704713845933674648514E-02;
    pat_points[257] = +0.140938864107824626142E-01;
    pat_points[258] = +0.211398533783310883350E-01;
    pat_points[259] = +0.281846489497456943394E-01;
    pat_points[260] = +0.352278828084410232603E-01;
    pat_points[261] = +0.422691647653636032124E-01;
    pat_points[262] = +0.493081047908686267156E-01;
    pat_points[263] = +0.563443130465927899720E-01;
    pat_points[264] = +0.633773999173222898797E-01;
    pat_points[265] = +0.704069760428551790633E-01;
    pat_points[266] = +0.774326523498572825675E-01;
    pat_points[267] = +0.844540400837108837102E-01;
    pat_points[268] = +0.914707508403553909095E-01;
    pat_points[269] = +0.984823965981192020903E-01;
    pat_points[270] = +0.105488589749541988533;
    pat_points[271] = +0.112488943133186625746;
    pat_points[272] = +0.119483070065440005133;
    pat_points[273] = +0.126470584372301966851;
    pat_points[274] = +0.133451100421161601344;
    pat_points[275] = +0.140424233152560174594;
    pat_points[276] = +0.147389598111939940054;
    pat_points[277] = +0.154346811481378108692;
    pat_points[278] = +0.161295490111305257361;
    pat_points[279] = +0.168235251552207464982;
    pat_points[280] = +0.175165714086311475707;
    pat_points[281] = +0.182086496759252198246;
    pat_points[282] = +0.188997219411721861059;
    pat_points[283] = +0.195897502711100153915;
    pat_points[284] = +0.202786968183064697557;
    pat_points[285] = +0.209665238243181194766;
    pat_points[286] = +0.216531936228472628081;
    pat_points[287] = +0.223386686428966881628;
    pat_points[288] = +0.230229114119222177156;
    pat_points[289] = +0.237058845589829727213;
    pat_points[290] = +0.243875508178893021593;
    pat_points[291] = +0.250678730303483176613;
    pat_points[292] = +0.257468141491069790481;
    pat_points[293] = +0.264243372410926761945;
    pat_points[294] = +0.271004054905512543536;
    pat_points[295] = +0.277749822021824315065;
    pat_points[296] = +0.284480308042725577496;
    pat_points[297] = +0.291195148518246681964;
    pat_points[298] = +0.297893980296857823437;
    pat_points[299] = +0.304576441556714043335;
    pat_points[300] = +0.311242171836871800300;
    pat_points[301] = +0.317890812068476683182;
    pat_points[302] = +0.324522004605921855207;
    pat_points[303] = +0.331135393257976833093;
    pat_points[304] = +0.337730623318886219621;
    pat_points[305] = +0.344307341599438022777;
    pat_points[306] = +0.350865196458001209011;
    pat_points[307] = +0.357403837831532152376;
    pat_points[308] = +0.363922917266549655269;
    pat_points[309] = +0.370422087950078230138;
    pat_points[310] = +0.376901004740559344802;
    pat_points[311] = +0.383359324198730346916;
    pat_points[312] = +0.389796704618470795479;
    pat_points[313] = +0.396212806057615939183;
    pat_points[314] = +0.402607290368737092671;
    pat_points[315] = +0.408979821229888672409;
    pat_points[316] = +0.415330064175321663764;
    pat_points[317] = +0.421657686626163300056;
    pat_points[318] = +0.427962357921062742583;
    pat_points[319] = +0.434243749346802558002;
    pat_points[320] = +0.440501534168875795783;
    pat_points[321] = +0.446735387662028473742;
    pat_points[322] = +0.452944987140767283784;
    pat_points[323] = +0.459130011989832332874;
    pat_points[324] = +0.465290143694634735858;
    pat_points[325] = +0.471425065871658876934;
    pat_points[326] = +0.477534464298829155284;
    pat_points[327] = +0.483618026945841027562;
    pat_points[328] = +0.489675444004456155436;
    pat_points[329] = +0.495706407918761460170;
    pat_points[330] = +0.501710613415391878251;
    pat_points[331] = +0.507687757533716602155;
    pat_points[332] = +0.513637539655988578507;
    pat_points[333] = +0.519559661537457021993;
    pat_points[334] = +0.525453827336442687395;
    pat_points[335] = +0.531319743644375623972;
    pat_points[336] = +0.537157119515795115982;
    pat_points[337] = +0.542965666498311490492;
    pat_points[338] = +0.548745098662529448608;
    pat_points[339] = +0.554495132631932548866;
    pat_points[340] = +0.560215487612728441818;
    pat_points[341] = +0.565905885423654422623;
    pat_points[342] = +0.571566050525742833992;
    pat_points[343] = +0.577195710052045814844;
    pat_points[344] = +0.582794593837318850840;
    pat_points[345] = +0.588362434447662541434;
    pat_points[346] = +0.593898967210121954393;
    pat_points[347] = +0.599403930242242892974;
    pat_points[348] = +0.604877064481584353319;
    pat_points[349] = +0.610318113715186400156;
    pat_points[350] = +0.615726824608992638014;
    pat_points[351] = +0.621102946737226402941;
    pat_points[352] = +0.626446232611719746542;
    pat_points[353] = +0.631756437711194230414;
    pat_points[354] = +0.637033320510492495071;
    pat_points[355] = +0.642276642509759513774;
    pat_points[356] = +0.647486168263572388782;
    pat_points[357] = +0.652661665410017496101;
    pat_points[358] = +0.657802904699713735422;
    pat_points[359] = +0.662909660024780595461;
    pat_points[360] = +0.667981708447749702165;
    pat_points[361] = +0.673018830230418479199;
    pat_points[362] = +0.678020808862644517838;
    pat_points[363] = +0.682987431091079228087;
    pat_points[364] = +0.687918486947839325756;
    pat_points[365] = +0.692813769779114702895;
    pat_points[366] = +0.697673076273711232906;
    pat_points[367] = +0.702496206491527078610;
    pat_points[368] = +0.707282963891961103412;
    pat_points[369] = +0.712033155362252034587;
    pat_points[370] = +0.716746591245747095767;
    pat_points[371] = +0.721423085370098915485;
    pat_points[372] = +0.726062455075389632685;
    pat_points[373] = +0.730664521242181261329;
    pat_points[374] = +0.735229108319491547663;
    pat_points[375] = +0.739756044352694758677;
    pat_points[376] = +0.744245161011347082309;
    pat_points[377] = +0.748696293616936602823;
    pat_points[378] = +0.753109281170558142523;
    pat_points[379] = +0.757483966380513637926;
    pat_points[380] = +0.761820195689839149173;
    pat_points[381] = +0.766117819303760090717;
    pat_points[382] = +0.770376691217076824278;
    pat_points[383] = +0.774596669241483377036;
    pat_points[384] = +0.778777615032822744702;
    pat_points[385] = +0.782919394118283016385;
    pat_points[386] = +0.787021875923539422170;
    pat_points[387] = +0.791084933799848361435;
    pat_points[388] = +0.795108445051100526780;
    pat_points[389] = +0.799092290960841401800;
    pat_points[390] = +0.803036356819268687782;
    pat_points[391] = +0.806940531950217611856;
    pat_points[392] = +0.810804709738146594361;
    pat_points[393] = +0.814628787655137413436;
    pat_points[394] = +0.818412667287925807395;
    pat_points[395] = +0.822156254364980407373;
    pat_points[396] = +0.825859458783650001088;
    pat_points[397] = +0.829522194637401400178;
    pat_points[398] = +0.833144380243172624728;
    pat_points[399] = +0.836725938168868735503;
    pat_points[400] = +0.840266795261030442350;
    pat_points[401] = +0.843766882672708601038;
    pat_points[402] = +0.847226135891580884381;
    pat_points[403] = +0.850644494768350279758;
    pat_points[404] = +0.854021903545468625813;
    pat_points[405] = +0.857358310886232156525;
    pat_points[406] = +0.860653669904299969802;
    pat_points[407] = +0.863907938193690477146;
    pat_points[408] = +0.867121077859315215614;
    pat_points[409] = +0.870293055548113905851;
    pat_points[410] = +0.873423842480859310192;
    pat_points[411] = +0.876513414484705269742;
    pat_points[412] = +0.879561752026556262568;
    pat_points[413] = +0.882568840247341906842;
    pat_points[414] = +0.885534668997285008926;
    pat_points[415] = +0.888459232872256998890;
    pat_points[416] = +0.891342531251319871666;
    pat_points[417] = +0.894184568335559022859;
    pat_points[418] = +0.896985353188316590376;
    pat_points[419] = +0.899744899776940036639;
    pat_points[420] = +0.902463227016165675048;
    pat_points[421] = +0.905140358813261595189;
    pat_points[422] = +0.907776324115058903624;
    pat_points[423] = +0.910371156957004292498;
    pat_points[424] = +0.912924896514370590080;
    pat_points[425] = +0.915437587155765040644;
    pat_points[426] = +0.917909278499077501636;
    pat_points[427] = +0.920340025470012420730;
    pat_points[428] = +0.922729888363349241523;
    pat_points[429] = +0.925078932907075652364;
    pat_points[430] = +0.927387230329536696843;
    pat_points[431] = +0.929654857429740056670;
    pat_points[432] = +0.931881896650953639345;
    pat_points[433] = +0.934068436157725787999;
    pat_points[434] = +0.936214569916450806625;
    pat_points[435] = +0.938320397779592883655;
    pat_points[436] = +0.940386025573669721370;
    pat_points[437] = +0.942411565191083059813;
    pat_points[438] = +0.944397134685866648591;
    pat_points[439] = +0.946342858373402905148;
    pat_points[440] = +0.948248866934137357063;
    pat_points[441] = +0.950115297521294876558;
    pat_points[442] = +0.951942293872573589498;
    pat_points[443] = +0.953730006425761136415;
    pat_points[444] = +0.955478592438183697574;
    pat_points[445] = +0.957188216109860962736;
    pat_points[446] = +0.958859048710200221356;
    pat_points[447] = +0.960491268708020283423;
    pat_points[448] = +0.962085061904651475741;
    pat_points[449] = +0.963640621569812132521;
    pat_points[450] = +0.965158148579915665979;
    pat_points[451] = +0.966637851558416567092;
    pat_points[452] = +0.968079947017759947964;
    pat_points[453] = +0.969484659502459231771;
    pat_points[454] = +0.970852221732792443256;
    pat_points[455] = +0.972182874748581796578;
    pat_points[456] = +0.973476868052506926773;
    pat_points[457] = +0.974734459752402667761;
    pat_points[458] = +0.975955916702011753129;
    pat_points[459] = +0.977141514639705714156;
    pat_points[460] = +0.978291538324758539526;
    pat_points[461] = +0.979406281670862683806;
    pat_points[462] = +0.980486047876721339416;
    pat_points[463] = +0.981531149553740106867;
    pat_points[464] = +0.982541908851080604251;
    pat_points[465] = +0.983518657578632728762;
    pat_points[466] = +0.984461737328814534596;
    pat_points[467] = +0.985371499598520371114;
    pat_points[468] = +0.986248305913007552681;
    pat_points[469] = +0.987092527954034067190;
    pat_points[470] = +0.987904547695124280467;
    pat_points[471] = +0.988684757547429479939;
    pat_points[472] = +0.989433560520240838716;
    pat_points[473] = +0.990151370400770159181;
    pat_points[474] = +0.990838611958294243677;
    pat_points[475] = +0.991495721178106132399;
    pat_points[476] = +0.992123145530863117683;
    pat_points[477] = +0.992721344282788615328;
    pat_points[478] = +0.993290788851684966211;
    pat_points[479] = +0.993831963212755022209;
    pat_points[480] = +0.994345364356723405931;
    pat_points[481] = +0.994831502800621000519;
    pat_points[482] = +0.995290903148810302261;
    pat_points[483] = +0.995724104698407188509;
    pat_points[484] = +0.996131662079315037786;
    pat_points[485] = +0.996514145914890273849;
    pat_points[486] = +0.996872143485260161299;
    pat_points[487] = +0.997206259372221959076;
    pat_points[488] = +0.997517116063472399965;
    pat_points[489] = +0.997805354495957274562;
    pat_points[490] = +0.998071634524930323302;
    pat_points[491] = +0.998316635318407392531;
    pat_points[492] = +0.998541055697167906027;
    pat_points[493] = +0.998745614468095114704;
    pat_points[494] = +0.998931050830810562236;
    pat_points[495] = +0.999098124967667597662;
    pat_points[496] = +0.999247618943342473599;
    pat_points[497] = +0.999380338025023581928;
    pat_points[498] = +0.999497112467187190535;
    pat_points[499] = +0.999598799671910683252;
    pat_points[500] = +0.999686286448317731776;
    pat_points[501] = +0.999760490924432047330;
    pat_points[502] = +0.999822363679787739196;
    pat_points[503] = +0.999872888120357611938;
    pat_points[504] = +0.999913081144678282800;
    pat_points[505] = +0.999943996207054375764;
    pat_points[506] = +0.999966730098486276883;
    pat_points[507] = +0.999982430354891598580;
    pat_points[508] = +0.999992298136257588028;
    pat_points[509] = +0.999997596379748464620;
    pat_points[510] = +0.999999672956734384381;

    pat_weights[0][0] = 2.0;

    pat_weights[1][0] = 0.555555555555555555556;
    pat_weights[1][1] = 0.888888888888888888889;
    pat_weights[1][2] = 0.555555555555555555556;

    pat_weights[2][0] = 0.104656226026467265194;
    pat_weights[2][1] = 0.268488089868333440729;
    pat_weights[2][2] = 0.401397414775962222905;
    pat_weights[2][3] = 0.450916538658474142345;
    pat_weights[2][4] = 0.401397414775962222905;
    pat_weights[2][5] = 0.268488089868333440729;
    pat_weights[2][6] = 0.104656226026467265194;

    pat_weights[3][ 0] = 0.0170017196299402603390;
    pat_weights[3][ 1] = 0.0516032829970797396969;
    pat_weights[3][ 2] = 0.0929271953151245376859;
    pat_weights[3][ 3] = 0.134415255243784220360;
    pat_weights[3][ 4] = 0.171511909136391380787;
    pat_weights[3][ 5] = 0.200628529376989021034;
    pat_weights[3][ 6] = 0.219156858401587496404;
    pat_weights[3][ 7] = 0.225510499798206687386;
    pat_weights[3][ 8] = 0.219156858401587496404;
    pat_weights[3][ 9] = 0.200628529376989021034;
    pat_weights[3][10] = 0.171511909136391380787;
    pat_weights[3][11] = 0.134415255243784220360;
    pat_weights[3][12] = 0.0929271953151245376859;
    pat_weights[3][13] = 0.0516032829970797396969;
    pat_weights[3][14] = 0.0170017196299402603390;

    pat_weights[4][ 0] = 0.00254478079156187441540;
    pat_weights[4][ 1] = 0.00843456573932110624631;
    pat_weights[4][ 2] = 0.0164460498543878109338;
    pat_weights[4][ 3] = 0.0258075980961766535646;
    pat_weights[4][ 4] = 0.0359571033071293220968;
    pat_weights[4][ 5] = 0.0464628932617579865414;
    pat_weights[4][ 6] = 0.0569795094941233574122;
    pat_weights[4][ 7] = 0.0672077542959907035404;
    pat_weights[4][ 8] = 0.0768796204990035310427;
    pat_weights[4][ 9] = 0.0857559200499903511542;
    pat_weights[4][10] = 0.0936271099812644736167;
    pat_weights[4][11] = 0.100314278611795578771;
    pat_weights[4][12] = 0.105669893580234809744;
    pat_weights[4][13] = 0.109578421055924638237;
    pat_weights[4][14] = 0.111956873020953456880;
    pat_weights[4][15] = 0.112755256720768691607;
    pat_weights[4][16] = 0.111956873020953456880;
    pat_weights[4][17] = 0.109578421055924638237;
    pat_weights[4][18] = 0.105669893580234809744;
    pat_weights[4][19] = 0.100314278611795578771;
    pat_weights[4][20] = 0.0936271099812644736167;
    pat_weights[4][21] = 0.0857559200499903511542;
    pat_weights[4][22] = 0.0768796204990035310427;
    pat_weights[4][23] = 0.0672077542959907035404;
    pat_weights[4][24] = 0.0569795094941233574122;
    pat_weights[4][25] = 0.0464628932617579865414;
    pat_weights[4][26] = 0.0359571033071293220968;
    pat_weights[4][27] = 0.0258075980961766535646;
    pat_weights[4][28] = 0.0164460498543878109338;
    pat_weights[4][29] = 0.00843456573932110624631;
    pat_weights[4][30] = 0.00254478079156187441540;

    pat_weights[5][ 0] = 0.000363221481845530659694;
    pat_weights[5][ 1] = 0.00126515655623006801137;
    pat_weights[5][ 2] = 0.00257904979468568827243;
    pat_weights[5][ 3] = 0.00421763044155885483908;
    pat_weights[5][ 4] = 0.00611550682211724633968;
    pat_weights[5][ 5] = 0.00822300795723592966926;
    pat_weights[5][ 6] = 0.0104982469096213218983;
    pat_weights[5][ 7] = 0.0129038001003512656260;
    pat_weights[5][ 8] = 0.0154067504665594978021;
    pat_weights[5][ 9] = 0.0179785515681282703329;
    pat_weights[5][10] = 0.0205942339159127111492;
    pat_weights[5][11] = 0.0232314466399102694433;
    pat_weights[5][12] = 0.0258696793272147469108;
    pat_weights[5][13] = 0.0284897547458335486125;
    pat_weights[5][14] = 0.0310735511116879648799;
    pat_weights[5][15] = 0.0336038771482077305417;
    pat_weights[5][16] = 0.0360644327807825726401;
    pat_weights[5][17] = 0.0384398102494555320386;
    pat_weights[5][18] = 0.0407155101169443189339;
    pat_weights[5][19] = 0.0428779600250077344929;
    pat_weights[5][20] = 0.0449145316536321974143;
    pat_weights[5][21] = 0.0468135549906280124026;
    pat_weights[5][22] = 0.0485643304066731987159;
    pat_weights[5][23] = 0.0501571393058995374137;
    pat_weights[5][24] = 0.0515832539520484587768;
    pat_weights[5][25] = 0.0528349467901165198621;
    pat_weights[5][26] = 0.0539054993352660639269;
    pat_weights[5][27] = 0.0547892105279628650322;
    pat_weights[5][28] = 0.0554814043565593639878;
    pat_weights[5][29] = 0.0559784365104763194076;
    pat_weights[5][30] = 0.0562776998312543012726;
    pat_weights[5][31] = 0.0563776283603847173877;
    pat_weights[5][32] = 0.0562776998312543012726;
    pat_weights[5][33] = 0.0559784365104763194076;
    pat_weights[5][34] = 0.0554814043565593639878;
    pat_weights[5][35] = 0.0547892105279628650322;
    pat_weights[5][36] = 0.0539054993352660639269;
    pat_weights[5][37] = 0.0528349467901165198621;
    pat_weights[5][38] = 0.0515832539520484587768;
    pat_weights[5][39] = 0.0501571393058995374137;
    pat_weights[5][40] = 0.0485643304066731987159;
    pat_weights[5][41] = 0.0468135549906280124026;
    pat_weights[5][42] = 0.0449145316536321974143;
    pat_weights[5][43] = 0.0428779600250077344929;
    pat_weights[5][44] = 0.0407155101169443189339;
    pat_weights[5][45] = 0.0384398102494555320386;
    pat_weights[5][46] = 0.0360644327807825726401;
    pat_weights[5][47] = 0.0336038771482077305417;
    pat_weights[5][48] = 0.0310735511116879648799;
    pat_weights[5][49] = 0.0284897547458335486125;
    pat_weights[5][50] = 0.0258696793272147469108;
    pat_weights[5][51] = 0.0232314466399102694433;
    pat_weights[5][52] = 0.0205942339159127111492;
    pat_weights[5][53] = 0.0179785515681282703329;
    pat_weights[5][54] = 0.0154067504665594978021;
    pat_weights[5][55] = 0.0129038001003512656260;
    pat_weights[5][56] = 0.0104982469096213218983;
    pat_weights[5][57] = 0.00822300795723592966926;
    pat_weights[5][58] = 0.00611550682211724633968;
    pat_weights[5][59] = 0.00421763044155885483908;
    pat_weights[5][60] = 0.00257904979468568827243;
    pat_weights[5][61] = 0.00126515655623006801137;
    pat_weights[5][62] = 0.000363221481845530659694;

    pat_weights[6][  0] = 0.0000505360952078625176247;
    pat_weights[6][  1] = 0.000180739564445388357820;
    pat_weights[6][  2] = 0.000377746646326984660274;
    pat_weights[6][  3] = 0.000632607319362633544219;
    pat_weights[6][  4] = 0.000938369848542381500794;
    pat_weights[6][  5] = 0.00128952408261041739210;
    pat_weights[6][  6] = 0.00168114286542146990631;
    pat_weights[6][  7] = 0.00210881524572663287933;
    pat_weights[6][  8] = 0.00256876494379402037313;
    pat_weights[6][  9] = 0.00305775341017553113613;
    pat_weights[6][ 10] = 0.00357289278351729964938;
    pat_weights[6][ 11] = 0.00411150397865469304717;
    pat_weights[6][ 12] = 0.00467105037211432174741;
    pat_weights[6][ 13] = 0.00524912345480885912513;
    pat_weights[6][ 14] = 0.00584344987583563950756;
    pat_weights[6][ 15] = 0.00645190005017573692280;
    pat_weights[6][ 16] = 0.00707248999543355546805;
    pat_weights[6][ 17] = 0.00770337523327974184817;
    pat_weights[6][ 18] = 0.00834283875396815770558;
    pat_weights[6][ 19] = 0.00898927578406413572328;
    pat_weights[6][ 20] = 0.00964117772970253669530;
    pat_weights[6][ 21] = 0.0102971169579563555237;
    pat_weights[6][ 22] = 0.0109557333878379016480;
    pat_weights[6][ 23] = 0.0116157233199551347270;
    pat_weights[6][ 24] = 0.0122758305600827700870;
    pat_weights[6][ 25] = 0.0129348396636073734547;
    pat_weights[6][ 26] = 0.0135915710097655467896;
    pat_weights[6][ 27] = 0.0142448773729167743063;
    pat_weights[6][ 28] = 0.0148936416648151820348;
    pat_weights[6][ 29] = 0.0155367755558439824399;
    pat_weights[6][ 30] = 0.0161732187295777199419;
    pat_weights[6][ 31] = 0.0168019385741038652709;
    pat_weights[6][ 32] = 0.0174219301594641737472;
    pat_weights[6][ 33] = 0.0180322163903912863201;
    pat_weights[6][ 34] = 0.0186318482561387901863;
    pat_weights[6][ 35] = 0.0192199051247277660193;
    pat_weights[6][ 36] = 0.0197954950480974994880;
    pat_weights[6][ 37] = 0.0203577550584721594669;
    pat_weights[6][ 38] = 0.0209058514458120238522;
    pat_weights[6][ 39] = 0.0214389800125038672465;
    pat_weights[6][ 40] = 0.0219563663053178249393;
    pat_weights[6][ 41] = 0.0224572658268160987071;
    pat_weights[6][ 42] = 0.0229409642293877487608;
    pat_weights[6][ 43] = 0.0234067774953140062013;
    pat_weights[6][ 44] = 0.0238540521060385400804;
    pat_weights[6][ 45] = 0.0242821652033365993580;
    pat_weights[6][ 46] = 0.0246905247444876769091;
    pat_weights[6][ 47] = 0.0250785696529497687068;
    pat_weights[6][ 48] = 0.0254457699654647658126;
    pat_weights[6][ 49] = 0.0257916269760242293884;
    pat_weights[6][ 50] = 0.0261156733767060976805;
    pat_weights[6][ 51] = 0.0264174733950582599310;
    pat_weights[6][ 52] = 0.0266966229274503599062;
    pat_weights[6][ 53] = 0.0269527496676330319634;
    pat_weights[6][ 54] = 0.0271855132296247918192;
    pat_weights[6][ 55] = 0.0273946052639814325161;
    pat_weights[6][ 56] = 0.0275797495664818730349;
    pat_weights[6][ 57] = 0.0277407021782796819939;
    pat_weights[6][ 58] = 0.0278772514766137016085;
    pat_weights[6][ 59] = 0.0279892182552381597038;
    pat_weights[6][ 60] = 0.0280764557938172466068;
    pat_weights[6][ 61] = 0.0281388499156271506363;
    pat_weights[6][ 62] = 0.0281763190330166021307;
    pat_weights[6][ 63] = 0.0281888141801923586938;
    pat_weights[6][ 64] = 0.0281763190330166021307;
    pat_weights[6][ 65] = 0.0281388499156271506363;
    pat_weights[6][ 66] = 0.0280764557938172466068;
    pat_weights[6][ 67] = 0.0279892182552381597038;
    pat_weights[6][ 68] = 0.0278772514766137016085;
    pat_weights[6][ 69] = 0.0277407021782796819939;
    pat_weights[6][ 70] = 0.0275797495664818730349;
    pat_weights[6][ 71] = 0.0273946052639814325161;
    pat_weights[6][ 72] = 0.0271855132296247918192;
    pat_weights[6][ 73] = 0.0269527496676330319634;
    pat_weights[6][ 74] = 0.0266966229274503599062;
    pat_weights[6][ 75] = 0.0264174733950582599310;
    pat_weights[6][ 76] = 0.0261156733767060976805;
    pat_weights[6][ 77] = 0.0257916269760242293884;
    pat_weights[6][ 78] = 0.0254457699654647658126;
    pat_weights[6][ 79] = 0.0250785696529497687068;
    pat_weights[6][ 80] = 0.0246905247444876769091;
    pat_weights[6][ 81] = 0.0242821652033365993580;
    pat_weights[6][ 82] = 0.0238540521060385400804;
    pat_weights[6][ 83] = 0.0234067774953140062013;
    pat_weights[6][ 84] = 0.0229409642293877487608;
    pat_weights[6][ 85] = 0.0224572658268160987071;
    pat_weights[6][ 86] = 0.0219563663053178249393;
    pat_weights[6][ 87] = 0.0214389800125038672465;
    pat_weights[6][ 88] = 0.0209058514458120238522;
    pat_weights[6][ 89] = 0.0203577550584721594669;
    pat_weights[6][ 90] = 0.0197954950480974994880;
    pat_weights[6][ 91] = 0.0192199051247277660193;
    pat_weights[6][ 92] = 0.0186318482561387901863;
    pat_weights[6][ 93] = 0.0180322163903912863201;
    pat_weights[6][ 94] = 0.0174219301594641737472;
    pat_weights[6][ 95] = 0.0168019385741038652709;
    pat_weights[6][ 96] = 0.0161732187295777199419;
    pat_weights[6][ 97] = 0.0155367755558439824399;
    pat_weights[6][ 98] = 0.0148936416648151820348;
    pat_weights[6][ 99] = 0.0142448773729167743063;
    pat_weights[6][100] = 0.0135915710097655467896;
    pat_weights[6][101] = 0.0129348396636073734547;
    pat_weights[6][102] = 0.0122758305600827700870;
    pat_weights[6][103] = 0.0116157233199551347270;
    pat_weights[6][104] = 0.0109557333878379016480;
    pat_weights[6][105] = 0.0102971169579563555237;
    pat_weights[6][106] = 0.00964117772970253669530;
    pat_weights[6][107] = 0.00898927578406413572328;
    pat_weights[6][108] = 0.00834283875396815770558;
    pat_weights[6][109] = 0.00770337523327974184817;
    pat_weights[6][110] = 0.00707248999543355546805;
    pat_weights[6][111] = 0.00645190005017573692280;
    pat_weights[6][112] = 0.00584344987583563950756;
    pat_weights[6][113] = 0.00524912345480885912513;
    pat_weights[6][114] = 0.00467105037211432174741;
    pat_weights[6][115] = 0.00411150397865469304717;
    pat_weights[6][116] = 0.00357289278351729964938;
    pat_weights[6][117] = 0.00305775341017553113613;
    pat_weights[6][118] = 0.00256876494379402037313;
    pat_weights[6][119] = 0.00210881524572663287933;
    pat_weights[6][120] = 0.00168114286542146990631;
    pat_weights[6][121] = 0.00128952408261041739210;
    pat_weights[6][122] = 0.000938369848542381500794;
    pat_weights[6][123] = 0.000632607319362633544219;
    pat_weights[6][124] = 0.000377746646326984660274;
    pat_weights[6][125] = 0.000180739564445388357820;
    pat_weights[6][126] = 0.0000505360952078625176247;

    pat_weights[7][  0] = 0.69379364324108267170E-05;
    pat_weights[7][  1] = 0.25157870384280661489E-04;
    pat_weights[7][  2] = 0.53275293669780613125E-04;
    pat_weights[7][  3] = 0.90372734658751149261E-04;
    pat_weights[7][  4] = 0.13575491094922871973E-03;
    pat_weights[7][  5] = 0.18887326450650491366E-03;
    pat_weights[7][  6] = 0.24921240048299729402E-03;
    pat_weights[7][  7] = 0.31630366082226447689E-03;
    pat_weights[7][  8] = 0.38974528447328229322E-03;
    pat_weights[7][  9] = 0.46918492424785040975E-03;
    pat_weights[7][ 10] = 0.55429531493037471492E-03;
    pat_weights[7][ 11] = 0.64476204130572477933E-03;
    pat_weights[7][ 12] = 0.74028280424450333046E-03;
    pat_weights[7][ 13] = 0.84057143271072246365E-03;
    pat_weights[7][ 14] = 0.94536151685852538246E-03;
    pat_weights[7][ 15] = 0.10544076228633167722E-02;
    pat_weights[7][ 16] = 0.11674841174299594077E-02;
    pat_weights[7][ 17] = 0.12843824718970101768E-02;
    pat_weights[7][ 18] = 0.14049079956551446427E-02;
    pat_weights[7][ 19] = 0.15288767050877655684E-02;
    pat_weights[7][ 20] = 0.16561127281544526052E-02;
    pat_weights[7][ 21] = 0.17864463917586498247E-02;
    pat_weights[7][ 22] = 0.19197129710138724125E-02;
    pat_weights[7][ 23] = 0.20557519893273465236E-02;
    pat_weights[7][ 24] = 0.21944069253638388388E-02;
    pat_weights[7][ 25] = 0.23355251860571608737E-02;
    pat_weights[7][ 26] = 0.24789582266575679307E-02;
    pat_weights[7][ 27] = 0.26245617274044295626E-02;
    pat_weights[7][ 28] = 0.27721957645934509940E-02;
    pat_weights[7][ 29] = 0.29217249379178197538E-02;
    pat_weights[7][ 30] = 0.30730184347025783234E-02;
    pat_weights[7][ 31] = 0.32259500250878684614E-02;
    pat_weights[7][ 32] = 0.33803979910869203823E-02;
    pat_weights[7][ 33] = 0.35362449977167777340E-02;
    pat_weights[7][ 34] = 0.36933779170256508183E-02;
    pat_weights[7][ 35] = 0.38516876166398709241E-02;
    pat_weights[7][ 36] = 0.40110687240750233989E-02;
    pat_weights[7][ 37] = 0.41714193769840788528E-02;
    pat_weights[7][ 38] = 0.43326409680929828545E-02;
    pat_weights[7][ 39] = 0.44946378920320678616E-02;
    pat_weights[7][ 40] = 0.46573172997568547773E-02;
    pat_weights[7][ 41] = 0.48205888648512683476E-02;
    pat_weights[7][ 42] = 0.49843645647655386012E-02;
    pat_weights[7][ 43] = 0.51485584789781777618E-02;
    pat_weights[7][ 44] = 0.53130866051870565663E-02;
    pat_weights[7][ 45] = 0.54778666939189508240E-02;
    pat_weights[7][ 46] = 0.56428181013844441585E-02;
    pat_weights[7][ 47] = 0.58078616599775673635E-02;
    pat_weights[7][ 48] = 0.59729195655081658049E-02;
    pat_weights[7][ 49] = 0.61379152800413850435E-02;
    pat_weights[7][ 50] = 0.63027734490857587172E-02;
    pat_weights[7][ 51] = 0.64674198318036867274E-02;
    pat_weights[7][ 52] = 0.66317812429018878941E-02;
    pat_weights[7][ 53] = 0.67957855048827733948E-02;
    pat_weights[7][ 54] = 0.69593614093904229394E-02;
    pat_weights[7][ 55] = 0.71224386864583871532E-02;
    pat_weights[7][ 56] = 0.72849479805538070639E-02;
    pat_weights[7][ 57] = 0.74468208324075910174E-02;
    pat_weights[7][ 58] = 0.76079896657190565832E-02;
    pat_weights[7][ 59] = 0.77683877779219912200E-02;
    pat_weights[7][ 60] = 0.79279493342948491103E-02;
    pat_weights[7][ 61] = 0.80866093647888599710E-02;
    pat_weights[7][ 62] = 0.82443037630328680306E-02;
    pat_weights[7][ 63] = 0.84009692870519326354E-02;
    pat_weights[7][ 64] = 0.85565435613076896192E-02;
    pat_weights[7][ 65] = 0.87109650797320868736E-02;
    pat_weights[7][ 66] = 0.88641732094824942641E-02;
    pat_weights[7][ 67] = 0.90161081951956431600E-02;
    pat_weights[7][ 68] = 0.91667111635607884067E-02;
    pat_weights[7][ 69] = 0.93159241280693950932E-02;
    pat_weights[7][ 70] = 0.94636899938300652943E-02;
    pat_weights[7][ 71] = 0.96099525623638830097E-02;
    pat_weights[7][ 72] = 0.97546565363174114611E-02;
    pat_weights[7][ 73] = 0.98977475240487497440E-02;
    pat_weights[7][ 74] = 0.10039172044056840798E-01;
    pat_weights[7][ 75] = 0.10178877529236079733E-01;
    pat_weights[7][ 76] = 0.10316812330947621682E-01;
    pat_weights[7][ 77] = 0.10452925722906011926E-01;
    pat_weights[7][ 78] = 0.10587167904885197931E-01;
    pat_weights[7][ 79] = 0.10719490006251933623E-01;
    pat_weights[7][ 80] = 0.10849844089337314099E-01;
    pat_weights[7][ 81] = 0.10978183152658912470E-01;
    pat_weights[7][ 82] = 0.11104461134006926537E-01;
    pat_weights[7][ 83] = 0.11228632913408049354E-01;
    pat_weights[7][ 84] = 0.11350654315980596602E-01;
    pat_weights[7][ 85] = 0.11470482114693874380E-01;
    pat_weights[7][ 86] = 0.11588074033043952568E-01;
    pat_weights[7][ 87] = 0.11703388747657003101E-01;
    pat_weights[7][ 88] = 0.11816385890830235763E-01;
    pat_weights[7][ 89] = 0.11927026053019270040E-01;
    pat_weights[7][ 90] = 0.12035270785279562630E-01;
    pat_weights[7][ 91] = 0.12141082601668299679E-01;
    pat_weights[7][ 92] = 0.12244424981611985899E-01;
    pat_weights[7][ 93] = 0.12345262372243838455E-01;
    pat_weights[7][ 94] = 0.12443560190714035263E-01;
    pat_weights[7][ 95] = 0.12539284826474884353E-01;
    pat_weights[7][ 96] = 0.12632403643542078765E-01;
    pat_weights[7][ 97] = 0.12722884982732382906E-01;
    pat_weights[7][ 98] = 0.12810698163877361967E-01;
    pat_weights[7][ 99] = 0.12895813488012114694E-01;
    pat_weights[7][100] = 0.12978202239537399286E-01;
    pat_weights[7][101] = 0.13057836688353048840E-01;
    pat_weights[7][102] = 0.13134690091960152836E-01;
    pat_weights[7][103] = 0.13208736697529129966E-01;
    pat_weights[7][104] = 0.13279951743930530650E-01;
    pat_weights[7][105] = 0.13348311463725179953E-01;
    pat_weights[7][106] = 0.13413793085110098513E-01;
    pat_weights[7][107] = 0.13476374833816515982E-01;
    pat_weights[7][108] = 0.13536035934956213614E-01;
    pat_weights[7][109] = 0.13592756614812395910E-01;
    pat_weights[7][110] = 0.13646518102571291428E-01;
    pat_weights[7][111] = 0.13697302631990716258E-01;
    pat_weights[7][112] = 0.13745093443001896632E-01;
    pat_weights[7][113] = 0.13789874783240936517E-01;
    pat_weights[7][114] = 0.13831631909506428676E-01;
    pat_weights[7][115] = 0.13870351089139840997E-01;
    pat_weights[7][116] = 0.13906019601325461264E-01;
    pat_weights[7][117] = 0.13938625738306850804E-01;
    pat_weights[7][118] = 0.13968158806516938516E-01;
    pat_weights[7][119] = 0.13994609127619079852E-01;
    pat_weights[7][120] = 0.14017968039456608810E-01;
    pat_weights[7][121] = 0.14038227896908623303E-01;
    pat_weights[7][122] = 0.14055382072649964277E-01;
    pat_weights[7][123] = 0.14069424957813575318E-01;
    pat_weights[7][124] = 0.14080351962553661325E-01;
    pat_weights[7][125] = 0.14088159516508301065E-01;
    pat_weights[7][126] = 0.14092845069160408355E-01;
    pat_weights[7][127] = 0.14094407090096179347E-01;
    pat_weights[7][128] = 0.14092845069160408355E-01;
    pat_weights[7][129] = 0.14088159516508301065E-01;
    pat_weights[7][130] = 0.14080351962553661325E-01;
    pat_weights[7][131] = 0.14069424957813575318E-01;
    pat_weights[7][132] = 0.14055382072649964277E-01;
    pat_weights[7][133] = 0.14038227896908623303E-01;
    pat_weights[7][134] = 0.14017968039456608810E-01;
    pat_weights[7][135] = 0.13994609127619079852E-01;
    pat_weights[7][136] = 0.13968158806516938516E-01;
    pat_weights[7][137] = 0.13938625738306850804E-01;
    pat_weights[7][138] = 0.13906019601325461264E-01;
    pat_weights[7][139] = 0.13870351089139840997E-01;
    pat_weights[7][140] = 0.13831631909506428676E-01;
    pat_weights[7][141] = 0.13789874783240936517E-01;
    pat_weights[7][142] = 0.13745093443001896632E-01;
    pat_weights[7][143] = 0.13697302631990716258E-01;
    pat_weights[7][144] = 0.13646518102571291428E-01;
    pat_weights[7][145] = 0.13592756614812395910E-01;
    pat_weights[7][146] = 0.13536035934956213614E-01;
    pat_weights[7][147] = 0.13476374833816515982E-01;
    pat_weights[7][148] = 0.13413793085110098513E-01;
    pat_weights[7][149] = 0.13348311463725179953E-01;
    pat_weights[7][150] = 0.13279951743930530650E-01;
    pat_weights[7][151] = 0.13208736697529129966E-01;
    pat_weights[7][152] = 0.13134690091960152836E-01;
    pat_weights[7][153] = 0.13057836688353048840E-01;
    pat_weights[7][154] = 0.12978202239537399286E-01;
    pat_weights[7][155] = 0.12895813488012114694E-01;
    pat_weights[7][156] = 0.12810698163877361967E-01;
    pat_weights[7][157] = 0.12722884982732382906E-01;
    pat_weights[7][158] = 0.12632403643542078765E-01;
    pat_weights[7][159] = 0.12539284826474884353E-01;
    pat_weights[7][160] = 0.12443560190714035263E-01;
    pat_weights[7][161] = 0.12345262372243838455E-01;
    pat_weights[7][162] = 0.12244424981611985899E-01;
    pat_weights[7][163] = 0.12141082601668299679E-01;
    pat_weights[7][164] = 0.12035270785279562630E-01;
    pat_weights[7][165] = 0.11927026053019270040E-01;
    pat_weights[7][166] = 0.11816385890830235763E-01;
    pat_weights[7][167] = 0.11703388747657003101E-01;
    pat_weights[7][168] = 0.11588074033043952568E-01;
    pat_weights[7][169] = 0.11470482114693874380E-01;
    pat_weights[7][170] = 0.11350654315980596602E-01;
    pat_weights[7][171] = 0.11228632913408049354E-01;
    pat_weights[7][172] = 0.11104461134006926537E-01;
    pat_weights[7][173] = 0.10978183152658912470E-01;
    pat_weights[7][174] = 0.10849844089337314099E-01;
    pat_weights[7][175] = 0.10719490006251933623E-01;
    pat_weights[7][176] = 0.10587167904885197931E-01;
    pat_weights[7][177] = 0.10452925722906011926E-01;
    pat_weights[7][178] = 0.10316812330947621682E-01;
    pat_weights[7][179] = 0.10178877529236079733E-01;
    pat_weights[7][180] = 0.10039172044056840798E-01;
    pat_weights[7][181] = 0.98977475240487497440E-02;
    pat_weights[7][182] = 0.97546565363174114611E-02;
    pat_weights[7][183] = 0.96099525623638830097E-02;
    pat_weights[7][184] = 0.94636899938300652943E-02;
    pat_weights[7][185] = 0.93159241280693950932E-02;
    pat_weights[7][186] = 0.91667111635607884067E-02;
    pat_weights[7][187] = 0.90161081951956431600E-02;
    pat_weights[7][188] = 0.88641732094824942641E-02;
    pat_weights[7][189] = 0.87109650797320868736E-02;
    pat_weights[7][190] = 0.85565435613076896192E-02;
    pat_weights[7][191] = 0.84009692870519326354E-02;
    pat_weights[7][192] = 0.82443037630328680306E-02;
    pat_weights[7][193] = 0.80866093647888599710E-02;
    pat_weights[7][194] = 0.79279493342948491103E-02;
    pat_weights[7][195] = 0.77683877779219912200E-02;
    pat_weights[7][196] = 0.76079896657190565832E-02;
    pat_weights[7][197] = 0.74468208324075910174E-02;
    pat_weights[7][198] = 0.72849479805538070639E-02;
    pat_weights[7][199] = 0.71224386864583871532E-02;
    pat_weights[7][200] = 0.69593614093904229394E-02;
    pat_weights[7][201] = 0.67957855048827733948E-02;
    pat_weights[7][202] = 0.66317812429018878941E-02;
    pat_weights[7][203] = 0.64674198318036867274E-02;
    pat_weights[7][204] = 0.63027734490857587172E-02;
    pat_weights[7][205] = 0.61379152800413850435E-02;
    pat_weights[7][206] = 0.59729195655081658049E-02;
    pat_weights[7][207] = 0.58078616599775673635E-02;
    pat_weights[7][208] = 0.56428181013844441585E-02;
    pat_weights[7][209] = 0.54778666939189508240E-02;
    pat_weights[7][210] = 0.53130866051870565663E-02;
    pat_weights[7][211] = 0.51485584789781777618E-02;
    pat_weights[7][212] = 0.49843645647655386012E-02;
    pat_weights[7][213] = 0.48205888648512683476E-02;
    pat_weights[7][214] = 0.46573172997568547773E-02;
    pat_weights[7][215] = 0.44946378920320678616E-02;
    pat_weights[7][216] = 0.43326409680929828545E-02;
    pat_weights[7][217] = 0.41714193769840788528E-02;
    pat_weights[7][218] = 0.40110687240750233989E-02;
    pat_weights[7][219] = 0.38516876166398709241E-02;
    pat_weights[7][220] = 0.36933779170256508183E-02;
    pat_weights[7][221] = 0.35362449977167777340E-02;
    pat_weights[7][222] = 0.33803979910869203823E-02;
    pat_weights[7][223] = 0.32259500250878684614E-02;
    pat_weights[7][224] = 0.30730184347025783234E-02;
    pat_weights[7][225] = 0.29217249379178197538E-02;
    pat_weights[7][226] = 0.27721957645934509940E-02;
    pat_weights[7][227] = 0.26245617274044295626E-02;
    pat_weights[7][228] = 0.24789582266575679307E-02;
    pat_weights[7][229] = 0.23355251860571608737E-02;
    pat_weights[7][230] = 0.21944069253638388388E-02;
    pat_weights[7][231] = 0.20557519893273465236E-02;
    pat_weights[7][232] = 0.19197129710138724125E-02;
    pat_weights[7][233] = 0.17864463917586498247E-02;
    pat_weights[7][234] = 0.16561127281544526052E-02;
    pat_weights[7][235] = 0.15288767050877655684E-02;
    pat_weights[7][236] = 0.14049079956551446427E-02;
    pat_weights[7][237] = 0.12843824718970101768E-02;
    pat_weights[7][238] = 0.11674841174299594077E-02;
    pat_weights[7][239] = 0.10544076228633167722E-02;
    pat_weights[7][240] = 0.94536151685852538246E-03;
    pat_weights[7][241] = 0.84057143271072246365E-03;
    pat_weights[7][242] = 0.74028280424450333046E-03;
    pat_weights[7][243] = 0.64476204130572477933E-03;
    pat_weights[7][244] = 0.55429531493037471492E-03;
    pat_weights[7][245] = 0.46918492424785040975E-03;
    pat_weights[7][246] = 0.38974528447328229322E-03;
    pat_weights[7][247] = 0.31630366082226447689E-03;
    pat_weights[7][248] = 0.24921240048299729402E-03;
    pat_weights[7][249] = 0.18887326450650491366E-03;
    pat_weights[7][250] = 0.13575491094922871973E-03;
    pat_weights[7][251] = 0.90372734658751149261E-04;
    pat_weights[7][252] = 0.53275293669780613125E-04;
    pat_weights[7][253] = 0.25157870384280661489E-04;
    pat_weights[7][254] = 0.69379364324108267170E-05;

    pat_weights[8][  0] = 0.945715933950007048827E-06;
    pat_weights[8][  1] = 0.345456507169149134898E-05;
    pat_weights[8][  2] = 0.736624069102321668857E-05;
    pat_weights[8][  3] = 0.125792781889592743525E-04;
    pat_weights[8][  4] = 0.190213681905875816679E-04;
    pat_weights[8][  5] = 0.266376412339000901358E-04;
    pat_weights[8][  6] = 0.353751372055189588628E-04;
    pat_weights[8][  7] = 0.451863674126296143105E-04;
    pat_weights[8][  8] = 0.560319507856164252140E-04;
    pat_weights[8][  9] = 0.678774554733972416227E-04;
    pat_weights[8][ 10] = 0.806899228014035293851E-04;
    pat_weights[8][ 11] = 0.944366322532705527066E-04;
    pat_weights[8][ 12] = 0.109085545645741522051E-03;
    pat_weights[8][ 13] = 0.124606200241498368482E-03;
    pat_weights[8][ 14] = 0.140970302204104791413E-03;
    pat_weights[8][ 15] = 0.158151830411132242924E-03;
    pat_weights[8][ 16] = 0.176126765545083195474E-03;
    pat_weights[8][ 17] = 0.194872642236641146532E-03;
    pat_weights[8][ 18] = 0.214368090034216937149E-03;
    pat_weights[8][ 19] = 0.234592462123925204879E-03;
    pat_weights[8][ 20] = 0.255525589595236862014E-03;
    pat_weights[8][ 21] = 0.277147657465187357459E-03;
    pat_weights[8][ 22] = 0.299439176850911730874E-03;
    pat_weights[8][ 23] = 0.322381020652862389664E-03;
    pat_weights[8][ 24] = 0.345954492129903871350E-03;
    pat_weights[8][ 25] = 0.370141402122251665232E-03;
    pat_weights[8][ 26] = 0.394924138246873704434E-03;
    pat_weights[8][ 27] = 0.420285716355361231823E-03;
    pat_weights[8][ 28] = 0.446209810101403247488E-03;
    pat_weights[8][ 29] = 0.472680758429262691232E-03;
    pat_weights[8][ 30] = 0.499683553312800484519E-03;
    pat_weights[8][ 31] = 0.527203811431658386125E-03;
    pat_weights[8][ 32] = 0.555227733977307579715E-03;
    pat_weights[8][ 33] = 0.583742058714979703847E-03;
    pat_weights[8][ 34] = 0.612734008012225209294E-03;
    pat_weights[8][ 35] = 0.642191235948505088403E-03;
    pat_weights[8][ 36] = 0.672101776960108194646E-03;
    pat_weights[8][ 37] = 0.702453997827572321358E-03;
    pat_weights[8][ 38] = 0.733236554224767912055E-03;
    pat_weights[8][ 39] = 0.764438352543882784191E-03;
    pat_weights[8][ 40] = 0.796048517297550871506E-03;
    pat_weights[8][ 41] = 0.828056364077226302608E-03;
    pat_weights[8][ 42] = 0.860451377808527848128E-03;
    pat_weights[8][ 43] = 0.893223195879324912340E-03;
    pat_weights[8][ 44] = 0.926361595613111283368E-03;
    pat_weights[8][ 45] = 0.959856485506936206261E-03;
    pat_weights[8][ 46] = 0.993697899638760857945E-03;
    pat_weights[8][ 47] = 0.102787599466367326179E-02;
    pat_weights[8][ 48] = 0.106238104885340071375E-02;
    pat_weights[8][ 49] = 0.109720346268191941940E-02;
    pat_weights[8][ 50] = 0.113233376051597664917E-02;
    pat_weights[8][ 51] = 0.116776259302858043685E-02;
    pat_weights[8][ 52] = 0.120348074001265964881E-02;
    pat_weights[8][ 53] = 0.123947911332878396534E-02;
    pat_weights[8][ 54] = 0.127574875977346947345E-02;
    pat_weights[8][ 55] = 0.131228086370221478128E-02;
    pat_weights[8][ 56] = 0.134906674928353113127E-02;
    pat_weights[8][ 57] = 0.138609788229672549700E-02;
    pat_weights[8][ 58] = 0.142336587141720519900E-02;
    pat_weights[8][ 59] = 0.146086246895890987689E-02;
    pat_weights[8][ 60] = 0.149857957106456636214E-02;
    pat_weights[8][ 61] = 0.153650921735128916170E-02;
    pat_weights[8][ 62] = 0.157464359003212166189E-02;
    pat_weights[8][ 63] = 0.161297501254393423070E-02;
    pat_weights[8][ 64] = 0.165149594771914570655E-02;
    pat_weights[8][ 65] = 0.169019899554346019117E-02;
    pat_weights[8][ 66] = 0.172907689054461607168E-02;
    pat_weights[8][ 67] = 0.176812249885838886701E-02;
    pat_weights[8][ 68] = 0.180732881501808930079E-02;
    pat_weights[8][ 69] = 0.184668895851282540913E-02;
    pat_weights[8][ 70] = 0.188619617015808475394E-02;
    pat_weights[8][ 71] = 0.192584380831993546204E-02;
    pat_weights[8][ 72] = 0.196562534503150547732E-02;
    pat_weights[8][ 73] = 0.200553436203751169944E-02;
    pat_weights[8][ 74] = 0.204556454679958293446E-02;
    pat_weights[8][ 75] = 0.208570968849203942640E-02;
    pat_weights[8][ 76] = 0.212596367401472533045E-02;
    pat_weights[8][ 77] = 0.216632048404649142727E-02;
    pat_weights[8][ 78] = 0.220677418916003329194E-02;
    pat_weights[8][ 79] = 0.224731894601603393082E-02;
    pat_weights[8][ 80] = 0.228794899365195972378E-02;
    pat_weights[8][ 81] = 0.232865864987842738864E-02;
    pat_weights[8][ 82] = 0.236944230779380495146E-02;
    pat_weights[8][ 83] = 0.241029443242563417382E-02;
    pat_weights[8][ 84] = 0.245120955750556483923E-02;
    pat_weights[8][ 85] = 0.249218228238276930060E-02;
    pat_weights[8][ 86] = 0.253320726907925325750E-02;
    pat_weights[8][ 87] = 0.257427923948908888092E-02;
    pat_weights[8][ 88] = 0.261539297272236109225E-02;
    pat_weights[8][ 89] = 0.265654330259352828314E-02;
    pat_weights[8][ 90] = 0.269772511525294586667E-02;
    pat_weights[8][ 91] = 0.273893334695947541201E-02;
    pat_weights[8][ 92] = 0.278016298199139435045E-02;
    pat_weights[8][ 93] = 0.282140905069222207923E-02;
    pat_weights[8][ 94] = 0.286266662764757868253E-02;
    pat_weights[8][ 95] = 0.290393082998878368175E-02;
    pat_weights[8][ 96] = 0.294519681581857582284E-02;
    pat_weights[8][ 97] = 0.298645978275408290247E-02;
    pat_weights[8][ 98] = 0.302771496658198544480E-02;
    pat_weights[8][ 99] = 0.306895764002069252174E-02;
    pat_weights[8][100] = 0.311018311158427546158E-02;
    pat_weights[8][101] = 0.315138672454287935858E-02;
    pat_weights[8][102] = 0.319256385597434736790E-02;
    pat_weights[8][103] = 0.323370991590184336368E-02;
    pat_weights[8][104] = 0.327482034651233969564E-02;
    pat_weights[8][105] = 0.331589062145094394706E-02;
    pat_weights[8][106] = 0.335691624518616761342E-02;
    pat_weights[8][107] = 0.339789275244138669739E-02;
    pat_weights[8][108] = 0.343881570768790591876E-02;
    pat_weights[8][109] = 0.347968070469521146972E-02;
    pat_weights[8][110] = 0.352048336613417922682E-02;
    pat_weights[8][111] = 0.356121934322919357659E-02;
    pat_weights[8][112] = 0.360188431545532431869E-02;
    pat_weights[8][113] = 0.364247399027690353194E-02;
    pat_weights[8][114] = 0.368298410292403911967E-02;
    pat_weights[8][115] = 0.372341041620379550870E-02;
    pat_weights[8][116] = 0.376374872034296338241E-02;
    pat_weights[8][117] = 0.380399483285952829161E-02;
    pat_weights[8][118] = 0.384414459846013158917E-02;
    pat_weights[8][119] = 0.388419388896099560998E-02;
    pat_weights[8][120] = 0.392413860322995774660E-02;
    pat_weights[8][121] = 0.396397466714742455513E-02;
    pat_weights[8][122] = 0.400369803358421688562E-02;
    pat_weights[8][123] = 0.404330468239442998549E-02;
    pat_weights[8][124] = 0.408279062042157838350E-02;
    pat_weights[8][125] = 0.412215188151643401528E-02;
    pat_weights[8][126] = 0.416138452656509745764E-02;
    pat_weights[8][127] = 0.420048464352596631772E-02;
    pat_weights[8][128] = 0.423944834747438184434E-02;
    pat_weights[8][129] = 0.427827178065384480959E-02;
    pat_weights[8][130] = 0.431695111253279479928E-02;
    pat_weights[8][131] = 0.435548253986604343679E-02;
    pat_weights[8][132] = 0.439386228676004195260E-02;
    pat_weights[8][133] = 0.443208660474124713206E-02;
    pat_weights[8][134] = 0.447015177282692726900E-02;
    pat_weights[8][135] = 0.450805409759782158001E-02;
    pat_weights[8][136] = 0.454578991327213285488E-02;
    pat_weights[8][137] = 0.458335558178039420335E-02;
    pat_weights[8][138] = 0.462074749284080687482E-02;
    pat_weights[8][139] = 0.465796206403469754658E-02;
    pat_weights[8][140] = 0.469499574088179046532E-02;
    pat_weights[8][141] = 0.473184499691503264714E-02;
    pat_weights[8][142] = 0.476850633375474925263E-02;
    pat_weights[8][143] = 0.480497628118194150483E-02;
    pat_weights[8][144] = 0.484125139721057135214E-02;
    pat_weights[8][145] = 0.487732826815870573054E-02;
    pat_weights[8][146] = 0.491320350871841897367E-02;
    pat_weights[8][147] = 0.494887376202437487201E-02;
    pat_weights[8][148] = 0.498433569972103029914E-02;
    pat_weights[8][149] = 0.501958602202842039909E-02;
    pat_weights[8][150] = 0.505462145780650125058E-02;
    pat_weights[8][151] = 0.508943876461803986674E-02;
    pat_weights[8][152] = 0.512403472879005351831E-02;
    pat_weights[8][153] = 0.515840616547381084096E-02;
    pat_weights[8][154] = 0.519254991870341614863E-02;
    pat_weights[8][155] = 0.522646286145300596306E-02;
    pat_weights[8][156] = 0.526014189569259311205E-02;
    pat_weights[8][157] = 0.529358395244259896547E-02;
    pat_weights[8][158] = 0.532678599182711857974E-02;
    pat_weights[8][159] = 0.535974500312596681161E-02;
    pat_weights[8][160] = 0.539245800482555593606E-02;
    pat_weights[8][161] = 0.542492204466865704951E-02;
    pat_weights[8][162] = 0.545713419970309863995E-02;
    pat_weights[8][163] = 0.548909157632945623482E-02;
    pat_weights[8][164] = 0.552079131034778706457E-02;
    pat_weights[8][165] = 0.555223056700346326850E-02;
    pat_weights[8][166] = 0.558340654103215637610E-02;
    pat_weights[8][167] = 0.561431645670402467678E-02;
    pat_weights[8][168] = 0.564495756786715368885E-02;
    pat_weights[8][169] = 0.567532715799029830087E-02;
    pat_weights[8][170] = 0.570542254020497332312E-02;
    pat_weights[8][171] = 0.573524105734693719020E-02;
    pat_weights[8][172] = 0.576478008199711142954E-02;
    pat_weights[8][173] = 0.579403701652197628421E-02;
    pat_weights[8][174] = 0.582300929311348057702E-02;
    pat_weights[8][175] = 0.585169437382850155033E-02;
    pat_weights[8][176] = 0.588008975062788803205E-02;
    pat_weights[8][177] = 0.590819294541511788161E-02;
    pat_weights[8][178] = 0.593600151007459827614E-02;
    pat_weights[8][179] = 0.596351302650963502011E-02;
    pat_weights[8][180] = 0.599072510668009471472E-02;
    pat_weights[8][181] = 0.601763539263978131522E-02;
    pat_weights[8][182] = 0.604424155657354634589E-02;
    pat_weights[8][183] = 0.607054130083414983949E-02;
    pat_weights[8][184] = 0.609653235797888692923E-02;
    pat_weights[8][185] = 0.612221249080599294931E-02;
    pat_weights[8][186] = 0.614757949239083790214E-02;
    pat_weights[8][187] = 0.617263118612191922727E-02;
    pat_weights[8][188] = 0.619736542573665996342E-02;
    pat_weights[8][189] = 0.622178009535701763157E-02;
    pat_weights[8][190] = 0.624587310952490748541E-02;
    pat_weights[8][191] = 0.626964241323744217671E-02;
    pat_weights[8][192] = 0.629308598198198836688E-02;
    pat_weights[8][193] = 0.631620182177103938227E-02;
    pat_weights[8][194] = 0.633898796917690165912E-02;
    pat_weights[8][195] = 0.636144249136619145314E-02;
    pat_weights[8][196] = 0.638356348613413709795E-02;
    pat_weights[8][197] = 0.640534908193868098342E-02;
    pat_weights[8][198] = 0.642679743793437438922E-02;
    pat_weights[8][199] = 0.644790674400605734710E-02;
    pat_weights[8][200] = 0.646867522080231481688E-02;
    pat_weights[8][201] = 0.648910111976869964292E-02;
    pat_weights[8][202] = 0.650918272318071200827E-02;
    pat_weights[8][203] = 0.652891834417652442012E-02;
    pat_weights[8][204] = 0.654830632678944064054E-02;
    pat_weights[8][205] = 0.656734504598007641819E-02;
    pat_weights[8][206] = 0.658603290766824937794E-02;
    pat_weights[8][207] = 0.660436834876456498276E-02;
    pat_weights[8][208] = 0.662234983720168509457E-02;
    pat_weights[8][209] = 0.663997587196526532519E-02;
    pat_weights[8][210] = 0.665724498312454708217E-02;
    pat_weights[8][211] = 0.667415573186258997654E-02;
    pat_weights[8][212] = 0.669070671050613006584E-02;
    pat_weights[8][213] = 0.670689654255504925648E-02;
    pat_weights[8][214] = 0.672272388271144108036E-02;
    pat_weights[8][215] = 0.673818741690825799086E-02;
    pat_weights[8][216] = 0.675328586233752529078E-02;
    pat_weights[8][217] = 0.676801796747810680683E-02;
    pat_weights[8][218] = 0.678238251212300746082E-02;
    pat_weights[8][219] = 0.679637830740619795480E-02;
    pat_weights[8][220] = 0.681000419582894688374E-02;
    pat_weights[8][221] = 0.682325905128564571420E-02;
    pat_weights[8][222] = 0.683614177908911221841E-02;
    pat_weights[8][223] = 0.684865131599535812903E-02;
    pat_weights[8][224] = 0.686078663022780697951E-02;
    pat_weights[8][225] = 0.687254672150094831613E-02;
    pat_weights[8][226] = 0.688393062104341470995E-02;
    pat_weights[8][227] = 0.689493739162046825872E-02;
    pat_weights[8][228] = 0.690556612755588354803E-02;
    pat_weights[8][229] = 0.691581595475321433825E-02;
    pat_weights[8][230] = 0.692568603071643155621E-02;
    pat_weights[8][231] = 0.693517554456992049848E-02;
    pat_weights[8][232] = 0.694428371707782549438E-02;
    pat_weights[8][233] = 0.695300980066273063177E-02;
    pat_weights[8][234] = 0.696135307942366551493E-02;
    pat_weights[8][235] = 0.696931286915342540213E-02;
    pat_weights[8][236] = 0.697688851735519545845E-02;
    pat_weights[8][237] = 0.698407940325846925786E-02;
    pat_weights[8][238] = 0.699088493783425207545E-02;
    pat_weights[8][239] = 0.699730456380953992594E-02;
    pat_weights[8][240] = 0.700333775568106572820E-02;
    pat_weights[8][241] = 0.700898401972830440494E-02;
    pat_weights[8][242] = 0.701424289402572916425E-02;
    pat_weights[8][243] = 0.701911394845431165171E-02;
    pat_weights[8][244] = 0.702359678471225911031E-02;
    pat_weights[8][245] = 0.702769103632498213858E-02;
    pat_weights[8][246] = 0.703139636865428709508E-02;
    pat_weights[8][247] = 0.703471247890678765907E-02;
    pat_weights[8][248] = 0.703763909614153052319E-02;
    pat_weights[8][249] = 0.704017598127683066242E-02;
    pat_weights[8][250] = 0.704232292709631209597E-02;
    pat_weights[8][251] = 0.704407975825415053266E-02;
    pat_weights[8][252] = 0.704544633127951476780E-02;
    pat_weights[8][253] = 0.704642253458020417748E-02;
    pat_weights[8][254] = 0.704700828844548013730E-02;
    pat_weights[8][255] = 0.704720354504808967346E-02;
    pat_weights[8][256] = 0.704700828844548013730E-02;
    pat_weights[8][257] = 0.704642253458020417748E-02;
    pat_weights[8][258] = 0.704544633127951476780E-02;
    pat_weights[8][259] = 0.704407975825415053266E-02;
    pat_weights[8][260] = 0.704232292709631209597E-02;
    pat_weights[8][261] = 0.704017598127683066242E-02;
    pat_weights[8][262] = 0.703763909614153052319E-02;
    pat_weights[8][263] = 0.703471247890678765907E-02;
    pat_weights[8][264] = 0.703139636865428709508E-02;
    pat_weights[8][265] = 0.702769103632498213858E-02;
    pat_weights[8][266] = 0.702359678471225911031E-02;
    pat_weights[8][267] = 0.701911394845431165171E-02;
    pat_weights[8][268] = 0.701424289402572916425E-02;
    pat_weights[8][269] = 0.700898401972830440494E-02;
    pat_weights[8][270] = 0.700333775568106572820E-02;
    pat_weights[8][271] = 0.699730456380953992594E-02;
    pat_weights[8][272] = 0.699088493783425207545E-02;
    pat_weights[8][273] = 0.698407940325846925786E-02;
    pat_weights[8][274] = 0.697688851735519545845E-02;
    pat_weights[8][275] = 0.696931286915342540213E-02;
    pat_weights[8][276] = 0.696135307942366551493E-02;
    pat_weights[8][277] = 0.695300980066273063177E-02;
    pat_weights[8][278] = 0.694428371707782549438E-02;
    pat_weights[8][279] = 0.693517554456992049848E-02;
    pat_weights[8][280] = 0.692568603071643155621E-02;
    pat_weights[8][281] = 0.691581595475321433825E-02;
    pat_weights[8][282] = 0.690556612755588354803E-02;
    pat_weights[8][283] = 0.689493739162046825872E-02;
    pat_weights[8][284] = 0.688393062104341470995E-02;
    pat_weights[8][285] = 0.687254672150094831613E-02;
    pat_weights[8][286] = 0.686078663022780697951E-02;
    pat_weights[8][287] = 0.684865131599535812903E-02;
    pat_weights[8][288] = 0.683614177908911221841E-02;
    pat_weights[8][289] = 0.682325905128564571420E-02;
    pat_weights[8][290] = 0.681000419582894688374E-02;
    pat_weights[8][291] = 0.679637830740619795480E-02;
    pat_weights[8][292] = 0.678238251212300746082E-02;
    pat_weights[8][293] = 0.676801796747810680683E-02;
    pat_weights[8][294] = 0.675328586233752529078E-02;
    pat_weights[8][295] = 0.673818741690825799086E-02;
    pat_weights[8][296] = 0.672272388271144108036E-02;
    pat_weights[8][297] = 0.670689654255504925648E-02;
    pat_weights[8][298] = 0.669070671050613006584E-02;
    pat_weights[8][299] = 0.667415573186258997654E-02;
    pat_weights[8][300] = 0.665724498312454708217E-02;
    pat_weights[8][301] = 0.663997587196526532519E-02;
    pat_weights[8][302] = 0.662234983720168509457E-02;
    pat_weights[8][303] = 0.660436834876456498276E-02;
    pat_weights[8][304] = 0.658603290766824937794E-02;
    pat_weights[8][305] = 0.656734504598007641819E-02;
    pat_weights[8][306] = 0.654830632678944064054E-02;
    pat_weights[8][307] = 0.652891834417652442012E-02;
    pat_weights[8][308] = 0.650918272318071200827E-02;
    pat_weights[8][309] = 0.648910111976869964292E-02;
    pat_weights[8][310] = 0.646867522080231481688E-02;
    pat_weights[8][311] = 0.644790674400605734710E-02;
    pat_weights[8][312] = 0.642679743793437438922E-02;
    pat_weights[8][313] = 0.640534908193868098342E-02;
    pat_weights[8][314] = 0.638356348613413709795E-02;
    pat_weights[8][315] = 0.636144249136619145314E-02;
    pat_weights[8][316] = 0.633898796917690165912E-02;
    pat_weights[8][317] = 0.631620182177103938227E-02;
    pat_weights[8][318] = 0.629308598198198836688E-02;
    pat_weights[8][319] = 0.626964241323744217671E-02;
    pat_weights[8][320] = 0.624587310952490748541E-02;
    pat_weights[8][321] = 0.622178009535701763157E-02;
    pat_weights[8][322] = 0.619736542573665996342E-02;
    pat_weights[8][323] = 0.617263118612191922727E-02;
    pat_weights[8][324] = 0.614757949239083790214E-02;
    pat_weights[8][325] = 0.612221249080599294931E-02;
    pat_weights[8][326] = 0.609653235797888692923E-02;
    pat_weights[8][327] = 0.607054130083414983949E-02;
    pat_weights[8][328] = 0.604424155657354634589E-02;
    pat_weights[8][329] = 0.601763539263978131522E-02;
    pat_weights[8][330] = 0.599072510668009471472E-02;
    pat_weights[8][331] = 0.596351302650963502011E-02;
    pat_weights[8][332] = 0.593600151007459827614E-02;
    pat_weights[8][333] = 0.590819294541511788161E-02;
    pat_weights[8][334] = 0.588008975062788803205E-02;
    pat_weights[8][335] = 0.585169437382850155033E-02;
    pat_weights[8][336] = 0.582300929311348057702E-02;
    pat_weights[8][337] = 0.579403701652197628421E-02;
    pat_weights[8][338] = 0.576478008199711142954E-02;
    pat_weights[8][339] = 0.573524105734693719020E-02;
    pat_weights[8][340] = 0.570542254020497332312E-02;
    pat_weights[8][341] = 0.567532715799029830087E-02;
    pat_weights[8][342] = 0.564495756786715368885E-02;
    pat_weights[8][343] = 0.561431645670402467678E-02;
    pat_weights[8][344] = 0.558340654103215637610E-02;
    pat_weights[8][345] = 0.555223056700346326850E-02;
    pat_weights[8][346] = 0.552079131034778706457E-02;
    pat_weights[8][347] = 0.548909157632945623482E-02;
    pat_weights[8][348] = 0.545713419970309863995E-02;
    pat_weights[8][349] = 0.542492204466865704951E-02;
    pat_weights[8][350] = 0.539245800482555593606E-02;
    pat_weights[8][351] = 0.535974500312596681161E-02;
    pat_weights[8][352] = 0.532678599182711857974E-02;
    pat_weights[8][353] = 0.529358395244259896547E-02;
    pat_weights[8][354] = 0.526014189569259311205E-02;
    pat_weights[8][355] = 0.522646286145300596306E-02;
    pat_weights[8][356] = 0.519254991870341614863E-02;
    pat_weights[8][357] = 0.515840616547381084096E-02;
    pat_weights[8][358] = 0.512403472879005351831E-02;
    pat_weights[8][359] = 0.508943876461803986674E-02;
    pat_weights[8][360] = 0.505462145780650125058E-02;
    pat_weights[8][361] = 0.501958602202842039909E-02;
    pat_weights[8][362] = 0.498433569972103029914E-02;
    pat_weights[8][363] = 0.494887376202437487201E-02;
    pat_weights[8][364] = 0.491320350871841897367E-02;
    pat_weights[8][365] = 0.487732826815870573054E-02;
    pat_weights[8][366] = 0.484125139721057135214E-02;
    pat_weights[8][367] = 0.480497628118194150483E-02;
    pat_weights[8][368] = 0.476850633375474925263E-02;
    pat_weights[8][369] = 0.473184499691503264714E-02;
    pat_weights[8][370] = 0.469499574088179046532E-02;
    pat_weights[8][371] = 0.465796206403469754658E-02;
    pat_weights[8][372] = 0.462074749284080687482E-02;
    pat_weights[8][373] = 0.458335558178039420335E-02;
    pat_weights[8][374] = 0.454578991327213285488E-02;
    pat_weights[8][375] = 0.450805409759782158001E-02;
    pat_weights[8][376] = 0.447015177282692726900E-02;
    pat_weights[8][377] = 0.443208660474124713206E-02;
    pat_weights[8][378] = 0.439386228676004195260E-02;
    pat_weights[8][379] = 0.435548253986604343679E-02;
    pat_weights[8][380] = 0.431695111253279479928E-02;
    pat_weights[8][381] = 0.427827178065384480959E-02;
    pat_weights[8][382] = 0.423944834747438184434E-02;
    pat_weights[8][383] = 0.420048464352596631772E-02;
    pat_weights[8][384] = 0.416138452656509745764E-02;
    pat_weights[8][385] = 0.412215188151643401528E-02;
    pat_weights[8][386] = 0.408279062042157838350E-02;
    pat_weights[8][387] = 0.404330468239442998549E-02;
    pat_weights[8][388] = 0.400369803358421688562E-02;
    pat_weights[8][389] = 0.396397466714742455513E-02;
    pat_weights[8][390] = 0.392413860322995774660E-02;
    pat_weights[8][391] = 0.388419388896099560998E-02;
    pat_weights[8][392] = 0.384414459846013158917E-02;
    pat_weights[8][393] = 0.380399483285952829161E-02;
    pat_weights[8][394] = 0.376374872034296338241E-02;
    pat_weights[8][395] = 0.372341041620379550870E-02;
    pat_weights[8][396] = 0.368298410292403911967E-02;
    pat_weights[8][397] = 0.364247399027690353194E-02;
    pat_weights[8][398] = 0.360188431545532431869E-02;
    pat_weights[8][399] = 0.356121934322919357659E-02;
    pat_weights[8][400] = 0.352048336613417922682E-02;
    pat_weights[8][401] = 0.347968070469521146972E-02;
    pat_weights[8][402] = 0.343881570768790591876E-02;
    pat_weights[8][403] = 0.339789275244138669739E-02;
    pat_weights[8][404] = 0.335691624518616761342E-02;
    pat_weights[8][405] = 0.331589062145094394706E-02;
    pat_weights[8][406] = 0.327482034651233969564E-02;
    pat_weights[8][407] = 0.323370991590184336368E-02;
    pat_weights[8][408] = 0.319256385597434736790E-02;
    pat_weights[8][409] = 0.315138672454287935858E-02;
    pat_weights[8][410] = 0.311018311158427546158E-02;
    pat_weights[8][411] = 0.306895764002069252174E-02;
    pat_weights[8][412] = 0.302771496658198544480E-02;
    pat_weights[8][413] = 0.298645978275408290247E-02;
    pat_weights[8][414] = 0.294519681581857582284E-02;
    pat_weights[8][415] = 0.290393082998878368175E-02;
    pat_weights[8][416] = 0.286266662764757868253E-02;
    pat_weights[8][417] = 0.282140905069222207923E-02;
    pat_weights[8][418] = 0.278016298199139435045E-02;
    pat_weights[8][419] = 0.273893334695947541201E-02;
    pat_weights[8][420] = 0.269772511525294586667E-02;
    pat_weights[8][421] = 0.265654330259352828314E-02;
    pat_weights[8][422] = 0.261539297272236109225E-02;
    pat_weights[8][423] = 0.257427923948908888092E-02;
    pat_weights[8][424] = 0.253320726907925325750E-02;
    pat_weights[8][425] = 0.249218228238276930060E-02;
    pat_weights[8][426] = 0.245120955750556483923E-02;
    pat_weights[8][427] = 0.241029443242563417382E-02;
    pat_weights[8][428] = 0.236944230779380495146E-02;
    pat_weights[8][429] = 0.232865864987842738864E-02;
    pat_weights[8][430] = 0.228794899365195972378E-02;
    pat_weights[8][431] = 0.224731894601603393082E-02;
    pat_weights[8][432] = 0.220677418916003329194E-02;
    pat_weights[8][433] = 0.216632048404649142727E-02;
    pat_weights[8][434] = 0.212596367401472533045E-02;
    pat_weights[8][435] = 0.208570968849203942640E-02;
    pat_weights[8][436] = 0.204556454679958293446E-02;
    pat_weights[8][437] = 0.200553436203751169944E-02;
    pat_weights[8][438] = 0.196562534503150547732E-02;
    pat_weights[8][439] = 0.192584380831993546204E-02;
    pat_weights[8][440] = 0.188619617015808475394E-02;
    pat_weights[8][441] = 0.184668895851282540913E-02;
    pat_weights[8][442] = 0.180732881501808930079E-02;
    pat_weights[8][443] = 0.176812249885838886701E-02;
    pat_weights[8][444] = 0.172907689054461607168E-02;
    pat_weights[8][445] = 0.169019899554346019117E-02;
    pat_weights[8][446] = 0.165149594771914570655E-02;
    pat_weights[8][447] = 0.161297501254393423070E-02;
    pat_weights[8][448] = 0.157464359003212166189E-02;
    pat_weights[8][449] = 0.153650921735128916170E-02;
    pat_weights[8][450] = 0.149857957106456636214E-02;
    pat_weights[8][451] = 0.146086246895890987689E-02;
    pat_weights[8][452] = 0.142336587141720519900E-02;
    pat_weights[8][453] = 0.138609788229672549700E-02;
    pat_weights[8][454] = 0.134906674928353113127E-02;
    pat_weights[8][455] = 0.131228086370221478128E-02;
    pat_weights[8][456] = 0.127574875977346947345E-02;
    pat_weights[8][457] = 0.123947911332878396534E-02;
    pat_weights[8][458] = 0.120348074001265964881E-02;
    pat_weights[8][459] = 0.116776259302858043685E-02;
    pat_weights[8][460] = 0.113233376051597664917E-02;
    pat_weights[8][461] = 0.109720346268191941940E-02;
    pat_weights[8][462] = 0.106238104885340071375E-02;
    pat_weights[8][463] = 0.102787599466367326179E-02;
    pat_weights[8][464] = 0.993697899638760857945E-03;
    pat_weights[8][465] = 0.959856485506936206261E-03;
    pat_weights[8][466] = 0.926361595613111283368E-03;
    pat_weights[8][467] = 0.893223195879324912340E-03;
    pat_weights[8][468] = 0.860451377808527848128E-03;
    pat_weights[8][469] = 0.828056364077226302608E-03;
    pat_weights[8][470] = 0.796048517297550871506E-03;
    pat_weights[8][471] = 0.764438352543882784191E-03;
    pat_weights[8][472] = 0.733236554224767912055E-03;
    pat_weights[8][473] = 0.702453997827572321358E-03;
    pat_weights[8][474] = 0.672101776960108194646E-03;
    pat_weights[8][475] = 0.642191235948505088403E-03;
    pat_weights[8][476] = 0.612734008012225209294E-03;
    pat_weights[8][477] = 0.583742058714979703847E-03;
    pat_weights[8][478] = 0.555227733977307579715E-03;
    pat_weights[8][479] = 0.527203811431658386125E-03;
    pat_weights[8][480] = 0.499683553312800484519E-03;
    pat_weights[8][481] = 0.472680758429262691232E-03;
    pat_weights[8][482] = 0.446209810101403247488E-03;
    pat_weights[8][483] = 0.420285716355361231823E-03;
    pat_weights[8][484] = 0.394924138246873704434E-03;
    pat_weights[8][485] = 0.370141402122251665232E-03;
    pat_weights[8][486] = 0.345954492129903871350E-03;
    pat_weights[8][487] = 0.322381020652862389664E-03;
    pat_weights[8][488] = 0.299439176850911730874E-03;
    pat_weights[8][489] = 0.277147657465187357459E-03;
    pat_weights[8][490] = 0.255525589595236862014E-03;
    pat_weights[8][491] = 0.234592462123925204879E-03;
    pat_weights[8][492] = 0.214368090034216937149E-03;
    pat_weights[8][493] = 0.194872642236641146532E-03;
    pat_weights[8][494] = 0.176126765545083195474E-03;
    pat_weights[8][495] = 0.158151830411132242924E-03;
    pat_weights[8][496] = 0.140970302204104791413E-03;
    pat_weights[8][497] = 0.124606200241498368482E-03;
    pat_weights[8][498] = 0.109085545645741522051E-03;
    pat_weights[8][499] = 0.944366322532705527066E-04;
    pat_weights[8][500] = 0.806899228014035293851E-04;
    pat_weights[8][501] = 0.678774554733972416227E-04;
    pat_weights[8][502] = 0.560319507856164252140E-04;
    pat_weights[8][503] = 0.451863674126296143105E-04;
    pat_weights[8][504] = 0.353751372055189588628E-04;
    pat_weights[8][505] = 0.266376412339000901358E-04;
    pat_weights[8][506] = 0.190213681905875816679E-04;
    pat_weights[8][507] = 0.125792781889592743525E-04;
    pat_weights[8][508] = 0.736624069102321668857E-05;
    pat_weights[8][509] = 0.345456507169149134898E-05;
    pat_weights[8][510] = 0.945715933950007048827E-06;
}

void GaussPatterson::set_pat_tolerance_inner(const double tol_in)
{
	pat_tolerance = tol_in;
}

void GaussPatterson::set_pat_tolerance_outer(const double tol_in)
{
	pat_tolerance_outer = tol_in;
}

GaussPatterson::~GaussPatterson()
{
	//if (pat_points != NULL) delete[] pat_points;
	//if (pat_funcs != NULL) delete[] pat_funcs;
	//if (pat_funcs2 != NULL) delete[] pat_funcs2;
	//if (pat_funcs_mult != NULL) {
		//for (int i=0; i < 6; i++) delete[] pat_funcs_mult[i];
		//delete[] pat_funcs_mult;
	//}
	if (pat_orders != NULL) delete[] pat_orders;
	if (pat_weights != NULL) {
		for (int i=0; i < 9; i++) {
			delete[] pat_weights[i];
		}
		delete[] pat_weights;
	}
}

double GaussPatterson::AdaptiveQuad(double (GaussPatterson::*func)(double), double a, double b, bool &converged, bool outer)
{
	// I tried using Wynn's epsilon algorithm to accelerate convergence but it just doesn't work well in general.
	// Perhaps it is only useful when it is slow to converge, e.g. near singularities (in which case, maybe better to use
	// Romberg integration). This doesn't seem worth implementing here.

	double result = 0, result_old;
	double tolerance = (outer) ? pat_tolerance_outer : pat_tolerance;
	int i, level = 0, istep, istart;
	double abavg = (a+b)/2, abdif = (b-a)/2;
	converged = true; // until proven otherwise
	double *funcptr = pat_funcs;
	//if (outer) funcptr = pat_funcs2;
	int order, j;
	do {
		result_old = result;
		order = pat_orders[level];
		istep = (pat_N+1) / (order+1);
		istart = istep - 1;
		istep *= 2;
		result = 0;
		for (j=0, i=istart; j < order; j += 2, i += istep) {
			funcptr[i] = (this->*func)(abavg + abdif*pat_points[i]);
			result += pat_weights[level][j]*funcptr[i];
		}
		istart = istep - 1;
		for (j=1, i=istart; j < order; j += 2, i += istep) {
			result += pat_weights[level][j]*funcptr[i];
		}
		if ((level > 1) and (abs(result-result_old) < tolerance*abs(result))) break;
	} while (++level < 9);

	/*
	int npoints;
	if (level==0) npoints=1;
	else if (level==1) npoints=3;
	else if (level==2) npoints=7;
	else if (level==3) npoints=15;
	else if (level==4) npoints=31;
	else if (level==5) npoints=63;
	else if (level==6) npoints=127;
	else if (level==7) npoints=255;
	else if (level==8) npoints=511;
	*/

	if (level==9) {
		converged = false;
		if (show_convergence_warning) warn("Gauss-Patterson quadrature did not achieve desired tolerance after NMAX=511 points");
	}
	return abdif*result;
}

