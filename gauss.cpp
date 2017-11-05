#include <iostream>
#include <cmath>
#include "gauss.h"
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

double GaussianIntegral::output(double (GaussianIntegral::*func)(double))
{
	double result = 0;

	for (int i = 0; i < numberOfPoints; i++)
		result += weights[i]*(this->*func)(points[i]);

	return result;
}

GaussianIntegral::~GaussianIntegral()
{
   if (points != NULL) delete[] points;
	if (weights != NULL) delete[] weights;
}

GaussLegendre::GaussLegendre(int N) : GaussianIntegral(N)
{
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
          weights[i] = -Gamma(alpha + N)/Gamma(double(N))/N/pp/p2;
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
          weights[i] = Gamma(alpha + N)*Gamma(beta + N)/Gamma(N + 1.0)/Gamma(N + alfbet + 1.0)*temp*pow(2.0, alfbet)/(pp*p2);
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

double GaussianIntegral::Gamma(const double xx)
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
