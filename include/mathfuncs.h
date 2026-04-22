#ifndef MATHHDR_H
#define MATHHDR_H
#include "errors.h"
#include <cmath>
#include <iostream>

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

template <typename T>
T *matrix(const int xN)
{
     T *temp = new T[xN];
     return temp;
}

template <typename T>
T **matrix(const int xN, const int yN)
{
     T **temp = new T*[xN];
     
     for (int i = 0; i < xN; i++)
	  temp[i] = new T[yN];
     
     return temp;
}

template <typename T>
T *matrix(const int xN, T in)
{
	T *temp = new T[xN];
	for (int i = 0; i < xN; i++)
		temp[i] = in;
     
	return temp;
}

template <typename T>
T **matrix(const int xN, const int yN, T in)
{
	T **temp = new T*[xN];
     
	for (int i = 0; i < xN; i++)
	{
		temp[i] = new T[yN];
		for (int j = 0; j < yN; j++)
			temp[i][j] = in;
	}
     
	return temp;
}

template <typename T>
T *matrix(const int xN, T *in)
{
	T *temp = new T[xN];
	for (int i = 0; i < xN; i++)
		temp[i] = in[i];
     
	return temp;
}

template <typename T>
T **matrix(const int xN, const int yN, T **in)
{
	T **temp = new T*[xN];
     
	for (int i = 0; i < xN; i++)
	{
		temp[i] = new T[yN];
		for (int j = 0; j < yN; j++)
			temp[i][j] = in[i][j];
	}
     
	return temp;
}

template <typename T>
void del(T *temp)
{
     delete[] temp;
}

template <typename T>
void del(T **temp, int xN)
{
     for (int i = 0; i < xN; i++) delete[] temp[i];
     delete[] temp;
}

template <typename QScalar>
QScalar Gamma(const QScalar x)
{
#ifdef USE_STAN
	using stan::math::sin;
	using stan::math::pow;
	using stan::math::exp;
#endif
     static const double cof[6] ={76.18009172947146, -86.50532032941677, 24.01409824083091,
                              -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};
     QScalar temp = 1.000000000190015;
     
     if (x < 0.0)
	  return M_PI/Gamma(1.0-x)/sin(x*M_PI);
     else
     {
		  for (int i = 0; i < 6; i++)
				temp += cof[i]/(x + i + 1);
		  return 2.5066282746310005*pow(x + 5.5, x + 0.5)*exp(-(x + 5.5))*temp/x;
     }
}

template <typename QScalar>
QScalar Beta(const QScalar x, const QScalar y)
{
	return Gamma(x)*Gamma(y)/Gamma(x+y);
}

template <typename QScalar>
QScalar DiGamma(const QScalar x)
{
#ifdef USE_STAN
	using stan::math::log;
#endif

	static double c = 8.5;
	static double euler_mascheroni = 0.57721566490153286060;
	QScalar r;
	QScalar value;
	QScalar x2;
	if ( x == 0.0 ) die("DiGamma function is singular at x=0");
	if (x < 0) return (DiGamma(1+x) - 1/x); // recursive identity to handle negative arguments
	if ( x <= 0.000001 ) {
		return (-euler_mascheroni - 1.0/x + 1.6449340668482264365 * x); //  Use approximation for small argument.
	}
	//  Reduce to DIGAMMA(X + N).
	value = 0.0;
	x2 = x;
	while (x2 < c) {
		value = value - 1.0/x2;
		x2 = x2 + 1.0;
	}
	//  Use Stirling's (actually de Moivre's) expansion.
	r = 1.0 / x2;
	value = value + log(x2) - 0.5 * r;
	r = r * r;

	value = value 
		- r * ( 1.0 / 12.0 
		- r * ( 1.0 / 120.0 
		- r * ( 1.0 / 252.0 
		- r * ( 1.0 / 240.0
		- r * ( 1.0 / 132.0 ) ) ) ) );

	return value;
}

template <typename QScalar>
inline QScalar G_Function(const QScalar b, const QScalar c, const QScalar x)  // This is the limit (2F1(a,b,c;x) - 1)/a as a --> 0
{
	int j=1;
	QScalar ans=0, fac=1;
	static double tol = 1e-12;
	do {
		fac *= ((b+j-1)/(c+j-1)) * x;
		ans += fac/j;
	} while (fac > tol*ans*j++);
	//cout << "Number of iterations: " << j << endl;
	return ans;
}

template <typename QScalar>
QScalar gammln(const QScalar xx)
{
#ifdef USE_STAN
	using stan::math::log;
#endif

	QScalar x,y,tmp,ser;
	static double cof[6] = {76.18009172947146,-86.50532032941677,
		24.01409824083091,-1.231739572450155,
		0.1208650973866179e-2,-0.5395239384953e-5};

	y = x = xx;
	tmp = x + 5.5;
	tmp -= (x+0.5)*log(tmp);
	ser = 1.000000000190015;
	for (int j=0; j <= 5; j++) ser += cof[j] / ++y;
	return (-tmp + log(2.5066282746310005*ser/x));
}

template <typename QScalar>
void gser(QScalar &gamser, const QScalar a, const QScalar x, QScalar &gln)
{
#ifdef USE_STAN
	using stan::math::exp;
	using stan::math::log;
#endif

	const int ITMAX = 100;
	const double EPS = 3.0e-7;
	QScalar sum,del,ap;

	gln = gammln(a);
	if (x <= 0.0) {
		if (x < 0.0) die("x less than 0 in routine gser");
		gamser = 0.0;
		return;
	} else {
		ap = a;
		del = sum = 1.0/a;
		for (int n=0; n < ITMAX; n++) {
			++ap;
			del *= x/ap;
			sum += del;
			if (fabs(del) < fabs(sum)*EPS) {
				gamser = sum * exp(-x + a*log(x) - (gln));
				return;
			}
		}
		die("a too large, ITMAX too small in routine gser");
		return;
	}
}

template <typename QScalar>
void gcf(QScalar &gammcf, const QScalar a, const QScalar x, QScalar &gln)
{
#ifdef USE_STAN
	using stan::math::exp;
	using stan::math::log;
#endif
	const int ITMAX = 100;
	const double EPS = 3.0e-7;
	const double FPMIN = 1.0e-30;
	int i;
	QScalar an,b,c,d,del,h;

	gln = gammln(a);
	b = x + 1.0 - a;
	c = 1.0/FPMIN;
	d = 1.0/b;
	h = d;
	for (i=1; i <= ITMAX; i++) {
		an = -i*(i-a);
		b += 2.0;
		d = an*d + b;
		if (fabs(d) < FPMIN) d=FPMIN;
		c = b + an/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d = 1.0/d;
		del = d*c;
		h *= del;
		if (fabs(del-1.0) < EPS) break;
	}
	if (i > ITMAX) die("a too large, ITMAX too small in gcf");
	gammcf = exp(-x + a*log(x) - (gln)) * h;
	return;
}

template <typename QScalar>
QScalar IncGamma(const QScalar a, const QScalar x)
{
	QScalar gamser,gammcf,gln;

	if ((x < 0.0) or (a <= 0.0)) die("Invalid arguments in routine gammp");
	if (x < (a+1.0)) {
		gser(gamser,a,x,gln);
		return gamser;
	} else {
		gcf(gammcf,a,x,gln);
		return 1.0-gammcf;
	}
}

template <typename QScalar>
void IncGammaP_and_Gamma(const QScalar a, const QScalar x, QScalar& GammaP, QScalar& gam)
{
	QScalar gln;

	if ((x < 0.0) or (a <= 0.0)) {
		if (x < 0.0) die("x < 0 in routine gammp (x=%g, a=%g)",x,a);
		if (a <= 0.0) die("a <= 0 in routine gammp (x=%g, a=%g)",x,a);
	}
	if (x < (a+1.0)) {
		QScalar gamser;
		gser(gamser,a,x,gln);
		GammaP = gamser;
	} else {
		QScalar gammcf;
		gcf(gammcf,a,x,gln);
		GammaP = 1.0-gammcf;
	}
	gam = exp(gln);
}

template <typename QScalar>
QScalar IncGammaUp(const QScalar a, const QScalar x)
{
	QScalar gln;

	if ((x < 0.0) or (a <= 0.0)) die("Invalid arguments in routine gammq");
	if (x < (a+1.0)) {
		QScalar gamser;
		gser(gamser,a,x,gln);
		return 1.0-gamser;
	} else {
		QScalar gammcf;
		gcf(gammcf,a,x,gln);
		return gammcf;
	}
}

template <typename QScalar>
QScalar erffc(const QScalar x)
{
	QScalar onehalf = 0.5;
	return x < 0.0 ? 1.0+IncGamma(onehalf,x*x) : IncGammaUp(onehalf,x*x);
}

template <typename QScalar>
QScalar erff(const QScalar x)
{
	QScalar onehalf = 0.5;
	return x < 0.0 ? -IncGamma(onehalf,x*x) : IncGamma(onehalf,x*x);
}

#define erfinv_a3 -0.140543331
#define erfinv_a2 0.914624893
#define erfinv_a1 -1.645349621
#define erfinv_a0 0.886226899

#define erfinv_b4 0.012229801
#define erfinv_b3 -0.329097515
#define erfinv_b2 1.442710462
#define erfinv_b1 -2.118377725
#define erfinv_b0 1

#define erfinv_c3 1.641345311
#define erfinv_c2 3.429567803
#define erfinv_c1 -1.62490649
#define erfinv_c0 -1.970840454

#define erfinv_d2 1.637067800
#define erfinv_d1 3.543889200
#define erfinv_d0 1

template <typename QScalar>
QScalar erfinv(QScalar x)
{
  QScalar x2, r, y;
  int  sign_x;

  if (x < -1 || x > 1)
    return NAN;

  if (x == 0)
    return 0;

  if (x > 0)
    sign_x = 1;
  else {
    sign_x = -1;
    x = -x;
  }

  if (x <= 0.7) {

    x2 = x * x;
    r =
      x * (((erfinv_a3 * x2 + erfinv_a2) * x2 + erfinv_a1) * x2 + erfinv_a0);
    r /= (((erfinv_b4 * x2 + erfinv_b3) * x2 + erfinv_b2) * x2 +
    erfinv_b1) * x2 + erfinv_b0;
  }
  else {
    y = sqrt (-log ((1 - x) / 2));
    r = (((erfinv_c3 * y + erfinv_c2) * y + erfinv_c1) * y + erfinv_c0);
    r /= ((erfinv_d2 * y + erfinv_d1) * y + erfinv_d0);
  }

  r = r * sign_x;
  x = x * sign_x;

  r -= (erff(r) - x) / (2 / sqrt (M_PI) * exp (-r * r));
  r -= (erff(r) - x) / (2 / sqrt (M_PI) * exp (-r * r));

  return r;
}

#undef erfinv_a3
#undef erfinv_a2
#undef erfinv_a1
#undef erfinv_a0

#undef erfinv_b4
#undef erfinv_b3
#undef erfinv_b2
#undef erfinv_b1
#undef erfinv_b0

#undef erfinv_c3
#undef erfinv_c2
#undef erfinv_c1
#undef erfinv_c0

#undef erfinv_d2
#undef erfinv_d1
#undef erfinv_d0

double Minor(double **A, int x, int y, int N);

inline double Determinant (double **A, int N)
{
	double result = 0.0;

	if (N == 1)
		result = A[0][0];
	else if (N == 2)
		result = A[0][0]*A[1][1] - A[0][1]*A[1][0];
	else
		for (int i = 0; i < N; i++)
			result += A[0][i]*Minor(A, 0, i, N);

	return result;
}

inline double Minor(double **A, int x, int y, int N)
{
	double answer = 0.0;
	double **B = matrix <double> (N-1, N-1);

	for (int i = 0, M = 0; i < N-1; i++,M++)
	{
		if (x == i) M++;
		for (int j = 0, O = 0; j < N-1; j++,O++)
		{
			if (y==j) O++;
			B[i][j] = A[M][O];
		}
	}
	answer = pow(-1.0, x+y)*Determinant(B, N-1);

	del <double> (B, N-1);
	return answer;
}

#endif
