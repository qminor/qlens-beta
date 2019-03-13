#include <cstdio>
#include <iomanip>
#include <cmath>
#include <iostream>
#include "GregsMathHdr.h"
#include "errors.h"
#include <iostream>
#include <fstream>

using namespace std;

inline void SHFT(double &a, double &b, double &c, double d) {a=b;b=c;c=d;}
inline double SIGN(const double a, const double b) {return (b) >= 0.0 ? fabs(a) : -fabs(a);}
inline double MAX(const double a, const double b){return b > a ? (b) : (a);}
inline void SWAP(double &a, double &b) {double dum=a; a=b; b=dum;}
inline void mov3(double &a, double &b, double &c, const double d, const double e, const double f){a=d; b=e; c=f;}
template<class T>
inline const T SQR(const T a) {return a*a;}

const double EPS = 1.0e-14;

double *SolveTriDiagMatrix(double *a, double *b, double *c, double *u, int N)
{
    double *ans = new double[N];
    double *temp = new double[N];
    double det;
    int i;

    ans[0] = u[0]/(det = b[0]);
    for (i = 1; i < N; i++)
    {
        temp[i] = c[i - 1]/det;
        det = b[i] - a[i]*temp[i];
        ans[i] = (u[i] - a[i]*ans[i-1])/det;
    }
    for(i = N-2; i>=0; i--)
          ans[i] -= temp[i+1]*ans[i+1];

    delete[] temp;

    return ans;
}

double *FindSplineMs(double *x, double *y, int N)
{
    double *a = new double[N-2];
    double *b = new double[N-2];
    double *c = new double[N-2];
    double *u = new double[N-2];
    double *m = new double[N];
    double *temp;
    m[0] = m[N-1] = 0;

    b[0] = 2.0 * (x [2] - x [0]);
    c[0] = x[2] - x[1];
    u[0] = 6.0*((y[2] - y[1])/(x[2] - x[1])-(y[1]-y[0])/(x[1]-x[0]));
    for (int i = 1; i < N - 3; i++)
    {
		a[i] = x[i+1] - x[i];
		b[i] = 2.0*(x[i+2] - x[i]);
		c[i] = x[i+2] - x[i+1];
		u[i] = 6.0*((y[i+2] - y[i+1])/(x[i+2] - x[i+1])-(y[i+1]-y[i])/(x[i+1]-x[i]));
    }
    a[N-3] = x[N-2] - x[N-3];
    b[N-3] = 2.0*(x[N-1] - x[N-3]);
    u[N-3] = 6.0*((y[N-1] - y[N-2])/(x[N-1] - x[N-2])-(y[N-2]-y[N-3])/(x[N-2]-x[N-3]));

    temp = SolveTriDiagMatrix(a, b, c, u, N - 2);
    for (int i = 1; i < N-1; i++)
        m[i] = temp [i-1];

    delete[] temp;
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] u;

    return m;
}

double *FindSplineMs(double *x, double **y, int pos, int N)
{
     double *a = new double[N-2];
     double *b = new double[N-2];
     double *c = new double[N-2];
     double *u = new double[N-2];
     double *m = new double[N];
     double *temp;
     m[0] = m[N-1] = 0;

     b[0] = 2.0 * (x [2] - x [0]);
     c[0] = x[2] - x[1];
     u[0] = 6.0*((y[2][pos] - y[1][pos])/(x[2] - x[1])-(y[1][pos]-y[0][pos])/(x[1]-x[0]));
     for (int i = 1; i < N - 3; i++)
     {
		a[i] = x[i+1] - x[i];
		b[i] = 2.0*(x[i+2] - x[i]);
		c[i] = x[i+2] - x[i+1];
		u[i] = 6.0*((y[i+2][pos] - y[i+1][pos])/(x[i+2] - x[i+1])-(y[i+1][pos]-y[i][pos])/(x[i+1]-x[i]));
     }
     a[N-3] = x[N-2] - x[N-3];
     b[N-3] = 2.0*(x[N-1] - x[N-3]);
     u[N-3] = 6.0*((y[N-1][pos] - y[N-2][pos])/(x[N-1] - x[N-2])-(y[N-2][pos]-y[N-3][pos])/(x[N-2]-x[N-3]));

     temp = SolveTriDiagMatrix(a, b, c, u, N - 2);
     for (int i = 1; i < N-1; i++)
	  m[i] = temp [i-1];

     delete[] temp;
     delete[] a;
     delete[] b;
     delete[] c;
     delete[] u;

     return m;
}

double *FindSplineMs(double **x, double **y, int xpos, int ypos, int N)
{
     double *a = new double[N-2];
     double *b = new double[N-2];
     double *c = new double[N-2];
     double *u = new double[N-2];
     double *m = new double[N];
     double *temp;
     m[0] = m[N-1] = 0;

     b[0] = 2.0 * (x[2][xpos] - x[0][xpos]);
     c[0] = x[2][xpos] - x[1][xpos];
     u[0] = 6.0*((y[2][ypos] - y[1][ypos])/(x[2][xpos] - x[1][xpos])-(y[1][ypos]-y[0][ypos])/(x[1][xpos]-x[0][xpos]));
     for (int i = 1; i < N - 3; i++)
     {
		a[i] = x[i+1][xpos] - x[i][xpos];
		b[i] = 2.0*(x[i+2][xpos] - x[i][xpos]);
		c[i] = x[i+2][xpos] - x[i+1][xpos];
		u[i] = 6.0*((y[i+2][ypos] - y[i+1][ypos])/(x[i+2][xpos] - x[i+1][xpos])-(y[i+1][ypos]-y[i][ypos])/(x[i+1][xpos]-x[i][xpos]));
     }
     a[N-3] = x[N-2][xpos] - x[N-3][xpos];
     b[N-3] = 2.0*(x[N-1][xpos] - x[N-3][xpos]);
     u[N-3] = 6.0*((y[N-1][ypos] - y[N-2][ypos])/(x[N-1][xpos] - x[N-2][xpos])-(y[N-2][ypos]-y[N-3][ypos])/(x[N-2][xpos]-x[N-3][xpos]));

     temp = SolveTriDiagMatrix(a, b, c, u, N - 2);
     for (int i = 1; i < N-1; i++)
		m[i] = temp [i-1];

     delete[] temp;
     delete[] a;
     delete[] b;
     delete[] c;
     delete[] u;

     return m;
}

double Gamma(const double x)
{
     static const double cof[6] ={76.18009172947146, -86.50532032941677, 24.01409824083091,
                              -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};
     double temp = 1.000000000190015;
     
     if (x < 0.0)
	  return M_PI/Gamma(1.0-x)/sin(x*M_PI);
     else
     {
		  for (int i = 0; i < 6; i++)
				temp += cof[i]/(x + i + 1);
		  return SQ2PI*pow(x + 5.5, x + 0.5)*exp(-(x + 5.5))*temp/x;
     }
}

double Beta(const double x, const double y)
{
	return Gamma(x)*Gamma(y)/Gamma(x+y);
}

double DiGamma(const double x)
{
	static double c = 8.5;
	static double euler_mascheroni = 0.57721566490153286060;
	double r;
	double value;
	double x2;
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

double G_Function(const double b, const double c, const double x)  // This is the limit (2F1(a,b,c;x) - 1)/a as a --> 0
{
	int j=1;
	double ans=0, fac=1;
	static double tol = 1e-12;
	do {
		fac *= ((b+j-1)/(c+j-1)) * x;
		ans += fac/j;
	} while (fac > tol*ans*j++);
	//cout << "Number of iterations: " << j << endl;
	return ans;
}

/*
// The following functions appear to be normalized differently by a factor of sqrt(M_PI)...check this later!
double IncGamma(const double a, const double x)
{
     double ga = Gamma(a);
     double result = 0.0, gs = ga, prev;
     //double expnx = exp(-x);
     double b = a;
     int i = 0;
     
     if (x == 0.0)
          return 0.0;
//     if (expnx == 0.0)
      if (exp(a*log(x) - x) < 1.0e-30)
 	{
 	  return ga;
 	}
     do
     {
	  gs *= b;
	  b++;
	  prev = result;
	  //result += ga/gs*pow(x, i + a);
	  result += exp(log(ga/gs) + (i+a)*log(x) - x);
	  i++;//cout << (gs) << endl;
     }
     while((fabs(result - prev)/result) != 0.0);
     
     return result;
}

double IncGammaUp(const double a, const double x)
{
	int i = 0;
	double an,b,c,d,del,h;
	const double FPMIN = 1.0e-30;

	b=x+1.0-a;
	c=1.0/FPMIN;
	d=1.0/b;
	h=d;

	do
	{
		i++;
		an = -i*(i-a);
		b += 2.0;
		d=an*d+b;
		if (fabs(d) < FPMIN) d=FPMIN;
		c=b+an/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d=1.0/d;
		del=d*c;
		h *= del;
	}
	while (fabs(del-1.0) > EPS);
	
	return exp(-x+a*log(x))*h;
}
*/

double gammln(const double xx)
{
	double x,y,tmp,ser;
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

void gser(double &gamser, const double a, const double x, double &gln)
{
	const int ITMAX = 100;
	const double EPS = 3.0e-7;
	double sum,del,ap;

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

void gcf(double &gammcf, const double a, const double x, double &gln)
{
	const int ITMAX = 100;
	const double EPS = 3.0e-7;
	const double FPMIN = 1.0e-30;
	int i;
	double an,b,c,d,del,h;

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



double IncGamma(const double a, const double x)
{
	double gamser,gammcf,gln;

	if ((x < 0.0) or (a <= 0.0)) die("Invalid arguments in routine gammp");
	if (x < (a+1.0)) {
		gser(gamser,a,x,gln);
		return gamser;
	} else {
		gcf(gammcf,a,x,gln);
		return 1.0-gammcf;
	}
}

double IncGammaP_and_Gamma(const double a, const double x, double& GammaP, double& gam)
{
	double gln;

	if ((x < 0.0) or (a <= 0.0)) {
		if (x < 0.0) die("x < 0 in routine gammp (x=%g, a=%g)",x,a);
		if (a <= 0.0) die("a <= 0 in routine gammp (x=%g, a=%g)",x,a);
	}
	if (x < (a+1.0)) {
		double gamser;
		gser(gamser,a,x,gln);
		GammaP = gamser;
	} else {
		double gammcf;
		gcf(gammcf,a,x,gln);
		GammaP = 1.0-gammcf;
	}
	gam = exp(gln);
}

double IncGammaUp(const double a, const double x)
{
	double gln;

	if ((x < 0.0) or (a <= 0.0)) die("Invalid arguments in routine gammq");
	if (x < (a+1.0)) {
		double gamser;
		gser(gamser,a,x,gln);
		return 1.0-gamser;
	} else {
		double gammcf;
		gcf(gammcf,a,x,gln);
		return gammcf;
	}
}

double erffc(const double x)
{
	return x < 0.0 ? 1.0+IncGamma(0.5,x*x) : IncGammaUp(0.5,x*x);
}

double erff(const double x)
{
	return x < 0.0 ? -IncGamma(0.5,x*x) : IncGamma(0.5,x*x);
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

double erfinv(double x)
{
  double x2, r, y;
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

void gaussj(double **a, int n, double **b, int m)
{
	int *indxc,*indxr,*ipiv;
	int i,icol=0,irow=0,j,k,l,ll;
	double big,dum,pivinv;

	indxc=new int[n];
	indxr=new int[n];
	ipiv=new int[n];
	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) {
		big=0.0;
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) {
					if (ipiv[k] == 0) {
						if (fabs(a[j][k]) >= big) {
							big=fabs(a[j][k]);
							irow=j;
							icol=k;
						}
					}
				}
				++(ipiv[icol]);
				if (irow != icol) {
					for (l=0;l<n;l++) SWAP(a[irow][l],a[icol][l]);
					for (l=0;l<m;l++) SWAP(b[irow][l],b[icol][l]);
				}
				indxr[i]=irow;
				indxc[i]=icol;
				//if (a[icol][icol] == 0.0) cout << "gaussj: Singular Matrix-2" << endl;
				pivinv=1.0/a[icol][icol];
				a[icol][icol]=1.0;
				for (l=0;l<n;l++) a[icol][l] *= pivinv;
				for (l=0;l<m;l++) b[icol][l] *= pivinv;
				for (ll=0;ll<n;ll++)
					if (ll != icol) {
					dum=a[ll][icol];
					a[ll][icol]=0.0;
					for (l=0;l<n;l++) a[ll][l] -= a[icol][l]*dum;
					for (l=0;l<m;l++) b[ll][l] -= b[icol][l]*dum;
					}
	}
	for (l=n-1;l>=0;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				SWAP(a[k][indxr[l]],a[k][indxc[l]]);
	}
	delete[] ipiv;
	delete[] indxr;
	delete[] indxc;
}

double Determinant (double **A, int N)
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

double Minor(double **A, int x, int y, int N)
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

LevenMarq::LevenMarq(const double tolin, const int Ntotin) : Ntot(Ntotin), tol(tolin)
{
	beta = atry = da = a = NULL;
	alpha = covar = oneda = NULL;
}

void LevenMarq::FindCof(double *ain)
{
	int i, j;
	for (i = 0; i < ma; i++)
	{
		for (j = 0; j < ma; j++)
			alpha[i][j] = 0.0;
		beta[i] = 0.0;
	}
	chisq=(this->*findCof)(ain, beta, alpha);
	return;
}

void LevenMarq::Calc()
{
	int j,k,l;

	if (alamda < 0.0) {
		alamda=0.001;
		FindCof(a);
		ochisq=chisq;
		for (j=0;j<ma;j++) atry[j]=a[j];
	}
	for (j=0;j<ma;j++) {
		for (k=0;k<ma;k++) covar[j][k]=alpha[j][k];
		covar[j][j]=alpha[j][j]*(1.0+alamda);
		oneda[j][0]=beta[j];	
	}
	gaussj(covar,ma,oneda,1);
	for (j=0;j<ma;j++) da[j]=oneda[j][0];
	if (alamda == 0.0) {
		return;
	}
	for (j=0,l=0;l<ma;l++)
	{
		atry[l]=a[l]+da[j++];
	}
	FindCof(atry);
	if (chisq < ochisq) {
		alamda *= 0.1;
		ochisq=chisq;
		for (j=0;j<ma;j++) {
			for (k=0;k<ma;k++) alpha[j][k]=covar[j][k];
			beta[j]=da[j];
		}
		for (l=0;l<ma;l++) a[l]=atry[l];
		accepted = true;
	} else {
		alamda *= 10.0;
		chisq=ochisq;
		accepted = false;
	}
	
	return;
}

void LevenMarq::LMFindMin(double *a0, const int main, double (LevenMarq::*fin)(double *, double *, double **))
{
	a = a0;
	ma = main;
	alpha = matrix <double> (ma, ma);
	covar = matrix <double> (ma, ma);
	beta = matrix <double> (ma);
	atry = matrix <double> (ma);
	da = matrix <double> (ma);
	oneda = matrix <double> (ma, 1);
	findCof = fin;
	double chi2old;
	alamda = -1.0;
	Calc();
	int count=0;

	do
	{
		chi2old = chisq;
		Calc();
		if (alamda > 1e100)
			break;
		if (accepted)
		{
			if ((chi2old - chisq)/chisq < tol)
				count++;
		}
		else
		{
			if (count != 0)
				count = 0;
		}
	}
	while(count != Ntot);

	del <double> (alpha, ma);
	del <double> (covar, ma);
	del <double> (beta);
	del <double> (atry);
	del <double> (da);
	del <double> (oneda, ma);
}

Derivative::Derivative(const double errin, const int tabin) : NTAB(tabin) {}

double Derivative::Ridders(double (Derivative::*func)(double), const double x, const double h)
{
	const double CON=1.4, CON2=(CON*CON);
	const double BIG=1.0e100;
	const double SAFE=2.0;
	int i,j;
	double errt,fac,hh,ans=0.0;
	double a[NTAB][NTAB];
	//double **a = matrix <double> (NTAB, NTAB);

	hh=h;
	a[0][0]=((this->*func)(x+hh)-(this->*func)(x-hh))/(2.0*hh);
	double err=BIG;
	for (i=1;i<NTAB;i++) {
		hh /= CON;
		a[0][i]=((this->*func)(x+hh)-(this->*func)(x-hh))/(2.0*hh);
		fac=CON2;
		for (j=1;j<=i;j++) {
			a[j][i]=(a[j-1][i]*fac-a[j-1][i-1])/(fac-1.0);
			fac=CON2*fac;
			errt=MAX(fabs(a[j][i]-a[j-1][i]),fabs(a[j][i]-a[j-1][i-1]));
			if (errt <= err) {
				err=errt;
				ans=a[j][i];
			}
		}
		if (fabs(a[i][i]-a[i-1][i-1]) >= SAFE*err) break;
	}
	return ans;
}

double Derivative::Ridders(double (Derivative::*func)(double *), double *x, const int in, const double h)
{
	const double CON=1.4, CON2=(CON*CON);
	const double BIG=1.0e100;
	const double SAFE=2.0;
	int i,j;
	double errt,fac,hh,ans=0.0;
	double x0 = x[in];
	double a[NTAB][NTAB];
	//double **a = matrix <double> (NTAB, NTAB);

	hh=h;
	x[in] = x0 + hh;
	a[0][0] = (this->*func)(x);
	x[in] = x0 - hh;
	a[0][0] -= (this->*func)(x);
	a[0][0] /= (2.0*hh);
	double err=BIG;
	for (i=1;i<NTAB;i++) {
		hh /= CON;
		x[in] = x0 + hh;
		a[0][i] = (this->*func)(x);
		x[in] = x0 - hh;
		a[0][i] -= (this->*func)(x);
		a[0][i] /= (2.0*hh);
		fac=CON2;
		for (j=1;j<=i;j++) {
			a[j][i]=(a[j-1][i]*fac-a[j-1][i-1])/(fac-1.0);
			fac=CON2*fac;
			errt=MAX(fabs(a[j][i]-a[j-1][i]),fabs(a[j][i]-a[j-1][i-1]));
			if (errt <= err) {
				err=errt;
				ans=a[j][i];
			}
		}
		if (fabs(a[i][i]-a[i-1][i-1]) >= SAFE*err) break;
	}
	x[in] = x0;
	return ans;
}

double Derivative::Ridders(double (Derivative::*func)(double *, int), double *x, const int in, const int jn, const double h)
{
	const double CON=1.4, CON2=(CON*CON);
	const double BIG=1.0e100;
	const double SAFE=2.0;
	int i,j;
	double errt,fac,hh,ans=0.0;
	double x0 = x[jn];
	double a[NTAB][NTAB];
	//double **a = matrix <double> (NTAB, NTAB);

	hh=h;
	x[jn] = x0 + hh;
	a[0][0] = (this->*func)(x, in);
	x[jn] = x0 - hh;
	a[0][0] -= (this->*func)(x, in);
	a[0][0] /= (2.0*hh);
	double err=BIG;
	for (i=1;i<NTAB;i++) {
		hh /= CON;
		x[jn] = x0 + hh;
		a[0][i] = (this->*func)(x, in);
		x[jn] = x0 - hh;
		a[0][i] -= (this->*func)(x, in);
		a[0][i] /= (2.0*hh);
		fac=CON2;
		for (j=1;j<=i;j++) {
			a[j][i]=(a[j-1][i]*fac-a[j-1][i-1])/(fac-1.0);
			fac=CON2*fac;
			errt=MAX(fabs(a[j][i]-a[j-1][i]),fabs(a[j][i]-a[j-1][i-1]));
			if (errt <= err) {
				err=errt;
				ans=a[j][i];
			}
		}
		if (fabs(a[i][i]-a[i-1][i-1]) >= SAFE*err) break;
	}
	x[jn] = x0;
	return ans;
}

Minimize::Minimize(double tolin, int intin) : tol(tolin), ITMAX(intin) {}

double Minimize::Brent(double ax, double bx, double cx, double (Minimize::*f)(double))
{
	const double CGOLD = 0.3819660;
	const double ZEPS = 1.0e-10;
	int iter;
	double a,b,d=0.0,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
	double e=0.0;

	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx;
	fw=fv=fx=(this->*f)(x);
	for (iter=0;iter<ITMAX;iter++) {
		xm=0.5*(a+b);
		tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
		if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
			xmin=x;
			return fx;
		}
		if (fabs(e) > tol1) {
			r=(x-w)*(fx-fv);
			q=(x-v)*(fx-fw);
			p=(x-v)*q-(x-w)*r;
			q=2.0*(q-r);
			if (q > 0.0) p = -p;
			q=fabs(q);
			etemp=e;
			e=d;
			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
				d=CGOLD*(e=(x >= xm ? a-x : b-x));
			else {
				d=p/q;
				u=x+d;
				if (u-a < tol2 || b-u < tol2)
					d=SIGN(tol1,xm-x);
			}
		} else {
			d=CGOLD*(e=(x >= xm ? a-x : b-x));
		}
		u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
		fu=(this->*f)(u);
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			SHFT(v,w,x,u);
			SHFT(fv,fw,fx,fu);
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				v=w;
				w=u;
				fv=fw;
				fw=fu;
			} else if (fu <= fv || v == x || v == w) {
				v=u;
				fv=fu;
			}
		}
	}
	cout << "Too many iterations in brent" << endl;
	xmin=x;
	return fx;
}

double Minimize::DBrent(const double ax, const double bx, const double cx, double (Minimize::*f)(const double), double (Minimize::*df)(const double))
{
	const int ITMAX=100;
	const double ZEPS=1.0e-10;
	bool ok1,ok2;
	int iter;
	double a,b,d=0.0,d1,d2,du,dv,dw,dx,e=0.0;
	double fu,fv,fw,fx,olde,tol1,tol2,u,u1,u2,v,w,x,xm;

	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx;
	fw=fv=fx=(this->*f)(x);
	dw=dv=dx=(this->*df)(x);
	for (iter=0;iter<ITMAX;iter++) {
		xm=0.5*(a+b);
		tol1=tol*fabs(x)+ZEPS;
		tol2=2.0*tol1;
		if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
			xmin=x;
			return fx;
		}
		if (fabs(e) > tol1) {
			d1=2.0*(b-a);
			d2=d1;
			if (dw != dx) d1=(w-x)*dx/(dx-dw);
			if (dv != dx) d2=(v-x)*dx/(dx-dv);
			u1=x+d1;
			u2=x+d2;
			ok1 = (a-u1)*(u1-b) > 0.0 && dx*d1 <= 0.0;
			ok2 = (a-u2)*(u2-b) > 0.0 && dx*d2 <= 0.0;
			olde=e;
			e=d;
			if (ok1 || ok2) {
				if (ok1 && ok2)
					d=(fabs(d1) < fabs(d2) ? d1 : d2);
				else if (ok1)
					d=d1;
				else
					d=d2;
				if (fabs(d) <= fabs(0.5*olde)) {
					u=x+d;
					if (u-a < tol2 || b-u < tol2)
						d=SIGN(tol1,xm-x);
				} else {
					d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
				}
			} else {
				d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
			}
		} else {
			d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
		}
		if (fabs(d) >= tol1) {
			u=x+d;
			fu=(this->*f)(u);
		} else {
			u=x+SIGN(tol1,d);
			fu=(this->*f)(u);
			if (fu > fx) {
				xmin=x;
				return fx;
			}
		}
		du=(this->*df)(u);
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			mov3(v,fv,dv,w,fw,dw);
			mov3(w,fw,dw,x,fx,dx);
			mov3(x,fx,dx,u,fu,du);
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				mov3(v,fv,dv,w,fw,dw);
				mov3(w,fw,dw,u,fu,du);
			} else if (fu < fv || v == x || v == w) {
				mov3(v,fv,dv,u,fu,du);
			}
		}
	}
	cout << "Too many iterations in routine dbrent" << endl;
	return 0.0;
}

void Minimize::MnBrak(double &ax, double &bx, double &cx, double (Minimize::*func)(double))
{
	const double GOLD=1.618034,GLIMIT=100.0,TINY=1.0e-20;
	double ulim,u,r,q,fu,fa,fb,fc;

	fa=(this->*func)(ax);
	fb=(this->*func)(bx);
	if (fb > fa) {
		SWAP(ax,bx);
		SWAP(fb,fa);
	}
	cx=bx+GOLD*(bx-ax);
	fc=(this->*func)(cx);
	while (fb > fc) {
		r=(bx-ax)*(fb-fc);
		q=(bx-cx)*(fb-fa);
		u=bx-((bx-cx)*q-(bx-ax)*r)/
				(2.0*SIGN(MAX(fabs(q-r),TINY),q-r));
		ulim=bx+GLIMIT*(cx-bx);
		if ((bx-u)*(u-cx) > 0.0) {
			fu=(this->*func)(u);
			if (fu < fc) {
				ax=bx;
				bx=u;
				fa=fb;
				fb=fu;
				return;
			} else if (fu > fb) {
				cx=u;
				fc=fu;
				return;
			}
			u=cx+GOLD*(cx-bx);
			fu=(this->*func)(u);
		} else if ((cx-u)*(u-ulim) > 0.0) {
			fu=(this->*func)(u);
			if (fu < fc) {
				SHFT(bx,cx,u,u+GOLD*(u-cx));
				SHFT(fb,fc,fu,(this->*func)(u));
			}
		} else if ((u-ulim)*(ulim-cx) >= 0.0) {
			u=ulim;
			fu=(this->*func)(u);
		} else {
			u=cx+GOLD*(cx-bx);
			fu=(this->*func)(u);
		}
		SHFT(ax,bx,cx,u);
		SHFT(fa,fb,fc,fu);
	}
}

double Minimize::F1Dim(const double x)
{
	for (int i = 0; i < n; i++)
		xt[i] = p[i]+x*xi[i];
	return (this->*F1func)(xt);
}

double Minimize::DF1Dim(const double x)
{
	double temp = 0.0;
	int i;
	for (i = 0; i < n; i++)
		xt[i] = p[i]+x*xi[i];
	(this->*DF1func)(xt, dxt);
	for (i = 0; i < n; i++)
		temp += xi[i]*dxt[i];
	return temp;
}

double Minimize::Linmin()
{
	int j;
	double xx,bx,ax,fret;

	ax=0.0;
	xx=1.0;
	MnBrak(ax,xx,bx,&Minimize::F1Dim);
	fret=Brent(ax,xx,bx,&Minimize::F1Dim);
	for (j=0;j<n;j++) {
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	return fret;
}

double Minimize::DLinmin()
{
	int j;
	double xx,bx,ax,fret;

	ax=0.0;
	xx=1.0;
	MnBrak(ax,xx,bx,&Minimize::F1Dim);
	fret=DBrent(ax,xx,bx,&Minimize::F1Dim,&Minimize::DF1Dim);
	for (j=0;j<n;j++) {
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	return fret;
}

double Minimize::Frprmn(double *pin, int nin, double (Minimize::*func)(double *), void (Minimize::*dfunc)(double *, double *))
{
	int j,its;
	double gg,gam,fp,dgg, fret;
	double *g,*h;
	const double EPS = 1.0e-10;

	F1func = func;
	n = nin;
	g=new double[n];
	h=new double[n];
	xi=new double[n];
	xt=new double[n];
	p=pin;
	fp=(this->*func)(p);
	(this->*dfunc)(p,xi);
	for (j=0;j<n;j++) {
		g[j] = -xi[j];
		xi[j]=h[j]=g[j];
	}
	
	for (its=0;its<ITMAX;its++) {
		iter=its;
		fret = Linmin();
		if (2.0*fabs(fret-fp) <= tol*(fabs(fret)+fabs(fp)+EPS)) {
			delete[] g;
			delete[] h;
			delete[] xi;
			delete[] xt;
			return fret;
		}
		fp= fret;
		(this->*dfunc)(p,xi);
		dgg=gg=0.0;
		for (j=0;j<n;j++) {
			gg += g[j]*g[j];
			dgg += (xi[j]+g[j])*xi[j];
		}
		if (gg == 0.0) {
			delete[] g;
			delete[] h;
			delete[] xi;
			delete[] xt;
			return fret;
		}
		gam=dgg/gg;
		for (j=0;j<n;j++) {
			g[j] = -xi[j];
			xi[j]=h[j]=g[j]+gam*h[j];
		}
	}
	cout << "Too many iterations in frprmn" << endl;
	
	return 0.0;
}

double Minimize::Frprmnd(double *pin, int nin, double (Minimize::*func)(double *), void (Minimize::*dfunc)(double *, double *))
{
	int j,its;
	double gg,gam,fp,dgg, fret;
	double *g,*h;
	const double EPS = 1.0e-10;

	F1func = func;
	DF1func = dfunc;
	n = nin;
	g=new double[n];
	h=new double[n];
	xi=new double[n];
	xt=new double[n];
	dxt=new double[n];
	p=pin;
	fp=(this->*func)(p);
	(this->*dfunc)(p,xi);
	for (j=0;j<n;j++) {
		g[j] = -xi[j];
		xi[j]=h[j]=g[j];
	}
	
	for (its=0;its<ITMAX;its++) {
		iter=its;
		fret = DLinmin();
		if (2.0*fabs(fret-fp) <= tol*(fabs(fret)+fabs(fp)+EPS)) {
			delete[] g;
			delete[] h;
			delete[] xi;
			delete[] xt;
			delete[] dxt;
			return fret;
		}
		fp= fret;
		(this->*dfunc)(p,xi);
		dgg=gg=0.0;
		for (j=0;j<n;j++) {
			gg += g[j]*g[j];
			dgg += (xi[j]+g[j])*xi[j];
		}
		if (gg == 0.0) {
			delete[] g;
			delete[] h;
			delete[] xi;
			delete[] xt;
			delete[] dxt;
			return fret;
		}
		gam=dgg/gg;
		for (j=0;j<n;j++) {
			g[j] = -xi[j];
			xi[j]=h[j]=g[j]+gam*h[j];
		}
	}
	cout << "Too many iterations in frprmn" << endl;
	
	return 0.0;
}

double Minimize::Powell(double *pin, int nin, double **xia, double (Minimize::*func)(double *))
{
	const double TINY=1.0e-25;
	int i,j,ibig;
	double del,fp,fptt,t,fret;

	F1func = func;
	n = nin;
	double *pt = new double[n];
	double *ptt = new double[n];
	xi = new double[n];
	xt=new double[n];
	p = pin;
	fret=(this->*func)(p);
	for (j=0;j<n;j++) pt[j]=p[j];
	for (iter=0;;++iter) {
		fp=fret;
		ibig=0;
		del=0.0;
		for (i=0;i<n;i++) {
			for (j=0;j<n;j++) xi[j]=xia[j][i];
			fptt=fret;
			fret = Linmin();
			if (fptt-fret > del) {
				del=fptt-fret;
				ibig=i+1;
			}
		}
		if (2.0*(fp-fret) <= tol*(fabs(fp)+fabs(fret))+TINY) 
		{
			delete[] pt;
			delete[] ptt;
			delete[] xi;
			delete[] xt;
			return fret;
		}
		if (iter == ITMAX) cout << "powell exceeding maximum iterations." << endl;
		for (j=0;j<n;j++) {
			ptt[j]=2.0*p[j]-pt[j];
			xi[j]=p[j]-pt[j];
			pt[j]=p[j];
		}
		fptt=(this->*func)(ptt);
		if (fptt < fp) {
			t=2.0*(fp-2.0*fret+fptt)*SQR(fp-fret-del)-del*SQR(fp-fptt);
			if (t < 0.0) {
				fret = Linmin();
				for (j=0;j<n;j++) {
					xia[j][ibig-1]=xia[j][n-1];
					xia[j][n-1]=xi[j];
				}
			}
		}
	}
}

Minimize::~Minimize(){}

