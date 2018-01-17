#ifndef GREGSMATHHDR_H
#define GREGSMATHHDR_H
#include <cmath>
#include <iostream>
using namespace std;
const double SQ2PI = 2.5066282746310005;
const double SQRT2PI = 2.5066282746310005;
const double PI = 3.14159265358979323846;
const double RT4PI_INV = 0.7511255444649425;
const double SM = 1.989e33;
const double PC = 3.0856e18;
const double G = 6.6720e-8;
const double mpcToKm = 3.0856e19;
const double SQRT2 = 1.414213562373095048801688;

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
T ***matrix(const int xN, const int yN, const int zN)
{
     T ***temp = new T**[xN];
     
     for (int i = 0; i < xN; i++)
     {
	  temp[i] = new T*[yN];
	  for (int j = 0; j < yN; j++)
	       temp[i][j] = new T[zN];
     }
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
T ***matrix(const int xN, const int yN, const int zN, T in)
{
	T ***temp = new T**[xN];
     
	for (int i = 0; i < xN; i++)
	{
		temp[i] = new T*[yN];
		for (int j = 0; j < yN; j++)
		{
			temp[i][j] = new T[zN];
			for (int k = 0; k < zN; k++)
				temp[i][j][k] = in;
		}
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
		T ***matrix(const int xN, const int yN, const int zN, T ***in)
{
	T ***temp = new T**[xN];
     
	for (int i = 0; i < xN; i++)
	{
		temp[i] = new T*[yN];
		for (int j = 0; j < yN; j++)
		{
			temp[i][j] = new T[zN];
			for (int k = 0; k < zN; k++)
				temp[i][j][k] = in[i][j][k];
		}
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
     for (int i = 0; i < xN; i++)
	  delete[] temp[i];
     delete[] temp;
}

template <typename T>
void del(T ***temp, int xN, int yN)
{
     for (int i = 0; i < xN; i++)
     {
		  for (int j = 0; j < yN; j++)
				 delete[] temp[i][j];
		  delete[] temp[i];
     }
     
     delete[] temp;
}

template <typename T>
T IfFlagUp(T flag, int N)
{
     T i = 1 << N;
     
     return flag & i;
}

template <typename T>
T FlagUp(T flag, int N)
{
     T i = 1 << N;
     
     return i | flag;
}

template <typename T>
T IfFlagDown(T flag, int N)
{
     T i = 1 << N;
     
     return ~(flag | (~i));
}

template <typename T>
T FlagDown(T flag, int N)
{
     T i = 1 << N;
     
     return flag&(~i);
}

template <typename T>
T FlagChange(T flag, int N)
{
     T i = 1 << N;
     
     return flag ^ i;
}

class LevenMarq
{
	private:
		double chisq;
		double ochisq;
		double alamda;
		double **alpha;
		double **covar;
		double *beta;
		double *a;
		double *atry;
		double *da;
		double **oneda;
		double *eigVal;
		double **eigVec;
		int ma;
		int Ntot;
		double tol;
		bool accepted;
		double (LevenMarq::*findCof)(double *, double *, double **);
		
	public:
		LevenMarq(const double, const int);
		void LMFindMin(double *, const int, double (LevenMarq::*)(double *, double *, double **));
		void Calc();
		void FindCof(double *ain);
};

class Derivative
{
	private:
		int NTAB;
		//double err;
	public:
		Derivative(double, int);
		double Ridders(double (Derivative::*f)(double), double, double);
		double Ridders(double (Derivative::*f)(double *), double *, int, double);
		double Ridders(double (Derivative::*f)(double *, int), double *, int, int, double);
};

class Minimize
{
	private:
		double tol;
		int ITMAX;
		double xmin;
		int iter;
		
		double *p;
		double *xi;
		int n ;
		double (Minimize::*F1func)(double *);
		void (Minimize::*DF1func)(double *, double *);
		double *xt;
		double *dxt;
		
	public:
		Minimize(double, int);
		double Brent(const double, const double, const double, double (Minimize::*)(const double));
		double DBrent(const double, const double, const double, double (Minimize::*f)(const double), double (Minimize::*df)(const double));
		void MnBrak(double &, double &, double &, double (Minimize::*)(double));
		double Min(){return xmin;}
		double F1Dim(const double);
		double DF1Dim(const double);
		double Linmin();
		double DLinmin();
		double Frprmn(double *, int, double (Minimize::*)(double *), void (Minimize::*)(double *, double *));
		double Frprmnd(double *, int, double (Minimize::*)(double *), void (Minimize::*)(double *, double *));
		double Powell(double *, int, double **, double (Minimize::*)(double *));
		~Minimize();
};

class GammaFunction
{
     private:
          double *GammaCs;
          double GammaAConst;
          
     public:
          GammaFunction(const double);
          double Output(const double);
          double Error(const double);
          ~GammaFunction();
};

double Beta(const double, const double);
double BetaInc(const double, const double, const double);
double Gamma(const double);
double DiGamma(const double);
double G_Function(const double x, const double b, const double c);  // This is the limit (2F1(a,b,c;x) - 1)/a as a --> 0
double Beta(const double x, const double y);
double IncGamma(const double, const double);
double IncGammaUp(const double, const double);
double gammln(const double xx);
void gser(double &gamser, const double a, const double x, double &gln);
void gcf(double &gammcf, const double a, const double x, double &gln);
double erfinv(double x);
double erffc(const double x);
double erff(const double x);

double Factorial(const int);
void gaussj(double **, int, double **, int);
double *SolveTriDiagMatrix(double *, double *, double *, double *, int);
double *FindSplineMs(double *, double *, int);
double *FindSplineMs(double *, double **, int, int);
double Minor(double **, int, int, int);
double Determinant(double **, int);

#endif
