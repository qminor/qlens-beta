#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <exception>
#include <csignal>
#include <string>
#include "GregsMathHdr.h"
#include "mcmchdr.h"
#include "random.h"
#include "errors.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

#define LOGLIKE(x) (this->*LogLikePtr)(x)

template<class T>
inline const T SQR(const T a) {return a*a;}
const string blank = string("                    \033[20D");
const string bblank = string("                              \033[30D");

class CovPoints
{
	private:
		int count;
		int num;
		int ma;
		double **data;

	public:
		CovPoints(const int numin, const int main) : count(0), num(numin), ma(main)
		{
			data = matrix <double> (num, ma);
		}
		CovPoints &operator = (double *ain)
		{
			for (int i = 0; i < ma; i++)
			{
				data[count][i] = ain[i];
			}
			count++;
			count %= num;
			return *this;
		}
		double operator [] (const int i)
		{
			return data[count][i];
		}
		~CovPoints()
		{
			del <double> (data, num);
		}
};

UCMC::UCMC() : Minimize(1.0e-6, 1000), LevenMarq(0.001, 3), Derivative(1.0e-6, 10)
{
	rand = 100;
	cvar = NULL;
	a = NULL;
	upperLimits = NULL;
	lowerLimits = NULL;
	LogLikePtr = &UCMC::LogLike;
	DerivedParamPtr = NULL;
	NDerivedParams = 0;
	dparam_list = NULL;
	mpi_id = 0;
	mpi_np = 1;
	mpi_ngroups = 1;
	mpi_group_id = mpi_group_num = 0;
	mpi_group_leader = NULL;
}

#ifdef USE_MPI
// use the following if each MPI process will perform a separate likelihood evaluation
void UCMC::Set_MCMC_MPI(const int mpi_np_in, const int mpi_id_in)
{
	mpi_np = mpi_np_in;
	mpi_id = mpi_id_in;
	mpi_ngroups = mpi_np;
	mpi_group_num = mpi_id;
	if (mpi_group_leader != NULL) delete[] mpi_group_leader;
	mpi_group_leader = new int[mpi_np];
	for (int i=0; i < mpi_np; i++) mpi_group_leader[i] = i;
}

// use the following if groups of MPI processes will perform separate likelihood evaluations
// (this is useful if each likelihood evaluation is MPI'd over the processes within a group, using sub-communicators)
void UCMC::Set_MCMC_MPI(const int mpi_np_in, const int mpi_id_in, const int mpi_ngroups_in, const int mpi_group_num_in, int *mpi_group_leader_in)
{
	mpi_np = mpi_np_in;
	mpi_id = mpi_id_in;
	mpi_ngroups = mpi_ngroups_in;
	mpi_group_num = mpi_group_num_in;
	if (mpi_group_leader != NULL) delete[] mpi_group_leader;
	mpi_group_leader = new int[mpi_ngroups];
	for (int i=0; i < mpi_ngroups; i++) mpi_group_leader[i] = mpi_group_leader_in[i];
}
#endif

void UCMC::SetNDerivedParams(const int nder)
{
	NDerivedParams = nder;
	if (dparam_list != NULL) delete[] dparam_list;
	dparam_list = new double[NDerivedParams];
}

double UCMC::LogLike(double *ain) {return 0.0;}

double UCMC::LogPrior(double *ain) {return 0.0;}

double UCMC::DLogLike(double *a, const int i)
{
	double h = 0.002;
	double tempp = Ridders(static_cast<double (Derivative::*)(double *)> (&UCMC::LogLike), a, i, h);
	return tempp;
}

double UCMC::DDLogLike(double *a, const int i, const int j)
{
	double h = 0.002;
	double tempp = Ridders(static_cast<double (Derivative::*)(double *, int)> (&UCMC::DLogLike), a, i, j, h);
	return tempp;
}

void UCMC::InputPoint(double *a0, double *ul, double *ll, int ain)
{
	ma = ain;
	a = matrix <double> (ma);
	upperLimits = matrix <double> (ma);
	lowerLimits = matrix <double> (ma);
	upperLimits_initial = matrix <double> (ma);
	lowerLimits_initial = matrix <double> (ma);
	cvar = matrix <double> (ma, ma, 0.0);
	for (int i = 0; i < ma; i++)
	{
		a[i] = a0[i];
		upperLimits[i] = ul[i];
		lowerLimits[i] = ll[i];
		upperLimits_initial[i] = ul[i];
		lowerLimits_initial[i] = ll[i];

		cvar[i][i] = SQR(ul[i] - ll[i])/3.0;
		//cvar[i][i] = SQR((ul[i] - ll[i])/10000);
	}
	
	return;
}

void UCMC::InputPoint(double *a0, double *ul, double *ll, double *ul0, double *ll0, int ain)
{
	ma = ain;
	a = matrix <double> (ma);
	upperLimits = matrix <double> (ma);
	lowerLimits = matrix <double> (ma);
	upperLimits_initial = matrix <double> (ma);
	lowerLimits_initial = matrix <double> (ma);
	cvar = matrix <double> (ma, ma, 0.0);
	for (int i = 0; i < ma; i++)
	{
		a[i] = a0[i];
		upperLimits[i] = ul[i];
		lowerLimits[i] = ll[i];
		upperLimits_initial[i] = ul0[i];
		lowerLimits_initial[i] = ll0[i];

		cvar[i][i] = SQR(ul[i] - ll[i])/3.0;
		//cvar[i][i] = SQR((ul[i] - ll[i])/10000);
	}
	
	return;
}

void UCMC::InputPoint(double *ul, double *ll, int ain)
{
	ma = ain;
	a = matrix <double> (ma);
	upperLimits = matrix <double> (ma);
	lowerLimits = matrix <double> (ma);
	cvar = matrix <double> (ma, ma, 0.0);
	for (int i = 0; i < ma; i++)
	{
		a[i] = (ul[i] + ll[i])/2.0;
		cvar[i][i] = SQR(ul[i] - ll[i])/3.0;
		upperLimits[i] = ul[i];
		lowerLimits[i] = ll[i];
	}
	
	return;
}

double UCMC::FindCof(double *ain, double *beta, double **alpha)
{
	for (int i = 0; i < ma; i++)
	{
		if(ain[i] >= upperLimits[i])
		{
			ain[i] = upperLimits[i];
		}
		else if(ain[i] <= lowerLimits[i])
		{
			ain[i] = lowerLimits[i];
		}
	}
	
	double chisq=2.0*LOGLIKE(ain);
	
	for (int i = 0; i < ma; i++) 
	{
		beta[i] = -DLogLike(ain, i);
		for (int j = 0; j <= i; j++)
		{
			alpha[i][j] = DDLogLike(ain, i, j);
		}
	}
	
	for (int i=1;i<ma;i++)
		for (int j=0;j<i;j++) 
			alpha[j][i]=alpha[i][j];

	return chisq;
}

void UCMC::CalcDir(double *p, double *d)
{
	for (int i = 0; i < ma; i++)
	{
		if(p[i] >= upperLimits[i])
		{
			p[i] = upperLimits[i];
		}
		else if(p[i] <= lowerLimits[i])
		{
			p[i] = lowerLimits[i];
		}
	}
	for (int i = 0; i < ma; i++)
	{
		d[i] = 2.0*DLogLike(p, i);
	}
}

double UCMC::Chi2(double *ain)
{
	for (int i = 0; i < ma; i++)
	{
		if(ain[i] >= upperLimits[i])
		{
			ain[i] = upperLimits[i];
		}
		else if(ain[i] <= lowerLimits[i])
		{
			ain[i] = lowerLimits[i];
		}
	}
	double temp = 2.0*LOGLIKE(ain);
	return temp;
}

void UCMC::ApproxCovMatrix()
{
	int i, j;
	
	for (i = 0; i < ma; i++)
	{
		for (j = 0; j < ma; j++)
			cvar[i][j] = 0.0;
	}
	
	for (i = 0; i < ma; i++)
	{
		cvar[i][i] = SQR(upperLimits[i] - lowerLimits[i])/3.0;	
	}
}

void UCMC::FindCovMatrix()
{
	cout << "calculating matrix ... " << flush;
	double **alpha = matrix <double>(ma, ma, 0.0);
	double *beta = matrix <double>(ma, 0.0);
	FindCof(a, beta, alpha);
	Cholesky chol(alpha, ma);
	chol.Inverse(cvar);
	del <double> (alpha, ma);
	delete[] beta;
	cout << "done" << endl;
}

void UCMC::SaveCovMatrix(const char *name)
{
	int i, j;
	ofstream out(name);
	
	for (i = 0; i < ma; i++)
	{
		for (j = 0; j < ma; j++)
		{
			out << cvar[i][j] << "   ";
		}
		out << endl;
	}
}

void UCMC::FindCovMatrix(const char *name)
{
	double temp;
	ifstream in(name);
	for (int i = 0; i < ma; i++)
	{
		for (int j = 0; j < ma; j++)
		{
			in >> temp;
			cvar[i][j] = temp;
		}
	}
}

void UCMC::FindCovMatrix(double *b)
{
	for (int i = 0; i < ma; i++)
		for (int j = 0; j < ma; j++)
		{
			cvar[i][j] = i == j ? b[i]*b[i] : 0.0;
		}
}

void UCMC::FindCovMatrix(double **b)
{
	int i, j;
	
	for (i = 0; i < ma; i++)
		for (j = 0; j < ma; j++)
	{
		cvar[i][j] = b[i][j];
	}
}

bool UCMC::checkLimits(double *ain)
{
	for (int i = 0; i < ma; i++)
	{
		if((ain[i] >= upperLimits[i])||(ain[i] <= lowerLimits[i])||(ain[i]*0.0 != 0.0))
		{
			//cout << "limit broken on " << i << endl;
			return false;
		}
	}
	return true;
}

bool UCMC::checkLimitsUni(double *ain)
{
	for (int i = 0; i < ma; i++)
	{
		if((ain[i] >= 1.0)||(ain[i] <= 0.0)||(ain[i]*0.0 != 0.0))
		{
			return false;
		}
	}
	return true;
}

bool UCMC::checkReplaceLimits(double *ain, RandomBasis &gDev)
{
	bool temp = true;
	for (int i = 0; i < ma; i++)
	{
		if(ain[i] > upperLimits[i])
		{
			gDev.Adjust(ain, upperLimits[i], i);
			if (temp)
				temp = false;
		}
		else if (ain[i] < lowerLimits[i])
		{
			gDev.Adjust(ain, lowerLimits[i], i);
			if (temp)
				temp = false;
		}
	}
	return temp;
}

void UCMC::replaceLowerLimits(double *ain, RandomBasis &gDev)
{
	gDev.Adjust(ain, lowerLimits[0], 0);
	for (int i = 1; i < ma; i++)
	{
		if(ain[i] > upperLimits[i])
		{
			gDev.Adjust(ain, upperLimits[i], i);
		}
		else if (ain[i] < lowerLimits[i])
		{
			gDev.Adjust(ain, lowerLimits[i], i);
		}
	}
	return;
}

void UCMC::replaceUpperLimits(double *ain, RandomBasis &gDev)
{
	gDev.Adjust(ain, upperLimits[0], 0);
	for (int i = 1; i < ma; i++)
	{
		if(ain[i] > upperLimits[i])
		{
			gDev.Adjust(ain, upperLimits[i], i);
		}
		else if (ain[i] < lowerLimits[i])
		{
			gDev.Adjust(ain, lowerLimits[i], i);
		}
	}
	return;
}

void UCMC::MetHas(const char *name, int N, const char flag)
{
	ofstream out(name);
	double chisq;
	double *aNext = matrix <double> (ma);
	double ans, chisqnext;
	int mult = 1;
	int count = 0, total = 0;
	int l, k;
	chisq = LOGLIKE(a) + LogPrior(a);
	MultiNormDev gDev(cvar, ma, 2.4, rand);

	cout << "Metropolis-Hastings Algorithm Started\n" << "\tpoints = " << "\n\taccept ratio = " << endl;
	
	do
	{
		do
		{			
			gDev.MultiDev(aNext, a);
		}
		while(!checkLimits(aNext));
		chisqnext = LOGLIKE(aNext) + LogPrior(aNext);

		if (((ans = chisqnext - chisq) <= 0.0)||(gDev.ExpDev() >= ans))
		{
			if (flag&MULTS)
			{
				out << mult << "   ";
				for (int k = 0; k < ma; k++)
				{
					out << a[k] << "   ";
					a[k] = aNext[k];
				}
				out << endl;
			}
			else
			{
				for (l = 0; l < mult; l++)
				{
					for (k = 0; k < ma; k++)
					{
						out << a[k] << "   ";
						a[k] = aNext[k];
					}
					out << endl;
				}
			}
			
			chisq = chisqnext;
			mult = 1;
			count++;
			cout << "\033[2A\tpoints = " << count << "\n\taccept ratio = " << blank << (double)count/(double)total << endl;
		}
		else
		{
			mult++;
		}
		total++;
	}
	while(count < N);

	del <double> (aNext);
}

void UCMC::Barker(const char *name, int N, const char flag)
{
	ofstream out(name);
	double chisq, chisqnext;
	double *aNext = matrix <double> (ma);
	int count = 0, total = 0;
	int k, l;
	int mult = 1;

	chisq = LOGLIKE(a);

	MultiNormDev gDev(cvar, ma, 2.4, rand);
	
	do
	{
		do
		{			
			gDev.MultiDev(aNext, a);
		}
		while(!checkLimits(aNext));

		chisqnext = LOGLIKE(aNext);

		if (gDev.Doub() <= 1.0/(1.0+exp(chisqnext-chisq)))
		{
			if (flag&MULTS)
			{
				out << mult << "   ";
				for (int k = 0; k < ma; k++)
				{
					out << a[k] << "   ";
					a[k] = aNext[k];
				}
				out << endl;
			}
			else
			{
				for (l = 0; l < mult; l++)
				{
					for (k = 0; k < ma; k++)
					{
						out << a[k] << "   ";
						a[k] = aNext[k];
					}
					out << endl;
				}
			}
			
			chisq = chisqnext;
			mult = 1;
			count++;
			cout << "Number = " << count << " rate = " << (double)count/(double)total << endl;
		}
		else
		{
			mult++;
		}
		total++;
	}
	while(count < N);

	del <double> (aNext);
}

inline double PDFRatio(Cholesky &chol, Cholesky &chol0, double *a, double *a0, const double f, const int num)
{
	double **inv = matrix <double> (num, num);
	double **inv0 = matrix <double> (num, num);
	chol.Inverse(inv);
	chol0.Inverse(inv0);
	int i, j;
	double arg = 0.0;
	for (i = 0; i < num; i++)
	{
		for (j = 0; j < num; j++)
		{
			arg += (a[i] - a0[i])*(inv[i][j] - inv0[i][j])*(a[j]-a0[j]);
		}
	}
	del <double> (inv, num);
	del <double> (inv0, num);
	
	return chol0.DetSqrt()*exp(-arg/2.0/f/f*num)/chol.DetSqrt();
}

inline double PDFRatio2(Cholesky &chol, Cholesky &chol0, double *a, double *a0, const double f, const int num)
{
	double **inv = matrix <double> (num, num);
	double **inv0 = matrix <double> (num, num);
	chol.Inverse(inv);
	chol0.Inverse(inv0);
	int i, j;
	double r = 0.0, r0 = 0.0;
	for (i = 0; i < num; i++)
	{
		for (j = 0; j < num; j++)
		{
			r += (a[i] - a0[i])*inv[i][j]*(a[j]-a0[j]);
			r0 += (a[i] - a0[i])*inv0[i][j]*(a[j]-a0[j]);
		}
	}
	r = sqrt(r)/f;
	r0 = sqrt(r0)/f;
	del <double> (inv, num);
	del <double> (inv0, num);
	
	return chol0.DetSqrt()*(1.34*r*exp(-r*r)+0.33*exp(-r))/(1.34*r0*exp(-r0*r0)+0.33*exp(-r0))*pow(r0/r, num-1.0)/chol.DetSqrt();
}

inline double PDFRatio(double **cov, double **cov0, double *a, double *a0, const double f, const int num)
{
	Cholesky chol(cov, num);
	Cholesky chol0(cov0, num);
	int i, j;
	double r = sqrt(chol.Square(a0, a))/f, r0 = sqrt(chol0.Square(a, a0))/f;
	
	return chol0.DetSqrt()*(1.34*r*exp(-r*r)+0.33*exp(-r))/(1.34*r0*exp(-r0*r0)+0.33*exp(-r0))*pow(r0/r, num-1.0)/chol.DetSqrt();
}

inline void TakeCov(double **a, const int dim, const int N, double **cov)
{
	double avg[dim];
	int t, i, j, k, l;
	
	for (i = 0; i < dim; i++)
	{
		avg[i] = 0.0;
		for (j = 0; j < dim; j++)
			cov[i][j] = 0.0;
	}
	
	for (t = 0; t < N; t++)
	{
		for (k = 0; k < dim; k++)
		{
			avg[k] += a[t][k];
			for (l = 0; l < dim; l++)
			{
				cov[k][l] += a[t][k]*a[t][l];
			}
		}
	}
	
	for (k = 0; k < dim; k++)
	{
		avg[k] /= N;
	}

	for (k = 0; k < dim; k++)
	{
		for (l = 0; l < dim; l++)
		{
			cov[k][l] = cov[k][l]/N - avg[k]*avg[l];
		}
	}
}

void UCMC::MetHasAdapt(const char *name, const double tol, const int Threads, const int cut, double *best_fit_params, const char flag)
{
	const int NThreads = Threads > ma ? Threads : 2*ma + 1;
	double chisq;
	double **aNext = matrix <double> (NThreads, ma);
	double **a0 = matrix <double> (NThreads, ma);
	double ans, chisqnext;
	int mult = 1;
	int totN[NThreads];
	int count = 0, totall = 0;
	int i, j, l, k, t;
	MultiNormDev *gDev[NThreads];
	
	double **coVar = matrix <double> (ma, ma), **coVarNext = matrix <double> (ma, ma);
	double *avg = matrix <double> (ma);
	double **covT = matrix <double> (NThreads, ma, 0.0);
	double **avgT = matrix <double> (NThreads, ma, 0.0);
	double *W = matrix <double> (ma, 0.0);
	double *Wb = matrix <double> (ma, 0.0);
	double *BnP = matrix <double> (ma, 0.0);
	double pdf;
	double *avgTot = matrix <double> (ma, 0.0);
	int *total = new int[NThreads];
	bool cont;
	bool contin[NThreads];
	ofstream *out = new ofstream[NThreads];
	double minchisq = 1e30;
	double Bn, R, Ravg;
	int ttotal;
	double davg, dcov;
	
	for (t = 0; t < NThreads; t++)
	{
		stringstream s;
		s << t;
		string end;
		s >> end;
		out[t].open((string(name)+string("_")+end).c_str());
		gDev[t] = new MultiNormDev(cvar, ma, 2.4, pow(rand, t));
		total[t] = 0;
	}

	for (t = 0; t < NThreads; t++)
	{
		totN[t] = 0;
		do for (j = 0; j < ma; j++)
			aNext[t][j] = a0[t][j] = lowerLimits_initial[j] + (gDev[0]->Doub())*(upperLimits_initial[j] - lowerLimits_initial[j]);
		while (0.0*LOGLIKE(a0[t]));
	}
	
	TakeCov(a0, ma, NThreads, coVar);
	cout << "Metropolis-Hastings Algorithm Started\n" << "\tpoints = " << "\n\taccept ratio = " << "\n\tR = "  << endl;
	
	do
	{
		Ravg = 0.0;
		
		t = int(NThreads*gDev[0]->Doub());
		do
		{	
			gDev[t]->MultiDev(coVar, aNext[t], a0[t]);
		}
		while(!checkLimits(aNext[t]));
		
		TakeCov(aNext, ma, NThreads, coVarNext);
		pdf = PDFRatio(coVarNext, coVar, aNext[t], a0[t], 2.4, ma);

		chisq = LOGLIKE(a0[t]) + LogPrior(a0[t]);
		chisqnext = LOGLIKE(aNext[t]) + LogPrior(aNext[t]);
		ans = chisqnext - chisq - log(pdf);
		if ((ans <= 0.0)||(gDev[t]->ExpDev() >= ans))
		{
			if (0.0*ans) cout << "Warning: absurd log-likelihood value obtained (dchisq=" << ans << ")" << endl;
			if (flag&MULTS)
			{
				out[t] << mult << "   ";
				for (k = 0; k < ma; k++)
				{
					out[t] << a0[t][k] << "   ";
					a0[t][k] = aNext[t][k];
				}
				out[t] << "   " << 2.0*chisq << endl;
			}
			else
			{
				for (l = 0; l < mult; l++)
				{
					for (k = 0; k < ma; k++)
					{
						out[t] << a0[t][k] << "   ";
						a0[t][k] = aNext[t][k];
					}
					out[t] << "   " << 2.0*chisq << endl;
				}
			}
			
			for (k = 0; k < ma; k++)
			{
				for (i = 0; i < ma; i++)
				{
					coVar[k][i] = coVarNext[k][i];
				}
			}
			if (chisq < minchisq) {
				minchisq = chisq;
				for (i=0; i < ma; i++) best_fit_params[i] = a0[t][i];
			}
			
			chisq = chisqnext;
			mult = 1;
			count++;
		}
		else
		{
			mult++;
		}
		
		if (total[t] >= cut) for (i = 0; i < ma; i++)
		{
			ttotal = total[t] - cut;
			davg = (a0[t][i]-avgT[t][i])/(ttotal+1.0);
			dcov = (-covT[t][i] + ttotal*(SQR(a0[t][i] - avgT[t][i]))/(ttotal+1.0))/(ttotal+1.0);
			BnP[i] += davg*(2.0*avgT[t][i] + davg);
			avgTot[i] += davg/NThreads;
			covT[t][i] += dcov;
			avgT[t][i] += davg;
			W[i] += dcov/NThreads;
		}
		total[t]++;
		cont = false;
		
		if (totall >= cut) for (i = 0; i < ma; i++)
		{
			Bn = 0;
			for (int ts = 0; ts < NThreads; ts++)
				Bn += covT[ts][i]*total[ts]/double(total[ts]-1.0); //(BnP[i] - (NThreads - 1.0)*SQR(avgTot))
			if (W[i]==0) R=0;
			else R = (W[i] + (1.0 + 1.0/NThreads)*(BnP[i] - (NThreads)*SQR(avgTot[i]))/(NThreads - 1.0))/W[i];//Bn*NThreads;
			if(W[i] <= 0.0 || R >= tol || R <= 0.0)
				cont = true;
			Ravg += R;
		}
		else
			cont = true;
		
		totall++;
		cout << "\033[3A\tpoints = " << count  << "( " << count/double(NThreads) << ")" << "\n\taccept ratio = " << blank << (double)count/(double)totall << "\n\tR = " << blank << Ravg/ma << endl;
	}
	while(cont || count < NThreads*2e4);

	delete[] out;
	del <double> (aNext, NThreads);
	del <double> (a0, NThreads);
	del <double> (covT, NThreads);
	del <double> (avgT, NThreads);
	del <double> (W);
	del <double> (avgTot);
	del <double> (BnP);
	del <double> (coVar, ma);
	del <double> (coVarNext, ma);
}

int KEEP_RUNNING = 1;

void sighandler(int sig)
{
	KEEP_RUNNING = 0;
}

void quitproc(int sig)
{
	exit(0);
}

void UCMC::TWalk(const char *name, const double div, const int proj, const double din, const double alim, const double alimt, const double tol, const int Threads, double *best_fit_params, bool logfile, double** initial_points)
{
	int NThreads = (Threads > ma+1) ? Threads : ma + 2;
	if (NThreads < 5+mpi_ngroups) NThreads = 5 + mpi_ngroups;
	if (NThreads <= proj) NThreads = proj + 1; // it might be ok for NThreads to be equal to proj, I'm not sure
	if (mpi_id==0) cout << "Number of chains for T-Walk algorithm: " << NThreads << endl << endl;
	vector<double> loglike(NThreads);
	vector<double> aNext(ma, 0.0);
	vector<vector<double> > a0 = vector<vector<double> > (NThreads, vector<double>(ma, 0.0));
	double ans, loglikenext;
	vector<int> mult(NThreads, 1);
	vector<int> count(NThreads, 1);
	int i,j,end;
	int t, tt, ttt;
	int total=1, ttotal=0;
	int Nlength=1;
#ifdef USE_OPENMP
	int n_loglikes=0;
	double time0, total_loglike_time=0;
#endif

	vector<vector<double> > covT(NThreads, vector<double>(ma, 0.0));
	vector<vector<double> > avgT(NThreads, vector<double>(ma, 0.0));
	double *W = new double[ma];
	double *avgTot = new double[ma];
	double *atrans = new double[ma];
	for (i=0; i < ma; i++) {
		W[i]=0;
		avgTot[i]=0;
	}
	bool cont;
	double logZ, Ravg = 0.0;
	double Rmax = -1e30;
	double minloglike = 1e30;
	ofstream logout;
#ifdef USE_MPI
	if (mpi_id==0)
	{
#endif
		if (logfile) {
			string log_filename = string(name) + ".twalk.log";
			logout.open(log_filename.c_str());
		}
#ifdef USE_MPI
	}
#endif

#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);

	vector<int> tints(NThreads);
	for (i=0; i < NThreads; i++) tints[i] = i;
	vector<int> talls(2*mpi_ngroups);
	bool leader = false;
	if (mpi_id == mpi_group_leader[mpi_group_num]) leader = true;
#endif

	RandomPlane gDev(proj, ma, din, alim, alimt, rand+mpi_group_num);

	ofstream *out;
	out = new ofstream[NThreads];
	for (t=0; t < NThreads; t++)
	{
		stringstream s,ps;
		s << t;
		string endstring;
		s >> endstring;
#ifdef USE_MPI
		if (leader) {
			if (mpi_ngroups > 1) {
				ps << mpi_group_num;
				string pstring;
				ps >> pstring;
				out[t].open((string(name)+string("_")+endstring+"."+pstring).c_str());
			}
			else out[t].open((string(name)+string("_")+endstring).c_str());
		}
#else
		out[t].open((string(name)+string("_")+endstring).c_str());
#endif
	}

	for (t=0; t < NThreads; t++)
	{
#ifdef USE_MPI
		if (mpi_group_num == 0)
		{
#endif
			if (initial_points==NULL) {
				for (j=0; j < ma; j++) {
					a0[t][j] = gDev.Doub();
				}
			} else {
				for (j=0; j < ma; j++) {
					//if (mpi_id==0) cout << t << " " << j << a0[t][j] << endl;
					a0[t][j] = (initial_points[t][j] - lowerLimits_initial[j]) / (upperLimits_initial[j] - lowerLimits_initial[j]);
				}
			}
#ifdef USE_MPI
		}
		MPI_Bcast (c_ptr(a0[t]), a0[t].size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (mpi_group_num == 0) {
#endif
			for (j=0; j < ma; j++) {
				atrans[j] = lowerLimits_initial[j] + a0[t][j]*(upperLimits_initial[j] - lowerLimits_initial[j]);
			}
			loglike[t] = LOGLIKE(atrans);
			//for (j=0; j < ma; j++) cout << atrans[j] << " ";
			//cout << 2*loglike[t] << endl << flush;
			
#ifdef USE_MPI
		}
#endif
	}

#ifdef USE_MPI
	MPI_Bcast (c_ptr(loglike), loglike.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (mpi_id==0)
	{
#endif
		if (logfile) logout << "Metropolis-Hastings/T-Walk Algorithm Started\n\n";
		cout << "Metropolis-Hastings/T-Walk Algorithm Started\n" << "\tpoints = " << "\n\taccept ratio = " << "\n\tR = "  << endl;
#ifdef USE_MPI
	}
#endif

	double b0, b1, b2;
	b0 = div/2.0;
	b1 = div;
	b2 = (1.0 + div)/2.0;

	int id, cnt, ts;
	int lastcnt=0;
	double ran, davg, dcov;
	double Bn, R;
	do
	{       
#ifdef USE_MPI
		if (mpi_id == 0) 
		{
			j = NThreads;
			for (i=0; i < mpi_ngroups; i++)
			{
				int temp = int((j--)*gDev.Doub());
				talls[i] = tints[temp];
				tints[temp] = tints[j];
				tints[j] = talls[i];
			}
            
			for (i=mpi_ngroups, end=talls.size(); i < end; i++)
			{
				talls[i] = tints[int(j*gDev.Doub())];
			}
		}
		else if (mpi_group_num == 0)
		{
			// the following ensures that all the processes in group 0 will be working together to perform the same
			// likelihood calculation with the same values (otherwise absurd results happen)
			for (i=0; i < mpi_ngroups; i++) gDev.Doub();
			for (i=mpi_ngroups, end=talls.size(); i < end; i++) gDev.Doub();
		}

		MPI_Bcast (c_ptr(talls), talls.size(), MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast (c_ptr(tints), tints.size(), MPI_INT, 0, MPI_COMM_WORLD);

		t = talls[mpi_group_num];
		tt = talls[mpi_group_num + mpi_ngroups];
#else
		t = int(NThreads*gDev.Doub());
		tt = int((NThreads - 1)*gDev.Doub());
		if (tt >= t) tt++;
#endif
		ran = gDev.Doub();
		if (ran < b0)
		{
			logZ = gDev.WalkDev(c_ptr(aNext), c_ptr(a0[t]), c_ptr(a0[tt]));
		}
		else if (ran < b1)
		{
			logZ = gDev.TransDev(c_ptr(aNext), c_ptr(a0[t]), c_ptr(a0[tt]));
		}
		else if (ran < b2)
		{
			vector<vector<double> > temp = a0;
#ifdef USE_MPI
			for (i=0, end=NThreads - mpi_ngroups; i < end; i++)
			{
				temp.push_back(a0[tints[i]]);
			}
#else
			for (i=0, end=a0.size(); i < end; i++)
			{
				if (i != tt)
					temp.push_back(a0[i]);
			}
#endif
			if (!gDev.EnterMat(calcCov(temp)))
			{
				gDev.EnterMat(calcIndent(temp));
			}

			gDev.MultiDev(c_ptr(aNext), c_ptr(a0[t]));
			logZ = 0.0;
		}
		else
		{
			vector<vector<double> > temp;
#ifdef USE_MPI
			for (i=0, end=NThreads - mpi_ngroups; i < end; i++)
			{
				temp.push_back(a0[tints[i]]);
			}
#else
			for (i=0, end=a0.size(); i < end; i++)
			{
				if (i != tt)
					temp.push_back(a0[i]);
			}
#endif
			if (!gDev.EnterMat(calcCov(temp)))
			{
				gDev.EnterMat(calcIndent(temp));
			}

			gDev.MultiDev(c_ptr(aNext), c_ptr(a0[tt]));
			logZ = 0.0;
		}

		if (!notUnit(aNext))
		{
			for (j=0; j < ma; j++) {
				atrans[j] = lowerLimits[j] + aNext[j]*(upperLimits[j] - lowerLimits[j]);
			}
#ifdef USE_OPENMP
			if (mpi_id==0) time0 = omp_get_wtime();
#endif
			loglikenext = LOGLIKE(atrans);
#ifdef USE_OPENMP
			if (mpi_id==0) {
				total_loglike_time += omp_get_wtime() - time0;
				n_loglikes++;
			}
#endif
			ans = loglikenext - loglike[t] - logZ;
			//cout << "rank " << mpi_id << ": a[0]=" << atrans[0] << " loglike=" << loglikenext << " " << ans << endl << flush;

			if ((ans <= 0.0)||(gDev.ExpDev() >= ans))
			{
#ifdef USE_MPI
				if (leader) {
#endif
					out[t] << mult[t] << "   ";
					for (i = 0; i < ma; i++)
					{
						out[t] << atrans[i] << "   ";
					}
					if (NDerivedParams > 0) {
						(this->*DerivedParamPtr)(atrans,dparam_list);
						for (i = 0; i < NDerivedParams; i++) {
							out[t] << dparam_list[i] << "   ";
						}
					}

					//out[t] << "   " << 2.0*loglike[t] << endl;
					out[t] << "   " << 2.0*loglikenext << endl << flush;
#ifdef USE_MPI
				}
#endif

				a0[t] = aNext;
				loglike[t] = loglikenext;
				mult[t] = 0;
				count[t]++;
			}
		}
#ifdef USE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		for (i=0; i < mpi_ngroups; i++)
		{
			id = mpi_group_leader[i];
			MPI_Bcast (c_ptr(a0[talls[i]]), a0[talls[i]].size(), MPI_DOUBLE, id, MPI_COMM_WORLD);
			MPI_Bcast (&loglike[talls[i]], 1, MPI_DOUBLE, id, MPI_COMM_WORLD);
			MPI_Bcast (&mult[talls[i]], 1, MPI_INT, id, MPI_COMM_WORLD);
			MPI_Bcast (&count[talls[i]], 1, MPI_INT, id, MPI_COMM_WORLD);
		}
#endif
		for (i=0; i < NThreads; i++)
			mult[i]++;

		total++;
#ifdef USE_MPI
		if (mpi_id == 0)
		{
#endif
			for (ttt=0; ttt < NThreads; ttt++) {
				if (loglike[ttt] < minloglike) {
					minloglike = loglike[ttt];
					for (i=0; i < ma; i++)
						best_fit_params[i] = lowerLimits[i] + a0[ttt][i]*(upperLimits[i] - lowerLimits[i]);
				}
			}

			cnt = 0;
			for (vector<int>::iterator it = count.begin(); it != count.end(); ++it)
			{
				cnt += *it;
			}

			cont = false;
			if (total%NThreads == 0) //cnt >= cut*NThreads && 
			{
				for (ttt=0; ttt < NThreads; ttt++) {
					for (i=0; i < ma; i++) {
						davg = (a0[ttt][i]-avgT[ttt][i])/(ttotal+1.0);
						dcov = ttotal*davg*davg - covT[ttt][i]/(ttotal+1.0);
						avgTot[i] += davg/NThreads;
						covT[ttt][i] += dcov;
						avgT[ttt][i] += davg;
						W[i] += dcov/NThreads;
					}
				}

				ttotal++;

				Ravg = 0.0;
				Rmax = -1e30;
				for (i = 0; i < ma; i++)
				{
					Bn = 0;
					for (ts = 0; ts < NThreads; ts++)
					{
						Bn += (avgT[ts][i] - avgTot[i])*(avgT[ts][i] - avgTot[i]);
					}
					Bn /= double(NThreads - 1);
											  
					R = 1.0 + double(NThreads + 1)*Bn/W[i]/double(NThreads);
					if (R > Rmax) Rmax = R;
											  
					if(W[i] <= 0.0 || R >= tol || R <= 0.0)
					{
						if (Nlength == 0)
						{
							cont = true;
						}
						else
						{
							cont = false;
							Nlength--;
							covT = vector<vector<double> > (NThreads, vector<double>(ma, 0.0));
							avgT = vector<vector<double> > (NThreads, vector<double>(ma, 0.0));
							for (j=0; j < ma; j++) {
								W[j] = 0.0;
								avgTot[j] = 0.0;
							}
							ttotal++;
						}
					}

					Ravg += R;
				}
			}
			else cont = true;

			if (logfile) {
				if ((cnt % 10 == 0) and (cnt != lastcnt)) {
					logout << "points = " << cnt  << " (" << cnt/double(NThreads) << ")" << " accept ratio=" << (double)cnt/(double)total/(double)mpi_ngroups << " R=" << Ravg/ma << " Rmax=" << Rmax;
#ifdef USE_OPENMP
					logout << "   avg_loglike_time = " << total_loglike_time / n_loglikes << " total_time = " << total_loglike_time << endl << flush;
#else
					logout << endl << flush;
#endif
					lastcnt = cnt;
				}
			}
			cout << "\033[3A\tpoints = " << cnt << " (" << cnt/double(NThreads) << ")" << "\n\taccept ratio = " << blank << (double)cnt/(double)total/(double)mpi_ngroups << "\n\tR = " << Ravg/ma << " Rmax=" << Rmax;
#ifdef USE_OPENMP
			cout << "   avg_loglike_time = " << total_loglike_time / n_loglikes << endl << flush;
#else
			cout << endl << flush;
#endif
#ifdef USE_MPI
		}
		MPI_Bcast(best_fit_params,ma,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast (&cont, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
		MPI_Bcast(&KEEP_RUNNING,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
		signal(SIGABRT, &sighandler);
		signal(SIGTERM, &sighandler);
		signal(SIGINT, &sighandler);
		signal(SIGUSR1, &sighandler);
		signal(SIGQUIT, &quitproc);
	}
	while((cont) and (KEEP_RUNNING));

	cout << "twalk for rank " << mpi_id << " has finished." << endl;

	delete[] W;
	delete[] avgTot;
	delete[] atrans;
	return;
}

void UCMC::McmcAd(const char *name, const int N)
{
	ofstream out(name);
	
	double u;
	double chisq;
	double pdf;
	double *aNext = matrix <double> (ma), *a0 = matrix <double> (ma);
	double **coVar = matrix <double> (ma, ma), **coVarNext = matrix <double> (ma, ma);
	double ans, chisqnext;
	double *avg = matrix <double> (ma);
	double avgi, avgj;
	int count = 0, total = 0;
	int i,j;
	int covCount=1000000;
	int covStart = 0;
	
	chisq = LOGLIKE(a);

	for (i = 0; i < ma; i++)
	{
		a0[i] = a[i];
		avg[i] = 0.0;
		for (j = 0; j < ma; j++)
			coVar[i][j] = coVarNext[i][j] = 0.0;
	}
	CovPoints aLast(covCount, ma);
	MultiNormalDev gDev(coVar, 2.381/sqrt(ma), rand, ma);
	MultiNormDev gDev2(cvar, ma, 2.4, rand);
	Cholesky gcDev(coVar, ma);

	do
	{
		
		do
		{			
			if (count > ma+covStart)
				gDev2.MultiDev(coVar, aNext, a0);
			else
				gDev2.MultiDev(aNext, a0);
		}
		while(!checkLimits(aNext));
		
		for (i=0; i < ma; i++)
		{
			for (j=0; j < ma; j++)
			{
				if(count < covCount)
				{
					coVarNext[i][j] = count*(coVar[i][j] + ((aNext[i] - avg[i])*(aNext[j] - avg[j]))/(count+1.0))/(count+1.0);
				}
				else
				{
					avgi = avg[i] + (aNext[i] - aLast[i])/covCount;
					avgj = avg[j] + (aNext[j] - aLast[j])/covCount;
					coVarNext[i][j] = coVar[i][j] + (aNext[i]*aNext[j] - aLast[i]*aLast[j])/covCount + avg[i]*avg[j] - avgi*avgj;
				}
			}
		}
		if (count > ma+1+covStart)
		{
			gcDev.EnterMat(coVarNext);
			pdf = PDFRatio2(gcDev, gDev2, aNext, a0, 2.4, ma);
		}
		else if (count == ma + 1+covStart)
		{
			pdf = 1.0;
		}
		else
			pdf = 1.0;
		chisqnext = LOGLIKE(aNext);
		cout << "pdf = " << count << ", " << ma+1+covStart << "  " << pdf << "\n";
		cout << "chi2n = " << chisqnext << "\n";
		cout << "chi2 = " << chisq << "\n";
		ans = pdf*exp(-(chisqnext - chisq)); 
		ans = (ans*0.0) ? 0.0 : ((ans > 1.0) ? 1.0 : ans);
		
		u = gDev.Doub();
		
		if (u <= ans)
		{
			for (int k = 0; k < ma; k++)
			{
				out << a0[k] << "   ";
				a0[k] = aNext[k];
				if(count < covCount)
					avg[k] += (a0[k]-avg[k])/(count+1.0);
				else
					avg[k] += (aNext[k] - aLast[k])/covCount;
				for (i = 0; i < ma; i++)
				{
					coVar[k][i] = coVarNext[k][i];
				}
			}
			aLast = a0;
			chisq = chisqnext;
			out << endl;
			count++;
		}
		total++;
		cout << ans << "   " << u << " rate = " << (double)count/(double)total << endl;
	}
	while(count < N);
	
	del <double> (coVar, ma);
	del <double> (coVarNext, ma);
	del <double> (a0);
	del <double> (aNext);
}

void UCMC::Slicing(const char *name, const int N, const char flag)
{
	ofstream out(name);
	double *aTemp = new double[ma];
	double w, y, *r = new double[ma], *l = new double[ma];
	double u, logLike, tempLogLike, Navg = 0.0;
	int count = 0, total = 0, i;
	RandomBasis *gDev;
	double (UCMC::*W)(RandomBasis &);
	if (flag&TRANSFORM)
	{
		gDev = new TransformRandomBasis(cvar, ma, rand);
		W = &UCMC::Wt;
	}
	else
	{
		gDev = new RandomBasis(ma, rand);
		W = &UCMC::Wu;
	}
	
	logLike = LOGLIKE(a);
	do
	{
		start:
		total += 2;
		y = logLike + gDev->ExpDev();
		w = (this->*W)(*gDev);
		gDev->RanMult(a, -w*gDev->Doub(), l);
		gDev->RanMult(l, w, r);
		if (checkReplaceLimits(l, *gDev))
		{
			tempLogLike = LOGLIKE(l);
			while((y > tempLogLike))
			{
				gDev->RanMult(l, -w, l);
				if (!checkReplaceLimits(l, *gDev))
					break;
				total++;
				tempLogLike = LOGLIKE(l);
			}
		}
		else 
			total--;
		
		if (checkReplaceLimits(r, *gDev))
		{
			tempLogLike = LOGLIKE(r);
			while((y > tempLogLike))
			{
				gDev->RanMult(r, w, r);
				if (!checkReplaceLimits(r, *gDev))
					break;
				total++;
				tempLogLike = LOGLIKE(r);
			}
		}
		else
			total--;
		
		while(true)
		{
			if (gDev->Mag(r, l) == 0.0)
			{
				gDev++;
				cout << "redo" << endl;
				goto start;
			}

			total++;
			u = gDev->Doub();
			for (i = 0; i < ma; i++)
				aTemp[i] = l[i] + u*(r[i]-l[i]);
			tempLogLike = LOGLIKE(aTemp);
			if (total > 100)
			{
				cout << "overflow" << endl;
			}

			if (y > tempLogLike)
			{
				logLike = tempLogLike;
				break;
			}
			if (gDev->Mag(aTemp, a) < 0.0)
			{
				for (i = 0; i < ma; i++)
					l[i] = aTemp[i];
			}
			else
			{
				for (i = 0; i < ma; i++)
				{
					r[i] = aTemp[i];
				}
			}
		}
		(*gDev)++;

		for (i = 0; i < ma; i++)
		{
			a[i] = aTemp[i];
			out << a[i] << "   ";
		}
 		out << endl;

		count++;
		Navg += (total - Navg)/count;
		cout << "points = " << count << "\nnumber of evals = " << total << "\nnumber averaged:  " << Navg << endl;
		total = 0;
	}
	while(count < N);
	
	delete gDev;
	delete[] aTemp;
	delete[] l;
	delete[] r;
}

void UCMC::SlicingFull(const char *name, const int N)
{
	ofstream out(name);
	double *aTemp = matrix <double> (ma);
	double y, *r = matrix <double> (ma), *l = matrix <double> (ma);
	double logLike, tempLogLike, Navg = 0.0;
	int count = 0, total = 0, i;
	BasicDevs gDev(rand);

	logLike = LOGLIKE(a);
	do
	{
		total = 2;
		y = logLike + gDev.ExpDev();
		
		for (i = 0; i < ma; i++)
		{
			r[i] = upperLimits[i];
			l[i] = lowerLimits[i];
		}
		
		while(true)
		{
			total++;
			for (i = 0; i < ma; i++)
				aTemp[i] = l[i] + gDev.Doub()*(r[i]-l[i]);
			tempLogLike = LOGLIKE(aTemp);
			if (total > 100)
			{
				cout << "overflow = " << total << endl;
			}

			if (y > tempLogLike)
			{
				logLike = tempLogLike;
				break;
			}
			
			for (i = 0; i < ma; i++)
			{
				if (a[i] < aTemp[i])
					r[i] = aTemp[i];
				else if (a[i] > aTemp[i])
					l[i] = aTemp[i];
					
			}
		}
		
		for (i = 0; i < ma; i++)
		{
			a[i] = aTemp[i];
			out << a[i] << "   ";
		}
		out << endl;
		
		count++;
		Navg += (total - Navg)/count;
		cout << "points = " << count << "\nnumber of evals = " << total << "\nnumber averaged:  " << Navg << endl;
	}
	while(count < N);
	
	del <double> (aTemp);
	del <double> (l);
	del <double> (r);
}

void UCMC::FindMinLM()
{
	LMFindMin(a, ma, static_cast <double (LevenMarq::*)(double *, double *, double **)> (&UCMC::FindCof));
}

void UCMC::FindMin()
{
	Frprmn(a, ma, static_cast <double (Minimize::*)(double *)> (&UCMC::Chi2), static_cast <void (Minimize::*)(double *, double *)> (&UCMC::CalcDir));
}

void UCMC::FindMinPow()
{
	double **nd = matrix <double> (ma, ma, 0.0);
	for (int i = 0; i < ma; i++)
		nd[i][i] = 1.0;
	
	Powell(a, ma, nd, static_cast <double (Minimize::*)(double *)> (&UCMC::Chi2));
	del <double> (nd, ma);
}

double UCMC::GridSearch(int iin)
{
	double chi2h, chi2l, chi2last, chi2, stepsize;

	stepsize = 10.0;
	chi2 = iin ? GridSearch(iin-1) : Chi2(a);
	for (int i = 0; i < 2; i++)
	{
		if((iin != 0)&&(iin != 4))
			a[iin] *= exp(stepsize);
		else
			a[iin] += stepsize;
		chi2h = iin ? GridSearch(iin-1) : Chi2(a);
		if ((iin != 0)&&(iin != 4))
			a[iin] /= exp(2.0*stepsize);
		else
			a[iin] -= 2.0*stepsize;
		chi2l = iin ? GridSearch(iin-1) : Chi2(a);
		if ((iin != 0)&&(iin != 4))
			a[iin] *= exp(stepsize);
		else
			a[iin] += stepsize;
		if (chi2l < chi2)
		{
			chi2 = chi2l;
			stepsize *= -1.0;
			if ((iin != 0)&&(iin != 4))
				a[iin] *= exp(stepsize);
			else
				a[iin] += stepsize;
		}
		else if (chi2h < chi2)
		{
			chi2 = chi2h;
			if ((iin != 0)&&(iin != 4))
				a[iin] *= exp(stepsize);
			else
				a[iin] += stepsize;
		}
		else 
		{
			goto here;
		}
		
		do
		{
			chi2last = chi2;
			if ((iin != 0)&&(iin != 4))
				a[iin] *= exp(stepsize);
			else
				a[iin] += stepsize;
			chi2 = iin ? GridSearch(iin-1) : Chi2(a);
		}
		while (chi2last > chi2);
		if ((iin != 0)&&(iin != 4))
			a[iin] /= exp(stepsize);
		else
			a[iin] -= stepsize;
		chi2 = chi2last;
		here:
		
		stepsize /= 10.0;
	}
	return chi2;
}

void UCMC::PrintPoint()
{
	for (int i = 0; i < ma; i++)
		cout << i << " = " << a[i] << endl;
}

UCMC::~UCMC()
{
	if (a != NULL) // just in case the input params weren't set up
	{
		del <double> (a);
		del <double> (upperLimits);
		del <double> (lowerLimits);
		del <double> (upperLimits_initial);
		del <double> (lowerLimits_initial);
		del <double> (cvar, ma);
	}
	if (mpi_group_leader != NULL) delete[] mpi_group_leader;
	if (dparam_list != NULL) delete[] dparam_list;
}

class Fit : private Minimize, private LevenMarq
{
	private:
		double *data;
		int N;
		int jMax;
		double a;
		double param[3];
		double (Minimize::*cPtr)(double *);
		void (Minimize::*dCPtr)(double *, double *);
		double (LevenMarq::*lMPtr)(double *, double *, double **);
		
	public:
		Fit(double *datain, const int Nin) : Minimize(1.0e-9, 1000), LevenMarq(0.0001, 5), N(Nin), jMax(Nin)
		{
			data = new double[N];
			for (int i = 0; i < N; i++)
				data[i] = datain[i];
			cPtr = static_cast <double (Minimize::*)(double *)> (&Fit::Chi2);
			dCPtr = static_cast <void (Minimize::*)(double *, double *)> (&Fit::DChi2);
			lMPtr = static_cast <double (LevenMarq::*)(double *, double *, double **)> (&Fit::Cof);
			param[0] = log(100.0);
			param[1] = log(0.1);
			param[2] = 2.0;
		}
		double Chi2(double *params)
		{
			double temp = 0.0;
			for (int i = 0; i < jMax; i++)
			{
				temp += pow(log(data[i]*(pow(PI*(i+1)/(N+1)/exp(params[1]), params[2]) + 1.0)) - params[0], 2.0);
			}
			return temp;
		}
		void DChi2(double *params, double *dirs)
		{
			double temp = 0.0;
			double k;
			dirs[0] = dirs[1] = dirs[2] = 0.0;
			
			for (int i = 0; i < jMax; i++)
			{
				k = pow(PI*(i+1)/(N+1)/exp(params[1]), params[2]);
				temp = -2.0*(log(data[i]*(k + 1.0)) - params[0]);
				dirs[0] += temp;
				dirs[1] += k*temp*params[2]/(k + 1.0);
				dirs[2] += k ? -k*temp*log(k)/params[2]/(k + 1.0): 0.0;
			}
			return;
		}
		double Cof(double *params, double *beta, double **alpha)
		{
			double dirs[3];
			double ki;
			double chi2 = 0.0;
			double temp;
			int i, j, k;
			for (i = 0; i < jMax; i++)
			{
				ki = pow(PI*(i+1)/(N+1)/exp(params[1]), params[2]);
				temp = (log(data[i]*(ki + 1.0)) - params[0]);
				chi2 += pow(temp, 2.0);
				dirs[0] = 1.0;
				dirs[1] = ki*params[2]/(ki + 1.0);
				dirs[2] = ki ? -ki*log(ki)/params[2]/(ki + 1.0) : 0.0;
				for (j = 0; j < 3; j++)
				{
					beta[j] += temp*dirs[j];
					for (k = 0; k < 3; k++)
					{
						alpha[j][k] += dirs[j]*dirs[k];
					}
				}
			}
			return chi2;
		}
		void FindMins()
		{
			LMFindMin(param, 3, lMPtr);
		}
		double R(){return exp(param[0])/(N+1.0)/2.0;}
		double JStar(){return exp(param[1])*(N+1)/PI;}
		double Output(const double kin){return exp(param[0])/(1.0 + pow(kin/exp(param[1]), param[2]));}
		~Fit()
		{
			delete[] data;
		}
};

inline double dummy(const double x){return x;}
inline double pow10(const double x){return pow(10.0, x);}
inline double pow102(const double x){return pow(10.0, x/2.0);}
inline double unit(const double x){return 1.0;}

class Group : public Cholesky
{
	protected:
	int dim;
	int N;
	int *map;
	int dimR;
	int imax;
	double **covar;
	double *avg;
	double *ulimit;
	double *llimit;
	double area;
	double factor;
	double X0;
	double x0;
	double volume;
	double volumeCorr;
	double E0;
	double enl;
	bool isbad;
	bool min;
	MultiNormDev *random;

	public:
	Group(double **pointsin, const int dimin, const double Xin, const double enl, int Nin, int *mapin, const int dimRin, MultiNormDev *randomin, const char flag) : Cholesky(dimin), dim(dimin), dimR(dimRin), N(Nin), X0(Xin), enl(enl), random(randomin)
	{
		int i, j, k;
		covar = matrix <double> (dim, dim, 0.0);
		avg = matrix <double> (dim, 0.0);
		ulimit = new double[dimR];
		llimit = new double[dimR];
		double *ptr1 = NULL;
		double **temp = pointsin;
		factor = pow(PI, dim/2.0)/Gamma(dim/2.0+1.0);
		//cout << " N = " << N << "   " << dim << flush;
		area = 1.0;
		
		map = new int[dimR];
		if (mapin == NULL)
		{
			for (i = 0; i < dimR; i++)
				map[i] = i;
		}
		else
		{
			for (i = 0; i < dimR; i++)
				map[i] = mapin[i];
		}
		
		for (i = 0; i < Nin; i++)
		{
			ptr1 = *pointsin++;
			for (j = 0; j < dim; j++)
			{
				avg[j] += ptr1[map[j]];
				for (k = 0; k < dim; k++)
				{
					covar[j][k] += ptr1[map[j]]*ptr1[map[k]];
				}
			}
		}
		
		for (i = 0; i < dim; i++)
		{
			avg[i] /= Nin;
		}
		
		for (i = 0; i < dim; i++)
		{
			for (j = 0; j < dim; j++)
			{
				covar[i][j] = covar[i][j]/Nin - avg[i]*avg[j];
			}
		}
		
		isbad = EnterMatM(covar, Nin);
		
		//enl = 1.0;
		CalcVolume(temp);
	}
	
	void FindLimits()
	{
		for (int i = 0; i < dim; i++)
		{
			double temp = sqrt(4.0*covar[i][i]);
			ulimit[i] = avg[i]+temp;
			if(ulimit[i] > 1.0) ulimit[i] = 1.0;
			llimit[i] = avg[i]-temp;
			if(llimit[i] < 0.0) llimit[i] = 0.0;
		}
			
	}
		
	void RemoveParam(double **points, const int param)
	{
		int i, j, k, m;
		double **covarold = covar;
		double *avgold = avg;
		dim--;
		covar = matrix <double> (dim, dim, 0.0);
		avg = matrix <double> (dim, 0.0);
		factor = pow(PI, dim/2.0)/Gamma(dim/2.0+1.0);
		
		for (j = 0, i = 0; j < dim; j++, i++)
		{
			if (j == param) i++;
			avg[j] = avgold[i];
			for (k = 0, m = 0; k < dim; k++, m++)
			{
				if (k == param) m++;
				covar[j][k] += covarold[i][m];
			}
		}
		del <double> (covarold, dim-1);
		del <double> (avgold);
		
		double sav = map[param];
		for (i = param; i < dim; i++)
		{
			map[i] = map[i+1];
		}
		map[dim] = sav;

		EnterMat(covar, dim);
		area *= (ulimit[map[i]] - llimit[map[i]]);
		//enl = 1.0;
		CalcVolume(points);
		//AdjustParam(points);
	}
	
	void CalcVolume(double **points)
	{
		double **temppts = points;
		double xtemp = x0 = Square(*points++, avg, map);
		int i;
		imax = 0;
		
		for (i = 1; i < N; i++)
		{
			xtemp = Square(*points++, avg, map);
			if (xtemp > x0)
			{
				x0 = xtemp;
				imax = i;
			}
		}
		x0 = sqrt(x0)*pow(enl, 1.0/dim);
		//x0 = sqrt(x0)*pow(2.0, 1.0/dim);
		//E0 = (factor-CorrLimits())*pow(x0, dim)*DetSqrt();
		E0 = (factor)*pow(x0, dim)*DetSqrt();//*CorrVolume();
		X0 /= area;
		if (E0 < X0)
		{
			x0 *= pow(X0/E0, 1.0/dim);
			volume = X0;//(factor)*pow(x0, dim)*DetSqrt()*CorrVolume();//X0*CorrVolume();
			//x0 *= pow(X0/volume, 1.0/dim);
			//volume = (factor)*pow(x0, dim)*DetSqrt()*CorrVolume();//X0*CorrVolume();
			min = false;
		}
		else
		{
			volume = E0;
			min = true;
		}
		volumeCorr = 0.0;//(factor-CorrLimits())*pow(x0, dim)*DetSqrt();
	}
	
	void AdjustParam2(double **temppts)
	{
		if (E0 > 2.0*X0)
		{
			double hi = abs(temppts[imax][map[0]] - avg[0]);
			int hii = 0;
			for (int i = 1; i < dim; i++)
			{
				double temp = abs(temppts[imax][map[i]] - avg[i]);
				if (temp > hi)
				{
					hi = temp;
					hii = i;
				}
			}
			if (covar[hii][hii] > 1.0/15.0)
				RemoveParam(temppts, hii);
		}
	}
	
	void AdjustParam(double **points)
	{
		int i, j;
		FindLimits();
		if (E0 > 2.0*X0)
		{
			double **temppts = points;
			for (i = 0; i < dim; i++)
			{
				points = temppts;
				for (j = 0; j < N; j++)
				{
					double temp = *(*points++ + i);
					if (temp > ulimit[i] || temp < llimit[i])
					{
						break;
					}
				}
				if (j == N)
					RemoveParam(temppts, i);
			}
			
		}
		
		area = 1;
		for (i = dim; i < dimR; i++)
		{
			area *= (ulimit[map[i]] - llimit[map[i]]);
		}
	}
	
	double CorrLimits()
	{
		double factor2 = pow(PI, (dim-1.0)/2.0)/Gamma((dim+1.0)/2.0);
		double corr = 0.0;
		
		for (int i = 0; i < dim; i++)
		{
			double l = x0*sqrt(covar[i][i]);
			if (avg[i] <= 0.0)
			{
				corr += (Beta(0.5, (dim+1.0)/2.0) + BetaInc(0.5, (dim+1.0)/2.0, SQR(avg[i]/l)))/2.0;
			}
			else if (avg[i] < l)
			{
				corr += BetaInc((dim+1.0)/2.0, 0.5, 1.0 - SQR(avg[i]/l))/2.0;
			}
			
			if (avg[i] >= 1.0)
			{
				corr += (Beta(0.5, (dim+1.0)/2.0) + BetaInc(0.5, (dim+1.0)/2.0, SQR((1.0-avg[i])/l)))/2.0;
			}
			else if ((1.0-avg[i]) < l)
			{
				corr += BetaInc((dim+1.0)/2.0, 0.5, 1.0 - SQR((1.0-avg[i])/l))/2.0;
			}
		}
		
		//volume -= factor2*pow(x0, dim)*DetSqrt()*corr;
		return factor2*corr;
	}
	
	double CorrVolume()
	{
		bool corr = false;
		
		for (int i = 0; i < dim; i++)
		{
			double l = x0*sqrt(covar[i][i]);
			if (avg[i] < l || (1.0-avg[i]) < l)
			{
				corr = true;
				break;
			}
		}
		
		if (corr)
		{
			int tot = dim*100;
			int count = tot;
			double *temp = new double[dim];
			for (int i = 0; i < tot; i++)
			{
				Random(temp);
				for (int j = 0; j < dim; j++)
					if (temp[j] < 0.0 || temp[j] > 1.0)
					{
						count--;
						break;
					}
			}
			delete[] temp;
			if (count <= 0) count = 1;
			return double(count)/double(tot);
		}
		else
			return 1.0;
	}
	
	void Random(double *ptr)
	{
		int i;
		double dist = 0.0;
		double temp;

		for (i = 0; i < dim; i++)
		{
			temp = random->Dev();
			ptr[i] = temp;
			dist += temp*temp;
		}

		ElMult(ptr);
		dist = pow(random->Doub(), 1.0/dim)/sqrt(dist);
				
		for (i = 0; i < dim; i++, ptr++)
		{
			(*ptr) = x0*dist*(*ptr) + avg[i];
		}
	}
			
	void test()
	{
		double **fish = matrix <double> (dim, dim);
		double **fish2 = matrix <double> (dim-1, dim-1);
		Inverse(fish);
		for (int i = 0; i < dim-1; i++)
		{
			for (int j = 0; j < dim-1; j++)
			{
				fish2[i][j] = fish[i][j];
			}
		}
		Cholesky chol(fish2, dim-1);
		cout << DetSqrt() << "   " << sqrt(covar[dim-1][dim-1])/chol.DetSqrt() << endl;
		del <double> (fish, dim);
		del <double> (fish2, dim-1);
		getchar();
	}
	
	bool IsMin(){return min;}
	
	void CalcVolume(double **points, bool *own)
	{
		double xtemp;// = x0 = Square(*points++, avg);
		int i, j;

		for (i = 0; i < N; i++)
		{
			if (own[i])
			{
				xtemp = x0 = Square(*points++, avg, map);
				break;
			}
		}
		
		for (j = i; j < N; j++)
		{
			if (own[j])
			{
				xtemp = Square(*points++, avg, map);
				if (xtemp > x0)
				{
					x0 = xtemp;
				}
			}
		}
		//x0 = sqrt(x0)*pow(enl, 1.0/dim);
		x0 = sqrt(x0)*pow(2.0, 1.0/dim);
		E0 = factor*pow(x0, dim)*DetSqrt();

		if (E0 < X0)
		{
			x0 *= pow(X0/E0, 1.0/dim);
			volume = X0;
		}
		else
		{
			volume = E0;
		}
	}
	
	void AddPoint(double *ptr, double **pointsin, bool *own)
	{
		int i, j;
		
		for (i = 0; i < dim; i++)
		{
			for (j = 0; j < dim; j++)
			{
				covar[i][j] = N*(covar[i][j] + ((ptr[map[i]] - avg[i])*(ptr[map[j]] - avg[j]))/(N+1.0))/(N+1.0);
			}
		}
		
		for (i = 0; i < dim; i++)
		{
			avg[i] += (ptr[map[i]]-avg[i])/(N+1.0);
		}
		X0 *= double(N+1)/(N);
		N++;
		isbad = EnterMatM(covar, N);
		CalcVolume(pointsin, own);
	}
	
	void DeletePoint(double *ptr, double **pointsin, bool *own)
	{
		int i, j;
		
		if (N == 1)
		{
			for (i = 0; i < dim; i++)
			{
				avg[i] = 0.0;
				for (j = 0; j < dim; j++)
				{
					covar[i][j] = 0.0;
				}
			}
		}
		else
		{
			for (i = 0; i < dim; i++)
			{
				for (j = 0; j < dim; j++)
				{
					covar[i][j] = N*(covar[i][j] - ((ptr[map[i]] - avg[i])*(ptr[map[j]] - avg[j]))/(N-1.0))/(N-1.0);
				}
			}
			
			for (i = 0; i < dim; i++)
			{
				avg[i] -= (ptr[map[i]]-avg[i])/(N-1.0);
			}
		}
		X0 *= double(N-1)/(N);
		N--;
		isbad = EnterMatM(covar, N);
		CalcVolume(pointsin, own);
	}
	
	void Set(double **points, const double X0in, const int NTemp)
	{
		X0 = X0in;
		//X0 *= double(N)/NTemp;
		int i, j, k;
// 			double **covart = matrix <double> (dim, dim, 0.0);
// 			double *avgt = matrix <double> (dim, 0.0);
		double *ptr1 = NULL;
		double **temp = points;
// 			factor = pow(PI, dim/2.0)/Gamma(dim/2.0+1.0);
		N = NTemp;
		for (i = 0; i < dim; i++)
		{
			avg[i] = 0.0;
			for (j = 0; j < dim; j++)
			{
				covar[i][j] = 0.0;
			}
		}
		
		for (i = 0; i < N; i++)
		{
			ptr1 = *temp++;
			for (j = 0; j < dim; j++)
			{
				avg[j] += ptr1[map[j]];
				for (k = 0; k < dim; k++)
				{
					covar[j][k] += ptr1[map[j]]*ptr1[map[k]];
				}
			}
		}
		
		for (i = 0; i < dim; i++)
		{
			avg[i] /= N;
		}
		
		for (i = 0; i < dim; i++)
		{
			for (j = 0; j < dim; j++)
			{
				covar[i][j] = covar[i][j]/N - avg[i]*avg[j];
			}
		}
		
		isbad = EnterMatM(covar, N);
		CalcVolume(points);
	}
	
	void OutputVolumes()
	{
		cout << "N = " << N << " x0 = " << x0 << "\nE0 = " << E0 << "\nX0 = " << X0 << "\nvolume = " << volume << endl;
	}
	
	double F(){return volume/X0;}
	
	double FCorr(){return volumeCorr/X0;}
	
	double Volume(){return volume;}
	
	double VolumeCorr(){return volumeCorr;}
	
	double h(double *u){return volume*Square(u, avg, map)/x0/x0/X0;}
	
	bool IsBad(){return isbad;}
	
	void printbads()
	{
		for (int i = dim; i < dimR; i++)
			cout << map[i] << " ";
	}
	
	~Group()
	{
		del <double> (avg);
		del <double> (covar, dim);
		delete[] ulimit;
		delete[] llimit;
		delete[] map;
	}
};

struct PointInput
{
	double **points;
	int dim;
	int N;
	double X0;
	double enl;
	MultiNormDev *random;
	char flags;
	void *group;
	
	PointInput(double **pointsin, const int dimin, const int Nin, const double Xin, MultiNormDev *randomin, const char flagin) : points(pointsin), dim(dimin), N(Nin), X0(Xin), random(randomin), flags(flagin){}
};

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

class Points : public Group
{
	private:
		double **points;
		int Npts;
		MultiNormDev *random;
		Points *group1, *group2;
		char flags;

	public:
		Points(double **pointsin, const int dimin, const int Nin, const double Xin, const int lvl, const double enl, MultiNormDev *randomin, const char flagin) : Group(pointsin, dimin, Xin, enl, Nin, NULL, dimin, randomin, flagin), group1(NULL), group2(NULL), random(randomin), flags(flagin), Npts(Nin)
		{
			double **ptr = points = new double * [Nin];
			for (int i = 0; i < Nin; i++)
			{
				*ptr++ = *pointsin++;
			}
			if (Nin > 2.0*dimin+1.0 && (lvl != 0) && F() > 1.0)
				KMeans(lvl-1);
			while (volume/X0 > 1.01)
				Shrink(volume, 1.0);
		}
		
		Points(double **pointsin, const int dimin, const int Nin, const double Xin, const double enl, int *mapin, const int dimRin, MultiNormDev *randomin, const char flagin) : Group(pointsin, dimin, Xin, enl, Nin, mapin, dimRin, randomin, flagin), group1(NULL), group2(NULL), flags(flagin), Npts(Nin)
		{
			double **ptr = points = new double * [Nin];
			for (int i = 0; i < Nin; i++)
			{
				*ptr++ = *pointsin++;
			}
		}
	
		void SetPoints(double **pointsin, const int lvl, MultiNormDev *randomin)
		{
			random = randomin;
			if (Npts > 2.0*dim+1.0 && (lvl != 0) && F() > 1.0)
				KMeans(lvl-1);
		}
		
		void Expand(const double ex)
		{
			
			if(group1 == NULL)
			{
				volume *= ex;
				x0 *= pow(ex, 1.0/dim);
			}
			else
			{
				group1->Expand(ex);
				group2->Expand(ex);
				volume = group1->Volume() + group2->Volume();
			}
		}
		
		void Shrink(const double X0in)
		{
			X0 = X0in/area;
			if(group1 == NULL)
			{
				if (E0 < X0)
				{
					x0 *= pow(X0/volume, 1.0/dim);
					volume = X0;
				}
			}
			else
			{
				group1->Shrink(double(group1->N)*X0in/N);
				group2->Shrink(double(group2->N)*X0in/N);
				volume = group1->Volume() + group2->Volume();
			}
		}
		
		bool IsContained(const double *ptr)
		{
			for (int i = 0; i < Npts; i++)
				if (points[i] == ptr)
					return true;
			return false;
		}
		
		void RemovePt(const double *ptr)
		{
			for (int i = 0; i < Npts; i++)
				if (points[i] == ptr)
				{
					Npts--;
					points[i] = points[Npts];
					return;
				}
		}
		
		void Shrink(double *ptr)
		{
			if (group1 != NULL)
			{
				Points *groupl;
				Points *grouph;
				
				if (group1->N < group2->N)
				{
					groupl = group1;
					grouph = group2;
				}
				else
				{
					groupl = group2;
					grouph = group1;
				}
				if(groupl->IsContained(ptr))
					groupl->Shrink(ptr);
				else
					grouph->Shrink(ptr);
			}
		}
		
		void Shrink(const double X0in, double enlin)
		{
			X0 = X0in/area;
			E0 *= enlin/enl;
			enl = enlin;
			if(group1 == NULL)
			{
				if (E0 < X0)
				{
					x0 *= pow(X0/volume, 1.0/dim);
					volume = X0;
				}
				else
				{
					x0 *= pow(E0/volume, 1.0/dim);
					volume = E0;
				}
			}
			else
			{
				group1->Shrink(double(group1->N)*X0in/N, enlin);
				group2->Shrink(double(group2->N)*X0in/N, enlin);
				volume = group1->Volume() + group2->Volume();
			}
		}
		
		static void *f(void *ptr)
		{
			PointInput *in = (PointInput *)ptr;
		}
		
		bool IsMin()
		{
			if (group1 == NULL)
				return (Npts == dim+1) ? true : false;
			else
				return (group1->IsMin() || group2->IsMin()) ? true : false; 
		}
		
		void KMeans(const int lvl)
		{
			double *center1 = new double[dim];
			double *center2 = new double[dim];
			double *mean1 = new double[dim];
			double *mean2 = new double[dim];
			double **temp1 = new double* [N];
			double **temp2 = new double* [N];
			bool *own = new bool[N];
			bool *own2 = new bool[N];
			double dist1, dist2;
			int num1 = 0, num2 = 0;
			double temp = 0.0;
			double *ptr, *ptr1, *ptr2;
			int i, j, k;
			int last1 = N;
			int last2 = N;
			int redos = 0;
			char flag;
			
			redo:
			if (redos > 100)
			{
				group1 = group2 = NULL;
				del <double *> (temp1);
				del <double *> (temp2);
				del <double> (center1);
				del <double> (center2);
				del <double> (mean1);
				del <double> (mean2);
				del <bool> (own);
				del <bool> (own2);
				return;
			}
			last1 = N;
			last2 = N;
			num1 = 0;
			num2 = 0;
			i = int(N*random->Doub());
  			do
  				j = int(N*random->Doub());
  			while(i == j);
 			
 			ptr1 = points[i];
 			ptr2 = points[j];
			
			for (i = 0; i < dim; i++)
			{
				center1[i] = ptr1[map[i]];
				center2[i] = ptr2[map[i]];
				mean1[i] = 0.0;
				mean2[i] = 0.0;
			}

			for (j = 0; j < N; j++)
			{
				dist1 = 0;
				dist2 = 0;
				ptr = points[j];
				
				for (i = 0; i < dim; i++)
				{
					temp = ptr[map[i]] - center1[i];
					dist1 += temp*temp;
					temp = ptr[map[i]] - center2[i];
					dist2 += temp*temp;
				}
				
				if (dist1 < dist2)
				{
					own[j] = true;
					own2[j] = false;
					num1++;
					for (i = 0; i < dim; i++)
						mean1[i] += ptr[map[i]];
				}
				else
				{
					num2++;
					own[j] = false;
					own2[j] = true;
					for (i = 0; i < dim; i++)
						mean2[i] += ptr[map[i]];
				}
			}

						
			if (num2 <= dim || num1 <= dim)
			{
				redos++;
				goto redo;
			}
			
			do
			{
				flag = 0x00;
				for (i = 0; i < dim; i++)
				{
					center1[i] = mean1[i]/num1;
					center2[i] = mean2[i]/num2;
				}
				
				for (j = 0; j < N; j++)
				{
					dist1 = 0;
					dist2 = 0;
					ptr = points[j];
					for (i = 0; i < dim; i++)
					{
						temp = ptr[map[i]] - center1[i];
						dist1 += temp*temp;
						temp = ptr[map[i]] - center2[i];
						dist2 += temp*temp;
					}
				
					if ((dist1 < dist2)&&(!own[j])&& (num2 > (dim + 1)))
					{
						own[j] = true;
						own2[j] = false;
						flag |= 0x01;
						num1++;
						num2--;
						for (i = 0; i < dim; i++)
						{
							mean1[i] += ptr[map[i]];
							mean2[i] -= ptr[map[i]];
						}
					}
					else if ((dist1 > dist2)&&(own[j])&& (num1 > (dim + 1)))
					{
						own[j] = false;
						own2[j] = true;
						flag |= 0x01;
						num2++;
						num1--;
						for (i = 0; i < dim; i++)
						{
							mean2[i] += ptr[map[i]];
							mean1[i] -= ptr[map[i]];
						}
					}
				}
			}
			while(flag);

			if (num1 > dim && num2 > dim)
			{
				j = 0;
				k = 0;

				for (i = 0; i < N; i++)
				{
					ptr = points[i];
					if (own[i])
						temp1[j++] = ptr;
					else
						temp2[k++] = ptr;
				}
				
				group1 = new Points(temp1, dim, j, double(j)*X0/double(N), enl, map, dimR, random, flags);
				group2 = new Points(temp2, dim, k, double(k)*X0/double(N), enl, map, dimR, random, flags);

				double h1, h2;
				bool good = true;
				int cc = 0;
				double fb, fa, fac, fbc;
				fb=(group1->volume + group2->volume)/X0;
				fbc=(group1->volumeCorr + group2->volumeCorr)/X0;
				
				for (;;)
				{
					cc++;
					j = 0;
					k = 0;
					flag = 0x00;
					if (group1->IsBad() || group2->IsBad())
					{
						good = false;
						break;
					}
					
					for (i = 0; i < N; i++)
					{
						ptr = points[i];
						h1 = group1->h(ptr);
						h2 = group2->h(ptr);
						if (own[i])
						{
							if (h1 > h2 && (num1 > (dim + 1)))
							{
								own[i] = false;
								own2[i] = true;
								num1--;
								num2++;
								flag |= 0x01;
								last1 = i;
								temp2[k++] = ptr;
								
							}
							else
							{
								temp1[j++] = ptr;
								own[i] = true;
								own2[i] = false;
							}
						}
						else
						{
							if (h1 < h2 && (num2 > (dim + 1)))
							{
								own[i] = true;
								own2[i] = false;
								num2--;
								num1++;

									flag |= 0x01;
								last2 = i;
								temp1[j++] = ptr;
							}
							else
							{
								temp2[k++] = ptr;
								own[i] = false;
								own2[i] = true;
							}
						}
					}
					if (!flag || cc > 10)
					{
						break;
					}
					
					if (k > dim && j > dim)
					{
						Points *t1 = new Points(temp1, dim, j, double(j)*X0/double(N), enl, map, dimR, random, flags);
						Points *t2 = new Points(temp2, dim, k, double(k)*X0/double(N), enl, map, dimR, random, flags);
						fa = (t1->volume + t2->volume)/X0;
						if (fa <= fb)
						{
							delete group1;
							delete group2;
							group1 = t1;
							group2 = t2;
							fb = fa;
						}
						else
						{
							
							delete t1;
							delete t2;
							j = group1->N;
							k = group2->N;
							break;
						}
					}
					else
					{
						j = group1->N;
						k = group2->N;
						good = false;
						break;
					}
				}
				if ((((group1->Volume() + group2->Volume()) < volume)||(volume > 2.0*X0))&&good&&((flags) ? ((num1 > dim + 1)&&(num2 > dim + 1)) : true))
				{
					group1->SetPoints(temp1, lvl, random);
					group2->SetPoints(temp2, lvl, random);
					volume = group1->Volume() + group2->Volume();
					volumeCorr = group1->VolumeCorr() + group2->VolumeCorr();
				}
				else
				{
					delete group1;
					delete group2;
					group1 = group2 = NULL;
				}
			}
			
			del <double *> (temp1);
			del <double *> (temp2);
			del <double> (center1);
			del <double> (center2);
			del <double> (mean1);
			del <double> (mean2);
			del <bool> (own);
			del <bool> (own2);
		}
	
		int Member(double *in) 
		{
			if (group1 == NULL)
				return ((Square(in, avg, map) > x0*x0) ? 0 : 1);
			else
				return group1->Member(in) + group2->Member(in);
		}
		
		void RandomC(double *ptr)
		{
			int i;
			double dist = 0.0;
			double temp;

			for (i = 0; i < dim; i++)
			{
				temp = random->Dev();
				ptr[i] = temp;
				dist += temp*temp;
			}
					
			ElMult(ptr);
					
			for (i = 0; i < dim; i++, ptr++)
			{
				(*ptr) = (*ptr) + avg[i];
			}
		}
		
		void GetPoint(double *ptr)
		{
			Points *pointptr;
			double tempptr[dim];
			do
			{
				pointptr = this;
				while(pointptr->group1 != NULL)
				{
					if(group1->Volume()/volume < random->Doub())
					{
						pointptr = pointptr->group2;
					}
					else
					{
						pointptr = pointptr->group1;
					}
				}
				pointptr->Random(tempptr);
				for (int i = 0; i < pointptr->dim; i++)
				{
					ptr[pointptr->map[i]] = tempptr[i];
				}
				for (int i = pointptr->dim; i < dimR; i++)
				{
					ptr[pointptr->map[i]] = (ulimit[pointptr->map[i]] - llimit[pointptr->map[i]])*random->Doub()+ llimit[pointptr->map[i]];
	
				}
			}
			while (1.0/double(this->Member(ptr)) < random->Doub());
		}
		
		void GetPoint2(double *ptr)
		{
			Points *pointptr;
			do
			{
				pointptr = this;
				while(pointptr->group1 != NULL)
				{
					if(group1->Volume()/volume < random->Doub())
					{
						pointptr = pointptr->group2;
					}
					else
					{
						pointptr = pointptr->group1;
					}
				}
				pointptr->Random(ptr);
			}
			while (1.0/double(this->Member(ptr)) < random->Doub());
		}
		
		int GetLevel(int count = 0)
		{
			if(group1 != NULL)
			{
				int int1 = group1->GetLevel(count + 1);
				int int2 = group2->GetLevel(count + 1);
				return (int1 > int2) ? int1 : int2;
			}
			else
				return count;
		}

		void Test(const char *name, int iin, int jin, int level = 0)
		{
			stringstream prefixs;
			string prefix;
			prefixs << level;
			prefixs >> prefix;
			string name2 = string(name)+string(".")+prefix;
			if (group1 == NULL)
			{
				ofstream outt(name2.c_str());
				outt << " NO NO " << endl;
				for (int i = 0; i < Npts; i++)
				{
					outt << points[i][iin] << "   " << points[i][jin] << endl;
				}
				outt << " NO NO " << endl;
				double **newcov = matrix <double> (2, 2);
				newcov[0][0] = covar[iin][iin];
				newcov[0][1] = covar[iin][jin];
				newcov[1][0] = covar[jin][iin];
				newcov[1][1] = covar[jin][jin];
				Cholesky chol(newcov, 2);
				for (int i = 0; i < 2000; i++)
				{
					double xx[2];
					xx[0] = cos(3.1416*i/1000.0);
					xx[1] = sin(3.1416*i/1000.0);
					double r[2];
					chol.ElMult(xx, r);
					outt << (r[0]*x0+avg[iin]) << "   " << (r[1]*x0+avg[jin]) << endl;
				}
				outt << " NO NO " << endl;
				del <double> (newcov, 2);
			}
			else
			{
				group1->Test(name2.c_str(), iin, jin, level+1);
				group2->Test(name2.c_str(), iin, jin, level-1);
			}
		}
		
		~Points()
		{
			delete[] points;
			delete group1;
			delete group2;
		}
};

void UCMC::Convert(double *ptrout, double *ptrin)
{
	double *lptr = lowerLimits;
	double *uptr = upperLimits;
	for (int i = 0; i < ma; i++, ptrout++, ptrin++, lptr++, uptr++)
		*ptrout = *lptr + (*ptrin)*(*uptr - *lptr);
}

void UCMC::Convert_initial(double *ptrout, double *ptrin)
{
	double *lptr = lowerLimits_initial;
	double *uptr = upperLimits_initial;
	for (int i = 0; i < ma; i++, ptrout++, ptrin++, lptr++, uptr++)
		*ptrout = *lptr + (*ptrin)*(*uptr - *lptr);
}

void UCMC::Convert_reverse_initial(double *ptrout, double *ptrin)
{
	double *lptr = lowerLimits_initial;
	double *uptr = upperLimits_initial;
	for (int i = 0; i < ma; i++, ptrout++, ptrin++, lptr++, uptr++)
		*ptrout = ((*ptrin) - (*lptr)) / ((*uptr) - (*lptr));
}

class Counter
{
	private:
		unsigned int N;
		bool *vals;
		unsigned int count;
		unsigned int hits;
		
	public:
		Counter (const unsigned int N) : N(N), count(0), hits(0) 
		{
			vals = new bool[N];
			for (int i = 0; i < N; i++)
				vals[i] = false;
		}
		void operator ++ (int)
		{
			count++;
			count %= N;
			if (vals[count])
			{
				hits--;
				vals[count] = false;
			}
		}
		void Hit()
		{
			vals[count] = true;
			hits++;
		}
		double Ratio() 
		{
			return double(hits)/double(N);
		}
		~Counter()
		{
			delete[] vals;
		}
};

void UCMC::MonoSample(const char *name, const int N, double &lnZ, double *best_fit_params, double *parameter_errors, bool logfile, double** initial_points)
{
	int i, j;
	double **points = matrix <double> (N, ma);
	double *cpt = matrix <double> (ma);
	double *logLikes = matrix <double> (N);
	double *logPriors = matrix <double> (N);
	double likeOld, likeMax, likeMin, Z=0.0, dZ, H = 0.0;
	double w0 = (1.0 - exp(-2.0/N))*0.5;
	double minloglike = 1e30;
#ifdef USE_MPI
	double *loglike_attempts = new double[mpi_np];
	int id;
#endif
	double lnZ_trans;
	double area = 1.0;

	ofstream out;
	ofstream binout;
	ofstream logout;

#ifdef USE_MPI
	if (mpi_id==0)
	{
#endif
		out.open(name);
		binout.open((string(name)+string(".temp")).c_str(), ios::binary);
		if (logfile) {
			string log_filename = string(name) + ".nest.log";
			logout.open(log_filename.c_str());
		}
#ifdef USE_MPI
	}
#endif

	const double tol = 0.5;
	const double enfac = 1.0;
	const double senfac = 2.0;
	const double enl = 1.0;
	const int lvl = -1.0;

	double temp, temp1, test, ratio;
	double ratio_all_procs;
	int imin, count = 0;
	int trystot = 0;
	double slope = 0.0;
	double likeLast = 1.0;
	Counter cRec(100);
	MultiNormDev random(ma, 1.0, rand+mpi_group_num);

	int iterations=0;
	double *ptr1, *ptr2;

	if (mpi_id==0) cout << "Status:  Preparing samples \nProgress:  [\033[20C]" << endl << endl << endl << endl << endl << flush;

#ifdef USE_OPENMP
	double total_time0, total_time;
#endif
	
	// divide this up among the processes
	int icount =0 ;
	int divisions = (N < 20) ? 1 : N/20;
	for (i = mpi_group_num; i < N; i += mpi_ngroups)
	{	
		ptr1 = points[i];
		if (initial_points==NULL) {
			for (j = 0; j < ma; j++)
			{
				ptr1[j] = random.Doub();
			}
			Convert_initial(cpt, ptr1);
		} else {
			for (j = 0; j < ma; j++)
			{
				cpt[j] = initial_points[i][j];
			}
			Convert_reverse_initial(ptr1, cpt);
		}
		logPriors[i] = LogPrior(cpt);
		logLikes[i] = LOGLIKE(cpt) + logPriors[i];

		if ((logLikes[i]*0.0) or (std::isinf(logLikes[i])))
		{
			i -= mpi_ngroups;
		}
		else if ((i % divisions) == 0)
		{
			icount++;
			if (mpi_id==0) {
				cout << "\033[5AProgress:  [" << flush;
				for (j=0; j < icount; j++) cout << "=" << flush;
				cout << "\033[4B" << endl << flush;
			}
		}
	}
#ifdef USE_MPI
	for (int group_num=0; group_num < mpi_ngroups; group_num++) {
		for (i=group_num; i < N; i += mpi_ngroups) {
			id = mpi_group_leader[group_num];
			MPI_Bcast(logLikes+i,1,MPI_DOUBLE,id,MPI_COMM_WORLD);
			MPI_Bcast(points[i],ma,MPI_DOUBLE,id,MPI_COMM_WORLD);
		}
	}
#endif

	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);
	signal(SIGUSR1, &sighandler);
	signal(SIGQUIT, &quitproc);
	
	Points *group;
	
	for (j = 0; j < ma; j++)
	{
		area *= (upperLimits[j] - lowerLimits[j]);
	}
	
	likeMax = likeMin = logLikes[0];
	imin = 0;
	for (i = 0; i < N; i++)
	{
		if (likeMax > logLikes[i])
		{
			likeMax = logLikes[i];
		}
		else if (likeMin < logLikes[i])
		{
			likeMin = logLikes[i];
			imin = i;
		}
	}
	
	ptr1 = points[imin];
	likeLast = likeMin;
	if (mpi_id==0) {
		cout << "\n\033[7AStatus:  Nested Sampling Started" << endl << flush;
		cout << "\033[K\tpoints = \033[K" << "\n\tinv slope = " << "\n\tneg loglike = " << "\n\taccept ratio = " << endl << endl << flush;
	}

	bool accepted;
	double time_lost=0;
#ifdef USE_OPENMP
	total_time0 = omp_get_wtime();
#endif
	bool first_interrupt=true;
	do
	{
		if (mpi_id==0) {
			Convert(cpt, points[imin]);
			binout.write((char *)(cpt), ma*sizeof(double));
			binout.write((char *)&likeMin, sizeof(double));
			binout.write((char *)(logPriors+imin), sizeof(double));
		}
		
		likeOld = likeMin;
		ptr2 = new double[ma];

		accepted = false;
		do
		{
			iterations++;
			for (j = 0; j < ma; j++)
			{
				ptr2[j] = random.Doub();
			}

			cRec++;
			Convert(cpt, ptr2);
			temp1 = LogPrior(cpt);
			temp = LOGLIKE(cpt) + temp1;

#ifdef USE_MPI
			loglike_attempts[mpi_id] = temp;
			i=0;
			do {
				id = mpi_group_leader[i];
				MPI_Bcast(loglike_attempts+i,1,MPI_DOUBLE,id,MPI_COMM_WORLD);
				if (loglike_attempts[i] < likeMin) {
					accepted = true;
					temp = loglike_attempts[i];
					MPI_Bcast(ptr2,ma,MPI_DOUBLE,id,MPI_COMM_WORLD);
					MPI_Bcast(&temp1,1,MPI_DOUBLE,id,MPI_COMM_WORLD);
				}
				trystot++;
			} while ((!accepted) and (++i < mpi_ngroups));
#else
			if (temp < likeMin) accepted = true;
			//if (accepted) cout << "ACCEPTED\n";
			//else cout << "NOT_ACCEPTED loglike=" << temp << " vs. " << likeMin << "\n";
			trystot++;
#endif
		}
		while (!accepted);

		if (temp < minloglike) {
			minloglike = temp;
			Convert(cpt, ptr2);
			for (j = 0; j < ma; j++) best_fit_params[j] = cpt[j];
		}

		ratio_all_procs = double(count+1.0)/iterations;
		ratio = double(count+1.0)/trystot;
		points[imin] = ptr2;
		logPriors[imin] = temp1;
		logLikes[imin] = temp;
		
		delete[] ptr1;
		
		if (likeMax > logLikes[imin])
		{
			likeMax = logLikes[imin];
		}

		likeMin = logLikes[0];
		imin = 0;
		for (i = 0; i < N; i++)
		{
			if (likeMin < logLikes[i])
			{
				likeMin = logLikes[i];
				imin = i;
			}
		}

		ptr1 = points[imin];
		
		slope = w0*exp(1.0/N) + exp(1.0/N-likeLast+likeOld)*slope;
		likeLast = likeOld;

		count++;
		cRec.Hit();
#ifdef USE_OPENMP
		total_time = omp_get_wtime() - total_time0;
		double time_per_it = total_time / iterations;
#endif
		if (mpi_id==0) {
			if (logfile) {
				if (count % 20 == 0) {
					logout << "points=" << count << " (" << cRec.Ratio() << ")" << " inv-slope=" << slope << " neg loglike=" << likeMin << " accept ratio=" << ratio << " mpi_r=" << ratio_all_procs << " it=" << iterations;
#ifdef USE_OPENMP
					logout << " t_it=" << time_per_it << endl;
#else
					logout << endl;
#endif
				}
			}
			cout << "\033[5A\tpoints = " << count << " (" << cRec.Ratio() << ")"
			<< "\n\tinv slope = " << blank << slope 
			<< "\n\tneg loglike = " << blank << likeMin 
			<< "\n\taccept ratio = " << blank << ratio << " mpi_r = " << blank << ratio_all_procs << " it = " << blank << iterations;
#ifdef USE_OPENMP
			cout << " t_it = " << blank << time_per_it << endl << endl;
#else
			cout << endl << endl;
#endif
		}
#ifdef USE_MPI
		MPI_Bcast(&KEEP_RUNNING,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
	}
	while(ratio > 1.0/senfac && KEEP_RUNNING);

	group = new Points(points, ma, N, exp(-double(count+1)/N)*senfac, lvl, enl, &random, 0x00);
	
	if (mpi_id==0) {
		cout << "\033[6AStatus:  MultNest Sampling Started\r\033[5B" << blank << endl;
		if (logfile) logout << "Status:  MultNest Sampling Started" << endl;
	}

	int overflow = 0;
	do
	{
		Convert(cpt, points[imin]);
		if (mpi_id==0) {
			binout.write((char *)(cpt), ma*sizeof(double));
			binout.write((char *)&likeMin, sizeof(double));
			binout.write((char *)(logPriors+imin), sizeof(double));
		}

		likeOld = likeMin;
		ptr2 = new double[ma];
		
		if (group->F() > 1.1 || true)
		{
			char flag = (ratio < 0.5 ? 0x00 : 0x00);
			double cor = (ratio < 0.5 ? 1.0 : 1.0);
			Points *newgroup = new Points(points, ma, N, exp(-double(count+1)/N)*senfac, lvl, enl, &random, flag);
			delete group;
			group = newgroup;
		}
		else
		{
			group->Shrink(exp(-double(count+1)/N)*senfac);
		}
	
		int trytemp = 0;
		accepted = false;
		do
		{
			iterations++;
			int ss = 0;
			do
			{
				group->GetPoint(ptr2);
			}
			while((!checkLimitsUni(ptr2)));

			trytemp++;
			cRec++;
			
			Convert(cpt, ptr2);
			temp1 = LogPrior(cpt);

			temp = LOGLIKE(cpt) + temp1;

#ifdef USE_MPI
			loglike_attempts[mpi_id] = temp;
			i=0;
			do {
				id = mpi_group_leader[i];
				MPI_Bcast(loglike_attempts+i,1,MPI_DOUBLE,id,MPI_COMM_WORLD);
				if (loglike_attempts[i] < likeMin) {
					accepted = true;
					temp = loglike_attempts[i];
					MPI_Bcast(ptr2,ma,MPI_DOUBLE,id,MPI_COMM_WORLD);
					MPI_Bcast(&temp1,1,MPI_DOUBLE,id,MPI_COMM_WORLD);
				}
				trystot++;
			} while ((!accepted) and (++i < mpi_ngroups));
#else
			if (temp < likeMin) accepted = true;
			trystot++;
#endif

			if (trytemp%100 == 0 && trytemp > 0)
			{
				overflow++;
				if (mpi_id==0) {
					cout << "\033[6AStatus:  MultNest Sampling Started (" << overflow << " overflow(s))\r\033[5B" << endl;
					if (logfile) logout << "Overflows: " << overflow << endl;
				}
			}
		}
		while (!accepted);

		ratio_all_procs = double(count+1.0)/iterations;
		ratio = double(count+1.0)/trystot;

		points[imin] = ptr2;
		logPriors[imin] = temp1;
		logLikes[imin] = temp;
		if (temp < minloglike) {
			minloglike = temp;
			Convert(cpt, ptr2);
			for (j = 0; j < ma; j++) best_fit_params[j] = cpt[j];
		}
		
		delete[] ptr1;
		
		if (likeMax > logLikes[imin])
		{
			likeMax = logLikes[imin];
		}

		likeMin = logLikes[0];
		imin = 0;
		for (i = 0; i < N; i++)
		{
			if (likeMin < logLikes[i])
			{
				likeMin = logLikes[i];
				imin = i;
			}
		}
		
		ptr1 = points[imin];
		
		slope = w0*exp(1.0/N) + exp(1.0/N-likeLast+likeOld)*slope;
		test = slope*exp(likeMax - likeOld);
		likeLast = likeOld;
		
		count++;
		cRec.Hit();
#ifdef USE_OPENMP
		total_time = omp_get_wtime() - total_time0;
		double time_per_it = total_time / iterations;
#endif
		if (mpi_id==0) {
			if (logfile) {
				if (count % 20 == 0) {
					logout << "points=" << count << " (" << cRec.Ratio() << ")" << " inv-slope=" << slope << " (test=" << test << ")" << " neg-loglike=" << likeMin << "(" << likeMax << ")" << " F=" << group->F() << ") accept ratio=" << ratio << " mpi_r=" << ratio_all_procs << " it=" << iterations;
#ifdef USE_OPENMP
					logout << " t_it=" << time_per_it << endl;
#else
					logout << endl;
#endif
				}
			}
			cout << "\033[5A\tpoints = " << count << " (" << cRec.Ratio() << ")"
				<< "\n\tinv slope = " << blank << slope << " (test = " << test << ")       "
				<< "\n\tneg loglike = " << blank << likeMin  << blank << "(" << likeMax << ")"
				<< "\n\tF = " << blank << group->F() << bblank << " (";
			group->printbads();
			cout << ")                                                            " << "\n\taccept ratio = " << blank << ratio << " mpi_r = " << blank << ratio_all_procs << " it = " << blank << iterations;
#ifdef USE_OPENMP
			cout << " t_it = " << blank << time_per_it << endl;
#else
			cout << endl;
#endif

		}
#ifdef USE_MPI
		MPI_Bcast(&KEEP_RUNNING,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
	}
	while(test < 1.0/tol && KEEP_RUNNING);
	
	Z = slope;
	
	for (i = 0; i < N; i++)
	{
		Z += exp(-logLikes[i] + likeOld)/N;
		Convert(cpt, points[i]);
		if (mpi_id==0) {
			binout.write((char *)(cpt), ma*sizeof(double));
			binout.write((char *)(logLikes+i), sizeof(double));
			binout.write((char *)(logPriors+i), sizeof(double));
		}
	}
	
	lnZ_trans = log(Z) - likeOld - double(count)/N;
	lnZ = lnZ_trans;
	//lnZ = lnZ_trans + log(area);
	//Z*=area;
	if (mpi_id==0) {
		binout.close();
		ifstream binin((string(name)+string(".temp")).c_str(), ios::binary);
		ptr1 = new double[ma];
		double weight;
		double weighttot = 0.0;
		double *avg = matrix <double> (ma, 0.0);
		double *cov = matrix <double> (ma, 0.0);
		int tot = count;
		count = 0;
		cout << endl << endl;
		if (NDerivedParams > 0) cout << "Calculating derived parameters: [\033[21C]\033[22D" << flush;

		out << "# Sampler: QLens nested sampler, n_livepts = " << N << endl;
		out << "# lnZ = " << lnZ << endl;
		for (i = 0; i < tot; i++, count++)
		{
			binin.read((char *)(ptr1), ma*sizeof(double));
			binin.read((char *)&likeOld, sizeof(double));
			binin.read((char *)&temp1, sizeof(double));
			H += -w0*exp(-likeOld-double(count)/N-lnZ_trans)*likeOld;
			out << w0*exp(-likeOld-double(count)/N-lnZ_trans) << "   ";
			for (j = 0; j < ma; j++)
				out << ptr1[j] << "   ";
			if (NDerivedParams > 0) {
				if (i==0) (this->*DerivedParamPtr)(ptr1,dparam_list); // This is a bit ugly, but it resets the raw chi-square to force its evaluation if it's being included as parameter
				(this->*DerivedParamPtr)(ptr1,dparam_list);
				for (int k = 0; k < NDerivedParams; k++) {
					out << dparam_list[k] << "   ";
				}
				if ((i % (tot/20)) == 0)
				{
					if (mpi_id==0) cout << "=" << flush;
				}
			}
			out << (likeOld-temp1)*2.0 << endl;
		}
		if (NDerivedParams > 0) cout << "]" << endl;
		for (i = 0; i < N; i++)
		{
			binin.read((char *)(ptr1), ma*sizeof(double));
			binin.read((char *)&likeOld, sizeof(double));
			binin.read((char *)&temp1, sizeof(double));
			H += -exp(-likeOld-double(count)/N-lnZ_trans)*likeOld/N;
			out << (weight=exp(-likeOld-double(count)/N-lnZ_trans)/N) << "   ";
			weighttot += weight;
			for (j = 0; j < ma; j++)
			{
				avg[j] += weight*ptr1[j];
				cov[j] += weight*ptr1[j]*ptr1[j];
				out << ptr1[j] << "   ";
			}
			if (NDerivedParams > 0) {
				(this->*DerivedParamPtr)(ptr1,dparam_list);
				for (int k = 0; k < NDerivedParams; k++) {
					out << dparam_list[k] << "   ";
				}
			}
			out << (likeOld-temp1)*2.0 << endl;
		}
		for (j = 0; j < ma; j++)
		{
			avg[j] /= weighttot;
			cov[j] = cov[j]/weighttot - avg[j]*avg[j];
		}
		/*
		// The following doesn't work if the likelihood is parallelized with MPI subgroups; need to rewrite
		double avg_loglike = LOGLIKE(avg) + LogPrior(cpt);
		if (avg_loglike < minloglike) {
			// if the posterior is Gaussian enough, the mean parameters might produce a better fit than any individual
			// point; if so, use this as our best-fit point.
			minloglike = avg_loglike;
			for (j = 0; j < ma; j++) best_fit_params[j] = avg[j];
		}
		*/
		for (j = 0; j < ma; j++) {
			parameter_errors[j] = sqrt(cov[j]);
		}
		
		ofstream outm((string(name)+string(".max")).c_str());
		for (j = 0; j < ma; j++)
		{
			outm << j << ":  " << avg[j] << " +/- " << sqrt(cov[j]/N) << " (sig = " << sqrt(cov[j]) << ")"<< endl;
		}
		delete[] ptr1;
		delete[] avg;
		delete[] cov;
	}
#ifdef USE_MPI
		MPI_Bcast(best_fit_params,ma,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(parameter_errors,ma,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	
	H -= lnZ;
	if (mpi_id==0) {
		if (logfile)
		{
			logout << "Status:  Finished\n";
			logout << count+N << " points, Z = " << exp(lnZ) << endl;
			logout << "lnZ = " << lnZ << " +/- " << sqrt(abs(H/N)) << endl;
		}
		if (NDerivedParams > 0)
			cout << "\033[7A\033[KStatus:  Finished\r\033[6B" << endl;
		else
			cout << "\033[6A\033[KStatus:  Finished\r\033[5B" << endl;
		cout << count+N << " points, Z = " << exp(lnZ) << "                                                   " << endl;
		cout << "lnZ = " << lnZ << " +/- " << sqrt(abs(H/N)) << "                                                   " << endl;
	}
	
	if (system((string("rm -f ") + string(name) + string(".temp")).c_str()) != 0) warn("could not delete temporary files for nested sampling output");
	
	signal(SIGABRT, SIG_DFL);
	signal(SIGTERM, SIG_DFL);
	signal(SIGINT, SIG_DFL);
	signal(SIGUSR1, SIG_DFL);
	signal(SIGQUIT, SIG_DFL);
	
	delete group;
	del <double> (points, N);
	del <double> (logLikes);
	del <double> (cpt);
	del <double> (logPriors);
#ifdef USE_MPI
	delete[] loglike_attempts;
#endif
}

void UCMC::HMC(const char *name, double tol, const char flag)
{
	ofstream out(name);
	double chisq;
	double *aNext = matrix <double> (ma);
	double *dchisqNext = matrix <double> (ma, 0.0);
	double *dchisq = matrix <double> (ma, 0.0);
	double *u=matrix <double> (ma);
	double *c = matrix <double> (ma, 0.0);
	double *dc = matrix <double> (ma, 0.0);
	double *dc1 = matrix <double> (ma, 0.0);
	double *dc2 = matrix <double> (ma, 0.0);
	double *avg = matrix <double> (ma, 0.0);
	double *avg2 = matrix <double> (ma, 0.0);
	double ans, chisqnext;
	double *aptr, *uptr, *dchi2ptr, *cptr, *dcptr, *dc1ptr, *dc2ptr, *avgptr, *avg2ptr, *anptr, *dchi2nptr;
	double kin, kinNext;
	bool finished = false;
	int mult = 1;
	int Nmult = 0;
	int Nmultp;
	int count = 0, total = 1;
	int i, j;
	chisq = LOGLIKE(a);
	BasicDevs gDev(rand);
	double step = 0.01;
	int Nsteps = 10.0;

	cout << "HMC Algorithm Started\n" << "\tpoints = " << "\n\taccept ratio = " << endl;
	for (i = 0, aptr = aNext, uptr = a, dchi2ptr = dchisq; i < ma; i++)
	{
		*aptr++ = *uptr++;
		*dchi2ptr++ = DLogLike(a, i);
	}
	do
	{
		step = 0.1*(1.0+9.0*gDev.Doub());
		Nsteps = 1.0*(1.0+9.0*gDev.Doub());
		kin = kinNext = 0.0;
		for (i = 0, uptr = u; i < ma; i++, uptr++)
		{
			*uptr = gDev.Dev();
			kin += (*uptr)*(*uptr);
		}
		
		for (i = 0, uptr = u; i < ma; i++)
		{
			*(uptr++) -= step*DLogLike(aNext, i)/2.0;
		}
		
		for (i = 1; i < Nsteps; i++)
		{
			for (j = 0, aptr = aNext, uptr = u; j < ma; j++)
			{
				*(aptr++) += step*(*(uptr++));
			}
			
			for (j = 0, uptr = u; j < ma; j++)
			{
				*(uptr++) -= step*DLogLike(aNext, j);
			}
		}
		
		for (i = 0, aptr = aNext, uptr = u; i < ma; i++)
		{
			*(aptr++) += step*(*(uptr++));
		}
		
		for (i = 0, uptr = u, dchi2ptr = dchisqNext; i < ma; i++, uptr++, dchi2ptr++)
		{
			*uptr -= step*(*dchi2ptr = DLogLike(aNext, i))/2.0;
			kinNext += (*uptr)*(*uptr);
		}

		chisqnext = LOGLIKE(aNext);

		if (((ans = (kinNext-kin)/2.0 + chisqnext - chisq) <= 0.0)||(gDev.ExpDev() >= ans))
		{

			
			out << mult << "   ";
			finished = true;
			Nmultp = Nmult + mult;
			for (i = 0, cptr = c, dcptr = dc, dc1ptr = dc1, dc2ptr = dc2, avgptr = avg, avg2ptr = avg2, 
				aptr = a, anptr = aNext, dchi2ptr = dchisq, dchi2nptr = dchisqNext; 
				i < ma; 
				i++, cptr++, dcptr++, dc1ptr++, dc2ptr++, avgptr++, avg2ptr++, 
				aptr++, anptr++, dchi2ptr++, dchi2nptr++)
			{
				*cptr = Nmult*(*cptr + mult*((*aptr - *avgptr)*(*aptr - *avgptr))/Nmultp)/Nmultp;
				*dcptr = (Nmult*(*dcptr) + mult*(*aptr)*(*aptr)*(*aptr)*(*dchi2ptr)/3.0)/Nmultp;
				*dc1ptr = (Nmult*(*dc1ptr) + mult*(*aptr)*(*dchi2ptr))/Nmultp;
				*dc2ptr = (Nmult*(*dc2ptr) + mult*(*dchi2ptr)/3.0)/Nmultp;
				*avgptr += mult*(*aptr-*avgptr)/Nmultp;
				*avg2ptr += mult*((*aptr)*(*aptr)*(*dchi2ptr)/2.0 - *avg2ptr)/Nmultp;
				if (finished && (abs(((*dcptr - (2.0 + *dc2ptr*(*avg2ptr) - *dc1ptr)*(*avg2ptr)*(*avg2ptr)) - *cptr)/(*cptr)) >= tol)) 
				{
					finished = false;
				}
				out << *aptr << "   ";
				*aptr = *anptr;
				*dchi2ptr = *dchi2nptr;
			}
			out << endl;
			
			chisq = chisqnext;
			Nmult = Nmultp;
			mult = 1;
			count++;//\033[2A\t
			cout << "\033[2A\tpoints = " << count << "\n\taccept ratio = " << blank << (double)count/(double)total << endl;
		}
		else
		{
			mult++;
			for (i = 0, aptr = a, anptr = aNext; i < ma; i++)
			{
				*anptr++ = *aptr++;
			}
		}
		total++;
	}
	while(!finished || count <= ma);

	del <double> (aNext);
	del <double> (u);
	del <double> (c);
	del <double> (dc);
	del <double> (dc1);
	del <double> (dc2);
	del <double> (avg);
	del <double> (avg2);
	del <double> (dchisq);
	del <double> (dchisqNext);
}

class BasicPoints
{
	private:
		double **points;
		double width;
		int N;
		int ma;
		
	public:
		BasicPoints(double **pts, const int N, const int ma) : points(pts), N(N), ma(ma)
		{
		}
		
		void Input(const double widthin){width = widthin;}
		
		void Point(double *a, BasicDevs &ran)
		{
			double *pts = points[int(N*ran.Doub())];
			
			for (int j = 0; j < ma; j++)
			{
				a[j] = pts[j] + width*ran.Dev();
			}
		}	
		
		double func(double *a)
		{
			double **pts = points;
			double arg, norm;
			
			double ans = 0.0;
			for (int i = 0; i < N; i++, pts++)
			{

				arg = 0.0;
				norm = 1.0;
				for (int j = 0; j < ma; j++)
				{
					arg += SQR((a[j]-(*pts)[j])/width);
				}
				if (arg < SQR(5.0))
				{
					ans += exp(-(arg)/2.0)/norm;
				}
				
			}
			
			return ans;
		}

		~BasicPoints()
		{
		}
};

