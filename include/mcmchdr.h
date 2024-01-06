#ifndef MCMCHDR_H
#define MCMCHDR_H
#include "random.h"
#include "mathexpr.h"
#include <vector>

#ifdef USE_MPI
#include "mpi.h"
#endif

using std::vector;

const char TRANSFORM = 0x01;
const char NOTRANSFORM = 0x00;
const char MULTS = 0x01;

inline bool notUnit(const vector<double> &in)
{
	vector<double>::const_iterator end = in.end();
	for (vector<double>::const_iterator it = in.begin(); it != end; ++it)
	{
		if(*it < 0.0 || *it > 1.0 || *it == -(*it))
			return true;
	}
	return false;
}

template <typename T>
inline typename T::iterator::pointer c_ptr(T &it){return &(*it.begin());}

inline vector<vector<double> > calcCov(const vector<vector<double> > &pts)
{
	static size_t dim = pts[0].size();
	static size_t N = pts.size();

	vector<vector<double> > covar(dim, vector<double>(dim, 0.0));
	vector<double> avg(dim, 0.0);
	size_t i, j;
			  
	vector<vector<double> >::const_iterator pt_end = pts.end();
	for (vector<vector<double> >::const_iterator it = pts.begin(); it != pt_end; ++it)
	{
		for (i = 0; i < dim; i++)
		{
			avg[i] += (*it)[i];
			for (j = 0; j < dim; j++)
			{
				covar[i][j] += (*it)[i]*(*it)[j];
			}
		}
	}
        
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			covar[i][j] = covar[i][j]/N - avg[i]*avg[j]/N/N;
		}
	}
	return covar;
}

inline vector<vector<double> > calcIndent (const vector<vector<double> > &pts)
{
	static size_t dim = pts[0].size();
			  
	vector<vector<double> > covar(dim, vector<double>(dim, 0.0));
	vector<double> hi(dim, 1.0), low(dim, 0.0);
	size_t i;
			  
	vector<vector<double> >::const_iterator end = pts.end();
	for (vector<vector<double> >::const_iterator it = pts.begin(); it != end; ++it)
	{
		for (i = 0; i < dim; i++)
		{
			if (hi[i] < (*it)[i])
			{
				hi[i] = (*it)[i];
			}

			if (low[i] > (*it)[i])
			{
				low[i] = (*it)[i];
			}
		}
	}

	for (i = 0; i < dim; i++)
	{
		covar[i][i] = (hi[i] - low[i])*(hi[i] - low[i])/12.0;
	}
	return covar;
}

class UCMC : public Minimize, private LevenMarq, private Derivative
{
	private:
		double **cvar;
		double *a;
		double *upperLimits;
		double *lowerLimits;
		double *upperLimits_initial;
		double *lowerLimits_initial;
		int ma;
		int NDerivedParams;
		double *dparam_list;
		unsigned long long int rand;
		int mpi_np, mpi_id, mpi_ngroups, mpi_group_num;
		int *mpi_group_leader;
		
	public:
		UCMC();
#ifdef USE_MPI
		void Set_MCMC_MPI(const int mpi_np_in, const int mpi_id_in); // Use this if the likelihood itself is not MPI'd (i.e., there will be a separate likelihood evaluation per MPI process)
		void Set_MCMC_MPI(const int mpi_np_in, const int mpi_id_in, const int mpi_ngroups_in, const int mpi_group_num_in, int *mpi_group_leader_in);
#endif
		void InputPoint(double *, double *, double *, double *, double *, int);
		void InputPoint(double *, double *, double *, int);
		void InputPoint(double *, double *, int);
		bool checkLimits(double *);
		bool checkLimitsUni(double *);
		void Convert(double *, double *);
		void Convert_initial(double *, double *);
		void Convert_reverse_initial(double *ptrout, double *ptrin);
		bool checkReplaceLimits(double *, RandomBasis &);
		void replaceUpperLimits(double *, RandomBasis &);
		void replaceLowerLimits(double *, RandomBasis &);
		inline double Wt(RandomBasis & gDev){return sqrt(gDev.RanMult(cvar));}
		inline double Wu(RandomBasis & gDev){return 1.0;}
		double Chi2(double *);
		void CalcDir(double *, double *);
		void MetHas(const char *, int, const char flag = 0x00);
		void Barker(const char *, int, const char flag = 0x00);
		void MetHasAdapt(const char *name, const double tol, const int Threads, const int cut, double *best_fit_params, const char flag = 0x00);
		void TWalk(const char *name, const double div, const int proj, const double din, const double alim, const double alimt, const double tol, const int Threads, double *best_fit_params, bool logfile, double** initial_points = NULL, std::string chain_info = "", std::string data_info = "");
		void McmcAd(const char *, int);
		void Slicing(const char *, int, const char flag = NOTRANSFORM);
		void SlicingFull(const char *, int);
		void MonoSample(const char *name, const int N, double &lnZ, double *best_fit_params, double *parameter_errors, bool logfile, double** initial_points = NULL, std::string chain_info = "", std::string data_info = "");
		void HMC(const char *name, double tol, const char flag);
		void ApproxCovMatrix();
		void FindCovMatrix();
		void FindCovMatrix(const char *);
		void FindCovMatrix(double *);
		void FindCovMatrix(double **);
		void SaveCovMatrix(const char *);
		void FindEig();
		double GridSearch(int);
		void FindMin();
		void FindMinPow();
		void FindMinLM();
		void PrintPoint();
		int Count(double, double, int, char*, int);
		double OutputParam(int i){return a[i];}
		void SetRan(int n){rand = n;};
		double (UCMC::*LogLikePtr)(double *);
		void (UCMC::*DerivedParamPtr)(double *, double *);
		void SetNDerivedParams(const int);
		virtual double LogLike(double *);
		virtual double LogPrior(double *);
		virtual double DLogLike(double *, const int);
		virtual double DDLogLike(double *, const int, const int);
		virtual double FindCof(double *, double *, double **);
		virtual ~UCMC();
};

#endif
