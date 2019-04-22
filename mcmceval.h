#ifndef MCMEVAL_H
#define MCMEVAL_H
#include "GregsMathHdr.h"
#include "random.h"
#include <vector>

//inline double SQR(const double s) { return s*s; }

const char LINEAR = 0x00;
const char EVAL_NONEG = 0x01;
const char LOG = 0x01;
const char XLOG = 0x01;
const char YLOG = 0x02;
const char NOADJX = 0x10;
const char NOADJY = 0x20;
const char INV = 0x04;
const char COL = 0x08;
const char SMOOTHONLY = 0x02;
const char CHIS = 0x04;
const char SIGS = 0x08;
const char EXTRALINES = 0x02;
const char NOADJ = 0x20;
const char HIST = 0x40;
const char SMOOTH = 0x80;
const char LOWER = 0x02;
const char UPPER = 0x04;
const char MULT = 0x01;
const char LOGAXIS = 0x10;
const char ZLOG = 0x04;
const char LIKE = 0x02;

enum EvalTransform { EVAL_NONE, EVAL_LOG_TRANSFORM, EVAL_GAUSS_TRANSFORM, INVERSE_EVAL_GAUSS_TRANSFORM, EVAL_EXP_TRANSFORM, EVAL_LINEAR_TRANSFORM };
enum EvalPriorWeight { EVAL_NO_WEIGHT, EVAL_LOG_PRIOR, EVAL_GAUSS_PRIOR, INVERSE_EVAL_GAUSS_PRIOR };

struct EvalParamTransform
{
	double gaussian_pos, gaussian_sig;
	double linear_A, linear_b;
	bool transform_name, transform_latex_name;
	string transformed_param_name, transformed_latex_name;
	EvalTransform transform;
	EvalParamTransform() { transform = EVAL_NONE; transform_name = false; transform_latex_name = false; }
	void set_none() { transform = EVAL_NONE; }
	void set_log() { transform = EVAL_LOG_TRANSFORM; }
	void set_exp() { transform = EVAL_EXP_TRANSFORM; }
	void set_linear(double &A_in, double &b_in) { transform = EVAL_LINEAR_TRANSFORM; linear_A = A_in; linear_b = b_in; }
	void set_gaussian(double &pos_in, double &sig_in) { transform = EVAL_GAUSS_TRANSFORM; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void set_inverse_gaussian(double &pos_in, double &sig_in) { transform = INVERSE_EVAL_GAUSS_TRANSFORM; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void transform_param_name(string &name_in) { transform_name = true; transformed_param_name = name_in; }
	void transform_latex_param_name(string &name_in) { transform_latex_name = true; transformed_latex_name = name_in; }
	void transform_parameter(double& param)
	{
		double p = param;
		if (transform==EVAL_NONE) return;
		else if (transform==EVAL_LOG_TRANSFORM) param = log(p)/M_LN10;
		else if (transform==EVAL_EXP_TRANSFORM) param = pow(10.0,p);
		else if (transform==EVAL_GAUSS_TRANSFORM)
			param = erff((p - gaussian_pos)/(M_SQRT2*gaussian_sig));
		else if (transform==INVERSE_EVAL_GAUSS_TRANSFORM)
			param = (gaussian_pos + M_SQRT2*gaussian_sig*erfinv(p));
		else if (transform==EVAL_LINEAR_TRANSFORM)
			param = linear_A*p + linear_b;
	}
};

struct ParamPriorWeight
{
	double normfac;
	double gaussian_pos, gaussian_sig;
	EvalPriorWeight prior_type;
	ParamPriorWeight() { prior_type = EVAL_NO_WEIGHT; }
	void set_none() { prior_type = EVAL_NO_WEIGHT; }
	void set_log(const double xmin, const double xmax) { prior_type = EVAL_LOG_PRIOR; normfac = 1.0/log(xmax/xmin); }
	//void set_gaussian(double &pos_in, double &sig_in) { prior_weight = EVAL_GAUSS_PRIOR; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	//void set_inverse_gaussian(double &pos_in, double &sig_in) { prior_weight = INVERSE_EVAL_GAUSS_PRIOR; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	double prior_weight(const double param)
	{
		if (prior_type==EVAL_NO_WEIGHT) return 1.0;
		else if (prior_type==EVAL_LOG_PRIOR) return normfac/param;
		//else if (prior_weight==EVAL_GAUSS_PRIOR)
		//else if (prior_weight==INVERSE_EVAL_GAUSS_PRIOR)
	}
};

class McmcEval
{
	private:
		double ***points;
		double **mults;
		double **chi2;
		double *minvals;
		double *maxvals;
		double *derived_param;
		double *derived_mults;
		EvalParamTransform *param_transforms;
		ParamPriorWeight *prior_weights;
		int *cut;
		int *numOfPoints;
		int totPts;
		double smoothWidth;
		int numOfParam;
		int numOfFiles;
		int min_chisq_pt_j, min_chisq_pt_m, min_chisq_pt_jj;
		double min_chisq_val;
		vector<string> chain_info;

		double rad; // for lensing
		
	public:
		McmcEval() { numOfParam = 0; mults = chi2 = NULL; cut = numOfPoints = NULL; points = NULL; minvals = maxvals = derived_param = derived_mults = NULL; param_transforms = NULL; }
		void input(const char *, int, int, double *, double *, const int mpi_np = 1, const int cut_val = 0, const char flag = 0x00, const bool silent = false, const int n_freeparams = -1, const bool transform_params=false, const char *transform_filename = NULL, const bool importance_sampling = false, const char *prior_weight_filename = NULL, const bool include_log_evidence = false);
		void input_parameter_transforms(const char *transform_filename);
		void input_prior_weights(const char *prior_weight_filename, double *minvals, double* maxvals);

		void remove_comments(string& instring);
		void transform_parameter_names(string *paramnames, string *latex_paramnames);
		void calculate_derived_param();
		void output_min_chisq_pt(void);
		void min_chisq_pt(double*);
		void get_final_points(const int nlist, double **params, double *chisq);
		void FindCoVar(const char *, double *avgs = NULL, double *sigs = NULL, double *minvals = NULL, double *maxvals = NULL);
		void FindCoVar(const char *, int *, const int);
		void FindDerivedSigs(double &center, double &sig);

		void FindHiLow(double &, double &, int, const char flag = 0x00);
		void FindHiLowDerived(double &hi, double &low, double *derived_param, int totPts, const char flag = 0x00);
		void FindMinChisq();

		void FindRanges(double *xminvals, double *xmaxvals, const int nbins, const double threshold);
		void FindRange(double &xmin, double &xmax, const int nbins, int iin, const double threshold);
		void MkHist(double, double, int, const char *, const int, const char flag = LINEAR, double * crap = NULL);
		void DerivedHist(double, double, int, const char *, double& center, double& sig, const char flag = LINEAR, double * crap = NULL);
		double DerivedParam(double *point);

		void MkHistTest(double, double, int, const char *, int, const char flag = LINEAR);
		bool MkHist2D(double, double, double, double, int, int, const char *, int, int, const char flag = LINEAR);
		void MkHist3D(double, double, double, double, int, int, const char *, int, int, int, const char flag = LINEAR);
		double cl(const double, const int, const double, const double, const char flag = LINEAR);
		double derived_cl(const double a, const char flag = LINEAR);

		void Prob(double, double, const int, const char *, const int, const char flag = LOWER);
		void AdjustWt(double (*)(double), int, int);
		void Sort(double *arr, const int n);
		void Sort(double *arr, double *brr, const int n);
		void get_nparams(int &nparams) { nparams = numOfParam; }
		~McmcEval();

		void setRadius(double r) { rad = r; }
		double enclosed_mass_spherical(const double r, const double k0, const double a, const double gamma, const double s);
		double enclosed_mass_spherical_nocore(const double rsq_prime, const double aprime, const double a, const double nprime, const double k0, const double gamma);
		double minchisq_derived_param() { return derived_param[min_chisq_pt_jj]; }
		void OutputChainInfo();
};

class FisherEval
{
	int numOfParam;
	double *bestfitpt;
	double *lower, *upper;
	double **pcov;

	public:
	FisherEval() { bestfitpt = NULL; pcov = NULL; lower = NULL; upper = NULL; numOfParam = 0; }
	void input(const char *file_root, const bool silent);
	~FisherEval();

	void MkDist(const int N, const char *name, const int iin);
	void MkDist2D(const int xN, const int yN, const char *name, const int iin, const int jin);
	void get_nparams(int &nparams) { nparams = numOfParam; }
};

#endif
