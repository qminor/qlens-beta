#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <sstream>
#include "GregsMathHdr.h"
#include "mcmceval.h"
#include "random.h"
#include "errors.h"

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;

inline double dummy(const double x){return x;}
inline double pow10(const double x){return pow(10.0, x);}
inline double unit(const double x){return 1.0;}

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
			double chisq = 0.0;
			double temp;
			int i, j, k;
			for (i = 0; i < jMax; i++)
			{
				ki = pow(PI*(i+1)/(N+1)/exp(params[1]), params[2]);
				temp = (log(data[i]*(ki + 1.0)) - params[0]);
				chisq += pow(temp, 2.0);
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
			return chisq;
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

void McmcEval::input(const char *name, int a, int filesin, double *lowLimit, double *hiLimit, const int mpi_np, const int cut_val, const char flag, const bool silent, const bool transform_params, const char *transform_filename)
{
	if (a < 0)
	{
		ifstream inlen;
		string suffix = "_0";
		if (mpi_np > 1) suffix += ".0";
		if (filesin > 1)
			inlen.open((string(name) + suffix).c_str());
		else
			inlen.open(name);
		string str;
		getline(inlen, str);
		istringstream iss(str);
		a = 0;
		while(iss >> str) a++;
		if (a==0) {
			numOfParam = 0;
			cerr << "Error: cannot read data file '" << string(name) + suffix << "'" << endl;
			return;
		}
		if(flag&MULT) a--;
		if(flag&LIKE) a--;
		if (!silent) cout << name << " has " << a << " parameters, " << filesin << " threads, " << mpi_np << " processes." << endl;
	}
	numOfParam = a;
	cout << "Number of parameters: " << a << endl;
	param_transforms = new ParamTransform[a];
	if (transform_params) input_parameter_transforms(transform_filename);
	if (a <= 0) die("no parameters found in input file");
	minvals = new double[a];
	maxvals = new double[a];
	double lowcut[a];
	double highcut[a];
	int k;
	for (k=0; k < a; k++) {
		lowcut[k] = minvals[k] = 1e30;
		highcut[k] = maxvals[k] = -1e30;
	}

	int i,j;
	string file_root(name);
	string paramranges_filename = file_root + ".ranges";
	ifstream paramranges_file(paramranges_filename.c_str());
	if (paramranges_file.is_open()) {
		for (i=0; i < a; i++) {
			if (!(paramranges_file >> lowcut[i])) die("not all parameter ranges are given in file '%s'",paramranges_filename.c_str());
			if (!(paramranges_file >> highcut[i])) die("not all parameter ranges are given in file '%s'",paramranges_filename.c_str());
			//if (i==1) {
				//double tmp = lowcut[1];
				//lowcut[1] = 2 - highcut[1];
				//highcut[1] = 2 - tmp;
			//}
			if (lowcut[i] > highcut[i]) die("cannot have minimum parameter value greater than maximum parameter value in file '%s'",paramranges_filename.c_str());

		}
		paramranges_file.close();
	} else warn("parameter range file '%s' not found",paramranges_filename.c_str());

	numOfFiles = filesin;
	numOfPoints = new int[numOfFiles];
	int **nlines_per_file;
	nlines_per_file = new int*[numOfFiles];
	for (i=0; i < numOfFiles; i++) {
		nlines_per_file[i] = new int[mpi_np];
		for (j=0; j < mpi_np; j++) nlines_per_file[i][j] = 0;
	}
	smoothWidth = 1.0;
	double temp;
	points = new double **[numOfFiles];
	chi2 = new double *[numOfFiles];
	mults = new double*[numOfFiles];
	cut = new int[numOfFiles];
	totPts = 0;
	
	int jmin, mmin;
	const int n_characters = 256;
	char line[n_characters];
	string dum;
	for (j = 0; j < numOfFiles; j++)
	{
		numOfPoints[j] = 0;
		for (k=0; k < mpi_np; k++)
		{
			string name2 = name;
			if (filesin > 1) {
				string name1;
				stringstream stuff; 
				stuff << j;
				stuff >> name1;
				if (mpi_np > 1) {
					string name1_suffix;
					stringstream suffix_str; 
					suffix_str << k;
					suffix_str >> name1_suffix;
					name1 += "." + name1_suffix;
				}
				name2 += "_" + name1;
			}
			ifstream in(name2.c_str());
			while ((in.getline(line,n_characters)) && (!in.eof())) {
				istringstream instream(line);
				if (instream >> dum) {
					numOfPoints[j]++;
					nlines_per_file[j][k]++;
				}
			}
		}
		totPts += numOfPoints[j];
		if (cut_val > numOfPoints[j]) die("cannot cut more points than the chain contains; adjust the cut using the '-c' argument");
		if (cut_val < 0) {
			cut[j] = numOfPoints[j]/10;
			// If no cut has been specified (as indicated by cut = -1), cut the first 10% of the points to be conservative
			//cout << "cut for chain " << j << ": " << cut[j] << endl;
		} else {
			cut[j] = cut_val;
		}
		points[j] = matrix <double> (numOfPoints[j], a);
		chi2[j] = matrix <double> (numOfPoints[j]);
		mults[j] = matrix <double> (numOfPoints[j]);
		string name4 = name;
		if (filesin > 1) {
			string name3;
			stringstream stuff; 
			stuff << j;
			stuff >> name3;
			name4 += "_" + name3;
		}
		if (mpi_np > 1) name4 += ".*";
		if (!silent) cout << "File '" << name4 << "' contains " << numOfPoints[j] << " points." << endl;
	}
	
	int l,m;
	bool remove_point;
	for (j = 0; j < numOfFiles; j++)
	{
		m=0;
		for (l=0; l < mpi_np; l++)
		{
			string name2 = name;
			if (filesin > 1) {
				string name1;
				stringstream stuff; 
				stuff << j;
				stuff >> name1;
				if (mpi_np > 1) {
					string name1_suffix;
					stringstream suffix_str; 
					suffix_str << l;
					suffix_str >> name1_suffix;
					name1 += "." + name1_suffix;
				}
				name2 += "_" + name1;
			}
			ifstream in(name2.c_str());
			bool column_error;
			for (i=0; i < nlines_per_file[j][l]; i++, m++)
			{
				column_error = false;
				in.getline(line,n_characters);
				istringstream instream(line);
				if (flag&MULT)
				{
					if (!(instream >> temp)) column_error = true;
					mults[j][m] = temp;
				}
				else
				{
					mults[j][m] = 1.0;
				}
				
				for (k = 0; k < a; k++)
				{
					if (!(instream >> temp)) column_error = true;
					if (((lowLimit == NULL)||(temp > lowLimit[k]))&&((hiLimit == NULL)||(temp < hiLimit[k])))
					{
						points[j][m][k] = temp;
					}
					else
					{
						for (k++; k < a; k++)
							instream >> temp;
						numOfPoints[j]--;
						totPts--;
						m--;
						break;
					}
				}
				remove_point = false;
				for (k = 0; k < a; k++) {
					if ((points[j][m][k] < lowcut[k]) or (points[j][m][k] > highcut[k])) {
						remove_point = true;
					}
				}
				
				if (flag&LIKE)
				{
					if (!(instream >> chi2[j][m])) column_error = true;
				}
				
				if ((remove_point) or (mults[j][m] <= 0.0))
				{
					numOfPoints[j]--;
					totPts--;
					m--;
				}
				else if (column_error) {
					cout << "Warning: missing column (or incorrect format) in file '" << name2 << "', line " << i << endl;
					cout << chi2[j][m] << endl;
					numOfPoints[j]--;
					totPts--;
					m--;
				}
				else if (instream >> dum) {
					cout << "Warning: extra columns in file '" << name2 << "', line " << i << "; discarding point to be safe (dum=" << dum << ")\n";
					numOfPoints[j]--;
					totPts--;
					m--;
				} else {
					for (k = 0; k < a; k++) {
						param_transforms[k].transform_parameter(points[j][m][k]);
						if (points[j][m][k] < minvals[k]) minvals[k] = points[j][m][k];
						if (points[j][m][k] > maxvals[k]) maxvals[k] = points[j][m][k];
					}
				}

			}
		}
	}
	for (i=0; i < numOfFiles; i++) delete[] nlines_per_file[i];
	delete[] nlines_per_file;
	if (!silent) {
		int cuttot = 0; for (i=0; i < numOfFiles; i++) cuttot += cut[i];
		if (cuttot==0) cout << "Total of " << totPts << " points." << endl;
		else cout << "Total of " << totPts << " points, cutting " << cuttot << " initial points, using " << totPts - cuttot << " points." << endl;
	}

	// this may not get used, but we allocate it regardless
	derived_param = new double[totPts];
	derived_mults = new double[totPts];

	FindMinChisq();
}

void McmcEval::input_parameter_transforms(const char *transform_filename)
{
	int nwords;
	string line;
	vector<string> words;
	stringstream* ws = NULL;
	ifstream transform_file(transform_filename);

	while (!transform_file.eof()) {
		bool transform_name = false;
		getline(transform_file,line);
		//cout << line << endl;
		words.clear();
		if (line=="") continue;
		istringstream linestream(line);
		string word;
		while (linestream >> word)
			words.push_back(word);
		nwords = words.size();
		if (ws != NULL) delete[] ws;
		ws = new stringstream[nwords];
		for (int i=0; i < nwords; i++) ws[i] << words[i];

		string new_name;
		if (words[nwords-1].find("name=")==0) {
			string lstr = words[nwords-1].substr(5);
			stringstream lstream;
			lstream << lstr;
			if (!(lstream >> new_name)) die("invalid parameter name");
			transform_name = true;
			stringstream* new_ws = new stringstream[nwords-1];
			words.erase(words.begin()+nwords-1);
			for (int i=0; i < nwords-1; i++) {
				new_ws[i] << words[i];
			}
			delete[] ws;
			ws = new_ws;
			nwords--;
		}

		if (nwords >= 2) {
			int param_num;
			if (!(ws[0] >> param_num)) die("Invalid parameter number");
			if (param_num >= numOfParam) die("Parameter number does not exist");
			if (transform_name) param_transforms[param_num].transform_param_name(new_name);
			if (words[1]=="none") param_transforms[param_num].set_none();
			else if (words[1]=="log") param_transforms[param_num].set_log();
			else if (words[1]=="exp") param_transforms[param_num].set_exp();
			else if (words[1]=="gaussian") {
				if (nwords != 4) die("gaussian requires two additional arguments (mean,sigma)");
				double sig, pos;
				if (!(ws[2] >> pos)) die("Invalid mean value for Gaussian transformation");
				if (!(ws[3] >> sig)) die("Invalid dispersion value for Gaussian transformation");
				param_transforms[param_num].set_gaussian(pos,sig);
			}
			else if (words[1]=="inverse_gaussian") {
				if (nwords != 4) die("inverse_gaussian requires two additional arguments (mean,sigma)");
				double sig, pos;
				if (!(ws[2] >> pos)) die("Invalid mean value for inverse Gaussian transformation");
				if (!(ws[3] >> sig)) die("Invalid dispersion value for inverse Gaussian transformation");
				param_transforms[param_num].set_inverse_gaussian(pos,sig);
			}
			else die("transformation type not recognized");
		}
		else die("the parameter transformation file requires at least two arguments (param_number,transformation_type)");
	}
	if (ws != NULL) delete[] ws;
	//for (int i=0; i < numOfParam; i++) {
		//cout << "Parameter " << i << ": ";
		//if (param_transforms[i].transform==NONE) cout << "none";
		//else if (param_transforms[i].transform==LOG_TRANSFORM) cout << "log";
		//else if (param_transforms[i].transform==EXP_TRANSFORM) cout << "exp";
		//else if (param_transforms[i].transform==GAUSS_TRANSFORM) cout << "gaussian (mean=" << param_transforms[i].gaussian_pos << ",sig=" << param_transforms[i].gaussian_sig << ")";
		//cout << endl;
	//}
}

void McmcEval::transform_parameter_names(string *paramnames)
{
	for (int i=0; i < numOfParam; i++) {
		if (param_transforms[i].transform_name==true) paramnames[i] = param_transforms[i].transformed_param_name;
	}
}


void McmcEval::calculate_derived_param()
{
	int f,i,j=0;
	for (f=0; f < numOfFiles; f++) {
		for (i = cut[f]; i < numOfPoints[f]; i++) {
			derived_param[j] = DerivedParam(points[f][i]);
			derived_mults[j] = mults[f][i];
			j++;
		}
	}
}

void McmcEval::output_min_chisq_pt(void)
{
	cout << "Minimum chi-square point: (chisq = " << min_chisq_val << ")\n";
	for (int k=0; k < numOfParam; k++) {
		cout << points[min_chisq_pt_j][min_chisq_pt_m][k] << " ";
	}
	cout << endl;
}

void McmcEval::FindDerivedSigs(double &center, double &sig)
{
	sig = 0;
	center = 0;
	double totNum = 0.0;
	int i, j;
	for (j=0; j < totPts; j++)
	{
		totNum += derived_mults[j];
		center += derived_mults[j]*derived_param[j];
		sig += derived_mults[j]*SQR(derived_param[j]); // this is actually N * <x^2>
	}
	
	center /= totNum;
	sig = sqrt(sig/totNum - SQR(center)); // sigma^2 = <x^2> - <x>^2
}

void McmcEval::FindCoVar(const char *name, double *avg, double *sigs, double *minvals, double *maxvals)
{
	bool use_ranges = false;
	bool allocated_avgs = false;
	if ((minvals != NULL) and (maxvals != NULL)) use_ranges = true;
	if (avg == NULL) {
		avg = matrix <double> (numOfParam, 0.0);
		allocated_avgs = true;
	}
	double **coVar = matrix <double> (numOfParam, numOfParam, 0.0);
	double totNum = 0.0;
	int i, j, k, l;
	for (i = 0; i < numOfParam; i++) avg[i] = 0;
	bool in_bounds;
	for (i = 0; i < numOfFiles; i++)
	{
		for (j = cut[i]; j < numOfPoints[i]; j++)
		{
			in_bounds = true;
			for (k = 0; k < numOfParam; k++)
			{
				if ((points[i][j][k] < minvals[k]) or (points[i][j][k] > maxvals[k])) in_bounds = false;
			}
			if (in_bounds==true) {
				totNum += mults[i][j];
				for (k = 0; k < numOfParam; k++)
				{
					avg[k] += mults[i][j]*points[i][j][k];
					for (l = 0; l < numOfParam; l++)
					{
						coVar[k][l] += mults[i][j]*points[i][j][k]*points[i][j][l];
					}
				}
			}
		}
	}
	for (k = 0; k < numOfParam; k++)
	{
		avg[k] /= totNum;
	}
	ofstream out(name);
	for (k = 0; k < numOfParam; k++)
	{
		if (sigs != NULL) sigs[k] = sqrt(coVar[k][k]/totNum - avg[k]*avg[k]);
		for (l = 0; l < numOfParam; l++)
		{
			coVar[k][l] = coVar[k][l]/totNum - avg[k]*avg[l];
			out << coVar[k][l] << "   ";
		}
		out << endl;
	}
	del <double> (coVar, numOfParam);
	if (allocated_avgs) del <double> (avg);
}

void McmcEval::FindCoVar(const char *name, int *nums, const int num)
{
	double *avg = matrix <double> (num, 0.0);
	double **coVar = matrix <double> (num, num, 0.0);
	double totNum = 0.0;
	int i, j, k, l;
	for (i = 0; i < numOfFiles; i++)
	{
		for (j = cut[i]; j < numOfPoints[i]; j++)
		{
			totNum += mults[i][j];
			for (k = 0; k < num; k++)
			{
				avg[k] += mults[i][j]*points[i][j][nums[k]];
				for (l = 0; l < num; l++)
				{
					coVar[k][l] += mults[i][j]*points[i][j][nums[k]]*points[i][j][nums[l]];
				}
			}
		}
	}
	for (k = 0; k < num; k++)
	{
		avg[k] /= totNum;
	}
	ofstream out(name);
	for (k = 0; k < num; k++)
	{
		for (l = 0; l < num; l++)
		{
			coVar[k][l] = coVar[k][l]/totNum - avg[k]*avg[l];
			out << coVar[k][l] << "   ";
		}
		out << endl;
	}
	for (k = 0; k < num; k++)
		out << avg[k] << "   ";
	out << endl;
	del <double> (coVar, num);
	del <double> (avg);
}

void McmcEval::FindHiLow(double &hi, double &low, int iin, const char flag)
{
	int i, j, start=0;
	if (flag != 0)
		while (points[0][start][iin] <= 0) start++;
	hi = low = points[0][start][iin];
	for (j = 0; j < numOfFiles; j++)
	{
		for (i = 0; i < numOfPoints[j]; i++)
		{
			if((flag == 0x00) || (points[j][i][iin] > 0.0))
			{
				if(points[j][i][iin] > hi)
					hi = points[j][i][iin];
				if(points[j][i][iin] < low)
					low = points[j][i][iin];
			}
		}
	}
}

void McmcEval::FindHiLowDerived(double &hi, double &low, double *derived_param, int totNumPts, const char flag)
{
	int i, j, start=0;
	if (flag != 0)
		while (derived_param[start] <= 0) start++;
	hi = low = derived_param[start];
	for (j = 0; j < totNumPts; j++)
	{
		if((flag == 0x00) || (derived_param[j] > 0.0))
		{
			if(derived_param[j] > hi)
				hi = derived_param[j];
			if(derived_param[j] < low)
				low = derived_param[j];
		}
	}
}

void McmcEval::FindRanges(double *xminvals, double *xmaxvals, const int nbins, const double threshold)
{
	for (int i=0; i < numOfParam; i++) {
		FindRange(xminvals[i],xmaxvals[i],nbins,i,threshold);
	}
}

void McmcEval::FindRange(double &xmin, double &xmax, const int nbins, int iin, const double threshold)
{
	double *xvals;
	double *weights;
	int i,j;
	int npoints=0;
	for (j=0; j < numOfFiles; j++)
		npoints += numOfPoints[j];
	xvals = new double[npoints];
	weights = new double[npoints];
	int k=0;
	double total_weight = 0;
	for (j=0; j < numOfFiles; j++) {
		for (i=cut[j]; i < numOfPoints[j]; i++) {
			xvals[k] = points[j][i][iin];
			weights[k] = mults[j][i];
			total_weight += weights[k];
			k++;
		}
	}

	double *bin_count;
	double *bin_left, *bin_right, *bin_center, *bin_vals;
	bin_count = new double[nbins];
	bin_vals = new double[nbins];
	bin_left = new double[nbins];
	bin_right = new double[nbins];
	bin_center = new double[nbins];

	double x, xstep;
	double maxval;

	if (xmin < minvals[iin]) xmin = minvals[iin];
	if (xmax > maxvals[iin]) xmax = maxvals[iin];
	bool rescale = true;
	do
	{
		xstep = (xmax-xmin)/nbins;
		for (i=0, x=xmin; i < nbins; i++, x += xstep)
		{
			bin_count[i] = 0;
			bin_left[i] = x;
			bin_right[i] = x + xstep;
			bin_center[i] = x + 0.5*xstep;
		}

		for (j=0; j < npoints; j++)
		{
			for (i=0; i < nbins; i++) {
				if ((xvals[j] > bin_left[i]) and (xvals[j] <= bin_right[i]))
				{
					bin_count[i] += weights[j];
				}
			}
		}
		maxval=-1e30;
		for (i=0; i < nbins; i++) {
			bin_vals[i] = bin_count[i] * (1.0/(xstep*total_weight));
			if (bin_vals[i] > maxval) maxval = bin_vals[i];
		}
		for (i=0; i < nbins; i++) {
			bin_vals[i] /= maxval;
		}

		if (rescale) {
			int min_i, max_i;
			min_i = 0;
			max_i = nbins - 1;
			while (bin_vals[min_i] < threshold) min_i++;
			while (bin_vals[max_i] < threshold) max_i--;
			if (min_i > 0) min_i--; // allow for a zero value in the first and last bins
			if (max_i < nbins-1) max_i++;
			if ((min_i==0) and (max_i==nbins-1)) {
				rescale = false;
			}
			else {
				xmin = bin_left[min_i];
				xmax = bin_right[max_i];
			}
		}
	}
	while (rescale);
}

void McmcEval::FindMinChisq()
{
	int i, j, jj, k, imin, jmin, jjmin;
	double minchisq=1e30;
	jj=0;
	for (j=0; j < numOfFiles; j++) {
		for (i=cut[j]; i < numOfPoints[j]; i++) {
				if (chi2[j][i] < minchisq) {
					minchisq = chi2[j][i];
					jmin=j; imin=i;
					jjmin=jj;
				}
			jj++;
		}
	}
	min_chisq_val = minchisq;
	min_chisq_pt_j = jmin;
	min_chisq_pt_m = imin;
	min_chisq_pt_jj = jjmin;
}

void McmcEval::MkHist(double al, double ah, const int N, const char *name, const int iin, const char flag, double *crap)
{
	int i, j, f;
	ofstream out(name);
	double *axis = matrix <double> (N);
	double *like  = matrix <double> (N);
	double *lpro = matrix <double> (N, 1e100);
	double hi = 0.0;
	double totNumMult = 0.0;
	int hibin;
	int lineNum = 0;
	double *lines = NULL;
	double *per = NULL;
	double (*fx)(double);
	double (*gx)(double);
	double (*facx)(double);
	double xhi, xlow;
	double step = (ah - al)/N;
	
	int fN;
	double *ftemp;
	double afl;
	double point;
	
	double *smoothy = NULL;
	double *smoothx = NULL;
	double smoothstep=0.0;
	int fNs=0;	
	double *smoothfy = NULL;
	double *smoothfx = NULL;
	double smoothfstep=0.0;
	double smoothHi=0.0;
	double smoothHix=0.0;
	const int smoothSize = 1000;
	smoothWidth = 1.0;
	
	if (flag&LOG)
	{
		fx = log10;
		gx = (flag&LOGAXIS) ? dummy : pow10;
		facx = (flag&NOADJ) ? unit : pow10;
		FindHiLow(xhi, xlow, iin, NONEG);
	}
	else
	{
		fx = dummy;
		gx = (flag&LOGAXIS) ? pow10 : dummy;
		facx = unit;
		FindHiLow(xhi, xlow, iin);
	}

	step = (ah - al)/N;
	
	if (flag&CHIS)
	{
		if (flag&EXTRALINES)
		{
			lineNum = 6;
			double linestemp[6] = {0.6827, 0.90, 0.9545, 0.99, 0.9973, 0.9999};
			lines = matrix <double> (6);
			per = matrix <double> (6, linestemp);
		}
		else
		{
			lineNum = 4;
			double linestemp[4] = {0.6827, 0.90, 0.9545, 0.99};
			lines = matrix <double> (4);
			per = matrix <double> (4, linestemp);
		}
	}
	else if (flag&SIGS)
	{
		if (flag&EXTRALINES)
		{
			lineNum = 3;
			double linestemp[3] = {0.6827, 0.9545, 0.9973};
			lines = matrix <double> (3);
			per = matrix <double> (3, linestemp);
		}
		else
		{
			lineNum = 2;
			double linestemp[2] = {0.6827, 0.9545};
			lines = matrix <double> (2);
			per = matrix <double> (2, linestemp);
		}
	}
	
	double *temp = matrix <double> (N, 0.0);
	double ntemp;
	int n;
	
	if (al > fx(xlow))
		fN = int((al - (fx(xlow)))/step);
	else
		fN = 0;
	afl = al - (fN+1)*step;
	if (fx(xhi) > ah)
		fN += int((fx(xhi) - ah)/step) + N + 2;
	else
		fN += N + 2;
	if (fN <= 0)
	{
		fN = 1;
		cout << "Error: " << iin << endl;
	}
	ftemp = matrix <double> (fN, 0.0);

	for (f = 0; f < numOfFiles; f++)
	{
		for (i = cut[f]; i < numOfPoints[f]; i++)
		{
			point = fx(points[f][i][iin]);
			ntemp = (point - al)/step;
			double lowchi2 = chi2[f][i];
			totNumMult += mults[f][i];
			n = (ntemp < 0) ? -1 : (int)ntemp;
			if ((n >= 0)&&(n < N))
			{
				temp[n] += mults[f][i];
				if (lowchi2 < lpro[n])
					lpro[n] = lowchi2;
			}
			
			ntemp = (point - afl)/step;
			n = (ntemp < 0) ? -1 : (int)ntemp;
			if ((n >= 0)&&(n < fN))
			{
				ftemp[n] += mults[f][i];
			}
		}
	}

	if(flag&SMOOTH)
	{
		int totNum = 0;
		for (f = 0; f < numOfFiles; f++)
		{
			totNum += numOfPoints[f] - cut[f];
		}
		double *sort = new double[totNum];
		double *wsort = new double[totNum];
		double *sortptr = sort;
		double *wsortptr = wsort;
		for (f = 0; f < numOfFiles; f++)
		{
			for (i = cut[f]; i < numOfPoints[f]; i++)
			{
				*(sortptr++) = fx(points[f][i][iin]);
				*(wsortptr++) = mults[f][i];
			}
		}
 		Sort(sort, wsort, totNum);
		smoothy = matrix <double> (smoothSize+1, 0.0);
		smoothx = matrix <double> (smoothSize+1);
		smoothstep = (ah - al)/smoothSize;
		for (i = 0; i <= smoothSize; i++)
			smoothx[i] = al + smoothstep*i;
		fNs = int((double(fN*smoothSize))/double(N)) + 1;
		
		smoothfy = matrix <double> (fNs+1, 0.0);
		smoothfx = matrix <double> (fNs+1);
		smoothfstep = (fN*step)/fNs;
		for (i = 0; i <= fNs; i++)
			smoothfx[i] = afl + smoothfstep*i;

		double width = step*smoothWidth;
		double bot = sqrt(2.0)*width;
		
		for (f = 0; f < numOfFiles; f++)
		{
			for (i = cut[f]; i < numOfPoints[f]; i++)
			{
				int hit, lowt;
				point = fx(points[f][i][iin]);
				ntemp = (point + 5.0*width - *smoothx)/smoothstep+1;
				if (ntemp < 0)
					hit = 0;
				else if (ntemp > smoothSize)
					hit = smoothSize;
				else
					hit = (int)ntemp;
				
				ntemp = (point - 5.0*width - *smoothx)/smoothstep;
				if (ntemp < 0)
					lowt = 0;
				else if (ntemp > smoothSize)
					lowt = smoothSize;
				else
					lowt = (int)ntemp;

				if (hit != lowt)
				{
					for (j = lowt; j <= hit; j++)
					{
						double temp = (smoothx[j] - point)/width;
						smoothy[j] += mults[f][i]*exp(-(temp*temp)/2.0)/SQRT2PI/width;
					}
				}
				
				ntemp = (point + 5.0*width - *smoothfx)/smoothfstep+1;
				if (ntemp < 0)
					hit = 0;
				else if (ntemp > fNs)
					hit = fNs;
				else
					hit = (int)ntemp;
				
				ntemp = (point - 5.0*width - *smoothfx)/smoothfstep;
				if (ntemp < 0)
					lowt = 0;
				else if (ntemp > fNs)
					lowt = fNs;
				else
					lowt = (int)ntemp;

				if (hit != lowt)
				{
					for (j = lowt; j <= hit; j++)
					{
						double temp = (smoothfx[j] - point)/width;
						smoothfy[j] += mults[f][i]*exp(-(temp*temp)/2.0)/SQRT2PI/width;
					}
				}
			}
		}

		for (i = 0; i <= smoothSize; i++)
			smoothy[i] /= facx(smoothx[i]);
	}
	
	if(flag&HIST)
	{
		for (i = 0; i < N; i++)
		{
			axis[i] = al + double(i)*step;
			like[i] = temp[i]/facx(axis[i]);
		}
	}
	else
	{
		for (i = 0; i < N; i++)
		{
			axis[i] = al + (double(i) + 0.5)*step;
			like[i] = temp[i]/facx(axis[i]);
		}
	}
	del <double> (temp);
	
	hi = like[0];
	hibin = 0;
	for (i = 1; i < N; i++)
	{
		if(like[i] > hi)
		{
			hi = like[i];
			hibin = i;
		}
	}
	
	double llow = lpro[0];
	for (i = 1; i < N; i++)
	{
		if(lpro[i] < llow)
		{
			llow = lpro[i];
		}
	}
		
	for (i = 0; i < N; i++)
		lpro[i] = exp((llow-lpro[i])/2.0);
	
	if (flag&SMOOTH)
	{
		smoothHi = smoothy[0];
		smoothHix = smoothx[0];
		int repeat = 1;
		for (i = 1; i <= smoothSize; i++)
		{
			if(smoothy[i] > smoothHi)
			{
				smoothHi = smoothy[i];
				smoothHix = smoothx[i];
				repeat = 1;
			}
			else if (smoothy[i] == smoothHi)
			{
				smoothHix += smoothx[i];
				repeat++;
			}
		}
		smoothHix /= repeat;
	}
	
	if (hi == 0)
	{
		cout << "Hist:  no points within boundaries, returning junk." << endl;
		out << "Hist:  no points within boundaries, returning junk." << endl;
	}
	else if(flag&HIST)
	{
		if (flag&SIGS)
		{
			if (flag&SMOOTH)
			{
				double totSoFar = 0;
				double *lsort = new double[fNs+1];
				double *smoothSave = new double[fNs+1];
				double *psort = new double[fNs+1];
				double tot = 0.0;
				
				for (i = 0; i < fN; i++)
					tot += ftemp[i];
				
				for (i = 0; i <= fNs; i++)
				{
					smoothSave[i] = lsort[i] = smoothfy[i]/facx(smoothfx[i]);
					smoothSave[i] /= smoothHi;
					psort[i] = smoothfy[i]*smoothfstep;
				}

				Sort(lsort, psort, fNs);
				j=0;
				cout << "Info for " << name << ":" << endl;
				cout << "\tHi value:  " << (crap[0]=gx(smoothHix)) << endl;
				for (i = fNs; i > 0; i--)
				{
					totSoFar += psort[i];
					if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/smoothHi;
						j++;
						if (j == lineNum)
							break;
					}
				}
				
				if(j < lineNum)
				{
					totSoFar += psort[i];
					if(totSoFar/tot >= per[j])
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/smoothHi;
						j++;
					}
					lineNum = j;
				}
				
				double next = 0, before = 0, line, temp;
				int *interNum = new int[lineNum];
				double **intercept = new double * [lineNum];
				for (i = 0; i < lineNum; i++)
				{
					interNum[i]=0;
				}
				
				for(i = 0; i <= fNs; i++)
				{
					next = smoothSave[i];
					for (j = 0; j < lineNum; j++)
					{
						line = lines[j];
						temp = next - line;
						if ((before - line)*temp < 0.0 || temp==0.0)
						{
							interNum[j]++;
						}
					}
					before = next;
				}
				
				for (i = 0; i < lineNum; i++)
				{
					intercept[i] = new double[interNum[i]];
					interNum[i]=0;
				}
				
				next = before = 0;
				for(i = 0; i <= fNs; i++)
				{
					next = smoothSave[i];
					for (j = 0; j < lineNum; j++)
					{
						line = lines[j];
						temp = next - line;
						if ((before - line)*temp < 0.0 || temp==0.0)
						{
							intercept[j][interNum[j]] = smoothfx[i] + smoothstep*(line - next)/(next-before);
							interNum[j]++;
						}
					}
					before = next;
				}
				
				crap[1] = gx(intercept[0][0]);
				crap[2] = gx(intercept[0][1]);
				if (interNum[0] > 2)
					cout << iin << " more than 2" << endl;
				
				for (i = 0; i < lineNum; i++)
				{
					cout << "\t" << per[i]*100.0 << "%:  " << lines[i] << " (";
					for (j = 0; j < interNum[i]-1; j++)
					{
						cout << gx(intercept[i][j])   << ", ";
					}
					cout << gx(intercept[i][j])  << ")" << ";  (";
					for (j = 0; j < interNum[i]-1; j++)
					{
						cout << gx(intercept[i][j])- gx(smoothHix)   << ", ";
					}
					cout << gx(intercept[i][j])- gx(smoothHix)  << ")" << endl;
				}
				
				for (i = 0; i < lineNum; i++)
				{
					delete[] intercept[i];
				}
				delete[] intercept;
				delete[] interNum;
				delete[] lsort;
				delete[] psort;
			}
			else
			{
				double totSoFar = 0;
				double *lsort = new double[fN];
				double *psort = new double[fN];
				double tot = 0.0;
				for (i = 0; i < fN; i++)
				{
					lsort[i] = ftemp[i]/facx(afl + (double(i) + 0.5)*step);
					tot += (int)(psort[i] = ftemp[i]);
				}
	
				Sort(lsort, psort, fN);
				j=0;
				cout << "Info for " << name << ":" << endl;
				cout << "\tHi value:  " << smoothHix << endl;
				for (i = fN-1; i > 0; i--)
				{
					totSoFar += psort[i];
					if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/hi;
						cout << "\t" << (per[j]*100.0) << "%:   " << lines[j] << endl;
						j++;
						if (j == lineNum)
							break;
					}
				}
				
				if(j < lineNum)
				{
					totSoFar += psort[i];
					if(totSoFar/tot >= per[j])
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/hi;
						j++;
					}
					lineNum = j;
				}
				delete[] lsort;
				delete[] psort;
			}
		}
		else
			lineNum = 0;
		
		if (flag&SMOOTH)
		{
			out << endl;
			for (j = 0; j <= smoothSize; j++)
				out << gx(smoothx[j]) << "   " << smoothy[j]/smoothHi << endl;
		}
		else
		{
			for (i = 0; i < N-1; i++)
			{
				out << gx(axis[i]) << "   " << like[i]/hi << endl;
				out << gx(axis[i+1]) << "   " << like[i]/hi << endl;
				out << gx(axis[i+1]) << "   " << like[i+1]/hi << endl;
			}
			out << gx(axis[i]) << "   " << like[i]/hi << endl;
			out << gx(ah) << "   " << like[i]/hi << endl;
		}
	}
	else
	{
		for (i = 0; i < N; i++)
		{
			out << gx(axis[i]) << "   " << like[i]/hi;
			for (j = 0; j < lineNum; j++)
				out << "   " << lines[j];
			out<< endl;
		}
	}

	del <double> (axis);
	del <double> (like);
	del <double> (lines);
	del <double> (per);
	if (flag&SMOOTH)
	{
		del <double> (smoothx);
		del <double> (smoothy);
		del <double> (smoothfx);
		del <double> (smoothfy);
	}
	del <double> (ftemp);
	
	return;
}

void McmcEval::DerivedHist(double al, double ah, const int N, const char *name, double& center, double& sig, const char flag, double *crap)
{
	int i, j, f, k;
	ofstream out(name);
	double *axis = matrix <double> (N);
	double *like  = matrix <double> (N);
	double *lpro = matrix <double> (N, 1e100);
	double hi = 0.0;
	double totNumMult = 0.0;
	int hibin;
	int lineNum = 0;
	double *lines = NULL;
	double *per = NULL;
	double (*fx)(double);
	double (*gx)(double);
	double (*facx)(double);
	double xhi, xlow;
	double step = (ah - al)/N;
	
	int fN;
	double *ftemp;
	double afl;
	double point;
	
	double *smoothy = NULL;
	double *smoothx = NULL;
	double smoothstep=0.0;
	int fNs=0;	
	double *smoothfy = NULL;
	double *smoothfx = NULL;
	double smoothfstep=0.0;
	double smoothHi=0.0;
	double smoothHix=0.0;
	const int smoothSize = 1000;
	smoothWidth = 1.0;
	
	if (flag&LOG)
	{
		fx = log10;
		gx = (flag&LOGAXIS) ? dummy : pow10;
		facx = (flag&NOADJ) ? unit : pow10;
		FindHiLowDerived(xhi, xlow, derived_param, totPts, NONEG);
	}
	else
	{
		fx = dummy;
		gx = (flag&LOGAXIS) ? pow10 : dummy;
		facx = unit;
		FindHiLowDerived(xhi, xlow, derived_param, totPts);
	}
	if (ah > 2*xhi) ah=xhi;
	if (al < 0.5*xlow) al=xlow;

	step = (ah - al)/N;
	
	if (flag&CHIS)
	{
		if (flag&EXTRALINES)
		{
			lineNum = 6;
			double linestemp[6] = {0.6827, 0.90, 0.9545, 0.99, 0.9973, 0.9999};
			lines = matrix <double> (6);
			per = matrix <double> (6, linestemp);
		}
		else
		{
			lineNum = 4;
			double linestemp[4] = {0.6827, 0.90, 0.9545, 0.99};
			lines = matrix <double> (4);
			per = matrix <double> (4, linestemp);
		}
	}
	else if (flag&SIGS)
	{
		if (flag&EXTRALINES)
		{
			lineNum = 3;
			double linestemp[3] = {0.6827, 0.9545, 0.9973};
			lines = matrix <double> (3);
			per = matrix <double> (3, linestemp);
		}
		else
		{
			lineNum = 2;
			double linestemp[2] = {0.6827, 0.9545};
			lines = matrix <double> (2);
			per = matrix <double> (2, linestemp);
		}
	}
	
	double *temp = matrix <double> (N, 0.0);
	double ntemp;
	int n;
	
	if (al > fx(xlow))
		fN = int((al - (fx(xlow)))/step);
	else
		fN = 0;
	afl = al - (fN+1)*step;
	if (fx(xhi) > ah)
		fN += int((fx(xhi) - ah)/step) + N + 2;
	else
		fN += N + 2;
	if (fN <= 0)
	{
		fN = 1;
		cout << "Error in DerivedHist" << endl;
	}
	ftemp = matrix <double> (fN, 0.0);

	j=0;
	for (f = 0; f < numOfFiles; f++)
	{
		for (i = cut[f]; i < numOfPoints[f]; i++) {
			point = derived_param[j];
			ntemp = (point - al)/step;
			double lowchi2 = chi2[f][i];
			totNumMult += derived_mults[j];
			n = (ntemp < 0) ? -1 : (int)ntemp;
			if ((n >= 0)&&(n < N))
			{
				temp[n] += derived_mults[j];
				if (lowchi2 < lpro[n])
					lpro[n] = lowchi2;
			}
			
			ntemp = (point - afl)/step;
			n = (ntemp < 0) ? -1 : (int)ntemp;
			if ((n >= 0)&&(n < fN))
			{
				ftemp[n] += derived_mults[j];
			}
			j++;
		}
	}

	if(flag&SMOOTH)
	{
		double *sort = new double[totPts];
		double *wsort = new double[totPts];
		double *sortptr = sort;
		double *wsortptr = wsort;
		for (j = 0; j < totPts; j++)
		{
			*(sortptr++) = derived_param[j];
			*(wsortptr++) = derived_mults[j];
		}
 		Sort(sort, wsort, totPts);
		smoothy = matrix <double> (smoothSize+1, 0.0);
		smoothx = matrix <double> (smoothSize+1);
		smoothstep = (ah - al)/smoothSize;
		for (i = 0; i <= smoothSize; i++)
			smoothx[i] = al + smoothstep*i;
		fNs = int((double(fN*smoothSize))/double(N)) + 1;
		
		smoothfy = matrix <double> (fNs+1, 0.0);
		smoothfx = matrix <double> (fNs+1);
		smoothfstep = (fN*step)/fNs;
		for (i = 0; i <= fNs; i++)
			smoothfx[i] = afl + smoothfstep*i;

		double width = step*smoothWidth;
		double bot = sqrt(2.0)*width;
		
		for (j = 0; j < totPts; j++)
		{
			int hit, lowt;
			point = derived_param[j];
			ntemp = (point + 5.0*width - *smoothx)/smoothstep+1;
			if (ntemp < 0)
				hit = 0;
			else if (ntemp > smoothSize)
				hit = smoothSize;
			else
				hit = (int)ntemp;
			
			ntemp = (point - 5.0*width - *smoothx)/smoothstep;
			if (ntemp < 0)
				lowt = 0;
			else if (ntemp > smoothSize)
				lowt = smoothSize;
			else
				lowt = (int)ntemp;

			if (hit != lowt)
			{
				for (k = lowt; k <= hit; k++)
				{
					double temp = (smoothx[k] - point)/width;
					smoothy[k] += derived_mults[j]*exp(-(temp*temp)/2.0)/SQRT2PI/width;
				}
			}
			
			ntemp = (point + 5.0*width - *smoothfx)/smoothfstep+1;
			if (ntemp < 0)
				hit = 0;
			else if (ntemp > fNs)
				hit = fNs;
			else
				hit = (int)ntemp;
			
			ntemp = (point - 5.0*width - *smoothfx)/smoothfstep;
			if (ntemp < 0)
				lowt = 0;
			else if (ntemp > fNs)
				lowt = fNs;
			else
				lowt = (int)ntemp;

			if (hit != lowt)
			{
				for (k = lowt; k <= hit; k++)
				{
					double temp = (smoothfx[k] - point)/width;
					smoothfy[k] += derived_mults[j]*exp(-(temp*temp)/2.0)/SQRT2PI/width;
				}
			}
		}

		for (i = 0; i <= smoothSize; i++)
			smoothy[i] /= facx(smoothx[i]);
	}
	
	if(flag&HIST)
	{
		for (i = 0; i < N; i++)
		{
			axis[i] = al + double(i)*step;
			like[i] = temp[i]/facx(axis[i]);
		}
	}
	else
	{
		for (i = 0; i < N; i++)
		{
			axis[i] = al + (double(i) + 0.5)*step;
			like[i] = temp[i]/facx(axis[i]);
		}
	}
	del <double> (temp);
	
	hi = like[0];
	hibin = 0;
	for (i = 1; i < N; i++)
	{
		if(like[i] > hi)
		{
			hi = like[i];
			hibin = i;
		}
	}
	
	double llow = lpro[0];
	for (i = 1; i < N; i++)
	{
		if(lpro[i] < llow)
		{
			llow = lpro[i];
		}
	}
		
	for (i = 0; i < N; i++)
		lpro[i] = exp((llow-lpro[i])/2.0);
	
	if (flag&SMOOTH)
	{
		smoothHi = smoothy[0];
		smoothHix = smoothx[0];
		int repeat = 1;
		for (i = 1; i <= smoothSize; i++)
		{
			if(smoothy[i] > smoothHi)
			{
				smoothHi = smoothy[i];
				smoothHix = smoothx[i];
				repeat = 1;
			}
			else if (smoothy[i] == smoothHi)
			{
				smoothHix += smoothx[i];
				repeat++;
			}
		}
		smoothHix /= repeat;
	}
	
	if (hi == 0)
	{
		cout << "Hist:  no points within boundaries, returning junk." << endl;
		out << "Hist:  no points within boundaries, returning junk." << endl;
	}
	else if(flag&HIST)
	{
		if (flag&SIGS)
		{
			if (flag&SMOOTH)
			{
				double totSoFar = 0;
				double *lsort = new double[fNs+1];
				double *smoothSave = new double[fNs+1];
				double *psort = new double[fNs+1];
				double tot = 0.0;
				
				for (i = 0; i < fN; i++)
					tot += ftemp[i];
				
				for (i = 0; i <= fNs; i++)
				{
					smoothSave[i] = lsort[i] = smoothfy[i]/facx(smoothfx[i]);
					smoothSave[i] /= smoothHi;
					psort[i] = smoothfy[i]*smoothfstep;
				}

				Sort(lsort, psort, fNs);
				j=0;
				cout << "Info for " << name << ":" << endl;
				cout << "\tHi value:  " << (crap[0]=gx(smoothHix)) << endl;
				for (i = fNs; i > 0; i--)
				{
					totSoFar += psort[i];
					if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/smoothHi;
						j++;
						if (j == lineNum)
							break;
					}
				}
				
				if(j < lineNum)
				{
					totSoFar += psort[i];
					if(totSoFar/tot >= per[j])
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/smoothHi;
						j++;
					}
					lineNum = j;
				}
				
				double next = 0, before = 0, line, temp;
				int *interNum = new int[lineNum];
				double **intercept = new double * [lineNum];
				for (i = 0; i < lineNum; i++)
				{
					interNum[i]=0;
				}
				
				for(i = 0; i <= fNs; i++)
				{
					next = smoothSave[i];
					for (j = 0; j < lineNum; j++)
					{
						line = lines[j];
						temp = next - line;
						if ((before - line)*temp < 0.0 || temp==0.0)
						{
							interNum[j]++;
						}
					}
					before = next;
				}
				
				for (i = 0; i < lineNum; i++)
				{
					intercept[i] = new double[interNum[i]];
					interNum[i]=0;
				}
				
				next = before = 0;
				for(i = 0; i <= fNs; i++)
				{
					next = smoothSave[i];
					for (j = 0; j < lineNum; j++)
					{
						line = lines[j];
						temp = next - line;
						if ((before - line)*temp < 0.0 || temp==0.0)
						{
							intercept[j][interNum[j]] = smoothfx[i] + smoothstep*(line - next)/(next-before);
							interNum[j]++;
						}
					}
					before = next;
				}
				
				crap[1] = gx(intercept[0][0]);
				crap[2] = gx(intercept[0][1]);
				if (interNum[0] > 2)
					cout << " derived param more than 2" << endl;
				
				for (i = 0; i < lineNum; i++)
				{
					cout << "\t" << per[i]*100.0 << "%:  " << lines[i] << " (";
					for (j = 0; j < interNum[i]-1; j++)
					{
						cout << gx(intercept[i][j])   << ", ";
					}
					cout << gx(intercept[i][j])  << ")" << ";  (";
					for (j = 0; j < interNum[i]-1; j++)
					{
						cout << gx(intercept[i][j])- gx(smoothHix)   << ", ";
					}
					cout << gx(intercept[i][j])- gx(smoothHix)  << ")" << endl;
				}
				
				for (i = 0; i < lineNum; i++)
				{
					delete[] intercept[i];
				}
				delete[] intercept;
				delete[] interNum;
				delete[] lsort;
				delete[] psort;
			}
			else
			{
				double totSoFar = 0;
				double *lsort = new double[fN];
				double *psort = new double[fN];
				double tot = 0.0;
				for (i = 0; i < fN; i++)
				{
					lsort[i] = ftemp[i]/facx(afl + (double(i) + 0.5)*step);
					tot += (int)(psort[i] = ftemp[i]);
				}
	
				Sort(lsort, psort, fN);
				j=0;
				cout << "Info for " << name << ":" << endl;
				cout << "\tHi value:  " << smoothHix << endl;
				for (i = fN-1; i > 0; i--)
				{
					totSoFar += psort[i];
					if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/hi;
						cout << "\t" << (per[j]*100.0) << "%:   " << lines[j] << endl;
						j++;
						if (j == lineNum)
							break;
					}
				}
				
				if(j < lineNum)
				{
					totSoFar += psort[i];
					if(totSoFar/tot >= per[j])
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/hi;
						j++;
					}
					lineNum = j;
				}
				delete[] lsort;
				delete[] psort;
			}
		}
		else
			lineNum = 0;
		
		if (flag&SMOOTH)
		{
			out << endl;
			for (j = 0; j <= smoothSize; j++)
				out << gx(smoothx[j]) << "   " << smoothy[j]/smoothHi << endl;
		}
		else
		{
			for (i = 0; i < N-1; i++)
			{
				out << gx(axis[i]) << "   " << like[i]/hi << endl;
				out << gx(axis[i+1]) << "   " << like[i]/hi << endl;
				out << gx(axis[i+1]) << "   " << like[i+1]/hi << endl;
			}
			out << gx(axis[i]) << "   " << like[i]/hi << endl;
			out << gx(ah) << "   " << like[i]/hi << endl;
		}
	}
	else
	{
		for (i = 0; i < N; i++)
		{
			out << gx(axis[i]) << "   " << like[i]/hi;
			for (j = 0; j < lineNum; j++)
				out << "   " << lines[j];
			out<< endl;
		}
	}

	del <double> (axis);
	del <double> (like);
	del <double> (lines);
	del <double> (per);
	if (flag&SMOOTH)
	{
		del <double> (smoothx);
		del <double> (smoothy);
		del <double> (smoothfx);
		del <double> (smoothfy);
	}
	del <double> (ftemp);

	sig = 0;
	center = 0;
	for (j = 0; j < totPts; j++)
	{
		center += derived_mults[j]*derived_param[j];
		sig += derived_mults[j]*SQR(derived_param[j]); // this is actually N * <x^2>
	}
	
	center /= totNumMult;
	sig = sqrt(sig/totNumMult - SQR(center)); // sigma^2 = <x^2> - <x>^2

	return;
}

double McmcEval::DerivedParam(double *point)
{
	double a,b;
	b = point[8];
	a = sqrt(point[0]*b);
	return M_PI*b*(rad - sqrt(a*a+rad*rad) + a) / (2 - point[1]);
}

void McmcEval::MkHistTest(double al, double ah, const int N, const char *name, int iin, const char flag)
{
	ofstream out(name);
	double *axis = matrix <double> (N);
	double *like  = matrix <double> (N);
	int i, j, f;
	double hi = 0.0;
	int hibin;
	int lineNum = 0;
	double *lines = NULL;
	double *per = NULL;
	double (*fx)(double);
	double (*gx)(double);
	double (*facx)(double);
	double xhi, xlow;
	double step = (ah - al)/N;
	
	int fN;
	int *ftemp;
	double afl;
	double point;
	
	double *smoothy = NULL;
	double *smoothx = NULL;
	double smoothstep=0.0;
	int fNs=0;	
	double *smoothDirs = NULL;
	int smoothStart = 0, smoothStop = 0;
	double smoothHi=0.0;
	double smoothHix=0.0;
	const int smoothSize = 1000;
	const double res = 5.0;
	int tot;
	smoothWidth = 1.0;
	
	if (flag&LOG)
	{
		fx = log10;
		gx = pow10;
		facx = (flag&NOADJ) ? unit : pow10;
		FindHiLow(xhi, xlow, iin, NONEG);
	}
	else
	{
		fx = dummy;
		gx = dummy;
		facx = unit;
		FindHiLow(xhi, xlow, iin);
	}
	
	if (flag&CHIS)
	{
		if (flag&EXTRALINES)
		{
			lineNum = 6;
			double linestemp[6] = {0.6827, 0.90, 0.9545, 0.99, 0.9973, 0.9999};
			lines = matrix <double> (6);
			per = matrix <double> (6, linestemp);
		}
		else
		{
			lineNum = 4;
			double linestemp[4] = {0.6827, 0.90, 0.9545, 0.99};
			lines = matrix <double> (4);
			per = matrix <double> (4, linestemp);
		}
	}
	else if (flag&SIGS)
	{
		if (flag&EXTRALINES)
		{
			lineNum = 3;
			double linestemp[3] = {0.6827, 0.9545, 0.9973};
			lines = matrix <double> (3);
			per = matrix <double> (3, linestemp);
		}
		else
		{
			lineNum = 2;
			double linestemp[2] = {0.6827, 0.9545};
			lines = matrix <double> (2);
			per = matrix <double> (2, linestemp);
		}
	}
	

	int *temp = matrix <int> (N);
	double ntemp;
	int n;
	
	if (al > fx(xlow))
		fN = int((al - (fx(xlow)))/step);
	else
		fN = 0;
	afl = al - (fN+1)*step;
	if (fx(xhi) > ah)
		fN += int((fx(xhi) - ah)/step) + N + 2;
	else
		fN += N + 2;

	ftemp = matrix <int> (fN);
	for (i = 0; i < fN; i++)
		ftemp[i] = 0;
	
	for (i = 0; i < N; i++)
		temp[i] = 0;
	
	for (f = 0; f < numOfFiles; f++)
	{
		for (i = cut[f]; i < numOfPoints[f]; i++)
		{
			point = fx(points[f][i][iin]);
			ntemp = (point - al)/step;
			n = (ntemp < 0) ? -1 : (int)ntemp;
			if ((n >= 0)&&(n < N))
				temp[n]++;
			
			ntemp = (point - afl)/step;
			n = (ntemp < 0) ? -1 : (int)ntemp;
			if ((n >= 0)&&(n < fN))
			{
				ftemp[n]++;
			}
		}
	}

	for (i = 0; i < fN; i++)
		tot += (int)ftemp[i];
	
	if(flag&SMOOTH)
	{
		smoothstep = (ah - al)/smoothSize;
		fNs = int((double(fN*smoothSize))/double(N)) + 1;
		smoothDirs = new double[fNs+1];
		smoothy = matrix <double> (fNs+1, 0.0);
		smoothx = matrix <double> (fNs+1);
		for (i = 0; i <= fNs; i++)
		{
			smoothx[i] = afl + smoothstep*i;
			smoothy[i] = 0.0;
			smoothDirs[i] = 0.0;
		}
		if (afl < al)
			smoothStart = int((al - afl)/smoothstep);
		else
			smoothStart = 0;
		smoothStop = smoothStart+smoothSize;

		double width = step*smoothWidth/2.0;
		
		for (f = 0; f < numOfFiles; f++)
		{
			for (i = cut[f]; i < numOfPoints[f]; i++)
			{
				int hit, lowt;
				point = fx(points[f][i][iin]);
				
				ntemp = (point + res*width - *smoothx)/smoothstep+1;
				if (ntemp < 0)
					hit = 0;
				else if (ntemp > fNs)
					hit = fNs;
				else
					hit = (int)ntemp;
				
				ntemp = (point - res*width - *smoothx)/smoothstep;
				if (ntemp < 0)
					lowt = 0;
				else if (ntemp > fNs)
					lowt = fNs;
				else
					lowt = (int)ntemp;

				if (hit != lowt)
				{
					for (j = lowt; j <= hit; j++)
					{
						double temp = (smoothx[j] - point)/width;
						smoothDirs[j] += exp(-(temp*temp)/2.0)/SQRT2PI/width;
					}
				}
			}
		}
		
		double errAvg = 0.0;
		for (i = 0; i <= fNs; i++)
		{
			errAvg += fabs(smoothDirs[i]);
		}
		errAvg /= double(fNs+ 1);
		width *= 10.0;
		for (f = 0; f < numOfFiles; f++)
		{
			for (i = cut[f]; i < numOfPoints[f]; i++)
			{
				int hit, lowt;
				point = fx(points[f][i][iin]);
				int t = int((point-afl)/smoothstep + 0.5);
				double widtht;
				widtht = width/sqrt(1.0 + smoothDirs[t]*smoothDirs[t]/double(tot)/double(tot));

				ntemp = (point + res*widtht - *smoothx)/smoothstep+1;
				if (ntemp < 0)
					hit = 0;
				else if (ntemp > fNs)
					hit = fNs;
				else
					hit = (int)ntemp;
				
				ntemp = (point - res*widtht - *smoothx)/smoothstep;
				if (ntemp < 0)
					lowt = 0;
				else if (ntemp > fNs)
					lowt = fNs;
				else
					lowt = (int)ntemp;

				if (hit != lowt)
				{
					for (j = lowt; j <= hit; j++)
					{
						double temp = (smoothx[j] - point)/widtht;
						smoothy[j] += exp(-(temp*temp)/2.0)/SQRT2PI/widtht;
					}
				}
			}
		}

		ofstream tout("widths.qdp");
		for (i = 0; i <= fNs; i++)
			if(smoothy[i] != 0.0)
				tout << smoothx[i] << "   " << width/sqrt(1.0 + smoothDirs[i]*smoothDirs[i]/double(tot)/double(tot)) << "   " << width << endl;
	}
	
	if(flag&HIST)
	{
		for (i = 0; i < N; i++)
		{
			axis[i] = al + double(i)*step;
			like[i] = temp[i]/facx(axis[i]);
		}
	}
	else
	{
		for (i = 0; i < N; i++)
		{
			axis[i] = al + (double(i) + 0.5)*step;
			like[i] = temp[i]/facx(axis[i]);
		}
	}
	
	del <int> (temp);

	hi = like[0];
	hibin = 0;
	for (i = 1; i < N; i++)
		if(like[i] > hi)
	{
		hi = like[i];
		hibin = i;
	}
		
	if (flag&SMOOTH)
	{
		smoothHix = smoothx[0];
		smoothHi = smoothy[0]/facx(smoothHix);
		int repeat = 1;
		for (i = smoothStart; i <= smoothStop; i++)
		{
			if(smoothy[i]/facx(smoothx[i]) > smoothHi)
			{
				smoothHix = smoothx[i];
				smoothHi = smoothy[i]/facx(smoothHix);
				repeat = 1;
			}
			else if (smoothy[i]/facx(smoothx[i]) == smoothHi)
			{
				smoothHix += smoothx[i];
				repeat++;
			}
		}
		smoothHix /= repeat;
	}
	
	if (hi == 0)
	{
		cout << "Hist:  no points within boundaries, returning junk." << endl;
		out << "Hist:  no points within boundaries, returning junk." << endl;
	}
	else if(flag&HIST)
	{
		if (flag&SIGS)
		{
			if (flag&SMOOTH)
			{
				double totSoFar = 0;
				double *lsort = new double[fNs+1];
				double *psort = new double[fNs+1];
				int tot = 0;
				//double tot2 = 0;
				
				for (i = 0; i < fN; i++)
					tot += (int)ftemp[i];
				
				for (i = 0; i <= fNs; i++)
				{
					lsort[i] = smoothy[i]/facx(smoothx[i]);
					psort[i] = smoothy[i]*smoothstep;
					//tot2 += (int)psort[i];
				}

				Sort(lsort, psort, fNs);
				j=0;
				cout << "Info for " << name << ":" << endl;
				cout << "\tHi value:  " << gx(smoothHix) << endl;
				for (i = fNs; i > 0; i--)
				{
					totSoFar += psort[i];
					if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/smoothHi;
						cout << "\t" << (per[j]*100.0) << "%:   " << lines[j] << endl;
						j++;
						if (j == lineNum)
							break;
					}
				}
				
				if(j < lineNum)
				{
					totSoFar += psort[i];
					if(totSoFar/tot >= per[j])
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/smoothHi;
						j++;
					}
					lineNum = j;
				}
				delete[] lsort;
				delete[] psort;
			}
			else
			{
				double totSoFar = 0;
				double *lsort = new double[fN];
				double *psort = new double[fN];
				int tot = 0;
				for (i = 0; i < fN; i++)
				{
					lsort[i] = ftemp[i]/facx(afl + (double(i) + 0.5)*step);
					tot += (int)(psort[i] = ftemp[i]);
				}
	
				Sort(lsort, psort, fN);
				j=0;
				cout << "Info for " << name << ":" << endl;
				cout << "\tHi value:  " << smoothHix << endl;
				for (i = fN-1; i > 0; i--)
				{
					totSoFar += psort[i];
					if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/hi;
						cout << "\t" << (per[j]*100.0) << "%:   " << lines[j] << endl;
						j++;
						if (j == lineNum)
							break;
					}
				}
				
				if(j < lineNum)
				{
					totSoFar += psort[i];
					if(totSoFar/tot >= per[j])
					{
						lines[j] = (lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i])/hi;
						j++;
					}
					lineNum = j;
				}
				delete[] lsort;
				delete[] psort;
			}
		}
		else
			lineNum = 0;
	
		if (flag&SMOOTH)
		{
			out << "NO NO" << endl;
			for (j = smoothStart; j <= smoothStop; j++)
				out << gx(smoothx[j]) << "   " << smoothy[j]/smoothHi/facx(smoothx[j]) << endl;
		}
		for (i = 0; i < lineNum; i++)
		{
			out << "NO NO" << endl;
			for (j = 0; j < N; j++)
				out << gx(axis[j]) << "   " << lines[i] << endl;
			out << gx(ah) << "   " << lines[i] << endl;
		}

		out << "NO NO" << endl;
		for (i = 0; i < N-1; i++)
		{
			out << gx(axis[i]) << "   " << like[i]/hi << endl;
			out << gx(axis[i+1]) << "   " << like[i]/hi << endl;
			out << gx(axis[i+1]) << "   " << like[i+1]/hi << endl;
		}
		out << gx(axis[i]) << "   " << like[i]/hi << endl;
		out << gx(ah) << "   " << like[i]/hi << endl;
	}
	else
	{
		for (i = 0; i < N; i++)
		{
			out << gx(axis[i]) << "   " << like[i]/hi;
			for (j = 0; j < lineNum; j++)
				out << "   " << lines[j];
			out<< endl;
		}
	}
	
	del <double> (axis);
	del <double> (like);
	del <double> (lines);
	del <double> (per);
	del <double> (smoothx);
	del <double> (smoothy);
	del <double> (smoothDirs);
	del <int> (ftemp);
	
	return;
}

void McmcEval::MkHist2D(double xl, double xh, double yl, double yh, const int xN, const int yN, const char *name, const int iin, const int jin, const char flag)
{
	double stepx = (xh - xl)/xN;
	double stepy = (yh - yl)/yN;
	double xlow, xhi, ylow, yhi;
	double **temp = matrix <double> (xN, yN, 0.0);
	double **like = matrix <double> (xN, yN);
	int i, j, f, k;
	double nxtemp, nytemp;
	
	double **smoothz = NULL;
	double *smoothy = NULL;
	double *smoothx = NULL;
	double smoothstepx, smoothstepy;
	int xfNs=0, yfNs=0;
	double **smoothfz = NULL;	
	double *smoothfy = NULL;
	double *smoothfx = NULL;
	double smoothfstepx=0.0, smoothfstepy=0.0;
	double smoothHi=0.0, smoothHiy=0.0;
	double smoothHix=0;
	const int smoothSize = 100;
	const double res = 4.0;
	smoothWidth = 1.0;
	
	double (*fx)(double);
	double (*fy)(double);
	double (*gx)(double);
	double (*gy)(double);
	double (*facx)(double);
	double (*facy)(double);
	
	if (flag&XLOG)
	{
		fx = log10;
		gx = pow10;
		facx = (flag&NOADJX) ? unit : pow10;
		FindHiLow(xhi, xlow, iin, NONEG);
	}
	else
	{
		fx = dummy;
		gx = dummy;
		facx = unit;
		FindHiLow(xhi, xlow, iin);
	}
	
	if (flag&YLOG)
	{
		fy = log10;
		gy = pow10;
		facy = (flag&NOADJY) ? unit : pow10;
		FindHiLow(yhi, ylow, jin, NONEG);
	}
	else
	{
		fy = dummy;
		gy = dummy;
		facy = unit;
		FindHiLow(yhi, ylow, jin);
	}
	
	double hi = 0.0;
	int nx, ny;
	
	int xfN;
	if (xl > fx(xlow))
		xfN = int((xl - (fx(xlow)))/stepx);
	else
		xfN = 0;
	int yfN;
	if (yl > fy(ylow))
		yfN = int((yl - (fy(ylow)))/stepy);
	else
		yfN = 0;
	double xfl = xl - (xfN+1)*stepx;
	double yfl = yl - (yfN+1)*stepy;
	if (fx(xhi) > xh)
		xfN += int((fx(xhi) - xh)/stepx) + xN + 2;
	else
		xfN += xN + 2;
	if (fy(yhi) > yh)
		yfN += int((fy(yhi) - yh)/stepy) + yN + 2;
	else
		yfN += yN + 2;
	if (xfN*yfN > 10000000)
	{
		cout << "selected range WAY smaller than actual range (really, WAAAAAAAAAY smaller)" << endl;
		//cout << "press enter and you may lock me up and freeze everything, then you'll not be happy" << endl;
		//getchar();
	}
	double **ftemp = matrix <double> (xfN, yfN, 0.0);
	
	for (f = 0; f < numOfFiles; f++)
	{
		for (i = cut[f]; i < numOfPoints[f]; i++)
		{
			double pointx = fx(points[f][i][iin]);
			double pointy = fy(points[f][i][jin]);
			nxtemp = (pointx - xl)/stepx;
			nytemp = (pointy - yl)/stepy;
			nx = (nxtemp < 0) ? -1 : (int)nxtemp;
			ny = (nytemp < 0) ? -1 : (int)nytemp;
			if ((nx >= 0)&&(nx < xN)&&(ny >= 0)&&(ny < yN))
				temp[nx][ny]+=mults[f][i];
			
			nxtemp = (pointx - xfl)/stepx;
			nytemp = (pointy - yfl)/stepy;
			nx = (nxtemp < 0) ? -1 : (int)nxtemp;
			ny = (nytemp < 0) ? -1 : (int)nytemp;
			if ((nx >= 0)&&(nx < xfN)&&(ny >= 0)&&(ny < yfN))
			{
				ftemp[nx][ny]+=mults[f][i];
			}
		}
	}
	double *smoothzx;
	if(flag&SMOOTH)
	{
		smoothz = matrix <double> (smoothSize+1, smoothSize+1, 0.0);
		smoothy = matrix <double> (smoothSize+1);
		smoothx = matrix <double> (smoothSize+1);
		smoothzx = matrix <double> (smoothSize+1, 0.0);
		smoothstepx = (xh - xl)/smoothSize;
		smoothstepy = (yh - yl)/smoothSize;
		for (i = 0; i <= smoothSize; i++)
		{
			smoothx[i] = xl + smoothstepx*i;
			smoothy[i] = yl + smoothstepy*i;
		}
		xfNs = int((double(xfN*smoothSize))/double(xN)) + 1;
		yfNs = int((double(yfN*smoothSize))/double(yN)) + 1;
				
		smoothfz = matrix <double> (xfNs+1, yfNs+1, 0.0);
		smoothfy = matrix <double> (yfNs+1);
		smoothfx = matrix <double> (xfNs+1);
		smoothfstepx = (xfN*stepx)/xfNs;
		smoothfstepy = (yfN*stepy)/yfNs;
		for (i = 0; i <= xfNs; i++)
			smoothfx[i] = xfl + smoothfstepx*i;
		for (i = 0; i <= yfNs; i++)
			smoothfy[i] = yfl + smoothfstepy*i;

		double widthx = stepx*smoothWidth;
		double widthy = stepy*smoothWidth;
		
		double s2[3] = {widthx*widthx, widthy*widthy, 0.0};
		
		double limitx = sqrt(s2[0]);
		double limity = sqrt(s2[1]);
	
		double det = s2[0]*s2[1] - s2[2]*s2[2];
		double fix = s2[1]/det;
		double fiy = s2[0]/det;
		double fixy = -s2[2]/det;
		//cout << widthx << "   " << widthy << endl;
		for (f = 0; f < numOfFiles; f++)
		{
			for (i = cut[f]; i < numOfPoints[f]; i++)
			{
				int hitx, lowtx, hity, lowty;
				double pointx = fx(points[f][i][iin]);
				double pointy = fy(points[f][i][jin]);
				nxtemp = (pointx + res*limitx - *smoothx)/smoothstepx+1;

				if (nxtemp < 0)
					hitx = 0;
				else if (nxtemp > smoothSize)
					hitx = smoothSize;
				else
					hitx = (int)nxtemp;
				
				nxtemp = (pointx - res*limitx - *smoothx)/smoothstepx;
				
				if (nxtemp < 0)
					lowtx = 0;
				else if (nxtemp > smoothSize)
					lowtx = smoothSize;
				else
					lowtx = (int)nxtemp;

				if (hitx != lowtx)
				{
					for (j = lowtx; j <= hitx; j++)
					{
						double xtemp = (smoothx[j] - pointx)/limitx;
						double intemp = 1.0-xtemp*xtemp/res/res;
						double widthtemp = (intemp < 0.0) ? 0.0 : res*limity*sqrt(intemp);
						nytemp = (pointy + widthtemp - *smoothy)/smoothstepy+1;
						if (nytemp < 0)
							hity = 0;
						else if (nytemp > smoothSize)
							hity = smoothSize;
						else
							hity = (int)nytemp;
						
						nytemp = (pointy - widthtemp - *smoothy)/smoothstepy;
						if (nytemp < 0)
							lowty = 0;
						else if (nytemp > smoothSize)
							lowty = smoothSize;
						else
							lowty = (int)nytemp;
						
						//smoothzx[j] += mults[f][i]*exp(-SQR(smoothx[j] - pointx)/limitx/limitx/2.0);
						
						if (hity != lowty)
						{
							for (k = lowty; k <= hity; k++)
							{
								//double ytemp = (smoothy[k] - pointy)/widthy;
								double dist2 = SQR(smoothx[j] - pointx)*fix + SQR(smoothy[k] - pointy)*fiy + 2.0*(smoothx[j] - pointx)*(smoothy[k] - pointy)*fixy;
								smoothz[j][k] += mults[f][i]*exp(-dist2/2.0)/sqrt(det)/2.0/PI;
							}
						}
					}
				}
				
				nxtemp = (pointx + res*limitx - *smoothfx)/smoothfstepx+1;
				if (nxtemp < 0)
					hitx = 0;
				else if (nxtemp > xfNs)
					hitx = xfNs;
				else
					hitx = (int)nxtemp;
				
				nxtemp = (pointx - res*limitx - *smoothfx)/smoothfstepx;
				if (nxtemp < 0)
					lowtx = 0;
				else if (nxtemp > xfNs)
					lowtx = xfNs;
				else
					lowtx = (int)nxtemp;

				if (hitx != lowtx)
				{
					for (j = lowtx; j <= hitx; j++)
					{
						double xtemp = (smoothfx[j] - pointx)/limitx;
						double intemp = 1.0-xtemp*xtemp/res/res;
						double widthtemp = (intemp < 0.0) ? 0.0 : res*limity*sqrt(intemp);
						nytemp = (pointy + widthtemp - *smoothfy)/smoothfstepy+1;
						if (nytemp < 0)
							hity = 0;
						else if (nytemp > yfNs)
							hity = yfNs;
						else
							hity = (int)nytemp;
						
						nytemp = (pointy - widthtemp - *smoothfy)/smoothfstepy;
						if (nytemp < 0)
							lowty = 0;
						else if (nytemp > yfNs)
							lowty = yfNs;
						else
							lowty = (int)nytemp;
						
						if (hity != lowty)
						{
							for (k = lowty; k <= hity; k++)
							{
								//double ytemp = (smoothfy[k] - pointy)/widthy;
								double dist2 = SQR(smoothfx[j] - pointx)*fix + SQR(smoothfy[k] - pointy)*fiy + 2.0*(smoothfx[j] - pointx)*(smoothfy[k] - pointy)*fixy;
								smoothfz[j][k] += mults[f][i]*exp(-dist2/2.0)/sqrt(det)/2.0/PI;
							}
						}
					}
				}
				
			}
		}

		for (i = 0; i <= smoothSize; i++)
		{
			for (j = 0; j <= smoothSize; j++)
			{
				smoothzx[i] += smoothz[i][j];
			}
		}
		
		for (i = 0; i <= smoothSize; i++)
		{
			for (j = 0; j <= smoothSize; j++)
			{
				smoothz[i][j] /= (facx(smoothx[i])*facy(smoothy[j]));//*smoothzx[i];
			}
		}
	}
	
	for (i = 0; i < xN; i++)
	{
		for (j = 0; j < yN; j++)
		{
			like[i][j] = temp[i][j]/facx((xl + (i + 0.5)*stepx))/facy((yl + (j + 0.5)*stepy));
		}
	}
	
	hi = like[0][0];
	for (i = 0; i < xN; i++)
	{
		for (j = 0; j < yN; j++)
		{
			if(like[i][j] > hi)
			{
				hi = like[i][j];
			}
		}
	}
	
	if (flag&SMOOTH)
	{
		int repeat = 1;
		smoothHi = smoothz[0][0];
		smoothHix = smoothx[0];
		smoothHiy = smoothy[0];
		for (i = 0; i <= smoothSize; i++)
		{
			for (j = 0; j <= smoothSize; j++)
			{
				if(smoothz[i][j] > smoothHi)
				{
					smoothHi = smoothz[i][j];
					smoothHix = smoothx[i];
					smoothHiy = smoothy[j];
					repeat = 1;
				}
				else if (smoothz[i][j] == smoothHi)
				{
					repeat++;
					smoothHix += smoothx[i];
					smoothHiy += smoothy[j];
				}
			}
		}
		smoothHix /= repeat;
		smoothHiy /= repeat;
	}
	
	ofstream outg, outx, outy;
	if (flag&HIST) {
		outg.open((name+string("_hist.dat")).c_str());
		outx.open((name+string("_hist.x")).c_str());
		outy.open((name+string("_hist.y")).c_str());
		for (i = 0; i < xN; i++)
		{
			outx << ((xl + (i + 0.5)*stepx)) << endl;
		}
		for (j = 0; j < yN; j++)
		{
			outy << ((yl + (j + 0.5)*stepy)) << endl;
		}
	}
	
	if (hi == 0)
	{
		cout << "Hist2D:  no points within boundaries, returning junk." << endl;
		if (flag&HIST) outg << "Hist2D:  no points within boundaries, returning junk." << endl;
	}
	else
	{
		const int linesNum = 2;
		double per[] = {0.6827, 0.9545};
		double *sigmas = new double[linesNum];
		if (flag&SMOOTH)
		{
			double tot = 0;
			double totSoFar = 0;
			int totNum = (xfNs+1)*(yfNs+1);
			double *lsort = new double[totNum];
			double *psort = new double[totNum];
			
			for (i = 0; i < xfN; i++)
			{
				for (j = 0; j < yfN; j++)
				{
					tot += ftemp[i][j];
				}
			}
			k = 0;
			for (i = 0; i <= xfNs; i++)
			{
				for (j = 0; j <= yfNs; j++)
				{
					lsort[k] = smoothfz[i][j]/facx(smoothfx[i])/facy(smoothfy[j]);
					psort[k++] = smoothfz[i][j]*smoothfstepx*smoothfstepy;
				}
			}
			Sort(lsort, psort, totNum);
			j=0;
			cout << "Info for " << name << ":" << endl;
			cout << "\tHi point:  (" << gx(smoothHix) << ", " << gy(smoothHiy) << ")" << endl;
			for (i = totNum-1; i > 0; i--)
			{
				totSoFar += psort[i];
				if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
				{
					double sig = lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i];
					sigmas[j] = sig/smoothHi;
					cout << "\t" << (per[j]*100.0) << "%:   " << sig/smoothHi << endl;
					j++;
					if (j == linesNum)
						break;
				}
			}
			if(j < linesNum)
			{
				totSoFar += psort[i];
				if(totSoFar/tot >= per[j])
				{
					double sig = lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i];
					sigmas[j] = sig/smoothHi;
					cout << (per[j]*100.0) << "%:   " << sig/smoothHi << endl;
				}
			}
			delete[] lsort;
			delete[] psort;
		}
		else
		{
			double tot = 0;
			double totSoFar = 0;
			int totNum = xfN*yfN;
			double *lsort = new double[totNum];
			double *psort = new double[totNum];
			
			k = 0;
			for (i = 0; i < xfN; i++)
			{
				for (j = 0; j < yfN; j++)
				{
					lsort[k] = ftemp[i][j]/facx((xfl + (i + 0.5)*stepx))/facy((yfl + (j + 0.5)*stepy));
					tot += (psort[k++] = ftemp[i][j]);
				}
			}
			Sort(lsort, psort, totNum);
			j=0;
			cout << "sigmas for " << name << ":" << endl;
			for (i = totNum-1; i > 0; i--)
			{
				totSoFar += psort[i];
				if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
				{
					double sig = lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i];
					sigmas[j] = sig/hi;
					cout << (per[j]*100.0) << "%:   " << sig/hi << endl;
					j++;
					if (j == linesNum)
						break;
				}
			}
			if(j < linesNum)
			{
				totSoFar += psort[i];
				if(totSoFar/tot >= per[j])
				{
					double sig = lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i];
					sigmas[j] = sig/hi;
					cout << (per[j]*100.0) << "%:   " << sig/hi << endl;
				}
			}
			delete[] lsort;
			delete[] psort;
		}
		
		if (flag&HIST)
		{
			if(flag&INV)
			{
				for (i = 0; i < yN; i++)
				{
					for (j = 0; j < xN; j++)
					{
						outg << like[j][i]/hi << "   ";
					}
					outg << endl;
				}
			}
			else if (flag&COL)
			{
				for (i = 0; i < xN; i++)
				{
					for (j = 0; j < yN; j++)
					{
						outg << pow(10.0, xl + (i+0.5)*stepx) << "   " << pow(10.0, yl + (j+0.5)*stepy) << "   " << like[i][j]/hi << endl;
					}
				}
			}
			else
			{
				for (j = 0; j < yN; j++)
				{
					for (i = 0; i < xN; i++)
					{
						outg << like[i][j]/hi << "   ";
					}
					outg << endl;
				}
			}
		}
		
		if (flag&SMOOTH)
		{
			ofstream outsg(name);
			ofstream outsx((name+string("_x")).c_str());
			ofstream outsy((name+string("_y")).c_str());
			
			for (i = 0; i <= smoothSize; i++)
			{
				outsx << (smoothx[i]) << endl;
			}
			
			for (j = 0; j <= smoothSize; j++)
			{
				outsy << (smoothy[j]) << endl;
			}
			
			for (j = 0; j <= smoothSize; j++)
			{
				for (i = 0; i <= smoothSize; i++)
				{
					outsg << smoothz[i][j]/smoothHi << "   ";
				}
				outsg << endl;
			}
			ofstream outcont((name+string("_cont")).c_str());
			for (i = linesNum-1; i >= 0; i--)
			{
				outcont << sigmas[i] << "   ";
			}
			outcont << endl;
		}
		delete[] sigmas;
	}
	
	del <double> (like, xN);
	del <double> (temp, xN);
	del <double> (ftemp, xfN);
	if (flag&SMOOTH)
	{
		del <double> (smoothx);
		del <double> (smoothy);
		del <double> (smoothz, smoothSize+1);
		del <double> (smoothfx);
		del <double> (smoothfy);
		del <double> (smoothfz, xfNs);
	}
	
	return;
}

void McmcEval::MkHist3D(double xl, double xh, double yl, double yh, const int xN, const int yN, const char *name, const int iin, const int jin, const int kin, const char flag)
{
	double stepx = (xh - xl)/xN;
	double stepy = (yh - yl)/yN;
	double xlow, xhi, ylow, yhi;
	double **temp = matrix <double> (xN, yN, 0.0);
	//double **tempNum = matrix <double> (xN, yN, 0.0);
	double **like = matrix <double> (xN, yN);
	int i, j, f, k;
	double nxtemp, nytemp;
	
	double **smoothz = NULL;
	double *smoothy = NULL;
	double *smoothx = NULL;
	double smoothstepx, smoothstepy;
	int xfNs=0, yfNs=0;
	double **smoothfz = NULL;	
	double *smoothfy = NULL;
	double *smoothfx = NULL;
	double smoothfstepx=0.0, smoothfstepy=0.0;
	double smoothHi=0.0, smoothHiy=0.0;
	double smoothHix=0;
	const int smoothSize = 100;
	const double res = 4.0;
	smoothWidth = 1.0;
	
	double (*fx)(double);
	double (*fy)(double);
	double (*gx)(double);
	double (*gy)(double);
	double (*facx)(double);
	double (*facy)(double);
	
	if (flag&XLOG)
	{
		fx = log10;
		gx = pow10;
		facx = (flag&NOADJX) ? unit : pow10;
		FindHiLow(xhi, xlow, iin, NONEG);
	}
	else
	{
		fx = dummy;
		gx = dummy;
		facx = unit;
		FindHiLow(xhi, xlow, iin);
	}
	
	if (flag&YLOG)
	{
		fy = log10;
		gy = pow10;
		facy = (flag&NOADJY) ? unit : pow10;
		FindHiLow(yhi, ylow, jin, NONEG);
	}
	else
	{
		fy = dummy;
		gy = dummy;
		facy = unit;
		FindHiLow(yhi, ylow, jin);
	}
	
	double hi = 0.0;
	int nx, ny;
	
	int xfN;
	if (xl > fx(xlow))
		xfN = int((xl - (fx(xlow)))/stepx);
	else
		xfN = 0;
	int yfN;
	if (yl > fy(ylow))
		yfN = int((yl - (fy(ylow)))/stepy);
	else
		yfN = 0;
	double xfl = xl - (xfN+1)*stepx;
	double yfl = yl - (yfN+1)*stepy;
	if (fx(xhi) > xh)
		xfN += int((fx(xhi) - xh)/stepx) + xN + 2;
	else
		xfN += xN + 2;
	if (fy(yhi) > yh)
		yfN += int((fy(yhi) - yh)/stepy) + yN + 2;
	else
		yfN += yN + 2;

	double **ftemp = matrix <double> (xfN, yfN, 0.0);
	//double **ftempNum = matrix <double> (xfN, yfN, 0.0);
	for (f = 0; f < numOfFiles; f++)
	{
		for (i = cut[f]; i < numOfPoints[f]; i++)
		{
			double pointx = fx(points[f][i][iin]);
			double pointy = fy(points[f][i][jin]);
			double pointz = points[f][i][kin];
			double mul = mults[f][i];
			nxtemp = (pointx - xl)/stepx;
			nytemp = (pointy - yl)/stepy;
			nx = (nxtemp < 0) ? -1 : (int)nxtemp;
			ny = (nytemp < 0) ? -1 : (int)nytemp;
			if ((nx >= 0)&&(nx < xN)&&(ny >= 0)&&(ny < yN))
			{
				//tempNum[nx][ny] += mul;
				temp[nx][ny]+=mul*pointz;
			}
			
			nxtemp = (pointx - xfl)/stepx;
			nytemp = (pointy - yfl)/stepy;
			nx = (nxtemp < 0) ? -1 : (int)nxtemp;
			ny = (nytemp < 0) ? -1 : (int)nytemp;
			if ((nx >= 0)&&(nx < xfN)&&(ny >= 0)&&(ny < yfN))
			{
				//ftempNum[nx][ny]+=mul;
				ftemp[nx][ny]+=mul*pointz;
			}
		}
	}
	
	if(flag&SMOOTH)
	{
		smoothz = matrix <double> (smoothSize+1, smoothSize+1, 0.0);
		smoothy = matrix <double> (smoothSize+1);
		smoothx = matrix <double> (smoothSize+1);
		smoothstepx = (xh - xl)/smoothSize;
		smoothstepy = (yh - yl)/smoothSize;
		for (i = 0; i <= smoothSize; i++)
		{
			smoothx[i] = xl + smoothstepx*i;
			smoothy[i] = yl + smoothstepy*i;
		}
		xfNs = int((double(xfN*smoothSize))/double(xN)) + 1;
		yfNs = int((double(yfN*smoothSize))/double(yN)) + 1;
				
		smoothfz = matrix <double> (xfNs+1, yfNs+1, 0.0);
		smoothfy = matrix <double> (yfNs+1);
		smoothfx = matrix <double> (xfNs+1);
		smoothfstepx = (xfN*stepx)/xfNs;
		smoothfstepy = (yfN*stepy)/yfNs;
		for (i = 0; i <= xfNs; i++)
			smoothfx[i] = xfl + smoothfstepx*i;
		for (i = 0; i <= yfNs; i++)
			smoothfy[i] = yfl + smoothfstepy*i;

		double widthx = stepx*smoothWidth;
		double widthy = stepy*smoothWidth;
		
		for (f = 0; f < numOfFiles; f++)
		{
			for (i = cut[f]; i < numOfPoints[f]; i++)
			{
				int hitx, lowtx, hity, lowty;
				double pointx = fx(points[f][i][iin]);
				double pointy = fy(points[f][i][jin]);
				double pointz = points[f][i][kin];
				double mul = mults[f][i];
				nxtemp = (pointx + res*widthx - *smoothx)/smoothstepx+1;

				if (nxtemp < 0)
					hitx = 0;
				else if (nxtemp > smoothSize)
					hitx = smoothSize;
				else
					hitx = (int)nxtemp;
				
				nxtemp = (pointx - res*widthx - *smoothx)/smoothstepx;
				
				if (nxtemp < 0)
					lowtx = 0;
				else if (nxtemp > smoothSize)
					lowtx = smoothSize;
				else
					lowtx = (int)nxtemp;

				if (hitx != lowtx)
				{
					for (j = lowtx; j <= hitx; j++)
					{
						double xtemp = (smoothx[j] - pointx)/widthx;
						double intemp = 1.0-xtemp*xtemp/res/res;
						double widthtemp = (intemp < 0.0) ? 0.0 : res*widthy*sqrt(intemp);
						nytemp = (pointy + widthtemp - *smoothy)/smoothstepy+1;
						if (nytemp < 0)
							hity = 0;
						else if (nytemp > smoothSize)
							hity = smoothSize;
						else
							hity = (int)nytemp;
						
						nytemp = (pointy - widthtemp - *smoothy)/smoothstepy;
						if (nytemp < 0)
							lowty = 0;
						else if (nytemp > smoothSize)
							lowty = smoothSize;
						else
							lowty = (int)nytemp;
						
						if (hity != lowty)
						{
							for (k = lowty; k <= hity; k++)
							{
								double ytemp = (smoothy[k] - pointy)/widthy;
								smoothz[j][k] += mul*pointz*exp(-(xtemp*xtemp + ytemp*ytemp)/2.0)/2.0/PI/widthx/widthy;
							}
						}
					}
				}
				
				nxtemp = (pointx + res*widthx - *smoothfx)/smoothfstepx+1;
				if (nxtemp < 0)
					hitx = 0;
				else if (nxtemp > xfNs)
					hitx = xfNs;
				else
					hitx = (int)nxtemp;
				
				nxtemp = (pointx - res*widthx - *smoothfx)/smoothfstepx;
				if (nxtemp < 0)
					lowtx = 0;
				else if (nxtemp > xfNs)
					lowtx = xfNs;
				else
					lowtx = (int)nxtemp;

				if (hitx != lowtx)
				{
					for (j = lowtx; j <= hitx; j++)
					{
						double xtemp = (smoothfx[j] - pointx)/widthx;
						double intemp = 1.0-xtemp*xtemp/res/res;
						double widthtemp = (intemp < 0.0) ? 0.0 : res*widthy*sqrt(intemp);
						nytemp = (pointy + widthtemp - *smoothfy)/smoothfstepy+1;
						if (nytemp < 0)
							hity = 0;
						else if (nytemp > yfNs)
							hity = yfNs;
						else
							hity = (int)nytemp;
						
						nytemp = (pointy - widthtemp - *smoothfy)/smoothfstepy;
						if (nytemp < 0)
							lowty = 0;
						else if (nytemp > yfNs)
							lowty = yfNs;
						else
							lowty = (int)nytemp;
						
						if (hity != lowty)
						{
							for (k = lowty; k <= hity; k++)
							{
								double ytemp = (smoothfy[k] - pointy)/widthy;
								smoothfz[j][k] += mul*pointz*exp(-(xtemp*xtemp + ytemp*ytemp)/2.0)/2.0/PI/widthx/widthy;
							}
						}
					}
				}
				
			}
		}

		for (i = 0; i <= smoothSize; i++)
			for (j = 0; j <= smoothSize; j++)
				smoothz[i][j] /= (facx(smoothx[i])*facy(smoothy[j]));
	}
	
	for (i = 0; i < xN; i++)
	{
		for (j = 0; j < yN; j++)
		{
			like[i][j] = temp[i][j]/facx((xl + (i + 0.5)*stepx))/facy((yl + (j + 0.5)*stepy));
		}
	}
	
	hi = like[0][0];
	for (i = 0; i < xN; i++)
	{
		for (j = 0; j < yN; j++)
		{
			if(like[i][j] > hi)
			{
				hi = like[i][j];
			}
		}
	}
	
	if (flag&SMOOTH)
	{
		int repeat = 1;
		smoothHi = smoothz[0][0];
		smoothHix = smoothx[0];
		smoothHiy = smoothy[0];
		for (i = 0; i <= smoothSize; i++)
		{
			for (j = 0; j <= smoothSize; j++)
			{
				if(smoothz[i][j] > smoothHi)
				{
					smoothHi = smoothz[i][j];
					smoothHix = smoothx[i];
					smoothHiy = smoothy[j];
					repeat = 1;
				}
				else if (smoothz[i][j] == smoothHi)
				{
					repeat++;
					smoothHix += smoothx[i];
					smoothHiy += smoothy[j];
				}
			}
		}
		smoothHix /= repeat;
		smoothHiy /= repeat;
	}
	
	ofstream outg((name+string(".z")).c_str());
	ofstream outx((name+string(".x")).c_str());
	ofstream outy((name+string(".y")).c_str());
	for (i = 0; i < xN; i++)
	{
		outx << ((xl + (i + 0.5)*stepx)) << "   ";
	}
	outx << endl;
	for (j = 0; j < yN; j++)
	{
		outy << ((yl + (j + 0.5)*stepy)) <<"   ";
	}
	outy << endl;
	
	if (hi == 0)
	{
		cout << "Hist2D:  no points within boundaries, returning junk." << endl;
		outg << "Hist2D:  no points within boundaries, returning junk." << endl;
	}
	else
	{
		if (flag&SMOOTH)
		{
			int tot = 0;
			double totSoFar = 0;
			int totNum = (xfNs+1)*(yfNs+1);
			const int linesNum = 3;
			double *lsort = new double[totNum];
			double *psort = new double[totNum];
			double per[] = {0.6827, 0.9, 0.9545};
			
			for (i = 0; i < xfN; i++)
			{
				for (j = 0; j < yfN; j++)
				{
					tot += (int)ftemp[i][j];
				}
			}
			k = 0;
			for (i = 0; i <= xfNs; i++)
			{
				for (j = 0; j <= yfNs; j++)
				{
					lsort[k] = smoothfz[i][j]/facx(smoothfx[i])/facy(smoothfy[j]);
					psort[k++] = smoothfz[i][j]*smoothfstepx*smoothfstepy;
				}
			}
			Sort(lsort, psort, totNum);
			j=0;
			cout << "Info for " << name << ":" << endl;
			cout << "\tHi point:  (" << gx(smoothHix) << ", " << gy(smoothHiy) << ")" << endl;
			for (i = totNum-1; i > 0; i--)
			{
				totSoFar += psort[i];
				if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
				{
					double sig = lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i];
					cout << "\t" << (per[j]*100.0) << "%:   " << sig/smoothHi << endl;
					j++;
					if (j == linesNum)
						break;
				}
			}
			if(j < linesNum)
			{
				totSoFar += psort[i];
				if(totSoFar/tot >= per[j])
				{
					double sig = lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i];
					cout << (per[j]*100.0) << "%:   " << sig/smoothHi << endl;
				}
			}
			delete[] lsort;
			delete[] psort;
		}
		else
		{
			int tot = 0;
			double totSoFar = 0;
			int totNum = xfN*yfN;
			const int linesNum = 3;
			double *lsort = new double[totNum];
			double *psort = new double[totNum];
			double per[] = {0.6827, 0.9, 0.9545};
			
			k = 0;
			for (i = 0; i < xfN; i++)
			{
				for (j = 0; j < yfN; j++)
				{
					lsort[k] = ftemp[i][j]/facx((xfl + (i + 0.5)*stepx))/facy((yfl + (j + 0.5)*stepy));
					tot += (int)(psort[k++] = ftemp[i][j]);
				}
			}
			Sort(lsort, psort, totNum);
			j=0;
			cout << "sigmas for " << name << ":" << endl;
			for (i = totNum-1; i > 0; i--)
			{
				totSoFar += psort[i];
				if((lsort[i] != lsort[i-1])&&(totSoFar/tot >= per[j]))
				{
					double sig = lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i];
					cout << (per[j]*100.0) << "%:   " << sig/hi << endl;
					j++;
					if (j == linesNum)
						break;
				}
			}
			if(j < linesNum)
			{
				totSoFar += psort[i];
				if(totSoFar/tot >= per[j])
				{
					double sig = lsort[i] + (lsort[i+1]-lsort[i])*(totSoFar - per[j]*tot)/psort[i];
					cout << (per[j]*100.0) << "%:   " << sig/hi << endl;
				}
			}
			delete[] lsort;
			delete[] psort;
		}
		
		if(flag&INV)
		{
			for (i = 0; i < yN; i++)
			{
				for (j = 0; j < xN; j++)
				{
					outg << like[j][i]/hi << "   ";
				}
				outg << endl;
			}
		}
		else if (flag&COL)
		{
			for (i = 0; i < xN; i++)
			{
				for (j = 0; j < yN; j++)
				{
					outg << pow(10.0, xl + (i+0.5)*stepx) << "   " << pow(10.0, yl + (j+0.5)*stepy) << "   " << like[i][j]/hi << endl;
				}
			}
		}
		else
		{
			for (i = 0; i < xN; i++)
			{
				for (j = 0; j < yN; j++)
				{
					outg << like[i][j]/hi << "   ";
				}
				outg << endl;
			}
		}
		
		if (flag&SMOOTH)
		{
			ofstream outsg((name+string(".dat")).c_str());
			ofstream outsx((name+string(".x")).c_str());
			ofstream outsy((name+string(".y")).c_str());
			
			for (i = 0; i <= smoothSize; i++)
			{
				outsx << (smoothx[i]) << "   ";
			}
			outsx << endl;
			
			for (j = 0; j <= smoothSize; j++)
			{
				outsy << (smoothy[j]) << "   ";
			}
			outsy << endl;
			
			for (i = 0; i <= smoothSize; i++)
			{
				for (j = 0; j <= smoothSize; j++)
				{
					outsg << smoothz[i][j]/smoothHi << "   ";
				}
				outsg << endl;
			}
		}
	}
	
	del <double> (like, xN);
	del <double> (temp, xN);
	del <double> (ftemp, xfN);
	if (flag&SMOOTH)
	{
		del <double> (smoothx);
		del <double> (smoothy);
		del <double> (smoothz, smoothSize+1);
		del <double> (smoothfx);
		del <double> (smoothfy);
		del <double> (smoothfz, xfNs);
	}
	
	return;
}

double McmcEval::cl(const double a, const int iin, const char flag)
{
	double (*fx)(double);
	int f, i;
	
	if (flag&LOG) fx = log10;
	else fx = dummy;
	
	double tot = 0.0;
	double *sort = new double[totPts];
	double *wsort = new double[totPts];
	double *sortptr = sort;
	double *wsortptr = wsort;
	for (f = 0; f < numOfFiles; f++)
	{
		for (i = cut[f]; i < numOfPoints[f]; i++)
		{
			*(sortptr++) = fx(points[f][i][iin]);
			*(wsortptr++) = mults[f][i];
			tot += mults[f][i];
		}
	}
	Sort(sort, wsort, totPts);
	double totsofar = 0.0;
	for (i = 0; i < totPts; i++)
	{
		totsofar += wsort[i];
		if (a <= totsofar/tot)
		{
			return sort[i] + (sort[i-1] - sort[i])*(totsofar - a*tot)/wsort[i];
			//cout << a*100.0 << "%:  " << ans << endl;
		}
	}
	delete[] sort;
	delete[] wsort;
	return 0;
}

double McmcEval::derived_cl(const double a, const char flag)
{
	double (*fx)(double);
	int f, i;
	
	if (flag&LOG) fx = log10;
	else fx = dummy;
	
	double tot = 0.0;
	double *sort = new double[totPts];
	double *wsort = new double[totPts];
	for (i=0; i < totPts; i++)
	{
		sort[i] = fx(derived_param[i]);
		wsort[i] = derived_mults[i];
		tot += derived_mults[i];

	}
	Sort(sort, wsort, totPts);
	double totsofar = 0.0;
	for (i = 0; i < totPts; i++)
	{
		totsofar += wsort[i];
		if (a <= totsofar/tot)
		{
			return sort[i] + (sort[i-1] - sort[i])*(totsofar - a*tot)/wsort[i];
			//cout << a*100.0 << "%:  " << ans << endl;
		}
	}
	delete[] sort;
	delete[] wsort;
	return 0;
}

void McmcEval::Sort(double *arr, const int n)
{
	const int M=7,NSTACK=64;
	int i,ir,j,k,jstack=-1,l=0;
	double a, temp;
	int *istack = new int[NSTACK];

	ir=n-1;
	for (;;) 
	{
		if (ir-l < M) 
		{
			for (j=l+1;j<=ir;j++) 
			{
				a=arr[j];
				for (i=j-1;i>=l;i--) 
				{
					if (arr[i] <= a) break;
					arr[i+1]=arr[i];
				}
				arr[i+1]=a;
			}
			if (jstack < 0) break;
			ir=istack[jstack--];
			l=istack[jstack--];
		} else 
		{
			k=(l+ir) >> 1;
			SWAP(arr[k],arr[l+1]);
			if (arr[l] > arr[ir]) 
			{
				SWAP(arr[l],arr[ir]);
			}
			if (arr[l+1] > arr[ir]) 
			{
				SWAP(arr[l+1],arr[ir]);
			}
			if (arr[l] > arr[l+1]) 
			{
				SWAP(arr[l],arr[l+1]);
			}
			i=l+1;
			j=ir;
			a=arr[l+1];
			for (;;) 
			{
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAP(arr[i],arr[j]);
			}
			arr[l+1]=arr[j];
			arr[j]=a;
			jstack += 2;
			if (jstack >= NSTACK) 
				cout << "NSTACK too small in sort." << endl;
			if (ir-i+1 >= j-l) 
			{
				istack[jstack]=ir;
				istack[jstack-1]=i;
				ir=j-1;
			} 
			else 
			{
				istack[jstack]=j-1;
				istack[jstack-1]=l;
				l=i;
			}
		}
	}
	
	delete[] istack;
}

void McmcEval::Sort(double *arr, double *brr, const int n)
{
	const int M=7,NSTACK=50;
	int i,ir,j,k,jstack=-1,l=0;
	double a,b,temp;
	int *istack = new int[NSTACK];

	ir=n-1;
	for (;;) {
		if (ir-l < M) {
			for (j=l+1;j<=ir;j++) {
				a=arr[j];
				b=brr[j];
				for (i=j-1;i>=l;i--) {
					if (arr[i] <= a) break;
					arr[i+1]=arr[i];
					brr[i+1]=brr[i];
				}
				arr[i+1]=a;
				brr[i+1]=b;
			}
			if (jstack < 0) break;
			ir=istack[jstack--];
			l=istack[jstack--];
		} else {
			k=(l+ir) >> 1;
			SWAP(arr[k],arr[l+1]);
			SWAP(brr[k],brr[l+1]);
			if (arr[l] > arr[ir]) {
				SWAP(arr[l],arr[ir]);
				SWAP(brr[l],brr[ir]);
			}
			if (arr[l+1] > arr[ir]) {
				SWAP(arr[l+1],arr[ir]);
				SWAP(brr[l+1],brr[ir]);
			}
			if (arr[l] > arr[l+1]) {
				SWAP(arr[l],arr[l+1]);
				SWAP(brr[l],brr[l+1]);
			}
			i=l+1;
			j=ir;
			a=arr[l+1];
			b=brr[l+1];
			for (;;) {
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAP(arr[i],arr[j]);
				SWAP(brr[i],brr[j]);
			}
			arr[l+1]=arr[j];
			arr[j]=a;
			brr[l+1]=brr[j];
			brr[j]=b;
			jstack += 2;
			if (jstack >= NSTACK) cout << "NSTACK too small in sort2." << endl;
			if (ir-i+1 >= j-l) {
				istack[jstack]=ir;
				istack[jstack-1]=i;
				ir=j-1;
			} else {
				istack[jstack]=j-1;
				istack[jstack-1]=l;
				l=i;
			}
		}
	}
	
	delete[] istack;
}

McmcEval::~McmcEval()
{
	for (int i = 0; i < numOfFiles; i++)
		del <double> (points[i], numOfPoints[i]);
 	if (points != NULL) delete[] points;
 	if (numOfPoints != NULL) delete[] numOfPoints;
 	if (cut != NULL) delete[] cut;
	if (param_transforms != NULL) delete[] param_transforms;
	if (minvals != NULL) delete[] minvals;
	if (maxvals != NULL) delete[] maxvals;
	if (derived_param != NULL) delete[] derived_param;
	if (derived_mults != NULL) delete[] derived_mults;
	if (mults != NULL) {
		for (int i=0; i < numOfFiles; i++) delete[] mults[i];
		delete[] mults;
	}
	if (chi2 != NULL) {
		for (int i=0; i < numOfFiles; i++) delete[] chi2[i];
		delete[] chi2;
	}
}

void FisherEval::input(const char *file_root, const bool silent)
{
	ifstream bf_file((string(file_root) + ".bf").c_str());
	string str, str2;
	getline(bf_file, str);
	bf_file.close();
	istringstream iss(str);
	istringstream iss2(str);
	int a = 0;
	while(iss2 >> str2) a++;
	if (a==0) {
		numOfParam = 0;
		cerr << "Error: cannot read data file '" << string(file_root) + ".bf'" << endl;
		return;
	}
	numOfParam = a;
	if (!silent) cout << "Number of parameters: " << a << endl;

	int i,j;
	bestfitpt = new double[numOfParam];
	for (i=0; i < numOfParam; i++) iss >> bestfitpt[i];
	pcov = new double*[numOfParam];
	for (i=0; i < numOfParam; i++) pcov[i] = new double[numOfParam];

	ifstream pcov_file((string(file_root) + ".pcov").c_str());
	for (i=0; i < numOfParam; i++) {
		for (j=0; j < numOfParam; j++) {
			pcov_file >> pcov[i][j];
			//cout << pcov[i][j] << " ";
		}
		//cout << endl;
	}
	pcov_file.close();
}

void FisherEval::MkDist(const int N, const char *name, const int iin)
{
	double sig, mean, x, xmin, xmax, xstep;
	sig = sqrt(abs(pcov[iin][iin]));
	mean = bestfitpt[iin];
	xmin = mean - 3.5*sig;
	xmax = mean + 3.5*sig;
	xstep = (xmax-xmin)/(N-1);

	double distval;
	int i;
	ofstream out(name);
	for (i=0, x=xmin; i < N; i++, x += xstep)
	{
		distval = exp(-0.5*SQR((x - mean)/sig));
		out << x << " " << distval << endl;
	}
}

void FisherEval::MkDist2D(const int xN, const int yN, const char *name, const int iin, const int jin)
{
	double pcov_submatrix[2][2], fisher_submatrix[2][2];
	pcov_submatrix[0][0] = pcov[iin][iin];
	pcov_submatrix[1][1] = pcov[jin][jin];
	pcov_submatrix[0][1] = pcov[iin][jin];
	pcov_submatrix[1][0] = pcov[jin][iin]; // this should be identical to the previous off-diagonal
	double det = pcov_submatrix[0][0]*pcov_submatrix[1][1] - pcov_submatrix[0][1]*pcov_submatrix[1][0];
	fisher_submatrix[0][0] = pcov_submatrix[1][1] / det;
	fisher_submatrix[1][1] = pcov_submatrix[0][0] / det;
	fisher_submatrix[0][1] = -pcov_submatrix[1][0] / det;
	fisher_submatrix[1][0] = -pcov_submatrix[0][1] / det;

	double sigx, xmean, x, xmin, xmax, xstep;
	sigx = sqrt(abs(pcov_submatrix[0][0]));
	xmean = bestfitpt[iin];
	xmin = xmean - 3.5*sigx;
	xmax = xmean + 3.5*sigx;
	xstep = (xmax-xmin)/(xN-1);

	double sigy, ymean, y, ymin, ymax, ystep;
	sigy = sqrt(abs(pcov_submatrix[1][1]));
	ymean = bestfitpt[jin];
	ymin = ymean - 3.5*sigy;
	ymax = ymean + 3.5*sigy;
	ystep = (ymax-ymin)/(yN-1);

	int i,j;
	double distval, neg2logdist;
	ofstream outg, outx, outy;
	outg.open(name);
	outx.open((name+string("_x")).c_str());
	outy.open((name+string("_y")).c_str());
	for (i=0, x=xmin; i < xN; i++, x += xstep) outx << x << endl;
	for (i=0, y=ymin; i < yN; i++, y += ystep) outy << y << endl;

	for (j=0, y=ymin; j < yN; j++, y += ystep) {
		for (i=0, x=xmin; i < xN; i++, x += xstep) {
			neg2logdist = fisher_submatrix[0][0]*SQR(x-xmean) + fisher_submatrix[1][1]*SQR(y-ymean) + (fisher_submatrix[0][1] + fisher_submatrix[1][0])*(x-xmean)*(y-ymean);
			distval = exp(-0.5*neg2logdist);
			outg << distval << "   ";
		}
		outg << endl;
	}

	const int linesNum = 2;
	double per[] = {0.6827, 0.9545};
	ofstream outcont((name+string("_cont")).c_str());
	for (i = linesNum-1; i >= 0; i--)
	{
		outcont << 1 - per[i] << "   ";
	}
	outcont << endl;
}

FisherEval::~FisherEval()
{
	if (bestfitpt != NULL) delete[] bestfitpt;
	if (pcov != NULL) {
		for (int i=0; i < numOfParam; i++) delete[] pcov[i];
		delete[] pcov;
	}
}

