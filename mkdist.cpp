// Flag options for MkHist and MkHist2D:

// HIST:  show histogram
// SMOOTH: show smooth distribution
// INV: invert x-y (which is exactly normal) in z file
// COL: put 2-D posteriors in one file (instead of 3)
// MULT: file has multiplicities in first column
// LIKE:  file has likelihood (chi2) in last column.
// SIGS: show 68, 95 percent cl.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include "mcmceval.h"
#include "errors.h"
#include <sys/stat.h>
using namespace std;

void usage_error();
char *advance(char *p);
bool file_exists(const string &filename);

int main(int argc, char *argv[])
{
	bool specify_nthreads = false;
	bool specify_processes = false;
	bool make_1d_posts = false;
	bool make_2d_posts = false;
	bool output_min_chisq_point = false;
	bool output_mean_and_errors = false;
	bool make_derived_posterior = false;
	bool plot_mass_profile_constraints = false;
	bool run_python_script = false;
	bool transform_parameters = false;
	bool use_fisher_matrix = false;
	char mprofile_name[100] = "mprofile.dat";
	char param_transform_filename[100] = "";
	bool smoothing = false;
	int nthreads=1, n_processes=1;
	double radius = 0.1;
	string file_root, file_label;
	int nparams;
	int nbins=60, nbins_2d=40;
	bool silent = false;
	bool include_shading = true;
	double threshold = 3e-3;
	if (argc < 2) {
		cerr << "Error: must enter at least one argument (file_root)\n";
		usage_error();
		return 0;
	}

	// if cut is not assigned by the user, McmcEval will cut the first 10% of points in each chain (unless it's
	// (it's a nested sampling run, in which case cut = 0 since there is no burn-in phase that case)
	int cut = -1;
	stringstream str1;
	stringstream str2;
	stringstream str3;
	str1 << argv[1];
	if (!(str1 >> file_label)) {
		"Error: invalid argument (file_label)\n";
		usage_error();
		return 0;
	}
	string output_dir = "chains_" + file_label;
	struct stat sb;
	stat(output_dir.c_str(),&sb);
	if (S_ISDIR(sb.st_mode)==false) output_dir = ".";

	int i,j,c;
	for (i=2; i < argc; i++)   // Process extra command-line arguments
	{
		if ((*argv[i] == '-') and (isalpha(*(argv[i]+1)))) {
			while (c = *++argv[i]) {
				switch (c) {
					case 'b': output_min_chisq_point = true; break;
					case 'e': output_mean_and_errors = true; break; // this option also outputs the parameter covariance matrix
					case 'P': run_python_script = true; break;
					case 'f':
						use_fisher_matrix = true;
						make_1d_posts = true;
						make_2d_posts = true;
						break;
					case 'B':
						if (sscanf(argv[i], "B%lf", &threshold)==0) usage_error();
						argv[i] = advance(argv[i]);
						break;
					case 'T':
						if (sscanf(argv[i], "T:%s", param_transform_filename)==1)
							argv[i] += (1 + strlen(param_transform_filename));
						transform_parameters = true;
						argv[i] = advance(argv[i]);
						break;
					case 'D': // find posterior in a derived parameter, which is defined in the function DerivedParam(...) in mcmceval.cpp
						if (sscanf(argv[i], "D%lf", &radius)==0) usage_error();
						make_derived_posterior = true;
						argv[i] = advance(argv[i]);
						break;
					case 'M': // this option is specific to lensing
						if (sscanf(argv[i], "M:%s", mprofile_name)==1)
							argv[i] += (1 + strlen(mprofile_name));
						plot_mass_profile_constraints = true;
						argv[i] = advance(argv[i]);
						break;
					case 'd':
						char dirchar[100];
						if (sscanf(argv[i], "d:%s", dirchar)==1)
							argv[i] += (1 + strlen(dirchar));
						argv[i] = advance(argv[i]);
						output_dir.assign(dirchar);
						break;
					case 'n':
						int try_nbins;
						try_nbins = -10000;
						if (sscanf(argv[i], "n%i", &try_nbins) != 0) {
							if (try_nbins != -10000) {
								if ((try_nbins <= 0) or (try_nbins > 30000)) { cerr << "Error: invalid number of 1d bins (usage: -n#, where # is number of bins)\n"; return 0; }
								nbins = try_nbins;
								argv[i] = advance(argv[i]);
							}
						}
						make_1d_posts = true;
						break;
					case 'N':
						int try_nbins_2d;
						try_nbins_2d = -10000;
						if (sscanf(argv[i], "N%i", &try_nbins_2d) != 0) {
							if (try_nbins_2d != -10000) {
								if ((try_nbins_2d <= 0) or (try_nbins_2d > 30000)) { cerr << "Error: invalid number of 2d bins (usage: -N#, where # is number of bins)\n"; return 0; }
								nbins_2d = try_nbins_2d;
								argv[i] = advance(argv[i]);
							}
						}
						make_2d_posts = true;
						break;
					case 'c':
						if (sscanf(argv[i], "c%i", &cut)==0) usage_error();
						argv[i] = advance(argv[i]);
						break;
					case 't':
						if (sscanf(argv[i], "t%i", &nthreads)==0) usage_error();
						specify_nthreads = true;
						argv[i] = advance(argv[i]);
						break;
					case 'p':
						if (sscanf(argv[i], "p%i", &n_processes)==0) usage_error();
						specify_processes = true;
						argv[i] = advance(argv[i]);
						break;
					case 'q': silent = true; break;
					case 's': include_shading = false; break;
					case 'S': smoothing = true; break;
					default: usage_error(); return 0; break;
				}
			}
		} else { usage_error(); return 0; }
	}

	file_root = output_dir + "/" + file_label;
	i=0;
	string filename, istring;
	if (!use_fisher_matrix) {
		for(;;)
		{
			stringstream istream;
			istream << i;
			istream >> istring;
			filename = file_root + "_0." + istring;
			if (file_exists(filename)) i++;
			else break;
		}
		if (i==0) {
			for(;;) {
				stringstream istream;
				istream << i;
				istream >> istring;
				filename = file_root + "_" + istring;
				if (file_exists(filename)) i++;
				else break;
			}
			if (i==0) {
				if (file_exists(file_root)) {
					// in this case the "chain" data is from nested sampling, and no cut needs to be made
					if (cut == -1) cut = 0;
					i++;
				}
				else die("No data files found");
			}
			if (!specify_nthreads) nthreads = i;
		} else {
			if (!specify_processes) n_processes = i;
			i = 0;
			for (;;) {
				stringstream istream;
				istream << i;
				istream >> istring;
				filename = file_root + "_" + istring + ".0";
				if (file_exists(filename)) i++;
				else break;
			}
			if (i==0) {
				if (file_exists(file_root)) {
					i++;
				}
				else die("No data files found");
			}
			if (!specify_nthreads) nthreads = i;
		}
	} else {
		filename = file_root + ".pcov";
		if (!file_exists(filename)) die("Inverse-Fisher matrix file not found");
	}

	McmcEval Eval;
	FisherEval FEval;

	if (use_fisher_matrix) {
		FEval.input(file_root.c_str(),silent);
		FEval.get_nparams(nparams);
	}
	else
	{
		Eval.input(file_root.c_str(),-1,nthreads,NULL,NULL,n_processes,cut,MULT|LIKE,silent,transform_parameters,param_transform_filename);
		Eval.get_nparams(nparams);
	}
	if (nparams==0) die();

	string *param_names = new string[nparams];
	string paramnames_filename = file_root + ".paramnames";
	ifstream paramnames_file(paramnames_filename.c_str());
	for (i=0; i < nparams; i++) {
		if (!(paramnames_file >> param_names[i])) die("not all parameter names are given in file '%s'",paramnames_filename.c_str());
	}
	paramnames_file.close();

	if (!use_fisher_matrix) Eval.transform_parameter_names(param_names); // should have this option for the Fisher analysis version too

	string out_paramnames_filename = file_root + ".py_paramnames";
	ofstream paramnames_out(out_paramnames_filename.c_str());
	for (i=0; i < nparams; i++) {
		paramnames_out << param_names[i] << endl;
	}
	paramnames_out.close();

	if (use_fisher_matrix) {
		if (make_1d_posts) {
			for (i=0; i < nparams; i++) {
				string dist_out;
				dist_out = file_root + "_p_" + param_names[i] + ".dat";
				FEval.MkDist(201, dist_out.c_str(), i);
			}
		}
		if (make_2d_posts) {
			for (i=0; i < nparams; i++) {
				for (j=i+1; j < nparams; j++) {
					string dist_out;
					dist_out = file_root + "_2D_" + param_names[j] + "_" + param_names[i];
					FEval.MkDist2D(61,61,dist_out.c_str(),i,j);
				}
			}
		}
	}
	else
	{
		Eval.transform_parameter_names(param_names);

		double *minvals = new double[nparams];
		double *maxvals = new double[nparams];
		for (i=0; i < nparams; i++) {
			minvals[i] = -1e30;
			maxvals[i] = 1e30;
		}

		if (make_1d_posts) {
			Eval.FindRanges(minvals,maxvals,nbins,threshold);
			double rap[20];
			for (i=0; i < nparams; i++) {
				string hist_out;
				hist_out = file_root + "_p_" + param_names[i] + ".dat";
				if (smoothing) Eval.MkHist(minvals[i], maxvals[i], nbins, hist_out.c_str(), i, HIST|SMOOTH, rap);
				else Eval.MkHist(minvals[i], maxvals[i], nbins, hist_out.c_str(), i, HIST, rap);
			}
		}

		if (make_derived_posterior) {
			double rap[20];
			double mean, sig;
			Eval.setRadius(radius);
			Eval.calculate_derived_param();
			if (smoothing) Eval.DerivedHist(0, 1e30, nbins, (file_root + "_p_derived.dat").c_str(), mean, sig, HIST|SMOOTH, rap);
			else Eval.DerivedHist(0, 1e30, nbins, (file_root + "_p_derived.dat").c_str(), mean, sig, HIST, rap);
			double cl_l1,cl_l2,cl_h1,cl_h2;
			cl_l1 = Eval.derived_cl(0.02275);
			cl_l2 = Eval.derived_cl(0.15865);
			cl_h1 = Eval.derived_cl(0.84135);
			cl_h2 = Eval.derived_cl(0.97725);
			double center,sigma;
			// NOTE: You need to enforce boundaries in FindDerivedSigs, otherwise outlier points will screw up the derived confidence limits
			Eval.FindDerivedSigs(center,sigma);
			cout << "Confidence limits: " << cl_l1 << " " << cl_l2 << " " << cl_h1 << " " << cl_h2 << endl;
			cout << "Sig: " << center << " " << sigma << endl;
		}

		if (plot_mass_profile_constraints) {
			double rap[20];
			//double mean, sig;
			int n_rpts = 100;
			double r, rmin, rmax, rstep;
			rmin = 0.001;
			rmax = 9.332543;
			//rstep = (rmax-rmin)/(n_rpts-1);
			rstep = pow(rmax/rmin,1.0/(n_rpts-1));

			double mass, mass_low, mass_high;
			ofstream mpfile(mprofile_name);
			for (i=0, r=rmin; i < n_rpts; i++, r *= rstep) {
				Eval.setRadius(r);
				Eval.calculate_derived_param();
				mass = Eval.minchisq_derived_param();
				mass_low = Eval.derived_cl(0.02275);
				mass_high = Eval.derived_cl(0.97725);
				//Eval.DerivedHist(0, 1e30, nbins, (file_root + "_p_derived.dat").c_str(), mean, sig, HIST|SMOOTH, rap);
				mpfile << r << " " << mass << " " << mass_low << " " << mass_high << endl;
				//cout << "center = " << mean << ", 2*sig = " << 2*sig << ", fractional error = " << 2*sig/mean << endl;
			}
			mpfile.close();
		}

		if (make_2d_posts) {
			Eval.FindRanges(minvals,maxvals,nbins_2d,threshold);
			for (i=0; i < nparams; i++) {
				for (j=i+1; j < nparams; j++) {
					string hist_out;
					hist_out = file_root + "_2D_" + param_names[j] + "_" + param_names[i];
					Eval.MkHist2D(minvals[i],maxvals[i],minvals[j],maxvals[j],nbins_2d,nbins_2d,hist_out.c_str(),i,j, SMOOTH);
				}
			}
		}
		if (output_min_chisq_point) {
			Eval.output_min_chisq_pt();
		}

		if (output_mean_and_errors) {
			Eval.FindRanges(minvals,maxvals,nbins,threshold);
			string covar_out = file_root + ".cov";
			double *centers = new double[nparams];
			double *sigs = new double[nparams];
			Eval.FindCoVar(covar_out.c_str(),centers,sigs,minvals,maxvals);
			for (i=0; i < nparams; i++) {
				// NOTE: The following errors are from standard deviation, not from CL's 
				cout << param_names[i] << ": " << centers[i] << " +/- " << sigs[i] << endl;
			}
			delete[] centers;
			delete[] sigs;
		}
		delete[] minvals;
		delete[] maxvals;
	}

	if (make_1d_posts)
	{
		string pyname = file_label + ".py";
		ofstream pyscript(pyname.c_str());
		pyscript << "import GetDistPlots, os" << endl;
		pyscript << "g=GetDistPlots.GetDistPlotter('" << output_dir << "/')" << endl;
		pyscript << "g.settings.setWithSubplotSize(3.0000)" << endl;
		pyscript << "outdir=''" << endl;
		pyscript << "roots=['" << file_label << "']" << endl;
		pyscript << "g.plots_1d(roots)" << endl;
		pyscript << "g.export(os.path.join(outdir,'" << file_label << ".pdf'))" << endl;
		pyscript.close();
		if (run_python_script) {
			string pycommand = "python " + pyname;
			if (system(pycommand.c_str()) == 0) {
				cout << "Plot for 1D posteriors saved to '" << file_label << ".pdf'\n";
				string rmcommand = "rm " + pyname;
				system(rmcommand.c_str());
			}
			else cout << "Error: Could not generate PDF file for 1D posteriors\n";
		} else {
			cout << "Plotting script for 1D posteriors saved to '" << pyname << "'\n";
		}
	}


	if (make_2d_posts)
	{
		string pyname = file_label + "_2D.py";
		ofstream pyscript2d(pyname.c_str());
		pyscript2d << "import GetDistPlots, os" << endl;
		pyscript2d << "g=GetDistPlots.GetDistPlotter('" << output_dir << "/')" << endl;
		pyscript2d << "g.settings.setWithSubplotSize(3.0000)" << endl;
		pyscript2d << "outdir=''" << endl;
		pyscript2d << "roots=['" << file_label << "']" << endl;
		pyscript2d << "pairs=[]" << endl;
		for (i=0; i < nparams; i++) {
			for (j=i+1; j < nparams; j++)
				pyscript2d << "pairs.append(['" << param_names[i] << "','" << param_names[j] << "'])\n";
		}
		pyscript2d << "g.plots_2d(roots,param_pairs=pairs,";
		if (include_shading) pyscript2d << "shaded=True";
		else pyscript2d << "shaded=False";
		pyscript2d << ")" << endl;
		pyscript2d << "g.export(os.path.join(outdir,'" << file_label << "_2D.pdf'))" << endl;
		if (run_python_script) {
			string pycommand = "python " + pyname;
			if (system(pycommand.c_str()) == 0) {
				cout << "Plot for 2D posteriors saved to '" << file_label << ".pdf'\n";
				string rmcommand = "rm " + pyname;
				system(rmcommand.c_str());
			}
			else cout << "Error: Could not generate PDF file for 2D posteriors\n";
		} else {
			cout << "Plotting script for 2D posteriors saved to '" << pyname << "'\n";
		}


		if (make_1d_posts) {
			// make script for triangle plot
			pyname = file_label + "_tri.py";
			ofstream pyscript(pyname.c_str());
			pyscript << "import GetDistPlots, os" << endl;
			pyscript << "g=GetDistPlots.GetDistPlotter('" << output_dir << "/')" << endl;
			pyscript << "g.settings.setWithSubplotSize(3.0000)" << endl;
			pyscript << "outdir=''" << endl;
			pyscript << "roots=['" << file_label << "']" << endl;
			pyscript << "g.triangle_plot(roots, [";
			for (i=0; i < nparams; i++) {
				pyscript << "'" << param_names[i] << "'";
				if (i != nparams-1) pyscript << ",";
			}
			pyscript << "],";
			if (include_shading) pyscript << "shaded=True";
			else pyscript << "shaded=False";
			pyscript << ")" << endl;
			pyscript << "g.export(os.path.join(outdir,'" << file_label << "_tri.pdf'))" << endl;
			if (run_python_script) {
				string pycommand = "python " + pyname;
				if (system(pycommand.c_str()) == 0) {
					cout << "Triangle plot (1D+2D posteriors) saved to '" << file_label << ".pdf'\n";
					string rmcommand = "rm " + pyname;
					system(rmcommand.c_str());
				}
				else cout << "Error: Could not generate PDF file for triangle plot (1d + 2d posteriors)\n";
			} else {
				cout << "Plotting script for triangle plot saved to '" << pyname << "'\n";
			}
		}
	}

	delete[] param_names;
	return 0;
}

bool file_exists(const string &filename)
{
	ifstream infile(filename.c_str());
	return infile.good();
}

char *advance(char *p)
{
	// This advances to the next flag (if there is one; 'e' is ignored because it might be part of a number in scientific notation)
	while ((*++p) and ((!isalpha(*p)) or (*p=='e'))) ;
	return --p;
}

void usage_error()
{
	cerr << "Usage:\n\n";
	cerr << "mkdist <file_root> ...\n\n";
	cerr << "Argument options (after the first required argument):\n"
			"  -d:<dir> set input/output directory to <dir> (default: 'chains_<file_root>/')\n"
			"  -n#      make 1d histograms with # bins\n"
			"  -N#      make 2d histograms with # by # bins\n"
			"  -P       execute Python scripts generated by mkdist to output posteriors as PDF files\n"
			"  -S       plot smoothed histograms\n"
			"  -s       do not include shading in 2D histograms\n"
			"  -b       output minimum chi-square point\n"
			"  -e       output mean parameters with standard errors in each parameter\n"
			"  -p#      specify number of MPI processes involved in making the chains\n"
			"  -t#      read in # chains per process (if more than one chain involved)\n"
			"  -B#      input minimum probability threshold used for defining parameter ranges\n"
			"              for plotting (default = 3e-3; higher threshold --> smaller ranges)\n"
			"  -f       use Fisher matrix to generate 1d,2d posteriors (MCMC data not required)\n"
			"  -q       quiet mode (non-verbose)\n" << endl;
	exit(1);
}

