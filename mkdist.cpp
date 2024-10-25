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

#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

void usage_error(const int mpi_id);
void show_transform_usage();
char *advance(char *p);
bool file_exists(const string &filename);
void adjust_ranges_to_include_markers(double *minvals, double *maxvals, double *markers, const int nparams_eff);

int main(int argc, char *argv[])
{
	int mpi_np=1, mpi_id=0;

#ifdef USE_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
#endif

	bool make_1d_posts = false;
	bool make_2d_posts = false;
	bool output_min_chisq_point = false;
	bool output_min_chisq_point_format2 = false;
	bool output_mean_and_errors = false;
	bool output_cl = false;
	bool cl_2sigma = false;
	bool make_derived_posterior = false;
	bool plot_mass_profile_constraints = false;
	bool run_python_script = false;
	bool transform_parameters = false;
	bool importance_sampling = false;
	bool output_transform_usage = false;
	bool use_fisher_matrix = false;
	bool exclude_derived_params = false;
	bool output_chain_header = false;
	char mprofile_name[100] = "mprofile.dat";
	bool show_markers = false;
	bool print_marker_values = false;
	bool make_subplot = false;
	bool add_title = false;
	bool latex_table_format = false;
	bool show_prior_ranges = false;
	bool fixed_precision = false;
	bool suppress_latex_names = false;
	string marker_filename = "";
	int n_markers_allowed = 10000;
	char param_transform_filename[100] = "";
	char prior_weight_filename[100] = "";
	char label_addendum[100] = "";
	bool skip_hist2d = false; // Useful if 2d histograms have already been generated, but something needs tweaking (e.g. markers)
	bool smoothing = false;
	int n_threads=1, n_processes=1;
	double radius = 0.1;
	bool file_label_set = false;
	string file_root, file_label;
	int nparams, nparams_eff, n_fitparams = -1;
	int nparams_subset = -1;
	int nbins=60, nbins_2d=40;
	int precision = 13; // this is for fixed precision format when outputting parameter percentiles/errors
	bool silent = false;
	bool include_shading = true;
	double threshold = 3e-3;
	double percentile = 0.5;
	bool output_percentile = false;
	bool show_uncertainties_as_percentiles = false;
	bool marker_filename_specified = false;
	bool use_bestfit_markers = false;
	bool add_dummy_params = false; // this option might be useful if you are making columns in a latex table with parameter values etc.
	int ndummy = 0;
	if (argc < 2) {
		cerr << "Error: must enter at least one argument (file_root)\n";
		usage_error(mpi_id);
		return 0;
	}

	// if cut is not assigned by the user, McmcEval will cut the first 10% of points in each chain (unless it's
	// (it's a nested sampling run, in which case cut = 0 since there is no burn-in phase that case)
	int cut = -1;
	string output_dir = ".";
	double pct_scaling = 1; // used if one wants to scale the uncertainties by a given factor
	//string output_dir = "chains_" + file_label;
	//struct stat sb;
	//stat(output_dir.c_str(),&sb);
	//if (S_ISDIR(sb.st_mode)==false) output_dir = ".";

	string output_file_label;
	int i,j,c;
	for (i=1; i < argc; i++)   // Process extra command-line arguments
	{
		if ((*argv[i] == '-') and (isalpha(*(argv[i]+1)))) {
			while ((c = *++argv[i])) {
				switch (c) {
					case 'b': output_min_chisq_point = true; break;
					case 'O': output_min_chisq_point_format2 = true; break;
					case 'e': output_mean_and_errors = true; break; // this option also outputs the parameter covariance matrix
					case 'E':
						output_cl = true;
						if (*(argv[i]+1)=='2') { cl_2sigma = true; argv[i]++; }
						break;
					case 'u':
						show_uncertainties_as_percentiles = true;
						if (sscanf(argv[i], "u%lf", &pct_scaling)==1) {
							argv[i] = advance(argv[i]);
						}
						break;
					case 'P': run_python_script = true; break;
					case 'x': exclude_derived_params = true; break;
					case 'i': output_chain_header = true; break; // Deprecated
					case 'h': output_chain_header = true; break;
					case 'f':
						use_fisher_matrix = true;
						make_1d_posts = true;
						make_2d_posts = true;
						break;
					case 'B':
						if (sscanf(argv[i], "B%lf", &threshold)==0) usage_error(mpi_id);
						argv[i] = advance(argv[i]);
						break;
					case 'T':
						if (sscanf(argv[i], "T:%s", param_transform_filename)==1) {
							argv[i] += (1 + strlen(param_transform_filename));
							transform_parameters = true;
							argv[i] = advance(argv[i]);
						}
						else output_transform_usage = true;
						break;
					case 'I':
						if (sscanf(argv[i], "I:%s", prior_weight_filename)==1) {
							argv[i] += (1 + strlen(prior_weight_filename));
							importance_sampling = true;
							argv[i] = advance(argv[i]);
						}
						break;
					case 'l': latex_table_format = true; break;
					case 'L': suppress_latex_names = true; break;
					case 'D': // find posterior in a derived parameter, which is defined in the function DerivedParam(...) in mcmceval.cpp
						//if (sscanf(argv[i], "D%lf", &radius)==0) usage_error();
						make_derived_posterior = true;
						argv[i] = advance(argv[i]);
						break;
					//case 'M': // this option is specific to lensing
						//if (sscanf(argv[i], "M:%s", mprofile_name)==1)
							//argv[i] += (1 + strlen(mprofile_name));
						//plot_mass_profile_constraints = true;
						//argv[i] = advance(argv[i]);
						//break;
					case 'd':
						char dirchar[100];
						if (sscanf(argv[i], "d:%s", dirchar)==1)
							argv[i] += (1 + strlen(dirchar));
						argv[i] = advance(argv[i]);
						output_dir.assign(dirchar);
						break;
					case 'v': print_marker_values = true; break;
					case 'm':
						show_markers = true;
						char marker_filename_char[100];
						if (sscanf(argv[i], "m:%s", marker_filename_char)==1) {
							marker_filename_specified = true;
							argv[i] += (1 + strlen(marker_filename_char));
							marker_filename.assign(marker_filename_char);
							if (marker_filename=="bestfit") {
								marker_filename_specified = false;
								use_bestfit_markers = true;
							}
						}
						argv[i] = advance(argv[i]);
						break;
					case 'M':
						int try_nmark;
						try_nmark = -10000;
						if (sscanf(argv[i], "M%i", &try_nmark) != 0) {
							if (try_nmark != -10000) {
								if ((try_nmark <= 0) or (try_nmark > 30000)) { cerr << "Error: invalid number of parameter markers (usage: -M#, where # is number of markers)\n"; return 0; }
								n_markers_allowed = try_nmark;
								argv[i] = advance(argv[i]);
							}
						}
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
						if (sscanf(argv[i], "c%i", &cut)==0) usage_error(mpi_id);
						argv[i] = advance(argv[i]);
						break;
					case 'p':
						if (sscanf(argv[i], "p%lf", &percentile)==0) usage_error(mpi_id);
						if ((percentile <= 0) or (percentile >= 1.0)) die("percentile must be between 0 and 1");
						output_percentile = true;
						argv[i] = advance(argv[i]);
						break;
					case 'a':
						if (sscanf(argv[i], "a:%s", label_addendum)==1) {
							argv[i] += (1 + strlen(label_addendum));
							string addstring(label_addendum);
							output_file_label += "." + addstring;
							argv[i] = advance(argv[i]);
						}
						break;
					case 'q': silent = true; break;
					case 'r': skip_hist2d = true; break;
					case 'R': show_prior_ranges = true; break;
					case 's': make_subplot = true; break;
					case 't': add_title = true; break;
					//case 'F': include_shading = false; break;
					case 'F':
						fixed_precision = true;
						if (sscanf(argv[i], "F%i", &precision) != 0) {
							if ((precision < 1) or (precision > 16)) { cerr << "Error: invalid precision (usage: -F#, where # is between 1 and 16)\n"; return 0; }
							argv[i] = advance(argv[i]);
						}
						break;
					case 'g':
						add_dummy_params = true;
						if (sscanf(argv[i], "g%i", &ndummy) != 0) {
							argv[i] = advance(argv[i]);
						}
						break;
					case 'C':
						if (sscanf(argv[i], "C%i", &nparams_subset)==0) usage_error(mpi_id);
						argv[i] = advance(argv[i]);
						break;
					case 'S': smoothing = true; break;
					default: usage_error(mpi_id); return 0; break;
				}
			}
		} else {
			if (!file_label_set) {
				stringstream stri;
				stri << argv[i];
				if (!(stri >> file_label)) {
					cerr << "Error: invalid argument (file_label)\n";
					usage_error(mpi_id);
					return 0;
				}
				file_label_set = true;
				output_file_label = file_label;
			} else {
				usage_error(mpi_id);
				return 0;
			}
		}
	}

	if (output_transform_usage) show_transform_usage();

	file_root = output_dir + "/" + file_label;
	i=0;
	j=0;
	string filename, istring, jstring;
	if (!use_fisher_matrix) {
		for(;;)
		{
			stringstream jstream;
			jstream << j;
			jstream >> jstring;
			filename = file_root + "_0." + jstring;
			if (file_exists(filename)) j++;
			else break;
		}
		if (j==0) {
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
		} else {
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
		}
		if (cut != 0) {
			if (i > 0) n_threads = i; // set number of chains for MCMC data
			if (j > 0) n_processes = j; // set number of MPI processes that were used to produce MCMC data
		}
	} else {
		filename = file_root + ".pcov";
		if (!file_exists(filename)) die("Inverse-Fisher matrix file not found");
	}

	string nparam_filename = file_root + ".nparam";
	ifstream nparam_file(nparam_filename.c_str());
	if (nparam_file.good()) {
		nparam_file >> n_fitparams;
		nparam_file.close();
	}

	McmcEval Eval;
	FisherEval FEval;
	double logev = 1e30;

	if (use_fisher_matrix) {
		FEval.input(file_root.c_str(),silent);
		FEval.get_nparams(nparams);
	}
	else
	{
		bool mpi_silent = true;
		if (mpi_id==0) mpi_silent = silent;
		Eval.input(file_root.c_str(),-1,n_threads,NULL,NULL,logev,n_processes,cut,MULT|LIKE,mpi_silent,n_fitparams,transform_parameters,param_transform_filename,importance_sampling,prior_weight_filename);
		Eval.get_nparams(nparams);
	}
	if ((mpi_id==0) and (output_chain_header)) Eval.OutputChainHeader();
	if (nparams==0) die();
	nparams_eff = nparams;
	if (n_fitparams==-1) n_fitparams = nparams;
	if (exclude_derived_params) nparams_eff = n_fitparams;
	if (nparams_subset < nparams) {
		if (nparams_subset > 0) nparams_eff = nparams_subset;
		else if (nparams_subset == 0) warn("specified subset number of parameters is equal to or less than zero; using all parameters");
	}

	// Make it so you can turn parameters on/off in this file! This will require revising nparams_eff after the flags are read in
	string *param_names = new string[nparams];
	string paramnames_filename, dummy;
	paramnames_filename = file_root + ".paramnames";
	ifstream paramnames_file(paramnames_filename.c_str());
	for (i=0; i < nparams; i++) {
		if (!(paramnames_file >> param_names[i])) die("not all parameter names are given in file '%s'",paramnames_filename.c_str());
	}
	paramnames_file.close();

	string *latex_param_names = new string[nparams];
	if ((latex_table_format) or (make_1d_posts) or (make_2d_posts)) {
		string latex_paramnames_filename = file_root + ".latex_paramnames";
		ifstream latex_paramnames_file(latex_paramnames_filename.c_str());
		string dummy;
		const int n_characters = 1024;
		char line[n_characters];
		for (i=0; i < nparams; i++) {
			if (!(latex_paramnames_file.getline(line,n_characters))) die("not all parameter names are given in file '%s'",latex_paramnames_filename.c_str());
			istringstream instream(line);
			if (!(instream >> dummy)) die("not all parameter names are given in file '%s'",latex_paramnames_filename.c_str());
			if (!(instream >> latex_param_names[i])) die("not all latex parameter names are given in file '%s'",latex_paramnames_filename.c_str());
			while (instream >> dummy) latex_param_names[i] += " " + dummy;
		}
		latex_paramnames_file.close();
	}

	double *prior_minvals = new double[nparams];
	double *prior_maxvals = new double[nparams];
	for (i=0; i < nparams; i++) {
		prior_minvals[i] = -1e30;
		prior_maxvals[i] = 1e30;
	}
	string paramranges_filename = file_root + ".ranges";
	ifstream paramranges_file(paramranges_filename.c_str());
	if (paramranges_file.is_open()) {
		for (i=0; i < nparams; i++) {
			if (!(paramranges_file >> prior_minvals[i])) die("not all parameter ranges are given in file '%s'",paramranges_filename.c_str());
			if (!(paramranges_file >> prior_maxvals[i])) die("not all parameter ranges are given in file '%s'",paramranges_filename.c_str());
			if (prior_minvals[i] > prior_maxvals[i]) die("cannot have minimum parameter value greater than maximum parameter value in file '%s'",paramranges_filename.c_str());
		}
		paramranges_file.close();
	} else warn("parameter range file '%s' not found",paramranges_filename.c_str());

	if ((make_1d_posts) or (make_2d_posts)) {
		if (!use_fisher_matrix) Eval.transform_parameter_names(param_names, latex_param_names); // should have this option for the Fisher analysis version too

		if (mpi_id==0) {
			string out_paramnames_filename = file_root + ".py_paramnames";
			ofstream paramnames_out(out_paramnames_filename.c_str());
			for (i=0; i < nparams_eff; i++) {
				paramnames_out << param_names[i] << endl;
			}
			paramnames_out.close();

			string out_latex_paramnames_filename = file_root + ".py_latex_paramnames";
			ofstream latex_paramnames_out(out_latex_paramnames_filename.c_str());
			for (i=0; i < nparams_eff; i++) {
				latex_paramnames_out << param_names[i] << "   " << latex_param_names[i] << endl;
			}
			latex_paramnames_out.close();
		}
	}

	bool *hist2d_active_params = new bool[nparams_eff];
	for (i=0; i < nparams_eff; i++) {
		hist2d_active_params[i] = true;
	}
	if (make_2d_posts) {
		string hist2d_paramnames_filename = file_root + ".hist2d_params";
		if (file_exists(hist2d_paramnames_filename)) {
			string *hist2d_param_names = new string[nparams_eff];
			ifstream hist2d_paramnames_file(hist2d_paramnames_filename.c_str());
			for (i=0; i < nparams_eff; i++) {
				if (!(hist2d_paramnames_file >> hist2d_param_names[i])) die("not all hist2d_parameter names are given in file '%s'",hist2d_paramnames_filename.c_str());
				if (hist2d_param_names[i] != param_names[i]) die("hist2d parameter names do not match names given in paramnames file");
				int pflag;
				if (!(hist2d_paramnames_file >> pflag)) die("hist2d parameter flag not given in file '%s'",hist2d_paramnames_filename.c_str());
				if (pflag == 0) hist2d_active_params[i] = false;
				else if (pflag == 1) hist2d_active_params[i] = true;
				else die("invalid hist2d parameter flag in file '%s'; should either be 0 or 1",hist2d_paramnames_filename.c_str());
			}
			hist2d_paramnames_file.close();
			delete[] hist2d_param_names;
		}
	}
	int nparams_eff_2d = 0;
	for (i=nparams_eff-1; i >= 0; i--) {
		if (hist2d_active_params[i]) {
			nparams_eff_2d = i+1;
			break;
		}
	}

	bool *subplot_active_params = new bool[nparams_eff];
	if (make_subplot) {
		string *subplot_param_names = new string[nparams_eff];
		string subplot_paramnames_filename = file_root + ".subplot_params";
		ifstream subplot_paramnames_file(subplot_paramnames_filename.c_str());
		for (i=0; i < nparams_eff; i++) {
			if (!(subplot_paramnames_file >> subplot_param_names[i])) die("not all subplot_parameter names are given in file '%s'",subplot_paramnames_filename.c_str());
			if (subplot_param_names[i] != param_names[i]) die("subplot parameter names do not match names given in paramnames file");
			int pflag;
			if (!(subplot_paramnames_file >> pflag)) die("subplot parameter flag not given in file '%s'",subplot_paramnames_filename.c_str());
			if (pflag == 0) subplot_active_params[i] = false;
			else if (pflag == 1) subplot_active_params[i] = true;
			else die("invalid subplot parameter flag in file '%s'; should either be 0 or 1",subplot_paramnames_filename.c_str());
			if ((subplot_active_params[i]) and (!hist2d_active_params[i])) die("subplot parameter '%s' must also have the hist2d flag set to 'true' in <label>.hist2d_params",subplot_param_names[i].c_str());
		}
		subplot_paramnames_file.close();
		delete[] subplot_param_names;
	}
	string title = "";
	if (add_title) {
		string title_filename = file_root + ".plot_title";
		ifstream title_file(title_filename.c_str());
		getline(title_file, title);
	}	
	
	double *markers = new double[nparams_eff];
	int n_markers = (n_markers_allowed < nparams_eff ? n_markers_allowed : nparams_eff);
	if (show_markers) {
		if (use_bestfit_markers) {
			double *bestfit = new double[nparams];
			Eval.min_chisq_pt(bestfit);
			for (i=0; i < n_markers; i++) markers[i] = bestfit[i];
			delete[] bestfit;
		} else {
			if (!marker_filename_specified) marker_filename = file_root + ".markers";
			ifstream marker_file(marker_filename.c_str());
			for (i=0; i < n_markers; i++) {
				if (!(marker_file >> markers[i])) {
					if (i==0) {
						if (mpi_id==0) cerr << "marker values could not be read from file '" << marker_filename << "'; will not use markers when plotting" << endl;
						show_markers = false;
						break;
					}
					n_markers = i;
					break;
				}
			}
		}
	}

	if (use_fisher_matrix) {
		if (mpi_id==0) {
			if (make_1d_posts) {
				for (i=0; i < nparams_eff; i++) {
					string dist_out;
					dist_out = file_root + "_p_" + param_names[i] + ".dat";
					FEval.MkDist(201, dist_out.c_str(), i);
				}
			}
			if (make_2d_posts) {
				for (i=0; i < nparams_eff_2d; i++) {
					if (hist2d_active_params[i]) {
						for (j=i+1; j < nparams_eff_2d; j++) {
							if (hist2d_active_params[j]) {
								string dist_out;
								dist_out = file_root + "_2D_" + param_names[j] + "_" + param_names[i];
								FEval.MkDist2D(61,61,dist_out.c_str(),i,j);
							}
						}
					}
				}
			}
		}
	}
	else
	{
		//Eval.transform_parameter_names(param_names, latex_paramnames);

		double *minvals = new double[nparams];
		double *maxvals = new double[nparams];
		for (i=0; i < nparams; i++) {
			minvals[i] = -1e30;
			maxvals[i] = 1e30;
		}

		if ((make_1d_posts) and (mpi_id==0)) {
			Eval.FindRanges(minvals,maxvals,nbins,threshold);
			if (show_markers) adjust_ranges_to_include_markers(minvals,maxvals,markers,n_markers);
			double rap[20];
			for (i=0; i < nparams_eff; i++) {
				string hist_out;
				hist_out = file_root + "_p_" + param_names[i] + ".dat";
				if (smoothing) Eval.MkHist(minvals[i], maxvals[i], nbins, hist_out.c_str(), i, HIST|SMOOTH, rap);
				else Eval.MkHist(minvals[i], maxvals[i], nbins, hist_out.c_str(), i, HIST, rap);
			}
		}

		if ((make_derived_posterior) and (mpi_id==0)) {
			double rap[20];
			double mean, sig;
			//Eval.setRadius(radius);
			Eval.calculate_derived_param();
			if (smoothing) Eval.DerivedHist(-1e30, 1e30, nbins, (file_root + "_p_derived.dat").c_str(), mean, sig, HIST|SMOOTH, rap);
			else Eval.DerivedHist(-1e30, 1e30, nbins, (file_root + "_p_derived.dat").c_str(), mean, sig, HIST, rap);
			double cl_l1,cl_l2,cl_h1,cl_h2;
			cl_l1 = Eval.derived_cl(0.02275);
			cl_l2 = Eval.derived_cl(0.15865);
			cl_h1 = Eval.derived_cl(0.84135);
			cl_h2 = Eval.derived_cl(0.97725);
			double center,sigma;
			// NOTE: You need to enforce boundaries in FindDerivedSigs, otherwise outlier points will screw up the derived confidence limits
			//Eval.FindDerivedSigs(center,sigma);
			cout << "Confidence limits: " << cl_l1 << " " << cl_l2 << " " << cl_h1 << " " << cl_h2 << endl;
			//cout << "Sig: " << center << " " << sigma << endl;
		}

		/*
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
		*/

		if ((make_2d_posts) and (!skip_hist2d)) {
			int omp_nthreads = 1;
#ifdef USE_OPENMP
			double wtime, wtime0;
			#pragma omp parallel
			{
				#pragma omp master
				omp_nthreads = omp_get_num_threads();
			}
			wtime0 = omp_get_wtime();
#endif
			bool derived_param_fail = false; // if contours can't be made for a derived parameter, we'll have it drop the derived parameters and try again
			Eval.FindRanges(minvals,maxvals,nbins_2d,threshold);
			if ((!make_1d_posts) and (show_markers)) adjust_ranges_to_include_markers(minvals,maxvals,markers,n_markers);
			do {
				int k, n_2dposts;
				vector<int> post2d_i, post2d_j;
				for (i=0; i < nparams_eff_2d; i++) {
					if (hist2d_active_params[i]) {
						for (j=i+1; j < nparams_eff_2d; j++) {
							if (hist2d_active_params[j]) {
								post2d_i.push_back(i);
								post2d_j.push_back(j);
							}
						}
					}
				}
				n_2dposts = post2d_i.size();
				if (mpi_id==0) {
					cout << "Generating 2D histograms (total of " << n_2dposts << ") with ";
					if (mpi_np > 1) cout << mpi_np << " processes and ";
					cout << omp_nthreads << " threads..." << endl;
				}
#ifdef USE_OPENMP
				int omp_nthreads0=omp_nthreads;
				while (n_2dposts < mpi_np*omp_nthreads) {
					omp_nthreads--;
				}
				if (omp_nthreads < omp_nthreads0) {
					omp_set_num_threads(omp_nthreads);
					if (mpi_id==0) cout << "Too many threads, reducing number of threads to " << omp_nthreads << "..." << endl;
				}
#endif

				int mpi_chunk, mpi_start, mpi_end;
				mpi_chunk = n_2dposts / mpi_np;
				mpi_start = mpi_id*mpi_chunk;
				if (mpi_id == mpi_np-1) mpi_chunk += (n_2dposts % mpi_np); // assign the remainder elements to the last mpi process
				mpi_end = mpi_start + mpi_chunk;

				if (derived_param_fail) derived_param_fail = false;
				#pragma omp parallel for private(i,j,k) schedule(dynamic)
				//for (k=0; k < n_2dposts; k++) {
				for (k=mpi_start; k < mpi_end; k++) {
					i = post2d_i[k];
					j = post2d_j[k];
					string hist_out;
					hist_out = file_root + "_2D_" + param_names[j] + "_" + param_names[i];
					if (!Eval.MkHist2D(minvals[i],maxvals[i],minvals[j],maxvals[j],nbins_2d,nbins_2d,hist_out.c_str(),i,j, SMOOTH)) {
						if ((i>=n_fitparams) or (j>=n_fitparams)) {
							derived_param_fail = true;
							warn("producing contours failed for derived parameter; we will drop the derived parameters and try again");
						}
					}
					if (derived_param_fail) i = nparams_eff_2d + 1; // make sure it exits loop
				}

				/*
				if (derived_param_fail) derived_param_fail = false;
				#pragma omp parallel for private(i,j) schedule(dynamic)
				for (i=0; i < nparams_eff_2d; i++) {
					if (hist2d_active_params[i]) {
						for (j=i+1; j < nparams_eff_2d; j++) {
							if (hist2d_active_params[j]) {
								string hist_out;
								hist_out = file_root + "_2D_" + param_names[j] + "_" + param_names[i];
								if (!Eval.MkHist2D(minvals[i],maxvals[i],minvals[j],maxvals[j],nbins_2d,nbins_2d,hist_out.c_str(),i,j, SMOOTH)) {
									if ((i>=n_fitparams) or (j>=n_fitparams)) {
										derived_param_fail = true;
										warn("producing contours failed for derived parameter; we will drop the derived parameters and try again");
										break;
									}
								}
							}
						}
						if (derived_param_fail) i = nparams_eff_2d + 1; // make sure it exits loop
					}
				}
				*/

				if (derived_param_fail) nparams_eff_2d = n_fitparams;
			} while (derived_param_fail);
#ifdef USE_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef USE_OPENMP
			wtime = omp_get_wtime() - wtime0;
			if (mpi_id==0) cout << "Wall time for calculating 2D histograms: " << wtime << endl;
#endif
		}
		if (mpi_id==0) {
			if (output_min_chisq_point) {
				Eval.output_min_chisq_pt(param_names);
			}
			if (output_min_chisq_point_format2) {
				Eval.output_min_chisq_pt2(param_names);
			}

			if (output_mean_and_errors) {
				Eval.FindRanges(minvals,maxvals,nbins,threshold);
				string covar_out = file_root + ".cov";
				double *centers = new double[nparams];
				double *sigs = new double[nparams];
				Eval.FindCoVar(covar_out.c_str(),centers,sigs,minvals,maxvals);
				for (i=0; i < nparams_eff; i++) {
					// NOTE: The following errors are from standard deviation, not from CL's 
					cout << param_names[i] << ": " << centers[i] << " +/- " << sigs[i] << endl;
				}
				cout << endl;
				delete[] centers;
				delete[] sigs;
			}
			if (output_cl) {
				Eval.FindRanges(minvals,maxvals,nbins,threshold);
				double *halfpct = new double[nparams];
				double *lowcl = new double[nparams];
				double *hicl = new double[nparams];
				int powers_of_ten;
				if (!silent) {
					if (cl_2sigma) cout << "50th percentile values and errors (based on 2.5\% and 97.5\% percentiles of marginalized posteriors):\n\n";
					else cout << "50th percentile values and errors (based on 15.8\% and 84.1\% percentiles of marginalized posteriors):\n\n";
				}
				if (fixed_precision) {
					//cout << resetiosflags(ios::scientific);
					cout << setprecision(precision);
					cout << fixed;
				}
				for (i=0; i < nparams_eff; i++) {
					if (cl_2sigma) {
						lowcl[i] = Eval.cl(0.025,i,minvals[i],maxvals[i]);
						hicl[i] = Eval.cl(0.975,i,minvals[i],maxvals[i]);
					} else {
						lowcl[i] = Eval.cl(0.15865,i,minvals[i],maxvals[i]);
						hicl[i] = Eval.cl(0.84135,i,minvals[i],maxvals[i]);
					}
					halfpct[i] = Eval.cl(0.5,i,minvals[i],maxvals[i]);
					if (show_uncertainties_as_percentiles) {
						double lowerr = pct_scaling*(halfpct[i] - lowcl[i]);
						double hierr = pct_scaling*(hicl[i] - halfpct[i]);
						double lowpct = halfpct[i] - lowerr;
						double hipct = halfpct[i] + hierr;
						//cout << param_names[i] << ": " << halfpct[i] << " " << lowcl[i] << " " << hicl[i] << endl;
						cout << param_names[i] << ": " << halfpct[i] << " " << lowpct << " " << hipct << endl;
					} else {
						if (!latex_table_format) {
							cout << param_names[i] << ": " << halfpct[i] << " -" << (halfpct[i]-lowcl[i]) << " / +" << (hicl[i] - halfpct[i]) << endl;
						} else {
							bool show_as_powers = false;
							bool increase_precision = false; // do this if a number is less than 0.1
							double half, hierr, lowerr;
							half = halfpct[i];
							hierr = hicl[i] - halfpct[i];
							lowerr = halfpct[i] - lowcl[i];
							if (half > 1e4) {
								show_as_powers = true;
								powers_of_ten = 0;
								do {
									half /= 10;
									hierr /= 10;
									lowerr /= 10;
									powers_of_ten++;
								} while (half > 10);
							}
							else if (abs(half) < 0.1) {
								increase_precision = true;
								powers_of_ten = -1;
								double halfdup = abs(half);
								do {
									halfdup *= 10;
									powers_of_ten--;
								} while (halfdup < 0.1);
								cout << setprecision(precision-powers_of_ten-1);
								cout << fixed;
							}
							if (!suppress_latex_names) {
								cout << "$" << latex_param_names[i];
								if (show_as_powers) cout << "(10^" << powers_of_ten << ")";
								cout << "$ & ";
								if (show_prior_ranges) {
									cout << defaultfloat;
									if ((prior_minvals[i] > -1e30) and (prior_maxvals[i] < 1e30)) {
										cout << "$(" << prior_minvals[i] << "," << prior_maxvals[i] << ")$ & ";
									} else {
										cout << "... & ";
									}
									cout << fixed;
								}
							}
							cout << "$" << half << "_{-" << lowerr << "}^{+" << hierr << "}$ & " << endl;
							if (increase_precision) {
								cout << setprecision(precision);
								cout << fixed;
							}
						}
					}
				}
				if (add_dummy_params) {
					for (int i=0; i < ndummy; i++) {
						if (!suppress_latex_names) {
							cout << "dummy" << i << " & ";
							if (show_prior_ranges) cout << "... & ";
						}
						cout << "... & " << endl;
					}
				}
				if (latex_table_format) {
					if (logev != 1e30) {
						if (!suppress_latex_names) {
							cout << "$\\ln\\mathcal{E}$ & "; 
							if (show_prior_ranges) cout << "... & ";
						}
						cout << logev << " & " << endl;
					}
				}
				cout << endl;
				delete[] halfpct;
				delete[] lowcl;
				delete[] hicl;
			}
			if (output_percentile) {
				Eval.FindRanges(minvals,maxvals,nbins,threshold);
				// The following gives the 
				cout << percentile << " percentile:\n\n";
				double val;
				for (i=0; i < nparams_eff; i++) {
					val = Eval.cl(percentile,i,minvals[i],maxvals[i]);
					cout << param_names[i] << " = " << val << endl;
				}
				cout << endl;
			}
			if (print_marker_values) {
				if (show_markers) {
					cout << "True parameter values (and bestfit values):\n";
					for (i=0; i < n_markers; i++) {
						// NOTE: The following errors are from standard deviation, not from CL's 
						cout << param_names[i] << ": " << markers[i] << " (" << Eval.output_min_chisq_value(i) << ")" << endl;
					}
					cout << endl;
				}
			}
		}

		delete[] minvals;
		delete[] maxvals;
	}

#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	if (mpi_id==0) {
		int system_returnval;
		if (make_1d_posts)
		{
			string pyname = output_file_label + ".py";
			ofstream pyscript(pyname.c_str());
			pyscript << "import GetDistPlots, os" << endl;
			pyscript << "g=GetDistPlots.GetDistPlotter('" << output_dir << "/')" << endl;
			pyscript << "g.settings.setSubplotSize(3.0000,width_scale=1.0)  # width_scale scales the width of all lines in the plot" << endl;
			pyscript << "outdir=''" << endl;
			pyscript << "roots=['" << file_label << "']" << endl;
			if (show_markers) {
				pyscript << "marker_list=[";
				for (i=0; i < n_markers; i++) {
					pyscript << markers[i];
					if (i < nparams_eff-1) pyscript << ",";
				}
				pyscript << "]" << endl;
			} else {
				pyscript << "marker_list=[]   # put parameter values in this list if you want to mark the 'true' or best-fit values on posteriors" << endl;
			}
			pyscript << "g.plots_1d(roots,markers=marker_list,marker_color='orange')" << endl;
			//if (add_title) pyscript << "g.add_title(r'" << title << "')" << endl; // 1d title doesn't look good
			pyscript << "g.export(os.path.join(outdir,'" << output_file_label << ".pdf'))" << endl;
			pyscript.close();
			if (run_python_script) {
				string pycommand = "python " + pyname;
				if (system(pycommand.c_str()) == 0) {
					cout << "Plot for 1D posteriors saved to '" << output_file_label << ".pdf'\n";
					//string rmcommand = "rm " + pyname;
					//system_returnval = system(rmcommand.c_str());
				}
				else cout << "Error: Could not generate PDF file for 1D posteriors\n";
			} else {
				cout << "Plotting script for 1D posteriors saved to '" << pyname << "'\n";
			}
		}

		if (make_2d_posts)
		{
			string pyname = output_file_label + "_2D.py";
			ofstream pyscript2d(pyname.c_str());
			pyscript2d << "import GetDistPlots, os" << endl;
			pyscript2d << "g=GetDistPlots.GetDistPlotter('" << output_dir << "/')" << endl;
			pyscript2d << "g.settings.setSubplotSize(3.0000,width_scale=1.0)  # width_scale scales the width of all lines in the plot" << endl;
			pyscript2d << "outdir=''" << endl;
			pyscript2d << "roots=['" << file_label << "']" << endl;
			pyscript2d << "pairs=[]" << endl;
			for (i=0; i < nparams_eff_2d; i++) {
				for (j=i+1; j < nparams_eff_2d; j++) {
					if ((hist2d_active_params[i]) and (hist2d_active_params[j])) {
						pyscript2d << "pairs.append(['" << param_names[i] << "','" << param_names[j] << "'])\n";
					}
				}
			}
			pyscript2d << "g.plots_2d(roots,param_pairs=pairs,";
			if (include_shading) pyscript2d << "shaded=True";
			else pyscript2d << "shaded=False";
			pyscript2d << ")" << endl;
			if (add_title) pyscript2d << "g.add_title(r'" << title << "')" << endl;
			pyscript2d << "g.export(os.path.join(outdir,'" << output_file_label << "_2D.pdf'))" << endl;
			/*
			if (run_python_script) {
				string pycommand = "python " + pyname;
				if (system(pycommand.c_str()) == 0) {
					cout << "Plot for 2D posteriors saved to '" << file_label << "_2D.pdf'\n";
					//string rmcommand = "rm " + pyname;
					//system_returnval = system(rmcommand.c_str());
				}
				else cout << "Error: Could not generate PDF file for 2D posteriors\n";
			} else {
				cout << "Plotting script for 2D posteriors saved to '" << pyname << "'\n";
			}
			*/


			if (make_1d_posts) {
				// make script for triangle plot
				int n_triplots = 1;
				if (make_subplot) n_triplots++;
				for (int k=0; k < n_triplots; k++) {
					if (k==0) pyname = output_file_label + "_tri.py";
					else pyname = output_file_label + "_subtri.py";
					ofstream pyscript(pyname.c_str());
					pyscript << "import GetDistPlots, os" << endl;
					pyscript << "g=GetDistPlots.GetDistPlotter('" << output_dir << "/')" << endl;
					pyscript << "g.settings.setSubplotSize(3.0000,width_scale=1.0)  # width_scale scales the width of all lines in the plot" << endl;
					pyscript << "outdir=''" << endl;
					pyscript << "roots=['" << file_label << "']" << endl;
					if (show_markers) {
						pyscript << "marker_list=[";
						for (i=0; i < n_markers; i++) {
							if ((hist2d_active_params[i]) and ((k==0) or (subplot_active_params[i]))) {
								pyscript << markers[i];
								if ((k==0) and (i != n_markers-1)) pyscript << ",";
								else if (k==1) {
									bool last_param = true;
									for (int ii=i+1; ii < n_markers; ii++) {
										if (subplot_active_params[ii]==true) last_param = false;
									}
									if (!last_param) pyscript << ",";
								}
							}
						}
						pyscript << "]" << endl;
					} else {
						pyscript << "marker_list=[]   # put parameter values in this list if you want to mark the 'true' or best-fit values on posteriors" << endl;
					}
					pyscript << "g.triangle_plot(roots, [";
					for (i=0; i < nparams_eff_2d; i++) {
						if ((hist2d_active_params[i]) and ((k==0) or (subplot_active_params[i]))) {
							pyscript << "'" << param_names[i] << "'";
							if ((k==0) and (i != nparams_eff_2d-1)) pyscript << ",";
							else if (k==1) {
								bool last_param = true;
								for (int ii=i+1; ii < nparams_eff_2d; ii++) {
									if (subplot_active_params[ii]==true) last_param = false;
								}
								if (!last_param) pyscript << ",";
							}
						}
					}
					pyscript << "],markers=marker_list,marker_color='orange',show_marker_2d=";
					if (show_markers) pyscript << "True";
					else pyscript << "False";
					pyscript << ",marker_2d='x',";
					if (include_shading) pyscript << "shaded=True";
					else pyscript << "shaded=False";
					pyscript << ")" << endl;
					if (add_title) pyscript << "g.add_title(r'" << title << "')" << endl;
					pyscript << "g.export(os.path.join(outdir,'" << output_file_label;
					if (k==0) pyscript << "_tri.pdf'))" << endl;
					else pyscript << "_subtri.pdf'))" << endl;
					if (run_python_script) {
						string pycommand = "python " + pyname;
						if (system(pycommand.c_str()) == 0) {
							if (k==0) cout << "Triangle plot (1D+2D posteriors) saved to '" << output_file_label << "_tri.pdf'\n";
							else cout << "Triangle subplot saved to '" << output_file_label << "_subtri.pdf'\n";
							//string rmcommand = "rm " + pyname;
							//system_returnval = system(rmcommand.c_str());
						}
						else cout << "Error: Could not generate PDF file for triangle plot (1d + 2d posteriors)\n";
					} else {
						cout << "Plotting script for triangle plot saved to '" << pyname << "'\n";
					}
				}
			}
		}
	}

	delete[] prior_minvals;
	delete[] prior_maxvals;
	delete[] param_names;
	delete[] latex_param_names;
	delete[] markers;
	delete[] subplot_active_params;
	delete[] hist2d_active_params;

#ifdef USE_MPI
	MPI_Finalize();
#endif

	return 0;
}

void adjust_ranges_to_include_markers(double *minvals, double *maxvals, double *markers, const int n_markers)
{
	const double extra_length_frac = 0.05;
	for (int i=0; i < n_markers; i++) {
		if (minvals[i] > markers[i]) {
			if ((maxvals[i]-markers[i]) > 4*(maxvals[i]-minvals[i])) warn("marker %i is WAY out of range of parameter chain; will not show marker",i);
			else {
				minvals[i] = markers[i];
				minvals[i] -= extra_length_frac*(maxvals[i]-minvals[i]);
			}
		}
		else if (maxvals[i] < markers[i]) {
			if ((markers[i]-minvals[i]) > 4*(maxvals[i]-minvals[i])) warn("marker %i is WAY out of range of parameter chain; will not show marker",i);
			else {
				maxvals[i] = markers[i];
				maxvals[i] += extra_length_frac*(maxvals[i]-minvals[i]);
			}
		}
	}
}

bool file_exists(const string &filename)
{
	ifstream infile(filename.c_str());
	bool exists = infile.good();
	infile.close();
	return exists;
}

char *advance(char *p)
{
	// This advances to the next flag (if there is one; 'e' is ignored because it might be part of a number in scientific notation)
	while ((*++p) and ((!isalpha(*p)) or (*p=='e'))) ;
	return --p;
}

void usage_error(const int mpi_id)
{
	if (mpi_id==0) {
		cerr << "Usage:\n\n";
		cerr << "mkdist <file_root> ...\n\n";
		cerr << "Argument options (after the first required argument):\n"
				"  -d:<dir> set input/output directory to <dir> (default: 'chains_<file_root>/')\n"
				"  -n#       make 1d histograms with # bins\n"
				"  -N#       make 2d histograms with # by # bins\n"
				"  -x        exclude the derived parameters when plotting histograms\n"
				"  -C#       truncate number of params, i.e. include only the first # parameters when plotting histograms\n"
				"  -s        make triangle subplot using file '<file_root>.subplot_params' containing flags (0 or 1) for each parameter\n"
				"  -h        output header from chain file (gives evidence, sampler, livepts, etc.)\n"
				"  -t        add a title for the triangle plots (contained in file <file_root>.plot_title)\n"
				"  -P        execute Python scripts generated by mkdist to output posteriors as PDF files\n"
				"  -S        plot smoothed histograms\n"
				"  -F        do not include shading in 2D histograms\n"
				"  -r        skip generating 2D histogram files, but write the plotting scripts\n"
				"  -b        output minimum chi-square point\n"
				"  -e        output mean parameters with standard errors in each parameter\n"
				"  -E        output best-fit parameters with errors given by 15.8\% and 84.1\% probability\n"
				"  -E2        output best-fit parameters with errors given by 2.5\% and 97.5\% probability\n"
				"  -u        show uncertainties as percentiels, rather than error bars (+/-)\n"
				"  -p#       output the #'th percentile for each parameter (where # must be between 0 and 1)\n"
				"  -c        number of initial points to cut from each MCMC chain (if no cut is specified,\n"
				"                the first 10% of points are cut by default)\n"
				"  -B#       input minimum probability threshold used for defining parameter ranges\n"
				"               for plotting (default = 3e-3; higher threshold --> smaller ranges)\n"
				"  -f        use Fisher matrix to generate 1d,2d posteriors (MCMC data not required)\n"
				"  -T:<file> transform parameters using an input script. For usage info, enter 'T' with\n"
				"               no argument.\n"
				"  -I:<file> define parameter priors for importance sampling using an input script.\n"
				"  -a:<suffix> Add <suffix> onto the filenames of output python scripts and PDF files.\n"
				"  -q        quiet mode (non-verbose)\n" << endl;
	}
#ifdef USE_MPI
	MPI_Finalize();
#endif
	exit(1);
}

void show_transform_usage()
{
	cout << "Usage: mkdist <file_root> -T:<file>\n\n"
			"Use this option to transform parameters and obtain posteriors in the transformed parameters.\n"
			"In the script <file>, the format is as follows:\n\n"
			"<paramnum> <transform_type> <params> [name=...] [latex_name=...]\n\n"
			"where the number of parameters <params> depends on the transformation type, and the 'name=...' and\n"
			"'latex_name=...' are optional and allow you to rename the parameter. The transformation types are as follows:\n\n"
			"log      -- transform to log(p) using the base 10 logarithm; no parameters to enter.\n"
			"exp      -- transform to 10^p; no parameters to enter.\n"
			"linear   -- transform to L{p} = A*p + b. The two parameter arguments are <A> and <b>, so e.g. 'fit transform\n"
			"              linear 2 5' will transform p --> 2*p + 5.\n"
			"gaussian -- transformation whose Jacobian is Gaussian. There are two arguments, <mean> and dispersion <sig>\n"
			"              (e.g., 'fit transform # gaussian 0.2 0.5' will be Gaussian with mean 0.2 and dispersion 0.5).\n"
			"inverse_gaussian -- inverse of the Gaussian transformation, with the same parameters <mean> and <sig>\n\n"
			"For example, to apply a linear transformation p --> 2*p - 1 to parameter 2, enter:\n"
			"2 linear 2 -1 name=newparam\n\n"
			"Again, the renaming is optional; for the log transformation, the parameter name is automatically changed from\n"
			"name --> log(name) by default. In the script file, you can transform as many parameters as you like by\n"
			"entering transform commands on separate lines.\n\n";
	exit(1);
}

