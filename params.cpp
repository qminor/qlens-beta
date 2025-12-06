#include "params.h"
#include <string>
#include <iostream>
using namespace std;

void ParamList::insert_params(const int pi, const int pf, string* names_in, string* latex_names_in, double* untransformed_values_in, double* stepsizes_in)
{
	int i, j, np = pf-pi;
	int new_nparams = nparams + np;
	int* new_paramnums = new int[nparams]; // this stores the update parameter numbers for parameters that already existed
	string* new_untransformed_param_names = new string[new_nparams];
	string* new_param_names = new string[new_nparams];
	string* new_untransformed_latex_names = new string[new_nparams];
	string* new_latex_names = new string[new_nparams];
	string* new_override_names = new string[new_nparams];
	ParamPrior** newpriors = new ParamPrior*[new_nparams];
	ParamTransform** newtransforms = new ParamTransform*[new_nparams];
	double* new_untransformed_values = new double[new_nparams];
	double* new_values = new double[new_nparams];
	double* new_stepsizes = new double[new_nparams];
	bool* new_auto_stepsize = new bool[new_nparams];
	bool* new_subplot_param = new bool[new_nparams];
	bool* new_hist2d_param = new bool[new_nparams];
	double* new_prior_norms = new double[new_nparams];
	double* new_untransformed_prior_limits_lo = new double[new_nparams];
	double* new_untransformed_prior_limits_hi = new double[new_nparams];
	bool* new_defined_prior_limits = new bool[new_nparams];
	double* new_prior_limits_lo = new double[new_nparams];
	double* new_prior_limits_hi = new double[new_nparams];
	//bool* new_override_prior_limits = new bool[new_nparams];

	for (i=0; i < pi; i++) {
		new_paramnums[i] = i;
		new_untransformed_param_names[i] = names_in[i]; // note, we update the names of all parameters in case there were any labeling changes due to multiples
		new_param_names[i] = param_names[i];
		new_untransformed_latex_names[i] = latex_names_in[i]; // note, we update the names of all parameters in case there were any labeling changes due to multiples
		new_latex_names[i] = latex_names[i];
		new_override_names[i] = override_names[i];
		newpriors[i] = new ParamPrior(priors[i]);
		newtransforms[i] = new ParamTransform(transforms[i]);
		new_untransformed_values[i] = untransformed_values[i]; 
		new_values[i] = values[i];
		new_auto_stepsize[i] = auto_stepsize[i];
		new_hist2d_param[i] = hist2d_param[i];
		new_subplot_param[i] = subplot_param[i];
		new_stepsizes[i] = stepsizes[i];
		new_prior_norms[i] = prior_norms[i];
		new_untransformed_prior_limits_lo[i] = untransformed_prior_limits_lo[i];
		new_untransformed_prior_limits_hi[i] = untransformed_prior_limits_hi[i];
		new_defined_prior_limits[i] = defined_prior_limits[i];
		new_prior_limits_lo[i] = prior_limits_lo[i];
		new_prior_limits_hi[i] = prior_limits_hi[i];
		//new_override_prior_limits[i] = override_prior_limits[i];
	}
	for (j=0,i=pi; i < pf; i++, j++) {
		new_untransformed_param_names[i] = names_in[i];
		new_param_names[i] = "";
		new_untransformed_latex_names[i] = latex_names_in[i];
		new_latex_names[i] = "";
		new_override_names[i] = "";
		newpriors[i] = new ParamPrior();
		newtransforms[i] = new ParamTransform();
		new_untransformed_values[i] = untransformed_values_in[j]; 
		new_values[i] = -VERY_LARGE; // hasn't been set yet
		new_stepsizes[i] = stepsizes_in[j];
		new_auto_stepsize[i] = true; // stepsizes for newly added parameters are set to 'auto'
		new_hist2d_param[i] = true; // stepsizes for newly added parameters are set to 'auto'
		new_subplot_param[i] = false; // stepsizes for newly added parameters are set to 'auto'
		new_prior_norms[i] = 1.0;
		new_untransformed_prior_limits_lo[i] = -VERY_LARGE;
		new_untransformed_prior_limits_hi[i] = VERY_LARGE;
		new_defined_prior_limits[i] = false;
		new_prior_limits_lo[i] = -VERY_LARGE;
		new_prior_limits_hi[i] = VERY_LARGE;
		//new_override_prior_limits[i] = false;
	}
	for (j=pf,i=pi; i < nparams; i++, j++) {
		new_paramnums[i] = j;
		new_untransformed_param_names[j] = names_in[j]; // note, we update the names of all parameters in case there were any labeling changes due to multiples
		new_param_names[j] = param_names[i];
		new_untransformed_latex_names[j] = latex_names_in[j]; // note, we update the names of all parameters in case there were any labeling changes due to multiples
		new_latex_names[j] = latex_names[i];
		new_override_names[j] = override_names[i];
		newpriors[j] = new ParamPrior(priors[i]);
		newtransforms[j] = new ParamTransform(transforms[i]);
		new_untransformed_values[j] = untransformed_values[i];
		new_values[j] = values[i];
		new_auto_stepsize[j] = auto_stepsize[i];
		new_hist2d_param[j] = hist2d_param[i];
		new_subplot_param[j] = subplot_param[i];
		new_stepsizes[j] = stepsizes[i];
		new_prior_norms[j] = prior_norms[i];
		new_untransformed_prior_limits_lo[j] = untransformed_prior_limits_lo[i];
		new_untransformed_prior_limits_hi[j] = untransformed_prior_limits_hi[i];
		new_defined_prior_limits[j] = defined_prior_limits[i];
		new_prior_limits_lo[j] = prior_limits_lo[i];
		new_prior_limits_hi[j] = prior_limits_hi[i];
		//new_override_prior_limits[j] = override_prior_limits[i];
	}
	if (nparams > 0) delete_param_ptrs();
	untransformed_param_names = new_untransformed_param_names;
	param_names = new_param_names;
	untransformed_latex_names = new_untransformed_latex_names;
	latex_names = new_latex_names;
	override_names = new_override_names;
	priors = newpriors;
	transforms = newtransforms;
	untransformed_values = new_untransformed_values;
	values = new_values;
	stepsizes = new_stepsizes;
	auto_stepsize = new_auto_stepsize;
	subplot_param = new_subplot_param;
	hist2d_param = new_hist2d_param;
	prior_norms = new_prior_norms;
	untransformed_prior_limits_lo = new_untransformed_prior_limits_lo;
	untransformed_prior_limits_hi = new_untransformed_prior_limits_hi;
	defined_prior_limits = new_defined_prior_limits;
	prior_limits_lo = new_prior_limits_lo;
	prior_limits_hi = new_prior_limits_hi;
	//override_prior_limits = new_override_prior_limits;
	nparams = new_nparams;
	update_reference_paramnums(new_paramnums);
	transform_parameters(pi,pf);
	transform_parameter_names();
	//cout << "NEW NPARAMS: " << nparams << endl;
	//for (int i=0; i < nparams; i++) {
		//cout << untransformed_param_names[i] << endl;
	//}
	delete[] new_paramnums;
}

bool ParamList::remove_params(const int pi, const int pf)
{
	if ((nparams==0) or (pf > nparams)) return false;
	int i, j, np = pf-pi;
	if (np==nparams) {
		clear_params();
		return true;
	}
	int new_nparams = nparams - np;
	int* new_paramnums = new int[nparams]; // this stores the update parameter numbers for parameters that already existed
	for (i=0; i < nparams; i++) new_paramnums[i] = -1;
	string* new_untransformed_param_names = new string[new_nparams];
	string* new_param_names = new string[new_nparams];
	string* new_untransformed_latex_names = new string[new_nparams];
	string* new_latex_names = new string[new_nparams];
	string* new_override_names = new string[new_nparams];
	ParamPrior** newpriors = new ParamPrior*[new_nparams];
	ParamTransform** newtransforms = new ParamTransform*[new_nparams];
	double* new_untransformed_values = new double[new_nparams];
	double* new_values = new double[new_nparams];
	double* new_stepsizes = new double[new_nparams];
	bool* new_auto_stepsize = new bool[new_nparams];
	bool* new_subplot_param = new bool[new_nparams];
	bool* new_hist2d_param = new bool[new_nparams];
	double* new_prior_norms = new double[new_nparams];
	double* new_untransformed_prior_limits_lo = new double[new_nparams];
	double* new_untransformed_prior_limits_hi = new double[new_nparams];
	bool* new_defined_prior_limits = new bool[new_nparams];
	double* new_prior_limits_lo = new double[new_nparams];
	double* new_prior_limits_hi = new double[new_nparams];
	//bool* new_override_prior_limits = new bool[new_nparams];
	for (i=0; i < pi; i++) {
		new_paramnums[i] = i;
		new_untransformed_param_names[i] = untransformed_param_names[i];
		new_param_names[i] = param_names[i];
		new_untransformed_latex_names[i] = untransformed_latex_names[i];
		new_latex_names[i] = latex_names[i];
		new_override_names[i] = override_names[i];
		newpriors[i] = new ParamPrior(priors[i]);
		newtransforms[i] = new ParamTransform(transforms[i]);
		new_untransformed_values[i] = untransformed_values[i];
		new_values[i] = values[i];
		new_stepsizes[i] = stepsizes[i];
		new_auto_stepsize[i] = auto_stepsize[i];
		new_subplot_param[i] = subplot_param[i];
		new_hist2d_param[i] = hist2d_param[i];
		new_prior_norms[i] = prior_norms[i];
		new_untransformed_prior_limits_lo[i] = untransformed_prior_limits_lo[i];
		new_untransformed_prior_limits_hi[i] = untransformed_prior_limits_hi[i];
		new_defined_prior_limits[i] = defined_prior_limits[i];
		new_prior_limits_lo[i] = prior_limits_lo[i];
		new_prior_limits_hi[i] = prior_limits_hi[i];
		//new_override_prior_limits[i] = override_prior_limits[i];
	}
	for (i=pf,j=pi; i < nparams; i++, j++) {
		new_paramnums[i] = j;
		new_untransformed_param_names[j] = untransformed_param_names[i];
		new_param_names[j] = param_names[i];
		new_untransformed_latex_names[j] = untransformed_latex_names[i];
		new_latex_names[j] = latex_names[i];
		new_override_names[j] = override_names[i];
		newpriors[j] = new ParamPrior(priors[i]);
		newtransforms[j] = new ParamTransform(transforms[i]);
		new_untransformed_values[j] = untransformed_values[i];
		new_values[j] = values[i];
		new_stepsizes[j] = stepsizes[i];
		new_auto_stepsize[j] = auto_stepsize[i];
		new_subplot_param[j] = subplot_param[i];
		new_hist2d_param[j] = hist2d_param[i];
		new_prior_norms[j] = prior_norms[i];
		new_untransformed_prior_limits_lo[j] = untransformed_prior_limits_lo[i];
		new_untransformed_prior_limits_hi[j] = untransformed_prior_limits_hi[i];
		new_defined_prior_limits[j] = defined_prior_limits[i];
		new_prior_limits_lo[j] = prior_limits_lo[i];
		new_prior_limits_hi[j] = prior_limits_hi[i];
		//new_override_prior_limits[j] = override_prior_limits[i];
	}
	delete_param_ptrs();
	untransformed_param_names = new_untransformed_param_names;
	param_names = new_param_names;
	untransformed_latex_names = new_untransformed_latex_names;
	latex_names = new_latex_names;
	override_names = new_override_names;
	priors = newpriors;
	transforms = newtransforms;
	untransformed_values = new_untransformed_values;
	values = new_values;
	stepsizes = new_stepsizes;
	auto_stepsize = new_auto_stepsize;
	subplot_param = new_subplot_param;
	hist2d_param = new_hist2d_param;
	prior_norms = new_prior_norms;
	untransformed_prior_limits_lo = new_untransformed_prior_limits_lo;
	untransformed_prior_limits_hi = new_untransformed_prior_limits_hi;
	defined_prior_limits = new_defined_prior_limits;
	prior_limits_lo = new_prior_limits_lo;
	prior_limits_hi = new_prior_limits_hi;
	//override_prior_limits = new_override_prior_limits;
	nparams = new_nparams;
	update_reference_paramnums(new_paramnums);
	delete[] new_paramnums;
	//cout << "NEW NPARAMS=" << nparams << endl;
	return true;
}

void ParamList::update_param_list(string* param_names_in, string* latex_names_in, double* stepsizes_in, const bool check_current_params)
{
	for (int i=0; i < nparams; i++) {
		untransformed_param_names[i] = param_names_in[i];
		untransformed_latex_names[i] = latex_names_in[i];
		if (auto_stepsize[i]) stepsizes[i] = stepsizes_in[i];
		transform_parameter_names();
		transform_stepsizes();
	}
}

void ParamList::add_dparam(DerivedParamType type_in, double param, int lensnum, double param2, bool use_kpc)
{
	DerivedParam** newlist = new DerivedParam*[n_dparams+1];
	string* new_dparam_names = new string[n_dparams+1];
	bool* new_subplot_dparam = new bool[n_dparams+1];
	bool* new_hist2d_dparam = new bool[n_dparams+1];
	if (n_dparams > 0) {
		for (int i=0; i < n_dparams; i++) {
			newlist[i] = dparams[i];
			new_dparam_names[i] = dparam_names[i];
			new_subplot_dparam[i] = subplot_dparam[i];
			new_hist2d_dparam[i] = hist2d_dparam[i];
		}
		delete_dparam_ptrs(false);
	}
	if (param2 == -1e30) newlist[n_dparams] = new DerivedParam(type_in,param,lensnum,-1,use_kpc);
	else newlist[n_dparams] = new DerivedParam(type_in,param,lensnum,param2,use_kpc);

	dparams = newlist;
	new_dparam_names[n_dparams] = dparams[n_dparams]->name;
	new_subplot_dparam[n_dparams] = false;
	new_hist2d_dparam[n_dparams] = true;

	dparam_names = new_dparam_names;
	subplot_dparam = new_subplot_dparam;
	hist2d_dparam = new_hist2d_dparam;
	n_dparams++;
}

bool ParamList::remove_dparam(int dparam_number)
{
	if ((dparam_number >= n_dparams) or (n_dparams == 0)) return false;

	DerivedParam** newlist;
	string* new_dparam_names;
	bool* new_subplot_dparam;
	bool* new_hist2d_dparam;
	if (n_dparams > 1) {
		newlist = new DerivedParam*[n_dparams-1];
		new_dparam_names = new string[n_dparams-1];
		new_subplot_dparam = new bool[n_dparams-1];
		new_hist2d_dparam = new bool[n_dparams-1];
		int i,j;
		for (i=0, j=0; i < n_dparams; i++) {
			if (i != dparam_number) {
				newlist[j] = dparams[i];
				new_dparam_names[j] = dparam_names[i];
				new_subplot_dparam[j] = subplot_dparam[i];
				new_hist2d_dparam[j] = hist2d_dparam[i];
				j++;
			} else delete dparams[i];
		}
	}
	delete_dparam_ptrs(false);
	n_dparams--;
	if (n_dparams > 0) {
		dparams = newlist;
		dparam_names = new_dparam_names;
		subplot_dparam = new_subplot_dparam;
		hist2d_dparam = new_hist2d_dparam;
	} else {
		dparams = NULL;
		dparam_names = NULL;
		subplot_dparam = NULL;
		hist2d_dparam = NULL;
	}
	return true;
}

void ParamList::print_priors_and_transforms()
{
	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter settings:\n";
	int max_length=0;
	for (int i=0; i < nparams; i++) {
		if (untransformed_param_names[i].length() > max_length) max_length = untransformed_param_names[i].length();
	}
	int extra_length;
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << untransformed_param_names[i] << ": ";
		extra_length = max_length - untransformed_param_names[i].length();
		for (int j=0; j < extra_length; j++) cout << " ";
		if ((nparams > 10) and (i < 10)) cout << " ";
		if (!output_prior(i)) die("Prior type unknown");
		if (transforms[i]->transform==NONE) ;
		else if (transforms[i]->transform==LOG_TRANSFORM) cout << ", log transformation";
		else if (transforms[i]->transform==GAUSS_TRANSFORM) cout << ", gaussian transformation (mean=" << transforms[i]->gaussian_pos << ", sigma=" << transforms[i]->gaussian_sig << ")";
		else if (transforms[i]->transform==LINEAR_TRANSFORM) cout << ", linear transformation A*" << untransformed_param_names[i] << " + b (A=" << transforms[i]->a << ", b=" << transforms[i]->b << ")";
		else if (transforms[i]->transform==RATIO) cout << ", ratio transformation " << untransformed_param_names[i] << "/" << untransformed_param_names[transforms[i]->ratio_paramnum];
		if (transforms[i]->include_jacobian==true) cout << " (include Jacobian in likelihood)";
		cout << endl;
	}
}

bool ParamList::output_prior(const int i)
{
	if (priors[i]->prior==UNIFORM_PRIOR) {
		cout << "uniform prior";
		if ((transforms[i]->transform != NONE) and (transforms[i]->include_jacobian)) {
			cout << " in " << untransformed_param_names[i];
		}
	}
	else if (priors[i]->prior==LOG_PRIOR) cout << "log prior";
	else if (priors[i]->prior==GAUSS_PRIOR) {
		cout << "gaussian prior (mean=" << priors[i]->gaussian_pos << ",sigma=" << priors[i]->gaussian_sig << ")";
	}
	else if (priors[i]->prior==GAUSS2_PRIOR) {
		cout << "multivariate gaussian prior, params=(" << i << "," << priors[i]->gauss_paramnums[1] << "), mean=(" << priors[i]->gauss_meanvals[0] << "," << priors[i]->gauss_meanvals[1] << "), sigs=(" << sqrt(priors[i]->covariance_matrix[0][0]) << "," << (sqrt(priors[i]->covariance_matrix[1][1])) << "), sqrt(sig12) = " << (sqrt(priors[i]->covariance_matrix[0][1]));
	} else if (priors[i]->prior==GAUSS2_PRIOR_SECONDARY) {
		cout << "multivariate gaussian prior, params=(" << priors[i]->gauss_paramnums[0] << "," << priors[i]->gauss_paramnums[1] << ")";
	} else return false;
	if (transforms[i]->include_jacobian==true) cout << " (including Jacobian in likelihood)";
	return true;
}

bool ParamList::print_priors_and_limits()
{
	int max_length=0;
	for (int i=0; i < nparams; i++) {
		if (param_names[i].length() > max_length) max_length = param_names[i].length();
	}
	int extra_length;
 
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << param_names[i] << ": " << flush;
		extra_length = max_length - param_names[i].length();
		for (int j=0; j < extra_length; j++) cout << " ";
		//if ((nparams > 10) and (i < 10)) cout << " ";
		cout << flush;
		if (!output_prior(i)) return false;
		cout << ", [";
		if (prior_limits_lo[i]==-VERY_LARGE) cout << "-inf";
		else cout << prior_limits_lo[i];
		cout << ",";
		if (prior_limits_hi[i]==VERY_LARGE) cout << "inf";
		else cout << prior_limits_hi[i];
		cout << "]";
		if ((values[i] < prior_limits_lo[i]) or (values[i] > prior_limits_hi[i])) cout << " *NOTE*: current value (" << values[i] << ") is outside prior range";
		cout << endl;
	}
	cout << endl;
	return true;
}

void ParamList::print_stepsizes()
{
	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter initial stepsizes:\n";
	//string *param_names = new string[nparams];
	//transform_parameter_names(untransformed_param_names,param_names,NULL,NULL);
	int max_length=0;
	for (int i=0; i < nparams; i++) {
		if (param_names[i].length() > max_length) max_length = param_names[i].length();
	}
	int extra_length;
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << param_names[i] << ": ";
		extra_length = max_length - param_names[i].length();
		for (int j=0; j < extra_length; j++) cout << " ";
		if ((nparams > 10) and (i < 10)) cout << " ";
		cout << stepsizes[i];
		if (auto_stepsize[i]) cout << " (auto)";
		cout << endl;
	}
	//delete[] param_names;
}

void ParamList::print_untransformed_prior_limits()
{
	//for (int i=0; i < nparams; i++) {
		//if (defined_prior_limits[i]==true) {
			//cout << "USE_LIMITS " << i << endl;
		//}
	//}

	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter limits imposed on chi-square:\n";
	int max_length=0;
	for (int i=0; i < nparams; i++) {
		if (untransformed_param_names[i].length() > max_length) max_length = untransformed_param_names[i].length();
	}
	int extra_length;
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << untransformed_param_names[i] << ": ";
		extra_length = max_length - untransformed_param_names[i].length();
		for (int j=0; j < extra_length; j++) cout << " ";
		if ((nparams > 10) and (i < 10)) cout << " ";
		if (defined_prior_limits[i]==false) cout << "none" << endl;
		else {
			cout << "[";
			if (untransformed_prior_limits_lo[i]==-VERY_LARGE) cout << "-inf";
			else cout << untransformed_prior_limits_lo[i];
			cout << ":";
			if (untransformed_prior_limits_hi[i]==VERY_LARGE) cout << "inf";
			else cout << untransformed_prior_limits_hi[i];
			cout << "]" << endl;
		}
	}
}

bool ParamList::print_parameter_values()
{
	if (nparams==0) return false;
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << param_names[i] << ": " << values[i] << endl;
	}
	cout << endl;
	return true;
}

string ParamList::mkstring_doub(const double db)
{
	stringstream dstr;
	string dstring;
	dstr << db;
	dstr >> dstring;
	return dstring;
}

string ParamList::mkstring_int(const int i)
{
	stringstream istr;
	string istring;
	istr << i;
	istr >> istring;
	return istring;
}

string ParamList::get_param_values_string()
{
	string paramstring = "";
	for (int i=0; i < nparams; i++) {
		paramstring += mkstring_int(i) + ". " + param_names[i] + ": " + mkstring_doub(values[i]) + "\n";
	}
	return paramstring;
}

