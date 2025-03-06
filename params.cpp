#include "params.h"
using namespace std;

void ParamSettings::update_params(const int nparams_in, vector<string>& names, double* stepsizes_in)
{
	int i;
	if (nparams==nparams_in) {
		// update parameter names just in case
		for (i=0; i < nparams_in; i++) {
			param_names[i] = names[i];
		}
		return;
	}
	int newparams = nparams_in - nparams;
	int* new_paramnums = new int[nparams]; // this stores the update parameter numbers for parameters that already existed
	for (i=0; i < nparams; i++) new_paramnums[i] = -1;
	ParamPrior** newpriors = new ParamPrior*[nparams_in];
	ParamTransform** newtransforms = new ParamTransform*[nparams_in];
	double* new_stepsizes = new double[nparams_in];
	bool* new_auto_stepsize = new bool[nparams_in];
	bool* new_subplot_param = new bool[nparams_in];
	bool* new_hist2d_param = new bool[nparams_in];
	double* new_prior_norms = new double[nparams_in];
	double* new_penalty_limits_lo = new double[nparams_in];
	double* new_penalty_limits_hi = new double[nparams_in];
	bool* new_use_penalty_limits = new bool[nparams_in];
	double* new_override_limits_lo = new double[nparams_in];
	double* new_override_limits_hi = new double[nparams_in];
	bool* new_override_prior_limits = new bool[nparams_in];
	if (param_names != NULL) delete[] param_names;
	param_names = new string[nparams_in];
	string* new_override_names = new string[nparams_in];
	if (nparams_in > nparams) {
		for (i=0; i < nparams; i++) {
			new_paramnums[i] = i;
			newpriors[i] = new ParamPrior(priors[i]);
			newtransforms[i] = new ParamTransform(transforms[i]);
			new_stepsizes[i] = stepsizes[i];
			new_auto_stepsize[i] = auto_stepsize[i];
			new_subplot_param[i] = subplot_param[i];
			new_hist2d_param[i] = hist2d_param[i];
			new_prior_norms[i] = prior_norms[i];
			new_penalty_limits_lo[i] = penalty_limits_lo[i];
			new_penalty_limits_hi[i] = penalty_limits_hi[i];
			new_use_penalty_limits[i] = use_penalty_limits[i];
			new_override_limits_lo[i] = override_limits_lo[i];
			new_override_limits_hi[i] = override_limits_hi[i];
			new_override_prior_limits[i] = override_prior_limits[i];
			param_names[i] = names[i];
			new_override_names[i] = override_names[i];
		}
		for (i=nparams; i < nparams_in; i++) {
			//cout << "New parameter " << i << ", setting plimits to false" << endl;
			newpriors[i] = new ParamPrior();
			newtransforms[i] = new ParamTransform();
			param_names[i] = names[i];
			new_stepsizes[i] = stepsizes_in[i];
			new_auto_stepsize[i] = true; // stepsizes for newly added parameters are set to 'auto'
			new_subplot_param[i] = false; 
			new_hist2d_param[i] = true; 
			new_prior_norms[i] = 1.0;
			new_penalty_limits_lo[i] = -1e30;
			new_penalty_limits_hi[i] = 1e30;
			new_use_penalty_limits[i] = false;
			new_override_limits_lo[i] = -1e30;
			new_override_limits_hi[i] = 1e30;
			new_override_prior_limits[i] = false;
			new_override_names[i] = "";
		}
	} else {
		for (i=0; i < nparams_in; i++) {
			new_paramnums[i] = i;
			newpriors[i] = new ParamPrior(priors[i]);
			newtransforms[i] = new ParamTransform(transforms[i]);
			new_stepsizes[i] = stepsizes[i];
			new_auto_stepsize[i] = auto_stepsize[i];
			new_subplot_param[i] = subplot_param[i];
			new_hist2d_param[i] = hist2d_param[i];
			new_prior_norms[i] = prior_norms[i];
			new_penalty_limits_lo[i] = penalty_limits_lo[i];
			new_penalty_limits_hi[i] = penalty_limits_hi[i];
			new_use_penalty_limits[i] = use_penalty_limits[i];
			new_override_limits_lo[i] = override_limits_lo[i];
			new_override_limits_hi[i] = override_limits_hi[i];
			new_override_prior_limits[i] = override_prior_limits[i];
			param_names[i] = names[i];
			new_override_names[i] = override_names[i];
		}
	}
	if (nparams > 0) {
		for (i=0; i < nparams; i++) {
			delete priors[i];
			delete transforms[i];
		}
		delete[] priors;
		delete[] transforms;
		delete[] stepsizes;
		delete[] auto_stepsize;
		delete[] subplot_param;
		delete[] hist2d_param;
		delete[] prior_norms;
		delete[] penalty_limits_lo;
		delete[] penalty_limits_hi;
		delete[] use_penalty_limits;
		delete[] override_limits_lo;
		delete[] override_limits_hi;
		delete[] override_prior_limits;
		delete[] override_names;
	}
	priors = newpriors;
	transforms = newtransforms;
	stepsizes = new_stepsizes;
	auto_stepsize = new_auto_stepsize;
	subplot_param = new_subplot_param;
	hist2d_param = new_hist2d_param;
	prior_norms = new_prior_norms;
	penalty_limits_lo = new_penalty_limits_lo;
	penalty_limits_hi = new_penalty_limits_hi;
	use_penalty_limits = new_use_penalty_limits;
	override_limits_lo = new_override_limits_lo;
	override_limits_hi = new_override_limits_hi;
	override_prior_limits = new_override_prior_limits;
	override_names = new_override_names;
	nparams = nparams_in;
	update_reference_paramnums(new_paramnums);
	delete[] new_paramnums;
}
void ParamSettings::insert_params(const int pi, const int pf, vector<string>& names, double* stepsizes_in)
{
	int i, j, np = pf-pi;
	int new_nparams = nparams + np;
	int* new_paramnums = new int[nparams]; // this stores the update parameter numbers for parameters that already existed
	ParamPrior** newpriors = new ParamPrior*[new_nparams];
	ParamTransform** newtransforms = new ParamTransform*[new_nparams];
	double* new_stepsizes = new double[new_nparams];
	bool* new_auto_stepsize = new bool[new_nparams];
	bool* new_subplot_param = new bool[new_nparams];
	bool* new_hist2d_param = new bool[new_nparams];
	double* new_prior_norms = new double[new_nparams];
	double* new_penalty_limits_lo = new double[new_nparams];
	double* new_penalty_limits_hi = new double[new_nparams];
	bool* new_use_penalty_limits = new bool[new_nparams];
	double* new_override_limits_lo = new double[new_nparams];
	double* new_override_limits_hi = new double[new_nparams];
	bool* new_override_prior_limits = new bool[new_nparams];
	string* new_param_names = new string[new_nparams];
	string* new_override_names = new string[new_nparams];
	for (i=0; i < pi; i++) {
		new_paramnums[i] = i;
		newpriors[i] = new ParamPrior(priors[i]);
		newtransforms[i] = new ParamTransform(transforms[i]);
		new_auto_stepsize[i] = auto_stepsize[i];
		new_hist2d_param[i] = hist2d_param[i];
		new_subplot_param[i] = subplot_param[i];
		new_stepsizes[i] = (auto_stepsize[i]) ? stepsizes_in[i] : stepsizes[i]; // if stepsizes are set to 'auto', use new auto stepsizes since they might have changed
		new_prior_norms[i] = prior_norms[i];
		new_penalty_limits_lo[i] = penalty_limits_lo[i];
		new_penalty_limits_hi[i] = penalty_limits_hi[i];
		new_use_penalty_limits[i] = use_penalty_limits[i];
		new_override_limits_lo[i] = override_limits_lo[i];
		new_override_limits_hi[i] = override_limits_hi[i];
		new_override_prior_limits[i] = override_prior_limits[i];
		new_param_names[i] = names[i];
		new_override_names[i] = override_names[i];
	}
	for (i=pi; i < pf; i++) {
		newpriors[i] = new ParamPrior();
		newtransforms[i] = new ParamTransform();
		new_stepsizes[i] = stepsizes_in[i];
		new_auto_stepsize[i] = true; // stepsizes for newly added parameters are set to 'auto'
		new_hist2d_param[i] = true; // stepsizes for newly added parameters are set to 'auto'
		new_subplot_param[i] = false; // stepsizes for newly added parameters are set to 'auto'
		new_prior_norms[i] = 1.0;
		new_penalty_limits_lo[i] = -1e30;
		new_penalty_limits_hi[i] = 1e30;
		new_use_penalty_limits[i] = false;
		new_override_limits_lo[i] = -1e30;
		new_override_limits_hi[i] = 1e30;
		new_override_prior_limits[i] = false;
		new_param_names[i] = names[i];
		new_override_names[i] = "";
	}
	for (j=pf,i=pi; i < nparams; i++, j++) {
		new_paramnums[i] = j;
		newpriors[j] = new ParamPrior(priors[i]);
		newtransforms[j] = new ParamTransform(transforms[i]);
		new_auto_stepsize[j] = auto_stepsize[i];
		new_hist2d_param[j] = hist2d_param[i];
		new_subplot_param[j] = subplot_param[i];
		new_stepsizes[j] = (auto_stepsize[i]) ? stepsizes_in[j] : stepsizes[i]; // if stepsizes are set to 'auto', use new auto stepsizes since they might have changed
		new_prior_norms[j] = prior_norms[i];
		new_penalty_limits_lo[j] = penalty_limits_lo[i];
		new_penalty_limits_hi[j] = penalty_limits_hi[i];
		new_use_penalty_limits[j] = use_penalty_limits[i];
		new_override_limits_lo[j] = override_limits_lo[i];
		new_override_limits_hi[j] = override_limits_hi[i];
		new_override_prior_limits[j] = override_prior_limits[i];
		new_param_names[j] = param_names[i];
		new_override_names[j] = override_names[i];
		//cout << "Penalty limit " << i << " is now penalty limit " << j << ", which is " << use_penalty_limits[i] << " with range " << penalty_limits_lo[i] << " " << penalty_limits_hi[i] << endl;
	}
	if (nparams > 0) {
		for (i=0; i < nparams; i++) {
			delete priors[i];
			delete transforms[i];
		}
		delete[] priors;
		delete[] transforms;
		delete[] stepsizes;
		delete[] auto_stepsize;
		delete[] subplot_param;
		delete[] hist2d_param;
		delete[] prior_norms;
		delete[] penalty_limits_lo;
		delete[] penalty_limits_hi;
		delete[] use_penalty_limits;
		delete[] override_limits_lo;
		delete[] override_limits_hi;
		delete[] override_prior_limits;
		delete[] param_names;
		delete[] override_names;
	}
	priors = newpriors;
	transforms = newtransforms;
	stepsizes = new_stepsizes;
	auto_stepsize = new_auto_stepsize;
	subplot_param = new_subplot_param;
	hist2d_param = new_hist2d_param;
	prior_norms = new_prior_norms;
	penalty_limits_lo = new_penalty_limits_lo;
	penalty_limits_hi = new_penalty_limits_hi;
	use_penalty_limits = new_use_penalty_limits;
	override_limits_lo = new_override_limits_lo;
	override_limits_hi = new_override_limits_hi;
	override_prior_limits = new_override_prior_limits;
	param_names = new_param_names;
	override_names = new_override_names;
	nparams = new_nparams;
	update_reference_paramnums(new_paramnums);
	delete[] new_paramnums;
}

bool ParamSettings::remove_params(const int pi, const int pf)
{
	if (pf > nparams) return false;
	int i, j, np = pf-pi;
	if (np==nparams) {
		clear_params();
		return true;
	}
	int new_nparams = nparams - np;
	int* new_paramnums = new int[nparams]; // this stores the update parameter numbers for parameters that already existed
	for (i=0; i < nparams; i++) new_paramnums[i] = -1;
	ParamPrior** newpriors = new ParamPrior*[new_nparams];
	ParamTransform** newtransforms = new ParamTransform*[new_nparams];
	double* new_stepsizes = new double[new_nparams];
	bool* new_auto_stepsize = new bool[new_nparams];
	bool* new_subplot_param = new bool[new_nparams];
	bool* new_hist2d_param = new bool[new_nparams];
	double* new_prior_norms = new double[new_nparams];
	double* new_penalty_limits_lo = new double[new_nparams];
	double* new_penalty_limits_hi = new double[new_nparams];
	bool* new_use_penalty_limits = new bool[new_nparams];
	double* new_override_limits_lo = new double[new_nparams];
	double* new_override_limits_hi = new double[new_nparams];
	bool* new_override_prior_limits = new bool[new_nparams];
	string* new_param_names = new string[new_nparams];
	string* new_override_names = new string[new_nparams];
	for (i=0; i < pi; i++) {
		new_paramnums[i] = i;
		newpriors[i] = new ParamPrior(priors[i]);
		newtransforms[i] = new ParamTransform(transforms[i]);
		new_stepsizes[i] = stepsizes[i];
		new_auto_stepsize[i] = auto_stepsize[i];
		new_subplot_param[i] = subplot_param[i];
		new_hist2d_param[i] = hist2d_param[i];
		new_prior_norms[i] = prior_norms[i];
		new_penalty_limits_lo[i] = penalty_limits_lo[i];
		new_penalty_limits_hi[i] = penalty_limits_hi[i];
		new_use_penalty_limits[i] = use_penalty_limits[i];
		new_override_limits_lo[i] = override_limits_lo[i];
		new_override_limits_hi[i] = override_limits_hi[i];
		new_override_prior_limits[i] = override_prior_limits[i];
		new_param_names[i] = param_names[i];
		new_override_names[i] = override_names[i];
	}
	for (i=pf,j=pi; i < nparams; i++, j++) {
		new_paramnums[i] = j;
		newpriors[j] = new ParamPrior(priors[i]);
		newtransforms[j] = new ParamTransform(transforms[i]);
		new_stepsizes[j] = stepsizes[i];
		new_auto_stepsize[j] = auto_stepsize[i];
		new_subplot_param[j] = subplot_param[i];
		new_hist2d_param[j] = hist2d_param[i];
		new_prior_norms[j] = prior_norms[i];
		new_penalty_limits_lo[j] = penalty_limits_lo[i];
		new_penalty_limits_hi[j] = penalty_limits_hi[i];
		new_use_penalty_limits[j] = use_penalty_limits[i];
		new_override_limits_lo[j] = override_limits_lo[i];
		new_override_limits_hi[j] = override_limits_hi[i];
		new_override_prior_limits[j] = override_prior_limits[i];
		new_param_names[j] = param_names[i];
		new_override_names[j] = override_names[i];
	}
	for (i=0; i < nparams; i++) {
		delete priors[i];
		delete transforms[i];
	}
	delete[] priors;
	delete[] transforms;
	delete[] stepsizes;
	delete[] auto_stepsize;
	delete[] subplot_param;
	delete[] hist2d_param;
	delete[] prior_norms;
	delete[] penalty_limits_lo;
	delete[] penalty_limits_hi;
	delete[] use_penalty_limits;
	delete[] override_limits_lo;
	delete[] override_limits_hi;
	delete[] override_prior_limits;
	delete[] param_names;
	delete[] override_names;
	priors = newpriors;
	transforms = newtransforms;
	stepsizes = new_stepsizes;
	auto_stepsize = new_auto_stepsize;
	subplot_param = new_subplot_param;
	hist2d_param = new_hist2d_param;
	prior_norms = new_prior_norms;
	penalty_limits_lo = new_penalty_limits_lo;
	penalty_limits_hi = new_penalty_limits_hi;
	use_penalty_limits = new_use_penalty_limits;
	override_limits_lo = new_override_limits_lo;
	override_limits_hi = new_override_limits_hi;
	override_prior_limits = new_override_prior_limits;
	param_names = new_param_names;
	override_names = new_override_names;
	nparams = new_nparams;
	update_reference_paramnums(new_paramnums);
	delete[] new_paramnums;
	return true;
}
void ParamSettings::add_dparam(string dparam_name)
{
	string* new_dparam_names = new string[n_dparams+1];
	bool* new_subplot_dparam = new bool[n_dparams+1];
	bool* new_hist2d_dparam = new bool[n_dparams+1];
	if (n_dparams > 0) {
		for (int i=0; i < n_dparams; i++) {
			new_dparam_names[i] = dparam_names[i];
			new_subplot_dparam[i] = subplot_dparam[i];
			new_hist2d_dparam[i] = hist2d_dparam[i];
		}
		delete[] dparam_names;
		delete[] subplot_dparam;
		delete[] hist2d_dparam;
	}
	new_dparam_names[n_dparams] = dparam_name;
	new_subplot_dparam[n_dparams] = false;
	new_hist2d_dparam[n_dparams] = true;
	n_dparams++;
	dparam_names = new_dparam_names;
	subplot_dparam = new_subplot_dparam;
	hist2d_dparam = new_hist2d_dparam;
}
void ParamSettings::remove_dparam(int dparam_number)
{
	string* new_dparam_names;
	bool* new_subplot_dparam;
	bool* new_hist2d_dparam;
	if (n_dparams > 1) {
		new_dparam_names = new string[n_dparams-1];
		new_subplot_dparam = new bool[n_dparams-1];
		new_hist2d_dparam = new bool[n_dparams-1];
		int i,j;
		for (i=0, j=0; i < n_dparams; i++) {
			if (i != dparam_number) {
				new_dparam_names[j] = dparam_names[i];
				new_subplot_dparam[j] = subplot_dparam[i];
				new_hist2d_dparam[j] = hist2d_dparam[i];
				j++;
			}
		}
	}
	delete[] dparam_names;
	delete[] subplot_dparam;
	delete[] hist2d_dparam;
	n_dparams--;
	if (n_dparams > 0) {
		dparam_names = new_dparam_names;
		subplot_dparam = new_subplot_dparam;
		hist2d_dparam = new_hist2d_dparam;
	} else {
		dparam_names = NULL;
		subplot_dparam = NULL;
		hist2d_dparam = NULL;
	}
}

void ParamSettings::print_priors()
{
	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter settings:\n";
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
		if (!output_prior(i)) die("Prior type unknown");
		if (transforms[i]->transform==NONE) ;
		else if (transforms[i]->transform==LOG_TRANSFORM) cout << ", log transformation";
		else if (transforms[i]->transform==GAUSS_TRANSFORM) cout << ", gaussian transformation (mean=" << transforms[i]->gaussian_pos << ", sigma=" << transforms[i]->gaussian_sig << ")";
		else if (transforms[i]->transform==LINEAR_TRANSFORM) cout << ", linear transformation A*" << param_names[i] << " + b (A=" << transforms[i]->a << ", b=" << transforms[i]->b << ")";
		else if (transforms[i]->transform==RATIO) cout << ", ratio transformation " << param_names[i] << "/" << param_names[transforms[i]->ratio_paramnum];
		if (transforms[i]->include_jacobian==true) cout << " (include Jacobian in likelihood)";
		cout << endl;
	}
}

bool ParamSettings::output_prior(const int i)
{
	if (priors[i]->prior==UNIFORM_PRIOR) ;
	if (priors[i]->prior==UNIFORM_PRIOR) {
		cout << "uniform prior";
		if ((transforms[i]->transform != NONE) and (transforms[i]->include_jacobian)) {
			cout << " in " << param_names[i];
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


void ParamSettings::print_stepsizes()
{
	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter initial stepsizes:\n";
	string *transformed_names = new string[nparams];
	transform_parameter_names(param_names,transformed_names,NULL,NULL);
	int max_length=0;
	for (int i=0; i < nparams; i++) {
		if (transformed_names[i].length() > max_length) max_length = transformed_names[i].length();
	}
	int extra_length;
	for (int i=0; i < nparams; i++) {
		cout << i << ". " << transformed_names[i] << ": ";
		extra_length = max_length - transformed_names[i].length();
		for (int j=0; j < extra_length; j++) cout << " ";
		if ((nparams > 10) and (i < 10)) cout << " ";
		cout << stepsizes[i];
		if (auto_stepsize[i]) cout << " (auto)";
		cout << endl;
	}
	delete[] transformed_names;
}

void ParamSettings::print_penalty_limits()
{
	//for (int i=0; i < nparams; i++) {
		//if (use_penalty_limits[i]==true) {
			//cout << "USE_LIMITS " << i << endl;
		//}
	//}

	if (nparams==0) { cout << "No fit parameters have been defined\n"; return; }
	cout << "Parameter limits imposed on chi-square:\n";
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
		if (use_penalty_limits[i]==false) cout << "none" << endl;
		else {
			cout << "[";
			if (penalty_limits_lo[i]==-1e30) cout << "-inf";
			else cout << penalty_limits_lo[i];
			cout << ":";
			if (penalty_limits_hi[i]==1e30) cout << "inf";
			else cout << penalty_limits_hi[i];
			cout << "]" << endl;
		}
	}
}


