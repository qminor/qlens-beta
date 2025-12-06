#include "modelparams.h"
#include "params.h"
#include <sstream>

using namespace std;

/********* Functions in class ModelParams (which is inherited by PointSource, SourcePixelGrid, DelaunayGrid *********/

void ModelParams::setup_parameter_arrays(const int npar)
{
	n_params = npar; // number of all possible parameters that can be varied (not necessarily at the same time)
	include_limits = false; // default
	active_params.input(n_params);
	vary_params.input(n_params);
	paramnames.resize(n_params);
	latex_paramnames.resize(n_params);
	latex_param_subscripts.resize(n_params);
	stepsizes.input(n_params);
	scale_stepsize_by_param_value.input(n_params);
	set_auto_penalty_limits.input(n_params);
	penalty_lower_limits.input(n_params);
	penalty_upper_limits.input(n_params);
	lower_limits.input(n_params);
	upper_limits.input(n_params);
	if (param != NULL) delete[] param;
	param = new double*[n_params];
	n_vary_params = 0;
	for (int i=0; i < n_params; i++) {
		scale_stepsize_by_param_value[i] = false; // the choices after initialization (below) will never change (judgment call determined by the type of parameter)
		vary_params[i] = false; // default
	}
	n_active_params = 0;
	for (int i=0; i < n_params; i++) {
		active_params[i] = false; // default
	}
}

void ModelParams::copy_param_arrays(const ModelParams* params_in)
{
	// REMEMBER: you must still run setup_parameters() in the inherited class to initialize and assign the pointers before running this!
	// note, we do not copy n_params, nor resize most of the vectors, because it is assumed that this has been setup already in setup_parameters()
	n_vary_params = params_in->n_vary_params;
	paramnames = params_in->paramnames;
	latex_paramnames = params_in->latex_paramnames;
	latex_param_subscripts = params_in->latex_param_subscripts;
	include_limits = params_in->include_limits;
	vary_params.input(params_in->vary_params);
	stepsizes.input(params_in->stepsizes);
	scale_stepsize_by_param_value.input(params_in->scale_stepsize_by_param_value);
	set_auto_penalty_limits.input(params_in->set_auto_penalty_limits);
	penalty_lower_limits.input(params_in->penalty_lower_limits);
	penalty_upper_limits.input(params_in->penalty_upper_limits);
	lower_limits.input(params_in->lower_limits);
	upper_limits.input(params_in->upper_limits);
}

void ModelParams::update_fit_parameters(const double* fitparams, int &index)
{
	if (n_vary_params > 0) {
		for (int i=0; i < n_params; i++) {
			if ((active_params[i]) and (vary_params[i]==true)) {
				*(param[i]) = fitparams[index++];
			}
		}
		update_meta_parameters(true);
	}
}

bool ModelParams::update_specific_parameter(const string name_in, const double& value)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (paramnames[i]==name_in)) {
			*(param[i]) = value;
			found_match = true;
			update_meta_parameters(false);
			break;
		}
	}
	update_fitparams_in_qlens();
	return found_match;
}

void ModelParams::update_fitparams_in_qlens()
{
	// this is a virtual function that must be defined for each inherited class
	return;
}

void ModelParams::get_parameter_number(const string name_in, int& paramnum)
{
	paramnum = -1;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (paramnames[i]==name_in)) {
			paramnum = i;
			break;
		}
	}
}

void ModelParams::get_parameter_vary_index(const string name_in, int& index)
{
	index = 0;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (paramnames[i]==name_in)) {
			break;
		}
		if (vary_params[i]) index++;
	}
}

bool ModelParams::set_varyflags(const boolvector& vary_in)
{
	if (vary_in.size() != n_active_params) {
		return false;
	}
	int pi, pf;
	get_parameter_numbers_from_qlens(pi,pf); // these are the old parameter numbers

	int i,j;
	n_vary_params = 0;
	for (i=0,j=0; i < n_params; i++) {
		if (active_params[i]) {
			vary_params[i] = vary_in[j];
			if (vary_in[j]) n_vary_params++;
			j++;
		}
	}
	if (qlens) {
		if (pf > pi) qlens->param_list->remove_params(pi,pf); // eliminate any old fit parameters so they can be replaced with the new ones
		return register_vary_parameters_in_qlens();
	}
	return true;
}

void ModelParams::get_parameter_numbers_from_qlens(int& pi, int& pf)
{
	// this is a virtual function that must be defined for each inherited class
	return;
}

bool ModelParams::register_vary_parameters_in_qlens()
{
	// this is a virtual function that must be defined for each inherited class
	return true;
}

bool ModelParams::update_specific_varyflag(const string name_in, const bool& vary_in)
{
	if (n_active_params==0) return false;
	int param_i = -1;
	int i;
	for (i=0; i < n_params; i++) {
		if (!active_params[i]) continue;
		if (paramnames[i]==name_in) {
			param_i = i;
			break;
		}
	}
	if (param_i != -1) {
		if (vary_params[param_i] != vary_in) {
			vary_params[param_i] = vary_in;
			if (vary_in) n_vary_params++;
			else n_vary_params--;
		}
	}
	return (param_i != -1);
}

void ModelParams::set_limits(const dvector& lower, const dvector& upper)
{
	if (lower.size() != n_vary_params) die("lower limits array does not match number of variable parameters in source object (%i vs %i)",lower.size(),n_vary_params);
	include_limits = true;

	int i,j;
	for (i=0,j=0; i < n_params; i++) {
		if ((active_params[i]) and (vary_params[i])) {
			lower_limits[i] = lower[j];
			upper_limits[i] = upper[j];
			j++;
		}
	}
	register_limits_in_qlens();
}

void ModelParams::update_limits(const double* lower, const double* upper, const bool* limits_changed, int& index)
{
	// in this case, the limits are being updated from the fitparams list, so there is no need to call register_lens_prior_limits
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (vary_params[i])) {
			lower_limits[i] = lower[index];
			upper_limits[i] = upper[index];
			index++;
		}
	}
}

bool ModelParams::set_limits_specific_parameter(const string name_in, const double& lower, const double& upper)
{
	int param_i = -1;
	int i;
	for (i=0; i < n_params; i++) {
		if (!active_params[i]) continue;
		if (paramnames[i]==name_in) {
			param_i = i;
			break;
		}
	}
	if (param_i != -1) {
		if (!include_limits) include_limits = true;
		lower_limits[param_i] = lower;
		upper_limits[param_i] = upper;
	}
	register_limits_in_qlens();
	return (param_i != -1);
}

void ModelParams::register_limits_in_qlens()
{
	// this is a virtual function that must be defined for each inherited class
	return;
}

void ModelParams::get_fit_parameters(double *fitparams, int &index)
{
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]==true) {
			fitparams[index++] = *(param[i]);
		}
	}
}

void ModelParams::get_auto_stepsizes(dvector& stepsizes_in, int &index)
{
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]) {
			if (scale_stepsize_by_param_value[i]) {
				stepsizes_in[index++] = stepsizes[i]*(*(param[i]));
			} else {
				stepsizes_in[index++] = stepsizes[i];
			}
		}
	}
}

void ModelParams::get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index)
{
	for (int i=0; i < n_params; i++) {
		if (vary_params[i]) {
			if (set_auto_penalty_limits[i]) {
				use_penalty_limits[index] = true;
				lower[index] = penalty_lower_limits[i];
				upper[index] = penalty_upper_limits[i];
			} else {
				use_penalty_limits[index] = false;
			}
			index++;
		}
	}
}


void ModelParams::get_fit_parameter_names(vector<string>& paramnames_vary, vector<string> *latex_paramnames_vary, vector<string> *latex_subscripts_vary)
{
	int i;
	for (i=0; i < n_params; i++) {
		if (vary_params[i]) {
			paramnames_vary.push_back(paramnames[i]);
			if (latex_paramnames_vary != NULL) latex_paramnames_vary->push_back(latex_paramnames[i]);
			if (latex_subscripts_vary != NULL) latex_subscripts_vary->push_back(latex_param_subscripts[i]);
		}
	}
}

bool ModelParams::get_specific_parameter(const string name_in, double& value)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (paramnames[i]==name_in)) {
			value = *(param[i]);
			found_match = true;
			break;
		}
	}
	return found_match;
}

bool ModelParams::get_specific_stepsize(const string name_in, double& step)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (paramnames[i]==name_in)) {
			if (scale_stepsize_by_param_value[i]) {
				step = stepsizes[i]*(*(param[i]));
			} else {
				step = stepsizes[i];
			}
			found_match = true;
			break;
		}
	}
	return found_match;
}

bool ModelParams::get_specific_varyflag(const string name_in, bool& flag)
{
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (paramnames[i]==name_in)) {
			flag = vary_params[i];
			found_match = true;
			break;
		}
	}
	return found_match;
}

bool ModelParams::get_specific_limit(const string name_in, double& lower, double& upper)
{
	if (include_limits==false) return false;
	bool found_match = false;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (vary_params[i]) and (paramnames[i]==name_in)) {
			lower = lower_limits[i];
			upper = upper_limits[i];
			found_match = true;
			break;
		}
	}
	return found_match;
}

void ModelParams::get_varyflags(boolvector& flags)
{
	flags.input(n_vary_params);
	int index = 0;
	for (int i=0; i < n_params; i++) {
		if (active_params[i]) {
			flags[index++] = vary_params[i];
		}
	}
}

bool ModelParams::get_limits(dvector& lower, dvector& upper)
{
	int index = 0;
	if (include_limits==false) return false;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (vary_params[i])) {
			lower[index] = lower_limits[i];
			upper[index] = upper_limits[i];
			index++;
		}
	}
	return true;
}

bool ModelParams::get_limits(dvector& lower, dvector& upper, int &index)
{
	if (include_limits==false) return false;
	for (int i=0; i < n_params; i++) {
		if ((active_params[i]) and (vary_params[i])) {
			lower[index] = lower_limits[i];
			upper[index] = upper_limits[i];
			index++;
		}
	}
	return true;
}

void ModelParams::print_parameters(const bool show_only_varying_params)
{
	if (n_active_params == 0) cout << "no active parameters";
	else {
		int j=0;
		for (int i=0; i < n_params; i++) {
			if (active_params[i]) {
				if ((!show_only_varying_params) or (vary_params[i])) {
					cout << paramnames[i] << "=";
					cout << *(param[i]);
					if (j != n_active_params-1) cout << ", ";
					j++;
				}
			}
		}
	}
	cout << endl;
}

string ModelParams::mkstring_int(const int i)
{
	stringstream istr;
	string istring;
	istr << i;
	istr >> istring;
	return istring;
}

string ModelParams::mkstring_doub(const double db)
{
	stringstream dstr;
	string dstring;
	dstr << db;
	dstr >> dstring;
	return dstring;
}

// This function is used by the Python wrapper
string ModelParams::get_parameters_string()
{
	string paramstring = "";
	if (entry_number >= 0) paramstring += mkstring_int(entry_number) + ". ";
	if (model_name != "") paramstring += model_name + ": ";
	if (n_active_params == 0) paramstring += "no active parameters";
	else {
		int j=0;
		for (int i=0; i < n_params; i++) {
			if (active_params[i]) {
				paramstring += paramnames[i] + "=";
				paramstring += mkstring_doub(*(param[i]));
				if (j != n_active_params-1) paramstring += ", ";
				j++;
			}
		}
	}

	return paramstring;
}



void ModelParams::print_vary_parameters()
{
	if (n_vary_params==0) {
		cout << "   parameters: none\n";
	} else {
		vector<string> paramnames_vary;
		get_fit_parameter_names(paramnames_vary);
		if (include_limits) {
			dvector lower_lims(n_vary_params);
			dvector upper_lims(n_vary_params);
			int indx=0;
			bool status = get_limits(lower_lims,upper_lims,indx);
			if (!status) cout << "   Warning: parameter limits not defined\n";
			else {
				cout << "   parameter limits:\n";
				for (int i=0; i < n_vary_params; i++) {
					cout << "   " << paramnames_vary[i] << ": [" << lower_lims[i] << ":" << upper_lims[i] << "]\n";
				}
			}
		} else {
			cout << "   parameters: ";
			cout << paramnames_vary[0];
			for (int i=1; i < n_vary_params; i++) cout << ", " << paramnames_vary[i];
			cout << endl;
		}
	}
}


