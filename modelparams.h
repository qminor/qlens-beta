#ifndef MODELPARAMS_H
#define MODELPARAMS_H

#include "vector.h"
#include <vector>
#include <string>

class ModelParams
{
	// This is the base class inherited by pixellated source and point source objects (lens/source analytic profiles
	// use separate functions for handling parameters)
	public:
	double **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model
	int n_params, n_vary_params, n_active_params;
	boolvector vary_params;
	boolvector active_params; // this keeps track of which parameters are actually being used, based on the mode of regularization, pixellation etc.
	std::string model_name;
	std::vector<std::string> paramnames;
	std::vector<std::string> latex_paramnames, latex_param_subscripts;
	boolvector set_auto_penalty_limits;
	dvector penalty_upper_limits, penalty_lower_limits;
	dvector stepsizes;
	boolvector scale_stepsize_by_param_value;
	bool include_limits;
	dvector lower_limits, upper_limits;
	dvector lower_limits_initial, upper_limits_initial;

	ModelParams() { param = NULL; }
	void setup_parameter_arrays(const int npar);
	virtual void setup_parameters(const bool initial_setup) {} 
	virtual void update_meta_parameters(const bool varied_only_fitparams) {}
	void copy_param_arrays(ModelParams* params_in);
	void update_active_params(const int id) {
		boolvector dummy;
		update_active_params(id,dummy);
	}
	void update_active_params(const int id, boolvector& turned_off_params) {
		turned_off_params.input(n_vary_params);
		setup_parameters(false);
		// check: if any parameters are no longer active, but they were being varied, turn off their vary flags
		int i,j;
		for (i=0,j=0; i < n_params; i++) {
			if ((!active_params[i]) and (vary_params[i])) {
				if (id==0) std::cout << "Parameter " << paramnames[i] << " is no longer active, so its vary flag is being turned off" << std::endl;
				vary_params[i] = false;
				n_vary_params--;
				turned_off_params[j++] = true;
			} else if (vary_params[i]) {
				turned_off_params[j++] = false;
			}
		}
	}


	void update_fit_parameters(const double* fitparams, int &index);
	bool update_specific_parameter(const std::string name_in, const double& value);
	bool set_varyflags(const boolvector& vary_in);
	bool update_specific_varyflag(const std::string name_in, const bool& vary_in);
	void set_limits(const dvector& lower, const dvector& upper, const dvector& lower_init, const dvector& upper_init);
	void set_limits(const dvector& lower, const dvector& upper) { set_limits(lower,upper,lower,upper); }
	bool set_limits_specific_parameter(const std::string name_in, const double& lower, const double& upper);
	bool get_limits(dvector& lower, dvector& upper, dvector& lower0, dvector& upper0, int &index);
	bool get_limits(dvector& lower, dvector& upper, int &index);
	void get_auto_stepsizes(dvector& stepsizes, int &index);
	void get_auto_ranges(boolvector& use_penalty_limits, dvector& lower, dvector& upper, int &index);

	void get_fit_parameters(dvector& fitparams, int &index);
	void get_fit_parameter_names(std::vector<std::string>& paramnames_vary, std::vector<std::string> *latex_paramnames_vary = NULL, std::vector<std::string> *latex_subscripts_vary = NULL);
	bool get_specific_parameter(const std::string name_in, double& value);
	bool get_specific_varyflag(const std::string name_in, bool& flag);
	void get_parameter_number(const std::string name_in, int& paramnum);
	void get_parameter_vary_index(const std::string name_in, int& index);

	void get_varyflags(boolvector& flags);
	void print_parameters(const bool show_only_varying_params = false);
	void print_vary_parameters();
	void set_include_limits(bool inc) { include_limits = inc; }
	int get_n_vary_params() { return n_vary_params; }
};

#endif // MODELPARAMS_H
