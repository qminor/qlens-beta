#ifndef MODELPARAMS_H
#define MODELPARAMS_H

#include "vector.h"
#include <vector>
#include <string>

#ifdef USE_STAN
#include <stan/math.hpp>
#endif

class QLens;

template <typename QScalar>
class ModelParams
{
	public:
	QScalar **param; // this is an array of pointers, each of which points to the corresponding indexed parameter for each model

	ModelParams() {
		param = NULL;
	}
};

class Model
{
	// This is the base class inherited by pixellated source and point source objects (lens/source analytic profiles
	// use separate functions for handling parameters) as well as the cosmology class and qlens class.
	protected:
	QLens* qlens;

	public:
	ModelParams<double>* modelparams; // this will point to the corresponding lensparams in the inherited classes
#ifdef USE_STAN
	ModelParams<stan::math::var>* modelparams_dif; // this will point to the corresponding lensparams in the inherited classes
#endif

	template <typename QScalar>
	ModelParams<QScalar>& assign_modelparam_object()
	{
#ifdef USE_STAN
		if constexpr (std::is_same_v<QScalar, stan::math::var>)
			return (*modelparams_dif);
		else
#endif
		return (*modelparams);
	}

	double **param; // this is about to get replaced by modelparams.param

	int n_params, n_vary_params, n_active_params;
	int entry_number; // if this object is in an array, keeps track of where it is in list
	boolvector vary_params;
	boolvector active_params; // this keeps track of which parameters are actually being used, based on the mode of regularization, pixellation etc.
	std::string model_name;
	std::vector<std::string> paramnames;
	std::vector<std::string> latex_paramnames, latex_param_subscripts;
	boolvector set_auto_penalty_limits;
	Vector<double> penalty_upper_limits, penalty_lower_limits;
	Vector<double> stepsizes;
	boolvector scale_stepsize_by_param_value;
	bool include_limits;
	Vector<double> lower_limits, upper_limits;

	Model() {
		modelparams = NULL;
#ifdef USE_STAN
		modelparams_dif = NULL;
#endif
		param = NULL;
		qlens = NULL;
		entry_number = -1;
	}
	void set_qlens(QLens* qlensptr) { qlens = qlensptr; }
	void setup_parameter_arrays(const int npar);
	virtual void setup_parameters(const bool initial_setup) {} 
	virtual void update_meta_parameters(const bool varied_only_fitparams) {}
	virtual void get_parameter_numbers_from_qlens(int& pi, int& pf);
	virtual bool register_vary_parameters_in_qlens();
	virtual void register_limits_in_qlens();
	virtual void update_fitparams_in_qlens();
#ifdef USE_STAN
	virtual void sync_autodif_parameters() {}
#endif
	void copy_param_arrays(const Model* params_in);
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

	template <typename QScalar>
	void update_fit_parameters(const QScalar* fitparams, int &index);
	void update_fit_parameters_doub(const double* fitparams, int &index);  // temporary for objects that don't have autodif implemented yet
	bool update_specific_parameter(const std::string name_in, const double& value);
	bool set_varyflags(const boolvector& vary_in);
	bool update_specific_varyflag(const std::string name_in, const bool& vary_in);
	void set_limits(const Vector<double>& lower, const Vector<double>& upper);
	bool set_limits_specific_parameter(const std::string name_in, const double& lower, const double& upper);
	void update_limits(const double* lower, const double* upper, const bool* limits_changed, int& index);
	bool get_limits(Vector<double>& lower, Vector<double>& upper);
	bool get_limits(Vector<double>& lower, Vector<double>& upper, int &index);
	void get_auto_stepsizes(Vector<double>& stepsizes, int &index);
	void get_auto_ranges(boolvector& use_penalty_limits, Vector<double>& lower, Vector<double>& upper, int &index);

	void get_fit_parameters(double *fitparams, int &index);
	void get_fit_parameter_names(std::vector<std::string>& paramnames_vary, std::vector<std::string> *latex_paramnames_vary = NULL, std::vector<std::string> *latex_subscripts_vary = NULL);
	bool check_parameter_name(const std::string name_in);
	bool get_specific_parameter(const std::string name_in, double& value);
	bool get_specific_varyflag(const std::string name_in, bool& flag);
	bool get_specific_stepsize(const std::string name_in, double& step);
	bool get_specific_limit(const std::string name_in, double& lower, double& upper);
	void get_parameter_number(const std::string name_in, int& paramnum);
	void get_parameter_vary_index(const std::string name_in, int& index);

	void get_varyflags(boolvector& flags);
	void print_parameters(const bool show_only_varying_params = false);
	void print_vary_parameters();
	std::string mkstring_int(const int i);
	std::string mkstring_doub(const double db);
	std::string get_parameters_string();
	void set_include_limits(bool inc) { include_limits = inc; }
	int get_n_vary_params() { return n_vary_params; }
	QLens* get_qlensptr() { return qlens; }
};

#endif // MODELPARAMS_H
