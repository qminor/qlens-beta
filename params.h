#ifndef PARAMS_H
#define PARAMS_H

#include "qlens.h"

using std::string;

struct ParamPrior
{
	double gaussian_pos, gaussian_sig;
	dmatrix covariance_matrix, inv_covariance_matrix;
	dvector gauss_meanvals;
	ivector gauss_paramnums;
	Prior prior;
	ParamPrior() { prior = UNIFORM_PRIOR; }
	ParamPrior(ParamPrior *prior_in)
	{
		prior = prior_in->prior;
		if (prior==GAUSS_PRIOR) {
			gaussian_pos = prior_in->gaussian_pos;
			gaussian_sig = prior_in->gaussian_sig;
		}
		else if (prior==GAUSS2_PRIOR) {
			gauss_paramnums.input(prior_in->gauss_paramnums);
			gauss_meanvals.input(prior_in->gauss_meanvals);
			inv_covariance_matrix.input(prior_in->inv_covariance_matrix);
		}
	}
	void set_uniform() { prior = UNIFORM_PRIOR; }
	void set_log() { prior = LOG_PRIOR; }
	void set_gaussian(double &pos_in, double &sig_in) { prior = GAUSS_PRIOR; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void set_gauss2(int p1, int p2, double &pos1_in, double &pos2_in, double &sig1_in, double &sig2_in, double &sig12_in) {
		prior = GAUSS2_PRIOR;
		gauss_paramnums.input(2);
		gauss_meanvals.input(2);
		covariance_matrix.input(2,2);
		gauss_paramnums[0] = p1;
		gauss_paramnums[1] = p2;
		gauss_meanvals[0] = pos1_in;
		gauss_meanvals[1] = pos2_in;
		covariance_matrix[0][0] = SQR(sig1_in);
		covariance_matrix[1][1] = SQR(sig2_in);
		covariance_matrix[0][1] = SQR(sig12_in);
		covariance_matrix[1][0] = covariance_matrix[0][1];
		inv_covariance_matrix.input(2,2);
		inv_covariance_matrix = covariance_matrix.inverse();
	}
	void set_gauss2_secondary(int p1, int p2) {
		prior = GAUSS2_PRIOR_SECONDARY;
		gauss_paramnums.input(2);
		gauss_paramnums[0] = p1;
		gauss_paramnums[1] = p2;
	}
};

struct ParamTransform
{
	double gaussian_pos, gaussian_sig;
	double a, b; // for linear transformations
	bool include_jacobian;
	int ratio_paramnum;
	Transform transform;
	ParamTransform() { transform = NONE; include_jacobian = false; }
	ParamTransform(ParamTransform *transform_in)
	{
		transform = transform_in->transform;
		include_jacobian = transform_in->include_jacobian;
		if (transform==GAUSS_TRANSFORM) {
			gaussian_pos = transform_in->gaussian_pos;
			gaussian_sig = transform_in->gaussian_sig;
		} else if (transform==LINEAR_TRANSFORM) {
			a = transform_in->a;
			b = transform_in->b;
		} else if (transform==RATIO) {
			ratio_paramnum = transform_in->ratio_paramnum;
		}
	}
	void set_none() { transform = NONE; }
	void set_log() { transform = LOG_TRANSFORM; }
	void set_linear(double &a_in, double &b_in) { transform = LINEAR_TRANSFORM; a = a_in; b = b_in; }
	void set_gaussian(double &pos_in, double &sig_in) { transform = GAUSS_TRANSFORM; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void set_ratio(int &paramnum_in) { transform = RATIO; ratio_paramnum = paramnum_in; }
	void set_include_jacobian(bool &include) { include_jacobian = include; }
};

struct DerivedParam
{
	DerivedParamType derived_param_type;
	double funcparam; // if funcparam == -1, then there is no parameter required
	double funcparam2;
	bool use_kpc_units;
	int int_param;
	string name, latex_name;
	DerivedParam(DerivedParamType type_in, double param, int lensnum, double param2 = -1, bool usekpc = false) // if lensnum == -1, then it uses *all* the lenses (if possible)
	{
		derived_param_type = type_in;
		funcparam = param;
		funcparam2 = param2;
		int_param = lensnum;
		use_kpc_units = usekpc;
		if (derived_param_type == KappaR) {
			name = "kappa"; latex_name = "\\kappa"; if (lensnum==-1) { name += "_tot"; latex_name += "_{tot}"; }
		} else if (derived_param_type == LambdaR) { // here lambda_R = 1 - <kappa>(R)
			name = "lambdaR"; latex_name = "\\lambda_R";
		} else if (derived_param_type == DKappaR) {
			name = "dkappa"; latex_name = "\\kappa'"; if (lensnum==-1) { name += "_tot"; latex_name += "_{tot}"; }
		} else if (derived_param_type == Mass2dR) {
			name = "mass2d"; latex_name = "M_{2D}";
		} else if (derived_param_type == Mass3dR) {
			name = "mass3d"; latex_name = "M_{3D}";
		} else if (derived_param_type == Einstein) {
			name = "re_zsrc"; latex_name = "R_{e}";
		} else if (derived_param_type == Einstein_Mass) {
			name = "mass_re"; latex_name = "M_{Re}";
		} else if (derived_param_type == Xi_Param) {
			name = "xi"; latex_name = "\\xi";
		} else if (derived_param_type == Kappa_Re) {
			name = "kappa_re"; latex_name = "\\kappa_{E}";
		} else if (derived_param_type == LensParam) {
			name = "lensparam"; latex_name = "\\lambda";
		} else if (derived_param_type == AvgLogSlope) {
			name = "logslope"; latex_name = "\\gamma_{avg}'";
		} else if (derived_param_type == Relative_Perturbation_Radius) {
			name = "r_perturb_rel"; latex_name = "\\Delta r_{\\delta c}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Perturbation_Radius) {
			name = "r_perturb"; latex_name = "r_{\\delta c}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Robust_Perturbation_Mass) {
			name = "mass_perturb"; latex_name = "M_{\\delta c}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Robust_Perturbation_Density) {
			name = "sigma_perturb"; latex_name = "\\Sigma_{\\delta c}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Chi_Square) {
			name = "raw_chisq"; latex_name = "\\chi^2";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Adaptive_Grid_qs) {
			name = "qs"; latex_name = "q_{s}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Adaptive_Grid_phi_s) {
			name = "phi_s"; latex_name = "\\phi_{s}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Adaptive_Grid_xavg) {
			name = "xavg_s"; latex_name = "x_{avg,s}";
			funcparam = -1e30; // no input parameter for this dparam
		} else if (derived_param_type == Adaptive_Grid_yavg) {
			name = "yavg_s"; latex_name = "y_{avg,s}";
			funcparam = -1e30; // no input parameter for this dparam
		} else die("no user defined function yet");

		if (funcparam != -1e30) {
			if (funcparam2==-1) {
				std::stringstream paramstr;
				string paramstring;
				paramstr << funcparam;
				paramstr >> paramstring;
				name += "(" + paramstring + ")";
				latex_name += "(" + paramstring + ")";
			} else {
				std::stringstream paramstr, paramstr2;
				string paramstring, paramstring2;
				paramstr << funcparam;
				paramstr >> paramstring;
				paramstr2 << funcparam2;
				paramstr2 >> paramstring2;
				name += "(" + paramstring + "," + paramstring2 + ")";
				latex_name += "(" + paramstring + "," + paramstring2 + ")";
			}
		}
	}
	double get_derived_param(QLens* lens_in)
	{
		if (derived_param_type == KappaR) return lens_in->total_kappa(funcparam,int_param,use_kpc_units);
		else if (derived_param_type == LambdaR) return (1 - lens_in->total_dkappa(funcparam,-1,use_kpc_units));
		else if (derived_param_type == DKappaR) return lens_in->total_dkappa(funcparam,int_param,use_kpc_units);
		else if (derived_param_type == Mass2dR) return lens_in->mass2d_r(funcparam,int_param,use_kpc_units);
		else if (derived_param_type == Mass3dR) return lens_in->mass3d_r(funcparam,int_param,use_kpc_units);
		else if (derived_param_type == Einstein) return lens_in->einstein_radius_single_lens(funcparam,int_param);
		else if (derived_param_type == Xi_Param) return lens_in->get_xi_parameter(funcparam,int_param);
		else if (derived_param_type == AvgLogSlope) return lens_in->calculate_average_log_slope(int_param,funcparam,funcparam2,use_kpc_units);
		else if (derived_param_type == Einstein_Mass) {
			double re = lens_in->einstein_radius_single_lens(funcparam,int_param);
			return lens_in->mass2d_r(re,int_param,false);
		} else if (derived_param_type == Kappa_Re) {
			double reav=0;
			lens_in->einstein_radius_of_primary_lens(lens_in->reference_zfactors[lens_in->lens_redshift_idx[lens_in->primary_lens_number]],reav);
			if (reav <= 0) return 0.0;
			else return lens_in->total_kappa(reav,-1,false);
		} else if (derived_param_type == LensParam) {
			return lens_in->get_lens_parameter_using_pmode((int)funcparam,int_param,(int)funcparam2);
		}
		else if (derived_param_type == Relative_Perturbation_Radius) {
			double rmax,avgsig,menc,rmax_z,avgkap_scaled;
			lens_in->calculate_critical_curve_perturbation_radius_numerical(int_param,false,rmax,avgsig,menc,rmax_z,avgkap_scaled,true);
			return rmax;
		} else if (derived_param_type == Perturbation_Radius) {
			double rmax,avgsig,menc,rmax_z,avgkap_scaled;
			lens_in->calculate_critical_curve_perturbation_radius_numerical(int_param,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
			return rmax;
		} else if (derived_param_type == Robust_Perturbation_Mass) {
			double rmax,avgsig,menc,rmax_z,avgkap_scaled;
			lens_in->calculate_critical_curve_perturbation_radius_numerical(int_param,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
			return menc;
		} else if (derived_param_type == Robust_Perturbation_Density) {
			double rmax,avgsig,menc,rmax_z,avgkap_scaled;
			lens_in->calculate_critical_curve_perturbation_radius_numerical(int_param,false,rmax,avgsig,menc,rmax_z,avgkap_scaled);
			return avgsig;
		} else if (derived_param_type == Adaptive_Grid_qs) {
			if (lens_in->lens_parent != NULL) {
				// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
				if (lens_in->raw_chisq==-1e30) lens_in->lens_parent->LogLikeFunc(NULL); // If a source inversion hasn't been performed yet, do it here
			} else if (lens_in->raw_chisq==-1e30) {
				lens_in->invert_surface_brightness_map_from_data(lens_in->raw_chisq, false, true); // If a source inversion hasn't been performed yet, do it here
			}
			double qs,phi_s,xavg,yavg;
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,xavg,yavg);
			return qs;
		} else if (derived_param_type == Adaptive_Grid_phi_s) {
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			if (lens_in->lens_parent != NULL) {
				// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
				if (lens_in->raw_chisq==-1e30) lens_in->lens_parent->LogLikeFunc(NULL); // If a source inversion hasn't been performed yet, do it here
			} else if (lens_in->raw_chisq==-1e30) {
				lens_in->invert_surface_brightness_map_from_data(lens_in->raw_chisq, false, true); // If a source inversion hasn't been performed yet, do it here
			}
			double qs,phi_s,phi_s_deg,xavg,yavg;
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,xavg,yavg);
			phi_s_deg = phi_s*180.0/M_PI;
			return phi_s_deg;
		} else if (derived_param_type == Adaptive_Grid_xavg) {
			if (lens_in->lens_parent != NULL) {
				// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
				if (lens_in->raw_chisq==-1e30) lens_in->lens_parent->LogLikeFunc(NULL); // If a source inversion hasn't been performed yet, do it here
			} else if (lens_in->raw_chisq==-1e30) {
				lens_in->invert_surface_brightness_map_from_data(lens_in->raw_chisq, false, true); // If a source inversion hasn't been performed yet, do it here
			}
			double qs,phi_s,xavg,yavg;
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,xavg,yavg);
			return xavg;
		} else if (derived_param_type == Adaptive_Grid_yavg) {
			if (lens_in->lens_parent != NULL) {
				// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
				if (lens_in->raw_chisq==-1e30) lens_in->lens_parent->LogLikeFunc(NULL); // If a source inversion hasn't been performed yet, do it here
			} else if (lens_in->raw_chisq==-1e30) {
				lens_in->invert_surface_brightness_map_from_data(lens_in->raw_chisq, false, true); // If a source inversion hasn't been performed yet, do it here
			}
			double qs,phi_s,xavg,yavg;
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,xavg,yavg);
			return yavg;
		} else if (derived_param_type == Chi_Square) {
			double chisq_out;
			if (lens_in->raw_chisq==-1e30) {
				if (lens_in->lens_parent != NULL) {
					// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
					lens_in->lens_parent->LogLikeFunc(NULL); // If the chi-square has not already been evaluated, evaluate it here
					chisq_out = lens_in->raw_chisq;
				} else {
					chisq_out = lens_in->chisq_single_evaluation(true,false,false,false);
				}
			} else chisq_out = -1e30;
			return chisq_out;
		}
		else die("no user defined function yet");
		return 0.0;
	}
	void print_param_description(QLens* lens_in)
	{
		string unitstring = (use_kpc_units) ? " kpc" : " arcsec";
		double dpar = get_derived_param(lens_in);
		//cout << name << ": ";
		if (derived_param_type == KappaR) {
			if (int_param==-1) std::cout << "Total kappa within r = " << funcparam << unitstring << std::endl;
			else std::cout << "kappa for lens " << int_param << " within r = " << funcparam << unitstring << std::endl;
		} else if (derived_param_type == LambdaR) {
			std::cout << "One minus average kappa at r = " << funcparam << unitstring << std::endl;
		} else if (derived_param_type == DKappaR) {
			if (int_param==-1) std::cout << "Derivative of total kappa within r = " << funcparam << unitstring << std::endl;
			else std::cout << "Derivative of kappa for lens " << int_param << " within r = " << funcparam << unitstring << std::endl;
		} else if (derived_param_type == Mass2dR) {
			std::cout << "Projected (2D) mass of lens " << int_param << " enclosed within r = " << funcparam << unitstring << std::endl;
		} else if (derived_param_type == Mass3dR) {
			std::cout << "Deprojected (3D) mass of lens " << int_param << " enclosed within r = " << funcparam << unitstring << std::endl;
		} else if (derived_param_type == Einstein) {
			std::cout << "Einstein radius of lens " << int_param << " for source redshift zsrc = " << funcparam << std::endl;
		} else if (derived_param_type == Einstein_Mass) {
			std::cout << "Projected mass within Einstein radius of lens " << int_param << " for source redshift zsrc = " << funcparam << std::endl;
		} else if (derived_param_type == Xi_Param) {
			std::cout << "Xi parameter of lens " << int_param << std::endl;
		} else if (derived_param_type == Kappa_Re) {
			std::cout << "Kappa at Einstein radius of primary lens (plus other lenses that are co-centered with primary), averaged over all angles" << std::endl;
		} else if (derived_param_type == LensParam) {
			std::cout << "Parameter " << ((int) funcparam) << " of lens " << int_param << " using pmode=" << ((int) funcparam2) << std::endl;
		} else if (derived_param_type == AvgLogSlope) {
			std::cout << "Average log-slope of kappa from lens " << int_param << " between r1=" << funcparam << unitstring << " and r2=" << funcparam2 << unitstring << std::endl;
		} else if (derived_param_type == Perturbation_Radius) {
			std::cout << "Critical curve perturbation radius of lens " << int_param << std::endl;
		} else if (derived_param_type == Relative_Perturbation_Radius) {
			std::cout << "Relative critical curve perturbation radius of lens " << int_param << std::endl;
		} else if (derived_param_type == Robust_Perturbation_Mass) {
			std::cout << "Projected mass within perturbation radius of lens " << int_param << std::endl;
		} else if (derived_param_type == Robust_Perturbation_Density) {
			std::cout << "Average projected density within perturbation radius of lens " << int_param << std::endl;
		} else if (derived_param_type == Adaptive_Grid_qs) {
			std::cout << "Axis ratio derived from source pixel covariance matrix using a " << int_param << "x" << int_param << " sampling" << std::endl;
		} else if (derived_param_type == Adaptive_Grid_phi_s) {
			std::cout << "Orientation angle derived of source pixel covariance matrix using a " << int_param << "x" << int_param << " sampling" << std::endl;
		} else if (derived_param_type == Adaptive_Grid_xavg) {
			std::cout << "Centroid x-coordinate of pixellated source using a " << int_param << "x" << int_param << " sampling" << std::endl;
		} else if (derived_param_type == Adaptive_Grid_yavg) {
			std::cout << "Centroid y-coordinate of pixellated source using a " << int_param << "x" << int_param << " sampling" << std::endl;
		} else if (derived_param_type == Chi_Square) {
			std::cout << "Raw chi-square value for given set of parameters" << std::endl;
		} else die("no user defined function yet");
		std::cout << "   name: '" << name << "', latex_name: '" << latex_name << "'" << std::endl;
		std::cout << "   " << name << " = " << dpar << std::endl;
	}
	void rename(const string new_name, const string new_latex_name)
	{
		name = new_name;
		latex_name = new_latex_name;
	}
};

struct ParamSettings
{
	int nparams;
	ParamPrior **priors;
	ParamTransform **transforms;
	string *param_names;
	string *override_names; // this allows to manually set names even after parameter transformations
	// ParamSettings should handle the latex names too, to simplify things; this would also allow for manual override of the latex names. Implement this!!!!!!
	double *prior_norms;
	double *penalty_limits_lo, *penalty_limits_hi;
	bool *use_penalty_limits;
	// It would be nice if penalty limits and override_limits could be merged. The tricky part is that the penalty limits deal with the	
	// untransformed parameters, while override_limits deal with the transformed parameters. Not sure yet what is the best way to handle this.
	double *override_limits_lo, *override_limits_hi;
	bool *override_prior_limits;
	double *stepsizes;
	bool *auto_stepsize;
	bool *hist2d_param;
	bool *hist2d_dparam;
	bool *subplot_param;
	bool *subplot_dparam;
	string *dparam_names;
	int n_dparams;
	ParamSettings() { priors = NULL; param_names = NULL; transforms = NULL; nparams = 0; stepsizes = NULL; auto_stepsize = NULL; hist2d_param = NULL; hist2d_dparam = NULL; subplot_param = NULL; dparam_names = NULL; subplot_dparam = NULL; nparams = 0; n_dparams = 0; }
	ParamSettings(ParamSettings& param_settings_in) {
		nparams = param_settings_in.nparams;
		n_dparams = param_settings_in.n_dparams;
		param_names = new string[nparams];
		override_names = new string[nparams];
		priors = new ParamPrior*[nparams];
		transforms = new ParamTransform*[nparams];
		stepsizes = new double[nparams];
		auto_stepsize = new bool[nparams];
		hist2d_param = new bool[nparams];
		subplot_param = new bool[nparams];
		prior_norms = new double[nparams];
		penalty_limits_lo = new double[nparams];
		penalty_limits_hi = new double[nparams];
		use_penalty_limits = new bool[nparams];
		override_limits_lo = new double[nparams];
		override_limits_hi = new double[nparams];
		override_prior_limits = new bool[nparams];
		for (int i=0; i < nparams; i++) {
			priors[i] = new ParamPrior(param_settings_in.priors[i]);
			transforms[i] = new ParamTransform(param_settings_in.transforms[i]);
			param_names[i] = param_settings_in.param_names[i];
			override_names[i] = param_settings_in.override_names[i];
			stepsizes[i] = param_settings_in.stepsizes[i];
			auto_stepsize[i] = param_settings_in.auto_stepsize[i];
			hist2d_param[i] = param_settings_in.hist2d_param[i];
			subplot_param[i] = param_settings_in.subplot_param[i];
			prior_norms[i] = param_settings_in.prior_norms[i];
			penalty_limits_lo[i] = param_settings_in.penalty_limits_lo[i];
			penalty_limits_hi[i] = param_settings_in.penalty_limits_hi[i];
			use_penalty_limits[i] = param_settings_in.use_penalty_limits[i];
			override_limits_lo[i] = param_settings_in.override_limits_lo[i];
			override_limits_hi[i] = param_settings_in.override_limits_hi[i];
			override_prior_limits[i] = param_settings_in.override_prior_limits[i];
		}
		if (n_dparams > 0) {
			dparam_names = new string[n_dparams];
			hist2d_dparam = new bool[n_dparams];
			subplot_dparam = new bool[n_dparams];
			for (int i=0; i < n_dparams; i++) {
				dparam_names[i] = param_settings_in.dparam_names[i];
				hist2d_dparam[i] = param_settings_in.hist2d_dparam[i];
				subplot_dparam[i] = param_settings_in.subplot_dparam[i];
			}
		}
	}
	void update_params(const int nparams_in, std::vector<string>& names, double* stepsizes_in);
	void insert_params(const int pi, const int pf, std::vector<string>& names, double* stepsizes_in);
	bool remove_params(const int pi, const int pf);
	void add_dparam(string dparam_name);
	void remove_dparam(int dparam_number);
	void rename_dparam(int dparam_number, string newname) { dparam_names[dparam_number] = newname; }
	void clear_dparams()
	{
		if (n_dparams > 0) {
			delete[] dparam_names;
			delete[] hist2d_dparam;
			delete[] subplot_dparam;
			n_dparams = 0;
		}
	}
	int lookup_param_number(const string pname)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		int pnum = -1;
		for (int i=0; i < nparams; i++) {
			if ((transformed_names[i]==pname) or (param_names[i]==pname)) { pnum = i; break; }
		}
		for (int i=0; i < n_dparams; i++) {
			if (dparam_names[i]==pname) pnum = nparams+i;
		}
		delete[] transformed_names;
		return pnum;
	}
	string lookup_param_name(const int i)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		string name = transformed_names[i];
		delete[] transformed_names;
		return name;
	}
	bool exclude_hist2d_param(const string pname)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		bool found_name = false;
		int i;
		for (i=0; i < nparams; i++) {
			if ((param_names[i]==pname) or (transformed_names[i]==pname)) {
				hist2d_param[i] = false;
				found_name = true;
				break;
			}
		}
		if (!found_name) {
			for (i=0; i < n_dparams; i++) {
				if (dparam_names[i]==pname) {
					hist2d_dparam[i] = false;
					found_name = true;
					break;
				}
			}
		}
		delete[] transformed_names;
		return found_name;
	}
	bool hist2d_params_defined()
	{
		bool active_param = false;
		int i;
		for (i=0; i < nparams; i++) {
			if (!hist2d_param[i]) {
				active_param = true;
				break;
			}
		}
		if (!active_param) {
			for (i=0; i < n_dparams; i++) {
				if (!hist2d_dparam[i]) {
					active_param = true;
					break;
				}
			}
		}
		return active_param;
	}
	bool hist2d_param_flag(const int i, string &name)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		bool flag;
		if (i < nparams) {
			name = transformed_names[i];
			flag = hist2d_param[i];
		} else {
			int j = i - nparams;
			name = dparam_names[j];
			flag = hist2d_dparam[j];
		}
		delete[] transformed_names;
		return flag;
	}
	string print_excluded_hist2d_params()
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		string pstring = "";
		int i;
		for (i=0; i < nparams; i++) {
			if (!hist2d_param[i]) pstring += transformed_names[i] + " ";
		}
		for (i=0; i < n_dparams; i++) {
			if (!hist2d_dparam[i]) pstring += dparam_names[i] + " ";
		}
		delete[] transformed_names;
		return pstring;
	}
	void reset_hist2d_params()
	{
		int i;
		for (i=0; i < nparams; i++) hist2d_param[i] = true;
		for (i=0; i < n_dparams; i++) hist2d_dparam[i] = true;
	}
	bool set_subplot_param(const string pname)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		bool found_name = false;
		int i;
		for (i=0; i < nparams; i++) {
			if ((param_names[i]==pname) or (transformed_names[i]==pname)) {
				subplot_param[i] = true;
				found_name = true;
				break;
			}
		}
		if (!found_name) {
			for (i=0; i < n_dparams; i++) {
				if (dparam_names[i]==pname) {
					subplot_dparam[i] = true;
					found_name = true;
					break;
				}
			}
		}
		delete[] transformed_names;
		return found_name;
	}
	bool subplot_params_defined()
	{
		bool active_param = false;
		int i;
		for (i=0; i < nparams; i++) {
			if (subplot_param[i]) {
				active_param = true;
				break;
			}
		}
		if (!active_param) {
			for (i=0; i < n_dparams; i++) {
				if (subplot_dparam[i]) {
					active_param = true;
					break;
				}
			}
		}
		return active_param;
	}
	bool subplot_param_flag(const int i, string &name)
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		bool flag;
		if (i < nparams) {
			name = transformed_names[i];
			flag = subplot_param[i];
		} else {
			int j = i - nparams;
			name = dparam_names[j];
			flag = subplot_dparam[j];
		}
		delete[] transformed_names;
		return flag;
	}
	string print_subplot_params()
	{
		string *transformed_names = new string[nparams];
		transform_parameter_names(param_names,transformed_names,NULL,NULL);
		string pstring = "";
		int i;
		for (i=0; i < nparams; i++) {
			if (subplot_param[i]) pstring += transformed_names[i] + " ";
		}
		for (i=0; i < n_dparams; i++) {
			if (subplot_dparam[i]) pstring += dparam_names[i] + " ";
		}
		delete[] transformed_names;
		return pstring;
	}
	void reset_subplot_params()
	{
		int i;
		for (i=0; i < nparams; i++) subplot_param[i] = false;
		for (i=0; i < n_dparams; i++) subplot_dparam[i] = false;
	}
	void clear_penalty_limits()
	{
		for (int i=0; i < nparams; i++) {
			use_penalty_limits[i] = false;
		}
	}
	void print_priors();
	bool output_prior(const int i);
	void print_stepsizes();
	void print_penalty_limits();
	void scale_stepsizes(const double fac)
	{
		for (int i=0; i < nparams; i++) {
			stepsizes[i] *= fac;
			auto_stepsize[i] = false;
		}
	}
	void reset_stepsizes(double *stepsizes_in)
	{
		for (int i=0; i < nparams; i++) {
			stepsizes[i] = stepsizes_in[i];
			auto_stepsize[i] = true;
		}
		transform_stepsizes();
	}
	void reset_stepsizes_no_transform(double *stepsizes_in)
	{
		for (int i=0; i < nparams; i++) {
			stepsizes[i] = stepsizes_in[i];
			auto_stepsize[i] = false;
		}
	}
	void set_stepsize(const int i, const double step)
	{
		if (i >= nparams) die("parameter chosen for stepsize is greater than total number of parameters (%i vs %i)",i,nparams);
		auto_stepsize[i] = false;
		stepsizes[i] = step;
	}
	void set_penalty_limit(const int i, const double lo, const double hi)
	{
		if (i >= nparams) die("parameter chosen for penalty limit is greater than total number of parameters (%i vs %i)",i,nparams);
		use_penalty_limits[i] = true;
		penalty_limits_lo[i] = lo;
		penalty_limits_hi[i] = hi;
	}
	void get_penalty_limits(boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		use_plimits.input(nparams);
		lower.input(nparams);
		upper.input(nparams);
		for (int i=0; i < nparams; i++) {
			use_plimits[i] = use_penalty_limits[i];
			lower[i] = penalty_limits_lo[i];
			upper[i] = penalty_limits_hi[i];
		}
	}
	void update_penalty_limits(boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		for (int i=0; i < nparams; i++) {
			use_penalty_limits[i] = use_plimits[i];
			penalty_limits_lo[i] = lower[i];
			penalty_limits_hi[i] = upper[i];
		}
	}
	void update_specific_penalty_limits(const int pi, const int pf, boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		int i, index;
		for (i=0, index=pi; index < pf; i++, index++) {
			use_penalty_limits[index] = use_plimits[i];
			penalty_limits_lo[index] = lower[i];
			penalty_limits_hi[index] = upper[i];
		}
	}
	void clear_penalty_limit(const int i)
	{
		if (i >= nparams) die("parameter chosen for penalty limit is greater than total number of parameters (%i vs %i)",i,nparams);
		use_penalty_limits[i] = true; // this ensures that it won't be overwritten by default values
		penalty_limits_lo[i] = -1e30;
		penalty_limits_hi[i] = 1e30;
	}
	void transform_parameters(double *params)
	{
		double *new_params = new double[nparams];
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->transform==NONE) new_params[i] = params[i];
			else if (transforms[i]->transform==LOG_TRANSFORM) new_params[i] = log(params[i])/M_LN10;
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				new_params[i] = erff((params[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				new_params[i] = transforms[i]->a * params[i] + transforms[i]->b;
			} else if (transforms[i]->transform==RATIO) {
				new_params[i] = params[i]/params[transforms[i]->ratio_paramnum];
			}
		}
		for (int i=0; i < nparams; i++) {
			params[i] = new_params[i];
		}
		delete[] new_params;
	}
	void transform_limits(double *lower, double *upper)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->transform==LOG_TRANSFORM) lower[i] = log(lower[i])/M_LN10;
			if (transforms[i]->transform==LOG_TRANSFORM) upper[i] = log(upper[i])/M_LN10;
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				lower[i] = erff((lower[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
				upper[i] = erff((upper[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				lower[i] = transforms[i]->a * lower[i] + transforms[i]->b;
				upper[i] = transforms[i]->a * upper[i] + transforms[i]->b;
				if (lower[i] > upper[i]) {
					double temp = lower[i]; lower[i] = upper[i]; upper[i] = temp;
				}
			} else if (transforms[i]->transform==RATIO) {
				lower[i] = 0; // these can be manually adjusted using 'fit priors range ...'
				upper[i] = 1; // these can be customized
			}
		}
	}
	void set_override_prior_limit(const int i, const double lo, const double hi)
	{
		if (i >= nparams) die("parameter chosen for prior limit is greater than total number of parameters (%i vs %i)",i,nparams);
		override_prior_limits[i] = true;
		override_limits_lo[i] = lo;
		override_limits_hi[i] = hi;
	}
	void override_limits(double *lower, double *upper)
	{
		for (int i=0; i < nparams; i++) {
			if (override_prior_limits[i]) {
				lower[i] = override_limits_lo[i];
				upper[i] = override_limits_hi[i];
			}
		}
	}
	void inverse_transform_parameters(double *params, double *transformed_params)
	{
		bool apply_ratio_transform_afterwards = false;
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->transform==NONE) transformed_params[i] = params[i];
			else if (transforms[i]->transform==LOG_TRANSFORM) transformed_params[i] = pow(10.0,params[i]);
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				transformed_params[i] = transforms[i]->gaussian_pos + M_SQRT2*transforms[i]->gaussian_sig*erfinv(params[i]);
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				transformed_params[i] = (params[i] - transforms[i]->b) / transforms[i]->a;
			} else if (transforms[i]->transform==RATIO) {
				if (transforms[i]->ratio_paramnum < i) {
					transformed_params[i] = params[i]*transformed_params[transforms[i]->ratio_paramnum];
				} else apply_ratio_transform_afterwards = true;
			}
		}
		if (apply_ratio_transform_afterwards) {
			for (int i=0; i < nparams; i++) {
				if (transforms[i]->transform==RATIO) transformed_params[i] = params[i]*transformed_params[transforms[i]->ratio_paramnum];
			}
		}
	}
	void inverse_transform_parameters(double *params)
	{
		inverse_transform_parameters(params,params);
	}
	void transform_parameter_names(string *names, string *transformed_names, string *latex_names, string *transformed_latex_names)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->transform==NONE) {
				transformed_names[i] = names[i];
				if (latex_names != NULL) transformed_latex_names[i] = latex_names[i];
			}
			else if (transforms[i]->transform==LOG_TRANSFORM) {
				transformed_names[i] = "log(" + names[i] + ")";
				if (latex_names != NULL) transformed_latex_names[i] = "\\log(" + latex_names[i] + ")";
			}
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				transformed_names[i] = "u{" + names[i] + "}";
				if (latex_names != NULL) transformed_latex_names[i] = "u\\{" + latex_names[i] + "\\}";
			}
			else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				transformed_names[i] = "L{" + names[i] + "}";
				if (latex_names != NULL) transformed_latex_names[i] = "L\\{" + latex_names[i] + "\\}";
			}
			else if (transforms[i]->transform==RATIO) {
				transformed_names[i] = names[i] + "_over_" + names[transforms[i]->ratio_paramnum];
				if (latex_names != NULL) transformed_latex_names[i] = latex_names[i] + "/" + latex_names[transforms[i]->ratio_paramnum];
			}
		}
		override_parameter_names(transformed_names); // allows for manually setting parameter names
	}
	bool set_override_parameter_name(const int i, const string name)
	{
		bool unique_name = true;
		for (int j=0; j < nparams; j++) {
			if ((i != j) and (((override_names[j] != "") and (override_names[j]==name)) or (param_names[j]==name))) unique_name = false;
		}
		if (!unique_name) return false;
		override_names[i] = name;
		return true;
	}
	void override_parameter_names(string* names)
	{
		for (int i=0; i < nparams; i++) {
			if (override_names[i] != "") names[i] = override_names[i];
		}
	}
	void transform_stepsizes()
	{
		// It would be better to have it pass in the current value of the parameters, then use the default
		// (untransformed) stepsize to define the transformed stepsize. For example, the log stepsize would
		// be log((pi+step)/pi). But passing in the parameter values is a bit of a pain...do this later
		for (int i=0; i < nparams; i++) {
			if (auto_stepsize[i]) {
				if (transforms[i]->transform==LOG_TRANSFORM) {
					stepsizes[i] = 0.5; // default for a log transform
				}
				else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				}
				else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				}
				else if (transforms[i]->transform==RATIO) {
					stepsizes[i] = 0.5; // default for a ratio
				}
			}
		}
	}
	void add_prior_terms_to_loglike(double *params, double& loglike)
	{
		//std::cout << "LOGLIKE00=" << (2*loglike) << std::endl;
		double dloglike,dloglike_tot=0;
		for (int i=0; i < nparams; i++) {
			if (priors[i]->prior!=UNIFORM_PRIOR) {
				//std::cout << "PRIOR NORM (param " << i << "): " << (2*log(prior_norms[i])) << std::endl;
				dloglike_tot += log(prior_norms[i]); // Normalize the prior for the bayesian evidence
				if (priors[i]->prior==LOG_PRIOR) {
					dloglike = log(params[i]);
					dloglike_tot += dloglike;
				}
				else if (priors[i]->prior==GAUSS_PRIOR) {
					dloglike = SQR((params[i] - priors[i]->gaussian_pos)/priors[i]->gaussian_sig)/2.0;
					//std::cout << "YO: " << params[i] << " " << priors[i]->gaussian_pos << " " << priors[i]->gaussian_sig << " " << dloglike << std::endl;
					dloglike_tot += dloglike;
				}
				else if (priors[i]->prior==GAUSS2_PRIOR) {
					int j = priors[i]->gauss_paramnums[1];
					dvector bvec, cvec;
					bvec.input(2);
					cvec.input(2);
					bvec[0] = params[i] - priors[i]->gauss_meanvals[0];
					bvec[1] = params[j] - priors[i]->gauss_meanvals[1];
					cvec = priors[i]->inv_covariance_matrix * bvec;
					dloglike = (bvec[0]*cvec[0] + bvec[1]*cvec[1]) / 2.0;
					dloglike_tot += dloglike;
				}
			}
		}
		//std::cout << "DLOGLIKE_TOT*2: " << (2*dloglike_tot) << " LOGLIKE0: " << (2*loglike) << std::endl;
		loglike += dloglike_tot;
		//std::cout << "NEW LOGLIKE: " << (2*loglike) << std::endl;
	}
	void update_reference_paramnums(int *new_paramnums)
	{
		// This updates any parameter numbers that are referenced by the priors or transforms; this is done any time the parameter list is changed
		int new_paramnum;
		for (int i=0; i < nparams; i++) {
			if (priors[i]->prior==GAUSS2_PRIOR) {
				new_paramnum = new_paramnums[priors[i]->gauss_paramnums[0]];
				if (new_paramnum==-1) {
					// parameter no longer exists; revert back to uniform prior
					priors[i]->set_uniform();
				} else {
					priors[i]->gauss_paramnums[0] = new_paramnum;
					priors[i]->gauss_paramnums[1] = new_paramnum;
				}
			}
			if (transforms[i]->transform==RATIO) {
				new_paramnum = new_paramnums[transforms[i]->ratio_paramnum];
				if (new_paramnum==-1) {
					// parameter no longer exists; remove transformation
					transforms[i]->set_none();
				} else {
					transforms[i]->ratio_paramnum = new_paramnum;
				}
			}
		}
	}
	void set_prior_norms(double *lower_limit, double* upper_limit)
	{
		// flat priors are automatically given a norm of 1.0, since we'll be transforming to the unit hypercube when doing nested sampling;
		// however a correction is required for other priors
		for (int i=0; i < nparams; i++) {
			if (priors[i]->prior!=UNIFORM_PRIOR) {
				if (priors[i]->prior==LOG_PRIOR) prior_norms[i] = log(upper_limit[i]/lower_limit[i]);
				else if (priors[i]->prior==GAUSS_PRIOR) {
					prior_norms[i] = (erff((upper_limit[i] - priors[i]->gaussian_pos)/(M_SQRT2*priors[i]->gaussian_sig)) - erff((lower_limit[i] - priors[i]->gaussian_pos)/(M_SQRT2*priors[i]->gaussian_sig))) * M_SQRT_HALFPI * priors[i]->gaussian_sig;
				}
				prior_norms[i] /= (upper_limit[i] - lower_limit[i]); // correction since we are transforming to the unit hypercube
			}
		}
	}
	void add_jacobian_terms_to_loglike(double *params, double& loglike)
	{
		for (int i=0; i < nparams; i++) {
			if (transforms[i]->include_jacobian==true) {
				if (transforms[i]->transform==LOG_TRANSFORM) loglike -= log(params[i]);
				else if (transforms[i]->transform==GAUSS_TRANSFORM) loglike -= SQR((params[i] - transforms[i]->gaussian_pos)/transforms[i]->gaussian_sig)/2.0;
				else if (transforms[i]->transform==RATIO) loglike += log(params[transforms[i]->ratio_paramnum]);
			}
		}
	}
	void clear_params()
	{
		if (nparams > 0) {
			delete[] param_names;
			delete[] override_names;
			for (int i=0; i < nparams; i++) {
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

		}
		priors = NULL;
		param_names = NULL;
		override_names = NULL;
		transforms = NULL;
		nparams = 0;
		stepsizes = NULL;
		auto_stepsize = NULL;
		subplot_param = NULL;
	}
	~ParamSettings()
	{
		if (nparams > 0) {
			delete[] param_names;
			delete[] override_names;
			for (int i=0; i < nparams; i++) {
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
		}
		if (n_dparams > 0) {
			delete[] dparam_names;
			delete[] subplot_dparam;
		}
	}
};

#endif // PARAMS_H

