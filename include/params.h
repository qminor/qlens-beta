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
	void set_gaussian(const double &pos_in, const double &sig_in) { prior = GAUSS_PRIOR; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void set_gauss2(int p1, int p2, const double &pos1_in, const double &pos2_in, const double &sig1_in, const double &sig2_in, const double &sig12_in) {
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
	void set_linear(const double a_in, const double b_in) { transform = LINEAR_TRANSFORM; a = a_in; b = b_in; }
	void set_gaussian(const double pos_in, const double sig_in) { transform = GAUSS_TRANSFORM; gaussian_pos = pos_in; gaussian_sig = sig_in; }
	void set_ratio(const int paramnum_in) { transform = RATIO; ratio_paramnum = paramnum_in; }
	void set_include_jacobian(const bool include) { include_jacobian = include; }
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
		} else if (derived_param_type == DlogKappaR) {
			name = "dlogkappa"; latex_name = "\\ln\\kappa'"; if (lensnum==-1) { name += "_tot"; latex_name += "_{tot}"; }
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
		} else if (derived_param_type == CC_Xi_Param) {
			name = "cc_xi"; latex_name = "\\xi_{cc}";
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
		} else if (derived_param_type == Adaptive_Grid_sig_s) {
			name = "sig_s"; latex_name = "\\sigma_{s}";
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
	DerivedParam(DerivedParam* dparam_in)
	{
		derived_param_type = dparam_in->derived_param_type;
		funcparam = dparam_in->funcparam;
		funcparam2 = dparam_in->funcparam2;
		int_param = dparam_in->int_param;
		use_kpc_units = dparam_in->use_kpc_units;
		name = dparam_in->name;
		latex_name = dparam_in->latex_name;
	}
	double get_derived_param(QLens* lens_in)
	{
		if (derived_param_type == KappaR) return lens_in->total_kappa(funcparam,int_param,use_kpc_units);
		else if (derived_param_type == LambdaR) return (1 - lens_in->total_kappa(funcparam,-1,use_kpc_units));
		else if (derived_param_type == DlogKappaR) return lens_in->total_dlogkappa(funcparam,int_param,use_kpc_units);
		else if (derived_param_type == Mass2dR) return lens_in->mass2d_r(funcparam,int_param,use_kpc_units);
		else if (derived_param_type == Mass3dR) return lens_in->mass3d_r(funcparam,int_param,use_kpc_units);
		else if (derived_param_type == Einstein) return lens_in->einstein_radius_single_lens(funcparam,int_param);
		else if (derived_param_type == Xi_Param) {
			if (int_param >= 0) return lens_in->get_xi_parameter(funcparam,int_param);
			else return lens_in->get_total_xi_parameter(funcparam);
		}
		else if (derived_param_type == CC_Xi_Param) {
			return lens_in->cc_xi_parameter(funcparam);
		}
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
			double qs,phi_s,sig_s,xavg,yavg;
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,sig_s,xavg,yavg);
			return qs;
		} else if (derived_param_type == Adaptive_Grid_phi_s) {
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			if (lens_in->lens_parent != NULL) {
				// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
				if (lens_in->raw_chisq==-1e30) lens_in->lens_parent->LogLikeFunc(NULL); // If a source inversion hasn't been performed yet, do it here
			} else if (lens_in->raw_chisq==-1e30) {
				lens_in->invert_surface_brightness_map_from_data(lens_in->raw_chisq, false, true); // If a source inversion hasn't been performed yet, do it here
			}
			double qs,phi_s,phi_s_deg,sig_s,xavg,yavg;
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,sig_s,xavg,yavg);
			phi_s_deg = phi_s*180.0/M_PI;
			return phi_s_deg;
		} else if (derived_param_type == Adaptive_Grid_sig_s) {
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			if (lens_in->lens_parent != NULL) {
				// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
				if (lens_in->raw_chisq==-1e30) lens_in->lens_parent->LogLikeFunc(NULL); // If a source inversion hasn't been performed yet, do it here
			} else if (lens_in->raw_chisq==-1e30) {
				lens_in->invert_surface_brightness_map_from_data(lens_in->raw_chisq, false, true); // If a source inversion hasn't been performed yet, do it here
			}
			double qs,phi_s,sig_s,xavg,yavg;
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,sig_s,xavg,yavg);
			return sig_s;
		} else if (derived_param_type == Adaptive_Grid_xavg) {
			if (lens_in->lens_parent != NULL) {
				// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
				if (lens_in->raw_chisq==-1e30) lens_in->lens_parent->LogLikeFunc(NULL); // If a source inversion hasn't been performed yet, do it here
			} else if (lens_in->raw_chisq==-1e30) {
				lens_in->invert_surface_brightness_map_from_data(lens_in->raw_chisq, false, true); // If a source inversion hasn't been performed yet, do it here
			}
			double qs,phi_s,sig_s,xavg,yavg;
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,sig_s,xavg,yavg);
			return xavg;
		} else if (derived_param_type == Adaptive_Grid_yavg) {
			if (lens_in->lens_parent != NULL) {
				// this means we're running it from the "fitmodel" QLens object, so the likelihood needs to be run from the parent QLens object
				if (lens_in->raw_chisq==-1e30) lens_in->lens_parent->LogLikeFunc(NULL); // If a source inversion hasn't been performed yet, do it here
			} else if (lens_in->raw_chisq==-1e30) {
				lens_in->invert_surface_brightness_map_from_data(lens_in->raw_chisq, false, true); // If a source inversion hasn't been performed yet, do it here
			}
			double qs,phi_s,sig_s,xavg,yavg;
			// Here, int_param is the number of pixels per side being sampled (so if funcparam=200, it's a 200x200 grid being sampled)
			lens_in->find_pixellated_source_moments(int_param,qs,phi_s,sig_s,xavg,yavg);
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
		} else if (derived_param_type == DlogKappaR) {
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
			if (int_param >= 0) {
				std::cout << "xi parameter of lens " << int_param << std::endl;
			} else {
				std::cout << "xi parameter of primary and co-centered lenses " << std::endl;
			}
		} else if (derived_param_type == CC_Xi_Param) {
			std::cout << "xi parameter of lens along critical curve " << std::endl;
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
		} else if (derived_param_type == Adaptive_Grid_sig_s) {
			std::cout << "Dispersion of source using a " << int_param << "x" << int_param << " sampling" << std::endl;
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
		if (latex_name != "") latex_name = new_latex_name;
	}
};

struct ParamList
{
	static constexpr double VERY_LARGE = 1e30;
	QLens* qlens;

	int nparams;
	ParamPrior **priors;
	ParamTransform **transforms;
	double *values, *untransformed_values;
	string *untransformed_param_names, *param_names;
	string *override_names; // this allows to manually set names even after parameter transformations
	string *untransformed_latex_names, *latex_names;
	double *prior_norms;
	double *untransformed_prior_limits_lo, *untransformed_prior_limits_hi;
	bool *defined_prior_limits;
	double *prior_limits_lo, *prior_limits_hi;
	double *stepsizes;
	bool *auto_stepsize;
	bool *hist2d_param;
	bool *subplot_param;

	ParamList() {
		set_null_ptrs_and_values();
	}
	ParamList(QLens* qlens_in) : ParamList() {
		qlens = qlens_in;
	}
	ParamList(ParamList& param_list_in) {
		nparams = param_list_in.nparams;
		untransformed_param_names = new string[nparams];
		param_names = new string[nparams];
		untransformed_latex_names = new string[nparams];
		latex_names = new string[nparams];
		override_names = new string[nparams];
		untransformed_values = new double[nparams];
		values = new double[nparams];
		priors = new ParamPrior*[nparams];
		transforms = new ParamTransform*[nparams];
		stepsizes = new double[nparams];
		auto_stepsize = new bool[nparams];
		hist2d_param = new bool[nparams];
		subplot_param = new bool[nparams];
		prior_norms = new double[nparams];
		untransformed_prior_limits_lo = new double[nparams];
		untransformed_prior_limits_hi = new double[nparams];
		defined_prior_limits = new bool[nparams];
		prior_limits_lo = new double[nparams];
		prior_limits_hi = new double[nparams];
		for (int i=0; i < nparams; i++) {
			priors[i] = new ParamPrior(param_list_in.priors[i]);
			transforms[i] = new ParamTransform(param_list_in.transforms[i]);
			untransformed_param_names[i] = param_list_in.untransformed_param_names[i];
			param_names[i] = param_list_in.param_names[i];
			untransformed_latex_names[i] = param_list_in.untransformed_latex_names[i];
			latex_names[i] = param_list_in.latex_names[i];
			override_names[i] = param_list_in.override_names[i];
			untransformed_values[i] = param_list_in.untransformed_values[i];
			values[i] = param_list_in.values[i];
			stepsizes[i] = param_list_in.stepsizes[i];
			auto_stepsize[i] = param_list_in.auto_stepsize[i];
			hist2d_param[i] = param_list_in.hist2d_param[i];
			subplot_param[i] = param_list_in.subplot_param[i];
			prior_norms[i] = param_list_in.prior_norms[i];
			untransformed_prior_limits_lo[i] = param_list_in.untransformed_prior_limits_lo[i];
			untransformed_prior_limits_hi[i] = param_list_in.untransformed_prior_limits_hi[i];
			defined_prior_limits[i] = param_list_in.defined_prior_limits[i];
			prior_limits_lo[i] = param_list_in.prior_limits_lo[i];
			prior_limits_hi[i] = param_list_in.prior_limits_hi[i];
		}
	}
	ParamList(ParamList& param_list_in, QLens* qlens_in) : ParamList(param_list_in) {
		qlens = qlens_in;
	}
	void update_param_list(string* param_names_in, string* latex_names_in, double* stepsizes_in, const bool check_current_params = false);
	void insert_params(const int pi, const int pf, string* param_names_in, string* latex_names_in, double* untransformed_values_in, double* stepsizes_in);
	bool remove_params(const int pi, const int pf);
	bool print_priors_and_limits();
	bool print_parameter_values();
	string mkstring_doub(const double db);
	string mkstring_int(const int i);
	string get_param_values_string();

	int lookup_param_number(const string pname)
	{
		int pnum = -1;
		for (int i=0; i < nparams; i++) {
			if ((param_names[i]==pname) or (untransformed_param_names[i]==pname)) { pnum = i; break; }
		}
		return pnum;
	}
	string lookup_param_name(const int i)
	{
		string name = param_names[i];
		return name;
	}
	bool exclude_hist2d_param(const string pname)
	{
		bool found_name = false;
		int i;
		for (i=0; i < nparams; i++) {
			if ((untransformed_param_names[i]==pname) or (param_names[i]==pname)) {
				hist2d_param[i] = false;
				found_name = true;
				break;
			}
		}
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
		return active_param;
	}
	bool hist2d_param_flag(const int i, string &name)
	{
		bool flag;
		if (i < nparams) {
			name = param_names[i];
			flag = hist2d_param[i];
		}
		return flag;
	}
	string print_excluded_hist2d_params()
	{
		string pstring = "";
		int i;
		for (i=0; i < nparams; i++) {
			if (!hist2d_param[i]) pstring += param_names[i] + " ";
		}
		return pstring;
	}
	void reset_hist2d_params()
	{
		int i;
		for (i=0; i < nparams; i++) hist2d_param[i] = true;
	}
	bool set_subplot_param(const string pname)
	{
		bool found_name = false;
		int i;
		for (i=0; i < nparams; i++) {
			if ((untransformed_param_names[i]==pname) or (param_names[i]==pname)) {
				subplot_param[i] = true;
				found_name = true;
				break;
			}
		}
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
		return active_param;
	}
	bool subplot_param_flag(const int i, string &name)
	{
		bool flag;
		if (i < nparams) {
			name = param_names[i];
			flag = subplot_param[i];
		}
		return flag;
	}
	string print_subplot_params()
	{
		string pstring = "";
		for (int i=0; i < nparams; i++) {
			if (subplot_param[i]) pstring += param_names[i] + " ";
		}
		return pstring;
	}
	void reset_subplot_params()
	{
		for (int i=0; i < nparams; i++) subplot_param[i] = false;
	}
	void clear_prior_limits()
	{
		for (int i=0; i < nparams; i++) {
			defined_prior_limits[i] = false;
			untransformed_prior_limits_lo[i] = -VERY_LARGE;
			untransformed_prior_limits_hi[i] = VERY_LARGE;
			prior_limits_lo[i] = -VERY_LARGE;
			prior_limits_hi[i] = VERY_LARGE;
			prior_norms[i] = 1.0;
		}
	}
	bool all_prior_limits_defined() {
		bool all_defined = true;
		for (int i=0; i < nparams; i++) {
			if (defined_prior_limits[i]==false) {
				all_defined = false;
			}
		}
		return all_defined;
	}
	void print_priors_and_transforms();
	bool output_prior(const int i);
	void print_stepsizes();
	void print_untransformed_prior_limits();
	void scale_stepsizes(const double fac)
	{
		for (int i=0; i < nparams; i++) {
			stepsizes[i] *= fac;
			auto_stepsize[i] = false;
		}
	}
	void scale_stepsize(const int paramnum, const double fac)
	{
		if ((paramnum < 0) or (paramnum >= nparams)) { warn("invalid ratio_paramnum"); return; }
		stepsizes[paramnum] *= fac;
		auto_stepsize[paramnum] = false;
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
	void update_untransformed_values(const int pi, const int pf, double* values_in)
	{
		int i, index;
		for (i=0, index=pi; index < pf; i++, index++) {
			untransformed_values[index] = values_in[i];
		}
		transform_parameters();
	}
	void update_untransformed_values(double* values_in)
	{
		update_untransformed_values(0,nparams,values_in);
	}
	bool update_param_values(const int pi, const int pf, double* values_in)
	{
		int i, index;
		for (i=0, index=pi; index < pf; i++, index++) {
			values[index] = values_in[i];
		}
		inverse_transform_parameters(); // we inverse transform all the parameters in case any of the transforms depend on the parameters that have just been updated
		if (qlens) {
			if (qlens->update_model(untransformed_values) != 0.0) return false;
		}
		return true;
	}
	bool update_param_values(double* values_in)
	{
		return update_param_values(0,nparams,values_in);
	}
	bool update_param_value(const int i, const double value_in)
	{
		values[i] = value_in;
		inverse_transform_parameter(i); // NOTE: what if another parameter has a ratio transform that depends on parameter i? The safe thing would be to transform all parameters, but I just wonder if it's overkill. Just transforming parameter i for now
		if (qlens) {
			if (qlens->update_model(untransformed_values) != 0.0) return false;
		}
		return true;
	}
	bool update_param_value(const string name, const double value_in)
	{
		int paramnum = lookup_param_number(name);
		if ((paramnum < 0) or (paramnum >= nparams)) return false;
		return update_param_value(paramnum,value_in);
	}
	void get_untransformed_prior_limits(boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		use_plimits.input(nparams);
		lower.input(nparams);
		upper.input(nparams);
		for (int i=0; i < nparams; i++) {
			use_plimits[i] = defined_prior_limits[i];
			lower[i] = untransformed_prior_limits_lo[i];
			upper[i] = untransformed_prior_limits_hi[i];
		}
	}
	void get_prior_limits(boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		use_plimits.input(nparams);
		lower.input(nparams);
		upper.input(nparams);
		for (int i=0; i < nparams; i++) {
			use_plimits[i] = defined_prior_limits[i];
			lower[i] = prior_limits_lo[i];
			upper[i] = prior_limits_hi[i];
		}
	}
	bool set_prior_limit(const int paramnum, const double lo, const double hi)
	{
		if (paramnum >= nparams) die("parameter chosen for prior limit is greater than total number of parameters (%i vs %i)",paramnum,nparams);
		prior_limits_lo[paramnum] = lo;
		prior_limits_hi[paramnum] = hi;
		defined_prior_limits[paramnum] = true;
		inverse_transform_prior_limit(paramnum);
		set_prior_norm(paramnum);
		bool *changed_limit = new bool[nparams];
		for (int i=0; i < nparams; i++) changed_limit[i] = false;
		changed_limit[paramnum] = true;
		if (qlens) qlens->update_prior_limits(untransformed_prior_limits_lo,untransformed_prior_limits_hi,changed_limit);
		delete[] changed_limit;
		return true;
	}
	bool set_prior_limit(const string name, const double lo, const double hi)
	{
		int paramnum = lookup_param_number(name);
		if ((paramnum < 0) or (paramnum >= nparams)) return false;
		return set_prior_limit(paramnum,lo,hi);
	}
	void set_untransformed_prior_limits(const int pi, const int pf, dvector& lower, dvector& upper, const bool update_model = false)
	{
		int i,j;
		for (i=pi,j=0; i < pf; i++,j++) {
			untransformed_prior_limits_lo[i] = lower[j];
			untransformed_prior_limits_hi[i] = upper[j];
			defined_prior_limits[i] = true;
		}
		transform_prior_limits();
		set_prior_norms();
		if ((update_model) and (qlens)) {
			bool *changed_limit = new bool[nparams];
			for (j=0; j < nparams; j++) changed_limit[j] = false;
			for (j=pi; j < pf; j++) changed_limit[j] = true;
			qlens->update_prior_limits(untransformed_prior_limits_lo,untransformed_prior_limits_hi,changed_limit);
			delete[] changed_limit;
		}
	}
	void set_untransformed_prior_limit(const int i, const double lo, const double hi, const bool update_model = false)
	{
		if (i >= nparams) die("parameter chosen for penalty limit is greater than total number of parameters (%i vs %i)",i,nparams);
		defined_prior_limits[i] = true;
		untransformed_prior_limits_lo[i] = lo;
		untransformed_prior_limits_hi[i] = hi;
		defined_prior_limits[i] = true;
		transform_prior_limit(i);
		set_prior_norm(i);
		// by default, update_model=false because the prior limits within the model objects were changed before calling this function
		if ((update_model) and (qlens)) {
			bool *changed_limit = new bool[nparams];
			for (int j=0; j < nparams; j++) changed_limit[j] = false;
			changed_limit[i] = true;
			qlens->update_prior_limits(untransformed_prior_limits_lo,untransformed_prior_limits_hi,changed_limit);
			delete[] changed_limit;
		}
	}
	void update_untransformed_prior_limits_from_auto_ranges(const int pi, const int pf, boolvector& use_plimits, dvector& lower, dvector& upper)
	{
		int i,j;
		for (i=pi,j=0; i < pf; i++,j++) {
			if (use_plimits[j]) {
				if (!defined_prior_limits[i]) {
					untransformed_prior_limits_lo[i] = lower[j];
					untransformed_prior_limits_hi[i] = upper[j];
					defined_prior_limits[i] = true;
				} else {
					if (untransformed_prior_limits_lo[i] < lower[j]) {
						untransformed_prior_limits_lo[i] = lower[j];
					}
					if (untransformed_prior_limits_hi[i] > upper[j]) {
						untransformed_prior_limits_hi[i] = upper[j];
					}
				}
			}
		}
		transform_prior_limits(pi,pf);
		set_prior_norms(pi,pf);
	}
	void clear_prior_limit(const int i)
	{
		if (i >= nparams) die("parameter chosen for penalty limit is greater than total number of parameters (%i vs %i)",i,nparams);
		defined_prior_limits[i] = false;
		untransformed_prior_limits_lo[i] = -VERY_LARGE;
		untransformed_prior_limits_hi[i] = VERY_LARGE;
		prior_limits_lo[i] = -VERY_LARGE;
		prior_limits_hi[i] = VERY_LARGE;
		prior_norms[i] = 1.0;
	}
	bool set_param_prior(const int param_num, const string prior_type_string, const vector<double> &prior_params, const int param_num2 = -1)
	{
		int n_required_params = -1;
		Prior prior_type;
		if (prior_type_string=="none") { prior_type = UNIFORM_PRIOR; n_required_params = 0; }
		else if (prior_type_string=="log") { prior_type = LOG_PRIOR; n_required_params = 0; }
		else if (prior_type_string=="gaussian") { prior_type = GAUSS_PRIOR; n_required_params = 2; }
		else if (prior_type_string=="gauss2") { prior_type = GAUSS2_PRIOR; n_required_params = 5; }
		else { warn("prioration type not recognized"); return false; }
		if (prior_params.size() != n_required_params) { warn("wrong number of prioration parameters passed into set_param_prior method"); return false; }

		if (prior_type==UNIFORM_PRIOR) priors[param_num]->set_uniform();
		else if (prior_type==LOG_PRIOR) priors[param_num]->set_log();
		else if (prior_type==GAUSS_PRIOR) priors[param_num]->set_gaussian(prior_params[0],prior_params[1]);
		else if (prior_type==GAUSS2_PRIOR) {
			priors[param_num]->set_gauss2(param_num,param_num2,prior_params[0],prior_params[1],prior_params[2],prior_params[3],prior_params[4]);
			priors[param_num2]->set_gauss2_secondary(param_num,param_num2);
		}
		set_prior_norm(param_num);
		return true;
	}
	bool set_param_transform(const int param_num, const string transform_type_string, const vector<double> &transform_params, const int param_num2 = -1)
	{
		int n_required_params = -1;
		Transform transform_type;
		if (transform_type_string=="none") { transform_type = NONE; n_required_params = 0; }
		else if (transform_type_string=="log") { transform_type = LOG_TRANSFORM; n_required_params = 0; }
		else if (transform_type_string=="ratio")
		{
			transform_type = RATIO;
			n_required_params = 0;
			if ((param_num2 < 0) or (param_num2 >= nparams)) {
				warn("invalid ratio_paramnum");
				return false;
			}
		}
		else if (transform_type_string=="gaussian") { transform_type = GAUSS_TRANSFORM; n_required_params = 2; }
		else if (transform_type_string=="linear") { transform_type = LINEAR_TRANSFORM; n_required_params = 2; }
		else { warn("transformation type not recognized"); return false; }
		if (transform_params.size() != n_required_params) { warn("wrong number of transformation parameters passed into set_param_transform method"); return false; }

		if (transform_type==NONE) transforms[param_num]->set_none();
		else if (transform_type==LOG_TRANSFORM) transforms[param_num]->set_log();
		else if (transform_type==RATIO) transforms[param_num]->set_ratio(param_num2);
		else if (transform_type==GAUSS_TRANSFORM) transforms[param_num]->set_gaussian(transform_params[0],transform_params[1]);
		else if (transform_type==LINEAR_TRANSFORM) transforms[param_num]->set_linear(transform_params[0],transform_params[1]);
		transform_parameter_name(param_num);
		transform_parameter(param_num);
		transform_stepsize(param_num);
		transform_prior_limit(param_num);
		set_prior_norm(param_num);
		return true;
	}
	void transform_parameters(const int pi = 0, int pf = -1)
	{
		if (pf==-1) pf = nparams;
		for (int i=pi; i < pf; i++) {
			if (transforms[i]->transform==NONE) values[i] = untransformed_values[i];
			else if (transforms[i]->transform==LOG_TRANSFORM) values[i] = log(untransformed_values[i])/M_LN10;
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				values[i] = erff((untransformed_values[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				values[i] = transforms[i]->a * untransformed_values[i] + transforms[i]->b;
			} else if (transforms[i]->transform==RATIO) {
				values[i] = untransformed_values[i]/untransformed_values[transforms[i]->ratio_paramnum];
			}
		}
	}
	void transform_parameter(const int paramnum)
	{
		transform_parameters(paramnum,paramnum+1);
	}
	void transform_prior_limits(const int pi = 0, int pf = -1)
	{
		if (pf==-1) pf = nparams;
		for (int i=pi; i < pf; i++) {
			if (transforms[i]->transform==NONE) {
				prior_limits_lo[i] = untransformed_prior_limits_lo[i];
				prior_limits_hi[i] = untransformed_prior_limits_hi[i];
			} else if (transforms[i]->transform==LOG_TRANSFORM) {
				prior_limits_lo[i] = log(untransformed_prior_limits_lo[i])/M_LN10;
				prior_limits_hi[i] = log(untransformed_prior_limits_hi[i])/M_LN10;
			} else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				prior_limits_lo[i] = erff((untransformed_prior_limits_lo[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
				prior_limits_hi[i] = erff((untransformed_prior_limits_hi[i] - transforms[i]->gaussian_pos)/(M_SQRT2*transforms[i]->gaussian_sig));
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				prior_limits_lo[i] = transforms[i]->a * untransformed_prior_limits_lo[i] + transforms[i]->b;
				prior_limits_hi[i] = transforms[i]->a * untransformed_prior_limits_hi[i] + transforms[i]->b;
				if (prior_limits_lo[i] > prior_limits_hi[i]) {
					double temp = prior_limits_lo[i]; prior_limits_lo[i] = prior_limits_hi[i]; prior_limits_hi[i] = temp;
				}
			} else if (transforms[i]->transform==RATIO) {
				prior_limits_lo[i] = 0; // these can be manually adjusted using 'fit priors range ...'
				prior_limits_hi[i] = 1; // these can be customized
			}
		}
	}
	void inverse_transform_prior_limits(const int pi = 0, int pf = -1)
	{
		if (pf==-1) pf = nparams;
		bool apply_ratio_transform_afterwards = false;
		for (int i=pi; i < pf; i++) {
			if (transforms[i]->transform==NONE) {
				untransformed_prior_limits_lo[i] = prior_limits_lo[i];
				untransformed_prior_limits_hi[i] = prior_limits_hi[i];
			} else if (transforms[i]->transform==LOG_TRANSFORM) {
				untransformed_prior_limits_lo[i] = pow(10.0,prior_limits_lo[i]);
				untransformed_prior_limits_hi[i] = pow(10.0,prior_limits_hi[i]);
			} else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				untransformed_prior_limits_lo[i] = transforms[i]->gaussian_pos + M_SQRT2*transforms[i]->gaussian_sig*erfinv(prior_limits_lo[i]);
				untransformed_prior_limits_hi[i] = transforms[i]->gaussian_pos + M_SQRT2*transforms[i]->gaussian_sig*erfinv(prior_limits_hi[i]);
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				untransformed_prior_limits_lo[i] = (prior_limits_lo[i] - transforms[i]->b) / transforms[i]->a;
				untransformed_prior_limits_hi[i] = (prior_limits_hi[i] - transforms[i]->b) / transforms[i]->a;
			} else if (transforms[i]->transform==RATIO) {
				untransformed_prior_limits_lo[i] = prior_limits_lo[i]*untransformed_prior_limits_lo[transforms[i]->ratio_paramnum];
				untransformed_prior_limits_hi[i] = prior_limits_hi[i]*untransformed_prior_limits_hi[transforms[i]->ratio_paramnum];
			}
		}
	}
	void transform_prior_limit(const int paramnum)
	{
		transform_prior_limits(paramnum,paramnum+1);
	}
	void inverse_transform_prior_limit(const int paramnum)
	{
		inverse_transform_prior_limits(paramnum,paramnum+1);
	}
	void inverse_transform_parameters(double *params, double *inverse_transformed_params, const int pi = 0, int pf = -1)
	{
		if (pf==-1) pf = nparams;
		bool apply_ratio_transform_afterwards = false;
		for (int i=pi; i < pf; i++) {
			if (transforms[i]->transform==NONE) inverse_transformed_params[i] = params[i];
			else if (transforms[i]->transform==LOG_TRANSFORM) inverse_transformed_params[i] = pow(10.0,params[i]);
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				inverse_transformed_params[i] = transforms[i]->gaussian_pos + M_SQRT2*transforms[i]->gaussian_sig*erfinv(params[i]);
			} else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				inverse_transformed_params[i] = (params[i] - transforms[i]->b) / transforms[i]->a;
			} else if (transforms[i]->transform==RATIO) {
				if (transforms[i]->ratio_paramnum < i) {
					inverse_transformed_params[i] = params[i]*inverse_transformed_params[transforms[i]->ratio_paramnum];
				} else apply_ratio_transform_afterwards = true;
			}
		}
		if (apply_ratio_transform_afterwards) {
			for (int i=pi; i < pf; i++) {
				if (transforms[i]->transform==RATIO) inverse_transformed_params[i] = params[i]*inverse_transformed_params[transforms[i]->ratio_paramnum];
			}
		}
	}
	void inverse_transform_parameters(const int pi = 0, int pf = -1)
	{
		if (pf==-1) pf = nparams;
		inverse_transform_parameters(values,untransformed_values,pi,pf);
	}
	void inverse_transform_parameter(const int paramnum)
	{
		inverse_transform_parameters(values,untransformed_values,paramnum,paramnum+1);
	}
	void inverse_transform_parameters(double *params)
	{
		inverse_transform_parameters(params,params);
	}
	void transform_parameter_names(const int pi = 0, int pf = -1)
	{
		if (pf==-1) pf = nparams;
		for (int i=pi; i < pf; i++) {
			if (transforms[i]->transform==NONE) {
				param_names[i] = untransformed_param_names[i];
				if (untransformed_latex_names != NULL) latex_names[i] = untransformed_latex_names[i];
			}
			else if (transforms[i]->transform==LOG_TRANSFORM) {
				param_names[i] = "log(" + untransformed_param_names[i] + ")";
				if (untransformed_latex_names != NULL) latex_names[i] = "\\log(" + untransformed_latex_names[i] + ")";
			}
			else if (transforms[i]->transform==GAUSS_TRANSFORM) {
				param_names[i] = "u{" + untransformed_param_names[i] + "}";
				if (untransformed_latex_names != NULL) latex_names[i] = "u\\{" + untransformed_latex_names[i] + "\\}";
			}
			else if (transforms[i]->transform==LINEAR_TRANSFORM) {
				param_names[i] = "L{" + untransformed_param_names[i] + "}";
				if (untransformed_latex_names != NULL) latex_names[i] = "L\\{" + untransformed_latex_names[i] + "\\}";
			}
			else if (transforms[i]->transform==RATIO) {
				param_names[i] = untransformed_param_names[i] + "_over_" + untransformed_param_names[transforms[i]->ratio_paramnum];
				if (untransformed_latex_names != NULL) latex_names[i] = untransformed_latex_names[i] + "/" + untransformed_latex_names[transforms[i]->ratio_paramnum];
			}
		}
		override_parameter_names(); // allows for manually setting parameter untransformed_param_names
	}
	void transform_parameter_name(const int paramnum)
	{
		transform_parameter_names(paramnum,paramnum+1);
	}
	bool set_override_parameter_name(const int i, const string name)
	{
		bool unique_name = true;
		for (int j=0; j < nparams; j++) {
			if ((i != j) and (((override_names[j] != "") and (override_names[j]==name)) or (untransformed_param_names[j]==name))) unique_name = false;
		}
		if (!unique_name) return false;
		override_names[i] = name;
		param_names[i] = name;
		return true;
	}
	void override_parameter_names()
	{
		for (int i=0; i < nparams; i++) {
			if (override_names[i] != "") param_names[i] = override_names[i];
		}
	}
	void transform_stepsize(const int i)
	{
		// It would be better to have it pass in the current value of the parameters, then use the default
		// (untransformed) stepsize to define the transformed stepsize. For example, the log stepsize would
		// be log((pi+step)/pi). But passing in the parameter values is a bit of a pain...do this later
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
	void transform_stepsizes()
	{
		for (int i=0; i < nparams; i++) transform_stepsize(i);
	}
	void add_prior_terms_to_loglike(double *params, double& loglike)
	{
		double dloglike,dloglike_tot=0;
		for (int i=0; i < nparams; i++) {
			if (priors[i]->prior!=UNIFORM_PRIOR) {
				dloglike_tot += log(prior_norms[i]); // Normalize the prior for the bayesian evidence
				if (priors[i]->prior==LOG_PRIOR) {
					dloglike = log(params[i]);
					dloglike_tot += dloglike;
				}
				else if (priors[i]->prior==GAUSS_PRIOR) {
					dloglike = SQR((params[i] - priors[i]->gaussian_pos)/priors[i]->gaussian_sig)/2.0;
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
		loglike += dloglike_tot;
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
	void set_prior_norms(const int pi = 0, int pf = -1)
	{
		// flat priors are automatically given a norm of 1.0, since we'll be transforming to the unit hypercube when doing nested sampling;
		// however a correction is required for other priors
		if (pf==-1) pf = nparams;
		for (int i=pi; i < pf; i++) {
			if (priors[i]->prior!=UNIFORM_PRIOR) {
				if (priors[i]->prior==LOG_PRIOR) prior_norms[i] = log(prior_limits_hi[i]/prior_limits_lo[i]);
				else if (priors[i]->prior==GAUSS_PRIOR) {
					prior_norms[i] = (erff((prior_limits_hi[i] - priors[i]->gaussian_pos)/(M_SQRT2*priors[i]->gaussian_sig)) - erff((prior_limits_lo[i] - priors[i]->gaussian_pos)/(M_SQRT2*priors[i]->gaussian_sig))) * M_SQRT_HALFPI * priors[i]->gaussian_sig;
				}
				prior_norms[i] /= (prior_limits_hi[i] - prior_limits_lo[i]); // correction since we are transforming to the unit hypercube
			}
		}
	}
	void set_prior_norm(const int paramnum)
	{
		set_prior_norms(paramnum,paramnum+1);
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
			delete_param_ptrs();
			set_null_param_ptrs();
		}
		nparams = 0;
	}
	void delete_param_ptrs() {
		if (nparams > 0) {
			delete[] untransformed_values;
			delete[] values;
			for (int i=0; i < nparams; i++) {
				delete priors[i];
				delete transforms[i];
			}
			delete[] priors;
			delete[] transforms;
			delete[] untransformed_param_names;
			delete[] param_names;
			delete[] untransformed_latex_names;
			delete[] latex_names;
			delete[] override_names;
			delete[] stepsizes;
			delete[] auto_stepsize;
			delete[] subplot_param;
			delete[] hist2d_param;
			delete[] prior_norms;
			delete[] untransformed_prior_limits_lo;
			delete[] untransformed_prior_limits_hi;
			delete[] defined_prior_limits;
			delete[] prior_limits_lo;
			delete[] prior_limits_hi;
		}

	}
	void set_null_param_ptrs()
	{
		priors = NULL;
		transforms = NULL;
		untransformed_param_names = NULL;
		param_names = NULL;
		untransformed_latex_names = NULL;
		latex_names = NULL;
		override_names = NULL;
		untransformed_values = NULL;
		values = NULL;
		stepsizes = NULL;
		auto_stepsize = NULL;
		hist2d_param = NULL;
		subplot_param = NULL;
		prior_norms = NULL;
		untransformed_prior_limits_lo = NULL;
		untransformed_prior_limits_hi = NULL;
		defined_prior_limits = NULL;
		prior_limits_lo = NULL;
		prior_limits_hi = NULL;
	}
	void set_null_ptrs_and_values()
	{
		nparams = 0;
		qlens = NULL;
		set_null_param_ptrs();
	}
	~ParamList()
	{
		delete_param_ptrs();
	}
};

struct DerivedParamList
{
	static constexpr double VERY_LARGE = 1e30;
	QLens* qlens;

	int n_dparams;
	DerivedParam** dparams;
	bool *hist2d_dparam;
	bool *subplot_dparam;
	string *dparam_names;
	DerivedParamList() {
		set_null_ptrs_and_values();
	}
	DerivedParamList(QLens* qlens_in) : DerivedParamList() {
		qlens = qlens_in;
	}
	DerivedParamList(DerivedParamList& dparam_list_in) {
		n_dparams = dparam_list_in.n_dparams;
		if (n_dparams > 0) {
			dparam_names = new string[n_dparams];
			hist2d_dparam = new bool[n_dparams];
			subplot_dparam = new bool[n_dparams];
			dparams = new DerivedParam*[n_dparams];
			for (int i=0; i < n_dparams; i++) {
				dparam_names[i] = dparam_list_in.dparam_names[i];
				hist2d_dparam[i] = dparam_list_in.hist2d_dparam[i];
				subplot_dparam[i] = dparam_list_in.subplot_dparam[i];
				dparams[i] = new DerivedParam(dparam_list_in.dparams[i]);
			}
		} else {
			dparams = NULL;
			dparam_names = NULL;
			hist2d_dparam = NULL;
			subplot_dparam = NULL;
		}
	}
	DerivedParamList(DerivedParamList& dparam_list_in, QLens* qlens_in) : DerivedParamList(dparam_list_in) {
		qlens = qlens_in;
	}
	//void update_param_list(string* param_names_in, string* latex_names_in, double* stepsizes_in, const bool check_current_params = false);
	//bool print_parameter_values();
	//string mkstring_doub(const double db);
	//string mkstring_int(const int i);
	//string get_param_values_string();

	bool add_dparam(const string param_type, const double param, const int lensnum, const double param2, const bool use_kpc);
	bool remove_dparam(const int dparam_number);
	bool rename_dparam(const int dparam_number, const string newname, const string new_latex_name) {
		if (dparam_number >= n_dparams) { warn("Specified derived parameter does not exist"); return false; }
		dparam_names[dparam_number] = newname;
		dparams[dparam_number]->rename(newname,new_latex_name);
		return true;
	}
	double get_dparam(const int i)
	{
		if (i < n_dparams) return dparams[i]->get_derived_param(qlens);
		else {
			die("specified derived parameter index has not been created");
		}
		return -VERY_LARGE;
	}
	void get_dparams(double *dparam_vals)
	{
		for (int i=0; i < n_dparams; i++) {
			dparam_vals[i] = dparams[i]->get_derived_param(qlens);
		}
	}
	void clear_dparams()
	{
		delete_dparam_ptrs();
		if (n_dparams > 0) {
			dparams = NULL;
			dparam_names = NULL;
			hist2d_dparam = NULL;
			subplot_dparam = NULL;
			n_dparams = 0;
		}
	}
	bool print_dparam_list()
	{
		bool status = true;
		if (n_dparams > 0) {
			if (qlens) {
				for (int i=0; i < n_dparams; i++) {
					std::cout << i << ". " << std::flush;
					dparams[i]->print_param_description(qlens);
				}
			} else {
				status = false;
			}
		}
		else {
			std::cout << "No derived parameters have been created" << std::endl;
		}
		return status;
	}
	int lookup_param_number(const string pname)
	{
		int pnum = -1;
		for (int i=0; i < n_dparams; i++) {
			if (dparam_names[i]==pname) pnum = i;
		}
		return pnum;
	}
	string lookup_param_name(const int i)
	{
		string name = dparam_names[i];
		return name;
	}
	bool exclude_hist2d_param(const string pname)
	{
		bool found_name = false;
		for (int i=0; i < n_dparams; i++) {
			if (dparam_names[i]==pname) {
				hist2d_dparam[i] = false;
				found_name = true;
				break;
			}
		}
		return found_name;
	}
	bool hist2d_params_defined()
	{
		bool active_param = false;
		for (int i=0; i < n_dparams; i++) {
			if (!hist2d_dparam[i]) {
				active_param = true;
				break;
			}
		}
		return active_param;
	}
	bool hist2d_param_flag(const int i, string &name)
	{
		bool flag;
		name = dparam_names[i];
		flag = hist2d_dparam[i];
		return flag;
	}
	string print_excluded_hist2d_params()
	{
		string pstring = "";
		for (int i=0; i < n_dparams; i++) {
			if (!hist2d_dparam[i]) pstring += dparam_names[i] + " ";
		}
		return pstring;
	}
	void reset_hist2d_params()
	{
		for (int i=0; i < n_dparams; i++) hist2d_dparam[i] = true;
	}
	bool set_subplot_param(const string pname)
	{
		bool found_name = false;
		for (int i=0; i < n_dparams; i++) {
			if (dparam_names[i]==pname) {
				subplot_dparam[i] = true;
				found_name = true;
				break;
			}
		}
		return found_name;
	}
	bool subplot_params_defined()
	{
		bool active_param = false;
		for (int i=0; i < n_dparams; i++) {
			if (subplot_dparam[i]) {
				active_param = true;
				break;
			}
		}
		return active_param;
	}
	bool subplot_param_flag(const int i, string &name)
	{
		bool flag;
		name = dparam_names[i];
		flag = subplot_dparam[i];
		return flag;
	}
	string print_subplot_params()
	{
		string pstring = "";
		for (int i=0; i < n_dparams; i++) {
			if (subplot_dparam[i]) pstring += dparam_names[i] + " ";
		}
		return pstring;
	}
	void reset_subplot_params()
	{
		for (int i=0; i < n_dparams; i++) subplot_dparam[i] = false;
	}
	void delete_dparam_ptrs(const bool include_dparam_objects = true) {
		if (n_dparams > 0) {
			if (include_dparam_objects) {
				// delete the actual derived parameter objects, and not just the arrays that point to them
				for (int i=0; i < n_dparams; i++) delete dparams[i];
			}
			delete[] dparams;
			delete[] dparam_names;
			delete[] hist2d_dparam;
			delete[] subplot_dparam;
		}
	}
	void set_null_dparam_ptrs()
	{
		dparams = NULL;
		dparam_names = NULL;
		hist2d_dparam = NULL;
		subplot_dparam = NULL;
	}
	void set_null_ptrs_and_values()
	{
		n_dparams = 0;
		qlens = NULL;
		set_null_dparam_ptrs();
	}
	~DerivedParamList()
	{
		delete_dparam_ptrs();
	}
};



#endif // PARAMS_H

