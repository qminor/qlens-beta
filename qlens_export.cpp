#include <pybind11/pybind11.h>
#include "qlens_wrapper.cpp"
#include "profile.h"
#include "params.h"
#include "cosmo.h"
#include "qlens.h"
#include <vector>

namespace py = pybind11;

double default_zlens = 0.5;
double default_zsrc = 2;
double default_zsrc_ref = 2;

QLens* global_qlens_ptr = NULL;

void process_init_lens_kwargs(int& pmode, Cosmology*& cosmo, QLens_Wrap*& qlens_ptr, double& z, double& zs, boolvector& vary_list, py::kwargs& kwargs)
{
	bool set_pmode=false, set_cosmo=false, set_qlens=false, set_z=false, set_zs=false, set_vary=false;
	if (kwargs) {
		for (auto item : kwargs) {
			if (py::cast<string>(item.first)=="pmode") {
				pmode = py::cast<int>(item.second);
				set_pmode = true;
			} else if (py::cast<string>(item.first)=="cosmology") {
				cosmo = py::cast<Cosmology*>(item.second);
				set_cosmo = true;
			} else if (py::cast<string>(item.first)=="qlens") {
				qlens_ptr = py::cast<QLens_Wrap*>(item.second);
				set_qlens = true;
			} else if (py::cast<string>(item.first)=="z") {
				z = py::cast<double>(item.second);
				set_z = true;
			} else if (py::cast<string>(item.first)=="zs") {
				zs = py::cast<double>(item.second);
				set_zs = true;
			} else if (py::cast<string>(item.first)=="vary") {
				py::list py_vary_list = py::cast<py::list>(item.second);
				vary_list.input(py_vary_list.size());
				int iter = 0;
				for (auto item : py_vary_list) {
					vary_list[iter] = py::cast<bool>(item); iter++;
				}
				set_vary = true;
			}
		}
		if (set_pmode) kwargs.attr("pop")("pmode");
		if (set_cosmo) kwargs.attr("pop")("cosmology");
		if (set_qlens) kwargs.attr("pop")("qlens");
		if (set_z) kwargs.attr("pop")("z");
		if (set_zs) kwargs.attr("pop")("zs");
		if (set_vary) kwargs.attr("pop")("vary");

		if ((set_qlens) and (!set_cosmo) and (qlens_ptr != NULL)) cosmo = qlens_ptr->cosmo;
	}
}	

PYBIND11_MODULE(qlens, m) {
	m.doc() = "QLens Python Plugin"; // optional module docstring

	//py::class_<QLens, std::unique_ptr<QLens, py::nodelete>>(m, "QLens_base")
		//.def(py::init<>([](){return new QLens();}))
	//;

	py::class_<ParamList>(m, "ParamList")
		.def(py::init<>([](QLens_Wrap* qlens_in){return new ParamList(qlens_in);}))
		.def("values", [](ParamList &current) {
			//current.print_parameter_values();
			py::list vals(current.nparams);
			for (int i=0; i < current.nparams; i++) vals[i] = current.values[i];
				return vals;
		})
		.def("__getitem__", [](ParamList &current, size_t index) {
			//current.print_parameter_values();
			return &current.values[index];
		})
		.def("print",&ParamList::print_parameter_values)
		.def("print_stepsizes",&ParamList::print_stepsizes)
		.def("print_priors",&ParamList::print_priors_and_limits)
		.def("print_transforms",&ParamList::print_priors_and_transforms)
		.def("priors",&ParamList::print_priors_and_limits)
		.def("transforms",&ParamList::print_priors_and_transforms)
		.def("stepsizes", [](ParamList &current) {
			py::list steps(current.nparams);
			for (int i=0; i < current.nparams; i++) steps[i] = current.stepsizes[i];
				return steps;
		})
		.def("update", [](ParamList &current, py::dict dict){
			for (auto item : dict) {
				if(!current.update_param_value(py::cast<string>(item.first), py::cast<double>(item.second)))
					return false;
			}
			return true;
		})
		.def("update", [](ParamList &current, const string name, const double value){
			return current.update_param_value(name, value);
		})
		.def("stepsize", [](ParamList &current, const string name, py::args &args, py::kwargs &kwargs){
			int paramnum = -1;
			if (name != "") {
				paramnum = current.lookup_param_number(name);
				if (paramnum < 0) throw std::runtime_error("specified parameter name '" + name + "' not recognized");
			}
			double stepsize = -1.0;
			double scale_factor = -1.0;
			if (args.size()>=1) {
				try {
					stepsize = args[0].cast<double>();
				} catch (...) {
					throw std::runtime_error("Invalid stepsize for argument '" + name + "'");
				}
				if (args.size() > 1) throw std::runtime_error("only two non-keyword arguments allowed to set_stepsize function (param_name, stepsize)");
			}
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="scale") {
					try {
						scale_factor = py::cast<double>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid scale factor for argument '" + name + "'");
					}
				}
			}
			if (stepsize > 0) current.set_stepsize(paramnum,stepsize);
			if (scale_factor > 0) {
				current.scale_stepsize(paramnum,scale_factor);
			}
			if ((stepsize <= 0) and (scale_factor <= 0)) {
				if (paramnum <= 0) current.print_stepsizes();
			};
		})
		.def("transform", [](ParamList &current, const string name, const string transform_type_in, py::kwargs &kwargs){
			int paramnum = current.lookup_param_number(name);
			if (paramnum < 0) throw std::runtime_error("specified parameter name '" + name + "' not recognized");
			const int n_transform_types = 5;
			string transform_types[n_transform_types] = {"none","log","ratio","gaussian","linear"};
			bool found_transform_type = false;
			for (int i=0; i < n_transform_types; i++) {
				if (transform_type_in==transform_types[i]) found_transform_type = true;
			}
			if (!found_transform_type) throw std::runtime_error("specified transformation type '" + transform_type_in + "' not valid");
			vector<double> transform_params;
			int ratio_paramnum = -1;

			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="a") {
					if (transform_type_in != "linear") throw std::runtime_error("specified transform parameter 'a' only valid for 'linear' transformation");
					transform_params.insert(transform_params.begin(),py::cast<double>(item.second));
				} else if (py::cast<string>(item.first)=="b") {
					if (transform_type_in != "linear") throw std::runtime_error("specified transform parameter 'b' only valid for 'linear' transformation");
					transform_params.push_back(py::cast<double>(item.second));
				} else if (py::cast<string>(item.first)=="mean") {
					if (transform_type_in != "gaussian") throw std::runtime_error("specified transform parameter 'mean' only valid for 'gaussian' transformation");
					transform_params.insert(transform_params.begin(),py::cast<double>(item.second));
				} else if (py::cast<string>(item.first)=="sig") {
					if (transform_type_in != "gaussian") throw std::runtime_error("specified transform parameter 'sig' only valid for 'gaussian' transformation");
					transform_params.push_back(py::cast<double>(item.second));
				} else if (py::cast<string>(item.first)=="ratio_paramnum") {
					if (transform_type_in != "ratio") throw std::runtime_error("specified transform parameter 'ratio_paramnum' only valid for 'ratio' transformation");
					ratio_paramnum = py::cast<int>(item.second);
				} else throw std::runtime_error("unknown argument to param.transform(...)");
			}
			if (!current.set_param_transform(paramnum,transform_type_in,transform_params,ratio_paramnum)) throw std::runtime_error("could not perform parameter transformation");
		})
		.def("set_prior", [](ParamList &current, const string name, const string prior_type_in, py::kwargs &kwargs){
			int paramnum = current.lookup_param_number(name);
			if (paramnum < 0) throw std::runtime_error("specified parameter name '" + name + "' not recognized");
			const int n_prior_types = 3;
			string prior_types[n_prior_types] = {"uniform","log","gaussian"};
			bool found_prior_type = false;
			for (int i=0; i < n_prior_types; i++) {
				if (prior_type_in==prior_types[i]) found_prior_type = true;
			}
			if (!found_prior_type) throw std::runtime_error("specified prior type '" + prior_type_in + "' not valid");
			vector<double> prior_params;
			int joint_prior_paramnum = -1;

			// add 'gauss2' prior later
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="mean") {
					if (prior_type_in != "gaussian") throw std::runtime_error("specified prior parameter 'mean' only valid for 'gaussian' prior");
					prior_params.insert(prior_params.begin(),py::cast<double>(item.second));
				} else if (py::cast<string>(item.first)=="sig") {
					if (prior_type_in != "gaussian") throw std::runtime_error("specified prior parameter 'sig' only valid for 'gaussian' prior");
					prior_params.push_back(py::cast<double>(item.second));
				} else throw std::runtime_error("unknown argument to param.set_prior(...)");
			}
			if (!current.set_param_prior(paramnum,prior_type_in,prior_params,joint_prior_paramnum)) throw std::runtime_error("could not set parameter prior");
		})
		.def("set_limits", [](ParamList &curr, const std::string &paramname, const double lower, const double upper){
			if (curr.set_prior_limit(paramname,lower,upper)==false) {
				throw std::runtime_error("could not set limits for given parameter " + paramname);
			}
		})
		.def("set_limits", [](ParamList &curr, py::list list){
			std::string paramname;
			double lower, upper;
			for (auto arr : list){
				try {
					std::tuple<std::string, double, double> extracted = py::cast<std::tuple<std::string, double, double>>(arr);
					paramname = std::get<0>(extracted);
					lower = std::get<1>(extracted);
					upper = std::get<2>(extracted);
				} catch (...) {
					throw std::runtime_error("Error setting parameter limits. Input should be an array of tuples. Ex: [(<paramname>, lower, upper), (<paramname>, lower, upper)]");
				}
				if (curr.set_prior_limit(paramname,lower,upper)==false) {
					throw std::runtime_error("could not set limits for given parameter " + paramname);
				}
			}
		})
		.def("clear",&ParamList::clear_params)
		.def("rename", [](ParamList &current, const string old_name, const string new_name){
			int paramnum = current.lookup_param_number(old_name);
			if (paramnum < 0) throw std::runtime_error("specified parameter name '" + old_name + "' not recognized");
			if (!current.set_override_parameter_name(paramnum,new_name)) throw std::runtime_error("parameter name not unique; parameter could not be renamed");
		})
		.def("__repr__", [](ParamList &a) {
			string outstring = a.get_param_values_string();
			return("\n" + outstring);
		})
		;


	py::class_<DerivedParamList>(m, "DerivedParamList")
		.def(py::init<>([](QLens_Wrap* qlens_in){return new DerivedParamList(qlens_in);}))
		.def("values", [](DerivedParamList &current) {
			py::list vals(current.n_dparams);
			for (int i=0; i < current.n_dparams; i++) vals[i] = current.get_dparam(i);
				return vals;
		})
		.def("print",&DerivedParamList::print_dparam_list)
		.def("add", [](DerivedParamList &current, const string param_type, py::args &args, py::kwargs &kwargs){
			double param1=-1.0, param2=-1.0;
			bool use_kpc = false;
			int lens_number = -1;
			if (args.size()>=1) {
				try {
					param1 = args[0].cast<double>();
				} catch (...) {
					throw std::runtime_error("Invalid param1 for derived parameter type '" + param_type + "'");
				}
				if (args.size()>=2) {
					try {
						param2 = args[1].cast<double>();
					} catch (...) {
						throw std::runtime_error("Invalid param2 for derived parameter type '" + param_type + "'");
					}
				}
				if (args.size()>=3) {
					throw std::runtime_error("Too many non-keyword arguments for derived parameter type '" + param_type + "'");
				}
			}
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="lens") {
					lens_number = py::cast<int>(item.second);
				} else if (py::cast<string>(item.first)=="use_kpc") {
					use_kpc = py::cast<bool>(item.second);
				} else {
					throw std::runtime_error("Keyword argument not recognized for derived parameter type '" + param_type + "'");
				}
			}
			return current.add_dparam(param_type,param1,lens_number,param2,use_kpc);
		})
		.def("remove",&DerivedParamList::remove_dparam)
		.def("clear",&DerivedParamList::clear_dparams)
		.def("rename", [](DerivedParamList &current, const string old_name, const string new_name, const string new_latex_name=""){
			int paramnum = current.lookup_param_number(old_name);
			if (paramnum < 0) throw std::runtime_error("specified parameter name '" + old_name + "' not recognized");
			if (!current.rename_dparam(paramnum,new_name,new_latex_name)) throw std::runtime_error("parameter name not unique; parameter could not be renamed");
		})
		//.def("__repr__", [](DerivedParamList &a) {
			//string outstring = a.get_param_values_string();
			//return("\n" + outstring);
		//})
		;


	py::class_<ModelParams>(m, "ModelParams")
		.def(py::init<>([](){return new ModelParams();}))
		.def("update", [](ModelParams &current, py::dict dict){
			for (auto item : dict) {
				if(!current.update_specific_parameter(py::cast<string>(item.first), py::cast<double>(item.second)))
					return false;
			}
			return true;
		})
		.def("update", [](ModelParams &current, const string name, const double value){
			return current.update_specific_parameter(name, value);
		})
		.def("print_params", [](ModelParams &current) { current.print_parameters(false); })
		.def("print_vary_params", &ModelParams::print_vary_parameters)
		.def("vary", [](ModelParams &current, py::list list){ 
			std::vector<double> lst = py::cast<std::vector<double>>(list);
			boolvector val(lst.size());
			int iter = 0;
			for (auto item : list) {
					val[iter] = py::cast<bool>(item); iter++;
			}
			if (current.set_varyflags(val)==false) {
				throw std::runtime_error("Number of input vary flags does not match number of lens parameters");
			}
		})
		.def("set_limits", [](ModelParams &curr, const std::string &param, const double lower, const double upper){
			if (curr.set_limits_specific_parameter(param,lower,upper)==false) {
				throw std::runtime_error("could not set limits for given parameter " + param);
			}
		})
		.def("set_limits", [](ModelParams &curr, py::list list){
			std::string paramname;
			double lower, upper;
			for (auto arr : list){
				try {
					std::tuple<std::string, double, double> extracted = py::cast<std::tuple<std::string, double, double>>(arr);
					paramname = std::get<0>(extracted);
					lower = std::get<1>(extracted);
					upper = std::get<2>(extracted);
				} catch (...) {
					throw std::runtime_error("Error setting parameter limits. Input should be an array of tuples. Ex: [(<paramname>, lower, upper), (<paramname>, lower, upper)]");
				}
				if (curr.set_limits_specific_parameter(paramname,lower,upper)==false) {
					throw std::runtime_error("could not set limits for given parameter " + paramname);
				}
			}
		})
		.def("__repr__", [](ModelParams &a) {
				string outstring = a.get_parameters_string();
				return("\n" + outstring);
		})
		;

	py::class_<Cosmology, ModelParams, std::unique_ptr<Cosmology, py::nodelete>>(m, "Cosmology")
		//.def(py::init<>([](){return new Cosmology();}))
		.def(py::init<>([](py::kwargs &kwargs){
			if (kwargs) {
				double omega_m = 0.3;
				double hubble = 0.7;
				for (auto item : kwargs) {
					if (py::cast<string>(item.first)=="omega_m") {
						omega_m = py::cast<double>(item.second);
					}
					else if (py::cast<string>(item.first)=="hubble") {
						hubble = py::cast<double>(item.second);
					} else throw std::runtime_error("unknown argument to Cosmo");
				}
				return new Cosmology(omega_m,hubble);
			} else {
				return new Cosmology();
			}
		}))
		.def_property("omega_m", &Cosmology::get_omega_m, &Cosmology::set_omega_m)
		.def_property("hubble", &Cosmology::get_hubble, &Cosmology::set_hubble)
		.def("sigma_crit_arcsec", &Cosmology::sigma_crit_arcsec)
		.def("sigma_crit_kpc", &Cosmology::sigma_crit_kpc)
		.def("angular_diameter_distance", [](Cosmology &curr, const double redshift) { return 1e-3*curr.angular_diameter_distance_exact(redshift); }) // units of Gpc
		.def("luminosity_distance", [](Cosmology &curr, const double redshift) { return 1e-3*curr.luminosity_distance_exact(redshift); }) // units of Gpc
		.def("comoving_distance", [](Cosmology &curr, const double redshift) { return 1e-3*curr.comoving_distance_exact(redshift); }) // units of Gpc
		.def("angular_diameter_distance_splined", [](Cosmology &curr, const double redshift) { return 1e-3*curr.angular_diameter_distance(redshift); }) // units of Gpc
		.def("luminosity_distance_splined", [](Cosmology &curr, const double redshift) { return 1e-3*curr.luminosity_distance(redshift); }) // units of Gpc
		.def("comoving_distance_splined", [](Cosmology &curr, const double redshift) { return 1e-3*curr.comoving_distance(redshift); }) // units of Gpc
		.def("kpc_to_arcsec", [](Cosmology &curr, const double redshift) { return 1.0e-3*(180/M_PI)*3600/curr.angular_diameter_distance_exact(redshift); })
		.def("arcsec_to_kpc", [](Cosmology &curr, const double redshift) { return 1.0e3*(M_PI/180)*(curr.angular_diameter_distance_exact(redshift)/3600); })
		.def("__repr__", [](Cosmology &a) {
				string outstring = a.get_parameters_string();
				return("cosmology (flat LCDM): " + outstring);
		})
		;

	py::class_<LensList>(m, "LensList")
		.def(py::init<>([](QLens_Wrap* qlens_in){return new LensList(qlens_in);}))
		//.def("add",&LensList::add_lens)
		.def("add", [](LensList &current, LensProfile* lens_in, py::kwargs &kwargs){
			Shear *extshear = NULL;
			int anchor_center_lensnum = -1;
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="shear") {
					try {
						extshear = py::cast<Shear*>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid Shear lens object passed in for external shear");
					}
				} else if (py::cast<string>(item.first)=="anchor_center") {
					try {
						anchor_center_lensnum = py::cast<int>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid lens number passed in to anchor center to");
					}
				} else {
					throw std::runtime_error("Keyword argument not recognized for lens.add");
				}
			}
			if (anchor_center_lensnum >= 0) {
				if (anchor_center_lensnum >= current.nlens) throw std::runtime_error("Lens number for anchoring is greater than nlens-1");
			}
			if (extshear != NULL) current.add_lens_extshear(lens_in,extshear);
			else current.add_lens(lens_in);
			if (anchor_center_lensnum >= 0) {
				if (extshear != NULL) current.anchor_lens_center(current.nlens-2,anchor_center_lensnum);
				else current.anchor_lens_center(current.nlens-1,anchor_center_lensnum);
			}
		})
		.def("clear", [](LensList &current){
			current.clear();
		})
		.def("clear", [](LensList &current, const int num){
			current.clear(num);
		})
		.def("clear", [](LensList &current, const int min, const int max){
			current.clear(min,max);
		})
		.def("clear",&LensList::clear)
		.def("__getitem__", [](LensList &current, size_t index) {
			//current.print_parameter_values();
			return current.lenslistptr[index];
		})
		.def("__len__", [](LensList &current) {
			return current.nlens;
		})
		.def("__repr__", [](LensList &current) {
			string lens_info;
			for (int i=0; i < current.nlens; i++) lens_info += "\n" + current.lenslistptr[i]->get_parameters_string();
			return(lens_info);
		})
		;

	py::class_<SourceList>(m, "SourceList")
		.def(py::init<>([](QLens_Wrap* qsrc_in){return new SourceList(qsrc_in);}))
		//.def("add",&SourceList::add_lens)
		.def("add", [](SourceList &current, SB_Profile* src_in, py::kwargs &kwargs){
			bool is_lensed = true;
			bool anchor_to_src = false;
			bool anchor_to_lens = false;
			int anchor_center_srcnum = -1;
			int anchor_center_lensnum = -1;
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="is_lensed") {
					try {
						is_lensed = py::cast<bool>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid boolean argument for 'is_lensed' (should be True or False)");
					}
				} else if (py::cast<string>(item.first)=="anchor_center") {
					if (anchor_to_lens) throw std::runtime_error("cannot anchor to both a lens and a source");
					try {
						anchor_center_srcnum = py::cast<int>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid src number passed in to anchor center to");
					}
					anchor_to_src = true;
				} else if (py::cast<string>(item.first)=="anchor_lens_center") {
					if (anchor_to_src) throw std::runtime_error("cannot anchor to both a lens and a source");
					try {
						anchor_center_lensnum = py::cast<int>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid lens number passed in to anchor center to");
					}
					anchor_to_lens = true;
				} else {
					throw std::runtime_error("Keyword argument not recognized for src.add");
				}
			}
			if (anchor_to_src) {
				if (anchor_center_srcnum >= current.n_sb) throw std::runtime_error("src number for anchoring is greater than n_sb-1");
			}
			current.add_source(src_in,is_lensed);
			if (anchor_to_src) {
				current.anchor_center_to_source(current.n_sb-1,anchor_center_srcnum);
			} else if (anchor_to_lens) {
				current.anchor_center_to_lens(current.n_sb-1,anchor_center_lensnum);
			}
		})
		.def("clear", [](SourceList &current){
			current.clear();
		})
		.def("clear", [](SourceList &current, const int num){
			current.clear(num);
		})
		.def("clear", [](SourceList &current, const int min, const int max){
			current.clear(min,max);
		})
		.def("clear",&SourceList::clear)
		.def("__getitem__", [](SourceList &current, size_t index) {
			//current.print_parameter_values();
			return current.srclistptr[index];
		})
		.def("__len__", [](SourceList &current) {
			return current.n_sb;
		})
		.def("__repr__", [](SourceList &current) {
			string src_info;
			for (int i=0; i < current.n_sb; i++) src_info += "\n" + current.srclistptr[i]->get_parameters_string();
			return(src_info);
		})
		;

	py::class_<PixSrcList>(m, "PixSrcList")
		.def(py::init<>([](QLens_Wrap* qsrc_in){return new PixSrcList(qsrc_in);}))
		//.def("add",&PixSrcList::add_lens)
		.def("add", [](PixSrcList &current, py::kwargs &kwargs){
			int band = 0;
			double zsrc = current.qlens->source_redshift;
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="band") {
					try {
						band = py::cast<int>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid band number");
					}
				} else if (py::cast<string>(item.first)=="z") {
					try {
						zsrc = py::cast<double>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid src redshift");
					}
				} else {
					throw std::runtime_error("Keyword argument not recognized for pixsrc.add");
				}
			}
			current.add_pixsrc(zsrc,band);
		})

		.def("clear", [](PixSrcList &current){
			current.clear();
		})
		.def("clear", [](PixSrcList &current, const int num){
			current.clear(num);
		})
		.def("clear", [](PixSrcList &current, const int min, const int max){
			current.clear(min,max);
		})
		.def("clear",&PixSrcList::clear)
		.def("__getitem__", [](PixSrcList &current, size_t index) {
			//current.print_parameter_values();
			return current.pixsrclist_ptr[index];
		})
		.def("__len__", [](PixSrcList &current) {
			return current.n_pixsrc;
		})
		.def("__repr__", [](PixSrcList &current) {
			string src_info;
			for (int i=0; i < current.n_pixsrc; i++) src_info += "\n" + current.pixsrclist_ptr[i]->get_parameters_string();
			return(src_info);
		})
		;

	py::class_<PtSrcList>(m, "PtSrcList")
		.def(py::init<>([](QLens_Wrap* qsrc_in){return new PtSrcList(qsrc_in);}))
		//.def("add",&PtSrcList::add_lens)
		.def("add", [](PtSrcList &current, const double x, const double y, py::kwargs &kwargs){
			double zsrc = current.qlens->source_redshift;
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="z") {
					try {
						zsrc = py::cast<double>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid src redshift");
					}
				} else {
					throw std::runtime_error("Keyword argument not recognized for ptsrc.add");
				}
			}
			lensvector pos;
			pos[0]=x;
			pos[1]=y;
			current.add_ptsrc(zsrc,pos,false);
		})

		.def("clear", [](PtSrcList &current){
			current.clear();
		})
		.def("clear", [](PtSrcList &current, const int num){
			current.clear(num);
		})
		.def("clear", [](PtSrcList &current, const int min, const int max){
			current.clear(min,max);
		})
		//.def("clear",&PtSrcList::clear)
		.def("__getitem__", [](PtSrcList &current, size_t index) {
			//current.print_parameter_values();
			return current.ptsrclist_ptr[index];
		})
		.def("__len__", [](PtSrcList &current) {
			return current.n_ptsrc;
		})
		.def("__repr__", [](PtSrcList &current) {
			string src_info;
			for (int i=0; i < current.n_ptsrc; i++) src_info += "\n" + current.ptsrclist_ptr[i]->get_parameters_string();
			return(src_info);
		})
		;

	py::class_<LensProfile>(m, "LensProfile")
		.def(py::init<>([](){return new LensProfile();}))
		.def(py::init<const LensProfile*>())
		// .def(py::init<>([](const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int& nn, const double &acc, const double &qx_in, const double &f_in, Lens* lens_in){
		//		return new LensProfile(splinefile, zlens_in, zsrc_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, qx_in, f_in, lens_in);
		// }))
		.def("print_params", &LensProfile::print_parameters)
		.def("print_vary_params", &LensProfile::print_vary_parameters)
		.def("get_model_name", &LensProfile::get_model_name)
		.def("update", [](LensProfile &current, py::dict dict){
			for (auto item : dict) {
				if(!current.update_specific_parameter(py::cast<string>(item.first), py::cast<double>(item.second)))
					return false;
			}
			return true;
		})
		.def("update", [](LensProfile &current, const string name, const double value){
			return current.update_specific_parameter(name, value);
		})
		.def("set_center", &LensProfile::set_center)
		.def("vary", [](LensProfile &current, py::list list){ 
			std::vector<double> lst = py::cast<std::vector<double>>(list);
			boolvector val(lst.size());
			int iter = 0;
			for (auto item : list) {
					val[iter] = py::cast<bool>(item); iter++;
			}
			if (current.set_vary_flags(val)==false) {
				throw std::runtime_error("Number of input vary flags does not match number of lens parameters");
			}
		})
		.def("set_limits", [](LensProfile &curr, const std::string &param, const double lower, const double upper){
			if (curr.set_limits_specific_parameter(param,lower,upper)==false) {
				throw std::runtime_error("could not set limits for given parameter " + param);
			}
		})
		.def("set_limits", [](LensProfile &curr, py::list list){
			std::string paramname;
			double lower, upper;
			for (auto arr : list){
				try {
					std::tuple<std::string, double, double> extracted = py::cast<std::tuple<std::string, double, double>>(arr);
					paramname = std::get<0>(extracted);
					lower = std::get<1>(extracted);
					upper = std::get<2>(extracted);
				} catch (...) {
					throw std::runtime_error("Error setting parameter limits. Input should be an array of tuples. Ex: [(<paramname>, lower, upper), (<paramname>, lower, upper)]");
				}
				if (curr.set_limits_specific_parameter(paramname,lower,upper)==false) {
					throw std::runtime_error("could not set limits for given parameter " + paramname);
				}
			}
		})
		//.def("get_prior_limits", [](LensProfile &curr){
			//int nparams = current.n_vary_params;
			//dvector lower(nparams);
			//dvector upper(nparams);
			//current.get_limits(lower,upper);
			//// I think it should return a list of tuples with lower limit and upper limit. Do this later
		//})
		.def("anchor_param", [](LensProfile &current, const string name, LensProfile* param_anchor_lens, const string anchor_param_name){
			int paramnum = -1;
			int anchor_paramnum = -1;
			if (!current.lookup_parameter_number(name,paramnum)) throw std::runtime_error("could not find parameter '" + name +"'");
			if (current.get_vary_flag(paramnum)==true) throw std::runtime_error("cannot anchor parameter if its vary flag is set to 'True'");
			if (!param_anchor_lens->lookup_parameter_number(anchor_param_name,anchor_paramnum)) throw std::runtime_error("could not find parameter '" + anchor_param_name +"'");
			current.assign_anchored_parameter(paramnum,anchor_paramnum,false,false,1.0,1.0,param_anchor_lens);
		})
		.def("anchor_param", [](LensProfile &current, const string name, SB_Profile* param_anchor_source, const string anchor_param_name){
			int paramnum = -1;
			int anchor_paramnum = -1;
			if (!current.lookup_parameter_number(name,paramnum)) throw std::runtime_error("could not find parameter '" + name +"'");
			if (current.get_vary_flag(paramnum)==true) throw std::runtime_error("cannot anchor parameter if its vary flag is set to 'True'");
			if (!param_anchor_source->lookup_parameter_number(anchor_param_name,anchor_paramnum)) throw std::runtime_error("could not find parameter '" + anchor_param_name +"'");
			current.assign_anchored_parameter(paramnum,anchor_paramnum,false,false,1.0,1.0,param_anchor_source);
		})
		.def("anchor_center",&LensProfile::anchor_center_to_lens)
		.def("__repr__", [](LensProfile &a) {
				string outstring = a.get_parameters_string();
				return("\n" + outstring);
		})
		.def("kappa", &LensProfile::kappa)
		.def("potential", &LensProfile::potential)
		.def("deflection", [](LensProfile &current, const double x, const double y){ 
			py::list def(2);
			lensvector def_vec;
			current.deflection(x,y,def_vec);
			def[0] = def_vec[0];
			def[1] = def_vec[1];
			return def;
		})
		;

	py::class_<SPLE_Lens, LensProfile, std::unique_ptr<SPLE_Lens, py::nodelete>>(m, "SPLE")
		//.def(py::init<>([](){return new SPLE_Lens();}))
		.def(py::init<const SPLE_Lens*>())
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int pmode=0;
			Cosmology* cosmo_in = NULL;
			QLens_Wrap* qlens_ptr = NULL;
			double zlens = default_zlens;
			double zsrc_ref = default_zsrc_ref;
			boolvector vary_list;

			process_init_lens_kwargs(pmode, cosmo_in, qlens_ptr, zlens, zsrc_ref, vary_list, kwargs);

			double b,p2,s,q1,q2,xc,yc;
			b = py::cast<double>(dict["b"]);
			if (pmode==0) {
				try {
					p2 = py::cast<double>(dict["alpha"]);
				} catch (...) {
					p2 = 1.0;
				}
			} else if (pmode==1) {
				try {
					p2 = py::cast<double>(dict["gamma"]);
				} catch (...) {
					p2 = 2.0;
				}
			} else throw std::runtime_error("Can only choose pmode=0 or pmode=1");
			try {
				s = py::cast<double>(dict["s"]);
			} catch (...) {
				s = 0.0;
			}
			LensProfile::extract_geometric_params_from_map(q1,q2,xc,yc,py::cast<std::map<std::string,double>>(dict));
			SPLE_Lens* sple = new SPLE_Lens(zlens,zsrc_ref,b,p2,s,q1,q2,xc,yc,pmode,cosmo_in);
			if (qlens_ptr != NULL) qlens_ptr->add_lens(sple);
			if (vary_list.size() > 0) {
				if (sple->set_vary_flags(vary_list)==false) {
					throw std::runtime_error("Number of input vary flags does not match number of lens parameters");
				}
			}
			return sple;
		}))
		//.def("initialize", [](SPLE_Lens &current, py::dict dict){
				// do you really need initialize? Should just require initialization when creating object
		//})
		//.def("__repr__", [](SPLE_Lens &a) {
				//string outstring = a.get_parameters_string();
				//return("\n" + outstring);
		//})
		;

	py::class_<Shear, LensProfile, std::unique_ptr<Shear, py::nodelete>>(m, "Shear")
		//.def(py::init<>([](){return new Shear();}))
		.def(py::init<const Shear*>())
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int pmode=0;
			Cosmology* cosmo_in = NULL;
			QLens_Wrap* qlens_ptr = NULL;
			double zlens = default_zlens;
			double zsrc_ref = default_zsrc_ref;
			boolvector vary_list;
			process_init_lens_kwargs(pmode, cosmo_in, qlens_ptr, zlens, zsrc_ref, vary_list, kwargs);
			double p1, p2, xc, yc;
			if (!Shear::use_shear_component_params) {
					p1 = py::cast<double>(dict["shear"]);
					p2 = py::cast<double>(dict["theta"]);
			} else {
					p1 = py::cast<double>(dict["shear1"]);
					p2 = py::cast<double>(dict["shear2"]);
			}
			try {
					xc = py::cast<double>(dict["xc"]);
			} catch (...) {
				xc = 0.0;
			}
			try {
					yc = py::cast<double>(dict["yc"]);
			} catch (...) {
				yc = 0.0;
			}
			Shear* shearlens = new Shear(zlens,zsrc_ref,p1,p2,xc,yc,cosmo_in);
			if (qlens_ptr != NULL) qlens_ptr->add_lens(shearlens);
			if (vary_list.size() > 0) {
				if (shearlens->set_vary_flags(vary_list)==false) {
					throw std::runtime_error("Number of input vary flags does not match number of lens parameters");
				}
			}
			return shearlens;
		}))
		;

	py::class_<dPIE_Lens, LensProfile, std::unique_ptr<dPIE_Lens, py::nodelete>>(m, "dPIE")
		//.def(py::init<>([](){return new dPIE_Lens();}))
		.def(py::init<const dPIE_Lens*>()) 
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int pmode=0;
			Cosmology* cosmo_in = NULL;
			QLens_Wrap* qlens_ptr = NULL;
			double zlens = default_zlens;
			double zsrc_ref = default_zsrc_ref;
			boolvector vary_list;
			process_init_lens_kwargs(pmode, cosmo_in, qlens_ptr, zlens, zsrc_ref, vary_list, kwargs);

			double p1,p2,p3,q1,q2,xc,yc;
			if (pmode==0) {
				p1 = py::cast<double>(dict["b"]);
				p2 = py::cast<double>(dict["a"]);
				try {
					p3 = py::cast<double>(dict["s"]);
				} catch (...) {
					p3 = 0.0;
				}
			} else if (pmode==1) {
				p1 = py::cast<double>(dict["sigma0"]);
				p2 = py::cast<double>(dict["a_kpc"]);
				try {
					p3 = py::cast<double>(dict["s_kpc"]);
				} catch (...) {
					p3 = 0.0;
				}
			} else if (pmode==2) {
				p1 = py::cast<double>(dict["mtot"]);
				p2 = py::cast<double>(dict["a_kpc"]);
				try {
					p3 = py::cast<double>(dict["s_kpc"]);
				} catch (...) {
					p3 = 0.0;
				}
			} else throw std::runtime_error("Can only choose pmode=0,1, or 2");
			LensProfile::extract_geometric_params_from_map(q1,q2,xc,yc,py::cast<std::map<std::string,double>>(dict));

			dPIE_Lens* dpie = new dPIE_Lens(zlens,zsrc_ref,p1,p2,p3,q1,q2,xc,yc,pmode,cosmo_in);
			if (qlens_ptr != NULL) qlens_ptr->add_lens(dpie);
			if (vary_list.size() > 0) {
				if (dpie->set_vary_flags(vary_list)==false) {
					throw std::runtime_error("Number of input vary flags does not match number of lens parameters");
				}
			}
			return dpie;
		}))
		;

	py::class_<NFW, LensProfile, std::unique_ptr<NFW, py::nodelete>>(m, "NFW")
		//.def(py::init<>([](){return new NFW();}))
		.def(py::init<const NFW*>())
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int pmode=0;
			Cosmology* cosmo_in = NULL;
			QLens_Wrap* qlens_ptr = NULL;
			double zlens = default_zlens;
			double zsrc_ref = default_zsrc_ref;
			boolvector vary_list;
			bool use_median_c = false;
			bool anchor_median_c = true; // keep c set to median if use_median_c is turned on
			double c_median_factor = 1.0;
			if (kwargs) {
				process_init_lens_kwargs(pmode, cosmo_in, qlens_ptr, zlens, zsrc_ref, vary_list, kwargs);
				for (auto item : kwargs) {
					if (py::cast<string>(item.first)=="c_median") {
						use_median_c = py::cast<bool>(item.second);
					} else if (py::cast<string>(item.first)=="c_median_init") {
						use_median_c = py::cast<bool>(item.second);
						anchor_median_c = false;
					} else if (py::cast<string>(item.first)=="c_median_factor") {
						c_median_factor = py::cast<double>(item.second);
					} else {
						throw std::runtime_error("unknown argument to NFW");
					}
				}
			}
	
			double p1,p2,q1,q2,xc,yc;
			if (pmode==0) {
				p1 = py::cast<double>(dict["ks"]);
				p2 = py::cast<double>(dict["rs"]);
			} else if (pmode==1) {
				p1 = py::cast<double>(dict["mvir"]);
				if (!use_median_c) {
					p2 = py::cast<double>(dict["c"]);
				} else {
					p2 = 1.0; // dummy value to input before it assigns median concentration
				}
			} else if (pmode==2) {
				p1 = py::cast<double>(dict["mvir"]);
				p2 = py::cast<double>(dict["rs_kpc"]);
			} else throw std::runtime_error("Can only choose pmode=0, 1, or 2");
			LensProfile::extract_geometric_params_from_map(q1,q2,xc,yc,py::cast<std::map<std::string,double>>(dict));

			if (cosmo_in==NULL) throw std::runtime_error("NFW requires cosmology object to be passed in when initializing");

			NFW* nfw = new NFW(zlens,zsrc_ref,p1,p2,q1,q2,xc,yc,pmode,cosmo_in);
			if (use_median_c) {
				nfw->assign_special_anchored_parameters(nfw,c_median_factor,true);
				if (!anchor_median_c) nfw->unassign_special_anchored_parameter();
			}
			if (qlens_ptr != NULL) qlens_ptr->add_lens(nfw);
			if (vary_list.size() > 0) {
				if (nfw->set_vary_flags(vary_list)==false) {
					throw std::runtime_error("Number of input vary flags does not match number of lens parameters");
				}
			}
			return nfw;
		}))
		;

	/*
	py::class_<Cored_NFW, LensProfile, std::unique_ptr<Cored_NFW, py::nodelete>>(m, "Cored_NFW")
		.def(py::init<>([](){return new Cored_NFW();}))
		.def(py::init<const Cored_NFW*>())
		;

	py::class_<Hernquist, LensProfile, std::unique_ptr<Hernquist, py::nodelete>>(m, "Hernquist")
		.def(py::init<>([](){return new Hernquist();}))
		.def(py::init<const Hernquist*>())
		;

	py::class_<ExpDisk, LensProfile, std::unique_ptr<ExpDisk, py::nodelete>>(m, "ExpDisk")
		.def(py::init<>([](){return new ExpDisk();}))
		.def(py::init<const ExpDisk*>())
		;

	py::class_<Multipole, LensProfile, std::unique_ptr<Multipole, py::nodelete>>(m, "Multipole")
		.def(py::init<>([](){return new Multipole();}))
		.def(py::init<const Multipole*>())
		;

	py::class_<PointMass, LensProfile, std::unique_ptr<PointMass, py::nodelete>>(m, "PointMass")
		.def(py::init<>([](){return new PointMass();}))
		.def(py::init<const PointMass*>())
		;

	py::class_<CoreCusp, LensProfile, std::unique_ptr<CoreCusp, py::nodelete>>(m, "CoreCusp")
		.def(py::init<>([](){return new CoreCusp();}))
		.def(py::init<const CoreCusp*>())
		;
	*/

	py::class_<SersicLens, LensProfile, std::unique_ptr<SersicLens, py::nodelete>>(m, "SersicLens")
		//.def(py::init<>([](){return new SersicLens();}))
		.def(py::init<const SersicLens*>())
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int pmode=0;
			Cosmology* cosmo_in = NULL;
			QLens_Wrap* qlens_ptr = NULL;
			double zlens = default_zlens;
			double zsrc_ref = default_zsrc_ref;
			boolvector vary_list;
			process_init_lens_kwargs(pmode, cosmo_in, qlens_ptr, zlens, zsrc_ref, vary_list, kwargs);

			double p1,p2,p3,q1,q2,xc,yc;
			if (pmode==0) {
				p1 = py::cast<double>(dict["kappa_e"]);
			} else if (pmode==1) {
				p1 = py::cast<double>(dict["Mstar"]);
			} else throw std::runtime_error("Can only choose pmode=0 or 1");
			p2 = py::cast<double>(dict["R_eff"]);
			p3 = py::cast<double>(dict["n"]);
			LensProfile::extract_geometric_params_from_map(q1,q2,xc,yc,py::cast<std::map<std::string,double>>(dict));

			SersicLens* sersic = new SersicLens(zlens,zsrc_ref,p1,p2,p3,q1,q2,xc,yc,pmode,cosmo_in);
			if (qlens_ptr != NULL) qlens_ptr->add_lens(sersic);
			if (vary_list.size() > 0) {
				if (sersic->set_vary_flags(vary_list)==false) {
					throw std::runtime_error("Number of input vary flags does not match number of lens parameters");
				}
			}
			return sersic;
		}))
		;

	/*
	py::class_<Cored_SersicLens, LensProfile, std::unique_ptr<Cored_SersicLens, py::nodelete>>(m, "Cored_SersicLens")
		.def(py::init<>([](){return new Cored_SersicLens();}))
		.def(py::init<const Cored_SersicLens*>())
		;

	py::class_<MassSheet, LensProfile, std::unique_ptr<MassSheet, py::nodelete>>(m, "MassSheet")
		.def(py::init<>([](){return new MassSheet();}))
		.def(py::init<const MassSheet*>())
		;

	py::class_<Deflection, LensProfile, std::unique_ptr<Deflection, py::nodelete>>(m, "Deflection")
		.def(py::init<>([](){return new Deflection();}))
		.def(py::init<const Deflection*>())
		;


	py::class_<Tabulated_Model, LensProfile, std::unique_ptr<Tabulated_Model, py::nodelete>>(m, "Tabulated_Model")
		.def(py::init<>([](){return new Tabulated_Model();}))
		.def(py::init<const Tabulated_Model*>())
		;

	py::class_<QTabulated_Model, LensProfile, std::unique_ptr<QTabulated_Model, py::nodelete>>(m, "QTabulated_Model")
		.def(py::init<>([](){return new QTabulated_Model();}))
		.def(py::init<const QTabulated_Model*>())
		;
	*/


	py::class_<SB_Profile>(m, "SrcProfile")
		.def(py::init<>([](){return new SB_Profile();}))
		.def(py::init<const SB_Profile*>())
		.def("update", [](SB_Profile &current, py::dict dict){
			for (auto item : dict) {
				if(!current.update_specific_parameter(py::cast<string>(item.first), py::cast<double>(item.second)))
					return false;
			}
			return true;
		})
		.def("update", [](SB_Profile &current, const string name, const double value){
			return current.update_specific_parameter(name, value);
		})
		.def("print_params", &SB_Profile::print_params)
		.def("print_vary_params", &SB_Profile::print_vary_parameters)
		.def("set_center", &SB_Profile::set_center)
		.def("get_model_name", &SB_Profile::get_model_name)
		.def("vary", [](SB_Profile &current, py::list list){ 
			std::vector<double> lst = py::cast<std::vector<double>>(list);
			boolvector val(lst.size());
			int iter = 0;
			for (auto item : list) {
					val[iter] = py::cast<bool>(item); iter++;
			}
			if (current.set_vary_flags(val)==false) {
				throw std::runtime_error("Number of input vary flags does not match number of lens parameters");
			}
		})
		.def("set_limits", [](SB_Profile &curr, const std::string &param, const double lower, const double upper){
			if (curr.set_limits_specific_parameter(param,lower,upper)==false) {
				throw std::runtime_error("could not set limits for given parameter " + param);
			}
		})
		.def("set_limits", [](SB_Profile &curr, py::list list){
			std::string paramname;
			double lower, upper;
			for (auto arr : list){
				try {
					std::tuple<std::string, double, double> extracted = py::cast<std::tuple<std::string, double, double>>(arr);
					paramname = std::get<0>(extracted);
					lower = std::get<1>(extracted);
					upper = std::get<2>(extracted);
				} catch (...) {
					throw std::runtime_error("Error setting parameter limits. Input should be an array of tuples. Ex: [(<paramname>, lower, upper), (<paramname>, lower, upper)]");
				}
				if (curr.set_limits_specific_parameter(paramname,lower,upper)==false) {
					throw std::runtime_error("could not set limits for given parameter " + paramname);
				}
			}
		})
		.def("sb", &SB_Profile::surface_brightness)
		.def("anchor_param", [](SB_Profile &current, const string name, SB_Profile* param_anchor_source, const string anchor_param_name){
			int paramnum = -1;
			int anchor_paramnum = -1;
			if (!current.lookup_parameter_number(name,paramnum)) throw std::runtime_error("could not find parameter '" + name +"'");
			if (current.get_vary_flag(paramnum)==true) throw std::runtime_error("cannot anchor parameter if its vary flag is set to 'True'");
			if (!param_anchor_source->lookup_parameter_number(anchor_param_name,anchor_paramnum)) throw std::runtime_error("could not find parameter '" + anchor_param_name +"'");
			current.assign_anchored_parameter(paramnum,anchor_paramnum,false,false,1.0,1.0,param_anchor_source);
		})
		//.def("anchor_center",&SB_Profile::anchor_center_to_source)
		.def("__repr__", [](SB_Profile &a) {
			string outstring = a.get_parameters_string();
			return("\n" + outstring);
		})
		;

	py::class_<Gaussian, SB_Profile, std::unique_ptr<Gaussian, py::nodelete>>(m, "Gaussian")
		.def(py::init<>([](){return new Gaussian();}))
		.def(py::init<const Gaussian*>())
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int band = 0;
			double zsrc = default_zsrc;
			if (kwargs) {
				for (auto item : kwargs) {
					if (py::cast<string>(item.first)=="band") {
						band = py::cast<int>(item.second);
					} else if (py::cast<string>(item.first)=="z") {
						zsrc = py::cast<double>(item.second);
					} else {
						throw std::runtime_error("unknown argument to Gaussian");
					}
				}
			}
	
			//if (kwargs) {
				//throw std::runtime_error("unknown argument to Gaussian"); // currently no kwargs arguments available for Gaussian
			//}

			double p1,p2,q1,q2,xc,yc;
			p1 = py::cast<double>(dict["sbmax"]);
			p2 = py::cast<double>(dict["sigma"]);
			SB_Profile::extract_geometric_params_from_map(q1,q2,xc,yc,py::cast<std::map<std::string,double>>(dict));
			return new Gaussian(band,zsrc,p1,p2,q1,q2,xc,yc,NULL);
		}))
		;



	py::class_<Sersic, SB_Profile, std::unique_ptr<Sersic, py::nodelete>>(m, "Sersic")
		.def(py::init<>([](){return new Sersic();}))
		.def(py::init<const Sersic*>())
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int band = 0;
			double zsrc = default_zsrc;
			int pmode=0;
			if (kwargs) {
				for (auto item : kwargs) {
					if (py::cast<string>(item.first)=="pmode") {
						pmode = py::cast<int>(item.second);
					} else if (py::cast<string>(item.first)=="band") {
						band = py::cast<int>(item.second);
					} else if (py::cast<string>(item.first)=="z") {
						zsrc = py::cast<double>(item.second);
					} else {
						throw std::runtime_error("unknown argument to Sersic");
					}
				}
			}

			double p1,p2,p3,q1,q2,xc,yc;
			if (pmode==0) {
				p1 = py::cast<double>(dict["s0"]);
			} else if (pmode==1) {
				p1 = py::cast<double>(dict["s_eff"]);
			} else throw std::runtime_error("Can only choose pmode=0 or 1");
			p2 = py::cast<double>(dict["R_eff"]);
			p3 = py::cast<double>(dict["n"]);
			SB_Profile::extract_geometric_params_from_map(q1,q2,xc,yc,py::cast<std::map<std::string,double>>(dict));

			return new Sersic(band,zsrc,p1,p2,p3,q1,q2,xc,yc,0,NULL);
		}))
		;

	py::class_<DelaunaySourceGrid, ModelParams, std::unique_ptr<DelaunaySourceGrid, py::nodelete>>(m, "DelaunaySrcGrid")
		.def(py::init<>([](QLens* qlens_in){return new DelaunaySourceGrid(qlens_in);}))
		.def(py::init<const DelaunaySourceGrid*>())
		;

	py::class_<QLens_Wrap>(m, "QLens")
		.def(py::init<>([](py::kwargs &kwargs){
			double zlens = default_zlens;
			double zsrc = default_zsrc;
			double zsrc_ref = default_zsrc_ref;
			bool set_zsrc = false;
			bool set_zsrc_ref = false;
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="zlens") {
					zlens = py::cast<double>(item.second);
				} else if (py::cast<string>(item.first)=="zsrc") {
					zsrc = py::cast<double>(item.second);
					set_zsrc = true;
				} else if (py::cast<string>(item.first)=="zsrc_ref") {
					zsrc_ref = py::cast<double>(item.second);
					set_zsrc_ref = true;
				} else throw std::runtime_error("unknown argument to Cosmo");
			}
			if ((set_zsrc) and (!set_zsrc_ref)) zsrc_ref = zsrc;
			return new QLens_Wrap(zlens,zsrc,zsrc_ref);
		}))
		.def(py::init<>([](Cosmology* cosmo_in, py::kwargs &kwargs){
			double zlens = default_zlens;
			double zsrc = default_zsrc;
			double zsrc_ref = default_zsrc_ref;
			bool set_zsrc = false;
			bool set_zsrc_ref = false;
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="zlens") {
					zlens = py::cast<double>(item.second);
				} else if (py::cast<string>(item.first)=="zsrc") {
					zsrc = py::cast<double>(item.second);
					set_zsrc = true;
				} else if (py::cast<string>(item.first)=="zsrc_ref") {
					zsrc_ref = py::cast<double>(item.second);
					set_zsrc_ref = true;
				} else throw std::runtime_error("unknown argument to Cosmo");
			}
			if ((set_zsrc) and (!set_zsrc_ref)) zsrc_ref = zsrc;

			return new QLens_Wrap(zlens,zsrc,zsrc_ref,cosmo_in);
		}))
		.def_readonly("cosmo", &QLens_Wrap::cosmo)
		.def("objects", [](QLens_Wrap &current){ 
			return std::make_tuple(current.lenslist, current.srclist, current.ptsrclist, current.pixsrclist, current.param_list, current.dparam_list);
		})

		.def("imgdata_display", &QLens_Wrap::imgdata_display)
		.def("imgdata_add", &QLens_Wrap::imgdata_add, 
				py::arg("x") = -1.0, py::arg("y") = -1.0)
		.def("imgdata_write", &QLens_Wrap::imgdata_write_file)
		.def("imgdata_clear", &QLens_Wrap::imgdata_clear, 
				py::arg("lower") = -1, py::arg("upper") = -1)
		.def("imgdata_read", &QLens_Wrap::imgdata_load_file)
		.def("sbmap_loadimg", &QLens_Wrap::sbmap_load_image_file)
		.def("sbmap_load_noisemap", &QLens_Wrap::sbmap_load_noise_map)
		.def("sbmap_load_psf", &QLens_Wrap::sbmap_load_psf)
		.def("sbmap_load_mask", &QLens_Wrap::sbmap_load_mask)
		.def_readwrite("outside_sb_prior", &QLens_Wrap::outside_sb_prior)
		.def_readwrite("outside_sb_frac_threshold", &QLens_Wrap::outside_sb_prior_threshold)
		.def_readwrite("outside_sb_noise_threshold", &QLens_Wrap::outside_sb_prior_noise_frac)
		.def_readwrite("nimg_prior", &QLens_Wrap::n_image_prior)
		.def_readwrite("nimg_threshold", &QLens_Wrap::n_image_threshold)

		.def("clear_lenses", &QLens_Wrap::lens_clear, py::arg("min_loc") = -1, py::arg("max_loc") = -1)
		.def_property("shear_components", &QLens_Wrap::get_shear_components_mode, &QLens_Wrap::set_shear_components_mode)
		.def_property("ellipticity_components", &QLens_Wrap::get_ellipticity_components_mode, &QLens_Wrap::set_ellipticity_components_mode)
		.def_property("split_imgpixels", &QLens_Wrap::get_split_imgpixels, &QLens_Wrap::set_split_imgpixels)
		.def("lens_list", &QLens_Wrap::lens_display)
		.def("src_list", &QLens_Wrap::src_display)
		.def("pixsrc_list", &QLens_Wrap::pixsrc_display)
		//.def("pixsrc_clear", &QLens_Wrap::pixsrc_clear, py::arg("min_loc") = -1, py::arg("max_loc") = -1)
		//.def("remove_pixsrc", &QLens_Wrap::remove_pixellated_source)
		//.def("lens", &QLens_Wrap::get_lens_pointer)
		.def_readonly("lens", &QLens_Wrap::lenslist)
		.def_readonly("src", &QLens_Wrap::srclist)
		.def_readonly("pixsrc", &QLens_Wrap::pixsrclist)
		.def_readonly("ptsrc", &QLens_Wrap::ptsrclist)
		.def("add_lens", &QLens_Wrap::add_lens_tuple, "Input should be a tuple that specifies the lens' zl and zs value. \nEx: (Lens1, zl1, zs1)")
		.def("add_lens", &QLens_Wrap::add_lens, "Input should be a lens object.")
		.def("add_lens_extshear", &QLens_Wrap::add_lens_extshear, "Input should be a	lens object, and a shear lens object to anchor to the original lens object.")
		//.def("add_lenses", &QLens_Wrap::batch_add_lenses_tuple, "Input should be an array of tuples. Each tuple must specify each lens' zl and zs values. \nEx: [(Lens1, zl1, zs1), (Lens2, zl2, zs2)]")
		.def("add_lenses", &QLens_Wrap::batch_add_lenses, "Input should be an array of lenses.")
		//.def("add_lenses", [](QLens_Wrap &self){
				//return "Pass in an array of lenses \n\tEx: [Lens1, Lens2, Lens3] \nor an array of tuples. Each tuple must contain the lens, the zl and zs values for each corresponding lens. \n\tEx: [(Lens1, zl1, zs1), (Lens2, zl2, zs2)]";
		//})
		//.def("remove_lens", &QLens_Wrap::remove_lens) // lens_clear is the better function to use
		//.def("clear_lenses", &QLens_Wrap::clear_lenses)

		.def("add_src", &QLens_Wrap::add_src_default, "Input should be a source object.")
		.def("add_sources", &QLens_Wrap::batch_add_sources, "Input should be an array of sources")
		//.def("add_sources", &QLens_Wrap::batch_add_sources_tuple, "Input should be an array of tuples. Each tuple must specify each source's zs values. \nEx: [(src1, zs1), (src2, zs2)]")
		.def("add_sources", [](QLens_Wrap &self){
				return "Pass in an array of sourcees \n\tEx: [src1, src2, src3] \nor an array of tuples. Each tuple must contain the source, the zs values for each corresponding source. \n\tEx: [(src1, zs1), (src2, zs2)]";
		})
		.def("src_clear", &QLens_Wrap::source_clear, py::arg("min_loc") = -1, py::arg("max_loc") = -1)

		.def("add_pixsrc", [](QLens_Wrap &current, py::kwargs &kwargs){ 
			int band=0;
			double zsrc = current.source_redshift;
			if (kwargs) {
				for (auto item : kwargs) {
					if (py::cast<string>(item.first)=="band") {
						band = py::cast<int>(item.second);
					} else if (py::cast<string>(item.first)=="z") {
						zsrc = py::cast<double>(item.second);
					} else {
						throw std::runtime_error("unknown argument; can specify either 'band' or 'z'");
					}
				}
			}
			current.add_pixellated_source(zsrc,band);
		})

		.def_property("regularization_method", &QLens_Wrap::get_regularization_method, &QLens_Wrap::set_regularization_method)

		.def("findimg", &QLens_Wrap::output_images_single_source,
				py::arg("x_source"), py::arg("y_source"), py::arg("verbal")=false,
				py::arg("flux")=-1, py::arg("show_labels")=false
				)
		// .def("get_imageset", &QLens_Wrap::get_imageset)
		.def("get_imageset", [](QLens_Wrap &curr, PointSource &imgset, double src_x=0.5, double src_y=0.1, bool verbal=false) {
				curr.get_imageset(src_x, src_y, imgset, verbal);
		},  py::arg("imgset"), py::arg("src_x") = 0.5, py::arg("src_y") = 0.1, py::arg("verbal")=false)		
		.def("get_fit_imagesets", &QLens_Wrap::get_fit_imagesets, 
				py::arg("status") = false, py::arg("min_dataset") = 0, py::arg("max_dataset") = -1, 
				py::arg("verbal") = false) 
		.def("get_data_imagesets", &QLens_Wrap::export_to_ImageDataSet)
		.def("run_fit", [](QLens_Wrap &curr, const std::string &fitmethod, py::kwargs &kwargs){
			bool adopt_bestfit = false;
			bool show_errors = true;
			bool resume = false;
			for (auto item : kwargs) {
				if (py::cast<string>(item.first)=="adopt") {
					try {
						adopt_bestfit = py::cast<bool>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid argument for 'adopt'; must be true or false");
					}
				} else if (py::cast<string>(item.first)=="resume") {
					try {
						adopt_bestfit = py::cast<bool>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid argument for 'resume'; must be true or false");
					}
				} else if (py::cast<string>(item.first)=="show_errors") {
					try {
						adopt_bestfit = py::cast<bool>(item.second);
					} catch (...) {
						throw std::runtime_error("Invalid argument for 'show_errors'; must be true or false");
					}
				} else {
					throw std::runtime_error("Unknown keyword argument for 'run_fit'");
				}
			}

			if(fitmethod=="simplex") {
					curr.chi_square_fit_simplex(show_errors);
			} else if (fitmethod=="powell") {
					curr.chi_square_fit_powell(show_errors);
			} else if (fitmethod=="nest") {
					curr.nested_sampling();
			} else if (fitmethod=="multinest") {
					curr.multinest(resume,false);
			} else if (fitmethod=="polychord") {
					curr.polychord(resume,false);
			} else if (fitmethod=="twalk") {
					curr.chi_square_twalk();
			} else {
					throw std::runtime_error("Available fitmethods: simplex (default), powell, nest, multinest, polychord, twalk");
			}
		})
		.def("set_source_mode", [](QLens_Wrap &curr, const std::string &source_mode="ptsource"){
			if(source_mode=="ptsource") {
					curr.source_fit_mode = Point_Source;
			} else if (source_mode=="cartesian") {
					curr.source_fit_mode = Cartesian_Source;
			} else if (source_mode=="delaunay") {
					curr.source_fit_mode = Delaunay_Source;
			} else if (source_mode=="sbprofile") {
					curr.source_fit_mode = Parameterized_Source;
			} else if (source_mode=="shapelet") {
					curr.source_fit_mode = Shapelet_Source;
			} else {
					throw std::runtime_error("Available source_modes: ptsource, cartesian, delaunay, sbprofile, shapelet");
			}
		})
		.def("source_mode", [](QLens_Wrap &curr){
			if (curr.source_fit_mode == Point_Source) cout << "ptsource" << endl;
			else if (curr.source_fit_mode == Cartesian_Source) cout << "cartesian" << endl;
			else if (curr.source_fit_mode == Delaunay_Source) cout << "delaunay" << endl;
			else if (curr.source_fit_mode == Parameterized_Source) cout << "sbprofile" << endl;
			else if (curr.source_fit_mode == Shapelet_Source) cout << "shapelet" << endl;
			else throw std::runtime_error("Source_modes not recognized");
		})
		//.def("use_bestfit", &QLens_Wrap::use_bestfit)
		.def("use_bestfit", [](QLens_Wrap &curr){
			curr.adopt_model(curr.bestfitparams);
		})
		.def("test_lens", &QLens_Wrap::test_lens_functions)
		.def("sort_critical_curves", &QLens_Wrap::sort_critical_curves)
		.def("init_fitmodel", &QLens_Wrap::initialize_fitmodel, py::arg("run_fit_in") = true)
		.def_readonly("params", &QLens_Wrap::param_list)
		.def_readonly("dparams", &QLens_Wrap::dparam_list)
		.def("adopt_model", [](QLens_Wrap &current, py::list param_list){ 
				std::vector<double> plst = py::cast<std::vector<double>>(param_list);
				dvector param_vec(plst.size());
				int iter = 0;
				for (auto item : param_list) {
						param_vec[iter] = py::cast<double>(item); iter++;
								//cout << "pvec " << iter << " = " << param_vec[iter] << endl;
				}
					return current.adopt_model(param_vec);
		})

		.def("fit_chisq",&QLens_Wrap::chisq_single_evaluation, py::arg("init_fitmodel") = false, py::arg("show_total_wtime") = false, py::arg("showdiag") = false, py::arg("show_status") = true, py::arg("show_lensmodel") = false)
		.def("LogLike", &QLens_Wrap::LogLikeListFunc)
		.def("sbmap_invert", [](QLens_Wrap &current){
			double chisq0;
			bool verbal = true;
				double chisq = current.invert_surface_brightness_map_from_data(chisq0, verbal);
		})
		.def_property("optimize_regparam", &QLens_Wrap::get_optimize_regparam, &QLens_Wrap::set_optimize_regparam)
		.def("set_sourcepts_auto",&QLens_Wrap::set_analytic_sourcepts, py::arg("verbal") = true)
		.def("fitmodel", &QLens_Wrap::print_fit_model)
		.def_readonly("sorted_critical_curve", &QLens_Wrap::sorted_critical_curve)
		.def_readonly("nlens", &QLens_Wrap::nlens)
		.def_readwrite("default_pixsize", &QLens_Wrap::default_data_pixel_size)
		.def_readwrite("psf_threshold", &QLens_Wrap::psf_threshold)
		.def_property("zsrc", &QLens_Wrap::get_source_redshift, &QLens_Wrap::set_source_redshift)
		.def_property("zsrc_ref", &QLens_Wrap::get_reference_source_redshift, &QLens_Wrap::set_reference_source_redshift)
		.def_property("analytic_bestfit_src", &QLens_Wrap::get_analytic_bestfit_src, &QLens_Wrap::set_analytic_bestfit_src)
		.def_readwrite("cc_splitlevels", &QLens_Wrap::cc_splitlevels)
		.def_readwrite("zlens", &QLens_Wrap::lens_redshift)
		.def_readwrite("imgplane_chisq", &QLens_Wrap::imgplane_chisq)
		.def_readwrite("nrepeat", &QLens_Wrap::n_repeats)
		.def_readwrite("flux_chisq", &QLens_Wrap::include_flux_chisq)
		.def_readwrite("chisqtol", &QLens_Wrap::chisq_tolerance)
		.def_readwrite("central_image", &QLens_Wrap::include_central_image)
		//.def_readwrite("sourcepts_fit", &QLens_Wrap::sourcepts_fit)
		.def_readwrite("n_livepts", &QLens_Wrap::n_livepts)
		.def_property("sci_notation", &QLens_Wrap::get_sci_notation, &QLens_Wrap::set_sci_notation)
		.def_property("fit_label", &QLens_Wrap::get_fit_label, &QLens_Wrap::set_fit_label)
		.def_readwrite("fit_output_dir", &QLens_Wrap::fit_output_dir)
		.def_readwrite_static("ansi_output", &QLens_Wrap::use_ansi_output_during_fit)
		;

	py::class_<QLens_Wrap::critical_curve>(m, "critical_curve")
		.def_readonly("cc_pts", &QLens_Wrap::critical_curve::cc_pts)
		.def_readonly("caustic_pts", &QLens_Wrap::critical_curve::caustic_pts)
		.def_readonly("length_of_cell", &QLens_Wrap::critical_curve::length_of_cell);
		
	py::class_<image>(m, "image")
		.def_readonly("pos", &image::pos)
		.def_readonly("mag", &image::mag)
		.def_readonly("td", &image::td)
		.def_readonly("parity", &image::parity)
		;
	
	py::class_<image_data>(m, "image_data")
		.def_readonly("pos", &image_data::pos)
		.def_readonly("flux", &image_data::flux)
		.def_readonly("td", &image_data::td)
		.def_readonly("sigma_pos", &image_data::sigma_pos)
		.def_readonly("sigma_flux", &image_data::sigma_flux)
		.def_readonly("sigma_td", &image_data::sigma_td)
		;
 
	py::class_<lensvector>(m, "lensvector")
		.def(py::init([](){ return new lensvector(); }))
		.def_property("x", &lensvector::xval, &lensvector::set_xval)
		.def_property("y", &lensvector::yval, &lensvector::set_yval)
		.def("pos", [](lensvector &lens){ return std::make_tuple(lens.v[0], lens.v[1]); })
		;

	py::class_<PointSource, ModelParams, std::unique_ptr<PointSource, py::nodelete>>(m, "PtSrc")
		.def(py::init<>([](QLens* qlens_in){ return new PointSource(qlens_in); }))
		// .def()
		// .def("print", &PointSource::print)
		//.def("print", [](&PointSource curr, bool include_time_delays = false, bool show_labels = true, ofstream* srcfile = NULL, ofstream* imgfile = NULL){
				//curr.print(include_time_delays, show_labels, srcfile, imgfile);
		//}, 
		.def("print", &PointSource::print,
				py::arg("include_time_delays") = false, py::arg("include_time_delays") = true)
		.def_readonly("n_images", &PointSource::n_images)
		.def_readonly("zsrc", &PointSource::zsrc)
		.def_readonly("srcflux", &PointSource::srcflux)
		.def_readonly("pos", &PointSource::pos)
		.def_readonly("images", &PointSource::images)
		;

	py::class_<ImageDataSet>(m, "ImageDataSet")
		.def(py::init<>([](){ return new ImageDataSet(); }))
		// .def()
		// .def("print", &PointSource::print)
		// .def("print_s", [](&PointSource curr, bool include_time_delays = false, bool show_labels = true, ofstream* srcfile = NULL, ofstream* imgfile = NULL){
		//		curr.print(include_time_delays, show_labels, srcfile, imgfile);
		// }, 
		//		py::arg("include_time_delays") = false, py::arg("include_time_delays") = true,
		//		py::arg("srcfile") = NULL, py::arg("imgfile") = NULL)
		.def_readonly("n_images", &ImageDataSet::n_images)
		.def_readonly("zsrc", &ImageDataSet::zsrc)
		.def_readonly("images", &ImageDataSet::images)
		;

}
