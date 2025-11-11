#include <pybind11/pybind11.h>
#include "qlens_wrapper.cpp"
#include "profile.h"
#include "cosmo.h"
#include "qlens.h"
#include <vector>

namespace py = pybind11;

double default_zlens = 0.5;
double default_zsrc = 2;
double default_zsrc_ref = 2;

void process_cosmology_and_pmode_kwargs(int& pmode, Cosmology*& cosmo, double& z, double& zs, py::kwargs& kwargs)
{
	bool set_pmode=false, set_cosmo=false, set_z=false, set_zs=false;
	if (kwargs) {
		for (auto item : kwargs) {
			if (py::cast<string>(item.first)=="pmode") {
				pmode = py::cast<int>(item.second);
				set_pmode = true;
			} else if (py::cast<string>(item.first)=="cosmology") {
				cosmo = py::cast<Cosmology*>(item.second);
				set_cosmo = true;
			} else if (py::cast<string>(item.first)=="z") {
				z = py::cast<double>(item.second);
				set_z = true;
			} else if (py::cast<string>(item.first)=="zs") {
				zs = py::cast<double>(item.second);
				set_zs = true;
			}
		}
		if (set_pmode) kwargs.attr("pop")("pmode");
		if (set_cosmo) kwargs.attr("pop")("cosmology");
		if (set_z) kwargs.attr("pop")("z");
		if (set_zs) kwargs.attr("pop")("zs");
	}
}	

PYBIND11_MODULE(qlens, m) {
	m.doc() = "QLens Python Plugin"; // optional module docstring

	//py::class_<QLens, std::unique_ptr<QLens, py::nodelete>>(m, "QLens_base")
		//.def(py::init<>([](){return new QLens();}))
	//;
	py::class_<Cosmology, std::unique_ptr<Cosmology, py::nodelete>>(m, "Cosmology")
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
	;

	py::class_<LensProfile>(m, "LensProfile")
		.def(py::init<>([](){return new LensProfile();}))
		.def(py::init<const LensProfile*>())
		// .def(py::init<>([](const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int& nn, const double &acc, const double &qx_in, const double &f_in, Lens* lens_in){
		//		return new LensProfile(splinefile, zlens_in, zsrc_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, qx_in, f_in, lens_in);
		// }))
		.def("update", [](LensProfile &current, py::dict dict){
			for (auto item : dict) {
				if(!current.update_specific_parameter(py::cast<string>(item.first), py::cast<double>(item.second)))
					return false;
			}
			return true;
		})
		.def("print_params", &LensProfile::print_parameters)
		.def("print_vary_params", &LensProfile::print_vary_parameters)
		.def("set_center", &LensProfile::set_center)
		//.def("extract_geometric_params", [](double& q1, double& q2, double& xcp, double& ycp, py::dict dict){ 
				//if (!LensProfile::use_ellipticity_components) {
					//q1 = py::cast<double>(dict["q"]);
				//try {
					//q2 = py::cast<double>(dict["theta"]);
				//} catch (...) {
					//q2 = 0.0;
				//}
				//} else {
					//q1 = py::cast<double>(dict["e1"]);
					//q2 = py::cast<double>(dict["e2"]);
				//}
			//try {
							//xcp = py::cast<double>(dict["xc"]);
			//} catch (...) {
				//xcp = 0.0;
			//}
			//try {
							//ycp = py::cast<double>(dict["yc"]);
			//} catch (...) {
				//ycp = 0.0;
			//}
		//})
		.def("get_model_name", &LensProfile::get_model_name)
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
		.def("set_prior_limits", [](LensProfile &curr, const std::string &param, const double lower, const double upper){
			if (curr.set_limits_specific_parameter(param,lower,upper)==false) {
				throw std::runtime_error("could not set limits for given parameter " + param);
			}
		})
		.def("set_prior_limits", [](LensProfile &curr, py::list list){
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
		.def("anchor_center",&LensProfile::anchor_center_to_lens)
			.def("__repr__", [](LensProfile &a) {
					string outstring = a.get_parameters_string();
					return("\n" + outstring);
				})
		;

	py::class_<SPLE_Lens, LensProfile, std::unique_ptr<SPLE_Lens, py::nodelete>>(m, "SPLE")
		.def(py::init<>([](){return new SPLE_Lens();}))
		.def(py::init<const SPLE_Lens*>())
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int pmode=0;
			Cosmology* cosmo_in = NULL;
			double zlens = default_zlens;
			double zsrc_ref = default_zsrc_ref;

			process_cosmology_and_pmode_kwargs(pmode, cosmo_in, zlens, zsrc_ref, kwargs);

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
			return new SPLE_Lens(b,p2,s,q1,q2,xc,yc,pmode);
		}))
		//.def("initialize", [](SPLE_Lens &current, py::dict dict){
				// do you really need initialize? Should just require initialization when creating object
		//})
		;

	py::class_<Shear, LensProfile, std::unique_ptr<Shear, py::nodelete>>(m, "Shear")
		.def(py::init<>([](){return new Shear();}))
		.def(py::init<const Shear*>())
		.def(py::init([](py::dict dict) {
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
			return new Shear(p1,p2,xc,yc);
		}))
		;

	py::class_<dPIE_Lens, LensProfile, std::unique_ptr<dPIE_Lens, py::nodelete>>(m, "dPIE")
		.def(py::init<>([](){return new dPIE_Lens();}))
		.def(py::init<const dPIE_Lens*>())
		;

	py::class_<NFW, LensProfile, std::unique_ptr<NFW, py::nodelete>>(m, "NFW")
		.def(py::init<>([](){return new NFW();}))
		.def(py::init<const NFW*>())
		.def(py::init([](py::dict dict, py::kwargs& kwargs) {
			int pmode=0;
			Cosmology* cosmo_in = NULL;
			double zlens = default_zlens;
			double zsrc_ref = default_zsrc_ref;
			bool use_median_c = false;
			bool anchor_median_c = true; // keep c set to median if use_median_c is turned on
			double c_median_factor = 1.0;
			if (kwargs) {
				process_cosmology_and_pmode_kwargs(pmode, cosmo_in, zlens, zsrc_ref, kwargs);
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
			return nfw;
		}))
		;

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

	py::class_<SersicLens, LensProfile, std::unique_ptr<SersicLens, py::nodelete>>(m, "SersicLens")
		.def(py::init<>([](){return new SersicLens();}))
		.def(py::init<const SersicLens*>())
		;


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

		.def("lens_clear", &QLens_Wrap::lens_clear,
				py::arg("min_loc") = -1, py::arg("max_loc") = -1)
		.def_property("shear_components", &QLens_Wrap::get_shear_components_mode, &QLens_Wrap::set_shear_components_mode)
		.def_property("ellipticity_components", &QLens_Wrap::get_ellipticity_components_mode, &QLens_Wrap::set_ellipticity_components_mode)
		.def_property("split_imgpixels", &QLens_Wrap::get_split_imgpixels, &QLens_Wrap::set_split_imgpixels)
		.def("lens_list", &QLens_Wrap::lens_display)
		//.def("lens", &QLens_Wrap::get_lens_pointer)
		.def_readonly("lens", &QLens_Wrap::lens_list_vec)
		.def("add_lens", &QLens_Wrap::add_lens_tuple, "Input should be a tuple that specifies the lens' zl and zs value. \nEx: (Lens1, zl1, zs1)")
		.def("add_lens", &QLens_Wrap::add_lens_default, "Input should be a lens object.")
		.def("add_lens_extshear", &QLens_Wrap::add_lens_extshear, "Input should be a	lens object, and a shear lens object to anchor to the original lens object.")
		.def("add_lenses", &QLens_Wrap::batch_add_lenses_tuple, "Input should be an array of tuples. Each tuple must specify each lens' zl and zs values. \nEx: [(Lens1, zl1, zs1), (Lens2, zl2, zs2)]")
		.def("add_lenses", [](QLens_Wrap &self){
				return "Pass in an array of lenses \n\tEx: [Lens1, Lens2, Lens3] \nor an array of tuples. Each tuple must contain the lens, the zl and zs values for each corresponding lens. \n\tEx: [(Lens1, zl1, zs1), (Lens2, zl2, zs2)]";
		})
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
		.def("run_fit", [](QLens_Wrap &curr, const std::string &fitmethod="simplex"){
				if(fitmethod=="simplex") {
						curr.chi_square_fit_simplex();
				} else if (fitmethod=="powell") {
						curr.chi_square_fit_powell();
				} else if (fitmethod=="nest") {
						curr.nested_sampling();
				} else if (fitmethod=="multinest") {
						curr.multinest(false,false);
				} else if (fitmethod=="polychord") {
						curr.polychord(true,false);
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
		//.def("use_bestfit", &QLens_Wrap::use_bestfit)
		.def("use_bestfit", [](QLens_Wrap &curr){
			curr.adopt_model(curr.bestfitparams);
		})
		.def("test_lens", &QLens_Wrap::test_lens_functions)
		.def("sort_critical_curves", &QLens_Wrap::sort_critical_curves)
		.def("setup_fitparams", &QLens_Wrap::setup_fit_parameters)
		.def("init_fitmodel", &QLens_Wrap::initialize_fitmodel, py::arg("run_fit_in") = true)
		.def("fitparams", [](QLens_Wrap &current){ 
			vector<double> fitparams_vec;
			for (int i=0; i < current.fitparams.size(); i++) fitparams_vec.push_back(current.fitparams[i]);
			py::list py_fitparams = py::cast(fitparams_vec);
			return py_fitparams;
		})
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

	py::class_<PointSource>(m, "PointSource")
		.def(py::init<>([](){ return new PointSource(); }))
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
