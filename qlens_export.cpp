#include <pybind11/pybind11.h>
#include "qlens_wrapper.cpp"
#include "profile.h"
#include "qlens.h"

namespace py = pybind11;

PYBIND11_MODULE(qlens, m) {
    m.doc() = "QLens Python Plugin"; // optional module docstring

    py::class_<LensProfile>(m, "LensProfile")
        .def(py::init<>([](){return new LensProfile();}))
        .def(py::init<const LensProfile*>())
        .def(py::init<>([](const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int& nn, const double &acc, const double &qx_in, const double &f_in, Lens* lens_in){
                return new LensProfile(splinefile, zlens_in, zsrc_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, qx_in, f_in, lens_in);
        }))
        .def("update", [](LensProfile &current, py::dict dict){
                for(auto item : dict) {
                        if(!current.update_specific_parameter(py::cast<string>(item.first), py::cast<double>(item.second)))
                                return false;
                }
                return true;
        })
        .def("print_parameters", &LensProfile::print_parameters)
        .def("print_vary_parameters", &LensProfile::print_vary_parameters)
        .def("set_center", &LensProfile::set_center)
        .def("get_model_name", &LensProfile::get_model_name);

    py::class_<Alpha, LensProfile, std::unique_ptr<Alpha, py::nodelete>>(m, "Alpha")
        .def(py::init<>([](){return new Alpha();}))
        .def(py::init<const Alpha*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &b_in, const double &alpha_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens* lens_in) {
                                return new Alpha(zlens_in, zsrc_in, b_in, alpha_in, s_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, lens_in);
                        }))
        .def("initialize", [](Alpha &current, py::dict dict){
                try {
                        double b = py::cast<double>(dict["b"]);
                        double alpha = py::cast<double>(dict["alpha"]);
                        double s = py::cast<double>(dict["s"]);
                        double q = py::cast<double>(dict["q"]);
                        double theta = py::cast<double>(dict["theta"]);
                        double xc = py::cast<double>(dict["xc"]);
                        double yc = py::cast<double>(dict["yc"]);
                        current.initialize_parameters(b, alpha, s, q, theta, xc, yc);
                } catch(...) {
                        throw std::runtime_error("Required parameters: b, alpha, s, q, theta, xc, yc.");
                }
                
        })
        .def("initialize", [](Alpha &current){
        })
        ;

    py::class_<Shear, LensProfile, std::unique_ptr<Shear, py::nodelete>>(m, "Shear")
        .def(py::init<>([](){return new Shear();}))
        .def(py::init<const Shear*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &shear_in, const double &theta_degrees, const double &xc_in, const double &yc_in, Lens* lens) {
                return new Shear(zlens_in, zsrc_in, shear_in, theta_degrees, xc_in, yc_in, lens);
        }))
        .def("initialize", [](Shear &current, py::dict dict){
                try {
                        double shear = py::cast<double>(dict["shear"]);
                        double theta = py::cast<double>(dict["theta"]);
                        double xc = py::cast<double>(dict["xc"]);
                        double yc = py::cast<double>(dict["yc"]);
                        current.initialize_parameters(shear, theta, xc, yc);
                } catch(...) {
                        throw std::runtime_error("Required parameters: shear, theta, xc, yc.");
                }
        })
        ;;

    py::class_<PseudoJaffe, LensProfile, std::unique_ptr<PseudoJaffe, py::nodelete>>(m, "PseudoJaffe")
        .def(py::init<>([](){return new PseudoJaffe();}))
        .def(py::init<const PseudoJaffe*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &b_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode, Lens* lens_in) {
                        return new PseudoJaffe(zlens_in, zsrc_in, b_in, a_in, s_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode, lens_in);
        }));

    py::class_<NFW, LensProfile, std::unique_ptr<NFW, py::nodelete>>(m, "NFW")
        .def(py::init<>([](){return new NFW();}))
        .def(py::init<const NFW*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new NFW(zlens_in, zsrc_in, ks_in, rs_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode_in, lens_in);
        }));

    py::class_<Cored_NFW, LensProfile, std::unique_ptr<Cored_NFW, py::nodelete>>(m, "Cored_NFW")
        .def(py::init<>([](){return new Cored_NFW();}))
        .def(py::init<const Cored_NFW*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &rt_in, const double &q_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new Cored_NFW(zlens_in, zsrc_in, ks_in, rs_in, rt_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode_in, lens_in);
        }));

    py::class_<Hernquist, LensProfile, std::unique_ptr<Hernquist, py::nodelete>>(m, "Hernquist")
        .def(py::init<>([](){return new Hernquist();}))
        .def(py::init<const Hernquist*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens* lens_in) {
                        return new Hernquist(zlens_in, zsrc_in, ks_in, rs_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, lens_in);
    }));

    py::class_<ExpDisk, LensProfile, std::unique_ptr<ExpDisk, py::nodelete>>(m, "ExpDisk")
        .def(py::init<>([](){return new ExpDisk();}))
        .def(py::init<const ExpDisk*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &k0_in, const double &R_d_in, const double &q_in, 
                const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens* lens_in) {
                        return new ExpDisk(zlens_in, zsrc_in, k0_in, R_d_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, lens_in);
    }));

    py::class_<Multipole, LensProfile, std::unique_ptr<Multipole, py::nodelete>>(m, "Multipole")
        .def(py::init<>([](){return new Multipole();}))
        .def(py::init<const Multipole*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const bool kap, Lens* lens_in, const bool sine=false) {
                        return new Multipole(zlens_in, zsrc_in, A_m_in, n_in, m_in, theta_degrees, xc_in, yc_in, kap, lens_in, sine);
    }));

    py::class_<PointMass, LensProfile, std::unique_ptr<PointMass, py::nodelete>>(m, "PointMass")
        .def(py::init<>([](){return new PointMass();}))
        .def(py::init<const PointMass*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &bb, const double &xc_in, const double &yc_in, const int parameter_mode_in, Lens* lens_in) {
                        return new PointMass(zlens_in, zsrc_in, bb, xc_in, yc_in, parameter_mode_in, lens_in);
    }));

    py::class_<CoreCusp, LensProfile, std::unique_ptr<CoreCusp, py::nodelete>>(m, "CoreCusp")
        .def(py::init<>([](){return new CoreCusp();}))
        .def(py::init<const CoreCusp*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &k0_in, const double &gamma_in, const double &n_in, const double &a_in, 
                const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new CoreCusp(zlens_in, zsrc_in, k0_in, gamma_in, n_in, a_in, s_in, q_in, theta_degrees, xc_in, yc_in, nn, acc,  parameter_mode_in, lens_in);
    }));

    py::class_<SersicLens, LensProfile, std::unique_ptr<SersicLens, py::nodelete>>(m, "SersicLens")
        .def(py::init<>([](){return new SersicLens();}))
        .def(py::init<const SersicLens*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &q_in, 
                const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new SersicLens(zlens_in, zsrc_in, kappa0_in, k_in, n_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode_in, lens_in);
    }));


    py::class_<Cored_SersicLens, LensProfile, std::unique_ptr<Cored_SersicLens, py::nodelete>>(m, "Cored_SersicLens")
        .def(py::init<>([](){return new Cored_SersicLens();}))
        .def(py::init<const Cored_SersicLens*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &rc_in, const double &q_in, 
                const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new Cored_SersicLens(zlens_in, zsrc_in, kappa0_in, k_in, n_in, rc_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode_in, lens_in);
    }));

    py::class_<MassSheet, LensProfile, std::unique_ptr<MassSheet, py::nodelete>>(m, "MassSheet")
        .def(py::init<>([](){return new MassSheet();}))
        .def(py::init<const MassSheet*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kext_in, const double &xc_in, const double &yc_in, Lens* lens_in) {
                        return new MassSheet(zlens_in, zsrc_in, kext_in, xc_in, yc_in, lens_in);
    }));

    py::class_<Deflection, LensProfile, std::unique_ptr<Deflection, py::nodelete>>(m, "Deflection")
        .def(py::init<>([](){return new Deflection();}))
        .def(py::init<const Deflection*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &defx_in, const double &defy_in, Lens* lens_in) {
                        return new Deflection(zlens_in, zsrc_in, defx_in, defy_in, lens_in);
    }));


    py::class_<Tabulated_Model, LensProfile, std::unique_ptr<Tabulated_Model, py::nodelete>>(m, "Tabulated_Model")
        .def(py::init<>([](){return new Tabulated_Model();}))
        .def(py::init<const Tabulated_Model*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double xc, 
                const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, Lens* lens) {
                        return new Tabulated_Model(zlens_in, zsrc_in, kscale_in, rscale_in, theta_in, xc, yc, lens_in, rmin, rmax, logr_N, phi_N, lens);}))
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double &xc, 
                const double &yc, ifstream& tabfile, const string& tab_filename, Lens* lens_in) {
                        return new Tabulated_Model(zlens_in, zsrc_in, kscale_in, rscale_in, theta_in, xc, yc, tabfile, tab_filename, lens_in);
    }));

    py::class_<QTabulated_Model, LensProfile, std::unique_ptr<QTabulated_Model, py::nodelete>>(m, "QTabulated_Model")
        .def(py::init<>([](){return new QTabulated_Model();}))
        .def(py::init<const QTabulated_Model*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, 
                const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, const double qmin, const int q_N, Lens* lens) {
                        return new QTabulated_Model(zlens_in, zsrc_in, kscale_in, rscale_in, q_in, theta_in, xc, yc, lens_in, rmin, rmax, logr_N, phi_N, qmin, q_N, lens);}))
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, 
                const double &xc, const double &yc, ifstream& tabfile, Lens* lens_in) {
                        return new QTabulated_Model(zlens_in, zsrc_in, kscale_in, rscale_in, q_in, theta_in, xc, yc, tabfile, lens_in);
    }));

    py::class_<Lens_Wrap>(m, "QLens")
        .def(py::init<>([](){return new Lens_Wrap();}))
        .def("imgdata_display", &Lens_Wrap::imgdata_display)
        .def("imgdata_add", &Lens_Wrap::imgdata_add, 
                py::arg("x") = -1.0, py::arg("y") = -1.0)
        .def("imgdata_write", &Lens_Wrap::imgdata_write_file)
        .def("imgdata_clear", &Lens_Wrap::imgdata_clear, 
                py::arg("lower") = -1, py::arg("upper") = -1)
        .def("imgdata_load", &Lens_Wrap::imgdata_load_file)
        .def("lens_clear", &Lens_Wrap::lens_clear,
                py::arg("min_loc") = -1, py::arg("max_loc") = -1)
        .def("lens_display", &Lens_Wrap::lens_display)
        .def("add_lenses", &Lens_Wrap::batch_add_lenses_tuple, "Input should be an array of tuples. Each tuple must specify each lens' zl and zs values. \nEx: [(Lens1, zl1, zs1), (Lens2, zl2, zs2)]")
        .def("findimg", &Lens_Wrap::output_images_single_source,
                py::arg("x_source"), py::arg("y_source"), py::arg("verbal")=false,
                py::arg("flux")=-1, py::arg("show_labels")=false
                )
        ;
}