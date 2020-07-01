#include <pybind11/pybind11.h>
#include "qlens_wrapper.cpp"
#include "profile.h"
#include "qlens.h"

namespace py = pybind11;

PYBIND11_MODULE(qlens, m) {
    m.doc() = "QLens Python Plugin"; // optional module docstring

    py::class_<LensProfile>(m, "LensProfile")
        .def(py::init<>())
        .def_property("theta", &LensProfile::get_theta, &LensProfile::set_theta)
        .def("print_parameters", &LensProfile::print_parameters)
        .def("set_center", &LensProfile::set_center)
        .def("get_model_name", &LensProfile::get_model_name);

    py::class_<Alpha, LensProfile>(m, "Alpha")
        .def(py::init<>())
        .def(py::init<const Alpha*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &b_in, const double &alpha_in, const double &s_in, const double &q_in, const double &theta_degrees,
			const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens* lens_in) {
                                return new Alpha(zlens_in, zsrc_in, b_in, alpha_in, s_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, lens_in);
                        }));

    py::class_<Shear>(m, "Shear")
        .def(py::init<>())
        .def(py::init<const Shear*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &shear_in, const double &theta_degrees, const double &xc_in, const double &yc_in, Lens* lens) {
                return new Shear(zlens_in, zsrc_in, shear_in, theta_degrees, xc_in, yc_in, lens);
        }));

    py::class_<PseudoJaffe>(m, "PseudoJaffe")
        .def(py::init<>())
        .def(py::init<const PseudoJaffe*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &b_in, const double &a_in, const double &s_in, const double &q_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode, Lens* lens_in) {
                        return new PseudoJaffe(zlens_in, zsrc_in, b_in, a_in, s_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode, lens_in);
        }));

    py::class_<NFW>(m, "NFW")
        .def(py::init<>())
        .def(py::init<const NFW*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new NFW(zlens_in, zsrc_in, ks_in, rs_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode_in, lens_in);
        }));

    py::class_<Cored_NFW>(m, "Cored_NFW")
        .def(py::init<>())
        .def(py::init<const Cored_NFW*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &rt_in, const double &q_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new Cored_NFW(zlens_in, zsrc_in, ks_in, rs_in, rt_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode_in, lens_in);
        }));

    py::class_<Hernquist>(m, "Hernquist")
        .def(py::init<>())
        .def(py::init<const Hernquist*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &ks_in, const double &rs_in, const double &q_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens* lens_in) {
                        return new Hernquist(zlens_in, zsrc_in, ks_in, rs_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, lens_in);
    }));

    py::class_<ExpDisk>(m, "ExpDisk")
        .def(py::init<>())
        .def(py::init<const ExpDisk*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &k0_in, const double &R_d_in, const double &q_in, 
                const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, Lens* lens_in) {
                        return new ExpDisk(zlens_in, zsrc_in, k0_in, R_d_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, lens_in);
    }));

    py::class_<Multipole>(m, "Multipole")
        .def(py::init<>())
        .def(py::init<const Multipole*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &A_m_in, const double n_in, const int m_in, const double &theta_degrees, 
                const double &xc_in, const double &yc_in, const bool kap, Lens* lens_in, const bool sine=false) {
                        return new Multipole(zlens_in, zsrc_in, A_m_in, n_in, m_in, theta_degrees, xc_in, yc_in, kap, lens_in, sine);
    }));

    py::class_<PointMass>(m, "PointMass")
        .def(py::init<>())
        .def(py::init<const PointMass*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &bb, const double &xc_in, const double &yc_in, const int parameter_mode_in, Lens* lens_in) {
                        return new PointMass(zlens_in, zsrc_in, bb, xc_in, yc_in, parameter_mode_in, lens_in);
    }));

    py::class_<CoreCusp>(m, "CoreCusp")
        .def(py::init<>())
        .def(py::init<const CoreCusp*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &k0_in, const double &gamma_in, const double &n_in, const double &a_in, 
                const double &s_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new CoreCusp(zlens_in, zsrc_in, k0_in, gamma_in, n_in, a_in, s_in, q_in, theta_degrees, xc_in, yc_in, nn, acc,  parameter_mode_in, lens_in);
    }));

    py::class_<SersicLens>(m, "SersicLens")
        .def(py::init<>())
        .def(py::init<const SersicLens*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &q_in, 
                const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new SersicLens(zlens_in, zsrc_in, kappa0_in, k_in, n_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode_in, lens_in);
    }));


    py::class_<Cored_SersicLens>(m, "Cored_SersicLens")
        .def(py::init<>())
        .def(py::init<const Cored_SersicLens*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kappa0_in, const double &k_in, const double &n_in, const double &rc_in, const double &q_in, 
                const double &theta_degrees, const double &xc_in, const double &yc_in, const int &nn, const double &acc, const int parameter_mode_in, Lens* lens_in) {
                        return new Cored_SersicLens(zlens_in, zsrc_in, kappa0_in, k_in, n_in, rc_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, parameter_mode_in, lens_in);
    }));

    py::class_<MassSheet>(m, "MassSheet")
        .def(py::init<>())
        .def(py::init<const MassSheet*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kext_in, const double &xc_in, const double &yc_in, Lens* lens_in) {
                        return new MassSheet(zlens_in, zsrc_in, kext_in, xc_in, yc_in, lens_in);
    }));

    py::class_<Deflection>(m, "Deflection")
        .def(py::init<>())
        .def(py::init<const Deflection*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &defx_in, const double &defy_in, Lens* lens_in) {
                        return new Deflection(zlens_in, zsrc_in, defx_in, defy_in, lens_in);
    }));


    py::class_<Tabulated_Model>(m, "Tabulated_Model")
        .def(py::init<>())
        .def(py::init<const Tabulated_Model*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double xc, 
                const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, Lens* lens) {
                        return new Tabulated_Model(zlens_in, zsrc_in, kscale_in, rscale_in, theta_in, xc, yc, lens_in, rmin, rmax, logr_N, phi_N, lens);}))
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &theta_in, const double &xc, 
                const double &yc, ifstream& tabfile, const string& tab_filename, Lens* lens_in) {
                        return new Tabulated_Model(zlens_in, zsrc_in, kscale_in, rscale_in, theta_in, xc, yc, tabfile, tab_filename, lens_in);
    }));

    py::class_<QTabulated_Model>(m, "QTabulated_Model")
        .def(py::init<>())
        .def(py::init<const QTabulated_Model*>())
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, 
                const double xc, const double yc, LensProfile* lens_in, const double rmin, const double rmax, const int logr_N, const int phi_N, const double qmin, const int q_N, Lens* lens) {
                        return new QTabulated_Model(zlens_in, zsrc_in, kscale_in, rscale_in, q_in, theta_in, xc, yc, lens_in, rmin, rmax, logr_N, phi_N, qmin, q_N, lens);}))
        .def(py::init([](const double zlens_in, const double zsrc_in, const double &kscale_in, const double &rscale_in, const double &q_in, const double &theta_in, 
                const double &xc, const double &yc, ifstream& tabfile, Lens* lens_in) {
                        return new QTabulated_Model(zlens_in, zsrc_in, kscale_in, rscale_in, q_in, theta_in, xc, yc, tabfile, lens_in);
    }));

    py::class_<Lens_Wrap>(m, "Lens")
        .def(py::init<>())
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
        .def("lens_add", &Lens_Wrap::lens_add,
                py::arg("b"), py::arg("alpha"), py::arg("s"),  
                py::arg("q"), py::arg("theta"), py::arg("xc"), py::arg("yc"),   
                py::arg("mode") = "alpha", py::arg("emode")=1,
                py::arg("zl_in") = 0, py::arg("reference_source_redshift") = 2)
        .def("lens_add_alpha", &Lens_Wrap::lens_add_alpha,
                py::arg("b"), py::arg("alpha"), py::arg("s"),  
                py::arg("q"), py::arg("theta") = 0.0, py::arg("xc")=0.0, py::arg("yc")=0.0,   
                py::arg("emode")=1.0, py::arg("reference_source_redshift")=2,
                py::arg("zl_in") = 0.0)
        .def("lens_add_shear", &Lens_Wrap::lens_add_shear,
                py::arg("shear"), py::arg("theta"), py::arg("xc"), py::arg("yc"),
                py::arg("zl_in")=0, py::arg("reference_source_redshift")=2)
        ;
}