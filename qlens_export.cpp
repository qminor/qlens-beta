#include <pybind11/pybind11.h>
#include "qlens_wrapper.cpp"
#include "profile.h"
#include "qlens.h"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(qlens, m) {
    m.doc() = "QLens Python Plugin"; // optional module docstring

    py::class_<LensProfile>(m, "LensProfile")
        .def(py::init<>([](){return new LensProfile();}))
        .def(py::init<const LensProfile*>())
        // .def(py::init<>([](const char *splinefile, const double zlens_in, const double zsrc_in, const double &q_in, const double &theta_degrees, const double &xc_in, const double &yc_in, const int& nn, const double &acc, const double &qx_in, const double &f_in, Lens* lens_in){
        //         return new LensProfile(splinefile, zlens_in, zsrc_in, q_in, theta_degrees, xc_in, yc_in, nn, acc, qx_in, f_in, lens_in);
        // }))
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
        .def("get_model_name", &LensProfile::get_model_name)
        .def("set_vary_flags", [](LensProfile &current, py::list list){ 
                std::vector<double> lst = py::cast<std::vector<double>>(list);
                boolvector val(lst.size());
                int iter = 0;
                for (auto item : list) {
                        val[iter] = py::cast<bool>(item); iter++;
                }
                current.set_vary_flags(val);
        })
        ;

    py::class_<Alpha, LensProfile, std::unique_ptr<Alpha, py::nodelete>>(m, "Alpha")
        .def(py::init<>([](){return new Alpha();}))
        .def(py::init<const Alpha*>())
        .def(py::init([](py::dict dict) {
                                return new Alpha(
                                        py::cast<double>(dict["b"]),
                                        py::cast<double>(dict["alpha"]),
                                        py::cast<double>(dict["s"]),
                                        py::cast<double>(dict["q"]),
                                        py::cast<double>(dict["theta"]),
                                        py::cast<double>(dict["xc"]),
                                        py::cast<double>(dict["yc"])
                                );
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
        ;

    py::class_<Shear, LensProfile, std::unique_ptr<Shear, py::nodelete>>(m, "Shear")
        .def(py::init<>([](){return new Shear();}))
        .def(py::init<const Shear*>())
        // .def(py::init([](const double zlens_in, const double zsrc_in, const double &shear_in, const double &theta_degrees, const double &xc_in, const double &yc_in, Lens* lens) {
        //         return new Shear(zlens_in, zsrc_in, shear_in, theta_degrees, xc_in, yc_in, lens);
        // }))
        .def(py::init([](py::dict dict) {
                return new Shear(
                        py::cast<double>(dict["shear"]),
                        py::cast<double>(dict["theta"]),
                        py::cast<double>(dict["xc"]),
                        py::cast<double>(dict["yc"])
                );
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
        ;

    py::class_<PseudoJaffe, LensProfile, std::unique_ptr<PseudoJaffe, py::nodelete>>(m, "PseudoJaffe")
        .def(py::init<>([](){return new PseudoJaffe();}))
        .def(py::init<const PseudoJaffe*>());

    py::class_<NFW, LensProfile, std::unique_ptr<NFW, py::nodelete>>(m, "NFW")
        .def(py::init<>([](){return new NFW();}))
        .def(py::init<const NFW*>())
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

    py::class_<Lens_Wrap>(m, "QLens")
        .def(py::init<>([](){return new Lens_Wrap();}))
        .def("imgdata_display", &Lens_Wrap::imgdata_display)
        .def("imgdata_add", &Lens_Wrap::imgdata_add, 
                py::arg("x") = -1.0, py::arg("y") = -1.0)
        .def("imgdata_write", &Lens_Wrap::imgdata_write_file)
        .def("imgdata_clear", &Lens_Wrap::imgdata_clear, 
                py::arg("lower") = -1, py::arg("upper") = -1)
        .def("imgdata_read", &Lens_Wrap::imgdata_load_file)
        .def("lens_clear", &Lens_Wrap::lens_clear,
                py::arg("min_loc") = -1, py::arg("max_loc") = -1)
        .def("lens_display", &Lens_Wrap::lens_display)
        .def("add_lenses", &Lens_Wrap::batch_add_lenses_tuple, "Input should be an array of tuples. Each tuple must specify each lens' zl and zs values. \nEx: [(Lens1, zl1, zs1), (Lens2, zl2, zs2)]")
        .def("add_lenses", [](Lens_Wrap &self){
                return "Pass in an array of lenses \n\tEx: [Lens1, Lens2, Lens3] \nor an array of tuples. Each tuple must contain the lens, the zl and zs values for each corresponding lens. \n\tEx: [(Lens1, zl1, zs1), (Lens2, zl2, zs2)]";
        })
        .def("findimg", &Lens_Wrap::output_images_single_source,
                py::arg("x_source"), py::arg("y_source"), py::arg("verbal")=false,
                py::arg("flux")=-1, py::arg("show_labels")=false
                )
        .def("get_imageset", &Lens_Wrap::get_imageset)
        .def("run_fit", [](Lens_Wrap &curr, const std::string &param="simplex"){
                curr.set_analytic_bestfit_src(true);
                if(param=="simplex") {
                        curr.chi_square_fit_simplex();
                } else if (param=="powell") {
                        curr.chi_square_fit_powell();
                } else if (param=="twalk") {
                        curr.chi_square_twalk();
                } else {
                        throw std::runtime_error("Available parameters: simplex (default), powell, twalk");
                }
        })
        .def("run_fit", &Lens_Wrap::chi_square_fit_simplex)
        .def("test_lens", &Lens_Wrap::test_lens_functions)
        ;

    py::class_<image>(m, "image")
        .def_readonly("pos", &image::pos)
        .def_readonly("mag", &image::mag)
        .def_readonly("td", &image::td)
        .def_readonly("parity", &image::parity)
        ;
    

    py::class_<lensvector>(m, "lensvector")
        .def(py::init([](){ return new lensvector(); }));

    py::class_<ImageSet>(m, "ImageSet")
        .def(py::init<>([](){ return new ImageSet(); }))
        // .def()
        // .def("print", &ImageSet::print)
        .def_readonly("n_images", &ImageSet::n_images)
        .def_readonly("src", &ImageSet::src)
        .def_readwrite("images", &ImageSet::images)
        ;
}