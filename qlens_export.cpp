#include <pybind11/pybind11.h>
#include "qlens_wrapper.cpp"

namespace py = pybind11;

PYBIND11_MODULE(qlens, m) {
    m.doc() = "QLens Python Plugin"; // optional module docstring

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
                py::arg("zl_in") = 0, py::arg("reference_source_redshift") = 2);
}