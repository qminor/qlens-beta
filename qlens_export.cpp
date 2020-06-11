#include <pybind11/pybind11.h>
#include "qlens_wrapper.cpp"

namespace py = pybind11;

PYBIND11_MODULE(qlens, m) {
    m.doc() = "QLens Python Plugin"; // optional module docstring

    py::class_<Lens_Wrap>(m, "Lens")
        .def(py::init<>())
        .def("imgdata_add", &Lens_Wrap::imgdata_add, 
                py::arg("x") = -1.0, py::arg("y") = -1.0)
        .def("imgdata_write", &Lens_Wrap::imgdata_write_file)
        .def("imgdata_clear", &Lens_Wrap::imgdata_clear, 
                py::arg("lower") = -1, py::arg("upper") = -1)
        .def("imgdata_load", &Lens_Wrap::imgdata_load_file);
}