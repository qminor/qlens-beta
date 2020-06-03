#include <pybind11/pybind11.h>
#include "qlens.h"

namespace py = pybind11;

class Pet {
public:
    Pet() : name("default") { }
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }
    int sample_add(int a, int b) {
        return a + b;
    }
private:
    std::string name;
};

PYBIND11_MODULE(qlens, m) {
    m.doc() = "QLens Python Plugin"; // optional module docstring

    py::class_<Pet>(m, "Pet", py::dynamic_attr())
        .def(py::init<const std::string &>())
        .def(py::init<>())
        .def("add", &Pet::sample_add);

    py::class_<Lens>(m, "Lens")
        .def(py::init<>())
        .def("add", &Lens::sample_add);
}