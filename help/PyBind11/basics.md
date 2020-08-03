# PyBind11 Basics

To read more about PyBind11, please refer to the docs here: https://pybind11.readthedocs.io/en/stable/

<!-- TOC -->

- [PyBind11 Basics](#pybind11-basics)
    - [QLens Implementation](#qlens-implementation)
        - [Python Classes](#python-classes)
        - [Processing User Inputs from Python](#processing-user-inputs-from-python)
        - [Multiple Signatures/Arguments in Python](#multiple-signaturesarguments-in-python)
        - [Default Arguments in PyBind11](#default-arguments-in-pybind11)

<!-- /TOC -->

## QLens Implementation
@Author: ExcelE

Much of the QLens integration to Python is written in `qlens_export.cpp`. 

### Python Classes

If in Python you want the user to create a object like so:

    from something import *
    Object1 = ClassObject()

You would add the following in the `qlens_export.cpp`:

    py::class_<ClassObject>(m, "ClassObject")

*Note: This works with structs, class within class or structs within structs.
See `py::class_<lensvector>` and `py::class<Lens_Wrap::critical_curve>`

### Processing User Inputs from Python

If in Python you want the user to pass a string like so:

    Object1.process1("option") # Python string

You would want to define:

    .def("option", [](ClassObject &current, py::str in){ // current is a reference to itself. Think of it as <this> in C++
        std::string input = py::cast<std::string>(in); // Converting Python str to string.
        ...
    })

It is important to note that if you want to deal with user data, you must do a type convertion from PyBind handlers to C++ native datatype. In simple terms, use `py::cast< [C++ native data type] >(PyBind object)` before doing any data manipulation/comparison. 

This goes with other types. For example:

    Object1.process1({"arg1": 12.0}) # Python Dictionary

You would want to define:

    .def("process1", [](ClassObject &current, py::dict in){  // notice py::dict
        double b = py::cast<double>(in["arg1"]);;
        ...
    })

### Multiple Signatures/Arguments in Python

Let's say you want to have one function that has two possible inputs:

    Object1.process1(1) # Prints to the console while running
    Object1.process1(1, verbose=False) # No outputs after finish

In C++, this is called multiple signatures where one function (in this case, `process1`) can either accept 1 or 2 arguments without returning an error. 

Let's look at this implementation:

    .def("add_lenses", &Lens_Wrap::batch_add_lenses_tuple)
    .def("add_lenses", [](Lens_Wrap &self){
            return "Pass in an array of lenses \n\tEx: [Lens1, Lens2, Lens3] \nor an array of tuples. Each tuple must contain the lens, the zl and zs values for each corresponding lens. \n\tEx: [(Lens1, zl1, zs1), (Lens2, zl2, zs2)]";
    })

In this case, if a user runs `L.add_lenses()`, it will print the string from the second definition. If the user matches the number of arguments from `&Lens_Wrap::batch_add_lenses_tuple` it will run the first definition.

### Default Arguments in PyBind11

Kind of similar to multiple signatures in terms of how the user experiences the function.

For example:

    Object1.process1(1)
    Object1.process1(data=1, verbose=False, save_to_disk=True)

This has the following implementation:

    .def("process1", 
        [](Object &obj, int data, bool verbose=false, 
            bool save_to_disk = false)
            {
                // Do something here
            },
        py::arg("data"),
        py::arg("verbose") = false,
        py::arg("save_to_disk") = false,
    )

Notice that after the lambda function, there is `py::arg(parameter)`. PyBind11 cannot automatically read the function's signature/inputs. So we have to explicitly tell pybind what are its inputs. 

If some parameters have defaults, you need to specify `py::arg(param) = false`. This default value must match the function default.
