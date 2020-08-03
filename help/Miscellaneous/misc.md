# Additional Notes 
@Author: ExcelE

## Parallel Core Processing

When ready to compile, you can run `make -j` to use multiple cores for compilation. You may have to deal with random race conditions. In the integration of PyBind11 for instance, `profile.o` always fails due to a `not found` error. This was easily fixed by moving `profile.o` to the top of the `objects` list in the Makefile, which pushes it to be compiled first. Your mileage may vary depending on the OS and the compiler you have.

## Missing Libraries (PyBind/Python) when Compilling

If you are trying to compile something from C++ to produce a Python object, you may get a missing library error (can't remember the specific error). You need to add `python3-config --ldflags` in the flags on the Makefile. I already added them on `LINKLIBS` but if you for some reason can't compile, maybe check this flag.

## C++ Lambda Expressions

If you check out `qlens_export.cpp`, you might notice different `.def(...)` implementations. For example:

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

On the first `def` statement, we have "get_model_name" as the python function name to be mapped to `&LensProfile::get_model_name`. `&LensProfile::get_model_name` is a reference to the memory address of that function in memory after it gets compiled. `&LensProfile::get_model_name`'s definition is already defined in another cpp file.

The second `def` statement maps the python function name "set_vary_flags" to a function we define here inline. It is equivalent to the following:

    void process(LensProfile &current, py::list list){ 
        std::vector<double> lst = py::cast<std::vector<double>>(list);
        boolvector val(lst.size());
        int iter = 0;
        for (auto item : list) {
                val[iter] = py::cast<bool>(item); iter++;
        }
        current.set_vary_flags(val);
    }

But because this function is only applicable to this one place (never used in another function), there is no need to define this helper function explicitly and call it on another. 

The format for the lambda expression is 

    [](<Function arguments>){...Rest of the code.}


Lambda expressions/functions are useful in creating a helper function inline. we could have created it in class definitions itself, but that may make the source code more cluttered. For example, let's say we want to have a function that processes a Python list. If we were to create it inside a class definition we would need to import PyBind11 headers inside that file. Then, we also need to make sure that the Makefile has all the settings matched to the python3 source files. These steps may be simple at first but if the project were to get complicated, debugging will be extremely hard as it would mean we have to look at PyBind and C++ issues together. 