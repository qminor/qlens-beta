#include "qlens.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "profile.h"

namespace py = pybind11;



class Lens_Wrap: public Lens {
public:

    Lens_Wrap() : Lens() {}

    void batch_add_lenses_tuple(py::list list) {
        double zl, zs;
        LensProfile* curr;
        for (auto arr : list){
            try {
                std::tuple<LensProfile*, double, double> extracted = py::cast<std::tuple<LensProfile*, double, double>>(arr);
                zl = std::get<1>(extracted);
                zs = std::get<2>(extracted);
                curr = std::get<0>(extracted);
            } catch(std::runtime_error) {
                zl = 0.5;
                zs = 1.0;
                curr = py::cast<LensProfile*>(arr);
            } catch (...) {
                throw std::runtime_error("Error adding lenses. Input should be an array of tuples. Ex: [(<Lens1>, zl1, zs2), (<Lens2>, zl2, zs2)]");
            }
            add_lens(curr, zl, zs);
        }
    }

    std::string imgdata_load_file(const std::string &name_) { 
        if (load_image_data(name_)==false) throw runtime_error("Unable to read data");
        update_parameter_list();
        return name_;
    }

    void imgdata_display() {
        if (n_sourcepts_fit==0) throw runtime_error("no image data has been loaded");
        print_image_data(true);
    }

    void imgdata_write_file(const std::string &name_) {
        try{
            write_image_data(name_);
        } catch(...) {
            throw runtime_error("Unable to write to: " + name_);
        }
    }

    bool imgdata_clear(int lower = -1, int upper = -1) {
        // No argument, clear all
        if (lower==-1) { clear_image_data(); return true; }

        if (upper > n_sourcepts_fit) {throw runtime_error("Specified max image dataset number exceeds number of data sets in list");}
        
        // Clear a range
        bool is_range_ok = upper > lower;
        if (is_range_ok && lower > -1 && upper != -1) {
            for (int i=upper; i >= lower; i--) remove_image_data(i);
            return true;
        }

        // User did not use the proper format
        throw runtime_error("Specify proper range or leave empty to clear all data.");
    }

    void imgdata_add(double x, double y) {
        if(x<=-1 || y<=-1) { throw runtime_error("Please specify a proper coodinate"); }

        lensvector src;
        src[0] = x;
        src[1] = y;
        if(add_simulated_image_data(src)) update_parameter_list();
    }

    void lens_display() {
        print_lens_list(false);
    }

    void lens_clear(int min_loc=-1, int max_loc=-1) {
        if(min_loc == -1 && max_loc == -1) {
            clear_lenses();
        } else if (min_loc != -1 && max_loc == -1) {
            remove_lens(min_loc);
        } else {
            for (int i=max_loc; i >= min_loc; i--) {
                remove_lens(i);
            }
        }
    }

    void update_lens(py::args args, py::kwargs kwargs, int loc=-1) {
        /* FEATURE IDEA: from https://pybind11.readthedocs.io/en/stable/advanced/functions.html#accepting-args-and-kwargs

            1. Accept *args, **kwargs from python user
            2. If an argument matches with lens parameter
        */
        // TODO: Verify with Quinn on where in commands.cpp gets the updated and then adapt here.
        if (loc == -1) throw runtime_error("Please specify the lens to update.");
        // if (kwargs) {
            
        // }
    }
        
private:
};