#include "qlens.h"

class Lens_Wrap: public Lens {
public:

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

    void imgdata_add(double x=-1.0, double y=-1.0) {
        if(x==-1 || y==-1) { throw runtime_error("Please specify a proper coodinate"); }

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

    void lens_add(double b, int alpha, int s, double q, int theta, double xc, double yc,
            std::string mode="alpha", int emode=1, int zl_in=0, int reference_source_redshift=2){
        add_lens(ALPHA, emode, zl_in, reference_source_redshift, b, alpha, s, q, theta, xc, yc);
        }
        
private:
};