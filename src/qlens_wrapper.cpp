#include "qlens.h"
#include "pixelgrid.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "profile.h"

namespace py = pybind11;

using namespace std;

class QLens_Wrap: public QLens {
public:
#ifdef USE_MPI
	MPI_Comm *subgroup_comm;
	MPI_Group *subgroup;
	MPI_Comm *onegroup_comm;
	MPI_Group *onegroup;
#endif

	int mpi_id, mpi_np;
	int ngroups;
    QLens_Wrap() : QLens()
	 {
		mpi_id=0;
		mpi_np=1;

#ifdef USE_MPI
		MPI_Init(NULL, NULL);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_np);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
#endif
		ngroups = mpi_np; // later, allow option to have mpi groups with multiple processes per group

		int n_omp_threads;
#ifdef USE_OPENMP
		#pragma omp parallel
		{
			#pragma omp master
			n_omp_threads = omp_get_num_threads();
		}
#else
		n_omp_threads = 1;
#endif
		Grid::allocate_multithreaded_variables(n_omp_threads);
		SourcePixelGrid::allocate_multithreaded_variables(n_omp_threads);
		QLens::allocate_multithreaded_variables(n_omp_threads);

#ifdef USE_MPI
		subgroup_comm = new MPI_Comm[ngroups];
		subgroup = new MPI_Group[ngroups];

		int subgroup_size[ngroups];
		int *subgroup_rank[ngroups];
		int subgroup_id, subgroup_id_sum;

		int n,i,j,group_number;
		MPI_Group world_group;

		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		for (n=0; n < ngroups; n++) {
			subgroup_size[n] = 0;
			for (i=n; i < mpi_np; i += ngroups) subgroup_size[n]++;
			subgroup_rank[n] = new int[subgroup_size[n]];
			for (j=0, i=n; i < mpi_np; j++, i += ngroups) {
				subgroup_rank[n][j] = i;
				//if (mpi_id==0) cout << "subgroup " << n << " process " << j << ": rank " << subgroup_rank[n][j] << endl;
			}

			MPI_Group_incl(world_group, subgroup_size[n], subgroup_rank[n], &subgroup[n]);
			MPI_Comm_create(MPI_COMM_WORLD, subgroup[n], &subgroup_comm[n]);

			if (mpi_id % ngroups == n) {
				MPI_Comm_rank(subgroup_comm[n],&subgroup_id);
				group_number = n;
			}
		}

		onegroup_comm = new MPI_Comm[mpi_np];
		onegroup = new MPI_Group[mpi_np];
		int onegroup_size[mpi_np];
		int *onegroup_rank[mpi_np];
		int onegroup_id, onegroup_id_sum;

		for (n=0; n < mpi_np; n++) {
			onegroup_size[n] = 1;
			onegroup_rank[n] = new int[1];
			onegroup_rank[n][0] = n;

			MPI_Group_incl(world_group, onegroup_size[n], onegroup_rank[n], &onegroup[n]);
			MPI_Comm_create(MPI_COMM_WORLD, onegroup[n], &onegroup_comm[n]);
		}
#endif

//#ifdef USE_OPENMP
		//if (disptime) set_show_wtime(true); // useful for optimizing the number of threads and MPI processes to minimize the wall time per likelihood evaluation
//#endif
#ifdef USE_MPI
		int mpi_group_leaders[ngroups];
		for (int i=0; i < ngroups; i++) mpi_group_leaders[i] = subgroup_rank[i][0];
		set_mpi_params(mpi_id,mpi_np,ngroups,group_number,subgroup_id,subgroup_size[group_number],mpi_group_leaders,&subgroup[group_number],&subgroup_comm[group_number],&onegroup[mpi_id],&onegroup_comm[mpi_id]);
		if (ngroups==mpi_np) {
			Set_MCMC_MPI(mpi_np,mpi_id);
		} else {
			Set_MCMC_MPI(mpi_np,mpi_id,ngroups,group_number,mpi_group_leaders);
		}
		for (n=0; n < ngroups; n++) delete[] subgroup_rank[n];
#else
		set_mpi_params(0,1); // no MPI, so we have one process and id=0
#endif
	 }

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
                zl = lens_redshift;
                zs = reference_source_redshift;
                curr = py::cast<LensProfile*>(arr);
            } catch (...) {
                throw std::runtime_error("Error adding lenses. Input should be an array of tuples. Ex: [(<Lens1>, zl1, zs2), (<Lens2>, zl2, zs2)]");
            }
            add_lens(curr, zl, zs);
        }
    }

    void add_lens_tuple(std::tuple<LensProfile*, double, double> lens_tuple) {
        double zl, zs;
        LensProfile* curr;
            zl = std::get<1>(lens_tuple);
            zs = std::get<2>(lens_tuple);
            curr = std::get<0>(lens_tuple);
        add_lens(curr, zl, zs);
    }

    void add_lens_default(LensProfile* lens_in) {
        add_lens(lens_in, lens_redshift, reference_source_redshift);
    }

    std::string imgdata_load_file(const std::string &name_) { 
        if (load_point_image_data(name_)==false) throw runtime_error("Unable to read data");
        update_parameter_list();
        return name_;
    }

    void imgdata_display() {
        if (n_ptsrc==0) throw runtime_error("no image data has been loaded");
        print_image_data(true);
    }

    void imgdata_write_file(const std::string &name_) {
        try{
            write_point_image_data(name_);
        } catch(...) {
            throw runtime_error("Unable to write to: " + name_);
        }
    }

    bool imgdata_clear(int lower = -1, int upper = -1) {
        // No argument, clear all
        if (lower==-1) { clear_image_data(); return true; }

        if (upper > n_ptsrc) {throw runtime_error("Specified max image dataset number exceeds number of data sets in list");}
        
        // Clear a range
        bool is_range_ok = upper > lower;
        if (is_range_ok && lower > -1 && upper != -1) {
            for (int i=upper; i >= lower; i--) remove_point_source(i);
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
	double LogLikeListFunc(py::list param_list)
	{
		if (fitmodel == NULL) return -1e30;
		vector<double> param_vec = py::cast<vector<double>>(param_list);
		return LogLikeVecFunc(param_vec);
	}
    ~QLens_Wrap()
	 {
		Grid::deallocate_multithreaded_variables();
		SourcePixelGrid::deallocate_multithreaded_variables();
		QLens::deallocate_multithreaded_variables();

#ifdef USE_MPI
		MPI_Finalize();
		//delete[] subgroup_comm;
		//delete[] subgroup;
		//delete[] onegroup_comm;
		//delete[] onegroup;
#endif
	 }
        
private:
};
