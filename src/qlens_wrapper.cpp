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
	QLens_Wrap(const double zlens, const double zsrc, const double zsrc_ref, Cosmology* cosmo_in = NULL) : QLens(cosmo_in)
	 {
		 lens_redshift = zlens;
		 source_redshift = zsrc;
		 reference_source_redshift = zsrc_ref;
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
		CartesianSourceGrid::allocate_multithreaded_variables(n_omp_threads);
		DelaunayGrid::allocate_multithreaded_variables(n_omp_threads);
		ImagePixelGrid::allocate_multithreaded_variables(n_omp_threads);
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
	 	set_verbal_mode(true);
		set_inversion_nthreads(n_omp_threads);
	}

	void batch_add_lenses(py::list list) {
		LensProfile* curr;
		for (auto arr : list){
			try {
				curr = py::cast<LensProfile*>(arr);
			} catch (...) {
				throw std::runtime_error("Error adding lenses. Input should be an array of tuples. Ex: [(<Lens1>, zl1, zs2), (<Lens2>, zl2, zs2)]");
			}
			add_lens(curr);
		}
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
			add_lens(curr);
		}
	}

	void add_lens_tuple(std::tuple<LensProfile*, double, double> lens_tuple) {
		double zl, zs;
		LensProfile* curr;
			zl = std::get<1>(lens_tuple);
			zs = std::get<2>(lens_tuple);
			curr = std::get<0>(lens_tuple);
		add_lens(curr);
	}

	void add_lens_extshear(LensProfile* lens_in, Shear* extshear) {
		add_lens(lens_in);
		add_lens((LensProfile*) extshear);
		  lens_list[nlens-1]->anchor_center_to_lens(nlens-2);
	}

	//void add_lens_default(LensProfile* lens_in) {
		//add_lens(lens_in);
	//}

	void batch_add_sources(py::list list) {
		SB_Profile* curr;
		for (auto arr : list){
			try {
				curr = py::cast<SB_Profile*>(arr);
			} catch (...) {
				throw std::runtime_error("Error adding sources. Input should be an array of sources");
			}
			add_source(curr, false);
		}
	}

	void add_src_default(SB_Profile* src_in) {
		add_source(src_in, false);
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

	bool get_shear_components_mode() { return Shear::use_shear_component_params; }
	void set_shear_components_mode(const bool comp) {
			Shear::use_shear_component_params = comp;
			reassign_lensparam_pointers_and_names();
	}
	bool get_ellipticity_components_mode() { return LensProfile::use_ellipticity_components; }
	void set_ellipticity_components_mode(const bool comp) {
			LensProfile::use_ellipticity_components = comp;
			reassign_lensparam_pointers_and_names();
	}

	bool get_split_imgpixels() { return split_imgpixels; }
	void set_split_imgpixels(const bool split) {
		bool old_setting = split_imgpixels;
		if ((split==false) and ((use_srcpixel_clustering) or (use_lum_weighted_srcpixel_clustering))) {
			if (mpi_id==0) cout << "NOTE: turning off source pixel clustering" << endl;
			use_srcpixel_clustering = false;
			use_lum_weighted_srcpixel_clustering = false;
		}
		split_imgpixels = split;
		if (split_imgpixels != old_setting) {
			if (psf_supersampling) {
				psf_supersampling = false;
				if (mpi_id==0) cout << "NOTE: Turning off PSF supersampling" << endl;
			}
			if (image_pixel_grids != NULL) {
				for (int i=0; i < n_extended_src_redshifts; i++) {
					if (image_pixel_grids[i] != NULL) {
						image_pixel_grids[i]->delete_ray_tracing_arrays();
						image_pixel_grids[i]->setup_ray_tracing_arrays();
						if (nlens > 0) image_pixel_grids[i]->calculate_sourcepts_and_areas(true);
					}
				}
			}
			if (fft_convolution) cleanup_FFT_convolution_arrays();
		}
	}

	bool get_optimize_regparam() { return optimize_regparam; }
	void set_optimize_regparam(const bool opt) {
		 optimize_regparam = opt;
		 int i;
			if (n_pixellated_src > 0) {
				bool vary_regparam = false;
				for (i=0; i < n_pixellated_src; i++) {
					srcgrids[i]->get_specific_varyflag("regparam",vary_regparam);
					if (vary_regparam) break;
				}
				if (vary_regparam) {
					if (mpi_id==0) cout << "NOTE: setting 'vary_regparam' to 'off' and updating parameters" << endl;
					for (i=0; i < n_pixellated_src; i++) {
						update_pixellated_src_varyflag(i,"regparam",false);
					}
					update_parameter_list();
				}
			}
			if ((opt==false) and (use_lum_weighted_regularization) and ((!use_saved_sbweights) or (!get_lumreg_from_sbweights))) {
				if (mpi_id==0) cout << "NOTE: setting 'lum_weighted_regularization' to 'off' (to keep it on, consider using sbweights via 'lumreg_from_sbweights' and 'use_saved_sbweights))" << endl;
				use_lum_weighted_regularization = false;
				for (i=0; i < n_pixellated_src; i++) {
					if (source_fit_mode==Delaunay_Source) update_pixsrc_active_parameters(i);
				}
				update_parameter_list();
			}
	 }

	std::string sbmap_load_image_file(const std::string &filename_, py::kwargs& kwargs) { 
			int band_i = 0;
			int hdu_indx = 1;
			bool show_header = false;
			double pixsize = default_data_pixel_size;
			double x_offset = 0.0, y_offset = 0.0;

			if (kwargs) {
				for (auto item : kwargs) {
						 if (py::cast<string>(item.first)=="band") {
							 band_i = py::cast<int>(item.second);
						 } else if (py::cast<string>(item.first)=="pixsize") {
							 pixsize = py::cast<double>(item.second);
						 } else if (py::cast<string>(item.first)=="hdu_indx") {
							 hdu_indx = py::cast<int>(item.second);
						 } else if (py::cast<string>(item.first)=="x_offset") {
							 x_offset = py::cast<double>(item.second);
						 } else if (py::cast<string>(item.first)=="y_offset") {
							 y_offset = py::cast<double>(item.second);
						 } else if (py::cast<string>(item.first)=="show_header") {
							 show_header = py::cast<bool>(item.second);
						 }
					 }
			}

			if (!load_image_pixel_data(band_i,filename_,pixsize,1.0,x_offset,y_offset,hdu_indx,show_header)) throw runtime_error("could not load image data");

		return filename_;
	}

	std::string sbmap_load_noise_map(const std::string &filename_, py::kwargs& kwargs) { 
			int band_i = 0;
			int hdu_indx = 1;
			bool show_header = false;

			if (kwargs) {
				 for (auto item : kwargs) {
					 if (py::cast<string>(item.first)=="band") {
						 band_i = py::cast<int>(item.second);
					 } else if (py::cast<string>(item.first)=="hdu_indx") {
						 hdu_indx = py::cast<int>(item.second);
					 } else if (py::cast<string>(item.first)=="show_header") {
						 show_header = py::cast<bool>(item.second);
					 }
				 }
			}

			if (band_i >= n_data_bands) throw runtime_error("image data for specified band has not been loaded yet");
			if (!imgdata_list[band_i]->load_noise_map_fits(filename_,hdu_indx,show_header)) throw runtime_error("could not load noise map fits file '" + filename_ + "'");

			use_noise_map = true;

		return filename_;
	}

	std::string sbmap_load_psf(const std::string &filename_, py::kwargs& kwargs) { 
			int band_i = 0;
			int hdu_indx = 1;
			bool show_header = false;
			bool load_supersampled_psf = false;
			bool verbal_mode = true;

			if (kwargs) {
				 for (auto item : kwargs) {
					 if (py::cast<string>(item.first)=="band") {
						 band_i = py::cast<int>(item.second);
					 } else if (py::cast<string>(item.first)=="hdu_indx") {
						 hdu_indx = py::cast<int>(item.second);
					 } else if (py::cast<string>(item.first)=="show_header") {
						 show_header = py::cast<bool>(item.second);
					 }
				 }
			}

			if (band_i > n_psf) throw runtime_error("band index is higher than n_psf. To create new PSF, set band_i=n_psf");
			if (band_i==n_psf) add_psf();
			if (!psf_list[band_i]->load_psf_fits(filename_,hdu_indx,load_supersampled_psf,show_header,verbal_mode and (mpi_id==0))) throw runtime_error("could not load PSF fits file '" + filename_ + "'");
			if (fft_convolution) cleanup_FFT_convolution_arrays();

		return filename_;
	}

	std::string sbmap_load_mask(const std::string &filename_, py::kwargs& kwargs) { 
			int band_i = 0;
			int mask_i = 0;
			bool add_mask = false;
			bool subtract_mask = false;
			bool foreground_mask = false;
			bool emask = false;

			if (kwargs) {
				 for (auto item : kwargs) {
					 if (py::cast<string>(item.first)=="band") {
						 band_i = py::cast<int>(item.second);
					 } else if (py::cast<string>(item.first)=="mask") {
						 mask_i = py::cast<int>(item.second);
					 } else if (py::cast<string>(item.first)=="fgmask") {
						 foreground_mask = py::cast<bool>(item.second);
					 } else if (py::cast<string>(item.first)=="emask") {
						 emask = py::cast<bool>(item.second);
					 } else if (py::cast<string>(item.first)=="add_mask") {
						 add_mask = py::cast<bool>(item.second);
					 } else if (py::cast<string>(item.first)=="subtract_mask") {
						 subtract_mask = py::cast<bool>(item.second);
					 }
				 }
			}

			if (band_i >= n_data_bands) throw runtime_error("image data for specified band has not been loaded yet");
			if (imgdata_list[band_i]->load_mask_fits(mask_i,filename_,foreground_mask,emask,add_mask,subtract_mask)==false) throw runtime_error("could not load mask file");

		return filename_;
	}

	void lens_display() {
		print_lens_list(false);
	}
	void src_display() {
		print_source_list(false);
	}
	void pixsrc_display() {
		print_pixellated_source_list(false);
	}

	void lens_clear(int min_loc=-1, int max_loc=-1) {
		if((min_loc == -1) and (max_loc == -1)) {
			for (int i=nlens-1; i >= 0; i--) {
				remove_lens(i,false);
			}
		} else if ((min_loc != -1) and (max_loc == -1)) {
			if (min_loc >= nlens) throw runtime_error("specified lens index does not exist");
			remove_lens(min_loc,false);
		} else {
			if ((min_loc < 0) or (min_loc >= nlens)) throw runtime_error("specified lens index does not exist");
			if ((max_loc < 0) or (max_loc >= nlens)) throw runtime_error("specified lens index does not exist");
			if (min_loc > max_loc) throw runtime_error("max index must be greater than min index");
			for (int i=max_loc; i >= min_loc; i--) {
				remove_lens(i,false);
			}
		}
	}

	void source_clear(int min_loc=-1, int max_loc=-1) {
		if((min_loc == -1) and (max_loc == -1)) {
			for (int i=n_sb-1; i >= 0; i--) {
				remove_source_object(i,false);
			}
		} else if ((min_loc != -1) and (max_loc == -1)) {
			if (min_loc >= n_sb) throw runtime_error("specified source index does not exist");
			remove_source_object(min_loc,false);
		} else {
			if ((min_loc < 0) or (min_loc >= n_sb)) throw runtime_error("specified source index does not exist");
			if ((max_loc < 0) or (max_loc >= n_sb)) throw runtime_error("specified source index does not exist");
			if (min_loc > max_loc) throw runtime_error("max index must be greater than min index");
			for (int i=max_loc; i >= min_loc; i--) {
				remove_source_object(i,false);
			}
		}
	}

	void pixsrc_clear(int min_loc=-1, int max_loc=-1) {
		if((min_loc == -1) and (max_loc == -1)) {
			for (int i=n_pixellated_src-1; i >= 0; i--) {
				remove_pixellated_source(i,false);
			}
		} else if ((min_loc != -1) and (max_loc == -1)) {
			if (min_loc >= n_pixellated_src) throw runtime_error("specified pixellated source index does not exist");
			remove_pixellated_source(min_loc,false);
		} else {
			if ((min_loc < 0) or (min_loc >= n_pixellated_src)) throw runtime_error("specified pixellated source index does not exist");
			if ((max_loc < 0) or (max_loc >= n_pixellated_src)) throw runtime_error("specified pixellated source index does not exist");
			if (min_loc > max_loc) throw runtime_error("max index must be greater than min index");
			for (int i=max_loc; i >= min_loc; i--) {
				remove_pixellated_source(i,false);
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

	void set_regularization_method(const std::string &method) {
		if (method=="none") {
			regularization_method = None;
			if (optimize_regparam) {
				optimize_regparam = false;
				if (mpi_id==0) cout << "NOTE: Turning 'optimize_regparam' off" << endl;
			}
		}
		else if (method=="norm") regularization_method = Norm;
		else if (method=="gradient") regularization_method = Gradient;
		else if (method=="sgradient") regularization_method = SmoothGradient;
		else if (method=="curvature") regularization_method = Curvature;
		else if (method=="scurvature") regularization_method = SmoothCurvature;
		else if (method=="matern_kernel") regularization_method = Matern_Kernel;
		else if (method=="exp_kernel") regularization_method = Exponential_Kernel;
		else if (method=="sqexp_kernel") regularization_method = Squared_Exponential_Kernel;
		else throw runtime_error("invalid argument to 'regularization_method'; must specify valid regularization method");
		for (int i=0; i < n_pixellated_src; i++) {
			update_pixsrc_active_parameters(i);
		}
	}

	string get_regularization_method() {
		string method;
		if (regularization_method == None) method = "none";
		else if (regularization_method == Norm) method = "norm";
		else if (regularization_method == Gradient) method = "gradient";
		else if (regularization_method == SmoothGradient) method = "sgradient";
		else if (regularization_method == Curvature) method = "curvature";
		else if (regularization_method == SmoothCurvature) method = "scurvature";
		else if (regularization_method == Matern_Kernel) method = "matern_kernel";
		else if (regularization_method == Exponential_Kernel) method = "exp_kernel";
		else if (regularization_method == Squared_Exponential_Kernel) method = "sqexp_kernel";
		return method;
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
		ImagePixelGrid::deallocate_multithreaded_variables();
		DelaunayGrid::deallocate_multithreaded_variables();
		CartesianSourceGrid::deallocate_multithreaded_variables();
		QLens::deallocate_multithreaded_variables();

#ifdef USE_MPI
		MPI_Finalize();
		//delete[] subgroup_comm;
		//delete[] subgroup;
		//delete[] onegroup_comm;
		//delete[] onegroup;
#endif
	 }
};

