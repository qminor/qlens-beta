// QLENS: Strong gravitational lensing software with a command-line interface (beta version)
//        by Quinn Minor (qeminor@gmail.com)

#include "qlens.h"
#include "pixelgrid.h"
#include "errors.h"
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>

using namespace std;

char *advance(char *p);
void usage_error(const int mpi_id);

int main(int argc, char *argv[])
{
	int mpi_id=0, mpi_np=1;

#ifdef USE_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
#endif

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
	Lens::allocate_multithreaded_variables(n_omp_threads);

	bool read_from_file = false;
	bool verbal_mode = true;
	bool quit_after_reading_file = false;
	char input_filename[40];
	bool find_total_time = false;
	int inversion_nthread = n_omp_threads;
	int ngroups = mpi_np;
	bool disptime=false;
	bool mumps_mpi=true;
	bool quit_if_error = true;
	bool suppress_plots = false;
	CosmologyParams cosmology;

	bool load_cosmology_file = false;
	char cosmo_params_file[30] = "";
	string cosmology_filename;

	for (int i = 1; i < argc; i++)   // Process command-line arguments
	{
		if ((*argv[i] == '-') and (isalpha(*(argv[i]+1)))) {
			int c;
			while (c = *++argv[i]) {
				switch (c) {
					case 's': verbal_mode = false; break;
					case 'q': quit_after_reading_file = true; break;
					case 'T': find_total_time = true; break;
					case 'w': disptime = true; break;
					case 'p': mumps_mpi = false; break;
					case 'Q': quit_if_error = false; break;
					case 'n': suppress_plots = true; break;
					case 't':
						if (sscanf(argv[i], "t%i", &inversion_nthread)==0) usage_error(mpi_id);
						argv[i] = advance(argv[i]);
						break;
					case 'f':
						read_from_file = true;
						if (sscanf(argv[i], "f:%s", input_filename)==1)
							argv[i] += (1 + strlen(input_filename));
						argv[i] = advance(argv[i]);
						break;
				case 'c':
					if (sscanf(argv[i], "c:%s", cosmo_params_file)==1) {
						load_cosmology_file = true;
						argv[i] += (1 + strlen(cosmo_params_file));
						cosmology_filename.assign(cosmo_params_file);
					}
					break;
					case 'g':
						if (sscanf(argv[i], "g%i", &ngroups)==0) usage_error(mpi_id);
						if (ngroups > mpi_np) {
#ifdef USE_MPI
							MPI_Finalize();
#endif
							if (mpi_id==0) cerr << "Error: cannot have more MPI groups than the total number of MPI processes running\n";
							exit(1);
						}
						argv[i] = advance(argv[i]);
						break;
					default: if (mpi_id==0) usage_error(mpi_id); break;
				}
			}
		} else {
			read_from_file = true;
			if (sscanf(argv[i], "%s", input_filename)==1)
				argv[i] += (1 + strlen(input_filename));
			argv[i] = advance(argv[i]);
			ifstream test_open(input_filename);
			if (!test_open.good()) {
				cerr << "Error: specified input file '" << input_filename << "' does not exist" << endl;
				usage_error(mpi_id);
			}
		}
	}
	if ((load_cosmology_file) and (cosmology.load_params(cosmology_filename)==false)) die();

#ifdef USE_MPI
	MPI_Comm subgroup_comm[ngroups];
	MPI_Group subgroup[ngroups];
	int subgroup_size[ngroups];
	int *subgroup_rank[ngroups];
	int subgroup_id, subgroup_id_sum;

	int n,i,j,group_number;
	MPI_Group world_group;
	MPI_Comm world_comm;

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

	MPI_Comm onegroup_comm[mpi_np];
	MPI_Group onegroup[mpi_np];
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

	Lens lens;
#ifdef USE_OPENMP
	if (disptime) lens.set_show_wtime(true); // useful for optimizing the number of threads and MPI processes to minimize the wall time per likelihood evaluation
#endif
#ifdef USE_MPI
	int mpi_group_leaders[ngroups];
	for (int i=0; i < ngroups; i++) mpi_group_leaders[i] = subgroup_rank[i][0];
	lens.set_mpi_params(mpi_id,mpi_np,ngroups,group_number,subgroup_id,subgroup_size[group_number],mpi_group_leaders,&subgroup[group_number],&subgroup_comm[group_number],&onegroup[mpi_id],&onegroup_comm[mpi_id]);
	if (load_cosmology_file) lens.set_cosmology(cosmology);
	if (ngroups==mpi_np) {
		lens.Set_MCMC_MPI(mpi_np,mpi_id);
	} else {
		lens.Set_MCMC_MPI(mpi_np,mpi_id,ngroups,group_number,mpi_group_leaders);
	}
	for (n=0; n < ngroups; n++) delete[] subgroup_rank[n];
#else
	lens.set_mpi_params(0,1); // no MPI, so we have one process and id=0
#endif

	lens.set_verbal_mode(verbal_mode);
	lens.set_inversion_nthreads(inversion_nthread);
	lens.set_mumps_mpi(mumps_mpi);
	lens.set_quit_after_error(quit_if_error);
	if (suppress_plots) lens.set_suppress_plots(true);
	if (read_from_file) {
		if (lens.open_command_file(input_filename)==false) {
			cerr << "Error: could not open input file '" << input_filename << "'\n\n";
			exit(1);
		}
		lens.set_quit_after_reading_file(quit_after_reading_file);
	}

	if ((mpi_id==0) and (verbal_mode==true)) {
		cout << "QLens by Quinn Minor (2018)\n";
		cout << "Type 'help' for a list of commands, or 'demo1' or 'demo2' to see demos (or 'q' to quit).\n\n";
	}

	double wtime0;
	clock_t clocktime0;
	if (find_total_time) disptime=true;
	if (disptime) {
#ifdef USE_OPENMP
		wtime0 = omp_get_wtime();
#else
		clocktime0 = clock();
#endif
	}

	lens.process_commands(read_from_file);

	if (disptime) {
		double wtime;
#ifdef USE_OPENMP
		wtime = omp_get_wtime() - wtime0;
#else
		wtime = ((double) (clock() - clocktime0)) / CLOCKS_PER_SEC;
#endif
		if (mpi_id==0) cout << "Total time: " << wtime << endl;
	}

#ifdef USE_MUMPS
	Lens::delete_mumps();
#endif
	Grid::deallocate_multithreaded_variables();
	SourcePixelGrid::deallocate_multithreaded_variables();
	Lens::deallocate_multithreaded_variables();

#ifdef USE_MPI
	MPI_Finalize();
#endif

	return 0;
}

char *advance(char *p)
{
	// This advances to the next flag (if there is one; 'e' is ignored because it might be part of a number in scientific notation)
	while ((*++p) and ((!isalpha(*p)) or (*p=='e'))) ;
	return --p;
}

void usage_error(const int mpi_id)
{
	if (mpi_id==0) {
		cout << "Usage: qlens [args] [script_filename]    (all arguments are optional)\n\n"
				"Argument options:\n"
				"  -f:<file> Read commands from input script with filename <file>\n"
				"  -c:<file> Load cosmology parameters from input file (default: 'planck.csm')\n"
				"  -s        Run qlens in nonverbal mode (does not echo commands read from file, etc.)\n"
				"  -q        Skip pauses and quit after reading input file (rather than enter interactive mode)\n"
				"  -Q        Do not quit if an error is encountered while running an input script\n";
#ifdef USE_OPENMP
		cout << "  -t        # of OpenMP threads for matrix inversion (only inversion part)\n";
		cout << "  -w        Show wall time for computationally intensive parts (matrix inversion, etc.)\n";
#endif
#ifdef USE_MPI
		cout << "  -g##      # of MPI groups for MCMC/nested sampling (default=# of MPI processes)\n";
#endif
#ifdef USE_MUMPS
		cout << "  -p        Run MUMPS in serial mode, rather than parallel mode\n";
#endif
		cout << endl;
	}
#ifdef USE_MPI
	MPI_Finalize();
#endif
	exit(0);
}

