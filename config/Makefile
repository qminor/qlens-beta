#
#topdir = ./MUMPS_5.0.1
#libdir = $(topdir)/lib
#include $(topdir)/Makefile.inc
#LIBMUMPS_COMMON = $(libdir)/libmumps_common$(PLAT)$(LIBEXT)
#LIBDMUMPS = $(libdir)/libdmumps$(PLAT)$(LIBEXT) $(LIBMUMPS_COMMON)

#CMULTINEST = # put multinest include directory here
#CPOLYCHORD = # put polychord include directory here
#MULTINEST_LIB = L/.../MultiNest/lib -lmultinest_mpi # enter in multinest library path, and add the folder to LD_LIBRARY_PATH
#POLYCHORD_LIB = -L/.../PolyChord/lib -lchord # enter in polychord library path, and add the folder to LD_LIBRARY_PATH

#FITPACKDIR=./contrib/fitpack
#F77=gfortran
#.f.o:
	#$(F77) -c -o  $@ $<

EXT := $(shell python3-config --extension-suffix)
MODULE := qlens$(shell python3-config --extension-suffix)
PYBIND11_CFLAGS := $(shell python3 -m pybind11 --includes)

STAN_INC = -I/Users/qminor/math -I/Users/qminor/math/lib/eigen_3.4.0 -I/Users/qminor/math/lib/boost_1.87.0 -I/Users/qminor/math/lib/sundials_6.1.1/include -I/Users/qminor/math/lib/sundials_6.1.1/src/sundials -include /Users/qminor/math/stan/math/prim/constraint/identity_free.hpp -I/Users/qminor/math/lib/tbb_2020.3/include

# Version without MUMPS
#default: qlens mkdist cosmocalc 
default: qlens mkdist qlens-wrap
CCOMP = g++
#CCOMP = mpicxx -DUSE_MPI
#OPTS = -w -fopenmp -O3
#OPTS = -g -w -fopenmp #for debugging
OPTS = -Wno-write-strings -O0 -std=c++17
OPTS_NO_OPT = -Wno-write-strings -O0 -std=c++17
#OPTS = -w -g
#FLAGS = -DUSE_READLINE -DUSE_FITS -DUSE_OPENMP -DUSE_UMFPACK -DUSE_MULTINEST -DUSE_POLYCHORD -DUSE_FITPACK
FLAGS = -DUSE_READLINE -DUSE_STAN -D_REENTRANT
#OTHERLIBS =  -lm -lreadline -ltcmalloc -lcfitsio
OTHERLIBS =  /Users/qminor/math/lib/tbb/libtbb.dylib -lm -lreadline
LINKLIBS = $(OTHERLIBS) $(MULTINEST_LIB) $(POLYCHORD_LIB)

# Version with MUMPS
#default: qlens mkdist cosmocalc
#CCOMP = mpicxx -DUSE_MPI
#OPTS = -Wno-write-strings -O3 -fopenmp
#OPTS_NO_OPT = -Wno-write-strings -fopenmp
#FLAGS = -DUSE_OPENMP -DUSE_MUMPS -DUSE_FITS -DUSE_UMFPACK
#CMUMPS = $(INCS) $(CDEFS) -I. -I$(topdir)/include -I$(topdir)/src
#MUMPSLIBS = $(LIBDMUMPS) $(LORDERINGS) $(LIBS) $(LIBBLAS) $(LIBOTHERS) -lgfortran
#OTHERLIBS =  -lm -lreadline -lcfitsio -ltcmalloc
##OTHERLIBS =  -lm -lreadline -lcfitsio
#LINKLIBS = $(MUMPSLIBS) $(OTHERLIBS)

CC   := $(CCOMP) $(OPTS) $(UMFOPTS) $(FLAGS) $(CMUMPS) $(INC) $(STAN_INC)
CC_NO_OPT   := $(CCOMP) $(OPTS_NO_OPT) $(UMFOPTS) $(FLAGS) $(CMUMPS) $(INC) $(STAN_INC)
CL   := $(CCOMP) $(OPTS) $(UMFOPTS) $(FLAGS)

objects = sbprofile.o profile.o egrad.o models.o qlens.o commands.o params.o modelparams.o lenscalc.o \
				lens.o imgsrch.o pixelgrid.o cg.o mcmchdr.o errors.o brent.o sort.o \
				trirectangle.o cosmo.o \
				simplex.o powell.o mcmceval.o

wrapper_objects = qlens_export.o qlens_wrapper.o profile.o mcmceval.o egrad.o commands.o params.o modelparams.o lenscalc.o \
				lens.o imgsrch.o pixelgrid.o cg.o mcmchdr.o models.o sbprofile.o errors.o brent.o \
				sort.o trirectangle.o cosmo.o \
				simplex.o powell.o 

mkdist_objects = mkdist.o
mkdist_shared_objects = errors.o mcmceval.o

qlens: $(objects) $(LIBDMUMPS)
	$(CL) -o qlens $(OPTL) $(objects) $(LINKLIBS) $(UMFPACK) $(UMFLIBS) 

mkdist: $(mkdist_objects) $(mkdist_shared_objects)
	$(CC) -o mkdist $(mkdist_objects) $(mkdist_shared_objects) -lm

qlens-wrap: $(wrapper_objects)
	$(CL) -shared $(wrapper_objects) $(LINKLIBS) -undefined dynamic_lookup -o $(MODULE)

qlens_wrapper.o: qlens_wrapper.cpp qlens.h pixelgrid.h modelparams.h params.h delaunay.h lensvec.h profile.h sbprofile.h 
	$(CC) $(PYBIND11_CFLAGS) -c qlens_wrapper.cpp -o qlens_wrapper.o

qlens_export.o: qlens_export.cpp qlens.h pixelgrid.h modelparams.h params.h delaunay.h lensvec.h profile.h sbprofile.h 
	$(CC) $(PYBIND11_CFLAGS) -c qlens_export.cpp -o qlens_export.o

mumps:
	(cd MUMPS_5.0.1; $(MAKE))

qlens.o: qlens.cpp qlens.h
	$(CC) -c qlens.cpp

commands.o: commands.cpp qlens.h lensvec.h profile.h sbprofile.h egrad.h pixelgrid.h modelparams.h
	$(CC_NO_OPT) -c commands.cpp

params.o: params.cpp params.h 
	$(CC) -c params.cpp

modelparams.o: modelparams.cpp modelparams.h 
	$(CC) -c modelparams.cpp

lenscalc.o: lenscalc.cpp qlens.h lensvec.h
	$(CC) -c lenscalc.cpp

lens.o: lens.cpp profile.h sbprofile.h qlens.h pixelgrid.h lensvec.h matrix.h simplex.h powell.h mcmchdr.h cosmo.h delaunay.h modelparams.h
	$(CC) -c lens.cpp

imgsrch.o: imgsrch.cpp qlens.h lensvec.h
	$(CC) -c imgsrch.cpp

pixelgrid.o: pixelgrid.cpp profile.h sbprofile.h lensvec.h pixelgrid.h qlens.h matrix.h cg.h egrad.h modelparams.h
	$(CC) -c pixelgrid.cpp

cg.o: cg.cpp cg.h
	$(CC) -c cg.cpp

mcmchdr.o: mcmchdr.cpp mcmchdr.h mathfuncs.h random.h
	$(CC) -c mcmchdr.cpp

egrad.o: egrad.cpp egrad.h 
	$(CC) -c egrad.cpp

profile.o: profile.h profile.cpp lensintegral.h lensvec.h egrad.h
	$(CC) -c profile.cpp

models.o: models.cpp profile.h 
	$(CC) -c models.cpp

sbprofile.o: sbprofile.cpp sbprofile.h egrad.h
	$(CC) -c sbprofile.cpp

errors.o: errors.cpp errors.h
	$(CC) -c errors.cpp

brent.o: brent.h brent.cpp
	$(CC) -c brent.cpp

simplex.o: simplex.h rand.h simplex.cpp
	$(CC) -c simplex.cpp

powell.o: powell.h powell.cpp
	$(CC) -c powell.cpp

sort.o: sort.h sort.cpp
	$(CC) -c sort.cpp

#gauss.o: gauss.cpp gauss.h
	#$(CC) -c gauss.cpp

#romberg.o: romberg.cpp romberg.h
	#$(CC) -c romberg.cpp

#qspline.o: qspline.cpp spline.h errors.h
	#$(CC) -c qspline.cpp

#spline.o: spline.cpp spline.h errors.h
	#$(CC) -c spline.cpp

trirectangle.o: trirectangle.cpp lensvec.h trirectangle.h
	$(CC) -c trirectangle.cpp

mcmceval.o: mcmceval.cpp mcmceval.h mathfuncs.h random.h errors.h
	$(CC) -c mcmceval.cpp

mkdist.o: mkdist.cpp mcmceval.h errors.h
	$(CC) -c mkdist.cpp

#hyp_2F1.o: hyp_2F1.cpp hyp_2F1.h complex_functions.h
	#$(CC) -c hyp_2F1.cpp

cosmo.o: cosmo.cpp cosmo.h
	$(CC) -c cosmo.cpp

clean_qlens:
	rm qlens $(objects)

clean:
	rm qlens mkdist $(objects) $(mkdist_objects) 

clmain:
	rm qlens.o

vim_run:
	qlens

