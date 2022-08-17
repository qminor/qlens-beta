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
#.src/f.o:
	#$(F77) -c -o  $@ $<

# Version without MUMPS
default: qlens mkdist cosmocalc 
CCOMP = g++ -I./include	
#CCOMP = mpicxx -DUSE_MPI
#OPTS = -w -fopenmp -O3
#OPTS = -g -w -fopenmp #for debugging
OPTS = -Wno-write-strings -O3 -std=c++11
OPTS_NO_OPT = -Wno-write-strings -std=c++11
#OPTS = -w -g
#FLAGS = -DUSE_READLINE -DUSE_FITS -DUSE_OPENMP -DUSE_UMFPACK -DUSE_MULTINEST -DUSE_POLYCHORD -DUSE_FITPACK
FLAGS = -DUSE_READLINE
#OTHERLIBS =  -lm -lreadline -ltcmalloc -lcfitsio
OTHERLIBS =  -lm -lreadline
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

CC   := $(CCOMP) $(OPTS) $(UMFOPTS) $(FLAGS) $(CMUMPS) $(INC) 
CC_NO_OPT   := $(CCOMP) $(OPTS_NO_OPT) $(UMFOPTS) $(FLAGS) $(CMUMPS) $(INC) 
CL   := $(CCOMP) $(OPTS) $(UMFOPTS) $(FLAGS)

objects = src/profile.o src/sbprofile.o src/egrad.o src/models.o src/qlens.o src/commands.o src/lens.o src/imgsrch.o src/pixelgrid.o \
				src/cg.o src/mcmchdr.o src/errors.o src/brent.o src/sort.o src/gauss.o src/romberg.o src/spline.o \
				src/trirectangle.o src/GregsMathHdr.o src/hyp_2F1.o src/cosmo.o \
				src/simplex.o src/powell.o src/mcmceval.o

mkdist_objects = src/mkdist.o
mkdist_shared_objects = src/GregsMathHdr.o src/errors.o src/mcmceval.o
cosmocalc_objects = src/cosmocalc.o
cosmocalc_shared_objects = src/errors.o src/spline.o src/romberg.o src/cosmo.o src/brent.o

qlens: $(objects) $(LIBDMUMPS)
	$(CL) -o bin/qlens $(OPTL) $(objects) $(LINKLIBS) $(UMFPACK) $(UMFLIBS) 

mkdist: $(mkdist_objects) $(mkdist_shared_objects)
	$(CC) -o bin/mkdist $(mkdist_objects) $(mkdist_shared_objects) -lm

cosmocalc: $(cosmocalc_objects)
	$(CC) -o bin/cosmocalc $(cosmocalc_objects) $(cosmocalc_shared_objects) -lm

mumps:
	(cd MUMPS_5.0.1; $(MAKE))

src/qlens.o: src/qlens.cpp include/qlens.h
	$(CC) -o src/qlens.o -c src/qlens.cpp

src/commands.o: src/commands.cpp include/qlens.h include/lensvec.h include/profile.h include/sbprofile.h include/egrad.h include/pixelgrid.h
	$(CC_NO_OPT) -o src/commands.o -c src/commands.cpp

src/lens.o: src/lens.cpp include/profile.h include/sbprofile.h include/qlens.h include/pixelgrid.h include/lensvec.h include/matrix.h include/simplex.h include/powell.h include/mcmchdr.h include/cosmo.h
	$(CC) -o src/lens.o -c src/lens.cpp

src/imgsrch.o: src/imgsrch.cpp include/qlens.h include/lensvec.h
	$(CC) -o src/imgsrch.o -c src/imgsrch.cpp

src/pixelgrid.o: src/pixelgrid.cpp include/profile.h include/sbprofile.h include/lensvec.h include/pixelgrid.h include/qlens.h include/matrix.h include/cg.h include/egrad.h
	$(CC) -o src/pixelgrid.o -c src/pixelgrid.cpp

src/cg.o: src/cg.cpp include/cg.h
	$(CC) -o src/cg.o -c src/cg.cpp

src/mcmchdr.o: src/mcmchdr.cpp include/mcmchdr.h include/GregsMathHdr.h include/random.h
	$(CC) -o src/mcmchdr.o -c src/mcmchdr.cpp

src/egrad.o: src/egrad.cpp include/egrad.h 
	$(CC) -o src/egrad.o -c src/egrad.cpp

src/profile.o: include/profile.h src/profile.cpp include/lensvec.h include/egrad.h
	$(CC) -o src/profile.o -c src/profile.cpp

src/models.o: src/models.cpp include/profile.h 
	$(CC) -o src/models.o -c src/models.cpp

src/sbprofile.o: src/sbprofile.cpp include/sbprofile.h include/egrad.h
	$(CC) -o src/sbprofile.o -c src/sbprofile.cpp

src/errors.o: src/errors.cpp include/errors.h
	$(CC) -o src/errors.o -c src/errors.cpp

src/brent.o: include/brent.h src/brent.cpp
	$(CC) -o src/brent.o -c src/brent.cpp

src/simplex.o: include/simplex.h include/rand.h src/simplex.cpp
	$(CC) -o src/simplex.o -c src/simplex.cpp

src/powell.o: include/powell.h src/powell.cpp
	$(CC) -o src/powell.o -c src/powell.cpp

src/sort.o: include/sort.h src/sort.cpp
	$(CC) -o src/sort.o -c src/sort.cpp

src/gauss.o: src/gauss.cpp include/gauss.h
	$(CC) -o src/gauss.o -c src/gauss.cpp

src/romberg.o: src/romberg.cpp include/romberg.h
	$(CC) -o src/romberg.o -c src/romberg.cpp

src/spline.o: src/spline.cpp include/spline.h include/errors.h
	$(CC) -o src/spline.o -c src/spline.cpp

src/trirectangle.o: src/trirectangle.cpp include/lensvec.h include/trirectangle.h
	$(CC) -o src/trirectangle.o -c src/trirectangle.cpp

src/GregsMathHdr.o: src/GregsMathHdr.cpp include/GregsMathHdr.h
	$(CC) -o src/GregsMathHdr.o -c src/GregsMathHdr.cpp

src/mcmceval.o: src/mcmceval.cpp include/mcmceval.h include/GregsMathHdr.h include/random.h include/errors.h
	$(CC) -o src/mcmceval.o -c src/mcmceval.cpp

src/mkdist.o: src/mkdist.cpp include/mcmceval.h include/errors.h
	$(CC) -o src/mkdist.o -c src/mkdist.cpp

src/hyp_2F1.o: src/hyp_2F1.cpp include/hyp_2F1.h include/complex_functions.h
	$(CC) -o src/hyp_2F1.o -c src/hyp_2F1.cpp

src/cosmocalc.o: src/cosmocalc.cpp include/errors.h include/cosmo.h
	$(CC) -o src/cosmocalc.o -c src/cosmocalc.cpp

src/cosmo.o: src/cosmo.cpp include/cosmo.h
	$(CC) -o src/cosmo.o -c src/cosmo.cpp

clean_qlens:
	rm qlens $(objects)

clean:
	rm qlens mkdist cosmocalc $(objects) $(mkdist_objects) $(cosmocalc_objects)

clmain:
	rm src/qlens.o

vim_run:
	qlens

