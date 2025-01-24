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

# Version without MUMPS
default: qlens mkdist cosmocalc 
CCOMP = g++
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

objects = profile.o sbprofile.o egrad.o models.o qlens.o commands.o params.o modelparams.o lenscalc.o \
				lens.o imgsrch.o pixelgrid.o cg.o mcmchdr.o errors.o brent.o sort.o gauss.o \
				romberg.o spline.o trirectangle.o GregsMathHdr.o hyp_2F1.o cosmo.o \
				simplex.o powell.o mcmceval.o

mkdist_objects = mkdist.o
mkdist_shared_objects = GregsMathHdr.o errors.o mcmceval.o
cosmocalc_objects = cosmocalc.o
cosmocalc_shared_objects = errors.o spline.o romberg.o modelparams.o cosmo.o brent.o

qlens: $(objects) $(LIBDMUMPS)
	$(CL) -o qlens $(OPTL) $(objects) $(LINKLIBS) $(UMFPACK) $(UMFLIBS) 

mkdist: $(mkdist_objects) $(mkdist_shared_objects)
	$(CC) -o mkdist $(mkdist_objects) $(mkdist_shared_objects) -lm

cosmocalc: $(cosmocalc_objects)
	$(CC) -o cosmocalc $(cosmocalc_objects) $(cosmocalc_shared_objects) -lm

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

mcmchdr.o: mcmchdr.cpp mcmchdr.h GregsMathHdr.h random.h
	$(CC) -c mcmchdr.cpp

egrad.o: egrad.cpp egrad.h 
	$(CC) -c egrad.cpp

profile.o: profile.h profile.cpp lensvec.h egrad.h
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

gauss.o: gauss.cpp gauss.h
	$(CC) -c gauss.cpp

romberg.o: romberg.cpp romberg.h
	$(CC) -c romberg.cpp

spline.o: spline.cpp spline.h errors.h
	$(CC) -c spline.cpp

trirectangle.o: trirectangle.cpp lensvec.h trirectangle.h
	$(CC) -c trirectangle.cpp

GregsMathHdr.o: GregsMathHdr.cpp GregsMathHdr.h
	$(CC) -c GregsMathHdr.cpp

mcmceval.o: mcmceval.cpp mcmceval.h GregsMathHdr.h random.h errors.h
	$(CC) -c mcmceval.cpp

mkdist.o: mkdist.cpp mcmceval.h errors.h
	$(CC) -c mkdist.cpp

hyp_2F1.o: hyp_2F1.cpp hyp_2F1.h complex_functions.h
	$(CC) -c hyp_2F1.cpp

cosmocalc.o: cosmocalc.cpp errors.h cosmo.h modelparams.h
	$(CC) -c cosmocalc.cpp

cosmo.o: cosmo.cpp cosmo.h
	$(CC) -c cosmo.cpp

clean_qlens:
	rm qlens $(objects)

clean:
	rm qlens mkdist cosmocalc $(objects) $(mkdist_objects) $(cosmocalc_objects)

clmain:
	rm qlens.o

vim_run:
	qlens

