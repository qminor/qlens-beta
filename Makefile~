# This is the most basic compilation without any additional packages

SRC_DIR := ./src
BIN_DIR := ./bin
PYTHON_DIR := ./python/qlens
BUILD_DIR := ./build
OBJ_DIR := $(BUILD_DIR)/obj
INCLUDE_DIR := ./include

MODULE := $(PYTHON_DIR)/qlens$(shell python3-config --extension-suffix)
PYBIND11_CFLAGS := $(shell python3 -m pybind11 --includes)

INC := -I$(INCLUDE_DIR)

default: mkbuilddir mkobjdir qlens mkdist qlens-wrap

CCOMP = g++
OPTS = -Wno-write-strings -Wno-vla-cxx-extension -O3 -std=c++17 
OPTS_NO_OPT = -Wno-write-strings -Wno-vla-cxx-extension -std=c++17 
OTHERLIBS = -lreadline -lm 

LINKLIBS = $(OTHERLIBS) 

CC   := $(CCOMP) $(OPTS) $(INC) $(FLAGS)
CC_NO_OPT   := $(CCOMP) $(OPTS_NO_OPT) $(INC) $(FLAGS) -DUSE_READLINE 
CL   := $(CCOMP) $(OPTS) 

core_objects = $(OBJ_DIR)/cosmo.o $(OBJ_DIR)/simplex.o $(OBJ_DIR)/spline.o $(OBJ_DIR)/profile.o $(OBJ_DIR)/sbprofile.o $(OBJ_DIR)/egrad.o $(OBJ_DIR)/models.o \
					$(OBJ_DIR)/params.o $(OBJ_DIR)/modelparams.o $(OBJ_DIR)/lenscalc.o $(OBJ_DIR)/lens.o $(OBJ_DIR)/imgsrch.o $(OBJ_DIR)/pixelgrid.o \
					$(OBJ_DIR)/cg.o $(OBJ_DIR)/mcmchdr.o $(OBJ_DIR)/errors.o $(OBJ_DIR)/brent.o $(OBJ_DIR)/sort.o $(OBJ_DIR)/gauss.o $(OBJ_DIR)/romberg.o \
					$(OBJ_DIR)/trirectangle.o $(OBJ_DIR)/GregsMathHdr.o $(OBJ_DIR)/hyp_2F1.o $(OBJ_DIR)/powell.o $(OBJ_DIR)/mcmceval.o

wrapper_objects = $(OBJ_DIR)/qlens_export.o $(OBJ_DIR)/qlens_wrapper.o $(core_objects)

objects = $(OBJ_DIR)/commands.o $(OBJ_DIR)/qlens.o $(core_objects)

mkdist_objects = $(OBJ_DIR)/mkdist.o
mkdist_shared_objects = $(OBJ_DIR)/GregsMathHdr.o $(OBJ_DIR)/errors.o $(OBJ_DIR)/mcmceval.o

mkbuilddir:
	mkdir -p $(BUILD_DIR)

mkobjdir:
	mkdir -p $(OBJ_DIR)

qlens: $(objects) $(LIBDMUMPS) 
	$(CL) -o $(BIN_DIR)/qlens $(objects) $(LINKLIBS) $(UMFPACK) $(UMFLIBS)  

qlens-wrap: $(wrapper_objects)
	$(CL) -shared $(wrapper_objects) $(LINKLIBS) -undefined dynamic_lookup -o $(MODULE)

mkdist: $(mkdist_objects) $(mkdist_shared_objects)
	$(CC) -o $(BIN_DIR)/mkdist $(mkdist_objects) $(mkdist_shared_objects) -lm

$(OBJ_DIR)/cosmo.o: $(SRC_DIR)/cosmo.cpp $(INCLUDE_DIR)/cosmo.h
	$(CC) -c $(SRC_DIR)/cosmo.cpp -o $(OBJ_DIR)/cosmo.o

$(OBJ_DIR)/qlens_wrapper.o: $(SRC_DIR)/qlens_wrapper.cpp
	$(CC) $(PYBIND11_CFLAGS) -c $(SRC_DIR)/qlens_wrapper.cpp -o $(OBJ_DIR)/qlens_wrapper.o

$(OBJ_DIR)/qlens_export.o: $(SRC_DIR)/qlens_export.cpp
	$(CC) $(PYBIND11_CFLAGS) -c $(SRC_DIR)/qlens_export.cpp -o $(OBJ_DIR)/qlens_export.o

$(OBJ_DIR)/qlens.o: $(SRC_DIR)/qlens.cpp $(INCLUDE_DIR)/qlens.h
	$(CC) -c $(SRC_DIR)/qlens.cpp -o $(OBJ_DIR)/qlens.o

$(OBJ_DIR)/commands.o: $(SRC_DIR)/commands.cpp $(INCLUDE_DIR)/qlens.h $(INCLUDE_DIR)/lensvec.h $(INCLUDE_DIR)/profile.h $(INCLUDE_DIR)/sbprofile.h $(INCLUDE_DIR)/params.h $(INCLUDE_DIR)/modelparams.h $(INCLUDE_DIR)/delaunay.h $(INCLUDE_DIR)/pixelgrid.h
	$(CC_NO_OPT) -c $(SRC_DIR)/commands.cpp -o $(OBJ_DIR)/commands.o

$(OBJ_DIR)/params.o: $(SRC_DIR)/params.cpp $(INCLUDE_DIR)/params.h 
	$(CC) -c $(SRC_DIR)/params.cpp -o $(OBJ_DIR)/params.o

$(OBJ_DIR)/modelparams.o: $(SRC_DIR)/modelparams.cpp $(INCLUDE_DIR)/modelparams.h 
	$(CC) -c $(SRC_DIR)/modelparams.cpp -o $(OBJ_DIR)/modelparams.o

$(OBJ_DIR)/lenscalc.o: $(SRC_DIR)/lenscalc.cpp $(INCLUDE_DIR)/qlens.h $(INCLUDE_DIR)/lensvec.h
	$(CC) -c $(SRC_DIR)/lenscalc.cpp -o $(OBJ_DIR)/lenscalc.o

$(OBJ_DIR)/lens.o: $(SRC_DIR)/lens.cpp $(INCLUDE_DIR)/profile.h $(INCLUDE_DIR)/sbprofile.h $(INCLUDE_DIR)/qlens.h $(INCLUDE_DIR)/pixelgrid.h $(INCLUDE_DIR)/lensvec.h $(INCLUDE_DIR)/matrix.h $(INCLUDE_DIR)/simplex.h $(INCLUDE_DIR)/powell.h $(INCLUDE_DIR)/mcmchdr.h $(INCLUDE_DIR)/cosmo.h $(INCLUDE_DIR)/delaunay.h $(INCLUDE_DIR)/modelparams.h $(INCLUDE_DIR)/params.h
	$(CC) -c $(SRC_DIR)/lens.cpp -o $(OBJ_DIR)/lens.o

$(OBJ_DIR)/imgsrch.o: $(SRC_DIR)/imgsrch.cpp $(INCLUDE_DIR)/qlens.h $(INCLUDE_DIR)/lensvec.h
	$(CC) -c $(SRC_DIR)/imgsrch.cpp -o $(OBJ_DIR)/imgsrch.o

$(OBJ_DIR)/pixelgrid.o: $(SRC_DIR)/pixelgrid.cpp $(INCLUDE_DIR)/profile.h $(INCLUDE_DIR)/sbprofile.h $(INCLUDE_DIR)/lensvec.h $(INCLUDE_DIR)/pixelgrid.h $(INCLUDE_DIR)/qlens.h $(INCLUDE_DIR)/matrix.h $(INCLUDE_DIR)/cg.h $(INCLUDE_DIR)/modelparams.h
	$(CC) -c $(SRC_DIR)/pixelgrid.cpp -o $(OBJ_DIR)/pixelgrid.o

$(OBJ_DIR)/cg.o: $(SRC_DIR)/cg.cpp $(INCLUDE_DIR)/cg.h
	$(CC) -c $(SRC_DIR)/cg.cpp -o $(OBJ_DIR)/cg.o

$(OBJ_DIR)/mcmchdr.o: $(SRC_DIR)/mcmchdr.cpp $(INCLUDE_DIR)/mcmchdr.h $(INCLUDE_DIR)/GregsMathHdr.h $(INCLUDE_DIR)/random.h
	$(CC) -c $(SRC_DIR)/mcmchdr.cpp -o $(OBJ_DIR)/mcmchdr.o

$(OBJ_DIR)/profile.o: $(INCLUDE_DIR)/profile.h $(SRC_DIR)/profile.cpp $(INCLUDE_DIR)/lensvec.h
	$(CC) -c $(SRC_DIR)/profile.cpp -o $(OBJ_DIR)/profile.o

$(OBJ_DIR)/models.o: $(INCLUDE_DIR)/profile.h $(SRC_DIR)/models.cpp
	$(CC) -c $(SRC_DIR)/models.cpp -o $(OBJ_DIR)/models.o

$(OBJ_DIR)/sbprofile.o: $(INCLUDE_DIR)/sbprofile.h $(SRC_DIR)/sbprofile.cpp
	$(CC) -c $(SRC_DIR)/sbprofile.cpp -o $(OBJ_DIR)/sbprofile.o

$(OBJ_DIR)/errors.o: $(SRC_DIR)/errors.cpp $(INCLUDE_DIR)/errors.h
	$(CC) -c $(SRC_DIR)/errors.cpp -o $(OBJ_DIR)/errors.o

$(OBJ_DIR)/brent.o: $(INCLUDE_DIR)/brent.h $(SRC_DIR)/brent.cpp
	$(CC) -c $(SRC_DIR)/brent.cpp -o $(OBJ_DIR)/brent.o

$(OBJ_DIR)/simplex.o: $(INCLUDE_DIR)/simplex.h $(INCLUDE_DIR)/rand.h $(SRC_DIR)/simplex.cpp
	$(CC) -c $(SRC_DIR)/simplex.cpp -o $(OBJ_DIR)/simplex.o

$(OBJ_DIR)/powell.o: $(INCLUDE_DIR)/powell.h $(SRC_DIR)/powell.cpp
	$(CC) -c $(SRC_DIR)/powell.cpp -o $(OBJ_DIR)/powell.o

$(OBJ_DIR)/sort.o: $(INCLUDE_DIR)/sort.h $(SRC_DIR)/sort.cpp
	$(CC) -c $(SRC_DIR)/sort.cpp -o $(OBJ_DIR)/sort.o

$(OBJ_DIR)/gauss.o: $(SRC_DIR)/gauss.cpp $(INCLUDE_DIR)/gauss.h
	$(CC) -c $(SRC_DIR)/gauss.cpp -o $(OBJ_DIR)/gauss.o

$(OBJ_DIR)/romberg.o: $(SRC_DIR)/romberg.cpp $(INCLUDE_DIR)/romberg.h
	$(CC) -c $(SRC_DIR)/romberg.cpp -o $(OBJ_DIR)/romberg.o

$(OBJ_DIR)/spline.o: $(SRC_DIR)/spline.cpp $(INCLUDE_DIR)/spline.h $(INCLUDE_DIR)/errors.h
	$(CC) -c $(SRC_DIR)/spline.cpp -o $(OBJ_DIR)/spline.o

$(OBJ_DIR)/trirectangle.o: $(SRC_DIR)/trirectangle.cpp $(INCLUDE_DIR)/lensvec.h $(INCLUDE_DIR)/trirectangle.h
	$(CC) -c $(SRC_DIR)/trirectangle.cpp -o $(OBJ_DIR)/trirectangle.o

$(OBJ_DIR)/GregsMathHdr.o: $(SRC_DIR)/GregsMathHdr.cpp $(INCLUDE_DIR)/GregsMathHdr.h
	$(CC) -c $(SRC_DIR)/GregsMathHdr.cpp -o $(OBJ_DIR)/GregsMathHdr.o

$(OBJ_DIR)/egrad.o: $(INCLUDE_DIR)/egrad.h $(SRC_DIR)/egrad.cpp $(INCLUDE_DIR)/lensvec.h $(INCLUDE_DIR)/qlens.h 
	$(CC) -c $(SRC_DIR)/egrad.cpp -o $(OBJ_DIR)/egrad.o

$(OBJ_DIR)/mcmceval.o: $(SRC_DIR)/mcmceval.cpp $(INCLUDE_DIR)/mcmceval.h $(INCLUDE_DIR)/GregsMathHdr.h $(INCLUDE_DIR)/random.h $(INCLUDE_DIR)/errors.h
	$(CC) -c $(SRC_DIR)/mcmceval.cpp -o $(OBJ_DIR)/mcmceval.o

$(OBJ_DIR)/mkdist.o: $(SRC_DIR)/mkdist.cpp $(INCLUDE_DIR)/mcmceval.h $(INCLUDE_DIR)/errors.h
	$(CC) -c $(SRC_DIR)/mkdist.cpp -o $(OBJ_DIR)/mkdist.o

$(OBJ_DIR)/hyp_2F1.o: $(SRC_DIR)/hyp_2F1.cpp $(INCLUDE_DIR)/hyp_2F1.h $(INCLUDE_DIR)/complex_functions.h
	$(CC) -c $(SRC_DIR)/hyp_2F1.cpp -o $(OBJ_DIR)/hyp_2F1.o

clean_qlens:
	rm $(BIN_DIR)/qlens $(MODULE) $(wrapper_objects)

clean:
	rm $(BIN_DIR)/qlens $(MODULE) $(BIN_DIR)/kdist $(objects) $(mkdist_objects) $(wrapper_objects)

vim_run:
	qlens

