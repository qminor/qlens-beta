include ./suitesparse/SuiteSparse_config/SuiteSparse_config.mk

I_WITH_PARTITION = 
LIB_WITH_PARTITION =
CONFIG1 = -DNCHOLMOD
CONFIG2 = -DNPARTITION
ifeq (,$(findstring -DNCHOLMOD, $(UMFPACK_CONFIG)))
    # CHOLMOD is requested.  See if it is available
    ifeq (./suitesparse/CHOLMOD, $(wildcard ./suitesparse/CHOLMOD))
        ifeq (./suitesparse/COLAMD, $(wildcard ./suitesparse/COLAMD))
            # CHOLMOD and COLAMD are available
            CONFIG1 =
            LIB_WITH_CHOLMOD = ./suitesparse/CHOLMOD/Lib/libcholmod.a \
                ./suitesparse/COLAMD/Lib/libcolamd.a
            # check if METIS is requested and available
            ifeq (,$(findstring -DNPARTITION, $(CHOLMOD_CONFIG)))
                # METIS is requested.  See if it is available
                ifeq ($(METIS_PATH), $(wildcard $(METIS_PATH)))
                    ifeq (./suitesparse/CAMD, $(wildcard ./suitesparse/CAMD))
                        ifeq (./suitesparse/CCOLAMD, $(wildcard ./suitesparse/CCOLAMD))
                            # METIS, CAMD, and CCOLAMD are available
                            LIB_WITH_PARTITION = $(METIS) \
                                ./suitesparse/CCOLAMD/Lib/libccolamd.a \
                                ./suitesparse/CAMD/Lib/libcamd.a
                            I_WITH_PARTITION = -I$(METIS_PATH)/Lib \
                                -I./suitesparse/CCOLAMD/Include -I./suitesparse/CAMD/Include
                            CONFIG2 =
                        endif
                    endif
                endif
            endif
        endif
    endif
endif


UMFOPTS = $(CF) $(UMFPACK_CONFIG) $(CONFIG1) $(CONFIG2) \
    -I./suitesparse/UMFPACK/Include -I./suitesparse/AMD/Include -I./suitesparse/SuiteSparse_config

INC = ./suitesparse/UMFPACK/Include/umfpack.h ./suitesparse/AMD/Include/amd.h ./suitesparse/SuiteSparse_config/SuiteSparse_config.h

UMFLIBS = $(BLAS) $(XERBLA) $(LIB) $(LIB_WITH_CHOLMOD) $(LIB_WITH_PARTITION)

UMFPACK = ./suitesparse/UMFPACK/Lib/libumfpack.a ./suitesparse/AMD/Lib/libamd.a \
    ./suitesparse/SuiteSparse_config/libsuitesparseconfig.a \
    $(LIB_WITH_CHOLMOD) $(LIB_WITH_PARTITION)


