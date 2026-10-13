FROM gcc:10

ARG LIBDIR=/home/libraries
ARG SRCDIR=/home/src

WORKDIR $SRCDIR


# 3rd party dependencies
#########################################################################################################
# Install gfortran, blas, lapack, python
RUN apt update \
    && apt install -y libblas-dev liblapack-dev python3 libsuitesparse-dev

# Install openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz \
    &&  tar -xvf openmpi-4.1.1.tar.gz \
    &&  cd openmpi-4.1.1 \
    &&  ./configure --prefix=$LIBDIR/openmpi \
    &&  make all install

# Install cfitsio
RUN wget https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-4.6.3.tar.gz \
    &&	tar -xvf cfitsio-4.6.3.tar.gz \
    &&	mv cfitsio-4.6.3 cfitsio \
    &&	cd cfitsio \
    &&	./configure --prefix=$LIBDIR/cfitsio \
    &&	make \
    &&	make install

# Install CCfits
RUN wget https://heasarc.gsfc.nasa.gov/docs/software/fitsio/CCfits-2.5/CCfits-2.5.tar.gz \
    &&	tar -xvf CCfits-2.5.tar.gz \
    &&	cd CCfits \
    &&	./configure --prefix=$LIBDIR/CCfits --with-cfitsio=$LIBDIR/cfitsio \
    &&	make \
    &&	make install

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.5/cmake-3.15.5-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    &&	chmod u+x /tmp/cmake-install.sh \
    &&	mkdir /usr/bin/cmake \
    &&	/tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
    &&	rm /tmp/cmake-install.sh
ENV PATH=$PATH:/usr/bin/cmake/bin

# Install Multinest
RUN git clone https://github.com/JohannesBuchner/MultiNest.git \
    &&  cd MultiNest/build \
    &&  cmake -DCMAKE_PREFIX_PATH=$LIBDIR/openmpi -DCMAKE_INSTALL_PREFIX=$LIBDIR/multinest .. \
    &&  make \
    &&  make install

# Install PolyChord
RUN git clone https://github.com/PolyChord/PolyChordLite.git
#    &&	cd PolyChordLite \
#    &&	make MPI=1
#    &&	python3 setup.py build

# Install jsoncpp
#RUN git clone https://github.com/open-source-parsers/jsoncpp.git \
#    &&	cd jsoncpp \
#    &&	mkdir -p build \
#    &&	cd build \
#    &&	cmake \
#    -DCMAKE_BUILD_TYPE=release \
#    -DBUILD_SHARED_LIBS=ON \
#    -DARCHIVE_INSTALL_DIR=. \
#    -DCMAKE_INSTALL_PREFIX=$LIBDIR/jsoncpp \
#    -G "Unix Makefiles" \
#    .. \
#    &&	make \
#    &&	make install



ENV PATH=$PATH:$LIBDIR/openmpi/bin
#:$LIBDIR/MultiNest/bin
ENV CPATH=$CPATH:$LIBDIR/cfitsio/include:$LIBDIR/CCfits/include:$LIBDIR/openmpi/include:$LIBDIR/multinest/include
#:$LIBDIR/gmp/include:$LIBDIR/mpfr/include:$LIBDIR/boost/include:$LIBDIR/CGAL/include:$LIBDIR/jsoncpp/include:$LIBDIR/gsl/include:$LIBDIR/eigen/include/eigen3
ENV LIBRARY_PATH=$LIBRARY_PATH:$LIBDIR/cfitsio/lib:$LIBDIR/CCfits/lib:$LIBDIR/openmpi/lib:$LIBDIR/multinest/lib
#:$LIBDIR/gmp/lib:$LIBDIR/mpfr/lib:$LIBDIR/boost/lib:$LIBDIR/CGAL/lib:$LIBDIR/jsoncpp/lib:$LIBDIR/gsl/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBDIR/cfitsio/lib:$LIBDIR/CCfits/lib:$LIBDIR/openmpi/lib:$LIBDIR/multinest/lib
#:$LIBDIR/gmp/lib:$LIBDIR/mpfr/lib:$LIBDIR/boost/lib:$LIBDIR/CGAL/lib:$LIBDIR/jsoncpp/lib:$LIBDIR/gsl/lib
#########################################################################################################




### Uncomment the line below to use cache up to here and update the libraries and code with their latest versions
#ARG DUMMY_DATE=2011-11-04


# Install 'gerlumphpp' and 'vkl_lib' libraries
#########################################################################################################
# Install vkl_lib
#RUN cd /home/ \
#    &&	git clone https://github.com/qminor/qlens-beta.git \
#    &&  mkdir qlens-beta/build \
#    &&	cd qlens-beta/build 
#    &&	cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
#########################################################################################################

WORKDIR /home/qlens-beta
