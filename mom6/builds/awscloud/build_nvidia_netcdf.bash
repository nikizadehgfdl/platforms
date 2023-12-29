#!/bin/bash

if [ ${1} = "" ] ; then
  echo "must contain an argument X.X.X for the version of NVidia HPC SDK you want to build with"
  exit
fi

export CC='mpicc'
export FC='mpif90'

h_ver=1.12.2

export CPP='cpp -E'
export CXXCPP='cpp -E'
export FCFLAGS="-fPIC -g"
export CFLAGS="-g"

wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_2/source/hdf5-1.12.2.tar.gz

\rm -rf hdf5-${h_ver}
tar xf hdf5-${h_ver}.tar.gz

cd hdf5-${h_ver}

./configure --prefix=/opt/hdf5/${h_ver}/NVHPC/${1} --enable-fortran --enable-cxx --with-szlib=/usr/lib64 --with-default-api-version=v114

make 
make check
make install
cd ..

export HDF5_ROOT='/home/hdf5-1.12.2/hdf5'
export CFLAGS='-O2 -fPIC'
export CXXFLAGS='-O2'
export FFLAGS='-O2 -fPIC'
export F90FLAGS='-O2 -fPIC'
export FCFLAGS='-O2 -fPIC'
export CPP='cpp -E'
export CXXCPP='cpp -E'
export CPPFLAGS=-I${HDF5_ROOT}/include
export LDFLAGS=-L${HDF5_ROOT}/lib
export RUN_LIBRARY_PATH=-L${HDF5_ROOT}/lib

wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.8.1.tar.gz
\rm -rf netcdf-c-${c_ver}
tar xf  v${c_ver}.tar.gz
pushd   netcdf-c-${c_ver}
./configure --prefix=/opt/netcdf/${c_ver}/NVHPC/${1} --enable-shared --disable-dap --enable-netcdf-4 \
                  --libdir=/opt/netcdf/${c_ver}/NVHPC/${1}/lib64 --disable-libxml2
make
make check
make install
cd ..


wget https://downloads.unidata.ucar.edu/netcdf-fortran/4.6.1/netcdf-fortran-4.6.1.tar.gz
\rm -rf netcdf-fortran-${f_ver}
tar xf netcdf-fortran-${f_ver}.tar.gz
cd netcdf-fortran-${f_ver}
echo "\nNVHPC VERSIONS"
which nvc
which nvfortran
sleep 5
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/netcdf/${c_ver}/NVHPC/${1}/lib64/
./configure --prefix=/opt/netcdf/${c_ver}/NVHPC/${1} --libdir=/opt/netcdf/${c_ver}/NVHPC/${1}/lib64/ \
                  CPPFLAGS=-I/opt/netcdf/${c_ver}/NVHPC/${1}/include/ LDFLAGS=-L/opt/netcdf/${c_ver}/NVHPC/${1}/lib64
make
make check
make install
cd ..

