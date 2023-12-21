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

pushd hdf5-${h_ver}

./configure --prefix=/opt/hdf5/${h_ver}/NVHPC/${1} --enable-fortran --enable-cxx --with-szlib=/usr/lib64 \
            --with-default-api-version=v114

make 
make check
make install
popd

