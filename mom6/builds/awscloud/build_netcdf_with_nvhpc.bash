  # test the compiler
  which nvfortran
  #nvfortran -mp=gpu -stdpar /source/platforms/samples/gpu/gpu_offload_test2d.f90 -o gpu_offload_test2d
  #
  export installDir='/contrib/Niki.Zadeh/opt'
  #Build hdf5
  export CC='/contrib/Niki.Zadeh/spack/var/spack/environments/nvidia3/.spack-env/view/Linux_x86_64/23.9/comm_libs/mpi/bin/mpicc'
  export FC='/contrib/Niki.Zadeh/spack/var/spack/environments/nvidia3/.spack-env/view/Linux_x86_64/23.9/comm_libs/mpi/bin/mpif90'

  export h_ver='1.12.2'
  export HDF5_ROOT=${installDir}/hdf5/${h_ver}/NVHPC

  export CPP='cpp -E'
  export CXXCPP='cpp -E'
  export FCFLAGS="-fPIC -g"
  export CFLAGS="-g"

#  wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_2/source/hdf5-1.12.2.tar.gz
#  \rm -rf hdf5-${h_ver}
#  tar xf hdf5-${h_ver}.tar.gz
#  cd hdf5-${h_ver}
#  ./configure --prefix=${HDF5_ROOT} --enable-fortran --enable-cxx --with-szlib=/usr/lib64 
#  make 
## make check
#  make install
## --prefix or --exec-prefix did not work and I had to make a sym link to creat the HDF5_ROOT
## cd /contrib/Niki.Zadeh/opt/hdf5/1.12.2;  ln -s ../../hdf5-1.12.2/hdf5 NVHPC
#
#  cd ..
  #cleanup
#  \rm -rf hdf5-${h_ver} hdf5-${h_ver}.tar.gz
  #
  #build netcdf-c
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

  export c_ver='4.8.1'
#  wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.8.1.tar.gz
#  \rm -rf netcdf-c-${c_ver}
#  tar xf  v${c_ver}.tar.gz
#  cd   netcdf-c-${c_ver}
#  ./configure --prefix=${installDir}/netcdf/${c_ver}/NVHPC/${1} --enable-shared --disable-dap --enable-netcdf-4 \
#                  --libdir=${installDir}/netcdf/${c_ver}/NVHPC/${1}/lib64 --disable-libxml2
#  make
## make check
#  make install
#  cd ..
##  exit 0
  #Build netcdf-fortran
  export f_ver='4.6.1'
#  wget https://downloads.unidata.ucar.edu/netcdf-fortran/4.6.1/netcdf-fortran-4.6.1.tar.gz
  \rm -rf netcdf-fortran-${f_ver}
  tar xf netcdf-fortran-${f_ver}.tar.gz
  cd netcdf-fortran-${f_ver}
#  echo "\nNVHPC VERSIONS"
  which nvc
  which nvfortran
  pwd
  sleep 5
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${installDir}/netcdf/${c_ver}/NVHPC/${1}/lib64/
  ./configure --prefix=${installDir}/netcdf/${c_ver}/NVHPC/${1} --libdir=${installDir}/netcdf/${c_ver}/NVHPC/${1}/lib64/ \
                  CPPFLAGS=-I${installDir}/netcdf/${c_ver}/NVHPC/${1}/include/ LDFLAGS=-L${installDir}/netcdf/${c_ver}/NVHPC/${1}/lib64
  make
## make check
  make install
  cd ..
  #cleanup
  pwd
#  \rm -rf hdf5-${h_ver} hdf5-${h_ver}.tar.gz v${c_ver}.tar.gz netcdf-c-${c_ver}
#  \rm -rf netcdf-fortran-${f_ver} netcdf-fortran-${f_ver}.tar.gz
