!
!Very simple example of openmp offloading to GPU adopted from
!https://gist.github.com/anjohan/9ee746295ea1a00d9ca69415f40fafc9

!To compile on GPU box:
!No omp
!\rm ./example1_fl ; /home/Niki.Zadeh/opt/llvm/install/bin/flang example1.f90 -o example1_fl  -L/opt/gcc/11.3.0/lib64 ; time ./example1_fl
! size        runtime sum
! 100000000   0.000  687.194763183593750
!real	5m47.690s
!user	5m47.332s
!
!nvfortran  example1.f90 -o example1_nvf ; time ./example1_nvf
! size        runtime sum                omp_num_threads
! 100000000   0.000  687.194763183593750   0
!real	4m5.391s
!user	4m5.226s
!
!-O3
!/home/Niki.Zadeh/opt/llvm/install/bin/flang example1.f90 -O3 -L/opt/gcc/11.3.0/lib64 -o example1_fl ; time ./example1_fl
! size        runtime sum                omp_num_threads
! 100000000   0.000  687.194763183593750****
!real	0m20.021s
!user	0m19.890s
!
!gfortran -O3 example1.f90 -o example1_gf; time ./example1_gf
! size        runtime sum                omp_num_threads
! 100000000   0.000  687.194763183593750****
!real	0m20.037s
!user	0m19.903s
!
!-fopenmp
!/home/Niki.Zadeh/opt/llvm/install/bin/flang example1.f90 -O3 -L/opt/gcc/11.3.0/lib64 -fopenmp -o example1_fl_omp ; time ./example1_fl_omp
! size        runtime sum                omp_num_threads
! 100000000  24.786  687.194763183593750   1
!real	0m25.071s
!user	6m31.081s
!
!gfortran -O3 example1.f90 -fopenmp -o example1_gf_omp; time ./example1_gf_omp
! size        runtime sum                omp_num_threads
! 100000000  24.755  687.194763183593750   1
!real	0m24.903s
!user	6m30.967s
!
!nvfortran  example1.f90 -o example1_nvf ; time ./example1_nvf
! size        runtime sum                omp_num_threads
! 100000000   0.000  687.194763183593750   0
!real	4m5.391s
!user	4m5.226s
!
!nvfortran  example1.f90 -O3 -o example1_nvf ; time ./example1_nvf
! size        runtime sum                omp_num_threads
! 100000000   0.000 1505.299316406250000****         !!!!!-O3 does it wrong
!real	0m4.871s
!user	0m4.301s
!
!nvfortran -Mpreprocess  -ta=tesla,cc60 -mp  example1.f90 -o example1_nvf ;time ./example1_nvf
! size        runtime sum                omp_num_threads
! 100000000   5.812  687.194763183593750   1
!real	0m6.545s
!user	1m30.543s
!
!module load cuda/11.7
!module load nvhpc-no-mpi/22.5
!nvfortran -Mpreprocess  -ta=tesla,cc60,cuda11.7 -mp  example1.f90 -o example1_nvf_target ;time ./example1_nvf_target
! size        runtime sum                omp_num_threads
! 100000000   5.599  687.194763183593750   1            !!!GPU idle!!!
!real	0m6.116s
!user	1m28.413s
!
!nvfortran -Mpreprocess  -ta=tesla,cc70,cuda11.7 -mp=gpu  example1.f90 -o example1_nvf_target ;time ./example1_nvf_target
! size        runtime sum                omp_num_threads
! 100000000   0.544  687.194763183593750   1
!real	0m0.920s
!
!
!ACC
!pgf90 -acc -ta=nvidia:managed -Minfo=accel -o example1_acc_managed example1.f90; time ./example1_acc_managed
! size        runtime sum                omp_num_threads
! 100000000   0.000  687.194763183593750****
!real	0m1.261s
!user	0m0.491s
!
!DO CONCURRENT with GPU offload (Note -stdpar and comment out $omp target 
!nvfortran -mp -stdpar  example1.f90 -o example1_nvf_docon ;time ./example1_nvf_docon 
! size        runtime sum                omp_num_threads
! 100000000   0.237  687.194763183593750   1

program example1
  implicit none
  integer,parameter :: N = 1e8
  integer :: i,j
  real, dimension (:),  allocatable :: x,y
  real*8 :: run_time, omp_get_wtime
  integer :: nthread, OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM
  real :: sum

  allocate ( x(0:N-1) , y(0:N-1) ) ; x=0.0; y=0.0
!$   nthread = omp_get_num_threads()
!$   run_time = omp_get_wtime() 
!NOacc data copyin(x) copyout(y)
!NOacc kernels present(x,y)
!Nomp target teams distribute parallel do map(tofrom: x, y)
!  do i=0,N-1
  do concurrent(i=0:N-1)
    x(i) = i*1.
    do j=1,1000
      y(i) = y(i)+ 3.*x(i)/N
    enddo
  enddo
!NOacc end kernels
!NOacc end data

!$   run_time = omp_get_wtime() - run_time;
  
  sum=0.0
  do i=0,N-1; sum = sum+y(i); enddo

  write(*,'(a)') ' size        runtime sum                omp_num_threads'
  write(*,'(i10,f8.3,f21.15, i4)') N, run_time, sum/N,nthread 

  deallocate(x,y)
end program example1

