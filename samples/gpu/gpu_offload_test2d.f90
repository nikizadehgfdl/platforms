!This is a quick test for openmp
!To compile and run do:
!\rm ./gpu_offload_test2d; nvfortran -mp=gpu gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!\rm ./gpu_offload_test2d; ifx -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
program test_omp
  implicit none
  include 'omp_lib.h'        
  integer, parameter :: m=10000,n=10000, iter_max=2000
  integer :: i, j, iter, itermax
  real, parameter :: pi=2.0*asin(1.0)
  real, parameter :: tol=1e-10
  real, dimension (:,:), allocatable :: A, B,A2
  real, dimension (:),  allocatable :: y0
  real :: sum0
  real :: res1  =0.000006629710014
  real :: res100=0.000000663121053
  real*8 :: run_time
  integer :: nthread, nthreads,nthr
  character (len = 50) :: subname
  nthread = 1


  allocate ( A(0:m-1,0:n-1),B(0:m-1,0:n-1) )
  allocate ( y0(0:n-1) )
  allocate ( A2(0:m-1,0:n-1))
  A=0.0; B=0.0
  ! Set B.C.
  y0 = sin(pi* (/ (j,j=0,n-1) /) /(n-1))
  A(:,0)   = 0.0
  A(:,n-1) = 0.0
  A(0,:)   = y0(:)
  A(m-1,:) = y0(:)*exp(-pi)
  do i=1,m-1
    B(i,0)   = 0.0
    B(i,n-1) = 0.0
  end do
  do j=1,n-1
    B(0,j)   = y0(j)
    B(m-1,j) = y0(j)*exp(-pi)
  end do

  sum0=sum(A)/n/m

  write(*,'(a)')  '      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine'

!The cpu openmp will be too slow (1000x) for large size arrays.
!  A2(:,:)=A(:,:)
!  nthread=1
!!$   run_time = omp_get_wtime()
!   call benchmark2d_omp(nthread, iter_max, m, n, A2)
!   subname='benchmark2d_omp'
!!$   run_time = omp_get_wtime() - run_time;
!   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

   A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=1 
   call benchmark2d_omp_gpu(nthread, iter_max, m, n, A2)
   subname='benchmark2d_omp_gpu'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

   A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=1 
   call benchmark2d_docon(nthread, iter_max, m, n, A2)
   subname='benchmark2d_docon'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname


   deallocate (A, B, y0, A2)
end program test_omp

subroutine benchmark2d_omp(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter,nthr
!$ call omp_set_dynamic(0)
!$ call omp_set_num_threads(nthread)

!$omp parallel 
!$  nthr = OMP_GET_NUM_THREADS();print *, ' We are using',nthr,' thread(s)'
!$omp end parallel
!$omp parallel do private(iter)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
!$omp end parallel do
end subroutine benchmark2d_omp

subroutine benchmark2d_omp0(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
!$ call omp_set_dynamic(0);
!$ call omp_set_num_threads(nthread);
!$omp parallel do private(iter) shared(A,iter_max)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
!$omp end parallel do
end subroutine benchmark2d_omp0

subroutine benchmark2d0(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine benchmark2d0

subroutine benchmark2d_omp_gpu(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

!$omp target parallel do collapse(2) map(tofrom: A(0:m-1,0:n-1)) private(iter)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine benchmark2d_omp_gpu

subroutine benchmark2d_docon(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
  do concurrent(j=0:n-1,i=0:m-1)
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
  enddo
end subroutine benchmark2d_docon

!Some results
!On Intel devcloud platform on 12/22/2022
!-O0
!u182325@s019-n007:~/platforms/samples/gpu/openmp$ \rm ./gpu_offload_test2d; ifx -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine
!      16000000     8.940     200    0.000165991194081    0.000015737372451    1     benchmark2d0
!      16000000     4.999     200    0.000165991194081    0.000015737372451    1     benchmark2d_omp_gpu
!      16000000     1.165     200    0.000165991194081    0.000015737372451    1     benchmark2d_docon
!-O2
!u182325@s019-n007:~/platforms/samples/gpu/openmp$ \rm ./gpu_offload_test2d; ifx -O2 -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine
!      16000000     4.244     200    0.000165990830283    0.000015737356080    1     benchmark2d0
!      16000000     0.727     200    0.000165990830283    0.000015737356080    1     benchmark2d_omp_gpu
!      16000000     0.069     200    0.000165990830283    0.000015737356080    1     benchmark2d_docon        
!u182325@s019-n007:~/platforms/samples/gpu/openmp$ \rm ./gpu_offload_test2d; ifx -O2 -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine
!      16000000     1.355    2000    0.000165990830283    0.000006629713880    1     benchmark2d_omp_gpu
!      16000000     0.502    2000    0.000165990830283    0.000006629713880    1     benchmark2d_docon
!u182325@s019-n007:~/platforms/samples/gpu/openmp$ \rm ./gpu_offload_test2d; ifx -O2 -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine
!     100000000     4.387    2000    0.000066406377300    0.000002652283911    1     benchmark2d_omp_gpu
!     100000000     3.140    2000    0.000066406377300    0.000002652283911    1     benchmark2d_docon
!u182325@s019-n007:~/platforms/samples/gpu/openmp$ \rm ./gpu_offload_test2d; ifx -O2 -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine
!     400000000    14.626    2000    0.000033204814827    0.000001326215852    1     benchmark2d_omp_gpu
!     400000000    12.551    2000    0.000033204814827    0.000001326215852    1     benchmark2d_docon   
!
!     16000000 bombs
!
!On gfdl gpubox on 12/22/2022
!Niki.Zadeh: ~/platforms/samples/gpu/openmp $ \rm ./gpu_offload_test2d; nvfortran -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine
!      16000000     6.226     200    0.000165991179529    0.000015737370632    1     benchmark2d0                                      
!      16000000     0.024     200    0.000165991179529    0.000015737370632    1     benchmark2d_omp_gpu                               
!      16000000     0.024     200    0.000165991179529    0.000015737370632    1     benchmark2d_docon        
!-O2
!Niki.Zadeh: ~/platforms/samples/gpu/openmp $ \rm ./gpu_offload_test2d; nvfortran -O2 -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads     subroutine
!      16000000     6.218     200    0.000165991063113    0.000015737357899    1      benchmark2d0                                      
!      16000000     0.023     200    0.000165991063113    0.000015737357899    1      benchmark2d_omp_gpu                               
!      16000000     0.024     200    0.000165991063113    0.000015737357899    1      benchmark2d_docon  
!Niki.Zadeh: ~/platforms/samples/gpu/openmp $ \rm ./gpu_offload_test2d; nvfortran -O2 -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads     subroutine
!     100000000     0.245    2000    0.000066406464612    0.000002652284593    1      benchmark2d_omp_gpu                               
!     100000000     0.144    2000    0.000066406464612    0.000002652284593    1      benchmark2d_docon  
!Niki.Zadeh: ~/platforms/samples/gpu/openmp $ \rm ./gpu_offload_test2d; nvfortran -O2 -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads     subroutine
!     400000000     0.822    2000    0.000033204873034    0.000001326215056    1      benchmark2d_omp_gpu                               
!     400000000     0.430    2000    0.000033204873034    0.000001326215056    1      benchmark2d_docon                                 
!x100
!Niki.Zadeh: ~/platforms/samples/gpu/openmp $ \rm ./gpu_offload_test2d; nvfortran -O2 -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d;./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads     subroutine
!    1600000000     3.203    2000    0.000016602891264    0.000000663109006    1      benchmark2d_omp_gpu                               
!    1600000000     1.919    2000    0.000016602891264    0.000000663109006    1      benchmark2d_docon   
