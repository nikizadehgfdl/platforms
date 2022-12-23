!This is a quick test for openmp
!To compile and run do:
!\rm ./test_omp_gpu; nvfortran -mp=gpu test_omp_gpu.f90 -o test_omp_gpu;./test_omp_gpu
!\rm ./test_omp_gpu; ifx -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 test_omp_gpu.f90 -o test_omp_gpu;./test_omp_gpu
program test_omp
  implicit none
  include 'omp_lib.h'        
  integer, parameter :: m=4000,n=4000, iter_max=200
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

  A2(:,:)=A(:,:)
  nthread=1
!$   run_time = omp_get_wtime()
   call benchmark2d0(nthread, iter_max, m, n, A2)
   subname='benchmark2d0'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

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
!On Intel devcloud platform
!-O0
!u182325@s019-n007:~/platforms/samples/gpu/openmp$ \rm ./test_omp_gpu; ifx -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 test_omp_gpu.f90 -o test_omp_gpu;./test_omp_gpu
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine
!      16000000     8.940     200    0.000165991194081    0.000015737372451    1     benchmark2d0
!      16000000     4.999     200    0.000165991194081    0.000015737372451    1     benchmark2d_omp_gpu
!      16000000     1.165     200    0.000165991194081    0.000015737372451    1     benchmark2d_docon
!-O2
!u182325@s019-n007:~/platforms/samples/gpu/openmp$ \rm ./test_omp_gpu; ifx -O2 -g -qopenmp -fopenmp-target-do-concurrent -fopenmp-targets=spir64 test_omp_gpu.f90 -o test_omp_gpu;./test_omp_gpu
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads    subroutine
!      16000000     4.244     200    0.000165990830283    0.000015737356080    1     benchmark2d0
!      16000000     0.727     200    0.000165990830283    0.000015737356080    1     benchmark2d_omp_gpu
!      16000000     0.069     200    0.000165990830283    0.000015737356080    1     benchmark2d_docon        
!
