!This is a quick test for openmp
!To compile and run do:
!\rm ./test_omp_gpu; nvfortran -mp=gpu test_omp_gpu.f90 -o test_omp_gpu;./test_omp_gpu
program test_omp
  implicit none
  include 'omp_lib.h'        
  integer, parameter :: m=400,n=400, iter_max=20
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
!$   run_time = omp_get_wtime() 
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A2(i,j) = A2(i,j)*(A2(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
   subname='benchmark2d_inline'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

  A2(:,:)=A(:,:)
  nthread=1
!$   run_time = omp_get_wtime() 
!$   call omp_set_dynamic(.FALSE.)
!$   call omp_set_num_threads(nthread)
!$omp parallel 
!$  nthr = OMP_GET_NUM_THREADS();print *, ' We are using',nthr,' thread(s)'
!$omp do private(iter)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A2(i,j) = A2(i,j)*(A2(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
!$omp end do
!$omp end parallel
   subname='benchmark2d_omp_inline'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

  A2(:,:)=A(:,:)
  nthread=2
!$   run_time = omp_get_wtime() 
!$   call omp_set_dynamic(.FALSE.)
!$   call omp_set_num_threads(nthread)
!$omp parallel 
!$  nthr = OMP_GET_NUM_THREADS();print *, ' We are using',nthr,' thread(s)'
!$omp do private(iter)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A2(i,j) = A2(i,j)*(A2(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
!$omp end do
!$omp end parallel
   subname='benchmark2d_omp_inline'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

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
   call benchmark2d_omp0(nthread, iter_max, m, n, A2)
   subname='benchmark2d_omp0'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

  A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=2 
   call benchmark2d_omp0(nthread, iter_max, m, n, A2)
   subname='benchmark2d_omp0'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

   A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=4 
   call benchmark2d_omp(nthread, iter_max, m, n, A2)
   subname='benchmark2d_omp'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

   A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=1 
   call benchmark2d_omp_gpu(nthread, iter_max, m, n, A2)
   subname='benchmark2d_omp_gpu'
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

!$omp target parallel do collapse(2) map(tofrom: A(0:m-1,0:n-1),iter)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine benchmark2d_omp_gpu

