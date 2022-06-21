
subroutine transform_2darray_omp(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

!$ call omp_set_dynamic(0);
!$ call omp_set_num_threads(nthread);
!$omp parallel do
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine transform_2darray_omp



program benchmark1
  implicit none

  integer, parameter :: m=1000,n=1000, iter_max=2000
  integer :: i, j, iter, itermax
  real, parameter :: pi=2.0*asin(1.0)
  real, dimension (:,:), allocatable :: A, B
  real, dimension (:),  allocatable :: y0
  real :: sum0
  real*8 :: run_time, omp_get_wtime
  integer :: nthread, OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM

  write(*,'(a)')  '   size      time(s) iterations initial_sum          final_sum        omp_nthreads'
  write(*,'(a)')  '2D arrays'
  allocate ( A(0:m-1,0:n-1),B(0:m-1,0:n-1) )
  allocate ( y0(0:n-1) )
!$ do nthread = 1,4
!$ call omp_set_num_threads(nthread)
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

!$   run_time = omp_get_wtime() 
      if(nthread > 0  ) call transform_2darray_omp(nthread,iter_max,m,n,A)
!$   run_time = omp_get_wtime() - run_time;

  write(*,'(i10,f8.3,i8,f21.15,f21.15,i5)')  n*m, run_time,iter_max,sum0,sum(A)/n/m,nthread 

!$ enddo


  deallocate (A, B, y0)

end program benchmark1


!Some results
!
!pgf90 -O3 -mp -Minfo -o benchmark1_pgf90 benchmark1.f90 ; ./benchmark1_pgf90
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!   1000000   4.482    2000    0.000663466518745    0.000026498451916    1
!   1000000   2.449    2000    0.000663466518745    0.000026498451916    2
!   1000000   1.550    2000    0.000663466518745    0.000026498451916    3
!   1000000   1.273    2000    0.000663466518745    0.000026498451916    4

!~/opt/llvm/install/bin/flang benchmark1.f90 -o benchmark1_flang  -O3 -fopenmp; ./benchmark1_flang
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!   1000000   4.339    2000    0.000663466402330    0.000026498400985    1
!   1000000   2.249    2000    0.000663466402330    0.000026498400985    2
!   1000000   1.681    2000    0.000663466402330    0.000026498400985    3

