subroutine benchmark2d(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine benchmark2d

subroutine benchmark2d_omp(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
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
end subroutine benchmark2d_omp
subroutine benchmark2d_omp_teams(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
!$ call omp_set_dynamic(0);
!$ call omp_set_num_threads(nthread);
!The following is an error with nvfortran
!NVFORTRAN-S-1233-TEAMS construct must be strictly nested to the TARGET. (benchmark1.f90: 39)
!c$omp teams distribute parallel do
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine benchmark2d_omp_teams

subroutine benchmark2d_omp_gpu_teams_2devs(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
  integer :: dev,j1,j2

!$omp parallel do private(dev,j1,j2,iter,i,j) shared(A,iter_max,m,n)
do dev=0,1
  if(dev==0) then
     j1=0;j2=n/2
  endif   
  if(dev==1) then
     j1=n/2+1; j2=n-1
  endif
!$omp target teams distribute parallel do map(tofrom: A(0:m-1,j1:j2)) device(dev)
  do j=j1,j2;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
enddo
end subroutine benchmark2d_omp_gpu_teams_2devs

subroutine benchmark2d_omp_gpu_teams(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

!$omp target teams distribute parallel do map(tofrom: A(0:m-1,0:n-1))
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine benchmark2d_omp_gpu_teams

subroutine benchmark2d_omp_gpu(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

!$omp target parallel do map(tofrom: A(0:m-1,0:n-1),iter)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine benchmark2d_omp_gpu

program benchmark1_omp
  implicit none
  integer, parameter :: m=4000,n=4000, iter_max=2000
  integer :: i, j, iter, itermax
  real*8, parameter :: pi=2.0*asin(1.0)
  real*8, parameter :: tol=1e-10
  real*8, dimension (:,:), allocatable :: A, B,A2
  real*8, dimension (:),  allocatable :: y0
  real*8 :: sum0
  real*8 :: res1=0.000006629710014
  real*8 :: run_time, omp_get_wtime
  integer :: nthread, nthreads, OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM
  character (len = 50) :: subname
  integer :: input
  READ(*,*) input
  nthreads = 1
  if(input .ge. 0) nthreads=input

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

  write(*,'(a)')  '      size      time(s) iterations initial_sum          final_sum        omp_nthreads    subroutine'

  do nthread = 1,nthreads
     A2(:,:)=A(:,:)
     sum0=sum(A2)/n/m

!$   run_time = omp_get_wtime() 
      if(input > 0  ) then
         call benchmark2d_omp(nthread,iter_max,m,n,A2)
         subname='benchmark2d_omp'
      endif
      if(input == -1) then
         call benchmark2d_omp_gpu(1,iter_max,m,n,A2)
         subname='benchmark2d_omp_gpu'
      endif
      if(input == -2) then
         call benchmark2d_omp_gpu_teams(1,iter_max,m,n,A2)
         subname='benchmark2d_omp_gpu_teams'
      endif
      if(input == -3) then
         call benchmark2d_omp_gpu_teams_2devs(1,iter_max,m,n,A2)
         subname='benchmark2d_omp_gpu_teams_2devs'
      endif
!$   run_time = omp_get_wtime() - run_time;

      write(*,'(i14,f8.3,i8,f21.15,f21.15,i5,5X,A)')  n*m, run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname
      if(dabs(sum(A2)/n/m-res1) .gt. tol) print*,'Wrong Result!!!!'
   enddo
   deallocate (A, B, y0, A2)
end program benchmark1_omp
!Some results
!bash
!module load cuda/11.7
!module load nvhpc-no-mpi/22.5
!ulimit -s unlimited 
!
!Note: -mp=gpu means offload openmp to gpu
!Note: cc70 in -ta stands for the compute capability of the gpu, not easy to find
!nvfortran -Mpreprocess  -ta=tesla,cc70,cuda11.7 -mp=gpu  benchmark1_omp.f90 -o benchmark1_omp ;time ./benchmark1_omp
!10
!      size      time(s) iterations initial_sum          final_sum        omp_nthreads    subroutine
!      16000000  68.689    2000    0.000165991179529    0.000006629719337    1     benchmark2d_omp            
!      16000000  34.430    2000    0.000165991179529    0.000006629719337    2     benchmark2d_omp
!      16000000  24.769    2000    0.000165991179529    0.000006629719337    3     benchmark2d_omp
!      16000000  17.540    2000    0.000165991179529    0.000006629719337    4     benchmark2d_omp
!      16000000  15.075    2000    0.000165991179529    0.000006629719337    5     benchmark2d_omp
!      16000000  11.840    2000    0.000165991179529    0.000006629719337    6     benchmark2d_omp
!      16000000  11.561    2000    0.000165991179529    0.000006629719337    7     benchmark2d_omp
!      16000000   8.912    2000    0.000165991179529    0.000006629719337    8     benchmark2d_omp 
!      16000000   7.925    2000    0.000165991179529    0.000006629719337    9     benchmark2d_omp
!      16000000   7.129    2000    0.000165991179529    0.000006629719337   10     benchmark2d_omp
!-1
!      size      time(s) iterations initial_sum          final_sum        omp_nthreads    subroutine
!      16000000   1.386    2000    0.000165991179529    0.000006629719337    1     benchmark2d_omp_gpu
!
!-2
!      size      time(s) iterations initial_sum          final_sum        omp_nthreads    subroutine
!      16000000   1.365    2000    0.000165991179529    0.000006629719337    1     benchmark2d_omp_gpu_teams  
!
!-3
!      size      time(s) iterations initial_sum          final_sum        omp_nthreads    subroutine
!      16000000   1.600    2000    0.000165991179529    0.000006629719337    1     benchmark2d_omp_gpu_teams_2devs 
