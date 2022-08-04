
subroutine transform_2darray(nthread, iter_max, m, n, A)
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
end subroutine transform_2darray

subroutine transform_2darray_doconcurrent(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
!!$omp target data map(A(0:m-1,0:n-1))
  do concurrent(j=0:n-1) 
    do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
    enddo
  enddo
!!$omp target end
end subroutine transform_2darray_doconcurrent

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

subroutine transform_2darray_omp_teams(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
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
end subroutine transform_2darray_omp_teams

subroutine transform_2darray_omp_gpu_teams(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

!$omp target teams distribute parallel do map(tofrom: A(0:m-1,0:n-1),iter)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine transform_2darray_omp_gpu_teams

subroutine transform_2darray_omp_gpu(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

!$omp target parallel do map(tofrom: A(0:m-1,0:n-1),iter)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine transform_2darray_omp_gpu



program benchmark1
  implicit none

  integer, parameter :: m=4000,n=4000, iter_max=2000
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
  do nthread = -4,0
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
      if(nthread ==  0) call transform_2darray(1,iter_max,m,n,A)
      if(nthread == -1) cycle !call transform_2darray_omp_teams(1,iter_max,m,n,A)
      if(nthread == -2) call transform_2darray_omp_gpu(1,iter_max,m,n,A)
      if(nthread == -3) call transform_2darray_omp_gpu_teams(1,iter_max,m,n,A)
      if(nthread == -4) call transform_2darray_doconcurrent(1,iter_max,m,n,A)
!$   run_time = omp_get_wtime() - run_time;

  write(*,'(i10,f8.3,i8,f21.15,f21.15,i5)')  n*m, run_time,iter_max,sum0,sum(A)/n/m,nthread 

  enddo


  deallocate (A, B, y0)

end program benchmark1


!Some results
!
!module load cuda/11.7
!module load nvhpc-no-mpi/22.5
!ulimit -s unlimited 
!
!Note: -mp=gpu means offload openmp to gpu
!Note: cc70 in -ta stands for the compute capability of the gpu, not easy to find
!
!nvfortran -Mpreprocess  -ta=tesla,cc70,cuda11.7 -mp=gpu  benchmark1.f90 -o benchmark1_nvf_target ;time ./benchmark1_nvf_target
!
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!  16000000   0.292    2000    0.000165991179529    0.000006629719337   -3!!GPU teams
!  16000000   0.078    2000    0.000165991179529    0.000006629719337   -2!!GPU 
!  16000000  68.899    2000    0.000165991179529    0.000006629719337    0!cpu
!  16000000  68.683    2000    0.000165991179529    0.000006629719337    1
!  16000000  34.441    2000    0.000165991179529    0.000006629719337    2
!  16000000  26.666    2000    0.000165991179529    0.000006629719337    3
!
!It is curious that "nvidia-smi -l 1" shows the GPU busy for ALL the cases above!
!Shouldn't gpu be idle when there is no "$omp target" directive for nthread>=0 above ?
!How can we ensure that the first two results come from GPU given that numerical values are the same as cpu? Why would they be the same to such accuracy? 
!
!Offload with do concurrent is blazing
!nvfortran -Mpreprocess -mp -stdpar  benchmark1.f90 -o benchmark1_nvf_docon ;time ./benchmark1_nvf_docon
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!  16000000   0.042    2000    0.000165991179529    0.000006629719337   -4
!  16000000   4.713    2000    0.000165991179529    0.000006629719337   -3
!  16000000   4.657    2000    0.000165991179529    0.000006629719337   -2
!  16000000  68.909    2000    0.000165991179529    0.000006629719337    0
!
!nvfortran -Mpreprocess -mp -stdpar=gpu -Minfo=stdpar  benchmark1.f90 -o benchmark1_nvf_docon ;time ./benchmark1_nvf_docon
!
!Note: we cannot use both -ta and -stdpar !!
!
!No offloading, replace -mp=gpu by -mp    
!nvfortran -Mpreprocess -ta=tesla -mp benchmark1.f90 -o benchmark1_nvf ; time ./benchmark1_nvf
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!  16000000   4.570    2000    0.000165991179529    0.000006629719337   -3
!  16000000   4.479    2000    0.000165991179529    0.000006629719337   -2
!  16000000  68.653    2000    0.000165991179529    0.000006629719337    0
!  16000000  68.634    2000    0.000165991179529    0.000006629719337    1
!  16000000  34.555    2000    0.000165991179529    0.000006629719337    2
!  16000000  24.625    2000    0.000165991179529    0.000006629719337    3
!  16000000  17.661    2000    0.000165991179529    0.000006629719337    4
!  16000000  15.080    2000    0.000165991179529    0.000006629719337    5
!  16000000  11.869    2000    0.000165991179529    0.000006629719337    6
!  16000000  11.465    2000    0.000165991179529    0.000006629719337    7
!  16000000   8.921    2000    0.000165991179529    0.000006629719337    8
!  16000000   7.976    2000    0.000165991179529    0.000006629719337    9
!  16000000   7.136    2000    0.000165991179529    0.000006629719337   10
!  16000000   6.531    2000    0.000165991179529    0.000006629719337   11
!  16000000   5.958    2000    0.000165991179529    0.000006629719337   12
!  16000000   5.511    2000    0.000165991179529    0.000006629719337   13
!  16000000   5.124    2000    0.000165991179529    0.000006629719337   14
!  16000000   4.793    2000    0.000165991179529    0.000006629719337   15
!  16000000   4.489    2000    0.000165991179529    0.000006629719337   16
!  16000000   5.494    2000    0.000165991179529    0.000006629719337   17
!  16000000   5.145    2000    0.000165991179529    0.000006629719337   18
!  16000000   5.064    2000    0.000165991179529    0.000006629719337   19
!  16000000   5.188    2000    0.000165991179529    0.000006629719337   20
!  16000000   5.511    2000    0.000165991179529    0.000006629719337   21
!  16000000   5.215    2000    0.000165991179529    0.000006629719337   22
!  16000000   5.328    2000    0.000165991179529    0.000006629719337   23
!  16000000   5.441    2000    0.000165991179529    0.000006629719337   24
!  16000000   5.230    2000    0.000165991179529    0.000006629719337   25
!  16000000   5.350    2000    0.000165991179529    0.000006629719337   26
!  16000000   5.326    2000    0.000165991179529    0.000006629719337   27
!  16000000   5.088    2000    0.000165991179529    0.000006629719337   28
!  16000000   5.008    2000    0.000165991179529    0.000006629719337   29
!  16000000   4.817    2000    0.000165991179529    0.000006629719337   30
!  16000000   4.757    2000    0.000165991179529    0.000006629719337   31
!  16000000   4.566    2000    0.000165991179529    0.000006629719337   32
!  16000000   5.015    2000    0.000165991179529    0.000006629719337   33
!  16000000   5.398    2000    0.000165991179529    0.000006629719337   34
!  16000000   4.977    2000    0.000165991179529    0.000006629719337   35
!  16000000   5.233    2000    0.000165991179529    0.000006629719337   36
!  16000000   4.950    2000    0.000165991179529    0.000006629719337   37
!  16000000   5.288    2000    0.000165991179529    0.000006629719337   38
!  16000000   5.376    2000    0.000165991179529    0.000006629719337   39
!  16000000   5.331    2000    0.000165991179529    0.000006629719337   40
!
!From above timings it looks like "$omp target parallel do "
!in the absence of gpu-offloading finds the optimal number of openmp threads 
!for the cpu. Its timing 4.479secs is almost the same as (and smaller than)  
!the 32 threads which gives the least time for openmp jobs.
!
!I wanted to see what -ta=tesla is doing for the no-offload case above. So I removed it and look what happened:
!nvfortran -Mpreprocess -mp benchmark1.f90 -o benchmark1_nvf ; time ./benchmark1_nvf
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!  16000000  121.394    2000    0.000165991179529    0.000006629719337    0
!  16000000  121.649    2000    0.000165991179529    0.000006629719337    1
!  16000000  529.560    2000    0.000165991179529    0.000002164418447    2
!  16000000  648.300    2000    0.000165991179529    0.000001493037303    3
!  16000000 1058.108    2000    0.000165991179529   -0.000000860898240   21
!  16000000 1084.153    2000    0.000165991179529   -0.000000942094857   22
!
!This is crawling and the final_sum is totally wrong for omp_threads > 1 ! 
!Maybe this is -O0 and -O2 or -O3 or -fast would speeds it up to the level of -ta=tesla? But why?
!nvfortran -mp benchmark1.f90 -o benchmark1_nvf -fast ; time ./benchmark1_nvf
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!  16000000  68.664    2000    0.000165991063113    0.000006629711152    0
!  16000000  68.628    2000    0.000165991063113    0.000006629711152    1
!  16000000  34.446    2000    0.000165991063113    0.000006629711152    2
!  16000000  23.986    2000    0.000165991063113    0.000006629711152    3
!  16000000  17.599    2000    0.000165991063113    0.000006629711152    4
!
!nvfortran -Mpreprocess -mp benchmark1.f90 -o benchmark1_nvf -O2  ; time ./benchmark1_nvf
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!  16000000  68.666    2000    0.000165991063113    0.000006629711152    0
!  16000000  68.624    2000    0.000165991063113    0.000006629711152    1
!  16000000  34.430    2000    0.000165991063113    0.000006629711152    2
!  16000000  24.070    2000    0.000165991063113    0.000006629711152    3
!  16000000  17.579    2000    0.000165991063113    0.000006629711152    4
!nvfortran -mp benchmark1.f90 -o benchmark1_nvf -O3 ; time ./benchmark1_nvf
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!  16000000  68.665    2000    0.000165991063113    0.000006629712061    0
!  16000000  68.624    2000    0.000165991063113    0.000006629712061    1
!  16000000  34.455    2000    0.000165991063113    0.000006629712061    2
!  16000000  23.838    2000    0.000165991063113    0.000006629712061    3
!  16000000  17.617    2000    0.000165991063113    0.000006629712061    4
!
!Note the timings get to the same level of -ta=tesla, but the answers deterirate a little.
!
!
!
!flang compiler
!No gpu offloading.
!~/opt/llvm/install/bin/flang benchmark1.f90 -o benchmark1_flang  -O3 -fopenmp; ./benchmark1_flang
!   size      time(s) iterations initial_sum          final_sum        omp_nthreads
!2D arrays
!   1000000   4.371    2000    0.000663466402330    0.000026498400985    1
!   1000000   2.397    2000    0.000663466402330    0.000026498400985    2
!   1000000   1.692    2000    0.000663466402330    0.000026498400985    3
!   1000000   1.224    2000    0.000663466402330    0.000026498400985    4

