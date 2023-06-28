!This is a quick test for openmp
!To compile and run do:
!\rm ./gpu_offload_test2d2; nvfortran -mp=gpu -stdpar gpu_offload_test2d2.f90 -o gpu_offload_test2d2;./gpu_offload_test2d2
!Output on gpubox
!      size        time(s) iterations initial_sum         final_sum    omp_nthreads    subroutine
!     100000000   429.689    2000    0.000066406416776    0.000002652284758    1   benchmark2d_omp_cpu                                   
!     100000000   235.189    2000    0.000066406416776    0.000002652284758    2   benchmark2d_omp_cpu                                   
!     100000000     0.458    2000    0.000066406416776    0.000002652284758    1   benchmark2d_omp_gpu                               
!     100000000     0.353    2000    0.000066406416776    0.000002652284758    1   benchmark2d_omp_gpu_subij                         
!     100000000     0.189    2000    0.000066406416776    0.000002652284758    1   benchmark2d_docon             
program test_omp
  implicit none
  include 'omp_lib.h'        
  integer, parameter :: m=10000,n=10000, iter_max=2000
  integer :: i, j, iter, itermax
  real*8, parameter :: pi=2.0*asin(1.0)
  real*8, parameter :: tol=1e-10
  real*8, dimension (:,:), allocatable :: A, A2
  real*8, dimension (:),  allocatable :: y0
  real*8 :: sum0
  real*8 :: run_time
  integer :: nthread, nthreads,nthr
  character (len = 50) :: subname
  nthread = 1


  allocate ( A(0:m-1,0:n-1) )
  allocate ( y0(0:n-1) )
  allocate ( A2(0:m-1,0:n-1))
  A=0.0
  ! Set B.C.
  y0 = sin(pi* (/ (j,j=0,n-1) /) /(n-1))
  A(:,0)   = 0.0
  A(:,n-1) = 0.0
  A(0,:)   = y0(:)
  A(m-1,:) = y0(:)*exp(-pi)

  sum0=sum(A)/n/m


  write(*,'(a)')  '     fully vectorizable subroutine Aij=Aij*(Aij-1)'

  write(*,'(a)')  '     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine'
if(.false.) then !The cpu openmp will be too slow (1000x) for large size arrays.
  A2(:,:)=A(:,:)
  nthread=1
!$   run_time = omp_get_wtime()
   call benchmark2d_omp(nthread, iter_max, m, n, A2)
   subname='benchmark2d_omp_cpu'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

  A2(:,:)=A(:,:)
  nthread=2
!$   run_time = omp_get_wtime()
   call benchmark2d_omp(nthread, iter_max, m, n, A2)
   subname='benchmark2d_omp_cpu'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname
endif 

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
   call benchmark2d_omp_gpu_subij(nthread, iter_max, m, n, A2)
   subname='benchmark2d_omp_gpu_subij'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

   A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=1 
   call benchmark2d_docon(nthread, iter_max, m, n, A2)
   subname='benchmark2d_docon'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

!!!!!2d average subs
  write(*,'(a)')  '     non-vectorizable subroutine Aij=(Ai-1,j + Ai+1,j + Ai,j-1 + Ai,j+1)/4'
  write(*,'(a)')  '     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine'

if(.false.) then !The cpu will be too slow (1000x) for large size arrays. Avoid this test unless you really want to.
   A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=1 
   call benchmark2d2_omp_cpu(nthread, iter_max, m, n, A2)
   subname='benchmark2d2_omp_cpu'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname
endif

   A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=1 
   call benchmark2d2_omp_gpu(nthread, iter_max, m, n, A2)
   subname='benchmark2d2_omp_gpu'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

   A2(:,:)=A(:,:)
!$   run_time = omp_get_wtime()
   nthread=1 
   call benchmark2d2_docon(nthread, iter_max, m, n, A2)
   subname='benchmark2d2_docon'
!$   run_time = omp_get_wtime() - run_time;
   write(*,'(i14,f10.3,i8,f21.15,f21.15,i5,5X,A)')  n*m,run_time,iter_max,sum0,sum(A2)/n/m,nthread,subname

   deallocate (A, y0, A2)
end program test_omp

subroutine benchmark2d_omp(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter,nthr
!$ call omp_set_dynamic(0)
!$ call omp_set_num_threads(nthread)

!$omp parallel 
!!!$  nthr = OMP_GET_NUM_THREADS();print *, ' We are using',nthr,' thread(s)'
!c$  print *, ' We are using',OMP_GET_NUM_THREADS(),' thread(s)'
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

subroutine benchmark2d_omp_gpu(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

!$omp target parallel do collapse(2) map(tofrom: A(0:m-1,0:n-1)) private(iter) device(1)
  do j=0,n-1;do i=0,m-1
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
   enddo; enddo
end subroutine benchmark2d_omp_gpu

subroutine benchmark2d_omp_gpu_subij(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A

!$omp target parallel do collapse(2) map(tofrom: A(0:m-1,0:n-1)) device(1)
  do j=0,n-1;do i=0,m-1
     call sub1ij(A(i,j),iter_max)
  enddo; enddo
end subroutine benchmark2d_omp_gpu_subij

subroutine sub1ij(x,itermax)
  real*8, intent(inout) :: x
  integer, intent(in) :: itermax
  integer :: iter
  iter=0
  do while (iter < itermax)
     x = x*(x-1.0)
     iter = iter+1
  enddo
end subroutine sub1ij

subroutine benchmark2d_docon(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
!$ACC set device_num(1)
  do concurrent(j=0:n-1,i=0:m-1)
     iter=0
     do while (iter < iter_max)
        A(i,j) = A(i,j)*(A(i,j)-1.0)
        iter = iter+1
     enddo
  enddo
end subroutine benchmark2d_docon

subroutine benchmark2d2_omp_cpu(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

!$ call omp_set_dynamic(0)
!$ call omp_set_num_threads(nthread)
  iter=0
  do while (iter < iter_max)
!$omp parallel do private(iter)
    do j=1,n-2;do i=1,m-2
      A(i,j) = 0.25*(A(i-1,j)+A(i+1,j)+A(i,j-1)+A(i,j+1))
    enddo; enddo
    iter = iter+1
  enddo
end subroutine benchmark2d2_omp_cpu

subroutine benchmark2d2_omp_gpu(nthread, iter_max, m, n, A)
!Answers are not repeatable and too different from cpu answers
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter

  iter=0
  do while (iter < iter_max)
!$omp target parallel do map(tofrom: A(0:m-1,0:n-1))
    do j=1,n-2;do i=1,m-2
      A(i,j) = 0.25*(A(i-1,j)+A(i+1,j)+A(i,j-1)+A(i,j+1))
    enddo; enddo
    iter = iter+1
  enddo
end subroutine benchmark2d2_omp_gpu

subroutine benchmark2d2_omp_gpu1(nthread, iter_max, m, n, A)
!Answers are  different from the above and from cpu answers
!takes a longer time
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
  real*8, dimension(0:m-1,0:n-1) :: Anew

  iter=0
  do while (iter < iter_max)
!$omp target map(tofrom: A(0:m-1,0:n-1),Anew(0:m-1,0:n-1))
!$omp parallel do
    do j=1,n-2;do i=1,m-2
      Anew(i,j) = 0.25*(A(i-1,j)+A(i+1,j)+A(i,j-1)+A(i,j+1))
    enddo; enddo

!$omp parallel do
    do j=1,n-2;do i=1,m-2
      A(i,j) = Anew(i,j)
    enddo; enddo
!$omp end target
    iter = iter+1
  enddo

end subroutine benchmark2d2_omp_gpu1

subroutine benchmark2d2_docon(nthread, iter_max, m, n, A)
  integer, intent(in) :: nthread, iter_max, m, n
  real*8, dimension(0:m-1,0:n-1), intent(inout) :: A
  integer :: iter
  iter=0
  do while (iter < iter_max)
!$ACC set device_num(1)
    do concurrent(j=1:n-2,i=1:m-2)
      A(i,j) = 0.25*(A(i-1,j)+A(i+1,j)+A(i,j-1)+A(i,j+1))
    enddo
    iter = iter+1
  enddo
end subroutine benchmark2d2_docon
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
!
!06/27/2023 
!It's strange that adding -stdpar to compile options speeds up the test by a factor of 5 (2.247/0.46)
!This means something is causing a slowdown without -stdpar !
!lscgpu50-d: /home/Niki.Zadeh/platforms/samples/gpu % nvfortran -mp=gpu gpu_offload_test2d.f90 -o gpu_offload_test2d ; ./gpu_offload_test2d
!      size      time(s) iterations initial_sum     final_sum        omp_nthreads subroutine
!     100000000     2.247    2000    0.000066406416776    0.000002652284758    1 benchmark2d_omp_gpu              
!lscgpu50-d: /home/Niki.Zadeh/platforms/samples/gpu % nvfortran -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d ; ./gpu_offload_test2d
!     size      time(s) iterations initial_sum     final_sum        omp_nthreads subroutine
!     100000000     0.460    2000    0.000066406416776    0.000002652284758    1 benchmark2d_omp_gpu               
