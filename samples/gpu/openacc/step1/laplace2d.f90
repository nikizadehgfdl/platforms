!laplace2d is a simple benchamrk test for openMP and openACC
!To compile
! pgf90 -O2 -o laplace2d_f90 laplace2d.f90
! pgf90 -O2 -mp -Minfo -o laplace2d_f90_omp laplace2d.f90
! pgf90 -O2 -acc -ta=nvidia:managed -Minfo=accel -o laplace2d_f90_acc_managed laplace2d.f90
! pgf90 -O2 -acc -ta=nvidia -Minfo=accel -o laplace2d_f90_acc laplace2d.f90
!
!
!some results
!    size  time(s) iterations initial_sum          final_sum
!./laplace2d_f90
!     512   0.396    2000    0.001294592395425    0.033113177865744
!    1000   1.260    2000    0.000663465878461    0.017047453671694
!    1024   1.344    2000    0.000647931126878    0.016649523749948
!./laplace2d_f90_acc
!     512   1.511    2000    0.001294592395425    0.033113177865744
!    1000   1.573    2000    0.000663465878461    0.017047453671694
!    1024   1.477    2000    0.000647931126878    0.016649523749948
!./laplace2d_f90_acc_managed
!     512   0.105    2000    0.001294592395425    0.033113177865744
!    1000   0.162    2000    0.000663465878461    0.017047453671694
!    1024   0.166    2000    0.000647931126878    0.016649523749948
!pgf90 -O2 -mp -Minfo -o laplace2d_f90_omp laplace2d.f90;  ./laplace2d_f90_omp
!    size  time(s) iterations initial_sum          final_sum          omp_nthreads
!    1000   3.917    2000    0.000663465878461    0.017047453671694    1
!    1000   2.010    2000    0.000663465878461    0.017047453671694    2
!    1000   1.697    2000    0.000663465878461    0.017047453671694    3
!    1000   1.074    2000    0.000663465878461    0.017047453671694    4
!    1000   1.162    2000    0.000663465878461    0.017047453671694    5
!    1000   0.774    2000    0.000663465878461    0.017047453671694    6
!    1000   0.824    2000    0.000663465878461    0.017047453671694    7
!    1000   0.601    2000    0.000663465878461    0.017047453671694    8
!

program laplace
  implicit none
  integer, parameter :: fp_kind=kind(1.0)
  integer, parameter :: n=1000, m=1000, iter_max=2000
  integer :: i, j, iter, itermax
  real(fp_kind), dimension (:,:), allocatable :: A, Anew
  real(fp_kind), dimension (:),   allocatable :: y0
  real(fp_kind) :: pi=2.0_fp_kind*asin(1.0_fp_kind), tol=1.0e-4_fp_kind, error=1.0_fp_kind
  real(fp_kind) :: start_time, stop_time,sum0
  real*8 :: run_time, omp_get_wtime
  integer :: nthread, OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM
  !write(*,'(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', n, ' x', m, ' mesh'
  nthread = 1
!$ do nthread = 1,8
!$ call omp_set_num_threads(nthread)
  allocate ( A(0:n-1,0:m-1), Anew(0:n-1,0:m-1) )
  allocate ( y0(0:m-1) )
  A = 0.0_fp_kind
  ! Set B.C.
  y0 = sin(pi* (/ (j,j=0,m-1) /) /(m-1))
  A(0,:)   = 0.0_fp_kind
  A(n-1,:) = 0.0_fp_kind
  A(:,0)   = y0
  A(:,m-1) = y0*exp(-pi)
  sum0=sum(A)/n/m
  !write(*,'(a,f21.15)')  'debug:  sum(A)=',sum0 

  !
  !Start timing
  !
  call cpu_time(start_time)
!$   run_time = omp_get_wtime();
 
!$acc kernels copyin(y0) create(Anew)
!$omp parallel do shared(Anew)
  do i=1,m-1
    Anew(0,i)   = 0.0_fp_kind
    Anew(n-1,i) = 0.0_fp_kind
  end do
!$end omp parallel do
!$omp parallel do shared(Anew)
  do i=1,n-1
    Anew(i,0)   = y0(i)
    Anew(i,m-1) = y0(i)*exp(-pi)
  end do
!$end omp parallel do
!$acc end kernels
  iter=0
  error=1.0_fp_kind
!$acc data copyin(A) create(Anew) !We do need to copy in A to device and allocate Anew on the device.
!  do while ( error .gt. tol )
  do while ( iter < iter_max )
!$acc kernels present(A,Anew)   !Tell compiler to reuse A,Anew on the device
    error=0.0_fp_kind
!$omp parallel shared(Anew, A) firstprivate(iter) 
!$omp do reduction( max:error )
    do j=1,m-2
      do i=1,n-2
        Anew(i,j) = 0.25_fp_kind * ( A(i+1,j  ) + A(i-1,j  ) + &
                                     A(i  ,j-1) + A(i  ,j+1) )
        error = max( error, abs(Anew(i,j)-A(i,j)) )
      end do
    end do
!$omp end do
!$omp do
    do j=1,m-2
      do i=1,n-2
        A(i,j) = Anew(i,j)
      end do
    end do
!$omp end do
!$omp end parallel
!$acc end kernels
    iter = iter +1
  end do
!$acc update self(A)
!$acc end data
  call cpu_time(stop_time) 
!$   run_time = omp_get_wtime() - run_time;

  if(nthread == 1) then
     write(*,'(a)')  '    size  time(s) iterations initial_sum          final_sum          omp_nthreads'
     run_time = stop_time-start_time !Gives wrong time for nthread>1 with Intel
  endif
  write(*,'(i8,f8.3,i8,f21.15,f21.15,i5)')  n, run_time,iter,sum0,sum(A)/n/m,nthread 
  deallocate (A,Anew,y0)
!$ enddo
end program laplace


! Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!  * Redistributions of source code must retain the above copyright
!    notice, this list of conditions and the following disclaimer.
!  * Redistributions in binary form must reproduce the above copyright
!    notice, this list of conditions and the following disclaimer in the
!    documentation and/or other materials provided with the distribution.
!  * Neither the name of NVIDIA CORPORATION nor the names of its
!    contributors may be used to endorse or promote products derived
!    from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
! PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
! OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
