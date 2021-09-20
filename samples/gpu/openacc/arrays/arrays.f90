!This is a very simple program to show the power of GPUs via openACC directives
!and the comparison of double precision vs. single precision
!and the comparison between "managed" and non-"managed" modes of ACC
!and the offloading of FORTRAN types by openACC 
!The problem is simply to 
!  send an input array to GPU, 
!  calculate another array on GPU based on the input array
!  send the result back to the host
!  calculate the excatly known sum of the elements of the output array
!
!In the following outputs the exact sum is known to be 1.0 (upto machine precision).
!The size of the real arrays are 1000000000
!GPU_M stands for the managed memory mode of ACC sompile -ta=nvidia:managed
!!!Single precision results:
!CPU   Array      time=1.447 seconds,  sum=    1.000990033149719
!CPU   type%Array time=2.046 seconds,  sum=    0.031250000000000
!GPU_M Array      time=2.310 seconds,  sum=    1.000990033149719
!GPU_M type%Array time=1.557 seconds,  sum=    0.031250000000000
!GPU   Array      time=3.497 seconds,  sum=    1.000990033149719
!GPU   type%Array uStreamSynchronize error 700: Illegal address during kernel execution
!
!!!Double precision results (-r8):
!CPU   Array      time=4.734 seconds,  sum=    0.999999992539933
!CPU   type%Array time=3.972 seconds,  sum=    0.999999992539933
!GPU_M Array      time=3.956 seconds,  sum=    0.999999999998084
!GPU_M type%Array time=2.396 seconds,  sum=    0.999999992539933
!GPU   Array      time=13.488 seconds,  sum=    0.999999999998084
!GPU   type%Array uStreamSynchronize error 700: Illegal address during kernel execution
!


program main
  implicit none
  integer, parameter :: isd=0,ied=10001,isc=1,iec=10000 
  integer, parameter :: jsd=0,jed=10001,jsc=1,jec=10000 
  integer, parameter :: nk=10
  integer :: i, j, k
  real, dimension (:,:,:), allocatable :: t_a, t_b
  real, dimension (:,:,:), allocatable :: dd
  type testype
   real, dimension (:,:,:), allocatable :: a,b
  end type testype
  type(testype) :: t
  real :: start_time, stop_time

  allocate ( dd(isd:ied,jsd:jed,1:nk) )
  allocate ( t_a(isd:ied,jsd:jed,1:nk) )
  allocate ( t_b(isd:ied,jsd:jed,1:nk) )
  allocate ( t%a(isd:ied,jsd:jed,1:nk) )
  allocate ( t%b(isd:ied,jsd:jed,1:nk) )

  dd = 0.0
  do k = 1,nk ; do j = jsc,jec ; do i = isc,iec
     dd(i,j,k) = 1.0/(iec-isc+1)/(jec-jsc+1)/nk
  enddo; enddo ; enddo

  t_a = 0.0
  t_b = 0.0
  write(*,'(a)') 'Array'
  call cpu_time(start_time) 
  !$ACC data copyin(dd,t_a,t_b) copyout(t_a,t_b)
  !$ACC parallel loop collapse(3)
  do k = 1,nk ; do j = jsc,jec ; do i = isc,iec
     t_a(i,j,k) = dd(i,j,k)
     t_b(i,j,k) = dd(i,j,k)*0.5*2.0
  enddo; enddo ; enddo
  !$ACC end data
  call cpu_time(stop_time) 
  write(*,'(a,i,a,f10.3,a,a,f21.15)') 'Array      size ',(iec-isc+1)*(jec-jsc+1)*nk, ' completed in ', stop_time-start_time, ' seconds, ',&
                                      ' sum=',sum(t_a) 
  deallocate (t_a,t_b)

  t%a = 0.0
  t%b = 0.0
  write(*,'(a)') 'type%Array'
  call cpu_time(start_time) 
  !$ACC data copyin(dd,t%a,t%b) copyout(t%a,t%b)
  !$ACC parallel loop collapse(3)
  do k = 1,nk ; do j = jsc,jec ; do i = isc,iec
     t%a(i,j,k) = dd(i,j,k)
     t%b(i,j,k) = dd(i,j,k)*0.5*2.0
  enddo; enddo ; enddo
  !$ACC end data
  call cpu_time(stop_time) 
  write(*,'(a,i,a,f10.3,a,a,f21.15)') 'type%Array size ',(iec-isc+1)*(jec-jsc+1)*nk, ' completed in ', stop_time-start_time, ' seconds, ',&
                                      ' sum=',sum(t%a) 
  deallocate (t%a,t%b)

  deallocate (dd)
end program main

