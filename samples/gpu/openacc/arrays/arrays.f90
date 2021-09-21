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
!In the following the exact values of "sums" should be 1.0 (upto machine precision.) 
!Array size=   1000000000
!GPU_M stands for the managed memory mode of ACC compilation -ta=nvidia:managed
!
!!!Single precision results:
!CPU   Array      time= 2.414 seconds,  sums=  1.000990033149719    0.000000000000000
!CPU   type%Array time=23.48  seconds,  sums=  0.031250000000000    0.000000000000000
!GPU_M Array      time= 1.713 seconds,  sums=  1.000990033149719    0.000000000000000
!GPU_M type%Array time= 1.236 seconds,  sums=  0.031250000000000    0.000000000000000
!GPU   Array      time= 5.445 seconds,  sums=  1.000990033149719    0.000000000000000
!GPU   type%Array uStreamSynchronize error 700: Illegal address during kernel execution
!
!!!Double precision results (-r8):
!CPU   Array      time=34.177 seconds, sums=  0.999999992539933    1.000000082739080
!CPU   type%Array time=34.442 seconds, sums=  0.999999992539933    1.000000082739080
!GPU_M Array      time= 3.776 seconds, sums=  0.999999999998084    1.000000082487792
!GPU_M type%Array time= 2.776 seconds, sums=  0.999999992539933    1.000000082739080
!GPU   Array      time=15.057 seconds, sums=  0.999999999998084    1.000000082487792
!GPU   type%Array uStreamSynchronize error 700: Illegal address during kernel execution
!
!NOTES:
!- Single precision may be much faster, but it does not give accurate answers,
!- Single precision gives totally wrong answers if Math function calls are involved (0.0 for the second sum)!
!- Single precision gives totally wrong answers for sum(A) if array A is a member element of a type!
!- Double precision is not as accurate as it should be (only 7-8 decimals rather than 14 decimals).
!- GPU with non-managed memory is very slow
!- GPU with non-managed memory cannot handle offloading of types
!- GPU double precision seems to be more accurate (closer to 1.) for array than type%array
!- On a positive note, GPU with managed memory is doing a fine job, both in acccuracy and speeding up the model.
!
!
!
!Naively, here's why I expect to see 14 digits accuracy with -r8
!  real :: x
!  x=0.123456789123456789
!  write(*,'(a,f28.25)') 'x = ',x
!  write(*,'(a,f28.25)') 'fx= ',2.0*log(exp(x*0.5))
!  Gives:
!Single precision output:
!!!x =  0.1234567910432815551757813
!!!fx=  0.1234567314386367797851563
!!!Warning: ieee_inexact is signaling
!Double precision (-r8) output:
!!!x =  0.1234567891234567837965841
!!!fx=  0.1234567891234567837965841
!!!Warning: ieee_inexact is signaling
!
!So double precision agrees to 17 decimal places whereas single precision to just 7
!Why don't I see this for the sums above?


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


  write(*,'(a,i)') 'Array size= ',(iec-isc+1)*(jec-jsc+1)*nk

  !Construct a data array
  dd = 0.0
  do k = 1,nk ; do j = jsc,jec ; do i = isc,iec
     dd(i,j,k) = 1.0/(iec-isc+1)/(jec-jsc+1)/nk
  enddo; enddo ; enddo

  !work arrays calculated from data array
  t_a = 0.0
  t_b = 0.0
  call cpu_time(start_time) 
  !$ACC data copyin(dd,t_a,t_b) copyout(t_a,t_b)
  !$ACC parallel loop collapse(3)
  do k = 1,nk ; do j = jsc,jec ; do i = isc,iec
     t_a(i,j,k) = dd(i,j,k)
     t_b(i,j,k) = exp(t_a(i,j,k)*0.5)
     t_b(i,j,k) = 2.0*log(t_b(i,j,k))
  enddo; enddo ; enddo
  !$ACC end data
  call cpu_time(stop_time) 
  write(*,'(a,f7.3,a,f19.15,f19.15)') 'Array,       time(secs)=', stop_time-start_time,&
                                      ' sums=',sum(t_a) ,sum(t_b) 
  deallocate (t_a,t_b)

  !same work arrays calculated from data array BUT this time they are elements of a type t
  t%a = 0.0
  t%b = 0.0
  call cpu_time(start_time) 
  !$ACC data copyin(dd,t%a,t%b) copyout(t%a,t%b)
  !$ACC parallel loop collapse(3)
  do k = 1,nk ; do j = jsc,jec ; do i = isc,iec
     t%a(i,j,k) = dd(i,j,k)
     t%b(i,j,k) = exp(t%a(i,j,k)*0.5)
     t%b(i,j,k) = 2.0*log(t%b(i,j,k))
  enddo; enddo ; enddo
  !$ACC end data
  call cpu_time(stop_time) 
  write(*,'(a,f7.3,a,f19.15,f19.15)') 'type%Array,  time(secs)=', stop_time-start_time,&
                                      ' sums=',sum(t%a) ,sum(t%b) 
  deallocate (t%a,t%b)

  deallocate (dd)
end program main

