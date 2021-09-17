
program main
  implicit none
  integer, parameter :: isd=0,ied=10001,isc=1,iec=10000 
  integer, parameter :: jsd=0,jed=10001,jsc=1,jec=10000 
  integer, parameter :: nk=10
  integer :: i, j, k
  real, dimension (:,:,:), allocatable :: t_a, t_b
  real, dimension (:,:,:), allocatable :: dd
  real :: start_time, stop_time

  allocate ( dd(isd:ied,jsd:jed,1:nk) )
  allocate ( t_a(isd:ied,jsd:jed,1:nk) )
  allocate ( t_b(isd:ied,jsd:jed,1:nk) )
  call cpu_time(start_time) 

  dd = 0.0
  do k = 1,nk ; do j = jsc,jec ; do i = isc,iec
     dd(i,j,k) = 1.0/(iec-isc+1)/(jec-jsc+1)/nk
  enddo; enddo ; enddo

  t_a = 0.0
  t_b = 0.0

  !$ACC data copyin(dd,t_a,t_b) copyout(t_a,t_b)
  !$ACC parallel loop collapse(3)
  do k = 1,nk ; do j = jsc,jec ; do i = isc,iec
     t_a(i,j,k) = dd(i,j,k)
     t_b(i,j,k) = dd(i,j,k)*0.5
  enddo; enddo ; enddo
  !$ACC end data
  call cpu_time(stop_time) 
  write(*,'(a,i,a,f10.3,a,a,f21.15,f21.15)') 'Problem size ',(iec-isc+1)*(jec-jsc+1)*nk, ' completed in ', stop_time-start_time, ' seconds, ',&
                                      ' sums=',sum(t_a),sum(t_b) 
  deallocate (dd,t_a,t_b)
end program main
