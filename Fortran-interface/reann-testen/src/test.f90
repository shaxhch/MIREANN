program mypes 
    use pes_mod
    implicit none
    character*40 a,pre
    integer(kind=4) :: i,j,npoint,istat,num
    real(kind=8) :: energy_ini,energy_cal
    real(kind=8),allocatable::coor(:), force_ini(:), force_cal(:)

    !part-1 reann model initialization
    call init_reann()

    allocate(coor(numatoms*3),force_ini(numatoms*3),force_cal(numatoms*3))
    npoint=0
    open(181,file="configuration")
    open(182,file="reann")

    do
      do j=1,5
        read(181,*,iostat=istat)
      end do
      do j=1,numatoms
        read(181,*,iostat=istat)
      end do
      read(181,*,iostat=istat)
      if (istat/=0) exit
      npoint=npoint+1
    end do
    rewind(181)
!    write(*,*) npoint  
    do i = 1,npoint
      read(181,*) pre,num
      do j=1,4
        read(181,*)
      end do
      do j=1,numatoms
        read(181,*) species(j),mass(j),coor(1+(j-1)*3:3+(j-1)*3),force_ini(1+(j-1)*3:3+(j-1)*3)
      end do
      read(181,*,iostat=istat) pre,energy_ini
      energy_cal=0.0
      force_cal=0.0

      !part-2 reann model inference
      call pes_ptr%reann_out(coor, energy_cal, force_cal)

      write(182,'(A8,I8,F18.9)') 'Point=',i,energy_cal
      do j=1,numatoms
        write(182,'(3F12.6,3F16.9)') coor(1+(j-1)*3:3+(j-1)*3),force_cal(1+(j-1)*3:3+(j-1)*3)
      end do
    end do
    close(181);close(182)

    !part-3 deallocate all variables realeated to reann
    call delete_reann()
    deallocate(coor,force_ini,force_cal)
end program
