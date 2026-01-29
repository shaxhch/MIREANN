program mypes 
    use pes_mod
    implicit none
    character*40 a
    integer(kind=4) :: i,j,npoint,istat
    real(kind=8) :: energy_ini, energy_cal
    real(kind=8),allocatable::coor(:), force_ini(:), force_cal(:)

    !part-1 reann model initialization
    call init_reann()

    allocate(coor(numatoms*3),force_ini(numatoms*3),force_cal(numatoms*3))
    open(181,file="REANN.module")
    open(182,file="EANN.xyz")
    open(183,file="EANN.dat")                                                                                                                                               
    read(181,*)
    do j = 1,3 
      read(181,*) cell(j,1:3)
    end do
    read(181,*) a,pbc
    do j=1,numatoms
      read(181,*) species(j),mass(j),coor(1+(j-1)*3:3+(j-1)*3),force_ini(1+(j-1)*3:3+(j-1)*3)
    end do
    read(181,*) a,energy_ini
    do j=1,numatoms
      read(182,*) coor(1+(j-1)*3:3+(j-1)*3)
    end do
    energy_cal=0.0
    force_cal=0.0

    !part-2 reann model inference
    call pes_ptr%reann_out(coor, energy_cal, force_cal)

    write(183,'(F19.9)') energy_cal
    do j=1,numatoms
      write(183,'(3F19.9)') force_cal(1+(j-1)*3:3+(j-1)*3)
    end do
    close(181);close(182);close(183)

    !part-3 deallocate all variables realeated to reann
    call delete_reann()
    deallocate(coor,force_ini,force_cal)
end program
