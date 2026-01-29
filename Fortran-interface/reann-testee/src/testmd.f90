program mypes 
    use Chpes_mod
    implicit none
    character*40 a
    integer(kind=4) :: i,j
    real(kind=8),allocatable:: energy_ini(:), energy_cal(:)
    real(kind=8),allocatable::coor(:),force_cal(:)

    !part-1 reann model initialization
    call init_chreann()

    allocate(energy_ini(chnumatoms),energy_cal(chnumatoms))
    allocate(coor(chnumatoms*3),force_cal(chnumatoms*chnumatoms*3))
    open(181,file="REANNCh.module")
    open(182,file="EANN.xyz")
    open(183,file="EANN.dat",access='append')

    read(181,*)
    do j = 1,3
      read(181,*) chcell(j,1:3)
    end do
    read(181,*) a,chpbc
    do j=1,chnumatoms
      read(181,*) chspecies(j),chmass(j),coor(1+(j-1)*3:3+(j-1)*3)
    end do
    read(181,*) a,energy_ini(:)
    do j=1,chnumatoms
      read(182,*) coor(1+(j-1)*3:3+(j-1)*3)
    end do
    energy_cal=0.0
    force_cal=0.0

    !part-2 reann model inference
    call Chpes_ptr%reann_chout(coor, energy_cal, force_cal)

    do j=1,chnumatoms
      write(183,'(F20.8)') energy_cal(j)
      do i=1,chnumatoms
        write(183,'(3F14.8)') force_cal(j*chnumatoms*3+i*3-80:j*chnumatoms*3+i*3-78)
      end do
    end do
    close(181);close(182);close(183)

    !part-3 deallocate all variables realeated to reann
    call delete_Chreann()
    deallocate(coor,energy_ini,energy_cal,force_cal)
end program
