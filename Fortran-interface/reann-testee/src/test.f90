program mypes 
    use chpes_mod
    implicit none
    character*40 a
    integer(kind=4) :: i,j,npoint,istat,num_mm
    real(kind=8) :: energy_cal,energy_cal2,tmp,energy_menn,energy_eenn,energy_mepoint,energy_eepoint,energy_ab
    real(kind=8),allocatable::coor_qm(:), coor_mm(:), cha_mm(:), force_qm(:), force_mm(:),cha_eenn(:),distance(:),  &
                              cha_menn(:),cha_ee(:), cha_me(:), pot_ee(:),pot_me(:),pot_eenn(:),pot_menn(:),        &
                              pot_eePoint(:),pot_mePoint(:), cha_mm_copy(:),cha_real(:)
    integer,parameter :: luckatom=8,RCnum=2
    integer::RCindex(2,RCnum)
    real(kind=8)::RCeff(RCnum),rc
    RCindex=reshape([5,8,5,6],[2,RCnum])
    RCeff=[-1,1]

    !part-1 reann model initialization
    call init_chreann()

    allocate(coor_qm(chnumatoms*3),coor_mm(chmaxnum_mm*3),cha_mm(chmaxnum_mm),force_qm(chnumatoms*3),distance(chmaxnum_mm), &
             force_mm(chmaxnum_mm*3),cha_eenn(chnumatoms),cha_menn(chnumatoms),cha_ee(chnumatoms),cha_me(chnumatoms),   &
             pot_ee(chmaxnum_mm),pot_me(chmaxnum_mm),pot_eenn(chmaxnum_mm),pot_menn(chmaxnum_mm),pot_mePoint(chmaxnum_mm), &
             pot_eePoint(chmaxnum_mm),cha_mm_copy(chmaxnum_mm),cha_real(chmaxnum_mm))
    npoint=1
    open(181,file="configuration")
    open(182,file="reann")

    do
      coor_mm = 0d0;cha_mm = 0d0;cha_ee = 0d0;cha_me = 0d0;cha_eenn = 0d0;cha_menn = 0d0;cha_mm_copy = 0d0
      pot_eenn = 0d0;pot_menn = 0d0;pot_ee = 0d0;pot_me = 0d0;rc = 0d0;pot_eePoint = 0d0;pot_mePoint = 0d0
      energy_menn = 0d0;energy_eenn = 0d0;energy_mepoint = 0d0;energy_eepoint = 0d0

      read(181,*,iostat=istat)
      if (istat/=0) exit
      do j = 1,3
        read(181,*,iostat=istat) chcell(j,1:3) 
      end do
      read(181,*,iostat=istat) a,chpbc
      do j=1,chnumatoms
        read(181,*,iostat=istat) chspecies(j),chmass(j),coor_qm(1+(j-1)*3:3+(j-1)*3)
      end do
      read(181,*,iostat=istat) 

      num_mm = 0
      do
        read(181,*,iostat=istat) coor_mm(num_mm*3+1:num_mm*3+3),cha_mm(num_mm+1),pot_ee(num_mm+1),cha_real(num_mm+1)
        if (istat/=0) then
          backspace(181)
          read(181,*,iostat=istat) a, energy_ab
          exit
        end if
        num_mm=num_mm+1
      end do
      write(*,*) num_mm

      call Chpes_ptr%reann_chout(coor_qm,coor_mm,cha_mm,energy_cal,force_qm,force_mm,cha_eenn)

      cha_mm = 0d0
      call Chpes_ptr%reann_chout(coor_qm,coor_mm,cha_mm,energy_cal2,force_qm,force_mm,cha_menn)

      write(182,'(2F18.6)') energy_ab,energy_cal
      npoint=npoint+1
    end do

    !part-3 deallocate all variables related to reann
    close(181);close(182)
    call delete_Chreann()
    deallocate(coor_qm,coor_mm,cha_mm,force_qm,force_mm,cha_eenn,cha_ee,cha_menn,cha_me,pot_ee,pot_me,pot_eenn,pot_menn,distance)
end program
