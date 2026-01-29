program mypes 
    use chpes_mod
    implicit none
    character*40 a
    integer(kind=4) :: i,j,npoint,istat,num_mm
    real(kind=8) :: energy_cal,tmp,energy_menn,energy_eenn,energy_mepoint,energy_eepoint
    real(kind=8),allocatable::coor_qm(:), coor_mm(:), cha_mm(:), force_qm(:), force_mm(:),cha_eenn(:),distance(:),  &
                              cha_menn(:),cha_ee(:), cha_me(:), pot_ee(:),pot_me(:),pot_eenn(:),pot_menn(:),        &
                              pot_eePoint(:),pot_mePoint(:), cha_mm_copy(:)
    real(kind=8) :: t_start, t_end
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
             pot_eePoint(chmaxnum_mm),cha_mm_copy(chmaxnum_mm))
    npoint=1
    open(181,file="EEconfiguration")
    open(183,file="MEconfiguration")
    open(182,file="reann")
    open(184,file="result")

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
        read(181,'(3F12.6,F9.4,F12.6)',iostat=istat) coor_mm(num_mm*3+1:num_mm*3+3),    &
                                                     cha_mm(num_mm+1), pot_ee(num_mm+1)
        if (istat/=0) then
          backspace(181)
          read(181,*,iostat=istat) a,cha_ee(:)
          exit
        end if
        num_mm=num_mm+1
      end do

      call Chpes_ptr%reann_chout(coor_qm,coor_mm,cha_mm,energy_cal,force_qm,force_mm,cha_eenn)

      cha_mm_copy = cha_mm
      read(183,*,iostat=istat) 
      do j = 1,3
        read(183,*,iostat=istat) chcell(j,1:3) 
      end do
      read(183,*,iostat=istat) a,chpbc
      do j=1,chnumatoms
        read(183,*,iostat=istat) chspecies(j),chmass(j),coor_qm(1+(j-1)*3:3+(j-1)*3)
      end do
      read(183,*,iostat=istat) 

      do i = 0,num_mm-1
        read(183,'(3F12.6,F5.1,F12.6)',iostat=istat) coor_mm(i*3+1:i*3+3), cha_mm(i+1), pot_me(i+1)
      end do
      read(183,*,iostat=istat) a,cha_me(:)

      call cpu_time(t_start)
      call Chpes_ptr%reann_chout(coor_qm,coor_mm,cha_mm,energy_cal,force_qm,force_mm,cha_menn)
      call cpu_time(t_end)
      write(*,'("Time for point",I5,": ",F10.6," seconds")') npoint, t_end - t_start

      do j = 1, num_mm
        do i = 1,chnumatoms
          tmp = dsqrt((coor_qm(i*3-2)-coor_mm(j*3-2))**2+(coor_qm(i*3-1)-coor_mm(j*3-1))**2  &
                     +(coor_qm(i*3)-coor_mm(j*3))**2)
          pot_eenn(j) = pot_eenn(j) + cha_eenn(i)/tmp
          pot_menn(j) = pot_menn(j) + cha_menn(i)/tmp
          pot_eePoint(j) = pot_eePoint(j) + cha_ee(i)/tmp
          pot_mePoint(j) = pot_mePoint(j) + cha_me(i)/tmp
        end do
        distance(j) = dsqrt((coor_qm(luckatom*3-2)-coor_mm(j*3-2))**2+(coor_qm(luckatom*3-1)-coor_mm(j*3-1))**2  &
                        +(coor_qm(luckatom*3)-coor_mm(j*3))**2)
      end do
      pot_eenn=pot_eenn*18.2223**2/627.510;pot_menn=pot_menn*18.2223**2/627.510
      pot_mePoint=pot_mePoint*18.2223**2/627.510;pot_eePoint=pot_eePoint*18.2223**2/627.510
      pot_eenn=2*pot_eenn-pot_menn

      do j = 1,RCnum
        rc = rc + RCeff(j) *dsqrt((coor_qm(RCindex(1,j)*3-2)-coor_qm(RCindex(2,j)*3-2))**2  &
                                 +(coor_qm(RCindex(1,j)*3-1)-coor_qm(RCindex(2,j)*3-1))**2  &
                                 +(coor_qm(RCindex(1,j)*3)-coor_qm(RCindex(2,j)*3))**2)
      end do
      do j = 1,num_mm
        write(182,'(2F10.4,6F16.8)') rc,distance(j),pot_me(j),pot_menn(j),pot_mePoint(j),pot_ee(j),pot_eenn(j),pot_eePoint(j) 
      end do
      write(184,'(F10.4,7F16.8)') rc,sum(pot_me(1:num_mm)*cha_mm_copy(1:num_mm))*627.510,       &
                                     sum(pot_menn(1:num_mm)*cha_mm_copy(1:num_mm))*627.510,     &
                                     sum(pot_mepoint(1:num_mm)*cha_mm_copy(1:num_mm))*627.510,  &
                                     sum(pot_ee(1:num_mm)*cha_mm_copy(1:num_mm))*627.510,       &
                                     sum(pot_eenn(1:num_mm)*cha_mm_copy(1:num_mm))*627.510,     &
                                     sum(pot_eepoint(1:num_mm)*cha_mm_copy(1:num_mm))*627.510,  &
                                    (sum(pot_ee(1:num_mm)*cha_mm_copy(1:num_mm))*627.510-       &
                                     sum(pot_me(1:num_mm)*cha_mm_copy(1:num_mm))*627.510)/2
      npoint=npoint+1
    end do

    !part-3 deallocate all variables related to reann
    close(181);close(182)
    call delete_Chreann()
    deallocate(coor_qm,coor_mm,cha_mm,force_qm,force_mm,cha_eenn,cha_ee,cha_menn,cha_me,pot_ee,pot_me,pot_eenn,pot_menn,distance)
end program
