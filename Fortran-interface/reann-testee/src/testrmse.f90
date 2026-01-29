program mypes 
    use chpes_mod
    implicit none
    character*40 a
    integer(kind=4) :: i,j,npoint,istat
    real(kind=8) :: energy_cal
    real(kind=8),allocatable::coor_qm(:), coor_mm(:), cha_mm(:), force_qm(:), force_mm(:),cha_qm(:)
    real(kind=8) :: t_start, t_end

    !part-1 reann model initialization
    call init_chreann()

    allocate(coor_qm(chnumatoms*3),coor_mm(chmaxnum_mm*3),cha_mm(chmaxnum_mm),force_qm(chnumatoms*3), &
             force_mm(chmaxnum_mm*3),cha_qm(chnumatoms))
    npoint=1
    open(181,file="configuration")
    open(182,file="reann")

    do
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

      coor_mm = 0d0
      i = 1
      do
        if (i > chmaxnum_mm) then
            print *, "Warning: Exceeded maximum MM atoms limit."
            exit
        end if
        read(181,'(3F12.6,F9.4)',iostat=istat) coor_mm((i-1)*3 + 1), coor_mm((i-1)*3 + 2), coor_mm((i-1)*3 + 3), cha_mm(i)
        if (istat/=0) exit
        i=i+1
      end do

      call cpu_time(t_start)
      call Chpes_ptr%reann_chout(coor_qm,coor_mm,cha_mm,energy_cal,force_qm,force_mm,cha_qm)
      call cpu_time(t_end)
      write(*,'("Time for point",I5,": ",F10.6," seconds")') npoint, t_end - t_start

      write(182,'(A8,I5,F14.8)') 'Point=',npoint,energy_cal
      npoint=npoint+1
    end do

    !part-3 deallocate all variables related to reann
    close(181);close(182)
    call delete_Chreann()
    deallocate(coor_qm,coor_mm,cha_mm,force_qm,force_mm,cha_qm)
end program
