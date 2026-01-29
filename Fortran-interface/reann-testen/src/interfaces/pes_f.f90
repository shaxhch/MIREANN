module pes_lib
    use iso_c_binding
    implicit none
    public :: pes
    interface
      function create_pes_c(cell, pbc, numatoms, species, maxnumtype, atomtype, mass) bind(C, name="create_pes")
        use iso_c_binding
        implicit none
        type(c_ptr) :: create_pes_c
        integer(c_int), value :: numatoms
        integer(c_int) :: pbc(3)
        real(c_double), dimension(3,3) :: cell
        type(c_ptr), dimension(numatoms)::species 
        integer(c_int), value :: maxnumtype
        type(c_ptr), dimension(maxnumtype)::atomtype
        real(c_double), dimension(numatoms) :: mass
      end function
    
      subroutine delete_pes_c(pes) bind(C, name="delete_pes")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: pes
      end subroutine
    
      subroutine pes_reann_out_c(pes, numatoms, cart, energy, force) bind(C, name="pes_reann_out")
        use iso_c_binding
        implicit none
        type(c_ptr), intent(in), value :: pes
        integer(c_int), value :: numatoms
        real(c_double), dimension(numatoms*3), intent(in) :: cart
        real(c_double), intent(inout) :: energy
        real(c_double), dimension(numatoms*3), intent(inout) :: force
      end subroutine
    end interface

    type pes
      private
      type(c_ptr) :: ptr
      integer::numatoms
    contains
      final :: delete_pes
      procedure :: delete => delete_pes_polymorph
      procedure :: reann_out => pes_reann_out
    end type

    interface pes
      procedure create_pes
    end interface

contains 
    function create_pes(cell, pbc, numatoms, species_f, maxnumtype, atomtype_f, mass)
      implicit none
      type(pes) :: create_pes
      double precision, dimension(3,3),intent(in) :: cell
      integer, dimension(3),intent(in):: pbc(3)
      integer,intent(in) :: numatoms
      character(len=2, kind=C_CHAR), dimension(numatoms),intent(in) :: species_f
      character(len=3, kind=C_CHAR), dimension(numatoms), target:: species_c
      type(c_ptr), dimension(numatoms)::species
      integer,intent(in) :: maxnumtype
      character(len=2, kind=C_CHAR), dimension(maxnumtype),intent(in) :: atomtype_f
      character(len=3, kind=C_CHAR), dimension(maxnumtype), target:: atomtype_c
      type(c_ptr), dimension(maxnumtype)::atomtype
      double precision, dimension(numatoms),intent(in) :: mass
      integer :: i
      do i=1,numatoms
        species_c(i)= species_f(i)//C_NULL_CHAR
        species(i)= c_loc(species_c(i))
      end do
      do i=1,maxnumtype
        atomtype_c(i)= atomtype_f(i)//C_NULL_CHAR
        atomtype(i)= c_loc(atomtype_c(i))
      end do
      create_pes%ptr = create_pes_c(cell,pbc,numatoms, species,maxnumtype,atomtype, mass)
      create_pes%numatoms = numatoms
    end function

    subroutine delete_pes(this)
      implicit none
      type(pes) :: this
      call delete_pes_c(this%ptr)
    end subroutine

    subroutine delete_pes_polymorph(this)
      implicit none
      class(pes) :: this
      call delete_pes_c(this%ptr)
    end subroutine

    subroutine pes_reann_out(this, cart, energy, force)
      implicit none
      class(pes), intent(in) :: this
      double precision, dimension(this%numatoms*3), intent(in) :: cart
      double precision, intent(inout) :: energy
      double precision, dimension(this%numatoms*3), intent(inout) :: force
      call pes_reann_out_c(this%ptr,this%numatoms, cart, energy, force)
    end subroutine

end module

module pes_mod
  use pes_lib
  implicit none
  type(pes) :: pes_ptr
  real(kind=8)::cell(3,3)
  integer(kind=4)::pbc(3)
  integer(kind=4)::numatoms,maxnumtype
  character(len=2),allocatable::species(:),atomtype(:)
  real(kind=8),allocatable::mass(:)
end module

subroutine readinput()
  use pes_mod
  implicit none
  integer(kind=4) i
  open(428,file="./input_reann")
  read(428,*)
  do i = 1,3
    read(428,*) cell(i,1:3)
  end do
  read(428,*)
  read(428,*) pbc
  read(428,*)
  read(428,*) numatoms
  read(428,*)
  if  (.not. allocated(species)) then
    allocate(species(numatoms),mass(numatoms))
  end if
  do i=1,numatoms
    read(428,*) species(i),mass(i)
  enddo
  read(428,*)
  read(428,*) maxnumtype
  if (.not. allocated(atomtype)) then
    allocate(atomtype(maxnumtype))
  end if
  read(428,*)
  do i=1,maxnumtype
    read(428,*) atomtype(i)
  enddo      
  close(428)
end subroutine

subroutine init_reann()
  use pes_mod
  implicit none
  call readinput()
  pes_ptr=pes(cell,pbc,numatoms, species, maxnumtype, atomtype, mass)
end subroutine

subroutine delete_reann()
  use pes_mod
  implicit none
  deallocate(species,mass,atomtype)
  call pes_ptr%delete
end subroutine 
