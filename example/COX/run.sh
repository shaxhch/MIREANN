export OMP_NUM_THREADS=1
mpirun -np 2 sander-qmcut7luck.MPI -O -i dyn.in -o dyn_restraint.out -p cox1-wat.prmtop -c cox1.rst  -r dyn.rst -x dyn_infixq.mdcrd
