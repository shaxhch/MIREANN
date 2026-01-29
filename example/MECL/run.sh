export OMP_NUM_THREADS=1
mpirun -np 2 sander-qmcut7luck.MPI -O -i dyn.in -o dyn.out -p MECL.prmtop -c MECL.rst  -r dyn.rst -x dyn.mdcrd
