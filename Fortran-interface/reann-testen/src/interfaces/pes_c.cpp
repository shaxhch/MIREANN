#include <iostream>
#include "pes_cpp.hpp"
#include "pes_c.h"

using namespace std;

PES* create_pes(double *cell, int *pbc, int numatoms, char **species, int maxnumtype, char **atomtype, double *mass) {
  return new Pes(cell, pbc, numatoms, species, maxnumtype, atomtype, mass);
}

void delete_pes(PES* pes){
  delete pes;
}

void pes_reann_out(PES* pes, int numatoms, double *cart, double *energy, double *force) {
  pes->reann_out(numatoms, cart, energy, force);
  return;
}
